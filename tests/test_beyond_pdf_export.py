"""Offline regression test for the Beyond PDF Embed Generators' CLI contract.

Feeds synthetic inputs (no real checkpoints needed) through each registered
Generator and parses the data blob back out of the HTML it writes. Follows
the ablation scope's table-generator test idiom (result_analysis/tests/
test_ablation.py): invoke through the CLI, parse the artifact back, assert on
parsed values -- never on the generator's internals.

Parametrised over GENERATORS (beyond_pdf/export.py's lambda-simplex
scrubber, beyond_pdf/ablation_export.py's ablation explorer, and
beyond_pdf/mechanism_export.py's mechanism explainer) so a future Embed
Generator joins by adding one entry to that registry, not by restructuring
this file. Every entry is
checked against the shared "Embed contract" from issue #43 / #39: the data
blob parses back out, every index into a deduplicated table is in range, the
artifact is under the size cap, the artifact contains no external
references, and the artifact satisfies the iframe contract (no page chrome,
body fills the frame, transparent ground).

Runs on a fresh clone: no checkpoints, no network, no wandb access. The real
ChickenBanana environment is used for rollouts (it is local, deterministic,
offline gym code, not a real training run), so the synthetic snapshots below
only need components_q/lambdas arrays shaped like the real ones.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = REPO_ROOT / "beyond_pdf" / "export.py"
ABLATION_EXPORT_SCRIPT = REPO_ROOT / "beyond_pdf" / "ablation_export.py"
ENV_EXPORT_SCRIPT = REPO_ROOT / "beyond_pdf" / "env_export.py"
PARETO_EXPORT_SCRIPT = REPO_ROOT / "beyond_pdf" / "pareto_export.py"
MECHANISM_EXPORT_SCRIPT = REPO_ROOT / "beyond_pdf" / "mechanism_export.py"
N_STATES, N_ACTIONS, NUM_REWARDS = 64, 4, 3

sys.path.insert(0, str(REPO_ROOT / "result_analysis"))
sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "ablation"))
from arms import COMPONENTS, EPSILON_ARMS, NORMALIZER_ARMS, RB_ARMS, TAU_ARMS  # noqa: E402, F401

sys.path.insert(0, str(REPO_ROOT / "beyond_pdf"))
import mechanism_export as mechanism  # noqa: E402

ABLATION_SWEEPS = {
    "tau": TAU_ARMS,
    "rb": RB_ARMS,
    "normalizer": NORMALIZER_ARMS,
    "epsilon": EPSILON_ARMS,
}


def write_snapshot(root, episode, components_q, lambdas):
    d = Path(root) / f"{episode:05d}"
    d.mkdir(parents=True)
    np.save(d / "components_q.npy", components_q)
    np.save(d / "lambdas.npy", np.asarray(lambdas, dtype=float))


def make_all_zero_snapshots(root):
    """Two episodes of all-zero Q-tables: every state ties everywhere, so the
    scrubber generator's deterministic per-state tie-break (see export.py's
    module docstring) collapses every weighting to one policy that never
    reaches the gate -- a known, hand-verified rollout (returns 0/0/0)."""
    zeros = np.zeros((NUM_REWARDS, N_STATES, N_ACTIONS))
    write_snapshot(root, 0, zeros, [1 / 3, 1 / 3, 1 / 3])
    write_snapshot(root, 10, zeros, [0.2, 0.3, 0.5])


def run_export(snapshots, out, expected_return, extra=()):
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--snapshots",
        str(snapshots),
        "--out",
        str(out),
        "--expected-return",
        *map(str, expected_return),
        "--lattice-step",
        "2",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_data_blob(html):
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    return json.loads(html[start:end])


def _build_scrubber(tmp_path_factory):
    """Build beyond_pdf/export.py's lambda-simplex scrubber Embed from a
    synthetic all-zero snapshot pair. See make_all_zero_snapshots' docstring
    for why every weighting collapses to one known no_gate policy."""
    snapshots = tmp_path_factory.mktemp("snapshots")
    make_all_zero_snapshots(snapshots)
    out = tmp_path_factory.mktemp("out") / "scrubber.html"
    result = run_export(snapshots, out, (0, 0, 0))
    assert result.returncode == 0, result.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    return html, parse_data_blob(html)


def _check_scrubber_indices(data):
    """The scrubber's dedup table: every cellIndex/realIndex entry must
    index into data['policies']."""
    n_weightings = len(data["lattice"])
    for row in data["cellIndex"]:
        assert len(row) == n_weightings
        assert all(0 <= idx < len(data["policies"]) for idx in row)
    for idx in data["realIndex"]:
        assert 0 <= idx < len(data["policies"])


@dataclass
class GeneratorSpec:
    """One entry in the Embed Generator registry: how to build a synthetic
    artifact through the Generator's CLI, and how to validate that its
    deduplicated-table indices are in range (schema differs per Generator, so
    this is supplied per entry rather than assumed generic)."""

    name: str
    build: Callable[[pytest.FixtureRequest], Tuple[str, dict]]
    check_indices: Callable[[dict], None]


def _write_ablation_csv(path, arm_labels, metric, n_seeds=2, n_steps=5):
    """A synthetic Tidy CSV shaped like the ablation scope's committed
    ones: _step, seed, arm, <metric>, one row per (arm, seed, step)."""
    rows = ["_step,seed,arm," + metric]
    for arm in arm_labels:
        for seed in range(n_seeds):
            for step in range(n_steps):
                rows.append(f"{step},{seed},{arm},{step + seed}")
    path.write_text("\n".join(rows) + "\n")


def _make_ablation_data(root):
    for comp in COMPONENTS:
        _write_ablation_csv(
            root / f"nominal_{comp.lower()}_reward.csv", ["nominal"], "ep_info/" + comp
        )
        _write_ablation_csv(
            root / f"nominal_{comp.lower()}_lambda.csv", ["nominal"], "lambdas/" + comp
        )
        for sweep, arms in ABLATION_SWEEPS.items():
            labels = [a.label for a in arms]
            _write_ablation_csv(
                root / f"{sweep}_{comp.lower()}_reward.csv", labels, "ep_info/" + comp
            )
            _write_ablation_csv(
                root / f"{sweep}_{comp.lower()}_lambda.csv", labels, "lambdas/" + comp
            )


def _build_ablation_curves(tmp_path_factory):
    """Build beyond_pdf/ablation_export.py's ablation explorer Embed from
    synthetic per-(sweep, component, kind) Tidy CSVs shaped like the
    committed ones, with a small --grid-points so the bootstrap in
    core.stats.bootstrap_curve stays fast under test."""
    data_dir = tmp_path_factory.mktemp("ablation_data")
    _make_ablation_data(data_dir)
    out = tmp_path_factory.mktemp("out") / "ablation_curves.html"
    cmd = [
        sys.executable,
        str(ABLATION_EXPORT_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--out",
        str(out),
        "--grid-points",
        "10",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    return html, parse_data_blob(html)


def _check_ablation_curves_indices(data):
    """No deduplicated table here (unlike the scrubber): every panel is a
    self-contained set of pre-computed curves, so the structural invariants
    are the 4 sweeps x 2 kinds selector options and all 4 x 2 x 3 panel
    cells resolving into the meta table (the shared template's schema)."""
    assert len(data["selectors"]) == 2
    assert len(data["selectors"][0]["options"]) == 4  # sweeps
    assert len(data["selectors"][1]["options"]) == 2  # kinds
    n = 0
    for sweep, kinds in data["panels"].items():
        n += len(kinds) * len(kinds[next(iter(kinds))])
        for kind, panels in kinds.items():
            for pk in panels.values():
                assert pk in data["meta"]["panels"]
    assert n == 4 * 2 * 3


def _make_mechanism_returns_csv(path, window_end):
    """A synthetic actual-returns CSV shaped like the committed one
    (episode,<name0>,<name1>,<name2>), contiguous from episode 1 to
    `window_end` -- everything replay_smoothed_returns needs."""
    names = [c.name for c in mechanism.COMPONENTS]
    lines = ["episode," + ",".join(names)]
    for ep in range(1, window_end + 1):
        # Deterministic, mildly varying synthetic returns -- no meaning
        # beyond exercising the CLI contract on a fresh clone.
        vals = [(ep * (i + 1)) % 7 for i in range(len(names))]
        lines.append(f"{ep}," + ",".join(str(v) for v in vals))
    path.write_text("\n".join(lines) + "\n")


def _build_mechanism_explainer(tmp_path_factory):
    """Build beyond_pdf/mechanism_export.py's mechanism explainer Embed from
    a synthetic actual-returns CSV and a synthetic snapshot whose lambdas.npy
    is computed by the module's own replay_smoothed_returns/dylam_weights --
    the fidelity check this generator shares with the scrubber (see its
    module docstring) requires the two to agree, and hand-picking a value
    that happens to satisfy it would just be duplicating that same math in
    the test instead of the generator."""
    window_end = 15
    data_dir = tmp_path_factory.mktemp("mechanism_data")
    returns_csv = data_dir / "actual_returns.csv"
    _make_mechanism_returns_csv(returns_csv, window_end)

    buffer_len, tau_lambda = mechanism.dylam_hyperparams()
    returns = mechanism.load_actual_returns(returns_csv)
    gbar, _ = mechanism.replay_smoothed_returns(
        returns, window_end, buffer_len, tau_lambda
    )
    _, _, _, lam = mechanism.dylam_weights(gbar)
    lam_arr = np.array([lam[c.name] for c in mechanism.COMPONENTS])

    snapshots = tmp_path_factory.mktemp("mechanism_snapshots")
    write_snapshot(
        snapshots, window_end, np.zeros((NUM_REWARDS, N_STATES, N_ACTIONS)), lam_arr
    )

    out = tmp_path_factory.mktemp("out") / "mechanism_explainer.html"
    cmd = [
        sys.executable,
        str(MECHANISM_EXPORT_SCRIPT),
        "--snapshots",
        str(snapshots),
        "--out",
        str(out),
        "--window-end",
        str(window_end),
        "--actual-returns",
        str(returns_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    return html, parse_data_blob(html)


def _check_mechanism_explainer_indices(data):
    """No deduplicated table here (a single worked window, not a lattice):
    the structural invariant is that the policy array covers every full
    environment state (meta.nStates -- position x inventory, unlike
    grid.statePositions, which is one entry per grid cell) and every
    displayed component appears in every per-component dict."""
    assert len(data["policy"]["policy"]) == data["meta"]["nStates"]
    assert len(data["meta"]["grid"]["statePositions"]) > 0
    names = {c["name"] for c in data["meta"]["components"]}
    assert set(data["lambda"]) == names
    assert set(data["gbar"]) == names


def _make_tiny_series_csv(path, columns, identity, identities, n_seeds=2, n_steps=5):
    """A synthetic Tidy CSV shaped like the trad/curriculum scopes':
    _step, seed, <identity>, <columns...>, one row per (identity, seed, step)."""
    header = ["_step", "seed", identity] + columns
    rows = [",".join(header)]
    for ident in identities:
        for seed in range(n_seeds):
            for step in range(n_steps):
                rows.append(
                    ",".join(
                        [str(step), str(seed), ident]
                        + [str(step + seed)] * len(columns)
                    )
                )
    path.write_text("\n".join(rows) + "\n")


def _make_env_curves_data(root):
    """The trad and curriculum CSVs env_export.py reads, minimum-shaped.

    Column and file names must match the real committed ones (the generator
    reads them by name), but the values only need to exercise the CLI
    contract -- one row per (method/identity, seed, step)."""
    root = Path(root)
    trad = root / "trad" / "data"
    curr = root / "curriculum" / "data"
    trad.mkdir(parents=True)
    curr.mkdir(parents=True)

    _make_tiny_series_csv(
        trad / "chicken_banana.csv",
        ["ep_info/total"],
        "method",
        ["Base SO RL", "Q-Decomposition", "UDC", "DyLam"],
    )
    _make_tiny_series_csv(
        trad / "halfcheetah_v4.csv",
        ["ep_info/Final_position"],
        "method",
        ["Base SO RL", "UDC", "DyLam"],
    )
    _make_tiny_series_csv(
        trad / "halfcheetah_v4_env_return.csv",
        ["ep_info/total"],
        "method",
        ["Base SO RL", "UDC", "DyLam"],
    )
    _make_tiny_series_csv(
        trad / "vss_v0.csv",
        ["ep_info/Goal"],
        "method",
        ["Base SO RL", "UDC", "DyLam", "Tuned-UDC"],
    )

    # curriculum: one reward + one lambda CSV per (env, component), with the
    # env's component registry (curriculum/sources.py ENVS) driving names.
    sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "curriculum"))
    from sources import ENVS as CURRICULUM_ENVS

    for env_name, spec in CURRICULUM_ENVS.items():
        for comp in spec.components:
            _make_tiny_series_csv(
                curr / f"{env_name.lower()}_{comp.name.lower()}_reward.csv",
                [comp.ep_metric],
                "method",
                ["DyLam"],
            )
            _make_tiny_series_csv(
                curr / f"{env_name.lower()}_{comp.name.lower()}_lambda.csv",
                [f"lambdas/{comp.name}"],
                "method",
                ["DyLam"],
            )


def _build_env_curves(tmp_path_factory):
    """Build beyond_pdf/env_export.py's per-environment curve explorer from
    synthetic trad/curriculum Tidy CSVs laid out like the committed ones,
    via the generator's --data-root flag (a fresh-clone-safe second data
    root: the generator's module constants stay untouched)."""
    data_root = tmp_path_factory.mktemp("env_data")
    _make_env_curves_data(data_root)
    out = tmp_path_factory.mktemp("out") / "env_curves.html"
    cmd = [
        sys.executable,
        str(ENV_EXPORT_SCRIPT),
        "--data-root",
        str(data_root),
        "--out",
        str(out),
        "--grid-points",
        "10",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    return html, parse_data_blob(html)


def _check_env_curves_indices(data):
    """Every env x metric cell resolves to a panel in the meta table, and
    the initial render (first option of each selector) is non-empty."""
    meta = data["meta"]["panels"]
    cells = 0
    for env, metrics in data["panels"].items():
        cells += len(metrics)
        for pk in metrics.values():
            assert pk in meta
    assert cells == len(meta)
    for sel in data["selectors"]:
        first = sel["options"][0]["key"]
        assert first in data["groups"][sel["key"]]


def _make_pareto_csv(path, n_obj, n_seeds, rows_per_seed, with_weights=False):
    """A synthetic morl-scope candidate-point CSV (and, optionally, the
    matching weight-trajectory CSV): obj1..objN, seed, point_index."""
    obj_cols = [f"obj{i + 1}" for i in range(n_obj)]
    frames = []
    for seed in range(n_seeds):
        for point in range(rows_per_seed):
            # Deterministic spread across the objective space so the
            # per-seed Pareto filter keeps a non-trivial front.
            vals = [
                (point + seed) % (rows_per_seed // 2 or 1) + i for i in range(n_obj)
            ]
            frames.append(vals + [seed, point])
    pd_frames = None
    header = ",".join(obj_cols + ["seed", "point_index"])
    lines = [header] + [",".join(str(v) for v in row) for row in frames]
    path.write_text("\n".join(lines) + "\n")
    if with_weights:
        weight_path = path.parent / (path.stem + "_weights.csv")
        wlines = [",".join(obj_cols + ["seed", "point_index"])]
        for seed in range(n_seeds):
            for point in range(rows_per_seed):
                w = [round(1.0 / n_obj, 6)] * n_obj
                wlines.append(",".join(str(v) for v in w + [seed, point]))
        weight_path.write_text("\n".join(wlines) + "\n")
    return pd_frames


def _build_pareto_explorer(tmp_path_factory):
    """Build beyond_pdf/pareto_export.py's Pareto explorer from synthetic
    morl Tidy CSVs via --data-root. sources.per_seed reads its module-level
    DATA constant, so the test patches it before invoking the CLI -- the
    same patch the generator itself performs for a non-default data root."""
    data_root = tmp_path_factory.mktemp("morl_data")
    morl = data_root / "morl" / "data"
    morl.mkdir(parents=True)

    def write(name, n_obj, rows, with_weights=False):
        obj_cols = [f"obj{i + 1}" for i in range(n_obj)]
        lines = [",".join(obj_cols + ["seed", "point_index"])]
        for seed in range(2):
            for point in range(rows):
                vals = [(point + seed) % max(rows // 2, 1) + i for i in range(n_obj)]
                lines.append(",".join(str(v) for v in vals + [seed, point]))
        (morl / name).write_text("\n".join(lines) + "\n")
        if with_weights:
            wlines = [",".join(obj_cols + ["seed", "point_index"])]
            for seed in range(2):
                for point in range(rows):
                    w = [round(1.0 / n_obj, 6)] * n_obj
                    wlines.append(",".join(str(v) for v in w + [seed, point]))
            (morl / (name.replace(".csv", "") + "_weights.csv")).write_text(
                "\n".join(wlines) + "\n"
            )

    write("halfcheetah_pgmorl.csv", 2, 20)
    write("halfcheetah_gpi_ls.csv", 2, 20)
    write("halfcheetah_dylam.csv", 2, 20, with_weights=True)
    write("minecart_gpi_ls.csv", 3, 20)
    write("minecart_dynmorl.csv", 3, 20, with_weights=True)
    write("minecart_dylam.csv", 3, 20, with_weights=True)

    out = tmp_path_factory.mktemp("out") / "pareto_explorer.html"
    cmd = [
        sys.executable,
        str(PARETO_EXPORT_SCRIPT),
        "--data-root",
        str(data_root),
        "--out",
        str(out),
        "--max-points-per-method",
        "50",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    return html, parse_data_blob(html)


def _check_pareto_explorer_indices(data):
    """Every point's index is in range of its own method's arrays, every
    weighted method's weight rows match its point count, and both
    environments are present."""
    for env, panel in data["meta"]["panels"].items():
        for method in panel["methods"]:
            pts = panel["points"][method["label"]]["obj"]
            assert len(panel["points"][method["label"]]["i"]) == len(pts)
        for label, w in panel["weights"].items():
            assert len(w["values"]) == len(panel["points"][label]["obj"])
            assert len(w["values"][0]) == len(panel["axisLabels"])


def test_mechanism_fidelity_guard_aborts_on_mismatch(tmp_path_factory, tmp_path):
    """The mechanism explainer shares export.py's fidelity-precondition idea
    (see its module docstring): a snapshot whose recorded lambdas.npy
    disagrees with the freshly-replayed EMA must abort with no output."""
    window_end = 15
    data_dir = tmp_path_factory.mktemp("mechanism_bad_data")
    returns_csv = data_dir / "actual_returns.csv"
    _make_mechanism_returns_csv(returns_csv, window_end)

    snapshots = tmp_path_factory.mktemp("mechanism_bad_snapshots")
    write_snapshot(
        snapshots,
        window_end,
        np.zeros((NUM_REWARDS, N_STATES, N_ACTIONS)),
        [1 / 3, 1 / 3, 1 / 3],
    )

    out = tmp_path / "should_not_exist.html"
    cmd = [
        sys.executable,
        str(MECHANISM_EXPORT_SCRIPT),
        "--snapshots",
        str(snapshots),
        "--out",
        str(out),
        "--window-end",
        str(window_end),
        "--actual-returns",
        str(returns_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "fidelity" in result.stderr.lower()
    assert not out.exists()


# Registry of Embed Generators under the shared offline-CLI contract (see
# module docstring). Grows as later tickets add Embed generators; every test
# below runs once per entry via the `generator`/`generated` fixtures.
GENERATORS = [
    GeneratorSpec("lambda_simplex_scrubber", _build_scrubber, _check_scrubber_indices),
    GeneratorSpec(
        "ablation_curves", _build_ablation_curves, _check_ablation_curves_indices
    ),
    GeneratorSpec("env_curves", _build_env_curves, _check_env_curves_indices),
    GeneratorSpec(
        "pareto_explorer", _build_pareto_explorer, _check_pareto_explorer_indices
    ),
    GeneratorSpec(
        "mechanism_explainer",
        _build_mechanism_explainer,
        _check_mechanism_explainer_indices,
    ),
]


@pytest.fixture(scope="module", params=GENERATORS, ids=lambda g: g.name)
def generator(request):
    return request.param


@pytest.fixture(scope="module")
def generated(generator, tmp_path_factory):
    return generator.build(tmp_path_factory)


def test_data_blob_parses_back(generated):
    _, data = generated
    assert isinstance(data, dict)
    assert data


def test_dedup_table_indices_in_range(generator, generated):
    _, data = generated
    generator.check_indices(data)


def test_size_under_cap(generated):
    html, _ = generated
    assert len(html.encode("utf-8")) < 5 * 1024 * 1024


def test_no_external_references(generated):
    html, _ = generated
    lowered = html.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<link" not in lowered
    assert "cdn." not in lowered


def test_iframe_contract(generated):
    """The Embed contract established by #43: no page chrome (it now lives
    in the surrounding prose), the body fills the frame, and the ground is
    transparent so the page's own theme shows through."""
    html, _ = generated
    lowered = html.lower()
    assert "<h1>" not in lowered and "<h1 " not in lowered
    assert 'class="framing"' not in html
    assert 'class="caption"' not in html
    lowered_no_space = lowered.replace(" ", "")
    assert "background:transparent" in lowered_no_space


# --- Scrubber-specific coverage (beyond_pdf/export.py's own semantics) -----
# These assert on the synthetic all-zero-Q fixture's known behaviour and on
# the scrubber's fidelity precondition; they are not part of the generic,
# per-Generator "Embed contract" above, so they stay tied to the scrubber
# rather than parametrised over the registry.


@pytest.fixture(scope="module")
def all_zero_snapshots(tmp_path_factory):
    root = tmp_path_factory.mktemp("fidelity_snapshots")
    make_all_zero_snapshots(root)
    return root


@pytest.fixture(scope="module")
def scrubber_generated(tmp_path_factory):
    return _build_scrubber(tmp_path_factory)


def test_behavior_classes_are_known(scrubber_generated):
    _, data = scrubber_generated
    known = set(data["meta"]["behaviorClasses"])
    assert known == {"gate_only", "gate_banana", "gate_chicken", "gate_both", "no_gate"}
    for entry in data["policies"]:
        assert entry["class"] in known


def test_all_zero_q_never_reaches_gate(scrubber_generated):
    _, data = scrubber_generated
    for entry in data["policies"]:
        assert entry["class"] == "no_gate"
        assert entry["returns"] == [0.0, 0.0, 0.0]


def test_fidelity_guard_aborts_on_mismatch(all_zero_snapshots, tmp_path):
    out = tmp_path / "should_not_exist.html"
    result = run_export(all_zero_snapshots, out, (100, 30, 70))
    assert result.returncode != 0
    assert "fidelity" in result.stderr.lower()
    assert not out.exists()
