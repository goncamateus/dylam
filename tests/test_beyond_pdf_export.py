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
MECHANISM_EXPORT_SCRIPT = REPO_ROOT / "beyond_pdf" / "mechanism_export.py"
N_STATES, N_ACTIONS, NUM_REWARDS = 64, 4, 3

sys.path.insert(0, str(REPO_ROOT / "result_analysis"))
sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "ablation"))
from arms import (COMPONENTS, EPSILON_ARMS, NORMALIZER_ARMS,  # noqa: E402
                  RB_ARMS, TAU_ARMS)

sys.path.insert(0, str(REPO_ROOT / "beyond_pdf"))
import mechanism_export as mechanism  # noqa: E402

ABLATION_SWEEPS = {"tau": TAU_ARMS, "rb": RB_ARMS, "normalizer": NORMALIZER_ARMS,
                   "epsilon": EPSILON_ARMS}


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
    cmd = [sys.executable, str(EXPORT_SCRIPT), "--snapshots", str(snapshots),
           "--out", str(out), "--expected-return", *map(str, expected_return),
           "--lattice-step", "2", *extra]
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
        _write_ablation_csv(root / f"nominal_{comp.lower()}_reward.csv", ["nominal"],
                            "ep_info/" + comp)
        _write_ablation_csv(root / f"nominal_{comp.lower()}_lambda.csv", ["nominal"],
                            "lambdas/" + comp)
        for sweep, arms in ABLATION_SWEEPS.items():
            labels = [a.label for a in arms]
            _write_ablation_csv(root / f"{sweep}_{comp.lower()}_reward.csv", labels,
                                "ep_info/" + comp)
            _write_ablation_csv(root / f"{sweep}_{comp.lower()}_lambda.csv", labels,
                                "lambdas/" + comp)


def _build_ablation_curves(tmp_path_factory):
    """Build beyond_pdf/ablation_export.py's ablation explorer Embed from
    synthetic per-(sweep, component, kind) Tidy CSVs shaped like the
    committed ones, with a small --grid-points so the bootstrap in
    core.stats.bootstrap_curve stays fast under test."""
    data_dir = tmp_path_factory.mktemp("ablation_data")
    _make_ablation_data(data_dir)
    out = tmp_path_factory.mktemp("out") / "ablation_curves.html"
    cmd = [sys.executable, str(ABLATION_EXPORT_SCRIPT), "--data-dir", str(data_dir),
           "--out", str(out), "--grid-points", "10"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    return html, parse_data_blob(html)


def _check_ablation_curves_indices(data):
    """No deduplicated table here (unlike the scrubber): every panel is a
    self-contained set of pre-computed curves, so the only structural
    invariant is that all 4 sweeps x 2 kinds x 3 components are present."""
    assert len(data["sweeps"]) == 4
    assert len(data["kinds"]) == 2
    assert len(data["components"]) == 3
    for sweep in data["sweeps"]:
        for kind in data["kinds"]:
            assert set(data["panels"][sweep["key"]][kind["key"]]) == set(data["components"])


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
    gbar, _, _ = mechanism.replay_smoothed_returns(returns, window_end, buffer_len, tau_lambda)
    _, _, _, lam = mechanism.dylam_weights(gbar)
    lam_arr = np.array([lam[c.name] for c in mechanism.COMPONENTS])

    snapshots = tmp_path_factory.mktemp("mechanism_snapshots")
    write_snapshot(snapshots, window_end, np.zeros((NUM_REWARDS, N_STATES, N_ACTIONS)), lam_arr)

    out = tmp_path_factory.mktemp("out") / "mechanism_explainer.html"
    cmd = [sys.executable, str(MECHANISM_EXPORT_SCRIPT), "--snapshots", str(snapshots),
           "--out", str(out), "--window-end", str(window_end),
           "--actual-returns", str(returns_csv)]
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


def test_mechanism_fidelity_guard_aborts_on_mismatch(tmp_path_factory, tmp_path):
    """The mechanism explainer shares export.py's fidelity-precondition idea
    (see its module docstring): a snapshot whose recorded lambdas.npy
    disagrees with the freshly-replayed EMA must abort with no output."""
    window_end = 15
    data_dir = tmp_path_factory.mktemp("mechanism_bad_data")
    returns_csv = data_dir / "actual_returns.csv"
    _make_mechanism_returns_csv(returns_csv, window_end)

    snapshots = tmp_path_factory.mktemp("mechanism_bad_snapshots")
    write_snapshot(snapshots, window_end, np.zeros((NUM_REWARDS, N_STATES, N_ACTIONS)),
                    [1 / 3, 1 / 3, 1 / 3])

    out = tmp_path / "should_not_exist.html"
    cmd = [sys.executable, str(MECHANISM_EXPORT_SCRIPT), "--snapshots", str(snapshots),
           "--out", str(out), "--window-end", str(window_end),
           "--actual-returns", str(returns_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "fidelity" in result.stderr.lower()
    assert not out.exists()


# Registry of Embed Generators under the shared offline-CLI contract (see
# module docstring). Grows as later tickets add Embed generators; every test
# below runs once per entry via the `generator`/`generated` fixtures.
GENERATORS = [
    GeneratorSpec("lambda_simplex_scrubber", _build_scrubber, _check_scrubber_indices),
    GeneratorSpec("ablation_curves", _build_ablation_curves, _check_ablation_curves_indices),
    GeneratorSpec("mechanism_explainer", _build_mechanism_explainer,
                  _check_mechanism_explainer_indices),
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
