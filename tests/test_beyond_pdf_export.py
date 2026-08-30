"""Offline regression test for beyond_pdf/export.py's CLI.

Feeds a synthetic snapshot directory (no real checkpoints needed) through the
generator and parses the data blob back out of the HTML it writes. Follows
the ablation scope's table-generator test idiom (result_analysis/tests/
test_ablation.py): invoke through the CLI, parse the artifact back, assert on
parsed values -- never on the generator's internals.

Runs on a fresh clone: no checkpoints, no network, no wandb access. The real
ChickenBanana environment is used for rollouts (it is local, deterministic,
offline gym code, not a real training run), so the two synthetic episodes
below only need components_q/lambdas arrays shaped like the real ones.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXPORT_SCRIPT = Path(__file__).resolve().parent.parent / "beyond_pdf" / "export.py"
N_STATES, N_ACTIONS, NUM_REWARDS = 64, 4, 3


def write_snapshot(root, episode, components_q, lambdas):
    d = Path(root) / f"{episode:05d}"
    d.mkdir(parents=True)
    np.save(d / "components_q.npy", components_q)
    np.save(d / "lambdas.npy", np.asarray(lambdas, dtype=float))


def run_export(snapshots, out, expected_return, extra=()):
    cmd = [sys.executable, str(EXPORT_SCRIPT), "--snapshots", str(snapshots),
           "--out", str(out), "--expected-return", *map(str, expected_return),
           "--lattice-step", "2", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_data_blob(html):
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    return json.loads(html[start:end])


@pytest.fixture(scope="module")
def all_zero_snapshots(tmp_path_factory):
    """Two episodes of all-zero Q-tables: every state ties everywhere, so the
    generator's deterministic per-state tie-break (see export.py's module
    docstring) collapses every weighting to one policy that never reaches the
    gate -- a known, hand-verified rollout (returns 0/0/0)."""
    root = tmp_path_factory.mktemp("snapshots")
    zeros = np.zeros((NUM_REWARDS, N_STATES, N_ACTIONS))
    write_snapshot(root, 0, zeros, [1 / 3, 1 / 3, 1 / 3])
    write_snapshot(root, 10, zeros, [0.2, 0.3, 0.5])
    return root


@pytest.fixture(scope="module")
def generated(all_zero_snapshots, tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "artifact.html"
    result = run_export(all_zero_snapshots, out, (0, 0, 0))
    assert result.returncode == 0, result.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    return html, parse_data_blob(html)


def test_lattice_cells_all_populated(generated):
    _, data = generated
    n_weightings = len(data["lattice"])
    assert n_weightings == 6  # step 2 over 3 components: C(4,2)
    for row in data["cellIndex"]:
        assert len(row) == n_weightings
        assert all(0 <= idx < len(data["policies"]) for idx in row)
    for idx in data["realIndex"]:
        assert 0 <= idx < len(data["policies"])


def test_behavior_classes_are_known(generated):
    _, data = generated
    known = set(data["meta"]["behaviorClasses"])
    assert known == {"gate_only", "gate_banana", "gate_chicken", "gate_both", "no_gate"}
    for entry in data["policies"]:
        assert entry["class"] in known


def test_all_zero_q_never_reaches_gate(generated):
    _, data = generated
    for entry in data["policies"]:
        assert entry["class"] == "no_gate"
        assert entry["returns"] == [0.0, 0.0, 0.0]


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


def test_fidelity_guard_aborts_on_mismatch(all_zero_snapshots, tmp_path):
    out = tmp_path / "should_not_exist.html"
    result = run_export(all_zero_snapshots, out, (100, 30, 70))
    assert result.returncode != 0
    assert "fidelity" in result.stderr.lower()
    assert not out.exists()
