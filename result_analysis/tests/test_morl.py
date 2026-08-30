"""Offline regression test for the morl scope's table generator.

Invokes morl/table.py through its CLI (--out-path) and parses the numbers
back out of the tabular it writes. FIXTURE is transcribed from
sections/results/morl/app.tex's tab:res/pareto/hv-cardinality and its
Mann-Whitney p-values (app:res_morl_tests) in the sibling DyLam-TMLR repo
-- not captured from this code.

Every cell reproduces exactly except GPI-LS on MO-HalfCheetah: that
(env_id, algo) pair has 31 finished runs in the openrlbenchmark/
MORL-Baselines project (a static, unchanging 2023 benchmark dataset, all
the others have exactly 10), so "the last 10 by creation time" -- this
migration's fetch rule, same as every other scope -- does not necessarily
recover whichever 10 the manuscript was built from; no other selection
rule is recorded anywhere. Its raw mean/std are marked xfail. Its p-values
for HV and cardinality (which depend on the specific 10 runs) are also
xfail, but its time p-value is not: DyLam is faster than every one of the
31 candidate runs, so the exact Mann-Whitney U hits complete separation
(and the same minimal p) regardless of which 10 are drawn.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

TABLE_SCRIPT = Path(__file__).resolve().parent.parent / "morl" / "table.py"

GPILS_HC_AMBIGUOUS = pytest.mark.xfail(
    reason="GPI-LS/MO-HalfCheetah has 31 candidate runs, not 10; which 10 the "
           "manuscript used is not recorded, not a code bug",
    strict=False,
)


def _p(*args, xfail=False):
    return pytest.param(*args, marks=GPILS_HC_AMBIGUOUS) if xfail else pytest.param(*args)


# (method, env, metric, mean, std, tolerance)
MEANS_FIXTURE = [
    _p("PGMORL", "HC", "hv", 5.014, 0.111, 6e-3),
    _p("PGMORL", "HC", "cardinality", 21, 4, 0.6),
    _p("PGMORL", "HC", "wall_time_min", 445, 11, 0.6),
    _p("GPI-LS", "HC", "hv", 5.636, 0.285, 6e-3, xfail=True),
    _p("GPI-LS", "HC", "cardinality", 11, 10, 0.6, xfail=True),
    _p("GPI-LS", "HC", "wall_time_min", 6155, 3459, 0.6, xfail=True),
    _p("DyLam", "HC", "hv", 5.644, 0.048, 6e-3),
    _p("DyLam", "HC", "cardinality", 22, 9, 0.6),
    _p("DyLam", "HC", "wall_time_min", 148, 12, 0.6),
    _p("GPI-LS", "MC", "hv", 1.776, 2.040, 6e-3),
    _p("GPI-LS", "MC", "cardinality", 49, 23, 0.6),
    _p("GPI-LS", "MC", "wall_time_min", 436, 22, 0.6),
    _p("DynMORL", "MC", "hv", 3.048, 0.001, 6e-3),
    _p("DynMORL", "MC", "cardinality", 2949, 531, 0.6),
    _p("DynMORL", "MC", "wall_time_min", 1285, 222, 0.6),
    _p("DyLam", "MC", "hv", 3.045, 0.003, 6e-3),
    _p("DyLam", "MC", "cardinality", 5090, 235, 0.6),
    _p("DyLam", "MC", "wall_time_min", 40, 13, 0.6),
]

# (rival, env, metric, p_holm)
P_FIXTURE = [
    _p("GPI-LS", "MC", "hv", 4.3e-5),
    _p("GPI-LS", "HC", "hv", 0.089, xfail=True),
    _p("PGMORL", "HC", "hv", 4.3e-5),
    _p("DynMORL", "MC", "hv", 0.0042),
    _p("GPI-LS", "MC", "cardinality", 4.3e-5),
    _p("GPI-LS", "HC", "cardinality", 0.0058, xfail=True),
    _p("DynMORL", "MC", "cardinality", 4.3e-5),
    _p("PGMORL", "HC", "cardinality", 0.912),
    _p("GPI-LS", "HC", "wall_time_min", 4.3e-5),
    _p("GPI-LS", "MC", "wall_time_min", 4.3e-5),
    _p("PGMORL", "HC", "wall_time_min", 4.3e-5),
    _p("DynMORL", "MC", "wall_time_min", 4.3e-5),
]

CELL_ORDER = [("HC", "hv"), ("HC", "cardinality"), ("HC", "wall_time_min"),
              ("MC", "hv"), ("MC", "cardinality"), ("MC", "wall_time_min")]


def _row_cells(line):
    """Split 'Label  & c1 & c2 & ... & c6 \\\\' into (label, [c1..c6])."""
    body = line.rsplit(r"\\", 1)[0]
    parts = [p.strip() for p in body.split("&")]
    return parts[0].strip(), parts[1:]


def parse_table(tex):
    means, pvals = {}, {}
    for raw in tex.splitlines():
        line = raw.strip()
        if "vs." in line or not line or line.startswith(("\\", "%")):
            continue
        method, cells = _row_cells(line)
        if len(cells) != 6 or not re.match(r"^[A-Za-z][\w\-]*$", method):
            continue
        for (env, metric), cell in zip(CELL_ORDER, cells):
            m = re.search(r"([\d.]+) \$\\pm\$ ([\d.]+)", cell)
            if m:
                means[(method, env, metric)] = (float(m.group(1)), float(m.group(2)))

    for raw in tex.splitlines():
        line = raw.strip()
        m = re.match(r"\\textit\{DyLam\} vs\.\\ (\S+)", line)
        if not m:
            continue
        rival = m.group(1)
        _, cells = _row_cells(line)
        for (env, metric), cell in zip(CELL_ORDER, cells):
            m2 = re.search(r"([\d.eE+-]+)~", cell)
            if m2:
                pvals[(rival, env, metric)] = float(m2.group(1))
    return means, pvals


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out_path = tmp_path_factory.mktemp("paper")
    subprocess.run([sys.executable, str(TABLE_SCRIPT), "--out-path", str(out_path)],
                   capture_output=True, text=True, check=True)
    fragment = out_path / "tables" / "morl" / "hv_cardinality.tex"
    assert fragment.exists()
    return parse_table(fragment.read_text())


@pytest.mark.parametrize("method,env,metric,mean,std,tol", MEANS_FIXTURE)
def test_means_match_manuscript(generated, method, env, metric, mean, std, tol):
    means, _ = generated
    got_mean, got_std = means[(method, env, metric)]
    assert got_mean == pytest.approx(mean, abs=tol)
    assert got_std == pytest.approx(std, abs=tol)


@pytest.mark.parametrize("rival,env,metric,p", P_FIXTURE)
def test_pvalues_match_manuscript(generated, rival, env, metric, p):
    _, pvals = generated
    assert pvals[(rival, env, metric)] == pytest.approx(p, rel=0.15)
