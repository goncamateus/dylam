"""Offline regression test for the ablation scope's table generator.

Invokes ablation/table.py through its CLI (--out-path) and parses the
numbers back out of the tabular it writes. FIXTURE is transcribed from
sections/ablation.tex's tab:ablation/normalizer in the sibling DyLam-TMLR
repo -- not captured from this code. Unlike tau/rb/epsilon (see arms.py),
this sweep was run cleanly, so every row is expected to reproduce exactly.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

TABLE_SCRIPT = Path(__file__).resolve().parent.parent / "ablation" / "table.py"

# (label, mean, std, iqm, ci_lo, ci_hi, success_n, success_total, p)
FIXTURE = [
    ("Exponential, $g(\\zeta) = \\mathrm{e}^\\zeta - 1$",
     178.92, 33.76, 188.19, 153.19, 200.00, 7, 10, None),
    ("Linear ($\\ell_1$), $g(\\zeta) = \\zeta$",
     178.90, 33.74, 188.16, 153.22, 200.00, 7, 10, 0.912),
    ("Min--max", 126.19, 95.63, 142.40, 48.60, 200.00, 6, 10, 0.579),
    ("Min--max, reversed routing (control)",
     127.78, 6.96, 130.00, 126.30, 130.00, 0, 10, 0.0029),
]

ROW_RE = re.compile(
    r"^(?P<label>.+?) & \$(?P<mean>[\d.]+) \\pm (?P<std>[\d.]+)\$ & "
    r"\$(?P<iqm>[\d.]+)\\ \[(?P<lo>[\d.]+), (?P<hi>[\d.]+)\]\$ & "
    r"\$(?P<hit>\d+)/(?P<tot>\d+)\$ & (?P<p>---|[\d.]+) \\\\$"
)


def parse_table(tex):
    rows = {}
    for line in tex.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            g = m.groupdict()
            rows[g["label"]] = dict(
                mean=float(g["mean"]), std=float(g["std"]), iqm=float(g["iqm"]),
                ci=(float(g["lo"]), float(g["hi"])), hit=int(g["hit"]), tot=int(g["tot"]),
                p=None if g["p"] == "---" else float(g["p"]),
            )
    return rows


@pytest.fixture(scope="module")
def generated_table(tmp_path_factory):
    out_path = tmp_path_factory.mktemp("paper")
    subprocess.run([sys.executable, str(TABLE_SCRIPT), "--out-path", str(out_path)],
                   capture_output=True, text=True, check=True)
    fragment = out_path / "tables" / "ablation" / "normalizer.tex"
    assert fragment.exists()
    return parse_table(fragment.read_text())


@pytest.mark.parametrize("label,mean,std,iqm,lo,hi,hit,tot,p", FIXTURE)
def test_row_matches_manuscript(generated_table, label, mean, std, iqm, lo, hi, hit, tot, p):
    got = generated_table[label]
    assert got["mean"] == pytest.approx(mean, abs=6e-3)
    assert got["std"] == pytest.approx(std, abs=6e-3)
    assert got["iqm"] == pytest.approx(iqm, abs=6e-3)
    assert got["ci"][0] == pytest.approx(lo, abs=6e-3)
    assert got["ci"][1] == pytest.approx(hi, abs=6e-3)
    assert (got["hit"], got["tot"]) == (hit, tot)
    if p is None:
        assert got["p"] is None
    else:
        assert got["p"] == pytest.approx(p, rel=0.1)
