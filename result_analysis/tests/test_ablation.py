"""Offline regression test for the ablation scope's table generator.

Invokes ablation/table.py through its CLI (--out-path) and parses the
numbers back out of the tabular it writes. FIXTURE is transcribed from
sections/ablation.tex's tab:ablation/normalizer in the sibling DyLam-TMLR
repo -- not captured from this code. Unlike tau/rb/epsilon (see arms.py),
this sweep was run cleanly, so every row is expected to reproduce exactly.

Also invokes the generator with --format html and checks the HTML sibling
fragment's rows carry the same values as the LaTeX rows -- both are rendered
from the same in-memory `compute()` rows (see ablation/table.py), so a
mismatch here would mean the two renderers drifted apart, not that the data
changed.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.html import detex  # noqa: E402

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


def parse_html_table(html):
    soup = BeautifulSoup(html, "html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]
    rows = {}
    for tr in soup.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        row = dict(zip(headers, cells))
        label = row["Transform"]
        mean_str, std_str = row["Reward"].split("±")
        iqm_str, ci_str = row["IQM [95% CI]"].split("[")
        lo_str, hi_str = ci_str.rstrip("]").split(",")
        hit_str, tot_str = row["Success"].split("/")
        p = row["p"]
        rows[label] = dict(
            mean=float(mean_str), std=float(std_str), iqm=float(iqm_str),
            ci=(float(lo_str), float(hi_str)), hit=int(hit_str), tot=int(tot_str),
            p=None if p in ("—", "---") else float(p),
        )
    return rows


@pytest.fixture(scope="module")
def generated_both(tmp_path_factory):
    out_path = tmp_path_factory.mktemp("paper")
    subprocess.run([sys.executable, str(TABLE_SCRIPT), "--out-path", str(out_path),
                    "--format", "both"],
                   capture_output=True, text=True, check=True)
    tex_fragment = out_path / "tables" / "ablation" / "normalizer.tex"
    html_fragment = out_path / "tables" / "ablation" / "normalizer.html"
    assert tex_fragment.exists()
    assert html_fragment.exists()
    return parse_table(tex_fragment.read_text()), parse_html_table(html_fragment.read_text())


@pytest.fixture(scope="module")
def generated_table(generated_both):
    return generated_both[0]


@pytest.fixture(scope="module")
def generated_html_table(generated_both):
    return generated_both[1]


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


@pytest.mark.parametrize("label,mean,std,iqm,lo,hi,hit,tot,p", FIXTURE)
def test_html_row_matches_manuscript(generated_html_table, label, mean, std, iqm,
                                     lo, hi, hit, tot, p):
    got = generated_html_table[detex(label)]
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


def test_html_rows_match_latex_rows(generated_table, generated_html_table):
    """The two renderers consume the same in-memory rows (ablation/table.py's
    `compute()`), so HTML and LaTeX must carry identical values exactly --
    this is the drift a hand-ported second table would not catch."""
    assert len(generated_table) == len(generated_html_table)
    for label, tex_row in generated_table.items():
        html_row = generated_html_table[detex(label)]
        assert html_row == tex_row
