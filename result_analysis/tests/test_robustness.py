"""Offline regression test for the robustness scope's table generator.

Invokes robustness/table.py through its actual CLI contract (--out-path) and
parses the numbers back out of the tabular it writes, exactly as a reviewer
or a future refactor would need to. No network access.

FIXTURE is transcribed from the numbers currently published in
sections/results/trad/app.tex (Table tab:res/robustness/summary) in the
sibling DyLam-TMLR repository -- not captured from this code -- so a
mismatch here means either this migration has a bug, or the manuscript does.
mean/std/IQM/CI/U/r are reported to the precision the manuscript prints them
at and compared tightly; p and p_Holm are manuscript-rounded to 1-2
significant figures, so those use a looser tolerance, plus a check that the
significance call (< 0.05) agrees.

Also invokes the generator with --format html and checks the HTML sibling
fragment's rows carry the same values as the LaTeX rows -- both are rendered
from the same in-memory `compute_rows()` rows (see robustness/table.py).
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.html import detex  # noqa: E402

TABLE_SCRIPT = Path(__file__).resolve().parent.parent / "robustness" / "table.py"
ALPHA = 0.05

FIXTURE = {
    "Nominal": dict(mean=0.852, std=0.021, iqm=0.848, ci=(0.838, 0.865)),
    "Move $-25\\%$": dict(mean=0.898, std=0.016, iqm=0.899, ci=(0.885, 0.911),
                           U=95, p=2.1e-4, p_holm=0.0006, r=0.90),
    "Move $+50\\%$": dict(mean=0.607, std=0.120, iqm=0.589, ci=(0.524, 0.698),
                           U=0, p=1.1e-5, p_holm=0.00006, r=-1.00),
    "Ball $-25\\%$": dict(mean=0.754, std=0.070, iqm=0.775, ci=(0.706, 0.800),
                          U=0, p=1.1e-5, p_holm=0.00006, r=-1.00),
    "Ball $+25\\%$": dict(mean=0.870, std=0.023, iqm=0.874, ci=(0.856, 0.885),
                          U=73, p=0.089, p_holm=0.089, r=0.46),
    "Move $+50\\%$, ball $+25\\%$": dict(mean=0.737, std=0.111, iqm=0.757, ci=(0.651, 0.810),
                                         U=7, p=4.9e-4, p_holm=0.0010, r=-0.86),
    "Move $-25\\%$, ball $-50\\%$": dict(mean=0.670, std=0.036, iqm=0.663, ci=(0.646, 0.696),
                                         U=0, p=1.1e-5, p_holm=0.00006, r=-1.00),
}

ROW_RE = re.compile(
    r"^(?P<label>[A-Za-z][^&]*?) & \$[^$]*\$ & "
    r"\$(?P<mean>[\d.]+) \\pm (?P<std>[\d.]+)\$ & "
    r"\$(?P<iqm>[\d.]+)\\ \[(?P<lo>[\d.]+), (?P<hi>[\d.]+)\]\$ & "
    r"(?P<U>---|\$\d+\$) & (?P<p>---|\$[^$]+\$) & (?P<ph>---|\$[^$]+\$) & "
    r"(?P<r>---|\$[+-][\d.]+\$) \\\\$"
)


def parse_scientific(field):
    """'$6.2 \\times 10^{-4}^{\\ast}$' / '$0.089~\\text{n.s.}$' -> float."""
    s = field.strip("$")
    s = re.sub(r"\^\{\\ast\}$", "", s)
    s = re.sub(r"~\\text\{n\.s\.\}$", "", s)
    m = re.match(r"([\d.]+)(?:\s*\\times 10\^\{(-?\d+)\})?$", s)
    mant = float(m.group(1))
    return mant * 10 ** int(m.group(2)) if m.group(2) else mant


def parse_table(tex):
    rows = {}
    for line in tex.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        row = dict(mean=float(g["mean"]), std=float(g["std"]), iqm=float(g["iqm"]),
                   ci=(float(g["lo"]), float(g["hi"])))
        if g["U"] != "---":
            row["U"] = int(g["U"].strip("$"))
            row["p"] = parse_scientific(g["p"])
            row["p_holm"] = parse_scientific(g["ph"])
            row["r"] = float(g["r"].strip("$"))
        rows[g["label"]] = row
    return rows


def parse_scientific_html(field):
    """'1.1×10^-5' / '0.089' -> float."""
    m = re.match(r"([\d.]+)(?:×10\^(-?\d+))?$", field)
    mant = float(m.group(1))
    return mant * 10 ** int(m.group(2)) if m.group(2) else mant


def parse_html_table(html):
    soup = BeautifulSoup(html, "html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]
    rows = {}
    for tr in soup.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        cell = dict(zip(headers, cells))
        mean_str, std_str = cell["Goal rate"].split("±")
        iqm_str, ci_str = cell["IQM [95% CI]"].split("[")
        lo_str, hi_str = ci_str.rstrip("]").split(",")
        row = dict(mean=float(mean_str), std=float(std_str), iqm=float(iqm_str),
                   ci=(float(lo_str), float(hi_str)))
        if cell["U"] != "—":
            row["U"] = int(cell["U"])
            row["p"] = parse_scientific_html(cell["p"])
            ph, _, sig = cell["p_Holm"].partition(" ")
            row["p_holm"] = parse_scientific_html(ph)
            row["sig"] = sig == "*"
            row["r"] = float(cell["r"])
        rows[cell["Condition"]] = row
    return rows


@pytest.fixture(scope="module")
def generated_both(tmp_path_factory):
    out_path = tmp_path_factory.mktemp("paper")
    subprocess.run(
        [sys.executable, str(TABLE_SCRIPT), "--out-path", str(out_path), "--format", "both"],
        capture_output=True, text=True, check=True,
    )
    tex_fragment = out_path / "tables" / "results" / "robustness_summary.tex"
    html_fragment = out_path / "tables" / "results" / "robustness_summary.html"
    assert tex_fragment.exists()
    assert html_fragment.exists()
    return parse_table(tex_fragment.read_text()), parse_html_table(html_fragment.read_text())


@pytest.fixture(scope="module")
def generated_table(generated_both):
    return generated_both[0]


@pytest.fixture(scope="module")
def generated_html_table(generated_both):
    return generated_both[1]


def test_all_conditions_present(generated_table):
    assert set(generated_table) == set(FIXTURE)


@pytest.mark.parametrize("label", list(FIXTURE))
def test_condition_matches_manuscript(generated_table, label):
    got, want = generated_table[label], FIXTURE[label]
    assert got["mean"] == pytest.approx(want["mean"], abs=6e-4)
    assert got["std"] == pytest.approx(want["std"], abs=6e-4)
    assert got["iqm"] == pytest.approx(want["iqm"], abs=6e-4)
    assert got["ci"][0] == pytest.approx(want["ci"][0], abs=6e-4)
    assert got["ci"][1] == pytest.approx(want["ci"][1], abs=6e-4)

    if "U" not in want:  # nominal: reference row, no test statistics
        return
    assert got["U"] == want["U"]
    assert got["r"] == pytest.approx(want["r"], abs=6e-3)
    assert got["p"] == pytest.approx(want["p"], rel=0.25)
    assert got["p_holm"] == pytest.approx(want["p_holm"], rel=0.25)
    assert (got["p_holm"] < ALPHA) == (want["p_holm"] < ALPHA)


def test_html_all_conditions_present(generated_html_table):
    assert set(generated_html_table) == {detex(label) for label in FIXTURE}


@pytest.mark.parametrize("label", list(FIXTURE))
def test_html_condition_matches_manuscript(generated_html_table, label):
    got, want = generated_html_table[detex(label)], FIXTURE[label]
    assert got["mean"] == pytest.approx(want["mean"], abs=6e-4)
    assert got["std"] == pytest.approx(want["std"], abs=6e-4)
    assert got["iqm"] == pytest.approx(want["iqm"], abs=6e-4)
    assert got["ci"][0] == pytest.approx(want["ci"][0], abs=6e-4)
    assert got["ci"][1] == pytest.approx(want["ci"][1], abs=6e-4)

    if "U" not in want:
        return
    assert got["U"] == want["U"]
    assert got["r"] == pytest.approx(want["r"], abs=6e-3)
    assert got["p"] == pytest.approx(want["p"], rel=0.25)
    assert got["p_holm"] == pytest.approx(want["p_holm"], rel=0.25)
    assert got["sig"] == (want["p_holm"] < ALPHA)


def test_html_rows_match_latex_rows(generated_table, generated_html_table):
    """Same in-memory rows (robustness/table.py's `compute_rows()`) feed both
    renderers, so HTML and LaTeX must carry identical values exactly."""
    assert len(generated_table) == len(generated_html_table)
    for label, tex_row in generated_table.items():
        html_row = generated_html_table[detex(label)]
        assert html_row["mean"] == pytest.approx(tex_row["mean"], abs=1e-9)
        assert html_row["std"] == pytest.approx(tex_row["std"], abs=1e-9)
        assert html_row["iqm"] == pytest.approx(tex_row["iqm"], abs=1e-9)
        assert html_row["ci"] == pytest.approx(tex_row["ci"], abs=1e-9)
        if "U" in tex_row:
            assert html_row["U"] == tex_row["U"]
            assert html_row["r"] == pytest.approx(tex_row["r"], abs=1e-9)
            assert html_row["p"] == pytest.approx(tex_row["p"], rel=1e-6)
            assert html_row["p_holm"] == pytest.approx(tex_row["p_holm"], rel=1e-6)
