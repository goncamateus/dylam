"""Offline regression test for the trad scope's table generator.

Invokes trad/table.py through its actual CLI contract (--out-path) and
parses the numbers back out of the three tabulars it writes.

FIXTURE holds the numbers published in the DyLam-TNNLS manuscript
(Table II, the DyLam-Lagrangian adaptive-baseline row, and the AUC /
threshold figures quoted in sections/results/trad/performance.tex), not
values captured blindly from this code. Chicken--Banana is the 0/1-reward
rerun on seeds 1..10; every cell has n=10.

Also invokes the generator with --format html and checks the HTML sibling
fragments' rows carry the same values as the LaTeX rows -- both are rendered
from the same in-memory `summary`/`frames`/`compute_efficiency()` values
(see trad/table.py).
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

TABLE_SCRIPT = Path(__file__).resolve().parent.parent / "trad" / "table.py"

def _p(*args):
    return pytest.param(*args)


SUMMARY_FIXTURE = [
   _p("Chicken--Banana", "Base SO RL", dict(mean=2.200, std=0.422, n=10)),
   _p("Chicken--Banana", "Q-Decomposition", dict(mean=2.124, std=0.614, n=10)),
   _p("Chicken--Banana", "UDC", dict(mean=2.100, std=0.316, n=10)),
   _p("Chicken--Banana", "DyLam", dict(mean=2.697, std=0.481, n=10)),
   _p("Chicken--Banana", "DyLam-Scalar", dict(mean=0.175, std=0.287, n=10)),
   _p("Chicken--Banana", "DyLam-OpenLoop", dict(mean=2.198, std=0.423, n=10)),
   _p("Chicken--Banana", "DyLam-Lagrangian", dict(mean=2.100, std=0.316, n=10)),
   _p("HalfCheetah-v4", "Base SO RL", dict(mean=409.407, std=36.048, n=10)),
   _p("HalfCheetah-v4", "UDC", dict(mean=284.942, std=83.861, n=10)),
   _p("HalfCheetah-v4", "DyLam", dict(mean=464.568, std=47.627, n=10)),
   _p("HalfCheetah-v4", "DyLam-Scalar", dict(mean=481.483, std=11.764, n=10)),
   _p("HalfCheetah-v4", "DyLam-OpenLoop", dict(mean=466.689, std=54.181, n=10)),
   _p("HalfCheetah-v4", "DyLam-Lagrangian", dict(mean=379.891, std=32.993, n=10)),
   _p("HalfCheetah-v4 (env return)", "Base SO RL", dict(mean=8043.4, std=708.8, n=10)),
   _p("HalfCheetah-v4 (env return)", "UDC", dict(mean=5593.5, std=1649.5, n=10)),
   _p("HalfCheetah-v4 (env return)", "DyLam", dict(mean=9086.5, std=962.1, n=10)),
   _p("HalfCheetah-v4 (env return)", "DyLam-Scalar", dict(mean=9426.3, std=239.1, n=10)),
   _p("HalfCheetah-v4 (env return)", "DyLam-OpenLoop", dict(mean=9141.9, std=1069.0, n=10)),
   _p("HalfCheetah-v4 (env return)", "DyLam-Lagrangian", dict(mean=7370.5, std=653.2, n=10)),
   _p("VSS-v0", "Base SO RL", dict(mean=0.070, std=0.023, n=10)),
   _p("VSS-v0", "UDC", dict(mean=0.054, std=0.011, n=10)),
   _p("VSS-v0", "Tuned-UDC", dict(mean=0.388, std=0.116, n=10)),
   _p("VSS-v0", "DyLam", dict(mean=0.852, std=0.021, n=10)),
   _p("VSS-v0", "DyLam-Scalar", dict(mean=0.669, std=0.032, n=10)),
   _p("VSS-v0", "DyLam-OpenLoop", dict(mean=0.759, std=0.134, n=10)),
   _p("VSS-v0", "DyLam-Lagrangian", dict(mean=0.853, std=0.048, n=10)),
]

IQM_FIXTURE = [
   _p("Chicken--Banana", "Base SO RL", (2.00, 2.00, 2.50)),
   _p("Chicken--Banana", "Q-Decomposition", (2.01, 1.70, 2.62)),
   _p("Chicken--Banana", "UDC", (2.00, 2.00, 2.17)),
   _p("Chicken--Banana", "DyLam", (2.83, 2.33, 3.00)),
   _p("Chicken--Banana", "DyLam-Scalar", (0.08, 0.00, 0.35)),
   _p("Chicken--Banana", "DyLam-OpenLoop", (2.00, 2.00, 2.50)),
   _p("Chicken--Banana", "DyLam-Lagrangian", (2.00, 2.00, 2.17)),
   _p("HalfCheetah-v4", "Base SO RL", (407.1, 382.6, 436.8)),
   _p("HalfCheetah-v4", "UDC", (271.7, 218.4, 347.5)),
   _p("HalfCheetah-v4", "DyLam", (472.6, 434.0, 496.9)),
   _p("HalfCheetah-v4", "DyLam-Scalar", (481.9, 473.1, 490.3)),
   _p("HalfCheetah-v4", "DyLam-OpenLoop", (481.0, 435.9, 500.3)),
   _p("HalfCheetah-v4", "DyLam-Lagrangian", (380.0, 356.1, 401.8)),
   _p("HalfCheetah-v4 (env return)", "Base SO RL", (7998.9, 7516.5, 8581.7)),
   _p("HalfCheetah-v4 (env return)", "UDC", (5332.7, 4283.2, 6822.8)),
   _p("HalfCheetah-v4 (env return)", "DyLam", (9249.9, 8467.7, 9742.9)),
   _p("HalfCheetah-v4 (env return)", "DyLam-Scalar", (9433.5, 9256.3, 9605.8)),
   _p("HalfCheetah-v4 (env return)", "DyLam-OpenLoop", (9423.5, 8537.2, 9806.0)),
   _p("HalfCheetah-v4 (env return)", "DyLam-Lagrangian", (7368.8, 6897.4, 7802.3)),
   _p("VSS-v0", "Base SO RL", (0.071, 0.056, 0.087)),
   _p("VSS-v0", "UDC", (0.052, 0.046, 0.061)),
   _p("VSS-v0", "Tuned-UDC", (0.374, 0.317, 0.446)),
   _p("VSS-v0", "DyLam", (0.848, 0.838, 0.865)),
   _p("VSS-v0", "DyLam-Scalar", (0.671, 0.651, 0.689)),
   _p("VSS-v0", "DyLam-OpenLoop", (0.791, 0.663, 0.847)),
   _p("VSS-v0", "DyLam-Lagrangian", (0.859, 0.818, 0.889)),
]

# (column, method, reached, total, median value or None, unit, auc_mean, auc_std)
EFFICIENCY_FIXTURE = [
   _p("Chicken--Banana", "Base SO RL", 2, 10, 135, "ep", 0.419, 0.272),
   _p("Chicken--Banana", "Q-Decomposition", 8, 10, 382, "ep", 0.494, 0.244),
   _p("Chicken--Banana", "UDC", 1, 10, 140, "ep", 0.353, 0.204),
   _p("Chicken--Banana", "DyLam", 7, 10, 551, "ep", 0.596, 0.266),
   _p("HalfCheetah-v4", "Base SO RL", 5, 10, 359, "k", 0.379, 0.069),
   _p("HalfCheetah-v4", "UDC", 1, 10, 418, "k", 0.128, 0.146),
   _p("HalfCheetah-v4", "DyLam", 9, 10, 227, "k", 0.541, 0.122),
   _p("VSS-v0", "Base SO RL", 1, 10, 492, "k", 0.068, 0.007),
   _p("VSS-v0", "UDC", 6, 10, 301, "k", 0.077, 0.006),
   _p("VSS-v0", "Tuned-UDC", 10, 10, 112, "k", 0.338, 0.050),
   _p("VSS-v0", "DyLam", 10, 10, 102, "k", 0.670, 0.030),
]


TABLE_COLUMNS = ["Chicken--Banana", "HalfCheetah-v4", "HalfCheetah-v4 (env return)", "VSS-v0"]


def _rows(tex):
    return [ln.strip() for ln in tex.splitlines() if re.match(r"[A-Za-z]", ln.strip())]


def _parse_grid(tex, pattern, build):
    """One (column, method) -> build(match) entry per cell matching `pattern`."""
    out = {}
    for line in _rows(tex):
        cols = [c.strip() for c in line.split("&")]
        method = cols[0].strip()
        for col, cell in zip(TABLE_COLUMNS, cols[1:5]):
            m = re.search(pattern, cell)
            if m:
                out[(col, method)] = build(m)
    return out


def parse_summary(tex):
    return _parse_grid(
        tex, r"\$([\d.]+) \\pm ([\d.]+)(\^\\ast)?(?:\^\\dagger)?\$.*n\{=\}(\d+)",
        lambda m: dict(mean=float(m.group(1)), std=float(m.group(2)),
                      n=int(m.group(4)), star=bool(m.group(3))),
    )


def parse_iqm(tex):
    return _parse_grid(
        tex, r"\$([\d.]+)\\ \[([\d.]+), ([\d.]+)\]\$",
        lambda m: tuple(float(g) for g in m.groups()),
    )


def parse_efficiency(tex):
    out = {}
    column = None
    for raw in tex.splitlines():
        line = raw.strip()
        m = re.match(r"\\multirow\{\d+\}\{\*\}\{(.+?) \(", line)
        if m:
            column = m.group(1)
            continue
        m = re.match(r"& ([A-Za-z][\w\- ]*?)\s*& \$(\d+)/(\d+)\$ & (.+?) & \$(.+?)\$ \\\\", line)
        if not m or column is None:
            continue
        method, hit, tot, med_field, auc_field = m.groups()
        method = method.strip()
        if med_field.strip() == "never":
            med, unit = None, None
        else:
            mm = re.search(r"\$(?:\\mathbf\{)?(\d+)\}?\$(k)? ?(?:steps?|ep\.)", med_field)
            med, unit = int(mm.group(1)), ("k" if mm.group(2) else "ep")
        auc_mean, auc_std = (float(x) for x in re.search(
            r"([\d.]+) \\pm ([\d.]+)", auc_field).groups())
        out[(column, method)] = (int(hit), int(tot), med, unit, auc_mean, auc_std)
    return out


def _html_rows(html):
    soup = BeautifulSoup(html, "html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]
    for tr in soup.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        yield dict(zip(headers, cells))


def parse_summary_html(html):
    out = {}
    for row in _html_rows(html):
        method = row["Method"]
        for col in TABLE_COLUMNS:
            m = re.search(r"([\d.]+) ± ([\d.]+)(\*)? \(n=(\d+)\)", row[col])
            if m:
                out[(col, method)] = dict(mean=float(m.group(1)), std=float(m.group(2)),
                                          n=int(m.group(4)), star=bool(m.group(3)))
    return out


def parse_iqm_html(html):
    out = {}
    for row in _html_rows(html):
        method = row["Method"]
        for col in TABLE_COLUMNS:
            m = re.search(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", row[col])
            if m:
                out[(col, method)] = tuple(float(g) for g in m.groups())
    return out


def parse_efficiency_html(html):
    out = {}
    column = None
    for row in _html_rows(html):
        if row["Environment"]:
            column = row["Environment"].split(" (")[0]
        method = row["Method"]
        hit, tot = (int(x) for x in row["Reached"].split("/"))
        med_field = row["Median budget"]
        if med_field == "never":
            med, unit = None, None
        else:
            mm = re.match(r"(\d+)(k)? ?(?:steps?|ep\.)", med_field)
            med, unit = int(mm.group(1)), ("k" if mm.group(2) else "ep")
        auc_mean, auc_std = (float(x) for x in
                             re.search(r"([\d.]+) ± ([\d.]+)", row["AUC"]).groups())
        out[(column, method)] = (hit, tot, med, unit, auc_mean, auc_std)
    return out


@pytest.fixture(scope="module")
def generated_all(tmp_path_factory):
    out_path = tmp_path_factory.mktemp("paper")
    subprocess.run([sys.executable, str(TABLE_SCRIPT), "--out-path", str(out_path),
                    "--format", "both"],
                   capture_output=True, text=True, check=True)
    root = out_path / "tables" / "results"
    tex = dict(
        summary=parse_summary((root / "trad_summary.tex").read_text()),
        iqm=parse_iqm((root / "trad_iqm.tex").read_text()),
        efficiency=parse_efficiency((root / "trad_efficiency.tex").read_text()),
    )
    html = dict(
        summary=parse_summary_html((root / "trad_summary.html").read_text()),
        iqm=parse_iqm_html((root / "trad_iqm.html").read_text()),
        efficiency=parse_efficiency_html((root / "trad_efficiency.html").read_text()),
    )
    return tex, html


@pytest.fixture(scope="module")
def generated(generated_all):
    return generated_all[0]


@pytest.fixture(scope="module")
def generated_html(generated_all):
    return generated_all[1]


SUMMARY_TOL = {  # half the last displayed decimal place, per column
    "Chicken--Banana": 5e-4,
    "HalfCheetah-v4": 5e-4,
    "HalfCheetah-v4 (env return)": 5e-2,
    "VSS-v0": 5e-4,
}


@pytest.mark.parametrize("col,method,want", SUMMARY_FIXTURE)
def test_summary_matches_manuscript(generated, col, method, want):
    got = generated["summary"][(col, method)]
    tol = SUMMARY_TOL[col]
    assert got["mean"] == pytest.approx(want["mean"], abs=tol)
    assert got["std"] == pytest.approx(want["std"], abs=tol)
    assert got["n"] == want["n"]


IQM_TOL = {  # half the last displayed decimal place, per column
    "Chicken--Banana": 5e-3,
    "HalfCheetah-v4": 5e-2,
    "HalfCheetah-v4 (env return)": 5e-2,
    "VSS-v0": 5e-4,
}


@pytest.mark.parametrize("col,method,want", IQM_FIXTURE)
def test_iqm_matches_manuscript(generated, col, method, want):
    got = generated["iqm"][(col, method)]
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=IQM_TOL[col])


@pytest.mark.parametrize("col,method,hit,tot,med,unit,auc_mean,auc_std", EFFICIENCY_FIXTURE)
def test_efficiency_matches_manuscript(generated, col, method, hit, tot, med, unit,
                                       auc_mean, auc_std):
    g_hit, g_tot, g_med, g_unit, g_auc_mean, g_auc_std = generated["efficiency"][(col, method)]
    assert (g_hit, g_tot) == (hit, tot)
    assert g_med == med
    assert g_unit == unit
    assert g_auc_mean == pytest.approx(auc_mean, abs=6e-4)
    assert g_auc_std == pytest.approx(auc_std, abs=6e-4)


@pytest.mark.parametrize("col,method,want", SUMMARY_FIXTURE)
def test_html_summary_matches_manuscript(generated_html, col, method, want):
    got = generated_html["summary"][(col, method)]
    tol = SUMMARY_TOL[col]
    assert got["mean"] == pytest.approx(want["mean"], abs=tol)
    assert got["std"] == pytest.approx(want["std"], abs=tol)
    assert got["n"] == want["n"]


@pytest.mark.parametrize("col,method,want", IQM_FIXTURE)
def test_html_iqm_matches_manuscript(generated_html, col, method, want):
    got = generated_html["iqm"][(col, method)]
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=IQM_TOL[col])


@pytest.mark.parametrize("col,method,hit,tot,med,unit,auc_mean,auc_std", EFFICIENCY_FIXTURE)
def test_html_efficiency_matches_manuscript(generated_html, col, method, hit, tot, med, unit,
                                            auc_mean, auc_std):
    g_hit, g_tot, g_med, g_unit, g_auc_mean, g_auc_std = \
        generated_html["efficiency"][(col, method)]
    assert (g_hit, g_tot) == (hit, tot)
    assert g_med == med
    assert g_unit == unit
    assert g_auc_mean == pytest.approx(auc_mean, abs=6e-4)
    assert g_auc_std == pytest.approx(auc_std, abs=6e-4)


def test_html_matches_latex(generated, generated_html):
    """Same in-memory `summary`/`frames`/`compute_efficiency()` values feed
    both renderers (trad/table.py), so every cell in the LaTeX tables must
    appear in the HTML tables with the identical value."""
    assert set(generated["summary"]) == set(generated_html["summary"])
    for key, tex_row in generated["summary"].items():
        html_row = generated_html["summary"][key]
        assert html_row["mean"] == pytest.approx(tex_row["mean"], abs=1e-9)
        assert html_row["std"] == pytest.approx(tex_row["std"], abs=1e-9)
        assert html_row["n"] == tex_row["n"]
        assert html_row["star"] == tex_row["star"]

    assert set(generated["iqm"]) == set(generated_html["iqm"])
    for key, tex_val in generated["iqm"].items():
        html_val = generated_html["iqm"][key]
        for g, w in zip(html_val, tex_val):
            assert g == pytest.approx(w, abs=1e-9)

    assert set(generated["efficiency"]) == set(generated_html["efficiency"])
    for key, tex_row in generated["efficiency"].items():
        h_hit, h_tot, h_med, h_unit, h_auc_mean, h_auc_std = generated_html["efficiency"][key]
        t_hit, t_tot, t_med, t_unit, t_auc_mean, t_auc_std = tex_row
        assert (h_hit, h_tot) == (t_hit, t_tot)
        assert h_med == t_med
        assert h_unit == t_unit
        assert h_auc_mean == pytest.approx(t_auc_mean, abs=1e-9)
        assert h_auc_std == pytest.approx(t_auc_std, abs=1e-9)
