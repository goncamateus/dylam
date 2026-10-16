"""Generators for Table 1 / RQ1 performance (tab:res/trad/summary,
tab:res/trad/iqm, tab:res/trad/efficiency) and the RQ1 significance figures
quoted in prose in sections/results/trad/performance.tex (sec:res_trad_rq1).

Reads only the committed tidy CSVs under data/; never touches the network.
Per-seed summary is the mean of the final 10% of training
(core.stats.seed_summary). DyLam is compared against the strongest rival
method per environment with an exact two-sided Mann-Whitney U,
Holm-Bonferroni corrected within the three-comparison RQ1 family
(Chicken--Banana, HalfCheetah-v4, VSS-v0); the HalfCheetah env-return
re-scoring is reported with its own (uncorrected) test, since it re-scores
runs already in the family rather than adding a fourth independent
comparison.
\\mbox{DyLam-Scalar} and \\mbox{DyLam-OpenLoop} are mechanism ablations: rows
of the summary and IQM tables, tested against DyLam in their own Holm family
(mechanism_tests), and excluded from the RQ1 family and the efficiency table
because they are not rival methods (OpenLoop replays DyLam's own schedule).

Emits three tabular-environment-only fragments (no \\begin{table}, caption,
or label -- those stay authored prose in the manuscript). --format html
emits the same three tables as semantic HTML <table> fragments for the
Beyond-PDF submission (issue #42), one HTML file per LaTeX fragment (same
stem, .html suffix) -- built from the same `summary`/`frames` this module
already computes for the LaTeX side, never hand-ported.

Usage: python table.py [--out-path PATH] [--format {latex,html,both}]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from methods import CELLS
from scipy.stats import binomtest

from core import stats
from core.html import strong, table as html_table

DATA = Path(__file__).parent / "data"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
ALPHA = 0.05

COLUMNS = ["Chicken--Banana", "HalfCheetah-v4", "HalfCheetah-v4 (env return)", "VSS-v0"]
EFFICIENCY_COLUMNS = ["Chicken--Banana", "HalfCheetah-v4", "VSS-v0"]
ROW_ORDER = ["Base SO RL", "Q-Decomposition", "UDC", "Tuned-UDC", "DyLam"]
DECIMALS = {"Chicken--Banana": 3, "HalfCheetah-v4": 3,
            "HalfCheetah-v4 (env return)": 1, "VSS-v0": 3}
IQM_DECIMALS = {"Chicken--Banana": 2, "HalfCheetah-v4": 1,
                "HalfCheetah-v4 (env return)": 1, "VSS-v0": 3}
THRESHOLD_DECIMALS = {"Chicken--Banana": 2, "HalfCheetah-v4": 1, "VSS-v0": 3}
METRIC = {c.column: c.metric for c in CELLS}
ABLATIONS = ["DyLam-Scalar", "DyLam-OpenLoop"]
# Primal-dual baseline (ADR 0003): own block in the tables, own Holm family.
ADAPTIVE = "DyLam-Lagrangian"
# Chicken--Banana seeds count as solved when most final-10% episodes collect
# all three objects (0/1 rewards, max 3); the outcome is bimodal (2 or 3).
CB_SOLVED = 2.5


def _slug(column):
    return re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")


def _load(column):
    return pd.read_csv(DATA / f"{_slug(column)}.csv")


def _frames(df, method):
    """Per-seed raw frames for one method, in fetch order."""
    d = df[df["method"] == method]
    return [g for _, g in d.groupby("seed", sort=False)]


def compute():
    dfs = {col: _load(col) for col in COLUMNS}
    frames, summary = {}, {}
    for col in COLUMNS:
        for method in ROW_ORDER + ABLATIONS + [ADAPTIVE]:
            fs = _frames(dfs[col], method)
            if fs:
                frames[(method, col)] = fs
                summary[(method, col)] = [stats.seed_summary(f, METRIC[col]) for f in fs]
    return frames, summary


def family_tests(summary):
    """DyLam vs. the strongest rival method, per column; Holm within the 3-env family."""
    fam, secondary = [], []
    for col in COLUMNS:
        dylam = summary.get(("DyLam", col))
        if not dylam:
            continue
        rivals = {m: summary[(m, col)] for m in ROW_ORDER
                  if m != "DyLam" and (m, col) in summary}
        if not rivals:
            continue
        best = max(rivals, key=lambda m: np.mean(rivals[m]))
        u, p, r = stats.exact_mw(dylam, rivals[best])
        entry = (col, best, len(dylam), len(rivals[best]), u, p, r)
        (secondary if "env return" in col else fam).append(entry)
    adj = dict(zip((e[0] for e in fam), stats.holm([e[5] for e in fam])))
    return fam, secondary, adj


def mechanism_tests(summary):
    """Each ablation vs. DyLam, per column; Holm within the mechanism family."""
    fam = []
    for col in COLUMNS:
        dylam = summary.get(("DyLam", col))
        for abl in ABLATIONS:
            vals = summary.get((abl, col))
            if dylam and vals and "env return" not in col:
                fam.append((col, abl, *stats.exact_mw(dylam, vals)))
    return fam, stats.holm([e[3] for e in fam])


def adaptive_tests(summary):
    """DyLam vs. DyLam-Lagrangian per column; Holm within the three primary
    columns, the HalfCheetah env return tested uncorrected as in RQ1."""
    tests = {col: stats.exact_mw(summary[("DyLam", col)], summary[(ADAPTIVE, col)])
             for col in COLUMNS if ("DyLam", col) in summary and (ADAPTIVE, col) in summary}
    prim = [c for c in tests if "env return" not in c]
    adj = dict(zip(prim, stats.holm([tests[c][1] for c in prim])))
    return {col: (*t, adj.get(col, t[1])) for col, t in tests.items()}


def report(summary, fam, adj, mech, mech_adj):
    """Prose-quoted numbers: RQ1 tests, mechanism family, CB solved counts."""
    for col, best, n1, n2, u, p, r in fam:
        print(f"RQ1 {col}: DyLam vs {best} U={u} p={p:.3g} p_holm={adj[col]:.3g} r={r:+.2f}",
              file=sys.stderr)
    for col, (u, p, r, ph) in adaptive_tests(summary).items():
        print(f"ADAPTIVE {col}: DyLam vs {ADAPTIVE} U={u} p={p:.3g} p_holm={ph:.3g} r={r:+.2f}",
              file=sys.stderr)
    for (col, abl, u, p, r), ph in zip(mech, mech_adj):
        print(f"MECH {col}: DyLam vs {abl} U={u} p={p:.3g} p_holm={ph:.3g} r={r:+.2f}",
              file=sys.stderr)
    succ = _load("Chicken--Banana success")
    for method, d in succ.groupby("method", sort=False):
        vals = [stats.seed_summary(g, METRIC["Chicken--Banana"]) for _, g in d.groupby("seed")]
        k, n = sum(v > CB_SOLVED for v in vals), len(vals)
        ci = binomtest(k, n).proportion_ci(method="exact")
        print(f"CB success {method}: {k}/{n} = {100 * k / n:.0f}% "
              f"[{100 * ci.low:.0f}, {100 * ci.high:.0f}]", file=sys.stderr)
    for method in ROW_ORDER + ABLATIONS + [ADAPTIVE]:
        vals = summary.get((method, "Chicken--Banana"))
        if vals:
            lo, hi = stats.boot_ci(vals)
            print(f"CB {method}: solved {sum(v > CB_SOLVED for v in vals)}/{len(vals)} "
                  f"mean {np.mean(vals):.3f}+-{np.std(vals, ddof=1):.3f} "
                  f"IQM {stats.iqm(vals):.2f} [{lo:.2f}, {hi:.2f}]", file=sys.stderr)


def _header(units_row):
    return [r"\begin{tabular}{lcccc}", r"\toprule",
            r"\textbf{Method} & \textbf{Chicken--Banana} & "
            r"\multicolumn{2}{c}{\textbf{HalfCheetah-v4}} & \textbf{VSS-v0} \\",
            r"\cmidrule(lr){3-4}",
            f"                & {units_row} \\\\",
            r"\midrule"]


def _row(method, cell):
    """One table row: `cell(method, col)` formats each of COLUMNS' cells."""
    return f"{method:16s}& " + " & ".join(cell(method, col) for col in COLUMNS) + r" \\"


def render_summary(summary, fam, secondary, adj):
    star_col = {col: p < ALPHA for col, *_, p, _ in secondary}
    star_col.update({col: adj[col] < ALPHA for col, *_ in fam})
    dagger = {col: t[3] < ALPHA and t[2] > 0 for col, t in adaptive_tests(summary).items()
              if "env return" not in col}

    def cell(method, col):
        vals = summary.get((method, col))
        if not vals:
            return "---"
        d = DECIMALS[col]
        star = r"^\ast" if method == "DyLam" and star_col.get(col) else ""
        star += r"^\dagger" if method == ADAPTIVE and dagger.get(col) else ""
        return (f"${np.mean(vals):.{d}f} \\pm {np.std(vals, ddof=1):.{d}f}{star}$ "
                f"\\, ($n{{=}}{len(vals)}$)")

    lines = _header(r"(final episode reward, max $3$) & (final $x$-position) & "
                    r"(env.\ return) & (goal rate)")
    for method in ROW_ORDER:
        lines.append(_row(method, cell))
        if method == "DyLam":
            lines += [r"\midrule", r"\multicolumn{5}{l}{\emph{Adaptive baseline}} \\",
                      _row(ADAPTIVE, cell), r"\midrule"]
            lines.append(r"\multicolumn{5}{l}{\emph{Ablation}} \\")
    lines += [_row(m, cell) for m in ABLATIONS]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_summary_html(summary, fam, secondary, adj):
    star_col = {col: p < ALPHA for col, *_, p, _ in secondary}
    star_col.update({col: adj[col] < ALPHA for col, *_ in fam})

    def cell(method, col):
        vals = summary.get((method, col))
        if not vals:
            return "—"
        d = DECIMALS[col]
        star = "*" if method == "DyLam" and star_col.get(col) else ""
        return f"{np.mean(vals):.{d}f} ± {np.std(vals, ddof=1):.{d}f}{star} (n={len(vals)})"

    headers = ["Method"] + COLUMNS
    body = [[method] + [cell(method, col) for col in COLUMNS] for method in ROW_ORDER]
    body += [[m] + [cell(m, col) for col in COLUMNS] for m in [ADAPTIVE] + ABLATIONS]
    return html_table(headers, body)


def render_iqm(summary):
    def cell(method, col):
        vals = summary.get((method, col))
        if not vals:
            return "---"
        lo, hi = stats.boot_ci(vals)
        d = IQM_DECIMALS[col]
        return f"${stats.iqm(vals):.{d}f}\\ [{lo:.{d}f}, {hi:.{d}f}]$"

    lines = _header(r"IQM $[95\%\ \mathrm{CI}]$ & $x$-position & env.\ return & goal rate")
    for method in ROW_ORDER:
        lines.append(_row(method, cell))
        if method == "DyLam":
            lines += [r"\midrule", _row(ADAPTIVE, cell), r"\midrule"]
    lines += [_row(m, cell) for m in ABLATIONS]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_iqm_html(summary):
    def cell(method, col):
        vals = summary.get((method, col))
        if not vals:
            return "—"
        lo, hi = stats.boot_ci(vals)
        d = IQM_DECIMALS[col]
        return f"{stats.iqm(vals):.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

    headers = ["Method"] + COLUMNS
    body = [[method] + [cell(method, col) for col in COLUMNS] for method in ROW_ORDER]
    body += [[m] + [cell(m, col) for col in COLUMNS] for m in [ADAPTIVE] + ABLATIONS]
    return html_table(headers, body)


def compute_efficiency(frames, summary):
    """Per efficiency column: the rival threshold and every method's
    (reached, median-budget, AUC) row. Shared by both renderers below."""
    out = {}
    for col in EFFICIENCY_COLUMNS:
        rivals = {m: summary[(m, col)] for m in ROW_ORDER
                  if m != "DyLam" and (m, col) in summary}
        dylam = summary.get(("DyLam", col))
        if not rivals or not dylam:
            continue
        best = max(rivals, key=lambda m: np.mean(rivals[m]))
        thr = float(np.mean(rivals[best]))
        allv = [x for m in list(rivals) + ["DyLam"] for x in summary[(m, col)]]
        lo, hi = min(allv), max(allv)
        computed = []
        for method in list(rivals) + ["DyLam"]:
            fs = frames[(method, col)]
            st = [stats.steps_to(f, METRIC[col], thr) for f in fs]
            hit = [s for s in st if s is not None]
            au = [stats.auc(f, METRIC[col], lo, hi) for f in fs]
            med = np.median(hit) if hit else None
            computed.append(dict(method=method, n_hit=len(hit), n_tot=len(fs), med=med,
                                 au_mean=np.mean(au),
                                 au_std=np.std(au, ddof=1) if len(au) > 1 else 0.0))
        best_auc = max(c["au_mean"] for c in computed)
        reached_medians = [c["med"] for c in computed if c["med"] is not None]
        best_median = min(reached_medians) if reached_medians else None
        for c in computed:
            c["bold_med"] = c["med"] is not None and c["med"] == best_median
            c["bold_auc"] = c["au_mean"] == best_auc
        out[col] = dict(threshold=thr, rows=computed)
    return out


def render_efficiency(data):
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"\textbf{Environment} & \textbf{Method} & \textbf{Reached} & "
             r"\textbf{Median budget} & \textbf{AUC} \\", r"\midrule"]
    for i, col in enumerate(EFFICIENCY_COLUMNS):
        if i:
            lines.append(r"\midrule")
        entry = data.get(col)
        if entry is None:
            continue
        thr = entry["threshold"]
        rows = entry["rows"]
        for j, c in enumerate(rows):
            if c["med"] is None:
                med_tex = "never"
            elif col == "Chicken--Banana":
                num = f"{c['med']:.0f}"
                med_tex = (f"$\\mathbf{{{num}}}$ ep." if c["bold_med"] else f"${num}$ ep.")
            else:
                num = f"{c['med'] / 1000:.0f}"
                med_tex = (f"$\\mathbf{{{num}}}$k steps" if c["bold_med"] else f"${num}$k steps")
            auc_num = f"{c['au_mean']:.3f} \\pm {c['au_std']:.3f}"
            auc_tex = f"$\\mathbf{{{auc_num}}}$" if c["bold_auc"] else f"${auc_num}$"
            if j == 0:
                thr_str = f"{thr:.{THRESHOLD_DECIMALS[col]}f}"
                lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{{col} ($\\geq {thr_str}$)}}")
            lines.append(f"  & {c['method']:16s} & ${c['n_hit']}/{c['n_tot']}$ & "
                         f"{med_tex} & {auc_tex} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_efficiency_html(data):
    headers = ["Environment", "Method", "Reached", "Median budget", "AUC"]
    body = []
    for col in EFFICIENCY_COLUMNS:
        entry = data.get(col)
        if entry is None:
            continue
        thr = entry["threshold"]
        thr_str = f"{thr:.{THRESHOLD_DECIMALS[col]}f}"
        rows = entry["rows"]
        for j, c in enumerate(rows):
            if c["med"] is None:
                med_str = "never"
            elif col == "Chicken--Banana":
                med_str = f"{c['med']:.0f} ep."
            else:
                med_str = f"{c['med'] / 1000:.0f}k steps"
            med_cell = strong(med_str) if c["bold_med"] else med_str
            auc_str = f"{c['au_mean']:.3f} ± {c['au_std']:.3f}"
            auc_cell = strong(auc_str) if c["bold_auc"] else auc_str
            env_cell = f"{col} (≥ {thr_str})" if j == 0 else ""
            body.append([env_cell, c["method"], f"{c['n_hit']}/{c['n_tot']}",
                        med_cell, auc_cell])
    return html_table(headers, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    ap.add_argument("--format", choices=["latex", "html", "both"], default="latex",
                     help="output format(s) to write (default: %(default)s)")
    args = ap.parse_args()

    frames, summary = compute()
    fam, secondary, adj = family_tests(summary)
    efficiency = compute_efficiency(frames, summary)
    report(summary, fam, adj, *mechanism_tests(summary))

    if args.format in ("latex", "both"):
        outputs = {
            "tables/results/trad_summary.tex": render_summary(summary, fam, secondary, adj),
            "tables/results/trad_iqm.tex": render_iqm(summary),
            "tables/results/trad_efficiency.tex": render_efficiency(efficiency),
        }
        for rel, tex in outputs.items():
            out = args.out_path / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(tex)
            print(tex)
            print(f"wrote {out}", file=sys.stderr)

    if args.format in ("html", "both"):
        outputs = {
            "tables/results/trad_summary.html": render_summary_html(summary, fam, secondary, adj),
            "tables/results/trad_iqm.html": render_iqm_html(summary),
            "tables/results/trad_efficiency.html": render_efficiency_html(efficiency),
        }
        for rel, html in outputs.items():
            out = args.out_path / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html)
            print(html)
            print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
