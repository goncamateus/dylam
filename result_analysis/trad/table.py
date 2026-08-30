"""Generators for Table 1 / RQ1 performance (tab:res/trad/summary,
tab:res/trad/iqm, tab:res/trad/efficiency) and the RQ1 significance figures
quoted in prose in sections/results/trad/performance.tex (sec:res_trad_rq1).

Reads only the committed tidy CSVs under data/; never touches the network.
Per-seed summary is the mean of the final 10% of training
(lib.stats.seed_summary). DyLam is compared against the strongest rival
method per environment with an exact two-sided Mann-Whitney U,
Holm-Bonferroni corrected within the three-comparison RQ1 family
(Chicken--Banana, HalfCheetah-v4, VSS-v0); the HalfCheetah env-return
re-scoring is reported with its own (uncorrected) test, since it re-scores
runs already in the family rather than adding a fourth independent
comparison.
\\mbox{DyLam-Scalar} is the scalar-critic ablation of Section 6.1.1: a row
of the summary and IQM tables, excluded from the RQ1 family (it is not a
rival method) and from the efficiency table (it never learns the task, so
sample-efficiency is not a meaningful comparison for it).

Emits three tabular-environment-only fragments (no \\begin{table}, caption,
or label -- those stay authored prose in the manuscript).

Usage: python table.py [--out-path PATH]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from methods import CELLS

from core import stats

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
        for method in ROW_ORDER + ["DyLam-Scalar"]:
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

    def cell(method, col):
        vals = summary.get((method, col))
        if not vals:
            return "---"
        d = DECIMALS[col]
        star = r"^\ast" if method == "DyLam" and star_col.get(col) else ""
        return (f"${np.mean(vals):.{d}f} \\pm {np.std(vals, ddof=1):.{d}f}{star}$ "
                f"\\, ($n{{=}}{len(vals)}$)")

    lines = _header(r"(final episode reward, max $200$) & (final $x$-position) & "
                    r"(env.\ return) & (goal rate)")
    for method in ROW_ORDER:
        lines.append(_row(method, cell))
        if method == "DyLam":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{5}{l}{\emph{Ablation}} \\")
    lines.append(_row("DyLam-Scalar", cell))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


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
            lines.append(r"\midrule")
    lines.append(_row("DyLam-Scalar", cell))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_efficiency(frames, summary):
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"\textbf{Environment} & \textbf{Method} & \textbf{Reached} & "
             r"\textbf{Median budget} & \textbf{AUC} \\", r"\midrule"]
    for i, col in enumerate(EFFICIENCY_COLUMNS):
        if i:
            lines.append(r"\midrule")
        rivals = {m: summary[(m, col)] for m in ROW_ORDER
                  if m != "DyLam" and (m, col) in summary}
        dylam = summary.get(("DyLam", col))
        if not rivals or not dylam:
            continue
        best = max(rivals, key=lambda m: np.mean(rivals[m]))
        thr = float(np.mean(rivals[best]))
        allv = [x for m in list(rivals) + ["DyLam"] for x in summary[(m, col)]]
        lo, hi = min(allv), max(allv)
        rows = list(rivals) + ["DyLam"]
        best_auc, best_median = None, None
        computed = []
        for method in rows:
            fs = frames[(method, col)]
            st = [stats.steps_to(f, METRIC[col], thr) for f in fs]
            hit = [s for s in st if s is not None]
            au = [stats.auc(f, METRIC[col], lo, hi) for f in fs]
            med = np.median(hit) if hit else None
            computed.append((method, len(hit), len(fs), med, np.mean(au),
                             np.std(au, ddof=1) if len(au) > 1 else 0.0))
        best_auc = max(c[4] for c in computed)
        reached_medians = [c[3] for c in computed if c[3] is not None]
        best_median = min(reached_medians) if reached_medians else None
        for j, (method, n_hit, n_tot, med, au_mean, au_std) in enumerate(computed):
            bold_med = med is not None and med == best_median
            bold_auc = au_mean == best_auc
            if med is None:
                med_tex = "never"
            elif col == "Chicken--Banana":
                num = f"{med:.0f}"
                med_tex = (f"$\\mathbf{{{num}}}$ ep." if bold_med else f"${num}$ ep.")
            else:
                num = f"{med / 1000:.0f}"
                med_tex = (f"$\\mathbf{{{num}}}$k steps" if bold_med else f"${num}$k steps")
            auc_num = f"{au_mean:.3f} \\pm {au_std:.3f}"
            auc_tex = f"$\\mathbf{{{auc_num}}}$" if bold_auc else f"${auc_num}$"
            if j == 0:
                thr_str = f"{thr:.{THRESHOLD_DECIMALS[col]}f}"
                lines.append(f"\\multirow{{{len(computed)}}}{{*}}{{{col} ($\\geq {thr_str}$)}}")
            lines.append(f"  & {method:16s} & ${n_hit}/{n_tot}$ & {med_tex} & {auc_tex} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    args = ap.parse_args()

    frames, summary = compute()
    fam, secondary, adj = family_tests(summary)

    outputs = {
        "tables/results/trad_summary.tex": render_summary(summary, fam, secondary, adj),
        "tables/results/trad_iqm.tex": render_iqm(summary),
        "tables/results/trad_efficiency.tex": render_efficiency(frames, summary),
    }
    for rel, tex in outputs.items():
        out = args.out_path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex)
        print(tex)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
