"""Generator for Table~\\ref{tab:res/robustness/summary} (VSS-v0 bound misspecification).

Owns: tab:res/robustness/summary in sections/results/trad/app.tex, and by
extension the goal rates / U / p / p_Holm / r figures quoted in prose in
sections/results/robust.tex (sec:res_robustness) -- both read off this same
table.

Reads only the committed tidy CSV under data/; never touches the network.
Per-seed summary is the mean of the final 10% of training (core.stats.seed_summary).
Comparisons are exact two-sided Mann-Whitney U against nominal, Holm-Bonferroni
corrected within the six-comparison RQ3 family, with rank-biserial effect size.

Emits the tabular environment only (no \\begin{table}, caption, or label --
those stay authored prose in the manuscript). p and p_Holm are both rendered
scientific below 1e-3 and fixed-point otherwise; this is a normalization of
the previously hand-typed table, which mixed conventions between the two
columns -- the numbers are unchanged, only their typesetting rule is now one
rule instead of two.

--format html emits the same rows as a semantic HTML <table> fragment for
the Beyond-PDF submission (issue #42) -- generated from the same `compute()`
values as the LaTeX, never hand-ported.

Usage: python table.py [--out-path PATH] [--format {latex,html,both}]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from conditions import CONDITIONS

from core import stats
from core.html import detex, table as html_table

METRIC = "ep_info/Goal"
DATA = Path(__file__).parent / "data" / "vss_ep_info_goal.csv"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
FRAGMENT = Path("tables/results/robustness_summary.tex")
HTML_FRAGMENT = FRAGMENT.with_suffix(".html")
ALPHA = 0.05

# Row order: grouped by which ceiling moved, matching the published table.
ROW_ORDER = [
    "nominal", "move_m25", "move_p50", "ball_m25", "ball_p25",
    "compound_move_p50_ball_p25", "compound_move_m25_ball_m50",
]


def fmt_p(p):
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / 10 ** exp
        return f"{mant:.1f} \\times 10^{{{exp}}}"
    return f"{p:.3f}"


def fmt_p_html(p):
    """Plain-text sibling of fmt_p -- same mantissa/exponent, no LaTeX."""
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / 10 ** exp
        return f"{mant:.1f}×10^{exp}"
    return f"{p:.3f}"


def per_seed(df, condition):
    """Per-seed summaries in fetch order (not sorted by seed id).

    Bootstrap CIs are order-sensitive at fixed sample size (the same seed
    resample draws different index->value pairings for a different input
    order), so this must match the order core.fetch wrote the tidy CSV in to
    reproduce the published CI bounds exactly.
    """
    d = df[df["condition"] == condition]
    return [stats.seed_summary(g, METRIC) for _, g in d.groupby("seed", sort=False)]


def compute():
    df = pd.read_csv(DATA)
    vals = {key: per_seed(df, key) for key in ROW_ORDER}
    nominal = vals["nominal"]

    family_keys = [key for key in ROW_ORDER if key != "nominal"]
    tests = {}
    for key in family_keys:
        tests[key] = stats.exact_mw(vals[key], nominal)
    adj = stats.holm([tests[key][1] for key in family_keys])
    p_holm = dict(zip(family_keys, adj))

    return vals, tests, p_holm


def compute_rows(vals, tests, p_holm):
    """Per-row values shared by both renderers below."""
    rows = []
    for key in ROW_ORDER:
        label, _, _, r_max, section = CONDITIONS[key]
        x = vals[key]
        mean, std = np.mean(x), np.std(x, ddof=1)
        lo, hi = stats.boot_ci(x)
        row = dict(key=key, label=label, r_max=r_max, section=section,
                  mean=mean, std=std, iqm=stats.iqm(x), lo=lo, hi=hi,
                  nominal=(key == "nominal"))
        if not row["nominal"]:
            u, p, r = tests[key]
            a = p_holm[key]
            row.update(u=u, p=p, p_holm=a, r=r, sig=a < ALPHA)
        rows.append(row)
    return rows


def render_tex(rows):
    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Condition} & $\boldsymbol{R_{\max}}$ & \textbf{Goal rate} & "
        r"\textbf{IQM [95\% CI]} & $\boldsymbol{U}$ & $\boldsymbol{p}$ & "
        r"$\boldsymbol{p_{\text{Holm}}}$ & $\boldsymbol{r}$ \\",
        r"\midrule",
    ]
    for row in rows:
        if row["section"]:
            lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{row['section']}}}}} \\\\")
        if row["nominal"]:
            line = (f"{row['label']} & ${row['r_max']}$ & "
                   f"${row['mean']:.3f} \\pm {row['std']:.3f}$ & "
                   f"${row['iqm']:.3f}\\ [{row['lo']:.3f}, {row['hi']:.3f}]$ & "
                   r"--- & --- & --- & --- \\")
        else:
            mark = r"^{\ast}" if row["sig"] else r"~\text{n.s.}"
            line = (f"{row['label']} & ${row['r_max']}$ & "
                   f"${row['mean']:.3f} \\pm {row['std']:.3f}$ & "
                   f"${row['iqm']:.3f}\\ [{row['lo']:.3f}, {row['hi']:.3f}]$ & "
                   f"${row['u']:.0f}$ & ${fmt_p(row['p'])}$ & "
                   f"${fmt_p(row['p_holm'])}{mark}$ & ${row['r']:+.2f}$ \\\\")
        lines.append(line)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_html(rows):
    headers = ["Condition", "R_max", "Goal rate", "IQM [95% CI]", "U", "p", "p_Holm", "r",
              "Section"]
    body = []
    for row in rows:
        cells = [detex(row["label"]), row["r_max"],
                f"{row['mean']:.3f} ± {row['std']:.3f}",
                f"{row['iqm']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]"]
        if row["nominal"]:
            cells += ["—", "—", "—", "—"]
        else:
            sig = "*" if row["sig"] else "n.s."
            cells += [f"{row['u']:.0f}", fmt_p_html(row["p"]),
                     f"{fmt_p_html(row['p_holm'])} {sig}", f"{row['r']:+.2f}"]
        cells.append(detex(row["section"]) if row["section"] else "")
        body.append(cells)
    return html_table(headers, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    ap.add_argument("--format", choices=["latex", "html", "both"], default="latex",
                     help="output format(s) to write (default: %(default)s)")
    args = ap.parse_args()

    vals, tests, p_holm = compute()
    rows = compute_rows(vals, tests, p_holm)

    if args.format in ("latex", "both"):
        tex = render_tex(rows)
        out = args.out_path / FRAGMENT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex)
        print(tex)
        print(f"wrote {out}", file=sys.stderr)

    if args.format in ("html", "both"):
        html = render_html(rows)
        out = args.out_path / HTML_FRAGMENT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(html)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
