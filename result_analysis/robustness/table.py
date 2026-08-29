"""Generator for Table~\\ref{tab:res/robustness/summary} (VSS-v0 bound misspecification).

Owns: tab:res/robustness/summary in sections/results/trad/app.tex, and by
extension the goal rates / U / p / p_Holm / r figures quoted in prose in
sections/results/robust.tex (sec:res_robustness) -- both read off this same
table.

Reads only the committed tidy CSV under data/; never touches the network.
Per-seed summary is the mean of the final 10% of training (lib.stats.final).
Comparisons are exact two-sided Mann-Whitney U against nominal, Holm-Bonferroni
corrected within the six-comparison RQ3 family, with rank-biserial effect size.

Emits the tabular environment only (no \\begin{table}, caption, or label --
those stay authored prose in the manuscript). p and p_Holm are both rendered
scientific below 1e-3 and fixed-point otherwise; this is a normalization of
the previously hand-typed table, which mixed conventions between the two
columns -- the numbers are unchanged, only their typesetting rule is now one
rule instead of two.

Usage: python table.py [--out-path PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from lib import stats

METRIC = "ep_info/Goal"
DATA = Path(__file__).parent / "data" / "vss_ep_info_goal.csv"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
FRAGMENT = Path("tables/results/robustness_summary.tex")
ALPHA = 0.05

# (arm key, label, R_max string, section header or None to continue the section)
ARMS = [
    ("nominal", "Nominal", "(150, 40, -100)", None),
    ("move_m25", "Move $-25\\%$", "(112.5, 40, -100)", "Move ceiling"),
    ("move_p50", "Move $+50\\%$", "(225, 40, -100)", None),
    ("ball_m25", "Ball $-25\\%$", "(150, 30, -100)", "Ball-to-goal ceiling"),
    ("ball_p25", "Ball $+25\\%$", "(150, 50, -100)", None),
    ("compound_move_p50_ball_p25", "Move $+50\\%$, ball $+25\\%$", "(225, 50, -100)", "Compound"),
    ("compound_move_m25_ball_m50", "Move $-25\\%$, ball $-50\\%$", "(112.5, 20, -100)", None),
]


def fmt_p(p):
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / 10 ** exp
        return f"{mant:.1f} \\times 10^{{{exp}}}"
    return f"{p:.3f}"


def per_seed(df, arm):
    """Per-seed summaries in fetch order (not sorted by seed id).

    Bootstrap CIs are order-sensitive at fixed sample size (the same seed
    resample draws different index->value pairings for a different input
    order), so this must match the order lib.fetch wrote the tidy CSV in to
    reproduce the published CI bounds exactly.
    """
    d = df[df["arm"] == arm]
    return [stats.final(g, METRIC) for _, g in d.groupby("seed", sort=False)]


def compute():
    df = pd.read_csv(DATA)
    vals = {key: per_seed(df, key) for key, *_ in ARMS}
    nominal = vals["nominal"]

    family_keys = [key for key, *_ in ARMS if key != "nominal"]
    tests = {}
    for key in family_keys:
        tests[key] = stats.exact_mw(vals[key], nominal)
    adj = stats.holm([tests[key][1] for key in family_keys])
    p_holm = dict(zip(family_keys, adj))

    return vals, tests, p_holm


def render(vals, tests, p_holm):
    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Condition} & $\boldsymbol{R_{\max}}$ & \textbf{Goal rate} & "
        r"\textbf{IQM [95\% CI]} & $\boldsymbol{U}$ & $\boldsymbol{p}$ & "
        r"$\boldsymbol{p_{\text{Holm}}}$ & $\boldsymbol{r}$ \\",
        r"\midrule",
    ]
    for key, label, r_max, section in ARMS:
        if section:
            lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{section}}}}} \\\\")
        x = vals[key]
        mean, std = np.mean(x), np.std(x, ddof=1)
        lo, hi = stats.boot_ci(x)
        iqm = stats.iqm(x)
        if key == "nominal":
            row = (f"{label} & ${r_max}$ & ${mean:.3f} \\pm {std:.3f}$ & "
                   f"${iqm:.3f}\\ [{lo:.3f}, {hi:.3f}]$ & --- & --- & --- & --- \\\\")
        else:
            u, p, r = tests[key]
            a = p_holm[key]
            mark = r"^{\ast}" if a < ALPHA else r"~\text{n.s.}"
            row = (f"{label} & ${r_max}$ & ${mean:.3f} \\pm {std:.3f}$ & "
                   f"${iqm:.3f}\\ [{lo:.3f}, {hi:.3f}]$ & ${u:.0f}$ & "
                   f"${fmt_p(p)}$ & ${fmt_p(a)}{mark}$ & ${r:+.2f}$ \\\\")
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    args = ap.parse_args()

    vals, tests, p_holm = compute()
    tex = render(vals, tests, p_holm)

    out = args.out_path / FRAGMENT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(tex)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
