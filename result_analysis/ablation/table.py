"""Generator for Table~\\ref{tab:ablation/normalizer} (deficiency-transform
ablation on Chicken--Banana, sections/ablation.tex, app:normalizer).

Reads only the committed tidy CSV under data/; never touches the network.
Per-seed summary is the mean of the final 10% of training
(core.stats.seed_summary); success counts seeds whose summary exceeds 160
(of a max 200). p is an exact two-sided Mann-Whitney test against the
Exponential (nominal) row -- no Holm correction, matching the manuscript
(each row is its own comparison against nominal, not a family).

This sweep (unlike tau/rb/epsilon, see arms.py) was run cleanly: verified
against wandb, its logged configs show nominal tau/rb/epsilon-decay
throughout.

Emits the tabular environment only (no \\begin{table}, caption, or label --
those stay authored prose in the manuscript). --format html emits the same
rows as a semantic HTML <table> fragment for the Beyond-PDF submission
(issue #42) -- generated from the same `compute()` rows as the LaTeX, never
hand-ported, so the two cannot silently diverge.

Usage: python table.py [--out-path PATH] [--format {latex,html,both}]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from arms import NORMALIZER_TABLE_ROWS

from core import stats
from core.html import detex, table as html_table

DATA = Path(__file__).parent / "data" / "normalizer_table_total.csv"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
FRAGMENT = Path("tables/ablation/normalizer.tex")
HTML_FRAGMENT = FRAGMENT.with_suffix(".html")
METRIC = "ep_info/total"
SUCCESS_THRESHOLD = 160
REFERENCE_LABEL = NORMALIZER_TABLE_ROWS[0][0]


def per_seed(df, label):
    d = df[df["arm"] == label]
    return [stats.seed_summary(g, METRIC) for _, g in d.groupby("seed", sort=False)]


def compute():
    """Per-row values shared by both renderers below."""
    df = pd.read_csv(DATA)
    vals = {label: per_seed(df, label) for label, _ in NORMALIZER_TABLE_ROWS}
    reference = vals[REFERENCE_LABEL]

    rows = []
    for label, _ in NORMALIZER_TABLE_ROWS:
        x = vals[label]
        mean, std = np.mean(x), np.std(x, ddof=1)
        lo, hi = stats.boot_ci(x)
        success = sum(v > SUCCESS_THRESHOLD for v in x)
        p = None if label == REFERENCE_LABEL else stats.exact_mw(x, reference)[1]
        rows.append(dict(label=label, mean=mean, std=std, iqm=stats.iqm(x),
                         lo=lo, hi=hi, success=success, total=len(x), p=p))
    return rows


def render_tex(rows):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Transform} & \textbf{Reward} & \textbf{IQM [95\% CI]} & "
             r"\textbf{Success} & $\boldsymbol{p}$ \\", r"\midrule"]
    for r in rows:
        p_str = "---" if r["p"] is None else f"{r['p']:.3g}"
        lines.append(f"{r['label']} & ${r['mean']:.2f} \\pm {r['std']:.2f}$ & "
                     f"${r['iqm']:.2f}\\ [{r['lo']:.2f}, {r['hi']:.2f}]$ & "
                     f"${r['success']}/{r['total']}$ & {p_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_html(rows):
    headers = ["Transform", "Reward", "IQM [95% CI]", "Success", "p"]
    body = []
    for r in rows:
        p_str = "—" if r["p"] is None else f"{r['p']:.3g}"
        body.append([
            detex(r["label"]),
            f"{r['mean']:.2f} ± {r['std']:.2f}",
            f"{r['iqm']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]",
            f"{r['success']}/{r['total']}",
            p_str,
        ])
    return html_table(headers, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    ap.add_argument("--format", choices=["latex", "html", "both"], default="latex",
                     help="output format(s) to write (default: %(default)s)")
    args = ap.parse_args()

    rows = compute()

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
