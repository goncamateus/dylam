"""Generator for Table~\\ref{tab:ablation/normalizer} (deficiency-transform
ablation on Chicken--Banana, sections/ablation.tex, app:normalizer).

Reads only the committed tidy CSV under data/; never touches the network.
Per-seed summary is the mean of the final 10% of training
(lib.stats.seed_summary); success counts seeds whose summary exceeds 160
(of a max 200). p is an exact two-sided Mann-Whitney test against the
Exponential (nominal) row -- no Holm correction, matching the manuscript
(each row is its own comparison against nominal, not a family).

This sweep (unlike tau/rb/epsilon, see arms.py) was run cleanly: verified
against wandb, its logged configs show nominal tau/rb/epsilon-decay
throughout.

Emits the tabular environment only (no \\begin{table}, caption, or label --
those stay authored prose in the manuscript).

Usage: python table.py [--out-path PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from arms import NORMALIZER_TABLE_ROWS
from lib import stats

DATA = Path(__file__).parent / "data" / "normalizer_table_total.csv"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
FRAGMENT = Path("tables/ablation/normalizer.tex")
METRIC = "ep_info/total"
SUCCESS_THRESHOLD = 160
REFERENCE_LABEL = NORMALIZER_TABLE_ROWS[0][0]


def per_seed(df, label):
    d = df[df["arm"] == label]
    return [stats.seed_summary(g, METRIC) for _, g in d.groupby("seed", sort=False)]


def render():
    df = pd.read_csv(DATA)
    vals = {label: per_seed(df, label) for label, _ in NORMALIZER_TABLE_ROWS}
    reference = vals[REFERENCE_LABEL]

    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Transform} & \textbf{Reward} & \textbf{IQM [95\% CI]} & "
             r"\textbf{Success} & $\boldsymbol{p}$ \\", r"\midrule"]
    for label, _ in NORMALIZER_TABLE_ROWS:
        x = vals[label]
        mean, std = np.mean(x), np.std(x, ddof=1)
        lo, hi = stats.boot_ci(x)
        success = sum(v > SUCCESS_THRESHOLD for v in x)
        p_str = "---" if label == REFERENCE_LABEL else f"{stats.exact_mw(x, reference)[1]:.3g}"
        lines.append(f"{label} & ${mean:.2f} \\pm {std:.2f}$ & "
                     f"${stats.iqm(x):.2f}\\ [{lo:.2f}, {hi:.2f}]$ & "
                     f"${success}/{len(x)}$ & {p_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    args = ap.parse_args()

    tex = render()
    out = args.out_path / FRAGMENT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(tex)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
