"""Generator for Figure~\\ref{fig:res/robustness/curves} (VSS-v0 goal rate under
bound misspecification, sections/results/robust.tex).

Reads only the committed tidy CSV under data/; never touches the network.
Per-seed rolling-mean smoothing (lib.style.rolling_smooth) before cross-seed
IQM + bootstrap aggregation (lib.stats.bootstrap_curve) is the protocol that
actually produced the currently published figure -- the other implementation
that once existed for this plot (a notebook, since deleted) used per-seed
Gaussian smoothing with min/max bands over a different, narrower set of
conditions and was never wired to write into the paper repo.

Figures have no automated test seam; check the output by eye.

Usage: python figure.py [--out-path PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lib import stats, style

METRIC = "ep_info/Goal"
DATA = Path(__file__).parent / "data" / "vss_ep_info_goal.csv"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
IMAGE = Path("images/results/robustness_curves.pdf")
GRID = 200

# (arm key, label -- must match a key in lib.style.CONDITION_COLORS)
ARMS = [
    ("nominal", "Nominal"),
    ("move_m25", "Move $-25\\%$"),
    ("move_p50", "Move $+50\\%$"),
    ("ball_p25", "Ball $+25\\%$"),
    ("ball_m25", "Ball $-25\\%$"),
    ("compound_move_p50_ball_p25", "Move $+50\\%$, ball $+25\\%$"),
    ("compound_move_m25_ball_m50", "Move $-25\\%$, ball $-50\\%$"),
]


def per_seed_series(df, arm):
    """List of (step, value) arrays, one per seed, in fetch order."""
    d = df[df["arm"] == arm]
    return [(g["_step"].to_numpy(dtype=float), g[METRIC].to_numpy(dtype=float))
            for _, g in d.groupby("seed", sort=False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    series = {key: per_seed_series(df, key) for key, _ in ARMS}
    grid_hi = min(steps.max() for arm in series.values() for steps, _ in arm)
    grid = np.linspace(0, grid_hi, GRID)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for key, label in ARMS:
        mat = np.array([
            np.interp(grid, step, style.rolling_smooth(value))
            for step, value in series[key]
        ])
        centre, lo, hi = stats.bootstrap_curve(mat)
        color = style.CONDITION_COLORS[label]
        ax.plot(grid, centre, color=color, linewidth=1.6)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Goal rate")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out = args.out_path / IMAGE
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
