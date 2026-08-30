"""Generators for fig:res/all (sections/results/trad.tex) and
fig:res/chicken_banana/components (sections/results/trad/app.tex).

Reads only the committed tidy CSVs under data/; never touches the network.

fig:res/all summarizes each seed's smoothed curve by IQM + 95% bootstrap CI
across seeds (core.style.rolling_smooth + core.stats.bootstrap_curve) -- the
protocol that actually produced the currently published figure. Colors are
core.style.METHOD_COLORS, keyed by the paper method label, not by whichever
algorithm name a panel's legend happens to use for it (Q-Learning/SAC are
both "Base SO RL").

fig:res/chicken_banana/components is a plainer per-component view: each
seed's smoothed curve, averaged across seeds with no CI band, matching this
figure's own caption ("mean over 10 seeds"). Its data source (the aggregated
wandb-UI exports chicken_banana.ipynb read) predates seeds being
recoverable at all; this generator reads the same tidy per-seed data as
everything else instead.

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
from methods import CHICKENBANANA_METHODS, COMPONENT_METRICS

from core import stats, style

DATA = Path(__file__).parent / "data"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
GRID = 200

# fig:res/all panels. No in-plot legend (the manuscript's caption names the
# colors), so only the method -- which is also the style.METHOD_COLORS key --
# matters here, not what a panel's legend would call it (Q-Learning/SAC are
# both "Base SO RL").
MAIN_PANELS = [
    dict(file="chicken_banana.csv", metric="ep_info/total", xlabel="Episode",
         ylabel="Cumulative episode reward",
         out="images/results/tradicional/chicken_banana/reward-total.pdf",
         methods=["Base SO RL", "Q-Decomposition", "UDC", "DyLam"]),
    dict(file="halfcheetah_v4.csv", metric="ep_info/Final_position", xlabel="Environment step",
         ylabel="Final $x$-position (m)",
         out="images/results/tradicional/halfcheetah/HalfCheetah-v4.pdf",
         methods=["Base SO RL", "UDC", "DyLam"]),
    dict(file="vss_v0.csv", metric="ep_info/Goal", xlabel="Environment step",
         ylabel="Goal rate",
         out="images/results/tradicional/vss/VSS-v0.pdf",
         methods=["Base SO RL", "UDC", "DyLam", "Tuned-UDC"]),
]

# fig:res/chicken_banana/components panels, in the manuscript's own order.
COMPONENT_PANELS = [
    ("Banana", "chicken_banana_banana.csv", COMPONENT_METRICS["Banana"], 30,
     "images/results/tradicional/chicken_banana/reward-banana.pdf"),
    ("Chicken", "chicken_banana_chicken.csv", COMPONENT_METRICS["Chicken"], 70,
     "images/results/tradicional/chicken_banana/reward-chicken.pdf"),
    ("Gate", "chicken_banana_gate.csv", COMPONENT_METRICS["Gate"], 100,
     "images/results/tradicional/chicken_banana/reward-gate.pdf"),
    ("Total", "chicken_banana.csv", "ep_info/total", 200,
     "images/results/tradicional/chicken_banana/reward-total-app.pdf"),
]


def _per_seed_series(df, column, key, metric):
    d = df[df[column] == key]
    return [(g["_step"].to_numpy(dtype=float), g[metric].to_numpy(dtype=float))
            for _, g in d.groupby("seed", sort=False)]


def _grid_for(series_by_method):
    hi = min(step.max() for series in series_by_method.values() for step, _ in series)
    return np.linspace(0, hi, GRID)


def draw_main(out_root, fmt="pdf"):
    for panel in MAIN_PANELS:
        df = pd.read_csv(DATA / panel["file"])
        series_by_method = {method: _per_seed_series(df, "method", method, panel["metric"])
                            for method in panel["methods"]}
        grid = _grid_for(series_by_method)

        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        for method in panel["methods"]:
            mat = np.array([np.interp(grid, step, style.rolling_smooth(value))
                            for step, value in series_by_method[method]])
            centre, lo, hi = stats.bootstrap_curve(mat)
            color = style.METHOD_COLORS[method]
            ax.plot(grid, centre, color=color, linewidth=1.6)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.20, linewidth=0)
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()

        out = out_root / panel["out"]
        for written in style.savefig(fig, out, fmt):
            print(f"wrote {written}", file=sys.stderr)
        plt.close(fig)


def draw_components(out_root, fmt="pdf"):
    for label, file, metric, r_max, out_rel in COMPONENT_PANELS:
        df = pd.read_csv(DATA / file)
        series_by_method = {m: _per_seed_series(df, "method", m, metric)
                            for m in CHICKENBANANA_METHODS}
        grid = _grid_for(series_by_method)

        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for method in CHICKENBANANA_METHODS:
            curves = [np.interp(grid, step, style.rolling_smooth(value)) / r_max
                     for step, value in series_by_method[method]]
            mean = np.mean(curves, axis=0)
            ax.plot(grid, mean, color=style.METHOD_COLORS[method], linewidth=1.6, label=method)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"{label} reward / $R_{{\\max}}$")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7)
        fig.tight_layout()

        out = out_root / out_rel
        for written in style.savefig(fig, out, fmt):
            print(f"wrote {written}", file=sys.stderr)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    ap.add_argument("--format", choices=style.FORMATS, default="pdf",
                     help="output format: pdf (manuscript, default), svg "
                          "(Beyond PDF), or both (default: %(default)s)")
    args = ap.parse_args()
    draw_main(args.out_path, args.format)
    draw_components(args.out_path, args.format)


if __name__ == "__main__":
    main()
