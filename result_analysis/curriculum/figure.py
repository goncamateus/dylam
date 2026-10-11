"""Generators for fig:curr/weights and fig:curr/components
(sections/results/trad/curriculum.tex, sections/results/trad/app.tex,
app:res_curriculum): DyLam's adaptive lambda-weight trajectory and the
per-component returns driving it, for ChickenBanana-v0, HalfCheetah-v4,
and VSS-v0.

Reads only the committed tidy CSVs under data/; never touches the
network. Colors are positional (core.style.COMPONENT_PALETTE, index i for
the i-th component in sources.ENVS[env].components) -- matching every
component's order in that env's own Dylam comp_names, and reused as-is
for HalfCheetah/VSS, whose components figures have no surviving
generating code (sources.py). Confirmed by rendering the currently
published PDFs: HalfCheetah's dashed reference lines sit at 1.0 (Run)
and -0.25 (Control = -200/800), which only matches normalizing each
component by max(|r_max|, |r_min|) -- the same convention utils.py's now
otherwise-dead plot_rewards used for Chicken-Banana, so all three envs
share it here for consistency, not just Chicken-Banana.

fig:curr/weights plots, per component, the interquartile mean over seeds
with a 95% bootstrap band (core.stats.bootstrap_curve), with a legend in
every panel; its colors and the sec:res_trad_rq2 prose that reads it must
agree (Velocity is blue in HalfCheetah; Ball's weight peaks before Move's
in VSS).

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
from sources import ENVS

from core import stats, style

DATA = Path(__file__).parent / "data"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
GRID = 200
OUT_DIR = {
    "CHICKENBANANA": "images/results/tradicional/chicken_banana",
    "HALFCHEETAH": "images/results/tradicional/halfcheetah",
    "VSS": "images/results/tradicional/vss",
}


def _grid(df, metric):
    """Per-seed smoothed series interpolated onto a shared grid: (grid, n_seeds x GRID)."""
    series = [(g["_step"].to_numpy(dtype=float), g[metric].to_numpy(dtype=float))
             for _, g in df.groupby("seed", sort=False)]
    hi = min(step.max() for step, _ in series)
    grid = np.linspace(0, hi, GRID)
    return grid, np.array([np.interp(grid, step, style.rolling_smooth(value)) for step, value in series])


def _grid_mean(df, metric):
    grid, mat = _grid(df, metric)
    return grid, mat.mean(axis=0)


def draw_weights(out_root, fmt="pdf"):
    for env, spec in ENVS.items():
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for i, comp in enumerate(spec.components):
            metric = f"lambdas/{comp.name}"
            df = pd.read_csv(DATA / f"{env.lower()}_{comp.name.lower()}_lambda.csv")
            grid, mat = _grid(df, metric)
            centre, lo, hi = stats.bootstrap_curve(mat)
            color = style.COMPONENT_PALETTE[i]
            ax.plot(grid, centre, color=color, linewidth=1.6, label=comp.label)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel(r"$\lambda$ weights")
        ax.set_ylim(0, 1)
        ax.set_title(spec.gym_label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = out_root / OUT_DIR[env] / f"{spec.gym_label}-weights.pdf"
        for written in style.savefig(fig, out, fmt):
            print(f"wrote {written}", file=sys.stderr)
        plt.close(fig)


def draw_components(out_root, fmt="pdf"):
    for env, spec in ENVS.items():
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for i, comp in enumerate(spec.components):
            df = pd.read_csv(DATA / f"{env.lower()}_{comp.name.lower()}_reward.csv")
            grid, mean = _grid_mean(df, comp.ep_metric)
            abs_max = max(abs(comp.r_max), abs(comp.r_min))
            color = style.COMPONENT_PALETTE[i]
            ax.plot(grid, mean / abs_max, color=color, linewidth=1.6, label=comp.label)
            ax.axhline(comp.r_max / abs_max, color=color, linestyle="--", linewidth=0.8)
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel("Cumulative episode reward")
        ax.set_title(spec.gym_label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = out_root / OUT_DIR[env] / f"{spec.gym_label}-components.pdf"
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
    draw_weights(args.out_path, args.format)
    draw_components(args.out_path, args.format)


if __name__ == "__main__":
    main()
