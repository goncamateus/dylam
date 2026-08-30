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

Two discrepancies found and NOT corrected here (out of scope -- captions
and the narrative they describe are authored prose):

- curriculum.tex's fig:curr/weights caption says HalfCheetah-v4 is
  "control cost (blue) and velocity (orange)", but fetching lambdas/Run
  vs lambdas/Control from wandb and comparing curve shapes against the
  published image shows the opposite of what's actually plotted -- Run
  is blue, Control is orange, the same comp_names-order-to-palette-index
  mapping every other env uses.
- curriculum.tex's VSS-v0 paragraph describes Move's weight
  concentrating first and transferring to Ball as Move's own return
  saturates (~10^5 steps). fig:curr/vss-comp (draw_components) matches
  that story cleanly: Move's return does saturate first. But
  fig:curr/vss-lam (draw_weights) doesn't show a clean Move-then-Ball
  handoff -- Ball's weight rises and peaks earlier than Move's does,
  the two converging to near parity by the end rather than Move ceding
  share monotonically. Checked for a fetch/plotting bug first (the
  per-component tidy CSVs' own metric columns are correctly
  lambdas/Move vs lambdas/Ball, no mix-up); none found, so this reads as
  a real property of the fetched trajectory, not a code defect.

Both flagged here and in the commit message for the author to judge.

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

from core import style

DATA = Path(__file__).parent / "data"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
GRID = 200
OUT_DIR = {
    "CHICKENBANANA": "images/results/tradicional/chicken_banana",
    "HALFCHEETAH": "images/results/tradicional/halfcheetah",
    "VSS": "images/results/tradicional/vss",
}


def _grid_mean(df, metric):
    series = [(g["_step"].to_numpy(dtype=float), g[metric].to_numpy(dtype=float))
             for _, g in df.groupby("seed", sort=False)]
    hi = min(step.max() for step, _ in series)
    grid = np.linspace(0, hi, GRID)
    mat = np.array([np.interp(grid, step, style.rolling_smooth(value)) for step, value in series])
    return grid, mat.mean(axis=0)


def draw_weights(out_root, fmt="pdf"):
    for env, spec in ENVS.items():
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for i, comp in enumerate(spec.components):
            metric = f"lambdas/{comp.name}"
            df = pd.read_csv(DATA / f"{env.lower()}_{comp.name.lower()}_lambda.csv")
            grid, mean = _grid_mean(df, metric)
            ax.plot(grid, mean, color=style.COMPONENT_PALETTE[i], linewidth=1.6, label=comp.label)
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel(r"$\lambda$ weights")
        ax.set_ylim(0, 1)
        ax.set_title(spec.gym_label)
        ax.grid(alpha=0.3)
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
