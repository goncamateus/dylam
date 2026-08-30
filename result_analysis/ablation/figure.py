"""Generators for the four Chicken--Banana ablation figures
(fig:ablation/{tau,rb,normalizer,epsilon}, sections/ablation.tex).

Reads only the committed tidy CSVs under data/; never touches the network.
Each panel is IQM + 95% bootstrap CI across seeds (lib.style.rolling_smooth
+ lib.stats.bootstrap_curve), the same protocol as trad/robustness --
replacing the per-sweep cumsum-smoothed mean/min-max bands of the two
notebooks this supersedes (ablation.ipynb, dead entirely; the combined
figures in ablation_combined.ipynb, whose two files disagreed on the
colour assigned to the same swept value, exactly the drift issue #37's
problem statement names).

fig:ablation/rb's caption promises a third row -- one seed's raw,
unaggregated lambda trace ("isolated run"), kept here since it's what the
caption says the figure shows; the other three figures' captions don't
commit to that extra row, so they get a plain two-row layout (reward,
aggregated lambda) instead of ablation_combined.ipynb's per-figure mix of
aggregated/isolated rows for reasons the notebook doesn't explain.

CONFOUNDED (see arms.py): the rb and epsilon sweeps are not clean
single-hyperparameter ablations due to a training-code bug in
scripts/ablation.py; reproduced faithfully from what wandb contains.

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
from arms import COMPONENTS, EPSILON_ARMS, NORMALIZER_ARMS, RB_ARMS, TAU_ARMS

from lib import stats, style

DATA = Path(__file__).parent / "data"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
GRID = 200
NOMINAL_COLOR = "dimgray"
ARM_COLORS = ["tab:blue", "tab:orange", "tab:green"]

# (sweep name, arms, has_isolated_row, out path)
FIGURES = [
    ("tau", TAU_ARMS, False, "images/ablation/tau/combined_results.pdf"),
    ("rb", RB_ARMS, True, "images/ablation/rb/combined_results.pdf"),
    ("normalizer", NORMALIZER_ARMS, False, "images/ablation/normalizer/combined_results.pdf"),
    ("epsilon", EPSILON_ARMS, False, "images/ablation/epsilon/combined_results.pdf"),
]


def _read(path):
    """arm labels like "0.5"/"50" must stay strings, not become floats/ints."""
    return pd.read_csv(path, dtype={"arm": str})


def _series(df, key):
    d = df[df["arm"] == key]
    metric = [c for c in d.columns if c not in ("_step", "seed", "arm")][0]
    return [(g["_step"].to_numpy(dtype=float), g[metric].to_numpy(dtype=float))
            for _, g in d.groupby("seed", sort=False)]


def _band_row(ax_row, sweep, comp, kind, arms):
    """One row of panels: nominal + each arm, IQM + bootstrap CI."""
    nominal = _read(DATA / f"nominal_{comp.lower()}_{kind}.csv")
    swept = _read(DATA / f"{sweep}_{comp.lower()}_{kind}.csv")
    series = {"nominal": _series(nominal, "nominal")}
    series.update({a.label: _series(swept, a.label) for a in arms})

    grid_hi = min(step.max() for s in series.values() for step, _ in s)
    grid = np.linspace(0, grid_hi, GRID)
    for label, color in [("nominal", NOMINAL_COLOR)] + list(zip((a.label for a in arms),
                                                                ARM_COLORS)):
        mat = np.array([np.interp(grid, step, style.rolling_smooth(value))
                        for step, value in series[label]])
        centre, lo, hi = stats.bootstrap_curve(mat)
        ax_row.plot(grid, centre, color=color, linewidth=1.4, label=label)
        ax_row.fill_between(grid, lo, hi, color=color, alpha=0.20, linewidth=0)
    return grid


def _isolated_row(ax_row, sweep, comp, arms):
    """One representative (first-fetched) seed's raw lambda trace per arm."""
    nominal = _read(DATA / f"nominal_{comp.lower()}_lambda.csv")
    swept = _read(DATA / f"{sweep}_{comp.lower()}_lambda.csv")
    step, value = _series(nominal, "nominal")[0]
    ax_row.plot(step, value, color=NOMINAL_COLOR, linewidth=1.2, label="nominal")
    for arm, color in zip(arms, ARM_COLORS):
        step, value = _series(swept, arm.label)[0]
        ax_row.plot(step, value, color=color, linewidth=1.2, label=arm.label)
    ax_row.set_ylim(0, 1)


def draw(out_root):
    for sweep, arms, isolated, out_rel in FIGURES:
        nrows = 3 if isolated else 2
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 4 * nrows), sharex=True)
        for col, comp in enumerate(COMPONENTS):
            _band_row(axes[0, col], sweep, comp, "reward", arms)
            axes[0, col].set_title(comp)
            if col == 0:
                axes[0, col].set_ylabel("Cumulative episode reward")
            _band_row(axes[1, col], sweep, comp, "lambda", arms)
            axes[1, col].set_ylim(0, 1)
            if col == 0:
                axes[1, col].set_ylabel("Lambda value")
            if isolated:
                _isolated_row(axes[2, col], sweep, comp, arms)
                if col == 0:
                    axes[2, col].set_ylabel("Lambda value (single run)")
                axes[2, col].set_xlabel("Episode")
            else:
                axes[1, col].set_xlabel("Episode")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()

        out = out_root / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    args = ap.parse_args()
    draw(args.out_path)


if __name__ == "__main__":
    main()
