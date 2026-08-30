"""Generators for the three Pareto-front figures (sections/results/morl/app.tex,
app:pareto_extended): fig:hc_pareto_a / fig:hc_pareto_b (MO-HalfCheetah
discovered front and weight space) and fig:res/minecart-pareto-group
(MO-Minecart pairwise objective and weight-simplex projections).

Reads only the committed tidy CSVs under data/; never touches the network.
Each method's plotted front is its per-seed candidate sets pooled and
re-filtered to the non-dominated subset (lib.pareto), matching the
manuscript's own "candidates pooled, duplicates removed, dominated points
filtered" protocol (app:res_morl_metrics) -- the same computation
table.py does per-seed, just merged across seeds here for the picture.

The weight-space panels include DyLam/DynMORL, which log an actual weight
trajectory, but not GPI-LS/PGMORL: neither exposes a per-policy weight
vector under any wandb key this migration could find (see
fetch_data.py's docstring), so the two points/curves the superseded
notebooks plotted for them came from untracked local CSVs of unknown
provenance. Omitted rather than reproduced from data with no traceable
source.

The HalfCheetah PGMORL/GPI-LS fronts render visibly sparser here than
pareto.ipynb's published figure. That notebook plotted
HalfCheetah/objs.txt and merged_fronts/{gpi_ls,dylam}_halfcheetah.csv --
static files with no generating code left in this repo -- directly,
with no re-filtering call in that cell, so their apparent density can't
be distinguished from "already someone's precomputed front" vs. "raw
unfiltered candidates". This migration instead pools and re-filters the
actual per-seed eval/front tables fetched from wandb (fetch_data.py),
the same data that reproduces the manuscript's HV/cardinality numbers
almost exactly in table.py's tests -- so the sparser front here is the
one traceable to source, not a bug to chase into matching an
unreproducible artifact.

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
from sources import ENVS, per_seed

from core import pareto, style

DATA = Path(__file__).parent / "data"
DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
HALFCHEETAH_R_MAX = (800.0, 800.0)  # R_max=(800,-200) on ctrl's 1000-shifted axis
MARKERS = {"PGMORL": "o", "GPI-LS": "s", "DynMORL": "o", "DyLam": "^"}


def merged_front(env, source):
    """Non-dominated front of every seed's candidates, pooled.

    Pools sources.per_seed's already-per-seed-filtered fronts (bounded,
    same cost table.py already pays per seed) and re-filters once more.
    Pareto-filtering is idempotent under union -- NDS(A u B) ==
    NDS(NDS(A) u NDS(B)) -- so this is the identical front a single pooled
    filter would give, just without ever materializing an O(n^2) dominance
    matrix over the full ~10^5-point pooled cloud.
    """
    seed_fronts = [front for _, _, front in per_seed(env, source)]
    return pareto.filter_dominated(np.concatenate(seed_fronts, axis=0))


def draw_halfcheetah_pareto(out_root):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for source in ENVS["HALFCHEETAH"]:
        front = merged_front("HALFCHEETAH", source)
        ax.scatter(front[:, 0], front[:, 1], color=style.METHOD_COLORS[source.label],
                  marker=MARKERS[source.label], alpha=0.6, edgecolors="w", label=source.label)
    ax.axvline(HALFCHEETAH_R_MAX[0], color="black", linestyle="--", linewidth=0.8)
    ax.axhline(HALFCHEETAH_R_MAX[1], color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_root / "images/results/pareto/halfcheetah_pareto.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def draw_halfcheetah_weights(out_root):
    df = pd.read_csv(DATA / "halfcheetah_dylam_weights.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.scatter(df["obj1"], df["obj2"], color=style.METHOD_COLORS["DyLam"],
              marker=MARKERS["DyLam"], s=10, alpha=0.6, label="DyLam")
    ax.set_xlabel("Run weight")
    ax.set_ylabel("Control weight")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_root / "images/results/pareto/halfcheetah_weights_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


MINECART_OBJ_LABELS = ["First Minerium", "Second Minerium", "Fuel"]
MINECART_WEIGHT_LABELS = ["$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def draw_minecart(out_root):
    fronts = {s.label: merged_front("MINECART", s) for s in ENVS["MINECART"]}
    weights = {
        "DyLam": pd.read_csv(DATA / "minecart_dylam_weights.csv"),
        "DynMORL": pd.read_csv(DATA / "minecart_dynmorl_weights.csv"),
    }

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    for col, (i, j) in enumerate(PAIRS):
        ax = axes[0, col]
        for label, front in fronts.items():
            ax.scatter(front[:, i], front[:, j], color=style.METHOD_COLORS[label],
                      marker=MARKERS[label], alpha=0.6, edgecolors="w", label=label, s=80)
        ax.set_xlabel(MINECART_OBJ_LABELS[i])
        ax.set_ylabel(MINECART_OBJ_LABELS[j])
        ax.set_title(f"{i + 1} vs {j + 1}")

        ax = axes[1, col]
        for label, df in weights.items():
            ax.scatter(df[f"obj{i + 1}"], df[f"obj{j + 1}"], color=style.METHOD_COLORS[label],
                      marker=MARKERS[label], alpha=0.6, edgecolors="w", label=label, s=80)
        ax.set_xlabel(MINECART_WEIGHT_LABELS[i])
        ax.set_ylabel(MINECART_WEIGHT_LABELS[j])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_root / "images/results/pareto/minecart_pareto_weights.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    args = ap.parse_args()
    draw_halfcheetah_pareto(args.out_path)
    draw_halfcheetah_weights(args.out_path)
    draw_minecart(args.out_path)


if __name__ == "__main__":
    main()
