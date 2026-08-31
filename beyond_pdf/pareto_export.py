"""Beyond PDF export: builds a single self-contained HTML file -- the
Pareto explorer Embed (issue #50) -- from the morl scope's committed Tidy
CSVs. See issue #39's "Pareto explorer" spec: front points where hovering
reveals the weighting that produced each one. A static scatter shows where
the fronts are but hides the mapping a reader most wants -- which weighting
produced which trade-off.

Reads only the committed CSVs under result_analysis/morl/data/; never
touches the network. The fronts are the morl scope's own per-seed
candidate fronts (sources.per_seed), each method's points colored with
core.style.METHOD_COLORS -- the same computation and the same colors as
figure.py's static scatters this Embed replaces.

Which points carry a weight vector? Only DyLam and DynMORL log one
(fetch_data.py's docstring records why GPI-LS/PGMORL have none under any
wandb key; the untracked local CSVs that once plotted some are exactly the
un-owned artifact this pipeline exists to eliminate). Those weight
trajectories are separate history samples of the same runs, so a front
point's weight is recovered by time-fraction: both fetches subsample a
run's history with wandb's uniform-spaced history(samples=N), so history
row i of N sits at time fraction i/(N-1), and the weight at that fraction
is linearly interpolated between the two bracketing weight rows.
Linear interpolation is safe here because DyLam's lambda trajectory is the
EMA-smoothed path whose per-step movement Proposition 1(3) bounds, but it
is still an interpolation between logged rows, not an exact logged value;
the embed's tooltip says so (weights carry a note to that effect), and
DynMORL's piecewise-constant schedule is interpolated the same way --
between two equal rows that returns the constant itself, so only its
schedule jumps are approximated, by the same construction.

This mirrors figure.py's own honestly-documented gap (the weight panels
plot the weight trajectories, not per-point weights); the embed goes one
step further than the static figure could and labels the extra step.

The fronts shown are the manuscript's: per-seed fronts pooled exactly as
figure.py's merged_front does for the static scatters this Embed replaces
( DynMORL/Minecart decimated uniformly -- see --max-points-per-method --
to stay under the size cap; the front's extent and shape are preserved,
its density is not, and the capped counts are stated on the page's prose).

Usage:
  python beyond_pdf/pareto_export.py --out PATH [--max-points-per-method 2500]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "result_analysis"))
sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "morl"))

import pandas as pd  # noqa: E402

from core import style  # noqa: E402
from sources import DATA as MORL_DATA, ENVS  # noqa: E402

# figure.py owns the front computation this embed mirrors.
from figure import per_seed  # noqa: E402

TEMPLATE = Path(__file__).parent / "pareto_template.html"
DATA_TOKEN = "/*__BEYOND_PDF_DATA__*/null"
SIZE_CAP_BYTES = 5 * 1024 * 1024

WEIGHT_FILES = {
    ("MINECART", "DyLam"): "minecart_dylam_weights.csv",
    ("MINECART", "DynMORL"): "minecart_dynmorl_weights.csv",
    ("HALFCHEETAH", "DyLam"): "halfcheetah_dylam_weights.csv",
}

# The per-component R_max dashed lines the static HalfCheetah figure draws
# (figure.py's HALFCHEETAH_R_MAX), carried over so the embed shows the same
# reference lines. Minecart's static figure draws none.
REFS = [
    {"env": "HALFCHEETAH", "axis": 0, "value": 800.0},
    {"env": "HALFCHEETAH", "axis": 1, "value": 800.0},
]

WEIGHT_NOTE = (
    "weights recovered by interpolating the logged λ trajectory "
    "at the point's time fraction"
)
WEIGHT_UNAVAILABLE_NOTE = "λ weights not available: this method did not log a per-policy weight trajectory"


def _weight_series(env, label, morl_data):
    """seed -> (M, n_obj) array of the logged weight trajectory, in log order."""
    df = pd.read_csv(Path(morl_data) / WEIGHT_FILES[(env, label)])
    out = {}
    obj_cols = [c for c in df.columns if c.startswith("obj")]
    for seed, g in df.groupby("seed", sort=False):
        out[seed] = g[obj_cols].to_numpy(dtype=float)
    return out


def _interpolate_weights(weights, n_history):
    """Per-point weight vectors for one seed's front, by time fraction.

    `weights` is the seed's (M, n_obj) logged weight trajectory; the front's
    points sit at history rows 0..n_history-1 of the same run (front points
    keep their row order under the Pareto filter -- filter_dominated
    returns a boolean mask over the input rows), so point j's time fraction
    is j/(n_history-1) and its weight interpolates the trajectory there.
    """
    m = len(weights)
    fr = np.linspace(0.0, 1.0, n_history)
    grid = np.linspace(0.0, 1.0, m)
    out = np.empty((n_history, weights.shape[1]))
    for k in range(weights.shape[1]):
        out[:, k] = np.interp(fr, grid, weights[:, k])
    return out


def _method_points(env, label, max_points, morl_data):
    """(points, weights-or-None) for one (env, method): the pooled front,
    split per seed so weight interpolation runs within a seed's own history,
    then concatenated in seed order."""
    per_seed_weights = None
    if (env, label) in WEIGHT_FILES:
        per_seed_weights = _weight_series(env, label, morl_data)

    chunks, weight_chunks = [], []
    for seed, g, front in per_seed(env, _source(env, label)):
        chunks.append(front)
        if per_seed_weights is not None and seed in per_seed_weights:
            weight_chunks.append(
                _interpolate_weights(per_seed_weights[seed], len(front))
            )
        elif per_seed_weights is not None:
            weight_chunks.append(None)

    points = np.concatenate(chunks, axis=0)
    # Uniform decimation keeps the front's extent; the seed-interleaved
    # order means a plain stride keeps every seed represented.
    if max_points and len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        points = points[keep]
        if weight_chunks and all(w is not None for w in weight_chunks):
            weights = np.concatenate(weight_chunks, axis=0)
            weights = weights[keep]
        else:
            weights = None
    elif weight_chunks and all(w is not None for w in weight_chunks):
        weights = np.concatenate(weight_chunks, axis=0)
    else:
        weights = None
    return points, weights


_SOURCE_CACHE = {}


def _source(env, label):
    if not _SOURCE_CACHE:
        for e, sources in ENVS.items():
            for s in sources:
                _SOURCE_CACHE[(e, s.label)] = s
    return _SOURCE_CACHE[(env, label)]


def build(out_path, max_points=2500, data_root=None):
    # --data-root points at a directory holding morl/data (default: the
    # repository's result_analysis/). sources.per_seed reads its own module
    # constant, so a synthetic data root additionally requires patching that
    # constant -- the offline Embed test does exactly that (see
    # tests/test_beyond_pdf_export.py); the CLI flag alone covers the real
    # data living anywhere other than the repo layout.
    morl_data = Path(data_root) / "morl" / "data" if data_root else MORL_DATA
    if data_root:
        import sources as morl_sources

        morl_sources.DATA = Path(data_root) / "morl" / "data"
    envs = {}
    meta_panels = {}
    for env in ("HALFCHEETAH", "MINECART"):
        methods = []
        points_by_method = {}
        weights_by_method = {}
        for source in ENVS[env]:
            points, weights = _method_points(env, source.label, max_points, morl_data)
            points_by_method[source.label] = points
            weights_by_method[source.label] = weights
            methods.append(
                {
                    "label": source.label,
                    "color": _css_color(style.METHOD_COLORS[source.label]),
                }
            )

        n_axes = points_by_method[ENVS[env][0].label].shape[1]
        axis_labels = ENV_AXIS_LABELS[env][:n_axes]
        env_points = {
            label: {
                "obj": [[round(float(v), 4) for v in p] for p in pts],
                "i": list(range(len(pts))),
                "r": 2.2 if weights_by_method[label] is None else 1.8,
            }
            for label, pts in points_by_method.items()
        }
        env_weights = {
            label: (
                {
                    "available": True,
                    "values": [[round(float(v), 3) for v in w] for w in weights],
                    "note": WEIGHT_NOTE,
                }
                if weights is not None
                else {"available": False, "note": WEIGHT_UNAVAILABLE_NOTE}
            )
            for label, weights in weights_by_method.items()
        }
        meta_panels[env] = {
            "axisLabels": axis_labels,
            "methods": methods,
            "points": env_points,
            "weights": env_weights,
            "refs": [r for r in REFS if r["env"] == env],
        }
        projections = []
        if n_axes == 3:
            projections = [
                {"key": "1v2", "label": "M1 vs M2", "axes": [0, 1]},
                {"key": "1v3", "label": "M1 vs Fuel", "axes": [0, 2]},
                {"key": "2v3", "label": "M2 vs Fuel", "axes": [1, 2]},
            ]
        else:
            projections = [{"key": "1v2", "label": "Run vs Control", "axes": [0, 1]}]
        envs[env] = projections

    data = {
        "selectors": [
            {
                "key": "env",
                "label": "Environment",
                "options": [
                    {"key": "HALFCHEETAH", "label": "MO-HalfCheetah"},
                    {"key": "MINECART", "label": "MO-Minecart"},
                ],
            },
        ],
        "groups": {"env": {"HALFCHEETAH": ["HALFCHEETAH"], "MINECART": ["MINECART"]}},
        "panels": {"HALFCHEETAH": ["HALFCHEETAH"], "MINECART": ["MINECART"]},
        "projections": envs,
        "meta": {"panels": meta_panels},
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if DATA_TOKEN not in template:
        sys.exit(f"template {TEMPLATE} is missing the data token {DATA_TOKEN!r}")
    html = template.replace(DATA_TOKEN, json.dumps(data, separators=(",", ":")))

    size = len(html.encode("utf-8"))
    if size > SIZE_CAP_BYTES:
        sys.exit(
            f"built artifact is {size} bytes, over the {SIZE_CAP_BYTES}-byte cap. "
            "Lower --max-points-per-method to reduce the point clouds."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    n_points = sum(
        len(p["obj"])
        for panel in meta_panels.values()
        for p in panel["points"].values()
    )
    print(
        f"wrote {out_path} ({size} bytes, {len(meta_panels)} environments, "
        f"{n_points} front points)",
        file=sys.stderr,
    )


def _css_color(color):
    named = {
        "tab:blue": "#1f77b4",
        "tab:orange": "#ff7f0e",
        "tab:green": "#2ca02c",
        "tab:purple": "#9467bd",
    }
    return named.get(color, color)


ENV_AXIS_LABELS = {
    "HALFCHEETAH": ["Run", "Control (shifted)"],
    "MINECART": ["First Minerium", "Second Minerium", "Fuel"],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="output HTML path")
    ap.add_argument("--max-points-per-method", type=int, default=2500)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="directory holding morl/data (default: the repository's result_analysis/)",
    )
    args = ap.parse_args()
    build(args.out, args.max_points_per_method, args.data_root)


if __name__ == "__main__":
    main()
