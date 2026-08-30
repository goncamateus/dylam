"""Canonical source metadata for tab:res/pareto/hv-cardinality and the three
Pareto-front figures (sections/results/morl.tex, app:res_morl).

Not ablation arms (CONTEXT.md's "Arm" is one swept-hyperparameter cell) --
this registers where each method's Pareto candidates come from.

Two data sources, two protocols: DyLam and DynMORL are this project's own
runs (goncamateus/DyLam), and a Pareto candidate set is built per seed by
sampling that seed's training history and Pareto-filtering it (`kind ==
"history"`). GPI-LS and PGMORL are `morl-baselines` reference runs logged
to a separate wandb project (openrlbenchmark/MORL-Baselines); their
candidate set is the `eval/front` table already logged at the end of
training (`kind == "front"`). `kind` selects which protocol fetch_data.py
uses for a given Source.

Objectives are stored untransformed in the tidy CSVs (what was actually
logged); `transform` is applied by table.py/figure.py at analysis time, so
the tidy data stays an honest record of the raw metric.
"""
from collections import namedtuple
from pathlib import Path

import pandas as pd

from lib import pareto

DATA = Path(__file__).parent / "data"

DYLAM_PROJECT = "goncamateus/DyLam"
MORL_PROJECT = "openrlbenchmark/MORL-Baselines"

# manuscript's "10^4 evenly spaced points" per DyLam/DynMORL seed
DYLAM_SAMPLES = 10_000


def slug(label):
    return label.lower().replace(" ", "_").replace("-", "_")


def halfcheetah_dylam_transform(points):
    """ep_info/ctrl is a penalty; shift to the maximizing axis R_c = 1000 - Control
    the manuscript scores HalfCheetah on, matching HALFCHEETAH_REF's units."""
    points = points.copy()
    points[:, 1] = 1000.0 + points[:, 1]
    return points


def halfcheetah_morl_transform(points):
    """eval/front's second objective is control cost on morl-baselines' own
    scale; matches the shift + scale the DyLam side applies above."""
    points = points.copy()
    points[:, 1] = 1000.0 + points[:, 1] * 10
    return points


Source = namedtuple("Source", "label kind project env_or_id setup_or_algo metrics transform")

HALFCHEETAH_REF = (-1.0, -1.0)
HALFCHEETAH_SOURCES = [
    Source("PGMORL", "front", MORL_PROJECT, "mo-halfcheetah-v4", "PGMORL",
           None, halfcheetah_morl_transform),
    Source("GPI-LS", "front", MORL_PROJECT, "mo-halfcheetah-v4", "GPI-LS Continuous Action",
           None, halfcheetah_morl_transform),
    Source("DyLam", "history", DYLAM_PROJECT, "HALFCHEETAH", "Dylam",
           ["ep_info/run", "ep_info/ctrl"], halfcheetah_dylam_transform),
]

MINECART_REF = (0.0, 0.0, -1000.0)
MINECART_SOURCES = [
    Source("GPI-LS", "front", MORL_PROJECT, "minecart-v0", "GPI-LS", None, None),
    Source("DynMORL", "history", DYLAM_PROJECT, "MINECART", "Dynmorl",
           ["ep_info/First_minerium", "ep_info/Second_minerium", "ep_info/Fuel"], None),
    Source("DyLam", "history", DYLAM_PROJECT, "MINECART", "Dylam",
           ["ep_info/First_minerium", "ep_info/Second_minerium", "ep_info/Fuel"], None),
]

# lambda (weight) keys for the two figures' bottom/second panels, same fetch
# protocol as reward objectives (a joint per-step history sample).
HALFCHEETAH_WEIGHT_METRICS = ["lambdas/Run", "lambdas/Control"]
MINECART_WEIGHT_METRICS = ["lambdas/First_minerium", "lambdas/Second_minerium", "lambdas/Fuel"]

METHOD_ORDER = ["PGMORL", "GPI-LS", "DynMORL", "DyLam"]
ENVS = {"HALFCHEETAH": HALFCHEETAH_SOURCES, "MINECART": MINECART_SOURCES}
MAX_SEEDS = 10


def per_seed(env, source):
    """Yield (seed, seed's rows, seed's own non-dominated front) for one source.

    Shared by table.py (needs the seed group for wall-clock time) and
    figure.py (needs only the front, pooled across seeds and re-filtered
    for the picture) -- both were computing this same read-group-transform-
    filter shape independently.
    """
    df = pd.read_csv(DATA / f"{env.lower()}_{slug(source.label)}.csv")
    obj_cols = [c for c in df.columns if c.startswith("obj")]
    for seed, g in df.groupby("seed", sort=False):
        pts = g[obj_cols].to_numpy(dtype=float)
        if source.transform is not None:
            pts = source.transform(pts)
        yield seed, g, pareto.filter_dominated(pts)
