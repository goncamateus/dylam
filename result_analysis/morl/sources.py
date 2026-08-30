"""Canonical source metadata for tab:res/pareto/hv-cardinality and the three
Pareto-front figures (sections/results/morl.tex, app:res_morl).

Two data sources, two protocols: DyLam and DynMORL are this project's own
runs (goncamateus/DyLam), and a Pareto candidate set is built per seed by
sampling that seed's training history and Pareto-filtering it. GPI-LS and
PGMORL are `morl-baselines` reference runs logged to a separate wandb
project (openrlbenchmark/MORL-Baselines); their candidate set is the
`eval/front` table already logged at the end of training. `kind` below
selects which protocol fetch_data.py uses for a given Source.

Objectives are stored untransformed in the tidy CSVs (what was actually
logged); `transform` is applied by table.py/figure.py at analysis time, so
the tidy data stays an honest record of the raw metric.
"""
from collections import namedtuple

DYLAM_PROJECT = "goncamateus/DyLam"
MORL_PROJECT = "openrlbenchmark/MORL-Baselines"

# manuscript's "10^4 evenly spaced points" per DyLam/DynMORL seed
DYLAM_SAMPLES = 10_000


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
    Source("PGMORL", "morl", MORL_PROJECT, "mo-halfcheetah-v4", "PGMORL",
           None, halfcheetah_morl_transform),
    Source("GPI-LS", "morl", MORL_PROJECT, "mo-halfcheetah-v4", "GPI-LS Continuous Action",
           None, halfcheetah_morl_transform),
    Source("DyLam", "dylam", DYLAM_PROJECT, "HALFCHEETAH", "Dylam",
           ["ep_info/run", "ep_info/ctrl"], halfcheetah_dylam_transform),
]

MINECART_REF = (0.0, 0.0, -1000.0)
MINECART_SOURCES = [
    Source("GPI-LS", "morl", MORL_PROJECT, "minecart-v0", "GPI-LS", None, None),
    Source("DynMORL", "dylam", DYLAM_PROJECT, "MINECART", "Dynmorl",
           ["ep_info/First_minerium", "ep_info/Second_minerium", "ep_info/Fuel"], None),
    Source("DyLam", "dylam", DYLAM_PROJECT, "MINECART", "Dylam",
           ["ep_info/First_minerium", "ep_info/Second_minerium", "ep_info/Fuel"], None),
]

# lambda (weight) keys for the two figures' bottom/second panels, same fetch
# protocol as reward objectives (a joint per-step history sample).
HALFCHEETAH_WEIGHT_METRICS = ["lambdas/Run", "lambdas/Control"]
MINECART_WEIGHT_METRICS = ["lambdas/First_minerium", "lambdas/Second_minerium", "lambdas/Fuel"]

METHOD_ORDER = ["PGMORL", "GPI-LS", "DynMORL", "DyLam"]
ENVS = {"HALFCHEETAH": HALFCHEETAH_SOURCES, "MINECART": MINECART_SOURCES}
MAX_SEEDS = 10
