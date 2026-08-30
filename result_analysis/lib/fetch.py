"""Wandb query + caching. The only module in result_analysis that touches the network.

The cache is disposable: it exists to avoid re-downloading a run's history on
every invocation, keyed by wandb run id, and lives outside the repository
under the system temp directory. It is not a data source of record -- the
tidy CSVs committed under each scope's data/ directory are.

This data-layer split (tidy CSVs committed, run cache ephemeral) is the
choice an ADR still needs to record and the author still needs to sign off
on -- see the issue's "Further Notes".
"""
import tempfile
from pathlib import Path

import pandas as pd
import wandb

ENTITY_PROJECT = "goncamateus/DyLam"
CACHE = Path(tempfile.gettempdir()) / "dylam_result_analysis_cache"
CACHE.mkdir(parents=True, exist_ok=True)

_api = None


def api():
    global _api
    if _api is None:
        _api = wandb.Api(timeout=180)
    return _api


def histories(env, setup, metric, max_seeds=10, samples=5000, refresh=False,
              entity_project=ENTITY_PROJECT, with_wall_time=False):
    """Per-seed (step, metric) frames for one (env, setup) pair, newest first.

    `metric` is one wandb history key, or a list of keys to fetch jointly so
    they line up on the same row -- morl's Pareto-front candidates need
    several objectives read at the same step, not one column at a time.

    Each frame carries a `seed` column (the run's `config.seed`, falling back
    to the run id when a run predates that config key). Capped at `max_seeds`
    finished runs. `entity_project` defaults to this project; morl's rival
    methods (GPI-LS, PGMORL) live in a different one. With
    `with_wall_time`, each frame also carries a constant `_wall_time_min`
    column (the run's total wall-clock time), for morl's timing comparison.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    runs = api().runs(
        entity_project,
        filters={"config.env": env, "config.setup": setup, "state": "finished"},
        order="-created_at",
    )
    out = []
    cache_key = "_".join(m.replace("/", "_") for m in metrics)
    if with_wall_time:
        cache_key += "_wt"
    for run in runs:
        if len(out) >= max_seeds:
            break
        f = CACHE / f"{run.id}_{cache_key}.csv"
        if f.exists() and not refresh:
            df = pd.read_csv(f)
        else:
            df = run.history(samples=samples, keys=metrics, pandas=True)
            df = df.dropna(subset=metrics) if all(m in df for m in metrics) else pd.DataFrame()
            if with_wall_time and len(df):
                runtime_s = run.summary.get("_runtime")
                df["_wall_time_min"] = runtime_s / 60.0 if runtime_s is not None else None
            df.to_csv(f, index=False)
        if all(m in df for m in metrics) and len(df):
            df = df.copy()
            df["seed"] = run.config.get("seed", run.id)
            out.append(df)
    return out


def tidy(dfs, metric, value, column="condition"):
    """Concatenate per-seed frames into one long-form frame with an identifying column.

    `column` names what `value` identifies -- "condition" for robustness's
    bound perturbations, "method" for trad's row labels, "arm" for
    ablation's swept hyperparameter values -- since that identity concept
    differs by scope (see CONTEXT.md).
    """
    parts = []
    for df in dfs:
        d = df[["_step", metric, "seed"]].copy()
        d[column] = value
        parts.append(d)
    return pd.concat(parts, ignore_index=True)
