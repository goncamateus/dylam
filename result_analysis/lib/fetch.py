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


def histories(env, setup, metric, max_seeds=10, samples=5000, refresh=False):
    """Per-seed (step, metric) frames for one (env, setup) pair, newest first.

    Each frame carries a `seed` column (the run's `config.seed`, falling back
    to the run id when a run predates that config key). Capped at `max_seeds`
    finished runs.
    """
    runs = api().runs(
        ENTITY_PROJECT,
        filters={"config.env": env, "config.setup": setup, "state": "finished"},
        order="-created_at",
    )
    out = []
    for run in runs:
        if len(out) >= max_seeds:
            break
        f = CACHE / f"{run.id}_{metric.replace('/', '_')}.csv"
        if f.exists() and not refresh:
            df = pd.read_csv(f)
        else:
            df = run.history(samples=samples, keys=[metric], pandas=True)
            df = df.dropna(subset=[metric]) if metric in df else pd.DataFrame()
            df.to_csv(f, index=False)
        if metric in df and len(df):
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
