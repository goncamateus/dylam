"""Fetch morl scope data into tidy CSVs: per-seed Pareto candidate point
clouds for every (env, method) in arms.py, plus DyLam/DynMORL's weight
(lambda) trajectories for the two weight-space figure panels.

Two protocols, matching arms.Source.kind:
  "dylam" -- this project's own runs. A candidate set is this seed's
    training history (10^4 sampled points, arms.DYLAM_SAMPLES), Pareto
    filtering happens at analysis time in table.py/figure.py.
  "morl"  -- morl-baselines reference runs in a separate wandb project.
    A candidate set is the eval/front table already logged at the end of
    training (a JSON-backed wandb Table, downloaded once per run).

GPI-LS and PGMORL do not log an accessible per-policy weight vector (no
history key, no config field enumerating it) under either project, so the
weight-space figures include only DyLam/DynMORL, which do. The two
notebooks this migration replaces plotted GPI-LS/PGMORL points there too,
from untracked local CSVs of unknown provenance -- exactly the kind of
un-owned, hand-copied artifact this migration exists to eliminate, not
reproduce.

Fetch step, not a generator: touches the network, not covered by the
offline test seam.

Usage: python fetch_data.py [--refresh]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from arms import (ENVS, HALFCHEETAH_WEIGHT_METRICS, MAX_SEEDS,
                  MINECART_WEIGHT_METRICS)

from lib import fetch

DATA = Path(__file__).parent / "data"


def _slug(label):
    return label.lower().replace(" ", "_").replace("-", "_")


def _to_tidy(df, metrics, with_wall_time=True):
    cols = list(metrics) + ["seed"]
    names = [f"obj{i + 1}" for i in range(len(metrics))] + ["seed"]
    if with_wall_time:
        cols.append("_wall_time_min")
        names.append("wall_time_min")
    d = df[cols].copy()
    d.columns = names
    d["point_index"] = range(len(d))
    return d


def fetch_dylam_source(source, refresh):
    dfs = fetch.histories(source.env_or_id, source.setup_or_algo, source.metrics,
                          samples=10_000, max_seeds=MAX_SEEDS, refresh=refresh,
                          with_wall_time=True, entity_project=source.project)
    return [_to_tidy(df, source.metrics) for df in dfs]


def fetch_weights(env_or_id, setup, project, metrics, refresh):
    dfs = fetch.histories(env_or_id, setup, metrics, samples=10_000, max_seeds=MAX_SEEDS,
                          refresh=refresh, entity_project=project)
    return [_to_tidy(df, metrics, with_wall_time=False) for df in dfs]


def _parse_front_table(run, last):
    if isinstance(last, dict) and "data" in last:
        cols = last.get("columns", [])
        return pd.DataFrame(last["data"], columns=cols or None).select_dtypes(include=[np.number])
    if isinstance(last, dict) and "path" in last:
        with tempfile.TemporaryDirectory() as tmp:
            dl = run.file(last["path"]).download(root=tmp, replace=True)
            with open(dl.name) as fp:
                raw = json.load(fp)
            return pd.DataFrame(raw.get("data", []), columns=raw.get("columns")) \
                .select_dtypes(include=[np.number])
    return None


def fetch_morl_source(source, refresh):
    runs = fetch.api().runs(
        source.project,
        filters={"config.env_id": source.env_or_id, "config.algo": source.setup_or_algo,
                 "state": "finished"},
        order="-created_at",
    )
    frames, n = [], 0
    for run in runs:
        if n >= MAX_SEEDS:
            break
        cache = fetch.CACHE / f"morlfront_{run.id}.csv"
        df = pd.read_csv(cache) if cache.exists() and not refresh else None
        if df is None:
            rows = list(run.history(keys=["eval/front"], pandas=False))
            valid = [r["eval/front"] for r in rows if r.get("eval/front") is not None]
            if not valid:
                continue
            df = _parse_front_table(run, valid[-1])
            if df is None or len(df) < 2:
                continue
            df.to_csv(cache, index=False)
        if len(df) < 2:
            continue
        d = df.copy()
        d.columns = [f"obj{i + 1}" for i in range(d.shape[1])]
        runtime_s = run.summary.get("_runtime")
        d["wall_time_min"] = runtime_s / 60.0 if runtime_s is not None else None
        d["seed"] = run.config.get("seed", run.id)
        d["point_index"] = range(len(d))
        frames.append(d)
        n += 1
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="bypass the run-history cache")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    for env, sources in ENVS.items():
        for source in sources:
            print(f"  {env:12s} {source.label:8s} ...", file=sys.stderr, end=" ")
            fetch_fn = fetch_dylam_source if source.kind == "dylam" else fetch_morl_source
            frames = fetch_fn(source, args.refresh)
            print(f"n={len(frames)}", file=sys.stderr)
            if not frames:
                sys.exit(f"no data for {env}/{source.label}")
            out = DATA / f"{env.lower()}_{_slug(source.label)}.csv"
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
            print(f"wrote {out}", file=sys.stderr)

    # (env, setup, project, metrics, out filename)
    weight_jobs = [
        ("HALFCHEETAH", "Dylam", "goncamateus/DyLam", HALFCHEETAH_WEIGHT_METRICS,
         "halfcheetah_dylam_weights.csv"),
        ("MINECART", "Dylam", "goncamateus/DyLam", MINECART_WEIGHT_METRICS,
         "minecart_dylam_weights.csv"),
        ("MINECART", "Dynmorl", "goncamateus/DyLam", MINECART_WEIGHT_METRICS,
         "minecart_dynmorl_weights.csv"),
    ]
    for env, setup, project, metrics, fname in weight_jobs:
        print(f"  {env:12s} {setup:8s} weights ...", file=sys.stderr, end=" ")
        frames = fetch_weights(env, setup, project, metrics, args.refresh)
        print(f"n={len(frames)}", file=sys.stderr)
        if not frames:
            sys.exit(f"no weight data for {env}/{setup}")
        out = DATA / fname
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
