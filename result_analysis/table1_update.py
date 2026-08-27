"""Regenerate Table 1 of the paper (learning-dynamics environments) from wandb.

Per-seed summary = mean of the final 10% of each run's logged history, which is
the unit of independence used for all statistical tests in the paper.

Prints a paste-ready LaTeX tabular body plus the significance test against the
strongest non-DyLam baseline in each environment.

Usage:  python table1_update.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy.stats import mannwhitneyu

CACHE = Path(__file__).parent / "table1_cache"
CACHE.mkdir(exist_ok=True)
ENTITY_PROJECT = "goncamateus/DyLam"
MAX_SEEDS = 10
HISTORY_SAMPLES = 5000

# paper label -> (wandb env, wandb setup)
COLUMNS = {
    "Chicken--Banana": dict(env="CHICKENBANANA", metric="ep_info/total", setups={
        "Base SO RL": "Baseline", "Q-Decomposition": "Decq",
        "UDC": "Drq", "DyLam": "Dylam"}),
    "HalfCheetah-v4": dict(env="HALFCHEETAH", metric="ep_info/Final_position", setups={
        "Base SO RL": "Baseline", "UDC": "Drq", "DyLam": "Dylam"}),
    "VSS-v0": dict(env="VSS", metric="ep_info/Goal", setups={
        "Base SO RL": "Baseline", "UDC": "Drq", "DyLam": "Dylam"}),
}
# Tuned-UDC lives in its own wandb env
TUNED = dict(env="VSS_TUNED", setup="Drq", metric="ep_info/Goal", column="VSS-v0")
ROW_ORDER = ["Base SO RL", "Q-Decomposition", "UDC", "Tuned-UDC", "DyLam"]

api = wandb.Api(timeout=120)


def run_history(run, metric, refresh=False):
    f = CACHE / f"{run.id}.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f)
    df = run.history(samples=HISTORY_SAMPLES, keys=[metric], pandas=True)
    df = df.dropna(subset=[metric])
    df.to_csv(f, index=False)
    return df


def per_seed(env, setup, metric, refresh=False):
    runs = api.runs(ENTITY_PROJECT,
                    filters={"config.env": env, "config.setup": setup, "state": "finished"},
                    order="-created_at")
    vals = []
    for run in runs:
        if len(vals) >= MAX_SEEDS:
            break
        df = run_history(run, metric, refresh)
        if metric not in df or len(df) == 0:
            continue
        tail = df[metric].iloc[int(0.9 * len(df)):]
        vals.append(float(tail.mean()))
    return vals


def boot_ci(x, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, (B, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(vals, decimals):
    if not vals:
        return "---"
    return f"{np.mean(vals):.{decimals}f} $\\pm$ {np.std(vals, ddof=1):.{decimals}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the local cache")
    args = ap.parse_args()

    data = {}       # (row, column) -> per-seed list
    for col, spec in COLUMNS.items():
        for row, setup in spec["setups"].items():
            data[(row, col)] = per_seed(spec["env"], setup, spec["metric"], args.refresh)
            print(f"  fetched {col:16s} {row:16s} n={len(data[(row, col)])}",
                  file=sys.stderr, flush=True)
    data[("Tuned-UDC", TUNED["column"])] = per_seed(TUNED["env"], TUNED["setup"],
                                                    TUNED["metric"], args.refresh)
    print(f"  fetched {TUNED['column']:16s} {'Tuned-UDC':16s} "
          f"n={len(data[('Tuned-UDC', TUNED['column'])])}", file=sys.stderr)

    decimals = {"Chicken--Banana": 3, "HalfCheetah-v4": 3, "VSS-v0": 3}
    stars, tests = {}, []
    for col in COLUMNS:
        dylam = data.get(("DyLam", col), [])
        rivals = {r: v for r, v in
                  ((r, data.get((r, col), [])) for r in ROW_ORDER if r != "DyLam") if v}
        if not dylam or not rivals:
            continue
        best = max(rivals, key=lambda r: np.mean(rivals[r]))
        u, p = mannwhitneyu(dylam, rivals[best], alternative="two-sided")
        stars[col] = p < 0.05
        tests.append((col, best, len(dylam), len(rivals[best]), u, p))

    print("\n% --- paste into Table~\\ref{tab:res/trad/summary} ---")
    for row in ROW_ORDER:
        cells = []
        for col in COLUMNS:
            vals = data.get((row, col), [])
            cell = fmt(vals, decimals[col])
            if row == "DyLam" and vals and stars.get(col):
                cell += "$^\\ast$"
            cells.append(cell)
        print(f"{row:16s} & " + " & ".join(cells) + r" \\")

    print("\n% n per cell")
    for row in ROW_ORDER:
        print("%   " + f"{row:16s} " +
              "  ".join(f"{col}: n={len(data.get((row, col), []))}" for col in COLUMNS))

    print("\n% Mann-Whitney, DyLam vs strongest baseline")
    for col, best, n1, n2, u, p in tests:
        ci = boot_ci(data[("DyLam", col)])
        print(f"%   {col:16s} vs {best:16s} n={n1}/{n2}  U={u:.0f}  p={p:.4g}"
              f"   DyLam 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")


if __name__ == "__main__":
    main()
