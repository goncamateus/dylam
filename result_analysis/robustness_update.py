"""Regenerate the RQ3 robustness table (VSS-v0 bound misspecification) from wandb.

Same protocol as table1_update.py: per-seed summary = mean of the final 10% of
each run's logged goal rate. Reports mean +/- std, 95% bootstrap CI, Mann-Whitney
U against the nominal configuration, and Holm-Bonferroni corrected p-values
within the RQ3 family.

The four conditions are the intended one-at-a-time +/-25% perturbations of the
nominal bounds r_max = (150, 40, -100). ROBUSTNESS_MOVE2 already exists; the
other three are launched by scripts/run_missing_seeds.sh.

Usage:  python robustness_update.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy.stats import mannwhitneyu

CACHE = Path(__file__).parent / "robustness_cache"
CACHE.mkdir(exist_ok=True)
ENTITY_PROJECT = "goncamateus/DyLam"
METRIC = "ep_info/Goal"
MAX_SEEDS = 10
HISTORY_SAMPLES = 5000

NOMINAL = ("VSS", "Dylam")

# (label, wandb env, r_max as run). The first four are the intended one-at-a-time
# +/-25% design; MOVE2 already exists, the other three are launched by
# scripts/run_missing_seeds.sh. The last three are the conditions that were
# actually run for the first submission: MOVE1 perturbs the move ceiling by
# +50%, and BALL1/BALL2 perturb the ball ceiling *on top of* a move
# perturbation, so they are compound rather than one-at-a-time.
CONDITIONS = [
    ("Move $-25\\%$",                    "ROBUSTNESS_MOVE2",    "(112.5, 40, -100)"),
    ("Move $+25\\%$",                    "ROBUSTNESS_MOVE_P25", "(187.5, 40, -100)"),
    ("Ball-to-goal $-25\\%$",            "ROBUSTNESS_BALL_M25", "(150, 30, -100)"),
    ("Ball-to-goal $+25\\%$",            "ROBUSTNESS_BALL_P25", "(150, 50, -100)"),
    ("Move $+50\\%$",                    "ROBUSTNESS_MOVE1",    "(225, 40, -100)"),
    ("Move $+50\\%$, ball $+25\\%$",    "ROBUSTNESS_BALL1",    "(225, 50, -100)"),
    ("Move $-25\\%$, ball $-50\\%$",    "ROBUSTNESS_BALL2",    "(112.5, 20, -100)"),
]

api = wandb.Api(timeout=120)


def per_seed(env, setup="Dylam", refresh=False):
    runs = api.runs(ENTITY_PROJECT,
                    filters={"config.env": env, "config.setup": setup, "state": "finished"},
                    order="-created_at")
    vals, bounds = [], None
    for run in runs:
        if len(vals) >= MAX_SEEDS:
            break
        f = CACHE / f"{run.id}.csv"
        if f.exists() and not refresh:
            df = pd.read_csv(f)
        else:
            df = run.history(samples=HISTORY_SAMPLES, keys=[METRIC], pandas=True)
            df = df.dropna(subset=[METRIC])
            df.to_csv(f, index=False)
        if METRIC not in df or len(df) == 0:
            continue
        vals.append(float(df[METRIC].iloc[int(0.9 * len(df)):].mean()))
        bounds = bounds or tuple(run.config.get("r_max") or ())
    return vals, bounds


def boot_ci(x, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    return tuple(np.percentile(rng.choice(x, (B, len(x)), replace=True).mean(axis=1), [2.5, 97.5]))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running = [0.0] * len(pvals), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        out[idx] = running
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    nom, nom_bounds = per_seed(*NOMINAL, refresh=args.refresh)
    print(f"  nominal n={len(nom)} r_max={nom_bounds}", file=sys.stderr)
    if not nom:
        sys.exit("no nominal runs found")

    rows, raw_p = [], []
    for label, env, expected in CONDITIONS:
        vals, bounds = per_seed(env, refresh=args.refresh)
        print(f"  {env:22s} n={len(vals)} r_max={bounds}  (expected {expected})", file=sys.stderr)
        if not vals:
            rows.append((label, expected, None, None, None, None))
            continue
        u, p = mannwhitneyu(vals, nom, alternative="two-sided")
        rows.append((label, expected, vals, u, p, bounds))
        raw_p.append(p)

    adj = holm(raw_p)
    adj_iter = iter(adj)

    lo, hi = boot_ci(nom)
    print("\n% --- paste into Table~\\ref{tab:res/robustness/summary} ---")
    print(f"Nominal & $(150, 40, -100)$ & ${np.mean(nom):.3f} \\pm {np.std(nom, ddof=1):.3f}$ "
          f"& $[{lo:.3f}, {hi:.3f}]$ & --- & --- & --- \\\\")
    for label, expected, vals, u, p, bounds in rows:
        if vals is None:
            print(f"{label} & ${expected}$ & \\emph{{pending}} & --- & --- & --- & --- \\\\")
            continue
        a = next(adj_iter)
        lo, hi = boot_ci(vals)
        mark = "^{\\ast}" if a < 0.05 else "~\\text{n.s.}"
        print(f"{label} & ${expected}$ & ${np.mean(vals):.3f} \\pm {np.std(vals, ddof=1):.3f}$ "
              f"& $[{lo:.3f}, {hi:.3f}]$ & ${u:.0f}$ & ${p:.4f}$ & ${a:.3f}{mark}$ \\\\")

    print("\n% seeds per condition")
    print(f"%   Nominal n={len(nom)}")
    for label, expected, vals, *_ in rows:
        print(f"%   {label} n={0 if vals is None else len(vals)}")


if __name__ == "__main__":
    main()
