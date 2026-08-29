"""Recompute every statistic the R1 revision needs, from wandb, per seed.

Adds to what table1_update.py already did:
  * true n per cell (several baselines have fewer than 10 finished seeds)
  * IQM with a bootstrap CI alongside mean +/- std
  * exact Mann-Whitney U, rank-biserial effect size, Holm within each family
  * HalfCheetah scored on the environment's own scalar return (ep_info/total)
  * steps-to-threshold and normalized learning-curve AUC (sample efficiency)
  * Chicken-Banana success rate

The RQ3 robustness family has moved to robustness/table.py, the one
generator for tab:res/robustness/summary; this script no longer computes it.

Usage:  python stats_r1.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy.stats import mannwhitneyu

CACHE = Path(__file__).parent / "r1_cache"
CACHE.mkdir(exist_ok=True)
EP = "goncamateus/DyLam"
MAX_SEEDS = 10
SAMPLES = 5000

api = wandb.Api(timeout=180)

# (paper label, wandb env, metric, {row label: setup})
ENVS = {
    "Chicken--Banana": ("CHICKENBANANA", "ep_info/total", {
        "Base SO RL": "Baseline", "Q-Decomposition": "Decq",
        "UDC": "Drq", "DyLam": "Dylam"}),
    "HalfCheetah-v4": ("HALFCHEETAH", "ep_info/Final_position", {
        "Base SO RL": "Baseline", "UDC": "Drq", "DyLam": "Dylam"}),
    "HalfCheetah-v4 (env return)": ("HALFCHEETAH", "ep_info/total", {
        "Base SO RL": "Baseline", "UDC": "Drq", "DyLam": "Dylam"}),
    "VSS-v0": ("VSS", "ep_info/Goal", {
        "Base SO RL": "Baseline", "UDC": "Drq", "DyLam": "Dylam"}),
}
TUNED = ("VSS_TUNED", "Drq", "ep_info/Goal", "VSS-v0", "Tuned-UDC")


def histories(env, setup, metric, refresh=False):
    """Per-seed (step, value) frames, newest runs first, capped at MAX_SEEDS."""
    runs = api.runs(EP, filters={"config.env": env, "config.setup": setup,
                                 "state": "finished"}, order="-created_at")
    out = []
    for run in runs:
        if len(out) >= MAX_SEEDS:
            break
        f = CACHE / f"{run.id}_{metric.replace('/', '_')}.csv"
        if f.exists() and not refresh:
            df = pd.read_csv(f)
        else:
            df = run.history(samples=SAMPLES, keys=[metric], pandas=True)
            df = df.dropna(subset=[metric]) if metric in df else pd.DataFrame()
            df.to_csv(f, index=False)
        if metric in df and len(df):
            out.append(df)
    return out


def final(df, metric):
    """Per-seed summary: mean of the final 10% of the run, as in the paper."""
    v = df[metric].to_numpy(dtype=float)
    return float(v[int(0.9 * len(v)):].mean())


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    lo, hi = int(np.floor(0.25 * len(x))), int(np.ceil(0.75 * len(x)))
    return float(x[lo:hi].mean())


def boot(x, stat=iqm, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    s = [stat(rng.choice(x, len(x), replace=True)) for _ in range(B)]
    return tuple(np.percentile(s, [2.5, 97.5]))


def exact_mw(a, b):
    """Exact two-sided Mann-Whitney + rank-biserial (positive => a > b)."""
    u, p = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    return u, p, 2.0 * u / (len(a) * len(b)) - 1.0


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def steps_to(df, metric, thr):
    """First logged step whose 20-point rolling mean reaches thr; None if never."""
    d = df.dropna(subset=[metric])
    roll = d[metric].rolling(20, min_periods=5).mean().to_numpy()
    step = d["_step"].to_numpy() if "_step" in d else np.arange(len(d))
    hit = np.where(roll >= thr)[0]
    return float(step[hit[0]]) if len(hit) else None


def auc(df, metric, lo, hi):
    v = df[metric].to_numpy(dtype=float)
    return float(np.clip((v - lo) / (hi - lo), 0, 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    data, curves = {}, {}
    for col, (env, metric, setups) in ENVS.items():
        for row, setup in setups.items():
            hs = histories(env, setup, metric, args.refresh)
            data[(row, col)] = [final(h, metric) for h in hs]
            curves[(row, col)] = hs
            print(f"  {col:30s} {row:16s} n={len(hs)}", file=sys.stderr, flush=True)
    env, setup, metric, col, row = TUNED
    hs = histories(env, setup, metric, args.refresh)
    data[(row, col)] = [final(h, metric) for h in hs]
    curves[(row, col)] = hs
    print(f"  {col:30s} {row:16s} n={len(hs)}", file=sys.stderr)

    print("\n%%%% TABLE 1: per-cell summary")
    for col, (_, metric, _) in ENVS.items():
        print(f"\n% {col}   metric={metric}")
        for (r, c), vals in data.items():
            if c != col or not vals:
                continue
            lo, hi = boot(vals)
            print(f"%   {r:16s} n={len(vals):2d}  mean {np.mean(vals):9.3f} +- {np.std(vals, ddof=1):7.3f}"
                  f"   IQM {iqm(vals):9.3f}  95% CI [{lo:9.3f}, {hi:9.3f}]")

    print("\n%%%% RQ1 FAMILY: DyLam vs strongest baseline, exact MW + Holm")
    fam, labels, secondary = [], [], []
    for col in ENVS:
        dl = data.get(("DyLam", col), [])
        rivals = {r: v for (r, c), v in data.items()
                  if c == col and r != "DyLam" and v}
        if not dl or not rivals:
            continue
        best = max(rivals, key=lambda r: np.mean(rivals[r]))
        # the env-return scoring of HalfCheetah is a re-scoring of a test already in
        # the family, not a fourth independent comparison, so it stays out of Holm
        target = secondary if "env return" in col else fam
        target.append((col, best, len(dl), len(rivals[best])) + exact_mw(dl, rivals[best]))
    adj = holm([row[5] for row in fam])
    for row, a in zip(fam, adj):
        col, best, n1, n2, u, p, r = row
        print(f"%   {col:30s} vs {best:16s} n={n1}/{n2}  U={u:6.1f}  p_exact={p:.3g}"
              f"  p_Holm={a:.3g}  rank-biserial r={r:+.3f}")
    for col, best, n1, n2, u, p, r in secondary:
        print(f"%   [outside family] {col:30s} vs {best:16s} n={n1}/{n2}  U={u:6.1f}"
              f"  p_exact={p:.3g}  rank-biserial r={r:+.3f}")

    print("\n%%%% CHICKEN-BANANA success rate (solved := final-10% mean > 160 of 200)")
    for r in ("Base SO RL", "Q-Decomposition", "UDC", "DyLam"):
        v = data.get((r, "Chicken--Banana"), [])
        if v:
            print(f"%   {r:16s} {sum(x > 160 for x in v)}/{len(v)}")

    print("\n%%%% RQ1 SAMPLE EFFICIENCY: steps to the strongest baseline's final mean")
    for col, (_, metric, _) in ENVS.items():
        rivals = {r: v for (r, c), v in data.items()
                  if c == col and r != "DyLam" and v}
        if not rivals or not data.get(("DyLam", col)):
            continue
        best = max(rivals, key=lambda r: np.mean(rivals[r]))
        thr = float(np.mean(rivals[best]))
        allv = [x for (r, c), v in data.items() if c == col for x in v]
        lo, hi = min(allv), max(allv)
        print(f"% {col}  threshold = {best} final mean = {thr:.3f}")
        for r in list(rivals) + ["DyLam"]:
            hs = curves.get((r, col), [])
            st = [steps_to(h, metric, thr) for h in hs]
            hit = [s for s in st if s is not None]
            au = [auc(h, metric, lo, hi) for h in hs]
            med = f"{np.median(hit):,.0f}" if hit else "never"
            print(f"%   {r:16s} reached {len(hit)}/{len(hs)} seeds, median steps {med:>12s}"
                  f"   AUC {np.mean(au):.3f} +- {np.std(au, ddof=1) if len(au) > 1 else 0:.3f}")

    # RQ3 robustness table now lives in robustness/table.py (one generator,
    # writing straight into the paper repo); `nom` is still needed below by
    # the R5 open-loop comparison, which has not migrated yet.
    nom = data.get(("DyLam", "VSS-v0"), [])

    print("\n%%%% R5 OPEN-LOOP REPLAY: DyLam vs its own lambda(t) schedule on VSS-v0")
    ol = [final(h, "ep_info/Goal")
          for h in histories("VSS", "Dylam_Openloop", "ep_info/Goal", args.refresh)]
    if not ol:
        print("%   no Dylam_Openloop runs yet (scripts/run_r1_vss.sh openloop)")
    else:
        lo, hi = boot(ol)
        u, p, r = exact_mw(nom, ol)
        print(f"%   Open-loop n={len(ol):2d} mean {np.mean(ol):.3f} +- {np.std(ol, ddof=1):.3f}"
              f"  IQM {iqm(ol):.3f} CI [{lo:.3f}, {hi:.3f}]")
        print(f"%   DyLam vs open-loop  U={u:5.1f}  p_exact={p:.4g}  r={r:+.3f}")


if __name__ == "__main__":
    main()
