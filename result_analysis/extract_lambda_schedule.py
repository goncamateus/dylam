"""Record DyLam's mean lambda(t) trajectory on VSS-v0 as an open-loop schedule.

Feeds the R5 ablation: replaying this schedule with no return feedback and no
bounds separates "adaptivity" from "this particular lambda(t) curve". The output
is a CSV of (step, one column per component) that SACStratOpenLoop reads.

Usage:  python extract_lambda_schedule.py [--env VSS] [--out ../scripts/schedules/vss_dylam_lambda.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

EP = "goncamateus/DyLam"
GRID = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="VSS")
    ap.add_argument("--setup", default="Dylam")
    ap.add_argument("--components", nargs="+", default=["Move", "Ball", "Energy"])
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "scripts/schedules/vss_dylam_lambda.csv"))
    args = ap.parse_args()

    keys = [f"lambdas/{c}" for c in args.components]
    api = wandb.Api(timeout=180)
    runs = api.runs(EP, filters={"config.env": args.env, "config.setup": args.setup,
                                 "state": "finished"}, order="-created_at")

    per_seed, horizon = [], []
    for run in list(runs)[:10]:
        df = run.history(samples=5000, keys=keys, pandas=True).dropna()
        if not len(df):
            continue
        per_seed.append(df)
        horizon.append(df["_step"].max())
        print(f"  {run.id}: {len(df)} points up to step {df['_step'].max():.0f}")

    if not per_seed:
        raise SystemExit("no lambda histories found")

    grid = np.linspace(0, min(horizon), GRID)
    stacked = np.stack([
        np.stack([np.interp(grid, df["_step"], df[k]) for k in keys])
        for df in per_seed
    ])                                   # (seeds, components, grid)
    mean = stacked.mean(axis=0)

    out = pd.DataFrame({"step": grid})
    for c, col in zip(args.components, mean):
        out[c] = col
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows, {len(per_seed)} seeds averaged")
    print(out.iloc[[0, len(out) // 4, len(out) // 2, -1]].to_string(index=False))


if __name__ == "__main__":
    main()
