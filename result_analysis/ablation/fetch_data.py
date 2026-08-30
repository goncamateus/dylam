"""Re-export the Chicken--Banana ablation sweeps from wandb into tidy CSVs.

One file per (arm group, component, metric): data/nominal_<comp>_<kind>.csv
for the shared nominal reference (fetched once, reused by all four ablation
figures), data/<type>_<comp>_<kind>.csv for each type's swept arms (`kind`
is "reward" or "lambda"), and data/normalizer_table_total.csv for
tab:ablation/normalizer's four rows (total episode reward only -- the
table doesn't need per-component or lambda data). Fetch step, not a
generator: touches the network, not covered by the offline test seam.

Usage: python fetch_data.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from arms import (COMPONENTS, ENV, EPSILON_ARMS, LAMBDA_METRICS, NOMINAL_SETUP,
                  NORMALIZER_ARMS, NORMALIZER_TABLE_ROWS, RB_ARMS,
                  REWARD_METRICS, TAU_ARMS)

from lib import fetch

DATA = Path(__file__).parent / "data"
TOTAL_METRIC = "ep_info/total"

SWEEPS = {"tau": TAU_ARMS, "rb": RB_ARMS, "epsilon": EPSILON_ARMS, "normalizer": NORMALIZER_ARMS}


def _fetch_one(setup, metric, refresh, comp=""):
    hs = fetch.histories(ENV, setup, metric, refresh=refresh)
    print(f"  {setup:28s} {comp:8s} {metric:16s} n={len(hs)}", file=sys.stderr)
    if not hs:
        sys.exit(f"no finished runs for {ENV}/{setup}")
    return hs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="bypass the run-history cache")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    for comp in COMPONENTS:
        for kind, metric in (("reward", REWARD_METRICS[comp]), ("lambda", LAMBDA_METRICS[comp])):
            hs = _fetch_one(NOMINAL_SETUP, metric, args.refresh, comp)
            fetch.tidy(hs, metric, "nominal", column="arm").to_csv(
                DATA / f"nominal_{comp.lower()}_{kind}.csv", index=False)

    for sweep, arms in SWEEPS.items():
        for comp in COMPONENTS:
            for kind, metric in (("reward", REWARD_METRICS[comp]),
                                 ("lambda", LAMBDA_METRICS[comp])):
                frames = [fetch.tidy(_fetch_one(a.wandb_setup, metric, args.refresh, comp),
                                     metric, a.label, column="arm") for a in arms]
                out = DATA / f"{sweep}_{comp.lower()}_{kind}.csv"
                pd.concat(frames, ignore_index=True).to_csv(out, index=False)
                print(f"wrote {out}", file=sys.stderr)

    frames = [fetch.tidy(_fetch_one(setup, TOTAL_METRIC, args.refresh), TOTAL_METRIC, label,
                         column="arm")
              for label, setup in NORMALIZER_TABLE_ROWS]
    out = DATA / "normalizer_table_total.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
