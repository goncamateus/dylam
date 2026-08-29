"""Re-export the RQ3 robustness runs from wandb into a committed tidy CSV.

VSS-v0, DyLam under the nominal bounds and the six bound perturbations,
metric ep_info/Goal. One row per (step, seed); the arm identifies the
condition. This is a fetch step, not a generator: it touches the network and
is not covered by the offline test seam. Re-run it (with --refresh to bypass
the run-history cache) whenever a condition gets new finished seeds.

Usage: python fetch_data.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from lib import fetch

METRIC = "ep_info/Goal"
OUT = Path(__file__).parent / "data" / "vss_ep_info_goal.csv"

# (arm, wandb env, wandb setup)
CONDITIONS = [
    ("nominal", "VSS", "Dylam"),
    ("move_m25", "ROBUSTNESS_MOVE2", "Dylam"),
    ("move_p50", "ROBUSTNESS_MOVE1", "Dylam"),
    ("ball_m25", "ROBUSTNESS_BALL_M25", "Dylam"),
    ("ball_p25", "ROBUSTNESS_BALL_P25", "Dylam"),
    ("compound_move_p50_ball_p25", "ROBUSTNESS_BALL1", "Dylam"),
    ("compound_move_m25_ball_m50", "ROBUSTNESS_BALL2", "Dylam"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="bypass the run-history cache")
    args = ap.parse_args()

    frames = []
    for arm, env, setup in CONDITIONS:
        hs = fetch.histories(env, setup, METRIC, refresh=args.refresh)
        print(f"  {arm:30s} {env:22s} n={len(hs)}", file=sys.stderr)
        if not hs:
            sys.exit(f"no finished runs for {env}/{setup}")
        frames.append(fetch.tidy(hs, METRIC, arm))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
