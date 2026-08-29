"""Re-export the RQ3 robustness runs from wandb into a committed tidy CSV.

VSS-v0, DyLam under the nominal bounds and the six bound perturbations,
metric ep_info/Goal. One row per (step, seed); the condition column
identifies which perturbation. This is a fetch step, not a generator: it
touches the network and is not covered by the offline test seam. Re-run it
(with --refresh to bypass the run-history cache) whenever a condition gets
new finished seeds.

Usage: python fetch_data.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from conditions import CONDITIONS
from lib import fetch

METRIC = "ep_info/Goal"
OUT = Path(__file__).parent / "data" / "vss_ep_info_goal.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="bypass the run-history cache")
    args = ap.parse_args()

    frames = []
    for key, c in CONDITIONS.items():
        hs = fetch.histories(c.env, c.setup, METRIC, refresh=args.refresh)
        print(f"  {key:30s} {c.env:22s} n={len(hs)}", file=sys.stderr)
        if not hs:
            sys.exit(f"no finished runs for {c.env}/{c.setup}")
        frames.append(fetch.tidy(hs, METRIC, key))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
