"""Re-export the RQ1 trad runs from wandb into committed tidy CSVs.

One file per paper column (methods.py's CELLS and COMPONENT_CELLS), each one
row per (step, seed) with a method column identifying which row of the
table -- or which curve of the figure -- it belongs to. This is a fetch
step, not a generator: it touches the network and is not covered by the
offline test seam. Re-run it (with --refresh to bypass the run-history
cache) whenever a method gets new finished seeds.

Usage: python fetch_data.py [--refresh]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from methods import CELLS, COMPONENT_CELLS

from core import fetch

DATA = Path(__file__).parent / "data"
# CB is rerun on explicit seeds: 1..10 for every table and test (as in the other
# environments), 1..100 only for the success-rate estimate.
CB_SEEDS = range(1, 11)
CB_SUCCESS_SEEDS = range(1, 101)


def slug(column):
    return re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="bypass the run-history cache")
    args = ap.parse_args()

    by_column = {}
    for cell in CELLS + COMPONENT_CELLS:
        by_column.setdefault(cell.column, []).append(cell)

    DATA.mkdir(parents=True, exist_ok=True)
    for column, cells in by_column.items():
        frames = []
        for cell in cells:
            hs = fetch.histories(cell.wandb_env, cell.wandb_setup, cell.metric, refresh=args.refresh,
                                 seeds=CB_SEEDS if cell.wandb_env == "CHICKENBANANA" else None)
            print(f"  {column:30s} {cell.method:16s} n={len(hs)}", file=sys.stderr)
            if not hs:
                sys.exit(f"no finished runs for {cell.wandb_env}/{cell.wandb_setup}")
            frames.append(fetch.tidy(hs, cell.metric, cell.method, column="method"))
        out = DATA / f"{slug(column)}.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"wrote {out}")

    # Success-rate estimate only (descriptive, no test): every CB arm on seeds 1..100.
    frames = []
    for cell in CELLS:
        if cell.column == "Chicken--Banana":
            hs = fetch.histories(cell.wandb_env, cell.wandb_setup, cell.metric,
                                 refresh=args.refresh, seeds=CB_SUCCESS_SEEDS)
            print(f"  CB success {cell.method:16s} n={len(hs)}", file=sys.stderr)
            frames.append(fetch.tidy(hs, cell.metric, cell.method, column="method"))
    out = DATA / "chicken_banana_success.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
