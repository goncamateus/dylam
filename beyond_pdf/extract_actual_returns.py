"""One-off data-prep step: extract the chosen seed's per-episode per-Component
returns from the curriculum scope's committed tidy CSVs, for export.py's ghost
bars. Mirrors curriculum/fetch_data.py's role (a fetch step, not a generator) --
except this reads only already-committed CSVs, so it never touches the network
and needs no test seam of its own (see beyond_pdf/export.py's module docstring
for what *is* tested).

Writes one small derived CSV: beyond_pdf/data/chickenbanana_actual_returns.csv,
columns episode,Objective,Banana,Chicken -- decoupled from
result_analysis/curriculum's own file layout so export.py never has to import
pandas or know about that scope's data/ directory.

Usage: python beyond_pdf/extract_actual_returns.py --seed 1764531329
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

CURRICULUM_DATA = Path(__file__).resolve().parent.parent / "result_analysis" / "curriculum" / "data"
OUT = Path(__file__).resolve().parent / "data" / "chickenbanana_actual_returns.csv"
COMPONENTS = ["Objective", "Banana", "Chicken"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    merged = None
    for name in COMPONENTS:
        df = pd.read_csv(CURRICULUM_DATA / f"chickenbanana_{name.lower()}_reward.csv")
        df = df[df["seed"] == args.seed][["_step", f"ep_info/{name}"]]
        df = df.rename(columns={f"ep_info/{name}": name, "_step": "episode"})
        merged = df if merged is None else merged.merge(df, on="episode")
    if merged is None or merged.empty:
        sys.exit(f"no rows for seed {args.seed} in {CURRICULUM_DATA}")

    merged = merged.sort_values("episode")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(merged)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
