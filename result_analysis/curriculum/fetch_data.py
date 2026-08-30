"""Fetch curriculum scope data: per-seed component-return and lambda-weight
histories for the three environments in fig:curr/weights and
fig:curr/components. DyLam only -- curriculum is about one policy's own
adaptive trajectory, not a method comparison.

One file per (environment, component, quantity) -- reward or lambda --
matching the tidy-CSV granularity every other scope uses (CONTEXT.md's
Tidy CSV entry), even though the figures overlay all of an env's
components on one axes: figure.py loads each file separately and aligns
them on a shared step grid, the same pattern trad/figure.py's
per-component panels already use.

Fetch step, not a generator: touches the network, not covered by the
offline test seam (curriculum has no generated table, so no test seam
at all -- see the issue's testing decisions, "six generated tables
across four scopes").

Usage: python fetch_data.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources import ENVS, MAX_SEEDS

from core import fetch

DATA = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="bypass the run-history cache")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    for env, spec in ENVS.items():
        for comp in spec.components:
            for kind, metric in [("reward", comp.ep_metric), ("lambda", f"lambdas/{comp.name}")]:
                print(f"  {env:14s} {comp.name:8s} {kind:6s} ...", file=sys.stderr, end=" ")
                dfs = fetch.histories(env, spec.setup, metric, max_seeds=MAX_SEEDS,
                                      refresh=args.refresh)
                print(f"n={len(dfs)}", file=sys.stderr)
                if not dfs:
                    sys.exit(f"no data for {env}/{comp.name}/{kind}")
                out = DATA / f"{env.lower()}_{comp.name.lower()}_{kind}.csv"
                fetch.tidy(dfs, metric, "DyLam", column="method").to_csv(out, index=False)
                print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
