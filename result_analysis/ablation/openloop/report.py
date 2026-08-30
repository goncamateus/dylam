"""R5 open-loop replay: DyLam vs its own lambda(t) schedule on VSS-v0.

In-flight work with no paper artifact yet (no fig:/tab: label references
it anywhere in the manuscript) -- given its own scope directory per user
story 30 ("I want the in-flight open-loop ablation given a scope
directory now, so that it does not become the next orphan") rather than
sitting orphaned at the top level, since -- per the issue's own solution
section -- "a script with no home is the mechanism that produced the
current state." Everything else this script's predecessor (stats_r1.py)
used to compute -- Table 1, the RQ1 significance family, sample
efficiency, Chicken-Banana success rate, and the RQ3 robustness table --
already lives in trad/table.py and robustness/table.py.

Not a generator: it prints a stats report to the console for the author
to read, not a LaTeX fragment or figure -- there is nothing to emit yet
because there is no published number to reproduce. It will grow into a
real generator (with its own data/ and, once R5 has a published table,
a test) once the manuscript actually cites this result.

Usage: python report.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from core import fetch, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    nom = [stats.seed_summary(h, "ep_info/Goal")
           for h in fetch.histories("VSS", "Dylam", "ep_info/Goal", refresh=args.refresh)]
    ol = [stats.seed_summary(h, "ep_info/Goal")
          for h in fetch.histories("VSS", "Dylam_Openloop", "ep_info/Goal", refresh=args.refresh)]

    print("\n%%%% R5 OPEN-LOOP REPLAY: DyLam vs its own lambda(t) schedule on VSS-v0")
    if not ol:
        print("%   no Dylam_Openloop runs yet (scripts/run_r1_vss.sh openloop)")
        return
    lo, hi = stats.boot_ci(ol)
    u, p, r = stats.exact_mw(nom, ol)
    print(f"%   Open-loop n={len(ol):2d} mean {np.mean(ol):.3f} +- {np.std(ol, ddof=1):.3f}"
          f"  IQM {stats.iqm(ol):.3f} CI [{lo:.3f}, {hi:.3f}]")
    print(f"%   DyLam vs open-loop  U={u:5.1f}  p_exact={p:.4g}  r={r:+.3f}")


if __name__ == "__main__":
    main()
