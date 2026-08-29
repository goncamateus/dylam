"""R5 open-loop replay: DyLam vs its own lambda(t) schedule on VSS-v0.

Everything else this script used to compute -- Table 1, the RQ1
significance family, sample efficiency, Chicken-Banana success rate, and
the RQ3 robustness table -- now lives in trad/table.py and
robustness/table.py. This is the one piece not yet migrated; it will
retire into ablation/openloop's own generator when that scope migrates.

Usage: python stats_r1.py [--refresh]
"""
import argparse

import numpy as np
from lib import fetch, stats


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
