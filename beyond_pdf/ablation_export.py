"""Beyond PDF export: builds a single self-contained HTML file -- the
ablation explorer Embed (issue #48) -- from the ablation scope's committed
Tidy CSVs. See issue #39's "Ablation explorer" spec: selectors over sweep
(tau, replay buffer E, deficiency-transform normalizer, exploration decay)
and over panel (per-Component reward or lambda-weight), driving a grid of
line panels -- one per Component -- for the selected (sweep, panel) pair.
This is the "shared curves template" issue #48 establishes: a categorical
selector driving a grid of line panels from Tidy CSVs, generic enough for
a later per-environment explorer to reuse with different data.

Reads only the committed CSVs under result_analysis/ablation/data/; never
touches the network. Each line is the same IQM + 95% bootstrap CI protocol
as the ablation scope's own figure.py (core.style.rolling_smooth +
core.stats.bootstrap_curve), so the Embed's curves are the same numbers the
static combined figures used to show, just reachable individually instead
of pre-composed into four fixed grids.

24 total panels are reachable: 4 sweeps x 2 kinds (reward, lambda) x 3
Components (Banana, Chicken, Gate) -- matching issue #39's count exactly.
The third ("isolated run") row some of the static figures carried is not
reproduced here: it is a single raw per-seed trace, not an IQM/CI band, and
issue #39 counts only the 24 aggregate panels as what the PDF hides.

Usage:
  python beyond_pdf/ablation_export.py --data-dir DIR --out PATH [--grid-points 200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "result_analysis"))
sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "ablation"))

from arms import (COMPONENTS, EPSILON_ARMS, NORMALIZER_ARMS,  # noqa: E402
                  RB_ARMS, TAU_ARMS)
from core import stats, style  # noqa: E402
# Reuses the ablation scope's own CSV reader instead of re-typing it.
from figure import _read, _series  # noqa: E402

TEMPLATE = Path(__file__).parent / "curves_template.html"
DATA_TOKEN = "/*__BEYOND_PDF_DATA__*/null"
SIZE_CAP_BYTES = 5 * 1024 * 1024
DEFAULT_DATA_DIR = REPO_ROOT / "result_analysis" / "ablation" / "data"

# (sweep key, display label, arms) -- the four sweeps of sections/ablation.tex.
SWEEPS = [
    ("tau", "Update rate (τλ)", TAU_ARMS),
    ("rb", "Replay buffer (E)", RB_ARMS),
    ("normalizer", "Deficiency transform", NORMALIZER_ARMS),
    ("epsilon", "Exploration decay (ε)", EPSILON_ARMS),
]
KINDS = [("reward", "Cumulative episode reward"), ("lambda", "λ weight")]
ARM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
NOMINAL_COLOR = "#7f7f7f"


def _band(data_dir, sweep, comp, kind, arms, grid_points):
    nominal = _read(data_dir / f"nominal_{comp.lower()}_{kind}.csv")
    swept = _read(data_dir / f"{sweep}_{comp.lower()}_{kind}.csv")
    series = {"nominal": _series(nominal, "nominal")}
    series.update({a.label: _series(swept, a.label) for a in arms})

    grid_hi = min(step.max() for s in series.values() for step, _ in s)
    grid = np.linspace(0, grid_hi, grid_points)

    lines = []
    for label, color in [("nominal", NOMINAL_COLOR)] + list(zip((a.label for a in arms), ARM_COLORS)):
        mat = np.array([np.interp(grid, step, style.rolling_smooth(value))
                        for step, value in series[label]])
        centre, lo, hi = stats.bootstrap_curve(mat)
        lines.append({
            "label": label, "color": color,
            "centre": [round(float(v), 6) for v in centre],
            "lo": [round(float(v), 6) for v in lo],
            "hi": [round(float(v), 6) for v in hi],
        })
    return {"grid": [round(float(v), 3) for v in grid], "lines": lines}


def build(data_dir, out_path, grid_points=200):
    data_dir = Path(data_dir)
    panels = {}
    for sweep, sweep_label, arms in SWEEPS:
        panels[sweep] = {}
        for kind, kind_label in KINDS:
            panels[sweep][kind] = {
                comp: _band(data_dir, sweep, comp, kind, arms, grid_points)
                for comp in COMPONENTS
            }

    data = {
        "sweeps": [{"key": s, "label": label} for s, label, _ in SWEEPS],
        "kinds": [{"key": k, "label": label} for k, label in KINDS],
        "components": COMPONENTS,
        "panels": panels,
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if DATA_TOKEN not in template:
        sys.exit(f"template {TEMPLATE} is missing the data token {DATA_TOKEN!r}")
    html = template.replace(DATA_TOKEN, json.dumps(data, separators=(",", ":")))

    size = len(html.encode("utf-8"))
    if size > SIZE_CAP_BYTES:
        sys.exit(
            f"built artifact is {size} bytes, over the {SIZE_CAP_BYTES}-byte cap. "
            "Coarsen --grid-points to reduce per-line point counts."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({size} bytes, {len(SWEEPS)} sweeps x {len(KINDS)} kinds x "
          f"{len(COMPONENTS)} components = "
          f"{len(SWEEPS) * len(KINDS) * len(COMPONENTS)} panels)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                     help="result_analysis/ablation/data (default: %(default)s)")
    ap.add_argument("--out", type=Path, required=True, help="output HTML path")
    ap.add_argument("--grid-points", type=int, default=200)
    args = ap.parse_args()
    build(args.data_dir, args.out, args.grid_points)


if __name__ == "__main__":
    main()
