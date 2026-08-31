"""Beyond PDF export: builds a single self-contained HTML file -- the
per-environment curve explorer Embed (issue #49) -- from the trad and
curriculum scopes' committed Tidy CSVs. See issue #39's "Per-environment
curve explorer" spec: selectors over environment and metric, replacing the
appendix figures a reader currently has to page between to compare DyLam
across environments.

This is the shared curves template from #48 with different data -- a second
data blob, not a second template.

Reads only the committed CSVs under result_analysis/{trad,curriculum}/data/;
never touches the network. Lines follow the same aggregation protocol as
the owning scopes' own figures:

  - trad: each method is the IQM + 95% bootstrap CI across seeds, using the
    trad scope's figure.py re-implementation here verbatim from
    core.style/core.stats -- the same numbers fig:res/all used to show.

  - curriculum (DyLam per-Component and lambda): the mean across seeds of
    each seed's smoothed curve, normalized per component by
    max(|r_max|, |r_min|) with the r_max dashed reference line -- the same
    convention curriculum/figure.py uses, with COMPONENT_PALETTE positional
    component colours.

Usage:
  python beyond_pdf/env_export.py --out PATH [--grid-points 200]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "result_analysis"))
sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "curriculum"))

from core import stats, style  # noqa: E402
from sources import ENVS as CURRICULUM_ENVS  # noqa: E402

TEMPLATE = Path(__file__).parent / "curves_template.html"
DATA_TOKEN = "/*__BEYOND_PDF_DATA__*/null"
SIZE_CAP_BYTES = 5 * 1024 * 1024
TRAD_DATA = REPO_ROOT / "result_analysis" / "trad" / "data"
CURRICULUM_DATA = REPO_ROOT / "result_analysis" / "curriculum" / "data"

GRID = 200

# (key, label, file, metric, xlabel, ylabel, methods) -- the trad learning
# curves of fig:res/all plus the per-environment appendix figures this
# Embed replaces. Method colors are core.style.METHOD_COLORS, keyed by the
# paper method label, exactly as the trad scope's own figure.py does.
TRAD_CURVES = [
    (
        "chickenbanana",
        "chickenbanana",
        "Chicken-Banana",
        "chicken_banana.csv",
        "ep_info/total",
        "Episode",
        "Cumulative episode reward",
        ["Base SO RL", "Q-Decomposition", "UDC", "DyLam"],
    ),
    (
        "halfcheetah",
        "halfcheetah",
        "HalfCheetah",
        "halfcheetah_v4.csv",
        "ep_info/Final_position",
        "Environment step",
        "Final x-position (m)",
        ["Base SO RL", "UDC", "DyLam"],
    ),
    (
        "halfcheetah_env",
        "halfcheetah",
        "HalfCheetah (env return)",
        "halfcheetah_v4_env_return.csv",
        "ep_info/total",
        "Environment step",
        "Environment scalar reward",
        ["Base SO RL", "UDC", "DyLam"],
    ),
    (
        "vss",
        "vss",
        "VSS",
        "vss_v0.csv",
        "ep_info/Goal",
        "Environment step",
        "Goal rate",
        ["Base SO RL", "UDC", "DyLam", "Tuned-UDC"],
    ),
]

# Curriculum panels: DyLam's per-Component returns (normalized by
# max(|r_max|, |r_min|), r_max dashed) and lambda weights, per environment --
# the curriculum scope's fig:curr/components and fig:curr/weights data.
CURRICULUM_CURVES = [
    ("chickenbanana", "Chicken-Banana", "CHICKENBANANA"),
    ("halfcheetah", "HalfCheetah", "HALFCHEETAH"),
    ("vss", "VSS", "VSS"),
]


def _hex(color):
    """css color for a matplotlib color name used by core.style."""
    named = {
        "tab:blue": "#1f77b4",
        "tab:orange": "#ff7f0e",
        "tab:green": "#2ca02c",
        "tab:red": "#d62728",
        "tab:purple": "#9467bd",
        "tab:brown": "#8c564b",
    }
    return named.get(color, color)


def _trad_band(df, metric, methods, grid_points):
    series_by_method = {}
    for method in methods:
        d = df[df["method"] == method]
        series_by_method[method] = [
            (g["_step"].to_numpy(dtype=float), g[metric].to_numpy(dtype=float))
            for _, g in d.groupby("seed", sort=False)
        ]
    grid_hi = min(
        step.max() for series in series_by_method.values() for step, _ in series
    )
    grid = np.linspace(0, grid_hi, grid_points)
    lines = []
    for method in methods:
        mat = np.array(
            [
                np.interp(grid, step, style.rolling_smooth(value))
                for step, value in series_by_method[method]
            ]
        )
        centre, lo, hi = stats.bootstrap_curve(mat)
        lines.append(
            {
                "label": method,
                "color": _hex(style.METHOD_COLORS[method]),
                "centre": [round(float(v), 6) for v in centre],
                "lo": [round(float(v), 6) for v in lo],
                "hi": [round(float(v), 6) for v in hi],
            }
        )
    return {"grid": [round(float(v), 3) for v in grid], "lines": lines}


def _grid_mean(df, metric, grid_points):
    series = [
        (g["_step"].to_numpy(dtype=float), g[metric].to_numpy(dtype=float))
        for _, g in df.groupby("seed", sort=False)
    ]
    hi = min(step.max() for step, _ in series)
    grid = np.linspace(0, hi, grid_points)
    mat = np.array(
        [np.interp(grid, step, style.rolling_smooth(value)) for step, value in series]
    )
    return grid, mat.mean(axis=0)


def _curriculum_panels(env_key, label, env_name, grid_points, curriculum_data=None):
    """One return panel + one weight panel for a curriculum environment.

    Returns panel, weights: (panel, lambda weights panel)."""
    spec = CURRICULUM_ENVS[env_name]
    curriculum_data = curriculum_data or CURRICULUM_DATA
    return_panel_lines, weight_panel_lines, refs = [], [], []
    for i, comp in enumerate(spec.components):
        color = style.COMPONENT_PALETTE[i]
        reward_df = pd.read_csv(
            curriculum_data / f"{env_name.lower()}_{comp.name.lower()}_reward.csv"
        )
        grid, mean = _grid_mean(reward_df, comp.ep_metric, grid_points)
        abs_max = max(abs(comp.r_max), abs(comp.r_min))
        return_panel_lines.append(
            {
                "label": comp.label,
                "color": _hex(color),
                "centre": [round(float(v), 6) for v in mean / abs_max],
                "lo": [round(float(v), 6) for v in mean / abs_max],
                "hi": [round(float(v), 6) for v in mean / abs_max],
            }
        )
        refs.append({"y": comp.r_max / abs_max, "color": _hex(color)})
        lambda_df = pd.read_csv(
            curriculum_data / f"{env_name.lower()}_{comp.name.lower()}_lambda.csv"
        )
        _, lam_mean = _grid_mean(lambda_df, f"lambdas/{comp.name}", grid_points)
        weight_panel_lines.append(
            {
                "label": comp.label,
                "color": _hex(color),
                "centre": [round(float(v), 6) for v in lam_mean],
                "lo": [round(float(v), 6) for v in lam_mean],
                "hi": [round(float(v), 6) for v in lam_mean],
            }
        )
    return_panel = {
        "key": f"{env_key}|Cumulative episode reward (normalized)",
        "label": f"{label} · Per-Component return (normalized)",
        "grid": [round(float(v), 3) for v in grid],
        "lines": return_panel_lines,
        "refs": refs,
    }
    weight_panel = {
        "key": f"{env_key}|λ weight",
        "label": f"{label} · λ weight",
        "grid": [round(float(v), 3) for v in grid],
        "lines": weight_panel_lines,
    }
    return return_panel, weight_panel


def build(out_path, grid_points=GRID, data_root=None):
    # --data-root points at a directory holding trad/data and
    # curriculum/data (default: the real result_analysis/). The offline
    # Embed test drives this generator through synthetic CSVs laid out the
    # same way, keeping the fresh-clone guarantee.
    trad_data = Path(data_root) / "trad" / "data" if data_root else TRAD_DATA
    curriculum_data = (
        Path(data_root) / "curriculum" / "data" if data_root else CURRICULUM_DATA
    )
    meta_panels = {}
    env_group = {}

    for env_key, _, label, fname, metric, _, ylabel, methods in TRAD_CURVES:
        df = pd.read_csv(trad_data / fname)
        band = _trad_band(df, metric, methods, grid_points)
        meta_panels[f"{env_key}|{ylabel}"] = {"label": f"{label} · {ylabel}", **band}
        env_group.setdefault(env_key, []).append(f"{env_key}|{ylabel}")

    for env_key, label, env_name in CURRICULUM_CURVES:
        return_panel, weight_panel = _curriculum_panels(
            env_key, label, env_name, grid_points, curriculum_data
        )
        meta_panels[return_panel["key"]] = return_panel
        meta_panels[weight_panel["key"]] = weight_panel
        env_group.setdefault(env_key, []).extend(
            [return_panel["key"], weight_panel["key"]]
        )

    # (metric key as stored, display label shown in the Metric selector).
    # The curriculum panels' keys spell their group name
    # "Cumulative episode reward (normalized)"; the display label says what
    # the reader is looking at -- DyLam's per-Component returns.
    METRIC_DISPLAY = [
        ("Cumulative episode reward", "Cumulative episode reward"),
        ("Final x-position (m)", "Final x-position (m)"),
        ("Environment scalar reward", "Environment scalar reward"),
        ("Goal rate", "Goal rate"),
        ("Cumulative episode reward (normalized)", "Per-Component return (normalized)"),
        ("λ weight", "λ weight"),
    ]

    def _panels_for(env, stored):
        """Panel keys under `env` whose group segment matches `stored`."""
        return [pk for pk in env_group.get(env, []) if pk.split("|", 1)[1] == stored]

    data = {
        "selectors": [
            {
                "key": "env",
                "label": "Environment",
                "options": [
                    {"key": "chickenbanana", "label": "Chicken-Banana"},
                    {"key": "halfcheetah", "label": "HalfCheetah"},
                    {"key": "vss", "label": "VSS"},
                ],
            },
            {
                "key": "metric",
                "label": "Metric",
                "options": [
                    {"key": stored, "label": label} for stored, label in METRIC_DISPLAY
                ],
            },
        ],
        # env group: the panel keys this env offers across metrics; metric
        # group: which env panels carry that metric. Their intersection
        # (via the template's innermost `panels` map) resolves to the
        # single panel matching both selectors.
        "groups": {
            "env": env_group,
            "metric": {
                stored: [pk for env2 in env_group for pk in _panels_for(env2, stored)]
                for stored, _ in METRIC_DISPLAY
            },
        },
        # panels[envKey][metricKey] = the single panel key shown; an env
        # that has no panel for a metric is simply absent from that
        # metric's mapping (the metric selector's first reachable option
        # per env always exists, so the initial render is never empty).
        "panels": {
            env: {
                stored: _panels_for(env, stored)[0]
                for stored, _ in METRIC_DISPLAY
                if _panels_for(env, stored)
            }
            for env in env_group
        },
        "meta": {"panels": meta_panels},
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
    n_panels = len(meta_panels)
    print(
        f"wrote {out_path} ({size} bytes, {len(env_group)} environments, "
        f"{n_panels} panels)",
        file=sys.stderr,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="output HTML path")
    ap.add_argument("--grid-points", type=int, default=GRID)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="directory holding trad/data and curriculum/data "
        "(default: the repository's result_analysis/)",
    )
    args = ap.parse_args()
    build(args.out, args.grid_points, args.data_root)


if __name__ == "__main__":
    main()
