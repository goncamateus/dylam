"""
Fetch Pareto front data from wandb, compute HV/cardinality per run,
then run Mann-Whitney U tests (HV, cardinality, wall-time).

Sources
-------
goncamateus/DyLam
  MINECART : setup in {Dylam, Dynmorl}, state=finished
  HALFCHEETAH : setup=Dylam, state=finished
  Front built from ALL ep_info/* history points → filter_pareto_dominated

openrlbenchmark/MORL-Baselines
  minecart-v0 : algo=GPI-LS, state=finished
  mo-halfcheetah-v4 : algo in {GPI-LS Continuous Action, PGMORL}, state=finished
  Front read directly from last logged eval/front table
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from pymoo.indicators.hv import HV
from scipy.stats import mannwhitneyu

CACHE_DIR = Path("wandb_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALPHA = 0.05
DYLAM_HISTORY_SAMPLES = 10_000  # rows sampled per run from scan_history
MAX_RUNS_PER_GROUP = 10         # last N finished runs per env-setup tuple

GPILS_MC = "GPI-LS"                   # minecart-v0 algo key in MORL-Baselines
GPILS_HC = "GPI-LS Continuous Action"  # mo-halfcheetah-v4 algo key in MORL-Baselines

# ep_info/* keys used in goncamateus/DyLam
MINECART_METRICS = [
    "ep_info/First_minerium",
    "ep_info/Second_minerium",
    "ep_info/Fuel",
]
HALFCHEETAH_METRICS = ["ep_info/run", "ep_info/ctrl"]

# Reference points for HV
MINECART_REF = np.array([0.0, 0.0, -1000.0])
HALFCHEETAH_REF = np.array([-1, -1])

# ---------------------------------------------------------------------------
# Pareto / HV utilities
# ---------------------------------------------------------------------------


def get_non_pareto_dominated_inds(candidates, remove_duplicates=True):
    candidates = np.array(candidates)
    _, indcs, invs, counts = np.unique(
        candidates,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        axis=0,
    )
    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)
    return np.logical_and(c1, c2) & to_keep


def filter_pareto_dominated(candidates, remove_duplicates=True):
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[get_non_pareto_dominated_inds(candidates, remove_duplicates)]


def get_hv(points, ref_point):
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

api = wandb.Api(timeout=120)


def _wall_time_minutes(run):
    t = run.summary.get("_runtime")  # seconds
    return t / 60.0 if t is not None else None


# ---------------------------------------------------------------------------
# Cached fetchers  (cache key = run.id; wall_time stored as extra column)
# ---------------------------------------------------------------------------

WALL_COL = "__wall_time_min__"


def fetch_dylam_points(run, metric_keys, n_samples=DYLAM_HISTORY_SAMPLES):
    """Return (pts: (N,D) array, wall_time_min: float|None), using cache."""
    cache = CACHE_DIR / f"dylam_{run.id}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        wt = float(df[WALL_COL].iloc[0]) if WALL_COL in df.columns else None
        pts = df[metric_keys].dropna().values
        return (pts if len(pts) >= 2 else None), wt

    try:
        df = run.history(samples=n_samples, keys=metric_keys, pandas=True)
    except Exception as exc:
        print(f"    history() failed ({run.id}): {exc}")
        return None, None

    df = df[metric_keys].dropna()
    if len(df) < 2:
        return None, None

    wt = _wall_time_minutes(run)
    cache_df = df.copy()
    cache_df[WALL_COL] = wt
    cache_df.to_csv(cache, index=False)

    return df.values, wt


def fetch_eval_front(run, key="eval/front"):
    """Return (pts: (N,D) array, wall_time_min: float|None), using cache."""
    cache = CACHE_DIR / f"morl_{run.id}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        wt = float(df[WALL_COL].iloc[0]) if WALL_COL in df.columns else None
        pts = (
            df.drop(columns=[WALL_COL], errors="ignore")
            .select_dtypes(include=[np.number])
            .values.astype(float)
        )
        return (pts if len(pts) >= 2 else None), wt

    def _parse_table(last):
        if isinstance(last, dict) and "data" in last:
            cols = last.get("columns", [])
            df = pd.DataFrame(last["data"], columns=cols or None)
            return df.select_dtypes(include=[np.number])
        if isinstance(last, dict) and "path" in last:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    dl = run.file(last["path"]).download(root=tmpdir, replace=True)
                    with open(dl.name) as fp:
                        raw = json.load(fp)
                    df = pd.DataFrame(raw.get("data", []), columns=raw.get("columns"))
                    return df.select_dtypes(include=[np.number])
                except Exception as exc:
                    print(f"    Table download failed ({run.id}): {exc}")
        return None

    try:
        rows = list(run.history(keys=[key], pandas=False))
    except Exception as exc:
        print(f"    history() failed ({run.id}): {exc}")
        return None, None

    valid = [r[key] for r in rows if key in r and r[key] is not None]
    if not valid:
        return None, None

    df = _parse_table(valid[-1])
    if df is None or len(df) < 2:
        return None, None

    wt = _wall_time_minutes(run)
    cache_df = df.copy()
    cache_df[WALL_COL] = wt
    cache_df.to_csv(cache, index=False)

    return df.values.astype(float), wt


# ---------------------------------------------------------------------------
# Per-run metric extraction
# ---------------------------------------------------------------------------


def run_metrics_dylam(run, metric_keys, ref_point, transform=None):
    pts, wt = fetch_dylam_points(run, metric_keys)
    if pts is None:
        return None
    if transform is not None:
        pts = transform(pts)
    pareto = filter_pareto_dominated(pts)
    return {"hv": get_hv(pareto, ref_point), "cardinality": len(pareto), "wall_time_min": wt, "pareto_points": pareto}


def run_metrics_morl(run, ref_point, transform=None):
    pts, wt = fetch_eval_front(run)
    if pts is None or len(pts) < 2:
        return None
    if transform is not None:
        pts = transform(pts)
    pareto = filter_pareto_dominated(pts)
    return {"hv": get_hv(pareto, ref_point), "cardinality": len(pareto), "wall_time_min": wt, "pareto_points": pareto}


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------


def halfcheetah_dylam_transform(pts):
    """DyLam ep_info/ctrl: 1000 + ctrl."""
    p = pts.copy()
    p[:, 1] = 1000.0 + p[:, 1]
    return p


def halfcheetah_morl_transform(pts):
    """MORL-Baselines eval/front obj2: 1000 + obj2 * 10."""
    p = pts.copy()
    p[:, 1] = 1000.0 + p[:, 1] * 10
    return p


def collect_dylam_runs(env, setups, metric_keys, ref_point, transform=None):
    results = {s: [] for s in setups}
    for setup in setups:
        filters = {"config.env": env, "config.setup": setup, "state": "finished"}
        runs = api.runs("goncamateus/DyLam", filters=filters, order="-created_at")
        count = 0
        for run in runs:
            if count >= MAX_RUNS_PER_GROUP:
                break
            print(f"  [{setup}] {run.id} ...", end=" ", flush=True)
            m = run_metrics_dylam(run, metric_keys, ref_point, transform)
            if m is None:
                print("skip (no data)")
                continue
            results[setup].append(m)
            count += 1
            print(f"HV={m['hv']:.4f}  card={m['cardinality']}  t={m['wall_time_min']:.1f}min")
    return results


def collect_morl_runs(env_id, algos, ref_point, transform=None):
    results = {a: [] for a in algos}
    for algo in algos:
        filters = {"config.env_id": env_id, "config.algo": algo, "state": "finished"}
        runs = api.runs("openrlbenchmark/MORL-Baselines", filters=filters, order="-created_at")
        count = 0
        for run in runs:
            if count >= MAX_RUNS_PER_GROUP:
                break
            print(f"  [{algo}] {run.id} ...", end=" ", flush=True)
            m = run_metrics_morl(run, ref_point, transform)
            if m is None:
                print("skip (no eval/front)")
                continue
            results[algo].append(m)
            count += 1
            print(f"HV={m['hv']:.4f}  card={m['cardinality']}  t={m['wall_time_min']:.1f}min")
    return results


# ---------------------------------------------------------------------------
# Run all collections
# ---------------------------------------------------------------------------

print("=== goncamateus/DyLam — MINECART ===")
minecart_dylam = collect_dylam_runs(
    "MINECART", ["Dylam", "Dynmorl"], MINECART_METRICS, MINECART_REF
)

print("\n=== goncamateus/DyLam — HALFCHEETAH ===")
halfcheetah_dylam = collect_dylam_runs(
    "HALFCHEETAH", ["Dylam"], HALFCHEETAH_METRICS, HALFCHEETAH_REF,
    transform=halfcheetah_dylam_transform,
)

print("\n=== openrlbenchmark/MORL-Baselines — minecart-v0 ===")
minecart_morl = collect_morl_runs("minecart-v0", [GPILS_MC], MINECART_REF)

print("\n=== openrlbenchmark/MORL-Baselines — mo-halfcheetah-v4 ===")
halfcheetah_morl = collect_morl_runs(
    "mo-halfcheetah-v4", [GPILS_HC, "PGMORL"], HALFCHEETAH_REF,
    transform=halfcheetah_morl_transform,
)

# ---------------------------------------------------------------------------
# Mann-Whitney U tests
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "hv": "HV",
    "cardinality": "Cardinality",
    "wall_time_min": "Wall-time (min)",
}


def mw_test(env, label_a, data_a, label_b, data_b):
    rows = []
    for metric, label in METRIC_LABELS.items():
        vals_a = [d[metric] for d in data_a if d.get(metric) is not None]
        vals_b = [d[metric] for d in data_b if d.get(metric) is not None]
        if len(vals_a) < 2 or len(vals_b) < 2:
            print(
                f"  {label_a} vs {label_b} [{label}]: "
                f"not enough samples ({len(vals_a)} vs {len(vals_b)})"
            )
            continue
        stat, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        med_a, med_b = np.median(vals_a), np.median(vals_b)
        direction = ">" if med_a > med_b else "<"
        sig = "  *" if p < ALPHA else ""
        print(
            f"  {label_a} {direction} {label_b} [{label}]  "
            f"n=({len(vals_a)},{len(vals_b)})  "
            f"med=({med_a:.3f},{med_b:.3f})  "
            f"U={stat:.0f}  p={p:.4f}{sig}"
        )
        rows.append(
            {
                "env": env,
                "method_a": label_a,
                "method_b": label_b,
                "metric": label,
                "n_a": len(vals_a),
                "n_b": len(vals_b),
                "median_a": round(med_a, 4),
                "median_b": round(med_b, 4),
                "U": stat,
                "p": round(p, 6),
                "significant": p < ALPHA,
            }
        )
    return rows


print("\n" + "=" * 70)
print(f"Mann-Whitney U tests  (two-sided, alpha={ALPHA})")
print("=" * 70)

all_rows = []

print("\n--- MINECART ---")
all_rows += mw_test("MINECART", "DyLam", minecart_dylam["Dylam"], "GPI-LS",  minecart_morl[GPILS_MC])
all_rows += mw_test("MINECART", "DyLam", minecart_dylam["Dylam"], "DynMORL", minecart_dylam["Dynmorl"])

print("\n--- HALFCHEETAH ---")
all_rows += mw_test("HALFCHEETAH", "DyLam", halfcheetah_dylam["Dylam"], "GPI-LS", halfcheetah_morl[GPILS_HC])
all_rows += mw_test("HALFCHEETAH", "DyLam", halfcheetah_dylam["Dylam"], "PGMORL", halfcheetah_morl["PGMORL"])

# ---------------------------------------------------------------------------
# Summary (CSV)
# ---------------------------------------------------------------------------

summary = pd.DataFrame(all_rows)
print("\n\nSummary table:")
print(summary.to_string(index=False))
summary.to_csv("pareto_mw_results.csv", index=False)
print("\nSaved -> result_analysis/pareto_mw_results.csv")

# ---------------------------------------------------------------------------
# LaTeX: single combined table (results + MW p-values)
# ---------------------------------------------------------------------------

METHOD_DATA = {
    "PGMORL":  {"halfcheetah": halfcheetah_morl.get("PGMORL", []),  "minecart": []},
    "GPI-LS":  {"halfcheetah": halfcheetah_morl.get(GPILS_HC, []),  "minecart": minecart_morl.get(GPILS_MC, [])},
    "DynMORL": {"halfcheetah": [],                                   "minecart": minecart_dylam.get("Dynmorl", [])},
    "DyLam":   {"halfcheetah": halfcheetah_dylam.get("Dylam", []),  "minecart": minecart_dylam.get("Dylam", [])},
}
METHOD_ORDER = ["PGMORL", "GPI-LS", "DynMORL", "DyLam"]

# MW rows: DyLam (anchor) vs each baseline
MW_ROWS = [
    ("DyLam", "GPI-LS"),   # both envs
    ("DyLam", "PGMORL"),   # halfcheetah only
    ("DyLam", "DynMORL"),  # minecart only
]

# Ordered to match the 6 table columns (3 HC + 3 MC)
COL_SPEC = [
    ("hv",           "HALFCHEETAH", "HV"),
    ("cardinality",  "HALFCHEETAH", "Cardinality"),
    ("wall_time_min","HALFCHEETAH", "Wall-time (min)"),
    ("hv",           "MINECART",    "HV"),
    ("cardinality",  "MINECART",    "Cardinality"),
    ("wall_time_min","MINECART",    "Wall-time (min)"),
]

mw_index = {
    (r["env"], r["method_a"], r["method_b"], r["metric"]): r
    for r in all_rows
}


def _stats(data, metric):
    """Return (mean, std) with log10 applied to HV; (None, None) if no data."""
    vals = [d[metric] for d in data if d.get(metric) is not None]
    if not vals:
        return None, None
    if metric == "hv":
        vals = [math.log10(v) for v in vals]
    return float(np.mean(vals)), float(np.std(vals))


def _best(col_dict, maximize=True):
    valid = {k: v for k, v in col_dict.items() if v is not None}
    if not valid:
        return None
    return max(valid, key=valid.get) if maximize else min(valid, key=valid.get)


def _val_cell(mean, std, best_method, this_method, fmt):
    if mean is None:
        return "---"
    s = f"{mean:{fmt}} $\\pm$ {std:{fmt}}"
    return f"\\textbf{{{s}}}" if this_method == best_method else s


def _p_cell(env_key, a, b, metric_label):
    r = mw_index.get((env_key, a, b, metric_label))
    if r is None:
        return "---"
    sig = r"$^{\ast}$" if r["significant"] else r"n.s."
    return f"{r['p']:.4f}~{sig}"


# Compute per-method (mean, std) for each (metric, env)
col = {
    (metric, env): {m: _stats(METHOD_DATA[m][env.lower()], metric) for m in METHOD_ORDER}
    for metric, env, _ in COL_SPEC
}

# Bold winner determined by mean only
MAXIMIZE = {"hv": True, "cardinality": True, "wall_time_min": False}
best = {
    (metric, env): _best(
        {m: v[0] for m, v in col[(metric, env)].items()},
        maximize=MAXIMIZE[metric],
    )
    for metric, env, _ in COL_SPEC
}

FMTS = {"hv": ".3f", "cardinality": ".0f", "wall_time_min": ".0f"}

print(r"""
% ---- Combined results + MW significance table ----
\begin{table}[ht]
\centering
\caption{Hypervolume ($\log_{10}$), cardinality, and wall-clock training time
  (minutes) of the approximated Pareto fronts on the two Pareto-oriented
  benchmarks. Best per column in \textbf{bold} (for time, lowest is best).
  All times measured on a single NVIDIA RTX~3060 (16 CPU cores, 16~GB RAM).
  PGMORL is restricted to continuous action spaces and is not applicable to
  Minecart; DynMORL was originally introduced for Minecart and is reported
  there as the primary methodological comparator.
  Lower block: two-sided Mann--Whitney $U$ $p$-values ($\alpha = 0.05$);
  $^{\ast}$ significant, n.s.\ not significant.}
\label{tab:res/pareto/hv-cardinality}
\resizebox{\columnwidth}{!}{%
\small
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{\textbf{MO-HalfCheetah}}
 & \multicolumn{3}{c}{\textbf{MO-Minecart}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Method}
  & \textbf{HV ($\log_{10}$)} & \textbf{Card.} & \textbf{Time (min)}
  & \textbf{HV ($\log_{10}$)} & \textbf{Card.} & \textbf{Time (min)} \\
\midrule""")

for m in METHOD_ORDER:
    cells = [
        _val_cell(*col[(metric, env)][m], best[(metric, env)], m, FMTS[metric])
        for metric, env, _ in COL_SPEC
    ]
    print(f"{m}  & " + " & ".join(cells) + r" \\")

print(r"\midrule")
print(
    r"\multicolumn{7}{l}{\textit{Mann--Whitney $U$ $p$-values"
    r" (two-sided, $\alpha = 0.05$)}} \\"
)
print(r"\midrule")

for a_method, b_method in MW_ROWS:
    label = f"\\textit{{{a_method}}} vs.\\ {b_method}"
    cells = [
        _p_cell(env_key, a_method, b_method, metric_label)
        for _, env_key, metric_label in COL_SPEC
    ]
    print(f"{label}  & " + " & ".join(cells) + r" \\")

print(r"""\bottomrule
\end{tabular}%
}
\end{table}""")

# ---------------------------------------------------------------------------
# Merged Pareto fronts  (all runs per method-env pooled → re-filtered)
# ---------------------------------------------------------------------------

MERGED_SOURCES = {
    ("DyLam",   "halfcheetah"): halfcheetah_dylam.get("Dylam",   []),
    ("DynMORL", "minecart"):    minecart_dylam.get("Dynmorl",    []),
    ("DyLam",   "minecart"):    minecart_dylam.get("Dylam",      []),
    ("GPI-LS",  "halfcheetah"): halfcheetah_morl.get(GPILS_HC,  []),
    ("PGMORL",  "halfcheetah"): halfcheetah_morl.get("PGMORL",  []),
    ("GPI-LS",  "minecart"):    minecart_morl.get(GPILS_MC,     []),
}

print("\n\n=== Merged Pareto fronts ===")
merged_fronts = {}
for (method, env), runs_data in MERGED_SOURCES.items():
    chunks = [d["pareto_points"] for d in runs_data if d.get("pareto_points") is not None]
    if not chunks:
        print(f"  [{method} / {env}] no data")
        continue
    all_pts = np.vstack(chunks)
    merged = filter_pareto_dominated(all_pts)
    merged_fronts[(method, env)] = merged
    print(f"  [{method} / {env}] {len(all_pts)} total pts → {len(merged)} Pareto pts")

    out_dir = Path("merged_fronts")
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"{method.lower().replace('-', '_')}_{env}.csv"
    pd.DataFrame(merged).to_csv(fname, index=False, header=False)
    print(f"    saved → {fname}")
