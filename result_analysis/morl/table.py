"""Generator for Table~\\ref{tab:res/pareto/hv-cardinality} (hypervolume,
cardinality, wall-clock time on MO-HalfCheetah and MO-Minecart;
sections/results/morl/app.tex, app:res_morl_metrics) and the pairwise
Mann-Whitney tests quoted in prose in the same appendix
(app:res_morl_tests) and in sections/results/morl.tex (sec:res_morl).

Reads only the committed tidy CSVs under data/; never touches the
network. Per-run HV and cardinality are computed by Pareto-filtering that
run's own candidate set (a 10^4-point sample of DyLam/DynMORL's training
history, or GPI-LS/PGMORL's already-approximate eval/front) against the
env's reference point (core.pareto). DyLam is compared against every
applicable rival with an exact two-sided Mann-Whitney U; each of the
three metrics (HV, cardinality, wall-time) is its own Holm-Bonferroni
family of the (up to) four comparisons it has across both environments,
matching the appendix's own family definition.

Emits the tabular environment only (no \\begin{table}, caption, or label
-- those stay authored prose in the manuscript).

--format html emits the same means and Mann-Whitney rows as one semantic
HTML <table> fragment for the Beyond-PDF submission (issue #42) -- built
from the same `data`/`tests` this module already computes for the LaTeX
side, never hand-ported.

Usage: python table.py [--out-path PATH] [--format {latex,html,both}]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sources import ENVS, HALFCHEETAH_REF, METHOD_ORDER, MINECART_REF, per_seed

from core import pareto, stats
from core.html import strong, table as html_table

DEFAULT_OUT = Path.home() / "doc/DyLam-TMLR"
FRAGMENT = Path("tables/morl/hv_cardinality.tex")
HTML_FRAGMENT = FRAGMENT.with_suffix(".html")
ALPHA = 0.05

REF_BY_ENV = {"HALFCHEETAH": HALFCHEETAH_REF, "MINECART": MINECART_REF}
METRICS = ["hv", "cardinality", "wall_time_min"]
MAXIMIZE = {"hv": True, "cardinality": True, "wall_time_min": False}
FMT = {"hv": ".3f", "cardinality": ".0f", "wall_time_min": ".0f"}
COL_LABEL = {"hv": "HV ($\\log_{10}$)", "cardinality": "Card.", "wall_time_min": "Time (min)"}
COL_LABEL_PLAIN = {"hv": "HV (log10)", "cardinality": "Card.", "wall_time_min": "Time (min)"}


def per_seed_metrics(env, label, source):
    ref = np.asarray(REF_BY_ENV[env])
    rows = []
    for seed, g, front in per_seed(env, source):
        rows.append(dict(hv=pareto.hypervolume(front, ref), cardinality=len(front),
                         wall_time_min=float(g["wall_time_min"].iloc[0])))
    return rows


def compute():
    data = {}  # (env, method) -> {metric: [per-seed values]}
    for env, sources in ENVS.items():
        for source in sources:
            rows = per_seed_metrics(env, source.label, source)
            data[(env, source.label)] = {m: [r[m] for r in rows] for m in METRICS}
    return data


def family_tests(data):
    """DyLam vs. every applicable rival, one Holm family per metric across both envs."""
    tests = {}  # metric -> {(env, rival): (u, p, r)}
    for metric in METRICS:
        entries = []
        for env, sources in ENVS.items():
            rivals = [s.label for s in sources if s.label != "DyLam"]
            dylam = data.get((env, "DyLam"), {}).get(metric)
            if not dylam:
                continue
            for rival in rivals:
                vals = data.get((env, rival), {}).get(metric)
                if not vals:
                    continue
                entries.append((env, rival, stats.exact_mw(dylam, vals)))
        adj = stats.holm([e[2][1] for e in entries])
        tests[metric] = {(env, rival): (u, p, a) for (env, rival, (u, p, r)), a
                         in zip(entries, adj)}
    return tests


def _shown(vals, m):
    return [math.log10(v) for v in vals] if m == "hv" else vals


def _best_per_cell(data):
    """(env, metric) -> the method with the best displayed (log10-for-HV)
    mean. Shared by both renderers so "best" always ranks the same quantity
    the cell displays -- ranking raw HV would let one high-variance outlier
    seed flip the bold marker onto a method whose displayed (log-mean) value
    is lower."""
    means = {(env, m): {method: (np.mean(_shown(v, m))
                                 if (v := data.get((env, method), {}).get(m)) else None)
                       for method in METHOD_ORDER}
             for env in ENVS for m in METRICS}
    best = {}
    for (env, m), method_means in means.items():
        present = [(method, v) for method, v in method_means.items() if v is not None]
        if present:
            pick = max if MAXIMIZE[m] else min
            best[(env, m)] = pick(present, key=lambda kv: kv[1])[0]
    return best


def render_tex(data, tests):
    metric_header = " & ".join(f"\\textbf{{{COL_LABEL[m]}}}" for m in METRICS)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{\textbf{MO-HalfCheetah}}"
             r" & \multicolumn{3}{c}{\textbf{MO-Minecart}} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             f"\\textbf{{Method}} & {metric_header} & {metric_header} \\\\",
             r"\midrule"]

    best = _best_per_cell(data)

    for method in METHOD_ORDER:
        cells = []
        for env in ENVS:
            for m in METRICS:
                vals = data.get((env, method), {}).get(m)
                if not vals:
                    cells.append("---")
                    continue
                show = _shown(vals, m)
                mean, sd = np.mean(show), np.std(show)
                s = f"{mean:{FMT[m]}} $\\pm$ {sd:{FMT[m]}}"
                cells.append(f"\\textbf{{{s}}}" if best.get((env, m)) == method else s)
        lines.append(f"{method}  & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{Mann--Whitney $U$ $p$-values "
                 r"(two-sided, Holm within each metric, $\alpha = 0.05$)}} \\")
    lines.append(r"\midrule")
    for rival in [m for m in METHOD_ORDER if m != "DyLam"]:
        cells = []
        for env in ENVS:
            for m in METRICS:
                entry = tests[m].get((env, rival))
                if entry is None:
                    cells.append("---")
                    continue
                _, _, a = entry
                sig = r"$^{\ast}$" if a < ALPHA else "n.s."
                cells.append(f"{a:.4g}~{sig}")
        lines.append(f"\\textit{{DyLam}} vs.\\ {rival}  & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_html(data, tests):
    headers = ["Method"] + [f"{env_label} {COL_LABEL_PLAIN[m]}"
                            for env_label in ("MO-HalfCheetah", "MO-Minecart")
                            for m in METRICS]
    best = _best_per_cell(data)

    body = []
    for method in METHOD_ORDER:
        cells = [method]
        for env in ENVS:
            for m in METRICS:
                vals = data.get((env, method), {}).get(m)
                if not vals:
                    cells.append("—")
                    continue
                show = _shown(vals, m)
                mean, sd = np.mean(show), np.std(show)
                s = f"{mean:{FMT[m]}} ± {sd:{FMT[m]}}"
                cells.append(strong(s) if best.get((env, m)) == method else s)
        body.append(cells)

    body.append(["Mann–Whitney U p-values (two-sided, Holm within each metric, α = 0.05)"]
               + [""] * (len(headers) - 1))

    for rival in [m for m in METHOD_ORDER if m != "DyLam"]:
        cells = [f"DyLam vs. {rival}"]
        for env in ENVS:
            for m in METRICS:
                entry = tests[m].get((env, rival))
                if entry is None:
                    cells.append("—")
                    continue
                _, _, a = entry
                sig = "*" if a < ALPHA else "n.s."
                cells.append(f"{a:.4g} {sig}")
        body.append(cells)

    return html_table(headers, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                     help="paper repository root (default: %(default)s)")
    ap.add_argument("--format", choices=["latex", "html", "both"], default="latex",
                     help="output format(s) to write (default: %(default)s)")
    args = ap.parse_args()

    data = compute()
    tests = family_tests(data)

    if args.format in ("latex", "both"):
        tex = render_tex(data, tests)
        out = args.out_path / FRAGMENT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex)
        print(tex)
        print(f"wrote {out}", file=sys.stderr)

    if args.format in ("html", "both"):
        html = render_html(data, tests)
        out = args.out_path / HTML_FRAGMENT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(html)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
