"""Regenerate the paper's learning-curve figures with IQM + bootstrap CI bands.

Replaces the min--max bands of the first submission, which at n = 10 with bimodal
outcomes are the least informative summary available. Writes straight into the
paper repo so the \includegraphics paths do not change.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stats_r1 import histories, iqm

OUT = Path.home() / "doc/DyLam-TMLR/images/results"
GRID = 200
B = 2000

PANELS = [
    dict(path=OUT / "tradicional/chicken_banana/reward-total.pdf",
         env="CHICKENBANANA", metric="ep_info/total", xlabel="Episode",
         ylabel="Cumulative episode reward",
         series=[("Q-Learning", "Baseline", "tab:orange"),
                 ("Q-Decomp", "Decq", "tab:purple"),
                 ("UDC", "Drq", "tab:blue"),
                 ("DyLam", "Dylam", "tab:green")]),
    dict(path=OUT / "tradicional/halfcheetah/HalfCheetah-v4.pdf",
         env="HALFCHEETAH", metric="ep_info/Final_position", xlabel="Environment step",
         ylabel="Final $x$-position (m)",
         series=[("SAC", "Baseline", "tab:orange"),
                 ("UDC", "Drq", "tab:blue"),
                 ("DyLam", "Dylam", "tab:green")]),
    dict(path=OUT / "tradicional/vss/VSS-v0.pdf",
         env="VSS", metric="ep_info/Goal", xlabel="Environment step",
         ylabel="Goal rate",
         series=[("SAC", "Baseline", "tab:orange"),
                 ("UDC", "Drq", "tab:blue"),
                 ("DyLam", "Dylam", "tab:green")],
         extra=[("Tuned-UDC", "VSS_TUNED", "Drq", "tab:purple")]),
]

def band(hs, metric, grid):
    """IQM curve and 95% bootstrap CI over seeds, on a common step grid."""
    mat = []
    for h in hs:
        step = h["_step"].to_numpy(dtype=float) if "_step" in h else np.arange(len(h))
        # per-seed rolling mean before aggregation: the episodic metrics are noisy
        # enough that an unsmoothed curve hides the trend the summary statistics report
        win = max(5, len(h) // 40)
        val = h[metric].rolling(win, min_periods=1).mean().to_numpy(dtype=float)
        mat.append(np.interp(grid, step, val))
    mat = np.asarray(mat)
    rng = np.random.default_rng(0)
    centre = np.array([iqm(col) for col in mat.T])
    boot = np.array([[iqm(col[rng.integers(0, len(col), len(col))]) for col in mat.T]
                     for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return centre, lo, hi


def curves(entries, metric):
    loaded = []
    for label, env, setup, color in entries:
        hs = histories(env, setup, metric)
        if hs:
            loaded.append((label, color, hs))
    if not loaded:
        return [], None
    hi = min(max(h["_step"].max() if "_step" in h else len(h) for h in hs)
             for _, _, hs in loaded)
    return loaded, np.linspace(0, hi, GRID)


def draw(loaded, grid, metric, path, xlabel, ylabel, title=None):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for label, color, hs in loaded:
        c, lo, hi = band(hs, metric, grid)
        ax.plot(grid, c, color=color, linewidth=1.6)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    for p in PANELS:
        entries = [(lab, p["env"], setup, col) for lab, setup, col in p["series"]]
        entries += [(lab, env, setup, col) for lab, env, setup, col in p.get("extra", [])]
        loaded, grid = curves(entries, p["metric"])
        draw(loaded, grid, p["metric"], p["path"], p["xlabel"], p["ylabel"])


if __name__ == "__main__":
    main()
