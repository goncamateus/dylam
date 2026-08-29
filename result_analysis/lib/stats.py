"""Shared statistics: seed summary, IQM, bootstrap CI, exact Mann-Whitney, Holm.

The seed summary is the unit of independence used by every statistical test
in the paper; every scope reads it from here rather than re-implementing it,
so the definition cannot drift between scopes.
"""
import numpy as np
from scipy.stats import mannwhitneyu


def seed_summary(df, metric, frac=0.1):
    """The seed summary: mean of the final `frac` of one run's logged metric."""
    v = df[metric].to_numpy(dtype=float)
    return float(v[int((1 - frac) * len(v)):].mean())


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    lo, hi = int(np.floor(0.25 * len(x))), int(np.ceil(0.75 * len(x)))
    return float(x[lo:hi].mean())


def boot_ci(x, stat=iqm, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    s = [stat(rng.choice(x, len(x), replace=True)) for _ in range(B)]
    return tuple(np.percentile(s, [2.5, 97.5]))


def exact_mw(a, b):
    """Exact two-sided Mann-Whitney + rank-biserial (positive => a > b)."""
    u, p = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    return u, p, 2.0 * u / (len(a) * len(b)) - 1.0


def bootstrap_curve(mat, B=2000, seed=0):
    """IQM curve + 95% bootstrap CI band across seeds.

    `mat` is (n_seeds, n_gridpoints); each bootstrap draw resamples seeds
    once and reuses that draw across every grid column, so the band reflects
    seed variability rather than per-column resampling noise.
    """
    rng = np.random.default_rng(seed)
    mat = np.asarray(mat, dtype=float)
    centre = np.array([iqm(col) for col in mat.T])
    boot = np.array([[iqm(col[rng.integers(0, len(col), len(col))]) for col in mat.T]
                      for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return centre, lo, hi


def holm(ps):
    """Holm-Bonferroni correction within one family of comparisons."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out
