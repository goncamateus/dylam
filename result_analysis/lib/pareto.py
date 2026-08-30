"""Pareto-dominance filtering and hypervolume, for the morl scope.

All objectives are maximized. Reward components that are naturally
penalties (e.g. control cost) must already be transformed into the
maximizing direction before reaching these functions -- see morl/arms.py.
"""
import numpy as np
from pymoo.indicators.hv import HV


def non_dominated_mask(points, remove_duplicates=True, chunk_size=2000):
    """Boolean mask selecting the non-dominated (Pareto-optimal) rows of `points`.

    Compares `chunk_size` rows against the full set at a time rather than
    building one (n, n, d) dominance tensor -- that tensor is what blew up
    memory on morl's ~10^4-30^4-point merged fronts (n=29488 alone needs
    multiple GB). Chunking bounds peak memory to O(chunk_size * n * d)
    regardless of n, with the identical result.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    _, first_idx, inverse, counts = np.unique(
        points, return_index=True, return_inverse=True, return_counts=True, axis=0)
    not_dominated = np.empty(n, dtype=bool)
    not_strictly_worse_somewhere = np.empty(n, dtype=bool)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[start:end]
        at_least_as_good = np.all(chunk[:, None, :] <= points[None, :, :], axis=-1)
        strictly_better = np.all(chunk[:, None, :] < points[None, :, :], axis=-1)
        not_dominated[start:end] = np.sum(at_least_as_good, axis=-1) == counts[inverse[start:end]]
        not_strictly_worse_somewhere[start:end] = np.any(~strictly_better, axis=-1)
    mask = not_dominated & not_strictly_worse_somewhere
    if remove_duplicates:
        keep_first = np.zeros(n, dtype=bool)
        keep_first[first_idx] = True
        mask &= keep_first
    return mask


def filter_dominated(points, remove_duplicates=True):
    """The non-dominated subset of `points`; unchanged if fewer than 2 rows."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    return points[non_dominated_mask(points, remove_duplicates)]


def hypervolume(points, ref_point):
    """Hypervolume dominated by `points` with respect to `ref_point` (both maximized)."""
    points = np.asarray(points, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    return float(HV(ref_point=-ref_point)(-points))
