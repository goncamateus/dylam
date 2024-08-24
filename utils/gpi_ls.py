import numpy as np


def unique_tol(a, tol=1e-4):
    """Returns unique elements of a list of np.arrays, within a tolerance."""
    if len(a) == 0:
        return a
    delete = np.array([False] * len(a))
    a = np.array(a)
    for i in range(len(a)):
        if delete[i]:
            continue
        for j in range(i + 1, len(a)):
            if np.allclose(a[i], a[j], tol):
                delete[j] = True
    return list(a[~delete])
