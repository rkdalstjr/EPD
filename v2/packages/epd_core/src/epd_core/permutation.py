"""순열 엔트로피 (Bandt-Pompe 2002)."""
from __future__ import annotations
import numpy as np
from math import factorial, log


def _ordinal_pattern(vec: np.ndarray) -> tuple:
    return tuple(np.argsort(vec, kind="stable").tolist())


def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """정규화된 순열 엔트로피 in [0, 1]."""
    x = np.asarray(x, dtype=float)
    n = x.size
    span = (m - 1) * tau + 1
    if n < span:
        return np.nan

    counts = {}
    total = 0
    for i in range(n - span + 1):
        vec = x[i : i + span : tau]
        key = _ordinal_pattern(vec)
        counts[key] = counts.get(key, 0) + 1
        total += 1

    if total == 0:
        return np.nan

    p = np.fromiter(counts.values(), dtype=float, count=len(counts)) / total
    h = -float(np.sum(p * np.log(p)))
    h_max = log(factorial(m))
    return h / h_max if h_max > 0 else np.nan
