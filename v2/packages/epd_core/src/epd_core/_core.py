"""롤링 유틸: PE, causal z-score, slope."""
from __future__ import annotations
import numpy as np
from .permutation import permutation_entropy


def rolling_pe(returns: np.ndarray, w_pe: int = 60,
               m: int = 3, tau: int = 1) -> np.ndarray:
    """PE_t = permutation_entropy(r[t-w_pe:t]). Causal."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    out = np.full(n, np.nan)
    for t in range(w_pe, n):
        out[t] = permutation_entropy(r[t - w_pe : t], m=m, tau=tau)
    return out


def rolling_zscore(x: np.ndarray, w_z: int = 252,
                   min_periods: int | None = None) -> np.ndarray:
    """Causal rolling z-score."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if min_periods is None:
        min_periods = w_z
    out = np.full(n, np.nan)
    for t in range(n):
        lo = t - w_z
        if lo < 0:
            continue
        win = x[lo:t]
        win = win[~np.isnan(win)]
        if win.size < min_periods:
            continue
        mu = win.mean()
        sd = win.std(ddof=1)
        if not np.isfinite(sd) or sd == 0.0:
            out[t] = 0.0
        else:
            out[t] = (x[t] - mu) / sd
    return out


def rolling_slope(y: np.ndarray, w: int = 60) -> np.ndarray:
    """y[t-w+1 : t+1] 단순선형회귀 기울기."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    xgrid = np.arange(w, dtype=float)
    xmean = xgrid.mean()
    xvar = ((xgrid - xmean) ** 2).sum()
    for t in range(w - 1, n):
        win = y[t - w + 1 : t + 1]
        if np.isnan(win).any():
            continue
        ymean = win.mean()
        out[t] = ((xgrid - xmean) * (win - ymean)).sum() / xvar
    return out
