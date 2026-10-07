"""EPD v2.0 - Z_r based, EMA14 smoothed."""
from __future__ import annotations
import numpy as np
from ._core import rolling_pe, rolling_zscore

DEFAULT_W_PE = 60
DEFAULT_M = 3
DEFAULT_TAU = 1
DEFAULT_W_Z = 252
DEFAULT_TANH_SCALE = 2.0
DEFAULT_EMA_SPAN = 14

THRESHOLD_HIGH = 70
THRESHOLD_LOW = 30


def _ema(x, span):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    for t in range(len(x)):
        xt = x[t]
        if not np.isfinite(xt):
            out[t] = prev
        else:
            prev = xt if not np.isfinite(prev) else alpha * xt + (1.0 - alpha) * prev
            out[t] = prev
    return out


def compute_epd(close, w_pe=DEFAULT_W_PE, m=DEFAULT_M, tau=DEFAULT_TAU,
                w_z=DEFAULT_W_Z, tanh_scale=DEFAULT_TANH_SCALE,
                ema_span=DEFAULT_EMA_SPAN):
    close = np.asarray(close, dtype=float)
    if close.size < 2:
        raise ValueError("close must have >= 2 elements")
    if np.any(close <= 0):
        raise ValueError("close must be strictly positive")

    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)

    pe = rolling_pe(r, w_pe=w_pe, m=m, tau=tau)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]

    z_r = rolling_zscore(r, w_z=w_z)
    z_e = rolling_zscore(dpe, w_z=w_z)

    raw = z_r - z_e
    ema14 = _ema(raw, ema_span)

    epd_raw_100 = 50.0 * (1.0 + np.tanh(raw / tanh_scale))
    epd_100 = 50.0 * (1.0 + np.tanh(ema14 / tanh_scale))

    return {
        "epd_100": epd_100,
        "epd_raw_100": epd_raw_100,
        "epd_raw": raw,
        "epd_ema": ema14,
        "epd_abs": np.abs(raw),
        "return_z": z_r,
        "entropy_z": z_e,
        "pe": pe,
        "dpe": dpe,
        "params": {
            "w_pe": w_pe, "m": m, "tau": tau, "w_z": w_z,
            "tanh_scale": tanh_scale, "ema_span": ema_span,
            "version": "2.0.0",
        },
    }
