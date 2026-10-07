"""EPD — Entropy-Price Divergence, RSI-style bounded indicator."""
from __future__ import annotations
import numpy as np
from ._core import rolling_pe, rolling_zscore

DEFAULT_W_PE = 60
DEFAULT_M = 3
DEFAULT_TAU = 1
DEFAULT_W_Z = 252
DEFAULT_TANH_SCALE = 2.0
THRESHOLD_HIGH = 70
THRESHOLD_LOW = 30


def compute_epd(returns, close=None, w_pe=DEFAULT_W_PE, m=DEFAULT_M,
                tau=DEFAULT_TAU, w_z=DEFAULT_W_Z, bounded=True,
                tanh_scale=DEFAULT_TANH_SCALE):
    r = np.asarray(returns, dtype=float)
    n = r.size

    pe = rolling_pe(r, w_pe=w_pe, m=m, tau=tau)
    dpe = np.full(n, np.nan)
    if n >= 2:
        dpe[1:] = pe[1:] - pe[:-1]

    z_r = rolling_zscore(r, w_z=w_z)
    z_e = rolling_zscore(dpe, w_z=w_z)
    epd_raw = z_r - z_e

    if bounded:
        epd = np.tanh(epd_raw / tanh_scale)
        epd_100 = 50.0 * (1.0 + epd)
    else:
        epd = epd_raw
        epd_100 = epd_raw

    with np.errstate(invalid="ignore"):
        prod = z_r * z_e
    alignment = np.sign(prod)
    alignment[np.isnan(prod)] = np.nan

    return {
        "epd": epd,
        "epd_100": epd_100,
        "epd_raw": epd_raw,
        "epd_abs": np.abs(epd),
        "alignment": alignment,
        "return_z": z_r,
        "entropy_z": z_e,
        "pe": pe,
        "dpe": dpe,
        "params": {"w_pe": w_pe, "m": m, "tau": tau, "w_z": w_z,
                   "bounded": bounded, "tanh_scale": tanh_scale},
    }
