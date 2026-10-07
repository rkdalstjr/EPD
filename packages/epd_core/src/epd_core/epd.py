"""EPD (Entropy-Price Divergence) — v1.0 final.

정의:
    EPD_raw  = Z_r − Z_e
    EPD_smooth = EMA(EPD_raw, span=14)
    EPD_100  = 50 · (1 + tanh(EPD_smooth / 2))  ∈ (0, 100)

파라미터 (이론적 고정, 변경 금지):
    w_pe = 60, m = 3, tau = 1     (Bandt-Pompe 순열 엔트로피)
    w_z = 252                      (1년 거래일, causal z-score)
    tanh_scale = 2                 (RSI 70/30 정렬)
    ema_span = 14                  (RSI Wilder smoothing과 동일)

임계값:
    EPD_100 > 70  →  과열 위험
    EPD_100 < 30  →  구조 형성 중

주의:
    EPD는 방향 예측 지표가 아님. "구조적 긴장(structural tension)"의
    강도를 표시하는 상태 지표. 방향 예측 용도로 사용 금지.
"""

from __future__ import annotations
import numpy as np
from ._core import rolling_pe, rolling_zscore

# ===== 이론적 고정 상수 =====
DEFAULT_W_PE = 60
DEFAULT_M = 3
DEFAULT_TAU = 1
DEFAULT_W_Z = 252
DEFAULT_TANH_SCALE = 2.0
DEFAULT_EMA_SPAN = 14

THRESHOLD_HIGH = 70
THRESHOLD_LOW = 30


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """지수이동평균. NaN은 이전 값 유지."""
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


def compute_epd(
    close: np.ndarray,
    w_pe: int = DEFAULT_W_PE,
    m: int = DEFAULT_M,
    tau: int = DEFAULT_TAU,
    w_z: int = DEFAULT_W_Z,
    tanh_scale: float = DEFAULT_TANH_SCALE,
    ema_span: int = DEFAULT_EMA_SPAN,
) -> dict:
    """EPD 계산.

    Parameters
    ----------
    close : 1D array, 양수
        종가 시계열.

    Returns
    -------
    dict:
        epd_100        : 0~100, EMA14 스무딩 (주 사용)
        epd_raw_100    : 0~100, EMA 없음 (원시)
        epd_raw        : Z_r − Z_e (raw z-score)
        epd_ema        : EMA14(Z_r − Z_e)
        epd_abs        : |Z_r − Z_e| (강도)
        return_z       : Z_r
        entropy_z      : Z_e
        pe             : 순열 엔트로피
        dpe            : ΔPE
        params         : dict
    """
    close = np.asarray(close, dtype=float)
    if close.size < 2:
        raise ValueError("close must have >= 2 elements")
    if np.any(close <= 0):
        raise ValueError("close must be strictly positive")

    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)

    # 순열 엔트로피
    pe = rolling_pe(r, w_pe=w_pe, m=m, tau=tau)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]

    # z-score (causal)
    z_r = rolling_zscore(r, w_z=w_z)
    z_e = rolling_zscore(dpe, w_z=w_z)

    # EPD
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
            "w_pe": w_pe,
            "m": m,
            "tau": tau,
            "w_z": w_z,
            "tanh_scale": tanh_scale,
            "ema_span": ema_span,
            "version": "1.0.0",
        },
    }
