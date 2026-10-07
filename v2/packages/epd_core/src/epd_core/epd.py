"""EPD v3.1 — 최종 확정.

정의:
    Z_r(t) = causal_zscore(r, w_z)
    PE_t   = permutation_entropy(r[t-w_pe:t], m, tau)
    Z_e(t) = causal_zscore(PE − E[PE | |Z_r|], w_z)
    EPD_mag = ||Z_r| − |Z_e||
    EPD_100 = 100 · tanh(EPD_mag / 2)         (0, 100)

파라미터 (Stage 0 SPEC 동결):
    w_pe=60, m=3, tau=1, w_z=252, resid_win=504, tanh_scale=2.0

해석:
    EPD_100 < 30   : 낮은 긴장 (Low tension)
    30 ≤ EPD_100 < 70 : 중간 (Mid)
    EPD_100 ≥ 70   : 높은 긴장 (High tension)

주의:
    EPD는 방향 예측 지표가 아님.
    "가격 스트레스와 엔트로피 이탈의 magnitude divergence"를
    측정하는 시장 상태 변수.
"""

from __future__ import annotations
import numpy as np
from ._core import rolling_pe, rolling_zscore

# ===== 이론적 고정 상수 =====
DEFAULT_W_PE = 60
DEFAULT_M = 3
DEFAULT_TAU = 1
DEFAULT_W_Z = 252
DEFAULT_RESID_WIN = 504
DEFAULT_TANH_SCALE = 2.0

THRESHOLD_HIGH = 70
THRESHOLD_LOW = 30
EPS = 1e-9


def _entropy_residual(pe, z_r, resid_win=DEFAULT_RESID_WIN):
    """E[PE | |Z_r|] 제거한 잔차."""
    n = len(pe)
    abs_zr = np.abs(z_r)
    resid = np.full(n, np.nan)
    for t in range(resid_win, n):
        win_zr = abs_zr[t - resid_win : t]
        win_pe = pe[t - resid_win : t]
        mask = np.isfinite(win_zr) & np.isfinite(win_pe)
        if mask.sum() < 100:
            continue
        w_zr, w_pe = win_zr[mask], win_pe[mask]
        if not (np.isfinite(abs_zr[t]) and np.isfinite(pe[t])):
            continue
        qs = np.quantile(w_zr, np.linspace(0, 1, 6))
        qs[0] -= EPS
        qs[-1] += EPS
        b = int(np.searchsorted(qs, abs_zr[t], side="right") - 1)
        b = max(0, min(4, b))
        bm = (w_zr >= qs[b]) & (w_zr < qs[b + 1])
        if bm.sum() < 5:
            continue
        resid[t] = pe[t] - w_pe[bm].mean()
    return resid


def compute_epd(
    close,
    w_pe=DEFAULT_W_PE,
    m=DEFAULT_M,
    tau=DEFAULT_TAU,
    w_z=DEFAULT_W_Z,
    resid_win=DEFAULT_RESID_WIN,
    tanh_scale=DEFAULT_TANH_SCALE,
):
    """EPD v3.1 — magnitude divergence indicator."""
    close = np.asarray(close, dtype=float)
    n = close.size
    if n < max(w_pe, w_z, resid_win) + 10:
        raise ValueError("close too short")

    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)

    pe = rolling_pe(r, w_pe=w_pe, m=m, tau=tau)
    z_r = rolling_zscore(r, w_z=w_z)

    resid = _entropy_residual(pe, z_r, resid_win=resid_win)
    z_e = rolling_zscore(resid, w_z=w_z)

    # 핵심: magnitude divergence
    epd_mag = np.abs(np.abs(z_r) - np.abs(z_e))
    epd_100 = 100.0 * np.tanh(epd_mag / tanh_scale)

    # 진단용
    epd_dir = np.sign(np.abs(z_r) - np.abs(z_e))
    epd_raw = z_r - z_e  # 기존 (참고)

    return {
        "epd_100": epd_100,  # 주 사용
        "epd_mag": epd_mag,  # raw magnitude
        "epd_dir": epd_dir,  # diagnostic
        "epd_raw": epd_raw,  # legacy
        "z_r": z_r,
        "z_e": z_e,
        "price_z": np.abs(z_r),
        "entropy_z": np.abs(z_e),
        "pe": pe,
        "resid": resid,
        "params": {
            "w_pe": w_pe,
            "m": m,
            "tau": tau,
            "w_z": w_z,
            "resid_win": resid_win,
            "tanh_scale": tanh_scale,
            "version": "3.1.0",
        },
    }
