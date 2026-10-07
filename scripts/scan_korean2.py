"""스캔 2 (최적화): PE 캐싱으로 3배 빠름.

목표: 한국 개별주(7종목) rho > 0.10 이면서 US_index rho > 0.05 (부호 유지).
전략: PE는 w_z 무관 → 한 번만 계산. z-score는 w_z별 캐싱.
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

warnings.filterwarnings("ignore")

# ---- 고정 파라미터 ----
W_PE, M, TAU = 60, 3, 1
HORIZONS = [10, 20]
WZ_LIST = [120, 252, 504]


def compute_base(close):
    """PE/dpe는 w_z 무관 → 한 번만 계산."""
    close = np.asarray(close, dtype=float)
    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)

    pe = rolling_pe(r, w_pe=W_PE, m=M, tau=TAU)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]
    return log_P, r, dpe


def z_at_wz(r, log_P, dpe, w_z):
    """w_z별 z-score 3종."""
    z_r = rolling_zscore(r, w_z)
    z_p = rolling_zscore(log_P, w_z)
    z_e = rolling_zscore(dpe, w_z)
    return z_r, z_p, z_e


def intensity(z_r, z_p, z_e, kind):
    """강도 측정 함수 (10종)."""
    ar, ap = np.abs(z_r), np.abs(z_p)
    if kind == "Zr":
        return ar
    if kind == "Zp":
        return ap
    if kind == "sum":
        return ar + ap
    if kind == "max":
        return np.maximum(ar, ap)
    if kind == "norm":
        return np.sqrt(z_r**2 + z_p**2)
    if kind == "prod":
        return ar * ap
    if kind == "Zr+Ze":
        return ar + np.abs(z_e)
    if kind == "Zp+Ze":
        return ap + np.abs(z_e)
    if kind == "ZrZpZe":
        return ar + ap + np.abs(z_e)
    if kind == "hypot3":
        return np.sqrt(z_r**2 + z_p**2 + z_e**2)
    raise ValueError(kind)


def forward_vol(r, h):
    """미래 h일 실현 변동성."""
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def spearman(x, y):
    """Spearman rank correlation."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def q_ratio(x, y):
    """상위 20% / 하위 20% 평균 비율."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return np.nan
    a, b = x[m], y[m]
    lo = np.quantile(a, 0.2)
    hi = np.quantile(a, 0.8)
    lm = a <= lo
    hm = a >= hi
    if lm.sum() < 20 or hm.sum() < 20:
        return np.nan
    return float(b[hm].mean() / b[lm].mean())


def main():
    import yfinance as yf
    import time

    tickers = [
        ("005930.KS", "삼성전자", "KR_stock"),
        ("000660.KS", "SK하이닉스", "KR_stock"),
        ("005380.KS", "현대차", "KR_stock"),
        ("035420.KS", "NAVER", "KR_stock"),
        ("051910.KS", "LG화학", "KR_stock"),
        ("005490.KS", "POSCO홀딩스", "KR_stock"),
        ("068270.KS", "셀트리온", "KR_stock"),
        ("^KQ11", "KOSDAQ지수", "KR_index"),
        ("^KS11", "KOSPI지수", "KR_index"),
        ("^GSPC", "SPX", "US_index"),
        ("^IXIC", "NASDAQ", "US_index"),
    ]

    # ---------- 1) 데이터 로드 + PE/기본 계산 캐시 ----------
    print("[1/3] 데이터 로드 + PE 계산 캐시...")
    cache = {}
    t0 = time.time()
    for tkr, name, grp in tickers:
        try:
            df = yf.download(
                tkr,
                start="2005-01-01",
                end="2025-12-31",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < 500:
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
            log_P, r, dpe = compute_base(close)
            cache[name] = {
                "grp": grp,
                "r": r,
                "log_P": log_P,
                "dpe": dpe,
                "z_cache": {},
            }
            print(f"  {name}: n={len(close)}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {name}: err {e}")

    # ---------- 2) w_z별 z-score 캐시 ----------
    print("[2/3] w_z별 z-score 캐시...")
    for w_z in WZ_LIST:
        for name, d in cache.items():
            z_r, z_p, z_e = z_at_wz(d["r"], d["log_P"], d["dpe"], w_z)
            d["z_cache"][w_z] = (z_r, z_p, z_e)
    print(f"  done ({time.time()-t0:.1f}s)")

    # ---------- 3) 스캔 ----------
    print("[3/3] 스캔...")
    fv_cache = {}
    for H in HORIZONS:
        fv_cache[H] = {name: forward_vol(d["r"], H) for name, d in cache.items()}

    kinds = [
        "Zr",
        "Zp",
        "sum",
        "max",
        "norm",
        "prod",
        "Zr+Ze",
        "Zp+Ze",
        "ZrZpZe",
        "hypot3",
    ]

    candidates = []
    total = len(kinds) * len(HORIZONS) * len(WZ_LIST)
    idx = 0
    for kind, H, w_z in product(kinds, HORIZONS, WZ_LIST):
        idx += 1
        rhos = {"KR_stock": [], "KR_index": [], "US_index": []}
        qr_kr = []
        for name, d in cache.items():
            z_r, z_p, z_e = d["z_cache"][w_z]
            inten = intensity(z_r, z_p, z_e, kind)
            fv = fv_cache[H][name]
            rho = spearman(inten, fv)
            if np.isfinite(rho):
                rhos[d["grp"]].append(rho)
            if d["grp"] == "KR_stock":
                qr = q_ratio(inten, fv)
                if np.isfinite(qr):
                    qr_kr.append(qr)

        m_kr = np.mean(rhos["KR_stock"]) if rhos["KR_stock"] else np.nan
        m_kri = np.mean(rhos["KR_index"]) if rhos["KR_index"] else np.nan
        m_us = np.mean(rhos["US_index"]) if rhos["US_index"] else np.nan
        m_qr = np.mean(qr_kr) if qr_kr else np.nan

        # 조건: KR_stock rho > 0.10 AND US_index rho > 0.05 (부호 유지)
        if (m_kr > 0.10) and (m_us > 0.05):
            candidates.append((kind, H, w_z, m_kr, m_kri, m_us, m_qr))

        # 진행 상황
        if idx % 10 == 0:
            print(
                f"  [{idx}/{total}] {kind} H={H} w_z={w_z}  "
                f"KR={m_kr:.3f} US={m_us:.3f}"
            )

    print(f"\n총 소요: {time.time()-t0:.1f}s")
    print("=" * 100)

    if not candidates:
        print("[결과] 조건 만족 조합 없음")
        print("→ |Z_r| 기반(H=20) rho≈0.093이 최선")
        return

    candidates.sort(key=lambda x: -x[3])
    print(f"[통과 조합] {len(candidates)}개\n")
    print(
        f"{'kind':>8s} {'H':>3s} {'w_z':>5s} | "
        f"{'KR_stock':>9s} {'KR_index':>9s} {'US_index':>9s} {'KR Q5/Q1':>10s}"
    )
    print("-" * 80)
    for kind, H, w_z, m_kr, m_kri, m_us, m_qr in candidates[:20]:
        print(
            f"{kind:>8s} {H:>3d} {w_z:>5d} | "
            f"{m_kr:>9.3f} {m_kri:>9.3f} {m_us:>9.3f} {m_qr:>10.3f}"
        )

    top = candidates[0]
    print(f"\n**최적**: kind={top[0]}, H={top[1]}, w_z={top[2]}")
    print(f"  KR_stock rho={top[3]:.3f}, US_index rho={top[5]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
