"""한국 개별주 통과를 위한 강도/기간 조합 스캔.

목표: 한국 개별주(7종목)에서 rho > 0.10, Q5/Q1 > 1.10
동시에 SPX/NASDAQ/KOSDAQ에서도 유지되는지 확인 (공정성).

강도 후보: |EPD|, |Z_r|, |Z_p|, |Z_r|+|Z_e|, |Z_p|+|Z_e|, max(|Z_r|,|Z_e|), sqrt(Z_r²+Z_e²)
Horizon: 5, 10, 20
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

W_PE, M, TAU = 60, 3, 1
W_Z = 252
HORIZONS = [5, 10, 20]


def compute_z(close):
    close = np.asarray(close, dtype=float)
    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)
    pe = rolling_pe(r, w_pe=W_PE, m=M, tau=TAU)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]
    z_r = rolling_zscore(r, W_Z)
    z_p = rolling_zscore(log_P, W_Z)
    z_e = rolling_zscore(dpe, W_Z)
    return z_r, z_p, z_e, r


def intensity(z_r, z_p, z_e, kind):
    if kind == "EPD":
        return np.abs(z_r - z_e)
    if kind == "Zr":
        return np.abs(z_r)
    if kind == "Zp":
        return np.abs(z_p)
    if kind == "Ze":
        return np.abs(z_e)
    if kind == "ZrZe":
        return np.abs(z_r) + np.abs(z_e)
    if kind == "ZpZe":
        return np.abs(z_p) + np.abs(z_e)
    if kind == "max":
        return np.maximum(np.abs(z_r), np.abs(z_e))
    if kind == "norm":
        return np.sqrt(z_r**2 + z_e**2)
    raise ValueError(kind)


def forward_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def q_ratio(x, y, q_lo=0.2, q_hi=0.8):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return np.nan
    a, b = x[m], y[m]
    lo = np.quantile(a, q_lo)
    hi = np.quantile(a, q_hi)
    lm = a <= lo
    hm = a >= hi
    if lm.sum() < 20 or hm.sum() < 20:
        return np.nan
    return float(b[hm].mean() / b[lm].mean())


def main():
    import yfinance as yf

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

    # 데이터 로딩
    data = {}
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
            z_r, z_p, z_e, r = compute_z(close)
            data[name] = {"grp": grp, "z_r": z_r, "z_p": z_p, "z_e": z_e, "r": r}
        except Exception:
            pass

    kinds = ["EPD", "Zr", "Zp", "ZrZe", "ZpZe", "max", "norm"]

    print("=" * 110)
    print("스캔: 강도 × horizon 별 mean rho (자산군)")
    print("=" * 110)
    print(
        f"{'kind':>6s}  {'H':>3s}  | "
        f"{'KR_stock rho':>13s}  {'KR_index rho':>13s}  {'US_index rho':>13s}  | "
        f"{'KR_stock Q5/Q1':>15s}  {'KR_stock pass?':>14s}"
    )
    print("-" * 110)

    best = None
    for kind, H in product(kinds, HORIZONS):
        rhos_by_grp = {"KR_stock": [], "KR_index": [], "US_index": []}
        qr_kr = []
        for name, d in data.items():
            inten = intensity(d["z_r"], d["z_p"], d["z_e"], kind)
            fv = forward_vol(d["r"], H)
            rho = spearman(inten, fv)
            if np.isfinite(rho):
                rhos_by_grp[d["grp"]].append(rho)
            if d["grp"] == "KR_stock":
                qr = q_ratio(inten, fv)
                if np.isfinite(qr):
                    qr_kr.append(qr)

        m_kr = np.mean(rhos_by_grp["KR_stock"]) if rhos_by_grp["KR_stock"] else np.nan
        m_kri = np.mean(rhos_by_grp["KR_index"]) if rhos_by_grp["KR_index"] else np.nan
        m_us = np.mean(rhos_by_grp["US_index"]) if rhos_by_grp["US_index"] else np.nan
        m_qr = np.mean(qr_kr) if qr_kr else np.nan
        pass_kr = (m_kr > 0.10) and (m_qr > 1.10)

        flag = "✅" if pass_kr else ""
        print(
            f"{kind:>6s}  {H:>3d}  | "
            f"{m_kr:>13.3f}  {m_kri:>13.3f}  {m_us:>13.3f}  | "
            f"{m_qr:>15.3f}  {flag:>14s}"
        )

        if pass_kr:
            if best is None or m_kr > best[2]:
                best = (kind, H, m_kr, m_kri, m_us, m_qr)

    print()
    if best is not None:
        print(f"**최적 조합**: kind={best[0]}, H={best[1]}")
        print(
            f"  KR_stock rho={best[2]:.3f}, KR_index={best[3]:.3f}, US_index={best[4]:.3f}, KR_stock Q5/Q1={best[5]:.3f}"
        )
    else:
        print("**조합 중 KR_stock 통과 없음** — 다른 접근 필요")


if __name__ == "__main__":
    raise SystemExit(main())
