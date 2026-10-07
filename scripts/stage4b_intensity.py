"""Stage 4b: 다양한 강도 측정 비교 — |EPD|, |Z_r|, |Z_e|, |Z_r|+|Z_e|."""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

warnings.filterwarnings("ignore")

W_PE, M, TAU, W_Z = 60, 3, 1, 252
TANH_SCALE = 2.0
H = 5


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


def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def quintile_ratio(intensity, fv, n_q=5):
    m = np.isfinite(intensity) & np.isfinite(fv)
    if m.sum() < n_q * 30:
        return np.nan
    a, b = intensity[m], fv[m]
    qs = np.quantile(a, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (a >= qs[i]) & (a < qs[i + 1])
        if mm.sum() < 20:
            return np.nan
        means.append(b[mm].mean())
    if means[0] <= 0:
        return np.nan
    return means[-1] / means[0]


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
        ("^KS11", "KOSPI지수", "KR_index"),
        ("^KQ11", "KOSDAQ지수", "KR_index"),
        ("^GSPC", "SPX", "US_index"),
        ("^IXIC", "NASDAQ", "US_index"),
    ]

    measures = {
        "|EPD|": lambda zr, zp, ze: np.abs(zr - ze),
        "|Z_r|": lambda zr, zp, ze: np.abs(zr),
        "|Z_p|": lambda zr, zp, ze: np.abs(zp),
        "|Z_e|": lambda zr, zp, ze: np.abs(ze),
        "|Z_r|+|Z_e|": lambda zr, zp, ze: np.abs(zr) + np.abs(ze),
        "|Z_p|+|Z_e|": lambda zr, zp, ze: np.abs(zp) + np.abs(ze),
        "|Z_r|*|Z_e|": lambda zr, zp, ze: np.abs(zr) * np.abs(ze),
    }

    print("=" * 110)
    print("Stage 4b: 강도 측정 비교 — Q5/Q1 (forward 5d vol)")
    print("=" * 110)

    # 헤더
    hdr = f"{'asset':>14s}  {'grp':>10s}  | " + "  ".join(f"{k:>10s}" for k in measures)
    print(hdr)
    print("-" * len(hdr))

    rows = []
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
            zr, zp, ze, r = compute_z(close)
            fv = forward_vol(r, H)
            res = {"name": name, "grp": grp}
            row_str = f"{name:>14s}  {grp:>10s}  | "
            for key, fn in measures.items():
                intens = fn(zr, zp, ze)
                ratio = quintile_ratio(intens, fv)
                res[key] = ratio
                row_str += f"  {ratio:>10.3f}"
            print(row_str)
            rows.append(res)
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    # 자산군 요약
    print()
    print("=" * 110)
    print("자산군별 평균 Q5/Q1")
    print("=" * 110)
    hdr = f"{'group':>14s}  {'n':>3s}  | " + "  ".join(f"{k:>10s}" for k in measures)
    print(hdr)
    print("-" * len(hdr))
    for grp in ["KR_stock", "KR_index", "US_index"]:
        sub = [r for r in rows if r["grp"] == grp]
        if not sub:
            continue
        row_str = f"{grp:>14s}  {len(sub):>3d}  | "
        for key in measures:
            vals = [r[key] for r in sub if np.isfinite(r[key])]
            mean = np.mean(vals) if vals else np.nan
            row_str += f"  {mean:>10.3f}"
        print(row_str)

    print()
    print("판정: 모든 자산군에서 가장 균일하게 > 1.05인 measure를 선택")


if __name__ == "__main__":
    raise SystemExit(main())
