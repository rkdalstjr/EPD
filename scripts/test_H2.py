"""H2 검증: |EPD| 극단값 -> 미래 변동성 증가 (부호 무관)."""

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


def compute_epd_abs(close):
    close = np.asarray(close, dtype=float)
    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)
    pe = rolling_pe(r, w_pe=W_PE, m=M, tau=TAU)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]
    z_r = rolling_zscore(r, W_Z)
    z_e = rolling_zscore(dpe, W_Z)
    raw = z_r - z_e
    epd = np.tanh(raw / TANH_SCALE)
    return np.abs(epd), r


def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def quintile_by_abs(epd_abs, fv, n_q=5):
    m = np.isfinite(epd_abs) & np.isfinite(fv)
    if m.sum() < n_q * 30:
        return None, np.nan
    e, f = epd_abs[m], fv[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i + 1])
        if mm.sum() < 20:
            return None, np.nan
        means.append(f[mm].mean())
    means = np.array(means)
    rq = np.arange(1, n_q + 1)
    rm = np.argsort(np.argsort(means)) + 1
    rho = float(np.corrcoef(rq, rm)[0, 1])
    return means, rho


def main():
    import yfinance as yf

    tickers = [
        ("005930.KS", "삼성전자"),
        ("000660.KS", "SK하이닉스"),
        ("005380.KS", "현대차"),
        ("035420.KS", "NAVER"),
        ("051910.KS", "LG화학"),
        ("005490.KS", "POSCO홀딩스"),
        ("068270.KS", "셀트리온"),
        ("373220.KS", "LG엔솔"),
        ("^GSPC", "SPX"),
        ("^IXIC", "NASDAQ"),
        ("^KS11", "KOSPI지수"),
        ("^KQ11", "KOSDAQ지수"),
    ]

    print("=" * 100)
    print(f"H2 검증: |EPD| 극단 -> 미래 {H}일 변동성 증가 (부호 무관)")
    print("=" * 100)
    print(
        f"{'asset':>14s}  {'n':>5s}  | "
        f"{'Q1(|EPD|낮음)':>14s}  {'Q5(|EPD|높음)':>14s}  "
        f"{'ratio':>8s}  {'rho':>6s}"
    )
    print("-" * 100)

    all_rhos = []
    for tkr, name in tickers:
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
                print(f"{name:>14s}  (skip)")
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
            epd_abs, r = compute_epd_abs(close)
            fv = forward_vol(r, H)
            means, rho = quintile_by_abs(epd_abs, fv)
            if means is None:
                print(f"{name:>14s}  {len(close):>5d}  (insufficient)")
                continue
            ratio = means[4] / means[0] if means[0] > 0 else np.nan
            print(
                f"{name:>14s}  {len(close):>5d}  | "
                f"{means[0]:>14.6f}  {means[4]:>14.6f}  "
                f"{ratio:>8.2f}  {rho:>6.2f}"
            )
            if np.isfinite(rho):
                all_rhos.append(rho)
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    print()
    print(f"평균 rho: {np.mean(all_rhos):.3f}")
    print(f"rho > 0 비율: {sum(1 for x in all_rhos if x > 0)}/{len(all_rhos)}")
    print()
    print("판정 기준:")
    print("  - 평균 rho >= 0.5: H2 PASS")
    print("  - 모든 종목 rho > 0: 부호 문제 완전 해결")


if __name__ == "__main__":
    raise SystemExit(main())
