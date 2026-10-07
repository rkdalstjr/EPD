"""KR 개별주 Stage 2 실패 원인 진단.

4가지 각도:
  1. 기간별 (2005-12, 13-19, 20-25) Q5/Q1
  2. Forward window 변화 (H=5, 10, 20)
  3. 강도 측정 변화 (|EPD_raw|, |Z_r|, |Z_p|, |Z_r|+|Z_e|)
  4. 종목별 편차 (baseline vol, 자기상관)
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd

warnings.filterwarnings("ignore")

FOCUS_STOCKS = [
    ("005930.KS", "삼성전자", "PASS"),
    ("000660.KS", "SK하이닉스", "FAIL"),
    ("005380.KS", "현대차", "FAIL"),
    ("035420.KS", "NAVER", "PASS"),
    ("051910.KS", "LG화학", "FAIL"),
    ("005490.KS", "POSCO홀딩스", "PASS"),
    ("068270.KS", "셀트리온", "PASS"),
    ("373220.KS", "LG엔솔", "FAIL"),
]

PERIODS = [
    ("2005-2012", "2005-01-01", "2012-12-31"),
    ("2013-2019", "2013-01-01", "2019-12-31"),
    ("2020-2025", "2020-01-01", "2025-12-31"),
]

HORIZONS = [5, 10, 20]


def forward_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def q_ratio(intensity, fv, n_q=5):
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
    return float(means[-1] / means[0])


def main():
    import yfinance as yf

    print("=" * 100)
    print("KR_stock 실패 원인 진단")
    print("=" * 100)

    # ============================================================
    # 진단 1: 기간별 Q5/Q1 (H=5, |EPD_raw|)
    # ============================================================
    print("\n### 진단 1. 기간별 Q5/Q1 (H=5, |EPD_raw|)\n")
    print(
        f"{'stock':>14s}  {'status':>7s}  | "
        f"{'2005-2012':>11s}  {'2013-2019':>11s}  {'2020-2025':>11s}"
    )
    print("-" * 80)

    for tkr, name, status in FOCUS_STOCKS:
        row = []
        for pname, start, end in PERIODS:
            try:
                df = yf.download(
                    tkr, start=start, end=end, progress=False, auto_adjust=True
                )
                close = df["Close"].squeeze().to_numpy(dtype=float)
                close = close[np.isfinite(close)]
                if close.size < 400:
                    row.append("      n/a")
                    continue
                out = compute_epd(close)
                r = np.full_like(close, np.nan)
                r[1:] = np.diff(np.log(close))
                fv = forward_vol(r, 5)
                qr = q_ratio(out["epd_abs"], fv)
                row.append(f"{qr:>11.3f}" if np.isfinite(qr) else "      n/a")
            except Exception:
                row.append("      err")
        print(f"{name:>14s}  {status:>7s}  | " + "  ".join(row))

    # ============================================================
    # 진단 2: Forward window 변화
    # ============================================================
    print("\n### 진단 2. Forward window 변화 (H=5, 10, 20)\n")
    print(
        f"{'stock':>14s}  {'status':>7s}  | " f"{'H=5':>9s}  {'H=10':>9s}  {'H=20':>9s}"
    )
    print("-" * 60)

    for tkr, name, status in FOCUS_STOCKS:
        try:
            df = yf.download(
                tkr,
                start="2005-01-01",
                end="2025-12-31",
                progress=False,
                auto_adjust=True,
            )
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
            out = compute_epd(close)
            r = np.full_like(close, np.nan)
            r[1:] = np.diff(np.log(close))

            ratios = []
            for h in HORIZONS:
                fv = forward_vol(r, h)
                qr = q_ratio(out["epd_abs"], fv)
                ratios.append(f"{qr:>9.3f}" if np.isfinite(qr) else "      n/a")
            print(f"{name:>14s}  {status:>7s}  | " + "  ".join(ratios))
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    # ============================================================
    # 진단 3: 강도 측정 변화
    # ============================================================
    print("\n### 진단 3. 강도 측정 변화 (H=5)\n")
    print(
        f"{'stock':>14s}  {'status':>7s}  | "
        f"{'|EPD_raw|':>10s}  {'|Z_r|':>9s}  {'|Z_e|':>9s}  "
        f"{'|Z_r|+|Z_e|':>12s}  {'max':>9s}"
    )
    print("-" * 100)

    for tkr, name, status in FOCUS_STOCKS:
        try:
            df = yf.download(
                tkr,
                start="2005-01-01",
                end="2025-12-31",
                progress=False,
                auto_adjust=True,
            )
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
            out = compute_epd(close)
            z_r = out["return_z"]
            z_e = out["entropy_z"]
            r = np.full_like(close, np.nan)
            r[1:] = np.diff(np.log(close))
            fv = forward_vol(r, 5)

            measures = {
                "|EPD|": out["epd_abs"],
                "|Z_r|": np.abs(z_r),
                "|Z_e|": np.abs(z_e),
                "|Z_r|+|Z_e|": np.abs(z_r) + np.abs(z_e),
                "max": np.maximum(np.abs(z_r), np.abs(z_e)),
            }
            vals = [q_ratio(v, fv) for v in measures.values()]
            row = "  ".join(
                f"{v:>9.3f}" if np.isfinite(v) else "      n/a" for v in vals
            )
            print(f"{name:>14s}  {status:>7s}  | {row}")
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    # ============================================================
    # 진단 4: 종목 특성 (baseline vol, 자기상관)
    # ============================================================
    print("\n### 진단 4. 종목 특성 비교\n")
    print(
        f"{'stock':>14s}  {'status':>7s}  | "
        f"{'mean vol':>10s}  {'autocorr1':>10s}  {'PE mean':>9s}"
    )
    print("-" * 70)

    for tkr, name, status in FOCUS_STOCKS:
        try:
            df = yf.download(
                tkr,
                start="2005-01-01",
                end="2025-12-31",
                progress=False,
                auto_adjust=True,
            )
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
            out = compute_epd(close)
            r = np.full_like(close, np.nan)
            r[1:] = np.diff(np.log(close))
            r_valid = r[np.isfinite(r)]

            mean_vol = float(r_valid.std(ddof=1) * np.sqrt(252))
            # AR(1)
            if r_valid.size > 100:
                ac1 = float(np.corrcoef(r_valid[:-1], r_valid[1:])[0, 1])
            else:
                ac1 = np.nan
            pe = out["pe"]
            pe_mean = float(np.nanmean(pe)) if np.isfinite(pe).any() else np.nan

            print(
                f"{name:>14s}  {status:>7s}  | "
                f"{mean_vol:>10.4f}  {ac1:>+10.4f}  {pe_mean:>9.4f}"
            )
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    # ============================================================
    # 요약
    # ============================================================
    print()
    print("=" * 100)
    print("요약")
    print("=" * 100)
    print("진단 1: 기간별 편차 → 특정 시기에만 약한가?")
    print("진단 2: window 변화 → H=5가 개별주에 너무 짧은가?")
    print("진단 3: 강도 측정 변화 → EPD_raw가 최적인가?")
    print("진단 4: 종목 특성 → FAIL 종목이 구조적으로 다른가?")


if __name__ == "__main__":
    raise SystemExit(main())
