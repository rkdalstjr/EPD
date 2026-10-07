"""Stage 10 — Time Stability.

R1 = |Zr|
R6 = ||Zr| - |Ze||

기간 분할 (4구간):
  2005-2010, 2010-2015, 2015-2020, 2020-2025

각 기간에서:
  - Spearman rho (R1, R6) vs vol_5/vol_10/vol_20
  - Partial Spearman (R1, R6 | |Zr|)  ← R6는 자기 자신으로 통제 시 무의미
    → 대신 |Zr| 통제 후 R6 partial만 측정

핵심 질문:
  R6 partial이 모든 기간에서 양수이고 안정적인가?
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd_v3

warnings.filterwarnings("ignore")

ASSETS = {
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "Samsung": "005930.KS",
    "SK Hynix": "000660.KS",
    "NAVER": "035420.KS",
    "LG Chem": "051910.KS",
}

PERIODS = [
    ("2005-2010", "2005-01-01", "2010-12-31"),
    ("2010-2015", "2011-01-01", "2015-12-31"),
    ("2015-2020", "2016-01-01", "2020-12-31"),
    ("2020-2025", "2021-01-01", "2025-12-31"),
]

HORIZONS = [5, 10, 20]


def safe_spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def partial_spearman(x, y, control):
    """rank-residualized partial Spearman."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if m.sum() < 50:
        return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    cr = pd.Series(control[m]).rank().to_numpy()
    X = np.column_stack([np.ones(len(cr)), cr])
    rx = xr - X @ np.linalg.lstsq(X, xr, rcond=None)[0]
    ry = yr - X @ np.linalg.lstsq(X, yr, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def fwd_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def load_asset(asset_name):
    import yfinance as yf

    tkr = ASSETS[asset_name]
    df = yf.download(
        tkr,
        start="2005-01-01",
        end="2025-12-31",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if df is None or len(df) < 1000:
        raise ValueError(f"{asset_name} insufficient")
    close = df["Close"].squeeze().to_numpy(dtype=float)
    valid = np.isfinite(close)
    close = close[valid]
    dates = df.index[valid]
    out = compute_epd_v3(close)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "close": close,
            "zr": out["z_r"],
            "ze": out["z_e"],
        }
    )


def main():
    rows = []
    for asset_name in ASSETS:
        try:
            df = load_asset(asset_name)
        except Exception as e:
            print(f"{asset_name} err: {e}")
            continue

        close = df["close"].to_numpy(dtype=float)
        zr = df["zr"].to_numpy(dtype=float)
        ze = df["ze"].to_numpy(dtype=float)
        dates = df["date"]

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))

        A = np.abs(zr)
        E_abs = np.abs(ze)
        R1 = A
        R6 = np.abs(A - E_abs)

        print(f"\n### {asset_name}")
        print(
            f"  {'period':>11s}  {'target':>7s}  "
            f"{'R1 rho':>9s}  {'R6 rho':>9s}  "
            f"{'R6 partial':>11s}  {'R6>0':>5s}"
        )
        print("  " + "-" * 62)

        for pname, start, end in PERIODS:
            mask_p = (dates >= start) & (dates <= end)
            if mask_p.sum() < 400:
                continue

            for h in HORIZONS:
                fv = fwd_vol(r, h)
                m = (
                    mask_p.to_numpy()
                    & np.isfinite(fv)
                    & np.isfinite(R1)
                    & np.isfinite(R6)
                )
                if m.sum() < 200:
                    continue

                r1_rho = safe_spearman(R1[m], fv[m])
                r6_rho = safe_spearman(R6[m], fv[m])
                # R6 partial: |Zr| 통제 후
                r6_partial = partial_spearman(R6[m], fv[m], R1[m])

                rows.append(
                    {
                        "asset": asset_name,
                        "period": pname,
                        "horizon": h,
                        "R1_rho": r1_rho,
                        "R6_rho": r6_rho,
                        "R6_partial": r6_partial,
                    }
                )

                flag = "✅" if r6_partial > 0 else "❌"
                print(
                    f"  {pname:>11s}  {h:>7d}  "
                    f"{r1_rho:>+9.4f}  {r6_rho:>+9.4f}  "
                    f"{r6_partial:>+11.4f}  {flag:>5s}"
                )

    # ============================================================
    # 요약
    # ============================================================
    out = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("기간별 평균 (8자산)")
    print("=" * 100)
    print(
        f"  {'period':>11s}  {'H':>3s}  "
        f"{'mean R1 rho':>13s}  {'mean R6 rho':>13s}  "
        f"{'mean R6 partial':>16s}  {'R6>0 비율':>10s}"
    )
    print("  " + "-" * 80)

    for pname, _, _ in PERIODS:
        for h in HORIZONS:
            sub = out[(out["period"] == pname) & (out["horizon"] == h)]
            if len(sub) == 0:
                continue
            print(
                f"  {pname:>11s}  {h:>3d}  "
                f"{sub['R1_rho'].mean():>+13.4f}  "
                f"{sub['R6_rho'].mean():>+13.4f}  "
                f"{sub['R6_partial'].mean():>+16.4f}  "
                f"{(sub['R6_partial'] > 0).mean():>9.1%}"
            )

    # 핵심 판정
    print("\n" + "=" * 100)
    print("최종 판정")
    print("=" * 100)

    r6_partials = out["R6_partial"].dropna()
    n_pos = (r6_partials > 0).sum()
    n_total = len(r6_partials)
    mean_part = r6_partials.mean()

    print(f"  R6 partial (전체 {n_total} 케이스):")
    print(f"    mean = {mean_part:+.4f}")
    print(f"    양수 비율 = {n_pos}/{n_total} ({n_pos/n_total:.1%})")
    print(f"    최소값 = {r6_partials.min():+.4f}")
    print(f"    최대값 = {r6_partials.max():+.4f}")

    # 기간별로
    print(f"\n  기간별 R6 partial 평균 (모든 horizon):")
    for pname, _, _ in PERIODS:
        sub = out[out["period"] == pname]
        if len(sub) == 0:
            continue
        pos_ratio = (sub["R6_partial"] > 0).mean()
        print(
            f"    {pname}: mean = {sub['R6_partial'].mean():+.4f}, "
            f"양수 = {pos_ratio:.1%}"
        )

    print()
    if (r6_partials > 0).mean() >= 0.80:
        print("  ✅ R6 시간 안정성 확인 — EPD = ||Zr|-|Ze|| 확정")
    elif (r6_partials > 0).mean() >= 0.60:
        print("  ⚠️ R6 부분 안정 — 조건부 확정")
    else:
        print("  ❌ R6 시간 불안정 — 재검토")

    # 저장
    outdir = ROOT / "docs" / "stage10_outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "time_stability.csv", index=False)
    print(f"\n  CSV → {outdir / 'time_stability.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
