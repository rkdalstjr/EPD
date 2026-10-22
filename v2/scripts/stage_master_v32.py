"""A (1일 Z_r) vs D (3일 RMS Z_r) — Stage 1,2,3,9,10,11 통합 재검증.

한 번에 두 후보를 side-by-side 비교.
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd, rolling_pe

warnings.filterwarnings("ignore")

ASSETS = {
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "Samsung": "005930.KS",
    "SK_Hynix": "000660.KS",
    "NAVER": "035420.KS",
    "LG_Chem": "051910.KS",
}
MODES = {"A": "daily", "D": "3day_rms"}
HORIZONS = [5, 10, 20]
PERIODS = [
    ("2005-2010", "2005-01-01", "2010-12-31"),
    ("2010-2015", "2011-01-01", "2015-12-31"),
    ("2015-2020", "2016-01-01", "2020-12-31"),
    ("2020-2025", "2021-01-01", "2025-12-31"),
]


# ---- 기술적 지표 ----
def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    out = np.full(len(c), np.nan)
    ag = al = np.nan
    for t in range(n, len(c)):
        if np.isnan(ag):
            ag, al = g[t - n + 1 : t + 1].mean(), l[t - n + 1 : t + 1].mean()
        else:
            ag = (ag * (n - 1) + g[t]) / n
            al = (al * (n - 1) + l[t]) / n
        out[t] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def ema(x, span):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    a = 2 / (span + 1)
    prev = np.nan
    for t in range(len(x)):
        if not np.isfinite(x[t]):
            out[t] = prev
            continue
        prev = x[t] if not np.isfinite(prev) else a * x[t] + (1 - a) * prev
        out[t] = prev
    return out


def macd(c):
    return np.abs(ema(c, 12) - ema(c, 26) - ema(ema(c, 12) - ema(c, 26), 9))


def rv(r, w=20):
    out = np.full_like(r, np.nan)
    for t in range(w, len(r)):
        v = r[t - w : t]
        v = v[np.isfinite(v)]
        if v.size >= w // 2:
            out[t] = v.std(ddof=1)
    return out


def ac1(r, w=20):
    out = np.full_like(r, np.nan)
    for t in range(w, len(r)):
        v = r[t - w : t]
        v = v[np.isfinite(v)]
        if v.size >= w - 2 and v.std() > 0:
            out[t] = np.corrcoef(v[:-1], v[1:])[0, 1]
    return out


def fwd_vol(r, h):
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(n - h):
        v = r[t + 1 : t + 1 + h]
        v = v[np.isfinite(v)]
        if v.size >= 3:
            out[t] = v.std(ddof=1)
    return out


def safe_spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def part_spear(x, y, c):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if m.sum() < 200:
        return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    cr = pd.Series(c[m]).rank().to_numpy()
    X = np.column_stack([np.ones(len(cr)), cr])
    rx = xr - X @ np.linalg.lstsq(X, xr, rcond=None)[0]
    ry = yr - X @ np.linalg.lstsq(X, yr, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def q_ratio(x, y, nq=5):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return np.nan
    a, b = x[m], y[m]
    qs = np.quantile(a, np.linspace(0, 1, nq + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(nq):
        mm = (a >= qs[i]) & (a < qs[i + 1])
        if mm.sum() < 20:
            return np.nan
        means.append(b[mm].mean())
    return float(means[-1] / means[0]) if means[0] > 0 else np.nan


def load(tkr):
    import yfinance as yf

    df = yf.download(
        tkr, start="2005-01-01", end="2025-12-31", progress=False, auto_adjust=True
    )
    if df is None or len(df) < 1000:
        return None, None
    c = df["Close"].squeeze().to_numpy(float)
    v = np.isfinite(c)
    return c[v], df.index[v]


def main():
    print("=" * 100)
    print("EPD v3.2 후보 D 검증 — A(1일) vs D(3일 RMS)")
    print("=" * 100)

    # 결과 수집
    rows_stage2 = []  # Q5/Q1 (Stage 2)
    rows_stage3 = []  # corr with indicators (Stage 3)
    rows_stage9 = []  # sign partial (Stage 9)
    rows_stage10 = []  # time stability (Stage 10)
    rows_stage11 = []  # distribution (Stage 11)

    for name, tkr in ASSETS.items():
        print(f"\n[{name}]")
        close, dates = load(tkr)
        if close is None:
            print("  skip")
            continue

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))

        # 지표 (모드 무관)
        ind = {
            "RSI": rsi(close),
            "MACD": macd(close),
            "RV20": rv(r, 20),
            "AC20": ac1(r, 20),
            "PE60": rolling_pe(r, w_pe=60, m=3, tau=1),
        }

        for label, mode in MODES.items():
            out = compute_epd(close, z_r_mode=mode)
            epd_mag = out["epd_mag"]
            z_r = out["z_r"]
            z_e = out["z_e"]

            # ---- Stage 2: Q5/Q1 ----
            for h in [5]:
                fv = fwd_vol(r, h)
                qr = q_ratio(epd_mag, fv)
                rows_stage2.append({"asset": name, "mode": label, "h": h, "Q5_Q1": qr})

            # ---- Stage 3: distinctiveness ----
            for k, v in ind.items():
                m = np.isfinite(epd_mag) & np.isfinite(v)
                if m.sum() < 200:
                    continue
                corr = safe_spear(epd_mag[m], v[m])
                rows_stage3.append(
                    {"asset": name, "mode": label, "indicator": k, "corr": corr}
                )

            # ---- Stage 9: Ze sign partial ----
            sign_ze = np.sign(z_e)
            for h in [5, 10, 20]:
                fv = fwd_vol(r, h)
                # sign partial (|Zr|, |Ze| 통제 후)
                # rank 기반 partial은 2-control 구현 필요 → 근사: |Ze| 통제 후
                p = part_spear(sign_ze, fv, np.abs(z_r))
                rows_stage9.append(
                    {"asset": name, "mode": label, "h": h, "sign_partial": p}
                )

            # ---- Stage 10: Time stability ----
            for pname, start, end in PERIODS:
                mask = (dates >= start) & (dates <= end)
                if mask.sum() < 400:
                    continue
                for h in [5]:
                    fv = fwd_vol(r, h)
                    m = mask & np.isfinite(fv) & np.isfinite(epd_mag)
                    if m.sum() < 300:
                        continue
                    p = part_spear(epd_mag[m], fv[m], np.abs(z_r[m]))
                    rows_stage10.append(
                        {
                            "asset": name,
                            "mode": label,
                            "period": pname,
                            "h": h,
                            "partial": p,
                        }
                    )

            # ---- Stage 11: Distribution ----
            v = epd_mag[np.isfinite(epd_mag)]
            if len(v) > 100:
                rows_stage11.append(
                    {
                        "asset": name,
                        "mode": label,
                        "mean": float(v.mean()),
                        "std": float(v.std(ddof=1)),
                        "Q50": float(np.percentile(v, 50)),
                        "Q75": float(np.percentile(v, 75)),
                        "Q95": float(np.percentile(v, 95)),
                    }
                )

    # ============================================================
    # 요약
    # ============================================================
    print("\n" + "=" * 100)
    print("Stage 2 — Q5/Q1 (forward vol)")
    print("=" * 100)
    df2 = pd.DataFrame(rows_stage2)
    piv2 = df2.pivot_table(
        index="asset", columns="mode", values="Q5_Q1", aggfunc="mean"
    )
    print(piv2.round(3).to_string())
    print(f"\nA mean: {piv2['A'].mean():.3f}  |  D mean: {piv2['D'].mean():.3f}")

    print("\n" + "=" * 100)
    print("Stage 3 — Distinctiveness (corr with existing indicators)")
    print("=" * 100)
    df3 = pd.DataFrame(rows_stage3)
    piv3 = df3.pivot_table(
        index=["asset", "indicator"], columns="mode", values="corr", aggfunc="mean"
    )
    print(piv3.round(3).to_string())
    print(f"\nmax|corr| per mode:")
    print(piv3.abs().groupby(level="indicator").max().round(3).to_string())

    print("\n" + "=" * 100)
    print("Stage 9 — Ze sign partial (should be ≈ 0)")
    print("=" * 100)
    df9 = pd.DataFrame(rows_stage9)
    print(
        df9.groupby(["mode", "h"])["sign_partial"]
        .agg(["mean", "std", lambda x: (x.abs() > 0.05).mean()])
        .round(4)
        .to_string()
    )

    print("\n" + "=" * 100)
    print("Stage 10 — Time stability")
    print("=" * 100)
    df10 = pd.DataFrame(rows_stage10)
    piv10 = df10.pivot_table(
        index=["asset", "period"], columns="mode", values="partial", aggfunc="mean"
    )
    print(piv10.round(3).to_string())

    print("\n양수 비율 (전체 케이스):")
    print(
        df10.groupby("mode")["partial"]
        .agg(mean="mean", pos_share=lambda x: (x > 0).mean())
        .round(3)
        .to_string()
    )

    print("\n" + "=" * 100)
    print("Stage 11 — Distribution (EPD_mag)")
    print("=" * 100)
    df11 = pd.DataFrame(rows_stage11)
    print(
        df11.groupby("mode")[["mean", "std", "Q50", "Q75", "Q95"]]
        .mean()
        .round(3)
        .to_string()
    )

    # ---- 최종 판정 ----
    print("\n" + "=" * 100)
    print("최종 판정")
    print("=" * 100)

    a_q = piv2["A"].mean()
    d_q = piv2["D"].mean()
    a_par = df10[df10["mode"] == "A"]["partial"]
    d_par = df10[df10["mode"] == "D"]["partial"]

    checks = {
        "Stage 2 (D Q5/Q1 > 1.10)": d_q > 1.10,
        "Stage 3 (D max|corr| < 0.7)": piv3["D"].abs().max() < 0.7,
        "Stage 9 (D sign |partial| < 0.05)": abs(
            df9[df9["mode"] == "D"]["sign_partial"].mean()
        )
        < 0.05,
        "Stage 10 (D partial > 0 비율 ≥ 80%)": (d_par > 0).mean() >= 0.80,
        "D > A (Stage 2)": d_q >= a_q,
        "D > A (Stage 10)": d_par.mean() >= a_par.mean(),
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'}  {k}")

    all_pass = all(checks.values())
    print(f"\n=== EPD v3.2 후보 D: {'확정 가능' if all_pass else '추가 검토 필요'} ===")


if __name__ == "__main__":
    raise SystemExit(main())
