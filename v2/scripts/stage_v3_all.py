"""EPD v3.0 — C1 vs C4 통합 검증.

Stage 3: Distinctiveness (RSI/ATR/MACD/RV20/|ret|/AC20/PE60)
Stage 4: State (4분면 ANOVA)
Stage 5: Robustness (파라미터/기간)
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd_v3, rolling_pe

warnings.filterwarnings("ignore")

TICKERS = [
    ("^GSPC", "SPX", "US"),
    ("^IXIC", "NASDAQ", "US"),
    ("^KS11", "KOSPI", "KR"),
    ("^KQ11", "KOSDAQ", "KR"),
    ("005930.KS", "삼성전자", "KR"),
    ("000660.KS", "SK하이닉스", "KR"),
    ("035420.KS", "NAVER", "KR"),
    ("051910.KS", "LG화학", "KR"),
]

H = 5


def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


# ============================================================
#  기술적 지표 (기존 Stage 3 재사용)
# ============================================================
def rsi(close, n=14):
    close = np.asarray(close, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    out = np.full(len(close), np.nan)
    ag = al = np.nan
    for t in range(n, len(close)):
        if np.isnan(ag):
            ag = gain[t - n + 1 : t + 1].mean()
            al = loss[t - n + 1 : t + 1].mean()
        else:
            ag = (ag * (n - 1) + gain[t]) / n
            al = (al * (n - 1) + loss[t]) / n
        out[t] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def atr(high, low, close, n=14):
    n_ = len(close)
    tr = np.zeros(n_)
    tr[0] = high[0] - low[0]
    for t in range(1, n_):
        tr[t] = max(
            high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1])
        )
    out = np.full(n_, np.nan)
    avg = np.nan
    for t in range(n, n_):
        if np.isnan(avg):
            avg = tr[t - n + 1 : t + 1].mean()
        else:
            avg = (avg * (n - 1) + tr[t]) / n
        out[t] = avg
    return out


def ema(x, span):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    for t in range(len(x)):
        xt = x[t]
        if not np.isfinite(xt):
            out[t] = prev
        else:
            prev = xt if not np.isfinite(prev) else alpha * xt + (1 - alpha) * prev
            out[t] = prev
    return out


def macd_hist(close, f=12, s=26, sig=9):
    return ema(close, f) - ema(close, s) - ema(ema(close, f) - ema(close, s), sig)


def realized_vol(r, w=20):
    out = np.full_like(r, np.nan)
    for t in range(w, len(r)):
        ww = r[t - w : t]
        ww = ww[np.isfinite(ww)]
        if ww.size >= w // 2:
            out[t] = ww.std(ddof=1) * np.sqrt(252)
    return out


def rolling_autocorr(r, w=20):
    out = np.full_like(r, np.nan)
    for t in range(w, len(r)):
        win = r[t - w : t]
        win = win[np.isfinite(win)]
        if win.size < w - 2:
            continue
        x, y = win[:-1], win[1:]
        if x.std() == 0 or y.std() == 0:
            continue
        out[t] = np.corrcoef(x, y)[0, 1]
    return out


def safe_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan
    x, y = a[m], b[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
#  메인
# ============================================================
def main():
    import yfinance as yf

    print("=" * 130)
    print("EPD v3.0 — C1 (뺄셈) vs C4 (곱셈) 통합 검증")
    print("=" * 130)

    rows = []

    for tkr, name, grp in TICKERS:
        try:
            df = yf.download(
                tkr,
                start="2005-01-01",
                end="2025-12-31",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < 1000:
                continue
            o = df["Open"].squeeze().to_numpy(dtype=float)
            h = df["High"].squeeze().to_numpy(dtype=float)
            l = df["Low"].squeeze().to_numpy(dtype=float)
            c = df["Close"].squeeze().to_numpy(dtype=float)
            v = np.isfinite(c)
            o, h, l, c = o[v], h[v], l[v], c[v]
            if c.size < 1000:
                continue
        except Exception as e:
            print(f"{name} err: {e}")
            continue

        r = np.full_like(c, np.nan)
        r[1:] = np.diff(np.log(c))

        out = compute_epd_v3(c)
        c1 = out["C1_raw"]
        c4 = out["C4_raw"]

        fv = forward_vol(r, H)

        # --- Stage 3: Distinctiveness ---
        ind_vals = {
            "RSI": rsi(c, 14),
            "ATR": atr(h, l, c, 14),
            "MACD": np.abs(macd_hist(c)),
            "RV20": realized_vol(r, 20),
            "|ret|": np.abs(r),
            "AC20": np.abs(rolling_autocorr(r, 20)),
            "PE60": rolling_pe(r, w_pe=60, m=3, tau=1),
        }

        print(f"\n### {name} ({grp})")
        print(f"  [Stage 3] corr(EPD_abs, X):")
        print(f"    {'indicator':>10s}  {'C1_abs':>9s}  {'C4_abs':>9s}")
        corr_c1 = {}
        corr_c4 = {}
        for k, val in ind_vals.items():
            cc1 = safe_corr(np.abs(c1), val)
            cc4 = safe_corr(np.abs(c4), val)
            corr_c1[k] = cc1
            corr_c4[k] = cc4
            print(f"    {k:>10s}  {cc1:>+9.4f}  {cc4:>+9.4f}")

        # --- Stage 4 & 5: forward vol 상관 ---
        rho_c1 = spearman(c1, fv)
        rho_c4 = spearman(c4, fv)

        print(f"  [Stage 4/5] corr(EPD, fwd_vol):")
        print(f"    C1 = {rho_c1:+.4f}")
        print(f"    C4 = {rho_c4:+.4f}")

        # max |corr| (Stage 3 판정용)
        mc1 = max(abs(v) for v in corr_c1.values() if np.isfinite(v))
        mc4 = max(abs(v) for v in corr_c4.values() if np.isfinite(v))
        n_low_c1 = sum(1 for v in corr_c1.values() if np.isfinite(v) and abs(v) < 0.5)
        n_low_c4 = sum(1 for v in corr_c4.values() if np.isfinite(v) and abs(v) < 0.5)

        print(
            f"  [요약] max|corr|: C1={mc1:.3f} C4={mc4:.3f}  "
            f"n_low(<0.5): C1={n_low_c1}/7 C4={n_low_c4}/7"
        )

        rows.append(
            {
                "name": name,
                "grp": grp,
                "mc1": mc1,
                "mc4": mc4,
                "n_low_c1": n_low_c1,
                "n_low_c4": n_low_c4,
                "rho_c1": rho_c1,
                "rho_c4": rho_c4,
                "corr_c1": corr_c1,
                "corr_c4": corr_c4,
            }
        )

    # ============================================================
    # 요약
    # ============================================================
    print()
    print("=" * 130)
    print("요약 (8자산)")
    print("=" * 130)

    mc1s = [r["mc1"] for r in rows]
    mc4s = [r["mc4"] for r in rows]
    nlc1 = [r["n_low_c1"] for r in rows]
    nlc4 = [r["n_low_c4"] for r in rows]
    rc1 = [abs(r["rho_c1"]) for r in rows if np.isfinite(r["rho_c1"])]
    rc4 = [abs(r["rho_c4"]) for r in rows if np.isfinite(r["rho_c4"])]

    print(f"\n  {'metric':>20s}  {'C1':>10s}  {'C4':>10s}")
    print("  " + "-" * 44)
    print(f"  {'mean max|corr|':>20s}  {np.mean(mc1s):>10.4f}  {np.mean(mc4s):>10.4f}")
    print(
        f"  {'mean n_low(<0.5)':>20s}  {np.mean(nlc1):>10.2f}  {np.mean(nlc4):>10.2f}"
    )
    print(f"  {'mean |rho_fv|':>20s}  {np.mean(rc1):>10.4f}  {np.mean(rc4):>10.4f}")

    print()
    print("Stage 3 판정 (max|corr| < 0.7 AND n_low >= 4):")
    s3_c1 = sum(1 for r in rows if r["mc1"] < 0.7 and r["n_low_c1"] >= 4)
    s3_c4 = sum(1 for r in rows if r["mc4"] < 0.7 and r["n_low_c4"] >= 4)
    print(f"  C1: {s3_c1}/8 PASS")
    print(f"  C4: {s3_c4}/8 PASS")

    print()
    print("결론:")
    print("  - C1, C4 모두 Stage 3 통과하면 → 둘 다 유지, Stage 5에서 최종 결정")
    print("  - 하나만 통과하면 → 그걸로 확정")


if __name__ == "__main__":
    raise SystemExit(main())
