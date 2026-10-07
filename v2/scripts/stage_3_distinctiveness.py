"""Stage 3 — Distinctiveness.

질문: EPD가 기존 지표와 다른 정보를 담고 있는가?

기존 지표:
  - RSI(14)
  - ATR(14)
  - MACD(12,26,9) histogram
  - Realized Vol(20)
  - |return| (1일)
  - Autocorr(20)
  - Permutation Entropy(60)

측정: corr(EPD_abs, 기존지표) — Pearson + Spearman
  (Pearson만 Stage 3 판정에 사용)

PASS:
  - 모든 기존 지표와 |corr| < 0.7 (Pearson)
  - 최소 4개 지표에서 |corr| < 0.5
  - 자산군별 통과율 >= 70%

FAIL (즉시 폐기): |corr| > 0.85 인 지표 존재
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd, rolling_pe

warnings.filterwarnings("ignore")

TICKERS = [
    ("^GSPC", "SPX", "US_index"),
    ("^IXIC", "NASDAQ", "US_index"),
    ("^KS11", "KOSPI", "KR_index"),
    ("^KQ11", "KOSDAQ", "KR_index"),
    ("005930.KS", "삼성전자", "KR_stock"),
    ("000660.KS", "SK하이닉스", "KR_stock"),
    ("005380.KS", "현대차", "KR_stock"),
    ("035420.KS", "NAVER", "KR_stock"),
    ("051910.KS", "LG화학", "KR_stock"),
    ("005490.KS", "POSCO홀딩스", "KR_stock"),
    ("068270.KS", "셀트리온", "KR_stock"),
    ("373220.KS", "LG엔솔", "KR_stock"),
]

GROUP_PASS_RATE = 0.70
CORR_FAIL_THRESHOLD = 0.85  # 즉시 폐기
CORR_LOW_THRESHOLD = 0.50  # 낮은 상관 개수
N_LOW_CORR_MIN = 4  # 최소 4개 이상 낮은 상관


# ---- 기술적 지표 계산 ----
def rsi(close, n=14):
    close = np.asarray(close, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    out = np.full(len(close), np.nan)
    avg_g = avg_l = np.nan
    for t in range(n, len(close)):
        if np.isnan(avg_g):
            avg_g = gain[t - n + 1 : t + 1].mean()
            avg_l = loss[t - n + 1 : t + 1].mean()
        else:
            avg_g = (avg_g * (n - 1) + gain[t]) / n
            avg_l = (avg_l * (n - 1) + loss[t]) / n
        if avg_l == 0:
            out[t] = 100.0
        else:
            rs = avg_g / avg_l
            out[t] = 100.0 - 100.0 / (1.0 + rs)
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
            prev = xt if not np.isfinite(prev) else alpha * xt + (1.0 - alpha) * prev
            out[t] = prev
    return out


def macd_hist(close, fast=12, slow=26, sig=9):
    close = np.asarray(close, dtype=float)
    m_line = ema(close, fast) - ema(close, slow)
    s_line = ema(m_line, sig)
    return m_line - s_line


def realized_vol(r, w=20):
    out = np.full_like(r, np.nan)
    for t in range(w, len(r)):
        win = r[t - w : t]
        win = win[np.isfinite(win)]
        if win.size >= max(3, w // 2):
            out[t] = win.std(ddof=1) * np.sqrt(252)
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


def safe_corr(a, b, method="pearson"):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan
    x, y = a[m], b[m]
    if method == "pearson":
        if x.std() == 0 or y.std() == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    else:  # spearman
        xr = np.argsort(np.argsort(x))
        yr = np.argsort(np.argsort(y))
        return float(np.corrcoef(xr, yr)[0, 1])


def main():
    import yfinance as yf

    print("=" * 120)
    print("Stage 3 — Distinctiveness")
    print(
        f"PASS: 모든 지표 |corr|<{CORR_FAIL_THRESHOLD}, "
        f"최소 {N_LOW_CORR_MIN}개 지표 |corr|<{CORR_LOW_THRESHOLD}"
    )
    print("=" * 120)

    indicators = ["RSI", "ATR", "MACD", "RV20", "|ret|", "AC20", "PE60"]
    all_rows = []

    print(
        f"\n{'asset':>14s}  {'group':>10s}  | "
        + "  ".join(f"{k:>7s}" for k in indicators)
        + f"  | {'max|r|':>7s}  {'n_low':>5s}  {'pass':>5s}"
    )
    print("-" * 130)

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
            if df is None or len(df) < 500:
                continue
            o = df["Open"].squeeze().to_numpy(dtype=float)
            h = df["High"].squeeze().to_numpy(dtype=float)
            l = df["Low"].squeeze().to_numpy(dtype=float)
            c = df["Close"].squeeze().to_numpy(dtype=float)
            valid = np.isfinite(c)
            h, l, c = h[valid], l[valid], c[valid]
            if c.size < 500:
                continue
        except Exception as e:
            print(f"{name:>14s}  err: {e}")
            continue

        r = np.full_like(c, np.nan)
        r[1:] = np.diff(np.log(c))

        epd_out = compute_epd(c)
        epd_abs = epd_out["epd_abs"]

        ind_vals = {
            "RSI": rsi(c, 14),
            "ATR": atr(h, l, c, 14),
            "MACD": np.abs(macd_hist(c)),
            "RV20": realized_vol(r, 20),
            "|ret|": np.abs(r),
            "AC20": np.abs(rolling_autocorr(r, 20)),
            "PE60": rolling_pe(r, w_pe=60, m=3, tau=1),
        }

        corrs = {}
        for k, v in ind_vals.items():
            corrs[k] = safe_corr(epd_abs, v, method="pearson")

        finite_corrs = [abs(v) for v in corrs.values() if np.isfinite(v)]
        max_corr = max(finite_corrs) if finite_corrs else np.nan
        n_low = sum(1 for v in finite_corrs if v < CORR_LOW_THRESHOLD)

        passed = (max_corr < CORR_FAIL_THRESHOLD) and (n_low >= N_LOW_CORR_MIN)

        def fmt(v):
            return f"{v:>7.3f}" if np.isfinite(v) else f"{'n/a':>7s}"

        row_str = "  ".join(fmt(corrs[k]) for k in indicators)
        print(
            f"{name:>14s}  {grp:>10s}  | {row_str}  | "
            f"{max_corr:>7.3f}  {n_low:>5d}  {'✅' if passed else '❌':>5s}"
        )

        all_rows.append(
            {
                "name": name,
                "grp": grp,
                "corrs": corrs,
                "max_corr": max_corr,
                "n_low": n_low,
                "passed": passed,
            }
        )

    # ----- 자산군별 요약 -----
    print()
    print("=" * 120)
    print("자산군별 요약")
    print("=" * 120)
    print(
        f"{'group':>12s}  {'n':>3s}  "
        f"{'mean max|r|':>12s}  {'mean n_low':>11s}  "
        f"{'통과율':>8s}  {'판정':>6s}"
    )
    print("-" * 80)

    group_res = {}
    for grp in ["US_index", "KR_index", "KR_stock"]:
        sub = [r for r in all_rows if r["grp"] == grp]
        if not sub:
            continue
        maxcs = [r["max_corr"] for r in sub if np.isfinite(r["max_corr"])]
        nlows = [r["n_low"] for r in sub]
        npass = sum(1 for r in sub if r["passed"])
        frac = npass / len(sub)
        ok = frac >= GROUP_PASS_RATE
        group_res[grp] = {"frac": frac, "ok": ok}
        print(
            f"{grp:>12s}  {len(sub):>3d}  "
            f"{np.mean(maxcs):>12.3f}  {np.mean(nlows):>11.1f}  "
            f"{frac:>7.1%}  {'✅' if ok else '❌':>6s}"
        )

    overall = all(g["ok"] for g in group_res.values())
    print()
    print(f"=== Stage 3 Result: {'PASS' if overall else 'FAIL'} ===")

    # ----- 리포트 -----
    rpt = ROOT / "docs" / "STAGE_3_DISTINCTIVENESS.md"
    L = [
        "# Stage 3 — Distinctiveness",
        "",
        f"- PASS 기준: 모든 \\|corr\\| < {CORR_FAIL_THRESHOLD}, "
        f"최소 {N_LOW_CORR_MIN}개 지표 \\|corr\\| < {CORR_LOW_THRESHOLD}",
        f"- 자산군 통과: {GROUP_PASS_RATE:.0%}",
        "",
        "## 상관 행렬 (Pearson, |corr(EPD_abs, X)|)",
        "",
        "| Asset | Group | "
        + " | ".join(indicators)
        + " | max\\|r\\| | n_low | Pass |",
        "|---" * (len(indicators) + 5) + "|",
    ]
    for r in all_rows:
        vals = " | ".join(
            f"{r['corrs'][k]:.3f}" if np.isfinite(r["corrs"][k]) else "n/a"
            for k in indicators
        )
        L.append(
            f"| {r['name']} | {r['grp']} | {vals} | "
            f"{r['max_corr']:.3f} | {r['n_low']} | "
            f"{'✅' if r['passed'] else '❌'} |"
        )

    L += [
        "",
        "## 자산군 요약",
        "",
        "| Group | n | mean max\\|r\\| | mean n_low | 통과율 | 판정 |",
        "|---|---|---|---|---|---|",
    ]
    for grp, g in group_res.items():
        sub = [r for r in all_rows if r["grp"] == grp]
        maxcs = [r["max_corr"] for r in sub if np.isfinite(r["max_corr"])]
        nlows = [r["n_low"] for r in sub]
        L.append(
            f"| {grp} | {len(sub)} | {np.mean(maxcs):.3f} | "
            f"{np.mean(nlows):.1f} | {g['frac']:.1%} | "
            f"{'✅' if g['ok'] else '❌'} |"
        )

    L += ["", f"**Overall**: {'PASS' if overall else 'FAIL'}"]
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {rpt}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
