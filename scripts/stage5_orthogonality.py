"""Stage 5: Incremental Information — |EPD_raw|가 RV 통제 후에도 추가 정보?"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

warnings.filterwarnings("ignore")

W_PE, M, TAU, W_Z = 60, 3, 1, 252
H = 5


def compute_signals(close):
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
    epd_raw = z_r - z_e
    epd_abs = np.abs(epd_raw)
    return epd_abs, r


def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def realized_vol(r, w):
    rv = np.full_like(r, np.nan)
    for t in range(w, len(r)):
        win = r[t - w : t]
        win = win[np.isfinite(win)]
        if win.size >= max(3, w // 2):
            rv[t] = win.std(ddof=1)
    return rv


def nw_ols(y, X, maxlag=10):
    """OLS with Newey-West HAC standard errors.

    y: (n,), X: (n, k) with intercept already included
    returns: beta (k,), se (k,), r2, tstats (k,), pvals (k,)
    """
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else np.nan

    # HAC
    S = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            s = 0.0
            for t in range(n):
                s += X[t, i] * X[t, j] * resid[t] ** 2
            for lag in range(1, maxlag + 1):
                w = 1 - lag / (maxlag + 1)
                for t in range(lag, n):
                    s += (
                        w
                        * (X[t, i] * X[t - lag, j] + X[t - lag, i] * X[t, j])
                        * resid[t]
                        * resid[t - lag]
                    )
            S[i, j] = s
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    tstats = beta / se
    # p-value via normal approx
    from math import erfc, sqrt

    pvals = np.array([erfc(abs(t) / sqrt(2)) for t in tstats])
    return {"beta": beta, "se": se, "r2": r2, "t": tstats, "p": pvals}


def fit_models(epd_abs, r, rv5, rv20, rv60, fv):
    """모델 5개 fit. 공통 mask 사용."""
    mask = (
        np.isfinite(epd_abs)
        & np.isfinite(fv)
        & np.isfinite(rv5)
        & np.isfinite(rv20)
        & np.isfinite(rv60)
    )
    n = mask.sum()
    if n < 200:
        return None

    y = fv[mask]
    e = epd_abs[mask]
    x5 = rv5[mask]
    x20 = rv20[mask]
    x60 = rv60[mask]
    c = np.ones(n)

    models = {
        "M0_const": c[:, None],
        "M1_RV20": np.column_stack([c, x20]),
        "M2_RV20_EPD": np.column_stack([c, x20, e]),
        "M3_HAR": np.column_stack([c, x5, x20, x60]),
        "M4_HAR_EPD": np.column_stack([c, x5, x20, x60, e]),
    }

    out = {}
    for name, X in models.items():
        res = nw_ols(y, X)
        if res is None:
            continue
        res["k"] = X.shape[1]
        res["n"] = n
        out[name] = res
    return out


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

    print("=" * 110)
    print("Stage 5: Incremental Information — |EPD_raw|가 RV 통제 후에도 정보?")
    print("=" * 110)
    print(
        f"{'asset':>14s}  {'grp':>10s}  {'n':>5s}  | "
        f"{'R2_M1':>7s} {'R2_M2':>7s} {'dR2(M2-M1)':>12s} {'p(EPD)':>9s}  | "
        f"{'R2_M3':>7s} {'R2_M4':>7s} {'dR2(M4-M3)':>12s} {'p(EPD)':>9s}"
    )
    print("-" * 130)

    summary = {"KR_stock": [], "KR_index": [], "US_index": []}
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

            epd_abs, r = compute_signals(close)
            fv = forward_vol(r, H)
            rv5 = realized_vol(r, 5)
            rv20 = realized_vol(r, 20)
            rv60 = realized_vol(r, 60)

            fits = fit_models(epd_abs, r, rv5, rv20, rv60, fv)
            if fits is None or "M4_HAR_EPD" not in fits:
                print(f"{name:>14s}  (insufficient)")
                continue

            m1, m2 = fits["M1_RV20"], fits["M2_RV20_EPD"]
            m3, m4 = fits["M3_HAR"], fits["M4_HAR_EPD"]
            dR2_a = m2["r2"] - m1["r2"]
            dR2_b = m4["r2"] - m3["r2"]
            p_epd_a = m2["p"][-1]
            p_epd_b = m4["p"][-1]

            print(
                f"{name:>14s}  {grp:>10s}  {m1['n']:>5d}  | "
                f"{m1['r2']:>7.4f} {m2['r2']:>7.4f} {dR2_a:>12.4f} {p_epd_a:>9.4f}  | "
                f"{m3['r2']:>7.4f} {m4['r2']:>7.4f} {dR2_b:>12.4f} {p_epd_b:>9.4f}"
            )

            rows.append(
                {
                    "name": name,
                    "grp": grp,
                    "r2_m1": m1["r2"],
                    "r2_m2": m2["r2"],
                    "dR2_m2_m1": dR2_a,
                    "p_epd_m2": p_epd_a,
                    "r2_m3": m3["r2"],
                    "r2_m4": m4["r2"],
                    "dR2_m4_m3": dR2_b,
                    "p_epd_m4": p_epd_b,
                }
            )
            summary[grp].append(rows[-1])
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    # 자산군 요약
    print()
    print("=" * 110)
    print("자산군별 요약")
    print("=" * 110)
    print(
        f"{'group':>12s}  {'n':>3s}  {'mean dR2(M2-M1)':>17s}  "
        f"{'frac p<0.05 (M2)':>18s}  {'mean dR2(M4-M3)':>17s}  {'frac p<0.05 (M4)':>18s}"
    )
    print("-" * 110)
    for grp in ["KR_stock", "KR_index", "US_index"]:
        sub = summary[grp]
        if not sub:
            continue
        dR2_a = [r["dR2_m2_m1"] for r in sub]
        dR2_b = [r["dR2_m4_m3"] for r in sub]
        p_a_ok = sum(1 for r in sub if r["p_epd_m2"] < 0.05) / len(sub)
        p_b_ok = sum(1 for r in sub if r["p_epd_m4"] < 0.05) / len(sub)
        print(
            f"{grp:>12s}  {len(sub):>3d}  {np.mean(dR2_a):>17.4f}  "
            f"{p_a_ok:>18.1%}  {np.mean(dR2_b):>17.4f}  {p_b_ok:>18.1%}"
        )

    print()
    print("판정:")
    print("  PASS 조건 A: mean dR2(M2-M1) > 0.01 AND p<0.05 비율 >= 70%")
    print("  PASS 조건 B: mean dR2(M4-M3) > 0.005 AND p<0.05 비율 >= 60%")

    # 리포트
    rpt = ROOT / "docs" / "incremental_report.md"
    lines = [
        "# Incremental Report — Stage 5",
        "",
        f"모델: M1=RV20, M2=RV20+|EPD_raw|, M3=HAR(5/20/60), M4=HAR+|EPD_raw|",
        f"표준오차: Newey-West HAC (maxlag=10)",
        "",
        "## 자산별 결과",
        "",
        "| Asset | Group | n | R2(M1) | R2(M2) | dR2(M2-M1) | p(EPD,M2) | "
        "R2(M3) | R2(M4) | dR2(M4-M3) | p(EPD,M4) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['grp']} | - | "
            f"{r['r2_m1']:.4f} | {r['r2_m2']:.4f} | {r['dR2_m2_m1']:.4f} | "
            f"{r['p_epd_m2']:.4f} | {r['r2_m3']:.4f} | {r['r2_m4']:.4f} | "
            f"{r['dR2_m4_m3']:.4f} | {r['p_epd_m4']:.4f} |"
        )
    lines.append("")
    lines.append("## 자산군 요약")
    lines.append("")
    lines.append(
        "| Group | n | mean dR2(M2-M1) | frac p<0.05 (M2) | "
        "mean dR2(M4-M3) | frac p<0.05 (M4) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for grp in ["KR_stock", "KR_index", "US_index"]:
        sub = summary[grp]
        if not sub:
            continue
        dR2_a = [r["dR2_m2_m1"] for r in sub]
        dR2_b = [r["dR2_m4_m3"] for r in sub]
        p_a_ok = sum(1 for r in sub if r["p_epd_m2"] < 0.05) / len(sub)
        p_b_ok = sum(1 for r in sub if r["p_epd_m4"] < 0.05) / len(sub)
        lines.append(
            f"| {grp} | {len(sub)} | {np.mean(dR2_a):.4f} | "
            f"{p_a_ok:.1%} | {np.mean(dR2_b):.4f} | {p_b_ok:.1%} |"
        )
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    raise SystemExit(main())
