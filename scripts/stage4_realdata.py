"""Stage 4: Real Data Discovery — H2 (|EPD| -> forward vol).

- 12자산 (한국 개별 8 + 한국 지수 2 + 미국 지수 2)
- |EPD_100 - 50| 5분위별 forward vol
- 절대 임계값 (20/30) 기반 분석
- 기간별 (2005-12, 13-19, 20-25) 안정성
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd  # noqa: E402

warnings.filterwarnings("ignore")

H = 5
THR_MOD = 20  # 중간 임계값 (|EPD-50| > 20)
THR_HI = 30  # 높은 임계값


def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def quintile_means(x, y, n_q=5):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < n_q * 30:
        return None
    a, b = x[m], y[m]
    qs = np.quantile(a, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (a >= qs[i]) & (a < qs[i + 1])
        if mm.sum() < 20:
            return None
        means.append(b[mm].mean())
    return np.array(means)


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
        ("373220.KS", "LG엔솔", "KR_stock"),
        ("^KS11", "KOSPI지수", "KR_index"),
        ("^KQ11", "KOSDAQ지수", "KR_index"),
        ("^GSPC", "SPX", "US_index"),
        ("^IXIC", "NASDAQ", "US_index"),
    ]

    periods = [
        ("full", "2005-01-01", "2025-12-31"),
        ("2005-2012", "2005-01-01", "2012-12-31"),
        ("2013-2019", "2013-01-01", "2019-12-31"),
        ("2020-2025", "2020-01-01", "2025-12-31"),
    ]

    print("=" * 110)
    print("Stage 4: Real Data Discovery — H2 (|EPD-50| -> forward 5d vol)")
    print(f"임계값: mod={THR_MOD}, hi={THR_HI}")
    print("=" * 110)

    # ------------------ A. 전체 기간 quintile + 임계값 ------------------
    print("\n### A. 전체 기간 |EPD-50| 5분위 forward vol + 임계값별\n")
    print(
        f"{'asset':>14s}  {'n':>5s}  | "
        f"{'Q1':>9s} {'Q2':>9s} {'Q3':>9s} {'Q4':>9s} {'Q5':>9s}  | "
        f"{'<20':>9s} {'20-30':>9s} {'>30':>9s}  {'Q5/Q1':>6s}"
    )
    print("-" * 130)

    rows_A = []
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

            out = compute_epd(close)
            epd_100 = out["epd_100"]
            r = np.full_like(close, np.nan)
            r[1:] = np.diff(np.log(close))
            fv = forward_vol(r, H)

            intensity = np.abs(epd_100 - 50.0)
            q = quintile_means(intensity, fv)
            if q is None:
                print(f"{name:>14s}  {len(close):>5d}  (insufficient)")
                continue

            # 임계값별 평균 vol
            m = np.isfinite(intensity) & np.isfinite(fv)
            intens_v = intensity[m]
            fv_v = fv[m]

            v_lo = (
                fv_v[intens_v < THR_MOD].mean()
                if (intens_v < THR_MOD).sum() > 20
                else np.nan
            )
            v_mid = (
                fv_v[(intens_v >= THR_MOD) & (intens_v < THR_HI)].mean()
                if ((intens_v >= THR_MOD) & (intens_v < THR_HI)).sum() > 20
                else np.nan
            )
            v_hi = (
                fv_v[intens_v >= THR_HI].mean()
                if (intens_v >= THR_HI).sum() > 20
                else np.nan
            )

            ratio = q[4] / q[0] if q[0] > 0 else np.nan

            print(
                f"{name:>14s}  {len(close):>5d}  | "
                f"{q[0]:>9.6f} {q[1]:>9.6f} {q[2]:>9.6f} {q[3]:>9.6f} {q[4]:>9.6f}  | "
                f"{v_lo:>9.6f} {v_mid:>9.6f} {v_hi:>9.6f}  {ratio:>6.2f}"
            )

            rows_A.append(
                {
                    "name": name,
                    "grp": grp,
                    "n": len(close),
                    "q": q,
                    "ratio": ratio,
                    "v_lo": v_lo,
                    "v_mid": v_mid,
                    "v_hi": v_hi,
                }
            )
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    # 요약
    print("\n### 자산군별 요약\n")
    print(f"{'group':>14s}  {'n':>3s}  {'mean Q5/Q1':>12s}  {'mean >30/<20':>15s}")
    print("-" * 60)
    for grp in ["KR_stock", "KR_index", "US_index"]:
        sub = [r for r in rows_A if r["grp"] == grp]
        if not sub:
            continue
        ratios = [r["ratio"] for r in sub if np.isfinite(r["ratio"])]
        # >30 / <20 비율
        r_hilo = []
        for r in sub:
            if np.isfinite(r["v_lo"]) and np.isfinite(r["v_hi"]) and r["v_lo"] > 0:
                r_hilo.append(r["v_hi"] / r["v_lo"])
        print(
            f"{grp:>14s}  {len(sub):>3d}  "
            f"{np.mean(ratios):>12.3f}  {np.mean(r_hilo) if r_hilo else np.nan:>15.3f}"
        )

    # ------------------ B. 기간별 ------------------
    print("\n### B. 기간별 Q5/Q1 비율\n")
    print(
        f"{'asset':>14s}  | "
        f"{'2005-2012':>12s}  | {'2013-2019':>12s}  | {'2020-2025':>12s}"
    )
    print("-" * 80)

    for tkr, name, grp in tickers:
        row = []
        for pname, start, end in periods[1:]:
            try:
                df = yf.download(
                    tkr,
                    start=start,
                    end=end,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if df is None or len(df) < 400:
                    row.append("     n/a")
                    continue
                close = df["Close"].squeeze().to_numpy(dtype=float)
                close = close[np.isfinite(close)]
                if close.size < 400:
                    row.append("     n/a")
                    continue
                out = compute_epd(close)
                epd_100 = out["epd_100"]
                r = np.full_like(close, np.nan)
                r[1:] = np.diff(np.log(close))
                fv = forward_vol(r, H)
                intensity = np.abs(epd_100 - 50.0)
                q = quintile_means(intensity, fv)
                if q is None or q[0] <= 0:
                    row.append("     n/a")
                else:
                    row.append(f"{q[4]/q[0]:>12.2f}")
            except Exception:
                row.append("     err")
        print(f"{name:>14s}  | " + "  | ".join(row))

    # ------------------ 리포트 저장 ------------------
    rpt = ROOT / "docs" / "realdata_report.md"
    lines = [
        "# Real Data Report — Stage 4 (H2, |EPD-50| -> forward vol)",
        "",
        f"- H (forward window): {H}일",
        f"- 임계값: mod={THR_MOD}, hi={THR_HI}",
        f"- 자산: 한국 개별 8, 한국 지수 2, 미국 지수 2",
        "",
        "## A. 전체 기간 quintile + 임계값",
        "",
        "| Asset | Group | n | Q1 | Q2 | Q3 | Q4 | Q5 | Q5/Q1 | <20 | 20-30 | >30 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_A:
        q = r["q"]
        lines.append(
            f"| {r['name']} | {r['grp']} | {r['n']} | "
            f"{q[0]:.6f} | {q[1]:.6f} | {q[2]:.6f} | {q[3]:.6f} | {q[4]:.6f} | "
            f"{r['ratio']:.3f} | "
            f"{r['v_lo']:.6f} | {r['v_mid']:.6f} | {r['v_hi']:.6f} |"
        )

    lines.append("")
    lines.append("## 자산군 요약")
    lines.append("")
    lines.append("| Group | n | mean Q5/Q1 | mean >30/<20 |")
    lines.append("|---|---|---|---|")
    for grp in ["KR_stock", "KR_index", "US_index"]:
        sub = [r for r in rows_A if r["grp"] == grp]
        if not sub:
            continue
        ratios = [r["ratio"] for r in sub if np.isfinite(r["ratio"])]
        r_hilo = []
        for r in sub:
            if np.isfinite(r["v_lo"]) and np.isfinite(r["v_hi"]) and r["v_lo"] > 0:
                r_hilo.append(r["v_hi"] / r["v_lo"])
        m1 = np.mean(ratios) if ratios else np.nan
        m2 = np.mean(r_hilo) if r_hilo else np.nan
        lines.append(f"| {grp} | {len(sub)} | {m1:.3f} | {m2:.3f} |")

    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    raise SystemExit(main())
