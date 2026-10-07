"""Stage 6: Walk-Forward Validation.

질문: |EPD_raw| -> forward 5d vol 관계가 OOS에서도 유지되는가?
- 전략/수익률/비용 없음.
- 순수 통계 검증.

기간:
  Train: 2005-2014
  Val:   2015-2017
  Test:  2018-2025

지표:
  - Q5/Q1: |EPD_raw| 상위 20% vs 하위 20%의 forward vol 비율
  - Spearman rho: |EPD_raw| vs forward vol (단조성)
  - 판정: Test에서 Q5/Q1 > 1.10 AND rho > 0.1
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

warnings.filterwarnings("ignore")

W_PE, M, TAU, W_Z = 60, 3, 1, 252
H = 5

PERIODS = [
    ("Train", "2005-01-01", "2014-12-31"),
    ("Val", "2015-01-01", "2017-12-31"),
    ("Test", "2018-01-01", "2025-12-31"),
]


def compute_epd_abs_and_r(close):
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
    epd_abs = np.abs(z_r - z_e)
    return epd_abs, r


def forward_vol(r, h=H):
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


def q_ratio(epd_abs, fv, q_lo=0.2, q_hi=0.8):
    m = np.isfinite(epd_abs) & np.isfinite(fv)
    if m.sum() < 200:
        return np.nan
    e, f = epd_abs[m], fv[m]
    lo = np.quantile(e, q_lo)
    hi = np.quantile(e, q_hi)
    low_mask = e <= lo
    high_mask = e >= hi
    if low_mask.sum() < 20 or high_mask.sum() < 20:
        return np.nan
    return float(f[high_mask].mean() / f[low_mask].mean())


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

    print("=" * 120)
    print("Stage 6: Walk-Forward — |EPD_raw| -> forward 5d vol 관계의 OOS 안정성")
    print("=" * 120)
    print(
        f"{'asset':>14s} {'grp':>10s} | "
        f"{'Train Q5/Q1':>12s} {'Val Q5/Q1':>11s} {'Test Q5/Q1':>12s} | "
        f"{'Train rho':>10s} {'Val rho':>9s} {'Test rho':>10s} | "
        f"{'판정':>6s}"
    )
    print("-" * 120)

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
            if df is None or len(df) < 1000:
                continue
            close_all = df["Close"].squeeze().to_numpy(dtype=float)
            idx = df.index
            valid = np.isfinite(close_all)
            close_all = close_all[valid]
            idx = idx[valid]
            if close_all.size < 1000:
                continue
        except Exception as e:
            print(f"{name:>14s}  err: {e}")
            continue

        # 전체 기간에 대해 EPD와 forward vol 계산 (한 번만)
        epd_abs, r = compute_epd_abs_and_r(close_all)
        fv = forward_vol(r, H)

        # 기간별 슬라이스
        res = {}
        for pname, start, end in PERIODS:
            mask = (idx >= start) & (idx <= end)
            if mask.sum() < 300:
                res[pname] = (np.nan, np.nan)
                continue
            e_p = epd_abs[mask]
            f_p = fv[mask]
            qr = q_ratio(e_p, f_p)
            rho = spearman(e_p, f_p)
            res[pname] = (qr, rho)

        # 판정: Test Q5/Q1 > 1.10 AND Test rho > 0.1
        t_qr, t_rho = res.get("Test", (np.nan, np.nan))
        if np.isfinite(t_qr) and np.isfinite(t_rho):
            passed = (t_qr > 1.10) and (t_rho > 0.1)
        else:
            passed = False

        def fmt(v):
            return f"{v:>12.3f}" if np.isfinite(v) else f"{'n/a':>12s}"

        def fmt2(v):
            return f"{v:>10.3f}" if np.isfinite(v) else f"{'n/a':>10s}"

        def fmt3(v):
            return f"{v:>9.3f}" if np.isfinite(v) else f"{'n/a':>9s}"

        def fmt4(v):
            return f"{v:>11.3f}" if np.isfinite(v) else f"{'n/a':>11s}"

        tr_qr, tr_rho = res["Train"]
        va_qr, va_rho = res["Val"]
        te_qr, te_rho = res["Test"]

        flag = "PASS" if passed else "FAIL"
        print(
            f"{name:>14s} {grp:>10s} | "
            f"{fmt(tr_qr)} {fmt4(va_qr)} {fmt(te_qr)} | "
            f"{fmt2(tr_rho)} {fmt3(va_rho)} {fmt2(te_rho)} | "
            f"{flag:>6s}"
        )

        rows.append(
            {
                "name": name,
                "grp": grp,
                "train_qr": tr_qr,
                "val_qr": va_qr,
                "test_qr": te_qr,
                "train_rho": tr_rho,
                "val_rho": va_rho,
                "test_rho": te_rho,
                "passed": passed,
            }
        )

    # 자산군별 요약
    print()
    print("=" * 120)
    print("자산군별 요약")
    print("=" * 120)
    print(
        f"{'group':>12s}  {'n':>3s}  "
        f"{'mean Train Q5/Q1':>17s}  {'mean Val Q5/Q1':>15s}  {'mean Test Q5/Q1':>16s}  "
        f"{'PASS':>6s}"
    )
    print("-" * 120)
    for grp in ["KR_stock", "KR_index", "US_index"]:
        sub = [r for r in rows if r["grp"] == grp]
        if not sub:
            continue
        t_qr = [r["train_qr"] for r in sub if np.isfinite(r["train_qr"])]
        v_qr = [r["val_qr"] for r in sub if np.isfinite(r["val_qr"])]
        te_qr = [r["test_qr"] for r in sub if np.isfinite(r["test_qr"])]
        n_pass = sum(1 for r in sub if r["passed"])
        print(
            f"{grp:>12s}  {len(sub):>3d}  "
            f"{np.mean(t_qr):>17.3f}  {np.mean(v_qr):>15.3f}  {np.mean(te_qr):>16.3f}  "
            f"{n_pass}/{len(sub):>4d}"
        )

    print()
    print("판정 기준:")
    print("  Test Q5/Q1 > 1.10 AND Test rho > 0.10 -> PASS (OOS에서 관계 유지)")
    print("  Train -> Test 열화 폭도 진단 정보로 기록")

    # 리포트
    rpt = ROOT / "docs" / "walkforward_report.md"
    lines = [
        "# Walk-Forward Report — Stage 6",
        "",
        "**질문**: |EPD_raw| → forward 5d vol 관계가 OOS에서 유지되는가?",
        "",
        "**방법**: Train 2005-2014, Val 2015-2017, Test 2018-2025",
        "각 기간별 Q5/Q1 비율 + Spearman rho 측정.",
        "",
        "**판정**: Test Q5/Q1 > 1.10 AND Test rho > 0.10",
        "",
        "## 자산별 결과",
        "",
        "| Asset | Group | Train Q5/Q1 | Val Q5/Q1 | Test Q5/Q1 | "
        "Train rho | Val rho | Test rho | PASS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:

        def s(v):
            return f"{v:.3f}" if np.isfinite(v) else "n/a"

        lines.append(
            f"| {r['name']} | {r['grp']} | {s(r['train_qr'])} | {s(r['val_qr'])} | "
            f"{s(r['test_qr'])} | {s(r['train_rho'])} | {s(r['val_rho'])} | "
            f"{s(r['test_rho'])} | {'✅' if r['passed'] else '❌'} |"
        )

    lines.append("")
    lines.append("## 자산군 요약")
    lines.append("")
    lines.append(
        "| Group | n | mean Train Q5/Q1 | mean Val Q5/Q1 | " "mean Test Q5/Q1 | PASS |"
    )
    lines.append("|---|---|---|---|---|---|")
    for grp in ["KR_stock", "KR_index", "US_index"]:
        sub = [r for r in rows if r["grp"] == grp]
        if not sub:
            continue
        t_qr = [r["train_qr"] for r in sub if np.isfinite(r["train_qr"])]
        v_qr = [r["val_qr"] for r in sub if np.isfinite(r["val_qr"])]
        te_qr = [r["test_qr"] for r in sub if np.isfinite(r["test_qr"])]
        n_pass = sum(1 for r in sub if r["passed"])
        lines.append(
            f"| {grp} | {len(sub)} | {np.mean(t_qr):.3f} | "
            f"{np.mean(v_qr):.3f} | {np.mean(te_qr):.3f} | "
            f"{n_pass}/{len(sub)} |"
        )
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    raise SystemExit(main())
