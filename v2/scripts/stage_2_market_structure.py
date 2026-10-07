"""Stage 2 — Market Structure Validation.

질문: EPD의 극단값이 실제 시장 상태와 연결되는가?

측정:
  - |EPD_raw| 5분위별 forward 5일 변동성 평균
  - Q5/Q1 비율 (극단 vs 안정)
  - 임계값 |EPD_raw|>1.5 vs <0.5 비교

PASS:
  - 자산별 Q5/Q1 > 1.10
  - 자산군별 통과율 >= 70%
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd

warnings.filterwarnings("ignore")

H = 5
Q5Q1_THRESHOLD = 1.10
GROUP_PASS_RATE = 0.70

TICKERS = [
    # (ticker, name, group)
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


def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def quintile_ratio(intensity, fv, n_q=5):
    """Q5/Q1 비율."""
    m = np.isfinite(intensity) & np.isfinite(fv)
    if m.sum() < n_q * 30:
        return None, None
    a, b = intensity[m], fv[m]
    qs = np.quantile(a, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (a >= qs[i]) & (a < qs[i + 1])
        if mm.sum() < 20:
            return None, None
        means.append(b[mm].mean())
    means = np.array(means)
    if means[0] <= 0:
        return None, means
    return float(means[-1] / means[0]), means


def threshold_ratio(intensity, fv, thr_hi=1.5, thr_lo=0.5):
    m = np.isfinite(intensity) & np.isfinite(fv)
    if m.sum() < 100:
        return None, None
    a, b = intensity[m], fv[m]
    hi = b[a >= thr_hi]
    lo = b[a < thr_lo]
    if hi.size < 20 or lo.size < 20 or lo.mean() <= 0:
        return None, None
    return float(hi.mean() / lo.mean()), (hi.mean(), lo.mean())


def main():
    import yfinance as yf

    print("=" * 110)
    print("Stage 2 — Market Structure Validation")
    print(f"forward window H={H}, Q5/Q1 threshold={Q5Q1_THRESHOLD}")
    print("=" * 110)

    print(
        f"\n{'asset':>14s}  {'group':>10s}  {'n':>5s}  | "
        f"{'Q1':>9s} {'Q2':>9s} {'Q3':>9s} {'Q4':>9s} {'Q5':>9s}  | "
        f"{'Q5/Q1':>6s}  {'>1.5/<0.5':>9s}  {'pass':>5s}"
    )
    print("-" * 130)

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
            if df is None or len(df) < 500:
                print(f"{name:>14s}  {grp:>10s}  (skip)")
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
        except Exception as e:
            print(f"{name:>14s}  err: {e}")
            continue

        out = compute_epd(close)
        epd_abs = out["epd_abs"]

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))
        fv = forward_vol(r, H)

        q_ratio, q_means = quintile_ratio(epd_abs, fv)
        t_ratio, _ = threshold_ratio(epd_abs, fv)

        passed = (q_ratio is not None) and (q_ratio > Q5Q1_THRESHOLD)

        if q_means is None:
            print(f"{name:>14s}  {grp:>10s}  {len(close):>5d}  (insufficient)")
            continue

        print(
            f"{name:>14s}  {grp:>10s}  {len(close):>5d}  | "
            f"{q_means[0]:>9.6f} {q_means[1]:>9.6f} {q_means[2]:>9.6f} "
            f"{q_means[3]:>9.6f} {q_means[4]:>9.6f}  | "
            f"{q_ratio:>6.3f}  {(t_ratio if t_ratio else float('nan')):>9.3f}  "
            f"{'✅' if passed else '❌':>5s}"
        )

        rows.append(
            {
                "name": name,
                "grp": grp,
                "n": len(close),
                "q_means": q_means,
                "q_ratio": q_ratio,
                "t_ratio": t_ratio,
                "passed": passed,
            }
        )

    # ----- 자산군별 요약 -----
    print()
    print("=" * 110)
    print("자산군별 요약")
    print("=" * 110)
    print(
        f"{'group':>12s}  {'n':>3s}  {'mean Q5/Q1':>12s}  "
        f"{'mean >1.5/<0.5':>15s}  {'통과율':>8s}  {'판정':>6s}"
    )
    print("-" * 90)

    group_results = {}
    for grp in ["US_index", "KR_index", "KR_stock"]:
        sub = [r for r in rows if r["grp"] == grp]
        if not sub:
            continue
        qr = [r["q_ratio"] for r in sub if r["q_ratio"] is not None]
        tr = [r["t_ratio"] for r in sub if r["t_ratio"] is not None]
        n_pass = sum(1 for r in sub if r["passed"])
        frac = n_pass / len(sub)
        ok = frac >= GROUP_PASS_RATE
        group_results[grp] = {
            "frac": frac,
            "ok": ok,
            "mean_qr": float(np.mean(qr)) if qr else np.nan,
            "mean_tr": float(np.mean(tr)) if tr else np.nan,
        }
        print(
            f"{grp:>12s}  {len(sub):>3d}  "
            f"{np.mean(qr):>12.3f}  {np.mean(tr):>15.3f}  "
            f"{frac:>7.1%}  {'✅' if ok else '❌':>6s}"
        )

    overall = all(g["ok"] for g in group_results.values())
    print()
    print(f"=== Stage 2 Result: {'PASS' if overall else 'FAIL'} ===")

    # ----- 리포트 -----
    rpt = ROOT / "docs" / "STAGE_2_MARKET_STRUCTURE.md"
    L = [
        "# Stage 2 — Market Structure Validation",
        "",
        f"- H (forward window): {H}일",
        f"- Q5/Q1 threshold: {Q5Q1_THRESHOLD}",
        f"- 자산군 통과 기준: {GROUP_PASS_RATE:.0%}",
        "",
        "## 자산별 결과",
        "",
        "| Asset | Group | n | Q1 | Q2 | Q3 | Q4 | Q5 | Q5/Q1 | >1.5/<0.5 | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q = r["q_means"]
        L.append(
            f"| {r['name']} | {r['grp']} | {r['n']} | "
            f"{q[0]:.6f} | {q[1]:.6f} | {q[2]:.6f} | {q[3]:.6f} | {q[4]:.6f} | "
            f"{r['q_ratio']:.3f} | "
            f"{(r['t_ratio'] if r['t_ratio'] else float('nan')):.3f} | "
            f"{'✅' if r['passed'] else '❌'} |"
        )

    L.append("")
    L.append("## 자산군 요약")
    L.append("")
    L.append("| Group | n | mean Q5/Q1 | mean >1.5/<0.5 | 통과율 | 판정 |")
    L.append("|---|---|---|---|---|---|")
    for grp, g in group_results.items():
        sub = [r for r in rows if r["grp"] == grp]
        L.append(
            f"| {grp} | {len(sub)} | {g['mean_qr']:.3f} | "
            f"{g['mean_tr']:.3f} | {g['frac']:.1%} | "
            f"{'✅' if g['ok'] else '❌'} |"
        )

    L.append("")
    L.append(f"**Overall**: {'PASS' if overall else 'FAIL'}")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {rpt}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
