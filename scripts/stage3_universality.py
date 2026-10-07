"""Stage 3: Universality — EPD v1.2, 18조합 (log-price)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import compute_epd  # noqa: E402

warnings.filterwarnings("ignore")

RHO_STRONG = 0.6
FRAC_STRONG_THRESHOLD = 0.70


def load_real_data():
    try:
        import yfinance as yf
    except ImportError:
        print("[data] yfinance 없음")
        return None

    tickers = {"SPX": "^GSPC", "KOSPI": "^KS11", "NASDAQ": "^IXIC"}
    out = {}
    for name, tkr in tickers.items():
        try:
            df = yf.download(tkr, start="2005-01-01", end="2025-12-31",
                             interval="1d", progress=False, auto_adjust=True)
            if df is None or len(df) < 1000:
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            r = np.diff(np.log(close))
            out[name] = {"close": close[1:], "returns": r}
            print(f"[data] {name}: n={len(r)}")
        except Exception as e:
            print(f"[data] {name} 에러: {e}")
    return out if out else None


def forward_return(r, h=5):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t+1:t+1+h].sum()
    return fr


def quintile_monotonicity(epd, fwd, n_q=5):
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return np.nan
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12; qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i + 1])
        if mm.sum() < 20:
            return np.nan
        means.append(f[mm].mean())
    rq = np.arange(1, n_q + 1)
    rm = np.argsort(np.argsort(np.array(means))) + 1
    return float(np.corrcoef(rq, rm)[0, 1])


def run_grid(data):
    w_pe_grid = [20, 60, 120]
    m_grid = [3, 4]
    w_z_grid = [60, 252, 504]
    rows = []
    for asset, d in data.items():
        close = d["close"]
        r = d["returns"]
        fwd = forward_return(r, 5)
        for w_pe, m, w_z in product(w_pe_grid, m_grid, w_z_grid):
            try:
                out = compute_epd(close, w_pe=w_pe, m=m, tau=1, w_z=w_z)
            except Exception:
                continue
            rho = quintile_monotonicity(out["epd_100"], fwd)
            rows.append({"asset": asset, "w_pe": w_pe, "m": m, "w_z": w_z,
                         "rho": rho, "abs_rho": abs(rho) if np.isfinite(rho) else np.nan})
    return rows


def main():
    print("=" * 70)
    print("Stage 3: Universality — EPD v1.2 (log-price)")
    print(f"판정: |rho| >= {RHO_STRONG} 비율 >= {FRAC_STRONG_THRESHOLD:.0%}")
    print("=" * 70)

    data = load_real_data()
    if data is None:
        print("[FAIL] 데이터 로드 실패")
        return 1

    rows = run_grid(data)
    valid = [r for r in rows if np.isfinite(r["rho"])]
    if not valid:
        print("[FAIL] 유효 조합 없음")
        return 1

    print(f"\n{'asset':>8s}  {'n':>4s}  {'frac|rho|>=0.6':>16s}  {'median_rho':>11s}  {'median|rho|':>12s}")
    print("-" * 70)
    results = {}
    for asset in sorted(set(r["asset"] for r in valid)):
        sub = [r for r in valid if r["asset"] == asset]
        frac_strong = sum(1 for r in sub if r["abs_rho"] >= RHO_STRONG) / len(sub)
        med_rho = float(np.median([r["rho"] for r in sub]))
        med_abs = float(np.median([r["abs_rho"] for r in sub]))
        results[asset] = frac_strong
        print(f"{asset:>8s}  {len(sub):>4d}  {frac_strong:>16.2%}  {med_rho:>11.4f}  {med_abs:>12.4f}")

    print(f"\n--- 자산별 판정 (frac >= {FRAC_STRONG_THRESHOLD:.0%}) ---")
    fail = []
    for asset, frac in results.items():
        ok = frac >= FRAC_STRONG_THRESHOLD
        print(f"  {asset:>8s}  {frac:.2%}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            fail.append(asset)

    overall = len(fail) == 0
    print(f"\n=== Stage 3 Result: {'PASS' if overall else 'FAIL'} ===")

    rpt = ROOT / "docs" / "universality_report.md"
    lines = ["# Universality Report — Stage 3 (v1.2, log-price)", "",
             f"판정: |rho| >= {RHO_STRONG} 비율 >= {FRAC_STRONG_THRESHOLD:.0%}",
             "", "| Asset | n | frac_strong | median_rho | Pass |",
             "|---|---|---|---|---|"]
    for asset, frac in results.items():
        sub = [r for r in valid if r["asset"] == asset]
        med = float(np.median([r["rho"] for r in sub]))
        ok = frac >= FRAC_STRONG_THRESHOLD
        lines.append(f"| {asset} | {len(sub)} | {frac:.2%} | {med:.4f} | "
                     f"{'✅' if ok else '❌'} |")
    lines.append("")
    lines.append(f"**Overall**: {'PASS' if overall else 'FAIL'}")
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
