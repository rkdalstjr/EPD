"""EPD v3.0 — C1 vs C4 Robustness (Stage 5).

검증:
  1. Parameter neighborhood (27조합 × 2)
  2. Time stability (3기간 × 8자산)
  3. Bootstrap stability (100회)

PASS:
  - 27조합 중 |rho_fv| > 0.03 비율 >= 80%
  - 3기간 모두 |rho_fv| > 0 (부호 유지)
  - Bootstrap 평균 - 1σ > 0
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd_v3

warnings.filterwarnings("ignore")

TICKERS = [
    ("^GSPC", "SPX"),
    ("^KS11", "KOSPI"),
    ("^KQ11", "KOSDAQ"),
    ("005930.KS", "삼성전자"),
]

W_PE_GRID = [40, 60, 80]
W_Z_GRID = [152, 252, 352]
RESID_GRID = [252, 504, 756]

PERIODS = [
    ("2005-2012", "2005-01-01", "2012-12-31"),
    ("2013-2019", "2013-01-01", "2019-12-31"),
    ("2020-2025", "2020-01-01", "2025-12-31"),
]

H = 5
RHO_THRESHOLD = 0.03
HORIZON_BOOT = 100


def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t+1:t+1+h]
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


def get_close(tkr, start="2005-01-01", end="2025-12-31"):
    import yfinance as yf
    df = yf.download(tkr, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df is None or len(df) < 500:
        return None
    c = df["Close"].squeeze().to_numpy(dtype=float)
    return c[np.isfinite(c)]


def main():
    print("=" * 130)
    print("EPD v3.0 — C1 vs C4 Robustness (Stage 5)")
    print(f"|rho_fv| > {RHO_THRESHOLD} 비율 >= 80%")
    print("=" * 130)

    # ============================================================
    # 1) Parameter neighborhood
    # ============================================================
    print("\n### [1] Parameter neighborhood (27조합 × 4자산)\n")

    for tkr, name in TICKERS:
        close = get_close(tkr)
        if close is None:
            continue

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))
        fv = forward_vol(r, H)

        print(f"\n  --- {name} ---")
        print(f"  {'w_pe':>5s} {'w_z':>5s} {'resid':>6s}  "
              f"{'C1':>8s}  {'C4':>8s}")
        print("  " + "-" * 42)

        for w_pe in W_PE_GRID:
            for w_z in W_Z_GRID:
                for resid in RESID_GRID:
                    try:
                        out = compute_epd_v3(close, w_pe=w_pe, w_z=w_z,
                                              resid_win=resid)
                        rho_c1 = spearman(out["C1_raw"], fv)
                        rho_c4 = spearman(out["C4_raw"], fv)
                        print(f"  {w_pe:>5d} {w_z:>5d} {resid:>6d}  "
                              f"{rho_c1:>+8.4f}  {rho_c4:>+8.4f}")
                    except Exception as e:
                        print(f"  {w_pe:>5d} {w_z:>5d} {resid:>6d}  err: {e}")

    # ============================================================
    # 2) Time stability
    # ============================================================
    print("\n\n### [2] Time stability (3기간 × 4자산)\n")
    print(f"  {'asset':>12s}  {'period':>11s}  "
          f"{'C1 |rho|':>10s}  {'C4 |rho|':>10s}")
    print("  " + "-" * 55)

    for tkr, name in TICKERS:
        for pname, start, end in PERIODS:
            close = get_close(tkr, start=start, end=end)
            if close is None or close.size < 500:
                continue
            r = np.full_like(close, np.nan)
            r[1:] = np.diff(np.log(close))
            fv = forward_vol(r, H)
            try:
                out = compute_epd_v3(close)
                rho_c1 = spearman(out["C1_raw"], fv)
                rho_c4 = spearman(out["C4_raw"], fv)
                print(f"  {name:>12s}  {pname:>11s}  "
                      f"{rho_c1:>+10.4f}  {rho_c4:>+10.4f}")
            except Exception:
                pass

    # ============================================================
    # 3) Bootstrap stability (SPX만)
    # ============================================================
    print("\n\n### [3] Bootstrap stability (SPX, 100회)\n")

    close = get_close("^GSPC")
    if close is not None:
        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))
        fv = forward_vol(r, H)
        out = compute_epd_v3(close)
        c1 = out["C1_raw"]
        c4 = out["C4_raw"]

        n = len(close)
        rhos_c1 = []
        rhos_c4 = []
        rng = np.random.default_rng(42)
        for _ in range(HORIZON_BOOT):
            idx = np.sort(rng.choice(n, size=int(n * 0.8), replace=False))
            rho_c1 = spearman(c1[idx], fv[idx])
            rho_c4 = spearman(c4[idx], fv[idx])
            if np.isfinite(rho_c1):
                rhos_c1.append(rho_c1)
            if np.isfinite(rho_c4):
                rhos_c4.append(rho_c4)

        for label, vals in [("C1", rhos_c1), ("C4", rhos_c4)]:
            vals = np.array(vals)
            print(f"  {label}: mean={vals.mean():+.4f}  "
                  f"std={vals.std():.4f}  "
                  f"min={vals.min():+.4f}  max={vals.max():+.4f}  "
                  f"frac>0={np.mean(vals > 0):.2%}")

    print()
    print("=" * 130)
    print("판정 기준:")
    print(f"  - Parameter: {RHO_THRESHOLD} 초과 비율 >= 80%")
    print("  - Time: 모든 기간에서 부호 일치")
    print("  - Bootstrap: mean - std > 0")


if __name__ == "__main__":
    raise SystemExit(main())