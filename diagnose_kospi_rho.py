"""KOSPI의 조합별 rho 확인."""

import sys
from pathlib import Path
from itertools import product
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT.parent / "packages" / "epd_core" / "src"))
from epd_core import compute_epd  # noqa: E402


def forward_return(r, h=5):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def quintile_monotonicity(epd, fwd, n_q=5):
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return np.nan, None
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i + 1])
        if mm.sum() < 20:
            return np.nan, None
        means.append(f[mm].mean())
    means = np.array(means)
    rq = np.arange(1, n_q + 1)
    rm = np.argsort(np.argsort(means)) + 1
    rho = float(np.corrcoef(rq, rm)[0, 1])
    return rho, means


for tkr in ["^KS11", "^GSPC"]:
    print("\n" + "=" * 80)
    print(f"=== {tkr} ===")
    print("=" * 80)
    df = yf.download(
        tkr,
        start="2005-01-01",
        end="2025-12-31",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    close = df["Close"].squeeze().to_numpy(dtype=float)
    close = close[np.isfinite(close)]
    r = np.diff(np.log(close))
    fwd = forward_return(r, 5)

    print(f"{'w_pe':>5s} {'m':>3s} {'w_z':>5s} | {'rho':>7s} | quintile means (Q1..Q5)")
    print("-" * 80)
    for w_pe, m, w_z in product([20, 60, 120], [3, 4], [60, 252, 504]):
        out = compute_epd(r, w_pe=w_pe, m=m, tau=1, w_z=w_z)
        rho, means = quintile_monotonicity(out["epd_100"], fwd)
        if means is None:
            print(f"{w_pe:>5d} {m:>3d} {w_z:>5d} | {rho:>7.3f} | (short)")
        else:
            ms = "  ".join(f"{v:+.4f}" for v in means)
            print(f"{w_pe:>5d} {m:>3d} {w_z:>5d} | {rho:>7.3f} | {ms}")
