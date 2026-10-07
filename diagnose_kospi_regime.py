"""KOSPI의 EPD 신호를 regime별로 분해."""

import sys
from pathlib import Path
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


def mono(epd, fwd, n_q=5):
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return np.nan
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i + 1])
        if mm.sum() < 20:
            return np.nan
        means.append(f[mm].mean())
    rq = np.arange(1, n_q + 1)
    rm = np.argsort(np.argsort(means)) + 1
    return float(np.corrcoef(rq, rm)[0, 1])


# 기간 분할
periods = [
    ("2005-2012", "2005-01-01", "2012-12-31"),
    ("2013-2019", "2013-01-01", "2019-12-31"),
    ("2020-2025", "2020-01-01", "2025-12-31"),
]

for tkr in ["^KS11", "^GSPC"]:
    print("\n" + "=" * 70)
    print(f"=== {tkr} (기간별) ===")
    print("=" * 70)
    print(
        f"{'period':>12s}  {'n':>6s}  {'rho':>7s}  {'Q1 mean':>10s}  {'Q5 mean':>10s}"
    )
    for name, start, end in periods:
        df = yf.download(
            tkr, start=start, end=end, interval="1d", progress=False, auto_adjust=True
        )
        close = df["Close"].squeeze().to_numpy(dtype=float)
        close = close[np.isfinite(close)]
        r = np.diff(np.log(close))
        fwd = forward_return(r, 5)
        out = compute_epd(r, w_pe=60, m=3, tau=1, w_z=252)
        epd = out["epd_100"]
        m = np.isfinite(epd) & np.isfinite(fwd)
        if m.sum() < 100:
            print(f"{name:>12s}  (n<100)")
            continue
        rho = mono(epd, fwd)
        e, f = epd[m], fwd[m]
        qs = np.quantile(e, [0.2, 0.8])
        q1 = f[e <= qs[0]].mean()
        q5 = f[e >= qs[1]].mean()
        print(f"{name:>12s}  {m.sum():>6d}  {rho:>7.3f}  {q1:>+10.4f}  {q5:>+10.4f}")
