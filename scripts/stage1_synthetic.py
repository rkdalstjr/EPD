"""Stage 1: Synthetic Validation — EPD v1.2 (log-price)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import compute_epd  # noqa: E402

RNG = np.random.default_rng(20261006)
N = 10_000


def returns_to_close(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    log_P = np.zeros(n)
    for i in range(1, n):
        log_P[i] = log_P[i-1] + r[i]
    return np.exp(log_P)


def gen_white_noise(n=N):
    return RNG.standard_normal(n)


def gen_garch(n=N, alpha=0.10, beta=0.85, seed=7):
    rng = np.random.default_rng(seed)
    omega = 1.0 - alpha - beta
    eps = rng.standard_normal(n)
    r = np.zeros(n); s2 = np.zeros(n)
    s2[0] = omega / max(1e-9, 1 - alpha - beta)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t-1]**2 + beta * s2[t-1]
        r[t] = np.sqrt(s2[t]) * eps[t]
    return r


def gen_regime(n=N, p_stay=0.98, seed=11):
    rng = np.random.default_rng(seed)
    state = 0; states = np.zeros(n, dtype=int)
    for t in range(n):
        if rng.random() > p_stay:
            state = 1 - state
        states[t] = state
    sig = np.where(states == 0, 1.0, 3.0)
    return rng.standard_normal(n) * sig, states


def gen_divergence(n=N, seed=13):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n)
    phi = 0.9
    b1 = np.zeros(1000)
    for i in range(1, 1000):
        b1[i] = 0.5 + phi * b1[i-1] + rng.standard_normal() * 0.3
    r[3000:4000] = b1
    b2 = np.zeros(1000)
    for i in range(1, 1000):
        b2[i] = -0.5 + phi * b2[i-1] + rng.standard_normal() * 0.3
    r[7000:8000] = b2
    return r


def forward_return(r, h):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t+1:t+1+h].sum()
    return fr


def forward_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fv[t] = r[t+1:t+1+h].std(ddof=1)
    return fv


def spearman_ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def ic_blocks(x, y, block=1000):
    ics = [spearman_ic(x[i:i+block], y[i:i+block])
           for i in range(0, len(x), block)]
    return np.array([v for v in ics if np.isfinite(v)])


def test_A_white_noise():
    r = gen_white_noise()
    epd = compute_epd(returns_to_close(r))["epd_100"]
    ic_ret = spearman_ic(epd, forward_return(r, 5))
    ic_vol = spearman_ic(epd, forward_vol(r, 5))
    blk = ic_blocks(epd, forward_return(r, 5), block=1000)
    frac = float(np.mean(np.abs(blk) > 2/np.sqrt(1000))) if blk.size else np.nan
    passed = (abs(ic_ret) < 0.05) and (abs(ic_vol) < 0.05)
    return {"name": "A_white_noise", "ic_ret_5d": ic_ret, "ic_vol_5d": ic_vol,
            "frac_blocks_2sigma": frac, "pass": bool(passed)}


def test_B_garch():
    r = gen_garch()
    epd = compute_epd(returns_to_close(r))["epd_100"]
    ic_ret = spearman_ic(epd, forward_return(r, 5))
    ic_vol = spearman_ic(epd, forward_vol(r, 5))
    return {"name": "B_garch", "ic_ret_5d": ic_ret, "ic_vol_5d": ic_vol,
            "pass": bool(abs(ic_ret) < 0.05)}


def test_C_regime():
    r, states = gen_regime()
    epd100 = compute_epd(returns_to_close(r))["epd_100"]
    trans = np.where(np.diff(states) != 0)[0] + 1
    hits = 0
    for t in trans:
        lo, hi = max(0, t-5), min(len(epd100), t+5)
        if np.any(np.abs(epd100[lo:hi] - 50) > 20):
            hits += 1
    frac = hits / max(1, len(trans))
    return {"name": "C_regime", "n_transitions": int(len(trans)),
            "hit_rate_within_5d": frac, "pass": bool(frac >= 0.5)}


def test_D_divergence():
    r = gen_divergence()
    out = compute_epd(returns_to_close(r))
    raw = out["epd_raw"]
    n = len(raw)
    ev = np.zeros(n, dtype=bool)
    ev[3020:3100] = True
    ev[7020:7100] = True
    bg = np.isfinite(raw) & ~ev
    ei = np.isfinite(raw) & ev
    thr = 1.5
    fe = float(np.mean(np.abs(raw[ei]) > thr)) if ei.sum() else 0.0
    fb = float(np.mean(np.abs(raw[bg]) > thr)) if bg.sum() else 0.0
    passed = (fe > 0.2) and (fe > 2 * fb)
    return {"name": "D_divergence", "frac_extreme_event": fe,
            "frac_extreme_bg": fb, "pass": bool(passed)}


def main():
    results = []
    for fn in (test_A_white_noise, test_B_garch, test_C_regime, test_D_divergence):
        res = fn(); results.append(res)
        flag = "PASS" if res["pass"] else "FAIL"
        print(f"[Stage1:{res['name']}] {flag}  "
              f"{ {k:v for k,v in res.items() if k not in ('name','pass')} }")

    passed = all(r["pass"] for r in results)
    print(f"\n=== Stage 1 Result: {'PASS' if passed else 'FAIL'} ===")

    rpt = ROOT / "docs" / "synthetic_report.md"
    lines = ["# Synthetic Report — Stage 1 (v1.2, log-price)", "",
             "Fixed params: w_pe=60, m=3, tau=1, w_z=252, tanh_scale=2", ""]
    for res in results:
        lines.append(f"## {res['name']}")
        for k, v in res.items():
            if k != "name":
                lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(f"**Overall**: {'PASS' if passed else 'FAIL'}")
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
