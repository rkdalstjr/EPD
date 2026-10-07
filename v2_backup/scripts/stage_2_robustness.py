"""Stage 2 — Robustness Validation.

Seed / Sample / Parameter 변화에 대한 안정성.
PASS: 전체 |rho| > 0.10 비율 >= 80%
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
SEEDS = list(range(10))


def returns_to_close(r, scale=0.01):
    r = np.asarray(r, dtype=float)
    return np.exp(np.cumsum(r) * scale)


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


def rho_of(r):
    close = returns_to_close(r)
    out = compute_epd(close)
    return spearman(out["epd_abs"], forward_vol(r))


# ---------- synthetic scenarios ----------
def gen_garch(seed, n=5000):
    rng = np.random.default_rng(seed)
    alpha, beta = 0.10, 0.85
    omega = 1.0 - alpha - beta
    r = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / max(1e-9, 1 - alpha - beta)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * eps[t]
    return r


def gen_regime(seed, n=5000):
    rng = np.random.default_rng(seed)
    state = 0
    states = np.zeros(n, dtype=int)
    for t in range(n):
        if rng.random() > 0.98:
            state = 1 - state
        states[t] = state
    return rng.standard_normal(n) * np.where(states == 0, 1.0, 3.0)


def gen_divergence(seed, n=5000):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n)
    phi = 0.9
    for start in (1500, 3500):
        b = np.zeros(1000)
        drift = 0.5 if start == 1500 else -0.5
        for i in range(1, 1000):
            b[i] = drift + phi * b[i - 1] + rng.standard_normal() * 0.3
        r[start : start + 1000] = b
    return r


# ---------- tests ----------
def test_seed_robustness():
    print("=== Seed Robustness ===")
    scenarios = {"garch": gen_garch, "regime": gen_regime, "divergence": gen_divergence}
    rows = []
    for name, fn in scenarios.items():
        rhos = []
        for seed in SEEDS:
            rho = rho_of(fn(seed))
            if np.isfinite(rho):
                rhos.append(rho)
        frac = sum(1 for v in rhos if abs(v) > 0.10) / len(rhos) if rhos else 0
        print(
            f"  {name:12s}: n={len(rhos)}, mean={np.mean(rhos):+.3f}, "
            f"frac|rho|>0.1={frac:.0%}"
        )
        rows.append({"scenario": name, "rhos": rhos, "frac": frac})
    return rows


def test_sample_robustness():
    print("\n=== Sample Robustness (real) ===")
    import yfinance as yf

    assets = [("^GSPC", "SPX"), ("^KS11", "KOSPI"), ("^KQ11", "KOSDAQ")]
    periods = [
        ("2005-2012", "2005-01-01", "2012-12-31"),
        ("2013-2019", "2013-01-01", "2019-12-31"),
        ("2020-2025", "2020-01-01", "2025-12-31"),
    ]
    rows = []
    for tkr, name in assets:
        for pname, start, end in periods:
            try:
                df = yf.download(
                    tkr, start=start, end=end, progress=False, auto_adjust=True
                )
                close = df["Close"].squeeze().to_numpy(dtype=float)
                close = close[np.isfinite(close)]
                if close.size < 300:
                    continue
                r = np.full_like(close, np.nan)
                r[1:] = np.diff(np.log(close))
                out = compute_epd(close)
                fv = forward_vol(r, H)
                rho = spearman(out["epd_abs"], fv)
                print(f"  {name:8s} {pname}: rho={rho:+.3f}")
                rows.append({"asset": name, "period": pname, "rho": rho})
            except Exception as e:
                print(f"  {name} {pname}: err {e}")
    return rows


def test_parameter_sensitivity():
    print("\n=== Parameter Sensitivity (SPX) ===")
    import yfinance as yf

    df = yf.download(
        "^GSPC", start="2005-01-01", end="2025-12-31", progress=False, auto_adjust=True
    )
    close = df["Close"].squeeze().to_numpy(dtype=float)
    close = close[np.isfinite(close)]
    r = np.full_like(close, np.nan)
    r[1:] = np.diff(np.log(close))
    fv = forward_vol(r, H)

    rows = []
    for w_pe in [40, 60, 80]:
        for w_z in [152, 252, 352]:
            for ema_span in [7, 14, 21]:
                out = compute_epd(close, w_pe=w_pe, w_z=w_z, ema_span=ema_span)
                rho = spearman(out["epd_abs"], fv)
                rows.append(
                    {"w_pe": w_pe, "w_z": w_z, "ema_span": ema_span, "rho": rho}
                )
    # 요약만 출력
    rhos = [r["rho"] for r in rows if np.isfinite(r["rho"])]
    frac = sum(1 for v in rhos if abs(v) > 0.10) / len(rhos) if rhos else 0
    print(
        f"  27 조합: mean={np.mean(rhos):+.3f}, "
        f"frac|rho|>0.1={frac:.0%}, min={np.min(rhos):+.3f}, max={np.max(rhos):+.3f}"
    )
    return rows


def main():
    print("=" * 80)
    print("Stage 2 — Robustness Validation (v2.0)")
    print("=" * 80)

    seed_rows = test_seed_robustness()
    sample_rows = test_sample_robustness()
    param_rows = test_parameter_sensitivity()

    all_rhos = []
    for r in seed_rows:
        all_rhos += r["rhos"]
    all_rhos += [r["rho"] for r in sample_rows if np.isfinite(r["rho"])]
    all_rhos += [r["rho"] for r in param_rows if np.isfinite(r["rho"])]

    n_valid = len(all_rhos)
    n_strong = sum(1 for v in all_rhos if abs(v) > 0.10)
    frac = n_strong / n_valid if n_valid else 0
    overall = frac >= 0.80

    print()
    print(f"전체: n={n_valid}, |rho|>0.10 = {n_strong} ({frac:.1%})")
    print(f"=== Stage 2 Result: {'PASS' if overall else 'FAIL'} ===")

    rpt = ROOT / "docs" / "STAGE_2_ROBUSTNESS.md"
    L = [
        "# Stage 2 — Robustness Validation",
        "",
        f"- Seed: {len(SEEDS)} × 3 scenarios",
        f"- Sample: 3자산 × 3기간",
        f"- Parameter: 27조합 (SPX)",
        f"- 총: {n_valid} 조합",
        f"- **|rho|>0.10 = {n_strong}/{n_valid} ({frac:.1%})**",
        "",
        f"**Overall**: {'PASS' if overall else 'FAIL'}",
        "",
        "## Seed Robustness",
        "",
        "| Scenario | n | mean rho | frac\\|rho\\|>0.1 |",
        "|---|---|---|---|",
    ]
    for r in seed_rows:
        L.append(
            f"| {r['scenario']} | {len(r['rhos'])} | "
            f"{np.mean(r['rhos']):+.3f} | {r['frac']:.0%} |"
        )
    L += ["", "## Sample Robustness", "", "| Asset | Period | rho |", "|---|---|---|"]
    for r in sample_rows:
        L.append(f"| {r['asset']} | {r['period']} | {r['rho']:+.3f} |")
    L += [
        "",
        "## Parameter Sensitivity",
        "",
        "| w_pe | w_z | ema | rho |",
        "|---|---|---|---|",
    ]
    for r in param_rows:
        L.append(f"| {r['w_pe']} | {r['w_z']} | {r['ema_span']} | {r['rho']:+.3f} |")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {rpt}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
