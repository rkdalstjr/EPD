"""EPD 소스 파일들 다시 채우기 (EPD 루트에서 실행)."""

from pathlib import Path

ROOT = Path(".")
FILES = {}

FILES[
    "packages/epd_core/src/epd_core/permutation.py"
] = '''"""순열 엔트로피 (Bandt-Pompe 2002)."""
from __future__ import annotations
import numpy as np
from math import factorial, log


def _ordinal_pattern(vec: np.ndarray) -> tuple:
    return tuple(np.argsort(vec, kind="stable").tolist())


def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """정규화된 순열 엔트로피 in [0, 1]."""
    x = np.asarray(x, dtype=float)
    n = x.size
    span = (m - 1) * tau + 1
    if n < span:
        return np.nan

    counts = {}
    total = 0
    for i in range(n - span + 1):
        vec = x[i : i + span : tau]
        key = _ordinal_pattern(vec)
        counts[key] = counts.get(key, 0) + 1
        total += 1

    if total == 0:
        return np.nan

    p = np.fromiter(counts.values(), dtype=float, count=len(counts)) / total
    h = -float(np.sum(p * np.log(p)))
    h_max = log(factorial(m))
    return h / h_max if h_max > 0 else np.nan
'''

FILES[
    "packages/epd_core/src/epd_core/_core.py"
] = '''"""롤링 유틸: PE, causal z-score, slope."""
from __future__ import annotations
import numpy as np
from .permutation import permutation_entropy


def rolling_pe(returns: np.ndarray, w_pe: int = 60,
               m: int = 3, tau: int = 1) -> np.ndarray:
    """PE_t = permutation_entropy(r[t-w_pe:t]). Causal."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    out = np.full(n, np.nan)
    for t in range(w_pe, n):
        out[t] = permutation_entropy(r[t - w_pe : t], m=m, tau=tau)
    return out


def rolling_zscore(x: np.ndarray, w_z: int = 252,
                   min_periods: int | None = None) -> np.ndarray:
    """Causal rolling z-score."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if min_periods is None:
        min_periods = w_z
    out = np.full(n, np.nan)
    for t in range(n):
        lo = t - w_z
        if lo < 0:
            continue
        win = x[lo:t]
        win = win[~np.isnan(win)]
        if win.size < min_periods:
            continue
        mu = win.mean()
        sd = win.std(ddof=1)
        if not np.isfinite(sd) or sd == 0.0:
            out[t] = 0.0
        else:
            out[t] = (x[t] - mu) / sd
    return out


def rolling_slope(y: np.ndarray, w: int = 60) -> np.ndarray:
    """y[t-w+1 : t+1] 단순선형회귀 기울기."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    xgrid = np.arange(w, dtype=float)
    xmean = xgrid.mean()
    xvar = ((xgrid - xmean) ** 2).sum()
    for t in range(w - 1, n):
        win = y[t - w + 1 : t + 1]
        if np.isnan(win).any():
            continue
        ymean = win.mean()
        out[t] = ((xgrid - xmean) * (win - ymean)).sum() / xvar
    return out
'''

FILES[
    "packages/epd_core/src/epd_core/epd.py"
] = '''"""EPD — Entropy-Price Divergence, RSI-style bounded indicator."""
from __future__ import annotations
import numpy as np
from ._core import rolling_pe, rolling_zscore

DEFAULT_W_PE = 60
DEFAULT_M = 3
DEFAULT_TAU = 1
DEFAULT_W_Z = 252
DEFAULT_TANH_SCALE = 2.0
THRESHOLD_HIGH = 70
THRESHOLD_LOW = 30


def compute_epd(returns, close=None, w_pe=DEFAULT_W_PE, m=DEFAULT_M,
                tau=DEFAULT_TAU, w_z=DEFAULT_W_Z, bounded=True,
                tanh_scale=DEFAULT_TANH_SCALE):
    r = np.asarray(returns, dtype=float)
    n = r.size

    pe = rolling_pe(r, w_pe=w_pe, m=m, tau=tau)
    dpe = np.full(n, np.nan)
    if n >= 2:
        dpe[1:] = pe[1:] - pe[:-1]

    z_r = rolling_zscore(r, w_z=w_z)
    z_e = rolling_zscore(dpe, w_z=w_z)
    epd_raw = z_r - z_e

    if bounded:
        epd = np.tanh(epd_raw / tanh_scale)
        epd_100 = 50.0 * (1.0 + epd)
    else:
        epd = epd_raw
        epd_100 = epd_raw

    with np.errstate(invalid="ignore"):
        prod = z_r * z_e
    alignment = np.sign(prod)
    alignment[np.isnan(prod)] = np.nan

    return {
        "epd": epd,
        "epd_100": epd_100,
        "epd_raw": epd_raw,
        "epd_abs": np.abs(epd),
        "alignment": alignment,
        "return_z": z_r,
        "entropy_z": z_e,
        "pe": pe,
        "dpe": dpe,
        "params": {"w_pe": w_pe, "m": m, "tau": tau, "w_z": w_z,
                   "bounded": bounded, "tanh_scale": tanh_scale},
    }
'''

FILES[
    "packages/epd_core/src/epd_core/__init__.py"
] = '''"""EPD Core — RSI-style bounded indicator."""
from .epd import (compute_epd, DEFAULT_W_PE, DEFAULT_M, DEFAULT_TAU,
                  DEFAULT_W_Z, DEFAULT_TANH_SCALE,
                  THRESHOLD_HIGH, THRESHOLD_LOW)
from .permutation import permutation_entropy
from ._core import rolling_pe, rolling_zscore, rolling_slope

__all__ = [
    "compute_epd", "permutation_entropy",
    "rolling_pe", "rolling_zscore", "rolling_slope",
    "DEFAULT_W_PE", "DEFAULT_M", "DEFAULT_TAU",
    "DEFAULT_W_Z", "DEFAULT_TANH_SCALE",
    "THRESHOLD_HIGH", "THRESHOLD_LOW",
]
__version__ = "0.1.0"
'''

FILES["packages/epd_core/pyproject.toml"] = """[project]
name = "epd-core"
version = "0.1.0"
description = "Entropy-Price Divergence (RSI-style bounded indicator)"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/epd_core"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

FILES["scripts/stage1_synthetic.py"] = '''"""Stage 1: Synthetic Validation — EPD_100."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import compute_epd  # noqa: E402

RNG = np.random.default_rng(20261006)
N = 10_000


def gen_white_noise(n=N):
    return RNG.standard_normal(n)


def gen_garch(n=N, alpha=0.10, beta=0.85, seed=7):
    rng = np.random.default_rng(seed)
    omega = 1.0 - alpha - beta
    eps = rng.standard_normal(n)
    r = np.zeros(n); s2 = np.zeros(n)
    s2[0] = omega / max(1e-9, 1 - alpha - beta)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
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
    eps = rng.standard_normal(n) * 0.5
    phi = np.linspace(0.0, 0.85, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.05 + phi[t] * r[t - 1] + eps[t]
    return r


def forward_return(r, h):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def forward_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fv[t] = r[t + 1 : t + 1 + h].std(ddof=1)
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
    epd = compute_epd(r)["epd_100"]
    ic_ret = spearman_ic(epd, forward_return(r, 5))
    ic_vol = spearman_ic(epd, forward_vol(r, 5))
    blk = ic_blocks(epd, forward_return(r, 5), block=1000)
    frac_fp = float(np.mean(np.abs(blk) > 0.05)) if blk.size else np.nan
    return {"name": "A_white_noise", "ic_ret_5d": ic_ret, "ic_vol_5d": ic_vol,
            "frac_blocks_absIC_gt_0.05": frac_fp, "pass": bool(frac_fp < 0.05)}


def test_B_garch():
    r = gen_garch()
    epd = compute_epd(r)["epd_100"]
    ic_ret = spearman_ic(epd, forward_return(r, 5))
    ic_vol = spearman_ic(epd, forward_vol(r, 5))
    return {"name": "B_garch", "ic_ret_5d": ic_ret, "ic_vol_5d": ic_vol,
            "pass": bool(abs(ic_ret) < 0.05)}


def test_C_regime():
    r, states = gen_regime()
    epd100 = compute_epd(r)["epd_100"]
    trans = np.where(np.diff(states) != 0)[0] + 1
    hits = 0
    for t in trans:
        lo, hi = max(0, t - 5), min(len(epd100), t + 5)
        if np.any(np.abs(epd100[lo:hi] - 50) > 20):
            hits += 1
    frac = hits / max(1, len(trans))
    return {"name": "C_regime", "n_transitions": int(len(trans)),
            "hit_rate_within_5d": frac, "pass": bool(frac >= 0.5)}


def test_D_divergence():
    r = gen_divergence()
    epd100 = compute_epd(r)["epd_100"]
    half = len(epd100) // 2
    first = epd100[:half][np.isfinite(epd100[:half])]
    second = epd100[half:][np.isfinite(epd100[half:])]
    delta = second.mean() - first.mean()
    sd = np.nanstd(epd100)
    return {"name": "D_divergence",
            "mean_first_half": float(first.mean()),
            "mean_second_half": float(second.mean()),
            "delta_over_sigma": float(delta / sd) if sd > 0 else np.nan,
            "pass": bool(abs(delta) > 0.3 * sd)}


def main():
    results = []
    for fn in (test_A_white_noise, test_B_garch, test_C_regime, test_D_divergence):
        res = fn(); results.append(res)
        flag = "PASS" if res["pass"] else "FAIL"
        extras = {k: v for k, v in res.items() if k not in ("name", "pass")}
        print(f"[Stage1:{res['name']}] {flag}  {extras}")

    passed = all(r["pass"] for r in results)
    print(f"\\n=== Stage 1 Result: {'PASS' if passed else 'FAIL'} ===")

    rpt = ROOT / "docs" / "synthetic_report.md"
    lines = ["# Synthetic Report — Stage 1", "",
             "Fixed params: w_pe=60, m=3, tau=1, w_z=252, tanh_scale=2", ""]
    for res in results:
        lines.append(f"## {res['name']}")
        for k, v in res.items():
            if k != "name":
                lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(f"**Overall**: {'PASS' if passed else 'FAIL'}")
    rpt.write_text("\\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main():
    n = 0
    for rel, content in FILES.items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        n += 1
        print(f"  wrote {rel}  ({p.stat().st_size} bytes)")
    print(f"\nDone. {n} files.")


if __name__ == "__main__":
    main()
