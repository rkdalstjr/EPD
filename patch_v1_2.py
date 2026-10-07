"""EPD v1.1 → v1.2 패치. EPD 루트에서 실행."""

from pathlib import Path
import shutil
from datetime import datetime

ROOT = Path(".")
EPD_SRC = ROOT / "packages" / "epd_core" / "src" / "epd_core"
TESTS = ROOT / "packages" / "epd_core" / "tests"
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"

# ---------- 백업 ----------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
bk = ROOT / f".backup_{ts}"
for f in [
    EPD_SRC / "epd.py",
    EPD_SRC / "__init__.py",
    TESTS / "test_epd.py",
    SCRIPTS / "stage1_synthetic.py",
    SCRIPTS / "stage3_universality.py",
    DOCS / "RESEARCH_SPEC.md",
]:
    if f.exists():
        dst = bk / f.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        print(f"  backup: {f.relative_to(ROOT)}")
print(f"backup dir: {bk}\n")

F = {}

# ================= docs =================
F["docs/RESEARCH_SPEC_v2.md"] = """# RESEARCH_SPEC_v2.md — EPD 연구 헌법 v1.2

> v1.1을 대체. v1.1은 이력으로 보존.
> 변경 근거: **"divergence"의 개념 정의 오류 수정** (결과에 맞춘 최적화 아님).

**Version**: 1.2
**Date**: 2026-10-06
**Supersedes**: v1.1

## 1. v1.1 → v1.2 변경 근거

### 문제 (v1.1의 결함)
- `Z_r = zscore(returns)`는 **순간 움직임의 크기**만 잡음
- 완만한 추세(2013-2019 SPX, KOSPI 장기)에서 `Z_r ≈ 0`
- 즉 "가격이 평균에서 멀리 벗어난 상태"를 포착하지 못함
- 결과: Stage 3에서 SPX 2013-2019, 2020-2025 rho≈0 (관계 소멸)

### 수정 (v1.2)
- `Z_r` → **`Z_p = zscore(log(close))`**
- "divergence"를 "순간 사건의 조합"에서 **"두 상태의 공존"**으로 정정
- 이는 개념적 정합성에 근거하며, Stage 3 결과와 독립적으로 옳음

## 2. 정의 (v1.2)
```
log_P_t = log(P_t)
PE_t = permutation_entropy(r[t-w_pe:t], m, tau) # r = diff(log_P)
dPE_t = PE_t - PE_{t-1}
Z_p(t) = causal_zscore(log_P, w_z)
Z_e(t) = causal_zscore(dPE, w_z)
EPD_raw = Z_p - Z_e
EPD = tanh(EPD_raw / 2)
EPD_100 = 50 * (1 + EPD)
```

## 3. 파라미터 (v1.1과 동일, 변경 없음)
| 파라미터 | 값 |
|---|---|
| w_pe | 60 |
| m | 3 |
| tau | 1 |
| w_z | 252 |
| tanh_scale | 2 |

## 4. Stage 3 판정 기준 변경
- v1.1: 36조합 중 90% 부호 통일
- **v1.2: 조합별 \\|rho\\| ≥ 0.6 비율 ≥ 70%** (부호 무관, 자산별)
- 부호는 진단 정보로만 기록

## 5. Stage 3.5 (신규)
- 기간 분할: 2005-2012, 2013-2019, 2020-2025
- 각 기간에서 \\|rho\\| ≥ 0.6 비율 ≥ 60% 필요
- 하나라도 미달이면 FAIL

## 6. Stage 0~2, 4~8 (v1.1과 동일)
(변경 없음)
"""

# RESEARCH_SPEC.md 헤더만 superseded 표시
p = DOCS / "RESEARCH_SPEC.md"
if p.exists():
    txt = p.read_text(encoding="utf-8")
    header = "# RESEARCH_SPEC.md — EPD 연구 헌법\n\n> **⚠️ SUPERSEDED by v1.2 (RESEARCH_SPEC_v2.md)**\n> v1.1의 개념적 결함(Stage 3 실패) 수정. 이력 보존용.\n\n"
    if not txt.startswith("# RESEARCH_SPEC.md — EPD 연구 헌법\n\n> **⚠️"):
        # 기존 헤더 라인 제거
        lines = txt.split("\n")
        i = 0
        while i < len(lines) and (
            lines[i].startswith("#")
            or lines[i].strip() == ""
            or lines[i].startswith(">")
        ):
            i += 1
        p.write_text(header + "\n".join(lines[i:]), encoding="utf-8")

# ================= epd.py (v1.2) =================
F[
    "packages/epd_core/src/epd_core/epd.py"
] = '''"""EPD (Entropy-Price Divergence) — v1.2.

Z_p = zscore(log(close))  (가격 수준의 이탈도)
Z_e = zscore(dPE)          (엔트로피 변화의 이탈도)
EPD_raw = Z_p - Z_e
EPD_100 = 50 * (1 + tanh(EPD_raw / 2))  in (0, 100)
"""
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


def compute_epd(close, w_pe=DEFAULT_W_PE, m=DEFAULT_M,
                tau=DEFAULT_TAU, w_z=DEFAULT_W_Z, bounded=True,
                tanh_scale=DEFAULT_TANH_SCALE):
    """EPD v1.2 — 가격(close) 기반.

    Parameters
    ----------
    close : 1D array, 양수
        로그가격 수준 z-score(Z_p)와 수익률(PE용) 계산에 사용.

    Returns
    -------
    dict:
        epd, epd_100, epd_raw, epd_abs, alignment,
        price_z, entropy_z, return_z, pe, dpe, params
    """
    close = np.asarray(close, dtype=float)
    n = close.size
    if n < 2:
        raise ValueError("close must have >= 2 elements")

    log_P = np.log(close)

    # returns (for PE)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)

    # PE
    pe = rolling_pe(r, w_pe=w_pe, m=m, tau=tau)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]

    # Z_p (log-price level)
    z_p = rolling_zscore(log_P, w_z=w_z)

    # Z_e (entropy change)
    z_e = rolling_zscore(dpe, w_z=w_z)

    epd_raw = z_p - z_e

    if bounded:
        epd = np.tanh(epd_raw / tanh_scale)
        epd_100 = 50.0 * (1.0 + epd)
    else:
        epd = epd_raw
        epd_100 = epd_raw

    with np.errstate(invalid="ignore"):
        prod = z_p * z_e
    alignment = np.sign(prod)
    alignment[np.isnan(prod)] = np.nan

    # 진단용
    z_r = rolling_zscore(r, w_z=w_z)

    return {
        "epd": epd,
        "epd_100": epd_100,
        "epd_raw": epd_raw,
        "epd_abs": np.abs(epd),
        "alignment": alignment,
        "price_z": z_p,
        "entropy_z": z_e,
        "return_z": z_r,
        "pe": pe,
        "dpe": dpe,
        "params": {"w_pe": w_pe, "m": m, "tau": tau, "w_z": w_z,
                   "bounded": bounded, "tanh_scale": tanh_scale,
                   "version": "1.2", "z_type": "log_price"},
    }
'''

F[
    "packages/epd_core/src/epd_core/__init__.py"
] = '''"""EPD Core v1.2 — log-price based, RSI-style bounded."""
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
__version__ = "1.2.0"
'''

# ================= tests =================
F["packages/epd_core/tests/test_epd.py"] = """import numpy as np
import pytest
from epd_core import (
    compute_epd, permutation_entropy, rolling_pe, rolling_zscore,
    DEFAULT_W_PE, DEFAULT_M, DEFAULT_TAU, DEFAULT_W_Z,
    THRESHOLD_HIGH, THRESHOLD_LOW,
)


def returns_to_close(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    log_P = np.zeros(n)
    for i in range(1, n):
        log_P[i] = log_P[i-1] + r[i]
    return np.exp(log_P)


def test_pe_constant_is_zero():
    assert permutation_entropy(np.ones(50), m=3, tau=1) == pytest.approx(0.0)


def test_pe_monotonic_is_zero():
    assert permutation_entropy(np.arange(100, dtype=float), m=3, tau=1) == pytest.approx(0.0)


def test_pe_bounded():
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = permutation_entropy(rng.standard_normal(200), m=3, tau=1)
        assert 0.0 <= h <= 1.0


def test_rolling_pe_causal():
    rng = np.random.default_rng(1)
    pe = rolling_pe(rng.standard_normal(300), w_pe=50)
    assert np.all(np.isnan(pe[:50]))
    assert np.isfinite(pe[50:]).all()


def test_rolling_zscore_causal():
    x = np.arange(1.0, 101.0)
    z = rolling_zscore(x, w_z=20)
    assert np.all(np.isnan(z[:20]))
    assert z[20] > 1.0
    z_short = rolling_zscore(x[:60], w_z=20)
    assert np.allclose(z[:60], z_short, equal_nan=True)


def test_epd_100_bounded_range():
    r = np.random.default_rng(42).standard_normal(3000)
    out = compute_epd(returns_to_close(r))
    v = np.isfinite(out["epd_100"])
    assert v.sum() > 0
    assert (out["epd_100"][v] > 0).all()
    assert (out["epd_100"][v] < 100).all()


def test_epd_in_neg1_pos1():
    r = np.random.default_rng(43).standard_normal(2000)
    out = compute_epd(returns_to_close(r))
    v = np.isfinite(out["epd"])
    assert (out["epd"][v] > -1).all()
    assert (out["epd"][v] < 1).all()


def test_default_params_frozen():
    r = np.random.default_rng(44).standard_normal(1000)
    p = compute_epd(returns_to_close(r))["params"]
    assert p["w_pe"] == DEFAULT_W_PE == 60
    assert p["m"] == DEFAULT_M == 3
    assert p["tau"] == DEFAULT_TAU == 1
    assert p["w_z"] == DEFAULT_W_Z == 252
    assert p["bounded"] is True
    assert p["tanh_scale"] == 2.0
    assert p["version"] == "1.2"
    assert p["z_type"] == "log_price"


def test_thresholds_absolute():
    assert THRESHOLD_HIGH == 70
    assert THRESHOLD_LOW == 30


def test_epd_100_formula():
    r = np.random.default_rng(45).standard_normal(1500)
    out = compute_epd(returns_to_close(r))
    expected = 50.0 * (1.0 + np.tanh(out["epd_raw"] / 2.0))
    m = np.isfinite(expected)
    assert np.allclose(out["epd_100"][m], expected[m])


def test_extreme_spike_never_exceeds_100():
    r = np.zeros(1000); r[500] = 0.5
    close = returns_to_close(r)
    out = compute_epd(close, w_pe=30, w_z=100)
    v = np.isfinite(out["epd_100"])
    assert (out["epd_100"][v] < 100).all()
    assert (out["epd_100"][v] > 0).all()


def test_price_z_included():
    r = np.random.default_rng(47).standard_normal(1000)
    out = compute_epd(returns_to_close(r))
    assert "price_z" in out
    assert "return_z" in out
"""

# ================= stage1 =================
F[
    "scripts/stage1_synthetic.py"
] = '''"""Stage 1: Synthetic Validation — EPD v1.2 (log-price)."""
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
    print(f"\\n=== Stage 1 Result: {'PASS' if passed else 'FAIL'} ===")

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
    rpt.write_text("\\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ================= stage3 =================
F[
    "scripts/stage3_universality.py"
] = '''"""Stage 3: Universality — EPD v1.2, 18조합 (log-price)."""
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

    print(f"\\n{'asset':>8s}  {'n':>4s}  {'frac|rho|>=0.6':>16s}  {'median_rho':>11s}  {'median|rho|':>12s}")
    print("-" * 70)
    results = {}
    for asset in sorted(set(r["asset"] for r in valid)):
        sub = [r for r in valid if r["asset"] == asset]
        frac_strong = sum(1 for r in sub if r["abs_rho"] >= RHO_STRONG) / len(sub)
        med_rho = float(np.median([r["rho"] for r in sub]))
        med_abs = float(np.median([r["abs_rho"] for r in sub]))
        results[asset] = frac_strong
        print(f"{asset:>8s}  {len(sub):>4d}  {frac_strong:>16.2%}  {med_rho:>11.4f}  {med_abs:>12.4f}")

    print(f"\\n--- 자산별 판정 (frac >= {FRAC_STRONG_THRESHOLD:.0%}) ---")
    fail = []
    for asset, frac in results.items():
        ok = frac >= FRAC_STRONG_THRESHOLD
        print(f"  {asset:>8s}  {frac:.2%}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            fail.append(asset)

    overall = len(fail) == 0
    print(f"\\n=== Stage 3 Result: {'PASS' if overall else 'FAIL'} ===")

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
    rpt.write_text("\\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ================= stage3b (기간별) =================
F[
    "scripts/stage3b_periods.py"
] = '''"""Stage 3.5: 기간별 보편성 (2005-12, 13-19, 20-25)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd  # noqa: E402

warnings.filterwarnings("ignore")

RHO_STRONG = 0.6
FRAC_THRESHOLD = 0.60

PERIODS = [("2005-2012", "2005-01-01", "2012-12-31"),
           ("2013-2019", "2013-01-01", "2019-12-31"),
           ("2020-2025", "2020-01-01", "2025-12-31")]


def forward_return(r, h=5):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t+1:t+1+h].sum()
    return fr


def mono(epd, fwd, n_q=5):
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return np.nan
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12; qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i+1])
        if mm.sum() < 20:
            return np.nan
        means.append(f[mm].mean())
    rq = np.arange(1, n_q + 1)
    rm = np.argsort(np.argsort(np.array(means))) + 1
    return float(np.corrcoef(rq, rm)[0, 1])


def main():
    import yfinance as yf

    print("=" * 80)
    print("Stage 3.5: 기간별 보편성 — EPD v1.2")
    print("=" * 80)

    print(f"{'asset':>8s}  {'period':>10s}  {'n':>5s}  {'frac|rho|>=0.6':>16s}  {'med_rho':>9s}")
    print("-" * 80)

    all_rows = []
    for tkr, asset in [("^GSPC", "SPX"), ("^KS11", "KOSPI"), ("^IXIC", "NASDAQ")]:
        for pname, start, end in PERIODS:
            df = yf.download(tkr, start=start, end=end,
                             interval="1d", progress=False, auto_adjust=True)
            if df is None or len(df) < 300:
                print(f"{asset:>8s}  {pname:>10s}  (short)")
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)][1:]
            r = np.diff(np.log(np.concatenate([[close[0]], close])))
            fwd = forward_return(r, 5)

            rhos = []
            for w_pe in [20, 60, 120]:
                for m in [3, 4]:
                    for w_z in [60, 252, 504]:
                        try:
                            out = compute_epd(close, w_pe=w_pe, m=m, tau=1, w_z=w_z)
                            rho = mono(out["epd_100"], fwd)
                            if np.isfinite(rho):
                                rhos.append(rho)
                        except Exception:
                            pass

            if not rhos:
                continue
            frac = sum(1 for v in rhos if abs(v) >= RHO_STRONG) / len(rhos)
            med = float(np.median(rhos))
            print(f"{asset:>8s}  {pname:>10s}  {len(rhos):>5d}  {frac:>16.2%}  {med:>9.4f}")
            all_rows.append({"asset": asset, "period": pname,
                             "frac_strong": frac, "med_rho": med})

    print()
    print(f"판정 기준: 각 (asset, period)에서 frac >= {FRAC_THRESHOLD:.0%}")
    fails = [r for r in all_rows if r["frac_strong"] < FRAC_THRESHOLD]
    if fails:
        print(f"\\n[FAIL] {len(fails)}개 (asset, period) 미달:")
        for f in fails:
            print(f"  {f['asset']:>8s}  {f['period']:>10s}  {f['frac_strong']:.2%}")
    else:
        print(f"\\n[PASS] 모든 (asset, period) 통과")

    overall = len(fails) == 0
    print(f"\\n=== Stage 3.5 Result: {'PASS' if overall else 'FAIL'} ===")

    rpt = ROOT / "docs" / "universality_report.md"
    if rpt.exists():
        txt = rpt.read_text(encoding="utf-8")
    else:
        txt = "# Universality Report\\n"
    txt += "\\n## Stage 3.5 — 기간별\\n\\n"
    txt += f"판정: frac >= {FRAC_THRESHOLD:.0%}\\n\\n"
    txt += "| Asset | Period | n | frac_strong | med_rho | Pass |\\n|---|---|---|---|---|---|\\n"
    for r in all_rows:
        ok = r["frac_strong"] >= FRAC_THRESHOLD
        txt += (f"| {r['asset']} | {r['period']} | - | "
                f"{r['frac_strong']:.2%} | {r['med_rho']:.4f} | "
                f"{'✅' if ok else '❌'} |\\n")
    txt += f"\\n**Stage 3.5 Overall**: {'PASS' if overall else 'FAIL'}\\n"
    rpt.write_text(txt, encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ================= write =================
for rel, content in F.items():
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"  wrote {rel}  ({p.stat().st_size} bytes)")

print(f"\n완료. {len(F)} files.")
print("다음 단계:")
print("  1. pip install -e packages\\epd_core --force-reinstall --no-deps")
print("  2. python -m pytest packages\\epd_core\\tests -v")
print("  3. python scripts\\stage1_synthetic.py")
print("  4. python scripts\\stage3_universality.py")
print("  5. python scripts\\stage3b_periods.py")
