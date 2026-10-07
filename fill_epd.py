"""epd_project 안의 파일들에 내용 채우기."""

from pathlib import Path

ROOT = Path("epd_project")

FILES = {}

# ================= docs =================
FILES["docs/RESEARCH_SPEC.md"] = """# RESEARCH_SPEC.md — EPD 프로젝트 연구 헌법

> **동결 선언**: 본 문서는 실험 중 수정 금지. 수정 시 `RESEARCH_SPEC_v2.md` 생성.
> **철학**: RSI처럼 어떤 자산·시대에도 동일 파라미터로 작동하는 지표.
> **금지**: 파라미터 최적화, ML 피처 결합.

**Version**: 1.1
**Date**: 2026-10-06

## 0. RSI 4원칙
- P1. 유계 출력 (EPD_100 ∈ (0,100))
- P2. 이론적 파라미터 (탐색 금지)
- P3. 스케일 불변 (로그수익률 + causal z-score)
- P4. 학습 없음 (순수 공식)

## 1. 가설
- H1: EPD_100 >70 또는 <30 이후 추세 지속성 약화
- H2: |EPD| 극단값 -> 미래 변동성 증가
- H3: EPD는 MOM/VOL/PE와 직교 (|corr|<0.5)
- **Primary**: H1 AND H3

## 2. 실패 조건 (사전 고정)
| 조건 | 판정 |
|---|---|
| A. 구조적 중복 | corr(EPD,MOM)>0.5 OR corr(EPD,VOL)>0.5 |
| B. Null FP | White noise 블록 중 |IC|>0.05 가 5% 이상 |
| C. OOS 소멸 | IS Sharpe 개선>0.2, OOS<0.05 |
| D. 파라미터 비보편성 | 36조합 중 90% 미만 동일 방향 |
| E. 비용 후 소멸 | 왕복 0.4354% 반영 후 초과수익<0 |

## 3. EPD 정의 (확정)
```
r_t = log(P_t / P_{t-1})
PE_t = permutation_entropy(r[t-w_pe:t], m, tau)
dPE_t = PE_t - PE_{t-1}
Z_r(t) = causal_zscore(r, w_z)
Z_e(t) = causal_zscore(dPE, w_z)
EPD_raw = Z_r - Z_e
EPD = tanh(EPD_raw / 2)
EPD_100 = 50 * (1 + EPD)
```

## 4. 파라미터 (고정, 변경 금지)
| 파라미터 | 값 | 근거 |
|---|---|---|
| w_pe | 60 | 6패턴 x 10관측, 분기 |
| m | 3 | Bandt-Pompe 2002 표준 |
| tau | 1 | 일봉 표준 |
| w_z | 252 | 1년 거래일 |
| tanh_scale | 2 | RSI 70/30 정렬 |

## 5. 임계값 (자산 무관)
- EPD_100 > 70: 과열 위험
- EPD_100 < 30: 구조 형성 중

## 6. Stage 1 판정
- A (White noise): 블록 |IC|>0.05 비율 <5%
- B (GARCH): return IC ~ 0
- C (Regime): 전환 +-5일 내 |EPD_100-50|>20 비율 >=50%
- D (Divergence): 후반부 EPD_100 편향 |delta| > 0.3 sigma
"""

FILES["docs/EPD_SPEC.md"] = """# EPD_SPEC.md
> 상태: DRAFT — Stage 3 완료 후 작성
> 예정 작성일: TBD
"""

for name in [
    "synthetic_report",
    "universality_report",
    "realdata_report",
    "walkforward_report",
    "final_decision",
]:
    FILES[f"docs/{name}.md"] = (
        f"# {name.replace('_', ' ').title()}\n> (자동 생성 예정)\n"
    )

# ================= package =================
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
    """Causal rolling z."""
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
] = '''"""EPD — RSI-style bounded indicator."""
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

FILES["packages/epd_core/src/epd_core/__init__.py"] = '''"""EPD Core."""
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

FILES["packages/epd_core/tests/test_epd.py"] = """import numpy as np
import pytest
from epd_core import (
    compute_epd, permutation_entropy, rolling_pe, rolling_zscore,
    DEFAULT_W_PE, DEFAULT_M, DEFAULT_TAU, DEFAULT_W_Z,
    THRESHOLD_HIGH, THRESHOLD_LOW,
)


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
    out = compute_epd(np.random.default_rng(42).standard_normal(3000))
    v = np.isfinite(out["epd_100"])
    assert v.sum() > 0
    assert (out["epd_100"][v] > 0).all()
    assert (out["epd_100"][v] < 100).all()


def test_epd_in_neg1_pos1():
    out = compute_epd(np.random.default_rng(43).standard_normal(2000))
    v = np.isfinite(out["epd"])
    assert (out["epd"][v] > -1).all()
    assert (out["epd"][v] < 1).all()


def test_default_params_frozen():
    p = compute_epd(np.random.default_rng(44).standard_normal(1000))["params"]
    assert p["w_pe"] == DEFAULT_W_PE == 60
    assert p["m"] == DEFAULT_M == 3
    assert p["tau"] == DEFAULT_TAU == 1
    assert p["w_z"] == DEFAULT_W_Z == 252
    assert p["bounded"] is True
    assert p["tanh_scale"] == 2.0


def test_thresholds_absolute():
    assert THRESHOLD_HIGH == 70
    assert THRESHOLD_LOW == 30


def test_epd_100_formula():
    out = compute_epd(np.random.default_rng(45).standard_normal(1500))
    expected = 50.0 * (1.0 + np.tanh(out["epd_raw"] / 2.0))
    m = np.isfinite(expected)
    assert np.allclose(out["epd_100"][m], expected[m])


def test_unbounded_option():
    out = compute_epd(np.random.default_rng(46).standard_normal(500), bounded=False)
    m = np.isfinite(out["epd"])
    assert np.allclose(out["epd"][m], out["epd_raw"][m])


def test_extreme_spike_never_exceeds_100():
    r = np.zeros(1000); r[500] = 50.0
    out = compute_epd(r, w_pe=30, w_z=100)
    v = np.isfinite(out["epd_100"])
    assert (out["epd_100"][v] < 100).all()
    assert (out["epd_100"][v] > 0).all()
"""

# ================= scripts =================
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

for s in [
    "stage3_universality",
    "stage4_realdata",
    "stage5_orthogonality",
    "stage6_walkforward",
]:
    FILES[f"scripts/{s}.py"] = f'"""TODO: {s}."""\n'

FILES[".gitignore"] = """__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
data/
*.parquet
*.csv
"""


def main():
    n = 0
    for rel, content in FILES.items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        n += 1
        print(f"  wrote {rel}")
    print(f"\nDone. {n} files written under {ROOT.resolve()}")


if __name__ == "__main__":
    main()
