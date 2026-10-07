"""Stage 1: Synthetic Validation — EPD_100."""

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
    r = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / max(1e-9, 1 - alpha - beta)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * eps[t]
    return r


def gen_regime(n=N, p_stay=0.98, seed=11):
    rng = np.random.default_rng(seed)
    state = 0
    states = np.zeros(n, dtype=int)
    for t in range(n):
        if rng.random() > p_stay:
            state = 1 - state
        states[t] = state
    sig = np.where(states == 0, 1.0, 3.0)
    return rng.standard_normal(n) * sig, states


def gen_divergence(n=N, seed=13):
    """Divergence 이벤트 삽입: iid 기반 + 이벤트 블록에서 (강한 drift + 높은 자기상관).

    - r의 자기상관이 높아지면 PE 급락 (예측 가능성 ↑)
    - 동시에 drift 추가 → Z_r 양수
    - 두 효과가 겹치면 EPD_raw = Z_r − Z_e가 크게 양수
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * 1.0

    # 이벤트 1: 3000~4000 (강한 상승 + 자기상관)
    phi = 0.9
    block1 = np.zeros(1000)
    for i in range(1, 1000):
        block1[i] = 0.5 + phi * block1[i - 1] + rng.standard_normal() * 0.3
    r[3000:4000] = block1

    # 이벤트 2: 7000~8000 (강한 하락 + 자기상관)
    block2 = np.zeros(1000)
    for i in range(1, 1000):
        block2[i] = -0.5 + phi * block2[i - 1] + rng.standard_normal() * 0.3
    r[7000:8000] = block2

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
    ics = [
        spearman_ic(x[i : i + block], y[i : i + block]) for i in range(0, len(x), block)
    ]
    return np.array([v for v in ics if np.isfinite(v)])


def test_A_white_noise():
    r = gen_white_noise()
    epd = compute_epd(r)["epd_100"]
    fr5 = forward_return(r, 5)
    fv5 = forward_vol(r, 5)

    ic_ret = spearman_ic(epd, fr5)
    ic_vol = spearman_ic(epd, fv5)

    # 진단용: 1000샘플 블록에서 2σ(=2/√1000≈0.063) 초과 비율
    blk = ic_blocks(epd, fr5, block=1000)
    sigma_blk = 1.0 / np.sqrt(1000.0)
    frac_2sigma = float(np.mean(np.abs(blk) > 2 * sigma_blk)) if blk.size else np.nan

    # 판정: 전체 IC가 임계값 이하 (블록 통계는 진단용)
    passed = (abs(ic_ret) < 0.05) and (abs(ic_vol) < 0.05)
    return {
        "name": "A_white_noise",
        "ic_ret_5d": ic_ret,
        "ic_vol_5d": ic_vol,
        "frac_blocks_2sigma": frac_2sigma,
        "pass": bool(passed),
    }


def test_B_garch():
    r = gen_garch()
    epd = compute_epd(r)["epd_100"]
    ic_ret = spearman_ic(epd, forward_return(r, 5))
    ic_vol = spearman_ic(epd, forward_vol(r, 5))
    return {
        "name": "B_garch",
        "ic_ret_5d": ic_ret,
        "ic_vol_5d": ic_vol,
        "pass": bool(abs(ic_ret) < 0.05),
    }


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
    return {
        "name": "C_regime",
        "n_transitions": int(len(trans)),
        "hit_rate_within_5d": frac,
        "pass": bool(frac >= 0.5),
    }


def test_D_divergence():
    """이벤트 구간 초반에서 EPD_raw가 극단값을 보여야 함.

    진입 시점(이벤트 시작 후 20~100 샘플)에 |EPD_raw| > 1.5 가
    배경(background) 대비 유의하게 자주 발생해야 함.
    """
    r = gen_divergence()
    out = compute_epd(r)
    raw = out["epd_raw"]
    n = len(raw)

    # 이벤트 진입 구간 (진입 직후 100 샘플)
    event_idx = np.zeros(n, dtype=bool)
    event_idx[3020:3100] = True
    event_idx[7020:7100] = True

    bg_idx = np.isfinite(raw) & ~event_idx
    ev_idx = np.isfinite(raw) & event_idx

    thr = 1.5
    frac_ev = float(np.mean(np.abs(raw[ev_idx]) > thr)) if ev_idx.sum() else 0.0
    frac_bg = float(np.mean(np.abs(raw[bg_idx]) > thr)) if bg_idx.sum() else 0.0

    # 판정: 이벤트 구간이 배경의 2배 이상 + 이벤트 구간 20% 이상
    passed = (frac_ev > 0.2) and (frac_ev > 2 * frac_bg)

    return {
        "name": "D_divergence",
        "frac_extreme_event_window": frac_ev,
        "frac_extreme_background": frac_bg,
        "threshold_abs_epd_raw": thr,
        "pass": bool(passed),
    }


def main():
    results = []
    for fn in (test_A_white_noise, test_B_garch, test_C_regime, test_D_divergence):
        res = fn()
        results.append(res)
        flag = "PASS" if res["pass"] else "FAIL"
        extras = {k: v for k, v in res.items() if k not in ("name", "pass")}
        print(f"[Stage1:{res['name']}] {flag}  {extras}")

    passed = all(r["pass"] for r in results)
    print(f"\n=== Stage 1 Result: {'PASS' if passed else 'FAIL'} ===")

    rpt = ROOT / "docs" / "synthetic_report.md"
    lines = [
        "# Synthetic Report — Stage 1",
        "",
        "Fixed params: w_pe=60, m=3, tau=1, w_z=252, tanh_scale=2",
        "",
    ]
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
