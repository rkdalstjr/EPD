"""Stage 1 — Synthetic Validation.

4 tests: White noise, GARCH, Regime switch, Divergence event.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# v2/packages/epd_core/src 를 path에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import compute_epd  # noqa: E402

SEED = 20261006
N = 10_000


# ============================================================
#  Synthetic data generators
# ============================================================
def gen_white_noise(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def gen_garch(n=N, alpha=0.10, beta=0.85, seed=SEED):
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


def gen_regime(n=N, p_stay=0.98, seed=SEED):
    rng = np.random.default_rng(seed)
    state = 0
    states = np.zeros(n, dtype=int)
    for t in range(n):
        if rng.random() > p_stay:
            state = 1 - state
        states[t] = state
    sigma = np.where(states == 0, 1.0, 3.0)
    r = rng.standard_normal(n) * sigma
    return r, states


def gen_divergence(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n)
    phi = 0.9
    b1 = np.zeros(1000)
    for i in range(1, 1000):
        b1[i] = 0.5 + phi * b1[i - 1] + rng.standard_normal() * 0.3
    r[3000:4000] = b1
    b2 = np.zeros(1000)
    for i in range(1, 1000):
        b2[i] = -0.5 + phi * b2[i - 1] + rng.standard_normal() * 0.3
    r[7000:8000] = b2
    return r


# ============================================================
#  Utilities
# ============================================================
def returns_to_close(r, scale=0.01):
    """cumsum이 무한히 커지지 않도록 scale로 압축.
    EPD는 scale-invariant (z-score 정규화)라서 결과에 영향 없음.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    log_P = np.zeros(n)
    for i in range(1, n):
        log_P[i] = log_P[i - 1] + r[i] * scale
    return np.exp(log_P)


def forward_return(r, h=5):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def forward_vol(r, h=5):
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


def ic_blocks(x, y, block=1000):
    ics = []
    for i in range(0, len(x), block):
        ics.append(spearman(x[i : i + block], y[i : i + block]))
    return np.array([v for v in ics if np.isfinite(v)])


# ============================================================
#  Tests
# ============================================================
def test_A_white_noise():
    r = gen_white_noise()
    close = returns_to_close(r)
    out = compute_epd(close)
    epd100 = out["epd_100"]

    ic_ret = spearman(epd100, forward_return(r, 5))
    ic_vol = spearman(epd100, forward_vol(r, 5))
    blk = ic_blocks(epd100, forward_return(r, 5), block=1000)
    sigma_blk = 1.0 / np.sqrt(1000.0)
    frac_2sigma = float(np.mean(np.abs(blk) > 2 * sigma_blk)) if blk.size else 1.0

    passed = abs(ic_ret) < 0.05 and abs(ic_vol) < 0.05 and frac_2sigma < 0.30
    return {
        "name": "A_white_noise",
        "ic_ret_5d": ic_ret,
        "ic_vol_5d": ic_vol,
        "frac_blocks_2sigma": frac_2sigma,
        "pass": bool(passed),
    }


def test_B_garch():
    r = gen_garch()
    close = returns_to_close(r)
    out = compute_epd(close)
    epd100 = out["epd_100"]

    ic_ret = spearman(epd100, forward_return(r, 5))
    ic_vol = spearman(epd100, forward_vol(r, 5))

    passed = abs(ic_ret) < 0.05
    return {
        "name": "B_garch",
        "ic_ret_5d": ic_ret,
        "ic_vol_5d": ic_vol,
        "pass": bool(passed),
    }


def test_C_regime():
    """raw EPD 사용 — EMA는 regime 전환 감지에 부적합."""
    r, states = gen_regime()
    close = returns_to_close(r)
    out = compute_epd(close)
    epd_raw100 = out["epd_raw_100"]  # ★ raw

    trans = np.where(np.diff(states) != 0)[0] + 1
    hits = 0
    for t in trans:
        lo, hi = max(0, t - 5), min(len(epd_raw100), t + 5)
        if np.any(np.abs(epd_raw100[lo:hi] - 50) > 20):
            hits += 1
    frac = hits / max(1, len(trans))

    passed = frac >= 0.50
    return {
        "name": "C_regime",
        "n_transitions": int(len(trans)),
        "hit_rate_within_5d": frac,
        "pass": bool(passed),
    }


def test_D_divergence():
    r = gen_divergence()
    close = returns_to_close(r)
    out = compute_epd(close)
    raw = out["epd_raw"]
    n = len(raw)

    event_idx = np.zeros(n, dtype=bool)
    event_idx[3020:3100] = True
    event_idx[7020:7100] = True

    bg_idx = np.isfinite(raw) & ~event_idx
    ev_idx = np.isfinite(raw) & event_idx

    thr = 1.5
    fe = float(np.mean(np.abs(raw[ev_idx]) > thr)) if ev_idx.sum() else 0.0
    fb = float(np.mean(np.abs(raw[bg_idx]) > thr)) if bg_idx.sum() else 0.0

    passed = (fe > 0.20) and (fe > 2 * fb)
    return {
        "name": "D_divergence",
        "frac_extreme_event": fe,
        "frac_extreme_bg": fb,
        "ratio_event_bg": fe / fb if fb > 0 else np.nan,
        "pass": bool(passed),
    }


# ============================================================
#  Runner
# ============================================================
def main():
    print("=" * 80)
    print("Stage 1 — Synthetic Validation (v2.0)")
    print(f"Seed={SEED}, N={N}")
    print("=" * 80)

    tests = [test_A_white_noise, test_B_garch, test_C_regime, test_D_divergence]

    results = []
    for fn in tests:
        res = fn()
        results.append(res)
        flag = "PASS" if res["pass"] else "FAIL"
        extras = {k: v for k, v in res.items() if k not in ("name", "pass")}
        print(f"[{res['name']:>14s}] {flag}  {extras}")

    n_pass = sum(1 for r in results if r["pass"])
    overall = n_pass == len(results)
    print()
    print(
        f"=== Stage 1 Result: {n_pass}/{len(results)} "
        f"({'PASS' if overall else 'FAIL'}) ==="
    )

    # 리포트 저장
    rpt = ROOT / "docs" / "STAGE_1_SYNTHETIC.md"
    lines = [
        "# Stage 1 — Synthetic Validation",
        "",
        f"- Seed: {SEED}",
        f"- N: {N}",
        f"- 결과: **{n_pass}/{len(results)} " f"({'PASS' if overall else 'FAIL'})**",
        "",
        "## 결과 표",
        "",
        "| Test | 핵심 지표 | 판정 |",
        "|---|---|---|",
    ]
    for res in results:
        if res["name"] == "A_white_noise":
            detail = (
                f"IC_ret={res['ic_ret_5d']:.4f}, "
                f"IC_vol={res['ic_vol_5d']:.4f}, "
                f"2σ={res['frac_blocks_2sigma']:.0%}"
            )
        elif res["name"] == "B_garch":
            detail = f"IC_ret={res['ic_ret_5d']:.4f}, " f"IC_vol={res['ic_vol_5d']:.4f}"
        elif res["name"] == "C_regime":
            detail = (
                f"전환 {res['n_transitions']}회, "
                f"적중 {res['hit_rate_within_5d']:.1%}"
            )
        elif res["name"] == "D_divergence":
            detail = (
                f"이벤트 {res['frac_extreme_event']:.1%}, "
                f"배경 {res['frac_extreme_bg']:.1%}, "
                f"비율 {res['ratio_event_bg']:.2f}"
            )
        else:
            detail = ""
        flag = "✅ PASS" if res["pass"] else "❌ FAIL"
        lines.append(f"| {res['name']} | {detail} | {flag} |")

    lines.append("")
    lines.append(f"**Overall**: {'PASS' if overall else 'FAIL'}")
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
