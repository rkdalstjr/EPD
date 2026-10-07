"""Stage 7 — Representation Selection (R1, R3, R4, R5, R6).

R1 = |Zr|
R3 = |Zr − Ze|              (기존 EPD)
R4 = ||Zr| − Ze|            (Stage 6 최우수)
R5 = |Zr − Ze| / (|Zr|+|Ze|+ε)
R6 = ||Zr| − |Ze||          (신규 — 완전 대칭 divergence)

평가:
  1. Marginal Spearman
  2. Partial | |Zr|
  3. Block bootstrap 95% CI (block=20, B=500)
  4. Multiple horizons (vol_5, vol_10, vol_20)
  5. 5개 핵심 시장 (SPX, NASDAQ, KOSPI, KOSDAQ, + 개별주 secondary)

최종 선택:
  - Partial이 안정적으로 0 초과
  - Block bootstrap CI가 0을 넘지 않음
  - 여러 horizon에서 일관
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
    ("^GSPC", "SPX", "US"),
    ("^IXIC", "NASDAQ", "US"),
    ("^KS11", "KOSPI", "KR"),
    ("^KQ11", "KOSDAQ", "KR"),
    ("005930.KS", "삼성전자", "KR2"),
    ("000660.KS", "SK하이닉스", "KR2"),
    ("035420.KS", "NAVER", "KR2"),
    ("051910.KS", "LG화학", "KR2"),
]

EPS = 1e-6
BLOCK = 20
B = 500
SEED = 42


def fwd_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


TARGETS = {
    "vol_5": lambda r: fwd_vol(r, 5),
    "vol_10": lambda r: fwd_vol(r, 10),
    "vol_20": lambda r: fwd_vol(r, 20),
}


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def partial_corr(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 100:
        return np.nan
    x, y, z = x[m], y[m], z[m]
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def block_bootstrap_ci(x, y, block=BLOCK, B=B, seed=SEED):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return np.nan, np.nan
    x, y = x[m], y[m]
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    rhos = []
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        rho = spearman(x[idx], y[idx])
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return np.nan, np.nan
    rhos = np.array(rhos)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def build_reps(z_r, z_e):
    return {
        "R1": np.abs(z_r),
        "R3": np.abs(z_r - z_e),
        "R4": np.abs(np.abs(z_r) - z_e),
        "R5": np.abs(z_r - z_e) / (np.abs(z_r) + np.abs(z_e) + EPS),
        "R6": np.abs(np.abs(z_r) - np.abs(z_e)),
    }


def main():
    import yfinance as yf

    print("=" * 130)
    print("Stage 7 — Representation Selection (R1, R3, R4, R5, R6)")
    print("=" * 130)

    reps = ["R1", "R3", "R4", "R5", "R6"]
    targets = list(TARGETS.keys())

    results = {}  # (asset, rep, target) -> dict
    asset_grp = {}

    for tkr, name, grp in TICKERS:
        try:
            df = yf.download(
                tkr,
                start="2005-01-01",
                end="2025-12-31",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < 1000:
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 1000:
                continue
        except Exception:
            continue

        asset_grp[name] = grp

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))

        out = compute_epd_v3(close)
        z_r, z_e = out["z_r"], out["z_e"]
        abs_zr = np.abs(z_r)

        rep_vals = build_reps(z_r, z_e)

        for rep_name, rep in rep_vals.items():
            for tgt_name, fn in TARGETS.items():
                tgt = fn(r)
                rho = spearman(rep, tgt)
                part = partial_corr(rep, tgt, abs_zr)
                lo, hi = block_bootstrap_ci(rep, tgt, block=BLOCK, B=B)
                results[(name, rep_name, tgt_name)] = {
                    "rho": rho,
                    "partial": part,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }

    # ============================================================
    # 요약: Rep × Target 평균
    # ============================================================
    def summarize(metric, groups=None):
        print(f"\n### {metric}")
        print(
            f"  {'rep':>4s}  "
            + "  ".join(f"{t:>9s}" for t in targets)
            + f"  {'mean':>9s}"
        )
        print("  " + "-" * 55)
        for rep in reps:
            vals = []
            for tgt in targets:
                sub = [
                    results[(a, rep, tgt)][metric]
                    for a in asset_grp
                    if (a, rep, tgt) in results
                    and (groups is None or asset_grp[a] in groups)
                    and np.isfinite(results[(a, rep, tgt)][metric])
                ]
                vals.append(np.mean(sub) if sub else np.nan)
            print(
                f"  {rep:>4s}  "
                + "  ".join(f"{v:>+9.4f}" for v in vals)
                + f"  {np.nanmean(vals):>+9.4f}"
            )

    print("\n" + "=" * 130)
    print("자산 평균 (8자산)")
    print("=" * 130)
    summarize("rho")
    summarize("partial")

    print("\n" + "=" * 130)
    print("핵심 시장만 (SPX, NASDAQ, KOSPI, KOSDAQ)")
    print("=" * 130)
    summarize("rho", groups={"US", "KR"})
    summarize("partial", groups={"US", "KR"})

    # ============================================================
    # Block bootstrap CI (vol_5)
    # ============================================================
    print("\n" + "=" * 130)
    print("Block Bootstrap 95% CI (vol_5) — R4 vs R6")
    print("=" * 130)
    print(
        f"  {'asset':>12s}  "
        f"{'R4 rho':>10s}  {'R4 CI':>20s}  "
        f"{'R6 rho':>10s}  {'R6 CI':>20s}"
    )
    print("  " + "-" * 90)

    r4_sig = 0
    r6_sig = 0
    n_assets = 0
    for name in asset_grp:
        n_assets += 1
        r4 = results.get((name, "R4", "vol_5"), {})
        r6 = results.get((name, "R6", "vol_5"), {})
        r4_ok = (
            np.isfinite(r4.get("ci_lo", np.nan))
            and np.isfinite(r4.get("ci_hi", np.nan))
            and r4["ci_lo"] > 0
            and r4["ci_hi"] > 0
        )
        r6_ok = (
            np.isfinite(r6.get("ci_lo", np.nan))
            and np.isfinite(r6.get("ci_hi", np.nan))
            and r6["ci_lo"] > 0
            and r6["ci_hi"] > 0
        )
        if r4_ok:
            r4_sig += 1
        if r6_ok:
            r6_sig += 1

        print(
            f"  {name:>12s}  "
            f"{r4.get('rho', np.nan):>+10.4f}  "
            f"[{r4.get('ci_lo', np.nan):>+7.4f}, {r4.get('ci_hi', np.nan):>+7.4f}]  "
            f"{r6.get('rho', np.nan):>+10.4f}  "
            f"[{r6.get('ci_lo', np.nan):>+7.4f}, {r6.get('ci_hi', np.nan):>+7.4f}]"
        )

    print()
    print(f"  R4 CI > 0: {r4_sig}/{n_assets}")
    print(f"  R6 CI > 0: {r6_sig}/{n_assets}")

    # ============================================================
    # 최종 판정
    # ============================================================
    print("\n" + "=" * 130)
    print("최종 판정")
    print("=" * 130)

    for rep in ["R3", "R4", "R6"]:
        parts = [
            results[(a, rep, "vol_5")]["partial"]
            for a in asset_grp
            if (a, rep, "vol_5") in results
            and np.isfinite(results[(a, rep, "vol_5")]["partial"])
        ]
        n_pos = sum(1 for p in parts if p > 0)
        print(
            f"  {rep}: mean partial(vol_5) = {np.mean(parts):+.4f}, "
            f"positive = {n_pos}/{len(parts)}"
        )

    print()
    print("  선택 기준:")
    print("    1. mean partial > 0.03 (모든 horizon)")
    print("    2. Block bootstrap CI > 0 비율 >= 60%")
    print("    3. R4 vs R6: 방향성 필요 여부 판정")


if __name__ == "__main__":
    raise SystemExit(main())
