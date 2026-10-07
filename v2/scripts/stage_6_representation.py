"""Stage 6 — Representation 비교.

5가지 magnitude 후보:
  R1: |Z_r|
  R2: |Z_e_res|
  R3: |Z_r − Z_e_res|         (C1)
  R4: ||Z_r| − Z_e_res|       (새 후보 — 이론적으로 자연스러움)
  R5: |Z_r − Z_e_res| / (|Z_r| + |Z_e_res| + ε)   (상대적 divergence)

Targets (6):
  fwd_vol_5, fwd_vol_10, fwd_vol_20
  fwd_abs_ret_5, fwd_range_5, fwd_max_abs_5

검증:
  - marginal Spearman
  - partial | |Z_r|
  - 4분면 (Z_r 부호 × Z_e_res 부호) × fwd_vol_5
  - block bootstrap 95% CI (block=20, B=500)
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
    ("005930.KS", "삼성전자", "KR"),
    ("000660.KS", "SK하이닉스", "KR"),
    ("035420.KS", "NAVER", "KR"),
    ("051910.KS", "LG화학", "KR"),
]

EPS = 1e-6
BLOCK = 20
B = 500


# ============================================================
#  Targets
# ============================================================
def fwd_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def fwd_abs_ret(r, h=5):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fv[t] = np.nansum(np.abs(r[t + 1 : t + 1 + h]))
    return fv


def fwd_range(r, h=5):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.max() - w.min()
    return fv


def fwd_max_abs(r, h=5):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = np.abs(r[t + 1 : t + 1 + h])
        w = w[np.isfinite(w)]
        if w.size >= 1:
            fv[t] = w.max()
    return fv


TARGETS = {
    "vol_5": lambda r: fwd_vol(r, 5),
    "vol_10": lambda r: fwd_vol(r, 10),
    "vol_20": lambda r: fwd_vol(r, 20),
    "abs_ret_5": lambda r: fwd_abs_ret(r, 5),
    "range_5": lambda r: fwd_range(r, 5),
    "max_abs_5": lambda r: fwd_max_abs(r, 5),
}


# ============================================================
#  통계
# ============================================================
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


def block_bootstrap_ci(x, y, block=BLOCK, B=B, seed=42):
    """Moving block bootstrap → rho의 95% CI."""
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


# ============================================================
#  Representation 정의
# ============================================================
def build_reps(z_r, z_e):
    return {
        "R1_absZr": np.abs(z_r),
        "R2_absZe": np.abs(z_e),
        "R3_absDiff": np.abs(z_r - z_e),
        "R4_absZrMinusZe": np.abs(np.abs(z_r) - z_e),
        "R5_ratio": np.abs(z_r - z_e) / (np.abs(z_r) + np.abs(z_e) + EPS),
    }


# ============================================================
#  Main
# ============================================================
def main():
    import yfinance as yf

    print("=" * 140)
    print("Stage 6 — Representation 비교 (5 후보 × 6 targets)")
    print("=" * 140)

    # 수집용
    all_reps = list(build_reps(np.array([0.0]), np.array([0.0])).keys())

    # 결과: (asset, rep, target) -> {rho, partial, ci_lo, ci_hi}
    results = {}
    asset_data = {}

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
        except Exception as e:
            print(f"{name} err: {e}")
            continue

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))

        out = compute_epd_v3(close)
        z_r, z_e = out["z_r"], out["z_e"]
        abs_zr = np.abs(z_r)

        reps = build_reps(z_r, z_e)

        print(f"\n### {name} ({grp})")
        print(
            f"  {'rep':>16s}  {'target':>11s}  "
            f"{'rho':>9s}  {'partial':>9s}  {'95% CI':>18s}"
        )
        print("  " + "-" * 75)

        asset_data[name] = {"grp": grp, "reps": reps, "z_r": z_r, "z_e": z_e, "r": r}

        for rname, rval in reps.items():
            for tname, fn in TARGETS.items():
                tgt = fn(r)
                rho = spearman(rval, tgt)
                part = partial_corr(rval, tgt, abs_zr)
                lo, hi = block_bootstrap_ci(rval, tgt, block=BLOCK, B=B)

                results[(name, rname, tname)] = {
                    "rho": rho,
                    "partial": part,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }

                print(
                    f"  {rname:>16s}  {tname:>11s}  "
                    f"{rho:>+9.4f}  {part:>+9.4f}  "
                    f"[{lo:>+7.4f}, {hi:>+7.4f}]"
                )
            print()

    # ============================================================
    # 요약: rep × target 평균
    # ============================================================
    print("=" * 140)
    print("자산 평균 |rho| (8자산)")
    print("=" * 140)
    print(
        f"  {'rep':>16s}  "
        + "  ".join(f"{t:>10s}" for t in TARGETS.keys())
        + f"  {'mean':>9s}"
    )
    print("  " + "-" * 130)

    for rname in all_reps:
        vals = []
        for tname in TARGETS.keys():
            rhos = [
                abs(results[(a, rname, tname)]["rho"])
                for a in asset_data
                if (a, rname, tname) in results
                and np.isfinite(results[(a, rname, tname)]["rho"])
            ]
            vals.append(np.mean(rhos) if rhos else np.nan)
        print(
            f"  {rname:>16s}  "
            + "  ".join(f"{v:>10.4f}" for v in vals)
            + f"  {np.nanmean(vals):>9.4f}"
        )

    # ============================================================
    # Partial correlation (|Z_r| 통제 후)
    # ============================================================
    print()
    print("=" * 140)
    print("Partial correlation (|Z_r| 통제 후)")
    print("=" * 140)
    print(
        f"  {'rep':>16s}  "
        + "  ".join(f"{t:>10s}" for t in TARGETS.keys())
        + f"  {'mean':>9s}"
    )
    print("  " + "-" * 130)

    for rname in all_reps:
        vals = []
        for tname in TARGETS.keys():
            parts = [
                results[(a, rname, tname)]["partial"]
                for a in asset_data
                if (a, rname, tname) in results
                and np.isfinite(results[(a, rname, tname)]["partial"])
            ]
            vals.append(np.mean(parts) if parts else np.nan)
        print(
            f"  {rname:>16s}  "
            + "  ".join(f"{v:>+10.4f}" for v in vals)
            + f"  {np.nanmean(vals):>+9.4f}"
        )

    # ============================================================
    # 4분면 × fwd_vol_5
    # ============================================================
    print()
    print("=" * 140)
    print("4분면 분석 × fwd_vol_5 (자산 평균)")
    print("=" * 140)
    print(
        f"  {'asset':>12s}  "
        f"{'Q1(Zr+,Ze+)':>13s}  {'Q2(Zr-,Ze+)':>13s}  "
        f"{'Q3(Zr-,Ze-)':>13s}  {'Q4(Zr+,Ze-)':>13s}  {'spread':>9s}"
    )
    print("  " + "-" * 90)

    spreads = []
    for name, d in asset_data.items():
        z_r, z_e, r = d["z_r"], d["z_e"], d["r"]
        fv = fwd_vol(r, 5)
        m = np.isfinite(z_r) & np.isfinite(z_e) & np.isfinite(fv)

        q1 = m & (z_r > 0) & (z_e > 0)
        q2 = m & (z_r < 0) & (z_e > 0)
        q3 = m & (z_r < 0) & (z_e < 0)
        q4 = m & (z_r > 0) & (z_e < 0)

        vols = []
        for mask in [q1, q2, q3, q4]:
            vols.append(fv[mask].mean() if mask.sum() > 20 else np.nan)

        spread = max(v for v in vols if np.isfinite(v)) - min(
            v for v in vols if np.isfinite(v)
        )
        spreads.append(spread)

        print(
            f"  {name:>12s}  "
            + "  ".join(f"{v:>13.6f}" for v in vols)
            + f"  {spread:>9.6f}"
        )

    print()
    print(
        f"  4분면 vol spread: mean = {np.mean(spreads):.6f}, "
        f"min = {np.min(spreads):.6f}, max = {np.max(spreads):.6f}"
    )

    # ============================================================
    # 최종 판정
    # ============================================================
    print()
    print("=" * 140)
    print("판정")
    print("=" * 140)

    # R1 vs R3 vs R4 (vol_5)
    for rname in ["R1_absZr", "R3_absDiff", "R4_absZrMinusZe"]:
        rhos = [
            abs(results[(a, rname, "vol_5")]["rho"])
            for a in asset_data
            if (a, rname, "vol_5") in results
        ]
        parts = [
            results[(a, rname, "vol_5")]["partial"]
            for a in asset_data
            if (a, rname, "vol_5") in results
            and np.isfinite(results[(a, rname, "vol_5")]["partial"])
        ]
        print(
            f"  {rname:>16s}: mean|rho|={np.mean(rhos):.4f}, "
            f"mean partial={np.mean(parts):+.4f}"
        )

    print()
    print("  R4 (||Z_r|−Z_e|)가 R1 (|Z_r|)을 능가하면 → EPD_v3.1 확정")
    print("  R1이 여전히 최강이면 → EPD 폐기, |Z_r| 단독")


if __name__ == "__main__":
    raise SystemExit(main())
