"""
Stage 8: Indicator Information Validation

목적
----
EPD를 투자전략으로 백테스트하지 않는다.

본 단계의 질문은 오직 하나다.

    "R6가 |Zr|만으로 설명되는 가격 충격의 크기와 별개로,
     추가적인 시장 상태 정보를 가지고 있는가?"

검증 대상
----------
R1 = |Zr|
R4 = ||Zr| - Ze|
R6 = ||Zr| - |Ze||

핵심 검증
----------
1. Marginal Spearman
2. Partial Spearman | |Zr|
3. Block Bootstrap CI
4. |Zr| 크기를 통제한 조건부 분석
5. |Zr| quintile 내부에서 R6의 일관성
6. R4 vs R6 비교
7. 자산별 / horizon별 안정성
8. R6와 |Zr|의 중복성 점검

주의
----
- 매매전략 없음
- PnL 없음
- position 없음
- threshold 최적화 없음
- transaction cost 없음
- Sharpe/MDD 없음

이 단계에서 미래 변동성은 "수익 목표"가 아니라
현재 시장 상태 표현이 미래 시장 불안정성과 어떤 관계를 가지는지
검증하기 위한 관측 대상(target)으로만 사용한다.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")


# ============================================================
# Project import
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import compute_epd_v3

# ============================================================
# Configuration
# ============================================================

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

START = "2005-01-01"
END = "2025-12-31"

HORIZONS = [5, 10, 20]

BLOCK = 20
BOOTSTRAP_B = 1000
SEED = 42

MIN_OBS = 300
N_BINS = 5

EPS = 1e-6


# ============================================================
# Forward volatility
# ============================================================


def fwd_vol(r: np.ndarray, h: int) -> np.ndarray:
    """
    Future h-day realized volatility.

    IMPORTANT:
    This is not used as a trading target.
    It is only an observable future market-state variable
    for validating whether the current indicator contains
    information about subsequent market instability.
    """
    out = np.full_like(r, np.nan, dtype=float)

    for t in range(len(r) - h):
        window = r[t + 1 : t + 1 + h]
        window = window[np.isfinite(window)]

        if window.size >= 3:
            out[t] = np.std(window, ddof=1)

    return out


TARGETS = {f"vol_{h}": lambda r, h=h: fwd_vol(r, h) for h in HORIZONS}


# ============================================================
# Representation
# ============================================================


def build_reps(z_r: np.ndarray, z_e: np.ndarray) -> dict[str, np.ndarray]:
    """
    Representation definitions.

    R1:
        |Zr|

    R4:
        ||Zr| - Ze|

    R6:
        ||Zr| - |Ze||

    R6 is the primary candidate.
    """

    return {
        "R1": np.abs(z_r),
        "R4": np.abs(np.abs(z_r) - z_e),
        "R6": np.abs(np.abs(z_r) - np.abs(z_e)),
    }


# ============================================================
# Spearman
# ============================================================


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Tie-aware Spearman correlation.
    """

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < MIN_OBS:
        return np.nan

    value = spearmanr(x[mask], y[mask]).statistic

    if not np.isfinite(value):
        return np.nan

    return float(value)


# ============================================================
# Partial Spearman
# ============================================================


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """
    Partial Spearman correlation.

    모든 변수를 rank-transform한 뒤,
    rank(z)를 통제하고 residual correlation을 계산한다.

    따라서:
        partial_spearman(R6, future_vol, |Zr|)

    은

        "|Zr|의 영향력을 제거한 뒤에도
         R6와 future volatility 사이에 관계가 남는가?"

    를 측정한다.
    """

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if mask.sum() < MIN_OBS:
        return np.nan

    x = x[mask]
    y = y[mask]
    z = z[mask]

    # Rank transform
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rz = rankdata(z, method="average")

    Z = np.column_stack(
        [
            np.ones(len(rz)),
            rz,
        ]
    )

    beta_x = np.linalg.lstsq(Z, rx, rcond=None)[0]
    beta_y = np.linalg.lstsq(Z, ry, rcond=None)[0]

    res_x = rx - Z @ beta_x
    res_y = ry - Z @ beta_y

    sx = np.std(res_x)
    sy = np.std(res_y)

    if sx == 0 or sy == 0:
        return np.nan

    return float(np.corrcoef(res_x, res_y)[0, 1])


# ============================================================
# Block bootstrap
# ============================================================


def block_indices(
    n: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:

    n_blocks = int(np.ceil(n / block))

    starts = rng.integers(
        0,
        n - block + 1,
        size=n_blocks,
    )

    idx = np.concatenate([np.arange(start, start + block) for start in starts])

    return idx[:n]


def bootstrap_statistic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    statistic,
    block: int = BLOCK,
    B: int = BOOTSTRAP_B,
    seed: int = SEED,
) -> tuple[float, float]:

    if z is None:
        mask = np.isfinite(x) & np.isfinite(y)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if mask.sum() < MIN_OBS:
        return np.nan, np.nan

    x = x[mask]
    y = y[mask]

    if z is not None:
        z = z[mask]

    n = len(x)

    rng = np.random.default_rng(seed)

    values = []

    for _ in range(B):

        idx = block_indices(
            n=n,
            block=block,
            rng=rng,
        )

        xb = x[idx]
        yb = y[idx]

        if z is None:
            value = statistic(xb, yb)
        else:
            zb = z[idx]
            value = statistic(xb, yb, zb)

        if np.isfinite(value):
            values.append(value)

    if len(values) < 100:
        return np.nan, np.nan

    values = np.asarray(values)

    lo = np.percentile(values, 2.5)
    hi = np.percentile(values, 97.5)

    return float(lo), float(hi)


# ============================================================
# |Zr| quintile analysis
# ============================================================


def conditional_bin_analysis(
    rep: np.ndarray,
    target: np.ndarray,
    control: np.ndarray,
    n_bins: int = N_BINS,
) -> list[dict]:

    mask = np.isfinite(rep) & np.isfinite(target) & np.isfinite(control)

    if mask.sum() < MIN_OBS:
        return []

    rep = rep[mask]
    target = target[mask]
    control = control[mask]

    # Quantile bins of |Zr|
    try:
        bins = pd.qcut(
            control,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return []

    results = []

    for b in sorted(np.unique(bins)):

        m = bins == b

        if m.sum() < 100:
            continue

        rho = spearman(
            rep[m],
            target[m],
        )

        results.append(
            {
                "bin": int(b + 1),
                "n": int(m.sum()),
                "control_mean": float(np.mean(control[m])),
                "rho": rho,
            }
        )

    return results


# ============================================================
# Asset preparation
# ============================================================


def load_asset(
    ticker: str,
    name: str,
) -> dict | None:

    try:

        df = yf.download(
            ticker,
            start=START,
            end=END,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if df is None or len(df) < 1000:
            return None

        close = df["Close"].squeeze().to_numpy(dtype=float)

        close = close[np.isfinite(close)]

        if close.size < 1000:
            return None

        # Log return
        r = np.full_like(
            close,
            np.nan,
            dtype=float,
        )

        r[1:] = np.diff(np.log(close))

        # EPD core
        out = compute_epd_v3(close)

        z_r = np.asarray(
            out["z_r"],
            dtype=float,
        )

        z_e = np.asarray(
            out["z_e"],
            dtype=float,
        )

        reps = build_reps(
            z_r,
            z_e,
        )

        return {
            "name": name,
            "close": close,
            "return": r,
            "z_r": z_r,
            "z_e": z_e,
            "abs_zr": np.abs(z_r),
            "reps": reps,
        }

    except Exception as exc:

        print(f"[WARN] {name}: {exc}")

        return None


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 120)
    print("Stage 8 — Indicator Information Validation")
    print("=" * 120)

    print()
    print("목적:")
    print("  R6가 |Zr|만으로 설명되지 않는 추가적인 시장 상태 정보를")
    print("  가지고 있는지 검증한다.")
    print()
    print("백테스트:")
    print("  사용하지 않음")
    print("  PnL / position / Sharpe / MDD / entry / exit 없음")
    print()

    assets = {}

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    for ticker, name, group in TICKERS:

        print(
            f"Loading {name:<12s} ...",
            end=" ",
        )

        data = load_asset(
            ticker,
            name,
        )

        if data is None:
            print("FAILED")
            continue

        data["group"] = group

        assets[name] = data

        print(f"OK ({len(data['close'])} observations)")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    conditional_results = []

    for name, data in assets.items():

        r = data["return"]
        abs_zr = data["abs_zr"]

        for target_name, target_fn in TARGETS.items():

            target = target_fn(r)

            for rep_name in ["R1", "R4", "R6"]:

                rep = data["reps"][rep_name]

                # --------------------------------------------
                # Marginal
                # --------------------------------------------

                rho = spearman(
                    rep,
                    target,
                )

                # --------------------------------------------
                # Partial |Zr|
                # --------------------------------------------

                partial = partial_spearman(
                    rep,
                    target,
                    abs_zr,
                )

                # --------------------------------------------
                # Bootstrap marginal
                # --------------------------------------------

                rho_lo, rho_hi = bootstrap_statistic(
                    rep,
                    target,
                    None,
                    spearman,
                    block=BLOCK,
                    B=BOOTSTRAP_B,
                    seed=SEED,
                )

                # --------------------------------------------
                # Bootstrap partial
                # --------------------------------------------

                partial_lo, partial_hi = bootstrap_statistic(
                    rep,
                    target,
                    abs_zr,
                    partial_spearman,
                    block=BLOCK,
                    B=BOOTSTRAP_B,
                    seed=SEED,
                )

                results.append(
                    {
                        "asset": name,
                        "group": data["group"],
                        "target": target_name,
                        "rep": rep_name,
                        "rho": rho,
                        "rho_lo": rho_lo,
                        "rho_hi": rho_hi,
                        "partial": partial,
                        "partial_lo": partial_lo,
                        "partial_hi": partial_hi,
                    }
                )

                # --------------------------------------------
                # Conditional |Zr| quintiles
                # --------------------------------------------

                if rep_name in ["R4", "R6"]:

                    bins = conditional_bin_analysis(
                        rep,
                        target,
                        abs_zr,
                    )

                    for item in bins:

                        conditional_results.append(
                            {
                                "asset": name,
                                "group": data["group"],
                                "target": target_name,
                                "rep": rep_name,
                                **item,
                            }
                        )

    result_df = pd.DataFrame(results)
    cond_df = pd.DataFrame(conditional_results)

    # ========================================================
    # 1. Main summary
    # ========================================================

    print()
    print("=" * 120)
    print("1. 전체 자산 — Marginal / Partial")
    print("=" * 120)

    summary = (
        result_df.groupby(["rep", "target"])[["rho", "partial"]].mean().reset_index()
    )

    for rep in ["R1", "R4", "R6"]:

        print()
        print(f"[{rep}]")

        sub = summary[summary["rep"] == rep]

        for _, row in sub.iterrows():

            print(
                f"  {row['target']:<8s} "
                f"rho={row['rho']:+.4f}   "
                f"partial={row['partial']:+.4f}"
            )

    # ========================================================
    # 2. Partial bootstrap significance
    # ========================================================

    print()
    print("=" * 120)
    print("2. Partial Spearman Block Bootstrap 95% CI")
    print("=" * 120)

    print()
    print(
        f"{'asset':<12s} "
        f"{'target':<8s} "
        f"{'rep':<5s} "
        f"{'partial':>10s} "
        f"{'95% CI':>24s}"
    )

    print("-" * 75)

    for _, row in result_df.iterrows():

        if row["rep"] not in ["R4", "R6"]:
            continue

        print(
            f"{row['asset']:<12s} "
            f"{row['target']:<8s} "
            f"{row['rep']:<5s} "
            f"{row['partial']:>+10.4f} "
            f"[{row['partial_lo']:>+.4f}, "
            f"{row['partial_hi']:>+.4f}]"
        )

    # ========================================================
    # 3. Sign consistency
    # ========================================================

    print()
    print("=" * 120)
    print("3. Partial Sign Consistency")
    print("=" * 120)

    for rep in ["R4", "R6"]:

        print()
        print(f"[{rep}]")

        for target in TARGETS:

            sub = result_df[(result_df["rep"] == rep) & (result_df["target"] == target)]

            positive = (sub["partial"] > 0).sum()

            ci_positive = (sub["partial_lo"] > 0).sum()

            print(
                f"  {target:<8s} "
                f"partial > 0: "
                f"{positive}/{len(sub)}   "
                f"CI > 0: "
                f"{ci_positive}/{len(sub)}"
            )

    # ========================================================
    # 4. |Zr| quintile conditional analysis
    # ========================================================

    print()
    print("=" * 120)
    print("4. Conditional Analysis — |Zr| Quintiles")
    print("=" * 120)

    print()
    print("질문:")
    print("  |Zr|가 작은 상태 / 중간 상태 / 큰 상태에서도")
    print("  R6와 미래 변동성의 관계가 유지되는가?")

    for rep in ["R4", "R6"]:

        print()
        print(f"[{rep}]")

        for target in TARGETS:

            sub = cond_df[(cond_df["rep"] == rep) & (cond_df["target"] == target)]

            if sub.empty:
                continue

            mean_by_bin = sub.groupby("bin")["rho"].mean()

            positive_bins = (mean_by_bin > 0).sum()

            print(
                f"  {target:<8s} "
                f"positive bins = "
                f"{positive_bins}/{len(mean_by_bin)}"
            )

            for b, value in mean_by_bin.items():

                print(f"      Q{int(b)}: " f"{value:+.4f}")

    # ========================================================
    # 5. R6 redundancy with |Zr|
    # ========================================================

    print()
    print("=" * 120)
    print("5. R6 Redundancy Check")
    print("=" * 120)

    print()
    print("R6와 |Zr|의 상관이 지나치게 높으면")
    print("R6가 단순한 |Zr| 변형일 가능성이 커진다.")

    redundancy_rows = []

    for name, data in assets.items():

        abs_zr = data["abs_zr"]
        r6 = data["reps"]["R6"]

        rho = spearman(
            r6,
            abs_zr,
        )

        redundancy_rows.append(
            {
                "asset": name,
                "rho_R6_absZr": rho,
            }
        )

    redundancy_df = pd.DataFrame(redundancy_rows)

    print()

    for _, row in redundancy_df.iterrows():

        print(
            f"  {row['asset']:<12s} "
            f"Spearman(R6, |Zr|) = "
            f"{row['rho_R6_absZr']:+.4f}"
        )

    print()
    print(
        f"  Mean absolute correlation: "
        f"{redundancy_df['rho_R6_absZr'].abs().mean():.4f}"
    )

    # ========================================================
    # 6. R4 vs R6
    # ========================================================

    print()
    print("=" * 120)
    print("6. R4 vs R6 — Information Comparison")
    print("=" * 120)

    comparison = (
        result_df[result_df["rep"].isin(["R4", "R6"])]
        .groupby(["rep", "target"])[["rho", "partial"]]
        .mean()
        .reset_index()
    )

    for target in TARGETS:

        r4 = comparison[
            (comparison["rep"] == "R4") & (comparison["target"] == target)
        ].iloc[0]

        r6 = comparison[
            (comparison["rep"] == "R6") & (comparison["target"] == target)
        ].iloc[0]

        print()
        print(f"[{target}]")

        print(f"  R4  rho={r4['rho']:+.4f} " f"partial={r4['partial']:+.4f}")

        print(f"  R6  rho={r6['rho']:+.4f} " f"partial={r6['partial']:+.4f}")

        print(f"  Δ partial " f"(R6 - R4) = " f"{r6['partial'] - r4['partial']:+.4f}")

    # ========================================================
    # 7. Stability score
    # ========================================================

    print()
    print("=" * 120)
    print("7. Representation Stability")
    print("=" * 120)

    for rep in ["R4", "R6"]:

        sub = result_df[result_df["rep"] == rep]

        positive_partial = (sub["partial"] > 0).mean()

        ci_above_zero = (sub["partial_lo"] > 0).mean()

        horizon_mean = sub.groupby("target")["partial"].mean()

        horizon_std = horizon_mean.std()

        print()
        print(f"[{rep}]")

        print(f"  Positive partial ratio : " f"{positive_partial:.1%}")

        print(f"  CI > 0 ratio           : " f"{ci_above_zero:.1%}")

        print(f"  Horizon mean std       : " f"{horizon_std:.4f}")

        print("  Horizon partial:")

        for target, value in horizon_mean.items():

            print(f"      {target:<8s}: " f"{value:+.4f}")

    # ========================================================
    # 8. Final interpretation
    # ========================================================

    print()
    print("=" * 120)
    print("8. Stage 8 Diagnostic")
    print("=" * 120)

    r6 = result_df[result_df["rep"] == "R6"]

    r6_positive = (r6["partial"] > 0).mean()

    r6_ci_positive = (r6["partial_lo"] > 0).mean()

    r6_mean_partial = r6["partial"].mean()

    print()

    print(f"R6 mean partial        : " f"{r6_mean_partial:+.4f}")

    print(f"R6 positive ratio      : " f"{r6_positive:.1%}")

    print(f"R6 bootstrap CI > 0    : " f"{r6_ci_positive:.1%}")

    print()

    if r6_mean_partial > 0.03 and r6_positive >= 0.75 and r6_ci_positive >= 0.50:
        print(
            "판정: R6는 |Zr|과 구별되는 "
            "추가적인 시장 상태 정보 후보로서 강한 증거를 보인다."
        )

    elif r6_mean_partial > 0.03:

        print(
            "판정: R6는 유망한 추가 정보 후보이나 "
            "자산 간 보편성은 아직 충분히 확립되지 않았다."
        )

    else:

        print("판정: R6의 추가 정보성에 대한 증거가 부족하다.")

    print()
    print("중요: 본 판정은 투자전략의 수익성이나 매매 성과를 " "평가하는 것이 아니다.")

    print(
        "목적은 EPD가 시장 상태를 표현하는 독립적인 축으로서 "
        "통계적으로 의미가 있는지를 검증하는 것이다."
    )


if __name__ == "__main__":
    raise SystemExit(main())
