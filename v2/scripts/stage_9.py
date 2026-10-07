"""
Stage 9: Directional Decomposition of EPD

Goal
----
Determine whether the sign of entropy residual (Ze) contains
incremental market-state information beyond:

    A  = |Zr|
    E  = Ze
    R4 = ||Zr| - Ze|
    R6 = ||Zr| - |Ze||

Core questions
--------------
1. Does R6 retain information after controlling |Zr|?
2. Does the sign of Ze add information beyond R6 and |Zr|?
3. Is R4's advantage over R6 explained by the sign of Ze?
4. Is the sign effect stable across assets / horizons / |Zr| regimes?

NO trading strategy / PnL / Sharpe / MDD is used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd_v3

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

HORIZONS = [5, 10, 20]

BLOCK_SIZE = 20
BOOTSTRAP_B = 1000
RANDOM_SEED = 42

# Ze near zero is treated as neutral.
# This avoids interpreting tiny numerical noise as a direction.
SIGN_EPS = 1e-8


# ============================================================
# Asset configuration
# ============================================================

ASSETS = {
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "Samsung": "005930.KS",
    "SK Hynix": "000660.KS",
    "NAVER": "035420.KS",
    "LG Chem": "051910.KS",
}


# ============================================================
# Utility
# ============================================================


def rank_array(x):
    """
    Tie-aware rank transform.
    """
    return pd.Series(x).rank(method="average").to_numpy(dtype=float)


def safe_spearman(x, y):
    """
    Tie-aware Spearman correlation.
    """
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 20:
        return np.nan

    r = spearmanr(x[mask], y[mask])

    return float(r.statistic)


def partial_spearman(x, y, control):
    """
    Partial Spearman correlation using rank residualization.

    Correlation between rank(x) and rank(y), controlling for rank(control).
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)

    if mask.sum() < 30:
        return np.nan

    xr = rank_array(x[mask])
    yr = rank_array(y[mask])
    cr = rank_array(control[mask])

    # Residualize X against control
    X = np.column_stack(
        [
            np.ones(len(cr)),
            cr,
        ]
    )

    beta_x = np.linalg.lstsq(X, xr, rcond=None)[0]
    beta_y = np.linalg.lstsq(X, yr, rcond=None)[0]

    rx = xr - X @ beta_x
    ry = yr - X @ beta_y

    sx = np.std(rx)
    sy = np.std(ry)

    if sx == 0 or sy == 0:
        return np.nan

    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman_two_controls(x, y, c1, c2):
    """
    Partial Spearman correlation controlling for two variables.

    Used for testing whether Ze sign adds information
    beyond |Zr| and R6.
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c1) & np.isfinite(c2)

    if mask.sum() < 50:
        return np.nan

    xr = rank_array(x[mask])
    yr = rank_array(y[mask])
    c1r = rank_array(c1[mask])
    c2r = rank_array(c2[mask])

    X = np.column_stack(
        [
            np.ones(len(c1r)),
            c1r,
            c2r,
        ]
    )

    beta_x = np.linalg.lstsq(X, xr, rcond=None)[0]
    beta_y = np.linalg.lstsq(X, yr, rcond=None)[0]

    rx = xr - X @ beta_x
    ry = yr - X @ beta_y

    sx = np.std(rx)
    sy = np.std(ry)

    if sx == 0 or sy == 0:
        return np.nan

    return float(np.corrcoef(rx, ry)[0, 1])


# ============================================================
# Block bootstrap
# ============================================================


def make_block_indices(n, block_size, rng):
    """
    Circular block bootstrap.
    """
    if n <= 0:
        return np.array([], dtype=int)

    starts = rng.integers(
        0,
        n,
        size=int(np.ceil(n / block_size)),
    )

    indices = []

    for s in starts:
        block = (s + np.arange(block_size)) % n
        indices.extend(block.tolist())

    return np.asarray(indices[:n], dtype=int)


def bootstrap_stat(
    arrays,
    statistic_fn,
    block_size=20,
    B=1000,
    seed=42,
):
    """
    Generic circular block bootstrap.
    """
    arrays = [np.asarray(a, dtype=float) for a in arrays]

    n = len(arrays[0])

    if n < block_size * 2:
        return np.nan, np.nan, np.nan

    if not all(len(a) == n for a in arrays):
        raise ValueError("All arrays must have the same length.")

    rng = np.random.default_rng(seed)

    observed = statistic_fn(*arrays)

    boots = []

    for _ in range(B):
        idx = make_block_indices(
            n,
            block_size,
            rng,
        )

        sample = [a[idx] for a in arrays]

        try:
            value = statistic_fn(*sample)
        except Exception:
            value = np.nan

        if np.isfinite(value):
            boots.append(value)

    if len(boots) < max(100, B // 5):
        return observed, np.nan, np.nan

    boots = np.asarray(boots)

    lo = np.percentile(boots, 2.5)
    hi = np.percentile(boots, 97.5)

    return observed, lo, hi


# ============================================================
# EPD decomposition
# ============================================================


def build_stage9_features(zr, ze):
    """
    Construct the representations required for Stage 9.

    A
        = |Zr|

    E
        = Ze

    E_abs
        = |Ze|

    R4
        = ||Zr| - Ze|

    R6
        = ||Zr| - |Ze||

    D
        = |Zr| - Ze

    sign_E
        = sign(Ze)

    sign_E_negative
        = 1 if Ze < 0
    """

    A = np.abs(zr)
    E = ze
    E_abs = np.abs(E)

    R4 = np.abs(A - E)
    R6 = np.abs(A - E_abs)

    D = A - E

    sign_E = np.zeros_like(E)

    sign_E[E > SIGN_EPS] = 1.0
    sign_E[E < -SIGN_EPS] = -1.0

    negative_E = (E < -SIGN_EPS).astype(float)
    positive_E = (E > SIGN_EPS).astype(float)

    return {
        "A": A,
        "E": E,
        "E_abs": E_abs,
        "R4": R4,
        "R6": R6,
        "D": D,
        "sign_E": sign_E,
        "negative_E": negative_E,
        "positive_E": positive_E,
    }


# ============================================================
# Future observable targets
# ============================================================


def make_future_targets(close):
    """
    Future volatility / movement observables.

    These are validation targets only.
    """

    close = pd.Series(close).astype(float)

    log_ret = np.log(close).diff()

    targets = {}

    for h in HORIZONS:
        targets[f"vol_{h}"] = log_ret.rolling(h).std().shift(-h + 1)

    return pd.DataFrame(targets, index=close.index)


# ============================================================
# Sign-specific analysis
# ============================================================


def sign_group_difference(feature, target, negative_mask):
    """
    Difference in target rank between Ze-negative and Ze-positive states.

    Positive value:
        negative-Ze state has larger future target.

    Uses rank target so scale differences across assets
    are less problematic.
    """

    mask = np.isfinite(feature) & np.isfinite(target) & np.isfinite(negative_mask)

    if mask.sum() < 50:
        return np.nan

    y = target[mask]
    g = negative_mask[mask].astype(bool)

    if g.sum() < 20 or (~g).sum() < 20:
        return np.nan

    y_rank = rank_array(y)

    return float(np.mean(y_rank[g]) - np.mean(y_rank[~g]))


def sign_partial_effect(
    sign_feature,
    target,
    A,
    R6,
):
    """
    Does Ze sign carry information beyond |Zr| and R6?

    sign_feature:
        1 for Ze < 0
        0 for Ze >= 0

    Because sign is binary, Spearman here is equivalent
    to a rank-based point-biserial-type association.

    Controls:
        |Zr|
        R6
    """

    return partial_spearman_two_controls(
        sign_feature,
        target,
        A,
        R6,
    )


# ============================================================
# One asset / one horizon
# ============================================================


def evaluate_one(
    df,
    zr_col="zr",
    ze_col="ze",
):
    """
    Evaluate all Stage 9 representations.
    """

    zr = df[zr_col].to_numpy(dtype=float)
    ze = df[ze_col].to_numpy(dtype=float)

    features = build_stage9_features(zr, ze)

    targets = make_future_targets(df["close"])

    rows = []

    for horizon in HORIZONS:

        target = targets[f"vol_{horizon}"].to_numpy(dtype=float)

        # ----------------------------------------------------
        # R4 / R6 baseline
        # ----------------------------------------------------

        r4_rho = safe_spearman(
            features["R4"],
            target,
        )

        r6_rho = safe_spearman(
            features["R6"],
            target,
        )

        r4_partial = partial_spearman(
            features["R4"],
            target,
            features["A"],
        )

        r6_partial = partial_spearman(
            features["R6"],
            target,
            features["A"],
        )

        # ----------------------------------------------------
        # Sign effect
        # ----------------------------------------------------

        sign_partial = sign_partial_effect(
            features["negative_E"],
            target,
            features["A"],
            features["R6"],
        )

        sign_diff = sign_group_difference(
            features["R6"],
            target,
            features["negative_E"],
        )

        # ----------------------------------------------------
        # Conditional R6 representations
        # ----------------------------------------------------

        R6_negative = features["R6"] * features["negative_E"]

        R6_positive = features["R6"] * features["positive_E"]

        neg_rho = safe_spearman(
            R6_negative,
            target,
        )

        pos_rho = safe_spearman(
            R6_positive,
            target,
        )

        neg_partial = partial_spearman(
            R6_negative,
            target,
            features["A"],
        )

        pos_partial = partial_spearman(
            R6_positive,
            target,
            features["A"],
        )

        # ----------------------------------------------------
        # Signed divergence D
        # ----------------------------------------------------

        D_rho = safe_spearman(
            features["D"],
            target,
        )

        D_partial = partial_spearman(
            features["D"],
            target,
            features["A"],
        )

        rows.append(
            {
                "horizon": horizon,
                "R4_rho": r4_rho,
                "R6_rho": r6_rho,
                "R4_partial": r4_partial,
                "R6_partial": r6_partial,
                "sign_partial": sign_partial,
                "sign_rank_diff": sign_diff,
                "R6_negative_rho": neg_rho,
                "R6_positive_rho": pos_rho,
                "R6_negative_partial": neg_partial,
                "R6_positive_partial": pos_partial,
                "D_rho": D_rho,
                "D_partial": D_partial,
            }
        )

    return pd.DataFrame(rows), features, targets


# ============================================================
# Bootstrap sign effect
# ============================================================


def bootstrap_sign_effect(
    features,
    target,
    B=1000,
    block_size=20,
    seed=42,
):
    """
    Bootstrap CI for incremental sign information.

    Statistic:
        partial Spearman(
            sign(Ze),
            future_vol,
            controls=[|Zr|, R6]
        )
    """

    A = features["A"]
    R6 = features["R6"]
    sign = features["negative_E"]

    mask = np.isfinite(A) & np.isfinite(R6) & np.isfinite(sign) & np.isfinite(target)

    A = A[mask]
    R6 = R6[mask]
    sign = sign[mask]
    target = target[mask]

    if len(target) < 100:
        return np.nan, np.nan, np.nan

    def stat(s, y, a, r6):
        return partial_spearman_two_controls(
            s,
            y,
            a,
            r6,
        )

    return bootstrap_stat(
        [
            sign,
            target,
            A,
            R6,
        ],
        stat,
        block_size=block_size,
        B=B,
        seed=seed,
    )


# ============================================================
# Conditional |Zr| quintile analysis
# ============================================================


def quintile_analysis(features, target):
    """
    Examine sign information within |Zr| quintiles.

    For each quintile:
        compare Ze-negative vs Ze-positive future volatility.

    Returns rank difference.
    """

    A = features["A"]
    negative = features["negative_E"]

    mask = np.isfinite(A) & np.isfinite(target) & np.isfinite(negative)

    A = A[mask]
    target = target[mask]
    negative = negative[mask]

    if len(A) < 100:
        return pd.DataFrame()

    q = pd.qcut(
        A,
        5,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for qi in sorted(np.unique(q)):

        m = q == qi

        if m.sum() < 50:
            continue

        diff = sign_group_difference(
            A[m],
            target[m],
            negative[m],
        )

        n_neg = int(np.sum(negative[m] == 1))
        n_pos = int(np.sum(negative[m] == 0))

        rows.append(
            {
                "quintile": int(qi + 1),
                "n": int(m.sum()),
                "n_Ze_negative": n_neg,
                "n_Ze_positive": n_pos,
                "rank_diff_negative_minus_positive": diff,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main Stage 9
# ============================================================


def run_stage9(
    load_asset_fn,
):
    """
    Parameters
    ----------
    load_asset_fn:
        Function:
            asset_name -> DataFrame

        Required columns:
            date
            close
            zr
            ze
    """

    all_results = []
    bootstrap_results = []
    quintile_results = []
    redundancy_results = []

    for asset_name in ASSETS:

        print("=" * 80)
        print(asset_name)
        print("=" * 80)

        df = load_asset_fn(asset_name).copy()

        if "date" in df.columns:
            df = df.sort_values("date")

        df = df.reset_index(drop=True)

        # ----------------------------------------------------
        # Stage 9 representations
        # ----------------------------------------------------

        results, features, targets = evaluate_one(df)

        results.insert(
            0,
            "asset",
            asset_name,
        )

        all_results.append(results)

        # ----------------------------------------------------
        # R6 redundancy with |Zr|
        # ----------------------------------------------------

        redundancy = safe_spearman(
            features["R6"],
            features["A"],
        )

        redundancy_results.append(
            {
                "asset": asset_name,
                "R6_vs_absZr_spearman": redundancy,
            }
        )

        # ----------------------------------------------------
        # Bootstrap sign effect
        # ----------------------------------------------------

        for horizon in HORIZONS:

            target = targets[f"vol_{horizon}"].to_numpy(dtype=float)

            observed, lo, hi = bootstrap_sign_effect(
                features,
                target,
                B=BOOTSTRAP_B,
                block_size=BLOCK_SIZE,
                seed=RANDOM_SEED + horizon,
            )

            bootstrap_results.append(
                {
                    "asset": asset_name,
                    "horizon": horizon,
                    "sign_partial": observed,
                    "CI_low": lo,
                    "CI_high": hi,
                    "CI_above_zero": (bool(lo > 0) if np.isfinite(lo) else False),
                }
            )

        # ----------------------------------------------------
        # Quintile analysis
        # ----------------------------------------------------

        for horizon in HORIZONS:

            target = targets[f"vol_{horizon}"].to_numpy(dtype=float)

            qdf = quintile_analysis(
                features,
                target,
            )

            if not qdf.empty:

                qdf.insert(
                    0,
                    "asset",
                    asset_name,
                )

                qdf.insert(
                    1,
                    "horizon",
                    horizon,
                )

                quintile_results.append(qdf)

    # ========================================================
    # Combine
    # ========================================================

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    bootstrap_df = pd.DataFrame(bootstrap_results)

    quintile_df = (
        pd.concat(
            quintile_results,
            ignore_index=True,
        )
        if quintile_results
        else pd.DataFrame()
    )

    redundancy_df = pd.DataFrame(redundancy_results)

    return (
        results_df,
        bootstrap_df,
        quintile_df,
        redundancy_df,
    )


# ============================================================
# Summary
# ============================================================


def summarize_stage9(
    results_df,
    bootstrap_df,
    quintile_df,
    redundancy_df,
):
    print("\n")
    print("#" * 80)
    print("STAGE 9 SUMMARY")
    print("#" * 80)

    # --------------------------------------------------------
    # 1. Mean representation performance
    # --------------------------------------------------------

    print("\n[1] Representation comparison")

    summary = results_df.groupby("horizon").agg(
        {
            "R4_rho": "mean",
            "R6_rho": "mean",
            "R4_partial": "mean",
            "R6_partial": "mean",
            "sign_partial": "mean",
            "sign_rank_diff": "mean",
            "D_partial": "mean",
        }
    )

    print(summary.to_string(float_format=lambda x: f"{x:+.4f}"))

    # --------------------------------------------------------
    # 2. Sign consistency
    # --------------------------------------------------------

    print("\n[2] Sign partial consistency")

    sign_consistency = results_df.groupby("horizon")["sign_partial"].agg(
        [
            ("mean", "mean"),
            ("positive_ratio", lambda x: np.mean(x > 0)),
            ("negative_ratio", lambda x: np.mean(x < 0)),
        ]
    )

    print(sign_consistency.to_string(float_format=lambda x: f"{x:.4f}"))

    # --------------------------------------------------------
    # 3. Bootstrap CI
    # --------------------------------------------------------

    print("\n[3] Bootstrap CI for Ze sign incremental effect")

    boot_summary = bootstrap_df.groupby("horizon").agg(
        sign_partial_mean=(
            "sign_partial",
            "mean",
        ),
        CI_above_zero_ratio=(
            "CI_above_zero",
            "mean",
        ),
    )

    print(boot_summary.to_string(float_format=lambda x: f"{x:+.4f}"))

    # --------------------------------------------------------
    # 4. R4 vs R6
    # --------------------------------------------------------

    print("\n[4] R4 vs R6")

    comparison = results_df.groupby("horizon").agg(
        {
            "R4_partial": "mean",
            "R6_partial": "mean",
        }
    )

    comparison["R6_minus_R4"] = comparison["R6_partial"] - comparison["R4_partial"]

    print(comparison.to_string(float_format=lambda x: f"{x:+.4f}"))

    # --------------------------------------------------------
    # 5. R6 redundancy
    # --------------------------------------------------------

    print("\n[5] R6 redundancy with |Zr|")

    print(
        redundancy_df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    print(
        "\nMean |corr(R6, |Zr|)| = "
        f"{redundancy_df['R6_vs_absZr_spearman'].abs().mean():.4f}"
    )

    # --------------------------------------------------------
    # 6. Quintile analysis
    # --------------------------------------------------------

    if not quintile_df.empty:

        print("\n[6] Ze sign effect by |Zr| quintile")

        qsummary = quintile_df.groupby(["horizon", "quintile"]).agg(
            {
                "rank_diff_negative_minus_positive": "mean",
                "n": "mean",
            }
        )

        print(qsummary.to_string(float_format=lambda x: f"{x:+.4f}"))

    # --------------------------------------------------------
    # 7. Diagnostic judgment
    # --------------------------------------------------------

    mean_sign = results_df["sign_partial"].mean()

    positive_ratio = np.mean(results_df["sign_partial"] > 0)

    ci_ratio = np.mean(bootstrap_df["CI_above_zero"])

    delta = results_df["R6_partial"].mean() - results_df["R4_partial"].mean()

    print("\n")
    print("#" * 80)
    print("STAGE 9 DIAGNOSTIC")
    print("#" * 80)

    print(f"Mean sign partial      : {mean_sign:+.4f}")

    print(f"Positive sign ratio    : {positive_ratio:.1%}")

    print(f"Bootstrap CI > 0 ratio : {ci_ratio:.1%}")

    print(f"Mean R6 - R4 partial   : {delta:+.4f}")

    if positive_ratio < 0.60 and ci_ratio < 0.50:
        print("\n=> Ze의 부호가 추가 정보를 제공한다는 " "근거가 약합니다.")

        print("=> R6의 magnitude representation을 " "우선 유지하는 것이 타당합니다.")

    elif positive_ratio >= 0.80 and ci_ratio >= 0.50:
        print(
            "\n=> Ze의 부호가 R6와 |Zr|을 통제한 뒤에도 "
            "추가 정보를 가질 가능성이 있습니다."
        )

        print(
            "=> 단일 magnitude 지표보다 "
            "magnitude + directional component를 "
            "검토할 가치가 있습니다."
        )

    else:
        print("\n=> Ze 부호 효과가 혼재합니다.")

        print(
            "=> R4를 즉시 폐기하거나 R6를 최종 확정하지 말고 "
            "시계열 구간별 안정성을 추가 검증해야 합니다."
        )


# ============================================================
# Example integration
# ============================================================

# ============================================================
#  Loader
# ============================================================
ASSETS_MAP = {
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "Samsung": "005930.KS",
    "SK Hynix": "000660.KS",
    "NAVER": "035420.KS",
    "LG Chem": "051910.KS",
}


def load_asset(asset_name):
    """yfinance → DataFrame(date, close, zr, ze)."""
    import yfinance as yf

    tkr = ASSETS_MAP[asset_name]
    df = yf.download(
        tkr,
        start="2005-01-01",
        end="2025-12-31",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if df is None or len(df) < 1000:
        raise ValueError(f"{asset_name} insufficient data")

    close = df["Close"].squeeze().to_numpy(dtype=float)
    valid = np.isfinite(close)
    close = close[valid]
    dates = df.index[valid]

    out = compute_epd_v3(close)

    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "zr": out["z_r"],
            "ze": out["z_e"],
        }
    )


if __name__ == "__main__":
    results_df, bootstrap_df, quintile_df, redundancy_df = run_stage9(
        load_asset_fn=load_asset
    )

    summarize_stage9(
        results_df,
        bootstrap_df,
        quintile_df,
        redundancy_df,
    )

    outdir = ROOT / "docs" / "stage9_outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outdir / "results.csv", index=False)
    bootstrap_df.to_csv(outdir / "bootstrap.csv", index=False)
    redundancy_df.to_csv(outdir / "redundancy.csv", index=False)
    if not quintile_df.empty:
        quintile_df.to_csv(outdir / "quintiles.csv", index=False)
    print(f"\nCSV → {outdir}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           