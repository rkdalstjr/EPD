#!/usr/bin/env python3
"""EPD entropy alternatives research script.

Compares four research candidates without changing the frozen EPD v3.1 baseline:
  A. Standard Permutation Entropy + daily Z_r (v3.1 baseline)
  B. Weighted Permutation Entropy (variance-weighted) + daily Z_r
  C. Dispersion Entropy (5 classes, causal rolling-window standardization) + daily Z_r
  D. Standard PE + 3-day RMS return-stress z-score (separate Z_r time-scale experiment)

Research use only: this script does not generate trading signals or optimize a strategy.
All entropy estimates are calculated from the trailing window ending before time t.
The conditional residual uses the frozen implementation shared by the user: rolling
quintile bins over prior resid_win observations, then Z_e is a causal rolling z-score.

Outputs:
  epd_close_prices.csv
  epd_entropy_candidate_metrics.csv
  epd_entropy_candidate_summary.csv
  epd_entropy_feature_diagnostics.csv
  epd_entropy_synthetic_checks.csv
"""
from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
import statsmodels.api as sm

# ============================ User settings ============================
START_DATE = "2005-01-01"
END_DATE = None  # None = most recent available close
W_PE = 60
M = 3
TAU = 1
W_Z = 252
RESID_WIN = 504
TANH_SCALE = 2.0
FORWARD_HORIZONS = (5, 10, 20)
MIN_METRIC_ROWS = 300
MIN_RESID_HISTORY = 100
MIN_BIN_COUNT = 5
EPS = 1e-9
DE_CLASSES = 5
OUTPUT_DIR = Path("epd_entropy_research_output")

# Yahoo Finance symbols. If an index/ticker is unavailable, the script skips it and reports why.
TICKERS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "Samsung": "005930.KS",
    "SK_Hynix": "000660.KS",
    "NAVER": "035420.KS",
    "LG_Chem": "051910.KS",
}

# Normal-distribution quintile cutpoints for the rolling-standardized Dispersion Entropy.
# This is a declared implementation choice, not the only possible DE definition.
DE_Z_CUTS = np.array([-0.8416212336, -0.2533471031, 0.2533471031, 0.8416212336])

CANDIDATES = {
    "A_standard_PE_daily_Zr": ("standard_pe", "daily"),
    "B_weighted_PE_daily_Zr": ("weighted_pe", "daily"),
    "C_dispersion_entropy_daily_Zr": ("dispersion_entropy", "daily"),
    "D_standard_PE_3day_RMS_Zr": ("standard_pe", "3day_rms"),
}


def ordinal_pattern(vec: np.ndarray) -> tuple[int, ...]:
    """Match the supplied implementation's stable argsort tie handling."""
    return tuple(np.argsort(vec, kind="stable").tolist())


def _normalized_entropy_from_counts(counts: Counter, max_states: int) -> float:
    total = float(sum(counts.values()))
    if total <= 0 or max_states <= 1:
        return np.nan
    probabilities = np.array(list(counts.values()), dtype=float) / total
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(max_states)


def permutation_entropy(x: np.ndarray, m: int = M, tau: int = TAU) -> float:
    """Normalized Bandt-Pompe permutation entropy in [0, 1]."""
    x = np.asarray(x, dtype=float)
    span = (m - 1) * tau + 1
    if x.size < span or not np.isfinite(x).all():
        return np.nan
    counts: Counter = Counter()
    for i in range(x.size - span + 1):
        vec = x[i : i + span : tau]
        counts[ordinal_pattern(vec)] += 1
    return _normalized_entropy_from_counts(counts, math.factorial(m))


def weighted_permutation_entropy(x: np.ndarray, m: int = M, tau: int = TAU) -> float:
    """Variance-weighted permutation entropy (WPE).

    Each embedding vector receives population variance as its weight. Pattern
    probabilities are the sums of weights for each ordinal pattern, normalized
    by total weight. The output is normalized by log(m!).
    """
    x = np.asarray(x, dtype=float)
    span = (m - 1) * tau + 1
    if x.size < span or not np.isfinite(x).all():
        return np.nan
    pattern_weights: defaultdict[tuple[int, ...], float] = defaultdict(float)
    total_weight = 0.0
    for i in range(x.size - span + 1):
        vec = x[i : i + span : tau]
        weight = float(np.var(vec, ddof=0))
        pattern_weights[ordinal_pattern(vec)] += weight
        total_weight += weight
    # Convention for a perfectly constant sequence: its only meaningful state is zero complexity.
    if total_weight <= EPS:
        return 0.0
    probs = np.array(list(pattern_weights.values()), dtype=float) / total_weight
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(math.factorial(m))


def dispersion_entropy(x: np.ndarray, m: int = M, tau: int = TAU, classes: int = DE_CLASSES) -> float:
    """Normalized Dispersion Entropy with causal-window-local standardization.

    For each trailing entropy window, values are standardized using that same
    trailing window, then mapped to `classes` symbols using fixed Gaussian-quantile
    thresholds. Patterns are symbol tuples of length m. Because the thresholds are
    applied to standardized values, the method is scale-invariant to multiplying
    the entire window by one positive constant; it captures distribution/dispersion
    patterns, not absolute volatility level.
    """
    x = np.asarray(x, dtype=float)
    span = (m - 1) * tau + 1
    if x.size < span or not np.isfinite(x).all():
        return np.nan
    sd = float(np.std(x, ddof=0))
    if sd <= EPS:
        return 0.0
    z = (x - float(np.mean(x))) / sd
    symbols = np.searchsorted(DE_Z_CUTS, z, side="right")
    counts: Counter = Counter()
    for i in range(x.size - span + 1):
        key = tuple(int(v) for v in symbols[i : i + span : tau])
        counts[key] += 1
    return _normalized_entropy_from_counts(counts, classes ** m)


def rolling_entropy_series(returns: np.ndarray, method: str, window: int = W_PE) -> np.ndarray:
    """At t, compute entropy from returns[t-window:t], excluding r[t]."""
    n = len(returns)
    out = np.full(n, np.nan)
    fn = {
        "standard_pe": permutation_entropy,
        "weighted_pe": weighted_permutation_entropy,
        "dispersion_entropy": dispersion_entropy,
    }[method]
    for t in range(window, n):
        sample = returns[t - window : t]
        if np.isfinite(sample).all():
            out[t] = fn(sample, m=M, tau=TAU)
    return out


def causal_rolling_zscore(x: np.ndarray, window: int = W_Z) -> np.ndarray:
    """Causal z-score: value at t vs prior `window` observations, ddof=1."""
    s = pd.Series(np.asarray(x, dtype=float))
    mean_prior = s.shift(1).rolling(window=window, min_periods=window).mean()
    std_prior = s.shift(1).rolling(window=window, min_periods=window).std(ddof=1)
    z = (s - mean_prior) / std_prior
    # Match the shared _core convention for a zero historical standard deviation.
    zero_sd = (std_prior == 0) & s.notna()
    z.loc[zero_sd] = 0.0
    return z.to_numpy(dtype=float)


def causal_rolling_rms(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Trailing RMS including current return; current value is known at close t."""
    s = pd.Series(np.asarray(x, dtype=float))
    return np.sqrt(s.pow(2).rolling(window=window, min_periods=window).mean()).to_numpy()


def entropy_residual(entropy: np.ndarray, z_r: np.ndarray, resid_win: int = RESID_WIN) -> np.ndarray:
    """Frozen user-provided E[entropy | |Z_r|] residual method."""
    n = len(entropy)
    abs_zr = np.abs(z_r)
    residual = np.full(n, np.nan)
    for t in range(resid_win, n):
        win_zr = abs_zr[t - resid_win : t]
        win_entropy = entropy[t - resid_win : t]
        mask = np.isfinite(win_zr) & np.isfinite(win_entropy)
        if mask.sum() < MIN_RESID_HISTORY:
            continue
        hist_zr = win_zr[mask]
        hist_entropy = win_entropy[mask]
        if not (np.isfinite(abs_zr[t]) and np.isfinite(entropy[t])):
            continue
        qs = np.quantile(hist_zr, np.linspace(0, 1, 6))
        qs[0] -= EPS
        qs[-1] += EPS
        bin_id = int(np.searchsorted(qs, abs_zr[t], side="right") - 1)
        bin_id = max(0, min(4, bin_id))
        in_bin = (hist_zr >= qs[bin_id]) & (hist_zr < qs[bin_id + 1])
        if int(in_bin.sum()) < MIN_BIN_COUNT:
            continue
        residual[t] = entropy[t] - float(hist_entropy[in_bin].mean())
    return residual


def compute_candidate(close: np.ndarray, entropy_method: str, price_mode: str) -> dict[str, np.ndarray]:
    """Run entropy -> conditional residual -> Z_e -> EPD for one candidate."""
    close = np.asarray(close, dtype=float)
    returns = np.full(close.size, np.nan)
    positive = close > 0
    log_close = np.full(close.size, np.nan)
    log_close[positive] = np.log(close[positive])
    returns[1:] = np.diff(log_close)

    entropy = rolling_entropy_series(returns, entropy_method, W_PE)
    if price_mode == "daily":
        z_r = causal_rolling_zscore(returns, W_Z)
    elif price_mode == "3day_rms":
        rms3 = causal_rolling_rms(returns, 3)
        z_r = causal_rolling_zscore(rms3, W_Z)
    else:
        raise ValueError(f"Unknown price_mode={price_mode}")

    residual = entropy_residual(entropy, z_r, RESID_WIN)
    z_e = causal_rolling_zscore(residual, W_Z)
    magnitude = np.abs(np.abs(z_r) - np.abs(z_e))
    epd100 = 100.0 * np.tanh(magnitude / TANH_SCALE)
    return {
        "returns": returns,
        "entropy": entropy,
        "z_r": z_r,
        "residual": residual,
        "z_e": z_e,
        "epd_mag": magnitude,
        "epd100": epd100,
    }


def forward_volatility(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Std of next h daily log returns: returns[t+1:t+h+1], no overlap with t."""
    n = len(returns)
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        future = returns[t + 1 : t + horizon + 1]
        if np.isfinite(future).all():
            out[t] = float(np.std(future, ddof=1))
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _partial_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Pearson partial correlation after linear residualization on one control."""
    X = np.column_stack([np.ones(len(control)), control])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return _pearson(rx, ry)


def evaluate_one(asset: str, candidate: str, data: dict[str, np.ndarray], horizon: int) -> dict:
    returns = data["returns"]
    future_vol = forward_volatility(returns, horizon)
    epd = data["epd100"]
    abs_zr = np.abs(data["z_r"])
    mask = np.isfinite(epd) & np.isfinite(abs_zr) & np.isfinite(future_vol)
    n = int(mask.sum())
    row = {
        "asset": asset,
        "candidate": candidate,
        "horizon_days": horizon,
        "n": n,
        "epd_mean": float(np.nanmean(epd)) if np.isfinite(epd).any() else np.nan,
        "epd_std": float(np.nanstd(epd)) if np.isfinite(epd).any() else np.nan,
        "share_epd_lt30": float(np.mean(epd[np.isfinite(epd)] < 30)) if np.isfinite(epd).any() else np.nan,
        "share_epd_ge70": float(np.mean(epd[np.isfinite(epd)] >= 70)) if np.isfinite(epd).any() else np.nan,
        "corr_epd_future_vol": np.nan,
        "spearman_epd_future_vol": np.nan,
        "partial_corr_control_abs_zr": np.nan,
        "beta_epd_hac": np.nan,
        "p_epd_hac": np.nan,
        "r2_base_abs_zr": np.nan,
        "r2_full_epd_plus_abs_zr": np.nan,
        "delta_r2_epd_incremental": np.nan,
    }
    if n < MIN_METRIC_ROWS:
        return row

    x = epd[mask]
    y = future_vol[mask]
    control = abs_zr[mask]
    row["corr_epd_future_vol"] = _pearson(x, y)
    row["spearman_epd_future_vol"] = float(spearmanr(x, y).statistic)
    row["partial_corr_control_abs_zr"] = _partial_corr(x, y, control)

    base_X = sm.add_constant(control)
    full_X = sm.add_constant(np.column_stack([x, control]))
    try:
        base_fit = sm.OLS(y, base_X).fit()
        full_fit = sm.OLS(y, full_X).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
        # full_X columns: constant, EPD, |Z_r|
        row["beta_epd_hac"] = float(full_fit.params[1])
        row["p_epd_hac"] = float(full_fit.pvalues[1])
        row["r2_base_abs_zr"] = float(base_fit.rsquared)
        row["r2_full_epd_plus_abs_zr"] = float(full_fit.rsquared)
        row["delta_r2_epd_incremental"] = float(full_fit.rsquared - base_fit.rsquared)
    except Exception as exc:
        warnings.warn(f"Regression failed for {asset}/{candidate}/h={horizon}: {exc}")
    return row


def download_close_prices() -> pd.DataFrame:
    """Download adjusted daily close via yfinance and preserve each ticker's history."""
    series = {}
    failures = []
    print("\n[1/4] Downloading daily adjusted close prices with yfinance...")
    for name, ticker in TICKERS.items():
        try:
            history = yf.Ticker(ticker).history(
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )
            if history is None or history.empty or "Close" not in history.columns:
                failures.append((name, ticker, "empty/no Close column"))
                print(f"  SKIP {name:10s} ({ticker}): empty data")
                continue
            idx = pd.DatetimeIndex(history.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            close = pd.Series(history["Close"].to_numpy(dtype=float), index=idx, name=name)
            close = close[~close.index.duplicated(keep="last")].sort_index().dropna()
            close = close[close > 0]
            if len(close) < 1500:
                print(f"  NOTE {name:10s} ({ticker}): only {len(close)} closes; useful but less robust")
            series[name] = close
            print(f"  OK   {name:10s} ({ticker}): {len(close):5d} closes, {close.index.min().date()} to {close.index.max().date()}")
        except Exception as exc:
            failures.append((name, ticker, repr(exc)))
            print(f"  SKIP {name:10s} ({ticker}): {exc}")
    if not series:
        raise RuntimeError("No price series downloaded. Check internet access / Yahoo Finance availability.")
    close_df = pd.concat(series.values(), axis=1).sort_index()
    close_df.columns = list(series.keys())
    close_df.to_csv(OUTPUT_DIR / "epd_close_prices.csv", index_label="date")
    if failures:
        pd.DataFrame(failures, columns=["asset", "ticker", "reason"]).to_csv(OUTPUT_DIR / "download_skips.csv", index=False)
    return close_df


def feature_diagnostics(asset: str, outputs: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Describe raw entropy series and pairwise relationships among entropy alternatives."""
    rows = []
    for candidate, data in outputs.items():
        v = data["entropy"]
        valid = np.isfinite(v)
        rows.append({
            "asset": asset,
            "candidate": candidate,
            "entropy_mean": float(np.mean(v[valid])) if valid.any() else np.nan,
            "entropy_std": float(np.std(v[valid])) if valid.any() else np.nan,
            "entropy_valid_n": int(valid.sum()),
            "epd_valid_n": int(np.isfinite(data["epd100"]).sum()),
            "epd_mean": float(np.nanmean(data["epd100"])) if np.isfinite(data["epd100"]).any() else np.nan,
            "epd_share_lt30": float(np.mean(data["epd100"][np.isfinite(data["epd100"])] < 30)) if np.isfinite(data["epd100"]).any() else np.nan,
            "epd_share_ge70": float(np.mean(data["epd100"][np.isfinite(data["epd100"])] >= 70)) if np.isfinite(data["epd100"]).any() else np.nan,
        })
    # Pairwise raw entropy correlation on aligned observations.
    keys = list(outputs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            x, y = outputs[a]["entropy"], outputs[b]["entropy"]
            mask = np.isfinite(x) & np.isfinite(y)
            rows.append({
                "asset": asset,
                "candidate": f"PAIRWISE_ENTROPY_CORR::{a}::{b}",
                "entropy_mean": np.nan,
                "entropy_std": np.nan,
                "entropy_valid_n": int(mask.sum()),
                "epd_valid_n": np.nan,
                "epd_mean": np.nan,
                "epd_share_lt30": np.nan,
                "epd_share_ge70": np.nan,
                "pairwise_entropy_corr": _pearson(x[mask], y[mask]) if mask.sum() >= 3 else np.nan,
            })
    return rows


def run_synthetic_checks() -> pd.DataFrame:
    """Sanity check rank invariance vs amplitude-sensitive alternatives."""
    rng = np.random.default_rng(20261022)
    x = rng.standard_normal(2400)
    monotonic_nonlinear = np.sign(x) * np.abs(x) ** 3  # strictly increasing transform
    scaled = 7.0 * x
    rows = []
    methods = {
        "standard_PE": permutation_entropy,
        "weighted_PE": weighted_permutation_entropy,
        "dispersion_entropy": dispersion_entropy,
    }
    for method_name, fn in methods.items():
        vals = {
            "x_original": fn(x, M, TAU),
            "x_monotonic_nonlinear": fn(monotonic_nonlinear, M, TAU),
            "x_scaled_7x": fn(scaled, M, TAU),
        }
        rows.append({
            "method": method_name,
            "original": vals["x_original"],
            "strictly_monotonic_nonlinear_transform": vals["x_monotonic_nonlinear"],
            "uniform_scale_7x": vals["x_scaled_7x"],
            "delta_nonlinear_minus_original": vals["x_monotonic_nonlinear"] - vals["x_original"],
            "delta_scaled_minus_original": vals["x_scaled_7x"] - vals["x_original"],
            "interpretation": "PE should be invariant to strict monotonic transforms; WPE/DE may respond to nonlinear amplitude distribution. All three window-local definitions should be invariant to uniform scaling (within numerical precision).",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close_df = download_close_prices()
    print("\n[2/4] Computing candidate indicators (this may take a few minutes)...")
    all_metrics: list[dict] = []
    all_diagnostics: list[dict] = []
    candidate_names = list(CANDIDATES.keys())

    for asset in close_df.columns:
        close = close_df[asset].dropna()
        if len(close) < (RESID_WIN + 2 * W_Z + W_PE + 100):
            print(f"  SKIP {asset}: too little history ({len(close)} closes) for reliable post-warmup analysis")
            continue
        close_values = close.to_numpy(dtype=float)
        asset_outputs = {}
        print(f"  Asset {asset}: {len(close_values)} closes")
        for candidate, (entropy_method, price_mode) in CANDIDATES.items():
            print(f"    computing {candidate} ...")
            asset_outputs[candidate] = compute_candidate(close_values, entropy_method, price_mode)

        all_diagnostics.extend(feature_diagnostics(asset, asset_outputs))
        for candidate in candidate_names:
            data = asset_outputs[candidate]
            for horizon in FORWARD_HORIZONS:
                all_metrics.append(evaluate_one(asset, candidate, data, horizon))

    if not all_metrics:
        raise RuntimeError("No candidate metrics generated. Check downloaded history length and parameters.")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "epd_entropy_candidate_metrics.csv", index=False)
    diagnostics_df = pd.DataFrame(all_diagnostics)
    diagnostics_df.to_csv(OUTPUT_DIR / "epd_entropy_feature_diagnostics.csv", index=False)
    synthetic_df = run_synthetic_checks()
    synthetic_df.to_csv(OUTPUT_DIR / "epd_entropy_synthetic_checks.csv", index=False)

    print("\n[3/4] Candidate summary across assets x horizons")
    valid = metrics_df[metrics_df["n"] >= MIN_METRIC_ROWS].copy()
    if valid.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = valid.groupby("candidate", as_index=False).agg(
            asset_horizon_cases=("partial_corr_control_abs_zr", "count"),
            mean_partial_corr=("partial_corr_control_abs_zr", "mean"),
            median_partial_corr=("partial_corr_control_abs_zr", "median"),
            positive_partial_corr_share=("partial_corr_control_abs_zr", lambda s: float((s > 0).mean())),
            mean_delta_r2=("delta_r2_epd_incremental", "mean"),
            positive_delta_r2_share=("delta_r2_epd_incremental", lambda s: float((s > 0).mean())),
            hac_p_lt_05_share=("p_epd_hac", lambda s: float((s < 0.05).mean())),
            median_n=("n", "median"),
        )
        summary_df.to_csv(OUTPUT_DIR / "epd_entropy_candidate_summary.csv", index=False)

    print("\n[4/4] Synthetic transform checks")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(synthetic_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        if not summary_df.empty:
            print("\nSUMMARY (descriptive, not a formal cross-asset significance test):")
            print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print("\nPer-asset/horizon results: epd_entropy_candidate_metrics.csv")
        print("Raw entropy diagnostics: epd_entropy_feature_diagnostics.csv")
        print("Prices used: epd_close_prices.csv")
        print(f"Output directory: {OUTPUT_DIR.resolve()}")

    print("\nIMPORTANT INTERPRETATION NOTES:")
    print("- Do not select a winner by one average correlation alone; inspect every asset and horizon.")
    print("- partial_corr_control_abs_zr is a linear Pearson partial correlation controlling the candidate's |Z_r|.")
    print("- HAC p-values use maxlags equal to the forward-volatility horizon; overlapping horizons still make cases dependent.")
    print("- WPE uses variance weights. DE uses a declared 5-class Gaussian-threshold implementation, standardized within each trailing PE window.")
    print("- Candidate D changes the price-stress definition to 3-day RMS; it is a separate ablation, not an EPD v3.1 replacement.")
    print("- This is indicator-information validation, not a trading strategy/backtest. No Sharpe, PnL, entries, or exits are computed.")


if __name__ == "__main__":
    main()
