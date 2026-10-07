import numpy as np
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
