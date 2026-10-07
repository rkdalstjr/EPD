"""EPD Core — Entropy-Price Divergence v1.0.

구조적 긴장(structural tension) 지표. 방향 예측 아님.
"""

from .epd import (
    compute_epd,
    DEFAULT_W_PE,
    DEFAULT_M,
    DEFAULT_TAU,
    DEFAULT_W_Z,
    DEFAULT_TANH_SCALE,
    DEFAULT_EMA_SPAN,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from .permutation import permutation_entropy
from ._core import rolling_pe, rolling_zscore, rolling_slope

__all__ = [
    "compute_epd",
    "permutation_entropy",
    "rolling_pe",
    "rolling_zscore",
    "rolling_slope",
    "DEFAULT_W_PE",
    "DEFAULT_M",
    "DEFAULT_TAU",
    "DEFAULT_W_Z",
    "DEFAULT_TANH_SCALE",
    "DEFAULT_EMA_SPAN",
    "THRESHOLD_HIGH",
    "THRESHOLD_LOW",
]
__version__ = "1.0.0"
