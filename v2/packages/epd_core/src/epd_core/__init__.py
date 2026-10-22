"""EPD Core v3.1"""

# v3.1이 공식 기본값. "3day_rms"는 v3.2 후보 (실험적).

from .epd import (
    compute_epd,
    DEFAULT_W_PE,
    DEFAULT_M,
    DEFAULT_TAU,
    DEFAULT_W_Z,
    DEFAULT_RESID_WIN,
    DEFAULT_TANH_SCALE,
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
    "DEFAULT_RESID_WIN",
    "DEFAULT_TANH_SCALE",
    "THRESHOLD_HIGH",
    "THRESHOLD_LOW",
]
__version__ = "3.1.0"
