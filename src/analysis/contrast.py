"""
Contrast analysis from the luminance histogram.

Measured statistics (all computed on the Y channel of YCrCb, 0-255):

* **Histogram spread** = (P_high - P_low) / 255 with P = percentiles (default 1st/99th).
  Fraction of the available dynamic range actually used by the image.
* **RMS contrast** (contrast score) = std(Y) / 255. Captures how strongly the pixel
  values deviate from the mean, i.e. whether most pixels are bunched together even if
  a few outliers span the full range.
* **Skewness** = E[((Y - mu) / sigma)^3]. Positive: mass piled up in the shadows with a
  long bright tail (under-exposure); negative: the opposite (over-exposure).
* **Mean brightness** = mean(Y) / 255 (informational).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config import ANALYSIS, AnalysisSettings
from src.utils.image_utils import rgb_to_gray


@dataclass
class ContrastStats:
    histogram_spread: float
    rms_contrast: float
    skewness: float
    mean_brightness: float
    p_low: float
    p_high: float
    occupied_bins: int

    def to_dict(self) -> dict:
        return asdict(self)


def luminance_histogram(gray: np.ndarray) -> np.ndarray:
    """256-bin histogram of an 8-bit single-channel image (counts)."""
    return np.bincount(gray.ravel(), minlength=256)[:256]


def histogram_skewness(gray: np.ndarray) -> float:
    """Fisher skewness of the intensity distribution (0 for constant images)."""
    values = gray.astype(np.float64).ravel()
    std = values.std()
    if std < 1e-9:
        return 0.0
    return float(np.mean(((values - values.mean()) / std) ** 3))


def analyze_contrast(image: np.ndarray, settings: AnalysisSettings = ANALYSIS) -> ContrastStats:
    """Compute histogram-based contrast statistics of an RGB or grayscale image."""
    gray = rgb_to_gray(image)
    p_low, p_high = np.percentile(
        gray, [settings.spread_low_percentile, settings.spread_high_percentile]
    )
    hist = luminance_histogram(gray)
    return ContrastStats(
        histogram_spread=float((p_high - p_low) / 255.0),
        rms_contrast=float(gray.std() / 255.0),
        skewness=histogram_skewness(gray),
        mean_brightness=float(gray.mean() / 255.0),
        p_low=float(p_low),
        p_high=float(p_high),
        occupied_bins=int(np.count_nonzero(hist)),
    )
