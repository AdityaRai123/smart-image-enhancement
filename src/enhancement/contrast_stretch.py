"""
Contrast stretching (linear, percentile-based).

    out = (in - P_low) * 255 / (P_high - P_low), clipped to [0, 255]

The percentiles are measured on the **luminance**, and the *same* linear map is applied
to R, G and B. Stretching each channel with its own percentiles would change the
channel ratios and introduce colour casts; a shared gain preserves hue.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import ENHANCEMENT
from src.utils.image_utils import ensure_rgb_uint8, rgb_to_gray


def contrast_stretch(
    image: np.ndarray,
    low_percentile: float = ENHANCEMENT.stretch_low_percentile,
    high_percentile: float = ENHANCEMENT.stretch_high_percentile,
) -> np.ndarray:
    """Linearly map the luminance percentiles ``[P_low, P_high]`` to ``[0, 255]``."""
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("Require 0 <= low_percentile < high_percentile <= 100.")
    rgb = ensure_rgb_uint8(image)
    p_low, p_high = np.percentile(rgb_to_gray(rgb), [low_percentile, high_percentile])
    if p_high - p_low < 1.0:  # constant image: nothing to stretch
        return rgb.copy()
    gain = 255.0 / (p_high - p_low)
    lut = np.clip(np.rint((np.arange(256) - p_low) * gain), 0, 255).astype(np.uint8)
    return cv2.LUT(rgb, lut)


def stretch_gain(image: np.ndarray, low_percentile: float, high_percentile: float) -> float:
    """Gain that :func:`contrast_stretch` would apply (for explanations)."""
    p_low, p_high = np.percentile(rgb_to_gray(ensure_rgb_uint8(image)), [low_percentile, high_percentile])
    return float(255.0 / max(p_high - p_low, 1.0))
