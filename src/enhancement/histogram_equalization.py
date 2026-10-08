"""
Histogram equalisation - **from-scratch implementation** (NumPy only) plus the OpenCV
built-in for comparison.

Steps of the custom implementation (Gonzalez & Woods, "Digital Image Processing"):

    1. grayscale / luminance extraction        Y (from YCrCb for colour images)
    2. histogram                               h[k]   = number of pixels with value k
    3. normalised histogram (PDF)              p[k]   = h[k] / N
    4. cumulative distribution function        cdf[k] = sum_{j<=k} p[j]
    5. intensity mapping (look-up table)       T[k]   = round(255 * (cdf[k] - cdf_min) / (1 - cdf_min))
    6. output image                            Y'     = T[Y]

``cdf_min`` (the CDF at the first occupied grey level) makes the darkest occupied level
map to 0, which is the same convention as ``cv2.equalizeHist``.

Colour images are **not** equalised per RGB channel - that changes the ratios between
channels and produces strong colour shifts. Only the luminance (Y of YCrCb) is equalised
and the chroma channels (Cr, Cb) are kept, so hues are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.image_utils import ensure_rgb_uint8, is_grayscale_content, merge_luminance, split_luminance


@dataclass
class EqualizationDetails:
    """Intermediate results of the custom equalisation (for plots and teaching)."""

    histogram: np.ndarray  # h[k], shape (256,)
    pdf: np.ndarray  # p[k]
    cdf: np.ndarray  # cdf[k]
    lut: np.ndarray  # T[k], uint8
    equalized_histogram: np.ndarray


def equalize_gray_from_scratch(gray: np.ndarray) -> tuple[np.ndarray, EqualizationDetails]:
    """Histogram-equalise a single-channel uint8 image using only NumPy."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("equalize_gray_from_scratch expects a 2-D uint8 array.")

    # Step 2: histogram (np.bincount counts occurrences of each grey level 0..255).
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    # Step 3: normalised histogram = probability density function.
    pdf = histogram / total
    # Step 4: cumulative distribution function.
    cdf = np.cumsum(pdf)
    # Step 5: mapping. cdf_min = CDF of the first grey level that actually occurs.
    occupied = np.flatnonzero(histogram)
    cdf_min = cdf[occupied[0]]
    if cdf_min >= 1.0:  # single grey level: equalisation is undefined -> identity
        lut = np.arange(256, dtype=np.uint8)
    else:
        lut = np.clip(np.rint((cdf - cdf_min) / (1.0 - cdf_min) * 255.0), 0, 255).astype(np.uint8)
    # Step 6: apply the look-up table to every pixel (fancy indexing).
    equalized = lut[gray]

    details = EqualizationDetails(
        histogram=histogram,
        pdf=pdf,
        cdf=cdf,
        lut=lut,
        equalized_histogram=np.bincount(equalized.ravel(), minlength=256).astype(np.float64),
    )
    return equalized, details


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Custom (from-scratch) equalisation of the luminance of an RGB or gray image."""
    rgb = ensure_rgb_uint8(image)
    if is_grayscale_content(rgb):
        equalized, _ = equalize_gray_from_scratch(rgb[..., 0])
        return np.stack([equalized] * 3, axis=-1)
    y, ycrcb = split_luminance(rgb)
    equalized_y, _ = equalize_gray_from_scratch(y)
    return merge_luminance(equalized_y, ycrcb)


def histogram_equalization_with_details(image: np.ndarray) -> tuple[np.ndarray, EqualizationDetails]:
    """Same as :func:`histogram_equalization` but also returns the intermediate arrays."""
    rgb = ensure_rgb_uint8(image)
    y, ycrcb = split_luminance(rgb)
    equalized_y, details = equalize_gray_from_scratch(y)
    if is_grayscale_content(rgb):
        return np.stack([equalized_y] * 3, axis=-1), details
    return merge_luminance(equalized_y, ycrcb), details


def histogram_equalization_opencv(image: np.ndarray) -> np.ndarray:
    """Reference implementation with ``cv2.equalizeHist`` on the luminance (comparison only)."""
    rgb = ensure_rgb_uint8(image)
    y, ycrcb = split_luminance(rgb)
    return merge_luminance(cv2.equalizeHist(y), ycrcb)
