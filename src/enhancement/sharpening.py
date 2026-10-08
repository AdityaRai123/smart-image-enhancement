"""
Unsharp-mask sharpening.

    blurred   = GaussianBlur(Y, sigma)
    mask      = Y - blurred                  (the high-frequency detail)
    sharpened = Y + amount * mask            where |mask| > threshold

* ``sigma``     - size of the details that are boosted (larger -> coarser edges)
* ``amount``    - strength (1.0 doubles the high-frequency detail)
* ``threshold`` - minimum local difference (grey levels) that is sharpened; small
  differences are mostly residual noise and are left untouched

Applied to the luminance only (Y of YCrCb): sharpening chroma creates coloured halos and
amplifies chroma noise.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.analysis.noise import gaussian_ksize
from src.utils.image_utils import ensure_rgb_uint8, is_grayscale_content, merge_luminance, split_luminance


def unsharp_mask_channel(
    channel: np.ndarray, sigma: float, amount: float, threshold: float = 0.0
) -> np.ndarray:
    """Unsharp masking of one uint8 channel; returns uint8."""
    src = channel.astype(np.float32)
    k = gaussian_ksize(sigma)
    blurred = cv2.GaussianBlur(src, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    mask = src - blurred
    if threshold > 0:
        mask = np.where(np.abs(mask) > threshold, mask, 0.0).astype(np.float32)
    return np.clip(np.rint(src + amount * mask), 0, 255).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray, sigma: float = 1.0, amount: float = 0.8, threshold: float = 0.0
) -> np.ndarray:
    """Sharpen the luminance of an RGB image with an unsharp mask."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if amount < 0:
        raise ValueError("amount must be non-negative.")
    rgb = ensure_rgb_uint8(image)
    if is_grayscale_content(rgb):
        sharpened = unsharp_mask_channel(rgb[..., 0], sigma, amount, threshold)
        return np.stack([sharpened] * 3, axis=-1)
    y, ycrcb = split_luminance(rgb)
    return merge_luminance(unsharp_mask_channel(y, sigma, amount, threshold), ycrcb)
