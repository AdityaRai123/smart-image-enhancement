"""
Denoising operators: Gaussian, median and bilateral filtering.

* **Gaussian** - linear low-pass; optimal for averaging out mild white noise, but blurs
  edges in proportion to its sigma.
* **Median** - non-linear rank filter; replaces each pixel by the median of its
  neighbourhood, which removes isolated outliers (salt-and-pepper impulses) completely
  while keeping step edges.
* **Bilateral** - edge-preserving; weights neighbours by spatial distance *and* by
  intensity difference (``sigma_color``), so pixels across an edge barely contribute.

All functions take and return RGB uint8 images.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.analysis.noise import gaussian_ksize
from src.utils.image_utils import ensure_rgb_uint8


def gaussian_denoise(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Gaussian smoothing with standard deviation ``sigma`` (pixels)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    rgb = ensure_rgb_uint8(image)
    k = gaussian_ksize(sigma)
    return cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def median_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter with an odd ``ksize x ksize`` window."""
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer >= 3.")
    return cv2.medianBlur(ensure_rgb_uint8(image), int(ksize))


def bilateral_denoise(
    image: np.ndarray, diameter: int = 7, sigma_color: float = 35.0, sigma_space: float = 3.0
) -> np.ndarray:
    """Edge-preserving bilateral filter."""
    if diameter < 1 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("diameter, sigma_color and sigma_space must be positive.")
    return cv2.bilateralFilter(ensure_rgb_uint8(image), int(diameter), float(sigma_color), float(sigma_space))
