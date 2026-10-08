"""
Stage 4 - Enhancement operators.

``OPERATIONS`` is the registry used by the pipelines: every operation the decision
engine can select is a named, individually implemented function with explicit
keyword parameters. Nothing is hidden behind a generic ``enhance()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.enhancement.clahe import clahe
from src.enhancement.contrast_stretch import contrast_stretch
from src.enhancement.denoise import bilateral_denoise, gaussian_denoise, median_denoise
from src.enhancement.histogram_equalization import (
    histogram_equalization,
    histogram_equalization_opencv,
)
from src.enhancement.sharpening import unsharp_mask


@dataclass(frozen=True)
class OperationInfo:
    key: str
    label: str
    category: str  # "contrast" | "denoise" | "sharpen" | "super_resolution"
    function: Callable[..., np.ndarray] | None
    description: str


OPERATIONS: dict[str, OperationInfo] = {
    "contrast_stretch": OperationInfo(
        "contrast_stretch", "Contrast stretching", "contrast", contrast_stretch,
        "Linear percentile stretch of the luminance range (same gain on R, G, B).",
    ),
    "histogram_equalization": OperationInfo(
        "histogram_equalization", "Histogram equalisation (custom)", "contrast", histogram_equalization,
        "From-scratch histogram -> PDF -> CDF -> LUT equalisation of the Y channel.",
    ),
    "histogram_equalization_opencv": OperationInfo(
        "histogram_equalization_opencv", "Histogram equalisation (OpenCV)", "contrast",
        histogram_equalization_opencv, "cv2.equalizeHist on the Y channel (reference implementation).",
    ),
    "clahe": OperationInfo(
        "clahe", "CLAHE", "contrast", clahe,
        "Contrast-limited adaptive histogram equalisation of the Y channel.",
    ),
    "gaussian_denoise": OperationInfo(
        "gaussian_denoise", "Gaussian denoising", "denoise", gaussian_denoise,
        "Linear Gaussian low-pass filter.",
    ),
    "median_denoise": OperationInfo(
        "median_denoise", "Median denoising", "denoise", median_denoise,
        "Non-linear median filter (removes impulse / salt-and-pepper noise).",
    ),
    "bilateral_denoise": OperationInfo(
        "bilateral_denoise", "Bilateral denoising", "denoise", bilateral_denoise,
        "Edge-preserving bilateral filter.",
    ),
    "unsharp_mask": OperationInfo(
        "unsharp_mask", "Unsharp-mask sharpening", "sharpen", unsharp_mask,
        "Y + amount * (Y - GaussianBlur(Y)) on the luminance.",
    ),
    # Super-resolution is executed through a SuperResolver object (AI or fallback),
    # see src/enhancement/super_resolution.py, so it has no plain function here.
    "super_resolution": OperationInfo(
        "super_resolution", "Super-resolution", "super_resolution", None,
        "Real-ESRGAN x4 (optional AI stage) or Lanczos fallback.",
    ),
}


def apply_operation(key: str, image: np.ndarray, **params) -> np.ndarray:
    """Apply a registered classical operation by name."""
    info = OPERATIONS.get(key)
    if info is None or info.function is None:
        raise KeyError(f"'{key}' is not a directly callable classical operation.")
    return info.function(image, **params)


__all__ = [
    "OPERATIONS",
    "OperationInfo",
    "apply_operation",
    "bilateral_denoise",
    "clahe",
    "contrast_stretch",
    "gaussian_denoise",
    "histogram_equalization",
    "histogram_equalization_opencv",
    "median_denoise",
    "unsharp_mask",
]
