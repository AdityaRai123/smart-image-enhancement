"""
Controlled synthetic degradations.

Used by ``scripts/create_test_dataset.py`` and by the "Built-in sample" demo mode of the
app. Because the clean image is known, PSNR/SSIM can be computed against a true
reference instead of against the degraded input (see ``src/evaluation/metrics.py``).

Every function is deterministic for a given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class DegradationSpec:
    """Parameters of one degradation recipe (recorded in the dataset manifest)."""

    name: str
    description: str
    contrast_factor: float | None = None
    blur_sigma: float | None = None
    noise_sigma: float | None = None
    salt_pepper_amount: float | None = None
    downscale_factor: int | None = None
    # Which defects the decision engine is expected to detect (ground truth labels).
    expected: tuple[str, ...] = field(default_factory=tuple)


# The categories requested for the evaluation set. "good" is the untouched clean image:
# it checks that the adaptive pipeline does NOT process images that need nothing.
DEGRADATIONS: dict[str, DegradationSpec] = {
    "good": DegradationSpec(
        name="good",
        description="Clean reference image, no degradation (checks for unnecessary processing)",
        expected=(),
    ),
    "low_contrast": DegradationSpec(
        name="low_contrast",
        description="Linear contrast compression around the mean (factor 0.45)",
        contrast_factor=0.45,
        expected=("contrast",),
    ),
    "noisy": DegradationSpec(
        name="noisy",
        description="Additive white Gaussian noise (sigma = 15 grey levels)",
        noise_sigma=15.0,
        expected=("noise",),
    ),
    "impulse_noise": DegradationSpec(
        name="impulse_noise",
        description="Salt-and-pepper noise (3% of pixels)",
        salt_pepper_amount=0.03,
        expected=("noise",),
    ),
    "blurry": DegradationSpec(
        name="blurry",
        description="Gaussian blur (sigma = 2.0 px)",
        blur_sigma=2.0,
        expected=("blur",),
    ),
    "low_resolution": DegradationSpec(
        name="low_resolution",
        description="4x downsampling (INTER_AREA)",
        downscale_factor=4,
        expected=("resolution",),
    ),
    "mixed": DegradationSpec(
        name="mixed",
        description="Blur (sigma 1.5) + contrast compression (0.55) + Gaussian noise (sigma 10)",
        blur_sigma=1.5,
        contrast_factor=0.55,
        noise_sigma=10.0,
        expected=("contrast", "noise", "blur"),
    ),
}


def reduce_contrast(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Compress contrast linearly around the mean luminance: ``I' = m + f (I - m)``."""
    img = rgb.astype(np.float32)
    mean = float(img.mean())
    out = mean + factor * (img - mean)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def add_gaussian_noise(rgb: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean white Gaussian noise with standard deviation ``sigma`` (grey levels)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=rgb.shape).astype(np.float32)
    return np.clip(np.rint(rgb.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def add_salt_and_pepper(rgb: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Set a fraction ``amount`` of pixels to pure black or white (all channels)."""
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    mask = rng.random(rgb.shape[:2])
    out[mask < amount / 2] = 0
    out[(mask >= amount / 2) & (mask < amount)] = 255
    return out


def gaussian_blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Optical-style defocus approximation with a Gaussian point-spread function."""
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def downsample(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Reduce resolution by an integer factor with area averaging (anti-aliased)."""
    h, w = rgb.shape[:2]
    new_size = (max(1, w // factor), max(1, h // factor))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def crop_to_multiple(rgb: np.ndarray, multiple: int) -> np.ndarray:
    """Crop so both sides are divisible by ``multiple`` (keeps x4 SR output size exact)."""
    h, w = rgb.shape[:2]
    return rgb[: h - h % multiple, : w - w % multiple].copy()


def apply_degradation(rgb: np.ndarray, spec: DegradationSpec, seed: int = 0) -> np.ndarray:
    """Apply a recipe in a camera-like order: blur -> contrast -> noise -> resampling."""
    out = rgb.copy()
    if spec.blur_sigma:
        out = gaussian_blur(out, spec.blur_sigma)
    if spec.contrast_factor is not None:
        out = reduce_contrast(out, spec.contrast_factor)
    if spec.noise_sigma:
        out = add_gaussian_noise(out, spec.noise_sigma, seed=seed)
    if spec.salt_pepper_amount:
        out = add_salt_and_pepper(out, spec.salt_pepper_amount, seed=seed)
    if spec.downscale_factor:
        out = downsample(out, spec.downscale_factor)
    return out
