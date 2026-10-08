"""
Noise estimation based on local standard deviation in homogeneous regions.

Idea (explainable in one sentence): *in a flat region of the scene the only thing that
makes neighbouring pixels differ is noise, so the local standard deviation there is an
estimate of the noise level.*

Algorithm (Gaussian-type noise)
    1. **Structure removal.** Each channel is filtered with the Immerkaer operator
       ``N = [[1,-2,1],[-2,4,-2],[1,-2,1]] / 6`` (J. Immerkaer, "Fast Noise Variance
       Estimation", CVIU 1996). ``N`` is the difference of two Laplacians, so it cancels
       constant, linear and quadratic intensity structure; for white noise of std
       ``sigma`` the output has std exactly ``sigma`` (``sqrt(sum(N^2)) = 1``).
    2. **Local standard deviation** of that residual in a ``w x w`` window, computed with
       two box filters: ``sqrt(E[r^2] - E[r]^2)``.
    3. **Homogeneous windows** are selected with the **Sobel** gradient magnitude of the
       smoothed luminance: windows whose mean gradient is in the lowest ``p`` percent
       and that contain no clipped (near 0 / 255) pixels.
    4. ``sigma = median(local std over homogeneous windows)``, measured on R, G and B
       separately and averaged (noise is usually independent per channel, and the
       luminance average would hide ~1/3 of it).

Impulse (salt-and-pepper) noise is detected separately: the fraction of pixels that are
near-black/near-white *and* differ strongly from their 3x3 median.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from config import ANALYSIS, AnalysisSettings
from src.utils.image_utils import ensure_rgb_uint8, is_grayscale_content, rgb_to_gray

# Immerkaer's noise operator, normalised so that white noise keeps its std.
IMMERKAER_KERNEL: np.ndarray = (
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64) / 6.0
)


@dataclass
class NoiseStats:
    sigma: float  # mean per-channel noise std (8-bit grey levels) - decision value
    sigma_luminance: float  # noise std in the luminance channel (used by the blur score)
    impulse_ratio: float  # fraction of salt-and-pepper candidate pixels
    flat_fraction: float  # fraction of pixels used as "homogeneous"
    reliable: bool  # enough homogeneous pixels for a trustworthy estimate

    def to_dict(self) -> dict:
        return asdict(self)


def gaussian_ksize(sigma: float) -> int:
    """Odd kernel size covering +-3 sigma (used consistently across the project)."""
    return 2 * int(np.ceil(3.0 * sigma)) + 1


def local_std(image: np.ndarray, window: int) -> np.ndarray:
    """Local standard deviation in a ``window x window`` neighbourhood (box filters)."""
    img = image.astype(np.float64)
    mean = cv2.blur(img, (window, window), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(img * img, (window, window), borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def noise_residual(channel: np.ndarray) -> np.ndarray:
    """Structure-cancelled residual (std == noise std for white noise)."""
    return cv2.filter2D(channel.astype(np.float64), -1, IMMERKAER_KERNEL, borderType=cv2.BORDER_REFLECT)


def homogeneous_mask(gray: np.ndarray, settings: AnalysisSettings = ANALYSIS) -> np.ndarray:
    """Boolean mask of windows that contain neither edges/texture nor clipped pixels."""
    window = settings.noise_window
    sigma = settings.noise_gradient_sigma
    k = gaussian_ksize(sigma)
    smooth = cv2.GaussianBlur(gray.astype(np.float64), (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    grad_window = cv2.blur(np.hypot(gx, gy), (window, window), borderType=cv2.BORDER_REFLECT)

    margin = settings.noise_saturation_margin
    clipped = ((gray <= margin) | (gray >= 255 - margin)).astype(np.float64)
    clipped_window = cv2.blur(clipped, (window, window), borderType=cv2.BORDER_REFLECT) > 0

    candidates = ~clipped_window
    if np.count_nonzero(candidates) < settings.noise_min_flat_pixels:
        candidates = np.ones_like(candidates, dtype=bool)
    threshold = np.percentile(grad_window[candidates], settings.noise_flat_gradient_percentile)
    return candidates & (grad_window <= threshold)


def _sigma_in_mask(channel: np.ndarray, mask: np.ndarray, window: int) -> float:
    std_map = local_std(noise_residual(channel), window)
    values = std_map[mask] if np.any(mask) else std_map.ravel()
    return float(np.median(values)) if values.size else 0.0


def noise_maps(image: np.ndarray, settings: AnalysisSettings = ANALYSIS) -> dict[str, np.ndarray]:
    """Intermediate maps of the estimator (displayed in the UI's diagnostic view)."""
    gray = rgb_to_gray(ensure_rgb_uint8(image))
    return {
        "residual": noise_residual(gray),
        "local_std": local_std(noise_residual(gray), settings.noise_window),
        "flat_mask": homogeneous_mask(gray, settings),
    }


def estimate_gaussian_noise(
    image: np.ndarray, settings: AnalysisSettings = ANALYSIS
) -> tuple[float, float, float, bool]:
    """Return ``(sigma_per_channel, sigma_luminance, flat_fraction, reliable)``."""
    rgb = ensure_rgb_uint8(image)
    gray = rgb_to_gray(rgb)
    mask = homogeneous_mask(gray, settings)
    n_flat = int(np.count_nonzero(mask))
    window = settings.noise_window

    sigma_lum = _sigma_in_mask(gray, mask, window)
    if is_grayscale_content(rgb):
        sigma = sigma_lum
    else:
        sigma = float(np.mean([_sigma_in_mask(rgb[..., c], mask, window) for c in range(3)]))
    return sigma, sigma_lum, n_flat / float(mask.size), n_flat >= settings.noise_min_flat_pixels


def impulse_mask(gray: np.ndarray, settings: AnalysisSettings = ANALYSIS) -> np.ndarray:
    """Boolean mask of salt-and-pepper candidates: near-black/near-white pixels that
    differ from their 3x3 median by at least ``impulse_min_diff`` grey levels."""
    median = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int16) - median.astype(np.int16))
    extreme = (gray <= settings.impulse_dark_level) | (gray >= settings.impulse_bright_level)
    return extreme & (diff >= settings.impulse_min_diff)


def impulse_noise_ratio(gray: np.ndarray, settings: AnalysisSettings = ANALYSIS) -> float:
    """Fraction of pixels that look like salt-and-pepper impulses."""
    return float(np.mean(impulse_mask(gray, settings)))


def suppress_impulses(image: np.ndarray, settings: AnalysisSettings = ANALYSIS) -> np.ndarray:
    """Switching median: replace ONLY the detected impulse pixels by their 3x3 median.

    Used for *measurement* when impulse noise is present: unlike a full median filter it
    leaves every other pixel untouched, so the blur and Gaussian-noise statistics of the
    underlying image are not altered by the pre-filter itself.
    """
    rgb = ensure_rgb_uint8(image)
    mask = impulse_mask(rgb_to_gray(rgb), settings)
    if not mask.any():
        return rgb
    out = rgb.copy()
    out[mask] = cv2.medianBlur(rgb, 3)[mask]
    return out


def analyze_noise(image: np.ndarray, settings: AnalysisSettings = ANALYSIS) -> NoiseStats:
    """Estimate Gaussian noise sigma and the impulse-noise ratio of an image."""
    rgb = ensure_rgb_uint8(image)
    impulse_ratio = impulse_noise_ratio(rgb_to_gray(rgb), settings)
    # Isolated impulses would dominate the local std; when present, measure the
    # Gaussian component after replacing them (switching median).
    source = suppress_impulses(rgb, settings) if impulse_ratio > settings.impulse_prefilter_ratio else rgb
    sigma, sigma_lum, flat_fraction, reliable = estimate_gaussian_noise(source, settings)
    return NoiseStats(
        sigma=sigma,
        sigma_luminance=sigma_lum,
        impulse_ratio=impulse_ratio,
        flat_fraction=flat_fraction,
        reliable=reliable,
    )
