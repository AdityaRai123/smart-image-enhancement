"""
Blur / sharpness analysis with the variance of the Laplacian.

The Laplacian is a second-derivative operator: it responds to rapid intensity changes
(edges, fine texture). A sharp image has many strong responses, so the *variance* of the
Laplacian response is high; blur removes high frequencies and the variance drops.

Raw Laplacian variance has two well-known confounders, which this module removes in an
explicit, inspectable way:

1. **Contrast.** The Laplacian is linear, so scaling intensities by ``k`` scales the
   variance by ``k^2``; a low-contrast but perfectly sharp image would look "blurry".
   -> The variance is multiplied by ``gain^2`` with ``gain = 255 / (P99 - P1)``, i.e. it
   is measured as if the histogram were stretched to the full range.
2. **Noise.** Noise has strong second derivatives, so a noisy *blurry* image can look
   "sharp". -> The luminance is first smoothed with a light Gaussian (sigma ~1 px,
   i.e. a Laplacian-of-Gaussian), which suppresses pixel-level noise far more than real
   edges. When noise is detected, the remaining expected noise contribution
   ``E * sigma_n^2`` (``E`` = energy of the exact discrete kernel) is subtracted.

The classic raw value (no smoothing, no normalisation) is reported alongside for
transparency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import cv2
import numpy as np

from config import ANALYSIS, AnalysisSettings
from src.analysis.noise import gaussian_ksize
from src.utils.image_utils import rgb_to_gray


@dataclass
class SharpnessStats:
    blur_score: float  # decision value: normalised, noise-compensated Laplacian variance
    laplacian_variance_raw: float  # classic var(Laplacian(Y)) on the unmodified luminance
    laplacian_variance_smoothed: float  # var(Laplacian(Gaussian(Y))) before normalisation
    contrast_gain: float
    noise_compensation: float  # amount subtracted from the normalised variance
    edge_density: float  # fraction of Canny edge pixels (informational)

    def to_dict(self) -> dict:
        return asdict(self)


def laplacian_variance(gray: np.ndarray) -> float:
    """Classic blur metric: variance of ``cv2.Laplacian`` (3x3 kernel, ksize=1)."""
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=1).var())


def _presmooth(gray: np.ndarray, sigma: float) -> np.ndarray:
    g = gray.astype(np.float64)
    if sigma <= 0:
        return g
    k = gaussian_ksize(sigma)
    return cv2.GaussianBlur(g, (k, k), sigma, borderType=cv2.BORDER_REFLECT)


@lru_cache(maxsize=16)
def laplacian_noise_energy(presmooth_sigma: float) -> float:
    """``sum(k^2)`` of the combined Gaussian+Laplacian kernel.

    For white noise of std ``s`` the Laplacian-response variance is ``E * s^2``.
    Without pre-smoothing ``E = 20`` (kernel [[0,1,0],[1,-4,1],[0,1,0]]).
    """
    size = 4 * gaussian_ksize(max(presmooth_sigma, 0.5)) + 1
    impulse = np.zeros((size, size), dtype=np.float64)
    impulse[size // 2, size // 2] = 1.0
    response = cv2.Laplacian(_presmooth(impulse, presmooth_sigma), cv2.CV_64F, ksize=1)
    return float(np.sum(response**2))


def canny_edges(gray: np.ndarray, sigma: float = ANALYSIS.canny_sigma) -> np.ndarray:
    """Canny edge map with median-based ("auto-Canny") thresholds."""
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(smoothed))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, max(lower + 1, (1.0 + sigma) * median)))
    return cv2.Canny(smoothed, lower, upper)


def canny_edge_density(gray: np.ndarray, sigma: float = ANALYSIS.canny_sigma) -> float:
    """Fraction of pixels marked as edges by Canny (informational statistic)."""
    edges = canny_edges(gray, sigma)
    return float(np.count_nonzero(edges)) / edges.size


def analyze_sharpness(
    image: np.ndarray,
    *,
    p_low: float,
    p_high: float,
    noise_sigma: float = 0.0,
    compensate_noise: bool = False,
    settings: AnalysisSettings = ANALYSIS,
) -> SharpnessStats:
    """Compute the sharpness statistics.

    Args:
        image: RGB or grayscale uint8 image.
        p_low, p_high: luminance percentiles from the contrast analysis.
        noise_sigma: estimated Gaussian noise std (grey levels).
        compensate_noise: subtract the expected noise contribution (only when noise
            was actually detected, so clean images are not penalised by texture).
    """
    gray = rgb_to_gray(image)
    raw = laplacian_variance(gray)

    smoothed_var = laplacian_variance(_presmooth(gray, settings.blur_presmooth_sigma))
    intensity_range = max(p_high - p_low, 255.0 / settings.max_contrast_normalisation_gain)
    gain = 255.0 / intensity_range
    normalised = smoothed_var * gain * gain

    compensation = 0.0
    if compensate_noise and noise_sigma > 0:
        energy = laplacian_noise_energy(settings.blur_presmooth_sigma)
        compensation = energy * (noise_sigma * gain) ** 2
    blur_score = max(normalised - compensation, 0.0)

    return SharpnessStats(
        blur_score=float(blur_score),
        laplacian_variance_raw=raw,
        laplacian_variance_smoothed=float(smoothed_var),
        contrast_gain=float(gain),
        noise_compensation=float(compensation),
        edge_density=canny_edge_density(gray, settings.canny_sigma),
    )
