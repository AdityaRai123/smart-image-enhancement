"""
Stage 2 - Quality analysis orchestrator.

Runs the four measurements in dependency order and packages them for the decision
engine:

    noise     -> impulse ratio first: if salt-and-pepper noise is present, every other
                 statistic is measured on the impulse-suppressed image (switching median),
                 because black/white outliers would set the histogram percentiles and
                 dominate the Laplacian
    contrast  -> percentiles needed to contrast-normalise the blur score
    sharpness -> Laplacian-variance blur score (noise-compensated with the noise sigma)
    resolution-> pixel dimensions and the low-resolution flag
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config import ANALYSIS, DECISION, AnalysisSettings, DecisionThresholds
from src.analysis.blur import SharpnessStats, analyze_sharpness
from src.analysis.contrast import ContrastStats, analyze_contrast
from src.analysis.noise import NoiseStats, analyze_noise, suppress_impulses
from src.analysis.resolution import ResolutionStats, analyze_resolution
from src.utils.image_utils import ensure_rgb_uint8


@dataclass
class QualityReport:
    """All Stage-2 statistics of one image."""

    contrast: ContrastStats
    noise: NoiseStats
    sharpness: SharpnessStats
    resolution: ResolutionStats
    analysis_time_ms: float
    impulses_suppressed: bool = False  # statistics measured after a switching median

    def features(self) -> dict[str, float | int | bool]:
        """Flat feature dictionary consumed by the decision engine."""
        return {
            "histogram_spread": self.contrast.histogram_spread,
            "contrast_score": self.contrast.rms_contrast,
            "histogram_skewness": self.contrast.skewness,
            "mean_brightness": self.contrast.mean_brightness,
            "blur_score": self.sharpness.blur_score,
            "laplacian_variance_raw": self.sharpness.laplacian_variance_raw,
            "noise_score": self.noise.sigma,
            "impulse_ratio": self.noise.impulse_ratio,
            "width": self.resolution.width,
            "height": self.resolution.height,
            "pixels": self.resolution.pixels,
            "resolution_flag": self.resolution.resolution_flag,
        }

    def to_dict(self) -> dict:
        return {
            "contrast": self.contrast.to_dict(),
            "noise": self.noise.to_dict(),
            "sharpness": self.sharpness.to_dict(),
            "resolution": self.resolution.to_dict(),
            "analysis_time_ms": self.analysis_time_ms,
            "impulses_suppressed": self.impulses_suppressed,
        }


def analyze_image(
    image: np.ndarray,
    thresholds: DecisionThresholds = DECISION,
    settings: AnalysisSettings = ANALYSIS,
) -> QualityReport:
    """Compute every quality statistic of an RGB (or grayscale) uint8 image."""
    start = time.perf_counter()
    rgb = ensure_rgb_uint8(image)

    noise = analyze_noise(rgb, settings)
    impulses = noise.impulse_ratio > thresholds.impulse_ratio_threshold
    measured = suppress_impulses(rgb, settings) if impulses else rgb

    contrast = analyze_contrast(measured, settings)
    sharpness = analyze_sharpness(
        measured,
        p_low=contrast.p_low,
        p_high=contrast.p_high,
        noise_sigma=noise.sigma_luminance,
        compensate_noise=noise.sigma > thresholds.noise_medium_sigma,
        settings=settings,
    )
    resolution = analyze_resolution(rgb.shape[1], rgb.shape[0], thresholds)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return QualityReport(contrast, noise, sharpness, resolution, elapsed_ms, impulses)
