"""
Stage 5 - Full-reference quality metrics (scikit-image).

* **PSNR** (peak signal-to-noise ratio, dB) = 10 log10(MAX^2 / MSE). Pixel-wise fidelity:
  higher means the test image is numerically closer to the reference. Identical images
  give +infinity.
* **SSIM** (structural similarity, Wang et al. 2004) compares local luminance, contrast
  and structure in Gaussian windows (sigma 1.5); 1.0 means structurally identical.

Which reference?
    The report evaluates the enhanced image **against the original upload** (there is no
    clean ground truth for a real photo). That measures *how much the enhancement changed
    the image*, not whether it improved it: "no enhancement" trivially scores PSNR = inf
    and SSIM = 1. Higher PSNR/SSIM against a degraded input therefore does NOT mean a
    better enhancement.

    When a clean **ground-truth reference** is available (synthetic degradations, or a
    user-supplied reference), the same functions measure true restoration quality. The
    ``reference_kind`` field records which case applies so results are never mixed up.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from skimage.metrics import structural_similarity

from config import EVALUATION, EvaluationSettings
from src.utils.image_utils import ensure_rgb_uint8, resize_to_match

logger = logging.getLogger(__name__)

REFERENCE_ORIGINAL = "original"
REFERENCE_GROUND_TRUTH = "ground_truth"


@dataclass
class MetricResult:
    psnr: float  # dB; math.inf when identical; nan on failure
    ssim: float  # [-1, 1]; nan on failure
    reference_kind: str  # "original" or "ground_truth"
    resized_for_comparison: bool = False
    note: str | None = None

    @property
    def psnr_text(self) -> str:
        return format_psnr(self.psnr)

    @property
    def ssim_text(self) -> str:
        return "n/a" if math.isnan(self.ssim) else f"{self.ssim:.4f}"

    def to_dict(self) -> dict:
        return asdict(self)


def format_psnr(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf (identical)"
    return f"{value:.2f} dB"


def cap_psnr(value: float, settings: EvaluationSettings = EVALUATION) -> float:
    """Cap infinite PSNR for averaging (documented in every generated report)."""
    if math.isnan(value):
        return value
    return min(value, settings.psnr_cap_db)


def compute_psnr(reference: np.ndarray, test: np.ndarray, data_range: float = EVALUATION.data_range) -> float:
    """PSNR in dB between two same-shaped uint8 images (``inf`` if identical)."""
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {test.shape}")
    mse = float(np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / mse)


def compute_ssim(
    reference: np.ndarray,
    test: np.ndarray,
    data_range: float = EVALUATION.data_range,
    settings: EvaluationSettings = EVALUATION,
) -> float:
    """Mean SSIM over colour channels (Wang et al. 2004 Gaussian-window variant)."""
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {test.shape}")
    min_side = min(reference.shape[:2])
    kwargs: dict = {"data_range": data_range}
    if reference.ndim == 3:
        kwargs["channel_axis"] = -1
    gaussian_window = 2 * int(3.5 * settings.ssim_sigma + 0.5) + 1  # skimage's window size
    if settings.ssim_gaussian_weights and min_side >= gaussian_window:
        kwargs.update(gaussian_weights=True, sigma=settings.ssim_sigma, use_sample_covariance=False)
    else:
        win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win < 3:
            raise ValueError(f"Image too small for SSIM ({reference.shape[1]}x{reference.shape[0]}).")
        kwargs["win_size"] = win
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(structural_similarity(reference, test, **kwargs))


def compare_images(
    reference: np.ndarray,
    test: np.ndarray,
    reference_kind: str = REFERENCE_ORIGINAL,
    settings: EvaluationSettings = EVALUATION,
) -> MetricResult:
    """PSNR + SSIM of ``test`` against ``reference``. Never raises.

    If the sizes differ (e.g. after super-resolution) the test image is resampled to the
    reference size first, and ``resized_for_comparison`` is set.
    """
    try:
        ref = ensure_rgb_uint8(reference)
        tst = ensure_rgb_uint8(test)
        resized = ref.shape[:2] != tst.shape[:2]
        if resized:
            tst = resize_to_match(tst, ref.shape[:2])
        note = None
        if resized:
            note = (
                f"Enhanced image ({test.shape[1]}x{test.shape[0]}) was resampled to the reference size "
                f"({ref.shape[1]}x{ref.shape[0]}) for comparison."
            )
        return MetricResult(
            psnr=compute_psnr(ref, tst, settings.data_range),
            ssim=compute_ssim(ref, tst, settings.data_range, settings),
            reference_kind=reference_kind,
            resized_for_comparison=resized,
            note=note,
        )
    except Exception as exc:  # metric failure must never crash the app
        logger.warning("Metric computation failed: %s", exc)
        return MetricResult(math.nan, math.nan, reference_kind, False, f"Metric computation failed: {exc}")
