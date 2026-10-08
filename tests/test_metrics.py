"""Stage 5: PSNR / SSIM and the baseline comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.evaluation import REFERENCE_GROUND_TRUTH, REFERENCE_ORIGINAL, cap_psnr, compare_images, compute_psnr, compute_ssim
from src.evaluation.comparison import compare_pipelines
from src.utils.degradation import add_gaussian_noise, downsample, gaussian_blur


def test_psnr_identical_is_infinite(scene):
    assert math.isinf(compute_psnr(scene, scene))
    assert cap_psnr(compute_psnr(scene, scene)) == 60.0


def test_psnr_matches_known_value():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((10, 10, 3), 10, np.uint8)  # MSE = 100
    assert compute_psnr(a, b) == pytest.approx(10 * math.log10(255**2 / 100))


def test_psnr_matches_scikit_image(natural):
    noisy = add_gaussian_noise(natural, 10, seed=1)
    assert compute_psnr(natural, noisy) == pytest.approx(peak_signal_noise_ratio(natural, noisy, data_range=255))


def test_ssim_properties(natural):
    assert compute_ssim(natural, natural) == pytest.approx(1.0)
    noisy = add_gaussian_noise(natural, 20, seed=2)
    mild = add_gaussian_noise(natural, 5, seed=2)
    assert compute_ssim(natural, noisy) < compute_ssim(natural, mild) < 1.0
    expected = structural_similarity(natural, mild, channel_axis=-1, data_range=255,
                                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    assert compute_ssim(natural, mild) == pytest.approx(expected)


def test_ssim_small_image_does_not_crash():
    tiny = np.random.default_rng(0).integers(0, 255, (8, 9, 3), dtype=np.uint8)
    assert -1.0 <= compute_ssim(tiny, tiny) <= 1.0


def test_shape_mismatch_raises_but_compare_images_resizes(natural):
    small = downsample(natural, 4)
    with pytest.raises(ValueError):
        compute_psnr(natural, small)
    result = compare_images(natural, small, REFERENCE_GROUND_TRUTH)
    assert result.resized_for_comparison and result.note
    assert 15 < result.psnr < 40


def test_compare_images_never_raises():
    result = compare_images(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8))
    assert math.isnan(result.ssim) and result.note  # too small for SSIM -> reported, not raised


def test_comparison_against_original(natural):
    comparison = compare_pipelines(natural, analyze_outputs=False, keep_intermediates=False)
    assert comparison.reference_kind == REFERENCE_ORIGINAL
    none_row = comparison.row("none")
    assert math.isinf(none_row.metrics.psnr) and none_row.metrics.ssim == pytest.approx(1.0)
    # A clean image: the adaptive pipeline does nothing, the fixed one changes it.
    assert comparison.row("adaptive").operations == []
    assert comparison.row("fixed").metrics.psnr < 40


def test_adaptive_beats_fixed_with_ground_truth(natural):
    degraded = add_gaussian_noise(gaussian_blur(natural, 1.5), 10, seed=3)
    comparison = compare_pipelines(degraded, reference=natural, analyze_outputs=True, keep_intermediates=False)
    assert comparison.reference_kind == REFERENCE_GROUND_TRUTH
    adaptive, fixed, none = comparison.row("adaptive"), comparison.row("fixed"), comparison.row("none")
    assert adaptive.metrics.ssim > fixed.metrics.ssim
    assert adaptive.metrics.ssim > none.metrics.ssim
    assert adaptive.after_quality is not None
