"""Stage 2: contrast, blur (Laplacian variance), noise and resolution analysis."""

from __future__ import annotations

import numpy as np
import pytest

from config import DECISION
from src.analysis import analyze_contrast, analyze_image, analyze_noise, analyze_resolution, laplacian_variance
from src.analysis.blur import laplacian_noise_energy
from src.analysis.contrast import histogram_skewness, luminance_histogram
from src.analysis.noise import impulse_noise_ratio, suppress_impulses
from src.utils.degradation import add_gaussian_noise, add_salt_and_pepper, downsample, gaussian_blur, reduce_contrast
from src.utils.image_utils import rgb_to_gray


# ----------------------------------------------------------------------------- contrast
def test_histogram_counts_every_pixel(scene):
    hist = luminance_histogram(rgb_to_gray(scene))
    assert hist.shape == (256,)
    assert hist.sum() == scene.shape[0] * scene.shape[1]


def test_spread_drops_with_contrast_reduction(natural):
    full = analyze_contrast(natural)
    low = analyze_contrast(reduce_contrast(natural, 0.4))
    assert full.histogram_spread > 0.9
    assert low.histogram_spread == pytest.approx(0.4 * full.histogram_spread, abs=0.03)
    assert low.rms_contrast < full.rms_contrast


def test_constant_image_has_zero_contrast():
    flat = np.full((64, 64, 3), 128, np.uint8)
    stats = analyze_contrast(flat)
    assert stats.histogram_spread == 0.0
    assert stats.rms_contrast == 0.0
    assert stats.skewness == 0.0


def test_skewness_sign():
    rng = np.random.default_rng(0)
    dark = np.clip(rng.exponential(20, (100, 100)), 0, 255).astype(np.uint8)  # long bright tail
    assert histogram_skewness(dark) > 1.0
    assert histogram_skewness(255 - dark) < -1.0


# ----------------------------------------------------------------------------- blur
def test_laplacian_variance_drops_with_blur(scene):
    sharp = laplacian_variance(rgb_to_gray(scene))
    blurred = laplacian_variance(rgb_to_gray(gaussian_blur(scene, 2.0)))
    assert blurred < sharp / 5


def test_laplacian_variance_of_flat_image_is_zero():
    assert laplacian_variance(np.full((32, 32), 77, np.uint8)) == 0.0


def test_raw_kernel_noise_energy_is_20():
    # The 3x3 Laplacian [[0,1,0],[1,-4,1],[0,1,0]] has sum(k^2) = 20.
    assert laplacian_noise_energy(0.0) == pytest.approx(20.0)
    assert laplacian_noise_energy(1.0) < 1.0  # pre-smoothing suppresses noise strongly


def test_blur_score_classifies_sharp_and_blurry(natural):
    sharp = analyze_image(natural).sharpness.blur_score
    blurry = analyze_image(gaussian_blur(natural, 2.0)).sharpness.blur_score
    assert sharp >= DECISION.blur_threshold
    assert blurry < DECISION.blur_severe_threshold


def test_blur_score_is_contrast_invariant(natural):
    a = analyze_image(natural).sharpness.blur_score
    b = analyze_image(reduce_contrast(natural, 0.45)).sharpness.blur_score
    assert b == pytest.approx(a, rel=0.15)


def test_noise_does_not_make_blurry_image_look_sharp(natural):
    noisy_blurry = add_gaussian_noise(gaussian_blur(natural, 2.0), 12, seed=1)
    report = analyze_image(noisy_blurry)
    assert report.sharpness.laplacian_variance_raw > 1000  # classic metric is fooled
    assert report.sharpness.blur_score < DECISION.blur_threshold  # compensated score is not


# ----------------------------------------------------------------------------- noise
@pytest.mark.parametrize("sigma", [5.0, 10.0, 20.0])
def test_noise_estimate_on_flat_image(sigma):
    flat = np.full((160, 160, 3), 120, np.uint8)
    est = analyze_noise(add_gaussian_noise(flat, sigma, seed=2)).sigma
    assert est == pytest.approx(sigma, rel=0.12)


@pytest.mark.parametrize("sigma", [5.0, 15.0])
def test_noise_estimate_on_natural_image(natural, sigma):
    est = analyze_noise(add_gaussian_noise(natural, sigma, seed=3)).sigma
    assert est == pytest.approx(sigma, rel=0.15)


def test_clean_image_has_low_noise(natural):
    assert analyze_noise(natural).sigma < DECISION.noise_medium_sigma


def test_impulse_detection(natural):
    assert impulse_noise_ratio(rgb_to_gray(natural)) < DECISION.impulse_ratio_threshold
    sp = add_salt_and_pepper(natural, 0.05, seed=4)
    assert impulse_noise_ratio(rgb_to_gray(sp)) > 0.02


def test_switching_median_only_touches_impulses(natural):
    sp = add_salt_and_pepper(natural, 0.03, seed=5)
    cleaned = suppress_impulses(sp)
    changed = np.any(cleaned != sp, axis=-1)
    impulses = np.any(sp != natural, axis=-1)
    hits = np.count_nonzero(changed & impulses)
    # Almost every replaced pixel was an impulse (a few genuine near-black/white fine
    # details are locally indistinguishable); most impulses are replaced - the missed
    # ones sit on already dark/bright areas where they are nearly invisible.
    assert hits / np.count_nonzero(changed) > 0.99
    assert hits / np.count_nonzero(impulses) > 0.7


# ----------------------------------------------------------------------------- resolution
def test_resolution_flags():
    assert analyze_resolution(128, 128).resolution_flag
    assert analyze_resolution(1000, 200).resolution_flag  # shorter side < 256
    mid = analyze_resolution(640, 480)
    assert not mid.resolution_flag and mid.category == "Acceptable"
    assert analyze_resolution(1920, 1080).category == "High"
    assert analyze_resolution(640, 480).megapixels == pytest.approx(0.3072)


def test_resolution_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        analyze_resolution(0, 10)


def test_analyze_image_features_are_complete(natural):
    report = analyze_image(downsample(natural, 4))
    features = report.features()
    for key in ("histogram_spread", "contrast_score", "blur_score", "noise_score", "width", "height", "resolution_flag"):
        assert key in features
    assert features["resolution_flag"] is True
    assert report.analysis_time_ms > 0
