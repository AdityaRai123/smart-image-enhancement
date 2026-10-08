"""Stage 4 operators: histogram equalisation (custom), stretching, CLAHE, denoising, sharpening, SR."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.enhancement import (
    OPERATIONS,
    apply_operation,
    bilateral_denoise,
    clahe,
    contrast_stretch,
    gaussian_denoise,
    histogram_equalization,
    median_denoise,
    unsharp_mask,
)
from src.enhancement.histogram_equalization import equalize_gray_from_scratch
from src.enhancement.super_resolution import (
    InterpolationUpscaler,
    SRUnavailableError,
    create_super_resolver,
    get_model_spec,
)
from src.evaluation import compute_psnr
from src.utils.degradation import add_gaussian_noise, add_salt_and_pepper, gaussian_blur, reduce_contrast
from src.utils.image_utils import rgb_to_gray


# ----------------------------------------------------------------------- histogram equalisation
def test_custom_he_matches_opencv(natural):
    gray = rgb_to_gray(reduce_contrast(natural, 0.5))
    ours, _ = equalize_gray_from_scratch(gray)
    reference = cv2.equalizeHist(gray)
    assert np.abs(ours.astype(int) - reference.astype(int)).max() <= 1


def test_custom_he_intermediates_are_consistent():
    gray = np.random.default_rng(0).integers(50, 120, (64, 64), dtype=np.uint8)
    out, details = equalize_gray_from_scratch(gray)
    assert details.histogram.sum() == gray.size
    assert details.pdf.sum() == pytest.approx(1.0)
    assert details.cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(details.cdf) >= 0)  # CDF is monotone
    assert np.all(np.diff(details.lut.astype(int)) >= 0)  # mapping is monotone
    assert out.min() == 0 and out.max() == 255  # full range after equalisation


def test_custom_he_constant_image_is_identity():
    gray = np.full((16, 16), 90, np.uint8)
    out, _ = equalize_gray_from_scratch(gray)
    assert np.array_equal(out, gray)


def test_color_he_preserves_chroma(natural):
    out = histogram_equalization(reduce_contrast(natural, 0.5))
    src_cr = cv2.cvtColor(reduce_contrast(natural, 0.5), cv2.COLOR_RGB2YCrCb)[..., 1:]
    out_cr = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)[..., 1:]
    # Only Y is equalised: chroma is unchanged up to colour-conversion rounding.
    assert np.abs(src_cr.astype(int) - out_cr.astype(int)).mean() < 2.0


def test_he_rejects_non_uint8():
    with pytest.raises(ValueError):
        equalize_gray_from_scratch(np.zeros((4, 4), np.float32))


# ----------------------------------------------------------------------- contrast
def test_contrast_stretch_restores_range(natural):
    low = reduce_contrast(natural, 0.4)
    out = contrast_stretch(low, 0.5, 99.5)
    gray = rgb_to_gray(out)
    assert np.percentile(gray, 99.5) - np.percentile(gray, 0.5) > 240
    assert compute_psnr(natural, out) > compute_psnr(natural, low) + 5


def test_contrast_stretch_constant_and_invalid():
    flat = np.full((20, 20, 3), 100, np.uint8)
    assert np.array_equal(contrast_stretch(flat), flat)
    with pytest.raises(ValueError):
        contrast_stretch(flat, 99, 1)


def test_clahe_increases_local_contrast(natural):
    low = reduce_contrast(natural, 0.4)
    assert rgb_to_gray(clahe(low, 2.0)).std() > rgb_to_gray(low).std()
    with pytest.raises(ValueError):
        clahe(low, 0)


def test_clahe_small_image():
    tiny = np.random.default_rng(1).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    assert clahe(tiny).shape == tiny.shape


# ----------------------------------------------------------------------- denoising
@pytest.mark.parametrize("fn,kwargs", [
    (gaussian_denoise, {"sigma": 1.0}),
    (bilateral_denoise, {"diameter": 7, "sigma_color": 60, "sigma_space": 3}),
    (median_denoise, {"ksize": 3}),
])
def test_denoisers_improve_psnr_on_gaussian_noise(natural, fn, kwargs):
    noisy = add_gaussian_noise(natural, 15, seed=6)
    assert compute_psnr(natural, fn(noisy, **kwargs)) > compute_psnr(natural, noisy) + 2


def test_median_removes_impulses_best(natural):
    sp = add_salt_and_pepper(natural, 0.03, seed=7)
    median = compute_psnr(natural, median_denoise(sp, 3))
    gaussian = compute_psnr(natural, gaussian_denoise(sp, 1.0))
    assert median > gaussian + 3


def test_denoise_parameter_validation(scene):
    with pytest.raises(ValueError):
        median_denoise(scene, 4)
    with pytest.raises(ValueError):
        gaussian_denoise(scene, 0)
    with pytest.raises(ValueError):
        bilateral_denoise(scene, 0, 10, 3)


# ----------------------------------------------------------------------- sharpening
def test_unsharp_mask_restores_blur(natural):
    blurred = gaussian_blur(natural, 1.0)
    out = unsharp_mask(blurred, sigma=1.5, amount=1.0)
    assert compute_psnr(natural, out) > compute_psnr(natural, blurred)
    assert cv2.Laplacian(rgb_to_gray(out), cv2.CV_64F).var() > cv2.Laplacian(rgb_to_gray(blurred), cv2.CV_64F).var()


def test_unsharp_mask_zero_amount_is_identity(scene):
    assert np.abs(unsharp_mask(scene, 1.0, 0.0).astype(int) - scene.astype(int)).max() <= 1


def test_unsharp_threshold_ignores_small_differences():
    flat = np.full((32, 32, 3), 128, np.uint8)
    noisy = add_gaussian_noise(flat, 1.0, seed=8)
    out = unsharp_mask(noisy, sigma=1.0, amount=2.0, threshold=10.0)
    assert np.abs(out.astype(int) - noisy.astype(int)).max() <= 1


def test_unsharp_validation(scene):
    with pytest.raises(ValueError):
        unsharp_mask(scene, sigma=0)
    with pytest.raises(ValueError):
        unsharp_mask(scene, amount=-1)


# ----------------------------------------------------------------------- registry & grayscale
def test_registry_and_grayscale_inputs():
    gray = np.random.default_rng(2).integers(0, 255, (40, 40), dtype=np.uint8)
    for key, info in OPERATIONS.items():
        if info.function is None:
            continue
        out = apply_operation(key, gray)
        assert out.shape == (40, 40, 3) and out.dtype == np.uint8
    with pytest.raises(KeyError):
        apply_operation("super_resolution", gray)


# ----------------------------------------------------------------------- super-resolution
def test_interpolation_fallback_upscales(scene):
    result = InterpolationUpscaler(scale=4).upscale(scene[:50, :60])
    assert result.image.shape == (200, 240, 3)
    assert result.is_ai is False and "not AI" in result.method


def test_unknown_model_is_reported_not_raised():
    resolver, status = create_super_resolver("does-not-exist")
    assert status.ready is False
    assert isinstance(resolver, InterpolationUpscaler)
    with pytest.raises(SRUnavailableError):
        get_model_spec("does-not-exist")


def test_missing_weights_without_download_falls_back(tmp_path):
    from dataclasses import replace

    from config import SUPER_RESOLUTION

    settings = replace(SUPER_RESOLUTION, model_dir=tmp_path)
    resolver, status = create_super_resolver(allow_download=False, settings=settings)
    assert status.ready is False
    assert resolver is not None and resolver.is_ai is False
    assert "not found" in status.message or "PyTorch" in status.message


def test_realesrgan_if_available(scene):
    """Runs only when torch and the model file are present (skipped otherwise)."""
    from src.enhancement.super_resolution import probe_sr_status

    status = probe_sr_status()
    if not (status.torch_installed and status.model_file_present):
        pytest.skip("Real-ESRGAN weights or PyTorch not available")
    resolver, status = create_super_resolver(allow_download=False)
    assert status.ready and resolver.is_ai
    result = resolver.upscale(scene[:48, :64])
    assert result.image.shape == (192, 256, 3)
