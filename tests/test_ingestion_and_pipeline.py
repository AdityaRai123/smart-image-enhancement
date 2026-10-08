"""Stage 1 error handling and pipeline robustness."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from config import INGESTION
from src.decision.decision_engine import PlannedOperation
from src.pipeline import execute_operations, run_adaptive_pipeline, run_fixed_pipeline
from src.utils.image_utils import ImageLoadError, ensure_rgb_uint8, load_image_bytes


def encode(array: np.ndarray, fmt: str, mode: str | None = None) -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(array, mode) if mode else Image.fromarray(array)
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("fmt,ext", [("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp")])
def test_loads_supported_formats(scene, fmt, ext):
    ingested = load_image_bytes(encode(scene, fmt), f"photo.{ext}")
    assert ingested.image.shape == scene.shape
    assert ingested.source_format == fmt


def test_grayscale_png(scene):
    gray = scene[..., 0]
    ingested = load_image_bytes(encode(gray, "PNG"), "gray.png")
    assert ingested.image.shape == (*gray.shape, 3)
    assert ingested.is_grayscale


def test_rgba_is_flattened_on_white():
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[:16, :, 3] = 255  # top half opaque red, bottom half transparent
    ingested = load_image_bytes(encode(rgba, "PNG"), "alpha.png")
    assert ingested.had_alpha
    assert tuple(ingested.image[0, 0]) == (200, 0, 0)
    assert tuple(ingested.image[-1, -1]) == (255, 255, 255)


def test_palette_and_16bit_images(scene):
    palette = Image.fromarray(scene).convert("P")
    buffer = io.BytesIO()
    palette.save(buffer, format="PNG")
    assert load_image_bytes(buffer.getvalue(), "p.png").image.shape == scene.shape

    deep = (scene[..., 0].astype(np.uint16) * 257)
    buffer = io.BytesIO()
    Image.fromarray(deep).save(buffer, format="PNG")
    ingested = load_image_bytes(buffer.getvalue(), "deep.png")
    assert ingested.image.dtype == np.uint8
    assert np.abs(ingested.image[..., 0].astype(int) - scene[..., 0].astype(int)).max() <= 1


def test_corrupted_file_is_rejected(scene):
    data = encode(scene, "PNG")
    with pytest.raises(ImageLoadError):
        load_image_bytes(data[: len(data) // 3], "broken.png")
    with pytest.raises(ImageLoadError):
        load_image_bytes(b"this is not an image", "fake.png")
    with pytest.raises(ImageLoadError):
        load_image_bytes(b"", "empty.png")


def test_unsupported_extension_and_format(scene):
    with pytest.raises(ImageLoadError, match="Unsupported file type"):
        load_image_bytes(encode(scene, "PNG"), "doc.pdf")
    with pytest.raises(ImageLoadError, match="not supported"):
        load_image_bytes(encode(scene, "GIF"), "anim.png")


def test_too_small_image_is_rejected():
    with pytest.raises(ImageLoadError, match="too small"):
        load_image_bytes(encode(np.zeros((8, 8, 3), np.uint8), "PNG"), "tiny.png")


def test_large_image_is_downscaled():
    big = np.random.default_rng(0).integers(0, 255, (300, INGESTION.processing_max_side + 500, 3), dtype=np.uint8)
    ingested = load_image_bytes(encode(big, "PNG"), "wide.png")
    assert ingested.was_resized
    assert max(ingested.image.shape[:2]) == INGESTION.processing_max_side


def test_decompression_bomb_guard(monkeypatch, scene):
    from dataclasses import replace

    tight = replace(INGESTION, max_decoded_pixels=1000)
    with pytest.raises(ImageLoadError, match="safety limit"):
        load_image_bytes(encode(scene, "PNG"), "big.png", settings=tight)


@pytest.mark.parametrize("bad", [np.zeros((0, 5)), np.zeros((4, 4, 2)), np.full((4, 4), np.nan)])
def test_invalid_arrays_rejected(bad):
    with pytest.raises(ImageLoadError):
        ensure_rgb_uint8(bad)


def test_float_and_bool_arrays_are_converted():
    assert ensure_rgb_uint8(np.ones((4, 4), float)).max() == 255
    assert ensure_rgb_uint8(np.ones((4, 4), bool)).max() == 255


def test_failing_step_does_not_crash_pipeline(scene):
    ops = [
        PlannedOperation("unsharp_mask", "USM", "sharpen", {"sigma": -1}, "bad params", "T", 0),
        PlannedOperation("contrast_stretch", "Stretch", "contrast", {}, "ok", "T", 1),
    ]
    output, steps, errors = execute_operations(scene, ops)
    assert len(errors) == 1 and steps[0].error
    assert steps[1].error is None
    assert output.shape == scene.shape


def test_sr_step_without_backend_uses_lanczos(scene):
    op = PlannedOperation("super_resolution", "SR", "super_resolution", {"method": "lanczos", "scale": 2}, "", "R1", 5)
    output, steps, errors = execute_operations(scene[:40, :40], [op])
    assert not errors and output.shape == (80, 80, 3)


def test_adaptive_pipeline_on_low_resolution_with_fallback(natural):
    from src.enhancement.super_resolution import InterpolationUpscaler
    from src.utils.degradation import downsample

    small = downsample(natural, 4)
    result = run_adaptive_pipeline(small, sr_enabled=True, super_resolver=InterpolationUpscaler())
    assert result.output.shape[0] == small.shape[0] * 4
    assert "Lanczos" in result.steps[-1].label


def test_fixed_pipeline_always_runs_three_steps(natural):
    result = run_fixed_pipeline(natural, keep_intermediates=False)
    assert [s.key for s in result.steps] == ["clahe", "bilateral_denoise", "unsharp_mask"]
