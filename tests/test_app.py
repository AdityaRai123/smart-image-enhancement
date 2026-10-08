"""End-to-end tests of the Streamlit app with Streamlit's headless AppTest runner."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def page_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [h.value for h in at.header] + [h.value for h in at.subheader]
    parts += [str(i.value) for i in at.info] + [str(s.value) for s in at.success] + [str(w.value) for w in at.warning]
    return "\n".join(parts)


def use_sample(at: AppTest, degradation: str) -> AppTest:
    at.sidebar.radio[0].set_value("Built-in sample + synthetic degradation").run()
    at.sidebar.selectbox[1].set_value(degradation).run()
    return at


def test_start_page_without_image():
    at = run_app()
    assert not at.exception
    assert any("Upload an image" in str(i.value) for i in at.info)


def test_sample_mode_runs_all_five_stages():
    at = use_sample(run_app(), "mixed")
    assert not at.exception and not at.error
    headers = [h.value for h in at.header]
    for stage in ("Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"):
        assert any(h.startswith(stage) for h in headers), stage
    text = page_text(at)
    assert "Why did the system choose these operations?" in [s.value for s in at.subheader]
    assert "Gaussian denoising" in text and "Unsharp-mask sharpening" in text
    assert any(m.label == "PSNR" for m in at.metric) and any(m.label == "SSIM" for m in at.metric)


def test_good_image_is_left_unchanged():
    at = use_sample(run_app(), "good")
    assert not at.exception
    assert any("No enhancement needed" in str(s.value) for s in at.success)


def test_fixed_pipeline_view():
    at = use_sample(run_app(), "noisy")
    at.sidebar.radio[1].set_value("fixed").run()
    assert not at.exception
    assert any("Fixed classical pipeline" in s.value for s in at.subheader)


def test_low_resolution_with_super_resolution_enabled():
    """Works with Real-ESRGAN when installed, otherwise with the labelled Lanczos fallback."""
    at = use_sample(run_app(), "low_resolution")
    assert any("super-resolution" in str(i.value) for i in at.info)  # recommendation while disabled
    at.sidebar.checkbox[0].check().run()
    assert not at.exception and not at.error
    text = page_text(at)
    assert "Real-ESRGAN" in text or "Lanczos" in text
    size_metric = next(m for m in at.metric if m.label == "Output size")
    assert size_metric.value != "128 x 128"  # the image was upscaled
