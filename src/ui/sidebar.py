"""Streamlit sidebar: image source, pipeline mode, AI super-resolution and advanced settings."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import streamlit as st

from config import APP, DECISION, INGESTION, SUPER_RESOLUTION, DecisionThresholds
from src.enhancement.super_resolution import probe_sr_status
from src.pipeline import MODE_ADAPTIVE, MODE_FIXED, MODE_LABELS, MODE_NONE
from src.utils.degradation import DEGRADATIONS
from src.utils.samples import SAMPLE_DESCRIPTIONS

SOURCE_UPLOAD = "Upload an image"
SOURCE_SAMPLE = "Built-in sample + synthetic degradation"


@dataclass
class SidebarState:
    source: str
    upload_name: str | None = None
    upload_bytes: bytes | None = None
    reference_name: str | None = None
    reference_bytes: bytes | None = None
    sample_name: str | None = None
    degradation: str | None = None
    display_mode: str = MODE_ADAPTIVE
    sr_enabled: bool = False
    sr_model: str = SUPER_RESOLUTION.default_model
    sr_allow_download: bool = True
    thresholds: DecisionThresholds = field(default_factory=lambda: DECISION)
    thresholds_modified: bool = False
    keep_intermediates: bool = True


def render_sidebar() -> SidebarState:
    sb = st.sidebar
    sb.header("1. Input image")
    source = sb.radio("Image source", [SOURCE_UPLOAD, SOURCE_SAMPLE], index=0)
    state = SidebarState(source=source)

    if source == SOURCE_UPLOAD:
        upload = sb.file_uploader(
            "Upload JPG / PNG / WebP",
            type=list(INGESTION.allowed_extensions),
            help=f"Max {INGESTION.max_upload_megabytes:.0f} MB. Images larger than "
            f"{INGESTION.processing_max_side} px are downscaled for CPU-friendly processing.",
        )
        if upload is not None:
            state.upload_name, state.upload_bytes = upload.name, upload.getvalue()
        with sb.expander("Optional: ground-truth reference"):
            st.caption(
                "If you have a clean version of the same scene, upload it here. PSNR/SSIM are then computed "
                "against this reference instead of against the uploaded (degraded) image."
            )
            ref = st.file_uploader("Reference image", type=list(INGESTION.allowed_extensions), key="reference")
            if ref is not None:
                state.reference_name, state.reference_bytes = ref.name, ref.getvalue()
    else:
        state.sample_name = sb.selectbox(
            "Clean sample image", list(APP.sample_images),
            format_func=lambda n: SAMPLE_DESCRIPTIONS.get(n, n),
        )
        state.degradation = sb.selectbox(
            "Synthetic degradation", list(DEGRADATIONS),
            index=list(DEGRADATIONS).index("mixed"),
            format_func=lambda k: f"{k.replace('_', ' ')} - {DEGRADATIONS[k].description}",
        )
        sb.caption("The clean sample is used as the ground-truth reference for PSNR/SSIM.")

    sb.header("2. Pipeline")
    state.display_mode = sb.radio(
        "Result shown in the before/after view",
        [MODE_ADAPTIVE, MODE_FIXED, MODE_NONE],
        format_func=lambda m: MODE_LABELS[m],
        help="All three pipelines are always computed for the comparison table; this selects which one is "
        "shown in the before/after section.",
    )

    sb.header("3. AI super-resolution")
    state.sr_enabled = sb.checkbox(
        "Enable AI super-resolution (Real-ESRGAN)",
        value=False,
        help="Optional. Only applied when the decision engine flags the image as low-resolution. "
        "Real-ESRGAN is much more expensive than the classical filters.",
    )
    if state.sr_enabled:
        state.sr_model = sb.selectbox(
            "Model", list(SUPER_RESOLUTION.models),
            format_func=lambda n: f"{n} - {SUPER_RESOLUTION.models[n].description}",
        )
        state.sr_allow_download = sb.checkbox("Allow model download on first use", value=True)
    status = probe_sr_status(state.sr_model)
    version = f"PyTorch {status.torch_version}. " if status.torch_version else ""
    sb.caption(version + status.message)

    with sb.expander("Advanced settings (decision thresholds)"):
        st.caption(
            "Defaults come from config/config.py, where each value is justified. Changes here apply to this "
            "session only, so you can see how the decisions react."
        )
        t = DECISION
        overrides = {
            "contrast_low_spread": st.slider("Low-contrast spread threshold", 0.10, 0.90, t.contrast_low_spread, 0.01),
            "contrast_low_rms": st.slider("Flat (low RMS) contrast threshold", 0.02, 0.30, t.contrast_low_rms, 0.01),
            "blur_threshold": st.slider("Blur score threshold (soft below)", 1.0, 150.0, t.blur_threshold, 1.0),
            "blur_severe_threshold": st.slider("Severe blur threshold", 0.5, 60.0, t.blur_severe_threshold, 0.5),
            "noise_medium_sigma": st.slider("Noise threshold (sigma)", 1.0, 25.0, t.noise_medium_sigma, 0.5),
            "low_res_min_side": st.slider("Low-resolution shorter-side threshold (px)", 64, 1024, t.low_res_min_side, 16),
        }
        state.thresholds = replace(DECISION, **overrides)
        state.thresholds_modified = state.thresholds != DECISION
        state.keep_intermediates = st.checkbox("Keep intermediate images of each step", value=True)
    return state
