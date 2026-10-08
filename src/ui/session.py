"""
Input loading, model caching and the per-image processing run for the Streamlit app.

Results are cached in ``st.session_state`` under a key derived from the image and the
settings, so moving a slider or switching tabs does not recompute the pipelines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import streamlit as st

from src.analysis.quality_analyzer import QualityReport, analyze_image
from src.enhancement.super_resolution import (
    InterpolationUpscaler,
    RealESRGANUpscaler,
    SRUnavailableError,
    SuperResolver,
    torch_available,
)
from src.evaluation.comparison import PipelineComparison, compare_pipelines
from src.pipeline import run_adaptive_pipeline
from src.ui.sidebar import SOURCE_SAMPLE, SidebarState
from src.utils.degradation import DEGRADATIONS, apply_degradation, crop_to_multiple
from src.utils.image_utils import IngestedImage, ingest_array, load_image_bytes
from src.utils.samples import load_sample

logger = logging.getLogger(__name__)

STAGES = (
    "Image Ingestion",
    "Quality Analysis",
    "Decision Engine",
    "Enhancement Execution",
    "Evaluation & Presentation",
)


@dataclass
class LoadedInput:
    ingested: IngestedImage
    reference: np.ndarray | None
    reference_label: str | None
    key: str


@dataclass
class SRBackend:
    resolver: SuperResolver | None
    ai_ready: bool
    message: str


@dataclass
class AppRun:
    quality: QualityReport
    comparison: PipelineComparison
    stage_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _digest(data: bytes | None) -> str:
    return hashlib.sha1(data).hexdigest() if data else "none"


def load_input(state: SidebarState) -> LoadedInput | None:
    """Stage 1. Returns ``None`` when there is nothing to process yet.

    Raises :class:`src.utils.image_utils.ImageLoadError` with a user-facing message.
    """
    if state.source == SOURCE_SAMPLE:
        clean = crop_to_multiple(load_sample(state.sample_name), 4)
        spec = DEGRADATIONS[state.degradation]
        degraded = apply_degradation(clean, spec, seed=0)
        ingested = ingest_array(
            degraded, filename=f"{state.sample_name}_{state.degradation}.png", source_format="synthetic"
        )
        ingested.notes.insert(0, f"Synthetic degradation applied: {spec.description}.")
        return LoadedInput(ingested, clean, f"clean '{state.sample_name}' sample", f"sample:{state.sample_name}:{state.degradation}")

    if not state.upload_bytes:
        return None
    ingested = load_image_bytes(state.upload_bytes, state.upload_name)
    reference = None
    label = None
    if state.reference_bytes:
        reference = load_image_bytes(state.reference_bytes, state.reference_name).image
        label = f"uploaded reference '{state.reference_name}'"
    key = f"upload:{_digest(state.upload_bytes)}:{_digest(state.reference_bytes)}"
    return LoadedInput(ingested, reference, label, key)


@st.cache_resource(show_spinner="Loading Real-ESRGAN (first use may download the model)...")
def _load_realesrgan(model_name: str, allow_download: bool) -> RealESRGANUpscaler:
    # Exceptions are not cached by Streamlit, so a failed download can be retried.
    return RealESRGANUpscaler.load(model_name, allow_download)


def get_sr_backend(state: SidebarState) -> SRBackend:
    """Load Real-ESRGAN (cached) when AI super-resolution is enabled; never raises."""
    if not state.sr_enabled:
        return SRBackend(None, False, "AI super-resolution disabled.")
    if not torch_available():
        return SRBackend(
            InterpolationUpscaler(), False,
            "PyTorch is not installed, so Real-ESRGAN is unavailable; low-resolution images use the "
            "classical Lanczos fallback. Install it with: pip install -r requirements-ai.txt",
        )
    try:
        resolver = _load_realesrgan(state.sr_model, state.sr_allow_download)
        return SRBackend(resolver, True, f"{resolver.name} loaded (CPU-friendly, cached).")
    except SRUnavailableError as exc:
        return SRBackend(InterpolationUpscaler(), False, f"{exc} Using the classical Lanczos fallback instead.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected Real-ESRGAN failure")
        return SRBackend(InterpolationUpscaler(), False, f"Real-ESRGAN failed to load ({exc}); using Lanczos fallback.")


def run_key(loaded: LoadedInput, state: SidebarState, sr: SRBackend) -> str:
    payload = {
        "input": loaded.key,
        "thresholds": asdict(state.thresholds),
        "sr_enabled": state.sr_enabled,
        "sr_model": state.sr_model,
        "sr_ai": sr.ai_ready,
        "keep": state.keep_intermediates,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def process(
    loaded: LoadedInput,
    state: SidebarState,
    sr: SRBackend,
    on_stage: Callable[[int, str], None],
) -> AppRun:
    """Stages 2-5 for one image. ``on_stage(index, status)`` drives the UI tracker."""
    image = loaded.ingested.image
    stage_ms: dict[str, float] = {}

    on_stage(1, "running")
    start = time.perf_counter()
    quality = analyze_image(image, state.thresholds)
    stage_ms["Quality Analysis"] = (time.perf_counter() - start) * 1000.0
    on_stage(1, "done")

    def adaptive_stage(name: str) -> None:
        if name == "decision":
            on_stage(2, "running")
        elif name == "execution":
            on_stage(2, "done")
            on_stage(3, "running")

    adaptive = run_adaptive_pipeline(
        image,
        thresholds=state.thresholds,
        sr_enabled=state.sr_enabled,
        super_resolver=sr.resolver if state.sr_enabled else None,
        quality=quality,
        keep_intermediates=state.keep_intermediates,
        on_stage=adaptive_stage,
    )
    stage_ms["Decision Engine"] = adaptive.decision_ms
    stage_ms["Enhancement Execution"] = adaptive.enhancement_ms
    on_stage(3, "done")

    on_stage(4, "running")
    start = time.perf_counter()
    comparison = compare_pipelines(
        image,
        reference=loaded.reference,
        thresholds=state.thresholds,
        quality=quality,
        keep_intermediates=state.keep_intermediates,
        adaptive_result=adaptive,
    )
    stage_ms["Evaluation & Presentation"] = (time.perf_counter() - start) * 1000.0
    on_stage(4, "done")
    return AppRun(quality=quality, comparison=comparison, stage_ms=stage_ms, errors=list(adaptive.errors))
