"""
The proposed adaptive pipeline: Stage 2 (analysis) -> Stage 3 (decision) -> Stage 4
(execution), with timing for every stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import DECISION, ENHANCEMENT, DecisionThresholds, EnhancementDefaults
from src.analysis.quality_analyzer import QualityReport, analyze_image
from src.decision.decision_engine import DecisionEngine, DecisionOptions, DecisionResult
from src.enhancement.super_resolution import SuperResolver
from src.pipeline.executor import StepResult, execute_operations
from src.utils.image_utils import ensure_rgb_uint8

MODE_NONE = "none"
MODE_FIXED = "fixed"
MODE_ADAPTIVE = "adaptive"

MODE_LABELS = {
    MODE_NONE: "No enhancement (original)",
    MODE_FIXED: "Fixed classical pipeline",
    MODE_ADAPTIVE: "Adaptive pipeline (proposed)",
}


@dataclass
class PipelineResult:
    mode: str
    output: np.ndarray
    steps: list[StepResult] = field(default_factory=list)
    quality: QualityReport | None = None
    decision: DecisionResult | None = None
    analysis_ms: float = 0.0
    decision_ms: float = 0.0
    enhancement_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    @property
    def total_ms(self) -> float:
        return self.analysis_ms + self.decision_ms + self.enhancement_ms

    @property
    def operation_labels(self) -> list[str]:
        return [s.label for s in self.steps if s.error is None]


def run_adaptive_pipeline(
    image: np.ndarray,
    *,
    thresholds: DecisionThresholds = DECISION,
    params: EnhancementDefaults = ENHANCEMENT,
    sr_enabled: bool = False,
    super_resolver: SuperResolver | None = None,
    quality: QualityReport | None = None,
    keep_intermediates: bool = True,
    on_stage: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Analyse -> decide -> execute.

    Args:
        image: RGB uint8 working image (Stage 1 output).
        thresholds / params: decision thresholds and operator parameters (overridable).
        sr_enabled: user opted in to super-resolution for low-resolution images.
        super_resolver: Real-ESRGAN or fallback backend (``None`` = not loaded).
        quality: pre-computed Stage-2 report (avoids re-analysing in the UI).
        on_stage: optional callback receiving "analysis", "decision", "execution" as each
            stage starts (used by the UI's stage tracker).
    """
    rgb = ensure_rgb_uint8(image)
    notify = on_stage or (lambda _stage: None)

    notify("analysis")
    start = time.perf_counter()
    report = quality if quality is not None else analyze_image(rgb, thresholds)
    analysis_ms = report.analysis_time_ms if quality is not None else (time.perf_counter() - start) * 1000.0

    notify("decision")
    start = time.perf_counter()
    options = DecisionOptions(
        sr_enabled=sr_enabled,
        sr_ai_available=bool(super_resolver is not None and getattr(super_resolver, "is_ai", False)),
        sr_fallback_available=True,
    )
    decision = DecisionEngine(thresholds, params).decide(report.features(), options)
    decision_ms = (time.perf_counter() - start) * 1000.0

    notify("execution")
    start = time.perf_counter()
    output, steps, errors = execute_operations(rgb, decision.operations, super_resolver, keep_intermediates)
    enhancement_ms = (time.perf_counter() - start) * 1000.0

    return PipelineResult(
        mode=MODE_ADAPTIVE,
        output=output,
        steps=steps,
        quality=report,
        decision=decision,
        analysis_ms=analysis_ms,
        decision_ms=decision_ms,
        enhancement_ms=enhancement_ms,
        errors=errors,
    )


def run_no_enhancement(image: np.ndarray) -> PipelineResult:
    """Baseline A: the input itself."""
    return PipelineResult(mode=MODE_NONE, output=ensure_rgb_uint8(image).copy())
