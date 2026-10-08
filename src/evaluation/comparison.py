"""
Baseline comparison: (A) no enhancement, (B) fixed classical pipeline, (C) adaptive
pipeline - as required by the "Expected Outcome" section of the report.

The same function is used by the Streamlit app (one image) and by
``scripts/evaluate_dataset.py`` (many images), so app and offline results are produced
by identical code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import DECISION, ENHANCEMENT, DecisionThresholds, EnhancementDefaults
from src.analysis.quality_analyzer import QualityReport, analyze_image
from src.enhancement.super_resolution import SuperResolver
from src.evaluation.metrics import REFERENCE_GROUND_TRUTH, REFERENCE_ORIGINAL, MetricResult, compare_images
from src.pipeline import (
    MODE_ADAPTIVE,
    MODE_FIXED,
    MODE_NONE,
    PipelineResult,
    run_adaptive_pipeline,
    run_fixed_pipeline,
    run_no_enhancement,
)
from src.utils.image_utils import ensure_rgb_uint8


@dataclass
class ComparisonRow:
    mode: str
    label: str
    operations: list[str]
    metrics: MetricResult
    time_ms: float
    output_shape: tuple[int, int]
    after_quality: QualityReport | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineComparison:
    rows: list[ComparisonRow]
    results: dict[str, PipelineResult]
    reference_kind: str

    def row(self, mode: str) -> ComparisonRow:
        return next(r for r in self.rows if r.mode == mode)


def compare_pipelines(
    image: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    thresholds: DecisionThresholds = DECISION,
    params: EnhancementDefaults = ENHANCEMENT,
    sr_enabled: bool = False,
    super_resolver: SuperResolver | None = None,
    quality: QualityReport | None = None,
    analyze_outputs: bool = True,
    keep_intermediates: bool = True,
    adaptive_result: PipelineResult | None = None,
) -> PipelineComparison:
    """Run the three pipelines on ``image`` and score each against the reference.

    ``reference`` is a clean ground-truth image when available; otherwise the input
    image itself is the reference (the report's protocol), which only measures how much
    each pipeline changed the input.
    """
    rgb = ensure_rgb_uint8(image)
    reference_kind = REFERENCE_GROUND_TRUTH if reference is not None else REFERENCE_ORIGINAL
    ref = ensure_rgb_uint8(reference) if reference is not None else rgb

    results = {
        MODE_NONE: run_no_enhancement(rgb),
        MODE_FIXED: run_fixed_pipeline(rgb, keep_intermediates=keep_intermediates),
        MODE_ADAPTIVE: adaptive_result if adaptive_result is not None else run_adaptive_pipeline(
            rgb,
            thresholds=thresholds,
            params=params,
            sr_enabled=sr_enabled,
            super_resolver=super_resolver,
            quality=quality,
            keep_intermediates=keep_intermediates,
        ),
    }

    rows: list[ComparisonRow] = []
    for mode, result in results.items():
        after = None
        if analyze_outputs:
            try:
                after = analyze_image(result.output, thresholds) if mode != MODE_NONE else (
                    quality or results[MODE_ADAPTIVE].quality
                )
            except Exception:  # analysis of an output must never break the comparison
                after = None
        rows.append(ComparisonRow(
            mode=mode,
            label=result.label,
            operations=result.operation_labels,
            metrics=compare_images(ref, result.output, reference_kind),
            time_ms=result.total_ms,
            output_shape=(int(result.output.shape[0]), int(result.output.shape[1])),
            after_quality=after,
            errors=list(result.errors),
        ))
    return PipelineComparison(rows=rows, results=results, reference_kind=reference_kind)
