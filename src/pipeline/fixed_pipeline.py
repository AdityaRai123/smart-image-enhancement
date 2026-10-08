"""
Baseline B: the fixed classical pipeline.

The same chain (default CLAHE -> bilateral denoising -> unsharp masking, mid-range
parameters from ``config.FIXED_PIPELINE``) is applied to every image, whatever its
measured quality. Comparing it with the adaptive pipeline isolates the value of the
decision layer, because both use the same operator implementations.
"""

from __future__ import annotations

import time

import numpy as np

from config import FIXED_PIPELINE, FixedPipelineSettings
from src.decision.decision_engine import PlannedOperation
from src.pipeline.adaptive_pipeline import MODE_FIXED, PipelineResult
from src.pipeline.executor import execute_operations

_FIXED_REASON = "Fixed pipeline: applied to every image regardless of its measured quality."


def fixed_operations(settings: FixedPipelineSettings = FIXED_PIPELINE) -> list[PlannedOperation]:
    """Build the planned-operation list of the fixed chain."""
    catalogue = {
        "clahe": PlannedOperation(
            "clahe", "CLAHE", "contrast",
            {"clip_limit": settings.clahe_clip_limit, "tile_grid": list(settings.clahe_tile_grid)},
            _FIXED_REASON, "FIXED", 0,
        ),
        "bilateral_denoise": PlannedOperation(
            "bilateral_denoise", "Bilateral denoising", "denoise",
            {"diameter": settings.bilateral_diameter, "sigma_color": settings.bilateral_sigma_color,
             "sigma_space": settings.bilateral_sigma_space},
            _FIXED_REASON, "FIXED", 0,
        ),
        "unsharp_mask": PlannedOperation(
            "unsharp_mask", "Unsharp-mask sharpening", "sharpen",
            {"sigma": settings.usm_sigma, "amount": settings.usm_amount, "threshold": settings.usm_threshold},
            _FIXED_REASON, "FIXED", 0,
        ),
    }
    unknown = [k for k in settings.operations if k not in catalogue]
    if unknown:
        raise ValueError(f"Unknown fixed-pipeline operations: {unknown}")
    return [catalogue[k] for k in settings.operations]


def run_fixed_pipeline(
    image: np.ndarray,
    settings: FixedPipelineSettings = FIXED_PIPELINE,
    keep_intermediates: bool = True,
) -> PipelineResult:
    start = time.perf_counter()
    output, steps, errors = execute_operations(image, fixed_operations(settings), None, keep_intermediates)
    return PipelineResult(
        mode=MODE_FIXED,
        output=output,
        steps=steps,
        enhancement_ms=(time.perf_counter() - start) * 1000.0,
        errors=errors,
    )
