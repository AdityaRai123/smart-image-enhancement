"""
Stage 4 - Enhancement execution.

Executes an ordered list of :class:`PlannedOperation` objects. Shared by the adaptive
and the fixed pipeline so both use exactly the same operator implementations - the
only difference between them is *which* operations run and with *which* parameters.

A failing step never crashes the pipeline: the error is recorded, the step is skipped
and execution continues with the previous image.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.decision.decision_engine import PlannedOperation
from src.enhancement import apply_operation
from src.enhancement.super_resolution import InterpolationUpscaler, SuperResolver
from src.utils.image_utils import ensure_rgb_uint8

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    key: str
    label: str
    params: dict[str, Any]
    reason: str
    elapsed_ms: float
    output_shape: tuple[int, int]
    image: np.ndarray | None = None  # intermediate result (kept for the UI step gallery)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def _plan_params(op: PlannedOperation) -> dict[str, Any]:
    """Convert planned parameters into keyword arguments of the operator function."""
    params = dict(op.params)
    if op.key == "clahe" and "tile_grid" in params:
        params["tile_grid"] = tuple(params["tile_grid"])
    if op.key == "histogram_equalization":
        params.pop("color_space", None)  # informational only
    return params


def execute_operations(
    image: np.ndarray,
    operations: list[PlannedOperation],
    super_resolver: SuperResolver | None = None,
    keep_intermediates: bool = True,
) -> tuple[np.ndarray, list[StepResult], list[str]]:
    """Run the operations in order. Returns ``(output, steps, errors)``."""
    current = ensure_rgb_uint8(image)
    steps: list[StepResult] = []
    errors: list[str] = []

    for op in operations:
        start = time.perf_counter()
        notes: list[str] = []
        error: str | None = None
        label = op.label
        try:
            if op.key == "super_resolution":
                resolver = super_resolver
                if op.params.get("method") == "lanczos" and (resolver is None or resolver.is_ai):
                    resolver = InterpolationUpscaler(scale=int(op.params.get("scale", 4)))
                if resolver is None:
                    raise RuntimeError("No super-resolution backend is available.")
                result = resolver.upscale(current)
                current = result.image
                notes.extend(result.notes)
                label = f"{op.label} - {result.method}"
            else:
                current = apply_operation(op.key, current, **_plan_params(op))
        except Exception as exc:  # one failing step must not crash the whole run
            error = f"{op.label} failed: {exc}"
            logger.exception("Operation %s failed", op.key)
            errors.append(error)
        elapsed = (time.perf_counter() - start) * 1000.0
        steps.append(StepResult(
            key=op.key,
            label=label,
            params=dict(op.params),
            reason=op.reason,
            elapsed_ms=elapsed,
            output_shape=(int(current.shape[0]), int(current.shape[1])),
            image=current.copy() if keep_intermediates else None,
            notes=notes,
            error=error,
        ))
    return current, steps, errors
