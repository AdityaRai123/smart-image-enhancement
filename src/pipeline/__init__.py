"""Stage 4 - Pipelines: adaptive (proposed), fixed classical baseline, no enhancement."""

from src.pipeline.adaptive_pipeline import (  # noqa: F401
    MODE_ADAPTIVE,
    MODE_FIXED,
    MODE_LABELS,
    MODE_NONE,
    PipelineResult,
    run_adaptive_pipeline,
    run_no_enhancement,
)
from src.pipeline.executor import StepResult, execute_operations  # noqa: F401
from src.pipeline.fixed_pipeline import fixed_operations, run_fixed_pipeline  # noqa: F401
