"""Stage 5 - Evaluation: PSNR / SSIM against the original or a ground-truth reference."""

from src.evaluation.metrics import (  # noqa: F401
    REFERENCE_GROUND_TRUTH,
    REFERENCE_ORIGINAL,
    MetricResult,
    cap_psnr,
    compare_images,
    compute_psnr,
    compute_ssim,
    format_psnr,
)

# Note: src.evaluation.comparison imports the pipelines; import it explicitly
# (``from src.evaluation.comparison import compare_pipelines``) to avoid import cycles.
