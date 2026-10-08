"""
Evaluate (A) no enhancement, (B) the fixed classical pipeline and (C) the adaptive
pipeline on a folder of test images, as required by the report's Expected Outcome.

Inputs  (default ``data/test``, created by ``scripts/create_test_dataset.py``):
    <category>/<image>.png  + optional reference/<image>.png  + optional manifest.csv
    Any folder of images works: without a reference the metrics are computed against
    the input itself (reference_kind = "original"), as in the report.

Outputs (default ``outputs/evaluation``):
    results.csv     one row per image x pipeline (PSNR, SSIM, time, operations, ...)
    summary.csv     mean PSNR / SSIM / time per category x pipeline
    detection.csv   diagnosis accuracy of the decision engine (synthetic labels)
    summary.md      human-readable report of the above
    psnr_by_category.png, ssim_by_category.png

Usage:
    python scripts/evaluate_dataset.py
    python scripts/evaluate_dataset.py --sr off          # no super-resolution
    python scripts/evaluate_dataset.py --sr fallback     # Lanczos instead of Real-ESRGAN
"""

from __future__ import annotations

import argparse
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401  (adds the project root to sys.path)

import numpy as np
import pandas as pd

from config import APP, EVALUATION
from src.enhancement.super_resolution import InterpolationUpscaler, create_super_resolver
from src.evaluation.comparison import compare_pipelines
from src.evaluation.dataset import discover_dataset, markdown_table
from src.evaluation.metrics import cap_psnr
from src.pipeline import MODE_ADAPTIVE, MODE_FIXED, MODE_NONE
from src.utils.image_utils import load_image_file, read_rgb, save_image
from src.visualization import plots

ASPECT_TO_DEFECT = {"contrast": "contrast", "noise": "noise", "sharpness": "blur", "resolution": "resolution"}
DEFECTS = ("contrast", "noise", "blur", "resolution")
MODES = [MODE_NONE, MODE_FIXED, MODE_ADAPTIVE]
logger = logging.getLogger("evaluate_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=APP.data_dir / "test")
    parser.add_argument("--output", type=Path, default=APP.outputs_dir / "evaluation")
    parser.add_argument("--sr", choices=["auto", "ai", "fallback", "off"], default="auto",
                        help="Super-resolution for low-resolution images: auto = Real-ESRGAN if available, "
                             "otherwise Lanczos; ai = require Real-ESRGAN; off = never upscale.")
    parser.add_argument("--save-images", action="store_true", help="Also save every pipeline output image.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images (0 = all).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.data_dir.is_dir():
        raise SystemExit(f"{args.data_dir} does not exist. Run: python scripts/create_test_dataset.py")
    items = discover_dataset(args.data_dir)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No images found in {args.data_dir}.")

    resolver, sr_enabled, sr_label = None, args.sr != "off", "disabled"
    if args.sr in ("auto", "ai"):
        resolver, status = create_super_resolver(allow_download=True)
        if args.sr == "ai" and not status.ready:
            raise SystemExit(f"Real-ESRGAN required but unavailable: {status.message}")
        sr_label = resolver.name if resolver else "unavailable"
    elif args.sr == "fallback":
        resolver = InterpolationUpscaler()
        sr_label = resolver.name
    logger.info("Super-resolution: %s", sr_label)

    args.output.mkdir(parents=True, exist_ok=True)
    rows, detection_rows = [], []
    start_all = time.perf_counter()
    for i, item in enumerate(items, start=1):
        try:
            image = load_image_file(item["path"]).image
            reference = read_rgb(item["reference"]) if item.get("reference") else None
            comparison = compare_pipelines(
                image, reference=reference, sr_enabled=sr_enabled, super_resolver=resolver,
                analyze_outputs=True, keep_intermediates=False,
            )
        except Exception as exc:  # one bad file must not stop the evaluation
            logger.error("Failed on %s: %s", item["path"], exc)
            continue

        decision = comparison.results[MODE_ADAPTIVE].decision
        detected = {ASPECT_TO_DEFECT[k] for k, d in decision.diagnosis.items() if d.needs_action}
        for r in comparison.rows:
            aq = r.after_quality
            rows.append({
                "image": item["image"], "category": item["category"], "pipeline": r.mode,
                "operations": " -> ".join(r.operations) or "(none)",
                "psnr_db": r.metrics.psnr, "psnr_capped_db": cap_psnr(r.metrics.psnr), "ssim": r.metrics.ssim,
                "reference_kind": comparison.reference_kind, "time_ms": r.time_ms,
                "output_width": r.output_shape[1], "output_height": r.output_shape[0],
                "resized_for_metric": r.metrics.resized_for_comparison,
                "blur_score_after": aq.sharpness.blur_score if aq else np.nan,
                "noise_sigma_after": aq.noise.sigma if aq else np.nan,
                "spread_after": aq.contrast.histogram_spread if aq else np.nan,
            })
            if args.save_images:
                save_image(comparison.results[r.mode].output,
                           args.output / "images" / item["category"] / f"{item['image']}_{r.mode}.png")
        if item.get("labelled"):
            expected = item["expected_defects"]
            detection_rows.append({
                "image": item["image"], "category": item["category"],
                "expected": ";".join(sorted(expected)) or "(none)",
                "detected": ";".join(sorted(detected)) or "(none)",
                "exact_match": detected == expected,
                "missed": ";".join(sorted(expected - detected)),
                "false_alarms": ";".join(sorted(detected - expected)),
                "adaptive_operations": " -> ".join(op.key for op in decision.operations) or "(none)",
            })
        logger.info("[%d/%d] %s/%s -> %s", i, len(items), item["category"], item["image"],
                    " -> ".join(op.key for op in decision.operations) or "no enhancement")
    elapsed = time.perf_counter() - start_all

    results = pd.DataFrame(rows)
    results.to_csv(args.output / "results.csv", index=False)
    summary = (results.groupby(["category", "pipeline"])
               .agg(psnr_db=("psnr_capped_db", "mean"), ssim=("ssim", "mean"), time_ms=("time_ms", "mean"),
                    images=("image", "nunique"))
               .reindex(MODES, level="pipeline"))
    summary.to_csv(args.output / "summary.csv")
    overall = results.groupby("pipeline").agg(psnr_db=("psnr_capped_db", "mean"), ssim=("ssim", "mean"),
                                               time_ms=("time_ms", "mean")).reindex(MODES)

    # Adaptive vs fixed head-to-head on SSIM.
    pivot = results.pivot_table(index=["category", "image"], columns="pipeline", values="ssim")
    wins = (pivot[MODE_ADAPTIVE] >= pivot[MODE_FIXED]).groupby(level=0).mean() if not pivot.empty else pd.Series()

    detection = pd.DataFrame(detection_rows)
    if not detection.empty:
        detection.to_csv(args.output / "detection.csv", index=False)

    # ---- plots ---------------------------------------------------------------------
    labels = {MODE_NONE: "No enhancement", MODE_FIXED: "Fixed pipeline", MODE_ADAPTIVE: "Adaptive (proposed)"}
    for metric, title, ylabel in (("psnr_db", "Mean PSNR by degradation category", "PSNR (dB, capped at "
                                   f"{EVALUATION.psnr_cap_db:g})"),
                                  ("ssim", "Mean SSIM by degradation category", "SSIM")):
        fig = plots.category_comparison(summary, metric, title, ylabel, MODES, labels)
        fig.savefig(args.output / f"{metric.split('_')[0]}_by_category.png", dpi=130, bbox_inches="tight")
        plots.plt.close(fig)

    # ---- markdown report --------------------------------------------------------------
    reference_kinds = sorted(results["reference_kind"].unique())
    lines = [
        "# Evaluation results",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/evaluate_dataset.py` on {platform.processor() or platform.machine()} "
        f"(CPU), Python {platform.python_version()}.",
        "",
        f"* Data: `{args.data_dir}` - {results['image'].nunique()} images x {results['category'].nunique()} categories "
        f"({len(items)} test files). **Synthetic degradations** produced by `scripts/create_test_dataset.py` "
        "from clean scikit-image sample photographs unless you supplied your own data.",
        f"* Metrics computed against: {', '.join(reference_kinds)} "
        "(ground_truth = the clean image before degradation).",
        f"* Super-resolution for low-resolution inputs: {sr_label}.",
        f"* PSNR of identical images is infinite; it is capped at {EVALUATION.psnr_cap_db:g} dB for averaging.",
        "* Outputs smaller than the reference (e.g. no upscaling of a low-resolution input) are resampled "
        "bicubically to the reference size before scoring, so 'no enhancement' = bicubic upscaling there.",
        f"* Total evaluation time: {elapsed:.1f} s.",
        "",
        "## Mean PSNR / SSIM per category",
        "",
        markdown_table(summary.reset_index().assign(pipeline=lambda d: d["pipeline"].map(labels))),
        "",
        "## Overall mean",
        "",
        markdown_table(overall.reset_index().assign(pipeline=lambda d: d["pipeline"].map(labels))),
        "",
        "## Adaptive vs fixed: share of images where adaptive SSIM >= fixed SSIM",
        "",
        markdown_table(wins.rename("adaptive_ssim_ge_fixed").reset_index()) if len(wins) else "(n/a)",
        "",
    ]
    if not detection.empty:
        per_cat = detection.groupby("category").agg(images=("image", "count"), exact_match=("exact_match", "mean"))
        lines += [
            "## Decision-engine diagnosis accuracy (synthetic labels)",
            "",
            "`exact_match` = share of images where the set of detected defects equals the injected set "
            "(for `good`, detecting nothing). Details per image in `detection.csv`.",
            "",
            markdown_table(per_cat.reset_index()),
            "",
        ]
        misses = detection[~detection["exact_match"]]
        if not misses.empty:
            lines += ["### Mismatches", "", markdown_table(misses[["category", "image", "expected", "detected"]]), ""]
    lines += [
        "## Reading these numbers",
        "",
        "* PSNR/SSIM reward pixel fidelity to the reference. GAN-based super-resolution (Real-ESRGAN) synthesises "
        "plausible texture that is not pixel-identical to the ground truth, so it typically scores *below* bicubic "
        "interpolation on PSNR/SSIM while looking sharper (perception-distortion trade-off, Blau & Michaeli, CVPR 2018). "
        "Compare `blur_score_after` in results.csv for the sharpness gain.",
        "* For the `good` category the adaptive pipeline is expected to do nothing; any processing there is a false alarm.",
    ]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines[:]) + "\n")
    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
