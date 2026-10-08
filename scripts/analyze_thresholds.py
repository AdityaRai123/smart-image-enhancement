"""
Threshold calibration helper.

Runs Stage-2 analysis on every image of the evaluation set and reports, per degradation
category, the distribution of each statistic that a decision rule depends on, next to
the current threshold. This is the "refine and validate the thresholds against a small
test set" step promised in the Review 1 report.

Outputs (default ``outputs/threshold_analysis``):
    features.csv            all statistics of every image
    threshold_report.md     min / median / max per category + separation margins
    <statistic>.png         strip plots with the threshold drawn as a line

Usage:
    python scripts/create_test_dataset.py      # once
    python scripts/analyze_thresholds.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401  (adds the project root to sys.path)

import numpy as np
import pandas as pd

from config import APP, DECISION
from src.analysis import analyze_image
from src.evaluation.dataset import discover_dataset, markdown_table
from src.utils.image_utils import load_image_file
from src.visualization import plots

# statistic -> (threshold attribute, flagged when value is below/above, log scale)
CHECKS = {
    "histogram_spread": ("contrast_low_spread", "below", False),
    "blur_score": ("blur_threshold", "below", True),
    "noise_score": ("noise_medium_sigma", "above", False),
    "impulse_ratio": ("impulse_ratio_threshold", "above", True),
}
logger = logging.getLogger("analyze_thresholds")


def strip_plot(df: pd.DataFrame, column: str, threshold: float, log: bool, path: Path) -> None:
    categories = list(dict.fromkeys(df["category"]))
    fig, (ax,) = plots._figure(7.5, 3.0)
    rng = np.random.default_rng(0)
    for i, cat in enumerate(categories):
        values = df.loc[df["category"] == cat, column].to_numpy(dtype=float)
        if log:
            values = np.maximum(values, 1e-4)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=36, color=plots.SERIES["original"],
                   edgecolor=plots.SURFACE, linewidth=1.5, zorder=3)
    ax.axhline(threshold, color=plots.SERIES["fixed"], linewidth=1.5, label=f"threshold = {threshold:g}")
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(categories)), [c.replace("_", " ") for c in categories], fontsize=8)
    plots._style(ax, column.replace("_", " "), None, column)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plots.plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=APP.data_dir / "test")
    parser.add_argument("--output", type=Path, default=APP.outputs_dir / "threshold_analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.data_dir.is_dir():
        raise SystemExit(f"{args.data_dir} does not exist. Run: python scripts/create_test_dataset.py")

    records = []
    for item in discover_dataset(args.data_dir):
        try:
            features = analyze_image(load_image_file(item["path"]).image).features()
        except Exception as exc:
            logger.error("Failed on %s: %s", item["path"], exc)
            continue
        records.append({"category": item["category"], "image": item["image"], **features})
    df = pd.DataFrame(records)
    df["rms_after_stretch"] = df["contrast_score"] / df["histogram_spread"].clip(lower=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "features.csv", index=False)

    lines = ["# Threshold analysis", "", f"{len(df)} images from `{args.data_dir}`.", ""]
    for column, (attr, direction, log) in CHECKS.items():
        threshold = getattr(DECISION, attr)
        stats = df.groupby("category")[column].agg(["min", "median", "max"]).reset_index()
        flagged = df.groupby("category")[column].apply(
            lambda s: float(np.mean(s < threshold if direction == "below" else s > threshold))
        ).rename("flagged_share").reset_index()
        table = stats.merge(flagged, on="category")
        lines += [f"## `{column}` - flagged when {direction} `{attr}` = {threshold:g}", "",
                  markdown_table(table, "{:.4g}"), ""]
        strip_plot(df, column, threshold, log, args.output / f"{column}.png")
    (args.output / "threshold_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    logger.info("Written to %s", args.output)


if __name__ == "__main__":
    main()
