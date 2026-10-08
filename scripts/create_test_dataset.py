"""
Create a controlled evaluation set by degrading clean images.

For every clean image the script stores the clean version as the ground-truth reference
and one degraded copy per category:

    data/test/
        reference/<name>.png        clean ground truth
        good/<name>.png             untouched (checks for unnecessary processing)
        low_contrast/<name>.png
        noisy/<name>.png
        impulse_noise/<name>.png
        blurry/<name>.png
        low_resolution/<name>.png   4x smaller than the reference
        mixed/<name>.png
        manifest.csv                image, category, paths, recipe, expected defects

Clean images are read from ``data/clean/`` (put your own photos or a DIV2K subset
there). If that folder is empty, the built-in scikit-image samples are exported into it
first, so the dataset is reproducible offline.

Usage:
    python scripts/create_test_dataset.py
    python scripts/create_test_dataset.py --clean-dir path/to/clean --output data/test --seed 0
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import _bootstrap  # noqa: F401  (adds the project root to sys.path)

from config import APP
from src.utils.degradation import DEGRADATIONS, apply_degradation, crop_to_multiple
from src.utils.image_utils import ImageLoadError, read_rgb, resize_max_side, save_image
from src.utils.samples import load_sample

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
logger = logging.getLogger("create_test_dataset")


def export_samples(clean_dir: Path) -> list[Path]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in APP.sample_images:
        try:
            paths.append(save_image(load_sample(name), clean_dir / f"{name}.png"))
        except ImageLoadError as exc:
            logger.warning("Skipping sample %s: %s", name, exc)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", type=Path, default=APP.data_dir / "clean")
    parser.add_argument("--output", type=Path, default=APP.data_dir / "test")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the noise degradations.")
    parser.add_argument("--max-side", type=int, default=1024,
                        help="Clean images larger than this are downscaled first (keeps evaluation fast).")
    parser.add_argument("--categories", nargs="*", default=list(DEGRADATIONS), choices=list(DEGRADATIONS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    clean_paths = sorted(p for p in args.clean_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) \
        if args.clean_dir.exists() else []
    if not clean_paths:
        logger.info("No clean images in %s - exporting the built-in scikit-image samples.", args.clean_dir)
        clean_paths = export_samples(args.clean_dir)
    if not clean_paths:
        raise SystemExit("No clean images available.")

    out = args.output
    (out / "reference").mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, path in enumerate(clean_paths):
        try:
            clean = read_rgb(path)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            continue
        clean, _ = resize_max_side(clean, args.max_side)
        clean = crop_to_multiple(clean, 4)  # x4 downsampling / SR gives back the exact size
        stem = path.stem
        reference_path = save_image(clean, out / "reference" / f"{stem}.png")
        for category in args.categories:
            spec = DEGRADATIONS[category]
            degraded = apply_degradation(clean, spec, seed=args.seed + index)
            target = save_image(degraded, out / category / f"{stem}.png")
            manifest_rows.append({
                "image": stem,
                "category": category,
                "path": target.relative_to(out).as_posix(),
                "reference": reference_path.relative_to(out).as_posix(),
                "description": spec.description,
                "expected_defects": ";".join(spec.expected),
                "width": degraded.shape[1],
                "height": degraded.shape[0],
            })
        logger.info("%-24s -> %d categories", stem, len(args.categories))

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    logger.info("Wrote %d degraded images + %d references to %s", len(manifest_rows), len(clean_paths), out)


if __name__ == "__main__":
    main()
