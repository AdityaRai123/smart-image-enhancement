"""Evaluation-set discovery and report helpers shared by the scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def discover_dataset(data_dir: Path) -> list[dict]:
    """List test images with category, reference path and expected defects.

    Uses ``manifest.csv`` (written by ``scripts/create_test_dataset.py``) when present;
    otherwise every image in ``<data_dir>/<category>/`` is used, with
    ``<data_dir>/reference/<same name>`` as ground truth if it exists.
    """
    data_dir = Path(data_dir)
    manifest = data_dir / "manifest.csv"
    if manifest.is_file():
        with open(manifest, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["path"] = data_dir / row["path"]
            row["reference"] = data_dir / row["reference"] if row.get("reference") else None
            row["expected_defects"] = set(filter(None, (row.get("expected_defects") or "").split(";")))
            row["labelled"] = True
        return rows
    items = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != "reference"):
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                ref = data_dir / "reference" / path.name
                items.append({
                    "image": path.stem, "category": folder.name, "path": path,
                    "reference": ref if ref.is_file() else None, "expected_defects": set(), "labelled": False,
                })
    return items


def markdown_table(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""

    def fmt(value) -> str:
        if isinstance(value, float):
            return "n/a" if math.isnan(value) else float_format.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *body])
