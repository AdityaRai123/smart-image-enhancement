"""
CLAHE - Contrast Limited Adaptive Histogram Equalisation (Zuiderveld, 1994).

The image is divided into tiles, each tile's histogram is equalised independently, and
histogram bins are clipped at ``clip_limit`` (times the mean bin height) before
redistribution - which limits the noise amplification of plain adaptive HE. Tile
borders are bilinearly interpolated. Applied to the luminance (Y of YCrCb) only.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import ENHANCEMENT
from src.utils.image_utils import ensure_rgb_uint8, merge_luminance, split_luminance


def clahe(
    image: np.ndarray,
    clip_limit: float = ENHANCEMENT.clahe_clip_limit_mild,
    tile_grid: tuple[int, int] = ENHANCEMENT.clahe_tile_grid,
) -> np.ndarray:
    """Apply CLAHE to the luminance channel of an RGB image."""
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive.")
    rows, cols = int(tile_grid[0]), int(tile_grid[1])
    if rows < 1 or cols < 1:
        raise ValueError("tile_grid entries must be >= 1.")
    rgb = ensure_rgb_uint8(image)
    # Small images: never use tiles smaller than ~8 px.
    h, w = rgb.shape[:2]
    rows, cols = max(1, min(rows, h // 8)), max(1, min(cols, w // 8))
    operator = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(cols, rows))
    y, ycrcb = split_luminance(rgb)
    return merge_luminance(operator.apply(y), ycrcb)
