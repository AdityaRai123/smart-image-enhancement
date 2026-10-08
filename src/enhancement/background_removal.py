"""
Optional background removal with ``rembg`` (listed as an optional AI tool in the report).

This is an *extra* utility, not part of the adaptive enhancement pipeline: background
removal changes image content rather than restoring quality, so the decision engine
never selects it. It is exposed in the UI only when ``rembg`` is installed.
"""

from __future__ import annotations

import importlib.util
import logging

import numpy as np

from src.utils.image_utils import ensure_rgb_uint8

logger = logging.getLogger(__name__)


class BackgroundRemovalUnavailable(RuntimeError):
    pass


def rembg_available() -> bool:
    return importlib.util.find_spec("rembg") is not None


def remove_background(image: np.ndarray) -> np.ndarray:
    """Return an RGBA image whose alpha channel masks out the background."""
    if not rembg_available():
        raise BackgroundRemovalUnavailable(
            "rembg is not installed. Install it with: pip install rembg  (downloads a ~170 MB model on first use)"
        )
    try:
        from rembg import remove
        from PIL import Image

        result = remove(Image.fromarray(ensure_rgb_uint8(image)))
        return np.asarray(result.convert("RGBA"))
    except Exception as exc:
        logger.warning("Background removal failed: %s", exc)
        raise BackgroundRemovalUnavailable(f"Background removal failed: {exc}") from exc
