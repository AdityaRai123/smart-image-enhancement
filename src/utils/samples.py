"""Built-in clean sample images (bundled with scikit-image, no download needed)."""

from __future__ import annotations

import logging

import numpy as np

from src.utils.image_utils import ImageLoadError, ensure_rgb_uint8

logger = logging.getLogger(__name__)

# Human-readable descriptions shown in the UI.
SAMPLE_DESCRIPTIONS: dict[str, str] = {
    "astronaut": "Astronaut portrait (512x512, colour)",
    "coffee": "Coffee cup (600x400, colour)",
    "chelsea": "Chelsea the cat (451x300, colour)",
    "rocket": "Rocket launch (640x427, colour)",
    "camera": "Cameraman (512x512, grayscale)",
    "immunohistochemistry": "Microscopy slide (512x512, colour)",
}


def load_sample(name: str) -> np.ndarray:
    """Return a scikit-image sample image as RGB uint8."""
    try:
        from skimage import data as skdata

        loader = getattr(skdata, name)
        return ensure_rgb_uint8(loader())
    except AttributeError as exc:
        raise ImageLoadError(f"Unknown sample image '{name}'.") from exc
    except Exception as exc:  # e.g. sample needs a network download
        logger.warning("Sample %s unavailable: %s", name, exc)
        raise ImageLoadError(f"Sample image '{name}' is not available offline: {exc}") from exc
