"""Shared test fixtures: small deterministic synthetic images (no files, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_scene(size: int = 192, seed: int = 0) -> np.ndarray:
    """A sharp, full-contrast RGB test scene: gradient + shapes + fine texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 30 + 190 * x / (size - 1)
    img = np.stack([base, base * 0.8 + 30, 255 - base], axis=-1)
    img[size // 4: size // 2, size // 4: size // 2] = (240, 30, 30)
    img[size // 2: 3 * size // 4, size // 2: 3 * size // 4] = (10, 10, 10)
    checker = ((x // 6 + y // 6) % 2 == 0) & (y > 0.8 * size)
    img = np.where(checker[..., None], 250.0, img)
    img += rng.normal(0, 0.5, img.shape)  # tiny realistic grain
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def scene() -> np.ndarray:
    return make_scene()


@pytest.fixture(scope="session")
def natural() -> np.ndarray:
    """A real photograph (scikit-image sample, bundled offline)."""
    from src.utils.degradation import crop_to_multiple
    from src.utils.samples import load_sample

    return crop_to_multiple(load_sample("astronaut"), 4)
