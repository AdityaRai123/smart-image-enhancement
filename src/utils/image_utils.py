"""
Stage 1 - Image ingestion and shared image helpers.

Internal convention used everywhere in the project:
    * images are NumPy arrays of dtype ``uint8``, shape ``(H, W, 3)``, channel order **RGB**
      (the order used by Pillow and Streamlit). OpenCV calls use explicit RGB conversion
      codes (``cv2.COLOR_RGB2...``), so no BGR/RGB confusion can occur.
    * grayscale / RGBA / 16-bit inputs are normalised into that representation at the
      ingestion boundary, and what was done is recorded in ``IngestedImage.notes``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import INGESTION, IngestionSettings

logger = logging.getLogger(__name__)


class ImageLoadError(ValueError):
    """Raised when an input cannot be turned into a valid working image.

    The message is written for end users and is displayed directly in the UI.
    """


@dataclass
class IngestedImage:
    """Result of Stage 1: a validated working image plus provenance information."""

    image: np.ndarray
    original_width: int
    original_height: int
    source_format: str | None = None
    source_mode: str | None = None
    filename: str | None = None
    file_size_bytes: int | None = None
    was_resized: bool = False
    had_alpha: bool = False
    is_grayscale: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


# --------------------------------------------------------------------------- #
# Array validation / conversion
# --------------------------------------------------------------------------- #
def ensure_rgb_uint8(
    array: np.ndarray,
    background_rgb: tuple[int, int, int] = INGESTION.alpha_background_rgb,
) -> np.ndarray:
    """Validate an arbitrary image array and convert it to RGB ``uint8`` ``(H, W, 3)``.

    Accepts 2-D grayscale, ``(H, W, 1)``, RGB, RGBA, boolean, float (0-1 or 0-255) and
    16-bit arrays. Raises :class:`ImageLoadError` for anything that is not an image.
    """
    if array is None:
        raise ImageLoadError("No image data was provided.")
    arr = np.asarray(array)
    if arr.size == 0:
        raise ImageLoadError("The image is empty (0 pixels).")
    if arr.ndim not in (2, 3):
        raise ImageLoadError(f"Expected a 2-D or 3-D image array, got {arr.ndim} dimensions.")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ImageLoadError(f"Unsupported number of channels: {arr.shape[2]}.")

    # --- dtype normalisation -------------------------------------------------------
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ImageLoadError("The image contains NaN or infinite values.")
        scale = 255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0
        arr = np.clip(np.rint(arr * scale), 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = (arr.astype(np.uint32) // 257).astype(np.uint8)
    elif arr.dtype != np.uint8:
        if not np.issubdtype(arr.dtype, np.integer):
            raise ImageLoadError(f"Unsupported pixel data type: {arr.dtype}.")
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # --- channel normalisation -----------------------------------------------------
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 4:
        arr = composite_alpha(arr, background_rgb)
    return np.ascontiguousarray(arr)


def composite_alpha(rgba: np.ndarray, background_rgb: tuple[int, int, int]) -> np.ndarray:
    """Alpha-composite an RGBA ``uint8`` image onto a solid background colour."""
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    background = np.array(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    out = rgb * alpha + background * (1.0 - alpha)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def is_grayscale_content(rgb: np.ndarray, tolerance: int = 2) -> bool:
    """True if all three channels are (almost) identical, i.e. the image is grey."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return bool(np.abs(r - g).max() <= tolerance and np.abs(g - b).max() <= tolerance)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma (same weights as the Y channel of YCrCb)."""
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def split_luminance(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Y, YCrCb)`` so the luminance can be processed independently of colour."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return ycrcb[..., 0].copy(), ycrcb


def merge_luminance(new_y: np.ndarray, ycrcb: np.ndarray) -> np.ndarray:
    """Replace the Y channel of a YCrCb image and convert back to RGB."""
    out = ycrcb.copy()
    out[..., 0] = np.clip(new_y, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)


def resize_max_side(
    image: np.ndarray, max_side: int, interpolation: int = cv2.INTER_AREA
) -> tuple[np.ndarray, float]:
    """Downscale so the longer side is at most ``max_side``. Returns ``(image, scale)``."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, new_size, interpolation=interpolation), scale


def resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize ``image`` to ``(height, width)``: INTER_AREA to shrink, INTER_CUBIC to enlarge."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = image.shape[:2]
    if (h, w) == (th, tw):
        return image
    interpolation = cv2.INTER_AREA if (th * tw) < (h * w) else cv2.INTER_CUBIC
    return cv2.resize(image, (tw, th), interpolation=interpolation)


# --------------------------------------------------------------------------- #
# Pillow -> array
# --------------------------------------------------------------------------- #
def _pil_to_rgb_array(pil: Image.Image, background_rgb: tuple[int, int, int]) -> tuple[np.ndarray, bool]:
    """Convert any Pillow mode to RGB uint8. Returns ``(array, had_alpha)``."""
    mode = pil.mode
    has_alpha_channel = mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in pil.info)

    if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        arr = np.asarray(pil, dtype=np.uint16)
        return ensure_rgb_uint8(arr), False
    if mode in ("I", "F"):
        arr = np.asarray(pil).astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo)
        return ensure_rgb_uint8(arr), False
    if has_alpha_channel:
        rgba = np.asarray(pil.convert("RGBA"))
        # Only report "transparency" when some pixel is actually not fully opaque.
        return composite_alpha(rgba, background_rgb), bool(rgba[..., 3].min() < 255)
    if mode in ("1", "L"):
        return ensure_rgb_uint8(np.asarray(pil.convert("L"))), False
    return np.asarray(pil.convert("RGB"), dtype=np.uint8).copy(), False


def ingest_array(
    array: np.ndarray,
    *,
    filename: str | None = None,
    source_format: str | None = None,
    source_mode: str | None = None,
    settings: IngestionSettings = INGESTION,
) -> IngestedImage:
    """Validate an in-memory array and apply the Stage-1 size policy."""
    rgb = ensure_rgb_uint8(array, settings.alpha_background_rgb)
    return _finalise(rgb, filename, source_format, source_mode, None, False, settings)


def _finalise(
    rgb: np.ndarray,
    filename: str | None,
    source_format: str | None,
    source_mode: str | None,
    file_size: int | None,
    had_alpha: bool,
    settings: IngestionSettings,
) -> IngestedImage:
    h, w = rgb.shape[:2]
    if min(h, w) < settings.min_side:
        raise ImageLoadError(
            f"The image is too small ({w}x{h} px). At least {settings.min_side} px per side "
            "is required for meaningful quality analysis."
        )
    notes: list[str] = []
    if had_alpha:
        notes.append(
            "Transparency was flattened onto a white background (alpha is not part of the analysis)."
        )
    grayscale = is_grayscale_content(rgb)
    if grayscale:
        notes.append("Grayscale image detected: processed as 3 identical channels.")

    working, scale = resize_max_side(rgb, settings.processing_max_side)
    resized = scale < 1.0
    if resized:
        notes.append(
            f"Large image downscaled from {w}x{h} to {working.shape[1]}x{working.shape[0]} "
            f"(longest side {settings.processing_max_side} px) for CPU-friendly processing."
        )
    return IngestedImage(
        image=np.ascontiguousarray(working),
        original_width=w,
        original_height=h,
        source_format=source_format,
        source_mode=source_mode,
        filename=filename,
        file_size_bytes=file_size,
        was_resized=resized,
        had_alpha=had_alpha,
        is_grayscale=grayscale,
        notes=notes,
    )


def load_image_bytes(
    data: bytes,
    filename: str | None = None,
    settings: IngestionSettings = INGESTION,
) -> IngestedImage:
    """Decode an uploaded file into a validated :class:`IngestedImage`.

    Handles corrupted/truncated files, unsupported formats, decompression bombs,
    EXIF orientation, transparency, palette, 16-bit and grayscale images.
    """
    if not data:
        raise ImageLoadError("The uploaded file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_megabytes:
        raise ImageLoadError(
            f"The file is {size_mb:.1f} MB; the limit is {settings.max_upload_megabytes:.0f} MB."
        )
    if filename:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext and ext not in settings.allowed_extensions:
            raise ImageLoadError(
                f"Unsupported file type '.{ext}'. Supported: "
                + ", ".join(sorted(set(settings.allowed_extensions)))
            )

    try:
        pil = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise ImageLoadError("The file is not a readable image (corrupted or unsupported format).") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ImageLoadError(f"Could not open the image: {exc}") from exc

    fmt = pil.format
    if fmt and fmt.upper() not in settings.allowed_formats:
        raise ImageLoadError(
            f"Image format '{fmt}' is not supported. Please upload JPEG, PNG or WebP."
        )
    w, h = pil.size
    if w * h > settings.max_decoded_pixels:
        raise ImageLoadError(
            f"The image is {w}x{h} ({w * h / 1e6:.0f} MP), above the "
            f"{settings.max_decoded_pixels / 1e6:.0f} MP safety limit."
        )
    try:
        pil.load()
        if getattr(pil, "n_frames", 1) > 1:
            pil.seek(0)
        pil = ImageOps.exif_transpose(pil)
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("The image is too large to decode safely.") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageLoadError(f"The image file is corrupted or truncated ({exc}).") from exc

    mode = pil.mode
    rgb, had_alpha = _pil_to_rgb_array(pil, settings.alpha_background_rgb)
    result = _finalise(rgb, filename, fmt, mode, len(data), had_alpha, settings)
    if mode.startswith("I;16"):
        result.notes.append("16-bit image converted to 8-bit for processing.")
    logger.debug("Ingested %s (%s, %s) -> %sx%s", filename, fmt, mode, result.width, result.height)
    return result


def load_image_file(path: str | Path, settings: IngestionSettings = INGESTION) -> IngestedImage:
    """Load an image from disk (used by scripts and tests)."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"File not found: {path}")
    return load_image_bytes(path.read_bytes(), filename=path.name, settings=settings)


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image from disk as RGB uint8 without the Stage-1 resize policy."""
    with Image.open(path) as pil:
        pil.load()
        rgb, _ = _pil_to_rgb_array(ImageOps.exif_transpose(pil), INGESTION.alpha_background_rgb)
    return rgb


def save_image(rgb: np.ndarray, path: str | Path) -> Path:
    """Save an RGB uint8 image (format inferred from the extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_rgb_uint8(rgb)).save(path)
    return path


def to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB image as PNG bytes (lossless, for downloads)."""
    buffer = io.BytesIO()
    Image.fromarray(ensure_rgb_uint8(rgb)).save(buffer, format="PNG")
    return buffer.getvalue()
