"""
Optional AI super-resolution with Real-ESRGAN, plus a classical fallback.

Design goals (from the Review 1 report):
    * **Optional.** The application never requires PyTorch or a GPU. ``torch`` is imported
      lazily; if it is missing, or the model cannot be downloaded/loaded, the caller gets
      a clear status message and (optionally) a clearly labelled Lanczos fallback.
    * **Quality-aware.** The decision engine only routes images that were flagged as
      low-resolution here; it is never applied to every image.
    * **CPU-first.** Inputs above ``max_input_pixels`` are downscaled before inference
      ("applied on a downscaled crop to keep inference time reasonable on CPU"), the
      lightweight ``realesr-general-x4v3`` model is the default, and inference is tiled to
      bound memory.

Usage::

    resolver, status = create_super_resolver(allow_download=True)
    result = resolver.upscale(rgb)          # SRResult(image, method, is_ai, ...)
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from config import SUPER_RESOLUTION, SuperResolutionSettings
from config.config import SRModelSpec
from src.utils.image_utils import ensure_rgb_uint8, resize_max_side

logger = logging.getLogger(__name__)


class SRUnavailableError(RuntimeError):
    """Real-ESRGAN cannot be used (dependency, download or load failure)."""


@dataclass
class SRResult:
    image: np.ndarray
    method: str  # human-readable method name
    is_ai: bool
    scale: float  # effective output/input size ratio
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)


@dataclass
class SRStatus:
    """What the UI shows about the optional AI stage."""

    torch_installed: bool
    torch_version: str | None
    model_name: str
    model_file_present: bool
    model_path: Path
    ready: bool  # model loaded successfully
    message: str


class SuperResolver(Protocol):
    name: str
    is_ai: bool

    def upscale(self, image: np.ndarray) -> SRResult: ...


# --------------------------------------------------------------------------- #
# Availability checks
# --------------------------------------------------------------------------- #
def torch_available() -> bool:
    """True if PyTorch can be imported (checked without importing it)."""
    return importlib.util.find_spec("torch") is not None


def torch_version() -> str | None:
    if not torch_available():
        return None
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # pragma: no cover - broken installation
        return None


def model_path(spec: SRModelSpec, settings: SuperResolutionSettings = SUPER_RESOLUTION) -> Path:
    return Path(settings.model_dir) / f"{spec.name}.pth"


def get_model_spec(model_name: str | None, settings: SuperResolutionSettings = SUPER_RESOLUTION) -> SRModelSpec:
    name = model_name or settings.default_model
    if name not in settings.models:
        raise SRUnavailableError(f"Unknown Real-ESRGAN model '{name}'. Options: {list(settings.models)}")
    return settings.models[name]


# --------------------------------------------------------------------------- #
# Classical fallback
# --------------------------------------------------------------------------- #
class InterpolationUpscaler:
    """Lanczos-4 interpolation. Not AI - used only when Real-ESRGAN is unavailable."""

    is_ai = False

    def __init__(self, scale: int = SUPER_RESOLUTION.fallback_scale,
                 output_max_side: int = SUPER_RESOLUTION.output_max_side) -> None:
        self.scale = int(scale)
        self.output_max_side = int(output_max_side)
        self.name = f"Lanczos x{self.scale} interpolation (classical fallback, not AI)"

    def upscale(self, image: np.ndarray) -> SRResult:
        start = time.perf_counter()
        rgb = ensure_rgb_uint8(image)
        h, w = rgb.shape[:2]
        target = (w * self.scale, h * self.scale)
        out = cv2.resize(rgb, target, interpolation=cv2.INTER_LANCZOS4)
        out, _ = resize_max_side(out, self.output_max_side)
        return SRResult(
            image=out,
            method=self.name,
            is_ai=False,
            scale=out.shape[1] / float(w),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            notes=["Real-ESRGAN unavailable: classical Lanczos interpolation used instead."],
        )


# --------------------------------------------------------------------------- #
# Real-ESRGAN
# --------------------------------------------------------------------------- #
def download_model(spec: SRModelSpec, settings: SuperResolutionSettings = SUPER_RESOLUTION) -> Path:
    """Download the official checkpoint into ``settings.model_dir`` (atomic write)."""
    target = model_path(spec, settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s", spec.name, spec.url)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp, urllib.request.urlopen(
            spec.url, timeout=settings.download_timeout_seconds
        ) as response:
            while chunk := response.read(1 << 20):
                tmp.write(chunk)
        if Path(tmp_name).stat().st_size < 100_000:
            raise SRUnavailableError("Downloaded model file is unexpectedly small (incomplete download?).")
        os.replace(tmp_name, target)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SRUnavailableError(
            f"Could not download '{spec.name}' ({exc}). Check the internet connection or "
            f"download it manually from {spec.url} and place it at {target}."
        ) from exc
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return target


class RealESRGANUpscaler:
    """Real-ESRGAN inference on CPU (or CUDA if available) with tiling and an input budget."""

    is_ai = True

    def __init__(self, spec: SRModelSpec, network, device, settings: SuperResolutionSettings) -> None:
        self.spec = spec
        self.network = network
        self.device = device
        self.settings = settings
        self.name = f"Real-ESRGAN ({spec.name}, x{spec.scale})"

    @classmethod
    def load(
        cls,
        model_name: str | None = None,
        allow_download: bool = True,
        settings: SuperResolutionSettings = SUPER_RESOLUTION,
    ) -> "RealESRGANUpscaler":
        """Build the network and load pretrained weights. Raises :class:`SRUnavailableError`."""
        if not torch_available():
            raise SRUnavailableError(
                "PyTorch is not installed, so Real-ESRGAN is unavailable. Install the optional AI "
                "dependencies with: pip install -r requirements-ai.txt"
            )
        spec = get_model_spec(model_name, settings)
        path = model_path(spec, settings)
        if not path.is_file():
            if not allow_download:
                raise SRUnavailableError(
                    f"Model file not found at {path}. Enable 'Allow model download' or download it "
                    f"manually from {spec.url}."
                )
            download_model(spec, settings)

        try:
            import torch

            from src.enhancement.realesrgan_arch import build_network

            network = build_network(spec.architecture, spec.scale, spec.num_feat, spec.num_conv, spec.num_block)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict):
                state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
            else:
                raise SRUnavailableError("Unexpected checkpoint format.")
            network.load_state_dict(state, strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            network.eval().to(device)
        except SRUnavailableError:
            raise
        except Exception as exc:
            raise SRUnavailableError(
                f"Failed to load Real-ESRGAN weights from {path}: {exc}. Delete the file to re-download."
            ) from exc
        logger.info("Loaded %s on %s", spec.name, device)
        return cls(spec, network, device, settings)

    def _run_tiled(self, tensor):
        """Tiled inference (bounded memory). ``tensor`` is 1x3xHxW in [0, 1]."""
        import torch

        scale = self.spec.scale
        tile, pad = self.settings.tile_size, self.settings.tile_pad
        _, channels, height, width = tensor.shape
        if tile <= 0 or (height <= tile and width <= tile):
            return self.network(tensor)
        output = tensor.new_zeros((1, channels, height * scale, width * scale))
        for y0 in range(0, height, tile):
            for x0 in range(0, width, tile):
                y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, height), min(x1 + pad, width)
                with torch.no_grad():
                    out_tile = self.network(tensor[:, :, py0:py1, px0:px1])
                oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
                output[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = out_tile[
                    :, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale
                ]
        return output

    def upscale(self, image: np.ndarray) -> SRResult:
        import torch

        start = time.perf_counter()
        rgb = ensure_rgb_uint8(image)
        in_h, in_w = rgb.shape[:2]
        notes: list[str] = []

        # CPU guard: bound the number of input pixels.
        budget = self.settings.max_input_pixels
        if in_h * in_w > budget:
            factor = (budget / float(in_h * in_w)) ** 0.5
            new_size = (max(1, int(in_w * factor)), max(1, int(in_h * factor)))
            rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
            notes.append(
                f"Input downscaled from {in_w}x{in_h} to {new_size[0]}x{new_size[1]} before AI "
                f"super-resolution to keep CPU inference time bounded."
            )

        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._run_tiled(tensor)
        out = output.squeeze(0).clamp_(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        out = np.rint(out * 255.0).astype(np.uint8)

        out, shrink = resize_max_side(out, self.settings.output_max_side)
        if shrink < 1.0:
            notes.append(f"Output limited to {self.settings.output_max_side} px on the longer side.")
        elapsed = (time.perf_counter() - start) * 1000.0
        return SRResult(
            image=out,
            method=self.name,
            is_ai=True,
            scale=out.shape[1] / float(in_w),
            elapsed_ms=elapsed,
            notes=notes,
        )


# --------------------------------------------------------------------------- #
# Factory used by the app and scripts
# --------------------------------------------------------------------------- #
def create_super_resolver(
    model_name: str | None = None,
    allow_download: bool = True,
    settings: SuperResolutionSettings = SUPER_RESOLUTION,
) -> tuple[SuperResolver | None, SRStatus]:
    """Try to load Real-ESRGAN; fall back to interpolation (if enabled) on any failure.

    Never raises: failures are reported through the returned :class:`SRStatus`.
    """
    spec_name = model_name or settings.default_model
    try:
        spec = get_model_spec(spec_name, settings)
        path = model_path(spec, settings)
    except SRUnavailableError as exc:
        return (InterpolationUpscaler() if settings.fallback_to_interpolation else None), SRStatus(
            torch_available(), torch_version(), spec_name, False, Path(), False, str(exc)
        )

    try:
        resolver: SuperResolver | None = RealESRGANUpscaler.load(spec_name, allow_download, settings)
        status = SRStatus(
            torch_installed=True,
            torch_version=torch_version(),
            model_name=spec.name,
            model_file_present=True,
            model_path=path,
            ready=True,
            message=f"Real-ESRGAN ready ({spec.description}).",
        )
    except SRUnavailableError as exc:
        logger.warning("Real-ESRGAN unavailable: %s", exc)
        resolver = InterpolationUpscaler() if settings.fallback_to_interpolation else None
        status = SRStatus(
            torch_installed=torch_available(),
            torch_version=torch_version(),
            model_name=spec.name,
            model_file_present=path.is_file(),
            model_path=path,
            ready=False,
            message=str(exc),
        )
    return resolver, status


def probe_sr_status(model_name: str | None = None, settings: SuperResolutionSettings = SUPER_RESOLUTION) -> SRStatus:
    """Cheap status check (does not load the model)."""
    spec = get_model_spec(model_name, settings)
    path = model_path(spec, settings)
    installed = torch_available()
    if not installed:
        msg = "PyTorch not installed - AI super-resolution unavailable (classical fallback only)."
    elif not path.is_file():
        msg = "PyTorch installed; model weights will be downloaded on first use (~5 MB for the compact model)."
    else:
        msg = "PyTorch installed and model weights present."
    return SRStatus(installed, torch_version() if installed else None, spec.name, path.is_file(), path, False, msg)
