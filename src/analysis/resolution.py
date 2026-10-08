"""Resolution analysis: pixel dimensions and the low-resolution flag."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from config import DECISION, DecisionThresholds


@dataclass
class ResolutionStats:
    width: int
    height: int
    pixels: int
    megapixels: float
    shorter_side: int
    aspect_ratio: float
    resolution_flag: bool  # True -> low resolution (super-resolution candidate)
    category: str  # "Low" | "Acceptable" | "High"

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_resolution(
    width: int, height: int, thresholds: DecisionThresholds = DECISION
) -> ResolutionStats:
    """Derive the resolution flag from the pixel dimensions.

    Low resolution  : shorter side < ``low_res_min_side`` OR pixels < ``low_res_max_pixels``
    High resolution : pixels >= ``high_res_min_pixels``
    Acceptable      : everything in between
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    pixels = int(width) * int(height)
    shorter = min(width, height)
    is_low = shorter < thresholds.low_res_min_side or pixels < thresholds.low_res_max_pixels
    if is_low:
        category = "Low"
    elif pixels >= thresholds.high_res_min_pixels:
        category = "High"
    else:
        category = "Acceptable"
    return ResolutionStats(
        width=int(width),
        height=int(height),
        pixels=pixels,
        megapixels=pixels / 1_000_000.0,
        shorter_side=int(shorter),
        aspect_ratio=float(width) / float(height),
        resolution_flag=is_low,
        category=category,
    )
