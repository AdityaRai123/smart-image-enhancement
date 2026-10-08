"""
Matplotlib figures used by the app and the evaluation script.

Visual conventions (kept consistent across every figure):
    * categorical colours in a fixed order, and colour follows the entity:
      original / no enhancement = blue, fixed pipeline = orange, adaptive = aqua
    * thin marks, solid hairline grid, recessive axes, a legend whenever >= 2 series
    * every chart has a table equivalent in the UI / CSV (colour is never the only channel)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend (Streamlit / scripts)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.utils.image_utils import ensure_rgb_uint8, rgb_to_gray  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = {
    "original": "#2a78d6",  # slot 1 blue
    "none": "#2a78d6",
    "fixed": "#eb6834",  # slot 2 orange
    "adaptive": "#1baf7a",  # slot 3 aqua
    "enhanced": "#1baf7a",
}


def _style(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8)


def _figure(width: float, height: float, ncols: int = 1) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    return fig, np.atleast_1d(axes)


def _hist(gray: np.ndarray) -> np.ndarray:
    hist = np.bincount(gray.ravel(), minlength=256)[:256].astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def histogram_comparison(original: np.ndarray, enhanced: np.ndarray, enhanced_label: str = "Enhanced") -> Figure:
    """Luminance histograms (left) and cumulative distributions (right), before vs after."""
    g0 = rgb_to_gray(ensure_rgb_uint8(original))
    g1 = rgb_to_gray(ensure_rgb_uint8(enhanced))
    h0, h1 = _hist(g0), _hist(g1)
    levels = np.arange(256)

    fig, (ax_h, ax_c) = _figure(9.0, 3.0, ncols=2)
    for hist, color, label in ((h0, SERIES["original"], "Original"), (h1, SERIES["enhanced"], enhanced_label)):
        ax_h.fill_between(levels, hist, color=color, alpha=0.18, linewidth=0, step="mid")
        ax_h.plot(levels, hist, color=color, linewidth=1.4, label=label, drawstyle="steps-mid")
        ax_c.plot(levels, np.cumsum(hist), color=color, linewidth=2.0, label=label)
    ax_c.plot([0, 255], [0, 1], color=MUTED, linewidth=0.8, label="Ideal (uniform)")
    _style(ax_h, "Luminance histogram", "Grey level (Y)", "Fraction of pixels")
    _style(ax_c, "Cumulative distribution (CDF)", "Grey level (Y)", "Cumulative fraction")
    ax_h.set_xlim(0, 255)
    ax_c.set_xlim(0, 255)
    ax_c.set_ylim(0, 1.0)
    for ax in (ax_h, ax_c):
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


def equalization_details(details) -> Figure:
    """Teaching figure for the from-scratch histogram equalisation (PDF, CDF, LUT)."""
    levels = np.arange(256)
    fig, axes = _figure(9.0, 2.6, ncols=3)
    axes[0].plot(levels, details.pdf, color=SERIES["original"], linewidth=1.2, label="Input")
    axes[0].plot(levels, details.equalized_histogram / max(details.equalized_histogram.sum(), 1.0),
                 color=SERIES["enhanced"], linewidth=1.2, label="Equalised")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY)
    _style(axes[0], "1-3. Normalised histogram p[k]", "k", "p[k]")
    axes[1].plot(levels, details.cdf, color=SERIES["original"], linewidth=2.0)
    _style(axes[1], "4. CDF  cdf[k] = sum p[j]", "k", "cdf[k]")
    axes[2].plot(levels, details.lut, color=SERIES["original"], linewidth=2.0)
    axes[2].plot([0, 255], [0, 255], color=MUTED, linewidth=0.8)
    _style(axes[2], "5. Mapping T[k] (LUT)", "input level k", "output level")
    for ax in axes:
        ax.set_xlim(0, 255)
    fig.tight_layout()
    return fig


def metric_bars(rows: list[dict], metric: str, title: str, ylabel: str) -> Figure:
    """Grouped bars: one group per pipeline. ``rows`` = [{"mode", "label", metric}, ...]."""
    fig, (ax,) = _figure(4.4, 2.8)
    labels = [r["label"] for r in rows]
    values = [r[metric] for r in rows]
    colors = [SERIES.get(r["mode"], MUTED) for r in rows]
    x = np.arange(len(rows))
    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.annotate(f"{value:.3f}" if metric == "ssim" else f"{value:.1f}",
                        (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x, labels, fontsize=8)
    _style(ax, title, None, ylabel)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def category_comparison(summary, metric: str, title: str, ylabel: str, modes: list[str], labels: dict[str, str]) -> Figure:
    """Grouped bar chart per degradation category (evaluation script).

    ``summary`` is a pandas DataFrame indexed by (category, mode) with a ``metric`` column.
    """
    categories = list(dict.fromkeys(summary.index.get_level_values(0)))
    fig, (ax,) = _figure(max(7.0, 1.1 * len(categories) + 2), 3.4)
    width = 0.8 / len(modes)
    x = np.arange(len(categories))
    for i, mode in enumerate(modes):
        values = [summary.loc[(cat, mode), metric] if (cat, mode) in summary.index else np.nan for cat in categories]
        ax.bar(x + (i - (len(modes) - 1) / 2) * width, values, width=width * 0.92,
               color=SERIES.get(mode, MUTED), label=labels.get(mode, mode), edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x, [c.replace("_", " ") for c in categories], fontsize=8)
    _style(ax, title, None, ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncols=len(modes), loc="upper left",
              bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    return fig


def diagnostic_maps(image: np.ndarray) -> Figure:
    """Laplacian response, Canny edges and the noise estimator's homogeneous-region mask."""
    import cv2

    from src.analysis.blur import canny_edges
    from src.analysis.noise import noise_maps

    gray = rgb_to_gray(ensure_rgb_uint8(image))
    lap = np.abs(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=1))
    edges = canny_edges(gray)
    maps = noise_maps(image)

    fig, axes = _figure(10.0, 3.0, ncols=4)
    panels = [
        (lap, "magma", "|Laplacian| (blur analysis)", np.percentile(lap, 99)),
        (edges, "gray", "Canny edges", 255),
        (maps["local_std"], "viridis", "Local std of noise residual", np.percentile(maps["local_std"], 99)),
        (maps["flat_mask"].astype(float), "gray", "Homogeneous regions (noise estimate)", 1),
    ]
    for ax, (data, cmap, title, vmax) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=max(float(vmax), 1e-6))
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    return fig
