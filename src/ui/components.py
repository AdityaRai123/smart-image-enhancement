"""Rendering helpers for each section of the Streamlit page (one function per section)."""

from __future__ import annotations

import html
import math
import platform
from dataclasses import asdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import APP, FIXED_PIPELINE, DecisionThresholds
from src.analysis.quality_analyzer import QualityReport
from src.decision.decision_engine import DecisionResult
from src.enhancement.background_removal import BackgroundRemovalUnavailable, rembg_available, remove_background
from src.enhancement.histogram_equalization import (
    histogram_equalization_opencv,
    histogram_equalization_with_details,
)
from src.evaluation.comparison import PipelineComparison
from src.evaluation.metrics import REFERENCE_GROUND_TRUTH, cap_psnr, format_psnr
from src.pipeline import MODE_ADAPTIVE, MODE_FIXED, MODE_LABELS, MODE_NONE, PipelineResult
from src.ui.session import STAGES
from src.utils.image_utils import IngestedImage, resize_max_side, resize_to_match, to_png_bytes
from src.visualization import plots

LEVEL_COLORS = {True: "orange", False: "green"}


def _close(fig) -> None:
    plt.close(fig)


def _display(image: np.ndarray) -> np.ndarray:
    """Downscale for on-screen display only (never used for metrics)."""
    return resize_max_side(image, APP.display_max_side)[0]


# --------------------------------------------------------------------------- header
def render_header() -> None:
    st.title(APP.title)
    st.markdown(f"#### {APP.subtitle}")
    st.caption(
        "Upload an image: the system measures what is wrong with it (contrast, blur, noise, resolution), a "
        "rule-based decision engine selects the minimal set of classical operations - in the right order - and "
        "optional Real-ESRGAN super-resolution is reserved for low-resolution images. Every decision is traced "
        "back to a measured statistic."
    )


def stage_tracker_html(statuses: list[str], times: dict[str, float] | None = None) -> str:
    """Five-stage pipeline diagram. ``statuses`` items: pending | running | done | error."""
    icons = {"pending": "&#9675;", "running": "&#9684;", "done": "&#10003;", "error": "&#9888;"}
    words = {"pending": "pending", "running": "running...", "done": "done", "error": "error"}
    cards = []
    for i, (name, status) in enumerate(zip(STAGES, statuses), start=1):
        t = ""
        if times and name in times and status == "done":
            ms = times[name]
            t = f"<div class='sea-time'>{'&lt;1' if ms < 1 else f'{ms:.0f}'} ms</div>"
        cards.append(
            f"<div class='sea-stage sea-{status}'><div class='sea-num'>Stage {i}</div>"
            f"<div class='sea-name'>{html.escape(name)}</div>"
            f"<div class='sea-status'>{icons[status]} {words[status]}</div>{t}</div>"
        )
    arrow = "<div class='sea-arrow'>&#10140;</div>"
    return (
        "<style>"
        ".sea-row{display:flex;flex-wrap:nowrap;align-items:stretch;gap:4px;margin:4px 0 12px 0}"
        ".sea-stage{flex:1 1 0;min-width:0;border:1px solid #e1e0d9;border-radius:10px;padding:8px 10px;background:#fcfcfb}"
        ".sea-num{font-size:11px;color:#898781;text-transform:uppercase;letter-spacing:.04em}"
        ".sea-name{font-weight:600;font-size:13px;line-height:1.25;color:#0b0b0b;margin:2px 0}"
        ".sea-status{font-size:12px;color:#52514e}"
        ".sea-time{font-size:11px;color:#898781}"
        ".sea-done{border-color:#0ca30c55;background:#f3faf3}"
        ".sea-running{border-color:#2a78d6;background:#eef5fd}"
        ".sea-error{border-color:#d03b3b;background:#fdf1f1}"
        ".sea-arrow{align-self:center;color:#898781;font-size:14px;flex:0 0 auto}"
        "@media (max-width:640px){.sea-row{flex-wrap:wrap}.sea-stage{flex:1 1 40%}.sea-arrow{display:none}}"
        "</style><div class='sea-row'>" + arrow.join(cards) + "</div>"
    )


# --------------------------------------------------------------------------- stage 1
def render_ingestion(ingested: IngestedImage, reference_label: str | None) -> None:
    st.header("Stage 1 - Image Ingestion")
    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.image(_display(ingested.image), caption="Input image (working copy)", width="stretch")
    with col_info:
        rows = {
            "File": ingested.filename or "-",
            "Format / mode": f"{ingested.source_format or '-'} / {ingested.source_mode or 'RGB'}",
            "Original size": f"{ingested.original_width} x {ingested.original_height} px",
            "Working size": f"{ingested.width} x {ingested.height} px",
            "File size": f"{ingested.file_size_bytes / 1024:.1f} KB" if ingested.file_size_bytes else "-",
            "Grayscale": "yes" if ingested.is_grayscale else "no",
            "Transparency flattened": "yes" if ingested.had_alpha else "no",
            "Evaluation reference": reference_label or "the input itself (no ground truth)",
        }
        st.dataframe(pd.DataFrame({"Property": rows.keys(), "Value": rows.values()}), hide_index=True)
        for note in ingested.notes:
            st.caption(f"- {note}")


# --------------------------------------------------------------------------- stage 2
def render_quality(quality: QualityReport, decision: DecisionResult, image: np.ndarray) -> None:
    st.header("Stage 2 - Quality Analysis")
    st.caption("Every label below is derived from the measured statistic shown with it - no hidden model.")
    c, n, s, r = quality.contrast, quality.noise, quality.sharpness, quality.resolution
    d = decision.diagnosis
    cards = [
        ("Contrast", d["contrast"], [
            ("Histogram spread", f"{c.histogram_spread:.3f}", "(P99 - P1) / 255 of the luminance histogram"),
            ("RMS contrast", f"{c.rms_contrast:.3f}", "std(Y) / 255"),
            ("Skewness", f"{c.skewness:+.2f}", "third standardised moment of Y"),
        ]),
        ("Sharpness", d["sharpness"], [
            ("Blur score", f"{s.blur_score:.1f}", "Laplacian variance of the smoothed, contrast-normalised "
             "luminance, minus the expected noise contribution"),
            ("Raw Laplacian var.", f"{s.laplacian_variance_raw:.1f}", "classic var(Laplacian(Y)) - informational"),
            ("Edge density", f"{s.edge_density * 100:.1f} %", "Canny edge pixels - informational"),
        ]),
        ("Noise", d["noise"], [
            ("Noise sigma", f"{n.sigma:.2f}", "median local std of the structure-free residual in flat regions"),
            ("Impulse ratio", f"{n.impulse_ratio * 100:.3f} %", "black/white outliers vs 3x3 median"),
            ("Flat area used", f"{n.flat_fraction * 100:.0f} %", "share of pixels used for the estimate"),
        ]),
        ("Resolution", d["resolution"], [
            ("Width x height", f"{r.width} x {r.height}", "working image"),
            ("Megapixels", f"{r.megapixels:.3f} MP", "width x height / 10^6"),
            ("Low-res flag", "yes" if r.resolution_flag else "no", "shorter side / pixel-count rule"),
        ]),
    ]
    cols = st.columns(4)
    for col, (title, diag, metrics) in zip(cols, cards):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.badge(diag.level, color=LEVEL_COLORS[diag.needs_action],
                     icon=":material/warning:" if diag.needs_action else ":material/check_circle:")
            for label, value, help_text in metrics:
                st.metric(label, value, help=help_text)

    st.caption(f"Analysis time: {quality.analysis_time_ms:.0f} ms on CPU.")
    with st.expander("How is each statistic computed?"):
        st.markdown(
            """
- **Histogram spread** = (P99 - P1) / 255 of the luminance Y (YCrCb). Fraction of the 0-255 range used by
  the central 98 % of pixels (percentiles ignore a few outliers).
- **RMS contrast** = std(Y) / 255. **Skewness** = E[((Y - mean)/std)^3]; |skew| > 1 means the pixel mass is
  piled up at one end of the histogram (under- or over-exposure).
- **Blur score** = Var(Laplacian(G_1px * Y)) x gain^2 - E x (gain x sigma_noise)^2.
  The Laplacian (second derivative) responds to edges; blur lowers its variance.
  *gain* = 255 / (P99 - P1) removes the effect of contrast (variance scales with contrast^2), the 1-px Gaussian
  pre-smoothing and the subtraction of the exact noise term *E* stop noise from masquerading as sharpness.
  The classic raw Laplacian variance is shown for reference.
- **Noise sigma**: each channel is filtered with Immerkaer's structure-cancelling kernel
  [[1,-2,1],[-2,4,-2],[1,-2,1]]/6, the local standard deviation is computed in 7x7 windows, and the median is
  taken over homogeneous windows (lowest 20 % Sobel gradient, no clipped pixels). In a flat region only noise
  makes neighbouring pixels differ.
- **Impulse ratio**: share of near-black / near-white pixels that differ from their 3x3 median by > 60 levels.
- **Resolution flag**: shorter side < threshold OR pixel count < threshold (see config/config.py).
"""
        )
    with st.expander("Diagnostic maps (Laplacian, Canny edges, noise estimator)"):
        fig = plots.diagnostic_maps(image)
        st.pyplot(fig)
        _close(fig)


# --------------------------------------------------------------------------- stage 3
def render_decision(decision: DecisionResult, thresholds: DecisionThresholds, modified: bool) -> None:
    st.header("Stage 3 - Decision Engine")
    if modified:
        st.info("Custom thresholds from the sidebar are active for this session.")

    col_diag, col_plan = st.columns([1, 1])
    with col_diag:
        st.subheader("Diagnosis")
        for diag in decision.diagnosis.values():
            marker = ":orange[**needs action**]" if diag.needs_action else ":green[ok]"
            st.markdown(f"**{diag.aspect.capitalize()} - {diag.level}** ({marker})  \n{diag.summary}")
            for ev in diag.evidence:
                st.caption(f"- {ev}")
    with col_plan:
        st.subheader("Recommended pipeline")
        if decision.no_enhancement_needed:
            st.success(
                "No enhancement needed - every statistic is within its threshold, so the image is returned "
                "unchanged (avoids adding artefacts to a good image)."
            )
        else:
            for i, op in enumerate(decision.operations, start=1):
                params = ", ".join(f"{k}={v}" for k, v in op.params.items())
                st.markdown(f"**{i}. {op.label}**  \n`{params}`  (rule {op.rule_id})")
        for rec in decision.recommendations:
            st.info(rec)

    st.subheader("Why did the system choose these operations?")
    for line in decision.reasoning:
        st.markdown(f"- {line}")
    if decision.skipped:
        st.markdown("**Considered but not applied:**")
        for line in decision.skipped:
            st.markdown(f"- {line}")

    with st.expander("Rule trace (every rule evaluated, with measured value and threshold)", expanded=False):
        trace = pd.DataFrame([
            {"Rule": r.rule_id, "Checks": r.description, "Condition": r.condition, "Measured": r.measured,
             "Fired": "yes" if r.fired else "no", "Outcome": r.outcome}
            for r in decision.rules
        ])
        st.dataframe(trace, hide_index=True)


# --------------------------------------------------------------------------- stage 4
def render_execution(result: PipelineResult, keep_intermediates: bool) -> None:
    st.header("Stage 4 - Enhancement Execution")
    if not result.steps:
        st.write("Nothing to execute: the decision engine selected no operations.")
        return
    rows = []
    for i, step in enumerate(result.steps, start=1):
        rows.append({
            "#": i,
            "Operation": step.label,
            "Parameters": ", ".join(f"{k}={v}" for k, v in step.params.items()),
            "Time (ms)": round(step.elapsed_ms, 1),
            "Output size": f"{step.output_shape[1]} x {step.output_shape[0]}",
            "Status": step.error or "ok",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    for step in result.steps:
        for note in step.notes:
            st.caption(f"- {note}")
        if step.error:
            st.error(step.error)
    if keep_intermediates and any(s.image is not None for s in result.steps):
        with st.expander("Intermediate result after each step", expanded=False):
            cols = st.columns(min(len(result.steps), 4))
            for i, step in enumerate(result.steps):
                if step.image is not None:
                    cols[i % len(cols)].image(_display(step.image), caption=f"{i + 1}. {step.label}", width="stretch")


# --------------------------------------------------------------------------- stage 5
def _composite(before: np.ndarray, after: np.ndarray, split: float) -> np.ndarray:
    """Left part from ``before``, right part from ``after``, with a divider line."""
    out = after.copy()
    x = int(round(split * after.shape[1]))
    out[:, :x] = before[:, :x]
    if 0 < x < out.shape[1]:
        out[:, max(0, x - 1):x + 1] = (255, 255, 255)
    return out


def render_before_after(original: np.ndarray, result: PipelineResult, run_ms: float) -> None:
    st.header("Stage 5 - Evaluation & Presentation")
    st.subheader(f"Before / after - {result.label}")
    enhanced = result.output
    tab_side, tab_slider = st.tabs(["Side by side", "Comparison slider"])
    with tab_side:
        c1, c2 = st.columns(2)
        c1.image(_display(original), caption=f"Original - {original.shape[1]} x {original.shape[0]} px", width="stretch")
        c2.image(_display(enhanced), caption=f"Enhanced - {enhanced.shape[1]} x {enhanced.shape[0]} px", width="stretch")
    with tab_slider:
        split = st.slider("Divider position (left = original, right = enhanced)", 0, 100, 50, key="split") / 100.0
        after_disp = _display(enhanced)
        before_disp = resize_to_match(original, after_disp.shape[:2])
        st.image(_composite(before_disp, after_disp, split), width="stretch")
        if original.shape[:2] != enhanced.shape[:2]:
            st.caption("The original is resampled to the enhanced size for this view only.")

    ops = " -> ".join(result.operation_labels) or "none (image returned unchanged)"
    m1, m2, m3 = st.columns([3, 1, 1])
    m1.markdown(f"**Operations:** {ops}")
    m2.metric("Output size", f"{enhanced.shape[1]} x {enhanced.shape[0]}")
    m3.metric("Processing time", f"{run_ms:.0f} ms")
    st.download_button(
        "Download enhanced image (PNG)", to_png_bytes(enhanced),
        file_name=f"enhanced_{result.mode}.png", mime="image/png", icon=":material/download:",
    )


def render_histograms(original: np.ndarray, result: PipelineResult) -> None:
    st.subheader("Histogram comparison (luminance)")
    fig = plots.histogram_comparison(original, result.output, enhanced_label=result.label)
    st.pyplot(fig)
    _close(fig)
    st.caption(
        "A wider histogram means more of the dynamic range is used; a CDF closer to the diagonal means a more "
        "uniform distribution of grey levels (the target of histogram equalisation)."
    )


def render_metrics(comparison: PipelineComparison, mode: str, quality_before: QualityReport) -> None:
    st.subheader("Quantitative evaluation - PSNR and SSIM")
    row = comparison.row(mode)
    ground_truth = comparison.reference_kind == REFERENCE_GROUND_TRUTH
    ref_text = "the clean ground-truth reference" if ground_truth else "the original uploaded image"
    c1, c2, c3 = st.columns(3)
    c1.metric("PSNR", row.metrics.psnr_text, help="Peak signal-to-noise ratio, 10 log10(255^2 / MSE).")
    c2.metric("SSIM", row.metrics.ssim_text, help="Structural similarity (Wang et al. 2004), 1 = identical structure.")
    c3.metric("Compared against", "ground truth" if ground_truth else "original")
    st.markdown(
        f"Both metrics compare the **{MODE_LABELS[mode].lower()}** output with {ref_text}.  \n"
        "- **PSNR** (dB) measures pixel-wise fidelity from the mean squared error; higher = numerically closer.  \n"
        "- **SSIM** (0-1) compares local luminance, contrast and structure; it tracks perceived similarity better."
    )
    if row.metrics.note:
        st.caption(row.metrics.note)
    if not ground_truth:
        st.warning(
            "Limitation: without a ground-truth image, PSNR/SSIM are computed against the degraded original "
            "(the report's protocol). They then measure *how much the image was changed*, not whether it was "
            "improved - 'no enhancement' trivially scores PSNR = inf and SSIM = 1. Use the built-in sample mode "
            "or upload a reference image for a true restoration score."
        )

    after = row.after_quality
    if after is not None:
        st.markdown("**Before -> after quality statistics** (no-reference, same analysis as Stage 2)")
        table = pd.DataFrame({
            "Statistic": ["Histogram spread", "RMS contrast", "Blur score", "Noise sigma", "Impulse ratio (%)"],
            "Before": [quality_before.contrast.histogram_spread, quality_before.contrast.rms_contrast,
                       quality_before.sharpness.blur_score, quality_before.noise.sigma,
                       quality_before.noise.impulse_ratio * 100],
            "After": [after.contrast.histogram_spread, after.contrast.rms_contrast, after.sharpness.blur_score,
                      after.noise.sigma, after.noise.impulse_ratio * 100],
        })
        st.dataframe(table.style.format({"Before": "{:.3f}", "After": "{:.3f}"}), hide_index=True)


def render_baselines(comparison: PipelineComparison, original: np.ndarray) -> None:
    st.subheader("Baseline comparison - no enhancement vs fixed vs adaptive")
    fixed_chain = " -> ".join(FIXED_PIPELINE.operations)
    st.caption(
        f"Fixed pipeline = `{fixed_chain}` with mid-range parameters, applied to every image. The adaptive pipeline "
        "uses the same operator implementations, so differences come from the decision layer alone."
    )
    records = []
    for r in comparison.rows:
        aq = r.after_quality
        records.append({
            "Pipeline": r.label,
            "Operations": " -> ".join(r.operations) or "(none)",
            "PSNR": format_psnr(r.metrics.psnr),
            "SSIM": r.metrics.ssim_text,
            "Time (ms)": round(r.time_ms, 1),
            "Output size": f"{r.output_shape[1]} x {r.output_shape[0]}",
            "Blur score after": round(aq.sharpness.blur_score, 1) if aq else None,
            "Noise sigma after": round(aq.noise.sigma, 2) if aq else None,
            "Spread after": round(aq.contrast.histogram_spread, 3) if aq else None,
        })
    st.dataframe(pd.DataFrame(records), hide_index=True)

    chart_rows = [
        {"mode": r.mode, "label": {MODE_NONE: "None", MODE_FIXED: "Fixed", MODE_ADAPTIVE: "Adaptive"}[r.mode],
         "psnr": cap_psnr(r.metrics.psnr), "ssim": r.metrics.ssim}
        for r in comparison.rows
    ]
    c1, c2 = st.columns(2)
    with c1:
        fig = plots.metric_bars(chart_rows, "psnr", "PSNR (dB, higher = closer)", "dB")
        st.pyplot(fig)
        _close(fig)
    with c2:
        fig = plots.metric_bars(chart_rows, "ssim", "SSIM (higher = more similar)", "SSIM")
        st.pyplot(fig)
        _close(fig)
    if any(math.isinf(r.metrics.psnr) for r in comparison.rows):
        st.caption("Infinite PSNR (identical images) is drawn at the 60 dB cap.")
    if comparison.reference_kind != REFERENCE_GROUND_TRUTH:
        st.caption(
            "Against the original, 'no enhancement' always wins by definition; the informative columns here are the "
            "operations chosen, the processing time and the before/after statistics."
        )

    cols = st.columns(3)
    for col, r in zip(cols, comparison.rows):
        col.image(_display(comparison.results[r.mode].output), caption=r.label, width="stretch")


# --------------------------------------------------------------------------- extras
def render_he_demo(image: np.ndarray) -> None:
    with st.expander("From-scratch histogram equalisation (teaching view)"):
        custom, details = histogram_equalization_with_details(image)
        reference = histogram_equalization_opencv(image)
        diff = int(np.abs(custom.astype(int) - reference.astype(int)).max())
        st.markdown(
            "Implemented with NumPy only in `src/enhancement/histogram_equalization.py`: histogram -> normalised "
            "histogram -> CDF -> look-up table -> output. Applied to the luminance (Y of YCrCb) so colours are "
            f"not distorted. Maximum difference from OpenCV's `cv2.equalizeHist`: **{diff} grey level(s)**."
        )
        c1, c2, c3 = st.columns(3)
        c1.image(_display(image), caption="Input", width="stretch")
        c2.image(_display(custom), caption="Custom equalisation", width="stretch")
        c3.image(_display(reference), caption="OpenCV equalizeHist", width="stretch")
        fig = plots.equalization_details(details)
        st.pyplot(fig)
        _close(fig)


def render_technical_details(decision: DecisionResult, thresholds: DecisionThresholds, sr_message: str,
                             image: np.ndarray, stage_ms: dict[str, float]) -> None:
    with st.expander("Technical details"):
        tab_json, tab_cfg, tab_color, tab_env = st.tabs(
            ["Decision engine output (JSON)", "Thresholds in effect", "Colour-space views", "Environment"]
        )
        with tab_json:
            st.json(decision.to_dict(), expanded=1)
        with tab_cfg:
            st.json(asdict(thresholds))
        with tab_color:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            channels = [(ycrcb[..., 0], "Y (luma) - analysed & enhanced"), (ycrcb[..., 1], "Cr"), (ycrcb[..., 2], "Cb"),
                        (hsv[..., 0], "H (hue)"), (hsv[..., 1], "S (saturation)"), (hsv[..., 2], "V (value)")]
            cols = st.columns(3)
            for i, (channel, caption) in enumerate(channels):
                cols[i % 3].image(_display(channel), caption=caption, width="stretch", clamp=True)
            st.caption(
                "Contrast, equalisation and sharpening operate on Y only and keep Cr/Cb, so hues are preserved; "
                f"mean HSV saturation = {hsv[..., 1].mean() / 255:.3f} (informational)."
            )
        with tab_env:
            import importlib.metadata as md

            def version(pkg: str) -> str:
                try:
                    return md.version(pkg)
                except md.PackageNotFoundError:
                    return "not installed"

            env = {"Python": platform.python_version(), "Platform": platform.platform()}
            env.update({pkg: version(pkg) for pkg in
                        ("numpy", "opencv-python-headless", "pillow", "scikit-image", "streamlit", "matplotlib", "torch")})
            env["Super-resolution"] = sr_message
            env.update({f"Stage time - {k}": f"{v:.1f} ms" for k, v in stage_ms.items()})
            st.dataframe(pd.DataFrame({"Item": env.keys(), "Value": env.values()}), hide_index=True)


def render_background_removal(image: np.ndarray) -> None:
    with st.expander("Optional extra: background removal (rembg)"):
        st.caption(
            "Listed in the report as an optional AI tool. It changes image content rather than restoring quality, "
            "so the decision engine never selects it."
        )
        if not rembg_available():
            st.info("rembg is not installed. Install with `pip install rembg` (downloads a ~170 MB model on first use).")
            return
        if st.button("Remove background", icon=":material/content_cut:"):
            try:
                with st.spinner("Running rembg..."):
                    rgba = remove_background(image)
                st.image(_display(rgba), caption="Background removed (transparent)", width="stretch")
            except BackgroundRemovalUnavailable as exc:
                st.error(str(exc))
