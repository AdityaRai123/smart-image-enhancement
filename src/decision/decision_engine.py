"""
Stage 3 - Rule-based, explainable decision engine.

Input: the flat feature dictionary produced by Stage 2
(``QualityReport.features()``)::

    {"histogram_spread": 0.41, "contrast_score": 0.10, "histogram_skewness": -0.2,
     "blur_score": 8.3, "noise_score": 9.7, "impulse_ratio": 0.0,
     "width": 512, "height": 512, "resolution_flag": False, ...}

Output: a :class:`DecisionResult` with

    diagnosis   - one human-readable verdict per quality aspect, each traceable to a
                  measured statistic and a threshold
    operations  - the ORDERED list of operations to execute, with parameters
    reasoning   - why each operation was selected (and why in this order)
    parameters  - the parameter dictionary of every selected operation
    rules       - every rule that was evaluated: condition, measured value, threshold,
                  fired / not fired  (the "rule trace")
    skipped     - operations that were considered but deliberately NOT applied

Rule summary (thresholds live in ``config/config.py``):

    N1  impulse ratio > T_imp                      -> median filter
    N2  noise sigma > T_noise (no impulses)        -> denoise
          N2a  image is soft (blur score < T_blur) -> Gaussian  (no sharp edges to keep)
          N2b  otherwise                           -> bilateral (edge-preserving)
    C1  histogram spread < T_spread                -> contrast stretching (linear)
    C2  RMS contrast after stretch < T_rms         -> equalisation
          C2a  |skewness| > T_skew                 -> histogram equalisation (custom)
          C2b  otherwise                           -> CLAHE
    S1  blur score < T_blur                        -> unsharp masking (strong if < T_severe)
    R1  resolution flag                            -> super-resolution (Real-ESRGAN or fallback)

    Order: impulse removal -> denoising -> contrast -> sharpening -> super-resolution.
    If Real-ESRGAN will run, N2 (Gaussian-type noise) and S1 are left to the network,
    whose training degradations include blur and noise.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from config import DECISION, ENHANCEMENT, DecisionThresholds, EnhancementDefaults

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = (
    "histogram_spread",
    "contrast_score",
    "histogram_skewness",
    "blur_score",
    "noise_score",
    "impulse_ratio",
    "width",
    "height",
    "resolution_flag",
)

# Execution order of operation categories (lower runs first).
ORDER_IMPULSE, ORDER_DENOISE, ORDER_CONTRAST, ORDER_EQUALIZE, ORDER_SHARPEN, ORDER_SR = range(6)


@dataclass
class Diagnosis:
    aspect: str  # "contrast" | "sharpness" | "noise" | "resolution"
    level: str  # e.g. "Low", "Medium", "High", "Blurry", "Moderate", "Sharp", "Impulse"
    needs_action: bool
    summary: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class PlannedOperation:
    key: str  # registry key in src.enhancement.OPERATIONS (or "super_resolution")
    label: str
    category: str
    params: dict[str, Any]
    reason: str
    rule_id: str
    order: int


@dataclass
class RuleEvaluation:
    rule_id: str
    description: str
    condition: str
    measured: str
    fired: bool
    outcome: str


@dataclass
class DecisionOptions:
    """Run-time options that influence the plan (not image statistics)."""

    sr_enabled: bool = False  # user ticked "Enable AI super-resolution"
    sr_ai_available: bool = False  # Real-ESRGAN loaded successfully
    sr_fallback_available: bool = True  # Lanczos fallback allowed


@dataclass
class DecisionResult:
    diagnosis: dict[str, Diagnosis]
    operations: list[PlannedOperation]
    reasoning: list[str]
    parameters: dict[str, dict[str, Any]]
    rules: list[RuleEvaluation]
    skipped: list[str]
    recommendations: list[str]
    features: dict[str, Any]

    @property
    def no_enhancement_needed(self) -> bool:
        return not self.operations

    @property
    def pipeline_labels(self) -> list[str]:
        return [op.label for op in self.operations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": {k: asdict(v) for k, v in self.diagnosis.items()},
            "operations": [op.key for op in self.operations],
            "operation_details": [asdict(op) for op in self.operations],
            "reasoning": list(self.reasoning),
            "parameters": dict(self.parameters),
            "rules": [asdict(r) for r in self.rules],
            "skipped": list(self.skipped),
            "recommendations": list(self.recommendations),
            "features": {k: _jsonable(v) for k, v in self.features.items()},
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _validate_features(features: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_FEATURES if k not in features]
    if missing:
        raise ValueError(f"Decision engine input is missing features: {missing}")
    clean: dict[str, Any] = {k: _jsonable(v) for k, v in features.items()}
    for key in REQUIRED_FEATURES:
        if key == "resolution_flag":
            clean[key] = bool(clean[key])
            continue
        value = float(clean[key])
        if not np.isfinite(value):
            raise ValueError(f"Feature '{key}' is not a finite number ({value}).")
        clean[key] = value
    return clean


class DecisionEngine:
    """Maps quality statistics to an ordered, parameterised enhancement plan."""

    def __init__(
        self,
        thresholds: DecisionThresholds = DECISION,
        params: EnhancementDefaults = ENHANCEMENT,
    ) -> None:
        self.t = thresholds
        self.p = params

    # ------------------------------------------------------------------ diagnosis
    def diagnose(self, f: Mapping[str, Any]) -> dict[str, Diagnosis]:
        t = self.t
        spread, rms, skew = f["histogram_spread"], f["contrast_score"], f["histogram_skewness"]
        rms_after_stretch = rms / spread if spread > 0 else 0.0
        effective_rms = rms_after_stretch if spread < t.contrast_low_spread else rms

        # Contrast
        if spread < t.contrast_very_low_spread:
            c_level, c_action = "Very low", True
            c_summary = "Very low contrast: the histogram occupies only a small part of the 0-255 range."
        elif spread < t.contrast_low_spread:
            c_level, c_action = "Low", True
            c_summary = "Low contrast: the histogram is compressed into a narrow range."
        elif effective_rms < t.contrast_low_rms:
            c_level, c_action = "Low", True
            c_summary = "Low contrast: the range is used, but most pixels are bunched together (flat/hazy)."
        elif spread >= t.contrast_high_spread and rms >= t.contrast_high_rms:
            c_level, c_action = "High", False
            c_summary = "High contrast: wide histogram and large tonal variation."
        else:
            c_level, c_action = "Medium", False
            c_summary = "Medium contrast: adequate range and tonal variation."
        contrast = Diagnosis(
            "contrast", c_level, c_action, c_summary,
            [
                f"histogram spread = {spread:.3f} (low < {t.contrast_low_spread}, very low < {t.contrast_very_low_spread})",
                f"RMS contrast = {rms:.3f} (predicted after stretching: {effective_rms:.3f}; flat < {t.contrast_low_rms})",
                f"skewness = {skew:+.2f} (strongly skewed if |skew| > {t.skew_threshold})",
            ],
        )

        # Noise
        sigma, impulse = f["noise_score"], f["impulse_ratio"]
        if impulse > t.impulse_ratio_threshold:
            n_level, n_action = "Impulse (salt-and-pepper)", True
            n_summary = f"Impulse noise detected: {impulse * 100:.2f} % of pixels are isolated black/white outliers."
        elif sigma >= t.noise_high_sigma:
            n_level, n_action = "High", True
            n_summary = f"High noise: estimated sigma = {sigma:.1f} grey levels."
        elif sigma > t.noise_medium_sigma:
            n_level, n_action = "Medium", True
            n_summary = f"Moderate noise: estimated sigma = {sigma:.1f} grey levels."
        else:
            n_level, n_action = "Low", False
            n_summary = f"Low noise: estimated sigma = {sigma:.1f} grey levels."
        noise = Diagnosis(
            "noise", n_level, n_action, n_summary,
            [
                f"noise sigma (local std in flat regions) = {sigma:.2f} (medium > {t.noise_medium_sigma}, high >= {t.noise_high_sigma})",
                f"impulse ratio = {impulse * 100:.3f} % (impulse noise if > {t.impulse_ratio_threshold * 100:.2f} %)",
            ],
        )

        # Sharpness
        blur = f["blur_score"]
        if blur < t.blur_severe_threshold:
            s_level, s_action, s_summary = "Blurry", True, "Strong blur: very little high-frequency detail."
        elif blur < t.blur_threshold:
            s_level, s_action, s_summary = "Moderate", True, "Slightly soft: edges are weaker than in a sharp image."
        else:
            s_level, s_action, s_summary = "Sharp", False, "Sharp: strong high-frequency detail."
        raw_lv = f.get("laplacian_variance_raw")
        sharp_evidence = [
            f"blur score (Laplacian variance, contrast-normalised, noise-compensated) = {blur:.1f} "
            f"(soft < {t.blur_threshold}, blurry < {t.blur_severe_threshold})"
        ]
        if raw_lv is not None:
            sharp_evidence.append(f"raw Laplacian variance = {float(raw_lv):.1f} (classic metric, noise-sensitive; informational)")
        sharpness = Diagnosis("sharpness", s_level, s_action, s_summary, sharp_evidence)

        # Resolution
        w, h = int(f["width"]), int(f["height"])
        pixels = w * h
        if f["resolution_flag"]:
            r_level, r_action = "Low", True
            r_summary = f"Low resolution: {w}x{h} ({pixels / 1e6:.3f} MP)."
        elif pixels >= t.high_res_min_pixels:
            r_level, r_action, r_summary = "High", False, f"High resolution: {w}x{h} ({pixels / 1e6:.2f} MP)."
        else:
            r_level, r_action, r_summary = "Acceptable", False, f"Acceptable resolution: {w}x{h} ({pixels / 1e6:.2f} MP)."
        resolution = Diagnosis(
            "resolution", r_level, r_action, r_summary,
            [f"shorter side = {min(w, h)} px (low < {t.low_res_min_side}); pixels = {pixels:,} (low < {t.low_res_max_pixels:,})"],
        )
        return {"contrast": contrast, "noise": noise, "sharpness": sharpness, "resolution": resolution}

    # ------------------------------------------------------------------ planning
    def decide(self, features: Mapping[str, Any], options: DecisionOptions | None = None) -> DecisionResult:
        """Run all rules and return the ordered, explained plan."""
        opts = options or DecisionOptions()
        f = _validate_features(features)
        t, p = self.t, self.p
        diagnosis = self.diagnose(f)

        ops: list[PlannedOperation] = []
        rules: list[RuleEvaluation] = []
        skipped: list[str] = []
        recommendations: list[str] = []

        spread, rms, skew = f["histogram_spread"], f["contrast_score"], f["histogram_skewness"]
        sigma, impulse, blur = f["noise_score"], f["impulse_ratio"], f["blur_score"]
        low_res = f["resolution_flag"]
        is_soft = blur < t.blur_threshold

        # R1 is evaluated first because Real-ESRGAN changes what the classical stages must do.
        ai_sr_will_run = low_res and opts.sr_enabled and opts.sr_ai_available
        fallback_sr_will_run = low_res and opts.sr_enabled and not opts.sr_ai_available and opts.sr_fallback_available

        # ---- N1: impulse noise -> median -------------------------------------------
        n1 = impulse > t.impulse_ratio_threshold
        ksize = p.median_ksize_strong if impulse > t.impulse_strong_ratio else p.median_ksize_mild
        rules.append(RuleEvaluation(
            "N1", "Impulse (salt-and-pepper) noise", f"impulse ratio > {t.impulse_ratio_threshold * 100:.2f} %",
            f"{impulse * 100:.3f} %", n1, f"Median filter {ksize}x{ksize}" if n1 else "no median filter",
        ))
        if n1:
            ops.append(PlannedOperation(
                "median_denoise", "Median denoising", "denoise", {"ksize": ksize},
                f"Median {ksize}x{ksize} selected because {impulse * 100:.2f} % of pixels are isolated "
                f"black/white outliers (threshold {t.impulse_ratio_threshold * 100:.2f} %). A median filter removes "
                "impulses completely, while Gaussian or bilateral filters would only smear them.",
                "N1", ORDER_IMPULSE,
            ))

        # ---- N2: Gaussian-type noise -> Gaussian or bilateral ---------------------------
        n2 = sigma > t.noise_medium_sigma and not n1
        n2_outcome = "no denoising"
        if n2 and ai_sr_will_run:
            n2_outcome = "deferred to Real-ESRGAN"
            skipped.append(
                f"Classical denoising skipped: noise sigma {sigma:.1f} > {t.noise_medium_sigma}, but Real-ESRGAN will run "
                "and its training degradations include Gaussian/Poisson noise, so it denoises while upscaling."
            )
        elif n2 and is_soft:
            g_sigma = float(np.clip(sigma * p.gaussian_sigma_per_noise, p.gaussian_sigma_min, p.gaussian_sigma_max))
            n2_outcome = f"Gaussian (sigma {g_sigma:.2f})"
            ops.append(PlannedOperation(
                "gaussian_denoise", "Gaussian denoising", "denoise", {"sigma": round(g_sigma, 2)},
                f"Gaussian smoothing selected because the noise estimate ({sigma:.1f}) exceeds {t.noise_medium_sigma} AND "
                f"the image is already soft (blur score {blur:.1f} < {t.blur_threshold}): with no sharp edges to "
                "preserve, uniform Gaussian averaging removes noise more evenly than a bilateral filter, which "
                "leaves blotchy noise when edges and noise have similar amplitude.",
                "N2a", ORDER_DENOISE,
            ))
        elif n2:
            diameter = p.bilateral_diameter_strong if sigma >= t.noise_high_sigma else p.bilateral_diameter_mild
            sigma_color = float(np.clip(sigma * p.bilateral_sigma_color_per_noise,
                                        p.bilateral_sigma_color_min, p.bilateral_sigma_color_max))
            n2_outcome = f"bilateral (d {diameter}, sigmaColor {sigma_color:.0f})"
            ops.append(PlannedOperation(
                "bilateral_denoise", "Bilateral denoising", "denoise",
                {"diameter": diameter, "sigma_color": round(sigma_color, 1), "sigma_space": p.bilateral_sigma_space},
                f"Bilateral filtering selected because the noise estimate ({sigma:.1f}) exceeds {t.noise_medium_sigma} "
                f"while the image still has sharp edges (blur score {blur:.1f} >= {t.blur_threshold}). The range kernel "
                f"(sigmaColor = {p.bilateral_sigma_color_per_noise:g} x noise = {sigma_color:.0f}) averages away noise "
                "but ignores neighbours across an edge, so edges are preserved.",
                "N2b", ORDER_DENOISE,
            ))
        rules.append(RuleEvaluation(
            "N2", "Gaussian-type noise", f"noise sigma > {t.noise_medium_sigma} (and no impulse noise)",
            f"{sigma:.2f}", n2, n2_outcome,
        ))
        if n2 and not ai_sr_will_run:
            rules.append(RuleEvaluation(
                "N2a/b", "Choice of denoiser", f"blur score < {t.blur_threshold} -> Gaussian, else bilateral",
                f"{blur:.1f}", True, n2_outcome,
            ))
        denoised = any(op.category == "denoise" for op in ops)

        # ---- C1: compressed range -> contrast stretching ---------------------------------
        c1 = spread < t.contrast_low_spread
        gain = 1.0 / spread if spread > 0 else float("inf")
        rules.append(RuleEvaluation(
            "C1", "Compressed histogram range", f"histogram spread < {t.contrast_low_spread}",
            f"{spread:.3f}", c1, f"contrast stretching (gain ~{gain:.2f}x)" if c1 else "no stretching",
        ))
        if c1:
            severity = "very low" if spread < t.contrast_very_low_spread else "low"
            ops.append(PlannedOperation(
                "contrast_stretch", "Contrast stretching", "contrast",
                {"low_percentile": p.stretch_low_percentile, "high_percentile": p.stretch_high_percentile},
                f"Contrast stretching selected because the histogram spread ({spread:.3f}) is below {t.contrast_low_spread} "
                f"({severity} contrast). The pixels use only {spread * 100:.0f} % of the available range, so a linear "
                f"remap (gain ~{gain:.1f}x, larger for lower contrast) restores it without changing the histogram's "
                "shape; the same gain is applied to R, G and B to avoid colour casts.",
                "C1", ORDER_CONTRAST,
            ))

        # ---- C2: bunched distribution -> HE / CLAHE ----------------------------------------
        rms_after = rms * gain if c1 else rms
        c2 = rms_after < t.contrast_low_rms
        if c2:
            if abs(skew) > t.skew_threshold:
                c2_outcome = "histogram equalisation (custom)"
                ops.append(PlannedOperation(
                    "histogram_equalization", "Histogram equalisation (custom)", "contrast", {"color_space": "YCrCb (Y only)"},
                    f"Global histogram equalisation selected because the RMS contrast stays low ({rms_after:.3f} < "
                    f"{t.contrast_low_rms}) even after restoring the range, AND the histogram is strongly skewed "
                    f"(skewness {skew:+.2f}, |skew| > {t.skew_threshold}): pixel mass is piled at one end, which only a "
                    "non-linear CDF-based remapping can redistribute. Applied to the luminance only.",
                    "C2a", ORDER_EQUALIZE,
                ))
            else:
                clip_limit = p.clahe_clip_limit_strong if spread < t.contrast_very_low_spread else p.clahe_clip_limit_mild
                c2_outcome = f"CLAHE (clip {clip_limit})"
                ops.append(PlannedOperation(
                    "clahe", "CLAHE", "contrast", {"clip_limit": clip_limit, "tile_grid": list(p.clahe_tile_grid)},
                    f"CLAHE selected because the RMS contrast stays low ({rms_after:.3f} < {t.contrast_low_rms}) even with "
                    f"the full range: most pixels are bunched in a narrow band (flat/hazy content) while the histogram "
                    f"is not strongly skewed (|{skew:.2f}| <= {t.skew_threshold}). Local, clip-limited equalisation "
                    "raises local contrast without the noise amplification of plain adaptive HE.",
                    "C2b", ORDER_EQUALIZE,
                ))
        else:
            c2_outcome = "no equalisation"
        rules.append(RuleEvaluation(
            "C2", "Pixels bunched in a narrow band", f"RMS contrast (after stretching) < {t.contrast_low_rms}",
            f"{rms_after:.3f}", c2, c2_outcome,
        ))

        # ---- S1: blur -> unsharp masking ------------------------------------------------------
        s1 = is_soft
        s1_outcome = "no sharpening"
        if s1 and ai_sr_will_run:
            s1_outcome = "deferred to Real-ESRGAN"
            skipped.append(
                f"Unsharp masking skipped: blur score {blur:.1f} < {t.blur_threshold}, but Real-ESRGAN will run and "
                "reconstructs high-frequency detail itself; pre-sharpened halos would be amplified by the network."
            )
        elif s1:
            strong = blur < t.blur_severe_threshold
            usm_sigma = p.usm_sigma_strong if strong else p.usm_sigma_mild
            amount = p.usm_amount_strong if strong else p.usm_amount_mild
            threshold = p.usm_threshold_after_denoise if denoised else p.usm_threshold
            s1_outcome = f"unsharp mask ({'strong' if strong else 'mild'})"
            order_note = (
                " It runs AFTER denoising because sharpening a noisy image amplifies the noise." if denoised else ""
            )
            ops.append(PlannedOperation(
                "unsharp_mask", "Unsharp-mask sharpening", "sharpen",
                {"sigma": usm_sigma, "amount": amount, "threshold": threshold},
                f"Unsharp masking ({'strong' if strong else 'mild'}) selected because the blur score ({blur:.1f}) is below "
                f"{t.blur_threshold}{' and even below ' + str(t.blur_severe_threshold) if strong else ''}: the image lacks "
                f"high-frequency detail. sigma={usm_sigma}, amount={amount}; differences below {threshold} grey levels "
                f"are not boosted.{order_note}",
                "S1", ORDER_SHARPEN,
            ))
        rules.append(RuleEvaluation(
            "S1", "Insufficient sharpness", f"blur score < {t.blur_threshold} (strong if < {t.blur_severe_threshold})",
            f"{blur:.1f}", s1, s1_outcome,
        ))

        # ---- R1: low resolution -> super-resolution ----------------------------------------------
        if low_res and opts.sr_enabled and (opts.sr_ai_available or opts.sr_fallback_available):
            method = "realesrgan" if ai_sr_will_run else "lanczos"
            r1_outcome = "Real-ESRGAN x4 (AI)" if ai_sr_will_run else "Lanczos x4 (classical fallback)"
            reason = (
                f"Super-resolution selected because the image is low-resolution ({int(f['width'])}x{int(f['height'])}: "
                f"shorter side < {t.low_res_min_side} px or < {t.low_res_max_pixels:,} pixels) and AI super-resolution "
                "is enabled. "
            )
            reason += (
                "Real-ESRGAN (AI) reconstructs plausible detail; it is reserved for this case because it is far more "
                "expensive than any classical filter. It runs last, on the cleaned-up image."
                if ai_sr_will_run else
                "Real-ESRGAN is unavailable, so classical Lanczos interpolation is used instead (it enlarges the image "
                "but cannot add detail)."
            )
            ops.append(PlannedOperation(
                "super_resolution", "AI super-resolution (Real-ESRGAN)" if ai_sr_will_run else "Upscaling (Lanczos fallback)",
                "super_resolution", {"method": method, "scale": 4}, reason, "R1", ORDER_SR,
            ))
        elif low_res and not opts.sr_enabled:
            r1_outcome = "recommended (AI super-resolution disabled)"
            recommendations.append(
                f"The image is low-resolution ({int(f['width'])}x{int(f['height'])}). Enable 'AI super-resolution' in the "
                "sidebar to upscale it with Real-ESRGAN (optional, slower on CPU)."
            )
            skipped.append("Super-resolution not applied: recommended, but AI super-resolution is disabled.")
        elif low_res:
            r1_outcome = "unavailable"
            skipped.append("Super-resolution not applied: Real-ESRGAN unavailable and fallback disabled.")
        else:
            r1_outcome = "not needed"
        rules.append(RuleEvaluation(
            "R1", "Low resolution",
            f"shorter side < {t.low_res_min_side} px OR pixels < {t.low_res_max_pixels:,}",
            f"{int(f['width'])}x{int(f['height'])}", low_res, r1_outcome,
        ))

        if fallback_sr_will_run:
            skipped.append("Real-ESRGAN unavailable: the classical Lanczos fallback is used for upscaling.")

        # ---- skipped operations of an already-good aspect --------------------------------
        if not s1:
            skipped.append(
                f"Sharpening skipped: blur score {blur:.1f} >= {t.blur_threshold} (already sharp). Unsharp masking "
                "would only add ringing/halo artefacts and amplify noise."
            )
        if not n1 and not n2:
            skipped.append(f"Denoising skipped: noise sigma {sigma:.1f} <= {t.noise_medium_sigma} and no impulse noise.")
        if not c1 and not c2:
            skipped.append(
                f"Contrast enhancement skipped: spread {spread:.3f} >= {t.contrast_low_spread} and RMS contrast "
                f"{rms:.3f} >= {t.contrast_low_rms}; equalising a well-exposed image would distort its tones."
            )

        ops.sort(key=lambda op: op.order)
        reasoning = self._reasoning(ops, diagnosis, denoised)
        parameters = {op.key: dict(op.params) for op in ops}
        return DecisionResult(diagnosis, ops, reasoning, parameters, rules, skipped, recommendations, f)

    @staticmethod
    def _reasoning(ops: list[PlannedOperation], diagnosis: dict[str, Diagnosis], denoised: bool) -> list[str]:
        if not ops:
            return [
                "All measured statistics are within their thresholds, so NO enhancement is applied. "
                "Processing a good image can only add artefacts (halos, noise amplification, tone shifts)."
            ]
        lines = [op.reason for op in ops]
        categories = [op.category for op in ops]
        if denoised and len(set(categories)) > 1:
            lines.append(
                "Order: denoising runs first because every later step (contrast gain, equalisation, sharpening, "
                "upscaling) amplifies whatever noise is still present."
            )
        if "sharpen" in categories and "contrast" in categories:
            lines.append("Order: sharpening runs after contrast correction so its strength applies to the final tonal range.")
        if "super_resolution" in categories and len(ops) > 1:
            lines.append(
                "Order: super-resolution runs last, on the corrected image, so classical filters operate on the small "
                "image (cheap) and the upscaler does not enlarge defects that were already fixed."
            )
        return lines


def decide(
    features: Mapping[str, Any],
    options: DecisionOptions | None = None,
    thresholds: DecisionThresholds = DECISION,
    params: EnhancementDefaults = ENHANCEMENT,
) -> DecisionResult:
    """Functional shortcut for ``DecisionEngine(thresholds, params).decide(features, options)``."""
    return DecisionEngine(thresholds, params).decide(features, options)
