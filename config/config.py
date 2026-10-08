"""
Central configuration: every threshold and default parameter lives here.

Why a single file?
    The Review 1 report states that "the specific thresholds used in the decision
    engine will be refined and validated against a small test set of representative
    degraded images". Keeping every number in one place (instead of scattering magic
    numbers through the code) makes that tuning loop explicit:

        1. ``python scripts/create_test_dataset.py``    -> controlled degraded images
        2. ``python scripts/analyze_thresholds.py``     -> per-category metric distributions
        3. edit the values below                         -> re-run evaluation

    The Streamlit sidebar ("Advanced settings") can also override the decision
    thresholds for a single session without editing this file.

All classes are frozen dataclasses: they are immutable at runtime and can be copied
with overrides via ``dataclasses.replace(DECISION, blur_threshold=80.0)``.

Units used throughout
    * Intensities are 8-bit (0-255) unless stated otherwise.
    * "Spread" and "RMS contrast" are normalised to 0-1 (divided by 255).
    * Noise sigma is the standard deviation of additive noise in 8-bit grey levels.
    * Laplacian variance is computed on the contrast-normalised luminance (0-255).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Stage 1 - Image ingestion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IngestionSettings:
    """Limits applied when an image enters the system."""

    # File types accepted by the uploader (the decoded format is checked as well).
    allowed_extensions: tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")
    # Pillow format names that correspond to the extensions above.
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO")
    # Upload size limit. Larger files are rejected before decoding.
    max_upload_megabytes: float = 25.0
    # Decompression-bomb guard: images with more decoded pixels than this are rejected.
    max_decoded_pixels: int = 60_000_000
    # CPU-first design: very large images are downscaled (INTER_AREA) to this longest
    # side before any processing. 2048 px keeps bilateral filtering < ~1 s on a laptop
    # CPU while preserving enough detail for blur/noise analysis.
    processing_max_side: int = 2048
    # Below this size the statistics (7x7 noise windows, SSIM windows) are meaningless.
    min_side: int = 16
    # Transparent pixels (RGBA / LA / palette transparency) are composited onto this colour.
    alpha_background_rgb: tuple[int, int, int] = (255, 255, 255)


# --------------------------------------------------------------------------- #
# Stage 2 - Quality analysis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalysisSettings:
    """Parameters of the measurement algorithms (not decision thresholds)."""

    # Histogram spread = (P_high - P_low) / 255 of the luminance histogram.
    # Using the 1st/99th percentiles instead of min/max makes the measure robust to a
    # handful of outlier pixels (specular highlights, dead pixels).
    spread_low_percentile: float = 1.0
    spread_high_percentile: float = 99.0

    # Noise estimation (local standard deviation in homogeneous regions), see
    # src/analysis/noise.py:
    # 1. residual = Immerkaer structure-cancelling filter (unit noise gain)
    # 2. local std of the residual in a window x window neighbourhood (box filters)
    # 3. keep only homogeneous windows: Sobel gradient magnitude of the luminance
    #    (smoothed with noise_gradient_sigma, so the selection is not driven by the noise
    #    itself) in the lowest noise_flat_gradient_percentile %, and no clipped pixels
    # 4. sigma = median(local std over homogeneous windows), averaged over R, G, B
    # Validation on the reference set: clean images 0.3-1.7, low-resolution texture up to
    # ~6.2, added sigma=15 noise measured as 13.8-14.6 and sigma=10 as 9.6-9.7
    # (see scripts/analyze_thresholds.py).
    noise_window: int = 7
    noise_gradient_sigma: float = 2.5
    noise_flat_gradient_percentile: float = 20.0
    # Pixels this close to 0 or 255 are excluded (clipping hides noise there).
    noise_saturation_margin: int = 3
    # Minimum number of flat pixels for a trustworthy estimate (fallback: all pixels).
    noise_min_flat_pixels: int = 500

    # Impulse (salt-and-pepper) detection: a pixel is an impulse candidate when it is
    # near-black/near-white AND differs from its 3x3 median by more than impulse_min_diff.
    impulse_dark_level: int = 10
    impulse_bright_level: int = 245
    impulse_min_diff: int = 60
    # Above this impulse ratio, the Gaussian-noise estimate is measured after a 3x3
    # median (otherwise isolated impulses would dominate the local standard deviation).
    impulse_prefilter_ratio: float = 0.001

    # Blur score = variance of the Laplacian of the lightly Gaussian-smoothed luminance.
    # The raw cv2.Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]] amplifies white noise by
    # sum(k^2) = 20, so noise easily masquerades as sharpness. A 1-px pre-smoothing
    # reduces that factor to ~0.2 (computed exactly in src/analysis/blur.py) while
    # real edges survive. Set to 0 to use the classic raw Laplacian variance.
    blur_presmooth_sigma: float = 1.0
    # The blur score is measured as if the histogram were stretched to full range
    # (gain = 255 / (P99 - P1)); the gain is capped for near-constant images.
    max_contrast_normalisation_gain: float = 8.0

    # Canny thresholds derived from the median intensity (lower = (1-s)*median,
    # upper = (1+s)*median); s = 0.33 is the widely used "auto-Canny" heuristic.
    canny_sigma: float = 0.33


# --------------------------------------------------------------------------- #
# Stage 3 - Decision engine thresholds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionThresholds:
    """
    Rule thresholds of the decision engine.

    Every value was set from first principles and then validated on the synthetic
    reference set (6 clean scikit-image photographs x 7 controlled degradations):
    ``python scripts/analyze_thresholds.py`` prints the measured distributions and the
    margins quoted below. Re-run it after changing any analysis setting.
    """

    # --- Contrast (luminance histogram) ---------------------------------------------
    # "Low contrast" = the central 98 % of pixels use less than ~half of the 0-255 range.
    # Measured: lowest clean reference 0.60 (chelsea), highest deliberately
    # low-contrast image 0.44 -> threshold at the midpoint of that gap.
    contrast_low_spread: float = 0.52
    # "Very low": about a third of the range or less (stretch gain >= ~3x).
    contrast_very_low_spread: float = 0.35
    # Display label "High" needs both a wide range and a large RMS contrast.
    contrast_high_spread: float = 0.85
    contrast_high_rms: float = 0.20
    # RMS contrast (std / 255) - predicted *after* stretching (= rms / spread) - below
    # this means pixels stay bunched in a narrow band even when the range is restored:
    # a distribution problem (haze, large flat areas) that only equalisation can fix.
    # Clean references: >= 0.119. 0.10 = a standard deviation of ~25 grey levels.
    contrast_low_rms: float = 0.10
    # |skewness| above this = histogram mass piled up at one end (under-/over-exposure)
    # -> global histogram equalisation instead of CLAHE for the distribution problem.
    skew_threshold: float = 1.0

    # --- Sharpness (blur score = normalised, noise-compensated Laplacian variance) -----
    # Measured: clean references 52-109; Gaussian blur sigma 1.0 -> 15-45,
    # sigma 1.5 -> 6-22, sigma 2.0 -> 2-11. 30 ~ geometric midpoint between the softest
    # clean image (52) and the blurred group (<= ~20 for sigma >= 1.5).
    blur_threshold: float = 30.0
    # Below this the strong unsharp-mask setting is used (sigma >= ~1.5 px blur).
    blur_severe_threshold: float = 10.0

    # --- Noise (estimated per-channel Gaussian sigma, 8-bit grey levels) ---------------
    # Measured: clean references 0.3-1.7; added sigma=5 noise reads 4.9-5.2. sigma ~5 is
    # where grain becomes visible in flat areas. Known confounder: fine texture of
    # 4x-downsampled images reads up to ~6.
    noise_medium_sigma: float = 5.0
    # "High" noise label; also switches the bilateral filter to its larger diameter.
    noise_high_sigma: float = 10.0
    # Fraction of salt-and-pepper candidate pixels. Clean/blurred/noisy images: <= 0.04 %;
    # 3 % impulse noise: >= 2.1 %. 0.4 % leaves a 10x margin on the clean side.
    impulse_ratio_threshold: float = 0.004
    # Above this ratio a 5x5 median is used (3x3 measured better up to ~20 % density).
    impulse_strong_ratio: float = 0.20

    # --- Resolution ---------------------------------------------------------------------
    # Low resolution: shorter side < 256 px OR fewer than 100 k pixels (thumbnail class,
    # between QVGA 320x240 = 76.8 k and VGA 640x480 = 307 k).
    low_res_min_side: int = 256
    low_res_max_pixels: int = 100_000
    # At/above this pixel count the resolution is labelled "High" (~Full-HD 1920x1080).
    high_res_min_pixels: int = 2_000_000


# --------------------------------------------------------------------------- #
# Stage 4 - Enhancement parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnhancementDefaults:
    """
    Parameters of the enhancement operators used by the adaptive pipeline.

    Chosen by restoring the clean references from controlled degradations and keeping
    the setting with the best mean PSNR/SSIM (numbers in the README, section
    "Decision-engine logic").
    """

    # Contrast stretching: luminance percentiles mapped to 0 and 255 (same gain on R,G,B).
    # 0.5/99.5 restored linear contrast loss better than 1/99 (29.3 vs 25.1 dB): less clipping.
    stretch_low_percentile: float = 0.5
    stretch_high_percentile: float = 99.5

    # CLAHE (applied on the Y channel of YCrCb).
    clahe_clip_limit_mild: float = 2.0
    clahe_clip_limit_strong: float = 3.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # Gaussian denoising (chosen for noisy AND soft images, where there are no sharp
    # edges for a bilateral filter to preserve): sigma = clip(noise * factor, min, max).
    # Best measured on blur 1.5 + noise 10: sigma ~1.0-1.3.
    gaussian_sigma_per_noise: float = 0.12
    gaussian_sigma_min: float = 0.8
    gaussian_sigma_max: float = 1.3

    # Median denoising kernel sizes (must be odd).
    median_ksize_mild: int = 3
    median_ksize_strong: int = 5

    # Bilateral denoising (noisy images with sharp edges): sigmaColor = noise * factor.
    # A range sigma of ~4x the noise sigma was best at every tested level (5, 8, 10, 15).
    bilateral_sigma_color_per_noise: float = 4.0
    bilateral_sigma_color_min: float = 15.0
    bilateral_sigma_color_max: float = 100.0
    bilateral_diameter_mild: int = 7
    bilateral_diameter_strong: int = 9
    bilateral_sigma_space: float = 3.0

    # Unsharp masking (applied on the Y channel): sharpened = Y + amount * (Y - blur(Y)),
    # only where |Y - blur(Y)| > threshold (prevents amplifying residual noise).
    # Best measured: blur sigma 1.0 -> (1.5, 1.0); blur sigma 1.5-2.0 -> (2.0, 1.5-2.0).
    usm_sigma_mild: float = 1.5
    usm_amount_mild: float = 1.0
    usm_sigma_strong: float = 2.0
    usm_amount_strong: float = 1.5
    usm_threshold: float = 2.0
    # After denoising some residual noise remains; a higher threshold avoids boosting it.
    usm_threshold_after_denoise: float = 3.0


# --------------------------------------------------------------------------- #
# Stage 4b - Optional AI super-resolution (Real-ESRGAN)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SRModelSpec:
    """Description of a downloadable Real-ESRGAN checkpoint."""

    name: str
    url: str
    architecture: str  # "srvgg" (compact) or "rrdb" (full RRDBNet)
    scale: int
    num_feat: int = 64
    num_conv: int = 32  # SRVGGNetCompact only
    num_block: int = 23  # RRDBNet only
    description: str = ""


@dataclass(frozen=True)
class SuperResolutionSettings:
    """Real-ESRGAN integration settings (CPU-first)."""

    models: dict[str, SRModelSpec] = field(
        default_factory=lambda: {
            # Lightweight variant (~1.2 M parameters, ~4.7 MB). Default for CPU use, as the
            # report specifies "a lightweight model variant".
            "realesr-general-x4v3": SRModelSpec(
                name="realesr-general-x4v3",
                url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                architecture="srvgg",
                scale=4,
                num_feat=64,
                num_conv=32,
                description="Compact SRVGG network (lightweight, CPU friendly)",
            ),
            # Full-size RRDBNet (~16.7 M parameters, ~64 MB). Higher quality, much slower on CPU.
            "RealESRGAN_x4plus": SRModelSpec(
                name="RealESRGAN_x4plus",
                url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                architecture="rrdb",
                scale=4,
                num_feat=64,
                num_block=23,
                description="Full RRDBNet generator (best quality, slow on CPU)",
            ),
        }
    )
    default_model: str = "realesr-general-x4v3"
    model_dir: Path = PROJECT_ROOT / "models"
    # CPU guard ("downscaled crop" in the report): inputs larger than this many pixels
    # are downscaled before inference so a x4 pass stays within seconds on a CPU.
    max_input_pixels: int = 400 * 400
    # Tiled inference bounds peak memory; pad avoids seams between tiles.
    tile_size: int = 200
    tile_pad: int = 10
    # Output longer side is capped (INTER_AREA) to keep the UI responsive.
    output_max_side: int = 2048
    download_timeout_seconds: int = 60
    # When Real-ESRGAN is unavailable but SR was requested, fall back to classical
    # Lanczos interpolation (clearly labelled as non-AI in the UI and results).
    fallback_to_interpolation: bool = True
    fallback_scale: int = 4


# --------------------------------------------------------------------------- #
# Baseline: fixed classical pipeline (for comparison with the adaptive pipeline)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FixedPipelineSettings:
    """
    A reasonable one-size-fits-all chain, as commonly found in tutorials and simple
    tools: CLAHE -> bilateral denoising -> unsharp masking, with mid-range parameters.
    It is applied to every image regardless of its measured quality.
    """

    operations: tuple[str, ...] = ("clahe", "bilateral_denoise", "unsharp_mask")
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    bilateral_diameter: int = 7
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 3.0
    usm_sigma: float = 1.0
    usm_amount: float = 0.8
    usm_threshold: float = 0.0


# --------------------------------------------------------------------------- #
# Stage 5 - Evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvaluationSettings:
    """Metric settings."""

    data_range: float = 255.0
    # PSNR of identical images is +infinity. For averaging in summary tables the value
    # is capped at this level (documented in the generated reports).
    psnr_cap_db: float = 60.0
    # SSIM Gaussian window (Wang et al. 2004 use 11x11 Gaussian, sigma 1.5).
    ssim_gaussian_weights: bool = True
    ssim_sigma: float = 1.5


# --------------------------------------------------------------------------- #
# Application / UI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppSettings:
    title: str = "Smart Multimedia Image Enhancement Assistant"
    subtitle: str = "An AI-Based Adaptive Enhancement System"
    # Images shown in the comparison slider are downscaled to this width (display only).
    display_max_side: int = 1100
    # Built-in sample images (scikit-image, bundled offline) for the synthetic demo mode.
    sample_images: tuple[str, ...] = (
        "astronaut",
        "coffee",
        "chelsea",
        "rocket",
        "camera",
        "immunohistochemistry",
    )
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    data_dir: Path = PROJECT_ROOT / "data"


INGESTION = IngestionSettings()
ANALYSIS = AnalysisSettings()
DECISION = DecisionThresholds()
ENHANCEMENT = EnhancementDefaults()
SUPER_RESOLUTION = SuperResolutionSettings()
FIXED_PIPELINE = FixedPipelineSettings()
EVALUATION = EvaluationSettings()
APP = AppSettings()
