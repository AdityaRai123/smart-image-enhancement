# Smart Multimedia Image Enhancement Assistant
### An AI-Based Adaptive Enhancement System

BITE314L Multimedia Systems course project (VIT, Fall 2026-27). This repository is the
implementation of the system proposed in the Review 1 report.

**Live demo:** <https://smart-image-enhancement.streamlit.app>

The hosted demo runs the full classical pipeline. The optional Real-ESRGAN stage needs
PyTorch, so it only runs when you run the app locally (see [Installation](#7-installation)
and [Real-ESRGAN setup](#9-real-esrgan-setup)). Free hosted apps sleep when unused, so the
first visit may take about a minute to wake it.

---

## 1. Project description

Upload an image and the system:

1. **measures** what is wrong with it: histogram spread/skew (contrast), Laplacian variance
   (blur), local-standard-deviation noise estimate, and pixel dimensions (resolution);
2. passes those statistics to an explicit **rule-based decision engine**, which selects the
   *minimal* set of classical operations and the *order* to run them in, and explains every
   choice with the measured value and threshold behind it;
3. **executes** the plan with classical OpenCV/NumPy operators (contrast stretching,
   histogram equalisation (including a from-scratch version), CLAHE, Gaussian/median/bilateral
   denoising, unsharp masking), plus **optional Real-ESRGAN super-resolution** used only for
   images flagged as low-resolution;
4. **evaluates** the result with PSNR and SSIM (scikit-image) and compares three pipelines:
   no enhancement, a fixed classical pipeline, and the proposed adaptive pipeline.

Everything runs on an ordinary CPU. Without a GPU, one 512×512 image is analysed,
enhanced and evaluated in well under a second.

## 2. Research problem

Real-world images suffer from a *mix* of degradations (low contrast, sensor noise, blur,
insufficient resolution), and how severe each one is varies from image to image. Freely
available tools either apply one fixed filter chain to every input or expose manual sliders
that need photographic expertise. A fixed chain is rarely right: an already-sharp image gains
nothing from unsharp masking and picks up halos instead. Deep models such as GAN
super-resolution are powerful but expensive, and they work as black boxes.

## 3. Research gap

1. Classical operators (CLAHE, sharpening, denoising) are mature but rarely **composed
   adaptively** in open, general-purpose tools.
2. AI super-resolution (Real-ESRGAN) is usually a standalone tool, not **one option inside a
   quality-aware pipeline**.
3. Existing adaptive pipelines are domain-specific (OCR, underwater, satellite) and are not
   evaluated as general-purpose systems with **transparent, reproducible metrics**.

## 4. Objectives (from the report) and where they are implemented

| Objective | Implementation |
|---|---|
| Classical modules with explicit, inspectable parameters | `src/enhancement/*.py`, parameters in `config/config.py` |
| From-scratch histogram equalisation | `src/enhancement/histogram_equalization.py` (`equalize_gray_from_scratch`) |
| Rule-based decision engine (spread, Laplacian variance, noise → subset + order) | `src/decision/decision_engine.py` |
| Optional Real-ESRGAN for low-resolution images, CPU-friendly | `src/enhancement/super_resolution.py`, `realesrgan_arch.py` |
| PSNR + SSIM, before/after comparison in Streamlit | `src/evaluation/metrics.py`, `app.py` |
| Compare no enhancement / fixed / adaptive on an evaluation set | `src/evaluation/comparison.py`, `scripts/evaluate_dataset.py` |
| Document design and limitations | this README |

## 5. Architecture – the five stages

```
Stage 1  Image Ingestion        src/utils/image_utils.py       validate, decode, EXIF-rotate, flatten alpha,
   |                                                            16-bit -> 8-bit, downscale > 2048 px (CPU guard)
Stage 2  Quality Analysis       src/analysis/                  contrast.py  blur.py  noise.py  resolution.py
   |                                                            -> feature dictionary
Stage 3  Decision Engine        src/decision/decision_engine.py  rules -> ordered operations + reasons + rule trace
   |
Stage 4  Enhancement Execution  src/pipeline/ + src/enhancement/ classical operators, optional Real-ESRGAN
   |
Stage 5  Evaluation & Presentation  src/evaluation/ + src/visualization/ + app.py
                                    PSNR/SSIM, histograms, before/after, baseline comparison
```

The Streamlit page has the same five sections, and a live five-stage tracker shows which
stage is running and how long each one took.

Stage 2 hands Stage 3 a plain dictionary, and Stage 3 is a pure function of it:

```python
features = analyze_image(rgb).features()
# {"histogram_spread": 0.44, "contrast_score": 0.10, "histogram_skewness": -0.2,
#  "blur_score": 8.3, "noise_score": 9.7, "impulse_ratio": 0.0,
#  "width": 512, "height": 512, "resolution_flag": False, ...}
result = decide(features)
result.to_dict()  # {"diagnosis": {...}, "operations": [...], "reasoning": [...], "parameters": {...}, "rules": [...]}
```

## 6. Technologies

| Category | Tools |
|---|---|
| Language & UI | Python 3, Streamlit |
| Classical processing | OpenCV (filters, CLAHE, Sobel, Canny, colour conversion RGB/YCrCb/HSV/Gray), NumPy, Pillow |
| Quality assessment | scikit-image (PSNR, SSIM), Matplotlib (histograms, charts) |
| AI enhancement | Real-ESRGAN via PyTorch (optional), rembg background removal (optional extra) |
| Testing | pytest, Streamlit `AppTest` |

## 7. Installation

Python 3.10 or newer (tested with Python 3.14.5 on Windows 11).

**Windows (PowerShell or cmd)**

```bash
cd smart-image-enhancement
python -m venv venv
```

(If `python` opens the Microsoft Store, use the launcher instead: `py -m venv venv`.)

```bash
venv\Scripts\activate
```

(PowerShell may block `activate` scripts; you can skip activation and call
`venv\Scripts\python.exe` directly in every command below.)

```bash
pip install -r requirements.txt
```

**Optional AI super-resolution (Real-ESRGAN, CPU):**

```bash
pip install -r requirements-ai.txt
```

**Linux / macOS:** same steps, but activate with `source venv/bin/activate`.

## 8. Running the application

```bash
streamlit run app.py
```

Then open http://localhost:8501. Without activating the venv on Windows:
`venv\Scripts\python.exe -m streamlit run app.py`.

In the sidebar:

* **Upload an image** (JPG/PNG/WebP/BMP/TIFF, ≤ 25 MB). You can also add an optional
  ground-truth reference.
* Or **Built-in sample + synthetic degradation**: a clean sample photo is degraded in a
  controlled way (low contrast, noise, impulse noise, blur, low resolution, mixed, or "good",
  meaning untouched) and the clean original is used as the ground truth for PSNR/SSIM. This
  mode is best for demonstrations.
* **Pipeline**: choose which output (adaptive / fixed / none) the before/after view shows.
  All three are always computed for the comparison table.
* **Enable AI super-resolution (Real-ESRGAN)**: allows the AI stage for images flagged as
  low-resolution.
* **Advanced settings**: override the decision thresholds for the current session to see how
  the decisions react.

## 9. Real-ESRGAN setup

* `pip install -r requirements-ai.txt` installs the CPU build of PyTorch (~200 MB).
* The first time you tick *Enable AI super-resolution* on a low-resolution image, the official
  weights `realesr-general-x4v3.pth` (4.7 MB, the **lightweight** compact model) are
  downloaded from the [xinntao/Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)
  into `models/`. You can also download them manually and place them there:
  * <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth> (default)
  * <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth> (optional, 64 MB, much slower on CPU)
* The `basicsr`/`realesrgan` pip packages are **not** used, because they break with current
  torchvision. The two generator architectures are defined in
  `src/enhancement/realesrgan_arch.py` with the official parameter names, and the weights load
  with `strict=True`.
* CPU safeguards: inputs above 400×400 px are downscaled before inference (the report's
  "downscaled crop"), inference is tiled (200 px tiles), and output is capped at 2048 px.
  Measured on a laptop CPU: 128×128 → 512×512 in ~50 ms, 400×400 → 1600×1600 in ~0.55 s.
* **If PyTorch or the weights are unavailable**, nothing crashes. The sidebar says why, and
  low-resolution images are upscaled with Lanczos interpolation, labelled everywhere as
  *"classical fallback, not AI"*.

## 10. Dataset generation

```bash
python scripts/create_test_dataset.py
```

* Clean images are read from `data/clean/`. If that folder is empty, the six bundled
  scikit-image sample photos (astronaut, coffee, chelsea, rocket, camera,
  immunohistochemistry) are exported there first, so it works offline and is reproducible.
  You can put your own clean photos, or a DIV2K subset, in `data/clean/`.
* The script writes `data/test/<category>/<name>.png`, the ground truth in
  `data/test/reference/<name>.png`, and `data/test/manifest.csv` (recipe and expected defects).
* Categories (`src/utils/degradation.py`):

| Category | Degradation | Expected diagnosis |
|---|---|---|
| good | none | nothing |
| low_contrast | linear contrast compression ×0.45 | contrast |
| noisy | Gaussian noise σ = 15 | noise |
| impulse_noise | salt-and-pepper 3 % | noise |
| blurry | Gaussian blur σ = 2 px | blur |
| low_resolution | 4× area downsampling | resolution |
| mixed | blur σ 1.5 + contrast ×0.55 + noise σ 10 | contrast, noise, blur |

## 11. Evaluation

```bash
python scripts/evaluate_dataset.py
```

Options: `--sr off` (no upscaling), `--sr fallback` (Lanczos), `--sr ai` (require
Real-ESRGAN), `--save-images`, `--data-dir`, `--output`.

Outputs in `outputs/evaluation/`: `results.csv` (one row per image × pipeline),
`summary.csv`, `detection.csv` (diagnosis accuracy), `summary.md`, `psnr_by_category.png`,
`ssim_by_category.png`.

Threshold calibration (the report's "refine and validate the thresholds"):

```bash
python scripts/analyze_thresholds.py
```

This writes per-category distributions of every decision statistic, next to the current
threshold, to `outputs/threshold_analysis/` (Markdown report plus strip plots).

### Results obtained on this machine

These numbers come from the **synthetic** evaluation set above (6 clean photos × 7
categories). They were produced by `scripts/evaluate_dataset.py` on an Intel laptop CPU with
Real-ESRGAN enabled, measured against the clean ground truth, and they change if you change
the data or thresholds. Re-run the script to regenerate them.

| Category | No enhancement PSNR / SSIM | Fixed pipeline | **Adaptive (proposed)** |
|---|---|---|---|
| good | ∞ (60*) / 1.000 | 19.30 / 0.774 | **∞ (60*) / 1.000**: untouched |
| low_contrast | 18.20 / 0.782 | 19.03 / 0.842 | **29.29 / 0.942** |
| noisy | 24.83 / 0.485 | 15.86 / 0.310 | **31.86 / 0.839** |
| impulse_noise | 20.16 / 0.517 | 16.67 / 0.437 | **31.50 / 0.900** |
| blurry | 26.79 / 0.762 | 20.49 / 0.724 | **28.07 / 0.798** |
| mixed | 18.63 / 0.363 | 17.11 / 0.286 | 18.55 / **0.580** |
| low_resolution | 27.13 / 0.771 (bicubic) | 17.50 / 0.673 | 25.71 / 0.746 (Real-ESRGAN, see §16) |
| **overall mean** | 27.96 / 0.669 | 18.00 / 0.578 | **32.14 / 0.829** |

\* identical images have infinite PSNR; it is capped at 60 dB for averaging.

Diagnosis accuracy (detected defect set = injected defect set): good 6/6, low_contrast 6/6,
noisy 6/6, impulse_noise 6/6, blurry 6/6, low_resolution 5/6, mixed 4/6 (**39/42**). The three
misses are discussed under *Limitations*.

## 12. Project structure

```
smart-image-enhancement/
├── app.py                      Streamlit entry point (wiring only)
├── requirements.txt            core dependencies (CPU)
├── requirements-ai.txt         optional PyTorch for Real-ESRGAN
├── pytest.ini
├── .streamlit/config.toml      theme, upload limit
├── config/
│   └── config.py               ALL thresholds and parameters, with justification
├── src/
│   ├── utils/                  Stage 1: image_utils.py (ingestion), degradation.py, samples.py
│   ├── analysis/               Stage 2: contrast.py, blur.py, noise.py, resolution.py, quality_analyzer.py
│   ├── decision/               Stage 3: decision_engine.py
│   ├── enhancement/            Stage 4: contrast_stretch.py, histogram_equalization.py, clahe.py,
│   │                                    denoise.py, sharpening.py, super_resolution.py,
│   │                                    realesrgan_arch.py, background_removal.py
│   ├── pipeline/               Stage 4: executor.py, adaptive_pipeline.py, fixed_pipeline.py
│   ├── evaluation/             Stage 5: metrics.py, comparison.py, dataset.py
│   ├── visualization/          Stage 5: plots.py (Matplotlib)
│   └── ui/                     Streamlit sidebar, session/caching, section renderers
├── scripts/
│   ├── create_test_dataset.py
│   ├── evaluate_dataset.py
│   └── analyze_thresholds.py
├── tests/                      95 tests (analysis, enhancement, decision engine, metrics,
│                               ingestion/error handling, pipelines, headless app tests)
├── data/                       clean/ and test/ (generated)
├── models/                     Real-ESRGAN weights (downloaded on demand)
└── outputs/                    evaluation and threshold-analysis results
```

## 13. Decision-engine logic

### How each statistic is measured (Stage 2)

* **Histogram spread** = (P99 − P1) / 255 of the luminance Y (YCrCb). This is the fraction of
  the range used by the central 98 % of pixels. **RMS contrast** = std(Y)/255.
  **Skewness** = third standardised moment.
* **Blur score** (Laplacian variance). The variance of `cv2.Laplacian` is the primary
  measure. Validation exposed two confounders, and both are removed explicitly:
  * *contrast*: the Laplacian is linear, so variance scales with contrast². The score is
    multiplied by (255 / (P99 − P1))², i.e. measured as if the histogram were stretched;
  * *noise*: noise has strong second derivatives, so a noisy **blurry** image had a raw
    variance of ~900, which reads as "sharp". The luminance is therefore lightly smoothed
    first (σ = 1 px, i.e. a Laplacian of Gaussian), and when noise is detected the exact
    expected noise term E·σ² is subtracted (E is computed from the discrete kernel; it is 20
    for the raw 3×3 Laplacian).

  The classic raw Laplacian variance is still shown in the UI for transparency.
* **Noise σ** (local standard deviation). Each channel is filtered with Immerkær's
  structure-cancelling kernel `[[1,-2,1],[-2,4,-2],[1,-2,1]]/6` (unit gain for white noise).
  The local std is taken in 7×7 windows, keeping only homogeneous windows (lowest 20 % Sobel
  gradient, no clipped pixels), and the result is the median over those windows, averaged
  over R, G, B. Measuring per channel matters: luminance averaging hides ⅓ of independent
  per-channel noise (√(0.299² + 0.587² + 0.114²) = 0.668). Accuracy: true σ 5/10/15 →
  measured 4.9–5.2 / 9.3–9.9 / 13.8–14.7.
* **Impulse ratio**: share of near-black/near-white pixels that differ from their 3×3 median
  by more than 60 levels. When impulse noise is present, all other statistics are measured
  after a *switching median* that replaces only those pixels, so the outliers do not set the
  histogram percentiles or dominate the Laplacian.
* **Resolution flag**: shorter side < 256 px OR fewer than 100 000 pixels.

### Rules (Stage 3)

| Rule | Condition | Action | Why |
|---|---|---|---|
| N1 | impulse ratio > 0.4 % | **median** 3×3 (5×5 above 20 %) | rank filter removes outliers completely; Gaussian/bilateral only smear them |
| N2a | noise σ > 5 **and** blur score < 30 | **Gaussian** (σ = 0.12·noise, 0.8–1.3) | no sharp edges left to preserve; bilateral leaves blotchy noise when edge and noise amplitudes are similar |
| N2b | noise σ > 5 and image sharp | **bilateral** (σ_color = 4·noise, d = 7, or 9 if σ ≥ 10) | edge-preserving; 4·σ was the best range sigma at every tested level |
| C1 | spread < 0.52 | **contrast stretching** (P0.5–P99.5 → 0–255, same gain on R,G,B) | range is compressed but the histogram shape is fine → linear remap, no colour cast |
| C2a | RMS contrast after stretching < 0.10 and \|skew\| > 1 | **histogram equalisation (custom)** on Y | pixel mass piled at one end; only a CDF remap redistributes it |
| C2b | RMS contrast after stretching < 0.10, \|skew\| ≤ 1 | **CLAHE** (clip 2, or 3 if spread < 0.35) | flat/hazy content → local, clip-limited equalisation |
| S1 | blur score < 30 | **unsharp mask** (σ 1.5, amount 1.0; strong σ 2, amount 1.5 if < 10) | lacks high-frequency detail; skipped otherwise (avoids halos) |
| R1 | low-resolution flag | **Real-ESRGAN ×4** if enabled (Lanczos if unavailable); otherwise a recommendation | AI stage reserved for the one case it is needed |

**Order:** impulse removal → denoising → contrast → sharpening → super-resolution. Every
later step amplifies leftover noise, so denoising comes first. Sharpening acts on the final
tonal range. SR runs last, on the small corrected image (cheap). If Real-ESRGAN will run, N2
(Gaussian-type noise) and S1 are left to the network, because its training degradations
include blur and noise.

**Ordering evidence** (mixed category, same three operators, bilateral + stretch + unsharp
mask, mean PSNR over the 6 references): denoise→stretch→sharpen 19.96 dB >
stretch→denoise→sharpen 19.72 dB > stretch→sharpen→denoise 18.70 dB. The fixed
CLAHE→bilateral→USM order scored 17.11 dB, below doing nothing (18.63 dB).

### Why these thresholds (validated with `scripts/analyze_thresholds.py`)

| Threshold | Value | Clean references | Degraded group |
|---|---|---|---|
| contrast_low_spread | 0.52 | ≥ 0.60 | low_contrast ≤ 0.44 (midpoint of the gap) |
| blur_threshold | 30 | ≥ 51.9 | blurry ≤ 10.9, mixed ≤ 19.7 (≈ geometric midpoint) |
| noise_medium_sigma | 5 | ≤ 1.7 | σ=5 noise reads ≥ 4.85 (grain becomes visible) |
| impulse_ratio_threshold | 0.4 % | ≤ 0.043 % | 3 % salt-and-pepper reads ≥ 2.07 % |
| contrast_low_rms | 0.10 | ≥ 0.119 | std of ~25 grey levels |

Operator parameters were chosen the same way: each candidate setting was used to restore
the clean references from controlled degradations, and the best mean PSNR/SSIM was kept.
Examples: stretch percentiles 0.5/99.5 beat 1/99 (29.3 vs 25.1 dB); global HE on linear
contrast loss scored *below doing nothing* (16.4 vs 18.2 dB), which is why HE is reserved
for distribution problems (rule C2a); median 3×3 beat 5×5 on impulse noise up to 10 %
density.

All values are in `config/config.py`, with comments. Change them there, or temporarily in the
app's *Advanced settings*.

## 14. Evaluation protocol and the PSNR/SSIM caveat

The report evaluates the enhanced image against the **original upload**. The app does that
whenever no ground truth exists, with a visible warning: against a degraded original,
PSNR/SSIM measure *how much the image changed*, not whether it improved, and "no
enhancement" always scores ∞ / 1.0. `compare_images()` takes an optional **ground-truth
reference** (`reference_kind = "ground_truth"`), which the sample mode, the optional
reference upload, and the evaluation scripts all use.

## 15. Testing

```bash
python -m pytest
```

95 tests cover:

* the from-scratch HE (matches `cv2.equalizeHist` within 1 grey level), contrast, Laplacian
  variance and noise estimation accuracy, impulse detection and resolution flags;
* every decision rule, ordering and option, with JSON serialisation;
* enhancement operators;
* PSNR/SSIM (cross-checked against scikit-image);
* ingestion errors: corrupted, truncated, empty, wrong format, tiny, huge, RGBA, palette,
  16-bit, grayscale;
* a failing pipeline step not crashing the run, and the Real-ESRGAN load/fallback paths;
* headless runs of the Streamlit app (all five stages, good image, fixed view,
  low-resolution with SR).

## 16. Limitations

* **Synthetic evaluation.** The quantitative results use six clean scikit-image photos with
  synthetic degradations. Real camera noise is signal-dependent and spatially correlated, and
  real blur is not Gaussian. The thresholds should be re-checked on real photos (DIV2K/RealSR
  subsets fit into `data/clean/`).
* **GAN super-resolution vs PSNR.** Real-ESRGAN scores *lower* PSNR/SSIM than bicubic
  interpolation (25.7 vs 27.1 dB). It synthesises plausible texture that is not
  pixel-identical to the ground truth (the perception–distortion trade-off, Blau & Michaeli,
  CVPR 2018), and the compact model is trained on heavy real-world degradations, not on clean
  downsampling. Its benefit is perceptual sharpness (see `blur_score_after` in
  `results.csv`), which PSNR/SSIM do not reward.
* **Texture vs noise.** Fine texture in downsampled images can read as noise (one false alarm,
  σ ≈ 6.2). When Real-ESRGAN runs, classical denoising is skipped anyway.
* **Borderline contrast.** In the mixed category, two images (spread 0.53/0.56) sit just above
  the 0.52 threshold, partly because added noise widens the percentile range, so they are not
  stretched.
* **"Full range" may overshoot the reference.** Stretching a flat-looking image to the full
  range is the textbook goal. When the clean reference was itself low-contrast (chelsea,
  immunohistochemistry), this lowers PSNR against it, even though the image is "enhanced".
* **Hard thresholds.** Rules switch at fixed values. An image just above a threshold gets no
  correction, and one just below gets the full operation.
* Gaussian-type noise σ is estimated assuming locally white noise; JPEG artefacts are not
  modelled separately.

## 17. Future improvements

* Validate thresholds on real degraded photos (DIV2K / RealSR subsets) and report confidence
  intervals.
* Soft rules: scale operator strength continuously with the distance from the threshold.
* Closed-loop refinement: re-analyse after each step and stop when statistics are within
  range.
* No-reference quality metrics (BRISQUE/NIQE) to complement PSNR/SSIM for SR outputs.
* JPEG-artefact detection (blockiness) and a deblocking operator; low-light/gamma rule.
* A denoising-strength-controllable SR model (`realesr-general-wdn-x4v3` interpolation).

---

### Quick demo script for Review 2

| Question | Where to show it |
|---|---|
| *Why did your system choose this filter?* | Stage 3 → "Why did the system choose these operations?" and the rule-trace table (measured value vs threshold) |
| *What happens if the image is already good?* | Sample mode → degradation **good**: "No enhancement needed"; switch the view to *Fixed* to see it degrade the image |
| *Difference from just applying CLAHE + sharpening?* | Baseline comparison table/charts (ground truth in sample mode); `outputs/evaluation/summary.md` |
| *Where is the AI?* | Sample mode → **low resolution** + tick *Enable AI super-resolution*: Real-ESRGAN runs only because the resolution flag fired |
| *What is your contribution?* | The quality-aware decision layer: measured statistics → minimal, ordered, explained operator set, with optional AI SR, evaluated against a fixed pipeline |
