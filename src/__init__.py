"""Smart Multimedia Image Enhancement Assistant - core package.

Package layout mirrors the five-stage architecture of the Review 1 report:

    utils/        Stage 1  Image ingestion (loading, validation, colour conversion)
    analysis/     Stage 2  Quality analysis (contrast, blur, noise, resolution)
    decision/     Stage 3  Rule-based decision engine
    enhancement/  Stage 4  Classical operators + optional Real-ESRGAN
    pipeline/     Stage 4  Adaptive / fixed / no-enhancement pipelines
    evaluation/   Stage 5  PSNR / SSIM and quality comparison
    visualization/ Stage 5 Matplotlib figures (histograms, comparison charts)
"""

__version__ = "1.0.0"
