"""Stage 3: rule-based decision engine (pure function of the feature dictionary)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from config import DECISION
from src.decision import DecisionOptions, decide

GOOD = {
    "histogram_spread": 0.90,
    "contrast_score": 0.25,
    "histogram_skewness": 0.1,
    "blur_score": 80.0,
    "laplacian_variance_raw": 900.0,
    "noise_score": 1.0,
    "impulse_ratio": 0.0,
    "width": 800,
    "height": 600,
    "resolution_flag": False,
}


def features(**overrides):
    return {**GOOD, **overrides}


def keys(result):
    return [op.key for op in result.operations]


def test_good_image_gets_no_processing():
    result = decide(features())
    assert result.no_enhancement_needed
    assert keys(result) == []
    assert "NO enhancement" in result.reasoning[0]
    assert all(not d.needs_action for d in result.diagnosis.values())
    assert any("Sharpening skipped" in s for s in result.skipped)


def test_low_contrast_triggers_stretching():
    result = decide(features(histogram_spread=0.40, contrast_score=0.10))
    assert keys(result) == ["contrast_stretch"]
    assert result.diagnosis["contrast"].level == "Low"


def test_very_low_contrast_label_and_larger_gain():
    result = decide(features(histogram_spread=0.30, contrast_score=0.08))
    assert result.diagnosis["contrast"].level == "Very low"
    assert "contrast_stretch" in keys(result)


def test_flat_distribution_triggers_clahe_or_he():
    # range is fine, but pixels are bunched (low RMS) -> equalisation
    clahe_result = decide(features(histogram_spread=0.7, contrast_score=0.05, histogram_skewness=0.2))
    assert keys(clahe_result) == ["clahe"]
    he_result = decide(features(histogram_spread=0.7, contrast_score=0.05, histogram_skewness=2.5))
    assert keys(he_result) == ["histogram_equalization"]


def test_high_noise_sharp_image_uses_bilateral():
    result = decide(features(noise_score=14.0))
    assert keys(result) == ["bilateral_denoise"]
    params = result.parameters["bilateral_denoise"]
    assert params["sigma_color"] == pytest.approx(56.0)  # 4 x sigma
    assert params["diameter"] == 9  # high noise -> larger diameter


def test_noisy_and_soft_image_uses_gaussian_then_sharpening():
    result = decide(features(noise_score=9.0, blur_score=12.0))
    assert keys(result) == ["gaussian_denoise", "unsharp_mask"]
    # denoise BEFORE sharpening, explained in the reasoning
    assert any("AFTER denoising" in op.reason for op in result.operations)
    assert result.parameters["unsharp_mask"]["threshold"] == 3.0


def test_impulse_noise_uses_median():
    result = decide(features(impulse_ratio=0.03, noise_score=20.0))
    assert keys(result) == ["median_denoise"]
    assert result.parameters["median_denoise"]["ksize"] == 3
    assert decide(features(impulse_ratio=0.25)).parameters["median_denoise"]["ksize"] == 5


def test_blur_strength_levels():
    mild = decide(features(blur_score=20.0))
    strong = decide(features(blur_score=5.0))
    assert mild.diagnosis["sharpness"].level == "Moderate"
    assert strong.diagnosis["sharpness"].level == "Blurry"
    assert strong.parameters["unsharp_mask"]["amount"] > mild.parameters["unsharp_mask"]["amount"]


def test_operation_order_for_mixed_degradation():
    result = decide(features(histogram_spread=0.35, contrast_score=0.06, noise_score=9.0, blur_score=8.0))
    assert keys(result) == ["gaussian_denoise", "contrast_stretch", "unsharp_mask"]


def test_low_resolution_recommendation_when_sr_disabled():
    result = decide(features(width=128, height=128, resolution_flag=True))
    assert keys(result) == []
    assert result.diagnosis["resolution"].level == "Low"
    assert result.recommendations and "super-resolution" in result.recommendations[0]


def test_low_resolution_with_ai_sr_defers_denoise_and_sharpen():
    opts = DecisionOptions(sr_enabled=True, sr_ai_available=True)
    result = decide(features(width=128, height=128, resolution_flag=True, noise_score=9.0, blur_score=20.0), opts)
    assert keys(result) == ["super_resolution"]
    assert result.parameters["super_resolution"]["method"] == "realesrgan"
    assert any("Real-ESRGAN" in s for s in result.skipped)


def test_low_resolution_fallback_when_ai_unavailable():
    opts = DecisionOptions(sr_enabled=True, sr_ai_available=False)
    result = decide(features(width=128, height=128, resolution_flag=True, histogram_spread=0.4), opts)
    assert keys(result) == ["contrast_stretch", "super_resolution"]  # SR runs last
    assert result.parameters["super_resolution"]["method"] == "lanczos"


def test_rule_trace_covers_every_rule():
    result = decide(features())
    ids = {r.rule_id for r in result.rules}
    assert {"N1", "N2", "C1", "C2", "S1", "R1"} <= ids
    assert all(r.measured for r in result.rules)


def test_custom_thresholds_change_decision():
    strict = replace(DECISION, blur_threshold=100.0)
    assert keys(decide(features(blur_score=80.0), thresholds=strict)) == ["unsharp_mask"]


def test_output_is_json_serialisable():
    import json

    result = decide(features(noise_score=12.0, blur_score=5.0))
    payload = json.dumps(result.to_dict())
    for key in ("diagnosis", "operations", "reasoning", "parameters"):
        assert key in json.loads(payload)


@pytest.mark.parametrize("bad", [{"blur_score": float("nan")}, {"noise_score": float("inf")}])
def test_invalid_features_raise(bad):
    with pytest.raises(ValueError):
        decide(features(**bad))


def test_missing_feature_raises():
    incomplete = features()
    incomplete.pop("blur_score")
    with pytest.raises(ValueError, match="missing"):
        decide(incomplete)
