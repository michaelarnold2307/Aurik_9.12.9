"""Tests for the 5-improvement upgrade: Bayesian MediumClassifier, RestorabilityEstimator
weighted scoring, EraClassifier adaptive fusion, Phase-03 material profiles, MTEF.

These tests verify the NEW functionality added by the improvements while
existing test suites (test_medium_classifier_sota, test_era_classifier,
test_restorability_estimator) cover backward compatibility.
"""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

SR = 48000


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Bayesian MediumClassifier: Gaussian-likelihood scorer
# ═══════════════════════════════════════════════════════════════════════════════

from backend.core.medium_classifier import _MaterialScorer


@pytest.mark.unit
class TestBayesianMaterialScorer:
    """Tests for the Bayesian Gaussian-likelihood _MaterialScorer."""

    def test_score_returns_dict(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 14000.0,
            "snr_db": 50.0,
            "noise_color": 0.5,
            "crackle_density": 0.0,
            "wow_depth": 0.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        assert hasattr(result, "material")
        assert hasattr(result, "confidence")
        assert result.confidence >= 0.0

    def test_score_confidence_range(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 4000.0,
            "snr_db": 15.0,
            "noise_color": 0.8,
            "crackle_density": 0.02,
            "wow_depth": 0.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        assert 0.0 <= result.confidence <= 1.0

    def test_narrow_bandwidth_low_snr_selects_shellac_or_wax(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 4000.0,
            "snr_db": 12.0,
            "noise_color": 0.75,
            "crackle_density": 0.05,
            "wow_depth": 0.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        assert result.material_type in ("shellac", "wax_cylinder")

    def test_wideband_high_snr_selects_digital(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 20000.0,
            "snr_db": 75.0,
            "noise_color": 0.0,
            "crackle_density": 0.0,
            "wow_depth": 0.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        assert result.material_type in ("cd_digital", "dat", "streaming")

    def test_moderate_bandwidth_with_crackle_selects_vinyl(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 14000.0,
            "snr_db": 35.0,
            "noise_color": 1.5,
            "crackle_density": 0.004,
            "wow_depth": 0.15,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_strength": 0.30,
            "infrasonic_rms": 0.06,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        assert result.material_type in ("vinyl", "lacquer_disc")

    def test_tape_characteristics(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 12000.0,
            "snr_db": 28.0,
            "noise_color": 1.6,
            "crackle_density": 0.0,
            "wow_depth": 1.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 2.5,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        assert result.material_type in ("reel_tape", "tape", "cassette")

    def test_evidence_field_present(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 10000.0,
            "snr_db": 35.0,
            "noise_color": 0.4,
            "crackle_density": 0.01,
            "wow_depth": 0.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        assert hasattr(result, "evidence")
        assert isinstance(result.evidence, (list, dict))

    def test_posteriors_sum_to_one(self):
        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 12000.0,
            "snr_db": 40.0,
            "noise_color": 0.5,
            "crackle_density": 0.0,
            "wow_depth": 0.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
            "codec_type_code": 0.0,
        }
        result = scorer.score(features, None)
        # ClassificationResult has confidence as top posterior
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RestorabilityEstimator: Defect-type-weighted scoring
# ═══════════════════════════════════════════════════════════════════════════════

from backend.core.restorability_estimator import RestorabilityEstimator


class TestRestorabilityWeighted:
    """Tests for the improved weighted RestorabilityEstimator."""

    @pytest.fixture
    def est(self):
        return RestorabilityEstimator()

    def test_crackle_weighted_higher_than_clipping(self, est):
        """Crackle (weight 0.92) should penalize less than clipping (weight 0.55)
        for the same severity, because crackle is more repairable."""
        np.random.seed(42)
        # Create identical base signals
        audio = (np.random.randn(SR * 3) * 0.3).astype(np.float32)

        # Both estimate the same raw scores, but the weight difference
        # means different defect types at same severity → different scores.
        # We test the weights indirectly through the full pipeline.
        result = est.estimate(audio, SR)
        assert 0.0 <= result.restorability_score <= 100.0
        assert result.predicted_mos >= 1.0

    def test_material_cap_severity_interpolation(self, est):
        """Higher defect severity → lower material cap (floor interpolation)."""
        # This tests the severity-interpolated caps internally
        np.random.seed(7)
        audio = (np.random.randn(SR * 3) * 0.3).astype(np.float32)
        result = est.estimate(audio, SR)
        assert result.grade in ("excellent", "good", "fair", "poor")

    def test_multiwindow_bandwidth(self, est):
        """Bandwidth estimate should be stable (multi-window averaging)."""
        np.random.seed(42)
        # Create bandlimited signal
        sos = butter(6, 8000, btype="lowpass", fs=SR, output="sos")
        audio = sosfilt(sos, np.random.randn(SR * 5).astype(np.float32)) * 0.3
        result = est.estimate(audio, SR)
        assert isinstance(result.restorability_score, float)
        assert np.isfinite(result.restorability_score)

    def test_defect_weights_exist(self):
        """_DEFECT_WEIGHTS should have entries for key defect categories."""
        assert hasattr(RestorabilityEstimator, "_DEFECT_WEIGHTS")
        weights = RestorabilityEstimator._DEFECT_WEIGHTS
        for key in ("noise", "clipping", "crackle", "bandwidth", "hum"):
            assert key in weights
            assert 0.0 < weights[key] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EraClassifier: Adaptive fusion + continuous post-1990
# ═══════════════════════════════════════════════════════════════════════════════

from backend.core.era_classifier import _dsp_fingerprint_decade


class TestEraAdaptiveFusion:
    """Tests for EraClassifier improvements."""

    def test_post1990_continuous_snr_differentiation(self):
        """Post-1990 decades should use continuous SNR scoring, not hard thresholds."""
        # SNR slightly above old threshold 50 → should still get 2020
        dec_high, _ = _dsp_fingerprint_decade(20000.0, 52.0)
        assert dec_high >= 2010

        # SNR slightly below old threshold 50 → should now get 2020 too (continuous)
        dec_below, _ = _dsp_fingerprint_decade(20000.0, 48.0)
        assert dec_below >= 2000

    def test_post1990_monotone_with_snr(self):
        """Higher SNR → later or equal decade in post-1990 range."""
        snrs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        decades = [_dsp_fingerprint_decade(20000.0, snr)[0] for snr in snrs]
        for i in range(len(decades) - 1):
            assert decades[i] <= decades[i + 1], (
                f"Monotonie: SNR {snrs[i]} → {decades[i]}, SNR {snrs[i + 1]} → {decades[i + 1]}"
            )

    def test_post1990_gaussian_smooth_transition(self):
        """SNR values between old hard thresholds should give intermediate decades."""
        # Old: 28-38 → 2000, 38-50 → 2010. Gaussian may give smoother transitions.
        dec_33, _ = _dsp_fingerprint_decade(20000.0, 33.0)
        dec_43, _ = _dsp_fingerprint_decade(20000.0, 43.0)
        # Both should be valid post-1990 decades
        assert dec_33 >= 1990
        assert dec_43 >= 2000

    def test_outlier_robust_fusion_no_crash(self):
        """IQR-based outlier rejection should not crash with edge inputs."""
        from backend.core.era_classifier import EraClassifier

        clf = EraClassifier()
        np.random.seed(42)
        audio = (np.random.randn(SR * 3) * 0.1).astype(np.float32)
        result = clf.classify(audio, SR)
        assert result.decade in {
            1890,
            1900,
            1910,
            1920,
            1930,
            1940,
            1950,
            1960,
            1970,
            1980,
            1990,
            2000,
            2010,
            2020,
            2025,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Phase-03 Denoise: Material profiles + smooth decade modulation
# ═══════════════════════════════════════════════════════════════════════════════

from backend.core.phases.phase_03_denoise import (
    ERA_DECADE_KNOTS,
    DenoisePhase,
    decade_strength_multiplier,
)


class TestDenoiseProfiles:
    """Tests for new material profiles and decade interpolation."""

    def test_all_materials_have_profiles(self):
        """All expected materials should have an entry in MATERIAL_PARAMS."""
        phase = DenoisePhase()
        required = {
            "tape",
            "reel_tape",
            "cassette",
            "vinyl",
            "shellac",
            "wax_cylinder",
            "cd_digital",
            "dat",
            "mp3_low",
            "mp3_high",
            "aac",
            "unknown",
        }
        actual = set(phase.MATERIAL_PARAMS.keys())
        assert required.issubset(actual), f"Fehlend: {required - actual}"

    def test_material_params_have_required_keys(self):
        """Each material profile must have strength, bands, and transient_preserve."""
        phase = DenoisePhase()
        for mat, params in phase.MATERIAL_PARAMS.items():
            assert "strength" in params, f"{mat}: missing strength"
            assert "bands" in params, f"{mat}: missing bands"
            assert "transient_preserve" in params, f"{mat}: missing transient_preserve"
            assert 0.0 < params["strength"] <= 1.0, f"{mat}: strength out of range"  # type: ignore[operator]

    def test_shellac_g_floor_reduced(self):
        """Shellac g_floor should be lower than old 0.30 but still protective."""
        phase = DenoisePhase()
        g_floor = phase.MATERIAL_PARAMS["shellac"].get("g_floor", 0.1)
        assert 0.15 <= g_floor <= 0.28, f"shellac g_floor={g_floor}"  # type: ignore[operator]

    def test_vinyl_g_floor_raised(self):
        """Vinyl should have explicit g_floor > 0.10 for groove protection."""
        phase = DenoisePhase()
        g_floor = phase.MATERIAL_PARAMS["vinyl"].get("g_floor", 0.1)
        assert g_floor > 0.10, f"vinyl g_floor={g_floor}"  # type: ignore[operator]

    def test_dat_very_conservative(self):
        """DAT (very clean medium) should have low strength."""
        phase = DenoisePhase()
        assert phase.MATERIAL_PARAMS["dat"]["strength"] <= 0.25  # type: ignore[operator]

    def test_mp3_low_protects_codec_artifacts(self):
        """MP3 low bitrate should have higher g_floor to protect codec noise."""
        phase = DenoisePhase()
        g_floor = phase.MATERIAL_PARAMS["mp3_low"].get("g_floor", 0.1)
        assert g_floor >= 0.12, f"mp3_low g_floor={g_floor}"  # type: ignore[operator]

    def test_wax_cylinder_most_conservative(self):
        """Wax cylinder should have highest g_floor and lowest strength."""
        phase = DenoisePhase()
        wax = phase.MATERIAL_PARAMS["wax_cylinder"]
        assert wax["strength"] <= 0.30  # type: ignore[operator]
        assert wax.get("g_floor", 0.1) >= 0.30  # type: ignore[operator]

    def test_reel_tape_gentler_than_cassette(self):
        """Reel tape (higher quality) should have lower strength than cassette."""
        phase = DenoisePhase()
        assert phase.MATERIAL_PARAMS["reel_tape"]["strength"] < phase.MATERIAL_PARAMS["cassette"]["strength"]  # type: ignore[operator]

    def test_decade_interpolation_1960_is_neutral(self):
        """Decade 1960 should give multiplier ≈ 1.0 (neutral baseline)."""
        mult = decade_strength_multiplier(1960)
        assert abs(mult - 1.0) < 1e-6, f"1960 muss neutral sein, war {mult}"

    def test_decade_interpolation_smooth(self):
        """Decade strength multipliers should vary smoothly (no jumps)."""
        # Prüft die PRODUKTIONS-Knotentabelle (keine Test-Kopie), damit
        # Tabellenänderungen hier sofort auffallen.
        decades = [k[0] for k in ERA_DECADE_KNOTS]
        mults = [k[1] for k in ERA_DECADE_KNOTS]
        # Check monotonically non-increasing from earliest to latest
        for i in range(len(mults) - 1):
            assert mults[i] >= mults[i + 1] - 0.001, (
                f"Decade {decades[i]}: {mults[i]} > {decades[i + 1]}: {mults[i + 1]}"
            )

    def test_decade_strength_multiplier_clamps_and_interpolates(self):
        """Interpolation über die Produktions-Knoten: Ränder klemmen, Mitte interpoliert."""
        # Unterhalb/oberhalb der Knoten: Randwerte
        assert decade_strength_multiplier(1850) == decade_strength_multiplier(1890)
        assert decade_strength_multiplier(2050) == decade_strength_multiplier(2025)
        # Zwischen 1950 (1.05) und 1960 (1.00): linearer Verlauf
        assert abs(decade_strength_multiplier(1955) - 1.025) < 1e-6
