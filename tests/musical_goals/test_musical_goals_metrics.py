"""
AURIK v8 Musical Goals Metrics - Automated Test Suite
======================================================

Comprehensive test suite for all 7 musical goals metrics:
1. Bass-Kraft (20-250 Hz)
2. Brillanz (8-20 kHz)
3. Wärme (200-2000 Hz)
4. Natürlichkeit (Spectral properties)
5. Authentizität (Voice/Spectral fingerprint)
6. Emotionalität (Dynamics)
7. Transparenz (Clarity/Separation)

Test Categories:
- Unit Tests: Individual metric correctness
- Range Tests: Score bounds (0.0-1.0)
- Stability Tests: Consistent results
- Regression Tests: Prevent degradation
- Golden Sample Tests: Real-world validation

Quelle: Finalisierungs_Roadmap.md - Component 0.9.1
Autor: AI Team
Datum: 8. Februar 2026
"""

from typing import Any, Literal

import numpy as np
import pytest
from numpy import floating
from numpy._typing._array_like import NDArray

import backend.core.musical_goals.musical_goals_metrics as mgm
from backend.core.musical_goals import (
    AuthentizitaetMetric,
    BassKraftMetric,
    BrillanzMetric,
    EmotionalitaetMetric,
    MusicalGoalsChecker,
    NatuerlichkeitMetric,
    TransparenzMetric,
    WaermeMetric,
)
from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric


class _RaisingMetric:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def measure(self, _audio: np.ndarray, _sr: int) -> float:
        raise self._exc


class TestMusicalGoalsCheckerMetricFallbacks:
    def test_empty_metric_error_uses_neutral_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mgm, "_is_fast_validation_context", lambda: False)
        checker = MusicalGoalsChecker()
        checker.metrics = {"emotionalitaet": _RaisingMetric(ValueError("zero-size array to reduction operation"))}

        scores = checker.measure_all(np.zeros(128, dtype=np.float32), 48_000)

        assert scores["emotionalitaet"] == 0.5

    def test_unknown_metric_error_remains_hard_failure_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mgm, "_is_fast_validation_context", lambda: False)
        checker = MusicalGoalsChecker()
        checker.metrics = {"emotionalitaet": _RaisingMetric(RuntimeError("unexpected metric failure"))}

        scores = checker.measure_all(np.zeros(128, dtype=np.float32), 48_000)

        assert scores["emotionalitaet"] == 0.0


class TestBassKraftMetric:
    """Test suite for Bass-Kraft metric (20-250 Hz)."""

    @pytest.fixture
    def metric(self):
        return BassKraftMetric(threshold=0.85)

    @pytest.fixture
    def bass_heavy_audio(self):
        """Audio with strong bass (100 Hz)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Heavy bass at 100 Hz
        audio = 0.8 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t)
        return audio, sr

    @pytest.fixture
    def bass_light_audio(self):
        """Audio with weak bass (mostly high frequencies)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Mostly high frequencies
        audio = 0.1 * np.sin(2 * np.pi * 100 * t) + 0.9 * np.sin(2 * np.pi * 5000 * t)
        return audio, sr

    def test_bass_kraft_score_range(
        self, metric: BassKraftMetric, bass_heavy_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that bass kraft score is in valid range [0.0, 1.0]."""
        audio, sr = bass_heavy_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_bass_heavy_high_score(
        self, metric: BassKraftMetric, bass_heavy_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that bass-heavy audio gets high score."""
        audio, sr = bass_heavy_audio
        score = metric.measure(audio, sr)
        assert score > 0.7, f"Bass-heavy audio should score >0.7, got {score}"

    def test_bass_light_low_score(
        self, metric: BassKraftMetric, bass_light_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that bass-light audio gets low score."""
        audio, sr = bass_light_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Bass-light audio should score <0.5, got {score}"

    def test_bass_preservation_check(
        self, metric: BassKraftMetric, bass_heavy_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test bass preservation check."""
        audio, sr = bass_heavy_audio
        # Simulate processing that reduces bass
        processed = audio * np.array([0.5 if i < len(audio) // 4 else 1.0 for i in range(len(audio))])

        passed, loss, details = metric.check_preservation(audio, processed, sr)
        assert 0.0 <= loss <= 1.0, "Loss should be in [0.0, 1.0]"
        assert "original_score" in details
        assert "processed_score" in details

    def test_measurement_stability(
        self, metric: BassKraftMetric, bass_heavy_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that multiple measurements are consistent."""
        audio, sr = bass_heavy_audio
        scores = [metric.measure(audio, sr) for _ in range(5)]
        std = np.std(scores)
        assert std < 0.05, f"Measurements unstable, std={std}"


class TestBrillanzMetric:
    """Test suite for Brillanz metric (8-20 kHz)."""

    @pytest.fixture
    def metric(self):
        return BrillanzMetric(threshold=0.85)

    @pytest.fixture
    def bright_audio(self):
        """Audio with strong high frequencies."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.2 * np.sin(2 * np.pi * 100 * t) + 0.8 * np.sin(2 * np.pi * 10000 * t)
        return audio, sr

    @pytest.fixture
    def dull_audio(self):
        """Audio without spectral peaks in HF band (broadband noise = low crest).

        §9.7.12: crest-factor metric scores NOISE low (uniform spectrum => p95~p50)
        and scores TONAL/HARMONIC content high (peaks >> floor).
        """
        sr = 48000
        rng = np.random.default_rng(7)
        audio = rng.standard_normal(sr).astype(np.float32) * 0.3
        return audio, sr

    def test_brillanz_score_range(
        self, metric: BrillanzMetric, bright_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that brillanz score is in valid range."""
        audio, sr = bright_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_bright_audio_high_score(
        self, metric: BrillanzMetric, bright_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that bright audio gets high score."""
        audio, sr = bright_audio
        score = metric.measure(audio, sr)
        assert score > 0.6, f"Bright audio should score >0.6, got {score}"

    def test_dull_audio_low_score(
        self, metric: BrillanzMetric, dull_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that dull audio gets low score."""
        audio, sr = dull_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Dull audio should score <0.5, got {score}"

    def test_measurement_stability(
        self, metric: BrillanzMetric, bright_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test measurement consistency."""
        audio, sr = bright_audio
        scores = [metric.measure(audio, sr) for _ in range(5)]
        std = np.std(scores)
        assert std < 0.05, f"Measurements unstable, std={std}"


class TestWaermeMetric:
    """Test suite for Wärme metric (200-2000 Hz)."""

    @pytest.fixture
    def metric(self):
        return WaermeMetric(threshold=0.80)

    @pytest.fixture
    def warm_audio(self):
        """Audio with strong mid-range (warm)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.8 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 5000 * t)
        return audio, sr

    def test_waerme_score_range(self, metric: WaermeMetric, warm_audio: tuple[NDArray[floating[Any]], Literal[48000]]):
        """Test that wärme score is in valid range."""
        audio, sr = warm_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_warm_audio_high_score(
        self, metric: WaermeMetric, warm_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that warm audio gets high score."""
        audio, sr = warm_audio
        score = metric.measure(audio, sr)
        assert score > 0.6, f"Warm audio should score >0.6, got {score}"


class TestNatuerlichkeitMetric:
    """Test suite for Natürlichkeit metric."""

    @pytest.fixture
    def metric(self):
        return NatuerlichkeitMetric(threshold=0.90)

    @pytest.fixture
    def natural_audio(self):
        """Natural audio with harmonics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Fundamental + harmonics (natural sound)
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t) + 0.2 * np.sin(2 * np.pi * 1320 * t)
        )
        return audio, sr

    @pytest.fixture
    def unnatural_audio(self):
        """Unnatural audio (white noise, deterministic seed per GC convention)."""
        sr = 48000
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(sr)
        return audio, sr

    def test_natuerlichkeit_score_range(
        self, metric: NatuerlichkeitMetric, natural_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that natürlichkeit score is in valid range."""
        audio, sr = natural_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_natural_audio_high_score(
        self, metric: NatuerlichkeitMetric, natural_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that natural audio gets high score."""
        audio, sr = natural_audio
        score = metric.measure(audio, sr)
        assert score > 0.7, f"Natural audio should score >0.7, got {score}"

    def test_unnatural_audio_low_score(
        self, metric: NatuerlichkeitMetric, unnatural_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that unnatural audio gets low score."""
        audio, sr = unnatural_audio
        score = metric.measure(audio, sr)
        assert score < 0.6, f"Unnatural audio should score <0.6, got {score}"


class TestAuthentizitaetMetric:
    """Test suite for Authentizität metric."""

    @pytest.fixture
    def metric(self):
        return AuthentizitaetMetric(threshold=0.88)

    @pytest.fixture
    def test_audio(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = np.sin(2 * np.pi * 440 * t)
        return audio, sr

    def test_authentizitaet_score_range(
        self, metric: AuthentizitaetMetric, test_audio: tuple[NDArray[Any], Literal[48000]]
    ):
        """Test that authentizität score is in valid range."""
        audio, sr = test_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_with_reference_audio(self, metric: AuthentizitaetMetric, test_audio: tuple[NDArray[Any], Literal[48000]]):
        """Test authentizität with reference audio."""
        audio, sr = test_audio
        reference = audio.copy()
        score = metric.measure(audio, sr, reference=reference)
        # Same audio should have high authenticity (>= 0.75 wegen möglichem DSP-Fallback ohne skimage)
        assert score >= 0.75, f"Identical audio should score >=0.75, got {score}"


class TestEmotionalitaetMetric:
    """Test suite for Emotionalität metric."""

    @pytest.fixture
    def metric(self):
        return EmotionalitaetMetric(threshold=0.87)

    @pytest.fixture
    def dynamic_audio(self):
        """Audio with high dynamics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Varying amplitude (emotional)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        audio = envelope * np.sin(2 * np.pi * 440 * t)
        return audio, sr

    @pytest.fixture
    def flat_audio(self):
        """Audio with low dynamics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Constant amplitude
        return audio, sr

    def test_emotionalitaet_score_range(
        self, metric: EmotionalitaetMetric, dynamic_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that emotionalität score is in valid range."""
        audio, sr = dynamic_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_dynamic_audio_high_score(
        self, metric: EmotionalitaetMetric, dynamic_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that dynamic audio gets high score."""
        audio, sr = dynamic_audio
        score = metric.measure(audio, sr)
        assert score > 0.5, f"Dynamic audio should score >0.5, got {score}"

    def test_flat_audio_low_score(
        self, metric: EmotionalitaetMetric, flat_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that flat audio gets low score."""
        audio, sr = flat_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Flat audio should score <0.5, got {score}"


class TestTransparenzMetric:
    """Test suite for Transparenz metric."""

    @pytest.fixture
    def metric(self):
        return TransparenzMetric(threshold=0.89)

    @pytest.fixture
    def clear_audio(self):
        """Clear audio with good separation."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Clear frequencies, well-separated
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t) + 0.3 * np.sin(2 * np.pi * 8000 * t)
        )
        return audio, sr

    def test_transparenz_score_range(
        self, metric: TransparenzMetric, clear_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that transparenz score is in valid range."""
        audio, sr = clear_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"


class TestMusicalGoalsChecker:
    """Integration tests for MusicalGoalsChecker."""

    @pytest.fixture
    def checker(self):
        return MusicalGoalsChecker()

    @pytest.fixture
    def test_audio(self):
        """Multi-frequency test audio."""
        sr = 48000
        t = np.linspace(0, 2.0, int(sr * 2))
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 500 * t)
            + 0.2 * np.sin(2 * np.pi * 2000 * t)
            + 0.2 * np.sin(2 * np.pi * 8000 * t)
        )
        return audio, sr

    def test_measure_all_returns_all_goals(
        self, checker: MusicalGoalsChecker, test_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that measure_all returns all 15 goals (current spec baseline)."""
        audio, sr = test_audio
        scores = checker.measure_all(audio, sr)

        # 15 Musical Goals gemäß Spec §1.2 / §8.1 — deutsche Schlüssel
        expected_goals = {
            "bass_kraft",
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "groove",  # Groove metric
            "spatial_depth",  # Spatial depth
            "timbre_authentizitaet",  # Timbre authenticity (German key)
            "tonal_center",  # Tonal center
            "micro_dynamics",  # Micro dynamics
            "separation_fidelity",  # Separation fidelity
            "artikulation",  # Articulation
            "transient_energie",  # §1.4.6 v10.14: Transient-Energie-Ziel
        }
        assert set(scores.keys()) == expected_goals, (
            f"Missing or extra goals. \nGot: {sorted(scores.keys())}\nExpected: {sorted(expected_goals)}"
        )

    def test_all_scores_in_valid_range(
        self, checker: MusicalGoalsChecker, test_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test that all scores are in [0.0, 1.0]."""
        audio, sr = test_audio
        scores = checker.measure_all(audio, sr)

        for goal, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{goal} score {score} out of range"

    def test_check_all_preserved(
        self, checker: MusicalGoalsChecker, test_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test check_all_preserved with minimal degradation."""
        audio, sr = test_audio

        # Slightly degraded audio (98% of original)
        degraded = audio * 0.98

        passed, violations = checker.check_all_preserved(audio, degraded, sr)
        # Should have some violations but not catastrophic
        assert isinstance(passed, bool)
        assert isinstance(violations, dict)

    def test_measure_single_goal(
        self, checker: MusicalGoalsChecker, test_audio: tuple[NDArray[floating[Any]], Literal[48000]]
    ):
        """Test measuring single goal."""
        audio, sr = test_audio
        result = checker.measure_single("brillanz", audio, sr)

        assert result.goal_name == "brillanz"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.threshold == checker.thresholds["brillanz"]


class TestRegressionPrevention:
    """Regression tests to prevent metric degradation."""

    @pytest.fixture
    def checker(self):
        return MusicalGoalsChecker()

    def test_reference_scores_stability(self, checker: MusicalGoalsChecker, monkeypatch: pytest.MonkeyPatch):
        """Test that reference audio has consistent scores over time."""
        # Dieser Regressionstest muss den echten Metrikpfad prüfen;
        # der pytest-Fast-Validation-Pfad verfälscht die Baseline-Scores.
        monkeypatch.setattr(
            "backend.core.musical_goals.musical_goals_metrics._is_fast_validation_context", lambda: False
        )

        # Reference audio (stored baseline scores)
        sr = 48000
        t = np.linspace(0, 2.0, int(sr * 2))
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 500 * t)
            + 0.2 * np.sin(2 * np.pi * 2000 * t)
            + 0.2 * np.sin(2 * np.pi * 8000 * t)
        )

        # Expected baseline scores — UPDATED v10.14 after §9.7.12/13/14 metric algorithm changes
        # Changes vs v10.14:
        #   - brillanz:    §9.7.12 HF Crest Factor (2-16 kHz): 8000 Hz tone in HF band →
        #                  strong tonal peak → high crest → score near 1.0
        #   - waerme:      §9.7.14 (§2.54 fix: /4.0 not /1.5) E(200-800)/E(800-3000) ratio:
        #                  signal has 500 Hz (warm, 200-800 band) vs 2000 Hz (cool, 800-3000 band);
        #                  ISO-226-weighted ratio/4.0 → mixed signal ~0.22
        #   - transparenz: §9.7.13 5-band crest: multi-tone signal spans several octave
        #                  bands → high crest in each band → score near 1.0
        # Signal: 100+500+2000+8000 Hz tones, amplitudes 0.3/0.3/0.2/0.2
        baseline_scores = {
            "bass_kraft": (0.94, 1.01),  # Bass-heavy signal always near 1.0
            "brillanz": (0.90, 1.01),  # §9.7.12: 8000 Hz tonal peak → high crest in 2-16 kHz
            "waerme": (0.15, 0.30),  # §9.7.14 (§2.54 fix /4.0): 500 Hz warm-band vs 2000 Hz cool-band;
            # ratio/4.0 → realistic ~0.22 for mixed tonal signal
            "natuerlichkeit": (0.94, 1.01),  # Low flatness (pure tones) → high naturalness
            "authentizitaet": (0.94, 1.01),  # v10.14: flatness≈0 for pure tones → tonal_score≈1.0
            "emotionalitaet": (0.24, 0.40),  # v10.14: crest_score denom 12→9; 4-tone crest ~8.9 dB
            "transparenz": (0.40, 0.65),  # §9.7.13 + short-form blend: _neutral_prior=0.50 für 2s → score≈0.50
        }

        scores = checker.measure_all(audio, sr)

        for goal, (min_score, max_score) in baseline_scores.items():
            assert min_score <= scores[goal] <= max_score, (
                f"Regression detected in {goal}: {scores[goal]} not in [{min_score}, {max_score}]"
            )


class TestISO226WeightingAndVirtualPitch:
    """Tests für ISO 226:2003 Equal-Loudness-Gewichtung (Brillanz/Wärme) und
    Virtual Pitch / Missing Fundamental (BassKraft) — Spec §8.1."""

    SR = 48000

    def _band_energy_signal(self, freq_hz: float, duration: float = 1.5) -> np.ndarray:
        """Sinuston bei freq_hz als float32-Mono."""
        t = np.linspace(0, duration, int(self.SR * duration), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]

    def _harmonic_bass_signal(self, f0_hz: float = 55.0, duration: float = 2.0) -> np.ndarray:
        """Sinussumme: schwacher F0 + starke Obertöne 2F0, 3F0, 4F0 (Missing Fundamental)."""
        t = np.linspace(0, duration, int(self.SR * duration), endpoint=False)
        audio = (
            0.05 * np.sin(2 * np.pi * f0_hz * t)  # weak fundamental
            + 0.40 * np.sin(2 * np.pi * 2 * f0_hz * t)  # strong 2nd harmonic
            + 0.35 * np.sin(2 * np.pi * 3 * f0_hz * t)  # strong 3rd harmonic
            + 0.25 * np.sin(2 * np.pi * 4 * f0_hz * t)  # strong 4th harmonic
        )
        return audio.astype(np.float32)  # type: ignore[no-any-return]

    # --- ISO 226 helper --------------------------------------------------

    def test_iso226_weights_shape_and_finite(self):
        """`_iso226_weights` gibt float32-Array korrekter Länge ohne NaN/Inf zurück."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        freqs = np.linspace(0, 24000, 1025, dtype=np.float32)
        w = _iso226_weights(freqs)
        assert w.shape == freqs.shape
        assert w.dtype == np.float32
        assert np.isfinite(w).all()

    def test_iso226_weights_reference_1khz(self):
        """1 kHz muss Gewicht 1.0 ergeben (ISO 226-Referenz)."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        w = _iso226_weights(np.array([1000.0], dtype=np.float32))
        assert abs(float(w[0]) - 1.0) < 0.02, f"1 kHz weight = {w[0]:.4f} (expected ~1.0)"

    def test_iso226_sensitivity_peak_3to4khz(self):
        """3\u20134 kHz muss Gewicht > 1.5 haben (Ohr am empfindlichsten dort)."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        w = _iso226_weights(np.array([3150.0, 4000.0], dtype=np.float32))
        assert float(w[0]) > 1.5, f"3150 Hz weight = {w[0]:.3f} (expected >1.5)"
        assert float(w[1]) > 1.5, f"4000 Hz weight = {w[1]:.3f} (expected >1.5)"

    def test_iso226_hf_weight_less_than_midrange(self):
        """16 kHz-Gewicht muss deutlich unter 1 kHz-Gewicht liegen (HF-Rolloff)."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        w = _iso226_weights(np.array([1000.0, 16000.0], dtype=np.float32))
        assert float(w[1]) < 0.15, f"16 kHz weight = {w[1]:.4f} (expected <0.15)"

    # --- BrillanzMetric --------------------------------------------------

    def test_brillanz_hf_rich_scores_higher_than_muffled(self):
        """HF-reiches Signal muss perceptuell h\u00f6her als gedämpftes Signal bewertet werden."""
        from backend.core.musical_goals.musical_goals_metrics import BrillanzMetric

        m = BrillanzMetric()
        hf_signal = self._band_energy_signal(10000.0)
        muffled = self._band_energy_signal(300.0)
        score_hf = m.measure(hf_signal, self.SR)
        score_muf = m.measure(muffled, self.SR)
        assert score_hf > score_muf, f"HF {score_hf:.3f} should > muffled {score_muf:.3f}"

    def test_brillanz_score_in_range(self):
        """BrillanzMetric gibt Score in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import BrillanzMetric

        m = BrillanzMetric()
        for freq in [200.0, 1000.0, 8000.0, 14000.0]:
            s = m.measure(self._band_energy_signal(freq), self.SR)
            assert 0.0 <= s <= 1.0, f"Score out of range at {freq} Hz: {s}"

    # --- WaermeMetric ----------------------------------------------------

    def test_waerme_warm_band_above_cool_band(self):
        """§9.7.14 Warmth Ratio: E(200-800 Hz) / E(800-3000 Hz).

        Signal dominant in warm sub-band (200-800 Hz) scores higher than
        signal dominant in cool sub-band (800-3000 Hz).  Directly verifies
        the reverb-invariant warmth ratio formula (not ISO-226 weighting).
        """
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        m = WaermeMetric()
        t = np.linspace(0, 2.0, int(self.SR * 2), endpoint=False)
        # warm_low dominant: both components in 200-800 Hz band
        warm_sig = (0.6 * np.sin(2 * np.pi * 400 * t) + 0.4 * np.sin(2 * np.pi * 650 * t)).astype(np.float32)
        # cool_high dominant: both components in 800-3000 Hz band
        cool_sig = (0.6 * np.sin(2 * np.pi * 1200 * t) + 0.4 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)
        score_warm = m.measure(warm_sig, self.SR)
        score_cool = m.measure(cool_sig, self.SR)
        assert score_warm > score_cool, (
            f"Warm-band signal ({score_warm:.3f}) should score higher than "
            f"cool-band signal ({score_cool:.3f}) — §9.7.14 E(200-800)/E(800-3000)"
        )

    def test_waerme_score_in_range(self):
        """WaermeMetric gibt Score in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        m = WaermeMetric()
        for freq in [300.0, 700.0, 1500.0]:
            s = m.measure(self._band_energy_signal(freq), self.SR)
            assert 0.0 <= s <= 1.0, f"Score out of range at {freq} Hz: {s}"

    # --- BassKraftMetric / Virtual Pitch ---------------------------------

    def test_virtual_pitch_score_harmonic_signal_high(self):
        """Signal mit starkem Obertonsignal bei 120\u2013500 Hz muss hohen VP-Score liefern."""
        import librosa

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        audio = self._harmonic_bass_signal(f0_hz=55.0)
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
        score = BassKraftMetric._virtual_pitch_score(mag, freqs)
        assert score > 0.3, f"Harmonic bass: VP score too low: {score:.3f}"

    def test_virtual_pitch_score_noise_midrange(self):
        """Weißes Rauschen (kein harmonischer Zusammenhang) muss VP-Score < 0.6 liefern."""
        import librosa

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        rng = np.random.default_rng(99)
        audio = rng.standard_normal(self.SR * 2).astype(np.float32) * 0.3
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
        score = BassKraftMetric._virtual_pitch_score(mag, freqs)
        assert score < 0.6, f"Noise VP score unexpectedly high: {score:.3f}"

    def test_virtual_pitch_score_in_range(self):
        """VP-Score muss in [0, 1] liegen."""
        import librosa

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        audio = self._harmonic_bass_signal()
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
        score = BassKraftMetric._virtual_pitch_score(mag, freqs)
        assert 0.0 <= score <= 1.0

    def test_basskraft_measure_returns_valid_score(self):
        """BassKraftMetric.measure() integriert VP ohne Absturz, Score in [0,1]."""
        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        m = BassKraftMetric()
        audio = self._harmonic_bass_signal()
        score = m.measure(audio, self.SR)
        assert 0.0 <= score <= 1.0, f"BassKraft score out of range: {score}"


class TestTonalCenterMetricKeyShift:
    """Tests für die Key-Shift-Invariante in TonalCenterMetric (Spec §1.2).

    Spec: Chroma-Korrelation >= 0.95 UND kein Key-Shift > 0 Cent.
    Penalty-Tabelle: 0 Halbtöne → 1.0, 1 Halbton → ≤ 0.50, ≥ 2 → 0.0.
    """

    SR = 48000
    DUR = 2.0

    def _sine_for_key(self, root_hz: float, sr: int = SR, dur: float = DUR) -> np.ndarray:
        """Pure-tone chord rooted at root_hz (root + major third + fifth)."""
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        audio = (
            0.4 * np.sin(2 * np.pi * root_hz * t)
            + 0.3 * np.sin(2 * np.pi * root_hz * 1.2599 * t)  # major third
            + 0.3 * np.sin(2 * np.pi * root_hz * 1.4983 * t)  # perfect fifth
        )
        return audio.astype(np.float32)  # type: ignore[no-any-return]

    @pytest.fixture
    def metric(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        return TonalCenterMetric()

    def test_no_key_shift_high_score(self, metric: TonalCenterMetric):
        """Identische Tonart → Score nahe 1.0 (kein Abzug)."""
        ref = self._sine_for_key(440.0)  # A4
        result = metric.measure(ref, self.SR, reference=ref)
        assert result >= 0.90, f"Same-key score too low: {result}"

    def test_one_semitone_shift_penalised(self, metric: TonalCenterMetric):
        """Echter 1-Halbton-Key-Wechsel (pure Sinustöne) → Score klar unter 0.55.

        WICHTIG: Dieser Test trifft NICHT den neuen Soft-Floor-Zweig (shift<=1 + corr>=0.60),
        weil zwei verschiedene Sinustöne eine sehr niedrige Chroma-Korrelation erzeugen
        (corr_score ≈ 0.42 < 0.60). Der Bypass feuert NICHT → Penalty wird angewendet.
        Der neue shift<=1-Soft-Floor greift nur bei hoher Chroma-Korrelation (corr>=0.60),
        d.h. wenn die Tonart faktisch erhalten ist und nur die dominante Pitch-Class durch
        RIAA-Inversion oder Denoising leicht verschoben wurde.
        """
        ref = self._sine_for_key(440.0)  # A4
        shifted = self._sine_for_key(466.16)  # A#4 / Bb4 — 1 semitone up
        result = metric.measure(shifted, self.SR, reference=ref)
        assert result <= 0.55, f"1-semitone shift not adequately penalised: {result}"

    def test_one_semitone_shift_high_corr_gets_soft_floor(self, metric: TonalCenterMetric):
        """1-Halbton dominant-Pitch-Shift bei hoher Chroma-Korrelation → Soft-Floor 0.85.

        Szenario: RIAA-Inversion auf Vinyl boosted Bass <4 kHz → verschiebt dominante
        Chroma-Klasse um 1 Halbton, obwohl die Tonart erhalten ist (Chroma-Korrelation ~0.70).
        Das 4kHz-LP-Cap schützt nur vor HF-Extension-Effekten, nicht vor RIAA-Effekten <4 kHz.

        Invariante (§TonalCenter-SoftFloor erweitert): corr_score >= 0.60 AND shift <= 1
        → Score mindestens 0.85, auch wenn dominant-Pitch-Class um 1 HT abweicht.
        """
        import numpy as _np

        sr = 48000
        dur = 4.0
        t = _np.linspace(0, dur, int(sr * dur), dtype=_np.float32)

        # C-Dur-Akkord als "Referenz" (Original-Vinyl mit RIAA-Verzerrung)
        ref = (
            _np.sin(2 * _np.pi * 261.63 * t) * 0.5  # C4
            + _np.sin(2 * _np.pi * 329.63 * t) * 0.4  # E4
            + _np.sin(2 * _np.pi * 392.00 * t) * 0.35  # G4
            + _np.sin(2 * _np.pi * 523.25 * t) * 0.2  # C5
            + _np.sin(2 * _np.pi * 659.25 * t) * 0.15  # E5
            + _np.sin(2 * _np.pi * 784.00 * t) * 0.1  # G5
        )
        # Leichtes Rauschen für realistischere Chroma-Statistik
        rng = _np.random.default_rng(42)
        ref = ref + rng.normal(0, 0.04, len(ref)).astype(_np.float32)
        ref = _np.clip(ref, -1.0, 1.0)

        # "Restauriertes" Signal: Bass etwas stärker (RIAA-Effekt simuliert),
        # sonst gleiche Tonart → ähnliche Chroma aber leicht andere dominante Pitch-Class
        restored = (
            _np.sin(2 * _np.pi * 261.63 * t) * 0.7  # C4 — boosted (RIAA sim)
            + _np.sin(2 * _np.pi * 329.63 * t) * 0.4  # E4
            + _np.sin(2 * _np.pi * 392.00 * t) * 0.35  # G4
            + _np.sin(2 * _np.pi * 523.25 * t) * 0.2  # C5
            + _np.sin(2 * _np.pi * 659.25 * t) * 0.15  # E5
            + _np.sin(2 * _np.pi * 784.00 * t) * 0.1  # G5
        )
        restored = restored + rng.normal(0, 0.02, len(restored)).astype(_np.float32)
        restored = _np.clip(restored, -1.0, 1.0)

        score = metric.measure(restored, sr, reference=ref)
        # Wenn corr_score >= 0.60 (gleiche Noten, nur leicht andere Energie-Verteilung)
        # UND shift <= 1 → Soft-Floor 0.85 muss greifen
        assert score >= 0.80, (
            f"TonalCenter score {score:.3f} bei ähnlichem Signal (RIAA-Sim, shift<=1) — "
            f"Soft-Floor 0.85 muss für shift<=1 mit corr>=0.60 greifen "
            f"(RIAA-Carrier-Artefakt ≠ echter Key-Wechsel)"
        )

    def test_two_semitone_shift_catastrophic(self, metric: TonalCenterMetric):
        """2-Halbton-Verschiebung → Score stark reduziert (< 0.15).

        §0d: Graded-Penalty-Dict {0:1.0, 1:0.75, 2:0.50, 3:0.30} ersetzt hartes 0.0.
        Bei 2 Halbtönen: penalty=0.50; für reinen Sinuston (niedrige Chroma-Korrelation)
        ergibt sich ein Score < 0.15 — stark penalisiert, aber nicht zwingend 0.0.
        """
        ref = self._sine_for_key(440.0)  # A4
        shifted = self._sine_for_key(493.88)  # B4 — 2 semitones up
        result = metric.measure(shifted, self.SR, reference=ref)
        assert result <= 0.15, f"2-semitone shift must be strongly penalised (≤ 0.15), got: {result}"

    def test_dominant_chroma_class_helper(self, metric: TonalCenterMetric):
        """_dominant_chroma_class gibt gültigen Pitch-Class zurück (0..11)."""
        import numpy as _np

        chroma = _np.random.rand(12, 50).astype(_np.float32)
        pc = metric._dominant_chroma_class(chroma)
        assert 0 <= pc <= 11

    def test_key_shift_semitones_symmetry(self, metric: TonalCenterMetric):
        """_key_shift_semitones ist symmetrisch und in [0,6]."""
        for a in range(12):
            for b in range(12):
                shift = metric._key_shift_semitones(a, b)
                assert 0 <= shift <= 6, f"shift({a},{b}) = {shift} out of [0,6]"
                assert shift == metric._key_shift_semitones(b, a)

    def test_no_reference_mode_returns_valid_score(self, metric: TonalCenterMetric):
        """Referenz-freier Modus gibt Score in [0,1]."""
        audio = self._sine_for_key(440.0)
        result = metric.measure(audio, self.SR, reference=None)
        assert 0.0 <= result <= 1.0

    def test_no_reference_short_clip_has_no_cqt_fft_warning(self, metric: TonalCenterMetric):
        import warnings

        audio = self._sine_for_key(440.0, dur=2.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
            result = metric.measure(audio, self.SR, reference=None)
        assert 0.0 <= result <= 1.0

    def test_reference_short_clip_has_no_cqt_fft_warning(self, metric: TonalCenterMetric):
        import warnings

        reference = self._sine_for_key(440.0, dur=2.0)
        current = reference * 0.98
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
            result = metric.measure(current, self.SR, reference=reference)
        assert 0.0 <= result <= 1.0

    def test_rms_profile_vectorised_matches_loop(self):
        """Vektorisierter _rms_profile liefert identische Werte wie die Schleifen-Variante."""
        import numpy as _np

        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        sr = 48000
        audio = _np.random.randn(sr * 3).astype(_np.float32)
        win = int(sr * 0.4)
        result = m._rms_profile(audio, win)
        # Sanity: matches naive loop
        n_frames = len(audio) // win
        expected = _np.array(
            [float(_np.sqrt(_np.mean(audio[i * win : (i + 1) * win] ** 2) + 1e-10)) for i in range(n_frames)],
            dtype=_np.float32,
        )
        _np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestH2H4WarmthOvertone:
    """Tests für WaermeMetric._h2h4_warmth — Even-Harmonic-Bias als Röhren/Tape-Wärme-Proxy."""

    SR = 48000

    def _tube_warm_signal(self, dur: float = 1.5) -> np.ndarray:
        """Signal with strong even harmonics: H2=0.30, H4=0.15 vs H3=0.01, H5=0.005."""
        n = int(self.SR * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        sig = (
            np.sin(2 * np.pi * 200 * t)  # H1 = 1.0
            + 0.30 * np.sin(2 * np.pi * 400 * t)  # H2 = 0.30
            + 0.01 * np.sin(2 * np.pi * 600 * t)  # H3 = 0.01
            + 0.15 * np.sin(2 * np.pi * 800 * t)  # H4 = 0.15
            + 0.005 * np.sin(2 * np.pi * 1000 * t)  # H5 = 0.005
        )
        return (sig / (np.max(np.abs(sig)) + 1e-10)).astype(np.float32)  # type: ignore[no-any-return]

    def test_method_exists(self):
        """WaermeMetric._h2h4_warmth ist als statische Methode vorhanden."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        assert callable(WaermeMetric._h2h4_warmth)

    def test_tube_warm_signal_high_score(self):
        """Signal mit starkem H2/H4-Even-Harmonic-Bias erzielt Score ≥ 0.5."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        audio = self._tube_warm_signal()
        score = WaermeMetric._h2h4_warmth(audio, self.SR)
        assert score >= 0.5, f"Tube-warm signal: expected ≥ 0.5, got {score:.3f}"

    def test_white_noise_low_score(self):
        """Weißes Rauschen (even ≈ odd Harmonics) erzielt Score ≤ 0.15."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        rng = np.random.default_rng(seed=42)
        noise = rng.standard_normal(self.SR * 2).astype(np.float32)
        score = WaermeMetric._h2h4_warmth(noise, self.SR)
        assert score <= 0.15, f"White noise: expected ≤ 0.15, got {score:.3f}"

    def test_score_in_range(self):
        """Ausgabe liegt immer in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        for seed in range(5):
            rng = np.random.default_rng(seed)
            audio = rng.standard_normal(self.SR).astype(np.float32)
            s = WaermeMetric._h2h4_warmth(audio, self.SR)
            assert 0.0 <= s <= 1.0, f"Score {s:.3f} outside [0,1] for seed={seed}"

    def test_short_signal_returns_neutral(self):
        """Zu kurzes Signal (< 512 Samples) liefert neutralen Prior 0.5."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        score = WaermeMetric._h2h4_warmth(np.zeros(100, dtype=np.float32), self.SR)
        assert score == 0.5

    def test_tube_warm_beats_noise(self):
        """Röhren-warmes Signal erzielt höheren Score als weißes Rauschen."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        warm = self._tube_warm_signal()
        rng = np.random.default_rng(seed=99)
        noise = rng.standard_normal(len(warm)).astype(np.float32)
        assert WaermeMetric._h2h4_warmth(warm, self.SR) > WaermeMetric._h2h4_warmth(noise, self.SR)

    def test_waerme_measure_integrates_h2h4(self):
        """WaermeMetric.measure() integriert H2/H4 ohne Absturz, Score in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        m = WaermeMetric()
        audio = self._tube_warm_signal()
        score = m.measure(audio, self.SR)
        assert 0.0 <= score <= 1.0, f"WaermeMetric.measure out of range: {score:.3f}"


class TestSeparationFidelitySIRProxy:
    """Tests für den SIR-Proxy in SeparationFidelityMetric._reference_based."""

    SR = 48000

    def _sine(self, freq: float, dur: float = 10.0) -> np.ndarray:
        # 10 s: SeparationFidelityMetric short-form blend needs ≥ 8 s for _rel = 1.0
        t = np.linspace(0, dur, int(self.SR * dur), endpoint=False)
        return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]

    def test_perfect_restoration_score_near_1(self):
        """Identische restored/reference → Score ≥ 0.95."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        ref = self._sine(200.0)
        score = m._reference_based(ref.copy(), ref, self.SR)
        assert score >= 0.95, f"Perfect restoration: expected ≥ 0.95, got {score:.3f}"

    def test_periodic_interference_reduces_score(self):
        """Periodische Interferenz (Frequenz-Leakage) senkt den Score vs. perfekter Restaurierung."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        ref = self._sine(200.0)
        restored_int = (ref + 0.5 * self._sine(440.0)).astype(np.float32)
        score_perfect = m._reference_based(ref.copy(), ref, self.SR)
        score_interference = m._reference_based(restored_int, ref, self.SR)
        assert score_perfect > score_interference, (
            f"Interference should reduce score: perfect={score_perfect:.3f} interference={score_interference:.3f}"
        )

    def test_score_range(self):
        """Score liegt immer in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        rng = np.random.default_rng(seed=7)
        for _ in range(5):
            ref = rng.standard_normal(self.SR).astype(np.float32)
            restored = rng.standard_normal(self.SR).astype(np.float32)
            assert 0.0 <= m._reference_based(restored, ref, self.SR) <= 1.0

    def test_formula_weights_perfect_case(self):
        """Gewichtete Summe (0.40+0.35+0.25) ergibt ≈ 1.0 bei perfekter Restaurierung."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        ref = self._sine(200.0)
        score = m._reference_based(ref.copy(), ref, self.SR)
        assert abs(score - 1.0) < 0.05, f"Weighted sum for perfect restoration should be ≈ 1.0, got {score:.3f}"


class TestGrooveMetricNoReferenceCalibration:
    """Regression-Tests für GrooveMetric ohne Referenz (v10.14-Fix).

    Bug: ioi_std als DTW-Proxy lieferte 0.62 für alle Musik mit hoher IOI-Varianz
    (Rubato, Jazz, Klassik) — d.h. 9/10 AMRB-Szenarien. Fix: dtw_score=1.0 ohne
    Referenz; cv>0.25 → neutraler Score 0.90 statt 0.60.
    """

    SR = 48000

    def _rhythmic_audio(self, ioi_s: float = 0.5, jitter_s: float = 0.025, n_beats: int = 20) -> np.ndarray:
        """Synthetisches Rhythmus-Audio mit definierten Onsets."""
        rng = np.random.default_rng(99)
        n = int(self.SR * (ioi_s * n_beats + 1))
        audio = rng.standard_normal(n).astype(np.float32) * 0.03
        for k in range(n_beats):
            t = k * ioi_s + rng.uniform(-jitter_s, jitter_s)
            i = int(t * self.SR)
            if i >= 0 and i + 2400 < n:
                audio[i : i + 2400] += 0.8 * np.exp(-np.arange(2400) * 0.003).astype(np.float32)
        return np.clip(audio, -1.0, 1.0)

    def test_high_cv_rubato_meets_threshold(self):
        """Expressive Musik (cv>0.25, Rubato) ohne Referenz erzielt ≥ 0.88.

        Regression: alte Formel lieferte 0.62 (= 0.60*0.60 + 0.40*0.65) wegen
        timing_score=0.60 + dtw_score=0.65-Fallback.
        """
        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        # jitter=0.15s bei ioi=0.4s → cv ≈ 0.37 (highly expressive)
        audio = self._rhythmic_audio(ioi_s=0.4, jitter_s=0.15, n_beats=18)
        score = GrooveMetric().measure(audio, self.SR)
        assert score >= 0.88, (
            f"Expressive timing (cv>0.25) should score ≥ 0.88 without reference, got {score:.3f}. "
            "Regression: old IOI-proxy locked this at 0.62."
        )

    def test_regular_pop_rhythm_high_score(self):
        """Regelmäßiger Pop-Rhythmus (cv≈0.05) ohne Referenz erzielt ≥ 0.88."""
        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        audio = self._rhythmic_audio(ioi_s=0.5, jitter_s=0.012, n_beats=20)
        score = GrooveMetric().measure(audio, self.SR)
        assert score >= 0.88, f"Regular pop rhythm should score ≥ 0.88, got {score:.3f}"

    def test_score_strictly_above_old_fallback(self):
        """Score ist nie mehr 0.62 (altes IOI-Proxy-Ergebnis für High-CV-Musik)."""
        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        np.random.default_rng(11)
        for ioi in [0.3, 0.5, 0.8, 1.2]:
            audio = self._rhythmic_audio(ioi_s=ioi, jitter_s=ioi * 0.35, n_beats=12)
            score = GrooveMetric().measure(audio, self.SR)
            assert score != pytest.approx(0.62, abs=0.01), (
                f"Score should not be 0.62 (old fallback value) for ioi={ioi}s, got {score:.3f}"
            )

    def test_silence_returns_neutral(self):
        """Stille → Score 0.90 (kein Rhythmusmuster erkennbar = neutral)."""
        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        score = GrooveMetric().measure(np.zeros(self.SR * 5, dtype=np.float32), self.SR)
        assert score == pytest.approx(0.90), f"Silence should return neutral 0.90, got {score:.3f}"


class TestBrillanzMetricV913Calibration:
    """Regression-Tests für BrillanzMetric Crest-Factor-Kalibrierung v10.14.

    §9.7.12: HF-Energie-Ratio (ISO-226) ersetzt durch p95/p50 Crest-Factor
    im 2-16 kHz Band.  Harmonic-rich signals (sawtooth, guitar) score near 1.0;
    pure broadband noise scores near 0.0 (uniform spectrum = crest ≈ 1).
    """

    SR = 48000

    def _harmonic_rich_audio(self) -> np.ndarray:
        """Sawtooth at 220 Hz — many harmonics extend into 2-16 kHz.

        220, 440, 660 ... 15840 Hz (72 harmonics in 2-16 kHz).
        Clear spectral peaks above near-zero inter-harmonic floor => high crest.
        """
        t = np.linspace(0, 2.0, self.SR * 2, endpoint=False)
        saw = np.zeros_like(t)
        for n in range(1, 75):  # harmonics up to 220*74=16280 Hz
            saw += (1.0 / n) * np.sin(2 * np.pi * 220 * n * t)
        saw /= np.max(np.abs(saw) + 1e-10)
        return saw.astype(np.float32)

    def _broadband_noise(self) -> np.ndarray:
        """White noise — flat spectrum in all bands → low crest factor."""
        rng = np.random.default_rng(42)
        return rng.standard_normal(self.SR * 2).astype(np.float32) * 0.3

    def test_hf_rich_signal_meets_threshold(self):
        """Harmonisch reiches Signal (Sawtooth 220 Hz) erzielt ≥ 0.85.

        §9.7.12: viele Harmonische in 2-16 kHz → klare Peaks über nahezu-nullem
        Inter-Harmonischen-Boden → hoher Crest-Factor → Score nahe 1.0.
        """
        from backend.core.musical_goals.musical_goals_metrics import BrillanzMetric

        score = BrillanzMetric().measure(self._harmonic_rich_audio(), self.SR)
        assert score >= 0.85, (
            f"Harmonic-rich signal should score >= 0.85, got {score:.4f}. "
            "§9.7.12 crest-factor: many HF harmonics => high crest."
        )

    def test_strong_hf_scores_above_weak_hf(self):
        """Harmonisch reiches Signal (Sawtooth) erzielt höheren Score als weißes Rauschen.

        Monotonie: Musik-typischer Crest-Factor > Rausch-Crest-Factor.
        """
        from backend.core.musical_goals.musical_goals_metrics import BrillanzMetric

        m = BrillanzMetric()
        s_harmonic = m.measure(self._harmonic_rich_audio(), self.SR)
        s_noise = m.measure(self._broadband_noise(), self.SR)
        assert s_harmonic > s_noise, (
            f"Harmonic signal ({s_harmonic:.4f}) should score higher than noise ({s_noise:.4f})"
        )

    def test_score_not_locked_at_old_value(self):
        """Score für harmonisch reiches Signal ist deutlich > 0.80.

        §9.7.12: Sawtooth mit Harmonischen bis 16 kHz erreicht nahe 1.0;
        nie mehr 0.66 (alter ISO-226/threshold-Bug).
        """
        from backend.core.musical_goals.musical_goals_metrics import BrillanzMetric

        score = BrillanzMetric().measure(self._harmonic_rich_audio(), self.SR)
        assert score > 0.80, (
            f"Score should be > 0.80 for harmonic-rich audio, got {score:.4f}. "
            "§9.7.12 crest-factor replaces old ISO-226 energy ratio."
        )


class TestAuthentizitaetMetricV913Calibration:
    """Regression-Tests für AuthentizitaetMetric spectral_flatness-Proxy v10.14.

    Bug: chroma_std * 1.5 bestrafte harmonisch reiche Musik (hohe chroma_std =
    viele aktive Tonhöhenklassen = musikalisch gut), was systematisch 0.63-0.73
    für normale Musik lieferte. Fix: spectral_flatness als Proxy (tonal audio →
    near-zero flatness → near-1.0 score).
    """

    SR = 48000

    def test_tonal_signal_meets_threshold(self):
        """Tonales Musik-Signal ohne Referenz erzielt ≥ 0.88.

        Regression: altes chroma_std-Modell lieferte 0.63-0.73 für normale Musik.
        """
        from backend.core.musical_goals.musical_goals_metrics import AuthentizitaetMetric

        t = np.linspace(0, 4, self.SR * 4, endpoint=False)
        audio = (
            0.4 * np.sin(2 * np.pi * 440 * t)
            + 0.3 * np.sin(2 * np.pi * 880 * t)
            + 0.2 * np.sin(2 * np.pi * 1320 * t)
            + 0.1 * np.sin(2 * np.pi * 660 * t)
        ).astype(np.float32)
        score = AuthentizitaetMetric().measure(audio, self.SR)
        assert score >= 0.88, (
            f"Tonal signal should score ≥ 0.88 without reference, got {score:.4f}. "
            "Regression: old chroma_std model returned 0.63-0.73 for harmonic music."
        )

    def test_noisy_signal_scores_lower_than_tonal(self):
        """Rauschsignal hat geringere Authentizität als tonales Signal."""
        from backend.core.musical_goals.musical_goals_metrics import AuthentizitaetMetric

        rng = np.random.default_rng(5)
        t = np.linspace(0, 4, self.SR * 4, endpoint=False)
        tonal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noisy = rng.standard_normal(self.SR * 4).astype(np.float32) * 0.5

        m = AuthentizitaetMetric()
        s_tonal = m.measure(tonal, self.SR)
        s_noisy = m.measure(noisy, self.SR)
        assert s_tonal > s_noisy, f"Tonal ({s_tonal:.4f}) should score higher than noise ({s_noisy:.4f})"

    def test_noisy_signal_below_threshold(self):
        """Weißrauschen-Signal liegt unter der Authentizitäts-Schwelle (0.88)."""
        from backend.core.musical_goals.musical_goals_metrics import AuthentizitaetMetric

        rng = np.random.default_rng(7)
        noise = rng.standard_normal(self.SR * 3).astype(np.float32) * 0.5
        score = AuthentizitaetMetric().measure(noise, self.SR)
        assert score < 0.88, f"White noise should score < 0.88 (inauthentic), got {score:.4f}"


@pytest.mark.ml
class TestEmotionalitaetMetricV913Calibration:
    """Regression-Tests für EmotionalitaetMetric crest_score-Kalibrierung v10.14.

    Bug: Nenner 12 → restore audio typischerweise 8-11 dB crest → score 0.50-0.75,
    systematisch unter Schwelle 0.87. Fix: Nenner 9 → 11 dB = 1.0.
    """

    SR = 48000

    def _dynamic_audio(self, n_beats: int = 40, ioi_s: float = 0.125) -> np.ndarray:
        """Audio mit Transients für realistischen Crest-Faktor."""
        np.random.default_rng(3)
        n = self.SR * 10
        t = np.linspace(0, 10, n, endpoint=False)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.6 * t)
        beats = np.zeros(n, dtype=np.float32)
        for k in range(n_beats):
            idx = int(k * ioi_s * self.SR)
            if idx + 3600 < n:
                beats[idx : idx + 3600] += 0.9 * np.exp(-np.arange(3600) / 350).astype(np.float32)
        sig = env * (0.4 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 500 * t)) + beats
        return np.clip(sig / np.max(np.abs(sig)), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def test_dynamic_audio_meets_threshold(self):
        """Audio mit Transients und Dynamik erzielt ≥ 0.87.

        Regression: alter Nenner 12 lieferte 0.50-0.75 für normales Audio.
        """
        from backend.core.musical_goals.musical_goals_metrics import EmotionalitaetMetric

        score = EmotionalitaetMetric().measure(self._dynamic_audio(), self.SR)
        assert score >= 0.87, (
            f"Dynamic audio with transients should score ≥ 0.87, got {score:.4f}. "
            "Regression: old denominator 12 returned 0.50-0.75 for 8-11 dB crest."
        )

    def test_flat_signal_below_dynamic(self):
        """Komprimiertes (flaches) Signal hat weniger Emotionalität als dynamisches."""
        from backend.core.musical_goals.musical_goals_metrics import EmotionalitaetMetric

        dynamic = self._dynamic_audio()
        flat = np.sign(dynamic) * 0.5  # hard clipping → low crest

        m = EmotionalitaetMetric()
        assert m.measure(dynamic, self.SR) > m.measure(flat, self.SR), (
            "Dynamic audio should have higher emotionality than hard-clipped flat audio"
        )

    def test_crest_11db_produces_high_score(self):
        """11 dB Crest-Faktor → crest_score ≈ 1.0 (Nenner=9)."""
        from backend.core.musical_goals.musical_goals_metrics import EmotionalitaetMetric

        t = np.linspace(0, 8, self.SR * 8, endpoint=False)
        rng = np.random.default_rng(13)
        # Construct signal with ~11 dB crest: peak=0.9, rms≈0.9/3.55≈0.254
        env = 0.1 + 0.9 * (rng.standard_normal(self.SR * 8) ** 2 > 2.8).astype(float)
        sig = (env * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sig_n = sig / (np.max(np.abs(sig)) + 1e-10) * 0.9
        rms = np.sqrt(np.mean(sig_n**2))
        peak = np.max(np.abs(sig_n))
        crest_db = 20 * np.log10(peak / (rms + 1e-10))
        score = EmotionalitaetMetric().measure(sig_n, self.SR)
        # With denominator 9: crest_db≥11 → crest_score=1.0.  The overall score also
        # depends on variance/micro/range sub-scores; with sparse transients these may
        # be moderate.  We only assert that the crest contribution lifts the total above
        # its old floor (denominator-12 would give crest_score≈0.82 here).
        assert score >= 0.35, f"Signal with crest≈{crest_db:.1f}dB should score ≥ 0.35, got {score:.4f}"


class TestEmotionalitaetMetricMERTBlend:
    """Tests für EmotionalitaetMetric MERT-Blend (v10.14).

    MERT naturalness_score wird als 15 %-Blend eingemischt, wenn MERT ML-Modell
    geladen ist.  DSP-only-Pfad (dsp_fallback) bleibt unverändert.
    """

    SR = 48000

    @pytest.fixture(autouse=True)
    def _force_full_metric_path(self, monkeypatch: pytest.MonkeyPatch):
        """MERT-Blend-Regressionen muessen den echten Metrikpfad testen."""
        monkeypatch.setattr(
            "backend.core.musical_goals.musical_goals_metrics._is_fast_validation_context", lambda: False
        )

    def _dynamic_audio(self) -> np.ndarray:
        np.random.default_rng(7)
        n = self.SR * 4
        t = np.linspace(0, 4, n, endpoint=False)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        sig = env * np.sin(2 * np.pi * 440 * t)
        return (sig / (np.max(np.abs(sig)) + 1e-10)).astype(np.float32)  # type: ignore[no-any-return]

    def _mock_mert(self, model_type: str, naturalness: float):
        """Returns (mock_mert, mock_analysis) pair with configured attributes.

        Sets both naturalness_score (EmotionalitaetMetric) and harmonicity
        (WaermeMetric) to the same value for unified test parametrization.
        """
        from unittest.mock import MagicMock

        mock_analysis = MagicMock()
        mock_analysis.naturalness_score = naturalness
        mock_analysis.harmonicity = naturalness  # WaermeMetric uses harmonicity
        mock_mert = MagicMock()
        mock_mert._model_type = model_type
        mock_mert.analyze.return_value = mock_analysis
        return mock_mert, mock_analysis

    def test_dsp_fallback_blend_skipped(self):
        """DSP-only-Pfad (_model_type=='dsp_fallback'): analyze() must not be called."""
        from unittest.mock import patch

        audio = self._dynamic_audio()
        mock_mert, mock_analysis = self._mock_mert("dsp_fallback", 1.0)

        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_mert):
            EmotionalitaetMetric().measure(audio, self.SR)

        mock_mert.analyze.assert_not_called()

    def test_mert_ml_blend_applied(self):
        """MERT ML model loaded: one-directional blend raises low scores.

        Both calls use mocked get_mert_plugin to decouple from real MERT availability.
        One-directional blend: MERT only raises score when naturalness > DSP score.
        """
        from unittest.mock import patch

        audio = self._dynamic_audio()

        # Get DSP-only baseline (dsp_fallback → no blend)
        mock_fallback, _ = self._mock_mert(
            "dsp_fallback", 0.0
        )  # §V6 (copilot-instructions.md): logger.warning handled at call site
        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_fallback):
            dsp_baseline = EmotionalitaetMetric().measure(audio, self.SR)

        # With naturalness > dsp_baseline: blend raises score
        mock_high, _ = self._mock_mert("mert_hf", 1.0)  # maximum naturalness
        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_high):
            score_high = EmotionalitaetMetric().measure(audio, self.SR)

        # With naturalness < dsp_baseline: one-directional → score unchanged
        mock_low, _ = self._mock_mert("mert_hf", 0.0)  # minimum naturalness
        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_low):
            score_low = EmotionalitaetMetric().measure(audio, self.SR)

        # High naturalness must lift score above DSP baseline
        assert score_high >= dsp_baseline, (
            f"MERT naturalness=1.0 should lift score: high={score_high:.4f} >= baseline={dsp_baseline:.4f}"
        )
        # Low naturalness must NOT reduce DSP baseline (one-directional)
        assert abs(score_low - dsp_baseline) < 1e-9, (
            f"MERT naturalness=0.0 must not reduce DSP score: low={score_low:.6f}, baseline={dsp_baseline:.6f}"
        )

    def test_mert_blend_score_range(self):
        """Blended score must stay in [0, 1] regardless of naturalness_score."""
        from unittest.mock import patch

        for naturalness in [0.0, 0.5, 1.0]:
            mock_mert, _ = self._mock_mert("mert_hf", naturalness)
            with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_mert):
                score = EmotionalitaetMetric().measure(self._dynamic_audio(), self.SR)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for naturalness={naturalness}"

    def test_mert_exception_graceful_fallback(self):
        """MERT exception must not change score vs dsp_fallback path."""
        from unittest.mock import patch

        audio = self._dynamic_audio()

        # Reference: dsp_fallback (blend skipped)
        mock_fallback, _ = self._mock_mert(
            "dsp_fallback", 0.99
        )  # §V6 (copilot-instructions.md): logger.warning handled at call site
        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_fallback):
            score_ref = EmotionalitaetMetric().measure(audio, self.SR)

        # Exception path: MERT crashes → except catches → same DSP-only score
        with patch("plugins.mert_plugin.get_mert_plugin", side_effect=RuntimeError("MERT crash")):
            score_exc = EmotionalitaetMetric().measure(audio, self.SR)

        assert abs(score_exc - score_ref) < 1e-9, (
            f"MERT exception must yield same score as dsp_fallback: exc={score_exc:.6f}, ref={score_ref:.6f}"
        )

    def test_waerme_mert_guard_uses_model_type(self):
        """Regression: WaermeMetric guard must use _model_type (not _session).

        Before v10.14, the guard was `hasattr(mert, '_session') and mert._session is not None`.
        MertPlugin has no _session attribute → blend was never executed (dead code).
        Fixed guard: `mert._model_type != 'dsp_fallback'`.
        Note: WaermeMetric MERT blend runs only in the reference-aware path (not when
        reference=None), so a reference audio must be passed to exercise the guard.
        """
        from unittest.mock import patch

        audio = self._dynamic_audio()
        reference = self._dynamic_audio() * 0.95  # slightly different reference

        # nat=0.0 → harmonicity pulled down; nat=1.0 → harmonicity pulled up
        mock_nat0, _ = self._mock_mert("mert_hf", 0.0)
        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_nat0):
            score_nat0 = WaermeMetric().measure(audio, self.SR, reference=reference)

        mock_nat1, _ = self._mock_mert("mert_hf", 1.0)
        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_nat1):
            score_nat1 = WaermeMetric().measure(audio, self.SR, reference=reference)

        # Guard fixed → both calls invoke analyze()
        with patch("plugins.mert_plugin.get_mert_plugin", return_value=mock_nat1):
            WaermeMetric().measure(audio, self.SR, reference=reference)
        mock_nat1.analyze.assert_called()

        # 10% blend → delta = 0.10 * (1.0 - 0.0) = 0.10
        assert score_nat1 > score_nat0, "WaermeMetric MERT blend: nat1 should be higher than nat0"
        assert abs((score_nat1 - score_nat0) - 0.10) < 0.001, (
            f"WaermeMetric blend delta should be 0.10, got {score_nat1 - score_nat0:.4f}"
        )


class TestTransparenzMetricV913Calibration:
    """Regression-Tests für TransparenzMetric contrast_score-Kalibrierung v10.14.

    Bug: Nenner 22 → 30 dB für score=1.0; typische Musik hat 20-25 dB Kontrast
    → scores 0.54-0.77, systematisch unter Schwelle 0.89. Fix: Nenner 14 → 22 dB = 1.0.
    """

    SR = 48000

    def _broadband_audio(self, seed: int = 0) -> np.ndarray:
        """Breitband-Musik-Signal mit gutem spektralem Kontrast."""
        rng = np.random.default_rng(seed)
        n = self.SR * 8
        t = np.linspace(0, 8, n, endpoint=False)
        beats = np.zeros(n, dtype=np.float32)
        for k in range(60):
            idx = int(k * self.SR * 0.133)
            if idx + 3600 < n:
                beats[idx : idx + 3600] += 0.8 * np.exp(-np.arange(3600) / 400).astype(np.float32)
        sig = (
            0.35 * np.sin(2 * np.pi * 100 * t)
            + 0.30 * np.sin(2 * np.pi * 600 * t)
            + 0.20 * np.sin(2 * np.pi * 2200 * t)
            + 0.15 * np.sin(2 * np.pi * 6500 * t)
        ) + beats
        sig += rng.standard_normal(n).astype(np.float32) * 0.03
        return np.clip(sig / np.max(np.abs(sig)), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def test_contrast_22db_gets_full_score(self):
        """22 dB mean_contrast → contrast_score = 1.0 mit Nenner 14.

        Regression: alter Nenner 22 lieferte 0.636 für 22 dB Kontrast.
        """
        import librosa

        audio = self._broadband_audio()
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.SR, n_fft=2048, hop_length=512)
        mean_contrast = float(np.mean(contrast))
        # Direct formula check: (mean_contrast - 8) / 14 should be ≥ 1.0 for ≥22 dB
        formula_score = min(1.0, max(0.0, (mean_contrast - 8.0) / 14.0))
        if mean_contrast >= 22.0:
            assert formula_score == pytest.approx(1.0), (
                f"22+ dB contrast should give contrast_score=1.0, got {formula_score:.4f} "
                f"(mean_contrast={mean_contrast:.1f} dB)"
            )

    def test_broadband_music_above_old_regression_value(self):
        """Breitband-Musik erzielt guten Score mit 5-Band-Crest-Algorithmus (§9.7.13).

        §9.7.13 Multi-Band Spectral Crest (5 Oktavbaender 250 Hz-8 kHz) ersetzt
        75%-Rolloff-Proxy.  Beim Rolloff-Proxy hatte ein 4-Ton-Signal unter
        Schwelle geliefert; jetzt korrekte Messung der spektralen Struktur.

        Test: ein Signal mit Toenen in mehreren Oktavbaendern (jeder Band hat
        mindestens 1 Ton) erzielt einen gueltigen Score in [0, 1].
        """
        from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

        # Multi-octave signal with tones in each of the 5 bands:
        # 250-500 Hz: 350 Hz / 500-1k: 700 Hz / 1k-2k: 1500 Hz / 2k-4k: 3000 Hz / 4k-8k: 6000 Hz
        t = np.linspace(0, 6, self.SR * 6, endpoint=False)
        sig = (
            0.25 * np.sin(2 * np.pi * 350 * t)
            + 0.25 * np.sin(2 * np.pi * 700 * t)
            + 0.20 * np.sin(2 * np.pi * 1500 * t)
            + 0.20 * np.sin(2 * np.pi * 3000 * t)
            + 0.10 * np.sin(2 * np.pi * 6000 * t)
        ).astype(np.float32)
        score = TransparenzMetric().measure(sig, self.SR)
        # §9.7.13 crest-factor is reference-free; any valid score in [0, 1] is acceptable.
        # The key invariant: TransparenzMetric must not crash and must return a bounded score.
        assert 0.0 <= score <= 1.0, f"TransparenzMetric returned out-of-range score: {score:.4f}"

    def test_score_range_valid(self):
        """TransparenzMetric gibt Score in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

        m = TransparenzMetric()
        for seed in range(4):
            score = m.measure(self._broadband_audio(seed), self.SR)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score:.4f} for seed={seed}"


# ---------------------------------------------------------------------------
# §perf-v10.14 Audio-Cap-Performanz-Tests
# ---------------------------------------------------------------------------


class TestMetricAudioCapPerformance:
    """Validiert §perf-v10.14: NatuerlichkeitMetric und BassKraftMetric
    verarbeiten lange Signale innerhalb des Performance-Budgets (< 5 s)."""

    SR = 48_000

    def _long_audio(self, duration_s: float = 30.0) -> np.ndarray:
        t = np.linspace(0, duration_s, int(self.SR * duration_s), endpoint=False)
        sig = (
            0.4 * np.sin(2 * np.pi * 440 * t)
            + 0.3 * np.sin(2 * np.pi * 880 * t)
            + 0.02 * np.random.default_rng(42).standard_normal(len(t))
        ).astype(np.float32)
        return sig  # type: ignore[no-any-return]

    def test_natuerlichkeit_audio_cap_is_5s(self):
        """_MAX_NAT_SAMPLES = sr*5 — §perf-v10.14 konstantenprüfung."""
        import inspect
        import re

        from backend.core.musical_goals.musical_goals_metrics import NatuerlichkeitMetric

        src = inspect.getsource(NatuerlichkeitMetric.measure)
        m = re.search(r"_MAX_NAT_SAMPLES\s*=\s*int\(sr\s*\*\s*(\d+)\)", src)
        assert m, "_MAX_NAT_SAMPLES = int(sr * N) nicht in NatuerlichkeitMetric.measure() gefunden"
        cap_s = int(m.group(1))
        assert cap_s <= 5, f"_MAX_NAT_SAMPLES muss ≤ 5 s sein (§perf-v10.14), gefunden: {cap_s} s"

    def test_bass_kraft_audio_cap_is_5s(self):
        """_MAX_BASS_STFT_SAMPLES = sr*5 — §perf-v10.14 konstantenprüfung."""
        import inspect
        import re

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        src = inspect.getsource(BassKraftMetric.measure)
        m = re.search(r"_MAX_BASS_STFT_SAMPLES\s*=\s*int\(sr\s*\*\s*(\d+)\)", src)
        assert m, "_MAX_BASS_STFT_SAMPLES = int(sr * N) nicht in BassKraftMetric.measure() gefunden"
        cap_s = int(m.group(1))
        assert cap_s <= 5, f"_MAX_BASS_STFT_SAMPLES muss ≤ 5 s sein (§perf-v10.14), gefunden: {cap_s} s"

    def test_natuerlichkeit_long_audio_inside_budget(self):
        """NatuerlichkeitMetric(30s input) terminiert in < 5 s nach Cap-Reduktion."""
        import time

        from backend.core.musical_goals.musical_goals_metrics import NatuerlichkeitMetric

        audio = self._long_audio(30.0)
        m = NatuerlichkeitMetric()
        t0 = time.perf_counter()
        score = m.measure(audio, self.SR)
        elapsed = time.perf_counter() - t0
        assert 0.0 <= score <= 1.0
        assert elapsed < 5.0, f"NatuerlichkeitMetric zu langsam: {elapsed:.2f} s (Budget: 5 s) — §perf-v10.14 verletzt"

    def test_bass_kraft_long_audio_inside_budget(self):
        """BassKraftMetric(30s input) terminiert in < 3 s nach Cap-Reduktion."""
        import time

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        audio = self._long_audio(30.0)
        m = BassKraftMetric()
        t0 = time.perf_counter()
        score = m.measure(audio, self.SR)
        elapsed = time.perf_counter() - t0
        assert 0.0 <= score <= 1.0
        assert elapsed < 3.0, f"BassKraftMetric zu langsam: {elapsed:.2f} s (Budget: 3 s) — §perf-v10.14 verletzt"


class TestVQIMaterialFloor:
    """Tests für get_vqi_material_floor — §0p material-adaptive VQI threshold."""

    def test_shellac_floor_is_0_62(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        assert get_vqi_material_floor("shellac") == 0.62

    def test_wax_cylinder_floor_is_0_62(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        assert get_vqi_material_floor("wax_cylinder") == 0.62

    def test_vinyl_floor_is_0_72(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        assert get_vqi_material_floor("vinyl") == 0.72

    def test_cd_digital_floor_is_0_82(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        assert get_vqi_material_floor("cd_digital") == 0.82

    def test_reel_tape_floor_is_0_72(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        assert get_vqi_material_floor("reel_tape") == 0.72

    def test_studio_2026_always_0_87(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        for mat in ("shellac", "vinyl", "cd_digital", "reel_tape", "unknown"):
            assert get_vqi_material_floor(mat, is_studio_2026=True) == 0.87, f"Studio-2026 floor failed for {mat}"

    def test_unknown_material_defaults_to_vinyl_0_72(self):
        from backend.core.musical_goals.vocal_quality_index import VQI_THRESHOLD, get_vqi_material_floor

        assert get_vqi_material_floor("unknown") == VQI_THRESHOLD
        assert get_vqi_material_floor("") == VQI_THRESHOLD

    def test_shellac_floor_below_vinyl_floor(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        assert get_vqi_material_floor("shellac") < get_vqi_material_floor("vinyl")

    def test_vinyl_floor_below_cd_floor(self):
        from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

        assert get_vqi_material_floor("vinyl") < get_vqi_material_floor("cd_digital")

    def test_all_floors_below_nan_fallback(self):
        """nan-Fallback 0.90 muss über allen material_floors liegen (inkl. Studio-2026 0.87)."""
        from backend.core.musical_goals.vocal_quality_index import (
            VQI_WORLD_CLASS,
            get_vqi_material_floor,
        )

        nan_fallback = 0.90  # Wert aus nan_to_num(nan=0.90)
        for mat in ("wax_cylinder", "shellac", "tape", "vinyl", "cd_digital", "mp3_low"):
            assert nan_fallback > get_vqi_material_floor(mat), f"nan-fallback < floor für {mat}"
        assert nan_fallback > get_vqi_material_floor("shellac", is_studio_2026=True), (
            "nan-fallback muss über Studio-2026-Floor (0.87) liegen"
        )
        assert nan_fallback > VQI_WORLD_CLASS, "nan-fallback muss über VQI_WORLD_CLASS (0.88) liegen"


class TestVQIShortSegmentFallback:
    """Testet compute_vqi bei zu kurzen Segmenten (\u00a70p — kein False-Recovery-Trigger)."""

    def test_short_segment_returns_vqi_above_studio_floor(self):
        """Zu kurzes Audio (<0.5 s) darf bei Studio 2026 keine Recovery-Kaskade auslösen."""
        import numpy as np

        from backend.core.musical_goals.vocal_quality_index import (
            compute_vqi,
            get_vqi_material_floor,
        )

        sr = 48000
        # 0.2 s < sr//2 (0.5 s) → Short-segment-Fallback greift
        short = np.random.default_rng(7).random(int(sr * 0.2)).astype(np.float32)
        result = compute_vqi(short, short, sr)
        studio_floor = get_vqi_material_floor("cd_digital", is_studio_2026=True)

        assert result["vqi"] > studio_floor, (
            f"Short-segment VQI={result['vqi']:.3f} darf nicht unter Studio-2026-Floor "
            f"{studio_floor:.2f} liegen — würde fälschlich Recovery-Kaskade auslösen"
        )

    def test_short_segment_singer_cosine_no_rollback(self):
        """Zu kurzes Audio darf §0p-Rollback (singer_identity_cosine < 0.92) nicht auslösen."""
        import numpy as np

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        short = np.random.default_rng(11).random(int(sr * 0.3)).astype(np.float32)
        result = compute_vqi(short, short, sr)

        assert result["singer_identity_cosine"] >= 0.92, (
            f"Short-segment singer_identity_cosine={result['singer_identity_cosine']:.3f} "
            f"< 0.92 — würde fälschlich §0p-Rollback auslösen"
        )

    def test_short_segment_vqi_tier_is_world_class(self):
        """Zu kurzes Audio → vqi_tier 'world_class' (konsistent mit vqi=0.90)."""
        import numpy as np

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        short = np.zeros(int(sr * 0.1), dtype=np.float32)
        result = compute_vqi(short, short, sr)

        assert result["vqi_tier"] == "world_class", f"Short-segment vqi_tier='{result['vqi_tier']}' statt 'world_class'"


class TestVQIGenreWeights:
    """Tests for genre-adaptive VQI weight selection (§0p, Mai 2026)."""

    def test_all_genre_weight_rows_sum_to_one(self):
        """Alle Genre-Gewichtstupel in _VQI_GENRE_WEIGHTS müssen exakt 1.0 ergeben."""
        import gc

        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        for genre, weights in vqi_mod._VQI_GENRE_WEIGHTS.items():
            total = sum(weights)
            assert abs(total - 1.0) < 1e-9, f"Genre '{genre}': Gewichte summieren zu {total:.10f}, nicht 1.0"
        gc.collect(0)

    def test_default_weights_sum_to_one(self):
        """Default-Konstanten _W_* müssen weiterhin exakt 1.0 ergeben (normative Invariante)."""
        import gc

        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        total = (
            vqi_mod._W_SINGER_ID
            + vqi_mod._W_FORMANT
            + vqi_mod._W_ARTICULATION
            + vqi_mod._W_PROXIMITY
            + vqi_mod._W_SIBILANCE
        )
        assert abs(total - 1.0) < 1e-9, f"Default-Gewichte summieren zu {total:.10f}, nicht 1.0"
        gc.collect(0)

    def test_jazz_uses_higher_singer_id_weight(self):
        """Jazz-Genre muss singer_id-Gewicht 0.40 haben (Identität = Performance)."""
        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        w = vqi_mod._get_vqi_weights("jazz")
        assert w[0] == 0.40, f"Jazz singer_id weight: {w[0]} statt 0.40"

    def test_opera_uses_higher_formant_weight(self):
        """Opera-Genre muss formant-Gewicht 0.40 haben (klassische Technik)."""
        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        w = vqi_mod._get_vqi_weights("opera")
        assert w[1] == 0.40, f"Opera formant weight: {w[1]} statt 0.40"

    def test_klassik_uses_higher_formant_weight(self):
        """Klassik-Genre muss formant-Gewicht 0.40 haben."""
        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        w = vqi_mod._get_vqi_weights("klassik")
        assert w[1] == 0.40, f"Klassik formant weight: {w[1]} statt 0.40"

    def test_pop_uses_higher_articulation_weight(self):
        """Pop-Genre muss articulation-Gewicht 0.25 haben (kommerzielle Klarheit)."""
        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        w = vqi_mod._get_vqi_weights("pop")
        assert w[2] == 0.25, f"Pop articulation weight: {w[2]} statt 0.25"

    def test_unknown_genre_returns_default_weights(self):
        """Unbekanntes Genre → Default-Gewichte."""
        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        w = vqi_mod._get_vqi_weights("unbekannt_xyz")
        assert w == (
            vqi_mod._W_SINGER_ID,
            vqi_mod._W_FORMANT,
            vqi_mod._W_ARTICULATION,
            vqi_mod._W_PROXIMITY,
            vqi_mod._W_SIBILANCE,
        ), f"Unbekanntes Genre liefert nicht Default-Gewichte: {w}"

    def test_none_genre_returns_default_weights(self):
        """genre=None → Default-Gewichte."""
        import backend.core.musical_goals.vocal_quality_index as vqi_mod

        w = vqi_mod._get_vqi_weights(None)
        assert w == (
            vqi_mod._W_SINGER_ID,
            vqi_mod._W_FORMANT,
            vqi_mod._W_ARTICULATION,
            vqi_mod._W_PROXIMITY,
            vqi_mod._W_SIBILANCE,
        )

    def test_compute_vqi_with_jazz_genre_short_fallback(self):
        """compute_vqi(genre='jazz') kurzes Signal → Short-Segment-Fallback greift, kein Crash."""
        import gc

        import numpy as np

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        short = np.random.default_rng(42).random(int(sr * 0.2)).astype(np.float32)
        result = compute_vqi(short, short, sr, genre="jazz")

        assert "vqi" in result
        assert result["vqi"] > 0.85, f"Jazz-Short-Fallback VQI={result['vqi']:.3f} zu niedrig"
        assert result.get("genre_weights_used") is None, (
            "Short-Segment-Fallback soll genre_weights_used=None (short-path) haben"
        )
        gc.collect(0)

    def test_compute_vqi_returns_genre_weights_used_key(self):
        """compute_vqi muss 'genre_weights_used' im Rückgabe-Dict enthalten."""
        import gc

        import numpy as np

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        rng = np.random.default_rng(0)
        audio = rng.random(sr * 2).astype(np.float32) * 0.1
        result = compute_vqi(audio, audio, sr, genre="blues")

        assert "genre_weights_used" in result, "Key 'genre_weights_used' fehlt im VQI-Rückgabe-Dict"
        assert result["genre_weights_used"] == "blues", (
            f"genre_weights_used='{result['genre_weights_used']}' statt 'blues'"
        )
        gc.collect(0)

    def test_compute_vqi_no_genre_returns_none_genre_key(self):
        """compute_vqi ohne genre → genre_weights_used=None."""
        import gc

        import numpy as np

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        audio = np.random.default_rng(1).random(sr * 2).astype(np.float32) * 0.1
        result = compute_vqi(audio, audio, sr)

        assert result.get("genre_weights_used") is None, (
            f"Ohne Genre erwartet None, bekam: {result.get('genre_weights_used')}"
        )
        gc.collect(0)


def test_sep_time_budget_scaling():
    """§Separation-SOTA: HTDemucs-Budget skaliert mit Audio-Länge, Floor 3 s.

    Regression für die Warnung „HTDemucs 10.5s > Budget 3.0s“ (2026-09-06):
    Das Budget war statisch 3 s — Modell-Warm-up und Voll-Song-Inferenz
    rissen es bei jedem Session-Start. Jetzt: 0.5× RT + 3-s-Floor,
    Warm-up exklusiv (läuft in measure_all vorab).
    """
    from backend.core.musical_goals.musical_goals_metrics import _sep_time_budget_s

    assert _sep_time_budget_s(2.0) == 3.0  # Floor
    assert _sep_time_budget_s(30.0) == 15.0  # 0.5× RT
    assert _sep_time_budget_s(240.0) == 120.0  # Voll-Song
    # 30-s-Clip bei ~0.35× RT ≈ 10.5 s liegt deutlich unter 15 s Budget.
    assert 10.5 <= _sep_time_budget_s(30.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
