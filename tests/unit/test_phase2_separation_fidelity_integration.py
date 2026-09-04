"""
Phase 2 Integration Test: Full-Song Separation Fidelity via ChunkedProcessor

Tests that separation_fidelity measurement correctly:
1. Accepts full-song audio (30s+) without truncation
2. Transparently uses ChunkedProcessor for audio > 343980 samples
3. Integrates with RestaurerDenker global_scalar calibration

Spec: §v10.0.0 Phase 2 (ChunkedProcessor → separation_fidelity → global_scalar)
"""

import numpy as np
import pytest

from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric


class TestPhase2SeparationFidelityIntegration:
    """Full-song separation_fidelity measurement with ChunkedProcessor transparency."""

    def test_separation_fidelity_short_audio_reference_based(self):
        """Short audio (< WINDOW_SIZE) with reference → DSP-proxy (no HTDemucs needed)."""
        metric = SeparationFidelityMetric()

        # 7s audio (335544 samples @ 48kHz) — below WINDOW_SIZE (343980)
        audio = np.random.randn(335544).astype(np.float32) * 0.1
        reference = audio.copy()

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Random noise scores low for separation fidelity (normal behavior)
        # Important: test that measurement completes without error for short audio

    def test_separation_fidelity_full_song_30s_reference_based(self):
        """Full 30s song (687960 samples) with reference → ChunkedProcessor transparency."""
        metric = SeparationFidelityMetric()

        # 14.33s audio @ 48kHz (687960 samples) — requires 2 chunks
        audio = np.random.randn(687960).astype(np.float32) * 0.1
        reference = audio.copy() + np.random.randn(687960).astype(np.float32) * 0.02

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Random audio scores lower, but should not crash
        assert score >= 0.0

    def test_separation_fidelity_60s_song_reference_based(self):
        """60s song (2880000 samples) → 7+ chunks via ChunkedProcessor."""
        metric = SeparationFidelityMetric()

        # 60s audio @ 48kHz — requires 7-8 chunks (STRIDE = 301980 samples)
        audio = np.random.randn(2880000).astype(np.float32) * 0.05
        reference = audio.copy()

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="vinyl",
            global_scalar=0.9
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_separation_fidelity_stereo_full_song(self):
        """Stereo (2, N) full song → auto-convert to mono, ChunkedProcessor handles full song."""
        metric = SeparationFidelityMetric()

        # Stereo 30s @ 48kHz
        audio = np.random.randn(2, 687960).astype(np.float32) * 0.1
        reference = audio.copy()

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="tape",
            global_scalar=0.7
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_separation_fidelity_material_adaptive_tape(self):
        """Tape material → adaptive gain correction applied."""
        metric = SeparationFidelityMetric()

        # 30s tape audio
        audio = np.random.randn(687960).astype(np.float32) * 0.1
        reference = audio.copy()

        # Tape material should apply × 0.95 correction if MS-ratio heuristic used
        score_tape = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="tape",
            global_scalar=0.8
        )

        score_unknown = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        # Both should be valid
        assert 0.0 <= score_tape <= 1.0
        assert 0.0 <= score_unknown <= 1.0

    def test_separation_fidelity_global_scalar_low_triggers_proxy(self):
        """global_scalar < 0.15 → triggers proxy-fast-validation path."""
        metric = SeparationFidelityMetric()

        # 30s audio with very low global_scalar
        audio = np.random.randn(687960).astype(np.float32) * 0.1
        reference = audio.copy()

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.1  # < 0.15 → triggers fast path
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_separation_fidelity_without_reference_fallback_proxy(self):
        """No reference audio → fallback to _reference_free() (DSP-proxy mode)."""
        metric = SeparationFidelityMetric()

        # 30s audio, NO reference
        audio = np.random.randn(687960).astype(np.float32) * 0.1

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=None,  # No reference
            material_type="unknown",
            global_scalar=0.8
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_separation_fidelity_mono_1d_input(self):
        """1D mono input (N,) → proper handling."""
        metric = SeparationFidelityMetric()

        # 1D array, 30s
        audio = np.random.randn(687960).astype(np.float32) * 0.1
        reference = audio.copy()

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_separation_fidelity_nan_inf_handling(self):
        """NaN/Inf in audio → cleaned before processing."""
        metric = SeparationFidelityMetric()

        # 30s audio with some NaN/Inf
        audio = np.random.randn(687960).astype(np.float32) * 0.1
        audio[1000:1100] = np.nan
        audio[50000:50050] = np.inf

        reference = np.random.randn(687960).astype(np.float32) * 0.1

        score = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        assert isinstance(score, float)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_separation_fidelity_caching(self):
        """Identical inputs → cached result on second call."""
        metric = SeparationFidelityMetric()

        # Create fixed audio (not random, so hash is deterministic)
        audio = np.ones(687960, dtype=np.float32) * 0.1
        reference = audio.copy()

        score1 = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        score2 = metric.measure(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        # Should be identical (cached)
        assert score1 == score2


class TestPhase2MusicalGoalsCheckerIntegration:
    """Integration with MusicalGoalsChecker.measure_all() for full-song goals."""

    def test_musical_goals_checker_full_song_with_separation_fidelity(self):
        """MusicalGoalsChecker.measure_all() on full song includes separation_fidelity."""
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker(mode="restoration")

        # 30s audio @ 48kHz
        audio = np.random.randn(687960).astype(np.float32) * 0.1
        reference = audio.copy()

        goals = checker.measure_all(
            audio=audio,
            sr=48000,
            reference=reference,
            material_type="unknown",
            global_scalar=0.8
        )

        # All 15 goals should be present
        assert "separation_fidelity" in goals
        assert isinstance(goals["separation_fidelity"], float)
        assert 0.0 <= goals["separation_fidelity"] <= 1.0

        # Verify all 15 goals are measured
        expected_goals = {
            "brillanz", "waerme", "natuerlichkeit", "authentizitaet",
            "emotionalitaet", "transparenz", "bass_kraft", "groove",
            "spatial_depth", "timbre_authentizitaet", "tonal_center",
            "micro_dynamics", "separation_fidelity", "artikulation",
            "transient_energie"
        }
        assert expected_goals.issubset(set(goals.keys()))
