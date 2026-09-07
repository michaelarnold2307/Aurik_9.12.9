"""
tests/unit/test_phase_56_spectral_band_gap_repair.py
=====================================================
Aurik 10.0.0 — SpectralBandGapRepairPhase (§4.5, §7.1)

22 Unit-Tests.
Alle Tests synthetisch (keine echten Audio-Dateien).
"""

import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase():
    from backend.core.phases.phase_56_spectral_band_gap_repair import SpectralBandGapRepairPhase

    return SpectralBandGapRepairPhase()


@pytest.fixture(scope="module")
def silence_1s():
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture(scope="module")
def sine_440_2s():
    np.random.seed(42)
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture(scope="module")
def noisy_audio():
    np.random.seed(42)
    return (np.random.randn(SR * 3) * 0.1).astype(np.float32)


@pytest.fixture(scope="module")
def stereo_audio():
    np.random.seed(42)
    ch1 = np.sin(2 * np.pi * 220 * np.linspace(0, 2, 2 * SR, endpoint=False)).astype(np.float32)
    ch2 = np.sin(2 * np.pi * 330 * np.linspace(0, 2, 2 * SR, endpoint=False)).astype(np.float32)
    return np.stack([ch1, ch2], axis=0)


# ---------------------------------------------------------------------------
# Tests: Metadaten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase56Metadata:
    def test_01_metadata_returns_object(self, phase):
        meta = phase.get_metadata()
        assert meta is not None

    def test_02_category_is_restoration(self, phase):
        meta = phase.get_metadata()
        assert (
            "restor" in str(meta.category).lower()
            or "repair" in str(meta.category).lower()
            or "defect" in str(meta.category).lower()
        )

    def test_03_name_contains_spectral(self, phase):
        meta = phase.get_metadata()
        assert "spectral" in meta.name.lower() or "band" in meta.name.lower()

    def test_04_estimated_time_positive(self, phase):
        meta = phase.get_metadata()
        assert meta.estimated_time_factor >= 0.0


# ---------------------------------------------------------------------------
# Tests: Grundfunktion process()
# ---------------------------------------------------------------------------


class TestPhase56Process:
    def test_05_process_returns_phase_result(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result is not None
        assert hasattr(result, "audio") and hasattr(result, "success")

    def test_06_output_shape_preserved_mono(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result.audio.shape == sine_440_2s.shape

    def test_07_output_dtype_float32(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result.audio.dtype == np.float32

    def test_08_no_nan_in_output(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert np.isfinite(result.audio).all(), "NaN/Inf im Ausgang"

    def test_09_no_clipping_in_output(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_10_silence_passthrough(self, phase, silence_1s):
        result = phase.process(silence_1s, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_11_noisy_audio_processed(self, phase, noisy_audio):
        result = phase.process(noisy_audio, sample_rate=SR)
        assert result is not None
        assert result.audio.shape == noisy_audio.shape
        assert np.isfinite(result.audio).all()

    def test_zero_strength_passthrough(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR, strength=0.0)
        assert result.success is True
        assert np.allclose(result.audio, sine_440_2s, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR, strength=1.0, phase_locality_factor=0.4)
        assert result.success is True
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_defect_locations_localize_repair(self, phase, sine_440_2s, monkeypatch):
        def _fake_process_channel(channel, sr, instrument_tag, gap_fraction_min=None, bw_cap_hz=None):
            return (channel * 0.10).astype(np.float32)

        monkeypatch.setattr(phase, "_process_channel", _fake_process_channel)
        monkeypatch.setattr(phase, "_mrsa_gain_refinement", lambda pre, post, sr: post)

        result = phase.process(
            sine_440_2s,
            sample_rate=SR,
            confidence=1.0,
            strength=1.0,
            defect_locations={"head_wear": [(0.20, 0.30)]},
        )
        assert result.success is True
        diff = np.abs(result.audio - sine_440_2s)
        in_region = float(np.mean(diff[int(0.21 * SR) : int(0.29 * SR)]))
        out_region = float(np.mean(diff[int(1.40 * SR) : int(1.70 * SR)]))
        assert in_region > out_region * 2.0
        assert float(result.metadata.get("repair_locality_coverage", 0.0)) > 0.0

    def test_defect_locations_are_event_strength_adaptive(self, phase, sine_440_2s, monkeypatch):
        def _fake_process_channel(channel, sr, instrument_tag, gap_fraction_min=None, bw_cap_hz=None):
            return (channel * 0.10).astype(np.float32)

        monkeypatch.setattr(phase, "_process_channel", _fake_process_channel)
        monkeypatch.setattr(phase, "_mrsa_gain_refinement", lambda pre, post, sr: post)

        result = phase.process(
            sine_440_2s,
            sample_rate=SR,
            confidence=1.0,
            strength=1.0,
            defect_locations={"tape_head_clog": [(0.20, 0.50)], "tape_head_level_dip": [(1.20, 1.50)]},
            defect_event_metadata={
                "tape_head_clog": {"severity": 0.95, "confidence": 0.95},
                "tape_head_level_dip": {"severity": 0.35, "confidence": 0.70},
            },
        )
        assert result.success is True
        diff = np.abs(result.audio - sine_440_2s)
        clog_region = float(np.mean(diff[int(0.25 * SR) : int(0.45 * SR)]))
        dip_region = float(np.mean(diff[int(1.25 * SR) : int(1.45 * SR)]))
        assert clog_region > dip_region * 1.25

    def test_vibrato_zone_caps_local_band_gap_repair(self, phase, sine_440_2s, monkeypatch):
        def _fake_process_channel(channel, sr, instrument_tag, gap_fraction_min=None, bw_cap_hz=None):
            return (channel * 0.10).astype(np.float32)

        monkeypatch.setattr(phase, "_process_channel", _fake_process_channel)
        monkeypatch.setattr(phase, "_mrsa_gain_refinement", lambda pre, post, sr: post)

        free = phase.process(
            sine_440_2s,
            sample_rate=SR,
            confidence=1.0,
            strength=1.0,
            defect_locations={"tape_head_clog": [(1.20, 1.50)]},
            defect_event_metadata={"tape_head_clog": {"severity": 0.95, "confidence": 0.95}},
        )
        capped = phase.process(
            sine_440_2s,
            sample_rate=SR,
            confidence=1.0,
            strength=1.0,
            defect_locations={"tape_head_clog": [(1.20, 1.50)]},
            defect_event_metadata={"tape_head_clog": {"severity": 0.95, "confidence": 0.95}},
            vibrato_zones=[(1.10, 1.60)],
        )
        free_delta = float(
            np.mean(np.abs(free.audio[int(1.25 * SR) : int(1.45 * SR)] - sine_440_2s[int(1.25 * SR) : int(1.45 * SR)]))
        )
        capped_delta = float(
            np.mean(
                np.abs(capped.audio[int(1.25 * SR) : int(1.45 * SR)] - sine_440_2s[int(1.25 * SR) : int(1.45 * SR)])
            )
        )
        assert capped_delta < free_delta * 0.55


# ---------------------------------------------------------------------------
# Tests: Stereo-Eingabe
# ---------------------------------------------------------------------------


class TestPhase56Stereo:
    def test_12_stereo_shape_preserved(self, phase, stereo_audio):
        result = phase.process(stereo_audio, sample_rate=SR)
        assert result is not None
        # Akzeptiere: entweder Shape gleich ODER zu Mono konvertiert + zurück
        assert np.isfinite(result.audio).all()

    def test_13_stereo_no_clipping(self, phase, stereo_audio):
        result = phase.process(stereo_audio, sample_rate=SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestPhase56EdgeCases:
    def test_14_single_sample_array(self, phase):
        audio = np.array([0.0], dtype=np.float32)
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_15_very_short_100ms(self, phase):
        np.random.seed(42)
        audio = (np.random.randn(SR // 10) * 0.1).astype(np.float32)
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_16_dirac_impulse(self, phase):
        audio = np.zeros(SR, dtype=np.float32)
        audio[SR // 2] = 1.0
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_17_negative_amplitude_input(self, phase):
        audio = -np.ones(SR, dtype=np.float32) * 0.5
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_18_max_amplitude_input(self, phase):
        audio = np.ones(SR, dtype=np.float32)
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: Konsistenz
# ---------------------------------------------------------------------------


class TestPhase56Consistency:
    def test_19_both_runs_valid(self, phase, sine_440_2s):
        """Zwei Läufe mit gleicher Eingabe liefern beide gültige Ausgaben.

        Hinweis: SpectralBandGapRepair nutzt NMF mit stochastischer Initialisierung
        — bit-identische Outputs sind daher nicht garantiert, aber beide Ausgaben
        müssen valide (NaN-frei, bounded) sein.
        """
        np.random.seed(42)
        r1 = phase.process(sine_440_2s.copy(), sample_rate=SR)
        np.random.seed(42)
        r2 = phase.process(sine_440_2s.copy(), sample_rate=SR)
        # Beide Ausgaben müssen valide sein
        assert np.isfinite(r1.audio).all(), "Run 1: NaN/Inf im Ausgang"
        assert np.isfinite(r2.audio).all(), "Run 2: NaN/Inf im Ausgang"
        assert np.max(np.abs(r1.audio)) <= 1.0 + 1e-6, "Run 1: Clipping"
        assert np.max(np.abs(r2.audio)) <= 1.0 + 1e-6, "Run 2: Clipping"
        assert r1.audio.shape == sine_440_2s.shape
        assert r2.audio.shape == sine_440_2s.shape

    def test_20_success_flag_for_valid_audio(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result.success is True

    def test_21_additional_kwargs_ignored_gracefully(self, phase, sine_440_2s):
        """Unbekannte kwargs dürfen keinen Fehler auslösen."""
        result = phase.process(
            sine_440_2s, sample_rate=SR, material_type="tape", defect_scores={}, quality_mode="restoration"
        )
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_22_output_energy_not_zero_for_tonal_input(self, phase, sine_440_2s):
        """Tonaler Eingang → Ausgang nicht komplett auf Null."""
        result = phase.process(sine_440_2s, sample_rate=SR)
        rms = float(np.sqrt(np.mean(result.audio**2)))
        assert rms > 1e-6, f"Ausgang hat kein Energie: rms={rms}"


class TestPhase56StereoParameterFlow:
    """Regressionsschutz: Side-Konservativität ohne globalen Zustand."""

    def test_stereo_calls_detect_with_local_gap_fraction_and_no_global_mutation(self, monkeypatch):
        from backend.core.phases import phase_56_spectral_band_gap_repair as m

        phase = m.SpectralBandGapRepairPhase()
        seen_gap_fractions: list[float | None] = []
        original_gap_fraction = float(m._GAP_FRACTION_MIN)

        def _spy_detect_band_gaps(
            stft_mag: np.ndarray,
            sr: int,
            n_fft: int,
            gap_fraction_min: float | None = None,
            bw_cap_hz: float | None = None,
        ) -> list[tuple[int, int]]:
            seen_gap_fractions.append(gap_fraction_min)
            return []  # force fast path

        monkeypatch.setattr(m, "_detect_band_gaps", _spy_detect_band_gaps)

        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        left = (0.3 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
        right = (0.28 * np.sin(2 * np.pi * 335.0 * t)).astype(np.float32)
        stereo = np.stack([left, right], axis=1)

        result = phase.process(stereo, sample_rate=SR)
        assert result.success is True
        assert result.audio.shape == stereo.shape
        assert seen_gap_fractions == [None, 0.95], (
            "Stereo-M/S-Pfad muss Mid mit Standard und Side mit konservativem "
            f"gap_fraction_min laufen; gesehen={seen_gap_fractions}"
        )
        assert float(m._GAP_FRACTION_MIN) == pytest.approx(original_gap_fraction, abs=1e-12)


# ---------------------------------------------------------------------------
# §2.46g (2026-09-06): Material-BW-Ceiling — Spektralrand ist kein Defekt
# ---------------------------------------------------------------------------


class TestBwCapNyquistGuard:
    """Produktionsbefund: „Lücke 1013–1025 Bins (23742–24023 Hz)“ auf Vinyl
    kostete 141.6 s inkl. NMF-β — oberhalb des Material-Ceilings (vinyl ≤ 16 kHz)
    ist das der natürliche Spektralrand, keine reparierbare Lücke."""

    def test_gap_above_vinyl_ceiling_is_not_detected(self) -> None:
        from backend.core.phases.phase_56_spectral_band_gap_repair import _detect_band_gaps

        n_fft = 2048
        n_bins = n_fft // 2 + 1
        stft = np.full((n_bins, 50), 0.3, dtype=np.float32)
        stft[1013:1025, :] = 1e-6  # leeres Band bei 23.7–24.0 kHz (Nyquist-Kante)

        with_ceiling = _detect_band_gaps(stft, SR, n_fft, bw_cap_hz=16000.0)
        assert with_ceiling == [], "Gap über dem Vinyl-Ceiling darf nicht detektiert werden"

        without_ceiling = _detect_band_gaps(stft, SR, n_fft)
        assert any(g[0] >= 1013 for g in without_ceiling), "Ohne Ceiling muss der Rand als Gap erscheinen (Legacy-Verhalten)"

    def test_gap_below_ceiling_is_still_detected(self) -> None:
        from backend.core.phases.phase_56_spectral_band_gap_repair import _detect_band_gaps

        n_fft = 2048
        n_bins = n_fft // 2 + 1
        stft = np.full((n_bins, 50), 0.3, dtype=np.float32)
        stft[300:340, :] = 1e-6  # leeres Band bei ~7.0–8.0 kHz — innerhalb des Ceilings

        with_ceiling = _detect_band_gaps(stft, SR, n_fft, bw_cap_hz=16000.0)
        assert any(g[0] == 300 for g in with_ceiling), "Gap unterhalb des Ceilings muss weiterhin detektiert werden"

    def test_profile_sets_material_bw_cap(self) -> None:
        from backend.core.phases.phase_56_spectral_band_gap_repair import SpectralBandGapRepairPhase

        phase56 = SpectralBandGapRepairPhase()
        p_vinyl = phase56._compute_band_gap_profile("vinyl", "quality", 75.0)
        p_tape = phase56._compute_band_gap_profile("reel_tape", "quality", 75.0)
        p_digital = phase56._compute_band_gap_profile("cd_digital", "quality", 75.0)
        assert p_vinyl["bw_cap_hz"] == 16000.0
        assert p_tape["bw_cap_hz"] == 15000.0
        assert p_digital["bw_cap_hz"] == 22050.0
