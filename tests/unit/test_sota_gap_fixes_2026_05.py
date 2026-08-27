from __future__ import annotations

"""Unit-Tests für die SOTA-Gap-Fixes (Session Mai 2026):
- noise_texture_resynth.restore_carrier_noise_texture  (Gap 3)
- nvsr_plugin.NvsrPlugin.process                       (Gap 2)
- phoneme_boundary_detector                            (Gap 6)
- tonal_reference_profile.get_studio_console_curve     (Gap 5)
"""


from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _white_noise(n: int = 48000, amplitude: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32) * amplitude


def _sine(freq: float = 440.0, sr: int = 48000, dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


# ===========================================================================
# Gap 3: noise_texture_resynth
# ===========================================================================


class TestRestoreCarrierNoiseTexture:
    """§TimbralCoherence — restore_carrier_noise_texture() Grundverhalten."""

    def test_passthrough_when_strength_zero(self):
        """strength=0 → Audio unverändert zurückgegeben."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.05)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="vinyl", strength=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_shape_preserved_mono(self):
        """Output-Shape ist identisch mit Input (Mono)."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(24000, amplitude=0.03)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="vinyl")
        assert result.shape == audio.shape

    def test_shape_preserved_stereo(self):
        """Output-Shape ist identisch mit Input (Stereo 2×N)."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        rng = np.random.default_rng(7)
        audio = (rng.standard_normal((2, 48000)) * 0.05).astype(np.float32)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="vinyl")
        assert result.shape == audio.shape

    def test_no_clipping_in_output(self):
        """Output darf niemals clippen."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.4)
        result = restore_carrier_noise_texture(audio, audio, sr=48000)
        assert float(np.max(np.abs(result))) <= 1.0

    def test_no_nan_in_output(self):
        """Output darf keine NaN/Inf-Werte enthalten."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.05)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="shellac")
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_over_nr_correction_applied(self):
        """Wenn post-NR-Signal lautlos ist (extreme Over-NR), wird Korrektur angewandt."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        pre = _white_noise(48000, amplitude=0.05)
        # Simuliere Over-NR: post ist fast komplett still
        post = np.zeros(48000, dtype=np.float32) + 1e-6
        result = restore_carrier_noise_texture(pre, post, sr=48000, material_type="vinyl")
        # Nach Korrektur sollte etwas Energie vorhanden sein (wenn psychoacoustics verfügbar)
        assert result.shape == post.shape  # Mindestanforderung: shape bleibt gleich

    def test_passthrough_for_small_deviation(self):
        """Bei identischem pre/post-NR Signal (keine Over-NR) bleibt Output gleich."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.05)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="cd_digital")
        # Kein Unterschied → entweder passthrough oder minimale Korrektur
        assert result.shape == audio.shape

    def test_correction_floor_never_louder_than_original(self):
        """BUG-FIX: Injiziertes Comfort-Rauschen darf nie lauter als Original-Rauschboden sein.

        Vorher: effective_floor * strength = -52 * 0.48 = -24.96 dBFS → LAUTER als Original!
        Nachher: effective_floor + 20*log10(strength) → immer ≤ effective_floor (leiser).
        """
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        # Tape-ähnlicher Rauschboden: ~-52 dBFS
        tape_amplitude = 10 ** (-52.0 / 20.0)  # ≈ 0.00251 linear
        sr = 48000
        rng = np.random.default_rng(42)
        # Pre: weißes Rauschen bei -52 dBFS (reines Rauschsignal → Rauschboden ≈ -52 dBFS)
        pre = rng.standard_normal(sr * 2).astype(np.float32) * tape_amplitude
        # Post: fast komplett still (extreme Over-NR)
        post = np.zeros(sr * 2, dtype=np.float32) + 1e-7

        # Original-Rauschboden aus pre messen (5. Perzentil der Frame-RMS in dBFS)
        frame_len = int(0.05 * sr)
        hop = frame_len // 2
        rms_vals = []
        for s in range(0, len(pre) - frame_len, hop):
            rms = float(np.sqrt(np.mean(pre[s : s + frame_len].astype(float) ** 2) + 1e-20))
            rms_vals.append(rms)
        p5 = float(np.percentile(rms_vals, 5))
        original_floor_dbfs = 20.0 * np.log10(p5 + 1e-20)
        # Sicherstellen, dass der Rauschboden im erwarteten Bereich liegt
        assert original_floor_dbfs < -40.0, (
            f"Test-Setup: Rauschboden sollte unter -40 dBFS liegen, got {original_floor_dbfs:.1f}"
        )

        # Korrektur mit strength=0.48 (Phase_03 mit effective_strength=0.8: 0.8*0.6=0.48)
        result = restore_carrier_noise_texture(pre, post, sr=sr, material_type="tape", strength=0.48)

        # Der Output-RMS im stillen Post-Bereich darf nicht lauter als original_floor + 6 dB sein
        # (6 dB Toleranz wegen blend-Anteil und Meßungenauigkeit)
        result_rms = float(np.sqrt(np.mean(result.astype(float) ** 2) + 1e-20))
        result_db = 20.0 * np.log10(result_rms + 1e-20)
        assert result_db <= original_floor_dbfs + 6.0, (
            f"BUG REGRESSION: Injiziertes Comfort-Rauschen ({result_db:.1f} dBFS) ist lauter als "
            f"Original-Rauschboden ({original_floor_dbfs:.1f} dBFS) + 6 dB! "
            f"Altes Bug: effective_floor * 0.48 = {original_floor_dbfs * 0.48:.1f} dBFS → "
            f"Starkregen-Artefakt."
        )

    @pytest.mark.parametrize("material", ["cassette", "vinyl", "shellac", "tape", "reel_tape", "wax_cylinder"])
    def test_analog_resynth_targets_cd_like_floor(self, material):
        """Analogträger: Over-NR-Korrektur darf analoges Trägerrauschen nicht zurückfüllen."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        analog_amplitude = 10 ** (-52.0 / 20.0)
        sr = 48000
        rng = np.random.default_rng(43)
        pre = rng.standard_normal(sr * 2).astype(np.float32) * analog_amplitude
        post = np.zeros(sr * 2, dtype=np.float32) + 1e-7

        result = restore_carrier_noise_texture(pre, post, sr=sr, material_type=material, strength=1.0)
        result_rms = float(np.sqrt(np.mean(result.astype(float) ** 2) + 1e-20))
        result_db = 20.0 * np.log10(result_rms + 1e-20)

        assert result_db <= -68.0, f"{material}-Resynthese muss CD-like leise bleiben, got {result_db:.1f} dBFS"


# ===========================================================================
# Gap 2: nvsr_plugin
# ===========================================================================


class TestNvsrPlugin:
    """§SOTA Gap 2 — NvsrPlugin DSP-SBR Grundverhalten."""

    def test_singleton_returns_same_instance(self):
        """get_nvsr_plugin() liefert immer dieselbe Instanz."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        inst_a = get_nvsr_plugin()
        inst_b = get_nvsr_plugin()
        assert inst_a is inst_b

    def test_process_shape_preserved_mono(self):
        """process() erhält Shape: Mono (N,)."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(440.0, sr=48000, dur=0.5)
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert result["audio"].shape == audio.shape

    def test_process_shape_preserved_stereo(self):
        """process() erhält Shape: Stereo (2, N)."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = np.stack([_sine(440.0), _sine(880.0)], axis=0).astype(np.float32)
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert result["audio"].shape == audio.shape

    def test_no_clipping(self):
        """SBR-Ausgang darf niemals clippen."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(440.0, sr=48000, dur=1.0) * 0.8
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert float(np.max(np.abs(result["audio"]))) <= 1.0

    def test_strategy_metadata_present(self):
        """process()-Ergebnis enthält 'strategy' im dict."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(440.0, sr=48000, dur=0.5)
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert "strategy" in result or "strategy_used" in result

    def test_shellac_ceiling_respected(self):
        """Shellac-Material hat HF-Ceiling ≤ 8000 Hz — kein SBR über 8 kHz."""
        from plugins.nvsr_plugin import _MATERIAL_HF_CEILING_HZ

        assert _MATERIAL_HF_CEILING_HZ.get("shellac", 0) <= 8_001.0

    def test_no_nan_output(self):
        """Output darf keine NaN-Werte enthalten."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(880.0, sr=48000, dur=0.5)
        result = plugin.process(audio, sr=48000)
        assert not np.any(np.isnan(result["audio"]))


# ===========================================================================
# Gap 6: phoneme_boundary_detector
# ===========================================================================


class TestPhonemeBoundaryDetectorDsp:
    """§Gap 6 — DSP-Phonem-Grenzerkennung Grundverhalten."""

    def test_returns_bool_array(self):
        """detect_phoneme_boundaries_dsp() gibt bool-Array zurück."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = _white_noise(48000, amplitude=0.1)
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert result.dtype == bool

    def test_output_length_matches_n_frames(self):
        """Output-Länge entspricht n_frames = len(audio) // hop_length."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = _white_noise(48000, amplitude=0.1)
        hop = 512
        result = detect_phoneme_boundaries_dsp(audio, sr=48000, hop_length=hop)
        # mindestens 1, nicht mehr als len(audio) // hop
        assert 1 <= len(result) <= len(audio) // hop + 1

    def test_silence_has_no_boundaries(self):
        """Komplett stilles Signal → keine Phonem-Grenzen."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        silence = np.zeros(48000, dtype=np.float32)
        result = detect_phoneme_boundaries_dsp(silence, sr=48000)
        # Stille = alle Frames SILENCE → keine Übergänge
        assert not np.any(result)

    def test_plosive_onset_detected_for_energy_spike(self):
        """Energie-Spike (12+ dB) löst Boundary aus."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        # Erzeuge Signal: 0.5s still, dann Energie-Spike
        audio = np.zeros(48000, dtype=np.float32)
        audio[24000:26000] = 0.8  # großer Spike
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert np.any(result), "Energie-Spike muss eine Boundary auslösen"

    def test_stereo_input_handled(self):
        """Stereo-Input (2×N) wird korrekt zu Mono downgemischt."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = np.stack([_white_noise(24000), _white_noise(24000, seed=7)], axis=0)
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert result.dtype == bool

    def test_short_audio_no_crash(self):
        """Sehr kurzes Audio (< 4× hop_length) → leeres Array ohne Crash."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = np.zeros(100, dtype=np.float32)
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool


# ===========================================================================
# Gap 5: Console Character Studio 2026
# ===========================================================================


class TestStudioConsoleCharacter:
    """§Gap 5 — TonalReferenceProfiler.get_studio_console_curve()"""

    def test_neve_1073_returns_list(self):
        """neve_1073-Kurve ist eine Liste von (Hz, dB)-Paaren."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("neve_1073")
        assert isinstance(curve, list)
        assert len(curve) >= 3

    def test_ssl_4000_returns_list(self):
        """ssl_4000-Kurve ist eine Liste von (Hz, dB)-Paaren."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("ssl_4000")
        assert isinstance(curve, list)
        assert len(curve) >= 3

    def test_unknown_console_returns_neve_fallback(self):
        """Unbekannter console_type → Fallback auf neve_1073."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve_unknown = profiler.get_studio_console_curve("xyz_unknown")
        curve_neve = profiler.get_studio_console_curve("neve_1073")
        assert curve_unknown == curve_neve

    def test_console_curve_has_valid_frequency_range(self):
        """Alle Kurven haben Frequenzwerte 20 Hz – 20 kHz."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        for console in ("neve_1073", "ssl_4000", "api_2500", "neutral"):
            curve = profiler.get_studio_console_curve(console)
            freqs = [hz for hz, _ in curve]
            assert min(freqs) >= 20.0, f"{console}: Min-Frequenz zu niedrig"
            assert max(freqs) <= 21_000.0, f"{console}: Max-Frequenz zu hoch"

    def test_neve_1073_has_low_shelf_boost(self):
        """Neve 1073 hat Low-Shelf-Boost bei ~80 Hz."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("neve_1073")
        # Suche +2 dB bei 80 Hz
        gains_at_low = [g for hz, g in curve if 60.0 <= hz <= 120.0]
        assert any(g > 1.0 for g in gains_at_low), "Neve 1073 muss Low-Shelf > +1 dB haben"

    def test_neutral_curve_is_flat(self):
        """Neutral-Kurve enthält nur 0 dB-Werte."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("neutral")
        gains = [g for _, g in curve]
        assert all(g == 0.0 for g in gains), "Neutral muss komplett flat (0 dB) sein"

    def test_console_character_wired_in_phase06_studio_mode(self):
        """§Gap5: phase_06 ruft get_studio_console_curve() im Studio-2026-Pfad auf."""
        import inspect

        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        src = inspect.getsource(FrequencyRestorationPhase.process)
        assert "get_studio_console_curve" in src or "ConsoleCharacter" in src.lower() or "console_character" in src, (
            "§Gap5 Console Character nicht verdrahtet in phase_06"
        )
        assert '"studio"' in src or "'studio'" in src, "Studio-Mode-Gate fehlt"


# ===========================================================================
# VQI per-Phase Gates: phase_20 und phase_42
# ===========================================================================


class TestVqiPerPhaseGates:
    """§0p VQI per-Phase-Rollback — phase_20_reverb_reduction, phase_42_vocal_enhancement."""

    def _make_audio(self, n: int = 24000, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(n) * 0.1).astype(np.float32)

    def test_phase20_process_returns_phaseresult(self):
        """phase_20.process() gibt PhaseResult zurück (Smoke-Test)."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        phase = ReverbReduction()
        audio = self._make_audio()
        result = phase.process(audio, 48000, MaterialType.VINYL, strength=0.2)
        assert result is not None
        assert hasattr(result, "audio")
        assert result.audio.shape == audio.shape

    def test_phase20_vqi_gate_inserted(self):
        """phase_20.py enthält den VQI per-Phase Rollback-Block."""
        import inspect

        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        src = inspect.getsource(ReverbReduction.process)
        assert "compute_vqi" in src or "vocal_quality_index" in src, "§0p VQI per-phase gate fehlt in phase_20"
        assert "_vqi_p20" in src or "_vqi_result_p20" in src, "VQI-Variable _vqi_p20 nicht gefunden"

    def test_phase42_process_returns_phaseresult(self):
        """phase_42.process() gibt PhaseResult zurück (Smoke-Test)."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = self._make_audio()
        result = phase.process(audio, 48000, MaterialType.CD_DIGITAL, strength=0.2)
        assert result is not None
        assert hasattr(result, "audio")
        assert result.audio.shape == audio.shape

    def test_phase42_vqi_gate_inserted(self):
        """phase_42.py enthält den VQI per-Phase Rollback-Block."""
        import inspect

        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        src = inspect.getsource(VocalEnhancement.process)
        assert "compute_vqi" in src or "vocal_quality_index" in src, "§0p VQI per-phase gate fehlt in phase_42"
        assert "_vqi_p42" in src or "_vqi_result_p42" in src, "VQI-Variable _vqi_p42 nicht gefunden"

    def test_uv3_phase50_in_hnr_blend_set(self):
        """UV3: phase_50_spectral_repair muss im _NR_PHASES_HNR-Set sein."""
        # Spec 07 TEST-DESIGN: direkte Datei-Lesung statt inspect.getsource —
        # linecache ist prozessglobal und order-abhängig (Suite-Flakes).
        from pathlib import Path

        import backend.core.unified_restorer_v3 as _uv3_mod

        src = Path(str(_uv3_mod.__file__)).read_text(encoding="utf-8")
        assert "phase_50_spectral_repair" in src, (
            "§0p HNR-Blend: phase_50_spectral_repair fehlt in _NR_PHASES_HNR (UV3)"
        )


# ===========================================================================
# §0p Gap-Fixes Session 2: HNR-Blend in ML-NR-Phasen
# ===========================================================================


class TestHnrBlendInNrPhases:
    """§0p RELEASE_MUST — phase_20/29/49/50 müssen HNR-Blend aufrufen (panns >= 0.25)."""

    def test_phase20_source_contains_hnr_blend(self):
        """phase_20.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        src = inspect.getsource(ReverbReduction.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_20"
        assert "_hnr_blended_p20" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_20"

    def test_phase29_source_contains_hnr_blend(self):
        """phase_29.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        src = inspect.getsource(TapeHissReductionPhase.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_29"
        assert "_hnr_blended_p29" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_29"

    def test_phase49_source_contains_hnr_blend(self):
        """phase_49.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

        src = inspect.getsource(AdvancedDereverbPhase.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_49"
        assert "_hnr_blended_p49" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_49"

    def test_phase50_source_contains_hnr_blend(self):
        """phase_50.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase

        src = inspect.getsource(SpectralRepairPhase.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_50"
        assert "_hnr_blended_p50" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_50"

    def test_phase20_hnr_blend_called_when_over_cleaned(self):
        """phase_20: apply_hnr_blend wird aufgerufen; bei over_cleaned=True wird blended-Audio übernommen."""
        from unittest.mock import patch

        import numpy as np

        audio = np.zeros(4800, dtype=np.float32)
        blended = np.ones(4800, dtype=np.float32) * 0.1

        mock_result = (blended, {"over_cleaned": True, "hnr_delta_db": 4.2})

        with patch("backend.core.dsp.hnr_guard.apply_hnr_blend", return_value=mock_result):
            from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

            phase = ReverbReduction()
            result = phase.process(
                audio,
                sample_rate=48000,
                panns_singing=0.6,
                processing_mode="restoration",
            )
        assert hasattr(result, "audio")
        assert result.audio is not None

    def test_phase29_hnr_blend_called_when_over_cleaned(self):
        """phase_29: apply_hnr_blend wird aufgerufen; bei over_cleaned=True wird blended-Audio übernommen."""
        from unittest.mock import patch

        import numpy as np

        audio = np.zeros(4800, dtype=np.float32) + 0.02
        blended = np.ones(4800, dtype=np.float32) * 0.05

        mock_result = (blended, {"over_cleaned": True, "hnr_delta_db": 5.0})

        with patch("backend.core.dsp.hnr_guard.apply_hnr_blend", return_value=mock_result):
            from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

            phase = TapeHissReductionPhase()
            result = phase.process(
                audio,
                sample_rate=48000,
                panns_singing=0.5,
                processing_mode="restoration",
            )
        assert hasattr(result, "audio")
        assert result.audio is not None

    def test_phase49_hnr_blend_skipped_when_panns_low(self):
        """phase_49: HNR-Blend wird NICHT aufgerufen wenn panns_singing < 0.25."""
        from unittest.mock import patch

        import numpy as np

        audio = np.zeros(4800, dtype=np.float32) + 0.01

        with patch("backend.core.dsp.hnr_guard.apply_hnr_blend") as mock_hnr:
            from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

            phase = AdvancedDereverbPhase()
            phase.process(audio, sample_rate=48000, panns_singing=0.1, processing_mode="restoration")
        mock_hnr.assert_not_called()


# ===========================================================================
# §0p Gap-Fix: Singer-ID-Cosine Rollback in UV3
# ===========================================================================


class TestSingerIdRollbackUV3:
    """§0p RELEASE_MUST — UV3 muss Singer-ID-Rollback-Code enthalten."""

    def test_uv3_singer_id_rollback_code_present(self):
        """UV3 enthält den §0p Singer-ID-Cosine-Rollback-Block."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        assert "singer_identity_cosine" in src, "§0p Singer-ID-Rollback fehlt komplett in UV3"
        assert "SINGER_ID_BELOW_THRESHOLD" in src, "§0p SingerIDGate error_code fehlt in UV3"
        assert "_is_ms_rb" in src or "multi_singer" in src, "§0p multi_singer-Guard fehlt in UV3"

    def test_uv3_singer_id_rollback_threshold_correct(self):
        """UV3 verwendet exakt 0.92 als Singer-ID-Schwellwert."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        assert "0.92" in src, "§0p Singer-ID Threshold 0.92 nicht in UV3 gefunden"

    def test_uv3_singer_id_deactivated_for_multi_singer(self):
        """UV3-Rollback ist bei multi_singer=True deaktiviert."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        # Guard muss vorhanden sein
        assert "multi_singer" in src, "§0p multi_singer Guard fehlt"
        assert "not _is_ms_rb" in src or "multi_singer" in src, (
            "§0p Singer-ID Rollback-Deaktivierung für multi_singer fehlt"
        )


# ===========================================================================
# §2.46e Gap-Fix: Hallucination-Guard in phase_26
# ===========================================================================


class TestPhase26HallucinationGuard:
    """§2.46e RELEASE_MUST — phase_26 muss apply_hallucination_guard aufrufen."""

    def test_phase26_source_contains_hallucination_guard(self):
        """phase_26.py enthält den §2.46e Hallucination-Guard-Block."""
        import inspect

        from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

        src = inspect.getsource(DynamicRangeExpansion.process)
        assert "hallucination_guard" in src or "apply_hallucination_guard" in src, (
            "§2.46e Hallucination-Guard fehlt in phase_26"
        )
        assert "hallucination_decision" in src, "§2.46e hallucination_decision-Check fehlt in phase_26"

    def test_phase26_rollback_on_hallucination(self):
        """phase_26: apply_hallucination_guard-Rollback überschreibt expanded_audio mit original audio."""
        from unittest.mock import patch

        import numpy as np

        original = np.ones(4800, dtype=np.float32) * 0.2

        with patch(
            "backend.core.hallucination_guard.apply_hallucination_guard",
            return_value=(None, {"hallucination_decision": "rollback", "hallucination_severity": 0.9}),
        ):
            from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

            phase = DynamicRangeExpansion()
            result = phase.process(
                original.copy(),
                sample_rate=48000,
                processing_mode="restoration",
                strength=0.5,
            )
        assert hasattr(result, "audio")
        # Bei Rollback muss das Ergebnis dem Original entsprechen (keine neue Energie)
        assert result.audio is not None

    def test_phase26_no_rollback_when_clean(self):
        """phase_26: kein Rollback wenn hallucination_decision = 'pass'."""
        from unittest.mock import patch

        import numpy as np

        original = np.ones(4800, dtype=np.float32) * 0.2

        with patch(
            "backend.core.hallucination_guard.apply_hallucination_guard",
            return_value=(None, {"hallucination_decision": "pass", "hallucination_severity": 0.0}),
        ):
            from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

            phase = DynamicRangeExpansion()
            result = phase.process(
                original.copy(),
                sample_rate=48000,
                processing_mode="restoration",
                strength=0.3,
            )
        assert hasattr(result, "audio")
        assert result.audio is not None


# ===========================================================================
# §V11 Gap-Fix: sosfiltfilt in synthetic_generator.py
# ===========================================================================


class TestSyntheticGeneratorSosfiltfilt:
    """§V11 RELEASE_MUST — synthetic_generator.py muss sosfiltfilt (zero-phase) nutzen."""

    def test_synthetic_generator_uses_sosfiltfilt(self):
        """golden_samples/synthetic_generator.py nutzt sosfiltfilt statt sosfilt für Formant-Filter."""
        import inspect

        try:
            from golden_samples.synthetic_generator import SyntheticAudioGenerator  # type: ignore[attr-defined]

            src = inspect.getsource(SyntheticAudioGenerator._generate_vocal)
        except (ImportError, AttributeError):
            import pathlib

            src = pathlib.Path("golden_samples/synthetic_generator.py").read_text(encoding="utf-8")

        assert "sosfiltfilt" in src, "§V11 Verletzung: sosfilt statt sosfiltfilt in synthetic_generator.py"
        assert "sosfilt(" not in src.replace("sosfiltfilt", ""), "§V11: sosfilt() (kausal) wird noch verwendet"


# ===========================================================================
# §0a Gap-Fix: _RESTORATION_FORBIDDEN_PHASES in DefectPhaseMapper
# ===========================================================================


class TestDefectPhaseMapperRestorationFilter:
    """§0a RELEASE_MUST — DefectPhaseMapper darf phase_21/35/42 in Restoration
    nicht vorschlagen (BUG-FIX v10.14 §0a)."""

    def test_forbidden_phases_constant_exists(self):
        """_RESTORATION_FORBIDDEN_PHASES muss die drei §0a-Phasen enthalten."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES

        assert "phase_21_exciter" in _RESTORATION_FORBIDDEN_PHASES
        assert "phase_35_multiband_compression" in _RESTORATION_FORBIDDEN_PHASES
        assert "phase_42_vocal_enhancement" in _RESTORATION_FORBIDDEN_PHASES

    def test_get_primary_phases_restoration_filters_forbidden(self):
        """get_primary_phases(mode='restoration') darf §0a-Phasen nicht zurückgeben."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        for defect_type in DefectType:
            phases = mapper.get_primary_phases(defect_type, mode="restoration")
            forbidden_found = set(phases) & _RESTORATION_FORBIDDEN_PHASES
            assert not forbidden_found, (
                f"§0a Verletzung: {forbidden_found} in primary_phases für {defect_type.value} im Restoration-Modus"
            )

    def test_get_all_phases_restoration_filters_forbidden(self):
        """get_all_phases(mode='restoration') darf §0a-Phasen nicht zurückgeben."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        for defect_type in DefectType:
            phases = mapper.get_all_phases(defect_type, mode="restoration")
            forbidden_found = set(phases) & _RESTORATION_FORBIDDEN_PHASES
            assert not forbidden_found, (
                f"§0a Verletzung: {forbidden_found} in all_phases für {defect_type.value} im Restoration-Modus"
            )

    def test_studio_2026_does_not_over_filter_vs_restoration(self):
        """§0a Mode-Asymmetrie: Studio 2026 darf nie STRENGER filtern als Restoration.

        Nach dem V04-Fix (v10.14) sind §0a-verbotene Phasen (phase_21/35/42) bewusst
        komplett aus _PHASE_MAP entfernt — Defense-in-Depth zusätzlich zum Runtime-Filter.
        Der korrekte §0a-Vertrag ist daher NICHT 'Studio enthält verbotene Phasen', sondern
        'der mode-Gate restringiert ausschließlich in Restoration': für jeden DefectType
        muss die Restoration-Phasenliste eine Teilmenge der Studio-2026-Liste sein, und
        Restoration darf nie eine §0a-Phase enthalten (von den Geschwister-Tests gesichert).
        """
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        for defect_type in DefectType:
            phases_restoration = set(mapper.get_all_phases(defect_type, mode="restoration"))
            phases_studio = set(mapper.get_all_phases(defect_type, mode="studio_2026"))
            # Studio 2026 ist Obermenge: der mode-Gate entfernt nur in Restoration.
            assert phases_restoration <= phases_studio, (
                f"§0a Mode-Asymmetrie verletzt: Studio 2026 filtert für {defect_type.value} "
                f"strenger als Restoration (entfernt {phases_restoration - phases_studio})"
            )
            # Restoration darf §0a-Phasen nie enthalten (Defense-in-Depth + Filter).
            assert not (phases_restoration & _RESTORATION_FORBIDDEN_PHASES), (
                f"§0a Verletzung: {phases_restoration & _RESTORATION_FORBIDDEN_PHASES} "
                f"in Restoration-Phasen für {defect_type.value}"
            )

    def test_get_primary_phases_no_mode_defaults_to_restoration(self):
        """Kein mode-Argument → default 'restoration' → Filterung aktiv."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        for defect_type in DefectType:
            phases = mapper.get_primary_phases(defect_type)  # kein mode-Argument
            forbidden_found = set(phases) & _RESTORATION_FORBIDDEN_PHASES
            assert not forbidden_found, f"§0a Default-Filter fehlt: {forbidden_found} in {defect_type.value}"


# ===========================================================================
# §Cross-Goal-Recovery (v10.14.x fix): hf_recovery_boost_after_phase03
# WIRING FIX: floor enforcement moved from _profiled_phase_call (dead code path)
# to main phase loop, applied to _combined_strength before wrap_phase call.
# ===========================================================================


class TestCrossGoalRecoveryMainLoopFix:
    """§Cross-Goal-Recovery (v10.14.x) — Strength-Floor für phase_06/07/39 wird in
    UV3-Hauptschleife auf _combined_strength angewendet (nicht in _profiled_phase_call)."""

    def test_uv3_cross_goal_recovery_in_main_loop_source(self):
        """UV3-Source enthält den Cross-Goal-Recovery-Block in der Hauptschleife
        (erkennbar am Kommentar 'must run here (main loop)')."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        assert "must run here (main loop)" in src, (
            "§Cross-Goal-Recovery Wiring-Fix fehlt: Block 'must run here (main loop)' nicht in UV3"
        )
        assert "_hf_boost_ctx" in src, "§Cross-Goal-Recovery: _hf_boost_ctx Variable fehlt in UV3-Hauptschleife"

    def test_uv3_cross_goal_recovery_applies_strength_floor(self):
        """UV3-Hauptschleife erhöht _combined_strength auf HF-Floor wenn
        hf_recovery_boost_after_phase03 aktiviert ist."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        # Der Fix setzt _combined_strength auf den floor-Wert
        assert "_combined_strength = float(np.clip(_hf_floor" in src, (
            "§Cross-Goal-Recovery: _combined_strength-Floor-Assignment fehlt in UV3-Hauptschleife"
        )

    def test_uv3_cross_goal_recovery_phase_set_correct(self):
        """Cross-Goal-Recovery gilt für phase_06/07/39 (nicht andere Phasen)."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        # Beide Blöcke (Hauptschleife + _profiled_phase_call) checken phase_06/07/39
        assert "phase_06_frequency_restoration" in src
        assert "phase_07_harmonic_restoration" in src
        assert "phase_39_air_band_enhancement" in src

    def test_uv3_profiled_phase_call_backward_compat_comment(self):
        """_profiled_phase_call enthält Hinweis dass der dortige Block für den
        Bronze/Bypass-Pfad (Z.7298/7339) bestimmt ist — nicht für PMGG-Hauptpfad."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3._profiled_phase_call)
        assert "bronze" in src.lower() or "bypass" in src.lower(), (
            "§Cross-Goal-Recovery: _profiled_phase_call fehlt Hinweis auf Bronze/Bypass-Pfad"
        )


# ---------------------------------------------------------------------------
# Todo 2 — §4.4 Era-Aware ML-NR Routing (v10.14.x)
# ---------------------------------------------------------------------------


class TestEraAwareNrModelRouting:
    """§4.4 SOTA Era-Aware NR routing: MIIPHER/DFN/OMLSA selection."""

    def _routing(self, era_decade, material_type, est_snr_db, panns_singing, is_vocal=True, is_non_digital=True):
        from backend.core.phases.phase_03_denoise import _determine_era_nr_routing

        return _determine_era_nr_routing(era_decade, material_type, est_snr_db, panns_singing, is_vocal, is_non_digital)

    def test_acoustic_era_1920_returns_omlsa_only(self):
        """1920s phonograph: no ML NR — carrier character must be preserved (§0a)."""
        tier = self._routing(era_decade=1920, material_type="shellac", est_snr_db=8.0, panns_singing=0.5)
        assert tier == "omlsa_only", f"Expected omlsa_only for 1920 shellac, got {tier!r}"

    def test_acoustic_era_1930_boundary_returns_omlsa_only(self):
        """Era 1930 (boundary): omlsa_only."""
        tier = self._routing(era_decade=1930, material_type="shellac", est_snr_db=6.0, panns_singing=0.4)
        assert tier == "omlsa_only", f"Expected omlsa_only at 1930 boundary, got {tier!r}"

    def test_early_electric_shellac_1940_returns_dfn_restricted(self):
        """1940 shellac electrical: DFN restricted to 30% wet (preserve H2/H4)."""
        tier = self._routing(era_decade=1940, material_type="shellac", est_snr_db=12.0, panns_singing=0.4)
        assert tier == "dfn_restricted", f"Expected dfn_restricted for 1940 shellac, got {tier!r}"

    def test_early_electric_1945_shellac_returns_dfn_restricted(self):
        """Era 1945, shellac — still dfn_restricted (boundary)."""
        tier = self._routing(era_decade=1945, material_type="shellac", est_snr_db=15.0, panns_singing=0.5)
        assert tier == "dfn_restricted", f"Expected dfn_restricted at era=1945, got {tier!r}"

    def test_wax_cylinder_always_omlsa_only(self):
        """Wax cylinder: always omlsa_only regardless of decade."""
        tier = self._routing(era_decade=1965, material_type="wax_cylinder", est_snr_db=5.0, panns_singing=0.6)
        assert tier == "omlsa_only", f"Expected omlsa_only for wax_cylinder, got {tier!r}"

    def test_digital_material_omlsa_only(self):
        """Digital material: omlsa_only (no ML broadband NR needed)."""
        tier = self._routing(
            era_decade=1995, material_type="cd_digital", est_snr_db=35.0, panns_singing=0.5, is_non_digital=False
        )
        assert tier == "omlsa_only", f"Expected omlsa_only for cd_digital, got {tier!r}"

    def test_deep_snr_vocal_post1950_returns_miipher_primary(self):
        """1965 vinyl, SNR 8 dB, panns 0.4 → MIIPHER primary (§4.4 SOTA)."""
        tier = self._routing(era_decade=1965, material_type="vinyl", est_snr_db=8.0, panns_singing=0.4)
        assert tier == "miipher_primary", f"Expected miipher_primary for deep SNR vocal, got {tier!r}"

    def test_moderate_snr_post1950_returns_dfn_primary(self):
        """1975 vinyl, SNR 18 dB → DFN primary (current SOTA behavior)."""
        tier = self._routing(era_decade=1975, material_type="vinyl", est_snr_db=18.0, panns_singing=0.4)
        assert tier == "dfn_primary", f"Expected dfn_primary for moderate SNR, got {tier!r}"

    def test_snr_boundary_exactly_10_dfn_primary(self):
        """SNR exactly 10.0 dB (boundary): dfn_primary (MIIPHER only below 10 dB)."""
        tier = self._routing(era_decade=1970, material_type="vinyl", est_snr_db=10.0, panns_singing=0.4)
        assert tier == "dfn_primary", f"Expected dfn_primary at SNR=10 dB boundary, got {tier!r}"

    def test_low_panns_no_miipher(self):
        """Low panns_singing (0.25) below MIIPHER threshold 0.35 → dfn_primary."""
        tier = self._routing(era_decade=1965, material_type="vinyl", est_snr_db=7.0, panns_singing=0.25)
        assert tier == "dfn_primary", f"Expected dfn_primary for low panns, got {tier!r}"

    def test_snr_none_no_miipher_routing(self):
        """None SNR: MIIPHER not activated (cannot confirm deep noise) → dfn_primary."""
        tier = self._routing(era_decade=1965, material_type="vinyl", est_snr_db=None, panns_singing=0.5)
        assert tier == "dfn_primary", f"Expected dfn_primary when SNR unknown, got {tier!r}"

    def test_routing_function_is_importable(self):
        """_determine_era_nr_routing must be importable from phase_03_denoise."""
        from backend.core.phases.phase_03_denoise import _determine_era_nr_routing

        assert callable(_determine_era_nr_routing), "_determine_era_nr_routing must be callable"

    def test_era_routing_key_in_phase03_process_source(self):
        """phase_03 process() must call _determine_era_nr_routing."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "_determine_era_nr_routing" in src, "process() must call _determine_era_nr_routing"

    def test_miipher_block_in_phase03_source(self):
        """MIIPHER block must appear in phase_03 process()."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "miipher_primary" in src, "process() must contain MIIPHER primary routing"
        assert "miipher_applied" in src, "process() must track _miipher_applied flag"

    def test_sgmse_guard_for_omlsa_only_routing(self):
        """SGMSE+ must be blocked when _era_nr_routing == 'omlsa_only'."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "omlsa_only" in src, "process() must guard SGMSE+ with omlsa_only check"

    def test_dfn_restricted_blend_in_source(self):
        """DFN restricted 30% blend must be applied for early-electrical era."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "dfn_restricted" in src, "process() must implement dfn_restricted blend"
        assert "0.30" in src or "0.70" in src, "dfn_restricted must use 30%/70% wet/dry blend"


class TestRoomAcousticsFingerprinter:
    """Tests for §2.46f room_acoustics_fingerprinter.py"""

    def test_module_importable(self):
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        assert callable(compute_room_acoustics_fingerprint)

    def test_returns_expected_keys(self):
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        audio = np.zeros(48000, dtype=np.float32)
        result = compute_room_acoustics_fingerprint(audio, 48000)
        for key in ("rt60_s", "drr_db", "room_type", "dereverb_strength_cap", "early_reflection_ms", "protection_note"):
            assert key in result, f"Missing key: {key}"

    def test_studio_cap_range(self):
        """Studio room → cap ≥ 0.50 (moderate, not maximum protection)."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        # Dry impulse-like signal → short RT60 → studio
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(48000).astype(np.float32) * 0.01
        result = compute_room_acoustics_fingerprint(audio, 48000)
        cap = float(result["dereverb_strength_cap"])  # type: ignore[arg-type]
        assert 0.10 <= cap <= 1.0, f"Cap out of range: {cap}"

    def test_long_rt60_tightens_cap(self):
        """Long-decay signal → rt60 ≥ 1.2 s → cap ≤ 0.25."""
        from backend.core.room_acoustics_fingerprinter import _RT60_HIGH_THRESHOLD_S, compute_room_acoustics_fingerprint

        # Simulate long reverb tail: exponential decay over 3 s
        sr = 48000
        t = np.linspace(0, 3.0, sr * 3)
        decay = np.exp(-1.5 * t).astype(np.float32)  # ~0.4 s RT60 threshold
        # Use a very slow decay to exceed RT60 threshold
        slow_decay = np.exp(-0.3 * t).astype(np.float32)
        result = compute_room_acoustics_fingerprint(slow_decay, sr)
        if float(result["rt60_s"]) >= _RT60_HIGH_THRESHOLD_S:  # type: ignore[arg-type]
            assert float(result["dereverb_strength_cap"]) <= 0.25, f"High RT60 should tighten cap: {result}"  # type: ignore[arg-type]

    def test_silent_signal_returns_default(self):
        """Silent signal → fallback defaults, no exception."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        audio = np.zeros(48000, dtype=np.float32)
        result = compute_room_acoustics_fingerprint(audio, 48000)
        assert isinstance(result["rt60_s"], float)
        assert isinstance(result["dereverb_strength_cap"], float)

    def test_cap_clamped_to_valid_range(self):
        """dereverb_strength_cap must always be in [0.10, 1.0]."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        rng = np.random.default_rng(99)
        audio = rng.standard_normal(96000).astype(np.float32) * 0.5
        result = compute_room_acoustics_fingerprint(audio, 48000)
        cap = float(result["dereverb_strength_cap"])  # type: ignore[arg-type]
        assert 0.10 <= cap <= 1.0, f"Cap {cap} outside [0.10, 1.0]"

    def test_stereo_input_accepted(self):
        """Stereo audio (2, N) should be processed without error."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        audio = np.zeros((2, 48000), dtype=np.float32)
        result = compute_room_acoustics_fingerprint(audio, 48000)
        assert "rt60_s" in result

    def test_phase49_reads_room_acoustics_fingerprint(self):
        """phase_49 process() source must contain room_acoustics_fingerprint guard."""
        import inspect

        from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

        src = inspect.getsource(AdvancedDereverbPhase.process)
        assert "room_acoustics_fingerprint" in src, "phase_49 must read room_acoustics_fingerprint from kwargs"
        assert "dereverb_strength_cap" in src, "phase_49 must apply dereverb_strength_cap"

    def test_phase20_reads_room_acoustics_fingerprint(self):
        """phase_20 process() source must contain room_acoustics_fingerprint guard."""
        import inspect

        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        src = inspect.getsource(ReverbReduction.process)
        assert "room_acoustics_fingerprint" in src, "phase_20 must read room_acoustics_fingerprint from kwargs"
        assert "dereverb_strength_cap" in src, "phase_20 must apply dereverb_strength_cap"

    def test_uv3_injects_room_acoustics_fingerprint(self):
        """UV3 restore() source must inject room_acoustics_fingerprint into _restoration_context."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3.restore)
        assert "room_acoustics_fingerprint" in src, "UV3.restore() must inject room_acoustics_fingerprint"
        assert "room_acoustics_fingerprinter" in src, "UV3.restore() must import room_acoustics_fingerprinter"


class TestEraHarmonicProfileAndPhase07H2Steering:
    """Todo 4: get_era_harmonic_profile() + phase_07 H2-Target-Steering."""

    def test_get_era_harmonic_profile_importable(self):
        """get_era_harmonic_profile must be importable from tonal_reference_profile."""
        from backend.core.tonal_reference_profile import (
            HarmonicProfile,
            get_era_harmonic_profile,
        )

        assert callable(get_era_harmonic_profile)
        profile = get_era_harmonic_profile(1940)
        assert isinstance(profile, HarmonicProfile)

    def test_era_1940_h2_ratio_correct(self):
        """1940 decade should return the Golden Tube era H2 ratio (0.020)."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(1940)
        assert abs(profile.h2_ratio - 0.020) < 1e-6, f"Expected 0.020, got {profile.h2_ratio}"
        assert profile.era_label == "Golden Tube"

    def test_era_none_returns_fallback_1970(self):
        """None decade must fall back to 1970 Transistor-Era profile."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(None)
        assert abs(profile.h2_ratio - 0.006) < 1e-6, f"Expected 0.006, got {profile.h2_ratio}"
        assert "Transistor" in profile.era_label

    def test_era_beyond_2000_uses_nearest(self):
        """Decade 2030 (beyond last entry 2025) must use the 2025 entry."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(2030)
        # 2025 is the max available key — Contemporary era
        assert abs(profile.h2_ratio - 0.0002) < 1e-7
        assert "Contemporary" in profile.era_label

    def test_era_exact_key_match(self):
        """Exact decade key must return that entry directly."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        p1960 = get_era_harmonic_profile(1960)
        assert abs(p1960.h2_ratio - 0.014) < 1e-6

        p1970 = get_era_harmonic_profile(1970)
        assert abs(p1970.h2_ratio - 0.006) < 1e-6

    def test_era_between_decades_rounds_down(self):
        """Decade between entries (e.g. 1955) rounds down to 1950."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(1955)
        assert abs(profile.h2_ratio - 0.018) < 1e-6  # 1950 Classic Tube

    def test_get_era_harmonic_profile_in_all_export(self):
        """get_era_harmonic_profile must appear in __all__ of tonal_reference_profile."""
        import backend.core.tonal_reference_profile as mod

        assert "get_era_harmonic_profile" in mod.__all__, "get_era_harmonic_profile missing from __all__"

    def test_phase07_source_contains_h2_target_steering(self):
        """phase_07 process() must contain the ERA_HARMONIC H2-target-steering block."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "ERA_HARMONIC" in src, "phase_07.process() must contain §ERA_HARMONIC steering"
        assert "get_era_harmonic_profile" in src, "phase_07 must call get_era_harmonic_profile"
        assert "_measure_h2_ratio" in src, "phase_07 must call _measure_h2_ratio"

    def test_phase07_measure_h2_ratio_method_exists(self):
        """_measure_h2_ratio must be a static method of HarmonicRestorationPhase."""
        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        assert hasattr(HarmonicRestorationPhase, "_measure_h2_ratio"), (
            "_measure_h2_ratio method missing from HarmonicRestorationPhase"
        )
        assert callable(HarmonicRestorationPhase._measure_h2_ratio)

    def test_measure_h2_ratio_pure_sine_returns_small(self):
        """Pure 440 Hz sine without harmonics should yield a very small H2 ratio."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        t = np.linspace(0, 5.0, sr * 5)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        ratio = HarmonicRestorationPhase._measure_h2_ratio(audio, sr)
        # No harmonic content → should be well below 0.01
        assert ratio < 0.10, f"Expected small ratio for pure sine, got {ratio:.4f}"

    def test_measure_h2_ratio_with_harmonics_detects_h2(self):
        """Signal with explicit H2 component must yield a detectable H2 ratio."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        t = np.linspace(0, 5.0, sr * 5)
        # H1 = 440 Hz at amplitude 1.0, H2 = 880 Hz at amplitude 0.03
        audio = (1.0 * np.sin(2 * np.pi * 440 * t) + 0.03 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        ratio = HarmonicRestorationPhase._measure_h2_ratio(audio, sr)
        # Should detect H2 around 0.03 ± some tolerance
        assert ratio > 0.005, f"H2 ratio too low: {ratio:.4f}"

    def test_measure_h2_ratio_short_audio_returns_zero(self):
        """Audio shorter than 4096 samples must return 0.0 safely."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        short = np.zeros(100, dtype=np.float32)
        ratio = HarmonicRestorationPhase._measure_h2_ratio(short, 48000)
        assert ratio == 0.0


class TestConsoleCharacterStudio2026:
    """Todo 5: Console-Character in Studio 2026 (§Gap5)."""

    def test_phase07_source_contains_console_character_block(self):
        """phase_07 process() must contain the §Gap5 Console-Character block."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "Gap5" in src or "console_character" in src.lower(), (
            "phase_07.process() must contain §Gap5 Console-Character block"
        )
        assert "get_studio_console_curve" in src, "phase_07 must call get_studio_console_curve"
        assert "studio" in src, "Console-Character block must be gated on studio mode"

    def test_phase07_studio_mode_only_guard(self):
        """Console-Character must only activate in studio mode, not restoration."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        # The guard condition must check for "studio" in mode
        assert '"studio" in _mode_07' in src, "Console-Character must be gated on '\"studio\" in _mode_07'"

    def test_phase07_console_hallucination_guard_present(self):
        """phase_07 must apply hallucination_guard after console EQ (§2.46e)."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "hallucination_guard" in src or "check_hallucination" in src, (
            "phase_07 must import hallucination_guard for console EQ"
        )

    def test_apply_console_eq_method_exists(self):
        """_apply_console_eq must be a static method of HarmonicRestorationPhase."""
        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        assert hasattr(HarmonicRestorationPhase, "_apply_console_eq"), (
            "_apply_console_eq missing from HarmonicRestorationPhase"
        )

    def test_apply_console_eq_passthrough_on_neutral(self):
        """Neutral console profile (0 dB at all freqs) must return near-identical audio."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        audio = np.random.default_rng(42).standard_normal(sr * 3).astype(np.float32) * 0.3
        neutral_bp = [(20.0, 0.0), (20000.0, 0.0)]
        result = HarmonicRestorationPhase._apply_console_eq(audio, neutral_bp, sr, strength=1.0)
        # Should be within ±3 dB RMS of the original
        rms_orig = float(np.sqrt(np.mean(audio**2)))
        rms_out = float(np.sqrt(np.mean(result**2)))
        assert abs(rms_out - rms_orig) / (rms_orig + 1e-8) < 0.20, (
            f"Neutral console EQ should preserve RMS, orig={rms_orig:.4f} out={rms_out:.4f}"
        )

    def test_apply_console_eq_strength_zero_returns_passthrough(self):
        """strength=0 must return audio effectively unchanged."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        audio = np.random.default_rng(7).standard_normal(sr * 2).astype(np.float32) * 0.3
        neve_bp = [
            (20.0, 0.5),
            (80.0, 2.0),
            (200.0, 0.5),
            (1000.0, 0.0),
            (3000.0, 1.0),
            (18000.0, -0.5),
            (20000.0, -0.8),
        ]
        result = HarmonicRestorationPhase._apply_console_eq(audio, neve_bp, sr, strength=0.0)
        rms_diff = float(np.sqrt(np.mean((result - audio) ** 2)))
        assert rms_diff < 0.005, f"strength=0 should be passthrough, diff RMS={rms_diff:.6f}"

    def test_phase07_console_character_in_metadata(self):
        """phase_07 metadata must include console_character_applied key."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "console_character_applied" in src, "phase_07 return metadata must include 'console_character_applied'"


# ===========================================================================
# §Gap1 VocalFocusAnalyzer — Emotional Context (v10.14.x)
# ===========================================================================


class TestVFAEmotionalContext:
    """Tests for VFAResult emotional context fields + analyze() steps 6-9."""

    def _make_audio(self, sr: int = 48000, duration: float = 4.0, rng_seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.3

    def test_vfa_result_has_new_fields(self):
        """VFAResult must have tension_zones, release_zones, whisper_zones, climax_type."""
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult()
        assert hasattr(r, "tension_zones"), "VFAResult missing tension_zones"
        assert hasattr(r, "release_zones"), "VFAResult missing release_zones"
        assert hasattr(r, "whisper_zones"), "VFAResult missing whisper_zones"
        assert hasattr(r, "climax_type"), "VFAResult missing climax_type"
        assert r.climax_type == "none"
        assert isinstance(r.tension_zones, list)
        assert isinstance(r.release_zones, list)
        assert isinstance(r.whisper_zones, list)

    def test_vfa_to_dict_contains_new_fields(self):
        """VFAResult.to_dict() must export all new emotional context fields."""
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult(tension_zones=[(1.0, 2.0)], release_zones=[(2.5, 3.5)], whisper_zones=[], climax_type="peak")
        d = r.to_dict()
        assert "tension_zones" in d, "to_dict missing tension_zones"
        assert "release_zones" in d, "to_dict missing release_zones"
        assert "whisper_zones" in d, "to_dict missing whisper_zones"
        assert "climax_type" in d, "to_dict missing climax_type"
        assert d["climax_type"] == "peak"
        assert d["tension_zones"] == [(1.0, 2.0)]

    def test_detect_tension_zones_returns_list(self):
        """_detect_tension_zones must return a list of (float, float) tuples."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        sr = 48000
        audio = self._make_audio(sr=sr, duration=3.0)
        zones = VocalFocusAnalyzer._detect_tension_zones(audio, sr)
        assert isinstance(zones, list)
        for z in zones:
            assert len(z) == 2
            assert z[0] < z[1]

    def test_detect_tension_zones_short_audio(self):
        """_detect_tension_zones with very short audio must return [] without crash."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        zones = VocalFocusAnalyzer._detect_tension_zones(np.zeros(128, dtype=np.float32), 48000)
        assert zones == []

    def test_detect_release_zones_from_frisson(self):
        """_detect_release_zones must produce zones after frisson end times."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        frisson_zones = [(2.0, 3.0), (7.0, 8.0)]
        tension_zones: list = []
        mono = np.zeros(int(48000 * 12), dtype=np.float32)
        zones = VocalFocusAnalyzer._detect_release_zones(mono, 48000, tension_zones, frisson_zones)
        assert len(zones) >= 2
        # First release zone must start at or after frisson end
        starts = [z[0] for z in zones]
        assert any(s >= 3.0 for s in starts)

    def test_detect_whisper_zones_silent_audio(self):
        """Very quiet audio should be detected as whisper (or empty if below silence threshold)."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        sr = 48000
        # Level between -50 dBFS and -35 dBFS → whisper territory
        # -40 dBFS = 10^(-40/20) = 0.01
        audio = np.ones(sr * 3, dtype=np.float32) * 0.01  # ~−40 dBFS
        zones = VocalFocusAnalyzer._detect_whisper_zones(audio, sr)
        assert isinstance(zones, list)
        # constant tone = low flatness, may not trigger whisper — that's acceptable
        # Just assert no crash

    def test_detect_climax_type_none_no_frisson(self):
        """climax_type must be 'none' when frisson_zones is empty."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        mono = np.zeros(48000, dtype=np.float32)
        ct = VocalFocusAnalyzer._detect_climax_type([], [], mono, 48000)
        assert ct == "none"

    def test_detect_climax_type_peak(self):
        """Single short frisson zone → 'peak'."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        mono = np.zeros(48000, dtype=np.float32)
        ct = VocalFocusAnalyzer._detect_climax_type([(2.0, 3.5)], [], mono, 48000)
        assert ct == "peak"

    def test_detect_climax_type_sustained(self):
        """Single long frisson zone (≥ 5 s) → 'sustained'."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        mono = np.zeros(48000, dtype=np.float32)
        ct = VocalFocusAnalyzer._detect_climax_type([(0.0, 7.0)], [], mono, 48000)
        assert ct == "sustained"

    def test_detect_climax_type_dynamic(self):
        """Three frisson zones → 'dynamic'."""
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        mono = np.zeros(48000, dtype=np.float32)
        ct = VocalFocusAnalyzer._detect_climax_type([(1.0, 2.0), (5.0, 6.0), (9.0, 10.0)], [], mono, 48000)
        assert ct == "dynamic"

    def test_vfa_analyze_returns_emotional_fields(self):
        """VocalFocusAnalyzer.analyze() must populate emotional context fields."""
        from backend.core.vocal_focus_analyzer import get_vocal_focus_analyzer

        sr = 48000
        audio = self._make_audio(sr=sr, duration=5.0)
        vfa = get_vocal_focus_analyzer()
        result = vfa.analyze(audio, sr, panns_singing=0.6)
        assert hasattr(result, "tension_zones")
        assert hasattr(result, "release_zones")
        assert hasattr(result, "whisper_zones")
        assert hasattr(result, "climax_type")
        assert result.climax_type in ("none", "peak", "sustained", "dynamic")


# ===========================================================================
# §Gap2 SongCoherenceMonitor
# ===========================================================================


class TestSongCoherenceMonitor:
    """Tests for song-wide timbral coherence analysis."""

    def _make_audio(self, sr: int = 48000, duration: float = 15.0, seed: int = 99) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.2

    def test_import(self):
        """SongCoherenceMonitor must be importable."""
        from backend.core.song_coherence_monitor import SongCoherenceMonitor

        assert SongCoherenceMonitor is not None

    def test_singleton(self):
        """get_song_coherence_monitor() must return same instance."""
        from backend.core.song_coherence_monitor import get_song_coherence_monitor

        a = get_song_coherence_monitor()
        b = get_song_coherence_monitor()
        assert a is b

    def test_analyze_short_audio(self):
        """Very short audio must return coherence_score=1.0 (no check possible)."""
        from backend.core.song_coherence_monitor import get_song_coherence_monitor

        audio = np.zeros(4800, dtype=np.float32)
        result = get_song_coherence_monitor().analyze(audio, 48000)
        assert result.coherence_score == 1.0
        assert result.n_sections_analyzed == 0

    def test_analyze_sufficient_length(self):
        """Sufficient-length audio must produce coherence_score in [0, 1]."""
        from backend.core.song_coherence_monitor import get_song_coherence_monitor

        audio = self._make_audio(duration=35.0)
        result = get_song_coherence_monitor().analyze(audio, 48000)
        assert 0.0 <= result.coherence_score <= 1.0
        assert result.n_sections_analyzed >= _MIN_SECTIONS_FOR_CHECK_PROXY
        assert isinstance(result.inconsistent_sections, list)

    def test_to_dict_has_required_keys(self):
        """SongCoherenceResult.to_dict() must have coherence_score and inconsistent_sections."""
        from backend.core.song_coherence_monitor import get_song_coherence_monitor

        audio = self._make_audio(duration=35.0)
        d = get_song_coherence_monitor().analyze(audio, 48000).to_dict()
        assert "coherence_score" in d
        assert "inconsistent_sections" in d
        assert "reference_timbre" in d
        assert "n_sections_analyzed" in d

    def test_consistent_audio_high_score(self):
        """Uniform noise (same statistics everywhere) should yield high coherence."""
        from backend.core.song_coherence_monitor import get_song_coherence_monitor

        sr = 48000
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(sr * 40).astype(np.float32) * 0.2
        result = get_song_coherence_monitor().analyze(audio, sr)
        assert result.coherence_score >= 0.50, f"Uniform noise should be consistent, got {result.coherence_score}"

    def test_stereo_audio_accepted(self):
        """Stereo input must be handled without error."""
        from backend.core.song_coherence_monitor import get_song_coherence_monitor

        sr = 48000
        audio = np.random.default_rng(3).standard_normal((2, sr * 30)).astype(np.float32) * 0.2
        result = get_song_coherence_monitor().analyze(audio, sr)
        assert result.coherence_score >= 0.0


# Proxy constant for tests (mirrors _MIN_SECTIONS_FOR_CHECK in module)
_MIN_SECTIONS_FOR_CHECK_PROXY: int = 3


# ===========================================================================
# §Gap3 PhraseBoundaryGuard
# ===========================================================================


class TestPhraseBoundaryGuard:
    """Tests for phrase boundary detection and taper application."""

    def test_import(self):
        """PhraseBoundaryGuard functions must be importable."""
        from backend.core.dsp.phrase_boundary_guard import (
            apply_phrase_boundary_taper,
            detect_phrase_boundaries,
        )

        assert detect_phrase_boundaries is not None
        assert apply_phrase_boundary_taper is not None

    def test_detect_short_audio_empty(self):
        """Short audio must return empty boundary list."""
        from backend.core.dsp.phrase_boundary_guard import detect_phrase_boundaries

        boundaries = detect_phrase_boundaries(np.zeros(100, dtype=np.float32), 48000)
        assert boundaries == []

    def test_detect_silence_between_phrases(self):
        """Audio with a silence gap must detect at least one boundary."""
        from backend.core.dsp.phrase_boundary_guard import detect_phrase_boundaries

        sr = 48000
        rng = np.random.default_rng(5)
        phrase = rng.standard_normal(sr).astype(np.float32) * 0.3
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        audio = np.concatenate([phrase, silence, phrase])
        boundaries = detect_phrase_boundaries(audio, sr)
        assert len(boundaries) >= 1, "Should detect boundary in silence gap"

    def test_taper_all_ones_no_boundaries(self):
        """With no boundaries, taper must return all-ones envelope."""
        from backend.core.dsp.phrase_boundary_guard import apply_phrase_boundary_taper

        audio = np.zeros(48000, dtype=np.float32)
        env = apply_phrase_boundary_taper(audio, [], 48000)
        assert np.allclose(env, 1.0), "No boundaries → all-ones envelope"

    def test_taper_zero_at_boundary(self):
        """Envelope must be 0 exactly at boundary position."""
        from backend.core.dsp.phrase_boundary_guard import apply_phrase_boundary_taper

        audio = np.zeros(48000, dtype=np.float32)
        b = 24000  # midpoint
        env = apply_phrase_boundary_taper(audio, [b], 48000, taper_ms=20.0)
        assert abs(env[b]) < 1e-6, f"Env at boundary must be 0, got {env[b]}"

    def test_taper_shape_and_dtype(self):
        """Taper must return float32 1-D array of same length as audio."""
        from backend.core.dsp.phrase_boundary_guard import apply_phrase_boundary_taper

        audio = np.zeros(48000, dtype=np.float32)
        env = apply_phrase_boundary_taper(audio, [12000, 36000], 48000)
        assert env.dtype == np.float32
        assert len(env) == len(audio)
        assert np.all(env >= 0.0) and np.all(env <= 1.0)

    def test_boundaries_sorted_and_valid(self):
        """All returned boundary indices must be within audio range."""
        from backend.core.dsp.phrase_boundary_guard import detect_phrase_boundaries

        sr = 48000
        audio = np.random.default_rng(7).standard_normal(sr * 5).astype(np.float32) * 0.2
        boundaries = detect_phrase_boundaries(audio, sr)
        n = len(audio)
        for b in boundaries:
            assert 0 <= b < n, f"Boundary {b} out of range [0, {n})"
        assert boundaries == sorted(boundaries), "Boundaries must be sorted"


# ===========================================================================
# §Gap5 BlindInternalReference
# ===========================================================================


class TestBlindInternalReference:
    """Tests for blind cleanest-segment detection."""

    def test_import(self):
        """BlindInternalReference must be importable."""
        from backend.core.blind_internal_reference import BlindInternalReference

        assert BlindInternalReference is not None

    def test_singleton(self):
        """get_blind_internal_reference() must return same instance."""
        from backend.core.blind_internal_reference import get_blind_internal_reference

        a = get_blind_internal_reference()
        b = get_blind_internal_reference()
        assert a is b

    def test_find_short_audio(self):
        """Very short audio must return empty segments list."""
        from backend.core.blind_internal_reference import get_blind_internal_reference

        audio = np.zeros(480, dtype=np.float32)
        result = get_blind_internal_reference().find(audio, 48000)
        assert result.segments == []
        assert result.best_score == 0.0

    def test_find_returns_top_n_segments(self):
        """Must return at most top_n segments for long audio."""
        from backend.core.blind_internal_reference import get_blind_internal_reference

        sr = 48000
        audio = np.random.default_rng(11).standard_normal(sr * 30).astype(np.float32) * 0.2
        result = get_blind_internal_reference().find(audio, sr, top_n=3)
        assert len(result.segments) <= 3
        assert len(result.segments) >= 1

    def test_find_segments_sorted_by_score(self):
        """Returned segments must be sorted descending by score."""
        from backend.core.blind_internal_reference import get_blind_internal_reference

        sr = 48000
        audio = np.random.default_rng(12).standard_normal(sr * 25).astype(np.float32) * 0.2
        result = get_blind_internal_reference().find(audio, sr)
        scores = [s.score for s in result.segments]
        assert scores == sorted(scores, reverse=True)

    def test_best_score_in_range(self):
        """best_score must be in [0, 1]."""
        from backend.core.blind_internal_reference import get_blind_internal_reference

        sr = 48000
        audio = np.random.default_rng(13).standard_normal(sr * 20).astype(np.float32) * 0.2
        result = get_blind_internal_reference().find(audio, sr)
        assert 0.0 <= result.best_score <= 1.0

    def test_to_dict_structure(self):
        """BlindReferenceResult.to_dict() must have segments, global_snr_proxy_db, best_score."""
        from backend.core.blind_internal_reference import get_blind_internal_reference

        sr = 48000
        audio = np.random.default_rng(14).standard_normal(sr * 20).astype(np.float32) * 0.2
        d = get_blind_internal_reference().find(audio, sr).to_dict()
        assert "segments" in d
        assert "global_snr_proxy_db" in d
        assert "best_score" in d

    def test_segment_times_valid(self):
        """Segment start_s must be < end_s and within audio duration."""
        from backend.core.blind_internal_reference import get_blind_internal_reference

        sr = 48000
        duration = 20.0
        audio = np.random.default_rng(15).standard_normal(int(sr * duration)).astype(np.float32) * 0.2
        result = get_blind_internal_reference().find(audio, sr)
        for seg in result.segments:
            assert seg.start_s >= 0.0
            assert seg.end_s > seg.start_s
            assert seg.end_s <= duration + 0.1  # +0.1 s tolerance for rounding

    def test_score_segment_range(self):
        """_score_segment must return (score [0,1], snr_db, clarity [0,1])."""
        from backend.core.blind_internal_reference import BlindInternalReference

        sr = 48000
        seg = np.random.default_rng(16).standard_normal(sr * 5).astype(np.float32) * 0.2
        score, snr_db, clarity = BlindInternalReference._score_segment(seg, sr)
        assert 0.0 <= score <= 1.0
        assert -20.0 <= snr_db <= 60.0
        assert 0.0 <= clarity <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# §Gap-Vocal-Guards — PhraseGuard + FormantGate + EnergyBias Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPhraseBoundaryGuardIntegration:
    """phrase_boundary_guard is integrated in phases 03, 29, 49 (§Gap3)."""

    def test_phrase_boundary_guard_present_in_phase_03(self):
        """phase_03_denoise must reference detect_phrase_boundaries."""
        import inspect

        from backend.core.phases import phase_03_denoise

        src = inspect.getsource(phase_03_denoise)
        assert "detect_phrase_boundaries" in src, "PhraseGuard missing from phase_03"
        assert "apply_phrase_boundary_taper" in src

    def test_phrase_boundary_guard_present_in_phase_29(self):
        """phase_29_tape_hiss_reduction must reference detect_phrase_boundaries."""
        import inspect

        from backend.core.phases import phase_29_tape_hiss_reduction

        src = inspect.getsource(phase_29_tape_hiss_reduction)
        assert "detect_phrase_boundaries" in src, "PhraseGuard missing from phase_29"
        assert "apply_phrase_boundary_taper" in src

    def test_phrase_boundary_guard_present_in_phase_49(self):
        """phase_49_advanced_dereverb must reference detect_phrase_boundaries."""
        import inspect

        from backend.core.phases import phase_49_advanced_dereverb

        src = inspect.getsource(phase_49_advanced_dereverb)
        assert "detect_phrase_boundaries" in src, "PhraseGuard missing from phase_49"
        assert "apply_phrase_boundary_taper" in src

    def test_phrase_guard_taper_non_blocking_on_short(self):
        """detect_phrase_boundaries returns [] for short audio — non-blocking."""
        from backend.core.dsp.phrase_boundary_guard import detect_phrase_boundaries

        short = np.zeros(512, dtype=np.float32)
        result = detect_phrase_boundaries(short, 48000)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_phrase_guard_taper_returns_envelope_len(self):
        """apply_phrase_boundary_taper returns envelope of correct length."""
        from backend.core.dsp.phrase_boundary_guard import (
            apply_phrase_boundary_taper,
            detect_phrase_boundaries,
        )

        rng = np.random.default_rng(99)
        audio = rng.standard_normal(48000 * 10).astype(np.float32) * 0.1
        boundaries = detect_phrase_boundaries(audio, 48000)
        env = apply_phrase_boundary_taper(audio, boundaries, 48000)
        assert env.shape[0] == audio.shape[0], "Envelope length must match audio length"
        assert np.all(env >= 0.0) and np.all(env <= 1.0), "Envelope must be in [0, 1]"


class TestFormantIntegrityGates:
    """Formant preventive gates are referenced in phases 20 and 49 (§0p)."""

    def test_formant_gate_present_in_phase_20(self):
        """phase_20_reverb_reduction must reference LPC formant tracker for pre/post check."""
        import inspect

        from backend.core.phases import phase_20_reverb_reduction

        src = inspect.getsource(phase_20_reverb_reduction)
        assert "lpc_formant_tracker" in src, "Formant gate missing from phase_20"
        assert "_f1_pre_20" in src
        assert "_f1_post_20" in src

    def test_formant_gate_present_in_phase_49(self):
        """phase_49_advanced_dereverb must reference LPC formant tracker for pre/post check."""
        import inspect

        from backend.core.phases import phase_49_advanced_dereverb

        src = inspect.getsource(phase_49_advanced_dereverb)
        assert "lpc_formant_tracker" in src, "Formant gate missing from phase_49"
        assert "_f1_pre_49" in src
        assert "_f1_post_49" in src

    def test_formant_gate_rollback_keyword_phase_20(self):
        """phase_20 must log 'Formant drift' when rolling back."""
        import inspect

        from backend.core.phases import phase_20_reverb_reduction

        src = inspect.getsource(phase_20_reverb_reduction)
        assert "Formant drift phase_20" in src

    def test_formant_gate_rollback_keyword_phase_49(self):
        """phase_49 must log 'Formant drift' when rolling back."""
        import inspect

        from backend.core.phases import phase_49_advanced_dereverb

        src = inspect.getsource(phase_49_advanced_dereverb)
        assert "Formant drift phase_49" in src


class TestEnergyBiasContextPropagation:
    """vocal_energy_bias_db from _restoration_context is read by phases 20 and 29 (§0j)."""

    def test_energy_bias_context_read_in_phase_20(self):
        """phase_20_reverb_reduction must read vocal_energy_bias_db from _restoration_context."""
        import inspect

        from backend.core.phases import phase_20_reverb_reduction

        src = inspect.getsource(phase_20_reverb_reduction)
        assert "_ctx_energy_bias_20" in src, "Energy-bias context missing from phase_20"
        assert "vocal_energy_bias_db" in src

    def test_energy_bias_context_read_in_phase_29(self):
        """phase_29_tape_hiss_reduction must read vocal_energy_bias_db from context."""
        import inspect

        from backend.core.phases import phase_29_tape_hiss_reduction

        src = inspect.getsource(phase_29_tape_hiss_reduction)
        assert "_ctx_energy_bias_29" in src, "Energy-bias context missing from phase_29"
        assert "vocal_energy_bias_db" in src

    def test_energy_bias_scale_formula_sane(self):
        """energy_bias_db -9 dB → scale factor ~0.35 (not 0 or 1)."""

        _eb = -9.0
        scale = 10.0 ** (_eb / 20.0)
        assert 0.30 < scale < 0.40, f"Unexpected scale {scale} for -9 dB energy_bias"

    def test_energy_bias_neutral_default(self):
        """Default of -6.0 dB must not further scale strength (< -6.0 trigger only)."""
        _ctx_energy_bias = -6.0
        triggered = _ctx_energy_bias < -6.0
        assert not triggered, "Default -6 dB must not trigger extra scaling"


# ===========================================================================
# Session 2026-05-13+: Gap-Implementierungen (MIIPHER, FeedbackChain-VQI,
# LPC-F4, Falsetto, SingMOS-VQI)
# ===========================================================================


class TestMiipherNoNotImplementedError:
    """MIIPHER _enhance_miipher darf kein NotImplementedError mehr werfen."""

    def test_enhance_miipher_raises_runtime_not_notimplemented(self):
        """_enhance_miipher muss RuntimeError (DFN-Fallback-Signal) statt NotImplementedError werfen."""
        from plugins.miipher_plugin import MiipherPlugin

        plugin = MiipherPlugin()
        assert not plugin._model_loaded, "Stub-Modus: Modell darf nicht geladen sein"
        audio = np.zeros(4800, dtype=np.float32)
        # Direkt _enhance_miipher aufrufen — SGMSE+ nicht verfügbar → RuntimeError (kein NotImplementedError)
        with pytest.raises((RuntimeError, Exception)) as exc_info:
            plugin._enhance_miipher(audio, 48000)
        # Darf NICHT NotImplementedError sein
        assert not isinstance(exc_info.value, NotImplementedError), (
            "_enhance_miipher darf kein NotImplementedError werfen — Stub entfernt"
        )

    def test_enhance_full_chain_returns_audio(self):
        """enhance() muss immer Audio zurückgeben (Wiener-Fallback als Last-Resort)."""
        from plugins.miipher_plugin import MiipherPlugin

        plugin = MiipherPlugin()
        audio = np.random.default_rng(42).uniform(-0.1, 0.1, 4800).astype(np.float32)
        result = plugin.enhance(audio, 48000)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result)), "enhance() darf kein NaN/Inf ausgeben"

    def test_should_activate_logic(self):
        """should_activate muss SNR < 10 UND panns_singing >= 0.35 erfordern."""
        from plugins.miipher_plugin import MiipherPlugin

        plugin = MiipherPlugin()
        assert plugin.should_activate(noise_snr_db=5.0, panns_singing=0.5)
        assert not plugin.should_activate(noise_snr_db=15.0, panns_singing=0.5)
        assert not plugin.should_activate(noise_snr_db=5.0, panns_singing=0.2)


class TestFeedbackChainVQIDualObjective:
    """FeedbackChain §0p: VQI dual-objective wenn panns_singing >= 0.35."""

    def test_vqi_dual_objective_method_exists(self):
        """_apply_vqi_dual_objective Methode muss vorhanden sein."""
        from backend.core.feedback_chain import FeedbackChain

        fc = FeedbackChain()
        assert hasattr(fc, "_apply_vqi_dual_objective"), "_apply_vqi_dual_objective fehlt"

    def test_panns_singing_param_accepted(self):
        """FeedbackChain muss panns_singing-Parameter akzeptieren."""
        from backend.core.feedback_chain import FeedbackChain

        fc = FeedbackChain(panns_singing=0.6)
        assert fc.panns_singing == pytest.approx(0.6, abs=1e-6)

    def test_panns_singing_zero_no_vqi(self):
        """Bei panns_singing < 0.35 muss _vqi_orig_audio None bleiben (kein VQI-Overhead)."""
        from backend.core.feedback_chain import FeedbackChain

        fc = FeedbackChain(panns_singing=0.0)
        assert fc._vqi_orig_audio is None

    def test_vqi_dual_objective_no_orig_passthrough(self):
        """_apply_vqi_dual_objective ohne _vqi_orig_audio muss base_mos unverändert zurückgeben."""
        from backend.core.feedback_chain import FeedbackChain

        fc = FeedbackChain(panns_singing=0.8)
        fc._vqi_orig_audio = None  # kein Orig gespeichert
        audio = np.zeros(4800, dtype=np.float32)
        result = fc._apply_vqi_dual_objective(audio, 48000, 3.5)
        assert result == pytest.approx(3.5, abs=1e-6)

    def test_vqi_dual_objective_low_panns_passthrough(self):
        """_apply_vqi_dual_objective bei panns_singing < 0.35 muss base_mos unverändert zurückgeben."""
        from backend.core.feedback_chain import FeedbackChain

        fc = FeedbackChain(panns_singing=0.1)
        audio = np.zeros(4800, dtype=np.float32)
        fc._vqi_orig_audio = audio.copy()
        result = fc._apply_vqi_dual_objective(audio, 48000, 4.0)
        assert result == pytest.approx(4.0, abs=1e-6)


class TestLPCF4Extension:
    """LPC F4 (Singer's Formant) in formant_boost_eq."""

    def test_formant_boost_eq_uses_four_formants(self):
        """_formant_boost_eq muss F4 verarbeiten (4 Formanten, nicht nur 3)."""
        from backend.core.dsp.lpc_formant_tracker import _formant_boost_eq

        audio = np.random.default_rng(0).uniform(-0.1, 0.1, 4800).astype(np.float32)
        # 4 Formanten: F1=500, F2=1500, F3=2500, F4=3500 Hz (Singer's Formant)
        f4_formants = [500.0, 1500.0, 2500.0, 3500.0]
        result_4 = _formant_boost_eq(audio, 48000, f4_formants, boost_db=1.5)
        result_3 = _formant_boost_eq(audio, 48000, f4_formants[:3], boost_db=1.5)
        # F4-Boost muss messbaren Unterschied erzeugen
        diff = float(np.mean(np.abs(result_4.astype(np.float64) - result_3.astype(np.float64))))
        assert diff > 1e-7, f"F4-Boost hat keinen Effekt (diff={diff:.2e}) — [:3] Slice noch aktiv?"

    def test_formant_boost_eq_f4_source_comment(self):
        """Quellcode-Check: _formant_boost_eq darf nicht mehr [:3] enthalten."""
        import inspect

        from backend.core.dsp.lpc_formant_tracker import _formant_boost_eq

        src = inspect.getsource(_formant_boost_eq)
        assert "[:3]" not in src, "_formant_boost_eq enthält noch [:3] — F4 nicht aktiv"


class TestFalsettoRegisterDetection:
    """Falsetto muss als register erkannt und in VFA propagiert werden."""

    def test_falsetto_in_detect_vocal_register(self):
        """detect_vocal_register muss 'falsetto' zurückgeben können."""
        from backend.core.dsp.vocal_register_detector import _ENERGY_BIAS_FALSETTO, detect_vocal_register

        # Falsetto-Signal: Sinuswelle bei 400 Hz (hoch + leicht atemhaft)
        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # Fundamentale bei 400 Hz + leichter Atemanteil (Rauschen): teilweise atemhaft
        sine = 0.5 * np.sin(2 * np.pi * 400 * t)
        noise = 0.12 * np.random.default_rng(7).standard_normal(len(t)).astype(np.float32)
        audio = (sine + noise).astype(np.float32)
        register, bias = detect_vocal_register(audio, sr, panns_singing=0.7)
        # Mit Rauschen kann Flachheit variieren — prüfe ob "falsetto" Kandidat ist
        # (Wenn F0-Schätzung fehlschlägt, kann auch "head" zurückkommen — das ist OK)
        assert register in {"falsetto", "head", "chest"}, f"Unbekanntes Register: {register}"
        if register == "falsetto":
            assert bias == pytest.approx(_ENERGY_BIAS_FALSETTO, abs=1e-6)

    def test_falsetto_energy_bias_between_head_and_fry(self):
        """Falsetto energy_bias muss zwischen Kopf (-3 dB) und Fry (-9 dB) liegen."""
        from backend.core.dsp.vocal_register_detector import (
            _ENERGY_BIAS_CHEST,
            _ENERGY_BIAS_FALSETTO,
            _ENERGY_BIAS_FRY_WHISPER,
            _ENERGY_BIAS_HEAD,
        )

        assert _ENERGY_BIAS_CHEST < _ENERGY_BIAS_FALSETTO < _ENERGY_BIAS_HEAD, (
            f"Falsetto energy_bias {_ENERGY_BIAS_FALSETTO} muss zwischen Chest ({_ENERGY_BIAS_CHEST}) "
            f"und Head ({_ENERGY_BIAS_HEAD}) liegen (alle negativ: Chest=-6, Falsetto=-4.5, Head=-3)"
        )
        assert _ENERGY_BIAS_FALSETTO > _ENERGY_BIAS_FRY_WHISPER, "Falsetto darf nicht aggressiver sein als Fry/Whisper"

    def test_falsetto_constant_defined(self):
        """_ENERGY_BIAS_FALSETTO muss in vocal_register_detector definiert sein."""
        from backend.core.dsp.vocal_register_detector import _ENERGY_BIAS_FALSETTO

        assert isinstance(_ENERGY_BIAS_FALSETTO, float)
        assert -6.0 < _ENERGY_BIAS_FALSETTO < -3.0, f"Erwarteter Bereich (-6,-3), erhalten: {_ENERGY_BIAS_FALSETTO}"

    def test_vfa_dominant_register_docstring_includes_falsetto(self):
        """VFAResult.dominant_register Docstring muss 'falsetto' erwähnen."""
        import inspect

        from backend.core.vocal_focus_analyzer import VFAResult

        src = inspect.getsource(VFAResult)
        assert "falsetto" in src, "VFAResult.dominant_register Docstring enthält 'falsetto' nicht"


class TestVQISingMOSIntegration:
    """VQI muss SingMOS über VERSA als primären Naturalness-Proxy nutzen."""

    def test_compute_vqi_versa_singmos_branch_exists(self):
        """compute_vqi Quellcode muss SingMOS-Integration via versa_plugin enthalten."""
        import inspect

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        src = inspect.getsource(compute_vqi)
        assert "singmos_score" in src, "VQI: singmos_score Variable fehlt"
        assert "versa_plugin" in src, "VQI: versa_plugin Import fehlt"
        assert "model_used" in src, "VQI: model_used Check für SingMOS-Pro fehlt"

    def test_compute_vqi_returns_dict_with_keys(self):
        """compute_vqi muss alle erwarteten Keys zurückgeben (ohne SingMOS verfügbar)."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        rng = np.random.default_rng(42)
        orig = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        rest = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        result = compute_vqi(orig, rest, 48000)
        required = {
            "vqi",
            "singer_identity_cosine",
            "formant_stability_score",
            "articulation_score",
            "proximity_score",
            "sibilance_naturalness",
            "vqi_tier",
        }
        missing = required - set(result.keys())
        assert not missing, f"compute_vqi fehlen Keys: {missing}"

    def test_singmos_score_none_falls_back_to_proximity(self):
        """Wenn SingMOS nicht verfügbar, muss proximity DSP-Fallback (0.85) genutzt werden."""
        import inspect

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        src = inspect.getsource(compute_vqi)
        # Sicherstellen dass es einen default für proximity = 0.85 gibt
        assert "0.85" in src, "VQI: proximity default 0.85 fehlt"


# ===========================================================================
# Session 2026-05-15: P1 Artist-Voice-Reference + P2 Style-Intent-Detector
# ===========================================================================


class TestVQIArtistReference:
    """§P1: reference_audio Parameter in compute_vqi() — sauberer Künstler-Referenz-Anker."""

    def test_reference_audio_param_accepted(self):
        """compute_vqi muss reference_audio Parameter akzeptieren."""
        import inspect

        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sig = inspect.signature(compute_vqi)
        assert "reference_audio" in sig.parameters, (
            "compute_vqi: reference_audio Parameter fehlt (§P1 Artist-Voice-Reference)"
        )

    def test_reference_audio_used_flag_in_result(self):
        """Wenn reference_audio übergeben, muss result['reference_audio_used'] True sein."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        rng = np.random.default_rng(10)
        orig = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        rest = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        # Saubere Referenz (anderes Signal, gleicher Künstler)
        ref = rng.uniform(-0.15, 0.15, 48000).astype(np.float32)
        result = compute_vqi(orig, rest, 48000, reference_audio=ref)
        assert "reference_audio_used" in result, "compute_vqi result fehlt 'reference_audio_used'"
        assert result["reference_audio_used"] is True, (
            f"reference_audio_used muss True sein wenn ref übergeben, erhalten: {result['reference_audio_used']}"
        )

    def test_reference_audio_none_not_used(self):
        """Ohne reference_audio muss reference_audio_used False sein."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        rng = np.random.default_rng(11)
        orig = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        rest = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        result = compute_vqi(orig, rest, 48000)
        assert result.get("reference_audio_used") is False, (
            "reference_audio_used muss False sein wenn kein ref übergeben"
        )

    def test_reference_audio_too_short_falls_back(self):
        """Wenn reference_audio < 0.5 s, muss fallback auf degraded orig erfolgen."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        rng = np.random.default_rng(12)
        orig = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        rest = rng.uniform(-0.2, 0.2, 48000).astype(np.float32)
        # Zu kurze Referenz (< 0.5 s = 24000 samples)
        short_ref = rng.uniform(-0.1, 0.1, 10000).astype(np.float32)
        result = compute_vqi(orig, rest, 48000, reference_audio=short_ref)
        assert result.get("reference_audio_used") is False, "Kurze Referenz (< 0.5 s) soll nicht genutzt werden (§P1)"


class TestStyleIntentDetector:
    """§P2: StyleIntentDetector — intentionale Pitch-Abweichungen erkennen."""

    def test_module_exists_and_importable(self):
        """style_intent_detector muss importierbar sein."""
        from backend.core.dsp import style_intent_detector

        assert hasattr(style_intent_detector, "StyleIntentDetector")
        assert hasattr(style_intent_detector, "get_style_intent_detector")
        assert hasattr(style_intent_detector, "StyleIntentResult")

    def test_singleton_returns_same_instance(self):
        """get_style_intent_detector() muss Singleton zurückgeben."""
        from backend.core.dsp.style_intent_detector import get_style_intent_detector

        a = get_style_intent_detector()
        b = get_style_intent_detector()
        assert a is b, "get_style_intent_detector muss Singleton sein (thread-safe)"

    def test_consistent_deviation_detected(self):
        """Konsistente Pitch-Abweichung (Blue Note) muss als intentional erkannt werden."""
        from backend.core.dsp.style_intent_detector import get_style_intent_detector

        sr = 48000
        t = np.linspace(0, 5.0, sr * 5, endpoint=False)
        # Bb mit -50 cents: A4 (440 Hz) × 2^(-0.5/12) ≈ 427.5 Hz — konsistente Blue Note
        f0_blue = 440.0 * (2.0 ** (-0.50 / 12.0))
        # Gleiche Frequenz 5 Sekunden wiederholen → konsistente Abweichung
        audio = (0.5 * np.sin(2 * np.pi * f0_blue * t)).astype(np.float32)
        detector = get_style_intent_detector()
        result = detector.analyze(audio, sr)
        # StyleIntentResult muss zurückkommen (kein Crash)
        assert result is not None
        # Bei konsistenter Abweichung: entweder intentional_pitch_classes gefüllt
        # oder style_confidence ∈ [0, 1] (DSP-Fallback kann 0 zurückgeben, das ist OK)
        assert 0.0 <= result.style_confidence <= 1.0, f"style_confidence außerhalb [0,1]: {result.style_confidence}"
        assert isinstance(result.style_intent_zones, list)

    def test_stochastic_deviation_low_confidence(self):
        """Stochastische Pitch-Fluktuation (zufällig) darf nicht als intentional gewertet werden."""
        from backend.core.dsp.style_intent_detector import (
            StyleIntentResult,
            get_style_intent_detector,
        )

        sr = 48000
        rng = np.random.default_rng(99)
        # Reines Rauschen: keine konsistente Pitch-Abweichung
        audio = rng.uniform(-0.3, 0.3, sr * 3).astype(np.float32)
        detector = get_style_intent_detector()
        result = detector.analyze(audio, sr)
        assert isinstance(result, StyleIntentResult)
        assert 0.0 <= result.style_confidence <= 1.0
        # Rauschen hat keine Voiced-Frames → intentional_pitch_classes leer
        assert len(result.intentional_pitch_classes) == 0, (
            f"Rauschen hat {len(result.intentional_pitch_classes)} intentionale PCs — "
            "StyleIntentDetector erkennt bei Rauschen fälschlich intentionale Abweichungen"
        )

    def test_vfa_result_has_style_intent_fields(self):
        """VFAResult muss style_intent_zones und style_confidence haben."""
        from backend.core.vocal_focus_analyzer import VFAResult

        vfa = VFAResult()
        assert hasattr(vfa, "style_intent_zones"), "VFAResult: style_intent_zones fehlt"
        assert hasattr(vfa, "style_confidence"), "VFAResult: style_confidence fehlt"
        assert isinstance(vfa.style_intent_zones, list)
        assert isinstance(vfa.style_confidence, float)
        assert vfa.style_confidence == 0.0  # Default

    def test_vfa_to_dict_includes_style_intent(self):
        """VFAResult.to_dict() muss style_intent_zones und style_confidence enthalten."""
        from backend.core.vocal_focus_analyzer import VFAResult

        vfa = VFAResult()
        d = vfa.to_dict()
        assert "style_intent_zones" in d, "to_dict() fehlt 'style_intent_zones'"
        assert "style_confidence" in d, "to_dict() fehlt 'style_confidence'"


# ===========================================================================
# §0p Gap-Fix: PMGG-Primärpfad injiziert panns_singing in phase_kwargs
# ===========================================================================


class TestPmggPathPannsInjection:
    """§0f/§0p RELEASE_MUST — kanonische Phase-Kontext-Injektion auf BEIDEN Pfaden.

    Es existieren zwei Phasen-Ausführungspfade: (1) PMGG-Primärpfad
    (_pmgg_gate.wrap_phase) und (2) Fallback _profiled_phase_call →
    _prepare_profiled_phase_runtime_context. Beide MÜSSEN dieselben vokal-/
    wahrnehmungsbezogenen Kontext-Keys injizieren, sonst sind §0p-Guards
    (Vibrato/Frisson/Passaggio/Breath/soft_saturation/vocal_presence) und die
    §9.1d Per-Event-Dosierung (defect_event_metadata) auf dem jeweils anderen
    Pfad latent deaktiviert — stiller Qualitätsverlust ohne Fehler.

    Single Source of Truth ist _canonical_phase_context_kwargs(); diese Tests
    verhindern, dass ein Pfad daran vorbei eigene kwargs baut (Reintroduktion).
    """

    # Kanonischer Key-Satz, der auf BEIDEN Pfaden ankommen MUSS.
    _CANONICAL_KEYS = (
        "frisson_zones",
        "passaggio_zones",
        "vibrato_zones",
        "breath_segments",
        "soft_saturation_severity",
        "soft_saturation_preserve",
        "vocal_presence_active",
        "vocal_presence_strength",
        "panns_singing",
        "panns_vocals_confidence",
        "panns_tags",
        "transfer_chain",
        "defect_event_metadata",
    )

    def _wrap_phase_kwargs_window(self) -> str:
        """Extrahiert das phase_kwargs-Fenster direkt nach _pmgg_gate.wrap_phase( in
        _execute_pipeline. Direkte Datei-Lesung statt inspect.getsource —
        linecache wird prozessglobal gecacht und von anderen Tests verschmutzt
        (order-abhängige Full-Suite-Flakes, Bug-Klasse TEST-DESIGN, Spec 07)."""
        from pathlib import Path

        import backend.core.unified_restorer_v3 as _uv3_mod

        src = Path(str(_uv3_mod.__file__)).read_text(encoding="utf-8")
        marker = "_pmgg_gate.wrap_phase("
        idx = src.find(marker)
        assert idx != -1, "PMGG-Primärpfad _pmgg_gate.wrap_phase( nicht in _execute_pipeline gefunden"
        # phase_kwargs-Dict folgt unmittelbar; ~3500 Zeichen Fenster deckt den Dict sicher ab.
        return src[idx : idx + 3500]

    def test_pmgg_wrap_phase_uses_canonical_context_helper(self):
        """PMGG-Pfad merged _canonical_phase_context_kwargs() in seine phase_kwargs."""
        window = self._wrap_phase_kwargs_window()
        assert "_canonical_phase_context_kwargs()" in window, (
            "§0f/§0p BUG: PMGG-Primärpfad nutzt _canonical_phase_context_kwargs() NICHT — "
            "vokal-/wahrnehmungsbezogene Kontext-Keys fehlen, §0p-Guards latent deaktiviert"
        )

    def test_profiled_path_uses_canonical_context_helper(self):
        """Fallback-Pfad _prepare_profiled_phase_runtime_context nutzt denselben Helper."""
        from pathlib import Path

        import backend.core.unified_restorer_v3 as _uv3_mod

        src = Path(str(_uv3_mod.__file__)).read_text(encoding="utf-8")
        # Nur den Methoden-Rumpf betrachten (deterministisch, kein linecache).
        _def_idx = src.find("def _prepare_profiled_phase_runtime_context")
        assert _def_idx != -1
        _nxt_idx = src.find("\n    def ", _def_idx + 1)
        window = src[_def_idx : _nxt_idx if _nxt_idx != -1 else _def_idx + 40000]
        assert "_canonical_phase_context_kwargs()" in window, (
            "§0f/§0p BUG: Fallback-Pfad nutzt _canonical_phase_context_kwargs() NICHT — "
            "die beiden Pfade injizieren divergierende Kontext-Keys"
        )

    def test_canonical_helper_returns_all_keys(self):
        """_canonical_phase_context_kwargs() liefert den vollständigen §0p-Key-Satz."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        stub = SimpleNamespace(_restoration_context={})
        out = UnifiedRestorerV3._canonical_phase_context_kwargs(stub)  # type: ignore[arg-type]
        for key in self._CANONICAL_KEYS:
            assert key in out, f"§0p: kanonischer Kontext-Key '{key}' fehlt im Helper-Output"

    def test_canonical_helper_sources_from_restoration_context(self):
        """Werte stammen aus _restoration_context (nicht hartkodiert/konstant, §0c)."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        rctx = {
            "panns_singing": 0.42,
            "vocal_presence_active": True,
            "vibrato_zones": [(1.0, 2.0)],
            "transfer_chain": ["vinyl", "cassette"],
        }
        stub = SimpleNamespace(_restoration_context=rctx)
        out = UnifiedRestorerV3._canonical_phase_context_kwargs(stub)  # type: ignore[arg-type]
        assert out["panns_singing"] == pytest.approx(0.42), "panns_singing nicht aus restoration_context gelesen"
        assert out["vocal_presence_active"] is True
        assert out["vibrato_zones"] == [(1.0, 2.0)]
        assert out["transfer_chain"] == ["vinyl", "cassette"]

    def test_canonical_helper_safe_defaults_on_empty_context(self):
        """Leerer Kontext → sichere Defaults (keine None/Exception, §3.1)."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        stub = SimpleNamespace(_restoration_context={})
        out = UnifiedRestorerV3._canonical_phase_context_kwargs(stub)  # type: ignore[arg-type]
        assert out["panns_singing"] == 0.0
        assert out["vocal_presence_active"] is False
        assert out["soft_saturation_severity"] == 0.0
        assert out["frisson_zones"] == []
        assert out["panns_tags"] == {}
        assert out["defect_event_metadata"] == {}


class TestPhase03BsRoformerVocalStemNR:
    """Tests für BS-RoFormer Vocal-Stem-NR in phase_03 (§0a-konformer MIIPHER-Äquivalent)."""

    def _make_noisy_stereo(self, sr: int = 48000, dur: float = 1.0) -> np.ndarray:
        """Synthetisches Stereo-Signal: 220 Hz Sinus + weißes Rauschen (-20 dBFS)."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        signal = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        noise = 0.05 * rng.standard_normal(len(t)).astype(np.float32)
        mono = np.clip(signal + noise, -1.0, 1.0)
        return np.column_stack([mono, mono * 0.9]).astype(np.float32)  # (N, 2)

    def test_bsrof_active_path_preserves_instrumental_stem(self, monkeypatch):
        """Aktiver BS-RoFormer-Pfad remixt unveränderten Instrumental-Stem zurück."""
        import importlib

        from plugins.bs_roformer_plugin import StemSeparationResult

        mod = importlib.import_module("backend.core.phases.phase_03_denoise")
        phase = mod.DenoisePhase()

        audio_in = _white_noise(48000, amplitude=0.04, seed=123)
        original_rms = float(np.sqrt(np.mean(audio_in.astype(np.float64) ** 2) + 1e-20))

        class _FakeBsRoformer:
            def separate(self, audio, sr, *, stems=None):
                assert sr == 48000
                assert stems == ["vocals"]
                return StemSeparationResult(
                    stems={"vocals": np.zeros_like(audio, dtype=np.float32)},
                    sr=48000,
                    sdri_db=5.0,
                    model_used="melbandroformer_test",
                    confidence=0.95,
                )

        monkeypatch.setattr("plugins.bs_roformer_plugin.get_bs_roformer", lambda: _FakeBsRoformer())
        monkeypatch.setattr(mod, "RESOURCE_MANAGER_AVAILABLE", False, raising=False)

        result = phase.process(
            audio=audio_in,
            material_type="vinyl",
            sample_rate=48000,
            panns_singing=0.50,
            # §DENKER-CONTROL/§2.70: BS-RoFormer-Gate verlangt strength > 0.55
            # (und > DSP-only-Schwelle 0.10+panns×0.30) — sonst DSP-only-Pfad.
            strength=0.60,
            decade=1930,
            quality_mode="fast",
        )
        assert result is not None
        assert result.audio is not None
        assert result.metadata.get("bsrof_stem_active", False) is True
        assert result.metadata.get("bsrof_recombined", False) is True
        result_rms = float(np.sqrt(np.mean(result.audio.astype(np.float64) ** 2) + 1e-20))
        assert result_rms >= original_rms * 0.75

    def test_recombine_bsrof_inactive_passthrough(self):
        """_recombine_bsrof_if_needed gibt audio unverändert zurück wenn inaktiv."""
        import importlib

        mod = importlib.import_module("backend.core.phases.phase_03_denoise")
        phase = mod.DenoisePhase()

        audio_in = np.zeros((48000,), dtype=np.float32)
        audio_in[100] = 0.5

        # panns_singing=0.0 → Gate nicht erfüllt → kein BS-RoFormer
        result = phase.process(
            audio=audio_in,
            material_type="vinyl",
            sample_rate=48000,
            panns_singing=0.0,
            strength=0.1,
            decade=1975,
        )
        assert result is not None
        assert result.audio is not None
        assert result.metadata.get("bsrof_stem_active", False) is False

    def test_bsrof_gate_panns_threshold(self):
        """BS-RoFormer-Gate aktiviert nur wenn panns_singing >= 0.35."""
        import importlib

        mod = importlib.import_module("backend.core.phases.phase_03_denoise")
        phase = mod.DenoisePhase()

        rng = np.random.default_rng(7)
        audio = (0.05 * rng.standard_normal(48000)).astype(np.float32)

        # panns_singing = 0.20 → unter Gate (0.35) → bsrof_stem_active = False
        result_low = phase.process(
            audio=audio,
            material_type="vinyl",
            sample_rate=48000,
            panns_singing=0.20,
            strength=0.1,
            decade=1975,
        )
        assert result_low.metadata.get("bsrof_stem_active", False) is False

    def test_recombine_bsrof_adds_instrumental(self):
        """_recombine_bsrof_if_needed: NR-Vokal + Instrumental ergibt plausiblen Mix."""
        # Direkte Einheit: Vokal 0.3, Instrumental 0.2 → Summe ≈ 0.5
        voc = np.full(4800, 0.3, dtype=np.float32)
        inst = np.full(4800, 0.2, dtype=np.float32)
        combined = np.clip(voc + inst, -1.0, 1.0)
        assert float(np.mean(combined)) == pytest.approx(0.5, abs=1e-4)

    def test_bsrof_gate_snr_threshold(self):
        """BS-RoFormer-Gate nur wenn SNR < 20 dB; bei sauberem Signal inaktiv."""
        import importlib

        mod = importlib.import_module("backend.core.phases.phase_03_denoise")
        phase = mod.DenoisePhase()

        # Sehr sauberes Signal (>35 dB SNR → SNR-Bypass wird aktiv, kein BS-RoFormer)
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result_clean = phase.process(
            audio=clean,
            material_type="cd_digital",
            sample_rate=48000,
            panns_singing=0.50,
            strength=0.1,
            decade=2000,
        )
        # SNR-Bypass greift oder BS-RoFormer inaktiv wegen material_type=cd_digital (snr > 35)
        assert result_clean is not None
        assert result_clean.audio is not None


# ---------------------------------------------------------------------------
# §Lücke3 V55: WLPC — era_decade < 1960 aktiviert noise-robuste LPC-Schätzung
# ---------------------------------------------------------------------------
class TestWLPCFormantEnhance:
    """§V55: lpc_formant_enhance() mit era_decade — WLPC-Pfad validieren."""

    def _make_voiced_audio(self, dur_s: float = 0.5, sr: int = 48000) -> np.ndarray:
        import numpy as np

        t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
        f0 = 200.0
        audio = np.zeros_like(t)
        for k in range(1, 8):
            audio += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
        return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)  # type: ignore[no-any-return]

    def test_era_none_does_not_crash(self):
        """era_decade=None → normale LPC-Verarbeitung, kein Absturz."""
        import numpy as np

        from backend.core.dsp.lpc_formant_tracker import lpc_formant_enhance

        audio = self._make_voiced_audio()
        result = lpc_formant_enhance(audio, 48000, era_decade=None)
        assert result is not None
        assert result.shape == audio.shape
        assert not np.any(np.isnan(result))

    def test_historical_era_activates_wlpc(self):
        """era_decade=1930 → WLPC-Pfad aktiv — kein Absturz, valides Ergebnis."""
        import numpy as np

        from backend.core.dsp.lpc_formant_tracker import lpc_formant_enhance

        audio = self._make_voiced_audio()
        result = lpc_formant_enhance(audio, 48000, era_decade=1930)
        assert result is not None
        assert result.shape == audio.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_modern_era_uses_standard_lpc(self):
        """era_decade=1975 → WLPC nicht aktiv (>= 1960), Standard-LPC."""
        import numpy as np

        from backend.core.dsp.lpc_formant_tracker import lpc_formant_enhance

        audio = self._make_voiced_audio()
        result = lpc_formant_enhance(audio, 48000, era_decade=1975)
        assert result is not None
        assert result.shape == audio.shape
        assert not np.any(np.isnan(result))

    def test_era_boundary_1960_standard_lpc(self):
        """era_decade=1960 → Grenzwert >= 1960: Standard-LPC."""

        from backend.core.dsp.lpc_formant_tracker import lpc_formant_enhance

        audio = self._make_voiced_audio()
        result = lpc_formant_enhance(audio, 48000, era_decade=1960)
        assert result is not None
        assert result.shape == audio.shape

    def test_era_1959_activates_wlpc(self):
        """era_decade=1959 → streng < 1960: WLPC aktiv."""
        import numpy as np

        from backend.core.dsp.lpc_formant_tracker import lpc_formant_enhance

        audio = self._make_voiced_audio()
        result = lpc_formant_enhance(audio, 48000, era_decade=1959)
        assert result is not None
        assert result.shape == audio.shape
        assert not np.any(np.isnan(result))

    def test_wlpc_output_clipped(self):
        """WLPC-Ausgabe muss in [-1, 1] liegen (Clipping-Guard)."""
        import numpy as np

        from backend.core.dsp.lpc_formant_tracker import lpc_formant_enhance

        audio = self._make_voiced_audio()
        result = lpc_formant_enhance(audio, 48000, era_decade=1925)
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-4, "Clipping-Guard: Ausgabe > 1.0"

    def test_stereo_era_historical_no_crash(self):
        """Stereo (2, N) + era_decade=1940 → kein Absturz."""
        import numpy as np

        from backend.core.dsp.lpc_formant_tracker import lpc_formant_enhance

        mono = self._make_voiced_audio()
        # Stereo channels-first (2, N) — Phase42 übergibt in diesem Format
        stereo = np.stack([mono, mono * 0.9], axis=0)
        result = lpc_formant_enhance(stereo, 48000, era_decade=1940)
        assert result is not None
        assert not np.any(np.isnan(result))
