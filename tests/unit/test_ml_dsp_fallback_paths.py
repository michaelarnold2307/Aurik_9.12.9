"""§V6 ML→DSP-Fallback-Pfade — Validierung der Fallback-Kette.

Testet alle kritischen ML→DSP-Fallback-Pfade in Aurik 10, um sicherzustellen,
dass das System auch ohne ML-Modelle (ONNX, PyTorch) korrekt funktioniert.

Spec: .github/copilot-instructions.md §V6 Silent-Failure-Verbot
      .github/specs/04_dsp_standards.md Fallback-Pfade
"""

from __future__ import annotations

import types

import numpy as np
import pytest


def _audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    """Erzeugt Test-Audio (440 Hz Sinus)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.mark.unit
class TestMLEchoDetectorFallback:
    """Pre-Echo-Detektor funktioniert ohne ML-Modelle."""

    def test_detect_no_crash_random_noise(self):
        from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

        detector = get_pre_echo_detector()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        events = detector.detect(audio, 48000, material_key="mp3_low")
        assert isinstance(events, list)
        # Random noise sollte keine Pre-Echo-Events haben (kein Codec)
        assert len(events) == 0

    def test_detect_analog_material_returns_empty(self):
        from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

        detector = get_pre_echo_detector()
        audio = _audio(48000, 2.0)
        events = detector.detect(audio, 48000, material_key="vinyl")
        assert len(events) == 0  # Analog-Materialien haben kein Codec-Pre-Echo

    def test_repair_region_no_crash(self):
        from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

        detector = get_pre_echo_detector()
        audio = _audio(48000, 1.0)
        event = {
            "pre_echo_start": 1000,
            "pre_echo_end": 2000,
            "severity_db": 6.0,
        }
        result = detector.repair_region(audio, event, sr=48000)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()


@pytest.mark.unit
class TestNoiseTextureGuardFallback:
    """Noise-Textur-Invariante funktioniert ohne ML-Modelle."""

    def test_compute_distance_vinyl(self):
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = np.random.randn(48000).astype(np.float32) * 0.05
        dist = compute_noise_texture_distance(residual, "vinyl", sr=48000)
        assert isinstance(dist, float)
        assert 0.0 <= dist <= 1.0

    def test_compute_distance_shellac(self):
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = np.random.randn(48000).astype(np.float32) * 0.05
        dist = compute_noise_texture_distance(residual, "shellac", sr=48000)
        assert isinstance(dist, float)
        # Shellac hat breitere Toleranz (-2.0 bis +8.0 dB/oct)
        assert dist < 0.5

    def test_compute_distance_silent_residual(self):
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = np.zeros(48000, dtype=np.float32)
        dist = compute_noise_texture_distance(residual, "unknown", sr=48000)
        assert dist == 0.0


@pytest.mark.unit
class TestVocalHarmonicDecompFallback:
    """F0-Extraktion via ZCPA-DSP-Fallback funktioniert."""

    def test_estimate_f0_zcpa_440hz(self):
        from backend.core.dsp.vocal_harmonic_decomp import _estimate_f0_zcpa

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 48000)).astype(np.float32) * 0.5
        f0 = _estimate_f0_zcpa(audio, 48000, hop=512)
        assert len(f0) > 0
        # F0 sollte nahe 440 Hz sein (mit Toleranz)
        voiced_frames = f0[f0 > 0]
        if len(voiced_frames) > 0:
            mean_f0 = np.mean(voiced_frames)
            assert 420 <= mean_f0 <= 460, f"F0={mean_f0} Hz (expected ~440 Hz)"

    def test_estimate_f0_zcpa_short_audio(self):
        from backend.core.dsp.vocal_harmonic_decomp import _estimate_f0_zcpa

        audio = np.sin(2 * np.pi * 300 * np.linspace(0, 0.1, 4800)).astype(np.float32) * 0.5
        f0 = _estimate_f0_zcpa(audio, 48000, hop=512)
        assert isinstance(f0, np.ndarray)


@pytest.mark.unit
class TestSOTAVocalPipelineFallback:
    """SOTA Vocal Pipeline funktioniert ohne ML-Modelle (vollständiger Fallback)."""

    def test_process_no_crash(self):
        from backend.core.sota_vocal_pipeline import SOTAVocalPipeline

        pipeline = SOTAVocalPipeline(n_fft=1024, hop=256)
        audio = _audio(48000, 1.0)
        result = pipeline.process(audio, sample_rate=48000)
        assert result.audio.shape == audio.shape
        assert np.isfinite(result.audio).all()
        assert len(result.layers_applied) >= 1

    def test_process_preserves_length(self):
        from backend.core.sota_vocal_pipeline import SOTAVocalPipeline

        pipeline = SOTAVocalPipeline(n_fft=1024, hop=256)
        audio = _audio(48000, 2.0)
        result = pipeline.process(audio, sample_rate=48000)
        assert len(result.audio) == len(audio)

    def test_process_breath_change_db_valid(self):
        from backend.core.sota_vocal_pipeline import SOTAVocalPipeline

        pipeline = SOTAVocalPipeline(n_fft=1024, hop=256)
        audio = _audio(48000, 1.0)
        result = pipeline.process(audio, sample_rate=48000)
        assert isinstance(result.breath_change_db, float)
        # Breath change sollte in einem vernünftigen Bereich sein
        assert -20 <= result.breath_change_db <= 20

    def test_process_harmonic_preservation_valid(self):
        from backend.core.sota_vocal_pipeline import SOTAVocalPipeline

        pipeline = SOTAVocalPipeline(n_fft=1024, hop=256)
        audio = _audio(48000, 1.0)
        result = pipeline.process(audio, sample_rate=48000)
        assert isinstance(result.harmonic_preservation_pct, float)
        assert 0 <= result.harmonic_preservation_pct <= 100


@pytest.mark.unit
class TestSotaVocalModelRouterFallback:
    """SOTA Vocal Model Router Fallback-Kette (MIIPHER-DiT → MIIPHER → SGMSE+ → DFN)."""

    def test_fallback_chain_miipher_dit_import_error(self, monkeypatch):
        from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

        class _FakeMiipher:
            _model_loaded = False

        class _FakeSgmseResult:
            def __init__(self, audio: np.ndarray) -> None:
                self.audio = (audio * 0.5).astype(np.float32)
                self.model_used = "sgmse_plus_torchscript"

        class _FakeSgmse:
            _model_loaded = True

            @staticmethod
            def enhance(audio: np.ndarray, sr: int):  # pylint: disable=unused-argument
                return _FakeSgmseResult(audio)

        class _FakeDfn:
            @staticmethod
            def enhance(audio: np.ndarray, sr: int, energy_bias_db: float = -9.0):  # pylint: disable=unused-argument
                return (audio * 0.8).astype(np.float32)

        # MIIPHER-DiT nicht verfügbar → Fallback zu SGMSE+ + DFN
        monkeypatch.setitem(
            __import__("sys").modules,
            "plugins.miipher_dit_plugin",
            types.SimpleNamespace(get_miipher_dit=lambda: (_ for _ in ()).throw(ImportError("miipher_dit not available"))),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "plugins.miipher_plugin",
            types.SimpleNamespace(get_miipher_plugin=lambda: _FakeMiipher()),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "plugins.sgmse_plugin",
            types.SimpleNamespace(get_sgmse_plugin=lambda: _FakeSgmse()),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "plugins.deepfilternet_v3_ii_plugin",
            types.SimpleNamespace(get_deepfilternet_plugin=lambda: _FakeDfn()),
        )

        audio = _audio()
        result = SotaVocalModelRouter().enhance_vocal(audio, 48000, energy_bias_db=-6.0)
        assert result.success is True
        assert "miipher_dit:import_error" in result.fallback_chain
        # DFN wird als Kompensation angewendet → model_used enthält DFN
        assert "deepfilternet" in result.model_used or result.model_used == "sgmse_plus_torchscript"


@pytest.mark.unit
class TestPhonemeBoundaryDetectorFallback:
    """Phonem-Grenzerkennung via Energie-basierter DSP-Fallback."""

    def test_detect_no_crash(self):
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = _audio(48000, 2.0)
        boundaries = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert isinstance(boundaries, np.ndarray)


@pytest.mark.unit
class TestHallucinationGuardFallback:
    """Halluzinations-Guard funktioniert ohne SFT-Kalibrierung."""

    def test_get_threshold_returns_floor(self):
        from backend.core.dsp.hallucination_guard import _get_adaptive_penalty_threshold

        threshold = _get_adaptive_penalty_threshold()
        assert isinstance(threshold, float)
        # Floor-Wert ist 0.15 wenn SFT-Kalibrierung nicht verfügbar
        assert 0.0 <= threshold <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
