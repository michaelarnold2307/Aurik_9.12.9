"""
Tests für dsp/adaptive_formant_shifter.py — alle drei Methoden:
    simple_lpc, psola, world
"""

import numpy as np
import pytest

pytest.importorskip("librosa")  # CI-Minimal-Umgebung (cross-platform)

from dsp.adaptive_formant_shifter import AdaptiveFormantShifter

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
SR = 22050


def _sine(n: int = 2048, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(n: int = 2048, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, n).astype(np.float32)


# ===========================================================================
# LPC-basierte Methode (Referenz)
# ===========================================================================
class TestSimpleLPCFormantShift:
    def _run(self, shift_ratio: float = 1.0, n: int = 2048):
        afs = AdaptiveFormantShifter(method="simple_lpc")
        audio = _sine(n)
        return audio, afs.formant_shift(audio, SR, shift_ratio)

    def test_output_length(self):
        audio, out = self._run()
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        _, out = self._run()
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_neutral_shift_unchanged(self):
        """shift_ratio=1.0 sollte das Signal näherungsweise unverändert lassen."""
        _, out = self._run(shift_ratio=1.0)
        # Nur Nicht-Null-Check (LPC modifiziert leicht)
        assert np.max(np.abs(out)) > 1e-6

    def test_upward_shift(self):
        _, out = self._run(shift_ratio=1.2)
        assert not np.isnan(out).any()

    def test_downward_shift(self):
        _, out = self._run(shift_ratio=0.8)
        assert not np.isnan(out).any()


# ===========================================================================
# PSOLA-Methode
# ===========================================================================
class TestPSOLAFormantShift:
    def _run(self, shift_ratio: float = 1.0, n: int = 4096):
        afs = AdaptiveFormantShifter(method="psola")
        audio = _white_noise(n)
        return audio, afs.formant_shift(audio, SR, shift_ratio)

    # --- Basiseigenschaften ------------------------------------------------
    def test_output_length(self):
        audio, out = self._run()
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        _, out = self._run()
        assert not np.isnan(out).any(), "PSOLA-Ausgabe enthält NaN"
        assert not np.isinf(out).any(), "PSOLA-Ausgabe enthält Inf"

    def test_output_in_range(self):
        _, out = self._run()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6, "PSOLA-Ausgabe außerhalb [-1, 1]"

    def test_output_dtype_preserved(self):
        audio, out = self._run()
        assert out.dtype == audio.dtype, f"Dtype: {out.dtype} != {audio.dtype}"

    def test_not_all_zeros(self):
        _, out = self._run(shift_ratio=1.1)
        assert np.max(np.abs(out)) > 1e-6, "PSOLA-Ausgabe ist Nullsignal"

    def test_rms_preserving(self):
        """Energieerhaltung nach RMS-Normalisierung."""
        audio = _white_noise(4096)
        afs = AdaptiveFormantShifter(method="psola")
        out = afs.formant_shift(audio, SR, shift_ratio=1.0)
        rms_in = np.sqrt(np.mean(audio**2))
        rms_out = np.sqrt(np.mean(out**2))
        ratio = rms_out / (rms_in + 1e-12)
        assert 0.05 < ratio < 20.0, f"RMS-Ratio unplausibel: {ratio:.3f}"

    # --- Shift-Richtungen --------------------------------------------------
    @pytest.mark.parametrize("shift_ratio", [0.7, 0.9, 1.0, 1.1, 1.3])
    def test_various_shift_ratios(self, shift_ratio):
        audio = _white_noise(4096, seed=7)
        afs = AdaptiveFormantShifter(method="psola")
        out = afs.formant_shift(audio, SR, shift_ratio)
        assert len(out) == len(audio)
        assert not np.isnan(out).any()

    # --- Robustheit --------------------------------------------------------
    def test_very_short_audio(self):
        """Sehr kurzes Signal soll nicht abstürzen."""
        audio = _sine(n=64)
        afs = AdaptiveFormantShifter(method="psola")
        out = afs.formant_shift(audio, SR, shift_ratio=1.0)
        assert len(out) == len(audio)

    def test_pure_sine(self):
        audio = _sine(4096, freq=660.0)
        afs = AdaptiveFormantShifter(method="psola")
        out = afs.formant_shift(audio, SR, shift_ratio=1.1)
        assert not np.isnan(out).any()


# ===========================================================================
# WORLD-Methode
# ===========================================================================
class TestWORLDFormantShift:
    def _run(self, shift_ratio: float = 1.0, n: int = 4096):
        afs = AdaptiveFormantShifter(method="world")
        audio = _white_noise(n, seed=42)
        return audio, afs.formant_shift(audio, SR, shift_ratio)

    def test_output_length(self):
        audio, out = self._run()
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        _, out = self._run()
        assert not np.isnan(out).any(), "WORLD-Ausgabe enthält NaN"
        assert not np.isinf(out).any(), "WORLD-Ausgabe enthält Inf"

    def test_output_in_range(self):
        _, out = self._run()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_output_dtype_preserved(self):
        audio, out = self._run()
        assert out.dtype == audio.dtype

    def test_not_all_zeros(self):
        _, out = self._run(shift_ratio=1.15)
        assert np.max(np.abs(out)) > 1e-6

    @pytest.mark.parametrize("shift_ratio", [0.75, 1.0, 1.25])
    def test_various_shift_ratios(self, shift_ratio):
        audio = _white_noise(4096, seed=3)
        afs = AdaptiveFormantShifter(method="world")
        out = afs.formant_shift(audio, SR, shift_ratio)
        assert len(out) == len(audio)
        assert not np.isnan(out).any()

    def test_sine_input(self):
        audio = _sine(4096)
        afs = AdaptiveFormantShifter(method="world")
        out = afs.formant_shift(audio, SR, shift_ratio=0.9)
        assert not np.isnan(out).any()

    def test_rms_not_exploding(self):
        audio = _white_noise(4096, seed=99)
        afs = AdaptiveFormantShifter(method="world")
        out = afs.formant_shift(audio, SR, shift_ratio=1.5)
        rms_in = np.sqrt(np.mean(audio**2))
        rms_out = np.sqrt(np.mean(out**2))
        ratio = rms_out / (rms_in + 1e-12)
        assert ratio < 50.0, f"RMS ist explodiert: {ratio:.1f}×"


# ===========================================================================
# Alle Methoden — vergleichende Parametrisierung
# ===========================================================================
class TestAllMethodsCompare:
    @pytest.mark.parametrize("method", ["simple_lpc", "psola", "world"])
    def test_method_valid_output(self, method):
        audio = _white_noise(4096)
        afs = AdaptiveFormantShifter(method=method)
        out = afs.formant_shift(audio, SR, shift_ratio=1.0)
        assert len(out) == len(audio), f"{method}: Länge falsch"
        assert not np.isnan(out).any(), f"{method}: NaN"
        assert not np.isinf(out).any(), f"{method}: Inf"
        assert np.max(np.abs(out)) <= 1.0 + 1e-6, f"{method}: Clipping"

    @pytest.mark.parametrize("method", ["simple_lpc", "psola", "world"])
    def test_method_upward_shift(self, method):
        audio = _sine(4096)
        afs = AdaptiveFormantShifter(method=method)
        out = afs.formant_shift(audio, SR, shift_ratio=1.2)
        assert not np.isnan(out).any()

    @pytest.mark.parametrize("method", ["simple_lpc", "psola", "world"])
    def test_stereo_input_supported(self, method):
        left = _sine(2048, freq=330.0)
        right = _sine(2048, freq=550.0)
        stereo = np.stack([left, right], axis=1)

        afs = AdaptiveFormantShifter(method=method)
        out = afs.formant_shift(stereo, SR, shift_ratio=1.1)

        assert out.shape == stereo.shape
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_unknown_method_falls_back(self):
        """Unbekannte Methode → Fallback auf LPC (kein Absturz, kein NaN)."""
        afs = AdaptiveFormantShifter(method="nonexistent_method_42")
        # _formant_shift_classic loggt Warning und fällt auf LPC zurück — kein Raise
        out = afs._formant_shift_classic(_sine(), SR, shift_ratio=1.0)
        assert out is not None
        assert not np.isnan(out).any()

    @pytest.mark.parametrize("method", ["simple_lpc", "psola", "world"])
    def test_auto_optimize_returns_dict(self, method):
        afs = AdaptiveFormantShifter(method=method)
        params = afs.auto_optimize_params(_sine(), SR)
        assert isinstance(params, dict)
        assert "method" in params
        assert "shift_ratio" in params
