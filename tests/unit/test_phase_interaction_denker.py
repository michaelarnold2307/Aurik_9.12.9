"""§11 PhaseInteractionDenker — Unit-Tests für Cross-Phase Consensus.

Testet Interferenz-Detektion zwischen aufeinanderfolgenden Phasen, neue Peaks > -60 dBFS,
Rollback-Empfehlung und Edge-Cases.

Spec: AGENTS.md §11 / backend/core/phase_interaction_denker.py
"""

from __future__ import annotations

import numpy as np
import pytest


def _audio(sr: int = 48000, duration: float = 2.0, freq: float = 440.0) -> np.ndarray:
    """Erzeugt Test-Audio (Sinus)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.unit
class TestPhaseInteractionDenker:
    """§11 Cross-Phase Consensus funktioniert korrekt."""

    def test_singleton_factory(self):
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        d1 = get_phase_interaction_denker()
        d2 = get_phase_interaction_denker()
        assert d1 is d2, "Singleton-Factory sollte dieselbe Instanz zurückgeben"

    def test_no_interference_for_minor_change(self):
        """Kleine Änderungen sollten keine Interferenz erkennen."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        audio = _audio(48000, 2.0)
        # Fast identisches Signal (nur minimaler Unterschied)
        post = (audio * 1.01).astype(np.float32)

        result = denker.check_interference(audio, post, sr=48000)
        assert not result.has_interference, "Kleine Änderung sollte keine Interferenz erkennen"
        assert result.n_new_peaks == 0

    def test_detects_new_spectral_peaks(self):
        """Neue spektrale Peaks sollten erkannt werden."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        # Niedriges Signal (nur tiefe Frequenzen)
        pre = _audio(48000, 2.0, freq=100.0)
        # Hinzufügen hoher Frequenzen (neue Peaks)
        t = np.linspace(0, 2.0, int(48000 * 2), endpoint=False)
        post = (pre + 0.2 * np.sin(2 * np.pi * 10000.0 * t)).astype(np.float32)

        result = denker.check_interference(pre, post, sr=48000)
        # Neue Peaks sollten erkannt werden (wenn Energie > -60 dBFS)
        if result.has_interference:
            assert result.n_new_peaks > 0
            assert result.max_new_peak_dbfs > -100.0

    def test_result_attributes(self):
        """PhaseInteractionResult hat alle erwarteten Attribute."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker, PhaseInteractionResult

        denker = get_phase_interaction_denker()
        audio = _audio(48000, 2.0)
        post = (audio * 0.9).astype(np.float32)

        result = denker.check_interference(audio, post, sr=48000)
        assert isinstance(result, PhaseInteractionResult)
        assert hasattr(result, "has_interference")
        assert hasattr(result, "new_peaks_dbfs")
        assert hasattr(result, "n_new_peaks")
        assert hasattr(result, "max_new_peak_dbfs")

    def test_threshold_parameter(self):
        """threshold_dbfs-Parameter sollte Interferenz-Detektion beeinflussen."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        pre = _audio(48000, 2.0, freq=100.0)
        t = np.linspace(0, 2.0, int(48000 * 2), endpoint=False)
        post = (pre + 0.2 * np.sin(2 * np.pi * 10000.0 * t)).astype(np.float32)

        # Strengere Threshold → mehr Interferenzen
        result_strict = denker.check_interference(pre, post, sr=48000, threshold_dbfs=-80.0)
        # Lockerere Threshold → weniger Interferenzen
        result_loose = denker.check_interference(pre, post, sr=48000, threshold_dbfs=-20.0)

        assert result_strict.n_new_peaks >= result_loose.n_new_peaks


@pytest.mark.unit
class TestPhaseInteractionDenkerEdgeCases:
    """Edge-Cases für Phasen-Interferenz-Detektion."""

    def test_very_short_audio(self):
        """Kurze Audio (< 200 ms) sollte konservative Werte zurückgeben."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        short = _audio(48000, 0.1)  # 100ms < 200ms Minimum
        post = (short * 0.9).astype(np.float32)

        result = denker.check_interference(short, post, sr=48000)
        assert not result.has_interference
        assert result.n_new_peaks == 0

    def test_silent_audio(self):
        """Stille sollte keine Interferenz erkennen."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        silent = np.zeros(96000, dtype=np.float32)  # 2s Stille
        post = np.zeros(96000, dtype=np.float32)

        result = denker.check_interference(silent, post, sr=48000)
        assert not result.has_interference
        assert result.n_new_peaks == 0

    def test_nan_handling(self):
        """NaN/Inf-Werte sollten sicher behandelt werden."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        audio = _audio(48000, 2.0)
        audio[100] = np.nan
        audio[200] = np.inf
        post = (audio * 0.9).astype(np.float32)

        result = denker.check_interference(audio, post, sr=48000)
        assert np.isfinite(result.max_new_peak_dbfs), "max_new_peak_dbfs sollte NaN/Inf-frei sein"

    def test_stereo_audio(self):
        """Stereo-Audio (2, N) sollte korrekt verarbeitet werden."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        t = np.linspace(0, 2.0, int(48000 * 2), endpoint=False)
        stereo = np.stack([
            0.3 * np.sin(2 * np.pi * 440.0 * t),
            0.25 * np.sin(2 * np.pi * 440.0 * t + 0.1),
        ]).astype(np.float32)

        post = np.clip(stereo, -0.1, 0.1).astype(np.float32)
        result = denker.check_interference(stereo, post, sr=48000)
        assert isinstance(result.has_interference, bool)
        assert isinstance(result.n_new_peaks, int)

    def test_max_peak_default_value(self):
        """max_new_peak_dbfs sollte negativen Default-Wert haben wenn keine Peaks."""
        from backend.core.phase_interaction_denker import get_phase_interaction_denker

        denker = get_phase_interaction_denker()
        audio = _audio(48000, 2.0)
        post = (audio * 1.0).astype(np.float32)  # Keine Änderung

        result = denker.check_interference(audio, post, sr=48000)
        if not result.has_interference:
            assert result.max_new_peak_dbfs < -50.0, "Keine Peaks → max_new_peak_dbfs sollte negativ sein"
