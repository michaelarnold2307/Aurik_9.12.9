"""§0p DynamicPreservationGuard — Unit-Tests für Dynamik-Erhaltungs-Guard.

Testet RMS/Peak-Verhältnis-Messung, Rollback-Entscheidung bei > 3 dB Reduktion,
strength_scalar-Berechnung und Edge-Cases (Stille, kurze Audio).

Spec: .github/specs/01_musical_goals.md §0p / AGENTS.md §V5
"""

from __future__ import annotations

import numpy as np
import pytest


def _audio(sr: int = 48000, duration: float = 2.0, freq: float = 440.0) -> np.ndarray:
    """Erzeugt Test-Audio (Sinus)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _compressed_audio(audio: np.ndarray, compression: float = 0.5) -> np.ndarray:
    """Komprimiert Audio durch Amplituden-Begrenzung."""
    return (np.clip(audio, -compression, compression) ** 1.5).astype(np.float32)


@pytest.mark.unit
class TestDynamicPreservationGuard:
    """Dynamik-Erhaltungs-Guard funktioniert korrekt."""

    def test_singleton_factory(self):
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        g1 = get_dynamic_preservation_guard()
        g2 = get_dynamic_preservation_guard()
        assert g1 is g2, "Singleton-Factory sollte dieselbe Instanz zurückgeben"

    def test_no_rollback_for_minor_change(self):
        """Kleine Änderungen (< 3 dB) sollten kein Rollback auslösen."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        audio = _audio(48000, 2.0)
        # Fast identisches Signal (nur minimaler Unterschied)
        post = (audio * 1.01).astype(np.float32)

        decision = guard.evaluate(audio, post, sr=48000)
        assert not decision.rollback, "Kleine Änderung sollte kein Rollback auslösen"
        assert decision.strength_scalar > 0.5
        assert decision.delta_db < 3.0

    def test_rollback_for_heavy_compression(self):
        """Starke Kompression (> 3 dB RMS/Peak-Reduktion) sollte Rollback auslösen."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        # Signal mit hohem Crest-Faktor (große Dynamik: Peaks >> RMS)
        t = np.linspace(0, 2.0, int(48000 * 2), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440.0 * t) +
                 0.1 * np.random.randn(len(t)) * 0.1).astype(np.float32)
        # Hard-Limiter: Peaks werden stark gekappt → RMS bleibt ähnlich, Peak sinkt
        post = np.clip(audio, -0.08, 0.08).astype(np.float32)

        decision = guard.evaluate(audio, post, sr=48000)
        assert decision.rollback, f"Starke Kompression sollte Rollback auslösen (delta={decision.delta_db:.2f} dB)"
        assert decision.strength_scalar <= 0.5
        assert decision.delta_db > 3.0

    def test_strength_scalar_bounds(self):
        """strength_scalar liegt immer in [0, 1]."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        audio = _audio(48000, 2.0)

        # Verschiedene Kompressionsstufen
        for comp in [0.9, 0.7, 0.5, 0.3, 0.1]:
            post = np.clip(audio, -comp, comp).astype(np.float32)
            decision = guard.evaluate(audio, post, sr=48000)
            assert 0.0 <= decision.strength_scalar <= 1.0, f"strength_scalar={decision.strength_scalar} für comp={comp}"

    def test_decision_attributes(self):
        """DynamicPreservationDecision hat alle erwarteten Attribute."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        audio = _audio(48000, 2.0)
        post = (audio * 0.9).astype(np.float32)

        decision = guard.evaluate(audio, post, sr=48000)
        assert hasattr(decision, "rollback")
        assert hasattr(decision, "strength_scalar")
        assert hasattr(decision, "rms_peak_ratio_pre")
        assert hasattr(decision, "rms_peak_ratio_post")
        assert hasattr(decision, "delta_db")

    def test_rms_peak_ratio_negative(self):
        """RMS/Peak-Verhältnis ist immer negativ (Peak > RMS)."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        audio = _audio(48000, 2.0)
        post = (audio * 0.9).astype(np.float32)

        decision = guard.evaluate(audio, post, sr=48000)
        assert decision.rms_peak_ratio_pre < 0, "RMS/Peak-Verhältnis sollte negativ sein"
        assert decision.rms_peak_ratio_post < 0


@pytest.mark.unit
class TestDynamicPreservationGuardEdgeCases:
    """Edge-Cases für Dynamik-Erhaltungs-Guard."""

    def test_very_short_audio(self):
        """Kurze Audio (< 500 ms) sollte kein Rollback auslösen (konservativ)."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        short = _audio(48000, 0.3)  # 300ms < 500ms Minimum
        post = np.clip(short, -0.1, 0.1).astype(np.float32)

        decision = guard.evaluate(short, post, sr=48000)
        assert not decision.rollback, "Kurze Audio sollte konservativ sein (kein Rollback)"

    def test_silent_audio(self):
        """Stille sollte konservative Werte zurückgeben."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        silent = np.zeros(96000, dtype=np.float32)  # 2s Stille
        post = np.zeros(96000, dtype=np.float32)

        decision = guard.evaluate(silent, post, sr=48000)
        assert not decision.rollback
        assert decision.strength_scalar == 1.0

    def test_nan_handling(self):
        """NaN/Inf-Werte sollten sicher behandelt werden."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        audio = _audio(48000, 2.0)
        audio[100] = np.nan
        audio[200] = np.inf
        post = (audio * 0.9).astype(np.float32)

        decision = guard.evaluate(audio, post, sr=48000)
        assert np.isfinite(decision.delta_db), "delta_db sollte NaN/Inf-frei sein"
        assert np.isfinite(decision.strength_scalar)

    def test_stereo_audio(self):
        """Stereo-Audio (2, N) sollte korrekt verarbeitet werden."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        t = np.linspace(0, 2.0, int(48000 * 2), endpoint=False)
        stereo = np.stack([
            0.3 * np.sin(2 * np.pi * 440.0 * t),
            0.25 * np.sin(2 * np.pi * 440.0 * t + 0.1),
        ]).astype(np.float32)  # (2, N) channel-first

        post = np.clip(stereo, -0.1, 0.1).astype(np.float32)
        decision = guard.evaluate(stereo, post, sr=48000)
        assert 0.0 <= decision.strength_scalar <= 1.0
        assert np.isfinite(decision.delta_db)

    def test_zero_signal(self):
        """Null-Signal sollte konservative Werte zurückgeben."""
        from backend.core.dynamic_preservation_guard import get_dynamic_preservation_guard

        guard = get_dynamic_preservation_guard()
        zeros = np.zeros(96000, dtype=np.float32)
        decision = guard.evaluate(zeros, zeros, sr=48000)
        assert not decision.rollback
        assert decision.strength_scalar == 1.0
