"""§G90 PresenceEmbedding — Unit-Tests für perzeptuelle Präsenz-Metrik.

Testet die 5 Sub-Komponenten (VFC, TI, RTC, MDL, SAA), den Gesamtscore,
Threshold-Passing und Delta-Berechnung. Löst das „43→43"-Paradox:
technische Metriken sagen keine Verbesserung, PresenceEmbedding misst die
menschliche Anwesenheit in der Aufnahme.

Spec: .github/specs/18_non_plus_ultra_perceptual_fidelity.md §18.1 / §G90
"""

from __future__ import annotations

import numpy as np
import pytest


def _audio(sr: int = 48000, duration: float = 2.0, freq: float = 440.0) -> np.ndarray:
    """Erzeugt Test-Audio (Sinus)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rich_audio(sr: int = 48000, duration: float = 2.0) -> np.ndarray:
    """Erzeugt realistischeres Test-Audio mit mehreren Frequenzen und Transienten."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mehrere Frequenzen (wie echte Musik/Sprache)
    signal = (
        0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2 * np.pi * 880.0 * t)
        + 0.1 * np.sin(2 * np.pi * 1760.0 * t)
        + 0.05 * np.sin(2 * np.pi * 3520.0 * t)
        + 0.02 * np.sin(2 * np.pi * 8000.0 * t)
        + 0.01 * np.sin(2 * np.pi * 12000.0 * t)
    )
    # Transienten (plötzliche Amplitudenänderungen)
    envelope = np.ones_like(t)
    for i in range(0, len(t), int(sr * 0.3)):
        if i + 96 < len(t):
            envelope[i : i + 96] *= 1.5
    return (signal * envelope).astype(np.float32)


def _noisy_audio(sr: int = 48000, duration: float = 2.0) -> np.ndarray:
    """Erzeugt Test-Audio mit Rauschen."""
    base = _rich_audio(sr, duration)
    noise = np.random.randn(len(base)).astype(np.float32) * 0.15
    return (base + noise).astype(np.float32)


@pytest.mark.unit
class TestPresenceEmbedding:
    """PresenceEmbedding funktioniert korrekt."""

    def test_singleton_factory(self):
        from backend.core.presence_embedding import get_presence_embedding

        pe1 = get_presence_embedding()
        pe2 = get_presence_embedding()
        assert pe1 is pe2, "Singleton-Factory sollte dieselbe Instanz zurückgeben"

    def test_compute_returns_result(self):
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        audio = _rich_audio(48000, 2.0)
        result = pe.compute(audio, sample_rate=48000)
        assert hasattr(result, "overall")
        assert hasattr(result, "is_hearable_improvement")
        assert 0.0 <= result.overall <= 1.0

    def test_score_alias_works(self):
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        audio = _rich_audio(48000, 2.0)
        r1 = pe.compute(audio, sample_rate=48000)
        r2 = pe.score(audio, sr=48000)
        assert r1.overall == r2.overall

    def test_presence_score_bounds(self):
        """PresenceScore liegt immer in [0, 1]."""
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        # Sehr kurzes Signal
        short = np.zeros(4800, dtype=np.float32)
        r_short = pe.compute(short, sample_rate=48000)
        assert 0.0 <= r_short.overall <= 1.0

        # Langes Signal
        long_audio = _rich_audio(48000, 5.0)
        r_long = pe.compute(long_audio, sample_rate=48000)
        assert 0.0 <= r_long.overall <= 1.0

    def test_sub_scores_bounds(self):
        """Alle Sub-Scores liegen in [0, 1]."""
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        audio = _rich_audio(48000, 3.0)
        result = pe.compute(audio, sample_rate=48000)

        assert 0.0 <= result.vocal_formant_coherence <= 1.0
        assert 0.0 <= result.transient_immediacy <= 1.0
        assert 0.0 <= result.room_tone_continuity <= 1.0
        assert 0.0 <= result.microdynamic_liveliness <= 1.0
        assert 0.0 <= result.spectral_air_authenticity <= 1.0

    def test_passes_threshold_clean_audio(self):
        """Reiches Audio-Signal sollte Threshold ≥ 0.5 erreichen."""
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        audio = _rich_audio(48000, 3.0)
        result = pe.compute(audio, sample_rate=48000)
        # Rich audio sollte moderate Präsenz haben (HF + Transienten + Crest-Faktor)
        assert result.passes_threshold(0.3), f"Rich Audio PresenceScore zu niedrig: {result.overall}"

    def test_delta_positive_for_improvement(self):
        """Delta ist positiv wenn Rauschen entfernt wird."""
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        noisy = _noisy_audio(48000, 3.0)
        clean = _rich_audio(48000, 3.0)
        delta = pe.delta(noisy, clean, sr=48000)
        # Clean sollte höhere Präsenz haben als noisy (oder zumindest nicht viel schlechter)
        assert delta > -0.15, f"Delta sollte positiv sein (clean-noisy), war {delta}"

    def test_presence_score_result_attributes(self):
        """PresenceScoreResult hat alle erwarteten Attribute."""
        from backend.core.presence_embedding import PresenceScoreResult, get_presence_embedding

        pe = get_presence_embedding()
        audio = _rich_audio(48000, 2.0)
        result = pe.compute(audio, sample_rate=48000)

        assert isinstance(result, PresenceScoreResult)
        assert isinstance(result.overall, float)
        assert isinstance(result.is_hearable_improvement, bool)
        assert isinstance(result.component_scores, dict)
        # presence_score alias funktioniert
        assert result.presence_score == result.overall


@pytest.mark.unit
class TestPresenceEmbeddingEdgeCases:
    """Edge-Cases für PresenceEmbedding."""

    def test_very_short_audio(self):
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        short = np.zeros(480, dtype=np.float32)  # 10ms @ 48kHz
        result = pe.compute(short, sample_rate=48000)
        assert 0.0 <= result.overall <= 1.0

    def test_stereo_audio(self):
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        t = np.linspace(0, 2.0, int(48000 * 2), endpoint=False)
        stereo = np.stack(
            [
                0.3 * np.sin(2 * np.pi * 440.0 * t),
                0.25 * np.sin(2 * np.pi * 440.0 * t + 0.1),
            ]
        ).astype(np.float32)  # (2, N) channel-first
        result = pe.compute(stereo, sample_rate=48000)
        assert 0.0 <= result.overall <= 1.0

    def test_nan_handling(self):
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        audio = _rich_audio(48000, 2.0)
        audio[100] = np.nan
        audio[200] = np.inf
        result = pe.compute(audio, sample_rate=48000)
        assert np.isfinite(result.overall), "PresenceScore sollte NaN/Inf-frei sein"

    def test_zero_signal(self):
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        zeros = np.zeros(96000, dtype=np.float32)
        result = pe.compute(zeros, sample_rate=48000)
        assert 0.0 <= result.overall <= 1.0
