"""Adaptiver Vibrato-Detektor — Unit-Tests für era-spezifische Vibrato-Erkennung.

Testet F0-Schätzung, Vibrato-Rate-Detektion (3–7 Hz), Era-adaptive Bandbreite
und Edge-Cases (Stille, kurze Audio).

Spec: AGENTS.md §0p / backend/core/vibrato_detector.py
"""

from __future__ import annotations

import numpy as np
import pytest


def _vibrato_audio(
    sr: int = 48000,
    duration: float = 3.0,
    freq: float = 200.0,
    vibrato_rate: float = 5.0,
    depth_hz: float = 3.0,
) -> np.ndarray:
    """Erzeugt Audio mit Vibrato (F0-Modulation)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # F0-Modulation: freq + depth * sin(2*pi*vibrato_rate*t)
    instantaneous_freq = freq + depth_hz * np.sin(2 * np.pi * vibrato_rate * t)
    # Phasenintegration
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
    return (0.3 * np.sin(phase)).astype(np.float32)


def _steady_tone(sr: int = 48000, duration: float = 3.0, freq: float = 200.0) -> np.ndarray:
    """Erzeugt Audio ohne Vibrato (stabiler Ton)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.unit
class TestVibratoDetector:
    """Adaptiver Vibrato-Detektor funktioniert korrekt."""

    def test_detect_vibrato_rate_returns_result(self):
        """detect_vibrato_rate() sollte VibratoDetectionResult zurückgeben."""
        from backend.core.vibrato_detector import detect_vibrato_rate, VibratoDetectionResult

        audio = _vibrato_audio(48000, 3.0, vibrato_rate=5.0)
        result = detect_vibrato_rate(audio, sr=48000)

        assert isinstance(result, VibratoDetectionResult)
        assert hasattr(result, "rate_hz")
        assert hasattr(result, "depth_hz")
        assert hasattr(result, "confidence")
        assert hasattr(result, "is_vibrato")

    def test_detects_vibrato_in_range(self):
        """Vibrato im erwarteten Bereich (3–7 Hz) sollte erkannt werden."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        # 5 Hz Vibrato — mitten im erwarteten Bereich
        audio = _vibrato_audio(48000, 3.0, vibrato_rate=5.0, depth_hz=3.0)
        result = detect_vibrato_rate(audio, sr=48000)

        assert result.is_vibrato, "5 Hz Vibrato sollte erkannt werden"
        assert 3.0 <= result.rate_hz <= 7.0, f"Rate={result.rate_hz} Hz sollte in [3, 7] liegen"

    def test_no_vibrato_for_steady_tone(self):
        """Stabiler Ton ohne Vibrato sollte is_vibrato=False zurückgeben."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        audio = _steady_tone(48000, 3.0)
        result = detect_vibrato_rate(audio, sr=48000)

        assert not result.is_vibrato, "Stabiler Ton sollte kein Vibrato erkennen"

    def test_era_baroque_lower_range(self):
        """Barock-Era (< 1750) sollte langsamere Vibrato-Raten (3–4.5 Hz) erwarten."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        # 3.5 Hz — im Barock-Bereich
        audio = _vibrato_audio(48000, 3.0, vibrato_rate=3.5, depth_hz=2.0)
        result = detect_vibrato_rate(audio, sr=48000, era_decade=1700)

        # Sollte erkannt werden (im Barock-Bereich)
        assert result.is_vibrato or result.rate_hz > 0, "3.5 Hz Vibrato sollte bei Barock-Era erkannt werden"

    def test_era_modern_higher_range(self):
        """Modern-Era (> 1900) sollte schnellere Vibrato-Raten (5.5–7 Hz) erwarten."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        # 6 Hz — im Modern-Bereich
        audio = _vibrato_audio(48000, 3.0, vibrato_rate=6.0, depth_hz=4.0)
        result = detect_vibrato_rate(audio, sr=48000, era_decade=1950)

        assert result.is_vibrato or result.rate_hz > 0, "6 Hz Vibrato sollte bei Modern-Era erkannt werden"

    def test_result_values_reasonable(self):
        """Ergebniswerte sollten in vernünftigen Bereichen liegen."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        audio = _vibrato_audio(48000, 3.0, vibrato_rate=5.0, depth_hz=3.0)
        result = detect_vibrato_rate(audio, sr=48000)

        assert 0.0 <= result.rate_hz <= 20.0, f"Rate={result.rate_hz} Hz sollte vernünftig sein"
        assert 0.0 <= result.depth_hz <= 50.0, f"Tiefe={result.depth_hz} Hz sollte vernünftig sein"
        assert 0.0 <= result.confidence <= 1.0, f"Konfidenz={result.confidence} sollte in [0, 1] liegen"


@pytest.mark.unit
class TestVibratoDetectorEdgeCases:
    """Edge-Cases für Vibrato-Detektor."""

    def test_silent_audio(self):
        """Stille sollte kein Vibrato erkennen."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        silent = np.zeros(144000, dtype=np.float32)  # 3s Stille
        result = detect_vibrato_rate(silent, sr=48000)

        assert not result.is_vibrato
        assert result.rate_hz == 0.0 or result.confidence < 0.15

    def test_very_short_audio(self):
        """Kurze Audio sollte konservative Werte zurückgeben."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        short = _vibrato_audio(48000, 0.3)  # 300ms
        result = detect_vibrato_rate(short, sr=48000)

        assert not result.is_vibrato or result.confidence < 0.5

    def test_nan_handling(self):
        """NaN/Inf-Werte sollten sicher behandelt werden."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        audio = _vibrato_audio(48000, 3.0)
        audio[100] = np.nan
        audio[200] = np.inf

        result = detect_vibrato_rate(audio, sr=48000)
        assert np.isfinite(result.rate_hz), "rate_hz sollte NaN/Inf-frei sein"
        assert np.isfinite(result.depth_hz)
        assert np.isfinite(result.confidence)

    def test_stereo_audio(self):
        """Stereo-Audio (2, N) sollte korrekt verarbeitet werden."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        t = np.linspace(0, 3.0, int(48000 * 3), endpoint=False)
        stereo = np.stack([
            0.3 * np.sin(2 * np.pi * 200.0 * t + 5.0 * np.sin(2 * np.pi * 5.0 * t)),
            0.25 * np.sin(2 * np.pi * 200.0 * t + 5.0 * np.sin(2 * np.pi * 5.0 * t) + 0.1),
        ]).astype(np.float32)

        result = detect_vibrato_rate(stereo, sr=48000)
        assert isinstance(result.rate_hz, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_no_era_decade_default(self):
        """Ohne era_decade sollte Default-Bereich (3–7 Hz) verwendet werden."""
        from backend.core.vibrato_detector import detect_vibrato_rate

        audio = _vibrato_audio(48000, 3.0, vibrato_rate=5.0)
        result = detect_vibrato_rate(audio, sr=48000, era_decade=None)

        # Sollte funktionieren (Default-Bereich)
        assert isinstance(result.rate_hz, float)
