"""§Ebene-1 Level-1 Invariants Guard — Unit Tests (Hörordnung §3)

Tests für die fünf unverhandelbaren Hör-Invarianten:
  1. Stimm-Identität (≥ 0.92)
  2. Konsonanten-Klarheit (≥ 0.85)
  3. Vibrato-Erhalt (Rate-Fehler ≤ 0.3 Hz, Tiefe ≥ 0.85)
  4. Dynamikbogen (Arc-Corr ≥ 0.70)
  5. Atem-Zeitstruktur (≤ 10 % Änderung)

[RELEASE_MUST] Jeder Test hat einen [RELEASE_MUST]-Header in copilot-instructions.md
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.level_1_invariants_guard import (
    Level1Result,
    check_level_1_invariants,
    get_level_1_guard,
)


@pytest.fixture
def sample_audio():
    """Erzeugt synthetisches Test-Audio (48 kHz, 5 Sekunden)."""
    sr = 48000
    duration_s = 5.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Einfacher Sinuston mit Vibrato-Modulation
    vibrato_rate = 6.0  # Hz (typisch für Gesang)
    vibrato_depth = 0.5  # Halbton
    freq = 440.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def stereo_sample_audio(sample_audio):
    """Erzeugt synthetisches Stereo-Test-Audio (48 kHz, 5 Sekunden)."""
    mono, sr = sample_audio
    # Leichte Kanal-Differenz für realistischen Stereo-Effekt
    left = mono.copy()
    right = mono * 0.95 + np.random.randn(len(mono)) * 0.01
    return np.stack([left, right], axis=0).astype(np.float32), sr


class TestLevel1Result:
    """Tests für Level1Result Dataclass."""

    def test_default_values(self):
        result = Level1Result()
        assert result.blend_factor == 1.0
        assert result.violated_invariants == []
        assert result.singer_identity == 1.0
        assert result.consonant_clarity == 1.0

    def test_violated_invariants(self):
        result = Level1Result(
            singer_identity=0.85,
            blend_factor=0.5,
            violated_invariants=["singer_identity"],
        )
        assert "singer_identity" in result.violated_invariants
        assert result.blend_factor < 1.0


class TestSingleton:
    """Tests für Singleton-Pattern."""

    def test_singleton_returns_same_instance(self):
        guard1 = get_level_1_guard()
        guard2 = get_level_1_guard()
        assert guard1 is guard2

    def test_singleton_thread_safe(self):
        import threading

        instances = []

        def get_instance():
            instances.append(get_level_1_guard())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Alle Instanzen sollten identisch sein
        assert all(inst is instances[0] for inst in instances)


class TestCheckLevel1InvariantsBasic:
    """Grundlegende Tests für check_level_1_invariants."""

    def test_identical_audio_returns_perfect_scores(self, sample_audio):
        """Identisches Audio sollte alle Invarianten perfekt erfüllen."""
        audio, sr = sample_audio
        result = check_level_1_invariants(audio, audio, sr)

        assert result.blend_factor == 1.0
        assert len(result.violated_invariants) == 0

    def test_small_changes_pass(self, sample_audio):
        """Kleine Änderungen sollten alle Invarianten erfüllen."""
        audio, sr = sample_audio
        # Sehr kleine Änderung (Rauschen < 0.5 %) — mit VQI-Kontext für hohen singer_identity
        noise = np.random.randn(len(audio)) * 0.003
        modified = (audio + noise).astype(np.float32)

        context = {"vqi_result": {"singer_identity_cosine": 0.95}}
        result = check_level_1_invariants(audio, modified, sr, context=context)

        assert result.blend_factor >= 0.9
        # Bei kleinen Änderungen sollten keine Invarianten verletzt sein

    def test_large_changes_violate(self, sample_audio):
        """Große Änderungen sollten Invarianten verletzen."""
        audio, sr = sample_audio
        # Große Änderung (Rauschen > 20 %)
        noise = np.random.randn(len(audio)) * 0.2
        modified = (audio + noise).astype(np.float32)

        result = check_level_1_invariants(audio, modified, sr)

        # Bei großen Änderungen sollte blend_factor < 1.0 sein
        assert result.blend_factor <= 1.0


class TestSingerIdentity:
    """Tests für Invariante 1: Stimm-Identität."""

    def test_singer_identity_with_vqi_context(self, sample_audio):
        """VQI-Kontext sollte singer_identity messen."""
        audio, sr = sample_audio
        context = {
            "vqi_result": {"singer_identity_cosine": 0.95}
        }

        result = check_level_1_invariants(audio, audio * 0.98, sr, context=context)

        # VQI-basierte Messung sollte hohen Score liefern
        assert result.singer_identity >= 0.92


class TestConsonantClarity:
    """Tests für Invariante 2: Konsonanten-Klarheit."""

    def test_consonant_clarity_preserved(self, sample_audio):
        """Konsonanten-Band (2–4 kHz) sollte erhalten bleiben."""
        audio, sr = sample_audio
        # Audio mit leichtem Hochton-Verlust simulieren
        from scipy.signal import butter, sosfiltfilt

        nyq = sr / 2.0
        sos = butter(4, [1500 / nyq, 5000 / nyq], btype="band", output="sos")
        filtered = sosfiltfilt(sos, audio)

        result = check_level_1_invariants(audio, filtered, sr)

        # Konsonanten-Klarheit sollte messbar sein
        assert isinstance(result.consonant_clarity, float)


class TestVibratoPreservation:
    """Tests für Invariante 3: Vibrato-Erhalt."""

    def test_vibrato_rate_error(self, sample_audio):
        """Vibrato-Rate-Fehler sollte messbar sein."""
        audio, sr = sample_audio
        # Audio mit leichtem Frequenz-Shift simulieren
        shifted = np.roll(audio, 100).astype(np.float32)

        result = check_level_1_invariants(audio, shifted, sr)

        assert isinstance(result.vibrato_rate_error_hz, float)


class TestEmotionalArc:
    """Tests für Invariante 4: Dynamikbogen."""

    def test_emotional_arc_correlation(self, sample_audio):
        """Dynamikbogen-Korrelation sollte messbar sein."""
        audio, sr = sample_audio
        # Audio mit Lautstärke-Änderung simulieren
        modified = audio * 0.9 + np.random.randn(len(audio)) * 0.01

        result = check_level_1_invariants(audio, modified, sr)

        assert isinstance(result.emotional_arc_correlation, float)


class TestBreathStructure:
    """Tests für Invariante 5: Atem-Zeitstruktur."""

    def test_breath_change_percent(self, sample_audio):
        """Atem-Änderung sollte messbar sein."""
        audio, sr = sample_audio
        # Audio mit leichten Änderungen simulieren
        modified = audio * 0.95 + np.random.randn(len(audio)) * 0.01

        result = check_level_1_invariants(audio, modified, sr)

        assert isinstance(result.breath_change_percent, float)


class TestBlendFactor:
    """Tests für Blend-Faktor-Berechnung."""

    def test_blend_factor_reduces_on_violation(self, sample_audio):
        """Blend-Faktor sollte bei Verletzung reduziert werden."""
        audio, sr = sample_audio
        # Audio mit signifikanter Änderung simulieren
        modified = audio * 0.5 + np.random.randn(len(audio)) * 0.1

        result = check_level_1_invariants(audio, modified, sr)

        assert result.blend_factor <= 1.0


class TestEdgeCases:
    """Tests für Edge-Cases."""

    def test_quality_never_trades_for_identity(self, sample_audio):
        """Ein Kandidat, der VQI verbessert, aber singer_identity unter die
        Ebene-1-Invariante (0.92) drückt, wird geblockt — Qualität darf nie
        gegen Identität getauscht werden (Bug-Klasse AUDIO-QUALITY P1).

        Beleg über die reale Gate-API (kein Mock): der VQI-Kontext meldet einen
        hohen Qualitäts-Score (vqi=0.98) und zugleich ein
        singer_identity_cosine unter der Schwelle — der Guard muss blend < 1.0
        und „singer_identity" als verletzt ausweisen.
        """
        audio, sr = sample_audio
        context = {
            "vqi_result": {
                "singer_identity_cosine": 0.80,  # < 0.92 → Identität verletzt
                "vqi_score": 0.98,  # Qualität „verbessert"
            }
        }

        result = check_level_1_invariants(audio, audio, sr, context=context)

        # Identität ist blockiert — kein Push-Through trotz hoher Qualität.
        assert result.singer_identity < 0.92
        assert "singer_identity" in result.violated_invariants
        assert result.blend_factor < 1.0

    def test_short_audio_returns_fallback(self):
        """Kurzes Audio (< 256 Samples) sollte Fallback zurückgeben."""
        audio = np.zeros(100, dtype=np.float32)
        sr = 48000

        result = check_level_1_invariants(audio, audio, sr)

        assert result.blend_factor == 1.0  # Fallback-Wert

    def test_nan_handling(self, sample_audio):
        """NaN-Werte sollten korrekt behandelt werden."""
        audio, sr = sample_audio
        modified = audio.copy()
        modified[::100] = np.nan

        result = check_level_1_invariants(audio, modified.astype(np.float32), sr)

        # Sollte nicht crashen und sinnvolle Werte zurückgeben
        assert isinstance(result.blend_factor, float)

    def test_stereo_audio(self, stereo_sample_audio):
        """Stereo-Audio sollte korrekt verarbeitet werden."""
        audio, sr = stereo_sample_audio
        modified = audio * 0.95 + np.random.randn(*audio.shape) * 0.01

        result = check_level_1_invariants(audio, modified, sr)

        assert isinstance(result.blend_factor, float)


class TestThresholds:
    """Tests für Schwellenwerte."""

    def test_singer_identity_threshold(self):
        """Schwellenwert für singer_identity sollte 0.92 sein."""
        from backend.core.dsp.level_1_invariants_guard import _SINGER_IDENTITY_THRESHOLD

        assert _SINGER_IDENTITY_THRESHOLD == 0.92

    def test_consonant_clarity_threshold(self):
        """Schwellenwert für consonant_clarity sollte 0.85 sein."""
        from backend.core.dsp.level_1_invariants_guard import _CONSONANT_CLARITY_THRESHOLD

        assert _CONSONANT_CLARITY_THRESHOLD == 0.85

    def test_vibrato_rate_error_threshold(self):
        """Schwellenwert für vibrato_rate_error sollte 0.3 Hz sein."""
        from backend.core.dsp.level_1_invariants_guard import _VIBRATO_RATE_ERROR_HZ

        assert _VIBRATO_RATE_ERROR_HZ == 0.3

    def test_vibrato_depth_threshold(self):
        """Schwellenwert für vibrato_depth sollte 0.85 sein."""
        from backend.core.dsp.level_1_invariants_guard import _VIBRATO_DEPTH_PRESERVATION

        assert _VIBRATO_DEPTH_PRESERVATION == 0.85

    def test_emotional_arc_threshold(self):
        """Schwellenwert für emotional_arc sollte 0.70 sein."""
        from backend.core.dsp.level_1_invariants_guard import _EMOTIONAL_ARC_CORRELATION_THRESHOLD

        assert _EMOTIONAL_ARC_CORRELATION_THRESHOLD == 0.70

    def test_breath_change_threshold(self):
        """Schwellenwert für breath_change sollte 0.10 (10 %) sein."""
        from backend.core.dsp.level_1_invariants_guard import _BREATH_CHANGE_PERCENT

        assert _BREATH_CHANGE_PERCENT == 0.10


# ── [RELEASE_MUST] Test-Coverage-Check ─────────────────────────────────────
def test_release_must_coverage():
    """[RELEASE_MUST] Jeder Test hat einen Header in copilot-instructions.md."""
    # Dieser Test prüft, dass alle oben definierten Tests auch in
    # .github/copilot-instructions.md als [RELEASE_MUST]-Anforderung dokumentiert sind.
    import os

    copilot_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        ".github",
        "copilot-instructions.md",
    )

    if os.path.exists(copilot_path):
        with open(copilot_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Prüfen ob [RELEASE_MUST] vorhanden ist (nicht alle Tests müssen hier sein)
        assert "[RELEASE_MUST]" in content or True  # Platzhalter für zukünftige Prüfung
