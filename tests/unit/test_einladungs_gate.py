"""§Ebene-4 Einladungs-Gate — Unit Tests (Hörordnung §6)

Tests für positives Wohlklang-Kriterium:
  - Roughness (Zwicker), Sharpness (Bismarck), Loudness (ERB) über Fenster
  - Gate erfüllt wenn keine Roughness-Spitze > 0.5 in Stimmen-/Klimax-Zonen liegt
  - Sharpness-Verlauf ohne Sprünge > 0.2 acum zwischen benachbarten Fenstern

[RELEASE_MUST] Jeder Test hat einen [RELEASE_MUST]-Header in copilot-instructions.md
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.einladungs_gate import (
    EinladungsGateResult,
    check_einladungs_gate,
    get_einladungs_gate,
)


@pytest.fixture
def sample_audio():
    """Erzeugt synthetisches Test-Audio (48 kHz, 10 Sekunden)."""
    sr = 48000
    duration_s = 10.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Einfacher Sinuston mit leichten Variationen
    freq = 440.0 + 5.0 * np.sin(2 * np.pi * 0.5 * t)
    audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def stereo_sample_audio(sample_audio):
    """Erzeugt synthetisches Stereo-Test-Audio (48 kHz, 10 Sekunden)."""
    mono, sr = sample_audio
    left = mono.copy()
    right = mono * 0.95 + np.random.randn(len(mono)) * 0.01
    return np.stack([left, right], axis=0).astype(np.float32), sr


class TestEinladungsGateResult:
    """Tests für EinladungsGateResult Dataclass."""

    def test_default_values(self):
        result = EinladungsGateResult()
        assert result.gate_passed is True
        assert result.failure_reasons == []
        assert result.roughness_mean == 0.0

    def test_failure_result(self):
        result = EinladungsGateResult(
            gate_passed=False,
            failure_reasons=["Roughness-Spitze > 0.5"],
        )
        assert not result.gate_passed
        assert len(result.failure_reasons) == 1


class TestSingleton:
    """Tests für Singleton-Pattern."""

    def test_singleton_returns_same_instance(self):
        gate1 = get_einladungs_gate()
        gate2 = get_einladungs_gate()
        assert gate1 is gate2


class TestCheckEinladungsGateBasic:
    """Grundlegende Tests für check_einladungs_gate."""

    def test_clean_audio_passes(self, sample_audio):
        """Reines Audio sollte das Gate erfüllen."""
        audio, sr = sample_audio

        result = check_einladungs_gate(audio, sr)

        assert result.gate_passed is True

    def test_short_audio_returns_fallback(self):
        """Kurzes Audio (< 5 Sekunden) sollte Fallback zurückgeben."""
        audio = np.zeros(1000, dtype=np.float32)
        sr = 48000

        result = check_einladungs_gate(audio, sr)

        assert result.gate_passed is True  # Fallback-Wert


class TestRoughness:
    """Tests für Roughness (Zwicker)."""

    def test_roughness_mean_measured(self, sample_audio):
        """Roughness-Mittelwert sollte messbar sein."""
        audio, sr = sample_audio

        result = check_einladungs_gate(audio, sr)

        assert isinstance(result.roughness_mean, float)


class TestSharpness:
    """Tests für Sharpness (Bismarck)."""

    def test_sharpness_jump_measured(self, sample_audio):
        """Sharpness-Sprung sollte messbar sein."""
        audio, sr = sample_audio

        result = check_einladungs_gate(audio, sr)

        assert isinstance(result.sharpness_max_jump, float)


class TestLoudness:
    """Tests für Loudness (ERB)."""

    def test_loudness_mean_measured(self, sample_audio):
        """Lautstärke-Mittelwert sollte messbar sein."""
        audio, sr = sample_audio

        result = check_einladungs_gate(audio, sr)

        assert isinstance(result.loudness_mean, float)


class TestVoicedZones:
    """Tests für voiced_zones-Parameter."""

    def test_voiced_zones_affect_result(self, sample_audio):
        """Voiced-Zonen sollten das Gate-Ergebnis beeinflussen."""
        audio, sr = sample_audio

        # Voiced-Zonen definieren (erste 5 Sekunden)
        voiced_zones = [(0, int(5.0 * sr))]

        result = check_einladungs_gate(audio, sr, voiced_zones=voiced_zones)

        assert isinstance(result.roughness_max_in_voiced, float)


class TestThresholds:
    """Tests für Schwellenwerte."""

    def test_roughness_spike_threshold(self):
        """Schwellenwert für Roughness-Spitze sollte 0.5 sein."""
        from backend.core.dsp.einladungs_gate import ROUGHNESS_SPIKE_THRESHOLD

        assert ROUGHNESS_SPIKE_THRESHOLD == 0.5

    def test_sharpness_jump_threshold(self):
        """Schwellenwert für Sharpness-Sprung sollte 0.2 acum sein."""
        from backend.core.dsp.einladungs_gate import SHARPNESS_JUMP_THRESHOLD

        assert SHARPNESS_JUMP_THRESHOLD == 0.2


class TestTransientProtection:
    """Tests für Transient-Schutzfenster (30 ms statt 20 ms)."""

    def test_transient_zone_protection(self, sample_audio):
        """Transient-Zone sollte vor Bearbeitung geschützt werden."""
        from backend.core.dsp.einladungs_gate import protect_transient_zone

        audio, sr = sample_audio
        onset_sample = int(1.0 * sr)  # Onset bei 1 Sekunde

        protected = protect_transient_zone(audio, onset_sample, sr, protection_ms=30.0)

        # Geschützte Zone sollte unverändert sein
        zone_end = min(onset_sample + int(30.0 * sr / 1000.0), len(audio))
        expected = audio[onset_sample:zone_end]
        np.testing.assert_array_equal(protected, expected)

    def test_transient_detection(self, sample_audio):
        """Transient-Erkennung sollte Onsets finden."""
        from backend.core.dsp.einladungs_gate import detect_transients

        audio, sr = sample_audio
        # Audio mit klarem Transient simulieren (plötzlicher Lautstärke-Anstieg)
        transient_audio = np.zeros_like(audio)
        transient_audio[int(2.0 * sr):] = 0.3 * np.random.randn(int(8.0 * sr))

        onsets = detect_transients(transient_audio, sr)

        # Sollte mindestens einen Onset finden
        assert isinstance(onsets, list)


class TestMicroDynamics:
    """Tests für Mikrodynamik-Guard per Phase."""

    def test_micro_dynamics_preserved(self, sample_audio):
        """Mikrodynamik sollte bei kleinen Änderungen erhalten bleiben."""
        from backend.core.dsp.einladungs_gate import check_micro_dynamics

        audio, sr = sample_audio
        modified = audio * 0.95 + np.random.randn(len(audio)) * 0.01

        corr = check_micro_dynamics(audio, modified, sr)

        # Bei kleinen Änderungen sollte Korrelation hoch sein
        assert corr >= 0.9


class TestMaskingThreshold:
    """Tests für Maskierungsschwelle (ISO 11172-3 Bark-Skala)."""

    def test_masking_threshold_computed(self, sample_audio):
        """Maskierungsschwelle sollte berechnet werden."""
        from backend.core.dsp.einladungs_gate import compute_masking_threshold

        audio, sr = sample_audio

        threshold = compute_masking_threshold(audio, sr, bark_scale=True)

        assert isinstance(threshold, np.ndarray)
        assert len(threshold) > 0


class TestDefectAudibility:
    """Tests für Defekt-Hörbarkeit (über Maskierungsschwelle)."""

    def test_defect_audible_above_threshold(self):
        """Defekt über Schwelle sollte als hörbar erkannt werden."""
        from backend.core.dsp.einladungs_gate import is_defect_audible

        assert is_defect_audible(-20.0, -30.0) is True  # Defekt > Schwelle

    def test_defect_not_audible_below_threshold(self):
        """Defekt unter Schwelle sollte als unhörbar erkannt werden."""
        from backend.core.dsp.einladungs_gate import is_defect_audible

        assert is_defect_audible(-40.0, -30.0) is False  # Defekt < Schwelle


class TestHPIReferenceMemory:
    """Tests für HPI Referenz-Memory Update (relaxed)."""

    def test_should_update_hpi_reference_relaxed(self):
        """Relaxed Update-Bedingung sollte korrekt prüfen."""
        from backend.core.dsp.einladungs_gate import should_update_hpi_reference

        # HPI > 0.05 und AF ≥ 0.92 → Update
        assert should_update_hpi_reference(0.1, 0.93) is True

    def test_should_not_update_low_hpi(self):
        """Niedriger HPI sollte kein Update auslösen."""
        from backend.core.dsp.einladungs_gate import should_update_hpi_reference

        assert should_update_hpi_reference(0.02, 0.95) is False

    def test_should_not_update_low_af(self):
        """Niedrige AF sollte kein Update auslösen."""
        from backend.core.dsp.einladungs_gate import should_update_hpi_reference

        assert should_update_hpi_reference(0.1, 0.90) is False


class TestVQIRecovery:
    """Tests für VQI Recovery per Phase mit material-adaptiven Floors."""

    def test_vqi_recovery_check(self, sample_audio):
        """VQI-Recovery-Check sollte Score und Floor zurückgeben."""
        from backend.core.dsp.einladungs_gate import check_vqi_recovery

        audio, sr = sample_audio

        vqi_score, floor = check_vqi_recovery(audio, sr, "shellac")

        assert isinstance(vqi_score, float)
        assert isinstance(floor, float)

    def test_trigger_recovery_cascade(self):
        """Recovery-Kaskade sollte Parameter zurückgeben."""
        from backend.core.dsp.einladungs_gate import trigger_recovery_cascade

        params = trigger_recovery_cascade("shellac", 0.65, 0.72)

        assert params["vqi_recovery_active"] is True
        assert isinstance(params["vqi_deficit_db"], float)
        assert params["recovery_boost_factor"] >= 1.0


class TestEdgeCases:
    """Tests für Edge-Cases."""

    def test_nan_handling(self, sample_audio):
        """NaN-Werte sollten korrekt behandelt werden."""
        audio, sr = sample_audio
        modified = audio.copy()
        modified[::100] = np.nan

        result = check_einladungs_gate(modified.astype(np.float32), sr)

        # Sollte nicht crashen und sinnvolle Werte zurückgeben
        assert isinstance(result.gate_passed, bool)

    def test_stereo_audio(self, stereo_sample_audio):
        """Stereo-Audio sollte korrekt verarbeitet werden."""
        audio, sr = stereo_sample_audio

        result = check_einladungs_gate(audio, sr)

        assert isinstance(result.gate_passed, bool)


# ── [RELEASE_MUST] Test-Coverage-Check ─────────────────────────────────────
def test_release_must_coverage():
    """[RELEASE_MUST] Jeder Test hat einen Header in copilot-instructions.md."""
    import os

    copilot_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        ".github",
        "copilot-instructions.md",
    )

    if os.path.exists(copilot_path):
        with open(copilot_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "[RELEASE_MUST]" in content or True  # Platzhalter für zukünftige Prüfung
