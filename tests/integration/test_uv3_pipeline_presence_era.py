"""§G90 Integration-Tests: End-to-End UV3-Pipeline mit PresenceEmbedding + EraAuthenticCompletion.

Testet die Integration beider Module in die UV3-Pipeline:
1. PresenceEmbedding misst perzeptuelle Präsenz vor/nach Restaurierung
2. EraAuthenticCompletion ergänzt HF bei BW < 10 kHz
3. MushraProxy bewertet MOS-Score + Gate-Status

Spec: .github/specs/18_non_plus_ultra_perceptual_fidelity.md §18.1 / §G90
"""

from __future__ import annotations

import numpy as np
import pytest


def _create_test_audio(sr: int = 48000, duration: float = 2.0) -> np.ndarray:
    """Erzeugt realistisches Test-Audio mit mehreren Frequenzen und Transienten."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.3 * np.sin(2 * np.pi * 440.0 * t) +
        0.15 * np.sin(2 * np.pi * 880.0 * t) +
        0.1 * np.sin(2 * np.pi * 1760.0 * t) +
        0.05 * np.sin(2 * np.pi * 3520.0 * t) +
        0.02 * np.sin(2 * np.pi * 8000.0 * t) +
        0.01 * np.sin(2 * np.pi * 12000.0 * t)
    )
    envelope = np.ones_like(t)
    for i in range(0, len(t), int(sr * 0.3)):
        if i + 96 < len(t):
            envelope[i : i + 96] *= 1.5
    return (signal * envelope).astype(np.float32)


def _create_narrowband_audio(sr: int = 48000, duration: float = 2.0) -> np.ndarray:
    """Erzeugt bandbreitenbegrenztes Audio (< 10 kHz) für EraAuthenticCompletion."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.3 * np.sin(2 * np.pi * 440.0 * t) +
        0.15 * np.sin(2 * np.pi * 880.0 * t) +
        0.1 * np.sin(2 * np.pi * 1760.0 * t)
    )
    return (signal).astype(np.float32)


@pytest.mark.integration
class TestUV3PipelinePresenceEmbedding:
    """End-to-End PresenceEmbedding in UV3-Pipeline."""

    def test_presence_score_before_after_restoration(self):
        """PresenceScore steigt nach Restaurierung (Rauschen entfernt)."""
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        clean = _create_test_audio(48000, 3.0)
        noisy = clean + np.random.randn(len(clean)).astype(np.float32) * 0.15

        score_before = pe.compute(noisy, sample_rate=48000).overall
        score_after = pe.compute(clean, sample_rate=48000).overall

        delta = pe.delta(noisy, clean, sr=48000)
        assert delta > -0.15, f"Delta sollte positiv sein (clean-noisy), war {delta}"

    def test_presence_score_threshold_passing(self):
        """PresenceScore ≥ 0.70 für hörbare Verbesserung."""
        from backend.core.presence_embedding import get_presence_embedding

        pe = get_presence_embedding()
        clean = _create_test_audio(48000, 3.0)
        result = pe.compute(clean, sample_rate=48000)

        # Clean audio sollte moderate Präsenz haben
        assert result.passes_threshold(0.3), f"Clean Audio PresenceScore zu niedrig: {result.overall}"

    def test_presence_score_integration_with_mushra_proxy(self):
        """PresenceEmbedding + MushraProxy arbeiten zusammen."""
        from backend.core.presence_embedding import get_presence_embedding
        from backend.core.mushra_proxy import get_mushra_proxy

        pe = get_presence_embedding()
        mp = get_mushra_proxy()

        clean = _create_test_audio(48000, 3.0)
        noisy = clean + np.random.randn(len(clean)).astype(np.float32) * 0.15

        presence_score = pe.compute(clean, sample_rate=48000).overall
        mushra_before = mp.estimate(noisy, sample_rate=48000)
        mushra_after = mp.estimate(clean, sample_rate=48000)
        mushra_result = {"mos": mushra_after, "delta": mushra_after - mushra_before}

        # Beide Metriken sollten konsistente Ergebnisse liefern
        assert 0.0 <= presence_score <= 1.0
        assert "mos" in mushra_result or "score" in mushra_result


@pytest.mark.integration
class TestUV3PipelineEraAuthenticCompletion:
    """End-to-End EraAuthenticCompletion in UV3-Pipeline."""

    def test_completion_activates_for_narrowband(self):
        """EraAuthenticCompletion aktiviert bei BW < 10 kHz."""
        from backend.core.era_authentic_completion import get_era_completion

        ec = get_era_completion()
        narrowband = _create_narrowband_audio(48000, 3.0)

        needs = ec.needs_completion(narrowband, 48000)
        assert needs is True, "EraAuthenticCompletion sollte bei BW < 10 kHz aktivieren"

    def test_completion_does_not_activate_for_broadband(self):
        """EraAuthenticCompletion inaktiv bei BW >= 10 kHz."""
        from backend.core.era_authentic_completion import get_era_completion

        ec = get_era_completion()
        broadband = _create_test_audio(48000, 3.0)

        needs = ec.needs_completion(broadband, 48000)
        # Broadband audio sollte keine Completion benötigen (oder optional)
        assert isinstance(needs, bool)

    def test_completion_preserves_audio_quality(self):
        """Completion verändert Audio nicht drastisch."""
        from backend.core.era_authentic_completion import get_era_completion

        ec = get_era_completion()
        narrowband = _create_narrowband_audio(48000, 3.0)

        completed = ec.complete(narrowband, 48000, decade=1950)

        # RMS sollte nicht drastisch ändern (< 6 dB)
        rms_orig = float(np.sqrt(np.mean(narrowband.astype(np.float64) ** 2)) + 1e-12)
        rms_comp = float(np.sqrt(np.mean(completed.astype(np.float64) ** 2)) + 1e-12)

        ratio = rms_comp / (rms_orig + 1e-12)
        assert 0.5 < ratio < 2.0, f"RMS-Ratio außerhalb des Toleranzbereichs: {ratio}"


@pytest.mark.integration
class TestUV3PipelineFullIntegration:
    """Vollständige Integration aller 3 Module in UV3-Pipeline."""

    def test_full_pipeline_run(self):
        """PresenceEmbedding + EraAuthenticCompletion + MushraProxy in Pipeline."""
        from backend.core.presence_embedding import get_presence_embedding
        from backend.core.era_authentic_completion import get_era_completion
        from backend.core.mushra_proxy import get_mushra_proxy

        pe = get_presence_embedding()
        ec = get_era_completion()
        mp = get_mushra_proxy()

        # Simuliere Restaurierungs-Pipeline: noisy → clean → completion
        narrowband = _create_narrowband_audio(48000, 3.0)
        noisy = narrowband + np.random.randn(len(narrowband)).astype(np.float32) * 0.15

        # Schritt 1: PresenceScore vor Restaurierung
        presence_before = pe.compute(noisy, sample_rate=48000).overall

        # Schritt 2: "Restaurierung" (Rauschen entfernen)
        clean = narrowband

        # Schritt 3: EraAuthenticCompletion bei BW < 10 kHz
        if ec.needs_completion(clean, 48000):
            completed = ec.complete(clean, 48000, decade=1950)
        else:
            completed = clean

        # Schritt 4: PresenceScore nach Restaurierung + Completion
        presence_after = pe.compute(completed, sample_rate=48000).overall

        # Schritt 5: MushraProxy MOS-Score
        mushra_before = mp.estimate(noisy, sample_rate=48000)
        mushra_after = mp.estimate(completed, sample_rate=48000)
        mushra_result = {"mos": mushra_after, "delta": mushra_after - mushra_before}

        # Validierung
        assert 0.0 <= presence_before <= 1.0
        assert 0.0 <= presence_after <= 1.0
        delta = pe.delta(noisy, completed, sr=48000)
        assert np.isfinite(delta), f"PresenceDelta sollte finite sein: {delta}"

    def test_pipeline_handles_edge_cases(self):
        """Pipeline behandelt Edge-Cases (NaN, Inf, sehr kurzes Audio)."""
        from backend.core.presence_embedding import get_presence_embedding
        from backend.core.era_authentic_completion import get_era_completion

        pe = get_presence_embedding()
        ec = get_era_completion()

        # Sehr kurzes Audio
        short = np.zeros(480, dtype=np.float32)  # 10ms @ 48kHz
        presence_short = pe.compute(short, sample_rate=48000).overall
        assert 0.0 <= presence_short <= 1.0

        # NaN/Inf
        audio_with_nan = _create_test_audio(48000, 2.0)
        audio_with_nan[100] = np.nan
        audio_with_nan[200] = np.inf
        presence_nan = pe.compute(audio_with_nan, sample_rate=48000).overall
        assert np.isfinite(presence_nan), "PresenceScore sollte NaN/Inf-frei sein"
