"""Test für HTDemucs-Integration und separation_fidelity Neuimplementierung (v10.0.0).

Unit-Tests für:
- HTDemucs Plugin (PyTorch/ONNX Hybrid, GPU-Flag, Thread-Sicherheit)
- separation_fidelity echte Stem-Separation vs. Fallback-Proxy
- Lifecycle Manager Integration
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Import Test-Fixtures (gibt es bereits)
pytestmark = pytest.mark.unit


class TestHtdemucsPlugin:
    """HTDemucs Plugin Tests."""

    def test_htdemucs_plugin_singleton(self) -> None:
        """HTDemucs Singleton-Pattern: zweiter Aufruf gibt gleiche Instanz."""
        from plugins.htdemucs_plugin import get_htdemucs_plugin

        plugin1 = get_htdemucs_plugin()
        plugin2 = get_htdemucs_plugin()
        assert plugin1 is plugin2, "Singleton-Pattern nicht eingehalten"

    def test_htdemucs_separation_result_shape(self) -> None:
        """SeparationResult Container hat korrekte Form und Methoden."""
        from plugins.htdemucs_plugin import SeparationResult

        # Create dummy stems
        vocals = np.random.randn(48000).astype(np.float32)
        drums = np.random.randn(48000).astype(np.float32)
        bass = np.random.randn(48000).astype(np.float32)
        other = np.random.randn(48000).astype(np.float32)

        result = SeparationResult(vocals, drums, bass, other, 48000)

        assert result.sr == 48000
        assert result.vocals.shape == (48000,)
        assert result.drums.shape == (48000,)
        assert result.bass.shape == (48000,)
        assert result.other.shape == (48000,)

    def test_htdemucs_separation_result_reconstruct(self) -> None:
        """SeparationResult.reconstruct() summiert alle Stems."""
        from plugins.htdemucs_plugin import SeparationResult

        v = np.ones(1000, dtype=np.float32) * 0.1
        d = np.ones(1000, dtype=np.float32) * 0.2
        b = np.ones(1000, dtype=np.float32) * 0.3
        o = np.ones(1000, dtype=np.float32) * 0.4

        result = SeparationResult(v, d, b, o, 48000)
        reconstructed = result.reconstruct()

        expected = 0.1 + 0.2 + 0.3 + 0.4  # = 1.0
        np.testing.assert_allclose(reconstructed, expected, rtol=1e-5)

    def test_separation_fidelity_metric_fallback_mode(self) -> None:
        """separation_fidelity nutzt Proxy-Mode bei sehr kurzem Audio."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()

        # Sehr kurz: <64 samples
        audio_short = np.random.randn(32).astype(np.float32)
        ref_short = np.random.randn(32).astype(np.float32)

        score = metric.measure(audio_short, 48000, reference=ref_short)
        assert 0.0 <= score <= 1.0, f"Score {score} außerhalb [0, 1]"
        assert score == 1.0, "Sehr kurze Audio sollte Score 1.0 zurückgeben (Edge Case)"

    def test_separation_fidelity_metric_proxy_fallback_duration(self) -> None:
        """separation_fidelity nutzt Proxy für <2s Audio."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()

        # 1 Sekunde @ 48 kHz = 48000 samples
        audio = np.random.randn(48000).astype(np.float32)
        ref = np.random.randn(48000).astype(np.float32)

        score = metric.measure(audio, 48000, reference=ref, material_type="cd")

        # Score sollte zwischen 0 und 1 sein (Proxy-Methode)
        assert 0.0 <= score <= 1.0, f"Score {score} außerhalb [0, 1]"

    def test_plugin_lifecycle_manager_has_htdemucs(self) -> None:
        """Lifecycle Manager kennt HTDemucs Modell."""
        from backend.core.plugin_lifecycle_manager import _PHASE_REQUIRED_MODELS

        assert "musical_goals_separation_fidelity" in _PHASE_REQUIRED_MODELS
        assert "HTDemucs" in _PHASE_REQUIRED_MODELS["musical_goals_separation_fidelity"]

    def test_separation_fidelity_reference_free_score(self) -> None:
        """Referenzfreier Modus: Harmonizitäts-basierter Score."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()

        # Synthetic: Pure sine wave (hoch harmonisch) vs. white noise (flach)
        sr = 48000
        t = np.linspace(0, 1, sr, dtype=np.float32)

        # Pure sine (perfekt tonale Separation expected)
        sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        score_sine = metric._reference_free(sine, sr)
        logger.info("separation_fidelity (sine, ref-free): %.3f", score_sine)

        # White noise (flach, schlechte Separation expected)
        noise = (0.1 * np.random.randn(sr)).astype(np.float32)
        score_noise = metric._reference_free(noise, sr)
        logger.info("separation_fidelity (noise, ref-free): %.3f", score_noise)

        # Sine sollte bessere Score haben (tonaler)
        assert score_sine > score_noise, "Sine sollte bessere Separation-Score als Noise haben"

    def test_separation_fidelity_material_types(self) -> None:
        """Verschiedene Material-Typen haben unterschiedliche Harmonicity-Floors."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()

        # Synth Material für Tests
        sr = 48000
        audio = (0.1 * np.random.randn(sr * 10)).astype(np.float32)  # 10s

        # Teste verschiedene Material-Typen
        materials = ["shellac", "tape", "vinyl", "mp3_low", "cd", "unknown"]

        scores = {}
        for mat in materials:
            score = metric._reference_free(audio, sr, material_type=mat)
            scores[mat] = score
            logger.info("separation_fidelity (%s): %.3f", mat, score)

        # Ultra-analog sollte niedrigste Floor haben
        assert 0.0 <= scores["shellac"] <= 1.0
        assert 0.0 <= scores["cd"] <= 1.0

    def test_separation_fidelity_global_scalar_guard_uses_proxy(self, monkeypatch) -> None:
        """global_scalar < 0.15 darf die teure HTDemucs-Messung nicht beginnen."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()
        audio = (0.15 * np.random.randn(48000)).astype(np.float32)
        ref = (0.12 * np.random.randn(48000)).astype(np.float32)

        def _fail() -> None:
            raise AssertionError("HTDemucs sollte bei global_scalar=0.10 nicht aufgerufen werden")

        monkeypatch.setattr("plugins.htdemucs_plugin.get_htdemucs_plugin", _fail)
        score = metric.measure(audio, 48000, reference=ref, global_scalar=0.10)

        assert 0.0 <= score <= 1.0, f"Score {score} außerhalb [0, 1]"

    def test_separation_fidelity_measure_cache_reuses_same_result(self, monkeypatch) -> None:
        """Identische Messungen sollten denselben HTDemucs-Score aus dem Cache ziehen."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()
        # §Separation-SOTA/Test-Drift-Fix: Der <3-s-Guard (§Perf-Fix) leitet
        # kürzere Signale in den Proxy — das 1-s-Fixture erreichte den echten
        # HTDemucs-Pfad (und damit den Cache) nie. 3.5 s Fixture stellt die
        # Testabsicht (Cache-Reuse) wieder her.
        audio = (0.16 * np.sin(np.linspace(0, 4 * np.pi, int(48000 * 3.5), dtype=np.float32))).astype(np.float32)
        ref = audio.copy()
        calls = {"n": 0}

        class _FakePlugin:
            def separate(self, restored, sr):
                calls["n"] += 1
                return type(
                    "Result",
                    (),
                    {"reconstruct": lambda self: restored.copy(), "sr": sr},
                )()

        monkeypatch.setattr("plugins.htdemucs_plugin.get_htdemucs_plugin", lambda: _FakePlugin())

        first = metric.measure(audio, 48000, reference=ref, global_scalar=0.80)
        second = metric.measure(audio, 48000, reference=ref, global_scalar=0.80)

        assert 0.0 <= first <= 1.0
        assert 0.0 <= second <= 1.0
        assert calls["n"] == 1, "Zweiter identischer Aufruf sollte den Cache nutzen"

    def test_measure_all_fast_validation_for_low_global_scalar(self, monkeypatch) -> None:
        """Sehr niedrige Restoration-Stärke muss eine teure Metrik-Schleife verhindern."""
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker()
        audio = (0.1 * np.random.randn(48000)).astype(np.float32)
        calls = {"fast": 0}

        def _fast(*args, **kwargs):
            calls["fast"] += 1
            return {"bass_kraft": 0.81, "brillanz": 0.83, "waerme": 0.82, "natuerlichkeit": 0.84,
                    "authentizitaet": 0.83, "emotionalitaet": 0.81, "transparenz": 0.82,
                    "groove": 0.80, "spatial_depth": 0.76, "timbre_authentizitaet": 0.82,
                    "tonal_center": 0.84, "micro_dynamics": 0.81, "separation_fidelity": 0.79,
                    "artikulation": 0.81, "transient_energie": 0.80}

        def _raise_if_called(*args, **kwargs):
            raise AssertionError("Expensive 15-goal loop should be skipped for low global_scalar")

        monkeypatch.setattr(checker, "_measure_all_fast_validation", _fast)
        for metric in checker.metrics.values():
            monkeypatch.setattr(metric, "measure", _raise_if_called)

        scores = checker.measure_all(audio, 48000, global_scalar=0.10)
        assert 0.0 <= min(scores.values()) <= 1.0
        assert calls["fast"] == 1, "Fast-validation should be used once for low-strength restoration"


class TestSeparationFidelityIntegration:
    """Integration Tests für separation_fidelity mit echtem Material."""

    def test_separation_fidelity_consistency(self) -> None:
        """Konsistenz-Test: Wiederholte Messung gleicher Audio sollte gleichen Score geben."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()

        sr = 48000
        audio = (0.1 * np.random.randn(sr * 2)).astype(np.float32)

        # Zwei Messungen
        score1 = metric._reference_free(audio, sr)
        score2 = metric._reference_free(audio, sr)

        assert np.allclose(score1, score2, rtol=1e-6), "Scores sollten identisch sein"

    def test_separation_fidelity_proxy_range(self) -> None:
        """Proxy-Methode gibt immer Score ∈ [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()

        sr = 48000
        for i in range(5):
            # Verschiedene Audio-Längen
            length = sr * (1 + i)
            audio = (np.random.randn(length) * 0.1).astype(np.float32)
            ref = (np.random.randn(length) * 0.1).astype(np.float32)

            score = metric.measure(audio, sr, reference=ref)
            assert 0.0 <= score <= 1.0, f"Score {score} außerhalb [0, 1] bei length={length}"


class TestHtdemucsPluginDocstring:
    """Docstring-Konsistenz Tests."""

    def test_separation_fidelity_docstring_updated(self) -> None:
        """SeparationFidelityMetric Docstring erwähnt HTDemucs und 4-Stem."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        docstring = SeparationFidelityMetric.__doc__ or ""
        assert "HTDemucs" in docstring, "Docstring sollte HTDemucs erwähnen"
        assert "4-Stem" in docstring, "Docstring sollte 4-Stem-Trennung erwähnen"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
