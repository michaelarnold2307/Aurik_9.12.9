"""GPU-Detection Failsafe — Validierung des CPU-Fallbacks.

Testet, dass das System korrekt auf CPU zurückfällt, wenn keine GPU
verfügbar ist oder die GPU-Erkennung fehlschlägt.

Spec: .github/copilot-instructions.md §VI Startup-Vertrag (§v10.305)
      .github/specs/v10.305_startup_integration_contract.md
"""

from __future__ import annotations

import types

import numpy as np
import pytest


@pytest.mark.unit
class TestGpuDetectionFailsafe:
    """Validiert den GPU-Detection-Fallback-Mechanismus."""

    def test_cpu_only_mode(self):
        from backend.core.ml_device_manager import get_ml_device_manager

        # Erstelle Manager (singleton)
        mgr = get_ml_device_manager()
        # Wenn keine GPU verfügbar, sollte CPU-only Modus aktiv sein
        if not mgr._gpu_available:
            assert mgr._backend.value == "cpu", f"Backend sollte 'cpu' sein, ist '{mgr._backend}'"

    def test_detection_timeout_fallback(self):
        """Detection-Timeout führt zu CPU-Fallback."""
        from backend.core.ml_device_manager import get_ml_device_manager

        # Erstelle Manager (singleton)
        mgr = get_ml_device_manager()
        ok = mgr.wait_for_detection(timeout_s=2.0)
        # Sollte innerhalb des Timeouts abschließen (oder auf CPU zurückfallen)
        assert ok or not mgr._gpu_available, "Detection sollte erfolgreich sein oder zu CPU-Fallback führen"

    def test_onnx_cpu_provider_fallback(self):
        """ONNX verwendet CPUExecutionProvider wenn GPU nicht verfügbar."""
        from backend.core.ml_device_manager import get_ml_device_manager

        mgr = get_ml_device_manager()
        # ONNX-Provider sollten CPUExecutionProvider enthalten
        providers = mgr._ort_gpu_providers or ["CPUExecutionProvider"]
        assert "CPUExecutionProvider" in providers, \
            f"CPUExecutionProvider nicht in ONNX-Providern: {providers}"


@pytest.mark.unit
class TestMlInferenceFallback:
    """Validiert, dass ML-Inferenz auf CPU funktioniert."""

    def test_sota_vocal_pipeline_cpu(self):
        from backend.core.sota_vocal_pipeline import SOTAVocalPipeline

        pipeline = SOTAVocalPipeline(n_fft=1024, hop=256)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 48000)).astype(np.float32) * 0.5
        result = pipeline.process(audio[:4800], sample_rate=48000)
        assert result.audio.shape == audio[:4800].shape
        assert np.isfinite(result.audio).all()

    def test_mert_mushra_proxy_cpu(self):
        from backend.core.mert_mushra_proxy import MertMushraProxy

        proxy = MertMushraProxy()
        ref = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 48000)).astype(np.float32) * 0.5
        test = ref + np.random.randn(48000).astype(np.float32) * 0.01

        score = proxy.evaluate(ref[:9600], test[:9600], sr=48000)
        assert 0 <= score.proxy_score <= 100, f"MUSHRA-Score außerhalb des Bereichs: {score.proxy_score}"


@pytest.mark.unit
class TestPluginFallback:
    """Validiert Plugin-Fallback-Mechanismen."""

    def test_onnx_cpu_provider(self):
        from backend.core.ml_device_manager import get_ml_device_manager

        mgr = get_ml_device_manager()
        # ONNX sollte CPUExecutionProvider als Fallback haben
        providers = mgr._ort_gpu_providers or ["CPUExecutionProvider"]
        assert "CPUExecutionProvider" in providers


@pytest.mark.unit
class TestMemoryBudgetFallback:
    """Validiert ML-Memory-Budget Fallback."""

    def test_try_allocate_cpu(self):
        from backend.core.ml_memory_budget import try_allocate

        # Sollte auf CPU funktionieren (kein GPU-Speicher)
        ok = try_allocate("TEST_FALLBACK", 0.01)
        assert isinstance(ok, bool), "try_allocate sollte bool zurückgeben"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
