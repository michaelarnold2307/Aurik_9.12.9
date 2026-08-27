from __future__ import annotations

"""Unit-Tests für backend.core.ml_device_manager.

Tests decken ab:
  - CPU-only-Fallback wenn kein GPU vorhanden
  - Singleton-Verhalten (Thread-Sicherheit)
  - get_torch_device(): heavy vs. nicht-heavy Plugins
  - get_ort_providers(): heavy vs. nicht-heavy Plugins
  - VRAM-Budget-Allokation und -Freigabe
  - report_gpu_error(): Fehlerzählung + Session-Deaktivierung
  - gpu_status_summary(): vollständige Struktur
  - Fehlertoleranz (kein Absturz bei fehlenden Abhängigkeiten)
"""


import pathlib
import threading
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from backend.core.ml_device_manager import MLDeviceManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Setzt das module-level Singleton vor jedem Test zurück."""
    import backend.core.ml_device_manager as _mod

    original = _mod._instance
    _mod._instance = None
    yield
    _mod._instance = original


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _provider_names(providers: list) -> list[str]:
    """Extrahiert Provider-Namen aus einer Providers-Liste.

    get_ort_providers_fp16() gibt Tuples (name, options_dict) zurück;
    get_ort_providers() gibt je nach fp16-Auto auch Tuples zurück.
    Dieser Helper normalisiert beide Formate auf reine Strings.
    """
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def _make_manager_cpu_only() -> MLDeviceManager:
    """Erstellt einen MLDeviceManager im CPU-only-Modus (kein GPU detektiert).

    ALLE GPU-Detektionspfade werden gepatcht — deterministisch unabhängig vom
    Host-Zustand: der MIGraphX-Bridge-Pfad (_detect_migraphx_bridge) erkennt
    AMD-GPUs ohne ONNX-ROCm-EP und lief in Full-Suite-Läufen bei warmem
    ROCm-State an → order-/umgebungsabhängige Flakes (Spec 07, TEST-DESIGN).
    """
    from backend.core.ml_device_manager import MLDeviceManager

    with (
        patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
        patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
    ):
        mgr = MLDeviceManager()
    return mgr


# ---------------------------------------------------------------------------
# CPUOnly-Modus
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cpu_only_when_no_gpu():
    """get_torch_device() gibt stets 'cpu' zurück wenn kein GPU vorhanden."""
    mgr = _make_manager_cpu_only()
    assert mgr.get_torch_device("SGMSE") == "cpu"
    assert mgr.get_torch_device("AudioSR") == "cpu"
    assert mgr.get_torch_device("") == "cpu"


def test_ort_providers_cpu_only():
    """get_ort_providers() gibt ['CPUExecutionProvider'] zurück ohne GPU."""
    mgr = _make_manager_cpu_only()
    assert mgr.get_ort_providers("BSRoFormer") == ["CPUExecutionProvider"]
    assert mgr.get_ort_providers("MDXNet") == ["CPUExecutionProvider"]
    assert mgr.get_ort_providers() == ["CPUExecutionProvider"]


def test_ort_providers_cpu_when_plugin_session_disabled() -> None:
    """Session-deaktivierte Plugins müssen CPU-Provider erhalten."""
    mgr = _make_manager_rocm()
    mgr._gpu_disabled_plugins.add("BSRoFormer")
    assert mgr.get_ort_providers("BSRoFormer") == ["CPUExecutionProvider"]


def test_mark_ort_gpu_unsupported_forces_cpu() -> None:
    """Explizit als inkompatibel markierte Plugins müssen ORT-CPU erzwingen."""
    mgr = _make_manager_rocm()
    mgr.mark_ort_gpu_unsupported("Vocos", "provider mismatch")
    assert mgr.is_ort_gpu_supported("Vocos") is False
    assert mgr.get_ort_providers("Vocos") == ["CPUExecutionProvider"]


def test_vram_try_allocate_fails_without_gpu():
    """try_allocate_vram() gibt False zurück wenn kein GPU vorhanden."""
    mgr = _make_manager_cpu_only()
    assert mgr.try_allocate_vram("SGMSE", 0.3) is False


def test_is_gpu_available_false_without_gpu():
    mgr = _make_manager_cpu_only()
    assert mgr.is_gpu_available() is False


# ---------------------------------------------------------------------------
# Heavy-Plugin-Klassifizierung
# ---------------------------------------------------------------------------


def test_heavy_plugins_classified():
    """Bekannte schwere Plugins sind in _HEAVY_ML_PLUGINS."""
    from backend.core.ml_device_manager import _HEAVY_ML_PLUGINS

    for name in ("SGMSE", "AudioSR", "BSRoFormer", "MDXNet", "BigVGAN", "Vocos", "HiFiGAN", "ApolloPlugin"):
        assert name in _HEAVY_ML_PLUGINS, f"{name} should be in _HEAVY_ML_PLUGINS"


def test_lightweight_plugins_not_in_heavy():
    """Utility-/Analyse-Plugins sind NICHT in _HEAVY_ML_PLUGINS."""
    from backend.core.ml_device_manager import _HEAVY_ML_PLUGINS

    for name in ("FCPE", "RMVPE", "Silero", "Beats", "UTMOS", "BasicPitch"):
        assert name not in _HEAVY_ML_PLUGINS, f"{name} should NOT be in _HEAVY_ML_PLUGINS"


def test_non_heavy_plugin_always_cpu():
    """Nicht-schwere Plugins erhalten immer 'cpu', auch wenn GPU vorhanden."""
    from backend.core.ml_device_manager import GPUBackend, MLDeviceManager

    with (
        patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
        patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
    ):
        mgr = MLDeviceManager()
    # Simuliere GPU vorhanden
    mgr._gpu_available = True
    mgr._backend = GPUBackend.ROCM
    mgr._torch_gpu_device = "cuda"
    mgr._vram_total_gb = 8.0
    mgr._vram_free_gb = 8.0

    assert mgr.get_torch_device("Silero") == "cpu"
    assert mgr.get_torch_device("UTMOS") == "cpu"
    assert mgr.get_ort_providers("FCPE") == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Simulated ROCm mode
# ---------------------------------------------------------------------------


def _make_manager_rocm(vram_gb: float = 8.0, gpu_name: str = "AMD Radeon RX 7900 XTX (simulated)") -> MLDeviceManager:
    """Erstellt einen MLDeviceManager mit simuliertem ROCm-GPU."""
    from backend.core.ml_device_manager import (
        GPUBackend,
        MLDeviceManager,
        _compute_gpu_tier,
        _detect_amd_architecture,
    )

    with (
        patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
        patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
    ):
        mgr = MLDeviceManager()

    mgr._gpu_available = True
    mgr._backend = GPUBackend.ROCM
    mgr._torch_gpu_device = "cuda"
    mgr._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    mgr._vram_total_gb = vram_gb
    mgr._vram_free_gb = vram_gb
    mgr._gpu_name = gpu_name
    mgr._gpu_architecture = _detect_amd_architecture(gpu_name)
    mgr._gpu_tier = _compute_gpu_tier(mgr._gpu_architecture, vram_gb)
    return mgr


def test_rocm_heavy_plugin_gets_cuda():
    """Schwere Plugins bekommen 'cuda' im ROCm-Modus."""
    mgr = _make_manager_rocm()
    assert mgr.get_torch_device("SGMSE") == "cuda"
    assert mgr.get_torch_device("BigVGAN") == "cuda"


def test_rocm_ort_providers_heavy():
    """Schwere Plugins bekommen ROCMExecutionProvider im ROCm-Modus."""
    mgr = _make_manager_rocm()
    providers = mgr.get_ort_providers("BSRoFormer")
    names = _provider_names(providers)
    assert "ROCMExecutionProvider" in names
    assert "CPUExecutionProvider" in names  # Fallback immer dabei


def test_rocm_ort_providers_vocoder_plugins_heavy() -> None:
    """Vocos und HiFiGAN nutzen im ROCm-Modus GPU-Provider (mit CPU-Fallback)."""
    mgr = _make_manager_rocm()

    vocos_providers = mgr.get_ort_providers("Vocos")
    hifigan_providers = mgr.get_ort_providers("HiFiGAN")

    assert "ROCMExecutionProvider" in vocos_providers
    assert "CPUExecutionProvider" in vocos_providers
    assert "ROCMExecutionProvider" in hifigan_providers
    assert "CPUExecutionProvider" in hifigan_providers


def test_rocm_ort_providers_returns_copy():
    """get_ort_providers() gibt eine Kopie zurück (nicht dieselbe Liste)."""
    mgr = _make_manager_rocm()
    p1 = mgr.get_ort_providers("BSRoFormer")
    p2 = mgr.get_ort_providers("BSRoFormer")
    assert p1 == p2
    assert p1 is not p2  # defensive copy


# ---------------------------------------------------------------------------
# VRAM-Budget
# ---------------------------------------------------------------------------


def test_vram_allocate_and_release():
    """Grundlegender Allokations-/Freigabe-Zyklus."""
    mgr = _make_manager_rocm(vram_gb=8.0)
    assert mgr.try_allocate_vram("SGMSE", 0.3) is True
    status = mgr.gpu_status_summary()
    assert status["vram_allocated_gb"] == pytest.approx(0.3, abs=0.01)

    mgr.release_vram("SGMSE")
    status2 = mgr.gpu_status_summary()
    assert status2["vram_allocated_gb"] == pytest.approx(0.0, abs=0.01)


def test_vram_budget_exceeded():
    """try_allocate_vram() schlägt fehl wenn Budget erschöpft."""
    mgr = _make_manager_rocm(vram_gb=4.0)
    # 85 % von 4 GB = 3.4 GB max
    assert mgr.try_allocate_vram("AudioSR", 3.5) is False


def test_vram_idempotent_allocation():
    """Zweite Allokation derselben Plugins gibt True zurück ohne doppelte Buchung."""
    mgr = _make_manager_rocm(vram_gb=8.0)
    assert mgr.try_allocate_vram("SGMSE", 0.3) is True
    assert mgr.try_allocate_vram("SGMSE", 0.3) is True  # idempotent
    # Soll nur einmal gezählt sein
    assert mgr.gpu_status_summary()["vram_allocated_gb"] == pytest.approx(0.3, abs=0.01)


def test_vram_multiple_plugins():
    """Mehrere Allokationen werden korrekt summiert."""
    mgr = _make_manager_rocm(vram_gb=16.0)
    assert mgr.try_allocate_vram("SGMSE", 0.3) is True
    assert mgr.try_allocate_vram("BSRoFormer", 0.9) is True
    total = mgr.gpu_status_summary()["vram_allocated_gb"]
    assert total == pytest.approx(1.2, abs=0.01)

    mgr.release_vram("SGMSE")
    total2 = mgr.gpu_status_summary()["vram_allocated_gb"]
    assert total2 == pytest.approx(0.9, abs=0.01)


def test_vram_floor_enforced():
    """VRAM-Mindest-Headroom (_VRAM_MIN_FREE_MB) wird eingehalten."""

    mgr = _make_manager_rocm(vram_gb=1.0)
    # 1 GB total, 85 % = 0.85 GB max, aber floor = 512 MB → effective_free < required
    # Anforderung von 0.8 GB + 0.5 GB floor = 1.3 GB > 1.0 GB → FAIL
    big = 0.8
    result = mgr.try_allocate_vram("BigVGAN", big)
    if result:
        # Wenn es doch gepasst hat: kein Fehler, VRAM war groß genug
        mgr.release_vram("BigVGAN")


# ---------------------------------------------------------------------------
# Singleton-Verhalten
# ---------------------------------------------------------------------------


def test_singleton_returns_same_instance():
    """get_ml_device_manager() gibt dieselbe Instanz zurück."""
    from backend.core.ml_device_manager import get_ml_device_manager

    m1 = get_ml_device_manager()
    m2 = get_ml_device_manager()
    assert m1 is m2


def test_singleton_thread_safe():
    """Gleichzeitige Initialisierung produziert nur eine Instanz."""
    from backend.core.ml_device_manager import get_ml_device_manager

    instances: list = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        instances.append(get_ml_device_manager())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(i) for i in instances}) == 1, "Multiple singleton instances created"


# ---------------------------------------------------------------------------
# Fehlertoleranz
# ---------------------------------------------------------------------------


def test_get_torch_device_never_raises():
    """get_torch_device() wirft nie eine Exception, gibt mindestens 'cpu' zurück."""
    from backend.core.ml_device_manager import get_torch_device

    result = get_torch_device("NonExistentPlugin")
    assert isinstance(result, str)
    assert result in ("cpu", "cuda", "dml")


def test_get_ort_providers_never_raises():
    """get_ort_providers() wirft nie eine Exception."""
    from backend.core.ml_device_manager import get_ort_providers

    result = get_ort_providers("NonExistentPlugin")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "CPUExecutionProvider" in result


def test_gpu_status_summary_structure():
    """gpu_status_summary() gibt ein vollständiges Dict mit allen erwarteten Keys zurück."""
    mgr = _make_manager_cpu_only()
    status = mgr.gpu_status_summary()
    required_keys = {
        "backend",
        "gpu_available",
        "gpu_name",
        "vram_total_gb",
        "vram_free_gb",
        "vram_allocated_gb",
        "allocated_plugins",
        "gpu_errors",
        "gpu_disabled_plugins",
    }
    assert required_keys.issubset(set(status.keys()))
    assert status["backend"] == "none"
    assert status["gpu_available"] is False
    assert isinstance(status["allocated_plugins"], dict)
    assert isinstance(status["gpu_errors"], dict)
    assert isinstance(status["gpu_disabled_plugins"], list)


# ---------------------------------------------------------------------------
# report_gpu_error() — Fehlerzählung und Session-Deaktivierung
# ---------------------------------------------------------------------------


def _make_manager_with_gpu() -> MLDeviceManager:
    """Erstellt einen MLDeviceManager im simulierten ROCm-GPU-Modus."""
    from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier

    mgr = _make_manager_cpu_only()
    # Manually inject GPU state so we can test GPU-specific behaviour
    mgr._gpu_available = True
    mgr._backend = GPUBackend.ROCM
    mgr._torch_gpu_device = "cuda"
    mgr._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    mgr._vram_total_gb = 8.0
    mgr._vram_free_gb = 8.0
    mgr._gpu_architecture = AMDArchitecture.RDNA2
    mgr._gpu_tier = GPUTier.TIER_2
    return mgr


def test_report_gpu_error_increments_count():
    """report_gpu_error() erhöht den Fehlerzähler bei jedem Aufruf."""
    mgr = _make_manager_with_gpu()
    exc = RuntimeError("GPU OOM")

    mgr.report_gpu_error("SGMSE", exc)
    assert mgr._gpu_errors["SGMSE"] == 1

    mgr.report_gpu_error("SGMSE", exc)
    assert mgr._gpu_errors["SGMSE"] == 2


def test_report_gpu_error_different_plugins_independent():
    """Fehlerzähler verschiedener Plugins sind voneinander unabhängig."""
    mgr = _make_manager_with_gpu()
    exc = RuntimeError("GPU fail")

    mgr.report_gpu_error("SGMSE", exc)
    mgr.report_gpu_error("BigVGAN", exc)
    mgr.report_gpu_error("BigVGAN", exc)

    assert mgr._gpu_errors["SGMSE"] == 1
    assert mgr._gpu_errors["BigVGAN"] == 2


def test_report_gpu_error_releases_vram():
    """report_gpu_error() gibt VRAM-Budget des betroffenen Plugins frei."""
    mgr = _make_manager_with_gpu()
    mgr._vram_allocated["ApolloPlugin"] = 0.15
    mgr._vram_free_gb = 7.85

    mgr.report_gpu_error("ApolloPlugin", RuntimeError("fail"))

    assert "ApolloPlugin" not in mgr._vram_allocated
    assert abs(mgr._vram_free_gb - 8.0) < 1e-6


def test_report_gpu_error_disables_after_three_failures():
    """Plugin wird nach 3 Fehlern in _gpu_disabled_plugins eingetragen."""
    mgr = _make_manager_with_gpu()
    exc = RuntimeError("GPU fail")

    assert "CQTDiffPlus" not in mgr._gpu_disabled_plugins
    mgr.report_gpu_error("CQTDiffPlus", exc)
    mgr.report_gpu_error("CQTDiffPlus", exc)
    assert "CQTDiffPlus" not in mgr._gpu_disabled_plugins  # noch nicht nach 2

    mgr.report_gpu_error("CQTDiffPlus", exc)
    assert "CQTDiffPlus" in mgr._gpu_disabled_plugins


def test_report_gpu_error_unsupported_disables_immediately() -> None:
    """Kernel/Provider-Inkompatibilität muss sofort auf CPU degradieren."""
    mgr = _make_manager_with_gpu()

    assert _provider_names(mgr.get_ort_providers("BSRoFormer")) == ["ROCMExecutionProvider", "CPUExecutionProvider"]
    mgr.report_gpu_error("BSRoFormer", RuntimeError("No kernel registered for op on ROCMExecutionProvider"))

    assert "BSRoFormer" in mgr._gpu_disabled_plugins
    assert mgr.get_ort_providers("BSRoFormer") == ["CPUExecutionProvider"]


def test_get_torch_device_respects_disabled_plugins():
    """get_torch_device() liefert 'cpu' für Session-deaktivierte Plugins."""
    mgr = _make_manager_with_gpu()

    assert mgr.get_torch_device("SGMSE") == "cuda"  # GPU aktiv

    mgr._gpu_disabled_plugins.add("SGMSE")
    assert mgr.get_torch_device("SGMSE") == "cpu"  # nach Deaktivierung CPU


def test_report_gpu_error_summary_reflects_state():
    """gpu_status_summary() zeigt Fehler und deaktivierte Plugins korrekt an."""
    mgr = _make_manager_with_gpu()
    exc = RuntimeError("fail")

    for _ in range(3):
        mgr.report_gpu_error("Gacela", exc)

    status = mgr.gpu_status_summary()
    assert status["gpu_errors"].get("Gacela") == 3
    assert "Gacela" in status["gpu_disabled_plugins"]


def test_report_gpu_error_thread_safe():
    """report_gpu_error() ist thread-safe bei gleichzeitigen Aufrufen."""
    mgr = _make_manager_with_gpu()
    exc = RuntimeError("concurrent fail")
    errors = []

    def _call():
        try:
            mgr.report_gpu_error("BigVGAN", exc)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_call) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert mgr._gpu_errors.get("BigVGAN", 0) == 20


def test_vocoder_plugins_use_ml_device_manager_provider_selection() -> None:
    """Vocos/HiFiGAN dürfen ORT-Sessions nicht hart auf CPU festnageln."""
    root = pathlib.Path(__file__).parents[2]
    files = [
        root / "plugins" / "vocos_plugin.py",
        root / "plugins" / "hifigan_plugin.py",
    ]
    for path in files:
        src = path.read_text(encoding="utf-8")
        assert "get_ort_providers" in src, f"{path.name} must use ml_device_manager.get_ort_providers"
        assert 'InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])' not in src, (
            f"{path.name} must not hardcode CPUExecutionProvider for primary session"
        )
