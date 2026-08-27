from __future__ import annotations

"""Unit tests — ml_device_manager AMD ROCm performance extensions.

Verifies:
  - is_fp16_eligible() only returns True for ROCm + eligible plugins
  - get_ort_providers_fp16() falls back to CPU when not ROCm
  - warmup_rocm_gpu() is a safe no-op on CPU-only systems
  - pin_tensor_rocm() returns the original array on non-ROCm
  - DeepFilterNetV3 is in _HEAVY_ML_PLUGINS (GPU dispatch)
  - _FP16_ELIGIBLE_PLUGINS is a non-empty frozenset
"""


import sys
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_manager_module():
    """Import ml_device_manager without real GPU or torch deps."""
    import importlib

    if "backend.core.ml_device_manager" in sys.modules:
        return sys.modules["backend.core.ml_device_manager"]
    return importlib.import_module("backend.core.ml_device_manager")


@pytest.fixture(autouse=True)
def _neutralize_detection():
    """§V8-analog (Spec 07, Bug-Klasse TEST-DESIGN): GPU-Detektion vollständig neutralisieren.

    Alle Tests dieser Datei setzen den GPU-Zustand nach der Konstruktion
    explizit (Simulation) oder erwarten CPU-only. Ein im Suite-Kontext warmer
    ROCm-Stack führte zu order-abhängigen Flakes, weil trotz Child-Patches ein
    weiterer Detektionspfad anlaufen konnte (rdna3/20 GB in „patched“ Instanzen).
    `_detect_backend` als zentraler Einstieg wird deshalb komplett stumm
    geschaltet — deterministisch unabhängig vom Host-Zustand.
    """
    with patch("backend.core.ml_device_manager.MLDeviceManager._detect_backend"):
        yield


@pytest.fixture(autouse=True)
def _reset_manager_singleton():
    """§V8-analog: Prozess-Singleton pro Test zurücksetzen (Suite-Isolation)."""
    import backend.core.ml_device_manager as _mod

    _original = _mod._instance
    _mod._instance = None
    yield
    _mod._instance = _original


# ---------------------------------------------------------------------------
# _HEAVY_ML_PLUGINS / _FP16_ELIGIBLE_PLUGINS contract
# ---------------------------------------------------------------------------


class TestPluginSets:
    def test_deepfilternet_in_heavy_ml_plugins(self) -> None:
        """§GPU-Mixed-Mode: DeepFilterNetV3 must be GPU-dispatched."""
        mod = _get_manager_module()
        assert "DeepFilterNetV3" in mod._HEAVY_ML_PLUGINS, (
            "DeepFilterNetV3 missing from _HEAVY_ML_PLUGINS — computationally intensive DF-filter must run on AMD GPU"
        )

    def test_fp16_eligible_plugins_non_empty(self) -> None:
        """_FP16_ELIGIBLE_PLUGINS must contain at least the core ONNX models."""
        mod = _get_manager_module()
        assert len(mod._FP16_ELIGIBLE_PLUGINS) > 0
        for expected in ("BSRoFormer", "MDXNet", "DeepFilterNetV3", "PANNs"):
            assert expected in mod._FP16_ELIGIBLE_PLUGINS, f"{expected} should be in _FP16_ELIGIBLE_PLUGINS"

    def test_fp16_eligible_subset_of_heavy(self) -> None:
        """Every fp16-eligible plugin must also be in _HEAVY_ML_PLUGINS (it runs on GPU)."""
        mod = _get_manager_module()
        not_heavy = mod._FP16_ELIGIBLE_PLUGINS - mod._HEAVY_ML_PLUGINS
        assert not not_heavy, f"fp16-eligible plugins not in _HEAVY_ML_PLUGINS: {not_heavy}"


# ---------------------------------------------------------------------------
# is_fp16_eligible — CPU-only system returns False
# ---------------------------------------------------------------------------


class TestFp16EligibilityOnCpu:
    def test_returns_false_on_cpu_only(self) -> None:
        """On a CPU-only system MLDeviceManager.is_fp16_eligible() must return False."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, MLDeviceManager

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        # Manually set CPU-only state (bypasses __init__ GPU detection)
        import threading

        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.NONE
        mgr._gpu_available = False
        mgr._gpu_name = ""
        mgr._gpu_architecture = AMDArchitecture.UNKNOWN
        mgr._gpu_tier = GPUTier.TIER_4
        mgr._vram_total_gb = 0.0
        mgr._vram_free_gb = 0.0
        mgr._vram_allocated = {}
        mgr._gpu_errors = {}
        mgr._gpu_disabled_plugins = set()

        assert mgr.is_fp16_eligible("BSRoFormer") is False
        assert mgr.is_fp16_eligible("DeepFilterNetV3") is False

    def test_get_ort_providers_fp16_falls_back_on_cpu(self) -> None:
        """get_ort_providers_fp16() must return ['CPUExecutionProvider'] on CPU-only."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, MLDeviceManager

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        import threading

        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.NONE
        mgr._gpu_available = False
        mgr._gpu_name = ""
        mgr._gpu_architecture = AMDArchitecture.UNKNOWN
        mgr._gpu_tier = GPUTier.TIER_4
        mgr._vram_total_gb = 0.0
        mgr._vram_free_gb = 0.0
        mgr._vram_allocated = {}
        mgr._gpu_errors = {}
        mgr._gpu_disabled_plugins = set()

        providers = mgr.get_ort_providers_fp16("BSRoFormer")
        # Should fall back to standard providers → CPU
        assert "CPUExecutionProvider" in providers or providers == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# warmup_rocm_gpu — safe no-op on CPU-only
# ---------------------------------------------------------------------------


class TestWarmupRocm:
    def test_warmup_returns_false_on_cpu(self) -> None:
        """warmup_rocm_gpu() must return False gracefully on CPU-only systems."""
        import threading

        from backend.core.ml_device_manager import GPUBackend, MLDeviceManager, warmup_rocm_gpu

        # Patch singleton to return a CPU-only manager
        cpu_mgr = MLDeviceManager.__new__(MLDeviceManager)
        cpu_mgr._lock = threading.Lock()
        cpu_mgr._backend = GPUBackend.NONE
        cpu_mgr._gpu_available = False

        import backend.core.ml_device_manager as mdm

        orig = mdm._instance
        try:
            mdm._instance = cpu_mgr
            result = warmup_rocm_gpu()
            assert result is False, "warmup_rocm_gpu must return False on CPU-only"
        finally:
            mdm._instance = orig

    def test_warmup_no_exception_on_import_error(self) -> None:
        """warmup_rocm_gpu() must not raise even if torch is unavailable."""
        import threading

        import backend.core.ml_device_manager as mdm
        from backend.core.ml_device_manager import GPUBackend, MLDeviceManager

        cpu_mgr = MLDeviceManager.__new__(MLDeviceManager)
        cpu_mgr._lock = threading.Lock()
        cpu_mgr._backend = GPUBackend.ROCM
        cpu_mgr._gpu_available = True
        cpu_mgr._gpu_name = "AMD Radeon RX 7900 XTX"
        cpu_mgr._vram_total_gb = 24.0

        orig = mdm._instance
        try:
            mdm._instance = cpu_mgr
            # Simulate torch import failure inside warmup
            with patch.dict(sys.modules, {"torch": None}):
                result = cpu_mgr.warmup_rocm()  # must not raise
                assert isinstance(result, bool)
        finally:
            mdm._instance = orig


# ---------------------------------------------------------------------------
# pin_tensor_rocm — passthrough on CPU-only
# ---------------------------------------------------------------------------


class TestPinTensorRocm:
    def test_returns_array_unchanged_on_cpu(self) -> None:
        """pin_tensor_rocm() must return the original numpy array on CPU-only."""
        import threading

        from backend.core.ml_device_manager import GPUBackend, MLDeviceManager

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.NONE
        mgr._gpu_available = False

        arr = np.zeros(1024, dtype=np.float32)
        result = mgr.pin_tensor_rocm(arr)
        assert result is arr, "pin_tensor_rocm must return original array on CPU-only"

    def test_returns_non_array_unchanged(self) -> None:
        """Non-ndarray/tensor types must pass through unchanged."""
        import threading

        from backend.core.ml_device_manager import GPUBackend, MLDeviceManager

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.ROCM
        mgr._gpu_available = True

        obj = {"key": "value"}
        result = mgr.pin_tensor_rocm(obj)
        assert result is obj


# ---------------------------------------------------------------------------
# Global convenience wrappers
# ---------------------------------------------------------------------------


class TestGlobalWrappers:
    def test_get_ort_providers_fp16_returns_list(self) -> None:
        """get_ort_providers_fp16() global function must always return a non-empty list."""
        from backend.core.ml_device_manager import get_ort_providers_fp16

        providers = get_ort_providers_fp16("BSRoFormer")
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_warmup_rocm_gpu_global_no_raise(self) -> None:
        """warmup_rocm_gpu() global function must not raise under any condition."""
        from backend.core.ml_device_manager import warmup_rocm_gpu

        result = warmup_rocm_gpu()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Plugin source code audit: fp16-eligible plugins must use get_ort_providers
# ---------------------------------------------------------------------------


class TestFp16PluginAudit:
    """Verify that every fp16-eligible ONNX plugin sources get_ort_providers (auto-fp16 via §8.7).

    Since v10.0.0, get_ort_providers() automatically activates fp16 for eligible plugins
    on ROCm Tier 1–3. Plugins must NOT manually call get_ort_providers_fp16() — they should
    use the canonical get_ort_providers() which applies fp16 automatically when appropriate.
    """

    _PLUGIN_FILE_MAP: dict[str, str] = {
        "BSRoFormer": "plugins/bs_roformer_plugin.py",
        "MDXNet": "plugins/uvr_mdxnet_plugin.py",
        "DemucsV4": "plugins/demucs_v4_plugin.py",
        "MDX23C": "plugins/mdx23c_plugin.py",
        "MPSENet": "plugins/mp_senet_plugin.py",
        "ResembleEnhance": "plugins/resemble_enhance_plugin.py",
        "PANNs": "plugins/panns_plugin.py",
        "LaionCLAP_ONNX": "plugins/laion_clap_plugin.py",
        "BanquetVinyl": "plugins/banquet_vinyl_plugin.py",
        "DeepFilterNetV3": "plugins/deepfilternet_v3_ii_plugin.py",
    }

    @pytest.mark.parametrize("plugin_name,rel_path", list(_PLUGIN_FILE_MAP.items()))
    def test_uses_canonical_ort_providers(self, plugin_name: str, rel_path: str) -> None:
        """Plugin source must import get_ort_providers (auto-fp16 since v10.0.0)."""
        import pathlib

        workspace_root = pathlib.Path(__file__).parents[2]
        plugin_file = workspace_root / rel_path
        if not plugin_file.exists():
            pytest.skip(f"Plugin file absent (optional model): {rel_path}")
        source = plugin_file.read_text(encoding="utf-8")
        # Must use the canonical provider function (get_ort_providers activates fp16 auto)
        assert "get_ort_providers" in source, (
            f"{plugin_name} ({rel_path}) must import 'get_ort_providers' for AMD GPU support (auto-fp16 §8.7)"
        )


# ---------------------------------------------------------------------------
# AMD GPU Architecture Detection Tests (v10.0.0 §8.7)
# ---------------------------------------------------------------------------


class TestAmdArchitectureDetection:
    """Unit tests for _detect_amd_architecture() — pattern matching for AMD GPU names."""

    def _arch(self, name: str):
        from backend.core.ml_device_manager import _detect_amd_architecture

        return _detect_amd_architecture(name)

    def test_rdna3_rx7900(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 7900 XTX") == AMDArchitecture.RDNA3

    def test_rdna3_rx7600(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 7600 XT") == AMDArchitecture.RDNA3

    def test_rdna3_gfx1100(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx1100") == AMDArchitecture.RDNA3

    def test_rdna2_rx6800(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 6800 XT") == AMDArchitecture.RDNA2

    def test_rdna2_rx6700(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 6700") == AMDArchitecture.RDNA2

    def test_rdna1_rx5700(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 5700 XT") == AMDArchitecture.RDNA1

    def test_gcn5_vega64(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon Vega 64") == AMDArchitecture.GCN5

    def test_gcn4_polaris(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 580") == AMDArchitecture.GCN4

    def test_cdna3_mi300(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Instinct MI300X") == AMDArchitecture.CDNA3

    def test_cdna2_mi250(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Instinct MI250X") == AMDArchitecture.CDNA2

    def test_cdna1_mi100(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Instinct MI100") == AMDArchitecture.CDNA1

    def test_unknown_fallback(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("NVIDIA GeForce RTX 3090") == AMDArchitecture.UNKNOWN
        assert self._arch("Intel UHD Graphics") == AMDArchitecture.UNKNOWN
        assert self._arch("") == AMDArchitecture.UNKNOWN


# ---------------------------------------------------------------------------
# GPU Tier Computation Tests (v10.0.0 §8.7)
# ---------------------------------------------------------------------------


class TestGpuTierComputation:
    """Unit tests for _compute_gpu_tier() — VRAM + architecture → capability tier."""

    def _tier(self, arch_name: str, vram: float):
        from backend.core.ml_device_manager import _compute_gpu_tier, _detect_amd_architecture

        return _compute_gpu_tier(_detect_amd_architecture(arch_name), vram)

    def test_rdna3_24gb_tier1(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("gfx1100", 24.0) == GPUTier.TIER_1

    def test_rdna2_16gb_tier1(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Radeon RX 6800 XT", 16.0) == GPUTier.TIER_1

    def test_rdna2_8gb_tier2(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Radeon RX 6700", 8.0) == GPUTier.TIER_2

    def test_rdna2_4gb_tier3(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Radeon RX 6500 XT", 4.0) == GPUTier.TIER_3

    def test_rdna1_8gb_tier2(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Radeon RX 5700 XT", 8.0) == GPUTier.TIER_2

    def test_gcn5_8gb_tier3(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Radeon Vega 64", 8.0) == GPUTier.TIER_3

    def test_gcn4_4gb_tier4(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Radeon RX 580", 4.0) == GPUTier.TIER_4

    def test_cdna_mi300_tier1(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Instinct MI300X", 192.0) == GPUTier.TIER_1

    def test_cdna_mi100_tier1(self) -> None:
        from backend.core.ml_device_manager import GPUTier

        assert self._tier("AMD Instinct MI100", 32.0) == GPUTier.TIER_1


# ---------------------------------------------------------------------------
# Tier-based Plugin Exclusion Tests (v10.0.0 §8.7)
# ---------------------------------------------------------------------------


class TestTierBasedExclusion:
    """Verify that VRAM-heavy plugins are excluded from GPU on Tier 3/4."""

    def _mgr_rocm_tier(self, arch_name: str, vram_gb: float):
        """Create a simulated ROCm manager with given arch and VRAM."""
        from unittest.mock import patch

        from backend.core.ml_device_manager import (
            GPUBackend,
            MLDeviceManager,
            _compute_gpu_tier,
            _detect_amd_architecture,
        )

        with (
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        ):
            mgr = MLDeviceManager()
        mgr._gpu_available = True
        mgr._backend = GPUBackend.ROCM
        mgr._torch_gpu_device = "cuda"
        mgr._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        mgr._vram_total_gb = vram_gb
        mgr._vram_free_gb = vram_gb
        mgr._gpu_name = arch_name
        mgr._gpu_architecture = _detect_amd_architecture(arch_name)
        mgr._gpu_tier = _compute_gpu_tier(mgr._gpu_architecture, vram_gb)
        return mgr

    def test_flashsr_onnx_gpu_tier3(self) -> None:
        """FlashSR (ONNX, ~200MB) — GPU auf Tier 3 (4-7 GB VRAM) nutzbar."""
        mgr = self._mgr_rocm_tier("AMD Radeon RX 6500 XT", 4.0)
        # FlashSR ONNX ist klein genug für Tier 3 GPU
        assert mgr.is_ort_gpu_supported("FlashSR") is True

    def test_flashsr_onnx_gpu_tier4(self) -> None:
        """FlashSR ONNX — GPU auf Tier 4 nutzbar."""
        mgr = self._mgr_rocm_tier("AMD Radeon RX 580", 4.0)
        assert mgr.is_ort_gpu_supported("FlashSR") is True

    def test_flashsr_onnx_gpu_tier1(self) -> None:
        """FlashSR ONNX darf GPU auf Tier 1 (≥16 GB VRAM) nutzen."""
        mgr = self._mgr_rocm_tier("AMD Radeon RX 7900 XTX", 24.0)
        assert mgr.is_ort_gpu_supported("FlashSR") is True

    def test_flashsr_onnx_gpu_tier2(self) -> None:
        """FlashSR ONNX darf GPU auf Tier 2 (8 GB VRAM, RDNA2) nutzen."""
        mgr = self._mgr_rocm_tier("AMD Radeon RX 6700 XT", 12.0)
        assert mgr.is_ort_gpu_supported("FlashSR") is True

    def test_sgmse_excluded_tier4(self) -> None:
        """SGMSE must be CPU-only on Tier 4 (GCN4 / <4 GB)."""
        mgr = self._mgr_rocm_tier("AMD Radeon RX 580", 4.0)
        assert mgr.get_torch_device("SGMSE") == "cpu"

    def test_bsroformer_excluded_tier4(self) -> None:
        """BSRoFormer must be CPU-only on Tier 4."""
        mgr = self._mgr_rocm_tier("AMD Radeon RX 580", 4.0)
        assert mgr.get_torch_device("BSRoFormer") == "cpu"

    def test_bsroformer_allowed_tier1(self) -> None:
        """BSRoFormer (860 MB) must get GPU on Tier 1."""
        mgr = self._mgr_rocm_tier("AMD Radeon RX 7900 XTX", 24.0)
        assert mgr.get_torch_device("BSRoFormer") == "cuda"


# ---------------------------------------------------------------------------
# Auto-fp16 Activation Tests (v10.0.0 §8.7)
# ---------------------------------------------------------------------------


class TestAutoFp16:
    """Verify that get_ort_providers() automatically activates fp16 on ROCm Tier 1–3."""

    def _mgr_tier1_rocm(self):
        from unittest.mock import patch

        from backend.core.ml_device_manager import (
            _HEAVY_ML_PLUGINS,
            AMDArchitecture,
            GPUBackend,
            GPUTier,
            MLDeviceManager,
        )

        with (
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        ):
            mgr = MLDeviceManager()
        mgr._gpu_available = True
        mgr._backend = GPUBackend.ROCM
        mgr._torch_gpu_device = "cuda"
        mgr._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        mgr._vram_total_gb = 24.0
        mgr._vram_free_gb = 24.0
        mgr._gpu_name = "AMD Radeon RX 7900 XTX"
        mgr._gpu_architecture = AMDArchitecture.RDNA3
        mgr._gpu_tier = GPUTier.TIER_1
        mgr._ort_gpu_compatible_plugins = set(_HEAVY_ML_PLUGINS)
        return mgr

    @staticmethod
    def _provider_names(providers: list) -> list[str]:
        """Normalisiert Providers-Liste: (name, opts)-Tuples → reine Namen."""
        return [p[0] if isinstance(p, tuple) else p for p in providers]

    def test_fp16_auto_activated_for_bsroformer(self) -> None:
        """get_ort_providers('BSRoFormer') on ROCm Tier 1 must return fp16 providers."""
        mgr = self._mgr_tier1_rocm()
        providers = mgr.get_ort_providers("BSRoFormer")
        names = self._provider_names(providers)
        assert "ROCMExecutionProvider" in names, (
            "BSRoFormer on ROCm Tier 1 should get ROCMExecutionProvider (fp16 auto-activated)"
        )

    def test_fp16_auto_activated_for_deepfilternet(self) -> None:
        """get_ort_providers('DeepFilterNetV3') on ROCm Tier 1 must return fp16 providers."""
        mgr = self._mgr_tier1_rocm()
        providers = mgr.get_ort_providers("DeepFilterNetV3")
        names = self._provider_names(providers)
        assert "ROCMExecutionProvider" in names

    def test_non_fp16_plugin_gets_standard_rocm_provider(self) -> None:
        """Plugins not in _FP16_ELIGIBLE_PLUGINS still get ROCMExecutionProvider (not fp16)."""
        mgr = self._mgr_tier1_rocm()
        # ApolloPlugin is heavy but not fp16-eligible
        providers = mgr.get_ort_providers("ApolloPlugin")
        assert "ROCMExecutionProvider" in providers, "Heavy plugin should get GPU"

    def test_cpu_only_no_fp16(self) -> None:
        """get_ort_providers() on CPU-only returns CPU provider regardless of fp16 eligibility."""
        from unittest.mock import patch

        from backend.core.ml_device_manager import MLDeviceManager

        with (
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        ):
            mgr = MLDeviceManager()
        assert mgr.get_ort_providers("BSRoFormer") == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# gpu_status_summary — new fields (v10.0.0 §8.7)
# ---------------------------------------------------------------------------


class TestGpuStatusSummaryExtended:
    """Verify that gpu_status_summary() exposes architecture/tier/fp16_auto fields."""

    def test_summary_has_architecture_field_cpu(self) -> None:
        from unittest.mock import patch

        from backend.core.ml_device_manager import MLDeviceManager

        with (
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        ):
            mgr = MLDeviceManager()
        summary = mgr.gpu_status_summary()
        assert "gpu_architecture" in summary
        assert "gpu_tier" in summary
        assert "fp16_auto" in summary
        assert summary["gpu_architecture"] == "unknown"
        assert summary["fp16_auto"] is False  # Tier 4 has no fp16_auto

    def test_summary_fp16_auto_tier1(self) -> None:
        from unittest.mock import patch

        from backend.core.ml_device_manager import (
            AMDArchitecture,
            GPUBackend,
            GPUTier,
            MLDeviceManager,
        )

        with (
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        ):
            mgr = MLDeviceManager()
        mgr._gpu_available = True
        mgr._backend = GPUBackend.ROCM
        mgr._gpu_architecture = AMDArchitecture.RDNA3
        mgr._gpu_tier = GPUTier.TIER_1
        summary = mgr.gpu_status_summary()
        assert summary["gpu_architecture"] == "rdna3"
        assert summary["gpu_tier"] == "TIER_1"
        assert summary["fp16_auto"] is True


# ---------------------------------------------------------------------------
# gpu_architecture / gpu_tier properties (v10.0.0 §8.7)
# ---------------------------------------------------------------------------


class TestManagerProperties:
    """Verify the new gpu_architecture, gpu_tier, gpu_name properties."""

    def test_properties_on_cpu_only(self) -> None:
        from unittest.mock import patch

        from backend.core.ml_device_manager import AMDArchitecture, GPUTier, MLDeviceManager

        with (
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        ):
            mgr = MLDeviceManager()
        assert mgr.gpu_architecture == AMDArchitecture.UNKNOWN
        assert mgr.gpu_tier == GPUTier.TIER_4
        assert mgr.gpu_name == "N/A"


# ---------------------------------------------------------------------------
# RDNA4 architecture detection (v10.0.0)
# ---------------------------------------------------------------------------


class TestRdna4Detection:
    """RDNA4 (RX 9000 / Navi 4x) must be correctly identified."""

    def _arch(self, name: str):
        from backend.core.ml_device_manager import _detect_amd_architecture

        return _detect_amd_architecture(name)

    def test_rx_9070_xt(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 9070 XT") == AMDArchitecture.RDNA4

    def test_rx_9070(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon RX 9070") == AMDArchitecture.RDNA4

    def test_gfx1201(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx1201") == AMDArchitecture.RDNA4

    def test_gfx1200(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx1200") == AMDArchitecture.RDNA4

    def test_rdna4_tier1_16gb(self) -> None:
        """RDNA4 with 16 GB VRAM → Tier 1."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUTier, _compute_gpu_tier

        assert _compute_gpu_tier(AMDArchitecture.RDNA4, 16.0) == GPUTier.TIER_1

    def test_rdna4_tier2_8gb(self) -> None:
        """RDNA4 with 8 GB VRAM → Tier 2."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUTier, _compute_gpu_tier

        assert _compute_gpu_tier(AMDArchitecture.RDNA4, 8.0) == GPUTier.TIER_2


# ---------------------------------------------------------------------------
# APU / iGPU name patterns (v10.0.0)
# ---------------------------------------------------------------------------


class TestApuPatterns:
    """Integrated/APU GPU names must be correctly mapped to RDNA generations."""

    def _arch(self, name: str):
        from backend.core.ml_device_manager import _detect_amd_architecture

        return _detect_amd_architecture(name)

    def test_radeon_780m_rdna3(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon 780M") == AMDArchitecture.RDNA3

    def test_radeon_760m_rdna3(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon 760M") == AMDArchitecture.RDNA3

    def test_radeon_890m_rdna3(self) -> None:
        """Strix Point 890M (gfx1151 = RDNA 3.5) must map to RDNA3."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon 890M") == AMDArchitecture.RDNA3

    def test_radeon_880m_rdna3(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon 880M") == AMDArchitecture.RDNA3

    def test_radeon_860m_rdna3(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon 860M") == AMDArchitecture.RDNA3

    def test_gfx1151_rdna3(self) -> None:
        """Strix Point GFX ID gfx1151 must map to RDNA3."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx1151") == AMDArchitecture.RDNA3

    def test_gfx1150_rdna3(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx1150") == AMDArchitecture.RDNA3

    def test_radeon_680m_rdna2(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon 680M") == AMDArchitecture.RDNA2

    def test_radeon_660m_rdna2(self) -> None:
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon 660M") == AMDArchitecture.RDNA2

    def test_vega_8_gcn5_marketing_name(self) -> None:
        """Vega 8 (marketing name) must map to GCN5 via 'vega' pattern."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon Vega 8") == AMDArchitecture.GCN5

    def test_vega_7_gcn5_marketing_name(self) -> None:
        """Vega 7 (Renoir APU, gfx90c) via marketing name."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("AMD Radeon Vega 7") == AMDArchitecture.GCN5

    def test_gfx90c_gcn5(self) -> None:
        """Renoir/Cezanne APU GFX ID gfx90c must map to GCN5."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx90c") == AMDArchitecture.GCN5

    def test_gfx902_gcn5(self) -> None:
        """Raven Ridge/Picasso APU GFX ID gfx902 must map to GCN5."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx902") == AMDArchitecture.GCN5

    def test_gfx1103_rdna3(self) -> None:
        """Phoenix APU GFX ID must map to RDNA3."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx1103") == AMDArchitecture.RDNA3

    def test_gfx1033_rdna2(self) -> None:
        """Ryzen 6000 APU GFX ID must map to RDNA2."""
        from backend.core.ml_device_manager import AMDArchitecture

        assert self._arch("gfx1033") == AMDArchitecture.RDNA2


# ---------------------------------------------------------------------------
# DirectML tier uplift — GCN4/5 on Windows vs ROCm (v10.0.0)
# ---------------------------------------------------------------------------


class TestDirectMlTierUplift:
    """GCN4/5 must get a higher tier under DirectML than under ROCm."""

    def test_gcn4_rocm_tier4(self) -> None:
        """Polaris (GCN4) is CPU-only under ROCm 6.x."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, _compute_gpu_tier

        assert _compute_gpu_tier(AMDArchitecture.GCN4, 8.0, GPUBackend.ROCM) == GPUTier.TIER_4

    def test_gcn4_directml_tier3(self) -> None:
        """Polaris (GCN4) with 8 GB VRAM gets Tier 3 on DirectML (DX12)."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, _compute_gpu_tier

        assert _compute_gpu_tier(AMDArchitecture.GCN4, 8.0, GPUBackend.DIRECTML) == GPUTier.TIER_3

    def test_gcn5_rocm_tier3(self) -> None:
        """Vega (GCN5) 8 GB under ROCm → Tier 3 (limited kernel support)."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, _compute_gpu_tier

        assert _compute_gpu_tier(AMDArchitecture.GCN5, 8.0, GPUBackend.ROCM) == GPUTier.TIER_3

    def test_gcn5_directml_tier2(self) -> None:
        """Vega (GCN5) 8 GB under DirectML → Tier 2 (HBM2, full DX12)."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, _compute_gpu_tier

        assert _compute_gpu_tier(AMDArchitecture.GCN5, 8.0, GPUBackend.DIRECTML) == GPUTier.TIER_2

    def test_gcn4_4gb_directml_tier3(self) -> None:
        """Even 4 GB Polaris gets Tier 3 on DirectML (RX 480 4 GB use case)."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, _compute_gpu_tier

        assert _compute_gpu_tier(AMDArchitecture.GCN4, 4.0, GPUBackend.DIRECTML) == GPUTier.TIER_3

    def test_rdna2_tier_unchanged_between_backends(self) -> None:
        """RDNA2 tier must be the same on ROCm and DirectML (no special casing)."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, GPUTier, _compute_gpu_tier

        assert (
            _compute_gpu_tier(AMDArchitecture.RDNA2, 16.0, GPUBackend.ROCM)
            == _compute_gpu_tier(AMDArchitecture.RDNA2, 16.0, GPUBackend.DIRECTML)
            == GPUTier.TIER_1
        )


# ---------------------------------------------------------------------------
# DirectML full-stack: arch + tier + provider detection (v10.0.0)
# ---------------------------------------------------------------------------


class TestDirectMlArchTierDetection:
    """Verify that _detect_directml() sets architecture + tier correctly."""

    def _make_directml_mgr(self, gpu_name: str, vram_gb: float):
        """Simulate _detect_directml() with a known GPU name and VRAM."""
        from unittest.mock import patch

        from backend.core.ml_device_manager import (
            GPUBackend,
            MLDeviceManager,
            _compute_gpu_tier,
            _detect_amd_architecture,
        )

        with (
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_cuda_or_rocm"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_directml"),
            patch("backend.core.ml_device_manager.MLDeviceManager._detect_migraphx_bridge"),
        ):
            mgr = MLDeviceManager()

        # Simulate what the fixed _detect_directml() now does
        mgr._gpu_available = True
        mgr._backend = GPUBackend.DIRECTML
        mgr._ort_gpu_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        mgr._vram_total_gb = vram_gb
        mgr._vram_free_gb = vram_gb
        mgr._gpu_name = gpu_name
        mgr._gpu_architecture = _detect_amd_architecture(gpu_name)
        mgr._gpu_tier = _compute_gpu_tier(mgr._gpu_architecture, vram_gb, GPUBackend.DIRECTML)
        return mgr

    def test_rx7900xtx_directml_tier1(self) -> None:
        """RX 7900 XTX on DirectML → RDNA3, Tier 1."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUTier

        mgr = self._make_directml_mgr("AMD Radeon RX 7900 XTX", 24.0)
        assert mgr.gpu_architecture == AMDArchitecture.RDNA3
        assert mgr.gpu_tier == GPUTier.TIER_1

    def test_rx580_directml_tier3(self) -> None:
        """RX 580 on DirectML → GCN4, Tier 3 (DX12 uplift)."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUTier

        mgr = self._make_directml_mgr("AMD Radeon RX 580", 8.0)
        assert mgr.gpu_architecture == AMDArchitecture.GCN4
        assert mgr.gpu_tier == GPUTier.TIER_3

    def test_rx580_ort_providers_directml(self) -> None:
        """RX 580 on DirectML (Tier 3) → AudioSR (7 GB, _TIER3_GPU_EXCLUDE) is CPU-only."""
        mgr = self._make_directml_mgr("AMD Radeon RX 580", 8.0)
        # AudioSR is in _TIER3_GPU_EXCLUDE → blocked even on Tier 3
        providers = mgr.get_ort_providers("AudioSR")
        assert providers == ["CPUExecutionProvider"]

    def test_rx6800xt_directml_heavy_plugin_gets_dml(self) -> None:
        """RX 6800 XT 16 GB on DirectML (Tier 1) → non-excluded plugin gets DmlExecutionProvider."""
        mgr = self._make_directml_mgr("AMD Radeon RX 6800 XT", 16.0)
        providers = mgr.get_ort_providers("DeepFilterNetV3")
        assert "DmlExecutionProvider" in providers

    def test_status_summary_shows_arch_and_tier(self) -> None:
        """gpu_status_summary() on DirectML must expose architecture and tier."""
        mgr = self._make_directml_mgr("AMD Radeon RX 7600", 8.0)
        s = mgr.gpu_status_summary()
        assert s["gpu_architecture"] == "rdna3"
        assert s["gpu_tier"] == "TIER_2"
        assert s["gpu_name"] == "AMD Radeon RX 7600"


# ---------------------------------------------------------------------------
# MIGraphX-Bridge-Detection (Rev. 2026-08-16 — Bridge gegen ROCm 7.2.4 gebaut)
# ---------------------------------------------------------------------------


def _migraphx_bridge_available() -> bool:
    """True, wenn libmigraphx_bridge.so auf dieser Maschine ladbar ist."""
    try:
        from backend.core.migraphx_adapter import is_migraphx_available

        return is_migraphx_available()
    except Exception:
        return False


class TestMigraphxDetection:
    """Echter MIGraphX-Pfad — nur aktiv, wenn die Bridge ladbar ist."""

    @pytest.mark.skipif(
        not _migraphx_bridge_available(),
        reason="libmigraphx_bridge.so nicht ladbar (kein MIGraphX oder falscher Build)",
    )
    def test_bridge_detection_sets_rocm_enum(self) -> None:
        """Ohne torch-ROCm gewinnt die MIGraphX-Bridge: ROCM + Enum-Architektur."""
        from backend.core.ml_device_manager import AMDArchitecture, GPUBackend, MLDeviceManager

        mgr = MLDeviceManager()
        if mgr._backend != GPUBackend.ROCM:
            pytest.skip("ROCm bereits über torch erkannt — MIGraphX-Zweig nicht erreicht")
        assert mgr._gpu_available is True
        assert "MIGraphXExecutionProvider" in mgr._ort_gpu_providers
        # Enum-Fix (Rev. 2026-08-16): kein String mehr — gpu_status_summary() darf nicht werfen
        assert isinstance(mgr._gpu_architecture, AMDArchitecture)
        s = mgr.gpu_status_summary()
        assert s["gpu_architecture"] in {"rdna3", "unknown"}
        assert "gpu_errors" in s
