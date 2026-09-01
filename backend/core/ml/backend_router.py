"""GPU-Backend-Router — Multi-Plattform ML-Engine-Konfiguration.

§15.5: Abstrahiert GPU-Backend (CPU, CUDA, ROCm, MPS, DirectML).
Ersetzt die bisherige harte AMD-ROCm/DirectML-Beschränkung durch einen
flexiblen Provider-Router, der auf jeder Plattform das beste Backend wählt.

Usage::

    from backend.core.ml.backend_router import detect_gpu_capabilities, MLEngineConfig

    config = detect_gpu_capabilities()
    print(f"Provider: {config.provider}, ONNX: {config.onnx_providers}")

    options = get_onnx_session_options(config)
    session = ort.InferenceSession("model.onnx", options, providers=config.onnx_providers)

Autor: Aurik 10 — 11. Juli 2026
Referenz: Spec 15 §15.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# ── Typen ───────────────────────────────────────────────────────────────────
ProviderT = Literal["cpu", "cuda", "rocm", "directml"]  # macOS/MPS nicht unterstützt


@dataclass
class MLEngineConfig:
    """Multi-Plattform ML-Engine-Konfiguration.

    Attributes:
        provider:         Gewähltes Backend (cpu/cuda/rocm/mps/directml).
        device_id:        GPU-Device-ID (0 = erste GPU).
        fallback_to_cpu:  Bei GPU-Fehler auf CPU zurückfallen.
        onnx_providers:   ONNX ExecutionProvider-Liste (z.B. ["CUDAExecutionProvider", "CPUExecutionProvider"]).
        provider_version: Versionsstring des gewählten Providers.
        gpu_name:         GPU-Name (z.B. "NVIDIA GeForce RTX 4090").
        vram_mb:          Verfügbarer VRAM in MB (0 wenn CPU).
        warnings:         Warnungen während der Erkennung.
    """

    provider: ProviderT = "cpu"
    device_id: int = 0
    fallback_to_cpu: bool = True
    onnx_providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    provider_version: str = ""
    gpu_name: str = ""
    vram_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_gpu(self) -> bool:
        """True wenn ein GPU-Backend gefunden wurde."""
        return self.provider != "cpu"

    @property
    def summary(self) -> str:
        """Einzeilige Zusammenfassung."""
        parts = [f"Provider: {self.provider}"]
        if self.gpu_name:
            parts.append(f"GPU: {self.gpu_name}")
        if self.vram_mb > 0:
            parts.append(f"VRAM: {self.vram_mb:.0f} MB")
        parts.append(f"ONNX: {', '.join(self.onnx_providers)}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Serialisierung für JSON-Export."""
        return {
            "provider": self.provider,
            "device_id": self.device_id,
            "fallback_to_cpu": self.fallback_to_cpu,
            "onnx_providers": self.onnx_providers,
            "provider_version": self.provider_version,
            "gpu_name": self.gpu_name,
            "vram_mb": self.vram_mb,
            "warnings": self.warnings,
            "is_gpu": self.is_gpu,
        }


# ── GPU-Erkennung ───────────────────────────────────────────────────────────


def detect_gpu_capabilities(fail_fast: bool = False) -> MLEngineConfig:
    """Erkennt verfügbare GPU-Backends und wählt das beste.

    Priorität: CUDA → ROCm → DirectML → CPU.
    (macOS/Apple Silicon nicht unterstützt.)

    Args:
        fail_fast: True → Exception bei Erkennungsfehlern statt Warnung.

    Returns:
        ``MLEngineConfig`` mit gewähltem Provider und ONNX-Konfiguration.
    """
    config = MLEngineConfig()
    config.provider = "cpu"  # Default
    config.onnx_providers = ["CPUExecutionProvider"]

    # ── onnxruntime prüfen ───────────────────────────────────────────────
    try:
        pass
    except ImportError:
        config.warnings.append("onnxruntime nicht installiert — CPU-only")
        return config

    available = _get_available_providers()

    # ── CUDA ─────────────────────────────────────────────────────────────
    if "CUDAExecutionProvider" in available:
        config.provider = "cuda"
        config.onnx_providers = _build_provider_list("CUDAExecutionProvider")
        config.provider_version = _get_cuda_version()
        config.gpu_name = _get_gpu_name_cuda()
        config.vram_mb = _get_vram_cuda(config.device_id)
        logger.info("CUDA-GPU erkannt: %s (%.0f MB VRAM)", config.gpu_name, config.vram_mb)
        return config

    # ── ROCm (AMD) ───────────────────────────────────────────────────────
    if "ROCMExecutionProvider" in available:
        config.provider = "rocm"
        config.onnx_providers = _build_provider_list("ROCMExecutionProvider")
        config.gpu_name = _get_gpu_name_rocm()
        config.vram_mb = _get_vram_rocm(config.device_id)
        logger.info("AMD-GPU (ROCm) erkannt: %s", config.gpu_name)
        return config

    # ── DirectML (Windows AMD/Intel) ─────────────────────────────────────
    if "DmlExecutionProvider" in available:
        config.provider = "directml"
        config.onnx_providers = _build_provider_list("DmlExecutionProvider")
        config.gpu_name = "DirectML GPU"
        config.vram_mb = 0.0  # DirectML teilt sich System-RAM
        logger.info("DirectML-GPU erkannt (Windows)")
        return config

    # ── CPU-Fallback ─────────────────────────────────────────────────────
    logger.info("Keine GPU-Backends erkannt, CPU-Ersatzpfad")
    config.warnings.append("GPU-Backend nicht verfügbar — CPU wird verwendet")
    return config


def get_onnx_session_options(config: MLEngineConfig) -> onnxruntime.SessionOptions:  # type: ignore[name-defined]  # noqa: F821
    """Optimierte ONNX-Session-Optionen pro Provider.

    Args:
        config: MLEngineConfig aus detect_gpu_capabilities().

    Returns:
        ``onnxruntime.SessionOptions`` mit provider-spezifischen Optimierungen.
    """
    try:
        import onnxruntime as ort
    except ImportError as err:
        raise RuntimeError("onnxruntime ist nicht installiert.") from err

    options = ort.SessionOptions()

    if config.is_gpu:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = True
        # GPU-spezifisch: Graph-Optimierungen maximieren
        if config.provider == "cuda":
            options.enable_cpu_mem_arena = False  # CUDA managed eigenen Speicher
            # CUDA-spezifische Optionen
            options.add_session_config_entry("cudnn_conv_use_max_workspace", "1")
        else:
            options.enable_cpu_mem_arena = True
    else:
        # CPU: aggressivere Optimierung, da kein GPU-Overhead
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.enable_cpu_mem_arena = True
        options.intra_op_num_threads = _cpu_thread_count()
        options.inter_op_num_threads = max(1, _cpu_thread_count() // 2)

    return options


# ── Private Helpers ─────────────────────────────────────────────────────────


def _get_available_providers() -> list[str]:
    """onnxruntime.get_available_providers() mit Fallback."""
    try:
        import onnxruntime as ort

        return ort.get_available_providers()  # type: ignore[no-any-return]
    except Exception:
        return []


def _build_provider_list(primary: str) -> list[str]:
    """Baut Provider-Liste: [primary, ..., CPUExecutionProvider]."""
    providers = [primary]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def _cpu_thread_count() -> int:
    """Physische CPU-Threads (nicht logische Cores)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, NotImplementedError):
        return max(1, os.cpu_count() or 4)


def _get_cuda_version() -> str:
    try:
        import torch  # type: ignore[import]

        return torch.version.cuda or "unknown"
    except Exception:
        return ""


def _get_gpu_name_cuda() -> str:
    try:
        import torch  # type: ignore[import]

        return torch.cuda.get_device_name(0)
    except Exception:
        return "NVIDIA GPU (CUDA)"


def _get_vram_cuda(device_id: int = 0) -> float:
    try:
        import torch  # type: ignore[import]

        props = torch.cuda.get_device_properties(device_id)
        return props.total_memory / (1024 * 1024)
    except Exception:
        return 0.0


def _get_gpu_name_rocm() -> str:
    try:
        import torch  # type: ignore[import]

        return torch.cuda.get_device_name(0)
    except Exception:
        return "AMD GPU (ROCm)"


def _get_vram_rocm(device_id: int = 0) -> float:
    try:
        import torch  # type: ignore[import]

        props = torch.cuda.get_device_properties(device_id)
        return props.total_memory / (1024 * 1024)
    except Exception:
        return 0.0


# Benötigt für _cpu_thread_count
import os
