"""ml_device_manager — Mixed CPU/GPU Device Selection for Aurik 10.0.0.

Detects available GPU acceleration (ROCm on Linux, DirectML on Windows) and
provides a unified interface for per-plugin device assignment.

Mixed-mode operation (§GPU-Mixed-Mode):
  - Heavy ML models (large transformers, diffusion models) → GPU when available
  - DSP modules, lightweight/utility models            → always CPU
  - Any GPU failure                                    → transparent CPU fallback
    (§2.47 ML-Failure-Degradationskaskade)

Platform support:
  Linux:   ROCm 6.x via PyTorch ROCm wheel (uses torch.cuda API identically to CUDA)
  Windows: DirectML via onnxruntime-directml + optionally torch-directml
  Both:    CPU-only fallback (preserves §9.5-compatible behaviour when no GPU present)

ROCm-Versionskope (Rev. 2026-08-16):
  Verifiziert: ROCm 6.x und ROCm 7.2.4.
  Produktions-Stack 7.2.4 (Rev. 2026-08-16): Radeon RX 7900 XTX (gfx1100),
  torch 2.11.0+rocm7.2 (hip 7.2.26015) + torchaudio 2.11.0+rocm7.2 +
  torchvision 0.26.0+rocm7.2 + onnxruntime-rocm 1.22.2.post3
  (ROCMExecutionProvider). Hinweis: torchaudio ist für ROCm 7.2 offiziell nur
  bis 2.11.0 verfügbar — neuere bräuchten Eigenbau.
  GPU-Gate tests/unit/test_gpu_determinism_gate.py: 6/6 grün auf der Hardware
  (Backend-Erkennung, kein §v10.304.30-Hang, GPU=CPU innerhalb 1e-5).
  §G5: Bit-Determinismus ist nur für den CPU-Pfad garantiert; GPU-Inferenz
  ist Toleranz-equal, nicht bit-identisch (keine deterministischen Kernel konfiguriert).

VRAM budget (§GPU-VRAM-Guard):
  Analog to ml_memory_budget for system RAM — tracks allocated VRAM per plugin and
  falls back to CPU when VRAM headroom is insufficient. On ROCm, free VRAM is queried
  via torch.cuda.mem_get_info(); on DirectML, manually tracked.

Singleton: get_ml_device_manager()
Convenience wrappers: get_torch_device(plugin_name), get_ort_providers(plugin_name)
"""

from __future__ import annotations

import enum
import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

logger = logging.getLogger(__name__)

# Lazy/optionale ML-Imports innerhalb von Methoden sind bewusste Aurik-Design-Entscheidung
# (RELEASE_MUST: optionale ML-Dependencies immer per try/except eingebunden — §E402).
# pylint: disable=import-outside-toplevel

# ---------------------------------------------------------------------------
# GPU Backend enum
# ---------------------------------------------------------------------------


class GPUBackend(enum.Enum):
    """GPU-Beschleunigungsbackend — CUDA (NVIDIA), ROCm (AMD Linux), DirectML (AMD Windows) oder CPU-Only."""

    CUDA = "cuda"  # NVIDIA GPU via CUDA (Linux & Windows)
    ROCM = "rocm"  # AMD GPU via ROCm 6.x / 7.2.4 (Linux only; 7.2.4 per GPU-Gate verifiziert)
    DIRECTML = "directml"  # AMD GPU via DirectML (Windows only)
    NONE = "none"  # CPU-only (no GPU or GPU suppressed)


class AMDArchitecture(enum.Enum):
    """AMD GPU micro-architecture family — determines kernel behaviour and limits."""

    RDNA4 = "rdna4"  # RX 9000 series (Navi 4x) — wave32, AI accelerators Gen2 (2025+)
    RDNA3 = "rdna3"  # RX 7000 series (Navi 3x) — wave32, AV1, AI accelerators
    RDNA2 = "rdna2"  # RX 6000 series (Navi 2x) — wave32, ray-tracing, Infinity Cache
    RDNA1 = "rdna1"  # RX 5000 series (Navi 1x) — wave32, first RDNA
    GCN5 = "gcn5"  # RX Vega 56/64 (Vega) — wave64, HBM2; DX12 on Windows
    GCN4 = "gcn4"  # RX 400/500 series (Polaris) — wave64; DX12 on Windows
    GCN3 = "gcn3"  # R9 380/390/Fury (Volcanic Islands / Hawaii) — DX12 on Windows
    CDNA3 = "cdna3"  # MI300 series — data-centre, HBM3, matrix cores
    CDNA2 = "cdna2"  # MI200 series — data-centre, HBM2e, matrix cores
    CDNA1 = "cdna1"  # MI100 — data-centre, HBM2, first CDNA
    UNKNOWN = "unknown"


class GPUTier(enum.Enum):
    """GPU capability tier — controls VRAM budgets, fp16 policy and batch sizing.

    Tier 1: ≥16 GB VRAM, modern architecture → all plugins GPU-eligible, fp16 auto
    Tier 2: 8–15 GB VRAM, RDNA2+ or CDNA → most plugins GPU, fp16 for heavy models
    Tier 3: 4–7 GB VRAM or legacy arch → selective GPU, conservative budgets
    Tier 4: <4 GB VRAM or very old arch → CPU-only (GPU overhead > benefit)
    """

    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
    TIER_4 = 4


# ---------------------------------------------------------------------------
# AMD GPU name → architecture mapping
#
# ROCm exposes torch.cuda.get_device_name() with the marketing name.
# We pattern-match to determine the architecture family.
# ---------------------------------------------------------------------------

_AMD_ARCH_PATTERNS: list[tuple[str, AMDArchitecture]] = [
    # ── CDNA (data-centre) — check first; names may overlap with consumer cards ──
    ("mi300", AMDArchitecture.CDNA3),
    ("mi250", AMDArchitecture.CDNA2),
    ("mi210", AMDArchitecture.CDNA2),
    ("mi200", AMDArchitecture.CDNA2),
    ("mi100", AMDArchitecture.CDNA1),
    ("mi50", AMDArchitecture.CDNA1),
    # ── RDNA 4 (Navi 4x) — RX 9000 series (2025) ────────────────────────────────
    ("navi4", AMDArchitecture.RDNA4),
    ("rx 9070", AMDArchitecture.RDNA4),
    ("rx 9060", AMDArchitecture.RDNA4),
    ("rx 9050", AMDArchitecture.RDNA4),
    ("gfx1200", AMDArchitecture.RDNA4),
    ("gfx1201", AMDArchitecture.RDNA4),
    # ── RDNA 3 (Navi 3x) — RX 7000 series ──────────────────────────────────────
    ("navi3", AMDArchitecture.RDNA3),
    # Discrete desktop/laptop (marketing names)
    ("7900", AMDArchitecture.RDNA3),
    ("7800", AMDArchitecture.RDNA3),
    ("7700", AMDArchitecture.RDNA3),
    ("7600", AMDArchitecture.RDNA3),
    # Professional (W-series)
    ("w7900", AMDArchitecture.RDNA3),
    ("w7800", AMDArchitecture.RDNA3),
    ("w7700", AMDArchitecture.RDNA3),
    ("w7600", AMDArchitecture.RDNA3),
    # APU / integrated (Phoenix, Hawk Point, Strix Point)
    ("890m", AMDArchitecture.RDNA3),  # Radeon 890M  = gfx1151 (Strix Point)
    ("880m", AMDArchitecture.RDNA3),  # Radeon 880M  = gfx1151 (Strix Point)
    ("870m", AMDArchitecture.RDNA3),  # Radeon 870M  = gfx1150 (Strix Point)
    ("860m", AMDArchitecture.RDNA3),  # Radeon 860M  = gfx1150 (Strix Point/Hawk Point Pro)
    ("780m", AMDArchitecture.RDNA3),  # Radeon 780M  = gfx1103 (Phoenix)
    ("760m", AMDArchitecture.RDNA3),  # Radeon 760M  = gfx1103
    ("740m", AMDArchitecture.RDNA3),  # Radeon 740M  = gfx1103
    # GFX IDs
    ("gfx1100", AMDArchitecture.RDNA3),
    ("gfx1101", AMDArchitecture.RDNA3),
    ("gfx1102", AMDArchitecture.RDNA3),
    ("gfx1103", AMDArchitecture.RDNA3),  # APU: Phoenix/Hawk Point
    ("gfx1150", AMDArchitecture.RDNA3),  # APU: Strix Point (RDNA 3.5 — same ISA)
    ("gfx1151", AMDArchitecture.RDNA3),  # APU: Strix Point / Strix Halo
    # ── RDNA 2 (Navi 2x) — RX 6000 series ──────────────────────────────────────
    ("navi2", AMDArchitecture.RDNA2),
    # Discrete desktop/laptop
    ("6900", AMDArchitecture.RDNA2),
    ("6800", AMDArchitecture.RDNA2),
    ("6750", AMDArchitecture.RDNA2),
    ("6700", AMDArchitecture.RDNA2),
    ("6650", AMDArchitecture.RDNA2),
    ("6600", AMDArchitecture.RDNA2),
    ("6500", AMDArchitecture.RDNA2),
    ("6400", AMDArchitecture.RDNA2),
    ("6300", AMDArchitecture.RDNA2),
    ("6200", AMDArchitecture.RDNA2),
    # Professional
    ("w6800", AMDArchitecture.RDNA2),
    ("w6700", AMDArchitecture.RDNA2),
    ("w6600", AMDArchitecture.RDNA2),
    ("w6400", AMDArchitecture.RDNA2),
    # APU / integrated (Van Gogh, Rembrandt, Mendocino)
    ("680m", AMDArchitecture.RDNA2),  # Radeon 680M = gfx1035 (Rembrandt)
    ("660m", AMDArchitecture.RDNA2),  # Radeon 660M = gfx1034
    ("610m", AMDArchitecture.RDNA2),  # Radeon 610M = gfx1036
    # GFX IDs
    ("gfx1030", AMDArchitecture.RDNA2),
    ("gfx1031", AMDArchitecture.RDNA2),
    ("gfx1032", AMDArchitecture.RDNA2),
    ("gfx1033", AMDArchitecture.RDNA2),  # Ryzen 6000 APU
    ("gfx1034", AMDArchitecture.RDNA2),
    ("gfx1035", AMDArchitecture.RDNA2),
    ("gfx1036", AMDArchitecture.RDNA2),
    # ── RDNA 1 (Navi 1x) — RX 5000 series ──────────────────────────────────────
    ("navi1", AMDArchitecture.RDNA1),
    ("5700", AMDArchitecture.RDNA1),
    ("5600", AMDArchitecture.RDNA1),
    ("5500", AMDArchitecture.RDNA1),
    ("5300", AMDArchitecture.RDNA1),
    # GFX IDs
    ("gfx1010", AMDArchitecture.RDNA1),
    ("gfx1011", AMDArchitecture.RDNA1),
    ("gfx1012", AMDArchitecture.RDNA1),
    ("gfx1013", AMDArchitecture.RDNA1),  # Integrated RDNA1 (Renoir/Cézanne)
    # ── GCN 5 (Vega / Vega 20 / Radeon VII + APU Renoir/Picasso) ────────────────
    ("vega", AMDArchitecture.GCN5),
    ("radeon vii", AMDArchitecture.GCN5),
    ("gfx900", AMDArchitecture.GCN5),  # Vega 10 (discrete)
    ("gfx906", AMDArchitecture.GCN5),  # Vega 20 / Radeon VII
    ("gfx90c", AMDArchitecture.GCN5),  # APU: Renoir (Ryzen 4000) / Cezanne (Ryzen 5000) — Vega 7/8
    ("gfx902", AMDArchitecture.GCN5),  # APU: Raven Ridge / Picasso (Ryzen 2000/3000) — Vega 8/11
    # Note: gfx908 is CDNA1 (MI100), not consumer GCN5
    # ── GCN 4 (Polaris) ─────────────────────────────────────────────────────────
    # ROCm 6.x: no official support; DirectML: DX12-capable → GPU acceleration
    ("polaris", AMDArchitecture.GCN4),
    ("rx 590", AMDArchitecture.GCN4),
    ("rx 580", AMDArchitecture.GCN4),
    ("rx 570", AMDArchitecture.GCN4),
    ("rx 560", AMDArchitecture.GCN4),
    ("rx 550", AMDArchitecture.GCN4),
    ("rx 480", AMDArchitecture.GCN4),
    ("rx 470", AMDArchitecture.GCN4),
    ("rx 460", AMDArchitecture.GCN4),
    ("gfx803", AMDArchitecture.GCN4),  # Polaris 10/11/12
    # ── GCN 3 (Volcanic Islands / Hawaii) ────────────────────────────────────────
    # ROCm: not supported; DirectML: DX12-capable on some models
    ("r9 390", AMDArchitecture.GCN3),
    ("r9 380", AMDArchitecture.GCN3),
    ("r9 285", AMDArchitecture.GCN3),
    ("fury", AMDArchitecture.GCN3),
    ("nano", AMDArchitecture.GCN3),  # R9 Nano (Fiji)
    ("gfx802", AMDArchitecture.GCN3),  # Tonga
    ("gfx801", AMDArchitecture.GCN3),  # Fiji
    ("gfx800", AMDArchitecture.GCN3),  # Carrizo APU (GCN3)
]


def _detect_amd_architecture(device_name: str) -> AMDArchitecture:
    """Match *device_name* against known AMD GPU patterns to determine architecture."""
    lower = device_name.lower()
    for pattern, arch in _AMD_ARCH_PATTERNS:
        if pattern in lower:
            return arch
    return AMDArchitecture.UNKNOWN


def _compute_gpu_tier(
    arch: AMDArchitecture,
    vram_gb: float,
    backend: GPUBackend = None,  # type: ignore[assignment]
) -> GPUTier:
    """Bestimmt GPU capability tier from architecture, VRAM and backend.

    ROCm (Linux):    tier reflects kernel / wavefront-mode support; GCN4+ unsupported.
    DirectML (Windows): DX12-based — GCN4 and GCN3 GPUs receive a one-tier uplift because
                        DirectML bypasses ROCm's kernel-coverage limitations.
    """
    # Resolve late import (GPUBackend is defined above in the same module scope)
    _be = backend  # may be None at module load time (pure-function use in tests)
    _is_directml = _be is not None and _be.value == "directml"

    # ── CDNA (data-centre) — always Tier 1 when enough VRAM ─────────────────
    if arch in (AMDArchitecture.CDNA3, AMDArchitecture.CDNA2, AMDArchitecture.CDNA1):
        return GPUTier.TIER_1 if vram_gb >= 8.0 else GPUTier.TIER_2

    # ── RDNA 4 — treat same as RDNA 3 (same tier thresholds, newer arch) ───
    if arch == AMDArchitecture.RDNA4:
        if vram_gb >= 16.0:
            return GPUTier.TIER_1
        if vram_gb >= 8.0:
            return GPUTier.TIER_2
        if vram_gb >= 4.0:
            return GPUTier.TIER_3
        return GPUTier.TIER_4

    # ── RDNA 3 / RDNA 2 — modern architectures, tier purely by VRAM ─────────
    if arch in (AMDArchitecture.RDNA3, AMDArchitecture.RDNA2):
        if vram_gb >= 16.0:
            return GPUTier.TIER_1
        if vram_gb >= 8.0:
            return GPUTier.TIER_2
        if vram_gb >= 4.0:
            return GPUTier.TIER_3
        return GPUTier.TIER_4

    # ── RDNA 1 — wave32, but older; cap at Tier 2 ───────────────────────────
    if arch == AMDArchitecture.RDNA1:
        if vram_gb >= 8.0:
            return GPUTier.TIER_2
        if vram_gb >= 4.0:
            return GPUTier.TIER_3
        return GPUTier.TIER_4

    # ── GCN 5 (Vega) — wave64; ROCm 6.x limited, DirectML full DX12 ─────────
    if arch == AMDArchitecture.GCN5:
        if _is_directml:
            # Vega has 8–16 GB HBM2 and DX12 feature-level 12_0 → Tier 2 on DirectML
            if vram_gb >= 8.0:
                return GPUTier.TIER_2
            return GPUTier.TIER_3
        # ROCm: limited kernel coverage
        if vram_gb >= 8.0:
            return GPUTier.TIER_3
        return GPUTier.TIER_4

    # ── GCN 4 (Polaris) — ROCm 6.x: unsupported; DirectML: DX12-capable ────
    if arch == AMDArchitecture.GCN4:
        if _is_directml:
            if vram_gb >= 8.0:
                return GPUTier.TIER_3
            if vram_gb >= 4.0:
                return GPUTier.TIER_3
            return GPUTier.TIER_4
        return GPUTier.TIER_4  # ROCm: not supported → CPU-only

    # ── GCN 3 (Volcanic Islands / Hawaii) — DirectML only, very conservative ─
    if arch == AMDArchitecture.GCN3:
        # Only light ONNX models via DirectML; cap at Tier 4 unless DirectML + decent VRAM
        if _is_directml and vram_gb >= 4.0:
            return GPUTier.TIER_4  # minimal benefit; still CPU-only for heavy plugins
        return GPUTier.TIER_4

    # ── UNKNOWN — conservative, VRAM-only estimate ───────────────────────────
    if vram_gb >= 16.0:
        return GPUTier.TIER_2
    if vram_gb >= 8.0:
        return GPUTier.TIER_3
    return GPUTier.TIER_4


# ---------------------------------------------------------------------------
# Tier-adaptive VRAM budget parameters
# ---------------------------------------------------------------------------

_TIER_VRAM_PARAMS: dict[GPUTier, dict[str, float]] = {
    # max_usage_ratio: fraction of total VRAM Aurik may use
    # min_free_mb: hard floor of free VRAM to preserve
    # fp16_auto: whether fp16 is automatically activated for eligible plugins
    GPUTier.TIER_1: {"max_usage_ratio": 0.85, "min_free_mb": 512.0, "fp16_auto": 1.0},
    GPUTier.TIER_2: {"max_usage_ratio": 0.80, "min_free_mb": 640.0, "fp16_auto": 1.0},
    GPUTier.TIER_3: {"max_usage_ratio": 0.70, "min_free_mb": 768.0, "fp16_auto": 1.0},
    GPUTier.TIER_4: {"max_usage_ratio": 0.50, "min_free_mb": 512.0, "fp16_auto": 0.0},
}


# Tier 3/4 plugins that are too large for small-VRAM GPUs → CPU-only
_TIER3_GPU_EXCLUDE: frozenset[str] = frozenset(
    {
        "AudioLDM2",  # ~1.3 GB peak — needs headroom
        "MERT-330M-fairseq",  # ~3.7 GB — only Tier 1/2
    }
)

_TIER4_GPU_EXCLUDE: frozenset[str] = frozenset(
    {
        *_TIER3_GPU_EXCLUDE,
        "MERT-330M-HF",  # ~1.2 GB — too large for <4 GB VRAM
        "BSRoFormer",  # ~860 MB peak
        "MDXNet",  # ~1.2 GB peak
        "BigVGAN",  # ~400 MB + intermediates
        "CQTDiffPlus",  # diffusion iterations need headroom
        "SGMSE",  # diffusion score-matching
    }
)


# ---------------------------------------------------------------------------
# Heavy ML plugins eligible for GPU acceleration
#
# Only plugins in this frozenset are dispatched to the GPU device.
# DSP modules, lightweight utility models (< ~100 MB), and quality-scoring
# plugins stay on CPU — their compute is too small to amortise the
# CPU↔GPU transfer overhead.
# ---------------------------------------------------------------------------

_HEAVY_ML_PLUGINS: frozenset[str] = frozenset(
    {
        # --- Stem Separation (>500 MB) ---
        "SGMSE",  # sgmse_plugin — diffusion score-matching (251 MB)
        "AudioSR",  # audiosr_plugin — diffusion super-resolution (~7 GB, heaviest plugin)
        "FlashSR",  # flashsr_plugin — FlashSR ONNX bandwidth extension (~2 GB)
        "BSRoFormer",  # bs_roformer_plugin — stem separation ONNX (860 MB)
        "MDXNet",  # uvr_mdxnet_plugin — stem separation ONNX (~1.2 GB)
        "DemucsV5",  # demucs_v5_wrapper — stem separation
        "MDX23C",  # mdx23c wrapper — stem separation
        "DemucsV4",  # demucs_v4_plugin — htdemucs_6s ONNX (~500 MB)
        # --- Neural Vocoders / Enhancement ---
        "BigVGAN",  # bigvgan_v2_plugin — neural vocoder (400 MB)
        "Vocos",  # vocos_plugin — primary neural vocoder in restore/export path
        "HiFiGAN",  # hifigan_plugin — neural vocoder fallback path
        "ApolloPlugin",  # apollo_plugin — restorative ML
        "CQTDiffPlus",  # cqtdiff_plus_plugin — diffusion inpainting
        "Gacela",  # gacela_plugin — audio inpainting
        "MPSENet",  # mp_senet plugin — speech enhancement
        "ResembleEnhance",  # resemble_enhance_plugin — ONNX (~722 MB)
        "AudioLDM2",  # audioldm2_plugin — diffusion ONNX (~1.3 GB)
        # --- Neural Enhancement (GPU-accelerated on AMD ROCm) ---
        "DeepFilterNetV3",  # deepfilternet_v3_ii_plugin — 3x enc/dec ONNX (~150 MB)
        #                     Computationally intensive DF-filter; ROCm accelerates
        #                     the iterative ERB-inverse + deep filtering significantly.
        # --- Quality / Scoring (heavy Transformer backbones) ---
        "MERT-330M-HF",  # mert_plugin — HuggingFace Transformer (~1.2 GB)
        "MERT-330M-fairseq",  # mert_plugin — fairseq checkpoint (~3.7 GB)
        "VersaSingMOS",  # versa_plugin — wav2vec2 backbone (~800 MB)
        "UTMOSv2",  # utmos_plugin — torch fold models (~800 MB)
        "BanquetVinyl",  # banquet_vinyl_plugin — ONNX (~800 MB)
        # --- Audio Understanding ---
        "PANNs",  # panns_plugin — CNN14 ONNX (~660 MB)
        "LaionCLAP_ONNX",  # laion_clap_plugin — ONNX audio encoder (~300 MB)
    }
)

# ---------------------------------------------------------------------------
# AMD ROCm fp16 ONNX inference — eligible plugins
# ROCm supports fp16 for all ONNX models via ROCMExecutionProvider.
# fp16 halves VRAM usage and doubles inference throughput at negligible quality loss
# for audio enhancement models (activations already clipped to [-1, 1]).
# ---------------------------------------------------------------------------

_FP16_ELIGIBLE_PLUGINS: frozenset[str] = frozenset(
    {
        # ONNX models where fp16 is safe (activations bounded, no integer ops):
        "BSRoFormer",
        "MDXNet",
        "DemucsV4",
        "MPSENet",
        "ResembleEnhance",
        "PANNs",
        "LaionCLAP_ONNX",
        "BanquetVinyl",
        "DeepFilterNetV3",
        # PyTorch models where .half() is safe on AMD ROCm:
        "BigVGAN",
        "MDX23C",
    }
)

# Error hints that indicate provider/kernel-level GPU incompatibility.
_GPU_UNSUPPORTED_ERROR_HINTS: tuple[str, ...] = (
    "not implemented",
    "unsupported",
    "no kernel",
    "kernel not found",
    "failed to find kernel",
    "not supported",
    "execution provider",
    "hiperror",
    "hip error",
    "invaliddevicefunction",
    "invalid device function",
)

# ---------------------------------------------------------------------------
# VRAM budget constants
# ---------------------------------------------------------------------------

# Keep at least this much VRAM free after allocations (hard floor).
_VRAM_MIN_FREE_MB: float = 512.0
# At most this fraction of total VRAM may be used by Aurik plugins combined.
# NOTE: This constant is superseded by tier-adaptive _TIER_VRAM_PARAMS[gpu_tier]["max_usage_ratio"].
# Retained only as a module-level documentation value; do NOT use it in logic.
_VRAM_MAX_USAGE_RATIO: float = 0.85  # see _TIER_VRAM_PARAMS for actual per-tier values

# ---------------------------------------------------------------------------
# AMD ROCm performance constants
# ---------------------------------------------------------------------------

# Warmup dummy tensor dimensions for ROCm runtime initialisation (sec §GPU-Perf)
_ROCM_WARMUP_TENSOR_LEN: int = 4096  # samples — tiny enough to be instant
# ROCm graph capture is not used (Triton compilation overhead > benefit for audio)
_ROCM_TORCH_THREADS: int = 1  # inter-op threads — GPU already parallel

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_instance: MLDeviceManager | None = None
_init_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public convenience wrappers
# ---------------------------------------------------------------------------


def get_ml_device_manager() -> MLDeviceManager:
    """Gibt the process-wide MLDeviceManager singleton (thread-safe, lazy-init) zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = MLDeviceManager()
    return _instance


def get_torch_device(plugin_name: str = "") -> str:
    """Gibt the PyTorch device string for *plugin_name* zurück.

    Returns ``"cpu"``, ``"cuda"`` (ROCm), or ``"dml"`` (DirectML+torch-directml).
    Heavy plugins (in ``_HEAVY_ML_PLUGINS``) get the GPU device when available;
    all other plugins always get ``"cpu"``.
    """
    try:
        return get_ml_device_manager().get_torch_device(plugin_name)
    except Exception as exc:
        logger.debug("get_torch_device Ersatzpfad to cpu: %s", exc)
        return "cpu"


def get_ort_providers(plugin_name: str = "") -> list[str]:
    """Gibt the ONNX Runtime provider list for *plugin_name* zurück.

    Heavy plugins get GPU provider + CPU fallback; others get CPU-only.
    """
    try:
        return get_ml_device_manager().get_ort_providers(plugin_name)
    except Exception as exc:
        logger.debug("get_ort_providers Ersatzpfad to CPU: %s", exc)
        return ["CPUExecutionProvider"]


def get_ort_providers_fp16(plugin_name: str = "") -> list[str]:
    """Gibt ORT providers with AMD ROCm fp16 hint for *plugin_name* zurück.

    For eligible ONNX plugins on ROCm, returns ROCMExecutionProvider with
    memory-efficient options.  Falls back to standard providers on CPU/DirectML.
    """
    try:
        return get_ml_device_manager().get_ort_providers_fp16(plugin_name)
    except Exception as exc:
        logger.debug("get_ort_providers_fp16 Ersatzpfad to CPU: %s", exc)
        return ["CPUExecutionProvider"]


def warmup_rocm_gpu() -> bool:
    """Initialisiert AMD-ROCm-Laufzeitumgebung, um Cold-Start-Latenz vor der ersten Inferenz zu eliminieren.

    Call once from the application startup path (e.g. BatchProcessingThread.__init__).
    Safe no-op on CPU-only systems or Windows (DirectML warmup is automatic).
    """
    try:
        return get_ml_device_manager().warmup_rocm()
    except Exception as exc:
        logger.debug("warmup_rocm_gpu fehlgeschlagen (unkritisch): %s", exc)
        return False


# ---------------------------------------------------------------------------
# MLDeviceManager
# ---------------------------------------------------------------------------


class MLDeviceManager:
    """Erkennt GPU backend and manages device assignment for ML plugins.

    Responsibilities:
      1. Platform-aware GPU backend detection (CUDA / ROCm / DirectML / None)
      2. Per-plugin device assignment (heavy plugins → GPU, rest → CPU)
      3. VRAM budget guard (analog to ml_memory_budget for system RAM)
      4. Transparent CPU fallback on any GPU failure or budget exhaustion
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: GPUBackend = GPUBackend.NONE
        self._torch_gpu_device: str = "cpu"  # "cuda" (ROCm) | "dml" (DirectML)
        self._ort_gpu_providers: list[str] = ["CPUExecutionProvider"]
        self._vram_allocated: dict[str, float] = {}  # plugin_name → GB
        self._vram_total_gb: float = 0.0
        self._vram_free_gb: float = 0.0
        self._gpu_available: bool = False
        self._gpu_name: str = "N/A"
        self._gpu_architecture: AMDArchitecture = AMDArchitecture.UNKNOWN
        self._gpu_tier: GPUTier = GPUTier.TIER_4
        self._gpu_errors: dict[str, int] = {}  # plugin_name → error count
        self._gpu_disabled_plugins: set[str] = set()  # session-disabled plugins
        self._ort_gpu_compatible_plugins: set[str] = set(_HEAVY_ML_PLUGINS)
        self._detection_complete = threading.Event()  # §v10.304.24: Signal für Warmup
        # §v10.304.28: AURIK_FORCE_CPU=1 überspringt GPU-Detektion komplett.
        # Nützlich bei ROCm-Treiberproblemen (Thread-Deadlocks, HIP-Kontext-Fehler).
        if os.environ.get("AURIK_FORCE_CPU", "").strip() in ("1", "true", "yes"):
            logger.info("MLDeviceManager: AURIK_FORCE_CPU=1 — GPU deaktiviert, CPU-only")
            self._detection_complete.set()
            return
        try:
            self._detect_backend()
            self._detection_complete.set()  # §v10.304.24: GPU-Erkennung abgeschlossen
        except Exception:
            self._detection_complete.set()  # Garantie: Event immer gesetzt, kein ewiges wait()
            raise

    # ── §v10.304.24: Warten auf GPU-Erkennung ──────────────────────────

    def wait_for_detection(self, timeout_s: float = 60.0) -> bool:
        """Blockiert bis GPU-Erkennung abgeschlossen ist (max timeout_s)."""
        return self._detection_complete.wait(timeout=timeout_s)

    # ── Detection ────────────────────────────────────────────────────────

    def _detect_backend(self) -> None:
        """Auto-detect available GPU backend. Silently falls back to CPU on error.

        Detection order (first match wins):
          All platforms: CUDA (NVIDIA) or ROCm (AMD) via torch.cuda
          Windows only:  DirectML (AMD) fallback
        """
        try:
            self._detect_cuda_or_rocm()
            if self._gpu_available:
                return
            # MIGraphX standalone bridge: AMD GPU without ONNX Runtime ROCm EP
            self._detect_migraphx_bridge()
            if self._gpu_available:
                return
            if sys.platform == "win32":
                self._detect_directml()
        except Exception as exc:
            logger.debug("MLDeviceManager: backend detection error (CPU Ersatzpfad): %s", exc)
            self._gpu_available = False
            self._backend = GPUBackend.NONE

        if self._gpu_available:
            arch_str = (
                self._gpu_architecture.value
                if hasattr(self._gpu_architecture, "value")
                else str(self._gpu_architecture)
            )
            logger.info(
                "MLDeviceManager: GPU backend=%s device=%s VRAM=%.1f GB gpu=%s arch=%s tier=%s",
                self._backend.value,
                self._torch_gpu_device,
                self._vram_total_gb,
                self._gpu_name,
                arch_str,
                self._gpu_tier.name,
            )
            # §v10.304.30: Klare, grep-bare Status-Zeile
            _backend_label = (
                "ROCm"
                if self._backend == GPUBackend.ROCM
                else "CUDA"
                if self._backend == GPUBackend.CUDA
                else self._backend.value
            )
            logger.info("══════════════════════════════════════════════")
            logger.info("GPU STATUS: %s AKTIV — %s (%.1f GB VRAM)", _backend_label, self._gpu_name, self._vram_total_gb)
            logger.info("  PyTorch device:  %s", self._torch_gpu_device)
            logger.info("  ONNX providers:  %s", " → ".join(self._ort_gpu_providers))
            logger.info("══════════════════════════════════════════════")
        else:
            logger.info(
                "MLDeviceManager: no GPU backend — CPU-only Betriebsart (CUDA/ROCm/DirectML not found or not installed)"
            )
            logger.info("══════════════════════════════════════════════")
            logger.info("GPU STATUS: DEAKTIVIERT — CPU-only")
            logger.info("══════════════════════════════════════════════")

    def _detect_cuda_or_rocm(self) -> None:
        """Erkennt NVIDIA CUDA oder AMD ROCm via torch.cuda.

        Beide Backends nutzen die gleiche torch.cuda API, unterscheiden sich aber in:
          - AMD:  torch.version.hip ist gesetzt → ROCMExecutionProvider
          - NVIDIA: torch.version.cuda ist gesetzt → CUDAExecutionProvider
        """
        try:
            import torch  # type: ignore[import]

            if not torch.cuda.is_available():
                logger.debug("MLDeviceManager: torch.cuda nicht verfuegbar — pruefe ONNX Ersatzpfad")
                self._detect_rocm_onnx_only()
                return

            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024**3)

            # Distinguish AMD ROCm vs NVIDIA CUDA
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            is_cuda = hasattr(torch.version, "cuda") and torch.version.cuda is not None

            if is_rocm:
                self._backend = GPUBackend.ROCM
                self._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
                self._gpu_architecture = _detect_amd_architecture(device_name)
                logger.info("MLDeviceManager: AMD ROCm erkannt — HIP %s", torch.version.hip)
                # §v10.304.30: ONNX ROCm-Sanity-Probe sofort nach Erkennung.
                # Prüft ob ROCMExecutionProvider tatsächlich funktioniert.
                # Defekte ONNX-ROCm-Installationen (hipErrorInvalidDeviceFunction)
                # werden sofort erkannt → GPU global deaktiviert statt dass
                # jeder einzelne Plugin-Inferenz-Call fehlschlägt.
                self._probe_rocm_onnx_pad()
            elif is_cuda:
                self._backend = GPUBackend.CUDA
                self._ort_gpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._gpu_architecture = AMDArchitecture.UNKNOWN
                logger.info("MLDeviceManager: NVIDIA CUDA erkannt — CUDA %s", torch.version.cuda)
            else:
                # Fallback: assume ROCm (backward compatibility)
                self._backend = GPUBackend.ROCM
                self._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
                self._gpu_architecture = _detect_amd_architecture(device_name)
                logger.info("MLDeviceManager: GPU erkannt (Ersatzpfad ROCm) — %s", device_name)

            self._torch_gpu_device = "cuda"
            self._vram_total_gb = round(total_vram, 2)
            self._vram_free_gb = self._query_vram_free()
            self._gpu_available = True
            self._gpu_name = device_name
            self._gpu_tier = _compute_gpu_tier(self._gpu_architecture, self._vram_total_gb, self._backend)

        except Exception as exc:
            logger.debug("MLDeviceManager: CUDA/ROCm detection fehlgeschlagen: %s — trying ONNX-only", exc)
            self._detect_rocm_onnx_only()

    def _detect_rocm_onnx_only(self) -> None:
        """Fallback: detect ROCm via ONNX Runtime providers (no torch needed)."""
        try:
            import onnxruntime as ort  # type: ignore[import]

            providers = ort.get_available_providers()
            if "ROCMExecutionProvider" in providers:
                self._backend = GPUBackend.ROCM
                self._torch_gpu_device = "cpu"
                self._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
                self._gpu_available = True
                self._gpu_name = "AMD GPU (ONNX ROCm, no torch)"
                self._vram_total_gb = 4.0  # conservative estimate
                self._vram_free_gb = 4.0
                logger.info("MLDeviceManager: ONNX ROCm provider erkannt — ONNX models will use GPU")
            else:
                logger.debug("MLDeviceManager: ONNX ROCm provider not verfuegbar")
        except ImportError:
            logger.debug("MLDeviceManager: onnxruntime not installed")
        except Exception as exc:
            logger.debug("MLDeviceManager: ONNX ROCm detection error: %s", exc)

    def _detect_migraphx_bridge(self) -> None:
        """Detect AMD GPU via standalone MIGraphX bridge (no ONNX Runtime ROCm EP required).

        Uses the custom libmigraphx_bridge.so that wraps AMD's MIGraphX C API.
        This is the primary GPU path for gfx1100 (RDNA3 / RX 7900 XTX) because
        the standard ONNX Runtime ROCm wheel lacks MIOpen kernels for this arch.

        On success sets:
          - _backend = GPUBackend.ROCM
          - _ort_gpu_providers = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
          - _gpu_available = True
        """
        try:
            from backend.core.migraphx_adapter import get_migraphx_device_info, is_migraphx_available

            if not is_migraphx_available():
                return

            device_info = get_migraphx_device_info()
            with self._lock:
                self._backend = GPUBackend.ROCM
                self._ort_gpu_providers = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
                self._gpu_available = True
                self._gpu_name = device_info.get("name", "AMD GPU (MIGraphX)")
                self._vram_total_gb = device_info.get("vram_gb", 8.0)
                self._vram_free_gb = self._vram_total_gb * 0.85
                # §Rev. 2026-08-16: _gpu_architecture MUSS ein AMDArchitecture-Enum sein —
                # gpu_status_summary() greift auf .value zu. Die frühere String-Zuweisung
                # (device_info["arch"]) war ein latenter AttributeError.
                _arch_str = str(device_info.get("arch", "") or "")
                _arch_enum = _detect_amd_architecture(self._gpu_name)
                if _arch_enum is AMDArchitecture.UNKNOWN and _arch_str:
                    _arch_enum = _detect_amd_architecture(_arch_str)
                self._gpu_architecture = _arch_enum
                # Tier 2: RDNA3 dGPU — capable, >8GB
                self._gpu_tier = GPUTier.TIER_2
            logger.info(
                "MLDeviceManager: MIGraphX-Bridge erkannt — %s (%s VRAM=%.1f GB) → ONNX GPU aktiv",
                self._gpu_name,
                self._gpu_architecture,
                self._vram_total_gb,
            )
        except ImportError:
            logger.debug("MLDeviceManager: MIGraphX bridge adapter not installed")
        except Exception as exc:
            logger.debug("MLDeviceManager: MIGraphX bridge detection error: %s", exc)

    def _probe_rocm_onnx_pad(self) -> None:
        """Probe ROCm ONNX Runtime with a minimal GPU-compute op.

        Some AMD APU/iGPU targets (e.g. gfx1103, Phoenix/Hawk Point) ship with
        ROCm ORT builds that do not include kernels compiled for that specific arch.
        Running even a trivial model then raises hipErrorInvalidDeviceFunction.
        This probe runs once at startup and blacklists ROCMExecutionProvider for all
        ONNX sessions if inference fails — preventing per-plugin retry cycles that
        exhaust the Wall-Time budget (§0d, §2.47).

        The probe model is a minimal Relu(float[1]) graph serialised as raw protobuf
        bytes to avoid any dependency on the ``onnx`` package.
        """
        try:
            import numpy as np
            import onnxruntime as ort  # type: ignore[import]

            # Minimal ONNX model: x (float[1]) → Relu → y (float[1])
            # Opset 7, IR version 7 — hand-computed protobuf bytes (no onnx dep).
            # Verified: 63 bytes total.
            #   ir_version=7, opset_import {version=7},
            #   graph { node {Relu x→y}, input float[1], output float[1] }
            # NodeProto field mapping: input=1, output=2, op_type=4 (not 3!)
            # TensorShapeProto field mapping: dim=1 (not 2!)
            _MODEL_BYTES: bytes = (
                b"\x08\x07"  # ir_version: 7
                b"\x42\x02\x10\x07"  # opset_import: {version: 7}
                b"\x3a\x37"  # graph: len=55
                b"\x0a\x0c"  # node: len=12
                b"\x0a\x01\x78"  # input: "x"
                b"\x12\x01\x79"  # output: "y"
                b"\x22\x04\x52\x65\x6c\x75"  # op_type: "Relu" (field 4)
                b"\x12\x05\x70\x72\x6f\x62\x65"  # name: "probe"
                b"\x5a\x0f"  # input ValueInfo: len=15
                b"\x0a\x01\x78"  # name: "x"
                b"\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01"  # dtype: float[1]
                b"\x62\x0f"  # output ValueInfo: len=15
                b"\x0a\x01\x79"  # name: "y"
                b"\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01"  # dtype: float[1]
            )

            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 4  # silent
            sess = ort.InferenceSession(
                _MODEL_BYTES,
                sess_opts=sess_opts,
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
            )
            inp = np.zeros((1,), dtype=np.float32)
            sess.run(None, {"x": inp})
            # Verify ROCMExecutionProvider was actually activated (not silent CPU fallback)
            active = sess.get_providers()
            if "ROCMExecutionProvider" not in active:
                logger.info(
                    "MLDeviceManager: ROCm ONNX Probe — Provider silently fell back to CPU "
                    "(ROCm libs nicht verfuegbar or not in LD_LIBRARY_PATH) → ORT CPU-only"
                )
                with self._lock:
                    self._ort_gpu_providers = ["CPUExecutionProvider"]
                    self._gpu_disabled_plugins.update(_HEAVY_ML_PLUGINS)
                    self._ort_gpu_compatible_plugins.clear()
            else:
                logger.info("MLDeviceManager: ROCm ONNX Probe OK — ROCMExecutionProvider aktiv")
                # Opportunistic MIGraphX upgrade: if MIGraphXExecutionProvider is
                # available, prepend it as highest-priority provider. ORT will use it
                # for ops it supports and fall back to ROCMExecutionProvider otherwise.
                # MIGraphX is AMD's graph-compiler backend — measurably faster than
                # ROCMExecutionProvider on RDNA2/3 for many inference graphs.
                _avail = ort.get_available_providers()
                if "MIGraphXExecutionProvider" in _avail:
                    with self._lock:
                        self._ort_gpu_providers = [
                            "MIGraphXExecutionProvider",
                            "ROCMExecutionProvider",
                            "CPUExecutionProvider",
                        ]
                    logger.info(
                        "MLDeviceManager: MIGraphXExecutionProvider verfügbar — "
                        "als primärer ORT-Provider eingetragen (ROCM als Ersatzpfad)"
                    )

        except Exception as exc:
            exc_str = str(exc).lower()
            _hip_hints = (
                "hiperror",
                "hip error",
                "invaliddevicefunction",
                "invalid device function",
                "no kernel",
                "kernel not found",
            )
            if any(h in exc_str for h in _hip_hints):
                logger.warning(
                    "MLDeviceManager: ROCm ONNX Probe FEHLGESCHLAGEN (%s) — "
                    "ROCMExecutionProvider für alle ONNX-Plugins deaktiviert (CPU-Ersatzpfad)",
                    exc,
                )
                with self._lock:
                    # Downgrade ORT providers to CPU-only.
                    self._ort_gpu_providers = ["CPUExecutionProvider"]
                    # Disable GPU for all ONNX-heavy plugins so is_ort_gpu_supported()
                    # returns False immediately — no per-plugin retry overhead.
                    self._gpu_disabled_plugins.update(_HEAVY_ML_PLUGINS)
                    self._ort_gpu_compatible_plugins.clear()
            else:
                # Unexpected probe error (e.g. import issue) — log but don't blacklist.
                logger.debug("MLDeviceManager: ROCm ONNX Probe-Fehler (nicht HIP): %s", exc)

    def _detect_directml(self) -> None:
        """Erkennt DirectML on Windows via onnxruntime-directml."""
        try:
            import onnxruntime as _ort  # type: ignore[import]

            available = _ort.get_available_providers()
            if "DmlExecutionProvider" not in available:
                logger.debug(
                    "MLDeviceManager: DmlExecutionProvider not verfuegbar "
                    "(onnxruntime-directml not installed?). Providers: %s",
                    available,
                )
                return

            self._backend = GPUBackend.DIRECTML
            self._ort_gpu_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self._gpu_available = True
            self._vram_total_gb = self._query_vram_directml()
            self._vram_free_gb = self._vram_total_gb  # assume clear at startup

            # Query real GPU name for architecture detection
            gpu_name = self._query_gpu_name_windows()
            self._gpu_name = gpu_name if gpu_name else "DirectML (AMD/DX12)"

            # Architecture & tier detection (same pipeline as ROCm)
            self._gpu_architecture = _detect_amd_architecture(self._gpu_name)
            self._gpu_tier = _compute_gpu_tier(self._gpu_architecture, self._vram_total_gb, self._backend)

            logger.info(
                "MLDeviceManager: DirectML GPU arch=%s tier=%s name=%s VRAM=%.1f GB",
                self._gpu_architecture.value,
                self._gpu_tier.name,
                self._gpu_name,
                self._vram_total_gb,
            )

            # Optional: torch-directml enables PyTorch models on DirectML.
            # Without it, only ONNX models get DirectML acceleration.
            try:
                import torch_directml  # type: ignore[import]  # pylint: disable=unused-import  # Side-Effect-Import: registriert DML-Device in PyTorch

                self._torch_gpu_device = "dml"
                logger.info("MLDeviceManager: torch-directml verfuegbar — PyTorch models can use DML device")
            except ImportError:
                self._torch_gpu_device = "cpu"
                logger.info(
                    "MLDeviceManager: torch-directml not installed — ONNX models use DML, PyTorch models stay on CPU"
                )

        except Exception as exc:
            logger.debug("MLDeviceManager: DirectML detection fehlgeschlagen: %s", exc)

    # ── VRAM query helpers ────────────────────────────────────────────────

    def _query_gpu_name_windows(self) -> str:
        """Query the GPU display name on Windows via WMIC.

        Returns the real GPU name (e.g. 'AMD Radeon RX 7900 XTX') so that
        _detect_amd_architecture() can determine the correct architecture.
        Returns empty string when WMIC is unavailable (non-Windows or no admin).
        """
        try:
            import subprocess  # nosec B404 — WMIC read-only hardware query

            result = subprocess.run(  # nosec B603 B607
                ["wmic", "path", "Win32_VideoController", "get", "Caption", "/value"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,  # WMIC-Probe: Fehlercode irrelevant, Ausgabe wird ausgewertet
            )
            for line in result.stdout.splitlines():
                if "Caption" in line and "=" in line:
                    name = line.split("=", 1)[1].strip()
                    if name:
                        return name
        except Exception as e:
            logger.warning("ml_device_manager.py::_query_gpu_name_windows Ersatzpfad: %s", e)
        return ""

    def _query_vram_free(self) -> float:
        """Query free VRAM in GB via torch.cuda.mem_get_info(device=0).

        Works for both NVIDIA CUDA and AMD ROCm (ROCm exposes CUDA API).
        """
        try:
            import torch  # type: ignore[import]

            free_bytes, _ = torch.cuda.mem_get_info(0)
            return round(free_bytes / (1024**3), 2)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("ml_device_manager.py::_query_vram_free Ersatzpfad: %s", e)
            return self._vram_total_gb  # assume empty on query failure

    def _query_vram_directml(self) -> float:
        """Schätzt VRAM for DirectML via WMIC (Windows) or conservative default."""
        try:
            import subprocess  # nosec B404 — WMIC read-only hardware query

            result = subprocess.run(  # nosec B603 B607
                ["wmic", "path", "Win32_VideoController", "get", "AdapterRAM", "/value"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,  # WMIC-Probe: Fehlercode irrelevant, Ausgabe wird ausgewertet
            )
            for line in result.stdout.splitlines():
                if "AdapterRAM" in line and "=" in line:
                    raw = line.split("=", 1)[1].strip()
                    if raw.isdigit():
                        return round(int(raw) / (1024**3), 2)
        except Exception as e:
            logger.warning("ml_device_manager.py::_query_vram_directml Ersatzpfad: %s", e)
        return 4.0  # conservative default when WMIC is unavailable

    # ── Public API ────────────────────────────────────────────────────────

    def get_gpu_backend(self) -> GPUBackend:
        """Gibt the detected GPU backend enum value zurück."""
        return self._backend

    def is_gpu_available(self) -> bool:
        """True when GPU acceleration is available on this system."""
        return self._gpu_available

    def is_heavy_plugin(self, plugin_name: str) -> bool:
        """True when *plugin_name* is eligible for GPU dispatch."""
        return plugin_name in _HEAVY_ML_PLUGINS

    @property
    def gpu_architecture(self) -> AMDArchitecture:
        """Gibt the detected AMD GPU architecture family zurück."""
        return self._gpu_architecture

    @property
    def gpu_tier(self) -> GPUTier:
        """Gibt the GPU capability tier (1=best, 4=CPU-only recommended) zurück."""
        return self._gpu_tier

    @property
    def gpu_name(self) -> str:
        """Gibt the GPU device name (e.g. 'AMD Radeon RX 7900 XTX') zurück."""
        return self._gpu_name

    def get_torch_device(self, plugin_name: str = "") -> str:
        """Gibt PyTorch device string for *plugin_name* zurück.

        Heavy plugins receive the GPU device string; all others receive ``"cpu"``.
        DirectML without torch-directml always returns ``"cpu"`` for PyTorch models.
        Tier-based exclusion: Tier 3/4 GPUs exclude VRAM-heavy plugins.
        """
        if not self._gpu_available:
            return "cpu"
        if plugin_name and plugin_name not in _HEAVY_ML_PLUGINS:
            return "cpu"
        if plugin_name and plugin_name in self._gpu_disabled_plugins:
            return "cpu"
        # Tier-based exclusion for plugins too large for the GPU
        if plugin_name and self._is_tier_excluded(plugin_name):
            return "cpu"
        # DirectML requires separate torch-directml package for PyTorch tensors.
        if self._backend == GPUBackend.DIRECTML and self._torch_gpu_device == "cpu":
            return "cpu"
        return self._torch_gpu_device

    def get_ort_providers(self, plugin_name: str = "") -> list[str]:
        """Gibt ONNX Runtime provider list for *plugin_name* zurück.

        Heavy plugins get the GPU provider (with CPU fallback); others get CPU-only.
        On ROCm, fp16-eligible plugins automatically receive fp16-optimised providers
        when the GPU tier supports it (Tier 1–3).
        Returns a defensive copy of the internal list.
        """
        if not self.is_ort_gpu_supported(plugin_name):
            return ["CPUExecutionProvider"]
        # Auto-fp16: if plugin is fp16-eligible and tier allows it, use fp16 providers
        if (
            self._backend == GPUBackend.ROCM
            and plugin_name
            and plugin_name in _FP16_ELIGIBLE_PLUGINS
            and _TIER_VRAM_PARAMS.get(self._gpu_tier, {}).get("fp16_auto", 0.0) >= 1.0
        ):
            return self.get_ort_providers_fp16(plugin_name)
        return list(self._ort_gpu_providers)

    def is_ort_gpu_supported(self, plugin_name: str = "") -> bool:
        """Gibt True if ORT-GPU should be used for *plugin_name* in this session zurück.

        Decision is conservative: GPU must be available, plugin must be in the heavy
        GPU-eligible set, and the plugin must not be session-disabled due to prior
        incompatibility/runtime failures. Tier-based exclusion also applies.
        """
        if not self._gpu_available:
            return False
        if plugin_name and plugin_name not in _HEAVY_ML_PLUGINS:
            return False
        if plugin_name and plugin_name in self._gpu_disabled_plugins:
            return False
        if plugin_name and plugin_name not in self._ort_gpu_compatible_plugins:
            return False
        if plugin_name and self._is_tier_excluded(plugin_name):
            return False
        # RX 5xx-Serie (GCN4): DirectML/ONNX-DML wird nicht unterstützt → CPU-only.
        if self._backend == GPUBackend.DIRECTML and self._gpu_architecture == AMDArchitecture.GCN4:
            return False
        return True

    def _is_tier_excluded(self, plugin_name: str) -> bool:
        """Gibt True if *plugin_name* is excluded from GPU on the current tier zurück."""
        if self._gpu_tier == GPUTier.TIER_4:
            return plugin_name in _TIER4_GPU_EXCLUDE
        if self._gpu_tier == GPUTier.TIER_3:
            return plugin_name in _TIER3_GPU_EXCLUDE
        return False

    def mark_ort_gpu_unsupported(self, plugin_name: str, reason: str = "") -> None:
        """Deaktiviert ORT-GPU for *plugin_name* for the current session.

        Use this when session creation/inference proves provider incompatibility.
        Subsequent calls to ``get_ort_providers(plugin_name)`` will return CPU-only.
        """
        if not plugin_name:
            return
        with self._lock:
            self._gpu_disabled_plugins.add(plugin_name)
            self._ort_gpu_compatible_plugins.discard(plugin_name)
        if reason:
            logger.info("MLDeviceManager: ORT-GPU für %s deaktiviert (%s) → CPU", plugin_name, reason)
        else:
            logger.info("MLDeviceManager: ORT-GPU für %s deaktiviert → CPU", plugin_name)

    def get_vram_free_gb(self) -> float:
        """Gibt current estimated free VRAM in GB (refreshed on ROCm via HW query) zurück."""
        with self._lock:
            if self._backend == GPUBackend.ROCM:
                self._vram_free_gb = self._query_vram_free()
            return self._vram_free_gb

    def try_allocate_vram(self, plugin_name: str, size_gb: float) -> bool:
        """Attempt to reserve *size_gb* GB of VRAM for *plugin_name*.

        Returns True  → proceed with GPU inference.
        Returns False → use CPU fallback (VRAM budget would be exceeded).

        Idempotent: already-allocated plugins return True immediately.
        """
        if not self._gpu_available:
            return False

        with self._lock:
            if self._vram_allocated.get(plugin_name, 0.0) > 0.0:
                return True  # already allocated

            # Refresh VRAM free estimate before deciding.
            if self._backend == GPUBackend.ROCM:
                self._vram_free_gb = self._query_vram_free()

            already_used = sum(v for k, v in self._vram_allocated.items())
            # Tier-adaptive VRAM budget parameters
            tier_params = _TIER_VRAM_PARAMS.get(self._gpu_tier, _TIER_VRAM_PARAMS[GPUTier.TIER_4])
            max_usage_ratio = tier_params["max_usage_ratio"]
            min_free_mb = tier_params["min_free_mb"]
            max_usable = self._vram_total_gb * max_usage_ratio

            if already_used + size_gb > max_usable:
                logger.warning(
                    "MLDeviceManager: VRAM Grenze exceeded for %s (%.2f GB)"
                    " — used %.2f / %.2f GB (tier=%s) → CPU Ersatzpfad",
                    plugin_name,
                    size_gb,
                    already_used,
                    max_usable,
                    self._gpu_tier.name,
                )
                return False

            required_with_floor = size_gb + (min_free_mb / 1024.0)
            effective_free = self._vram_free_gb - already_used
            if effective_free < required_with_floor:
                logger.warning(
                    "MLDeviceManager: insufficient VRAM for %s — "
                    "need %.2f GB (incl. %.0f MB floor), have %.2f GB (tier=%s) → CPU Ersatzpfad",
                    plugin_name,
                    required_with_floor,
                    min_free_mb,
                    effective_free,
                    self._gpu_tier.name,
                )
                return False

            self._vram_allocated[plugin_name] = size_gb
            logger.info(
                "MLDeviceManager: VRAM allocated %s=%.2f GB (total used=%.2f / %.2f GB total)",
                plugin_name,
                size_gb,
                sum(self._vram_allocated.values()),
                self._vram_total_gb,
            )
            return True

    def release_vram(self, plugin_name: str) -> None:
        """Release VRAM budget reservation for *plugin_name* (call on model unload)."""
        with self._lock:
            freed = self._vram_allocated.pop(plugin_name, 0.0)
            if freed > 0.0:
                logger.debug("MLDeviceManager: VRAM released %s=%.2f GB", plugin_name, freed)

    def gpu_status_summary(self) -> dict[str, Any]:
        """Gibt a status snapshot dict for UI display and diagnostics zurück."""
        with self._lock:
            return {
                "backend": self._backend.value,
                "gpu_available": self._gpu_available,
                "gpu_name": self._gpu_name,
                "gpu_architecture": self._gpu_architecture.value,
                "gpu_tier": self._gpu_tier.name,
                "vram_total_gb": self._vram_total_gb,
                "vram_free_gb": self._vram_free_gb,
                "vram_allocated_gb": sum(self._vram_allocated.values()),
                "allocated_plugins": dict(self._vram_allocated),
                "gpu_errors": dict(self._gpu_errors),
                "gpu_disabled_plugins": list(self._gpu_disabled_plugins),
                "fp16_auto": _TIER_VRAM_PARAMS.get(self._gpu_tier, {}).get("fp16_auto", 0.0) >= 1.0,
            }

    def report_gpu_error(self, plugin_name: str, exc: Exception) -> None:
        """Record a runtime GPU inference failure and release VRAM budget.

        Called by plugins when GPU inference fails at runtime so the manager
        can track error frequency and proactively disable GPU for that plugin
        in future sessions (telemetry only — no pipeline impact).
        """
        with self._lock:
            self._gpu_errors[plugin_name] = self._gpu_errors.get(plugin_name, 0) + 1
            count = self._gpu_errors[plugin_name]
            lower_msg = str(exc).lower()
            if any(hint in lower_msg for hint in _GPU_UNSUPPORTED_ERROR_HINTS):
                # Provider/kernel incompatibility: disable immediately.
                self._gpu_disabled_plugins.add(plugin_name)
                self._ort_gpu_compatible_plugins.discard(plugin_name)
                logger.info(
                    "MLDeviceManager: GPU-Inkompatibilität für %s erkannt (%s) — sofort CPU-Ersatzpfad",
                    plugin_name,
                    exc,
                )
                return
            logger.warning(
                "MLDeviceManager: GPU-Inferenz-Fehler #%d für %s (%s)",
                count,
                plugin_name,
                exc,
            )
            # Release VRAM budget so subsequent CPU load fits
            freed = self._vram_allocated.pop(plugin_name, 0.0)
            if freed > 0.0:
                self._vram_free_gb = min(self._vram_total_gb, self._vram_free_gb + freed)
            # After 3 consecutive failures, disable GPU for this plugin this session
            if count >= 3:
                self._gpu_disabled_plugins.add(plugin_name)
                logger.info(
                    "MLDeviceManager: GPU f\u00fcr %s nach %d Fehlern deaktiviert (Sitzung)",
                    plugin_name,
                    count,
                )

    # ── AMD ROCm performance extensions ──────────────────────────────────

    def is_fp16_eligible(self, plugin_name: str) -> bool:
        """True when *plugin_name* supports fp16 inference on AMD ROCm.

        fp16 halves VRAM usage and typically doubles ONNX/PyTorch inference throughput
        on AMD GPUs with no perceptible quality difference for bounded audio signals.
        Returns False on CPU-only or DirectML (DML handles precision internally).
        """
        return (
            self._backend == GPUBackend.ROCM
            and self._gpu_available
            and plugin_name in _FP16_ELIGIBLE_PLUGINS
            and plugin_name not in self._gpu_disabled_plugins
        )

    def get_ort_providers_fp16(self, plugin_name: str = "") -> list[str]:
        """Gibt ORT providers with fp16 precision hint for AMD ROCm zurück.

        Uses ROCMExecutionProvider with memory-efficient options when eligible;
        falls back to standard providers otherwise (always CPU-safe).
        """
        if not self.is_fp16_eligible(plugin_name):
            return self.get_ort_providers(plugin_name)
        return [  # type: ignore[list-item]  # ORT accepts (str, dict) tuples as providers
            (  # type: ignore[list-item]
                "ROCMExecutionProvider",
                {
                    "device_id": 0,
                    "gpu_mem_limit": int(
                        self._vram_total_gb
                        * _TIER_VRAM_PARAMS.get(self._gpu_tier, _TIER_VRAM_PARAMS[GPUTier.TIER_4])["max_usage_ratio"]
                        * 1024**3
                    ),
                    "arena_extend_strategy": "kSameAsRequested",
                    # NOTE: enable_cuda_graph is a CUDA-only option — not supported by ROCMExecutionProvider
                    # (onnxruntime >= 1.20 raises FAIL: Unknown provider option: "enable_cuda_graph")
                },
            ),
            "CPUExecutionProvider",
        ]

    def force_cpu_fallback(self, reason: str = "") -> None:
        """§v10.304.23: GPU global deaktivieren — alle Plugins auf CPU.

        Wird aufgerufen wenn der Bridge-Warmup einen GPU-Timeout detektiert.
        Verhindert dass nachfolgende Phasen/Plugins erneut versuchen die
        defekte GPU zu nutzen.
        """
        with self._lock:
            if self._gpu_available:
                self._gpu_available = False
                self._backend = GPUBackend.NONE
                self._torch_gpu_device = "cpu"
                self._ort_gpu_providers = ["CPUExecutionProvider"]
                logger.warning(
                    "MLDeviceManager: GPU deaktiviert (%s) — CPU-Ersatzpfad für alle Plugins",
                    reason or "unknown",
                )
                logger.info("══════════════════════════════════════════════")
                logger.info("GPU STATUS: DEAKTIVIERT — %s", reason or "unknown")
                logger.info("══════════════════════════════════════════════")

    def warmup_rocm(self) -> bool:
        """Initialisiert ROCm-Laufzeitumgebung mit Dummy-Inferenz (Timeout-geschützt).

        Running this once during application startup eliminates the ~500 ms–2 s first-
        inference latency caused by ROCm kernel compilation (HIP JIT). Returns True on
        success, False if warmup failed or ROCm is not active.

        §v10.304.30: torch.zeros("cuda") kann auf defekten ROCm-Installationen
        hängen. Geschützt durch ThreadPoolExecutor mit 10 s Timeout.
        §v10.304.31: self._lock wird NICHT während der GPU-Operation gehalten —
        sonst Deadlock wenn force_cpu_fallback denselben Lock braucht.
        """
        if self._backend != GPUBackend.ROCM or not self._gpu_available:
            return False
        try:
            import torch  # type: ignore[import]

            def _warmup_op() -> bool:
                try:
                    _t = torch.zeros(_ROCM_WARMUP_TENSOR_LEN, dtype=torch.float32, device="cuda")
                    _t_back = _t.cpu()
                    del _t, _t_back
                    torch.cuda.synchronize()
                    torch.set_num_interop_threads(_ROCM_TORCH_THREADS)
                    return True
                except Exception:
                    return False

            from concurrent.futures import ThreadPoolExecutor as _WarmupTPE

            _pool = _WarmupTPE(max_workers=1, thread_name_prefix="rocm_warmup")
            _fut = _pool.submit(_warmup_op)
            _ok = _fut.result(timeout=10.0)
            _pool.shutdown(wait=False)
            if _ok:
                logger.info("MLDeviceManager: ROCm warmup vollstaendig — HIP runtime ready (%s)", self._gpu_name)
                return True
        except Exception as exc:
            logger.warning("MLDeviceManager: ROCm warmup fehlgeschlagen/Zeitlimit — GPU deaktiviert: %s", exc)
            self.force_cpu_fallback("warmup_timeout")
        return False

    def pin_tensor_rocm(self, array: Any) -> Any:
        """Gibt a pinned-memory copy of *array* for zero-copy CPU→GPU transfers zurück.

        Only active on ROCm; returns the original array unchanged on all other backends.
        Pinned memory avoids an extra memcopy during `.to('cuda')` and reduces plugin
        inference latency by 10–25 % for large tensors (FlashSR, MERT, etc.).
        """
        if self._backend != GPUBackend.ROCM or not self._gpu_available:
            return array
        try:
            import numpy as np
            import torch  # type: ignore[import]

            if isinstance(array, np.ndarray):
                t = torch.from_numpy(array)
            elif isinstance(array, torch.Tensor):
                t = array
            else:
                return array
            if not t.is_pinned():
                t = t.pin_memory()
            return t
        except Exception as exc:
            logger.debug("pin_tensor_rocm fehlgeschlagen (unkritisch, using Originalsignal): %s", exc)
            return array


def ort_run_with_cpu_fallback(
    session: Any,
    feeds: dict[str, Any],
    *,
    output_names: Any = None,
    rebuild_cpu_factory: Callable[[], Any] | None = None,
    label: str = "ONNX",
) -> Any:
    """§ROCm-Fallback: run() mit einmaligem CPU-Retry bei MIOpen-Kernel-Fehlern.

    MIOpen kann auf manchen ROCm-Versionen Kernels nicht bauen
    ("Code object build failed", v. a. AveragePool/Pooling-Nodes) — die Inferenz
    wirft dann, obwohl das Modell gesund ist. Diese Helper-Funktion ist DIE
    zentrale Stelle für den CPU-Fallback (Spec 23-Kontext, 2026-08-16):
    Plugins bauen per ``rebuild_cpu_factory`` dieselbe Session CPU-only neu und
    wiederholen die Inferenz EINMAL.

    Args:
        session: onnxruntime.InferenceSession (oder kompatibles Mock-Objekt).
        feeds: input_name → Tensor-Dict wie für session.run().
        output_names: Optional; entspricht session.run(output_names, feeds).
        rebuild_cpu_factory: Callable, das eine CPU-only Session für denselben
            Modellpfad baut; None → kein Retry, Exception propagiert.
        label: Log-Präfix für die §V6-Warnung.

    Returns:
        session.run(...)-Ergebnis.
    """
    try:
        return session.run(output_names, feeds)
    except Exception as exc:
        if rebuild_cpu_factory is None:
            raise
        # §V6 (copilot-instructions.md): ML→DSP/CPU-Fallback mit Warnung + Begründung.
        logger.warning("%s: ONNX-Inferenz fehlgeschlagen (%s) — CPU-Fallback-Retry", label, exc)
        cpu_session = rebuild_cpu_factory()
        if cpu_session is None:
            raise
        return cpu_session.run(output_names, feeds)
