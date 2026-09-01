"""GPU-Determinismus-Gate (§G5 (GEBOTE.md), Rev. 2026-08-16).

Beantwortet die §G5-Frage auf GPU-Hardware mit Messung statt Annahme:

  1. Basis (läuft überall): Der CPU-Pfad ist bit-deterministisch (§G5 (GEBOTE.md)).
  2. GPU-Capability (nur auf GPU-Hardware): Der MLDeviceManager meldet das
     Backend korrekt, und warmup_rocm_gpu() hängt nicht (§v10.304.30).
  3. GPU-Inferenz (nur auf GPU-Hardware): Toleranz-equal zum CPU-Pfad,
     NICHT bit-identisch — dokumentiert die §G5-Grenze auf GPU.

ROCm 7.x: Dieser Gate-Lauf auf der Zielhardware ist die Voraussetzung, um
einen neuen Versionskope im ml_device_manager zu deklarieren (siehe
Modul-Docstring „ROCm-Versionskope").

Auf CPU-only-Systemen werden die GPU-Tests mit Begründung geskippt.
"""

from __future__ import annotations

import numpy as np
import pytest

_GPU_SKIP_REASON = "Keine GPU verfügbar — GPU-Tests laufen nur auf GPU-CI-Hardware"


def _gpu_available() -> bool:
    """True wenn PyTorch eine CUDA/ROCm/DirectML-GPU sieht (defensiv, nie hängend)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _deterministic_dsp_chain(seed: int = 7) -> np.ndarray:
    """Kanonischer seeded DSP-Pfad (CPU) — §G5-Referenz ohne ML-Modelle."""
    from scipy.signal import istft, stft

    rng = np.random.RandomState(seed)
    t = np.arange(48000) / 48000
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.001 * rng.randn(48000)
    _, _, z_stft = stft(sig.astype(np.float64), fs=48000, nperseg=1024, noverlap=768)
    _, rec = istft(z_stft, fs=48000, nperseg=1024, noverlap=768)
    return np.asarray(rec, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. §G5-Basis: CPU-Bit-Determinismus (läuft überall)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g5_cpu_reference_bit_identical() -> None:
    """Zwei Läufe des kanonischen DSP-Pfads müssen bit-identisch sein (§G5 (GEBOTE.md))."""
    a = _deterministic_dsp_chain()
    b = _deterministic_dsp_chain()
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a, b), "CPU-Pfad ist nicht bit-deterministisch — §G5 (GEBOTE.md) verletzt"


@pytest.mark.unit
def test_g5_cpu_reference_bit_identical_after_seed_reset() -> None:
    """Gleicher Seed zweimal frisch gesetzt → bit-identisch (Seeds pro Session, §G5 (GEBOTE.md))."""
    a = _deterministic_dsp_chain(seed=11)
    b = _deterministic_dsp_chain(seed=11)
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_g5_cpu_reference_rejects_different_seed() -> None:
    """Sanity: Unterschiedliche Seeds müssen unterschiedliche Ausgaben liefern —
    sonst misst dieses Gate nichts (Rauschanteil wäre wegoptimiert)."""
    a = _deterministic_dsp_chain(seed=11)
    b = _deterministic_dsp_chain(seed=12)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 2. GPU-Capability (nur auf GPU-Hardware)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _gpu_available(), reason=_GPU_SKIP_REASON)
def test_gpu_manager_reports_backend() -> None:
    """Mit GPU muss der Manager ein GPU-Backend (ROCM/CUDA/DIRECTML) melden."""
    from backend.core.ml_device_manager import GPUBackend, get_ml_device_manager

    mgr = get_ml_device_manager()
    assert mgr._backend in (GPUBackend.ROCM, GPUBackend.CUDA, GPUBackend.DIRECTML), (
        f"GPU vorhanden, aber Manager meldet {mgr._backend}"
    )


@pytest.mark.unit
@pytest.mark.skipif(not _gpu_available(), reason=_GPU_SKIP_REASON)
def test_warmup_rocm_does_not_hang() -> None:
    """§v10.304.30-Regression: Der ROCm-Warmup darf nicht hängen (torch.zeros-Hang-Klasse)."""
    from backend.core.ml_device_manager import warmup_rocm_gpu

    result = warmup_rocm_gpu()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 3. GPU-Inferenz: Toleranz-equal, NICHT bit-identisch (§G5-Grenze)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _gpu_available(), reason=_GPU_SKIP_REASON)
def test_gpu_inference_tolerance_equal_to_cpu() -> None:
    """GPU-Inferenz muss numerisch gleichwertig (1e-5), aber darf NICHT als
    bit-identisch angenommen werden — deshalb führt Aurik export- und
    entscheidungskritische Pfade deterministisch auf der CPU (§G5 (GEBOTE.md))."""
    import torch

    dev = torch.device("cuda")
    x = torch.randn(8192, dtype=torch.float64)
    ref = torch.fft.rfft(x)  # complex128, CPU-Referenz
    gpu = torch.fft.rfft(x.to(dev)).cpu()  # complex128, GPU→CPU
    assert torch.allclose(ref, gpu, atol=1e-5, rtol=1e-5), "GPU-Ergebnis weicht zu stark von der CPU-Referenz ab"
