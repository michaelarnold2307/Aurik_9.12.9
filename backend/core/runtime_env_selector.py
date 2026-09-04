"""Laufzeitumgebungs-Auswahl — Backend-agnostisch (CUDA + ROCm + CPU).

Probes available virtual environments for GPU acceleration.
Prefers GPU (.venv_gpu) over CPU (.venv_aurik).
Supports: NVIDIA CUDA, AMD ROCm, CPU-only fallback.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROBE_CODE = r"""
import json, sys
result = {"torch_import": False, "torch_cuda_available": False, "torch_hip": None,
          "torch_cuda_version": None, "onnxruntime_import": False, "ort_providers": []}
try:
    import torch
    result["torch_import"] = True
    result["torch_cuda_available"] = bool(torch.cuda.is_available())
    result["torch_hip"] = getattr(getattr(torch, "version", None), "hip", None)
    result["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
except Exception:
    # expected: torch may not be installed in subprocess probe
    pass
try:
    import onnxruntime as ort
    result["onnxruntime_import"] = True
    result["ort_providers"] = list(ort.get_available_providers())
except Exception:
    # expected: onnxruntime may not be installed in subprocess probe
    pass
sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
"""


@dataclass(frozen=True)
class RuntimeProbe:
    python_path: Path
    torch_import: bool
    torch_cuda_available: bool
    torch_hip: str | None
    torch_cuda_version: str | None
    onnxruntime_import: bool
    ort_providers: tuple[str, ...]

    @property
    def has_rocm(self) -> bool:
        return (
            self.torch_import
            and self.torch_cuda_available
            and bool(self.torch_hip)
            and self.onnxruntime_import
            and "ROCMExecutionProvider" in self.ort_providers
        )

    @property
    def has_cuda(self) -> bool:
        return (
            self.torch_import
            and self.torch_cuda_available
            and bool(self.torch_cuda_version)
            and self.onnxruntime_import
            and "CUDAExecutionProvider" in self.ort_providers
        )

    @property
    def has_gpu(self) -> bool:
        return self.has_cuda or self.has_rocm


def _candidate_paths(repo_root: Path) -> list[Path]:
    """Backend-agnostic venv candidates. GPU first, CPU fallback.

    Reihenfolge (Rev. 2026-08-16): venv_rocm72 (ROCm 7.2.4, GPU-Gate-verifiziert) →
    venv_rocm (ROCm 6.2, Legacy) → Repo-venvs → CPU-Fallback.
    """
    _aurik_home = Path.home() / ".local" / "share" / "aurik"
    return [
        _aurik_home / "venv_rocm72" / "bin" / "python",
        _aurik_home / "venv_rocm" / "bin" / "python",
        repo_root / ".venv_gpu" / "bin" / "python",
        repo_root / ".venv_gpu" / "Scripts" / "python.exe",
        repo_root / ".venv_aurik" / "bin" / "python",
        repo_root / ".venv_aurik" / "Scripts" / "python.exe",
    ]


def probe_python_runtime(python_path: Path) -> RuntimeProbe | None:
    if not python_path.exists() or not os.access(python_path, os.X_OK):
        return None
    try:
        proc = subprocess.run(
            [str(python_path), "-c", _PROBE_CODE],
            check=True,
            capture_output=True,
            text=True,
            timeout=12,
        )
    except FileNotFoundError:
        logger.debug("GPU-Probe: Python-Pfad nicht gefunden: %s", python_path)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("GPU-Probe: Zeitlimit (12s) für %s — Python hängt oder Umgebung korrupt", python_path)
        return None
    except Exception:
        logger.warning("GPU-Probe fehlgeschlagen für %s", python_path, exc_info=True)
        return None
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError as e:
        logger.debug("§V6 GPU-Probe JSON-Dekodierung fehlgeschlagen für %s — None zurückgegeben: %s", python_path, e)
        return None
    return RuntimeProbe(
        python_path=python_path,
        torch_import=bool(payload.get("torch_import")),
        torch_cuda_available=bool(payload.get("torch_cuda_available")),
        torch_hip=payload.get("torch_hip"),
        torch_cuda_version=payload.get("torch_cuda_version"),
        onnxruntime_import=bool(payload.get("onnxruntime_import")),
        ort_providers=tuple(str(p) for p in (payload.get("ort_providers") or [])),
    )


def select_runtime_python(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    probes: list[RuntimeProbe] = []
    for candidate in _candidate_paths(root):
        probe = probe_python_runtime(candidate)
        if probe is not None:
            probes.append(probe)

    # Prefer GPU (CUDA or ROCm)
    for probe in probes:
        if probe.has_gpu:
            logger.info("runtime_env_selector: GPU venv selected — %s", probe.python_path)
            return probe.python_path

    # Fallback: CPU venv
    for probe in probes:
        if probe.python_path.parent.parent.name == ".venv_aurik":
            return probe.python_path

    # Ultimate fallback: current Python
    current = Path(sys.executable).resolve()
    if current.exists():
        return current
    return root / ".venv_aurik" / "bin" / "python"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[2]
    selected = select_runtime_python(repo_root)
    if args == ["--print-python"]:
        sys.stdout.write(f"{selected}\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
