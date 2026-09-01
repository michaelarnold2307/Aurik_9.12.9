#!/usr/bin/env bash
# Aurik 10 — Startskript mit venv-Python (.venv_aurik, Python 3.10.12)
# GPU-Modus: .venv_gpu → venv_rocm72 (ROCm 7.2.4) → venv_rocm (ROCm 6.2, Legacy)
# wird automatisch erkannt
# Verwendung: ./run_aurik.sh [Argumente]
#   AURIK_FORCE_CPU=1  ./run_aurik.sh  — erzwingt CPU-only
#
# GPU-Unterstützung:
#   NVIDIA CUDA → .venv_gpu mit torch+cuda + onnxruntime-gpu
#   AMD ROCm    → .venv_gpu mit torch+rocm + onnxruntime-rocm
#   Kein GPU    → .venv_aurik (CPU-only)
#   Windows/AMD → DirectML in .venv_aurik (kein separates venv nötig)
#
# Hinweis für ROCm: GPU-venv sollte auf ext4 liegen, da ROCm GPU Code Objects
# per mmap() aus ELF-Sektionen geladen werden und FUSE/fuseblk (NTFS) dieses
# mmap nicht unterstützt → hipErrorInvalidDeviceFunction.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# §v10: __pycache__ vor jedem Start löschen
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
VENV_CPU="$SCRIPT_DIR/.venv_aurik/bin/python"
VENV_GPU="$SCRIPT_DIR/.venv_gpu/bin/python"
PID_FILE="$SCRIPT_DIR/temp_repro/aurik_gui.pid"
LOG_FILE="$SCRIPT_DIR/logs/aurik_frontend.out"

# Release-Default: MIOpen/ROCm-Logging auf Fehler beschränken
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-1}"

check_rocm_torchaudio_abi() {
    "$VENV_PYTHON" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"ROCM_STACK_ERR torch import failed: {exc}")
    raise SystemExit(10)

try:
    import torchaudio
except Exception as exc:
    print(f"ROCM_STACK_ERR torchaudio import failed: {exc}")
    raise SystemExit(11)

torch_ver = str(getattr(torch, "__version__", ""))
audio_ver = str(getattr(torchaudio, "__version__", ""))
torch_build = torch_ver.split("+", 1)[1] if "+" in torch_ver else ""
audio_build = audio_ver.split("+", 1)[1] if "+" in audio_ver else ""

if torch_build and audio_build and torch_build != audio_build:
    print(
        "ROCM_STACK_ERR build mismatch: "
        f"torch={torch_ver} torchaudio={audio_ver}"
    )
    raise SystemExit(12)

print(f"ROCM_STACK_OK torch={torch_ver} torchaudio={audio_ver}")
PY
}

repair_rocm_torchaudio() {
    if [[ ! -x "$PIP_GPU" ]]; then
        echo "ROCM_STACK_ERR pip im ROCm-venv fehlt: $PIP_GPU" >&2
        return 1
    fi

    local torch_version rocm_tag
    torch_version="$($VENV_PYTHON - <<'PY'
import torch
print(getattr(torch, "__version__", ""))
PY
)"

    if [[ -z "$torch_version" || "$torch_version" != *+rocm* ]]; then
        echo "ROCM_STACK_ERR keine ROCm-Torch-Version erkannt: $torch_version" >&2
        return 1
    fi

    rocm_tag="${torch_version#*+}"
    echo "ROCM_STACK_REPAIR installiere torchaudio==$torch_version via $rocm_tag ..."
    "$PIP_GPU" install --upgrade --index-url "https://download.pytorch.org/whl/$rocm_tag" \
        "torchaudio==$torch_version"
}

# GPU-Erkennung: ROCm-venv (ext4) + KFD-Device vorhanden und nicht explizit deaktiviert
# Prüft .venv_gpu (neu), venv_rocm72 (ROCm 7.2.4) und venv_rocm (legacy 6.2)
_GPU_PYTHON="$VENV_GPU"
if [[ ! -x "$_GPU_PYTHON" ]]; then
    _GPU_PYTHON="$HOME/.local/share/aurik/venv_rocm72/bin/python"  # ROCm 7.2.4 (Rev. 2026-08-16)
fi
if [[ ! -x "$_GPU_PYTHON" ]]; then
    _GPU_PYTHON="$HOME/.local/share/aurik/venv_rocm/bin/python"  # ROCm 6.2 (Legacy-Fallback)
fi
if [[ "${AURIK_FORCE_CPU:-0}" != "1" && -x "$_GPU_PYTHON" && -e "/dev/kfd" ]]; then
    VENV_PYTHON="$_GPU_PYTHON"
    PIP_GPU="$(dirname "$VENV_PYTHON")/pip"
    _GPU_MODE="ROCm (AMD GPU)"
    # ORT's libonnxruntime_providers_rocm.so benötigt ROCm-Laufzeitbibliotheken
    # (ROCm 7.2.4: libhipblas.so.3, libhipfft.so.0; Legacy 6.2: libhipblas.so.2).
    # Diese liegen im PyTorch-lib-Verzeichnis des gewählten ROCm-venv (ext4) —
    # robust aus dem tatsächlichen torch des selektierten venv ableiten (3.10-kompatibel).
    _TORCH_LIB="$("$VENV_PYTHON" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")' 2>/dev/null || true)"
    if [[ -d "$_TORCH_LIB" ]]; then
        export LD_LIBRARY_PATH="${_TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # .pth-Bridge: aurik_bridge.pth im ROCm-venv-Site-Packages verweist auf venv_aurik-Pakete.
    # .pth-Dateien werden NACH den eigenen Site-Packages geladen → ROCm-torch hat Vorrang.
    set +e
    check_rocm_torchaudio_abi
    _rocm_stack_rc=$?
    set -e
    if [[ "$_rocm_stack_rc" -ne 0 ]]; then
        echo "Warnung: ROCm-Audio-Stack inkonsistent (torch/torchaudio), rc=${_rocm_stack_rc}." >&2
        if [[ "${AURIK_DISABLE_TORCHAUDIO_AUTO_REPAIR:-0}" != "1" ]] && repair_rocm_torchaudio; then
            set +e
            check_rocm_torchaudio_abi
            _rocm_stack_rc=$?
            set -e
        fi
        if [[ "$_rocm_stack_rc" -eq 0 ]]; then
            echo "ROCM_STACK_REPAIR erfolgreich." >&2
        elif [[ "$_rocm_stack_rc" -eq 11 || "$_rocm_stack_rc" -eq 12 ]]; then
            echo "Warnung: torchaudio bleibt defekt/inkompatibel; GPU bleibt AKTIV, torchaudio-abhängige Phasen fallen auf CPU/DSP zurück." >&2
            echo "Hinweis: Für erneuten Reparaturversuch AURIK_DISABLE_TORCHAUDIO_AUTO_REPAIR=0 setzen." >&2
            export AURIK_TORCHAUDIO_DEGRADED=1
            _GPU_MODE="ROCm (AMD GPU, torchaudio degraded → selective CPU/DSP fallback)"
        else
            echo "Warnung: ROCm-Basisstack defekt (torch nicht nutzbar). Fallback auf CPU-venv." >&2
            VENV_PYTHON="$VENV_CPU"
            _GPU_MODE="CPU-only (ROCm-Stack defekt)"
        fi
    fi
    # Stack-Mismatch-Guard (Rev. 2026-08-16): torch+rocm6.2 auf System-ROCm 7.2.4
    # erzeugt harte Native-Crashes ohne Python-Dump (Beleg: amdgpu-VM-Leak-Meldung
    # im Kernel-Log beim Prozess-Exit). Warnen, wenn der venv nicht zum System-ROCm passt.
    _sys_rocm_v=""
    _rocm_target="$(readlink -f /opt/rocm 2>/dev/null || echo /opt/rocm)"
    _rocm_base="${_rocm_target##*/}"  # z.B. rocm-7.2.4
    if [[ "$_rocm_base" =~ ^rocm-([0-9]+)\.([0-9]+) ]]; then
        _sys_rocm_v="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    fi
    _torch_hip="$("$VENV_PYTHON" -c 'import torch; print(getattr(torch.version, "hip", "") or "")' 2>/dev/null || true)"
    if [[ -n "$_sys_rocm_v" && -n "$_torch_hip" ]]; then
        _hip_mm="$(echo "$_torch_hip" | cut -d. -f1,2)"
        if [[ "$_hip_mm" != "$_sys_rocm_v" ]]; then
            echo "WARNUNG: Stack-Mismatch — System-ROCm ${_sys_rocm_v}, torch-HIP ${_hip_mm} (${VENV_PYTHON})." >&2
            echo "         torch+rocm${_hip_mm} auf System-ROCm ${_sys_rocm_v} kann hart crashen (Rev. 2026-08-16, amdgpu-VM-Leak-Beleg)." >&2
            echo "         Empfohlen: venv mit torch 2.11.0+rocm7.2 (venv_rocm72)." >&2
        fi
    fi
else
    VENV_PYTHON="$VENV_CPU"
    _GPU_MODE="CPU-only"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "FEHLER: venv-Python nicht gefunden: $VENV_PYTHON" >&2
    echo "Bitte zuerst: bash scripts/install_aurik.sh" >&2
    echo "Alternativ: python3 -m venv .venv_aurik && .venv_aurik/bin/pip install -r requirements/requirements_aurik.txt" >&2
    exit 1
fi

mkdir -p "$SCRIPT_DIR/temp_repro" "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

echo "Aurik GPU-Modus: ${_GPU_MODE} (Python: ${VENV_PYTHON})"

# NICHT NUMBA_DISABLE_JIT setzen (entfernt Rev. 2026-08-16): Der alte
# Workaround gegen den numba-cgutils-Circular-Import-Crash (< 0.57) ist mit
# numba 0.67 obsolet — und NUMBA_DISABLE_JIT=1 lässt @guvectorize-Dekoratoren
# als plain functions zurück („'function' object has no attribute
# 'get_call_template'“) → librosa-Submodule degradierten, DSP-Ersatzpfade.
# Der thread-sichere Import wird jetzt über backend/core/librosa_bootstrap.py
# sichergestellt (Hauptthread-Warmup vor Worker-Start).

# Kein Doppelstart: verhindert UI-Konflikte und wiederholte Force-Quit-Dialoge.
if pgrep -f "[A]urik10/main.py" >/dev/null 2>&1; then
    _pid="$(pgrep -f "[A]urik10/main.py" | head -n 1)"
    echo "Aurik läuft bereits (PID ${_pid})."
    exit 0
fi

# In VS Code-Terminals detach starten, damit VS Code den GUI-Prozess nicht verwaltet.
if [[ "${TERM_PROGRAM:-}" == "vscode" ]]; then
    nohup "$VENV_PYTHON" -B -W ignore::FutureWarning -W ignore::DeprecationWarning Aurik10/main.py "$@" >>"$LOG_FILE" 2>&1 &
    _pid="$!"
    echo "$_pid" >"$PID_FILE"
    echo "Aurik detached gestartet (PID ${_pid}). Log: $LOG_FILE"
    exit 0
fi

exec "$VENV_PYTHON" -B -W ignore::FutureWarning -W ignore::DeprecationWarning Aurik10/main.py "$@"
