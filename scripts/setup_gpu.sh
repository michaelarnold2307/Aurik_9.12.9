#!/usr/bin/env bash
# setup_gpu.sh — Aurik GPU acceleration setup (Linux) v10.0.5
# ============================================================
# Auto-detects GPU (NVIDIA CUDA / AMD ROCm / none) and creates
# .venv_gpu with appropriate PyTorch + ONNX Runtime packages.
#
# Usage:
#   ./scripts/setup_gpu.sh              # auto-detect GPU
#   ./scripts/setup_gpu.sh --cuda       # force NVIDIA CUDA
#   ./scripts/setup_gpu.sh --rocm       # force AMD ROCm
#   ./scripts/setup_gpu.sh --dry-run    # show what would be installed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_GPU="$REPO_ROOT/.venv_gpu"
VENV_CPU="$REPO_ROOT/.venv_aurik"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}=== Aurik GPU Setup (Linux) v10.0.5 ===${NC}"

# ── Parse args ──────────────────────────────────────────────────────────
FORCE=""; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)     FORCE="cuda"; shift ;;
        --rocm)     FORCE="rocm"; shift ;;
        --cpu)      FORCE="cpu"; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)  echo "Usage: $0 [--cuda|--rocm|--cpu] [--dry-run]"; exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Detect GPU ──────────────────────────────────────────────────────────
detect_gpu() {
    if [[ -n "$FORCE" ]]; then echo "$FORCE"; return; fi
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "cuda"; return
    fi
    if lspci 2>/dev/null | grep -qi "nvidia"; then echo "cuda"; return; fi
    if lspci 2>/dev/null | grep -qi "amd\|advanced micro" && [[ -e /dev/kfd ]]; then
        # Detect ROCm version from installed packages
        if command -v rocminfo &>/dev/null; then echo "rocm"; return; fi
        # ROCm-capable hardware but no ROCm installed
        echo "rocm"; return
    fi
    echo "cpu"
}

GPU_TYPE=$(detect_gpu)
echo -e "GPU: ${YELLOW}$GPU_TYPE${NC}"

if [[ "$GPU_TYPE" == "cpu" ]]; then
    echo -e "${YELLOW}No NVIDIA/AMD GPU with drivers detected. Skipping GPU setup.${NC}"
    echo "Aurik runs CPU-only (.venv_aurik)."
    exit 0
fi

# ── Check prerequisites ─────────────────────────────────────────────────
if [[ ! -f "$VENV_CPU/bin/python" ]]; then
    echo -e "${RED}ERROR: .venv_aurik not found at $VENV_CPU${NC}"
    echo "Run the main Aurik setup first: python3 -m venv .venv_aurik"
    exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN — would create .venv_gpu with:${NC}"
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        echo "  pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124"
        echo "  pip install onnxruntime-gpu"
    elif [[ "$GPU_TYPE" == "rocm" ]]; then
        echo "  pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm7.2"
        echo "  pip install onnxruntime-rocm"
    fi
    exit 0
fi

# ── Create GPU venv ─────────────────────────────────────────────────────
echo -e "${GREEN}Creating GPU virtual environment at .venv_gpu ...${NC}"
"$VENV_CPU/bin/python" -m venv --clear "$VENV_GPU" || {
    echo -e "${RED}Failed to create venv. Check disk space and permissions.${NC}"
    exit 1
}

PIP="$VENV_GPU/bin/pip"
PYTHON="$VENV_GPU/bin/python"

"$PIP" install --upgrade pip setuptools wheel -q || {
    echo -e "${RED}Failed to upgrade pip. Check network connection.${NC}"
    exit 1
}

# ── Install GPU packages ────────────────────────────────────────────────
install_status=0
if [[ "$GPU_TYPE" == "cuda" ]]; then
    echo -e "${GREEN}Installing PyTorch CUDA 12.4 + ONNX Runtime GPU ...${NC}"
    "$PIP" install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124 || install_status=1
    "$PIP" install onnxruntime-gpu || install_status=1
elif [[ "$GPU_TYPE" == "rocm" ]]; then
    echo -e "${GREEN}Installing PyTorch ROCm 7.2.4 + ONNX Runtime ROCm ...${NC}"
    "$PIP" install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm7.2 || install_status=1
    "$PIP" install onnxruntime-rocm || install_status=1
fi

if [[ $install_status -ne 0 ]]; then
    echo -e "${RED}Package installation failed. Check network and PyTorch index URL.${NC}"
    echo "Falling back to CPU-only (.venv_aurik)."
    exit 1
fi

# ── Verify ──────────────────────────────────────────────────────────────
echo -e "${GREEN}Verifying GPU installation ...${NC}"
if "$PYTHON" -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA/ROCm not available after install', file=sys.stderr)
    sys.exit(1)
print(f'OK: PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')
import onnxruntime as ort
providers = ort.get_available_providers()
gpu_providers = [p for p in providers if 'CUDA' in p or 'ROCM' in p or 'Tensorrt' in p]
if not gpu_providers:
    print('WARNING: No GPU ONNX provider found. ONNX models will run on CPU.', file=sys.stderr)
else:
    print(f'OK: ONNX GPU providers: {gpu_providers}')
" 2>&1; then
    echo -e "${GREEN}GPU setup complete! .venv_gpu is ready.${NC}"
else
    echo -e "${RED}GPU verification failed.${NC}"
    echo "  1. Check GPU drivers: nvidia-smi (NVIDIA) or rocminfo (AMD)"
    echo "  2. Check CUDA/ROCm toolkit installation"
    echo "  3. Try: AURIK_FORCE_CPU=1 ./run_aurik.sh (CPU-only fallback)"
    exit 1
fi
