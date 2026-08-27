#!/usr/bin/env bash
# build_migraphx_bridge.sh — baut backend/core/lib/libmigraphx_bridge.so
# gegen die installierte ROCm-/MIGraphX-Version (Produktion: ROCm 7.2.4,
# MIGraphX 2.15.0; Legacy: 6.2 wurde ersetzt).
#
# Verwendung:
#   bash scripts/build_migraphx_bridge.sh            # baut im Repo
#   bash scripts/build_migraphx_bridge.sh --clean    # löscht Build-Artefakte vorher
# Voraussetzungen (Ubuntu 24.04, ROCm 7.2.4):
#   sudo apt install migraphx migraphx-dev g++
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUT="$REPO_ROOT/backend/core/lib/libmigraphx_bridge.so"
SRC="$REPO_ROOT/backend/core/lib/migraphx_bridge.cpp"

# ── ROCm-Pfade version-agnostisch ermitteln (7.2.4 primär, ältere als Fallback) ──
ROCM_DIR="/opt/rocm"
if [[ -L /opt/rocm ]]; then
    ROCM_DIR="$(readlink -f /opt/rocm)"
elif [[ ! -d "$ROCM_DIR" ]]; then
    for d in /opt/rocm-*; do
        [[ -d "$d/include" && -d "$d/lib" ]] || continue
        ROCM_DIR="$d"
        break
    done
fi

if [[ ! -d "$ROCM_DIR/include/migraphx" ]]; then
    echo "FEHLER: MIGraphX-Header nicht gefunden ($ROCM_DIR/include/migraphx)." >&2
    echo "        Bitte installieren: sudo apt install migraphx migraphx-dev" >&2
    exit 1
fi

MGX_LIB="$ROCM_DIR/lib/migraphx/lib"
[[ -d "$MGX_LIB" ]] || MGX_LIB="$ROCM_DIR/lib"

if [[ "${1:-}" == "--clean" ]]; then
    rm -f "$OUT"
fi

echo "ROCm:      $ROCM_DIR"
echo "MIGraphX:  $(dpkg -s migraphx 2>/dev/null | grep -m1 '^Version' | awk '{print $2}')"
echo "Quelle:    $SRC"
echo "Ziel:      $OUT"

g++ -std=c++17 -fPIC -shared -O2 -D__HIP_PLATFORM_AMD__ \
    -I"$ROCM_DIR/include" \
    -L"$ROCM_DIR/lib" \
    -L"$MGX_LIB" \
    "$SRC" \
    -lmigraphx_c -lmigraphx_onnx -lmigraphx_gpu -lmigraphx -lamdhip64 \
    -Wl,-rpath,"$ROCM_DIR/lib:$MGX_LIB" \
    -o "$OUT"

echo "OK: $OUT"
ldd "$OUT" | grep -E "migraphx|amdhip" || true
