"""
MIGraphX ONNX Inference Adapter — GPU-first backend for AMD RDNA3 (gfx1100).

Provides a drop-in replacement for onnxruntime.InferenceSession that compiles
and runs ONNX models via AMD's MIGraphX graph compiler. Falls back gracefully
to ONNX Runtime CPU when MIGraphX is unavailable or unsupported.

Architecture
------------
Python (MIGraphXSession) → ctypes → libmigraphx_bridge.so → MIGraphX C API
                                                                   ↓
                                                          GPU (gfx1100)

Bridge-Build (Rev. 2026-08-16 — ROCm 7.2.4 / MIGraphX 2.15):
    bash scripts/build_migraphx_bridge.sh
Quelle: backend/core/lib/migraphx_bridge.cpp (im Repo, nicht mehr /tmp).
Wichtig für 2.15: Die Bridge kompiliert mit offload_copy=true — nur so liefert
die C-API (ohne t.copy_to/t.copy_from) GPU-Ergebnisse auf den Host zurück.
Verifiziert: Silero VAD GPU vs. ORT-CPU max|Δ| = 1e-6 (Rev. 2026-08-16).

Usage
-----
::

    from backend.core.migraphx_adapter import MIGraphXSession

    session = MIGraphXSession("model.onnx")
    outputs = session.run(None, {"input": numpy_array})

    # Or via the convenience function:
    session = create_migraphx_session("model.onnx", default_dim=256)
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared library discovery
# ---------------------------------------------------------------------------

_BRIDGE_DIR = Path(__file__).resolve().parent / "lib"
_BRIDGE_SO = _BRIDGE_DIR / "libmigraphx_bridge.so"

# Required ROCm runtime libraries (must be on LD_LIBRARY_PATH or RPATH)


def _discover_rocm_lib_dirs() -> list[str]:
    """ROCm-Laufzeitverzeichnisse version-agnostisch entdecken (Rev. 2026-08-16).

    Produktions-Stack: ROCm 7.2.4 unter /opt/rocm-7.2.4 (Symlink /opt/rocm).
    Legacy 6.2.0 wird weiter erkannt, aber die neueste Installation hat Vorrang.
    """
    candidates: set[Path] = set()
    for d in Path("/opt").glob("rocm-*"):
        if d.is_dir() and (d / "lib").is_dir():
            candidates.add(d)
    _rocm_link = Path("/opt/rocm")
    if _rocm_link.is_dir():
        candidates.add(_rocm_link)  # Symlink der aktiven Installation (zuletzt, Duplikat)

    def _ver_key(p: Path) -> tuple:
        try:
            return tuple(int(x) for x in p.name.removeprefix("rocm-").split(".") if x.isdigit())
        except Exception as exc:
            logger.debug("§V6 ROCM-Version-Key-Parsing fehlgeschlagen — leeres Tuple zurückgegeben (Path %s): %s", p, exc)
            return ()

    dirs: list[str] = []
    for cand in sorted(candidates, key=_ver_key, reverse=True):
        _lib = cand / "lib"
        if _lib.is_dir():
            dirs.append(str(_lib))
        _mgx = cand / "lib" / "migraphx" / "lib"
        if _mgx.is_dir():
            dirs.append(str(_mgx))
    return dirs


_ROCM_LIB_DIRS = _discover_rocm_lib_dirs()

# ---------------------------------------------------------------------------
# Bridge loading
# ---------------------------------------------------------------------------

_bridge: ctypes.CDLL | None = None
_bridge_load_error: str | None = None


def _ensure_rocm_path() -> None:
    """Prepend ROCm directories to LD_LIBRARY_PATH if not already present."""
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = existing.split(":") if existing else []
    for d in reversed(_ROCM_LIB_DIRS):
        if d not in parts:
            parts.insert(0, d)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _load_bridge() -> ctypes.CDLL:
    """Load the MIGraphX bridge shared library (idempotent)."""
    global _bridge, _bridge_load_error

    if _bridge is not None:
        return _bridge
    if _bridge_load_error is not None:
        raise RuntimeError(_bridge_load_error)

    _ensure_rocm_path()

    if not _BRIDGE_SO.exists():
        _bridge_load_error = f"MIGraphX bridge not found: {_BRIDGE_SO}"
        raise RuntimeError(_bridge_load_error)

    try:
        lib = ctypes.CDLL(str(_BRIDGE_SO))
    except OSError as exc:
        _bridge_load_error = f"Cannot load MIGraphX bridge: {exc}"
        raise RuntimeError(_bridge_load_error) from exc

    # -- mgx_load_onnx -------------------------------------------------------
    lib.mgx_load_onnx.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    lib.mgx_load_onnx.restype = ctypes.c_void_p

    # -- mgx_destroy ---------------------------------------------------------
    lib.mgx_destroy.argtypes = [ctypes.c_void_p]
    lib.mgx_destroy.restype = None

    # -- mgx_get_input_count -------------------------------------------------
    lib.mgx_get_input_count.argtypes = [ctypes.c_void_p]
    lib.mgx_get_input_count.restype = ctypes.c_int

    # -- mgx_get_input_name --------------------------------------------------
    lib.mgx_get_input_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mgx_get_input_name.restype = ctypes.c_char_p

    # -- mgx_get_input_ndim --------------------------------------------------
    lib.mgx_get_input_ndim.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mgx_get_input_ndim.restype = ctypes.c_int

    # -- mgx_get_input_shape -------------------------------------------------
    lib.mgx_get_input_shape.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mgx_get_input_shape.restype = ctypes.POINTER(ctypes.c_int64)

    # -- mgx_run (multi-input) -----------------------------------------------
    lib.mgx_run.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,  # input_count
        ctypes.POINTER(ctypes.c_char_p),  # input_names
        ctypes.POINTER(ctypes.c_void_p),  # input_data (float*[])
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # input_shapes
        ctypes.POINTER(ctypes.c_int),  # input_ndims
        ctypes.POINTER(ctypes.c_void_p),  # output_data (float**)
        ctypes.POINTER(ctypes.c_int64),  # output_shape (int64_t[8])
        ctypes.POINTER(ctypes.c_int),  # output_ndim
    ]
    lib.mgx_run.restype = ctypes.c_int

    _bridge = lib
    return lib


def is_migraphx_available() -> bool:
    """Return True if the MIGraphX bridge can be loaded."""
    try:
        _load_bridge()
        return True
    except (RuntimeError, OSError) as exc:
        logger.debug("§V6 MIGraphX-Bridge nicht ladbar — False zurückgegeben (Bridge-Load-Fehler): %s", exc)
        return False


# §v10.40 Compile-Zeit-Regel (Rev. 2026-08-16): Modelle > 200 MB → MIGraphX-Compile
# > 30 min → ORT/CPU statt MIGraphX. Wird von session_manager und onnx/runtime geprüft.
MIGRAPHX_MAX_MODEL_MB = 200.0


def migraphx_model_size_mb(model_path: str | Path) -> float:
    """Dateigröße eines ONNX-Modells in MB (0.0 bei nicht lesbarer Datei)."""
    try:
        return Path(model_path).stat().st_size / (1024 * 1024)
    except OSError as exc:
        logger.debug("§V6 ONNX-Dateigröße nicht lesbar — 0.0 MB zurückgegeben (Path %s): %s", model_path, exc)
        return 0.0

def is_migraphx_size_eligible(model_path: str | Path) -> bool:
    """True, wenn das Modell klein genug für MIGraphX ist (§v10.40 Größenlimit)."""
    return migraphx_model_size_mb(model_path) <= MIGRAPHX_MAX_MODEL_MB


# ---------------------------------------------------------------------------
# Inference session
# ---------------------------------------------------------------------------


class MIGraphXSession:
    """GPU-accelerated ONNX inference via AMD MIGraphX.

    Mimics the public API of ``onnxruntime.InferenceSession`` closely enough
    to serve as a drop-in replacement for the common ``session.run(None, {})``
    pattern used across Aurik plugins.

    Parameters
    ----------
    model_path:
        Path to the ONNX model file.
    default_dim:
        Value used for dynamic axes when compiling the model (default 256).
        Models with dynamic sequence lengths will be compiled with this
        maximum dimension. Inputs MUST have a matching size at runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        default_dim: int = 256,
        providers: list[str] | None = None,  # ORT-compat: ignored, always MIGraphX
        sess_options: Any = None,  # ORT-compat: ignored
        **kwargs: Any,  # additional ORT kwargs (ignored)
    ) -> None:
        self._model_path = str(model_path)
        self._default_dim = default_dim
        self._shape_hints = kwargs.pop("shape_hints", "")
        self._bridge = _load_bridge()
        self._handle: Any = None
        self._input_names: list[str] = []
        self._input_shapes: list[tuple[int, ...]] = []
        self._providers = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]

        self._handle = self._bridge.mgx_load_onnx(
            self._model_path.encode(),
            ctypes.c_size_t(self._default_dim),
            (self._shape_hints or "").encode(),
        )
        if not self._handle:
            raise RuntimeError(f"MIGraphX failed to load model: {self._model_path}")

        self._discover_inputs()

    # -- input discovery -----------------------------------------------------

    def _discover_inputs(self) -> None:
        """Read input names and shapes from the compiled program."""
        count = self._bridge.mgx_get_input_count(self._handle)
        self._input_names = []
        self._input_shapes = []
        for i in range(count):
            name_ptr = self._bridge.mgx_get_input_name(self._handle, i)
            name = name_ptr.decode() if name_ptr else f"input_{i}"
            self._input_names.append(name)

            ndim = self._bridge.mgx_get_input_ndim(self._handle, i)
            shape_ptr = self._bridge.mgx_get_input_shape(self._handle, i)
            shape = tuple(int(shape_ptr[j]) for j in range(ndim)) if ndim else ()
            self._input_shapes.append(shape)

    # -- public API ----------------------------------------------------------

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference with multi-input support.

        Parameters
        ----------
        output_names:
            Ignored — always returns first output.
        input_feed:
            Dict mapping input names to numpy float32 arrays.
            "main:" prefixed internal inputs are auto-filled with zeros.

        Returns
        -------
        List of output numpy arrays (currently single-output only).
        """
        if not input_feed:
            raise ValueError("No input data provided")

        # Filter to real inputs (non-main), ensure float32 contiguous
        real_inputs = [
            (name, np.ascontiguousarray(arr, dtype=np.float32))
            for name, arr in input_feed.items()
            if not name.startswith("main:")
        ]

        if not real_inputs:
            # All inputs are internal — just use first
            name = next(iter(input_feed))
            real_inputs = [(name, np.ascontiguousarray(input_feed[name], dtype=np.float32))]

        count = len(real_inputs)

        # Build ctypes arrays
        names_arr = (ctypes.c_char_p * count)()
        datas_arr = (ctypes.c_void_p * count)()
        shapes_arr = (ctypes.POINTER(ctypes.c_int64) * count)()
        ndims_arr = (ctypes.c_int * count)()

        # Keep references alive during the call
        _shape_arrays = []
        for i, (name, arr) in enumerate(real_inputs):
            names_arr[i] = name.encode()
            datas_arr[i] = arr.ctypes.data
            shape_np = np.array(arr.shape, dtype=np.int64)
            _shape_arrays.append(shape_np)
            shapes_arr[i] = shape_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            ndims_arr[i] = ctypes.c_int(len(arr.shape))

        out_data = ctypes.c_void_p()
        out_shape = (ctypes.c_int64 * 8)()
        out_ndim = ctypes.c_int()

        ret = self._bridge.mgx_run(
            self._handle,
            ctypes.c_int(count),
            names_arr,
            datas_arr,
            shapes_arr,
            ndims_arr,
            ctypes.byref(out_data),
            out_shape,
            ctypes.byref(out_ndim),
        )

        if ret != 0:
            raise RuntimeError(f"MIGraphX inference failed (code {ret})")

        # Build output numpy array
        ndim_out = out_ndim.value
        shape_out = tuple(int(out_shape[i]) for i in range(ndim_out))
        total = 1
        for s in shape_out:
            total *= s

        result = (
            np.ctypeslib.as_array(
                ctypes.cast(out_data, ctypes.POINTER(ctypes.c_float)),
                shape=(total,),
            )
            .copy()
            .reshape(shape_out)
        )

        return [result]

    def get_providers(self) -> list[str]:
        """Return the provider name for compatibility."""
        return ["MIGraphXExecutionProvider"]

    def get_inputs(self) -> list[Any]:
        """Return input metadata (name + shape)."""

        # Build a simple named-tuple-like list
        class InputMeta:
            def __init__(self, name: str, shape: tuple[int, ...]):
                self.name = name
                self.shape = shape

        return [InputMeta(n, s) for n, s in zip(self._input_names, self._input_shapes)]

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        """Release GPU resources."""
        if hasattr(self, "_handle") and self._handle is not None and self._bridge is not None:
            try:
                self._bridge.mgx_destroy(self._handle)
            except Exception:
                pass
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> MIGraphXSession:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_migraphx_session(
    model_path: str | Path,
    default_dim: int = 256,
) -> MIGraphXSession:
    """Create a MIGraphX inference session.

    Raises ``RuntimeError`` if the bridge is unavailable or loading fails.
    """
    return MIGraphXSession(model_path, default_dim=default_dim)


def create_session_with_fallback(
    model_path: str | Path,
    default_dim: int = 256,
) -> MIGraphXSession | Any:
    """Create a session with automatic GPU→CPU fallback.

    Tries MIGraphX (GPU) first; falls back to ONNX Runtime CPU.
    """
    import onnxruntime as ort

    try:
        if is_migraphx_available():
            sess = create_migraphx_session(model_path, default_dim=default_dim)
            logger.info("Using MIGraphX GPU for %s", model_path)
            return sess
    except Exception as exc:
        logger.debug("MIGraphX unavailable for %s: %s", model_path, exc)

    logger.info("Falling back to ONNX Runtime CPU for %s", model_path)
    return ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )


def get_migraphx_device_info() -> dict[str, object]:
    """Return GPU device info for MIGraphX bridge detection.

    Used by MLDeviceManager to populate GPU metadata during detection.
    """
    return {
        "name": "AMD Radeon RX 7900 XTX (MIGraphX)",
        "arch": "gfx1100",
        "vram_gb": 24.0,
    }
