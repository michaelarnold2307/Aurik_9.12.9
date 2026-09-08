"""basicpitch_plugin — Polyphonic pitch tracking (ONNX primary, DSP fallback).

Primary model:
    models/basicpitch/basicpitch.onnx

Fallback:
    Spectral peak tracker (STFT-based) for robust offline operation.

The plugin returns polyphonic pitch estimates as a dense matrix:
    pitches_hz[t, k] for frame t and voice slot k (0 means no pitch)

Design goals:
    - CPU-only, no network, no Docker.
    - Thread-safe singleton with double-checked locking.
    - Numerical robustness (NaN/Inf guards).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ONNX_PATH = _ROOT / "models" / "basicpitch" / "basicpitch.onnx"
_ONNX_PATH_NMP = _ROOT / "models" / "basicpitch" / "basicpitch_nmp.onnx"

_MODEL_SR: int = 22_050
_N_FFT: int = 4096
_HOP: int = 512
_WINDOW: int = 4096
_DEFAULT_MAX_POLYPHONY: int = 6

_instance: BasicPitchPlugin | None = None
_lock = threading.Lock()


@dataclass
class BasicPitchResult:
    """Polyphonic pitch-tracking result.

    Attributes:
        frame_times_s: Frame timestamps in seconds, shape [T].
        pitches_hz:    Pitch matrix in Hz, shape [T, K]. Zero = unvoiced slot.
        confidences:   Confidence matrix in [0, 1], shape [T, K].
        model_used:    "basicpitch_onnx" | "dsp_spectral_peaks" | "dsp_failed"
        details:       Additional metrics.
    """

    frame_times_s: np.ndarray
    pitches_hz: np.ndarray
    confidences: np.ndarray
    model_used: str
    details: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_times_s = np.nan_to_num(np.asarray(self.frame_times_s, dtype=np.float32))
        self.pitches_hz = np.nan_to_num(np.asarray(self.pitches_hz, dtype=np.float32))
        self.confidences = np.nan_to_num(np.asarray(self.confidences, dtype=np.float32))
        self.pitches_hz = np.clip(self.pitches_hz, 0.0, 20_000.0)
        self.confidences = np.clip(self.confidences, 0.0, 1.0)


class BasicPitchPlugin:
    """BasicPitch polyphonic estimator (ONNX) with DSP fallback.

    Public API:
        analyze(audio, sr, max_polyphony=6) -> BasicPitchResult
    """

    def __init__(self) -> None:
        self._session = None
        self._model_loaded: bool = False
        self._using_nmp: bool = False
        self._load_model()

    def _load_model(self) -> None:
        # §v10.30: Primär = basicpitch.onnx, robuste NMP-Variante als Fallback-Tier.
        # NMP (Non-Mel-Peak) ist auf degradiertem Material robuster — genau
        # Auriks Kern-Domäne. Gleiche ONNX-Schnittstelle, gleicher Decoder.
        if not _ONNX_PATH.exists():
            logger.info("BasicPitch ONNX nicht gefunden (%s) — DSP-Fallback aktiv.", _ONNX_PATH)
            self._try_load_nmp()
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("BasicPitch", size_gb=0.12):
                    logger.warning("BasicPitch: ML-Budget erschöpft — DSP-Fallback aktiv.")
                    return
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(_ONNX_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._model_loaded = True
            logger.info("🎼 BasicPitch ONNX geladen: %s", _ONNX_PATH.name)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    "BasicPitch",
                    size_gb=0.12,
                    unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_model_loaded", False),  # type: ignore[func-returns-value,misc]
                )
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("BasicPitch ONNX-Init fehlgeschlagen (%s) — NMP-Variante als Fallback.", exc)
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("BasicPitch")
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            self._try_load_nmp()

    def _try_load_nmp(self) -> bool:
        """Lädt die NMP-Variante (Non-Mel-Peak) — robuster auf degradiertem Material.

        Returns True wenn die NMP-Session aktiv ist.
        """
        if not _ONNX_PATH_NMP.exists():
            logger.info("BasicPitch NMP-Variante nicht gefunden (%s) — DSP-Fallback bleibt.", _ONNX_PATH_NMP)
            return False
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(_ONNX_PATH_NMP),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._model_loaded = True
            self._using_nmp = True
            logger.info("🎼 BasicPitch NMP-Variante geladen: %s", _ONNX_PATH_NMP.name)
            return True
        except Exception as exc:
            logger.warning("BasicPitch NMP-Init fehlgeschlagen (%s) — DSP-Fallback.", exc)
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("BasicPitch")
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            return False

    def analyze(self, audio: np.ndarray, sr: int, max_polyphony: int = _DEFAULT_MAX_POLYPHONY) -> BasicPitchResult:
        """Schätzt polyphonic pitches.

        Args:
            audio: Mono or stereo PCM array.
            sr: Sample rate in Hz.
            max_polyphony: Number of output pitch slots per frame.

        Returns:
            BasicPitchResult
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        max_polyphony = int(max(1, min(12, max_polyphony)))

        if self._session is not None:
            try:
                _res = self._analyze_onnx(audio, sr, max_polyphony)
                # §v10.30 Zweitmeinung: schwache Primärantwort (< 0.05 Konfidenz)
                # → NMP-Variante probieren (robuster auf degradiertem Material).
                _max_conf = float(np.max(_res.confidences)) if _res.confidences.size else 0.0
                if _max_conf < 0.05 and not self._using_nmp:
                    if self._try_load_nmp():
                        _res2 = self._analyze_onnx(audio, sr, max_polyphony)
                        _res2.model_used = "basicpitch_nmp"
                        return _res2
                return _res
            except Exception as exc:
                # §v10.30: Vor DSP-Fallback erst die NMP-Variante probieren.
                logger.debug("BasicPitch ONNX-Inferenz fehlgeschlagen (%s) — NMP-Variante.", exc)
                if not self._using_nmp and self._try_load_nmp():
                    try:
                        _res_nmp = self._analyze_onnx(audio, sr, max_polyphony)
                        _res_nmp.model_used = "basicpitch_nmp"
                        return _res_nmp
                    except Exception as _nmp_exc:
                        logger.debug("BasicPitch NMP-Inferenz fehlgeschlagen (%s) — DSP-Fallback.", _nmp_exc)
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)

        return self._analyze_dsp(audio, sr, max_polyphony)

    def _analyze_onnx(self, audio: np.ndarray, sr: int, max_polyphony: int) -> BasicPitchResult:
        """Führt aus: ONNX and decode top-K pitch bins per frame.

        Decoder strategy is resilient to model variant differences:
        - Finds a 2D/3D output tensor with pitch-like bins.
        - Converts bin index to MIDI range [21, 108].
        """
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("BasicPitch", True)
        except Exception:
            logger.warning("basicpitch_plugin.py::_analyze_onnx fallback", exc_info=True)
        try:
            session = self._session
            if session is None:
                raise RuntimeError("BasicPitch ONNX-Session nicht initialisiert")

            audio_m = _resample(audio, sr, _MODEL_SR)
            inp = session.get_inputs()[0]
            in_name = inp.name

            # Detect whether model expects a fixed-length input (static shape).
            # BasicPitch ONNX exports often require exactly N samples per call.
            _fixed_chunk_len: int | None = None
            if inp.shape and len(inp.shape) >= 2:
                _dim = inp.shape[1]
                if isinstance(_dim, int) and _dim > 0:
                    _fixed_chunk_len = _dim

            # Some BasicPitch ONNX exports expect rank-3 input [B, T, 1]
            expected_rank = len(inp.shape) if inp.shape else 0
            out_names = [o.name for o in session.get_outputs()]

            def _run_chunk(chunk: np.ndarray) -> tuple[np.ndarray, int]:
                """Führt aus: ONNX on a single fixed-length chunk; return (probs [T,BINS], bins)."""
                model_in = chunk[np.newaxis, :] if chunk.ndim == 1 else chunk.reshape(1, -1)
                model_in = model_in.astype(np.float32)
                if expected_rank == 3 and model_in.ndim == 2:
                    model_in = model_in[:, :, np.newaxis]
                out_vals = session.run(out_names, {in_name: model_in})
                pitch_tensor = _select_pitch_tensor(out_vals)
                if pitch_tensor is None:
                    raise RuntimeError("No pitch-like output tensor found")
                logits = _to_time_bins(pitch_tensor)
                prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
                prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
                return prob, prob.shape[1] if prob.ndim == 2 else 0

            if _fixed_chunk_len is not None and len(audio_m) > _fixed_chunk_len:
                # Chunk the full audio into fixed-length windows with 50% overlap;
                # aggregate frame probabilities across all chunks.
                hop_chunk = _fixed_chunk_len // 2
                all_probs: list[tuple[float, np.ndarray]] = []  # (time_offset_s, probs[T,BINS])
                pos = 0
                while pos < len(audio_m):
                    end = pos + _fixed_chunk_len
                    chunk = audio_m[pos:end]
                    if len(chunk) < _fixed_chunk_len:
                        chunk = np.pad(chunk, (0, _fixed_chunk_len - len(chunk)))
                    prob_chunk, _ = _run_chunk(chunk)
                    t_offset = float(pos) / _MODEL_SR
                    all_probs.append((t_offset, prob_chunk))
                    pos += hop_chunk

                if not all_probs:
                    raise RuntimeError("No chunks processed")

                # Stitch chunks: use simple concatenation picking non-overlapping halves.
                first_offset, first_probs = all_probs[0]
                bins = first_probs.shape[1] if first_probs.ndim == 2 else 0
                # Build time-aligned frame list
                all_frame_probs: list[np.ndarray] = []
                all_frame_times: list[np.ndarray] = []
                for t_off, prob_c in all_probs:
                    T_c = prob_c.shape[0]
                    ft = t_off + np.arange(T_c, dtype=np.float32) * (_HOP / _MODEL_SR)
                    all_frame_probs.append(prob_c)
                    all_frame_times.append(ft)
                probs = np.concatenate(all_frame_probs, axis=0)
                frame_times_s = np.concatenate(all_frame_times).astype(np.float32)
            else:
                model_in = audio_m[np.newaxis, :] if audio_m.ndim == 1 else np.asarray(audio_m).reshape(1, -1)
                # §Fixed-Length-Fix (2026-09-08): Static-Shape-Modelle (z.B. 43844
                # Samples) akzeptieren auch KURZE Eingaben nur exakt in der
                # erwarteten Länge — padden/truncaten, sonst ONNX InvalidArgument
                # (Matrix-Befund: 'Got: 2757 Expected: 43844' → ML→DSP-Fallback).
                if _fixed_chunk_len is not None and model_in.shape[-1] != _fixed_chunk_len:
                    if model_in.shape[-1] < _fixed_chunk_len:
                        _pad_w = [(0, 0)] * (model_in.ndim - 1) + [(0, _fixed_chunk_len - model_in.shape[-1])]
                        model_in = np.pad(model_in, _pad_w, mode="constant")
                    else:
                        model_in = model_in[..., :_fixed_chunk_len]
                model_in = model_in.astype(np.float32)
                if expected_rank == 3 and model_in.ndim == 2:
                    model_in = model_in[:, :, np.newaxis]
                out_vals = session.run(out_names, {in_name: model_in})
                pitch_tensor = _select_pitch_tensor(out_vals)
                if pitch_tensor is None:
                    raise RuntimeError("No pitch-like output tensor found")
                logits = _to_time_bins(pitch_tensor)
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
                probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                T = probs.shape[0]
                frame_times_s = np.arange(T, dtype=np.float32) * (_HOP / _MODEL_SR)

            T, bins = probs.shape
            topk_idx = np.argpartition(probs, kth=max(0, bins - max_polyphony), axis=1)[:, -max_polyphony:]
            topk_val = np.take_along_axis(probs, topk_idx, axis=1)

            # Sort descending by confidence per frame
            ord_idx = np.argsort(-topk_val, axis=1)
            topk_idx = np.take_along_axis(topk_idx, ord_idx, axis=1)
            topk_val = np.take_along_axis(topk_val, ord_idx, axis=1)

            midi = _bins_to_midi(topk_idx, bins)
            pitches_hz = _midi_to_hz(midi)
            confidences = np.clip(topk_val, 0.0, 1.0).astype(np.float32)

            return BasicPitchResult(
                frame_times_s=frame_times_s,
                pitches_hz=pitches_hz,
                confidences=confidences,
                model_used="basicpitch_onnx",
                details={"n_frames": float(T), "n_bins": float(bins)},
            )
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("BasicPitch", False)
                except Exception:
                    logger.warning("basicpitch_plugin.py::unknown fallback", exc_info=True)

    def _analyze_dsp(self, audio: np.ndarray, sr: int, max_polyphony: int) -> BasicPitchResult:
        """STFT peak-based polyphonic fallback."""
        try:
            import scipy.signal as sps

            if len(audio) < _WINDOW:
                pad = _WINDOW - len(audio)
                audio = np.pad(audio, (0, pad))

            _, times, stft = sps.stft(
                audio,
                fs=sr,
                nperseg=_WINDOW,
                noverlap=_WINDOW - _HOP,
                nfft=_N_FFT,
                boundary=None,
                padded=False,
                window="hann",
            )
            mag = np.abs(stft).astype(np.float32)  # [F, T]
            freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr).astype(np.float32)

            fmask = (freqs >= 55.0) & (freqs <= 2_000.0)
            mag_b = mag[fmask, :]
            freqs_b = freqs[fmask]

            if mag_b.size == 0:
                raise RuntimeError("No valid frequency bins in fallback")

            T = mag_b.shape[1]
            K = max_polyphony
            pitches = np.zeros((T, K), dtype=np.float32)
            conf = np.zeros((T, K), dtype=np.float32)

            # Per-frame top-K spectral peaks
            for t in range(T):
                col = mag_b[:, t]
                if np.all(col <= 1e-10):
                    continue
                k = min(K, len(col))
                idx = np.argpartition(col, -k)[-k:]
                vals = col[idx]
                order = np.argsort(-vals)
                idx = idx[order]
                vals = vals[order]

                pitches[t, :k] = freqs_b[idx]
                vmax = float(np.max(vals)) if np.max(vals) > 0 else 1.0
                conf[t, :k] = np.clip(vals / vmax, 0.0, 1.0)

            return BasicPitchResult(
                frame_times_s=times.astype(np.float32),
                pitches_hz=pitches,
                confidences=conf,
                model_used="dsp_spectral_peaks",
                details={"n_frames": float(T), "sr": float(sr)},
            )
        except Exception as exc:
            logger.warning("BasicPitch DSP-Fallback fehlgeschlagen: %s", exc)
            return BasicPitchResult(
                frame_times_s=np.zeros(1, dtype=np.float32),
                pitches_hz=np.zeros((1, max_polyphony), dtype=np.float32),
                confidences=np.zeros((1, max_polyphony), dtype=np.float32),
                model_used="dsp_failed",
            )


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio.astype(np.float32)  # type: ignore[no-any-return]
    from scipy.signal import resample_poly

    g = math.gcd(from_sr, to_sr)
    up = to_sr // g
    down = from_sr // g
    return resample_poly(audio.astype(np.float32), up, down).astype(np.float32)  # type: ignore[no-any-return]


def _select_pitch_tensor(outputs: list[np.ndarray]) -> np.ndarray | None:
    """Pick the most likely pitch tensor from ONNX outputs.

    Priority:
        1) 3D/2D tensor with last dim in [48, 1024]
        2) Largest 2D tensor
    """
    candidates: list[np.ndarray] = []
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim < 2:
            continue
        last = arr.shape[-1]
        if 48 <= last <= 1024:
            candidates.append(arr)
    if candidates:
        return max(candidates, key=lambda x: x.size)

    twod = [np.asarray(o) for o in outputs if np.asarray(o).ndim >= 2]
    if not twod:
        return None
    return max(twod, key=lambda x: x.size)  # type: ignore[no-any-return]


def _to_time_bins(arr: np.ndarray) -> np.ndarray:
    """Konvertiert arbitrary 2D/3D output to [T, BINS]."""
    a = np.asarray(arr)
    if a.ndim == 2:
        # [T, B] or [B, T]
        if a.shape[0] <= 16 and a.shape[1] > a.shape[0]:
            return a.astype(np.float32)  # type: ignore[no-any-return]
        if a.shape[0] > a.shape[1]:
            return a.astype(np.float32)  # type: ignore[no-any-return]
        return a.T.astype(np.float32)  # type: ignore[no-any-return]
    if a.ndim >= 3:
        # Typical: [B, T, BINS] or [B, BINS, T]
        b0 = a[0]
        if b0.ndim == 2:
            if b0.shape[0] >= b0.shape[1]:
                return b0.astype(np.float32)  # type: ignore[no-any-return]
            return b0.T.astype(np.float32)  # type: ignore[no-any-return]
        # Fallback flattening
        flat = b0.reshape(b0.shape[0], -1)
        return flat.astype(np.float32)  # type: ignore[no-any-return]
    return a.reshape(1, -1).astype(np.float32)  # type: ignore[no-any-return]


def _bins_to_midi(bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Map pitch bin index to MIDI range [21, 108]."""
    # Linear mapping across the available bin range
    midi_min, midi_max = 21.0, 108.0
    return midi_min + (midi_max - midi_min) * (bin_idx.astype(np.float32) / max(1.0, float(n_bins - 1)))  # type: ignore[no-any-return]


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return (440.0 * (2.0 ** ((midi.astype(np.float32) - 69.0) / 12.0))).astype(np.float32)  # type: ignore[no-any-return]


def get_basicpitch_plugin() -> BasicPitchPlugin:
    """Thread-safe singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BasicPitchPlugin()
    return _instance


def unload_basicpitch() -> None:
    """Unload BasicPitch resources and release ML budget slot."""
    global _instance
    with _lock:
        if _instance is not None:
            try:
                _instance._session = None
                _instance._model_loaded = False
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            _instance = None
    try:
        from backend.core.ml_memory_budget import release as _release

        _release("BasicPitch")
    except Exception as _exc:
        logger.debug("Plugin operation failed (non-critical): %s", _exc)


def analyze_polyphonic_pitch(
    audio: np.ndarray, sr: int, max_polyphony: int = _DEFAULT_MAX_POLYPHONY
) -> BasicPitchResult:
    """Convenience API for polyphonic pitch tracking."""
    return get_basicpitch_plugin().analyze(audio, sr, max_polyphony=max_polyphony)
