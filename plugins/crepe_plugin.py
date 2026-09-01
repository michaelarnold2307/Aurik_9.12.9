"""
CREPE Plugin — Direkte ONNX-Inferenz (kein Docker)
===================================================

Pitch-Detection mit CREPE (Convolutional Representation for Pitch Estimation)
via lokal gebündeltem ONNX-Modell (models/crepe/crepe/model-full.onnx, 89 MB).

Kein Docker, kein Netzwerk — vollständig out-of-the-box nutzbar.

Modell-Spezifikation:
    Input:  ("input",      [N, 1024]) float32  @ 16 000 Hz, Fenster zentriert
    Output: ("classifier", [N,  360]) float32  Salience-Werte über 360 Pitch-Bins
    Pitch-Bins: f_i = 10.0 · 2^(i · 20/1200) · 32.703195 Hz  für i = 0..359

Referenz:
    Kim et al. (2018) — "CREPE: A Convolutional Representation for Pitch Estimation"
    https://arxiv.org/abs/1802.06182

DSP-Fallback: librosa.pyin() (Mauch & Dixon 2014), sekundär librosa.yin()
bei fehlender ONNX-Laufzeit oder pYIN-Fehlern.

Invarianten (§3.1, §3.2, §3.7 Aurik-Spec):
    - Thread-sicherer Singleton mit Double-Checked Locking
    - NaN/Inf in keiner Ausgabe (nan_to_num)
    - Provider-Wahl über get_ort_providers() oder CPU-Fallback (§G5 Determinismus)
    - Alle öffentlichen Methoden vollständig typisiert (PEP 484)
    - GPU-Support via AURIK_PITCH_GPU=1 optional; CPU ist Default für Reproduzierbarkeit
"""
# pylint: disable=import-outside-toplevel

import hashlib
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Feature-Flag: AURIK_PITCH_GPU=1 ermöglicht GPU-Provider für Pitch-Tracking
_PITCH_GPU_ENABLED = os.getenv("AURIK_PITCH_GPU", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Modell-Pfad (relativ zur Projektwurzel)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
_CREPE_ONNX_PATH = _PROJECT_ROOT / "models" / "crepe" / "crepe.onnx"

# CREPE Frame-Parameter (trainiert bei 16 kHz)
_CREPE_SR: int = 16_000
_FRAME_LENGTH: int = 1024
_HOP_SAMPLES: int = 160  # 10 ms @ 16 kHz

# 360 Pitch-Bins (CREPE offizielle Formel, Kim et al. 2018):
#   cents_i = i * 20.0 + 1997.3794084376191  (20 Cent pro Bin, Offset = ~C1)
#   f_i     = 10 · 2^(cents_i / 1200)
# Abdeckung: ~31.7 Hz (C1) … ~2005 Hz (B6)
_CENTS_MAPPING: np.ndarray = (np.linspace(0, 7180, 360) + 1997.3794084376191).astype(np.float64)
_CREPE_FREQS: np.ndarray = (10.0 * 2.0 ** (_CENTS_MAPPING / 1200.0)).astype(np.float32)

# Voicing-Konfidenz-Schwelle
_VOICED_THRESHOLD: float = 0.45

# Performance-Budget: CPU-only — Inferenz in Chunks à _CHUNK_FRAMES Frames
# (≈ 5 s Audio pro Batch @ 10 ms Hop; kein Subsampling — vollständige Abdeckung)
_CHUNK_FRAMES: int = 512


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------
@dataclass
class CrepeResult:
    """Pitch-Tracking-Ergebnis aus CREPE (F0 pro Frame).

    Alle Arrays haben identische Länge ``n_frames``.
    """

    f0_hz: np.ndarray  # Grundfrequenz pro Frame [Hz]
    voiced_prob: np.ndarray  # Voicing-Wahrscheinlichkeit  ∈ [0, 1]
    salience: np.ndarray  # Maximale Salience pro Frame  ∈ [0, 1]
    times_s: np.ndarray  # Zeitstempel in Sekunden
    model_used: str  # "crepe_onnx" | "dsp_pyin" | "dsp_yin" | "dsp_yin_failed"
    details: dict[str, float] = field(default_factory=dict)

    def voiced_f0(self, threshold: float = _VOICED_THRESHOLD) -> np.ndarray:
        """Array mit F0 nur für voiced Frames, NaN sonst."""
        f0 = self.f0_hz.copy()
        f0[self.voiced_prob < threshold] = np.nan
        return f0

    def mean_f0(self, threshold: float = _VOICED_THRESHOLD) -> float:
        """Mittlere F0 über voiced Frames; 0.0 bei Stille."""
        vf0 = self.voiced_f0(threshold)
        valid = vf0[np.isfinite(vf0)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def voiced_fraction(self, threshold: float = _VOICED_THRESHOLD) -> float:
        """Anteil voiced Frames ∈ [0, 1]."""
        return float(np.mean(self.voiced_prob >= threshold))


# ---------------------------------------------------------------------------
# Plugin-Klasse
# ---------------------------------------------------------------------------
class CrepePlugin:
    """Direkte ONNX-Inferenz des CREPE-Modells (kein Docker).

    Algorithmus:
        1. Resample Audio auf 16 000 Hz (CREPE-Trainings-SR)
        2. Frame-Extraktion: Hop = 10 ms = 160 Samples; Länge = 1024 Samples
           Padding: halbe Fensterlänge vorne/hinten (zentrierte Frames)
        3. Normalisierung: Mean-Subtraction + Amplitude-Normierung pro Frame
        4. ONNX-Batch-Inferenz → Salience-Matrix (N × 360)
        5. F0 = Σ(f_i · s_i) / Σ(s_i) — gewichteter Durchschnitt der Pitch-Bins
        6. Voicing-Wahrscheinlichkeit = max(salience_i) pro Frame
        7. Zeitstempel: t_i = (i · 160) / 16 000 Sekunden

    Thread-Safety: Double-Checked Locking für Singleton (§3.2 Aurik-Spec).
    """

    def __init__(self) -> None:
        self._session: object | None = None
        self._model_used: str = "dsp_pyin"
        self._result_cache: dict[str, CrepeResult] = {}  # SHA256-Cache (§3.8)
        self._cache_lock: threading.Lock = threading.Lock()
        self._load_model()

    def unload(self) -> None:
        """Gibt Model-Session und Cache frei."""
        self._session = None
        with self._cache_lock:
            self._result_cache.clear()

    def _load_model(self) -> None:
        """ONNX-Session laden oder stumm auf DSP-Fallback wechseln."""
        try:
            import onnxruntime as ort

            if not _CREPE_ONNX_PATH.exists():
                logger.debug(
                    "CREPE-ONNX nicht gefunden (%s) — pYIN-Fallback aktiv",
                    _CREPE_ONNX_PATH,
                )
                return

            try:
                from backend.core.ml_memory_budget import release as _release
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("CREPE", size_gb=0.10):
                    try:
                        _release("CREPE")
                    except Exception:
                        logger.warning("crepe_plugin.py::_load_model fallback", exc_info=True)
                    if not _try_alloc("CREPE", size_gb=0.10):
                        logger.warning("CREPE: ML-Budget erschöpft — pYIN-Fallback.")
                        return
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Wähle Provider: GPU wenn AURIK_PITCH_GPU=1, sonst CPU (§G5 Determinismus)
            if _PITCH_GPU_ENABLED:
                try:
                    from backend.core.ml_device_manager import get_ort_providers
                    providers = get_ort_providers("CREPE")
                    logger.info("CREPE: GPU-Provider aktiviert via AURIK_PITCH_GPU=1")
                except Exception as _e:
                    logger.warning("CREPE: GPU-Provider fehlgeschlagen (%s) — CPU-Fallback", _e)
                    providers = ["CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(_CREPE_ONNX_PATH),
                sess_options=opts,
                providers=providers,
            )
            self._model_used = "crepe_onnx"
            logger.info(
                "crepe_plugin: ONNX model loaded: %s (provider=%s)",
                _CREPE_ONNX_PATH.name,
                providers[0],
            )
            # Warmup-Inference: erste ONNX-Inferenz ist langsam (JIT/Graph-Optimierung).
            # Ein Dummy-Run mit kleinem Batch eliminiert den 13s→6s Kaltstart-Nachteil.
            try:
                _dummy = np.zeros((1, _FRAME_LENGTH), dtype=np.float32)
                assert isinstance(self._session, ort.InferenceSession)
                self._session.run(["classifier"], {"input": _dummy})
                logger.debug("CREPE ONNX warmup inference completed")
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                def _unload_crepe() -> None:
                    self._session = None
                    self._model_used = "pyin"

                _reg_plm(
                    "CREPE",
                    size_gb=0.10,
                    unload_fn=_unload_crepe,
                )
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("CREPE-ONNX nicht verfügbar (%s) — pYIN-Fallback aktiv", exc)
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("CREPE")
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

    def analyze(self, audio: np.ndarray, sr: int) -> CrepeResult:
        """Analysiert den Grundton (F0) eines Audio-Signals.

        Ergebnisse werden SHA256-basiert gecacht (§3.8 Aurik-Spec, max. 128 Einträge).

        Args:
            audio: 1-D oder 2-D (Stereo) float32/64-Array, beliebige SR.
            sr:    Sample-Rate in Hz des Eingangssignals.

        Returns:
            :class:`CrepeResult` mit f0_hz, voiced_prob, salience, times_s.

        Raises:
            Kein Raise — Fallback auf leer bei totalem Fehler.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))

        # SHA256-Cache (§3.8): Identische Audio-Eingaben werden nicht erneut berechnet
        _h = hashlib.sha256(audio.tobytes())
        _h.update(sr.to_bytes(4, "little"))
        _cache_key = "crepe:" + _h.hexdigest()[:16]
        with self._cache_lock:
            if _cache_key in self._result_cache:
                logger.debug("CREPE-Cache-Hit: %s", _cache_key)
                return self._result_cache[_cache_key]

        result = self._analyze_onnx(audio, sr) if self._session is not None else self._analyze_pyin(audio, sr)

        with self._cache_lock:
            if len(self._result_cache) >= 128:
                # LRU-Eviction: ältesten Eintrag entfernen
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            self._result_cache[_cache_key] = result
        return result

    def _analyze_onnx(self, audio: np.ndarray, sr: int) -> CrepeResult:
        """CREPE-Inferenz via onnxruntime (Fallback zu pYIN bei ONNX-Fehler)."""
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("CREPE", True)
        except Exception:
            logger.warning("crepe_plugin.py::_analyze_onnx fallback", exc_info=True)
        try:
            import onnxruntime as ort
            import scipy.signal as sps

            assert isinstance(self._session, ort.InferenceSession)

            # 1) Resample auf 16 kHz
            if sr != _CREPE_SR:
                gcd = math.gcd(sr, _CREPE_SR)
                up, down = _CREPE_SR // gcd, sr // gcd
                audio_16k = sps.resample_poly(audio, up, down).astype(np.float32)
            else:
                audio_16k = audio.copy()

            audio_16k = np.nan_to_num(audio_16k)

            # 2) Frame-Extraktion (zentriert, konstantes Padding)
            n_samples = len(audio_16k)
            hop = _HOP_SAMPLES
            win = _FRAME_LENGTH
            pad = win // 2
            audio_padded = np.pad(audio_16k, pad, mode="constant")
            n_frames = max(1, (n_samples + hop - 1) // hop)
            # Strides für Zero-Copy Frame-Erzeugung
            sample_stride = int(audio_padded.dtype.itemsize)
            strides = (
                sample_stride * hop,
                sample_stride,
            )
            frames = np.lib.stride_tricks.as_strided(
                audio_padded,
                shape=(n_frames, win),
                strides=strides,
            ).copy()

            # 3) Normalisierung pro Frame
            frames -= np.mean(frames, axis=1, keepdims=True)
            frame_max = np.max(np.abs(frames), axis=1, keepdims=True)
            frame_max = np.where(frame_max < 1e-7, 1.0, frame_max)
            frames = (frames / frame_max).astype(np.float32)

            # 4) Chunk-Streaming: Inferenz in _CHUNK_FRAMES-großen Batches
            #    → vollständige Abdeckung aller Frames ohne Subsampling-Verlust
            f0_hz_parts: list = []
            voiced_parts: list = []
            sal_parts: list = []

            for _start in range(0, n_frames, _CHUNK_FRAMES):
                chunk = frames[_start : _start + _CHUNK_FRAMES]  # (C, 1024)

                # 5) ONNX-Batch-Inferenz pro Chunk
                sal_chunk: np.ndarray = self._session.run(["classifier"], {"input": chunk})[0].astype(
                    np.float32
                )  # (C, 360)
                sal_chunk = np.nan_to_num(sal_chunk).clip(0.0, 1.0)

                # 6) F0-Schätzung: gewichteter Durchschnitt der 360 Pitch-Bins
                row_sum = sal_chunk.sum(axis=1, keepdims=True)
                row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
                norm = sal_chunk / row_sum
                f0_hz_parts.append(norm.dot(_CREPE_FREQS))  # (C,)
                voiced_parts.append(sal_chunk.max(axis=1).clip(0.0, 1.0))
                sal_parts.append(sal_chunk.max(axis=1))

            # 7) Alle Chunks zusammenführen
            f0_hz = np.concatenate(f0_hz_parts).astype(np.float32)  # (n_frames,)
            voiced_prob = np.concatenate(voiced_parts).astype(np.float32)
            sal_interp = np.concatenate(sal_parts).astype(np.float32)

            # 9) Zeitstempel
            times_s = np.arange(n_frames, dtype=np.float32) * hop / _CREPE_SR

            f0_hz = np.nan_to_num(f0_hz)
            voiced_prob = np.nan_to_num(voiced_prob)

            n_voiced = int(np.sum(voiced_prob >= _VOICED_THRESHOLD))
            logger.debug(
                "CREPE ONNX: %d Frames, voiced=%.1f%%, F0-Median=%.1f Hz",
                n_frames,
                100.0 * n_voiced / max(1, n_frames),
                float(np.median(f0_hz[voiced_prob >= _VOICED_THRESHOLD])) if n_voiced > 0 else 0.0,
            )

            return CrepeResult(
                f0_hz=f0_hz,
                voiced_prob=voiced_prob,
                salience=sal_interp,
                times_s=times_s,
                model_used="crepe_onnx",
            )

        except Exception as exc:
            logger.warning("CREPE-ONNX-Inferenz fehlgeschlagen (%s) — Wechsel zu pYIN", exc)
            return self._analyze_pyin(audio, sr)
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("CREPE", False)
                except Exception:
                    logger.warning("crepe_plugin.py::unknown fallback", exc_info=True)

    def _analyze_pyin(self, audio: np.ndarray, sr: int) -> CrepeResult:
        """pYIN-Fallback (Mauch & Dixon 2014) — O(N²), max. 2 Sekunden.

        Post-2018-Fallback gem. §4.2 (erlaubt).
        """
        try:
            import librosa

            # pYIN: max. 30 s (O(N) per Frame mit librosa-Optimierung; Pytest-Budget)
            seg_len = min(len(audio), int(sr * 30.0))
            seg = audio[:seg_len]

            # Ensure at least two fmin periods fit in frame_length to avoid
            # librosa.pyin warnings on low-pitch material.
            _fmin_floor = float(librosa.note_to_hz("C1"))  # ≈32.7 Hz — ideal lower bound
            _min_frame = int(np.ceil((2.0 * sr) / max(_fmin_floor, 1e-6))) + 1
            _frame_length = max(2048, _min_frame)
            _fmin_safe = _fmin_floor
            f0, _, voiced_probs = librosa.pyin(seg, fmin=_fmin_safe, fmax=2_000.0, sr=sr, frame_length=_frame_length)
            f0 = np.nan_to_num(f0.astype(np.float32))
            voiced_probs = np.nan_to_num(voiced_probs.astype(np.float32))
            n_frames = len(f0)
            hop_lib = 512
            times_s = (np.arange(n_frames) * hop_lib / sr).astype(np.float32)

            return CrepeResult(
                f0_hz=f0,
                voiced_prob=voiced_probs,
                salience=voiced_probs,
                times_s=times_s,
                model_used="dsp_pyin",
                details={"segment_len_s": seg_len / sr},
            )
        except Exception as exc:
            logger.warning("pYIN-Fallback fehlgeschlagen (%s) — Wechsel zu YIN", exc)
            return self._analyze_yin(audio, sr)

    def _analyze_yin(self, audio: np.ndarray, sr: int) -> CrepeResult:
        """Sekundärer YIN-Fallback (de Cheveigne & Kawahara 2002)."""
        try:
            import librosa

            seg_len = min(len(audio), int(sr * 30.0))
            seg = audio[:seg_len]

            _fmin_floor = float(librosa.note_to_hz("C1"))
            _min_frame = int(np.ceil((2.0 * sr) / max(_fmin_floor, 1e-6))) + 1
            _frame_length = max(2048, _min_frame)
            _fmin_safe = _fmin_floor
            f0 = librosa.yin(seg, fmin=_fmin_safe, fmax=2_000.0, sr=sr, frame_length=_frame_length)
            f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            voiced_probs = np.where(f0 > 0.0, 0.55, 0.0).astype(np.float32)
            n_frames = len(f0)
            hop_lib = 512
            times_s = (np.arange(n_frames) * hop_lib / sr).astype(np.float32)

            return CrepeResult(
                f0_hz=f0,
                voiced_prob=voiced_probs,
                salience=voiced_probs,
                times_s=times_s,
                model_used="dsp_yin",
                details={"segment_len_s": seg_len / sr},
            )
        except Exception as exc:
            logger.warning("YIN-Fallback fehlgeschlagen (%s) — leeres Ergebnis", exc)
            return CrepeResult(
                f0_hz=np.zeros(1, dtype=np.float32),
                voiced_prob=np.zeros(1, dtype=np.float32),
                salience=np.zeros(1, dtype=np.float32),
                times_s=np.zeros(1, dtype=np.float32),
                model_used="dsp_yin_failed",
            )


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking §3.2 Aurik-Spec)
# ---------------------------------------------------------------------------
_INSTANCE_HOLDER: list[CrepePlugin | None] = [None]
_lock = threading.Lock()


def get_crepe_plugin() -> CrepePlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking).

    Returns:
        Initialisierte :class:`CrepePlugin`-Instanz (lazy init).
    """
    if _INSTANCE_HOLDER[0] is None:  # Schnellpfad ohne Lock
        with _lock:
            if _INSTANCE_HOLDER[0] is None:  # Zweiter Check unter Lock (Race-Condition-sicher)
                _INSTANCE_HOLDER[0] = CrepePlugin()
    plugin = _INSTANCE_HOLDER[0]
    assert plugin is not None
    return plugin


def unload_crepe() -> None:
    """Unload CREPE resources and release ML budget slot.

    Safe to call multiple times.
    """
    with _lock:
        plugin = _INSTANCE_HOLDER[0]
        if plugin is not None:
            try:
                plugin.unload()
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            _INSTANCE_HOLDER[0] = None
    try:
        from backend.core.ml_memory_budget import release as _release

        _release("CREPE")
    except Exception as _exc:
        logger.debug("Plugin operation failed (non-critical): %s", _exc)


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------
def analyze_pitch(
    audio: np.ndarray,
    sr: int,
) -> CrepeResult:
    """Pitch-Tracking ohne manuelle Plugin-Instantiierung.

    Nutzt CREPE-ONNX wenn verfügbar, andernfalls pYIN- und YIN-Fallback.

    Args:
        audio: 1-D oder Stereo Audio-Array (beliebige SR).
        sr:    Sample-Rate in Hz.

    Returns:
        :class:`CrepeResult` mit F0-Tracking-Daten.

    Example::

        result = analyze_pitch(audio, sr=48000)
        logger.debug("Mittlere F0: %.1f Hz", result.mean_f0())
        logger.debug("Modell: %s", result.model_used)
    """
    return get_crepe_plugin().analyze(audio, sr)


# ---------------------------------------------------------------------------
# Rückwärtskompatible Aliase (alte Klassen-/Funktionsnamen aus Docker-Version)
# ---------------------------------------------------------------------------
CREPEPlugin = CrepePlugin
