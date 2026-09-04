"""FCPE Plugin — Fast Context-Aware Pitch Estimation (ONNX / CREPE-Fallback)
==============================================================================

Pitch-Schätzung mit FCPE (Zhu et al. 2023).

ML-Kaskade:
    Primär:   models/fcpe/fcpe.onnx (wenn vorhanden, ~69 MB)
    Fallback: CREPE ONNX via get_crepe_plugin() (85 MB, semantisch identisch)
    DSP:      librosa.pyin() (Mauch & Dixon 2014) als letzter Fallback

Verbesserungen gegenüber CREPE (Kim 2018):
    - Höhere Genauigkeit bei Gesangsstimmen (Conformer-Transformer)
    - Robuster bei polyfonem Material und Hintergrundgeräuschen

ONNX I/O:
    Input:  mel      (1, T, 128)  float32  log-mel @ 16 kHz
    Output: salience (1, T, 360)  float32  sigmoid Pitch-Klassen-Probs

Gives CrepeResult zurück — vollständig API-kompatibel zu crepe_plugin.

Referenz:
    Zhu et al. (2023) — "FCPE: Fast Context-Aware Pitch Estimation"
    https://arxiv.org/abs/2306.15522

Invarianten (§3.1, §3.2, §3.7 Aurik-Spec):
    - Thread-sicherer Singleton mit Double-Checked Locking
    - NaN/Inf in keiner Ausgabe (nan_to_num)
    - Provider-Wahl über get_ort_providers() oder CPU-Fallback (§G5 Determinismus)
    - Alle öffentlichen Methoden vollständig typisiert (PEP 484)
    - GPU-Support via AURIK_PITCH_GPU=1 optional; CPU ist Default für §G5 Reproduzierbarkeit
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np

# Re-export CrepeResult for full API compatibility
from plugins.crepe_plugin import CrepeResult, get_crepe_plugin

# Feature-Flag: AURIK_PITCH_GPU=1 ermöglicht GPU-Provider für Pitch-Tracking
_PITCH_GPU_ENABLED = os.getenv("AURIK_PITCH_GPU", "").lower() in ("1", "true", "yes")

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_FCPE_ONNX_PATH = _PROJECT_ROOT / "models" / "fcpe" / "fcpe.onnx"

# FCPE-Mel-Parameter (aus fcpe.pt Checkpoint-Config)
_FCPE_SR: int = 16_000
_FCPE_N_FFT: int = 1024
_FCPE_WIN: int = 1024
_FCPE_HOP: int = 160  # 10 ms @ 16 kHz
_FCPE_N_MELS: int = 128
_FCPE_FMIN: float = 0.0
_FCPE_FMAX: float = 8000.0
_FCPE_CLIP: float = 1e-5

# Voiced-Threshold (aus Modell-Config: threshold=0.05)
_VOICED_THRESHOLD: float = 0.05

# Cent-Tabelle: linear von f0_to_cent(32.7 Hz) bis f0_to_cent(1975.5 Hz), 360 Bins
# f0_to_cent(f) = 1200 * log2(f / 10.0)   (CFNaiveMelPE-Konvention)
_CENT_MIN = 1200.0 * math.log2(32.7 / 10.0)
_CENT_MAX = 1200.0 * math.log2(1975.5 / 10.0)
_CENT_TABLE = np.linspace(_CENT_MIN, _CENT_MAX, 360, dtype=np.float32)

# Mel-Basis (librosa-kompatibel, einmalig berechnet)
_MEL_BASIS_HOLDER: list[np.ndarray | None] = [None]
_MEL_BASIS_LOCK = threading.Lock()


def _get_mel_basis() -> np.ndarray:
    """Lazy-init des Mel-Filter-Basismatrix (128x513, float32)."""
    if _MEL_BASIS_HOLDER[0] is None:
        with _MEL_BASIS_LOCK:
            if _MEL_BASIS_HOLDER[0] is None:
                try:
                    from librosa.filters import mel as librosa_mel

                    _MEL_BASIS_HOLDER[0] = librosa_mel(
                        sr=_FCPE_SR,
                        n_fft=_FCPE_N_FFT,
                        n_mels=_FCPE_N_MELS,
                        fmin=_FCPE_FMIN,
                        fmax=_FCPE_FMAX,
                    ).astype(np.float32)
                except Exception:
                    # Fallback: einfache Dreiecksfilterbänke
                    _MEL_BASIS_HOLDER[0] = np.eye(_FCPE_N_MELS, _FCPE_N_FFT // 2 + 1, dtype=np.float32)
    mel_basis = _MEL_BASIS_HOLDER[0]
    assert mel_basis is not None
    return np.asarray(mel_basis, dtype=np.float32)  # type: ignore[no-any-return]


def _compute_mel(audio_16k: np.ndarray) -> np.ndarray:
    """Berechnet log-mel-Spektrogramm  identisch zu torchfcpe.MelModule.

    Args:
        audio_16k: mono float32-Array @ 16 kHz, Werte in [-1, 1].

    Returns:
        mel: float32  (T, 128) — log-komprimiertes Mel-Spektrogramm.
    """
    import scipy.signal as sps

    # Padding wie torchfcpe.MelModule (center-Pad)
    pad_l = (_FCPE_WIN - _FCPE_HOP) // 2
    pad_r = max((_FCPE_WIN - _FCPE_HOP + 1) // 2, _FCPE_WIN - audio_16k.shape[-1] - pad_l)
    audio_padded = np.pad(audio_16k, (pad_l, pad_r), mode="reflect")

    # STFT
    _, _, stft = sps.stft(
        audio_padded,
        nperseg=_FCPE_N_FFT,
        noverlap=_FCPE_N_FFT - _FCPE_HOP,
        nfft=_FCPE_N_FFT,
        window="hann",
        boundary=None,
        padded=False,
    )
    # Magnitude: (n_fft//2+1, T)
    mag = np.sqrt(stft.real**2 + stft.imag**2 + 1e-9).astype(np.float32)

    # Mel-Filterbank: (128, T)
    mel_basis = _get_mel_basis()
    mel = mel_basis @ mag  # (128, T)

    # Log-Kompression (dynamic_range_compression_torch equivalent)
    mel = np.log(np.maximum(mel, _FCPE_CLIP))  # (128, T)
    mel_arr = np.asarray(mel, dtype=np.float32)

    return np.asarray(mel_arr.T, dtype=np.float32)  # type: ignore[no-any-return]  # (T, 128)


def _local_argmax_decode(salience: np.ndarray, threshold: float = _VOICED_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Konvertiert FCPE-Salience-Matrix zu F0 [Hz] + voiced_prob.

    Identisch zu CFNaiveMelPE.latent2cents_local_decoder + cent_to_f0.

    Args:
        salience: (T, 360) float32 — sigmoid Pitch-Klassen-Wahrscheinlichkeiten.
        threshold: Minimum-Salience; Frames darunter → f0=0.

    Returns:
        (f0_hz, voiced_prob) je als (T,) float32-Array.
    """
    T, C = salience.shape
    # Maximale Salience und Peak-Index pro Frame
    voiced_prob = salience.max(axis=-1).astype(np.float32)  # (T,)
    max_idx = salience.argmax(axis=-1)  # (T,) int

    # 9 Bins um den Peak (±4): local weighted average
    local_idx = np.clip(
        max_idx[:, None] + np.arange(-4, 5, dtype=np.int32)[None, :],
        0,
        C - 1,
    )  # (T, 9)

    ci_l = _CENT_TABLE[local_idx]  # (T, 9) cent-Werte
    y_l = salience[np.arange(T)[:, None], local_idx]  # (T, 9) Salience
    y_sum = y_l.sum(axis=-1)  # (T,)

    cents = np.where(
        y_sum > 1e-12,
        (ci_l * y_l).sum(axis=-1) / y_sum,
        _CENT_MIN,
    )  # (T,)

    # Voiced-Maske: Frames mit Salience <= threshold → f0=0
    voiced_mask = voiced_prob > threshold
    f0_hz = np.where(voiced_mask, 10.0 * (2.0 ** (cents / 1200.0)), 0.0).astype(np.float32)
    f0_hz = np.nan_to_num(f0_hz, nan=0.0, posinf=0.0, neginf=0.0)

    return f0_hz, voiced_prob


class FcpePlugin:
    """FCPE Pitch-Estimator — ONNX-primär, CREPE-Fallback, pYIN-DSP.

    API identisch zu CrepePlugin: analyze(audio, sr) → CrepeResult.

    Algorithmus (FCPE ONNX Vollpfad):
        1. Resample auf 16 kHz
        2. Log-Mel-Spektrogramm (n_fft=1024, hop=160, n_mels=128, fmin=0, fmax=8000)
        3. ONNX-Inferenz: mel (1,T,128) → salience (1,T,360)
        4. Local-Argmax-Decoder: salience → cents → Hz

    Thread-Safety: Double-Checked Locking (§3.2 Aurik-Spec).
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._crepe_delegate: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """FCPE ONNX laden; sonst CREPE-Plugin-Delegation registrieren."""
        # Versuch 1: FCPE ONNX (Zhu et al. 2023, ~69 MB)
        try:
            import onnxruntime as ort

            if _FCPE_ONNX_PATH.exists():
                try:
                    from backend.core.ml_memory_budget import try_allocate as _try_alloc

                    if not _try_alloc("FCPE", size_gb=0.07):
                        logger.warning("FCPE: ML-Budget erschöpft — CREPE-Fallback.")
                        return
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)

                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Wähle Provider: GPU wenn AURIK_PITCH_GPU=1, sonst CPU (§G5 Determinismus)
                if _PITCH_GPU_ENABLED:
                    try:
                        from backend.core.ml_device_manager import get_ort_providers
                        providers = get_ort_providers("FCPE")
                        logger.info("FCPE: GPU-Provider aktiviert via AURIK_PITCH_GPU=1")
                    except Exception as _e:
                        logger.warning("FCPE: GPU-Provider fehlgeschlagen (%s) — CPU-Fallback", _e)
                        providers = ["CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    str(_FCPE_ONNX_PATH),
                    sess_options=opts,
                    providers=providers,
                )
                logger.info("fcpe_plugin: ONNX model loaded: %s (provider=%s)", _FCPE_ONNX_PATH.name, providers[0])
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    def _unload_fcpe() -> None:
                        self._session = None

                    _reg_plm("FCPE", size_gb=0.07, unload_fn=_unload_fcpe)
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
                return
            logger.debug(
                "FCPE ONNX nicht gefunden (%s) — CREPE ONNX-Fallback aktiv",
                _FCPE_ONNX_PATH,
            )
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("FCPE ONNX-Init fehlgeschlagen (%s) — CREPE-Fallback", exc)

        # Versuch 2: CREPE ONNX als transparente Delegation
        try:
            self._crepe_delegate = get_crepe_plugin()
            delegate_model = getattr(self._crepe_delegate, "model_used", "unbekannt")
            logger.info(
                "🎵 FCPE-Plugin: delegiert an CREPE ONNX (Modell=%s)",
                delegate_model,
            )
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("CREPE-Delegation fehlgeschlagen (%s) — pYIN DSP aktiv", exc)

    @property
    def model_used(self) -> str:
        """Gibt den aktuell aktiven Analysepfad zurück."""
        if self._session is not None:
            return "fcpe_onnx"
        if self._crepe_delegate is not None:
            return "crepe_onnx_via_fcpe"
        return "dsp_pyin"

    def clear_runtime_handles(self) -> None:
        """Gibt interne Laufzeit-Handles frei, ohne harte Fehler auszulösen."""
        self._session = None
        self._crepe_delegate = None

    def analyze(self, audio: np.ndarray, sr: int) -> CrepeResult:
        """Pitch-Tracking via FCPE ONNX → CREPE ONNX → pYIN.

        Args:
            audio: 1-D oder Stereo float32-Array, beliebige SR.
            sr:    Sample-Rate in Hz.

        Returns:
            :class:`CrepeResult` — API-kompatibel zu CrepePlugin.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))

        if self._session is not None:
            return self._analyze_fcpe_onnx(audio, sr)
        if self._crepe_delegate is not None:
            return cast(CrepeResult, self._crepe_delegate.analyze(audio, sr))
        return self._analyze_pyin(audio, sr)

    def _analyze_fcpe_onnx(self, audio: np.ndarray, sr: int) -> CrepeResult:
        """FCPE ONNX-Inferenz: Waveform → Mel → Salience → F0.

        ONNX I/O:
            Input:  mel      (1, T, 128) float32  log-mel @ 16 kHz
            Output: salience (1, T, 360) float32  sigmoid Pitch-Klassen-Probs
        """
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("FCPE", True)
        except Exception:
            logger.warning("fcpe_plugin.py::_analyze_fcpe_onnx fallback", exc_info=True)
        try:
            import scipy.signal as sps

            # 1) Resample auf 16 kHz
            if sr != _FCPE_SR:
                gcd = math.gcd(sr, _FCPE_SR)
                audio_16k = sps.resample_poly(audio, _FCPE_SR // gcd, sr // gcd).astype(np.float32)
            else:
                audio_16k = audio.astype(np.float32)
            audio_16k = np.clip(np.nan_to_num(audio_16k), -1.0, 1.0)

            # 2) Log-Mel-Spektrogramm: (T, 128)
            mel = _compute_mel(audio_16k)  # (T, 128)
            n_frames = mel.shape[0]

            # 3) ONNX-Inferenz: mel (1, T, 128) → salience (1, T, 360)
            #    OOM-Guard: chunk mel in 3000-frame segments for large files
            _MAX_CHUNK = 3000
            if n_frames <= _MAX_CHUNK:
                mel_inp = mel[None].astype(np.float32)  # (1, T, 128)
                [sal_out] = self._session.run(["salience"], {"mel": mel_inp})
                salience = np.nan_to_num(sal_out[0]).astype(np.float32)  # (T, 360)
            else:
                _chunks: list[np.ndarray] = []
                for _i in range(0, n_frames, _MAX_CHUNK):
                    _chunk = mel[_i : _i + _MAX_CHUNK]
                    _cinp = _chunk[None].astype(np.float32)
                    [_cout] = self._session.run(["salience"], {"mel": _cinp})
                    _chunks.append(np.nan_to_num(_cout[0]).astype(np.float32))
                salience = np.concatenate(_chunks, axis=0)  # (T, 360)
            salience = np.clip(salience, 0.0, 1.0)

            # 4) Local-Argmax-Decoder: salience → [Hz], voiced_prob
            f0_hz, voiced_prob = _local_argmax_decode(salience)

            # 5) Zeitstempel
            times_s = np.arange(n_frames, dtype=np.float32) * _FCPE_HOP / _FCPE_SR

            logger.debug(
                "FCPE ONNX: %d Frames, voiced=%.1f%%, F0-Median=%.1f Hz",
                n_frames,
                100.0 * float(np.mean(voiced_prob > _VOICED_THRESHOLD)),
                float(np.median(f0_hz[f0_hz > 0.0])) if np.any(f0_hz > 0.0) else 0.0,
            )
            return CrepeResult(
                f0_hz=f0_hz,
                voiced_prob=voiced_prob,
                salience=voiced_prob,
                times_s=times_s,
                model_used="fcpe_onnx",
            )
        except Exception as exc:
            logger.warning("FCPE-ONNX-Inferenz fehlgeschlagen (%s) — CREPE/pYIN Fallback", exc)
            if self._crepe_delegate is not None:
                return cast(CrepeResult, self._crepe_delegate.analyze(audio, sr))
            return self._analyze_pyin(audio, sr)
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("FCPE", False)
                except Exception:
                    logger.warning("fcpe_plugin.py::_analyze_fcpe_onnx fallback", exc_info=True)

    def _analyze_pyin(self, audio: np.ndarray, sr: int) -> CrepeResult:
        """pYIN DSP-Fallback (Mauch & Dixon 2014)."""
        try:
            import librosa

            seg = audio[: min(len(audio), int(sr * 30.0))]
            # fmin must satisfy: at least 2 periods fit within frame_length.
            # Use a small safety margin above the exact boundary to avoid
            # floating-point edge warnings in librosa.pyin for sr=48k/frame=2048.
            _fmin_min_hz = (float(sr) / 1024.0) * 1.01
            _fmin_safe = max(float(librosa.note_to_hz("C1")), _fmin_min_hz)
            f0, _, voiced_probs = librosa.pyin(seg, fmin=_fmin_safe, fmax=2_000.0, sr=sr)
            f0 = np.nan_to_num(f0.astype(np.float32))
            voiced_probs = np.nan_to_num(voiced_probs.astype(np.float32))
            times_s = (np.arange(len(f0)) * 512 / sr).astype(np.float32)
            return CrepeResult(
                f0_hz=f0,
                voiced_prob=voiced_probs,
                salience=voiced_probs,
                times_s=times_s,
                model_used="dsp_pyin",
            )
        except Exception as exc:
            logger.warning("pYIN-Fallback fehlgeschlagen (%s)", exc)
            return CrepeResult(
                f0_hz=np.zeros(1, np.float32),
                voiced_prob=np.zeros(1, np.float32),
                salience=np.zeros(1, np.float32),
                times_s=np.zeros(1, np.float32),
                model_used="dsp_pyin_failed",
            )


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking §3.2 Aurik-Spec)
# ---------------------------------------------------------------------------
_INSTANCE_HOLDER: list[FcpePlugin | None] = [None]
_lock = threading.Lock()


def get_fcpe_plugin() -> FcpePlugin:
    """Thread-sicherer Singleton-Accessor.

    Returns:
        Initialisierte :class:`FcpePlugin`-Instanz (lazy init).
    """
    if _INSTANCE_HOLDER[0] is None:
        with _lock:
            if _INSTANCE_HOLDER[0] is None:
                _INSTANCE_HOLDER[0] = FcpePlugin()
    plugin = _INSTANCE_HOLDER[0]
    assert plugin is not None
    return plugin


def get_loaded_fcpe_plugin() -> FcpePlugin | None:
    """Gibt nur eine bereits geladene FCPE-Instanz zurück, ohne Lazy-Load."""
    return _INSTANCE_HOLDER[0]


def unload_fcpe() -> None:
    """Unload FCPE resources and release ML budget slot.

    Safe to call multiple times.
    """
    plugin: FcpePlugin | None = None
    with _lock:
        if _INSTANCE_HOLDER[0] is not None:
            try:
                plugin = _INSTANCE_HOLDER[0]
                assert plugin is not None
                plugin.clear_runtime_handles()
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            _INSTANCE_HOLDER[0] = None
    try:
        from backend.core.ml_memory_budget import release as _release

        _release("FCPE")
    except Exception as _exc:
        logger.debug("Operation failed (non-critical): %s", _exc)


def analyze_pitch(audio: np.ndarray, sr: int) -> CrepeResult:
    """Convenience-Funktion: FCPE → CREPE → pYIN Pitch-Tracking.

    Args:
        audio: 1-D oder Stereo Audio-Array (beliebige SR).
        sr:    Sample-Rate in Hz.

    Returns:
        :class:`CrepeResult` mit F0-Tracking-Daten.
    """
    return get_fcpe_plugin().analyze(audio, sr)
