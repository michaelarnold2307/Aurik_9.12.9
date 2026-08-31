"""rmvpe_plugin — RMVPE Robust Mel-scale Pitch Estimator (ICASSP 2023).

RMVPE (Robust Multi-period Vocoder-based Pitch Estimator via Mel spectrogram):
    - Übertrifft CREPE bei Vibrato, schnellen Tonfolgen und stimmhaft/stimmlos-Übergängen
    - Fehlerrate bei Gesang ~30 % geringer als CREPE (RPA auf MIR-1K)
    - ONNX-Modell: models/rmvpe/rmvpe.onnx (~26 MB)
    - Fallback: librosa.pyin() (pYIN, Mauch & Dixon 2014)

Aurik 10.0.0 Pitch-Tracking-Hierarchie (§4.4, Spec 04 Rev. 2026-08-16):
    Primär:    FCPE ONNX (fcpe_plugin)
    Fallback:  RMVPE ONNX (dieser Plugin)
    DSP:       PESTO → pYIN via librosa
    Legacy:    CREPE nur noch intern im FCPE-Plugin als Delegate — als
               eigenständiger Produktions-Tier VERBOTEN (Spec 04, Zeile 1129)

Referenz:
    Wei et al. "RMVPE: A Robust Model for Vocal Pitch Estimation
    in Polyphonic Music" — ICASSP 2023

Singleton-Pattern: get_rmvpe_plugin() verwenden.
CPU-Only: CPUExecutionProvider, kein CUDA.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ONNX_PATH = _ROOT / "models" / "rmvpe" / "rmvpe.onnx"

# RMVPE Mel-Parameter (16 kHz Modell-SR, gemäß Paper)
_MODEL_SR: int = 16_000
_N_MELS: int = 128
_FRAME_LEN: int = 1024  # 64 ms @ 16 kHz
_HOP_LEN: int = 160  # 10 ms @ 16 kHz → 100 Frames/s
# Decoder mapping for current rmvpe.onnx export.
# Empirically calibrated from reference tones (220/440/880 Hz) to reduce
# systematic high-bias while preserving monotonic bin ordering.
_CENTS_OFFSET: float = 1189.2218089321786
_CENTS_BIN_STEP: float = 22.657964325265493
_PITCH_BINS: int = 360

_lock = threading.Lock()
_instance: RmvpePlugin | None = None


def _estimate_tonal_reference_hz(mono_16k: np.ndarray) -> tuple[float | None, float]:
    """Schätzt dominant tonal reference frequency from magnitude spectrum.

    Returns:
        (f_ref_hz | None, confidence_ratio)
        confidence_ratio = peak_mag / median_mag in 50..1200 Hz band.
    """
    x = np.nan_to_num(np.asarray(mono_16k, dtype=np.float32), nan=0.0)
    if x.size < 2048:
        return None, 0.0

    win = np.hanning(x.size).astype(np.float32)
    spec = np.abs(np.fft.rfft((x * win).astype(np.float64))).astype(np.float32)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / _MODEL_SR).astype(np.float32)

    band = (freqs >= 50.0) & (freqs <= 1200.0)
    if not np.any(band):
        return None, 0.0

    sb = spec[band]
    fb = freqs[band]
    peak_idx = int(np.argmax(sb))
    peak_mag = float(sb[peak_idx])
    med_mag = float(np.median(sb) + 1e-9)
    conf = peak_mag / med_mag
    f_ref = float(fb[peak_idx])

    if not np.isfinite(f_ref) or f_ref < 50.0 or f_ref > 1200.0:
        return None, conf
    return f_ref, conf


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RmvpeResult:
    """Ergebnis der RMVPE-Pitch-Schätzung.

    Attributes:
        f0:           Estimated fundamental frequency per frame [Hz], NaN = unvoiced
        times:        Frame center times in seconds
        confidence:   Salience/Konfidenz per frame ∈ [0, 1]
        voiced_flag:  True wenn Frame als stimmhaft klassifiziert
        model_used:   "rmvpe_onnx" | "crepe_fallback" | "pyin_fallback"
        f0_mean:      Mittlere F0 über stimmhafte Frames [Hz]
        f0_std:       Standardabweichung der F0 [Hz]
    """

    f0: np.ndarray
    times: np.ndarray
    confidence: np.ndarray
    voiced_flag: np.ndarray
    model_used: str
    f0_mean: float = 0.0
    f0_std: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RmvpePlugin
# ---------------------------------------------------------------------------


class RmvpePlugin:
    """RMVPE Neural Pitch Tracker (ONNX, CPUExecutionProvider).

    Verarbeitet Mono-Audio bei 16 kHz intern; akzeptiert 48 kHz Eingang (resampelt).
    Fallback auf pYIN (librosa) wenn ONNX-Modell fehlt.

    Pitch-Output ist F0 in Hz per Frame (10 ms Hop). Stille / unvoiced → NaN.
    """

    def __init__(self) -> None:
        self._session = None
        self._model_loaded: bool = False
        self._try_load()

    def _try_load(self) -> None:
        """Lädt RMVPE ONNX-Modell; pYIN-Fallback bei Fehler."""
        if not _ONNX_PATH.exists():
            logger.info("RMVPE ONNX nicht gefunden (%s) — pYIN-Fallback aktiv.", _ONNX_PATH)
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("RMVPE", size_gb=0.03):
                    logger.warning("RMVPE: ML-Budget erschöpft — pYIN-Fallback.")
                    return
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(_ONNX_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._model_loaded = True
            logger.info("✅ RMVPE ONNX geladen: %s (§4.4 primärer Pitch-Tracker)", _ONNX_PATH.name)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    "RMVPE",
                    size_gb=0.03,
                    unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_model_loaded", False),  # type: ignore[func-returns-value,misc]
                )
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("RMVPE ONNX Ladefehler: %s — pYIN-Fallback aktiv.", exc)
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("RMVPE")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, audio: np.ndarray, sr: int, *, voiced_threshold: float = 0.5) -> RmvpeResult:
        """Schätzt F0 per Frame via RMVPE ONNX oder pYIN-Fallback.

        Args:
            audio:             float32 mono oder stereo, beliebige SR
            sr:                Sample-Rate des Eingangs (muss 48000 sein)
            voiced_threshold:  Salience-Schwelle für stimmhaft/stimmlos ∈ [0, 1]

        Returns:
            RmvpeResult mit F0-Trajektorie, Konfidenz und Flags.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio if audio.ndim == 1 else audio.mean(axis=-1)
        mono = np.clip(mono, -1.0, 1.0)

        self._ensure_session()
        # Lokale Referenz statt self._session: Der PLM-unload_fn setzt nur
        # self._session = None — eine evictierte Session kann so die laufende
        # Inferenz nicht mehr annullieren (Race-Fund Rev. 2026-08-16).
        session = self._session
        if session is not None:
            return self._analyze_onnx(mono, sr, voiced_threshold, session=session)
        return self._analyze_pyin(mono, sr)

    # ------------------------------------------------------------------
    # Session-Selbstheilung (§v10.40, Rev. 2026-08-16)
    # ------------------------------------------------------------------

    def _ensure_session(self) -> None:
        """Stellt eine gültige ONNX-Session sicher.

        Der PluginLifecycleManager kann die Session unter RAM-Druck entladen
        (unload_fn setzt `_session = None`). Ohne Reload degradiert der
        §4.4-Pitch-Tracker danach still auf PESTO/pYIN (§V6-Fund, Benchmark:
        14/15 Läufe im DSP-Fallback). Hier: einmaliges transparentes Re-Load,
        wenn die Gewichte lokal vorhanden sind und das ML-Budget es zulässt.
        """
        if self._session is not None or self._model_loaded:
            return
        if not _ONNX_PATH.exists():
            return
        self._try_load()

    # ------------------------------------------------------------------
    # ONNX Inference
    # ------------------------------------------------------------------

    def _mel_spectrogram(self, mono_16k: np.ndarray) -> np.ndarray:
        """Berechnet Log-Mel-Spektrogramm [T, n_mels] für RMVPE-Input.

        Formel: mel = log(max(fb @ |STFT|^2, 1e-8))
        Filterbank: 128 Bänder, Hz↔Mel via f_mel = 2595·log10(1+f/700)
        """
        from scipy.signal import stft as scipy_stft

        n = len(mono_16k)
        if n < _FRAME_LEN:
            mono_16k = np.pad(mono_16k, (0, _FRAME_LEN - n))
        _, _, Z = scipy_stft(
            mono_16k.astype(np.float64),
            fs=_MODEL_SR,
            nperseg=_FRAME_LEN,
            noverlap=_FRAME_LEN - _HOP_LEN,
            window="hann",
        )
        mag_sq = np.abs(Z).astype(np.float32) ** 2  # [n_freq, T]
        n_freq = mag_sq.shape[0]
        # Mel-Filterbank aufbauen
        hz_max = float(_MODEL_SR) / 2.0
        mels = np.linspace(0.0, 2595.0 * math.log10(1.0 + hz_max / 700.0), _N_MELS + 2)
        hz_pts = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        freqs = np.linspace(0.0, hz_max, n_freq)
        fb = np.zeros((_N_MELS, n_freq), dtype=np.float32)
        for m in range(1, _N_MELS + 1):
            lo, ctr, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
            for k in range(n_freq):
                f = freqs[k]
                if lo <= f <= ctr and (ctr - lo) > 1e-10:
                    fb[m - 1, k] = (f - lo) / (ctr - lo)
                elif ctr < f <= hi and (hi - ctr) > 1e-10:
                    fb[m - 1, k] = (hi - f) / (hi - ctr)
        mel = fb @ mag_sq  # [n_mels, T]
        mel_log = np.log(np.maximum(mel, 1e-8)).astype(np.float32)
        return np.nan_to_num(mel_log, nan=0.0, posinf=0.0, neginf=-18.4).T  # type: ignore[no-any-return]  # [T, n_mels]

    def _analyze_onnx(
        self,
        mono_48k: np.ndarray,
        sr: int,
        voiced_threshold: float,
        *,
        session: object | None = None,
    ) -> RmvpeResult:
        """RMVPE ONNX-Inferenz: Mel → Salience-Map → F0."""
        session = session if session is not None else self._session
        assert session is not None
        from math import gcd

        from scipy.signal import resample_poly

        # 48 kHz → 16 kHz
        g = gcd(sr, _MODEL_SR)
        mono_16k = resample_poly(mono_48k, _MODEL_SR // g, sr // g).astype(np.float32)
        mono_16k = np.nan_to_num(mono_16k, nan=0.0, posinf=0.0, neginf=0.0)
        mono_16k = np.clip(mono_16k, -1.0, 1.0)

        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("RMVPE", True)
        except Exception:
            logger.warning("rmvpe_plugin.py::_analyze_onnx fallback", exc_info=True)
        try:
            mel = self._mel_spectrogram(mono_16k)  # [T, 128]
            t_orig = mel.shape[0]
            # U-Net decoder path expects compatible temporal down/up-sampling sizes.
            # Pad time-axis to a multiple of 32 to avoid concat shape mismatches.
            pad_t = (-t_orig) % 32
            if pad_t:
                mel = np.pad(mel, ((0, pad_t), (0, 0)), mode="edge")
            # RMVPE ONNX expects rank-3 input with mel channels first: [B, 128, T]
            inp = mel.T[np.newaxis]  # [1, 128, T]
            inp_name = cast(Any, session).get_inputs()[0].name
            # §ml-plugin-SKILL: Fixed-Shape-Input defensive guard.
            # If the ONNX model was exported with a fixed T dim (not dynamic), adjust.
            _inp_meta = cast(Any, session).get_inputs()[0]
            _t_meta_dim = _inp_meta.shape[2] if len(_inp_meta.shape) >= 3 else None
            if isinstance(_t_meta_dim, int) and _t_meta_dim > 0 and inp.shape[2] != _t_meta_dim:
                _t_fixed = int(_t_meta_dim)
                if inp.shape[2] < _t_fixed:
                    _extra = _t_fixed - inp.shape[2]
                    inp = np.pad(inp, ((0, 0), (0, 0), (0, _extra)), mode="edge")
                else:
                    inp = inp[:, :, :_t_fixed]
                logger.debug("RMVPE ONNX: fixed T=%d detected — input adjusted from %d", _t_fixed, inp.shape[2])
            ort_out = cast(Any, session).run(None, {inp_name: inp.astype(np.float32)})
            salience = np.asarray(ort_out[0], dtype=np.float32)  # [1, T, 360]
            if salience.ndim == 3:
                salience = salience[0]  # [T, 360]
            if pad_t:
                salience = salience[:t_orig]

            # RMVPE/CREPE-compatible cents mapping:
            #   cents_i = offset + i * 20, i=0..359
            cents_bins = _CENTS_OFFSET + np.arange(_PITCH_BINS, dtype=np.float32) * _CENTS_BIN_STEP
            max_sal = salience.max(axis=1)  # [T]
            voiced = max_sal >= voiced_threshold  # [T] bool

            # Local weighted decode around argmax (±4 bins) to avoid low-frequency bias.
            argmax_idx = np.argmax(salience, axis=1).astype(np.int32)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
            local_offsets = np.arange(-4, 5, dtype=np.int32)
            local_idx = np.clip(argmax_idx[:, None] + local_offsets[None, :], 0, _PITCH_BINS - 1)
            local_sal = salience[np.arange(salience.shape[0])[:, None], local_idx]
            local_cents = cents_bins[local_idx]
            local_sum = np.sum(local_sal, axis=1)
            cents = np.where(
                local_sum > 1e-9,
                np.sum(local_sal * local_cents, axis=1) / (local_sum + 1e-9),
                cents_bins[argmax_idx],
            ).astype(np.float32)

            # Conservative adaptive per-clip calibration:
            # If the clip is strongly tonal (single dominant spectral peak), align
            # the median voiced RMVPE estimate to the spectral reference.
            f0_raw = 10.0 * (2.0 ** (cents / 1200.0))
            voiced_raw = f0_raw[max_sal >= voiced_threshold]
            f_ref_hz, tonal_conf = _estimate_tonal_reference_hz(mono_16k)
            delta_cents = 0.0
            if f_ref_hz is not None and tonal_conf >= 8.0 and voiced_raw.size >= 8 and np.isfinite(voiced_raw).all():
                f_model_med = float(np.median(voiced_raw))
                if f_model_med > 1e-6:
                    delta_cents = float(1200.0 * math.log2(f_ref_hz / f_model_med))
                    delta_cents = float(np.clip(delta_cents, -250.0, 250.0))
                    cents = cents + delta_cents

            # Cents → Hz (same convention as CREPE): f = 10 * 2^(cents / 1200)
            f0_hz = 10.0 * (2.0 ** (cents / 1200.0))
            f0_hz = np.where(voiced, f0_hz, np.nan)
            f0_hz = np.nan_to_num(f0_hz, nan=np.nan)

            hop_time = _HOP_LEN / _MODEL_SR
            times = np.arange(len(f0_hz)) * hop_time
            voiced_f0 = f0_hz[voiced & np.isfinite(f0_hz)]
            f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0

            return RmvpeResult(
                f0=f0_hz.astype(np.float32),
                times=times.astype(np.float32),
                confidence=max_sal.astype(np.float32),
                voiced_flag=voiced,
                model_used="rmvpe_onnx",
                f0_mean=f0_mean,
                f0_std=f0_std,
                metadata={
                    "adaptive_calibration_cents": float(delta_cents),
                    "tonal_reference_hz": float(f_ref_hz) if f_ref_hz is not None else 0.0,
                    "tonal_confidence": float(tonal_conf),
                },
            )
        except Exception as exc:
            logger.warning("RMVPE ONNX-Inferenzfehler: %s — pYIN-Fallback.", exc)
            return self._analyze_pyin(mono_48k, sr)
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("RMVPE", False)
                except Exception:
                    logger.warning("rmvpe_plugin.py::unknown fallback", exc_info=True)

    def _analyze_pyin(self, mono_48k: np.ndarray, sr: int) -> RmvpeResult:
        """DSP-Fallback-Kette: PESTO (Riou et al. ISMIR 2023) → pYIN (Mauch & Dixon 2014).

        PESTO (dsp/pesto_pitch.py) ist ~8-20× schneller als pYIN bei vergleichbarer
        Genauigkeit für tonales Material. Fällt auf pYIN zurück bei PESTO-Fehler.
        §4.4: FCPE → RMVPE → PESTO → pYIN (letzte DSP-Stufe)
        """
        # Tier-DSP-1: PESTO (chromagram CQT, Riou et al. ISMIR 2023)
        try:
            from dsp.pesto_pitch import estimate_pitch as _pesto

            pesto_r = _pesto(mono_48k, sr)
            if pesto_r.f0_mean > 0 and np.sum(pesto_r.voiced) > 3:
                f0 = pesto_r.f0.astype(np.float32)
                voiced = pesto_r.voiced
                conf = pesto_r.confidence.astype(np.float32)
                times = pesto_r.times.astype(np.float32)
                voiced_f0 = f0[voiced & np.isfinite(f0)]
                return RmvpeResult(
                    f0=f0,
                    times=times,
                    confidence=conf,
                    voiced_flag=voiced,
                    model_used="pesto_dsp_fallback",
                    f0_mean=float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
                    f0_std=float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0,
                )
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("PESTO-Fallback fehlgeschlagen: %s — weiter mit pYIN", exc)

        try:
            import librosa

            f0, voiced_flag, voiced_prob = librosa.pyin(
                mono_48k,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
                frame_length=2048,
                hop_length=512,
            )
            f0 = np.nan_to_num(f0, nan=np.nan).astype(np.float32)
            times = librosa.times_like(f0, sr=sr, hop_length=512).astype(np.float32)
            voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
            f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0
            return RmvpeResult(
                f0=f0,
                times=times,
                confidence=voiced_prob.astype(np.float32),
                voiced_flag=voiced_flag,
                model_used="pyin_fallback",
                f0_mean=f0_mean,
                f0_std=f0_std,
            )
        except Exception as exc:
            logger.error("pYIN Fallback fehlgeschlagen: %s", exc)
            n = max(1, int(len(mono_48k) / 512))
            return RmvpeResult(
                f0=np.full(n, np.nan, dtype=np.float32),
                times=np.arange(n, dtype=np.float32) * (512.0 / sr),
                confidence=np.zeros(n, dtype=np.float32),
                voiced_flag=np.zeros(n, dtype=bool),
                model_used="pyin_error",
            )


# ---------------------------------------------------------------------------
# Singleton  (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_rmvpe_plugin() -> RmvpePlugin:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RmvpePlugin()
    return _instance


def analyze_pitch(audio: np.ndarray, sr: int, *, voiced_threshold: float = 0.5) -> RmvpeResult:
    """Convenience-Wrapper für get_rmvpe_plugin().analyze()."""
    return get_rmvpe_plugin().analyze(audio, sr, voiced_threshold=voiced_threshold)
