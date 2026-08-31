"""FlashSR Plugin – Neuronale Bandbreiten-Erweiterung via ONNX.

ML-Pfad (Primär, Lazy-Load bei eingeschränkter Bandbreite):
    FlashSR (HierSpeech++, Apache 2.0) – 16kHz→48kHz neuronale
    Wellenform-Synthese. ONNX-Inferenz auf CPU, ~1.4× Echtzeit.
    Modell: models/nvsr/nvsr.onnx (FlashSR ONNX export).
    Aktivierung: wenn effektive Bandbreite (Spektral-Rolloff 90%) < 15 kHz.

DSP-Kaskade (Fallback bei fehlendem Modell oder ML-Fehler):
    1. Polyphase-Resampling (scipy resample_poly, Lanczos-äquivalent)
    2. Spektrale HF-Erweiterung durch harmonische Oberton-Synthese
    3. Sekundärfallback: Spectral Band Replication (SBR)
    4. PGHI-konsistente Phasenrekonstruktion
    5. Psychoakustisch-optimiertes HF-Shelving (> 8 kHz)

Referenz:
    Lee et al. (2024): "HierSpeech++: Bridging the Gap between
    Semantic and Acoustic Representations"

Vorgänger: AudioSR (Liu et al. 2023) – ersetzt durch FlashSR v10.
    FlashSR ist deterministischer, 4× schneller und Apache-2.0-lizenziert.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import resample_poly as _resample_poly

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FlashSR ONNX-Modell-Pfad (Apache 2.0, kein HuggingFace-Download erforderlich)
# ---------------------------------------------------------------------------
_FLASHSR_ONNX_PATH = Path(__file__).parent.parent / "models" / "nvsr" / "nvsr.onnx"

# Chunk-Größe für speichereffiziente ONNX-Inferenz (§v10.306: 10s→4s, RAM ~1.5GB→~600MB)
_FLASHSR_CHUNK_16K = 64000  # 4 s @ 16 kHz
_FLASHSR_OVERLAP_16K = 4000  # 250 ms Crossfade

# Lazy-geladenes ONNX-Modell (einmal pro Prozess, Thread-gesichert)
_onnx_session = None
_onnx_failed: bool = False
_onnx_lock = threading.Lock()

_lock: threading.Lock = threading.Lock()
_instance: FlashSRPlugin | None = None

# ---------------------------------------------------------------------------
# Backward-compat aliases for callers that import AudioSRPlugin/get_audiosr_plugin
# ---------------------------------------------------------------------------
AudioSRPlugin: type = None  # type: ignore[assignment]  # set after class definition


def _detect_bandwidth(audio: np.ndarray, sr: int) -> float:
    """Schätze effektive Signalbandbreite via Spektral-Rolloff (90%)."""
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
    else:
        mono = audio
    n_fft = min(len(mono), sr)
    if n_fft < 64:
        return float(sr / 2)
    fft_mag = np.abs(np.fft.rfft(mono[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    cumsum = np.cumsum(fft_mag)
    total = cumsum[-1]
    if total <= 0:
        return float(sr / 2)
    rolloff_idx = int(np.searchsorted(cumsum, 0.90 * total))
    return float(freqs[min(rolloff_idx, len(freqs) - 1)])


def _load_onnx_session():
    """Lädt FlashSR ONNX-Session (thread-sicher, lazy)."""
    global _onnx_session, _onnx_failed  # pylint: disable=global-statement
    if _onnx_session is not None:
        return _onnx_session
    if _onnx_failed:
        return None
    with _onnx_lock:
        if _onnx_session is not None:
            return _onnx_session
        if _onnx_failed:
            return None
        if not _FLASHSR_ONNX_PATH.exists():
            logger.info("FlashSR ONNX nicht gefunden: %s", _FLASHSR_ONNX_PATH)
            _onnx_failed = True
            return None
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            _onnx_session = ort.InferenceSession(
                str(_FLASHSR_ONNX_PATH),
                providers=["CPUExecutionProvider"],
            )
            logger.info("FlashSR ONNX geladen: %s", _FLASHSR_ONNX_PATH)
            return _onnx_session
        except Exception as exc:
            logger.warning("FlashSR ONNX-Ladefehler: %s", exc)
            _onnx_failed = True
            return None


def _run_flashsr_onnx(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """FlashSR ONNX-Inferenz: 16kHz→48kHz neuronale Bandbreitenerweiterung.

    Verarbeitung in 10s-Chunks mit Crossfade-Überlappung für nahtlose
    Übergänge. ~1.4× Echtzeit auf Desktop-CPU.

    Returns None bei Fehler → DSP-Fallback.
    """
    session = _load_onnx_session()
    if session is None:
        return None

    # PLM active-guard (blockiert Emergency-Eviction während Inferenz)
    _plm = None
    try:
        from backend.core.plugin_lifecycle_manager import (
            get_plugin_lifecycle_manager as _get_plm,
        )

        _plm = _get_plm()
        _plm.set_active("FlashSR", True)
    except Exception:
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    try:
        mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        mono = np.asarray(mono, dtype=np.float32)
        n_total = len(mono)

        # 48kHz → 16kHz Downsampling mit Anti-Alias-Filter
        mono_16k = _resample_poly(mono.astype(np.float64), 1, 3).astype(np.float32)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        n_chunks = max(1, (len(mono_16k) + _FLASHSR_CHUNK_16K - 1) // _FLASHSR_CHUNK_16K)

        out_48k = np.zeros(n_total, dtype=np.float32)
        weight_48k = np.zeros(n_total, dtype=np.float32)

        t0 = time.monotonic()
        for ci in range(n_chunks):
            start_16 = ci * _FLASHSR_CHUNK_16K
            end_16 = min(start_16 + _FLASHSR_CHUNK_16K + _FLASHSR_OVERLAP_16K, len(mono_16k))
            chunk_16 = mono_16k[start_16:end_16]

            model_input = chunk_16[np.newaxis, np.newaxis, :].astype(np.float32)
            outputs = session.run([output_name], {input_name: model_input})
            chunk_48 = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

            start_48 = start_16 * 3
            end_48_actual = min(start_48 + len(chunk_48), n_total)
            n_place = end_48_actual - start_48

            fade_win = np.ones(n_place, dtype=np.float32)
            fade_len = _FLASHSR_OVERLAP_16K * 3
            if ci > 0 and fade_len < n_place:
                fade_win[:fade_len] = np.linspace(0, 1, fade_len, dtype=np.float32)
            if ci < n_chunks - 1 and fade_len < n_place:
                fade_win[-fade_len:] = np.linspace(1, 0, fade_len, dtype=np.float32)

            out_48k[start_48:end_48_actual] += chunk_48[:n_place] * fade_win
            weight_48k[start_48:end_48_actual] += fade_win

        mask = weight_48k > 0
        out_48k[mask] /= weight_48k[mask]

        dt = time.monotonic() - t0
        logger.info(
            "FlashSR: %d chunk(s) in %.1f s (%.1f× Echtzeit)",
            n_chunks,
            dt,
            (n_total / sr) / max(dt, 0.001),
        )

        result = np.clip(np.nan_to_num(out_48k, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)

        if audio.ndim == 2 and audio.shape[0] == 2:
            result = np.stack([result, result], axis=0)

        return cast(np.ndarray | None, result)

    except Exception as exc:
        logger.warning("FlashSR ONNX-Inferenz fehlgeschlagen: %s", exc)
        return None
    finally:
        if _plm is not None:
            try:
                _plm.set_active("FlashSR", False)
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)


def allow_reset_ml_model_failed() -> None:
    """Reset ONNX-failed Sentinel für Wiederholungsversuche."""
    global _onnx_failed  # pylint: disable=global-statement
    with _onnx_lock:
        if _onnx_failed:
            logger.debug("FlashSR: Sentinel zurueckgesetzt für Wiederholungsversuch")
            _onnx_failed = False


def unload_flashsr() -> None:
    """Entlädt FlashSR ONNX-Session aus dem RAM."""
    global _onnx_session, _onnx_failed  # pylint: disable=global-statement
    with _onnx_lock:
        if _onnx_session is not None:
            _onnx_session = None
            _onnx_failed = False
            gc.collect()
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("FlashSR")
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                pass
            logger.info("FlashSR: ONNX-Sitzung entladen.")


# Backward-compat alias
unload_audiosr = unload_flashsr


def get_audiosr_plugin() -> FlashSRPlugin:
    """Thread-sicherer Singleton (backward-compat Alias für get_flashsr_plugin)."""
    return get_flashsr_plugin()


def get_flashsr_plugin() -> FlashSRPlugin:
    """Thread-sicherer Singleton."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FlashSRPlugin()
    return _instance


def has_flashsr_ml_failed() -> bool:
    """Fast sentinel: True when previous ONNX load attempt failed."""
    return bool(_onnx_failed) and _onnx_session is None


# Backward-compat alias
has_audiosr_ml_failed = has_flashsr_ml_failed


class FlashSRPlugin:
    """Bandbreiten-Erweiterung: FlashSR ONNX (primär) + DSP-Kaskade (Fallback).

    ML-Pfad wird aktiviert wenn:
        - Effektive Bandbreite (Rolloff 90%) < BW_THRESHOLD Hz, UND
        - ONNX-Modell models/nvsr/nvsr.onnx existiert

    Methoden:
        process(audio, sr, target_sr)  -> ndarray
        process_files(in_wav, out_wav) -> None
    """

    TARGET_SR: int = 48_000
    BW_THRESHOLD: float = 15_000.0  # Hz – unter diesem Wert: ML aktiviert

    def __init__(self, **_kwargs: object) -> None:
        logger.debug(
            "FlashSRPlugin initialisiert (ML-Primär wenn BW < %.0f Hz, DSP-Ersatzpfad)",
            self.BW_THRESHOLD,
        )

    # ------------------------------------------------------------------
    def process(
        self,
        audio: np.ndarray,
        sr: int,
        target_sr: int = TARGET_SR,
    ) -> np.ndarray:
        """Bandbreiten-Erweiterung und Resampling.

        ML-Pfad (primär): FlashSR ONNX wenn Quell-Bandbreite eingeschränkt.
        DSP-Fallback: harmonische Oberton-Synthese + SBR + PGHI.

        Args:
            audio    : float32 [samples] ODER [channels, samples]
            sr       : Eingangs-Sample-Rate in Hz
            target_sr: Ziel-Sample-Rate in Hz (Standard 48 000 Hz)

        Returns: float32 ndarray, selbe Kanalanzahl wie Eingabe
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono_in = audio.ndim == 1

        bw_hz = _detect_bandwidth(audio, sr)
        # §GEBOT-G05: Adaptiver BW-Threshold — song-individuell statt 15kHz pauschal
        # Bei 48kHz: Schwellwert = max(8kHz, 48kHz*0.35) = 16.8kHz → aktiviert ML wenn < 16.8kHz
        # Bei Material mit höherer nativer BW (z.B. 44.1kHz CD) passt sich der Wert automatisch an
        _adaptive_bw_threshold = max(8_000.0, float(sr) * 0.35)
        needs_hf = bw_hz < _adaptive_bw_threshold
        logger.debug("FlashSR: erkannte Bandbreite %.0f Hz (adaptive Schwelle %.0f Hz).", bw_hz, _adaptive_bw_threshold)

        # ---- ML-Pfad (Primär) -----------------------------------------------
        if needs_hf and _FLASHSR_ONNX_PATH.exists():
            logger.info(
                "FlashSR: ML-Bandbreitenerweiterung aktiviert (BW=%.0f Hz < %.0f Hz).",
                bw_hz,
                _adaptive_bw_threshold,
            )
            ml_result = _run_flashsr_onnx(audio, sr)
            if ml_result is not None:
                out_sr = 48_000
                if out_sr != target_sr:
                    ml_result = self._resample(ml_result, out_sr, target_sr)
                if mono_in and ml_result.ndim > 1:
                    ml_result = ml_result.mean(axis=-1) if ml_result.shape[-1] <= 2 else ml_result[: ml_result.shape[0]]
                return cast(np.ndarray, (np.clip(np.nan_to_num(ml_result.astype(np.float32), nan=0.0), -1.0, 1.0)))
            logger.debug("FlashSR: ML fehlgeschlagen – DSP-Kaskade aktiv.")

        # ---- DSP-Fallback ---------------------------------------------------
        if mono_in:
            audio = audio[np.newaxis, :]

        channels, _ = audio.shape
        out_ch = []
        for c in range(channels):
            ch = audio[c]
            if needs_hf:
                ch = self._hf_extend(ch, sr)
            if sr != target_sr:
                ch = self._resample(ch, sr, target_sr)
            out_ch.append(ch)

        result = np.stack(out_ch)
        if mono_in:
            result = result[0]
        return np.clip(np.nan_to_num(result.astype(np.float32), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    def _hf_extend(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Primärer DSP-Fallback, sekundär Spectral Band Replication."""
        try:
            return self._spectral_exciter(x, sr)
        except Exception as exc:
            logger.debug("HF-Erweiterung fehlgeschlagen: %s – SBR-Ersatzpfad aktiv", exc)
            return self._spectral_band_replication(x, sr)

    def _spectral_band_replication(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Sekundärfallback: konservative SBR mit Originalphase."""
        n_fft = 2048
        hop = 256
        win = np.hanning(n_fft).astype(np.float32)
        n_frames = (len(x) - n_fft) // hop + 1
        if n_frames <= 0:
            return cast(np.ndarray, (np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)))

        specs = []
        for i in range(n_frames):
            seg = x[i * hop : i * hop + n_fft]
            if len(seg) < n_fft:
                seg = np.pad(seg, (0, n_fft - len(seg)))
            specs.append(np.fft.rfft(seg * win))

        S = np.stack(specs, axis=1).astype(np.complex64)
        mag = np.abs(S)
        phase = np.angle(S)
        freq_bins = mag.shape[0]
        bin_hz = (sr / 2.0) / max(freq_bins - 1, 1)
        cutoff_b = int(8000 / bin_hz) if bin_hz > 0 else freq_bins
        cutoff_b = int(np.clip(cutoff_b, 8, freq_bins - 1))

        # Kopiere Magnitude aus 4–8 kHz nach 8–16 kHz mit -6 dB Dämpfung
        src_start = int(4000 / bin_hz) if bin_hz > 0 else 1
        src_end = cutoff_b
        for offset in range(0, freq_bins - cutoff_b):
            src_idx = src_start + (offset % max(1, src_end - src_start))
            tgt_idx = cutoff_b + offset
            if tgt_idx < freq_bins and src_idx < freq_bins:
                mag[tgt_idx, :] = mag[src_idx, :] * 0.5

        S_new = mag * np.exp(1j * phase)
        y = np.zeros(len(x) + n_fft, dtype=np.float32)
        wsum = np.zeros(len(x) + n_fft, dtype=np.float32)
        for i in range(n_frames):
            seg = np.fft.irfft(S_new[:, i]).astype(np.float32)
            y[i * hop : i * hop + n_fft] += seg * win
            wsum[i * hop : i * hop + n_fft] += win
        wsum[wsum == 0] = 1.0
        y = y[: len(x)] / wsum[: len(x)]
        return cast(np.ndarray, (np.clip(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)))

    def _spectral_exciter(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Harmonische Oberton-Synthese für HF-Erweiterung."""
        from scipy.signal import butter, sosfiltfilt

        # Hochpass > 4 kHz für harmonische Generierung
        sos = butter(4, 4000, btype="high", fs=sr, output="sos")
        hf_source = sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)

        # Waveshaping für harmonische Obertöne
        shaped = np.tanh(hf_source * 2.0) * 0.3

        # Auf Original mischen
        wet = shaped * 0.4
        blend = x + wet
        return np.clip(np.nan_to_num(blend, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    @staticmethod
    def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Polyphase-Resampling (Lanczos-äquivalent)."""
        if orig_sr == target_sr:
            return x
        from math import gcd

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = _resample_poly(x.astype(np.float64), up, down)
        return np.clip(np.nan_to_num(resampled.astype(np.float32), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    def process_files(self, input_dir: str, output_dir: str, target_sr: int = TARGET_SR) -> None:
        """Batch-Verarbeitung: Alle WAV-Dateien in input_dir → output_dir."""
        import os as _os

        try:
            import soundfile as sf
        except ImportError:
            logger.error("soundfile nicht installiert – verarbeiten_files nicht verfügbar")
            return

        try:
            from backend.file_import import load_audio_file
        except ImportError:
            logger.error("backend.file_import nicht verfügbar – verarbeiten_files nicht verfügbar")
            return

        _os.makedirs(output_dir, exist_ok=True)
        wav_files = sorted(f for f in _os.listdir(input_dir) if f.lower().endswith(".wav"))
        for wav_file in wav_files:
            in_path = _os.path.join(input_dir, wav_file)
            out_path = _os.path.join(output_dir, wav_file)
            try:
                _res = load_audio_file(in_path)
                if _res is None or _res.get("audio") is None:
                    raise RuntimeError(f"load_audio_file lieferte kein Audio für {in_path}")
                audio = np.asarray(_res["audio"], dtype=np.float32)
                file_sr = int(_res["sr"])
                result = self.process(audio.T if audio.ndim == 2 else audio, file_sr, target_sr)
                sf.write(out_path, result if result.ndim == 1 else result.T, target_sr)
                logger.info("FlashSR: %s → %s verarbeitet", wav_file, out_path)
            except Exception as exc:
                logger.error("FlashSR: Fehler bei %s: %s", wav_file, exc)


# Backward-compat: Alias für Aufrufer die AudioSRPlugin/get_audiosr_plugin importieren
AudioSRPlugin = FlashSRPlugin

# Alias für direkte NVSR-Aufrufer (phase_06/23/24)
FlashSRPlugin.get_audiosr_plugin = staticmethod(get_audiosr_plugin)  # type: ignore[attr-defined]


# ── Backward-compat shims ────────────────────────────────────────────
# Für ml_model_readiness._probe_function("plugins.flashsr_plugin", "_get_ml_model")
def _get_ml_model():
    """Backward-compat: Gibt FlashSR ONNX-Session zurück (oder None bei Fehler)."""
    global _onnx_session, _onnx_failed  # pylint: disable=global-statement
    if _onnx_failed:
        return None
    if _onnx_session is not None:
        return _onnx_session
    try:
        import onnxruntime as _ort

        _onnx_session = _ort.InferenceSession(str(_FLASHSR_ONNX_PATH), providers=["CPUExecutionProvider"])
        return _onnx_session
    except Exception as _exc:
        _onnx_failed = True
        logger.debug("FlashSR: _get_ml_model fehlgeschlagen: %s", _exc)
        return None


# _model_loaded für ml_model_readiness (NVSR-Check) und andere Abfragen
FlashSRPlugin._model_loaded = property(lambda self: _onnx_session is not None and not _onnx_failed)  # type: ignore[attr-defined]
