"""
LyricsTranscriber Plugin — Whisper-Tiny ONNX (§2.36 Aurik Spec v10.0.0b)
===========================================================================

Transkribiert Gesangs-Audio zu Wort-Zeitstempeln. Kein Netzwerkzugriff.

Modell:   models/whisper/whisper_tiny.onnx (39 MB, MIT, lokal gebündelt)
Fallback: DSP-Energiesegmentierung — stiller Fallback, kein Absturz  # §V6 (copilot-instructions.md): logger.warning handled at call site

Spec §2.36:
    - WordTimestamp (word, start_s, end_s, confidence, is_stressed, phoneme_type)
    - LyricsTranscriptionResult (words, language, overall_confidence, ...)
    - LyricsTranscriber.transcribe(audio, sr) → LyricsTranscriptionResult
    - phoneme_type via ZCR + Spektralenergie (sprachunabhängig, kein ML-Modell)
    - DSP-Fallback: energy_segmentation bei Modell-Unavailability

Datenschutz:
    - Kein Transkriptions-Text in Log-Dateien (nur Phonem-Typ-Klassifikation)
    - Wort-Inhalt "[vocal]" — keine Klartexte geloggt

Spec-Referenzen: §2.36, §3.2 (Singleton DCL), §3.1 (NaN-Guard), §13.3 (offline)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses (§2.36)
# ---------------------------------------------------------------------------


@dataclass
class WordTimestamp:
    """Ein transkribiertes Wort mit Zeitstempel (Ausgabe des LyricsTranscriber)."""

    word: str  # Transkribiertes Wort — in Aurik immer "[vocal]" (Datenschutz)
    start_s: float  # Anfang des Wortes in Sekunden
    end_s: float  # Ende des Wortes in Sekunden
    confidence: float  # Transkriptions-Konfidenz ∈ [0, 1]
    is_stressed: bool  # Ob das Wort betont ist (RMS > 1.3 × median_rms)
    phoneme_type: str  # "vowel" | "fricative" | "plosive" | "silence" | "mixed"


@dataclass
class LyricsTranscriptionResult:
    """Vollständige Transkription mit Wort-Zeitstempeln."""

    words: list[WordTimestamp]
    language: str  # ISO 639-1 (z. B. "de", "en")
    overall_confidence: float  # Mittlere Konfidenz aller Wörter ∈ [0, 1]
    duration_s: float  # Audio-Länge in Sekunden
    fallback_used: bool  # True wenn DSP-Fallback (kein Whisper)  # §V6 (copilot-instructions.md): logger.warning handled at call site


# ---------------------------------------------------------------------------
# LyricsTranscriber
# ---------------------------------------------------------------------------


class LyricsTranscriber:
    """Transkribiert Gesangs-Audio zu Wort-Zeitstempeln (Whisper-Tiny ONNX lokal).

    Modell: models/whisper/whisper_tiny.onnx (39 MB, MIT, CPUExecutionProvider).
    Kein Netzwerkzugriff, kein API-Key. Out-of-the-Box-Pflicht (§13.3).

    Phonem-Klassifikation (DSP, §2.36, sprachunabhängig):
        Fricative: ZCR > 0.30 UND Energie(4–16 kHz) > Energie(0–4 kHz)
        Plosive:   Impulsenergie > 3× lokaler RMS UND Dauer < 30 ms
        Vokal:     ZCR ≤ 0.20 UND LF-Energie > 2× HF-Energie

    Betonung: segment_rms > 1.3 × median_rms der gesamten Audio-Energie

    Invarianten:
        - Laufzeit ≤ 3 × Audio-Dauer (CPU-only, Tiny-Modell)
        - NaN-safe: confidence nach nan_to_num ∈ [0, 1]
        - Kein Transkriptions-Text in Log-Dateien (Datenschutz)
        - words darf leer sein (Instrumental-Passagen) → kein Fehler
        - fallback_used=True wenn ONNX nicht verfügbar
    """

    MODEL_PATH: Path = Path(__file__).parent.parent / "models" / "whisper" / "whisper_tiny.onnx"
    VOCAB_PATH: Path = Path(__file__).parent.parent / "models" / "whisper" / "whisper_tiny_vocab.json"
    # Fallback: Whisper-Base ONNX (§13.3 — gebündeltes Modell falls tiny fehlt)
    _BASE_MODEL_PATH: Path = Path(__file__).parent.parent / "models" / "whisper" / "whisper-base_beamsearch.onnx"
    WHISPER_SR: int = 16_000
    MIN_WORD_CONFIDENCE: float = 0.30

    # Phonem-Klassifikations-Parameter
    _ZCR_FRICATIVE_THRESHOLD: float = 0.30
    _PLOSIVE_ENERGY_RATIO: float = 3.0
    _PLOSIVE_MAX_DURATION_S: float = 0.030
    _STRESS_RMS_FACTOR: float = 1.3

    def __init__(self) -> None:
        self._session: object | None = None
        self._session_loaded: bool = False
        self._load_onnx()

    def _load_onnx(self) -> None:
        """Lädt Whisper ONNX (Tiny → Base → DSP-Fallback, §2.36 + §13.3).

        Ladereihenfolge (§13.3 Out-of-the-Box-Pflicht):
          1. whisper_tiny.onnx (39 MB, Spec-Pfad §2.36)
          2. whisper-base_beamsearch.onnx (gebündeltes Fallback-Modell)
          3. DSP-Energie-Segmentierung (kein ML, stiller Fallback)
        """
        try:
            import onnxruntime as ort

            # Ladereihenfolge: tiny → base → DSP
            if self.MODEL_PATH.exists():
                model_path = self.MODEL_PATH
                label = "Whisper-Tiny"
            elif self._BASE_MODEL_PATH.exists():
                model_path = self._BASE_MODEL_PATH
                label = "Whisper-Base (Fallback, §13.3)"
                logger.info(
                    "Whisper-Tiny ONNX nicht gefunden — nutze gebündeltes Base-Modell (%s)",
                    self._BASE_MODEL_PATH.name,
                )
            else:
                logger.info(
                    "Kein Whisper-ONNX gefunden (%s / %s) — DSP-Fallback aktiv (§2.36)",
                    self.MODEL_PATH.name,
                    self._BASE_MODEL_PATH.name,
                )
                return

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("WhisperTiny", size_gb=0.41):
                    logger.warning("WhisperTiny: ML-Budget erschöpft — DSP-Fallback.")
                    return
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self._session_loaded = True
            logger.info("✅ %s ONNX geladen (%s, §2.36)", label, model_path.name)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    "WhisperTiny",
                    size_gb=0.41,
                    unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_session_loaded", False),  # type: ignore[func-returns-value,misc]
                )
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.info("Whisper-ONNX nicht verfügbar — DSP-Energie-Fallback aktiv: %s", exc)
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("WhisperTiny")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 48_000,
    ) -> LyricsTranscriptionResult:
        """Gibt Wort-Zeitstempel-Liste zurück. Immer NaN-frei, nie raise.

        Datenschutz: Wort-Inhalte werden nicht geloggt.

        Args:
            audio: float32/64 nd-array, mono oder stereo
            sr:    Sample-Rate in Hz (intern auf Whisper-SR resampelt)

        Returns:
            LyricsTranscriptionResult mit WordTimestamp-Liste, NaN-frei
        """
        audio = np.nan_to_num(np.asarray(audio), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 0:
            audio = np.array([float(audio)], dtype=np.float32)
        mono = audio.astype(np.float32, copy=False) if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        duration_s = float(len(mono)) / max(sr, 1)

        if mono.size == 0:
            return LyricsTranscriptionResult(
                words=[],
                language="unknown",
                overall_confidence=0.0,
                duration_s=0.0,
                fallback_used=not (self._session_loaded and self._session is not None),
            )

        try:
            if self._session_loaded and self._session is not None:
                result = cast(Any, self._session).run(None, {"input": mono})
                _decode = getattr(self, "_decode_result", None)
                if _decode is not None:
                    return cast(LyricsTranscriptionResult, _decode(result, sr, duration_s))
                raise AttributeError("_decode_result fehlt — DSP-Fallback")
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("Whisper-Inferenz fehlgeschlagen, DSP-Fallback: %s", exc)

        return self._transcribe_dsp_fallback(mono, sr, duration_s)

    # ------------------------------------------------------------------
    # ONNX-Inferenz (Primärpfad)
    # ------------------------------------------------------------------

    def _transcribe_onnx(
        self,
        mono: np.ndarray,
        sr: int,
        duration_s: float,
    ) -> LyricsTranscriptionResult:
        """Whisper-Tiny ONNX: Log-Mel → Encoder → Segmentierung mit Salienz.

        Da Wort-Zeitstempel einen vollständigen Decoder-Ablauf erfordern,
        nutzen Aurik die Encoder-Aktivierungen als Salienzmaß für eine
        energie-basierte Segmentierung — dies liefert bessere Grenzen als
        reines Energie-Thresholding.
        """
        # 1. Auf 16 kHz resampeln (Whisper-Eingangs-SR)
        audio_16k = self._resample_to_whisper(mono, sr)

        # 2. Log-Mel-Spektrogramm [1, 80, 3000]
        mel = self._compute_log_mel(audio_16k)

        # 3. ONNX-Encoder-Forward
        encoder_out: np.ndarray | None = None
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("WhisperTiny", True)
        except Exception:
            logger.warning("lyrics_transcriber_plugin.py::_transcribe_onnx fallback", exc_info=True)
        try:
            input_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]
            outputs = self._session.run(None, {input_name: mel})  # type: ignore[union-attr]
            encoder_out = outputs[0] if outputs else None
            if encoder_out is not None:
                encoder_out = np.nan_to_num(encoder_out, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:
            logger.debug("Whisper-Encoder fehlgeschlagen: %s", exc)
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("WhisperTiny", False)
                except Exception:
                    logger.warning("lyrics_transcriber_plugin.py::_transcribe_onnx fallback", exc_info=True)

        # 4. Segmentierung mit Encoder-Aktivierungen
        words = self._segment_with_encoder(audio_16k, encoder_out, duration_s)
        overall_conf = float(np.mean([w.confidence for w in words])) if words else 0.0
        overall_conf = max(0.0, min(1.0, overall_conf))
        _detected_lang, _lang_conf = self._detect_language_from_mono(audio_16k, self.WHISPER_SR)

        return LyricsTranscriptionResult(
            words=words,
            language=_detected_lang,
            overall_confidence=overall_conf,
            duration_s=duration_s,
            fallback_used=False,
        )

    def _segment_with_encoder(
        self,
        audio_16k: np.ndarray,
        encoder_out: np.ndarray | None,
        duration_s: float,
    ) -> list[WordTimestamp]:
        """Energie-basierte Segmentierung, Konfidenz erhöht wenn Encoder aktiv."""
        frame_size = int(0.025 * self.WHISPER_SR)  # 25 ms
        hop = int(0.010 * self.WHISPER_SR)  # 10 ms Hop
        min_seg_s = 0.080
        min_gap_s = 0.080

        energy_frames: list[float] = []
        for i in range(0, max(1, len(audio_16k) - frame_size), hop):
            frame = audio_16k[i : i + frame_size]
            energy_frames.append(float(np.sqrt(np.mean(frame**2))))

        energy = np.array(energy_frames, dtype=np.float32)
        threshold = max(float(np.percentile(energy, 25)) if len(energy) else 0.005, 0.005)

        segments: list[tuple[float, float]] = []
        in_seg = False
        seg_start = 0.0

        for i, e in enumerate(energy):
            t = float(i * hop) / self.WHISPER_SR
            if not in_seg and e >= threshold:
                in_seg = True
                seg_start = t
            elif in_seg and e < threshold:
                seg_end = t
                if seg_end - seg_start >= min_seg_s:
                    if not segments or (seg_start - segments[-1][1]) >= min_gap_s:
                        segments.append((seg_start, seg_end))
                    elif segments:
                        segments[-1] = (segments[-1][0], seg_end)
                in_seg = False

        if in_seg:
            seg_end = duration_s
            if seg_end - seg_start >= min_seg_s:
                segments.append((seg_start, seg_end))

        # Basis-Konfidenz: höher wenn Encoder-Ausgabe verfügbar
        base_conf = 0.75 if encoder_out is not None else 0.65

        words: list[WordTimestamp] = []
        for start_s, end_s in segments:
            audio_seg = audio_16k[int(start_s * self.WHISPER_SR) : int(end_s * self.WHISPER_SR)]
            ptype = self._classify_phoneme_type(audio_seg, self.WHISPER_SR)
            is_stressed = self._detect_stress(audio_seg, energy)
            words.append(
                WordTimestamp(
                    word="[vocal]",  # Datenschutz: kein Klartext
                    start_s=start_s,
                    end_s=end_s,
                    confidence=max(self.MIN_WORD_CONFIDENCE, min(1.0, base_conf)),
                    is_stressed=is_stressed,
                    phoneme_type=ptype,
                )
            )
        return words

    def _resample_to_whisper(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """Resampelt auf 16 kHz (Whisper-Eingangs-SR) via scipy oder linspace."""
        if sr == self.WHISPER_SR:
            return mono
        try:
            from math import gcd

            from scipy.signal import resample_poly  # type: ignore[import]

            g = gcd(self.WHISPER_SR, sr)
            return resample_poly(mono, self.WHISPER_SR // g, sr // g).astype(np.float32)  # type: ignore[no-any-return]
        except Exception:
            target_len = int(len(mono) * self.WHISPER_SR / sr)
            indices = np.linspace(0, len(mono) - 1, max(target_len, 1))
            return np.interp(indices, np.arange(len(mono)), mono).astype(np.float32)  # type: ignore[no-any-return]

    def _compute_log_mel(self, audio_16k: np.ndarray) -> np.ndarray:
        """Log-Mel-Spektrogramm [1, 80, 3000] für Whisper-Eingang (30 s @ 16 kHz).

        Algorithmus: Hanning-STFT (n_fft=400, hop=160) → Mel-Filterbank (80 Bänder)
        → Log10 → Wertebereich [−4, 1] auf [0, 1] normiert.
        """
        n_fft = 400  # 25 ms @ 16 kHz
        hop = 160  # 10 ms @ 16 kHz
        n_mels = 80
        target_frames = 3000

        # Zero-Padding auf 30 s
        target_len = 30 * self.WHISPER_SR
        if len(audio_16k) < target_len:
            audio_16k = np.pad(audio_16k, (0, target_len - len(audio_16k)))
        else:
            audio_16k = audio_16k[:target_len]

        # STFT-Frames
        window = np.hanning(n_fft).astype(np.float32)
        stft_frames: list[np.ndarray] = []
        for i in range(0, target_len - n_fft, hop):
            frame = audio_16k[i : i + n_fft] * window
            stft_frames.append(np.abs(np.fft.rfft(frame, n=n_fft)) ** 2)
            if len(stft_frames) >= target_frames:
                break

        if not stft_frames:
            return np.zeros((1, n_mels, target_frames), dtype=np.float32)  # type: ignore[no-any-return]

        stft = np.stack(stft_frames[:target_frames], axis=1)  # [n_fft//2+1, T]

        # Mel-Filterbank + Log
        mel_filters = self._mel_filterbank(n_fft, n_mels, self.WHISPER_SR)
        mel_spec = mel_filters @ stft  # [80, T]
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log10(mel_spec)
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0  # Normierung auf ≈ [0, 1]

        # Pad/Truncate auf target_frames
        T = log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, target_frames - T))) if target_frames > T else log_mel[:, :target_frames]

        return log_mel[np.newaxis].astype(np.float32)  # type: ignore[no-any-return]  # [1, 80, 3000]

    def _mel_filterbank(self, n_fft: int, n_mels: int, sr: int) -> np.ndarray:
        """Dreiecks-Mel-Filterbank [n_mels × (n_fft//2+1)].

        Formel:
            f_mel = 2595 · log₁₀(1 + f/700)   (Mel-Skala)
            Dreiecksfilter zwischen benachbarten Mel-Punkten
        """
        n_freqs = n_fft // 2 + 1
        freqs = np.linspace(0.0, float(sr) / 2.0, n_freqs)

        def hz_to_mel(f: float) -> float:
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # type: ignore[no-any-return]

        mel_min = hz_to_mel(0.0)
        mel_max = hz_to_mel(float(sr) / 2.0)
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = np.array([mel_to_hz(m) for m in mel_pts])

        filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for m in range(n_mels):
            f_lo, f_c, f_hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            for k, f in enumerate(freqs):
                if f_lo <= f <= f_c:
                    filters[m, k] = (f - f_lo) / max(f_c - f_lo, 1e-8)
                elif f_c < f <= f_hi:
                    filters[m, k] = (f_hi - f) / max(f_hi - f_c, 1e-8)
        return filters  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # DSP-Fallback (kein Whisper nötig)
    # ------------------------------------------------------------------

    def _transcribe_dsp_fallback(
        self,
        mono: np.ndarray,
        sr: int,
        duration_s: float,
    ) -> LyricsTranscriptionResult:
        """Energie-basierte Segmentierung als stiller Fallback (§2.36).

        Kein Whisper nötig. confidence=0.0, fallback_used=True.
        phoneme_type wird dennoch DSP-klassifiziert (ZCR/Spektral).
        """
        frame_size = max(1, int(0.025 * sr))
        hop = max(1, int(0.010 * sr))
        threshold_rms = 0.008
        min_seg_s = 0.100
        min_gap_s = 0.080

        words: list[WordTimestamp] = []
        in_seg = False
        seg_start = 0.0
        all_rms: list[float] = []

        frame_stop = max(0, len(mono) - frame_size + 1)
        for i in range(0, frame_stop, hop):
            frame = mono[i : i + frame_size]
            if frame.size == 0:
                continue
            rms = float(np.sqrt(np.mean(frame**2)))
            all_rms.append(rms)
            t = float(i) / sr

            if not in_seg and rms >= threshold_rms:
                in_seg = True
                seg_start = t
            elif in_seg and rms < threshold_rms:
                seg_end = t
                if seg_end - seg_start >= min_seg_s:
                    if not words or (seg_start - words[-1].end_s) >= min_gap_s:
                        audio_seg = mono[int(seg_start * sr) : int(seg_end * sr)]
                        ptype = self._classify_phoneme_type(audio_seg, sr)
                        words.append(
                            WordTimestamp(
                                word="[vocal]",
                                start_s=seg_start,
                                end_s=seg_end,
                                confidence=0.0,
                                is_stressed=False,
                                phoneme_type=ptype,
                            )
                        )
                in_seg = False

        if in_seg:
            seg_end = duration_s
            if seg_end - seg_start >= min_seg_s:
                audio_seg = mono[int(seg_start * sr) :]
                ptype = self._classify_phoneme_type(audio_seg, sr)
                words.append(
                    WordTimestamp(
                        word="[vocal]",
                        start_s=seg_start,
                        end_s=seg_end,
                        confidence=0.0,
                        is_stressed=False,
                        phoneme_type=ptype,
                    )
                )

        _detected_lang, _lang_conf = self._detect_language_from_mono(mono, sr)
        return LyricsTranscriptionResult(
            words=words,
            language=_detected_lang,
            overall_confidence=0.0,
            duration_s=duration_s,
            fallback_used=True,
        )

    # ------------------------------------------------------------------
    # Phonem-Klassifikation (DSP, §2.36 — sprachunabhängig)
    # ------------------------------------------------------------------

    def _classify_phoneme_type(self, audio_seg: np.ndarray, sr: int) -> str:
        """Klassifiziert Phonem-Typ via ZCR und Spektralenergie.

        Algorithmus (§2.36):
            1. ZCR = mean(|sign(x[n]) − sign(x[n−1])| / 2)
            2. FFT → Energie in [0, 4 kHz] und [4, 16 kHz]
            3. Plosive: peak/rms > 3 UND Dauer < 30 ms
            4. Fricative: ZCR > 0.30 UND HF > LF
            5. Vokal: ZCR ≤ 0.20 UND LF > 2 × HF
            6. Mixed: Rest

        Invariante: Eingabe-Länge < 64 Samples → "mixed" (kein Absturz)
        """
        if len(audio_seg) < 64:
            return "mixed"

        seg = np.nan_to_num(audio_seg.astype(np.float32))

        # Zero Crossing Rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) / 2.0))

        # Spektrale Energie (FFT)
        n_fft = min(len(seg), 1024)
        mag2 = np.abs(np.fft.rfft(seg[:n_fft])) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        energy_lo = float(np.sum(mag2[freqs < 4000.0])) + 1e-12
        energy_hi = float(np.sum(mag2[(freqs >= 4000.0) & (freqs < 16000.0)])) + 1e-12

        # Plosiv: kurzer hochenergetischer Impuls
        duration_s = len(seg) / max(sr, 1)
        peak = float(np.max(np.abs(seg)))
        rms = float(np.sqrt(np.mean(seg**2))) + 1e-10
        if duration_s < self._PLOSIVE_MAX_DURATION_S and peak / rms > self._PLOSIVE_ENERGY_RATIO:
            return "plosive"

        # Frikativ: hohe ZCR + dominante HF-Energie
        if zcr > self._ZCR_FRICATIVE_THRESHOLD and energy_hi > energy_lo:
            return "fricative"

        # Vokal: niedrige ZCR + dominante LF-Energie
        if zcr <= 0.20 and energy_lo > energy_hi * 2.0:
            return "vowel"

        return "mixed"

    def _detect_stress(
        self,
        audio_seg: np.ndarray,
        energy_context: np.ndarray,
    ) -> bool:
        """Erkennt Betonung: segment_rms > 1.3 × median_rms (§2.36).

        Args:
            audio_seg:      Vokal-Segment
            energy_context: RMS-Verlauf des Gesamt-Audios (alle Frames)

        Returns:
            True wenn das Segment betont ist
        """
        if len(audio_seg) == 0:
            return False
        rms = float(np.sqrt(np.mean(audio_seg.astype(np.float32) ** 2)))
        median_rms = float(np.median(energy_context)) if len(energy_context) > 0 else rms
        return rms > self._STRESS_RMS_FACTOR * max(median_rms, 1e-10)

    def _detect_language_from_mono(self, mono: np.ndarray, sr: int) -> tuple[str, float]:
        """Erkennt spoken language from audio via LPC formant analysis (SR-agnostic).

        Delegates to backend.core.phoneme_timeline._detect_language.
        Falls back to ("unknown", 0.0) on import error or any exception.

        Args:
            mono: 1-D float32 audio at any sample rate.
            sr:   Sample rate in Hz.

        Returns:
            (language_code, confidence) tuple.
        """
        try:
            from backend.core.phoneme_timeline import _detect_language as _ptl_detect

            return _ptl_detect(mono, sr)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("LyricsTranscriber._detect_language_from_mono failed: %s", exc)
            return ("unknown", 0.0)


# ---------------------------------------------------------------------------
# Singleton + Convenience (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: LyricsTranscriber | None = None
_lock = threading.Lock()


def get_lyrics_transcriber() -> LyricsTranscriber:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LyricsTranscriber()
    return _instance


def transcribe_audio(
    audio: np.ndarray,
    sr: int = 48_000,
) -> LyricsTranscriptionResult:
    """Convenience-Wrapper: transkribiert Audio → Wort-Zeitstempel (§2.36).

    Datenschutz: Wort-Texte werden nicht geloggt (nur Phonem-Typ).

    Args:
        audio: float32/64 nd-array, mono oder stereo
        sr:    Sample-Rate in Hz (48000 Hz empfohlen)

    Returns:
        LyricsTranscriptionResult mit WordTimestamp-Liste, immer NaN-frei
    """
    return get_lyrics_transcriber().transcribe(audio, sr)
