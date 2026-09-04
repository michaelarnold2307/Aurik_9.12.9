from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.audio_utils import safe_filtfilt  # §v10.101 padlen-guard

logger = logging.getLogger(__name__)
# Lazy imports of optional dependencies (scipy, PyQt5, onnxruntime, backend plugins) are
# intentional throughout this module to avoid hard dependencies at import time.
# pylint: disable=import-outside-toplevel


@dataclass
class WordTimestamp:
    """Timestamped word from lyrics transcription with phoneme classification."""

    word: str
    start_s: float
    end_s: float
    confidence: float
    is_stressed: bool
    phoneme_type: str


@dataclass
class LyricsTranscriptionResult:
    """Full transcription result including per-word timestamps and metadata."""

    words: list[WordTimestamp]
    language: str
    overall_confidence: float
    duration_s: float
    fallback_used: bool


def _assert_no_lyrics_in_log(words: list[WordTimestamp]) -> None:
    """§v10.303.50 Datenschutz-Guard: Lyrics-Text NACH Verarbeitung löschen.

    Stellt sicher, dass ``word.word`` vor Logging/Metadaten-Pfaden
    geleert ist. Im HF-Decoder-Pfad (§v10.303.50) wird der Text bewusst
    im RAM gehalten (für Sentiment/Semantic-DSP) und erst hier gelöscht.

    Privacy invariant: "Lyrics-Text NIEMALS in Logs oder Metadaten."
    """
    for w in words:
        if w.word:
            logger.debug(
                "§v10.303.50 Privacy: lösche %d Zeichen Lyrics-Text vor Logging.",
                len(w.word or ""),
            )
            object.__setattr__(w, "word", "")


class LyricsTranscriber:
    """Delegate transcription to the §2.36 core implementation.

    This keeps a small public transcriber object for callers that expect a
    ``.transcribe()`` API while routing all real work through
    ``LyricsGuidedEnhancement._transcribe_internal``.
    """

    def __init__(self) -> None:
        self._enhancement: LyricsGuidedEnhancement | None = None

    def bind_enhancement(self, enhancement: LyricsGuidedEnhancement) -> None:
        """Bind the LyricsGuidedEnhancement instance for transcription delegation."""
        self._enhancement = enhancement

    def transcribe(self, audio: np.ndarray, sr: int = 48_000) -> LyricsTranscriptionResult:
        """Transcribe audio via §2.36 core — routes to LyricsGuidedEnhancement."""
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        dur = float(mono.shape[0] / max(1, sr))
        enhancement = self._enhancement
        if enhancement is None:
            enhancement = get_lyrics_guided_enhancement()
        return enhancement._transcribe_internal(mono, sr, dur)  # type: ignore[attr-defined]  # pylint: disable=protected-access


class ContentAwareProcessor:
    """Weights phoneme-type segments for perceptual saliency in NR bypass decisions."""

    SALIENCY_BOOST: dict[str, float] = {
        "fricative_stressed": 1.55,  # §8.3 Tiefen-Immersion: fricative ×1.55
        "fricative_unstressed": 1.55,  # §8.3 Tiefen-Immersion: fricative ×1.55
        "vowel_stressed": 1.35,  # §8.3 Tiefen-Immersion: vowel_stressed ×1.35
        "vowel_unstressed": 1.0,
        "plosive": 1.40,  # §8.3 Tiefen-Immersion: plosive ×1.40
        "silence": 0.70,  # §8.3 Tiefen-Immersion: silence ×0.70
        "mixed": 1.0,
    }

    @staticmethod
    def _lpc_burg_coeffs(signal: np.ndarray, order: int) -> np.ndarray:
        """Schätzt LPC coefficients with Burg recursion (stable AR fit).

        Returns polynomial A(z) with A[0] = 1.0 and length order+1.
        Falls back to [1.0] for degenerate inputs.
        """
        x = np.asarray(signal, dtype=np.float64)
        n = int(x.size)
        if order < 1 or n <= (order + 2):
            return np.array([1.0], dtype=np.float64)  # type: ignore[no-any-return]

        ef = x[1:].copy()
        eb = x[:-1].copy()
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0

        for m in range(order):
            if ef.size < 2 or eb.size < 2:
                break

            den = float(np.dot(ef, ef) + np.dot(eb, eb) + 1e-12)
            k = float(-2.0 * np.dot(eb, ef) / den)
            k = float(np.clip(k, -0.995, 0.995))

            a_prev = a.copy()
            for i in range(1, m + 1):
                a[i] = a_prev[i] + k * a_prev[m + 1 - i]
            a[m + 1] = k

            ef_new = ef[1:] + k * eb[1:]
            eb_new = eb[:-1] + k * ef[:-1]
            ef, eb = ef_new, eb_new

        if not np.isfinite(a).all() or abs(a[0]) < 1e-12:
            return np.array([1.0], dtype=np.float64)  # type: ignore[no-any-return]
        return a  # type: ignore[no-any-return]

    def _apply_phoneme_dsp(
        self,
        segment: np.ndarray,
        phoneme_type: str,
        sr: int,
        strength: float,
    ) -> np.ndarray:
        """Per-phoneme spectral treatment per spec §2.36a.

        4 branches (all require PGHI after spectral modification):

        fricative_stressed / fricative_unstressed:
            Ramp-gain g(f) = 1 + strength × ramp(4 kHz → 8 kHz).
            NO Wiener smoothing in the 4–8 kHz band — preserves fricative texture.

        plosive:
            TransientShapeGuard: onset window (0–5 ms) gain = 1.0 (frozen).
            Burst enhancement  100–350 Hz × 1.40.
            Aspiration boost   3–8 kHz   × 1.20.

        vowel_stressed:
            LPC Burg Ord. 30–40 → F1–F4 peaks → symmetric ±2-semitone
            bandpass boost around each formant.

        silence:
            OMLSA-inspired Wiener gain with G_floor = 0.05 and
            energy_bias = −12 dB for aggressive but artefact-free NR.

        PGHI phase continuation is approximated by passing the STFT phase
        unchanged into ISTFT (scipy.signal.istft internally applies PGHI-like
        overlap-add consistency).

        Post-invariants (asserted upstream by MusicalGoalsChecker):
            TimbralAuthenticityMetric ≥ 0.87, ArticulationMetric ≥ 0.85.

        Args:
            segment:      1-D float32 mono audio segment.
            phoneme_type: One of 'fricative_stressed', 'fricative_unstressed',
                          'plosive', 'vowel_stressed', 'silence'; others unchanged.
            sr:           Sample rate (48 000 Hz).
            strength:     Processing intensity 0–1.

        Returns:
            Processed mono float32 array, same length as input.
        """
        from scipy import signal as _sig

        seg = np.asarray(segment, dtype=np.float64)
        n = len(seg)
        if n < 32 or strength < 1e-6:
            return segment.astype(np.float32)  # type: ignore[no-any-return]

        seg_out: np.ndarray

        if "fricative" in phoneme_type:
            # Ramp-gain 4–8 kHz; no Wiener smoothing in this band
            nperseg = min(512, n)
            noverlap = nperseg // 2
            _f, _t, Zxx = _sig.stft(seg, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            f4k = int(np.searchsorted(_f, 4_000.0))
            f8k = int(np.searchsorted(_f, 8_000.0))
            n_bins = f8k - f4k
            if n_bins > 0:
                ramp = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
                gain = 1.0 + strength * ramp
                Zxx[f4k:f8k, :] *= gain[:, np.newaxis]
            _, seg_out = _sig.istft(Zxx, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            seg_out = seg_out[:n]

        elif phoneme_type == "plosive":
            # TransientShapeGuard: freeze onset, boost burst + aspiration
            nperseg = min(256, n)
            noverlap = nperseg * 3 // 4
            hop = nperseg - noverlap
            _f, _t, Zxx = _sig.stft(seg, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            onset_frames = max(1, int(sr * 0.005 / max(1, hop)))  # 0–5 ms frames
            # Onset window: leave untouched (gain = 1.0 implicit — no modification)
            # Burst 100–350 Hz × 1.40 (post-onset only)
            f100 = int(np.searchsorted(_f, 100.0))
            f350 = int(np.searchsorted(_f, 350.0))
            if f350 > f100:
                Zxx[f100:f350, onset_frames:] *= 1.0 + strength * 0.40
            # Aspiration 3–8 kHz × 1.20 (post-onset only)
            f3k = int(np.searchsorted(_f, 3_000.0))
            f8k = int(np.searchsorted(_f, 8_000.0))
            if f8k > f3k:
                Zxx[f3k:f8k, onset_frames:] *= 1.0 + strength * 0.20
            _, seg_out = _sig.istft(Zxx, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            seg_out = seg_out[:n]

        elif phoneme_type == "vowel_stressed":
            # LPC Burg Ord. 30–40 → F1–F4 → symmetric shelving ±2 semitones
            try:
                from scipy.signal import filtfilt as _lge_filtfilt
                from scipy.signal import lfilter

                order = min(36, n // 4)  # Ord 30–40 preferred; cap for short segments
                A = self._lpc_burg_coeffs(seg - np.mean(seg), order)
                # Guard: degenerate LPC polynomial triggers LAPACK DLASCL via np.roots companion matrix
                if A.size < 2 or not np.isfinite(A).all():
                    raise ValueError("Degenerate LPC polynomial — passthrough")
                roots = np.roots(A)
                roots = roots[(np.abs(roots) < 1.0) & (np.imag(roots) > 0)]
                formant_freqs: list[float] = sorted(
                    [
                        float(np.angle(r) * sr / (2.0 * np.pi))
                        for r in roots
                        if 80.0 < float(np.angle(r) * sr / (2.0 * np.pi)) < 8_000.0
                    ]
                )[:4]  # F1–F4 only
                seg_out = seg.copy()
                for ff in formant_freqs:
                    fl = ff * (2.0 ** (-2.0 / 12.0))
                    fh = ff * (2.0 ** (2.0 / 12.0))
                    nyq = sr / 2.0
                    if fh >= nyq or fl <= 0 or (fh - fl) < 5.0:
                        continue
                    _butter_ba = _sig.butter(2, [fl / nyq, min(0.999, fh / nyq)], btype="band", output="ba")
                    b, a_filt = _butter_ba[0], _butter_ba[1]  # type: ignore[index]
                    # §2.51: filtfilt (zero-phase) statt lfilter — Gruppenversatz IIR-BP
                    # erzeugte Comb-Filter wenn Band zum Original addiert wird
                    _lge_n = len(seg_out)
                    seg_band = safe_filtfilt(b, a_filt, seg_out) if _lge_n >= 15 else lfilter(b, a_filt, seg_out)
                    seg_out = seg_out + strength * 0.30 * seg_band  # additive formant lift
            except Exception:
                seg_out = seg.copy()

        elif phoneme_type == "silence":
            # OMLSA-inspired aggressive NR: G_floor=0.05, energy_bias=−12 dB
            try:
                from scipy.ndimage import minimum_filter1d

                nperseg = min(2048, n)
                noverlap = nperseg * 3 // 4
                _f, _t, Zxx = _sig.stft(seg, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
                magnitude = np.abs(Zxx)
                phase = np.angle(Zxx)
                # Sliding-minimum noise floor estimate (IMCRA-inspired, M=15 frames)
                noise_floor = minimum_filter1d(magnitude * 1.66, size=15, axis=1, mode="nearest")
                G_floor = 0.05
                snr = magnitude / (noise_floor + 1e-10)
                G = np.clip(1.0 - 1.0 / (snr + 1e-10), G_floor, 1.0)
                # energy_bias = −12 dB per spec §2.36a
                energy_bias = 10.0 ** (-12.0 / 20.0)
                G_eff = G * energy_bias * strength + (1.0 - strength)
                G_eff = np.clip(G_eff, G_floor, 1.0)
                Zxx_out = magnitude * G_eff * np.exp(1j * phase)
                _, seg_out = _sig.istft(Zxx_out, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
                seg_out = seg_out[:n]
            except Exception:
                seg_out = seg.copy()
        else:
            seg_out = seg.copy()

        # Pad/trim to original length; clip and NaN-guard
        seg_out = np.asarray(seg_out, dtype=np.float64)
        if len(seg_out) < n:
            seg_out = np.pad(seg_out, (0, n - len(seg_out)))
        else:
            seg_out = seg_out[:n]
        return np.clip(np.nan_to_num(seg_out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def apply_phoneme_dsp_to_audio(
        self,
        audio: np.ndarray,
        transcription: LyricsTranscriptionResult,
        sr: int = 48_000,
        strength: float = 0.50,
        language_profile: dict[str, float] | None = None,  # §v10.303.51
    ) -> np.ndarray:
        """Wendet _apply_phoneme_dsp() auf jedes Transkriptionssegment in-place an.

        Iterates over transcription.words (phoneme_type only — no word text
        is accessed, satisfying §2.36 privacy invariant) and writes the
        spectrally processed segment back into a copy of audio.

        Args:
            audio:         float32 ndarray, mono (N,) or stereo (N, 2).
            transcription: LyricsTranscriptionResult from LyricsGuidedEnhancement.
            sr:            Sample rate (48 000 Hz).
            strength:      Per-phoneme DSP intensity 0–1.

        Returns:
            float32 array, same shape as audio, values ∈ [−1, 1].
        """
        if transcription.fallback_used or not transcription.words:
            return audio

        # §v10.303.50: Text NUR im RAM; erst NACH _assert_no_lyrics_in_log löschen.
        # Hier noch nicht prüfen — Sentiment/Semantic-DSP brauchen den Text.

        out = np.asarray(audio, dtype=np.float32).copy()
        n_samples = out.shape[0]
        is_stereo = out.ndim == 2

        # §v10.303.51: Language-specific phoneme gain offsets
        _lp = language_profile or {}
        _fric_gain = 1.0 + float(_lp.get("fricative_gain_db", 0.0)) / 20.0
        _plos_gain = 1.0 + float(_lp.get("plosive_burst_db", 0.0)) / 20.0
        _vowel_gain = 1.0 + float(_lp.get("vowel_formant_db", 0.0)) / 20.0
        _sil_nr = float(_lp.get("silence_nr_strength", 1.0))

        for word in transcription.words:
            i0 = max(0, min(n_samples, int(word.start_s * sr)))
            i1 = max(i0, min(n_samples, int(word.end_s * sr)))
            if i1 - i0 < 32:
                continue
            # §v10.303.51: Per-phoneme language-specific strength scaling
            _phoneme_str = strength
            if "fricative" in word.phoneme_type:
                _phoneme_str *= _fric_gain
            elif "plosive" in word.phoneme_type:
                _phoneme_str *= _plos_gain
            elif "vowel" in word.phoneme_type:
                _phoneme_str *= _vowel_gain
            _phoneme_str = float(np.clip(_phoneme_str, 0.1, 1.5))
            if is_stereo:
                for ch in range(out.shape[1]):
                    seg_out = self._apply_phoneme_dsp(out[i0:i1, ch], word.phoneme_type, sr, _phoneme_str)
                    seg_len = min(len(seg_out), i1 - i0)
                    out[i0 : i0 + seg_len, ch] = seg_out[:seg_len]
            else:
                seg_out = self._apply_phoneme_dsp(out[i0:i1], word.phoneme_type, sr, _phoneme_str)
                seg_len = min(len(seg_out), i1 - i0)
                out[i0 : i0 + seg_len] = seg_out[:seg_len]

        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    def compute_lyrics_saliency(
        self,
        base_saliency: np.ndarray,
        transcription: LyricsTranscriptionResult,
        sr: int = 48_000,
    ) -> np.ndarray:
        """Wendet an: phoneme-class SALIENCY_BOOST to a pre-computed base saliency map.

        Each WordTimestamp region in ``transcription`` overwrites the corresponding
        sample range in ``base_saliency`` with the class-specific boost factor.
        Result is clipped to [0.3, 2.0] per §2.36 spec.

        Args:
            base_saliency: 1-D float32 array of length n_samples (starting at 1.0).
            transcription: result from LyricsGuidedEnhancement._transcribe_internal.
            sr:            sample rate of the audio that base_saliency refers to.

        Returns:
            float32 array of shape (n_samples,), values ∈ [0.3, 2.0].
        """
        sal = np.nan_to_num(np.asarray(base_saliency, dtype=np.float32))
        if sal.ndim != 1:
            return np.clip(sal, 0.3, 2.0)  # type: ignore[no-any-return]
        if transcription.fallback_used or not transcription.words:
            return np.clip(sal, 0.3, 2.0)  # type: ignore[no-any-return]
        n = len(sal)
        for word in transcription.words:
            boost = self.SALIENCY_BOOST.get(word.phoneme_type, 1.0)
            i0 = max(0, min(n, int(word.start_s * sr)))
            i1 = max(i0, min(n, int(word.end_s * sr)))
            if i1 > i0:
                sal[i0:i1] = boost
        return np.clip(sal, 0.3, 2.0)  # type: ignore[no-any-return]


class LyricsGuidedTimeline:
    """Visualisiert Wort-Zeitstempel im WaveformWidget (§2.36).

    Zeichnet farbige Phonem-Overlays auf den Waveform-Canvas.
    Datenschutz: Kein Lyrics-Text in Tooltips oder Logs — nur Phonem-Typ.
    """

    COLOR_MAP: dict[str, str] = {
        "vowel_stressed": "#4CAF50",
        "fricative_stressed": "#FF9800",
        "fricative_unstressed": "#FFD54F",
        "plosive": "#29B6F6",
        "silence": "#B0BEC5",
    }
    SHORTCUT: str = "L"

    def render_overlay(
        self,
        painter: Any,
        transcription: LyricsTranscriptionResult,
        widget_width_px: int,
        audio_duration_s: float,
    ) -> None:
        """Renders phoneme-type color bands on the WaveformWidget canvas.

        Each phoneme category has a unique visual treatment:
        - fricative_stressed:   orange band with top triangle marker (sibilant energy indicator)
        - fricative_unstressed: amber band (softer sibilants)
        - plosive:              cyan band with diamond marker (onset indicator at start)
        - vowel_stressed:       green band with filled bar (formant prominence)
        - silence:              very subtle dark band (NR zone indicator)

        Privacy: no transcription text rendered — phoneme_type labels only.
        """
        if transcription.fallback_used or not transcription.words:
            return
        if audio_duration_s <= 0 or widget_width_px <= 0:
            return
        try:
            from PyQt5.QtCore import QPointF, QRectF, Qt
            from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF

            OVERLAY_H = 44  # px height of overlay band at top of waveform
            BAR_H = {
                "vowel_stressed": OVERLAY_H,
                "fricative_stressed": 30,
                "fricative_unstressed": 22,
                "plosive": OVERLAY_H,
                "silence": 10,
            }
            ALPHA = {
                "vowel_stressed": 75,
                "fricative_stressed": 95,
                "fricative_unstressed": 60,
                "plosive": 85,
                "silence": 22,
            }

            if not hasattr(painter, "fillRect"):
                return

            for word in transcription.words:
                ptype = word.phoneme_type
                color_hex = self.COLOR_MAP.get(ptype, "")
                if not color_hex:
                    continue

                x_start = int(word.start_s / audio_duration_s * widget_width_px)
                x_end = max(x_start + 2, int(word.end_s / audio_duration_s * widget_width_px))
                width = x_end - x_start
                bh = BAR_H.get(ptype, 20)
                alpha = ALPHA.get(ptype, 60)

                c = QColor(color_hex)
                c.setAlpha(alpha)
                painter.fillRect(QRectF(x_start, 0, width, bh), c)  # type: ignore[arg-type]

                # Phoneme type accent decorations
                if ptype == "plosive":
                    # Diamond onset marker at segment start
                    dia_x = x_start + 1
                    dia_y = bh // 2
                    diamond = QPolygonF()
                    diamond.append(QPointF(dia_x, dia_y - 5))
                    diamond.append(QPointF(dia_x + 4, dia_y))
                    diamond.append(QPointF(dia_x, dia_y + 5))
                    diamond.append(QPointF(dia_x - 4, dia_y))
                    accent = QColor(color_hex)
                    accent.setAlpha(200)
                    painter.setBrush(QBrush(accent))  # type: ignore[attr-defined]
                    painter.setPen(Qt.PenStyle.NoPen)  # type: ignore[attr-defined]
                    painter.drawPolygon(diamond)  # type: ignore[attr-defined]

                elif ptype == "fricative_stressed":
                    # Small triangles at top — sibilance energy markers
                    acc2 = QColor(color_hex)
                    acc2.setAlpha(200)
                    n_marks = max(1, width // 8)
                    for mi in range(n_marks):
                        tx = x_start + int((mi + 0.5) * width / n_marks)
                        tri = QPolygonF()
                        tri.append(QPointF(tx, 0))
                        tri.append(QPointF(tx - 3, 6))
                        tri.append(QPointF(tx + 3, 6))
                        painter.setBrush(QBrush(acc2))  # type: ignore[attr-defined]
                        painter.setPen(Qt.PenStyle.NoPen)  # type: ignore[attr-defined]
                        painter.drawPolygon(tri)  # type: ignore[attr-defined]

                elif ptype == "vowel_stressed":
                    # Horizontal line at 60% height — formant prominence indicator
                    acc3 = QColor(color_hex)
                    acc3.setAlpha(160)
                    painter.setPen(QPen(acc3, 1.5))  # type: ignore[attr-defined]
                    _line_y = int(bh * 0.6)
                    painter.drawLine(x_start, _line_y, x_end, _line_y)  # type: ignore[attr-defined]
                    painter.setPen(Qt.PenStyle.NoPen)  # type: ignore[attr-defined]

        except (ImportError, Exception):
            pass  # No Qt or rendering error — silent fallback


_transcriber: LyricsTranscriber | None = None
_transcriber_lock = threading.Lock()
_processor: ContentAwareProcessor | None = None
_processor_lock = threading.Lock()
_timeline: LyricsGuidedTimeline | None = None
_timeline_lock = threading.Lock()
_lge_instance: LyricsGuidedEnhancement | None = None
_lge_lock = threading.Lock()


def is_lyrics_guided_loaded() -> bool:
    """Gibt True only if the LGE singleton is already initialised (models loaded) zurück.

    Use this guard before calling get_lyrics_guided_enhancement() in
    latency-sensitive paths (e.g. pre-analysis, genre classification during
    file opening) to avoid loading Whisper + wav2vec2 unexpectedly.
    """
    return _lge_instance is not None


def get_lyrics_transcriber() -> LyricsTranscriber:
    """Thread-safe singleton accessor for LyricsTranscriber."""
    global _transcriber  # pylint: disable=global-statement
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                _transcriber = LyricsTranscriber()
    return _transcriber


def get_content_aware_processor() -> ContentAwareProcessor:
    """Thread-safe singleton accessor for ContentAwareProcessor."""
    global _processor  # pylint: disable=global-statement
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                _processor = ContentAwareProcessor()
    return _processor


def get_lyrics_guided_timeline() -> LyricsGuidedTimeline:
    """Thread-safe singleton accessor for LyricsGuidedTimeline."""
    global _timeline  # pylint: disable=global-statement
    if _timeline is None:
        with _timeline_lock:
            if _timeline is None:
                _timeline = LyricsGuidedTimeline()
    return _timeline


class LyricsGuidedEnhancement:
    """§2.36 LyricsGuidedEnhancement — mandatory from Aurik 10.0.0.x.

    Pipeline:
      1. Resample to 16 kHz → 80-channel log-mel spectrogram
      2. Whisper-Tiny ONNX encoder (models/whisper/whisper_tiny.onnx,
         CPUExecutionProvider, no network access)
      3. Frame-level RMS of encoder hidden states → vocal-activity segments
      4. ContentAwareProcessor.SALIENCY_BOOST mapping → sample-level gain curve
      5. audio_out = clip(nan_to_num(audio * saliency), -1, 1)

    Fallback (ONNX unavailable):
      DSP energy segmentation: 20 ms frames, RMS, 60th-percentile threshold.

    Privacy invariant: lyrics text is NEVER written to any log, variable, or
    RestorationResult.metadata.  Only phoneme-type labels are used internally.

    Singleton access: get_lyrics_guided_enhancement().
    """

    # Whisper-Tiny constants (Radford et al. 2022)
    _ONNX_SR: int = 16_000
    _N_MELS: int = 80
    _N_FFT: int = 400
    _HOP: int = 160
    _MAX_FRAMES: int = 3_000  # 30 s at 100 frames/s → 1500 encoder output frames

    # wav2vec2 forced-alignment ONNX (125 MB, CPUExecutionProvider) — §2.36 Pflicht
    _WAV2VEC2_SR: int = 16_000  # wav2vec2 operates at 16 kHz

    def __init__(self) -> None:
        self._cap = ContentAwareProcessor()
        self._tl = LyricsGuidedTimeline()
        self._ort_session: Any = None  # Whisper ONNX InferenceSession
        self._aligner_session: Any = None  # wav2vec2 forced-alignment ONNX session
        self._whisper_hf_processor: Any = None  # §v10.303.50 HF WhisperProcessor
        self._whisper_hf_model: Any = None  # §v10.303.50 HF WhisperForConditionalGeneration
        self._try_load_hf_whisper()  # §v10.303.50: HF decoder path (preferred)
        if self._whisper_hf_model is None:
            self._try_load_onnx()  # Fallback: ONNX encoder-only
        self._try_load_aligner()
        self._transcriber = get_lyrics_transcriber()
        self._transcriber.bind_enhancement(self)

    def is_loaded(self) -> bool:
        """Return True if at least one model backend is ready."""
        return self._whisper_hf_model is not None or self._ort_session is not None or self._aligner_session is not None

    # ── ONNX bootstrap ─────────────────────────────────────────────────────

    def _try_load_onnx(self) -> None:
        """Lädt whisper_tiny.onnx with CPUExecutionProvider (no GPU, no network)."""
        # [RELEASE_MUST] memory budget guard before InferenceSession (§2.37 Checkliste)
        _release_on_fail: Callable[[], None] | None = None
        try:
            from backend.core.ml_memory_budget import (
                release as _ml_release,
            )
            from backend.core.ml_memory_budget import (
                try_allocate as _try_alloc,
            )

            if not _try_alloc("lyrics_transcriber_whisper", size_gb=0.04):
                logger.info(
                    "LyricsGuidedEnhancement: ML-Grenze erschöpft (Whisper) — DSP-Ersatzpfad aktiv.",
                )
                return
            _release_on_fail = partial(_ml_release, "lyrics_transcriber_whisper")
        except ImportError:
            pass  # budget module absent → attempt load anyway
        _loaded = False
        try:
            import onnxruntime as ort  # type: ignore[import]

            model_path = Path(__file__).resolve().parents[2] / "models" / "whisper" / "whisper_tiny.onnx"
            if model_path.exists():
                self._ort_session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                logger.info(
                    "LyricsGuidedEnhancement: whisper_tiny.onnx geladen (%.1f MB)",
                    model_path.stat().st_size / 1e6,
                )
                _loaded = True
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        "lyrics_transcriber_whisper",
                        size_gb=0.04,
                        unload_fn=lambda: setattr(self, "_ort_session", None),
                    )
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            else:
                logger.debug(
                    "LyricsGuidedEnhancement: whisper_tiny.onnx not found at %s — DSP Ersatzpfad active",
                    model_path,
                )
        except Exception as exc:
            logger.debug(
                "LyricsGuidedEnhancement: ONNX laden fehlgeschlagen (%s) — DSP Ersatzpfad active",
                exc,
            )
        finally:
            if not _loaded and _release_on_fail is not None:
                try:
                    _release_on_fail()  # type: ignore[operator, call-arg]
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    def _try_load_aligner(self) -> None:
        """Lädt wav2vec2_forced_alignment.onnx (§2.36 PFLICHT: Phonem-Alignment).

        Model: 125 MB, CPUExecutionProvider, no network access.
        Fallback: DSP energy-threshold segmentation + Whisper token ID phoneme prior.  # §V6 (copilot-instructions.md): logger.warning handled at call site
        """
        # [RELEASE_MUST] memory budget guard before InferenceSession (§2.37 Checkliste)
        _release_on_fail: Callable[[], None] | None = None
        try:
            from backend.core.ml_memory_budget import (
                release as _ml_release,
            )
            from backend.core.ml_memory_budget import (
                try_allocate as _try_alloc,
            )

            if not _try_alloc("lyrics_aligner_wav2vec2", size_gb=0.13):
                logger.info(
                    "LyricsGuidedEnhancement: ML-Grenze erschöpft (wav2vec2 Aligner) — DSP-Ersatzpfad aktiv.",
                )
                return
            _release_on_fail = partial(_ml_release, "lyrics_aligner_wav2vec2")
        except ImportError:
            pass  # budget module absent → attempt load anyway
        _loaded = False
        try:
            import onnxruntime as ort  # type: ignore[import]

            model_path = Path(__file__).resolve().parents[2] / "models" / "wav2vec2" / "wav2vec2_forced_alignment.onnx"
            if model_path.exists():
                self._aligner_session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                logger.info(
                    "LyricsGuidedEnhancement: wav2vec2_forced_alignment.onnx geladen (%.1f MB)",
                    model_path.stat().st_size / 1e6,
                )
                _loaded = True
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        "lyrics_aligner_wav2vec2",
                        size_gb=0.13,
                        unload_fn=lambda: setattr(self, "_aligner_session", None),
                    )
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            else:
                logger.debug(
                    "LyricsGuidedEnhancement: wav2vec2_forced_alignment.onnx not found at %s"
                    " — DSP phoneme-prior Ersatzpfad active",
                    model_path,
                )
        except Exception as exc:
            logger.debug(
                "LyricsGuidedEnhancement: aligner ONNX laden fehlgeschlagen (%s) — DSP phoneme-prior Ersatzpfad active",
                exc,
            )
        finally:
            if not _loaded and _release_on_fail is not None:
                try:
                    _release_on_fail()  # type: ignore[operator, call-arg]
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    # ── §v10.303.50 HF Whisper Decoder bootstrap ────────────────────────────

    def _try_load_hf_whisper(self) -> None:
        """Lädt HuggingFace WhisperForConditionalGeneration (ENCODER + DECODER).

        Modellpfad: models/whisper/ (lokales HF-Format, kein Netzwerk).
        Dies ist der PREFERRED Pfad — der Decoder liefert echte Wort-Transkripte
        für NLP-Sentiment-Analyse und semantik-gesteuertes DSP.

        Fallback: ONNX whisper_tiny.onnx (encoder-only) + DSP-Energie-Segmentierung.  # §V6 (copilot-instructions.md): logger.warning handled at call site

        Privacy: Das HF-Modell läuft vollständig lokal. Kein Text verlässt den RAM.
        """
        _release_on_fail: Callable[[], None] | None = None
        try:
            from backend.core.ml_memory_budget import (
                release as _ml_release,
            )
            from backend.core.ml_memory_budget import (
                try_allocate as _try_alloc,
            )

            if not _try_alloc("lyrics_whisper_hf", size_gb=0.25):
                logger.info(
                    "LyricsGuidedEnhancement: ML-Grenze erschöpft (HF Whisper) — ONNX-Encoder/DSP-Ersatzpfad aktiv.",
                )
                return
            _release_on_fail = partial(_ml_release, "lyrics_whisper_hf")
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        _loaded = False
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            model_path = Path(__file__).resolve().parents[2] / "models" / "whisper"
            if not (model_path / "config.json").exists():
                logger.debug(
                    "LyricsGuidedEnhancement: HF Whisper model not found at %s — ONNX/DSP Ersatzpfad active",
                    model_path,
                )
                return

            self._whisper_hf_processor = WhisperProcessor.from_pretrained(  # nosec B615 — local_files_only=True, kein Download
                str(model_path), local_files_only=True
            )
            self._whisper_hf_model = WhisperForConditionalGeneration.from_pretrained(  # nosec B615 — local_files_only=True, kein Download
                str(model_path), local_files_only=True
            )
            self._whisper_hf_model.eval()
            logger.info(
                "§v10.303.50 LyricsGuidedEnhancement: HF Whisper Decoder geladen "
                "(models/whisper/) — echte Wort-Transkription aktiv",
            )
            _loaded = True
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                def _unload_hf() -> None:
                    self._whisper_hf_processor = None
                    self._whisper_hf_model = None
                    try:
                        _ml_release("lyrics_whisper_hf")
                    except Exception:
                        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

                _reg_plm(
                    "lyrics_whisper_hf",
                    size_gb=0.25,
                    unload_fn=_unload_hf,
                )
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        except Exception as exc:
            logger.debug(
                "§v10.303.50 HF Whisper laden fehlgeschlagen (%s) — ONNX/DSP Ersatzpfad active",
                exc,
            )
        finally:
            if not _loaded and _release_on_fail is not None:
                try:
                    _release_on_fail()
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    # Minimum samples required by wav2vec2 feature extractor (7 Conv1d layers,
    # cumulative receptive field: kernels [10,3,3,3,3,2,2], strides [5,2,2,2,2,2,2]
    # → min input ≥ 400 samples at 16 kHz = 25 ms).  Inputs shorter than this
    # cause an OrtInvalidArgument / "Invalid input shape: {N}" error.  Any segment
    # shorter than 25 ms is treated as silence for phoneme classification.
    _MIN_WAV2VEC2_SAMPLES: int = 400

    def _align_phonemes(
        self,
        words: list[WordTimestamp],
        mono_16k: np.ndarray,
        sr_16k: int = 16_000,
    ) -> list[WordTimestamp]:
        """Refine phoneme types using wav2vec2 forced alignment (§2.36 PFLICHT).

        If ``self._aligner_session`` is available, runs wav2vec2 to obtain
        frame-level CTC emission probabilities, then recomputes the phoneme_type
        for each word segment using the dominant phoneme class from the CTC output.

        Fallback (aligner unavailable): returns ``words`` unchanged — the
        ``_classify_phoneme_type`` DSP assignment from the transcription step is used.

        Fallback (audio too short): any call where ``len(mono_16k) < _MIN_WAV2VEC2_SAMPLES``
        (< 25 ms at 16 kHz) returns ``words`` unchanged — the wav2vec2 Conv1d
        feature extractor requires at least 400 samples to produce a valid output
        frame.  Shorter inputs cause OrtInvalidArgument "Invalid input shape: {N}".

        Privacy: no text content is forwarded to the model; only raw waveform
        frames corresponding to each word's time span are processed.

        Args:
            words:    List of WordTimestamp segments from Whisper/DSP transcription.
            mono_16k: Mono audio at 16 kHz (float32).
            sr_16k:   Sample rate of mono_16k (must be 16 000 Hz).

        Returns:
            Updated list of WordTimestamp with refined phoneme_type labels.
        """
        if self._aligner_session is None or not words:
            return words  # fallback: keep DSP classification  # §V6 (copilot-instructions.md): logger.warning handled at call site

        try:
            # Wav2vec2 expects float32 input of shape [1, T]
            audio_input = mono_16k.astype(np.float32)
            if audio_input.ndim != 1:
                return words

            # §2.36 Mindestlängen-Guard: wav2vec2 Conv1d requires ≥ 400 samples
            # (25 ms @ 16 kHz).  Shorter inputs → OrtInvalidArgument "Invalid
            # input shape: {N}".  Return DSP classification unchanged.
            if len(audio_input) < self._MIN_WAV2VEC2_SAMPLES:
                logger.debug(
                    "LyricsGuidedEnhancement._align_phonemes: Eingabe too short"
                    " (%d samples < %d min) — DSP silence Ersatzpfad",
                    len(audio_input),
                    self._MIN_WAV2VEC2_SAMPLES,
                )
                return words

            # Normalise to [-1, 1]
            amax = float(np.abs(audio_input).max()) or 1.0
            audio_input = audio_input / amax

            # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during wav2vec2 ONNX inference
            _plm_w2v: Any = None
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_w2v

                _plm_w2v = _get_plm_w2v()
                _plm_w2v.set_active("lyrics_aligner_wav2vec2", True)
            except Exception as e:
                logger.warning("lyrics_guided_enhancement.py::_align_phonemes Ersatzpfad: %s", e)
            # OOM-Guard: chunk wav2vec2 into 30 s segments to prevent 34+ GB
            # intermediate allocation on long files (§2.36 — root cause of OOM).
            _MAX_W2V_CHUNK = 30 * sr_16k  # 480 000 samples @ 16 kHz
            try:
                if len(audio_input) <= _MAX_W2V_CHUNK:
                    # Short file — single pass
                    outputs = self._aligner_session.run(  # type: ignore[attr-defined]
                        None, {"input_values": audio_input[np.newaxis, :]}
                    )
                    logits = outputs[0]  # (1, T_frames, vocab_size)
                else:
                    # Chunked inference for long files
                    _logit_parts: list[np.ndarray] = []
                    for _cstart in range(0, len(audio_input), _MAX_W2V_CHUNK):
                        _cchunk = audio_input[_cstart : _cstart + _MAX_W2V_CHUNK]
                        if len(_cchunk) < self._MIN_WAV2VEC2_SAMPLES:
                            break
                        _cout = self._aligner_session.run(  # type: ignore[attr-defined]
                            None, {"input_values": _cchunk[np.newaxis, :]}
                        )
                        _logit_parts.append(_cout[0][0])  # (T_chunk, vocab)
                    if not _logit_parts:
                        return words
                    logits = np.concatenate(_logit_parts, axis=0)[np.newaxis, :]  # (1, T_total, vocab)
            finally:
                if _plm_w2v is not None:
                    try:
                        _plm_w2v.set_active("lyrics_aligner_wav2vec2", False)  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning("lyrics_guided_enhancement.py::unbekannter Ersatzpfad: %s", e)

            # Run encoder: output is (1, T_frames, vocab_size) CTC log-probs
            if logits.ndim != 3:
                return words
            logits = logits[0]  # (T_frames, vocab_size)
            n_frames, _vocab_size = logits.shape

            # Frame duration at 16 kHz (wav2vec2 conv-fe downsamples by 320×)
            frames_per_sec = sr_16k / 320.0  # ≈ 50 frames/s

            # Phoneme-class mapping based on CTC token clusters:
            # Vowel tokens: roughly index range 4–20 (language-dependent approximation)
            # Fricative tokens: 21–35 | Plosive tokens: 36–50 | Silence: 0–3
            # Note: These are heuristic boundaries valid for multilingual wav2vec2.
            VOWEL_RANGE = (4, 20)
            FRICATIVE_RANGE = (21, 35)
            PLOSIVE_RANGE = (36, 50)

            updated: list = []
            for word in words:
                # §2.36: Use correct WordTimestamp fields (start_s / end_s, not start_time / end_time)
                frame_start = max(0, int(word.start_s * frames_per_sec))
                frame_end = min(n_frames, int(word.end_s * frames_per_sec) + 1)
                if frame_start >= frame_end:
                    updated.append(word)
                    continue

                seg_logits = logits[frame_start:frame_end]  # (T_seg, vocab)
                # Mean probability per token class (softmax approximation)
                probs = np.exp(seg_logits - np.max(seg_logits, axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True) + 1e-9
                mean_probs = probs.mean(axis=0)  # (vocab_size,)

                vowel_p = float(mean_probs[VOWEL_RANGE[0] : VOWEL_RANGE[1]].sum())
                fric_p = float(mean_probs[FRICATIVE_RANGE[0] : FRICATIVE_RANGE[1]].sum())
                plos_p = float(mean_probs[PLOSIVE_RANGE[0] : PLOSIVE_RANGE[1]].sum())

                # Determine dominant class
                is_stressed = word.phoneme_type.endswith("_stressed")
                if plos_p >= max(vowel_p, fric_p) * 0.8:
                    new_type = "plosive"
                elif fric_p >= vowel_p * 0.7:
                    new_type = "fricative_stressed" if is_stressed else "fricative_unstressed"
                else:
                    new_type = "vowel_stressed" if is_stressed else "vowel_unstressed"

                updated.append(
                    WordTimestamp(
                        word=getattr(word, "word", "") or "",  # §v10.303.50: preserve word if present (HF decoder)
                        start_s=word.start_s,
                        end_s=word.end_s,
                        confidence=float(max(vowel_p, fric_p, plos_p)),
                        is_stressed=is_stressed,
                        phoneme_type=new_type,
                    )
                )
            return updated
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("LyricsGuidedEnhancement._align_phonemes fehlgeschlagen (%s) — DSP Ersatzpfad", exc)
            return words

    # ── §v10.303.52 Semantic keyword-guided DSP ─────────────────────────────

    _SEMANTIC_DSP_PARAMS: dict[str, dict[str, float]] = {
        # EQ-Skulptierung pro semantischem Konzept
        # Jeder Eintrag: {low_db, mid_db, high_db, dynamics_scale, space_scale, groove_ms}
        # low_db:     Sub-Bass/Bass-Korrektur (60–250 Hz)
        # mid_db:     Präsenz (1–4 kHz)
        # high_db:    Brillanz/Luft (8–16 kHz)
        # dynamics_scale: 1.0=neutral, <1.0=weichere Kompression, >1.0=stärker
        # space_scale:    1.0=neutral, >1.0=mehr Raum/Akustik-Erhalt
        # groove_ms:     Timing-Verschiebung in ms (negativ=fester, positiv=lockerer)
        "liebe": {"low_db": 1.0, "mid_db": 0.5, "high_db": 0.0, "dyn": 0.75, "space": 1.15, "groove": 2.0},
        "love": {"low_db": 1.0, "mid_db": 0.5, "high_db": 0.0, "dyn": 0.75, "space": 1.15, "groove": 2.0},
        "herz": {"low_db": 1.5, "mid_db": 0.0, "high_db": -0.5, "dyn": 0.70, "space": 1.10, "groove": 3.0},
        "heart": {"low_db": 1.5, "mid_db": 0.0, "high_db": -0.5, "dyn": 0.70, "space": 1.10, "groove": 3.0},
        "tränen": {"low_db": 0.0, "mid_db": -1.0, "high_db": -1.0, "dyn": 0.60, "space": 1.25, "groove": 5.0},
        "tears": {"low_db": 0.0, "mid_db": -1.0, "high_db": -1.0, "dyn": 0.60, "space": 1.25, "groove": 5.0},
        "schmerz": {"low_db": -1.0, "mid_db": -1.5, "high_db": -1.5, "dyn": 0.55, "space": 1.30, "groove": 7.0},
        "pain": {"low_db": -1.0, "mid_db": -1.5, "high_db": -1.5, "dyn": 0.55, "space": 1.30, "groove": 7.0},
        "tanz": {"low_db": 2.0, "mid_db": 1.0, "high_db": 0.5, "dyn": 1.15, "space": 0.90, "groove": -3.0},
        "dance": {"low_db": 2.0, "mid_db": 1.0, "high_db": 0.5, "dyn": 1.15, "space": 0.90, "groove": -3.0},
        "freude": {"low_db": 1.0, "mid_db": 1.5, "high_db": 1.5, "dyn": 1.10, "space": 1.00, "groove": -2.0},
        "joy": {"low_db": 1.0, "mid_db": 1.5, "high_db": 1.5, "dyn": 1.10, "space": 1.00, "groove": -2.0},
        "glück": {"low_db": 1.0, "mid_db": 1.5, "high_db": 1.5, "dyn": 1.10, "space": 1.00, "groove": -2.0},
        "happy": {"low_db": 1.0, "mid_db": 1.5, "high_db": 1.5, "dyn": 1.10, "space": 1.00, "groove": -2.0},
        "traum": {"low_db": -0.5, "mid_db": 0.5, "high_db": 2.0, "dyn": 0.85, "space": 1.20, "groove": 1.0},
        "dream": {"low_db": -0.5, "mid_db": 0.5, "high_db": 2.0, "dyn": 0.85, "space": 1.20, "groove": 1.0},
        "hoffen": {"low_db": 0.5, "mid_db": 1.0, "high_db": 1.0, "dyn": 0.90, "space": 1.10, "groove": 0.0},
        "hope": {"low_db": 0.5, "mid_db": 1.0, "high_db": 1.0, "dyn": 0.90, "space": 1.10, "groove": 0.0},
        "feuer": {"low_db": 1.0, "mid_db": 2.0, "high_db": 2.0, "dyn": 1.20, "space": 0.85, "groove": -2.0},
        "fire": {"low_db": 1.0, "mid_db": 2.0, "high_db": 2.0, "dyn": 1.20, "space": 0.85, "groove": -2.0},
        "stark": {"low_db": 2.0, "mid_db": 1.5, "high_db": 1.0, "dyn": 1.15, "space": 0.90, "groove": -3.0},
        "strong": {"low_db": 2.0, "mid_db": 1.5, "high_db": 1.0, "dyn": 1.15, "space": 0.90, "groove": -3.0},
        "einsam": {"low_db": -2.0, "mid_db": -1.0, "high_db": -1.0, "dyn": 0.50, "space": 1.30, "groove": 8.0},
        "lonely": {"low_db": -2.0, "mid_db": -1.0, "high_db": -1.0, "dyn": 0.50, "space": 1.30, "groove": 8.0},
        "nacht": {"low_db": -1.0, "mid_db": -0.5, "high_db": -0.5, "dyn": 0.65, "space": 1.25, "groove": 4.0},
        "night": {"low_db": -1.0, "mid_db": -0.5, "high_db": -0.5, "dyn": 0.65, "space": 1.25, "groove": 4.0},
        "wind": {"low_db": -0.5, "mid_db": 0.0, "high_db": 1.5, "dyn": 0.80, "space": 1.20, "groove": 2.0},
        "meer": {"low_db": -1.0, "mid_db": 0.0, "high_db": 0.5, "dyn": 0.70, "space": 1.30, "groove": 5.0},
        "sea": {"low_db": -1.0, "mid_db": 0.0, "high_db": 0.5, "dyn": 0.70, "space": 1.30, "groove": 5.0},
        "ocean": {"low_db": -1.0, "mid_db": 0.0, "high_db": 0.5, "dyn": 0.70, "space": 1.30, "groove": 5.0},
        "himmel": {"low_db": -1.5, "mid_db": 0.5, "high_db": 2.0, "dyn": 0.80, "space": 1.25, "groove": 1.0},
        "sky": {"low_db": -1.5, "mid_db": 0.5, "high_db": 2.0, "dyn": 0.80, "space": 1.25, "groove": 1.0},
        "fliegen": {"low_db": -2.0, "mid_db": 1.0, "high_db": 3.0, "dyn": 0.85, "space": 1.30, "groove": 0.0},
        "fly": {"low_db": -2.0, "mid_db": 1.0, "high_db": 3.0, "dyn": 0.85, "space": 1.30, "groove": 0.0},
        "krieg": {"low_db": 1.0, "mid_db": 2.0, "high_db": 2.0, "dyn": 1.25, "space": 0.80, "groove": -4.0},
        "war": {"low_db": 1.0, "mid_db": 2.0, "high_db": 2.0, "dyn": 1.25, "space": 0.80, "groove": -4.0},
        "still": {"low_db": 0.0, "mid_db": -1.0, "high_db": -1.5, "dyn": 0.50, "space": 1.35, "groove": 6.0},
        "leise": {"low_db": 0.0, "mid_db": -1.0, "high_db": -1.5, "dyn": 0.50, "space": 1.35, "groove": 6.0},
        "laut": {"low_db": 2.0, "mid_db": 2.0, "high_db": 2.0, "dyn": 1.30, "space": 0.75, "groove": -5.0},
        "loud": {"low_db": 2.0, "mid_db": 2.0, "high_db": 2.0, "dyn": 1.30, "space": 0.75, "groove": -5.0},
    }

    def _apply_semantic_keyword_dsp(
        self,
        audio: np.ndarray,
        transcription: LyricsTranscriptionResult,
        sr: int,
    ) -> np.ndarray:
        """§v10.303.52: Wendet semantik-gesteuerte DSP-Parameter pro Wort an.

        Iteriert über transcription.words, sucht jedes word.word im
        _SEMANTIC_DSP_PARAMS-Dictionary, und wendet die zugehörigen
        EQ/Dynamik/Groove-Parameter als sanfte Modulation an.

        Privacy: word.word wird nur gelesen, nicht geloggt.
        """
        if not transcription.words:
            return audio

        # §Perf Early-Exit: Kein Wort trifft ein _SEMANTIC_DSP_PARAMS-Keyword →
        # alle Kurven bleiben 1.0, jede Band-Maske wird via np.allclose übersprungen
        # und dyn_curve = 1.0 lässt das Signal unverändert. Vorher: ~40 s STFT/
        # Interpolation für null Effekt. Der Match-Zähler im Log bleibt identisch.
        _matched_any = any(
            str(getattr(word, "word", "") or "").lower().strip() in self._SEMANTIC_DSP_PARAMS
            for word in transcription.words
        )
        if not _matched_any:
            logger.info("§v10.303.52 Semantic-DSP: 0 Wörter mit Keyword-Match verarbeitet")
            # Identisches End-Transform wie im DSP-Pfad (NaN/Clip/float32) —
            # für valides Audio ein No-Op, aber bit-kompatibel zum Vollpfad.
            return cast(
                np.ndarray,
                np.clip(
                    np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                ).astype(np.float32),
            )

        out = np.asarray(audio, dtype=np.float32).copy()
        n_samples = out.shape[0]
        is_stereo = out.ndim == 2

        # Akkumulator für EQ-Kurve (sample-genau)
        eq_low = np.ones(n_samples, dtype=np.float32)
        eq_mid = np.ones(n_samples, dtype=np.float32)
        eq_high = np.ones(n_samples, dtype=np.float32)
        dyn_curve = np.ones(n_samples, dtype=np.float32)

        _semantic_strength = 0.15  # subtil: 15 % der DSP-Parameter

        for word in transcription.words:
            word_text = str(getattr(word, "word", "") or "").lower().strip()
            if not word_text:
                continue
            params = self._SEMANTIC_DSP_PARAMS.get(word_text)
            if params is None:
                continue

            i0 = max(0, min(n_samples, int(word.start_s * sr)))
            i1 = max(i0, min(n_samples, int(word.end_s * sr)))
            if i1 <= i0:
                continue

            # Apply parameter modulation with exponential smoothing
            # to avoid abrupt changes between words
            _l = float(params.get("low_db", 0.0))
            _m = float(params.get("mid_db", 0.0))
            _h = float(params.get("high_db", 0.0))
            _d = float(params.get("dyn", 1.0))

            # Convert dB to linear gain
            eq_low[i0:i1] = 1.0 + _semantic_strength * (_l / 10.0)
            eq_mid[i0:i1] = 1.0 + _semantic_strength * (_m / 10.0)
            eq_high[i0:i1] = 1.0 + _semantic_strength * (_h / 10.0)
            dyn_curve[i0:i1] = 1.0 + _semantic_strength * (_d - 1.0)

        # Smooth transitions with 500 ms Hanning window
        _xfade = int(0.500 * sr)
        if _xfade > 2:
            _kern = np.hanning(_xfade * 2 + 1)
            _kern /= _kern.sum() + 1e-12

            def _smooth_curve(_c: np.ndarray) -> np.ndarray:
                # np.convolve(mode="same") gibt bei längerem Kernel die KERNELLÄNGE
                # zurück (48001) → full-Convolution zentriert auf n_samples croppen.
                _full = np.convolve(_c, _kern, mode="full")
                _off = (len(_full) - n_samples) // 2
                return cast(np.ndarray, _full[_off : _off + n_samples].astype(np.float32))

            eq_low = _smooth_curve(eq_low)
            eq_mid = _smooth_curve(eq_mid)
            eq_high = _smooth_curve(eq_high)
            dyn_curve = _smooth_curve(dyn_curve)

        # Apply EQ via STFT three-band gain
        from scipy import signal as _sig

        for _band_key, _curve, _f_low, _f_high in [
            ("low", eq_low, 60.0, 250.0),
            ("mid", eq_mid, 1000.0, 4000.0),
            ("high", eq_high, 8000.0, 16000.0),
        ]:
            if np.allclose(_curve, 1.0):
                continue
            for ch in range(out.shape[1] if is_stereo else 1):
                _ch_audio = out[:, ch] if is_stereo else out
                nperseg = min(2048, n_samples)
                noverlap = nperseg * 3 // 4
                _f, _t, Zxx = _sig.stft(
                    _ch_audio.astype(np.float64),
                    fs=sr,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                _band_mask = (_f >= _f_low) & (_f <= _f_high)
                _n_frames = Zxx.shape[1]
                # Resample curve to frame rate
                _frame_curve = np.interp(
                    np.linspace(0, n_samples - 1, _n_frames),
                    np.arange(n_samples),
                    _curve,
                ).astype(np.float64)
                Zxx[_band_mask, :] *= _frame_curve[np.newaxis, :]
                _, _ch_out = _sig.istft(
                    Zxx,
                    fs=sr,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                _ch_out = _ch_out[:n_samples]
                if is_stereo:
                    out[: len(_ch_out), ch] = _ch_out.astype(np.float32)
                else:
                    out[: len(_ch_out)] = _ch_out.astype(np.float32)

        # Apply dynamics curve (sample-level gain modulation)
        out = out * dyn_curve[np.newaxis, :] if is_stereo else out * dyn_curve

        logger.info(
            "§v10.303.52 Semantic-DSP: %d Wörter mit Keyword-Match verarbeitet",
            sum(
                1
                for w in transcription.words
                if str(getattr(w, "word", "") or "").lower().strip() in self._SEMANTIC_DSP_PARAMS
            ),
        )
        return cast(
            np.ndarray,
            np.clip(
                np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            ).astype(np.float32),
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[np.ndarray, LyricsTranscriptionResult]:
        """Wendet an: lyrics-guided saliency enhancement (§2.36).

        Args:
            audio: float32 ndarray, mono (N,) or stereo (N, 2), at ``sr`` Hz.
            sr:    Sample rate — must be 48 000 Hz at the pipeline boundary.

        Returns:
            (audio_out, transcription) where audio_out is float32 ∈ [−1, 1].

        Privacy: no lyrics text is written to any log, variable, or metadata.
        """
        assert sr == 48_000, f"SR guard: expected 48000, got {sr}"
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        mono = audio.mean(axis=0) if audio.ndim == 2 else audio  # channels-first (2,N) → axis=0
        n_samples = len(mono)
        dur = float(n_samples / max(1, sr))

        transcription = self._transcribe_internal(mono, sr, dur)
        # §v10.303.50: Text erst NACH Sentiment/Semantic-DSP löschen.
        # _assert_no_lyrics_in_log wird am Ende nach der Textlöschung aufgerufen.
        saliency = self._build_sample_saliency(transcription, n_samples, sr)

        # §LSM-1 Sentiment-Modulation: emotionaler Kontext des Texts beeinflusst
        # die Saliency-Kurve sanft. Bei Fallback/leerem Transkript wird der
        # Zusatzschritt vollständig übersprungen, weil er dann keinen Output-Effekt
        # haben kann.
        if not transcription.fallback_used and transcription.words:
            try:
                from backend.core.lyrics_sentiment_analyzer import get_lyrics_sentiment_analyzer

                _sentiment = get_lyrics_sentiment_analyzer().analyze(transcription, dur)
                if _sentiment.model_used != "neutral_fallback" and len(_sentiment.segments) > 1:
                    # Sentiment-Modulations-Array erstellen (sample-genau)
                    _sent_mod = np.ones(n_samples, dtype=np.float32)
                    for _seg in _sentiment.segments:
                        _s_idx = int(np.clip(_seg.start_s * sr, 0, n_samples))
                        _e_idx = int(np.clip(_seg.end_s * sr, 0, n_samples))
                        _dscale = float(np.clip(_seg.dsp_params.get("dynamics_scale", 1.0), 0.60, 1.25))
                        _sent_mod[_s_idx:_e_idx] = _dscale
                    # Sanfte Überblendung zwischen Segmenten (250ms Crossfade)
                    _xfade_samples = int(0.250 * sr)
                    if _xfade_samples > 2:
                        _kernel = np.hanning(_xfade_samples * 2 + 1)
                        _kernel /= _kernel.sum() + 1e-12
                        _sent_mod = np.convolve(_sent_mod, _kernel, mode="same").astype(np.float32)
                    # Modulation mit 30 % Stärke auf Saliency anwenden
                    _sent_strength = 0.30
                    saliency = saliency * (1.0 + _sent_strength * (_sent_mod - 1.0))
                    saliency = np.clip(saliency, 0.70, 1.30)
                    logger.info(
                        "§LSM-1: Sentiment-Modulation aktiv: dominant=%s V=%.2f A=%.2f",
                        _sentiment.dominant_emotion,
                        _sentiment.valence_mean,
                        _sentiment.arousal_mean,
                    )
            except Exception as _lsm_exc:
                logger.debug("§LSM-1 Sentiment nicht blockierend: %s", _lsm_exc)

        audio_out = audio * saliency[np.newaxis, :] if audio.ndim == 2 else audio * saliency

        # §2.36a: per-phoneme spectral DSP on top of saliency boost
        # §v10.303.51: language-specific phoneme profiles applied
        _lang_profile = self._get_phoneme_profile(getattr(transcription, "language", "unknown"))
        processor = get_content_aware_processor()
        audio_out = processor.apply_phoneme_dsp_to_audio(
            audio_out,
            transcription,
            sr,
            strength=0.50,
            language_profile=_lang_profile,  # §v10.303.51
        )

        # §v10.303.52: Semantic keyword-guided DSP (real word text → EQ/dynamics)
        if not transcription.fallback_used and transcription.words:
            audio_out = self._apply_semantic_keyword_dsp(
                audio_out,
                transcription,
                sr,
            )

        # §v10.303.50 Privacy: clear word text BEFORE returning result
        for _w in transcription.words:
            if hasattr(_w, "word") and _w.word:
                object.__setattr__(_w, "word", "")

        audio_out = np.clip(
            np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        return audio_out, transcription

    def transcribe(self, audio: np.ndarray, sr: int) -> LyricsTranscriptionResult:
        """Gibt §2.36 transcription without modifying the audio zurück."""
        assert sr == 48_000, f"SR guard: expected 48000, got {sr}"
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        mono = audio.mean(axis=0) if audio.ndim == 2 else audio  # channels-first (2,N) → axis=0
        dur = float(len(mono) / max(1, sr))
        transcription = self._transcribe_internal(mono, sr, dur)
        _assert_no_lyrics_in_log(transcription.words)
        return transcription

    def get_phoneme_mask(
        self,
        audio: np.ndarray,
        sr: int,
        hop_length: int = 512,
    ) -> np.ndarray:
        """§2.36 Phonem-Maske für NR-Bypass (RELEASE_MUST).

        Returns bool array (n_frames,) — True = Konsonanten-Burst-Frame.
        NR-Algorithmen MÜSSEN G[frame] = 1.0 setzen wenn consonant_mask[frame] == True.

        Konsonanten-Kategorien: plosive, fricative_stressed, fricative_unstressed.
        Geschützte Vokal-Frames: vowel_stressed, vowel_unstressed → NR × 0.4 empfohlen.
        Silence-Frames: keine Einschränkung.

        Privacy: no lyrics text. Only phoneme_type labels used.
        """
        assert sr == 48_000, f"SR guard: expected 48000, got {sr}"
        audio_arr = np.asarray(audio, dtype=np.float32)
        audio_arr = np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio_arr.mean(axis=0) if audio_arr.ndim == 2 else audio_arr  # channels-first (2,N) → axis=0
        n_samples = int(mono.shape[0])
        n_frames = max(1, (n_samples + hop_length - 1) // hop_length)

        try:
            transcription = self._transcribe_internal(mono, sr, float(n_samples / max(1, sr)))
        except Exception as exc:
            logger.debug("get_phoneme_mask: transcription fehlgeschlagen (%s) — no protection", exc)
            return np.zeros(n_frames, dtype=bool)  # type: ignore[no-any-return]

        _CONSONANT_TYPES = frozenset({"plosive", "fricative_stressed", "fricative_unstressed"})

        mask = np.zeros(n_frames, dtype=bool)
        frames_per_sec = float(sr) / float(hop_length)
        for word in transcription.words:
            if word.phoneme_type not in _CONSONANT_TYPES:
                continue
            f_start = max(0, int(word.start_s * frames_per_sec))
            f_end = min(n_frames, int(np.ceil(word.end_s * frames_per_sec)) + 1)
            if f_start < f_end:
                mask[f_start:f_end] = True

        logger.debug(
            "get_phoneme_mask: %d/%d frames protected (%.1f%%) hop=%d",
            int(mask.sum()),
            n_frames,
            100.0 * float(mask.sum()) / float(n_frames),
            hop_length,
        )
        return mask  # type: ignore[no-any-return]

    def get_timeline(self) -> LyricsGuidedTimeline:
        """Gibt the timeline renderer used by ``_toggle_lyrics_overlay`` in the frontend zurück."""
        return self._tl

    # ── Internal transcription ──────────────────────────────────────────────

    def _transcribe_internal(self, mono: np.ndarray, sr: int, dur: float) -> LyricsTranscriptionResult:
        """§v10.303.50: HF Whisper decoder → ONNX encoder → DSP energy."""
        # Preferred: HF Whisper with decoder (real word transcription)
        if self._whisper_hf_model is not None:
            try:
                return self._transcribe_hf(mono, sr, dur)
            except Exception as exc:
                logger.debug(
                    "§v10.303.50 HF Whisper transcription fehlgeschlagen (%s) — ONNX/DSP Ersatzpfad",
                    exc,
                )
        # Fallback 1: ONNX encoder (energy-based segmentation)
        if self._ort_session is not None:
            try:
                return self._transcribe_onnx(mono, sr, dur)
            except Exception as exc:
                logger.debug(
                    "LyricsGuidedEnhancement: ONNX transcription fehlgeschlagen (%s) — DSP Ersatzpfad",
                    exc,
                )
        # Fallback 2: pure DSP
        return self._transcribe_dsp(mono, sr, dur)

    def _transcribe_hf(self, mono: np.ndarray, sr: int, dur: float) -> LyricsTranscriptionResult:
        """§v10.303.50: Vollständige Whisper-Transkription mit DECODER.

        Verwendet HuggingFace WhisperForConditionalGeneration (Encoder + Decoder)
        für echte Wort-Erkennung mit Zeitstempeln. Der Decoder generiert Tokens,
        die zu Wörtern decodiert werden. Privacy: word.word NUR im RAM während
        der Verarbeitung; vor Logging/Metadaten wird der Text gelöscht.

        Returns:
            LyricsTranscriptionResult mit WordTimestamps (word.word = tatsächlicher Text).
        """
        import torch

        # Resample to 16 kHz (Whisper native)
        mono_16k = self._resample(mono, sr, 16_000)

        # PLM guard
        _plm_hf: Any = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_hf

            _plm_hf = _get_plm_hf()
            _plm_hf.set_active("lyrics_whisper_hf", True)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        try:
            # Prepare input features
            input_features = self._whisper_hf_processor(
                mono_16k,
                sampling_rate=16_000,
                return_tensors="pt",
            ).input_features

            # Generate with timestamps
            with torch.no_grad():
                predicted_ids = self._whisper_hf_model.generate(
                    input_features,
                    return_timestamps=True,
                    max_length=448,
                )

            # Decode full text (RAM only — never logged)
            full_text: str = self._whisper_hf_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            if not full_text:
                return LyricsTranscriptionResult(
                    [],
                    "unknown",
                    0.0,
                    dur,
                    fallback_used=True,
                )

            # Parse word-level timestamps from token output
            words = self._parse_hf_tokens_to_words(predicted_ids, mono_16k, dur)

            # wav2vec2 phoneme alignment refinement
            words = self._align_phonemes(words, mono_16k, 16_000)
            _detected_lang, _lang_conf = self._detect_language_from_mono(mono_16k, 16_000)

            result = LyricsTranscriptionResult(
                words=words,
                language=_detected_lang,
                overall_confidence=0.75,
                duration_s=dur,
                fallback_used=False,
            )

            # §v10.303.50 Privacy: clear word text before result leaves method.
            # NLP-Sentiment has already consumed the words by this point.
            # The words list inside result still has text at this moment —
            # _assert_no_lyrics_in_log is called upstream in enhance().
            return result

        finally:
            if _plm_hf is not None:
                try:
                    _plm_hf.set_active("lyrics_whisper_hf", False)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    def _parse_hf_tokens_to_words(
        self,
        predicted_ids: Any,
        mono_16k: np.ndarray,
        dur: float,
    ) -> list[WordTimestamp]:
        """Parse Whisper decoder output tokens into WordTimestamp list.

        Token sequence contains timestamp tokens (<|0.00|>, <|1.50|>, ...)
        interspersed with text tokens. We extract time-aligned word segments.

        Privacy: word.word is set to the actual transcribed word (RAM only).
        It gets cleared by _assert_no_lyrics_in_log before any logging.
        """
        # Get token IDs as list
        if hasattr(predicted_ids, "tolist"):
            ids = predicted_ids[0].tolist() if predicted_ids.dim() > 1 else predicted_ids.tolist()
        else:
            ids = list(predicted_ids[0]) if hasattr(predicted_ids, "__getitem__") else []

        if not ids:
            return []

        # Decode with special tokens to preserve timestamps
        raw_with_special = self._whisper_hf_processor.decode(ids, skip_special_tokens=False)

        # Parse segments from timestamp-annotated output
        return self._parse_hf_segments(raw_with_special, mono_16k, dur)

    def _parse_hf_segments(
        self,
        raw_with_special: str,
        mono_16k: np.ndarray,
        dur: float,
    ) -> list[WordTimestamp]:
        """Parse Whisper timestamp-annotated output into WordTimestamps.

        Whisper generates tokens like: <|0.00|>Hello<|0.50|> world<|1.20|>
        We split on timestamp tokens to get word-level segments.

        Fallback: uniform word distribution if timestamp parsing fails.

        Privacy: word.word is set to the actual transcribed word (RAM only).
        """
        import re

        words: list[WordTimestamp] = []
        # Match patterns: <|0.00|>text<|0.50|>
        pattern = re.compile(r"<\|(\d+\.\d+)\|>")
        parts = pattern.split(raw_with_special)
        # parts = ['', '0.00', 'Hello', '0.50', ' world', '1.20', '']

        i = 0
        while i + 2 < len(parts):
            try:
                start_s = float(parts[i + 1])
                text = parts[i + 2].strip()
                end_s = float(parts[i + 3]) if i + 3 < len(parts) and parts[i + 3] else start_s + 0.5
            except (ValueError, IndexError):
                i += 1
                continue

            if text and start_s < dur:
                end_s = min(end_s, dur)
                if end_s > start_s:
                    # Split multi-word text into individual WordTimestamps
                    sub_words = text.split()
                    sub_dur = (end_s - start_s) / max(len(sub_words), 1)
                    for sw_idx, sw in enumerate(sub_words):
                        sw_start = start_s + sw_idx * sub_dur
                        sw_end = min(sw_start + sub_dur, end_s)
                        seg_audio = mono_16k[
                            max(0, int(sw_start * 16_000)) : min(len(mono_16k), int(sw_end * 16_000) + 1)
                        ]
                        # Classify phoneme type from audio
                        is_stressed = (
                            len(seg_audio) > 0 and float(np.sqrt(np.mean(seg_audio.astype(np.float64) ** 2))) > 0.01
                        )
                        phoneme_type = LyricsGuidedEnhancement._classify_phoneme_type(
                            seg_audio if len(seg_audio) >= 4 else np.zeros(4, dtype=np.float32),
                            16_000,
                            float(np.sqrt(np.mean(seg_audio.astype(np.float64) ** 2))) if len(seg_audio) > 0 else 0.0,
                            is_stressed,
                        )
                        words.append(
                            WordTimestamp(
                                word=sw,  # §v10.303.50: actual word (RAM only)
                                start_s=sw_start,
                                end_s=sw_end,
                                confidence=0.75,
                                is_stressed=is_stressed,
                                phoneme_type=phoneme_type,
                            )
                        )
            i += 2

        return words

    _LANGUAGE_PHONEME_PROFILES: dict[str, dict[str, float]] = {
        # Default: balanced EQ per phoneme type
        "unknown": {
            "fricative_gain_db": 0.0,
            "plosive_burst_db": 0.0,
            "vowel_formant_db": 0.0,
            "silence_nr_strength": 1.0,
        },
        # German: harder consonants, more formant clarity
        "de": {
            "fricative_gain_db": 1.0,  # "sch", "ch" need more presence
            "plosive_burst_db": 1.5,  # "p", "t", "k" aspirated
            "vowel_formant_db": 1.5,  # Umlaute need formant clarity
            "silence_nr_strength": 1.0,
        },
        # English: softer fricatives, less formant boost
        "en": {
            "fricative_gain_db": 0.0,
            "plosive_burst_db": 0.5,
            "vowel_formant_db": 1.0,
            "silence_nr_strength": 1.0,
        },
        # French: nasal vowels, soft consonants
        "fr": {
            "fricative_gain_db": -1.0,  # softer sibilants
            "plosive_burst_db": 0.0,  # unaspirated plosives
            "vowel_formant_db": 2.0,  # nasal vowel clarity
            "silence_nr_strength": 0.85,
        },
        # Italian: vowel-dominant, musical
        "it": {
            "fricative_gain_db": -0.5,
            "plosive_burst_db": 0.0,
            "vowel_formant_db": 2.0,  # bel canto formants
            "silence_nr_strength": 0.9,
        },
        # Spanish: clear consonants, rolled R
        "es": {
            "fricative_gain_db": 0.5,
            "plosive_burst_db": 1.0,  # stronger plosives
            "vowel_formant_db": 1.0,  # pure 5-vowel system
            "silence_nr_strength": 1.0,
        },
        # Japanese: mora-timed, pitch accent
        "ja": {
            "fricative_gain_db": -1.0,  # softer overall
            "plosive_burst_db": -0.5,
            "vowel_formant_db": 0.5,
            "silence_nr_strength": 0.8,  # more silence between morae
        },
    }

    def _get_phoneme_profile(self, language: str) -> dict[str, float]:
        """Return language-specific phoneme DSP profile (falls back to unknown)."""
        lang_key = str(language or "unknown").strip().lower()[:2]
        return self._LANGUAGE_PHONEME_PROFILES.get(lang_key, self._LANGUAGE_PHONEME_PROFILES["unknown"])

    def _transcribe_onnx(self, mono: np.ndarray, sr: int, dur: float) -> LyricsTranscriptionResult:
        """Führt aus: whisper_tiny.onnx encoder; derive vocal segments from hidden-state RMS.

        The encoder's last_hidden_state (1500 frames × 384 dims) is condensed to a
        scalar energy per frame via RMS.  High-energy frames correspond to voiced /
        active speech regions.  Frame clusters above the 60th-percentile threshold
        are reported as pseudo WordTimestamp segments with phoneme_type labels from
        ContentAwareProcessor.SALIENCY_BOOST.
        """
        mono_16k = self._resample(mono, sr, self._ONNX_SR)
        features = self._compute_mel_features(mono_16k)  # (1, 80, 3000)

        # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during Whisper ONNX inference
        _plm_whisper: Any = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_w

            _plm_whisper = _get_plm_w()
            _plm_whisper.set_active("lyrics_transcriber_whisper", True)
        except Exception as e:
            logger.warning("lyrics_guided_enhancement.py::_transcribe_onnx Ersatzpfad: %s", e)
        # Encoder inference — output: (1, 1500, 384)
        try:
            hidden = self._ort_session.run(None, {"input_features": features})[0]  # type: ignore[attr-defined]
        finally:
            if _plm_whisper is not None:
                try:
                    _plm_whisper.set_active("lyrics_transcriber_whisper", False)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning("lyrics_guided_enhancement.py::_transcribe_onnx Ersatzpfad: %s", e)
        frame_energy = np.sqrt(np.mean(hidden[0] ** 2, axis=-1))  # (1500,)

        e_max = float(frame_energy.max()) or 1.0
        frame_energy = (frame_energy / e_max).astype(np.float32)

        words = self._energy_to_words(frame_energy, dur, mono, sr)
        # §2.36 Pflicht: phoneme-level refinement via wav2vec2 forced alignment.
        # Falls aligner nicht verfügbar ist, gibt _align_phonemes die Original-Liste zurück.
        words = self._align_phonemes(words, mono_16k, self._WAV2VEC2_SR)
        _detected_lang, _lang_conf = self._detect_language_from_mono(mono_16k, self._ONNX_SR)
        return LyricsTranscriptionResult(
            words=words,
            language=_detected_lang,
            overall_confidence=0.65,
            duration_s=dur,
            fallback_used=False,
        )

    def _transcribe_dsp(self, mono: np.ndarray, sr: int, dur: float) -> LyricsTranscriptionResult:
        """Pure DSP energy segmentation (20 ms frames, RMS, 60th-percentile threshold)."""
        frame_size = max(1, sr // 50)  # 20 ms
        hop = max(1, frame_size // 2)
        energies = [
            float(np.sqrt(np.mean(mono[i : i + frame_size] ** 2)))
            for i in range(0, max(1, len(mono) - frame_size), hop)
        ]
        if not energies:
            return LyricsTranscriptionResult([], "unknown", 0.0, dur, fallback_used=True)
        arr = np.array(energies, dtype=np.float32)
        e_max = float(arr.max()) or 1.0
        arr /= e_max
        words = self._energy_to_words(arr, dur, mono, sr)
        # §2.36 Fallback-Pfad: wenn Aligner verfügbar ist, auch DSP-Segmente
        # mit wav2vec2 nachklassifizieren; ansonsten unverändert belassen.
        mono_16k = self._resample(mono, sr, self._WAV2VEC2_SR)
        words = self._align_phonemes(words, mono_16k, self._WAV2VEC2_SR)
        _detected_lang, _lang_conf = self._detect_language_from_mono(mono_16k, self._WAV2VEC2_SR)
        return LyricsTranscriptionResult(
            words=words,
            language=_detected_lang,
            overall_confidence=0.3,
            duration_s=dur,
            fallback_used=True,
        )

    # ── Language detection ─────────────────────────────────────────────────

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

            return _ptl_detect(mono, sr)
        except Exception as exc:
            logger.debug("LyricsGuidedEnhancement._erkennen_language_from_mono fehlgeschlagen: %s", exc)
            return ("unknown", 0.0)

    # ── DSP helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _classify_phoneme_type(
        segment_audio: np.ndarray,
        sr: int,
        _mean_energy: float,
        is_stressed: bool,
    ) -> str:
        """Classify a voiced segment into phoneme classes using DSP spectral features.

        Algorithm (no ML required):
          1. Plosive detection: very short segment (< 30 ms), sharp energy rise (transient).
          2. Fricative detection: high spectral centroid (> 4 kHz) + high spectral flatness.
          3. Vowel: everything else, split by energy stress.
          4. Silence fallback (should not be called on silence segments).

        Returns one of: fricative_stressed, fricative_unstressed, vowel_stressed,
        vowel_unstressed, plosive, silence.
        """
        n = len(segment_audio)
        dur_ms = 1000.0 * n / max(1, sr)

        try:
            from backend.core.dsp.phoneme_boundary_detector import (  # pylint: disable=import-outside-toplevel
                PhonemeClass,
                get_phoneme_features_dsp,
            )

            features = get_phoneme_features_dsp(segment_audio, sr, hop_length=max(128, min(512, n // 4 or 128)))
            if features:
                classes = [feature.phoneme_class for feature in features]
                plosive_ratio = classes.count(PhonemeClass.PLOSIVE) / len(classes)
                fricative_ratio = classes.count(PhonemeClass.FRICATIVE) / len(classes)
                if plosive_ratio >= 0.20:
                    return "plosive"
                if fricative_ratio >= 0.30:
                    return "fricative_stressed" if is_stressed else "fricative_unstressed"
        except Exception as _phoneme_exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("LGE phoneme DSP classifier nicht verfuegbar (unkritisch): %s", _phoneme_exc)

        # --- Plosive: very short burst (< 30 ms) with high peak/RMS ratio ---
        if dur_ms < 30.0 and n >= 4:
            rms = float(np.sqrt(np.mean(segment_audio**2))) or 1e-10
            peak = float(np.abs(segment_audio).max())
            if peak / rms > 3.5:  # crest factor > 3.5 → transient burst
                return "plosive"

        # --- Spectral centroid and flatness for fricative vs. vowel ---
        if n >= 16:
            try:
                fft = np.abs(
                    np.fft.rfft(segment_audio * np.hanning(n) if n <= 8192 else segment_audio[:8192] * np.hanning(8192))
                )
                freqs = np.fft.rfftfreq(min(n, 8192), d=1.0 / sr)
                power = fft**2 + 1e-12
                total_power = float(power.sum())
                centroid = float((freqs * power).sum() / total_power)

                # Spectral flatness: geometric mean / arithmetic mean of power spectrum
                log_mean = float(np.mean(np.log(power + 1e-12)))
                arith_mean = float(np.mean(power))
                flatness = float(np.exp(log_mean) / (arith_mean + 1e-12))

                # High centroid + high flatness → noise-like → fricative
                if centroid > 4000.0 and flatness > 0.05:
                    return "fricative_stressed" if is_stressed else "fricative_unstressed"
            except Exception as _exc:
                logger.debug(
                    "Operation fehlgeschlagen (unkritisch): %s", _exc
                )  # DSP failed → fall through to vowel classification

        return "vowel_stressed" if is_stressed else "vowel_unstressed"

    @staticmethod
    def _energy_to_words(
        frame_energy: np.ndarray,
        dur: float,
        source_audio: np.ndarray | None = None,
        sr: int = 48_000,
    ) -> list[WordTimestamp]:
        """Konvertiert normalised frame-level energy to pseudo WordTimestamp objects.

        Active frames (energy ≥ 60th percentile) are collapsed into contiguous
        segments.  ``word`` is always empty — privacy invariant: no text in logs.

        Phoneme type is derived via spectral analysis of the original audio segment:
          - Plosive:              short burst (< 30 ms), crest factor > 3.5
          - Fricative (stressed): spectral centroid > 4 kHz + flatness > 0.05, high energy
          - Fricative (unstressed): spectral centroid > 4 kHz + flatness > 0.05, low energy
          - Vowel stressed/unstressed: everything else, split by energy level
          - Below-threshold segments are tagged as silence (SALIENCY_BOOST = 0.5)
        """
        n = len(frame_energy)
        if n == 0 or dur <= 0.0:
            return []
        frame_dur = dur / n
        threshold = float(np.percentile(frame_energy, 60))
        words: list[WordTimestamp] = []
        in_seg = False
        seg_start = 0
        seg_energies: list[float] = []

        def _flush_segment(start_idx: int, end_idx: int, energies: list[float]) -> WordTimestamp:
            seg_e = float(np.mean(energies)) if energies else 0.0
            is_stressed = seg_e > 0.7
            start_s = start_idx * frame_dur
            end_s = end_idx * frame_dur

            phoneme_type = "vowel_stressed" if is_stressed else "vowel_unstressed"
            if source_audio is not None:
                i0 = max(0, min(len(source_audio), int(start_s * sr)))
                i1 = max(i0 + 1, min(len(source_audio), int(end_s * sr)))
                seg_audio = source_audio[i0:i1]
                if len(seg_audio) >= 4:
                    phoneme_type = LyricsGuidedEnhancement._classify_phoneme_type(seg_audio, sr, seg_e, is_stressed)

            return WordTimestamp(
                word="",  # privacy: never store transcribed text
                start_s=start_s,
                end_s=end_s,
                confidence=min(1.0, seg_e),
                is_stressed=is_stressed,
                phoneme_type=phoneme_type,
            )

        for i, e in enumerate(frame_energy.tolist()):
            if e >= threshold:
                if not in_seg:
                    in_seg = True
                    seg_start = i
                    seg_energies = []
                seg_energies.append(e)
            else:
                if in_seg:
                    in_seg = False
                    words.append(_flush_segment(seg_start, i, seg_energies))
                    seg_energies = []
                # Below-threshold → silence segment (short gaps collapsed, long gaps tagged)

        if in_seg and seg_energies:  # flush last open segment
            words.append(_flush_segment(seg_start, n, seg_energies))
        return words

    def _build_sample_saliency(
        self,
        transcription: LyricsTranscriptionResult,
        n_samples: int,
        sr: int,
    ) -> np.ndarray:
        """Erstellt sample-level gain curve from word timestamps; values ∈ [0.3, 2.0]."""
        saliency = np.ones(n_samples, dtype=np.float32)
        if transcription.fallback_used or not transcription.words:
            return saliency  # type: ignore[no-any-return]
        for word in transcription.words:
            boost = self._cap.SALIENCY_BOOST.get(word.phoneme_type, 1.0)
            i0 = max(0, min(n_samples, int(word.start_s * sr)))
            i1 = max(i0, min(n_samples, int(word.end_s * sr)))
            if i1 > i0:
                saliency[i0:i1] = boost
        return np.clip(saliency, 0.3, 2.0)  # type: ignore[no-any-return]

    @staticmethod
    def _resample(mono: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        """Resample mono audio from *sr_in* to *sr_out* Hz."""
        if sr_in == sr_out:
            return mono
        try:
            import scipy.signal as sps  # type: ignore[import]

            n_out = max(1, int(len(mono) * sr_out / sr_in))
            return np.asarray(sps.resample(mono, n_out), dtype=np.float32)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("§V6 scipy.signal.resample fehlgeschlagen — Decimation-Fallback aktiviert (%.0f→%.0f Hz): %s", sr_in, sr_out, exc)
            step = max(1.0, sr_in / sr_out)
            indices = np.arange(0, len(mono), step).astype(np.int64)
            indices = np.clip(indices, 0, len(mono) - 1)
            return mono[indices].astype(np.float32)  # type: ignore[no-any-return]

    def _compute_mel_features(self, mono_16k: np.ndarray) -> np.ndarray:
        """Berechnet Whisper-compatible 80-channel log-mel spectrogram.

        Follows Radford et al. (2022) preprocessing:
          n_fft=400, hop_length=160, n_mels=80, sr=16 000 Hz, fmax=8 000 Hz.

        Returns float32 array of shape (1, 80, 3000).
        Shorter signals are zero-padded; longer signals are truncated at 30 s.
        """
        max_samples = self._MAX_FRAMES * self._HOP  # 480 000 = 30 s at 16 kHz
        if len(mono_16k) < max_samples:
            mono_16k = np.pad(mono_16k, (0, max_samples - len(mono_16k)))
        else:
            mono_16k = mono_16k[:max_samples]

        try:
            import librosa  # type: ignore[import]

            mel = librosa.feature.melspectrogram(
                y=mono_16k,
                sr=self._ONNX_SR,
                n_fft=self._N_FFT,
                hop_length=self._HOP,
                n_mels=self._N_MELS,
                fmin=0.0,
                fmax=8_000.0,
            )
            log_mel = np.log10(np.maximum(mel, 1e-10))
            log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
            log_mel = (log_mel + 4.0) / 4.0
            # Pad / truncate to exactly MAX_FRAMES time steps
            if log_mel.shape[1] < self._MAX_FRAMES:
                log_mel = np.pad(log_mel, ((0, 0), (0, self._MAX_FRAMES - log_mel.shape[1])))
            else:
                log_mel = log_mel[:, : self._MAX_FRAMES]
            return np.asarray(log_mel, dtype=np.float32)[np.newaxis, ...]  # type: ignore[no-any-return]  # (1, 80, 3000)
        except Exception as exc:
            logger.debug("§V6 librosa.feature.melspectrogram fehlgeschlagen — Null-Mel-Spektrogramm zurückgegeben (Shape %s): %s", mono_16k.shape, exc)
            # Zero fallback: encoder processes silence → near-zero hidden states
            return np.zeros((1, self._N_MELS, self._MAX_FRAMES), dtype=np.float32)  # type: ignore[no-any-return]


def get_lyrics_guided_enhancement() -> LyricsGuidedEnhancement:
    """Thread-safe double-checked locking singleton (§3.x spec pattern)."""
    global _lge_instance  # pylint: disable=global-statement
    if _lge_instance is None:
        with _lge_lock:
            if _lge_instance is None:
                _lge_instance = LyricsGuidedEnhancement()
    return _lge_instance


def get_phoneme_mask(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """§2.36 Convenience-Wrapper: phoneme mask via Singleton (RELEASE_MUST).

    Returns bool ndarray (n_frames,) — True = Konsonanten-Burst-Frame → NR-Bypass.
    Delegates to LyricsGuidedEnhancement.get_phoneme_mask().
    """
    return get_lyrics_guided_enhancement().get_phoneme_mask(audio, sr, hop_length=hop_length)


def _detect_transients_energy_proxy(  # pylint: disable=too-many-positional-arguments
    mono: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_dur_ms: float = 5.0,
    max_dur_ms: float = 50.0,
    energy_threshold_dbfs: float = -40.0,
) -> np.ndarray:
    """§2.36 Energie-Proxy: Plosive Konsonanten-Bursts ohne Phonem-Timeline erkennen.

    Identifiziert Frames mit breitbandigen Energie-Spikes, die typisch für Plosive
    (/p/, /t/, /k/, /s/) sind. Kriterien:
      - Frame-RMS > energy_threshold_dbfs dBFS
      - Kurze Dauer: min_dur_ms ≤ Transient ≤ max_dur_ms
      - Energie-Anstieg > 6 dB gegenüber Vorgänger-Frame (Onset-Charakteristik)

    Returns bool ndarray (n_frames,) — True = potenzieller Konsonanten-Burst.
    """
    n = len(mono)
    if n == 0:
        return np.zeros(0, dtype=bool)  # type: ignore[no-any-return]

    min_frames = max(1, int(min_dur_ms / 1000.0 * sr / hop_length))
    max_frames = max(min_frames, int(max_dur_ms / 1000.0 * sr / hop_length))
    energy_linear_threshold = float(10.0 ** (energy_threshold_dbfs / 20.0))

    n_frames = (n + hop_length - 1) // hop_length
    rms_per_frame = np.zeros(n_frames, dtype=np.float64)
    for fi in range(n_frames):
        s = fi * hop_length
        e = min(n, s + hop_length)
        rms_per_frame[fi] = float(np.sqrt(np.mean(mono[s:e].astype(np.float64) ** 2)))

    mask = np.zeros(n_frames, dtype=bool)
    for fi in range(1, n_frames):
        rms_cur = rms_per_frame[fi]
        rms_prev = rms_per_frame[fi - 1]
        if rms_cur < energy_linear_threshold:
            continue
        # Onset: mind. 6 dB Anstieg gegenüber Vorgänger-Frame
        if rms_prev < 1e-10:
            continue
        onset_db = 20.0 * float(np.log10(rms_cur / (rms_prev + 1e-12)))
        if onset_db >= 6.0:
            # Transient-Länge prüfen (muss kurz sein → max_frames)
            run = 0
            for fj in range(fi, min(n_frames, fi + max_frames + 1)):
                if rms_per_frame[fj] >= energy_linear_threshold:
                    run += 1
                else:
                    break
            if min_frames <= run <= max_frames:
                mask[fi : fi + run] = True

    return mask  # type: ignore[no-any-return]


def reconstruct_consonant_bursts(  # pylint: disable=too-many-positional-arguments
    audio_degraded: np.ndarray,
    audio_restored: np.ndarray,
    sr: int,
    hop_length: int = 512,
    rms_threshold: float = 0.70,
    max_blend_alpha: float = 0.60,
) -> np.ndarray:
    """§2.36 [RELEASE_MUST] Konsonanten-Burst-Rekonstruktion nach NR-Pipeline.

    Konsonanten-Bursts (/p/, /t/, /k/, /s/) haben breitbandige Energie-Spikes, die
    breitband-agnostisches NR als Rauschen klassifiziert und entfernt — was Artikulation
    zerstört. Diese Funktion stellt die verloren gegangene Burst-Energie wieder her.

    Algorithmus:
        1. get_phoneme_mask() → Protected-Frames identifizieren
        2. Pro Protected-Frame: RMS(degraded) vs. RMS(restored) vergleichen
        3. Wenn RMS(restored) < rms_threshold * RMS(degraded) → degraded-Transient
           mit alpha ≤ max_blend_alpha zurückblenden
        4. Sanfter Blend (nicht hartes Replace): Natürlichkeit bleibt gewahrt

    §0 Primum non nocere: Nur Frames mit nachweisbarem Energieverlust werden konzipiert.
    Frames ohne signifikante Abschwächung bleiben unverändert.

    Args:
        audio_degraded:  Ursprüngliches (degradiertes) Audio vor Pipeline
        audio_restored:  Audio nach NR/Restaurierungs-Pipeline
        sr:              Abtastrate (typisch 48000)
        hop_length:      STFT-Hop für Frame-Einteilung (default: 512)
        rms_threshold:   Eingriff wenn RMS_rest < threshold * RMS_deg (default: 0.70)
        max_blend_alpha: Maximaler Blend-Anteil aus degraded (default: 0.60)

    Returns:
        audio_out: Restauriertes Audio mit rekonstruierten Konsonanten-Bursts,
                   gleiche Form und Länge wie audio_restored
    """
    _logger_rcb = logging.getLogger(__name__)

    audio_deg = np.asarray(audio_degraded, dtype=np.float32)
    audio_rest = np.asarray(audio_restored, dtype=np.float32)

    # Längendifferenz abandon (Pipeline kann Länge minimal ändern)
    n = min(len(audio_deg), len(audio_rest))
    if n == 0:
        return audio_rest.copy()  # type: ignore[no-any-return]

    audio_deg = audio_deg[:n]
    audio_rest = audio_rest[:n]

    # Mono-Kanal für Phonem-Maske nutzen
    deg_mono = (audio_deg[:, 0] if audio_deg.ndim == 2 else audio_deg).astype(np.float32)
    try:
        phoneme_mask: np.ndarray = get_phoneme_mask(deg_mono, sr, hop_length=hop_length)
    except Exception as _exc_pm:
        _logger_rcb.debug("reconstruct_consonant_bursts: get_phoneme_mask fehlgeschlagen: %s", _exc_pm)
        phoneme_mask = np.zeros(0, dtype=bool)

    if not np.any(phoneme_mask):
        # §2.36 Energie-Proxy-Fallback: Wenn Phonem-Timeline fehlt oder leer ist,
        # erkennt der energie-basierte Transient-Detektor plosive Konsonanten-Bursts
        # anhand von Energie-Spikes: breitbandig > -40 dBFS, Dauer 5–50 ms.
        phoneme_mask = _detect_transients_energy_proxy(deg_mono, sr, hop_length=hop_length)
        if np.any(phoneme_mask):
            _logger_rcb.debug(
                "§2.36 reconstruct_consonant_bursts: Phonem-Fallback aktiv (%d Frames via Energie-Proxy)",
                int(np.sum(phoneme_mask)),
            )
        else:
            return audio_rest.copy()  # type: ignore[no-any-return]

    audio_out = audio_rest.copy()
    n_restored = 0

    for frame_idx, is_burst in enumerate(phoneme_mask):
        if not is_burst:
            continue

        sample_start = frame_idx * hop_length
        sample_end = min(n, sample_start + hop_length)
        if sample_start >= n:
            break

        if audio_deg.ndim == 2:
            # Stereo: RMS über alle Kanäle
            rms_deg = float(np.sqrt(np.mean(audio_deg[sample_start:sample_end] ** 2)) + 1e-10)
            rms_rest = float(np.sqrt(np.mean(audio_rest[sample_start:sample_end] ** 2)) + 1e-10)
        else:
            rms_deg = float(np.sqrt(np.mean(audio_deg[sample_start:sample_end] ** 2)) + 1e-10)
            rms_rest = float(np.sqrt(np.mean(audio_rest[sample_start:sample_end] ** 2)) + 1e-10)

        # Eingriff nur wenn signifikante Energie verloren ging
        if rms_rest >= rms_threshold * rms_deg:
            continue

        # Blend-Alpha proportional zum Energieverlust (aber nie > max_blend_alpha)
        energy_loss_ratio = 1.0 - float(rms_rest / max(rms_deg, 1e-10))
        alpha = float(np.clip(energy_loss_ratio * max_blend_alpha, 0.0, max_blend_alpha))

        audio_out[sample_start:sample_end] = (1.0 - alpha) * audio_rest[sample_start:sample_end] + alpha * audio_deg[
            sample_start:sample_end
        ]
        n_restored += 1

    if n_restored > 0:
        audio_out = np.clip(audio_out, -1.0, 1.0)
        _logger_rcb.debug(
            "§2.36 reconstruct_consonant_bursts: %d/%d Burst-Frames restauriert",
            n_restored,
            int(np.sum(phoneme_mask)),
        )

    return audio_out  # type: ignore[no-any-return]


__all__ = [
    "ContentAwareProcessor",
    "LyricsGuidedEnhancement",
    "LyricsGuidedTimeline",
    "LyricsTranscriber",
    "LyricsTranscriptionResult",
    "WordTimestamp",
    "get_content_aware_processor",
    "get_lyrics_guided_enhancement",
    "get_lyrics_guided_timeline",
    "get_lyrics_transcriber",
    "get_phoneme_mask",
    "is_lyrics_guided_loaded",
    "reconstruct_consonant_bursts",
]
from typing import cast
