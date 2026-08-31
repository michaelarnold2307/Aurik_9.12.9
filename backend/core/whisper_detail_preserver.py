"""
whisper_detail_preserver.py — v10 Gesangsdetail-Erhalt
==============================================================

Schutzt leise Gesangsdetails (Flustern, Atmer, leise Passagen) vor
dem Verlust durch aggressive Rauschunterdruckung und Entzerrung.

Das Problem: Noise Reduction arbeitet mit SNR-Schwellen. Leise
Gesangsdetails (< -40 dBFS) werden oft als "Rauschen" klassifiziert
und entfernt. Das zerstort die emotionale Intimitat einer Aufnahme.

Losung: Whisper-Aware Preservation Mask (WAPM):
  1. Erkennt leise, aber tonal-relevante Segmente (< -30 dBFS RMS)
  2. Erkennt spektrale Signaturen menschlicher Stimme (Formanten)
  3. Erstellt eine Preservation-Mask die diese Bereiche schutzt
  4. Gibt maximale erlaubte Attenuation pro Frame zuruck

Wissenschaftliche Basis:
  - Titze (1994): Principles of Voice Production
  - Sundberg (1987): The Science of the Singing Voice
  - ANSI S3.5: Speech Intelligibility Index

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Konstanten
WHISPER_RMS_THRESHOLD_DBFS: float = -40.0  # Alles leiser = Whisper-Kandidat
WHISPER_RMS_FLOOR_DBFS: float = -75.0  # Alles darunter = Stille/Ignorieren
WHISPER_MIN_DURATION_S: float = 0.1  # Minimale Dauer fur Whisper-Segment
WHISPER_FORMANT_MIN_CORRELATION: float = 0.3  # Min. Formant-Korrelation fur "Stimme"
WHISPER_MAX_ATTENUATION_DB: float = 3.0  # Max erlaubte Dampfung in Whisper-Zonen
WHISPER_SAFETY_MARGIN: float = 0.15  # 15% Sicherheitspuffer um Segmente

# Geschatzte Formant-Frequenzen (Mittelwerte nach Titze)
# F1-F3 fur /a/, /i/, /u/ — decken Grossteil der Vokale ab
_REFERENCE_FORMANTS: dict[str, list[float]] = {
    "male": [650, 1100, 2500],  # F1, F2, F3 fur /a/ (m)
    "female": [850, 1350, 2900],  # F1, F2, F3 fur /a/ (f)
}


@dataclass
class WhisperSegment:
    """Ein erkanntes Whisper/leises Gesangs-Segment."""

    start_s: float
    end_s: float
    rms_dbfs: float
    formant_score: float  # 0-1, wie "stimmhaft" ist das Segment
    confidence: float  # 0-1, Gesamt-Konfidenz dass es Gesang ist


@dataclass
class WhisperPreservationResult:
    """Ergebnis der Whisper-Detail-Analyse."""

    segments: list[WhisperSegment] = field(default_factory=list)
    preservation_mask: np.ndarray | None = None  # 1D, Lange = n_frames
    max_attenuation_db: float = 0.0
    whisper_ratio: float = 0.0  # Anteil des Signals der "whisper" ist
    recommendation: str = ""


def _rms_db(frame: np.ndarray) -> float:
    """RMS in dBFS."""
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
    return float(20.0 * np.log10(rms))


def _compute_formant_score(frame: np.ndarray, sr: int, formant_refs: list[float]) -> float:
    """Berechnet wie gut die spektrale Struktur zu Referenz-Formanten passt."""
    n_fft = min(2048, len(frame))
    spec = np.abs(np.fft.rfft(frame[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(sr, 1))

    scores = []
    for f_ref in formant_refs:
        # Finde Energie in 50Hz-Band um Referenz-Formanten
        idx = int(np.argmin(np.abs(freqs - f_ref)))
        bandwidth = max(1, int(25.0 / (sr / n_fft)))
        lo = int(max(0, idx - bandwidth))
        hi = int(min(len(spec) - 1, idx + bandwidth))

        peak_energy = float(np.max(spec[lo : hi + 1]))  # type: ignore[operator]
        mean_energy = float(np.mean(spec[1:]))  # ohne DC

        if mean_energy > 0:
            scores.append(min(1.0, peak_energy / (mean_energy * 3.0)))
        else:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def analyze_whisper_detail(
    audio: np.ndarray,
    sr: int,
    *,
    gender: str = "auto",
    min_duration_s: float = WHISPER_MIN_DURATION_S,
) -> WhisperPreservationResult:
    """Analysiert Audio auf leise Gesangsdetails und erstellt Preservation-Mask.

    Args:
        audio: Eingabe-Audio (mono oder stereo)
        sr: Sample-Rate
        gender: "male", "female", oder "auto"
        min_duration_s: Minimale Dauer fur ein Whisper-Segment

    Returns:
        WhisperPreservationResult mit Segmenten und Preservation-Mask
    """
    arr = np.asarray(audio, dtype=np.float64)

    # Mono
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()

    total_duration_s = len(mono) / max(sr, 1)
    if total_duration_s < min_duration_s:
        return WhisperPreservationResult(recommendation="Signal zu kurz fur Whisper-Analyse")

    # Frame-basierte Analyse: 50ms Fenster, 25ms Hop
    frame_len = int(0.050 * sr)
    hop_len = int(0.025 * sr)
    n_frames = max(1, (len(mono) - frame_len) // hop_len + 1)

    # Formant-Referenzen wahlen
    if gender == "male":
        formant_refs = _REFERENCE_FORMANTS["male"]
    elif gender == "female":
        formant_refs = _REFERENCE_FORMANTS["female"]
    else:
        formant_refs = _REFERENCE_FORMANTS["male"]  # Default

    frame_rms_db: list[float] = []
    frame_formant: list[float] = []
    frame_times: list[float] = []

    for i in range(n_frames):
        start = i * hop_len
        frame = mono[start : start + frame_len]

        rms_db_val = _rms_db(frame)
        formant_score = _compute_formant_score(frame, sr, formant_refs)

        frame_rms_db.append(rms_db_val)
        frame_formant.append(formant_score)
        frame_times.append(start / sr)

    frame_rms_db = np.array(frame_rms_db)  # type: ignore[assignment]
    frame_formant = np.array(frame_formant)  # type: ignore[assignment]

    # Whisper-Kandidaten: RMS zwischen Floor und Threshold, + Formant-Score > Schwelle
    whisper_candidates = (
        (frame_rms_db > WHISPER_RMS_FLOOR_DBFS)  # type: ignore[operator]
        & (frame_rms_db < WHISPER_RMS_THRESHOLD_DBFS)  # type: ignore[operator]
        & (frame_formant > WHISPER_FORMANT_MIN_CORRELATION)  # type: ignore[operator]
    )

    # Segmente gruppieren (zusammenhangende Frames)
    segments: list[WhisperSegment] = []
    in_segment = False
    seg_start = 0.0
    seg_rms_vals: list[float] = []
    seg_formant_vals: list[float] = []

    for i in range(n_frames):
        if whisper_candidates[i] and not in_segment:
            in_segment = True
            seg_start = frame_times[i]
            seg_rms_vals = [frame_rms_db[i]]
            seg_formant_vals = [frame_formant[i]]
        elif whisper_candidates[i] and in_segment:
            seg_rms_vals.append(frame_rms_db[i])
            seg_formant_vals.append(frame_formant[i])
        elif not whisper_candidates[i] and in_segment:
            in_segment = False
            seg_end = frame_times[i]
            duration = seg_end - seg_start

            if duration >= min_duration_s:
                avg_rms = float(np.mean(seg_rms_vals))
                avg_formant = float(np.mean(seg_formant_vals))
                confidence = float(
                    np.clip(avg_formant * (1.0 + (WHISPER_RMS_THRESHOLD_DBFS - avg_rms) / 30.0), 0.0, 1.0)
                )
                segments.append(
                    WhisperSegment(
                        start_s=seg_start,
                        end_s=seg_end,
                        rms_dbfs=avg_rms,
                        formant_score=avg_formant,
                        confidence=confidence,
                    )
                )

    # Letztes Segment abschliessen
    if in_segment:
        seg_end = frame_times[-1]
        duration = seg_end - seg_start
        if duration >= min_duration_s:
            avg_rms = float(np.mean(seg_rms_vals))
            avg_formant = float(np.mean(seg_formant_vals))
            confidence = float(np.clip(avg_formant * (1.0 + (WHISPER_RMS_THRESHOLD_DBFS - avg_rms) / 30.0), 0.0, 1.0))
            segments.append(
                WhisperSegment(
                    start_s=seg_start,
                    end_s=seg_end,
                    rms_dbfs=avg_rms,
                    formant_score=avg_formant,
                    confidence=confidence,
                )
            )

    # Preservation-Mask erstellen: 1.0 = voll schutzen, 0.0 = normal
    preserv_mask = np.zeros(n_frames, dtype=np.float32)
    for seg in segments:
        safety_start = max(0.0, seg.start_s - WHISPER_SAFETY_MARGIN)
        safety_end = min(total_duration_s, seg.end_s + WHISPER_SAFETY_MARGIN)

        start_frame = int(safety_start * sr / hop_len)
        end_frame = min(n_frames, int(safety_end * sr / hop_len) + 1)

        # Gauss-formige Maske um Segment-Mitte
        seg_center = (seg.start_s + seg.end_s) / 2.0
        seg_half = (seg.end_s - seg.start_s) / 2.0 + WHISPER_SAFETY_MARGIN

        for j in range(start_frame, end_frame):
            t = j * hop_len / sr
            dist = abs(t - seg_center) / max(seg_half, 0.01)
            weight = float(np.exp(-(dist**2) / 2.0)) * seg.confidence
            preserv_mask[j] = max(preserv_mask[j], weight)

    # Berechne Whisper-Anteil
    total_whisper_s = sum(s.end_s - s.start_s for s in segments)
    whisper_ratio = total_whisper_s / max(total_duration_s, 0.01)

    max_atten = WHISPER_MAX_ATTENUATION_DB  # Default
    if whisper_ratio > 0.3:
        max_atten = 1.5  # Viele leise Passagen → konservativer
    elif whisper_ratio < 0.05:
        max_atten = 5.0  # Wenig leise Passagen → mehr Spielraum

    # Recommendation
    if not segments:
        rec = "Keine leisen Gesangsdetails erkannt — normale Verarbeitung."
    elif whisper_ratio > 0.2:
        rec = (
            f"Viele leise Gesangspassagen ({whisper_ratio:.0%} des Signals) — "
            f"maximale Dampfung auf {max_atten:.1f}dB begrenzen, "
            f"um emotionale Intimitat zu erhalten."
        )
    else:
        rec = (
            f"Leise Gesangsdetails in {len(segments)} Segmenten erkannt — "
            f"Dampfung in diesen Bereichen auf {max_atten:.1f}dB begrenzen."
        )

    return WhisperPreservationResult(
        segments=segments,
        preservation_mask=preserv_mask if len(segments) > 0 else None,
        max_attenuation_db=max_atten,
        whisper_ratio=whisper_ratio,
        recommendation=rec,
    )


def apply_whisper_preservation(
    audio: np.ndarray,
    sr: int,
    whisper: WhisperPreservationResult,
    *,
    processed_audio: np.ndarray,
) -> np.ndarray:
    """Mischt Original und bearbeitetes Audio basierend auf Whisper-Maske.

    In Whisper-Zonen wird mehr Original-Anteil beibehalten, um leise Gesangsdetails
    nicht durch aggressive Rauschunterdruckung zu verlieren.

    Args:
        audio: Original-Audio
        sr: Sample-Rate
        whisper: WhisperPreservationResult
        processed_audio: Das durch die Pipeline verarbeitete Audio

    Returns:
        Gemischtes Audio mit erhaltenen Whisper-Details
    """
    if whisper.preservation_mask is None or len(whisper.segments) == 0:
        return processed_audio

    orig = np.asarray(audio, dtype=np.float64)
    proc = np.asarray(processed_audio, dtype=np.float64)

    # Resample preservation_mask auf Sample-Ebene
    hop_s = 0.025  # 25ms Hop (analog zur Analyse)
    n_out: int

    if orig.ndim == 1:
        n_out = len(orig)
    else:
        n_out = orig.shape[0]

    # Erstelle Sample-genaue Maske
    mask_samples = np.zeros(n_out, dtype=np.float64)

    for i in range(len(whisper.preservation_mask)):
        t = i * hop_s
        idx_start = int(t * sr)
        idx_end = min(n_out, int((t + hop_s) * sr))
        if idx_start < n_out:
            mask_samples[idx_start:idx_end] = float(whisper.preservation_mask[i])

    # Weichzeichnen (20ms Gauss)
    blur_samples = int(0.020 * sr)
    if blur_samples > 2:
        kernel = np.exp(-(np.linspace(-3, 3, blur_samples) ** 2) / 2.0)
        kernel /= kernel.sum()
        if orig.ndim == 1:
            mask_samples = np.convolve(mask_samples, kernel, mode="same")
        else:
            for ch in range(min(orig.shape[1], 2)):
                mask_samples[:, ch] = np.convolve(mask_samples[:, ch], kernel, mode="same")

    mask_samples = np.clip(mask_samples, 0.0, 1.0)

    # Mische: Original * mask + Processed * (1 - mask)
    if orig.ndim == 1:
        mask_2d = mask_samples
        result = orig * mask_2d + proc * (1.0 - mask_2d)
    else:
        mask_2d = np.atleast_2d(mask_samples).T if mask_samples.ndim == 1 else mask_samples
        result = orig * mask_2d + proc * (1.0 - mask_2d)

    return cast(np.ndarray, result.astype(np.float32))


# Singleton


class WhisperDetailAnalyzer:
    """Thread-sicherer Analyzer fur Whisper/leise Gesangsdetails."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_result: WhisperPreservationResult | None = None

    def analyze(self, audio: np.ndarray, sr: int, **kwargs: Any) -> WhisperPreservationResult:
        result = analyze_whisper_detail(audio, sr, **kwargs)
        with self._lock:
            self._last_result = result
        return result

    @property
    def last_result(self) -> WhisperPreservationResult | None:
        return self._last_result


_whisper_instance: WhisperDetailAnalyzer | None = None


def get_whisper_singleton() -> WhisperDetailAnalyzer:
    """Thread-sicherer Singleton-Accessor."""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperDetailAnalyzer()
    return _whisper_instance
