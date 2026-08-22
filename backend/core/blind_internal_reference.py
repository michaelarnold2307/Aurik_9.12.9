"""
BlindInternalReference — §Gap5 Cleanest-Segment Finder (Aurik 10.0.0.x)
=======================================================================

Finds the highest-quality (least degraded) segments within a song
to use as internal quality reference — without any external reference.

These reference segments are used for:
  - Calibrating timbral_fidelity gates (§0d best_carrier_checkpoint)
  - Providing a clean template for Wiener filter estimation (phase_29)
  - Grounding the HolisticPerceptualGate against the song's own best quality

Algorithm (per 5-second window):
  1. SNR proxy: noise_floor (RMS of quietest 10 % of sub-frames) vs signal RMS
  2. Spectral clarity: 1 − spectral_entropy (cleaner → lower entropy)
  3. Transient richness: peak/RMS ratio (higher → more dynamic range)
  Combined score = 0.40 * snr_proxy + 0.35 * spectral_clarity + 0.25 * transient_richness

Returns top-N cleanest segments with score and audio_slice.

Usage in UV3:
    from backend.core.blind_internal_reference import get_blind_internal_reference
    _bir = get_blind_internal_reference()
    _bir_result = _bir.find(audio, sr)
    context["blind_reference"] = _bir_result.to_dict()

Singleton-Pattern. Non-blocking.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_S: float = 5.0
_HOP_S: float = 2.5
_TOP_N: int = 3
_MIN_SEGMENT_DUR_S: float = 2.0
_SUB_FRAME_N: int = 512
_N_FFT: int = 2048


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BlindReferenceSegment:
    """single candidate 'cleanest' segment."""

    start_s: float
    end_s: float
    score: float
    snr_proxy_db: float
    spectral_clarity: float

    def to_dict(self) -> dict:
        """Serialisiert this segment as a dictionary for UV3 context injection."""
        return {
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "score": float(self.score),
            "snr_proxy_db": float(self.snr_proxy_db),
            "spectral_clarity": float(self.spectral_clarity),
        }


@dataclass
class BlindReferenceResult:
    """Top-N cleanest segments found in the song."""

    segments: list[BlindReferenceSegment] = field(default_factory=list)
    """Top-N cleanest segments, sorted by score descending."""

    global_snr_proxy_db: float = 0.0
    """Median SNR proxy across all windows (dB); low = heavily degraded material."""

    best_score: float = 0.0
    """Score of the single best segment [0, 1]."""

    def to_dict(self) -> dict:
        """Serialisiert the reference result as a dictionary for UV3 context injection."""
        return {
            "segments": [s.to_dict() for s in self.segments],
            "global_snr_proxy_db": float(self.global_snr_proxy_db),
            "best_score": float(self.best_score),
        }


# ---------------------------------------------------------------------------
# BlindInternalReference
# ---------------------------------------------------------------------------


class BlindInternalReference:
    """Findet die saubersten Segmente eines Songs als interne Referenz."""

    def find(self, audio: np.ndarray, sr: int, top_n: int = _TOP_N) -> BlindReferenceResult:
        """Analysiert einen Song und gibt die N saubersten Segmente zurück.

        Args:
            audio:  Input audio (mono or stereo, any length).
            sr:     Sample rate of ``audio``.
            top_n:  Maximum number of segments to return.

        Returns:
            BlindReferenceResult with ranked segment list and global stats.
        """
        result = BlindReferenceResult()
        try:
            mono = _to_mono(audio)
            n = len(mono)
            min_n = int(_MIN_SEGMENT_DUR_S * sr)
            if n < min_n:
                return result

            window_n = int(_WINDOW_S * sr)
            hop_n = int(_HOP_S * sr)
            candidates: list[BlindReferenceSegment] = []

            pos = 0
            while pos + window_n <= n:
                seg = mono[pos : pos + window_n]
                seg_score, snr_db, clarity = self._score_segment(seg, sr)
                candidates.append(
                    BlindReferenceSegment(
                        start_s=pos / sr,
                        end_s=(pos + window_n) / sr,
                        score=seg_score,
                        snr_proxy_db=snr_db,
                        spectral_clarity=clarity,
                    )
                )
                pos += hop_n

            # Include tail if ≥ min duration
            if n - pos >= min_n:
                seg = mono[pos:]
                seg_score, snr_db, clarity = self._score_segment(seg, sr)
                candidates.append(
                    BlindReferenceSegment(
                        start_s=pos / sr,
                        end_s=n / sr,
                        score=seg_score,
                        snr_proxy_db=snr_db,
                        spectral_clarity=clarity,
                    )
                )

            if not candidates:
                return result

            # Sort by score descending, pick top-N
            candidates.sort(key=lambda c: c.score, reverse=True)
            result.segments = candidates[:top_n]
            result.best_score = candidates[0].score
            result.global_snr_proxy_db = float(np.median([c.snr_proxy_db for c in candidates]))

            logger.info(
                "BlindInternalReference: best_Wert=%.3f snr_proxy=%.1fdB (bir_snr_proxy, advisory) top=%d",
                result.best_score,
                result.global_snr_proxy_db,
                len(result.segments),
            )
        except Exception as exc:
            logger.debug("BlindInternalReference nicht blockierend: %s", exc)
        return result

    @staticmethod
    def _score_segment(seg: np.ndarray, _sr: int) -> tuple[float, float, float]:
        """Berechnet quality score, SNR proxy (dB), and spectral clarity for a segment.

        Returns:
            (combined_score [0,1], snr_proxy_db, spectral_clarity [0,1])
        """
        # 1. SNR proxy via noise floor
        n_sub = max(1, len(seg) // _SUB_FRAME_N)
        rms_vals = []
        for i in range(n_sub):
            chunk = seg[i * _SUB_FRAME_N : (i + 1) * _SUB_FRAME_N]
            if len(chunk) > 0:
                rms_vals.append(float(np.sqrt(np.mean(chunk**2) + 1e-12)))

        if not rms_vals:
            return 0.0, -60.0, 0.0

        rms_arr = np.array(rms_vals)
        signal_rms = float(np.percentile(rms_arr, 75))  # signal level
        noise_rms = float(np.percentile(rms_arr, 10))  # noise floor proxy
        snr_db = float(20.0 * np.log10(signal_rms / (noise_rms + 1e-12) + 1e-6))
        snr_db = float(np.clip(snr_db, -20.0, 60.0))
        snr_norm = float((snr_db + 20.0) / 80.0)  # map [-20,60] → [0,1]

        # 2. Spectral clarity: 1 − normalized spectral entropy
        mag = np.abs(np.fft.rfft(seg[:_N_FFT] * np.hanning(min(_N_FFT, len(seg))), n=_N_FFT)) + 1e-10
        prob = mag / (mag.sum() + 1e-10)
        entropy = float(-np.sum(prob * np.log(prob + 1e-10)))
        max_entropy = float(np.log(len(prob) + 1e-10))
        spectral_clarity = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

        # 3. Transient richness: crest factor (peak/RMS)
        peak = float(np.percentile(np.abs(seg), 99.9))
        rms_total = float(np.sqrt(np.mean(seg**2) + 1e-12))
        crest = float(np.clip(peak / (rms_total + 1e-12), 1.0, 20.0))
        crest_norm = float((crest - 1.0) / 19.0)  # map [1,20] → [0,1]

        combined = 0.40 * snr_norm + 0.35 * spectral_clarity + 0.25 * crest_norm
        return float(np.clip(combined, 0.0, 1.0)), snr_db, spectral_clarity


# ---------------------------------------------------------------------------
# Singleton-Zugriff
# ---------------------------------------------------------------------------

_instance: BlindInternalReference | None = None
_lock = threading.Lock()


def get_blind_internal_reference() -> BlindInternalReference:
    """Thread-sicherer Singleton-Zugriff."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BlindInternalReference()
                logger.info("BlindInternalReference initialisiert (§Gap5)")
    return _instance


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)  # type: ignore[no-any-return]
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            return audio.mean(axis=0).astype(np.float32)  # type: ignore[no-any-return]
        return audio.mean(axis=-1).astype(np.float32)  # type: ignore[no-any-return]
    return audio.flatten().astype(np.float32)  # type: ignore[no-any-return]
