"""
§v10.101 Segment-weise Qualitätsbewertung — Aurik 10

Zweck: Lokale Qualitätsspitzen/-täler sind hörbar relevanter als der globale Mittelwert.
Misst OQS/HPI in 5-Sekunden-Fenstern. Segmente < Threshold werden isoliert neu
restauriert mit angepasster Phasen-Konfiguration.

Usage:
    from backend.core.segment_quality_scorer import SegmentQualityScorer

    scorer = SegmentQualityScorer(window_seconds=5.0)
    scores = scorer.score(audio, sr=48000)
    bad_segments = [s for s in scores if s.score < 0.7]
    # → isoliert neu restaurieren
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Segment-Score-Datenstruktur ─────────────────────────────────────────


@dataclass
class SegmentScore:
    """Qualitätsscore für ein einzelnes Segment.

    Attributes:
        start_sample: Start-Sample des Segments.
        end_sample: Ende-Sample des Segments.
        start_time_s: Startzeit in Sekunden.
        duration_s: Dauer in Sekunden.
        score: Qualitätsscore [0, 1].
        rms_dbfs: RMS-Pegel in dBFS.
        peak_dbfs: Peak-Pegel in dBFS.
        snr_estimate_db: Geschätzter SNR in dB (positiv = gut).
    """

    start_sample: int
    end_sample: int
    start_time_s: float
    duration_s: float
    score: float
    rms_dbfs: float
    peak_dbfs: float
    snr_estimate_db: float


def _compute_segment_score(
    segment: np.ndarray,
    sr: int,
) -> tuple[float, float, float, float]:
    """Berechnet Qualitätsscore für ein Segment.

    Nutzt RMS, Peak und SNR-Schätzung als Proxy für lokale Qualität.

    Returns:
        (score [0,1], rms_dbfs, peak_dbfs, snr_estimate_db)
    """
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # Mono
    if segment.ndim == 2:
        mono = np.mean(segment, axis=0)
    else:
        mono = segment

    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.abs(mono).max())

    if peak < 1e-10:
        return 0.0, -100.0, -100.0, 0.0

    rms_dbfs = 20.0 * np.log10(rms + 1e-10)
    peak_dbfs = 20.0 * np.log10(peak + 1e-10)

    # SNR-Schätzung: RMS vs. Rauschboden (unterste 5% der Energie)
    energy = mono**2
    noise_floor = float(np.percentile(energy, 5))
    signal_level = float(np.mean(energy))

    if noise_floor < 1e-20 or signal_level < 1e-20:
        snr_db = 0.0
    else:
        snr_db = 10.0 * np.log10(signal_level / (noise_floor + 1e-20))

    # Score aus SNR und RMS/Peak-Verhältnis kombinieren
    rms_peak_ratio = 20.0 * np.log10(rms / (peak + 1e-10)) if peak > 1e-10 else -20.0

    # SNR-Komponente [0, 1]: 30 dB → 1.0, 0 dB → 0.5
    snr_component = float(np.clip(snr_db / 60.0 + 0.5, 0.0, 1.0))

    # Dynamik-Komponente [0, 1]: -12 dB (gut) → 1.0, -2 dB (komprimiert) → 0.5
    dynamic_component = float(np.clip((rms_peak_ratio + 20.0) / 18.0, 0.0, 1.0))

    # Gewichtete Kombination
    score = float(np.clip(0.6 * snr_component + 0.4 * dynamic_component, 0.0, 1.0))

    return score, rms_dbfs, peak_dbfs, snr_db


class SegmentQualityScorer:
    """§v10.101 Segment-weise Qualitätsbewertung mit gleitenden Fenstern.

    Misst Qualität in überlappenden Segmenten (5s Fenster, 2.5s Hop).
    Segmente < Threshold können isoliert neu restauriert werden.

    Invarianten:
        - Segmente überlappen sich um 50% für kontinuierliche Abdeckung.
        - Score [0, 1] — höher = besser.
        - Lokale Qualitätsspitzen/-täler sind relevanter als globaler Mittelwert.
    """

    def __init__(self, window_seconds: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._window_s = max(1.0, min(window_seconds, 30.0))  # 1–30 s erlaubt

    @property
    def window_seconds(self) -> float:
        """Fensterlänge in Sekunden."""
        return self._window_s

    def score(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[SegmentScore]:
        """Bewertet die Qualität segment-weise mit gleitenden Fenstern.

        Args:
            audio: Audio-Signal (float32/float64, Mono oder Stereo).
            sr: Sample-Rate in Hz.

        Returns:
            Liste von SegmentScore für jedes Fenster.
        """
        with self._lock:
            # NaN/Inf-Schutz (§0a)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            win_len = int(self._window_s * sr)
            hop_len = win_len // 2  # 50% Überlappung

            if len(audio) < win_len and audio.ndim == 1:
                logger.debug(
                    "§v10.101 Segment-Scorer: Audio zu kurz (%.1f s < %.1f s Fenster)",
                    len(audio) / sr,
                    self._window_s,
                )
                return []

            scores = []
            # n_samples: letzte Dimension (samples), egal ob Mono (N,) oder Stereo (2,N)/(N,2)
            n_samples = audio.shape[-1]

            for start in range(0, n_samples - win_len + 1, hop_len):
                end = start + win_len
                segment = audio[start:end] if audio.ndim == 1 else audio[:, start:end]

                score_val, rms_dbfs, peak_dbfs, snr_db = _compute_segment_score(segment, sr)

                scores.append(SegmentScore(
                    start_sample=start,
                    end_sample=end,
                    start_time_s=start / sr,
                    duration_s=self._window_s,
                    score=score_val,
                    rms_dbfs=rms_dbfs,
                    peak_dbfs=peak_dbfs,
                    snr_estimate_db=snr_db,
                ))

            if scores:
                mean_score = float(np.mean([s.score for s in scores]))
                min_score = float(np.min([s.score for s in scores]))
                logger.debug(
                    "§v10.101 Segment-Scorer: %d Segmente, mean=%.3f, min=%.3f",
                    len(scores),
                    mean_score,
                    min_score,
                )

            return scores

    def get_bad_segments(
        self,
        audio: np.ndarray,
        sr: int,
        threshold: float = 0.7,
    ) -> list[SegmentScore]:
        """Gibt Segmente zurück, die unterhalb des Thresholds liegen.

        Diese Segmente können isoliert neu restauriert werden mit angepasster
        Phasen-Konfiguration (reduzierte Stärke, andere Phasen-Auswahl).

        Args:
            audio: Audio-Signal.
            sr: Sample-Rate in Hz.
            threshold: Mindest-Score [0, 1]. Segmente < threshold sind „schlecht".

        Returns:
            Liste der schlechten Segmente (sortiert nach Score aufsteigend).
        """
        all_scores = self.score(audio, sr)
        bad = [s for s in all_scores if s.score < threshold]
        bad.sort(key=lambda s: s.score)

        if bad:
            logger.info(
                "§v10.101 Segment-Scorer: %d schlechte Segmente (< %.2f) — isolierte Restaurierung empfohlen",
                len(bad),
                threshold,
            )

        return bad


# ── Thread-safe Singleton ────────────────────────────────────────────────

_scorer_instance: SegmentQualityScorer | None = None
_scorer_lock = threading.Lock()


def get_segment_quality_scorer(window_seconds: float = 5.0) -> SegmentQualityScorer:
    """Singleton-Zugriff auf den segment-weisen Qualitäts-Scorer."""
    global _scorer_instance  # pylint: disable=global-statement
    if _scorer_instance is None:
        with _scorer_lock:
            if _scorer_instance is None:
                _scorer_instance = SegmentQualityScorer(window_seconds=window_seconds)
    return _scorer_instance
