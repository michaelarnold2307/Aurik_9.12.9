"""
Listener-in-the-Loop Feedback-Schleife — Aurik 10

Zweck: Kein automatischer Score ersetzt das menschliche Gehör.
Ermöglicht segmentweise A/B-Bewertung durch den Hörer. Segmente mit < 6
werden mit reduzierter Phasendichte neu restauriert.

Usage:
    from backend.core.listener_feedback_loop import ListenerFeedbackLoop

    feedback = ListenerFeedbackLoop()
    # Nach Restaurierung: Segment-Präsentation und Bewertung
    segments = feedback.get_segments_for_review(audio, sr=48000)
    # User bewertet jedes Segment 0–10
    for seg in segments:
        score = get_user_input(seg)  # z. B. über UI
        feedback.record_feedback(seg.segment_id, score)

    # Segmente mit < 6 neu restaurieren
    bad_segments = feedback.get_low_score_segments(threshold=6.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Segment-Datenstruktur für Review ────────────────────────────────────


@dataclass
class ReviewSegment:
    """Segment zur manuellen Bewertung.

    Attributes:
        segment_id: Eindeutige ID des Segments.
        start_sample: Start-Sample.
        end_sample: Ende-Sample.
        start_time_s: Startzeit in Sekunden.
        duration_s: Dauer in Sekunden.
        audio_restored: Restauriertes Segment (float32).
        audio_original: Original-Segment (float32) für A/B-Vergleich.
        user_score: Benutzerbewertung [0, 10] (None = noch nicht bewertet).
    """

    segment_id: str
    start_sample: int
    end_sample: int
    start_time_s: float
    duration_s: float
    audio_restored: np.ndarray
    audio_original: np.ndarray
    user_score: float | None


@dataclass
class FeedbackResult:
    """Ergebnis der Listener-Feedback-Schleife.

    Attributes:
        mean_score: Mittelwert aller Bewertungen [0, 10].
        n_segments: Anzahl der bewerteten Segmente.
        n_low_score: Anzahl der Segmente < threshold.
        needs_rerestoration: True wenn Segmente neu restauriert werden sollen.
    """

    mean_score: float
    n_segments: int
    n_low_score: int
    needs_rerestoration: bool


class ListenerFeedbackLoop:
    """Listener-in-the-Loop Feedback-Schleife — menschliches Gehör als finale Instanz.

    Teilt das restaurierte Audio in Segmente (5s), präsentiert sie zur Bewertung
    und speichert die Scores. Segmente < 6 werden markiert für neu Restaurierung.

    Invarianten:
        - User-Score [0, 10] — 0 = sehr schlecht, 10 = perfekt.
        - Segmente < 6 → neu restaurieren mit reduzierter Phasendichte.
        - Kein Segment wird stillschweigend ignoriert (alle müssen bewertet werden).
    """

    def __init__(self, segment_duration_s: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._segment_s = max(1.0, min(segment_duration_s, 30.0))
        self._segments: dict[str, ReviewSegment] = {}

    def get_segments_for_review(
        self,
        audio_restored: np.ndarray,
        audio_original: np.ndarray,
        sr: int,
        song_id: str | None = None,
    ) -> list[ReviewSegment]:
        """Teilt Audio in Segmente zur manuellen Bewertung.

        Args:
            audio_restored: Restauriertes Audio (float32).
            audio_original: Original-Audio für A/B-Vergleich.
            sr: Sample-Rate in Hz.
            song_id: Song-Identifikation (für Segment-ID).

        Returns:
            Liste von ReviewSegment zur Bewertung.
        """
        with self._lock:
            # NaN/Inf-Schutz (§0a)
            restored = np.nan_to_num(audio_restored, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            original = np.nan_to_num(audio_original, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            win_len = int(self._segment_s * sr)
            hop_len = win_len  # Keine Überlappung für Review-Segmente

            # n_samples: letzte Dimension (samples), egal ob Mono (N,) oder Stereo (2,N)/(N,2)
            n_samples = restored.shape[-1]

            segments: list[ReviewSegment] = []
            song_prefix = song_id or "unknown"

            for idx, start in enumerate(range(0, n_samples - win_len + 1, hop_len)):
                end = start + win_len
                seg_restored = restored[start:end] if restored.ndim == 1 else restored[:, start:end]
                seg_original = original[start:end] if original.ndim == 1 else original[:, start:end]

                segment_id = f"{song_prefix}_seg_{idx:04d}"

                review_seg = ReviewSegment(
                    segment_id=segment_id,
                    start_sample=start,
                    end_sample=end,
                    start_time_s=start / sr,
                    duration_s=self._segment_s,
                    audio_restored=seg_restored,
                    audio_original=seg_original,
                    user_score=None,
                )

                segments.append(review_seg)
                self._segments[segment_id] = review_seg

            logger.info(
                "Listener-Feedback: %d Segmente für Review (%.1f s je Segment)",
                len(segments),
                self._segment_s,
            )

            return segments

    def record_feedback(self, segment_id: str, user_score: float) -> None:
        """Speichert die Benutzerbewertung für ein Segment.

        Args:
            segment_id: ID des Segments.
            user_score: Bewertung [0, 10].
        """
        with self._lock:
            if segment_id not in self._segments:
                logger.warning("Listener-Feedback: Segment %s nicht gefunden", segment_id)
                return

            score = float(np.clip(user_score, 0.0, 10.0))
            self._segments[segment_id].user_score = score

            if score < 6.0:
                logger.info(
                    "Listener-Feedback: Segment %s Score=%.1f < 6.0 — neu Restaurierung empfohlen",
                    segment_id,
                    score,
                )

    def get_low_score_segments(self, threshold: float = 6.0) -> list[ReviewSegment]:
        """Gibt Segmente zurück, die unterhalb des Thresholds liegen.

        Diese Segmente können isoliert neu restauriert werden mit reduzierter
        Phasen-Konfiguration (weniger aggressive NR, geringere Kompression).

        Args:
            threshold: Mindest-Score [0, 10]. Segmente < threshold sind „schlecht".

        Returns:
            Liste der schlechten Segmente (sortiert nach Score aufsteigend).
        """
        with self._lock:
            bad = [
                seg for seg in self._segments.values()
                if seg.user_score is not None and seg.user_score < threshold
            ]
            bad.sort(key=lambda s: s.user_score or 10.0)

            if bad:
                logger.info(
                    "Listener-Feedback: %d Segmente < %.1f — neu Restaurierung empfohlen",
                    len(bad),
                    threshold,
                )

            return bad

    def get_feedback_result(self, threshold: float = 6.0) -> FeedbackResult:
        """Gibt das Gesamtergebnis der Listener-Feedback-Schleife zurück.

        Args:
            threshold: Mindest-Score [0, 10].

        Returns:
            FeedbackResult mit Mittelwert und Segment-Anzahlen.
        """
        with self._lock:
            scored = [seg for seg in self._segments.values() if seg.user_score is not None]

            if not scored:
                return FeedbackResult(
                    mean_score=0.0,
                    n_segments=0,
                    n_low_score=0,
                    needs_rerestoration=False,
                )

            scores = [seg.user_score for seg in scored]
            mean_score = float(np.mean(scores))
            n_low = sum(1 for s in scores if s < threshold)

            return FeedbackResult(
                mean_score=mean_score,
                n_segments=len(scored),
                n_low_score=n_low,
                needs_rerestoration=n_low > 0,
            )


# ── Thread-safe Singleton ────────────────────────────────────────────────

_feedback_instance: ListenerFeedbackLoop | None = None
_feedback_lock = threading.Lock()


def get_listener_feedback_loop(segment_duration_s: float = 5.0) -> ListenerFeedbackLoop:
    """Singleton-Zugriff auf die Listener-Feedback-Schleife."""
    global _feedback_instance  # pylint: disable=global-statement
    if _feedback_instance is None:
        with _feedback_lock:
            if _feedback_instance is None:
                _feedback_instance = ListenerFeedbackLoop(segment_duration_s=segment_duration_s)
    return _feedback_instance
