"""
§0p Dynamik-Erhaltungs-Guard — Aurik 10

Zweck: Verhindert, dass Kompressions-/Limiting-Phasen die natürliche Dynamik
übermäßig reduzieren. Das menschliche Ohr empfindet Dynamik als „Lebendigkeit".

Misst RMS/Peak-Verhältnis pro Song. Wenn Kompression das Verhältnis um > 3 dB
reduziert → Phase skippen oder Stärke halbieren.

Usage:
    from backend.core.dynamic_preservation_guard import DynamicPreservationGuard

    guard = DynamicPreservationGuard()
    decision = guard.evaluate(pre_audio, post_audio, sr=48000)
    if decision.rollback:
        # Phase skippen oder Stärke halbieren
        ...
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Schwellwerte (psychoakustisch kalibriert) ────────────────────────────

# Maximale erlaubte RMS/Peak-Reduktion in dB (§0p)
_MAX_RMS_PEAK_REDUCTION_DB = 3.0

# Minimale Audiolänge für Analyse (ms)
_MIN_AUDIO_MS = 500


@dataclass
class DynamicPreservationDecision:
    """Entscheidung des Dynamik-Erhaltungs-Gates.

    Attributes:
        rollback: True wenn Phase zurückgerollt werden soll.
        strength_scalar: Multiplikator für Phasenstärke [0, 1].
        rms_peak_ratio_pre: RMS/Peak-Verhältnis vor der Phase (dB).
        rms_peak_ratio_post: RMS/Peak-Verhältnis nach der Phase (dB).
        delta_db: Reduktion des RMS/Peak-Verhältnisses (positiv = Dynamik verloren).
    """

    rollback: bool
    strength_scalar: float
    rms_peak_ratio_pre: float
    rms_peak_ratio_post: float
    delta_db: float


def _compute_rms_peak_ratio(audio: np.ndarray) -> float:
    """Berechnet RMS/Peak-Verhältnis in dB.

    Negativer Wert: je negativer, desto mehr Dynamik (Peak > RMS).
    Wert nahe 0: komprimiertes Signal (RMS ≈ Peak).

    Returns:
        RMS/Peak-Verhältnis in dB (negativ, z. B. -12 dB = gute Dynamik).
    """
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Mono
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.abs(mono).max())

    if peak < 1e-10 or rms < 1e-10:
        return -20.0  # Stille → konservativer Default

    return float(20.0 * np.log10(rms / peak))


class DynamicPreservationGuard:
    """§0p Dynamik-Erhaltungs-Guard — verhindert Überkompression.

    Misst RMS/Peak-Verhältnis vor/nach Phase. Wenn Reduktion > 3 dB → Rollback.

    Invarianten:
        - Bei rollback=True MUSS die Phase übersprungen oder auf 50% Stärke reduziert werden.
        - strength_scalar ∈ [0, 1] — nur Dämpfung, kein Boost (§1.4b).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def evaluate(
        self,
        pre_audio: np.ndarray,
        post_audio: np.ndarray,
        sr: int,
    ) -> DynamicPreservationDecision:
        """Bewertet die Dynamik-Erhaltung vor/nach einer Phase.

        Args:
            pre_audio: Audio VOR der Phase (float32).
            post_audio: Audio NACH der Phase.
            sr: Sample-Rate in Hz.

        Returns:
            DynamicPreservationDecision mit Rollback-Empfehlung.
        """
        with self._lock:
            # Länge-Check
            min_samples = int(_MIN_AUDIO_MS / 1000.0 * sr)
            if len(pre_audio) < min_samples or len(post_audio) < min_samples:
                return DynamicPreservationDecision(
                    rollback=False,
                    strength_scalar=1.0,
                    rms_peak_ratio_pre=-20.0,
                    rms_peak_ratio_post=-20.0,
                    delta_db=0.0,
                )

            # RMS/Peak-Verhältnis vor/nach Phase
            ratio_pre = _compute_rms_peak_ratio(pre_audio)
            ratio_post = _compute_rms_peak_ratio(post_audio)

            # Delta: positiv = Dynamik verloren (ratio_post näher an 0 als ratio_pre)
            # ratio_pre und ratio_post sind negativ; wenn ratio_post > ratio_pre → Dynamik verloren
            delta_db = float(np.clip(ratio_post - ratio_pre, 0.0, 20.0))

            rollback = delta_db > _MAX_RMS_PEAK_REDUCTION_DB

            # Stärke-Scalar: proportional zur Dynamik-Erhaltung
            strength_scalar = float(np.clip(1.0 - delta_db / 6.0, 0.0, 1.0))
            if rollback:
                strength_scalar *= 0.5  # §0p: Rollback → 50% Stärke

            decision = DynamicPreservationDecision(
                rollback=rollback,
                strength_scalar=strength_scalar,
                rms_peak_ratio_pre=ratio_pre,
                rms_peak_ratio_post=ratio_post,
                delta_db=delta_db,
            )

            if rollback:
                logger.info(
                    "§0p Dynamik-Erhaltungs-Guard: ROLLBACK — delta=%.2f dB > %.1f dB (RMS/Peak: %.1f → %.1f dB)",
                    delta_db,
                    _MAX_RMS_PEAK_REDUCTION_DB,
                    ratio_pre,
                    ratio_post,
                )

            return decision


# ── Thread-safe Singleton ────────────────────────────────────────────────

_guard_instance: DynamicPreservationGuard | None = None
_guard_lock = threading.Lock()


def get_dynamic_preservation_guard() -> DynamicPreservationGuard:
    """Singleton-Zugriff auf den Dynamik-Erhaltungs-Guard."""
    global _guard_instance  # pylint: disable=global-statement
    if _guard_instance is None:
        with _guard_lock:
            if _guard_instance is None:
                _guard_instance = DynamicPreservationGuard()
    return _guard_instance
