"""
§11 Cross-Phase Consensus — Aurik 10

Zweck: Detektiert Interferenzen zwischen aufeinanderfolgenden Phasen.
Zwei Phasen können sich gegenseitig aufheben (z. B. Click-Removal erzeugt
Transienten, die Declipper als Klippen interpretiert).

Nach jeder Phase wird ein kurzer Spektral-Vergleich mit dem Vorzustand
durchgeführt. Wenn neue Peaks > -60 dBFS erscheinen → Interferenz-Flag.

Usage:
    from backend.core.phase_interaction_denker import get_phase_interaction_denker

    denker = get_phase_interaction_denker()
    result = denker.check_interference(pre_audio, post_audio, sr=48000)
    if result.has_interference:
        # Phasenstärke reduzieren oder Phase überspringen
        ...
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Schwellwerte (psychoakustisch kalibriert) ────────────────────────────

# Maximale erlaubte neue Peak-Energie in dBFS (§11 Cross-Phase Consensus)
_NEW_PEAK_THRESHOLD_DBFS = -60.0

# Minimale Audiolänge für Analyse (ms)
_MIN_AUDIO_MS = 200


@dataclass
class PhaseInteractionResult:
    """Ergebnis der Phasen-Interferenz-Detektion.

    Attributes:
        has_interference: True wenn neue Peaks > -60 dBFS erscheinen.
        new_peaks_dbfs: Liste der neuen Peak-Energien in dBFS.
        n_new_peaks: Anzahl der neuen Peaks.
        max_new_peak_dbfs: Maximale neue Peak-Energie in dBFS.
    """

    has_interference: bool
    new_peaks_dbfs: list[float]
    n_new_peaks: int
    max_new_peak_dbfs: float


def _detect_new_spectral_peaks(
    pre_audio: np.ndarray,
    post_audio: np.ndarray,
    sr: int,
    threshold_dbfs: float = -60.0,
) -> list[float]:
    """Detektiert neue spektrale Peaks nach einer Phase.

    Vergleicht das Spektrum vor/nach Phase und identifiziert Frequenzbänder,
    in denen die Energie um > 10 dB angestiegen ist (und über dem Threshold).

    Args:
        pre_audio: Audio VOR der Phase.
        post_audio: Audio NACH der Phase.
        sr: Sample-Rate in Hz.
        threshold_dbfs: Mindest-Energie für Peak-Detektion in dBFS.

    Returns:
        Liste der neuen Peak-Energien in dBFS (über Threshold).
    """
    # NaN/Inf-Schutz (§0a)
    pre = np.nan_to_num(pre_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    post = np.nan_to_num(post_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # Mono
    if pre.ndim == 2:
        pre_mono = np.mean(pre, axis=0)
    else:
        pre_mono = pre

    if post.ndim == 2:
        post_mono = np.mean(post, axis=0)
    else:
        post_mono = post

    n = min(len(pre_mono), len(post_mono))
    pre_seg = pre_mono[:n]
    post_seg = post_mono[:n]

    # FFT-Analyse (4096 bins)
    n_fft = 4096
    if n < n_fft:
        return []

    # Log-Magnitude-Spektren
    window = np.hanning(n_fft)
    spec_pre = np.abs(np.fft.rfft(pre_seg[:n_fft] * window)) ** 2
    spec_post = np.abs(np.fft.rfft(post_seg[:n_fft] * window)) ** 2

    # Nach dBFS konvertieren
    db_pre = 10.0 * np.log10(spec_pre + 1e-20)
    db_post = 10.0 * np.log10(spec_post + 1e-20)

    # Energie-Anstieg > 10 dB und über Threshold
    delta = db_post - db_pre
    new_peak_mask = (delta > 10.0) & (db_post > threshold_dbfs)

    if not np.any(new_peak_mask):
        return []

    # Neue Peaks extrahieren
    new_peaks = list(db_post[new_peak_mask])
    new_peaks.sort(reverse=True)

    return [float(p) for p in new_peaks[:20]]  # Max 20 Peaks zurückgeben


class PhaseInteractionDenker:
    """§11 Cross-Phase Consensus — Phasen-Interferenz-Detektion.

    Nach jeder Phase wird ein Spektral-Vergleich durchgeführt. Wenn neue Peaks
    > -60 dBFS erscheinen → Interferenz-Flag. Verhindert, dass aufeinanderfolgende
    Phasen sich gegenseitig aufheben oder neue Artefakte produzieren.

    Invarianten:
        - Bei has_interference=True MUSS die Phase überprüft werden (Stärke reduzieren).
        - Neue Peaks > -40 dBFS → sofortiger Rollback empfohlen.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def check_interference(
        self,
        pre_audio: np.ndarray,
        post_audio: np.ndarray,
        sr: int,
        threshold_dbfs: float = -60.0,
    ) -> PhaseInteractionResult:
        """Prüft auf spektrale Interferenzen nach einer Phase.

        Args:
            pre_audio: Audio VOR der Phase (float32).
            post_audio: Audio NACH der Phase.
            sr: Sample-Rate in Hz.
            threshold_dbfs: Mindest-Energie für Peak-Detektion in dBFS.

        Returns:
            PhaseInteractionResult mit Interferenz-Flag und neuen Peaks.
        """
        with self._lock:
            # Länge-Check
            min_samples = int(_MIN_AUDIO_MS / 1000.0 * sr)
            if len(pre_audio) < min_samples or len(post_audio) < min_samples:
                return PhaseInteractionResult(
                    has_interference=False,
                    new_peaks_dbfs=[],
                    n_new_peaks=0,
                    max_new_peak_dbfs=-100.0,
                )

            new_peaks = _detect_new_spectral_peaks(pre_audio, post_audio, sr, threshold_dbfs)

            has_interference = len(new_peaks) > 0
            max_peak = float(max(new_peaks)) if new_peaks else -100.0

            result = PhaseInteractionResult(
                has_interference=has_interference,
                new_peaks_dbfs=new_peaks,
                n_new_peaks=len(new_peaks),
                max_new_peak_dbfs=max_peak,
            )

            if has_interference:
                severity = "CRITICAL" if max_peak > -40.0 else "WARNING"
                logger.info(
                    "§11 Cross-Phase Consensus: %s — %d neue Peaks, max=%.1f dBFS",
                    severity,
                    len(new_peaks),
                    max_peak,
                )

            return result


# ── Thread-safe Singleton ────────────────────────────────────────────────

_denker_instance: PhaseInteractionDenker | None = None
_denker_lock = threading.Lock()


def get_phase_interaction_denker() -> PhaseInteractionDenker:
    """Singleton-Zugriff auf den Phasen-Interferenz-Denker."""
    global _denker_instance  # pylint: disable=global-statement
    if _denker_instance is None:
        with _denker_lock:
            if _denker_instance is None:
                _denker_instance = PhaseInteractionDenker()
    return _denker_instance
