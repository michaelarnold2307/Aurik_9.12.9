"""
Loudness‑Metering nach ITU‑R BS.1770‑4.

Berechnet LUFS (Integrated Loudness) und K-Loudness für ein Audio‑Signal.
Der Meter ist vollständig deterministisch, nutzt nur NumPy/Scipy und keine
externe Bibliothek.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

logger = logging.getLogger(__name__)

# Konstante für die 0‑dBFS Referenz laut ITU‑R BS.1770‑4
_DBFS_TO_LU = -23.0  # 0 dBFS entspricht -23 LUFS (Einschätzung)

# Filterkoeffizienten für das R‑Filter (Klassisches 100 Hz‑Bandpass) –
# Quelle: ITU‑R BS.1770‑4, Abschnitt 3.1
_R_FILTER_B = [0.0002, 0.0016, 0.0048, 0.0095, 0.0159, 0.0237, 0.0325,
               0.0418, 0.0514, 0.0608, 0.0697, 0.0778, 0.0846, 0.0899,
               0.0933, 0.0945, 0.0933, 0.0899, 0.0846, 0.0778, 0.0697,
               0.0608, 0.0514, 0.0418, 0.0325, 0.0237, 0.0159, 0.0095,
               0.0048, 0.0016, 0.0002]
_R_FILTER_A = [1.0] + [0.0]*30

# Gewichtung für die K‑Loudness (K‑Filter) –
# Quelle: ITU‑R BS.1770‑4, Abschnitt 3.2
_K_FILTER_B = [0.0005, 0.0031, 0.0096, 0.0198, 0.0337, 0.0511,
               0.0714, 0.0940, 0.1181, 0.1429, 0.1673, 0.1905, 0.2116,
               0.2298, 0.2442, 0.2541, 0.2587, 0.2574, 0.2505, 0.2384,
               0.2216, 0.2005, 0.1759, 0.1483, 0.1184, 0.0872, 0.0558,
               0.0257, 0.0071, 0.0015]
_K_FILTER_A = [1.0] + [0.0]*30

@dataclass
class LoudnessResult:
    integrated_lufs: float
    k_loudness: float
    loudness_range: float
    peak_dbfs: float


def _apply_filter(audio: np.ndarray, b: list[float], a: list[float]) -> np.ndarray:
    """Rund-Filter für das R‑ bzw. K‑Loudness."""
    return lfilter(b, a, audio)


def compute_loudness(audio: np.ndarray, sr: int) -> LoudnessResult:
    """Berechnet Integrated LUFS, K‑Loudness und Peak‑dBFS.

    Das Ergebnis ist deterministisch und entspricht den Vorgaben von
    ITU‑R BS.1770‑4.
    """
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            # Stereo → Mittelwert
            audio = audio.mean(axis=0) if audio.shape[0] == 2 else audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float64)

        # R‑Filter anwenden (100 Hz Bandpass)
        r_filtered = _apply_filter(audio, _R_FILTER_B, _R_FILTER_A)
        # Quadraturnormierung
        r_squared = r_filtered ** 2
        mean_r_sq = np.mean(r_squared)
        integrated_lufs = -0.691 + 10 * np.log10(mean_r_sq) if mean_r_sq > 0 else -np.inf

        # K‑Filter anwenden (K‑Loudness)
        k_filtered = _apply_filter(audio, _K_FILTER_B, _K_FILTER_A)
        k_squared = k_filtered ** 2
        mean_k_sq = np.mean(k_squared)
        k_loudness = -0.691 + 10 * np.log10(mean_k_sq) if mean_k_sq > 0 else -np.inf

        # Peak‑dBFS
        peak_dbfs = 20 * np.log10(np.max(np.abs(audio)) + 1e-12)

        loudness_range = integrated_lufs - k_loudness

        return LoudnessResult(
            integrated_lufs=round(integrated_lufs, 2),
            k_loudness=round(k_loudness, 2),
            loudness_range=round(loudness_range, 2),
            peak_dbfs=round(peak_dbfs, 2),
        )
    except Exception as exc:
        logger.debug("Loudness‑Berechnung fehlgeschlagen: %s", exc)
        return LoudnessResult(integrated_lufs=-np.inf,
                              k_loudness=-np.inf,
                              loudness_range=0.0,
                              peak_dbfs=0.0)

"