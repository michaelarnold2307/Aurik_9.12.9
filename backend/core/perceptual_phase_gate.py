"""
§v10.101 Perzeptuelles Phase-Gate — Aurik 10

Zweck: Jede Phase prüft vor dem Commit, ob die Veränderung hörbar ist.
Nutzt Maskierungsschwelle (ISO 11172-3 Bark-Skala) als harten Gate.

„Ist der Unterschied hörbar?" → Wenn nein, Phase überspringen.

Implementiert nach Spec 01 §v10.101.2 JND-basierte Goal-Relevanz.

Usage:
    from backend.core.perceptual_phase_gate import PerceptualPhaseGate

    gate = PerceptualPhaseGate()
    decision = gate.evaluate(pre_audio, post_audio, sr=48000)
    if not decision.is_audible:
        # Phase überspringen — Unterschied nicht hörbar
        ...
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Bark-Skala Konstanten (ISO 11172-3) ────────────────────────────────
# 24 kritische Bänder bis 22 kHz — für Maskierungsschwelle-Berechnung

_BARK_BAND_EDGES_HZ: list[float] = [
    0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 630.0, 770.0, 920.0,
    1080.0, 1270.0, 1500.0, 1750.0, 2050.0, 2400.0, 2800.0, 3300.0,
    3900.0, 4600.0, 5500.0, 6700.0, 8200.0, 10100.0, 12700.0, 16000.0
]

# JND pro Bark-Band (psychoakustisch kalibriert nach Zwicker & Fastl 1999)
_JND_PER_BAND_DB: list[float] = [
    3.0, 2.5, 2.0, 1.8, 1.5, 1.3, 1.2, 1.0, 0.9,
    0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4,
    0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2
]


@dataclass
class PerceptualGateDecision:
    """Entscheidung des perzeptuellen Phase-Gates.

    Attributes:
        is_audible: True wenn die Veränderung in ≥2 Bark-Bändern die JND überschreitet.
        audible_bands: Liste der Bark-Bänder, in denen die Veränderung hörbar ist.
        max_delta_db: Maximale Abweichung über alle Bänder (in dB).
        mean_delta_db: Mittlere Abweichung über alle Bänder (in dB).
        bands_above_jnd: Anzahl der Bänder, die JND überschreiten.
    """

    is_audible: bool
    audible_bands: list[int]
    max_delta_db: float
    mean_delta_db: float
    bands_above_jnd: int


def _compute_bark_energy(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 4096,
) -> np.ndarray:
    """Berechnet Energie pro Bark-Band.

    Args:
        audio: Mono-Audio (float32/float64).
        sr: Sample-Rate in Hz.
        n_fft: FFT-Größe.

    Returns:
        Energie pro Bark-Band (dBFS, Länge 24).
    """
    # Mono
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    n = len(mono)
    if n < n_fft:
        mono = np.pad(mono, (0, n_fft - n), mode="edge")

    # FFT
    spectrum = np.abs(np.fft.rfft(mono[:n_fft])) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Energie pro Bark-Band summieren
    bark_energy = np.zeros(len(_BARK_BAND_EDGES_HZ) - 1, dtype=np.float64)
    for band_idx in range(len(bark_energy)):
        f_lo = _BARK_BAND_EDGES_HZ[band_idx]
        f_hi = _BARK_BAND_EDGES_HZ[band_idx + 1]
        mask = (freqs >= f_lo) & (freqs < f_hi)
        bark_energy[band_idx] = np.sum(spectrum[mask])

    # Nach dBFS konvertieren (mit Floor gegen Division durch Null)
    return 10.0 * np.log10(bark_energy + 1e-20)


def _check_masking_threshold(
    pre_bark: np.ndarray,
    post_bark: np.ndarray,
) -> PerceptualGateDecision:
    """Prüft, ob die Veränderung pro Bark-Band die Maskierungsschwelle (JND) überschreitet.

    §v10.101.2: Ein Goal wird nur dann als „verletzt" gewertet, wenn die Abweichung
    vom Target in ≥2 Bark-Bändern die JND überschreitet.

    Args:
        pre_bark: Energie pro Bark-Band VOR Phase (dBFS).
        post_bark: Energie pro Bark-Band NACH Phase (dBFS).

    Returns:
        PerceptualGateDecision mit Hörbarkeits-Entscheidung.
    """
    delta = np.abs(post_bark - pre_bark)
    jnd_thresholds = np.array(_JND_PER_BAND_DB, dtype=np.float64)

    # Finde Bänder, die JND überschreiten
    above_jnd = delta > jnd_thresholds
    audible_bands = list(np.where(above_jnd)[0])

    max_delta = float(np.max(delta)) if len(delta) > 0 else 0.0
    mean_delta = float(np.mean(delta)) if len(delta) > 0 else 0.0

    # §v10.101.2: Hörbar wenn ≥2 Bänder JND überschreiten
    is_audible = len(audible_bands) >= 2

    return PerceptualGateDecision(
        is_audible=is_audible,
        audible_bands=audible_bands,
        max_delta_db=max_delta,
        mean_delta_db=mean_delta,
        bands_above_jnd=len(audible_bands),
    )


class PerceptualPhaseGate:
    """§v10.101 Perzeptuelles Phase-Gate — „Ist der Unterschied hörbar?"

    Nutzt Bark-Skala (ISO 11172-3) mit JND pro Band als Maskierungsschwelle.
    Wenn die Veränderung in <2 Bark-Bändern die JND überschreitet → Phase überspringen.

    Invarianten:
        - Nur Dämpfung (`scalar <= 1.0`), kein Boost (§1.4b).
        - Bei `is_audible=False` MUSS die Phase übersprungen werden.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def evaluate(
        self,
        pre_audio: np.ndarray,
        post_audio: np.ndarray,
        sr: int,
        n_fft: int = 4096,
    ) -> PerceptualGateDecision:
        """Bewertet, ob die Veränderung einer Phase hörbar ist.

        Args:
            pre_audio: Audio VOR der Phase (float32).
            post_audio: Audio NACH der Phase.
            sr: Sample-Rate in Hz.
            n_fft: FFT-Größe für Analyse.

        Returns:
            PerceptualGateDecision mit Hörbarkeits-Entscheidung.
        """
        with self._lock:
            # NaN/Inf-Schutz (§0a)
            pre = np.nan_to_num(pre_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            post = np.nan_to_num(post_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # Bark-Energie vor/nach Phase
            pre_bark = _compute_bark_energy(pre, sr, n_fft)
            post_bark = _compute_bark_energy(post, sr, n_fft)

            decision = _check_masking_threshold(pre_bark, post_bark)

            if not decision.is_audible:
                logger.debug(
                    "§v10.101 Perzeptuelles Gate: NICHT hörbar — max_delta=%.2f dB, bands_above_jnd=%d",
                    decision.max_delta_db,
                    decision.bands_above_jnd,
                )
            else:
                logger.debug(
                    "§v10.101 Perzeptuelles Gate: HÖRBAR — bands %s, max_delta=%.2f dB",
                    decision.audible_bands,
                    decision.max_delta_db,
                )

            return decision


# ── Thread-safe Singleton ────────────────────────────────────────────────

_gate_instance: PerceptualPhaseGate | None = None
_gate_lock = threading.Lock()


def get_perceptual_phase_gate() -> PerceptualPhaseGate:
    """Singleton-Zugriff auf das perzeptuelle Phase-Gate."""
    global _gate_instance  # pylint: disable=global-statement
    if _gate_instance is None:
        with _gate_lock:
            if _gate_instance is None:
                _gate_instance = PerceptualPhaseGate()
    return _gate_instance
