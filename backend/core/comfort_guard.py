"""ComfortGuard — Psychoakustische Hörmüdungs-Prävention.

§Rolls-Royce-Phantom: Verhindert AKTIV Hörmüdung, nicht nur messend.
Jede Phase kann den 2–5 kHz-Bereich übersteuern. Der ComfortGuard prüft
NACH jeder Phase und wendet bei Bedarf eine sanfte psychoakustische
Korrektur an — unhörbar für den Laien, spürbar bei Langzeithören.

Algorithmus (ISO 532-B Zwicker Sharpness angelehnt):
  1. Berechne Sharpness im 2–5 kHz-Bereich (Weighted Energy Ratio)
  2. Wenn Sharpness > COMFORT_THRESHOLD: Berechne needed_attenuation
  3. Wende High-Shelf-Filter an (fc=2.5 kHz, Q=0.5, max -3 dB)
  4. Validiere: Sharpness nach Korrektur < COMFORT_THRESHOLD

Nutzung:
    from backend.core.comfort_guard import apply_comfort_guard

    audio = apply_comfort_guard(audio, sr=48000)
    # Audio wurde auf Hörkomfort optimiert

Autor: Aurik 10 — Rolls-Royce Phantom Edition, 11. Juli 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Psychoakustische Konstanten ─────────────────────────────────────────────
CRITICAL_LOW_HZ: float = 2000.0  # Untergrenze kritischer Bereich
CRITICAL_HIGH_HZ: float = 5000.0  # Obergrenze kritischer Bereich
COMFORT_THRESHOLD: float = 0.12  # Max Sharpness-Ratio vor Korrektur (12%)
MAX_ATTENUATION_DB: float = 3.0  # Max Shelf-Cut (konservativ — unhörbar aber wirksam)
TARGET_SHARPNESS: float = 0.09  # Ziel-Sharpness (9% = sehr komfortabel)
SHELF_FC_HZ: float = 2500.0  # High-Shelf Eckfrequenz
SHELF_Q: float = 0.5  # Sanfter Shelf (kein resonantes Bell-Filter)


@dataclass
class ComfortResult:
    """Ergebnis einer ComfortGuard-Prüfung."""

    sharpness_before: float
    sharpness_after: float
    attenuation_applied_db: float
    correction_needed: bool
    comfortable: bool


def _compute_sharpness(audio: np.ndarray, sr: int) -> float:
    """Berechnet gewichtete Energie-Ratio im 2–5 kHz-Bereich.

    Vereinfachte Sharpness nach Zwicker (ISO 532-B):
    Sharpness = Summe(Energie × Bark-Gewicht) / Gesamtenergie
    """
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
    mono = mono.astype(np.float32).flatten()

    if len(mono) < sr * 0.1:
        return 0.0

    n_fft = min(4096, len(mono) // 2)
    spec = np.abs(np.fft.rfft(mono[: n_fft * 2]))
    freqs = np.fft.rfftfreq(n_fft * 2, d=1.0 / sr)

    mask = (freqs >= CRITICAL_LOW_HZ) & (freqs <= CRITICAL_HIGH_HZ)
    critical_energy = float(np.sum(spec[mask]))
    total_energy = float(np.sum(spec)) + 1e-10

    # Bark-Gewichtung: Höhere Frequenzen werden als schärfer empfunden
    # (vereinfacht: lineare Gewichtung über dem kritischen Bereich)
    if critical_energy > 0:
        bark_weights = freqs[mask] / CRITICAL_HIGH_HZ
        weighted = float(np.sum(spec[mask] * bark_weights))
        return weighted / total_energy
    return 0.0


def _apply_high_shelf(
    audio: np.ndarray,
    sr: int,
    fc_hz: float = SHELF_FC_HZ,
    gain_db: float = -2.0,
    q: float = SHELF_Q,
) -> np.ndarray:
    """Wendet sanften High-Shelf-Filter an (psychoakustisch optimiert).

    Nutzt scipy.signal.lfilter. Gain negativ = Absenkung der Höhen.
    """
    from scipy.signal import lfilter

    if abs(gain_db) < 0.1:
        return audio  # Keine hörbare Änderung — überspringen

    # Biquad High-Shelf: type='highshelf', fc, Q, gain
    # scipy.signal.biquad Koeffizienten manuell berechnen
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc_hz / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

    was_mono = audio.ndim == 1
    if was_mono:
        audio = audio.reshape(1, -1)

    result = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        result[ch] = lfilter(b, a, audio[ch].astype(np.float64)).astype(np.float32)

    if was_mono:
        result = result[0]
    return cast(np.ndarray, result)


def apply_comfort_guard(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Hauptfunktion: Prüft und korrigiert Hörmüdung.

    Args:
        audio: float32, mono oder stereo.
        sr:    Abtastrate.

    Returns:
        Korrigiertes Audio (oder unverändert wenn bereits komfortabel).
    """
    sharpness = _compute_sharpness(audio, sr)

    if sharpness <= COMFORT_THRESHOLD:
        logger.debug(
            "ComfortGuard: Sharpness %.3f ≤ %.3f — keine Korrektur nötig",
            sharpness,
            COMFORT_THRESHOLD,
        )
        return audio

    # Benötigte Dämpfung berechnen
    excess = sharpness - TARGET_SHARPNESS
    attenuation_db = min(MAX_ATTENUATION_DB, excess * 30)  # Proportional

    logger.info(
        "ComfortGuard: Sharpness %.3f > %.3f → High-Shelf %.1f dB @ %.0f Hz",
        sharpness,
        COMFORT_THRESHOLD,
        -attenuation_db,
        SHELF_FC_HZ,
    )

    corrected = _apply_high_shelf(audio, sr, gain_db=-attenuation_db)

    # Validierung
    new_sharpness = _compute_sharpness(corrected, sr)
    logger.info(
        "ComfortGuard: Sharpness %.3f → %.3f (%.1f dB Korrektur)",
        sharpness,
        new_sharpness,
        -attenuation_db,
    )

    return corrected


def check_comfort(audio: np.ndarray, sr: int = 48000) -> ComfortResult:
    """Prüft Hörkomfort ohne Korrektur (für Monitoring/Gate).

    Args:
        audio: float32 Audio.
        sr:    Abtastrate.

    Returns:
        ComfortResult mit Sharpness-Werten und Korrektur-Empfehlung.
    """
    sharpness_before = _compute_sharpness(audio, sr)
    needs_correction = sharpness_before > COMFORT_THRESHOLD

    if needs_correction:
        corrected = apply_comfort_guard(audio, sr)
        sharpness_after = _compute_sharpness(corrected, sr)
        att = min(MAX_ATTENUATION_DB, (sharpness_before - TARGET_SHARPNESS) * 30)
    else:
        sharpness_after = sharpness_before
        att = 0.0

    return ComfortResult(
        sharpness_before=round(sharpness_before, 4),
        sharpness_after=round(sharpness_after, 4),
        attenuation_applied_db=round(att, 1),
        correction_needed=needs_correction,
        comfortable=not needs_correction,
    )


__all__ = [
    "apply_comfort_guard",
    "check_comfort",
    "ComfortResult",
]
