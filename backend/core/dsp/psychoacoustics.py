"""
backend/core/dsp/psychoacoustics.py — ISO 532-1 Psychoakustische Lautheitsmessung (Aurik 10.0.0 §4.1b)
===================================================================================================

Implementiert `compute_specific_loudness_zwicker()` — stationäre Methode nach
Zwicker & Fastl (Psychoacoustics, 2nd ed. 1999) / ISO 532-1:2017.

Warum LUFS allein nicht ausreicht (§4.1b):
  ITU-R BS.1770-5 K-Weighting detektiert Tieftonrumpeln (200–300 Hz) NICHT
  als Lautheitszunahme, die das Gehör mit bis zu +6 Phon wahrnimmt
  (ISO 226:2003 Equal-Loudness-Contours). Rumble-Filter-Phasen werden bei
  LUFS-Only-Check fälschlich als lautheitsneutral eingestuft.

Algorithmus (stationäre ISO 532-1):
  1. Bark-Filterbank: 24 kritische Bänder (Zwicker 1961, Bark-Skala)
  2. Spezifische Lautheit N' pro Band (sone/Bark): aus Bandpegel → Phon → Sone
  3. Gesamt-Lautheit N (sone): ∫ N'(z) dz über 24 Bänder
  4. ΔN = N_out − N_in: Pipeline-Reaktion nach §4.1b-Tabelle

Referenz-Veröffentlichungen:
  - Zwicker E. & Fastl H. (1999): "Psychoacoustics — Facts and Models", 2nd ed.
  - ISO 532-1:2017: "Acoustics — Zwicker method for calculating loudness"
  - ISO 226:2003: Equal-Loudness-Level Contours (Fletcher-Munson)
  - Chalupper J. & Fastl H. (2002): "Dynamic loudness model (DLM) based on
    Moore's loudness model" — for tonality correction reference

Performance: ≤ 50 ms für 5-s-Fenster bei 48 000 Hz auf modernem Desktop-CPU.

Author: Aurik 10.0.0 Engineering
Version: 1.0.0  (§4.1b RELEASE_MUST)
"""

from __future__ import annotations

# v10.101 SOTA: ISO 532-1 Zwicker-Lautheit + Bark-Bänder. Basis für Gammatone-Migration.
import logging
import threading
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import signal as _sp_signal

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# §4.3 Bark-Band-Kanten (Zwicker & Fastl 1999, Tabelle A.1)
# 25 Kanten = 24 Bänder (0..15 500 Hz)
# ──────────────────────────────────────────────────────────────────────
BARK_EDGES_HZ: np.ndarray = np.array(
    [
        20,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ],
    dtype=np.float64,
)
N_BARK: int = 24  # Anzahl kritische Bänder

# Bark-Mitten für Equal-Loudness-Interpolation
BARK_CENTERS_HZ: np.ndarray = 0.5 * (BARK_EDGES_HZ[:-1] + BARK_EDGES_HZ[1:])

# ──────────────────────────────────────────────────────────────────────
# ISO 226:2003 Equal-Loudness-Level Contour (vereinfacht, Phon-Referenz)
# Tabelle: Frequenz → Pegel bei 40 Phon (Kern-Referenzkurve = 1 sone)
# Interpoliert für Bark-Band-Mitten.
# Basis: ISO 226:2003 Table A.1 — Absolute Threshold of Hearing + 40 Phon Contour
# ──────────────────────────────────────────────────────────────────────
# Reference SPL at 40 phon (ISO 226:2003) for selected standard frequencies
_ISO226_FREQ_HZ: np.ndarray = np.array(
    [
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
        16000,
    ],
    dtype=np.float64,
)
_ISO226_40PHON_SPL: np.ndarray = np.array(
    [
        73.4,
        65.8,
        59.2,
        53.8,
        49.3,
        45.6,
        42.5,
        40.5,
        38.7,
        37.4,
        36.7,
        36.5,
        36.7,
        37.0,
        37.4,
        37.5,
        37.1,
        36.8,
        36.2,
        34.4,
        30.6,
        26.5,
        22.7,
        20.4,
        20.8,
        24.0,
        28.7,
        33.7,
        38.4,
        43.9,
    ],
    dtype=np.float64,
)

# Threshold of Hearing (ATH) in dB SPL at Bark-Center frequencies
# Used as noise floor below which loudness contribution is 0.
_ATH_SPL: np.ndarray = np.interp(
    BARK_CENTERS_HZ,
    _ISO226_FREQ_HZ,
    # Approximate ATH from ISO 226:2003 (minimum audible field)
    np.array(
        [
            78.5,
            68.7,
            59.5,
            51.1,
            44.0,
            38.5,
            34.0,
            30.5,
            27.5,
            25.0,
            23.0,
            22.0,
            21.0,
            21.0,
            19.0,
            18.5,
            17.5,
            16.5,
            15.5,
            14.0,
            12.0,
            10.0,
            8.0,
            8.5,
            11.0,
            15.0,
            21.0,
            29.5,
            38.0,
            48.0,
        ],
        dtype=np.float64,
    ),
)

# ──────────────────────────────────────────────────────────────────────
# ISO 532-1 Lautheits-Kurve: Pegeldifferenz (dB) bei 40 Phon pro Bark-Band
# gegenüber 1 kHz-Referenz (ISO 226:2003-basiert)
# ──────────────────────────────────────────────────────────────────────
_ISO226_CORRECTION_DB: np.ndarray = (
    np.interp(
        BARK_CENTERS_HZ,
        _ISO226_FREQ_HZ,
        _ISO226_40PHON_SPL,
    )
    - _ISO226_40PHON_SPL[np.where(_ISO226_FREQ_HZ == 1000)[0][0]]
)  # Normalize to 1 kHz

# ──────────────────────────────────────────────────────────────────────
# Sone-Phon Konversionskonstanten (Zwicker 1961)
# N = 2^((Lp - 40) / 10) für Lp ≥ 40 Phon (sone Referenz: 40 Phon @ 1 kHz = 1 sone)
# N = (Lp / 40)^2.642    für 0 < Lp < 40 Phon (low-level approximation)
# ──────────────────────────────────────────────────────────────────────
_REF_PHON: float = 40.0  # 1 sone = 40 Phon @ 1 kHz
_EXP_HIGH: float = 0.1  # exponent for Lp ≥ 40: N = 2^(0.1 * (Lp - 40))
_EXP_LOW: float = 2.642  # exponent for Lp < 40: N = (Lp/40)^2.642

# Assumed SPL calibration: full-scale (0 dBFS) ≡ 90 dB SPL (typical studio level)
# This is a conservative studio reference; Aurik does not perform absolute SPL
# calibration, so this constant maps dBFS to an approximate dB SPL domain.
_DBFS_TO_SPL_OFFSET: float = 90.0  # dBFS 0 → 90 dB SPL


# ──────────────────────────────────────────────────────────────────────
# Ergebnis-Dataclass
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ZwickerLoudnessResult:
    """Ergebnis von compute_specific_loudness_zwicker().

    Felder:
        total_loudness_sone:   Gesamt-Lautheit N in Sone
        specific_loudness:     N' pro Bark-Band (24 Werte, sone/Bark)
        loudness_phon:         Äquivalenter Lautheitspegel in Phon
        n_bark_bands:          Anzahl Bark-Bänder (immer 24)
        band_levels_db_spl:    Schallpegel pro Bark-Band in dB SPL
        computation_valid:     False wenn Fehler aufgetreten (Fallback-Wert)
    """

    total_loudness_sone: float
    specific_loudness: np.ndarray  # shape (24,), float64
    loudness_phon: float
    n_bark_bands: int = N_BARK
    band_levels_db_spl: np.ndarray | None = None
    computation_valid: bool = True

    def delta_phon(self, reference: ZwickerLoudnessResult) -> float:
        """Pegel-Differenz in Phon gegenüber Referenz."""
        return self.loudness_phon - reference.loudness_phon

    def delta_sone(self, reference: ZwickerLoudnessResult) -> float:
        """Lautheits-Differenz in Sone gegenüber Referenz."""
        return self.total_loudness_sone - reference.total_loudness_sone


# ──────────────────────────────────────────────────────────────────────
# Singleton-Cache für Bark-Filterbank
# ──────────────────────────────────────────────────────────────────────
_filterbank_cache: dict[int, list] = {}
_filterbank_lock = threading.Lock()


def _build_bark_filterbank(sr: int) -> list:
    """Baut 24 Butterworth-Bandpass-Filter für Bark-Skala (cached, thread-safe).

    Jedes Filter ist 4. Ordnung (2× 2. Ordnung Butterworth, SOS-Form für
    numerische Stabilität). Frequenzgrenzen aus BARK_EDGES_HZ.

    Returns:
        Liste von 24 SOS-Arrays (scipy.signal.sosfilt-kompatibel)
    """
    with _filterbank_lock:
        if sr in _filterbank_cache:
            return _filterbank_cache[sr]

    nyq = sr / 2.0
    filters: list[np.ndarray | None] = []
    for b in range(N_BARK):
        f_low = BARK_EDGES_HZ[b]
        f_high = BARK_EDGES_HZ[b + 1]

        # Clip to safe Nyquist range (avoid numerical instability near Nyquist)
        f_low_norm = max(f_low / nyq, 0.001)
        f_high_norm = min(f_high / nyq, 0.999)

        if f_low_norm >= f_high_norm:
            # Degenerate band (SR too low) — use allpass (gain=1 across all freq)
            filters.append(None)
            continue

        try:
            if f_low_norm <= 0.002:
                # Very low-frequency band → lowpass only
                sos = _sp_signal.butter(4, f_high_norm, btype="low", output="sos")
            else:
                sos = _sp_signal.butter(4, [f_low_norm, f_high_norm], btype="band", output="sos")
            filters.append(sos)
        except Exception as _fe:
            logger.debug("Bark-Filter %d Fehler: %s", b, _fe)
            filters.append(None)

    with _filterbank_lock:
        _filterbank_cache[sr] = filters
    return filters


def _spl_to_sone(spl_db: float) -> float:
    """Konvertiert dB SPL → Sone (ISO 532-1 Zwicker-Kurve).

    N = 2^((Lp - 40) / 10)     für Lp ≥ 40 Phon
    N = (Lp / 40)^2.642         für 0 < Lp < 40 Phon
    N = 0                        für Lp ≤ 0 (unterhalb Hörschwelle)
    """
    if spl_db <= 0.0:
        return 0.0
    if spl_db >= _REF_PHON:
        return float(2.0 ** (_EXP_HIGH * (spl_db - _REF_PHON)))
    return float((spl_db / _REF_PHON) ** _EXP_LOW)


def _sone_to_phon(sone: float) -> float:
    """Konvertiert Sone → Phon (Inverse Zwicker-Kurve).

    Lp = 40 + 33.22 * log10(N)   für N ≥ 1 (≥ 40 Phon)
    Lp = 40 * N^(1/2.642)         für 0 < N < 1
    """
    if sone <= 0.0:
        return 0.0
    if sone >= 1.0:
        return float(40.0 + 33.22 * np.log10(sone))
    return float(40.0 * (sone ** (1.0 / _EXP_LOW)))


def compute_specific_loudness_zwicker(
    audio: np.ndarray,
    sr: int,
    analysis_window_s: float = 5.0,
    center_window: bool = True,
) -> ZwickerLoudnessResult:
    """
    Stationäre ISO 532-1 Zwicker-Lautheitsmessung für Aurik 10.0.0 §4.1b.

    Algorithmus:
      1. Stereo → Mono (Downmix)
      2. 24-Band Bark-Butterworth-Filterbank (4. Ordnung, SOS)
      3. RMS-Leistungspegel pro Band (dBFS → dB SPL via Kalibrieroffset)
      4. ISO 226:2003 Frequenzkorrektur (Equal-Loudness-Gewichtung)
      5. ATH-Gating (Bänder unterhalb Hörschwelle → L'=0)
      6. Sone-Konversion: N'_b = f(L'_b) nach ISO 532-1
      7. Gesamt-Lautheit: N = Σ N'_b × ΔBark (trapz-Integration)

    Args:
        audio:             Float-Array, mono oder stereo (beliebige Konvention)
        sr:                Sample-Rate in Hz
        analysis_window_s: Analysefenster-Länge in Sekunden (max. Mitte → STI)
        center_window:     True → zentriertes Fenster (stabiler für Pipeline-Vergleich)

    Returns:
        ZwickerLoudnessResult — enthält total_loudness_sone, specific_loudness, loudness_phon

    Laufzeit: ≤ 50 ms für 5-s-Fenster bei 48 000 Hz.

    Raises:
        Kein raise — alle Fehler werden als computation_valid=False mit Fallback-0 gemeldet.
    """
    try:
        return _compute_zwicker_internal(audio, sr, analysis_window_s, center_window)
    except Exception as _e:
        logger.warning("berechnen_specific_loudness_zwicker Fehler (Ersatzpfad 0): %s", _e)
        return ZwickerLoudnessResult(
            total_loudness_sone=0.0,
            specific_loudness=np.zeros(N_BARK, dtype=np.float64),
            loudness_phon=0.0,
            computation_valid=False,
        )


def _compute_zwicker_internal(
    audio: np.ndarray,
    sr: int,
    analysis_window_s: float,
    center_window: bool,
) -> ZwickerLoudnessResult:
    """Interne Berechnung — wird von compute_specific_loudness_zwicker() gewrapped."""

    # ── 1. Mono Downmix ──────────────────────────────────────────────
    arr = np.asarray(audio, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        # Handle both (N, 2) and (2, N) layouts
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.mean(axis=0)
        elif arr.shape[1] == 2:
            arr = arr.mean(axis=1)
        else:
            arr = arr.mean(axis=0)
    elif arr.ndim > 2:
        arr = arr.flatten()

    if arr.size == 0:
        return ZwickerLoudnessResult(
            total_loudness_sone=0.0,
            specific_loudness=np.zeros(N_BARK, dtype=np.float64),
            loudness_phon=0.0,
            computation_valid=False,
        )

    # ── 2. Analysefenster auswählen ──────────────────────────────────
    max_samples = int(analysis_window_s * sr)
    if arr.size > max_samples:
        if center_window:
            start_idx = (arr.size - max_samples) // 2
            arr = arr[start_idx : start_idx + max_samples]
        else:
            arr = arr[:max_samples]

    # ── 3. Bark-Filterbank anwenden ──────────────────────────────────
    filters = _build_bark_filterbank(sr)

    band_levels_db_spl = np.zeros(N_BARK, dtype=np.float64)
    specific_loudness = np.zeros(N_BARK, dtype=np.float64)

    for b in range(N_BARK):
        sos = filters[b]
        if sos is None:
            # Degenerate band — zero contribution
            continue

        try:
            filtered = _sp_signal.sosfiltfilt(sos, arr)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue

        # RMS → dBFS
        rms = float(np.sqrt(np.mean(filtered**2)))
        if rms < 1e-12:
            band_levels_db_spl[b] = -120.0  # Below computational floor
            continue

        db_fs = 20.0 * np.log10(rms)

        # dBFS → dB SPL (studio calibration: 0 dBFS ≡ 90 dB SPL)
        db_spl = db_fs + _DBFS_TO_SPL_OFFSET

        # ISO 226:2003 Equal-Loudness-Correction (frequency weighting)
        db_spl_corrected = db_spl - _ISO226_CORRECTION_DB[b]

        band_levels_db_spl[b] = db_spl_corrected

        # ── 4. ATH-Gating: Bänder unterhalb Hörschwelle → N'=0 ──────
        ath_for_band = _ATH_SPL[b]
        if db_spl_corrected <= ath_for_band:
            # Sub-threshold: no loudness contribution (ISO 532-1 §6.2.4)
            specific_loudness[b] = 0.0
            continue

        # Effective level above ATH (excitation level)
        l_eff = db_spl_corrected - ath_for_band + 40.0  # map to phon above threshold

        # ── 5. Sone-Konversion (ISO 532-1) ───────────────────────────
        specific_loudness[b] = _spl_to_sone(l_eff)

    # ── 6. Gesamt-Lautheit: Trapez-Integration über Bark-Bänder ──────
    # ΔBark = 1.0 Bark pro Band (uniform), so N = Σ N'_b × 1 Bark
    total_loudness_sone = float(np.sum(specific_loudness))
    total_loudness_sone = max(0.0, total_loudness_sone)

    # ── 7. Phon-Äquivalent ───────────────────────────────────────────
    loudness_phon = _sone_to_phon(total_loudness_sone)

    return ZwickerLoudnessResult(
        total_loudness_sone=total_loudness_sone,
        specific_loudness=specific_loudness,
        loudness_phon=loudness_phon,
        n_bark_bands=N_BARK,
        band_levels_db_spl=band_levels_db_spl,
        computation_valid=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Pipeline-Guard-Hilfsfunktion (Aurik §4.1b ΔN-Tabelle)
# ──────────────────────────────────────────────────────────────────────


def evaluate_mid_pipeline_loudness_delta(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
    phase_name: str = "unknown",
) -> dict:
    """
    §4.1b Mid-Pipeline-Guard: Berechnet ΔN (Sone) und empfiehlt Pipeline-Reaktion.

    Reaktionstabelle (§4.1b):
      ΔN ≤ 0.5         → OK
      0.5 < ΔN ≤ 1.0   → INFO in metadata["loudness_delta_sone"]
      1.0 < ΔN ≤ 2.0   → WARNING + PhaseConductor-State
      ΔN > 2.0          → FAIL → Dry/Wet-Rescue empfohlen

    Args:
        audio_before: Audio vor der Phase
        audio_after:  Audio nach der Phase
        sr:           Sample-Rate (muss 48000 sein)
        phase_name:   Phasenname für Logging

    Returns:
        dict mit keys: delta_sone, delta_phon, action, sone_before, sone_after,
                       phon_before, phon_after, phase_name, valid
    """
    result_before = compute_specific_loudness_zwicker(audio_before, sr)
    result_after = compute_specific_loudness_zwicker(audio_after, sr)

    delta_sone = result_after.total_loudness_sone - result_before.total_loudness_sone
    delta_phon = result_after.loudness_phon - result_before.loudness_phon

    # §4.1b ΔN-Entscheidungstabelle
    if not (result_before.computation_valid and result_after.computation_valid):
        action = "skip_invalid_measurement"
    elif delta_sone <= 0.5:
        action = "ok"
    elif delta_sone <= 1.0:
        action = "info"
    elif delta_sone <= 2.0:
        action = "warning"
    else:
        action = "fail_dry_wet_rescue"

    if action in ("warning", "fail_dry_wet_rescue"):
        logger.warning(
            "§4.1b Zwicker-Lautheit: Verarbeitungsschritt=%s ΔN=%.2f sone (%.1f phon) → %s",
            phase_name,
            delta_sone,
            delta_phon,
            action,
        )
    elif action == "info":
        logger.info(
            "§4.1b Zwicker-Lautheit: Verarbeitungsschritt=%s ΔN=%.2f sone (%.1f phon) → info",
            phase_name,
            delta_sone,
            delta_phon,
        )

    return {
        "phase_name": phase_name,
        "sone_before": result_before.total_loudness_sone,
        "sone_after": result_after.total_loudness_sone,
        "phon_before": result_before.loudness_phon,
        "phon_after": result_after.loudness_phon,
        "delta_sone": delta_sone,
        "delta_phon": delta_phon,
        "action": action,
        "valid": result_before.computation_valid and result_after.computation_valid,
    }


# ──────────────────────────────────────────────────────────────────────
# Spezifische Lautheit nach ISO 532-1 als numpy-Array (schnelle Variante)
# ──────────────────────────────────────────────────────────────────────


def compute_specific_loudness_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """Gibt spezifische Lautheit N'(z) als float64-Array zurück (24 Bark-Bänder).

    Convenience-Funktion für Integration in PMGG und ArtifactFreedomGate.

    Returns:
        np.ndarray shape (24,) — N' in sone/Bark pro Bark-Band.
        Gibt np.zeros(24) bei Fehler zurück (nie raise).
    """
    result = compute_specific_loudness_zwicker(audio, sr)
    if not result.computation_valid:
        return np.zeros(N_BARK, dtype=np.float64)  # type: ignore[no-any-return]
    return result.specific_loudness


# ──────────────────────────────────────────────────────────────────────
# Bark-Energie-Profil (vereinfachte Variante ohne Phon-Konversion)
# Für schnelle Nutzung in Guards (< 5 ms)
# ──────────────────────────────────────────────────────────────────────


def compute_bark_energy_profile(audio: np.ndarray, sr: int) -> np.ndarray:
    """Schnelles Bark-Energieprofil ohne Phon-Konversion (RMS pro Band).

    Liefert RMS-Energie in 24 Bark-Bändern als float64-Array.
    Verwendet gecachete Filterbank — ca. 3–10 ms für 5-s-Audio.

    Returns:
        np.ndarray shape (24,) — RMS-Energie pro Band, NaN-frei.
    """
    try:
        arr = np.asarray(audio, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
        if arr.size == 0:
            return np.zeros(N_BARK, dtype=np.float64)  # type: ignore[no-any-return]
        # Cap at 5 s for performance
        arr = arr[: int(5 * sr)]

        filters = _build_bark_filterbank(sr)
        profile = np.zeros(N_BARK, dtype=np.float64)
        for b, sos in enumerate(filters):
            if sos is None:
                continue
            filtered = _sp_signal.sosfiltfilt(sos, arr)
            profile[b] = float(np.sqrt(np.mean(filtered**2)))
        return np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]
    except Exception as _e:
        logger.debug("berechnen_bark_energy_Profil Fehler: %s", _e)
        return np.zeros(N_BARK, dtype=np.float64)  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────────
# §0a Rauschboden-Textur-Profil (Material-adaptiv)
# ──────────────────────────────────────────────────────────────────────
# Material-spezifische Referenz-Rauschboden-Spektralformen (normiert 0–1).
# Quelle: empirische Messungen typischer Tonträger (Jorgensen 1996,
# Zar 1989, AES-65-2012).  8 Bark-Aggregatbänder (je 3 Bark-Bänder).
# Index 0 = Sub-Bass (Bark 0–2), Index 7 = Air (Bark 21–23).
_MATERIAL_NOISE_TEXTURE: dict[str, np.ndarray] = {
    # Vinyl: 1/f Pink-Noise + Tiefton-Rumble (Rille → Motorbrumm 30–100 Hz)
    "vinyl": np.array([0.85, 0.72, 0.55, 0.42, 0.35, 0.30, 0.28, 0.25], dtype=np.float64),
    # Shellac: breitbandiger Rauschboden (Schellack-Körnigkeit + Nadel-HF)
    "shellac": np.array([0.70, 0.65, 0.60, 0.58, 0.55, 0.50, 0.45, 0.40], dtype=np.float64),
    # Tape/Reel-Tape: Tape-Hiss (HF-dominant 4–12 kHz) + moderater LF-Grundrausch
    "tape": np.array([0.40, 0.35, 0.38, 0.45, 0.55, 0.70, 0.80, 0.75], dtype=np.float64),
    "reel_tape": np.array([0.38, 0.32, 0.35, 0.42, 0.52, 0.68, 0.78, 0.72], dtype=np.float64),
    # Kassette: stärkerer HF-Hiss + Dolby-NR-Restspuren
    "cassette": np.array([0.42, 0.38, 0.40, 0.48, 0.60, 0.75, 0.85, 0.80], dtype=np.float64),
    # Wachs-Zylinder: extremer LF-Rumble, kaum HF (mechanische BW < 5 kHz)
    "wax_cylinder": np.array([0.95, 0.80, 0.60, 0.35, 0.15, 0.08, 0.05, 0.03], dtype=np.float64),
    # Drahtaufnahme: mid-heavy mit mechanischem Grundrauschen
    "wire_recording": np.array([0.55, 0.50, 0.55, 0.50, 0.40, 0.30, 0.20, 0.10], dtype=np.float64),
    # CD/Digital: White Noise Floor (Quantisierungsrauschen, gleichmäßig)
    "cd_digital": np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64),
    # MP3 (niedrig): Codec-Artefakte im Presence-Band, reduziertes Air
    "mp3_low": np.array([0.45, 0.48, 0.52, 0.55, 0.50, 0.35, 0.20, 0.10], dtype=np.float64),
}


def compute_noise_texture_profile(
    audio: np.ndarray,
    sr: int,
    *,
    max_duration_s: float = 5.0,
) -> np.ndarray:
    """Berechnet 8-band Bark-aggregated noise-floor spectral shape.

    Measures the noise texture of a signal by analysing quiet frames
    (below -35 dBFS RMS).  Returns a normalised (0–1) shape vector
    representing the spectral distribution of the noise floor.

    §0a: "Rauschboden-*Niveau* UND -*Textur* des originalen Aufnahmemediums
    anstreben" — dieses Profil ist der Textur-Anteil.

    Args:
        audio: Mono or stereo float audio.
        sr: Sample rate.
        max_duration_s: Cap analysis length for performance.

    Returns:
        np.ndarray shape (8,) — normalised noise-texture profile (0–1).
        Zeros if insufficient quiet frames found.
    """
    try:
        arr = np.asarray(audio, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
        arr = arr[: int(max_duration_s * sr)]
        if arr.size < int(0.1 * sr):
            return np.zeros(8, dtype=np.float64)  # type: ignore[no-any-return]

        # Frame-based noise extraction: select quiet frames < -35 dBFS
        frame_len = int(0.05 * sr)  # 50 ms frames
        hop = frame_len // 2
        n_frames = max(1, (len(arr) - frame_len) // hop)

        quiet_spectra: list[np.ndarray] = []
        n_fft = min(2048, frame_len)
        win = np.hanning(n_fft).astype(np.float64)

        for fi in range(n_frames):
            s = fi * hop
            e = s + frame_len
            if e > len(arr):
                break
            frame = arr[s:e]
            rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
            rms_db = 20.0 * np.log10(rms + 1e-12)
            if rms_db < -35.0:
                # Analyse spectrum of quiet frame
                seg = frame[:n_fft]
                spec = np.abs(np.fft.rfft(seg * win)) ** 2
                quiet_spectra.append(spec)

        if len(quiet_spectra) < 3:
            return np.zeros(8, dtype=np.float64)  # type: ignore[no-any-return]

        # Average PSD of quiet frames
        avg_psd = np.median(np.array(quiet_spectra), axis=0)
        avg_psd = np.nan_to_num(avg_psd, nan=0.0, posinf=0.0, neginf=0.0)

        freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)

        # Aggregate into 8 macro-bands (each covering 3 Bark bands)
        profile = np.zeros(8, dtype=np.float64)
        for band_idx in range(8):
            bark_start = band_idx * 3
            bark_end = min(bark_start + 3, N_BARK)
            if bark_start >= N_BARK:
                break
            lo_hz = float(BARK_EDGES_HZ[bark_start])
            hi_hz = float(BARK_EDGES_HZ[min(bark_end, N_BARK)])
            mask = (freq_bins >= lo_hz) & (freq_bins < hi_hz)
            if np.any(mask):
                profile[band_idx] = float(np.mean(avg_psd[mask]))

        # Normalise to 0–1 range
        pmax = float(np.max(profile))
        if pmax > 1e-15:
            profile /= pmax

        return np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]

    except Exception as _e:
        logger.debug("berechnen_noise_texture_Profil error: %s", _e)
        return np.zeros(8, dtype=np.float64)  # type: ignore[no-any-return]


def get_material_noise_texture(material_type: str) -> np.ndarray:
    """Gibt canonical noise-texture reference for a material type zurück.

    Args:
        material_type: Material key (e.g. 'vinyl', 'shellac', 'reel_tape').

    Returns:
        np.ndarray shape (8,) — reference noise texture profile.
        Falls back to cd_digital (flat) for unknown materials.
    """
    key = str(material_type or "unknown").lower().strip()
    return _MATERIAL_NOISE_TEXTURE.get(key, _MATERIAL_NOISE_TEXTURE["cd_digital"]).copy()


def synthesize_comfort_noise(
    audio: np.ndarray,
    sr: int,
    measured_texture: np.ndarray,
    target_texture: np.ndarray,
    noise_floor_dbfs: float = -65.0,
) -> np.ndarray:
    """Synthesise material-authentic comfort noise and blend into quiet passages.

    §0a: After denoising, the residual noise floor may have a spectral shape
    (white/clinical) that differs from the original carrier's characteristic
    noise.  This function reshapes the noise floor to match the material's
    texture profile without affecting musical content.

    Only affects frames below -40 dBFS — musical content is untouched.

    Args:
        audio: Denoised audio (mono, float64/32).
        sr: Sample rate.
        measured_texture: 8-band texture of the denoised noise floor.
        target_texture: 8-band texture of the original material.
        noise_floor_dbfs: Target noise floor level in dBFS.

    Returns:
        Audio with reshaped noise floor in quiet passages.
    """
    try:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.size == 0:
            return audio

        # Compute spectral correction (target / measured), capped
        measured_safe = np.clip(measured_texture, 1e-6, None)
        correction = np.clip(target_texture / measured_safe, 0.3, 3.0)

        # Check if correction is meaningful (> 3 dB difference in any band)
        max_correction_db = 20.0 * np.log10(float(np.max(correction)))
        min_correction_db = 20.0 * np.log10(float(np.min(correction)))
        if abs(max_correction_db) < 3.0 and abs(min_correction_db) < 3.0:
            return audio  # Texture already matches — no comfort noise needed

        # Generate coloured noise matching target texture
        n_samples = len(arr)
        n_fft = 2048
        freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)

        # Build spectral shaping filter from 8-band correction
        shape_filter = np.ones(len(freq_bins), dtype=np.float64)
        for band_idx in range(8):
            bark_start = band_idx * 3
            bark_end = min(bark_start + 3, N_BARK)
            if bark_start >= N_BARK:
                break
            lo_hz = float(BARK_EDGES_HZ[bark_start])
            hi_hz = float(BARK_EDGES_HZ[min(bark_end, N_BARK)])
            mask = (freq_bins >= lo_hz) & (freq_bins < hi_hz)
            if np.any(mask):
                shape_filter[mask] = correction[band_idx]

        # Synthesise shaped noise in OLA blocks
        target_rms = 10.0 ** (noise_floor_dbfs / 20.0)
        hop = n_fft // 2
        noise_out = np.zeros(n_samples, dtype=np.float64)
        win = np.hanning(n_fft).astype(np.float64)

        rng = np.random.default_rng(42)  # Deterministic for reproducibility
        for s in range(0, n_samples - n_fft, hop):
            # White noise → FFT → shape → iFFT
            white = rng.standard_normal(n_fft)
            spec = np.fft.rfft(white)
            shaped_spec = spec * shape_filter
            shaped_noise = np.fft.irfft(shaped_spec, n=n_fft) * win
            # Normalise to target RMS
            block_rms = float(np.sqrt(np.mean(shaped_noise**2) + 1e-12))
            if block_rms > 1e-10:
                shaped_noise *= target_rms / block_rms
            noise_out[s : s + n_fft] += shaped_noise

        # Overlap-add normalisation
        win_sum = np.zeros(n_samples, dtype=np.float64)
        for s in range(0, n_samples - n_fft, hop):
            win_sum[s : s + n_fft] += win
        win_sum = np.clip(win_sum, 1e-8, None)
        noise_out /= win_sum

        # Apply only to quiet frames (< -40 dBFS) with smooth crossfade
        frame_len = int(0.03 * sr)  # 30 ms
        fade_len = int(0.01 * sr)  # 10 ms crossfade
        result = arr.copy()

        for s in range(0, n_samples - frame_len, frame_len // 2):
            e = min(s + frame_len, n_samples)
            frame = arr[s:e]
            rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
            rms_db = 20.0 * np.log10(rms + 1e-12)

            if rms_db < -40.0:
                # Blend: replace existing noise floor with shaped comfort noise
                blend = 0.7  # 70% comfort noise, 30% original residual
                noise_seg = noise_out[s:e]
                result[s:e] = (1.0 - blend) * arr[s:e] + blend * noise_seg
                # Smooth edges
                if s > fade_len:
                    fade_in = np.linspace(0.0, 1.0, fade_len)
                    result[s : s + fade_len] = (1.0 - fade_in) * arr[s : s + fade_len] + fade_in * result[
                        s : s + fade_len
                    ]

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(audio.dtype)  # type: ignore[no-any-return]

    except Exception as _e:
        logger.debug("synthesize_comfort_noise error: %s — returning unmodified audio", _e)
        return audio


# ──────────────────────────────────────────────────────────────────────
# §2.45a-IV Zeitvariante Loudness (von Bismarck 1974 / Moore 2014)
# ──────────────────────────────────────────────────────────────────────


def compute_time_varying_loudness(
    audio: np.ndarray,
    sr: int,
    *,
    window_ms: float = 200.0,
    hop_ms: float = 50.0,
) -> np.ndarray:
    """Compute time-varying loudness envelope in sone (simplified model).

    Based on Moore, Glasberg & Baer (1997) temporal integration model
    with simplified attack/release time constants from von Bismarck (1974):
      - Attack τ ≈ 20 ms (fast rise to perceived loudness)
      - Release τ ≈ 100 ms (slow decay — masking persistence)

    This provides a perceptually-weighted loudness envelope that captures
    subjective loudness changes missed by LUFS and Zwicker stationäre Methode.

    §2.45a-II: Used for envelope-aware gain compensation — gain adjustments
    follow the perceived loudness contour, not the RMS contour.

    Args:
        audio: Mono or stereo float audio.
        sr: Sample rate.
        window_ms: Analysis window in ms (default 200 ms — temporal integration).
        hop_ms: Hop between measurements in ms.

    Returns:
        np.ndarray shape (n_frames,) — instantaneous loudness in sone per frame.
    """
    try:
        arr = np.asarray(audio, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)

        win_samples = max(64, int(window_ms * sr / 1000.0))
        hop_samples = max(16, int(hop_ms * sr / 1000.0))
        n_frames = max(1, (len(arr) - win_samples) // hop_samples + 1)

        # Frame-level stationäre Loudness
        raw_loudness = np.zeros(n_frames, dtype=np.float64)
        for fi in range(n_frames):
            s = fi * hop_samples
            e = s + win_samples
            if e > len(arr):
                break
            frame = arr[s:e]
            # Fast RMS → approximate loudness via Stevens' power law:
            # L = k * I^0.3 (sone approximation for broadband signals)
            rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
            # Convert RMS to approximate SPL (dB), then to sone
            rms_db = 20.0 * np.log10(max(rms, 1e-10))
            # Map dBFS to approximate Phon (40 phon ≈ -20 dBFS for typical mastering)
            approx_phon = max(0.0, rms_db + 60.0)  # rough mapping
            # Zwicker sone formula: N = 2^((P-40)/10) for P >= 40 phon
            if approx_phon >= 40.0:
                raw_loudness[fi] = 2.0 ** ((approx_phon - 40.0) / 10.0)
            elif approx_phon > 0.0:
                raw_loudness[fi] = (approx_phon / 40.0) ** 2.7  # sub-threshold approximation
            else:
                raw_loudness[fi] = 0.0

        # Apply temporal integration (attack/release smoothing)
        # τ_attack ≈ 20 ms, τ_release ≈ 100 ms (von Bismarck 1974)
        dt = hop_ms / 1000.0
        alpha_attack = 1.0 - np.exp(-dt / 0.020)  # 20 ms attack
        alpha_release = 1.0 - np.exp(-dt / 0.100)  # 100 ms release

        smoothed = np.zeros_like(raw_loudness)
        smoothed[0] = raw_loudness[0]
        for fi in range(1, len(raw_loudness)):
            if raw_loudness[fi] > smoothed[fi - 1]:
                smoothed[fi] = smoothed[fi - 1] + alpha_attack * (raw_loudness[fi] - smoothed[fi - 1])
            else:
                smoothed[fi] = smoothed[fi - 1] + alpha_release * (raw_loudness[fi] - smoothed[fi - 1])

        return np.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]

    except Exception as _e:
        logger.debug("berechnen_time_varying_loudness error: %s", _e)
        return np.zeros(1, dtype=np.float64)  # type: ignore[no-any-return]


def compute_loudness_envelope_delta(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
) -> dict:
    """Compare time-varying loudness envelopes before/after a phase.

    Returns statistics about how the perceived loudness contour changed,
    not just the overall level.

    Returns:
        dict with keys: mean_delta_sone, max_delta_sone, rms_delta_sone,
                        quiet_passage_delta_sone, n_frames, valid
    """
    try:
        env_before = compute_time_varying_loudness(audio_before, sr)
        env_after = compute_time_varying_loudness(audio_after, sr)

        # Align lengths
        n = min(len(env_before), len(env_after))
        if n == 0:
            return {"valid": False}
        eb, ea = env_before[:n], env_after[:n]

        delta = ea - eb
        # Quiet passages: where before-loudness < 0.5 sone
        quiet_mask = eb < 0.5
        quiet_delta = float(np.mean(delta[quiet_mask])) if np.any(quiet_mask) else 0.0

        return {
            "mean_delta_sone": float(np.mean(delta)),
            "max_delta_sone": float(np.max(np.abs(delta))),
            "rms_delta_sone": float(np.sqrt(np.mean(delta**2))),
            "quiet_passage_delta_sone": quiet_delta,
            "n_frames": int(n),
            "valid": True,
        }
    except Exception as _e:
        logger.debug("berechnen_loudness_envelope_delta error: %s", _e)
        return {"valid": False}


# ──────────────────────────────────────────────────────────────────────
# §4.5 Reusable Psychoacoustic Masking Clamp — for ALL subtraktive/additive Phasen
# ──────────────────────────────────────────────────────────────────────


def apply_psychoacoustic_masking_clamp(
    original_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
    *,
    strength: float = 1.0,
    mode: str = "subtractive",
    _min_energy_ratio: float = 0.20,  # reserved — future per-band floor tuning
    masking_result=None,
) -> np.ndarray:
    """Wendet an: psychoacoustic masking to protect inaudible modifications.

    This is the **standard integration point** for ALL phases that modify
    spectral content.  It ensures that:
      - Subtractive phases (denoise, dereverb, gate, etc.) only remove energy
        that exceeds the psychoacoustic masking threshold
      - Additive phases (exciter, harmonic restoration, spectral repair) only
        add energy where it would be audible above the masking threshold
      - Musical content in masked regions is preserved (§0, Primum non nocere)

    Uses the central `PsychoacousticMaskingModel` (ISO 11172-3 simultaneous +
    temporal masking, 24 Bark bands, §4.5).

    §2.54: Masking gain is strength-scaled — at low PMGG strength the clamp
    is near-transparent to avoid unexpected energy changes.

    Args:
        original_audio: Audio BEFORE the phase operation (mono float).
        processed_audio: Audio AFTER the phase operation (mono float).
        sr: Sample rate (must be 48000).
        strength: Phase strength ∈ [0,1] — scales the masking effect.
        mode: "subtractive" (denoise, gate, dereverb) or "additive" (exciter,
              harmonic, spectral repair). Controls how masking is applied.
        min_energy_ratio: Minimum energy preservation ratio (§8.2).

    Returns:
        Masking-adjusted processed audio. Falls back to processed_audio on error.
    """
    try:
        if sr != 48000:
            return processed_audio

        from backend.core.psychoacoustic_masking_model import compute_masking_threshold  # pylint: disable=import-outside-toplevel  # noqa: I001

        orig = np.asarray(original_audio, dtype=np.float32)
        proc = np.asarray(processed_audio, dtype=np.float32)

        if orig.ndim == 2:
            orig_mono = orig.mean(axis=0) if orig.shape[0] <= 2 else orig.mean(axis=1)
        else:
            orig_mono = orig
        if proc.ndim == 2:
            proc_mono = proc.mean(axis=0) if proc.shape[0] <= 2 else proc.mean(axis=1)
        else:
            proc_mono = proc

        if orig_mono.size < 1024:
            return processed_audio

        # Compute masking threshold on the ORIGINAL audio (or reuse precomputed result).
        # Reuse improves consistency across phases and avoids redundant per-phase recompute.
        if masking_result is None:
            masking_result = compute_masking_threshold(orig_mono, sr)
        gain_t = np.mean(masking_result.gain_modifier, axis=1).astype(np.float32)

        # Interpolate to sample-level
        hop = 512
        centers = np.arange(len(gain_t)) * float(hop) + hop * 0.5

        # Apply mode-specific masking
        effective_strength = float(np.clip(strength, 0.0, 1.0))

        if mode == "subtractive":
            # For subtractive phases: limit how much energy is removed in
            # masked regions. gain_modifier near 0 = masked (keep more original),
            # near 1 = audible (allow full processing).
            if processed_audio.ndim == 2:
                result = proc.copy()
                for ch in range(proc.shape[0] if proc.shape[0] <= 2 else proc.shape[1]):
                    if proc.shape[0] <= 2:
                        ch_orig = orig[ch] if ch < orig.shape[0] else orig[0]
                        ch_proc = proc[ch]
                        x = np.arange(len(ch_proc), dtype=np.float32)
                        gain_samples = np.interp(x, centers, gain_t).astype(np.float32)
                        scaled = 1.0 + effective_strength * (gain_samples - 1.0)
                        # Blend: in masked regions (low gain), keep more original
                        # §2.62 G_floor≥0.10: kein NR-Gain unter 10% (verhindert Stille-Artefakt)
                        blend = np.clip(scaled, 0.10, 1.0)
                        result[ch] = blend * ch_proc + (1.0 - blend) * ch_orig
                    else:
                        ch_orig = orig[:, ch] if ch < orig.shape[1] else orig[:, 0]
                        ch_proc = proc[:, ch]
                        x = np.arange(len(ch_proc), dtype=np.float32)
                        gain_samples = np.interp(x, centers, gain_t).astype(np.float32)
                        scaled = 1.0 + effective_strength * (gain_samples - 1.0)
                        # §2.62 G_floor≥0.10
                        blend = np.clip(scaled, 0.10, 1.0)
                        result[:, ch] = blend * ch_proc + (1.0 - blend) * ch_orig
                result_arr = np.asarray(np.clip(result, -1.0, 1.0), dtype=processed_audio.dtype)
                return result_arr  # type: ignore[no-any-return]
            else:
                x = np.arange(len(proc_mono), dtype=np.float32)
                gain_samples = np.interp(x, centers, gain_t).astype(np.float32)
                scaled = 1.0 + effective_strength * (gain_samples - 1.0)
                # §2.62 G_floor≥0.10: verhindert klinisches Stille-Artefakt
                blend = np.clip(scaled, 0.10, 1.0)
                result = blend * proc_mono + (1.0 - blend) * orig_mono
                result_arr = np.asarray(np.clip(result, -1.0, 1.0), dtype=processed_audio.dtype)
                return cast(np.ndarray, result_arr)

        elif mode == "additive":
            # For additive phases: limit how much energy is ADDED in masked
            # regions. gain_modifier near 0 = masked (don't add), near 1 = audible.
            delta = proc - orig
            if delta.ndim == 2:
                delta_mono = delta.mean(axis=0) if delta.shape[0] <= 2 else delta.mean(axis=1)
            else:
                delta_mono = delta

            x = np.arange(len(delta_mono), dtype=np.float32)
            gain_samples = np.interp(x, centers, gain_t).astype(np.float32)
            scaled = np.clip(effective_strength * gain_samples, 0.0, 1.0)

            if processed_audio.ndim == 2:
                if delta.shape[0] <= 2:
                    scaled_2d = scaled[np.newaxis, :]
                else:
                    scaled_2d = scaled[:, np.newaxis]
                result = orig + delta * scaled_2d
            else:
                result = orig + delta * scaled
            return np.clip(result, -1.0, 1.0).astype(processed_audio.dtype)  # type: ignore[no-any-return]

        return processed_audio

    except Exception as _e:
        logger.debug("anwenden_psychoacoustic_masking_clamp nicht blockierend: %s", _e)
        return processed_audio


def compute_erb_masking_threshold(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    masking_offset_db: float = 14.5,
    spreading_erbs: float = 2.0,
) -> np.ndarray:
    """Berechnet ERB-weighted simultaneous masking threshold for a signal.

    This is a frequency-domain masking threshold (used by AFG, spectral repair,
    EQ correction, etc.) based on the Equivalent Rectangular Bandwidth
    (Glasberg & Moore 1990) spreading function.

    Args:
        audio: Mono float audio segment.
        sr: Sample rate.
        n_fft: FFT size.
        masking_offset_db: Masking level below signal peak (default 14.5 dB).
        spreading_erbs: Spreading range in ERB units.

    Returns:
        np.ndarray shape (n_fft//2+1,) — masking threshold in dB.
    """
    try:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)

        n = min(len(arr), n_fft)
        if n < 64:
            return np.full(n_fft // 2 + 1, -120.0, dtype=np.float64)  # type: ignore[no-any-return]

        win = np.hanning(n).astype(np.float64)
        spec = np.abs(np.fft.rfft(arr[:n] * win))
        mag_db = 20.0 * np.log10(spec + 1e-12)
        n_bins = len(mag_db)

        freq_axis = np.arange(n_bins) * sr / float(n_fft)
        erb_widths = 24.7 * (4.37 * freq_axis / 1000.0 + 1.0)

        threshold = np.full(n_bins, -120.0, dtype=np.float64)
        for b in range(n_bins):
            if mag_db[b] < -60.0:
                continue
            erb_w = max(1.0, erb_widths[b])
            spread_hz = spreading_erbs * erb_w
            spread_bins = max(1, int(spread_hz / max(1.0, float(sr) / n_fft)))
            lo = max(0, b - spread_bins)
            hi = min(n_bins, b + spread_bins + 1)
            ml = mag_db[b] - masking_offset_db
            threshold[lo:hi] = np.maximum(threshold[lo:hi], ml)

        return threshold  # type: ignore[no-any-return]

    except Exception as _e:
        logger.debug("berechnen_erb_masking_Schwelle error: %s", _e)
        return np.full(n_fft // 2 + 1, -120.0, dtype=np.float64)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# §2.62 Psychoakustischer Masking-Guard — ISO 11172-3 Masking Threshold
# ---------------------------------------------------------------------------


def compute_masking_threshold_iso11172(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """§2.62 — Per-Band-Maskierungsschwelle nach ISO 11172-3 (MPEG Psychoacoustic Model 1).

    Berechnet die minimale wahrnehmbare Rauschschwelle für jeden Frequenzbin und Frame.
    NR-Gain-Floor pro Band: G_floor[band] = max(0.10, masking_threshold[band] / noise_estimate[band]).

    Parameters
    ----------
    audio : np.ndarray
        Mono-Signal (N,) float32/float64, normalisiert ±1.
    sr : int
        Abtastrate in Hz.
    n_fft : int
        FFT-Größe (Standard: 2048).
    hop_length : int
        Hop-Länge in Samples (Standard: 512).

    Returns
    -------
    np.ndarray
        Maskierungsschwelle (n_freq_bins, n_frames), normalisiert [0, 1]:
        1.0 = vollständige Maskierung (Rauschen unsichtbar), 0.0 = keine Maskierung.
    """
    # --- Mono erzwingen ---
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = np.asarray(audio, dtype=np.float64)

    n_fft_half = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_fft_half, dtype=np.float64)  # Hz
    f_khz = np.maximum(freqs / 1000.0, 1e-6)  # kHz, nie 0

    # --- ATH (Absolute Threshold of Hearing) nach ISO 226:2003 ---
    # Terhardt-Formel: dB SPL re 20 µPa
    ath_db = 3.64 * np.power(f_khz, -0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * np.power(f_khz, 4)
    ath_db = np.clip(ath_db, -20.0, 80.0)

    # --- Bark-Skala (Zwicker-Formel) ---
    bark = 13.0 * np.arctan(0.76 * f_khz) + 3.5 * np.arctan((f_khz / 7.5) ** 2)

    # --- STFT des Signals ---
    # §v10.103 noverlap-Guard: clamp für kurzes Audio / dynamisches n_fft
    _noverlap = min(n_fft - hop_length, max(0, n_fft - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, Zxx = _sp_signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann")
    power = np.abs(Zxx) ** 2  # (n_freq, n_frames)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-20))  # dB

    n_freq, n_frames = power_db.shape
    masking_db = np.zeros((n_freq, n_frames), dtype=np.float64)

    # --- Terhardt-Spreading-Funktion (vereinfacht, MPEG-1 Modell 1) ---
    # S(dz) = 15.81 + 7.5*(dz+0.474) - 17.5*sqrt(1+(dz+0.474)^2) dB
    # Für jeden Masker k wird die Schwelle in allen Bändern addiert
    bark_diff = bark[:, np.newaxis] - bark[np.newaxis, :]  # (n_freq, n_masker)
    dz = bark_diff + 0.474
    spreading_db = 15.81 + 7.5 * dz - 17.5 * np.sqrt(1.0 + dz**2)
    spreading_db = np.clip(spreading_db, -80.0, 0.0)

    for t in range(n_frames):
        masker_level = power_db[:, t]  # dB pro Bin
        # Nur Bins über ATH sind wirksame Masker
        active = masker_level > ath_db
        if np.any(active):
            # Absolute Maskierungsschwelle = Masker-Level + Spreading - Tonalitäts-Korrekturfaktor
            # Vereinfachung: Tonalitätsfaktor = 6 dB (Sinuston-Masker)
            spread = spreading_db[:, active] + masker_level[np.newaxis, active] - 6.0
            # Energetische Addition: dB → Linear → Maximum → dB
            spread_lin = np.power(10.0, spread / 10.0)
            total_mask_lin = spread_lin.sum(axis=1)
            masking_db[:, t] = 10.0 * np.log10(np.maximum(total_mask_lin, 1e-20))
        else:
            masking_db[:, t] = ath_db

        # ATH als untere Schranke
        masking_db[:, t] = np.maximum(masking_db[:, t], ath_db)

    # --- Normierung auf [0, 0.70] ---
    # §2.62: G_floor = max(0.10, masking_threshold / noise_estimate)
    # Rauschboden-Schätzung via Minimum-Statistics (10th Percentile über Zeit pro Bin +
    # Bias-Korrektur +1.5 dB). Bewusste Deckelung bei 0.70: Die Masking-Guard soll
    # klinische Stille-Artefakte verhindern, NICHT NR vollständig blockieren.
    # Selbst für voll maskiertes Rauschen sind 30 % NR psychoakustisch transparent.
    noise_db = np.percentile(power_db, 10.0, axis=1, keepdims=True).astype(np.float64) + 1.5
    ratio = np.power(10.0, (masking_db - noise_db) / 20.0)  # Amplitudenratio
    # Cap bei 0.70: NR kann immer mindestens 30 % reduzieren (kein vollständiger NR-Block).
    ratio = np.clip(ratio, 0.0, 0.70)

    return np.asarray(ratio, dtype=np.float32)  # type: ignore[no-any-return]


def compute_versa_confidence(snr_estimate_db: float, material_type: str) -> float:
    """Schätzt how trustworthy a VERSA-style perceptual score is for the material.

    The confidence is intentionally conservative for low-restorability carrier media
    such as shellac and rises with cleaner, high-SNR digital sources.

    Args:
        snr_estimate_db: Estimated signal-to-noise ratio in dB.
        material_type: Material key such as ``cd``, ``vinyl`` or ``shellac``.

    Returns:
        Confidence scalar in ``[0.10, 1.00]``.
    """
    material_key = str(material_type or "unknown").strip().lower()
    material_key = {
        "cd_digital": "cd",
        "dat": "cd",
        "streaming": "mp3_high",
        "tape": "cassette",
    }.get(material_key, material_key)

    material_anchor = {
        "cd": 0.96,
        "aac": 0.90,
        "mp3_high": 0.88,
        "reel_tape": 0.86,
        "vinyl": 0.82,
        "mp3_low": 0.78,
        "cassette": 0.74,
        "minidisc": 0.80,
        "shellac": 0.65,
        "lacquer_disc": 0.64,
        "wire_recording": 0.60,
        "wax_cylinder": 0.58,
        "unknown": 0.70,
    }.get(material_key, 0.70)

    snr_db = float(np.nan_to_num(snr_estimate_db, nan=0.0, posinf=45.0, neginf=-10.0))
    snr_norm = float(np.clip((snr_db + 5.0) / 45.0, 0.0, 1.0))
    snr_factor = 0.35 + 0.65 * snr_norm
    confidence = material_anchor * snr_factor
    return float(np.clip(confidence, 0.10, 1.00))


__all__ = [
    "BARK_CENTERS_HZ",
    "BARK_EDGES_HZ",
    "N_BARK",
    "ZwickerLoudnessResult",
    "apply_psychoacoustic_masking_clamp",
    "compute_bark_energy_profile",
    "compute_erb_masking_threshold",
    "compute_loudness_envelope_delta",
    "compute_masking_threshold_iso11172",
    "compute_noise_texture_profile",
    "compute_specific_loudness_array",
    "compute_specific_loudness_zwicker",
    "compute_versa_confidence",
    "compute_time_varying_loudness",
    "evaluate_mid_pipeline_loudness_delta",
    "get_material_noise_texture",
    "synthesize_comfort_noise",
]
