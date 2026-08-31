"""
===============================================================

Prüft vor jeder DSP-Operation: "Ist der Unterschied hörbar?"
Basiert auf Bark-Bändern, ISO-226-Hörschwellen und psychoakustischer Maskierung.

Prinzip: Wenn die erwartete Änderung in KEINEM hörbaren Frequenzband
oberhalb der Maskierungsschwelle liegt → Phase überspringen (kein Nutzen).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

from backend.core.dsp.bark_lufs_util import (
    BARK_EDGES_HZ,
    BARK_HEARING_THRESHOLD_DB,
    hz_to_bark,
    split_into_bark_bands,
)
from backend.core.dsp.bark_lufs_util import (
    N_GAMMATONE as N_BARK,
)

# ---------------------------------------------------------------------------
# JND (Just Noticeable Difference) pro Bark-Band
# Quelle: Zwicker & Fastl (1999), Tabelle 7.1
# ---------------------------------------------------------------------------
_JND_DB_PER_BARK = np.array(
    [
        2.0,
        1.8,
        1.5,
        1.3,
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.6,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
        1.8,
        2.0,
        2.5,
        3.0,
        3.5,
    ],
    dtype=np.float32,
)


def compute_perceptual_threshold(
    audio: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Berechnet die perzeptuelle Maskierungsschwelle pro Bark-Band.

    Kombiniert:
    1. Absolute Hörschwelle (ISO 226)
    2. Simultane Maskierung (Spektrale Energie maskiert Nachbarbänder)
    3. JND-Level (Minimal hörbare Pegeländerung)

    Returns:
        threshold_db: (n_bark,) float32 - Pegel in dB, unter dem Änderungen unhörbar sind
    """
    from backend.core.dsp.bark_lufs_util import measure_lufs_per_bark

    bands = split_into_bark_bands(
        audio if audio.ndim == 1 else np.mean(audio, axis=0),
        sr,
    )
    lufs = measure_lufs_per_bark(bands, sr)

    # Spread-Funktion: Energie in Band i maskiert Bänder j
    spread = np.zeros((N_BARK, N_BARK), dtype=np.float32)
    for i in range(N_BARK):
        for j in range(N_BARK):
            dz = abs(i - j)
            if dz <= 1:
                spread[i, j] = 1.0
            elif dz <= 4:
                spread[i, j] = 10.0 ** (-dz / 4.0)
            elif dz <= 10:
                spread[i, j] = 10.0 ** (-1.0 - (dz - 4) / 2.0)

    # Maskierungsschwelle = Hörschwelle + Spread-Beitrag + JND
    threshold_db = np.zeros(N_BARK, dtype=np.float32)
    for b in range(N_BARK):
        hearing = BARK_HEARING_THRESHOLD_DB[b]
        jnd = _JND_DB_PER_BARK[b]

        # Spread-Maskierung: Summe der Beiträge aller Bänder
        spread_contrib = 0.0
        for src in range(N_BARK):
            if lufs[src] > -60 and spread[src, b] > 0.001:
                # Maskierung = Quellpegel - 6dB pro Bark Distanz
                mask_db = lufs[src] - 6.0 * abs(src - b)
                spread_contrib = max(spread_contrib, mask_db)

        # Effektive Schwelle = max(Hörschwelle, Maskierung) + JND
        threshold_db[b] = float(max(hearing, spread_contrib) + jnd)

    return cast(np.ndarray, threshold_db)


def should_skip_phase(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
    *,
    min_audible_bands: int = 2,
    material_type: str = "unknown",
    genre: str = "unknown",
) -> bool:
    """Prüft, ob eine Phase hörbare Änderungen bewirkt hat.

    §v10.116: Material-adaptive JND + Genre-Tuning.
    Kassette/Rauschen → höhere JND (Änderungen sind schwerer hörbar).
    Klassik → niedrigere JND (Hörer sind kritischer).

    Spread-Masking ist für Codec-Design ("kann ich Rauschen verstecken?"),
    nicht für Restaurations-Validierung ("hat die Phase etwas geändert?").

    Algorithmus:
    1. Teile before/after in 24 Bark-Bänder
    2. Messe LUFS pro Band
    3. Zähle Bänder mit |delta| > JND UND Pegel > Hörschwelle
    4. Wenn < min_audible_bands → Phase war unhörbar

    Returns:
        True wenn Phase übersprungen werden sollte (keine hörbare Änderung)
    """
    from backend.core.perceptual_tuning import get_genre_jnd_factor, get_material_jnd_factor

    _mat_factor = get_material_jnd_factor(material_type)
    _genre_factor = get_genre_jnd_factor(genre)
    _combined_jnd_factor = _mat_factor * _genre_factor  # Multiplikativ: beide Faktoren wirken

    delta = audio_after.astype(np.float64) - audio_before.astype(np.float64)
    delta_rms = float(np.sqrt(np.mean(delta**2)) + 1e-12)
    if delta_rms < 1e-8:
        return True  # Keine Änderung

    # Schnell-Check: Globaler Pegelunterschied > 0.5 dB → garantiert hörbar
    rms_before = float(np.sqrt(np.mean(audio_before.astype(np.float64) ** 2)) + 1e-12)
    rms_after = float(np.sqrt(np.mean(audio_after.astype(np.float64) ** 2)) + 1e-12)
    if abs(20.0 * np.log10(rms_after / rms_before)) > 0.5:
        return False  # Globaler Pegelsprung → hörbar

    # Bark-Band-Analyse
    from backend.core.dsp.bark_lufs_util import measure_lufs_per_bark

    bands_before = split_into_bark_bands(
        audio_before if audio_before.ndim == 1 else np.mean(audio_before, axis=0),
        sr,
    )
    bands_after = split_into_bark_bands(
        audio_after if audio_after.ndim == 1 else np.mean(audio_after, axis=0),
        sr,
    )

    lufs_before = measure_lufs_per_bark(bands_before, sr)
    lufs_after = measure_lufs_per_bark(bands_after, sr)

    # Einfacher, robuster Test: Zähle Bänder mit hörbarer Änderung
    n_audible = 0
    for b in range(N_BARK):
        if lufs_before[b] < -60 and lufs_after[b] < -60:
            continue  # Beide unhörbar leise

        delta_db = float(abs(lufs_after[b] - lufs_before[b]))
        jnd = (
            float(_JND_DB_PER_BARK[b]) * 0.7 * _combined_jnd_factor
        )  # §v10.116: Material+Genre-adaptiv, §v10.101: ×0.7 konservativ

        # Eine Änderung ist hörbar wenn:
        # 1. Delta > JND des Bandes (Zwicker, ×0.7 konservativ)
        # 2. Mindestens ein Signal über der Hörschwelle liegt
        # §v10.101: Hörschwelle ist in dB SPL (ISO 226), Umrechnung auf dBFS:
        # 0 dBFS ≈ 100 dB SPL bei typischer Abhörlautstärke
        _hearing_dbfs = float(BARK_HEARING_THRESHOLD_DB[b]) - 100.0
        above_hearing = max(lufs_before[b], lufs_after[b]) > _hearing_dbfs

        if delta_db > jnd and above_hearing:
            n_audible += 1

    should_skip = n_audible < min_audible_bands
    return should_skip


def perceptual_loudness_normalize(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -18.0,
) -> np.ndarray:
    """LUFS-basierte Lautheitsnormalisierung mit Bark-Band-Korrektur.

    Im Gegensatz zur reinen dB-basierten Normalisierung berücksichtigt
    diese Funktion die Frequenzabhängigkeit der menschlichen Lautheitswahrnehmung.

    Args:
        audio: Audio-Signal float32
        sr: Sample-Rate
        target_lufs: Ziel-Lautheit in LUFS

    Returns:
        Lautheitsnormalisiertes Audio
    """
    from backend.core.dsp.bark_lufs_util import (
        bark_dynamics_target,
        measure_lufs_per_bark,
    )

    is_stereo = audio.ndim == 2
    mono = audio if not is_stereo else np.mean(audio, axis=0)

    # Bark-Band-Analyse
    bands = split_into_bark_bands(mono.astype(np.float32), sr)
    lufs_in = measure_lufs_per_bark(bands, sr)

    # Perzeptuelle Gain-Korrektur
    gain_db = bark_dynamics_target(lufs_in, lufs_in, target_lufs)

    # Synthese: Gain pro Bark-Band aufsummieren
    output = np.zeros_like(mono, dtype=np.float32)
    for b in range(N_BARK):
        if abs(gain_db[b]) < 0.1:
            output += bands[b]
        else:
            gain_lin = 10.0 ** (float(gain_db[b]) / 20.0)
            output += bands[b] * gain_lin

    if is_stereo:
        # Stereo-Mix: gleiche Gain-Kurve für beide Kanäle
        gain_ratio = output / (mono + 1e-12)
        output = np.column_stack(
            [
                audio[:, 0] * gain_ratio,
                audio[:, 1] * gain_ratio,
            ]
        ).astype(np.float32)

    return cast(np.ndarray, (np.clip(output, -1.0, 1.0)))


from typing import cast
