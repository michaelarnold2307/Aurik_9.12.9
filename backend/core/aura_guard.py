"""
§v10 Aura Guard — Bewahrt den unverwechselbaren Charakter jeder Aufnahme.

Die Aura ist das, was eine Aufnahme EINZIGARTIG macht:
- Der warme, leicht verhallte Klang eines Ballsaals von 1932
- Die intime Nähe einer Jazz-Club-Aufnahme von 1958
- Die druckvolle Präsenz einer 80er-Jahre-Studio-Produktion
- Die rohe Energie eines Live-Mitschnitts von 1975

Technische Restaurierung DARF diese Aura nicht zerstören.
Eine perfekt entrauschte, aber seelenlose Aufnahme ist wertlos.

Aura-Dimensionen (5):
1. ERA_CHARACTER:     Epochentypischer Frequenzgang (z.B. Schellack 300-5000Hz)
2. SPATIAL_SIGNATURE: Raumeindruck (trocken/hallig/intim/welt)
3. NOISE_PERSONALITY: Charakter des Rauschbodens (warmes Tape-Hiss vs. kaltes Digital)
4. DYNAMIC_CHARACTER: Epochentypische Dynamik (Mikrofon-Nahbesprechung vs. Raumaufnahme)
5. EMOTIONAL_TEMPERATURE: Warm/kalt, nah/distanziert, energetisch/ruhig

Composite AS (Aura Score) ∈ [0,1]:
- AS ≥ 0.80 = Aura perfekt erhalten
- AS < 0.40 = Aura zerstört — die Aufnahme klingt „falsch"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AuraProfile:
    """Das Aura-Profil einer Aufnahme."""

    era_character: float  # 0=modern, 1=authentisch epochentypisch
    spatial_signature: float  # 0=flach/steril, 1=räumlich lebendig
    noise_personality: float  # 0=steril, 1=charaktervoller Rauschboden
    dynamic_character: float  # 0=komprimiert, 1=epochentypische Dynamik
    emotional_temperature: float  # 0=kalt/distanziert, 1=warm/nah

    aura_score: float = 0.0
    era_label: str = ""
    spatial_label: str = ""
    label: str = ""  # "Authentisch 50er Jazz-Club" etc.
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Era-Character-Profile: typische Frequenzgänge pro Ära
# ─────────────────────────────────────────────────────────────────────────

ERA_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "acoustic_78rpm": {  # 1900-1925
        "freq_range": (300, 5000),
        "spectral_tilt": (-4.0, -8.0),  # dB/Oktave — starker Höhenabfall
        "noise_character": "broadband_crackle",  # type: ignore[dict-item]
        "dynamic_expectation": (15, 25),  # dB dynamic range
        "spatial_expectation": "mono_intimate",  # type: ignore[dict-item]
    },
    "electric_78rpm": {  # 1925-1948
        "freq_range": (200, 7000),
        "spectral_tilt": (-2.5, -5.0),
        "noise_character": "surface_hiss",  # type: ignore[dict-item]
        "dynamic_expectation": (20, 35),
        "spatial_expectation": "mono_present",  # type: ignore[dict-item]
    },
    "early_vinyl": {  # 1948-1960
        "freq_range": (150, 12000),
        "spectral_tilt": (-1.5, -3.0),
        "noise_character": "warm_hiss",  # type: ignore[dict-item]
        "dynamic_expectation": (25, 45),
        "spatial_expectation": "mono_early_stereo",  # type: ignore[dict-item]
    },
    "vinyl_golden": {  # 1960-1980
        "freq_range": (80, 16000),
        "spectral_tilt": (-0.5, -2.0),
        "noise_character": "vinyl_surface",  # type: ignore[dict-item]
        "dynamic_expectation": (30, 55),
        "spatial_expectation": "stereo_natural",  # type: ignore[dict-item]
    },
    "tape_analog": {  # 1950-1990
        "freq_range": (60, 18000),
        "spectral_tilt": (-0.5, -1.5),
        "noise_character": "tape_hiss_warm",  # type: ignore[dict-item]
        "dynamic_expectation": (35, 60),
        "spatial_expectation": "stereo_warm",  # type: ignore[dict-item]
    },
    "digital_early": {  # 1980-2000
        "freq_range": (40, 20000),
        "spectral_tilt": (0.0, -1.0),
        "noise_character": "digital_clean",  # type: ignore[dict-item]
        "dynamic_expectation": (40, 70),
        "spatial_expectation": "stereo_precise",  # type: ignore[dict-item]
    },
    "digital_modern": {  # 2000+
        "freq_range": (30, 20000),
        "spectral_tilt": (0.0, -0.5),
        "noise_character": "near_silent",  # type: ignore[dict-item]
        "dynamic_expectation": (30, 90),  # Kann auch loudness-war sein
        "spatial_expectation": "stereo_wide",  # type: ignore[dict-item]
    },
}


def detect_aura(
    audio: np.ndarray,
    sr: int,
    *,
    known_era: str | None = None,
    known_medium: str | None = None,
) -> AuraProfile:
    """Erkennt und charakterisiert die Aura einer Aufnahme.

    Args:
        audio:       Mono Audio
        sr:          Sample-Rate
        known_era:   Optional: bereits klassifizierte Ära
        known_medium: Optional: bereits klassifiziertes Medium

    Returns:
        AuraProfile mit allen 5 Dimensionen
    """
    arr = np.asarray(audio, dtype=np.float64)
    mono = arr.mean(axis=1) if arr.ndim == 2 else arr
    mono = np.atleast_1d(mono).ravel()

    # ── 1. Era Character ──
    era_char, era_label = _measure_era_character(mono, sr, known_era, known_medium)

    # ── 2. Spatial Signature ──
    spatial, spatial_label = _measure_spatial_signature(mono, sr, arr)

    # ── 3. Noise Personality ──
    noise_pers = _measure_noise_personality(mono, sr)

    # ── 4. Dynamic Character ──
    dynamic_char = _measure_dynamic_character(mono, sr)

    # ── 5. Emotional Temperature ──
    emotional_temp = _measure_emotional_temperature(mono, sr)

    # ── Composite Aura Score ──
    aura_score = float(
        np.clip(
            era_char * 0.30 + spatial * 0.20 + noise_pers * 0.15 + dynamic_char * 0.20 + emotional_temp * 0.15, 0.0, 1.0
        )
    )

    # ── Label ──
    warnings = []
    if era_char < 0.4:
        warnings.append("Epochencharakter gefährdet")
    if spatial < 0.4:
        warnings.append("Raumeindruck verloren")
    if noise_pers < 0.3:
        warnings.append("Rauschboden-Charakter zerstört")

    if aura_score >= 0.80:
        label = f"Authentisch {era_label} {spatial_label}"
        rec = "Aura perfekt erhalten — die Aufnahme klingt genau richtig für ihre Zeit."
    elif aura_score >= 0.50:
        label = f"{era_label} Charakter"
        rec = "Aura weitgehend erhalten. " + (warnings[0] if warnings else "")
    else:
        label = "Aura gefährdet"
        rec = "Die Restaurierung zerstört den Charakter der Aufnahme. " + (warnings[0] if warnings else "")

    return AuraProfile(
        era_character=era_char,
        spatial_signature=spatial,
        noise_personality=noise_pers,
        dynamic_character=dynamic_char,
        emotional_temperature=emotional_temp,
        aura_score=aura_score,
        era_label=era_label,
        spatial_label=spatial_label,
        label=label,
        recommendation=rec,
        warnings=warnings,
    )


def compare_aura(original: np.ndarray, restored: np.ndarray, sr: int) -> dict:
    """Vergleicht die Aura vor und nach der Restaurierung.

    Returns dict mit 'aura_preserved', 'delta', 'verdict'.
    """
    aura_orig = detect_aura(original, sr)
    aura_rest = detect_aura(restored, sr)

    delta = aura_rest.aura_score - aura_orig.aura_score
    preserved = delta > -0.08

    if preserved and aura_rest.aura_score >= 0.60:
        verdict = "Aura erhalten — die Aufnahme klingt nach ihrer Zeit."
    elif preserved:
        verdict = "Aura leicht verändert, aber Charakter erkennbar."
    else:
        verdict = (
            f"AURA-VERLUST: Die Restaurierung hat den Charakter zerstört (Δ={delta:+.2f}). {aura_rest.recommendation}"
        )

    return {
        "aura_preserved": preserved,
        "delta_aura": float(delta),
        "original_aura": aura_orig.aura_score,
        "restored_aura": aura_rest.aura_score,
        "original_label": aura_orig.label,
        "restored_label": aura_rest.label,
        "verdict": verdict,
        "warnings": aura_rest.warnings,
    }


# ── Messfunktionen ──────────────────────────────────────────────────────


def _measure_era_character(
    mono: np.ndarray,
    sr: int,
    known_era: str | None = None,
    known_medium: str | None = None,
) -> tuple[float, str]:
    """Misst, wie authentisch der Frequenzgang zur Ära passt."""
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.6, "unbekannt"

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Bestimme die wahrscheinlichste Ära aus dem Frequenzgang
    if known_era is None:
        # Heuristik: wo fällt die Energie ab?
        energy = np.cumsum(spec**2)
        total = energy[-1] + 1e-12
        # 95% Energie-Grenzfrequenz
        cutoff_idx = np.searchsorted(energy, 0.95 * total)
        cutoff_freq = freqs[min(cutoff_idx, len(freqs) - 1)]

        if cutoff_freq < 5000:
            era = "acoustic_78rpm"
        elif cutoff_freq < 8000:
            era = "electric_78rpm"
        elif cutoff_freq < 13000:
            era = "early_vinyl"
        elif cutoff_freq < 17000:
            era = "vinyl_golden"
        else:
            era = "digital_modern"
    else:
        era = known_era

    profile = ERA_PROFILES.get(era, ERA_PROFILES["vinyl_golden"])
    f_low, f_high = profile["freq_range"]

    # Prüfe Frequenzgang-Übereinstimmung
    lf_energy = float(np.sum(spec[(freqs >= f_low) & (freqs < f_high)] ** 2))
    total_energy = float(np.sum(spec**2))

    # Charakteristischer Frequenzanteil
    character_ratio = lf_energy / (total_energy + 1e-12)

    # Spektraler Tilt (dB/Oktave)
    mid_freqs = freqs[(freqs >= 300) & (freqs <= 8000)]
    if len(mid_freqs) > 10:
        mid_spec = spec_db[(freqs >= 300) & (freqs <= 8000)]
        coeffs = np.polyfit(np.log2(mid_freqs + 1), mid_spec, 1)
        tilt = coeffs[0]
    else:
        tilt = -3.0

    era_names = {
        "acoustic_78rpm": "Akustische 78rpm",
        "electric_78rpm": "Elektrische 78rpm",
        "early_vinyl": "Frühes Vinyl",
        "vinyl_golden": "Goldenes Vinyl-Zeitalter",
        "tape_analog": "Analoges Tonband",
        "digital_early": "Frühes Digital",
        "digital_modern": "Modernes Digital",
    }

    score = 0.85  # Default: gute Übereinstimmung
    if character_ratio < 0.4:
        score = 0.45
    if abs(tilt) < 0.5 and era in ("acoustic_78rpm", "electric_78rpm"):
        score = 0.30  # Zu flach für alte Ära → wurde vermutlich modernisiert

    return score, era_names.get(era, era)


def _measure_spatial_signature(mono: np.ndarray, sr: int, arr: np.ndarray) -> tuple[float, str]:
    """Misst den Raumeindruck."""
    # Reverb-Detektion via Abklingzeit
    n_fft = 2048
    if len(mono) < 4 * n_fft:
        return 0.7, "trocken"

    # Einfache RT60-Schätzung via Energie-Abklingkurve
    energy = []
    hop = n_fft // 2
    for i in range(0, len(mono) - n_fft, hop):
        chunk = mono[i : i + n_fft] * np.hanning(n_fft)
        energy.append(float(np.sum(chunk**2)))
    energy = np.array(energy)  # type: ignore[assignment]
    energy_db = 10.0 * np.log10(energy + 1e-12)  # type: ignore[operator]

    # Abklingzeit: wie lange bis Energie um 60dB fällt
    peak_idx = np.argmax(energy_db)
    peak_val = energy_db[peak_idx]
    decay = energy_db[peak_idx:]
    below_60 = decay < (peak_val - 60)

    if np.any(below_60):
        rt60_ms = (np.argmax(below_60) * hop / sr) * 1000
    else:
        rt60_ms = 200  # type: ignore  # Default

    if rt60_ms < 100:
        return 0.60, "sehr trocken (Nahaufnahme)"
    elif rt60_ms < 500:
        return 0.85, "natürlicher Raum"
    elif rt60_ms < 1500:
        return 0.70, "hallig (großer Raum)"
    else:
        return 0.50, "extrem hallig (Kirche/Halle)"


def _measure_noise_personality(mono: np.ndarray, sr: int) -> float:
    """Misst den Charakter des Rauschbodens — warm oder steril."""
    n_fft = 4096
    if len(mono) < 2 * n_fft:
        return 0.6

    # Rauschboden-Frequenzgang
    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))

    # Warmes Rauschen: fällt mit 1/f ab (mehr Bass)
    # Kaltes/steriles Rauschen: flach oder ansteigend
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    low_mask = freqs < 500
    high_mask = (freqs >= 2000) & (freqs < 8000)

    low_noise = float(np.mean(spec_db[low_mask])) if np.any(low_mask) else -80
    high_noise = float(np.mean(spec_db[high_mask])) if np.any(high_mask) else -80

    tilt = low_noise - high_noise

    if tilt > 15:
        return 0.90  # Starkes Bass-Rauschen = sehr warm/analog
    elif tilt > 8:
        return 0.75  # Warm
    elif tilt > 3:
        return 0.55  # Neutral
    elif tilt > 0:
        return 0.35  # Kalt
    else:
        return 0.15  # Steril — kein Rauschboden-Charakter


def _measure_dynamic_character(mono: np.ndarray, sr: int) -> float:
    """Misst epochentypische Dynamik."""
    win = int(0.1 * sr)
    if len(mono) < 10 * win:
        return 0.6

    rms_vals = [float(np.sqrt(np.mean(mono[i : i + win] ** 2))) for i in range(0, len(mono) - win, win)]
    rms_db = 20.0 * np.log10(np.array(rms_vals) + 1e-12)

    dr = float(np.max(rms_db) - np.percentile(rms_db, 10))

    # Epochentypische Dynamik-Bereiche
    if dr > 50:
        return 0.90  # Sehr dynamisch — Klassik/Jazz
    elif dr > 35:
        return 0.80  # Natürliche Dynamik
    elif dr > 20:
        return 0.55  # Moderate Kompression
    elif dr > 10:
        return 0.30  # Stark komprimiert
    else:
        return 0.10  # Loudness-War


def _measure_emotional_temperature(mono: np.ndarray, sr: int) -> float:
    """Misst die emotionale Wärme/Kälte des Klangs."""
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.6

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Wärme-Indikatoren
    bass_mask = (freqs >= 60) & (freqs <= 300)
    presence_mask = (freqs >= 2000) & (freqs <= 5000)

    bass_energy = np.sum(spec[bass_mask] ** 2) if np.any(bass_mask) else 0.0
    presence_energy = np.sum(spec[presence_mask] ** 2) if np.any(presence_mask) else 0.0
    total = np.sum(spec**2) + 1e-12

    bass_ratio = bass_energy / total
    presence_ratio = presence_energy / total

    # Warm = viel Bass, wenig Präsenz (entspannt, nah)
    # Kalt = wenig Bass, viel Präsenz (distanziert, analytisch)
    warmth = bass_ratio * 3.0 - presence_ratio * 2.0
    return float(np.clip(warmth * 10.0 + 0.5, 0.0, 1.0))
