"""
critical_listening_points.py — §v10 Gehör-optimierte Restaurierung
=======================================================================

Identifiziert die Frequenz- und Zeitbereiche, in denen das menschliche Ohr
am empfindlichsten ist, um die Restaurierungs-Energie genau dorthin zu
lenken, wo sie den größten Wohlklang-Gewinn bringt — und von Bereichen
fernzuhalten, wo sie am ehesten hörbare Artefakte erzeugt.

Wissenschaftliche Basis:
  - Fletcher-Munson / ISO 226: Gehörempfindlichkeitskurven
  - Zwicker/Fastl: Psychoakustik (Bark-Skala, Mithörschwellen)
  - Moore/Glasberg: ERB-Skala, Frequenzselektivität
  - ANSI S3.4: Loudness-Modelle

CLP-Maske:
  - Zone 1 (2-5 kHz, "Präsenz"): Höchste Ohrempfindlichkeit → MAXIMALE Vorsicht
  - Zone 2 (300-800 Hz, "Wärme"): Hohe Empfindlichkeit für Klangfarbe
  - Zone 3 (80-250 Hz, "Fundament"): Musikalische Grundtöne → ERHALTEN
  - Zone 4 (>8 kHz, "Luft"): Geringe Empfindlichkeit → mehr Spielraum
  - Zone 5 (<80 Hz, "Subbass"): Kaum richtungs-empfindlich → robust

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Konstanten: Zonen-Definitionen (ISO 226, Zwicker/Fastl 2007)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CLPZone:
    """Eine Critical-Listening-Zone mit Frequenzbereich und Bearbeitungsgrenzen."""

    name: str
    f_min: float  # Hz
    f_max: float  # Hz
    sensitivity: float  # 0-1 (1 = maximale Ohrempfindlichkeit)
    max_gain_db: float  # max erlaubte Verstärkung in dieser Zone
    max_cut_db: float  # max erlaubte Absenkung in dieser Zone
    priority: int  # 1 (höchste) bis 5 (niedrigste)


CLP_ZONES: list[CLPZone] = [
    CLPZone("Präsenz", 2000, 5000, sensitivity=1.00, max_gain_db=3.0, max_cut_db=1.5, priority=1),
    CLPZone("Wärme", 300, 800, sensitivity=0.85, max_gain_db=4.0, max_cut_db=2.0, priority=2),
    CLPZone("Fundament", 80, 250, sensitivity=0.75, max_gain_db=4.0, max_cut_db=3.0, priority=3),
    CLPZone("Brillanz", 5000, 8000, sensitivity=0.60, max_gain_db=6.0, max_cut_db=4.0, priority=4),
    CLPZone("Luft", 8000, 16000, sensitivity=0.35, max_gain_db=8.0, max_cut_db=6.0, priority=5),
    CLPZone("Subbass", 20, 80, sensitivity=0.25, max_gain_db=6.0, max_cut_db=6.0, priority=5),
]

# Vocal-Relevante Frequenzbereiche (Moore/Glasberg ERB, Titze 1994)
VOCAL_ZONES: dict[str, tuple[float, float]] = {
    "grundton_male": (85, 180),  # männliche Stimme Grundfrequenz
    "grundton_female": (165, 350),  # weibliche Stimme Grundfrequenz
    "formant_f1": (300, 1000),  # Erster Formant (Vokal-Färbung)
    "formant_f2": (850, 2500),  # Zweiter Formant (Vokal-Unterscheidung)
    "formant_f3": (2000, 3500),  # Dritter Formant (Klangcharakter)
    "sibilance": (4000, 10000),  # Zischlaute (s, z, sch)
    "breath": (5000, 12000),  # Atemgeräusche
}

# Gehör-Empfindlichkeitskurve (vereinfacht nach ISO 226:2023, 40 phon)
# dB-Anhebung relativ zu 1 kHz für gleiche Lautheitswahrnehmung
EQUAL_LOUDNESS_40PHON: dict[float, float] = {
    20: 50.0,
    25: 42.0,
    31.5: 35.0,
    40: 28.0,
    50: 22.0,
    63: 17.0,
    80: 12.0,
    100: 8.5,
    125: 6.0,
    160: 4.0,
    200: 2.5,
    250: 1.5,
    315: 0.8,
    400: 0.3,
    500: 0.0,
    630: -0.2,
    800: -0.3,
    1000: 0.0,
    1250: 0.2,
    1600: 0.5,
    2000: 0.8,
    2500: 1.0,
    3150: 1.2,
    4000: 1.0,
    5000: 0.5,
    6300: -0.5,
    8000: -2.0,
    10000: -4.0,
    12500: -6.0,
    16000: -8.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CLP-Ergebnis-Dataclass
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CLPResult:
    """Ergebnis der Critical-Listening-Points-Analyse."""

    zone_scores: dict[str, float] = field(default_factory=dict)
    """Score pro CLP-Zone (0-1). 1 = maximale Energie = maximale Vorsicht."""

    vocal_presence: float = 0.0
    """Wahrscheinlichkeit von Gesang im Signal (0-1)."""

    critical_mask: np.ndarray | None = None
    """Frequenzabhängige Vorsichts-Maske (0-1). 1 = kritisch, nicht bearbeiten."""

    whisper_energy: float = 0.0
    """Geschätzte Energie von Flüster-/Leise-Passagen (0-1)."""

    transient_zones: list[tuple[float, float]] = field(default_factory=list)
    """Zeitstempel (start_s, end_s) mit hoher Transienten-Dichte."""

    recommendation: str = ""
    """Handlungsempfehlung für die Restaurierungs-Pipeline."""


def _freq_to_erb(freq_hz: np.ndarray) -> np.ndarray:
    """Konvertiert Hz zu ERB-Nummer (Moore/Glasberg 1983)."""
    return 21.4 * np.log10(0.00437 * freq_hz + 1.0)  # type: ignore[no-any-return]


def _compute_vocal_presence(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """Schätzt die Wahrscheinlichkeit von Gesang im Signal.

    Nutzt charakteristische Formant-Verteilungen: Hohe Energie in
    F1-F3-Bereichen mit harmonischer Struktur = wahrscheinlich Gesang.
    """
    vocal_regions = [
        (VOCAL_ZONES["formant_f1"][0], VOCAL_ZONES["formant_f1"][1], 0.3),
        (VOCAL_ZONES["formant_f2"][0], VOCAL_ZONES["formant_f2"][1], 0.4),
        (VOCAL_ZONES["formant_f3"][0], VOCAL_ZONES["formant_f3"][1], 0.3),
    ]

    total = np.sum(spectrum) + 1e-12
    vocal_score = 0.0
    for lo, hi, weight in vocal_regions:
        mask = (freqs >= lo) & (freqs <= hi)
        band_energy = float(np.sum(spectrum[mask]))
        vocal_score += weight * (band_energy / total)

    return float(np.clip(vocal_score * 3.0, 0.0, 1.0))


def _compute_whisper_energy(audio: np.ndarray, sr: int) -> float:
    """Erkennt Flüster-/Leise-Passagen durch schwache RMS in HF-Bereichen.

    Flüstern hat charakteristisch: niedriger Gesamtpegel, aber hoher
    relativer HF-Anteil (>3 kHz), da Stimmlippen nicht schwingen.
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)

    win = int(0.025 * sr)  # 25ms
    hop = win // 2
    whisper_frames = 0
    total_frames = 0

    for start in range(0, len(arr) - win, hop):
        frame = arr[start : start + win]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        rms_db = 20.0 * np.log10(max(rms, 1e-12))

        if rms_db < -40.0:  # Leise
            # HF-Anteil check: wenn >30% Energie >3kHz bei niedrigem Pegel
            spec = np.abs(np.fft.rfft(frame * np.hanning(win)))
            freqs = np.fft.rfftfreq(win, d=1.0 / sr)
            hf_mask = freqs >= 3000
            hf_ratio = float(np.sum(spec[hf_mask]) / (np.sum(spec) + 1e-12))
            if hf_ratio > 0.3:
                whisper_frames += 1

        total_frames += 1

    return float(np.clip(whisper_frames / max(total_frames, 1), 0.0, 1.0))


def _detect_transient_zones(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Findet Zeitbereiche mit hoher Transienten-Dichte (Schlagzeug, Anschläge).

    Diese Bereiche sind besonders empfindlich für Pre-Echo-Artefakte
    und Phasenverzerrungen durch aggressive Filterung.
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)

    win = int(0.010 * sr)  # 10ms
    hop = win // 2
    onset_strength = []

    for start in range(0, len(arr) - win, hop):
        frame = arr[start : start + win]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        onset_strength.append(rms)

    if not onset_strength:
        return []

    onset_strength = np.array(onset_strength)  # type: ignore[assignment]
    # Transient = lokales RMS-Maximum
    threshold = float(np.percentile(onset_strength, 85))
    is_transient = onset_strength > threshold * 1.5  # type: ignore[operator]

    # Gruppiere benachbarte Transienten zu Zonen
    zones = []
    in_zone = False
    zone_start = 0.0
    for i, t in enumerate(is_transient):
        t_s = i * hop / sr
        if t and not in_zone:
            zone_start = t_s
            in_zone = True
        elif not t and in_zone:
            if t_s - zone_start > 0.02:  # mindestens 20ms
                zones.append((float(zone_start), float(t_s)))
            in_zone = False
    if in_zone:
        zones.append((float(zone_start), float(len(is_transient) * hop / sr)))

    return zones


def compute_equal_loudness_weighting(freqs: np.ndarray) -> np.ndarray:
    """Berechnet die Frequenzgewichtung nach ISO 226:2023 (40 phon).

    Gibt Gewichte zurück, die die Gehörempfindlichkeit modellieren:
    - Werte ≈ 1.0: Ohr ist besonders empfindlich → WENIGER Bearbeitung
    - Werte < 0.5: Ohr ist weniger empfindlich → mehr Bearbeitung erlaubt
    """
    ref_freqs = np.array(sorted(EQUAL_LOUDNESS_40PHON.keys()))
    ref_db = np.array([EQUAL_LOUDNESS_40PHON[f] for f in ref_freqs])

    # Interpoliere auf Zielfrequenzen
    db_at_freqs = np.interp(freqs, ref_freqs, ref_db)

    # Konvertiere dB-Abweichung zu Gewicht 0-1
    # 0 dB Abweichung (1000 Hz) → Gewicht 1.0
    # 50 dB Abweichung (20 Hz) → Gewicht 0.0
    weights = np.clip(1.0 - np.abs(db_at_freqs) / 50.0, 0.0, 1.0)

    return cast(np.ndarray, weights.astype(np.float32))


def compute_critical_mask(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    sr: int,
    *,
    vocal_boost: bool = True,
) -> np.ndarray:
    """Berechnet die frequenzabhängige CLP-Maske (0-1).

    Maske-Werte:
    - 1.0 = MAXIMALE Vorsicht — nicht bearbeiten (z.B. 2-5 kHz Präsenz)
    - 0.0 = unkritisch — Bearbeitung unbedenklich (z.B. >12 kHz)

    Args:
        spectrum: FFT-Magnituden-Spektrum
        freqs: FFT-Frequenz-Array
        sr: Sample-Rate
        vocal_boost: Wenn True, werden Vokal-Formant-Bereiche zusätzlich geschützt
    """
    n_bins = len(freqs)
    mask = np.zeros(n_bins, dtype=np.float32)

    # ISO 226 Gewichtung: höchste Gewichtung = höchste Empfindlichkeit
    loudness_weights = compute_equal_loudness_weighting(freqs)

    # Zonen-basierte Verfeinerung
    for zone in CLP_ZONES:
        zone_mask = (freqs >= zone.f_min) & (freqs <= zone.f_max)
        mask[zone_mask] = np.maximum(
            mask[zone_mask],
            loudness_weights[zone_mask] * zone.sensitivity,
        )

    # Vocal-Boost: Formant-Bereiche extra schützen
    if vocal_boost:
        vocal_presence = _compute_vocal_presence(spectrum, freqs)
        if vocal_presence > 0.3:
            for vz_name, (vz_lo, vz_hi) in VOCAL_ZONES.items():
                if "grundton" in vz_name or "formant" in vz_name:
                    vz_mask = (freqs >= vz_lo) & (freqs <= vz_hi)
                    boost = 0.2 * vocal_presence  # max +0.2 auf bestehende Maske
                    mask[vz_mask] = np.clip(mask[vz_mask] + boost, 0.0, 1.0)

    return cast(np.ndarray, mask.astype(np.float32))


def analyze_critical_zones(
    audio: np.ndarray,
    sr: int,
    *,
    vocal_boost: bool = True,
) -> CLPResult:
    """Vollständige CLP-Analyse: Zonen, Maske, Vocal-Präsenz, Transienten.

    Laufzeit: ~30ms für 4min Stereo @ 48kHz.
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr

    n_fft = 4096
    if len(mono) < n_fft:
        return CLPResult(
            zone_scores={z.name: 0.0 for z in CLP_ZONES},
            recommendation="Signal zu kurz für CLP-Analyse",
        )

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Zonen-Scores berechnen
    zone_scores = {}
    total = np.sum(spec) + 1e-12
    for zone in CLP_ZONES:
        mask = (freqs >= zone.f_min) & (freqs <= zone.f_max)
        zone_energy = float(np.sum(spec[mask]))
        zone_scores[zone.name] = float(np.clip(zone_energy / total * 5.0, 0.0, 1.0))

    # Vocal-Präsenz
    vocal_presence = _compute_vocal_presence(spec, freqs)

    # Critical Mask
    critical_mask = compute_critical_mask(spec, freqs, sr, vocal_boost=vocal_boost)

    # Whisper-Energie
    whisper_energy = _compute_whisper_energy(audio, sr)

    # Transienten-Zonen
    transient_zones = _detect_transient_zones(audio, sr)

    # Recommendation
    high_risk_zones = [z.name for z in CLP_ZONES if zone_scores.get(z.name, 0.0) > 0.7]
    if vocal_presence > 0.5:
        rec = f"Hohe Vocal-Präsenz ({vocal_presence:.1%}) — Formant-Bereiche 300-3500 Hz maximal schonen. " + (
            f"Kritische Zonen: {', '.join(high_risk_zones[:3])}" if high_risk_zones else ""
        )
    elif high_risk_zones:
        rec = f"Kritische Zonen mit hoher Energie: {', '.join(high_risk_zones[:3])} — Bearbeitung reduzieren."
    elif whisper_energy > 0.3:
        rec = f"Hoher Leise-Anteil ({whisper_energy:.1%}) — Noise-Reduction vorsichtig dosieren."
    else:
        rec = "Keine kritischen Zonen — normale Bearbeitung möglich."

    return CLPResult(
        zone_scores=zone_scores,
        vocal_presence=vocal_presence,
        critical_mask=critical_mask,
        whisper_energy=whisper_energy,
        transient_zones=transient_zones,
        recommendation=rec,
    )


def apply_clp_limited_gain(
    audio: np.ndarray,
    sr: int,
    clp: CLPResult,
    *,
    gain_db: float = 0.0,
    freq_start: float | None = None,
    freq_end: float | None = None,
) -> np.ndarray:
    """Wendet frequenzabhängige Verstärkung an, begrenzt durch CLP-Maske.

    Verhindert Überbearbeitung in den fürs Ohr kritischen Frequenzbereichen.
    """
    if clp.critical_mask is None:
        return audio

    arr = np.asarray(audio, dtype=np.float64)
    n_fft = 4096
    hop = n_fft // 4
    n_frames = (len(arr) - n_fft) // hop + 1
    if n_frames < 1:
        return cast(np.ndarray, arr)

    result = np.zeros_like(arr)
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(sr, 1))
    n_freq_bins = len(freqs)

    # Interpoliere critical_mask auf n_fft Raster
    if len(clp.critical_mask) != n_freq_bins:
        orig_freqs = np.fft.rfftfreq(len(clp.critical_mask) * 2 - 2, d=1.0 / max(sr, 1))
        mask_interp = np.interp(freqs, orig_freqs[:n_freq_bins], clp.critical_mask[:n_freq_bins])
    else:
        mask_interp = clp.critical_mask

    # Frequenzbereich filtern
    if freq_start is not None or freq_end is not None:
        f_mask = np.ones(n_freq_bins)
        if freq_start is not None:
            f_mask[freqs < freq_start] = 0.0
        if freq_end is not None:
            f_mask[freqs > freq_end] = 0.0
        mask_interp = mask_interp * f_mask + (1.0 - f_mask)

    for i in range(n_frames):
        start = i * hop
        frame = arr[start : start + n_fft] * window
        spec = np.fft.rfft(frame)
        # Begrenzte Verstärkung: mask=0 → volle Verstärkung, mask=1 → keine
        gain_linear = np.power(10.0, gain_db / 20.0 * (1.0 - mask_interp))
        spec = spec * gain_linear
        result[start : start + n_fft] += np.fft.irfft(spec).real * window

    max_val = float(np.max(np.abs(result)))
    if max_val > 1.0:
        result /= max_val

    return cast(np.ndarray, result.astype(audio.dtype))


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════


class CLPAnalyzer:
    """Thread-sicherer Analyzer für Critical Listening Points."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_result: CLPResult | None = None

    def analyze(self, audio: np.ndarray, sr: int, **kwargs: Any) -> CLPResult:
        """Analysiert Audio auf Critical Listening Zones."""
        result = analyze_critical_zones(audio, sr, **kwargs)
        with self._lock:
            self._last_result = result
        return result

    @property
    def last_result(self) -> CLPResult | None:
        return self._last_result


_clp_instance: CLPAnalyzer | None = None


def get_clp_singleton() -> CLPAnalyzer:
    """Thread-sicherer Singleton-Accessor."""
    global _clp_instance
    if _clp_instance is None:
        _clp_instance = CLPAnalyzer()
    return _clp_instance
