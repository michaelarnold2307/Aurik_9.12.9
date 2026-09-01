"""
defect_to_audibility.py — §v10.210 Defect-to-Audibility Engine

Berechnet die EXAKTE Phasen-Stärke, die nötig ist, um einen Defekt
unter die psychoakustische Hörschwelle zu drücken.

Prinzip: „Make Defects Inaudible" statt „Do No Harm".
Nicht mehr Stärke als nötig. Nicht weniger als erforderlich.

Architektur:
  1. Defekt-Schwere → geschätztes dB-Level des Defekts
  2. Simultaneous-Masking → wie viel maskiert das Musiksignal?
  3. Target-Reduktion → wie viel dB muss der Defekt sinken?
  4. Phase → Stärke → erwartete dB-Reduktion → benötigte Stärke

Author: Aurik 10 Development
Version: 1.0.0 — §v10.210
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.core.calibration_context import get_calibration_context

logger = logging.getLogger(__name__)

MATERIAL_EXPECTED_BW = 20000


def _resolve_transfer_chain_depth(value: int | None) -> int:
    """§G86: Bezieht transfer_chain_depth aus dem CalibrationContext statt einem stillen Default."""
    if value is not None:
        return value
    ctx = get_calibration_context()
    return ctx.transfer_chain_depth if ctx is not None else 1


# ---------------------------------------------------------------------------
# Defect → dB-Level Mapping (psychoakustisch kalibriert)
# ---------------------------------------------------------------------------
# Jeder Defekttyp hat ein charakteristisches dB-Level bei severity=1.0.
# Bei niedrigerer Severity wird linear interpoliert.
# Quellen: ITU-R BS.1770-4, Moore (2003), Zwicker (1961)

DEFECT_DB_AT_SEVERITY_1: dict[str, float] = {
    # Breitband-Rauschen: −55 dBFS bei Kassette, −96 dBFS bei CD
    "high_freq_noise": 35.0,  # Hiss/Rauschen: typ. 20-40 dB über Noise-Floor
    "hiss": 32.0,
    "tape_hiss": 35.0,
    "modulation_noise": 28.0,  # Dolby-Atmung: 15-30 dB
    "surface_noise": 30.0,  # Vinyl-Oberflächenrauschen
    # Impulsive Defekte
    "clicks": 45.0,  # Einzel-Click: 30-50 dB über Umgebung
    "crackle": 35.0,  # Knistern: 20-40 dB
    "dropout": 30.0,  # Pegel-Einbruch: 20-40 dB
    "dropout_oxide": 30.0,
    # Tonale Defekte
    "hum": 25.0,  # 50/60 Hz Brumm: 15-35 dB
    "rumble": 20.0,  # Subsonic: 10-30 dB
    # Zeitvariante Defekte
    "wow": 18.0,  # Pitch-Modulation: 5-25 dB eff. SNR
    "flutter": 18.0,
    "wow_flutter": 18.0,
    # Bandbreiten-Verlust
    "bandwidth_loss": 30.0,  # HF-Verlust: 10-40 dB bei 10+ kHz
    "hf_loss": 30.0,
    # Digitale Artefakte
    "quantization_noise": 20.0,
    "compression_artifacts": 15.0,
    "digital_artifacts": 15.0,
    # Clipping
    "clipping": 40.0,
    "digital_clip": 40.0,
    "soft_saturation": 25.0,
    # Hall/Raum
    "reverb_excess": 20.0,
    # Sonstige
    "bias_error": 18.0,
    "azimuth_error": 12.0,
    "phase_issues": 12.0,
    "stereo_imbalance": 10.0,
    "groove_echo": 15.0,
    "inner_groove_distortion": 20.0,
    "print_through": 15.0,
    "sticky_shed_residue": 25.0,
    "tape_head_clog": 28.0,
    "tape_head_level_dip": 22.0,
    "tape_splice_artifact": 35.0,
    "transport_bump": 25.0,
    "motor_interference": 20.0,
    "dolby_nr_mismatch": 15.0,
    "cassette_azimuth_tolerance": 12.0,
    "lacquer_disc_degradation": 30.0,
    "generation_loss": 25.0,
    "jitter_artifacts": 12.0,
    "nr_breathing_artifact": 20.0,
    "room_mode_resonance": 18.0,
    "multiband_wow_flutter": 20.0,
    "crosstalk": 15.0,
}

# ---------------------------------------------------------------------------
# Phase → dB-Reduktion bei Strength=1.0 (psychoakustisch kalibriert)
# ---------------------------------------------------------------------------
# Gemessen als erwartete Defekt-Reduktion in dB bei maximaler Phasen-Stärke.

PHASE_MAX_DB_REDUCTION: dict[str, float] = {
    # Phase-ID → max dB Reduktion bei strength=1.0
    "phase_01_click_removal": 50.0,
    "phase_02_hum_removal": 40.0,
    "phase_03_denoise": 25.0,  # OMLSA/IMCRA: 15-25 dB SNR-Verbesserung
    "phase_04_eq_correction": 12.0,
    "phase_05_rumble_filter": 30.0,
    "phase_06_frequency_restoration": 20.0,  # SBR: 10-20 dB HF-Rekonstruktion
    "phase_07_harmonic_restoration": 15.0,
    "phase_08_transient_preservation": 8.0,  # Erhalt, nicht Reduktion
    "phase_09_crackle_removal": 40.0,
    "phase_12_wow_flutter_fix": 25.0,
    "phase_14_phase_correction": 10.0,
    "phase_15_stereo_balance": 8.0,
    "phase_16_final_eq": 10.0,
    "phase_18_noise_gate": 20.0,
    "phase_19_de_esser": 15.0,
    "phase_20_reverb_reduction": 20.0,
    "phase_23_spectral_repair": 30.0,
    "phase_24_dropout_repair": 35.0,
    "phase_25_azimuth_correction": 12.0,
    "phase_26_dynamic_range_expansion": 10.0,
    "phase_27_click_pop_removal": 50.0,
    "phase_28_surface_noise_profiling": 25.0,
    "phase_29_tape_hiss_reduction": 30.0,  # OMLSA/IMCRA: 20-30 dB
    "phase_31_speed_pitch_correction": 15.0,
    "phase_36_transient_shaper": 8.0,
    "phase_39_air_band_enhancement": 10.0,
    "phase_40_loudness_normalization": 6.0,
    "phase_43_ml_deesser": 15.0,
    "phase_47_truepeak_limiter": 3.0,
    "phase_49_advanced_dereverb": 25.0,
    "phase_50_spectral_repair": 25.0,
    "phase_54_transparent_dynamics": 10.0,
    "phase_56_spectral_band_gap_repair": 20.0,
    "phase_57_print_through_reduction": 20.0,
    "phase_59_modulation_noise_reduction": 25.0,
    "phase_60_inner_groove_distortion_repair": 20.0,
    "phase_61_groove_echo_cancellation": 20.0,
    "phase_64_tape_splice_repair": 35.0,
    "phase_65_vocal_naturalness_restoration": 10.0,
}

# ---------------------------------------------------------------------------
# Defekt → Primär-Phase Mapping (1:1 von defect_phase_mapper)
# ---------------------------------------------------------------------------
# Redundanz ist gewollt — diese Engine ist autark, keine Import-Abhängigkeit.

DEFECT_TO_PRIMARY_PHASE: dict[str, str] = {
    "clicks": "phase_01_click_removal",
    "hum": "phase_02_hum_removal",
    "high_freq_noise": "phase_03_denoise",
    "hiss": "phase_29_tape_hiss_reduction",
    "tape_hiss": "phase_29_tape_hiss_reduction",
    "surface_noise": "phase_28_surface_noise_profiling",
    "crackle": "phase_09_crackle_removal",
    "dropout": "phase_24_dropout_repair",
    "dropout_oxide": "phase_24_dropout_repair",
    "wow": "phase_12_wow_flutter_fix",
    "flutter": "phase_12_wow_flutter_fix",
    "wow_flutter": "phase_12_wow_flutter_fix",
    "multiband_wow_flutter": "phase_12_wow_flutter_fix",
    "modulation_noise": "phase_59_modulation_noise_reduction",
    "rumble": "phase_05_rumble_filter",
    "quantization_noise": "phase_03_denoise",
    "compression_artifacts": "phase_50_spectral_repair",
    "digital_artifacts": "phase_23_spectral_repair",
    "digital_clip": "phase_23_spectral_repair",
    "clipping": "phase_23_spectral_repair",
    "soft_saturation": "phase_23_spectral_repair",
    "reverb_excess": "phase_20_reverb_reduction",
    "bandwidth_loss": "phase_06_frequency_restoration",
    "hf_loss": "phase_06_frequency_restoration",
    "bias_error": "phase_04_eq_correction",
    "azimuth_error": "phase_25_azimuth_correction",
    "phase_issues": "phase_14_phase_correction",
    "stereo_imbalance": "phase_15_stereo_balance",
    "groove_echo": "phase_61_groove_echo_cancellation",
    "inner_groove_distortion": "phase_60_inner_groove_distortion_repair",
    "print_through": "phase_57_print_through_reduction",
    "sticky_shed_residue": "phase_24_dropout_repair",
    "tape_head_clog": "phase_24_dropout_repair",
    "tape_head_level_dip": "phase_54_transparent_dynamics",
    "tape_splice_artifact": "phase_64_tape_splice_repair",
    "transport_bump": "phase_08_transient_preservation",
    "motor_interference": "phase_02_hum_removal",
    "dolby_nr_mismatch": "phase_04_eq_correction",
    "cassette_azimuth_tolerance": "phase_25_azimuth_correction",
    "nr_breathing_artifact": "phase_59_modulation_noise_reduction",
    "room_mode_resonance": "phase_04_eq_correction",
    "generation_loss": "phase_06_frequency_restoration",
    "jitter_artifacts": "phase_23_spectral_repair",
    "lacquer_disc_degradation": "phase_09_crackle_removal",
    "crosstalk": "phase_14_phase_correction",
}


# ---------------------------------------------------------------------------
# Psychoakustisches Masking-Modell
# ---------------------------------------------------------------------------


def _signal_level_in_defect_band(audio: np.ndarray, sr: int, defect_type: str) -> float:
    """Signalpegel (dB) im Defekt-Frequenzband."""
    defect_band = _defect_frequency_band(defect_type)
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=-1)
    frame_samples = int(0.05 * sr)
    centre = len(mono) // 2
    start = max(0, centre - frame_samples // 2)
    segment = mono[start : start + frame_samples]
    if len(segment) < 256:
        return -60.0
    n_fft = min(2048, len(segment))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    spectrum = np.abs(np.fft.rfft(segment[:n_fft] * np.hanning(n_fft)))
    band_mask = (freqs >= defect_band[0]) & (freqs <= defect_band[1])
    if not np.any(band_mask):
        return -60.0
    band_rms = float(np.sqrt(np.mean(spectrum[band_mask] ** 2) + 1e-20))
    return float(20.0 * np.log10(band_rms + 1e-10))


def _estimate_noise_floor(audio: np.ndarray, sr: int) -> float:
    """Schätzt den Rauschboden (dBFS) via 5%-Perzentil der Kurzzeit-RMS."""
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=-1)
    frame_len = int(0.05 * sr)
    hop = frame_len // 2
    rms_vals = []
    for i in range(0, len(mono) - frame_len, hop):
        frame = mono[i : i + frame_len]
        rms_vals.append(float(np.sqrt(np.mean(frame**2) + 1e-20)))
    if not rms_vals:
        return -60.0
    noise_rms = float(np.percentile(rms_vals, 5))
    return float(20.0 * np.log10(noise_rms + 1e-10))


def compute_simultaneous_masking(
    audio: np.ndarray,
    sr: int,
    defect_type: str,
    *,
    frame_ms: float = 50.0,
) -> float:
    """Berechnet die Maskierungsschwelle im Defekt-Frequenzbereich (dB).

    Kombiniert zwei Mechanismen:
    1. In-Band-Masking: Signal im Defektband maskiert Defekt im selben Band
    2. Cross-Band-Masking: Dominante Signalenergie streut in andere Bänder

    Modell nach ISO 11172-3 + ERB-Skalierung (Glasberg & Moore 1990).

    Returns:
        Maskierungsschwelle in dB. Defekte UNTER diesem Pegel sind unhörbar.
    """
    defect_band = _defect_frequency_band(defect_type)

    mono = audio if audio.ndim == 1 else np.mean(audio, axis=-1)
    frame_samples = int(frame_ms / 1000.0 * sr)
    centre = len(mono) // 2
    start = max(0, centre - frame_samples // 2)
    segment = mono[start : start + frame_samples]

    if len(segment) < 256:
        return -80.0

    n_fft = min(2048, len(segment))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    win = np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(segment[:n_fft] * win))
    spec_db = 20.0 * np.log10(spectrum + 1e-10)

    # --- 1. In-Band-Masking ---
    band_mask = (freqs >= defect_band[0]) & (freqs <= defect_band[1])
    if np.any(band_mask):
        in_band_db = float(np.max(spec_db[band_mask]))
        in_band_threshold = in_band_db - 20.0  # ~20 dB unter In-Band-Peak
    else:
        in_band_threshold = -100.0

    # --- 2. Cross-Band-Masking (Spreading) ---
    peak_bin = int(np.argmax(spectrum))
    peak_freq = freqs[peak_bin]
    peak_db = spec_db[peak_bin]
    defect_centre = (defect_band[0] + defect_band[1]) / 2.0

    bark_peak = _hz_to_bark(peak_freq)
    bark_defect = _hz_to_bark(defect_centre)
    bark_distance = abs(bark_peak - bark_defect)

    # ISO 11172-3: Spreading = 14.5 + bark_distance × 7.5 dB
    cross_band_threshold = peak_db - (14.5 + bark_distance * 7.5)

    # Die effektive Maskierungsschwelle ist das MAXIMUM beider Mechanismen
    masking_threshold_db = max(in_band_threshold, cross_band_threshold)

    # --- 3. Temporale Maskierung (Forward + Backward) ---
    _temporal_boost = _compute_temporal_masking_boost(mono, sr, defect_band)
    masking_threshold_db = max(masking_threshold_db, _temporal_boost)

    return float(np.clip(masking_threshold_db, -80.0, 120.0))


def _compute_temporal_masking_boost(audio: np.ndarray, sr: int, defect_band: tuple[float, float]) -> float:
    """Temporale Maskierung: Forward (200ms) + Backward (5ms) nach ISO 11172-3.

    Nach einem lauten Transienten ist das Ohr für ~200 ms desensibilisiert.
    Defekte in diesem Zeitfenster sind teilweise maskiert.

    Returns:
        Zusätzliche Maskierungsschwelle in dB durch temporale Effekte.
    """
    if len(audio) < sr // 10:
        return -80.0

    # Kurzzeit-RMS-Hüllkurve (10ms Fenster)
    frame_len = int(0.010 * sr)
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop
    if n_frames < 4:
        return -80.0

    rms_frames = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop
        rms_frames[i] = float(np.sqrt(np.mean(audio[s : s + frame_len] ** 2) + 1e-12))

    rms_db = 20.0 * np.log10(rms_frames + 1e-10)

    # Transienten finden: Peaks > 15 dB über lokalem Median
    local_median = np.zeros(n_frames)
    for i in range(n_frames):
        lo = max(0, i - 5)
        hi = min(n_frames, i + 6)
        local_median[i] = np.median(rms_db[lo:hi])

    transient_mask = rms_db > (local_median + 15.0)
    if not np.any(transient_mask):
        return -80.0

    # Forward-Masking-Kurve: exponentieller Abfall über 200 ms
    fwd_mask_duration_frames = int(0.200 * sr / hop)  # 200 ms
    temporal_boost_db = np.full(n_frames, -80.0)

    for i in np.where(transient_mask)[0]:
        peak_db = rms_db[i]
        for offset in range(1, min(fwd_mask_duration_frames, n_frames - i)):
            # Exponentieller Abfall: 10 dB / 100 ms
            decay_db = 10.0 * offset / (0.100 * sr / hop)
            boost = peak_db - decay_db - 20.0  # 20 dB unter Transienten-Pegel
            if boost > temporal_boost_db[i + offset]:
                temporal_boost_db[i + offset] = boost
        # Backward masking: 5 ms vor dem Transienten
        for offset in range(1, min(int(0.005 * sr / hop), i)):
            boost = peak_db - 30.0  # Rückwärtsmaskierung ist schwächer
            if boost > temporal_boost_db[i - offset]:
                temporal_boost_db[i - offset] = boost

    # Maximaler temporaler Boost über alle Frames
    return float(np.max(temporal_boost_db))


def _defect_frequency_band(defect_type: str) -> tuple[float, float]:
    """Gibt das charakteristische Frequenzband eines Defekttyps zurück (Hz)."""
    bands = {
        "high_freq_noise": (6000, MATERIAL_EXPECTED_BW),
        "hiss": (6000, MATERIAL_EXPECTED_BW),
        "tape_hiss": (6000, MATERIAL_EXPECTED_BW),
        "surface_noise": (4000, MATERIAL_EXPECTED_BW),
        "modulation_noise": (2000, 8000),
        "clicks": (2000, MATERIAL_EXPECTED_BW),
        "crackle": (2000, 16000),
        "hum": (40, 400),
        "rumble": (10, 120),
        "wow": (0, 20),
        "flutter": (4, 20),
        "bandwidth_loss": (8000, MATERIAL_EXPECTED_BW),
        "hf_loss": (8000, MATERIAL_EXPECTED_BW),
        "dropout": (100, MATERIAL_EXPECTED_BW),
        "dropout_oxide": (100, MATERIAL_EXPECTED_BW),
        "clipping": (100, MATERIAL_EXPECTED_BW),
        "digital_clip": (100, MATERIAL_EXPECTED_BW),
        "reverb_excess": (200, 8000),
    }
    return bands.get(defect_type, (500, 8000))


def _hz_to_bark(freq_hz: float) -> float:
    """Konvertiert Hz in Bark-Skala (Zwicker 1961)."""
    return float(13.0 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500.0) ** 2))


# ---------------------------------------------------------------------------
# Kern-Funktion: Berechnet exakte benötigte Phasen-Stärke
# ---------------------------------------------------------------------------


@dataclass
class AudibilityTarget:
    """Ergebnis der Defect-to-Audibility-Analyse."""

    defect_type: str
    severity: float  # 0–1 vom DefectScanner
    estimated_db: float  # geschätztes Defekt-dB-Level
    masking_db: float  # wie viel maskiert das Signal?
    target_reduction_db: float  # benötigte dB-Reduktion
    primary_phase: str  # Phase-ID
    max_phase_db: float  # max dB-Reduktion dieser Phase bei strength=1.0
    required_strength: float  # benötigte Stärke [0, 1]
    audible: bool  # True wenn Defekt hörbar ist
    reason: str = ""


def required_strength(
    defect_type: str,
    severity: float,
    audio: np.ndarray,
    sr: int,
    *,
    transfer_chain_depth: int | None = None,
    safety_margin_db: float = 3.0,
    stem_audio: np.ndarray | None = None,  # §v10.210: Pro-Stem-Analyse
) -> AudibilityTarget:
    """Berechnet die benötigte Phasen-Stärke für unhörbare Defekt-Reduktion.

    Args:
        defect_type: Defekt-Typ (z.B. "hiss", "clicks", "wow")
        severity: Defekt-Schwere 0–1 vom DefectScanner
        audio: Original-Audio (float32, mono oder stereo)
        sr: Sample-Rate
        transfer_chain_depth: Tiefe der Transfer-Kette (1 = Studio-Master)
        safety_margin_db: dB-Sicherheitsabstand unter Maskierungsschwelle
        stem_audio: §v10.210 Optionales Stem-Audio für pro-Stem-Analyse.
                   Wenn gesetzt, wird die Maskierung NUR auf dem Stem berechnet.
                   BSRoFormer liefert z.B. "vocals" oder "instruments".

    Args:
        defect_type: Defekt-Typ (z.B. "hiss", "clicks", "wow")
        severity: Defekt-Schwere 0–1 vom DefectScanner
        audio: Original-Audio (float32, mono oder stereo)
        sr: Sample-Rate
        transfer_chain_depth: Tiefe der Transfer-Kette (1 = Studio-Master)
        safety_margin_db: dB-Sicherheitsabstand unter Maskierungsschwelle

    Returns:
        AudibilityTarget mit required_strength ∈ [0, 1]
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    depth = max(1, _resolve_transfer_chain_depth(transfer_chain_depth))

    # §v10.210: Stem-Aware Maskierung — wenn Stem-Audio vorhanden,
    # berechne Maskierung NUR auf dem Stem (z.B. Vocals-only für De-Essing)
    _masking_audio = stem_audio if stem_audio is not None else audio

    # 1. Defekt-dB-Level (absolut, dBFS) aus Severity und Noise-Floor
    max_db_above_noise = DEFECT_DB_AT_SEVERITY_1.get(defect_type, 30.0)
    noise_floor_db = _estimate_noise_floor(audio, sr)
    estimated_db = noise_floor_db + max_db_above_noise * severity

    # 2. Simultaneous Masking — Schwelle relativ zum Signal im Defektband
    # §v10.210: Stem-Aware: Maskierung auf Stem-Audio (oder Full-Mix)
    masking_db = compute_simultaneous_masking(_masking_audio, sr, defect_type)
    signal_in_band_db = _signal_level_in_defect_band(_masking_audio, sr, defect_type)
    # Maskierungs-Hub: wie viel dB unter dem Signalpegel wird maskiert?
    masking_headroom_db = signal_in_band_db - masking_db  # Positiv = viel Maskierung

    # 3. Target-Reduktion: Defekt muss unter Maskierungsschwelle + Sicherheitsabstand
    target_reduction_db = max(0.0, estimated_db - masking_headroom_db + safety_margin_db)

    # 4. Phase finden (case-insensitive lookup)
    _dt_lower = defect_type.lower()
    phase_id = DEFECT_TO_PRIMARY_PHASE.get(_dt_lower, DEFECT_TO_PRIMARY_PHASE.get(defect_type, ""))
    max_phase_db = PHASE_MAX_DB_REDUCTION.get(phase_id, 20.0)

    # 5. Stärke berechnen: linear aus dB-Reduktion / max dB-Reduktion
    if max_phase_db > 0 and target_reduction_db > 0:
        required = target_reduction_db / max_phase_db
        # Depth-Korrektur: tiefere Ketten brauchen proportional mehr Stärke
        # weil das Signal-Rausch-Verhältnis schlechter ist
        if depth >= 4:
            required *= 1.30  # +30% für 4-stufige Kette
        elif depth >= 3:
            required *= 1.15
        required = float(np.clip(required, 0.0, 1.0))
    else:
        required = 0.0

    audible = target_reduction_db > 0.5  # > 0.5 dB = hörbar

    reason_parts = []
    if not audible:
        reason_parts.append("maskiert — keine Restauration nötig")
    else:
        reason_parts.append(f"Defekt {estimated_db:.1f} dB, Signal maskiert {masking_db:.1f} dB")
        reason_parts.append(f"→ {target_reduction_db:.1f} dB Reduktion nötig")
        reason_parts.append(f"→ {phase_id} bei Stärke {required:.2f}")

    return AudibilityTarget(
        defect_type=defect_type,
        severity=severity,
        estimated_db=round(estimated_db, 1),
        masking_db=round(masking_db, 1),
        target_reduction_db=round(target_reduction_db, 1),
        primary_phase=phase_id,
        max_phase_db=max_phase_db,
        required_strength=round(required, 3),
        audible=audible,
        reason="; ".join(reason_parts),
    )


def compute_all_strengths(
    defect_scores: dict[str, float],
    audio: np.ndarray,
    sr: int,
    *,
    transfer_chain_depth: int | None = None,
    min_severity: float = 0.05,
) -> dict[str, AudibilityTarget]:
    """Berechnet benötigte Stärken für alle Defekte.

    Args:
        defect_scores: {defect_type: severity} vom DefectScanner
        audio: Original-Audio
        sr: Sample-Rate
        transfer_chain_depth: Tiefe der Transfer-Kette
        min_severity: Defekte unter dieser Schwelle werden ignoriert

    Returns:
        {defect_type: AudibilityTarget}
    """
    results: dict[str, AudibilityTarget] = {}
    for defect_type, severity in defect_scores.items():
        if severity < min_severity:
            continue
        target = required_strength(defect_type, severity, audio, sr, transfer_chain_depth=transfer_chain_depth)
        if target.audible:
            results[defect_type] = target
    return results


def phase_strength_map(
    defect_scores: dict[str, float],
    audio: np.ndarray,
    sr: int,
    *,
    transfer_chain_depth: int | None = None,
) -> dict[str, float]:
    """Berechnet {phase_id: strength} für alle hörbaren Defekte.

    Wenn mehrere Defekte auf die gleiche Phase zeigen, wird die
    MAXIMALE benötigte Stärke verwendet.

    Returns:
        {phase_id: strength} — direkt als kwargs für Phase-Wrapper nutzbar
    """
    all_targets = compute_all_strengths(defect_scores, audio, sr, transfer_chain_depth=transfer_chain_depth)

    phase_strengths: dict[str, float] = {}
    for target in all_targets.values():
        pid = target.primary_phase
        if pid:
            phase_strengths[pid] = max(phase_strengths.get(pid, 0.0), target.required_strength)

    return phase_strengths


def is_defect_still_audible(
    defect_type: str,
    severity: float,
    audio_after: np.ndarray,
    sr: int,
    *,
    transfer_chain_depth: int | None = None,
) -> tuple[bool, float]:
    """§v10.210 Feedback-Loop: Prüft ob ein Defekt nach der Phase noch hörbar ist.

    Berechnet die benötigte Stärke auf dem VERARBEITETEN Audio.
    Wenn required_strength > 0, ist der Defekt noch hörbar → Phase sollte
    mit höherer Stärke wiederholt werden.

    Returns:
        (still_audible, redo_strength): True wenn Defekt noch hörbar,
        und die benötigte Stärke für eine Wiederholung.
    """
    target = required_strength(defect_type, severity, audio_after, sr, transfer_chain_depth=transfer_chain_depth)
    if target.audible and target.required_strength > 0.02:
        return True, target.required_strength
    return False, 0.0
