"""
Causal Defect Reasoner — Aurik 10.0.0
=====================================
Bayesianische Ursachendiagnose für mangelhafte Tonträger-Aufnahmen.
Gegeben eine Menge von Fehlermerkmalen (DefectResult + akustische Signalmerkmale)
und dem Trägermaterial berechnet dieses Modul die posteriore Wahrscheinlichkeit
über 8 mögliche Wurzelursachen und erstellt einen priorisierten Restaurierungsplan.

Das System ist ein diskretisiertes Bayesnetz:
    P(cause | observations, material) ∝ P(observations | cause) · P(cause | material)

Ursachen (11) — Spec §2.4:
    tape_dropout        — Magnetband-Aussetzer (Dropout)
    tape_hiss           — Bandrauschen (thermisches & Partikelrauschen)
    vinyl_crackle       — Vinyl-Knistern (Oberflächendefekte)
    vinyl_warp          — Plattenwellung (Pitch-Instabilität + Intermodulation)
    electrical_hum      — Netzbrumm 50/60 Hz + Obertöne
    head_misalignment   — Tonkopf-Fehlausrichtung (Azimut-Fehler, HF-Verlust)
    dc_offset           — DC-Versatz (einseitiger Betriebspunkt)
    digital_clip        — Digitales Clipping (Harddist, Integer-Overflow)
    soft_saturation     — Röhren-/Tape-Sättigung (gerade Obertöne) — kein Eingriff (BEWAHREN)
    head_wear           — Komplette Frequenzband-Auslöschung durch Kopfverschleiß (→ phase_56)
    print_through       — Magnetisches Tape-Übersprechen / Vorecho (Adaptive Temporal Subtraction)

Ausgabe:
    RestorationPlan mit geordneten Restaurierungsphasen + Parametern

Referenzen:
    - Pearl, Causality (2000) — Bayesnetze
    - Maher, Audio Restoration (IEEE, 1993)
    - Lahat et al., Temporal and Spectral Audio Forensics (2013)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.core.defect_scanner import MaterialType  # §v10.113

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typ-Definitionen
# ---------------------------------------------------------------------------

# 62 Kausal-Ursachen (Spec §2.4): 10 Magnetband + 4 Vinyl + 2 Elektrik + 9 Digital/Codec + 9 v10.0.0
# + 2 Spektral + 2 Stereo + 5 Pitch/Dynamik/Vokal + 1 Vintage + 2 Transport + 12 v10.0.0
CAUSES = [
    # ── Analoge Magnetband-Ursachen ──────────────────────────────────────────
    "tape_dropout",
    "tape_hiss",
    "transport_bump",  # Impulsartige Mikro-Geschwindigkeitssprünge (Kassette/Tape-Holpern) → phase_12
    "print_through",  # Magnetisches Vorecho → Adaptive Temporal Subtraction
    "head_wear",  # Frequenzband-Auslöschung → phase_56_spectral_band_gap_repair
    "head_misalignment",
    "bias_error",  # Falscher Vormagnetisierungsstrom → phase_04 + phase_29
    "wow",  # Tonhöhenschwankung < 0.5 Hz (Motorexzentrizität)
    "flutter",  # Tonhöhenschwankung 0.5–200 Hz (Bandschwankung)
    "wow_flutter",  # Kombiniert wow + flutter
    # ── Vinyl-/Schellack-Ursachen ────────────────────────────────────────────
    "vinyl_crackle",
    "vinyl_warp",
    "riaa_curve_error",  # Falsche Disc-Entzerrungskurve (AES/NAB/FFRR)
    "low_freq_rumble",  # Subsonic-Störung (Plattenteller, Motor)
    # ── Elektrik / Mechanik ──────────────────────────────────────────────────
    "electrical_hum",
    "dc_offset",
    # ── Digital / Codec ──────────────────────────────────────────────────────
    "digital_clip",
    "clipping",  # Generisches Clipping (analog + digital)
    "digital_artifacts",  # Breitband-Codec-Artefakte (Quantisierungsreste, Ringmodulation)
    "compression_artifacts",  # MP3/AAC/OGG Codec-Artefakte (Butterfly-Noise, Birdie)
    "quantization_noise",  # Bit-Tiefe-bedingtes Rauschen (8-Bit, Resampling)
    "jitter_artifacts",  # Zeitgitter-Fehler D/A-Wandlung
    "pre_echo",  # Codec-Pre-Echo vor Transienten (MP3 Long-Window)
    "aliasing",  # ADC-Spiegelfrequenzen bei fehlendem AA-Filter
    "dynamic_compression_excess",  # Loudness-War-Artefakte (Over-Limiting)
    # ── Spektrale Ursachen ───────────────────────────────────────────────────
    "bandwidth_loss",  # HF-Rolloff / Bandbreitenbegrenzung
    "high_freq_noise",  # Hochfrequenzrauschen (distinct from tape_hiss)
    # ── Stereo / Phase ───────────────────────────────────────────────────────
    "stereo_imbalance",  # L/R-Pegelunterschied
    "phase_issues",  # Phasenverschiebung zwischen Kanälen
    # ── Pitch / Dynamik ──────────────────────────────────────────────────────
    "pitch_drift",  # Konstanter Geschwindigkeitsfehler (Motor/Tape-Stretch)
    "reverb_excess",  # Übermäßiger Raumhall
    "transient_smearing",  # Ansatzverschmierung (Comp/Limiter)
    "vocal_harshness",  # Vokale Härte/Verzerrung (2-6 kHz)
    "sibilance",  # Zischlautüberbetonung > 6 kHz
    # ── Vintage (Schutz) ────────────────────────────────────────────────────
    "soft_saturation",  # Tube-/Tape-Sättigung — BEWAHREN, P(phases) = leer
    # ── Transport-Mechanik (v10.0.0) ───────────────────────────────────────
    "tape_start_instability",  # Cassette head engagement + motor startup (first 20 s)
    "tape_head_contact_instability",  # Gradual level dips from head-tape pressure variation / capstan irregularity
    # ── v10.0.0: 12 neue Kausal-Ursachen ────────────────────────────────────
    "modulation_noise",  # Signal-dependent noise modulation (tape media)
    "inner_groove_distortion",  # IGD: THD increasing with groove radius (vinyl/shellac)
    "groove_echo",  # Pre-echo from adjacent groove deformation (~1.8 s @ 33⅓)
    "crosstalk",  # Channel separation degradation in early stereo
    "intermodulation_distortion",  # Nonlinear sum/difference products (amplifier chain)
    "tape_splice_artifact",  # Click + level jump + phase discontinuity at splices
    "hf_remanence_loss",  # Magnetic particle demagnetization over decades
    "stylus_damage",  # Asymmetric distortion from worn/damaged stylus
    "sticky_shed_residue",  # Post-baking tape degradation: level dips + noise bursts
    "multiband_wow_flutter",  # Frequency-dependent speed fluctuations (head gap)
    "generation_loss",  # Cumulative degradation from tape dubbing/transcoding
    "motor_interference",  # Motor harmonics 80–300 Hz from DC/sync motors
    # ── v10.0.0: Pegelveränderung ────────────────────────────────────────────
    "amplitude_drift",  # Gradual level rise/fall over song duration (AGC, oxide drift, motor temp)
    # ── v10.0.0: DefectType→CAUSE-Lücken geschlossen ─────────────────────────
    "clicks",  # DefectType.CLICKS → phase_01 (impulsartige Einzelstörungen)
    "dolby_nr_mismatch",  # DefectType.DOLBY_NR_MISMATCH → phase_04 HF-Shelf-Korrektur
    # (Dolby B/C/S encode ohne Dekodierung)
    "tape_head_level_dip",  # DefectType.TAPE_HEAD_LEVEL_DIP → phase_12/phase_24 (Bandkopf-Kontaktdruckvariation)
    "scrape_flutter",  # DefectType.SCRAPE_FLUTTER → phase_12/phase_31 (hochfrequente Bandführungsmodulation)
    "tape_head_clog",  # DefectType.TAPE_HEAD_CLOG → phase_56/phase_25 (temporäre HF-Auslöschung)
    # ── v10.0.0: 9 neue Kausal-Ursachen — Carrier-Lücken geschlossen ─────────
    # Nahbesprechungseffekt (Richtmikrofon ≤30 cm) → LF +6–12 dB ≤250 Hz; häufig Vokal 1940–1970.
    "proximity_effect_excess",
    # Raumresonanz-Stehwellen 40–200 Hz ≠ diffuser Hall → schmalbandige Q>8-Peaks.
    "room_mode_resonance",
    # Dolby/dbx NR Pumpen/Atmen (korrekt dekodiert) → pulsierender Rauschboden.
    "nr_breathing_artifact",
    "flutter_spectral_sidebands",  # Flutter-Seitenbänder um Spektralpeaks (metallisch) → phase_12 + phase_23
    # Fester Geschwindigkeitsfehler (Motor 50/60 Hz, rpm-Verwechslung) ≠ zeitvarianter pitch_drift.
    "speed_calibration_error",
    # Analoger Preamp/Console-Klirr: H3/H5-dominant, asymmetrisch (5–15 % THD).
    "overload_distortion",
    # Acetat-Zersetzung: Substrat-Rissbildung + Lackschicht-Oxidation.
    "lacquer_disc_degradation",
    # Kassetten-Shell-Azimuth-Toleranz → HF-Kammfilterung 8–14 kHz.
    "cassette_azimuth_tolerance",
    # Drahtband: Draht-Knoten-Clicks + Wicklungs-Wow + nichtlinearer Magnetisierungsverlauf.
    "wire_recording_specific",
    # Vokal-Naturalness-Degradierung: VQI-Abfall durch kumulative NR/Kompressor-Eingriffe →
    # DSP-Korrektiv (HNR-Blend + Spektral-Tilt + Formant-Tilt) via phase_65 (§0a-konform).
    "vocal_quality_degradation",
    # ── v10.0.0: Stem-Targeted NR (phase_66) ────────────────────────────────
    # Kombination aus Vokal-Rauschen + Begleitungs-Rauschen, die wideband-NR nicht
    # sauber voneinander trennen kann → stem-spezifische NR via BSRoFormer-Separation.
    "vocal_stem_noise",  # Vokal-Stem + Begleitung haben unterschiedliche Rauschprofile
]

# Material-Typen — Priors für alle 34 Kausal-Ursachen (v10.0.0b)
# Priors pro Material nicht zwingend exakt auf 1.0 normiert — _infer() normalisiert Posterioren.
MATERIAL_PRIORS: dict[str, dict[str, float]] = {
    "tape": {
        "tape_dropout": 0.27,
        "tape_hiss": 0.27,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.13,
        "head_misalignment": 0.10,
        "dc_offset": 0.05,
        "digital_clip": 0.05,
        "soft_saturation": 0.05,
        "head_wear": 0.04,
        "print_through": 0.02,
        "transport_bump": 0.12,
        "tape_start_instability": 0.15,  # v10.0.0: cassette head/motor startup (first 20 s)
        "tape_head_contact_instability": 0.12,  # v10.0.0.x: gradual level dips from head-tape pressure variation
        # v10.0.0b: 22 erweiterte Ursachen
        "bandwidth_loss": 0.06,
        "high_freq_noise": 0.03,
        "stereo_imbalance": 0.02,
        "phase_issues": 0.02,
        "pitch_drift": 0.03,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.01,
        "compression_artifacts": 0.01,
        "quantization_noise": 0.01,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.01,
        "pre_echo": 0.01,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.01,
        "clipping": 0.02,
        "riaa_curve_error": 0.01,
        "aliasing": 0.01,
        "bias_error": 0.05,
        "sibilance": 0.01,
        "wow": 0.06,
        "flutter": 0.05,
        "wow_flutter": 0.08,
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.18,  # very common on tape
        "inner_groove_distortion": 0.01,  # N/A for tape
        "groove_echo": 0.01,  # N/A for tape
        "crosstalk": 0.04,  # early stereo tape
        "intermodulation_distortion": 0.03,  # tube amp chains
        "tape_splice_artifact": 0.12,  # common on reel/cassette
        "hf_remanence_loss": 0.15,  # primary tape issue
        "stylus_damage": 0.01,  # N/A for tape
        "sticky_shed_residue": 0.14,  # polyester tape degradation
        "multiband_wow_flutter": 0.08,  # head geometry artifact
        "generation_loss": 0.10,  # common in tape dubbing
        "motor_interference": 0.06,  # capstan motor
    },
    "vinyl": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.04,
        "vinyl_crackle": 0.38,
        "vinyl_warp": 0.18,
        "electrical_hum": 0.09,
        "head_misalignment": 0.03,
        "dc_offset": 0.05,
        "digital_clip": 0.13,
        "soft_saturation": 0.05,
        "head_wear": 0.02,
        "print_through": 0.01,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.04,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.02,
        "phase_issues": 0.01,
        "pitch_drift": 0.02,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.01,
        "compression_artifacts": 0.01,
        "quantization_noise": 0.01,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.01,
        "pre_echo": 0.01,
        "low_freq_rumble": 0.06,
        "transient_smearing": 0.01,
        "clipping": 0.02,
        "riaa_curve_error": 0.08,
        "aliasing": 0.01,
        "bias_error": 0.01,
        "sibilance": 0.01,
        "wow": 0.03,
        "flutter": 0.01,
        "wow_flutter": 0.03,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.18,
        "groove_echo": 0.15,
        "crosstalk": 0.06,
        "intermodulation_distortion": 0.05,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.15,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.05,  # vinyl→cassette→cd dubbing chains common in 1970s–1990s
        "motor_interference": 0.1,
    },
    "shellac": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.09,
        "vinyl_crackle": 0.42,
        "vinyl_warp": 0.14,
        "electrical_hum": 0.11,
        "head_misalignment": 0.05,
        "dc_offset": 0.04,
        "digital_clip": 0.07,
        "soft_saturation": 0.04,
        "head_wear": 0.02,
        "print_through": 0.01,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.10,
        "high_freq_noise": 0.03,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.03,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.01,
        "compression_artifacts": 0.01,
        "quantization_noise": 0.01,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.01,
        "pre_echo": 0.01,
        "low_freq_rumble": 0.04,
        "transient_smearing": 0.01,
        "clipping": 0.01,
        "riaa_curve_error": 0.10,
        "aliasing": 0.01,
        "bias_error": 0.01,
        "sibilance": 0.01,
        "wow": 0.02,
        "flutter": 0.01,
        "wow_flutter": 0.02,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.2,
        "groove_echo": 0.12,
        "crosstalk": 0.02,
        "intermodulation_distortion": 0.06,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.18,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.01,
        "motor_interference": 0.12,
    },
    "digital": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.04,
        "vinyl_crackle": 0.02,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.18,
        "head_misalignment": 0.02,
        "dc_offset": 0.09,
        "digital_clip": 0.50,
        "soft_saturation": 0.07,
        "head_wear": 0.01,
        "print_through": 0.04,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.02,
        "high_freq_noise": 0.03,
        "stereo_imbalance": 0.02,
        "phase_issues": 0.02,
        "pitch_drift": 0.01,
        "reverb_excess": 0.02,
        "digital_artifacts": 0.08,
        "compression_artifacts": 0.04,
        "quantization_noise": 0.04,
        "jitter_artifacts": 0.03,
        "dynamic_compression_excess": 0.06,
        "pre_echo": 0.02,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.03,
        "clipping": 0.10,
        "riaa_curve_error": 0.01,
        "aliasing": 0.02,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.02,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.01,
        "motor_interference": 0.01,
    },
    "unknown": {
        "tape_dropout": 0.10,
        "tape_hiss": 0.10,
        "vinyl_crackle": 0.10,
        "vinyl_warp": 0.07,
        "electrical_hum": 0.13,
        "head_misalignment": 0.09,
        "dc_offset": 0.09,
        "digital_clip": 0.17,
        "soft_saturation": 0.07,
        "head_wear": 0.05,
        "print_through": 0.03,
        "transport_bump": 0.04,
        "bandwidth_loss": 0.04,
        "high_freq_noise": 0.03,
        "stereo_imbalance": 0.02,
        "phase_issues": 0.02,
        "pitch_drift": 0.02,
        "reverb_excess": 0.02,
        "digital_artifacts": 0.03,
        "compression_artifacts": 0.03,
        "quantization_noise": 0.02,
        "jitter_artifacts": 0.02,
        "dynamic_compression_excess": 0.03,
        "pre_echo": 0.02,
        "low_freq_rumble": 0.02,
        "transient_smearing": 0.02,
        "clipping": 0.05,
        "riaa_curve_error": 0.02,
        "aliasing": 0.02,
        "bias_error": 0.02,
        "sibilance": 0.02,
        "wow": 0.03,
        "flutter": 0.02,
        "wow_flutter": 0.03,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.05,
        "inner_groove_distortion": 0.05,
        "groove_echo": 0.04,
        "crosstalk": 0.03,
        "intermodulation_distortion": 0.03,
        "tape_splice_artifact": 0.03,
        "hf_remanence_loss": 0.04,
        "stylus_damage": 0.04,
        "sticky_shed_residue": 0.03,
        "multiband_wow_flutter": 0.03,
        "generation_loss": 0.03,
        "motor_interference": 0.03,
    },
    # ── Digitale / Codec-Quellen ─────────────────────────────────────────────
    "mp3_low": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.07,
        "head_misalignment": 0.01,
        "dc_offset": 0.04,
        "digital_clip": 0.50,
        "soft_saturation": 0.07,
        "head_wear": 0.01,
        "print_through": 0.25,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.08,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.01,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.06,
        "compression_artifacts": 0.15,
        "quantization_noise": 0.03,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.04,
        "pre_echo": 0.10,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.02,
        "clipping": 0.05,
        "riaa_curve_error": 0.01,
        "aliasing": 0.03,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.02,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.06,  # mp3_low endpoint of multi-gen tape→cd→mp3 chains
        "motor_interference": 0.01,
    },
    "mp3_high": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.09,
        "head_misalignment": 0.01,
        "dc_offset": 0.04,
        "digital_clip": 0.36,
        "soft_saturation": 0.05,
        "head_wear": 0.01,
        "print_through": 0.39,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.04,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.01,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.04,
        "compression_artifacts": 0.08,
        "quantization_noise": 0.02,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.04,
        "pre_echo": 0.06,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.02,
        "clipping": 0.04,
        "riaa_curve_error": 0.01,
        "aliasing": 0.02,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.01,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.04,  # mp3_high endpoint of dubbing chains
        "motor_interference": 0.01,
    },
    "aac": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.09,
        "head_misalignment": 0.01,
        "dc_offset": 0.04,
        "digital_clip": 0.36,
        "soft_saturation": 0.05,
        "head_wear": 0.01,
        "print_through": 0.39,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.04,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.01,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.04,
        "compression_artifacts": 0.10,
        "quantization_noise": 0.02,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.04,
        "pre_echo": 0.08,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.02,
        "clipping": 0.04,
        "riaa_curve_error": 0.01,
        "aliasing": 0.02,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.01,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.01,
        "motor_interference": 0.01,
    },
    "cd_digital": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.11,
        "head_misalignment": 0.01,
        "dc_offset": 0.07,
        "digital_clip": 0.41,
        "soft_saturation": 0.06,
        "head_wear": 0.01,
        "print_through": 0.28,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.02,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.01,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.05,
        "compression_artifacts": 0.03,
        "quantization_noise": 0.03,
        "jitter_artifacts": 0.03,
        "dynamic_compression_excess": 0.06,
        "pre_echo": 0.02,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.03,
        "clipping": 0.08,
        "riaa_curve_error": 0.01,
        "aliasing": 0.01,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.01,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.03,  # cd_digital often from analog masters with generation loss
        "motor_interference": 0.01,
    },
    "streaming": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.07,
        "head_misalignment": 0.01,
        "dc_offset": 0.03,
        "digital_clip": 0.32,
        "soft_saturation": 0.05,
        "head_wear": 0.01,
        "print_through": 0.46,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.06,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.01,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.05,
        "compression_artifacts": 0.12,
        "quantization_noise": 0.02,
        "jitter_artifacts": 0.02,
        "dynamic_compression_excess": 0.05,
        "pre_echo": 0.05,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.02,
        "clipping": 0.04,
        "riaa_curve_error": 0.01,
        "aliasing": 0.02,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.01,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.04,  # streaming masters often from multi-gen source chains
        "motor_interference": 0.01,
    },
    "dat": {
        "tape_dropout": 0.07,
        "tape_hiss": 0.04,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.11,
        "head_misalignment": 0.02,
        "dc_offset": 0.04,
        "digital_clip": 0.27,
        "soft_saturation": 0.03,
        "head_wear": 0.05,
        "print_through": 0.35,
        "transport_bump": 0.03,
        "bandwidth_loss": 0.03,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.02,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.04,
        "compression_artifacts": 0.02,
        "quantization_noise": 0.03,
        "jitter_artifacts": 0.04,
        "dynamic_compression_excess": 0.03,
        "pre_echo": 0.01,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.02,
        "clipping": 0.05,
        "riaa_curve_error": 0.01,
        "aliasing": 0.01,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.03,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.02,
        "intermodulation_distortion": 0.01,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.02,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.02,
        "motor_interference": 0.01,
    },
    "minidisc": {
        "tape_dropout": 0.04,
        "tape_hiss": 0.03,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.07,
        "head_misalignment": 0.01,
        "dc_offset": 0.03,
        "digital_clip": 0.41,
        "soft_saturation": 0.04,
        "head_wear": 0.02,
        "print_through": 0.33,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.05,
        "high_freq_noise": 0.02,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.01,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.05,
        "compression_artifacts": 0.10,
        "quantization_noise": 0.02,
        "jitter_artifacts": 0.02,
        "dynamic_compression_excess": 0.03,
        "pre_echo": 0.04,
        "low_freq_rumble": 0.01,
        "transient_smearing": 0.02,
        "clipping": 0.04,
        "riaa_curve_error": 0.01,
        "aliasing": 0.02,
        "bias_error": 0.01,
        "sibilance": 0.02,
        "wow": 0.01,
        "flutter": 0.01,
        "wow_flutter": 0.01,
        "tape_start_instability": 0.001,  # N/A: no tape transport mechanism
        "tape_head_contact_instability": 0.001,  # N/A: no tape head contact
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.01,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.01,
        "motor_interference": 0.01,
    },
    # ── Historische Medien ───────────────────────────────────────────────────
    "wax_cylinder": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.23,
        "vinyl_crackle": 0.37,
        "vinyl_warp": 0.09,
        "electrical_hum": 0.07,
        "head_misalignment": 0.05,
        "dc_offset": 0.04,
        "digital_clip": 0.02,
        "soft_saturation": 0.02,
        "head_wear": 0.08,
        "print_through": 0.01,
        "transport_bump": 0.02,
        "bandwidth_loss": 0.15,
        "high_freq_noise": 0.04,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.06,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.01,
        "compression_artifacts": 0.01,
        "quantization_noise": 0.01,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.01,
        "pre_echo": 0.01,
        "low_freq_rumble": 0.04,
        "transient_smearing": 0.01,
        "clipping": 0.01,
        "riaa_curve_error": 0.01,
        "aliasing": 0.01,
        "bias_error": 0.01,
        "sibilance": 0.01,
        "wow": 0.04,
        "flutter": 0.02,
        "wow_flutter": 0.05,
        "tape_start_instability": 0.001,  # N/A: no magnetic tape head
        "tape_head_contact_instability": 0.001,  # N/A: no magnetic tape head
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.15,
        "groove_echo": 0.08,
        "crosstalk": 0.01,
        "intermodulation_distortion": 0.04,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.2,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.01,
        "motor_interference": 0.15,
    },
    "lacquer_disc": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.13,
        "vinyl_crackle": 0.41,
        "vinyl_warp": 0.11,
        "electrical_hum": 0.09,
        "head_misalignment": 0.04,
        "dc_offset": 0.04,
        "digital_clip": 0.02,
        "soft_saturation": 0.03,
        "head_wear": 0.09,
        "print_through": 0.02,
        "transport_bump": 0.01,
        "bandwidth_loss": 0.10,
        "high_freq_noise": 0.03,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.04,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.01,
        "compression_artifacts": 0.01,
        "quantization_noise": 0.01,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.01,
        "pre_echo": 0.01,
        "low_freq_rumble": 0.04,
        "transient_smearing": 0.01,
        "clipping": 0.01,
        "riaa_curve_error": 0.08,
        "aliasing": 0.01,
        "bias_error": 0.01,
        "sibilance": 0.01,
        "wow": 0.02,
        "flutter": 0.01,
        "wow_flutter": 0.02,
        "tape_start_instability": 0.001,  # N/A: no magnetic tape
        "tape_head_contact_instability": 0.001,  # N/A: no magnetic tape
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.01,
        "inner_groove_distortion": 0.16,
        "groove_echo": 0.12,
        "crosstalk": 0.03,
        "intermodulation_distortion": 0.05,
        "tape_splice_artifact": 0.01,
        "hf_remanence_loss": 0.01,
        "stylus_damage": 0.16,
        "sticky_shed_residue": 0.01,
        "multiband_wow_flutter": 0.01,
        "generation_loss": 0.01,
        "motor_interference": 0.1,
    },
    "wire_recording": {
        "tape_dropout": 0.18,
        "tape_hiss": 0.22,
        "vinyl_crackle": 0.04,
        "vinyl_warp": 0.13,
        "electrical_hum": 0.13,
        "head_misalignment": 0.09,
        "dc_offset": 0.04,
        "digital_clip": 0.02,
        "soft_saturation": 0.02,
        "head_wear": 0.10,
        "print_through": 0.03,
        "transport_bump": 0.08,
        "bandwidth_loss": 0.08,
        "high_freq_noise": 0.03,
        "stereo_imbalance": 0.01,
        "phase_issues": 0.01,
        "pitch_drift": 0.04,
        "reverb_excess": 0.01,
        "digital_artifacts": 0.01,
        "compression_artifacts": 0.01,
        "quantization_noise": 0.01,
        "jitter_artifacts": 0.01,
        "dynamic_compression_excess": 0.01,
        "pre_echo": 0.01,
        "low_freq_rumble": 0.02,
        "transient_smearing": 0.02,
        "clipping": 0.01,
        "riaa_curve_error": 0.01,
        "aliasing": 0.01,
        "bias_error": 0.03,
        "sibilance": 0.01,
        "wow": 0.05,
        "flutter": 0.04,
        "wow_flutter": 0.07,
        "tape_start_instability": 0.10,  # v10.0.0: wire recording motor startup
        "tape_head_contact_instability": 0.08,  # v10.0.0.x: wire head contact less affected
        # v10.0.0: 12 neue Ursachen
        "modulation_noise": 0.1,
        "inner_groove_distortion": 0.01,
        "groove_echo": 0.01,
        "crosstalk": 0.03,
        "intermodulation_distortion": 0.03,
        "tape_splice_artifact": 0.08,
        "hf_remanence_loss": 0.12,
        "stylus_damage": 0.01,
        "sticky_shed_residue": 0.05,
        "multiband_wow_flutter": 0.06,
        "generation_loss": 0.08,
        "motor_interference": 0.1,
    },
}

# Ensure newly introduced causes are present in every material prior table.
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("vocal_harshness", 0.01)

# v10.0.0: amplitude_drift — per-material priors (AGC/oxide/motor-temperature drift)
_AMPLITUDE_DRIFT_MATERIAL_PRIORS: dict[str, float] = {
    "tape": 0.20,
    "cassette": 0.18,
    "reel_tape": 0.15,
    "wire_recording": 0.15,
    "wax_cylinder": 0.15,
    "shellac": 0.08,
    "lacquer_disc": 0.08,
    "vinyl": 0.06,
    "dat": 0.03,
    "minidisc": 0.02,
    "cd_digital": 0.01,
    "mp3_low": 0.01,
    "mp3_high": 0.01,
    "aac": 0.01,
    "streaming": 0.005,
    "digital": 0.01,
    "unknown": 0.07,
}
for _mat, _ad_prior in _AMPLITUDE_DRIFT_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("amplitude_drift", _ad_prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("amplitude_drift", 0.03)

# v10.0.0: clicks, dolby_nr_mismatch, tape_head_level_dip — per-material priors
_CLICKS_MATERIAL_PRIORS: dict[str, float] = {
    "shellac": 0.25,
    "vinyl": 0.20,
    "lacquer_disc": 0.15,
    "wax_cylinder": 0.18,
    "tape": 0.12,
    "cassette": 0.12,
    "reel_tape": 0.08,
    "wire_recording": 0.14,
    "dat": 0.05,
    "minidisc": 0.04,
    "cd_digital": 0.05,
    "mp3_low": 0.04,
    "mp3_high": 0.03,
    "aac": 0.03,
    "streaming": 0.02,
    "digital": 0.03,
    "unknown": 0.10,
}
_DOLBY_NR_MISMATCH_MATERIAL_PRIORS: dict[str, float] = {
    "cassette": 0.18,  # Dolby B/C/S — häufigster Consumer-Fall (1975–2000)
    "reel_tape": 0.12,  # Dolby A/SR — Broadcast-Dekoder fehlt oft
    "tape": 0.08,  # Open-reel-Consumer mit Dolby B
    "wire_recording": 0.001,
    "wax_cylinder": 0.001,
    "lacquer_disc": 0.001,
    "shellac": 0.001,  # vor Dolby NR (Erfindung 1966)
    "vinyl": 0.001,
    "cd_digital": 0.001,
    "dat": 0.001,
    "minidisc": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "digital": 0.001,
    "unknown": 0.04,
}
_TAPE_HEAD_LEVEL_DIP_MATERIAL_PRIORS: dict[str, float] = {
    "cassette": 0.18,  # Capstan/Andruckrolle — häufig
    "tape": 0.16,  # Bandkopf-Kontaktdruckvariation
    "reel_tape": 0.14,  # Studio-Reel-Tape — Kopfverschleiß
    "wire_recording": 0.12,  # Drahtband-Kopf instabil
    "shellac": 0.001,
    "vinyl": 0.001,
    "lacquer_disc": 0.001,
    "wax_cylinder": 0.001,
    "cd_digital": 0.001,
    "dat": 0.001,
    "minidisc": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "digital": 0.001,
    "unknown": 0.05,
}
_SCRAPE_FLUTTER_MATERIAL_PRIORS: dict[str, float] = {
    "cassette": 0.17,
    "tape": 0.15,
    "reel_tape": 0.12,
    "wire_recording": 0.18,
    "shellac": 0.01,
    "vinyl": 0.02,
    "lacquer_disc": 0.01,
    "wax_cylinder": 0.02,
    "cd_digital": 0.001,
    "dat": 0.001,
    "minidisc": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "digital": 0.001,
    "unknown": 0.04,
}
_TAPE_HEAD_CLOG_MATERIAL_PRIORS: dict[str, float] = {
    "cassette": 0.16,
    "tape": 0.14,
    "reel_tape": 0.13,
    "wire_recording": 0.10,
    "shellac": 0.001,
    "vinyl": 0.001,
    "lacquer_disc": 0.001,
    "wax_cylinder": 0.001,
    "cd_digital": 0.001,
    "dat": 0.001,
    "minidisc": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "digital": 0.001,
    "unknown": 0.04,
}
for _new_cause, _new_priors in [
    ("clicks", _CLICKS_MATERIAL_PRIORS),
    ("dolby_nr_mismatch", _DOLBY_NR_MISMATCH_MATERIAL_PRIORS),
    ("tape_head_level_dip", _TAPE_HEAD_LEVEL_DIP_MATERIAL_PRIORS),
    ("scrape_flutter", _SCRAPE_FLUTTER_MATERIAL_PRIORS),
    ("tape_head_clog", _TAPE_HEAD_CLOG_MATERIAL_PRIORS),
]:
    for _mat, _prior in _new_priors.items():
        if _mat in MATERIAL_PRIORS:
            MATERIAL_PRIORS[_mat].setdefault(_new_cause, _prior)
    for _priors in MATERIAL_PRIORS.values():
        _priors.setdefault(_new_cause, 0.01)

# v10.0.0: 9 neue Kausal-Ursachen — Material-Priors
_PROXIMITY_EFFECT_MATERIAL_PRIORS: dict[str, float] = {
    # Nahbesprechungseffekt: häufig bei Ribbon/Kondensator-Mikrofonen ≤30 cm
    "shellac": 0.15,  # Richtmikrofone 1930–1950 bei Rundfunk
    "vinyl": 0.08,  # Studio-Vinyl: weniger problematisch
    "tape": 0.18,  # Kassette: Heimaufnahmen mit zu nahem Mikrofon häufig
    "cassette": 0.20,  # Heimkassette: Nahbesprechung sehr häufig
    "reel_tape": 0.22,  # Profi-Studio 1940–1970: häufig bei Röhren-Mikrofonen
    "lacquer_disc": 0.25,  # Heimschnitt: primitivere Technik, Mikrofon-Nahaufstellung
    "wax_cylinder": 0.12,  # Trichteraufnahme: Nahbesprechung bauartbedingt
    "wire_recording": 0.18,  # Wire: Nahbesprechung bei Feldaufnahmen
    "dat": 0.05,
    "minidisc": 0.06,
    "cd_digital": 0.04,  # Moderne Produktion: bewusste Entscheidung, keine Fehler
    "mp3_low": 0.03,
    "mp3_high": 0.03,
    "aac": 0.03,
    "streaming": 0.03,
    "digital": 0.03,
    "unknown": 0.10,
}
_ROOM_MODE_MATERIAL_PRIORS: dict[str, float] = {
    # Stehwellen-Raumresonanzen: Aufnahmestudios 1930–1970 mit parallelen Wänden
    "shellac": 0.14,
    "vinyl": 0.10,
    "tape": 0.12,
    "cassette": 0.10,
    "reel_tape": 0.18,  # Profi-Studios 1940–1970: oft problematische Akustik
    "lacquer_disc": 0.15,  # Heimaufnahme: keine akustische Behandlung
    "wax_cylinder": 0.16,  # Frühe Studios: kaum akustische Behandlung
    "wire_recording": 0.14,
    "dat": 0.06,
    "minidisc": 0.06,
    "cd_digital": 0.05,
    "mp3_low": 0.04,
    "mp3_high": 0.04,
    "aac": 0.04,
    "streaming": 0.04,
    "digital": 0.04,
    "unknown": 0.08,
}
_NR_BREATHING_MATERIAL_PRIORS: dict[str, float] = {
    # NR Pumpen/Atmen: nur bei korrekt dekodierten Analog-NR-Systemen
    "cassette": 0.22,  # Dolby B/C/S Consumer-Kassette 1975–2000
    "tape": 0.12,  # Consumer Reel mit Dolby B
    "reel_tape": 0.14,  # Profi-Reel mit Dolby A/SR
    "wire_recording": 0.06,  # AGC-Interaktion, nicht echtes NR-System
    "shellac": 0.001,  # vor Dolby NR (1966)
    "vinyl": 0.001,
    "lacquer_disc": 0.001,
    "wax_cylinder": 0.001,
    "cd_digital": 0.001,
    "dat": 0.001,
    "minidisc": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "digital": 0.001,
    "unknown": 0.05,
}
_FLUTTER_SIDEBANDS_MATERIAL_PRIORS: dict[str, float] = {
    # Flutter-Seitenbänder: bei > 3 Hz Flutter auf Tonträgern mit Sustained-Tönen
    "cassette": 0.16,
    "tape": 0.14,
    "reel_tape": 0.10,
    "wire_recording": 0.22,  # Drahtband: starkes Flutter durch Drahtgeometrie
    "shellac": 0.08,
    "vinyl": 0.06,
    "lacquer_disc": 0.08,
    "wax_cylinder": 0.10,
    "dat": 0.001,
    "cd_digital": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "minidisc": 0.001,
    "digital": 0.001,
    "unknown": 0.06,
}
_SPEED_CALIB_MATERIAL_PRIORS: dict[str, float] = {
    # Fester Geschwindigkeitsfehler: Motorfrequenz-Verwechslung, rpm-Fehler
    "shellac": 0.18,  # 78 rpm: 50/60 Hz Motor-Verwechslung sehr häufig
    "vinyl": 0.12,  # 33⅓/45 rpm Verwechslung
    "tape": 0.10,
    "cassette": 0.10,
    "reel_tape": 0.08,  # Professionell: besser kalibriert
    "lacquer_disc": 0.18,  # Heimschnitt: billige Motoren
    "wax_cylinder": 0.20,  # Sehr frühe Aufnahmen: keine Standardisierung
    "wire_recording": 0.16,
    "dat": 0.001,
    "cd_digital": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "minidisc": 0.001,
    "digital": 0.001,
    "unknown": 0.08,
}
_OVERLOAD_DISTORTION_MATERIAL_PRIORS: dict[str, float] = {
    # Analoger Preamp/Console-Klirr: Live-Aufnahmen 1960–1980, Broadcast
    "shellac": 0.12,
    "vinyl": 0.14,
    "tape": 0.16,
    "cassette": 0.14,
    "reel_tape": 0.18,  # Profi-Live-Aufnahmen: Eingangsstufen oft übersteuert
    "lacquer_disc": 0.10,
    "wax_cylinder": 0.08,
    "wire_recording": 0.12,
    "dat": 0.08,  # Analog-Eingang kann übersteuern
    "cd_digital": 0.06,  # Mastering-Overload aus analoger Kette
    "mp3_low": 0.04,
    "mp3_high": 0.04,
    "aac": 0.04,
    "streaming": 0.04,
    "minidisc": 0.06,
    "digital": 0.05,
    "unknown": 0.08,
}
_LACQUER_DEGRADATION_MATERIAL_PRIORS: dict[str, float] = {
    # Acetat-Zersetzung: nur Lacquer Disc
    "lacquer_disc": 0.55,  # primäres Target — Acetat-Zersetzung nach Jahrzehnten
    "shellac": 0.001,
    "vinyl": 0.001,
    "tape": 0.001,
    "cassette": 0.001,
    "reel_tape": 0.001,
    "wax_cylinder": 0.001,
    "wire_recording": 0.001,
    "dat": 0.001,
    "cd_digital": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "minidisc": 0.001,
    "digital": 0.001,
    "unknown": 0.05,
}
_CASSETTE_AZIMUTH_MATERIAL_PRIORS: dict[str, float] = {
    # Kassetten-Shell-Azimuth-Toleranz: nur Kassetten-Format
    "cassette": 0.20,
    "tape": 0.08,  # Consumer Open-Reel: etwas besser
    "reel_tape": 0.04,  # Profi: gute Toleranzen
    "shellac": 0.001,
    "vinyl": 0.001,
    "lacquer_disc": 0.001,
    "wax_cylinder": 0.001,
    "wire_recording": 0.001,
    "dat": 0.001,
    "cd_digital": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "minidisc": 0.001,
    "digital": 0.001,
    "unknown": 0.04,
}
_WIRE_SPECIFIC_MATERIAL_PRIORS: dict[str, float] = {
    # Drahtband-spezifische Defekte: nur Wire Recording
    "wire_recording": 0.45,  # primäres Target
    "tape": 0.001,
    "cassette": 0.001,
    "reel_tape": 0.001,
    "shellac": 0.001,
    "vinyl": 0.001,
    "lacquer_disc": 0.001,
    "wax_cylinder": 0.001,
    "dat": 0.001,
    "cd_digital": 0.001,
    "mp3_low": 0.001,
    "mp3_high": 0.001,
    "aac": 0.001,
    "streaming": 0.001,
    "minidisc": 0.001,
    "digital": 0.001,
    "unknown": 0.05,
}
for _new_cause_v9129, _new_priors_v9129 in [
    ("proximity_effect_excess", _PROXIMITY_EFFECT_MATERIAL_PRIORS),
    ("room_mode_resonance", _ROOM_MODE_MATERIAL_PRIORS),
    ("nr_breathing_artifact", _NR_BREATHING_MATERIAL_PRIORS),
    ("flutter_spectral_sidebands", _FLUTTER_SIDEBANDS_MATERIAL_PRIORS),
    ("speed_calibration_error", _SPEED_CALIB_MATERIAL_PRIORS),
    ("overload_distortion", _OVERLOAD_DISTORTION_MATERIAL_PRIORS),
    ("lacquer_disc_degradation", _LACQUER_DEGRADATION_MATERIAL_PRIORS),
    ("cassette_azimuth_tolerance", _CASSETTE_AZIMUTH_MATERIAL_PRIORS),
    ("wire_recording_specific", _WIRE_SPECIFIC_MATERIAL_PRIORS),
]:
    for _mat, _prior in _new_priors_v9129.items():
        if _mat in MATERIAL_PRIORS:
            MATERIAL_PRIORS[_mat].setdefault(_new_cause_v9129, _prior)
    for _priors in MATERIAL_PRIORS.values():
        _priors.setdefault(_new_cause_v9129, 0.01)

# v10.0.0: vocal_quality_degradation — Vokal-Naturalness-Degradierung durch
# kumulative NR/Kompressor-Eingriffe. Korrektiv: phase_65 (§0a-konform).
_VOCAL_QUALITY_DEGRADATION_MATERIAL_PRIORS: dict[str, float] = {
    "mp3_low": 0.13,  # Codec-Artefakte degradieren Stimmqualität stark
    "minidisc": 0.11,  # ATRAC-Codec: bekannt für Stimm-Artefakte
    "cassette": 0.12,  # Schlechte Aufnahmequalität + aggressive NR
    "streaming": 0.09,  # Variable Codec-Qualität
    "tape": 0.08,  # Häufige aggressive NR-Anwendung
    "reel_tape": 0.09,  # Professionelle NR (Dolby A/SR)
    "aac": 0.08,  # AAC-Codec-Artefakte in Vokalregion
    "mp3_high": 0.07,  # Weniger Codec-Degradierung
    "shellac": 0.07,  # Aggressive NR nötig
    "wax_cylinder": 0.07,  # Schwere Verarbeitungskette
    "lacquer_disc": 0.06,  # Ähnlich Shellac
    "wire_recording": 0.05,  # Unübliche Verarbeitungskette
    "vinyl": 0.05,  # Moderate NR
    "digital": 0.04,  # Wenig NR nötig
    "unknown": 0.06,
    "dat": 0.03,  # Hohe Qualität, minimale Verarbeitung
    "cd_digital": 0.02,  # Minimale Verarbeitung nötig
}
for _mat, _prior in _VOCAL_QUALITY_DEGRADATION_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("vocal_quality_degradation", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("vocal_quality_degradation", 0.04)

# v10.0.0: vocal_stem_noise — Vokal-Stem + Begleitung haben unterschiedliche Rauschprofile.
# Häufig bei Kassette/Tape (Hiss dominant) und MP3 (codec NR ungleichmäßig).
_VOCAL_STEM_NOISE_MATERIAL_PRIORS: dict[str, float] = {
    "tape": 0.14,  # Tape-Hiss im Begleit-Stem häufig dominanter als im Vokal-Stem
    "cassette": 0.16,  # Kassetten-Hiss + Dolby-Inkonsistenz: Stems divergieren stark
    "reel_tape": 0.12,  # Professionelle Aufnahme: geringerer Hiss-Unterschied
    "shellac": 0.10,  # Oberflächenrauschen — beide Stems betroffen
    "wax_cylinder": 0.09,
    "vinyl": 0.08,  # Knistern beide Stems; weniger Stem-Divergenz
    "mp3_low": 0.11,  # Codec NR erzeugt unterschiedliche Residuen je Stem
    "aac": 0.09,
    "digital": 0.04,
    "cd_digital": 0.03,
    "unknown": 0.08,
}
for _mat, _prior in _VOCAL_STEM_NOISE_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("vocal_stem_noise", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("vocal_stem_noise", 0.05)

# --- Tier 2: Neue Ursachen-Priors (MPEG / Stereo / Phase / Dropout-Subtypen) ---

_MPEG_FRAME_LOSS_MATERIAL_PRIORS: dict[str, float] = {
    "mp3_low": 0.18,  # Stark komprimiert → Brickwall hörbar
    "mp3_high": 0.12,  # Mittel → weniger auffällig
    "aac": 0.10,
    "minidisc": 0.08,  # ATRAC: eigene Artefakte
    "streaming": 0.14,
    "cd_digital": 0.02,  # CD: nur als Transcode
    "cassette": 0.0,  # Analog
    "tape": 0.0,
    "reel_tape": 0.0,
    "shellac": 0.0,
    "vinyl": 0.0,
    "wax_cylinder": 0.0,
    "dat": 0.05,
    "unknown": 0.05,
}
for _mat, _prior in _MPEG_FRAME_LOSS_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("mpeg_frame_loss", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("mpeg_frame_loss", 0.02)

_STEREO_COLLAPSE_MATERIAL_PRIORS: dict[str, float] = {
    "shellac": 0.10,  # Meist Mono → fake Stereo
    "wax_cylinder": 0.10,
    "cassette": 0.08,
    "tape": 0.06,
    "mp3_low": 0.07,  # Joint Stereo MS
    "lacquer_disc": 0.08,
    "vinyl": 0.04,
    "cd_digital": 0.02,
    "unknown": 0.05,
}
for _mat, _prior in _STEREO_COLLAPSE_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("stereo_field_collapse", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("stereo_field_collapse", 0.03)

_PHASE_ROTATION_MATERIAL_PRIORS: dict[str, float] = {
    "cassette": 0.10,  # Dolby NR + EQ
    "tape": 0.08,
    "reel_tape": 0.07,
    "wire_recording": 0.09,
    "shellac": 0.05,
    "wax_cylinder": 0.05,
    "mp3_low": 0.06,
    "unknown": 0.04,
}
for _mat, _prior in _PHASE_ROTATION_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("phase_rotation", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("phase_rotation", 0.03)

_DROPOUT_OXIDE_MATERIAL_PRIORS: dict[str, float] = {
    "cassette": 0.18,  # Kassette: häufigste Oxid-Dropouts
    "tape": 0.14,
    "reel_tape": 0.10,
    "wire_recording": 0.12,
    "shellac": 0.0,
    "vinyl": 0.0,
    "unknown": 0.05,
}
for _mat, _prior in _DROPOUT_OXIDE_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("dropout_oxide", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("dropout_oxide", 0.02)

_DROPOUT_HEAD_CONTACT_MATERIAL_PRIORS: dict[str, float] = {
    "cassette": 0.15,
    "tape": 0.14,
    "reel_tape": 0.12,
    "wire_recording": 0.10,
    "shellac": 0.0,
    "vinyl": 0.0,
    "unknown": 0.04,
}
for _mat, _prior in _DROPOUT_HEAD_CONTACT_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("dropout_head_contact", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("dropout_head_contact", 0.02)

_DROPOUT_SPLICE_MATERIAL_PRIORS: dict[str, float] = {
    "reel_tape": 0.15,  # Profi-Band = Schnitte
    "tape": 0.10,
    "cassette": 0.05,
    "wire_recording": 0.05,
    "shellac": 0.0,
    "vinyl": 0.0,
    "unknown": 0.03,
}
for _mat, _prior in _DROPOUT_SPLICE_MATERIAL_PRIORS.items():
    if _mat in MATERIAL_PRIORS:
        MATERIAL_PRIORS[_mat].setdefault("dropout_splice", _prior)
for _priors in MATERIAL_PRIORS.values():
    _priors.setdefault("dropout_splice", 0.01)

# Phase-Empfehlungen pro Ursache (kanonische phase_id = Dateiname ohne .py)
CAUSE_TO_PHASES: dict[str, list[str]] = {
    # ── Magnetband ────────────────────────────────────────────────────────────
    "tape_dropout": [
        "phase_24_dropout_repair",
        "phase_55_diffusion_inpainting",  # §7.2 DiffWave-Inpainting
        "phase_01_click_removal",
        "phase_03_denoise",
    ],
    "tape_hiss": [
        "phase_29_tape_hiss_reduction",
        "phase_03_denoise",
        "phase_04_eq_correction",
        "phase_40_loudness_normalization",
    ],
    # ── Vinyl ────────────────────────────────────────────────────────────────
    "vinyl_crackle": [
        "phase_09_crackle_removal",
        "phase_01_click_removal",
        "phase_28_surface_noise_profiling",
        "phase_03_denoise",
    ],
    "vinyl_warp": [
        "phase_12_wow_flutter_fix",
        "phase_31_speed_pitch_correction",
        "phase_04_eq_correction",
        "phase_03_denoise",
    ],
    # §7.2 Spec 06 — getrennte Routing-Einträge für WOW/FLUTTER DefectTypes
    "wow_flutter": [
        "phase_12_wow_flutter_fix",
        "phase_31_speed_pitch_correction",
        "phase_04_eq_correction",
        "phase_03_denoise",
    ],
    "wow": [
        "phase_12_wow_flutter_fix",
        "phase_31_speed_pitch_correction",
        "phase_04_eq_correction",
    ],
    "flutter": [
        "phase_12_wow_flutter_fix",
        "phase_08_transient_preservation",
        "phase_31_speed_pitch_correction",
        "phase_03_denoise",
    ],
    # ── Elektrik / Mechanik ──────────────────────────────────────────────────
    "electrical_hum": ["phase_02_hum_removal", "phase_03_denoise", "phase_04_eq_correction"],
    "head_misalignment": [
        "phase_06_frequency_restoration",
        "phase_04_eq_correction",
        "phase_14_phase_correction",
        "phase_25_azimuth_correction",
        "phase_03_denoise",
    ],
    "dc_offset": ["phase_30_dc_offset_removal", "phase_40_loudness_normalization"],
    "digital_clip": ["phase_23_spectral_repair", "phase_06_frequency_restoration", "phase_40_loudness_normalization"],
    # ── Spektrale Defekte ────────────────────────────────────────────────────
    "bandwidth_loss": [
        "phase_06_frequency_restoration",
        "phase_07_harmonic_restoration",
        # phase_39_air_band_enhancement ENTFERNT (BUG-FIX v10.0.0 §6.2c):
        # phase_39 erzeugt Halluzinationen über BW-Ceiling analoger Materialien im Restoration-Modus.
    ],
    "high_freq_noise": ["phase_29_tape_hiss_reduction", "phase_03_denoise", "phase_18_noise_gate"],
    # ── Stereo / Phase ───────────────────────────────────────────────────────
    "stereo_imbalance": ["phase_15_stereo_balance", "phase_33_stereo_width_limiter", "phase_34_mid_side_processing"],
    "phase_issues": ["phase_14_phase_correction", "phase_25_azimuth_correction"],
    # ── Pitch / Dynamik ──────────────────────────────────────────────────────
    "pitch_drift": ["phase_31_speed_pitch_correction", "phase_12_wow_flutter_fix"],
    "reverb_excess": ["phase_20_reverb_reduction", "phase_49_advanced_dereverb"],
    "print_through": [
        "phase_57_print_through_reduction",  # Bidirektionale LMS (Pre+Post-Echo) — Spec §7.x Primär
        "phase_29_tape_hiss_reduction",
        "phase_24_dropout_repair",  # §4.5+§7.2
        "phase_03_denoise",
        "phase_23_spectral_repair",
    ],
    # ── Digital / Codec ──────────────────────────────────────────────────────
    "digital_artifacts": ["phase_23_spectral_repair", "phase_50_spectral_repair", "phase_06_frequency_restoration"],  # §4.7c POCS vor PGHI; §2.57a HF-Guard für analoge Harmoniken
    "compression_artifacts": [
        "phase_23_spectral_repair",  # Apollo pre-proc + IMCRA inpainting (primary codec path); §4.7c POCS n_iter=2–5
        "phase_50_spectral_repair",  # STFT spike interpolation DSP (no Apollo — fallback/complement); §2.57a HF-Guard _hf_protected_bin_start aktiv
        "phase_26_dynamic_range_expansion",
        "phase_06_frequency_restoration",
        "phase_54_transparent_dynamics",
    ],
    "quantization_noise": ["phase_23_spectral_repair", "phase_03_denoise", "phase_06_frequency_restoration"],
    # Jitter ist kein mechanisches Wow/Flutter-Problem: keine PSOLA-Phase_12-Zuordnung.
    "jitter_artifacts": ["phase_23_spectral_repair", "phase_14_phase_correction"],
    "dynamic_compression_excess": [
        "phase_26_dynamic_range_expansion",
        "phase_54_transparent_dynamics",
        # phase_35_multiband_compression ENTFERNT (BUG-FIX v10.0.0 §0a): Stem-Enhancement VERBOTEN in Restoration
    ],
    "head_wear": [
        "phase_56_spectral_band_gap_repair",  # §4.5/§7.2
        "phase_14_phase_correction",
        "phase_06_frequency_restoration",
    ],
    "soft_saturation": [],  # §2.1/§6.3 — BEWAHREN, kein destruktiver Eingriff
    # ── MP3/AAC-Codec-Pre-Echo ───────────────────────────────────────────────
    "pre_echo": [
        "phase_23_spectral_repair",  # §6.3 — Codec-Pre-Echo vor Transienten
        "phase_50_spectral_repair",
        "phase_08_transient_preservation",
    ],
    # ── Tieffrequenz / Transienten / Clipping ────────────────────────────────
    "low_freq_rumble": [
        "phase_05_rumble_filter",  # §7.2 — Subsonic-/Rumble-Filter
        "phase_03_denoise",
        "phase_04_eq_correction",
    ],
    "transient_smearing": [
        "phase_08_transient_preservation",  # §7.2 — Transienten-Restaurierung
        "phase_36_transient_shaper",
        "phase_23_spectral_repair",
    ],
    "clipping": [
        "phase_07_declipper",
        "phase_23_spectral_repair",
        "phase_06_frequency_restoration",
    ],  # §v10.18: declipper als Primäraktion
    # ── Entzerrungs- & Digitalisierungsfehler (§6.3, §7.2 v10.0.0) ─────────
    "riaa_curve_error": [
        "phase_04_eq_correction",  # RIAA/AES/NAB/FFRR-Entzerrungs-Fehler
        "phase_06_frequency_restoration",
        "phase_07_harmonic_restoration",
    ],
    "aliasing": [
        # Aliasing ist kohärente Spiegelfrequenz-Struktur, kein stationäres Rauschen.
        "phase_23_spectral_repair",
        "phase_50_spectral_repair",
    ],
    "bias_error": [
        "phase_04_eq_correction",  # Falscher Vormagnetisierungsstrom (Bandaufnahme)
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "phase_29_tape_hiss_reduction",
    ],
    # ── Sibilanten (§6.3 v10.0.0) ──────────────────────────────────────────
    "sibilance": [
        "phase_19_de_esser",  # Primär: De-Esser (Sibilantenreduktion)
        "phase_43_ml_deesser",  # ML-gestützter De-Esser
        # phase_42_vocal_enhancement ENTFERNT (BUG-FIX v10.0.0 §0a): Stem-Enhancement VERBOTEN in Restoration
    ],
    # ── Transport-Bump (v10.0.0b — Kassetten-Holpern) ───────────────────────
    "transport_bump": [
        "phase_12_wow_flutter_fix",  # Primär: lokale PSOLA-Korrektur der Pitch-Sprünge
        "phase_08_transient_preservation",  # Amplitude: Attack-Restoration der Pegel-Dips
        "phase_24_dropout_repair",  # Fallback: Amplitude-Reparatur bei tiefem Signal-Einbruch
        "phase_31_speed_pitch_correction",  # Zusätzlich: globale Speed-Stabilisierung
    ],
    # ── Tape-Start-Instability (v10.0.0 — Kassettenkopf-Einrastfehler) ─────
    # Combined multi-defect cause for the first 20 s of cassette playback:
    # motor spin-up speed ramp, head engagement azimuth drift, tape slack
    # tension equalization.  Requires coordinated WOW+AZIMUTH+SPEED correction.
    # Scientific basis: Camras (1988) Ch. 7 "Transport Mechanisms — Start Transients";
    # McKnight (1969) AES Convention 36 — measured cassette start-up characteristics.
    "tape_start_instability": [
        "phase_12_wow_flutter_fix",  # Motor speed ramp → wow/flutter in intro
        "phase_25_azimuth_correction",  # Head engagement → time-varying azimuth drift
        "phase_31_speed_pitch_correction",  # Constant speed offset after motor stabilization
        "phase_14_phase_correction",  # Residual L/R phase misalignment
        "phase_24_dropout_repair",  # Tape-slack can cause momentary signal loss
    ],
    # ── Tape Head Contact Instability (level dips from pressure variation) ──
    # Gradual envelope dips (60-100 ms onset, 100-400 ms total, 10-25 dB deep)
    # caused by capstan irregularity, worn pinch roller, oxide shedding, or
    # tape tension variation.  Phase 12 contains the autonomous Tape Level
    # Stabilizer (Step 6c) that detects and compensates these dips.
    # Scientific basis: Camras (1988) Ch.7 — head-tape spacing modulates output.
    "tape_head_contact_instability": [
        "phase_12_wow_flutter_fix",  # Primary: Tape Level Stabilizer (Step 6c)
        "phase_24_dropout_repair",  # Deep dips that cross dropout threshold
        "phase_26_dynamic_range_expansion",  # Restore micro-dynamics after leveling
    ],
    # ── Vocal-Harshness (v10.0.0 — Vokal-Härte/Übersteuerung/Kratzigkeit) ──
    "vocal_harshness": [
        # phase_42_vocal_enhancement ENTFERNT (BUG-FIX v10.0.0 §0a): Stem-Enhancement VERBOTEN in Restoration
        "phase_19_de_esser",  # De-Esser reduziert auch obere Härte-Frequenzen
        "phase_43_ml_deesser",  # ML-De-Esser (zweiter Pass, Frikativ-Kontrolle)
        "phase_23_spectral_repair",  # Spektrale Reparatur bei starker Verzerrung
    ],
    # ── v10.0.0: 12 neue Kausal-Ursachen → Phase-Mappings ──────────────────
    "modulation_noise": [
        "phase_59_modulation_noise_reduction",  # Primary: signal-dependent noise reduction
        "phase_03_denoise",  # OMLSA/ResembleEnhance secondary pass
        "phase_29_tape_hiss_reduction",  # Complementary HF denoising
    ],
    "inner_groove_distortion": [
        "phase_60_inner_groove_distortion_repair",  # Primary: adaptive THD reduction
        "phase_23_spectral_repair",  # Spectral inpainting for severe cases; §4.7c POCS n_iter=2–5 vor PGHI
        "phase_04_eq_correction",  # HF tilt compensation
    ],
    "groove_echo": [
        "phase_61_groove_echo_cancellation",  # Primary: template-based echo removal
        "phase_20_reverb_reduction",  # SGMSE+ for residual echo energy
        "phase_03_denoise",  # Final noise cleanup
    ],
    "crosstalk": [
        "phase_62_crosstalk_cancellation",  # Primary: BSS-based channel separation
        "phase_14_phase_correction",  # Phase alignment after separation
        "phase_34_mid_side_processing",  # §7.2 Spec 06 — M/S für Restfehler-Kontrolle
    ],
    "intermodulation_distortion": [
        "phase_63_intermodulation_reduction",  # Primary: Volterra-based IMD removal
        "phase_23_spectral_repair",  # Spectral inpainting for IMD products; §4.7c POCS-Konsistenz-Projektion
        "phase_04_eq_correction",  # Spectral tilt correction
    ],
    "tape_splice_artifact": [
        "phase_64_tape_splice_repair",  # Primary: splice artifact removal
        "phase_01_click_removal",  # Impulse component
        "phase_24_dropout_repair",  # Level discontinuity component
    ],
    "hf_remanence_loss": [
        "phase_06_frequency_restoration",  # Primary: AudioSR bandwidth extension
        "phase_23_spectral_repair",  # Spectral envelope reconstruction
        "phase_04_eq_correction",  # HF shelf compensation
        "phase_24_dropout_repair",  # Sekundär: Oxid-Degradierung → Dropout-Risiko (physikalisch koppliert)
    ],
    "stylus_damage": [
        "phase_09_crackle_removal",  # Primary: broadband distortion removal
        "phase_23_spectral_repair",  # Spectral repair of harmonic distortion
        "phase_60_inner_groove_distortion_repair",  # Shared asymmetric distortion logic
    ],
    "sticky_shed_residue": [
        "phase_24_dropout_repair",  # Primary: level dip repair
        "phase_29_tape_hiss_reduction",  # Modulated noise burst removal
        "phase_03_denoise",  # Residual noise cleanup
    ],
    "multiband_wow_flutter": [
        "phase_12_wow_flutter_fix",  # Primary: multi-band pitch correction
        "phase_31_speed_pitch_correction",  # Global speed correction
        "phase_08_transient_preservation",  # Transient integrity after correction
    ],
    "generation_loss": [
        "phase_06_frequency_restoration",  # Bandwidth extension
        "phase_03_denoise",  # Cumulative noise removal
        "phase_23_spectral_repair",  # Spectral coherence restoration
        "phase_04_eq_correction",  # Spectral tilt correction
    ],
    "motor_interference": [
        "phase_02_hum_removal",  # Primary: harmonic removal in 80–300 Hz
        "phase_03_denoise",  # §7.2 Spec 06 — OMLSA/DFNet für Motor-Breitbandrauschen
        "phase_29_tape_hiss_reduction",  # §7.2 Spec 06 — HF-Anteile der Motor-Interferenz
        "phase_04_eq_correction",  # Residual spectral correction
    ],
    # ── v10.0.0: Pegelveränderung ────────────────────────────────────────────
    "amplitude_drift": [
        "phase_40_loudness_normalization",  # Primary: time-varying inverse gain envelope
        # (amplitude_drift_correction=True)
    ],
    # ── v10.0.0: DefectType→CAUSE-Lücken ─────────────────────────────────────
    "clicks": [
        "phase_01_click_removal",  # Primary: Bayesian click/impulse detection
        "phase_09_crackle_removal",  # Secondary: handles dense click fields
    ],
    "dolby_nr_mismatch": [
        "phase_04_eq_correction",  # Primary: high-shelf EQ inversion (Dolby B/C/S expand) ≈ −6 to −20 dB HF
        "phase_29_tape_hiss_reduction",  # Secondary: Dolby NR typically also encodes tape hiss
        "phase_03_denoise",  # Tertiary: residual noise after HF correction
    ],
    "tape_head_level_dip": [
        "phase_54_transparent_dynamics",  # Primary: Envelope-Re-Smoothing für graduelle Dips
        "phase_24_dropout_repair",  # Deep dips cross dropout threshold
        "phase_40_loudness_normalization",  # Slow level drift correction
    ],
    "scrape_flutter": [
        "phase_12_wow_flutter_fix",  # Primary: transport-/reibungsbedingte FM-Modulation korrigieren
        "phase_31_speed_pitch_correction",  # Fine correction for fast scrape-induced pitch wobble
        "phase_08_transient_preservation",  # Sustain shimmer glätten ohne Attack-Verlust
    ],
    "tape_head_clog": [
        "phase_56_spectral_band_gap_repair",  # Primary: lokale HF-Bandlücken rekonstruieren
        "phase_25_azimuth_correction",  # Head geometry / spacing cleanup
        "phase_24_dropout_repair",  # tiefe Clogs können dropout-artig erscheinen
    ],
    # ── v10.0.0: 9 neue Kausal-Ursachen → Phase-Mappings ────────────────────
    "proximity_effect_excess": [
        "phase_04_eq_correction",  # Primary: LF-Shelf-Absenkung ~150–250 Hz (−4 bis −8 dB, material-adaptiv)
        "phase_05_rumble_filter",  # Sekundär: Sub-Bass-Begleiterscheinung ausfiltern
    ],
    "room_mode_resonance": [
        "phase_04_eq_correction",  # Primary: parametrischer Notch-EQ 40–200 Hz (Q=12, −8 dB)
        "phase_16_final_eq",  # Sekundär: Feinkorrektur verbleibender Resonanzspitzen
        "phase_05_rumble_filter",  # Tertiär: Sub-Bass-Begleitenergie (Rolloff)
    ],
    "nr_breathing_artifact": [
        "phase_54_transparent_dynamics",  # Primary: Envelope-Re-Smoothing der NR-Modulationszonen
        "phase_08_transient_preservation",  # Sekundär: Transienten-Integrität an Pumping-Grenzen
        # KEIN phase_03/phase_29 — weiteres NR auf NR-Artefakt verstärkt das Pumpen (§4.11, V28)
    ],
    "flutter_spectral_sidebands": [
        "phase_12_wow_flutter_fix",  # Primary: PSOLA-Korrektur eliminiert Seitenband-Quelle (Pitch-Instabilität)
        "phase_23_spectral_repair",  # Sekundär: spektrale Inpainting der Seitenband-Energie um Peaks
        "phase_08_transient_preservation",  # Tertiär: Transienten-Integrität nach Pitch-Korrektur
    ],
    "speed_calibration_error": [
        "phase_12_wow_flutter_fix",  # Primary: globale Pitch-Verschiebung (Constant-Rate-Mode)
        "phase_31_speed_pitch_correction",  # Sekundär: Feinabstimmung nach Grob-Korrektur
    ],
    "overload_distortion": [
        "phase_09_crackle_removal",  # Primary: asymmetrische Wellenform + transiente Peakrekonstruktion
        "phase_23_spectral_repair",  # Sekundär: Spektral-Inpainting harmonischer Klirr-Produkte (H2/H3)
        "phase_14_phase_correction",  # Tertiär: Phasenverzerrung durch nichtlineares Overload
        # KEIN phase_63 (IMD-Reduktion) — Harmonische ≠ Intermodulationsprodukte (§4.11, V29)
    ],
    "lacquer_disc_degradation": [
        "phase_03_denoise",  # Primary: Substrat-Rauschen + Breitband-HF-Verlust
        "phase_09_crackle_removal",  # Sekundär: Rissbildungs-Clicks (dicht, periodisch)
        "phase_01_click_removal",  # Tertiär: einzelne scharfe Riss-Events
        "phase_06_frequency_restoration",  # Quartär: HF-Wiederherstellung (Lackschicht-Oxidation ≥ 8 kHz)
    ],
    "cassette_azimuth_tolerance": [
        "phase_14_phase_correction",  # Primary: Shell-Toleranz-bedingter HF-Phasen-Slope L/R
        "phase_25_azimuth_correction",  # Sekundär: residualer Azimuth-Drift nach Shell-Korrektur
        "phase_06_frequency_restoration",  # Tertiär: HF-Kammfilterung kompensieren (8–14 kHz)
    ],
    "wire_recording_specific": [
        "phase_12_wow_flutter_fix",  # Primary: Draht-Wicklungs-Wow + Knoten-Geschwindigkeitsspitzen
        "phase_24_dropout_repair",  # Sekundär: Draht-Knoten-Signalunterbrechungen
        "phase_03_denoise",  # Tertiär: Drahtband-Rauschen (nichtlinear, breitbandig)
        "phase_01_click_removal",  # Quartär: Draht-Knoten-Clicks
    ],
    # §0p Vocal-Supremacy + §0a-Invariante: VQI-Abfall durch kumulative Phasen-Eingriffe.
    # phase_65 ist das einzige §0a-konforme Korrektiv in Restoration (DSP: HNR-Blend +
    # Spektral-Tilt + Formant-Tilt). phase_42_vocal_enhancement VERBOTEN in Restoration (§0a).
    "vocal_quality_degradation": [
        "phase_65_vocal_naturalness_restoration",  # Primary: §0a DSP-Korrektiv (§7.10)
        "phase_03_denoise",  # Sekundär: Rausch-Basis absenken → VQI stabilisieren
        "phase_19_de_esser",  # Tertiär: Sibilanten-Harshness als VQI-Treiber dämpfen
    ],
    # ── v10.0.0: Stem-Targeted NR (phase_66) ─────────────────────────────────
    # Wenn Vokal + Begleitung unterschiedliche Rauschprofile haben, die wideband-NR
    # (phase_03/phase_29) nicht sauber entkoppeln kann. BSRoFormer-Separation erlaubt
    # stem-spezifische NR-Parameter (energy_bias −6 dB Vokal, −9 dB Instrumental).
    # §0a: Erlaubt in Restoration UND Studio 2026 (kein §0a-Ausschluss).
    "vocal_stem_noise": [
        "phase_66_stem_targeted_nr",  # Primary: BSRoFormer + stem-spezifische DFN-NR
        "phase_03_denoise",  # Sekundär: wideband-NR als Ergänzung
        "phase_65_vocal_naturalness_restoration",  # Tertiär: VQI-Korrektiv wenn nötig
    ],
}

# §2.67 Koalitions-Priorisierung bereits auf Ursache→Phase-Ebene.
# Damit bleiben zusammengehörige Reparaturphasen auch vor der UV3-Ausführung
# dichter beieinander und werden weniger durch globale Einzelscores getrennt.
_CAUSAL_PHASE_COALITIONS: dict[str, tuple[str, ...]] = {
    "digital_repair_chain": ("phase_23_spectral_repair", "phase_50_spectral_repair"),
    "hiss_harmonic_rebuild": ("phase_29_tape_hiss_reduction", "phase_07_harmonic_restoration"),
    "stereo_alignment": ("phase_14_phase_correction", "phase_25_azimuth_correction"),
    "generation_loss_rebuild": ("phase_23_spectral_repair", "phase_07_harmonic_restoration"),
}

# §6.2b/c Era-Verarbeitungsrichtlinien: Materialspezifische Phasen-Ausschlüsse.
# Phasen die für ein bestimmtes Material VERBOTEN sind — unabhängig von der Defect-Ursache.
# Spec copilot-instructions.md §ERA-Tabelle:
#   1900–1925 (wax_cylinder): VERBOTEN phase_07 (keine Harmonik-Ergänzung, BW ≤ 3 kHz)
#   wire_recording: kein Harmonik-Enhancement (keine verlässliche f0-Basis)
_MATERIAL_PHASE_EXCLUSIONS: dict[str, frozenset] = {
    # Akustische Ära (Trichteraufnahmen): Harmonik-Ergänzung verfälscht das Material — §ERA 1900-1925
    "wax_cylinder": frozenset(
        {
            "phase_07_harmonic_restoration",  # VERBOTEN per spec: keine Harmonik für Trichteraufnahmen
        }
    ),
    # Drahtaufnahmen: keine verlässliche harmonische Basis für additive Synthese
    "wire_recording": frozenset(
        {
            "phase_07_harmonic_restoration",
        }
    ),
}

# §SOTA #8: Sekundäre Kausalketten — von der primären Ursache zur Wurzel
_SECONDARY_CAUSE_CHAIN: dict[str, tuple[str, ...]] = {
    "vinyl_crackle": ("dust_contamination", "dry_playback", "static_discharge"),
    "vinyl_wow": ("off_center_pressing", "bearing_wear", "belt_slippage"),
    "vinyl_flutter": ("motor_instability", "capstan_irregularity"),
    "tape_dropout": ("oxide_shedding", "head_clog", "binder_hydrolysis"),
    "tape_hiss": ("magnetic_particle_noise", "bias_undercorrection"),
    "tape_wow": ("capstan_eccentricity", "pinch_roller_degradation"),
    "cassette_azimuth": ("shell_warp", "head_misalignment", "transport_wear"),
    "mp3_artifact": ("mdct_quantization", "pre_echo_ringing", "band21_sfb21_cutoff"),
    "clipping": ("preamp_overdrive", "adc_saturation", "limiter_misuse"),
    "hum": ("ground_loop", "psu_ripple", "electromagnetic_induction"),
    "reverb_excess": ("room_reflection", "mic_distance", "no_acoustic_treatment"),
}

# Empfohlene Parameter pro Ursache
CAUSE_PARAMS: dict[str, dict[str, Any]] = {
    "tape_dropout": {
        "noise_reduction_strength": 0.55,
        "ar_order": 64,
        "ola_crossfade_ms": 20.0,
        "inpaint_context_ms": 50.0,
    },
    "tape_hiss": {
        "noise_reduction_strength": 0.70,
        "ar_order": 32,
        "hpf_cutoff_hz": 60.0,
        "nr_smoothing_ms": 80.0,
    },
    "vinyl_crackle": {
        "click_threshold_sigma": 4.5,
        "noise_reduction_strength": 0.40,
        "ar_order": 48,
        "declicker_window_ms": 5.0,
    },
    "vinyl_warp": {
        "pitch_correction_semitones": 0.5,
        "wow_flutter_filter_hz": 0.5,
        "noise_reduction_strength": 0.25,
    },
    "electrical_hum": {
        "hum_fundamental_hz": 50.0,  # wird auto-erkannt
        "hum_harmonics": 5,
        "hum_notch_q": 20.0,
        "noise_reduction_strength": 0.20,
    },
    "head_misalignment": {
        "azimuth_correction_deg": 0.0,  # wird optimiert
        "hf_boost_db": 3.0,
        "eq_high_shelf_hz": 8000.0,
    },
    "dc_offset": {
        "hpf_cutoff_hz": 10.0,
        "normalization_lufs": -23.0,
    },
    "digital_clip": {
        "declip_threshold": 0.98,
        "harmonic_boost_db": 1.5,
        "noise_reduction_strength": 0.10,
    },
    "transport_bump": {
        "bump_correction_strength": 0.85,
        "bump_psola_crossfade_ms": 15.0,
        "bump_envelope_smooth_ms": 10.0,
    },
    # v10.0.0b: Erweiterte Ursachen-Parameter
    "bandwidth_loss": {
        "hf_extension_target_hz": 16000.0,
        "harmonic_boost_db": 2.0,
    },
    "compression_artifacts": {
        "spectral_repair_strength": 0.60,
        "noise_reduction_strength": 0.30,
    },
    "dynamic_compression_excess": {
        "expansion_ratio": 1.5,
        "expansion_threshold_db": -20.0,
    },
    "reverb_excess": {
        "dereverb_strength": 0.70,
        "dry_level": 0.85,
    },
    "vocal_harshness": {
        "deesser_strength": 0.65,
        "presence_attenuation_db": 2.5,
        "vocal_harshness_threshold": 0.35,
    },
    "low_freq_rumble": {
        "hpf_cutoff_hz": 30.0,
        "rumble_filter_order": 4,
    },
    "wow": {
        "wow_flutter_filter_hz": 0.5,
        "pitch_correction_semitones": 0.3,
    },
    "flutter": {
        "wow_flutter_filter_hz": 8.0,
        "pitch_correction_semitones": 0.2,
    },
    "wow_flutter": {
        "wow_flutter_filter_hz": 4.0,
        "pitch_correction_semitones": 0.4,
        "noise_reduction_strength": 0.20,
    },
    "clipping": {
        "declip_threshold": 0.97,
        "harmonic_boost_db": 1.0,
    },
    "bias_error": {
        "noise_reduction_strength": 0.50,
        "eq_high_shelf_hz": 6000.0,
        "hf_boost_db": 2.0,
    },
    "riaa_curve_error": {
        "eq_correction_strength": 0.80,
    },
    "sibilance": {
        "deesser_threshold_db": -12.0,
        "deesser_frequency_hz": 6500.0,
    },
    "pre_echo": {
        "spectral_repair_strength": 0.50,
        "transient_preserve_strength": 0.80,
    },
    # ── v10.0.0: Parameter für 12 neue Ursachen ────────────────────────────
    "modulation_noise": {
        "noise_reduction_strength": 0.60,
        "signal_dependent_gate": True,
        "modulation_tracking_ms": 20.0,
    },
    "inner_groove_distortion": {
        "thd_reduction_strength": 0.55,
        "hf_compensation_db": 2.0,
        "position_adaptive": True,
    },
    "groove_echo": {
        "echo_cancellation_strength": 0.65,
        "revolution_delay_s": 1.8,
        "spectral_subtraction_floor": -40.0,
    },
    "crosstalk": {
        "separation_strength": 0.50,
        "bss_iterations": 20,
        "preserve_stereo_image": True,
    },
    "intermodulation_distortion": {
        "imd_reduction_strength": 0.55,
        "volterra_order": 3,
        "spectral_notch_width_hz": 50.0,
    },
    "tape_splice_artifact": {
        "crossfade_ms": 15.0,
        "impulse_removal_strength": 0.70,
        "level_smoothing_ms": 50.0,
    },
    "hf_remanence_loss": {
        "hf_restoration_strength": 0.60,
        "ghost_harmonic_boost_db": 3.0,
        "rolloff_compensation_slope": 6.0,
    },
    "stylus_damage": {
        "distortion_reduction_strength": 0.55,
        "asymmetry_correction": True,
        "harmonic_rebalancing": True,
    },
    "sticky_shed_residue": {
        "level_dip_repair_strength": 0.65,
        "noise_burst_removal_strength": 0.60,
        "dip_detection_threshold_db": 4.0,
    },
    "multiband_wow_flutter": {
        "multiband_correction": True,
        "n_bands": 4,
        "flutter_smoothing_ms": 30.0,
    },
    "generation_loss": {
        "noise_reduction_strength": 0.55,
        "bandwidth_extension_strength": 0.50,
        "phase_coherence_restoration": True,
    },
    "motor_interference": {
        "harmonic_removal_strength": 0.65,
        "motor_freq_range_hz": [80, 300],
        "sideband_removal": True,
    },
    # v10.0.0: Parameter für tape-transport-spezifische Ursachen
    "tape_start_instability": {
        "wow_flutter_filter_hz": 3.0,  # motor spin-up wow rate
        "pitch_correction_semitones": 0.5,
        "azimuth_correction_deg": 0.0,  # auto-optimiert
        "start_segment_s": 20.0,  # Korrektur auf erste 20 Sekunden begrenzen
    },
    "tape_head_contact_instability": {
        "dip_detection_threshold_db": 6.0,  # Tiefe Dips: 10–25 dB onset
        "dip_onset_ms": 80.0,  # typische Onset-Zeit (Camras 1988)
        "dip_duration_ms": 250.0,  # typische Gesamtdauer
        "level_smoothing_ms": 120.0,
        "noise_reduction_strength": 0.40,
    },
    # ── v10.0.0: DefectType→CAUSE-Lücken ─────────────────────────────────────
    "clicks": {
        "ar_order": 64,  # AR-Prädiktionsordnung für Janssen-Interpolation
        "click_threshold": 0.10,  # Relativ zum Signal-Peak
        "ola_crossfade_ms": 4.0,
        "noise_reduction_strength": 0.30,
    },
    "dolby_nr_mismatch": {
        "eq_correction_strength": 0.70,
        "hf_shelf_hz": 1000.0,  # Dolby B Eckfrequenz ≈ 1 kHz
        "hf_shelf_db": -8.0,  # Typische Dolby-B-Decode-Rücknahme
        "era_filter": "cassette_1975_2000",  # Nur bei passender Ära aktiv
    },
    "tape_head_level_dip": {
        "dip_detection_threshold_db": 4.0,  # flachere Schwelle als tape_head_contact_instability
        "dip_onset_ms": 100.0,
        "dip_duration_ms": 500.0,
        "level_smoothing_ms": 200.0,
        "noise_reduction_strength": 0.25,
    },
    "scrape_flutter": {
        "scrape_rate_hz_min": 40.0,
        "scrape_rate_hz_max": 120.0,
        "wow_flutter_filter_hz": 85.0,
        "pitch_correction_semitones": 0.12,
        "transient_protect_strength": 0.80,
    },
    "tape_head_clog": {
        "band_gap_lo_hz": 4500.0,
        "band_gap_hi_hz": 12000.0,
        "spectral_repair_strength": 0.55,
        "azimuth_correction_deg": 0.0,
        "dropout_assist_threshold_db": 6.0,
    },
    # ── v10.0.0: CAUSE_PARAMS für neue Kausal-Ursachen ───────────────────────
    "proximity_effect_excess": {
        "lf_shelf_hz": 250.0,  # Proximty-Übergangsfrequenz (Olson 1948)
        "lf_shelf_gain_db": -6.0,  # Adaptive: −4 bis −8 dB basierend auf Severity
        "sub_bass_protect_hz": 60.0,  # Sub-Bass unter 60 Hz nicht anrühren (Rumble-Schutz)
        "eq_slope_db_per_octave": 6.0,  # Sanfter Low-Shelf-Abfall
    },
    "room_mode_resonance": {
        "notch_q": 12.0,  # Schmalbandiger Notch (Q=12 für Raumresonanz)
        "notch_depth_db": -8.0,
        "freq_range_lo_hz": 40.0,
        "freq_range_hi_hz": 200.0,
        "min_peak_prominence_db": 6.0,
    },
    "nr_breathing_artifact": {
        "gain_smooth_ms": 200.0,  # Dolby B charakteristische Release-Zeit
        "modulation_depth_threshold": 0.25,  # Mindest-Modulationstiefe für Aktivierung
        "noise_reduction_strength": 0.35,
    },
    "flutter_spectral_sidebands": {
        "sideband_search_radius_hz": 12.0,  # Suchradius um Peak für Seitenbänder
        "min_sideband_prominence_db": 3.0,
        "min_sideband_count": 3,
    },
    "speed_calibration_error": {
        "pitch_correction_range_semitones": 3.0,  # Maximal 3 HT Korrektur (>3 = anderes Problem)
        "constant_mode": True,  # Kein zeitvarianter Modus — globale Korrektur
        "confidence_threshold": 0.60,
    },
    "overload_distortion": {
        "thd_target_percent": 1.0,  # Ziel nach Korrektur: < 1 % THD
        "odd_harmonic_gate_db": 4.0,  # Mindest H3/H5 über Noise Floor
        "min_loud_frames_percent": 0.30,  # Nur wenn ≥ 30 % der Frames übersteuert
    },
    "lacquer_disc_degradation": {
        "click_threshold_sigma": 4.0,
        "hf_extension_target_hz": 8000.0,  # Lacquer-Ceiling (Acetat degradiert ≥ 8 kHz)
        "noise_reduction_strength": 0.55,  # Stärkere NR für Substrat-Rauschen
        "material_specific": True,  # Nur für MaterialType.LACQUER_DISC
    },
    "cassette_azimuth_tolerance": {
        "azimuth_tolerance_degrees": 0.5,  # Kassetten-Shell-Toleranz ≈ ±0.5°
        "hf_comb_notch_hz": 10000.0,  # Kammfilter-Nullstelle bei 10 kHz (typisch)
        "phase_correction_strength": 0.70,
    },
    "wire_recording_specific": {
        "knot_click_interval_ms": 200.0,  # Typischer Drahtknoten-Abstand bei Heimformat
        "flutter_rate_hz": 5.0,  # Wire-Wicklungs-Flutter ≈ 4–7 Hz
        "noise_reduction_strength": 0.50,
        "material_specific": True,  # Nur für MaterialType.WIRE_RECORDING
    },
}


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SpectralFeatures:
    """Kompakte Signalmerkmale für Bayesianische Diagnose."""

    rms: float = 0.0
    peak: float = 0.0
    dc_offset: float = 0.0
    crest_factor_db: float = 0.0
    spectral_rolloff_hz: float = 5000.0
    hf_energy_ratio: float = 0.5  # Energie oberhalb 4 kHz / Gesamt
    lf_energy_ratio: float = 0.2  # Energie unterhalb 200 Hz / Gesamt
    hum_score: float = 0.0  # Stärke der harmonischen Linien bei 50/60 Hz
    click_density: float = 0.0  # Clicks pro Sekunde
    dropout_density: float = 0.0  # Dropouts pro Sekunde
    pitch_instability: float = 0.0  # σ(F0-Varianz)
    stereo_correlation: float = 1.0  # [-1, 1]
    clip_fraction: float = 0.0  # Anteil geclippter Samples


@dataclass
class RestorationPlan:
    """Priorisierter Restaurierungsplan des CausalDefectReasoner."""

    primary_cause: str
    cause_probabilities: dict[str, float]  # normierte Posterioren
    ranked_causes: list[tuple[str, float]]  # absteigend nach Prob.
    recommended_phases: list[str]  # geordnet
    phase_parameters: dict[str, Any]  # {phase_id: {param: value}}
    confidence: float  # max. Posterior-Wert
    reasoning: str  # menschenlesbare Erklärung
    material: str


# ---------------------------------------------------------------------------
# Merkmals-Extraktion
# ---------------------------------------------------------------------------


def extract_spectral_features(audio: np.ndarray, sample_rate: int) -> SpectralFeatures:
    """
    Extrahiert kompakte Signalmerkmale für das Bayesnetz.

    Args:
        audio:       np.ndarray, mono oder stereo
        sample_rate: Abtastrate
    Returns:
        SpectralFeatures
    """
    if audio.ndim == 2:
        # Stereo: Kanäle trennen für Korrelation
        if audio.shape[0] <= 2:
            ch_l, ch_r = audio[0], audio[1]
            mono = 0.5 * (ch_l + ch_r)
        else:
            mono = np.mean(audio, axis=1)
            ch_l = ch_r = mono
    else:
        mono = audio.astype(np.float64)
        ch_l = ch_r = mono

    mono = mono.astype(np.float64)
    n = len(mono)
    if n < 32:
        return SpectralFeatures()

    sr = sample_rate

    # Basismerkmale
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    dc = float(np.mean(mono))
    cf = float(20 * math.log10(peak / (rms + 1e-9) + 1e-9))

    # FFT-Merkmale
    n_fft = min(4096, _next_pow2(n))
    hop = n_fft // 4
    # Mittleres Powerspektrum
    n_frames = max(1, (n - n_fft) // hop)
    ps = np.zeros(n_fft // 2 + 1)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + n_fft]
        ps += np.abs(np.fft.rfft(seg * np.hanning(n_fft))) ** 2
    ps /= n_frames + 1e-12

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total_energy = np.sum(ps) + 1e-12

    # Spektrale Rolloff
    cumsum = np.cumsum(ps)
    rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # HF / LF Energieverhältnis
    hf_mask = freqs >= 4000.0
    lf_mask = freqs <= 200.0
    hf_ratio = float(np.sum(ps[hf_mask]) / total_energy)
    lf_ratio = float(np.sum(ps[lf_mask]) / total_energy)

    # Brumm-Score: harmonische Linien bei 50/60 Hz
    hum_score = _compute_hum_score(ps, freqs)

    # Click-Dichte (Impuls-Detektor via Z-Score)
    click_density = _compute_click_density(mono, sr)

    # Dropout-Dichte (kurze Stille-Intervalle)
    dropout_density = _compute_dropout_density(mono, sr)

    # Pitch-Instabilität (näherungsweise via AutoKorrelation)
    pitch_instability = _compute_pitch_instability(mono, sr)

    # Stereo-Korrelation
    if ch_l is not ch_r:
        _sl = ch_l[: min(len(ch_l), len(ch_r))]
        _sr = ch_r[: min(len(ch_l), len(ch_r))]
        if np.std(_sl) > 1e-9 and np.std(_sr) > 1e-9:
            _sla = _sl - _sl.mean()
            _sra = _sr - _sr.mean()
            _nl = float(np.linalg.norm(_sla))
            _nr = float(np.linalg.norm(_sra))
            corr = float(np.dot(_sla, _sra) / (_nl * _nr + 1e-10))
        else:
            corr = 1.0 if (np.std(_sl) < 1e-9 and np.std(_sr) < 1e-9) else 0.0
        stereo_corr = float(np.clip(corr, -1.0, 1.0))
    else:
        stereo_corr = 1.0

    # Clipping-Anteil
    clip_thr = 0.97
    clip_frac = float(np.mean(np.abs(mono) >= clip_thr))

    return SpectralFeatures(
        rms=rms,
        peak=peak,
        dc_offset=dc,
        crest_factor_db=cf,
        spectral_rolloff_hz=rolloff_hz,
        hf_energy_ratio=hf_ratio,
        lf_energy_ratio=lf_ratio,
        hum_score=hum_score,
        click_density=click_density,
        dropout_density=dropout_density,
        pitch_instability=pitch_instability,
        stereo_correlation=stereo_corr,
        clip_fraction=clip_frac,
    )


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _compute_hum_score(ps: np.ndarray, freqs: np.ndarray) -> float:
    """Stärke harmonischer Linien bei 50 oder 60 Hz."""
    score_50 = _harmonic_line_score(ps, freqs, 50.0, n_harmonics=5)
    score_60 = _harmonic_line_score(ps, freqs, 60.0, n_harmonics=5)
    return float(max(score_50, score_60))


def _harmonic_line_score(ps: np.ndarray, freqs: np.ndarray, f0: float, n_harmonics: int = 5) -> float:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    score = 0.0
    for k in range(1, n_harmonics + 1):
        fk = k * f0
        idx = round(fk / df)
        idx = min(max(idx, 0), len(ps) - 1)
        # Schmales Fenster ± 2 Bins
        window = ps[max(0, idx - 2) : idx + 3]
        line_energy = np.max(window) if len(window) > 0 else 0.0
        # Vergleich mit lokal umgebendem Hintergrund
        bg_lo = ps[max(0, idx - 10) : max(0, idx - 3)]
        bg_hi = ps[min(len(ps) - 1, idx + 3) : min(len(ps), idx + 10)]
        bg = np.mean(np.concatenate([bg_lo, bg_hi])) + 1e-12
        score += min(line_energy / bg, 10.0)  # type: ignore[operator]  # SNR der Linie
    return score / (n_harmonics * 10.0)  # normiert auf [0, 1]


def _compute_click_density(mono: np.ndarray, sr: int) -> float:
    """Clicks pro Sekunde mittels robustem Z-Score."""
    diff = np.diff(mono)
    np.median(np.abs(diff))
    mad = np.median(np.abs(diff - np.median(diff))) + 1e-9
    z = np.abs(diff - np.median(diff)) / (1.4826 * mad)
    clicks = float(np.sum(z > 8.0))
    dur_s = len(mono) / (sr + 1e-9)
    return clicks / (dur_s + 1e-9)


def _compute_dropout_density(mono: np.ndarray, sr: int) -> float:
    """Dropouts pro Sekunde: kurze Intervalle mit sehr kleiner Energie."""
    frame_len = int(sr * 0.005)  # 5 ms Frames
    if frame_len < 1 or len(mono) < frame_len:
        return 0.0
    n_frames = len(mono) // frame_len
    frame_rms = np.array([np.sqrt(np.mean(mono[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)])
    global_rms = np.mean(frame_rms) + 1e-9
    dropout_frames: int = int(np.sum(frame_rms < 0.05 * global_rms))
    dur_s = len(mono) / (sr + 1e-9)
    return float(dropout_frames * 0.005 / dur_s)


def _compute_pitch_instability(mono: np.ndarray, sr: int) -> float:
    """Grobe Pitch-Instabilität mittels autokorrelationsbasierter F0-Verfolgung."""
    frame_len = int(sr * 0.050)  # 50 ms Frames
    hop = frame_len // 2
    if frame_len < 16 or len(mono) < frame_len:
        return 0.0
    f0_values = []
    for i in range(0, len(mono) - frame_len, hop):
        seg = mono[i : i + frame_len]
        # Autokorrelation — FFT-based O(N log N)
        from backend.core.core_utils import fft_autocorr  # pylint: disable=import-outside-toplevel

        ac = fft_autocorr(seg)
        ac = ac / (ac[0] + 1e-9)
        # Suche Peak zwischen 2 ms und 20 ms (50 Hz – 500 Hz)
        lo = max(1, int(sr * 0.002))
        hi = min(len(ac) - 1, int(sr * 0.020))
        if lo >= hi:
            continue
        peak_idx = int(np.argmax(ac[lo:hi])) + lo
        if ac[peak_idx] > 0.3:
            f0 = sr / (peak_idx + 1)
            f0_values.append(f0)
    if len(f0_values) < 2:
        return 0.0
    f0_arr = np.array(f0_values)
    return float(np.std(f0_arr) / (np.mean(f0_arr) + 1e-9))


# ---------------------------------------------------------------------------
# Likelihood-Funktionen P(Merkmale | Ursache)
# ---------------------------------------------------------------------------


def _likelihood_tape_dropout(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """Bedingte Wahrscheinlichkeit für Tape-Dropout."""
    p = 0.0
    p += _gaussian_score(sf.dropout_density, mu=0.5, sigma=0.3) * 0.40
    p += _gaussian_score(sf.click_density, mu=0.5, sigma=0.5) * 0.15
    p += _sigmoid_score(defect_scores.get("dropout_severity", 0.0), k=8, x0=0.4) * 0.30
    p += _sigmoid_score(defect_scores.get("silence_ratio", 0.0), k=6, x0=0.2) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_tape_hiss(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    p = 0.0
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.35, sigma=0.15) * 0.40
    p += _sigmoid_score(defect_scores.get("noise_floor_db", -60.0) + 60.0, k=0.1, x0=30.0) * 0.35
    p += (1.0 - sf.hum_score) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_vinyl_crackle(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    p = 0.0
    crackle_sev = float(defect_scores.get("crackle", 0.0))
    click_sev = float(defect_scores.get("click_severity", 0.0))
    direct_surface_evidence = max(crackle_sev, click_sev)
    p += _sigmoid_score(crackle_sev, k=8, x0=0.30) * 0.35
    p += _sigmoid_score(click_sev, k=8, x0=0.35) * 0.25
    p += _sigmoid_score(sf.click_density, k=1.0, x0=2.0) * 0.25
    if direct_surface_evidence > 0.0 or sf.click_density >= 2.0:
        p += (1.0 - abs(sf.dc_offset)) * 0.08
        p += (1.0 - sf.clip_fraction) * 0.07
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_vinyl_warp(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    p = 0.0
    p += _sigmoid_score(sf.pitch_instability, k=20, x0=0.02) * 0.50
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.30, sigma=0.15) * 0.30
    p += _gaussian_score(float(defect_scores.get("wow_flutter", 0.0)), mu=0.4, sigma=0.2) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_electrical_hum(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:  # pylint: disable=unused-argument
    p = 0.0
    p += sf.hum_score * 0.60
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.35, sigma=0.15) * 0.25
    p += sf.stereo_correlation * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_head_misalignment(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    p = 0.0
    # HF-Verlust: rolloff deutlich unter 10 kHz
    hf_loss = max(0.0, 1.0 - sf.spectral_rolloff_hz / 10000.0)
    p += hf_loss * 0.45
    p += (1.0 - sf.hf_energy_ratio) * 0.30
    p += _sigmoid_score(float(defect_scores.get("azimuth_error", 0.0)), k=10, x0=0.3) * 0.25
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_dc_offset(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:  # pylint: disable=unused-argument
    p = 0.0
    p += _sigmoid_score(abs(sf.dc_offset), k=20, x0=0.03) * 0.70
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.40, sigma=0.15) * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_digital_clip(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    p = 0.0
    p += _sigmoid_score(sf.clip_fraction, k=30, x0=0.02) * 0.55
    p += _sigmoid_score(sf.crest_factor_db, k=0.1, x0=0.0) * 0.25  # niedriger Crestfaktor
    p += _sigmoid_score(float(defect_scores.get("clip_severity", 0.0)), k=8, x0=0.3) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_soft_saturation(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:  # pylint: disable=unused-argument
    """P(Merkmale | soft_saturation) — Röhren-/Tape-Sättigung (gerade Obertöne).

    Soft-Saturation erzeugt gerade Harmonische (H2, H4) und runde Wellenformen
    ohne Flat-Tops. Spectral Flatness im Clip-Bereich ist niedrig (< 0.3).
    Spec §6.3: BEWAHREN — Phasen-Liste bleibt leer, aber die Ursache muss
    im Bayes-Posterior korrekt erscheinen, damit keine destruktiven Phasen
    (z.B. phase_23_spectral_repair als Clipping-Fix) fälschlicherweise aktiviert werden.
    """
    p = 0.0
    # Hinweis auf Sättigung: Clip-Anteil niedrig, Crest-Faktor moderat
    p += _gaussian_score(sf.clip_fraction, mu=0.005, sigma=0.01) * 0.40  # kaum Flat-Tops
    p += _gaussian_score(sf.crest_factor_db, mu=8.0, sigma=4.0) * 0.30  # typischer Crest
    p += (1.0 - sf.clip_fraction) * 0.30
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_head_wear(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | head_wear) — Kopfverschleiß / Frequenzband-Auslöschung.

    Charakteristisch: vollständiger HF-Energie-Verlust (> 3 kHz) über die
    gesamte Dateilänge, sehr niedriger Spectral-Rolloff (§4.5, §6.3).
    """
    p = 0.0
    # Niedriger HF-Anteil — Hauptmerkmal
    hf_loss = max(0.0, 1.0 - sf.hf_energy_ratio / 0.05)  # < 5 % HF-Energie
    p += hf_loss * 0.50
    # Rolloff deutlich unter 5 kHz
    rolloff_loss = max(0.0, 1.0 - sf.spectral_rolloff_hz / 5000.0)
    p += rolloff_loss * 0.35
    p += _sigmoid_score(float(defect_scores.get("azimuth_error", 0.0)), k=8, x0=0.5) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_print_through(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | print_through) — Magnetisches Vorecho / Nachecho (Tape).

    Print-Through äußert sich als sehr schwaches Geister-Signal (typisch −20 bis
    −30 dB) kurz vor/nach dem Hauptsignal. Als Proxy: hoher Dropout-Score bei
    gleichzeitig vorhandenem Tape-Hiss-Profil (Reel-Tape-Context).
    """
    p = 0.0
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.30, sigma=0.12) * 0.35  # Tape-HF-Profil
    p += _sigmoid_score(defect_scores.get("noise_floor_db", -60.0) + 60.0, k=0.08, x0=25.0) * 0.35
    p += _gaussian_score(sf.dropout_density, mu=0.1, sigma=0.15) * 0.30
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_transport_bump(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | transport_bump) — Impulsartige Mikro-Geschwindigkeitssprünge (Kassette/Tape).

    Transport bumps manifest as short (50–300 ms) pitch+amplitude excursions,
    similar to dropout but with preserved signal energy (no silence gap).
    Proxy: wow/flutter indicators with short transient character.
    """
    p = 0.0
    # Wow/Flutter-like pitch instability
    p += _sigmoid_score(float(defect_scores.get("wow_severity", 0.0)), k=6, x0=0.3) * 0.30
    p += _sigmoid_score(float(defect_scores.get("flutter_severity", 0.0)), k=6, x0=0.3) * 0.20
    # Dropout indicators (bumps can resemble short dropouts)
    p += _gaussian_score(sf.dropout_density, mu=0.05, sigma=0.10) * 0.25
    # Tape-like HF profile (transport bumps are tape/cassette artifacts)
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.25, sigma=0.15) * 0.15
    # Pitch instability — bumps cause sudden pitch changes
    p += _sigmoid_score(sf.pitch_instability, k=4, x0=0.4) * 0.10
    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Likelihood-Funktionen für die 22 erweiterten Ursachen (v10.0.0b)
# ---------------------------------------------------------------------------


def _likelihood_bandwidth_loss(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | bandwidth_loss) — HF-Rolloff / Bandbreitenbegrenzung."""
    p = 0.0
    # Niedriger Spectral Rolloff — Hauptmerkmal
    rolloff_loss = max(0.0, 1.0 - sf.spectral_rolloff_hz / 12000.0)
    p += rolloff_loss * 0.40
    # Wenig HF-Energie
    p += (1.0 - sf.hf_energy_ratio) * 0.30
    # DefectScanner bandwidth_loss severity
    p += _sigmoid_score(float(defect_scores.get("bandwidth_loss", 0.0)), k=8, x0=0.3) * 0.30
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_high_freq_noise(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | high_freq_noise) — Hochfrequenzrauschen (distinct from tape_hiss)."""
    p = 0.0
    # Hoher HF-Anteil (Rauschen im HF-Band)
    p += _sigmoid_score(sf.hf_energy_ratio, k=8, x0=0.40) * 0.40
    p += _sigmoid_score(float(defect_scores.get("high_freq_noise", 0.0)), k=8, x0=0.25) * 0.35
    # Kein Brumm (unterscheidet von electrical_hum)
    p += (1.0 - sf.hum_score) * 0.15
    # Kein Clipping
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_stereo_imbalance(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | stereo_imbalance) — L/R-Pegelunterschied."""
    p = 0.0
    # Low stereo correlation suggests imbalance
    p += (1.0 - abs(sf.stereo_correlation)) * 0.45
    p += _sigmoid_score(float(defect_scores.get("stereo_imbalance", 0.0)), k=8, x0=0.25) * 0.40
    # Normal clip fraction (not a clipping issue)
    p += (1.0 - sf.clip_fraction) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_phase_issues(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | phase_issues) — Phasenverschiebung zwischen Kanälen."""
    p = 0.0
    # Anti-phase correlation is the primary indicator
    anti_phase = max(0.0, -sf.stereo_correlation)
    p += anti_phase * 0.50
    p += _sigmoid_score(float(defect_scores.get("phase_issues", 0.0)), k=8, x0=0.25) * 0.35
    # Low HF energy from cancellation
    p += (1.0 - sf.hf_energy_ratio) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_pitch_drift(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | pitch_drift) — Konstanter Geschwindigkeitsfehler."""
    p = 0.0
    # Moderate pitch instability (steady drift, not random flutter)
    p += _gaussian_score(sf.pitch_instability, mu=0.03, sigma=0.02) * 0.40
    p += _sigmoid_score(float(defect_scores.get("pitch_drift", 0.0)), k=8, x0=0.25) * 0.40
    # No dropout (distinguishes from tape dropout)
    p += (1.0 - sf.dropout_density) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_reverb_excess(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | reverb_excess) — Übermäßiger Raumhall."""
    p = 0.0
    # High LF energy ratio (reverb tail energy)
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.30, sigma=0.12) * 0.30
    p += _sigmoid_score(float(defect_scores.get("reverb_excess", 0.0)), k=8, x0=0.25) * 0.45
    # Moderate HF (reverb preserves some HF)
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.20, sigma=0.10) * 0.15
    # No clipping
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_digital_artifacts(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | digital_artifacts) — Codec-Artefakte (Quantisierungsreste, Ringmodulation)."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("digital_artifacts", 0.0)), k=8, x0=0.25) * 0.45
    # Moderate clip fraction (digital distortion)
    p += _sigmoid_score(sf.clip_fraction, k=20, x0=0.005) * 0.25
    # High crest factor (intermittent artifacts)
    p += _gaussian_score(sf.crest_factor_db, mu=12.0, sigma=5.0) * 0.20
    p += (1.0 - sf.hum_score) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_compression_artifacts(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | compression_artifacts) — MP3/AAC Codec-Artefakte."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("compression_artifacts", 0.0)), k=8, x0=0.25) * 0.45
    # Bandwidth loss above ~16 kHz typical for lossy codecs
    rolloff_codec = max(0.0, 1.0 - sf.spectral_rolloff_hz / 16000.0)
    p += rolloff_codec * 0.25
    # Low HF energy
    p += (1.0 - sf.hf_energy_ratio) * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_quantization_noise(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | quantization_noise) — Bit-Tiefe-bedingtes Rauschen."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("quantization_noise", 0.0)), k=8, x0=0.25) * 0.45
    # Flat spectral noise floor (quantization noise is white-ish)
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.50, sigma=0.15) * 0.25
    # No clicks (distinguishes from vinyl)
    p += (1.0 - min(1.0, sf.click_density / 5.0)) * 0.15
    p += (1.0 - sf.hum_score) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_jitter_artifacts(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | jitter_artifacts) — Zeitgitter-Fehler D/A-Wandlung."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("jitter_artifacts", 0.0)), k=8, x0=0.25) * 0.50
    # Slight pitch instability from clock jitter
    p += _gaussian_score(sf.pitch_instability, mu=0.005, sigma=0.005) * 0.25
    # HF distortion from sample-hold errors
    p += _sigmoid_score(sf.hf_energy_ratio, k=5, x0=0.35) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_dynamic_compression_excess(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | dynamic_compression_excess) — Loudness-War Over-Limiting."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("dynamic_compression_excess", 0.0)), k=8, x0=0.25) * 0.40
    # Very low crest factor — hallmark of over-compression
    low_crest = max(0.0, 1.0 - sf.crest_factor_db / 6.0)
    p += low_crest * 0.30
    # High clip fraction from inter-sample peaks
    p += _sigmoid_score(sf.clip_fraction, k=15, x0=0.01) * 0.20
    # High RMS (loud master)
    p += _sigmoid_score(sf.rms, k=8, x0=0.3) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_pre_echo(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | pre_echo) — Codec-Pre-Echo vor Transienten (MP3 Long-Window)."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("pre_echo", 0.0)), k=8, x0=0.25) * 0.50
    # Moderate click density (pre-echo resembles soft transient artifacts)
    p += _gaussian_score(sf.click_density, mu=0.5, sigma=0.5) * 0.20
    # Bandwidth loss from codec
    rolloff_codec = max(0.0, 1.0 - sf.spectral_rolloff_hz / 16000.0)
    p += rolloff_codec * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_low_freq_rumble(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | low_freq_rumble) — Subsonic-/LF-Störung."""
    p = 0.0
    # High LF energy — primary indicator
    p += _sigmoid_score(sf.lf_energy_ratio, k=8, x0=0.25) * 0.40
    p += _sigmoid_score(float(defect_scores.get("low_freq_rumble", 0.0)), k=8, x0=0.25) * 0.35
    # No hum (distinguishes from electrical_hum — rumble is broadband LF)
    p += (1.0 - sf.hum_score) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_transient_smearing(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | transient_smearing) — Ansatzverschmierung durch Kompression/Limiter."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("transient_smearing", 0.0)), k=8, x0=0.25) * 0.45
    # Low crest factor (compressed dynamics flatten transients)
    low_crest = max(0.0, 1.0 - sf.crest_factor_db / 8.0)
    p += low_crest * 0.30
    # Low click density (transients are smoothed out)
    p += (1.0 - min(1.0, sf.click_density / 3.0)) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_vocal_harshness(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | vocal_harshness) — Vokalhärte im Präsenzband (2-6 kHz)."""
    p = 0.0
    harsh_sev = float(defect_scores.get("vocal_harshness", 0.0))
    p += _sigmoid_score(harsh_sev, k=8, x0=0.20) * 0.55
    sib_sev = float(defect_scores.get("sibilance", 0.0))
    p += _sigmoid_score(sib_sev, k=6, x0=0.20) * 0.20
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.38, sigma=0.18) * 0.15
    p += _gaussian_score(sf.spectral_rolloff_hz, mu=7000.0, sigma=2500.0) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_clipping(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | clipping) — Generisches Clipping (analog + digital)."""
    p = 0.0
    p += _sigmoid_score(sf.clip_fraction, k=25, x0=0.02) * 0.45
    p += _sigmoid_score(float(defect_scores.get("clipping", 0.0)), k=8, x0=0.3) * 0.30
    # Low crest factor (clipped peaks)
    low_crest = max(0.0, 1.0 - sf.crest_factor_db / 4.0)
    p += low_crest * 0.15
    p += _sigmoid_score(sf.peak, k=10, x0=0.95) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_riaa_curve_error(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | riaa_curve_error) — Falsche Disc-Entzerrungskurve."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("riaa_curve_error", 0.0)), k=8, x0=0.25) * 0.50
    # Abnormal spectral tilt (either too bright or too dark)
    spectral_imbalance = abs(sf.hf_energy_ratio - 0.25)
    p += _sigmoid_score(spectral_imbalance, k=8, x0=0.15) * 0.30
    # High LF energy (bass boost from wrong curve)
    p += _sigmoid_score(sf.lf_energy_ratio, k=6, x0=0.30) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_aliasing(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | aliasing) — ADC-Spiegelfrequenzen bei fehlendem AA-Filter."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("aliasing", 0.0)), k=8, x0=0.25) * 0.50
    # Unusually high HF energy (mirror frequencies fold back)
    p += _sigmoid_score(sf.hf_energy_ratio, k=6, x0=0.40) * 0.25
    # No clicks (not vinyl crackle)
    p += (1.0 - min(1.0, sf.click_density / 5.0)) * 0.15
    p += (1.0 - sf.hum_score) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_bias_error(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | bias_error) — Falscher Vormagnetisierungsstrom bei Bandaufnahme."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("bias_error", 0.0)), k=8, x0=0.25) * 0.45
    # Spectral tilt — under-biased: bright+noisy; over-biased: dull+distorted
    spectral_tilt = abs(sf.hf_energy_ratio - 0.20)
    p += _sigmoid_score(spectral_tilt, k=6, x0=0.15) * 0.30
    # Elevated noise floor (bias-related noise)
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.35, sigma=0.12) * 0.15
    # Tape context (low clip, no click)
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_sibilance(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | sibilance) — Zischlautüberbetonung > 6 kHz."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("sibilance", 0.0)), k=8, x0=0.25) * 0.50
    # High HF energy (sibilance is HF-dominant)
    p += _sigmoid_score(sf.hf_energy_ratio, k=6, x0=0.35) * 0.25
    # High rolloff (energy extends to high frequencies)
    rolloff_high = min(1.0, sf.spectral_rolloff_hz / 15000.0)
    p += rolloff_high * 0.15
    # No clipping
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_wow(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | wow) — Tonhöhenschwankung < 0.5 Hz (Motor-Exzentrizität)."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("wow", 0.0)), k=8, x0=0.25) * 0.40
    # Pitch instability — wow causes slow pitch modulation
    p += _sigmoid_score(sf.pitch_instability, k=15, x0=0.02) * 0.35
    # Low dropout density (wow ≠ dropout)
    p += (1.0 - sf.dropout_density) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_flutter(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | flutter) — Tonhöhenschwankung 0.5–200 Hz (Bandschwankung)."""
    p = 0.0
    p += _sigmoid_score(float(defect_scores.get("flutter", 0.0)), k=8, x0=0.25) * 0.40
    # Higher pitch instability than wow
    p += _sigmoid_score(sf.pitch_instability, k=10, x0=0.04) * 0.30
    # Click density from flutter-induced amplitude modulation
    p += _gaussian_score(sf.click_density, mu=0.3, sigma=0.3) * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_wow_flutter(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | wow_flutter) — Kombiniertes Wow+Flutter."""
    p = 0.0
    wow_sev = float(defect_scores.get("wow", 0.0))
    flutter_sev = float(defect_scores.get("flutter", 0.0))
    combined = max(wow_sev, flutter_sev)
    p += _sigmoid_score(combined, k=6, x0=0.25) * 0.40
    p += _sigmoid_score(sf.pitch_instability, k=12, x0=0.03) * 0.35
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.20, sigma=0.10) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_tape_head_contact(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | tape_head_contact_instability) — level dips from head pressure variation."""
    p = 0.0
    # Primary evidence: detected tape head level dips
    dip_sev = float(defect_scores.get("tape_head_level_dip", 0.0))
    p += _sigmoid_score(dip_sev, k=8, x0=0.15) * 0.50
    # Secondary: dropout evidence (deep dips cross dropout threshold)
    dropout_sev = float(defect_scores.get("dropouts", 0.0))
    p += _sigmoid_score(dropout_sev, k=5, x0=0.20) * 0.20
    # Tertiary: also accompanied by wow/flutter (same transport mechanism)
    wow_sev = float(defect_scores.get("wow", 0.0))
    flutter_sev = float(defect_scores.get("flutter", 0.0))
    p += _sigmoid_score(max(wow_sev, flutter_sev), k=4, x0=0.15) * 0.15
    # Low clip fraction expected (analog source, not digital clipping)
    p += (1.0 - sf.clip_fraction) * 0.15
    return float(np.clip(p, 0.0, 1.0))


# ── v10.0.0: 12 neue Likelihood-Funktionen ─────────────────────────────────


def _likelihood_modulation_noise(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | modulation_noise) — signal-dependent noise modulation."""
    p = 0.0
    mod_sev = float(defect_scores.get("modulation_noise", 0.0))
    p += _sigmoid_score(mod_sev, k=8, x0=0.25) * 0.45
    hiss_sev = float(defect_scores.get("tape_hiss", 0.0))
    p += _sigmoid_score(hiss_sev, k=5, x0=0.20) * 0.25
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.30, sigma=0.15) * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_inner_groove_distortion(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | inner_groove_distortion) — THD increasing with groove radius."""
    p = 0.0
    igd_sev = float(defect_scores.get("inner_groove_distortion", 0.0))
    p += _sigmoid_score(igd_sev, k=8, x0=0.20) * 0.50
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.40, sigma=0.20) * 0.20
    crackle_sev = float(defect_scores.get("crackle", 0.0))
    p += _sigmoid_score(crackle_sev, k=4, x0=0.15) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_groove_echo(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | groove_echo) — pre-echo from adjacent groove deformation."""
    p = 0.0
    echo_sev = float(defect_scores.get("groove_echo", 0.0))
    p += _sigmoid_score(echo_sev, k=8, x0=0.20) * 0.50
    crackle_sev = float(defect_scores.get("crackle", 0.0))
    p += _sigmoid_score(crackle_sev, k=4, x0=0.10) * 0.20
    p += _gaussian_score(sf.stereo_correlation, mu=0.85, sigma=0.15) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_crosstalk(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | crosstalk) — channel separation degradation."""
    p = 0.0
    xt_sev = float(defect_scores.get("crosstalk", 0.0))
    p += _sigmoid_score(xt_sev, k=8, x0=0.20) * 0.50
    # High stereo correlation = poor separation = crosstalk
    p += _sigmoid_score(sf.stereo_correlation, k=5, x0=0.85) * 0.30
    p += (1.0 - sf.clip_fraction) * 0.10
    p += (1.0 - abs(sf.dc_offset)) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_intermodulation_distortion(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | intermodulation_distortion) — nonlinear sum/difference products."""
    p = 0.0
    imd_sev = float(defect_scores.get("intermodulation_distortion", 0.0))
    p += _sigmoid_score(imd_sev, k=8, x0=0.20) * 0.50
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.35, sigma=0.20) * 0.20
    clip_sev = float(defect_scores.get("clipping", 0.0))
    p += _sigmoid_score(clip_sev, k=4, x0=0.15) * 0.15
    p += (1.0 - abs(sf.dc_offset)) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_tape_splice(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | tape_splice_artifact) — click + level jump at splices."""
    p = 0.0
    splice_sev = float(defect_scores.get("tape_splice_artifact", 0.0))
    p += _sigmoid_score(splice_sev, k=8, x0=0.20) * 0.50
    click_sev = float(defect_scores.get("click_severity", 0.0))
    p += _sigmoid_score(click_sev, k=4, x0=0.20) * 0.20
    dropout_sev = float(defect_scores.get("dropouts", 0.0))
    p += _sigmoid_score(dropout_sev, k=4, x0=0.15) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_hf_remanence_loss(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | hf_remanence_loss) — magnetic particle demagnetization."""
    p = 0.0
    hf_loss_sev = float(defect_scores.get("hf_remanence_loss", 0.0))
    p += _sigmoid_score(hf_loss_sev, k=8, x0=0.20) * 0.45
    bw_sev = float(defect_scores.get("bandwidth_loss", 0.0))
    p += _sigmoid_score(bw_sev, k=5, x0=0.20) * 0.25
    # Low HF energy = HF loss
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.10, sigma=0.10) * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_stylus_damage(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | stylus_damage) — asymmetric distortion from worn stylus."""
    p = 0.0
    stylus_sev = float(defect_scores.get("stylus_damage", 0.0))
    p += _sigmoid_score(stylus_sev, k=8, x0=0.20) * 0.50
    crackle_sev = float(defect_scores.get("crackle", 0.0))
    p += _sigmoid_score(crackle_sev, k=4, x0=0.15) * 0.20
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.40, sigma=0.20) * 0.15
    p += (1.0 - abs(sf.dc_offset)) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_sticky_shed(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | sticky_shed_residue) — post-baking tape degradation."""
    p = 0.0
    shed_sev = float(defect_scores.get("sticky_shed_residue", 0.0))
    p += _sigmoid_score(shed_sev, k=8, x0=0.20) * 0.50
    dropout_sev = float(defect_scores.get("dropouts", 0.0))
    p += _sigmoid_score(dropout_sev, k=5, x0=0.15) * 0.20
    hiss_sev = float(defect_scores.get("tape_hiss", 0.0))
    p += _sigmoid_score(hiss_sev, k=4, x0=0.15) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_multiband_wow_flutter(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | multiband_wow_flutter) — frequency-dependent speed fluctuations."""
    p = 0.0
    mb_sev = float(defect_scores.get("multiband_wow_flutter", 0.0))
    p += _sigmoid_score(mb_sev, k=8, x0=0.20) * 0.45
    wf_sev = float(defect_scores.get("wow_flutter", 0.0))
    p += _sigmoid_score(wf_sev, k=5, x0=0.15) * 0.25
    p += _sigmoid_score(sf.pitch_instability, k=20, x0=0.02) * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_generation_loss(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | generation_loss) — cumulative dubbing/transcoding degradation."""
    p = 0.0
    gen_sev = float(defect_scores.get("generation_loss", 0.0))
    p += _sigmoid_score(gen_sev, k=8, x0=0.20) * 0.45
    bw_sev = float(defect_scores.get("bandwidth_loss", 0.0))
    p += _sigmoid_score(bw_sev, k=5, x0=0.15) * 0.20
    hiss_sev = float(defect_scores.get("tape_hiss", 0.0))
    p += _sigmoid_score(hiss_sev, k=4, x0=0.15) * 0.20
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.15, sigma=0.10) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_motor_interference(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | motor_interference) — motor harmonics in 80–300 Hz."""
    p = 0.0
    motor_sev = float(defect_scores.get("motor_interference", 0.0))
    p += _sigmoid_score(motor_sev, k=8, x0=0.20) * 0.45
    hum_sev = float(defect_scores.get("hum", 0.0))
    p += _sigmoid_score(hum_sev, k=5, x0=0.15) * 0.20
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.35, sigma=0.15) * 0.20
    p += _gaussian_score(sf.hum_score, mu=0.40, sigma=0.20) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_amplitude_drift(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | amplitude_drift) — gradual level rise/fall (AGC, oxide, motor-temp). v10.0.0"""
    p = 0.0
    # Primary: DefectScanner severity is the strongest signal
    drift_sev = float(defect_scores.get("amplitude_drift", 0.0))
    p += _sigmoid_score(drift_sev, k=10, x0=0.30) * 0.75
    # Secondary: high crest_factor variance (loudness War-free signals show stable CF)
    p += _gaussian_score(sf.crest_factor_db, mu=12.0, sigma=8.0) * 0.15
    # Tertiary: low RMS (drifting signals often start quiet) as weak tie-breaker
    p += _gaussian_score(sf.rms, mu=0.10, sigma=0.12) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_clicks(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | clicks) — impulsartige Einzelstörungen (Vinyl/Shellac/Band). v10.0.0"""
    p = 0.0
    # Primary: direct DefectScanner score (DefectType.CLICKS)
    click_sev = float(defect_scores.get("clicks", 0.0))
    p += _sigmoid_score(click_sev, k=10, x0=0.25) * 0.70
    # Secondary: SpectralFeatures click_density (Clicks pro Sekunde)
    p += _sigmoid_score(sf.click_density, k=8, x0=0.30) * 0.20
    # Tertiary: high crest_factor indicates transient spikes
    p += _sigmoid_score(sf.crest_factor_db, k=3, x0=20.0) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_dolby_nr_mismatch(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | dolby_nr_mismatch) — Dolby B/C/S encode ohne Dekodierung → +6..20 dB HF. v10.0.0

    Signature: excessive high-frequency energy (spectral_rolloff very high, hf_energy_ratio elevated)
    combined with a DefectScanner DOLBY_NR_MISMATCH score.
    """
    p = 0.0
    # Primary: direct DefectScanner score
    dolby_sev = float(defect_scores.get("dolby_nr_mismatch", 0.0))
    p += _sigmoid_score(dolby_sev, k=10, x0=0.30) * 0.60
    # Secondary: elevated HF energy ratio (Dolby encoded → too much treble)
    # Dolby B pushes HF up by 6–10 dB; above 8 kHz HF ratio typically > 0.35
    p += _sigmoid_score(sf.hf_energy_ratio, k=15, x0=0.35) * 0.25
    # Tertiary: high spectral rolloff (HF-heavy signal)
    p += _sigmoid_score(sf.spectral_rolloff_hz, k=0.0003, x0=14000.0) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_tape_head_level_dip(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | tape_head_level_dip) — graduelle Pegeleinbrüche durch Bandkopf-Kontaktdruckvariation. v10.0.0"""
    p = 0.0
    # Primary: direct DefectScanner score (DefectType.TAPE_HEAD_LEVEL_DIP)
    dip_sev = float(defect_scores.get("tape_head_level_dip", 0.0))
    p += _sigmoid_score(dip_sev, k=10, x0=0.25) * 0.65
    # Secondary: dropout density (head-tape contact loss manifests as dropout-like dips)
    p += _sigmoid_score(sf.dropout_density, k=8, x0=0.20) * 0.20
    # Tertiary: low average RMS (sustained dips lower mean energy)
    p += _gaussian_score(sf.rms, mu=0.06, sigma=0.08) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_scrape_flutter(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | scrape_flutter) — hochfrequente Bandführungsmodulation. v10.0.0"""
    p = 0.0
    scrape_sev = float(defect_scores.get("scrape_flutter", 0.0))
    p += _sigmoid_score(scrape_sev, k=10, x0=0.20) * 0.60
    flutter_sev = float(defect_scores.get("flutter", 0.0))
    sideband_sev = float(defect_scores.get("flutter_spectral_sidebands", 0.0))
    p += _sigmoid_score(max(flutter_sev, sideband_sev), k=8, x0=0.18) * 0.25
    p += _gaussian_score(sf.spectral_rolloff_hz, mu=6500.0, sigma=3000.0) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_tape_head_clog(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | tape_head_clog) — lokale HF-Auslöschung durch zugesetzten Magnetkopf. v10.0.0"""
    p = 0.0
    clog_sev = float(defect_scores.get("tape_head_clog", 0.0))
    p += _sigmoid_score(clog_sev, k=10, x0=0.20) * 0.65
    p += _sigmoid_score(float(defect_scores.get("head_wear", 0.0)), k=8, x0=0.30) * 0.15
    hf_absence = float(1.0 - np.clip(sf.hf_energy_ratio / 0.08, 0.0, 1.0))
    p += hf_absence * 0.10
    p += _sigmoid_score(sf.dropout_density, k=8, x0=0.08) * 0.10
    return float(np.clip(p, 0.0, 1.0))


# ── v10.0.0: Likelihood-Funktionen für 9 neue Kausal-Ursachen ────────────────


def _likelihood_proximity_effect_excess(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | proximity_effect_excess) — Nahbesprechungseffekt bei Richtmikrofonen. v10.0.0"""
    p = 0.0
    # Primary: direct DefectScanner score
    prox_sev = float(defect_scores.get("proximity_effect_excess", 0.0))
    p += _sigmoid_score(prox_sev, k=10, x0=0.25) * 0.60
    # Secondary: low spectral rolloff (HF absent, LF dominant)
    p += _gaussian_score(sf.spectral_rolloff_hz, mu=3500.0, sigma=2000.0) * 0.20
    # Tertiary: low HF energy (proximity shifts energy down)
    hf_absence = float(1.0 - np.clip(sf.hf_energy_ratio * 4.0, 0.0, 1.0))
    p += hf_absence * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_room_mode_resonance(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | room_mode_resonance) — Stehwellen-Resonanzen 40–200 Hz. v10.0.0"""
    p = 0.0
    # Primary: direct DefectScanner score
    mode_sev = float(defect_scores.get("room_mode_resonance", 0.0))
    p += _sigmoid_score(mode_sev, k=10, x0=0.20) * 0.60
    # Secondary: low spectral rolloff (room modes → energy accumulates in LF)
    p += _gaussian_score(sf.spectral_rolloff_hz, mu=4000.0, sigma=2500.0) * 0.25
    # Tertiary: non-zero RMS (room modes only audible in signal-bearing material)
    p += _sigmoid_score(sf.rms, k=15, x0=0.05) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_nr_breathing_artifact(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | nr_breathing_artifact) — Dolby/dbx NR Pumpen/Atmen. v10.0.0"""
    p = 0.0
    # Primary: direct DefectScanner score
    breath_sev = float(defect_scores.get("nr_breathing_artifact", 0.0))
    p += _sigmoid_score(breath_sev, k=10, x0=0.20) * 0.70
    # Secondary: Dolby NR mismatch score (related phenomenon)
    dolby_sev = float(defect_scores.get("dolby_nr_mismatch", 0.0))
    p += _sigmoid_score(dolby_sev, k=8, x0=0.30) * 0.20
    # Tertiary: high-frequency noise pattern (NR breathing affects HF region)
    p += _sigmoid_score(sf.hf_energy_ratio, k=8, x0=0.3) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_flutter_spectral_sidebands(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | flutter_spectral_sidebands) — Flutter-Seitenbänder um tonale Peaks. v10.0.0"""
    p = 0.0
    # Primary: direct DefectScanner score
    sb_sev = float(defect_scores.get("flutter_spectral_sidebands", 0.0))
    p += _sigmoid_score(sb_sev, k=10, x0=0.20) * 0.60
    # Secondary: flutter score (sidebands require flutter source)
    flutter_sev = float(defect_scores.get("flutter", 0.0))
    p += _sigmoid_score(flutter_sev, k=8, x0=0.25) * 0.25
    # Tertiary: pitch instability (correlated with flutter mechanism)
    p += _sigmoid_score(sf.pitch_instability, k=8, x0=0.20) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_speed_calibration_error(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | speed_calibration_error) — Konstanter Geschwindigkeitsfehler. v10.0.0"""
    p = 0.0
    # Primary: direct DefectScanner score
    speed_sev = float(defect_scores.get("speed_calibration_error", 0.0))
    p += _sigmoid_score(speed_sev, k=10, x0=0.20) * 0.65
    # Secondary: pitch drift (speed error presents as flat offset in pitch)
    pitch_drift = float(defect_scores.get("pitch_drift", 0.0))
    p += _sigmoid_score(pitch_drift, k=8, x0=0.30) * 0.25
    # Tertiary: pitch instability (low for speed error vs high for wow/flutter)
    # Speed calib: globally offset but temporally stable → lower instability
    p += _gaussian_score(sf.pitch_instability, mu=0.05, sigma=0.10) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_overload_distortion(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | overload_distortion) — Analoger Preamp/Console-Klirr H3/H5. v10.0.0"""
    _ = sf.rms
    p = 0.0
    # Primary: direct DefectScanner score
    over_sev = float(defect_scores.get("overload_distortion", 0.0))
    p += _sigmoid_score(over_sev, k=10, x0=0.20) * 0.55
    # Secondary: intermodulation distortion (shares harmonic distortion mechanism)
    imd_sev = float(defect_scores.get("intermodulation_distortion", 0.0))
    p += _sigmoid_score(imd_sev, k=8, x0=0.25) * 0.25
    # Tertiary: clipping (analog overload often co-occurs with near-clipping levels)
    clip_sev = float(defect_scores.get("clipping", 0.0))
    p += _sigmoid_score(clip_sev, k=8, x0=0.20) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_lacquer_disc_degradation(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | lacquer_disc_degradation) — Acetat-Zersetzung (LACQUER_DISC). v10.0.0"""
    _ = sf.rms
    p = 0.0
    # Primary: direct DefectScanner score
    lacquer_sev = float(defect_scores.get("lacquer_disc_degradation", 0.0))
    p += _sigmoid_score(lacquer_sev, k=10, x0=0.15) * 0.60
    # Secondary: click density (substrate cracking)
    click_sev = float(defect_scores.get("clicks", 0.0))
    p += _sigmoid_score(click_sev, k=8, x0=0.30) * 0.20
    # Tertiary: bandwidth loss (HF oxidation)
    bw_sev = float(defect_scores.get("bandwidth_loss", 0.0))
    p += _sigmoid_score(bw_sev, k=8, x0=0.35) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_cassette_azimuth_tolerance(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | cassette_azimuth_tolerance) — Kassetten-Shell-HF-Kammfilterung. v10.0.0"""
    _ = sf.rms
    p = 0.0
    # Primary: azimuth error (shell tolerance manifests as azimuth-like HF loss)
    azimuth_sev = float(defect_scores.get("azimuth_error", 0.0))
    p += _sigmoid_score(azimuth_sev, k=10, x0=0.20) * 0.55
    # Secondary: bandwidth loss (azimuth→comb filter → HF cancellation)
    bw_sev = float(defect_scores.get("bandwidth_loss", 0.0))
    p += _sigmoid_score(bw_sev, k=8, x0=0.30) * 0.25
    # Tertiary: phase issues (comb filter creates inter-channel phase divergence)
    phase_sev = float(defect_scores.get("phase_issues", 0.0))
    p += _sigmoid_score(phase_sev, k=8, x0=0.25) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_wire_recording_specific(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(features | wire_recording_specific) — Drahtband: Knoten, Wicklungs-Wow. v10.0.0"""
    p = 0.0
    # Primary: click density (wire knots create impulsive events)
    click_sev = float(defect_scores.get("clicks", 0.0))
    p += _sigmoid_score(click_sev, k=10, x0=0.20) * 0.35
    # Secondary: wow/flutter (wire winding irregularities)
    wow_sev = float(defect_scores.get("wow", 0.0))
    flutter_sev = float(defect_scores.get("flutter", 0.0))
    p += _sigmoid_score(max(wow_sev, flutter_sev), k=8, x0=0.25) * 0.30
    # Tertiary: dropout density (magnetization loss at wire nodes)
    p += _sigmoid_score(sf.dropout_density, k=8, x0=0.20) * 0.20
    # Quaternary: high-frequency noise (non-linear magnetization of wire)
    p += _sigmoid_score(sf.hf_energy_ratio, k=6, x0=0.35) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_vocal_quality_degradation(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | vocal_quality_degradation) — Vokal-Qualitätsverlust durch
    kumulative NR/Kompressor-Eingriffe (VQI-Abfall). Korrektiv: phase_65. v10.0.0"""
    p = 0.0
    # Primär: Vokalhärte (NR-Überbearbeitung erzeugt Harshness in der Stimme)
    harsh_sev = float(defect_scores.get("vocal_harshness", 0.0))
    p += _sigmoid_score(harsh_sev, k=8, x0=0.15) * 0.45
    # Sekundär: NR-Pumpeffekt/Atemartefakt (NR-Nebeneffekt auf Stimm-Textur)
    nr_breath = float(defect_scores.get("nr_breathing_artifact", defect_scores.get("nr_breathing", 0.0)))
    p += _sigmoid_score(nr_breath, k=8, x0=0.20) * 0.25
    # Tertiär: Kompressionsartefakte (dynamische Kompression zerstört Vokal-Mikrodynamik)
    comp_sev = float(defect_scores.get("compression_artifacts", 0.0))
    p += _sigmoid_score(comp_sev, k=6, x0=0.20) * 0.15
    # Quartär: niedriger Crest-Factor (Kompression reduziert Vokal-Peaks)
    low_crest = float(np.clip(1.0 - sf.crest_factor_db / 10.0, 0.0, 1.0))
    p += low_crest * 0.10
    # Quinär: keine physischen Träger-Defekte (Verarbeitungsartefakt, keine Clicks)
    p += float(np.clip(1.0 - sf.click_density / 5.0, 0.0, 1.0)) * 0.05
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_vocal_stem_noise(sf: SpectralFeatures, defect_scores: dict[str, float]) -> float:
    """P(Merkmale | vocal_stem_noise) — Vokal-Stem und Begleitung haben unterschiedliche
    Rauschprofile; BSRoFormer-Stem-Trennung + stem-spezifische NR sinnvoll. v10.0.0"""
    p = 0.0
    # Primär: HF-Rauschen vorhanden (Stem-Trennung bringt Gewinn)
    hf_noise = float(defect_scores.get("high_freq_noise", 0.0))
    p += _sigmoid_score(hf_noise, k=8, x0=0.20) * 0.40
    # Sekundär: Tape-Hiss (häufigste Trägerform für vocal_stem_noise) — Alias: tape_hiss oder high_freq_noise
    tape_hiss_sev = float(defect_scores.get("tape_hiss", hf_noise))
    p += _sigmoid_score(tape_hiss_sev, k=7, x0=0.25) * 0.30
    # Tertiär: breites Rauschen (high_freq_noise als Proxy) + hoher Crest-Factor deutet auf klare Stimme hin
    p += _sigmoid_score(hf_noise, k=6, x0=0.15) * 0.20
    # Quartär: kein Knistern (das würde andere Phasen bevorzugen)
    crackle_sev = float(defect_scores.get("crackle", 0.0))
    p += float(np.clip(1.0 - crackle_sev * 2.0, 0.0, 1.0)) * 0.10
    return float(np.clip(p, 0.0, 1.0))


LIKELIHOOD_FNS = {
    # ── Original 12 ──────────────────────────────────────────────────────────
    "tape_dropout": _likelihood_tape_dropout,
    "tape_hiss": _likelihood_tape_hiss,
    "vinyl_crackle": _likelihood_vinyl_crackle,
    "vinyl_warp": _likelihood_vinyl_warp,
    "electrical_hum": _likelihood_electrical_hum,
    "head_misalignment": _likelihood_head_misalignment,
    "dc_offset": _likelihood_dc_offset,
    "digital_clip": _likelihood_digital_clip,
    "soft_saturation": _likelihood_soft_saturation,
    "head_wear": _likelihood_head_wear,
    "print_through": _likelihood_print_through,
    "transport_bump": _likelihood_transport_bump,
    # ── Erweiterte 22 (v10.0.0b) ───────────────────────────────────────────
    "bandwidth_loss": _likelihood_bandwidth_loss,
    "high_freq_noise": _likelihood_high_freq_noise,
    "stereo_imbalance": _likelihood_stereo_imbalance,
    "phase_issues": _likelihood_phase_issues,
    "pitch_drift": _likelihood_pitch_drift,
    "reverb_excess": _likelihood_reverb_excess,
    "digital_artifacts": _likelihood_digital_artifacts,
    "compression_artifacts": _likelihood_compression_artifacts,
    "quantization_noise": _likelihood_quantization_noise,
    "jitter_artifacts": _likelihood_jitter_artifacts,
    "dynamic_compression_excess": _likelihood_dynamic_compression_excess,
    "pre_echo": _likelihood_pre_echo,
    "low_freq_rumble": _likelihood_low_freq_rumble,
    "transient_smearing": _likelihood_transient_smearing,
    "vocal_harshness": _likelihood_vocal_harshness,
    "clipping": _likelihood_clipping,
    "riaa_curve_error": _likelihood_riaa_curve_error,
    "aliasing": _likelihood_aliasing,
    "bias_error": _likelihood_bias_error,
    "sibilance": _likelihood_sibilance,
    "wow": _likelihood_wow,
    "flutter": _likelihood_flutter,
    "wow_flutter": _likelihood_wow_flutter,
    "tape_start_instability": _likelihood_wow_flutter,  # same transport mechanism
    "tape_head_contact_instability": _likelihood_tape_head_contact,
    # ── v10.0.0: 12 neue Ursachen ──────────────────────────────────────────
    "modulation_noise": _likelihood_modulation_noise,
    "inner_groove_distortion": _likelihood_inner_groove_distortion,
    "groove_echo": _likelihood_groove_echo,
    "crosstalk": _likelihood_crosstalk,
    "intermodulation_distortion": _likelihood_intermodulation_distortion,
    "tape_splice_artifact": _likelihood_tape_splice,
    "hf_remanence_loss": _likelihood_hf_remanence_loss,
    "stylus_damage": _likelihood_stylus_damage,
    "sticky_shed_residue": _likelihood_sticky_shed,
    "multiband_wow_flutter": _likelihood_multiband_wow_flutter,
    "generation_loss": _likelihood_generation_loss,
    "motor_interference": _likelihood_motor_interference,
    # ── v10.0.0 ──────────────────────────────────────────────────────────────
    "amplitude_drift": _likelihood_amplitude_drift,
    # ── v10.0.0: DefectType→CAUSE-Lücken geschlossen ─────────────────────────
    "clicks": _likelihood_clicks,
    "dolby_nr_mismatch": _likelihood_dolby_nr_mismatch,
    "tape_head_level_dip": _likelihood_tape_head_level_dip,
    "scrape_flutter": _likelihood_scrape_flutter,
    "tape_head_clog": _likelihood_tape_head_clog,
    # ── v10.0.0: 9 neue Kausal-Ursachen ──────────────────────────────────────
    "proximity_effect_excess": _likelihood_proximity_effect_excess,
    "room_mode_resonance": _likelihood_room_mode_resonance,
    "nr_breathing_artifact": _likelihood_nr_breathing_artifact,
    "flutter_spectral_sidebands": _likelihood_flutter_spectral_sidebands,
    "speed_calibration_error": _likelihood_speed_calibration_error,
    "overload_distortion": _likelihood_overload_distortion,
    "lacquer_disc_degradation": _likelihood_lacquer_disc_degradation,
    "cassette_azimuth_tolerance": _likelihood_cassette_azimuth_tolerance,
    "wire_recording_specific": _likelihood_wire_recording_specific,
    # ── v10.0.0: vocal_quality_degradation ──────────────────────────────────
    "vocal_quality_degradation": _likelihood_vocal_quality_degradation,
    # ── v10.0.0: vocal_stem_noise ─────────────────────────────────────────────
    "vocal_stem_noise": _likelihood_vocal_stem_noise,
}


def _gaussian_score(x: float, mu: float, sigma: float) -> float:
    """Gaussianischer Ähnlichkeitsscore ∈ [0, 1]."""
    return float(math.exp(-0.5 * ((x - mu) / (sigma + 1e-9)) ** 2))


def _sigmoid_score(x: float, k: float = 5.0, x0: float = 0.5) -> float:
    """Sigmoidaler Score ∈ (0, 1). Höheres x → höhere Wahrscheinlichkeit."""
    return float(1.0 / (1.0 + math.exp(-k * (x - x0))))


def _normalize_defect_scores(defect_scores: dict[str, float]) -> dict[str, float]:
    """Normalisiert/aliasiert Defektbewertungs-Schlüssel und bereinigt Werte.

    Handles naming drift between legacy reasoner keys (e.g. click_severity)
    and current DefectScanner keys (e.g. clicks). Values are clamped to [0, 1]
    where appropriate; non-finite values are mapped to 0.0.
    """
    norm: dict[str, float] = {}
    for k, v in defect_scores.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            fv = 0.0
        # Keep values bounded for severity-like keys.
        if k.endswith(("severity", "_error")) or k in {
            "clicks",
            "dropouts",
            "clipping",
            "wow",
            "flutter",
            "wow_flutter",
            "sibilance",
            "pre_echo",
            "phase_issues",
            "stereo_imbalance",
        }:
            fv = float(np.clip(fv, 0.0, 1.0))
        norm[k] = fv

    # Bidirectionale Alias-Gruppen (Spec-/Code-Drift-Resilienz):
    # Scanner-Keys, Legacy-Keys und CAUSE-Namen werden auf denselben
    # Evidenzstand harmonisiert, damit Erkennung und Behebung konsistent bleiben.
    alias_groups = [
        ("clicks", "click_severity"),
        ("dropouts", "dropout_severity", "tape_dropout"),
        ("clipping", "clip_severity", "digital_clip"),
        ("wow", "wow_severity"),
        ("flutter", "flutter_severity"),
        ("tape_hiss", "high_freq_noise"),
        ("electrical_hum", "hum"),
        ("nr_breathing_artifact", "nr_breathing"),
        ("azimuth_error", "head_misalignment", "cassette_azimuth_tolerance"),
    ]
    for group in alias_groups:
        _vals = [float(norm[k]) for k in group if k in norm]
        if not _vals:
            continue
        _merged = float(np.clip(max(_vals), 0.0, 1.0))
        for k in group:
            norm[k] = _merged

    # Derive combined wow/flutter if only components are present.
    if "wow_flutter" not in norm and ("wow" in norm or "flutter" in norm):
        norm["wow_flutter"] = float(max(norm.get("wow", 0.0), norm.get("flutter", 0.0)))

    # Dropout proxy for silence ratio when not explicitly provided.
    if "silence_ratio" not in norm and "dropout_severity" in norm:
        norm["silence_ratio"] = float(np.clip(norm["dropout_severity"] * 0.8, 0.0, 1.0))

    return norm


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class CausalDefectReasoner:
    """
    Bayesianische Ursachendiagnose für Tonträger-Fehler.

    Verwendung::

        reasoner = CausalDefectReasoner()
        plan = reasoner.reason(
            defect_scores={"dropout_severity": 0.7, "noise_floor_db": -45.0},
            material="tape",
            audio=waveform,
            sample_rate=44100,
        )
        logger.debug(plan.primary_cause)
        logger.debug(plan.recommended_phases)
    """

    def __init__(self, detect_hum_hz: float | None = None):
        """
        Args:
            detect_hum_hz: Bekannte Netzfrequenz (50 oder 60 Hz). Wenn None,
                           wird automatisch erkannt.
        """
        self._known_hum_hz = detect_hum_hz

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def reason(
        self,
        defect_scores: dict[str, float],
        material: str = "unknown",
        audio: np.ndarray | None = None,
        sample_rate: int = 48000,
        sr: int | None = None,
        codec_contamination: dict[str, float] | None = None,
    ) -> RestorationPlan:
        """
        Berechnet den Restaurierungsplan für die gegebene Aufnahme.

        Args:
            defect_scores: Fehlermerkmale {"dropout_severity": 0.7, ...}
            material:      Trägermaterial ("tape"|"vinyl"|"shellac"|"digital"|"unknown")
            audio:         Optional – rohes Audio-Array für Signal-Merkmale
            sample_rate:   Abtastrate des Audio-Arrays
        Returns:
            RestorationPlan
        """
        if sr is not None:
            sample_rate = sr
        if audio is not None and len(audio) > 0:
            assert sample_rate == 48000, f"CausalDefectReasoner.reason() erwartet SR=48000, erhalten: {sample_rate}"
        material = material.lower().strip()
        if material not in MATERIAL_PRIORS:
            material = "unknown"

        # Signal-Merkmale extrahieren
        if audio is not None and len(audio) > 0:
            sf = extract_spectral_features(audio, sample_rate)
        else:
            sf = SpectralFeatures()

        defect_scores_norm = _normalize_defect_scores(defect_scores)
        return self._infer(defect_scores_norm, sf, material, codec_contamination=codec_contamination)

    # ------------------------------------------------------------------
    # Bayes-Inferenz
    # ------------------------------------------------------------------

    def _infer(
        self,
        defect_scores: dict[str, float],
        sf: SpectralFeatures,
        material: str,
        codec_contamination: dict[str, float] | None = None,
    ) -> RestorationPlan:
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        priors = MATERIAL_PRIORS.get(_mk, MATERIAL_PRIORS.get(MaterialType.UNKNOWN, {}))  # type: ignore[call-overload]
        posteriors: dict[str, float] = {}

        # §CODEC: Adjustiere Bayesian-Priors für analoge Ursachen wenn Codec-Contamination vorliegt.
        # Ohne diesen Guard: CausalReasoner leitet VINYL_WEAR aus MP3-Artefakten ab → falsche Phasen.
        _cc = codec_contamination or {}
        if _cc:
            _avg_discount = sum(_cc.values()) / max(len(_cc), 1)
            _digital_causes = {"digital_compression", "codec_artifact", "streaming_loss"}
            _analog_causes = {
                "vinyl_wear",
                "tape_degradation",
                "shellac_deterioration",
                "wow_flutter_damage",
                "surface_damage",
            }
            for cause, prior_val in list(priors.items()):
                if cause in _analog_causes and _avg_discount < 0.80:
                    priors[cause] = prior_val * _avg_discount
                elif cause in _digital_causes and _avg_discount < 0.80:
                    priors[cause] = prior_val * (1.0 + (1.0 - _avg_discount) * 0.5)

        for cause in CAUSES:
            prior = priors.get(cause, 1.0 / len(CAUSES))
            # Strong direct evidence for vocal harshness must not be drowned by global digital priors.
            if cause == "vocal_harshness":
                _vh_sev = float(defect_scores.get("vocal_harshness", 0.0))
                if _vh_sev > 0.0:
                    prior = max(prior, min(0.28, 0.03 + 0.25 * _vh_sev))
            likelihood = LIKELIHOOD_FNS[cause](sf, defect_scores)
            posteriors[cause] = prior * likelihood

        # Normierung
        total = sum(posteriors.values()) + 1e-12
        posteriors = {c: v / total for c, v in posteriors.items()}

        # Sortierung absteigend
        ranked = sorted(posteriors.items(), key=lambda kv: kv[1], reverse=True)
        primary_cause = ranked[0][0]
        # Kalibrierte Kausal-Konfidenz statt rohem Max-Posterior:
        # Berücksichtigt Trennschärfe (Top-2-Margin), Entropie und Defekt-Evidenz,
        # damit robuste Mehrfach-Evidenz nicht als "niedrige Konfidenz" kollabiert.
        _top1 = float(ranked[0][1])
        _top2 = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        _margin = max(0.0, _top1 - _top2)
        _probs = np.asarray([float(v) for _, v in ranked], dtype=np.float64)
        _probs = np.clip(_probs, 1e-12, 1.0)
        _norm = float(np.sum(_probs))
        if _norm > 0.0:
            _probs /= _norm
        _entropy = float(-np.sum(_probs * np.log(_probs)))
        _entropy_norm = _entropy / max(math.log(float(len(_probs))), 1e-9)
        _entropy_conf = float(np.clip(1.0 - _entropy_norm, 0.0, 1.0))

        _sev_vals = [float(v) for v in defect_scores.values() if isinstance(v, (int, float)) and np.isfinite(v)]
        if _sev_vals:
            _sev_vals_sorted = sorted(_sev_vals, reverse=True)
            _top3_mean = float(np.mean(_sev_vals_sorted[:3]))
            _evidence_conf = float(np.clip(0.55 * _sev_vals_sorted[0] + 0.45 * _top3_mean, 0.0, 1.0))
        else:
            _evidence_conf = 0.5

        _top_conf = float(np.clip(_top1 / 0.25, 0.0, 1.0))
        _margin_conf = float(np.clip(_margin / 0.18, 0.0, 1.0))
        confidence = float(
            np.clip(
                0.42 * _top_conf + 0.20 * _margin_conf + 0.18 * _entropy_conf + 0.20 * _evidence_conf,
                0.0,
                1.0,
            )
        )
        if _top1 < 0.08 and _evidence_conf < 0.35:
            confidence = min(confidence, 0.45)

        # Fusions-Plan: Phasen der Top-3 Ursachen zusammenführen
        seen_phases: set = set()
        ordered_phases: list[str] = []
        merged_params: dict[str, Any] = {}

        for cause, prob in ranked[:3]:
            if prob < 0.05:
                break
            for phase in CAUSE_TO_PHASES.get(cause, []):
                if phase not in seen_phases:
                    ordered_phases.append(phase)
                    seen_phases.add(phase)
            for param, val in CAUSE_PARAMS.get(cause, {}).items():
                if param not in merged_params:
                    merged_params[param] = val

        # §v10 SNR-adaptive param scaling: Noise-Reduction-Strength und
        # verwandte Parameter aus dem gemessenen SNR ableiten, nicht aus
        # statischen CAUSE_PARAMS. Cleaner Song → weniger NR nötig.
        if hasattr(self, "_last_snr_estimate") and self._last_snr_estimate > 0:
            _snr = self._last_snr_estimate
            _snr_scale = float(np.clip(25.0 / max(5.0, _snr), 0.5, 1.5))
            for _key in list(merged_params.keys()):
                if "strength" in _key or "boost" in _key:
                    merged_params[_key] = float(merged_params[_key] * _snr_scale)
            logger.debug("§v10 CausalDefectReasoner SNR-adaptive: snr=%.1fdB scale=%.2f", _snr, _snr_scale)

        # §6.2b/c Material-Phase-Exclusion-Filter: Era-spezifische Verbote durchsetzen.
        # Entfernt materialspezifisch verbotene Phasen aus dem Restaurierungsplan
        # (z.B. wax_cylinder: phase_07 VERBOTEN per spec §ERA 1900-1925).
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        _mat_exclusions = _MATERIAL_PHASE_EXCLUSIONS.get(_mk, frozenset())
        if _mat_exclusions:
            _excluded = [_p for _p in ordered_phases if _p in _mat_exclusions]
            if _excluded:
                ordered_phases = [_p for _p in ordered_phases if _p not in _mat_exclusions]
                logger.info(
                    "§6.2b Material-Verarbeitungsschritt-Exclusion: material=%s hat %d Verarbeitungsschritt(n) blockiert: %s",
                    material,
                    len(_excluded),
                    _excluded,
                )

        # §7.2a Severity-Weighted Phase-Reorder (spec Y1):
        # When ≥3 defects have severity ≥ 0.70, reorder phases so that phases
        # belonging to the highest-posterior causes are processed first.
        _high_sev_defects = [k for k, v in defect_scores.items() if v >= 0.70]
        if len(_high_sev_defects) >= 3:
            _phase_priority: dict[str, float] = {}
            for _cause_k, _cause_prob in ranked[:10]:
                for _phase_k in CAUSE_TO_PHASES.get(_cause_k, []):
                    _phase_priority[_phase_k] = max(_phase_priority.get(_phase_k, 0.0), _cause_prob)

            _original_order = {_phase_id: _idx for _idx, _phase_id in enumerate(ordered_phases)}

            ordered_phases = sorted(
                ordered_phases,
                key=lambda _p: (_phase_priority.get(_p, 0.0), -_original_order.get(_p, 999)),
                reverse=True,
            )
            logger.info(
                "§7.2a Severity-Reorder: %d high-severity defects (≥0.70) → reordered %d phases by cause posterior",
                len(_high_sev_defects),
                len(ordered_phases),
            )

        # §2.67 Koalitions-Priorisierung: Nach dem globalen Severity-Reorder
        # zusammengehörige Phasen sanft zusammenziehen.
        _coalition_ordered = self._apply_phase_coalition_priority(ordered_phases, ranked)
        if _coalition_ordered != ordered_phases:
            logger.info(
                "§2.67 Causal-Coalition-Priority: reordered %d phases for coalition continuity",
                len(_coalition_ordered),
            )
            ordered_phases = _coalition_ordered

        reasoning = self._build_reasoning(primary_cause, ranked, sf, material)

        return RestorationPlan(
            primary_cause=primary_cause,
            cause_probabilities=posteriors,
            ranked_causes=ranked,
            recommended_phases=ordered_phases,
            phase_parameters=merged_params,
            confidence=confidence,
            reasoning=reasoning,
            material=material,
        )

    @staticmethod
    def _apply_phase_coalition_priority(
        ordered_phases: list[str],
        ranked_causes: list[tuple[str, float]],
        phase_coalitions: dict[str, tuple[str, ...]] | None = None,
    ) -> list[str]:
        """Wendet eine milde §2.67-Koalitionspriorisierung auf die Phasenliste an."""
        if not ordered_phases:
            return ordered_phases

        phase_priority: dict[str, float] = {}
        for cause, prob in ranked_causes[:10]:
            for phase in CAUSE_TO_PHASES.get(cause, []):
                phase_priority[phase] = max(phase_priority.get(phase, 0.0), float(prob))

        coalitions = phase_coalitions if isinstance(phase_coalitions, dict) else _CAUSAL_PHASE_COALITIONS
        for members in coalitions.values():
            present = [phase_id for phase_id in members if phase_id in ordered_phases]
            if len(present) < 2:
                continue
            dominant = max(phase_priority.get(phase_id, 0.0) for phase_id in present)
            coalition_floor = max(0.0, float(dominant) * 0.92)
            for phase_id in present:
                phase_priority[phase_id] = max(float(phase_priority.get(phase_id, 0.0)), coalition_floor)

        original_order = {phase_id: idx for idx, phase_id in enumerate(ordered_phases)}
        return sorted(
            ordered_phases,
            key=lambda phase_id: (phase_priority.get(phase_id, 0.0), -original_order.get(phase_id, 999)),
            reverse=True,
        )

    def _build_reasoning(
        self,
        primary_cause: str,
        ranked: list[tuple[str, float]],
        sf: SpectralFeatures,
        material: str,
    ) -> str:
        lines = [
            f"Trägermaterial: {material}",
            f"Primäre Ursache: {primary_cause} (Posterior={ranked[0][1]:.3f})",
        ]
        if len(ranked) > 1 and ranked[1][1] > 0.10:
            lines.append(f"Zweitwahrscheinlichste: {ranked[1][0]} ({ranked[1][1]:.3f})")

        # §SOTA #8: Sekundäre Kausalkette — welche tieferen Ursachen stecken dahinter?
        _secondary = _SECONDARY_CAUSE_CHAIN.get(primary_cause)
        if _secondary:
            lines.append(f"Sekundär-Ursachen: {' → '.join(_secondary)}")

        lines.append(f"Klick-Dichte: {sf.click_density:.2f}/s")
        lines.append(f"Dropout-Dichte: {sf.dropout_density:.3f}")
        lines.append(f"Brumm-Score: {sf.hum_score:.3f}")
        lines.append(f"Clipping-Anteil: {sf.clip_fraction:.4f}")
        lines.append(f"Spektraler Rolloff: {sf.spectral_rolloff_hz:.0f} Hz")
        return " | ".join(lines)


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------

_reasoner: CausalDefectReasoner | None = None
_reasoner_lock = threading.Lock()


def get_reasoner() -> CausalDefectReasoner:
    """Globaler Singleton-Reasoner."""
    global _reasoner  # pylint: disable=global-statement
    if _reasoner is None:
        with _reasoner_lock:
            if _reasoner is None:
                _reasoner = CausalDefectReasoner()
    return _reasoner


def reason_about_defects(
    defect_scores: dict[str, float],
    material: str = "unknown",
    audio: np.ndarray | None = None,
    sample_rate: int = 44100,
    sr: int | None = None,
    codec_contamination: dict[str, float] | None = None,
) -> RestorationPlan:
    """Convenience-Funktion für direkten Aufruf."""
    return get_reasoner().reason(
        defect_scores, material, audio, sample_rate, sr=sr, codec_contamination=codec_contamination
    )


# Spec §3.2 / §2.4: kanonischer Fabrik-Name
get_causal_reasoner = get_reasoner
