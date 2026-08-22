"""Global calibration helpers for derived universal meta-parameters (§09.10).

All functions are pure and bounded. They only use existing pipeline inputs
and are safe to call from gates and target estimators.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_tcci(transfer_chain: list[str] | None) -> float:
    """Transfer-Chain-Complexity-Index in [0, 1]."""
    chain = [str(m).strip().lower() for m in (transfer_chain or []) if str(m).strip()]
    n = max(1, len(chain))
    lossy = sum(1 for m in chain if m in {"mp3_low", "aac", "streaming"})
    analog = sum(
        1 for m in chain if m in {"wax_cylinder", "shellac", "vinyl", "tape", "cassette", "wire_recording", "reel_tape"}
    )
    score = 0.18 * float(n - 1) + 0.22 * float(lossy) + 0.10 * float(max(0, analog - 1))
    return float(np.clip(score, 0.0, 1.0))


def compute_ibs(restorability: float, defect_severity_mean: float, tcci: float) -> float:
    """Intervention-Budget-Scalar in [0.15, 0.95]."""
    r = 1.0 - float(np.clip(restorability / 100.0, 0.0, 1.0))
    d = float(np.clip(defect_severity_mean, 0.0, 1.0))
    c = float(np.clip(tcci, 0.0, 1.0))
    budget = 0.55 * r + 0.30 * d + 0.15 * c
    return float(np.clip(budget, 0.15, 0.95))


def blend_targets_with_confidence(
    canonical: dict[str, float],
    song_targets: dict[str, float],
    medium_conf: float,
    era_conf: float,
    genre_conf: float,
) -> dict[str, float]:
    """Mischt per-song targets with canonical floors based on confidence."""
    conf = float(np.clip(0.45 * medium_conf + 0.30 * era_conf + 0.25 * genre_conf, 0.0, 1.0))
    blended: dict[str, float] = {}
    for goal, floor in canonical.items():
        t = float(song_targets.get(goal, floor))
        blended[goal] = float((1.0 - conf) * float(floor) + conf * t)
    return blended


def compute_cpb(material_ceiling: float, current_value: float, mode: str) -> float:
    """Ceiling-Proximity-Budget in [0, material_ceiling]."""
    mc = max(0.0, float(material_ceiling))
    cv = max(0.0, float(current_value))
    margin = max(0.0, mc - cv)
    safety = 0.70 if str(mode).strip().lower() == "restoration" else 0.50
    return float(np.clip(safety * margin, 0.0, mc))


def compute_retry_temperature(restorability: float, tcci: float, artifact_freedom_score: float) -> float:
    """Retry aggressiveness temperature in [0, 1]."""
    hard_song = 1.0 - float(np.clip(restorability / 100.0, 0.0, 1.0))
    chain = float(np.clip(tcci, 0.0, 1.0))
    artifact_risk = 1.0 - float(np.clip(artifact_freedom_score, 0.0, 1.0))
    t = 0.50 * hard_song + 0.30 * chain + 0.20 * artifact_risk
    return float(np.clip(t, 0.0, 1.0))


def compute_export_reliability(
    hpi: float,
    artifact_freedom: float,
    passed_goals: int,
    total_goals: int,
    reference_confidence: float,
) -> float:
    """Export reliability score in [0, 1]."""
    total = int(total_goals)
    ratio = 0.0 if total <= 0 else float(np.clip(float(passed_goals) / float(total), 0.0, 1.0))
    score = (
        0.35 * float(np.clip(hpi, 0.0, 1.0))
        + 0.30 * float(np.clip(artifact_freedom, 0.0, 1.0))
        + 0.20 * ratio
        + 0.15 * float(np.clip(reference_confidence, 0.0, 1.0))
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_goal_coverage_index(musical_goals_passed: dict[str, bool] | None) -> float:
    """Priority-weighted musical-goal coverage in [0, 1]."""
    passed = dict(musical_goals_passed or {})
    if not passed:
        return 0.0

    weights = {
        # P1
        "natuerlichkeit": 1.4,
        "authentizitaet": 1.4,
        # P2
        "tonal_center": 1.2,
        "tonalcenter": 1.2,
        "timbre_authentizitaet": 1.2,
        "artikulation": 1.2,
        # P3
        "emotionalitaet": 1.0,
        "mikrodynamik": 1.0,
        "micro_dynamics": 1.0,
        "groove": 1.0,
        # P4
        "transparenz": 0.8,
        "waerme": 0.8,
        "basskraft": 0.8,
        "bass_kraft": 0.8,
        "separation_fidelity": 0.8,
        # P5
        "brillanz": 0.6,
        "raumtiefe": 0.6,
        "spatial_depth": 0.6,
    }

    score = 0.0
    total = 0.0
    for g, ok in passed.items():
        k = str(g).strip().lower()
        w = float(weights.get(k, 1.0))
        total += w
        if bool(ok):
            score += w

    if total <= 0.0:
        return 0.0
    return float(np.clip(score / total, 0.0, 1.0))


def compute_reference_confidence(target_confidence: float, tcci: float, carrier_chain_recovery_ratio: float) -> float:
    """Kalibrierte Referenz-Konfidenz in [0, 1] aus vorhandenen Zuverlässigkeitssignalen."""
    tc = float(np.clip(target_confidence, 0.0, 1.0))
    chain_stability = 1.0 - float(np.clip(tcci, 0.0, 1.0))
    # High carrier-recovery-ratio means stronger intentional divergence from degraded input.
    # This lowers confidence in strict input-referenced proxies.
    carrier_stability = 1.0 - float(np.clip(carrier_chain_recovery_ratio / 0.35, 0.0, 1.0))
    conf = 0.65 * tc + 0.20 * chain_stability + 0.15 * carrier_stability
    return float(np.clip(conf, 0.0, 1.0))


def compute_recovery_pressure_index(
    fallback_attempts: int,
    rollback_count: int,
    goal_deficit_ratio: float,
) -> float:
    """Recovery pressure in [0, 1] based on recovery attempts, rollbacks and missing goals."""
    fa = float(np.clip(float(fallback_attempts) / 3.0, 0.0, 1.0))
    rb = float(np.clip(float(rollback_count) / 8.0, 0.0, 1.0))
    gd = float(np.clip(goal_deficit_ratio, 0.0, 1.0))
    rpi = 0.40 * fa + 0.35 * rb + 0.25 * gd
    return float(np.clip(rpi, 0.0, 1.0))


# ---------------------------------------------------------------------------
# §09.1 Kanonische Schwellwerte (Single Source of Truth)
# ---------------------------------------------------------------------------

CANONICAL_THRESHOLDS_RESTORATION: dict[str, float] = {
    "natuerlichkeit": 0.90,
    "authentizitaet": 0.88,
    "tonal_center": 0.95,
    "tonalcenter": 0.95,
    "timbre_authentizitaet": 0.87,
    "artikulation": 0.88,  # §09.1 Spec P2 (v10.0.0: 0.85→0.88)
    "emotionalitaet": 0.84,  # §09.1 Spec P3 (v10.0.0: 0.82→0.84)
    "mikrodynamik": 0.88,
    "micro_dynamics": 0.88,
    "groove": 0.83,
    "transparenz": 0.82,
    "waerme": 0.77,  # §V25 Wärmeband-Guard v10.0.0 (200–800 Hz, ≤2.5 dB kumulativer Verlust)
    "bass_kraft": 0.78,
    "basskraft": 0.78,
    "separation_fidelity": 0.80,  # §09.1 Spec P4 (v10.0.0: 0.78→0.80)
    "brillanz": 0.78,
    "raumtiefe": 0.70,
    "spatial_depth": 0.70,
    "transient_energie": 0.80,  # §§1.4.6 v10.0.0: Transient-Energie-Ziel (P3)
}

CANONICAL_THRESHOLDS_STUDIO2026: dict[str, float] = {
    "natuerlichkeit": 0.92,
    "authentizitaet": 0.90,
    "tonal_center": 0.96,
    "tonalcenter": 0.96,
    "timbre_authentizitaet": 0.89,
    "artikulation": 0.90,  # §09.1 Spec P2 (v10.0.0: 0.87→0.90)
    "emotionalitaet": 0.87,  # §09.1 Spec P3 (v10.0.0: 0.84→0.87)
    "mikrodynamik": 0.90,
    "micro_dynamics": 0.90,
    "groove": 0.85,
    "transparenz": 0.85,
    "waerme": 0.78,
    "bass_kraft": 0.80,
    "basskraft": 0.80,
    "separation_fidelity": 0.83,  # §09.1 Spec P4 (v10.0.0: 0.80→0.83)
    "brillanz": 0.82,
    "raumtiefe": 0.74,
    "spatial_depth": 0.74,
    "transient_energie": 0.83,  # §§1.4.6 v10.0.0: Transient-Energie-Ziel (P3, Studio 2026)
}

# ---------------------------------------------------------------------------
# §09.2 Per-Song Goal-Targets — Era × Material × Genre Bias-Tabellen
# ---------------------------------------------------------------------------

_ERA_BIAS: dict[str, dict[str, float]] = {
    # Pre-1926: Rein akustische Trichteraufnahme — kein Mikrofon, kein Verstärker.
    # BW typisch 150–3000 Hz; Bass < 200 Hz physikalisch abgeschnitten.
    # Horn-Diffraktion verfärbt Einsätze; keine Raumkontrolle möglich.
    "acoustic_era": {
        "brillanz": -0.38,  # BW-Hartgrenze ~3 kHz durch Trichtermechanik
        "transparenz": -0.24,  # Kein Elektronikrauschen, aber physik. Oberflächenrauschen dominant
        "raumtiefe": -0.18,  # Mono-Pflicht; Aufnahmeraum auf Trichterwirkung optimiert
        "waerme": +0.12,  # Mittenbetonter Charakter klingt warm durch Tiefpasswirkung
        "authentizitaet": +0.12,  # Historische Authentizität ist inhärent; Erwartung niedriger
        "natuerlichkeit": +0.06,  # Akustisch, aber stark durch Trichterfarbe geprägt
        "bass_kraft": -0.30,  # Horn schneidet Bass unter ~200 Hz physikalisch ab
        "groove": -0.18,  # Kein Click-Track; physische Trägheit der Trichter-Sessions
        "micro_dynamics": -0.14,  # Trichter: DR < 30 dB, schwere Dynamikkompression
        "emotionalitaet": -0.08,  # Stark limitiertes Frequenz- und Dynamikfenster
        "artikulation": -0.10,  # Horn-Diffraktion + Schneidstichel: Onset-Verschmierung
        "separation_fidelity": -0.20,  # Mono; keine Quellentrennung möglich
    },
    # 1926–1949: Frühelektrische Aufnahme — Kondensator-/Bändchenmikrofon, Röhrenverstärker.
    # BW typisch 50–7000 Hz; AGC-Schaltungen; Schellack-Presswerk; kein Magnetband.
    "early_electric": {
        "brillanz": -0.20,  # BW ~7 kHz besser als akustisch, aber noch stark begrenzt
        "transparenz": -0.12,  # Röhrenrauschen + Schellack-Oberflächenrauschen
        "raumtiefe": -0.10,  # Überwiegend Mono; Stereo-Experimente 1930s kaum verbreitet
        "waerme": +0.14,  # Röhrenverstärkung: ausgeprägte 2.-Harmonische-Wärme
        "authentizitaet": +0.10,  # Historische Authentizität; Epoche-Erwartung angepasst
        "natuerlichkeit": +0.08,  # Elektrisches Mikrofon gibt natürlicheres Timbre als Trichter
        "groove": -0.08,  # Schellack-Schnitt instabil; kein Magnetband; Rest-Gleichlaufschwankung
        "micro_dynamics": -0.08,  # Frühe AGC/Kompressoren: reduzierter Dynamikbereich
        "emotionalitaet": -0.04,  # Begrenzt, aber signifikant besser als akustische Ära
        "artikulation": -0.06,  # Frühe-Scheibenaufnahme: Onset-Unschärfe reduziert
    },
    "1950s": {
        "brillanz": -0.14,
        "transparenz": -0.08,
        "waerme": +0.10,
        "authentizitaet": +0.08,
        # Early electronic recording technique: residual speed instability + compressed DR.
        "groove": -0.04,  # Early tape/disc: residual wow/flutter beyond medium limits
        "micro_dynamics": -0.04,  # Early electronic compressors: reduced dynamic headroom
    },
    "1960s": {
        # Early stereo (1960–1969): 4→8-track, British Invasion, Motown, bossa nova.
        # Warm analog tape, organic live performances — but stereo imaging still narrow.
        "waerme": +0.06,  # Analog tape warmth at its cleanest (7.5/15 ips)
        "authentizitaet": +0.04,  # Live studio sessions, minimal post-processing
        "natuerlichkeit": +0.04,  # Natural room acoustics, pre-digital aesthetic
        "spatial_depth": -0.04,  # Early narrow stereo: hard-panned L/C/R only, no automation
    },
    "1970s": {
        # High-DR analog era (1970–1979): 16–24-track, funk/soul, prog rock, jazz-fusion.
        # Natural dynamics peak, groove-oriented performance culture.
        "brillanz": +0.04,
        "transparenz": +0.04,
        "waerme": +0.02,
        "groove": +0.06,  # Funk/soul: studio musicians, extremely tight rhythmic playing
        "authentizitaet": +0.04,  # Organic, pre-digital recording culture
        "natuerlichkeit": +0.04,  # Minimal electronic processing, natural acoustics
    },
    "1980s": {
        # Digital effects era (1980–1989): Dolby NR, gated reverb, synth pop, hair metal.
        # Bright mix aesthetic, heavy processing — technically clean but sonically artificial.
        "brillanz": +0.08,  # Bright, HF-pushed mix aesthetic of the decade
        "transparenz": +0.06,  # Clean tape/early digital — low noise floor
        "waerme": -0.06,  # Cold digital/electronic aesthetic dominates
        "natuerlichkeit": -0.06,  # Heavy processing: gated snare, digital reverb, synths
        "spatial_depth": +0.06,  # Large reverbs → pronounced spatial impression
        "groove": -0.04,  # Drum machines + gated reverb reduce organic groove feel
    },
    "1990s": {
        # CD peak era (1990–1999): grunge, britpop, R&B, excellent dynamic range.
        # Digital mixing standard — transparent, articulate, but slightly sterile warmth.
        "brillanz": +0.10,
        "transparenz": +0.10,
        "artikulation": +0.06,
        "waerme": -0.04,
        "groove": +0.04,  # CD masters: faithful rhythmic reproduction, tight grid
        "natuerlichkeit": +0.04,  # Relatively unprocessed vs. 1980s; grunge/live aesthetic
    },
    "2000s": {
        # Loudness War peak (2000–2009): heavy limiting, DR 5–8 dB, MP3 distribution.
        # Over-compressed masters — loudness at the cost of dynamics and naturalness.
        "brillanz": +0.06,  # HF pushed in mastering for perceived loudness
        "transparenz": -0.08,  # Heavy limiting destroys micro-detail transparency
        "micro_dynamics": -0.14,  # Loudness War: DR often 5–8 dB (peak recording technique impact)
        "natuerlichkeit": -0.08,  # Over-compressed, fatiguing, harsh
        "emotionalitaet": -0.06,  # Compression reduces emotional arc and dynamic contrast
        "groove": +0.04,  # Hip-hop/electronic: tight rhythmic grid despite limiting
        "waerme": -0.06,  # Clinical, compressed, over-bright
    },
    "2010s": {
        # Streaming era (2010–2026): −14 LUFS normalization, DR recovery, spatial audio.
        # Partial reversal of Loudness War — more balanced dynamics, high-res digital.
        "brillanz": +0.08,  # Modern bright productions, but more balanced than 2000s
        "transparenz": +0.10,  # High-resolution digital: excellent spectral clarity
        "micro_dynamics": +0.08,  # DR recovery vs. 2000s; −14 LUFS streaming normalization
        "natuerlichkeit": +0.06,  # Less extreme Loudness War compression than 2000s
        "artikulation": +0.10,  # Modern digital: excellent transient precision (24-bit/96 kHz)
        "waerme": -0.02,  # Slightly clinical but less extreme than 1980s
        "spatial_depth": +0.04,  # Spatial audio (Dolby Atmos, binaural) growing standard
    },
}

# ---------------------------------------------------------------------------
# §R4 [RELEASE_MUST] Per-Material-Goal-Overrides — überschreiben bias-basierten Floor-Wert.
# Für physikalisch begrenzte Träger, die kein realistisches Ziel über dem Bias-Wert erreichen.
# Wird in get_material_floor() vor der Bias-Berechnung geprüft — normativ bindend.
# ---------------------------------------------------------------------------
_MATERIAL_GOAL_FLOOR_OVERRIDES: dict[str, dict[str, float]] = {
    # §R4: spatial_depth für Lossy-Material (MP3/AAC Joint-Stereo-Coding reduziert IACC deutlich).
    # MP3 128 kbps Stereo: IACC typisch 0.35–0.50 (psychoakustisches Stereo-Masking → räuml. Info verloren).
    # Ohne Override wäre Floor 0.70 (Canonical) — physikalisch unerreichbar für 128 kbps MP3.
    # IEC 11172-3 §8: MP3 Joint-Stereo-Coding erlaubt Phasencancellung in Mid/Side-Kanälen.
    "mp3_low": {"spatial_depth": 0.38},
    "mp3_high": {"spatial_depth": 0.44},
    "aac": {"spatial_depth": 0.44},
    "streaming": {"spatial_depth": 0.44},
    "minidisc": {"spatial_depth": 0.36},  # ATRAC Stereo-Masking aggressiver als MP3
    # §09.19 Kassette — IEC 60094-1 Type I physikalische Grenzen (v10.0.0):
    # brillanz:          12-kHz-BW-Ceiling (IEC Type I) begrenzt HF-Restaurierung auf ~0.68;
    #                    tape_analog-Bias -0.22 ergibt ~0.721 — zu optimistisch nach phase_06.
    # spatial_depth:     Kassetten-Stereo: L/R-Übersprechen ~40 dB (IEC 60094-1 §5.4);
    #                    physikalisches Ceiling ~0.55 (tape_analog-Bias -0.04 ergibt ~0.689).
    # separation_fidelity: Kassetten-Bleed: SDR < 15 dB; Ceiling ~0.60
    #                    (tape_analog-Bias -0.22 ergibt ~0.741 — deutlich zu hoch).
    "cassette": {
        "brillanz": 0.68,
        "spatial_depth": 0.55,
        "separation_fidelity": 0.60,
    },
    "kassette": {
        "brillanz": 0.68,
        "spatial_depth": 0.55,
        "separation_fidelity": 0.60,
    },
}

_MATERIAL_BIAS: dict[str, dict[str, float]] = {
    # Ultra-analog (Shellac, Wax, Wire, Lacquer)
    # P1/P2: Authentizität und Timbre stark richtungsgebend (Trägersignatur), aber BW stark limitiert.
    # P3/P4: Physikalische Grenzen des Trägers — keine SGT-Kalibrierung ohne diese Biases führt
    # dazu, dass groove/bass_kraft/sep_fidelity auf kanonischen Werten (0.83/0.78/0.78) verbleiben,
    # die für Shellac/Wax physikalisch unerreichbar sind (§0a Material-Ceiling Ebene 1).
    "ultra_analog": {
        "brillanz": -0.24,
        "transparenz": -0.12,
        "waerme": +0.10,
        "authentizitaet": +0.10,
        # §09.2 Material-adaptive floor: Shellac SNR~15dB, BW~7kHz → natuerlichkeit physikalisch ≤0.72
        # Formula: floor = canonical(0.90) + kappa_min(0.27) * bias → bias = (0.72-0.90)/0.27 = -0.667
        "natuerlichkeit": -0.67,
        # P3 — dynamisch/rhythmisch limitiert durch Wow/Flutter, surface noise floor, limited DR
        "groove": -0.18,  # Wow/Flutter, Crackle-Onsets maskieren rhythmische Präzision
        "micro_dynamics": -0.14,  # Rauschboden und limitierte DR reduzieren Dynamik-Headroom
        "emotionalitaet": -0.08,  # Begrenzte tonale + dynamische Bandbreite reduziert Arousal-Range
        # P4 — spektral/strukturell limitiert
        "bass_kraft": -0.22,  # LF-Rolloff: Shellac ≤ 100 Hz praktisch, Wax ≤ 80 Hz (§0 §6.2c)
        "separation_fidelity": -0.24,  # Mono oder Mono-kompatibles Narrow-Stereo → keine Stem-Sep möglich
        # §Bug#fix v10.0.0: get_material_floor() nutzt "spatial_depth" (Metrik-Dict-Key), nicht
        # "raumtiefe" (Legacy-Key). Legacy-Key bleibt für SGT-Alias-Propagation erhalten.
        "raumtiefe": -0.15,  # Legacy-Key: SGT-Alias-Propagation (estimate_song_goal_targets)
        "spatial_depth": -0.15,  # Canonical Key: get_material_floor("shellac", "spatial_depth")
        # §09.2 Weitere physikalische Grenzen Shellac/Wax:
        # artikulation: Crackle-Onsets + limited DR maskieren Transienten stark
        # Formula: physical ceiling ~0.60 → bias = (0.60-0.85)/0.27 = -0.926 ≈ -0.93
        "artikulation": -0.93,
        # tonal_center: Wow/Flutter (1-2% WRMS) + keine RIAA-Standardisierung bis 1954
        # Formula: physical ceiling ~0.72 → bias = (0.72-0.95)/0.27 = -0.852 ≈ -0.85
        "tonal_center": -0.85,
        # timbre_authentizitaet: Shellac-Resonanzen/Sättigung IST der authentische Klang → positive Richtung
        # Formula: ceiling ~0.884 → bias = (0.884-0.87)/0.27 = +0.05
        "timbre_authentizitaet": +0.05,
    },
    # Normal-analog Vinyl (LP, Vinyl)
    # Vinyl: Schneidebeschränkungen <80 Hz (bass_kraft), leichtes Wow/Flutter (groove)
    "analog": {
        "waerme": +0.10,
        "brillanz": -0.06,
        "authentizitaet": +0.08,
        # §09.2 Material-adaptive floor: Vinyl SNR~55dB, BW~16kHz → natuerlichkeit physikalisch ≤0.82
        # Formula: floor = canonical(0.90) + kappa_min(0.27) * bias → bias = (0.82-0.90)/0.27 = -0.296
        "natuerlichkeit": -0.30,
        # P3/P4: milde Biases — physikalische Einschränkungen des Carriers ohne Extremwerte
        "groove": -0.04,  # Leichtes Wow/Flutter
        "bass_kraft": -0.06,  # Vinyl-Schneidebeschränkung LF
        "separation_fidelity": -0.06,  # Narrow-Stereo, Early-Stereo-Artefakte
        # §09.2 Ergänzende Vinyl-Biases (v10.0.0):
        # transparenz: Oberflächenrauschen reduziert Spektralklarheit; ceiling ~0.793
        # Formula: (0.793-0.82)/0.27 = -0.10
        "transparenz": -0.10,
        # artikulation: Wow/Flutter (~0.10-0.15% WRMS) beeinträchtigt Transienten; ceiling ~0.828
        # Formula: (0.828-0.85)/0.27 = -0.08
        "artikulation": -0.08,
        # micro_dynamics: Oberflächenrauschen maskiert feine Dynamik; ceiling ~0.858
        # Formula: (0.858-0.88)/0.27 = -0.08
        "micro_dynamics": -0.08,
    },
    # Tape (Reel-Tape, Kassette) — physikalisch stärker limitiert als Vinyl
    # §09.2 (v10.0.0): Separatklasse tape_analog, da Tape-spezifische Einschränkungen
    # sich grundlegend von Vinyl unterscheiden:
    # - Tape-Hiss-NR reduziert HF stärker als Vinyl-Rausch-NR → brillanz Ceiling tiefer
    # - Wow/Flutter bei Tape stärker → groove/artikulation eingeschränkter
    # - Tape-Sättigung (H2/H4) ist authentic, aber ISO-226-gewichtetes Wärme-Verhältnis
    #   bleibt niedrig (Waerme-Bias 0.0, nicht +0.10 wie bei Vinyl) — Metrik-Divisor
    #   übernimmt die Anpassung (§9.12.8 WaermeMetric material_type)
    # - Narrow/Mono-Stereo → SepFidelity stark eingeschränkt
    "tape_analog": {
        "waerme": +0.00,  # Kein Bias: Metrik-Divisor (§9.12.8) übernimmt Anpassung
        "brillanz": -0.22,  # Tape-Hiss-NR reduziert HF: floor ~0.72 bei kappa_min
        "authentizitaet": +0.06,  # Tape-Sättigung ist authentisch
        "natuerlichkeit": -0.30,  # Gleich wie Vinyl (SNR~40-50 dB nach NR)
        "groove": -0.08,  # Stärkeres Wow/Flutter als Vinyl
        "bass_kraft": -0.06,  # Tape-Bias-Sättigung LF
        "separation_fidelity": -0.22,  # Tape Narrow/Mono: SDR<3dB triggert ref-free path
        "artikulation": -0.15,  # Dropout+Hiss maskiert Transienten
        "timbre_authentizitaet": -0.04,
        "spatial_depth": -0.04,
        # §09.2 Ergänzende Tape-Biases (v10.0.0):
        # micro_dynamics: Tape-Hiss maskiert feine Dynamik auch nach NR; ceiling ~0.839
        # Formula: (0.839-0.88)/0.27 = -0.15
        "micro_dynamics": -0.15,
        # tonal_center: Wow/Flutter (~0.3% WRMS) beeinträchtigt Tonstabilität; ceiling ~0.869
        # Formula: (0.869-0.95)/0.27 = -0.30
        "tonal_center": -0.30,
        # §09.2 Ergänzende Tape-Biases (v10.0.0 Kalibrierungslücke §GOAL_BASELINE_CHECK):
        # transparenz: Kassetten-/Bandhiss reduziert Spektralklarheit stärker als Vinyl (-0.10).
        # Physikalisches Ceiling Kassette ~0.776 (Type-I) → bias = (0.776-0.82)/0.27 = -0.163 ≈ -0.15
        "transparenz": -0.15,
        # transient_energie: Kassetten-Hiss + Dropouts maskieren Onset-Energie; Ziel blieb
        # bisher auf kanonisch 0.80 (CD-Niveau) — zu hoch für Tape/Kassette → unnötige
        # Recovery-Phasen in §GOAL_BASELINE_CHECK.
        # Physikalisches Ceiling Kassette ~0.746 → bias = (0.746-0.80)/0.27 = -0.200
        "transient_energie": -0.20,
        # §09.2 Realmesspflichtige Ergänzung (v10.0.0 Echtmessung 2026-05-20):
        # emotionalitaet: Kassetten-AGC-Kompressionsschaltkreis reduziert dynamische Modulation;
        # 12-kHz-BW-Ceiling (IEC 60094-1 Type I) begrenzt tonale Bandbreite für Arousal-Messungen.
        # Echtmessung Original-Kassette (Elke Best, 1970er, Schlager, measure_all() mit panns=0.7):
        #   Original = 0.782. Canon = 0.840. Kappa_min = 0.27.
        #   Physikalisches Ceiling ~0.781 → bias = (0.781-0.840)/0.27 = -0.219 ≈ -0.22.
        # Ohne diesen Bias löst §GOAL_BASELINE_CHECK fälschlicherweise Recovery-Phasen aus,
        # da das Original das Floor physikalisch nicht erreichen kann.
        "emotionalitaet": -0.22,
    },
    # Digital (CD, DAT, Streaming) — near-lossless; natuerlichkeit at full canonical floor 0.90
    "digital": {
        "transparenz": +0.08,
        "artikulation": +0.06,
        "brillanz": +0.06,
    },
    # Lossy (mp3_low, mp3_high, aac, minidisc) — codec artifacts reduce naturalness
    # §09.2 Material-adaptive floor: mp3_low 128kbps → natuerlichkeit physikalisch ≤0.78
    # Formula: floor = canonical(0.90) + kappa_min(0.27) * bias → bias = (0.78-0.90)/0.27 = -0.444
    "lossy": {
        "natuerlichkeit": -0.44,
        # §Bug#fix v10.0.0: war +0.04 (falsche Richtung!). mp3 Pre-Echo + HF-Rolloff (~16 kHz
        # bei 128 kbps) begrenzen Transparenz; physikalisches Ceiling ~0.793.
        # Formula: (0.793 - 0.82) / 0.27 = -0.10. CD/DAT ("digital") hat +0.08 ✓.
        "transparenz": -0.10,
        "artikulation": +0.04,
        "brillanz": +0.04,
        # §09.2 Lossy-SDR-Ceiling: Codec-Quantisierungsrauschen begrenzt Vocal-Stem-SDR auf ~10 dB
        # (_MATERIAL_MAX_SDR["mp3_low"] = 10.0; musical_goals.instructions.md sep_fidelity).
        # Im reference-free Modus → Harmonicity-Score floor ~0.70–0.75 (codec reduziert HF-Harmonicity).
        # Formula: target_floor ≈ 0.75 → bias = (0.75 - 0.78) / 0.27 = -0.111 ≈ -0.10.
        # Vergleich: vinyl analog = -0.06 (floor 0.764); tape_analog = -0.22 (floor 0.721).
        # Lossy ist schlechter als Vinyl (Codec-Quantisierung) aber besser als Tape (Stereo erhalten).
        "separation_fidelity": -0.10,
        # §09.2 Ergänzende Lossy-Biases (v10.0.0):
        # bass_kraft: Codec-Quantisierung beeinträchtigt Bass-Klarheit; ceiling ~0.758
        # Formula: (0.758 - 0.78) / 0.27 = -0.08
        "bass_kraft": -0.08,
        # timbre_authentizitaet: Pre-Echo-Verschmierung verändert Timbre; ceiling ~0.843
        # Formula: (0.843 - 0.87) / 0.27 = -0.10
        "timbre_authentizitaet": -0.10,
        # authentizitaet: Codec-Phasen-/Spektralartefakte; ceiling ~0.853
        # Formula: (0.853 - 0.88) / 0.27 = -0.10
        "authentizitaet": -0.10,
        # micro_dynamics: Psychoakustisches Masking komprimiert Mikrodynamik; ceiling ~0.858
        # Formula: (0.858 - 0.88) / 0.27 = -0.08
        "micro_dynamics": -0.08,
    },
}

_MATERIAL_CLASS: dict[str, str] = {
    "wax_cylinder": "ultra_analog",
    "shellac": "ultra_analog",
    "wire_recording": "ultra_analog",
    "lacquer_disc": "ultra_analog",
    "vinyl": "analog",
    "lp": "analog",
    "tape": "tape_analog",  # §09.2 (v10.0.0): eigene Klasse, nicht mit Vinyl zusammen
    "reel_tape": "tape_analog",  # §09.2 (v10.0.0): tape_analog mit korrekten HF/Sep-Böden
    "cassette": "tape_analog",  # §09.2 (v10.0.0): Kassette physikalisch näher an Tape als Vinyl
    "kassette": "tape_analog",  # Alias
    "cd_digital": "digital",
    "cd": "digital",
    "dat": "digital",
    "mp3_low": "lossy",
    "mp3_high": "lossy",
    "aac": "lossy",
    "minidisc": "lossy",
    "streaming": "lossy",
}

_GENRE_BIAS: dict[str, dict[str, float]] = {
    "klassik": {
        "raumtiefe": +0.18,
        "natuerlichkeit": +0.12,
        "mikrodynamik": +0.10,
        "brillanz": -0.08,
    },
    "jazz": {
        "waerme": +0.12,
        "natuerlichkeit": +0.10,
        "authentizitaet": +0.10,
        "transparenz": -0.04,
    },
    "schlager": {
        # bass_kraft −0.32: Schlager hat physikalisch geringen Bassanteil (heller Vokal-Mix,
        # Vinyl-Schneidebeschränkungen <80 Hz). bass_ratio typisch 0.002–0.005 statt 0.05
        # → Score max. ~0.20 ohne künstliche Bass-Anhebung (§0-Verletzung).
        # Threshold 0.78 → 0.46 ist genre-realistisch und §0-konform.
        "waerme": +0.10,
        "groove": +0.06,
        "authentizitaet": +0.08,
        "bass_kraft": -0.32,
    },
    "pop": {
        # bass_kraft −0.12: Heller Pop-Mix (1970s–1990s) hat weniger Bassanteil als
        # Rock/Electronic. Threshold leicht senken ohne §0-Verletzung.
        "transparenz": +0.08,
        "artikulation": +0.08,
        "brillanz": +0.08,
        "bass_kraft": -0.12,
    },
    "rock": {
        "bass_kraft": +0.08,
        "mikrodynamik": +0.06,
        "groove": +0.08,
    },
    "electronic": {
        "transparenz": +0.10,
        "bass_kraft": +0.10,
        "brillanz": +0.08,
    },
    "folk": {
        "natuerlichkeit": +0.12,
        "authentizitaet": +0.10,
        "waerme": +0.08,
    },
    # Soul / R&B (Motown, Stax, Atlantic, Northern Soul)
    # Tight rhythm section, emotional vocal performance, warm analog room acoustics.
    # Groove and authenticity are definitional — sterilization is worse than noise.
    "soul": {
        "groove": +0.10,  # Rhythm section precision: Motown/Stax studio musicians
        "authentizitaet": +0.12,  # Raw emotional performance is the product — preserve at all cost
        "waerme": +0.08,  # Warm room acoustics, natural reverb, analog warmth
        "micro_dynamics": +0.06,  # Vocal swells, breath control, expressive micro-dynamic
        "bass_kraft": +0.06,  # Prominent bass lines (Jamerson, Duck Dunn style)
    },
    # R&B aliases — GenreClassifier may output any of these
    "rnb": {"groove": +0.10, "authentizitaet": +0.12, "waerme": +0.08, "micro_dynamics": +0.06, "bass_kraft": +0.06},
    "r&b": {"groove": +0.10, "authentizitaet": +0.12, "waerme": +0.08, "micro_dynamics": +0.06, "bass_kraft": +0.06},
    # Blues (Delta, Chicago, Electric) — rawness IS the art form
    "blues": {
        "waerme": +0.12,  # Warm, dark, organic character
        "authentizitaet": +0.12,  # Raw imperfections preserve blues authenticity — never sterilize
        "groove": +0.08,  # Shuffle/swing feel defines genre
        "brillanz": -0.06,  # Dark, warm recordings — not bright/shrill
        "transparenz": -0.06,  # Deliberate analog warmth preferred over clinical transparency
    },
    # Hip-Hop / Rap — beat precision + sub-bass is the product
    "hip_hop": {
        "groove": +0.12,  # Beat is the dominant feature — rhythmic precision paramount
        "bass_kraft": +0.14,  # Sub-bass defines genre identity (808s, sampled kicks)
        "artikulation": +0.10,  # Lyrical intelligibility is primary vocal quality metric
        "spatial_depth": +0.06,  # Stereo field deliberately used for immersive effect
        "micro_dynamics": -0.08,  # Often intentionally side-chain compressed; respect the aesthetic
        "natuerlichkeit": -0.06,  # Electronic/heavily processed by design — not a defect
    },
    # Hip-Hop aliases
    "hip-hop": {
        "groove": +0.12,
        "bass_kraft": +0.14,
        "artikulation": +0.10,
        "spatial_depth": +0.06,
        "micro_dynamics": -0.08,
        "natuerlichkeit": -0.06,
    },
    "rap": {
        "groove": +0.12,
        "bass_kraft": +0.14,
        "artikulation": +0.10,
        "spatial_depth": +0.06,
        "micro_dynamics": -0.08,
        "natuerlichkeit": -0.06,
    },
    # Disco / Funk — dancefloor dynamics and bass groove
    "disco": {
        "groove": +0.12,  # Four-on-the-floor beat precision
        "bass_kraft": +0.10,  # Prominent bass: bassline drives the track
        "spatial_depth": +0.06,  # Wide stereo mix standard for disco
        "micro_dynamics": -0.04,  # Light compression typical for dancefloor
    },
    "funk": {"groove": +0.12, "bass_kraft": +0.10, "spatial_depth": +0.04, "micro_dynamics": +0.04},
    # Reggae / Dancehall — Off-Beat-Skank, prominenter Bass, jamaikanische Raumästhetik
    "reggae": {
        "groove": +0.10,  # Off-Beat-Skank ist präzise definiert; rhythmisches Kern-Feature
        "bass_kraft": +0.14,  # Riddim-Bass führt melodisch wie rhythmisch — zentrales Element
        "waerme": +0.12,  # Kingston-Studio-Akustik: warmer Raum, analoge Röhrengeräte
        "authentizitaet": +0.10,  # Produktions-Ästhetik darf nicht sterilisiert werden
        "natuerlichkeit": +0.08,  # Organische Aufnahmen; Raum-Ambience ist Stilmerkmal
        "spatial_depth": +0.06,  # Dub-Tradition: Raum und Echo sind kompositorische Mittel
        "brillanz": -0.06,  # Dunkler, bassbetonter Klangcharakter; keine HF-Überhöhung
    },
    "dancehall": {"groove": +0.10, "bass_kraft": +0.14, "waerme": +0.08, "authentizitaet": +0.08, "brillanz": -0.06},
    # Country — Nashville Sound / Bakersfield / Outlaw; Vokal-Artikulation primär
    "country": {
        "authentizitaet": +0.12,  # Authentizität ist Genre-definierend (kein Overproducing)
        "artikulation": +0.10,  # Textverständlichkeit ist primäres Vokal-Ziel
        "waerme": +0.10,  # Warmer Vokal-Mix, natürliche Instrumente
        "natuerlichkeit": +0.08,  # Live-Raum, minimale Elektronik (außer 1980s Nashville)
        "spatial_depth": +0.04,  # Moderat; Nashville-Raum-Ästhetik erwartet
        "micro_dynamics": +0.04,  # Expressive Vocals brauchen Dynamikspielraum
        "brillanz": -0.04,  # Warmer, nicht heller Mix-Charakter
    },
    # Gospel — Vokal-Supremacy maximal; Chor-Ensemble; emotionale Intensität ist das Produkt
    "gospel": {
        "emotionalitaet": +0.16,  # Emotionale Intensität ist das einzige Ziel — höchste Prio
        "authentizitaet": +0.14,  # Sterilisierung zerstört Genre-Essenz
        "natuerlichkeit": +0.12,  # Kirchen-Raumakustik und organischer Chor-Blend
        "spatial_depth": +0.10,  # Kirchenhall ist kompositorisches Stilmittel
        "micro_dynamics": +0.10,  # Vokal-Swells und Chor-Crescendos: Dynamik ist Sprache
        "waerme": +0.08,  # Warmer Kirchenraum, keine kalte Elektronik
        "groove": +0.06,  # Call-and-Response-Rhythmik; Swing-Gospel-Feel
        "bass_kraft": -0.06,  # Gospel-Mix ist vokalzentriert, nicht bassbetont
    },
    # Metal (Heavy, Thrash, Death, Doom) — Transient-Energie und Dichte primär
    "metal": {
        "transient_energie": +0.12,  # Schlagzeug-Attack und Gitarren-Chug sind das Produkt
        "brillanz": +0.10,  # Präsenz-Boost für Gitarren-Attack (2–5 kHz)
        "bass_kraft": +0.08,  # Tiefe Stimmen und Downtuned-Gitarren: Boden-Fundament
        "groove": +0.08,  # Rhythmische Präzision (Blast-Beat bis Groove-Metal)
        "micro_dynamics": -0.08,  # Typisch stärker komprimiert als Rock; bewusster Stil
        "natuerlichkeit": -0.10,  # Stark prozessiert by design (Gate, Trigger, Amp-Sim)
        "waerme": -0.06,  # Kühler, präsenzbetonter Klangcharakter dominiert
    },
    "heavy_metal": {
        "transient_energie": +0.12,
        "brillanz": +0.10,
        "bass_kraft": +0.08,
        "groove": +0.08,
        "micro_dynamics": -0.08,
        "natuerlichkeit": -0.10,
    },
    # Latin (Salsa, Cumbia, Bolero) — Afro-Kubanische Groove-Perkussion, warme Aufnahmen
    "latin": {
        "groove": +0.14,  # Afro-kubanische Rhythmik: Clave-Präzision ist Genre-definierend
        "authentizitaet": +0.12,  # Aufnahme-Ästhetik (Havanna, NYC, Bogotá) erhalten
        "waerme": +0.10,  # Warme analoge Studioakustik (1940s–1970s)
        "spatial_depth": +0.08,  # Live-Raumambience; Salsa-Sessions in großen Studios
        "artikulation": +0.08,  # Perkussions-Onset-Präzision ist rhythmisches Signal
        "natuerlichkeit": +0.06,  # Organische Live-Aufnahmen; keine Sterilisierung
    },
    "salsa": {
        "groove": +0.14,
        "authentizitaet": +0.12,
        "waerme": +0.10,
        "spatial_depth": +0.08,
        "artikulation": +0.08,
        "natuerlichkeit": +0.06,
    },
    # Bossa Nova — Kammermusik-Dichte, intimem Vokal-Gitarren-Dialog, São Paulo-Studioästhetik
    "bossa_nova": {
        "natuerlichkeit": +0.14,  # Intime Akustik-Ästhetik; über-prozessieren ist ein Sakrileg
        "authentizitaet": +0.12,  # João Gilberto-Estetik: Stille und Subtilität bewahren
        "waerme": +0.10,  # Warme analoge Gitarren-Vokal-Balance
        "micro_dynamics": +0.10,  # Flüstergesang und Pianissimo-Gitarre: Dynamik ist Stilmerkmal
        "groove": +0.08,  # Samba-abgeleiteter Puls; subtil aber präzise
        "brillanz": -0.08,  # Dunkler, intimer Klang; keine HF-Überbetonung
        "spatial_depth": -0.04,  # Enge, intime Raumdarstellung ist Stilmerkmal
    },
    "bossa nova": {
        "natuerlichkeit": +0.14,
        "authentizitaet": +0.12,
        "waerme": +0.10,
        "micro_dynamics": +0.10,
        "groove": +0.08,
        "brillanz": -0.08,
        "spatial_depth": -0.04,
    },
    # Oper / Klassischer Gesang — Saalakustik, Vokal-Projektion, Artikulation als Kunst
    "oper": {
        "spatial_depth": +0.22,  # Opernsaal: Hallradius und Diffusivität sind Klangerlebnis-Kern
        "natuerlichkeit": +0.14,  # Akustische Aufnahme ohne Elektronik-Verfremdung
        "emotionalitaet": +0.12,  # Dramatische Intensität ist das Produkt
        "artikulation": +0.10,  # Textverständlichkeit: Libretto-Dikton ist künstlerisches Ziel
        "brillanz": +0.06,  # Opernsopran/tenor trägt über Orchester: HF-Präsenz nötig
        "micro_dynamics": +0.10,  # Pianissimo bis Fortissimo: DR-Bandbreite ist Kunststilmittel
        "authentizitaet": +0.08,  # Historische Aufnahme-Ästhetik (1930s–1960s) bewahren
        "waerme": -0.04,  # Kühler Saal-Charakter (Stein/Holz) vs. Studio-Wärme
    },
    "opera": {
        "spatial_depth": +0.22,
        "natuerlichkeit": +0.14,
        "emotionalitaet": +0.12,
        "artikulation": +0.10,
        "brillanz": +0.06,
        "micro_dynamics": +0.10,
        "authentizitaet": +0.08,
        "waerme": -0.04,
    },
}


def _era_key(decade: int | None) -> str:
    """Map decade to bias bucket (10 Buckets: acoustic_era, early_electric, 1950s–2010s)."""
    if decade is None:
        return "1970s"
    if decade < 1926:
        return "acoustic_era"  # Rein akustische Trichteraufnahme (pre-Mikrofon)
    if decade < 1950:
        return "early_electric"  # Frühelektrisch: Mikrofon + Röhre, kein Magnetband
    if decade < 1960:
        return "1950s"
    if decade < 1970:
        return "1960s"
    if decade < 1980:
        return "1970s"
    if decade < 1990:
        return "1980s"
    if decade < 2000:
        return "1990s"
    if decade < 2010:
        return "2000s"
    return "2010s"


def estimate_song_goal_targets(
    *,
    is_studio_2026: bool = False,
    goal_weights: dict[str, float] | None = None,
    restorability_score: float | None = None,
    era_decade: int | None = None,
    genre_label: str | None = None,
    material_type: str | None = None,
    transfer_chain: list[str] | None = None,
    production_profile: object | None = None,
    restoration_prior: dict | None = None,
) -> dict[str, float]:
    """Berechnet per-song goal targets as studio-day reconstruction targets.

    Returns a dict mapping each goal name to its target value in [0.30, 0.99].
    Targets are blended from canonical floors (§09.1), era-/material-/genre-biases
    (§09.2), goal importance weights (§2.56) and restorability.

    These targets indicate where the pipeline *should stop* — the reconstructed
    studio-day score for this specific song.  They are NOT hard gates; they inform
    PhaseConductor strength recommendations and PMGG over-processing detection.

    Args:
        is_studio_2026: Studio 2026 mode uses higher canonical floors.
        goal_weights:   Per-song goal importance from §2.56 (1.0 = default).
        restorability_score: 0–100 from RestorabilityEstimator.
        era_decade:        Decade (e.g. 1970) from EraClassifier.
        genre_label:        Genre string (e.g. "schlager") from GenreClassifier.
        material_type:      Primary material (e.g. "vinyl") from MediumDetector.
        transfer_chain:     Full chain list (e.g. ["vinyl","tape","mp3_low"]).
        production_profile: Optional ProductionProfile from RecordingProductionKB.
                            Its goal_adjustments are applied as 4th bias layer
                            with kappa_provenance=0.30 (conservative).
        restoration_prior:  Optional Prior aus RestorationMemory (§2.70).
                            Wenn hpi_achieved > 0.75, wird kappa leicht erhöht
                            (max +0.10, gedeckelt auf kappa_base) — adaptive
                            Annäherung an das bekannte Optimum dieser Era/Material-
                            Kombination.

    Returns:
        dict[str, float]: Per-goal targets, same keys as CANONICAL_THRESHOLDS.
    """
    # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 70.0).
    from backend.core.calibration_context import resolve_restorability_score

    restorability_score = resolve_restorability_score(restorability_score)
    canonical = CANONICAL_THRESHOLDS_STUDIO2026 if is_studio_2026 else CANONICAL_THRESHOLDS_RESTORATION
    weights = goal_weights or {}
    rest_norm = float(np.clip(restorability_score / 100.0, 0.0, 1.0))

    # Resolve material from chain if not given directly
    primary_mat = (str(material_type or "").strip().lower()) or (
        str((transfer_chain or [""])[0]).strip().lower() if transfer_chain else ""
    )
    mat_class = _MATERIAL_CLASS.get(primary_mat, "analog")
    era_bucket = _era_key(era_decade)
    # Normalize genre: lower + collapse spaces and hyphens to underscores
    # e.g. "Hip Hop" → "hip_hop", "Hip-Hop" → "hip_hop", "R&B" → "r&b"
    genre_key = str(genre_label or "").strip().lower().replace(" ", "_").replace("-", "_")
    # "R B" / "R-B" → "r_b" after normalization; map back to the "r&b" dict key
    if genre_key == "r_b":
        genre_key = "r&b"

    # Accumulate biases
    bias: dict[str, float] = {}
    for b_dict in [
        _ERA_BIAS.get(era_bucket, {}),
        _MATERIAL_BIAS.get(mat_class, {}),
        _GENRE_BIAS.get(genre_key, {}),
    ]:
        for goal, delta in b_dict.items():
            bias[goal] = bias.get(goal, 0.0) + float(delta)

    # §09.2 Transfer-chain compound bias: secondary links contribute partial material biases.
    # Each subsequent medium adds further real degradation on top of the primary.
    # Weights: link[1]=0.40, link[2+]=0.20 — diminishing returns per generation.
    # Example: vinyl+tape+mp3_low → tape_analog at 0.40 + lossy at 0.20.
    # Only applied when the secondary class differs from the primary (no double-counting).
    if transfer_chain and len(transfer_chain) > 1:
        for _ci, _cm in enumerate(transfer_chain[1:4], start=1):  # cap at 3 extra links
            _cm_key = str(_cm).strip().lower()
            _cm_class = _MATERIAL_CLASS.get(_cm_key, "")
            if _cm_class and _cm_class != mat_class:
                _link_w = 0.40 if _ci == 1 else 0.20
                for _g, _d in _MATERIAL_BIAS.get(_cm_class, {}).items():
                    bias[_g] = bias.get(_g, 0.0) + _link_w * float(_d)

    # §Bug#2 Alias-Propagation: bias tables use legacy keys (raumtiefe, mikrodynamik,
    # tonalcenter, basskraft). Canonical dicts carry both old and new keys.
    # Propagate biases so new-key lookups (spatial_depth, micro_dynamics, etc.) work.
    _OLD_TO_NEW: dict[str, str] = {
        "raumtiefe": "spatial_depth",
        "mikrodynamik": "micro_dynamics",
        "tonalcenter": "tonal_center",
        "basskraft": "bass_kraft",
    }
    _NEW_TO_OLD: dict[str, str] = {v: k for k, v in _OLD_TO_NEW.items()}
    for old_k, new_k in _OLD_TO_NEW.items():
        if old_k in bias and new_k not in bias:
            bias[new_k] = bias[old_k]
        elif new_k in bias and old_k not in bias:
            bias[old_k] = bias[new_k]

    # kappa: how strongly biases are applied (low restorability → more conservative)
    # Restoration: 0.45; Studio 2026: 0.65; modulated by restorability.
    # §Lücke5 S-Kurve: logistische Funktion statt linearer Interpolation.
    # Gleiche Grenzen [0.60·kappa_base, kappa_base] wie zuvor, aber mittlere
    # Restorability-Bereiche reagieren stärker (steile Kurvenphase), Extreme flacher.
    # Formel: sigmoid(x) = 1 / (1 + exp(-k*(x-0.5))); k=10 ergibt [0.007, 0.993] für x in [0,1].
    # Skaliert auf [0.60, 1.0] → mit kappa_base multipliziert: [0.27, 0.65] Restoration.
    kappa_base = 0.65 if is_studio_2026 else 0.45
    _sigmoid = 1.0 / (1.0 + float(np.exp(-10.0 * (rest_norm - 0.5))))
    kappa = float(np.clip(kappa_base * (0.60 + 0.40 * _sigmoid), 0.0, kappa_base))

    # §2.70 RestorationMemory-Prior als kappa-Modulator:
    # Bekannte gute Ergebnisse (hpi_achieved > 0.75) für diese Era/Material/Defect-
    # Kombination → kappa leicht anheben, damit Targets dichter am nachgewiesen
    # erreichbaren Niveau liegen. Deckel: kappa_base (kein Über-Boost).
    if restoration_prior is not None:
        _prior_hpi = float(restoration_prior.get("hpi_achieved", 0.0))
        if _prior_hpi > 0.75:
            _kappa_memory_boost = float(np.clip((_prior_hpi - 0.75) * 0.40, 0.0, 0.10))
            kappa = float(np.clip(kappa + _kappa_memory_boost, 0.0, kappa_base))

    # §v10.200 Depth-adaptive kappa: tiefere Transfer-Ketten haben physikalisch
    # begrenzte erreichbare Ziele. Der kappa-Faktor bestimmt, wie stark Era/Genre/
    # Material-Bias vom kanonischen Floor abheben. Bei depth≥4 wird kappa reduziert.
    _depth = max(1, len(transfer_chain or []))
    if _depth >= 5:
        _depth_kappa_factor = 0.50
    elif _depth >= 4:
        _depth_kappa_factor = 0.65
    elif _depth >= 3:
        _depth_kappa_factor = 0.85
    else:
        _depth_kappa_factor = 1.0
    kappa *= _depth_kappa_factor

    # Provenance bias (4th layer) — kappa_provenance is fixed at 0.30 (conservative).
    # RecordingProductionKB adjustments are more specific but also more uncertain
    # than era/genre/material biases, so they are applied with smaller weight.
    _kappa_prov = 0.30
    _prov_adj: dict[str, float] = {}
    if production_profile is not None:
        _raw_adj = getattr(production_profile, "goal_adjustments", {})
        if isinstance(_raw_adj, dict):
            for _g, _v in _raw_adj.items():
                _prov_adj[str(_g)] = float(_v)

    # §09.2 Goal-class-adaptive weight-shift scale.
    # Vocal-sensitive P2/P3 goals respond more strongly to goal_weights
    # (e.g. panns_singing → artikulation/emotionalitaet weight boost from §2.56 SGI).
    # P5 goals are less weight-sensitive (material ceiling dominates there).
    _WEIGHT_SHIFT_SCALE: dict[str, float] = {
        "natuerlichkeit": 0.08,
        "authentizitaet": 0.08,  # P1
        "tonal_center": 0.10,
        "tonalcenter": 0.10,
        "timbre_authentizitaet": 0.10,
        "artikulation": 0.10,  # P2 — vocal-sensitive
        "emotionalitaet": 0.10,
        "micro_dynamics": 0.09,
        "mikrodynamik": 0.09,  # P3
        "groove": 0.07,
        "transparenz": 0.07,
        "waerme": 0.06,  # P4
        "bass_kraft": 0.06,
        "basskraft": 0.06,
        "separation_fidelity": 0.06,
        "brillanz": 0.06,
        "raumtiefe": 0.05,
        "spatial_depth": 0.05,  # P5
    }

    targets: dict[str, float] = {}
    for goal, floor in canonical.items():
        b = bias.get(goal, 0.0) + _kappa_prov * _prov_adj.get(goal, 0.0)
        w = float(weights.get(goal, 1.0))
        # goal weight > 1.0 → song needs this goal → stay closer to or above floor
        # goal weight < 1.0 → goal less important for this song → can tolerate lower target
        weight_shift = (w - 1.0) * _WEIGHT_SHIFT_SCALE.get(goal, 0.06)
        target = floor + kappa * b + weight_shift
        # Hard bounds: never below 0.30, never above 0.99 (1.0 is unreachable)
        targets[goal] = float(np.clip(target, 0.30, 0.99))

    return targets


# ---------------------------------------------------------------------------
# §09.7 Expected Quality Score (UI Baseline Prediction)
# ---------------------------------------------------------------------------

_MATERIAL_QUALITY_CEILING: dict[str, float] = {
    "wax_cylinder": 0.55,
    "shellac": 0.70,
    "lacquer_disc": 0.68,
    "wire_recording": 0.65,
    "vinyl": 0.88,
    "lp": 0.88,
    "tape": 0.85,
    "reel_tape": 0.86,
    "cassette": 0.80,
    "cd_digital": 0.95,
    "cd": 0.95,
    "dat": 0.92,
    "minidisc": 0.85,
    "mp3_low": 0.78,
    "mp3_high": 0.88,
    # Aliases / additional lossy keys — same ceiling as parent class
    "kassette": 0.80,  # Alias for cassette — German spelling; same physical medium
    "aac": 0.88,  # AAC 256+ kbps ≥ MP3 320kbps perceptually (Fraunhofer IIS 2022)
    "streaming": 0.88,  # Streaming quality (Spotify AAC 256 / Apple 256) ≈ mp3_high
}


# ---------------------------------------------------------------------------
# §09.8 Material-adaptive goal floors — get_material_floor (RELEASE_MUST)
# ---------------------------------------------------------------------------


def get_material_floor(
    material_type: str,
    goal: str,
    is_studio_2026: bool = False,
    transfer_chain: list[str] | None = None,
) -> float:
    """Gibt the minimum achievable goal floor for a given material type (§09.1) zurück.

    Unlike the canonical thresholds (CD-equivalent), this returns the
    material-adjusted minimum floor accounting for physical medium limits.
    REQUIRED: callers MUST use this instead of CANONICAL_THRESHOLDS directly
    when evaluating goals against material-specific limits (§0a, §2.44, §2.45b).

    Args:
        material_type: e.g. "shellac", "vinyl", "cd_digital"
        goal: goal key e.g. "brillanz", "natuerlichkeit"
        is_studio_2026: mode flag

    Returns:
        float: minimum achievable floor ∈ [0.30, 0.99]

    Examples:
        shellac brillanz → ~0.72 (physical BW limit)
        vinyl  brillanz → ~0.76
        cd     brillanz → ~0.80
    """
    canonical = CANONICAL_THRESHOLDS_STUDIO2026 if is_studio_2026 else CANONICAL_THRESHOLDS_RESTORATION
    floor = float(canonical.get(goal, 0.70))

    mat = str(material_type or "").strip().lower()

    # §R4: Physikalisch begrenzte Material-Goal-Overrides prüfen (VOR Bias-Berechnung).
    # Diese Overrides ersetzen den bias-basierten Floor für trägerphysikalisch unlösbare Goals.
    _mat_override = _MATERIAL_GOAL_FLOOR_OVERRIDES.get(mat)
    if _mat_override and goal in _mat_override:
        return float(np.clip(_mat_override[goal], 0.30, 0.99))

    # §R4/S4: Chain-End-Codec-Override — falls primäres Material kein Override hat,
    # aber die Transfer-Chain mit einem Codec endet, dessen Override greift.
    # Typischer Fall: Kassette→mp3_low (primary='cassette', chain=['mp3_low']).
    # MP3-Joint-Stereo begrenzt spatial_depth unabhängig vom analogen Primärträger.
    _CODEC_CHAIN_ENDINGS_CM = frozenset({"mp3_low", "mp3_high", "aac", "streaming", "minidisc"})
    if transfer_chain:
        _chain_last_cm = str(transfer_chain[-1]).lower().replace(" ", "_")
        if _chain_last_cm in _CODEC_CHAIN_ENDINGS_CM and _chain_last_cm != mat:
            _chain_override = _MATERIAL_GOAL_FLOOR_OVERRIDES.get(_chain_last_cm)
            if _chain_override and goal in _chain_override:
                return float(np.clip(_chain_override[goal], 0.30, 0.99))

    mat_class = _MATERIAL_CLASS.get(mat, "analog")

    # Apply material bias at minimum restorability kappa (worst-case → lowest floor).
    # kappa_min = kappa_base * 0.60 (restorability=0, §09.2 kappa range).
    kappa_base = 0.65 if is_studio_2026 else 0.45
    kappa_min = kappa_base * 0.60

    bias_val = float(_MATERIAL_BIAS.get(mat_class, {}).get(goal, 0.0))
    material_floor = floor + kappa_min * bias_val

    # §0a/§9.12.3 Studio2026-Invarianz: Studio2026-Floor darf NIEMALS unter dem
    # Restoration-Floor für dasselbe Material liegen. Studio2026 ist eine Obermenge
    # von Restoration (macht alles was Restoration tut + Enhancement). Wenn der größere
    # kappa_base (0.65 vs 0.45) zusammen mit negativem Material-Bias die Studio2026-
    # Formel unter den Restoration-Wert drückt, ist das ein Kalibrierungsfehler:
    # vinyl/natuerlichkeit: studio=0.92-0.39*0.30=0.803 < restoration=0.90-0.27*0.30=0.819.
    # Fix: Studio2026-Floor = max(computed, restoration_floor) → Monotonie-Invariante.
    if is_studio_2026:
        rest_canonical = float(CANONICAL_THRESHOLDS_RESTORATION.get(goal, 0.70))
        rest_kappa_min = 0.45 * 0.60  # kappa_min für Restoration
        rest_floor = rest_canonical + rest_kappa_min * bias_val
        rest_floor = float(np.clip(rest_floor, 0.30, 0.99))
        material_floor = max(material_floor, rest_floor)

    return float(np.clip(material_floor, 0.30, 0.99))


# ---------------------------------------------------------------------------
# §09.12 [RELEASE_MUST] Restorability-adaptive Floor-Skalierung
# ---------------------------------------------------------------------------

# Minimaler Skalierungsfaktor (restorability=0 → Boden × 0.72)
# Verhindert, dass hoffnungslose Tracks einen unerreichbaren Floor gesetzt bekommen.
RESTORABILITY_SCALE_MIN: float = 0.72


def get_effective_material_floor(
    material_type: str,
    goal_name: str | None = None,
    goal: str | None = None,
    restorability_score: float = 100.0,
    is_studio_2026: bool = False,
    **kwargs: Any,
) -> float:
    """Gibt restorability-skalierten Floor für §GOAL_BASELINE_CHECK zurück.

    Unterschied zu get_material_floor():
    - get_material_floor() = normative Böden (PMGG, UI, Tests) — UNVERÄNDERLICH
    - get_effective_material_floor() = adaptiver Floor für §GOAL_BASELINE_CHECK in UV3

    Formel (§09.12):
        floor_base = get_material_floor(material_type, goal_name, is_studio_2026)
        scale = max(RESTORABILITY_SCALE_MIN, restorability_score / 100)
        floor_eff = floor_base × scale

    Sonderfälle:
        restorability < 30 → metadata["degraded_restorability"] = True soll in UV3 gesetzt werden
        (diese Funktion setzt das Flag nicht — UV3 trägt dafür die Verantwortung)

    Args:
        material_type:      e.g. "shellac", "vinyl", "mp3_low"
        goal_name:          Kanonischer Goal-Schlüssel, e.g. "brillanz"
        goal:               Expliziter Alias für goal_name (Legacy-Callsites)
        kwargs["goal"]:     Rueckwaertskompatibler Alias fuer goal_name
        restorability_score: 0–100
        is_studio_2026:     Modus-Flag

    Returns:
        float: Effektiver Floor ∈ [0.20, 0.99]
    """
    _goal_key = str(goal_name or goal or kwargs.get("goal") or "").strip()
    if not _goal_key:
        _goal_key = "natuerlichkeit"
    floor_base = get_material_floor(material_type, _goal_key, is_studio_2026=is_studio_2026)
    scale = max(RESTORABILITY_SCALE_MIN, float(np.clip(float(restorability_score) / 100.0, 0.0, 1.0)))
    return float(np.clip(floor_base * scale, 0.20, 0.99))


_CHAIN_END_GOAL_CEILINGS: dict[str, dict[str, float]] = {
    "shellac": {
        "natuerlichkeit": 0.68,
        "authentizitaet": 0.65,
        "tonal_center": 0.70,
        "timbre_authentizitaet": 0.65,
        "artikulation": 0.61,
        "waerme": 0.62,
        "brillanz": 0.52,
        "transparenz": 0.55,
        "separation_fidelity": 0.60,
    },
    "wax_cylinder": {
        "natuerlichkeit": 0.66,
        "authentizitaet": 0.63,
        "tonal_center": 0.68,
        "timbre_authentizitaet": 0.63,
        "artikulation": 0.59,
        "waerme": 0.60,
        "brillanz": 0.40,
        "transparenz": 0.50,
        "separation_fidelity": 0.55,
    },
    "wire_recording": {
        "natuerlichkeit": 0.66,
        "authentizitaet": 0.63,
        "tonal_center": 0.68,
        "timbre_authentizitaet": 0.63,
        "artikulation": 0.59,
        "waerme": 0.60,
        "brillanz": 0.44,
        "transparenz": 0.52,
        "separation_fidelity": 0.56,
    },
    "lacquer_disc": {
        "natuerlichkeit": 0.69,
        "authentizitaet": 0.66,
        "tonal_center": 0.71,
        "timbre_authentizitaet": 0.66,
        "artikulation": 0.62,
        "waerme": 0.63,
        "brillanz": 0.50,
        "transparenz": 0.56,
        "separation_fidelity": 0.60,
    },
    "vinyl": {
        "natuerlichkeit": 0.82,
        "authentizitaet": 0.79,
        "tonal_center": 0.84,
        "timbre_authentizitaet": 0.79,
        "artikulation": 0.76,
        "waerme": 0.74,
        "brillanz": 0.82,
        "transparenz": 0.78,
        "separation_fidelity": 0.76,
    },
    "reel_tape": {
        "natuerlichkeit": 0.82,
        "authentizitaet": 0.79,
        "tonal_center": 0.84,
        "timbre_authentizitaet": 0.79,
        "artikulation": 0.76,
        "waerme": 0.74,
        "brillanz": 0.82,
        "transparenz": 0.78,
        "separation_fidelity": 0.76,
    },
    "tape": {
        "natuerlichkeit": 0.78,
        "authentizitaet": 0.75,
        "tonal_center": 0.80,
        "timbre_authentizitaet": 0.75,
        "artikulation": 0.72,
        "waerme": 0.72,
        "brillanz": 0.78,
        "transparenz": 0.74,
        "separation_fidelity": 0.72,
    },
    "cassette": {
        "natuerlichkeit": 0.76,
        "authentizitaet": 0.73,
        "tonal_center": 0.78,
        "timbre_authentizitaet": 0.73,
        "artikulation": 0.70,
        "waerme": 0.70,
        "brillanz": 0.72,
        "transparenz": 0.68,
        "separation_fidelity": 0.68,
    },
    "mp3_low": {
        "natuerlichkeit": 0.76,
        "authentizitaet": 0.74,
        "tonal_center": 0.78,
        "timbre_authentizitaet": 0.74,
        "artikulation": 0.70,
        "waerme": 0.70,
        "brillanz": 0.45,
        # MP3-low: Pre-Echo + MDCT smear begrenzen attack-klarheit; transient target cappen
        # damit §2.54 keine ueberphysikalischen Sollwerte auf alten Kassetten-MP3-Rips fordert.
        "transient_energie": 0.70,
        "transparenz": 0.60,
        "separation_fidelity": 0.62,
        # §R4/S4: MP3-Joint-Stereo begrenzt IACC-basierte Raumtiefe physikalisch.
        # Ceiling = get_material_floor("mp3_low", "spatial_depth") = 0.38.
        "spatial_depth": 0.38,
    },
    "mp3_high": {
        "natuerlichkeit": 0.82,
        "authentizitaet": 0.80,
        "tonal_center": 0.84,
        "timbre_authentizitaet": 0.80,
        "artikulation": 0.76,
        "waerme": 0.74,
        "brillanz": 0.65,
        "transient_energie": 0.76,
        "transparenz": 0.70,
        "separation_fidelity": 0.70,
        "spatial_depth": 0.44,  # §R4/S4: MP3 320kbps joint-stereo noch aktiv, aber schwächer.
    },
    "aac": {
        "natuerlichkeit": 0.84,
        "authentizitaet": 0.82,
        "tonal_center": 0.86,
        "timbre_authentizitaet": 0.82,
        "artikulation": 0.78,
        "waerme": 0.76,
        "brillanz": 0.72,
        "transient_energie": 0.79,
        "transparenz": 0.74,
        "separation_fidelity": 0.74,
        "spatial_depth": 0.44,  # §R4/S4: AAC HE-v2 parametric stereo → IACC-Limit.
    },
    "minidisc": {
        "natuerlichkeit": 0.80,
        "authentizitaet": 0.78,
        "tonal_center": 0.82,
        "timbre_authentizitaet": 0.78,
        "artikulation": 0.74,
        "waerme": 0.72,
        "brillanz": 0.64,
        "transient_energie": 0.74,
        "transparenz": 0.68,
        "separation_fidelity": 0.68,
        "spatial_depth": 0.36,  # §R4/S4: ATRAC joint-stereo/parametric stereo → IACC-Limit.
    },
    "dat": {
        "natuerlichkeit": 0.88,
        "authentizitaet": 0.86,
        "tonal_center": 0.92,
        "timbre_authentizitaet": 0.86,
        "artikulation": 0.84,
        "waerme": 0.80,
        "brillanz": 0.82,
        "transparenz": 0.82,
        "separation_fidelity": 0.82,
    },
}


def estimate_chain_end_goal_ceiling(transfer_chain: list[str] | tuple[str, ...] | None) -> dict[str, float]:
    """Gibt Goal-Ceilings des letzten Tonträgerketten-Glieds zurück (§2.46a)."""
    chain = [str(stage).strip().lower() for stage in (transfer_chain or []) if str(stage).strip()]
    if not chain:
        return {}
    return dict(_CHAIN_END_GOAL_CEILINGS.get(chain[-1], {}))


def resolve_effective_goal_targets(
    *,
    is_studio_2026: bool = False,
    goal_weights: dict[str, float] | None = None,
    restorability_score: float | None = None,
    era_decade: int | None = None,
    genre_label: str | None = None,
    material_type: str | None = None,
    transfer_chain: list[str] | tuple[str, ...] | None = None,
    physical_ceiling: dict[str, float] | None = None,
    applicable_goals: set[str] | list[str] | tuple[str, ...] | None = None,
    production_profile: object | None = None,
) -> dict[str, float]:
    """Berechnet effektive, physikalisch gedeckelte Goal-Zielwerte (§1.2b/§09.2b)."""
    # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 70.0).
    from backend.core.calibration_context import resolve_restorability_score

    restorability_score = resolve_restorability_score(restorability_score)
    canonical = CANONICAL_THRESHOLDS_STUDIO2026 if is_studio_2026 else CANONICAL_THRESHOLDS_RESTORATION
    mat = str(material_type or "").strip().lower() or (
        str((transfer_chain or [""])[0]).strip().lower() if transfer_chain else "unknown"
    )
    song_targets = estimate_song_goal_targets(
        is_studio_2026=is_studio_2026,
        goal_weights=goal_weights,
        restorability_score=restorability_score,
        era_decade=era_decade,
        genre_label=genre_label,
        material_type=mat,
        transfer_chain=list(transfer_chain or []),
        production_profile=production_profile,
    )
    physical = dict(physical_ceiling or {})
    chain_cap = estimate_chain_end_goal_ceiling(transfer_chain)
    goals = list(applicable_goals) if applicable_goals else list(canonical.keys())

    resolved: dict[str, float] = {}
    for raw_goal in goals:
        goal = str(raw_goal or "").strip().lower()
        if not goal:
            continue
        floor_eff = get_effective_material_floor(
            mat, goal, restorability_score=restorability_score, is_studio_2026=is_studio_2026
        )
        target = max(floor_eff, float(song_targets.get(goal, canonical.get(goal, 0.70))))
        target = min(target, float(physical.get(goal, 0.99)), float(chain_cap.get(goal, 0.99)))
        resolved[goal] = float(np.clip(target, 0.20, 0.99))
    return resolved


# ---------------------------------------------------------------------------
# §09.9 Material-adaptive phase strength ranges — get_phase_strength_range
# ---------------------------------------------------------------------------

# (min_strength, max_strength) per material class and phase.
# max_strength caps over-processing on fragile media (§2.45b, §0a BW-Ceiling).
# min_strength ensures the phase is effective enough to be worth running.
# Phases not listed fall back to _DEFAULT_STRENGTH_RANGE.
_PHASE_STRENGTH_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "ultra_analog": {
        # Shellac / wax_cylinder / wire_recording / lacquer_disc
        # BW ≤ 8 kHz, SNR ~15 dB, Mono — aggressive processing would hallucinate.
        "phase_03_denoise": (0.15, 0.60),  # OMLSA/DFN: hard cap — SNR ~15 dB
        "phase_06_frequency_restoration": (0.05, 0.35),  # BW-Ceiling ≤ 8 kHz (§6.2c)
        "phase_07_harmonic_restoration": (0.05, 0.30),  # BW-Ceiling strict
        "phase_09_crackle_removal": (0.20, 0.75),  # Key phase for shellac/vinyl crackle
        "phase_12_wow_flutter_fix": (0.10, 0.55),  # Wow/Flutter present but gentle
        "phase_20_reverb_reduction": (0.05, 0.20),  # Studio early reflections protect
        "phase_23_spectral_repair": (0.10, 0.50),  # Spectral gaps limited
        "phase_26_dynamic_range_expansion": (0.00, 0.25),  # DR ceiling shellac ≤ 45 dB
        "phase_29_tape_hiss_reduction": (0.10, 0.55),  # Surface noise, not tape hiss
        "phase_35_multiband_compression": (0.05, 0.20),  # Gentle — original dynamics key
        "phase_46_spatial_enhancement": (0.00, 0.10),  # Mono source — no stereo widening
        "phase_48_stereo_width_enhancer": (0.00, 0.00),  # VERBOTEN on mono material
        "phase_49_advanced_dereverb": (0.05, 0.20),  # Early reflections cap §2.46f
        "phase_55_diffusion_inpainting": (0.10, 0.45),  # Dropout repair — careful
    },
    "analog": {
        # Vinyl / tape / reel_tape / cassette
        # Moderate limitations — standard restoration range.
        "phase_03_denoise": (0.10, 0.75),
        "phase_06_frequency_restoration": (0.10, 0.60),  # BW vinyl ≤ 16 kHz
        "phase_07_harmonic_restoration": (0.10, 0.55),
        "phase_09_crackle_removal": (0.20, 0.85),
        "phase_12_wow_flutter_fix": (0.15, 0.70),
        "phase_20_reverb_reduction": (0.05, 0.35),
        "phase_23_spectral_repair": (0.15, 0.70),
        "phase_26_dynamic_range_expansion": (0.00, 0.50),  # vinyl ≤ 70 dB
        "phase_29_tape_hiss_reduction": (0.15, 0.70),
        "phase_35_multiband_compression": (0.05, 0.40),
        "phase_46_spatial_enhancement": (0.05, 0.40),  # Vinyl stereo (post-1954) — moderate OK
        "phase_48_stereo_width_enhancer": (0.00, 0.30),  # Vinyl has natural stereo — cap at 0.30
        "phase_49_advanced_dereverb": (0.05, 0.35),
        "phase_55_diffusion_inpainting": (0.15, 0.65),
    },
    "lossy": {
        # mp3_low / mp3_high / aac / minidisc
        # Codec artifacts — spectral repair is key, denoise moderate.
        "phase_03_denoise": (0.05, 0.55),
        "phase_06_frequency_restoration": (0.15, 0.70),  # HF reconstruction primary
        "phase_07_harmonic_restoration": (0.15, 0.65),
        "phase_12_wow_flutter_fix": (0.00, 0.10),  # MP3/AAC: no wow/flutter — near-prohibit
        "phase_23_spectral_repair": (0.25, 0.90),  # Primary phase for lossy
        "phase_29_tape_hiss_reduction": (0.00, 0.10),  # MP3/AAC: no tape hiss — near-prohibit
        "phase_35_multiband_compression": (0.05, 0.40),
        "phase_50_spectral_repair": (0.25, 0.85),
    },
    "tape_analog": {
        # Reel tape / cassette — tape hiss primary, dropout secondary, narrow/mono stereo.
        # tape_analog biases differ significantly from vinyl (analog) — must NOT fall back
        # to default (0.05, 1.0) which would allow unrestricted processing on fragile medium.
        "phase_03_denoise": (0.15, 0.80),  # Tape hiss more severe than vinyl surface noise
        "phase_06_frequency_restoration": (0.10, 0.55),  # BW ceiling: reel <=14kHz, cassette <=14kHz (§6.2c)
        "phase_07_harmonic_restoration": (0.10, 0.45),  # BW ceiling lower than vinyl
        "phase_09_crackle_removal": (0.05, 0.40),  # Tape has dropouts, not click crackle → low
        "phase_12_wow_flutter_fix": (0.15, 0.75),  # Tape W&F stronger than vinyl (IEC 0.2% WRMS)
        "phase_20_reverb_reduction": (0.05, 0.35),
        "phase_23_spectral_repair": (0.15, 0.70),  # Tape spectral gaps, dropouts
        "phase_26_dynamic_range_expansion": (0.00, 0.45),  # Tape DR ≤ 55–60 dB
        "phase_29_tape_hiss_reduction": (0.25, 0.85),  # Primary phase for tape — high priority
        "phase_35_multiband_compression": (0.05, 0.40),
        "phase_46_spatial_enhancement": (0.05, 0.25),  # Many tapes narrow stereo
        "phase_48_stereo_width_enhancer": (0.00, 0.20),  # Narrow stereo — cap hard (not 0 like mono)
        "phase_49_advanced_dereverb": (0.05, 0.35),
        "phase_55_diffusion_inpainting": (0.20, 0.70),  # Tape dropouts more common than vinyl
    },
    "digital": {
        # cd_digital / dat — near-lossless, minimal intervention (§2.45b).
        # Analog defect phases (crackle, tape hiss, wow/flutter) must be near-zero
        # to prevent unnecessary processing and micro-artifact introduction.
        "phase_03_denoise": (0.05, 0.35),
        "phase_06_frequency_restoration": (0.05, 0.40),
        "phase_07_harmonic_restoration": (0.05, 0.35),
        "phase_09_crackle_removal": (0.05, 0.30),
        "phase_12_wow_flutter_fix": (0.00, 0.10),  # CD/DAT: no wow/flutter — near-prohibit
        "phase_23_spectral_repair": (0.10, 0.55),
        "phase_26_dynamic_range_expansion": (0.00, 0.40),
        "phase_29_tape_hiss_reduction": (0.00, 0.10),  # CD/DAT: no tape hiss — near-prohibit
        "phase_35_multiband_compression": (0.05, 0.30),
        "phase_49_advanced_dereverb": (0.05, 0.30),
    },
}

_DEFAULT_STRENGTH_RANGE: tuple[float, float] = (0.05, 1.0)


def get_phase_strength_range(
    phase_id: str,
    material_type: str | None = None,
    restorability_score: float | None = None,
) -> tuple[float, float]:
    """Gibt (min_strength, max_strength) for a phase given material and restorability zurück.

    The max_strength caps over-processing on fragile material (§0a, §0h).
    The min_strength ensures the phase applies enough correction to be meaningful.
    High restorability reduces max_strength toward near-passthrough (§2.45b).

    Args:
        phase_id: e.g. "phase_03_denoise"
        material_type: e.g. "shellac", "vinyl", "cd_digital"
        restorability_score: 0–100 from RestorabilityEstimator

    Returns:
        tuple[float, float]: (min_strength, max_strength) both ∈ [0.0, 1.0]
    """
    mat = str(material_type or "").strip().lower()
    mat_class = _MATERIAL_CLASS.get(mat, "analog")
    # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 70.0).
    from backend.core.calibration_context import resolve_restorability_score

    restorability_score = resolve_restorability_score(restorability_score)

    mat_ranges = _PHASE_STRENGTH_RANGES.get(mat_class, {})
    min_s, max_s = mat_ranges.get(phase_id, _DEFAULT_STRENGTH_RANGE)

    # §2.45b: high restorability → near-passthrough → scale down max_strength.
    # Above 80 the scaling is linear from 1.0 to 0.30 at restorability=100.
    rest = float(np.clip(restorability_score, 0.0, 100.0))
    if rest > 80.0:
        passthrough_factor = 1.0 - 0.70 * ((rest - 80.0) / 20.0)
        passthrough_factor = float(np.clip(passthrough_factor, 0.30, 1.0))
        max_s = float(np.clip(max_s * passthrough_factor, min_s, max_s))

    return (float(min_s), float(max_s))


def predict_quality_score(
    material_type: str,
    restorability: float,
    defect_severity_mean: float,
    is_studio_2026: bool = False,
) -> float:
    """Predict expected OQS (Overall Quality Score) after full pipeline.

    Pure function — safe to call from UI/Bridge before processing starts.
    Returns a value ∈ [0.0, 0.99].
    """
    mat = str(material_type or "").strip().lower()
    ceiling = _MATERIAL_QUALITY_CEILING.get(mat, 0.75)
    rest_norm = float(np.clip(restorability / 100.0, 0.0, 1.0))
    defect_penalty = float(np.clip(defect_severity_mean, 0.0, 1.0)) * 0.15
    studio_boost = 0.08 if is_studio_2026 else 0.0
    score = rest_norm * ceiling - defect_penalty + studio_boost
    return float(np.clip(score, 0.0, 0.99))


# ---------------------------------------------------------------------------
# §09.10 Goal-to-Recovery-Phases mapping — get_goal_recovery_phases [RELEASE_MUST]
# ---------------------------------------------------------------------------
# Maps each of the 15 Musical Goals to its primary recovery phase(s).
# Used by UV3 §GOAL_BASELINE_CHECK to ensure recovery phases are in
# selected_phases when a goal proxy is below its material floor BEFORE the
# phase pipeline runs.
#
# RESTORATION list: corrective/subtractive phases only.
#   - No §0a-forbidden phases (phase_21_exciter, phase_35_multiband_compression,
#     phase_42_vocal_enhancement are NOT listed here).
#   - Ordering: most effective/safest first (primary recovery phase is index 0).
#
# STUDIO_EXTRAS list: additional additive/enhancement phases for Studio 2026.
#   - Only phases that §0a permits in Studio 2026 mode.
#
# Single Source of Truth for goal → phase mapping; CausalDefectReasoner
# CAUSE_TO_PHASES covers defect-driven selection. This covers goal-driven
# selection when no matching defect is detected (the "silent gap").

_GOAL_TO_RECOVERY_PHASES_RESTORATION: dict[str, list[str]] = {
    # Ordering principle — §2.46 Carrier-Chain hierarchy (normative, not phase-number order):
    #   1. Subtraktive / corrective phases BEFORE additive / enhancement phases
    #   2. Mechanical / physical-layer defects (Stufen 1–4) BEFORE digital processing
    #   3. Broadest-impact intervention first within the same Stufe tier
    # PRIMARY phase (index 0) = inserted by §GOAL_BASELINE_CHECK when goal < floor × 0.95.
    # Only the primary is inserted (§2.45 minimal-intervention). Subsequent phases serve
    # FeedbackChain iteration.
    #
    # P0 — Vocal timbre (carrier-level contributors; VQI-Gate handles vocal quality directly)
    "timbre_authentizitaet": [
        "phase_04_eq_correction",  # Stufe 2: RIAA / carrier transfer-function error — universal root
        "phase_25_azimuth_correction",  # Stufe 2: tape azimuth error causes comb-filter timbral distortion;
        # correct BEFORE EQ polish so EQ does not mask azimuth residual
        "phase_16_final_eq",  # Stufe 6: perceptual spectral rebalancing post correction
    ],
    # P1 — Naturalness & authenticity
    "natuerlichkeit": [
        "phase_03_denoise",  # Stufe 4: broadband NR — universally the largest single
        # contributor to perceived naturalness; all carrier types
        "phase_29_tape_hiss_reduction",  # Stufe 4: carrier-specific tape / groove surface noise
        "phase_02_hum_removal",  # Stufe 3: 50/60 Hz hum (electrical era, 1925–1960) —
        # continuous tonal interference that destroys naturalness
        "phase_59_modulation_noise_reduction",  # Stufe 4: signal-dependent modulation noise
        # (analog tape) — distinct from stationary hiss
    ],
    "authentizitaet": [
        "phase_09_crackle_removal",  # Stufe 3: systematic crackle / burst noise — most
        # damaging to authenticity; not always defect-detected
        "phase_24_dropout_repair",  # Stufe 3: tape dropouts create sudden silence — equal
        # severity to crackle; missed by scanner when infrequent
        "phase_29_tape_hiss_reduction",  # Stufe 4: continuous background noise degrades continuity
        "phase_57_print_through_reduction",  # Stufe 3: tape print-through — ghost pre/post-echo
        # from adjacent tape layer; very damaging to authenticity
        "phase_01_click_removal",  # Stufe 3: click events — _NEVER_SKIP so always runs;
        # listed here only for FeedbackChain completeness
    ],
    # P2 — Tonal / timbral / articulation
    "tonal_center": [
        "phase_12_wow_flutter_fix",  # Stufe 2: irregular mechanical rotation — MUST precede
        # any digital pitch processing; a time-varying pitch
        # deviation cannot be digitally stabilised if the
        # physical mechanism is still active
        "phase_25_azimuth_correction",  # Stufe 2: azimuth misalignment causes phase-dependent
        # HF loss that shifts perceived tonal center; correct
        # BEFORE digital pitch alignment
        "phase_31_speed_pitch_correction",  # residual digital pitch alignment on now-stable signal
    ],
    "timbre": [
        "phase_04_eq_correction",  # Stufe 2: carrier EQ correction (subtraktive, causal)
        "phase_25_azimuth_correction",  # Stufe 2: tape azimuth comb-filter — before EQ polish
        "phase_16_final_eq",  # Stufe 6: additive spectral rebalancing
    ],
    "artikulation": [
        "phase_08_transient_preservation",  # transient attack / decay envelope is the primary
        # carrier of consonant articulation clarity
        "phase_23_spectral_repair",  # spectral masking (codec, dropout) as secondary
    ],
    # P3 — Emotional / dynamic / rhythmic
    "emotionalitaet": [
        "phase_54_transparent_dynamics",  # §V28 primär: Envelope-Re-Smoothing stellt Mikrodynamik-
        # Profil nach NR-Glättung (DFN/SGMSE+/OMLSA) wieder her — kein neues Compression-Artefakt;
        # wirkt auf variance + micro + range, nicht nur auf crest (Gegensatz zu phase_26)
        "phase_26_dynamic_range_expansion",  # sekundär: Dynamic-Contrast-Erweiterung wenn Glättung
        # allein nicht ausreicht — hebt hauptsächlich crest_score
        "phase_08_transient_preservation",  # tertiär: Transient-Schärfe für emotionale Peaks
    ],
    "micro_dynamics": [
        "phase_26_dynamic_range_expansion",  # over-compression is the primary cause of
        # micro-dynamic loss across all mastering eras
        "phase_08_transient_preservation",  # fine-grained attack / decay micro-structure
    ],
    "groove": [
        "phase_12_wow_flutter_fix",  # Stufe 2: irregular motor speed is the direct physical
        # cause of timing instability and groove loss
        "phase_31_speed_pitch_correction",  # residual digital timing / pitch alignment
        "phase_61_groove_echo_cancellation",  # vinyl inner-groove echo imprints rhythmic ghost beats
        # that disrupt the temporal groove perception
    ],
    # P4 — Transparency / warmth / bass / separation
    "transparenz": [
        "phase_03_denoise",  # broadband noise masking — primary transparency barrier
        "phase_29_tape_hiss_reduction",  # carrier-specific noise as secondary layer
        "phase_02_hum_removal",  # 50/60 Hz hum masks fine spectral detail between harmonics
        "phase_23_spectral_repair",  # codec pre-echo / ringing artifacts
        "phase_61_groove_echo_cancellation",  # groove echo creates smeared micro-detail on vinyl
    ],
    "waerme": [
        "phase_04_eq_correction",  # Stufe 2: tonal balance 200–600 Hz is the physical foundation
        # of warmth — subtraktive EQ correction BEFORE additive saturation
        "phase_22_tape_saturation",  # Stufe 5: harmonic enrichment ON TOP of corrected tonal base
    ],
    "bass_kraft": [
        "phase_04_eq_correction",  # RIAA / EQ error in bass region — primary cause of
        # low-frequency power deficiency across all carriers
        "phase_26_dynamic_range_expansion",  # dynamic compression reduces bass transient authority;
        # expansion restores perceptible bass impact
    ],
    "separation_fidelity": [
        "phase_49_advanced_dereverb",  # reverb bleed between sources — most common cause;
        # WPE dereverb is the most effective remedy
        "phase_62_crosstalk_cancellation",  # Stufe 2: physical L/R channel bleed (tape heads,
        # vinyl cutting) — mechanical origin, broad applicability
        "phase_20_reverb_reduction",  # lighter reverb reduction for residual acoustic bleed
    ],
    # P5 — Brilliance / spatial
    "brillanz": [
        "phase_06_frequency_restoration",  # BW extension (AudioSR) — primary path for HF content
        # lost across all analog carriers
        "phase_23_spectral_repair",  # codec/pre-echo spectral smear recovery before harmonic lift
        "phase_07_harmonic_restoration",  # harmonic HF reconstruction for material where HF was
        # never captured (era guards handle 1900–1925 exclusion)
        "phase_39_air_band_enhancement",  # DSP air band (>12 kHz psychoacoustic shimmer) — safe
        # fallback when AudioSR/harmonic rollback leaves brillanz below floor;
        # §0a: NOT forbidden in Restoration (only phase_21/35/42 are forbidden)
    ],
    "spatial_depth": [
        "phase_46_spatial_enhancement",  # low spatial_depth = INSUFFICIENT spatial cues;
        # add inter-aural cues and stereo field width.
        # NEVER phase_49 as primary: dereverb REMOVES reverb
        # which REDUCES spatial depth — causal inversion!
        # §0p/§2.46e CONSTRAINT: phase_46 darf in Restoration NICHT bei Vokalmaterial
        # (panns_singing ≥ 0.25) injiziert werden — Delay-Reflexionen werden als Echo
        # wahrgenommen. §GOAL_BASELINE_CHECK und _select_phases() kennen dieses Constraint.
        # Bei Vocal-Sperre: Fallback auf phase_06 (HF/HRTF-Cues).
        "phase_06_frequency_restoration",  # HF > 8 kHz carries HRTF / air cues — the dominant
        # perceptual carrier of perceived spatial depth
    ],
    # P0 — Formant fidelity (§0p Vocal-Supremacy; activated when formant_fidelity < floor × 0.95)
    # Restoration uses EQ-based formant correction only — §0a FORBIDS phase_42 in Restoration.
    "formant_fidelity": [
        "phase_04_eq_correction",  # Stufe 2: spectral tilt correction shapes F1–F4 region (200–4000 Hz)
        "phase_16_final_eq",  # Stufe 6: perceptual rebalancing of formant energy distribution
    ],
    # P0 — Vocal Quality (VQI-Gate §0p; activated when VQI < material_floor × 0.95)
    # §0a: phase_42_vocal_enhancement VERBOTEN in Restoration.
    # Phase_65 = DSP-Korrektiv (Stufen: Spektral-Tilt + HNR-Blend + Formant-Tilt).
    "vocal_quality": [
        "phase_65_vocal_naturalness_restoration",  # §7.10 DSP-Korrektiv, Restoration-only
        "phase_03_denoise",  # wenn NR-Überstärkung Ursache ist
    ],
    # P2 — Transient-Energie (§1.4.6; Onset-Amplitude-Ratio nach subtraktiven Phasen)
    # PHASE_GOAL_EXCLUSIONS: phase_18 + phase_26 für transient_energie
    "transient_energie": [
        "phase_26_dynamic_range_expansion",  # §1.4.6 primär: Onset-Energie via Dynamikbearbeitung
        "phase_08_transient_preservation",  # sekundär: Transient-Erhalt als fallback
        "phase_23_spectral_repair",  # codec/pre-echo smear recovery fuer attack-klarheit
    ],
}

_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS: dict[str, list[str]] = {
    # Additional phases enabled in Studio 2026 (additive / enhancement).
    # §0a: phase_21/35/42 are Studio 2026 only — and only where goal-appropriate.
    # §2.36: phase_58_lyrics_guided_enhancement is Pflicht for vocal articulation.
    "timbre_authentizitaet": [
        "phase_42_vocal_enhancement",  # §0a Studio 2026 only: ML vocal timbre / formant correction
    ],
    "artikulation": [
        "phase_42_vocal_enhancement",  # ML vocal enhancement improves consonant clarity
        "phase_43_ml_deesser",  # removes sibilance that masks consonant articulation
        "phase_58_lyrics_guided_enhancement",  # §2.36 Pflicht: phoneme-level guided enhancement
    ],
    "brillanz": [
        "phase_07_harmonic_restoration",  # harmonic HF reconstruction (era guards apply)
        "phase_39_air_band_enhancement",  # air band > 12 kHz — dominant perceptual carrier of
        # brilliance in modern playback; Studio 2026 only
    ],
    "waerme": ["phase_37_bass_enhancement"],
    "bass_kraft": ["phase_37_bass_enhancement"],
    "spatial_depth": [
        "phase_48_stereo_width_enhancer",  # stereo field widening
        "phase_34_mid_side_processing",  # M/S matrix processing directly shapes spatial depth
    ],
    "emotionalitaet": ["phase_21_exciter"],
    "micro_dynamics": ["phase_36_transient_shaper"],
    # P0 — Formant fidelity Studio 2026: ML vocal enhancement corrects formant resonances
    # §0a: phase_42 is permitted in Studio 2026 ONLY.
    "formant_fidelity": ["phase_42_vocal_enhancement"],
    # P0 — Vocal Quality Studio 2026: ML vocal enhancement erlaubt (§0a Studio 2026 only)
    "vocal_quality": ["phase_42_vocal_enhancement"],
}


def get_goal_recovery_phases(
    goal: str,
    is_studio_2026: bool = False,
    transfer_chain: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Gibt ordered list of recovery phase IDs for a failing Musical Goal zurück.

    Used by UV3 §GOAL_BASELINE_CHECK (pre-pipeline) to add missing phases
    when a goal proxy score is below its material floor.

    Args:
        goal: canonical goal name (e.g. "brillanz", "natuerlichkeit").
              Aliases "raumtiefe", "mikrodynamik", "basskraft" are normalised.
        is_studio_2026: if True, also include Studio 2026 enhancement extras.
        transfer_chain: Optional transfer chain (e.g. ["cassette", "mp3_low"]).
            Used for chain-aware codec recovery prioritisation.

    Returns:
        Deduplicated list of phase IDs, most effective first.
        Empty list for unknown goals (non-blocking).

    §0a guarantee: §0a-forbidden phases (phase_21, phase_35, phase_42) are
    never returned when is_studio_2026=False.
    """
    # Normalise common aliases to canonical key
    _alias_map: dict[str, str] = {
        "raumtiefe": "spatial_depth",
        "mikrodynamik": "micro_dynamics",
        "basskraft": "bass_kraft",
        "tonalcenter": "tonal_center",
    }
    goal_key = _alias_map.get(str(goal or "").strip().lower(), str(goal or "").strip().lower())

    phases: list[str] = list(_GOAL_TO_RECOVERY_PHASES_RESTORATION.get(goal_key, []))

    # Chain-aware codec recovery: if the chain ends in lossy coding, pre-echo/smear repair
    # should be considered for HF/detail-sensitive goals before additive HF synthesis.
    _chain_norm = [str(s).strip().lower() for s in (transfer_chain or []) if str(s).strip()]
    _lossy_chain = bool(_chain_norm and _chain_norm[-1] in {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"})
    if _lossy_chain and goal_key in {"brillanz", "transient_energie", "transparenz", "artikulation"}:
        if "phase_50_spectral_repair" not in phases:
            if "phase_06_frequency_restoration" in phases:
                _i06 = phases.index("phase_06_frequency_restoration")
                phases.insert(_i06 + 1, "phase_50_spectral_repair")
            elif goal_key == "transient_energie" and "phase_08_transient_preservation" in phases:
                _i08 = phases.index("phase_08_transient_preservation")
                phases.insert(_i08 + 1, "phase_50_spectral_repair")
            else:
                phases.insert(0, "phase_50_spectral_repair")

    if is_studio_2026:
        for _p in _GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS.get(goal_key, []):
            if _p not in phases:
                phases.append(_p)
    return phases


__all__ = [
    "blend_targets_with_confidence",
    "compute_cpb",
    "compute_export_reliability",
    "compute_goal_coverage_index",
    "compute_ibs",
    "compute_recovery_pressure_index",
    "compute_reference_confidence",
    "compute_retry_temperature",
    "compute_tcci",
    "estimate_chain_end_goal_ceiling",
    "estimate_song_goal_targets",
    "get_effective_material_floor",
    "get_goal_recovery_phases",
    "get_material_floor",
    "get_phase_strength_range",
    "predict_quality_score",
    "resolve_effective_goal_targets",
    "RESTORABILITY_SCALE_MIN",
]
