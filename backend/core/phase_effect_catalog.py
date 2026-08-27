"""§2.60 Denker-Intelligenz: PhaseEffectProfile — Wissen was Phasen bewirken.

Damit der PhaseInteractionDenker Intensitäten PROAKTIV kalibrieren kann
(statt nur PMGG-rollback REACTIV), braucht er ein Modell jeder Phase:
  - Welche Musical Goals werden beeinflusst?
  - In welche Richtung? (boost/dampen)
  - Wie stark ist der typische Effekt?
  - Welche Risiken gibt es? (vocal_distortion, transient_smearing, etc.)
  - Welche Vorbedingungen braucht die Phase?

Die Profile werden vom Denker mit dem aktuellen Audio-Zustand (SNR, Panns,
Bandbreite, Defekte) kombiniert und ergeben eine kalibrierte Intensität.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseEffectProfile:
    """Beschreibt die Wirkung einer Phase auf Musical Goals und ihre Risiken."""

    phase_id: str
    # Goal-Impact: goal_name → typical_delta (positiv = verbessert, negativ = verschlechtert)
    goal_impact: dict[str, float] = field(default_factory=dict)
    # Risiko-Typen (wenn diese Bedingungen vorliegen, Intensität reduzieren)
    risks: list[str] = field(default_factory=list)
    # Vorbedingungen für optimale Wirkung
    preconditions: dict[str, Any] = field(default_factory=dict)
    # Maximal sichere Stärke (0-1) für verschiedene Materialien
    max_strength_by_material: dict[str, float] = field(default_factory=dict)
    # Zeitaufwand-Kategorie: "fast" (<5s), "medium" (5-30s), "slow" (30-120s), "heavy" (>120s)
    time_profile: str = "medium"
    # Minimale Defekt-Schwere damit Phase sinnvoll ist (0-1)
    min_severity: float = 0.0
    # Default-Stärke (1.0 = volle Stärke, vom Joint-Calibrator überschreibbar)
    base_strength: float = 1.0
    # Kommentar für Debugging
    note: str = ""


# ── §2.60 Phase-Wissensbasis ──────────────────────────────────────────────

PHASE_EFFECT_CATALOG: dict[str, PhaseEffectProfile] = {
    # ── Defekt-Entfernung ──────────────────────────────────────────
    "phase_01_click_removal": PhaseEffectProfile(
        phase_id="phase_01_click_removal",
        goal_impact={
            "transparenz": +0.03,
            "artikulation": +0.02,
            "natuerlichkeit": +0.01,
        },
        risks=["transient_smearing"],  # Zu aggressiv → Ansätze verschmiert
        preconditions={"click_density": "> 100/s"},
        max_strength_by_material={"shellac": 1.0, "vinyl": 0.9, "tape": 0.6, "cd_digital": 0.3},
        time_profile="fast",
        min_severity=0.2,
        note="Median-Filter; bei zu hoher Stärke werden Transienten verschmiert",
    ),
    "phase_02_hum_removal": PhaseEffectProfile(
        phase_id="phase_02_hum_removal",
        goal_impact={
            "transparenz": +0.02,
            "natuerlichkeit": +0.02,
            "waerme": -0.01,  # Notch kann Wärme minimal reduzieren
        },
        risks=["bass_loss"],
        preconditions={"hum_energy_db": "> -50"},
        max_strength_by_material={"vinyl": 1.0, "tape": 0.8, "cd_digital": 0.3},
        time_profile="fast",
        min_severity=0.1,
        note="IIR-Notch 50/60Hz+Harmonische; sehr gezielt, kaum Kollateralschaden",
    ),
    # ── Rauschunterdrückung ──────────────────────────────────────
    "phase_03_denoise": PhaseEffectProfile(
        phase_id="phase_03_denoise",
        goal_impact={
            "transparenz": +0.06,
            "artikulation": +0.04,
            "waerme": -0.03,  # ML kann Wärme reduzieren
            "natuerlichkeit": -0.02,  # ML kann künstlich klingen
            "emotionalitaet": -0.02,
        },
        risks=["vocal_distortion", "ml_artifact", "energy_loss"],
        preconditions={"snr_db": "< 20", "bypass_if": "snr_unknown AND vocal_heavy"},
        max_strength_by_material={"vinyl": 0.85, "tape": 0.90, "shellac": 0.95, "cd_digital": 0.40},
        time_profile="heavy",  # BS-RoFormer + MIIPHER + Resemble = 9+ Minuten!
        min_severity=0.3,
        note="Schwerste ML-Phase; Codec-Degradation (mp3_low/streaming/aac/minidisc) → MIIPHER-Sigma konservativ (0.25-0.40)",
    ),
    # ── Frequenz-Entzerrung ──────────────────────────────────────
    "phase_04_eq_correction": PhaseEffectProfile(
        phase_id="phase_04_eq_correction",
        goal_impact={
            "brillanz": +0.03,
            "waerme": +0.02,
            "natuerlichkeit": +0.01,
        },
        risks=["over_brightening"],
        preconditions={"bandwidth_hz": "< 15000"},
        max_strength_by_material={"vinyl": 0.8, "tape": 0.7, "shellac": 0.9},
        time_profile="fast",
        min_severity=0.2,
        note="Material-adaptive EQ; unkritisch, nur bei Bandbreitenverlust stark",
    ),
    # ── Wow/Flutter ─────────────────────────────────────────────
    "phase_12_wow_flutter_fix": PhaseEffectProfile(
        phase_id="phase_12_wow_flutter_fix",
        goal_impact={
            "tonal_center": +0.04,
            "emotionalitaet": +0.03,
            "waerme": +0.01,
        },
        risks=["pitch_artifact", "phase_distortion"],
        preconditions={"wow_severity": "> 0.3"},
        max_strength_by_material={"vinyl": 0.7, "tape": 0.9, "cassette": 1.0},
        time_profile="medium",
        min_severity=0.3,
        note="Polyphonic-Speed-Korrektur; bei vinyl konservativ (mechanisch, nicht elektrisch)",
    ),
    # ── Vocal/De-Esser ──────────────────────────────────────────
    "phase_19_de_esser": PhaseEffectProfile(
        phase_id="phase_19_de_esser",
        goal_impact={
            "artikulation": +0.04,
            "natuerlichkeit": +0.01,
            "brillanz": -0.01,  # kann HF marginal dämpfen
        },
        risks=["vocal_dulling", "gender_mismatch"],
        preconditions={"panns_singing": "> 0.20", "gender_detected": "valid"},
        max_strength_by_material={"vinyl": 0.85, "tape": 0.80, "cd_digital": 0.60},
        time_profile="medium",
        min_severity=0.1,
        note="Gender-abhängige Sibilanz-Bänder; female→6-10kHz, male→4-8kHz",
    ),
    # ── Reverb/Dereverb ──────────────────────────────────────────
    "phase_20_reverb_reduction": PhaseEffectProfile(
        phase_id="phase_20_reverb_reduction",
        goal_impact={
            "transparenz": +0.04,
            "artikulation": +0.03,
            "waerme": -0.02,  # Hall-Entfernung reduziert Wärme
            "spatial_depth": -0.03,  # Weniger Hall = weniger Raumtiefe
        },
        risks=["over_drying", "vocal_thinning"],
        preconditions={"rt60_s": "> 0.5"},
        max_strength_by_material={"vinyl": 0.6, "tape": 0.7, "cd_digital": 0.5},
        time_profile="medium",
        min_severity=0.2,
        note="DSP+DNN-Hybrid; bei church/broadcast cap durch RoomAcoustics",
    ),
    # ── Präsenz-Boost ───────────────────────────────────────────
    "phase_38_presence_boost": PhaseEffectProfile(
        phase_id="phase_38_presence_boost",
        goal_impact={
            "brillanz": +0.05,
            "artikulation": +0.03,
            "waerme": -0.01,
            "natuerlichkeit": -0.02,  # kann künstlich wirken
        },
        risks=["over_brightening", "vocal_harshness"],
        preconditions={"bandwidth_loss": "present"},
        max_strength_by_material={"vinyl": 0.7, "tape": 0.6, "cd_digital": 0.3},
        time_profile="fast",
        min_severity=0.3,
        note="HF-Anhebung; nur bei echten Bandbreitenverlust, nicht als Default-Enhancement",
    ),
    # ── §2.76: Bisher fehlende Phasen im Catalog → Joint-Calibrator kannte sie nicht
    "phase_57_print_through_reduction": PhaseEffectProfile(
        phase_id="phase_57_print_through_reduction",
        goal_impact={
            "transparenz": +0.05,
            "artikulation": +0.04,
            "natuerlichkeit": +0.03,
            "emotionalitaet": +0.02,
            "waerme": -0.01,  # kann leicht dünner klingen nach Echo-Entfernung
        },
        risks=["transient_smearing", "phase_artifact"],
        preconditions={"print_through": "> 0.3"},
        max_strength_by_material={"reel_tape": 0.95, "cassette": 0.85, "tape": 0.80, "vinyl": 0.0},
        time_profile="medium",
        min_severity=0.3,
        note="Bidirektionale LMS-Adaptive Subtraction; Pre+Post-Echo getrennt (Magnetband-Durchdruck)",
    ),
    "phase_63_intermodulation_reduction": PhaseEffectProfile(
        phase_id="phase_63_intermodulation_reduction",
        goal_impact={
            "transparenz": +0.05,
            "timbre_authentizitaet": +0.04,
            "natuerlichkeit": +0.03,
            "brillanz": -0.01,  # IMD oft in Höhen am stärksten
        },
        risks=["phase_distortion", "energy_loss"],
        preconditions={"intermodulation_distortion": "> 0.2"},
        max_strength_by_material={"vinyl": 0.8, "tape": 0.7, "cassette": 0.7, "cd_digital": 0.3},
        time_profile="heavy",
        min_severity=0.2,
        note="Volterra-basierte IMD-Tilgung; harmonische Verzerrungsprodukte entfernen",
    ),
    "phase_59_modulation_noise_reduction": PhaseEffectProfile(
        phase_id="phase_59_modulation_noise_reduction",
        goal_impact={
            "transparenz": +0.04,
            "natuerlichkeit": +0.04,
            "waerme": +0.02,
            "transient_energie": -0.01,
        },
        risks=["transient_smearing", "energy_loss"],
        preconditions={"modulation_noise": "> 0.2"},
        max_strength_by_material={"tape": 0.9, "reel_tape": 0.9, "cassette": 0.85, "vinyl": 0.0},
        time_profile="medium",
        min_severity=0.2,
        note="Rauschmodulations-Entfernung (signalabhängiges Rauschen auf Magnetband)",
    ),
}


# ── §2.60 Kalibrierungs-Logik ─────────────────────────────────────────────


def calibrate_phase_intensity(
    phase_id: str,
    base_strength: float,
    *,
    defect_severity: float = 0.0,
    material: str = "vinyl",
    panns_singing: float = 0.0,
    snr_db: float | None = None,
    rt60_s: float = 0.5,
    bandwidth_hz: float = 20000,
    era_decade: int = 1980,
    genre_is_schlager: bool = False,
    soft_saturation_preserve: bool = False,
    # ── §2.60 L1-Max: Zusätzliche Messwerte ─────────────────
    crest_db: float = 12.0,
    hf_ratio: float = 0.0,
    transient_ratio: float = 0.0,
    micro_dynamic_db: float = 6.0,
    rms_dbfs: float = -20.0,
    chain_has_cassette: bool = False,
    chain_has_mp3: bool = False,
    restorability: float = 0.5,
    pipeline_confidence: float = 0.75,
    defect_count_total: int = 0,
    terminal_codec: str | None = None,
    codec_avg_discount: float = 1.0,
) -> float:
    """§2.60: Kalibriert die Phasen-Intensität proaktiv.

    Nutzt das PhaseEffectProfile + Audio-Zustand um die optimale
    Intensität VOR der Ausführung zu berechnen. PMGG validiert danach.

    Returns: kalibrierte Stärke in [0.0, 1.0]
    """
    profile = PHASE_EFFECT_CATALOG.get(phase_id)
    if profile is None:
        return base_strength  # Kein Profil → Original-Stärke

    strength = float(base_strength)

    # 1. Defekt-Schwere-Skalierung
    if profile.min_severity > 0 and defect_severity < profile.min_severity:
        strength *= max(0.1, defect_severity / max(profile.min_severity, 0.01))

    # 2. Material-Cap
    mat_cap = profile.max_strength_by_material.get(material, 0.9)
    strength = min(strength, mat_cap)

    # 3. Risiko-basierte Reduktion
    for risk in profile.risks:
        if risk == "vocal_distortion" and panns_singing > 0.25:
            # Gesang im Signal → ML-Phasen konservativer
            vocal_factor = 1.0 - (panns_singing - 0.25) * 0.8  # 0.25→1.0, 0.35→0.92, 0.50→0.80
            strength *= max(0.4, vocal_factor)
        if risk == "energy_loss" and snr_db is not None and snr_db < 8:
            strength *= 0.6  # Sehr niedriger SNR → Energie-Verlust-Risiko
        if risk == "over_brightening" and soft_saturation_preserve:
            strength *= 0.5  # Sättigung erhalten → nicht zusätzlich aufhellen
        if risk == "over_drying" and rt60_s > 2.0:
            strength *= 0.7  # Sehr hallig → nicht zu viel Hall entfernen (war Aufnahme-Charakter)
        if risk == "vocal_dulling" and panns_singing > 0.3:
            strength *= 0.85  # De-Esser bei starkem Gesang etwas zurückhaltender
        if risk == "ml_artifact" and era_decade < 1980:
            strength *= 0.7  # Vintage-Material → ML-Artefakte wahrscheinlicher

    # 4. Genre-spezifische Anpassung
    if genre_is_schlager:
        if "waerme" in profile.goal_impact and profile.goal_impact.get("waerme", 0) < 0:
            strength *= 0.75  # Schlager braucht Wärme → Phasen die Wärme reduzieren drosseln
        if phase_id == "phase_20_reverb_reduction":
            strength *= 0.6  # Schlager-HALL ist erwünscht!

    # 5. SNR-Abhängigkeit für ML-Phasen
    if snr_db is not None and "ml_artifact" in profile.risks:
        if snr_db < 5:
            strength *= 0.5  # Sehr niedriger SNR → ML tut mehr Schaden als Nutzen
        elif snr_db > 15:
            strength = min(strength, 0.4)  # Hoher SNR → ML kaum nötig

    # 6. Crest-Faktor: hoher Crest → Transienten-Phasen verstärken
    if crest_db > 15.0 and "transient_smearing" in profile.risks:
        strength *= 0.6  # Ohnehin schon spitzig → nicht weiter schärfen

    # 7. HF-Anteil: viel HF → De-Esser/Brillanz-Phasen anpassen
    if hf_ratio > 0.15 and phase_id == "phase_19_de_esser":
        strength = min(strength, 1.0)  # Viel HF → De-Esser darf voll ran
    if hf_ratio < 0.02 and "over_brightening" in profile.risks:
        strength *= 0.3  # Kaum HF → Aufhell-Phasen kaum nötig, Risiko künstlich

    # 8. Mikrodynamik: flache Dynamik → Expansions-Phasen verstärken
    if micro_dynamic_db < 3.0 and phase_id == "phase_26_dynamic_range_expansion":
        strength = min(strength * 1.3, 1.0)  # Flach → mehr Expansion wagen

    # 9. Pegel: sehr leise Aufnahme → konservativer (Rauschen wird sonst hochgezogen)
    if rms_dbfs < -30.0 and "ml_artifact" in profile.risks:
        strength *= 0.5

    # 10. Transfer-Kette: MP3 in der Kette → ML-Phasen vorsichtiger
    if chain_has_mp3 and "ml_artifact" in profile.risks:
        strength *= 0.7  # MP3-Artefakte + ML = Gefahr

    # 13. §CODEC: Terminal-Codec-Kalibrierung — Denker entscheidet dynamisch
    # Je nach Codec-Typ werden analog-spezifische Phasen gedämpft,
    # weil ihre Defekt-Signatur durch Kompressionsartefakte maskiert ist.
    # Aber: Tape-Level-Dips (phase_12) und Kassetten-Hiss (phase_29) sind ECHT!
    if terminal_codec and codec_avg_discount < 0.90:
        _codec_factor = max(0.35, codec_avg_discount)
        # ML-Phasen: stärker dämpfen (MP3 + ML = doppeltes Risiko)
        if "ml_artifact" in profile.risks:
            strength *= _codec_factor
        # Analog-spezifische Phasen ohne echte Defekte: deutlich dämpfen
        if phase_id in (
            "phase_28_surface_noise_profiling",
            "phase_20_reverb_reduction",
            "phase_49_advanced_dereverb",
            "phase_60_inner_groove_distortion_repair",
        ):
            strength *= _codec_factor * 0.7
        # Wow/Flutter-Detektor: Codec-Artefakte → false positives, aber Tape-Dips sind real
        if phase_id == "phase_12_wow_flutter_fix":
            strength *= max(0.55, _codec_factor)  # Nicht unter 0.55 — Tape-Dips müssen leben

    # 11. Restorability: schlechte Ausgangslage → weniger invasive Eingriffe
    if restorability < 0.4:
        strength *= 0.7  # Ohnehin schwer → nicht zu viel riskieren

    # 12. Pipeline-Unsicherheit: unklare Diagnose → defensiver
    if pipeline_confidence < 0.7 and "ml_artifact" in profile.risks:
        strength *= 0.6  # Wenn unsicher, dann ML lieber weglassen

    # 13. Viele Defekte: zu viele Baustellen → nicht alle Phasen voll aufdrehen
    if defect_count_total > 40:
        strength *= 0.8  # Kaskadierende Phasen → jede etwas zurückhaltender

    return max(0.0, min(1.0, strength))


def get_phase_risk_level(phase_id: str, **audio_state) -> str:
    """Bewertet das Risiko-Level einer Phase im aktuellen Audio-Kontext.

    Returns: "low" | "medium" | "high" | "critical"
    """
    profile = PHASE_EFFECT_CATALOG.get(phase_id)
    if profile is None:
        return "low"

    risk_score = 0.0
    panns = float(audio_state.get("panns_singing", 0))
    snr = audio_state.get("snr_db")
    era = int(audio_state.get("era_decade", 1980))

    if "vocal_distortion" in profile.risks and panns > 0.25:
        risk_score += (panns - 0.25) * 3.0
    if "ml_artifact" in profile.risks and era < 1980:
        risk_score += 0.5
    if "ml_artifact" in profile.risks and (snr is None or snr < 8):
        risk_score += 1.0

    if risk_score > 2.0:
        return "critical"
    elif risk_score > 1.0:
        return "high"
    elif risk_score > 0.3:
        return "medium"
    return "low"


# ── §2.60 Katalog-Singleton ─────────────────────────────────────────────

_catalog_instance = None


def get_phase_effect_catalog():
    """Singleton-Zugriff auf den PhaseEffectCatalog."""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = _CatalogHelper()
    return _catalog_instance


class _CatalogHelper:
    """Hilfsklasse für calibrate_all()."""

    def calibrate_all(self, phase_ids: list[str], audio_ctx: dict) -> dict[str, float]:
        """Kalibriert alle gegebenen Phasen für einen Audio-Kontext."""
        result = {}
        for pid in phase_ids:
            profile = PHASE_EFFECT_CATALOG.get(pid)
            if profile is None:
                result[pid] = 1.0
                continue
            calibrated = calibrate_phase_intensity(
                pid,
                profile.base_strength,
                defect_severity=float(audio_ctx.get("defect_severity", 0)),
                material=str(audio_ctx.get("material_type", "vinyl")),
                panns_singing=float(audio_ctx.get("panns_singing", 0)),
                snr_db=audio_ctx.get("snr_db"),
                bandwidth_hz=float(audio_ctx.get("bandwidth_hz") or 20000),
                era_decade=int(audio_ctx.get("era_decade", 1980)),
                rt60_s=float(audio_ctx.get("rt60_s", 0.5)),
                crest_db=float(audio_ctx.get("crest_db", 12.0)),
                hf_ratio=float(audio_ctx.get("hf_ratio", 0.0)),
                transient_ratio=float(audio_ctx.get("transient_ratio", 0.0)),
                micro_dynamic_db=float(audio_ctx.get("micro_dynamic_db", 6.0)),
                rms_dbfs=float(audio_ctx.get("rms_dbfs", -20.0)),
                chain_has_cassette=bool(audio_ctx.get("chain_has_cassette", False)),
                chain_has_mp3=bool(audio_ctx.get("chain_has_mp3", False)),
                restorability=float(audio_ctx.get("restorability", 0.5)),
                pipeline_confidence=float(audio_ctx.get("pipeline_confidence", 0.75)),
                defect_count_total=int(audio_ctx.get("defect_count_total", 0)),
                terminal_codec=audio_ctx.get("terminal_codec"),
                codec_avg_discount=float(audio_ctx.get("codec_avg_discount", 1.0)),
            )
            result[pid] = calibrated
        return result
