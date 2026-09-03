"""
Aurik 10.0.0 — Phase-Typ-Ontologie §2.48a [RELEASE_MUST]
=====================================================
Zentrales Register für Phase-Operationstypen.

Architektur-Prinzip (Inversion gegenüber Ausnahmelisten):
  Jede Guard-Regel fragt: "Ist meine Voraussetzung für diese Phase ERFÜLLT?"
  — nicht: "Steht diese Phase in meiner Ausnahmeliste?"

Wissenschaftliche Grundlage:
  Jede Phase hat einen intrinsischen Operationstyp, der bestimmt, welche
  Qualitäts-Messungen valide sind:

  SUBTRACTIVE  — entfernt Signalkomponenten (Rauschen, Artefakte, Nachhall).
                 Residual = entfernter Defekt. Rauschtextur-Check valide.
                 Pre-Echo-Detektor NICHT valide (Residual ≠ Prä-Transient-Energie).
                 Metrik-Baseline ist defekt-inflationiert → Capping pflicht.
                 Quelle: ITU-R BS.1387 § 4.2 — Spektralsubtraktionsresidual ist kein
                 Artefakt, sondern entfernte Störung.

  ADDITIVE     — fügt neue Signalkomponenten hinzu (Obertöne, Bandbreite, Raum).
                 Pre-Echo-Detektor NICHT valide (keine STFT-Quantisierungsstruktur).
                 Rauschtextur-Check NICHT valide (Spektral-Tilt ändert sich intentional).
                 GDD-Check NICHT valide: Additive Synthese erzeugt keine STFT-Phasenfehler.
                 Quelle: Schwarz & Grill (2004), SBR — BW-Erweiterung verändert
                 Spektral-Tilt intentional; kein Artefakt.

  CORRECTIVE   — korrigiert zeitliche oder spektrale Eigenschaften ohne Energiegewinn/-verlust
                 (EQ, Azimuth, Wow/Flutter, DC-Offset, Phasenkorrektur).
                 Pre-Echo NICHT valide (Timing-Korrekturen ändern Prä-Attack-Energie).
                 Rauschtextur-Check NICHT valide (EQ verändert Geräuschform intentional).
                 GDD-Check prinzipiell valide, aber nur für IIR-Kaskaden (nicht STFT).
                 Quelle: Zölzer 2008, Digital Audio Signal Processing §4.

  ML_GENERATIVE — ML-Diffusion/Flow-Matching erzeugt neuen Inhalt (kein STFT-kohärenter
                 Ausgang). GDD-Check strukturell invalide (Diffusionsausgang ist nicht
                 STFT-phasenkohärent zum Eingang).
                 Quelle: Richter et al. (SGMSE+, TASLP 2022) — Score-basierter
                 Diffusionsprozess löst die STFT-Phasenkontinuität bewusst auf.

  DYNAMICS     — verändert die zeitliche Dynamik-Hüllkurve (Kompressor, Gate, Limiter,
                 Expander). Metriken für Groove, MikroDynamik, Emotionalität werden durch
                 intentionale Hüllkurven-Änderung systematisch verfälscht.
                 Quelle: Brecht De Man et al. (2013) — Dynamikverarbeitung und
                 Lautheitsnormierung erzeugen systembedingte Metrik-Artefakte.

  ANALYSIS_ONLY — erzeugt keine Audio-Änderung (Sematic-Analysis, Profiling).
                  Alle Checks überflüssig.

  ENHANCEMENT  — kombiniert mehrere Typen; default für Phasen ohne speziellen Charakter.

Reference: Spec 02 §2.48a (v10.0.0)
"""

from __future__ import annotations

from enum import Enum, auto


class PhaseOperationType(Enum):
    """Intrinsischer Operationstyp einer Phase.

    Bestimmt, welche Guards/Checks valide Ergebnisse liefern können.
    """

    SUBTRACTIVE = auto()  # Rauschen/Artefakt-Entfernung
    ADDITIVE = auto()  # Bandbreiten-/Harmonik-Erweiterung
    CORRECTIVE = auto()  # EQ, Azimuth, Timing-Korrektur
    ML_GENERATIVE = auto()  # Diffusion/Flow-Matching (kein STFT-kohärenter Ausgang)
    DYNAMICS = auto()  # Hüllkurven-Verarbeitung (Kompressor, Gate, Limiter)
    ANALYSIS_ONLY = auto()  # Keine Audio-Änderung
    ENHANCEMENT = auto()  # Mix / nicht eindeutig klassifizierbar


# ── Normatives Phase-Typ-Register ──────────────────────────────────────────
#
# Schlüssel: Phase-ID-Präfix (startswith-kompatibel).
# Neue Phasen müssen hier eingetragen werden — sonst greift ENHANCEMENT als Default.
# INVARIANTE: Dieses Register IST die Wahrheit. Ausnahmelisten in Guards und PMGG
# MÜSSEN sich von diesem Register ableiten und dürfen es nicht widersprechen.
#
PHASE_TYPE_REGISTRY: dict[str, PhaseOperationType] = {
    # ── Subtraktiv ─────────────────────────────────────────────────────────
    "phase_01": PhaseOperationType.SUBTRACTIVE,  # Click removal
    "phase_02": PhaseOperationType.SUBTRACTIVE,  # Hum removal (Notch-Filter)
    "phase_03": PhaseOperationType.SUBTRACTIVE,  # Broadband denoise
    "phase_05": PhaseOperationType.SUBTRACTIVE,  # Rumble filter
    "phase_09": PhaseOperationType.SUBTRACTIVE,  # Crackle removal (BANQUET)
    "phase_18": PhaseOperationType.SUBTRACTIVE,  # Noise gate
    "phase_20": PhaseOperationType.SUBTRACTIVE,  # Reverb reduction (SGMSE+ primär)
    "phase_24": PhaseOperationType.SUBTRACTIVE,  # Dropout repair (Interpolation → subtraktiv-dominant)
    "phase_27": PhaseOperationType.SUBTRACTIVE,  # Click/pop removal (2. Pass)
    "phase_28": PhaseOperationType.SUBTRACTIVE,  # Surface noise profiling (Vinyl)
    "phase_29": PhaseOperationType.SUBTRACTIVE,  # Tape hiss reduction (DeepFilterNet)
    "phase_49": PhaseOperationType.SUBTRACTIVE,  # Advanced dereverb (WPE)
    "phase_50": PhaseOperationType.SUBTRACTIVE,  # Spectral repair (STFT bin interpolation)
    "phase_57": PhaseOperationType.SUBTRACTIVE,  # Print-through reduction (LMS)
    "phase_59": PhaseOperationType.SUBTRACTIVE,  # Modulation noise reduction
    "phase_61": PhaseOperationType.SUBTRACTIVE,  # Groove echo cancellation
    "phase_62": PhaseOperationType.SUBTRACTIVE,  # Crosstalk cancellation
    "phase_63": PhaseOperationType.SUBTRACTIVE,  # Intermodulation reduction
    # ── Additiv ────────────────────────────────────────────────────────────
    "phase_06": PhaseOperationType.ADDITIVE,  # Bandwidth extension / Shelving EQ
    "phase_07": PhaseOperationType.ADDITIVE,  # Harmonic restoration (H2-H4)
    "phase_21": PhaseOperationType.ADDITIVE,  # Harmonic exciter
    "phase_22": PhaseOperationType.ADDITIVE,  # Tape saturation (tanh soft-sat)
    "phase_23": PhaseOperationType.CORRECTIVE,  # Spectral inpainting (AudioSR gap-fill / MRSA DSP repair); §4.7c POCS n_iter=2–5 vor PGHI
    "phase_37": PhaseOperationType.ADDITIVE,  # Bass enhancement
    "phase_38": PhaseOperationType.ADDITIVE,  # Presence boost (Bell EQ)
    "phase_39": PhaseOperationType.ADDITIVE,  # Air band enhancement
    "phase_44": PhaseOperationType.ADDITIVE,  # Guitar enhancement
    "phase_45": PhaseOperationType.ADDITIVE,  # Brass enhancement
    "phase_56": PhaseOperationType.ADDITIVE,  # Spectral band gap repair (HEAD_WEAR)
    # ── Korrektur ──────────────────────────────────────────────────────────
    "phase_04": PhaseOperationType.CORRECTIVE,  # Parametric EQ correction
    "phase_12": PhaseOperationType.CORRECTIVE,  # Wow/flutter correction (pitch)
    "phase_14": PhaseOperationType.CORRECTIVE,  # Phase correction (all-pass)
    "phase_15": PhaseOperationType.CORRECTIVE,  # Stereo balance L/R
    "phase_16": PhaseOperationType.CORRECTIVE,  # Final EQ trim
    "phase_25": PhaseOperationType.CORRECTIVE,  # Azimuth correction (fractional-delay)
    "phase_30": PhaseOperationType.CORRECTIVE,  # DC offset removal
    "phase_31": PhaseOperationType.CORRECTIVE,  # Speed/pitch correction (PSOLA)
    "phase_41": PhaseOperationType.CORRECTIVE,  # Output format optimization
    "phase_60": PhaseOperationType.CORRECTIVE,  # Inner groove distortion repair
    # ── ML-Generativ ──────────────────────────────────────────────────────
    "phase_55": PhaseOperationType.ML_GENERATIVE,  # Diffusion inpainting (CQTdiff+/FlowMatching)
    "phase_42": PhaseOperationType.ML_GENERATIVE,  # Vocal enhancement (BSRoFormer + ML synthesis)
    "phase_36": PhaseOperationType.ML_GENERATIVE,  # Transient shaper (ML-assisted)
    "phase_64": PhaseOperationType.ML_GENERATIVE,  # Tape splice repair (ML reconstruction)
    # ── Dynamik ───────────────────────────────────────────────────────────
    "phase_08": PhaseOperationType.DYNAMICS,  # Transient preservation (TDP/HPSS)
    "phase_10": PhaseOperationType.DYNAMICS,  # Compression (multiband parallel)
    "phase_11": PhaseOperationType.DYNAMICS,  # Limiting (4-band brick-wall)
    "phase_17": PhaseOperationType.DYNAMICS,  # Mastering polish (multiband comp+EQ)
    "phase_19": PhaseOperationType.DYNAMICS,  # De-esser (sibilant attenuation)
    "phase_26": PhaseOperationType.DYNAMICS,  # Dynamic range expansion
    "phase_33": PhaseOperationType.DYNAMICS,  # Stereo width limiter (M/S Side-compression)
    "phase_34": PhaseOperationType.DYNAMICS,  # Mid/Side processing (4-band)
    "phase_35": PhaseOperationType.DYNAMICS,  # Multiband compression (transparent)
    "phase_40": PhaseOperationType.DYNAMICS,  # Loudness normalization (LUFS ITU-R BS.1770-5)
    "phase_43": PhaseOperationType.DYNAMICS,  # ML de-esser (MP-SENet)
    "phase_47": PhaseOperationType.DYNAMICS,  # TruePeak limiter
    "phase_51": PhaseOperationType.DYNAMICS,  # Drums enhancement (compression+transient)
    "phase_52": PhaseOperationType.DYNAMICS,  # Piano restoration (expansion+resonance)
    "phase_54": PhaseOperationType.DYNAMICS,  # Transparent dynamics (psychoacoustic comp)
    # ── Analysis-Only ──────────────────────────────────────────────────────
    "phase_53": PhaseOperationType.ANALYSIS_ONLY,  # Semantic audio (BPM, key, genre-hint)
    # ── Enhancement (Mix) ─────────────────────────────────────────────────
    "phase_13": PhaseOperationType.ENHANCEMENT,  # Stereo enhancement (Haas+M/S)
    "phase_32": PhaseOperationType.ENHANCEMENT,  # Mono-to-stereo (Schroeder)
    "phase_46": PhaseOperationType.ENHANCEMENT,  # Spatial enhancement (cross-feed)
    "phase_48": PhaseOperationType.ENHANCEMENT,  # Stereo width enhancer (STFT M/S)
    "phase_58": PhaseOperationType.ENHANCEMENT,  # Lyrics-guided enhancement (phonem-DSP)
}
# phase_24 erscheint zweimal (SUBTRACTIVE und ENHANCEMENT) — letzter Eintrag gewinnt.
# AudioSR-Dropout-Repair ist dominant interpolativ (subtraktiv-artig), nicht generativ.
# Explizit als SUBTRACTIVE behalten:
PHASE_TYPE_REGISTRY["phase_24"] = PhaseOperationType.SUBTRACTIVE


def get_phase_type(phase_id: str) -> PhaseOperationType:
    """Gibt den normativen Operationstyp einer Phase zurück.

    Matching via startswith — robust gegen Suffix-Varianten (z.B. 'phase_03_denoise').
    Fallback: ENHANCEMENT (konservativ — keine Guard-Exemption).
    """
    for prefix, ptype in PHASE_TYPE_REGISTRY.items():
        if phase_id.startswith(prefix):
            return ptype
    return PhaseOperationType.ENHANCEMENT


# ── Guard-Applicability-Matrix ─────────────────────────────────────────────
#
# Für jeden Guard-Typ: Welche PhaseOperationTypes liefern VALIDE Messungen?
# INVALIDE → Guard darf NICHT feuern (würde perceptuell korrekte Ergebnisse rollbacken).
#
# Wissenschaftliche Begründung pro Zeile: siehe Modul-Docstring.


# §2.49 Noise-Texture-Check: Nur valide für SUBTRACTIVE (Rauschresidual messbar)
NOISE_TEXTURE_VALID_TYPES: frozenset[PhaseOperationType] = frozenset(
    {
        PhaseOperationType.SUBTRACTIVE,
    }
)

# §2.49 Pre-Echo-Detektor: Valide nur für STFT+Quantisierungs-Artefakte.
# In Aurik: nur DYNAMICS-Phasen mit STFT-Output können Pre-Echo erzeugen (Kompressor-Overshoot).
# SUBTRACTIVE (Residual ≠ Prä-Transient), ADDITIVE (Synthese, keine MDCT-Quantisierung),
# CORRECTIVE (Timing-Korrektur ändert Prä-Attack), ML_GENERATIVE (Diffusion): alle invalide.
PRE_ECHO_VALID_TYPES: frozenset[PhaseOperationType] = frozenset(
    {
        PhaseOperationType.DYNAMICS,  # Kompressor-Attack-Overshoot möglich
        PhaseOperationType.ENHANCEMENT,  # Mix-Typen können Pre-Echo einführen
    }
)

# §2.48 GDD-Check (STFT-Gruppenlaufzeit): Valide nur für DSP-STFT-Verarbeitung.
# ML_GENERATIVE: SGMSE+ Diffusionsausgang nicht STFT-phasenkohärent (Richter 2022).
# ADDITIVE: Synthese erzeugt keine Phasenfehler — neue Bins haben eigene Phase.
# CORRECTIVE (All-pass/Delay): GDD definitionsgemäß 0 für linearphasige FIR; nur IIR relevant.
GDD_VALID_TYPES: frozenset[PhaseOperationType] = frozenset(
    {
        PhaseOperationType.SUBTRACTIVE,  # STFT-Spektralsubtraktion kann Phasen verzerren
        PhaseOperationType.DYNAMICS,  # STFT-basierte Dynamikverarbeitung
        PhaseOperationType.ENHANCEMENT,
    }
)

# §2.29c Baseline-Capping: Nur valide für SUBTRACTIVE (defekt-inflationierte Baseline).
# ADDITIVE/CORRECTIVE: Baseline-Werte sind real (keine Defekt-Inflation).
BASELINE_CAPPING_VALID_TYPES: frozenset[PhaseOperationType] = frozenset(
    {
        PhaseOperationType.SUBTRACTIVE,
    }
)

# §2.48 P1/P2-Drift-Check: Valide für alle Typen außer ANALYSIS_ONLY.
# (Audio unverändert → Drift trivial 0.0.)
# ABER: P1/P2-Goals mit defekt-inflationierter Baseline → Capping zuerst anwenden.
P1P2_DRIFT_CHECK_INVALID_TYPES: frozenset[PhaseOperationType] = frozenset(
    {
        PhaseOperationType.ANALYSIS_ONLY,
    }
)

# ── §2.29e Phasen-Konflikt-Register ───────────────────────────────────────
#
# Explizite Paare, bei denen Phase B die Arbeit von Phase A NICHT neutralisieren darf:
#   key   = abgeschlossene Phase (hat etwas repariert)
#   value = Folgephasen, die das Ergebnis nicht revertieren dürfen
#
# Semantik: „phase_09 hat Crackle entfernt → phase_50 darf diese Bins nicht als
# Codec-Spikes einstufen und erneut löschen."
#
# Matching via startswith in get_conflict_phases() — phase_09_crackle → trifft „phase_09".
# Invariante: CONFLICT_REGISTRY definiert, was UV3 als conflict_with_prior_phases injiziert;
# die Phase selbst entscheidet, wie sie damit umgeht (conservative processing, skip, log).
#
CONFLICT_REGISTRY: dict[str, frozenset[str]] = {
    # Crackle/Click repariert → Spectral-Repair darf reparierte Bins nicht als Spikes sehen
    "phase_09": frozenset({"phase_50"}),
    "phase_01": frozenset({"phase_50", "phase_27"}),
    # Harmonik restauriert → Denoise/Spectral-Repair darf neue Obertöne nicht entfernen
    "phase_07": frozenset({"phase_50", "phase_03", "phase_29"}),
    # Bandbreite erweitert → HF-entfernende Phasen dürfen HF-Extension nicht rückgängig machen
    "phase_06": frozenset({"phase_28", "phase_29", "phase_50"}),
    # Spektrales Inpainting abgeschlossen → Denoise darf den Inhalt nicht nochmals entfernen
    "phase_23": frozenset({"phase_03", "phase_29"}),
    # Diffusions-Inpainting → Denoise darf halluzinierten Inhalt nicht entfernen
    "phase_55": frozenset({"phase_03", "phase_29"}),
    # Dropout repariert → Spectral-Repair + Denoise dürfen frisch interpolierten Inhalt nicht entfernen
    "phase_24": frozenset({"phase_50", "phase_03", "phase_29"}),
    # Bandlücke repariert (HEAD_WEAR) → Denoise darf restauriertes Band nicht re-entrauschen
    "phase_56": frozenset({"phase_29", "phase_03"}),
    # Broadband-Denoise → nachfolgende Harmonic/Freq-Restaurierung konservativ (§2.46 Invariante:
    # subtraktiv vor additiv; eine zweite Denoise-Runde nach Harmonik-Enhancement ist ein Konflikt)
    "phase_03": frozenset({"phase_29"}),
    # §v10.62 Exciter synthetisiert Harmonische → Denoiser entfernt sie als "Rauschen"
    # (F10: stumme Auslöschung — Phase_21-Arbeit wird von Phase_03/29 komplett vernichtet)
    "phase_21": frozenset({"phase_03", "phase_29"}),
    # §v10.62 Air-Band-Enhancement addiert 10–20 kHz → Tape-Hiss-Removal entfernt es
    # (F11: synthetisierte Luft wird von Phase_29 als "Hiss" erkannt und gelöscht)
    "phase_39": frozenset({"phase_29"}),
    # §v10.62 Bass-Enhancement addiert Subharmonische → Rumble-Filter könnte sie entfernen
    "phase_37": frozenset({"phase_05"}),
}


def get_conflict_phases(completed_phase_id: str) -> frozenset[str]:
    """Gibt phase IDs that should behave conservatively after completed_phase_id ran zurück.

    Uses prefix matching — robust against suffix variants (e.g. 'phase_09_crackle').
    Returns empty frozenset when no conflict is registered.

    Args:
        completed_phase_id: ID of the phase that already ran and produced content.

    Returns:
        frozenset of phase ID prefixes that should receive 'conflict_with_prior_phases'.
    """
    for _prefix, _conflicts in CONFLICT_REGISTRY.items():
        if completed_phase_id.startswith(_prefix):
            return _conflicts
    return frozenset()
