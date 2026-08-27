"""
spec_constitution.py — v10 Normative Spec-Konstitution
=========================================================

Zentrale, maschinenlesbare Konstitution aller Aurik-Spezifikationen.
Dient als SINGLE SOURCE OF TRUTH fuer Agent und Watchdog.

WAS HIER STEHT, GILT. Keine Ausnahmen, keine Overrides ohne Spec-Änderung.

Integration:
  from backend.core.spec_constitution import get_constitution
  const = get_constitution()
  const.check_artifact_freedom(audio, sr)     → (bool, str)
  const.get_forbidden_patterns()               → list[ForbiddenPattern]
  const.get_musical_goal_thresholds(material)  → dict[str, float]
  const.validate_against_paragraph_zero(audio) → list[str]

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.defect_scanner import MaterialType  # §v10.113

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# §0 — Oberstes Prinzip: Klangwahrheit
# ═══════════════════════════════════════════════════════════════════════════════

KLANGWAHRHEIT = """
Das Ziel jeder Restaurierung ist, dass der Hörer die Augen schließt und
die originale Performance hört — nicht eine technisch korrekte
Signalverarbeitung, und nicht eine 'verbesserte' Version.
"""

PRIMUM_NON_NOCERE = "Füge dem Klang keinen Schaden zu."
MINIMAL_INTERVENTION = "Greife nur ein, wo der Defekt hörbar ist."
PERCEPTUAL_IMPROVEMENT = "Der Export muss näher am Original-Klang liegen als der degradierte Input."

# §0h Music-Death-Shield
MUSIC_DEATH_SHIELD = {
    "artifact_freedom_min": 0.95,
    "hpi_min": 0.0,
    "vqi_recovery_trigger": 0.72,  # §0p: panns_singing >= 0.35 → Vokal-Vorrang
    "vqi_restoration_target": 0.82,
    "vqi_studio2026_target": 0.87,
    "oqs_min": 80,
    "timbral_fidelity_min": 0.93,
}

# §0i Perceptual Transparency Guarantee
PERCEPTUAL_TRANSPARENCY = {
    "musical_noise_max_above_carrier_db": 0.0,
    "frisson_zones_preserved": True,
}

# §0k Maximum-Achievable-Score
MAS_PRINCIPLE = "Jeder Song wird unter Berücksichtigung der Quelldatei und der physikalischen Grenzen bis zum maximal möglichen Ergebnis restauriert."

# §0m Maximal-Ausbaustufe Defektintelligenz
MAX_DEFECT_INTELLIGENCE = "Beide Modi (restoration + studio2026) immer auf maximaler Ausbaustufe für Defekterkennung, -Differenzierung und -Dosierung."

# §0p Primus-inter-Pares (Vocals first)
PRIMUS_INTER_PARES = "Wenn panns_singing >= 0.25, erhält Stimmqualität Vorrang vor allen anderen Zielen."

# ═══════════════════════════════════════════════════════════════════════════════
# FORBIDDEN — VERBOTEN-Regeln (normativ, aus .github/VERBOTEN.md)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RequiredPattern:
    """Eine GEBOTEN-Regel — was MUSS vorhanden sein."""

    id: str  # G01-G40
    category: str  # Logging, Audio, Pipeline, Test
    pattern: str  # Erkennungsmuster (muss vorhanden sein)
    description: str  # Was verlangt wird
    severity: str  # critical / warning


@dataclass
class ForbiddenPattern:
    """Eine VERBOTEN-Regel mit Erkennungsmuster und Begründung."""

    id: str  # V01-V58
    category: str  # Logging, API, DSP, Guard, ...
    pattern: str  # Erkennungsmuster (für AST/Linter)
    description: str  # Was verboten ist
    correct: str  # Was stattdessen zu tun ist
    severity: str  # critical / warning


FORBIDDEN_PATTERNS: list[ForbiddenPattern] = [
    # ── Kritische Architektur-Regeln ──
    ForbiddenPattern(
        "V01", "Logging", "print(", "print()-Aufrufe in Produktionscode", "logger.info/warning/error()", "critical"
    ),
    ForbiddenPattern("V02", "API", "return dict", "dict als API-Returnwert", "@dataclass", "warning"),
    ForbiddenPattern(
        "V03", "Cache", "_cache = {}", "Ungeschützter Module-Level-Cache", "threading.Lock() + Dict", "critical"
    ),
    ForbiddenPattern(
        "V04",
        "Guard",
        "gate_dbfs=-36.0",
        "Fester Gate-Schwellwert ohne reference_for_gate",
        "compute_signal_relative_gate_dbfs(ref)",
        "critical",
    ),
    ForbiddenPattern(
        "V05",
        "GPU",
        'map_location="cuda"',
        "GPU-Zugriff ohne ml_device_manager",
        "get_torch_device('PluginName')",
        "critical",
    ),
    ForbiddenPattern(
        "V06",
        "Audio",
        "sf.read|librosa.load",
        "Direkter Audio-Import ohne load_audio_file()",
        "load_audio_file(filepath)",
        "critical",
    ),
    ForbiddenPattern(
        "V07",
        "DSP",
        "sosfilt(sos, audio)",
        "Kausaler Filter bei Signal-Addition → Phasenversatz",
        "sosfiltfilt(sos, audio) (zero-phase)",
        "critical",
    ),
    ForbiddenPattern(
        "V08", "Normalize", "RMS|peak.*normali", "RMS/Peak-Normalisierung", "LUFS ITU-R BS.1770-5", "warning"
    ),
    ForbiddenPattern("V09", "Metric", "pesq|dnsmos|nisqa", "Veraltete Metriken", "PQS-MOS, VERSA, SingMOS", "warning"),
    ForbiddenPattern(
        "V10",
        "SongCal",
        "clip.*0\\.0.*2\\.0",
        "global_scalar außerhalb [0.50, 1.50]",
        "global_scalar ∈ [0.50,1.50], family ∈ [0.30,1.80]",
        "critical",
    ),
    ForbiddenPattern(
        "V11",
        "Guard",
        "MAX_DRIFT = -0\\.05|regression > 0\\.02",
        "Feste Guard-Schwellwerte",
        "compute_adaptive_drift_tolerance()",
        "warning",
    ),
    ForbiddenPattern(
        "V12",
        "Phase",
        "Phase_21|Phase_35|Phase_42.*CAUSE_TO_PHASES",
        "Phase 21/35/42 in Restoration-CAUSE_TO_PHASES",
        "Diese Phasen NIE für Restoration-Cause einplanen",
        "critical",
    ),
    ForbiddenPattern(
        "V13",
        "DC",
        "np\\.mean.*subtract|lfilter.*DC",
        "DC-Offset via np.mean/lfilter bei reel_tape",
        "filtfilt([1,-1],[1,-0.9995]) zero-phase",
        "warning",
    ),
    ForbiddenPattern(
        "V14",
        "Import",
        "from Aurik10.*import.*backend",
        "Backend-Import aus Frontend (Aurik10)",
        "Bridge-API via backend/api/bridge.py",
        "critical",
    ),
    ForbiddenPattern(
        "V15",
        "Gain",
        "audio \\*= gain_factor",
        "Uniformer Gain (auch auf Stille)",
        "_musical_gain_envelope()",
        "critical",
    ),
    ForbiddenPattern(
        "V16",
        "Limiter",
        "soft.limit|hard.limit.*routin",
        "Routine-Limiter ohne Peak-Check",
        "NUR wenn peak > 0.98",
        "warning",
    ),
    ForbiddenPattern(
        "V17",
        "Phase_skip",
        "severity.*<.*0\\.05.*skip",
        "Phase-Skip nur nach Severity",
        "_salience_adjusted_severity() Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V18",
        "ML",
        "except.*pass",
        "Stilles ML-Fallback ohne Logging",
        "logger.warning('ML->DSP fallback: ...')",
        "critical",
    ),
    ForbiddenPattern(
        "V19",
        "Noise",
        "noise.texture.*ohne.*V19",
        "Noise-Textur-Prüfung nach NR-Phase ohne V19-Check",
        "compute_noise_texture_distance()",
        "warning",
    ),
    ForbiddenPattern(
        "V20",
        "Correlation",
        "frame.energy.*ohne.*V20",
        "Mikrodynamik-Check nach NR ohne V20",
        "frame_energy_correlation() auf voiced-Zonen",
        "warning",
    ),
    # ── V10.0.7: Fortgeschrittene Regeln (V21-V42, 7 neue)
    ForbiddenPattern(
        "V21",
        "Noise",
        "noise.texture.*resynth",
        "Noise-Textur-Resynthese ohne Original-Referenz",
        "Noise-Textur aus Original extrahieren",
        "warning",
    ),
    ForbiddenPattern(
        "V33",
        "Material",
        "MaterialType.*missing|CASSETTE.*not in",
        "Neues MaterialType ohne vollstaendige dict-Eintraege",
        "Jedes dict[MaterialType,...] MUSS vollstaendig sein",
        "critical",
    ),
    ForbiddenPattern(
        "V38",
        "Strength",
        "uniform.*strength.*loop|strength.*for.*in.*events",
        "Einheitliche Strength fuer disparate Defekt-Events",
        "Per-Event-Strength-Oracle Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V39",
        "Phase",
        "phase_21.*CAUSE|phase_35.*CAUSE|phase_42.*CAUSE",
        "Phase 21/35/42 in Restoration-CAUSE_TO_PHASES",
        "Diese Phasen NIE fuer Restoration-Cause vorschlagen",
        "critical",
    ),
    ForbiddenPattern(
        "V40",
        "NR",
        "denoise.*ohne.*NMR|phase_03.*ohne.*feedback",
        "NR-Phase ohne NMR-Feedback",
        "compute_nmr_score() Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V41",
        "Masking",
        "additive.*ohne.*ForwardMasking",
        "Additive Phase ohne ForwardMaskingGuard",
        "ForwardMaskingGuard bei panns_singing>=0.25",
        "warning",
    ),
    ForbiddenPattern(
        "V42",
        "Roughness",
        "NR.*ohne.*roughness|phase_29.*ohne.*check",
        "NR-Phase ohne Roughness-Check",
        "check_roughness_regression() Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V22",
        "PreEcho",
        "pre.echo.*ohne.*guard|transient.*ohne.*PreEcho",
        "Pre-Echo-Schutz ohne Guard",
        "PreEchoPrevention Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V23",
        "Mono",
        "mono.*kompatibilitaet.*ohne.*check|MonoCompat",
        "Mono-Kompatibilitaet ohne Check",
        "Mono-Kompatibilitaets-Check Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V24",
        "Spectral",
        "spektralfarbe.*ohne.*guard|spectral.*color.*ohne",
        "Spektralfarbe ohne Guard",
        "SpectralColorGuard Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V25",
        "Waerme",
        "waermeband.*ohne.*guard|warmth.*band.*ohne",
        "Waermeband ohne Guard",
        "Waermeband-Guard Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V26",
        "Onset",
        "onset.*preserv.*ohne.*guard|onset.*guard.*fehlt",
        "Onset-Preservation ohne Guard",
        "OnsetPreservationGuard Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V27",
        "Defect",
        "JITTER.*phase_12|jitter.*wow_flutter",
        "JITTER mit phase_12 behandeln (falscher Algorithmus)",
        "phase_14+phase_23 fuer digitale Jitter",
        "critical",
    ),
    ForbiddenPattern(
        "V28",
        "Defect",
        "NR_BREATHING.*phase_03|breathing.*denoise",
        "NR-Atmen mit weiterer NR behandeln",
        "phase_54+phase_08 fuer NR-Artefakte",
        "critical",
    ),
    ForbiddenPattern(
        "V29",
        "Defect",
        "OVERLOAD.*phase_63|overload.*intermodulation",
        "Overload mit IMD-Reduktion behandeln",
        "phase_09+phase_23 fuer harmonische Verzerrung",
        "critical",
    ),
    ForbiddenPattern(
        "V30",
        "Defect",
        "ALIASING.*phase_03|aliasing.*denoise",
        "Aliasing mit Denoise behandeln",
        "Nur phase_23+phase_50 fuer Alias-Frequenzen",
        "critical",
    ),
    ForbiddenPattern(
        "V31",
        "Defect",
        "ROOM_MODE.*phase_05|room.*rumble",
        "Raumresonanzen mit Rumble-Filter (zu breit)",
        "phase_04 Notch-EQ als Primary",
        "warning",
    ),
    ForbiddenPattern(
        "V32",
        "Guard",
        "transparenz.*ohne.*drift_exclusion|Hiss.*ohne.*exclusion",
        "NR-Phase ohne transparenz in DRIFT_EXCLUSIONS",
        "transparenz in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS",
        "warning",
    ),
    ForbiddenPattern(
        "V34",
        "Phase",
        "MaterialType.*ohne.*CASSETTE|CASSETTE.*fehlt.*dict",
        "MaterialType.CASSETTE fehlt in dict",
        "Alle dict[MaterialType,...] vollstaendig befuellen",
        "critical",
    ),
    ForbiddenPattern(
        "V35",
        "Phase",
        "phase_63.*ohne.*M/S|IMD.*ohne.*mid.side",
        "Phase_63 ohne M/S-Domain",
        "M/S-Domain: Notch aus Mid, symmetrisch auf Mid+Side",
        "warning",
    ),
    ForbiddenPattern(
        "V36",
        "Phase",
        "phase.*ohne.*strength_feedback|wetness.*ohne.*feedback",
        "Phase-Wetness ohne Mess-Feedback",
        "PhaseConductor.recommend() mit 4D-State-Vektor",
        "warning",
    ),
    ForbiddenPattern(
        "V37",
        "Guard",
        "feste.*guard.*schwelle|MAX_DRIFT.*ohne.*adaptive",
        "Feste Guard-Schwellwerte",
        "compute_adaptive_drift_tolerance()",
        "warning",
    ),
    ForbiddenPattern(
        "V43",
        "Formant",
        "formant.*guard.*±1dB|jnd.*ohne.*resolve",
        "Formant-Guard mit uniformem ±1dB",
        "resolve_jnd_tolerance_db(freq_hz) Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V44",
        "Spatial",
        "spatial_depth.*ohne.*IACC|spatial.*ohne.*iacc",
        "spatial_depth ohne IACC",
        "compute_iacc(audio,sr) in spatial_depth",
        "warning",
    ),
    ForbiddenPattern(
        "V45",
        "Emotion",
        "emotionalitaet.*ohne.*VAT|emotion.*ohne.*valence",
        "emotionalitaet ohne VAT-Blend",
        "VATEmotionEstimator als Blend",
        "warning",
    ),
    ForbiddenPattern(
        "V46",
        "dBFS",
        r"dBFS.*\*.*strength|dbfs.*multiply",
        "dBFS mit linearem Faktor multipliziert",
        "level_db + 20*log10(max(strength,1e-6))",
        "critical",
    ),
    ForbiddenPattern(
        "V47",
        "Clip",
        r"clipping.*0\.999.*ohne.*sub|FLAT_TOPS.*ohne.*adjacent",
        "Clipping nur via FLAT_TOPS ohne Sub-Ceiling",
        "detect_sub_ceiling_clipping()+Adjacent-Ratio",
        "critical",
    ),
    ForbiddenPattern(
        "V48",
        "GAF",
        "goal_applicability.*ohne.*transfer_chain|GAF.*ohne.*chain",
        "GAF ohne transfer_chain-Parameter",
        "evaluate_goal_applicability(transfer_chain=...)",
        "warning",
    ),
    ForbiddenPattern(
        "V49",
        "Goals",
        "goals_passed.*ohne.*inapplicable|goals_passed.*ohne.*Ausschluss",
        "goals_passed ohne inapplicable-Ausschluss",
        "_count_passed schliesst _inappl aus",
        "warning",
    ),
    ForbiddenPattern(
        "V50",
        "Goals",
        "messe_ziele.*ohne.*reference|measure_all.*ohne.*ref",
        "messe_ziele() ohne reference-Parameter",
        "messe_ziele(audio,sr,reference=pre_audio)",
        "warning",
    ),
    ForbiddenPattern(
        "V51",
        "Dataclass",
        "RestaurierErgebnis.*ohne.*goal_applicability|Ergebnis.*ohne.*goal_applic",
        "Dataclass ohne goal_applicability-Feld",
        "goal_applicability:dict in Dataclass",
        "warning",
    ),
    ForbiddenPattern(
        "V52",
        "GAF",
        "separation_fidelity.*ohne.*near.mono|separation.*ohne.*codec",
        "separation_fidelity ohne Near-Mono-Codec",
        "Joint-Stereo-Codec->separation_fidelity inapplicable",
        "warning",
    ),
    ForbiddenPattern(
        "V53",
        "Singer",
        "singer_id.*rollback.*ohne.*dsp_fallback|singer.*ohne.*fallback",
        "Singer-ID-Rollback ohne DSP-Fallback-Guard",
        "if sic<0.92 and not singer_id_dsp_fallback",
        "critical",
    ),
    ForbiddenPattern(
        "V54",
        "HPG",
        r"update_reference_memory.*nie.*gerufen|_hg\.update.*never",
        "update_reference_memory() nie aufgerufen",
        "HPI>0+af>=0.95 -> _hg.update_reference_memory()",
        "critical",
    ),
    ForbiddenPattern(
        "V55",
        "LPC",
        "lpc_formant.*ohne.*era_decade|enhance.*ohne.*era",
        "lpc_formant_enhance ohne era_decade",
        "era_decade<1960->WLPC-Pfad aktivieren",
        "warning",
    ),
    ForbiddenPattern(
        "V56",
        "Frontend",
        r"_AURIK_VERSION.*=.*9\.\d+|hartcodierte.*Version.*Frontend",
        "Frontend-Version hartcodiert",
        "_AURIK_VERSION=unknown als Fallback",
        "critical",
    ),
    ForbiddenPattern(
        "V57",
        "Masking",
        "additive.*phase.*ohne.*ForwardMasking.*panns|phase_.*add.*ohne.*mask",
        "Neue additive Phase ohne ForwardMaskingGuard",
        "ForwardMaskingGuard bei panns_singing>=0.25",
        "warning",
    ),
    ForbiddenPattern(
        "V58",
        "Mypy",
        "return.*ndarray.*no-any-return|->.*ndarray.*Any",
        "no-any-return in ndarray-Funktionen",
        "cast(np.ndarray,result) oder type:ignore",
        "warning",
    ),
]

REQUIRED_PATTERNS: list[RequiredPattern] = [
    RequiredPattern(
        "G01", "Logging", "logger = logging.getLogger", "Jede .py-Datei braucht Logger-Initialisierung", "critical"
    ),
    RequiredPattern("G02", "NaN", "nan_to_num|isfinite", "Ausgabe-Audio MUSS NaN/Inf-Schutz haben", "critical"),
    RequiredPattern("G03", "Type", r"def \w+\([^)]*:\s*\w+", "Oeffentliche Funktionen brauchen Type-Hints", "warning"),
    RequiredPattern("G05", "Data", "@dataclass", "API-Rueckgaben MÜSSEN @dataclass sein", "warning"),
    RequiredPattern("G06", "Lock", r"threading\.Lock\(\)", "Singletons brauchen threading.Lock()", "critical"),
    RequiredPattern("G08", "GPU", "get_torch_device", "GPU-Zugriff NUR via get_torch_device()", "critical"),
    RequiredPattern("G09", "Audio", "load_audio_file", "Audio-Import NUR via load_audio_file()", "critical"),
    RequiredPattern("G12", "ZeroPhase", "sosfiltfilt", "Signal-Addition braucht zero-phase Filter", "critical"),
    RequiredPattern("G13", "Artifact", r"artifact_freedom.*0\.95", "Ausgabe MUSS artifact_freedom>=0.95", "critical"),
    RequiredPattern("G16", "Gate", "reference_for_gate", "Gate braucht reference_for_gate", "critical"),
    RequiredPattern(
        "G18", "Envelope", "_musical_gain_envelope", "Gain MUSS _musical_gain_envelope() nutzen", "critical"
    ),
    RequiredPattern(
        "G41",
        "Pleasantness",
        "compute_pleasantness|check_phase_start",
        "HPE-Check vor/nach JEDER Phase Pflicht",
        "critical",
    ),
    RequiredPattern(
        "G42", "Consistent", "deterministic|seed|reproducible", "Gleicher Input = gleicher Output", "critical"
    ),
    RequiredPattern(
        "G43",
        "CLP",
        "clp_max_attenuation|CLPZone|get_clp_attenuation",
        "NR-Staerke MUSS CLP-Maske respektieren",
        "critical",
    ),
    RequiredPattern(
        "G44",
        "Whisper",
        "whisper_detail|whisper_preservation|WhisperPreservation",
        "Leise-Passagen MUESSEN geschuetzt werden",
        "critical",
    ),
    RequiredPattern(
        "G45",
        "Dynamics",
        "dynamics_preserver|check_and_restore_dynamics",
        "Nach Kompressor/Limiter Dynamik pruefen",
        "critical",
    ),
    RequiredPattern(
        "G46", "Adaptive", "compute_adaptive_thresholds", "Goal-Schwellen aus Materialphysik berechnen", "critical"
    ),
    RequiredPattern(
        "G47", "Version", "from backend.core.version import|AURIK_VERSION", "Version NUR aus version.py", "warning"
    ),
    RequiredPattern(
        "G48", "Error", "logger.warning|logger.error|exc_info=True", "Exceptions MUESSEN geloggt werden", "critical"
    ),
    RequiredPattern("G49", "Dither", "dither|TPDF|triangular", "16-bit Export MUSS gedithert werden", "warning"),
    RequiredPattern(
        "G50",
        "PhaseOrder",
        "phase_order.*material|adaptive.*phase.*order",
        "Phasen-Reihenfolge materialabhaengig",
        "warning",
    ),
    RequiredPattern(
        "G51", "Watchdog", "WatchdogMonitor|get_watchdog|pipeline_guard", "Watchdog in JEDEM Lauf aktiv", "critical"
    ),
    RequiredPattern(
        "G52", "Timeout", "timeout|_MAX_PHASE_SECONDS|phase_timeout", "Jede Phase braucht Timeout", "critical"
    ),
    RequiredPattern(
        "G53", "Memory", "ml_memory_budget|try_allocate|memory_limit", "Speicher vor ML-Ladung pruefen", "critical"
    ),
    RequiredPattern("G54", "ColdStart", "COLDSTART|_coldstart|cold_start", "Cold-Start braucht Zeitbudget", "warning"),
    RequiredPattern(
        "G55", "GPUFallback", "CPUExecutionProvider|cpu.*fallback.*onnx", "GPU-Fehler -> ONNX-CPU weiter", "critical"
    ),
    RequiredPattern(
        "G56", "Chain", "transfer_chain|chain_string|Tontraegerkette", "Transfer-Ketten erkennen", "critical"
    ),
    RequiredPattern(
        "G57", "Bandwidth", "original.*medium|oldest.*carrier", "Aeltester Traeger = Bandbreiten-Ziel", "warning"
    ),
    RequiredPattern(
        "G58", "Codec", "codec.*artifact|mpeg.*frame|mp3.*artifact", "Codec-Artefakte getrennt behandeln", "warning"
    ),
    RequiredPattern("G59", "Dolby", "DolbyNR|dolby_nr|dolby_type", "Dolby vor NR erkennen", "warning"),
    RequiredPattern("G60", "RIAA", "RIAA|riaa_eq|riaa_curve", "RIAA vor Vinyl invers anwenden", "warning"),
    RequiredPattern(
        "G61", "Primus", "panns_singing.*0.25|primus.*inter.*pares", "Stimmqualitaet VORRANG bei Gesang", "critical"
    ),
    RequiredPattern(
        "G62", "Formant", "formant.*guard|formant.*integrity", "Formant-Integritaet nach Vokal-Phase", "critical"
    ),
    RequiredPattern(
        "G63",
        "Sibilance",
        "sibilance.*preserv|deesser.*intelligibility",
        "Sibilanten nach De-Essing unterscheidbar",
        "warning",
    ),
    RequiredPattern("G64", "Vibrato", "vibrato.*preserv|vibrato.*guard", "Vibrato MUSS erhalten bleiben", "critical"),
    RequiredPattern("G65", "Breath", "breath.*preserv|breath.*emotion", "Atem als musikalisch intentional", "warning"),
    RequiredPattern(
        "G66", "Export", "artifact_freedom.*0.95.*pleasantness", "Vor Export: af>=0.95 UND hpe>=0.35", "critical"
    ),
    RequiredPattern(
        "G67",
        "PleasantnessGate",
        "pleasantness.*worse|hpe.*rollback",
        "Wenn schlechter -> Original ausgeben",
        "critical",
    ),
    RequiredPattern(
        "G68", "Regression", "test.*regression|regression.*test", "Spec-Aenderung braucht Regression-Test", "warning"
    ),
    RequiredPattern(
        "G69", "Agent", "get_constitution|SpecConstitution", "KI-Agenten nutzen SpecConstitution", "warning"
    ),
    RequiredPattern(
        "G70", "VersionCheck", "__version__|AURIK_VERSION", "Keine hartcodierten Versionsnummern", "critical"
    ),
    RequiredPattern(
        "G04",
        "Doc",
        "def test_|def check_|def get_|def compute_",
        "Oeffentliche Funktionen brauchen Docstrings",
        "warning",
    ),
    RequiredPattern(
        "G07",
        "ML",
        "DSP.*fallback|cpu.*fallback|CPUExecutionProvider",
        "ML-Plugins brauchen DSP-Fallback",
        "critical",  # §V6 (copilot-instructions.md): logger.warning handled at call site
    ),
    RequiredPattern("G10", "Test", r"test_.*\.py|def test_", "Jede Spec-$-Referenz MUSS einen Test haben", "warning"),
    RequiredPattern("G11", "Loudness", "loudness|LUFS|lufs", "Ausgabe MUSS LUFS-normalisiert sein", "warning"),
    RequiredPattern("G14", "Vocal", "vqi|vocal_quality|VocalQuality", "panns_singing>=0.25 -> VQI Pflicht", "critical"),
    RequiredPattern("G15", "DC", "filtfilt.*1.*-1.*0.9995", "reel_tape: zero-phase DC-Filter", "warning"),
    RequiredPattern("G17", "Peak", "percentile.*99.9", "Peak-Guard: np.percentile(99.9)", "warning"),
    RequiredPattern("G19", "Material", "MaterialType|material_type", "dict[MaterialType,...] vollstaendig", "critical"),
    RequiredPattern("G20", "NR", "compute_nmr_score|nmr_score", "NR->NMR-Score Pflicht", "warning"),
    RequiredPattern("G21", "Glue", "glue_stage|phase_glue|GlueStage", "Glue-Stage in ALLEN Modi", "critical"),
    RequiredPattern("G22", "PIM", "pim|perceptual_intensity", "PIM vor Phasen-Loop", "warning"),
    RequiredPattern("G23", "RLP", "rlp|reflective_listening", "RLP nach Phasen-Loop", "warning"),
    RequiredPattern("G24", "Export", "artifact_freedom.*0.95.*export", "Export: artifact_freedom pruefen", "critical"),
    RequiredPattern("G25", "Warmup", "warmup|_ROCM_WARMUP|warmup_rocm", "ROCm-GPU warmup Pflicht", "warning"),
    RequiredPattern(
        "G26", "Recovery", "cpu.*fallback|CPU.*fallback|CPUExecution", "GPU-Fehler -> CPU Fallback", "critical"
    ),
    RequiredPattern("G27", "Budget", "_3X_RT_LIMIT|rt_factor|rt_limit", "8xRT-Budget Pflicht", "critical"),
    RequiredPattern(
        "G28", "Checkpoint", "save_checkpoint|checkpoint|recovery", "Crash-Recovery Checkpoints", "warning"
    ),
    RequiredPattern("G29", "Memory", "ml_memory_budget|memory_budget", "ML-Memory vor Allocation", "warning"),
    RequiredPattern(
        "G30", "CrossPhase", "CumulativeInteractionGuard|cumulative", "Cross-Phase-Guards Pflicht", "warning"
    ),
    RequiredPattern("G31", "UnitTest", r"test_phase_.*\.py|def test_phase", "Phase-Unit-Test Pflicht", "warning"),
    RequiredPattern("G32", "CI", "workflows|ci_benchmark|CI_GATE", "CI: artifact_freedom-Regression", "warning"),
    RequiredPattern("G33", "Coverage", "coverage|pytest-cov", "Test-Coverage >=80%", "warning"),
    RequiredPattern("G34", "Linter", "ruff|mypy|pre-commit-config", "Pre-Commit: ruff+mypy", "critical"),
    RequiredPattern("G35", "Benchmark", "amrb|AMRB|worldclass.*benchmark", "AMRB-Benchmark Pflicht", "warning"),
    RequiredPattern("G36", "Version", "CHANGELOG|changelog", "CHANGELOG.md Pflicht", "warning"),
    RequiredPattern("G37", "Lock", "with.*_lock|with.*Lock", "Lock mit with-Statement", "warning"),
    RequiredPattern("G38", "Path", r"Path\(|pathlib", "pathlib.Path Pflicht", "warning"),
    RequiredPattern("G39", "Encoding", "encoding.*utf", "UTF-8 Pflicht", "warning"),
    RequiredPattern("G40", "License", "Aurik 10|Copyright.*Aurik", "Lizenzheader Pflicht", "warning"),
    # ── §v10.305 Startup-Integrations-Vertrag (§G71 (GEBOTE.md)–§G80 (GEBOTE.md)) ──────────────
    RequiredPattern(
        "G71",
        "StartupEvent",
        "_detection_complete\\.set|complete\\.set",
        "Jedes threading.Event MUSS in finally/except-garantiert gesetzt werden",
        "critical",
    ),
    RequiredPattern(
        "G72",
        "LockFreeImport",
        "# LOCK_FREE_IMPORT|NO_LOCK_DURING_IMPORT",
        "Lock NICHT während Import/I/O halten — in finally/ausserhalb importieren",
        "critical",
    ),
    RequiredPattern(
        "G73",
        "WarmupValidator",
        "warmup.*validator|validate.*warmup.*accessor|_verify_warmup",
        "Warmup MUSS Plugin-Zugriffsnamen vor erstem Lauf validieren",
        "critical",
    ),
    RequiredPattern(
        "G74",
        "WatchdogSelfTest",
        "watchdog.*self.*test|_self_test.*watchdog|_probe_rocm|_probe.*pad",
        "Jeder Watchdog/Probe MUSS Startup-Selbsttest haben + aufgerufen werden",
        "critical",
    ),
    RequiredPattern(
        "G75",
        "CacheInvalidation",
        "__pycache__|PYTHONDONTWRITEBYTECODE|python.*-B",
        "Startup-Skript MUSS -B-Flag oder PYTHONDONTWRITEBYTECODE setzen",
        "warning",
    ),
    RequiredPattern(
        "G76",
        "HappyPathGate",
        " detected_medium_label\\.setText|_update_all|_dispatch_to_gui.*_update_all",
        "GUI MUSS mindestens einen Pfad haben, der Analyse-Labels setzt — Warning bei stillem Skip",
        "critical",
    ),
    RequiredPattern(
        "G77",
        "StartupSmokeTest",
        "test_startup_smoke|test_startup_integration|startup.*smoke",
        "Startup-Smoke-Test: GPU+Warmup+PreAnalysis in <60s",
        "warning",
    ),
    RequiredPattern(
        "G78",
        "ImportCheck",
        "^import os$|^from os import",
        "Jedes Modul MUSS alle verwendeten Standard-Imports haben (ruff F821)",
        "critical",
    ),
    RequiredPattern(
        "G79",
        "EventFinally",
        "finally:.*_detection_complete|except.*_detection_complete",
        "_detection_complete MUSS in finally-Block gesetzt werden",
        "critical",
    ),
    RequiredPattern(
        "G80",
        "ProbeInvocation",
        "_probe_.*\\(\\)|self\\._probe_",
        "Jede _probe_*-Methode MUSS in __init__ oder _detect_backend aufgerufen werden",
        "critical",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 15 Musical Goals — Schwellwerte (Spec 01)
# ═══════════════════════════════════════════════════════════════════════════════

MUSICAL_GOALS = {
    "authentizitaet": {"threshold": 0.70, "weight": 0.10, "priority": 1},
    "natuerlichkeit": {"threshold": 0.70, "weight": 0.12, "priority": 1},
    "brillanz": {"threshold": 0.65, "weight": 0.06, "priority": 2},
    "timbre": {"threshold": 0.65, "weight": 0.08, "priority": 2},
    "groove": {"threshold": 0.60, "weight": 0.07, "priority": 2},
    "micro_dynamics": {"threshold": 0.60, "weight": 0.08, "priority": 3},
    "artikulation": {"threshold": 0.65, "weight": 0.08, "priority": 2},
    "waerme": {"threshold": 0.70, "weight": 0.07, "priority": 2},
    "tiefe": {"threshold": 0.65, "weight": 0.06, "priority": 3},
    "durchsetzung": {"threshold": 0.65, "weight": 0.06, "priority": 3},
    "transparenz": {"threshold": 0.65, "weight": 0.06, "priority": 2},
    "kohaerenz": {"threshold": 0.65, "weight": 0.05, "priority": 3},
    "fokus": {"threshold": 0.60, "weight": 0.05, "priority": 3},
    "balance": {"threshold": 0.65, "weight": 0.04, "priority": 3},
    "stimmung": {"threshold": 0.60, "weight": 0.03, "priority": 3},
}

# Material-adaptive Floor-Toleranzen (fragile Materialien haben niedrigere Erwartungen)
MATERIAL_GOAL_FLOOR: dict[str, dict[str, float]] = {
    # Wachswalze (1877-1929): Frequenzbereich 200-5000 Hz, SNR ~20dB
    "wax_cylinder": {
        "authentizitaet": 0.55,
        "natuerlichkeit": 0.55,
        "brillanz": 0.40,
        "timbre": 0.50,
        "groove": 0.45,
        "artikulation": 0.55,
        "waerme": 0.60,
        "transparenz": 0.45,
        "kohaerenz": 0.50,
        "micro_dynamics": 0.40,
        "fokus": 0.45,
        "balance": 0.50,
        "tiefe": 0.45,
        "durchsetzung": 0.48,
        "stimmung": 0.55,
    },
    # Schellack (1895-1958): 100-8000 Hz, SNR ~30dB, RIAA-EQ fehlt
    # Code uebertrifft Specs konstant → angehoben per Continuous Improvement
    "shellac": {
        "authentizitaet": 0.65,
        "natuerlichkeit": 0.63,
        "brillanz": 0.50,
        "timbre": 0.58,
        "groove": 0.58,
        "artikulation": 0.63,
        "waerme": 0.68,
        "transparenz": 0.53,
        "kohaerenz": 0.58,
        "micro_dynamics": 0.52,
        "fokus": 0.55,
        "balance": 0.58,
        "tiefe": 0.55,
        "durchsetzung": 0.58,
        "stimmung": 0.60,
    },
    # Vinyl (1948-): 20-20000 Hz, SNR ~60dB, RIAA-EQ
    # Code uebertrifft Specs konstant → angehoben per Continuous Improvement
    "vinyl": {
        "authentizitaet": 0.70,
        "natuerlichkeit": 0.68,
        "brillanz": 0.63,
        "timbre": 0.68,
        "groove": 0.68,
        "artikulation": 0.68,
        "waerme": 0.72,
        "transparenz": 0.63,
        "kohaerenz": 0.68,
    },
    # Tonband (1930-): 30-18000 Hz, SNR ~55dB, Bandsättigung
    "tape": {
        "authentizitaet": 0.65,
        "natuerlichkeit": 0.65,
        "brillanz": 0.60,
        "timbre": 0.65,
        "groove": 0.65,
        "artikulation": 0.65,
        "waerme": 0.70,
        "transparenz": 0.60,
        "kohaerenz": 0.65,
    },
    # Kassette (1963-): 30-16000 Hz, SNR ~50dB, Dolby B/C/S
    "cassette": {
        "authentizitaet": 0.60,
        "natuerlichkeit": 0.60,
        "brillanz": 0.55,
        "timbre": 0.60,
        "groove": 0.60,
        "artikulation": 0.60,
        "waerme": 0.65,
        "transparenz": 0.55,
        "kohaerenz": 0.60,
        "micro_dynamics": 0.50,
        "fokus": 0.55,
        "balance": 0.58,
        "tiefe": 0.55,
        "durchsetzung": 0.58,
        "stimmung": 0.58,
    },
    # CD/DAT/High-Quality Digital (1982-): 20-20000 Hz, SNR >90dB
    "cd_digital": {
        "authentizitaet": 0.75,
        "natuerlichkeit": 0.75,
        "brillanz": 0.70,
        "timbre": 0.75,
        "groove": 0.75,
        "artikulation": 0.75,
        "waerme": 0.70,
        "transparenz": 0.75,
        "kohaerenz": 0.75,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Bug & Gap Detection — 5-Layer-Scan-Protokoll (Spec 10)
# ═══════════════════════════════════════════════════════════════════════════════

BUG_GAP_LAYERS = {
    "L1_Frontend": {"scope": "Aurik10/ui/, Aurik10/__init__.py", "tools": "version-check, grep hardcoded"},
    "L2_Bridge_CLI": {"scope": "backend/api/bridge.py, cli/", "tools": "contract-completeness"},
    "L3_Denker": {"scope": "denker/*.py", "tools": "goal_applicability, reference params"},
    "L4_UV3": {"scope": "backend/core/unified_restorer_v3.py", "tools": "SSIP, Rescheduler, Memory, HPG"},
    "L5_Phases_DSP": {
        "scope": "backend/core/phases/, dsp/, plugins/",
        "tools": "V33-V42, ForwardMasking, Halluzination",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# SpecConstitution — Der programmatische Zugang
# ═══════════════════════════════════════════════════════════════════════════════


class SpecConstitution:
    """Zentrale, maschinenlesbare Aurik-Konstitution.

    Lädt und validiert alle normativen Regeln aus den Spec-Dokumenten.
    Wird von Agent und Watchdog als Single Source of Truth verwendet.

    Usage:
        const = get_constitution()
        ok, issues = const.check_paragraph_zero(audio, sr)
        patterns = const.get_forbidden_patterns()
        goals = const.get_musical_goal_thresholds("vinyl")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forbidden: list[ForbiddenPattern] = FORBIDDEN_PATTERNS
        self._musical_goals: dict[str, dict[str, float]] = MUSICAL_GOALS
        self._material_floors: dict[str, dict[str, float]] = MATERIAL_GOAL_FLOOR
        self._shield: dict[str, float] = MUSIC_DEATH_SHIELD
        self._transparency: dict[str, Any] = PERCEPTUAL_TRANSPARENCY

    # ── §0 Validation ────────────────────────────────────────────────────

    def check_paragraph_zero(
        self,
        audio: np.ndarray,
        sr: int,
        artifact_freedom: float = 1.0,
        hpi: float = 0.5,
        has_vocals: bool = False,
        vqi: float = 0.8,
        chain_depth: int = 1,  # §v10.119: für depth-adaptiven artifact_freedom_min
        restorability: float = 50.0,  # §v10.119: für material-adaptiven Schwellwert
    ) -> list[str]:
        """Prüft gegen §0 Klangwahrheit und Music-Death-Shield.

        Returns:
            Liste von Verletzungen (leer = alle Checks bestanden).
        """
        issues: list[str] = []

        # §0h: Music-Death-Shield — Artifact Freedom ist primäres Veto
        # §v10.119: Depth-adaptiver Schwellwert. Studio-Master (depth=1)
        # kann 0.95 erreichen, aber 4-stufige Transfer-Chain (depth≥4)
        # ist physikalisch auf ~0.70 limitiert.
        _af_min = self._shield["artifact_freedom_min"]
        if chain_depth >= 4:
            _af_min = 0.70
        elif chain_depth == 3:
            _af_min = 0.80
        elif chain_depth == 2:
            _af_min = 0.88
        # §v10.119: Material-Adaptivität — schwer restaurierbares Quellmaterial
        # (schmale Bandbreite, starke Vorbelastung) ist physikalisch auf niedrigere
        # artifact_freedom limitiert. Der Parameter existierte, wurde aber nie
        # ausgewertet (Befund 2026-08-16). Greift erst UNTER „fair“ (rs<50),
        # damit die depth-Schwellen bei unklarem Material unverändert bleiben.
        if restorability < 35:
            _af_min = max(0.65, _af_min - 0.10)
        elif restorability < 50:
            _af_min = max(0.65, _af_min - 0.05)
        if artifact_freedom < _af_min:
            issues.append(
                f"§0h VETO: artifact_freedom={artifact_freedom:.3f} < "
                f"{_af_min} (depth={chain_depth}) — Export muss blockiert werden"
            )

        # §0h: HPI ≤ 0 → Over-Processing
        if hpi <= self._shield["hpi_min"]:
            issues.append(f"§0h VETO: HPI={hpi:.3f} ≤ 0 — Signal wurde verschlechtert. Original-Input exportieren.")

        # §0p: Primus-inter-Pares — Vokal-Vorrang
        if has_vocals and vqi < self._shield["vqi_recovery_trigger"]:
            issues.append(
                f"§0p RECOVERY: VQI={vqi:.3f} < {self._shield['vqi_recovery_trigger']} "
                f"bei erkanntem Gesang — _recovery_cascade() auslösen"
            )

        return issues

    def check_primum_non_nocere(
        self,
        original_rms_dbfs: float,
        restored_rms_dbfs: float,
        crest_delta_db: float,
    ) -> list[str]:
        """Prüft gegen Primum non nocere.

        Erkennt:
          - RMS-Kollaps (Signal wurde zu leise / verschwand)
          - Crest-Verlust (Dynamik wurde plattgebügelt)
          - Mögliche Artefakt-Einführung durch Überbearbeitung
        """
        issues: list[str] = []

        if restored_rms_dbfs < -80.0:
            issues.append("Primum non nocere: RMS-Kollaps — Signal wurde zerstört")
        if restored_rms_dbfs < original_rms_dbfs - 15.0:
            issues.append(
                f"Primum non nocere: RMS-Abfall {original_rms_dbfs - restored_rms_dbfs:.1f}dB — möglicher Signalkollaps"
            )
        if crest_delta_db < -6.0:
            issues.append(f"Primum non nocere: Crest-Verlust {crest_delta_db:.1f}dB — Dynamik wurde plattgebügelt")

        return issues

    # ── Forbidden Patterns ──────────────────────────────────────────────

    def get_forbidden_patterns(self, severity: str | None = None) -> list[ForbiddenPattern]:
        """Gibt alle VERBOTEN-Regeln zurück, optional gefiltert nach Severity."""
        if severity:
            return [p for p in self._forbidden if p.severity == severity]
        return list(self._forbidden)

    def get_critical_forbidden(self) -> list[ForbiddenPattern]:
        """Nur kritische VERBOTEN-Regeln (müssen immer gelten)."""
        return [p for p in self._forbidden if p.severity == "critical"]

    def check_forbidden_in_code(self, code_text: str) -> list[tuple[ForbiddenPattern, str]]:
        """Scannt Code-Text auf VERBOTEN-Muster und gibt Treffer zurück."""
        import re

        hits: list[tuple[ForbiddenPattern, str]] = []
        for fp in self._forbidden:
            try:
                matches = re.finditer(fp.pattern, code_text, re.MULTILINE)
                for m in matches:
                    line = code_text[max(0, m.start() - 20) : m.end() + 20].replace("\n", " ")
                    hits.append((fp, line.strip()))
            except re.error:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        return hits

    # ── Musical Goals ───────────────────────────────────────────────────

    def get_required_patterns(self) -> list[RequiredPattern]:
        return list(REQUIRED_PATTERNS)

    def check_required_in_code(self, code_text: str) -> list[tuple[RequiredPattern, bool]]:
        """Prueft ob alle PFLICHT-Regeln im Code erfuellt sind.

        Returns: list of (pattern, is_present)
        """
        import re

        results: list[tuple[RequiredPattern, bool]] = []
        for rp in REQUIRED_PATTERNS:
            try:
                present = bool(re.search(rp.pattern, code_text, re.MULTILINE))
                results.append((rp, present))
            except re.error:
                results.append((rp, False))
        return results

    def find_missing_requirements(self, code_text: str) -> list[RequiredPattern]:
        """Gibt alle VERLETZTEN PFLICHT-Regeln zurueck."""
        return [rp for rp, present in self.check_required_in_code(code_text) if not present]

    def get_musical_goal_thresholds(self, material: str = "unknown") -> dict[str, float]:
        """Gibt Goal-Schwellwerte zurück, material-adaptiv mit Floor-Toleranzen.

        Für fragile Materialien (Wachswalze, Schellack) sind die Schwellwerte
        NIEDRIGER als die Defaults — sie respektieren die physikalischen Grenzen.
        """
        thresholds: dict[str, float] = {}
        _mk = material.value if isinstance(material, MaterialType) else material  # type: ignore[name-defined]  # §v10.113
        floor = self._material_floors.get(_mk, {})
        for goal, cfg in self._musical_goals.items():
            floor_val = floor.get(goal, cfg["threshold"])
            # Floor ist das physikalisch maximal Erreichbare für dieses Material.
            # Der Schwellwert ist der Floor-Wert (niedriger als Default bei fragilen Trägern).
            thresholds[goal] = floor_val
        return thresholds

    def compute_adaptive_thresholds(
        self,
        material: str = "unknown",
        *,
        effective_bandwidth_hz: float = 20000.0,
        effective_snr_db: float = 60.0,
        era_decade: int = 2000,
    ) -> dict[str, float]:
        """Berechnet Goal-Schwellwerte DYNAMISCH aus physikalischen Materialeigenschaften.

        Statt statischer Hardcoded-Werte werden die Schwellwerte aus den
        fundamentalen physikalischen Limits des Tonträgers abgeleitet:

        - brillanz ∝ log(bandwidth): Wachswalze 5kHz→0.40, CD 20kHz→0.85
        - transparenz ∝ SNR: Shellac 30dB→0.50, Digital 90dB→0.75
        - authentizitaet: höher für ältere Aufnahmen (Era-Bonus)
        - natuerlichkeit ∝ sqrt(bandwidth × SNR / reference)

        Wissenschaftliche Basis:
          - Nyquist-Shannon: maximale Frequenzauflösung = bandwidth/2
          - Fletcher-Munson: Hörschwelle steigt unter 200Hz und über 8kHz
          - Zwicker/Fastl: Psychoakustische Schärfe ∝ Bandbreite
          - ITU-R BS.1770: Loudness-Wahrnehmung frequenzabhängig

        Args:
            material: Material-Typ (z.B. "wax_cylinder", "vinyl")
            effective_bandwidth_hz: Effektive Bandbreite in Hz (-3dB Punkt)
            effective_snr_db: Effektives SNR in dB
            era_decade: Aufnahmedekade (z.B. 1920, 1965, 2000)

        Returns:
            dict von Goal-Name → physikalisch berechneter Schwellwert [0,1]
        """
        import math

        # Normalisiere Bandbreite auf [200, 20000] Hz
        _bw_log = math.log10(max(effective_bandwidth_hz, 200.0))
        bw_norm = max(0.0, min(1.0, (_bw_log - math.log10(200.0)) / (math.log10(20000.0) - math.log10(200.0))))

        # Normalisiere SNR auf [15, 90] dB
        snr_norm = max(0.0, min(1.0, (effective_snr_db - 15.0) / 75.0))

        # Era-Faktor: ältere Aufnahmen haben inhärent weniger Bandbreite/SNR,
        # daher sind die Schwellwerte niedriger (Authentizität wird höher gewichtet)
        era_age = max(0, 2026 - era_decade) / 100.0  # 0.0 (heute) bis 1.26 (1877)
        era_factor = max(0.0, min(1.0, 1.0 - era_age * 0.5))

        # ── Physikalisch berechnete Schwellwerte ──
        computed: dict[str, float] = {}

        # brillanz: direkt proportional zur effektiven Bandbreite
        # Wachswalze 5kHz→0.40, Schellack 8kHz→0.50, Vinyl 20kHz→0.85
        computed["brillanz"] = round(0.40 + bw_norm * 0.45, 3)

        # transparenz: proportional zu SNR (Rauschabstand bestimmt Durchsichtigkeit)
        computed["transparenz"] = round(0.45 + snr_norm * 0.35, 3)

        # authentizitaet: höher für historische Aufnahmen (Era-Bonus)
        base_auth = 0.65 + snr_norm * 0.15
        era_bonus = (1.0 - era_factor) * 0.15
        computed["authentizitaet"] = round(min(0.85, base_auth + era_bonus), 3)

        # natuerlichkeit: kombiniert Bandbreite + SNR
        nat_base = 0.60 + (bw_norm * 0.5 + snr_norm * 0.5) * 0.20
        computed["natuerlichkeit"] = round(min(0.85, nat_base), 3)

        # waerme: inverse zu Bandbreite (schmalbandig = wärmer)
        computed["waerme"] = round(0.75 - bw_norm * 0.15, 3)

        # timbre: proportional zu Bandbreite (mehr Obertöne = reichere Klangfarbe)
        computed["timbre"] = round(0.55 + bw_norm * 0.25, 3)

        # artikulation: SNR-abhängig (höheres SNR = bessere Verständlichkeit)
        computed["artikulation"] = round(0.55 + snr_norm * 0.25, 3)

        # groove: leicht SNR-abhängig (Rauschen maskiert rhythmische Details)
        computed["groove"] = round(0.55 + snr_norm * 0.15, 3)

        # micro_dynamics: SNR-abhängig (Rauschen maskiert leise Details)
        computed["micro_dynamics"] = round(0.50 + snr_norm * 0.25, 3)

        # kohaerenz: Bandbreite-abhängig (mehr Frequenzen = mehr Kohärenz-Risiko)
        computed["kohaerenz"] = round(0.60 + (1.0 - bw_norm) * 0.10, 3)

        # tiefe: leicht Bandbreite-abhängig (mehr Höhen = mehr räumliche Tiefe)
        computed["tiefe"] = round(0.55 + bw_norm * 0.15, 3)

        # durchsetzung: SNR-abhängig
        computed["durchsetzung"] = round(0.55 + snr_norm * 0.20, 3)

        # fokus: SNR-abhängig (Rauschen verschmiert Fokus)
        computed["fokus"] = round(0.55 + snr_norm * 0.15, 3)

        # balance: relativ konstant, leicht SNR-abhängig
        computed["balance"] = round(0.60 + snr_norm * 0.10, 3)

        # stimmung: Era-abhängig (ältere Aufnahmen haben inhärent mehr "Stimmung")
        computed["stimmung"] = round(0.55 + (1.0 - era_factor) * 0.15, 3)

        # Physikalisches Limit: Floor = maximal erreichbarer Wert für dieses Material.
        # Das dynamische Modell darf den Floor NICHT überschreiten.
        _mk = material.value if isinstance(material, MaterialType) else material  # type: ignore[name-defined]  # §v10.113
        floor = self._material_floors.get(_mk, {})
        for goal in computed:
            if goal in floor:
                computed[goal] = min(computed[goal], floor[goal])

        return computed

    def get_goal_weights(self) -> dict[str, float]:
        """Gibt Goal-Gewichte für Composite-Score zurück."""
        return {g: c["weight"] for g, c in self._musical_goals.items()}

    def evaluate_goals(self, scores: dict[str, float], material: str = "unknown") -> tuple[int, int, list[str]]:
        """Evaluiert Goal-Scores gegen material-adaptive Schwellwerte.

        Returns:
            (passed, total, failed_goals)
        """
        thresholds = self.get_musical_goal_thresholds(material)
        passed = 0
        total = 0
        failed: list[str] = []
        for goal, score in scores.items():
            if goal in thresholds:
                total += 1
                if score >= thresholds[goal]:
                    passed += 1
                else:
                    failed.append(f"{goal}: {score:.3f} < {thresholds[goal]:.3f}")
        return passed, total, failed

    # ── Music Death Shield ──────────────────────────────────────────────

    def get_shield_thresholds(self) -> dict[str, float]:
        return dict(self._shield)

    def is_export_blocked(
        self,
        artifact_freedom: float,
        hpi: float,
        chain_depth: int = 1,
        restorability: float = 50.0,
    ) -> tuple[bool, str]:
        """Prüft ob der Export durch §0h blockiert wird.

        §v10.119: chain_depth relaxes the artifact_freedom threshold
        for deep transfer chains (cassette=4 → 0.70 statt 0.95).
        restorability moduliert zusätzlich material-adaptiv.
        """
        _af_min = self._shield["artifact_freedom_min"]
        if chain_depth >= 4:
            _af_min = 0.70
        elif chain_depth == 3:
            _af_min = 0.80
        elif chain_depth == 2:
            _af_min = 0.88
        # §v10.119: Material-Adaptivität (identisch zu check_paragraph_zero)
        if restorability < 35:
            _af_min = max(0.65, _af_min - 0.10)
        elif restorability < 50:
            _af_min = max(0.65, _af_min - 0.05)
        if artifact_freedom < _af_min:
            return True, f"§0h: artifact_freedom={artifact_freedom:.3f} < {_af_min} (depth={chain_depth})"
        if hpi <= 0:
            return True, f"§0h: HPI={hpi:.3f} ≤ 0 — Over-Processing"
        return False, ""

    # ── Bug-Gap-Strategie ───────────────────────────────────────────────

    def get_bug_gap_layers(self) -> dict[str, dict[str, str]]:
        return dict(BUG_GAP_LAYERS)

    def validate_layer_completeness(self, layer: str, artifacts: dict[str, Any]) -> list[str]:
        """Validiert ob eine Bug-Gap-Layer vollständig geprüft wurde."""
        if layer not in BUG_GAP_LAYERS:
            return [f"Unbekannte Layer: {layer}"]
        expected = BUG_GAP_LAYERS[layer]
        missing = []
        for tool in expected["tools"].split(", "):
            if tool not in str(artifacts):
                missing.append(f"Fehlendes Tool: {tool} in {layer}")
        return missing

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def forbidden_count(self) -> int:
        return len(self._forbidden)

    @property
    def required_count(self) -> int:
        return len(REQUIRED_PATTERNS)

    @property
    def goal_count(self) -> int:
        return len(self._musical_goals)

    @property
    def shield(self) -> dict[str, float]:
        return dict(self._shield)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_constitution: SpecConstitution | None = None
_constitution_lock = threading.Lock()


def get_constitution() -> SpecConstitution:
    """Thread-sicherer Singleton-Accessor."""
    global _constitution
    with _constitution_lock:
        if _constitution is None:
            _constitution = SpecConstitution()
    return _constitution
