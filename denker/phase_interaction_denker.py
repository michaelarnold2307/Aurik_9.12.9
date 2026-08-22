"""
denker/phase_interaction_denker.py — PhaseInteractionDenker
=============================================================
Steuernde Intelligenz für Phase-Orchestrierung in Aurik 10.

Übernimmt von UnifiedRestorerV3:
  - semantische Typ-Annotation jeder Phase
  - typ-paar-basierte Konflikterkennung (§2.48)
  - constraint-getriebene Reihenfolge (Spec §2.46 Carrier-Chain-Inversion)

Bislang war diese Logik in UV3._select_phases() (1400+ Zeilen) und
UV3._optimize_phase_plan_intelligence() als hartcodierte if/elif-Kaskaden
und Einzelregel-Guards eingebettet. Neue Phasen erforderten neue Guards.

Dieser Denker ersetzt das durch semantische Regeln:
  1. UV3._select_phases() läuft als Werkzeug (Defekt-zu-Phasen-Mapping bleibt dort)
  2. UV3._optimize_phase_plan_intelligence() läuft als Werkzeug (kausal-bewusstes Ordering)
  3. PhaseInteractionDenker verfeinert das Ergebnis: Typ-basierte Konflikte + Constraints
  4. PhasePlan.phases wird via precomputed_phase_plan-kwarg an UV3.restore() übergeben

UV3 ist dann reiner Executor — keine Orchestrierungs-Entscheidungen mehr in restore().

Aufruf (AurikDenker Stufe 5b, nach StrategieDenker, vor _run_rest):
    plan = get_phase_interaction_denker().plan(
        defect_result=cached_defect_result,
        material=material,
        mode=effective_mode,
        chain_info=chain_info,
        defekt_hint=_defekt_hint,
        audio=aktuelles_audio,
        sr=sr,
        restorability_score=float(getattr(cached_restorability_result, "restorability_score", 70.0)),
    )
    # plan.phases → precomputed_phase_plan → RestaurierDenker → UV3.restore()

Spec §2.46 Carrier-Chain-Inversion + §2.47 Adaptive-Intelligence + §2.48 Kumulative-Guard
v10.0.0 — PhaseInteractionDenker
"""

from __future__ import annotations

import logging
import threading
from copy import copy
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_SINGLETON: dict[str, PhaseInteractionDenker | None] = {"instance": None}
_lock = threading.Lock()

# §v10.303.3 Denker-Feedback-Loop: UV3 schreibt hier die Familien rein,
# die es bei Low-Confidence gestrippt hat. Der Denker liest sie beim
# nächsten Planungszyklus und plant diese Familien gar nicht erst ein.
_low_confidence_stripped_cache: frozenset[str] = frozenset()


def record_low_confidence_stripped_families(families: set[str] | frozenset[str]) -> None:
    """§v10.303.3: UV3 ruft dies nach dem Strip, damit der Denker lernt."""
    global _low_confidence_stripped_cache
    _low_confidence_stripped_cache = frozenset(families)


def get_low_confidence_stripped_families() -> frozenset[str]:
    """§v10.303.3: Denker liest dies vor der Planung."""
    return _low_confidence_stripped_cache


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Lädt Symbole lazy, um schwere/zyklische Imports zu vermeiden."""
    return getattr(import_module(module_name), symbol_name)


def get_phase_interaction_denker() -> PhaseInteractionDenker:
    """Gibt die thread-sichere Singleton-Instanz des PhaseInteractionDenker zurück."""
    _instance = _SINGLETON["instance"]
    if _instance is None:
        with _lock:
            _instance = _SINGLETON["instance"]
            if _instance is None:
                _instance = PhaseInteractionDenker()
                _SINGLETON["instance"] = _instance
    return _instance


def _goal_risk_threshold_from_signal(
    signal_signature: dict[str, float] | None,
    restorability_score: float,
    *,
    perceptual_context: dict[str, float] | None = None,
) -> float:
    """Leitet eine signal-/restorability-adaptive Goal-Risk-Schwelle ab.

    Faktoren ein (JND-Dichte, Bark-Energie-Verteilung), wenn verfügbar.
    """
    threshold = float(_GOAL_RISK_THRESHOLD)
    # ── Technische Faktoren ──
    if signal_signature:
        transient_ratio = float(signal_signature.get("transient_ratio", 0.0))
        crest_db = float(signal_signature.get("crest_db", 0.0))
        hf_ratio = float(signal_signature.get("hf_ratio", 0.0))
        if transient_ratio >= 0.01:
            threshold -= 0.05
        if crest_db >= 18.0:
            threshold -= 0.03
        if hf_ratio >= 0.12:
            threshold -= 0.02
    if restorability_score >= 80.0:
        threshold += 0.03
    elif restorability_score <= 40.0:
        threshold -= 0.03
    # ── §v10.101 Perzeptuelle Faktoren ──
    if perceptual_context:
        # JND-Dichte: viele hörbare Änderungen → konservativer (höhere Schwelle)
        _jnd_count = float(perceptual_context.get("jnd_audible_bands", 0))
        if _jnd_count > 15:
            threshold += 0.04  # Sehr viele hörbare Details → vorsichtig
        elif _jnd_count > 8:
            threshold += 0.02
        # Bark-Flatness: flaches Spektrum → mehr Risiko-Toleranz
        _bark_flatness = float(perceptual_context.get("bark_flatness", 0.5))
        if _bark_flatness > 0.7:
            threshold -= 0.03  # Komplexes Material → mehr erlauben
        # Vocal-Präsenz: Stimme → konservativer
        _vocal = float(perceptual_context.get("vocal_confidence", 0.0))
        if _vocal > 0.5:
            threshold += 0.04
    return float(np.clip(threshold, 0.40, 0.80))


# ---------------------------------------------------------------------------
# §0a Crossfire-Modus-Invariante — verbotene Phasen-Präfixe in Restoration
# ---------------------------------------------------------------------------
# Kanonische Quelle: backend.core.adaptive_phase_rescheduler._RESTORATION_FORBIDDEN
# (phase_21_exciter, phase_35_multiband_compression, phase_42_vocal_enhancement).
# Präfix-Matching (phase_21/phase_35/phase_42) ist robust gegen Namensvarianten
# (z. B. phase_21_exciter vs. phase_21_harmonic_exciter) — phase_21/35/42 sind
# eindeutig dem Exciter / der Multiband-Kompression / dem Vocal-Enhancement
# zugeordnet, daher ist Nummer-Präfix-Matching korrekt und sicher.
_RESTORATION_FORBIDDEN_PREFIXES: frozenset[str] | None = None


def _restoration_forbidden_prefixes() -> frozenset[str]:
    """Leitet §0a-verbotene Phasen-Nummer-Präfixe aus der kanonischen Quelle ab (cached).

    Vermeidet eine Parallelwelt-Definition: liest ``_RESTORATION_FORBIDDEN`` aus dem
    AdaptivePhaseRescheduler und reduziert jeden Eintrag auf sein ``phase_NN``-Präfix.
    Fällt bei Import-Fehler auf die drei kanonischen IDs zurück.
    """
    global _RESTORATION_FORBIDDEN_PREFIXES
    if _RESTORATION_FORBIDDEN_PREFIXES is not None:
        return _RESTORATION_FORBIDDEN_PREFIXES
    try:
        canonical = _load_symbol("backend.core.adaptive_phase_rescheduler", "_RESTORATION_FORBIDDEN")
    except Exception:
        canonical = frozenset({"phase_21_exciter", "phase_35_multiband_compression", "phase_42_vocal_enhancement"})
    prefixes: set[str] = set()
    for pid in canonical:
        parts = str(pid).split("_")
        if len(parts) >= 2 and parts[0] == "phase":
            prefixes.add(f"{parts[0]}_{parts[1]}")
    _RESTORATION_FORBIDDEN_PREFIXES = frozenset(prefixes)
    return _RESTORATION_FORBIDDEN_PREFIXES


# ---------------------------------------------------------------------------
# Semantische Typ-Taxonomie (§2.48)
# ---------------------------------------------------------------------------
# Jede Phase erhält eine Menge semantischer Tags.
# Neue Phasen: einfach eintragen — Konfliktregeln greifen automatisch.
# Präfix-Matching: "phase_03_denoise_ml" erbt Tags von "phase_03_denoise".

_PHASE_SEMANTICS: dict[str, frozenset[str]] = {
    # ── Subtraktiv: Rauschen / Artefakte entfernen ──────────────────────────
    "phase_01_click_removal": frozenset({"SUBTRACTIVE", "TRANSIENT_SUPPRESS"}),
    "phase_02_hum_removal": frozenset({"SUBTRACTIVE", "SPECTRAL_NOTCH"}),
    "phase_03_denoise": frozenset({"SUBTRACTIVE", "DENOISE", "BROADBAND"}),
    "phase_05_rumble_filter": frozenset({"SUBTRACTIVE", "LF_SUPPRESS"}),
    "phase_09_crackle_removal": frozenset({"SUBTRACTIVE", "TRANSIENT_SUPPRESS"}),
    "phase_18_noise_gate": frozenset({"SUBTRACTIVE", "GATING"}),
    "phase_20_reverb_reduction": frozenset({"SUBTRACTIVE", "REVERB"}),
    "phase_27_click_pop_removal": frozenset({"SUBTRACTIVE", "TRANSIENT_SUPPRESS"}),
    "phase_28_dc_offset_normalizer": frozenset({"SUBTRACTIVE", "DC"}),
    "phase_29_tape_hiss_reduction": frozenset({"SUBTRACTIVE", "DENOISE", "SPECTRAL"}),
    "phase_30_dc_offset_removal": frozenset({"SUBTRACTIVE", "DC"}),
    "phase_43_mp_senet_enhancement": frozenset({"SUBTRACTIVE", "DENOISE", "ML"}),
    "phase_49_advanced_dereverb": frozenset({"SUBTRACTIVE", "REVERB"}),
    # ── Additiv: Spektrum / Energie ergänzen ────────────────────────────────
    "phase_06_frequency_restoration": frozenset({"ADDITIVE", "FREQUENCY_EXT"}),
    "phase_07_harmonic_restoration": frozenset({"ADDITIVE", "HARMONIC"}),
    "phase_07_declipper": frozenset({"WAVEFORM_REPAIR", "ADDITIVE"}),  # §v10.18: Selbstkalibrierender Declipper
    "phase_11_transient_shaper": frozenset({"ADDITIVE", "TRANSIENT_ADD"}),
    "phase_21_harmonic_exciter": frozenset({"ADDITIVE", "HARMONIC"}),
    "phase_38_presence_boost": frozenset({"ADDITIVE", "FREQUENCY_EXT", "HF"}),
    "phase_55_diffusion_inpainting": frozenset({"ADDITIVE", "INPAINTING"}),
    # ── Dynamik-Expansion ───────────────────────────────────────────────────
    "phase_26_dynamic_range_expansion": frozenset({"DYNAMICS_EXPANDING"}),
    # ── Dynamik-Kompression ─────────────────────────────────────────────────
    "phase_10_compression": frozenset({"DYNAMICS_COMPRESSING"}),
    "phase_35_multiband_compression": frozenset({"DYNAMICS_COMPRESSING"}),
    "phase_54_multiband_dynamics": frozenset({"DYNAMICS_COMPRESSING"}),
    # ── Stereo: Korrigierend (muss vor Azimuth stehen — §7.2) ───────────────
    "phase_14_phase_correction": frozenset({"STEREO_CORRECTIVE", "PRE_AZIMUTH_REQUIRED"}),
    "phase_15_stereo_balance": frozenset({"STEREO_CORRECTIVE"}),
    # ── Stereo: Einengend ───────────────────────────────────────────────────
    "phase_33_stereo_width_limiter": frozenset({"STEREO_NARROWING"}),
    # ── Stereo: Erweiternd ──────────────────────────────────────────────────
    "phase_13_stereo_enhancement": frozenset({"STEREO_WIDENING"}),
    "phase_48_stereo_width_enhancer": frozenset({"STEREO_WIDENING"}),
    # ── Azimuth (benötigt PRE_AZIMUTH_REQUIRED davor — Spec §7.2) ───────────
    "phase_25_azimuth_correction": frozenset({"AZIMUTH", "NEEDS_PRE_AZIMUTH"}),
    # ── Groove-Echo (muss Reverb-Reduktion vorangehen) ──────────────────────
    "phase_61_groove_echo_cancellation": frozenset({"PRE_REVERB_REQUIRED"}),
    # ── Mastering / Output (EBU R128-Kette) ─────────────────────────────────
    "phase_16_final_eq": frozenset({"MASTERING", "EQ"}),
    "phase_17_mastering_polish": frozenset({"MASTERING", "POLISH"}),
    "phase_40_loudness_normalization": frozenset({"MASTERING", "LOUDNESS_NORMALIZATION"}),
    "phase_47_truepeak_limiter": frozenset({"MASTERING", "LIMITER", "NEEDS_LOUDNESS"}),
    "phase_41_output_format_optimization": frozenset({"OUTPUT"}),
    # §v10.18: Phasen ohne Annotation nachgetragen (für Konfliktregeln und DAG-Reorder)
    "phase_08_transient_preservation": frozenset({"TRANSIENT", "WAVEFORM_REPAIR"}),  # Zölzer §8.1: vor NR
    "phase_19_de_esser": frozenset({"SUBTRACTIVE", "SIBILANCE"}),  # De-Essing nach Denoise
    "phase_28_surface_noise_profiling": frozenset({"SUBTRACTIVE", "NOISE_PROFILING"}),  # Profiling vor Gate
}

# ---------------------------------------------------------------------------
# Konfliktregeln — semantisch, nicht phasenname-gebunden (§2.48)
# ---------------------------------------------------------------------------
# Tupel (Auslöser-Tags, Ziel-Tags):
#   Wenn Phase A Auslöser-Tags hat UND Phase B Ziel-Tags hat UND B nach A kommt
#   → B wird supprimiert (§2.48 "suppress_later").
# Neue Konflikte: einfach eintragen — kein Phasen-Code ändern.

_CONFLICT_RULES: list[tuple[frozenset[str], frozenset[str]]] = [
    # Dynamik-Expansion → kein anschließendes Komprimieren (hebt Expansion auf)
    (frozenset({"DYNAMICS_EXPANDING"}), frozenset({"DYNAMICS_COMPRESSING"})),
    # Stereo-Einengung → kein anschließendes Erweitern (hebt Einengung auf)
    (frozenset({"STEREO_NARROWING"}), frozenset({"STEREO_WIDENING"})),
    # §v10.18 Declipper: Waveform-Repair → kein Komprimieren danach
    # (Compression hebt de-clippte Peaks wieder auf → kreisförmige Verarbeitung)
    (frozenset({"WAVEFORM_REPAIR"}), frozenset({"DYNAMICS_COMPRESSING"})),
]

# ---------------------------------------------------------------------------
# Reihenfolge-Constraints — deklarative Tabelle (§2.46 / §7.2)
# ---------------------------------------------------------------------------
# Tupel (Phase A, Phase B): A muss im Plan vor B stehen.
# Gilt nur wenn beide Phasen selektiert sind.

_ORDER_CONSTRAINTS: list[tuple[str, str]] = [
    # Spec §7.2 CAUSE_TO_PHASES azimuth_error: Phase-Korrektur vor Azimuth
    ("phase_14_phase_correction", "phase_25_azimuth_correction"),
    # Groove-Echo muss vor Reverb-Reduktion stehen (sonst Fehlidentifikation als Raumhall)
    ("phase_61_groove_echo_cancellation", "phase_20_reverb_reduction"),
    ("phase_61_groove_echo_cancellation", "phase_49_advanced_dereverb"),
    # EBU R128: LUFS-Normalisierung vor TruePeak-Limiter (§6.1)
    ("phase_40_loudness_normalization", "phase_47_truepeak_limiter"),
    # Spec §2.46 Carrier-Chain-Inversion: subtraktiv vor additiv
    # §v10.18 Declipper: muss VOR Denoising laufen (Godsill & Rayner 1998, §5.2).
    # Clipped samples haben keine HF-Information → Noise-Model auf Clips lernt falsch.
    ("phase_07_declipper", "phase_03_denoise"),
    ("phase_07_declipper", "phase_04_eq_correction"),  # EQ nicht auf Clips anwenden
    ("phase_07_declipper", "phase_06_frequency_restoration"),  # Freq-Restore auf sauberer Wellenform
    ("phase_03_denoise", "phase_07_harmonic_restoration"),
    ("phase_03_denoise", "phase_06_frequency_restoration"),
    ("phase_03_denoise", "phase_21_harmonic_exciter"),
    ("phase_29_tape_hiss_reduction", "phase_07_harmonic_restoration"),
    ("phase_29_tape_hiss_reduction", "phase_06_frequency_restoration"),
    ("phase_29_tape_hiss_reduction", "phase_21_harmonic_exciter"),
    # §v10.18 Wissenschaftliche Reihenfolge-Korrekturen:
    # Godsill & Rayner (1998) §5.2: impulsive repair → broadband NR
    ("phase_01_click_removal", "phase_03_denoise"),
    ("phase_09_crackle_removal", "phase_03_denoise"),
    # Schmalband → Breitband-Prinzip (Zölzer 2008, §9.2)
    ("phase_05_rumble_filter", "phase_03_denoise"),
    # Zölzer (2008) §8.1: Waveform-Repair vor Spectral-Processing
    ("phase_08_transient_preservation", "phase_03_denoise"),
    # De-Esser braucht sauberes Signal für Sibilanz-Detektion (4-12 kHz)
    ("phase_03_denoise", "phase_19_de_esser"),
    # Noise Gate NACH Denoise (Cano et al. 2013: Gate vor NR erzeugt Pumping)
    ("phase_03_denoise", "phase_18_noise_gate"),
    # Surface Noise Profiling vor Gate (vollständiges Rauschprofil nötig)
    ("phase_28_surface_noise_profiling", "phase_18_noise_gate"),
    # Mastering: Stereo-Widening vor finalem EQ
    ("phase_13_stereo_enhancement", "phase_16_final_eq"),
]

# ---------------------------------------------------------------------------
# Goal-Risiko-Schwelle + schützende Phasen (§GoalRisk, ExzellenzDenker-Integration)
# ---------------------------------------------------------------------------
# Prophylaktische Phasen-Injektion wenn ExzellenzDenker.prognostiziere() Risiko meldet.
_GOAL_RISK_THRESHOLD: float = 0.60
# Risikoschwelle ∈ [0, 1] ab der eine schützende Phase injiziert wird.

_GOAL_RISK_PROTECTIVE_PHASES: dict[str, str] = {
    "natuerlichkeit": "phase_03_denoise",  # Rauschen → Natürlichkeit
    "authentizitaet": "phase_03_denoise",  # Rauschen → Authentizität
    "brillanz": "phase_06_frequency_restoration",  # HF-Verlust → Brillanz
    "timbre": "phase_07_harmonic_restoration",  # HF-Verlust → Timbre
    "groove": "phase_09_crackle_removal",  # Transient-Armut → Groove
    "micro_dynamics": "phase_29_tape_hiss_reduction",  # Rausch-Maskierung → MikroDynamik
    "artikulation": "phase_24_dropout_repair",  # Dropout → Artikulation
}


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class PhasePlan:
    """Semantisch aufgelöster, konfliktfreier Phase-Ausführungsplan.

    Wird von AurikDenker als precomputed_phase_plan an UV3.restore() übergeben.
    UV3 verwendet dann diesen Plan direkt — keine eigene Orchestrierung mehr.
    """

    phases: list[str]
    """Geordnete, deduplizierte, konfliktfreie Phasenliste für UV3."""

    suppressed: dict[str, str] = field(default_factory=dict)
    """phase_id → Grund der Supprimierung (§2.48 Konflikt-Guard)."""

    ordering_applied: list[tuple[str, str]] = field(default_factory=list)
    """Angewandte Reihenfolge-Constraints (before, after)."""

    conflict_notes: list[str] = field(default_factory=list)
    """Menschenlesbare Konflikt-Protokoll-Einträge."""

    semantic_annotations: dict[str, list[str]] = field(default_factory=dict)
    """phase_id → Semantik-Tags (für Diagnose/Logging)."""

    material: str = ""
    mode: str = ""

    phase_quality_tiers: dict[str, str] = field(default_factory=dict)
    """Qualitäts-Tier pro Phase: 'maximum' | 'quality' | 'fast' (aus StrategieDenker).
    Advisory — wird von UV3 für zukünftige per-Phase-Algorithmus-Selektion genutzt."""

    policy_hints: dict[str, Any] = field(default_factory=dict)

    surgical_routing: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    """§2.59: phase_id → [(start_s, end_s), ...] für chirurgische Verarbeitung.
    Leerer Eintrag = Phase läuft global über das gesamte Audio."""
    """Hörbezogene Hinweise für das zentrale restoration_policy_profile."""

    # ── §v10.2 Repair-Policy (zentrale Entscheidung, vom AurikDenker konsumiert) ──
    repair_policy: dict[str, str] = field(default_factory=dict)
    """Reparatur-Entscheidungen für den AurikDenker._run_rest()-Closure.

    Von PhaseInteractionDenker._determine_repair_policy() befüllt, basierend auf
    defect_scores, material, era, signal_signature und transfer_chain.

    Jeder Key bezeichnet eine Reparatur-Operation, der Value den Eingriffsgrad:

        clicks:   "off" | "mild" | "aggressive"
        hum:      "off" | "mild" | "aggressive"
        clipping: "off" | "mild" | "aggressive"

    Der AurikDenker konsumiert diese Policy deterministisch — keine eigenen
    Schwellen-Entscheidungen mehr im Closure.  Der ReparaturDenker erhält
    pro Operation einen booleschen Schalter (True/False) plus ggf. den
    Eingriffsgrad als severity-proportionalen Parameter.

    Aus der Policy werden im AurikDenker die booleschen Flags abgeleitet:
        remove_clicks    = policy != "off"
        remove_hum       = policy != "off"
        repair_clipping  = policy != "off"

    Der Eingriffsgrad ("mild" vs "aggressive") wird als separater Hinweis
    an den ReparaturDenker durchgereicht (z. B. defect_scores-Skalierung).
    """

    @property
    def is_valid(self) -> bool:
        """True wenn der Plan mindestens eine Phase enthält."""
        return bool(self.phases)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PhaseInteractionDenker:
    """Steuernde Intelligenz für Phasen-Orchestrierung.

    Workflow:
      1. _select_via_uv3(): UV3 selektiert Phasen (Werkzeug, nicht Orchestrator)
      2. _annotate(): semantische Typ-Annotation
      3. _resolve_conflicts(): Typ-Paar-basierte Supprimierung (§2.48)
      4. _apply_order_constraints(): deklarative Reihenfolge-Constraints (§2.46)
      5. PhasePlan zurückgeben → AurikDenker → UV3.restore(precomputed_phase_plan=...)
    """

    def plan(
        self,
        defect_result: Any,
        material: str = "unknown",
        mode: str = "quality",
        *,
        chain_info: Any | None = None,
        chain_result: Any | None = None,
        defekt_hint: Any | None = None,
        audio: np.ndarray | None = None,
        sr: int = 48000,
        causal_plan: Any | None = None,
        restorability_score: float = 70.0,
        pipeline_confidence: Any | None = None,
        goal_risk_map: dict[str, float] | None = None,
        strategie_plan: Any | None = None,
        signal_signature: dict[str, float] | None = None,
        sections: list[tuple[float, float, str]] | None = None,
    ) -> PhasePlan:
        """Erstellt einen konfliktfreien, semantisch geordneten Phasenplan.

        Bei Fehlern wird ein leerer PhasePlan zurückgegeben — UV3 übernimmt
        dann die Selektion autonom (fail-safe).

        Args:
            defect_result:        DefectAnalysisResult vom DefektDenker.
            material:             Erkanntes Trägermedium (z. B. "vinyl").
            mode:                 Qualitätsmodus ("restoration", "studio2026").
            chain_info:           Tonträgerketten-Dict (optional).
            defekt_hint:          Heuristik-Phasenliste vom DefektDenker (optional).
            audio:                Eingabe-Audio für interne UV3-Analysen (optional).
            sr:                   Samplerate in Hz.
            causal_plan:          CausalDefectReasoner-Ergebnis (optional).
            restorability_score:  Restorability 0–100 für UV3-Optimizer.
            pipeline_confidence:  UQ-Konfidenz für UV3-Optimizer (optional).

        Returns:
            PhasePlan mit geordneter, konfliktfreier Phasenliste.
            Bei Fehler: leerer PhasePlan (UV3 übernimmt Selektion).
        """
        if defect_result is None:
            logger.debug("PhaseInteractionDenker: kein defect_Ergebnis — UV3 übernimmt Selektion.")
            return PhasePlan(
                phases=[], material=material, mode=mode, conflict_notes=["Kein defect_result — UV3-Fallback."]
            )
        try:
            return self._plan_internal(
                defect_result=defect_result,
                material=material,
                mode=mode,
                chain_info=chain_info,
                chain_result=chain_result,
                defekt_hint=defekt_hint,
                audio=audio,
                sr=sr,
                causal_plan=causal_plan,
                restorability_score=restorability_score,
                pipeline_confidence=pipeline_confidence,
                goal_risk_map=goal_risk_map,
                strategie_plan=strategie_plan,
                signal_signature=signal_signature,
                sections=sections,
            )
        except Exception as exc:
            logger.warning(
                "PhaseInteractionDenker.plan() fehlgeschlagen: %s — UV3 übernimmt Selektion.",
                exc,
            )
            return PhasePlan(
                phases=[],
                material=material,
                mode=mode,
                conflict_notes=[f"PID fehlgeschlagen (fail-safe): {exc}"],
            )

    # ── Interne Pipeline ─────────────────────────────────────────────────────

    def _plan_internal(
        self,
        *,
        defect_result: Any,
        material: str,
        mode: str,
        chain_info: Any | None,
        chain_result: Any | None,
        defekt_hint: Any | None,
        audio: np.ndarray | None,
        sr: int,
        causal_plan: Any | None,
        restorability_score: float,
        pipeline_confidence: Any | None,
        goal_risk_map: dict[str, float] | None,
        strategie_plan: Any | None,
        signal_signature: dict[str, float] | None,
        sections: list[tuple[float, float, str]] | None = None,
    ) -> PhasePlan:
        # 1. Phase-Selektion via UV3 (UV3 = Werkzeug, nicht Orchestrator)
        uv3_phases = self._select_via_uv3(
            defect_result=self._with_policy_material(defect_result, material),
            mode=mode,
            chain_info=chain_info,
            defekt_hint=defekt_hint,
            audio=audio,
            sr=sr,
            causal_plan=causal_plan,
            restorability_score=restorability_score,
            pipeline_confidence=pipeline_confidence,
        )

        if not uv3_phases:
            return PhasePlan(
                phases=[],
                material=material,
                mode=mode,
                conflict_notes=["UV3-Selektion lieferte keine Phasen — UV3-Fallback."],
            )

        merged_phases = list(uv3_phases)
        injected_notes: list[str] = []

        # §v10.303.4 Aktiver Confidence-Check: Auch ohne Cache (Song 1) wird
        # der adaptive Threshold angewendet. Verhindert Song-1-Inkonsistenz.
        _mat_conf = pipeline_confidence
        if isinstance(_mat_conf, (int, float)) and isinstance(restorability_score, (int, float)):
            try:
                from backend.core.unified_restorer_v3 import UnifiedRestorerV3

                _threshold = UnifiedRestorerV3._compute_low_confidence_threshold(
                    restorability_score=float(restorability_score)
                )
                if _mat_conf < _threshold:
                    _family_map = getattr(UnifiedRestorerV3, "_PHASE_INTERVENTION_CLASS", {})
                    _no_skip = {
                        "subtractive_cleanup",
                        "reconstruction_inpainting",
                        "time_pitch_transport",
                        "distortion_repair",
                        "dynamics_repair",
                        "tonal_restoration",
                        "tonal_mastering",
                        "sibilance_control",
                        "noise_reduction",
                        "vocal_enhancement",
                        "stereo_phase_geometry",
                    }
                    _pre = len(merged_phases)
                    merged_phases = [_p for _p in merged_phases if _family_map.get(_p, "general") in _no_skip]
                    _post = len(merged_phases)
                    if _post < _pre:
                        logger.info(
                            "§v10.303.4 PID-Confidence-Strip: %d/%d Phasen entfernt "
                            "(conf=%.3f < Schwelle=%.3f, rs=%.1f)",
                            _pre - _post,
                            _pre,
                            _mat_conf,
                            _threshold,
                            restorability_score,
                        )
                        injected_notes.append(
                            f"§v10.303.4 Confidence-Strip: {_pre - _post} Phasen "
                            f"(conf={_mat_conf:.3f} < thr={_threshold:.3f})"
                        )
            except Exception:
                logger.debug("Verarbeitungsschritt_interaction_denker.py:567: Silent exception absorbed", exc_info=True)

        # §v10.303.3 Denker-Feedback-Loop: Familien die UV3 beim letzten Run
        # bei Low-Confidence gestrippt hat, werden gar nicht erst eingeplant.
        _stripped_cache = get_low_confidence_stripped_families()
        if _stripped_cache:
            try:
                from backend.core.unified_restorer_v3 import UnifiedRestorerV3

                _family_map = getattr(UnifiedRestorerV3, "_PHASE_INTERVENTION_CLASS", {})
                _pre_count = len(merged_phases)
                merged_phases = [_p for _p in merged_phases if _family_map.get(_p, "general") not in _stripped_cache]
                _post_count = len(merged_phases)
                if _post_count < _pre_count:
                    logger.info(
                        "§v10.303.3 Denker-Feedback: %d/%d Phasen aus Plan gestrichen (gelernte useless families: %s)",
                        _pre_count - _post_count,
                        _pre_count,
                        ", ".join(sorted(_stripped_cache)),
                    )
                    injected_notes.append(
                        f"§v10.303.3 Feedback-Strip: {_pre_count - _post_count} Phasen "
                        f"aus {len(_stripped_cache)} gelernten Familien"
                    )
            except Exception:
                logger.debug("Verarbeitungsschritt_interaction_denker.py:592: Silent exception absorbed", exc_info=True)

        # 2. Ketten-Pflicht-Phasen (§2.46 Feature 1: TontraegerketteDenker)
        # Injiziert must_have_phases aus der erkannten Trägerkette,
        # unabhängig vom DefectScanner-Score (§6.2a Komplement).
        if chain_result is not None:
            try:
                _get_tontraegerkette_denker = _load_symbol(
                    "denker.tontraegerkette_denker", "get_tontraegerkette_denker"
                )
                _chain_plan = _get_tontraegerkette_denker().leite_phasen_ab(chain_result)
                for must in _chain_plan.must_have_phases:
                    if must not in merged_phases:
                        merged_phases.append(must)
                        note = f"§6.2a Ketten-Injektion [{_chain_plan.chain_string}]: {must}"
                        injected_notes.append(note)
                        logger.info("PhaseInteractionDenker %s", note)
            except Exception as _ce:
                logger.debug("PhaseInteractionDenker: leite_phasen_ab() fehlgeschlagen: %s", _ce)

        # §2.5a Material-Kritische-Phasen-Injektion
        # Erzwingt Reparatur-Phasen für Bandmaterial (Cassette/Tape/Reel)
        # unabhängig von Defekt-Confidence — Bandkopfdefekte sind inhärent.
        _MATERIAL_CRITICAL: dict[str, list[str]] = {
            "cassette": [
                "phase_14_phase_correction",
                "phase_25_azimuth_correction",
                "phase_56_spectral_band_gap_repair",
                "phase_24_dropout_repair",
            ],
            "tape": [
                "phase_14_phase_correction",
                "phase_25_azimuth_correction",
                "phase_56_spectral_band_gap_repair",
                "phase_24_dropout_repair",
            ],
            "reel_tape": [
                "phase_14_phase_correction",
                "phase_25_azimuth_correction",
                "phase_56_spectral_band_gap_repair",
            ],
            "vinyl": [
                "phase_09_crackle_removal",
                "phase_28_surface_noise_profiling",
            ],
            "shellac": [
                "phase_09_crackle_removal",
                "phase_28_surface_noise_profiling",
            ],
        }
        _mat_critical = _MATERIAL_CRITICAL.get(str(material or "").lower(), [])
        for _mc_phase in _mat_critical:
            if _mc_phase not in merged_phases:
                merged_phases.append(_mc_phase)
                note = f"§2.5a Material-Kritisch [{material}]: {_mc_phase}"
                injected_notes.append(note)
                logger.info("PhaseInteractionDenker %s", note)

        # 3. Goal-Risk-Injektion (§GoalRisk Feature 2: ExzellenzDenker.prognostiziere())
        # Injiziert schützende Phasen wenn ein Musical Goal mit Risiko >= Schwelle bedroht ist.
        if goal_risk_map:
            _goal_threshold = _goal_risk_threshold_from_signal(signal_signature, restorability_score)
            for goal, risk in goal_risk_map.items():
                if risk >= _goal_threshold:
                    protective = _GOAL_RISK_PROTECTIVE_PHASES.get(goal)
                    if protective and protective not in merged_phases:
                        merged_phases.append(protective)
                        note = f"§GoalRisk-Injektion [{goal}={risk:.2f}, thr={_goal_threshold:.2f}]: {protective}"
                        injected_notes.append(note)
                        logger.info("PhaseInteractionDenker %s", note)

        # 3b. Wissenschaftsbasierte Signal-Injektion (advisory, no-harm):
        # schützt Transienten/Artikulation und Sibilanz bei risikoreichem Material.
        if signal_signature:
            transient_ratio = float(signal_signature.get("transient_ratio", 0.0))
            hf_ratio = float(signal_signature.get("hf_ratio", 0.0))
            if transient_ratio >= 0.01 and "phase_08_transient_preservation" not in merged_phases:
                merged_phases.append("phase_08_transient_preservation")
                note = f"§Signal-Injektion [transient_ratio={transient_ratio:.4f}]: phase_08_transient_preservation"
                injected_notes.append(note)
                logger.info("PhaseInteractionDenker %s", note)
            if hf_ratio >= 0.12 and "phase_19_de_esser" not in merged_phases:
                merged_phases.append("phase_19_de_esser")
                note = f"§Signal-Injektion [hf_ratio={hf_ratio:.3f}]: phase_19_de_esser"
                injected_notes.append(note)
                logger.info("PhaseInteractionDenker %s", note)

        # 3c. §0a Crossfire-Modus-Invariante (Defense-in-Depth):
        # phase_21 (Exciter), phase_35 (Multiband-Kompression), phase_42
        # (Vocal-Enhancement) dürfen in Restoration NIEMALS in einen PhasePlan
        # gelangen — auch nicht über Ketten-/Goal-Risk-/Signal-Injektion oben.
        # UV3 und der Rescheduler erzwingen denselben Guard; der
        # PhaseInteractionDenker als primäre planbildende Schicht erzwingt ihn
        # hier am eigenen Ausgabe-Rand, bevor Annotation/Konflikt/Ordering laufen.
        merged_phases, _forbidden_removed = self._strip_restoration_forbidden(merged_phases, mode)
        for _fr in _forbidden_removed:
            note = f"§0a Crossfire-Guard [restoration]: {_fr} entfernt (verbotene Phase)"
            injected_notes.append(note)
            logger.info("PhaseInteractionDenker %s", note)

        # §CODEC: Extrahiere Terminal-Codec für Denker-gesteuerte Kalibrierung
        # Keine hartcodierten Phasen-Listen — Denker entscheidet dynamisch.
        _terminal_codec: str | None = None
        _codec_avg_discount: float = 1.0
        _scores = getattr(defect_result, "scores", {}) or {}
        _discounts: list[float] = []
        for _ds in _scores.values() if isinstance(_scores, dict) else []:
            if hasattr(_ds, "metadata"):
                _tc = (_ds.metadata or {}).get("chain_contamination_terminal_codec")
                if _tc and not _terminal_codec:
                    _terminal_codec = str(_tc)
                _d = float((_ds.metadata or {}).get("chain_contamination_discount", 1.0))
                if _d < 0.99:
                    _discounts.append(_d)
        if _discounts:
            _codec_avg_discount = sum(_discounts) / len(_discounts)
        if _terminal_codec:
            injected_notes.append(f"§CODEC Context: terminal={_terminal_codec} discount={_codec_avg_discount:.2f}")
            logger.info(
                "PhaseInteractionDenker §CODEC Context: %s (avg discount=%.2f)", _terminal_codec, _codec_avg_discount
            )

        # 4. Semantische Annotation
        annotations = self._annotate(merged_phases)

        # 5. Semantische Konflikterkennung + Auflösung (§2.48)
        resolved, suppressed, conflict_notes = self._resolve_conflicts(merged_phases, annotations)

        # 6. Reihenfolge-Constraints erzwingen (§2.46 / §7.2)
        ordered, ordering_applied = self._apply_order_constraints(resolved)

        # 6a. §v10.706 Denker-IQ: Material-Fremdlauf-Detektion
        # Warnt bei Phasen die für anderes Medium designed sind, reduziert
        # aber NICHT die Stärke — ActiveIntervention lehnt wirkungslose
        # Phasen bereits ab. Phase_60/61 erkennen selbst ob Groove-Defekte
        # vorhanden sind. Phase_64 funktioniert auch auf Vinyl (Tape-Master).
        _MATERIAL_FOREIGN_PHASES: dict[str, frozenset[str]] = {
            "cassette": frozenset({"phase_60_inner_groove_distortion_repair", "phase_61_groove_echo_cancellation"}),
            "reel_tape": frozenset({"phase_60_inner_groove_distortion_repair", "phase_61_groove_echo_cancellation"}),
            "tape": frozenset({"phase_60_inner_groove_distortion_repair", "phase_61_groove_echo_cancellation"}),
            "cd_digital": frozenset(
                {
                    "phase_60_inner_groove_distortion_repair",
                    "phase_61_groove_echo_cancellation",
                    "phase_64_tape_splice_repair",
                }
            ),
        }
        _mat_foreign = _MATERIAL_FOREIGN_PHASES.get(material.lower(), frozenset())
        _foreign_found = [p for p in ordered if p in _mat_foreign]
        if _foreign_found:
            conflict_notes.append(
                f"§v10.706 Material-Fremdlauf: {len(_foreign_found)} Phase(n) "
                f"für anderes Medium designed ({', '.join(_foreign_found)}) — "
                f"laufen auf {material} (Defekte können durch Kette transferiert sein)"
            )
            logger.info(
                "PhaseInteractionDenker: %d Material-fremde Verarbeitungsschritt(n) erkannt: %s "
                "(Kette kann Defekte übertragen haben — Phasen laufen normal)",
                len(_foreign_found),
                _foreign_found,
            )

        # 7. Qualitäts-Tier je Phase (Feature 3: StrategieDenker.schaetze_phasen_tier())
        # Advisory-only — UV3 kann diese Infos für per-Phase-Algorithmus-Selektion nutzen.
        phase_quality_tiers: dict[str, str] = {}
        if strategie_plan is not None:
            try:
                _get_strategie_denker = _load_symbol("denker.strategie_denker", "get_strategie_denker")
                phase_quality_tiers = _get_strategie_denker().schaetze_phasen_tier(
                    strategie_plan,
                    ordered,
                    restorability_score=restorability_score,
                )
                if phase_quality_tiers:
                    logger.debug(
                        "PhaseInteractionDenker: %d Tier-Empfehlungen (%d maximum, %d quality, %d fast)",
                        len(phase_quality_tiers),
                        sum(1 for t in phase_quality_tiers.values() if t == "maximum"),
                        sum(1 for t in phase_quality_tiers.values() if t == "quality"),
                        sum(1 for t in phase_quality_tiers.values() if t == "fast"),
                    )
            except Exception as _te:
                logger.debug("PhaseInteractionDenker: schaetze_phasen_tier() fehlgeschlagen: %s", _te)

        policy_hints: dict[str, Any] = {
            "phase_count": len(ordered),
            "suppressed_count": len(suppressed),
            "injected_count": len(injected_notes),
            "semantic_density": float(np.clip(len(ordered) / 24.0, 0.0, 1.0)),
            "phase_quality_tiers": dict(phase_quality_tiers),
        }
        if goal_risk_map:
            policy_hints["goal_risk_map"] = {str(k): float(v) for k, v in goal_risk_map.items()}
        if signal_signature:
            policy_hints["signal_signature"] = {str(k): float(v) for k, v in signal_signature.items()}

        # ── 8. Repair-Policy (zentral statt inline im AurikDenker) ────────────
        repair_policy = self._determine_repair_policy(
            material=material,
            defekt_hint=defekt_hint,
            defect_result=defect_result,
            signal_signature=signal_signature,
            chain_info=chain_info,
            mode=mode,
        )
        logger.info(
            "PhaseInteractionDenker Repair-Policy: clicks=%s hum=%s clipping=%s",
            repair_policy.get("clicks", "?"),
            repair_policy.get("hum", "?"),
            repair_policy.get("clipping", "?"),
        )

        logger.info(
            "PhaseInteractionDenker: %d→%d Phasen | "
            "+%d injiziert | %d supprimiert | "
            "%d Ordnungsänderungen | material=%s Betriebsart=%s",
            len(uv3_phases),
            len(ordered),
            len(injected_notes),
            len(suppressed),
            len(ordering_applied),
            material,
            mode,
        )
        for note in conflict_notes:
            logger.info("§2.48 Konflikt-Guard: %s", note)

        # §U: Phase-Ordering-Intelligence — akustische Kopplungen optimieren
        try:
            from backend.core.phase_intelligence import PhaseOrderIntelligence

            _poi = PhaseOrderIntelligence()
            _order_result = _poi.optimize(list(ordered))
            if _order_result.changes:
                ordered = _order_result.optimized_order
                logger.info(
                    "§U Verarbeitungsschritt-Ordering: %d Änderungen (Wert=%.2f)",
                    len(_order_result.changes),
                    _order_result.score,
                )
                for _ch in _order_result.changes[:5]:
                    logger.debug("  §U %s: Pos %s→%s — %s", _ch["phase"], _ch["from_pos"], _ch["to_pos"], _ch["reason"])
                conflict_notes.append(
                    f"§U PhaseOrder: {len(_order_result.changes)} akustische "
                    f"Kopplungen optimiert (score={_order_result.score:.2f})"
                )
        except Exception as _u_exc:
            logger.debug("§U Verarbeitungsschritt-Ordering nicht verfügbar: %s", _u_exc)

        # ── §2.60 Denker-Intelligenz: PhaseEffectCatalog → Intensitäten ────
        try:
            from backend.core.phase_effect_catalog import get_phase_effect_catalog

            _catalog = get_phase_effect_catalog()
            _audio_ctx = {
                "snr_db": (signal_signature or {}).get("snr_db"),
                "panns_singing": (signal_signature or {}).get("panns_singing", 0.0),
                "bandwidth_hz": (signal_signature or {}).get("bandwidth_hz"),
                "era_decade": (signal_signature or {}).get("era_decade", 1970),
                "material_type": str(material).lower() if material else "unknown",
                "restorability": restorability_score / 100.0 if restorability_score else 0.5,
                "defect_severity": (signal_signature or {}).get("severity", 0.5),
                # §2.60 L1-Max: Alle Signal-Signature-Felder
                "crest_db": (signal_signature or {}).get("crest_db", 12.0),
                "hf_ratio": (signal_signature or {}).get("hf_ratio", 0.0),
                "transient_ratio": (signal_signature or {}).get("transient_ratio", 0.0),
                "micro_dynamic_db": (signal_signature or {}).get("micro_dynamic_db", 6.0),
                "rms_dbfs": (signal_signature or {}).get("rms_dbfs", -20.0),
                "chain_has_cassette": "cassette" in str(chain_info.get("chain_str", "")).lower()
                if chain_info
                else False,
                "chain_has_mp3": "mp3" in str(chain_info.get("chain_str", "")).lower() if chain_info else False,
                "pipeline_confidence": pipeline_confidence if pipeline_confidence else 0.75,
                "defect_count_total": len(defect_result.scores)
                if defect_result and hasattr(defect_result, "scores")
                else 0,
                # §CODEC: Denker-Kalibrierung codec-aware
                "terminal_codec": _terminal_codec,
                "codec_avg_discount": _codec_avg_discount,
            }
            _calibration = _catalog.calibrate_all(ordered, _audio_ctx)
            policy_hints["phase_calibration"] = _calibration
            _n_cal = sum(1 for v in _calibration.values() if v != 1.0)
            # ── §2.60.1 Fahrplan: Denker als Dirigent ────────────────────
            try:
                from backend.core.fahrplan import build_fahrplan

                _goal_prios = {k: float(v) for k, v in (goal_risk_map or {}).items()}
                _fahrplan = build_fahrplan(
                    phase_ids=list(ordered),
                    sections=sections if sections else [],
                    goal_priorities=_goal_prios,
                    phase_effect_catalog=None,
                    audio_ctx=_audio_ctx,
                )
                policy_hints["fahrplan"] = _fahrplan
                logger.info(
                    "§2.60.1 Fahrplan: %d Phasen, %d Sektionen → %s",
                    len(_fahrplan.phase_order),
                    len(_fahrplan.sections),
                    _fahrplan.note,
                )
            except Exception:
                logger.warning("Verarbeitungsschritt_interaction_denker.py::unbekannter Ersatzpfad", exc_info=True)
            if _n_cal > 0:
                logger.info(
                    "§2.60 Denker-Kalibrierung: %d/%d Phasen intensitäts-angepasst",
                    _n_cal,
                    len(ordered),
                )
        except Exception:
            logger.warning("Verarbeitungsschritt_interaction_denker.py::unbekannter Ersatzpfad", exc_info=True)

        # ── §ROADMAP-4: Dynamic Phase Ordering (DAG) ──
        # Sortiert Phasen topologisch nach Frequenzbereich-Überlappungen,
        # material-abhängig: EQ vor Denoise bei bandbreitenbegrenztem Material
        # (Shellac), Denoise vor EQ bei rauschdominiertem Material (Tape).
        try:
            ordered = self._dag_reorder(ordered, material=material)
            ordering_applied = ["dag_reorder"] + (ordering_applied or [])  # type: ignore[assignment]
            conflict_notes.append(f"§ROADMAP-4 DAG-Reorder: {len(ordered)} Phasen topologisch sortiert")
        except Exception as _dag_exc:
            logger.debug("§ROADMAP-4 DAG-Reorder nicht blockierend: %s", _dag_exc)

        # §IV (copilot-instructions.md) + §2.46/§7.2: HARTE Reihenfolge-Constraints
        # (EBU R128: LUFS vor RLP; RLP immer letztes Glied der Export-Kette) müssen
        # auch nach §U- und DAG-Optimierungen gelten. Weiche Optimierungen dürfen
        # harte Constraints nie verletzen → finaler Durchsetzungs-Pass.
        ordered, _final_order_applied = self._apply_order_constraints(ordered)
        if _final_order_applied:
            ordering_applied = list(_final_order_applied) + (ordering_applied or [])
            conflict_notes.append(
                f"§ORDER final: {len(_final_order_applied)} harte Constraints nach §U/DAG wiederhergestellt"
            )

        return PhasePlan(
            phases=ordered,
            suppressed=suppressed,
            ordering_applied=ordering_applied,
            conflict_notes=conflict_notes + injected_notes,
            semantic_annotations={p: sorted(annotations.get(p, frozenset())) for p in ordered},
            phase_quality_tiers=phase_quality_tiers,
            policy_hints=policy_hints,
            repair_policy=repair_policy,
            material=material,
            mode=mode,
        )

    @staticmethod
    def _determine_repair_policy(
        *,
        material: str,
        defekt_hint: Any | None,
        defect_result: Any,
        signal_signature: dict[str, float] | None,
        chain_info: Any | None,
        mode: str,
    ) -> dict[str, str]:
        """Zentrale Repair-Policy: clicks / hum / clipping → "off" | "mild" | "aggressive".

        Ersetzt die bisherige inline-Heuristik im AurikDenker._run_rest()-Closure
        (drei separate boolesche Entscheidungen mit Hardcoded-Schwellen 0.3/0.3/0.4).

        Entscheidungskriterien (nach Wichtigkeit):
          1. recommended_phases aus DefektDenker (ph01/ph09/ph27 → clicks, ph02 → hum, ph23/ph06 → clipping)
          2. overall_severity (material-adaptive Schwelle)
          3. Material-Historie (historische/fragile Materialien sensibler)
          4. signal_signature (transient_ratio > 1% → clicks sensibler)
          5. Mode (studio2026 → konservativer)
          6. Codec-Chain (mp3/aac → mildere click_iqr)
        """
        # ── recommended_phases aus defekt_hint extrahieren ──────────────
        _ph: set[str] = set()
        # §2.59: Chirurgische Defekte — Phasen die NUR lokal arbeiten müssen
        _surgical_defects: list[str] = []
        if defekt_hint is not None:
            try:
                _ph = set(defekt_hint.get("recommended_phases", []) or [])
                _surgical_defects = list(defekt_hint.get("surgical_defect_types", []) or [])
            except Exception:
                logger.debug("_determine_repair_policy: silent except suppressed", exc_info=True)
        if _surgical_defects:
            logger.info(
                "PhaseInteractionDenker: %d chirurgische Defekte erkannt — "
                "diese Phasen werden PRIORISIERT und NICHT supprimiert: %s",
                len(_surgical_defects),
                ", ".join(_surgical_defects[:5]),
            )
        # Fallback: aus defect_result extrahieren
        if not _ph and defect_result is not None:
            try:
                _ph = set(getattr(defect_result, "recommended_phases", None) or [])
            except Exception:
                logger.debug("_determine_repair_policy: silent except suppressed", exc_info=True)

        # ── overall_severity ────────────────────────────────────────────
        _sev: float = 1.0
        if defect_result is not None:
            try:
                _sev = float(getattr(defect_result, "overall_severity", 1.0))
            except Exception:
                _sev = 1.0

        # ── Material-adaptive Schwellen ─────────────────────────────────
        _historical_or_fragile = material.lower() in {
            "wax_cylinder",
            "shellac",
            "lacquer_disc",
            "wire_recording",
            "vinyl",
            "tape",
            "reel_tape",
            "cassette",
            "cassette_dolby_b",
            "cassette_dolby_c",
            "cassette_dolby_s",
        }
        _modern_digital = material.lower() in {
            "cd_digital",
            "dat",
            "aac",
            "mp3_high",
            "streaming",
        }
        # Historische Materialien: niedrigere Schwellen (früherer Eingriff)
        _click_threshold = 0.25 if _historical_or_fragile else 0.30
        _hum_threshold = 0.25 if _historical_or_fragile else 0.30
        _clip_threshold = 0.35 if _historical_or_fragile else 0.40
        # Studio 2026: konservativer (höhere Schwellen)
        if mode in ("studio2026", "maximum"):
            _click_threshold += 0.05
            _hum_threshold += 0.05
            _clip_threshold += 0.05
        # Digitale Quellen: nur bei explizitem Phasen-Hinweis
        if _modern_digital and not _ph:
            return {"clicks": "off", "hum": "off", "clipping": "off"}

        # §v10.47 Restorability-Modulation: schlechtere Qualität → niedrigere Schwellen
        # (früherer Eingriff bei stark degradiertem Material)
        _rs = float(
            getattr(defect_result, "restorability_score", 0)
            or getattr(getattr(defect_result, "metadata", None), "restorability_score", 0)
            or 50.0
        )
        _rest_factor = float(np.clip(1.0 - (_rs / 100.0) * 0.50, 0.75, 1.0))
        _click_threshold *= _rest_factor
        _hum_threshold *= _rest_factor
        _clip_threshold *= _rest_factor

        # ── Signal-Signatur-Fehlerjustage ───────────────────────────────
        if signal_signature:
            _tr = float(signal_signature.get("transient_ratio", 0.0))
            _cr = float(signal_signature.get("crest_db", 0.0))
            if _tr >= 0.01:
                _click_threshold -= 0.03  # Viele Transienten → clicks früher behandeln
            if _cr >= 18.0:
                _click_threshold -= 0.02  # Hoher Crest-Faktor → clicks hörbarer

        # ── Einzelentscheidungen ────────────────────────────────────────
        _has_click_phases = bool(
            _ph
            & {
                "phase_01_click_removal",
                "phase_09_crackle_removal",
                "phase_27_click_pop_removal",
            }
        )
        _has_hum_phases = bool(_ph & {"phase_02_hum_removal"})
        _has_clip_phases = bool(
            _ph
            & {
                "phase_07_declipper",  # §v10.18: Primär-Declipper
                "phase_23_spectral_repair",
                "phase_06_frequency_restoration",
            }
        )

        def _decide(
            has_phase_hint: bool,
            severity: float,
            threshold: float,
        ) -> str:
            if has_phase_hint and severity >= threshold * 1.5:
                return "aggressive"
            if has_phase_hint or severity >= threshold:
                return "mild"
            if severity >= threshold * 0.8:
                return "mild"
            return "off"

        policy = {
            "clicks": _decide(_has_click_phases, _sev, _click_threshold),
            "hum": _decide(_has_hum_phases, _sev, _hum_threshold),
            "clipping": _decide(_has_clip_phases, _sev, _clip_threshold),
        }

        # ── Codec-Chain-IQR-Floor: Terminal-Codec → clicks nur "mild" ──
        # (Brandenburg 1999: mp3/aac-Klick-Statistik nicht mit analog
        #  identisch; zu aggressiver click_iqr schadet der Transparenz)
        if policy["clicks"] == "aggressive" and chain_info is not None:
            try:
                _chain: list[str] = list(chain_info.get("transfer_chain") or chain_info.get("chain") or [])
                _terminal_codec = any(
                    str(n).strip().lower() in {"mp3_low", "mp3_high", "aac"}
                    for n in _chain[-2:]  # letztes Glied + Vorletztes
                )
                if _terminal_codec:
                    policy["clicks"] = "mild"
            except Exception:
                logger.debug("_decide: silent except suppressed", exc_info=True)

        return policy

    def _select_via_uv3(
        self,
        *,
        defect_result: Any,
        mode: str,
        chain_info: Any | None,
        defekt_hint: Any | None,
        audio: np.ndarray | None,
        sr: int,
        causal_plan: Any | None,
        restorability_score: float,
        pipeline_confidence: Any | None,
    ) -> list[str]:
        """Ruft UV3._select_phases() + _optimize_phase_plan_intelligence() ab.

        UV3 ist hier ein *Werkzeug* zur Defekt→Phasen-Übersetzung — kein Orchestrator.
        Der PhaseInteractionDenker verfeinert das Ergebnis anschließend semantisch.
        """
        try:
            _quality_mode_cls = _load_symbol("backend.core.unified_restorer_v3", "QualityMode")
            _restoration_config_cls = _load_symbol("backend.core.unified_restorer_v3", "RestorationConfig")
            _unified_restorer_v3_cls = _load_symbol("backend.core.unified_restorer_v3", "UnifiedRestorerV3")

            # Leicht-Instanz nur für Selektion (kein ML, kein Lazy-Load)
            _is_studio = mode in ("studio2026", "studio_2026", "maximum")
            _qmode = _quality_mode_cls.MAXIMUM if _is_studio else _quality_mode_cls.QUALITY
            _cfg = _restoration_config_cls(
                mode=_qmode,
                studio_2026=_is_studio,
                enforce_3x_rt=False,
                enable_performance_guard=False,
                enable_adaptive_skipping=False,
                enable_phase_gate=False,
            )
            _uv3 = _unified_restorer_v3_cls(config=_cfg)
            _select_fn = object.__getattribute__(_uv3, "_select_phases")
            _optimize_fn = object.__getattribute__(_uv3, "_optimize_phase_plan_intelligence")

            raw = _select_fn(
                defect_result,
                causal_plan=causal_plan,
                chain_info=chain_info,
                defekt_hint=defekt_hint,
                audio=audio,
                sr=sr,
            )
            optimized = _optimize_fn(
                raw,
                causal_plan=causal_plan,
                pipeline_confidence=pipeline_confidence,
                restorability_score=restorability_score,
            )
            logger.debug(
                "PhaseInteractionDenker: UV3-Selektion → %d Phasen (vor Semantic-Refine)",
                len(optimized),
            )
            if not isinstance(optimized, (list, tuple)):
                logger.warning(
                    "PhaseInteractionDenker: UV3-Optimizer lieferte unerwarteten Typ %s — UV3 übernimmt Selektion.",
                    type(optimized).__name__,
                )
                return []
            return [str(phase_id) for phase_id in optimized]

        except Exception as exc:
            logger.warning(
                "PhaseInteractionDenker: UV3-Phasenselektion fehlgeschlagen: %s",
                exc,
            )
            return []

    def _annotate(self, phases: list[str]) -> dict[str, frozenset[str]]:
        """Weist jeder Phase ihre semantischen Tags zu.

        Präfix-Matching: "phase_03_denoise_ml" erbt Tags von "phase_03_denoise".
        Unbekannte Phasen erhalten leere Tag-Menge.
        """
        result: dict[str, frozenset[str]] = {}
        for phase in phases:
            tags = _PHASE_SEMANTICS.get(phase)
            if tags is None:
                # Präfixsuche
                for known, known_tags in _PHASE_SEMANTICS.items():
                    if phase.startswith(known):
                        tags = known_tags
                        break
            result[phase] = tags if tags is not None else frozenset()
        return result

    def _resolve_conflicts(
        self,
        phases: list[str],
        annotations: dict[str, frozenset[str]],
    ) -> tuple[list[str], dict[str, str], list[str]]:
        """Erkennt und löst semantische Phasen-Konflikte (§2.48).

        Für jede Konflikt-Regel: wenn Phase A Auslöser-Tags hat, wird die
        erste nachfolgende Phase mit Ziel-Tags supprimiert.

        Invariante: Supprimierung ist deterministisch (First-Wins-Prinzip).
        """
        suppressed: dict[str, str] = {}
        conflict_notes: list[str] = []

        for trigger_tags, target_tags in _CONFLICT_RULES:
            trigger_phase: str | None = None
            for phase in phases:
                if phase in suppressed:
                    continue
                tags = annotations.get(phase, frozenset())
                if trigger_phase is None:
                    if tags & trigger_tags == trigger_tags:
                        trigger_phase = phase
                        continue
                # Auslöser wurde bereits gefunden — suche Konflikt-Kandidaten
                if trigger_phase is not None and tags & target_tags == target_tags:
                    reason = (
                        f"§2.48 [{', '.join(sorted(trigger_tags))}] "
                        f"→ [{', '.join(sorted(target_tags))}]: "
                        f"{trigger_phase} vor {phase} — {phase} supprimiert"
                    )
                    suppressed[phase] = reason
                    conflict_notes.append(reason)
                    logger.info("PhaseInteractionDenker: %s", reason)
                    # Only first target per trigger suppressed; reset for next occurrence
                    trigger_phase = None

        resolved = [p for p in phases if p not in suppressed]
        return resolved, suppressed, conflict_notes

    def _apply_order_constraints(
        self,
        phases: list[str],
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Erzwingt deklarative Reihenfolge-Constraints (§2.46 / §7.2).

        Für jedes Tupel (A, B): wenn A nach B steht, wird A vor B verschoben.
        Konvergiert in O(n × constraints) — keine Endlosschleifen möglich.
        """
        result = list(phases)
        applied: list[tuple[str, str]] = []

        for before, after in _ORDER_CONSTRAINTS:
            if before not in result or after not in result:
                continue
            idx_before = result.index(before)
            idx_after = result.index(after)
            if idx_before > idx_after:
                result.remove(before)
                idx_after_new = result.index(after)
                result.insert(idx_after_new, before)
                applied.append((before, after))
                logger.debug(
                    "PhaseInteractionDenker: §ORDER %s → vor → %s",
                    before,
                    after,
                )

        return result, applied

    @staticmethod
    def _with_policy_material(defect_result: Any, material: str) -> Any:
        """Erzeugt eine Plan-View mit zentralem Denker-Material statt Scanner-Fallback."""
        material_key = str(material or "").strip().lower()
        if not material_key:
            return defect_result
        try:
            material_type_cls = _load_symbol("backend.core.defect_scanner", "MaterialType")
            policy_material = material_type_cls(material_key)
        except Exception:
            logger.warning(
                "Verarbeitungsschritt_interaction_denker.py::_with_policy_material Ersatzpfad", exc_info=True
            )
            return defect_result
        if getattr(defect_result, "material_type", None) == policy_material:
            return defect_result
        try:
            planned = copy(defect_result)
            planned.material_type = policy_material
            logger.info(
                "PhaseInteractionDenker: material policy override for UV3-Selektion: %s -> %s",
                getattr(
                    getattr(defect_result, "material_type", None),
                    "value",
                    getattr(defect_result, "material_type", None),
                ),
                material_key,
            )
            return planned
        except Exception:
            logger.warning(
                "Verarbeitungsschritt_interaction_denker.py::_with_policy_material Ersatzpfad", exc_info=True
            )
            return defect_result

    @staticmethod
    def _strip_restoration_forbidden(phases: list[str], mode: str) -> tuple[list[str], list[str]]:
        """§0a Crossfire-Modus-Invariante: entfernt verbotene Phasen in Restoration.

        phase_21 (Exciter), phase_35 (Multiband-Kompression) und phase_42
        (Vocal-Enhancement) dürfen in ``restoration`` NIEMALS Teil eines
        ``PhasePlan`` sein — bidirektional, auch nicht als Fallback oder über
        eine Injektionsquelle. Studio-2026 behält die Phasen.

        Matching per ``phase_NN``-Nummer-Präfix → robust gegen Namensvarianten.
        Rückgabe: (bereinigte Phasenliste, entfernte Phasen) — deterministisch,
        Reihenfolge der verbleibenden Phasen unverändert.
        """
        _is_studio = mode in ("studio2026", "studio_2026", "maximum")
        if _is_studio:
            return list(phases), []
        forbidden_prefixes = _restoration_forbidden_prefixes()
        kept: list[str] = []
        removed: list[str] = []
        for p in phases:
            parts = str(p).split("_")
            prefix = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else str(p)
            if prefix in forbidden_prefixes:
                removed.append(p)
            else:
                kept.append(p)
        return kept, removed

    @staticmethod
    def get_adaptive_post_processing_order(
        material: str = "unknown",
        defect_types: list[str] | None = None,
        era_decade: int = 1970,
    ) -> list[str]:
        """Adaptive 8-stage post-processing order based on material/defect profile.

        Static order: Breitband→Impulsiv→Rauschen→Spektral→Räumlich→Dynamik→Enhancement→Ausgabe

        Material-specific reordering:
        - cassette/tape: Breitband AFTER Impulsiv (preserve noise profile for Stage 3)
        - reel_tape: Räumlich BEFORE Dynamik (azimuth correction before compression)
        - vinyl/shellac: static order (noise is broadband, needs Stage 1 first)
        """
        stages = [
            "breitband",  # 1: Hum, Rumpel, DC
            "impulsiv",  # 2: Clicks, Kratzer, Dropouts
            "rauschen",  # 3: Phase 03, 29
            "spektral",  # 4: AntiMuffling
            "raeumlich",  # 5: SmartTapeRepair, EchoRemoval
            "dynamik",  # 6: Phase 10, 26, 54
            "enhancement",  # 7: SibilanceMax, VocalClarity
            "ausgabe",  # 8: Humanization, PerceptualOptimizer
        ]

        mat = str(material).lower()

        if mat in ("cassette", "tape"):
            # Move Breitband AFTER Impulsiv — preserve noise profile
            stages.remove("breitband")
            impulsiv_idx = stages.index("impulsiv")
            stages.insert(impulsiv_idx + 1, "breitband")

        elif mat == "reel_tape":
            # Move Räumlich BEFORE Spektral — azimuth correction before spectral cleanup
            stages.remove("raeumlich")
            spektral_idx = stages.index("spektral")
            stages.insert(spektral_idx, "raeumlich")

        # vinyl, shellac, digital: static order

        return stages

    # ── Guard-Modulation (zentrale Entscheidungs-Intelligenz) ─────────────

    def _dag_reorder(self, phases: list[str], *, material: str = "unknown") -> list[str]:
        """§3.4 ECHTES DAG: Topologische Sortierung mit Kahn's Algorithmus.

        Jede Phase deklariert in PHASE_FREQ_PROFILES:
          - affects:   [(f_low, f_high, intensity), ...] — bearbeitete Frequenzbänder
          - category:  "subtractive" | "additive" | "dynamics" | "corrective" | "other"
          - requires:  [freq_band_name, ...] — Bänder, die VORHER restauriert sein müssen
          - conflicts: [phase_id, ...] — Phasen, die NICHT adjazent laufen sollen
          - after:     [phase_id, ...] — explizite Vorgänger (harte Kanten)

        Kahn's Algorithmus:
          1. Build adjacency list from requires/after + category rules + conflicts
          2. Compute in-degree for each node
          3. Queue nodes with in-degree 0, sorted by priority
          4. Process queue, append to result, reduce in-degree of neighbors
          5. Cycle detection: if result shorter than input → break cycles by priority

        Material-adaptive Regeln (zusätzlich zu deklarierten Kanten):
          - Tape/Kassette:  subtractive → dynamics → additive (NR vor EQ)
          - Shellac/Wachs:  additive → dynamics → subtractive (BW-Restaurierung vor NR)
          - Vinyl/Digital:  corrective → subtractive → dynamics → additive (Default)
          - reel_tape:      corrective → subtractive → additive → dynamics
        """
        if len(phases) <= 1:
            return list(phases)

        try:
            from denker.cross_phase_coordinator import PHASE_FREQ_PROFILES
        except ImportError:
            return list(phases)

        phase_set = set(phases)
        mat = str(material).lower()

        # ── 0. Material-adaptive Kategorie-Reihenfolge (VOR Kantenbau) ──
        if mat in ("cassette", "tape", "reel_tape"):
            cat_order = ["corrective", "subtractive", "dynamics", "additive", "other"]
        elif mat in ("shellac", "wax_cylinder", "lacquer_disc"):
            cat_order = ["corrective", "additive", "dynamics", "subtractive", "other"]
        else:
            cat_order = ["corrective", "subtractive", "dynamics", "additive", "other"]

        # ── 1. Baue Adjazenzliste und In-Degree ─────────────────────────
        adj: dict[str, list[str]] = {p: [] for p in phases}
        in_degree: dict[str, int] = dict.fromkeys(phases, 0)

        def add_edge(src: str, dst: str) -> None:
            """Fügt gerichtete Kante src → dst hinzu (src muss vor dst laufen)."""
            if src not in phase_set or dst not in phase_set:
                return
            if src == dst:
                return
            if dst not in adj[src]:
                adj[src].append(dst)
                in_degree[dst] += 1
                logger.debug("DAG edge: %s → %s", src, dst)

        # ── 1a. Deklarierte Kanten aus PHASE_FREQ_PROFILES ─────────────
        for pid in phases:
            profile = PHASE_FREQ_PROFILES.get(pid, {})
            # Explizite Vorgänger (after) — materialadaptiv überspringbar
            for predecessor in profile.get("after", []):
                # Bei Konflikt mit Material-Regel: after-Kante überspringen
                # (z.B. Shellac: additive vor subtractive → after "denoise vor presence" ignorieren)
                pred_cat = PHASE_FREQ_PROFILES.get(predecessor, {}).get("category", "other")
                pid_cat = profile.get("category", "other")
                if _categories_conflict_with_material(pred_cat, pid_cat, mat, cat_order):
                    logger.debug(
                        "DAG: skipping after edge %s (%s) → %s (%s) — conflicts with material rule for %s",
                        predecessor,
                        pred_cat,
                        pid,
                        pid_cat,
                        mat,
                    )
                    continue
                add_edge(predecessor, pid)
            # requires: Phasen, die die benötigten Frequenzbänder bereitstellen
            for required_band in profile.get("requires", []):
                for other in phases:
                    if other == pid:
                        continue
                    other_profile = PHASE_FREQ_PROFILES.get(other, {})
                    other_affects = other_profile.get("affects", [])
                    for band_low, band_high, _ in other_affects:
                        band_name = _freq_range_to_band_name(band_low, band_high)
                        if band_name == required_band and other_profile.get("category") in (
                            "additive",
                            "corrective",
                        ):
                            # Bei Konflikt mit Material-Regel: requires-Kante überspringen
                            other_cat = other_profile.get("category", "other")
                            pid_cat_req = profile.get("category", "other")
                            if not _categories_conflict_with_material(other_cat, pid_cat_req, mat, cat_order):
                                add_edge(other, pid)
                            break

        # ── 1b. Konflikt-Kanten: conflicting phases → separieren ──────
        for pid in phases:
            profile = PHASE_FREQ_PROFILES.get(pid, {})
            for conflict_id in profile.get("conflicts", []):
                if conflict_id in phase_set and conflict_id != pid:
                    # Der Phase mit niedrigerer Priorität wird als Nachfolger einsortiert
                    prio_a = profile.get("priority", 5)
                    prio_b = PHASE_FREQ_PROFILES.get(conflict_id, {}).get("priority", 5)
                    if prio_a < prio_b:
                        add_edge(pid, conflict_id)
                    elif prio_b < prio_a:
                        add_edge(conflict_id, pid)
                    # Bei gleicher Priorität: Kategorie-basierte Entscheidung
                    else:
                        cat_a = profile.get("category", "")
                        cat_b = PHASE_FREQ_PROFILES.get(conflict_id, {}).get("category", "")
                        if cat_a == "subtractive" and cat_b != "subtractive":
                            add_edge(pid, conflict_id)
                        elif cat_b == "subtractive" and cat_a != "subtractive":
                            add_edge(conflict_id, pid)

        # ── 1c. Material-adaptive Kategorie-Kanten ─────────────────────
        # cat_order bereits in Schritt 0 definiert

        # Phasen nach Kategorie gruppieren
        cat_phases: dict[str, list[str]] = {c: [] for c in cat_order}
        for pid in phases:
            profile = PHASE_FREQ_PROFILES.get(pid, {})
            cat = profile.get("category", "other")
            if cat not in cat_phases:
                cat = "other"
            cat_phases.setdefault(cat, []).append(pid)

        # Kanten zwischen aufeinanderfolgenden Kategorien
        for i in range(len(cat_order) - 1):
            for src in cat_phases.get(cat_order[i], []):
                for dst in cat_phases.get(cat_order[i + 1], []):
                    # Nur wenn keine explizite reverse-Kante existiert
                    if src not in adj.get(dst, []):
                        add_edge(src, dst)

        # ── 2. Kahn's Algorithmus ──────────────────────────────────────
        # Priority queue: (priority, phase_id) — niedrigere priority = früher
        import heapq

        queue: list[tuple[int, str]] = []
        for pid in phases:
            if in_degree[pid] == 0:
                prio = PHASE_FREQ_PROFILES.get(pid, {}).get("priority", 5)
                heapq.heappush(queue, (prio, pid))

        result: list[str] = []
        while queue:
            _, pid = heapq.heappop(queue)
            result.append(pid)
            for neighbor in adj.get(pid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    prio = PHASE_FREQ_PROFILES.get(neighbor, {}).get("priority", 5)
                    heapq.heappush(queue, (prio, neighbor))

        # ── 3. Cycle-Breaking: verbleibende Phasen nach Priorität ──────
        remaining = [p for p in phases if p not in result]
        if remaining:
            logger.warning(
                "DAG cycle erkannt: %d phases not reachable (material=%s). Breaking by priority.",
                len(remaining),
                mat,
            )
            remaining.sort(key=lambda p: PHASE_FREQ_PROFILES.get(p, {}).get("priority", 5))
            result.extend(remaining)

        # ── 4. Validierung: alle Phasen vorhanden? ─────────────────────
        if set(result) != phase_set:
            missing = phase_set - set(result)
            extra = set(result) - phase_set
            logger.error(
                "DAG integrity error: missing=%s extra=%s — falling back to category sort",
                missing,
                extra,
            )
            return self._dag_reorder_fallback(phases, material=mat)

        logger.info(
            "§3.4 DAG reorder: %d phases, material=%s → %s",
            len(result),
            mat,
            " → ".join(result[:5]) + (" …" if len(result) > 5 else ""),
        )
        return result

    def _dag_reorder_fallback(self, phases: list[str], *, material: str = "unknown") -> list[str]:
        """Fallback: Kategorie-basierte Sortierung (ursprüngliche Logik)."""
        if len(phases) <= 1:
            return list(phases)
        try:
            from denker.cross_phase_coordinator import PHASE_FREQ_PROFILES
        except ImportError:
            return list(phases)
        subtractive: list[Any] = []
        additive: list[Any] = []
        dynamics: list[Any] = []
        corrective: list[Any] = []
        other: list[Any] = []
        for pid in phases:
            cat = PHASE_FREQ_PROFILES.get(pid, {}).get("category", "other")
            {"subtractive": subtractive, "additive": additive, "dynamics": dynamics, "corrective": corrective}.get(
                cat, other
            ).append(pid)
        mat = str(material).lower()
        if mat in ("shellac", "wax_cylinder", "lacquer_disc"):
            return corrective + additive + dynamics + subtractive + other
        return corrective + subtractive + dynamics + additive + other


def _freq_range_to_band_name(f_low: float, f_high: float) -> str:
    """Mapped einen Frequenzbereich auf den nächstgelegenen FREQ_BANDS-Namen."""
    try:
        from denker.cross_phase_coordinator import FREQ_BANDS
    except ImportError:
        return "mid"
    center = (f_low + f_high) / 2
    best_band = "mid"
    best_dist = float("inf")
    for band_name, b_low, b_high, _ in FREQ_BANDS:
        b_center = (b_low + b_high) / 2
        dist = abs(center - b_center)
        if dist < best_dist:
            best_dist = dist
            best_band = band_name
    return best_band


def _categories_conflict_with_material(src_cat: str, dst_cat: str, material: str, cat_order: list[str]) -> bool:
    """Prüft, ob eine after-Kante (src→dst) der materialadaptiven Kategorie-Reihenfolge widerspricht.

    Wenn src in cat_order NACH dst kommt, würde die after-Kante src→dst
    einen Zyklus mit den Kategorie-Kanten erzeugen → Konflikt → Kante überspringen.

    Returns True wenn die Kante der Material-Regel widerspricht.
    """
    if src_cat not in cat_order or dst_cat not in cat_order:
        return False
    src_idx = cat_order.index(src_cat)
    dst_idx = cat_order.index(dst_cat)
    # Konflikt: src (Vorgänger) kommt in cat_order NACH dst (Nachfolger)
    # → after-Kante würde rückwärts durch die Kategorien zeigen
    return src_idx > dst_idx

    # ── Guard-Modulation (zentrale Entscheidungs-Intelligenz) ─────────────

    _CRITICAL_PHASES: frozenset[str] = frozenset(
        {
            "phase_14_phase_correction",
            "phase_25_azimuth_correction",
            "phase_56_spectral_band_gap_repair",
            "phase_24_dropout_repair",
            "phase_01_click_removal",
            "phase_09_crackle_removal",
            "phase_27_click_pop_removal",
            "phase_03_denoise",
            "phase_02_hum_removal",
            "phase_05_rumble_filter",
        }
    )

    @staticmethod  # type: ignore[misc]
    def resolve_guard_modulation(
        base_strength: float,
        *,
        goal_budget: Any = None,
        guard_wisdom: Any = None,
        cross_guard_results: dict[str, Any] | None = None,
        phase_id: str = "",
        material: str = "unknown",
    ) -> float:
        """Zentrale Guard-Modulation — eine Stimme, gewichtete Entscheidung.

        Ersetzt die blinde Multiplikation dreier Guards in _profiled_phase_call.
        Gewichtet die Guard-Einflüsse statt sie zu multiplizieren:
          - GoalBudget:  40 % Gewicht (Budget-Erschöpfung)
          - GuardWisdom: 50 % Gewicht (Lern-Historie)
          - CrossGuard:  10 % Gewicht (phasenübergreifende Konflikte)

        Returns modulierte Stärke ∈ [0.0, 1.0], nie unter Material-Floor.
        """
        penalties: list[tuple[float, float]] = []  # (faktor, gewicht)

        # ── GoalBudget (40 %) ──
        if goal_budget is not None and hasattr(goal_budget, "fraction_left"):
            try:
                wf = min(goal_budget.fraction_left(g) for g in ("waerme", "brillanz", "punch"))
                if wf < 0.5:
                    penalties.append((max(0.5, wf), 0.40))
            except Exception:
                logger.debug("resolve_guard_modulation: silent except suppressed", exc_info=True)

        # ── GuardWisdom (50 %) ──
        if guard_wisdom is not None and hasattr(guard_wisdom, "get_strength_mod"):
            sm = guard_wisdom.get_strength_mod()
            if sm < 1.0:
                penalties.append((sm, 0.50))

        # ── CrossGuard (10 %) ──
        if isinstance(cross_guard_results, dict) and cross_guard_results.get("verdict") == "degraded":
            penalties.append((0.85, 0.10))

        # ── Gewichtete Modulation (nicht multiplikativ) ──
        if penalties:
            total_weight = sum(w for _, w in penalties)
            if total_weight > 0:
                weighted_factor = sum(f * w for f, w in penalties) / total_weight
                base_strength *= weighted_factor

        # ── Material-adaptive Mindest-Stärke ──
        mat = str(material).lower()
        _min = 0.30
        if phase_id in PhaseInteractionDenker._CRITICAL_PHASES:
            if mat in ("cassette", "tape", "reel_tape"):
                _min = 0.40
            elif mat in ("vinyl", "shellac"):
                _min = 0.35

        return max(_min, min(1.0, base_strength))
