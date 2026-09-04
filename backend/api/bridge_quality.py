"""Aurik 10 — Bridge: Qualitätsbewertung (§8.1)
===================================================
Qualitätsmetriken und -bewertungen für Frontend/CLI → Backend-Core.

Enthält:
  - MusicalGoalsChecker-Klasse (§8.1, 15 Goals mit AMRB-kalibrierten Schwellwerten)
  - Adaptive Goal Thresholds + Config (§2.31, material-/ära-adaptiv)
  - MUSHRA-Evaluator-Singleton (§8.1.1 OQS, algorithmische PEAQ-Approximation)
  - PerceptualQualityScorer-Singleton (§8.1 PQS, alle vier Metriken)
  - Experience Insights (Joy/Fatigue/Frisson, Empfehlungen, Recovery-Certainty)
  - Goal Feedback Recording (§C10 Bayesian EMA Calibration)

Referenz: AGENTS.md §1 (Normative Kette), .github/copilot-instructions.md §V4 Bridge-Verbot.
"""

# pylint: disable=import-outside-toplevel
# cspell:disable

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _coerce_dict_str_any(raw: Any) -> dict[str, Any]:
    """Normalisiert optionale Metadaten auf ein dict[str, Any]."""
    return dict(raw) if isinstance(raw, dict) else {}


def _coerce_list_any(raw: Any) -> list[Any]:
    """Normalisiert optionale Metadaten auf eine Liste."""
    return list(raw) if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# MusicalGoalsChecker (§8.1) — 15 Goals mit AMRB-kalibrierten Schwellwerten
# ---------------------------------------------------------------------------


def get_musical_goals_checker() -> type:
    """Gibt ``MusicalGoalsChecker``-Klasse zurück (lazy import, §8.1).

    Die zurückgegebene **Klasse** kann instanziiert werden::

        checker = get_musical_goals_checker()()
        scores = checker.measure_all(audio, sr)  # Dict[str, float]

    15 Musical Goals mit AMRB-kalibrierten Schwellwerten (§8.1).
    Adaptive Schwellwerte via ``get_adaptive_goals_fn()`` — nicht statisch!
    """
    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker  # type: ignore[import]

    return MusicalGoalsChecker  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Adaptive Goal Thresholds + Config (§2.31)
# ---------------------------------------------------------------------------


def get_adaptive_goals_fn():
    """Gibt ``get_adaptive_goals_and_config``-Funktion zurück (lazy import, §2.31).

    Signatur::

        get_adaptive_goals_and_config(
            audio: np.ndarray,
            sr: int,
        ) -> tuple[AdaptiveGoalThresholds, dict, MaterialQualityAssessment]

    **Pflicht vor jeder Restaurierung**: statische Schwellwerte sind verboten
    als alleinige Entscheidungsbasis (§2.31 AdaptiveGoalThresholds).
    Schwellwerte werden material-, ära- und restorability-adaptiv skaliert.
    """
    from backend.core.musical_goals.adaptive_goals_system import (  # type: ignore[import]
        get_adaptive_goals_and_config,
    )

    return get_adaptive_goals_and_config


# ---------------------------------------------------------------------------
# MUSHRA-Evaluator-Singleton (§8.1.1 OQS) — algorithmische PEAQ-Approximation
# ---------------------------------------------------------------------------


def get_mushra_evaluator():
    """Gibt den ``MushraEvaluator``-Singleton zurück (lazy import, §8.1.1 OQS).

    OQS = algorithmische PEAQ-Approximation (kein ITU-R-MUSHRA).
    In externen Berichten stets "OQS (algorithmisch)" schreiben.

    Schwellwerte::

        OQS ≥ 91  → Excellent (A)
        OQS ≥ 80  → Good (B)  — Pflicht für jede neue Phase / jedes Plugin
        OQS ≥ 60  → Fair (C)

    Verwendung::

        evaluator = get_mushra_evaluator()
        result = evaluator.evaluate(audio, sr)
        assert result.oqs >= 80, f"OQS unter Good-Schwelle: {result.oqs}"
    """
    from backend.core.mushra_evaluator import get_mushra_evaluator as _get  # type: ignore[import]

    return _get()


# ---------------------------------------------------------------------------
# PerceptualQualityScorer-Singleton (§8.1 PQS) — alle vier Metriken
# ---------------------------------------------------------------------------


def get_perceptual_quality_scorer():
    """Gibt den ``PerceptualQualityScorer``-Singleton zurück (lazy import, §8.1 PQS).

    Prüft **alle vier PQS-Metriken** — nie nur MOS allein (§8.1)::

        PQS MOS            ≥ 3.8 (generell) / ≥ 4.5 (nur cd_digital/dat/mp3_high/aac)
        PQS NSIM           ≥ 0.70
        MCD (dB)           ≤ 8.0
        Spectral Coherence ≥ 0.60

    ABSOLUT VERBOTEN als Musikmetrik: klassische Sprachqualitaets-Metriken und CDPAM.

    Verwendung::

        pqs = get_perceptual_quality_scorer()
        result = pqs.score(audio, sr)
        assert result.mos >= 3.8, f"PQS MOS zu niedrig: {result.mos}"
    """
    from backend.core.perceptual_quality_scorer import (  # type: ignore[import]
        get_perceptual_quality_scorer as _get,
    )

    return _get()


# ---------------------------------------------------------------------------
# Experience Insights — Joy/Fatigue/Frisson, Empfehlungen, Recovery-Certainty
# ---------------------------------------------------------------------------


def resolve_pipeline_fail_reason(
    *,
    typed_fail_reason=None,
    metadata: dict | None = None,
    stage_notes: dict | None = None,
    fail_reasons: list[dict] | None = None,
) -> str:
    """Löst ``fail_reason`` aus typed Feld, Metadata und Stage-Notes auf (lazy import)."""
    from backend.core.pipeline_health_state import resolve_fail_reason as _resolve  # type: ignore[import]

    return _resolve(  # type: ignore[no-any-return]
        typed_fail_reason=typed_fail_reason,
        metadata=metadata,
        stage_notes=stage_notes,
        fail_reasons=fail_reasons,
    )


def get_experience_insights(result: Any) -> dict[str, Any]:
    """Extrahiert normalized joy/fatigue/recommendation insights from a result object.

    Frontend-safe helper for AurikErgebnis/RestorationResult-like objects.
    Returns stable keys even if metadata is partially missing.
    """
    _meta_raw = getattr(result, "metadata", None)
    _meta: dict[str, Any] = _coerce_dict_str_any(_meta_raw)

    _joy = _coerce_dict_str_any(_meta.get("joy_runtime_index"))
    _auto = _coerce_dict_str_any(_meta.get("auto_improvement_recommendations"))
    _song_cal = _coerce_dict_str_any(_meta.get("song_calibration"))
    _cluster = _coerce_dict_str_any(_song_cal.get("cluster_policy"))
    _fqf = _coerce_dict_str_any(_meta.get("fallback_quality_floor"))
    _rc = _coerce_dict_str_any(_meta.get("recovery_certainty"))
    _stage_notes: dict[str, Any] = _coerce_dict_str_any(getattr(result, "stage_notes", None))

    _rec_raw = _auto.get("recommendations")
    _recommendations: list[Any] = list(_rec_raw) if isinstance(_rec_raw, list) else []

    def _safe01(v: Any) -> float:
        try:
            vf = float(v)
        except Exception:
            logger.warning("bridge.py::_safe01 Ersatzpfad", exc_info=True)
            return 0.0
        if not np.isfinite(vf):
            return 0.0
        return float(np.clip(vf, 0.0, 1.0))

    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            vf = float(v)
        except Exception:
            logger.warning("bridge.py::_safe_float Ersatzpfad", exc_info=True)
            return float(default)
        if not np.isfinite(vf):
            return float(default)
        return vf

    _normalized_recommendations: list[dict[str, Any]] = []
    for _r in _recommendations:
        if not isinstance(_r, dict):
            continue
        _normalized_recommendations.append(
            {
                "priority": str(_r.get("priority", "info") or "info"),
                "focus": str(_r.get("focus", "") or ""),
                "reason": str(_r.get("reason", "") or ""),
                "action": str(_r.get("action", "") or ""),
            }
        )

    _cnt_raw = _auto.get("count", len(_normalized_recommendations))
    try:
        _cnt = int(_cnt_raw)
    except Exception:
        logger.debug("bridge: Auto-Improvement-Zählung konnte nicht geparst werden, nutze len(recommendations)")
        _cnt = len(_normalized_recommendations)
    _cnt = max(_cnt, len(_normalized_recommendations), 0)

    _tc = _coerce_dict_str_any(_meta.get("team_coordination"))
    _tc_events_raw_val = _tc.get("events")
    _tc_events_raw: list[Any] = list(_tc_events_raw_val) if isinstance(_tc_events_raw_val, list) else []
    _tc_events: list[dict[str, Any]] = []
    for _tce in _tc_events_raw:
        if not isinstance(_tce, dict):
            continue
        _tc_events.append(
            {
                "phase_id": str(_tce.get("phase_id", "") or ""),
                "action": str(_tce.get("action", "") or ""),
                "reason": str(_tce.get("reason", "") or ""),
                "excluded_goals": list(_tce.get("excluded_goals", []) or []),
            }
        )
    try:
        _tc_count = int(_tc.get("event_count", len(_tc_events)))
    except Exception:
        logger.debug("bridge: Team-Koordination event_count konnte nicht geparst werden, nutze len(events)")
        _tc_count = len(_tc_events)
    _pt_summary = dict(_tc.get("phase_type_summary", {}) or {})
    _fqf_trace_raw_val = _fqf.get("recovery_trace")
    _fqf_trace_raw: list[Any] = list(_fqf_trace_raw_val) if isinstance(_fqf_trace_raw_val, list) else []
    _fqf_trace: list[dict[str, Any]] = []
    for _tr in _fqf_trace_raw:
        if not isinstance(_tr, dict):
            continue
        _fqf_trace.append(
            {
                "attempt": int(_tr.get("attempt", 0)) if isinstance(_tr.get("attempt", 0), (int, float)) else 0,
                "candidate": str(_tr.get("candidate", "") or ""),
                "action": str(_tr.get("action", "") or ""),
                "result": str(_tr.get("result", "") or ""),
            }
        )

    _fail_reasons: list[Any] = _coerce_list_any(_meta.get("fail_reasons"))
    if not _fail_reasons and isinstance(_stage_notes.get("fail_reasons"), list):
        _fail_reasons = list(_stage_notes.get("fail_reasons") or [])

    _primary_fail_reason = resolve_pipeline_fail_reason(
        typed_fail_reason=getattr(result, "fail_reason", None),
        metadata=_meta,
        stage_notes=_stage_notes,
        fail_reasons=_fail_reasons,
    )
    _raw_degradation = (
        getattr(result, "degradation_status", None)
        or _meta.get("degradation_status", "")
        or _stage_notes.get("degradation_status", "")
    )
    from backend.api.bridge_audio import normalize_pipeline_health_state

    _degradation_status = normalize_pipeline_health_state(_raw_degradation).value

    _fqf_triggered = bool(_fqf.get("triggered", False))
    _fqf_status = str(_fqf.get("status", "") or "").strip().lower()
    _fqf_attempts = int(_fqf.get("attempts", 0)) if isinstance(_fqf.get("attempts", 0), (int, float)) else 0
    _exp_profile = str(_meta.get("export_gate_profile", "") or "").strip()
    _exp_material = str(_meta.get("export_gate_material", "") or "").strip()
    _exp_thresholds = _coerce_dict_str_any(_meta.get("export_gate_thresholds"))
    _exp_signature = _coerce_dict_str_any(_meta.get("export_gate_signal_signature"))
    _exp_preserve_signal = _safe01(_meta.get("export_gate_preserve_signal", 0.0))
    _xp_stage_profile = _coerce_dict_str_any(_stage_notes.get("exzellenz_recovery_profile"))
    if _xp_stage_profile:
        _exp_preserve_signal = max(_exp_preserve_signal, _safe01(_xp_stage_profile.get("preserve_signal", 0.0)))
    if not _exp_profile:
        if _exp_preserve_signal >= 0.55:
            _exp_profile = "fragile_or_transient_risk"
        elif _exp_preserve_signal <= 0.20 and _degradation_status == "ok":
            _exp_profile = "modern_stable"
        else:
            _exp_profile = "neutral"

    # Keep bridge and export-workflow semantics aligned for recovered/degraded fallback-floor runs.
    if _fqf_triggered and _fqf_status in {"recovered", "degraded", "failed", "fail"}:
        if _degradation_status == "ok":
            _degradation_status = "recovered" if _fqf_status == "recovered" else "degraded"
        if not _primary_fail_reason:
            _primary_fail_reason = str(_fqf.get("reason", "fallback_quality_floor_triggered") or "")

    _primary_error_code = ""
    if _fail_reasons and isinstance(_fail_reasons[0], dict):
        _primary_error_code = str(_fail_reasons[0].get("error_code", "") or "")
    _wcs_gate = _coerce_dict_str_any(_meta.get("worldclass_composite_gate"))
    _threshold_evidence = _coerce_dict_str_any(_meta.get("threshold_evidence"))
    _qe_threshold = _safe_float(_exp_thresholds.get("quality_estimate", 0.0), 0.0)
    _root_cause = str(_primary_fail_reason or "").strip()
    _root_cause_l = _root_cause.lower()
    _pipeline_like_failure = (
        _root_cause_l.startswith("pipeline_blocked:")
        or "pipeline-fehler" in _root_cause_l
        or "pipeline_fehler" in _root_cause_l
        or "unexpected keyword argument" in _root_cause_l
        or "missing 1 required positional argument" in _root_cause_l
    )
    _failure_class = "none"
    if _degradation_status in {"blocked", "critical_degraded", "degraded"}:
        if _pipeline_like_failure or (_qe_threshold <= 0.0001 and bool(_root_cause)):
            _failure_class = "technical_failure"
        else:
            _failure_class = "quality_failure"
    if _root_cause_l.startswith("pipeline_blocked:"):
        _root_cause = _root_cause.split(":", 1)[1].strip() or _root_cause

    _tone = "focus"
    if _degradation_status in {"blocked", "critical_degraded", "degraded"}:
        _tone = "caution"
    elif _safe01(_joy.get("joy_index", 0.0)) >= 0.72 and _safe01(_joy.get("fatigue_index", 0.0)) <= 0.30:
        _tone = "confidence"

    _headline = "Verarbeitung stabil"
    if _tone == "caution":
        _headline = "Ergebnis mit Schutzpriorität"
    elif _tone == "confidence":
        _headline = "Klangbild auf Kurs"

    _next_actions: list[str] = []
    if _degradation_status in {"blocked", "critical_degraded", "degraded"}:
        _next_actions.append("Konservative Recovery-Kaskade bevorzugen")
    if _safe01(_joy.get("fatigue_index", 0.0)) >= 0.45:
        _next_actions.append("Ermüdung senken: Dynamik-/HF-Eingriffe reduzieren")
    if _safe01(_joy.get("frisson_index", 0.0)) <= 0.35:
        _next_actions.append("Emotionale Akzente in Frisson-Zonen schonen")
    if not _next_actions:
        _next_actions.append("Aktuellen Kurs beibehalten")

    _quality_band = "mittel"
    _joy_idx = _safe01(_joy.get("joy_index", 0.0))
    _fat_idx = _safe01(_joy.get("fatigue_index", 0.0))
    if _joy_idx >= 0.75 and _fat_idx <= 0.30:
        _quality_band = "hoch"
    elif _joy_idx <= 0.45 or _fat_idx >= 0.55:
        _quality_band = "kritisch"

    return {
        "joy_index": _safe01(_joy.get("joy_index", 0.0)),
        "fatigue_index": _safe01(_joy.get("fatigue_index", 0.0)),
        "frisson_index": _safe01(_joy.get("frisson_index", 0.0)),
        "cluster_key": str(_song_cal.get("cluster_key", "") or ""),
        "cluster_policy": dict(_cluster) if isinstance(_cluster, dict) else {},
        "recommendations": _normalized_recommendations,
        "recommendation_count": _cnt,
        "team_coordination": {
            "event_count": _tc_count,
            "events": _tc_events,
            "phase_type_summary": _pt_summary,
        },
        "fallback_quality_floor": {
            "triggered": bool(_fqf.get("triggered", False)),
            "passed": bool(_fqf.get("passed", True)),
            "status": str(_fqf.get("status", "passed") or "passed"),
            "reason": str(_fqf.get("reason", "") or ""),
            "recovered": bool(_fqf.get("recovered", False)),
            "attempts": int(_fqf.get("attempts", 0)) if isinstance(_fqf.get("attempts", 0), (int, float)) else 0,
            "fallback_count": (
                int(_fqf.get("fallback_count", 0)) if isinstance(_fqf.get("fallback_count", 0), (int, float)) else 0
            ),
            "artifact_freedom": _safe01(_fqf.get("artifact_freedom", 1.0)),
            "hpi_passed": bool(_fqf.get("hpi_passed", False)),
            "hpi": _safe_float(_fqf.get("hpi", 0.0), 0.0),
            "best_candidate": str(_fqf.get("best_candidate", "") or ""),
            "recovery_trace": _fqf_trace,
        },
        "quality_gate": {
            "passed": bool(_degradation_status == "ok"),
            "degradation_status": str(_degradation_status),
            "primary_fail_reason": str(_primary_fail_reason or ""),
            # §v10.202: Guardian-Revert-Info für Layman-Kommunikation
            "do_no_harm_reverted": bool((_meta.get("do_no_harm") or {}).get("reverted", False)),
            "do_no_harm_reason": str((_meta.get("do_no_harm") or {}).get("reason", "")),
            "uqm_override_applied": bool((_meta.get("uqm") or {}).get("override_applied", False)),
            "uqm_quality_score": float((_meta.get("uqm") or {}).get("quality_score", 0.0)),
            "root_cause": str(_root_cause),
            "failure_class": str(_failure_class),
            "primary_error_code": str(_primary_error_code),
            "required_gates": ["musical_goals", "pqs", "oqs", "fallback_quality_floor"],
            "recovery_attempted": bool(_fqf_attempts > 0),
            "best_possible_reached": bool(_fqf_status == "recovered"),
            "fallback_quality_floor_status": str(_fqf.get("status", "passed") or "passed"),
            "profile": str(_exp_profile),
            "material": str(_exp_material),
            "preserve_signal": float(_exp_preserve_signal),
            "thresholds": {
                "quality_estimate": _qe_threshold,
                "level_drop_db": _safe_float(_exp_thresholds.get("level_drop_db", 0.0), 0.0),
            },
            "signal_signature": {
                "crest_db": _safe_float(_exp_signature.get("crest_db", 0.0), 0.0),
                "hf_ratio": _safe01(_exp_signature.get("hf_ratio", 0.0)),
                "transient_ratio": _safe01(_exp_signature.get("transient_ratio", 0.0)),
                "micro_dynamic_db": _safe_float(_exp_signature.get("micro_dynamic_db", 0.0), 0.0),
            },
            "worldclass_composite_gate": {
                "wcs": _safe01(_wcs_gate.get("wcs", 0.0)),
                "threshold": _safe01(_wcs_gate.get("threshold", 0.0)),
                "profile": str(_wcs_gate.get("profile", "") or ""),
                "artifact_veto": bool(_wcs_gate.get("artifact_veto", False)),
                "passed": bool(_wcs_gate.get("passed", False)),
            },
        },
        "threshold_evidence": dict(_threshold_evidence) if _threshold_evidence else {},
        "user_guidance": {
            "tone": _tone,
            "headline": _headline,
            "next_actions": _next_actions,
            "degradation_status": str(_degradation_status),
        },
        "quality_scale": {
            "band": _quality_band,
            "joy_index": _joy_idx,
            "fatigue_index": _fat_idx,
            "frisson_index": _safe01(_joy.get("frisson_index", 0.0)),
        },
        "recovery_certainty": {
            "recoverability_ceiling": _safe01(_rc.get("recoverability_ceiling", 0.0)),
            "uncertainty_index": _safe01(_rc.get("uncertainty_index", 1.0)),
            "conservative_audio_scalar": _safe01(_rc.get("conservative_audio_scalar", 1.0)),
            "confidence_band": str(_rc.get("confidence_band", "") or ""),
            "restorability_score": _safe_float(_rc.get("restorability_score", 0.0), 0.0),
            "transfer_generation_count": (
                int(_rc.get("transfer_generation_count", 0))
                if isinstance(_rc.get("transfer_generation_count", 0), (int, float))
                else 0
            ),
        },
        # §0/§2.46 HF-Hallucination-Guard: Treffer-Aggregation für UI-Klangtreue-Hinweis
        "hf_hallucination_guard": {
            "guard_fired_count": int((_meta.get("hf_hallucination_guard") or {}).get("guard_fired_count", 0) or 0),
            "phases_guarded": list((_meta.get("hf_hallucination_guard") or {}).get("phases_guarded", []) or []),
            "max_delta_ratio": _safe_float(
                (_meta.get("hf_hallucination_guard") or {}).get("max_delta_ratio", 0.0), 0.0
            ),
            "min_cap_hz": (
                _safe_float((_meta.get("hf_hallucination_guard") or {}).get("min_cap_hz", 0.0), 0.0)
                if isinstance((_meta.get("hf_hallucination_guard") or {}).get("min_cap_hz", None), (int, float))
                else None
            ),
        },
        # §2.46b Spectral Tilt Drift Guard: Treffer-Aggregation für UI-Klangtreue-Hinweis
        "spectral_tilt_guard": {
            "guard_fired_count": int((_meta.get("spectral_tilt_guard") or {}).get("guard_fired_count", 0) or 0),
            "phases_guarded": list((_meta.get("spectral_tilt_guard") or {}).get("phases_guarded", []) or []),
            "max_deviation_db_per_oct": _safe_float(
                (_meta.get("spectral_tilt_guard") or {}).get("max_deviation_db_per_oct", 0.0), 0.0
            ),
            "max_wet_cap_applied": _safe_float(
                (_meta.get("spectral_tilt_guard") or {}).get("max_wet_cap_applied", 0.0), 0.0
            ),
        },
        # §2.47b JND Sub-Threshold Phase Telemetrie — für Diagnose und UI
        "sub_threshold_phases": list(_meta.get("sub_threshold_phases", []) or []),
        # §2.47 ML-Fallback-Transparenz — Invariante: Kein ML-Failure darf Pipeline abbrechen
        "ml_fallbacks_used": [
            {
                "phase": str(fb.get("phase", "") or ""),
                "model": str(fb.get("model", "") or ""),
                "fallback": str(fb.get("fallback", "") or ""),
                "reason": str(fb.get("reason", "") or ""),
            }
            for fb in (list(_meta["ml_fallbacks_used"]) if isinstance(_meta.get("ml_fallbacks_used"), list) else [])
            if isinstance(fb, dict)
        ],
        # §0d Carrier-Chain-Recovery-Ratio — Pflichtfeld
        "carrier_chain_recovery_ratio": _safe_float(_meta.get("carrier_chain_recovery_ratio", 0.0), 0.0),
        "carrier_reference_shifted": bool(_meta.get("reference_shifted", False)),
    }


# ---------------------------------------------------------------------------
# Goal Feedback Recording (§C10 Bayesian EMA Calibration)
# ---------------------------------------------------------------------------


def record_goal_feedback(
    winning_goals: list[str],
    failing_goals: list[str],
    rating_thumbs_up: bool = True,
    genre: str = "",
    material: str = "",
    era: str = "",
) -> None:
    """§C10 Record listener thumbs-up/down feedback for Bayesian EMA calibration.

    Stores a UserFeedbackEntry and updates per-goal EMA nudges in
    sessions/goal_feedback.json (non-blocking — errors are logged, not raised).
    """
    try:
        from backend.core.song_goal_importance import (  # type: ignore[import]
            UserFeedbackEntry,
            get_feedback_store,
        )

        entry = UserFeedbackEntry(
            genre=str(genre or ""),
            material=str(material or ""),
            era=str(era or ""),
            rating_thumbs_up=bool(rating_thumbs_up),
            winning_goals=list(winning_goals or []),
            failing_goals=list(failing_goals or []),
        )
        get_feedback_store().record_feedback(entry)
    except Exception as _fb_exc:
        logger.warning("§C10 aufzeichnen_goal_feedback fehlgeschlagen: %s", _fb_exc)


# ---------------------------------------------------------------------------
# Public API — explizite Export-Liste
# ---------------------------------------------------------------------------

__all__ = [
    # MusicalGoalsChecker (§8.1)
    "get_musical_goals_checker",
    # Adaptive Goal Thresholds + Config (§2.31)
    "get_adaptive_goals_fn",
    # MUSHRA-Evaluator-Singleton (§8.1.1 OQS)
    "get_mushra_evaluator",
    # PerceptualQualityScorer-Singleton (§8.1 PQS)
    "get_perceptual_quality_scorer",
    # Experience Insights — Joy/Fatigue/Frisson, Empfehlungen, Recovery-Certainty
    "get_experience_insights",
    "resolve_pipeline_fail_reason",
    # Goal Feedback Recording (§C10 Bayesian EMA Calibration)
    "record_goal_feedback",
]
