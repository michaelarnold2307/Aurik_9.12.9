"""
debug_api.py — Hochrangige Debug-API für Aurik Pipeline-Telemetrie.

LEGACY_NON_RELEASE: Debug-/Forensik-API für Telemetrie-Auswertung. Diese Datei
ist kein Release-Einstieg und darf den kanonischen Bridge/Denker/Exporter-
Vertrag der Desktop-Produktpfade nicht umgehen.

Einheitlicher Zugriffspunkt für alle Debug-Daten aus einem RestorationResult.

Usage:
    from backend.api.debug_api import get_debug_summary, format_full_report

    result = restorer.restore(audio, sr, enable_debug_trace=True)
    print(format_full_report(result))
    summary = get_debug_summary(result)
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def get_debug_summary(result: Any) -> dict[str, Any]:
    """
    Gibt alle Debug-relevanten Daten aus einem RestorationResult als strukturiertes Dict zurück.

    Kein Raten: alle Werte sind direkt aus dem Result extrahiert, mit Fallback-Defaults.
    """
    meta = _get_meta(result)
    config = getattr(result, "config", None)
    material = getattr(result, "material_type", None)

    # robusten Zugriff auf material/value (vermeidet AttributeError für None)
    if material is None:
        _material_str = ""
    else:
        _mat_val = getattr(material, "value", None)
        _material_str = str(_mat_val) if _mat_val is not None else str(material)

    summary: dict[str, Any] = {
        # --- Grundinfos ---
        "material": _material_str,
        "mode": str(getattr(config, "mode", None) or ""),
        "era_decade": str(getattr(result, "era_decade", "") or ""),
        "restorability": _safe_float(getattr(result, "restorability", 0)),
        "total_time_s": _safe_float(getattr(result, "total_time_seconds", 0)),
        "rt_factor": _safe_float(getattr(result, "rt_factor", 0)),
        "quality_estimate": _safe_float(getattr(result, "quality_estimate", 0)),
        "confidence": _safe_float(getattr(result, "confidence", 0)),
        # --- Phase-Statistiken ---
        "phases_executed": list(getattr(result, "phases_executed", []) or []),
        "phases_skipped": list(getattr(result, "phases_skipped", []) or []),
        "phase_gate_log": list(getattr(result, "phase_gate_log", []) or []),
        "phases_executed_count": len(getattr(result, "phases_executed", []) or []),
        "phases_skipped_count": len(getattr(result, "phases_skipped", []) or []),
        # --- Musical Goals ---
        "final_goals": _safe_goals(getattr(result, "musical_goals", None)),
        "adaptive_thresholds": _safe_goals(getattr(result, "adaptive_thresholds", None)),
        # --- Fail-Reasons / Warnings ---
        "fail_reasons": _safe_list(meta.get("fail_reasons", [])),
        "warnings": list(getattr(result, "warnings", []) or []),
        # --- Pipeline-Insights ---
        "team_coordination": _safe_dict(meta.get("team_coordination", {})),
        "interaction_guard": _safe_dict(meta.get("interaction_guard", {})),
        "sub_threshold_phases": _safe_list(meta.get("sub_threshold_phases", [])),
        "ml_fallbacks_used": _safe_list(meta.get("ml_fallbacks_used", [])),
        # --- Experience ---
        "joy_runtime_index": _safe_dict(meta.get("joy_runtime_index", {})),
        "auto_improvement_recommendations": _safe_list(meta.get("auto_improvement_recommendations", [])),
        # --- Song-Calibration ---
        "song_calibration": _safe_dict(meta.get("song_calibration", {})),
        # --- Carrier-Chain ---
        "carrier_chain_recovery_ratio": _safe_float(meta.get("carrier_chain_recovery_ratio", 0)),
        # --- Debug-Trace (vorhanden wenn enable_debug_trace=True) ---
        "has_phase_goal_data": "pmgg_log_entries" in meta and bool(meta["pmgg_log_entries"]),
        "pmgg_log_entries_count": len(meta.get("pmgg_log_entries", []) or []),
        # --- Source-Material-Baseline (§2.50) ---
        "source_material_baseline": _safe_dict(meta.get("source_material_baseline", {})),
        # --- Goosebumps ---
        "goosebumps_score": _safe_float(getattr(result, "goosebumps_score", 0)),
        # --- Chroma / Loudness ---
        "chroma_correlation": _safe_float(getattr(result, "chroma_correlation", 0)),
        "lufs_delta": _safe_float(getattr(result, "lufs_delta", 0)),
    }

    return summary


def get_goals_timeline(result: Any) -> dict[str, list[float]]:
    """
    Gibt die 15 Musical-Goals als Zeitreihe über alle Phasen zurück.

    Benötigt enable_debug_trace=True im restore()-Aufruf (sonst leer).
    """
    try:
        from backend.core.pipeline_trace import build_from_result

        trace = build_from_result(result)
        return trace.goal_timeline  # type: ignore[no-any-return]
    except Exception as e:
        logger.debug("get_goals_timeline fehlgeschlagen: %s", e)
        return {}


def get_phase_decisions(result: Any) -> list[dict[str, Any]]:
    """
    Gibt alle Phasen-Entscheidungen (Gate-Decision, Strength, Retries, Regressions) zurück.

    Benötigt enable_debug_trace=True für vollständige Goal-Daten.
    """
    try:
        from backend.core.pipeline_trace import build_from_result

        trace = build_from_result(result)
        return [p.to_dict() for p in trace.phases]
    except Exception as e:
        logger.debug("get_Verarbeitungsschritt_decisions fehlgeschlagen: %s", e)
        return []


def format_goals_table(result: Any) -> str:
    """
    Gibt die ASCII Goal-Matrix zurück: 15 Goals × Phasen.
    Benötigt enable_debug_trace=True für vollständige Daten.
    """
    try:
        from backend.core.pipeline_trace import build_from_result
        from backend.core.pipeline_trace import format_goals_table as _fmt

        trace = build_from_result(result)
        return _fmt(trace)  # type: ignore[no-any-return]
    except Exception as e:
        return f"(format_goals_table fehlgeschlagen: {e})"


def format_full_report(result: Any) -> str:
    """
    Gibt einen vollständigen menschenlesbaren Debug-Bericht zurück.

    Empfohlen mit enable_debug_trace=True für vollständige Goal-Zeitreihen.
    Funktioniert auch ohne (eingeschränkte Daten, post_hoc-Modus).
    """
    try:
        from backend.core.pipeline_trace import build_from_result, store_trace
        from backend.core.pipeline_trace import format_full_report as _fmt

        trace = build_from_result(result)
        store_trace(trace)  # Für get_last_trace() Zugriff
        return _fmt(trace)  # type: ignore[no-any-return]
    except Exception as e:
        return f"(format_full_report fehlgeschlagen: {e})"


def build_from_result(result: Any) -> Any:
    """Baut den RestoreTrace aus einem RestorationResult.

    §V4-konformer Zugang: CLI/UI importiert pipeline_trace nie direkt,
    sondern nur über die backend.api-Schicht.
    """
    from backend.core.pipeline_trace import build_from_result as _build

    return _build(result)


def format_goal_deltas(trace: Any) -> str:
    """Goal-Deltas als Text (Trace-basiert, wie pipeline_trace)."""
    try:
        from backend.core.pipeline_trace import format_goal_deltas as _fmt

        return _fmt(trace)  # type: ignore[no-any-return]
    except Exception as e:
        return f"(format_goal_deltas fehlgeschlagen: {e})"


def format_phase_decisions(trace: Any) -> str:
    """Phasen-Entscheidungen als Text (Trace-basiert, wie pipeline_trace)."""
    try:
        from backend.core.pipeline_trace import format_phase_decisions as _fmt

        return _fmt(trace)  # type: ignore[no-any-return]
    except Exception as e:
        return f"(format_phase_decisions fehlgeschlagen: {e})"


def save_trace_json(result: Any, path: str) -> bool:
    """
    Speichert den vollständigen Trace als JSON-Datei.

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    try:
        from backend.core.pipeline_trace import build_from_result, store_trace

        trace = build_from_result(result)
        store_trace(trace)
        with open(path, "w", encoding="utf-8") as f:
            f.write(trace.to_json())
        logger.info("Pipeline-Trace gespeichert: %s", path)
        return True
    except Exception as e:
        logger.error("speichern_trace_json fehlgeschlagen (%s): %s", path, e)
        return False


def get_worst_phases(result: Any, n: int = 5) -> list[dict[str, Any]]:
    """
    Gibt die N Phasen mit den stärksten Goal-Regressionen zurück.

    Nützlich für schnelles Debugging: Wo verliert die Pipeline am meisten?
    """
    try:
        from backend.core.pipeline_trace import build_from_result

        trace = build_from_result(result)
        phases_with_regressions = [
            (p, sum(abs(v) for v in p.goal_regressions.values())) for p in trace.phases if p.goal_regressions
        ]
        phases_with_regressions.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "phase_id": p.phase_id,
                "gate_decision": p.gate_decision,
                "total_regression": round(total, 4),
                "regressions": {k: round(v, 4) for k, v in p.goal_regressions.items()},
            }
            for p, total in phases_with_regressions[:n]
        ]
    except Exception as e:
        logger.debug("get_worst_phases fehlgeschlagen: %s", e)
        return []


def get_goal_fails(result: Any, mode: str | None = None) -> list[dict[str, Any]]:
    """
    Gibt alle Goals zurück, die am Pipeline-Ende unter dem Schwellwert liegen.

    Args:
        result: RestorationResult
        mode: "restoration" oder "studio_2026" (auto-detektiert wenn None)
    """
    try:
        from backend.api.bridge import normalize_user_mode
        from backend.core.pipeline_trace import (
            CANONICAL_GOALS,
            RESTORATION_THRESHOLDS,
            STUDIO_THRESHOLDS,
            build_from_result,
        )

        trace = build_from_result(result)
        _canonical_mode = normalize_user_mode(mode or trace.mode or "restoration")
        thresholds = STUDIO_THRESHOLDS if _canonical_mode == "Studio 2026" else RESTORATION_THRESHOLDS
        fails = []
        for g in CANONICAL_GOALS:
            val = trace.final_goals.get(g)
            thr = thresholds.get(g, 0.0)
            if val is not None and val < thr:
                fails.append(
                    {
                        "goal": g,
                        "value": round(val, 4),
                        "threshold": thr,
                        "delta": round(val - thr, 4),
                    }
                )
        return sorted(fails, key=lambda x: x["delta"])
    except Exception as e:
        logger.debug("get_goal_fails fehlgeschlagen: %s", e)
        return []


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _get_meta(result: Any) -> dict[str, Any]:
    meta = getattr(result, "metadata", {})
    return meta if isinstance(meta, dict) else {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return 0.0 if not math.isfinite(f) else round(f, 4)
    except (TypeError, ValueError):
        return default


def _safe_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [str(v)]


def _safe_dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _safe_goals(v: Any) -> dict[str, float]:
    if not isinstance(v, dict):
        return {}
    return {str(k): _safe_float(val) for k, val in v.items() if val is not None}
