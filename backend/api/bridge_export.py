"""Aurik 10 — Bridge: Export-Guard (§3.1 Spec 08)
===================================================
Export-Absicherung für Frontend/CLI → Backend-Core.

Enthält:
  - export_guard (NaN/Inf-frei, [-1,1] geclippt) — PFLICHT vor jedem sf.write
  - Export-Transparenz (§v10.700 A5): Resample-Kette, True-Peak, LUFS, Dateigröße
  - validate_export_quality (chroma_correlation ≥ 0.80 catastrophic check)
  - build_export_quality_gate_payload (export_workflow-compatible payload)
  - build_export_metadata (ExportMetadata mit fidelity_guards Telemetrie)

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
# Export-Guard (PFLICHT vor jedem sf.write / AudioExporter.export)
# ---------------------------------------------------------------------------


def export_guard(audio: np.ndarray) -> np.ndarray:
    """Stellt sicher, dass Audio NaN/Inf-frei und auf [-1, 1] geclippt ist.

    Muss vor jedem ``sf.write()`` oder ``AudioExporter.export()`` aufgerufen
    werden. Entspricht der Numerischen Robustheit-Pflicht (§3.1 Spec 08).

    Args:
        audio: Audio-Array (float32 oder float64).

    Returns:
        Bereinigtes Audio (float32, kein NaN/Inf, Werte ∈ [-1, 1]).
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


# ---------------------------------------------------------------------------
# Export-Transparenz (§v10.700 A5) — Resample-Kette, True-Peak, LUFS, Dateigröße
# ---------------------------------------------------------------------------


def get_export_transparency(
    input_path: str = "",
    output_path: str = "",
    output_audio: np.ndarray | None = None,
    output_sr: int = 48000,
    original_audio: np.ndarray | None = None,
    original_sr: int = 48000,
    export_bit_depth: int = 24,
    export_format: str = "FLAC",
    dither_method: str = "POW-r 3",
) -> dict[str, Any]:
    """§v10.700 A5: Export-Transparenz — berechnet Export-Metadaten für die GUI.

    Liefert: Resample-Kette, True-Peak, LUFS, Dateigröße vorher/nachher, Dither.
    Kann von CLI und GUI nach dem Export aufgerufen werden.
    """
    import os
    from pathlib import Path

    _report: dict[str, Any] = {
        "resample_chain": f"{original_sr} Hz → {output_sr} Hz"
        if original_sr != output_sr
        else f"{original_sr} Hz (kein Resampling)",
        "resample_method": "Lanczos-4 (scipy.signal.resample_poly)" if original_sr != output_sr else "—",
        "export_format": export_format,
        "export_bit_depth": export_bit_depth,
        "dither_method": dither_method if export_bit_depth < 32 else "— (32-bit, kein Dither nötig)",
    }

    # True-Peak
    if output_audio is not None:
        try:
            from backend.core.audio_exporter import _approx_true_peak

            _tp_db = float(_approx_true_peak(output_audio, output_sr))
            _report["true_peak_dbtp"] = round(_tp_db, 2)
            _report["true_peak_ok"] = _tp_db <= -1.0  # EBU R128: ≤ -1 dBTP
        except Exception as _tp_exc:
            logger.warning("§G93 bridge: True-Peak-Messung fehlgeschlagen → None: %s", _tp_exc, exc_info=True)
            _report["true_peak_dbtp"] = None
            _report["true_peak_ok"] = None

    # LUFS
    if output_audio is not None:
        try:
            mono = output_audio if output_audio.ndim == 1 else output_audio.mean(axis=-1)
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)) + 1e-12)
            _lufs_approx = float(20.0 * np.log10(rms)) - 0.0  # RMS ≈ LUFS für stationäre Signale
            _report["integrated_lufs"] = round(_lufs_approx, 1)
        except Exception as _lufs_exc:
            logger.warning("§G93 bridge: LUFS-Messung fehlgeschlagen → None: %s", _lufs_exc, exc_info=True)
            _report["integrated_lufs"] = None

    # Dateigröße
    if input_path and Path(input_path).exists():
        _report["input_size_mb"] = round(os.path.getsize(input_path) / (1024 * 1024), 2)
        _report["input_size_label"] = f"{_report['input_size_mb']:.1f} MB"
    if output_path and Path(output_path).exists():
        _report["output_size_mb"] = round(os.path.getsize(output_path) / (1024 * 1024), 2)
        _report["output_size_label"] = f"{_report['output_size_mb']:.1f} MB"
        if "input_size_mb" in _report:
            _delta = _report["output_size_mb"] - _report["input_size_mb"]
            _report["size_delta_label"] = f"{_report['input_size_label']} → {_report['output_size_label']}"

    return _report


# ---------------------------------------------------------------------------
# validate_export_quality — chroma_correlation ≥ 0.80 catastrophic check
# ---------------------------------------------------------------------------


def validate_export_quality(result: object) -> tuple[bool, list[str]]:
    """Validiert export quality based on RestorationResult fields.

    Delegates to :func:`backend.exporter.validate_export_quality`.
    Returns ``(passed, warnings)`` — *passed* is False only on catastrophic
    tonal shift (chroma < 0.80).
    """
    try:
        from backend.exporter import validate_export_quality as _veq

        return _veq(result)  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("validieren_Ausgabe_quality nicht verfuegbar -> fail-closed: %s", exc)
        return False, ["Bridge-Export-Gate nicht verfügbar (fail-closed)"]


# ---------------------------------------------------------------------------
# build_export_quality_gate_payload — export_workflow-compatible payload
# ---------------------------------------------------------------------------


def build_export_quality_gate_payload(result: object) -> dict[str, Any]:
    """Erstellt export_workflow-compatible quality_gate payload from a result object.

    This is the canonical bridge-side payload builder used by frontend/CLI callers
    before calling ``backend.core.export_workflow.export_audio``.
    """
    passed, warnings = validate_export_quality(result)
    meta_raw = getattr(result, "metadata", None)
    meta: dict[str, Any] = _coerce_dict_str_any(meta_raw)

    fail_reasons: list[Any] = _coerce_list_any(meta.get("fail_reasons"))
    primary_fail_reason = str(meta.get("fail_reason", "") or "")
    degradation_status = str(meta.get("degradation_status", "") or "")
    fqf = _coerce_dict_str_any(meta.get("fallback_quality_floor"))
    export_gate_profile = str(meta.get("export_gate_profile", "") or "")
    export_gate_material = str(meta.get("export_gate_material", "") or "")
    _export_gate_thresholds_raw = meta.get("export_gate_thresholds")
    export_gate_thresholds = _coerce_dict_str_any(_export_gate_thresholds_raw)
    _export_gate_signal_signature_raw = meta.get("export_gate_signal_signature")
    export_gate_signal_signature = _coerce_dict_str_any(_export_gate_signal_signature_raw)
    export_gate_preserve_signal = float(np.clip(float(meta.get("export_gate_preserve_signal", 0.0) or 0.0), 0.0, 1.0))

    _degradation_norm = degradation_status.strip().lower()
    _has_structured_gate_issue = _degradation_norm not in {"", "ok"} or bool(fail_reasons)
    if _has_structured_gate_issue:
        passed = False

    fqf_triggered = bool(fqf.get("triggered", False))
    fqf_status = str(fqf.get("status", "")).strip().lower()
    fqf_attempts_raw = fqf.get("attempts", 0)
    fqf_attempts = int(fqf_attempts_raw) if isinstance(fqf_attempts_raw, (int, float)) else 0

    # Deterministic coupling: if fallback floor indicates recovered/degraded, do not
    # emit a contradictory passed=True payload.
    if fqf_triggered and fqf_status in {"recovered", "degraded", "failed", "fail"}:
        passed = False
        if not primary_fail_reason:
            primary_fail_reason = str(
                fqf.get("reason", "fallback_quality_floor_triggered") or "fallback_quality_floor_triggered"
            )
        if not degradation_status:
            degradation_status = "recovered" if fqf_status == "recovered" else "degraded"

    if not primary_fail_reason and fail_reasons:
        first = fail_reasons[0]
        if isinstance(first, dict):
            primary_fail_reason = str(first.get("error_code", "QUALITY_GATE_FAILED") or "QUALITY_GATE_FAILED")
    if not primary_fail_reason and warnings:
        primary_fail_reason = str(warnings[0])

    if not degradation_status:
        degradation_status = "ok" if passed else "degraded"

    # Music-Lover Telemetrie: liefert musikalisch relevante Exportindikatoren
    # für UI/Reporter, ohne bestehende Gate-Semantik zu verändern.
    _goals_meta: dict[str, Any] = _coerce_dict_str_any(meta.get("musical_goals"))
    _goal_scores: dict[str, Any] = _coerce_dict_str_any(_goals_meta.get("scores"))
    _goal_thresholds: dict[str, Any] = _coerce_dict_str_any(_goals_meta.get("thresholds"))
    _goal_gaps: list[dict[str, Any]] = []
    for _goal_name, _thr_val in _goal_thresholds.items():
        try:
            _gap = max(0.0, float(_thr_val) - float(_goal_scores.get(_goal_name, 0.0)))
        except Exception:
            logger.debug("bridge: Goal-Gap-Berechnung für '%s' fehlgeschlagen, nutze 0.0", _goal_name)
            _gap = 0.0
        if _gap > 0.0:
            _goal_gaps.append({"goal": str(_goal_name), "gap": round(float(_gap), 4)})
    _goal_gaps.sort(key=lambda e: float(e.get("gap", 0.0)), reverse=True)

    _temporal_cont: dict[str, Any] = _coerce_dict_str_any(meta.get("temporal_continuity"))
    _temporal_hotspots: list[dict[str, Any]] = []
    for _phase_id, _entry in _temporal_cont.items():
        if not isinstance(_entry, dict):
            continue
        try:
            _gain_step = float(_entry.get("gain_step_db", 0.0) or 0.0)
            _variance_ratio = float(_entry.get("variance_ratio", 1.0) or 1.0)
        except Exception:
            logger.debug(
                "bridge: Temporal-Continuity für '%s' konnte nicht geparst werden, nutze Standardwerte", _phase_id
            )
            _gain_step = 0.0
            _variance_ratio = 1.0
        _hot = (abs(_gain_step) > 1.5) or (_variance_ratio > 2.5)
        if _hot:
            _temporal_hotspots.append(
                {
                    "phase": str(_phase_id),
                    "gain_step_db": round(_gain_step, 3),
                    "variance_ratio": round(_variance_ratio, 3),
                }
            )
    _temporal_hotspots = _temporal_hotspots[:5]

    _vqi_val = float(meta.get("vqi", getattr(result, "vqi", 0.0)) or 0.0)
    _sid_val = float(meta.get("singer_identity_cosine", 0.0) or 0.0)
    _qe_val = float(getattr(result, "quality_estimate", 0.0) or 0.0)
    _chroma_val = float(getattr(result, "chroma_correlation", 0.0) or 0.0)
    _lufs_delta_val = float(getattr(result, "lufs_delta", 0.0) or 0.0)

    _mcg: dict[str, Any] = _coerce_dict_str_any(meta.get("model_capability_report"))
    _mcg_summary: dict[str, Any] = _coerce_dict_str_any(_mcg.get("summary"))
    _all_sota_raw = _mcg_summary.get("all_sota_real")
    _vocal_cap_status = str(meta.get("vocal_restoration_capability_status", "") or "")
    _all_sota_real = True
    if isinstance(_all_sota_raw, bool):
        _all_sota_real = bool(_all_sota_raw)
    if _vocal_cap_status and _vocal_cap_status != "sota_real":
        _all_sota_real = False

    _degraded_caps = _coerce_list_any(_mcg_summary.get("degraded_capabilities"))
    _wcs_gate = _coerce_dict_str_any(meta.get("worldclass_composite_gate"))
    _threshold_evidence = _coerce_dict_str_any(meta.get("threshold_evidence"))
    _qe_threshold = float(export_gate_thresholds.get("quality_estimate", 0.0) or 0.0)
    _root_cause = str(primary_fail_reason or "").strip()
    _root_cause_l = _root_cause.lower()
    _pipeline_like_failure = (
        _root_cause_l.startswith("pipeline_blocked:")
        or "pipeline-fehler" in _root_cause_l
        or "pipeline_fehler" in _root_cause_l
        or "unexpected keyword argument" in _root_cause_l
        or "missing 1 required positional argument" in _root_cause_l
    )
    _failure_class = "none"
    if degradation_status in {"blocked", "critical_degraded", "degraded"}:
        if _pipeline_like_failure or (_qe_threshold <= 0.0001 and bool(_root_cause)):
            _failure_class = "technical_failure"
        else:
            _failure_class = "quality_failure"
    if _root_cause_l.startswith("pipeline_blocked:"):
        _root_cause = _root_cause.split(":", 1)[1].strip() or _root_cause

    _manual_action_required = False
    _allowed_user_decisions = ["mode_selection"]
    _export_policy = "normal_export"
    _confidence_level = "hoch"
    _listener_message = "Aurik hat die Restaurierung autonom als gehoersicher freigegeben."
    if degradation_status in {"blocked", "critical_degraded", "degraded"}:
        _export_policy = "input_or_best_safe_checkpoint"
        _confidence_level = "geschuetzt"
        _listener_message = (
            "Aurik hat ein Hoerrisiko erkannt und schuetzt den Nutzer mit dem besten sicheren Checkpoint."
        )
    elif degradation_status == "recovered" or fqf_status == "recovered":
        _export_policy = "best_available_restoration"
        _confidence_level = "begrenzt"
        _listener_message = (
            "Aurik hat die bestmoegliche Restaurierung erreicht und verbleibende Grenzen transparent markiert."
        )

    payload = {
        "passed": bool(passed),
        "fail_reason": primary_fail_reason,
        "root_cause": _root_cause,
        "failure_class": _failure_class,
        "fail_reasons": list(fail_reasons),
        "required_gates": ["musical_goals", "pqs", "oqs", "fallback_quality_floor"],
        "recovery_attempted": bool(fqf_attempts > 0),
        "best_possible_reached": bool(fqf_status == "recovered"),
        "degradation_status": degradation_status,
        "fallback_quality_floor": dict(fqf) if fqf else {},
        "profile": export_gate_profile,
        "material": export_gate_material,
        "preserve_signal": export_gate_preserve_signal,
        "thresholds": {
            "quality_estimate": _qe_threshold,
            "level_drop_db": float(export_gate_thresholds.get("level_drop_db", 0.0) or 0.0),
        },
        "signal_signature": {
            "crest_db": float(export_gate_signal_signature.get("crest_db", 0.0) or 0.0),
            "hf_ratio": float(export_gate_signal_signature.get("hf_ratio", 0.0) or 0.0),
            "transient_ratio": float(export_gate_signal_signature.get("transient_ratio", 0.0) or 0.0),
            "micro_dynamic_db": float(export_gate_signal_signature.get("micro_dynamic_db", 0.0) or 0.0),
        },
        "worldclass_composite_gate": {
            "wcs": float(np.clip(float(_wcs_gate.get("wcs", 0.0) or 0.0), 0.0, 1.0)),
            "threshold": float(np.clip(float(_wcs_gate.get("threshold", 0.0) or 0.0), 0.0, 1.0)),
            "profile": str(_wcs_gate.get("profile", "") or ""),
            "artifact_veto": bool(_wcs_gate.get("artifact_veto", False)),
            "passed": bool(_wcs_gate.get("passed", False)),
        },
        "threshold_evidence": dict(_threshold_evidence) if _threshold_evidence else {},
        "user_confidence_summary": {
            "confidence_level": _confidence_level,
            "listener_message": _listener_message,
            "manual_action_required": _manual_action_required,
            "allowed_user_decisions": list(_allowed_user_decisions),
            "export_policy": _export_policy,
            "why_user_can_trust": [
                "Export-Gates pruefen Hoerschutz, Musical Goals und Fallback Quality Floor.",
                "Aurik faellt bei Risiko auf den besten sicheren Zustand zurueck.",
                "Der Nutzer muss keine Klangparameter setzen; nur die Moduswahl ist erlaubt.",
            ],
        },
        "musiclover": {
            "vocal_integrity": {
                "vqi": _vqi_val,
                "singer_identity_cosine": _sid_val,
                "vqi_tier": str(meta.get("vqi_tier", "") or ""),
                "vocal_no_harm_rollback": bool(meta.get("vocal_no_harm_rollback", False)),
            },
            "musical_goals": {
                "remaining_count": int(len(_goal_gaps)),
                "top_remaining_goals": list(_goal_gaps[:3]),
            },
            "stereo_integrity": {
                "mono_compatibility_warning": bool(meta.get("mono_compatibility_warning", False)),
            },
            "temporal_risk": {
                "hotspot_count": int(len(_temporal_hotspots)),
                "phase_hotspots": list(_temporal_hotspots),
            },
            "mastering": {
                "quality_estimate": _qe_val,
                "chroma_correlation": _chroma_val,
                "lufs_delta": _lufs_delta_val,
            },
            "decision_trace": {
                "degradation_status": str(degradation_status),
                "fail_reason": str(primary_fail_reason),
                "fail_reason_count": int(len(fail_reasons)),
                "recovery_attempted": bool(fqf_attempts > 0),
                "export_policy": _export_policy,
                "all_sota_real": bool(_all_sota_real),
                "vocal_restoration_capability_status": _vocal_cap_status,
                "degraded_capabilities": list(_degraded_caps) if isinstance(_degraded_caps, list) else [],
            },
        },
        "warnings": [str(w) for w in warnings],
    }

    try:
        meta_obj = getattr(result, "metadata", None)
        if isinstance(meta_obj, dict):
            meta_obj.setdefault("fail_reason", primary_fail_reason)
            meta_obj.setdefault("degradation_status", degradation_status)
            if fail_reasons and not isinstance(meta_obj.get("fail_reasons"), list):
                meta_obj["fail_reasons"] = list(fail_reasons)
            meta_obj["quality_gate_payload"] = payload
            meta_obj["export_quality_gate_payload"] = payload
        elif meta_obj is None and hasattr(result, "metadata"):
            result.metadata = {  # type: ignore[attr-defined]
                "fail_reason": primary_fail_reason,
                "degradation_status": degradation_status,
                "fail_reasons": list(fail_reasons),
                "quality_gate_payload": payload,
                "export_quality_gate_payload": payload,
            }
    except Exception as exc:
        logger.debug("build_Ausgabe_quality_gate_payload mirror uebersprungen: %s", exc)

    return payload


# ---------------------------------------------------------------------------
# build_export_metadata — ExportMetadata mit fidelity_guards Telemetrie
# ---------------------------------------------------------------------------


def build_export_metadata(result: object, **tag_kwargs):
    """Erstellt an ExportMetadata instance populated with fidelity-guard telemetry.

    Reads ``spectral_tilt_guard`` and ``hf_hallucination_guard`` from
    ``result.metadata`` (both written by UV3) and stores them under the
    ``fidelity_guards`` field so they appear in the JSON sidecar written by
    ``backend.core.export_workflow._write_metadata_sidecar``.

    Args:
        result: RestorationResult (or any object with a ``.metadata`` dict).
        **tag_kwargs: Optional id-tag overrides forwarded to ExportMetadata
                      (title, artist, album, …).

    Returns:
        Populated ExportMetadata instance (``fidelity_guards`` is None when
        both guards are absent from result metadata).
    """
    from backend.core.export_workflow import ExportMetadata

    meta = getattr(result, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}

    def _safe_guard(raw: object) -> dict | None:
        """Gibt guard dict with only JSON-safe numeric / list values, or None zurück."""
        if not isinstance(raw, dict):
            return None
        out: dict = {}
        for k, v in raw.items():
            if isinstance(v, (int, float, str, bool)):
                try:
                    import math

                    out[k] = 0.0 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
                except Exception:
                    logger.warning("bridge.py::_safe_guard Ersatzpfad", exc_info=True)
            elif isinstance(v, (list, tuple)):
                out[k] = [str(x) for x in v]
        return out or None

    _stg = _safe_guard(meta.get("spectral_tilt_guard"))
    _hfg = _safe_guard(meta.get("hf_hallucination_guard"))
    _guards: dict | None = None
    if _stg is not None or _hfg is not None:
        _guards = {}
        if _stg is not None:
            _guards["spectral_tilt_guard"] = _stg
        if _hfg is not None:
            _guards["hf_hallucination_guard"] = _hfg

    em = ExportMetadata(
        title=tag_kwargs.get("title") or None,
        artist=tag_kwargs.get("artist") or None,
        album=tag_kwargs.get("album") or None,
        date=tag_kwargs.get("date") or None,
        genre=tag_kwargs.get("genre") or None,
        comment=tag_kwargs.get("comment") or None,
        fidelity_guards=_guards,
    )
    return em


# ---------------------------------------------------------------------------
# Public API — explizite Export-Liste
# ---------------------------------------------------------------------------

__all__ = [
    # NaN/Inf-Guard (PFLICHT vor jedem sf.write / AudioExporter.export)
    "export_guard",
    # Export-Transparenz (§v10.700 A5)
    "get_export_transparency",
    # validate_export_quality — chroma_correlation ≥ 0.80 catastrophic check
    "validate_export_quality",
    # build_export_quality_gate_payload — export_workflow-compatible payload
    "build_export_quality_gate_payload",
    # build_export_metadata — ExportMetadata mit fidelity_guards Telemetrie
    "build_export_metadata",
]
