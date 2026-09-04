"""Real-audio Golden-Set gate for executed strategy and export safety.

This gate validates the layer after autonomous strategy planning: the strategy
manifest is converted into a precomputed phase plan, UV3 executes that plan on
real audio, and the resulting metadata plus export contract are checked for
phase execution, no-harm gates, artifact freedom, HPI/VQI, and safe export.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from backend.api.bridge import build_export_quality_gate_payload
from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES
from backend.core.export_workflow import ExportMetadata, export_audio
from backend.core.performance_guard import QualityMode
from backend.core.real_audio_defect_golden_gate import _load_manifest
from backend.core.real_audio_strategy_golden_gate import StrategyCaseResult, _scan_strategy_case
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3
from backend.file_import import load_audio_file


@dataclass(frozen=True)
class ExecutionGateThresholds:
    """Thresholds for the real-audio execution/export Golden-Set gate."""

    min_phase_execution_recall: float = 0.90
    min_phase_delta_coverage: float = 0.75
    min_artifact_contract_rate: float = 1.0
    min_hpi_contract_rate: float = 1.0
    min_vocal_contract_rate: float = 1.0
    min_export_contract_rate: float = 1.0
    max_forbidden_phase_executions: int = 0
    max_runtime_factor: float = 25.0
    runtime_duration_floor_seconds: float = 4.0


@dataclass(frozen=True)
class ExecutionCaseResult:
    """Serialisierbares Ausführungs-/Export-Bewertungsergebnis pro Fall."""

    case_id: str
    required_phases: tuple[str, ...]
    planned_phases: tuple[str, ...]
    phases_executed: tuple[str, ...]
    phases_skipped: tuple[str, ...]
    missing_required_executions: tuple[str, ...]
    forbidden_executed: tuple[str, ...]
    phase_delta_phases: tuple[str, ...]
    missing_phase_deltas: tuple[str, ...]
    artifact_freedom: float | None
    artifact_contract_passed: bool
    hpi: float | None
    hpi_contract_passed: bool
    vqi: float | None
    vocal_required: bool
    vocal_contract_passed: bool
    export_contract_passed: bool
    export_strategy: str
    export_blocked: bool
    degradation_status: str
    fail_reasons: tuple[str, ...]
    runtime_seconds: float
    duration_seconds: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExecutionGateResult:
    """Aggregate execution/export gate result."""

    passed: bool
    phase_execution_recall: float
    phase_delta_coverage: float
    artifact_contract_rate: float
    hpi_contract_rate: float
    vocal_contract_rate: float
    export_contract_rate: float
    forbidden_phase_executions: int
    runtime_factor: float
    fail_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RealAudioExecutionGoldenGateReport:
    """Serialisierbares Ergebnis eines Real-Audio-Ausführungs-/Export-Gate-Laufs."""

    gate: ExecutionGateResult
    cases: list[ExecutionCaseResult]
    skipped_cases: list[dict[str, str]]
    manifest_path: str
    scanned_cases: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Gibt a JSON-serializable representation zurück."""
        return {
            "gate": asdict(self.gate),
            "cases": [asdict(case) for case in self.cases],
            "skipped_cases": self.skipped_cases,
            "manifest_path": self.manifest_path,
            "scanned_cases": self.scanned_cases,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _thresholds_from_manifest(payload: dict[str, Any]) -> ExecutionGateThresholds:
    raw = payload.get("execution_thresholds")
    raw = raw if isinstance(raw, dict) else {}
    return ExecutionGateThresholds(
        min_phase_execution_recall=float(raw.get("min_phase_execution_recall", 0.90)),
        min_phase_delta_coverage=float(raw.get("min_phase_delta_coverage", 0.75)),
        min_artifact_contract_rate=float(raw.get("min_artifact_contract_rate", 1.0)),
        min_hpi_contract_rate=float(raw.get("min_hpi_contract_rate", 1.0)),
        min_vocal_contract_rate=float(raw.get("min_vocal_contract_rate", 1.0)),
        min_export_contract_rate=float(raw.get("min_export_contract_rate", 1.0)),
        max_forbidden_phase_executions=int(raw.get("max_forbidden_phase_executions", 0)),
        max_runtime_factor=float(raw.get("max_runtime_factor", 25.0)),
        runtime_duration_floor_seconds=float(raw.get("runtime_duration_floor_seconds", 4.0)),
    )


def _load_audio_for_execution(path: Path, target_sr: int, max_seconds: float) -> tuple[np.ndarray, int, float]:
    loaded = load_audio_file(str(path), target_sr=target_sr, mono=False, do_carrier_analysis=False)
    if not loaded or loaded.get("audio") is None:
        error = loaded.get("error") if isinstance(loaded, dict) else "unknown import error"
        raise RuntimeError(f"Audio import failed for {path}: {error}")
    audio = np.asarray(loaded["audio"], dtype=np.float32)
    sr = int(loaded.get("sr") or target_sr)
    duration = float(loaded.get("duration") or (audio.shape[0] / max(sr, 1)))
    if max_seconds > 0.0:
        audio = audio[: min(audio.shape[0], int(sr * max_seconds))]
    return np.nan_to_num(np.clip(audio, -1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0), sr, duration


def _export_shape(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] in {1, 2} and arr.shape[1] > 2:
        arr = arr.T
    return np.nan_to_num(np.clip(arr, -1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]


def _error_codes(fail_reasons: object) -> tuple[str, ...]:
    if not isinstance(fail_reasons, list):
        return ()
    codes: list[str] = []
    for entry in fail_reasons:
        if isinstance(entry, dict):
            code = str(entry.get("error_code", "") or "").strip()
            if code:
                codes.append(code)
    return tuple(codes)


def _contract_passed_for_failed_gate(passed: bool | None, fail_codes: tuple[str, ...], expected_code: str) -> bool:
    return bool(passed is True or expected_code in fail_codes)


def _check_export_contract(
    result: object,
    case_id: str,
    sr: int,
    output_dir: Path,
) -> tuple[bool, str, bool, dict[str, Any]]:
    payload = build_export_quality_gate_payload(result)
    result_meta = result.metadata if hasattr(result, "metadata") else None
    if not isinstance(result_meta, dict):
        result_meta = {}
    result_degradation = str(result_meta.get("degradation_status", "") or "").strip().lower()
    blocked = False
    sidecar_payload: dict[str, Any] = {}
    try:
        if not hasattr(result, "audio"):
            raise RuntimeError("Result object has no 'audio' attribute")
        export_path = export_audio(
            _export_shape(result.audio),
            sr,
            f"{case_id}_execution_gate",
            metadata=ExportMetadata(title=case_id, comment="real_audio_execution_golden_gate"),
            quality_gate=payload,
            output_dir=str(output_dir),
        )
        sidecar_path = Path(export_path).with_suffix(".json")
        if sidecar_path.exists():
            sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_ok = "quality_gate_passed" in sidecar_payload and "export_strategy" in sidecar_payload
        sidecar_degradation = str(sidecar_payload.get("quality_gate_degradation_status", "") or "").strip().lower()
        if result_degradation not in {"", "ok"} and sidecar_degradation in {"", "ok"}:
            return False, str(sidecar_payload.get("export_strategy", "unknown")), False, sidecar_payload
        if payload.get("passed") is False and not payload.get("recovery_attempted"):
            return False, str(sidecar_payload.get("export_strategy", "unknown")), False, sidecar_payload
        return bool(sidecar_ok), str(sidecar_payload.get("export_strategy", "success")), False, sidecar_payload
    except RuntimeError as e:
        logger.warning("§V6 Execution-Gate RuntimeError — contract fallback aktiviert: %s", e)
        blocked = True
        contract_ok = payload.get("passed") is False and not payload.get("recovery_attempted")
        return bool(contract_ok), "blocked", blocked, sidecar_payload


def _is_vocal_case(raw_case: dict[str, Any]) -> bool:
    if "requires_vocal_gate" in raw_case:
        return bool(raw_case.get("requires_vocal_gate"))
    text = " ".join(
        [
            str(raw_case.get("case_id", "") or ""),
            str(raw_case.get("path", "") or ""),
            str(raw_case.get("description", "") or ""),
        ]
    ).lower()
    return any(token in text for token in ("vocal", "choir", "voice", "singing", "gesang"))


def _measure_manifest_vqi(
    audio_orig: np.ndarray,
    audio_restored: np.ndarray,
    sr: int,
    *,
    raw_case: dict[str, Any],
) -> tuple[float | None, float | None, str]:
    """Misst VQI for manifest-declared vocal cases independent of UV3 metadata."""
    try:
        from backend.core.musical_goals.era_vocal_profile import (  # pylint: disable=import-outside-toplevel
            get_era_vocal_profile,
        )
        from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
            compute_vqi,
            get_vqi_material_floor,
        )

        case_text = " ".join(
            [
                str(raw_case.get("case_id", "") or ""),
                str(raw_case.get("description", "") or ""),
            ]
        ).lower()
        era_decade: int | None = None
        for _era_key in ("era_decade", "decade", "era_year", "year"):
            _raw = raw_case.get(_era_key)
            if isinstance(_raw, (int, float)):
                _val = int(_raw)
                if _val >= 1000:
                    era_decade = (_val // 10) * 10
                    break
                if 190 <= _val <= 210:
                    era_decade = _val * 10
                    break
        era_profile = get_era_vocal_profile(era_decade) if isinstance(era_decade, int) and era_decade > 0 else None

        result = compute_vqi(
            audio_orig=audio_orig,
            audio_restored=audio_restored,
            sr=sr,
            skip_singer_identity=("choir" in case_text or "chor" in case_text),
            era_profile=era_profile,
        )
        raw_vqi = result.get("vqi")
        material = str(raw_case.get("material_type", "unknown") or "unknown")
        floor = float(get_vqi_material_floor(material, is_studio_2026=False))
        return (float(raw_vqi) if isinstance(raw_vqi, (int, float)) else None), floor, "manifest_vqi"
    except Exception as exc:  # pragma: no cover - optional metric stack can be unavailable in slim envs
        logger.debug("§V6 _measure_manifest_vqi fehlgeschlagen — (None, None) zurückgegeben: %s", exc.__class__.__name__)
        return None, None, f"manifest_vqi_unavailable:{exc.__class__.__name__}"


def _scan_execution_case(
    raw_case: dict[str, Any],
    repo_root: Path,
    target_sr: int,
    output_dir: Path,
) -> ExecutionCaseResult:
    case_id = str(raw_case.get("case_id", "") or "").strip()
    if not case_id:
        raise ValueError("Execution case is missing 'case_id'")
    rel_path = str(raw_case.get("path", "") or "").strip()
    if not rel_path:
        raise ValueError(f"Execution case {case_id} is missing 'path'")
    path = repo_root / rel_path
    if not path.exists():
        raise FileNotFoundError(str(path))

    strategy_case: StrategyCaseResult = _scan_strategy_case(raw_case, repo_root, target_sr)
    planned_phases = tuple(strategy_case.combined_phases)
    max_seconds = float(raw_case.get("execution_max_seconds", min(float(raw_case.get("max_seconds", 20.0)), 8.0)))
    audio, sr, imported_duration = _load_audio_for_execution(path, target_sr=target_sr, max_seconds=max_seconds)

    start_time = time.time()
    cfg = RestorationConfig(
        mode=QualityMode.FAST,
        material_type=None,
        enable_performance_guard=True,
        enable_phase_gate=True,
        enable_phase_skipping=False,
        num_cores=1,
    )
    restorer = UnifiedRestorerV3(config=cfg)
    vocal_required = _is_vocal_case(raw_case)
    case_text = " ".join(
        [
            str(raw_case.get("case_id", "") or ""),
            str(raw_case.get("description", "") or ""),
        ]
    ).lower()
    result = restorer.restore(
        audio,
        sample_rate=sr,
        mode="fast",
        material=str(raw_case.get("material_type", "unknown") or "unknown"),
        precomputed_phase_plan=list(planned_phases),
        ml_runtime_budget_s=float(raw_case.get("ml_runtime_budget_s", 6.0)),
        vocal_material_prior=vocal_required,
        multi_singer_prior=("choir" in case_text or "chor" in case_text),
    )
    runtime_seconds = float(time.time() - start_time)

    meta = getattr(result, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
    fail_codes = _error_codes(meta.get("fail_reasons"))
    required_phases = tuple(strategy_case.required_phases)
    executed_phases = tuple(str(phase) for phase in getattr(result, "phases_executed", []) or [])
    skipped_phases = tuple(str(phase) for phase in getattr(result, "phases_skipped", []) or [])
    executed_or_skipped = set(executed_phases) | set(skipped_phases)
    missing_required = tuple(phase for phase in required_phases if phase not in executed_or_skipped)
    forbidden_executed = tuple(
        phase for phase in sorted(set(_RESTORATION_FORBIDDEN_PHASES)) if phase in set(executed_phases)
    )

    _pd = meta.get("phase_deltas")
    phase_deltas = _pd if isinstance(_pd, dict) else {}
    phase_delta_phases = tuple(sorted(str(phase) for phase in phase_deltas.keys()))
    missing_phase_deltas = tuple(
        phase for phase in required_phases if phase in set(executed_phases) and phase not in phase_deltas
    )

    _am = meta.get("artifact_freedom")
    artifact_meta = _am if isinstance(_am, dict) else {}
    artifact_score_raw = artifact_meta.get("score")
    artifact_score = float(artifact_score_raw) if isinstance(artifact_score_raw, (int, float)) else None
    artifact_passed = bool(artifact_meta.get("passed", artifact_score is not None and artifact_score >= 0.95))
    artifact_contract = _contract_passed_for_failed_gate(artifact_passed, fail_codes, "ARTIFACT_FREEDOM_VETO")

    hpi_meta = meta.get("holistic_perceptual_gate") if isinstance(meta.get("holistic_perceptual_gate"), dict) else None
    hpi_value = None
    hpi_passed = False
    if hpi_meta is not None:
        hpi_raw = hpi_meta.get("hpi")
        hpi_value = float(hpi_raw) if isinstance(hpi_raw, (int, float)) else None
        hpi_passed = bool(hpi_meta.get("passed", False))
    hpi_contract = hpi_meta is not None and _contract_passed_for_failed_gate(hpi_passed, fail_codes, "HPI_FAIL")

    vqi_raw = meta.get("vqi")
    vqi_value = float(vqi_raw) if isinstance(vqi_raw, (int, float)) else None
    vqi_floor = None
    vqi_source = "uv3_metadata" if vqi_value is not None else "none"
    if vocal_required:
        try:
            from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
                get_vqi_material_floor,
            )

            material = str(raw_case.get("material_type", "unknown") or "unknown")
            vqi_floor = float(get_vqi_material_floor(material, is_studio_2026=False))
        except Exception:  # pragma: no cover - optional metric stack can be unavailable in slim envs
            vqi_floor = None
    if vocal_required and vqi_value is None:
        restored_for_vqi = getattr(result, "audio", None)
        if restored_for_vqi is not None:
            vqi_value, vqi_floor, vqi_source = _measure_manifest_vqi(
                audio,
                np.asarray(restored_for_vqi, dtype=np.float32),
                sr,
                raw_case=raw_case,
            )
    vocal_contract = (not vocal_required) or (
        vqi_value is not None and (vqi_floor is None or vqi_value >= vqi_floor or "VQI_BELOW_THRESHOLD" in fail_codes)
    )

    export_ok, export_strategy, export_blocked, sidecar_payload = _check_export_contract(
        result,
        case_id,
        sr,
        output_dir,
    )
    degradation_status = str(meta.get("degradation_status", "") or "")

    return ExecutionCaseResult(
        case_id=case_id,
        required_phases=required_phases,
        planned_phases=planned_phases,
        phases_executed=executed_phases,
        phases_skipped=skipped_phases,
        missing_required_executions=missing_required,
        forbidden_executed=forbidden_executed,
        phase_delta_phases=phase_delta_phases,
        missing_phase_deltas=missing_phase_deltas,
        artifact_freedom=artifact_score,
        artifact_contract_passed=artifact_contract,
        hpi=hpi_value,
        hpi_contract_passed=hpi_contract,
        vqi=vqi_value,
        vocal_required=vocal_required,
        vocal_contract_passed=vocal_contract,
        export_contract_passed=export_ok,
        export_strategy=export_strategy,
        export_blocked=export_blocked,
        degradation_status=degradation_status,
        fail_reasons=fail_codes,
        runtime_seconds=runtime_seconds,
        duration_seconds=float(min(imported_duration, max_seconds) if max_seconds > 0 else imported_duration),
        metadata={
            "path": rel_path,
            "material": str(raw_case.get("material_type", "unknown") or "unknown"),
            "sample_rate": sr,
            "quality_estimate": float(getattr(result, "quality_estimate", 0.0) or 0.0),
            "goal_directed_candidate_recovery": meta.get("goal_directed_candidate_recovery"),
            "noise_texture_repair": meta.get("noise_texture_repair"),
            "frisson_goosebumps_recovery": meta.get("frisson_goosebumps_recovery"),
            "vqi_floor": vqi_floor,
            "vqi_source": vqi_source,
            "sidecar_quality_gate_passed": sidecar_payload.get("quality_gate_passed"),
            "sidecar_degradation_status": sidecar_payload.get("quality_gate_degradation_status"),
        },
    )


def _failed_execution_case(raw_case: dict[str, Any], exc: Exception) -> ExecutionCaseResult:
    case_id = str(raw_case.get("case_id", "") or "unknown")
    required = tuple(str(phase) for phase in raw_case.get("required_phases", []) if str(phase or "").strip())
    return ExecutionCaseResult(
        case_id=case_id,
        required_phases=required,
        planned_phases=(),
        phases_executed=(),
        phases_skipped=(),
        missing_required_executions=required,
        forbidden_executed=(),
        phase_delta_phases=(),
        missing_phase_deltas=(),
        artifact_freedom=None,
        artifact_contract_passed=False,
        hpi=None,
        hpi_contract_passed=False,
        vqi=None,
        vocal_required=_is_vocal_case(raw_case),
        vocal_contract_passed=not _is_vocal_case(raw_case),
        export_contract_passed=False,
        export_strategy="error",
        export_blocked=False,
        degradation_status="failed",
        fail_reasons=(f"EXECUTION_EXCEPTION:{type(exc).__name__}",),
        runtime_seconds=0.0,
        duration_seconds=1.0,
        metadata={
            "path": str(raw_case.get("path", "") or ""),
            "error": str(exc),
        },
    )


def _evaluate_execution_gate(
    cases: list[ExecutionCaseResult], thresholds: ExecutionGateThresholds
) -> ExecutionGateResult:
    total_cases = max(len(cases), 1)
    required_total = sum(len(case.required_phases) for case in cases)
    missing_total = sum(len(case.missing_required_executions) for case in cases)
    phase_recall = 1.0 if required_total == 0 else (required_total - missing_total) / required_total

    executed_total = sum(
        len([phase for phase in case.required_phases if phase in set(case.phases_executed)]) for case in cases
    )
    missing_delta_total = sum(len(case.missing_phase_deltas) for case in cases)
    phase_delta_coverage = 1.0 if executed_total == 0 else (executed_total - missing_delta_total) / executed_total

    artifact_rate = sum(1 for case in cases if case.artifact_contract_passed) / total_cases
    hpi_rate = sum(1 for case in cases if case.hpi_contract_passed) / total_cases
    vocal_cases = [case for case in cases if case.vocal_required]
    vocal_rate = (
        1.0 if not vocal_cases else sum(1 for case in vocal_cases if case.vocal_contract_passed) / len(vocal_cases)
    )
    export_rate = sum(1 for case in cases if case.export_contract_passed) / total_cases
    forbidden_total = sum(len(case.forbidden_executed) for case in cases)
    runtime = sum(case.runtime_seconds for case in cases)
    # Kurze Clips (<4s) tragen fixe Init-/Model-Load-Overheads unverhältnismäßig stark,
    # obwohl diese Kosten nicht proportional mit Programmlänge wachsen. Der Floor
    # stabilisiert die Gate-Bewertung für kurze Real-Audio-Snippets.
    duration_floor = max(float(thresholds.runtime_duration_floor_seconds), 1e-9)
    duration = sum(max(case.duration_seconds, duration_floor, 1e-9) for case in cases)
    runtime_factor = runtime / duration

    fail_reasons: list[str] = []
    if phase_recall < thresholds.min_phase_execution_recall:
        fail_reasons.append(f"phase_execution_recall {phase_recall:.3f} < {thresholds.min_phase_execution_recall:.3f}")
    if phase_delta_coverage < thresholds.min_phase_delta_coverage:
        fail_reasons.append(
            f"phase_delta_coverage {phase_delta_coverage:.3f} < {thresholds.min_phase_delta_coverage:.3f}"
        )
    if artifact_rate < thresholds.min_artifact_contract_rate:
        fail_reasons.append(f"artifact_contract_rate {artifact_rate:.3f} < {thresholds.min_artifact_contract_rate:.3f}")
    if hpi_rate < thresholds.min_hpi_contract_rate:
        fail_reasons.append(f"hpi_contract_rate {hpi_rate:.3f} < {thresholds.min_hpi_contract_rate:.3f}")
    if vocal_cases and vocal_rate < thresholds.min_vocal_contract_rate:
        fail_reasons.append(f"vocal_contract_rate {vocal_rate:.3f} < {thresholds.min_vocal_contract_rate:.3f}")
    if export_rate < thresholds.min_export_contract_rate:
        fail_reasons.append(f"export_contract_rate {export_rate:.3f} < {thresholds.min_export_contract_rate:.3f}")
    if forbidden_total > thresholds.max_forbidden_phase_executions:
        fail_reasons.append(
            f"forbidden_phase_executions {forbidden_total} > {thresholds.max_forbidden_phase_executions}"
        )
    if runtime_factor > thresholds.max_runtime_factor:
        fail_reasons.append(f"runtime_factor {runtime_factor:.3f} > {thresholds.max_runtime_factor:.3f}")

    return ExecutionGateResult(
        passed=not fail_reasons,
        phase_execution_recall=float(phase_recall),
        phase_delta_coverage=float(phase_delta_coverage),
        artifact_contract_rate=float(artifact_rate),
        hpi_contract_rate=float(hpi_rate),
        vocal_contract_rate=float(vocal_rate),
        export_contract_rate=float(export_rate),
        forbidden_phase_executions=int(forbidden_total),
        runtime_factor=float(runtime_factor),
        fail_reasons=tuple(fail_reasons),
    )


def run_real_audio_execution_golden_gate(
    *,
    manifest_path: Path,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    allow_missing: bool = False,
    allow_empty: bool = False,
    max_cases: int | None = None,
) -> RealAudioExecutionGoldenGateReport:
    """Führt aus: the manifest-driven real-audio execution/export Golden-Set gate."""
    start_time = time.time()
    manifest_path = manifest_path.resolve()
    repo_root = (repo_root or manifest_path.parents[1]).resolve()
    payload = _load_manifest(manifest_path)
    thresholds = _thresholds_from_manifest(payload)
    target_sr = int(payload.get("target_sample_rate", 48_000))

    if output_dir is None:
        temp_root = tempfile.TemporaryDirectory(prefix="aurik_execution_gate_")
        output_path = Path(temp_root.name)
    else:
        temp_root = None
        output_path = output_dir.resolve()
        output_path.mkdir(parents=True, exist_ok=True)

    scanned_cases: list[ExecutionCaseResult] = []
    skipped_cases: list[dict[str, str]] = []
    try:
        for raw_case in payload["cases"]:
            if not isinstance(raw_case, dict):
                raise ValueError("Every real-audio execution manifest case must be an object")
            case_id = str(raw_case.get("case_id", "") or "unknown")
            if raw_case.get("active", True) is False:
                skipped_cases.append({"case_id": case_id, "reason": "inactive"})
                continue
            if max_cases is not None and len(scanned_cases) >= max_cases:
                skipped_cases.append({"case_id": case_id, "reason": "max_cases"})
                continue
            try:
                scanned_cases.append(_scan_execution_case(raw_case, repo_root, target_sr, output_path))
            except FileNotFoundError as exc:
                if not allow_missing:
                    raise
                skipped_cases.append({"case_id": case_id, "reason": f"missing_file:{exc}"})
            except Exception as exc:
                scanned_cases.append(_failed_execution_case(raw_case, exc))
    finally:
        if temp_root is not None:
            temp_root.cleanup()

    if not scanned_cases and not allow_empty:
        raise RuntimeError("Real-audio Execution Golden-Set gate has no scanned active cases")

    gate = _evaluate_execution_gate(scanned_cases, thresholds)
    elapsed = float(time.time() - start_time)
    return RealAudioExecutionGoldenGateReport(
        gate=gate,
        cases=scanned_cases,
        skipped_cases=skipped_cases,
        manifest_path=str(manifest_path),
        scanned_cases=len(scanned_cases),
        elapsed_seconds=elapsed,
    )


__all__ = [
    "ExecutionCaseResult",
    "ExecutionGateResult",
    "ExecutionGateThresholds",
    "RealAudioExecutionGoldenGateReport",
    "run_real_audio_execution_golden_gate",
]
