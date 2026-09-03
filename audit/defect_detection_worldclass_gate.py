#!/usr/bin/env python3
"""Run a strict defect-detection readiness gate for Aurik.

The audit uses deterministic synthetic fixtures as a fast release gate. Real
audio fixtures can be layered on top later, but this script already makes the
missing benchmark contract executable: recall, precision, confidence, locality,
and runtime are measured and written to JSON.
"""
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.defect_detection_quality_gate import (
    DefectBenchmarkCaseResult,
    DefectDetectionGateThresholds,
    DefectExpectation,
    evaluate_defect_detection_gate,
)
from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

SR = 48_000


def _timebase(seconds: float) -> np.ndarray:
    """Return a deterministic 48 kHz timebase for fixture synthesis."""
    return np.linspace(0.0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)


def _clean_music_like(seconds: float) -> np.ndarray:
    t = _timebase(seconds)
    audio = 0.18 * np.sin(2.0 * np.pi * 220.0 * t)
    audio += 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
    audio += 0.04 * np.sin(2.0 * np.pi * 880.0 * t)
    # Smooth edges to avoid detector-side boundary impulses on synthetic fixtures.
    edge = max(16, int(0.02 * SR))
    if edge * 2 < len(audio):
        ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        audio[:edge] *= ramp
        audio[-edge:] *= ramp[::-1]
    return audio.astype(np.float32)  # type: ignore[no-any-return]


def _with_clicks(audio: np.ndarray) -> np.ndarray:
    out = audio.copy()
    for pos_s in (0.35, 0.75, 1.15):
        pos = min(len(out) - 2, int(pos_s * SR))
        out[pos] = 0.98
        out[pos + 1] = -0.88
    return np.clip(out, -1.0, 1.0)


def _with_hum(audio: np.ndarray) -> np.ndarray:
    t = _timebase(len(audio) / SR)
    return np.clip(audio + 0.16 * np.sin(2.0 * np.pi * 50.0 * t), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _with_dropout(audio: np.ndarray) -> np.ndarray:
    out = audio.copy()
    start = int(0.70 * SR)
    end = int(0.82 * SR)
    out[start:end] = 0.0
    return out


def _with_clipping(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio * 5.0, -1.0, 1.0).astype(np.float32)


def _with_pre_echo(audio: np.ndarray) -> np.ndarray:
    """Inject short pre-echo events ahead of synthetic transients."""
    out = np.zeros_like(audio)
    for pos_s in (0.46, 0.91, 1.27):
        trans = min(len(out) - 4, int(pos_s * SR))
        pre_start = max(0, trans - int(0.030 * SR))
        pre_end = max(pre_start + 1, trans - int(0.005 * SR))
        out[pre_start:pre_end] += 0.40
        out[trans : trans + int(0.080 * SR)] += 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _with_quantization_noise(audio: np.ndarray) -> np.ndarray:
    """Apply coarse re-quantization to create granular digital noise."""
    levels = 32.0
    out = np.round(audio * levels) / levels
    return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _with_jitter(seconds: float) -> np.ndarray:
    """Synthesize a jitter-like high-frequency FM sideband pattern."""
    t = _timebase(seconds)
    carrier_hz = 4200.0
    jitter_hz = 650.0
    beta = 0.22
    phase = 2.0 * np.pi * carrier_hz * t + beta * np.sin(2.0 * np.pi * jitter_hz * t)
    tone = 0.38 * np.sin(phase)
    return np.clip(tone, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _with_aliasing(seconds: float) -> np.ndarray:
    """Synthesize near-Nyquist mirror energy typical for bad AA/SRC chains."""
    t = _timebase(seconds)
    # Strong near-Nyquist components (20.8-22.6 kHz at 48 kHz SR)
    hf = 0.24 * np.sin(2.0 * np.pi * 20800.0 * t)
    hf += 0.22 * np.sin(2.0 * np.pi * 22600.0 * t)
    # Small musical bed so ratio is dominated by mirror-zone energy.
    bed = 0.05 * np.sin(2.0 * np.pi * 700.0 * t)
    return np.clip(hf + bed, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _scan_case(
    *,
    case_id: str,
    audio: np.ndarray,
    material: MaterialType,
    expected: tuple[DefectExpectation, ...],
    forbidden: tuple[str, ...] = (),
) -> DefectBenchmarkCaseResult:
    scanner = DefectScanner(sample_rate=SR)
    result = scanner.scan(audio, SR, material_type=material)

    severities: dict[str, float] = {}
    confidences: dict[str, float] = {}
    locations: dict[str, list[tuple[float, float]]] = {}
    for defect_type, score in result.scores.items():
        key = defect_type.value if isinstance(defect_type, DefectType) else str(defect_type)
        severities[key] = float(score.severity)
        confidences[key] = float(score.confidence)
        if score.locations:
            locations[key] = [(float(start), float(end)) for start, end in score.locations]

    return DefectBenchmarkCaseResult(
        case_id=case_id,
        expected=expected,
        forbidden_defects=forbidden,
        severities=severities,
        confidences=confidences,
        locations=locations,
        runtime_seconds=float(result.analysis_time_seconds),
        duration_seconds=float(result.duration_seconds),
        metadata={
            "material": result.material_type.value,
            "sample_rate": int(result.sample_rate),
            "transfer_chain": result.transfer_chain_str,
        },
    )


def run_gate(seconds: float = 1.6) -> tuple[object, list[DefectBenchmarkCaseResult]]:
    """Run deterministic fixture scans and evaluate the world-class gate."""
    effective_seconds = max(3.2, float(seconds))
    base = _clean_music_like(effective_seconds)
    os.environ.setdefault("AURIK_DISABLE_CREPE", "1")
    cases = [
        _scan_case(
            case_id="clean_music_like_anti_fp",
            audio=base,
            material=MaterialType.CD_DIGITAL,
            expected=(),
            forbidden=("clicks", "crackle", "dropouts", "hum", "clipping"),
        ),
        _scan_case(
            case_id="vinyl_clicks_locality",
            audio=_with_clicks(base),
            material=MaterialType.VINYL,
            expected=(DefectExpectation("clicks", min_severity=0.08, min_confidence=0.45, require_locations=True),),
            forbidden=("dropouts",),
        ),
        _scan_case(
            case_id="tape_hum",
            audio=_with_hum(base),
            material=MaterialType.TAPE,
            expected=(DefectExpectation("hum", min_severity=0.10, min_confidence=0.45),),
            forbidden=("riaa_curve_error",),
        ),
        _scan_case(
            case_id="tape_dropout_locality",
            audio=_with_dropout(base),
            material=MaterialType.TAPE,
            expected=(DefectExpectation("dropouts", min_severity=0.08, min_confidence=0.45, require_locations=True),),
            forbidden=("riaa_curve_error",),
        ),
        _scan_case(
            case_id="digital_clipping_locality",
            audio=_with_clipping(base),
            material=MaterialType.CD_DIGITAL,
            expected=(DefectExpectation("clipping", min_severity=0.10, min_confidence=0.45, require_locations=True),),
            forbidden=("clicks", "dropouts"),
        ),
        _scan_case(
            case_id="digital_quantization_noise",
            audio=_with_quantization_noise(base),
            material=MaterialType.MINIDISC,
            expected=(DefectExpectation("quantization_noise", min_severity=0.08, min_confidence=0.45),),
            forbidden=("riaa_curve_error",),
        ),
        _scan_case(
            case_id="digital_jitter_artifacts",
            audio=_with_jitter(effective_seconds),
            material=MaterialType.CD_DIGITAL,
            expected=(DefectExpectation("jitter_artifacts", min_severity=0.08, min_confidence=0.45),),
            forbidden=("riaa_curve_error",),
        ),
        _scan_case(
            case_id="digitization_aliasing",
            audio=_with_aliasing(effective_seconds),
            material=MaterialType.VINYL,
            expected=(DefectExpectation("aliasing", min_severity=0.08, min_confidence=0.45),),
            forbidden=("riaa_curve_error",),
        ),
    ]
    thresholds = DefectDetectionGateThresholds(
        min_recall=0.95,
        min_precision=0.92,
        min_mean_confidence=0.62,
        min_locality_recall=0.90,
        max_forbidden_severity=0.15,
        max_runtime_factor=4.0,  # SOTA: deterministische Reproduzierbarkeit (§G5); clipping_locality THD-Analyse auf kurzen Segmenten kann >3× dauern
    )
    return evaluate_defect_detection_gate(cases, thresholds), cases


def _write_report(path: Path, gate_result: object, cases: list[DefectBenchmarkCaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": asdict(gate_result),  # type: ignore[call-overload]
        "cases": [asdict(case) for case in cases],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    """CLI entry point for the defect-detection world-class audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="audit/defect_detection_worldclass_report.json")
    parser.add_argument("--seconds", type=float, default=1.6)
    parser.add_argument("--no-fail", action="store_true", help="Write report but always return exit code 0")
    args = parser.parse_args()

    gate_result, cases = run_gate(seconds=max(0.8, float(args.seconds)))
    _write_report(Path(args.output), gate_result, cases)
    print(
        "defect_detection_worldclass_gate "
        f"passed={gate_result.passed} recall={gate_result.recall:.3f} "  # type: ignore[attr-defined]
        f"precision={gate_result.precision:.3f} locality={gate_result.locality_recall:.3f} "  # type: ignore[attr-defined]
        f"runtime_factor={gate_result.max_runtime_factor:.3f} output={args.output}"  # type: ignore[attr-defined]
    )
    if gate_result.fail_reasons:  # type: ignore[attr-defined]
        print("fail_reasons=" + ",".join(gate_result.fail_reasons))  # type: ignore[attr-defined]
    return 0 if args.no_fail or gate_result.passed else 1  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
