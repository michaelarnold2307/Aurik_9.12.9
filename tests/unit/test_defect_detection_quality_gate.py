from __future__ import annotations

import pytest

from backend.core.defect_detection_quality_gate import (
    DefectBenchmarkCaseResult,
    DefectDetectionGateThresholds,
    DefectExpectation,
    evaluate_defect_detection_gate,
)


@pytest.mark.unit
def test_worldclass_gate_passes_high_recall_precision_locality_and_runtime() -> None:
    cases = [
        DefectBenchmarkCaseResult(
            case_id="vinyl_clicks",
            expected=(DefectExpectation("clicks", min_severity=0.20, min_confidence=0.70, require_locations=True),),
            forbidden_defects=("dropouts",),
            severities={"clicks": 0.44, "dropouts": 0.02},
            confidences={"clicks": 0.86},
            locations={"clicks": [(0.5, 0.501), (1.2, 1.201)]},
            runtime_seconds=0.9,
            duration_seconds=1.0,
        ),
        DefectBenchmarkCaseResult(
            case_id="tape_hum",
            expected=(DefectExpectation("hum", min_severity=0.15, min_confidence=0.65),),
            forbidden_defects=("riaa_curve_error",),
            severities={"hum": 0.37, "riaa_curve_error": 0.0},
            confidences={"hum": 0.78},
            runtime_seconds=1.0,
            duration_seconds=1.0,
        ),
    ]

    result = evaluate_defect_detection_gate(cases)

    assert result.passed is True
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.locality_recall == 1.0
    assert result.fail_reasons == ()


def test_worldclass_gate_fails_missed_expected_defect() -> None:
    case = DefectBenchmarkCaseResult(
        case_id="missed_clicks",
        expected=(DefectExpectation("clicks", min_severity=0.20),),
        severities={"clicks": 0.03},
        confidences={"clicks": 0.80},
    )

    result = evaluate_defect_detection_gate([case])

    assert result.passed is False
    assert result.recall == 0.0
    assert any(reason.startswith("recall:") for reason in result.fail_reasons)
    assert "missed:clicks" in result.case_failures["missed_clicks"][0]


def test_worldclass_gate_fails_low_confidence_even_when_severity_detects() -> None:
    case = DefectBenchmarkCaseResult(
        case_id="low_confidence_vocal_harshness",
        expected=(DefectExpectation("vocal_harshness", min_severity=0.10, min_confidence=0.72),),
        severities={"vocal_harshness": 0.32},
        confidences={"vocal_harshness": 0.31},
    )

    result = evaluate_defect_detection_gate([case])

    assert result.passed is False
    assert result.recall == 1.0
    assert any(
        "low_confidence:vocal_harshness" in reason for reason in result.case_failures["low_confidence_vocal_harshness"]
    )


def test_worldclass_gate_fails_forbidden_false_positive_and_precision() -> None:
    case = DefectBenchmarkCaseResult(
        case_id="clean_harmonic",
        expected=(DefectExpectation("none", min_severity=1.01, critical=False),),
        forbidden_defects=("clicks", "crackle"),
        severities={"clicks": 0.31, "crackle": 0.04},
        confidences={"clicks": 0.77},
    )
    thresholds = DefectDetectionGateThresholds(min_recall=0.0, min_precision=0.92)

    result = evaluate_defect_detection_gate([case], thresholds)

    assert result.passed is False
    assert result.false_positive_total >= 1
    assert any("false_positive:clicks" in reason for reason in result.case_failures["clean_harmonic"])


def test_worldclass_gate_fails_missing_required_locations() -> None:
    case = DefectBenchmarkCaseResult(
        case_id="dropout_without_locations",
        expected=(DefectExpectation("dropouts", min_severity=0.10, min_confidence=0.50, require_locations=True),),
        severities={"dropouts": 0.60},
        confidences={"dropouts": 0.80},
        locations={},
    )

    result = evaluate_defect_detection_gate([case])

    assert result.passed is False
    assert result.locality_recall == 0.0
    assert any("missing_locations:dropouts" in reason for reason in result.case_failures["dropout_without_locations"])


def test_worldclass_gate_fails_runtime_factor() -> None:
    case = DefectBenchmarkCaseResult(
        case_id="slow_real_audio_scan",
        expected=(DefectExpectation("hum", min_severity=0.10),),
        severities={"hum": 0.30},
        confidences={"hum": 0.80},
        runtime_seconds=1.8,
        duration_seconds=1.0,
    )

    result = evaluate_defect_detection_gate([case], DefectDetectionGateThresholds(max_runtime_factor=1.2))

    assert result.passed is False
    assert result.max_runtime_factor == 1.8
    assert any(reason.startswith("runtime_factor:") for reason in result.fail_reasons)


def test_defect_scanner_worldclass_audit_fixtures_pass_gate() -> None:
    from audit.defect_detection_worldclass_gate import run_gate

    result, cases = run_gate(seconds=2.0)  # SOTA: leicht erhöht für deterministische Reproduzierbarkeit (§G5)

    assert len(cases) == 8
    assert result.passed is True  # type: ignore[attr-defined]
    assert result.recall == 1.0  # type: ignore[attr-defined]
    assert result.precision == 1.0  # type: ignore[attr-defined]
    assert result.locality_recall == 1.0  # type: ignore[attr-defined]
    assert result.fail_reasons == ()  # type: ignore[attr-defined]
