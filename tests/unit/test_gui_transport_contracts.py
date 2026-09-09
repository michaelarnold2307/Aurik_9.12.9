from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")  # CI-Minimal-Umgebung (cross-platform)

from Aurik10.ipc.pipeline_process import PipelineProcess, PipelineStatus
from Aurik10.ui.results_summary import build_results_data, interpret_hpi_score, interpret_mushra_score


class _FakePipe:
    def __init__(self, messages: list[str]):
        self._messages = list(messages)

    def poll(self) -> bool:
        return bool(self._messages)

    def recv(self) -> str:
        return self._messages.pop(0)


def test_pipeline_process_poll_preserves_result_fields():
    status = PipelineStatus(
        state="warning",
        progress_pct=100.0,
        result_quality=69.4,
        result_reverted=True,
        result_revert_reason="ARTIFACT_VETO",
        result_mushra=38.9,
        result_hpi=0.59,
        result_phases_done=39,
        result_warnings=["degraded"],
    )
    process = PipelineProcess()
    process._parent_pipe = _FakePipe([status.to_json()])  # type: ignore[assignment]

    assert process.poll() is True

    latest = process.latest_status
    assert latest.result_quality == 69.4
    assert latest.result_reverted is True
    assert latest.result_revert_reason == "ARTIFACT_VETO"
    assert latest.result_mushra == 38.9
    assert latest.result_hpi == 0.59
    assert latest.result_phases_done == 39
    assert latest.result_warnings == ["degraded"]


def test_results_summary_data_preserves_degraded_and_no_effect_metadata():
    restoration_result = SimpleNamespace(
        quality_estimate=0.69,
        phases_executed=["phase_01", "phase_02"],
        phases_skipped=["phase_21"],
        deferred_phases=["phase_55"],
        metadata={
            "restorability_score": 66.4,
            "mushra": {"mushra_score": 38.9},
            "hpi_score": 0.59,
            "degradation_status": "degraded",
            "fail_reason": "RESTORATION_OQS_GATE_DEGRADED",
            "no_effect_phase_count": 12,
            "defect_countdown": {"remaining_audible": 34},
        },
    )

    data = build_results_data(restoration_result=restoration_result)

    assert data["degradation_status"] == "degraded"
    assert data["fail_reason"] == "RESTORATION_OQS_GATE_DEGRADED"
    assert data["no_effect_phase_count"] == 12
    assert data["residual_audible_defects"] == 34
    assert data["mushra_score"] == 38.9
    assert data["hpi_score"] == 0.59
    assert data["phases_skipped"] == 1
    assert data["deferred_phase_count"] == 1


def test_results_summary_interprets_mushra_for_laypeople():
    assert "weltklasse" in interpret_mushra_score(94.0).lower()
    assert "sehr gute" in interpret_mushra_score(84.0).lower()
    assert "hörbar" in interpret_mushra_score(70.0).lower()
    assert "sicheren checkpoint" in interpret_mushra_score(38.9).lower()


def test_results_summary_interprets_hpi_for_laypeople():
    assert "hoher sicherheit" in interpret_hpi_score(0.9).lower()
    assert "vertrauenswürdig" in interpret_hpi_score(0.75).lower()
    assert "vorsichtig" in interpret_hpi_score(0.59).lower()
    assert "schutzmodus" in interpret_hpi_score(0.31).lower()
