"""Tests für Aurik10/ui/hearing_gates_summary.py (T1, headless)."""

from __future__ import annotations

from Aurik10.ui.hearing_gates_summary import (
    apply_resolved_defects,
    hearing_gate_status,
    hearing_gates_details,
    hearing_gates_line,
)


def _meta(**kw) -> dict:
    base: dict = {}
    base.update(kw)
    return base


class TestStatus:
    def test_green_empty(self) -> None:
        assert hearing_gate_status(_meta()) == "green"

    def test_red_audibility(self) -> None:
        m = _meta(audibility_gate={"gate_passed": False, "n_audible_unmasked": 3,
                                   "improvable_types": ["hum"]})
        assert hearing_gate_status(m) == "red"

    def test_yellow_corrected(self) -> None:
        m = _meta(einladungs_gate_passed=False, einladungs_gate_corrected=True)
        assert hearing_gate_status(m) == "yellow"

    def test_yellow_queue(self) -> None:
        m = _meta(audibility_gate={"gate_passed": True, "improvable_types": ["clicks"]})
        assert hearing_gate_status(m) == "yellow"

    def test_red_vocal_revert(self) -> None:
        m = _meta(vocal_drive_hard_revert=True)
        assert hearing_gate_status(m) == "red"


class TestLines:
    def test_line_has_icon(self) -> None:
        assert "Hör-Gates" in hearing_gates_line(_meta())

    def test_details_audibility(self) -> None:
        m = _meta(audibility_gate={"gate_passed": False, "n_audible_unmasked": 2,
                                   "improvable_types": ["hum", "clicks"]})
        det = hearing_gates_details(m)
        assert any("Restdefekte" in d for d in det)
        assert any("Stufe-2-Queue" in d for d in det)

    def test_details_wohlklang(self) -> None:
        det = hearing_gates_details(_meta(einladungs_gate_passed=True))
        assert any("Wohlklang: erfüllt" in d for d in det)

    def test_details_empty(self) -> None:
        det = hearing_gates_details(_meta())
        assert any("n/a" in d for d in det)

    def test_blend_telemetry(self) -> None:
        det = hearing_gates_details(_meta(vocal_drive_blend=0.85))
        assert any("Vocal-Drive" in d for d in det)


class TestApplyResolvedDefects:
    def test_subtract_and_total(self) -> None:
        counts = {"clicks": 5, "hum": 2}
        new, total, done = apply_resolved_defects(counts, ["clicks", "hum", "hum"])
        assert new == {"clicks": 4, "hum": 0}
        assert total == 4
        assert done == ["hum"]

    def test_unknown_ignored_and_no_negative(self) -> None:
        new, total, done = apply_resolved_defects({"clicks": 0}, ["clicks", "fremd"])
        assert new == {"clicks": 0}
        assert total == 0
        assert done == []

    def test_none_safe(self) -> None:
        new, total, done = apply_resolved_defects({}, None)
        assert new == {} and total == 0 and done == []

    def test_original_untouched(self) -> None:
        counts = {"hum": 3}
        apply_resolved_defects(counts, ["hum"])
        assert counts == {"hum": 3}
