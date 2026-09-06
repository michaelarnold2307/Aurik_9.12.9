"""Tests für backend/core/defect_audibility_gate.py (Hörbarkeits-Gate, m1).

Hörordnung Ebene 2 (§4): „Reparatur gilt als abgeschlossen, wenn ein Defekt
unter der Maskierungsschwelle liegt — nicht wenn sein Messwert Null ist."
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_audibility_gate import (
    AUDIBLE_BASE,
    MATERIAL_JND_OFFSET,
    DefectAudibilityReport,
    audible_threshold,
    evaluate_defect_audibility,
)


def _entry(pre: float, post: float, masked: int = 0) -> dict:
    return {
        "pre": pre,
        "post": post,
        "reduction": round(max(0.0, pre - post), 4),
        "reduction_pct": round(max(0.0, pre - post) / max(pre, 0.001) * 100, 1),
        "masked_events": masked,
    }


class TestAudibleThreshold:
    def test_vinyl_base(self) -> None:
        assert audible_threshold("vinyl", 1) == pytest.approx(0.06)  # 0.08 - 0.02

    def test_cassette_higher_floor(self) -> None:
        assert audible_threshold("cassette", 1) == pytest.approx(0.12)

    def test_mp3_low_offset(self) -> None:
        assert audible_threshold("mp3_low", 1) == pytest.approx(0.13)

    def test_chain_depth_adds_masking(self) -> None:
        assert audible_threshold("vinyl", 3) == pytest.approx(0.08)

    def test_clipping_bounds(self) -> None:
        assert audible_threshold("cassette", 5) <= 0.15
        assert audible_threshold("vinyl", 1) >= 0.03

    def test_unknown_material_neutral(self) -> None:
        assert audible_threshold("banana", 1) == pytest.approx(AUDIBLE_BASE)

    def test_offset_table_complete(self) -> None:
        # In unified_restorer_v3.py wird die Tabelle aus diesem Modul importiert —
        # sie muss die historischen Materialien weiterhin abdecken.
        for mat in ("cassette", "vinyl", "shellac", "reel_tape", "cd_digital", "mp3_low", "aac"):
            assert mat in MATERIAL_JND_OFFSET


class TestEvaluateAudibility:
    def test_all_resolved_gate_passes(self) -> None:
        data = {"clicks": _entry(0.5, 0.02), "hiss": _entry(0.4, 0.03)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is True
        assert rep.n_resolved == 2
        assert rep.n_audible_unmasked == 0

    def test_audible_remaining_fails_gate(self) -> None:
        data = {"clicks": _entry(0.5, 0.20)}  # post 0.20 > thr 0.06, keine Maskierung
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is False
        assert rep.n_audible_unmasked == 1
        assert rep.improvable_types == ["clicks"]

    def test_masked_events_cover_residual(self) -> None:
        # post über Schwelle, aber ERB-Masking hat Events abgedeckt und post ist
        # moderat → gilt als abgedeckt, Gate besteht.
        data = {"clicks": _entry(0.5, 0.15, masked=12)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is True
        assert rep.n_masked == 1

    def test_masked_but_very_loud_still_fails(self) -> None:
        # Maskierte Events hin oder her: post >= 0.35 bleibt sicher exponiert.
        data = {"clicks": _entry(0.8, 0.60, masked=3)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is False
        assert rep.n_audible_unmasked == 1

    def test_physical_cap_types_accepted(self) -> None:
        data = {"bandwidth_loss": _entry(0.7, 0.35)}
        rep = evaluate_defect_audibility(
            data, material_key="mp3_low", physical_cap_types={"codec_artifacts"}
        )
        assert rep.gate_passed is True
        assert rep.n_physical_cap == 1

    def test_never_audible_ignored(self) -> None:
        data = {"wow": _entry(0.02, 0.01)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        assert rep.gate_passed is True
        assert rep.n_never_audible == 1

    def test_empty_and_malformed(self) -> None:
        rep = evaluate_defect_audibility(None, material_key="vinyl")
        assert rep.gate_passed is True
        rep2 = evaluate_defect_audibility({"x": "kaputt", "y": {"pre": np.nan, "post": None}}, material_key="vinyl")
        assert rep2.gate_passed is True

    def test_report_metadata_jsonable(self) -> None:
        data = {"clicks": _entry(0.5, 0.20)}
        rep = evaluate_defect_audibility(data, material_key="vinyl")
        meta = rep.to_metadata()
        assert meta["gate_passed"] is False
        assert meta["improvable_types"] == ["clicks"]
        assert isinstance(meta["per_type"]["clicks"]["pre"], float)
