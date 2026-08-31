"""Tests für die Hörordnungs-Verdrahtung (Ebenen 2/3, Phasen-Entscheidungen)."""

from __future__ import annotations

import numpy as np

from backend.core.goal_priority_protocol import GoalPriorityProtocol
from backend.core.unified_restorer_v3 import UnifiedRestorerV3

# ── Hörordnung Ebene 3: Lexikografische Dominanz ────────────────────────────


def test_hearing_tier_mapping() -> None:
    gpp = GoalPriorityProtocol()
    assert gpp.hearing_tier("natuerlichkeit") == 1
    assert gpp.hearing_tier("authentizitaet") == 1
    assert gpp.hearing_tier("waerme") == 2
    assert gpp.hearing_tier("transparenz") == 3
    assert gpp.hearing_tier("brillanz") == 4
    assert gpp.hearing_tier("unbekanntes_goal") == 3  # neutraler Default


def test_would_violate_hearing_order() -> None:
    gpp = GoalPriorityProtocol()
    # Brillanz-Gewinn auf Kosten von Wärme → Verstoß
    assert gpp.would_violate_hearing_order("brillanz", "waerme") is True
    # Wärme-Gewinn auf Kosten von Natürlichkeit → Verstoß
    assert gpp.would_violate_hearing_order("waerme", "natuerlichkeit") is True
    # Wärme-Gewinn auf Kosten von Brillanz → KEIN Verstoß (Stufe 2 > Stufe 4)
    assert gpp.would_violate_hearing_order("waerme", "brillanz") is False
    # Gleichrangiger Tausch (transparenz vs. separation_fidelity) → kein Verstoß
    assert gpp.would_violate_hearing_order("transparenz", "separation_fidelity") is False


# ── Hörordnung Ebene 2, Stufe B: Maskierungs-Skip ───────────────────────────


def _make_uv3_stub(defect_scores: dict, material_key: str = "") -> UnifiedRestorerV3:
    obj = UnifiedRestorerV3.__new__(UnifiedRestorerV3)  # ohne __init__ (leichtgewichtig)
    obj._defect_result_scores = defect_scores
    obj._restoration_context = {"material_key": material_key}
    return obj


def _score(sev: float, salience: float | None) -> object:
    md: dict[str, float] = {}
    if salience is not None:
        md["perceptual_salience"] = salience
    return type("S", (), {"severity": sev, "metadata": md})()


def _mapped_phase_id() -> str:
    from backend.core.defect_phase_mapper import get_reverse_phase_map

    rmap = get_reverse_phase_map()
    for pid, defects in rmap.items():
        if defects:
            return pid
    raise AssertionError("keine gemappte Phase gefunden")


def test_skip_masked_phase_when_all_masked() -> None:
    from backend.core.defect_phase_mapper import get_reverse_phase_map

    pid = _mapped_phase_id()
    defects = get_reverse_phase_map()[pid]
    scores = {dt: _score(0.5, 0.2) for dt in defects}  # alle vollständig maskiert
    obj = _make_uv3_stub(scores)
    assert obj._should_skip_masked_phase(pid) is True


def test_no_skip_when_any_defect_salient() -> None:
    from backend.core.defect_phase_mapper import get_reverse_phase_map

    pid = _mapped_phase_id()
    defects = list(get_reverse_phase_map()[pid])
    scores = {dt: _score(0.5, 0.2) for dt in defects}
    scores[defects[0]] = _score(0.5, 0.8)  # einer hörbar → kein Skip
    obj = _make_uv3_stub(scores)
    assert obj._should_skip_masked_phase(pid) is False


def test_no_skip_when_defects_absent_or_salience_missing() -> None:
    from backend.core.defect_phase_mapper import get_reverse_phase_map

    pid = _mapped_phase_id()
    defects = get_reverse_phase_map()[pid]
    # sev < 0.03 → zählt nicht als messbar → kein Skip
    obj_absent = _make_uv3_stub({dt: _score(0.01, 0.2) for dt in defects})
    assert obj_absent._should_skip_masked_phase(pid) is False
    # Salience fehlt (Pass-Through-Neutralisierung) → default 1.0 → kein Skip
    obj_nosal = _make_uv3_stub({dt: _score(0.5, None) for dt in defects})
    assert obj_nosal._should_skip_masked_phase(pid) is False


def test_no_skip_for_enhancement_or_analog_phase06() -> None:
    obj = _make_uv3_stub({})
    # Enhancement-Phase ohne Mapping → kein Skip
    assert obj._should_skip_masked_phase("phase_37_bass_enhancement") is False
    # Phase 06 für analoges Material → nie skip
    obj_analog = _make_uv3_stub({}, material_key="vinyl")
    assert obj_analog._should_skip_masked_phase("phase_06_frequency_restoration") is False
