"""P0-2 Tier-Map-Sync-Regressionstest (Tiefenanalyse 2026-09-08).

Schützt vor stiller Drift zwischen PRIORITY_MAP (§2.29-FC-Abort-Achse) und
HEARING_TIER_MAP (Hörordnungs-Dominanz-Achse, hoerordnung.instructions.md §5):
beide sind bewusst verschieden, aber jede Prioritäts-Goal muss eine explizite
Tier-Zuordnung haben (kein stilles Default-3) und die normativen Anker müssen
halten. Läuft automatisch in aurik-unit-smoke + Coverage-Gate.
"""

from __future__ import annotations

from backend.core.goal_priority_protocol import GoalPriorityProtocol


def test_map_consistency_no_problems() -> None:
    problems = GoalPriorityProtocol.verify_map_consistency()
    assert not problems, "Tier-/Priority-Map-Drift erkannt:\n" + "\n".join(problems)


def test_normative_tier_anchors() -> None:
    gpp = GoalPriorityProtocol()
    assert gpp.hearing_tier("natuerlichkeit") == 1
    assert gpp.hearing_tier("waerme") == 2
    assert gpp.hearing_tier("transparenz") == 3
    assert gpp.hearing_tier("brillanz") == 4
    assert gpp.hearing_tier("spatial_depth") == 4


def test_priority_anchors() -> None:
    gpp = GoalPriorityProtocol()
    assert gpp.priority_of("natuerlichkeit") == 1
    assert gpp.priority_of("brillanz") == 5


def test_aliases_resolve_to_canonical() -> None:
    gpp = GoalPriorityProtocol()
    assert gpp.canonical_goal("timbre") == "timbre_authentizitaet"
    assert gpp.canonical_goal("mikrodynamik") == "micro_dynamics"
    assert gpp.canonical_goal("sep_fidelity") == "separation_fidelity"
    assert gpp.canonical_goal("raumtiefe") == "spatial_depth"
    # Alias-Lookups landen auf derselben Stufe wie das kanonische Goal
    assert gpp.hearing_tier("raumtiefe") == gpp.hearing_tier("spatial_depth")
    assert gpp.priority_of("timbre") == gpp.priority_of("timbre_authentizitaet")


def test_unknown_goal_defaults_are_stable() -> None:
    gpp = GoalPriorityProtocol()
    assert gpp.hearing_tier("voellig_unbekannt") == 3
    assert gpp.priority_of("voellig_unbekannt") == 5
