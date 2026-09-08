"""§GUI-T1 — musical_goals_radar: headless-testbare Datenlogik.

Die Entscheidungslogik (Balken-Farbe) und die Result-Extraktion wurden aus
dem Qt-Widget in pure Funktionen gezogen — damit sind die 15-Goal-Anzeige
und ihre Status-Zustände ohne GUI verifizierbar.
"""

from __future__ import annotations

from types import SimpleNamespace

from Aurik10.ui.musical_goals_radar import (
    DEFAULT_GOALS,
    build_radar_update_payload,
    goal_bar_state,
)


def test_default_goals_complete() -> None:
    """Die 15 Musical Goals (§1.2) müssen vollständig und eindeutig sein."""
    keys = [g.key for g in DEFAULT_GOALS]
    assert len(keys) == 15
    assert len(set(keys)) == 15  # keine Duplikate
    for g in DEFAULT_GOALS:
        assert 0.0 < g.threshold <= 1.0
        assert g.label


def test_goal_bar_state_matrix() -> None:
    t = 0.85
    assert goal_bar_state(0.90, t) == "pass"  # ≥ t+0.04
    assert goal_bar_state(0.86, t) == "warn"  # t ≤ score < t+0.04
    assert goal_bar_state(0.84, t) == "fail"
    assert goal_bar_state(0.85, t) == "warn"  # exakt auf Schwelle = warn
    assert goal_bar_state(0.85, t, applicable=False) == "na"
    assert goal_bar_state(0.50, t, synthesized=True) == "synth"


def test_payload_extraction_full() -> None:
    result = SimpleNamespace(
        musical_goals={"brillanz": 0.9, "waerme": 0.5},
        adaptive_thresholds=SimpleNamespace(
            thresholds={"brillanz": 0.7},
            adaptations={"brillanz": "Material fair"},
        ),
        goal_applicability=SimpleNamespace(
            applicable=["brillanz"],
            reasons={"waerme": "nicht messbar"},
        ),
        genealogy=SimpleNamespace(
            operations=[SimpleNamespace(operation_type="synthesize_brillanz")]
        ),
    )
    p = build_radar_update_payload(result)
    assert p["scores"] == {"brillanz": 0.9, "waerme": 0.5}
    assert p["adaptive_thresholds"] == {"brillanz": 0.7}
    assert p["applicable_goals"] == {"brillanz"}
    assert p["inapplicable_reasons"] == {"waerme": "nicht messbar"}
    assert p["synthesized_goals"] == {"brillanz"}
    assert p["adaptation_reasons"] == {"brillanz": "Material fair"}


def test_payload_extraction_graceful_empty() -> None:
    """Leeres Result → keine Crash, nur leere Payloads."""
    p = build_radar_update_payload(SimpleNamespace())
    assert p["scores"] == {}
    assert p["adaptive_thresholds"] is None
    assert p["applicable_goals"] is None
    assert p["synthesized_goals"] is None


def test_payload_mushra_percent_conversion() -> None:
    result = SimpleNamespace(mushra_scores={"brillanz": 91.0, "waerme": 0.55})
    p = build_radar_update_payload(result)
    # Prozentwerte (>1) werden auf [0,1] normiert; 0.55 bleibt
    assert abs(p["scores"]["brillanz"] - 0.91) < 1e-9
    assert abs(p["scores"]["waerme"] - 0.55) < 1e-9
