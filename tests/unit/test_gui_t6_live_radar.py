"""§GUI-T6 — Live-15-Ziel-Radar während der Restaurierung.

Die Engine sendet je Fortschritt-Meldung den PMGG-Goal-Snapshot
(live_metrics["goals"]); ModernMainWindow._update_live_goal_radar() reicht
die Scores rein datengetrieben an das Radar-Widget weiter. Fehler im
Radar-Update dürfen den Restaurierungs-Progress nie unterbrechen
(absorbierter, geloggter Pfad).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from Aurik10.ui.modern_window import ModernMainWindow


class _RadarStub:
    def __init__(self) -> None:
        self.scores: dict | None = None

    def update_scores(self, scores: dict) -> None:
        self.scores = scores


class _RaisingRadar:
    def update_scores(self, scores: dict) -> None:
        raise RuntimeError("radar kaputt")


@pytest.mark.unit
def test_update_live_goal_radar_forwards_scores() -> None:
    radar = _RadarStub()
    dummy = SimpleNamespace(radar_widget=radar)
    goals = {"brillanz": 0.4, "spatial_depth": 0.7, "timbre_authentizitaet": 0.55}
    ModernMainWindow._update_live_goal_radar(dummy, dict(goals))  # type: ignore[arg-type]
    assert radar.scores == goals


@pytest.mark.unit
def test_update_live_goal_radar_no_widget_is_noop() -> None:
    dummy = SimpleNamespace(radar_widget=None)
    ModernMainWindow._update_live_goal_radar(dummy, {"brillanz": 0.4})  # type: ignore[arg-type]


@pytest.mark.unit
def test_update_live_goal_radar_absorbs_widget_errors() -> None:
    dummy = SimpleNamespace(radar_widget=_RaisingRadar())
    ModernMainWindow._update_live_goal_radar(dummy, {"brillanz": 0.4})  # type: ignore[arg-type]


@pytest.mark.unit
def test_batch_progress_wires_goals_to_live_radar() -> None:
    """§GUI-T6: _on_batch_progress liest metrics["goals"] und ruft den Radar-Hook."""
    import pathlib

    src = pathlib.Path("Aurik10/ui/modern_window.py").read_text(encoding="utf-8")
    assert 'metrics.get("goals")' in src
    assert "_update_live_goal_radar(dict(_live_goals))" in src
    assert 'def _update_live_goal_radar(self, goals: dict) -> None:' in src
