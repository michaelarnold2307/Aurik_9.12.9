from __future__ import annotations

import numpy as np
import pytest

from backend.core.excellence_optimizer import ExcellenceOptimizer


class _DummyChecker:
    def __init__(self, custom_thresholds: dict[str, float] | None = None):
        self._calls = 0

    def measure_all(self, _audio: np.ndarray, _sr: int) -> dict[str, float]:
        self._calls += 1
        # Erstaufruf: Referenz (vorher), Zweitaufruf: nach Optimizer
        if self._calls == 1:
            return {
                "natuerlichkeit": 0.95,
                "authentizitaet": 0.94,
                "spatial_depth": 0.80,
                "transient_energie": 0.85,
            }
        return {
            "natuerlichkeit": 0.90,  # -0.05 -> über 0.015, muss Rollback auslösen
            "authentizitaet": 0.93,
            "spatial_depth": 0.82,
            "transient_energie": 0.86,
        }


@pytest.mark.unit
def test_excellence_optimizer_rolls_back_on_core_goal_regression(monkeypatch):
    # Monkeypatch die get_checker()-FACTORY statt der Klasse: musical_goals_metrics
    # cached einen Singleton; ein früherer Test mit echter Instanz würde den
    # Klassen-Patch unwirksam machen (Order-Flaky-Fund, Rev. 2026-08-16).
    monkeypatch.setattr(
        "backend.core.musical_goals.musical_goals_metrics.get_checker",
        lambda custom_thresholds=None: _DummyChecker(custom_thresholds),
    )

    sr = 48_000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False, dtype=np.float32)
    audio = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    opt = ExcellenceOptimizer(sample_rate=sr)
    out, result = opt.optimize(audio)

    np.testing.assert_allclose(out, audio, atol=1e-6)
    assert "core_guard_rollback" in result.applied_steps
    assert result.core_guard_triggered is True
    assert any("natuerlichkeit:" in r for r in result.core_guard_regressions)
    assert result.delta_rms_db == 0.0
