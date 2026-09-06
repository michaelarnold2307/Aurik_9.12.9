"""Unit-Tests für scripts/benchmark_effizienz_matrix.py (Budget-Gate, Bootstrap-CI, Seeds).

Deterministisch, ohne Audio-I/O — die getesteten Helfer sind reine Funktionen
auf Ergebnis-Dicts. Reference: Spec 07 §9.1d (specs/07_quality_and_tests.md),
§Performance-Budget (copilot-instructions.md).
"""

from __future__ import annotations

from scripts.benchmark_effizienz_matrix import (
    _bootstrap_percentile_ci,
    _collect_quality_observations,
    _enforce_budget,
    _repeat_seed_schedule,
)


def _entry(**overrides) -> dict:
    base = {
        "cell": "fast",
        "wall_s": 10.0,
        "engine_total_s": 9.0,
        "quality_estimate": 0.72,
        "pqs_mos": 4.1,
        "hpi": 0.82,
        "artifact_freedom": 0.99,
        "pipeline_budget_timings": {
            "defect_scanner_s": 1.0,
            "phase_pipeline_s": 100.0,
            "feedback_chain_s": 50.0,
            "excellence_optimizer_s": 20.0,
            "restorability_estimator_s": 1.0,
            "export_flac_s": None,
        },
    }
    base.update(overrides)
    return base


def test_repeat_seed_schedule_deterministic():
    # §G5: gleiche Eingabe ⇒ gleiche Folge; 0-basiert ab Basis-Seed.
    assert _repeat_seed_schedule(42, 3) == [42, 43, 44]
    assert _repeat_seed_schedule(42, 1) == [42]
    assert _repeat_seed_schedule(7, 0) == [7]


def test_bootstrap_ci_null_for_single_observation():
    assert _bootstrap_percentile_ci([0.72]) is None


def test_bootstrap_ci_with_repeats_non_degenerate():
    reps = [_entry(quality_estimate=0.70 + 0.01 * i, pqs_mos=4.0 + 0.05 * i) for i in range(5)]
    main = _entry(repeats=reps)
    q_obs, m_obs = _collect_quality_observations(main)
    assert len(q_obs) >= 5
    assert len(m_obs) >= 5
    qci = _bootstrap_percentile_ci(q_obs)
    mci = _bootstrap_percentile_ci(m_obs)
    assert qci is not None and qci[0] < qci[1]
    assert mci is not None and mci[0] < mci[1]
    # Determinismus (§G5): identischer Aufruf ⇒ identisches Intervall.
    assert _bootstrap_percentile_ci(q_obs) == qci


def test_collect_quality_observations_without_repeats():
    main = _entry()
    q_obs, m_obs = _collect_quality_observations(main)
    assert 0.72 in q_obs
    assert 4.1 in m_obs
    assert 0.82 in q_obs  # hpi
    assert 0.99 in q_obs  # artifact_freedom


def test_enforce_budget_uses_pipeline_timings():
    # 30 s Audio ⇒ 0.5 min; defect_scanner 1.0 s ⇒ 2.0 s/min ≤ 4 — OK.
    violations, checks = _enforce_budget(_entry(), 0.5)
    assert violations == []
    for _op in (
        "defect_scanner",
        "phase_pipeline_total",
        "feedback_chain",
        "excellence_optimizer",
        "restorability_estimator",
    ):
        assert checks[_op] is not None
    # Export läuft außerhalb des Restorers — ehrlich null, nie geschätzt.
    assert checks["export_flac"] is None


def test_enforce_budget_violation_detected():
    # phase_pipeline 300 s auf 30 s Audio ⇒ 600 s/min > 240 ⇒ Verstoß.
    _pt = dict(_entry()["pipeline_budget_timings"])
    _pt["phase_pipeline_s"] = 300.0
    violations, checks = _enforce_budget(_entry(pipeline_budget_timings=_pt), 0.5)
    assert any(x["operation"] == "phase_pipeline_total" for x in violations)
    assert checks["phase_pipeline_total"]["violation"] is True


def test_enforce_budget_per_operation_violation():
    _pt = dict(_entry()["pipeline_budget_timings"])
    _pt["restorability_estimator_s"] = 5.0  # 10 s/min > 5 ⇒ Verstoß
    violations, _ = _enforce_budget(_entry(pipeline_budget_timings=_pt), 0.5)
    assert any(x["operation"] == "restorability_estimator" for x in violations)


def test_enforce_budget_legacy_fallback_without_timings():
    # 121 s auf 30 s Audio ⇒ 242 s/min > 240 ⇒ Verstoß über den Legacy-Fallback.
    e = _entry(pipeline_budget_timings=None, engine_total_s=121.0)
    violations, checks = _enforce_budget(e, 0.5)
    assert any(x["operation"] == "phase_pipeline_total" for x in violations)
    for _op in ("defect_scanner", "feedback_chain", "excellence_optimizer", "restorability_estimator"):
        assert checks[_op] is None
