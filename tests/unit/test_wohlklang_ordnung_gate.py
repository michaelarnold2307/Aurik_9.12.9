"""§Ebene-3 Wohlklang-Ordnungs-Gate — Unit Tests (Hörordnung §5).

Prüft das Audit-Gate für die lexikografische Wohlklang-Ordnung:
  - PASS bei ordnungskonformer Entscheidung.
  - VIOLATION bei lexikografischem Verstoß (niederrangiges Goal verbessert
    auf Kosten eines höherrangigen).
  - NOT_APPLICABLE bei leeren/NaN/Inf/kaputten Eingaben.
  - JSON-/dict-Metadaten und Determinismus.

Kein Audio-I/O; die Hörordnungs-Stufen werden ausschließlich aus
GoalPriorityProtocol recycelt (Stufe 1 Natürlichkeit > 2 Wärme >
3 Klarheit > 4 Brillanz).
"""

from __future__ import annotations

from backend.core.goal_priority_protocol import GoalPriorityProtocol
from backend.core.wohlklang_ordnung_gate import (
    NOT_APPLICABLE,
    PASS,
    VIOLATION,
    WohlklangOrdnungGate,
    WohlklangOrdnungResult,
)


def make_gate() -> WohlklangOrdnungGate:
    return WohlklangOrdnungGate()


class TestBasics:
    def test_status_constants(self) -> None:
        assert PASS == "PASS"
        assert VIOLATION == "VIOLATION"
        assert NOT_APPLICABLE == "NOT_APPLICABLE"

    def test_result_default(self) -> None:
        r = WohlklangOrdnungResult()
        assert r.status == NOT_APPLICABLE
        assert r.violated_goals == []
        assert r.violations == []


class TestPass:
    def test_pure_improvement_passes(self) -> None:
        r = make_gate().evaluate({"natuerlichkeit": 0.05, "waerme": 0.03})
        assert r.status == PASS
        assert r.violated_goals == []
        assert r.violations == []

    def test_improve_lower_at_no_higher_cost_passes(self) -> None:
        # Brillanz (Stufe 4) verbessert, kein höherrangiges Goal gesenkt.
        r = make_gate().evaluate({"brillanz": 0.02})
        assert r.status == PASS

    def test_same_tier_tradeoff_allowed(self) -> None:
        # transparenz & separation_fidelity sind beide Stufe 3 → Tie erlaubt.
        r = make_gate().evaluate({"transparenz": 0.05, "separation_fidelity": -0.04})
        assert r.status == PASS

    def test_sub_epsilon_noise_ignored(self) -> None:
        # Wirksame Verbesserung + nicht relevanter "Verlust" unter _EPSILON.
        r = make_gate().evaluate({"brillanz": 0.01, "natuerlichkeit": -1e-12})
        assert r.status == PASS


class TestViolation:
    def test_lower_improved_at_cost_of_higher(self) -> None:
        # transparenz (Stufe 3) verbessert auf Kosten von natuerlichkeit (Stufe 1).
        r = make_gate().evaluate({"transparenz": 0.10, "natuerlichkeit": -0.08})
        assert r.status == VIOLATION
        assert "natuerlichkeit" in r.violated_goals

    def test_brillanz_at_cost_of_waerme(self) -> None:
        r = make_gate().evaluate({"brillanz": 0.06, "waerme": -0.03})
        assert r.status == VIOLATION
        assert r.violated_goals == ["waerme"]

    def test_violations_recorded_with_tiers(self) -> None:
        r = make_gate().evaluate({"brillanz": 0.06, "natuerlichkeit": -0.05})
        assert r.status == VIOLATION
        assert len(r.violations) == 1
        v = r.violations[0]
        assert v["improving_goal"] == "brillanz"
        assert v["at_cost_of"] == "natuerlichkeit"
        assert v["improving_tier"] > v["cost_tier"]

    def test_multiple_violated_goals_sorted(self) -> None:
        # brillanz (4) verbessert auf Kosten von waerme (2) und natuerlichkeit (1).
        r = make_gate().evaluate(
            {"brillanz": 0.10, "waerme": -0.04, "natuerlichkeit": -0.05}
        )
        assert r.status == VIOLATION
        assert r.violated_goals == sorted(["waerme", "natuerlichkeit"])

    def test_threshold_gates_degradation(self) -> None:
        # Mit expliziter Schwelle zählt ein nur minimaler Verlust am
        # höherrangigen Goal nicht als Verstoß.
        r = make_gate().evaluate(
            {"brillanz": 0.10, "waerme": -0.001}, threshold=0.05
        )
        assert r.status == PASS


class TestNotApplicable:
    def test_none_input(self) -> None:
        r = make_gate().evaluate(None)
        assert r.status == NOT_APPLICABLE

    def test_empty_dict(self) -> None:
        r = make_gate().evaluate({})
        assert r.status == NOT_APPLICABLE

    def test_all_invalid_values(self) -> None:
        r = make_gate().evaluate(
            {"brillanz": float("nan"), "waerme": float("inf"), "natuerlichkeit": float("-inf")}
        )
        assert r.status == NOT_APPLICABLE

    def test_non_numeric_value_ignored(self) -> None:
        r = make_gate().evaluate({"waerme": "nicht-eine-Zahl"})
        assert r.status == NOT_APPLICABLE

    def test_mixed_invalid_falls_back_to_clean(self) -> None:
        # Nur der gültige Eintrag bleibt; reine Verbesserung → PASS.
        r = make_gate().evaluate({"brillanz": 0.05, "waerme": float("nan")})
        assert r.status == PASS


class TestMetadata:
    def test_to_dict_serializable(self) -> None:
        r = make_gate().evaluate({"brillanz": 0.06, "waerme": -0.03})
        d = r.to_dict()
        assert d["status"] == VIOLATION
        assert isinstance(d["violated_goals"], list)
        assert isinstance(d["violations"], list)

    def test_pass_dict(self) -> None:
        d = make_gate().evaluate({"natuerlichkeit": 0.05}).to_dict()
        assert d["status"] == PASS
        assert d["violated_goals"] == []
        assert d["violations"] == []


class TestDeterminism:
    def test_same_input_same_result(self) -> None:
        gate = make_gate()
        a = gate.evaluate({"brillanz": 0.06, "waerme": -0.03})
        b = gate.evaluate({"brillanz": 0.06, "waerme": -0.03})
        assert a.status == b.status == VIOLATION
        assert a.to_dict() == b.to_dict()

    def test_order_independent_violations(self) -> None:
        gate = make_gate()
        d1 = {"brillanz": 0.06, "waerme": -0.03}
        d2 = {"waerme": -0.03, "brillanz": 0.06}
        assert gate.evaluate(d1).to_dict() == gate.evaluate(d2).to_dict()


class TestRecycleNoNewThresholds:
    def test_tiers_match_protocol(self) -> None:
        """Die Stufen des Gates müssen exakt mit GoalPriorityProtocol übereinstimmen."""
        gate = make_gate()
        proto = GoalPriorityProtocol()
        for goal in proto.HEARING_TIER_MAP:
            assert gate._hearing_tier(goal) == proto.hearing_tier(goal)

    def test_no_hardcoded_tier_map(self) -> None:
        """Das Gate darf keine eigene, neu erfundene Stufen-Reihenfolge besitzen."""
        assert not hasattr(WohlklangOrdnungGate, "HEARING_TIER_MAP")
