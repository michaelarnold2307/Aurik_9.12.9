from __future__ import annotations

import threading
from dataclasses import dataclass, field

# pylint: disable=too-many-positional-arguments


@dataclass(frozen=True)
class ConflictResolutionResult:
    """Ergebnis einer Goal-Konflikt-Auflösung mit Gewinner, Verlierer und Begründung."""

    winner: str
    loser: str
    reason: str
    priority_winner: int
    priority_loser: int


@dataclass(frozen=True)
class IterationAbortResult:
    """Ergebnis der Iterationsabbruch-Prüfung mit Status und degradierten Goals."""

    should_abort: bool
    reason: str
    degraded_goals: list[str] = field(default_factory=list)


class GoalPriorityProtocol:
    """Priorisiert und vermittelt zwischen konkurrierenden Musical Goals (§2.29)."""

    PRIORITY_MAP: dict[str, int] = {
        "natuerlichkeit": 1,
        "authentizitaet": 1,
        "tonal_center": 2,
        "timbre_authentizitaet": 2,
        "artikulation": 2,
        "transient_energie": 2,
        "emotionalitaet": 3,
        "micro_dynamics": 3,
        "groove": 3,
        "transparenz": 4,
        "waerme": 4,
        "bass_kraft": 4,
        "separation_fidelity": 4,
        "brillanz": 5,
        "spatial_depth": 5,
    }

    # §Goal-Aliases (2026-09-08, P0-2 Tier-Map-Sync): Kanonische Namen für
    # Goal-Aliase aus musical_goals.instructions.md — verhindert stilles
    # Defaulten (Priorität 5 / Tier 3) bei abweichenden Schreibweisen.
    GOAL_ALIASES: dict[str, str] = {
        "timbre": "timbre_authentizitaet",
        "mikrodynamik": "micro_dynamics",
        "sep_fidelity": "separation_fidelity",
        "raumtiefe": "spatial_depth",
    }

    @classmethod
    def canonical_goal(cls, goal: str) -> str:
        """Kanonisiert einen Goal-Namen (Alias → kanonisch, Case/Zeichen-Normalisierung)."""
        _g = str(goal or "").lower().replace(" ", "_").replace("-", "_")
        _seen: set[str] = set()
        while _g in cls.GOAL_ALIASES and _g not in _seen:
            _seen.add(_g)
            _g = cls.GOAL_ALIASES[_g]
        return _g

    @classmethod
    def verify_map_consistency(cls) -> list[str]:
        """Prüft die Konsistenz von PRIORITY_MAP und HEARING_TIER_MAP.

        Beide Maps sind BEWUSST verschiedene Achsen (§2.29-Priorität für den
        FC-Abort vs. Hörordnungs-Dominanzstufe) — der Sync-Test schützt vor
        stiller Drift: (a) jede Prioritäts-Goal muss eine explizite Tier-
        Zuordnung haben (kein Default-3), (b) normative Anker müssen halten,
        (c) Aliase müssen auflösbar und zyklenfrei sein.
        """
        _problems: list[str] = []
        _tier = cls.HEARING_TIER_MAP
        # (a) Jede PRIORITY_MAP-Goal explizit in der Tier-Map?
        for _g in cls.PRIORITY_MAP:
            _c = cls.canonical_goal(_g)
            if _c not in _tier:
                _problems.append(f"PRIORITY_MAP-Goal '{_g}' (kanonisch '{_c}') fehlt in HEARING_TIER_MAP — würde still auf Tier 3 defaulten")
        # (b) Normative Hörordnungs-Anker (hoerordnung.instructions.md §5)
        _anchors = {"natuerlichkeit": 1, "waerme": 2, "transparenz": 3, "brillanz": 4}
        for _g, _expected in _anchors.items():
            if _tier.get(_g) != _expected:
                _problems.append(f"Tier-Anker '{_g}' = {_tier.get(_g)}, erwartet {_expected}")
        # (c) Aliase auflösbar und zyklenfrei (canonical_goal terminiert)
        for _alias, _target in cls.GOAL_ALIASES.items():
            _c = cls.canonical_goal(_alias)
            if _c == _alias:
                _problems.append(f"Alias '{_alias}' ist ein Selbst-Zyklus")
            if _c not in _tier:
                _problems.append(f"Alias '{_alias}' → '{_c}' fehlt in HEARING_TIER_MAP")
        return _problems

    ABORT_PRIORITY_THRESHOLD: int = 2
    REGRESSION_EPSILON: float = (
        0.012  # §2.29: GOOD threshold (0.001 was too strict, caused FC abort on numerical noise)
    )

    # Hörordnung Ebene 3 (hoerordnung.instructions.md §5): Lexikografische
    # Wohlklang-Ordnung. Strikte Dominanz: Ein Eingriff, der ein Goal einer
    # NIEDRIGEREN Stufe (kleinere Zahl = höherrangig) senkt, ist verboten,
    # auch wenn er ein Goal einer höheren Stufe verbessert.
    #   Stufe 1 Natürlichkeit > Stufe 2 Wärme > Stufe 3 Klarheit > Stufe 4 Brillanz
    # Ergänzt die Spec-PRIORITY_MAP (§2.34) — diese bleibt für deren Regeln
    # normativ; hearing_tier() ist die zusätzliche Hörordnungs-Dominanzstufe.
    HEARING_TIER_MAP: dict[str, int] = {
        "natuerlichkeit": 1,
        "authentizitaet": 1,
        "timbre_authentizitaet": 1,
        "micro_dynamics": 1,
        "emotionalitaet": 1,
        "formant_fidelity": 1,
        "vocal_quality": 1,
        "waerme": 2,
        "bass_kraft": 2,
        "tonal_center": 3,
        "artikulation": 3,
        "transparenz": 3,
        "separation_fidelity": 3,
        "brillanz": 4,
        "spatial_depth": 4,
        "transient_energie": 4,
        "groove": 4,
        "raumtiefe": 4,
    }

    def hearing_tier(self, goal: str) -> int:
        """Hörordnungs-Stufe eines Goals (1 = Natürlichkeit … 4 = Brillanz).

        Unbekannte Goals → Stufe 3 (neutral, weder dominierend noch dominiert).
        """
        _g = self.canonical_goal(goal)
        return int(self.HEARING_TIER_MAP.get(_g, 3))

    def would_violate_hearing_order(self, improving_goal: str, at_cost_of: str) -> bool:
        """True, wenn ein Gewinn bei `improving_goal` die Hörordnung verletzt,
        weil er auf Kosten eines höherrangigen Goals (`at_cost_of`) ginge.

        Hörordnung Ebene 3: strikte Dominanz — keine weiche Gewichtung.
        """
        return self.hearing_tier(improving_goal) > self.hearing_tier(at_cost_of)

    def resolve_conflict(
        self,
        goal_a: str,
        goal_b: str,
        delta_a: float,
        delta_b: float,
        headroom_a: float = 0.0,
        headroom_b: float = 0.0,
        goal_weights: dict[str, float] | None = None,
    ) -> ConflictResolutionResult:
        """Löst einen Konflikt zwischen zwei Goals nach Priorität, Headroom und Delta auf."""
        prio_a = self.priority_of(goal_a)
        prio_b = self.priority_of(goal_b)

        if prio_a < prio_b:
            return ConflictResolutionResult(goal_a, goal_b, "higher-priority goal wins", prio_a, prio_b)
        if prio_b < prio_a:
            return ConflictResolutionResult(goal_b, goal_a, "higher-priority goal wins", prio_b, prio_a)

        # §2.56: When priorities are equal, use song-specific weights as tiebreaker
        if goal_weights:
            w_a = goal_weights.get(goal_a, 1.0)
            w_b = goal_weights.get(goal_b, 1.0)
            if abs(w_a - w_b) > 0.05:  # meaningful weight difference
                if w_a > w_b:
                    return ConflictResolutionResult(
                        goal_a, goal_b, "equal priority, higher song-specific weight", prio_a, prio_b
                    )
                return ConflictResolutionResult(
                    goal_b, goal_a, "equal priority, higher song-specific weight", prio_b, prio_a
                )

        if headroom_a > headroom_b:
            return ConflictResolutionResult(goal_a, goal_b, "equal priority, higher headroom", prio_a, prio_b)
        if headroom_b > headroom_a:
            return ConflictResolutionResult(goal_b, goal_a, "equal priority, higher headroom", prio_b, prio_a)

        if delta_a >= delta_b:
            return ConflictResolutionResult(goal_a, goal_b, "equal priority/headroom, larger delta", prio_a, prio_b)
        return ConflictResolutionResult(goal_b, goal_a, "equal priority/headroom, larger delta", prio_b, prio_a)

    def should_abort_iteration(
        self,
        scores_before: dict[str, float],
        scores_after: dict[str, float],
        goal_weights: dict[str, float] | None = None,
    ) -> IterationAbortResult:
        """Prüft ob eine FeedbackChain-Iteration abgebrochen werden soll (kritische Goal-Regression)."""
        degraded: list[str] = []
        for goal, before in scores_before.items():
            after = scores_after.get(goal, before)
            # §2.56: Weight modulates the effective epsilon — important goals abort sooner
            w = goal_weights.get(goal, 1.0) if goal_weights else 1.0
            effective_epsilon = self.REGRESSION_EPSILON / max(w, 0.3)
            if before - after > effective_epsilon and self.priority_of(goal) <= self.ABORT_PRIORITY_THRESHOLD:
                degraded.append(goal)

        if degraded:
            return IterationAbortResult(True, "critical goal regression", degraded)
        return IterationAbortResult(False, "ok", [])

    def priority_of(self, goal: str) -> int:
        """Gibt die Prioritätsstufe eines Goals zurück (1 = höchste, 5 = niedrigste)."""
        return int(self.PRIORITY_MAP.get(self.canonical_goal(goal), 5))

    def sort_goals_by_priority(self, goals: list[str]) -> list[str]:
        """Sortiert Goals aufsteigend nach Prioritätsstufe."""
        return sorted(goals, key=self.priority_of)

    def goals_at_priority(self, level: int) -> list[str]:
        """Gibt alle Goals einer bestimmten Prioritätsstufe zurück."""
        return [g for g, p in self.PRIORITY_MAP.items() if p == level]

    def would_violate_priority(self, improving_goal: str, at_cost_of: str) -> bool:
        """Gibt True zurück wenn Verbesserung von improving_goal auf Kosten eines höherprioren Goals geht."""
        return self.priority_of(improving_goal) > self.priority_of(at_cost_of)

    def user_message_for_failure(self, goal: str) -> str:
        """Gibt eine deutschsprachige Fehlermeldung für ein nicht erreichtes Goal zurück."""
        if self.priority_of(goal) <= 2:
            return (
                "Die Restaurierung konnte zentrale Klangziele nicht voll erreichen. "
                "Das bestmögliche Ergebnis wurde dennoch ausgegeben."
            )
        return (
            "Einige zusätzliche Klangziele konnten nicht voll erreicht werden. "
            "Das ist bei diesem Material physikalisch bedingt."
        )


_instance: GoalPriorityProtocol | None = None
_lock = threading.Lock()


def get_goal_priority_protocol() -> GoalPriorityProtocol:
    """Thread-sicherer Singleton-Accessor für GoalPriorityProtocol."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GoalPriorityProtocol()
    return _instance


def resolve_goal_conflict(
    goal_a: str,
    goal_b: str,
    delta_a: float,
    delta_b: float,
    headroom_a: float = 0.0,
    headroom_b: float = 0.0,
    goal_weights: dict[str, float] | None = None,
) -> ConflictResolutionResult:
    """Convenience-Wrapper: löst einen Goal-Konflikt über den Singleton."""
    return get_goal_priority_protocol().resolve_conflict(
        goal_a, goal_b, delta_a, delta_b, headroom_a, headroom_b, goal_weights=goal_weights
    )


def check_iteration_abort(
    scores_before: dict[str, float],
    scores_after: dict[str, float],
    goal_weights: dict[str, float] | None = None,
) -> IterationAbortResult:
    """Convenience-Wrapper: prüft FeedbackChain-Iterationsabbruch über den Singleton."""
    return get_goal_priority_protocol().should_abort_iteration(scores_before, scores_after, goal_weights=goal_weights)


__all__ = [
    "ConflictResolutionResult",
    "GoalPriorityProtocol",
    "IterationAbortResult",
    "check_iteration_abort",
    "get_goal_priority_protocol",
    "resolve_goal_conflict",
]
