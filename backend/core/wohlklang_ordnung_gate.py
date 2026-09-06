"""§Ebene-3 (Hörordnung) Wohlklang-Ordnungs-Gate — Audit/Verifikation.

Prüft, ob eine gegebene Goal-/Kandidaten-Entscheidung (z. B. das Ranking aus
`goal_priority_protocol.py` oder FeedbackChain-Kandidaten) die **lexikografische
Wohlklang-Ordnung** respektiert.

Hörordnung Ebene 3 (§5 (hoerordnung.instructions.md)): Bei Zielkonflikt gilt die
Reihenfolge strikt — ein höherrangiges Ziel darf für kein niederrangiges gesenkt
werden. Konkret dominiert

    1. Natürlichkeit  >  2. Wärme  >  3. Klarheit/Durchhörbarkeit  >  4. Brillanz

(strikte Dominanz, kein weiches Gewichten; „Teamwork- statt Dominanz-Prinzip",
Spec 01 §1.2c). Die Reihenfolge wird hier **nicht neu erfunden**, sondern
ausschließlich aus `GoalPriorityProtocol.HEARING_TIER_MAP`/`hearing_tier()`
und `PRIORITY_MAP` recycelt (§2.34 (specs/01_musical_goals.md)).

Dieses Modul ist ein **Audit-Gate**: Es verifiziert eine bereits getroffene
Entscheidung; es trifft selbst keine. Ein Verstoß liegt vor, wenn ein Kandidat
ein niederrangiges Goal verbessert (positives Delta) und zugleich ein
höherrangiges Goal verschlechtert (negatives Delta), ohne dass die
Dominanzbedingung der Hörordnung greift.

Deterministisch, numpy-frei, rein deklarativ über die übergebenen Deltas.
Reference: .github/instructions/hoerordnung.instructions.md §5, §8 (Ebene 3).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Status-Konstanten ──────────────────────────────────────────────────────
PASS = "PASS"
VIOLATION = "VIOLATION"
NOT_APPLICABLE = "NOT_APPLICABLE"

# Minimale Delta-Magnitude, ab der eine Score-Änderung als „wirksam" zählt.
# Dies ist KEIN psychoakustischer Schwellwert, sondern nur eine numerische
# Untergrenze gegen Gleitkomma-Rauschen (regt keine Reparatur an).
_EPSILON = 1e-9


@dataclass
class WohlklangOrdnungResult:
    """Ergebnis des Ebene-3-Audit-Gates.

    Attributes:
        status: "PASS" | "VIOLATION" | "NOT_APPLICABLE".
        violated_goals: Liste der höherrangigen Goals, die zugunsten eines
            niederrangigen gesenkt wurden (bei VIOLATION gefüllt).
        detail: Deutschlandsprachige Kurzbeschreibung des Befunds.
        violations: Liste der einzelnen Verstoß-Paare
            ({"improving_goal", "at_cost_of", "improving_tier",
            "cost_tier"}), für Doku/Metadaten.
    """

    status: str = NOT_APPLICABLE
    violated_goals: list[str] = field(default_factory=list)
    detail: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flache JSON-serialisierbare Repräsentation."""
        return {
            "status": self.status,
            "violated_goals": list(self.violated_goals),
            "detail": self.detail,
            "violations": [dict(v) for v in self.violations],
        }


class WohlklangOrdnungGate:
    """Audit-Gate für die lexikografische Wohlklang-Ordnung (Ebene 3).

    Verifiziert Goal-/Kandidaten-Entscheidungen gegen die strikte
    Dominanzreihenfolge aus GoalPriorityProtocol — ohne eigene Schwellwerte
    oder Neu-Implementierung der Priorisierungslogik.
    """

    def __init__(self, protocol: Any | None = None) -> None:
        """Erzeugt das Gate; injizierter Protocol optional (Default: Singleton)."""
        if protocol is not None:
            self._protocol = protocol
            self._hearing_tier = self._tier_from_protocol(protocol)
        else:
            from backend.core.goal_priority_protocol import get_goal_priority_protocol

            self._protocol = get_goal_priority_protocol()
            self._hearing_tier = self._protocol.hearing_tier

    @staticmethod
    def _tier_from_protocol(protocol: Any):
        """Bindet `hearing_tier` an ein injiziertes Protokoll (falls vorhanden)."""
        if hasattr(protocol, "hearing_tier"):
            return protocol.hearing_tier
        # Fallback über HEARING_TIER_MAP (rekursiv recyceln statt neu erfinden).
        tier_map = getattr(protocol, "HEARING_TIER_MAP", {}) or {}

        def _tier(goal: str) -> int:
            key = str(goal or "").lower().replace(" ", "_").replace("-", "_")
            return int(tier_map.get(key, 3))

        return _tier

    def evaluate(
        self,
        goal_deltas: dict[str, float] | None,
        *,
        threshold: float | None = None,
    ) -> WohlklangOrdnungResult:
        """Prüft eine Kandidaten-Entscheidung gegen die Wohlklang-Ordnung.

        Args:
            goal_deltas: Mapping `{goal: delta}` mit der Score-Änderung je Goal
                (positiv = Verbesserung, negativ = Verschlechterung), wie sie
                ein Kandidat/Eingriff erzeugt (z. B. `scores_after - scores_before`
                aus der FeedbackChain oder ein End-Gate-Ranking).
            threshold: Optionaler Delta-Betrag, ab dem eine Änderung als
                relevant gilt. Default: `None` → nur die numerische
                `_EPSILON`-Untergrenze (kein neuer Wahrnehmungs-Schwellwert).

        Returns:
            WohlklangOrdnungResult mit ``status`` (PASS/VIOLATION/
            NOT_APPLICABLE), ``violated_goals`` und ``detail``.
        """
        data = goal_deltas or {}
        if not isinstance(data, dict) or not data:
            return WohlklangOrdnungResult(
                status=NOT_APPLICABLE,
                detail="keine Goal-Deltas übergeben (keine Entscheidung zu prüfen)",
            )

        # Numerische Robustheit: nicht-endliche/unsaubere Einträge filtern.
        cleaned: dict[str, float] = {}
        for goal, delta in data.items():
            key = str(goal)
            try:
                value = float(delta)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            cleaned[key] = value

        if not cleaned:
            return WohlklangOrdnungResult(
                status=NOT_APPLICABLE,
                detail="keine auswertbaren Goal-Deltas (leer/NaN/Inf)",
            )

        improving: dict[str, float] = {}  # tier -> [goals], wir speichern je Goal
        degrading: dict[str, float] = {}

        for goal, delta in cleaned.items():
            if delta > _EPSILON:
                improving[goal] = delta
            elif delta < -_EPSILON:
                degrading[goal] = delta

        violations: list[dict[str, Any]] = []
        relevant = abs(float(threshold)) if threshold is not None else 0.0

        for imp_goal, imp_delta in improving.items():
            imp_tier = self._hearing_tier(imp_goal)
            for cost_goal, cost_delta in degrading.items():
                cost_tier = self._hearing_tier(cost_goal)
                # Strikte Dominanz: Verbesserung eines NIEDERRANGIGEN Goals
                # (größere Stufen-Zahl) auf Kosten eines HÖHERRANGIGEN
                # (kleinere Stufen-Zahl) ist verboten.
                # Zielkonflikte innerhalb derselben Stufe sind erlaubt (Tie).
                if imp_tier <= cost_tier:
                    continue
                # Dominanzbedingung: nur ein *wirksamer* Verlust am höherrangigen
                # Goal zählt (Betrag über Schwelle). Verbesserung muss nicht
                # relevant sein — die Richtung allein entscheidet (strikt).
                cost_magnitude = abs(cost_delta)
                if cost_magnitude <= relevant:
                    continue
                violations.append(
                    {
                        "improving_goal": imp_goal,
                        "at_cost_of": cost_goal,
                        "improving_tier": imp_tier,
                        "cost_tier": cost_tier,
                    }
                )

        if violations:
            # Eindeutige, deterministische Reihenfolge der verletzten Goals.
            violated = sorted({v["at_cost_of"] for v in violations})
            for v in violations:
                logger.warning(
                    "§Ebene-3 Wohlklang-Ordnung verletzt: %s (Stufe %d) verbessert "
                    "auf Kosten von %s (Stufe %d)",
                    v["improving_goal"],
                    v["improving_tier"],
                    v["at_cost_of"],
                    v["cost_tier"],
                )
            return WohlklangOrdnungResult(
                status=VIOLATION,
                violated_goals=violated,
                detail=(
                    f"{len(violations)} Verstoß/Verstöße: niederrangige(s) Goal(s) "
                    f"verbessert auf Kosten höherrangiger ({', '.join(violated)})"
                ),
                violations=violations,
            )

        return WohlklangOrdnungResult(
            status=PASS,
            detail="Entscheidung respektiert die lexikografische Wohlklang-Ordnung",
        )


__all__ = [
    "NOT_APPLICABLE",
    "PASS",
    "VIOLATION",
    "WohlklangOrdnungGate",
    "WohlklangOrdnungResult",
]
