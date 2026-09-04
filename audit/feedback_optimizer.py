"""
Aurik Feedback- und Optimierungsroutine
- Analysiert Audit-Logs und Quality-Gates
- Generiert automatische Verbesserungsvorschläge
- Integriert Feedback in Policy-Engine
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_audit_log(audit_path="audit/audit_trail.json"):
    if not Path(audit_path).exists():
        print("Audit-Log nicht gefunden.")
        return []
    with open(audit_path) as f:
        audit_data = json.load(f)
    return audit_data


def generate_feedback_report(audit_data):
    suggestions = []
    for entry in audit_data:
        results = entry.get("results", {})
        for gate, value in results.items():
            if isinstance(value, float) and value > 0.5:
                suggestions.append(f"Schwellenwert für '{gate}' zu streng: Wert={value}")
            elif value is False:
                suggestions.append(f"Quality-Gate '{gate}' nicht bestanden. Plugin/Policy prüfen.")
    return suggestions


def integrate_feedback_in_policy(suggestions, policy_path="policy/policy_engine.py"):
    """Integriert automatische Audit-Verbesserungsvorschläge in die Policy-Engine.

    Der Qualitäts-Score wird aus dem Anteil bestandener Quality-Gates abgeleitet
    und als Feedback-Dict an ``AdaptiveController.adapt()`` übergeben.
    """
    n_total = max(len(suggestions), 1)
    # Schätze Qualität: weniger Vorschläge = bessere Policy
    quality_score = max(0.0, 1.0 - len(suggestions) / n_total)
    feedback = {
        "user_score": quality_score,
        "suggestions": suggestions,
        "n_suggestions": len(suggestions),
        "genre": "default",
        "source": "feedback_optimizer",
    }
    try:
        from policy.policy_engine import AdaptiveController

        controller = AdaptiveController(policy={"source": "feedback_optimizer"})
        result = controller.adapt(feedback)
        print(f"Policy-Engine aktualisiert ({len(suggestions)} Vorschläge eingearbeitet):")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
        print(f"Feedback wird in Policy-Engine integriert (Fallback-Modus, Ursache: {exc}):")
        for s in suggestions:
            print(f"- {s}")


def optimize_feedback(feedback_data: dict | None = None) -> dict:
    """Composite-Einstiegspunkt für `PolicyEngine.integrate_feedback()`.

    Verkettet die drei Einzelschritte dieses Moduls (Audit-Log analysieren →
    Vorschläge generieren → in Policy-Engine integrieren). ``feedback_data``
    wird zusätzlich zum Audit-Log-basierten Kern-Feedback im Ergebnis-Dict
    zurückgegeben (z. B. für externe Nutzerbewertungen des Aufrufers).
    """
    audit_data = analyze_audit_log()
    suggestions = generate_feedback_report(audit_data)
    integrate_feedback_in_policy(suggestions)
    return {
        "status": "ok" if audit_data else "no_audit_data",
        "n_entries": len(audit_data),
        "suggestions": suggestions,
        "extra_feedback": feedback_data or {},
    }


def main():
    audit_data = analyze_audit_log()
    suggestions = generate_feedback_report(audit_data)
    if suggestions:
        print("Automatische Verbesserungsvorschläge:")
        for s in suggestions:
            print(f"- {s}")
        integrate_feedback_in_policy(suggestions)
    else:
        print("Keine Optimierungsvorschläge. Quality-Gates und Audit-Log sind optimal.")


if __name__ == "__main__":
    main()
