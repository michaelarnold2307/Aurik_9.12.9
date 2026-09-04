"""
Aurik User-Feedback-Analyse und Community-Bewertung
- Analysiert User-Feedbacks und Community-Bewertungen nach jedem Release/Testlauf
- Erkennt Trends, Schwachstellen und Verbesserungspotenziale
- Integriert Feedback in Policy-Engine und Quality-Gates
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_feedback(feedback_path="audit/user_feedback.json"):
    if not Path(feedback_path).exists():
        print("User-Feedback nicht gefunden.")
        return []
    with open(feedback_path) as f:
        return json.load(f)


def analyze_feedback(feedback_data):
    trends: dict[str, list[str]] = {}
    for entry in feedback_data:
        rating = entry.get("rating", 0)
        comment = entry.get("comment", "")
        if rating < 4:
            trends.setdefault("Verbesserungspotenzial", []).append(comment)
        else:
            trends.setdefault("Stärken", []).append(comment)
    return trends


def integrate_feedback_in_policy(trends, policy_path="policy/policy_engine.py"):
    """Integriert analysierte Feedback-Trends in die Aurik Policy-Engine.

    Berechnet einen aggregierten user_score und übergibt ihn als strukturiertes
    Feedback-Dict an ``AdaptiveController.adapt()``. Bei Import-Fehler fällt die
    Funktion auf einfaches stdout-Reporting zurück.
    """
    all_comments = [c for v in trends.values() for c in v]
    low_ratings = trends.get("Verbesserungspotenzial", [])
    user_score = max(0.0, 1.0 - len(low_ratings) / max(len(all_comments), 1))
    feedback = {
        "user_score": user_score,
        "trends": {k: len(v) for k, v in trends.items()},
        "n_feedback": len(all_comments),
        "genre": "default",
        "source": "user_feedback_analyzer",
    }
    try:
        from policy.policy_engine import AdaptiveController

        controller = AdaptiveController(policy={"source": "user_feedback_analyzer"})
        result = controller.adapt(feedback)
        print(f"Policy-Engine aktualisiert (user_score={user_score:.2f}):")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
        print(f"Policy-Engine-Integration (Fallback-Modus, Ursache: {exc}):")
        for k, v in trends.items():
            for comment in v:
                print(f"- {k}: {comment}")


def main():
    feedback_data = load_feedback()
    trends = analyze_feedback(feedback_data)
    if trends:
        print("User-Feedback-Analyse:")
        for k, v in trends.items():
            print(f"{k}:")
            for comment in v:
                print(f"  - {comment}")
        integrate_feedback_in_policy(trends)
    else:
        print("Keine User-Feedbacks vorhanden.")


if __name__ == "__main__":
    main()
