"""Preference-Learning — Aurik lernt vom Nutzer-Feedback.

Speichert Nutzer-Präferenzen pro Kategorie:
- "zu viel De-Essing" → reduziert Sibilance-Stärke
- "zu wenig Bass" → erhöht Bass-Enhancement
- "klingt künstlich" → reduziert globale Stärke
- "perfekt" → speichert Parameter als Gold-Standard

Persistenz: ~/.aurik/preferences.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)
PREF_PATH = Path.home() / ".aurik" / "preferences.json"

FEEDBACK_CATEGORIES = {
    "too_much_deesser": {"target": "phase_19", "adjust": -0.10},
    "too_little_deesser": {"target": "phase_19", "adjust": +0.10},
    "too_much_denoise": {"target": "phase_03", "adjust": -0.10},
    "too_little_denoise": {"target": "phase_03", "adjust": +0.10},
    "too_bright": {"target": "phase_38", "adjust": -0.10},
    "too_dull": {"target": "phase_38", "adjust": +0.10},
    "too_much_bass": {"target": "phase_37", "adjust": -0.10},
    "too_little_bass": {"target": "phase_37", "adjust": +0.10},
    "sounds_artificial": {"target": "global_scalar", "adjust": -0.05},
    "sounds_untouched": {"target": "global_scalar", "adjust": +0.05},
    "perfect": {"target": "lock", "adjust": 0.0},
}


class PreferenceLearner:
    """Sammelt Nutzer-Feedback und passt zukünftige Läufe an."""

    def __init__(self) -> None:
        self._prefs: dict[str, dict[str, float]] = {}  # material → {phase_id: adjustment}
        self._history: list[dict[str, Any]] = []
        self._gold_standards: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if PREF_PATH.exists():
            try:
                with open(PREF_PATH) as f:
                    data = json.load(f)
                self._prefs = data.get("preferences", {})
                self._history = data.get("history", [])
                self._gold_standards = data.get("gold_standards", {})
            except Exception as _load_exc:
                logger.debug("Präferenz-Datei nicht lesbar: %s", _load_exc)

    def _save(self) -> None:
        PREF_PATH.parent.mkdir(exist_ok=True)
        with open(PREF_PATH, "w") as f:
            json.dump(
                {"preferences": self._prefs, "history": self._history[-100:], "gold_standards": self._gold_standards},
                f,
                indent=2,
            )

    def record_feedback(
        self, material: str, phase_strengths: dict[str, float], feedback: str, user_comment: str = ""
    ) -> None:
        """Verarbeitet Nutzer-Feedback und aktualisiert Präferenzen."""
        if feedback not in FEEDBACK_CATEGORIES:
            logger.warning("Unbekanntes Feedback: %s", feedback)
            return

        cat = FEEDBACK_CATEGORIES[feedback]
        target = str(cat["target"])
        adjust = cast(float, cat["adjust"])

        if target == "lock":
            self._gold_standards[material] = {
                "strengths": phase_strengths,
                "timestamp": __import__("time").time(),
                "comment": user_comment,
            }
        else:
            mat_prefs = self._prefs.setdefault(material, {})
            current = mat_prefs.get(target, 0.0)
            mat_prefs[target] = np.clip(current + adjust, -0.5, 0.5)

        self._history.append(
            {"material": material, "feedback": feedback, "comment": user_comment, "time": __import__("time").time()}
        )
        self._save()
        logger.info("Preference gelernt [%s]: %s → %s %+.2f", material, feedback, target, adjust)

    def get_adjustments(self, material: str) -> dict[str, float]:
        """Gibt alle gelernten Anpassungen für ein Material zurück."""
        mat_prefs = self._prefs.get(material, {})
        # Interpoliere mit ähnlichen Materialien
        if not mat_prefs:
            for mat, prefs in self._prefs.items():
                if mat[:3] == material[:3]:  # Ähnliches Material-Präfix
                    for k, v in prefs.items():
                        mat_prefs[k] = mat_prefs.get(k, 0.0) * 0.5 + v * 0.5
        return mat_prefs

    def get_gold_standard(self, material: str) -> dict[str, Any] | None:
        return self._gold_standards.get(material)

    def get_learning_summary(self) -> dict[str, Any]:
        return {
            "n_materials": len(self._prefs),
            "n_feedbacks": len(self._history),
            "n_gold_standards": len(self._gold_standards),
            "most_common_feedback": max(
                {h["feedback"] for h in self._history[-50:]},
                key=lambda x: sum(1 for h in self._history[-50:] if h["feedback"] == x),
            )
            if self._history
            else "none",
        }
