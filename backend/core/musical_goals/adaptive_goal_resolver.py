"""Adaptive goal threshold resolver — standalone module (P1-2 Modularisation Phase 1).

Extracted from ``backend.core.unified_restorer_v3.UnifiedRestorerV3._resolve_adaptive_goal_thresholds``
to make it independently importable, testable and reusable outside UV3.

Spec references:
    §2.31 AdaptiveGoalThresholds — Material- und Ära-adaptiv
    §2.33 PhysicalCeilingEstimator
    §1.2  15 Musical Goals

Usage:
    from backend.core.musical_goals.adaptive_goal_resolver import resolve_adaptive_goal_thresholds
    thresholds = resolve_adaptive_goal_thresholds(payload)
"""

from __future__ import annotations

import math
from typing import Any

#: Canonical goal names + accepted alias keys (incl. legacy hyphenated variants).
GOAL_ALIASES: dict[str, tuple[str, ...]] = {
    "bass_kraft": ("bass_kraft", "bass-kraft"),
    "brillanz": ("brillanz",),
    "waerme": ("waerme",),
    "natuerlichkeit": ("natuerlichkeit",),
    "authentizitaet": ("authentizitaet",),
    "emotionalitaet": ("emotionalitaet",),
    "transparenz": ("transparenz",),
    "groove": ("groove",),
    "spatial_depth": ("spatial_depth",),
    "timbre_authentizitaet": ("timbre_authentizitaet",),
    "tonal_center": ("tonal_center",),
    "micro_dynamics": ("micro_dynamics",),
    "separation_fidelity": ("separation_fidelity",),
    "artikulation": ("artikulation",),
    "transient_energie": ("transient_energie",),
}


def resolve_adaptive_goal_thresholds(adaptive_goals_payload: Any) -> dict[str, float]:
    """Extrahiert canonical adaptive thresholds for all 15 musical goals from mixed payloads.

    Supported payload shapes:
    - ``tuple`` / ``list``: e.g. the 3-tuple returned by
      ``get_adaptive_goals_and_config(audio, sr)``.  Each element is searched
      in sequence.
    - ``dict``: direct mapping with canonical *or* legacy hyphenated keys
      (``bass-kraft`` → ``bass_kraft``).
    - Object with direct attributes such as an ``AdaptiveGoalThresholds``
      dataclass (``obj.brillanz``, ``obj.bass_kraft`` …).
    - Object exposing a ``.thresholds`` dict attribute.

    Returns:
        Dict mapping canonical goal name → float threshold. Only goals whose
        value could be resolved are included; missing goals must fall back to
        the static defaults in ``MusicalGoalsChecker.thresholds``.
    """
    sources: list[Any] = [adaptive_goals_payload]
    if isinstance(adaptive_goals_payload, (tuple, list)):
        sources.extend(list(adaptive_goals_payload))

    resolved: dict[str, float] = {}

    for canonical, aliases in GOAL_ALIASES.items():
        for src in sources:
            if src is None:
                continue

            # 1) Dict payload (canonical or alias key)
            if isinstance(src, dict):
                for alias in aliases:
                    if alias in src:
                        try:
                            val = float(src[alias])
                            if math.isfinite(val):
                                resolved[canonical] = val
                                break
                        except Exception:
                            continue
                if canonical in resolved:
                    break

            # 2) Object with a .thresholds dict
            thresholds_dict = getattr(src, "thresholds", None)
            if isinstance(thresholds_dict, dict):
                for alias in aliases:
                    if alias in thresholds_dict:
                        try:
                            val = float(thresholds_dict[alias])
                            if math.isfinite(val):
                                resolved[canonical] = val
                                break
                        except Exception:
                            continue
                if canonical in resolved:
                    break

            # 3) Direct attribute on the object
            for alias in aliases:
                attr = getattr(src, alias, None)
                if attr is None:
                    continue
                try:
                    val = float(attr)
                    if math.isfinite(val):
                        resolved[canonical] = val
                        break
                except Exception:
                    continue
            if canonical in resolved:
                break

    return resolved
logger = logging.getLogger(__name__)
