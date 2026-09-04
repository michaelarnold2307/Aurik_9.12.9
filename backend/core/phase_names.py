"""Human-readable phase name mapping for log messages.

Usage:
    from backend.core.phase_names import phase_human_name
    name = phase_human_name("phase_03_denoise")  # → "Entrauschen"
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def phase_human_name(phase_id: str) -> str:
    """Return a human-readable name for a phase_id, or the id itself.

    Delegates to the canonical registry in backend.core.phase_icons.
    """
    # Strip common prefixes
    for prefix in ("backend/core/phases/", "phases/"):
        if phase_id.startswith(prefix):
            phase_id = phase_id[len(prefix) :]
    if phase_id.endswith(".py"):
        phase_id = phase_id[:-3]
    try:
        from backend.core.phase_icons import phase_name_de

        return phase_name_de(phase_id)
    except ImportError as exc:
        logger.debug("§V6 phase_icons.phase_name_de nicht verfügbar — Phase-ID unverändert zurückgegeben: %s", exc)
        return phase_id


def phase_human_name_with_icon(phase_id: str) -> str:
    """Return icon + human-readable name, e.g. '🔍 Knackser-Entfernung'."""
    try:
        from backend.core.phase_icons import phase_display

        return phase_display(phase_id)
    except ImportError as exc:
        logger.debug("§V6 phase_icons.phase_display nicht verfügbar — phase_human_name Fallback aktiviert: %s", exc)
        return phase_human_name(phase_id)
