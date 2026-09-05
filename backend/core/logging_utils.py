"""
backend/core/logging_utils.py — Zentrale Logging-Hilfsfunktionen

Bereitet standardisierte Logger-Instanzen für alle Aurik-Module vor.
Nutzt Python's logging.getLogger mit Modul-Namen als Identifier.

Usage:
    from backend.core.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("§v10 Nachricht")
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Gibt einen Logger mit dem angegebenen Modul-Namen zurück.

    Args:
        name: Modul-Name (typischerweise `__name__`).

    Returns:
        Konfigurierter Logger für das Modul.
    """
    return logging.getLogger(name)


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    fmt: str | None = None,
) -> logging.Logger:
    """Konfiguriert einen Logger mit Level und Format.

    Args:
        name: Modul-Name (typischerweise `__name__`).
        level: Logging-Level (default DEBUG).
        fmt: Log-Format-String (optional).

    Returns:
        Konfigurierter Logger für das Modul.
    """
    logger = get_logger(name)
    if not fmt:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
