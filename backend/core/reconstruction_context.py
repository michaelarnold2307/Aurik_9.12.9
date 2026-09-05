"""
backend/core/reconstruction_context.py — Rekonstruktions-Kontext für Pipeline

Speichert den Zustand der Lücken-Rekonstruktion zwischen Denker-Stufen.
Wird vom RekonstruktionsDenker (Stufe 7) erstellt und an RestaurierDenker
(Stufe 8) sowie UV3 weitergereicht.

Spec: .github/specs/03_cognitive_modules.md §2.43 (v10.0.0)

Usage:
    from backend.core.reconstruction_context import ReconstructionContext

    ctx = ReconstructionContext(
        gaps_found=5,
        gaps_repaired=4,
        total_repaired_ms=120.0,
        bandwidth_limited=True,
        estimated_original_bandwidth_hz=20000.0,
        reconstruction_quality=0.85,
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReconstructionContext:
    """Kontext für Lücken-Rekonstruktion zwischen Pipeline-Stufen.

    Pflicht-Felder nach Spec 03 §2.43 (v10.0.0):
        gaps_found:                  Anzahl erkannter Lücken
        gaps_repaired:               Anzahl erfolgreich gefüllter Lücken
        total_repaired_ms:           Gesamte reparierte Zeitdauer in ms
        bandwidth_limited:           True wenn BANDWIDTH_LOSS erkannt
        estimated_original_bandwidth_hz: Geschätzte Original-Bandbreite
        reconstruction_quality:      Qualität der Rekonstruktion [0, 1]

    Invarianten:
        - gaps_repaired <= gaps_found (nie mehr repariert als gefunden)
        - reconstruction_quality ∈ [0, 1]
        - total_repaired_ms >= 0
    """

    gaps_found: int = 0
    """Anzahl erkannter Lücken/Dropouts."""

    gaps_repaired: int = 0
    """Anzahl erfolgreich reparierter Lücken."""

    total_repaired_ms: float = 0.0
    """Gesamte reparierte Zeitdauer in Millisekunden."""

    bandwidth_limited: bool = False
    """True wenn BANDWIDTH_LOSS in Defekt-Analyse erkannt wurde."""

    estimated_original_bandwidth_hz: float = 20000.0
    """Geschätzte Original-Bandbreite vor Degradation (Hz)."""

    reconstruction_quality: float = 0.0
    """Qualität der Rekonstruktion ∈ [0, 1]."""

    def __post_init__(self) -> None:
        """Invarianten prüfen und Werte normalisieren."""
        # gaps_repaired darf nicht größer als gaps_found sein
        if self.gaps_repaired > self.gaps_found:
            self.gaps_repaired = self.gaps_found

        # reconstruction_quality auf [0, 1] begrenzen
        self.reconstruction_quality = max(0.0, min(1.0, self.reconstruction_quality))

        # total_repaired_ms muss >= 0 sein
        if self.total_repaired_ms < 0:
            self.total_repaired_ms = 0.0

    @property
    def repair_ratio(self) -> float:
        """Verhältnis reparierter zu gefundenen Lücken [0, 1]."""
        if self.gaps_found == 0:
            return 1.0
        return self.gaps_repaired / self.gaps_found
