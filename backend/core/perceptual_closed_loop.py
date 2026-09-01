#!/usr/bin/env python3
"""
§v10.620: Perceptual Closed-Loop — Wahrnehmungs-Regelkreis für die SOTA-Kette.

Problem: Die Repair-Kette misst Qualität an synthetischen Metriken (MSE),
nicht an menschlicher Wahrnehmung. Ein Schritt kann MSE verbessern, aber
MOS verschlechtern (z.B. zu aggressives De-Essing klingt "gequetscht").

Lösung: Nach jedem Repair-Schritt misst UTMOS v2 (trainierter MOS-Prädiktor)
die wahrgenommene Qualität VOR und NACH dem Schritt. Bei Verschlechterung
wird automatisch zurückgeblendet und die Strength reduziert.

Zusätzlich: Golden-Sample-Vergleich — wenn ein Referenz-Master existiert,
wird der Abstand zum Golden Sample gemessen und als Zielwert genutzt.

Regelkreis:
  restore → UTMOS messen → vergleichen → adaptieren → wiederholen
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

log = logging.getLogger(__name__)

SR = 48000

# Schwellwerte
MOS_DEGRADATION_TOLERANCE = 0.05  # Max. 0.05 MOS Verschlechterung pro Schritt
MOS_MIN_ABSOLUTE = 1.5  # Unter diesem MOS immer zurückblenden
GOLDEN_DISTANCE_TOLERANCE = 0.3  # Max. MOS-Abstand zum Golden Sample


@dataclass
class PerceptualResult:
    """Ergebnis einer Wahrnehmungs-Prüfung."""

    passed: bool
    mos_pre: float
    mos_post: float
    mos_delta: float
    golden_mos: float | None = None
    golden_distance: float | None = None
    adapted: bool = False  # Wurde Strength adaptiert?
    blend_ratio: float = 1.0  # 1.0 = kein Blend


class PerceptualClosedLoop:
    """
    Wahrnehmungs-Regelkreis mit UTMOS + Golden-Sample-Vergleich.

    Nutzung:
        loop = PerceptualClosedLoop()
        result = loop.evaluate(audio_pre, audio_post, sr, golden_sample=None)
        if not result.passed:
            audio_post = loop.blend_back(audio_pre, audio_post, result)
    """

    def __init__(self):
        self._utmos = None
        self._init_utmos()

    def _init_utmos(self):
        try:
            from plugins.utmos_plugin import get_utmos

            self._utmos = get_utmos()
            log.info("Perceptual Loop: UTMOS v2 geladen")
        except Exception as exc:
            log.warning("Perceptual Loop: UTMOS nicht verfügbar (%s) — Fallback auf RMS", exc)

    def estimate_mos(self, audio: np.ndarray, sr: int = SR) -> float:
        """Schätzt den Mean Opinion Score (1-5) via UTMOS v2."""
        if self._utmos is not None:
            try:
                result = self._utmos.estimate_mos(audio, sr)
                mos = getattr(result, "mos", None)
                if mos is None:
                    d = result.as_dict()
                    mos = d.get("mos")
                if mos is not None:
                    return float(mos)
            except Exception as exc:
                log.debug("UTMOS Schätzung fehlgeschlagen (%s) — RMS-Fallback", exc)

        # RMS-Fallback: lauter + dynamischer = besser (grobe Heuristik)
        rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-10))
        return float(np.clip(rms * 10 + 2.0, 1.0, 5.0))

    def evaluate(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sr: int = SR,
        golden_sample: np.ndarray | None = None,
    ) -> PerceptualResult:
        """
        Vergleicht wahrgenommene Qualität vor/nach einem Repair-Schritt.

        Returns:
            PerceptualResult mit passed-Flag und Adaptions-Empfehlung.
        """
        mos_pre = self.estimate_mos(audio_pre, sr)
        mos_post = self.estimate_mos(audio_post, sr)
        mos_delta = mos_post - mos_pre

        golden_mos = None
        golden_distance = None

        # Golden-Sample-Vergleich
        if golden_sample is not None:
            try:
                golden_mos = self.estimate_mos(golden_sample, sr)
                golden_distance = abs(mos_post - golden_mos)
            except Exception as _gold_exc:
                log.debug("Golden-MOS-Schätzung nicht verfügbar: %s", _gold_exc)

        passed = mos_delta >= -MOS_DEGRADATION_TOLERANCE and mos_post >= MOS_MIN_ABSOLUTE

        if golden_mos is not None and golden_distance is not None:
            if golden_distance > GOLDEN_DISTANCE_TOLERANCE:
                passed = False

        # Adaptions-Empfehlung: Blend-Ratio basierend auf Verschlechterung
        blend_ratio = 1.0
        if not passed:
            # Je schlechter der Schritt, desto mehr Original
            degradation = max(0.0, -mos_delta)
            blend_ratio = min(0.95, 0.5 + degradation * 2.0)

        return PerceptualResult(
            passed=passed,
            mos_pre=mos_pre,
            mos_post=mos_post,
            mos_delta=mos_delta,
            golden_mos=golden_mos,
            golden_distance=golden_distance,
            adapted=not passed,
            blend_ratio=blend_ratio,
        )

    def blend_back(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        result: PerceptualResult,
    ) -> np.ndarray:
        """Blendet basierend auf dem PerceptualResult zurück."""
        ratio = result.blend_ratio
        blended = ratio * audio_pre + (1 - ratio) * audio_post
        return cast(np.ndarray, blended.astype(np.float32))


@dataclass
class LoopReport:
    """Gesamtbericht des Regelkreises."""

    steps_evaluated: int = 0
    steps_adapted: int = 0
    final_mos: float = 0.0
    initial_mos: float = 0.0
    mos_improvement: float = 0.0
    golden_distance_final: float | None = None
    details: list[dict[str, Any]] = field(default_factory=list)


def run_closed_loop(
    audio_original: np.ndarray,
    audio_processed: np.ndarray,
    sr: int = SR,
    golden_sample: np.ndarray | None = None,
    max_iterations: int = 3,
    strength_decay: float = 0.7,
) -> tuple[np.ndarray, LoopReport]:
    """
    Iterativer Wahrnehmungs-Regelkreis:
    Verarbeitet, misst, adaptiert — bis zu max_iterations Runden.

    Args:
        audio_original: Unbearbeitetes Original
        audio_processed: Ergebnis der Repair-Kette
        golden_sample: Optionaler Referenz-Master
        max_iterations: Max. Adaptions-Runden
        strength_decay: Wie stark die Strength pro Runde reduziert wird

    Returns:
        (finales_audio, LoopReport)
    """
    loop = PerceptualClosedLoop()
    report = LoopReport()

    report.initial_mos = loop.estimate_mos(audio_original, sr)

    current = audio_processed
    prev = audio_original

    for iteration in range(max_iterations):
        report.steps_evaluated += 1
        result = loop.evaluate(prev, current, sr, golden_sample)
        report.details.append(
            {
                "iteration": iteration,
                "mos_pre": result.mos_pre,
                "mos_post": result.mos_post,
                "mos_delta": result.mos_delta,
                "passed": result.passed,
                "blend_ratio": result.blend_ratio,
            }
        )

        if result.passed:
            break

        # Verschlechterung → zurückblenden
        current = loop.blend_back(prev, current, result)
        report.steps_adapted += 1
        prev = current  # Nächste Runde vergleicht gegen das Geblendete

    report.final_mos = loop.estimate_mos(current, sr)
    report.mos_improvement = report.final_mos - report.initial_mos

    if golden_sample is not None:
        try:
            report.golden_distance_final = abs(report.final_mos - loop.estimate_mos(golden_sample, sr))
        except Exception as _gold_exc:
            log.debug("Golden-Abstand nicht verfügbar: %s", _gold_exc)

    return current, report
