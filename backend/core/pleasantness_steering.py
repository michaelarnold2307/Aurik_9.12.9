"""
§v10 Pleasantness Steering — „Nicht aufgeben, sondern nachsteuern."

Ein menschlicher Toningenieur sagt nicht „das klingt schlecht, ich höre auf."
Er sagt: „Das klingt nicht gut — ich versuche es mit weniger Kompression."
Oder: „Dieser EQ macht es schlimmer — ich überspringe ihn."
Oder: „Die letzten drei Schritte haben es verschlechtert — zurück auf Stand vorher."

PleasantnessSteering automatisiert genau dieses Verhalten:

  Vor jedem Schritt:  HPE-Baseline messen
  Nach jedem Schritt:  HPE-Delta messen
  Wenn ΔP > 0:        ✅ Besser! Weiter.
  Wenn ΔP ≈ 0:        ⚠️  Neutral. Trotzdem weitermachen (nicht jede Verbesserung ist sofort hörbar).
  Wenn ΔP < -0.02:    🔄 Leichter wiederholen (reduzierte Intensität).
  Wenn ΔP < -0.05:    ⏭️  Schritt überspringen, Audio zurücksetzen.
  Wenn mehrmals ΔP<0: ⏪ Zurückrollen zum besten Zwischenstand.

Die Philosophie: JEDER Eingriff muss das Hörerlebnis VERBESSERN.
Tut er das nicht, wird er nicht akzeptiert — aber Aurik GIBT NICHT AUF.
Es wird nachgesteuert, bis das Optimum gefunden ist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Steering-Aktionen
# ═══════════════════════════════════════════════════════════════════════════


class SteerAction(Enum):
    """Aktionen, die PleasantnessSteering nach einem Schritt entscheiden kann."""

    CONTINUE = "continue"
    RETRY_LIGHTER = "retry_lighter"
    SKIP_AND_REVERT = "skip_and_revert"
    ROLLBACK_TO_BEST = "rollback_to_best"
    STOP_WITH_BEST = "stop_with_best"


@dataclass
class SteerDecision:
    """Entscheidung nach einem Pipeline-Schritt."""

    action: SteerAction
    reason: str
    delta_pleasantness: float = 0.0


@dataclass
class StepSnapshot:
    """Snapshot eines Pipeline-Schritts für Rollback."""

    step_id: int
    audio: np.ndarray
    pleasantness: float
    label: str


class PleasantnessSteering:
    """Steuert die Pipeline anhand psychoakustischer Angenehmheit.

    Nutzung:
        steering = PleasantnessSteering()
        steering.set_reference(original_audio, sr)

        for step in pipeline:
            steering.pre_step(audio)
            processed = step.run(audio)
            decision = steering.post_step(processed)

            if decision.action == SteerAction.CONTINUE:
                audio = processed
            elif decision.action == SteerAction.RETRY_LIGHTER:
                processed2 = step.run(audio, intensity*0.7)
                decision2 = steering.post_step(processed2)
                ...
            elif decision.action == SteerAction.SKIP_AND_REVERT:
                audio = steering.get_best_audio()
                continue
            ...
    """

    # Konfiguration
    DELTA_NEUTRAL: float = 0.015  # |ΔP| < 0.015 → neutral
    DELTA_RETRY: float = 0.03  # |ΔP| zwischen 0.015 und 0.05 → leichter wiederholen
    DELTA_SKIP: float = 0.06  # |ΔP| > 0.05 → überspringen
    MAX_CONSECUTIVE_DROPS: int = 2  # Nach so vielen Drops in Folge → Rollback
    MAX_RETRIES_PER_STEP: int = 2  # Max. Wiederholungen pro Schritt
    INTENSITY_REDUCTION_PER_RETRY: float = 0.35  # 35% weniger Intensität pro Retry

    def __init__(self) -> None:
        self._sr: int = 44100
        self._reference_audio: np.ndarray | None = None
        self._reference_pleasantness: float = 0.0

        # Tracking
        self._snapshots: list[StepSnapshot] = []
        self._best_snapshot: StepSnapshot | None = None
        self._consecutive_drops: int = 0
        self._retries_this_step: int = 0
        self._current_pre_audio: np.ndarray | None = None
        self._current_pre_pleasantness: float = 0.0
        self._total_steps: int = 0
        self._pleasantness_history: list[float] = []

    # ── Öffentliche API ──────────────────────────────────────────────────

    def set_reference(self, audio: np.ndarray, sr: int) -> None:
        """Setzt das Referenz-Audio (Original vor Pipeline-Start).

        Wird EINMAL vor der Pipeline aufgerufen.
        """
        self._sr = sr
        self._reference_audio = np.asarray(audio, dtype=np.float32).copy()
        self._reference_pleasantness = self._measure(audio)

        # Initialer Snapshot
        snap = StepSnapshot(
            step_id=0,
            audio=self._reference_audio.copy(),
            pleasantness=self._reference_pleasantness,
            label="Original",
        )
        self._snapshots = [snap]
        self._best_snapshot = snap
        self._consecutive_drops = 0
        self._retries_this_step = 0
        self._total_steps = 0
        self._pleasantness_history = [self._reference_pleasantness]

        logger.info(
            "Steering initialisiert: Referenz-Pleasantness = %.3f",
            self._reference_pleasantness,
        )

    def pre_step(self, audio: np.ndarray) -> None:
        """Vor einem Schritt: Audio-Zustand speichern."""
        self._current_pre_audio = np.asarray(audio, dtype=np.float32).copy()
        self._current_pre_pleasantness = self._measure(audio)
        self._retries_this_step = 0

    def post_step(self, processed: np.ndarray, step_info: str = "") -> SteerDecision:
        """Nach einem Schritt: HPE-Delta prüfen und SteerDecision zurückgeben.

        Args:
            processed:  Das verarbeitete Audio
            step_info:  Optionaler Name des Schritts für Logging

        Returns:
            SteerDecision mit der nächsten Aktion
        """
        self._total_steps += 1
        post_pleasantness = self._measure(processed)
        delta = post_pleasantness - self._current_pre_pleasantness

        self._pleasantness_history.append(post_pleasantness)

        # ── Entscheidungslogik ──

        # Case 1: Deutliche Verbesserung → WEITER!
        if delta > self.DELTA_NEUTRAL:
            self._consecutive_drops = 0
            self._retries_this_step = 0
            self._record_snapshot(self._total_steps, processed, post_pleasantness, step_info)
            logger.info("HPE ✅ ΔP=%+.3f — Verbesserung! (%s)", delta, step_info)
            return SteerDecision(
                action=SteerAction.CONTINUE,
                reason=f"Angenehmheit verbessert (ΔP={delta:+.3f})",
                delta_pleasantness=delta,
            )

        # Case 2: Neutral → trotzdem weitermachen
        if abs(delta) <= self.DELTA_NEUTRAL:
            self._consecutive_drops = 0
            self._retries_this_step = 0
            self._record_snapshot(self._total_steps, processed, post_pleasantness, step_info)
            logger.info("HPE ≈  ΔP=%+.3f — Neutral, weiter. (%s)", delta, step_info)
            return SteerDecision(
                action=SteerAction.CONTINUE,
                reason=f"Angenehmheit neutral (ΔP={delta:+.3f})",
                delta_pleasantness=delta,
            )

        # Case 3: Leichte Verschlechterung → RETRY mit weniger Intensität
        if delta > -self.DELTA_SKIP:
            self._retries_this_step += 1

            if self._retries_this_step <= self.MAX_RETRIES_PER_STEP:
                reduction = self.INTENSITY_REDUCTION_PER_RETRY * self._retries_this_step
                logger.info(
                    "HPE 🔄 ΔP=%+.3f — Leichte Verschlechterung. Wiederholung %d/%d mit %d%% weniger Intensität. (%s)",
                    delta,
                    self._retries_this_step,
                    self.MAX_RETRIES_PER_STEP,
                    int(reduction * 100),
                    step_info,
                )
                return SteerDecision(
                    action=SteerAction.RETRY_LIGHTER,
                    reason=f"Angenehmheit gesunken (ΔP={delta:+.3f}) — leichter wiederholen",
                    delta_pleasantness=delta,
                )

            # Max Retries erreicht → SKIP
            self._consecutive_drops += 1
            logger.info(
                "HPE ⏭️  ΔP=%+.3f — Nach %d Retries keine Verbesserung. Schritt überspringen. (%s)",
                delta,
                self.MAX_RETRIES_PER_STEP,
                step_info,
            )
            return SteerDecision(
                action=SteerAction.SKIP_AND_REVERT,
                reason=f"Nach {self.MAX_RETRIES_PER_STEP} Retries keine Verbesserung — Schritt übersprungen",
                delta_pleasantness=delta,
            )

        # Case 4: Deutliche Verschlechterung → SOFORT SKIP
        self._consecutive_drops += 1
        logger.warning(
            "HPE ⏭️  ΔP=%+.3f — Deutliche Verschlechterung! Schritt überspringen. (%s)",
            delta,
            step_info,
        )

        # Prüfe ob Rollback nötig
        if self._consecutive_drops >= self.MAX_CONSECUTIVE_DROPS:
            best = self._best_snapshot
            if best is not None:
                logger.warning(
                    "HPE ⏪ %d Schritte in Folge verschlechtert. Rollback zu Schritt %d (P=%.3f).",
                    self._consecutive_drops,
                    best.step_id,
                    best.pleasantness,
                )
                return SteerDecision(
                    action=SteerAction.ROLLBACK_TO_BEST,
                    reason=(
                        f"{self._consecutive_drops} Schritte in Folge verschlechtert. "
                        f"Rollback zu Schritt {best.step_id} (P={best.pleasantness:.3f})"
                    ),
                    delta_pleasantness=delta,
                )

        return SteerDecision(
            action=SteerAction.SKIP_AND_REVERT,
            reason=f"Deutliche Verschlechterung (ΔP={delta:+.3f}) — Schritt übersprungen",
            delta_pleasantness=delta,
        )

    def final_evaluate(self, final_audio: np.ndarray) -> SteerDecision:
        """Abschließende Bewertung nach allen Pipeline-Schritten.

        Vergleicht das Endergebnis mit dem besten Zwischenstand.
        Wenn das Endergebnis schlechter ist → Rollback zum Besten.
        """
        final_p = self._measure(final_audio)

        best = self._best_snapshot
        if best is None:
            return SteerDecision(action=SteerAction.STOP_WITH_BEST, reason="Kein Referenz-Snapshot")

        delta_vs_best = final_p - best.pleasantness
        delta_vs_ref = final_p - self._reference_pleasantness

        if delta_vs_best < -self.DELTA_NEUTRAL and best.step_id > 0:
            # Endergebnis schlechter als bester Zwischenstand → Rollback
            logger.warning(
                "HPE ⏪ Final=%.3f schlechter als Best=%.3f (Schritt %d). Rollback zum Optimum.",
                final_p,
                best.pleasantness,
                best.step_id,
            )
            return SteerDecision(
                action=SteerAction.ROLLBACK_TO_BEST,
                reason=(
                    f"Endergebnis (P={final_p:.3f}) schlechter als "
                    f"bester Zwischenstand Schritt {best.step_id} (P={best.pleasantness:.3f})"
                ),
                delta_pleasantness=delta_vs_best,
            )

        # Endergebnis ist ok
        if delta_vs_ref > self.DELTA_NEUTRAL:
            return SteerDecision(
                action=SteerAction.STOP_WITH_BEST,
                reason=f"Pipeline erfolgreich: ΔP={delta_vs_ref:+.3f} vs Original",
                delta_pleasantness=delta_vs_ref,
            )
        else:
            return SteerDecision(
                action=SteerAction.STOP_WITH_BEST,
                reason=f"Pipeline abgeschlossen: ΔP={delta_vs_ref:+.3f} vs Original",
                delta_pleasantness=delta_vs_ref,
            )

    # ── Hilfsmethoden ────────────────────────────────────────────────────

    def get_best_audio(self) -> np.ndarray | None:
        """Gibt das Audio mit der höchsten Pleasantness zurück."""
        if self._best_snapshot is not None:
            return self._best_snapshot.audio.copy()
        return None

    def get_best_pleasantness(self) -> float:
        """Gibt den höchsten erreichten Pleasantness-Score zurück."""
        if self._best_snapshot is not None:
            return self._best_snapshot.pleasantness
        return 0.0

    def get_pleasantness_trajectory(self) -> list[float]:
        """Gibt den Pleasantness-Verlauf über alle Schritte zurück."""
        return list(self._pleasantness_history)

    def reset(self) -> None:
        """Setzt alle Zustände zurück."""
        self._snapshots.clear()
        self._best_snapshot = None
        self._consecutive_drops = 0
        self._retries_this_step = 0
        self._current_pre_audio = None
        self._current_pre_pleasantness = 0.0
        self._total_steps = 0
        self._pleasantness_history.clear()

    # ── Interne Methoden ─────────────────────────────────────────────────

    def _measure(self, audio: np.ndarray) -> float:
        """Misst die psychoakustische Angenehmheit. Robust gegen Fehler."""
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            result = compute_pleasantness(audio, self._sr)
            return float(result.score)
        except Exception as e:
            logger.debug("HPE-Messung fehlgeschlagen: %s", e)
            return 0.5  # Neutraler Fallback

    def _record_snapshot(self, step_id: int, audio: np.ndarray, pleasantness: float, label: str) -> None:
        """Zeichnet einen Snapshot auf und aktualisiert den Bestwert."""
        snap = StepSnapshot(
            step_id=step_id,
            audio=np.asarray(audio, dtype=np.float32).copy(),
            pleasantness=pleasantness,
            label=label,
        )
        self._snapshots.append(snap)

        if self._best_snapshot is None or pleasantness > self._best_snapshot.pleasantness:
            self._best_snapshot = snap


# ═══════════════════════════════════════════════════════════════════════════
# Companion: Ersetzt die alte should_stop_pipeline()
# ═══════════════════════════════════════════════════════════════════════════

# Alias für Rückwärtskompatibilität — die alte Stop-Rule wird durch
# PleasantnessSteering ersetzt. Bestehende Aufrufer von should_stop_pipeline()
# können stattdessen steering.post_step() verwenden.


def create_steering(reference_audio: np.ndarray, sr: int) -> PleasantnessSteering:
    """Factory-Funktion: Erstellt ein konfiguriertes PleasantnessSteering."""
    steering = PleasantnessSteering()
    steering.set_reference(reference_audio, sr)
    return steering
