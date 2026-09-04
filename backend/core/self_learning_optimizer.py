"""
core/self_learning_optimizer.py
Self-Learning Optimizer (SLO)
================================

Lernt vollautomatisch aus jedem Verarbeitungsergebnis, welche
Processing-Variante für welches Material und welchen Defekt-Typ am besten
funktioniert.  Kein Nutzereingriff nötig.

Algorithmus: UCB1 Multi-Armed Bandit pro (Material × Variante)-Kombination.
  - Jede Kombination (Material, Variante) ist ein "Arm".
  - Bei jeder neuen Restaurierung wird der Arm mit dem höchsten UCB1-Score gewählt.
  - Nach Abschluss wird das Ergebnis (quality_delta) zurückgemeldet und die
    Statistik aktualisiert.
  - Persistenz: Zustand wird in `logs/self_learning_state.json` gespeichert
    und beim nächsten Start wiederhergestellt.

Öffentliche API (verwendet von AutonomousRestorationEngine):
  - record_result(material, variant, defect_profile, quality_delta)
  - recommend_variant(material, defect_profile) → variant_name | None

Legacy-API (rückwärtskompatibel, deprecated):
  - update_from_feedback(features, feedback)
  - predict(features)
  - optimize(features)

Author: Aurik Development Team
Version: 2.0.0 "UCB1 Autonomy Edition"
Date: 2026-02-17
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.core.processing_modes import ProcessingMode

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = "logs/self_learning_state.json"


# ---------------------------------------------------------------------------
# Arm-Statistik (UCB1)
# ---------------------------------------------------------------------------


@dataclass
class ArmStats:
    """Statistik für einen (Material, Variante)-Arm."""

    count: float = 0.0
    """Anzahl der bisherigen Pulls (float via Decay)."""
    mean_delta: float = 0.0
    """Gleitender Mittelwert der quality_delta-Werte."""
    sum_sq: float = 0.0
    """Summe der Quadrate (für Varianz-Berechnung)."""
    last_updated: float = field(default_factory=time.time)

    def update(self, delta: float) -> None:
        """Inkrementelles Update mit exponentiell gerichtetem Glättungsfaktor."""
        self.count += 1.0
        alpha = 1.0 / max(0.5, self.count)  # Klassisches inkrementelles Mittel
        self.mean_delta += alpha * (delta - self.mean_delta)
        self.sum_sq += delta**2
        self.last_updated = time.time()

    def apply_decay(self, decay_factor: float) -> None:
        """Wendet einen exponentiellen Decay auf alle Statistiken an.

        Args:
            decay_factor: Faktor im [0, 1]-Bereich. 1.0 = kein Decay.
                         0.99 = 1 % Decay pro Aufruf.
        """
        if decay_factor >= 1.0:
            return
        self.count = max(0.0, self.count * decay_factor)
        self.sum_sq *= decay_factor
        # mean_delta bleibt, aber effektives Gewicht sinkt mit count.
        # Bei count → 0 wird mean_delta unsicherer, was UCB1 via
        # Exploration-Term automatisch berücksichtigt.

    def ucb1_score(self, total_pulls: int, exploration_factor: float = 1.4) -> float:
        """
        UCB1-Score: mean + C * sqrt(ln(N) / n).
        Unter 1000 Pulls: UCB1-Exploration. Über 1000: GP-Standardfehler.
        """
        if self.count < 0.5:
            return float("inf")
        if total_pulls <= 1:
            return self.mean_delta

        exploitation = self.mean_delta

        # §v10.7 Gaussian Process Upgrade: ab 1000 Pulls nutze Standardfehler
        if total_pulls > 1000 and self.count > 5:
            variance = max(0.0, self.sum_sq / max(1.0, self.count) - self.mean_delta**2)
            std_err = math.sqrt(variance / max(1.0, self.count))
            exploration = 2.0 * std_err  # 95% Konfidenzintervall
        else:
            exploration = exploration_factor * math.sqrt(math.log(max(2, total_pulls)) / max(0.5, self.count))

        return exploitation + exploration

    def gp_confidence(self) -> tuple[float, float]:
        """Gaussian Process: (mean, 95%-confidence-half-width).
        Nur aussagekräftig wenn count > 5.
        """
        if self.count < 2:
            return self.mean_delta, float("inf")
        variance = max(0.0, self.sum_sq / max(1.0, self.count) - self.mean_delta**2)
        std_err = math.sqrt(variance / max(1.0, self.count))
        return self.mean_delta, 2.0 * std_err  # 95% CI

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArmStats:
        return cls(**d)


# ---------------------------------------------------------------------------
# Self-Learning Optimizer
# ---------------------------------------------------------------------------


class SelfLearningOptimizer:
    """
    UCB1-basierter Self-Learning Optimizer für Aurik.

    Lernt vollautomatisch, welche (Material, Variante)-Kombination die
    beste Qualitätsverbesserung erzielt.  Session-übergreifende Persistenz
    via JSON.
    """

    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        state_path: str = _DEFAULT_STATE_PATH,
        exploration_factor: float = 1.4,
        decay_factor: float = 0.99,
    ):
        self.mode = mode
        self.state_path = state_path
        self.exploration_factor = exploration_factor
        self.decay_factor = max(0.0, min(1.0, decay_factor))
        self._lock = threading.RLock()

        # Kern-Datenstruktur: {(material_str, variant_str): ArmStats}
        self._arms: dict[tuple[str, str], ArmStats] = {}
        self._total_pulls: int = 0

        # Legacy-API Kompatibilität
        self.learning_rate: float = 0.05
        self.model_params: dict[str, float] = {}
        self._legacy_history: list[dict[str, Any]] = []
        self._max_legacy_history: int = 100  # Cap to prevent unbounded memory growth

        # Zustand laden
        self._load_state()
        logger.info(
            "SelfLearningOptimizer geladen | Modus=%s | Arme=%d | Gesamt-Pulls=%d | Decay=%.3f",
            mode.value,
            len(self._arms),
            self._total_pulls,
            self.decay_factor,
        )

    # ------------------------------------------------------------------
    # Öffentliche API (neu)
    # ------------------------------------------------------------------

    def record_result(
        self,
        material,  # MaterialType oder str
        variant: str,
        defect_profile: Any,
        quality_delta: float,
    ) -> None:
        """
        Registriert das Ergebnis eines Processing-Durchlaufs.

        Wendet vor dem Update den decay_factor auf ALLE Arme an,
        sodass ältere Ergebnisse graduell an Gewicht verlieren.

        Args:
            material:       Erkanntes Material (MaterialType oder str).
            variant:        Name der verwendeten Variante (z. B. 'balanced').
            defect_profile: DefectAnalysisResult (für zukünftige Feature-Extraktion).
            quality_delta:  Qualitätsverbesserung (positiv = besser, negativ = schlechter).
        """
        mat_str = material.value if hasattr(material, "value") else str(material)
        key = (mat_str, variant)

        with self._lock:
            # Apply decay to all existing arms before recording new result
            if self.decay_factor < 1.0:
                for arm in self._arms.values():
                    arm.apply_decay(self.decay_factor)

            if key not in self._arms:
                self._arms[key] = ArmStats()
            self._arms[key].update(quality_delta)
            self._total_pulls += 1

        logger.debug(
            "SLO aufzeichnen | %s × %s | Δ=%.3f | Count=%d | Mean=%.3f",
            mat_str,
            variant,
            quality_delta,
            self._arms[key].count,
            self._arms[key].mean_delta,
        )
        self._save_state()

    def recommend_variant(
        self,
        material,  # MaterialType oder str
        defect_profile: Any,
    ) -> str | None:
        """
        Empfiehlt die Variante mit dem höchsten UCB1-Score für das gegebene Material.

        Returns:
            Varianten-Name oder None wenn noch keine Erfahrungswerte vorliegen.
        """
        mat_str = material.value if hasattr(material, "value") else str(material)

        with self._lock:
            # Alle Arme für dieses Material
            material_arms = {variant: stats for (m, variant), stats in self._arms.items() if m == mat_str}

            if not material_arms:
                logger.debug("SLO: Noch keine Daten für Material %s.", mat_str)
                return None

            # Besten Arm nach UCB1-Score
            best_variant = max(
                material_arms,
                key=lambda v: material_arms[v].ucb1_score(self._total_pulls, self.exploration_factor),
            )
            best_score = material_arms[best_variant].ucb1_score(self._total_pulls, self.exploration_factor)
            logger.debug(
                "SLO Empfehlung: %s × %s (UCB1=%.3f)",
                mat_str,
                best_variant,
                best_score,
            )
            return best_variant

    def recommend_with_confidence(self, material) -> dict:
        """§v10.7 GP: Empfehlung mit 95%-Konfidenzintervall."""
        mat_str = material.value if hasattr(material, "value") else str(material)
        with self._lock:
            arms = {v: s for (m, v), s in self._arms.items() if m == mat_str}
            if not arms:
                return {"variant": None, "pulls": 0, "gp_active": False}
            best = max(arms, key=lambda v: arms[v].ucb1_score(self._total_pulls))
            mean, ci = arms[best].gp_confidence()
            return {
                "variant": best,
                "mean": round(mean, 4),
                "ci95": round(ci, 4),
                "pulls": int(arms[best].count),
                "gp_active": self._total_pulls > 1000,
            }

    def get_statistics(self) -> dict[str, Any]:
        """Gibt vollständige Lernstatistiken zurück (für Audit/Debugging)."""
        with self._lock:
            return {
                "mode": self.mode.value,
                "total_pulls": self._total_pulls,
                "arms": {f"{m}×{v}": stats.to_dict() for (m, v), stats in self._arms.items()},
            }

    def total_pulls(self) -> int:
        """Öffentlicher Accessor: Gesamtzahl der bisherigen Processing-Durchläufe."""
        return self._total_pulls

    def arm_mean_delta(self, material: Any, variant: str) -> float | None:
        """
        Öffentlicher Accessor: Mittleres quality_delta für (material, variant).

        Returns:
            mean_delta oder None wenn keine Daten vorliegen.
        """
        mat_str = material.value if hasattr(material, "value") else str(material)
        key = (mat_str, variant)
        with self._lock:
            if key not in self._arms:
                return None
            return self._arms[key].mean_delta

    def has_data_for(self, material: Any, variant: str) -> bool:
        """True wenn mindestens ein Ergebnis für (material, variant) vorliegt."""
        mat_str = material.value if hasattr(material, "value") else str(material)
        key = (mat_str, variant)
        with self._lock:
            return key in self._arms and self._arms[key].count > 0

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Öffentliche Methode: Persistiert den aktuellen UCB1-State nach
        ``logs/self_learning_state.json``.

        Kann jederzeit aufgerufen werden, z. B. vor Pipeline-Shutdown.
        """
        self._save_state()

    def load_state(self) -> bool:
        """Öffentliche Methode: Lädt den zuvor persistierten State.

        Kann nach einem Pipeline-Neustart aufgerufen werden, um das
        Lernen aus der vorherigen Session fortzusetzen.

        Returns:
            True wenn State erfolgreich geladen wurde, False sonst.
        """
        return self._load_state()

    def _save_state(self) -> None:
        """Speichert den aktuellen Zustand in JSON."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with self._lock:
                state = {
                    "mode": self.mode.value,
                    "total_pulls": self._total_pulls,
                    "decay_factor": self.decay_factor,
                    "exploration_factor": self.exploration_factor,
                    "arms": {f"{m}|{v}": stats.to_dict() for (m, v), stats in self._arms.items()},
                }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as exc:
            logger.warning("SLO: Zustand konnte nicht gespeichert werden: %s", exc)

    def _load_state(self) -> bool:
        """Lädt gespeicherten Zustand aus JSON (falls vorhanden).

        Returns:
            True wenn erfolgreich geladen, False sonst.
        """
        if not os.path.isfile(self.state_path):
            return False
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
            with self._lock:
                self._total_pulls = int(state.get("total_pulls", 0))
                self.decay_factor = float(state.get("decay_factor", self.decay_factor))
                self.exploration_factor = float(state.get("exploration_factor", self.exploration_factor))
                for key_str, arm_dict in state.get("arms", {}).items():
                    parts = key_str.split("|", 1)
                    if len(parts) == 2:
                        self._arms[(parts[0], parts[1])] = ArmStats.from_dict(arm_dict)
            return True
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("SLO: Zustand konnte nicht geladen werden: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Legacy-API (rückwärtskompatibel, deprecated)
    # ------------------------------------------------------------------

    def update_from_feedback(self, features: dict[str, float], feedback: float) -> None:
        """[Deprecated] Nutze record_result() stattdessen."""
        for k, v in features.items():
            if k not in self.model_params:
                self.model_params[k] = 1.0
            grad = (feedback - self.predict(features)) * v
            self.model_params[k] += self.learning_rate * grad
        self._legacy_history.append({"features": features, "feedback": feedback})
        # Cap history to prevent unbounded memory growth
        if len(self._legacy_history) > self._max_legacy_history:
            self._legacy_history = self._legacy_history[-self._max_legacy_history:]

    def predict(self, features: dict[str, float]) -> float:
        """[Deprecated] Lineare Vorhersage basierend auf gelernten Gewichten."""
        return sum(self.model_params.get(k, 1.0) * v for k, v in features.items())

    def optimize(self, features: dict[str, float]) -> dict[str, float]:
        """[Deprecated] Gibt gewichtete Features zurück."""
        return {k: v * self.model_params.get(k, 1.0) for k, v in features.items()}

    def get_history(self) -> list[dict[str, Any]]:
        """[Deprecated] Legacy-History."""
        return self._legacy_history.copy()
