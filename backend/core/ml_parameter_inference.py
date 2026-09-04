"""
ML Parameter Inference Engine für Aurik

Dieses Modul stellt die Grundlage für eine datengetriebene, adaptive Parameterauswahl bereit.
Ziel: Für neue, unbekannte oder schwierige Medientypen automatisch optimale Verarbeitungsparameter
ableiten, um die musikalische Exzellenz weiter zu maximieren.

Features (geplant):
- Automatische Inferenz optimaler Parameter auf Basis von Medieneigenschaften und Qualitätsmetriken
- Unterstützung für supervised/unsupervised/transfer learning
- Schnittstelle zur Integration in den Module Coordinator und die Adaptive Excellence Pipeline
- Logging und Nachvollziehbarkeit der getroffenen Entscheidungen

Status: Skelett/Platzhalter – Implementierung erforderlich
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of MLParameterInferenceEngine.infer_parameters().

    Backward-compatible with dict access via __getitem__ / get().
    """

    strategy: str
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:  # backward compat with dict callers
        return {"strategy": self.strategy, "confidence": self.confidence, "params": self.params}[key]

    def get(self, key: str, default: Any = None) -> Any:  # backward compat
        try:
            return self[key]
        except KeyError as exc:
            logger.debug("§V6 ML-Parameter-Key nicht gefunden — Default-Wert zurückgegeben (Key %s): %s", key, exc)
            return default


_instance: MLParameterInferenceEngine | None = None
_lock = threading.Lock()


def get_ml_inference_engine(model_path: str | None = None) -> MLParameterInferenceEngine:
    """Gibt zurück: or create MLParameterInferenceEngine singleton.

    Args:
        model_path: Path to pre-trained model (only used on first call)

    Returns:
        MLParameterInferenceEngine singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MLParameterInferenceEngine(model_path)
    return _instance


class MLParameterInferenceEngine:
    """
    Engine zur automatischen Inferenz optimaler Verarbeitungsparameter für Aurik.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """
        Initialisiert die ML Parameter Inference Engine.
        Optional: Lädt ein bestehendes Modell.
        """
        self.model: Any | None = None
        self.is_trained = False
        self.last_features: dict[str, Any] | None = None
        self.last_prediction: Any = None
        self.model_path = model_path
        if model_path:
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
                logger.info("Model geladen from %s", model_path)
            except Exception as e:
                logger.warning("konnte nicht laden model from %s: %s", model_path, e)
                self.model = None
                self.is_trained = False
        logger.info("MLParameterInferenceEngine initialisiert")

    def train(self, features: list[dict[str, Any]], targets: list[dict[str, Any]]) -> None:
        """
        Trainiert das ML-Modell auf Basis von Features und Zielparametern.
        :param features: Liste von Feature-Dictionaries
        :param targets: Liste von Zielparametern (Dictionary)
        """
        # Feature-Vektoren extrahieren
        X = [self._dict_to_vector(f) for f in features]
        y = [self._dict_to_vector(t) for t in targets]
        # Für Regression: Zielvektor als np.array
        y = np.array(y)  # type: ignore[assignment]
        self.model = RandomForestRegressor(n_estimators=32)
        self.model.fit(X, y)
        self.is_trained = True
        if self.model_path:
            joblib.dump(self.model, self.model_path)

    def infer_parameters(self, medium_features: dict[str, Any]) -> InferenceResult:
        """
        Leitet optimale Verarbeitungsparameter für gegebene Medieneigenschaften ab.
        :param medium_features: Merkmalsvektor des Mediums
        :return: InferenceResult mit strategy, confidence, params
        """
        self.last_features = medium_features
        if not self.is_trained or not self.model:
            return InferenceResult(strategy="default", confidence=0.0)
        X = np.array([self._dict_to_vector(medium_features)])
        pred = self.model.predict(X)[0]
        self.last_prediction = pred
        return InferenceResult(
            strategy="ml_inferred",
            confidence=0.95,
            params=self._vector_to_dict(pred),
        )

    def explain_last_inference(self) -> str:
        """
        Gibt eine Erklärung für die letzte getroffene Parameterentscheidung zurück.
        """
        if self.last_features is None or self.last_prediction is None:
            return "Keine Inferenz durchgeführt."
        return f"Features: {self.last_features}, Prediction: {self.last_prediction}"

    def _dict_to_vector(self, d: dict[str, Any]) -> list[float]:
        """
        Wandelt ein Dictionary in einen Featurevektor um (vereinfachtes Beispiel).
        """
        # Nur numerische Werte, sortiert nach Schlüssel
        return [float(d[k]) if isinstance(d[k], (int, float)) else 0.0 for k in sorted(d.keys())]

    def _vector_to_dict(self, v: Any) -> dict[str, float]:
        """
        Wandelt einen Vektor zurück in ein Dictionary (vereinfachtes Beispiel).
        """
        # Beispiel: Rückgabe als 'param_0', 'param_1', ...
        return {f"param_{i}": float(val) for i, val in enumerate(v)}
