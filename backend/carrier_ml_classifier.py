"""
backend/carrier_ml_classifier.py — Kompatibilitäts-Shim (Aurik 6.0 → 9.x)
=============================================

Dieses Modul ist ein reiner Re-Export-Shim für
``backend.core.medium_classifier``.

Migrationsanleitung::

    # Alt (Aurik 6.0):
    from backend.carrier_ml_classifier import CarrierMLClassifier
    # Neu (Aurik 10.0.0.x):
    from backend.core.medium_classifier import MediumClassifier, classify_medium

Referenz: §2.1 Aurik-9-Spec, MediumClassifier (§6.1 MaterialType)
"""

from __future__ import annotations

import logging
import warnings as _warnings

import numpy as _np

logger = logging.getLogger(__name__)

from backend.core.medium_classifier import (
    ClassificationResult,
    MediumClassifier,
    classify_medium,
    get_medium_classifier,
)

_warnings.warn(
    "backend.carrier_ml_classifier ist veraltet (Aurik 6.0). "
    "Verwende 'from backend.core.medium_classifier import MediumClassifier, classify_medium'.",
    DeprecationWarning,
    stacklevel=2,
)

# Aurik-6.0-kompatibler Alias
CarrierMLClassifier = MediumClassifier


def classify_carrier_ml(audio: _np.ndarray | None = None, sr: int = 48000, features: dict[str, object] | None = None) -> dict:
    """Classify audio carrier type from actual audio signal.

    SOTA-Update (Rev. 2026-09-04): Nutzt echtes Audio statt Dummy-Signal.
    Delegates to :func:`classify_medium` via real audio and returns a legacy-format dict.

    Parameters
    ----------
    audio:
        Actual audio samples (mono or stereo). If None, falls back to features-only path.
    sr:
        Sample rate of the audio signal.
    features:
        Optional feature dict produced by ``analyze_carrier_forensics`` for explainability.

    Returns
    -------
    dict with keys ``"carrier_ml"``, ``"confidence"``, ``"probas"``, ``"explain"``.
    """
    try:
        if audio is not None and audio.size > 0:
            result = classify_medium(audio, sr)
        else:
            # Fallback: features-only path (legacy compatibility)
            _audio = _np.zeros(int(sr * 0.1), dtype=_np.float32)
            result = classify_medium(_audio, sr)

        _material = getattr(result, "material", None)
        _material_value = getattr(_material, "value", None)
        if _material_value is not None:
            carrier = str(_material_value)
        elif _material is not None:
            carrier = str(_material)
        else:
            carrier = str(result)

        confidence = float(result.confidence) if hasattr(result, "confidence") else 0.5

        # Build explainability from features if available
        feature_count = len(features) if features is not None else 0
        explain_parts = [f"Classified as {carrier}"]
        if feature_count > 0:
            explain_parts.append(f"features={feature_count}")
        if audio is not None and audio.size > 0:
            explain_parts.append(f"samples={audio.size}")

        return {
            "carrier_ml": carrier,
            "confidence": confidence,
            "probas": {},
            "explain": " (".join(explain_parts) + ")" if len(explain_parts) > 1 else explain_parts[0],
        }
    except Exception as exc:
        logger.debug("§V6 Carrier-ML-Klassifizierung fehlgeschlagen — Fallback auf Unbekannt: %s", exc)
        return {
            "carrier_ml": "Unbekannt",
            "confidence": 0.0,
            "probas": {},
            "explain": str(exc),
        }


__all__ = [
    "CarrierMLClassifier",
    "ClassificationResult",
    "MediumClassifier",
    "classify_carrier_ml",
    "classify_medium",
    "get_medium_classifier",
]

# --- Aurik-6.0-Original-Code entfernt (2026-03-11, §9.4 Anti-Parallelwelten) ---
# Originaldatei war: carrier_ml_classifier.py für Aurik 6.0
# Nachfolger: backend.core.medium_classifier (MediumClassifier)
