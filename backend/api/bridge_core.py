"""Aurik Bridge — Core Entry Points & Analysis (§11 Spec 08)
============================================================
Lazy-import wrappers for Enums, Restorer classes, Denker, DefectScanner,
MediumClassifier, Era/Genre classifiers and RestorabilityEstimator.

Public API:
    get_quality_mode, get_medium_type_enum, get_processing_mode_enum
    normalize_user_mode, is_preview_mode
    get_restorer_classes, get_unified_restorer_v3_instance, get_ml_device_manager
    get_aurik_denker_class, get_aurik_denker_instance
    get_defect_scanner, get_audio_file_validator, get_defect_type
    get_medium_classifier_fn, get_era_classifier_fn, get_genre_classifier_fn
    get_restorability_estimator_class, get_medium_detector, get_carrier_forensics_fn

Referenz: Spec 08 §11 Softwareschichten-Architektur.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Mode Normalization
# ---------------------------------------------------------------------------

def get_quality_mode() -> type:
    """Gibt die ``QualityMode``-Enum zurück (lazy import)."""
    from backend.core.performance_guard import QualityMode  # type: ignore[import]

    return QualityMode  # type: ignore[no-any-return]


def get_medium_type_enum() -> type:
    """Gibt die ``MediumType``-Enum zurück (lazy import)."""
    from backend.core.enums import MediumType  # type: ignore[import]

    return MediumType  # type: ignore[no-any-return]


def get_processing_mode_enum() -> type:
    """Gibt die ``ProcessingMode``-Enum zurück (lazy import)."""
    from backend.core.enums import ProcessingMode  # type: ignore[import]

    return ProcessingMode  # type: ignore[no-any-return]


def normalize_user_mode(mode: str | None) -> str:
    """Normalisiert Nutzer-Mode-Aliase auf die kanonischen Release-Modi.

    Canonical Contract:
      - ``"Restoration"``
      - ``"Studio 2026"``

    Unbekannte Eingaben fallen fail-safe auf ``"Restoration"`` zurück.
    """
    raw = str(mode or "Restoration").strip().lower().replace("_", "").replace(" ", "")
    aliases = {
        "restoration": "Restoration",
        "fast": "Restoration",
        "balanced": "Restoration",
        "quality": "Restoration",
        "maximum": "Studio 2026",  # Canonical Contract: maximum ist Legacy-Alias für Studio 2026
        "studio2026": "Studio 2026",
        "studio": "Studio 2026",
        "preview": "Preview",  # §3.5: 30s preview before full restoration
    }
    return aliases.get(raw, "Restoration")


def is_preview_mode(mode: str | None) -> bool:
    """§3.5: Prüft, ob der Modus ein Preview ist (30s Vorschau)."""
    return normalize_user_mode(mode) == "Preview"


# ---------------------------------------------------------------------------
# Core Entry Points (Restorer, Denker, ML-Device)
# ---------------------------------------------------------------------------

def get_restorer_classes() -> tuple[type, type]:
    """Gibt ``(RestorationConfig, UnifiedRestorerV3)`` zurück (lazy import)."""
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore[import]

    return RestorationConfig, UnifiedRestorerV3


def get_unified_restorer_v3_instance():
    """Gibt den UV3-Prozess-Singleton zurück (lazy import über Bridge)."""
    from backend.core.unified_restorer_v3 import get_restorer  # type: ignore[import]

    return get_restorer()


def get_ml_device_manager():
    """Gibt den MLDeviceManager-Singleton zurück (lazy import, §v10.305)."""
    from backend.core.ml_device_manager import get_ml_device_manager as _fn  # type: ignore[import]

    return _fn()


def get_aurik_denker_class() -> type:
    """Gibt ``AurikDenker``-Klasse zurück (lazy import, §2.2 Spec 08).

    Primary entry point for the full 8-stage restoration with carrier analysis,
    DefektDenker, MusikalischerGlobalplan, VERSA MOS scoring and ExzellenzDenker.
    Use this instead of UnifiedRestorerV3 for production pipelines.
    """
    from denker.aurik_denker import AurikDenker  # type: ignore[import]

    return AurikDenker  # type: ignore[no-any-return]


def get_aurik_denker_instance():
    """Gibt den thread-sicheren AurikDenker-Prozess-Singleton zurück (lazy, §2.2 Spec 08).

    Primary production accessor for BatchProcessingThread.
    Ensures Single-Orchestrator Ownership per process (No-Competing-Instances-Protokoll).
    Use ``get_aurik_denker_class()`` only for testing / mocking scenarios.
    """
    from denker.aurik_denker import get_aurik_denker  # type: ignore[import]

    return get_aurik_denker()


# ---------------------------------------------------------------------------
# Analysis & Classification (Defect, Medium, Era/Genre, Restorability)
# ---------------------------------------------------------------------------

def get_defect_scanner() -> type:
    """Gibt die ``DefectScanner``-Klasse zurück (lazy import)."""
    from backend.core.defect_scanner import DefectScanner  # type: ignore[import]

    return DefectScanner  # type: ignore[no-any-return]


def get_audio_file_validator():
    """Gibt den ``AudioFileValidator``-Singleton zurück (lazy import, §10.5).

    Pflicht-Gate vor jedem ``_bg_load``-Thread-Start.  Wirf
    ``AudioLoadError`` (mit ``.message_user`` auf Deutsch) bei ungültiger Datei.
    """
    from backend.core.audio_file_validator import get_audio_file_validator as _get  # type: ignore[import]

    return _get()


def get_defect_type() -> type:
    """Gibt die ``DefectType``-Enum-Klasse zurück (lazy import).

    Wird von ``_defect_analysis_to_display`` und ``_result_scores_to_display``
    im Frontend benötigt, um DefectScanner-Scores zu indizieren.
    """
    from backend.core.defect_scanner import DefectType  # type: ignore[import]

    return DefectType  # type: ignore[no-any-return]


def get_medium_classifier_fn():
    """Gibt einen MediumDetector-basierten Legacy-Kompat-Callable zurück.

    Signatur-kompatibel zu ``classify_medium(mono_audio, sr)`` für Altaufrufer,
    intern jedoch detector-only (kein direkter MediumClassifier-Aufruf).
    """
    import numpy as np  # noqa: E402 — needed for _CompatMediumResult

    from forensics.medium_detector import get_medium_detector as _get_md  # type: ignore[import]

    class _CompatMediumResult:
        def __init__(self, primary_material: str, confidence: float, transfer_chain: list[str], chain_label: str):
            self.material_type = primary_material
            self.material = primary_material
            self.primary_material = primary_material
            self.confidence = float(confidence)
            self.transfer_chain = list(transfer_chain)
            self.chain_label = chain_label

    def _classify_medium_compat(mono_audio: np.ndarray, sr: int) -> _CompatMediumResult:
        _res = _get_md().detect(mono_audio, sr, file_ext="")
        _chain = list(getattr(_res, "transfer_chain", None) or [str(_res.primary_material)])
        _chain_label = str(getattr(_res, "chain_label", " -> ".join(_chain)))
        return _CompatMediumResult(
            primary_material=str(_res.primary_material),
            confidence=float(getattr(_res, "confidence", 0.0)),
            transfer_chain=_chain,
            chain_label=_chain_label,
        )

    return _classify_medium_compat


def get_era_classifier_fn():
    """Gibt ``classify_era``-Funktion zurück (lazy import, §2.4).

    Signatur: ``classify_era(audio: np.ndarray, sr: int) -> EraResult``
    """
    from backend.core.era_classifier import classify_era  # type: ignore[import]

    return classify_era


def get_genre_classifier_fn():
    """Gibt ``classify_genre``-Funktion zurück (lazy import).

    Signatur: ``classify_genre(audio: np.ndarray, sr: int) -> GenreResult``
    """
    from backend.core.genre_classifier import classify_genre  # type: ignore[import]

    return classify_genre


def get_restorability_estimator_class() -> type:
    """Gibt ``RestorabilityEstimator``-Klasse zurück (lazy import, §2.3).

    Verwendung: ``get_restorability_estimator_class()().estimate(audio, sr)``
    """
    from backend.core.restorability_estimator import RestorabilityEstimator  # type: ignore[import]

    return RestorabilityEstimator  # type: ignore[no-any-return]


def get_medium_detector():
    """Gibt the ``MediumDetector`` singleton (lazy import, §6.1 / §11.1) zurück.

    Canonical forensic carrier-chain detector.  Preferred over
    ``get_medium_classifier_fn()`` in all production paths because
    ``MediumDetector.detect()`` supplies the required ``file_ext`` context
    for codec-format digital-file prior adjustment (§6.7b).

    Invariante: ``primary_material`` is always a key from SUPPORTED_MATERIALS
    (cassette → tape, reel_wire → wire_recording, etc. normalised internally).

    Usage::

        md = get_medium_detector()
        result = md.detect(audio, sr, file_ext=Path(file_path).suffix)
        material = result.primary_material  # e.g. "tape", "vinyl"
    """
    try:
        from forensics.medium_detector import get_medium_detector as _get  # type: ignore[import]

        return _get()
    except ImportError as exc:
        logger.debug("§V6 MediumDetector nicht verfügbar — Stub-Recorder aktiviert: %s", exc)
        # Import bridge.py to access the stub recorder (circular-safe at runtime)
        from backend.api.bridge import _record_medium_detector_stub_activation  # noqa: E402, F401

        return _record_medium_detector_stub_activation(exc)


def get_carrier_forensics_fn():
    """Gibt ``analyze_carrier_forensics``-Funktion zurück (lazy import).

    Signatur: ``analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict``
    Rückgabe-Keys: ``"carrier_forensic"`` (str), ``"score"`` (float).

    Intern wird ``MediumDetector.detect`` genutzt (detector-only).
    """
    from forensics.medium_detector import get_medium_detector as _get_md  # type: ignore[import]

    def _analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict:
        result = _get_md().detect(mono, sr, file_ext="")
        return {"carrier_forensic": str(result.primary_material), "score": float(result.confidence)}

    return _analyze_carrier_forensics
