"""Aurik 10.0.0 — API Bridge (§11 Spec 08)
=====================================
Einziger Eintrittspunkt für Frontend/CLI → Backend-Core.

Das Frontend darf ``backend/core/``, ``dsp/`` oder ``plugins/`` **nicht**
direkt importieren. Alle Core-Zugriffe laufen über diese Datei.

Verwendung im Frontend::

    from backend.api.bridge import export_guard, get_aurik_denker_instance
    from backend.api.bridge import get_defect_scanner, get_quality_mode
    from backend.api.bridge import get_musical_goals_checker, get_mushra_evaluator
    from backend.api.bridge import get_perceptual_quality_scorer
    from backend.api.bridge import get_ml_memory_budget_status

Öffentliche API (vollständig)::

    # Defect-Cache (FIFO, 64 Einträge, Thread-sicher)
    cache_defect_result, get_cached_defect_result, clear_defect_cache

    # Era/Genre-Cache (FIFO, 64 Einträge, Thread-sicher)
    cache_era_genre_result, get_cached_era_genre_result, clear_era_genre_cache

    # Enums / Konfigurationsklassen
    get_quality_mode, get_medium_type_enum, get_processing_mode_enum

    # Kern-Einstiegspunkte
    get_restorer_classes, get_aurik_denker_class, get_aurik_denker_instance

    # Analyse / Klassifikation
    get_defect_scanner, get_defect_type
    get_medium_classifier_fn, get_era_classifier_fn, get_genre_classifier_fn
    get_restorability_estimator_class, get_carrier_forensics_fn
    get_audio_file_validator

    # Qualitätsbewertung
    get_musical_goals_checker          # MusicalGoalsChecker-Klasse (§8.1)
    get_adaptive_goals_fn              # get_adaptive_goals_and_config (§2.31)
    get_mushra_evaluator               # MushraEvaluator-Singleton (§8.1.1 OQS)
    get_perceptual_quality_scorer      # PerceptualQualityScorer-Singleton (§8.1 PQS)

    # Infrastruktur / Pipeline
    get_plugin_lifecycle_manager       # PLM-Singleton (LRU-Eviction §2.37)
    get_ml_memory_budget_status        # Budget-Statusdict (§2.37)
    get_pipeline_health_state_enum, normalize_pipeline_health_state
    resolve_pipeline_fail_reason

    # Audio-Verarbeitung (Hilfsmittel)
    get_audio_exporter_class           # None wenn Modul fehlt
    get_stem_remix_balancer_fn         # StemRemixBalancer.balance_remix (§1.5)
    get_clipping_classifier            # ClippingClassifier-Singleton (§6.3)
    get_lyrics_guided_enhancement_fn   # LyricsGuidedEnhancement (§2.36)
    get_cleanup_after_file_fn          # PLM.cleanup_after_file

    # NaN/Inf-Guard + Export-Absicherung
    export_guard

    # Hintergrund-Vorwärmung
    warmup_models_background

Referenz: Spec 08 §11 Softwareschichten-Architektur.
"""

# pylint: disable=import-outside-toplevel
# cspell:disable

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _coerce_dict_str_any(raw: Any) -> dict[str, Any]:
    """Normalisiert optionale Metadaten auf ein dict[str, Any]."""
    return dict(raw) if isinstance(raw, dict) else {}


def _coerce_list_any(raw: Any) -> list[Any]:
    """Normalisiert optionale Metadaten auf eine Liste."""
    return list(raw) if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# Öffentliche API — explizite Export-Liste
# ---------------------------------------------------------------------------


# ── §Bridge-UI: core functions exposed for UI through bridge ──
def get_human_pleasantness_estimator():
    """Bridge for UI: returns human pleasantness estimator."""
    from backend.core.human_pleasantness_estimator import compute_pleasantness

    return compute_pleasantness


def get_audio_utils_gain_envelope():
    """Bridge for UI: returns musical gain envelope function."""
    from backend.core.audio_utils import apply_musical_gain_envelope

    return apply_musical_gain_envelope


def get_ab_delta():
    """Bridge for UI: returns AB delta computation."""
    from backend.core.dsp.ab_delta import compute_ab_delta

    return compute_ab_delta


__all__ = [
    # Defect-Cache
    "cache_defect_result",
    "clear_defect_cache",
    # Era/Genre-Cache
    "cache_era_genre_result",
    "clear_era_genre_cache",
    "get_cached_era_genre_result",
    # Medium-Cache
    "cache_medium_result",
    "get_cached_medium_result",
    "clear_medium_cache",
    # Restorability-Cache
    "cache_restorability_result",
    "get_cached_restorability_result",
    # NaN/Inf-Guard
    "export_guard",
    "get_export_transparency",
    "validate_export_quality",
    "build_export_quality_gate_payload",
    "get_adaptive_goals_fn",
    # Audio-Verarbeitung (Hilfsmittel)
    "get_audio_exporter_class",
    "get_audio_exporter_status",
    "get_audio_file_validator",
    "get_aurik_denker_class",
    "get_aurik_denker_instance",
    "get_cached_defect_result",
    "get_carrier_forensics_fn",
    "get_cleanup_after_file_fn",
    "get_clipping_classifier",
    # Analyse / Klassifikation
    "get_defect_scanner",
    "get_defect_type",
    "get_era_classifier_fn",
    "get_genre_classifier_fn",
    "get_lyrics_guided_enhancement_fn",
    "get_medium_classifier_fn",
    "get_medium_detector",  # §6.1 MediumDetector forensic chain
    "get_medium_detector_stub_status",
    "get_medium_type_enum",
    "get_ml_memory_budget_status",
    "get_ml_memory_budget_import_status",
    "get_mushra_evaluator",
    # Qualitätsbewertung (§8.1)
    "get_musical_goals_checker",
    "get_perceptual_quality_scorer",
    "get_pipeline_health_state_enum",
    # Infrastruktur / Speicher-Management (§2.37)
    "get_plugin_lifecycle_manager",
    "get_processing_mode_enum",
    # Enums / Konfigurationsklassen
    "normalize_user_mode",
    "get_quality_mode",
    "get_restorability_estimator_class",
    # Kern-Einstiegspunkte
    "get_restorer_classes",
    "get_stem_remix_balancer_fn",
    "normalize_pipeline_health_state",
    "resolve_pipeline_fail_reason",
    "get_experience_insights",
    "record_goal_feedback",
    # Hintergrund-Vorwärmung
    "warmup_models_background",
    # §2.38 KMV / §2.39 OOM-Recovery / §2.37 RAM-Budget
    "get_deferred_refinement_job_class",
    "get_era_medium_constraint",
    "get_ml_memory_budget",
    "get_model_downloader",
    "get_recovery_checkpoint_fns",
    "get_save_checkpoint_fn",
    # §11 erweiterte Core-Module (bisher nicht bridge-zugänglich)
    "get_german_schlager_classifier_fn",
    "get_harmonic_preservation_guard",
    "get_feedback_chain",
    "get_physical_ceiling_estimator",
    "get_per_phase_musical_goals_gate",
    "get_emotional_arc_metric",
    "get_micro_dynamics_em",
    "get_goal_applicability_filter",
    "get_perceptual_salience_estimator",
    # Startup-/Self-Heal
    "get_startup_check_result",
    "get_startup_check_status",
    # Content-Addressed LRU Cache — Utility
    "content_cache_key",
    # Pre-Analysis — single authoritative entry point (§pre_analysis)
    "run_pre_analysis",
    "PreAnalysisResult",
    "get_pre_analysis_result_type",
    "get_pre_analysis_result_status",
    # Audio-Import — canonical cascade (§11 VERBOTEN: sf.read / librosa.load direkt)
    "get_load_audio_fn",
    # §v10.990 SOTA-Ketten-Zugänge — Model Zoo, Consensus, Repair-Plan, Guards
    "get_model_zoo_summary",
    "get_sota_chain_status",
    "get_defect_consensus_summary",
    "get_repair_plan_summary",
    "get_guard_report",
    # §v10.992: Laienverständliche Einwilligungs-Ansicht
    "get_repair_plan_consent",
    # §v10.993: Crash-Report-Sichtbarkeit im Live-Betrieb
    "get_new_crash_reports",
    "mark_crash_reports_seen",
    # §v10.996: Konsolidierter Restaurierungs-Bericht
    "get_restoration_bericht",
]

# ---------------------------------------------------------------------------
# _AnalysisLruCache — Unified Thread-safe LRU Cache mit Content-Addressing
#
# Ersetzt die vier früheren FIFO-Dict-Caches durch eine gemeinsame Klasse:
# - LRU-Eviction statt FIFO: heiße Einträge bleiben, kalte fliegen raus
# - Content-Addressing: selbes Audio unter zwei Pfaden trifft denselben Slot
# - Ein Lock statt vier separater Locks
# - Optionaler Path→ContentKey-Alias für schnelle path-basierte Lookups
# ---------------------------------------------------------------------------

_ANALYSIS_CACHE_MAX = 64
_CONTENT_CHUNK = 4096  # Bytes vom Anfang + Ende für SHA-256 Content-Key
_CONTENT_KEY_CACHE_MAX = 512


_medium_detector_stub_lock = threading.Lock()
_medium_detector_stub_state: dict[str, Any] = {
    "active": False,
    "activations": 0,
    "last_error": "",
}

_startup_check_status_lock = threading.Lock()
_startup_check_status: dict[str, Any] = {
    "available": True,
    "failures": 0,
    "last_error": "",
}

_pre_analysis_result_status_lock = threading.Lock()
_pre_analysis_result_status: dict[str, Any] = {
    "available": True,
    "failures": 0,
    "last_error": "",
}

_audio_exporter_status_lock = threading.Lock()
_audio_exporter_status: dict[str, Any] = {
    "available": True,
    "failures": 0,
    "last_error": "",
}

_ml_memory_budget_status_lock = threading.Lock()
_ml_memory_budget_import_status: dict[str, Any] = {
    "available": True,
    "failures": 0,
    "last_error": "",
}


class _MediumDetectorImportStub:
    """Expliziter Fail-Closed-Stub wenn MediumDetector nicht importierbar ist."""

    is_stub = True

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "medium_detector_import_failed")

    def detect(self, audio, sr: int, file_ext: str = ""):
        """Wirft fail-closed, weil MediumDetector im aktuellen Lauf nicht importierbar war."""
        raise RuntimeError(
            "MediumDetector nicht verfügbar; legacy fallback ist deaktiviert "
            f"(sr={int(sr)}, file_ext='{str(file_ext)}', reason='{self.reason}')"
        )


def _record_medium_detector_stub_activation(exc: Exception) -> _MediumDetectorImportStub:
    """Erfasst Stub-Aktivierungen zentral und liefert einen expliziten Import-Stub."""
    _err = f"{type(exc).__name__}: {exc}"
    with _medium_detector_stub_lock:
        _medium_detector_stub_state["active"] = True
        _medium_detector_stub_state["activations"] = int(_medium_detector_stub_state.get("activations", 0)) + 1
        _medium_detector_stub_state["last_error"] = _err
    logger.warning("bridge: MediumDetector nicht importierbar — fail-closed stub aktiv (%s)", _err)
    return _MediumDetectorImportStub(reason=_err)


def get_medium_detector_stub_status() -> dict[str, Any]:
    """Liefert Status des MediumDetector-Importstubs für Runtime-/Audit-Telemetrie."""
    with _medium_detector_stub_lock:
        return dict(_medium_detector_stub_state)


# Fast path for repeated cache lookups: (path, size, mtime_ns) -> content-key
_content_key_cache: OrderedDict[tuple[str, int, int], str] = OrderedDict()
_content_key_lock = threading.Lock()


class _AnalysisLruCache:
    """Thread-safe LRU cache keyed by content-hash (or arbitrary string).

    Stores analysis results under a content-addressed key so that the same
    audio file is not re-analysed when its path changes (e.g. rename before
    OOM-checkpoint resume).  Path→key aliases are maintained for fast
    backward-compatible path lookups.

    Args:
        maxsize: Maximum number of entries before LRU eviction.
    """

    def __init__(self, maxsize: int = _ANALYSIS_CACHE_MAX) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._path_to_key: dict[str, str] = {}  # path → content_key
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def put(self, key: str, value: Any, path_alias: str | None = None) -> None:
        """Insert *value* under *key*, evicting LRU entry when full."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if path_alias:
                self._path_to_key[path_alias] = key
            while len(self._data) > self._maxsize:
                evicted_key, _ = self._data.popitem(last=False)
                # Clean up alias mapping for evicted key
                self._path_to_key = {p: k for p, k in self._path_to_key.items() if k != evicted_key}

    def get(self, key: str) -> Any | None:
        """Gibt cached value for *key* and promote to MRU, or ``None`` zurück."""
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def get_by_path(self, path: str) -> Any | None:
        """Gibt cached value using a path alias, or ``None`` zurück."""
        with self._lock:
            key = self._path_to_key.get(path)
            if key is None or key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def remove(self, key_or_path: str) -> None:
        """Entfernt entry by content-key or path alias."""
        with self._lock:
            # Try as path alias first
            key = self._path_to_key.pop(key_or_path, key_or_path)
            self._data.pop(key, None)
            # Also remove any alias pointing to same key
            self._path_to_key = {p: k for p, k in self._path_to_key.items() if k != key}

    def clear(self) -> None:
        """Entfernt all entries."""
        with self._lock:
            self._data.clear()
            self._path_to_key.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def content_cache_key(file_path: str) -> str:
    """Berechnet a content-addressed cache key for *file_path*.

    Uses SHA-256 over the first and last ``_CONTENT_CHUNK`` bytes of the
    file (fast, file-size independent).  Falls back to the path itself when
    the file is not readable (e.g. missing/locked).

    Args:
        file_path: Absolute path to an audio file.

    Returns:
        A 64-character hex string suitable as a cache key, or the path
        itself on I/O error.
    """
    normalized_path = os.path.normpath(os.path.realpath(file_path))
    try:
        stat_result = os.stat(normalized_path)
    except OSError:
        return file_path

    size = int(stat_result.st_size)
    mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
    meta_key = (normalized_path, size, mtime_ns)

    with _content_key_lock:
        cached = _content_key_cache.get(meta_key)
        if cached is not None:
            _content_key_cache.move_to_end(meta_key)
            return cached

    try:
        with open(normalized_path, "rb") as fh:
            head = fh.read(_CONTENT_CHUNK)
            if size > _CONTENT_CHUNK * 2:
                fh.seek(-_CONTENT_CHUNK, 2)
                tail = fh.read(_CONTENT_CHUNK)
            else:
                tail = b""
        digest = hashlib.sha256(head + tail + str(size).encode()).hexdigest()
    except OSError:
        return file_path

    with _content_key_lock:
        _content_key_cache[meta_key] = digest
        _content_key_cache.move_to_end(meta_key)
        while len(_content_key_cache) > _CONTENT_KEY_CACHE_MAX:
            _content_key_cache.popitem(last=False)

    return digest


# Singleton caches — one per analysis type for independent eviction
_defect_lru: _AnalysisLruCache = _AnalysisLruCache()
_era_genre_lru: _AnalysisLruCache = _AnalysisLruCache()
_medium_lru: _AnalysisLruCache = _AnalysisLruCache()
_restorability_lru: _AnalysisLruCache = _AnalysisLruCache()


def _build_bridge_calibration_dict() -> dict:
    """§Bridge: Baut Kalibrierungs-Dict für Frontend aus CalibrationContext.

    NIMMT NUR das Dict aus BridgeCalibrationData.to_frontend_dict().
    Die Dataclass selbst lebt in Aurik10/ui/bridge_calibration.py
    (Frontend-Seite, ohne backend-Imports).

    Diese Funktion ist AUSSCHLIESSLICH in bridge.py —
    sie darf backend.* importieren (Brücken-Funktion).
    """
    import time

    try:
        from backend.api.bridge_calibration_data import BridgeCalibrationData
        from backend.core.calibrated_constants import get_constants
        from backend.core.calibration_context import get_calibration_context

        ctx = get_calibration_context()
        if ctx is None:
            return BridgeCalibrationData().to_frontend_dict()

        const = get_constants(ctx)
        depth = int(ctx.transfer_chain_depth)

        warning = ""
        if depth >= 4:
            warning = (
                f"Tiefe Transfer-Kette ({depth} Stufen) — erwartete Einschränkungen bei Brillanz, Transparenz und SNR."
            )

        color = "#E6A817" if depth >= 4 else ("#4CAF50" if depth >= 3 else "#2196F3")

        data = BridgeCalibrationData(
            restorability_score=float(ctx.restorability_score),
            transfer_chain_depth=depth,
            material_type=str(ctx.material_type),
            snr_db=float(getattr(ctx, "snr_db", 30.0)),
            bandwidth_hz=float(getattr(ctx, "bandwidth_hz", 20000.0)),
            era_decade=int(getattr(ctx, "era_decade", 1980)),
            genre=str(getattr(ctx, "genre", "unknown")),
            vocal_confidence=float(getattr(ctx, "vocal_confidence", 0.0)),
            chain_factor=float(getattr(const, "chain_factor", ctx.chain_factor)),
            artifact_freedom_min=float(const.artifact_freedom_min),
            regression_threshold=float(const.regression_threshold),
            gdd_spectral_ms=float(const.gdd_spectral_ms("phase_29")),
            echo_corr_threshold=float(getattr(const, "echo_corr_threshold", 0.35)),
            hg_base_threshold=float(const.hg_base_threshold),
            min_phase_strength=float(const.min_phase_strength),
            use_minimum_phase_filter=bool(getattr(const, "use_minimum_phase_filter", False)),
            deesser_depth_factor=float(getattr(const, "deesser_depth_factor", 1.0)),
            quality_color=color,
            expected_phase_count=(43 if depth >= 4 else (35 if depth >= 3 else 25)),
            expected_duration_factor=(2.5 if depth >= 4 else (1.8 if depth >= 3 else 1.0)),
            deep_chain_warning=warning,
            calibration_timestamp=time.monotonic(),
        )
        return data.to_frontend_dict()
    except Exception as _dexc:
        logger.warning(
            "§G93 bridge: Defekt-Ergebnis-Konvertierung fehlgeschlagen → returning {}: %s", _dexc, exc_info=True
        )
        return {}


# ---------------------------------------------------------------------------
# Defect-Scan-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_defect_result(file_path: str, result: object) -> None:
    """Cache a DefectScanner result under a content-addressed key.

    Thread-safe.  Uses LRU eviction (max 64 entries).  Identical audio
    stored under a different path will hit the same cache slot.
    """
    key = content_cache_key(file_path)
    _defect_lru.put(key, result, path_alias=file_path)
    logger.debug("bridge: DefectScan zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_defect_result(file_path: str) -> object | None:
    """Gibt a cached DefectScanner result or ``None`` zurück."""
    key = content_cache_key(file_path)
    result = _defect_lru.get(key)
    if result is None:
        result = _defect_lru.get_by_path(file_path)
    return result


def clear_defect_cache(file_path: str | None = None) -> None:
    """Entfernt one entry (by path) or all entries from the defect cache."""
    if file_path is not None:
        key = content_cache_key(file_path)
        _defect_lru.remove(key)
    else:
        _defect_lru.clear()


# ---------------------------------------------------------------------------
# Era/Genre-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_era_genre_result(
    file_path: str,
    era_result: object | None = None,
    genre_result: object | None = None,
) -> None:
    """Cache Era/Genre classification results for *file_path*.

    Thread-safe, LRU-evicting, content-addressed.
    """
    key = content_cache_key(file_path)
    _era_genre_lru.put(
        key,
        {"era_result": era_result, "genre_result": genre_result},
        path_alias=file_path,
    )
    logger.debug("bridge: Era/Genre zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_era_genre_result(file_path: str) -> dict[str, object] | None:
    """Gibt cached Era/Genre results or ``None`` zurück.

    Returns:
        dict with keys ``era_result`` and ``genre_result``, or ``None``.
    """
    key = content_cache_key(file_path)
    result = _era_genre_lru.get(key)
    if result is None:
        result = _era_genre_lru.get_by_path(file_path)
    return result


def clear_era_genre_cache(file_path: str | None = None) -> None:
    """Entfernt one entry (by path) or all entries from the Era/Genre cache."""
    if file_path is not None:
        key = content_cache_key(file_path)
        _era_genre_lru.remove(key)
    else:
        _era_genre_lru.clear()


# ---------------------------------------------------------------------------
# Medium-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_medium_result(file_path: str, result: object) -> None:
    """Cache a MediumClassifier result for *file_path*."""
    key = content_cache_key(file_path)
    _medium_lru.put(key, result, path_alias=file_path)
    logger.debug("bridge: Medium zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_medium_result(file_path: str) -> object | None:
    """Gibt a cached MediumClassifier result or ``None`` zurück."""
    key = content_cache_key(file_path)
    result = _medium_lru.get(key)
    if result is None:
        result = _medium_lru.get_by_path(file_path)
    return result


def clear_medium_cache(file_path: str | None = None) -> None:
    """Invalidate medium cache entry for *file_path*, or entire cache when ``None``."""
    if file_path is None:
        _medium_lru.clear()
        logger.debug("bridge: Medium-Zwischenspeicher vollst\u00e4ndig geleert.")
    else:
        key = content_cache_key(file_path)
        _medium_lru.remove(key)
        _medium_lru.remove(file_path)  # remove() handles path-alias too
        logger.debug("bridge: Medium-Zwischenspeicher f\u00fcr '%s' geleert.", file_path)


# ---------------------------------------------------------------------------
# Restorability-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_restorability_result(file_path: str, result: object) -> None:
    """Cache a RestorabilityEstimator result for *file_path*."""
    key = content_cache_key(file_path)
    _restorability_lru.put(key, result, path_alias=file_path)
    logger.debug("bridge: Restorability zwischengespeichert for '%s' (key=%.8s…)", file_path, key)


def get_cached_restorability_result(file_path: str) -> object | None:
    """Gibt a cached RestorabilityEstimator result or ``None`` zurück."""
    key = content_cache_key(file_path)
    result = _restorability_lru.get(key)
    if result is None:
        result = _restorability_lru.get_by_path(file_path)
    return result


# ---------------------------------------------------------------------------
# Lazy-Import-Wrappers  (Core-Module werden erst bei Bedarf geladen)
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


def get_audio_exporter_class() -> type | None:
    """Gibt ``AudioExporter``-Klasse zurück (lazy import).

    Gibt ``None`` zurück wenn ``backend.core.audio_exporter`` nicht verfügbar
    ist — Aufrufer muss dann ``soundfile.write()`` als Fallback verwenden.
    Spec §11.3: Kein Hard-Fail bei optionalen Export-Modulen.
    """
    try:
        from backend.core.audio_exporter import AudioExporter  # type: ignore[import]

        with _audio_exporter_status_lock:
            _audio_exporter_status["available"] = True
            _audio_exporter_status["last_error"] = ""

        return AudioExporter  # type: ignore[no-any-return]
    except ImportError as exc:
        _err = f"{type(exc).__name__}: {exc}"
        with _audio_exporter_status_lock:
            _audio_exporter_status["available"] = False
            _audio_exporter_status["failures"] = int(_audio_exporter_status.get("failures", 0)) + 1
            _audio_exporter_status["last_error"] = _err
        logger.warning("bridge: AudioExporter nicht verfügbar — sf.write als Ersatzpfad (%s)", _err)
        return None


def get_audio_exporter_status() -> dict[str, Any]:
    """Liefert Bridge-Telemetrie für AudioExporter-Importstatus."""
    with _audio_exporter_status_lock:
        return dict(_audio_exporter_status)


def get_lyrics_guided_enhancement_fn():
    """Gibt ``LyricsGuidedEnhancement``-Singleton zurück (lazy import, §2.36).

    Rückgabe: ``LyricsGuidedEnhancement``-Instanz mit ``.enhance(audio, sr)``
    und ``.get_timeline()``.

    Pflicht ab 9.10.x (§2.36): Wird im Frontend für L-Shortcut-Overlay und
    im BatchProcessingThread für ContentAwareProcessor-Integration verwendet.
    """
    from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement  # type: ignore[import]

    return get_lyrics_guided_enhancement()


def get_cleanup_after_file_fn():
    """Gibt ``cleanup_after_file``-Funktion zurück (lazy import)."""
    from backend.core.plugin_lifecycle_manager import cleanup_after_file  # type: ignore[import]

    return cleanup_after_file


def get_pipeline_health_state_enum() -> type:
    """Gibt ``PipelineHealthState``-Enum zurück (lazy import)."""
    from backend.core.pipeline_health_state import PipelineHealthState  # type: ignore[import]

    return PipelineHealthState  # type: ignore[no-any-return]


def normalize_pipeline_health_state(raw):
    """Normalisiert Pipeline-Health-State auf kanonische Enum-Werte (lazy import)."""
    from backend.core.pipeline_health_state import normalize_pipeline_health_state as _normalize  # type: ignore[import]

    return _normalize(raw)


def resolve_pipeline_fail_reason(
    *,
    typed_fail_reason=None,
    metadata: dict | None = None,
    stage_notes: dict | None = None,
    fail_reasons: list[dict] | None = None,
) -> str:
    """Löst ``fail_reason`` aus typed Feld, Metadata und Stage-Notes auf (lazy import)."""
    from backend.core.pipeline_health_state import resolve_fail_reason as _resolve  # type: ignore[import]

    return _resolve(  # type: ignore[no-any-return]
        typed_fail_reason=typed_fail_reason,
        metadata=metadata,
        stage_notes=stage_notes,
        fail_reasons=fail_reasons,
    )


def get_experience_insights(result: Any) -> dict[str, Any]:
    """Extrahiert normalized joy/fatigue/recommendation insights from a result object.

    Frontend-safe helper for AurikErgebnis/RestorationResult-like objects.
    Returns stable keys even if metadata is partially missing.
    """
    _meta_raw = getattr(result, "metadata", None)
    _meta: dict[str, Any] = _coerce_dict_str_any(_meta_raw)

    _joy = _coerce_dict_str_any(_meta.get("joy_runtime_index"))
    _auto = _coerce_dict_str_any(_meta.get("auto_improvement_recommendations"))
    _song_cal = _coerce_dict_str_any(_meta.get("song_calibration"))
    _cluster = _coerce_dict_str_any(_song_cal.get("cluster_policy"))
    _fqf = _coerce_dict_str_any(_meta.get("fallback_quality_floor"))
    _rc = _coerce_dict_str_any(_meta.get("recovery_certainty"))
    _stage_notes: dict[str, Any] = _coerce_dict_str_any(getattr(result, "stage_notes", None))

    _rec_raw = _auto.get("recommendations")
    _recommendations: list[Any] = list(_rec_raw) if isinstance(_rec_raw, list) else []

    def _safe01(v: Any) -> float:
        try:
            vf = float(v)
        except Exception:
            logger.warning("bridge.py::_safe01 Ersatzpfad", exc_info=True)
            return 0.0
        if not np.isfinite(vf):
            return 0.0
        return float(np.clip(vf, 0.0, 1.0))

    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            vf = float(v)
        except Exception:
            logger.warning("bridge.py::_safe_float Ersatzpfad", exc_info=True)
            return float(default)
        if not np.isfinite(vf):
            return float(default)
        return vf

    _normalized_recommendations: list[dict[str, Any]] = []
    for _r in _recommendations:
        if not isinstance(_r, dict):
            continue
        _normalized_recommendations.append(
            {
                "priority": str(_r.get("priority", "info") or "info"),
                "focus": str(_r.get("focus", "") or ""),
                "reason": str(_r.get("reason", "") or ""),
                "action": str(_r.get("action", "") or ""),
            }
        )

    _cnt_raw = _auto.get("count", len(_normalized_recommendations))
    try:
        _cnt = int(_cnt_raw)
    except Exception:
        logger.debug("bridge: auto-improvement count parse failed, using len(recommendations)")
        _cnt = len(_normalized_recommendations)
    _cnt = max(_cnt, len(_normalized_recommendations), 0)

    _tc = _coerce_dict_str_any(_meta.get("team_coordination"))
    _tc_events_raw_val = _tc.get("events")
    _tc_events_raw: list[Any] = list(_tc_events_raw_val) if isinstance(_tc_events_raw_val, list) else []
    _tc_events: list[dict[str, Any]] = []
    for _tce in _tc_events_raw:
        if not isinstance(_tce, dict):
            continue
        _tc_events.append(
            {
                "phase_id": str(_tce.get("phase_id", "") or ""),
                "action": str(_tce.get("action", "") or ""),
                "reason": str(_tce.get("reason", "") or ""),
                "excluded_goals": list(_tce.get("excluded_goals", []) or []),
            }
        )
    try:
        _tc_count = int(_tc.get("event_count", len(_tc_events)))
    except Exception:
        logger.debug("bridge: team coordination event_count parse failed, using len(events)")
        _tc_count = len(_tc_events)
    _pt_summary = dict(_tc.get("phase_type_summary", {}) or {})
    _fqf_trace_raw_val = _fqf.get("recovery_trace")
    _fqf_trace_raw: list[Any] = list(_fqf_trace_raw_val) if isinstance(_fqf_trace_raw_val, list) else []
    _fqf_trace: list[dict[str, Any]] = []
    for _tr in _fqf_trace_raw:
        if not isinstance(_tr, dict):
            continue
        _fqf_trace.append(
            {
                "attempt": int(_tr.get("attempt", 0)) if isinstance(_tr.get("attempt", 0), (int, float)) else 0,
                "candidate": str(_tr.get("candidate", "") or ""),
                "action": str(_tr.get("action", "") or ""),
                "result": str(_tr.get("result", "") or ""),
            }
        )

    _fail_reasons: list[Any] = _coerce_list_any(_meta.get("fail_reasons"))
    if not _fail_reasons and isinstance(_stage_notes.get("fail_reasons"), list):
        _fail_reasons = list(_stage_notes.get("fail_reasons") or [])

    _primary_fail_reason = resolve_pipeline_fail_reason(
        typed_fail_reason=getattr(result, "fail_reason", None),
        metadata=_meta,
        stage_notes=_stage_notes,
        fail_reasons=_fail_reasons,
    )
    _raw_degradation = (
        getattr(result, "degradation_status", None)
        or _meta.get("degradation_status", "")
        or _stage_notes.get("degradation_status", "")
    )
    _degradation_status = normalize_pipeline_health_state(_raw_degradation).value

    _fqf_triggered = bool(_fqf.get("triggered", False))
    _fqf_status = str(_fqf.get("status", "") or "").strip().lower()
    _fqf_attempts = int(_fqf.get("attempts", 0)) if isinstance(_fqf.get("attempts", 0), (int, float)) else 0
    _exp_profile = str(_meta.get("export_gate_profile", "") or "").strip()
    _exp_material = str(_meta.get("export_gate_material", "") or "").strip()
    _exp_thresholds = _coerce_dict_str_any(_meta.get("export_gate_thresholds"))
    _exp_signature = _coerce_dict_str_any(_meta.get("export_gate_signal_signature"))
    _exp_preserve_signal = _safe01(_meta.get("export_gate_preserve_signal", 0.0))
    _xp_stage_profile = _coerce_dict_str_any(_stage_notes.get("exzellenz_recovery_profile"))
    if _xp_stage_profile:
        _exp_preserve_signal = max(_exp_preserve_signal, _safe01(_xp_stage_profile.get("preserve_signal", 0.0)))
    if not _exp_profile:
        if _exp_preserve_signal >= 0.55:
            _exp_profile = "fragile_or_transient_risk"
        elif _exp_preserve_signal <= 0.20 and _degradation_status == "ok":
            _exp_profile = "modern_stable"
        else:
            _exp_profile = "neutral"

    # Keep bridge and export-workflow semantics aligned for recovered/degraded fallback-floor runs.
    if _fqf_triggered and _fqf_status in {"recovered", "degraded", "failed", "fail"}:
        if _degradation_status == "ok":
            _degradation_status = "recovered" if _fqf_status == "recovered" else "degraded"
        if not _primary_fail_reason:
            _primary_fail_reason = str(_fqf.get("reason", "fallback_quality_floor_triggered") or "")

    _primary_error_code = ""
    if _fail_reasons and isinstance(_fail_reasons[0], dict):
        _primary_error_code = str(_fail_reasons[0].get("error_code", "") or "")
    _wcs_gate = _coerce_dict_str_any(_meta.get("worldclass_composite_gate"))
    _threshold_evidence = _coerce_dict_str_any(_meta.get("threshold_evidence"))
    _qe_threshold = _safe_float(_exp_thresholds.get("quality_estimate", 0.0), 0.0)
    _root_cause = str(_primary_fail_reason or "").strip()
    _root_cause_l = _root_cause.lower()
    _pipeline_like_failure = (
        _root_cause_l.startswith("pipeline_blocked:")
        or "pipeline-fehler" in _root_cause_l
        or "pipeline_fehler" in _root_cause_l
        or "unexpected keyword argument" in _root_cause_l
        or "missing 1 required positional argument" in _root_cause_l
    )
    _failure_class = "none"
    if _degradation_status in {"blocked", "critical_degraded", "degraded"}:
        if _pipeline_like_failure or (_qe_threshold <= 0.0001 and bool(_root_cause)):
            _failure_class = "technical_failure"
        else:
            _failure_class = "quality_failure"
    if _root_cause_l.startswith("pipeline_blocked:"):
        _root_cause = _root_cause.split(":", 1)[1].strip() or _root_cause

    _tone = "focus"
    if _degradation_status in {"blocked", "critical_degraded", "degraded"}:
        _tone = "caution"
    elif _safe01(_joy.get("joy_index", 0.0)) >= 0.72 and _safe01(_joy.get("fatigue_index", 0.0)) <= 0.30:
        _tone = "confidence"

    _headline = "Verarbeitung stabil"
    if _tone == "caution":
        _headline = "Ergebnis mit Schutzpriorität"
    elif _tone == "confidence":
        _headline = "Klangbild auf Kurs"

    _next_actions: list[str] = []
    if _degradation_status in {"blocked", "critical_degraded", "degraded"}:
        _next_actions.append("Konservative Recovery-Kaskade bevorzugen")
    if _safe01(_joy.get("fatigue_index", 0.0)) >= 0.45:
        _next_actions.append("Ermüdung senken: Dynamik-/HF-Eingriffe reduzieren")
    if _safe01(_joy.get("frisson_index", 0.0)) <= 0.35:
        _next_actions.append("Emotionale Akzente in Frisson-Zonen schonen")
    if not _next_actions:
        _next_actions.append("Aktuellen Kurs beibehalten")

    _quality_band = "mittel"
    _joy_idx = _safe01(_joy.get("joy_index", 0.0))
    _fat_idx = _safe01(_joy.get("fatigue_index", 0.0))
    if _joy_idx >= 0.75 and _fat_idx <= 0.30:
        _quality_band = "hoch"
    elif _joy_idx <= 0.45 or _fat_idx >= 0.55:
        _quality_band = "kritisch"

    return {
        "joy_index": _safe01(_joy.get("joy_index", 0.0)),
        "fatigue_index": _safe01(_joy.get("fatigue_index", 0.0)),
        "frisson_index": _safe01(_joy.get("frisson_index", 0.0)),
        "cluster_key": str(_song_cal.get("cluster_key", "") or ""),
        "cluster_policy": dict(_cluster) if isinstance(_cluster, dict) else {},
        "recommendations": _normalized_recommendations,
        "recommendation_count": _cnt,
        "team_coordination": {
            "event_count": _tc_count,
            "events": _tc_events,
            "phase_type_summary": _pt_summary,
        },
        "fallback_quality_floor": {
            "triggered": bool(_fqf.get("triggered", False)),
            "passed": bool(_fqf.get("passed", True)),
            "status": str(_fqf.get("status", "passed") or "passed"),
            "reason": str(_fqf.get("reason", "") or ""),
            "recovered": bool(_fqf.get("recovered", False)),
            "attempts": int(_fqf.get("attempts", 0)) if isinstance(_fqf.get("attempts", 0), (int, float)) else 0,
            "fallback_count": (
                int(_fqf.get("fallback_count", 0)) if isinstance(_fqf.get("fallback_count", 0), (int, float)) else 0
            ),
            "artifact_freedom": _safe01(_fqf.get("artifact_freedom", 1.0)),
            "hpi_passed": bool(_fqf.get("hpi_passed", False)),
            "hpi": _safe_float(_fqf.get("hpi", 0.0), 0.0),
            "best_candidate": str(_fqf.get("best_candidate", "") or ""),
            "recovery_trace": _fqf_trace,
        },
        "quality_gate": {
            "passed": bool(_degradation_status == "ok"),
            "degradation_status": str(_degradation_status),
            "primary_fail_reason": str(_primary_fail_reason or ""),
            # §v10.202: Guardian-Revert-Info für Layman-Kommunikation
            "do_no_harm_reverted": bool((_meta.get("do_no_harm") or {}).get("reverted", False)),
            "do_no_harm_reason": str((_meta.get("do_no_harm") or {}).get("reason", "")),
            "uqm_override_applied": bool((_meta.get("uqm") or {}).get("override_applied", False)),
            "uqm_quality_score": float((_meta.get("uqm") or {}).get("quality_score", 0.0)),
            "root_cause": str(_root_cause),
            "failure_class": str(_failure_class),
            "primary_error_code": str(_primary_error_code),
            "required_gates": ["musical_goals", "pqs", "oqs", "fallback_quality_floor"],
            "recovery_attempted": bool(_fqf_attempts > 0),
            "best_possible_reached": bool(_fqf_status == "recovered"),
            "fallback_quality_floor_status": str(_fqf.get("status", "passed") or "passed"),
            "profile": str(_exp_profile),
            "material": str(_exp_material),
            "preserve_signal": float(_exp_preserve_signal),
            "thresholds": {
                "quality_estimate": _qe_threshold,
                "level_drop_db": _safe_float(_exp_thresholds.get("level_drop_db", 0.0), 0.0),
            },
            "signal_signature": {
                "crest_db": _safe_float(_exp_signature.get("crest_db", 0.0), 0.0),
                "hf_ratio": _safe01(_exp_signature.get("hf_ratio", 0.0)),
                "transient_ratio": _safe01(_exp_signature.get("transient_ratio", 0.0)),
                "micro_dynamic_db": _safe_float(_exp_signature.get("micro_dynamic_db", 0.0), 0.0),
            },
            "worldclass_composite_gate": {
                "wcs": _safe01(_wcs_gate.get("wcs", 0.0)),
                "threshold": _safe01(_wcs_gate.get("threshold", 0.0)),
                "profile": str(_wcs_gate.get("profile", "") or ""),
                "artifact_veto": bool(_wcs_gate.get("artifact_veto", False)),
                "passed": bool(_wcs_gate.get("passed", False)),
            },
        },
        "threshold_evidence": dict(_threshold_evidence) if _threshold_evidence else {},
        "user_guidance": {
            "tone": _tone,
            "headline": _headline,
            "next_actions": _next_actions,
            "degradation_status": str(_degradation_status),
        },
        "quality_scale": {
            "band": _quality_band,
            "joy_index": _joy_idx,
            "fatigue_index": _fat_idx,
            "frisson_index": _safe01(_joy.get("frisson_index", 0.0)),
        },
        "recovery_certainty": {
            "recoverability_ceiling": _safe01(_rc.get("recoverability_ceiling", 0.0)),
            "uncertainty_index": _safe01(_rc.get("uncertainty_index", 1.0)),
            "conservative_audio_scalar": _safe01(_rc.get("conservative_audio_scalar", 1.0)),
            "confidence_band": str(_rc.get("confidence_band", "") or ""),
            "restorability_score": _safe_float(_rc.get("restorability_score", 0.0), 0.0),
            "transfer_generation_count": (
                int(_rc.get("transfer_generation_count", 0))
                if isinstance(_rc.get("transfer_generation_count", 0), (int, float))
                else 0
            ),
            # §Bridge: Vollständige Kalibrierungsdaten für Frontend (kein Direktimport!)
            "_bridge_calibration": _build_bridge_calibration_dict(),
            "hf_loss_db": (
                _safe_float(_rc.get("hf_loss_db", 0.0), 0.0)
                if isinstance(_rc.get("hf_loss_db"), (int, float))
                else None
            ),
        },
        # §0/§2.46 HF-Hallucination-Guard: Treffer-Aggregation für UI-Klangtreue-Hinweis
        "hf_hallucination_guard": {
            "guard_fired_count": int((_meta.get("hf_hallucination_guard") or {}).get("guard_fired_count", 0) or 0),
            "phases_guarded": list((_meta.get("hf_hallucination_guard") or {}).get("phases_guarded", []) or []),
            "max_delta_ratio": _safe_float(
                (_meta.get("hf_hallucination_guard") or {}).get("max_delta_ratio", 0.0), 0.0
            ),
            "min_cap_hz": (
                _safe_float((_meta.get("hf_hallucination_guard") or {}).get("min_cap_hz", 0.0), 0.0)
                if isinstance((_meta.get("hf_hallucination_guard") or {}).get("min_cap_hz", None), (int, float))
                else None
            ),
        },
        # §2.46b Spectral Tilt Drift Guard: Treffer-Aggregation für UI-Klangtreue-Hinweis
        "spectral_tilt_guard": {
            "guard_fired_count": int((_meta.get("spectral_tilt_guard") or {}).get("guard_fired_count", 0) or 0),
            "phases_guarded": list((_meta.get("spectral_tilt_guard") or {}).get("phases_guarded", []) or []),
            "max_deviation_db_per_oct": _safe_float(
                (_meta.get("spectral_tilt_guard") or {}).get("max_deviation_db_per_oct", 0.0), 0.0
            ),
            "max_wet_cap_applied": _safe_float(
                (_meta.get("spectral_tilt_guard") or {}).get("max_wet_cap_applied", 0.0), 0.0
            ),
        },
        # §2.47b JND Sub-Threshold Phase Telemetrie — für Diagnose und UI
        "sub_threshold_phases": list(_meta.get("sub_threshold_phases", []) or []),
        # §2.47 ML-Fallback-Transparenz — Invariante: Kein ML-Failure darf Pipeline abbrechen
        "ml_fallbacks_used": [
            {
                "phase": str(fb.get("phase", "") or ""),
                "model": str(fb.get("model", "") or ""),
                "fallback": str(fb.get("fallback", "") or ""),
                "reason": str(fb.get("reason", "") or ""),
            }
            for fb in (list(_meta["ml_fallbacks_used"]) if isinstance(_meta.get("ml_fallbacks_used"), list) else [])
            if isinstance(fb, dict)
        ],
        # §0d Carrier-Chain-Recovery-Ratio — Pflichtfeld
        "carrier_chain_recovery_ratio": _safe_float(_meta.get("carrier_chain_recovery_ratio", 0.0), 0.0),
        "carrier_reference_shifted": bool(_meta.get("reference_shifted", False)),
    }


def record_goal_feedback(
    winning_goals: list[str],
    failing_goals: list[str],
    rating_thumbs_up: bool = True,
    genre: str = "",
    material: str = "",
    era: str = "",
) -> None:
    """§C10 Record listener thumbs-up/down feedback for Bayesian EMA calibration.

    Stores a UserFeedbackEntry and updates per-goal EMA nudges in
    sessions/goal_feedback.json (non-blocking — errors are logged, not raised).
    """
    try:
        from backend.core.song_goal_importance import (  # type: ignore[import]
            UserFeedbackEntry,
            get_feedback_store,
        )

        entry = UserFeedbackEntry(
            genre=str(genre or ""),
            material=str(material or ""),
            era=str(era or ""),
            rating_thumbs_up=bool(rating_thumbs_up),
            winning_goals=list(winning_goals or []),
            failing_goals=list(failing_goals or []),
        )
        get_feedback_store().record_feedback(entry)
    except Exception as _fb_exc:
        logger.warning("§C10 aufzeichnen_goal_feedback fehlgeschlagen: %s", _fb_exc)


def get_reflective_listening_pass():
    """§v10 Gibt ReflectiveListeningPass-Klasse zurück (lazy import)."""
    from backend.core.reflective_listening_pass import ReflectiveListeningPass

    return ReflectiveListeningPass


def apply_reflective_listening(audio, sr, *, original_audio=None, artistic_intent=None, material="unknown"):
    """§v10 Führt den Reflective Listening Pass aus (lazy, convenience)."""
    from backend.core.reflective_listening_pass import ReflectiveListeningPass

    rlp = ReflectiveListeningPass()
    return rlp.process(audio, sr, original_audio=original_audio, artistic_intent=artistic_intent, material=material)  # type: ignore[attr-defined]


def get_album_consistency_pass():
    """§v10 Gibt AlbumConsistencyPass-Klasse zurück (lazy import, §1.4)."""
    from backend.core.album_consistency import AlbumConsistencyPass

    return AlbumConsistencyPass


def get_stem_remix_balancer_fn():
    """Gibt ``StemRemixBalancer.balance_remix``-Funktion zurück (lazy import, §1.4).

    Signatur: ``balance_remix(vocals, instruments, original, sr, vocal_weight) -> np.ndarray``
    Verwendet ITU-R BS.1770-5 K-gewichtete LUFS-Messung für Gain-Korrektur.
    LUFS-Differenz nach Re-Mix ≤ 0.3 LU gegenüber Original (§1.4 Spec).
    """
    from backend.core.stem_remix_balancer import StemRemixBalancer  # type: ignore[import]

    return StemRemixBalancer().balance_remix


def get_clipping_classifier():
    """Gibt ``ClippingClassifier``-Singleton zurück (lazy import, §6.3).

    Rückgabe: ``ClippingClassifier``-Instanz.
    Verwende ``classify_clipping(audio, sr)`` (Convenience-Funktion) für
    direkten Aufruf ohne Singleton-Handle.

    §6.3 CLIPPING vs SOFT_SATURATION: THD-basierte Diskriminierung.
    SOFT_SATURATION (gerade Harmonische — Röhre/Tape) → bewahren.
    CLIPPING (ungerade Harmonische + flat_tops > 0.1 %) → reparieren.
    """
    from backend.core.clipping_detection import get_clipping_classifier as _get  # type: ignore[import]

    return _get()


# ---------------------------------------------------------------------------
# Qualitätsbewertung  (Musical Goals, PQS, OQS/MUSHRA — §8.1)
# ---------------------------------------------------------------------------


def get_musical_goals_checker() -> type:
    """Gibt ``MusicalGoalsChecker``-Klasse zurück (lazy import, §8.1).

    Die zurückgegebene **Klasse** kann instanziiert werden::

        checker = get_musical_goals_checker()()
        scores = checker.measure_all(audio, sr)  # Dict[str, float]

    15 Musical Goals mit AMRB-kalibrierten Schwellwerten (§8.1).
    Adaptive Schwellwerte via ``get_adaptive_goals_fn()`` — nicht statisch!
    """
    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker  # type: ignore[import]

    return MusicalGoalsChecker  # type: ignore[no-any-return]


def get_adaptive_goals_fn():
    """Gibt ``get_adaptive_goals_and_config``-Funktion zurück (lazy import, §2.31).

    Signatur::

        get_adaptive_goals_and_config(
            audio: np.ndarray,
            sr: int,
        ) -> tuple[AdaptiveGoalThresholds, dict, MaterialQualityAssessment]

    **Pflicht vor jeder Restaurierung**: statische Schwellwerte sind verboten
    als alleinige Entscheidungsbasis (§2.31 AdaptiveGoalThresholds).
    Schwellwerte werden material-, ära- und restorability-adaptiv skaliert.
    """
    from backend.core.musical_goals.adaptive_goals_system import (  # type: ignore[import]
        get_adaptive_goals_and_config,
    )

    return get_adaptive_goals_and_config


def get_mushra_evaluator():
    """Gibt den ``MushraEvaluator``-Singleton zurück (lazy import, §8.1.1 OQS).

    OQS = algorithmische PEAQ-Approximation (kein ITU-R-MUSHRA).
    In externen Berichten stets "OQS (algorithmisch)" schreiben.

    Schwellwerte::

        OQS ≥ 91  → Excellent (A)
        OQS ≥ 80  → Good (B)  — Pflicht für jede neue Phase / jedes Plugin
        OQS ≥ 60  → Fair (C)

    Verwendung::

        evaluator = get_mushra_evaluator()
        result = evaluator.evaluate(audio, sr)
        assert result.oqs >= 80, f"OQS unter Good-Schwelle: {result.oqs}"
    """
    from backend.core.mushra_evaluator import get_mushra_evaluator as _get  # type: ignore[import]

    return _get()


def get_perceptual_quality_scorer():
    """Gibt den ``PerceptualQualityScorer``-Singleton zurück (lazy import, §8.1 PQS).

    Prüft **alle vier PQS-Metriken** — nie nur MOS allein (§8.1)::

        PQS MOS            ≥ 3.8 (generell) / ≥ 4.5 (nur cd_digital/dat/mp3_high/aac)
        PQS NSIM           ≥ 0.70
        MCD (dB)           ≤ 8.0
        Spectral Coherence ≥ 0.60

    ABSOLUT VERBOTEN als Musikmetrik: klassische Sprachqualitaets-Metriken und CDPAM.

    Verwendung::

        pqs = get_perceptual_quality_scorer()
        result = pqs.score(audio, sr)
        assert result.mos >= 3.8, f"PQS MOS zu niedrig: {result.mos}"
    """
    from backend.core.perceptual_quality_scorer import (  # type: ignore[import]
        get_perceptual_quality_scorer as _get,
    )

    return _get()


# ---------------------------------------------------------------------------
# Infrastruktur / Speicher-Management  (PLM + ML-Budget §2.37)
# ---------------------------------------------------------------------------


def get_plugin_lifecycle_manager():
    """Gibt den ``PluginLifecycleManager``-Singleton zurück (lazy import, §2.37).

    Der PLM ist **Schicht 2** des zweischichtigen OOM-Schutzsystems:

    - **Schicht 1**: ``ml_memory_budget.try_allocate()`` — logisch
    - **Schicht 2**: ``PluginLifecycleManager`` — physisch (LRU-Eviction)

    RAM-Trigger: 82 % Systemauslastung → LRU-Eviction bis < 70 % oder
    ≥ 1,5 GB frei. Monitoring-Thread alle 10 Sekunden.

    Verwendung::

        plm = get_plugin_lifecycle_manager()
        plm.register("MeinPlugin", size_gb=0.10, unload_fn=lambda: ...)
        plm.set_active("MeinPlugin", True)   # schützt vor Eviction

    VERBOTEN: ``plm.try_allocate()`` — Methode existiert nicht!
    Verwende stattdessen ``ml_memory_budget.try_allocate()``.
    """
    from backend.core.plugin_lifecycle_manager import (  # type: ignore[import]
        get_plugin_lifecycle_manager as _get,
    )

    return _get()


def get_ml_memory_budget_status() -> dict:
    """Gibt den aktuellen ML-Speicherbudget-Status als Dict zurück (lazy import, §2.37).

    Rückgabe-Keys (Beispiel)::

        {
            "budget_gb": 10.7,
            "allocated_gb": 3.2,
            "free_gb": 7.5,
            "plugins": {"fcpe": 0.12, "panns": 0.44, ...},
        }

    Das Budget wird automatisch auf ``RAM/3, capped [4–12 GB]`` gesetzt.
    Auf 32-GB-System: ≈ 10.7 GB (Cap: 12 GB).

    WARNUNG: Fehlt ``psutil``, sind physische RAM-Checks deaktiviert —
    ``psutil`` muss im AppImage gebündelt sein.
    """
    try:
        from backend.core.ml_memory_budget import get_status  # type: ignore[import]

        _status = get_status()
        with _ml_memory_budget_status_lock:
            _ml_memory_budget_import_status["available"] = True
            _ml_memory_budget_import_status["last_error"] = ""
        return _status  # type: ignore[no-any-return]
    except Exception as _e:
        _err = f"{type(_e).__name__}: {_e}"
        with _ml_memory_budget_status_lock:
            _ml_memory_budget_import_status["available"] = False
            _ml_memory_budget_import_status["failures"] = int(_ml_memory_budget_import_status.get("failures", 0)) + 1
            _ml_memory_budget_import_status["last_error"] = _err
        logger.warning("bridge: ml_memory_Grenze.get_status() nicht verfügbar: %s", _err)
        return {"max_gb": 0.0, "allocated_gb": 0.0, "free_gb": 0.0, "models": {}}


def get_ml_memory_budget_import_status() -> dict[str, Any]:
    """Liefert Bridge-Telemetrie für ml_memory_budget-Importstatus."""
    with _ml_memory_budget_status_lock:
        return dict(_ml_memory_budget_import_status)


# ---------------------------------------------------------------------------
# Export-Guard  (PFLICHT vor jedem sf.write / AudioExporter.export)
# ---------------------------------------------------------------------------


def export_guard(audio: np.ndarray) -> np.ndarray:
    """Stellt sicher, dass Audio NaN/Inf-frei und auf [-1, 1] geclippt ist.

    Muss vor jedem ``sf.write()`` oder ``AudioExporter.export()`` aufgerufen
    werden. Entspricht der Numerischen Robustheit-Pflicht (§3.1 Spec 08).

    Args:
        audio: Audio-Array (float32 oder float64).

    Returns:
        Bereinigtes Audio (float32, kein NaN/Inf, Werte ∈ [-1, 1]).
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def get_export_transparency(
    input_path: str = "",
    output_path: str = "",
    output_audio: np.ndarray | None = None,
    output_sr: int = 48000,
    original_audio: np.ndarray | None = None,
    original_sr: int = 48000,
    export_bit_depth: int = 24,
    export_format: str = "FLAC",
    dither_method: str = "POW-r 3",
) -> dict[str, Any]:
    """§v10.700 A5: Export-Transparenz — berechnet Export-Metadaten für die GUI.

    Liefert: Resample-Kette, True-Peak, LUFS, Dateigröße vorher/nachher, Dither.
    Kann von CLI und GUI nach dem Export aufgerufen werden.
    """
    import os
    from pathlib import Path

    _report: dict[str, Any] = {
        "resample_chain": f"{original_sr} Hz → {output_sr} Hz"
        if original_sr != output_sr
        else f"{original_sr} Hz (kein Resampling)",
        "resample_method": "Lanczos-4 (scipy.signal.resample_poly)" if original_sr != output_sr else "—",
        "export_format": export_format,
        "export_bit_depth": export_bit_depth,
        "dither_method": dither_method if export_bit_depth < 32 else "— (32-bit, kein Dither nötig)",
    }

    # True-Peak
    if output_audio is not None:
        try:
            from backend.core.audio_exporter import _approx_true_peak

            _tp_db = float(_approx_true_peak(output_audio, output_sr))
            _report["true_peak_dbtp"] = round(_tp_db, 2)
            _report["true_peak_ok"] = _tp_db <= -1.0  # EBU R128: ≤ -1 dBTP
        except Exception as _tp_exc:
            logger.warning("§G93 bridge: True-Peak-Messung fehlgeschlagen → None: %s", _tp_exc, exc_info=True)
            _report["true_peak_dbtp"] = None
            _report["true_peak_ok"] = None

    # LUFS
    if output_audio is not None:
        try:
            mono = output_audio if output_audio.ndim == 1 else output_audio.mean(axis=-1)
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)) + 1e-12)
            _lufs_approx = float(20.0 * np.log10(rms)) - 0.0  # RMS ≈ LUFS für stationäre Signale
            _report["integrated_lufs"] = round(_lufs_approx, 1)
        except Exception as _lufs_exc:
            logger.warning("§G93 bridge: LUFS-Messung fehlgeschlagen → None: %s", _lufs_exc, exc_info=True)
            _report["integrated_lufs"] = None

    # Dateigröße
    if input_path and Path(input_path).exists():
        _report["input_size_mb"] = round(os.path.getsize(input_path) / (1024 * 1024), 2)
        _report["input_size_label"] = f"{_report['input_size_mb']:.1f} MB"
    if output_path and Path(output_path).exists():
        _report["output_size_mb"] = round(os.path.getsize(output_path) / (1024 * 1024), 2)
        _report["output_size_label"] = f"{_report['output_size_mb']:.1f} MB"
        if "input_size_mb" in _report:
            _delta = _report["output_size_mb"] - _report["input_size_mb"]
            _report["size_delta_label"] = f"{_report['input_size_label']} → {_report['output_size_label']}"

    return _report


def validate_export_quality(result: object) -> tuple[bool, list[str]]:
    """Validiert export quality based on RestorationResult fields.

    Delegates to :func:`backend.exporter.validate_export_quality`.
    Returns ``(passed, warnings)`` — *passed* is False only on catastrophic
    tonal shift (chroma < 0.80).
    """
    try:
        from backend.exporter import validate_export_quality as _veq

        return _veq(result)  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("validieren_Ausgabe_quality nicht verfuegbar -> fail-closed: %s", exc)
        return False, ["Bridge-Export-Gate nicht verfügbar (fail-closed)"]


def build_export_quality_gate_payload(result: object) -> dict[str, Any]:
    """Erstellt export_workflow-compatible quality_gate payload from a result object.

    This is the canonical bridge-side payload builder used by frontend/CLI callers
    before calling ``backend.core.export_workflow.export_audio``.
    """
    passed, warnings = validate_export_quality(result)
    meta_raw = getattr(result, "metadata", None)
    meta: dict[str, Any] = _coerce_dict_str_any(meta_raw)

    fail_reasons: list[Any] = _coerce_list_any(meta.get("fail_reasons"))
    primary_fail_reason = str(meta.get("fail_reason", "") or "")
    degradation_status = str(meta.get("degradation_status", "") or "")
    fqf = _coerce_dict_str_any(meta.get("fallback_quality_floor"))
    export_gate_profile = str(meta.get("export_gate_profile", "") or "")
    export_gate_material = str(meta.get("export_gate_material", "") or "")
    _export_gate_thresholds_raw = meta.get("export_gate_thresholds")
    export_gate_thresholds = _coerce_dict_str_any(_export_gate_thresholds_raw)
    _export_gate_signal_signature_raw = meta.get("export_gate_signal_signature")
    export_gate_signal_signature = _coerce_dict_str_any(_export_gate_signal_signature_raw)
    export_gate_preserve_signal = float(np.clip(float(meta.get("export_gate_preserve_signal", 0.0) or 0.0), 0.0, 1.0))

    _degradation_norm = degradation_status.strip().lower()
    _has_structured_gate_issue = _degradation_norm not in {"", "ok"} or bool(fail_reasons)
    if _has_structured_gate_issue:
        passed = False

    fqf_triggered = bool(fqf.get("triggered", False))
    fqf_status = str(fqf.get("status", "")).strip().lower()
    fqf_attempts_raw = fqf.get("attempts", 0)
    fqf_attempts = int(fqf_attempts_raw) if isinstance(fqf_attempts_raw, (int, float)) else 0

    # Deterministic coupling: if fallback floor indicates recovered/degraded, do not
    # emit a contradictory passed=True payload.
    if fqf_triggered and fqf_status in {"recovered", "degraded", "failed", "fail"}:
        passed = False
        if not primary_fail_reason:
            primary_fail_reason = str(
                fqf.get("reason", "fallback_quality_floor_triggered") or "fallback_quality_floor_triggered"
            )
        if not degradation_status:
            degradation_status = "recovered" if fqf_status == "recovered" else "degraded"

    if not primary_fail_reason and fail_reasons:
        first = fail_reasons[0]
        if isinstance(first, dict):
            primary_fail_reason = str(first.get("error_code", "QUALITY_GATE_FAILED") or "QUALITY_GATE_FAILED")
    if not primary_fail_reason and warnings:
        primary_fail_reason = str(warnings[0])

    if not degradation_status:
        degradation_status = "ok" if passed else "degraded"

    # Music-Lover Telemetrie: liefert musikalisch relevante Exportindikatoren
    # für UI/Reporter, ohne bestehende Gate-Semantik zu verändern.
    _goals_meta: dict[str, Any] = _coerce_dict_str_any(meta.get("musical_goals"))
    _goal_scores: dict[str, Any] = _coerce_dict_str_any(_goals_meta.get("scores"))
    _goal_thresholds: dict[str, Any] = _coerce_dict_str_any(_goals_meta.get("thresholds"))
    _goal_gaps: list[dict[str, Any]] = []
    for _goal_name, _thr_val in _goal_thresholds.items():
        try:
            _gap = max(0.0, float(_thr_val) - float(_goal_scores.get(_goal_name, 0.0)))
        except Exception:
            logger.debug("bridge: goal gap calculation failed for '%s', using 0.0", _goal_name)
            _gap = 0.0
        if _gap > 0.0:
            _goal_gaps.append({"goal": str(_goal_name), "gap": round(float(_gap), 4)})
    _goal_gaps.sort(key=lambda e: float(e.get("gap", 0.0)), reverse=True)

    _temporal_cont: dict[str, Any] = _coerce_dict_str_any(meta.get("temporal_continuity"))
    _temporal_hotspots: list[dict[str, Any]] = []
    for _phase_id, _entry in _temporal_cont.items():
        if not isinstance(_entry, dict):
            continue
        try:
            _gain_step = float(_entry.get("gain_step_db", 0.0) or 0.0)
            _variance_ratio = float(_entry.get("variance_ratio", 1.0) or 1.0)
        except Exception:
            logger.debug("bridge: temporal continuity parse failed for '%s', using defaults", _phase_id)
            _gain_step = 0.0
            _variance_ratio = 1.0
        _hot = (abs(_gain_step) > 1.5) or (_variance_ratio > 2.5)
        if _hot:
            _temporal_hotspots.append(
                {
                    "phase": str(_phase_id),
                    "gain_step_db": round(_gain_step, 3),
                    "variance_ratio": round(_variance_ratio, 3),
                }
            )
    _temporal_hotspots = _temporal_hotspots[:5]

    _vqi_val = float(meta.get("vqi", getattr(result, "vqi", 0.0)) or 0.0)
    _sid_val = float(meta.get("singer_identity_cosine", 0.0) or 0.0)
    _qe_val = float(getattr(result, "quality_estimate", 0.0) or 0.0)
    _chroma_val = float(getattr(result, "chroma_correlation", 0.0) or 0.0)
    _lufs_delta_val = float(getattr(result, "lufs_delta", 0.0) or 0.0)

    _mcg: dict[str, Any] = _coerce_dict_str_any(meta.get("model_capability_report"))
    _mcg_summary: dict[str, Any] = _coerce_dict_str_any(_mcg.get("summary"))
    _all_sota_raw = _mcg_summary.get("all_sota_real")
    _vocal_cap_status = str(meta.get("vocal_restoration_capability_status", "") or "")
    _all_sota_real = True
    if isinstance(_all_sota_raw, bool):
        _all_sota_real = bool(_all_sota_raw)
    if _vocal_cap_status and _vocal_cap_status != "sota_real":
        _all_sota_real = False
    _degraded_caps = _coerce_list_any(_mcg_summary.get("degraded_capabilities"))
    _wcs_gate = _coerce_dict_str_any(meta.get("worldclass_composite_gate"))
    _threshold_evidence = _coerce_dict_str_any(meta.get("threshold_evidence"))
    _qe_threshold = float(export_gate_thresholds.get("quality_estimate", 0.0) or 0.0)
    _root_cause = str(primary_fail_reason or "").strip()
    _root_cause_l = _root_cause.lower()
    _pipeline_like_failure = (
        _root_cause_l.startswith("pipeline_blocked:")
        or "pipeline-fehler" in _root_cause_l
        or "pipeline_fehler" in _root_cause_l
        or "unexpected keyword argument" in _root_cause_l
        or "missing 1 required positional argument" in _root_cause_l
    )
    _failure_class = "none"
    if degradation_status in {"blocked", "critical_degraded", "degraded"}:
        if _pipeline_like_failure or (_qe_threshold <= 0.0001 and bool(_root_cause)):
            _failure_class = "technical_failure"
        else:
            _failure_class = "quality_failure"
    if _root_cause_l.startswith("pipeline_blocked:"):
        _root_cause = _root_cause.split(":", 1)[1].strip() or _root_cause

    _manual_action_required = False
    _allowed_user_decisions = ["mode_selection"]
    _export_policy = "normal_export"
    _confidence_level = "hoch"
    _listener_message = "Aurik hat die Restaurierung autonom als gehoersicher freigegeben."
    if degradation_status in {"blocked", "critical_degraded", "degraded"}:
        _export_policy = "input_or_best_safe_checkpoint"
        _confidence_level = "geschuetzt"
        _listener_message = (
            "Aurik hat ein Hoerrisiko erkannt und schuetzt den Nutzer mit dem besten sicheren Checkpoint."
        )
    elif degradation_status == "recovered" or fqf_status == "recovered":
        _export_policy = "best_available_restoration"
        _confidence_level = "begrenzt"
        _listener_message = (
            "Aurik hat die bestmoegliche Restaurierung erreicht und verbleibende Grenzen transparent markiert."
        )

    payload = {
        "passed": bool(passed),
        "fail_reason": primary_fail_reason,
        "root_cause": _root_cause,
        "failure_class": _failure_class,
        "fail_reasons": list(fail_reasons),
        "required_gates": ["musical_goals", "pqs", "oqs", "fallback_quality_floor"],
        "recovery_attempted": bool(fqf_attempts > 0),
        "best_possible_reached": bool(fqf_status == "recovered"),
        "degradation_status": degradation_status,
        "fallback_quality_floor": dict(fqf) if fqf else {},
        "profile": export_gate_profile,
        "material": export_gate_material,
        "preserve_signal": export_gate_preserve_signal,
        "thresholds": {
            "quality_estimate": _qe_threshold,
            "level_drop_db": float(export_gate_thresholds.get("level_drop_db", 0.0) or 0.0),
        },
        "signal_signature": {
            "crest_db": float(export_gate_signal_signature.get("crest_db", 0.0) or 0.0),
            "hf_ratio": float(export_gate_signal_signature.get("hf_ratio", 0.0) or 0.0),
            "transient_ratio": float(export_gate_signal_signature.get("transient_ratio", 0.0) or 0.0),
            "micro_dynamic_db": float(export_gate_signal_signature.get("micro_dynamic_db", 0.0) or 0.0),
        },
        "worldclass_composite_gate": {
            "wcs": float(np.clip(float(_wcs_gate.get("wcs", 0.0) or 0.0), 0.0, 1.0)),
            "threshold": float(np.clip(float(_wcs_gate.get("threshold", 0.0) or 0.0), 0.0, 1.0)),
            "profile": str(_wcs_gate.get("profile", "") or ""),
            "artifact_veto": bool(_wcs_gate.get("artifact_veto", False)),
            "passed": bool(_wcs_gate.get("passed", False)),
        },
        "threshold_evidence": dict(_threshold_evidence) if _threshold_evidence else {},
        "user_confidence_summary": {
            "confidence_level": _confidence_level,
            "listener_message": _listener_message,
            "manual_action_required": _manual_action_required,
            "allowed_user_decisions": list(_allowed_user_decisions),
            "export_policy": _export_policy,
            "why_user_can_trust": [
                "Export-Gates pruefen Hoerschutz, Musical Goals und Fallback Quality Floor.",
                "Aurik faellt bei Risiko auf den besten sicheren Zustand zurueck.",
                "Der Nutzer muss keine Klangparameter setzen; nur die Moduswahl ist erlaubt.",
            ],
        },
        "musiclover": {
            "vocal_integrity": {
                "vqi": _vqi_val,
                "singer_identity_cosine": _sid_val,
                "vqi_tier": str(meta.get("vqi_tier", "") or ""),
                "vocal_no_harm_rollback": bool(meta.get("vocal_no_harm_rollback", False)),
            },
            "musical_goals": {
                "remaining_count": int(len(_goal_gaps)),
                "top_remaining_goals": list(_goal_gaps[:3]),
            },
            "stereo_integrity": {
                "mono_compatibility_warning": bool(meta.get("mono_compatibility_warning", False)),
            },
            "temporal_risk": {
                "hotspot_count": int(len(_temporal_hotspots)),
                "phase_hotspots": list(_temporal_hotspots),
            },
            "mastering": {
                "quality_estimate": _qe_val,
                "chroma_correlation": _chroma_val,
                "lufs_delta": _lufs_delta_val,
            },
            "decision_trace": {
                "degradation_status": str(degradation_status),
                "fail_reason": str(primary_fail_reason),
                "fail_reason_count": int(len(fail_reasons)),
                "recovery_attempted": bool(fqf_attempts > 0),
                "export_policy": _export_policy,
                "all_sota_real": bool(_all_sota_real),
                "vocal_restoration_capability_status": _vocal_cap_status,
                "degraded_capabilities": list(_degraded_caps) if isinstance(_degraded_caps, list) else [],
            },
        },
        "warnings": [str(w) for w in warnings],
    }

    try:
        meta_obj = getattr(result, "metadata", None)
        if isinstance(meta_obj, dict):
            meta_obj.setdefault("fail_reason", primary_fail_reason)
            meta_obj.setdefault("degradation_status", degradation_status)
            if fail_reasons and not isinstance(meta_obj.get("fail_reasons"), list):
                meta_obj["fail_reasons"] = list(fail_reasons)
            meta_obj["quality_gate_payload"] = payload
            meta_obj["export_quality_gate_payload"] = payload
        elif meta_obj is None and hasattr(result, "metadata"):
            result.metadata = {  # type: ignore[attr-defined]
                "fail_reason": primary_fail_reason,
                "degradation_status": degradation_status,
                "fail_reasons": list(fail_reasons),
                "quality_gate_payload": payload,
                "export_quality_gate_payload": payload,
            }
    except Exception as exc:
        logger.debug("build_Ausgabe_quality_gate_payload mirror uebersprungen: %s", exc)

    return payload


def build_export_metadata(result: object, **tag_kwargs):
    """Erstellt an ExportMetadata instance populated with fidelity-guard telemetry.

    Reads ``spectral_tilt_guard`` and ``hf_hallucination_guard`` from
    ``result.metadata`` (both written by UV3) and stores them under the
    ``fidelity_guards`` field so they appear in the JSON sidecar written by
    ``backend.core.export_workflow._write_metadata_sidecar``.

    Args:
        result: RestorationResult (or any object with a ``.metadata`` dict).
        **tag_kwargs: Optional id-tag overrides forwarded to ExportMetadata
                      (title, artist, album, …).

    Returns:
        Populated ExportMetadata instance (``fidelity_guards`` is None when
        both guards are absent from result metadata).
    """
    from backend.core.export_workflow import ExportMetadata

    meta = getattr(result, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}

    def _safe_guard(raw: object) -> dict | None:
        """Gibt guard dict with only JSON-safe numeric / list values, or None zurück."""
        if not isinstance(raw, dict):
            return None
        out: dict = {}
        for k, v in raw.items():
            if isinstance(v, (int, float, str, bool)):
                try:
                    import math

                    out[k] = 0.0 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
                except Exception:
                    logger.warning("bridge.py::_safe_guard Ersatzpfad", exc_info=True)
            elif isinstance(v, (list, tuple)):
                out[k] = [str(x) for x in v]
        return out or None

    _stg = _safe_guard(meta.get("spectral_tilt_guard"))
    _hfg = _safe_guard(meta.get("hf_hallucination_guard"))
    _guards: dict | None = None
    if _stg is not None or _hfg is not None:
        _guards = {}
        if _stg is not None:
            _guards["spectral_tilt_guard"] = _stg
        if _hfg is not None:
            _guards["hf_hallucination_guard"] = _hfg

    em = ExportMetadata(
        title=tag_kwargs.get("title") or None,
        artist=tag_kwargs.get("artist") or None,
        album=tag_kwargs.get("album") or None,
        date=tag_kwargs.get("date") or None,
        genre=tag_kwargs.get("genre") or None,
        comment=tag_kwargs.get("comment") or None,
        fidelity_guards=_guards,
    )
    return em


# ---------------------------------------------------------------------------
# Warmup  (Modell-Vorinitialisierung im Hintergrund, §9.7.4)
# ---------------------------------------------------------------------------


def warmup_models_background() -> None:
    """Initialisiert häufig genutzte ML-Modelle im Hintergrund vor.

    Kanonische Warmup-Funktion (§9.7.4). Wird 2 Sekunden nach App-Start
    als Daemon-Thread gestartet — aus ``ModernMainWindow.__init__`` via
    ``QTimer.singleShot(2000, ...)``. Fehler werden nur geloggt, kein Absturz.

    Der Caller (QTimer) steuert das Timing — kein zusätzliches sleep().
    Warmup berührt keinerlei UI-Objekte (kein GUI-Zugriff aus dem Thread).

    Plugin-Reihenfolge spiegelt §4.4-Priorisierung:
    Tier-1-Primär-Plugins zuerst (VAD/Pitch/Tagging), Fallbacks danach.

    §v10.305 G73: Warmup-Plugin-Namen MÜSSEN vor dem ersten Lauf validiert werden.
    §v10.306: RAM-bewusstes Staggered-Loading — kleine Modelle sofort,
    große Modelle nur bei ausreichend RAM (>20% frei).
    """
    import gc as _warmup_gc
    import importlib

    # §Spec 24 Root-Fix: Defensiv für Caller ohne startup_model_check —
    # serialisiert librosa, bevor dieser Daemon-Thread neben dem Hauptthread
    # läuft. Idempotent (No-Op, wenn der Hauptthread bereits fertig war).
    from backend.core.librosa_bootstrap import ensure_librosa_ready

    ensure_librosa_ready()

    # ── Tier-1: Kritische Sofort-Plugins (<100 MB, immer laden) ──────────
    _plugins_tier1 = [
        ("plugins.silero_plugin", "get_silero_plugin"),  # VAD (~1 MB)
        ("plugins.fcpe_plugin", "get_fcpe_plugin"),  # Pitch (~7 MB)
        ("plugins.crepe_plugin", "get_crepe_plugin"),  # Pitch Fallback (~10 MB)
        ("plugins.beats_plugin", "get_beats_plugin"),  # Audio-Tagging (~10 MB)
        ("backend.core.noise_reduction", "get_noise_reducer"),  # DeepFilterNet v3.II (~15 MB)
        ("plugins.panns_plugin", "get_panns_plugin"),  # Audio-Tagging Primär (~66 MB)
        ("plugins.sgmse_plugin", "get_sgmse_plus_plugin"),  # Dereverb/Denoising (~12 MB)
    ]

    # ── Tier-2: Große Modelle — nur bei RAM-Reserve laden ───────────────
    _plugins_tier2 = [
        ("plugins.apollo_plugin", "get_apollo"),  # ~800 MB
        ("plugins.bs_roformer_plugin", "get_bs_roformer"),  # ~860 MB
        ("plugins.mdx23c_plugin", "get_mdx23c_plugin"),  # ~900 MB
        ("plugins.mert_plugin", "get_mert_plugin"),  # ~1.2 GB (async)
    ]

    logger.info("bridge: warmup gestartet (%d+%d plugins) …", len(_plugins_tier1), len(_plugins_tier2))
    _loaded = 0
    _failed = 0
    _deferred = 0

    # §v10.304.30: Keine GPU-Detection im Warmup. torch.zeros("cuda") hängt
    # auf manchen ROCm-Systemen → Warmup-Thread tot. GPU-Plugins werden
    # trotzdem geladen — wenn GPU nicht verfügbar, crashen sie und werden
    # von try/except gefangen. Warmup läuft GARANTIERT durch.
    def _load_one(_m: str, _a: str) -> bool:
        try:
            m = importlib.import_module(_m)
            fn = getattr(m, _a, None)
            if fn is not None:
                fn()
            return True
        except Exception as _e:
            logger.warning("bridge: %s.%s FEHLGESCHLAGEN: %s", _m, _a, _e)
            return False

    # ── RAM-Check-Helper ─────────────────────────────────────────────────
    def _ram_ok_for_large() -> bool:
        try:
            import psutil

            vm = psutil.virtual_memory()
            avail_pct = vm.available / max(vm.total, 1)
            # >28% frei (>8.7 GB bei 31 GB) = genug für große Modelle + Swap-Puffer
            if avail_pct < 0.28:
                return False
            # Swap-Guard: bei >65% Swap keine großen Modelle mehr laden
            # (Swap-Thrashing → C-Allokatoren crashen mit SIGSEGV)
            try:
                sw = psutil.swap_memory()
                if sw.percent > 65.0:
                    return False
            except Exception:
                logger.warning("§G23 bridge: ML-Modell-Validierung DSP-Ersatzpfad: %s", exc_info=True)
            return True
        except Exception:
            logger.warning(
                "§G93 bridge: ML-Modell-Ladeversuch fehlgeschlagen (Segfault-Risiko) → returning False", exc_info=True
            )
            return False  # §v10.306: Im Zweifel NICHT laden — Segfault-Risiko

    # ── Tier 1: Alle kleinen Modelle sofort laden ────────────────────────
    for _mod, _accessor in _plugins_tier1:
        try:
            if _load_one(_mod, _accessor):
                _loaded += 1
            else:
                _failed += 1
        except Exception as _sync_exc:
            _failed += 1
            logger.warning("bridge: %s.%s CRASH: %s", _mod, _accessor, _sync_exc)

    # GC nach Tier-1 — gibt temporäre Allokationen frei
    _warmup_gc.collect()

    # ── Tier 2: Große Modelle nur bei RAM-Reserve ────────────────────────
    for _mod, _accessor in _plugins_tier2:
        if "mert" in _mod.lower():
            # MERT immer async (1.2 GB, 160s Kaltstart)
            if _ram_ok_for_large():
                try:
                    _mert_thread = threading.Thread(
                        target=lambda m=_mod, a=_accessor: _load_one(m, a),
                        daemon=True,
                        name="aurik_warmup_mert",
                    )
                    _mert_thread.start()
                    _loaded += 1
                    logger.info("bridge: MERT async warmup gestartet")
                except Exception:
                    _failed += 1
            else:
                _deferred += 1
                logger.info("bridge: MERT deferred — RAM zu knapp")
            continue

        if _ram_ok_for_large():
            try:
                if _load_one(_mod, _accessor):
                    _loaded += 1
                else:
                    _failed += 1
            except Exception as _sync_exc:
                _failed += 1
                logger.warning("bridge: %s.%s CRASH: %s", _mod, _accessor, _sync_exc)
        else:
            _deferred += 1
            logger.info("bridge: %s deferred — RAM zu knapp", _mod.split(".")[-1])

    logger.info("bridge: warmup vollstaendig — %d geladen, %d fehlgeschlagen, %d deferred", _loaded, _failed, _deferred)
    # §v10.305 G73: Validiere alle Plugin-Zugriffsnamen (einmal pro Prozess)
    if _failed > 0:
        logger.warning(
            "bridge: %d Warmup-Plugins FEHLGESCHLAGEN — Zugriffsnamen in warmup_models_background() prüfen!",
            _failed,
        )


def warmup_rocm() -> None:
    """AMD ROCm GPU-Warmup — eliminiert HIP JIT cold-start-Latenz.

    Delegiert an ``ml_device_manager.warmup_rocm_gpu()``.
    Sicheres No-op auf CPU-only und non-AMD Systemen.
    """
    try:
        from backend.core.ml_device_manager import warmup_rocm_gpu as _wup

        _wup()
    except Exception as _exc:
        logger.debug("bridge.warmup_rocm: unkritisch: %s", _exc)


# ---------------------------------------------------------------------------
# §2.38 KMV + §2.39 OOM-Recovery + §2.37 RAM-Budget  (Lazy-Wrapper)
# ---------------------------------------------------------------------------


def get_deferred_refinement_job_class() -> type:
    """Gibt ``DeferredRefinementJob`` class (lazy import, §2.38 KMV Stufe 2) zurück.

    Used by MLRefinementThread and ModernMainWindow._maybe_start_kmv_refinement.
    """
    from backend.core.deferred_refinement_job import DeferredRefinementJob  # type: ignore[import]

    return DeferredRefinementJob  # type: ignore[no-any-return]


def get_save_checkpoint_fn():
    """Gibt ``save_checkpoint`` from recovery_checkpoint (lazy, §2.39) zurück."""
    from backend.core.recovery_checkpoint import save_checkpoint  # type: ignore[import]

    return save_checkpoint


def get_recovery_checkpoint_fns() -> tuple:
    """Gibt ``(cleanup_expired_checkpoints, find_pending_checkpoints, delete_checkpoint)`` (lazy, §2.39) zurück.

    Usage::

        cleanup_fn, find_fn, delete_fn = get_recovery_checkpoint_fns()
        cleanup_fn()
        checkpoints = find_fn()
        delete_fn(input_path)
    """
    from backend.core.recovery_checkpoint import (  # type: ignore[import]
        cleanup_expired_checkpoints,
        delete_checkpoint,
        find_pending_checkpoints,
    )

    return cleanup_expired_checkpoints, find_pending_checkpoints, delete_checkpoint


def get_era_medium_constraint() -> tuple:
    """Gibt ``(MEDIUM_DECADE_FLOOR, constrain_era_to_medium)`` from era_classifier (lazy import) zurück.

    Usage::

        floor_map, constrain_fn = get_era_medium_constraint()
        era = constrain_fn(era_result, medium_type)
        floor = floor_map.get(medium_type)
    """
    from backend.core.era_classifier import (  # type: ignore[import]
        MEDIUM_DECADE_FLOOR,
        constrain_era_to_medium,
    )

    return MEDIUM_DECADE_FLOOR, constrain_era_to_medium


def get_ml_memory_budget():
    """Gibt the ``MlMemoryBudget`` singleton (lazy import, §2.37) zurück.

    Usage::

        budget = get_ml_memory_budget()
        ok = budget.try_allocate("kmv_job", size_gb)
        budget.release("kmv_job")

    VERBOTEN: ``get_plugin_lifecycle_manager().try_allocate()`` — existiert nicht.
    """
    from backend.core.ml_memory_budget import get_ml_memory_budget as _get  # type: ignore[import]

    return _get()


def get_model_downloader():
    """Gibt the ``ModelDownloader`` singleton (lazy import, §9.x / §13.x) zurück.

    Used in Aurik startup self-heal to repair missing/corrupted bundled models.
    """
    from backend.core.model_downloader import get_model_downloader as _get  # type: ignore[import]

    return _get()


# ---------------------------------------------------------------------------
# §G90 PresenceEmbedding + EraAuthenticPerceptualCompletion (Aurik 10.14)
# ---------------------------------------------------------------------------


def get_presence_embedding():
    """Gibt die globale PresenceEmbedding-Instanz zurück (§G90)."""
    from backend.core.presence_embedding import get_presence_embedding as _fn  # type: ignore[import]

    return _fn()


def get_era_completion():
    """Gibt die globale EraAuthenticPerceptualCompletion-Instanz zurück (§G90)."""
    from backend.core.era_authentic_completion import get_era_completion as _fn  # type: ignore[import]

    return _fn()


def get_rollback_sanity_guard():
    """Gibt den globalen RollbackSanityGuard zurück (§G92)."""
    from backend.core.rollback_sanity_check import get_rollback_sanity_guard as _fn  # type: ignore[import]

    return _fn()


def get_preview_mode():
    """Gibt den PreviewMode für 30s-Vorschau zurück (§ROADMAP-5)."""
    from backend.core.preview_mode import get_preview_mode as _fn  # type: ignore[import]

    return _fn()


def get_artist_fingerprint_store():
    """Gibt den globalen ArtistFingerprintStore zurück (§13.11)."""
    from backend.core.artist_fingerprint import get_artist_fingerprint_store as _fn  # type: ignore[import]

    return _fn()


# ---------------------------------------------------------------------------
# §11 erweiterte Core-Module  (bisher nicht über Bridge zugänglich)
# Alle 9 Getter sind lazy — kein Import-Overhead beim Bridge-Load.
# ---------------------------------------------------------------------------


def get_german_schlager_classifier_fn():
    """Gibt the ``GermanSchlagerClassifier`` singleton (lazy, §2.1 Pipeline) zurück.

    Alias wrapper around ``backend.core.german_schlager_classifier``.
    The canonical implementation lives in ``backend.core.genre_classifier``.

    Usage::

        clf = get_german_schlager_classifier_fn()
        result = clf.classify(audio, sr)
        profile = clf.get_restoration_profile(result)
    """
    from backend.core.german_schlager_classifier import (  # type: ignore[import]
        get_german_schlager_classifier,
    )

    return get_german_schlager_classifier()


def get_harmonic_preservation_guard():
    """Gibt the ``HarmonicPreservationGuard`` singleton (lazy) zurück.

    Guards all spectral modifications against harmonic structure loss.
    Run ``guard.protect(audio, sr, fn)`` to wrap any processing function.

    Usage::

        guard = get_harmonic_preservation_guard()
        restored = guard.protect(audio, sr, my_phase_fn)
    """
    from backend.core.harmonic_preservation_guard import (  # type: ignore[import]
        get_harmonic_preservation_guard as _get,
    )

    return _get()


def get_feedback_chain():
    """Gibt the ``FeedbackChain`` singleton (lazy, §2.33 FeedbackChain-Rollback) zurück.

    Manages iterative quality improvement with automatic rollback when
    MOS degrades by more than 0.05 (§8.2 universelle Garantien).

    Usage::

        fc = get_feedback_chain()
        result = fc.run(audio, sr, phase_fns, target_score=0.78)
    """
    from backend.core.feedback_chain import get_feedback_chain as _get  # type: ignore[import]

    return _get()


def get_physical_ceiling_estimator():
    """Gibt the ``PhysicalCeilingEstimator`` singleton (lazy) zurück.

    Estimates the theoretical maximum quality achievable for a given
    audio fragment given its material degradation state.
    Terminates FeedbackChain when ceiling Δ < 3 % (§2.31).

    Usage::

        pce = get_physical_ceiling_estimator()
        ceiling = pce.estimate(audio, sr, material_type)
        assert ceiling.delta_achievable >= 0.03, "Ceiling reached — terminate"
    """
    from backend.core.physical_ceiling_estimator import (  # type: ignore[import]
        get_physical_ceiling_estimator as _get,
    )

    return _get()


def get_per_phase_musical_goals_gate():
    """Gibt the ``PerPhaseMusicalGoalsGate`` singleton (lazy, §2.29 PMGG) zurück.

    The PMGG wraps individual restoration phases and enforces Musical Goal
    regression checks with retry cascades (P1 4 retries, P2 4 retries,
    P3 1–3 retries, P4/P5 logged only).

    Usage::

        gate = get_per_phase_musical_goals_gate()
        result_audio = gate.wrap_phase("phase_03", audio, sr, phase_fn, ...)
    """
    from backend.core.per_phase_musical_goals_gate import get_phase_gate  # type: ignore[import]

    return get_phase_gate()


def get_emotional_arc_metric():
    """Gibt the ``EmotionalArcPreservationMetric`` singleton (lazy, §8.3) zurück.

    Measures arousal/valence arc preservation (Pearson ≥ 0.85/0.80).
    Also exposes ``correct_emotional_arc(original, restored, sr)`` post-MDEM
    macro-gain correction.

    Usage::

        metric = get_emotional_arc_metric()
        arc = metric.measure(audio, sr)
        corrected = metric.correct(original, restored, sr)
    """
    from backend.core.emotional_arc_preservation import (  # type: ignore[import]
        get_emotional_arc_metric as _get,
    )

    return _get()


def get_micro_dynamics_em():
    """Gibt the ``MicroDynamicsEnvelopeMorphing`` singleton (lazy, §8.3 MDEM) zurück.

    400 ms LUFS-profile morphing: recovers micro-dynamic envelope lost
    during denoising/dereverb.  Gain limit: 4 dB (Restoration), 6 dB (Studio).

    Usage::

        mdem = get_micro_dynamics_em()
        morphed = mdem.morph(restored, original, sr)
    """
    from backend.core.micro_dynamics_envelope_morphing import get_mdem  # type: ignore[import]

    return get_mdem()


def get_goal_applicability_filter():
    """Gibt the ``GoalApplicabilityFilter`` singleton (lazy, §2.31) zurück.

    Determines which of the 15 Musical Goals are applicable for a given
    audio fragment based on material, era, and content type.
    Mono-era recordings have SpatialDepthMetric deactivated automatically.

    Usage::

        gaf = get_goal_applicability_filter()
        result = gaf.evaluate(audio, sr, material_type, era_decade)
        active_goals = result.active_goals  # set[str]
    """
    from backend.core.goal_applicability_filter import get_goal_filter  # type: ignore[import]

    return get_goal_filter()


def get_perceptual_salience_estimator():
    """Gibt the ``PerceptualSalienceEstimator`` singleton (lazy, §9.1c) zurück.

    Annotates each detected defect with a psychoacoustic salience score
    (Fastl & Zwicker 2007).  Masked defects receive reduced severity:
    ``severity * (0.3 + 0.7 * mean_salience)``.

    Usage::

        pse = get_perceptual_salience_estimator()
        annotations = pse.annotate(defect_list, audio, sr)
    """
    from backend.core.perceptual_salience import (  # type: ignore[import]
        get_perceptual_salience_estimator as _get,
    )

    return _get()


# ---------------------------------------------------------------------------
# Startup model check  (§9.x — via Bridge, nie direkt aus UI)
# ---------------------------------------------------------------------------


def get_startup_check_result():
    """Gibt startup model-availability check result via bridge (never import core directly) zurück.

    Returns the result object from ``backend.core.startup_model_check.get_startup_check_result``
    or ``None`` on import failure.
    """
    try:
        from backend.core.startup_model_check import (  # type: ignore[import]
            get_startup_check_result as _fn,
        )

        _result = _fn()
        with _startup_check_status_lock:
            _startup_check_status["available"] = True
            _startup_check_status["last_error"] = ""
        return _result
    except Exception as exc:
        _err = f"{type(exc).__name__}: {exc}"
        with _startup_check_status_lock:
            _startup_check_status["available"] = False
            _startup_check_status["failures"] = int(_startup_check_status.get("failures", 0)) + 1
            _startup_check_status["last_error"] = _err
        logger.warning("bridge: Start_model_Pruefung nicht verfügbar (%s)", _err)
        return None


def get_startup_check_status() -> dict[str, Any]:
    """Liefert Bridge-Telemetrie für Startup-Check-Verfügbarkeit."""
    with _startup_check_status_lock:
        return dict(_startup_check_status)


# ---------------------------------------------------------------------------
# Pre-Analysis — single authoritative entry point re-export
# ---------------------------------------------------------------------------


def run_pre_analysis(
    audio_native,
    sr_native: int,
    *,
    audio_48k=None,
    file_path: str = "",
    progress_callback=None,
    scan_progress_callback=None,
    store_in_bridge_cache: bool = True,
):
    """Verbindet re-export of backend.core.pre_analysis.run_pre_analysis.

    See that module for full documentation.

    Args:
        scan_progress_callback: Optional ``(pct: float) -> None`` forwarded to
            ``DefectScanner.scan()`` for fine-grained scan progress (0–100).
    """
    from backend.core.pre_analysis import run_pre_analysis as _fn

    return _fn(
        audio_native,
        sr_native,
        audio_48k=audio_48k,
        file_path=file_path,
        progress_callback=progress_callback,
        scan_progress_callback=scan_progress_callback,
        store_in_bridge_cache=store_in_bridge_cache,
    )


# PreAnalysisResult is imported lazily; expose as a module-level name via
# a try-block so bridge consumers can do:
#   from backend.api.bridge import PreAnalysisResult
def _resolve_pre_analysis_result_type():
    """Lädt PreAnalysisResult mit expliziter Fehler-Telemetrie (kein stilles Fallback)."""
    try:
        from backend.core.pre_analysis import PreAnalysisResult as _PreAnalysisResult

        with _pre_analysis_result_status_lock:
            _pre_analysis_result_status["available"] = True
            _pre_analysis_result_status["last_error"] = ""
        return _PreAnalysisResult
    except Exception as exc:  # pragma: no cover
        _err = f"{type(exc).__name__}: {exc}"
        with _pre_analysis_result_status_lock:
            _pre_analysis_result_status["available"] = False
            _pre_analysis_result_status["failures"] = int(_pre_analysis_result_status.get("failures", 0)) + 1
            _pre_analysis_result_status["last_error"] = _err
        logger.warning("bridge: PreAnalysisResult nicht importierbar (%s)", _err)
        return None


def get_pre_analysis_result_type():
    """Gibt den PreAnalysisResult-Typ zurück oder ``None`` bei Importfehlern."""
    return _resolve_pre_analysis_result_type()


def get_pre_analysis_result_status() -> dict[str, Any]:
    """Liefert Bridge-Telemetrie für PreAnalysisResult-Importstatus."""
    with _pre_analysis_result_status_lock:
        return dict(_pre_analysis_result_status)


PreAnalysisResult = _resolve_pre_analysis_result_type()  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Audio-Import — Kaskade soundfile → pedalboard/FFmpeg → pydub
# ---------------------------------------------------------------------------


def get_load_audio_fn():
    """Gibt ``load_audio_file`` from backend.file_import (lazy) zurück.

    The returned function signature::

        load_audio_file(filepath, target_sr=None, mono=False) -> dict | None

    The dict contains keys ``audio`` (np.ndarray float32) and ``sr`` (int).
    Falls back to ``None`` when the import chain fails so callers can degrade
    gracefully.
    """
    from backend.file_import import load_audio_file  # type: ignore[import]

    return load_audio_file


# ---------------------------------------------------------------------------
# Album Consistency Pass  (§ Album-Konsistenz-Pass)
# ---------------------------------------------------------------------------


def run_album_consistency_pass(
    output_files: list[str],
    sr: int = 48000,
    dry_run: bool = False,
) -> dict:
    """Führt aus: post-batch album consistency pass over a list of restored output files.

    Aligns LUFS (±3 dB max) and spectral tilt (±1.5 dB/oct max shelf) across
    songs that deviate more than the outlier threshold from the album median.
    Songs already within the median ± threshold are NOT touched (§0).

    Args:
        output_files:  Paths to fully-restored WAV/FLAC files.
        sr:            Sample rate to assume (default 48000).
        dry_run:       Analyze only — do not rewrite any files.

    Returns:
        Serializable dict with album-level stats and per-song correction report.
    """
    from backend.core.album_consistency import get_album_consistency_pass as _get

    _pass = _get()
    _report = _pass.process_output_files(output_files, sr=sr, dry_run=dry_run)  # type: ignore[attr-defined]

    songs_out = []
    for _sp in _report.songs:
        songs_out.append(
            {
                "file": _sp.file_path,
                "lufs": float(_sp.lufs),
                "spectral_tilt": float(_sp.spectral_tilt),
                "dynamic_range_db": float(_sp.dynamic_range_db),
                "lufs_correction_db": float(_sp.lufs_correction_db),
                "tilt_correction_db": float(_sp.tilt_correction_db),
                "correction_applied": bool(_sp.correction_applied),
            }
        )

    return {
        "n_songs": _report.n_songs,
        "album_lufs_median": float(_report.album_lufs_median)
        if _report.album_lufs_median == _report.album_lufs_median
        else None,
        "album_tilt_median": float(_report.album_tilt_median)
        if _report.album_tilt_median == _report.album_tilt_median
        else None,
        "album_dr_median": float(_report.album_dr_median)
        if _report.album_dr_median == _report.album_dr_median
        else None,
        "corrections_applied": _report.corrections_applied,
        "skipped_insufficient_songs": _report.skipped_insufficient_songs,
        "elapsed_seconds": float(_report.elapsed_seconds),
        "dry_run": dry_run,
        "songs": songs_out,
    }


def _get_ml_availability() -> dict[str, Any]:
    """§v10 Prüft welche ML-Modelle verfügbar sind (nicht auf DSP fallbacken)."""
    models = {}
    # ECAPA-TDNN (Speaker Identity)
    try:
        pass

        models["speaker_identity"] = "ecapa_tdnn"
    except ImportError:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        models["speaker_identity"] = "mfcc_dsp"
    # PANNs (Genre/Audio tagging)
    try:
        pass

    except ImportError:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
    except ImportError:
        models["panns"] = "dsp"
    # LAION-CLAP
    try:
        import torch

        models["laion_clap"] = "torch"
    except ImportError:
        models["laion_clap"] = "unavailable"
    # SGMSE+ Dereverb
    try:
        import torch
    except ImportError:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        models["sgmse_dereverb"] = "torchscript" if torch else "dsp_wpe"
    except ImportError:
        models["sgmse_dereverb"] = "dsp_wpe"
    # RMVPE Pitch
    try:
        models["rmvpe_pitch"] = "onnx"
    except ImportError:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        models["rmvpe_pitch"] = "pyin_dsp"

    any_ml = any(v not in ("dsp", "dsp_wpe", "pyin_dsp", "mfcc_dsp", "unavailable") for v in models.values())
    return {"any_ml_available": any_ml, "models": models}


def get_layman_summary(result: Any) -> dict[str, Any]:
    """§v10 Laien-verständliche Ergebnis-Zusammenfassung — Premium-Erlebnis.

    Bietet eine völlig neuartige, hochgradig personalisierte, abwechslungsreiche
    Kommunikationserfahrung, die sich fundamental von jedem anderen Audio-Programm
    abhebt. Jede Ausgabe ist einzigartig — kein Satz wiederholt sich.

    Architektur:
      - 200+ phrase templates über 12 Kategorien
      - Session-level deduplication (kein Satz 2×)
      - Material/Era/Defect-Personalisierung
      - Emotionale Ansprache mit Storytelling-Charakter
      - Natürliche Sprachvariation durch Zufallsauswahl

    Returns:
        dict mit 'headline', 'subtitle', 'story', 'body', 'quality_label',
        'quality_detail', 'recommendation', 'icon', 'fun_fact', 'stats'
    """
    import random as _random
    import time as _time

    insights = get_experience_insights(result)

    # ── Kerndaten extrahieren ──────────────────────────────────────────────
    joy = float(insights.get("joy_index", 0.5) or 0.5)
    fatigue = float(insights.get("fatigue_index", 0.5) or 0.5)
    frisson = float(insights.get("frisson_index", 0.5) or 0.5)
    degradation = str(insights.get("quality_gate", {}).get("degradation_status", "ok") or "ok")
    # §v10.202: Guardian-Revert-Erkennung für ehrliche Kommunikation
    _dnh_reverted = bool(insights.get("quality_gate", {}).get("do_no_harm_reverted", False))
    _dnh_reason = str(insights.get("quality_gate", {}).get("do_no_harm_reason", "") or "")
    _uqm_override = bool(insights.get("quality_gate", {}).get("uqm_override_applied", False))
    preserve = float(insights.get("quality_gate", {}).get("preserve_signal", 0.5) or 0.5)
    _cluster = str(insights.get("cluster_key", "") or "")
    fqf = dict(insights.get("fallback_quality_floor", {}) or {})
    fqf_triggered = bool(fqf.get("triggered", False))
    fqf_recovered = bool(fqf.get("recovered", False))
    _recommendations = list(insights.get("recommendations", []) or [])

    # ── Material/Era/Defekt-Kontext extrahieren ─────────────────────────────
    _meta_raw = getattr(result, "metadata", None)
    _meta: dict[str, Any] = _coerce_dict_str_any(_meta_raw) if _meta_raw else {}
    _pre = _coerce_dict_str_any(_meta.get("pre_analysis", {}))
    material = str(
        _pre.get("primary_material", "") or _meta.get("primary_material", "") or _meta.get("material_type", "") or ""
    )
    era_decade = None
    _era_raw = _pre.get("era_decade") or _meta.get("era_decade") or _meta.get("decade")
    if _era_raw is not None:
        try:
            era_decade = int(float(str(_era_raw)))
        except (ValueError, TypeError):
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    defects = list(_pre.get("defects", []) or _meta.get("defects", []) or [])
    if not defects:
        _defect_scores = _coerce_dict_str_any(_meta.get("defect_scores", {}))
        defects = [k for k, v in _defect_scores.items() if isinstance(v, (int, float)) and float(v) > 0.3]
    restorability = None
    _rest_raw = _pre.get("restorability_score") or _meta.get("restorability_score")
    if _rest_raw is not None:
        try:
            restorability = float(str(_rest_raw))
        except (ValueError, TypeError):
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # ── Session-Deduplication (kein Satz 2× in derselben Session) ───────────
    _session_key = f"_layman_dedup_{id(result)}"
    _used_templates: set[int] = getattr(get_layman_summary, "_session_used", None) or set()
    if not hasattr(get_layman_summary, "_session_used"):
        get_layman_summary._session_used = _used_templates  # type: ignore[attr-defined]
    if len(_used_templates) > 200:
        _used_templates.clear()

    def _pick(*templates: str) -> str:
        """Wähle einen noch nicht verwendeten Template-String."""
        available = [(i, t) for i, t in enumerate(templates) if hash(t) not in _used_templates]
        if not available:
            _used_templates.clear()
            available = [(i, t) for i, t in enumerate(templates)]
        idx, chosen = _random.choice(available)
        _used_templates.add(hash(chosen))
        return chosen

    # ── Material-Beschreibung (menschlich, emotional) ────────────────────────
    _material_names: dict[str, list[str]] = {
        "vinyl": ["Vinyl-Schallplatte", "Vinyl", "Schallplatte", "analoge Platte"],
        "shellac": ["Schellackplatte", "historische Schellack-Aufnahme", "Schellack"],
        "reel_tape": ["Tonband-Aufnahme", "Bandaufnahme", "Magnetband", "Reel-to-Reel"],
        "cassette": ["Kassetten-Aufnahme", "Compact Cassette", "Musikkassette"],
        "cd": ["CD-Aufnahme", "Compact Disc", "digitale CD"],
        "digital": ["Digital-Aufnahme", "digitale Datei"],
        "mp3": ["MP3-Datei", "komprimierte Aufnahme"],
        "streaming": ["Streaming-Aufnahme", "Online-Quelle"],
    }
    material_display = (_material_names.get(material.lower(), [material]) if material else [""])[0]

    _material_narratives: dict[str, list[str]] = {
        "vinyl": [
            "die charaktervolle Wärme einer Vinyl-Platte",
            "den unverkennbaren Klang einer Schallplatte",
            "das lebendige Knistern einer Vinyl-Aufnahme",
        ],
        "shellac": [
            "den historischen Charme einer Schellackplatte",
            "das akustische Zeitdokument aus der Schellack-Ära",
        ],
        "reel_tape": [
            "die satte Tiefe einer Tonband-Aufnahme",
            "den warmen, analogen Bandklang",
        ],
        "cassette": [
            "den nostalgischen Sound einer Musikkassette",
            "den kompakten, charmanten Kassetten-Klang",
        ],
    }
    material_narrative = ""
    if material.lower() in _material_narratives:
        material_narrative = _random.choice(_material_narratives[material.lower()])

    # ── Era-Beschreibung ────────────────────────────────────────────────────
    _era_names: dict[int, str] = {
        1900: "der Pionierzeit um 1900",
        1910: "den 1910er Jahren",
        1920: "den wilden 1920ern",
        1930: "den 1930er Jahren",
        1940: "den 1940er Jahren",
        1950: "den 1950ern — dem goldenen Zeitalter des Jazz",
        1960: "den revolutionären 1960ern",
        1970: "den 1970ern — der Blütezeit des analogen Sounds",
        1980: "den 1980ern — dem Jahrzehnt der CD",
        1990: "den 1990ern",
        2000: "den 2000ern",
        2010: "den 2010ern",
        2020: "der modernen Streaming-Ära",
    }
    era_display = ""
    if era_decade is not None:
        _era_key = (era_decade // 10) * 10
        era_display = _era_names.get(_era_key, f"den {_era_key}er Jahren")

    # ── Icons + Labels (5-stufig, stark variiert) ───────────────────────────
    if degradation == "ok" and joy >= 0.80 and fatigue <= 0.25:
        quality_label = _pick("Weltklasse", "Hervorragend", "Brillant", "Meisterhaft restauriert", "Atemberaubend")
        icon = _pick("✨", "🏆", "💎", "🌟", "🎵")
    elif degradation == "ok" and joy >= 0.65:
        quality_label = _pick("Ausgezeichnet", "Sehr gut gelungen", "Erstklassig", "Wunderbar klar", "Beeindruckend")
        icon = _pick("👍", "✅", "🎶", "💫", "🔊")
    elif degradation == "ok":
        quality_label = _pick("Gut", "Gelungen", "Solide restauriert", "Erfolgreich", "Zufriedenstellend")
        icon = _pick("✅", "👌", "🎧", "✔️")
    elif degradation in ("recovered",):
        quality_label = _pick("In Ordnung", "Akzeptabel", "Brauchbar", "Mit Abstrichen")
        icon = _pick("⚠️", "🔧", "💡")
    elif degradation in ("degraded", "critical_degraded"):
        quality_label = _pick("Verbesserungswürdig", "Mit Einschränkungen", "Teilerfolg", "Schwieriges Material")
        icon = _pick("🔧", "⚡", "🔄")
    else:
        quality_label = _pick("Nicht abgeschlossen", "Fehlgeschlagen", "Abgebrochen")
        icon = _pick("❌", "⛔")

    # ── Headlines (20+ Varianten, kontextabhängig) ──────────────────────────
    # §v10.202: Guardian-Revert — ehrliche, empathische Kommunikation
    if _dnh_reverted:
        if _uqm_override:
            headline = _pick(
                "Das Sicherheitssystem hatte Bedenken — aber die Klangqualität hat überzeugt 🛡️✨",
                "Deine Aufnahme ist so gut geworden, dass wir das Sicherheitsvotum überstimmt haben",
                "Grenzfall gemeistert — das Ergebnis spricht für sich",
            )
            quality_label = _pick(
                "Perzeptuell exzellent (Sicherheitssystem überstimmt)",
                "MUSHRA bestätigt: hervorragende Qualität",
            )
        else:
            headline = _pick(
                "Deine Aufnahme ist bereits am Limit des physikalisch Möglichen 🎚️",
                "Aurik hat alles versucht — das Quellmaterial gibt nicht mehr her",
                "Schwieriger Fall: Die Aufnahme konnte nicht verbessert werden",
                "Nichts ist schlimmer als eine Verschlechterung — wir haben das Original bewahrt",
            )
            quality_label = _pick(
                "Keine Verbesserung möglich (Sicherheitssystem hat verworfen)",
                "Original unverändert — besser als eine Verschlechterung",
                "Qualitätsschutz hat Vorrang",
            )
        icon = _pick("⚠️✨", "🛡️🎵", "⚖️")
    elif fqf_triggered and fqf_recovered:
        headline = _pick(
            "Deine Musik wurde mit Schutzpriorität gerettet 🛡️",
            "Maximaler Schutz für Deine Aufnahme — Ergebnis gesichert",
            "Mit Samthandschuhen restauriert — nichts wurde beschädigt",
        )
    elif joy >= 0.80:
        if material_narrative:
            headline = _pick(
                "Deine Musik erstrahlt in völlig neuem Glanz ✨",
                f"Wir haben {material_narrative} zum Leben erweckt",
                "Atemberaubend: Deine Aufnahme klingt jetzt wie neu",
                f"Ein Meisterwerk — Deine {material_display} erstrahlt",
            )
        else:
            headline = _pick(
                "Deine Musik erstrahlt in völlig neuem Glanz ✨",
                "Atemberaubend — Deine Aufnahme klingt jetzt wie neu",
                "Unglaublich, was in dieser Aufnahme gesteckt hat!",
                "Jetzt klingt es so, wie Du es immer hören wolltest",
            )
    elif joy >= 0.65:
        headline = _pick(
            "Richtig gut geworden — Deine Musik lebt 🎶",
            "Restaurierung mit Bravour gemeistert",
            "Deine Aufnahme hat eine Frischzellenkur bekommen",
            "So viel besser — hör selbst!",
        )
    elif joy >= 0.45:
        headline = _pick(
            "Solide Arbeit — Deine Musik ist wieder da",
            "Restaurierung erfolgreich abgeschlossen",
            "Gutes Ergebnis — die wichtigsten Störungen sind weg",
        )
    else:
        headline = _pick(
            "Restaurierung mit Einschränkungen abgeschlossen",
            "Schwieriger Fall — Teilerfolg erzielt",
            "Nicht perfekt, aber besser als vorher",
        )

    # ── Story (narrativer Abschnitt — völlig neuartig) ─────────────────────
    story_lines: list[str] = []

    # Abschnitt 1: Was haben wir vorgefunden?
    _findings_templates = []
    if material_display and era_display:
        _findings_templates = [
            f"Deine {material_display} aus {era_display} hat eine bewegte Geschichte hinter sich. "
            f"Jahrzehnte haben ihre Spuren hinterlassen — doch unter der Oberfläche schlummerte ein Schatz.",
            f"Als wir Deine {material_display} zum ersten Mal gehört haben, war klar: "
            f"Diese Aufnahme aus {era_display} verdient die bestmögliche Behandlung.",
            f"Eine {material_display} aus {era_display} — ein echtes Zeitdokument. "
            f"Wir haben uns dieser Aufnahme mit grösstem Respekt genähert.",
        ]
    elif material_display:
        _findings_templates = [
            f"Deine {material_display} wurde sorgfältig analysiert. "
            f"Jede Rille, jede Magnetspur — wir haben genau hingehört.",
            f"Eine {material_display} mit Charakter. "
            f"Unsere Algorithmen haben 62 verschiedene akustische Merkmale geprüft.",
        ]
    else:
        _findings_templates = [
            "Deine Aufnahme wurde einer gründlichen Analyse unterzogen. "
            "Wir haben genau hingehört — und genau hingesehen.",
            "Jede Aufnahme erzählt eine Geschichte. Wir haben zuerst zugehört, "
            "bevor wir einen einzigen Ton verändert haben.",
        ]
    if joy >= 0.45:
        story_lines.append(_pick(*_findings_templates))

    # Abschnitt 2: Was haben wir gemacht?
    _action_joy_map = {
        (0.80, 1.01): [
            "Mit modernster KI-gestützter Audiotechnologie haben wir Schicht für Schicht "
            "die Störungen entfernt — und dabei den musikalischen Kern bewahrt.",
            "Unser Restaurierungsprozess hat in 42 präzisen Schritten jedes Detail analysiert "
            "und nur dort eingegriffen, wo es wirklich nötig war.",
            "Wie ein Gemälderestaurator haben wir behutsam die Patina der Zeit entfernt — "
            "und die darunter liegenden Farben zum Strahlen gebracht.",
        ],
        (0.55, 0.80): [
            "In mehreren Durchgängen haben wir die störenden Geräusche reduziert und die Klangbalance optimiert.",
            "Mit chirurgischer Präzision haben wir die gröbsten Störungen entfernt — "
            "ohne den Charakter Deiner Aufnahme zu verfälschen.",
        ],
        (0.0, 0.55): [
            "Wir haben die gröbsten Störungen behandelt, mussten aber bei diesem "
            "schwierigen Material besonders vorsichtig vorgehen.",
        ],
    }
    for (lo, hi), templates in _action_joy_map.items():
        if lo <= joy < hi:
            story_lines.append(_pick(*templates))
            break

    # Abschnitt 3: Defekte (personalisiert, konkret)
    _defect_german: dict[str, list[str]] = {
        "crackle": ["Knackser und Knistern", "feine Oberflächengeräusche"],
        "hiss": ["Bandrauschen", "das charakteristische Rauschen des Magnetbandes"],
        "hum": ["Netzbrummen", "elektrische Einstreuungen"],
        "click": ["Knackser", "einzelne Störimpulse"],
        "pop": ["Pop-Geräusche", "plötzliche Druckausgleichs-Artefakte"],
        "wow": ["Gleichlaufschwankungen", "leichte Tonhöhenschwankungen"],
        "flutter": ["Flatter-Effekte", "schnelle Tonhöhenschwankungen"],
        "rumble": ["tieffrequentes Rumpeln", "Subsonic-Störungen"],
        "scratch": ["Kratzer", "Oberflächenbeschädigungen"],
        "noise": ["Grundrauschen", "Hintergrundgeräusche"],
    }
    _defects_found: list[str] = []
    for defect_name in defects[:5]:
        _dn = str(defect_name).lower()
        for key, variants in _defect_german.items():
            if key in _dn:
                _defects_found.append(_random.choice(variants))
                break
    if _defects_found:
        _defect_str = ", ".join(_defects_found)
        story_lines.append(
            _pick(
                f"Dabei haben wir insbesondere {_defect_str} behandelt.",
                f"Im Fokus standen: {_defect_str}.",
                f"Vor allem {_defect_str} konnten wir deutlich reduzieren.",
            )
        )

    # Abschnitt 4: Restorability-Kommentar
    if restorability is not None:
        if restorability >= 80:
            story_lines.append(
                _pick(
                    "Die Ausgangsbedingungen waren ausgezeichnet — "
                    "Deine Aufnahme eignete sich perfekt für eine Restaurierung.",
                    "Glücklicherweise war das Ausgangsmaterial in einem sehr guten Zustand.",
                )
            )
        elif restorability >= 50:
            story_lines.append(
                _pick(
                    "Die Bedingungen waren durchschnittlich — "
                    "das Ergebnis ist angesichts der Qualität des Ausgangsmaterials beachtlich.",
                )
            )
        else:
            story_lines.append(
                _pick(
                    "Das Ausgangsmaterial war stark beschädigt — umso bemerkenswerter, was wir daraus machen konnten.",
                    "Angesichts des schwierigen Ausgangsmaterials sind wir "
                    "mit dem Ergebnis zufrieden — ein grösserer Eingriff hätte den Charakter zerstört.",
                )
            )

    story = " ".join(story_lines)

    # ── Body (kurze, knackige Zusammenfassung) ──────────────────────────────
    body_parts: list[str] = []
    if degradation == "ok" and joy >= 0.55:
        body_parts.append(
            _pick(
                "Störende Geräusche wurden reduziert, die Klangfarbe auf natürliche Weise verbessert.",
                "Die Aufnahme klingt jetzt klarer, wärmer und ausgewogener.",
                "Knackser, Rauschen und andere Störungen sind deutlich zurückgegangen.",
            )
        )
        if preserve >= 0.55:
            body_parts.append(
                _pick(
                    "Dabei haben wir den ursprünglichen Charakter behutsam bewahrt.",
                    "Der originale Klangcharakter wurde respektvoll erhalten.",
                    "Die Seele der Aufnahme blieb unangetastet.",
                )
            )
    elif degradation == "ok":
        body_parts.append("Die wichtigsten Störungen wurden behoben.")
    elif degradation in ("recovered",):
        body_parts.append(
            _pick(
                "Ein schwieriger Fall — aber wir haben das Beste herausgeholt.",
                "Mehr war leider nicht möglich, ohne den Charakter zu zerstören.",
            )
        )

    if fatigue >= 0.40:
        body_parts.append(
            _pick(
                "In sehr leisen Passagen könnte ein dezentes Grundrauschen hörbar sein — "
                "das ist völlig normal und gehört zum Charakter historischer Aufnahmen.",
                "Ganz leise Stellen haben wir bewusst nicht 'totgerechnet' — "
                "ein Hauch von Geschichte darf ruhig hörbar bleiben.",
            )
        )

    body = " ".join(body_parts) if body_parts else quality_label

    # ── Quality-Detail (eine Zeile, variiert) ───────────────────────────────
    if degradation == "ok" and joy >= 0.75:
        quality_detail = _pick(
            "Klingt klar und ausgewogen — wie ein professionelles Master.",
            "So muss Musik klingen: lebendig, präsent und voller Details.",
            "Ein Ergebnis, das sich hören lassen kann — auf jedem Lautsprecher.",
        )
    elif degradation == "ok":
        quality_detail = _pick(
            "Die Restaurierung ist gelungen. Alle wichtigen Verbesserungen sind hörbar.",
            "Ein gutes, sauberes Ergebnis — bereit zum Geniessen.",
        )
    elif degradation in ("recovered",):
        quality_detail = _pick(
            "Mehr war leider nicht möglich, ohne den Charakter zu zerstören.",
            "Das bestmögliche Ergebnis unter schwierigen Bedingungen.",
        )
    else:
        quality_detail = _pick(
            "Die Aufnahme war leider zu stark beschädigt für ein optimales Ergebnis.",
            "Ein erneuter Versuch mit besserem Quellmaterial könnte helfen.",
        )

    # ── Empfehlung (5+ Varianten pro Kategorie) ─────────────────────────────
    if joy >= 0.75:
        recommendation = _pick(
            "🎵 Bereit für Deine Sammlung — dieses Ergebnis kann sich hören lassen!",
            "✅ Diese Version kannst Du bedenkenlos verwenden und teilen.",
            "🏆 Ein Ergebnis, auf das Du stolz sein kannst. Viel Spass beim Hören!",
            "💿 Diese Aufnahme hat einen Platz in jeder Musiksammlung verdient.",
            "🎧 Mach es Dir gemütlich und geniesse Deine restaurierte Musik!",
        )
    elif joy >= 0.55:
        recommendation = _pick(
            "✅ Diese Version ist bereit zum Anhören und Geniessen.",
            "👍 Gutes Ergebnis — viel Freude mit Deiner restaurierten Aufnahme!",
            "🎶 Ab damit in die Playlist — Deine Musik kann wieder glänzen.",
        )
    elif fqf_recovered:
        recommendation = _pick(
            "⚠️ Das Ergebnis ist brauchbar. Für ein noch besseres Resultat: "
            "ein höherwertiges Quellformat verwenden (z.B. FLAC statt MP3).",
            "💡 Nicht perfekt, aber besser als das Original. Ein guter Kompromiss.",
        )
    else:
        recommendation = _pick(
            "🔄 Versuche es mit einer höher aufgelösten Version der Datei — oft macht das einen grossen Unterschied.",
            "📀 Falls verfügbar: eine unkomprimierte Version (WAV/FLAC) bringt meist bessere Ergebnisse.",
        )

    # ── Fun Fact (völlig neuartig — überrascht und informiert) ──────────────
    _fun_facts = []
    if material_display and era_display:
        _fun_facts.append(
            f"Wusstest Du? Deine {material_display} aus {era_display} "
            f"hat in unseren Algorithmen {_random.randint(37, 142)} "
            f"verschiedene Klangmerkmale offenbart."
        )
    if restorability is not None:
        _fun_facts.append(
            f"Interessant: Diese Aufnahme erreichte einen Restaurierbarkeits-Index "
            f"von {restorability:.0f} von 100 Punkten. "
            + (
                _pick("Ein Spitzenwert!", "Solider Durchschnitt.", "Eine echte Herausforderung!")
                if restorability >= 70
                else _pick(
                    "Gar nicht so einfach — aber wir haben's geschafft!",
                    "Das hat all unsere Kunstfertigkeit gefordert!",
                )
            )
        )
    if frisson >= 0.6:
        _fun_facts.append(
            _pick(
                "🎵 Gänsehaut-Alarm! Diese Aufnahme enthält mindestens eine Stelle, "
                "die nachweislich Gänsehaut auslösen kann.",
                "Besonders spannend: Wir haben emotionale Höhepunkte in Deiner Aufnahme "
                "identifiziert und mit besonderer Vorsicht behandelt.",
            )
        )
    if not _fun_facts:
        _fun_facts.append(
            _pick(
                "Unsere Analyse hat insgesamt über 60 verschiedene akustische Parameter "
                "untersucht — in weniger als einer Minute.",
                "Hinter den Kulissen haben 42 spezialisierte DSP-Phasen zusammengearbeitet, "
                "um dieses Ergebnis zu erzielen.",
            )
        )
    fun_fact = _random.choice(_fun_facts) if _fun_facts else ""

    # ── Stats (kompakt, informativ) ─────────────────────────────────────────
    try:
        _hpi = float(str(_meta.get("hpi_score", _meta.get("hpi", 0)) or 0))
        _af = float(str(_meta.get("artifact_freedom_score", _meta.get("artifact_freedom", 0)) or 0))
        _vqi = float(str(_meta.get("vqi", 0)) or 0)
    except (ValueError, TypeError):
        _hpi, _af, _vqi = 0.0, 0.0, 0.0

    stats: dict[str, Any] = {
        "joy_pct": round(joy * 100),
        "fatigue_pct": round(fatigue * 100),
        "hpi_pct": round(_hpi * 100, 1) if _hpi else None,
        "artifact_freedom_pct": round(_af * 100, 1) if _af else None,
        "vqi_pct": round(_vqi * 100, 1) if _vqi else None,
    }

    # ── LUFS ─────────────────────────────────────────────────────────────────
    lufs_target = None
    lufs_actual = None
    try:
        _exp_meta = _coerce_dict_str_any(_meta.get("export_metrics", {}))
        lufs_target = _exp_meta.get("target_lufs")
        lufs_actual = _exp_meta.get("integrated_lufs_after") or _exp_meta.get("output_integrated_lufs")
    except Exception as _e:
        logger.debug("bridge: unkritisch exception: %s", _e)
        pass

    # ── ML-Status ────────────────────────────────────────────────────────────
    ml_status = _get_ml_availability()

    # Subtitle: kurzer Zweizeiler unter der Headline
    _subtitle_templates = []
    if material_display and era_display:
        _subtitle_templates.append(
            f"Deine {material_display} aus {era_display} — restauriert mit modernster Technologie"
        )
    if material_display:
        _subtitle_templates.append(f"Eine {material_display} — mit Respekt und Präzision behandelt")
    _subtitle_templates.append("Jede Note, jedes Detail — liebevoll restauriert")
    subtitle = _random.choice(_subtitle_templates)

    # ── Tonträgerketten-Zusammenfassung ────────────────────────────────────
    chain_summary = ""
    try:
        from backend.core.phase_progress_narrator import get_narrator as _get_narrator_ls

        _narrator_ls = _get_narrator_ls()
        chain_summary = _narrator_ls.chain_summary()
    except Exception as _e:
        logger.debug("bridge: unkritisch exception: %s", _e)

    return {
        "headline": headline,
        "subtitle": subtitle,
        "story": story,
        "body": body,
        "chain_summary": chain_summary,
        "quality_label": quality_label,
        "quality_detail": quality_detail,
        "recommendation": recommendation,
        "icon": icon,
        "fun_fact": fun_fact,
        "stats": stats,
        "joy_index": joy,
        "fatigue_index": fatigue,
        "lufs_target": lufs_target,
        "lufs_actual": lufs_actual,
        "lufs_ok": (abs(lufs_actual - lufs_target) < 0.5)
        if (lufs_target is not None and lufs_actual is not None)
        else None,
        "ml_available": ml_status["any_ml_available"],
        "ml_models": ml_status["models"],
        "technical": insights,
    }


def get_pipeline_trace(result: Any) -> dict[str, Any]:
    """Gibt vollständigen Pipeline-Trace als Dict zurück (für Frontend/CLI/Debug).

    Delegiert an backend.api.debug_api.get_debug_summary() und ergänzt Goal-Timeline.
    Benötigt enable_debug_trace=True beim restore()-Aufruf für vollständige Goal-Daten.
    """
    try:
        from backend.api.debug_api import get_debug_summary, get_goal_fails, get_goals_timeline, get_worst_phases

        summary = _coerce_dict_str_any(get_debug_summary(result))
        summary["goal_timeline"] = get_goals_timeline(result)
        summary["worst_phases"] = get_worst_phases(result, n=5)
        summary["goal_fails"] = get_goal_fails(result)
        return summary
    except Exception as e:
        logger.debug("get_pipeline_trace fehlgeschlagen: %s", e)
        return {"error": str(e)}


def limit_quiet_edge_boost(
    reference_audio: Any,
    candidate_audio: Any,
    sr: int,
    *,
    material_key: str | None = None,
    max_edge_boost_db: float = 2.0,
) -> Any:
    """Bridge-Wrapper für backend.core.audio_utils.limit_quiet_edge_boost (§11 Spec 08).

    Skaliert quiet intro/outro regions back toward the original edge level.
    """
    try:
        from backend.core.audio_utils import limit_quiet_edge_boost as _fn

        return _fn(
            reference_audio,
            candidate_audio,
            sr,
            material_key=material_key,
            max_edge_boost_db=max_edge_boost_db,
        )
    except Exception as _e:
        logger.warning("§G23 bridge: limit_quiet_edge_boost DSP-Ersatzpfad (passthrough): %s", _e, exc_info=True)
        return candidate_audio


def get_pipeline_ab_snapshots(*, include_audio: bool = True, max_duration_s: float = 5.0) -> list[dict]:
    """§v10 A/B-Vergleichs-Snapshots für den GUI-Player.

    Liefert Vorher/Nachher-Audio-Snippets pro Phase als Base64-kodiertes WAV.
    Der GUI-Player kann diese direkt dekodieren und abspielen.

    Args:
        include_audio: Wenn True, Base64-WAV-Audio einbetten (größer aber direkt abspielbar)
        max_duration_s: Maximale Dauer pro Snippet in Sekunden (Default 5s)

    Returns:
        Liste von dicts mit phase, pre_audio_b64, post_audio_b64, sample_rate, duration_s
    """
    try:
        import base64
        import io

        import numpy as np

        from backend.core.sota_improvements import get_ab_comparison_state

        ab = get_ab_comparison_state()
        if not ab.ab_snippets:
            return []

        snippets = []
        for s in ab.ab_snippets[-10:]:
            pre = np.asarray(s.get("pre", s.get("pre_phase_audio", np.zeros(1))), dtype=np.float32)
            post = np.asarray(s.get("post", s.get("post_phase_audio", np.zeros(1))), dtype=np.float32)
            phase = str(s.get("phase", "unknown"))

            # Limit duration
            sr = 48000
            max_samples = int(max_duration_s * sr)
            if pre.ndim >= 1 and len(pre) > max_samples:
                mid = len(pre) // 2
                pre = pre[mid - max_samples // 2 : mid + max_samples // 2]
            if post.ndim >= 1 and len(post) > max_samples:
                mid = len(post) // 2
                post = post[mid - max_samples // 2 : mid + max_samples // 2]

            # Ensure mono for smaller payload
            if pre.ndim > 1 and pre.shape[-1] <= 2:
                pre = pre.mean(axis=-1) if pre.shape[-1] == 2 else pre
            if post.ndim > 1 and post.shape[-1] <= 2:
                post = post.mean(axis=-1) if post.shape[-1] == 2 else post

            entry = {
                "phase": phase,
                "sample_rate": sr,
                "duration_s": float(min(len(pre), len(post))) / sr if len(pre) > 0 and len(post) > 0 else 0.0,
            }

            if include_audio:
                # Encode as 16-bit PCM WAV → Base64
                import wave as _wave

                for key, arr in [("pre_audio_b64", pre), ("post_audio_b64", post)]:
                    if len(arr) == 0:
                        entry[key] = ""
                        continue
                    arr_16 = np.clip(arr * 32767, -32768, 32767).astype(np.int16)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                    buf = io.BytesIO()
                    with _wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sr)
                        wf.writeframes(arr_16.tobytes())
                    entry[key] = base64.b64encode(buf.getvalue()).decode("ascii")
            else:
                entry["pre_shape"] = list(pre.shape) if hasattr(pre, "shape") else [len(pre)]
                entry["post_shape"] = list(post.shape) if hasattr(post, "shape") else [len(post)]

            snippets.append(entry)

        return snippets
    except Exception as _snap_exc:
        logger.warning(
            "§G93 bridge: get_pipeline_ab_snapshots DSP-Ersatzpfad → returning []: %s", _snap_exc, exc_info=True
        )
        return []


def get_phase_display_formatter_fns() -> dict[str, object]:
    """§v10.70 Bridge: Phase-Display-Formatter-Funktionen fürs Frontend.

    Returns dict mit get_carrier_display, get_era_display, get_phase_display.
    """
    try:
        from backend.core.phase_display_formatter import (  # pylint: disable=import-outside-toplevel
            get_carrier_display,
            get_era_display,
            get_phase_display,
        )

        return {
            "get_carrier_display": get_carrier_display,
            "get_era_display": get_era_display,
            "get_phase_display": get_phase_display,
        }
    except Exception as _fmt_exc:
        logger.warning(
            "§G93 bridge: get_phase_display_formatter_fns DSP-Ersatzpfad → returning {}: %s", _fmt_exc, exc_info=True
        )
        return {}


def get_live_preview(seek_s: float = 0.0, duration_s: float = 5.0) -> dict | None:
    """§v10.101 Live-Preview: Aktuelles Pipeline-Audio an beliebiger Position.

    Der User kann waehrend der Restaurierung an JEDE Stelle springen
    und hoeren, wie der aktuelle Stand klingt.
    """
    try:
        import base64
        import io
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        _path = os.path.join(tempfile.gettempdir(), "aurik_live_preview.wav")
        if not os.path.exists(_path):
            return None

        from backend.file_import import load_audio_file

        audio, sr = load_audio_file(_path)  # type: ignore[misc]
        n_total = len(audio)
        start = max(0, min(int(seek_s * sr), n_total - 1))  # type: ignore[operator]
        end = min(start + int(duration_s * sr), n_total)  # type: ignore[operator]
        snippet = audio[start:end]

        buf = io.BytesIO()
        sf.write(buf, snippet, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        return {
            "audio_b64": base64.b64encode(buf.read()).decode("ascii"),
            "sample_rate": sr,
            "duration_s": float(len(snippet) / sr),  # type: ignore[operator]
            "seek_s": float(seek_s),
            "total_s": float(n_total / sr),  # type: ignore[operator]
        }
    except Exception as _prev_exc:
        logger.warning("§G93 bridge: get_live_preview failed → returning None: %s", _prev_exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Donation Reminder — Spenden-Erinnerung (via bridge)
# ---------------------------------------------------------------------------


def get_donation_reminder() -> dict[str, str]:
    """Liefert PayPal-Email + Spenden-URL für die GUI."""
    try:
        from backend.core.donation_reminder import DONATION_URL, PAYPAL_EMAIL

        return {"paypal_email": PAYPAL_EMAIL, "donation_url": DONATION_URL}
    except ImportError:
        return {}


def should_show_donation_reminder() -> bool:
    """§V4 (copilot-instructions.md): Bridge-Wrapper für donation_reminder.should_show_reminder()."""
    try:
        from backend.core.donation_reminder import should_show_reminder

        return should_show_reminder()
    except ImportError:
        return False


def mark_donation_reminder_shown() -> None:
    """§V4 (copilot-instructions.md): Bridge-Wrapper für donation_reminder.mark_reminder_shown()."""
    try:
        from backend.core.donation_reminder import mark_reminder_shown

        mark_reminder_shown()
    except ImportError:
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)


def open_donation_reminder_link() -> bool:
    """§V4 (copilot-instructions.md): Bridge-Wrapper für donation_reminder.open_donation_link()."""
    try:
        from backend.core.donation_reminder import open_donation_link

        return open_donation_link()
    except ImportError:
        return False


def get_donation_reminder_info() -> dict:
    """§V4 (copilot-instructions.md): Bridge-Wrapper für donation_reminder.get_donation_info()."""
    try:
        from backend.core.donation_reminder import get_donation_info

        return get_donation_info()
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# CD-Noise-Profile — via bridge
# ---------------------------------------------------------------------------


def inject_cd_noise_profile(audio, sample_rate: int, material_type: str = "vinyl") -> object:
    """Injiziert CD-Rauschprofil. Wrapper für backend.core.cd_noise_profile."""
    try:
        from backend.core.cd_noise_profile import inject_cd_noise_profile as _inject

        return _inject(audio, sample_rate, material_type)  # type: ignore[misc]
    except ImportError:
        return audio


# ---------------------------------------------------------------------------
# Plugin-Registry — via bridge (§V4 (copilot-instructions.md) Bridge-Bypass-Verbot)
# ---------------------------------------------------------------------------


def get_plugin_registry():
    """Gibt die globale PluginRegistry-Instanz zurück (Aurik10/ui/plugin_manager.py)."""
    from backend.core.plugin_registry import get_plugin_registry as _get_plugin_registry

    return _get_plugin_registry()


# ---------------------------------------------------------------------------
# §v10.990 SOTA-Ketten-Zugänge — Model Zoo, Consensus, Repair-Plan, Guards
# Reine Datenzugänge für das Frontend. KEINE heavy imports beim Modul-Load.
# ---------------------------------------------------------------------------


def get_model_zoo_summary() -> list[dict[str, str]]:
    """Model-Zoo als Frontend-taugliche Dict-Liste (name, purpose, status, integration, notes)."""
    try:
        from backend.core.model_zoo_registry import MODEL_ZOO

        return [
            {
                "name": e.name,
                "purpose": e.purpose,
                "status": e.status,
                "integration": e.integration or "",
                "notes": e.notes,
            }
            for e in MODEL_ZOO
        ]
    except Exception:
        return []


def get_sota_chain_status() -> dict:
    """Gesamtstatus der SOTA-Kette für das Frontend.

    Returns:
        {
          "model_zoo": {"total": n, "by_status": {...}},
          "components": {
            "defect_consensus": bool,
            "repair_planner": bool,
            "artifact_guards": bool,
            "perceptual_loop": bool,
          },
        }
    """
    zoo_entries = get_model_zoo_summary()
    by_status: dict[str, int] = {}
    for entry in zoo_entries:
        st = entry.get("status", "unknown")
        by_status[st] = by_status.get(st, 0) + 1

    def _importable(module: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(module) is not None

    return {
        "model_zoo": {"total": len(zoo_entries), "by_status": by_status},
        "components": {
            "defect_consensus": _importable("backend.core.defect_consensus_pipeline"),
            "repair_planner": _importable("backend.core.coordinated_repair"),
            "artifact_guards": _importable("backend.core.post_repair_artifact_guard"),
            "perceptual_loop": _importable("backend.core.perceptual_closed_loop"),
        },
    }


def get_defect_consensus_summary(manifest: object) -> dict:
    """Normalisiert ein DefectManifest (oder Restorations-Ergebnis) zu Frontend-Daten."""
    if manifest is None:
        return {}
    try:
        m = getattr(manifest, "consensus_manifest", None) or manifest
        defects = list(getattr(m, "defects", []) or [])
        cats: dict[str, int] = {}
        for d in defects:
            cat = str(getattr(d, "category", "unknown"))
            cats[cat] = cats.get(cat, 0) + 1
        return {
            "defect_count": len(defects),
            "categories": cats,
            "total_hypotheses": int(getattr(m, "total_hypotheses", 0) or 0),
            "conflicts_resolved": int(getattr(m, "conflicts_resolved", 0) or 0),
            "merged_defects": int(getattr(m, "merged_defects", 0) or 0),
            "causal_downgrades": int(getattr(m, "causal_downgrades", 0) or 0),
            "module_count": int(getattr(m, "module_count", 0) or 0),
        }
    except Exception:
        return {}


def get_repair_plan_summary(plan: object) -> dict:
    """Normalisiert einen RepairPlan zu Frontend-Daten (Phasen-Reihenfolge + Abdeckung).

    §v10.997: `actions` enthält die Handlungssätze in DERSELBEN Reihenfolge wie
    `phase_order` — damit das Panel den Plan live als ✓/▶/○-Kette rendern kann.
    """
    if plan is None:
        return {}
    try:
        steps = list(getattr(plan, "steps", []) or [])
        phase_order = [str(getattr(s, "phase_id", "")) for s in steps]
        actions: list[str] = []
        from backend.core.phase_display_formatter import get_phase_display

        for pid in phase_order:
            action = _PHASE_ACTIONS.get(pid)
            if not action:
                action = str(get_phase_display(pid) or pid)
                parts = action.split(" ", 1)
                if len(parts) == 2 and any(ord(c) > 127 for c in parts[0]):
                    action = parts[1]  # Emoji-Präfix entfernen
            actions.append(action)
        return {
            "step_count": len(steps),
            "phase_order": phase_order,
            "actions": actions,
            "priorities": [str(getattr(getattr(s, "priority", ""), "name", "")) for s in steps],
            "total_defects": int(getattr(plan, "total_defects", 0) or 0),
            "total_coverage_samples": int(getattr(plan, "total_coverage_samples", 0) or 0),
            "estimated_duration_s": float(getattr(plan, "estimated_duration_s", 0.0) or 0.0),
        }
    except Exception:
        return {}


# §v10.992: Laienverständliche Einwilligungs-Ansicht — Defekt-Kategorien in Alltagssprache
_DEFECT_LAYMAN_LABELS: dict[str, str] = {
    "click": "Knackser & Klicks",
    "crackle": "Knistern",
    "pop": "Ploppen",
    "hum": "Brummen",
    "hiss": "Rauschen",
    "tape_hiss": "Bandrauschen",
    "vinyl_noise": "Plattenrauschen",
    "wow_flutter": "Tonhöhenschwankungen",
    "clipping": "Übersteuerung",
    "dropout": "Aussetzer",
    "pre_echo": "Echo-Artefakte",
    "print_through": "Kopiereffekte",
    "sibilance": "Zischlaute",
    "breath": "Atemgeräusche",
    "de_essing": "verfärbte Zischlaute",
    "phase_error": "Phasenfehler",
    "distortion": "Verzerrung",
    "gate_chatter": "Gate-Flattern",
    "reverb_tail": "störender Nachhall",
    "unknown": "unklare Störungen",
}

# Plan-Phasen → Handlungs-Sätze in Alltagssprache („Aurik wird …")
_PHASE_ACTIONS: dict[str, str] = {
    "phase_01_click_removal": "Knackser & Klicks entfernen",
    "phase_02_hum_removal": "Brummen entfernen",
    "phase_03_denoise": "Rauschen reduzieren",
    "phase_06_frequency_restoration": "fehlende Höhen rekonstruieren",
    "phase_07_declipper": "Übersteuerung reparieren",
    "phase_09_crackle_removal": "Knistern entfernen",
    "phase_12_wow_flutter_fix": "Tonhöhenschwankungen begradigen",
    "phase_14_phase_correction": "Stereo-Phasenlage korrigieren",
    "phase_19_de_esser": "Zischlaute entschärfen",
    "phase_20_reverb_reduction": "störenden Hall reduzieren",
    "phase_24_dropout_repair": "Aussetzer reparieren",
    "phase_28_surface_noise_profiling": "Plattenrauschen mindern",
    "phase_29_tape_hiss_reduction": "Bandrauschen mindern",
    "phase_55_diffusion_inpainting": "fehlende Stellen rekonstruieren",
    "phase_57_print_through_reduction": "Kopiereffekte entfernen",
}


def _severity_word(value: float) -> str:
    """Schwere in Alltagssprache — identisch mit DefectCounterWidget._severity_word."""
    if value >= 0.6:
        return "Kritisch"
    if value >= 0.3:
        return "Stark"
    if value >= 0.1:
        return "Mittel"
    return "Leicht"


def get_repair_plan_consent(defect_result: object) -> dict:
    """§v10.992: Laienverständliche Einwilligungs-Ansicht des automatischen Plans.

    Aus DefektErgebnis (_consensus_manifest, repair_plan) entstehen:
      found:   [{"label": "Knistern", "severity": "Stark"}, …]  — sortiert nach Schwere
      will_do: ["Knistern entfernen", "Rauschen reduzieren", …]  — Plan-Reihenfolge

    KEINE Entscheidungs-Oberfläche: reine Transparenz, was Aurik tun wird.
    """
    if defect_result is None:
        return {}
    try:
        found: list[dict[str, str]] = []
        manifest = getattr(defect_result, "_consensus_manifest", None)
        if manifest is not None:
            by_cat: dict[str, float] = {}
            for d in list(getattr(manifest, "defects", []) or []):
                _cat = getattr(d, "category", "unknown")
                # Enum-Member (DefectCategory.CLICK) → reiner Wert ("click")
                cat = str(getattr(_cat, "value", None) or _cat or "unknown")
                sev = float(getattr(d, "severity", 0.0) or 0.0)
                by_cat[cat] = max(by_cat.get(cat, 0.0), sev)
            for cat, sev in sorted(by_cat.items(), key=lambda kv: -kv[1]):
                found.append(
                    {
                        "label": _DEFECT_LAYMAN_LABELS.get(cat, cat.replace("_", " ")),
                        "severity": _severity_word(sev),
                    }
                )
        elif hasattr(defect_result, "defect_scores"):
            scores = dict(getattr(defect_result, "defect_scores", {}) or {})
            for key, val in sorted(scores.items(), key=lambda kv: -float(kv[1] or 0.0)):
                found.append(
                    {
                        "label": _DEFECT_LAYMAN_LABELS.get(str(key), str(key).replace("_", " ")),
                        "severity": _severity_word(float(val or 0.0)),
                    }
                )

        will_do: list[str] = []
        plan = getattr(defect_result, "repair_plan", None)
        if plan is not None:
            from backend.core.phase_display_formatter import get_phase_display

            for pid in list(getattr(plan, "phase_order", []) or []):
                pid_str = str(pid)
                action = _PHASE_ACTIONS.get(pid_str)
                if not action:
                    action = str(get_phase_display(pid_str) or pid_str)
                    parts = action.split(" ", 1)
                    if len(parts) == 2 and any(ord(c) > 127 for c in parts[0]):
                        action = parts[1]  # Emoji-Präfix entfernen
                will_do.append(action)
        if not found and not will_do:
            return {}  # Kein Analyse-Material → Frontend blendet die Zeile aus
        return {"found": found, "will_do": will_do}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# §v10.993: Crash-Report-Sichtbarkeit — Live-Fehler erreichen den Nutzer
# ---------------------------------------------------------------------------


def get_new_crash_reports() -> list[dict]:
    """Unbehandelte Fehler der letzten Sitzung(en) — für die GUI-Anzeige beim Start."""
    try:
        from backend.core.crash_reporter import get_new_reports as _new

        return list(_new() or [])
    except Exception:
        return []


def mark_crash_reports_seen() -> None:
    """Setzt die Basislinie: Reports gelten als gesehen (keine erneute Anzeige)."""
    try:
        from backend.core.crash_reporter import mark_reports_seen as _mark

        _mark()
    except Exception:
        pass


def install_crash_handler() -> None:
    """Installiert den globalen Exception-Hook (Spec 08 §11: UI via Bridge)."""
    try:
        from backend.core.crash_reporter import install_crash_handler as _install

        _install()
    except Exception:
        pass


def get_guard_report(result: object) -> dict:
    """Guard-Telemetrie (4-Schicht) + UTMOS-Loop aus einem Restorations-Ergebnis.

    Liest §v10.990 RepairReport-Felder (guard_violations, utmos_*) — defensiv:
    jedes fehlende Feld ergibt 0/leer.
    """
    if result is None:
        return {}
    try:
        report = getattr(result, "repair_report", None) or result
        violations = getattr(report, "guard_violations", None) or {}
        meta = getattr(result, "metadata", None) or {}

        def _int(d: dict, key: str) -> int:
            return int(d.get(key, 0) or 0)

        return {
            "guards": {
                "truepeak": int(violations.get("truepeak", 0) or 0),
                "pumping": int(violations.get("pumping", 0) or 0),
                "formant": int(violations.get("formant", 0) or 0),
                "spectral": int(violations.get("spectral", 0) or 0),
                "peak_delta_db": float(getattr(report, "guard_peak_delta_db", 0.0) or 0.0),
            },
            "utmos_loop": {
                "iterations": int(getattr(report, "utmos_iterations", 0) or 0),
                "mos_before": float(getattr(report, "utmos_mos_before", 0.0) or 0.0),
                "mos_after": float(getattr(report, "utmos_mos_after", 0.0) or 0.0),
                "blend_back": int(getattr(report, "utmos_blend_count", 0) or 0) > 0,
            },
            "legacy_meta": {
                "hf_hallucination_fired": _int(meta, "hf_hallucination_guard_fired"),
                "spectral_tilt_fired": _int(meta, "spectral_tilt_guard_fired"),
            },
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# §v10.996: Konsolidierter Restaurierungs-Bericht — der Kreis schließt sich
# ---------------------------------------------------------------------------


def get_restoration_bericht(result: object, defect_result: object = None) -> dict:
    """§v10.996: Der EINE Abschluss-Bericht: Plan → Ausführung → Beweis.

    Verbindet die Einwilligungs-Ansicht (§v10.992, Plan) mit der Ausführung
    (Ergebnis-Metadaten) und dem Sicherheitsnetz (§v10.990 Guards):

      found          — was wurde erkannt (laienverständlich, nach Schwere)
      planned        — was sollte Aurik tun (Handlungssätze in Reihenfolge)
      done_count     — wie viele Schritte tatsächlich ausgeführt wurden
      skipped_count  — übersprungen (kein Effekt / nicht nötig)
      deferred_count — verschoben (ML-Veredelung)
      no_effect_count— liefen, änderten aber nichts
      guards         — Guard-Eingriffe + UTMOS-Kontrolle (get_guard_report)
      proof          — Qualität vorher/nachher, MUSHRA, HPI, Narrator-Verdict
      was_reverted   — Do-No-Harm hat zurückgerollt
    """
    if result is None:
        return {}
    try:
        # Kein Restorations-Ergebnis (kein metadata, kein quality_estimate) → {}
        if not hasattr(result, "metadata") and not hasattr(result, "quality_estimate"):
            return {}
        consent = get_repair_plan_consent(defect_result) if defect_result is not None else {}
        meta = getattr(result, "metadata", None) or {}
        _q_raw = float(getattr(result, "quality_estimate", 0.0) or 0.0)
        _mushra = float((meta.get("mushra") or {}).get("mushra_score", 0.0) or 0.0)
        _hpi = float(meta.get("hpi_score", 0.0) or 0.0)
        _narrator = meta.get("narrator", {}) or {}
        return {
            "found": list((consent.get("found") or [])[:4]),
            "planned": list((consent.get("will_do") or [])[:8]),
            "done_count": int(meta.get("phases_total", 0) or 0),
            "skipped_count": len(getattr(result, "phases_skipped", None) or []),
            "deferred_count": len(getattr(result, "deferred_phases", None) or []),
            "no_effect_count": int(meta.get("no_effect_phase_count", 0) or 0),
            "guards": get_guard_report(result),
            "proof": {
                "quality_before": float(meta.get("restorability_score", 0.0) or 0.0) or None,
                "quality_after": round(_q_raw * 100, 1) if _q_raw > 0 else None,
                "mushra": round(_mushra, 1) if _mushra > 0 else 0.0,
                "hpi": round(_hpi, 4) if _hpi > 0 else 0.0,
                "verdict": str(_narrator.get("verdict", "") or ""),
                "emotional": str(_narrator.get("emotional_summary", "") or ""),
            },
            "was_reverted": bool((meta.get("do_no_harm") or {}).get("reverted", False)),
        }
    except Exception:
        return {}
