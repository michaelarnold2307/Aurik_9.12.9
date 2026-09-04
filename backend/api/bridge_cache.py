"""Aurik Bridge — Cache Infrastructure (§11 Spec 08)
===================================================
Thread-safe LRU caches for analysis results (defect, era/genre, medium,
restorability). Content-addressed keys prevent redundant re-analysis when
files are renamed or moved.

Public API:
    _AnalysisLruCache (class)
    content_cache_key
    cache_defect_result, get_cached_defect_result, clear_defect_cache
    cache_era_genre_result, get_cached_era_genre_result, clear_era_genre_cache
    cache_medium_result, get_cached_medium_result, clear_medium_cache
    cache_restorability_result, get_cached_restorability_result

Referenz: Spec 08 §11 Softwareschichten-Architektur.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANALYSIS_CACHE_MAX = 64
_CONTENT_CHUNK = 4096  # Bytes vom Anfang + Ende für SHA-256 Content-Key
_CONTENT_KEY_CACHE_MAX = 512


# ---------------------------------------------------------------------------
# Fast path for repeated cache lookups: (path, size, mtime_ns) -> content-key
# ---------------------------------------------------------------------------

_content_key_cache: OrderedDict[tuple[str, int, int], str] = OrderedDict()
_content_key_lock = threading.Lock()


# ---------------------------------------------------------------------------
# LRU Cache Class
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Content-Addressed Keying
# ---------------------------------------------------------------------------

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
    except OSError as exc:
        logger.debug("§V6 os.stat fehlgeschlagen — Dateipfad als Cache-Key verwendet: %s", exc)
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
    except OSError as exc:
        logger.debug("§V6 Datei-Lesen fehlgeschlagen — Dateipfad als Cache-Key verwendet: %s", exc)
        return file_path

    with _content_key_lock:
        _content_key_cache[meta_key] = digest
        _content_key_cache.move_to_end(meta_key)
        while len(_content_key_cache) > _CONTENT_KEY_CACHE_MAX:
            _content_key_cache.popitem(last=False)

    return digest


# ---------------------------------------------------------------------------
# Singleton caches — one per analysis type for independent eviction
# ---------------------------------------------------------------------------

_defect_lru: _AnalysisLruCache = _AnalysisLruCache()
_era_genre_lru: _AnalysisLruCache = _AnalysisLruCache()
_medium_lru: _AnalysisLruCache = _AnalysisLruCache()
_restorability_lru: _AnalysisLruCache = _AnalysisLruCache()


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
        logger.debug("bridge: Medium-Zwischenspeicher vollständig geleert.")
    else:
        key = content_cache_key(file_path)
        _medium_lru.remove(key)
        _medium_lru.remove(file_path)  # remove() handles path-alias too
        logger.debug("bridge: Medium-Zwischenspeicher für '%s' geleert.", file_path)


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
