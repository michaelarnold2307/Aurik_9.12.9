"""InferenceSessionManager — Zentraler ONNX-Session-Lifecycle.

§15.9 [RELEASE_MUST]: Alle ONNX-Sessions MÜSSEN über diesen Singleton verwaltet werden.
Direkte ``onnxruntime.InferenceSession(...)``-Aufrufe sind außerhalb dieses Moduls
verboten (Ausnahme: Test-Fixtures in conftest.py).

Features:
    - LRU-Cache mit konfigurierbarer Max-Größe (default: 4 Sessions)
    - Memory-Monitoring (Warnung bei >2 GB)
    - Thread-sicherer Zugriff
    - Lazy-Import von onnxruntime

Usage::

    from backend.core.ml.session_manager import get_session_manager

    mgr = get_session_manager(max_sessions=4)
    session = mgr.acquire("panns", "models/panns/cnn14.onnx")
    # ... Inferenz ...
    mgr.release("panns")
    print(f"Memory: {mgr.get_total_memory_mb():.1f} MB")

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class InferenceSessionManager:
    """Singleton: Zentraler ONNX-Session-Lifecycle-Manager.

    Verwaltet ONNX-InferenceSession-Instanzen mit LRU-Eviction.
    Thread-sicher via ``threading.Lock``.

    Attributes:
        max_sessions: Maximale Anzahl gleichzeitig geladener Sessions.
        _cache:        Dict model_name → (session, model_size_mb, last_access).
        _access_order: LRU-Queue (zuletzt genutzt am Ende).
        _lock:         Reentrant-Lock für Thread-Safety.
    """

    _instance: InferenceSessionManager | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self, max_sessions: int = 4, memory_limit_mb: float = 2048.0) -> None:
        """Initialisiert den Session-Manager.

        Args:
            max_sessions:    Maximale Anzahl gecachter Sessions (LRU-Eviction).
            memory_limit_mb: Warn-Schwelle für Gesamtspeicher in MB.
        """
        self.max_sessions: int = max(max_sessions, 1)
        self.memory_limit_mb: float = memory_limit_mb
        self._cache: dict[str, tuple[Any, float, float]] = {}
        # cache: model_name → (InferenceSession, size_mb, last_access_timestamp)
        self._access_order: list[str] = []
        self._lock: threading.RLock = threading.RLock()

    # ── Singleton-Zugriff ─────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, max_sessions: int = 4) -> InferenceSessionManager:
        """Singleton-Instanz (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(max_sessions=max_sessions)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Singleton zurücksetzen (nur für Tests)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.release_all()
                cls._instance = None

    # ── Session-Management ────────────────────────────────────────────────

    def acquire(self, model_name: str, model_path: str | Path) -> Any:
        """ONNX-Session laden oder aus Cache zurückgeben.

        Args:
            model_name: Eindeutiger Name (z.B. "panns", "whisper_tiny").
            model_path: Pfad zur .onnx-Datei.

        Returns:
            ``onnxruntime.InferenceSession``.

        Raises:
            RuntimeError: Wenn onnxruntime nicht importiert werden kann.
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf/
                Fail: Wenn die Modelldatei nicht ladbar ist.
        """
        model_path = Path(model_path)

        with self._lock:
            # ── Cache-Hit? ────────────────────────────────────────────────
            if model_name in self._cache:
                session, size_mb, _ = self._cache[model_name]
                self._touch(model_name)
                logger.debug("Sitzung-Zwischenspeicher-Hit: %s (%.1f MB)", model_name, size_mb)
                return session

            # ── Cache voll? LRU-Eviction ──────────────────────────────────
            while len(self._cache) >= self.max_sessions and self._access_order:
                evict_name = self._access_order[0]
                self._evict(evict_name)

            # ── Neue Session laden ────────────────────────────────────────
            session, size_mb = self._load_session(model_path)
            self._cache[model_name] = (session, size_mb, time.monotonic())
            self._access_order.append(model_name)

            # ── Memory-Warnung ─────────────────────────────────────────────
            total_mb = self._total_memory_mb()
            if total_mb > self.memory_limit_mb:
                logger.warning(
                    "ONNX-Sitzung-Memory > %.0f MB: %.1f MB (Modelle: %s). Erwäge max_sessions zu reduzieren.",
                    self.memory_limit_mb,
                    total_mb,
                    list(self._cache.keys()),
                )

            logger.info(
                "Sitzung geladen: %s (%.1f MB) [%d/%d Sessions, %.1f MB total]",
                model_name,
                size_mb,
                len(self._cache),
                self.max_sessions,
                total_mb,
            )
            return session

    def release(self, model_name: str) -> None:
        """Eine Session aus dem Cache entfernen."""
        with self._lock:
            self._evict(model_name)

    def release_all(self) -> None:
        """Alle Sessions freigeben."""
        with self._lock:
            for name in list(self._cache.keys()):
                self._evict(name)
            self._access_order.clear()
            logger.info("Alle %d Sessions freigegeben", len(self._cache))

    # ── Memory-Monitoring ─────────────────────────────────────────────────

    def get_total_memory_mb(self) -> float:
        """Geschätzte Gesamtmemory-Nutzung in MB."""
        with self._lock:
            return self._total_memory_mb()

    def get_session_sizes(self) -> dict[str, float]:
        """Memory-Nutzung pro Session in MB."""
        with self._lock:
            return {name: size_mb for name, (_, size_mb, _) in self._cache.items()}

    # ── Private ───────────────────────────────────────────────────────────

    def _total_memory_mb(self) -> float:
        """Summe aller Session-Sizes (ohne Lock — Aufrufer muss locken)."""
        return sum(size_mb for _, (_, size_mb, _) in self._cache.items())

    def _touch(self, model_name: str) -> None:
        """LRU-Position aktualisieren (ans Ende schieben)."""
        if model_name in self._access_order:
            self._access_order.remove(model_name)
        self._access_order.append(model_name)
        # Timestamp updaten
        if model_name in self._cache:
            session, size_mb, _ = self._cache[model_name]
            self._cache[model_name] = (session, size_mb, time.monotonic())

    def _evict(self, model_name: str) -> None:
        """Session aus Cache und LRU-Queue entfernen."""
        if model_name in self._cache:
            _, size_mb, _ = self._cache[model_name]
            del self._cache[model_name]
            if model_name in self._access_order:
                self._access_order.remove(model_name)
            logger.debug("Sitzung evictet: %s (%.1f MB freigegeben)", model_name, size_mb)

    @staticmethod
    def _load_session(model_path: Path) -> tuple[Any, float]:
        """ONNX-Session laden und Größe schätzen.

        Returns:
            (InferenceSession, estimated_size_mb).
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime ist nicht installiert. pip install onnxruntime (oder onnxruntime-gpu für CUDA)."
            ) from exc

        # Provider-Präferenz: CUDA > ROCm > CoreML > CPU
        providers = ort.get_available_providers()
        logger.debug("ONNX-Provider verfügbar: %s", providers)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        # MIGraphX GPU backend (gfx1100/RDNA3). §v10.40 Compile-Zeit-Regel:
        # Modelle > 200 MB überspringen (Compile > 30 min) → ORT statt MIGraphX.
        try:
            size_mb = model_path.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0
        try:
            from backend.core.migraphx_adapter import (
                MIGRAPHX_MAX_MODEL_MB,
                MIGraphXSession,
                is_migraphx_available,
                is_migraphx_size_eligible,
            )

            if is_migraphx_available():
                if not is_migraphx_size_eligible(model_path):
                    logger.debug(
                        "SessionManager: %s überschreitet MIGraphX-Größenlimit (§v10.40, %.1f MB > %.0f MB) → ORT-Pfad",
                        model_path.name,
                        size_mb,
                        MIGRAPHX_MAX_MODEL_MB,
                    )
                else:
                    logger.debug("SessionManager: trying MIGraphX GPU for %s", model_path.name)
                    session = MIGraphXSession(
                        str(model_path),
                        providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
                        sess_options=sess_options,
                    )
                    return session, size_mb
        except Exception as _mgx_exc:
            logger.debug("SessionManager: MIGraphX not available (%s), using ORT CPU", _mgx_exc)

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # ── Größe schätzen (Dateigröße ≈ Modellgröße im RAM) ───────────
        try:
            size_mb = model_path.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0

        return session, size_mb

    def get_active_count(self) -> int:
        """Anzahl aktuell gecachter Sessions (Spec 15 §9.5)."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Alle Sessions freigeben (Spec 15 §9.5, Alias fuer release_all)."""
        self.release_all()

    def __len__(self) -> int:
        """Anzahl gecachter Sessions."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, model_name: str) -> bool:
        with self._lock:
            return model_name in self._cache


# ── Convenience-Funktion ────────────────────────────────────────────────────


def get_session_manager(max_sessions: int = 4) -> InferenceSessionManager:
    """Singleton-Instanz des InferenceSessionManager (thread-safe).

    Args:
        max_sessions: Maximale Anzahl gecachter Sessions.

    Returns:
        Globale InferenceSessionManager-Instanz.
    """
    return InferenceSessionManager.get_instance(max_sessions=max_sessions)
