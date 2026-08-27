"""
Adaptive Resource Manager für Aurik 10.0.0
Automatische CPU- und Speicher-Auslastungsüberwachung und dynamische Core-Zuteilung für ML-Phasen.
Ermöglicht Fallback auf lightweight-Algorithmen bei Ressourcenknappheit.
Funktioniert out-of-the-box, keine Nutzerinteraktion nötig.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

_MONITOR_JOIN_TIMEOUT_S = 1.0


_instance: AdaptiveResourceManager | None = None
_lock_singleton = threading.Lock()


def get_resource_manager() -> AdaptiveResourceManager:
    """Gibt zurück: or create AdaptiveResourceManager singleton.

    Returns:
        AdaptiveResourceManager singleton instance
    """
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock_singleton:
            if _instance is None:
                _instance = AdaptiveResourceManager()
    return _instance


class AdaptiveResourceManager:
    """Überwacht CPU/Speicher-Auslastung und passt Core-Zuteilung dynamisch an."""

    # Lock-order: Priority 3 (ARM) — see §3.9.8.
    # evict_stale_plugins() is called OUTSIDE the ARM lock to respect lock hierarchy.
    # Never call back into PLM or MLMemoryBudget while holding self.lock.
    def __init__(
        self,
        min_cores: int = 2,
        max_cores: int | None = None,
        check_interval: float = 2.0,
        cpu_threshold: int = 80,
        memory_threshold: int = 78,
    ) -> None:
        self.system_cores = mp.cpu_count()
        self.max_cores = max_cores or self.system_cores
        self.min_cores = min_cores
        self.cpu_threshold = cpu_threshold  # in Prozent
        self.memory_threshold = memory_threshold  # in Prozent
        self.check_interval = check_interval
        self.current_cores = self.max_cores
        self.running = False
        self.lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self.use_lightweight = False  # Fallback-Flag
        # §Perf: get_cpu_usage() wurde im Fast-Mode-Gate 277×/Lauf aufgerufen
        # (27,4 s). psutil.cpu_percent(interval=None) kostet ~0,1–0,3 s pro Aufruf
        # (cpu_times-/proc-Read auf belastetem Desktop). CPU-Last ändert sich im
        # Sekundenbereich nicht relevant — 5-s-Cache liefert dieselben adaptiven
        # Entscheidungen (Lightweight-Mode-Wechsel sind Minuten-Entscheidungen).
        self._cpu_cache_value: float | None = None
        self._cpu_cache_ts: float = 0.0
        self._CPU_CACHE_TTL = 5.0  # Sekunden
        self._mem_cache_value: float | None = None
        self._mem_cache_ts: float = 0.0
        self._MEM_CACHE_TTL = 5.0  # Sekunden
        logger.info("AdaptiveResourceManager initialisiert: %s-%s cores", self.min_cores, self.max_cores)

    def get_cpu_usage(self) -> float:
        """Gibt zurück: current CPU usage percentage."""
        if psutil:
            # §Perf: 5-s-Cache — identische Entscheidungsgrundlage, kein Polling-Sturm.
            _now = time.monotonic()
            if self._cpu_cache_value is not None and (_now - self._cpu_cache_ts) < self._CPU_CACHE_TTL:
                return float(self._cpu_cache_value)
            # interval=None: non-blocking, gibt Wert seit letztem Aufruf zurück
            self._cpu_cache_value = float(psutil.cpu_percent(interval=None))
            self._cpu_cache_ts = _now
            return float(self._cpu_cache_value)
        else:
            return 0.0  # Fallback: keine Überwachung

    def get_memory_usage(self) -> float:
        """Gibt zurück: current system memory usage percentage."""
        if psutil:
            try:
                # §Perf: 5-s-Cache wie bei get_cpu_usage — virtual_memory()
                # kostete im Gate-Profil 11,4 s (390 Aufrufe).
                _now = time.monotonic()
                if self._mem_cache_value is not None and (_now - self._mem_cache_ts) < self._MEM_CACHE_TTL:
                    return float(self._mem_cache_value)
                self._mem_cache_value = float(psutil.virtual_memory().percent)
                self._mem_cache_ts = _now
                return float(self._mem_cache_value)
            except AttributeError:
                return 0.0  # Mock-Objekt ohne .percent (z. B. in Tests)
        else:
            return 0  # Fallback: keine Überwachung

    def get_available_memory_mb(self) -> float:
        """Gibt zurück: available system memory in MB."""
        if psutil:
            try:
                return float(psutil.virtual_memory().available) / (1024 * 1024)
            except AttributeError:
                return float("inf")  # Mock-Objekt ohne .available
        else:
            return float("inf")  # Fallback: assume unlimited

    def start_monitoring(self) -> None:
        """Startet den Hintergrund-Monitor-Thread für Ressourcen-Überwachung."""
        with self.lock:
            if self.running:
                return
            self.running = True
            self._stop_event = threading.Event()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="ARM_monitor")
            self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stoppt den Monitor-Thread und gibt Ressourcen frei."""
        self.running = False
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        thread = self._monitor_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_MONITOR_JOIN_TIMEOUT_S)
        self._monitor_thread = None

    def _monitor_loop(self) -> None:
        stop_event = getattr(self, "_stop_event", None)
        while self.running:
            cpu_usage = self.get_cpu_usage()
            memory_usage = self.get_memory_usage()

            with self.lock:
                # Determine if lightweight mode is needed
                self.use_lightweight = cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold

                # Adjust cores based on CPU usage
                if cpu_usage > self.cpu_threshold:
                    self.current_cores = max(self.min_cores, self.current_cores - 1)
                elif cpu_usage < self.cpu_threshold * 0.7 and self.current_cores < self.max_cores:
                    if memory_usage < self.memory_threshold * 0.8:
                        self.current_cores = min(self.max_cores, self.current_cores + 1)

            # Active eviction when RAM is critically high — outside lock to avoid deadlock
            if memory_usage > self.memory_threshold:
                try:
                    from backend.core.plugin_lifecycle_manager import (
                        evict_stale_plugins,  # pylint: disable=import-outside-toplevel
                    )

                    n_evicted = evict_stale_plugins()
                    if n_evicted > 0:
                        logger.info(
                            "ARM: RAM %.1f%% > %d%% Schwelle — evicted %d plugin(s)",
                            memory_usage,
                            self.memory_threshold,
                            n_evicted,
                        )
                except Exception as evict_exc:
                    logger.debug("ARM: eviction Pruefung fehlgeschlagen: %s", evict_exc)
                gc.collect()

            # Interruptible sleep — stoppt sofort bei stop_monitoring()
            if stop_event is not None:
                stop_event.wait(timeout=self.check_interval)
                if stop_event.is_set():
                    break
            else:
                time.sleep(self.check_interval)

    def get_num_cores(self) -> int:
        """Gibt die aktuell verfügbare Core-Anzahl für ML-Verarbeitung zurück."""
        with self.lock:
            return self.current_cores

    def should_use_lightweight_mode(self) -> bool:
        """
        Prüft if system resources are constrained and lightweight algorithms should be used.

        Returns:
            bool: True if resources are constrained, False otherwise
        """
        with self.lock:
            return self.use_lightweight

    def check_memory_availability(self, required_mb: float) -> bool:
        """
        Prüft if required memory is available.

        Args:
            required_mb: Required memory in MB

        Returns:
            bool: True if memory is available, False otherwise
        """
        available_mb = self.get_available_memory_mb()

        # Add 20% safety margin
        safety_margin = 1.2
        required_with_margin = required_mb * safety_margin

        return available_mb >= required_with_margin


# Modul-Level-Singleton — start_monitoring() wird NICHT automatisch
# aufgerufen, damit Test-Imports keinen blockierenden Hintergrund-Thread starten.
# Produktions-Code (UnifiedRestorerV3 etc.) ruft start_monitoring() explizit auf.
adaptive_resource_manager = AdaptiveResourceManager(
    min_cores=2, max_cores=mp.cpu_count(), check_interval=2.0, cpu_threshold=80, memory_threshold=78
)
