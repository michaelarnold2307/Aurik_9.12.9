"""§v10.17 PipelineHealthMonitor — Error-Budget + Circuit-Breaker.

Verhindert pathologische Pipeline-Läufe durch globales Retry- und Zeit-Limit.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TOTAL_RETRIES: int = 60  # Global: max 60 retries across all phases
_MAX_PHASE_DURATION_S: float = 300.0  # 5 min per phase max
_MAX_PIPELINE_DURATION_S: float = 7200.0  # 2 hours total max
_CIRCUIT_BREAKER_FAILURES: int = 8  # >8 hard failures → abort pipeline


@dataclass
class PipelineHealth:
    total_phases: int = 0
    total_retries: int = 0
    total_failures: int = 0
    total_hard_failures: int = 0
    phases_skipped: int = 0
    phases_pss_rejected: int = 0
    pipeline_start_time: float = 0.0
    circuit_breaker_triggered: bool = False
    error_log: list[dict[str, Any]] = field(default_factory=list)
    phase_durations: dict[str, float] = field(default_factory=dict)


class PipelineHealthMonitor:
    """Globales Health-Monitoring mit Circuit-Breaker."""

    def __init__(self):
        self._health = PipelineHealth(pipeline_start_time=time.time())
        self._lock = threading.Lock()

    def record_phase_start(self, phase_id: str) -> float:
        return time.time()

    def record_phase_end(self, phase_id: str, start_time: float, retries: int, success: bool, error_type: str = ""):
        with self._lock:
            self._health.total_phases += 1
            self._health.total_retries += retries
            dur = time.time() - start_time
            self._health.phase_durations[phase_id] = dur
            if not success:
                self._health.total_failures += 1
                if error_type:
                    self._health.total_hard_failures += int(error_type == "hard")
                self._health.error_log.append(
                    {"phase": phase_id, "retries": retries, "error": error_type, "duration_s": round(dur, 2)}
                )
            if dur > _MAX_PHASE_DURATION_S:
                logger.warning("HealthMonitor: %s exceeded Verarbeitungsschritt time limit (%.0fs)", phase_id, dur)

    def check_circuit_breaker(self) -> bool:
        """True = OK to continue. False = abort pipeline."""
        with self._lock:
            if self._health.total_retries > _MAX_TOTAL_RETRIES:
                self._health.circuit_breaker_triggered = True
                logger.error("CIRCUIT BREAKER: %d retries > %d limit", self._health.total_retries, _MAX_TOTAL_RETRIES)
                return False
            if self._health.total_hard_failures > _CIRCUIT_BREAKER_FAILURES:
                self._health.circuit_breaker_triggered = True
                logger.error(
                    "CIRCUIT BREAKER: %d hard failures > %d limit",
                    self._health.total_hard_failures,
                    _CIRCUIT_BREAKER_FAILURES,
                )
                return False
            if time.monotonic() - self._health.pipeline_start_time > _MAX_PIPELINE_DURATION_S:
                self._health.circuit_breaker_triggered = True
                logger.error("CIRCUIT BREAKER: pipeline time exceeded %.0fh limit", _MAX_PIPELINE_DURATION_S / 3600)
                return False
            return True

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_phases": self._health.total_phases,
                "total_retries": self._health.total_retries,
                "total_failures": self._health.total_failures,
                "hard_failures": self._health.total_hard_failures,
                "phases_skipped": self._health.phases_skipped,
                "pss_rejected": self._health.phases_pss_rejected,
                "circuit_breaker": self._health.circuit_breaker_triggered,
                "pipeline_duration_s": round(time.monotonic() - self._health.pipeline_start_time, 1),
                "errors": self._health.error_log[-10:],
            }


_instance: PipelineHealthMonitor | None = None
_lock = threading.Lock()


def get_health_monitor() -> PipelineHealthMonitor:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PipelineHealthMonitor()
    return _instance
