"""
backend/core/dsp/cumulative_hallucination_tracker.py — §CHT-1
=============================================================
Cross-phase cumulative spectral novelty tracking for Aurik 10.0.0.

The per-phase hallucination_guard.py catches single-phase synthetic additions.
But five phases each adding 3 % synthetic energy are individually "safe" yet
collectively 15 % — enough to audibly colour the restoration.

§CHT-1 tracks the running sum of spectral novelty across ALL phases in one UV3
run and triggers a graduated alarm system:

    "ok"       cumulative novelty ≤ WARN threshold
    "warn"     WARN < novelty ≤ CRITICAL — log warning, no rollback
    "critical" novelty > CRITICAL — UV3 rolls back to checkpoint where novelty
               first exceeded WARN, then re-runs the remaining phases at 50 %
               of their original strength

Thresholds (Restoration / Studio 2026):
    Restoration:  WARN=0.20  CRITICAL=0.35
    Studio 2026:  WARN=0.30  CRITICAL=0.50

Spectral novelty per phase: cosine distance of normalised log-magnitude spectra
before and after the phase, averaged across 3 analysis windows.

Integration (UV3):
    tracker = get_cumulative_hallucination_tracker()
    tracker.reset(mode="restoration")
    # After every phase in _profiled_phase_call_with_delta:
    level = tracker.record_phase(phase_id, pre_audio, post_audio, sr)
    if level == "critical":
        uv3._cumulative_novelty_rollback(tracker.rollback_checkpoint)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alarm thresholds per mode
# ---------------------------------------------------------------------------

_THRESHOLDS: dict[str, tuple[float, float]] = {
    "restoration": (0.20, 0.35),
    "studio": (0.30, 0.50),
}

AlarmLevel = Literal["ok", "warn", "critical"]

# ---------------------------------------------------------------------------
# Per-phase record
# ---------------------------------------------------------------------------


@dataclass
class PhaseNoveltyRecord:
    """Novelty data for one phase in the pipeline."""

    phase_id: str
    novelty_delta: float  # contribution of this phase to the cumulative total
    cumulative_after: float  # cumulative novelty after this phase was applied
    alarm_level: AlarmLevel  # alarm level AFTER this phase


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class CumulativeHallucinationTracker:
    """Cross-phase cumulative spectral novelty tracker (§CHT-1).

    Thread-safe (per-instance lock).  Designed to be reset once per UV3 run.
    """

    # Number of analysis windows for per-phase novelty estimation
    _N_WINDOWS: int = 3
    # FFT size for spectral analysis
    _N_FFT: int = 2048

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode: str = "restoration"
        self._cumulative: float = 0.0
        self._records: list[PhaseNoveltyRecord] = []
        # Index into _records where novelty first exceeded WARN
        self._warn_checkpoint_idx: int | None = None
        # §3 Hard-Limit nach Phase 5: alle folgenden ML-Phasen bypassen
        self._phase_count: int = 0
        self._hard_limit_reached: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, mode: str = "restoration") -> None:
        """Setzt zurück: for a new UV3 run.

        Args:
            mode: "restoration" or "studio" (controls thresholds).
        """
        with self._lock:
            self._mode = mode if mode in _THRESHOLDS else "restoration"
            self._cumulative = 0.0
            self._records = []
            self._warn_checkpoint_idx = None
            self._phase_count = 0
            self._hard_limit_reached = False
            logger.debug("§CHT-1 CumulativeHallucinationTracker zurueckgesetzt: Betriebsart=%s", self._mode)

    # ------------------------------------------------------------------
    # Phase recording
    # ------------------------------------------------------------------

    def record_phase(
        self,
        phase_id: str,
        pre_audio: np.ndarray,
        post_audio: np.ndarray,
        sr: int,
    ) -> AlarmLevel:
        """Record one phase's spectral novelty contribution.

        Args:
            phase_id:   Phase identifier string (e.g. "phase_07_harmonic_exc")
            pre_audio:  Audio array BEFORE the phase (channels-last)
            post_audio: Audio array AFTER the phase (channels-last)
            sr:         Sample rate (48 000 Hz)

        Returns:
            Current alarm level after this phase: "ok" | "warn" | "critical"
        """
        with self._lock:
            self._phase_count += 1
            delta = self._compute_novelty(pre_audio, post_audio, sr)
            self._cumulative = float(np.clip(self._cumulative + delta, 0.0, 1.0))
            level = self._alarm_level(self._cumulative)

            # §3 Hard-Limit nach Phase 5: wenn cumulative > CRITICAL → alle folgenden ML-Phasen bypassen
            warn_t, crit_t = _THRESHOLDS[self._mode]
            if self._phase_count >= 5 and self._cumulative > crit_t and not self._hard_limit_reached:
                self._hard_limit_reached = True
                logger.warning(
                    "§CHT-1 Hard-Limit nach Phase %d: cumulative=%.4f > CRITICAL=%.2f — ML-Phasen bypassen",
                    self._phase_count,
                    self._cumulative,
                    crit_t,
                )

            # Record first WARN crossing (rollback checkpoint)
            if level in ("warn", "critical") and self._warn_checkpoint_idx is None:
                # The checkpoint is the phase BEFORE this one
                self._warn_checkpoint_idx = max(0, len(self._records) - 1)

            rec = PhaseNoveltyRecord(
                phase_id=phase_id,
                novelty_delta=float(delta),
                cumulative_after=float(self._cumulative),
                alarm_level=level,
            )
            self._records.append(rec)

            if level != "ok":
                logger.debug(
                    "§CHT-1 Verarbeitungsschritt=%s delta=%.4f cumulative=%.4f alarm=%s",
                    phase_id,
                    delta,
                    self._cumulative,
                    level,
                )
            return level

    # ------------------------------------------------------------------
    # Hard-Limit für ML-Phasen (nach Phase 5)
    # ------------------------------------------------------------------

    def should_bypass_ml_phase(self) -> bool:
        """Prüft, ob nach Phase 5 alle folgenden ML-Phasen bypassed werden sollen.

        §3: Wenn cumulative_novelty > CRITICAL nach Phase 5 → Hard-Limit erreicht.
        Verhindert kumulative „AI-Glaze" durch mehrere ML-Phasen.

        Returns:
            True wenn ML-Phase übersprungen werden soll.
        """
        with self._lock:
            return self._hard_limit_reached

    @property
    def phase_count(self) -> int:
        """Anzahl der bisher aufgerufenen Phasen."""
        with self._lock:
            return self._phase_count

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def check_alarm(self) -> AlarmLevel:
        """Gibt current alarm level without recording a new phase zurück."""
        with self._lock:
            return self._alarm_level(self._cumulative)

    @property
    def cumulative_novelty(self) -> float:
        """Current cumulative spectral novelty (0..1)."""
        with self._lock:
            return self._cumulative

    @property
    def rollback_checkpoint(self) -> int | None:
        """Index of the phase where WARN was first triggered.

        UV3 should roll back to the audio state BEFORE this phase.
        None if no WARN has been triggered yet.
        """
        with self._lock:
            return self._warn_checkpoint_idx

    def get_report(self) -> dict:
        """Gibt a diagnostic report of all recorded phase novelties zurück."""
        with self._lock:
            return {
                "mode": self._mode,
                "cumulative_novelty": float(self._cumulative),
                "alarm_level": self._alarm_level(self._cumulative),
                "warn_checkpoint_idx": self._warn_checkpoint_idx,
                "phase_count": self._phase_count,
                "hard_limit_reached": self._hard_limit_reached,
                "phases": [
                    {
                        "phase_id": r.phase_id,
                        "novelty_delta": round(r.novelty_delta, 5),
                        "cumulative_after": round(r.cumulative_after, 5),
                        "alarm_level": r.alarm_level,
                    }
                    for r in self._records
                ],
                "thresholds": {
                    "warn": _THRESHOLDS[self._mode][0],
                    "critical": _THRESHOLDS[self._mode][1],
                },
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alarm_level(self, cumulative: float) -> AlarmLevel:
        warn_t, crit_t = _THRESHOLDS[self._mode]
        if cumulative > crit_t:
            return "critical"
        if cumulative > warn_t:
            return "warn"
        return "ok"

    def _compute_novelty(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,  # pylint: disable=unused-argument
    ) -> float:
        """Berechnet spectral novelty as cosine distance of log-magnitude spectra.

        Averaged over _N_WINDOWS windows drawn from different positions in the
        audio to reduce sensitivity to local artefacts.

        Returns:
            novelty in [0, 1]  (0 = identical spectra, 1 = completely different)
        """
        pre_a = np.nan_to_num(np.asarray(pre, dtype=np.float32))
        post_a = np.nan_to_num(np.asarray(post, dtype=np.float32))

        # Collapse to mono
        pre_m = pre_a if pre_a.ndim == 1 else pre_a.mean(axis=1)
        post_m = post_a if post_a.ndim == 1 else post_a.mean(axis=1)

        n = min(len(pre_m), len(post_m))
        if n < self._N_FFT:
            return 0.0

        n_fft = self._N_FFT
        novelties: list[float] = []

        for wi in range(self._N_WINDOWS):
            # Spread windows across the signal
            center = int(n * (wi + 1) / (self._N_WINDOWS + 1))
            start = max(0, center - n_fft // 2)
            end = start + n_fft
            if end > n:
                start = max(0, n - n_fft)
                end = n

            seg_pre = pre_m[start:end]
            seg_post = post_m[start:end]

            if len(seg_pre) < n_fft or len(seg_post) < n_fft:
                continue

            # Log-magnitude spectra
            window = np.hanning(n_fft)
            spec_pre = np.abs(np.fft.rfft(seg_pre * window)).astype(np.float32)
            spec_post = np.abs(np.fft.rfft(seg_post * window)).astype(np.float32)

            # Log-compress (perceptual)
            log_pre = np.log1p(spec_pre)
            log_post = np.log1p(spec_post)

            # Cosine distance
            norm_pre = np.linalg.norm(log_pre) + 1e-8
            norm_post = np.linalg.norm(log_post) + 1e-8
            cosine_sim = float(np.dot(log_pre / norm_pre, log_post / norm_post))
            novelties.append(float(np.clip(1.0 - cosine_sim, 0.0, 1.0)))

        if not novelties:
            return 0.0
        return float(np.mean(novelties))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: CumulativeHallucinationTracker | None = None
_lock = threading.Lock()


def get_cumulative_hallucination_tracker() -> CumulativeHallucinationTracker:
    """Thread-safe singleton (double-checked locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CumulativeHallucinationTracker()
    return _instance


__all__ = [
    "AlarmLevel",
    "CumulativeHallucinationTracker",
    "PhaseNoveltyRecord",
    "get_cumulative_hallucination_tracker",
]
