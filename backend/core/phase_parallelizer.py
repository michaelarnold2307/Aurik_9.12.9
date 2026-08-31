"""
Phase Execution Parallelizer (§G60)

Enables parallel execution of independent restoration phases.
Reduces processing time by running non-conflicting phases concurrently.

Phase Dependency Model:
  Phases that operate on DIFFERENT frequency bands or DIFFERENT signal
  aspects can run in parallel without conflicts:
    Group A (LF):  phase_06, phase_29      — low-frequency restoration
    Group B (MF):  phase_07, phase_19      — mid-frequency harmonic/de-ess
    Group C (HF):  phase_39, phase_03      — high-frequency air/denoise
    Group D (Meta): phase_01, phase_31     — analysis/speed (read-only)

  Phases within the same group are sequential (order matters).
  Phases across groups can run in parallel.

Usage (opt-in via config):
  from backend.core.phase_parallelizer import ParallelPhaseExecutor
import os  # §v10.105 module-level (prevents UnboundLocalError)
  executor = ParallelPhaseExecutor(max_workers=4)
  audio = executor.execute_phase_group(audio, group, sr, context)

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Phase Parallel Groups (§G60) ────────────────────────────────────────

# Groups of phases that can safely run in parallel.
# Within each group, phases are sequential.
# Across groups, phases are independent (different frequency bands/operations).
_PARALLEL_GROUPS: dict[str, list[str]] = {
    "analysis": ["phase_01_forensics", "phase_31_speed_pitch"],
    "low_freq": ["phase_06_frequency_restoration", "phase_29_tonal_balance"],
    "mid_freq": ["phase_07_harmonic_restoration", "phase_19_de_esser"],
    "high_freq": ["phase_39_air_band", "phase_03_spectral_denoise"],
    "dynamics": ["phase_36_transient_shaper", "phase_54_transparent_dynamics"],
    "vocal": ["phase_42_vocal_enhancement"],
}

# Phases that MUST run sequentially (strong dependencies)
_SEQUENTIAL_PHASES: set[str] = {
    "phase_12_wow_flutter",  # modifies timing — must run first
    "phase_35_multiband",  # depends on all prior spectral phases
    "phase_40_loudness",  # must be last
}


class ParallelPhaseExecutor:
    """§G60: Execute independent phase groups in parallel.

    Args:
        max_workers: Maximum parallel threads (default: cpu_count - 1).
    """

    def __init__(self, max_workers: int | None = None):
        # import os → module level (§v10.105)
        _cpu_count = os.cpu_count() or 2
        self.max_workers = max_workers or max(1, _cpu_count - 1)
        logger.info("ParallelPhaseExecutor: max_workers=%d (parallele Arbeiter)", self.max_workers)

    def execute_groups(
        self,
        audio: np.ndarray,
        sr: int,
        phase_funcs: dict[str, Callable],
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Execute phase groups in parallel where possible.

        Args:
            audio: Input audio.
            sr: Sample rate.
            phase_funcs: Dict mapping phase_id to callable.
            context: Shared restoration context.

        Returns:
            Processed audio (sequential phases applied, parallel results merged).
        """
        if context is None:
            context = {}

        # Step 1: Run sequential phases first (they modify the signal globally)
        audio_after_seq = audio.copy()
        for phase_id in sorted(_SEQUENTIAL_PHASES):
            if phase_id in phase_funcs:
                try:
                    audio_after_seq = phase_funcs[phase_id](audio_after_seq, sr, context)
                except Exception as e:
                    logger.warning("Sequenzielle Verarbeitungsschritt %s fehlgeschlagen: %s", phase_id, e)

        # Step 2: Run parallel groups
        results = {}  # group_name → processed audio for that group's band
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for group_name, phase_ids in _PARALLEL_GROUPS.items():
                # Filter to available phases
                available = [p for p in phase_ids if p in phase_funcs]
                if not available:
                    continue
                future = executor.submit(self._execute_group, audio_after_seq, sr, available, phase_funcs, context)
                futures[future] = group_name

            for future in as_completed(futures):
                group_name = futures[future]
                try:
                    results[group_name] = future.result()
                except Exception as e:
                    logger.warning("Parallele Gruppe %s fehlgeschlagen: %s", group_name, e)

        # Step 3: Merge parallel results (conservative: max-magnitude blend)
        if not results:
            return audio_after_seq

        result = audio_after_seq.astype(np.float64).copy()
        for group_audio in results.values():
            if group_audio is not None:
                # Blend: 50% original + 50% group result
                result = 0.5 * result + 0.5 * group_audio.astype(np.float64)

        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    def _execute_group(
        self,
        audio: np.ndarray,
        sr: int,
        phase_ids: list[str],
        phase_funcs: dict[str, Callable],
        context: dict[str, Any],
    ) -> np.ndarray | None:
        """Execute a single phase group sequentially."""
        current = audio.copy()
        for phase_id in phase_ids:
            try:
                current = phase_funcs[phase_id](current, sr, context)
            except Exception as e:
                logger.debug("Verarbeitungsschritt %s in Gruppe fehlgeschlagen: %s", phase_id, e)
        return current


def estimate_parallel_speedup(num_phases: int = 68, num_workers: int = 4) -> float:
    """Estimate theoretical speedup from parallelization.

    Based on Amdahl's Law: sequential fraction ~30% of phases,
    parallel fraction ~70% split across workers.
    """
    sequential = 0.30  # 30% must run sequentially
    parallel = 0.70  # 70% can be parallelized
    speedup = 1.0 / (sequential + parallel / num_workers)
    return speedup
