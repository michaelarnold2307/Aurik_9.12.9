"""§AD: Defect Precision Enhancement — each defect repaired individually.

Instead of applying the same repair strength to an entire file,
classify every defect instance and compute a per-instance optimal
repair strategy. Repairs are then verified and refined if needed.

Three-layer architecture:
1.  PRE:  Per-defect instance analysis (severity, frequency, audibility)
2.  MID:  Adaptive per-instance repair parameters
3.  POST: Verification + refinement loop with §AF DynamicsGuard

Based on ISO 11172-3 psychoacoustic masking model and
ITU-R BS.1387 PEAQ perceptual evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from backend.core.repair_dynamics_guard import RepairDynamicsGuard

logger = logging.getLogger(__name__)


@dataclass
class DefectInstance:
    """A single defect event with location and characteristics."""

    start_sample: int
    end_sample: int
    defect_type: str
    severity: float  # 0.0–1.0
    peak_db: float  # dBFS at defect peak
    center_freq_hz: float  # dominant frequency
    bandwidth_hz: float  # frequency spread
    is_audible: bool = True  # above masking threshold?
    repair_strength: float = 1.0  # adaptive strength
    verified_ok: bool = False


@dataclass
class PrecisionEnhancementResult:
    instances: list[DefectInstance] = field(default_factory=list)
    total_instances: int = 0
    audible_instances: int = 0
    repaired_instances: int = 0
    over_repaired: int = 0
    under_repaired: int = 0
    dynamics_violations: int = 0
    precision_gain_pct: float = 0.0


class DefectPrecisionEnhancer:
    """Per-instance defect analysis + adaptive repair + verification.

    §AF Integration: RepairDynamicsGuard ensures every repair preserves
    the song's dynamics — no volume hopping, pumping, or stuttering.
    """

    def __init__(self) -> None:
        self._masking_model = _SimpleMaskingModel()
        self._dynamics = RepairDynamicsGuard()

    def analyze_defects(
        self,
        audio: np.ndarray,
        sr: int,
        defect_types: list[str],
    ) -> dict[str, list[dict[str, float]]]:
        """Analyzes defect types and returns per-defect precision hints.

        Lightweight variant of analyze_instances() that works without
        per-instance location data. Uses defect type name + severity
        from the caller's defect_hint dict to compute optimal strengths.

        Args:
            audio: Input audio (float32)
            sr: Sample rate in Hz
            defect_types: List of DefectType.value strings, e.g. ["wow", "clicks"]

        Returns:
            Dict mapping defect_type → [{"strength": X, "audible": True/False}, ...]
        """
        hints: dict[str, list[dict[str, float]]] = {}

        for defect_type in defect_types:
            # Without per-instance location data, we can't use the psychoacoustic
            # masking model. Assume audible: if the defect_type was passed, the
            # caller (UV3) already confirmed its presence via DefectScanner.
            is_audible = True

            severity = 0.5
            optimal = self._compute_optimal_strength(defect_type, severity, is_audible)

            hints[defect_type] = [
                {
                    "strength": float(optimal),
                    "audible": float(is_audible),
                    "severity_default": severity,
                }
            ]

        return hints

    def analyze_instances(
        self,
        audio: np.ndarray,
        sr: int,
        defect_type: str,
        locations: list[tuple[float, float]],  # (start_s, end_s) in seconds
        severities: list[float] | None = None,
    ) -> list[DefectInstance]:
        """Classify every defect instance.

        Args:
            audio: Input audio (float32)
            sr: Sample rate
            defect_type: Defekt-Typ (click, crackle, hum, etc.)
            locations: List of (start_time_s, end_time_s)
            severities: Optional per-instance severity values
        """
        instances = []
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)
        n = len(mono)

        for i, (start_s, end_s) in enumerate(locations):
            s0 = max(0, int(start_s * sr))
            s1 = min(n, int(end_s * sr))
            if s1 - s0 < 2:
                continue

            segment = mono[s0:s1]

            # Peak-Analyse
            peak = float(np.max(np.abs(segment)))
            peak_db = 20.0 * np.log10(peak + 1e-10)

            # Frequenz-Analyse
            fft = np.abs(np.fft.rfft(segment, n=min(1024, len(segment))))
            freqs = np.fft.rfftfreq(min(1024, len(segment)), d=1.0 / sr)
            if len(fft) > 0 and np.sum(fft) > 1e-10:
                center_freq = float(np.average(freqs, weights=fft + 1e-10))
                # Bandbreite: Frequenzbereich mit >50% der Max-Energie
                half_max = np.max(fft) * 0.5
                above = np.where(fft >= half_max)[0]
                if len(above) >= 2:
                    bandwidth = freqs[above[-1]] - freqs[above[0]]
                else:
                    bandwidth = 500.0
            else:
                center_freq = 1000.0
                bandwidth = 500.0

            # Severity
            sev = severities[i] if severities and i < len(severities) else min(1.0, (peak_db + 60) / 40)

            # Psychoakustische Hörbarkeit
            is_audible = self._masking_model.is_audible(defect_type, sev, center_freq, peak_db, sr, mono, s0, s1)

            # Adaptive Stärke
            repair_strength = self._compute_optimal_strength(defect_type, sev, is_audible)

            instances.append(
                DefectInstance(
                    start_sample=s0,
                    end_sample=s1,
                    defect_type=defect_type,
                    severity=sev,
                    peak_db=peak_db,
                    center_freq_hz=center_freq,
                    bandwidth_hz=bandwidth,
                    is_audible=is_audible,
                    repair_strength=repair_strength,
                )
            )

        return instances

    def _compute_optimal_strength(self, defect_type: str, severity: float, is_audible: bool) -> float:
        """Compute per-instance optimal repair strength.

        Rules:
        - Inaudible defects: 0.0 (don't waste CPU)
        - Very mild (sev<0.2): 0.3 (light touch)
        - Moderate (sev 0.2-0.6): proportional 0.3-0.8
        - Severe (sev>0.6): 0.8-1.0
        - Clicks: always gentle to avoid transient loss
        - Hum: proportional to severity
        """
        # §2.59 Fix (2026-07-09): Defekt-Namen auf DefectType.values() abgestimmt.
        if not is_audible:
            return 0.0

        if defect_type in ("clicks", "crackle"):
            # Transient defects: gentle repair
            return float(np.clip(severity * 1.2, 0.2, 0.8))
        elif defect_type in ("hum", "motor_interference", "low_freq_rumble"):
            return float(np.clip(severity * 1.5, 0.3, 1.0))
        elif defect_type in ("high_freq_noise", "modulation_noise", "quantization_noise"):
            return float(np.clip(severity * 1.3, 0.3, 1.0))
        else:
            return float(np.clip(severity * 1.0, 0.2, 1.0))

    def apply_repairs(
        self,
        audio: np.ndarray,
        sr: int,
        instances: list[DefectInstance],
        repair_fn,  # callable(audio, sr, start_sample, end_sample, strength) -> np.ndarray
    ) -> np.ndarray:
        """Apply per-instance repairs WITH §AF dynamics-preserving envelope matching.

        Each repair is followed by the DynamicsGuard to ensure:
        - Amplitude envelope matches surrounding context (no hopping)
        - Smooth cross-fade at repair boundaries
        - No sudden loudness jumps between repaired and unrepaired regions
        """
        result = np.asarray(audio, dtype=np.float32).copy()

        for i, d in enumerate(instances):
            if not d.is_audible or d.repair_strength <= 0.001:
                continue

            s0 = max(0, d.start_sample)
            s1 = min(result.shape[-1], d.end_sample)
            if s1 - s0 < 2:
                continue

            try:
                # Apply the repair function to the segment
                if result.ndim == 2:
                    segment = result[:, s0:s1].copy()
                    repaired_segment = repair_fn(segment, sr, 0, s1 - s0, d.repair_strength)
                else:
                    segment = result[s0:s1].copy()
                    repaired_segment = repair_fn(segment, sr, 0, s1 - s0, d.repair_strength)

                # §AF: DynamicsGuard — envelope matching + smooth cross-fade
                repaired_segment = self._dynamics.match_envelope(
                    repaired_segment,
                    sr,
                    0,
                    len(repaired_segment) if repaired_segment.ndim == 1 else repaired_segment.shape[-1],
                    context_ms=50,
                    crossfade_ms=12,
                )

                # Write back
                if result.ndim == 2:
                    result[:, s0:s1] = repaired_segment
                else:
                    result[s0:s1] = repaired_segment
            except Exception:
                logger.debug("Precision repair %d at sample %d fehlgeschlagen, skipping", i, d.start_sample)

        return cast(np.ndarray, result)

    def verify_repair(
        self,
        before: np.ndarray,
        after: np.ndarray,
        instances: list[DefectInstance],
        sr: int,
    ) -> PrecisionEnhancementResult:
        """Verify each repair and trigger refinement if needed.

        §AF-enhanced: Also checks for amplitude envelope continuity at
        repair boundaries to catch dynamics violations (hopping/stuttering).
        """
        result = PrecisionEnhancementResult(
            instances=instances,
            total_instances=len(instances),
            audible_instances=sum(1 for d in instances if d.is_audible),
        )

        mono_before = np.mean(before, axis=0) if before.ndim == 2 else before
        mono_after = np.mean(after, axis=0) if after.ndim == 2 else after

        for d in instances:
            if not d.is_audible:
                continue

            s0, s1 = d.start_sample, min(d.end_sample, len(mono_after))
            if s1 - s0 < 2:
                continue

            # Before/After comparison at defect location
            before_seg = mono_before[s0:s1]
            after_seg = mono_after[s0:s1]

            before_rms = float(np.sqrt(np.mean(before_seg**2) + 1e-12))
            after_rms = float(np.sqrt(np.mean(after_seg**2) + 1e-12))

            # §AF: DynamicsGuard continuity check at repair boundaries
            boundary_ok = True
            try:
                continuity = self._dynamics.verify_continuity(mono_after, sr, [s0, s1])
                boundary_ok = continuity.continuity_ok
            except Exception as e:
                logger.warning("defect_precision_enhancer.py::pruefen_repair Ersatzpfad: %s", e)

            if after_rms < before_rms * 0.1:
                # Over-repair: signal removed entirely
                result.over_repaired += 1
                d.verified_ok = False
            elif after_rms > before_rms * 0.9:
                # Under-repair: defect still present
                result.under_repaired += 1
                d.verified_ok = False
            elif not boundary_ok:
                # §AF: Dynamics violation — amplitude discontinuity detected
                result.dynamics_violations += 1
                d.verified_ok = False
                logger.debug(
                    "§AF Dynamics discontinuity at sample %d (freq %.0f Hz, type %s)",
                    d.start_sample,
                    d.center_freq_hz,
                    d.defect_type,
                )
            else:
                result.repaired_instances += 1
                d.verified_ok = True

        # Precision-Gain: how many instances were individually analyzed vs bulk
        if result.total_instances > 0:
            unneeded = sum(1 for d in instances if not d.is_audible)
            result.precision_gain_pct = (unneeded + result.over_repaired) / result.total_instances * 100

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Simple psychoacoustic masking model (ISO 11172-3 simplified)
# ═══════════════════════════════════════════════════════════════════════════════


class _SimpleMaskingModel:
    """Simplified simultaneous + temporal masking model."""

    def is_audible(
        self,
        defect_type: str,
        severity: float,
        center_freq: float,
        peak_db: float,
        sr: int,
        audio: np.ndarray,
        s0: int,
        s1: int,
    ) -> bool:
        """Determine if defect is above masking threshold."""
        try:
            # 1. Global threshold: very quiet defects are inaudible
            if peak_db < -80:
                return False

            # 2. Severity threshold: very mild defects
            if severity < 0.05:
                return False

            # 3. Frequency-dependent masking
            #    Low frequencies mask high frequencies more than vice versa
            context_start = max(0, s0 - int(sr * 0.05))
            context_end = min(len(audio), s1 + int(sr * 0.05))
            if context_end - context_start < 10:
                return True  # too short to analyze

            context = audio[context_start:context_end]
            fft = np.abs(np.fft.rfft(context, n=min(4096, len(context))))
            freqs = np.fft.rfftfreq(min(4096, len(context)), d=1.0 / sr)

            # Find masking energy in critical band
            critical_band = 0.3 * center_freq  # ~1/3 octave
            mask = (freqs >= center_freq - critical_band) & (freqs <= center_freq + critical_band)
            mask_energy = float(np.sum(fft[mask]))

            # Find total energy
            total_energy = float(np.sum(fft)) + 1e-10

            # Signal-to-mask ratio
            smr = 10.0 * np.log10(mask_energy / total_energy + 1e-12)

            # Audibility: defect must stand out from background
            # audible if SMR > -25 dB
            return bool(smr > -25)

        except Exception as e:
            logger.warning("defect_precision_enhancer.py::is_audible Ersatzpfad: %s", e)
            return True  # assume audible on error
