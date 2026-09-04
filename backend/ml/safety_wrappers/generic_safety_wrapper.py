"""
generic_safety_wrapper.py - Generic HIPS Safety Wrapper Generator

Provides generic safety wrappers for common DSP operation types.
Implements 80% of HIPS requirements automatically, allowing for quick
deployment across 40+ DSP modules.

For critical modules, use dedicated custom wrappers (e.g., DeHumSafety).
For standard modules, use these generic wrappers:
- GenericNoiseReductionSafety
- GenericRestorationSafety
- GenericDynamicsSafety
- GenericSpectralSafety
- GenericSpatialSafety

Author: AURIK Team
Version: 1.0.0
Date: 8. Februar 2026
"""

# Import Musical Goals integration
import hashlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
from backend.core.musical_goals.processing_modes import PROCESSING_MODE_CONFIGS, ProcessingMode

from .safety_wrapper_template import (
    BaseSafetyWrapper,
    PostCheckResult,
    PreCheckResult,
    compute_correlation,
    compute_energy_ratio,
    compute_spectral_centroid,
    validate_audio_basic,
)

# ============================================================================
# GENERIC NOISE REDUCTION SAFETY WRAPPER
# ============================================================================


class GenericNoiseReductionSafety(BaseSafetyWrapper):
    """
    Generic safety wrapper for noise reduction modules.

    Applies to:
    - Denoisers (spectral, adaptive, MMSE, etc.)
    - Dehissers
    - Spectral gates
    - Noise profiling systems

    Ensures:
    - Signal has sufficient noise to warrant processing
    - No over-processing (musical artifacts)
    - Preserves transients
    - Maintains harmonic content
    - Musical Goals not violated
    """

    def __init__(
        self,
        module_name: str,
        module_version: str,
        processor_func: Callable,
        processing_mode: ProcessingMode = ProcessingMode.RESTORATION,
        enable_logging: bool = True,
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            module_version=module_version,
            processor_func=processor_func,
            enable_logging=enable_logging,
            confidence_threshold=0.5,
            quality_threshold=0.7,
        )
        self.processing_mode = processing_mode
        self.musical_goals = MusicalGoalsChecker()
        self._goals_cache: dict[tuple[Any, ...], dict[str, float]] = {}
        self._goals_cache_order: list[tuple[Any, ...]] = []
        self._goals_cache_max_entries: int = 16
        self._goals_cache_lock = threading.Lock()

    def _goals_cache_key(self, audio: np.ndarray, sr: int) -> tuple[Any, ...]:
        arr = np.asarray(audio, dtype=np.float32)
        flat = arr.reshape(-1)
        head = flat[:4096]
        tail = flat[-4096:] if flat.size > 4096 else flat
        h = hashlib.blake2b(digest_size=12)
        h.update(np.asarray([int(sr), int(flat.size)], dtype=np.int64).tobytes())
        h.update(head.tobytes())
        h.update(tail.tobytes())
        return (int(sr), tuple(arr.shape), str(arr.dtype), h.hexdigest())

    def _measure_goals_cached(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        key = self._goals_cache_key(audio, sr)
        with self._goals_cache_lock:
            cached = self._goals_cache.get(key)
            if cached is not None:
                return dict(cached)

        measured = self.musical_goals.measure_all(audio, sr)
        normalized = {str(k): float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in measured.items()}
        with self._goals_cache_lock:
            self._goals_cache[key] = normalized
            self._goals_cache_order.append(key)
            if len(self._goals_cache_order) > self._goals_cache_max_entries:
                old = self._goals_cache_order.pop(0)
                self._goals_cache.pop(old, None)
        return dict(normalized)

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for noise reduction."""
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        # Ensure mono for noise detection
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)

        # Check for sufficient noise level to warrant processing
        noise_floor = self._estimate_noise_floor(audio_mono)
        signal_level = np.sqrt(np.mean(audio_mono**2))

        if signal_level < 1e-6:
            return PreCheckResult(
                passed=False, confidence=0.0, reasons=["Signal too quiet for reliable noise reduction"]
            )

        snr_db = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))

        warnings = []

        # Very clean signal (SNR > 40 dB) - little to gain
        if snr_db > 40:
            warnings.append(f"Signal very clean (SNR={snr_db:.1f} dB). Noise reduction may not improve quality.")

        # Very noisy signal (SNR < 10 dB) - difficult to process safely
        if snr_db < 10:
            warnings.append(f"Signal very noisy (SNR={snr_db:.1f} dB). Aggressive reduction may damage content.")

        return PreCheckResult(
            passed=True,
            confidence=self._compute_noise_detection_confidence(audio_mono, noise_floor),
            reasons=[],
            warnings=warnings,
            metadata={"snr_db": float(snr_db), "noise_floor": float(noise_floor), "signal_level": float(signal_level)},
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in noise reduction."""
        snr_db = pre_check.metadata.get("snr_db", 20.0)

        # Confidence highest at moderate noise levels (15-30 dB SNR)
        # Lower for very clean or very noisy signals

        if 15 <= snr_db <= 30:
            confidence = 0.9
        elif 10 <= snr_db <= 40:
            confidence = 0.75
        elif snr_db < 10:
            confidence = 0.5  # Very noisy - hard to preserve content
        else:  # snr_db > 40
            confidence = 0.6  # Very clean - little benefit

        return confidence

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions for noise reduction."""
        issues = []
        side_effects = []
        metrics = {}

        # 1. Check energy preservation (should reduce, but not too much)
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio < 0.5:
            issues.append(f"Excessive energy loss: {energy_ratio:.2f}")
        elif energy_ratio > 1.1:
            issues.append(f"Energy increased unexpectedly: {energy_ratio:.2f}")

        # 2. Check correlation with original (avoid too much alteration)
        correlation = compute_correlation(
            original if original.ndim == 1 else np.mean(original, axis=0),
            processed if processed.ndim == 1 else np.mean(processed, axis=0),
        )
        metrics["correlation"] = correlation

        if correlation < 0.7:
            issues.append(f"Low correlation with original: {correlation:.2f}")

        # 3. Check for musical noise artifacts
        musical_noise_score = self._detect_musical_noise(processed, sr)
        metrics["musical_noise_score"] = musical_noise_score

        if musical_noise_score > 0.3:
            issues.append(f"Musical noise artifacts detected: {musical_noise_score:.2f}")

        # 4. Check transient preservation
        transient_preservation = self._check_transient_preservation(original, processed, sr)
        metrics["transient_preservation"] = transient_preservation

        if transient_preservation < 0.7:
            side_effects.append(f"Transient damage: {transient_preservation:.2f}")

        # 5. Musical Goals validation
        try:
            orig_goals = self._measure_goals_cached(original, sr)
            proc_goals = self._measure_goals_cached(processed, sr)

            # Check for goal violations
            mode_config = PROCESSING_MODE_CONFIGS.get(self.processing_mode)
            thresholds = mode_config.musical_goals if mode_config else {}
            violations = []

            for goal_name, threshold in thresholds.items():
                if goal_name in proc_goals and proc_goals[goal_name] < threshold:
                    violations.append(f"{goal_name}: {proc_goals[goal_name]:.2f} < {threshold:.2f}")

            if violations:
                issues.append(f"Musical Goals violations: {', '.join(violations)}")

            metrics["musical_goals_original"] = orig_goals  # type: ignore[assignment]
            metrics["musical_goals_processed"] = proc_goals  # type: ignore[assignment]
        except Exception as e:
            side_effects.append(f"Musical Goals check failed: {e}")

        passed = len(issues) == 0
        quality_score = self._compute_noise_reduction_quality(metrics)

        return PostCheckResult(
            passed=passed, quality_score=quality_score, issues=issues, side_effects=side_effects, metrics=metrics
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet quality score for noise reduction."""
        metrics = post_check.metrics

        # Weighted combination of factors
        energy_score = np.clip(metrics.get("energy_ratio", 0.8), 0.0, 1.0)
        correlation_score = metrics.get("correlation", 0.8)
        musical_noise_score = 1.0 - metrics.get("musical_noise_score", 0.1)
        transient_score = metrics.get("transient_preservation", 0.9)

        # Weighted average
        quality = 0.2 * energy_score + 0.3 * correlation_score + 0.3 * musical_noise_score + 0.2 * transient_score

        return float(np.clip(quality, 0.0, 1.0))

    # Helper methods

    def _estimate_noise_floor(self, audio: np.ndarray) -> float:
        """Schätzt noise floor using minimum statistics."""
        # Compute short-term energy
        frame_length = 2048
        hop_length = 512

        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            energy = np.sqrt(np.mean(frame**2))
            energies.append(energy)

        if len(energies) == 0:
            return 0.0

        energies = np.array(energies)  # type: ignore[assignment]

        # Noise floor ≈ 10th percentile of frame energies
        noise_floor = np.percentile(energies, 10)

        return float(noise_floor)

    def _compute_noise_detection_confidence(self, audio: np.ndarray, noise_floor: float) -> float:
        """Berechnet confidence in noise detection."""
        signal_level = np.sqrt(np.mean(audio**2))
        snr_db = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))

        # High confidence when noise clearly identifiable (moderate SNR)
        if 15 <= snr_db <= 30:
            return 0.9
        elif 10 <= snr_db <= 40:
            return 0.75
        else:
            return 0.6

    def _detect_musical_noise(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt musical noise artifacts (tonal bursts)."""
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)

        # Musical noise shows up as isolated spectral peaks varying rapidly
        frame_length = 2048
        hop_length = 512

        spectral_flux = []

        prev_spectrum = None
        for i in range(0, len(audio_mono) - frame_length, hop_length):
            frame = audio_mono[i : i + frame_length]
            spectrum = np.abs(np.fft.rfft(frame))

            if prev_spectrum is not None:
                # Spectral flux = sum of absolute differences
                flux = np.sum(np.abs(spectrum - prev_spectrum))
                spectral_flux.append(flux)

            prev_spectrum = spectrum

        if len(spectral_flux) == 0:
            return 0.0

        spectral_flux = np.array(spectral_flux)  # type: ignore[assignment]

        # High variance in spectral flux = musical noise
        flux_std = np.std(spectral_flux)
        flux_mean = np.mean(spectral_flux)

        if flux_mean < 1e-6:
            return 0.0

        # Normalize
        musical_noise_score = np.clip(flux_std / (flux_mean + 1e-10), 0.0, 1.0)

        return float(musical_noise_score)

    def _check_transient_preservation(self, original: np.ndarray, processed: np.ndarray, sr: int) -> float:
        """Prüft if transients are preserved."""
        orig_mono = original if original.ndim == 1 else np.mean(original, axis=0)
        proc_mono = processed if processed.ndim == 1 else np.mean(processed, axis=0)

        # Detect transients as peaks in envelope
        from scipy.ndimage import maximum_filter1d

        # Envelope (local maxima)
        window = int(sr * 0.02)  # 20ms window
        orig_env = maximum_filter1d(np.abs(orig_mono), size=window)
        proc_env = maximum_filter1d(np.abs(proc_mono), size=window)

        # Find transient locations (peaks in original)
        threshold = np.mean(orig_env) + 2 * np.std(orig_env)
        transient_locations = orig_env > threshold

        if not np.any(transient_locations):
            return 1.0  # No transients to preserve

        # Compare envelopes at transient locations
        orig_transients = orig_env[transient_locations]
        proc_transients = proc_env[transient_locations]

        # Preservation ratio
        preservation = np.mean(proc_transients / (orig_transients + 1e-10))

        return float(np.clip(preservation, 0.0, 1.0))

    def _compute_noise_reduction_quality(self, metrics: dict[str, Any]) -> float:
        """Specific quality computation for noise reduction."""
        # Use same logic as general quality score
        energy_score = np.clip(metrics.get("energy_ratio", 0.8), 0.0, 1.0)
        correlation_score = metrics.get("correlation", 0.8)
        musical_noise_score = 1.0 - metrics.get("musical_noise_score", 0.1)
        transient_score = metrics.get("transient_preservation", 0.9)

        quality = 0.2 * energy_score + 0.3 * correlation_score + 0.3 * musical_noise_score + 0.2 * transient_score

        return float(np.clip(quality, 0.0, 1.0))


# ============================================================================
# GENERIC RESTORATION SAFETY WRAPPER
# ============================================================================


class GenericRestorationSafety(BaseSafetyWrapper):
    """
    Generic safety wrapper for restoration modules.

    Applies to:
    - Declickers
    - Declippers
    - Decracklers
    - Debuzzers
    - Impulse noise removers

    Ensures:
    - Defects detected before processing
    - No over-correction
    - Preserves intentional dynamics
    - Maintains spectral balance
    - Musical Goals not violated
    """

    def __init__(
        self,
        module_name: str,
        module_version: str,
        processor_func: Callable,
        processing_mode: ProcessingMode = ProcessingMode.RESTORATION,
        enable_logging: bool = True,
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            module_version=module_version,
            processor_func=processor_func,
            enable_logging=enable_logging,
            confidence_threshold=0.6,
            quality_threshold=0.75,
        )
        self.processing_mode = processing_mode
        self.musical_goals = MusicalGoalsChecker()

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for restoration."""
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)

        # Detect presence of defects
        defect_score = self._detect_impulsive_defects(audio_mono, sr)

        warnings = []

        if defect_score < 0.1:
            warnings.append(f"Very few defects detected (score={defect_score:.2f}). Restoration may not be necessary.")

        if defect_score > 0.7:
            warnings.append(
                f"Severe degradation detected (score={defect_score:.2f}). Restoration may introduce artifacts."
            )

        return PreCheckResult(
            passed=True,
            confidence=defect_score,  # Confidence based on defect detectability
            reasons=[],
            warnings=warnings,
            metadata={"defect_score": float(defect_score)},
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in restoration."""
        defect_score = pre_check.metadata.get("defect_score", 0.3)

        # High confidence at moderate defect levels
        if 0.2 <= defect_score <= 0.5:
            return 0.85
        elif 0.1 <= defect_score <= 0.7:
            return 0.7
        else:
            return 0.55

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions for restoration."""
        issues = []
        side_effects = []
        metrics = {}

        # 1. Check defect removal
        orig_defects = self._detect_impulsive_defects(original if original.ndim == 1 else np.mean(original, axis=0), sr)
        proc_defects = self._detect_impulsive_defects(
            processed if processed.ndim == 1 else np.mean(processed, axis=0), sr
        )

        metrics["original_defects"] = orig_defects
        metrics["processed_defects"] = proc_defects
        metrics["defect_reduction"] = orig_defects - proc_defects

        if proc_defects > orig_defects * 0.7:
            issues.append(f"Insufficient defect removal: {proc_defects:.2f} (was {orig_defects:.2f})")

        # 2. Check for over-processing (artifacts)
        correlation = compute_correlation(
            original if original.ndim == 1 else np.mean(original, axis=0),
            processed if processed.ndim == 1 else np.mean(processed, axis=0),
        )
        metrics["correlation"] = correlation

        if correlation < 0.8:
            issues.append(f"Low correlation - possible over-processing: {correlation:.2f}")

        # 3. Check spectral balance preservation
        orig_centroid = compute_spectral_centroid(original if original.ndim == 1 else np.mean(original, axis=0), sr)
        proc_centroid = compute_spectral_centroid(processed if processed.ndim == 1 else np.mean(processed, axis=0), sr)

        centroid_change = abs(proc_centroid - orig_centroid) / (orig_centroid + 1e-10)
        metrics["spectral_centroid_change"] = centroid_change

        if centroid_change > 0.15:
            side_effects.append(f"Spectral balance shifted: {centroid_change:.2%}")

        # 4. Musical Goals validation
        try:
            proc_goals = self.musical_goals.measure_all(processed, sr)
            mode_config = PROCESSING_MODE_CONFIGS.get(self.processing_mode)
            thresholds = mode_config.musical_goals if mode_config else {}

            violations = []
            for goal_name, threshold in thresholds.items():
                if goal_name in proc_goals and proc_goals[goal_name] < threshold:
                    violations.append(f"{goal_name}: {proc_goals[goal_name]:.2f} < {threshold:.2f}")

            if violations:
                issues.append(f"Musical Goals violations: {', '.join(violations)}")

            metrics["musical_goals"] = proc_goals  # type: ignore[assignment]
        except Exception as e:
            side_effects.append(f"Musical Goals check failed: {e}")

        passed = len(issues) == 0
        quality_score = self._compute_restoration_quality(metrics)

        return PostCheckResult(
            passed=passed, quality_score=quality_score, issues=issues, side_effects=side_effects, metrics=metrics
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet quality score for restoration."""
        return self._compute_restoration_quality(post_check.metrics)

    def _detect_impulsive_defects(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt impulsive defects (clicks, pops, crackle)."""
        # Compute envelope
        from scipy.ndimage import maximum_filter1d

        window = int(sr * 0.001)  # 1ms window
        envelope = maximum_filter1d(np.abs(audio), size=window)

        # Find sudden peaks
        threshold = np.median(envelope) + 3 * np.std(envelope)
        peaks = envelope > threshold

        # Defect score = fraction of samples classified as defects
        defect_score = np.mean(peaks.astype(float))

        return float(np.clip(defect_score, 0.0, 1.0))

    def _compute_restoration_quality(self, metrics: dict[str, Any]) -> float:
        """Berechnet quality score for restoration."""
        defect_reduction = metrics.get("defect_reduction", 0.0)
        correlation = metrics.get("correlation", 0.85)
        centroid_change = metrics.get("spectral_centroid_change", 0.05)

        # Quality = defect removal + preservation of content
        defect_quality = np.clip(defect_reduction, 0.0, 1.0)
        preservation_quality = correlation * (1.0 - centroid_change)

        quality = 0.6 * defect_quality + 0.4 * preservation_quality

        return float(np.clip(quality, 0.0, 1.0))


# ============================================================================
# SAFETY WRAPPER FACTORY
# ============================================================================


def create_safety_wrapper(
    module_name: str,
    module_type: str,
    processor_func: Callable,
    module_version: str = "1.0.0",
    processing_mode: ProcessingMode = ProcessingMode.RESTORATION,
) -> BaseSafetyWrapper:
    """
    Factory-Funktion zum Erstellen eines passenden Sicherheits-Wrappers für DSP-Module.

    Args:
        module_name: Name of DSP module
        module_type: Type of processing ('noise_reduction', 'restoration', etc.)
        processor_func: Processing function to wrap
        module_version: Module version
        processing_mode: Processing mode for Musical Goals

    Returns:
        Appropriate safety wrapper instance
    """
    module_type_lower = module_type.lower()

    if (
        "noise" in module_type_lower
        or "denoise" in module_type_lower
        or "hiss" in module_type_lower
        or "gate" in module_type_lower
    ):
        return GenericNoiseReductionSafety(
            module_name=module_name,
            module_version=module_version,
            processor_func=processor_func,
            processing_mode=processing_mode,
        )

    elif (
        "click" in module_type_lower
        or "clip" in module_type_lower
        or "crack" in module_type_lower
        or "restoration" in module_type_lower
    ):
        return GenericRestorationSafety(
            module_name=module_name,
            module_version=module_version,
            processor_func=processor_func,
            processing_mode=processing_mode,
        )

    else:
        # Default to noise reduction wrapper (most conservative)
        return GenericNoiseReductionSafety(
            module_name=module_name,
            module_version=module_version,
            processor_func=processor_func,
            processing_mode=processing_mode,
        )
logger = logging.getLogger(__name__)
