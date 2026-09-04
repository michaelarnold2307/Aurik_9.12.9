"""
Context-Aware De-Esser Safety Wrapper
======================================

HIPS-compliant safety wrapper for phoneme-aware de-essing.

This wrapper ensures:
- Phoneme detection is functioning
- Sibilants are actually present
- Processing doesn't degrade intelligibility
- No over-processing artifacts
- Confidence-based validation

Author: Aurik Development Team
Version: 2.0.0
Date: 8. Februar 2026
"""

from typing import Any

import logging
import numpy as np

logger = logging.getLogger(__name__)

from backend.ml.safety_wrappers.safety_wrapper_template import (
    PostCheckResult,
    PreCheckResult,
)

try:
    from backend.ml.phoneme_aware.phoneme_classifier import PhonemeClassifier
    from backend.ml.phoneme_aware.phoneme_detector import DetectionConfig, PhonemeDetector

    _PHONEME_STACK_AVAILABLE = True
except Exception:
    PhonemeClassifier = None  # type: ignore[misc, assignment]
    DetectionConfig = None  # type: ignore[misc, assignment]
    PhonemeDetector = None  # type: ignore[misc, assignment]
    _PHONEME_STACK_AVAILABLE = False

# Import validation utilities from deesser_safety (which has them inline)
try:
    from backend.ml.safety_wrappers.deesser_safety import (
        compute_correlation as _compute_correlation_impl,
    )
    from backend.ml.safety_wrappers.deesser_safety import (
        compute_energy_ratio as _compute_energy_ratio_impl,
    )

    def compute_energy_ratio(before: np.ndarray, after: np.ndarray) -> float:
        """Berechnet energy ratio in dB."""
        return float(_compute_energy_ratio_impl(before, after))

    def compute_correlation(before: np.ndarray, after: np.ndarray) -> float:
        """Berechnet correlation between before/after."""
        return float(_compute_correlation_impl(before, after))

except ImportError:
    logger.debug("§V6 deesser_safety-Import fehlgeschlagen — Fallback-Implementierungen aktiviert")

    def compute_energy_ratio(before: np.ndarray, after: np.ndarray) -> float:
        """Berechnet energy ratio in dB."""
        energy_before = np.sum(before**2)
        energy_after = np.sum(after**2)
        if energy_before == 0:
            return 0.0
        ratio = energy_after / energy_before
        if ratio <= 0:
            return -np.inf
        return float(10 * np.log10(ratio))

    def compute_correlation(before: np.ndarray, after: np.ndarray) -> float:
        """Berechnet correlation between before/after."""
        # Ensure same length
        min_len = min(len(before), len(after))
        before = before[:min_len]
        after = after[:min_len]
        # Compute correlation
        if np.std(before) == 0 or np.std(after) == 0:
            return 0.0
        _a = before - before.mean()
        _b = after - after.mean()
        _na = float(np.linalg.norm(_a))
        _nb = float(np.linalg.norm(_b))
        r = float(np.dot(_a, _b) / (_na * _nb + 1e-10))
        return r if np.isfinite(r) else 0.0


def validate_audio_basic(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Basic audio validation."""
    issues = []

    if audio.size == 0:
        issues.append("Audio is empty")
    if sr <= 0:
        issues.append(f"Invalid sample rate: {sr}")
    if np.all(audio == 0):
        issues.append("Audio is silent")
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        issues.append("Audio contains NaN or Inf values")

    return {"is_valid": len(issues) == 0, "message": "; ".join(issues) if issues else "Valid"}


# ============================================================================
# PHONEME-AWARE SIBILANCE ANALYSIS
# ============================================================================


def detect_phoneme_based_sibilance(
    audio: np.ndarray,
    sr: int,
) -> tuple[bool, float, dict[str, Any]]:
    """
    Erkennt sibilance using phoneme detection.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate

    Returns:
        (has_sibilance, intensity, metrics): Detection result
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Try phoneme-based detection first
    try:
        if not _PHONEME_STACK_AVAILABLE:
            raise RuntimeError("phoneme stack not available")
        assert PhonemeDetector is not None
        assert DetectionConfig is not None
        assert PhonemeClassifier is not None
        detector = PhonemeDetector(config=DetectionConfig(min_confidence=0.5, use_gpu=False))
        classifier = PhonemeClassifier()

        # Detect phonemes
        phonemes = detector.detect(audio, sr)

        # Count sibilants
        sibilant_count = 0
        sibilant_duration = 0.0

        for phoneme_seg in phonemes:
            phoneme_info = classifier.classify(phoneme_seg.phoneme)
            if phoneme_info.is_sibilant:  # type: ignore[attr-defined]
                sibilant_count += 1
                sibilant_duration += phoneme_seg.duration

        # Calculate metrics
        total_duration = len(audio) / sr
        sibilance_ratio = sibilant_duration / total_duration if total_duration > 0 else 0.0

        has_sibilance = sibilant_count > 0
        intensity = float(np.clip(sibilance_ratio * 10.0, 0.0, 1.0))  # Normalize

        metrics = {
            "phonemes_detected": len(phonemes),
            "sibilants_detected": sibilant_count,
            "sibilance_duration_sec": float(sibilant_duration),
            "sibilance_ratio": float(sibilance_ratio),
            "detection_method": "phoneme_based",
        }

        return has_sibilance, intensity, metrics

    except Exception as e:
        logger.debug("§V6 phoneme-basierte Sibilanz-Erkennung fehlgeschlagen — Frequency-Fallback aktiviert: %s", e)
        return _detect_frequency_based_sibilance(audio, sr)


def _detect_frequency_based_sibilance(
    audio: np.ndarray,
    sr: int,
) -> tuple[bool, float, dict[str, Any]]:
    """
    Fallback: Frequency-based sibilance detection.

    Args:
        audio: Input audio (mono)
        sr: Sample rate

    Returns:
        (has_sibilance, intensity, metrics): Detection result
    """
    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Sibilant frequency band (4-10 kHz)
    s_band = (freqs >= 4000) & (freqs <= 10000)

    s_energy = np.mean(spectrum[s_band]) if np.any(s_band) else 0.0
    total_energy = np.mean(spectrum)

    if total_energy == 0:
        return False, 0.0, {}

    s_ratio = s_energy / total_energy

    has_sibilance = s_ratio > 0.15
    intensity = float(np.clip(s_ratio / 0.3, 0.0, 1.0))

    metrics = {
        "sibilant_band_energy": float(s_energy),
        "sibilance_ratio": float(s_ratio),
        "detection_method": "frequency_based",
    }

    return has_sibilance, intensity, metrics


def measure_intelligibility(audio: np.ndarray, sr: int) -> float:
    """
    Misst speech intelligibility using spectral balance.

    Rough approximation based on:
    - Mid-frequency energy (1-4 kHz) - vowels, formants
    - High-frequency energy (4-8 kHz) - consonants, sibilants

    Args:
        audio: Input audio (mono)
        sr: Sample rate

    Returns:
        intelligibility_score: Score from 0-1 (higher = better)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Frequency bands
    mid_band = (freqs >= 1000) & (freqs <= 4000)
    high_band = (freqs >= 4000) & (freqs <= 8000)

    mid_energy = np.mean(spectrum[mid_band]) if np.any(mid_band) else 0.0
    high_energy = np.mean(spectrum[high_band]) if np.any(high_band) else 0.0
    total_energy = np.mean(spectrum)

    if total_energy == 0:
        return 0.0

    # Intelligibility heuristic: balanced mid and high frequencies
    # Optimal ratio: mid ~40%, high ~30%
    mid_ratio = mid_energy / total_energy
    high_ratio = high_energy / total_energy

    # Score based on proximity to optimal
    mid_score = 1.0 - abs(mid_ratio - 0.4) / 0.4
    high_score = 1.0 - abs(high_ratio - 0.3) / 0.3

    # Combine scores
    intelligibility = float(np.clip((mid_score + high_score) / 2.0, 0.0, 1.0))

    return intelligibility


# ============================================================================
# SAFETY WRAPPER
# ============================================================================


class ContextAwareDeEsserSafety:
    """
    HIPS-compliant safety wrapper for Context-Aware De-Esser v2.0.

    Pre-checks:
    - Validate audio format
    - Detect sibilance presence (phoneme-based or frequency-based)
    - Measure baseline intelligibility
    - Check phoneme detection availability

    Post-checks:
    - Verify sibilance reduction (not over-processing)
    - Ensure intelligibility preservation (>= 95% of original)
    - Check for artifacts (correlation > 0.85)
    - Validate no excessive gain changes
    """

    def __init__(self):
        self.name = "Context-Aware De-Esser Safety"
        self.min_intelligibility_preservation = 0.95
        self.min_correlation = 0.85
        self.max_energy_change_db = 3.0

    def run_pre_check(
        self,
        audio: np.ndarray,
        sr: int,
        params: dict[str, Any],
    ) -> PreCheckResult:
        """
        Pre-processing safety checks.

        Args:
            audio: Input audio
            sr: Sample rate
            params: De-esser parameters (mode, device, etc.)

        Returns:
            PreCheckResult with PASS/FAIL/WARN
        """
        issues = []
        warnings_list: list[Any] = []
        metrics = {}
        params = params or {}
        if params:
            metrics["requested_mode"] = str(params.get("mode", ""))
            metrics["requested_device"] = str(params.get("device", ""))

        # Basic validation
        validation_result = validate_audio_basic(audio, sr)
        if not validation_result["is_valid"]:
            issues.append(validation_result["message"])
            return PreCheckResult(
                passed=False,
                confidence=0.0,
                reasons=issues,
                warnings=warnings_list,
                metadata=metrics,
            )

        # Ensure mono for analysis (average if stereo)
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        # Always compute baseline intelligibility so metadata remains complete,
        # even when pre-check exits early (e.g. no sibilance detected).
        baseline_intelligibility = measure_intelligibility(audio_mono, sr)
        metrics["baseline_intelligibility"] = baseline_intelligibility  # type: ignore[assignment]

        # Check for sibilance
        has_sibilance, sibilance_intensity, sibilance_metrics = detect_phoneme_based_sibilance(audio_mono, sr)
        metrics.update(sibilance_metrics)
        metrics["sibilance_intensity"] = sibilance_intensity  # type: ignore[assignment]

        if not has_sibilance:
            issues.append("No sibilance detected in audio. De-essing may not be necessary.")
            return PreCheckResult(
                passed=False,
                confidence=0.0,
                reasons=issues,
                warnings=warnings_list,
                metadata=metrics,
            )

        if sibilance_intensity < 0.3:
            warnings_list.append(f"Low sibilance intensity ({sibilance_intensity:.2f}). Consider using GENTLE mode.")

        if baseline_intelligibility < 0.5:
            warnings_list.append(
                f"Low baseline intelligibility ({baseline_intelligibility:.2f}). De-essing may further reduce clarity."
            )

        # Check phoneme detection availability
        if _PHONEME_STACK_AVAILABLE:
            metrics["phoneme_detection_available"] = True  # type: ignore[assignment]
        else:
            metrics["phoneme_detection_available"] = False  # type: ignore[assignment]
            warnings_list.append("Phoneme detection not available. Using frequency-based fallback.")

        # All checks passed
        return PreCheckResult(
            passed=True,
            confidence=1.0,
            reasons=[],
            warnings=warnings_list,
            metadata=metrics,
        )

    def run_post_check(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        params: dict[str, Any],
        pre_check_result: PreCheckResult,
    ) -> PostCheckResult:
        """
        Post-processing safety checks.

        Args:
            audio_before: Original audio
            audio_after: Processed audio
            sr: Sample rate
            params: De-esser parameters
            pre_check_result: Pre-check results

        Returns:
            PostCheckResult with PASS/FAIL/WARN
        """
        issues = []
        warnings_list = []
        metrics = {}
        params = params or {}
        if params:
            metrics["requested_mode"] = str(params.get("mode", ""))
            metrics["requested_device"] = str(params.get("device", ""))

        # Ensure mono for analysis
        before_mono = np.mean(audio_before, axis=0) if audio_before.ndim > 1 else audio_before
        after_mono = np.mean(audio_after, axis=0) if audio_after.ndim > 1 else audio_after

        # 1. Check intelligibility preservation
        baseline_intel = pre_check_result.metadata.get("baseline_intelligibility", 0.0)
        after_intel = measure_intelligibility(after_mono, sr)
        metrics["after_intelligibility"] = after_intel  # type: ignore[assignment]
        # Backward-compatible contract: key is always present for downstream checks/tests.
        metrics["intelligibility_preservation"] = 1.0  # type: ignore[assignment]

        if baseline_intel > 0:
            intel_preservation = after_intel / baseline_intel
            metrics["intelligibility_preservation"] = intel_preservation

            if intel_preservation < self.min_intelligibility_preservation:
                issues.append(
                    f"Intelligibility degraded: {intel_preservation:.1%} of original "
                    f"(threshold: {self.min_intelligibility_preservation:.1%})"
                )

        # 2. Check correlation (detect artifacts)
        correlation = compute_correlation(before_mono, after_mono)
        metrics["correlation"] = correlation  # type: ignore[assignment]

        if correlation < self.min_correlation:
            issues.append(f"Low correlation ({correlation:.3f}). Possible over-processing or artifacts.")
        elif correlation < 0.92:
            warnings_list.append(f"Moderate correlation ({correlation:.3f}). Check for subtle artifacts.")

        # 3. Check energy change (gain validation)
        energy_ratio_db = compute_energy_ratio(before_mono, after_mono)
        metrics["energy_change_db"] = energy_ratio_db  # type: ignore[assignment]

        if abs(energy_ratio_db) > self.max_energy_change_db:
            warnings_list.append(
                f"Significant energy change: {energy_ratio_db:+.1f} dB. "
                "Consider adjusting reduction or applying makeup gain."
            )

        # 4. Verify sibilance reduction (should be lower after processing)
        _, before_sib_intensity, _ = detect_phoneme_based_sibilance(before_mono, sr)
        _, after_sib_intensity, _ = detect_phoneme_based_sibilance(after_mono, sr)
        metrics["sibilance_before"] = before_sib_intensity  # type: ignore[assignment]
        metrics["sibilance_after"] = after_sib_intensity  # type: ignore[assignment]

        if after_sib_intensity >= before_sib_intensity * 0.95:
            warnings_list.append(
                "Minimal sibilance reduction: "
                f"{before_sib_intensity:.2f} → {after_sib_intensity:.2f}. "
                "Consider more aggressive settings."
            )

        sibilance_reduction_percent = (
            (before_sib_intensity - after_sib_intensity) / before_sib_intensity * 100.0
            if before_sib_intensity > 0
            else 0.0
        )
        metrics["sibilance_reduction_percent"] = sibilance_reduction_percent  # type: ignore[assignment]

        # Determine pass/fail
        passed = len(issues) == 0

        return PostCheckResult(
            passed=passed,
            quality_score=1.0 if passed else 0.0,
            issues=issues,
            side_effects=warnings_list,
            metrics=metrics,  # type: ignore[arg-type]
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def validate_deessing_pre(
    audio: np.ndarray,
    sr: int,
    params: dict[str, Any] | None = None,
) -> PreCheckResult:
    """
    Convenience function for pre-check validation.

    Args:
        audio: Input audio
        sr: Sample rate
        params: De-esser parameters

    Returns:
        PreCheckResult
    """
    wrapper = ContextAwareDeEsserSafety()
    return wrapper.run_pre_check(audio, sr, params or {})


def validate_deessing_post(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
    params: dict[str, Any] | None = None,
    pre_check: PreCheckResult | None = None,
) -> PostCheckResult:
    """
    Convenience function for post-check validation.

    Args:
        audio_before: Original audio
        audio_after: Processed audio
        sr: Sample rate
        params: De-esser parameters
        pre_check: Pre-check results (will run if not provided)

    Returns:
        PostCheckResult
    """
    wrapper = ContextAwareDeEsserSafety()

    if pre_check is None:
        pre_check = wrapper.run_pre_check(audio_before, sr, params or {})

    return wrapper.run_post_check(audio_before, audio_after, sr, params or {}, pre_check)
