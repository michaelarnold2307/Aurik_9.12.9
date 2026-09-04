"""
safety_wrapper_template.py - HIPS-Compliant Safety Wrapper Template

This template provides the standard structure for all HIPS safety wrappers
in AURIK v9.x. Each wrapper ensures:
- Pre-condition validation
- Epistemic confidence assessment
- Post-processing validation
- Audit trail logging
- Reversibility guarantee

Usage:
    Copy this template and customize for your specific DSP module.
    All TODOs must be implemented for HIPS compliance.

Author: AURIK Team
Version: 1.0.0
Date: 7. Februar 2026
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

# ============================================================================
# ENUMS & DATA MODELS
# ============================================================================


class ConfidenceLevel(Enum):
    """Epistemic confidence levels"""

    VERY_LOW = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 0.95


class ProcessingDecision(Enum):
    """Verarbeitet decision outcomes."""

    PROCEED = "proceed"
    ABORT = "abort"
    REDUCE_STRENGTH = "reduce_strength"
    FALLBACK_MODE = "fallback_mode"


@dataclass
class PreCheckResult:
    """Result of pre-processing validation"""

    passed: bool
    confidence: float  # 0.0-1.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostCheckResult:
    """Result of post-processing validation"""

    passed: bool
    quality_score: float  # 0.0-1.0
    issues: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessingReport:
    """Complete audit trail for processing operation"""

    timestamp: str
    module_name: str
    module_version: str

    # Input characteristics
    input_metadata: dict[str, Any]

    # Pre-checks
    pre_check_result: PreCheckResult

    # Processing decision
    decision: ProcessingDecision
    processing_params: dict[str, Any]

    # Post-checks
    post_check_result: PostCheckResult | None

    # Output characteristics
    output_metadata: dict[str, Any] | None

    # Reversibility info
    reversibility_data: dict[str, Any] | None

    # Performance
    processing_time_ms: float


# ============================================================================
# BASE SAFETY WRAPPER
# ============================================================================


class BaseSafetyWrapper:
    """
    Base class for all HIPS-compliant safety wrappers.

    All wrappers must implement:
    - _validate_pre_conditions()
    - _assess_epistemic_confidence()
    - _validate_post_conditions()
    - _compute_quality_score()
    - _log_audit_trail()

    Optional overrides:
    - _prepare_reversibility_data()
    - _apply_fallback_processing()
    """

    def __init__(
        self,
        module_name: str,
        module_version: str,
        processor_func: Callable,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        confidence_threshold: float = 0.5,
        quality_threshold: float = 0.6,
    ):
        """
        Initialisiert safety wrapper.

        Args:
            module_name: Name of wrapped DSP module
            module_version: Version of DSP module
            processor_func: The actual processing function to wrap
            enable_logging: Enable audit trail logging
            log_dir: Directory for audit logs (default: ./logs/audit/)
            confidence_threshold: Minimum confidence to proceed (0.0-1.0)
            quality_threshold: Minimum quality score for success (0.0-1.0)
        """
        self.module_name = module_name
        self.module_version = module_version
        self.processor_func = processor_func
        self.enable_logging = enable_logging
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold

        # Setup logging
        self.logger = logging.getLogger(f"HIPS.{module_name}")
        if log_dir is None:
            log_dir = Path("./logs/audit")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.aborted_calls = 0
        self.reduced_strength_calls = 0
        self.quality_scores: list[float] = []  # Track all quality scores

    def process(self, audio: np.ndarray, sr: int, **processing_params) -> tuple[np.ndarray, ProcessingReport]:
        """
        Haupt-processing entry point with full HIPS compliance.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            **processing_params: Module-specific parameters

        Returns:
            (processed_audio, report): Processed audio and audit report
        """
        self.total_calls += 1
        start_time = datetime.now()

        # =================================================================
        # PHASE 1: PRE-CHECKS
        # =================================================================
        input_metadata = self._extract_input_metadata(audio, sr)

        pre_check = self._validate_pre_conditions(audio, sr, **processing_params)

        if not pre_check.passed:
            self.aborted_calls += 1
            return self._create_abort_response(audio, input_metadata, pre_check, start_time)

        # =================================================================
        # PHASE 2: EPISTEMIC CONFIDENCE ASSESSMENT
        # =================================================================
        confidence = self._assess_epistemic_confidence(audio, sr, pre_check, **processing_params)

        # Decide processing strategy based on confidence
        if confidence < self.confidence_threshold:
            self.logger.warning(
                "Low confidence (%.2f < %.2f). Aborting processing.",
                confidence,
                self.confidence_threshold,
            )
            pre_check.warnings.append(f"Insufficient epistemic confidence: {confidence:.2f}")
            self.aborted_calls += 1
            return self._create_abort_response(audio, input_metadata, pre_check, start_time)

        # Adjust processing strength if medium confidence
        decision = ProcessingDecision.PROCEED
        adjusted_params = processing_params.copy()

        if confidence < 0.8:
            decision = ProcessingDecision.REDUCE_STRENGTH
            adjusted_params = self._adjust_params_for_confidence(processing_params, confidence)
            self.reduced_strength_calls += 1
            self.logger.info("Medium confidence (%.2f). Reducing processing strength.", confidence)

        # =================================================================
        # PHASE 3: PROCESSING
        # =================================================================
        # Store original for reversibility
        reversibility_data = self._prepare_reversibility_data(audio, sr)

        try:
            processed_audio = self.processor_func(audio, sr, **adjusted_params)
        except Exception as e:
            logger.error("§V6 Processing fehlgeschlagen — Abort-Response zurückgegeben: %s", e)
            self.aborted_calls += 1
            pre_check.passed = False
            pre_check.reasons.append(f"Processing exception: {e!s}")
            return self._create_abort_response(audio, input_metadata, pre_check, start_time)

        # =================================================================
        # PHASE 4: POST-CHECKS
        # =================================================================
        post_check = self._validate_post_conditions(original=audio, processed=processed_audio, sr=sr, **adjusted_params)

        if not post_check.passed:
            self.logger.warning("Post-check failed. Issues: %s", post_check.issues)
            # Apply fallback or return original
            processed_audio = self._apply_fallback_processing(audio, sr, post_check, **adjusted_params)
            post_check.side_effects.append("Fallback processing applied")

        # Quality assessment
        quality_score = self._compute_quality_score(
            original=audio, processed=processed_audio, sr=sr, post_check=post_check
        )

        if quality_score < self.quality_threshold:
            self.logger.warning(
                "Quality score (%.2f) below threshold (%.2f). Returning original.",
                quality_score,
                self.quality_threshold,
            )
            self.aborted_calls += 1
            return self._create_abort_response(
                audio, input_metadata, pre_check, start_time, reason=f"Quality score too low: {quality_score:.2f}"
            )

        # =================================================================
        # PHASE 5: AUDIT TRAIL
        # =================================================================
        self.successful_calls += 1

        output_metadata = self._extract_output_metadata(processed_audio, sr)

        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        report = ProcessingReport(
            timestamp=datetime.now().isoformat(),
            module_name=self.module_name,
            module_version=self.module_version,
            input_metadata=input_metadata,
            pre_check_result=pre_check,
            decision=decision,
            processing_params=adjusted_params,
            post_check_result=post_check,
            output_metadata=output_metadata,
            reversibility_data=reversibility_data,
            processing_time_ms=processing_time_ms,
        )

        # Track quality score for statistics
        if post_check and post_check.quality_score is not None:
            self.quality_scores.append(post_check.quality_score)

        if self.enable_logging:
            self._log_audit_trail(report)

        return processed_audio, report

    # =========================================================================
    # METHODS TO IMPLEMENT IN SUBCLASSES
    # =========================================================================

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """
        Validiert pre-conditions before processing.

        MUST IMPLEMENT in subclass with module-specific checks:
        - Input validity (NaN, Inf, clipping)
        - Signal characteristics (frequency content, dynamics)
        - Module-specific requirements

        Returns:
            PreCheckResult with validation outcome
        """
        if params:
            self.logger.debug("_validate_pre_conditions: params=%s (Basisimplementierung)", list(params.keys()))
        reasons: list = []
        warnings_list: list = []
        # NaN/Inf-Guard
        if not np.all(np.isfinite(audio)):
            reasons.append("Audio contains NaN or Inf values")
        # Clipping warning
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.005:
            warnings_list.append(f"Audio clips at peak={peak:.4f}")
        # Minimum length: at least 10 ms
        min_samples = max(1, int(sr * 0.01))
        if audio.size < min_samples:
            reasons.append(f"Audio too short: {audio.size} samples (< {min_samples} = 10 ms @ {sr} Hz)")
        passed = len(reasons) == 0
        confidence = ConfidenceLevel.HIGH.value if passed else ConfidenceLevel.LOW.value
        return PreCheckResult(
            passed=passed,
            confidence=confidence,
            reasons=reasons,
            warnings=warnings_list,
            metadata={"peak": peak, "n_samples": int(audio.size)},
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """
        Assess epistemic confidence for this processing operation.

        MUST IMPLEMENT in subclass with module-specific logic:
        - Can we reliably detect what needs processing?
        - Do we understand the signal characteristics?
        - Is this the right tool for this audio?

        Returns:
            Confidence score (0.0-1.0)
        """
        self.logger.debug("_assess_epistemic_confidence: sr=%d (Basisimplementierung)", sr)
        if params:
            self.logger.debug("_assess_epistemic_confidence: params=%s (Basisimplementierung)", list(params.keys()))
        if not pre_check.passed:
            return ConfidenceLevel.VERY_LOW.value
        if not np.all(np.isfinite(audio)) or audio.size == 0:
            return ConfidenceLevel.VERY_LOW.value
        # Proxy-SNR: RMS vs. 10th-percentile floor
        abs_audio = np.abs(audio)
        rms = float(np.sqrt(np.mean(audio**2))) + 1e-12
        noise_floor = float(np.percentile(abs_audio, 10)) + 1e-12
        snr_db = 20.0 * np.log10(rms / noise_floor)
        # Map SNR 0..60 dB → confidence LOW..VERY_HIGH
        raw = float(np.clip(snr_db / 60.0, 0.0, 1.0))
        # Floor at LOW so we never return 0 for valid audio
        return float(np.clip(raw, ConfidenceLevel.LOW.value, ConfidenceLevel.VERY_HIGH.value))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """
        Validiert post-conditions after processing.

        MUST IMPLEMENT in subclass with module-specific checks:
        - No new artifacts introduced
        - Intended effect achieved
        - Side effects within acceptable bounds
        - Energy/phase coherence preserved

        Returns:
            PostCheckResult with validation outcome
        """
        self.logger.debug("_validate_post_conditions: sr=%d (Basisimplementierung)", sr)
        if params:
            self.logger.debug("_validate_post_conditions: params=%s (Basisimplementierung)", list(params.keys()))
        issues: list = []
        # NaN/Inf in processed output
        if not np.all(np.isfinite(processed)):
            issues.append("Processed audio contains NaN or Inf values")
        # Energy ratio sanity check
        orig_rms = float(np.sqrt(np.mean(original**2))) + 1e-12
        proc_rms = float(np.sqrt(np.mean(processed**2))) + 1e-12
        energy_ratio = proc_rms / orig_rms
        if energy_ratio < 0.05 or energy_ratio > 20.0:
            issues.append(f"Energy ratio out of bounds: {energy_ratio:.3f} (expected 0.05..20.0)")
        passed = len(issues) == 0
        quality_score = float(np.clip(1.0 - abs(1.0 - energy_ratio) * 0.5, 0.0, 1.0))
        return PostCheckResult(
            passed=passed,
            quality_score=quality_score,
            issues=issues,
            side_effects=[],
            metrics={"energy_ratio": energy_ratio, "orig_rms": orig_rms},
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """
        §v10 Berechnet quality score — HPE-basiert statt rein technisch.

        Früher: 60% energy preservation + 40% SNR ratio (technisch)
        Jetzt:   50% HPE-Vergleich + 30% Energy preservation + 20% SNR

        Returns:
            Quality score (0.0-1.0)
        """
        self.logger.debug("_compute_quality_score: sr=%d", sr)
        if not post_check.passed:
            return 0.0

        # Component 1: post_check quality score (energy preservation)
        struct_score = float(np.clip(post_check.quality_score, 0.0, 1.0))

        # Component 2: proxy SNR improvement
        snr_ratio = 0.5
        if original.size > 0 and processed.size > 0:
            n_common = min(original.size, processed.size)
            diff = processed[:n_common] - original[:n_common]
            orig_rms = float(np.sqrt(np.mean(original[:n_common] ** 2))) + 1e-12
            diff_rms = float(np.sqrt(np.mean(diff**2))) + 1e-12
            snr_ratio = float(np.clip(orig_rms / diff_rms, 0.0, 100.0)) / 100.0

        # §v10 Component 3: HPE pleasantness comparison
        hpe_score = 0.5
        try:
            from backend.core.human_pleasantness_estimator import compare_pleasantness

            hpe_cmp = compare_pleasantness(
                np.asarray(original, dtype=np.float32),
                np.asarray(processed, dtype=np.float32),
                sr,
            )
            hpe_delta = float(hpe_cmp.get("delta_score", 0.0))
            # Map delta [-1,1] to score [0,1] where positive delta = high score
            hpe_score = float(np.clip(0.5 + hpe_delta * 0.5, 0.0, 1.0))
        except Exception:
            logger.warning("safety_wrapper_template.py::_berechnen_quality_Wert Ersatzpfad", exc_info=True)

        # §v10: 50% HPE + 30% structure + 20% SNR
        return float(np.clip(0.50 * hpe_score + 0.30 * struct_score + 0.20 * snr_ratio, 0.0, 1.0))

    def _log_audit_trail(self, report: ProcessingReport) -> None:
        """
        Protokolliert audit trail to disk.

        MUST IMPLEMENT in subclass with module-specific format.
        Default: JSONL format
        """
        log_file = self.log_dir / f"{self.module_name}_audit.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            # Convert report to dict (simplified)
            log_entry = {
                "timestamp": report.timestamp,
                "module": report.module_name,
                "decision": report.decision.name,  # Use .name instead of .value for uppercase
                "confidence": report.pre_check_result.confidence,
                "quality": report.post_check_result.quality_score if report.post_check_result else 0.0,
                "processing_time_ms": report.processing_time_ms,
            }
            f.write(json.dumps(log_entry) + "\n")

    # =========================================================================
    # OPTIONAL METHODS (can override in subclasses)
    # =========================================================================

    def _prepare_reversibility_data(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Prepare data needed for reversibility.

        Optional override for module-specific reversibility needs.
        Default: Store hash of original
        """
        return {"original_hash": hash(audio.tobytes()), "original_shape": audio.shape, "sample_rate": sr}

    def _apply_fallback_processing(
        self, audio: np.ndarray, sr: int, post_check: PostCheckResult, **params
    ) -> np.ndarray:
        """
        Wendet an: fallback processing if post-check fails.

        Optional override for module-specific fallback.
        Default: Return original audio
        """
        self.logger.info(
            "Applying fallback: returning original audio (sr=%d, post_check.passed=%s)",
            sr,
            post_check.passed,
        )
        if params:
            self.logger.debug("_apply_fallback_processing: params=%s (Basisimplementierung)", list(params.keys()))
        return audio

    def _adjust_params_for_confidence(self, params: dict[str, Any], confidence: float) -> dict[str, Any]:
        """
        Adjust processing parameters based on confidence level.

        Optional override for module-specific parameter adjustment.
        Default: Scale all numeric params by confidence
        """
        adjusted = params.copy()

        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Scale by confidence (conservative approach)
                adjusted[key] = value * confidence

        return adjusted

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _extract_input_metadata(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Extrahiert metadata from input audio."""
        return {
            "shape": audio.shape,
            "dtype": str(audio.dtype),
            "sample_rate": sr,
            "duration_sec": audio.shape[-1] / sr,
            "channels": 1 if audio.ndim == 1 else audio.shape[0],
            "peak_amplitude": float(np.max(np.abs(audio))),
            "rms_level": float(np.sqrt(np.mean(audio**2))),
            "has_nan": bool(np.any(np.isnan(audio))),
            "has_inf": bool(np.any(np.isinf(audio))),
        }

    def _extract_output_metadata(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Extrahiert metadata from output audio."""
        return self._extract_input_metadata(audio, sr)

    def _create_abort_response(
        self,
        original_audio: np.ndarray,
        input_metadata: dict[str, Any],
        pre_check: PreCheckResult,
        start_time: datetime,
        reason: str = "Pre-check failed",
    ) -> tuple[np.ndarray, ProcessingReport]:
        """Erstellt response when processing is aborted."""
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        report = ProcessingReport(
            timestamp=datetime.now().isoformat(),
            module_name=self.module_name,
            module_version=self.module_version,
            input_metadata=input_metadata,
            pre_check_result=pre_check,
            decision=ProcessingDecision.ABORT,
            processing_params={},
            post_check_result=None,
            output_metadata=input_metadata,  # Same as input
            reversibility_data=None,
            processing_time_ms=processing_time_ms,
        )

        self.logger.warning("Processing aborted: %s", reason)

        if self.enable_logging:
            self._log_audit_trail(report)

        return original_audio, report

    def get_statistics(self) -> dict[str, Any]:
        """Gibt zurück: wrapper statistics."""
        stats = {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "aborted_calls": self.aborted_calls,
            "reduced_strength_calls": self.reduced_strength_calls,
            "success_rate": self.successful_calls / max(1, self.total_calls),
            "abort_rate": self.aborted_calls / max(1, self.total_calls),
        }

        # Add average quality score if available
        if self.quality_scores:
            stats["average_quality_score"] = sum(self.quality_scores) / len(self.quality_scores)
            stats["min_quality_score"] = min(self.quality_scores)
            stats["max_quality_score"] = max(self.quality_scores)

        return stats


# ============================================================================
# COMMON VALIDATION HELPERS
# ============================================================================


def validate_audio_basic(audio: np.ndarray) -> tuple[bool, list[str]]:
    """
    Basic audio validation checks.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Check for NaN
    if np.any(np.isnan(audio)):
        errors.append("Audio contains NaN values")

    # Check for Inf
    if np.any(np.isinf(audio)):
        errors.append("Audio contains Inf values")

    # Check for clipping
    if np.max(np.abs(audio)) > 0.99:
        errors.append("Audio appears to be clipping (peak > 0.99)")

    # Check for silence
    if np.max(np.abs(audio)) < 1e-6:
        errors.append("Audio is silent or near-silent")

    # Check shape
    if audio.ndim not in [1, 2]:
        errors.append(f"Invalid audio shape: {audio.shape} (expected 1D or 2D)")

    if audio.ndim == 2 and audio.shape[0] > 2:
        errors.append(f"Too many channels: {audio.shape[0]} (expected 1 or 2)")

    return len(errors) == 0, errors


def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Berechnet spectral centroid (brightness measure)."""
    # Simple FFT-based centroid
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    return float(centroid)


def compute_energy_ratio(original: np.ndarray, processed: np.ndarray) -> float:
    """Berechnet energy preservation ratio."""
    energy_orig = np.sum(original**2)
    energy_proc = np.sum(processed**2)

    if energy_orig == 0:
        return 0.0

    return float(energy_proc / energy_orig)


def compute_correlation(original: np.ndarray, processed: np.ndarray) -> float:
    """Berechnet correlation between original and processed."""
    # Ensure same length
    min_len = min(len(original), len(processed))
    orig = original[:min_len]
    proc = processed[:min_len]

    # Normalize
    orig_norm = (orig - np.mean(orig)) / (np.std(orig) + 1e-8)
    proc_norm = (proc - np.mean(proc)) / (np.std(proc) + 1e-8)

    # Guarded Pearson correlation — avoids NaN on near-constant signals (§VERBOTEN: np.corrcoef)
    _no = float(np.linalg.norm(orig_norm))
    _np_ = float(np.linalg.norm(proc_norm))
    if _no < 1e-12 or _np_ < 1e-12:
        return 0.0
    correlation = float(np.dot(orig_norm, proc_norm) / (_no * _np_))
    return float(np.clip(correlation, -1.0, 1.0))
