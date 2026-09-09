"""
generic_safety_wrapper_extended.py - Extended Generic HIPS Safety Wrappers

Provides additional generic safety wrappers for:
- Dynamics processing (compressor, limiter, gate, expander)
- Spectral processing (EQ, filters, enhancement)
- Spatial processing (stereo widener, panner)

Author: AURIK Team
Version: 1.0.0
Date: 8. Februar 2026
"""

# Import Musical Goals integration
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.signal as signal

from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
from backend.core.musical_goals.processing_modes import PROCESSING_MODE_CONFIGS, ProcessingMode

from .safety_wrapper_template import (
    BaseSafetyWrapper,
    PostCheckResult,
    PreCheckResult,
    compute_correlation,
    compute_spectral_centroid,
    validate_audio_basic,
)

# ============================================================================
# GENERIC DYNAMICS SAFETY WRAPPER
# ============================================================================


class GenericDynamicsSafety(BaseSafetyWrapper):
    """
    Generic safety wrapper for dynamics processing modules.

    Applies to:
    - Compressors (multiband, broadband)
    - Limiters
    - Gates
    - Expanders
    - AGC systems

    Ensures:
    - Dynamic range changes are reasonable
    - No pumping/breathing artifacts
    - Transients preserved appropriately
    - No distortion introduced
    - Musical Goals (Emotionalität, Natürlichkeit) preserved
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
        """Validiert pre-conditions for dynamics processing."""
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)

        # Measure dynamic range
        dynamic_range_db = self._measure_dynamic_range(audio_mono)
        crest_factor_db = self._measure_crest_factor(audio_mono)

        warnings = []

        # Very limited dynamic range - little to process
        if dynamic_range_db < 10:
            warnings.append(
                f"Limited dynamic range ({dynamic_range_db:.1f} dB). Dynamics processing may have limited effect."
            )

        # Extremely high dynamic range - difficult to process safely
        if dynamic_range_db > 60:
            warnings.append(
                f"Very high dynamic range ({dynamic_range_db:.1f} dB). Careful processing needed to avoid artifacts."
            )

        return PreCheckResult(
            passed=True,
            confidence=0.8,  # Usually high confidence for dynamics
            reasons=[],
            warnings=warnings,
            metadata={"dynamic_range_db": float(dynamic_range_db), "crest_factor_db": float(crest_factor_db)},
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in dynamics processing."""
        dynamic_range = pre_check.metadata.get("dynamic_range_db", 30.0)

        # Confidence highest for moderate dynamic range
        if 20 <= dynamic_range <= 50:
            return 0.9
        elif 10 <= dynamic_range <= 60:
            return 0.75
        else:
            return 0.6

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions for dynamics processing."""
        issues = []
        side_effects = []
        metrics = {}

        orig_mono = original if original.ndim == 1 else np.mean(original, axis=0)
        proc_mono = processed if processed.ndim == 1 else np.mean(processed, axis=0)

        # 1. Check dynamic range change
        orig_dr = self._measure_dynamic_range(orig_mono)
        proc_dr = self._measure_dynamic_range(proc_mono)
        dr_change = abs(proc_dr - orig_dr)

        metrics["original_dynamic_range_db"] = orig_dr
        metrics["processed_dynamic_range_db"] = proc_dr
        metrics["dynamic_range_change_db"] = dr_change

        if dr_change > 20:
            issues.append(f"Excessive dynamic range change: {dr_change:.1f} dB")

        # 2. Check for pumping artifacts
        pumping_score = self._detect_pumping(proc_mono, sr)
        metrics["pumping_score"] = pumping_score

        if pumping_score > 0.3:
            issues.append(f"Pumping artifacts detected: {pumping_score:.2f}")

        # 3. Check for distortion
        thd_original = self._estimate_thd(orig_mono, sr)
        thd_processed = self._estimate_thd(proc_mono, sr)
        thd_increase = thd_processed - thd_original

        metrics["thd_increase"] = thd_increase

        if thd_increase > 0.05:  # 5% THD increase
            issues.append(f"Distortion introduced: +{thd_increase:.2%} THD")

        # 4. Check transient preservation
        transient_ratio = self._check_transient_ratio(original, processed, sr)
        metrics["transient_ratio"] = transient_ratio

        if transient_ratio < 0.7 or transient_ratio > 1.3:
            side_effects.append(f"Transient level changed: {transient_ratio:.2f}x")

        # 5. Musical Goals validation (focus on Emotionalität & Natürlichkeit)
        try:
            orig_goals = self.musical_goals.measure_all(original, sr)
            proc_goals = self.musical_goals.measure_all(processed, sr)

            # Dynamics processing especially affects Emotionalität
            emotionalitaet_change = abs(proc_goals.get("emotionalitaet", 0.8) - orig_goals.get("emotionalitaet", 0.8))

            if emotionalitaet_change > 0.15:
                issues.append(f"Emotionalität significantly changed: {emotionalitaet_change:.2f}")

            metrics["musical_goals_original"] = orig_goals  # type: ignore[assignment]
            metrics["musical_goals_processed"] = proc_goals  # type: ignore[assignment]
        except Exception as e:
            side_effects.append(f"Musical Goals check failed: {e}")

        passed = len(issues) == 0
        quality_score = self._compute_dynamics_quality(metrics)

        return PostCheckResult(
            passed=passed, quality_score=quality_score, issues=issues, side_effects=side_effects, metrics=metrics
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet quality score for dynamics processing."""
        return self._compute_dynamics_quality(post_check.metrics)

    # Helper methods

    def _measure_dynamic_range(self, audio: np.ndarray) -> float:
        """Misst dynamic range in dB."""
        # RMS of loudest 5% vs quietest 5%
        sorted_abs = np.sort(np.abs(audio))
        n = len(sorted_abs)

        top_5_percent = sorted_abs[int(0.95 * n) :]
        bottom_5_percent = sorted_abs[: int(0.05 * n)]

        rms_top = np.sqrt(np.mean(top_5_percent**2))
        rms_bottom = np.sqrt(np.mean(bottom_5_percent**2))

        if rms_bottom < 1e-10:
            return 100.0  # Max measurable

        dr_db = 20 * np.log10(rms_top / rms_bottom)
        return float(dr_db)

    def _measure_crest_factor(self, audio: np.ndarray) -> float:
        """Misst crest factor (peak/RMS ratio) in dB."""
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))

        if rms < 1e-10:
            return 0.0

        crest_db = 20 * np.log10(peak / rms)
        return float(crest_db)

    def _detect_pumping(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt pumping artifacts (gain modulation at ~2-10 Hz)."""
        # Compute envelope
        from scipy.signal import hilbert

        analytic = hilbert(audio)
        analytic_arr = np.asarray(analytic, dtype=np.complex128)
        envelope = np.abs(analytic_arr)

        # Smooth envelope
        from scipy.ndimage import gaussian_filter1d

        smoothed = gaussian_filter1d(envelope, sigma=sr // 100)

        # Compute envelope modulation spectrum
        env_spectrum = np.abs(np.fft.rfft(smoothed))
        env_freqs = np.fft.rfftfreq(len(smoothed), 1 / sr)

        # Look for energy in pumping frequency range (2-10 Hz)
        pumping_band = (env_freqs >= 2) & (env_freqs <= 10)

        if not np.any(pumping_band):
            return 0.0

        pumping_energy = np.sum(env_spectrum[pumping_band] ** 2)
        total_energy = np.sum(env_spectrum**2)

        pumping_score = pumping_energy / (total_energy + 1e-10)

        return float(np.clip(pumping_score, 0.0, 1.0))

    def _estimate_thd(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt Total Harmonic Distortion."""
        # Simplified THD: ratio of HF energy to total energy
        sos_lp = signal.butter(2, 1000, "low", fs=sr, output="sos")
        sos_hp = signal.butter(2, 1000, "high", fs=sr, output="sos")

        fundamental = signal.sosfilt(sos_lp, audio)
        harmonics = signal.sosfilt(sos_hp, audio)

        fund_energy = np.mean(fundamental**2)
        harm_energy = np.mean(harmonics**2)

        if fund_energy < 1e-10:
            return 0.0

        thd = np.sqrt(harm_energy / fund_energy)
        return float(np.clip(thd, 0.0, 1.0))

    def _check_transient_ratio(self, original: np.ndarray, processed: np.ndarray, sr: int) -> float:
        """Misst how transient levels changed."""
        from scipy.ndimage import maximum_filter1d

        orig_mono = original if original.ndim == 1 else np.mean(original, axis=0)
        proc_mono = processed if processed.ndim == 1 else np.mean(processed, axis=0)

        window = int(sr * 0.01)  # 10ms

        orig_env = maximum_filter1d(np.abs(orig_mono), size=window)
        proc_env = maximum_filter1d(np.abs(proc_mono), size=window)

        # Find transients (peaks)
        threshold = np.mean(orig_env) + 2 * np.std(orig_env)
        transients = orig_env > threshold

        if not np.any(transients):
            return 1.0

        orig_transient_level = np.mean(orig_env[transients])
        proc_transient_level = np.mean(proc_env[transients])

        ratio = proc_transient_level / (orig_transient_level + 1e-10)
        return float(ratio)

    def _compute_dynamics_quality(self, metrics: dict[str, Any]) -> float:
        """Berechnet quality score for dynamics processing."""
        dr_change = metrics.get("dynamic_range_change_db", 10.0)
        pumping = metrics.get("pumping_score", 0.1)
        thd_increase = metrics.get("thd_increase", 0.01)
        transient_ratio = metrics.get("transient_ratio", 1.0)

        # Penalties
        dr_penalty = np.clip(dr_change / 20.0, 0.0, 1.0)
        pumping_penalty = pumping
        thd_penalty = np.clip(thd_increase / 0.05, 0.0, 1.0)
        transient_penalty = abs(transient_ratio - 1.0)

        quality = 1.0 - (0.3 * dr_penalty + 0.3 * pumping_penalty + 0.2 * thd_penalty + 0.2 * transient_penalty)

        return float(np.clip(quality, 0.0, 1.0))


# ============================================================================
# GENERIC SPECTRAL SAFETY WRAPPER
# ============================================================================


class GenericSpectralSafety(BaseSafetyWrapper):
    """
    Generic safety wrapper for spectral processing modules.

    Applies to:
    - EQ (parametric, graphic, shelving)
    - Filters (highpass, lowpass, bandpass, notch)
    - Harmonic exciters
    - Bandwidth extenders
    - Spectral tilt processors

    Ensures:
    - Spectral balance remains musical
    - No harsh resonances introduced
    - Phase coherence maintained
    - Musical Goals (Brillanz, Wärme, Natürlichkeit) preserved
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
            confidence_threshold=0.7,
            quality_threshold=0.75,
        )
        self.processing_mode = processing_mode
        self.musical_goals = MusicalGoalsChecker()

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for spectral processing."""
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)

        # Analyze spectral characteristics
        spectral_centroid = compute_spectral_centroid(audio_mono, sr)
        spectral_flatness = self._measure_spectral_flatness(audio_mono)

        warnings = []

        # Very bright signal - EQ boost may cause harshness
        if spectral_centroid > sr / 4:
            warnings.append(
                f"Very bright signal (centroid={spectral_centroid:.0f} Hz). High-frequency boost may cause harshness."
            )

        # Very flat spectrum - may be noise-like
        if spectral_flatness > 0.9:
            warnings.append(f"Flat spectrum (flatness={spectral_flatness:.2f}). Signal may be primarily noise.")

        return PreCheckResult(
            passed=True,
            confidence=0.85,
            reasons=[],
            warnings=warnings,
            metadata={"spectral_centroid_hz": float(spectral_centroid), "spectral_flatness": float(spectral_flatness)},
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in spectral processing."""
        # Generally high confidence for spectral processing
        flatness = pre_check.metadata.get("spectral_flatness", 0.5)

        # Lower confidence for noise-like signals
        if flatness > 0.9:
            return 0.6
        else:
            return 0.85

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions for spectral processing."""
        issues = []
        side_effects = []
        metrics = {}

        orig_mono = original if original.ndim == 1 else np.mean(original, axis=0)
        proc_mono = processed if processed.ndim == 1 else np.mean(processed, axis=0)

        # 1. Check spectral centroid change
        orig_centroid = compute_spectral_centroid(orig_mono, sr)
        proc_centroid = compute_spectral_centroid(proc_mono, sr)
        centroid_change_pct = abs(proc_centroid - orig_centroid) / (orig_centroid + 1e-10)

        metrics["original_centroid_hz"] = orig_centroid
        metrics["processed_centroid_hz"] = proc_centroid
        metrics["centroid_change_pct"] = centroid_change_pct

        if centroid_change_pct > 0.5:  # 50% change
            issues.append(f"Excessive spectral shift: {centroid_change_pct:.1%}")

        # 2. Check for harsh resonances
        harshness_score = self._detect_harshness(proc_mono, sr)
        metrics["harshness_score"] = harshness_score

        if harshness_score > 0.4:
            issues.append(f"Harsh resonances detected: {harshness_score:.2f}")

        # 3. Check phase coherence (for stereo)
        if original.ndim == 2 and processed.ndim == 2:
            orig_coherence = self._measure_phase_coherence(original)
            proc_coherence = self._measure_phase_coherence(processed)
            coherence_loss = orig_coherence - proc_coherence

            metrics["phase_coherence_loss"] = coherence_loss

            if coherence_loss > 0.2:
                side_effects.append(f"Phase coherence reduced: -{coherence_loss:.2f}")

        # 4. Musical Goals validation (Brillanz, Wärme, Natürlichkeit)
        try:
            orig_goals = self.musical_goals.measure_all(original, sr)
            proc_goals = self.musical_goals.measure_all(processed, sr)

            # Check important spectral goals
            for goal in ["brillanz", "waerme", "natuerlichkeit"]:
                if goal in orig_goals and goal in proc_goals:
                    change = abs(proc_goals[goal] - orig_goals[goal])
                    if change > 0.2:
                        side_effects.append(f"{goal.capitalize()} changed significantly: {change:.2f}")

            metrics["musical_goals_original"] = orig_goals  # type: ignore[assignment]
            metrics["musical_goals_processed"] = proc_goals  # type: ignore[assignment]
        except Exception as e:
            side_effects.append(f"Musical Goals check failed: {e}")

        passed = len(issues) == 0
        quality_score = self._compute_spectral_quality(metrics)

        return PostCheckResult(
            passed=passed, quality_score=quality_score, issues=issues, side_effects=side_effects, metrics=metrics
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet quality score for spectral processing."""
        return self._compute_spectral_quality(post_check.metrics)

    # Helper methods

    def _measure_spectral_flatness(self, audio: np.ndarray) -> float:
        """Misst spectral flatness (Wiener entropy)."""
        spectrum = np.abs(np.fft.rfft(audio))
        spectrum = spectrum + 1e-10  # Avoid log(0)

        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)

        flatness = geometric_mean / arithmetic_mean
        return float(flatness)

    def _detect_harshness(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt harsh high-frequency resonances."""
        # Filter to harsh frequency range (3-7.5 kHz to avoid Nyquist)
        max_freq = min(7500, sr / 2 - 100)
        sos = signal.butter(4, [3000, max_freq], "bp", fs=sr, output="sos")
        harsh_band = signal.sosfilt(sos, audio)

        # Measure energy in harsh band relative to total
        harsh_energy = np.mean(harsh_band**2)
        total_energy = np.mean(audio**2)

        harshness = harsh_energy / (total_energy + 1e-10)

        # Also check for sharp peaks in this range
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        harsh_freqs = (freqs >= 3000) & (freqs <= 8000)
        if np.any(harsh_freqs):
            harsh_spectrum = spectrum[harsh_freqs]
            peak_to_mean = np.max(harsh_spectrum) / (np.mean(harsh_spectrum) + 1e-10)

            # Sharp peak = potential resonance
            if peak_to_mean > 5:
                harshness *= 1.5

        return float(np.clip(harshness, 0.0, 1.0))

    def _measure_phase_coherence(self, stereo_audio: np.ndarray) -> float:
        """Misst phase coherence between stereo channels."""
        if stereo_audio.shape[0] != 2:
            return 1.0

        left = stereo_audio[0]
        right = stereo_audio[1]

        # Cross-correlation (NaN-safe)
        _sl = float(np.std(left))
        _sr = float(np.std(right))
        if _sl > 1e-8 and _sr > 1e-8:
            _la = left - left.mean()
            _ra = right - right.mean()
            _nl = float(np.linalg.norm(_la))
            _nr = float(np.linalg.norm(_ra))
            correlation = float(np.dot(_la, _ra) / (_nl * _nr + 1e-10))
            if not np.isfinite(correlation):
                correlation = 1.0
        else:
            correlation = 1.0

        # Map to 0-1 (0 = out of phase, 1 = in phase)
        coherence = (correlation + 1.0) / 2.0

        return float(coherence)

    def _compute_spectral_quality(self, metrics: dict[str, Any]) -> float:
        """Berechnet quality score for spectral processing."""
        centroid_change = metrics.get("centroid_change_pct", 0.1)
        harshness = metrics.get("harshness_score", 0.2)
        coherence_loss = metrics.get("phase_coherence_loss", 0.0)

        # Penalties
        centroid_penalty = np.clip(centroid_change / 0.5, 0.0, 1.0)
        harshness_penalty = harshness
        coherence_penalty = np.clip(coherence_loss / 0.2, 0.0, 1.0)

        quality = 1.0 - (0.4 * centroid_penalty + 0.4 * harshness_penalty + 0.2 * coherence_penalty)

        return float(np.clip(quality, 0.0, 1.0))


# ============================================================================
# GENERIC SPATIAL SAFETY WRAPPER
# ============================================================================


class GenericSpatialSafety(BaseSafetyWrapper):
    """
    Generic safety wrapper for spatial processing modules.

    Applies to:
    - Stereo wideners
    - Panners
    - Mid/Side processors
    - Spatial enhancers

    Ensures:
    - Mono compatibility maintained
    - No phase cancellation
    - Center image preserved
    - Musical Goals (Transparenz, Natürlichkeit) preserved
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
            quality_threshold=0.7,
        )
        self.processing_mode = processing_mode
        self.musical_goals = MusicalGoalsChecker()

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for spatial processing."""
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        # Spatial processing requires stereo
        if audio.ndim == 1:
            return PreCheckResult(passed=False, confidence=0.0, reasons=["Spatial processing requires stereo input"])

        if audio.shape[0] != 2:
            return PreCheckResult(passed=False, confidence=0.0, reasons=[f"Expected 2 channels, got {audio.shape[0]}"])

        # Measure stereo width
        stereo_width = self._measure_stereo_width(audio)

        warnings = []

        if stereo_width < 0.2:
            warnings.append(f"Nearly mono signal (width={stereo_width:.2f}). Widening may sound artificial.")

        if stereo_width > 0.8:
            warnings.append(f"Already very wide (width={stereo_width:.2f}). Further widening may cause phase issues.")

        return PreCheckResult(
            passed=True,
            confidence=0.8,
            reasons=[],
            warnings=warnings,
            metadata={"original_stereo_width": float(stereo_width)},
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in spatial processing."""
        width = pre_check.metadata.get("original_stereo_width", 0.5)

        # Confidence highest for moderate width signals
        if 0.3 <= width <= 0.7:
            return 0.85
        else:
            return 0.65

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions for spatial processing."""
        issues = []
        side_effects = []
        metrics = {}

        # 1. Check stereo width change
        orig_width = self._measure_stereo_width(original)
        proc_width = self._measure_stereo_width(processed)

        metrics["original_width"] = orig_width
        metrics["processed_width"] = proc_width
        metrics["width_change"] = proc_width - orig_width

        if proc_width > 0.95:
            issues.append(f"Excessive stereo width: {proc_width:.2f}")

        # 2. Check mono compatibility
        mono_compat = self._check_mono_compatibility(original, processed)
        metrics["mono_compatibility"] = mono_compat

        if mono_compat < 0.7:
            issues.append(f"Poor mono compatibility: {mono_compat:.2f}")

        # 3. Check center image preservation
        center_preservation = self._check_center_preservation(original, processed)
        metrics["center_preservation"] = center_preservation

        if center_preservation < 0.75:
            side_effects.append(f"Center image weakened: {center_preservation:.2f}")

        # 4. Musical Goals validation
        try:
            self.musical_goals.measure_all(original, sr)
            proc_goals = self.musical_goals.measure_all(processed, sr)

            # Spatial processing affects Transparenz & Natürlichkeit
            mode_config = PROCESSING_MODE_CONFIGS.get(self.processing_mode)
            thresholds = mode_config.musical_goals if mode_config else {}

            for goal in ["transparenz", "natuerlichkeit"]:
                if goal in proc_goals and goal in thresholds:
                    threshold = thresholds[goal]
                    if proc_goals[goal] < threshold:
                        issues.append(f"{goal.capitalize()}: {proc_goals[goal]:.2f} < {threshold:.2f}")

            metrics["musical_goals_processed"] = proc_goals  # type: ignore[assignment]
        except Exception as e:
            side_effects.append(f"Musical Goals check failed: {e}")

        passed = len(issues) == 0
        quality_score = self._compute_spatial_quality(metrics)

        return PostCheckResult(
            passed=passed, quality_score=quality_score, issues=issues, side_effects=side_effects, metrics=metrics
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet quality score for spatial processing."""
        return self._compute_spatial_quality(post_check.metrics)

    # Helper methods

    def _measure_stereo_width(self, stereo_audio: np.ndarray) -> float:
        """Misst stereo width (0 = mono, 1 = maximum width)."""
        left = stereo_audio[0]
        right = stereo_audio[1]

        # Correlation-based width metric (NaN-safe)
        _sl = float(np.std(left))
        _sr = float(np.std(right))
        if _sl > 1e-8 and _sr > 1e-8:
            _la = left - left.mean()
            _ra = right - right.mean()
            _nl = float(np.linalg.norm(_la))
            _nr = float(np.linalg.norm(_ra))
            correlation = float(np.dot(_la, _ra) / (_nl * _nr + 1e-10))
            if not np.isfinite(correlation):
                correlation = 1.0
        else:
            correlation = 1.0  # Constant signals — treat as mono

        # Width = 1 - correlation
        width = 1.0 - correlation

        return float(np.clip(width, 0.0, 1.0))

    def _check_mono_compatibility(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Prüft mono compatibility (sum to mono without phase cancellation)."""
        # Sum to mono
        orig_mono = np.sum(original, axis=0)
        proc_mono = np.sum(processed, axis=0)

        # Energy comparison
        orig_energy = np.mean(orig_mono**2)
        proc_energy = np.mean(proc_mono**2)

        # Good mono compatibility if mono sum has similar energy
        if orig_energy < 1e-10:
            return 1.0

        compatibility = proc_energy / orig_energy

        # Should be close to 1.0 (or 2.0 if originally mono)
        # Penalize if much lower (phase cancellation)
        if compatibility < 0.5:
            compatibility *= 2  # Scale back up

        return float(np.clip(compatibility, 0.0, 1.0))

    def _check_center_preservation(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Prüft if center image (common L+R content) is preserved."""
        # Mid channel
        orig_mid = (original[0] + original[1]) / 2.0
        proc_mid = (processed[0] + processed[1]) / 2.0

        # Correlation between mid channels
        correlation = compute_correlation(orig_mid, proc_mid)

        return float(np.clip(correlation, 0.0, 1.0))

    def _compute_spatial_quality(self, metrics: dict[str, Any]) -> float:
        """Berechnet quality score for spatial processing."""
        proc_width = metrics.get("processed_width", 0.5)
        mono_compat = metrics.get("mono_compatibility", 0.9)
        center_pres = metrics.get("center_preservation", 0.9)

        # Penalize excessive width
        width_penalty = 0.0 if proc_width < 0.9 else (proc_width - 0.9) / 0.1

        quality = 0.4 * mono_compat + 0.4 * center_pres + 0.2 * (1.0 - width_penalty)

        return float(np.clip(quality, 0.0, 1.0))
logger = logging.getLogger(__name__)
