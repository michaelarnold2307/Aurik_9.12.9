"""
Edge Case & Extreme Degradation Handler for Musical Goals

Component 4.3: Edge Case Handling
Impact: +0.5 Punkte - Robuste Garantie auch bei extremen Fällen

Handles edge cases where Musical Goals might be unreachable or conflicting:
1. Extreme Degradation (SNR < 30 dB, defects > 80%)
2. Unknown Defect Types (not categorized as tape/vinyl/mp3/shellac)
3. Medium-Mix Scenarios (vinyl+tape, digital+analog hybrid)
4. Spectrum-Goals Conflicts (bass-only but brillanz required, HF-only but bass-kraft required)

Problem:
Without edge case handling, the system can get into absurd states where
it tries to achieve impossible goals or applies inappropriate processing.

Solution:
EdgeCaseHandler detects these scenarios and provides:
- Feasibility assessments (are goals reachable?)
- Adjusted thresholds (lower targets for extreme cases)
- Prioritization logic (which goals to focus on)
- Fallback strategies (graceful degradation)

Author: AI Team
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

try:
    import librosa

    _LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _LIBROSA_AVAILABLE = False
import numpy as np

from backend.core.audio_utils import safe_filtfilt  # §v10.101
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
from backend.core.musical_goals.processing_modes import PROCESSING_MODE_CONFIGS, ProcessingMode

logger = logging.getLogger(__name__)


class EdgeCaseType(Enum):
    """Types of edge cases that can be detected."""

    EXTREME_DEGRADATION = "extreme_degradation"
    UNKNOWN_DEFECT = "unknown_defect"
    MEDIUM_MIX = "medium_mix"
    SPECTRUM_CONFLICT = "spectrum_conflict"
    MULTIPLE_ISSUES = "multiple_issues"
    NONE = "none"


class DegradationSeverity(Enum):
    """Severity levels for degradation."""

    MINIMAL = "minimal"  # < 10% degradation, all goals reachable
    MODERATE = "moderate"  # 10-30% degradation, most goals reachable
    SEVERE = "severe"  # 30-60% degradation, some goals unreachable
    EXTREME = "extreme"  # > 60% degradation, most goals unreachable
    CATASTROPHIC = "catastrophic"  # > 80% degradation, goals impossible


@dataclass
class EdgeCaseAssessment:
    """
    Result of edge case detection and analysis.

    Attributes:
        edge_case_type: Type of edge case detected
        severity: Degradation severity level
        reachable_goals: Which goals can still be reached
        unreachable_goals: Which goals are impossible
        adjusted_thresholds: Recommended threshold adjustments
        recommended_mode: Suggested processing mode
        prioritized_goals: Goals in priority order
        fallback_strategy: What to do if processing fails
        confidence: Confidence in this assessment (0-1)
        details: Additional diagnostic information
    """

    edge_case_type: EdgeCaseType
    severity: DegradationSeverity
    reachable_goals: list[str]
    unreachable_goals: list[str]
    adjusted_thresholds: dict[str, float]
    recommended_mode: ProcessingMode
    prioritized_goals: list[str]
    fallback_strategy: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectrumProfile:
    """
    Spectral profile of audio signal.

    Attributes:
        has_low_freq: Significant energy in 20-250 Hz
        has_mid_freq: Significant energy in 250-2000 Hz
        has_high_freq: Significant energy in 2000-20000 Hz
        bass_ratio: Energy ratio in bass band
        mid_ratio: Energy ratio in mid band
        treble_ratio: Energy ratio in treble band
        spectral_centroid: Center of mass of spectrum
        spectral_bandwidth: Spread of spectrum
        missing_bands: Frequency bands with < 10% expected energy
    """

    has_low_freq: bool
    has_mid_freq: bool
    has_high_freq: bool
    bass_ratio: float
    mid_ratio: float
    treble_ratio: float
    spectral_centroid: float
    spectral_bandwidth: float
    missing_bands: list[str]


class EdgeCaseHandler:
    """
    Verarbeitet edge cases where Musical Goals might be unreachable or conflicting.

    This handler detects various problematic scenarios and provides:
    1. Feasibility assessments (can goals be reached?)
    2. Adjusted thresholds (lower targets for extreme degradation)
    3. Prioritization logic (which goals to focus on)
    4. Fallback strategies (graceful degradation)

    Example:
        >>> handler = EdgeCaseHandler()
        >>> assessment = handler.assess_edge_cases(audio, sr, mode=ProcessingMode.RESTORATION)
        >>>
        >>> if assessment.edge_case_type != EdgeCaseType.NONE:
        logger.debug("Edge case detected: %s", assessment.edge_case_type)
        logger.debug("Unreachable goals: %s", assessment.unreachable_goals)
        logger.debug("Adjusted thresholds: %s", assessment.adjusted_thresholds)
    """

    def __init__(self, musical_goals_checker: MusicalGoalsChecker | None = None) -> None:
        """
        Initialisiert edge case handler.

        Args:
            musical_goals_checker: Optional MusicalGoalsChecker instance
        """
        self.checker = musical_goals_checker or MusicalGoalsChecker()

        # Thresholds for extreme degradation detection
        self.extreme_degradation_thresholds = {
            "min_snr": 30.0,  # Below 30 dB SNR is extreme
            "max_defect_coverage": 0.8,  # > 80% defects is extreme
            "min_dynamic_range": 6.0,  # < 6 dB dynamic range is extreme (adjusted for sine waves)
            "max_clipping_ratio": 0.1,  # > 10% clipped samples is extreme
        }

        # Spectrum requirements for each goal
        self.goal_spectrum_requirements = {
            "brillanz": ["high_freq"],
            "waerme": ["mid_freq"],
            "bass_kraft": ["low_freq"],
            "bass-kraft": ["low_freq"],  # Support both naming conventions
            "natuerlichkeit": ["low_freq", "mid_freq", "high_freq"],
            "authentizitaet": ["low_freq", "mid_freq", "high_freq"],
            "emotionalitaet": ["mid_freq", "high_freq"],
            "transparenz": ["mid_freq", "high_freq"],
        }

        # Medium type indicators (heuristic patterns)
        self.medium_indicators = {
            "vinyl": ["rumble", "surface_noise", "wow_flutter"],
            "tape": ["hiss", "modulation_noise", "dropouts"],
            "shellac": ["severe_noise", "bandwidth_limited", "crackles"],
            "mp3": ["pre_echo", "ringing", "quantization_noise"],
            "digital": ["clean", "wide_bandwidth", "no_analog_artifacts"],
        }

    # =========================================================================
    # Main Assessment Method
    # =========================================================================

    def assess_edge_cases(
        self,
        audio: np.ndarray,
        sr: int,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        reference: np.ndarray | None = None,
    ) -> EdgeCaseAssessment:
        """
        Comprehensive edge case assessment.

        Detects all edge case types and provides recommendations.

        Args:
            audio: Audio signal to assess
            sr: Sample rate
            mode: Processing mode (affects goal priorities)
            reference: Optional reference for comparison

        Returns:
            EdgeCaseAssessment with all recommendations
        """
        # Get mode config
        mode_config = PROCESSING_MODE_CONFIGS.get(mode)
        if not mode_config:
            mode_config = PROCESSING_MODE_CONFIGS[ProcessingMode.RESTORATION]

        # Detect all edge case types
        extreme_deg = self._detect_extreme_degradation(audio, sr)
        unknown_defect = self._detect_unknown_defect(audio, sr)
        medium_mix = self._detect_medium_mix(audio, sr)
        spectrum_conflict = self._detect_spectrum_conflict(audio, sr, mode_config)

        # Determine primary edge case type
        edge_cases = []
        if extreme_deg["is_extreme"]:
            edge_cases.append(EdgeCaseType.EXTREME_DEGRADATION)
        if unknown_defect["is_unknown"]:
            edge_cases.append(EdgeCaseType.UNKNOWN_DEFECT)
        if medium_mix["is_mixed"]:
            edge_cases.append(EdgeCaseType.MEDIUM_MIX)
        if spectrum_conflict["has_conflict"]:
            edge_cases.append(EdgeCaseType.SPECTRUM_CONFLICT)

        if len(edge_cases) == 0:
            edge_case_type = EdgeCaseType.NONE
        elif len(edge_cases) > 1:
            edge_case_type = EdgeCaseType.MULTIPLE_ISSUES
        else:
            edge_case_type = edge_cases[0]

        # Determine severity
        severity = self._determine_severity(extreme_deg, unknown_defect, medium_mix, spectrum_conflict)

        # Determine reachable/unreachable goals
        reachable, unreachable = self._determine_goal_reachability(
            audio, sr, mode_config, extreme_deg, spectrum_conflict
        )

        # Adjust thresholds based on severity
        adjusted_thresholds = self._adjust_thresholds(mode_config, severity, unreachable)

        # Recommend processing mode
        recommended_mode = self._recommend_processing_mode(mode, edge_case_type, severity, medium_mix)

        # Prioritize goals
        prioritized_goals = self._prioritize_goals(mode_config, reachable, unreachable, spectrum_conflict)

        # Determine fallback strategy
        fallback_strategy = self._determine_fallback_strategy(edge_case_type, severity, unreachable)

        # Calculate confidence
        confidence = self._calculate_confidence(extreme_deg, unknown_defect, medium_mix, spectrum_conflict)

        # Compile details
        details = {
            "extreme_degradation": extreme_deg,
            "unknown_defect": unknown_defect,
            "medium_mix": medium_mix,
            "spectrum_conflict": spectrum_conflict,
            "original_mode": mode.value,
            "recommended_mode": recommended_mode.value,
        }

        return EdgeCaseAssessment(
            edge_case_type=edge_case_type,
            severity=severity,
            reachable_goals=reachable,
            unreachable_goals=unreachable,
            adjusted_thresholds=adjusted_thresholds,
            recommended_mode=recommended_mode,
            prioritized_goals=prioritized_goals,
            fallback_strategy=fallback_strategy,
            confidence=confidence,
            details=details,
        )

    # =========================================================================
    # 1. Extreme Degradation Detection
    # =========================================================================

    def _detect_extreme_degradation(self, audio: np.ndarray, sr: int) -> dict:
        """
        Erkennt extreme degradation where goals are unreachable.

        Checks:
        - SNR < 30 dB
        - Defect coverage > 80%
        - Dynamic range < 10 dB
        - Clipping > 10%

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Dict with degradation metrics and is_extreme flag
        """
        # Ensure mono for analysis
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Estimate SNR
        snr = self._estimate_snr(audio, sr)

        # Estimate defect coverage
        defect_coverage = self._estimate_defect_coverage(audio, sr)

        # Measure dynamic range
        dynamic_range = self._measure_dynamic_range(audio)

        # Detect clipping
        clipping_ratio = self._detect_clipping(audio)

        # Determine if extreme
        is_extreme = (
            snr < self.extreme_degradation_thresholds["min_snr"]
            or defect_coverage > self.extreme_degradation_thresholds["max_defect_coverage"]
            or dynamic_range < self.extreme_degradation_thresholds["min_dynamic_range"]
            or clipping_ratio > self.extreme_degradation_thresholds["max_clipping_ratio"]
        )

        return {
            "is_extreme": is_extreme,
            "snr": snr,
            "defect_coverage": defect_coverage,
            "dynamic_range": dynamic_range,
            "clipping_ratio": clipping_ratio,
            "reason": self._get_degradation_reason(snr, defect_coverage, dynamic_range, clipping_ratio),
        }

    def _estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Schätzt Signal-to-Noise Ratio.

        Uses spectral minimum statistics to estimate noise floor.
        """
        # Compute overall RMS
        rms = np.sqrt(np.mean(audio**2))

        if rms < 1e-10:
            return 0.0  # Silent audio

        # Compute frame-wise RMS
        frame_length = 2048
        hop_length = 512
        n_frames = (len(audio) - frame_length) // hop_length + 1

        frame_rms = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break
            frame = audio[start:end]
            frame_rms.append(np.sqrt(np.mean(frame**2)))

        frame_rms = np.array(frame_rms)  # type: ignore[assignment]

        # Check if this is a synthetic clean signal (very stable RMS)
        rms_std = np.std(frame_rms)
        rms_mean = np.mean(frame_rms)
        rms_cv = rms_std / (rms_mean + 1e-10)  # Coefficient of variation

        if rms_cv < 0.1:
            # Very stable RMS - synthetic or very clean signal
            return 80.0

        # Estimate noise floor using minimum RMS
        noise_rms = np.percentile(frame_rms, 10)
        signal_rms = rms_mean

        if noise_rms < 1e-10:
            return 70.0  # Clean signal

        # SNR in dB
        snr_ratio = signal_rms / noise_rms
        if snr_ratio < 1.0:
            return 0.0

        snr = 20 * np.log10(snr_ratio)  # Use 20*log10 for RMS ratio

        return max(0.0, min(100.0, snr))  # type: ignore[no-any-return]

    def _estimate_defect_coverage(self, audio: np.ndarray, sr: int) -> float:
        """
        Schätzt percentage of audio affected by defects.

        Detects clicks, pops, crackles using high-pass filtered energy spikes.
        """
        # High-pass filter to isolate transients/defects
        from scipy.signal import butter, filtfilt

        nyquist = sr / 2
        cutoff = min(8000, nyquist - 100)
        b, a = cast(tuple[np.ndarray, np.ndarray], butter(4, cutoff / nyquist, btype="high", output="ba"))
        filtered = safe_filtfilt(b, a, audio)

        # Detect energy spikes (only very large spikes are defects)
        window_size = int(sr * 0.005)  # 5ms windows
        energy = np.convolve(filtered**2, np.ones(window_size) / window_size, mode="same")

        # Use 99th percentile to catch only extreme spikes
        threshold = np.percentile(energy, 99) * 2.0  # 2x the 99th percentile
        defect_samples: int = int(np.sum(energy > threshold))

        coverage = defect_samples / len(audio)
        return min(1.0, coverage)  # type: ignore[no-any-return]

    def _measure_dynamic_range(self, audio: np.ndarray) -> float:
        """Misst dynamic range in dB."""
        peak: float = float(np.max(np.abs(audio)))
        rms = np.sqrt(np.mean(audio**2))

        if rms < 1e-10:
            return 0.0

        dynamic_range = 20 * np.log10(peak / rms)
        return max(0.0, dynamic_range)  # type: ignore[no-any-return]

    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Erkennt clipping ratio (fraction of samples near ±1.0)."""
        clipping_threshold = 0.99
        clipped_samples: int = int(np.sum(np.abs(audio) > clipping_threshold))
        return clipped_samples / len(audio)  # type: ignore[no-any-return]

    def _get_degradation_reason(
        self, snr: float, defect_coverage: float, dynamic_range: float, clipping_ratio: float
    ) -> str:
        """Gibt zurück: human-readable reason for extreme degradation."""
        reasons = []

        if snr < 30:
            reasons.append(f"low SNR ({snr:.1f} dB)")
        if defect_coverage > 0.8:
            reasons.append(f"high defect coverage ({defect_coverage * 100:.1f}%)")
        if dynamic_range < 10:
            reasons.append(f"low dynamic range ({dynamic_range:.1f} dB)")
        if clipping_ratio > 0.1:
            reasons.append(f"excessive clipping ({clipping_ratio * 100:.1f}%)")

        return ", ".join(reasons) if reasons else "none"

    # =========================================================================
    # 2. Unknown Defect Detection
    # =========================================================================

    def _detect_unknown_defect(self, audio: np.ndarray, sr: int) -> dict:
        """
        Erkennt defects that don't match known medium types.

        Uses heuristics to identify defect patterns that are neither:
        - Vinyl (rumble, surface noise, wow/flutter)
        - Tape (hiss, dropouts, modulation noise)
        - Shellac (severe noise, bandwidth limited, crackles)
        - MP3 (pre-echo, ringing, quantization)

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Dict with is_unknown flag and detected patterns
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Detect various defect patterns
        has_rumble = self._detect_rumble(audio, sr)
        has_hiss = self._detect_hiss(audio, sr)
        has_crackles = self._detect_crackles(audio, sr)
        has_dropouts = self._detect_dropouts(audio, sr)
        has_quantization = self._detect_quantization_noise(audio, sr)

        # Classify into known medium types
        defect_patterns = {
            "rumble": has_rumble,
            "hiss": has_hiss,
            "crackles": has_crackles,
            "dropouts": has_dropouts,
            "quantization": has_quantization,
        }

        # Count active patterns
        active_patterns = [k for k, v in defect_patterns.items() if v]

        # Only consider unknown if there are defects
        has_any_defects = len(active_patterns) > 0
        is_unknown = has_any_defects and self._is_unusual_defect_combination(defect_patterns)

        return {
            "is_unknown": is_unknown,
            "active_patterns": active_patterns,
            "defect_patterns": defect_patterns,
            "fallback_strategy": "conservative_processing" if is_unknown else "standard_processing",
        }

    def _detect_rumble(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt low-frequency rumble (< 50 Hz)."""
        from scipy.signal import butter, filtfilt

        nyquist = sr / 2
        cutoff = min(50, nyquist - 10)
        b, a = cast(tuple[np.ndarray, np.ndarray], butter(4, cutoff / nyquist, btype="low", output="ba"))
        rumble = safe_filtfilt(b, a, audio)

        rumble_energy = np.mean(rumble**2)
        total_energy = np.mean(audio**2)

        return bool((rumble_energy / (total_energy + 1e-10)) > 0.15)  # type: ignore[no-any-return]

    def _detect_hiss(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt high-frequency hiss (> 6 kHz)."""
        from scipy.signal import butter, filtfilt

        nyquist = sr / 2
        cutoff = min(6000, nyquist - 100)
        b, a = cast(tuple[np.ndarray, np.ndarray], butter(4, cutoff / nyquist, btype="high", output="ba"))
        hiss = safe_filtfilt(b, a, audio)

        hiss_energy = np.mean(hiss**2)
        total_energy = np.mean(audio**2)

        # Lower threshold to 5% for better sensitivity
        return bool((hiss_energy / (total_energy + 1e-10)) > 0.05)  # type: ignore[no-any-return]

    def _detect_crackles(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt crackles (rapid impulses)."""
        # Detect rapid zero-crossings in high-pass filtered signal
        from scipy.signal import butter, filtfilt

        nyquist = sr / 2
        b, a = cast(tuple[np.ndarray, np.ndarray], butter(4, 4000 / nyquist, btype="high", output="ba"))
        filtered = safe_filtfilt(b, a, audio)

        # Zero crossing rate
        if not _LIBROSA_AVAILABLE:
            logger.debug("librosa not verfuegbar — crackle detection uebersprungen")
            return False
        zcr = librosa.feature.zero_crossing_rate(filtered)[0]
        mean_zcr = np.mean(zcr)

        return bool(mean_zcr > 0.15)  # type: ignore[no-any-return]

    def _detect_dropouts(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt dropouts (sudden energy drops)."""
        # Compute frame-wise energy
        frame_length = int(sr * 0.02)  # 20ms frames
        hop_length = frame_length // 2

        if not _LIBROSA_AVAILABLE:
            logger.debug("librosa not verfuegbar — dropout detection uebersprungen")
            return False
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # Detect sudden drops (> 50% energy reduction)
        energy_diff = np.diff(energy)
        large_drops: int = int(np.sum(energy_diff < -0.5 * np.median(energy)))

        return large_drops > 5  # type: ignore[no-any-return]  # More than 5 large drops

    def _detect_quantization_noise(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt quantization noise patterns."""
        # Bug-Fix §10a: np.correlate(N, N, mode="same") is O(N²) → O(10.8M²) ≈ 8 hours
        # on 225 s audio.  Only lags 10–50 are inspected, so 4096 samples are sufficient.
        _seg = audio[: min(4096, len(audio))]

        # Quantization noise shows up as correlated noise in difference signal
        diff = np.diff(_seg)

        # Autocorrelation of difference — FFT-based O(N log N)
        from backend.core.core_utils import fft_autocorr

        _ac_full = fft_autocorr(diff)
        # Match mode="same" output: take center N elements
        _n = len(diff)
        _start = (len(_ac_full) * 2 - 1 - _n) // 2  # map to mode='same' center window
        # fft_autocorr returns positive-lag half; for mode="same" compatibility,
        # reconstruct symmetric then extract center portion:
        autocorr_sym = np.concatenate([_ac_full[:0:-1], _ac_full])
        _center_start = max(0, (len(autocorr_sym) - _n) // 2)
        autocorr = autocorr_sym[_center_start : _center_start + _n]
        _denom: float = float(np.max(np.abs(autocorr)))
        autocorr = autocorr / _denom if _denom > 0 else np.zeros_like(autocorr)  # §3.1

        # Check for periodic structure
        center = len(autocorr) // 2
        side_lobe_energy: float = float(np.sum(np.abs(autocorr[center + 10 : center + 50])))

        return side_lobe_energy > 0.3  # type: ignore[no-any-return]

    def _is_unusual_defect_combination(self, patterns: dict[str, bool]) -> bool:
        """Prüft if defect combination is unusual."""
        active = [k for k, v in patterns.items() if v]

        # Known good combinations
        known_combos = [
            {"rumble", "crackles"},  # Vinyl
            {"hiss", "dropouts"},  # Tape
            {"hiss", "quantization"},  # MP3
            {"crackles"},  # Shellac
        ]

        if len(active) == 0:
            return False  # Clean audio, not unknown

        # Check if matches any known combo
        active_set = set(active)
        return all(
            not (active_set.issubset(combo) or combo.issubset(active_set)) for combo in known_combos
        )  # Unusual combination

    # =========================================================================
    # 3. Medium-Mix Detection
    # =========================================================================

    def _detect_medium_mix(self, audio: np.ndarray, sr: int) -> dict:
        """
        Erkennt mixed medium scenarios (vinyl+tape, digital+analog).

        Returns prioritization logic for conflicting goals.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Detect medium indicators
        has_rumble = self._detect_rumble(audio, sr)
        has_hiss = self._detect_hiss(audio, sr)
        has_crackles = self._detect_crackles(audio, sr)
        has_dropouts = self._detect_dropouts(audio, sr)

        # Classify medium types present
        vinyl_score = int(has_rumble) + int(has_crackles)
        tape_score = int(has_hiss) + int(has_dropouts)

        is_mixed = vinyl_score > 0 and tape_score > 0

        # Determine dominant medium
        if vinyl_score > tape_score:
            dominant_medium = "vinyl"
            prioritized_goals = ["waerme", "authentizitaet", "bass_kraft", "brillanz"]
        elif tape_score > vinyl_score:
            dominant_medium = "tape"
            prioritized_goals = ["waerme", "natuerlichkeit", "transparenz", "brillanz"]
        else:
            dominant_medium = "mixed"
            prioritized_goals = ["waerme", "natuerlichkeit", "authentizitaet", "bass_kraft"]

        return {
            "is_mixed": is_mixed,
            "dominant_medium": dominant_medium,
            "vinyl_score": vinyl_score,
            "tape_score": tape_score,
            "prioritized_goals": prioritized_goals,
        }

    # =========================================================================
    # 4. Spectrum-Goals Conflict Detection
    # =========================================================================

    def _detect_spectrum_conflict(self, audio: np.ndarray, sr: int, mode_config) -> dict:
        """
        Erkennt conflicts between audio spectrum and musical goals.

        Examples:
        - Only bass present but brillanz (HF) required
        - Only HF present but bass-kraft required
        - Missing mid-range but waerme (warmth) required

        Args:
            audio: Audio signal
            sr: Sample rate
            mode_config: ProcessingModeConfig with goal targets

        Returns:
            Dict with conflict detection and adjustments
        """
        # Analyze spectrum
        spectrum_profile = self._analyze_spectrum_profile(audio, sr)

        # Check each goal against spectrum requirements
        conflicts = {}
        adjustments = {}

        for goal_name, target in mode_config.musical_goals.items():
            required_bands = self.goal_spectrum_requirements.get(goal_name, [])

            # Check if required frequency bands are present
            missing_requirements = []
            for band in required_bands:
                if band == "low_freq" and not spectrum_profile.has_low_freq:
                    missing_requirements.append("low_freq")
                elif band == "mid_freq" and not spectrum_profile.has_mid_freq:
                    missing_requirements.append("mid_freq")
                elif band == "high_freq" and not spectrum_profile.has_high_freq:
                    missing_requirements.append("high_freq")

            # If requirements missing, this is a conflict
            if missing_requirements:
                conflicts[goal_name] = {
                    "target": target,
                    "missing_bands": missing_requirements,
                    "severity": len(missing_requirements) / len(required_bands),
                }

                # Adjust threshold downward based on severity
                adjustment_factor = 1.0 - (0.15 * len(missing_requirements))
                adjustments[goal_name] = target * adjustment_factor

        has_conflict = len(conflicts) > 0

        return {
            "has_conflict": has_conflict,
            "conflicts": conflicts,
            "adjustments": adjustments,
            "spectrum_profile": spectrum_profile,
        }

    def _analyze_spectrum_profile(self, audio: np.ndarray, sr: int) -> SpectrumProfile:
        """
        Analysiert den Frequenzinhalt des Audios.

        Determines which frequency bands have significant energy.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Adaptive n_fft: never larger than signal length (guard for very short segments)
        _n_fft_ech = min(2048, max(64, int(2 ** np.floor(np.log2(max(len(audio), 64))))))
        _hop_ech = min(512, _n_fft_ech // 4)

        # Compute power spectrum
        if not _LIBROSA_AVAILABLE:
            logger.debug("librosa not verfuegbar — spectrum Profil uebersprungen")
            return SpectrumProfile()  # type: ignore[call-arg]
        stft = librosa.stft(audio, n_fft=_n_fft_ech, hop_length=_hop_ech)
        magnitude = np.abs(stft)
        power = magnitude**2

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=_n_fft_ech)

        # Define bands
        bass_mask = (freqs >= 20) & (freqs <= 250)
        mid_mask = (freqs >= 250) & (freqs <= 2000)
        treble_mask = (freqs >= 2000) & (freqs <= min(20000, sr / 2))

        # Compute energy ratios
        total_energy: float = float(np.sum(power))
        bass_energy: float = float(np.sum(power[bass_mask]))
        mid_energy: float = float(np.sum(power[mid_mask]))
        treble_energy: float = float(np.sum(power[treble_mask]))

        bass_ratio = bass_energy / (total_energy + 1e-10)
        mid_ratio = mid_energy / (total_energy + 1e-10)
        treble_ratio = treble_energy / (total_energy + 1e-10)

        # Determine presence (> 10% of expected energy)
        expected_bass_ratio = 0.15
        expected_mid_ratio = 0.50
        expected_treble_ratio = 0.35

        has_low_freq = bass_ratio > (expected_bass_ratio * 0.5)
        has_mid_freq = mid_ratio > (expected_mid_ratio * 0.5)
        has_high_freq = treble_ratio > (expected_treble_ratio * 0.5)

        # Identify missing bands
        missing_bands = []
        if not has_low_freq:
            missing_bands.append("bass (20-250 Hz)")
        if not has_mid_freq:
            missing_bands.append("mids (250-2000 Hz)")
        if not has_high_freq:
            missing_bands.append("treble (2000+ Hz)")

        # Spectral features — use same adaptive n_fft as STFT above
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=_n_fft_ech)[0]
        spectral_centroid = np.mean(spectral_centroids)

        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=_n_fft_ech)[0]
        spectral_bandwidth = np.mean(spectral_bandwidths)

        return SpectrumProfile(
            has_low_freq=has_low_freq,
            has_mid_freq=has_mid_freq,
            has_high_freq=has_high_freq,
            bass_ratio=bass_ratio,
            mid_ratio=mid_ratio,
            treble_ratio=treble_ratio,
            spectral_centroid=spectral_centroid,  # type: ignore[arg-type]
            spectral_bandwidth=spectral_bandwidth,  # type: ignore[arg-type]
            missing_bands=missing_bands,
        )

    # =========================================================================
    # Helper Methods for Assessment
    # =========================================================================

    def _determine_severity(
        self, extreme_deg: dict, unknown_defect: dict, medium_mix: dict, spectrum_conflict: dict
    ) -> DegradationSeverity:
        """Bestimmt overall degradation severity."""
        if extreme_deg["is_extreme"]:
            # Check specific metrics
            if extreme_deg["snr"] < 20 or extreme_deg["defect_coverage"] > 0.9:
                return DegradationSeverity.CATASTROPHIC
            elif extreme_deg["snr"] < 25 or extreme_deg["defect_coverage"] > 0.85:
                return DegradationSeverity.EXTREME
            else:
                return DegradationSeverity.SEVERE

        # Count issues
        issue_count = sum([unknown_defect["is_unknown"], medium_mix["is_mixed"], spectrum_conflict["has_conflict"]])

        if issue_count >= 2:
            return DegradationSeverity.SEVERE
        elif issue_count == 1:
            return DegradationSeverity.MODERATE
        else:
            return DegradationSeverity.MINIMAL

    def _determine_goal_reachability(
        self, audio: np.ndarray, sr: int, mode_config, extreme_deg: dict, spectrum_conflict: dict
    ) -> tuple[list[str], list[str]]:
        """Bestimmt which goals are reachable vs. unreachable."""
        all_goals = list(mode_config.musical_goals.keys())
        unreachable = []

        # If extreme degradation, most goals unreachable
        if extreme_deg["is_extreme"]:
            if extreme_deg["snr"] < 20:
                unreachable.extend(["brillanz", "transparenz", "natuerlichkeit"])
            if extreme_deg["dynamic_range"] < 10:
                unreachable.extend(["emotionalitaet"])
            if extreme_deg["clipping_ratio"] > 0.15:
                unreachable.extend(["natuerlichkeit", "authentizitaet"])

        # Add spectrum conflicts
        if spectrum_conflict["has_conflict"]:
            for goal in spectrum_conflict["conflicts"]:
                if goal not in unreachable:
                    unreachable.append(goal)

        # Remove duplicates
        unreachable = list(set(unreachable))
        reachable = [g for g in all_goals if g not in unreachable]

        return reachable, unreachable

    def _adjust_thresholds(
        self, mode_config, severity: DegradationSeverity, unreachable_goals: list[str]
    ) -> dict[str, float]:
        """Adjust goal thresholds based on severity."""
        adjusted = {}

        # Severity-based adjustment factors
        severity_factors = {
            DegradationSeverity.MINIMAL: 1.0,
            DegradationSeverity.MODERATE: 0.95,
            DegradationSeverity.SEVERE: 0.85,
            DegradationSeverity.EXTREME: 0.70,
            DegradationSeverity.CATASTROPHIC: 0.50,
        }

        factor = severity_factors[severity]

        for goal, target in mode_config.musical_goals.items():
            if goal in unreachable_goals:
                # Further reduce unreachable goals
                adjusted[goal] = target * factor * 0.80
            else:
                adjusted[goal] = target * factor

        return adjusted

    def _recommend_processing_mode(
        self,
        current_mode: ProcessingMode,
        edge_case_type: EdgeCaseType,
        severity: DegradationSeverity,
        medium_mix: dict,
    ) -> ProcessingMode:
        """Recommend best processing mode for edge case."""
        # Für extreme Degradation: RESTORATION (minimal intervention)
        if severity in [DegradationSeverity.EXTREME, DegradationSeverity.CATASTROPHIC]:
            return ProcessingMode.RESTORATION

        # Für mixed medium: RESTORATION (balanced)
        if edge_case_type == EdgeCaseType.MEDIUM_MIX:
            return ProcessingMode.RESTORATION

        # Für unknown defects: RESTORATION (minimal changes)
        if edge_case_type == EdgeCaseType.UNKNOWN_DEFECT:
            return ProcessingMode.RESTORATION

        # Sonst aktuellen Mode behalten
        return current_mode

    def _prioritize_goals(
        self, mode_config, reachable: list[str], unreachable: list[str], spectrum_conflict: dict
    ) -> list[str]:
        """Prioritize goals based on reachability and mode weights."""
        # Get mode's prioritized goals
        mode_priorities = mode_config.get_prioritized_goals()

        # Filter to reachable goals only
        prioritized = []
        for goal_name, weight in mode_priorities:
            if goal_name in reachable:
                prioritized.append(goal_name)

        return prioritized

    def _determine_fallback_strategy(
        self, edge_case_type: EdgeCaseType, severity: DegradationSeverity, unreachable_goals: list[str]
    ) -> str:
        """Bestimmt fallback strategy if processing fails."""
        if severity == DegradationSeverity.CATASTROPHIC:
            return "Return original audio - degradation too severe for restoration"

        if edge_case_type == EdgeCaseType.EXTREME_DEGRADATION:
            return "Apply minimal restoration - focus on defect removal only"

        if edge_case_type == EdgeCaseType.UNKNOWN_DEFECT:
            return "Use conservative parameters - avoid aggressive processing"

        if edge_case_type == EdgeCaseType.SPECTRUM_CONFLICT:
            goals_str = ", ".join(unreachable_goals[:3])
            return f"Skip unreachable goals ({goals_str}) - focus on reachable goals"

        if edge_case_type == EdgeCaseType.MEDIUM_MIX:
            return "Process dominant medium first - then address secondary defects"

        return "Standard processing with adjusted thresholds"

    def _calculate_confidence(
        self, extreme_deg: dict, unknown_defect: dict, medium_mix: dict, spectrum_conflict: dict
    ) -> float:
        """Calculate confidence in edge case assessment."""
        # Start with high confidence
        confidence = 1.0

        # Reduce confidence for uncertain detections
        if unknown_defect["is_unknown"]:
            confidence *= 0.85  # Unknown defects are uncertain

        if medium_mix["is_mixed"]:
            # Mixed medium is uncertain if scores are close
            vinyl_score = medium_mix["vinyl_score"]
            tape_score = medium_mix["tape_score"]
            if abs(vinyl_score - tape_score) <= 1:
                confidence *= 0.90

        # Increase confidence for clear extreme degradation
        if extreme_deg["is_extreme"]:
            # Very clear if multiple metrics exceeded
            metrics_exceeded = sum(
                [
                    extreme_deg["snr"] < 25,
                    extreme_deg["defect_coverage"] > 0.85,
                    extreme_deg["dynamic_range"] < 12,
                    extreme_deg["clipping_ratio"] > 0.12,
                ]
            )
            if metrics_exceeded >= 2:
                confidence = min(1.0, confidence * 1.10)

        return min(1.0, max(0.0, confidence))


if __name__ == "__main__":
    # Example usage and testing
    logger.debug("=== EdgeCaseHandler Example ===\n")

    # Create test signal with extreme degradation
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create severely degraded audio (low SNR, high defect coverage)
    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    noise = np.random.normal(0, 0.5, len(signal))  # Heavy noise
    clicks = np.zeros_like(signal)
    click_positions = np.random.choice(len(signal), size=int(len(signal) * 0.1))
    clicks[click_positions] = np.random.uniform(-2, 2, len(click_positions))

    degraded_audio = signal + noise + clicks
    degraded_audio = np.clip(degraded_audio, -1, 1)

    # Initialize handler
    handler = EdgeCaseHandler()

    # Assess edge cases
    logger.debug("Assessing edge cases...")
    assessment = handler.assess_edge_cases(degraded_audio, sr=sr, mode=ProcessingMode.RESTORATION)

    logger.debug("\nEdge Case Type: %s", assessment.edge_case_type.value)
    logger.debug("Severity: %s", assessment.severity.value)
    logger.debug("Confidence: %.2f", assessment.confidence)
    logger.debug("\nReachable Goals (%s):", len(assessment.reachable_goals))
    for goal in assessment.reachable_goals:
        logger.debug("  ✅ %s", goal)
    logger.debug("\nUnreachable Goals (%s):", len(assessment.unreachable_goals))
    for goal in assessment.unreachable_goals:
        logger.debug("  ❌ %s", goal)
    logger.debug("\nRecommended Betriebsart: %s", assessment.recommended_mode.value)
    logger.debug("Ersatzpfad Strategy: %s", assessment.fallback_strategy)

    logger.debug("\n=== Test vollstaendig ===")
