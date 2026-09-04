"""
Intelligibility Scorer
======================

Automatic vocal/speech clarity assessment for broadcast-quality validation.

This module provides comprehensive intelligibility scoring based on:
- Formant clarity (F1/F2/F3 prominence and separation)
- Consonant-to-vowel energy ratio
- Spectral balance across frequency bands
- Temporal clarity (attack/decay characteristics)

Use Cases:
- Podcast/Broadcast quality validation
- Vocal restoration quality assessment
- Before/after processing comparison
- Reference-based similarity scoring

Author: Aurik Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.signal as signal

if TYPE_CHECKING:
    from backend.ml.phoneme_aware.phoneme_detector import PhonemeSegment as _PhonemeSegmentT
else:
    _PhonemeSegmentT = Any

# Optional: phoneme detection integration
try:
    from backend.ml.phoneme_aware.phoneme_classifier import PhonemeCategory, PhonemeClassifier

    PHONEME_DETECTION_AVAILABLE = True
except ImportError:
    PHONEME_DETECTION_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


class QualityLevel(Enum):
    """Intelligibility quality levels."""

    EXCELLENT = "excellent"  # 0.85-1.0
    GOOD = "good"  # 0.70-0.85
    ACCEPTABLE = "acceptable"  # 0.55-0.70
    POOR = "poor"  # 0.40-0.55
    VERY_POOR = "very_poor"  # 0.0-0.40


@dataclass
class FormantData:
    """Formant frequency data."""

    f1: float  # First formant (Hz)
    f2: float  # Second formant (Hz)
    f3: float  # Third formant (Hz)
    f1_bandwidth: float  # F1 bandwidth (Hz)
    f2_bandwidth: float  # F2 bandwidth (Hz)
    f3_bandwidth: float  # F3 bandwidth (Hz)


@dataclass
class IntelligibilityReport:
    """
    Comprehensive intelligibility assessment report.

    Attributes:
        overall_score: Overall intelligibility (0.0-1.0)
        quality_level: Quality classification
        formant_clarity: Formant prominence score (0.0-1.0)
        consonant_clarity: Consonant clarity score (0.0-1.0)
        spectral_balance: Frequency balance score (0.0-1.0)
        temporal_clarity: Temporal characteristic score (0.0-1.0)
        cv_ratio: Consonant-to-vowel energy ratio
        formant_data: Detected formant frequencies
        recommendations: List of improvement suggestions
        metrics: Detailed sub-metrics
    """

    overall_score: float
    quality_level: QualityLevel
    formant_clarity: float
    consonant_clarity: float
    spectral_balance: float
    temporal_clarity: float
    cv_ratio: float
    formant_data: FormantData | None = None
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"IntelligibilityReport(overall={self.overall_score:.2f}, quality={self.quality_level.value})"


# ============================================================================
# INTELLIGIBILITY SCORER
# ============================================================================


class IntelligibilityScorer:
    """
    Assess vocal/speech intelligibility for quality validation.

    Provides multi-dimensional intelligibility scoring:
    - Formant clarity (F1/F2/F3 analysis)
    - Consonant-to-vowel ratio
    - Spectral balance (low/mid/high frequencies)
    - Temporal clarity (attack/decay characteristics)

    Example:
        >>> scorer = IntelligibilityScorer()
        >>> report = scorer.score(audio, sr=48000)
        >>> print(f"Quality: {report.quality_level.value}")
        >>> print(f"Overall score: {report.overall_score:.2f}")
    """

    def __init__(
        self,
        use_phoneme_detection: bool = True,
        min_formant_prominence: float = 0.3,
        optimal_cv_ratio: tuple[float, float] = (0.4, 0.6),
    ):
        """
        Initialisiert intelligibility scorer.

        Args:
            use_phoneme_detection: Use phoneme detection if available
            min_formant_prominence: Minimum formant prominence for clarity
            optimal_cv_ratio: Optimal consonant-to-vowel ratio range
        """
        self.use_phoneme_detection = use_phoneme_detection and PHONEME_DETECTION_AVAILABLE
        self.min_formant_prominence = min_formant_prominence
        self.optimal_cv_ratio = optimal_cv_ratio

        if self.use_phoneme_detection:
            self.phoneme_classifier = PhonemeClassifier()
            logger.debug("Intelligibility scorer initialisiert with phoneme detection")
        else:
            self.phoneme_classifier = None  # type: ignore[assignment]
            logger.debug("Intelligibility scorer initialisiert without phoneme detection")

    def score(
        self,
        audio: np.ndarray,
        sr: int,
        phonemes: list[_PhonemeSegmentT] | None = None,
        reference: np.ndarray | None = None,
    ) -> IntelligibilityReport:
        """
        Berechnet comprehensive intelligibility score.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            phonemes: Optional phoneme segments from PhonemeDetector
            reference: Optional reference audio for comparison

        Returns:
            IntelligibilityReport with scores and recommendations
        """
        # Ensure mono
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        logger.debug("Scoring %.2fs audio at %s Hz", len(audio_mono) / sr, sr)

        # 1. Formant analysis
        formant_data = self._extract_formants(audio_mono, sr)
        formant_clarity = self._assess_formant_clarity(formant_data)

        # 2. Consonant/vowel analysis
        if phonemes and self.phoneme_classifier:
            cv_ratio = self._compute_cv_ratio(audio_mono, sr, phonemes)
            consonant_clarity = self._assess_consonant_clarity(audio_mono, sr, phonemes)
        else:
            # Fallback: spectral estimation
            cv_ratio = self._estimate_cv_ratio(audio_mono, sr)
            consonant_clarity = self._estimate_consonant_clarity(audio_mono, sr)

        # 3. Spectral balance
        spectral_balance = self._assess_spectral_balance(audio_mono, sr)

        # 4. Temporal clarity
        temporal_clarity = self._assess_temporal_clarity(audio_mono, sr)

        # 5. Overall score (weighted average)
        overall = self._compute_overall_score(
            formant_clarity,
            consonant_clarity,
            spectral_balance,
            temporal_clarity,
        )

        # 6. Quality level
        quality_level = self._classify_quality(overall)

        # 7. Recommendations
        recommendations = self._generate_recommendations(
            formant_clarity,
            consonant_clarity,
            spectral_balance,
            temporal_clarity,
            cv_ratio,
        )

        # 8. Detailed metrics
        metrics = {
            "formant_f1": formant_data.f1 if formant_data else 0.0,
            "formant_f2": formant_data.f2 if formant_data else 0.0,
            "formant_f3": formant_data.f3 if formant_data else 0.0,
            "cv_ratio": cv_ratio,
            "formant_separation_f1_f2": formant_data.f2 - formant_data.f1 if formant_data else 0.0,
        }

        # 9. Reference comparison (optional)
        if reference is not None:
            similarity = self._compare_to_reference(audio_mono, reference, sr)
            metrics["reference_similarity"] = similarity

        return IntelligibilityReport(
            overall_score=overall,
            quality_level=quality_level,
            formant_clarity=formant_clarity,
            consonant_clarity=consonant_clarity,
            spectral_balance=spectral_balance,
            temporal_clarity=temporal_clarity,
            cv_ratio=cv_ratio,
            formant_data=formant_data,
            recommendations=recommendations,
            metrics=metrics,
        )

    def _extract_formants(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> FormantData | None:
        """
        Extrahiert formant frequencies using LPC analysis.

        Args:
            audio: Mono audio
            sr: Sample rate

        Returns:
            FormantData or None if extraction fails
        """
        try:
            # Limit audio length for performance (max 2 seconds for formant analysis)
            max_samples = int(2 * sr)
            if len(audio) > max_samples:
                # Use middle section for better quality
                start = (len(audio) - max_samples) // 2
                audio = audio[start : start + max_samples]

            # Pre-emphasis filter (boost high frequencies)
            pre_emphasized = signal.lfilter([1, -0.97], [1], audio)

            # Downsample for faster correlation (16kHz is sufficient for formants)
            target_sr = 16000
            if sr > target_sr:
                downsample_factor = sr // target_sr
                pre_emphasized = pre_emphasized[::downsample_factor]
                effective_sr = sr // downsample_factor
            else:
                effective_sr = sr

            # LPC order based on EFFECTIVE sample rate (after downsampling).
            # §2.8 Spec: Downsampling auf 16 kHz → LPC Ordnung 16 korrekt (Rabiner & Schafer 1978).
            # rule-of-thumb 2+sr_kHz würde bei 16 kHz → 18 liefern; auf Spec-konformes 16 begrenzt.
            lpc_order = min(int(effective_sr / 1000), 16)

            # Compute LPC coefficients
            from scipy.linalg import solve_toeplitz

            # Autocorrelation (use FFT-based method for speed)
            r = signal.correlate(pre_emphasized, pre_emphasized, mode="full", method="fft")
            r = r[len(r) // 2 :]
            r = r[: lpc_order + 1]

            # Solve Yule-Walker equations
            if len(r) < lpc_order + 1:
                return None

            R = r[:lpc_order]
            r_rest = r[1 : lpc_order + 1]

            try:
                a = solve_toeplitz(R, r_rest)
            except np.linalg.LinAlgError as e:
                logger.debug("§V6 LPC-Toeplitz-Lösung fehlgeschlagen — None zurückgegeben (LinAlgError): %s", e)
                return None

            # Guard: degenerate LPC → LAPACK DLASCL failure
            if not np.all(np.isfinite(a)):
                return None

            # Find roots of LPC polynomial
            try:
                roots = np.roots(np.r_[1, -a])
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.debug("§V6 LPC-Polynom-Roots fehlgeschlagen — None zurückgegeben: %s", e)
                return None

            # Convert complex roots to frequencies
            # IMPORTANT: use effective_sr (after downsampling), not original sr
            roots = roots[np.imag(roots) >= 0]  # Keep upper half plane
            angles = np.angle(roots)
            frequencies = angles * (effective_sr / (2 * np.pi))

            # Sort formants
            frequencies = np.sort(frequencies)
            frequencies = frequencies[frequencies > 50]  # Filter out DC
            frequencies = frequencies[frequencies < effective_sr / 2]  # Filter out Nyquist

            # Extract F1, F2, F3
            if len(frequencies) >= 3:
                f1, f2, f3 = frequencies[:3]

                # Estimate bandwidths (rough approximation)
                bw1 = 50.0
                bw2 = 70.0
                bw3 = 110.0

                return FormantData(
                    f1=float(f1),
                    f2=float(f2),
                    f3=float(f3),
                    f1_bandwidth=bw1,
                    f2_bandwidth=bw2,
                    f3_bandwidth=bw3,
                )

            return None

        except Exception as e:
            logger.warning("Formant extraction fehlgeschlagen: %s", e)
            return None

    def _assess_formant_clarity(
        self,
        formant_data: FormantData | None,
    ) -> float:
        """
        Assess formant clarity based on prominence and separation.

        Args:
            formant_data: Extracted formant frequencies

        Returns:
            Clarity score (0.0-1.0)
        """
        if formant_data is None:
            return 0.5  # Unknown

        # Check formant separation (F1-F2 should be > 700 Hz ideally)
        f1_f2_sep = formant_data.f2 - formant_data.f1
        f2_f3_sep = formant_data.f3 - formant_data.f2

        # Ideal separations
        ideal_f1_f2_sep = 800.0  # Hz
        ideal_f2_f3_sep = 1000.0  # Hz

        # Compute separation scores
        sep_score_1 = np.clip(f1_f2_sep / ideal_f1_f2_sep, 0.0, 1.0)
        sep_score_2 = np.clip(f2_f3_sep / ideal_f2_f3_sep, 0.0, 1.0)

        # Check formant positions (typical ranges)
        # F1: 200-900 Hz, F2: 800-2500 Hz, F3: 1700-3500 Hz
        position_score = 0.0
        if 200 <= formant_data.f1 <= 900:
            position_score += 0.33
        if 800 <= formant_data.f2 <= 2500:
            position_score += 0.33
        if 1700 <= formant_data.f3 <= 3500:
            position_score += 0.34

        # Combined clarity score
        clarity = sep_score_1 * 0.3 + sep_score_2 * 0.3 + position_score * 0.4

        return float(np.clip(clarity, 0.0, 1.0))

    def _compute_cv_ratio(
        self,
        audio: np.ndarray,
        sr: int,
        phonemes: list[_PhonemeSegmentT],
    ) -> float:
        """
        Berechnet consonant-to-vowel energy ratio using phoneme detection.

        Args:
            audio: Mono audio
            sr: Sample rate
            phonemes: Detected phoneme segments

        Returns:
            C/V ratio
        """
        if not self.phoneme_classifier:
            return 0.5

        consonant_energy = 0.0
        vowel_energy = 0.0

        for phoneme_seg in phonemes:
            # Get phoneme info
            phoneme_info = self.phoneme_classifier.classify(phoneme_seg.phoneme)

            # Extract audio segment
            start_idx = int(phoneme_seg.start_time * sr)
            end_idx = int(phoneme_seg.end_time * sr)
            segment = audio[start_idx:end_idx]

            # Compute energy
            energy = np.sum(segment**2)

            # Accumulate by type
            if phoneme_info.category in [  # type: ignore[attr-defined]
                PhonemeCategory.VOWEL_OPEN,
                PhonemeCategory.VOWEL_MID,
                PhonemeCategory.VOWEL_CLOSE,
            ]:
                vowel_energy += energy
            else:
                consonant_energy += energy

        # Compute ratio
        total_energy = consonant_energy + vowel_energy
        if total_energy == 0:
            return 0.5

        cv_ratio = consonant_energy / total_energy

        return float(cv_ratio)

    def _assess_consonant_clarity(
        self,
        audio: np.ndarray,
        sr: int,
        phonemes: list[_PhonemeSegmentT],
    ) -> float:
        """
        Assess consonant clarity using phoneme detection.

        Args:
            audio: Mono audio
            sr: Sample rate
            phonemes: Detected phoneme segments

        Returns:
            Clarity score (0.0-1.0)
        """
        if not self.phoneme_classifier:
            return 0.5

        # Analyze high-frequency content during consonants
        consonant_hf_energy = 0.0
        consonant_count = 0

        for phoneme_seg in phonemes:
            phoneme_info = self.phoneme_classifier.classify(phoneme_seg.phoneme)

            # Only process consonants
            if phoneme_info.category in [  # type: ignore[attr-defined]
                PhonemeCategory.VOWEL_OPEN,
                PhonemeCategory.VOWEL_MID,
                PhonemeCategory.VOWEL_CLOSE,
            ]:
                continue

            # Extract segment
            start_idx = int(phoneme_seg.start_time * sr)
            end_idx = int(phoneme_seg.end_time * sr)
            segment = audio[start_idx:end_idx]

            if len(segment) < 64:
                continue

            # Compute spectrum
            spectrum = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1 / sr)

            # High-frequency band (2-8 kHz) for consonants
            hf_band = (freqs >= 2000) & (freqs <= 8000)
            hf_energy = np.mean(spectrum[hf_band]) if np.any(hf_band) else 0.0

            consonant_hf_energy += hf_energy
            consonant_count += 1

        if consonant_count == 0:
            return 0.5

        # Average HF energy
        avg_hf_energy = consonant_hf_energy / consonant_count

        # Normalize to 0-1 range (heuristic)
        clarity = np.clip(avg_hf_energy / 1000.0, 0.0, 1.0)

        return float(clarity)

    def _estimate_cv_ratio(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Schätzt consonant-to-vowel ratio without phoneme detection.

        Uses spectral heuristics:
        - Vowels: strong low-mid frequencies (100-2000 Hz)
        - Consonants: strong high frequencies (2000-8000 Hz)

        Args:
            audio: Mono audio
            sr: Sample rate

        Returns:
            Estimated C/V ratio
        """
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Vowel band (100-2000 Hz)
        vowel_band = (freqs >= 100) & (freqs <= 2000)
        vowel_energy = np.sum(spectrum[vowel_band]) if np.any(vowel_band) else 0.0

        # Consonant band (2000-8000 Hz)
        consonant_band = (freqs >= 2000) & (freqs <= 8000)
        consonant_energy = np.sum(spectrum[consonant_band]) if np.any(consonant_band) else 0.0

        # Compute ratio
        total_energy = consonant_energy + vowel_energy
        if total_energy == 0:
            return 0.5

        cv_ratio = consonant_energy / total_energy

        return float(cv_ratio)

    def _estimate_consonant_clarity(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Schätzt consonant clarity without phoneme detection.

        Uses high-frequency content analysis.

        Args:
            audio: Mono audio
            sr: Sample rate

        Returns:
            Estimated clarity score (0.0-1.0)
        """
        # Compute power spectrum
        spectrum = np.abs(np.fft.rfft(audio)) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # High-frequency band (2-8 kHz) — carries consonant / clarity information
        hf_band = (freqs >= 2000) & (freqs <= 8000)

        total_energy = np.sum(spectrum) + 1e-30
        hf_energy = np.sum(spectrum[hf_band]) if np.any(hf_band) else 0.0

        # Relative HF ratio: score 1.0 when ≥30 % of energy is in HF band
        # (pure speech consonants typically reach 20–40 %)
        hf_ratio = hf_energy / total_energy
        clarity = np.clip(hf_ratio / 0.30, 0.0, 1.0)

        return float(clarity)

    def _assess_spectral_balance(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Assess spectral balance across frequency bands.

        Ideal balance:
        - Low (100-500 Hz): warmth, ~20%
        - Mid (500-2000 Hz): presence, ~40%
        - High (2000-8000 Hz): intelligibility, ~30%

        Args:
            audio: Mono audio
            sr: Sample rate

        Returns:
            Balance score (0.0-1.0)
        """
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Define bands
        low_band = (freqs >= 100) & (freqs <= 500)
        mid_band = (freqs >= 500) & (freqs <= 2000)
        high_band = (freqs >= 2000) & (freqs <= 8000)

        # Compute energies
        low_energy = np.sum(spectrum[low_band]) if np.any(low_band) else 0.0
        mid_energy = np.sum(spectrum[mid_band]) if np.any(mid_band) else 0.0
        high_energy = np.sum(spectrum[high_band]) if np.any(high_band) else 0.0

        total_energy = low_energy + mid_energy + high_energy
        if total_energy == 0:
            return 0.5

        # Compute ratios
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

        # Ideal ratios
        ideal_low = 0.20
        ideal_mid = 0.40
        ideal_high = 0.30

        # Compute balance score (penalty for deviation)
        low_score = 1.0 - abs(low_ratio - ideal_low) / ideal_low
        mid_score = 1.0 - abs(mid_ratio - ideal_mid) / ideal_mid
        high_score = 1.0 - abs(high_ratio - ideal_high) / ideal_high

        # Weighted average (mid most important)
        balance = low_score * 0.25 + mid_score * 0.5 + high_score * 0.25

        return float(np.clip(balance, 0.0, 1.0))

    def _assess_temporal_clarity(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Assess temporal clarity (attack/decay characteristics).

        Good temporal clarity:
        - Clear transients
        - Distinct attack phases
        - Proper decay envelopes

        Args:
            audio: Mono audio
            sr: Sample rate

        Returns:
            Clarity score (0.0-1.0)
        """
        # Compute envelope
        analytic_signal = signal.hilbert(audio)
        analytic_arr = np.asarray(analytic_signal, dtype=np.complex128)
        amplitude_envelope = np.abs(analytic_arr)

        # Smooth envelope
        window_size = int(0.01 * sr)  # 10ms window
        if window_size % 2 == 0:
            window_size += 1
        # Guard: kernel_size must not exceed signal length
        window_size = min(window_size, len(amplitude_envelope))
        if window_size % 2 == 0:
            window_size = max(1, window_size - 1)
        envelope_smooth = signal.medfilt(amplitude_envelope, window_size)

        # Detect transients using first derivative
        envelope_diff = np.diff(envelope_smooth)

        # Count strong attacks (positive peaks)
        threshold = np.std(envelope_diff) * 2.0
        attacks = np.sum(envelope_diff > threshold)

        # Normalize by duration
        duration_sec = len(audio) / sr
        attacks_per_sec = attacks / duration_sec if duration_sec > 0 else 0

        # Ideal: 2-6 attacks per second for speech/singing
        if 2 <= attacks_per_sec <= 6:
            temporal_clarity = 1.0
        elif attacks_per_sec < 2:
            # Too smooth (possibly over-processed)
            temporal_clarity = attacks_per_sec / 2.0
        else:
            # Too many attacks (possibly noisy)
            temporal_clarity = np.clip(6.0 / attacks_per_sec, 0.0, 1.0)

        return float(temporal_clarity)

    def _compute_overall_score(
        self,
        formant_clarity: float,
        consonant_clarity: float,
        spectral_balance: float,
        temporal_clarity: float,
    ) -> float:
        """
        Berechnet weighted overall intelligibility score.

        Args:
            formant_clarity: Formant clarity score
            consonant_clarity: Consonant clarity score
            spectral_balance: Spectral balance score
            temporal_clarity: Temporal clarity score

        Returns:
            Overall score (0.0-1.0)
        """
        # Weights (formant and consonant most important)
        weights = {
            "formant": 0.35,
            "consonant": 0.35,
            "spectral": 0.20,
            "temporal": 0.10,
        }

        overall = (
            formant_clarity * weights["formant"]
            + consonant_clarity * weights["consonant"]
            + spectral_balance * weights["spectral"]
            + temporal_clarity * weights["temporal"]
        )

        return float(np.clip(overall, 0.0, 1.0))

    def _classify_quality(
        self,
        overall_score: float,
    ) -> QualityLevel:
        """
        Classify quality level based on overall score.

        Args:
            overall_score: Overall intelligibility score

        Returns:
            QualityLevel classification
        """
        if overall_score >= 0.85:
            return QualityLevel.EXCELLENT
        elif overall_score >= 0.70:
            return QualityLevel.GOOD
        elif overall_score >= 0.55:
            return QualityLevel.ACCEPTABLE
        elif overall_score >= 0.40:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR

    def _generate_recommendations(
        self,
        formant_clarity: float,
        consonant_clarity: float,
        spectral_balance: float,
        temporal_clarity: float,
        cv_ratio: float,
    ) -> list[str]:
        """
        Generiert actionable recommendations for improvement.

        Args:
            formant_clarity: Formant clarity score
            consonant_clarity: Consonant clarity score
            spectral_balance: Spectral balance score
            temporal_clarity: Temporal clarity score
            cv_ratio: Consonant-to-vowel ratio

        Returns:
            List of recommendations
        """
        recommendations = []

        # Formant clarity
        if formant_clarity < 0.6:
            recommendations.append(
                "Low formant clarity. Consider applying formant enhancement or EQ boost in vowel regions (200-2500 Hz)."
            )

        # Consonant clarity
        if consonant_clarity < 0.6:
            recommendations.append(
                "Low consonant clarity. Consider applying high-frequency "
                "enhancement (2-8 kHz) or reducing excessive de-essing."
            )

        # Spectral balance
        if spectral_balance < 0.6:
            recommendations.append(
                "Poor spectral balance. Apply adaptive EQ to balance low/mid/high frequency content."
            )

        # Temporal clarity
        if temporal_clarity < 0.6:
            recommendations.append(
                "Low temporal clarity. Reduce excessive compression or ensure transient preservation in processing."
            )

        # C/V ratio
        if cv_ratio < self.optimal_cv_ratio[0]:
            recommendations.append(
                f"Consonant-to-vowel ratio ({cv_ratio:.2f}) is low. "
                "Boost high-frequency content or reduce excessive warmth."
            )
        elif cv_ratio > self.optimal_cv_ratio[1]:
            recommendations.append(
                f"Consonant-to-vowel ratio ({cv_ratio:.2f}) is high. Reduce harshness or apply gentle de-essing."
            )

        if not recommendations:
            recommendations.append("Intelligibility is excellent. No improvements needed.")

        return recommendations

    def _compare_to_reference(
        self,
        audio: np.ndarray,
        reference: np.ndarray,
        sr: int,
    ) -> float:
        """
        Compare audio to reference for similarity scoring.

        Args:
            audio: Test audio
            reference: Reference audio
            sr: Sample rate

        Returns:
            Similarity score (0.0-1.0)
        """
        # Ensure same length
        min_len = min(len(audio), len(reference))
        audio = audio[:min_len]
        reference = reference[:min_len]

        # Compute spectral similarity
        audio_spectrum = np.abs(np.fft.rfft(audio))
        ref_spectrum = np.abs(np.fft.rfft(reference))

        # Normalize spectra
        audio_spectrum = audio_spectrum / (np.max(audio_spectrum) + 1e-10)
        ref_spectrum = ref_spectrum / (np.max(ref_spectrum) + 1e-10)

        # Compute cosine similarity
        similarity = np.dot(audio_spectrum, ref_spectrum) / (
            np.linalg.norm(audio_spectrum) * np.linalg.norm(ref_spectrum) + 1e-10
        )

        return float(np.clip(similarity, 0.0, 1.0))


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def assess_intelligibility(
    audio: np.ndarray,
    sr: int,
    phonemes: list[_PhonemeSegmentT] | None = None,
    reference: np.ndarray | None = None,
) -> IntelligibilityReport:
    """
    Convenience function for intelligibility assessment.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
        phonemes: Optional phoneme segments from PhonemeDetector
        reference: Optional reference audio for comparison

    Returns:
        IntelligibilityReport with comprehensive scores

    Example:
        >>> report = assess_intelligibility(audio, sr=48000)
        >>> print(f"Quality: {report.quality_level.value}")
        >>> print(f"Score: {report.overall_score:.2f}")
    """
    scorer = IntelligibilityScorer()
    return scorer.score(audio, sr, phonemes=phonemes, reference=reference)
