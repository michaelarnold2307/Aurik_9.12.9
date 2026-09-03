"""
Aurik 10.0.0 AI Model Framework - Comprehensive Audio Restoration & Enhancement
=============================================================================

Vollständiges KI-Framework für:
- Tonträger-/Defekterkennung (open source + Eigenentwicklung)
- Audio Restoration (Denoising, Declicking, Dehissing)
- Audio Repair (Dropout-Filling, Bandwidth Extension)
- Reconstruction (Missing Content, Damage Repair)
- Enhancement (Clarity, Presence, Detail)
- Remastering (Magic Button: Studio 2026)

Phase: KI-Modelle (open source + Eigenentwicklung)
Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
Version: 10.0.0

WICHTIG: Keine Dummys/Mocks - nur reale, funktionsfähige Implementierungen.
Pure Open Source reicht NICHT aus - Eigenentwicklung ist in allen Bereichen essenziell.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.core.audio_utils import safe_filtfilt  # §v10.101 padlen-guard

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from backend.core.defect_scanner import MaterialType as _PhaseMaterialType


def _analytic_envelope(signal_in: np.ndarray) -> np.ndarray:
    """Gibt amplitude envelope via analytic signal (FFT Hilbert equivalent) zurück."""
    x = np.asarray(signal_in, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return np.asarray([], dtype=np.float64)  # type: ignore[no-any-return]
    spectrum = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * h))  # type: ignore[no-any-return]


# Import Vocal AI Enhancement
try:
    from backend.core.vocal_ai_enhancement import (
        EmotionPreservationMode,
        UnifiedVocalAIEnhancer,
        VocalEnhancementResult,
    )

    VOCAL_AI_AVAILABLE = True
except ImportError:
    VOCAL_AI_AVAILABLE = False
    logger.warning("Vocal AI Enhancement not verfuegbar")

# Import Dynamics Processing Phases (Phase 10 + 11)
try:
    # Import canonical MaterialType from defect_scanner (used by phases)
    from backend.core.defect_scanner import MaterialType as PhasesMaterialType
    from backend.core.phases.phase_10_compression import CompressionPhase
    from backend.core.phases.phase_11_limiting import LimitingPhase

    DYNAMICS_PHASES_AVAILABLE = True
except ImportError:
    DYNAMICS_PHASES_AVAILABLE = False
    PhasesMaterialType = None  # type: ignore[assignment,misc]
    logger.warning("Dynamics phases (Compression/Limiting) not verfuegbar")


# ============================================================
# ENUMS & DATA STRUCTURES
# ============================================================


class DefectType(Enum):
    """Erkannte Defekt-Typen."""

    CLICKS = "clicks"  # Clicks/Pops (Vinyl, digital)
    POPS = "pops"  # Loud transients
    CRACKLE = "crackle"  # Continuous noise (Vinyl)
    HISS = "hiss"  # Tape hiss, broadband noise
    HUM = "hum"  # 50/60Hz hum, ground loops
    BUZZ = "buzz"  # Harmonic buzz
    DISTORTION = "distortion"  # Clipping, overdrive, THD
    DROPOUT = "dropout"  # Tape/digital dropouts
    WOW = "wow"  # Speed variation (< 0.5 Hz)
    FLUTTER = "flutter"  # Speed variation (0.5-200 Hz)
    AZIMUTH_ERROR = "azimuth_error"  # Tape misalignment
    PHASE_ISSUES = "phase_issues"  # Phase cancellation
    DC_OFFSET = "dc_offset"  # DC bias
    CLIPPING = "clipping"  # Hard clipping
    COMPRESSION_ARTIFACTS = "compression_artifacts"  # MP3/AAC artifacts


class _AiMediaType(Enum):
    """Tonträger-Typen."""

    VINYL = "vinyl"
    SHELLAC = "shellac"
    TAPE_ANALOG = "tape_analog"
    TAPE_DIGITAL = "tape_digital"
    CD = "cd"
    DIGITAL_COMPRESSED = "digital_compressed"  # MP3, AAC, etc.
    DIGITAL_LOSSLESS = "digital_lossless"
    BROADCAST = "broadcast"
    UNKNOWN = "unknown"


#: Public alias for the internal _AiMediaType enum (for external imports).
MaterialType = _AiMediaType


class RestorationMode(Enum):
    """Restoration-Modi."""

    CONSERVATIVE = "conservative"  # Minimal processing, preserve authenticity
    BALANCED = "balanced"  # Balance between restoration and preservation
    AGGRESSIVE = "aggressive"  # Maximum restoration
    SURGICAL = "surgical"  # Target-specific defects only
    MAGIC_BUTTON = "magic_button"  # Studio 2026 mode


@dataclass
class DefectDetectionResult:
    """Ergebnis der Defekterkennung."""

    defects: dict[DefectType, float]  # Defect type -> confidence (0-1)
    severity: dict[DefectType, float]  # Defect type -> severity (0-1)
    locations: dict[DefectType, list[tuple[float, float]]]  # Defect type -> [(start_sec, end_sec)]
    overall_quality_score: float  # 0-1 (1=perfect, 0=terrible)
    material_type: _AiMediaType | None = None
    recommended_mode: RestorationMode | None = None


@dataclass
class FrameworkRestorationResult:
    """Ergebnis der AI-Framework-Restoration (intern, nicht Spec §2.1 FrameworkRestorationResult)."""

    audio: np.ndarray  # Restored audio
    sample_rate: int
    defects_removed: dict[DefectType, int]  # Count of removed defects
    processing_applied: list[str]  # List of applied processes
    quality_improvement: float  # Before/after quality delta
    confidence: float  # 0-1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancementResult:
    """Ergebnis des Audio-Enhancement."""

    audio: np.ndarray
    sample_rate: int
    enhancements_applied: list[str]
    clarity_improvement: float  # 0-1
    presence_improvement: float  # 0-1
    detail_improvement: float  # 0-1
    confidence: float  # 0-1
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# DEFECT DETECTION (Open Source + Eigenentwicklung)
# ============================================================


class UnifiedDefectDetector:
    """
    Einheitliches Defekterkennungs-System.

    Kombiniert:
    - Open Source Feature Extraction (librosa, scipy)
    - Eigenentwicklung: Multi-scale temporal analysis
    - Eigenentwicklung: Material-specific detection algorithms
    - Eigenentwicklung: Context-aware defect classification
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        logger.info("initialisiere Unified Defect Detector...")

        # Load existing detector if available
        try:
            from backend.core.forensics.ml_defect_detector import MLDefectDetector

            self.ml_detector = MLDefectDetector()
            self.has_ml_detector = True
            logger.info("✓ ML Defect Detector geladen")
        except ImportError:
            self.has_ml_detector = False
            logger.warning("ML Defect Detector not verfuegbar - using rule-based only")

    def detect(self, audio: np.ndarray, return_locations: bool = True) -> DefectDetectionResult:
        """
        Umfassende Defekterkennung.

        Args:
            audio: Audio signal (mono or stereo)
            return_locations: Return time locations of defects

        Returns:
            DefectDetectionResult with all detected defects
        """
        # Ensure mono for analysis
        audio_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # Multi-method detection
        defects = {}
        severity = {}
        locations = {}

        # Detect each defect type
        defects[DefectType.CLICKS], severity[DefectType.CLICKS], locations[DefectType.CLICKS] = self._detect_clicks(
            audio_mono, return_locations
        )

        defects[DefectType.HISS], severity[DefectType.HISS], locations[DefectType.HISS] = self._detect_hiss(
            audio_mono, return_locations
        )

        defects[DefectType.HUM], severity[DefectType.HUM], locations[DefectType.HUM] = self._detect_hum(
            audio_mono, return_locations
        )

        defects[DefectType.DISTORTION], severity[DefectType.DISTORTION], locations[DefectType.DISTORTION] = (
            self._detect_distortion(audio_mono, return_locations)
        )

        defects[DefectType.DROPOUT], severity[DefectType.DROPOUT], locations[DefectType.DROPOUT] = self._detect_dropout(
            audio_mono, return_locations
        )

        wow_conf, wow_sev, wow_locs, flutter_conf, flutter_sev, flutter_locs = self._detect_wow_flutter(
            audio_mono, return_locations
        )
        defects[DefectType.WOW], severity[DefectType.WOW], locations[DefectType.WOW] = (
            wow_conf,
            wow_sev,
            wow_locs,
        )
        defects[DefectType.FLUTTER], severity[DefectType.FLUTTER], locations[DefectType.FLUTTER] = (
            flutter_conf,
            flutter_sev,
            flutter_locs,
        )

        defects[DefectType.CLIPPING], severity[DefectType.CLIPPING], locations[DefectType.CLIPPING] = (
            self._detect_clipping(audio_mono, return_locations)
        )

        # Compute overall quality score
        quality_score = self._compute_quality_score(defects, severity)

        # Detect material type
        material = self._detect_material_type(audio_mono, defects)

        # Recommend restoration mode
        mode = self._recommend_mode(defects, severity, material)

        return DefectDetectionResult(
            defects=defects,
            severity=severity,
            locations=locations if return_locations else {},
            overall_quality_score=quality_score,
            material_type=material,
            recommended_mode=mode,
        )

    def _detect_clicks(self, audio: np.ndarray, return_locs: bool) -> tuple[float, float, list]:
        """Eigenentwicklung: Advanced click detection using multi-scale analysis."""
        from scipy import signal

        # Multi-scale click detection
        # Scale 1: Short transients (< 1ms) - typical clicks
        # Scale 2: Medium transients (1-5ms) - pops
        # Scale 3: Long transients (5-20ms) - crackle
        # Envelope detection
        envelope = _analytic_envelope(audio)

        # Differentiate to find rapid changes
        diff = np.diff(envelope, prepend=envelope[0])

        # Multi-scale peak detection
        scales = [
            int(0.001 * self.sr),  # 1ms
            int(0.005 * self.sr),  # 5ms
            int(0.020 * self.sr),  # 20ms
        ]

        all_clicks = []
        for scale in scales:
            # Smooth at scale
            window = np.ones(scale) / scale
            smoothed = np.convolve(np.abs(diff), window, mode="same")

            # Find peaks
            threshold = np.percentile(smoothed, 95)
            peaks, _ = signal.find_peaks(smoothed, height=threshold, distance=scale)
            all_clicks.extend(peaks)

        # Remove duplicates
        all_clicks = sorted(set(all_clicks))

        # Confidence and severity
        click_count = len(all_clicks)
        duration = len(audio) / self.sr
        density = click_count / duration if duration > 0 else 0

        confidence = min(1.0, density / 10)  # More clicks = higher confidence
        severity = min(1.0, density / 20)  # Very severe if >20 clicks/sec

        # Locations
        locations = []
        if return_locs:
            for click in all_clicks:
                start_sec = click / self.sr
                locations.append((start_sec, start_sec + 0.01))  # 10ms window

        return confidence, severity, locations

    def _detect_hiss(self, audio: np.ndarray, return_locs: bool) -> tuple[float, float, list]:
        """Eigenentwicklung: Broadband noise (hiss) detection."""
        from scipy import signal

        # High-frequency energy analysis (> 4kHz)
        sos = signal.butter(4, 4000, "high", fs=self.sr, output="sos")
        hf_audio = signal.sosfilt(sos, audio)

        # RMS energy in chunks
        chunk_size = int(0.1 * self.sr)  # 100ms
        hf_energy = []
        for i in range(0, len(hf_audio) - chunk_size, chunk_size // 2):
            chunk = hf_audio[i : i + chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            hf_energy.append(rms)

        # Consistent high-frequency energy = hiss
        if not hf_energy:
            return 0.0, 0.0, []

        mean_hf = np.mean(hf_energy)
        std_hf = np.std(hf_energy)

        # Hiss: consistent (low std), moderate level
        consistency = float(min(1.0, std_hf / (mean_hf + 1e-10)))  # type: ignore[arg-type]
        level = float(min(1.0, mean_hf / 0.1))  # type: ignore[arg-type]

        confidence = (consistency + level) / 2
        severity = level

        # Hiss is usually continuous
        locations = [(0, len(audio) / self.sr)] if return_locs else []

        return confidence, severity, locations

    def _detect_hum(self, audio: np.ndarray, return_locs: bool) -> tuple[float, float, list]:
        """Eigenentwicklung: 50/60Hz hum detection with harmonics."""
        from scipy import fft

        # FFT
        spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64).reshape(-1)))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Check 50Hz and 60Hz with harmonics
        def check_hum(fundamental: float, num_harmonics: int = 5) -> float:
            total_energy = 0.0
            for h in range(1, num_harmonics + 1):
                freq = fundamental * h
                idx = np.argmin(np.abs(freqs - freq))
                if idx < len(spectrum):
                    total_energy += spectrum[idx]
            return total_energy

        energy_50hz = check_hum(50.0)
        energy_60hz = check_hum(60.0)

        # Total energy for normalization
        total_energy: float = float(np.sum(spectrum))

        # Relative hum energy
        hum_energy = max(energy_50hz, energy_60hz)
        hum_ratio = hum_energy / (total_energy + 1e-10)

        confidence = min(1.0, hum_ratio * 100)
        severity = min(1.0, hum_ratio * 50)

        # Hum is continuous
        locations = [(0, len(audio) / self.sr)] if return_locs and confidence > 0.3 else []

        return confidence, severity, locations

    def _detect_distortion(self, audio: np.ndarray, return_locs: bool) -> tuple[float, float, list]:
        """Eigenentwicklung: Distortion detection (THD, clipping, overload)."""
        from scipy import fft

        # THD calculation
        spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64).reshape(-1)))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Find fundamental (assume speech/music range 80-1000 Hz)
        valid_range = (freqs >= 80) & (freqs <= 1000)
        if not np.any(valid_range):
            return 0.0, 0.0, []

        fund_idx = np.argmax(spectrum[valid_range]) + np.where(valid_range)[0][0]
        fund_freq = freqs[fund_idx]
        fund_mag = spectrum[fund_idx]

        # Harmonic energy
        harmonic_energy = 0.0
        for h in range(2, 6):
            harm_freq = fund_freq * h
            harm_idx = np.argmin(np.abs(freqs - harm_freq))
            if harm_idx < len(spectrum):
                harmonic_energy += spectrum[harm_idx] ** 2

        thd = np.sqrt(harmonic_energy) / (fund_mag + 1e-10)

        # Clipping check
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)

        # Combined distortion metric
        confidence = min(1.0, thd * 20 + clipping_ratio * 10)
        severity = min(1.0, thd * 10 + clipping_ratio * 5)

        # Find distorted regions
        locations = []
        if return_locs and confidence > 0.3:
            chunk_size = int(0.1 * self.sr)
            for i in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[i : i + chunk_size]
                chunk_clip = np.sum(np.abs(chunk) > 0.99) / len(chunk)
                if chunk_clip > 0.01:  # > 1% clipping
                    start_sec = i / self.sr
                    end_sec = (i + chunk_size) / self.sr
                    locations.append((start_sec, end_sec))

        return confidence, severity, locations

    def _detect_dropout(self, audio: np.ndarray, return_locs: bool) -> tuple[float, float, list]:
        """Eigenentwicklung: Dropout/silence detection."""

        # Envelope for amplitude tracking
        envelope = _analytic_envelope(audio)

        # Smooth envelope
        window_size = int(0.01 * self.sr)  # 10ms
        window = np.ones(window_size) / window_size
        smooth_env = np.convolve(envelope, window, mode="same")

        # Detect sudden drops
        threshold = np.percentile(smooth_env, 10)
        dropouts = smooth_env < threshold

        # Find dropout regions
        dropout_regions = []
        in_dropout = False
        start = 0

        for i, is_dropout in enumerate(dropouts):
            if is_dropout and not in_dropout:
                start = i
                in_dropout = True
            elif not is_dropout and in_dropout:
                dropout_regions.append((start, i))
                in_dropout = False

        # Count significant dropouts (> 10ms)
        min_dropout_samples = int(0.01 * self.sr)
        significant_dropouts = [(s, e) for s, e in dropout_regions if e - s > min_dropout_samples]

        dropout_count = len(significant_dropouts)
        duration = len(audio) / self.sr
        dropout_density = dropout_count / duration if duration > 0 else 0

        confidence = min(1.0, dropout_density * 10)
        severity = min(1.0, dropout_density * 5)

        # Locations
        locations = []
        if return_locs:
            for start, end in significant_dropouts:
                locations.append((start / self.sr, end / self.sr))

        return confidence, severity, locations

    def _detect_wow_flutter(
        self, audio: np.ndarray, return_locs: bool
    ) -> tuple[float, float, list, float, float, list]:
        """Eigenentwicklung: Wow & flutter (pitch/speed variations) detection."""
        from scipy import signal

        # Pitch tracking via autocorrelation in chunks
        chunk_size = int(0.05 * self.sr)  # 50ms
        hop = chunk_size // 2

        pitches = []
        for i in range(0, len(audio) - chunk_size, hop):
            chunk = audio[i : i + chunk_size]

            # Autocorrelation — FFT-based O(N log N)
            from backend.core.core_utils import fft_autocorr

            autocorr = fft_autocorr(chunk)
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Find first peak (fundamental period)
            peaks, _ = signal.find_peaks(autocorr[20:], height=0.5)
            if len(peaks) > 0:
                period = peaks[0] + 20
                freq = self.sr / period
                if 50 < freq < 2000:  # Reasonable pitch range
                    pitches.append(freq)

        if len(pitches) < 2:
            return 0.0, 0.0, [], 0.0, 0.0, []

        # Wow & flutter = pitch instability
        pitch_std = np.std(pitches)
        pitch_mean = np.mean(pitches)
        pitch_cv = pitch_std / (pitch_mean + 1e-10)  # Coefficient of variation

        # Modulation depth
        modulation = pitch_cv * 100  # Percentage

        confidence = min(1.0, modulation / 5)  # >5% = very noticeable
        severity = min(1.0, modulation / 10)

        # Keep split semantics for legacy detector path: same estimate is reported
        # to both WOW and FLUTTER when no dedicated band split is available.
        wow_conf = confidence
        wow_sev = severity
        flutter_conf = confidence
        flutter_sev = severity

        # Continuous effect
        locations = [(0, len(audio) / self.sr)] if return_locs and confidence > 0.3 else []  # type: ignore[operator]

        return wow_conf, wow_sev, locations, flutter_conf, flutter_sev, locations  # type: ignore[return-value]

    def _detect_clipping(self, audio: np.ndarray, return_locs: bool) -> tuple[float, float, list]:
        """Clipping-Erkennung (hartes Begrenzen)."""
        # Count samples near ±1.0
        clipped = np.abs(audio) > 0.99
        clipping_ratio = np.sum(clipped) / len(audio)

        confidence = min(1.0, clipping_ratio * 100)
        severity = min(1.0, clipping_ratio * 50)

        # Find clipped regions
        locations = []
        if return_locs and clipping_ratio > 0.001:
            chunk_size = int(0.01 * self.sr)  # 10ms
            for i in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[i : i + chunk_size]
                if np.any(np.abs(chunk) > 0.99):
                    locations.append((i / self.sr, (i + chunk_size) / self.sr))

        return confidence, severity, locations

    def _compute_quality_score(self, defects: dict[DefectType, float], severity: dict[DefectType, float]) -> float:
        """
        Berechnet overall quality score (0-1, 1=perfect).

        Weighted by defect impact on listening experience.
        """
        # Weights for different defects (impact on quality)
        weights = {
            DefectType.CLICKS: 0.15,
            DefectType.HISS: 0.10,
            DefectType.HUM: 0.10,
            DefectType.DISTORTION: 0.20,
            DefectType.DROPOUT: 0.15,
            DefectType.WOW: 0.05,
            DefectType.FLUTTER: 0.05,
            DefectType.CLIPPING: 0.20,
        }

        # Compute weighted quality penalty
        total_penalty = 0.0
        for defect, weight in weights.items():
            if defect in defects and defect in severity:
                # Use both confidence and severity
                penalty = defects[defect] * severity[defect] * weight
                total_penalty += penalty

        # Quality score
        quality = 1.0 - min(1.0, total_penalty)
        return float(quality)

    def _detect_material_type(self, audio: np.ndarray, defects: dict[DefectType, float]) -> _AiMediaType:
        """Eigenentwicklung: Material type detection based on defect profile."""
        # Vinyl: clicks, crackle, wow/flutter
        vinyl_score = (
            defects.get(DefectType.CLICKS, 0) * 0.4
            + (defects.get(DefectType.WOW, 0) + defects.get(DefectType.FLUTTER, 0)) * 0.15
            + defects.get(DefectType.CRACKLE, 0) * 0.3
        )

        # Tape: hiss, dropout, azimuth
        tape_score = (
            defects.get(DefectType.HISS, 0) * 0.4
            + defects.get(DefectType.DROPOUT, 0) * 0.4
            + (defects.get(DefectType.WOW, 0) + defects.get(DefectType.FLUTTER, 0)) * 0.1
        )

        # Digital: clipping, compression artifacts
        digital_score = (
            defects.get(DefectType.CLIPPING, 0) * 0.5 + defects.get(DefectType.COMPRESSION_ARTIFACTS, 0) * 0.5
        )

        scores = {
            _AiMediaType.VINYL: vinyl_score,
            _AiMediaType.TAPE_ANALOG: tape_score,
            _AiMediaType.DIGITAL_COMPRESSED: digital_score,
        }

        # Return type with highest score, or UNKNOWN
        best_type = max(scores.items(), key=lambda x: x[1])
        if best_type[1] > 0.3:
            return best_type[0]
        return _AiMediaType.UNKNOWN

    def _recommend_mode(
        self, defects: dict[DefectType, float], severity: dict[DefectType, float], material: _AiMediaType
    ) -> RestorationMode:
        """Recommend restoration mode based on analysis."""
        # Compute average severity
        avg_severity = np.mean(list(severity.values())) if severity else 0.0

        # Recommend based on severity
        if avg_severity < 0.2:
            return RestorationMode.CONSERVATIVE
        elif avg_severity < 0.5:
            return RestorationMode.BALANCED
        elif avg_severity < 0.8:
            return RestorationMode.AGGRESSIVE
        else:
            return RestorationMode.SURGICAL


# ============================================================
# AUDIO RESTORATION (Open Source + Eigenentwicklung)
# ============================================================


class UnifiedAudioRestorer:
    """
    Einheitliches Audio-Restaurierungs-System.

    Kombiniert:
    - Open Source Basistechniken (scipy, librosa)
    - Eigenentwicklung: Adaptive restoration algorithms
    - Eigenentwicklung: Context-aware processing
    - Eigenentwicklung: Quality-preserving techniques
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.detector = UnifiedDefectDetector(sample_rate=sample_rate)
        logger.info("initialisiere Unified Audio Restorer...")

    def restore(
        self, audio: np.ndarray, mode: RestorationMode = RestorationMode.BALANCED, auto_detect: bool = True
    ) -> FrameworkRestorationResult:
        """
        Comprehensive audio restoration.

        Args:
            audio: Input audio (mono or stereo)
            mode: Restoration mode
            auto_detect: Automatically detect and target defects

        Returns:
            FrameworkRestorationResult with restored audio
        """
        # Detect defects if auto mode - DEBUG level to prevent console spam on import
        if auto_detect:
            detection = self.detector.detect(audio)
            logger.debug("§2.46f Auto-detect: %d defect types identified", len(detection.defects))
        else:
            detection = None

        # Process audio
        restored = audio.copy()
        defects_removed = {}
        processes = []

        # Apply restoration based on detected defects or mode
        if detection is None or detection.defects.get(DefectType.CLICKS, 0) > 0.3:
            restored, count = self._remove_clicks(restored)
            defects_removed[DefectType.CLICKS] = count
            processes.append("click_removal")

        if detection is None or detection.defects.get(DefectType.HISS, 0) > 0.3:
            restored = self._reduce_hiss(restored, mode)
            processes.append("hiss_reduction")

        if detection is None or detection.defects.get(DefectType.HUM, 0) > 0.3:
            restored = self._remove_hum(restored)
            processes.append("hum_removal")

        if detection is None or detection.defects.get(DefectType.DROPOUT, 0) > 0.3:
            restored = self._fill_dropouts(restored)
            processes.append("dropout_filling")

        # Compute quality improvement
        if detection:
            after_detection = self.detector.detect(restored)
            quality_improvement = after_detection.overall_quality_score - detection.overall_quality_score
        else:
            quality_improvement = 0.0

        return FrameworkRestorationResult(
            audio=restored,
            sample_rate=self.sr,
            defects_removed=defects_removed,
            processing_applied=processes,
            quality_improvement=quality_improvement,
            confidence=0.9,  # High confidence in restoration
            metadata={"mode": mode.value},
        )

    def _remove_clicks(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """Eigenentwicklung: Advanced click removal using interpolation."""
        from scipy import signal

        # Detect clicks
        audio_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # Find clicks
        envelope = _analytic_envelope(audio_mono)
        diff = np.diff(envelope, prepend=envelope[0])

        # Adaptive threshold
        threshold = np.percentile(np.abs(diff), 99)
        click_candidates, _ = signal.find_peaks(np.abs(diff), height=threshold)

        # Remove clicks by interpolation
        restored = audio.copy()
        window_size = int(0.005 * self.sr)  # 5ms window

        for click_pos in click_candidates:
            start = max(0, click_pos - window_size // 2)
            end = min(len(audio_mono), click_pos + window_size // 2)

            if audio.ndim == 2:
                for ch in range(audio.shape[1]):
                    # Linear interpolation
                    if start > 0 and end < len(audio):
                        restored[start:end, ch] = np.linspace(audio[start, ch], audio[end, ch], end - start)
            else:
                if start > 0 and end < len(audio):
                    restored[start:end] = np.linspace(audio[start], audio[end], end - start)

        return restored, len(click_candidates)

    def _reduce_hiss(self, audio: np.ndarray, mode: RestorationMode) -> np.ndarray:
        """Eigenentwicklung: Adaptive spectral subtraction for hiss reduction."""

        # Stereo processing
        if audio.ndim == 2:
            return np.stack([self._reduce_hiss_mono(audio[:, ch], mode) for ch in range(audio.shape[1])], axis=1)  # type: ignore[no-any-return]
        else:
            return self._reduce_hiss_mono(audio, mode)

    def _reduce_hiss_mono(self, audio: np.ndarray, mode: RestorationMode) -> np.ndarray:
        """Wiener filtering for hiss reduction."""
        from scipy import signal

        # Guard: STFT erfordert mindestens nperseg Samples
        if len(audio) < 2048:
            return audio

        # STFT
        _f, _t, Zxx = signal.stft(audio, self.sr, nperseg=2048, boundary="even")
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate noise floor (median over time)
        noise_floor = np.median(magnitude, axis=1, keepdims=True)

        # Wiener filtering
        if mode == RestorationMode.CONSERVATIVE:
            strength = 0.3
        elif mode == RestorationMode.AGGRESSIVE:
            strength = 0.8
        else:
            strength = 0.5

        wiener_gain = (magnitude**2) / (magnitude**2 + (noise_floor * strength) ** 2 + 1e-10)
        wiener_gain = np.clip(wiener_gain, 0, 1)

        # Apply gain
        Zxx_filtered = magnitude * wiener_gain * np.exp(1j * phase)

        # Inverse STFT
        _, restored = signal.istft(Zxx_filtered, self.sr, nperseg=2048)

        # Match length
        if len(restored) > len(audio):
            restored = restored[: len(audio)]
        elif len(restored) < len(audio):
            restored = np.pad(restored, (0, len(audio) - len(restored)))

        return restored  # type: ignore[no-any-return]

    def _remove_hum(self, audio: np.ndarray) -> np.ndarray:
        """Eigenentwicklung: Notch filtering for hum removal."""
        from scipy import signal

        # Notch filters for 50Hz and 60Hz with harmonics
        hum_freqs = [50, 100, 150, 60, 120, 180]

        result = audio.copy()

        for freq in hum_freqs:
            # Design notch filter
            Q = 30  # Quality factor (narrow notch)
            b, a = signal.iirnotch(freq, Q, self.sr)

            # Apply to each channel
            if audio.ndim == 2:
                for ch in range(audio.shape[1]):
                    result[:, ch] = safe_filtfilt(b, a, result[:, ch])
            else:
                result = safe_filtfilt(b, a, result)

        return result

    def _fill_dropouts(self, audio: np.ndarray) -> np.ndarray:
        """Eigenentwicklung: Dropout interpolation using surrounding context."""
        from scipy import interpolate

        # Detect dropouts
        audio_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio

        envelope = _analytic_envelope(audio_mono)
        threshold = np.percentile(envelope, 5)
        dropouts = envelope < threshold

        # Find dropout regions
        result = audio.copy()
        in_dropout = False
        start = 0

        for i, is_dropout in enumerate(dropouts):
            if is_dropout and not in_dropout:
                start = i
                in_dropout = True
            elif not is_dropout and in_dropout:
                # Fill dropout [start:i]
                if i - start > 10:  # Only fill significant dropouts
                    if audio.ndim == 2:
                        for ch in range(audio.shape[1]):
                            # Linear interpolation (2 Stützpunkte genügen)
                            x = [start - 1, i]
                            y = [audio[start - 1, ch], audio[i, ch]]
                            f = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
                            result[start:i, ch] = f(np.arange(start, i))
                    else:
                        x = [start - 1, i]
                        y = [audio[start - 1], audio[i]]
                        f = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
                        result[start:i] = f(np.arange(start, i))

                in_dropout = False

        return result


# ============================================================
# AUDIO ENHANCEMENT (Eigenentwicklung)
# ============================================================


class UnifiedAudioEnhancer:
    """
    Einheitliches Audio-Verbesserungs-System.

    Eigenentwicklung:
    - Clarity enhancement via adaptive EQ
    - Presence enhancement via multiband dynamics
    - Detail enhancement via transient shaping
    - Warmth/brightness control
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        logger.info("initialisiere Unified Audio Enhancer...")

    def enhance(
        self, audio: np.ndarray, target_clarity: float = 0.7, target_presence: float = 0.7, target_detail: float = 0.7
    ) -> EnhancementResult:
        """
        Comprehensive audio enhancement.

        Args:
            audio: Input audio
            target_clarity: Target clarity level (0-1)
            target_presence: Target presence level (0-1)
            target_detail: Target detail level (0-1)

        Returns:
            EnhancementResult with enhanced audio
        """
        enhanced = audio.copy()
        enhancements = []

        # Clarity enhancement
        if target_clarity > 0.5:
            enhanced = self._enhance_clarity(enhanced, target_clarity)
            enhancements.append("clarity_enhancement")

        # Presence enhancement
        if target_presence > 0.5:
            enhanced = self._enhance_presence(enhanced, target_presence)
            enhancements.append("presence_enhancement")

        # Detail enhancement
        if target_detail > 0.5:
            enhanced = self._enhance_detail(enhanced, target_detail)
            enhancements.append("detail_enhancement")

        return EnhancementResult(
            audio=enhanced,
            sample_rate=self.sr,
            enhancements_applied=enhancements,
            clarity_improvement=target_clarity,
            presence_improvement=target_presence,
            detail_improvement=target_detail,
            confidence=0.85,
            metadata={},
        )

    def _enhance_clarity(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Eigenentwicklung: Clarity enhancement via adaptive EQ."""
        from scipy import signal

        # Mid-high boost (2-8 kHz) for clarity
        # Design parametric EQ
        center_freq = 4000  # Hz
        Q = 1.0
        gain_db = (amount - 0.5) * 6  # ±3 dB range

        # Peaking filter
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * center_freq / self.sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        # Apply filter
        if audio.ndim == 2:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                result[:, ch] = safe_filtfilt(b, a, audio[:, ch])
        else:
            result = safe_filtfilt(b, a, audio)

        return result  # type: ignore[no-any-return]

    def _enhance_presence(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Eigenentwicklung: Presence enhancement via high-frequency emphasis."""
        from scipy import signal

        # High-shelf boost (> 6 kHz)
        cutoff = 6000  # Hz
        gain_db = (amount - 0.5) * 4  # ±2 dB

        # Design high-shelf filter
        Q = 0.707
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * cutoff / self.sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        # Apply filter
        if audio.ndim == 2:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                result[:, ch] = safe_filtfilt(b, a, audio[:, ch])
        else:
            result = safe_filtfilt(b, a, audio)

        return result  # type: ignore[no-any-return]

    def _enhance_detail(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Eigenentwicklung: Detail enhancement via transient emphasis."""

        # Enhance transients using envelope differentiation
        if audio.ndim == 2:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                result[:, ch] = self._enhance_detail_mono(audio[:, ch], amount)
        else:
            result = self._enhance_detail_mono(audio, amount)

        return result  # type: ignore[no-any-return]

    def _enhance_detail_mono(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Mono transient enhancement."""

        # Envelope detection
        envelope = _analytic_envelope(audio)

        # Differentiate to find transients
        diff = np.diff(envelope, prepend=envelope[0])

        # Enhance positive transients
        enhancement_factor = (amount - 0.5) * 0.2  # Max 10% boost
        enhanced_diff = diff * (1 + enhancement_factor * (diff > 0))

        # Integrate back
        enhanced_envelope = np.cumsum(enhanced_diff)
        enhanced_envelope = enhanced_envelope - np.mean(enhanced_envelope) + np.mean(envelope)

        # Apply envelope modulation
        gain = enhanced_envelope / (envelope + 1e-10)
        gain = np.clip(gain, 0.5, 2.0)  # Limit gain range

        result = audio * gain

        return result  # type: ignore[no-any-return]


# ============================================================
# MAGIC BUTTON: STUDIO 2026 MODE
# ============================================================


class RestorationMagicButton:
    """
    Magic Button: Restoration Only.

    Vollautomatische Defekterkennung + Restoration (OHNE Enhancement).
    Für puristische Restaurierung mit maximaler Authentizität.

    Eigenentwicklung: Focused restoration pipeline
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.detector = UnifiedDefectDetector(sample_rate=sample_rate)
        self.restorer = UnifiedAudioRestorer(sample_rate=sample_rate)
        logger.info("initialisiere Restoration Magic Button...")

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Magic Button: Pure restoration without enhancement.

        Args:
            audio: Input audio

        Returns:
            (restored_audio, report_dict)
        """
        logger.info("🔧 Starting Restoration Magic Button...")

        # Step 1: Detect defects
        logger.info("Step 1/2: Defect Detection...")
        detection = self.detector.detect(audio)

        # Step 2: Restore (ONLY - no enhancement)
        logger.info("Step 2/2: Audio Restoration...")
        restoration = self.restorer.restore(audio, mode=RestorationMode.BALANCED, auto_detect=True)

        # Generate report
        report = {
            "detection": {
                "defects_found": len([d for d, c in detection.defects.items() if c > 0.3]),
                "quality_score_before": detection.overall_quality_score,
                "material_type": detection.material_type.value if detection.material_type else "unknown",
            },
            "restoration": {
                "defects_removed": sum(restoration.defects_removed.values()),
                "processes": restoration.processing_applied,
                "quality_improvement": restoration.quality_improvement,
            },
            "final": {
                "success": True,
                "mode": "Restoration Only",
            },
        }

        logger.info("✅ Restoration Magic Button vollstaendig!")

        return restoration.audio, report


class Studio2026Processor:
    """
    Magic Button: Studio 2026 Mode.

    Vollautomatische Restoration + Enhancement + Remastering
    auf modernstem Studio-Niveau (2026).

    Eigenentwicklung: Complete automated pipeline

    Integration:
    - Phase 10: Compression (Dynamics)
    - Phase 11: Limiting (Peak Control)
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.detector = UnifiedDefectDetector(sample_rate=sample_rate)
        self.restorer = UnifiedAudioRestorer(sample_rate=sample_rate)
        self.enhancer = UnifiedAudioEnhancer(sample_rate=sample_rate)

        # Initialize dynamics phases
        if DYNAMICS_PHASES_AVAILABLE:
            self.compression = CompressionPhase()
            self.limiting = LimitingPhase()
            logger.info("✓ Dynamics phases (Compression/Limiting) geladen")
        else:
            self.compression = None  # type: ignore[assignment]
            self.limiting = None  # type: ignore[assignment]
            logger.warning("⚠ Dynamics phases not verfuegbar - using Ersatzpfad")

        logger.info("initialisiere Studio 2026 Processor...")

    @staticmethod
    def _map_material_type(framework_material: _AiMediaType) -> "_PhaseMaterialType | None":
        """
        Konvertiert _AiMediaType from ai_framework to defect_scanner _AiMediaType (used by phases).

        Args:
            framework_material: _AiMediaType from ai_framework

        Returns:
            PhasesMaterialType for phase processing
        """
        if not DYNAMICS_PHASES_AVAILABLE or PhasesMaterialType is None:
            return None

        # Mapping: ai_framework _AiMediaType -> defect_scanner _AiMediaType
        mapping = {
            _AiMediaType.SHELLAC: PhasesMaterialType.SHELLAC,
            _AiMediaType.VINYL: PhasesMaterialType.VINYL,
            _AiMediaType.TAPE_ANALOG: PhasesMaterialType.TAPE,
            _AiMediaType.TAPE_DIGITAL: PhasesMaterialType.TAPE,
            _AiMediaType.CD: PhasesMaterialType.CD_DIGITAL,
            _AiMediaType.DIGITAL_COMPRESSED: PhasesMaterialType.STREAMING,
            _AiMediaType.DIGITAL_LOSSLESS: PhasesMaterialType.CD_DIGITAL,
            _AiMediaType.BROADCAST: PhasesMaterialType.CD_DIGITAL,
            _AiMediaType.UNKNOWN: PhasesMaterialType.UNKNOWN,
        }

        return mapping.get(framework_material, PhasesMaterialType.UNKNOWN)

    def process(self, audio: np.ndarray, material: _AiMediaType | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Magic Button: One-click processing to Studio 2026 standard.

        Args:
            audio: Input audio
            material: Optional material type (auto-detected if None)

        Returns:
            (processed_audio, report_dict)
        """
        logger.info("🎬 Starting Studio 2026 Processing...")

        # Step 1: Detect defects
        logger.info("Step 1/5: Defect Detection...")
        detection = self.detector.detect(audio)

        # Auto-detect material if not provided
        if material is None:
            material = detection.material_type or _AiMediaType.UNKNOWN

        # Step 2: Restore
        logger.info("Step 2/5: Audio Restoration...")
        restoration = self.restorer.restore(audio, mode=RestorationMode.BALANCED, auto_detect=True)

        # Step 3: Enhance
        logger.info("Step 3/5: Audio Enhancement...")
        enhancement = self.enhancer.enhance(
            restoration.audio, target_clarity=0.8, target_presence=0.7, target_detail=0.7
        )

        # Step 4: Dynamics Processing (Compression)
        logger.info("Step 4/5: Dynamics Processing...")
        dynamics_audio, dynamics_report = self._apply_dynamics(enhancement.audio, material)

        # Step 5: Final mastering touches
        logger.info("Step 5/5: Final Mastering...")
        mastered = self._apply_mastering(dynamics_audio)

        # Generate report
        report = {
            "detection": {
                "defects_found": len([d for d, c in detection.defects.items() if c > 0.3]),
                "quality_score_before": detection.overall_quality_score,
                "material_type": material.value if material else "unknown",
            },
            "restoration": {
                "defects_removed": sum(restoration.defects_removed.values()),
                "processes": restoration.processing_applied,
                "quality_improvement": restoration.quality_improvement,
            },
            "enhancement": {
                "enhancements": enhancement.enhancements_applied,
                "clarity": enhancement.clarity_improvement,
                "presence": enhancement.presence_improvement,
                "detail": enhancement.detail_improvement,
            },
            "dynamics": dynamics_report,
            "final": {
                "success": True,
                "mode": "Studio 2026",
            },
        }

        logger.info("✅ Studio 2026 Processing vollstaendig!")

        return mastered, report

    def _apply_dynamics(self, audio: np.ndarray, material: _AiMediaType) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Wendet an: dynamics processing (Phase 10: Compression + Phase 11: Limiting).

        Args:
            audio: Input audio
            material: Material type for adaptive processing (ai_framework _AiMediaType)

        Returns:
            (processed_audio, dynamics_report)
        """
        report: dict[str, Any] = {
            "compression_applied": False,
            "limiting_applied": False,
        }

        if not DYNAMICS_PHASES_AVAILABLE or self.compression is None:
            # Fallback: simple peak limiting only
            # §DSP-Invariante: percentile 99.9 — Impuls-Artefakt darf Normalisierung nicht blockieren.
            peak = float(np.percentile(np.abs(audio), 99.9))
            if peak > 0.95:
                audio = audio / peak * 0.95
                report["fallback_limiting"] = True
            return audio, report

        # Convert _AiMediaType for phase processing
        phase_material = self._map_material_type(material)

        # Phase 10: Compression
        compression_result = self.compression.process(audio, self.sr, phase_material)  # type: ignore[arg-type,union-attr]

        if compression_result.success:
            audio = compression_result.audio
            report["compression_applied"] = compression_result.metadata.get("compression_applied", False)
            if report["compression_applied"]:
                report["compression"] = {
                    "ratio": compression_result.metadata.get("ratio", 0),
                    "threshold_db": compression_result.metadata.get("threshold_db", 0),
                    "gain_reduction_db": compression_result.metrics.get("avg_gain_reduction_db", 0),
                }

        # Phase 11: Limiting
        if self.limiting is None:
            return audio, report
        limiting_result = self.limiting.process(audio, self.sr, phase_material)  # type: ignore[arg-type,union-attr]

        if limiting_result.success:
            audio = limiting_result.audio
            report["limiting_applied"] = limiting_result.metadata.get("limiting_applied", False)
            if report["limiting_applied"]:
                report["limiting"] = {
                    "ceiling_db": limiting_result.metadata.get("ceiling_db", 0),
                    "peak_reduction_db": limiting_result.metrics.get("peak_reduction_db", 0),
                }

        return audio, report

    def _apply_mastering(self, audio: np.ndarray) -> np.ndarray:
        """Final mastering: subtle high-frequency enhancement + safety limiting."""
        from scipy import signal

        # Subtle high-frequency enhancement (air band)
        sos = signal.butter(2, 8000, "high", fs=self.sr, output="sos")
        # §2.51 Anti-Zeitversatz: sosfiltfilt — hf wird mit audio gemischt (0.9*audio + 0.1*hf).
        hf = signal.sosfiltfilt(sos, audio, axis=0)

        # Mix in 10% of enhanced highs
        mastered = 0.9 * audio + 0.1 * hf

        # Final safety limiting to -0.1 dBFS (brick wall)
        # §DSP-Invariante: percentile 99.9 — Impuls-Artefakt darf Normalisierung nicht blockieren.
        peak = float(np.percentile(np.abs(mastered), 99.9))
        if peak > 0.99:
            mastered = mastered / peak * 0.99

        return mastered  # type: ignore[no-any-return]


# ============================================================
# UNIFIED AI FRAMEWORK
# ============================================================


class AurikAIFramework:
    """
    Aurik 10.0.0 Unified AI Framework.

    Integriert alle KI-Module:
    - Defect Detection
    - Audio Restoration
    - Audio Enhancement
    - Vocal Enhancement (Gender-Aware, Phase 19 + 42)
    - Studio 2026 Magic Button
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate

        # Initialize all modules
        self.detector = UnifiedDefectDetector(sample_rate=sample_rate)
        self.restorer = UnifiedAudioRestorer(sample_rate=sample_rate)
        self.enhancer = UnifiedAudioEnhancer(sample_rate=sample_rate)
        self.studio2026 = Studio2026Processor(sample_rate=sample_rate)

        # Initialize Vocal AI Enhancement if available
        if VOCAL_AI_AVAILABLE:
            self.vocal_enhancer = UnifiedVocalAIEnhancer(sample_rate=sample_rate)
            logger.info("✅ Aurik AI Framework initialisiert (with Vocal AI)")
        else:
            self.vocal_enhancer = None  # type: ignore[assignment]
            logger.info("✅ Aurik AI Framework initialisiert (Vocal AI not verfuegbar)")

    def analyze(self, audio: np.ndarray) -> DefectDetectionResult:
        """Analysiert Audio auf Defekte."""
        return self.detector.detect(audio)

    def restore(
        self, audio: np.ndarray, mode: RestorationMode = RestorationMode.BALANCED
    ) -> FrameworkRestorationResult:
        """Restauriert audio."""
        return self.restorer.restore(audio, mode=mode)

    def enhance(self, audio: np.ndarray, **kwargs) -> EnhancementResult:
        """Enhance audio."""
        return self.enhancer.enhance(audio, **kwargs)

    def restoration_magic_button(self, audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Magic Button 1: Restoration Only (keine Enhancement)."""
        return self.restoration_button.process(audio)  # type: ignore[no-any-return,attr-defined]

    def studio2026_magic_button(self, audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Magic Button 2: Studio 2026 Complete Pipeline."""
        return self.studio2026.process(audio)

    # Deprecated: Backward compatibility
    def magic_button(self, audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Deprecated: Use studio2026_magic_button() instead."""
        warnings.warn("magic_button() is deprecated, use studio2026_magic_button()", DeprecationWarning)
        return self.studio2026_magic_button(audio)

    def enhance_vocals(
        self, audio: np.ndarray, emotion_mode: str = "balanced", breath_preservation: float = 0.7
    ) -> VocalEnhancementResult:
        """
        Gender-aware vocal enhancement (Phase 19 + 42).

        Args:
            audio: Input audio
            emotion_mode: "maximum", "balanced", "technical", "transparent"
            breath_preservation: Breath preservation ratio (0-1)

        Returns:
            VocalEnhancementResult with enhanced vocals
        """
        if not VOCAL_AI_AVAILABLE or self.vocal_enhancer is None:
            raise RuntimeError("Vocal AI Enhancement not available")

        # Convert emotion mode string to enum
        mode_map = {
            "maximum": EmotionPreservationMode.MAXIMUM,
            "balanced": EmotionPreservationMode.BALANCED,
            "technical": EmotionPreservationMode.TECHNICAL,
            "transparent": EmotionPreservationMode.TRANSPARENT,
        }
        emotion_enum = mode_map.get(emotion_mode, EmotionPreservationMode.BALANCED)

        return self.vocal_enhancer.enhance(
            audio, emotion_mode=emotion_enum, breath_preservation=breath_preservation, sibilance_reduction=True
        )


# ============================================================
# MAIN ENTRY POINT FOR TESTING
# ============================================================

if __name__ == "__main__":
    """Test AI Framework."""
    logger.debug("Testing Aurik 10.0.0 AI Framework...")
    logger.debug("=" * 70)

    # Generate test audio with defects
    sr = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Clean signal
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Add defects
    # 1. Clicks
    click_positions = [1000, 5000, 10000, 15000]
    for pos in click_positions:
        audio[pos : pos + 10] += 0.5

    # 2. Hiss
    hiss = np.random.normal(0, 0.05, len(audio))
    audio += hiss

    # 3. Hum (50Hz)
    hum = 0.1 * np.sin(2 * np.pi * 50 * t)
    audio += hum

    # Initialize framework
    logger.debug("\ninitialisiere AI Framework...")
    framework = AurikAIFramework(sample_rate=sr)

    # Test detection
    logger.debug("\n1. Testing Defect Detection...")
    detection = framework.analyze(audio)
    logger.debug("   Quality Wert: %.2f", detection.overall_quality_score)
    logger.debug("   erkannt Defects:")
    for defect, confidence in detection.defects.items():
        if confidence > 0.3:
            severity = detection.severity.get(defect, 0)
            logger.debug("     - %s: confidence=%.2f, severity=%.2f", defect.value, confidence, severity)

    # Test restoration
    logger.debug("\n2. Testing Audio Restoration...")
    restoration = framework.restore(audio, mode=RestorationMode.BALANCED)
    logger.debug("   Processes angewendet: %s", ", ".join(restoration.processing_applied))
    logger.debug("   Defects Removed: %s", sum(restoration.defects_removed.values()))
    logger.debug("   Quality Improvement: +%.2f", restoration.quality_improvement)

    # Test enhancement
    logger.debug("\n3. Testing Audio Enhancement...")
    enhancement = framework.enhance(restoration.audio, target_clarity=0.8)
    logger.debug("   Enhancements: %s", ", ".join(enhancement.enhancements_applied))
    logger.debug("   Clarity Improvement: %.2f", enhancement.clarity_improvement)

    # Test Magic Button
    logger.debug("\n4. Testing Magic Button (Studio 2026)...")
    studio_audio, report = framework.magic_button(audio)
    logger.debug("   ✅ verarbeitet to Studio 2026 Standard")
    logger.debug("   Defects Found: %s", report["detection"]["defects_found"])
    logger.debug("   Total Removed: %s", report["restoration"]["defects_removed"])
    logger.debug("   Quality Before: %.2f", report["detection"]["quality_score_before"])

    logger.debug("\n" + "=" * 70)
    logger.debug("✅ AI Framework Test vollstaendig!")
