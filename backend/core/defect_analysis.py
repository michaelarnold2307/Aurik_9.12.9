"""
Defect Analysis für AURIK

Analysiert Audio auf Defekte um intelligentes Phase-Skipping zu ermöglichen:
- Clipping Detection
- Click/Pop Detection
- Dropout Detection
- Noise Level
- Spectral Artifacts

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-08
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore
import logging

from scipy import signal

logger = logging.getLogger(__name__)


class SourceMedium(Enum):
    """Source medium type."""

    UNKNOWN = "unknown"
    DIGITAL = "digital"  # CD, WAV, FLAC (clean source)
    VINYL = "vinyl"  # Vinyl record (clicks, crackle)
    SHELLAC = "shellac"  # 78rpm shellac (heavy clicks, noise)
    CASSETTE = "cassette"  # Cassette tape (dropouts, wow/flutter)
    REEL_TAPE = "reel_tape"  # Reel-to-reel tape (professional, better quality)
    DAT = "dat"  # DAT tape (occasional dropouts, digital)
    MP3_LOW = "mp3_low"  # Low-bitrate MP3 <128kbps (heavy codec artifacts)
    MP3_HIGH = "mp3_high"  # High-bitrate MP3 ≥128kbps (moderate artifacts)
    AAC = "aac"  # AAC/M4A (efficient compression, better than MP3)
    MINIDISC = "minidisc"  # MiniDisc ATRAC (90s/2000s, aggressive compression)
    STREAMING = "streaming"  # Streaming (glitches, packet loss)
    WAX_CYLINDER = "wax_cylinder"  # Wachszylinder
    WIRE_RECORDING = "wire_recording"  # Drahtaufnahme
    LACQUER_DISC = "lacquer_disc"  # Lackfolien-Disc
    TAPE = "tape"  # Generic tape
    CD_DIGITAL = "cd_digital"  # Compact Disc


@dataclass
class DefectAnalysis:
    """Complete defect analysis of audio."""

    # Source medium
    medium: SourceMedium = SourceMedium.UNKNOWN
    medium_confidence: float = 0.0

    # Clipping
    clipping_percentage: float = 0.0  # 0-100% of samples clipped
    clipping_severity: float = 0.0  # 0-1 (how hard clipped)

    # Clicks/Pops
    click_count: int = 0
    click_density: float = 0.0  # Clicks per second

    # Dropouts
    dropout_count: int = 0
    dropout_duration_total_sec: float = 0.0

    # Noise
    noise_floor_db: float = -60.0  # Background noise level
    has_hiss: bool = False
    has_hum: bool = False

    # Spectral artifacts
    has_aliasing: bool = False
    has_codec_artifacts: bool = False

    # Overall quality
    overall_quality: float = 1.0  # 0-1 (1 = pristine)

    def is_clean(self) -> bool:
        """Prüft if audio is relatively clean (minimal defects)."""
        return (
            self.clipping_percentage < 1.0
            and self.click_density < 0.5  # Less than 0.5 clicks/sec
            and self.dropout_count == 0
            and self.noise_floor_db < -50.0
            and self.overall_quality > 0.8
        )

    def needs_declipping(self) -> bool:
        """Prüft if declipping is needed."""
        return self.clipping_percentage >= 0.5  # >0.5% clipped samples

    def needs_click_removal(self) -> bool:
        """Prüft if click removal is needed."""
        return self.click_count > 0 or self.medium in [SourceMedium.VINYL, SourceMedium.SHELLAC]

    def needs_dropout_repair(self) -> bool:
        """Prüft if dropout repair is needed."""
        return self.dropout_count > 0 or self.medium in [
            SourceMedium.CASSETTE,
            SourceMedium.REEL_TAPE,
            SourceMedium.DAT,
        ]

    def needs_denoising(self) -> bool:
        """Prüft if denoising is needed."""
        return self.noise_floor_db > -50.0 or self.has_hiss or self.has_hum

    def __repr__(self) -> str:
        return (
            f"DefectAnalysis(medium={self.medium.value}, "
            f"clipping={self.clipping_percentage:.1f}%, "
            f"clicks={self.click_count}, "
            f"dropouts={self.dropout_count}, "
            f"quality={self.overall_quality:.2f})"
        )


class DefectAnalyzer:
    """Analysiert Audio auf Defekte."""

    def __init__(self):
        """Initialisiert defect analyzer."""

    def analyze(self, audio: np.ndarray, sr: int) -> DefectAnalysis:
        """
        Analysiert Audio auf Defekte.

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            DefectAnalysis with detected defects
        """
        analysis = DefectAnalysis()

        # Detect clipping
        analysis.clipping_percentage, analysis.clipping_severity = self._detect_clipping(audio)

        # Detect clicks
        analysis.click_count = self._detect_clicks(audio, sr)
        duration_sec = len(audio) / sr
        analysis.click_density = analysis.click_count / duration_sec if duration_sec > 0 else 0

        # Detect dropouts
        analysis.dropout_count, analysis.dropout_duration_total_sec = self._detect_dropouts(audio, sr)

        # Estimate noise floor
        analysis.noise_floor_db = self._estimate_noise_floor(audio)

        # Detect hiss
        analysis.has_hiss = self._detect_hiss(audio, sr)

        # Detect hum
        analysis.has_hum = self._detect_hum(audio, sr)

        # Detect medium (heuristic based on defects)
        analysis.medium, analysis.medium_confidence = self._detect_medium(analysis)

        # Compute overall quality
        analysis.overall_quality = self._compute_overall_quality(analysis)

        return analysis

    def _detect_clipping(self, audio: np.ndarray) -> tuple[float, float]:
        """Erkennt clipping."""
        # Check samples near +/- 1.0
        threshold = 0.99
        clipped = np.abs(audio) >= threshold
        clipping_percentage = 100.0 * np.sum(clipped) / len(audio)

        # Severity: how close to actual clipping
        if clipping_percentage > 0:
            clipped_samples = audio[clipped]
            severity = float(np.mean(np.abs(clipped_samples)))
        else:
            severity = 0.0

        return clipping_percentage, severity

    def _detect_clicks(self, audio: np.ndarray, sr: int) -> int:
        """Erkennt clicks/pops."""
        if librosa is None:
            return 0
        # Use onset detection with high threshold
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)  # type: ignore[attr-defined]
        onset_frames = librosa.onset.onset_detect(  # type: ignore[attr-defined]
            onset_envelope=onset_env,
            sr=sr,
            delta=0.5,  # High threshold for clicks
        )

        # Filter for very short, sharp onsets (clicks)
        click_count = 0
        for onset_frame in onset_frames:
            onset_sample = int(librosa.frames_to_samples(onset_frame))

            # Check if onset is very sharp (click characteristic)
            window_samples = int(0.005 * sr)  # 5ms window
            if onset_sample + window_samples < len(audio):
                segment = audio[onset_sample : onset_sample + window_samples].astype(float)
                envelope = np.abs(signal.hilbert(np.asarray(segment, dtype=np.float64)))  # type: ignore[call-overload]

                # Clicks have very rapid attack
                if len(envelope) > 1:
                    max_gradient: float = float(np.max(np.abs(np.gradient(envelope))))
                    if max_gradient > 0.2:  # Very sharp
                        click_count += 1

        return click_count

    def _detect_dropouts(self, audio: np.ndarray, sr: int) -> tuple[int, float]:
        """Erkennt dropouts (sudden amplitude drops)."""
        # Compute RMS in short windows
        window_size = int(0.01 * sr)  # 10ms
        hop_size = window_size // 2

        rms = np.array(
            [np.sqrt(np.mean(audio[i : i + window_size] ** 2)) for i in range(0, len(audio) - window_size, hop_size)]
        )

        # Detect sudden drops
        rms_threshold = np.median(rms) * 0.1  # Drop to 10% of median
        dropout_frames = rms < rms_threshold

        # Count continuous dropout regions
        dropout_count = 0
        dropout_duration_total = 0.0
        in_dropout = False
        dropout_start = 0

        for i, is_dropout in enumerate(dropout_frames):
            if is_dropout and not in_dropout:
                dropout_start = i
                in_dropout = True
            elif not is_dropout and in_dropout:
                dropout_count += 1
                dropout_duration = (i - dropout_start) * hop_size / sr
                dropout_duration_total += dropout_duration
                in_dropout = False

        return dropout_count, dropout_duration_total

    def _estimate_noise_floor(self, audio: np.ndarray) -> float:
        """Schätzt noise floor in dB."""
        # Use quietest 10% of audio
        rms_values = []
        window_size = len(audio) // 100

        for i in range(0, len(audio) - window_size, window_size):
            segment_rms = np.sqrt(np.mean(audio[i : i + window_size] ** 2))
            rms_values.append(segment_rms)

        if not rms_values:
            return -60.0

        # Noise floor = 10th percentile
        noise_floor = np.percentile(rms_values, 10)
        noise_floor_db = 20 * np.log10(noise_floor + 1e-10)

        return float(np.clip(noise_floor_db, -80, 0))

    def _detect_hiss(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt tape hiss (high-frequency noise)."""
        # High-pass filter above 4 kHz
        sos = signal.butter(4, 4000, "high", fs=sr, output="sos")
        filtered = signal.sosfilt(sos, audio)

        # Compute energy ratio
        high_freq_energy = np.mean(filtered**2)
        total_energy = np.mean(audio**2)

        if total_energy > 0:
            ratio = high_freq_energy / total_energy
            return bool(ratio > 0.1)  # type: ignore[no-any-return]  # More than 10% high-freq energy

        return False

    def _detect_hum(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt electrical hum (50/60 Hz)."""
        # FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        magnitudes = np.abs(fft)

        # Check for peaks at 50 Hz and 60 Hz
        def check_frequency(target_hz, tolerance=5):
            mask = (freqs >= target_hz - tolerance) & (freqs <= target_hz + tolerance)
            if np.any(mask):
                peak: float = float(np.max(magnitudes[mask]))
                median = np.median(magnitudes)
                return peak > median * 10  # 10x above median
            return False

        has_50hz = check_frequency(50)
        has_60hz = check_frequency(60)

        return has_50hz or has_60hz  # type: ignore[no-any-return]

    def _detect_medium(self, analysis: DefectAnalysis) -> tuple[SourceMedium, float]:
        """Erkennt source medium based on defect patterns."""
        # Heuristic-based detection

        # Vinyl: clicks + hiss
        if analysis.click_density > 1.0 and analysis.has_hiss:
            return SourceMedium.VINYL, 0.8

        # Shellac: many clicks
        if analysis.click_density > 5.0:
            return SourceMedium.SHELLAC, 0.7

        # Tape: dropouts
        if analysis.dropout_count > 0:
            if analysis.dropout_count > 5:
                return SourceMedium.CASSETTE, 0.6
            else:
                return SourceMedium.REEL_TAPE, 0.6

        # Digital with clipping
        if analysis.clipping_percentage > 5.0:
            return SourceMedium.DIGITAL, 0.7  # Over-mastered CD

        # Clean digital
        if analysis.click_count == 0 and analysis.dropout_count == 0 and analysis.noise_floor_db < -50:
            return SourceMedium.DIGITAL, 0.9

        return SourceMedium.UNKNOWN, 0.0

    def _compute_overall_quality(self, analysis: DefectAnalysis) -> float:
        """Berechnet overall quality score (0-1)."""
        quality = 1.0

        # Penalize defects
        quality -= analysis.clipping_percentage / 100.0 * 0.5
        quality -= min(analysis.click_density / 10.0, 0.3)
        quality -= min(analysis.dropout_count / 10.0, 0.2)

        # Penalize high noise
        if analysis.noise_floor_db > -40:
            quality -= 0.2
        elif analysis.noise_floor_db > -50:
            quality -= 0.1

        return float(np.clip(quality, 0, 1))


if __name__ == "__main__":
    # Demo
    logger.debug("AURIK Defect Analyse")
    logger.debug("=" * 60)
    logger.debug("\nDefect Types erkannt:")
    logger.debug("  • Clipping (>0.99 amplitude)")
    logger.debug("  • Clicks/Pops (sharp transients)")
    logger.debug("  • Dropouts (amplitude drops)")
    logger.debug("  • Noise Floor (background noise)")
    logger.debug("  • Hiss (high-frequency noise)")
    logger.debug("  • Hum (50/60 Hz electrical)")
    logger.debug("\nSource Medium Detection:")
    logger.debug("  • Digital, Vinyl, Shellac, Cassette, Reel Tape, DAT")
