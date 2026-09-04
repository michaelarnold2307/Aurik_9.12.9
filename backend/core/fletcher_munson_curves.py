#!/usr/bin/env python3
"""
Fletcher-Munson Equal-Loudness Curves
======================================

Implementiert Equal-Loudness Contours nach ISO 226:2003.
Diese Kurven beschreiben, wie laut verschiedene Frequenzen bei gleichem SPL klingen.

Psychoacoustic Phänomen:
- Das menschliche Ohr ist nicht linear über Frequenzen
- 1 kHz = Referenz (bei 40 dB SPL = 40 Phon
)
- Tiefe Frequenzen (<200 Hz) benötigen mehr SPL für gleiche Lautheit
- Hohe Frequenzen (>10 kHz) benötigen ebenfalls mehr SPL
- Maximum Sensitivity: ~3-4 kHz (evolutionär: Sprachbereich)

Fletcher-Munson vs. ISO 226:
- **Fletcher-Munson (1933)**: Historische Messungen, weniger präzise
- **ISO 226:2003**: Moderne, standardisierte Equal-Loudness Contours
- Wir verwenden ISO 226 als Basis mit Fletcher-Munson Name (historisch etabliert)

Anwendungen:
- Loudness Compensation (z.B. bei niedrigen Lautstärken)
- Perceptual EQ
- Tonal Balance Restoration
- Mastering
- Quality Assessment

Autor: Aurik v8.0 - Psychoacoustic Core
Lizenz: Apache 2.0
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EQUAL-LOUDNESS CONTOUR DATA (ISO 226:2003 Approximation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Frequency (Hz) → SPL (dB) für verschiedene Phon Levels
# Format: Frequency (Hz) → {Phon_Level: SPL_dB}
# Simplified ISO 226:2003 data (key frequencies)

EQUAL_LOUDNESS_CONTOURS = {
    # 20 Phon (sehr leise)
    20: [
        (20, 78),
        (25, 70),
        (31.5, 63),
        (40, 56),
        (50, 50),
        (63, 44),
        (80, 38),
        (100, 34),
        (125, 30),
        (160, 26),
        (200, 23),
        (250, 20),
        (315, 17),
        (400, 15),
        (500, 13),
        (630, 11),
        (800, 9),
        (1000, 8),
        (1250, 7),
        (1600, 6),
        (2000, 5),
        (2500, 4),
        (3150, 3),
        (4000, 2),
        (5000, 2),
        (6300, 3),
        (8000, 5),
        (10000, 10),
        (12500, 17),
        (16000, 28),
    ],
    # 40 Phon (leise Unterhaltung)
    40: [
        (20, 96),
        (25, 88),
        (31.5, 81),
        (40, 74),
        (50, 68),
        (63, 62),
        (80, 56),
        (100, 51),
        (125, 47),
        (160, 43),
        (200, 40),
        (250, 37),
        (315, 34),
        (400, 31),
        (500, 29),
        (630, 27),
        (800, 25),
        (1000, 24),
        (1250, 23),
        (1600, 21),
        (2000, 20),
        (2500, 18),
        (3150, 17),
        (4000, 16),
        (5000, 16),
        (6300, 17),
        (8000, 19),
        (10000, 23),
        (12500, 29),
        (16000, 40),
    ],
    # 60 Phon (normal conversation)
    60: [
        (20, 112),
        (25, 104),
        (31.5, 97),
        (40, 90),
        (50, 84),
        (63, 78),
        (80, 72),
        (100, 67),
        (125, 63),
        (160, 59),
        (200, 56),
        (250, 53),
        (315, 50),
        (400, 47),
        (500, 45),
        (630, 43),
        (800, 41),
        (1000, 40),
        (1250, 39),
        (1600, 37),
        (2000, 36),
        (2500, 34),
        (3150, 32),
        (4000, 31),
        (5000, 31),
        (6300, 32),
        (8000, 34),
        (10000, 38),
        (12500, 43),
        (16000, 53),
    ],
    # 80 Phon (laute Musik)
    80: [
        (20, 128),
        (25, 120),
        (31.5, 113),
        (40, 106),
        (50, 100),
        (63, 94),
        (80, 88),
        (100, 83),
        (125, 79),
        (160, 75),
        (200, 72),
        (250, 69),
        (315, 66),
        (400, 63),
        (500, 61),
        (630, 59),
        (800, 57),
        (1000, 56),
        (1250, 55),
        (1600, 53),
        (2000, 52),
        (2500, 50),
        (3150, 48),
        (4000, 46),
        (5000, 46),
        (6300, 47),
        (8000, 49),
        (10000, 53),
        (12500, 58),
        (16000, 67),
    ],
    # 100 Phon (sehr laut)
    100: [
        (20, 144),
        (25, 136),
        (31.5, 129),
        (40, 122),
        (50, 116),
        (63, 110),
        (80, 104),
        (100, 99),
        (125, 95),
        (160, 91),
        (200, 88),
        (250, 85),
        (315, 82),
        (400, 79),
        (500, 77),
        (630, 75),
        (800, 73),
        (1000, 72),
        (1250, 71),
        (1600, 69),
        (2000, 68),
        (2500, 66),
        (3150, 64),
        (4000, 62),
        (5000, 62),
        (6300, 62),
        (8000, 64),
        (10000, 68),
        (12500, 73),
        (16000, 82),
    ],
}


@dataclass
class EqualLoudnessContour:
    """
    Represents an equal-loudness contour at a specific phon level.

    Attributes:
        phon_level: Loudness level in Phons
        frequencies: Frequency points (Hz)
        spl_levels: Sound Pressure Level (dB SPL) at each frequency
    """

    phon_level: int
    frequencies: np.ndarray
    spl_levels: np.ndarray

    def get_spl_at_frequency(self, freq_hz: "float | np.ndarray") -> "float | np.ndarray":
        """
        Gibt zurück: SPL at specific frequency (interpolated).

        Args:
            freq_hz: Frequency in Hz

        Returns:
            SPL in dB SPL
        """
        # Clamp to valid range
        freq_hz = np.clip(freq_hz, self.frequencies[0], self.frequencies[-1])

        # Interpolate (log-space for frequency)
        # §Perf: interp1d-Instanz pro Contour nur EINMAL bauen und cachen —
        # get_correction_curve() ruft diese Methode pro Frequenzpunkt (bis zu
        # 288k Aufrufe pro phase_40-Lauf). Vorher: 67,8 s reine Interpolator-
        # Konstruktion. Identische Werte, nur ohne Wiederaufbau.
        interpolator = getattr(self, "_spl_interpolator_cache", None)
        if interpolator is None:
            interpolator = interp1d(
                np.log10(self.frequencies),
                self.spl_levels,
                kind="linear",  # Changed from cubic to linear (more stable)
                fill_value="extrapolate",
                bounds_error=False,
            )
            self._spl_interpolator_cache = interpolator
        _spl_vals = interpolator(np.log10(freq_hz))
        if isinstance(freq_hz, np.ndarray):
            return np.asarray(_spl_vals, dtype=float)  # type: ignore[no-any-return]
        return float(_spl_vals)

    def get_relative_loudness_curve(self, reference_freq: float = 1000.0) -> np.ndarray:
        """
        Gibt zurück: relative loudness curve (normalized to reference frequency).

        Args:
            reference_freq: Reference frequency (default: 1 kHz)

        Returns:
            Relative loudness in dB (same shape as frequencies)
        """
        ref_spl = self.get_spl_at_frequency(reference_freq)
        relative = self.spl_levels - ref_spl
        return relative  # type: ignore[no-any-return]


@dataclass
class FletcherMunsonConfig:
    """
    Configuration for Fletcher-Munson compensation.

    Attributes:
        target_phon: Target phon level for compensation
        reference_phon: Reference phon level (usually 80-100 for mastering)
        apply_boost: Apply boost compensation (lower frequencies get boosted at low volumes)
        max_correction_db: Maximum correction in dB (safety limit)
        smooth_curve: Apply smoothing to correction curve
    """

    target_phon: int = 60
    reference_phon: int = 80
    apply_boost: bool = True
    max_correction_db: float = 12.0
    smooth_curve: bool = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLETCHER-MUNSON PROCESSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FletcherMunsonProcessor:
    """
    Applies Fletcher-Munson equal-loudness compensation.

    Use Cases:
    1. **Loudness Compensation**: Boost bass/treble at low listening volumes
    2. **Perceptual EQ**: Flatten perceived frequency response
    3. **Tonal Balance**: Restore natural balance after processing

    Example:
        >>> processor = FletcherMunsonProcessor()
        >>> # Compensate for 60 phon listening level (reference: 80 phon mastering)
        >>> correction_curve = processor.get_correction_curve(
        ...     frequencies=np.linspace(20, 20000, 1000),
        ...     target_phon=60,
        ...     reference_phon=80
        ... )
    """

    def __init__(self, config: FletcherMunsonConfig | None = None):
        """
        Initialisiert Fletcher-Munson processor.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or FletcherMunsonConfig()

        # Build contour interpolators
        self.contours = {}
        for phon_level, data in EQUAL_LOUDNESS_CONTOURS.items():
            freqs = np.array([f for f, _ in data])
            spls = np.array([spl for _, spl in data])
            self.contours[phon_level] = EqualLoudnessContour(phon_level=phon_level, frequencies=freqs, spl_levels=spls)

        logger.debug("FletcherMunsonProcessor initialisiert (target=%s phon)", self.config.target_phon)

    def get_contour(self, phon_level: int) -> EqualLoudnessContour:
        """
        Gibt zurück: equal-loudness contour for specific phon level.

        Args:
            phon_level: Phon level (20, 40, 60, 80, 100)

        Returns:
            EqualLoudnessContour object
        """
        if phon_level not in self.contours:
            # Interpolate between available contours
            available = sorted(self.contours.keys())
            if phon_level < available[0]:
                return self.contours[available[0]]
            elif phon_level > available[-1]:
                return self.contours[available[-1]]
            else:
                # Find bracketing levels
                lower = max(p for p in available if p <= phon_level)
                upper = min(p for p in available if p >= phon_level)

                if lower == upper:
                    return self.contours[lower]

                # Interpolate
                weight = (phon_level - lower) / (upper - lower)
                contour_lower = self.contours[lower]
                contour_upper = self.contours[upper]

                spl_interp = (1 - weight) * contour_lower.spl_levels + weight * contour_upper.spl_levels

                return EqualLoudnessContour(
                    phon_level=phon_level, frequencies=contour_lower.frequencies, spl_levels=spl_interp
                )

        return self.contours[phon_level]

    def get_correction_curve(
        self, frequencies: np.ndarray, target_phon: int | None = None, reference_phon: int | None = None
    ) -> np.ndarray:
        """
        Gibt zurück: frequency-dependent correction curve in dB.

        This curve represents the EQ needed to compensate for equal-loudness effects.

        Args:
            frequencies: Frequency points (Hz)
            target_phon: Target listening level (uses config if None)
            reference_phon: Reference level (uses config if None)

        Returns:
            Correction in dB (same shape as frequencies)
        """
        target_phon = target_phon or self.config.target_phon
        reference_phon = reference_phon or self.config.reference_phon

        # Get contours
        target_contour = self.get_contour(target_phon)
        reference_contour = self.get_contour(reference_phon)

        # Interpolate to requested frequencies
        # §Perf: Vektorisiert über den gecachten Interpolator statt 288k Einzel-
        # Aufrufe mit je einer interp1d-Konstruktion (vorher ~22,6 s _evaluate
        # + 32,3 s __init__ pro phase_40). Werte identisch: derselbe Interpolator,
        # dieselben Stützstellen.
        target_spl = np.asarray(target_contour.get_spl_at_frequency(frequencies))
        reference_spl = np.asarray(reference_contour.get_spl_at_frequency(frequencies))

        # Correction = difference in SPL
        # If target < reference: bass needs boost, treble needs boost
        correction_db = target_spl - reference_spl

        if not self.config.apply_boost:
            # Only apply attenuation, not boost
            correction_db = np.minimum(correction_db, 0)

        # Clip to max correction
        correction_db = np.clip(correction_db, -self.config.max_correction_db, self.config.max_correction_db)

        # Smooth curve if requested
        if self.config.smooth_curve and len(correction_db) > 10:
            # Simple moving average
            window = 5
            kernel = np.ones(window) / window
            correction_db = np.convolve(correction_db, kernel, mode="same")

        return correction_db  # type: ignore[no-any-return]

    def apply_compensation(
        self, audio: np.ndarray, sr: int, target_phon: int | None = None, reference_phon: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Wendet an: Fletcher-Munson compensation to audio.

        Args:
            audio: Input audio (mono)
            sr: Sample rate
            target_phon: Target listening level
            reference_phon: Reference level

        Returns:
            (compensated_audio, correction_curve_db)
        """
        # Compute FFT
        spectrum = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Get correction curve
        correction_db = self.get_correction_curve(freqs, target_phon, reference_phon)

        # Convert dB to linear
        correction_linear = 10 ** (correction_db / 20)

        # Apply correction
        spectrum_corrected = spectrum * correction_linear

        # IFFT
        audio_corrected = np.fft.irfft(spectrum_corrected, n=len(audio))
        audio_corrected = np.nan_to_num(audio_corrected, nan=0.0, posinf=0.0, neginf=0.0)
        audio_corrected = np.clip(audio_corrected, -1.0, 1.0)

        return audio_corrected, correction_db


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_adaptive_phon(
    audio: np.ndarray | None = None,
    measured_lufs: float | None = None,
    sr: int = 48000,
    default_phon: int = 60,
) -> int:
    """§v10.0.5: Berechnet LUFS-adaptives target_phon für Fletcher-Munson.

    Leise Aufnahmen (historische Schellacks, -30 LUFS) brauchen ein
    niedrigeres Phon-Level (~40 phon) für korrekte Equal-Loudness-
    Kompensation. Laute moderne Aufnahmen (-8 LUFS) nutzen 70+ phon.

    Formel: phon = clamp(60 + (LUFS + 16) * 1.5, 30, 80)

    Args:
        audio: Optionales Audio (wird auf LUFS gemessen falls kein lufs-Wert)
        measured_lufs: Bereits gemessener LUFS-Wert (überspringt Messung)
        sr: Sample-Rate
        default_phon: Fallback wenn keine Messung möglich

    Returns:
        Adaptives Phon-Level ∈ [30, 80]
    """
    if measured_lufs is None and audio is not None:
        try:
            import pyloudnorm as pyln

            mono = audio if audio.ndim == 1 else audio.mean(axis=0)
            meter = pyln.Meter(sr)
            measured_lufs = float(meter.integrated_loudness(mono.astype(float)))
        except Exception as exc:
            logger.debug("§V6 pyloudnorm.integrated_loudness fehlgeschlagen — Default-Phon zurückgegeben (%d): %s", default_phon, exc)
            return default_phon

    if measured_lufs is None:
        return default_phon

    # Leise Aufnahme → niedrigeres Phon (das Ohr ist bei niedrigen Pegeln
    # weniger sensibel für Bässe und Höhen → sanftere Kompensation).
    # Laute Aufnahme → höheres Phon (das Ohr ist bei hohen Pegeln
    # gleichmäßiger → weniger Kompensation nötig).
    adaptive = int(round(60.0 + (measured_lufs + 16.0) * 1.5))
    return max(30, min(80, adaptive))


def get_fletcher_munson_curve(frequencies: np.ndarray, target_phon: int = 60, reference_phon: int = 80) -> np.ndarray:
    """
    Quick Fletcher-Munson correction curve.

    Args:
        frequencies: Frequency points (Hz)
        target_phon: Target listening level
        reference_phon: Reference level

    Returns:
        Correction curve in dB
    """
    processor = FletcherMunsonProcessor()
    return processor.get_correction_curve(frequencies, target_phon, reference_phon)


def apply_loudness_compensation(
    audio: np.ndarray,
    sr: int,
    listening_level: str = "normal",
    target_phon: float | None = None,
    reference_phon: float | None = None,
) -> np.ndarray:
    """
    Wendet an: loudness compensation based on listening level.

    Args:
        audio: Input audio
        sr: Sample rate
        listening_level: 'quiet' (40 phon), 'normal' (60 phon), 'loud' (80 phon)
        target_phon: Override target phon level (e.g. from §ISO-226 context)
        reference_phon: Override reference phon level

    Returns:
        Compensated audio
    """
    level_map = {"quiet": (40, 80), "normal": (60, 80), "loud": (80, 100)}

    if target_phon is not None and reference_phon is not None:
        pass  # Use explicit values — bypass level_map lookup
    else:
        target_phon, reference_phon = level_map.get(listening_level, (60, 80))

    config = FletcherMunsonConfig(target_phon=target_phon, reference_phon=reference_phon)  # type: ignore[arg-type]
    processor = FletcherMunsonProcessor(config)

    compensated, _ = processor.apply_compensation(audio, sr)
    return compensated


if __name__ == "__main__":
    """Demo Fletcher-Munson processor"""
    logger.debug("\n" + "=" * 70)
    logger.debug("FLETCHER-MUNSON EQUAL-LOUDNESS CURVES - Demo")
    logger.debug("=" * 70 + "\n")

    # Initialize processor
    processor = FletcherMunsonProcessor()

    # Test frequencies
    test_freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])

    logger.debug("Equal-Loudness Comparison (dB SPL required for equal loudness):\n")
    logger.debug("Frequency | 20 Phon | 40 Phon | 60 Phon | 80 Phon | 100 Phon")
    logger.debug("-" * 70)

    for freq in test_freqs:
        values = []
        for phon in [20, 40, 60, 80, 100]:
            contour = processor.get_contour(phon)
            spl = contour.get_spl_at_frequency(freq)
            values.append(f"{spl:7.1f}")
        logger.debug('%5.0f Hz | " + " | ".join(values) + " |', freq)

    # Correction curves
    logger.debug("\n\nLoudness Compensation Curves (dB correction):\n")

    freqs = np.logspace(np.log10(20), np.log10(20000), 20)

    logger.debug("Listening at 60 phon (normal), mastered at 80 phon (loud):")
    correction_60 = processor.get_correction_curve(freqs, target_phon=60, reference_phon=80)

    logger.debug("\nFreq (Hz) | Correction (dB)")
    logger.debug("-" * 35)
    for f, c in zip(freqs, correction_60):
        logger.debug("%8.0f | %+6.2f", f, c)

    # Test with audio
    logger.debug("\n\nApplying compensation to test signal...")
    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-tone signal
    audio = (
        np.sin(2 * np.pi * 100 * t)  # 100 Hz bass
        + np.sin(2 * np.pi * 1000 * t)  # 1 kHz mid
        + np.sin(2 * np.pi * 10000 * t)  # 10 kHz treble
    )
    audio = audio / np.abs(audio).max()

    compensated_quiet, _ = processor.apply_compensation(audio, sr, target_phon=40, reference_phon=80)
    compensated_normal, _ = processor.apply_compensation(audio, sr, target_phon=60, reference_phon=80)

    logger.debug("  Originalsignal RMS: %.4f", np.sqrt(np.mean(audio**2)))
    logger.debug("  Quiet (40 phon) RMS: %.4f", np.sqrt(np.mean(compensated_quiet**2)))
    logger.debug("  Normal (60 phon) RMS: %.4f", np.sqrt(np.mean(compensated_normal**2)))

    logger.debug("\n" + "=" * 70)
    logger.debug("Demo vollstaendig!")
    logger.debug("=" * 70 + "\n")
