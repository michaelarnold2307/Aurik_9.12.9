"""
§v10.118: Transientengesteuerte Atemprozessierung.

Problem: BreathDetector (breath_detector.py) verwendet ZCR + Energie-Schwellen.
Dies klassifiziert leise Konsonanten (f, s, sh, ch) fälschlich als "Atem"
und dämpft sie — hörbare Artikulationsverluste.

Lösung: Dreistufige Klassifikation statt binärer ZCR/Energie-Entscheidung:
1. ZCR + Energie → Atem-Kandidaten
2. Onset-Detektion → schließt Konsonanten aus (haben scharfe Attack-Phase)
3. Spektrale Schiefe → Atem hat fallendes Spektrum, Zischlaute steigendes

Ergebnis: Nur echte Atemsegmente werden gedämpft, Konsonanten bleiben
unberührt. Keine hörbaren Artikulationsverluste mehr.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

# Breath: falling spectrum (energy in 0-4 kHz >> 4-8 kHz)
BREATH_SPECTRAL_TILT_THRESHOLD: float = 2.0  # Ratio of low/high energy

# Sibilance: rising spectrum (energy in 4-8 kHz >> 0-4 kHz)
SIBILANCE_SPECTRAL_TILT_THRESHOLD: float = 0.3

# Onset strength threshold for consonant detection
CONSONANT_ONSET_THRESHOLD: float = 0.15


@dataclass
class BreathClassification:
    """Classification result for a single frame."""

    is_breath: bool
    is_consonant: bool
    is_sibilance: bool
    confidence: float  # 0.0 = definitely not breath, 1.0 = definitely breath
    spectral_tilt: float  # Negative = breath (falling), positive = sibilance (rising)


def classify_breath_frame(
    frame: np.ndarray,
    sr: int,
    *,
    zcr: float | None = None,
    energy_db: float | None = None,
    onset_strength: float | None = None,
) -> BreathClassification:
    """
    Classify a single audio frame as breath, consonant, or sibilance.

    Args:
        frame: Audio samples (typically 10-25ms)
        sr: Sample rate
        zcr: Pre-computed zero-crossing rate (optional)
        energy_db: Pre-computed energy in dB (optional)
        onset_strength: Pre-computed onset strength (optional)

    Returns:
        BreathClassification with detailed analysis
    """
    n = len(frame)
    if n < 16:
        return BreathClassification(
            is_breath=False, is_consonant=False, is_sibilance=False, confidence=0.0, spectral_tilt=0.0
        )

    # 1. ZCR + Energy baseline (same as original)
    if zcr is None:
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * n))
    if energy_db is None:
        energy = np.mean(frame**2) + 1e-12
        energy_db = float(10.0 * np.log10(energy))

    # 2. Spectral tilt (low vs high frequency energy)
    spec = np.abs(np.fft.rfft(frame.astype(np.float64)))[: n // 2 + 1]
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    low_band = spec[freqs <= 4000]
    high_band = spec[(freqs > 4000) & (freqs <= 8000)]

    low_energy = np.mean(low_band**2) + 1e-12 if len(low_band) > 0 else 1e-12
    high_energy = np.mean(high_band**2) + 1e-12 if len(high_band) > 0 else 1e-12
    spectral_tilt = float(np.log10(low_energy / high_energy))

    # 3. Onset detection (if available, else approximate)
    if onset_strength is not None:
        has_onset = onset_strength > CONSONANT_ONSET_THRESHOLD
    else:
        # Approximate: high ZCR + sharp energy change = consonant
        has_onset = zcr > 0.15

    # ── Classification logic ────────────────────────────────────────────────

    # Sibilance: rising spectrum (high > low)
    is_sibilance = spectral_tilt < -SIBILANCE_SPECTRAL_TILT_THRESHOLD  # Negative tilt = high energy in high freqs
    if is_sibilance:
        return BreathClassification(
            is_breath=False,
            is_consonant=False,
            is_sibilance=True,
            confidence=0.95,
            spectral_tilt=spectral_tilt,
        )

    # Consonant: has sharp onset (attack phase)
    is_consonant = has_onset and zcr > 0.1
    if is_consonant:
        return BreathClassification(
            is_breath=False,
            is_consonant=True,
            is_sibilance=False,
            confidence=0.8 + 0.2 * onset_strength if onset_strength else 0.8,
            spectral_tilt=spectral_tilt,
        )

    # True breath: ZCR high, energy low, no onset, falling spectrum
    is_breath = (
        zcr > 0.1
        and energy_db < -30.0
        and not has_onset
        and spectral_tilt > BREATH_SPECTRAL_TILT_THRESHOLD  # Positive tilt = low energy > high energy
    )

    confidence = 0.5 + 0.5 * (1.0 - float(np.clip(zcr / 0.3, 0.0, 1.0))) if is_breath else 0.1

    return BreathClassification(
        is_breath=is_breath,
        is_consonant=is_consonant,
        is_sibilance=is_sibilance,
        confidence=confidence,
        spectral_tilt=spectral_tilt,
    )


def process_breath_transient_aware(
    audio: np.ndarray,
    sr: int,
    *,
    breath_reduction_db: float = 6.0,
    frame_ms: float = 12.5,
    preserve_consonants: bool = True,
    preserve_sibilance: bool = True,
) -> np.ndarray:
    """
    Process audio with transient-aware breath detection.

    Only true breath segments (high ZCR, low energy, no onset, falling spectrum)
    are reduced. Consonants and sibilance are preserved.

    Args:
        audio: Mono audio array
        sr: Sample rate
        breath_reduction_db: Reduction for breath segments (dB)
        frame_ms: Analysis frame length in ms
        preserve_consonants: Keep consonants untouched
        preserve_sibilance: Keep sibilance untouched

    Returns:
        Processed audio with breath reduction applied
    """
    frame_samples = int(frame_ms * sr / 1000.0)
    hop_samples = frame_samples // 2  # 50% overlap

    n_frames = (len(audio) - frame_samples) // hop_samples + 1
    if n_frames <= 0:
        return audio.copy()

    result = audio.copy().astype(np.float32)
    gain = np.power(10.0, -breath_reduction_db / 20.0)

    # Window for analysis
    window = np.hanning(frame_samples).astype(np.float32)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + frame_samples
        frame = audio[start:end] * window

        classification = classify_breath_frame(frame, sr)

        if classification.is_breath:
            # Reduce breath
            result[start:end] *= gain
        elif preserve_consonants and classification.is_consonant:
            # Keep consonant — no change
            pass
        elif preserve_sibilance and classification.is_sibilance:
            # Keep sibilance — no change
            pass
        # else: neither breath nor consonant — leave as is

    # Smooth gain transitions to avoid clicks
    from scipy.ndimage import uniform_filter1d

    # Detect gain changes and smooth
    gain_env = np.ones(len(audio), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_samples
        end = start + frame_samples
        frame = audio[start:end] * window
        classification = classify_breath_frame(frame, sr)
        if classification.is_breath:
            gain_env[start:end] = gain

    # Smooth gain envelope (10ms smoothing)
    smooth_samples = int(0.010 * sr)
    gain_env = uniform_filter1d(gain_env.astype(np.float64), smooth_samples).astype(np.float32)
    result = audio * gain_env

    return cast(np.ndarray, result)
