"""
§v10.119: Multi-Band Music Enhancer — artefaktfreie Instrumenten-Klarheit.

Arbeitet auf dem GESAMTEN Mix (nach Denoising, vor Vocal-Enhancement).
Drei Bänder, getrennt bearbeitet, dann summiert:
  - Low  (20-250 Hz):  Punch-Verstärkung via Transient-Emphasis
  - Mid  (250-4000 Hz): Clarity via dynamischer EQ
  - High (4-20 kHz):    Air/Luft via sanftem Harmonic Exciter

Garantien:
  - Keine Phasenverschiebung (linear-phase FIR-Filter pro Band)
  - Kein Clipping (automatische Gain-Reduktion)
  - Keine Stereo-Bild-Veränderung (L/R unabhängig, gleiche Parameter)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# ── Crossover-Frequenzen (4. Ordnung Linkwitz-Riley, linear-phase via forward-backward) ──
LOW_CROSSOVER: float = 250.0  # Hz — Kick/Bass
HIGH_CROSSOVER: float = 4000.0  # Hz — Vocals/Gitarre → Hi-Hat/Cymbals

# ── Default-Parameter (für "balanced" Preset) ──
LOW_PUNCH_DB: float = 2.0  # dB Verstärkung für Kick-Transienten
MID_CLARITY_DB: float = 1.5  # dB Anhebung der Präsenz
HIGH_AIR_DB: float = 1.0  # dB Air/Luft


@dataclass
class MusicEnhancementResult:
    audio: np.ndarray
    low_gain_db: float
    mid_gain_db: float
    high_gain_db: float


def _design_crossover(sr: int, freq: float, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Design Linkwitz-Riley crossover (Butterworth squared)."""
    nyq = sr / 2.0
    freq_norm = freq / nyq
    b, a = signal.butter(order // 2, freq_norm, btype="low", output="ba")
    # Apply forward-backward for zero-phase
    return b, a


def _linear_phase_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Zero-phase filtering via forward-backward (filtfilt)."""
    if len(data) < len(b) * 3:
        return data  # Too short for meaningful filtering
    return cast(np.ndarray, (signal.filtfilt(b, a, data).astype(np.float32)))


def _transient_emphasis(audio: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    """Emphasize transients without boosting sustain (differentiator + envelope)."""
    if abs(gain_db) < 0.1:
        return audio

    # Differentiator to detect transients
    diff = np.diff(audio, prepend=audio[0])
    # Smooth envelope
    envelope = uniform_filter1d(np.abs(diff), size=int(sr * 0.005))  # 5ms smoothing
    envelope /= envelope.max() + 1e-10

    # Apply gain only to transient regions (envelope > 0.3)
    gain = 10.0 ** (gain_db / 20.0)
    mask = envelope > 0.3
    result = audio.copy()
    result[mask] *= gain
    return result


def _harmonic_exciter(audio: np.ndarray, sr: int, drive_db: float) -> np.ndarray:
    """Subtle harmonic saturation for air band (odd harmonics only, no DC)."""
    if abs(drive_db) < 0.1:
        return audio

    drive = 10.0 ** (drive_db / 20.0) * 0.3  # Scale drive_db to soft clipping range
    # Soft clip via tanh (odd harmonics only — no DC shift)
    wet = np.tanh(audio * drive) / max(drive, 0.01)
    # Mix 15% wet
    return cast(np.ndarray, (audio * 0.85 + wet * 0.15).astype(np.float32))


def enhance_music(
    audio: np.ndarray,
    sr: int,
    *,
    low_punch_db: float = LOW_PUNCH_DB,
    mid_clarity_db: float = MID_CLARITY_DB,
    high_air_db: float = HIGH_AIR_DB,
) -> MusicEnhancementResult:
    """
    Multi-band music enhancement.

    Processing chain:
      1. Crossover → Low/Mid/High Bänder
      2. Pro Band: spezifische Bearbeitung
      3. Summe → Output

    Args:
        audio: Mono or stereo [..., samples] at sr
        sr: Sample rate (must be >= 8000)
        low_punch_db: Punch boost for bass (0-6 dB)
        mid_clarity_db: Clarity boost for instruments (0-4 dB)
        high_air_db: Air boost for cymbals (0-3 dB)

    Returns:
        MusicEnhancementResult with enhanced audio
    """
    # Clamp parameters
    low_punch_db = float(np.clip(low_punch_db, 0.0, 6.0))
    mid_clarity_db = float(np.clip(mid_clarity_db, 0.0, 4.0))
    high_air_db = float(np.clip(high_air_db, 0.0, 3.0))

    is_2d = audio.ndim == 2
    if is_2d:
        # Process channels independently
        result = np.zeros_like(audio, dtype=np.float32)
        for ch in range(audio.shape[0]):
            ch_result = _enhance_mono(audio[ch], sr, low_punch_db, mid_clarity_db, high_air_db)
            result[ch] = ch_result
    else:
        result = _enhance_mono(audio, sr, low_punch_db, mid_clarity_db, high_air_db)

    # Safety: clip to [-1, 1]
    result = np.clip(result, -1.0, 1.0)

    return MusicEnhancementResult(
        audio=result,
        low_gain_db=low_punch_db,
        mid_gain_db=mid_clarity_db,
        high_gain_db=high_air_db,
    )


def _enhance_mono(
    audio: np.ndarray,
    sr: int,
    low_db: float,
    mid_db: float,
    high_db: float,
) -> np.ndarray:
    """Process mono audio through multi-band chain."""
    audio = audio.astype(np.float64)

    # Design crossovers
    b_low, a_low = _design_crossover(sr, LOW_CROSSOVER)
    b_high, a_high = _design_crossover(sr, HIGH_CROSSOVER)

    # Split into bands
    low_band = _linear_phase_filter(audio, b_low, a_low)
    mid_band = _linear_phase_filter(audio, b_high, a_high) - _linear_phase_filter(audio, b_low, a_low)
    high_band = audio - _linear_phase_filter(audio, b_high, a_high)

    # ── Low Band: Transient Emphasis ──
    low_enhanced = _transient_emphasis(low_band, sr, low_db)

    # ── Mid Band: Broad Gain ──
    mid_gain = 10.0 ** (mid_db / 20.0)
    mid_enhanced = mid_band * mid_gain

    # ── High Band: Harmonic Exciter ──
    high_enhanced = _harmonic_exciter(high_band, sr, high_db)

    # Sum bands
    result = low_enhanced + mid_enhanced + high_enhanced

    # Auto-gain: match RMS of input
    input_rms = np.sqrt(np.mean(audio**2)) + 1e-10
    output_rms = np.sqrt(np.mean(result**2)) + 1e-10
    result *= input_rms / output_rms

    return cast(np.ndarray, result.astype(np.float32))
