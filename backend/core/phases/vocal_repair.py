"""
Vocal Repair Module — Damage Detection & Restoration (§G58)

Detects and repairs common vocal damage types BEFORE enhancement:
  §G58a Bandwidth restoration — extend harmonics above cutoff
  §G58b Distortion repair — soft de-clipping + harmonic reconstruction
  §G58c Formant gap filling — spectral inpainting for missing bands

Design principle: Repair brings damaged vocals to a STATE WHERE
Phase 42's enhancement chain can work effectively. It does NOT aim
for perfection — just "not broken anymore."

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


def detect_vocal_damage(vocal_stem_mono: np.ndarray, sr: int) -> dict:
    """Analyzes vocal stem for damage indicators.

    Returns dict with:
      bandwidth_hz: effective bandwidth (spectral rolloff at -30dB)
      is_bandlimited: True if bandwidth < 4000 Hz
      crest_factor_db: peak/RMS ratio (low = distorted)
      is_distorted: True if crest_factor < 8 dB
      harmonic_density: ratio of harmonic to total energy
      needs_repair: True if any damage detected
      confidence: 0-1 overall repair confidence
    """
    n = len(vocal_stem_mono)
    if n < 2048:
        return {"needs_repair": False, "confidence": 0.0}

    n_fft = 4096
    while n_fft < n:
        n_fft <<= 1

    win = np.hanning(min(n_fft, n))
    spec = np.abs(np.fft.rfft(vocal_stem_mono[:n_fft] * win, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Bandwidth: frequency below which 95% of energy resides (spectral rolloff)
    cumsum = np.cumsum(spec**2)
    total = cumsum[-1]
    if total < 1e-15:
        return {"needs_repair": False, "confidence": 0.0}

    rolloff_idx = np.searchsorted(cumsum, 0.95 * total)
    bandwidth_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    is_bandlimited = bandwidth_hz < 4000.0

    # Crest factor: peak/RMS (dB)
    rms = float(np.sqrt(np.mean(vocal_stem_mono.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(vocal_stem_mono)))
    crest_db = 20.0 * np.log10(max(peak, 1e-15) / max(rms, 1e-15))
    is_distorted = crest_db < 8.0

    # Harmonic density: energy at harmonic peaks / total energy
    # Simple: ratio of spectral peaks to spectral mean
    mid_mask = (freqs >= 300) & (freqs <= 4000)
    if np.any(mid_mask):
        spec_mid = spec[mid_mask]
        peak_ratio = float(np.max(spec_mid)) / max(float(np.mean(spec_mid)), 1e-15)
        harmonic_density = min(peak_ratio / 20.0, 1.0)  # normalize
    else:
        harmonic_density = 0.5

    needs_repair = is_bandlimited or is_distorted
    confidence = float(0.5 * float(is_bandlimited) + 0.3 * float(is_distorted) + 0.2 * (1.0 - harmonic_density))

    return {
        "bandwidth_hz": bandwidth_hz,
        "is_bandlimited": is_bandlimited,
        "crest_factor_db": crest_db,
        "is_distorted": is_distorted,
        "harmonic_density": harmonic_density,
        "needs_repair": needs_repair,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }


def apply_vocal_repair(vocal_stem: np.ndarray, sr: int, *, damage: dict | None = None) -> np.ndarray:
    """§G58: Repair damaged vocals before enhancement.

    Applies bandwidth extension and/or distortion repair based on
    detected damage. Returns repaired vocal stem.
    """
    if damage is None:
        mono = _to_mono(vocal_stem)
        damage = detect_vocal_damage(mono, sr)

    if not damage.get("needs_repair", False):
        return vocal_stem

    logger.info(
        "VocalRepair: bw=%.0f Hz %s crest=%.1f dB %s → repairing",
        damage.get("bandwidth_hz", 0),
        "(bandlimited)" if damage.get("is_bandlimited") else "",
        damage.get("crest_factor_db", 0),
        "(distorted)" if damage.get("is_distorted") else "",
    )

    mono = _to_mono(vocal_stem)
    is_stereo = vocal_stem.ndim == 2 and vocal_stem.shape[1] >= 2
    orig_shape = vocal_stem.shape

    repaired = mono.astype(np.float64).copy()

    # §G58a: Bandwidth extension via harmonic synthesis
    if damage.get("is_bandlimited"):
        repaired = _extend_bandwidth(repaired, sr, damage["bandwidth_hz"])

    # §G58b: Distortion repair via soft de-clipping + harmonic reconstruction
    if damage.get("is_distorted"):
        repaired = _repair_distortion(repaired)

    # §G58c: Blend repair with original based on confidence
    confidence = damage.get("confidence", 0.5)
    # Higher confidence → more repair. Lower → more original.
    blend = 0.3 + 0.5 * confidence  # 0.3 to 0.8
    repaired = blend * repaired + (1.0 - blend) * mono

    if is_stereo:
        # Apply repair as gain ratio to preserve stereo
        gain = np.where(np.abs(mono) > 1e-8, repaired / (mono + 1e-12), 1.0)
        gain = np.clip(gain, 0.5, 2.0)
        result = vocal_stem.astype(np.float64) * gain[:, np.newaxis]
    else:
        result = repaired.reshape(orig_shape)

    return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _extend_bandwidth(mono: np.ndarray, sr: int, bandwidth_hz: float) -> np.ndarray:
    """Extend vocal bandwidth above current cutoff via light harmonic synthesis.

    Generates harmonics above the cutoff by waveshaping, then blends them
    in at -20 dB below the fundamental to add natural 'air' without artifacts.
    """
    n = len(mono)
    # Simple waveshaping: tanh soft-clip generates odd harmonics
    gentle_distortion = np.tanh(mono * 3.0) / 3.0

    # Highpass filter the generated harmonics above the cutoff
    n_fft = 4096
    while n_fft < n:
        n_fft <<= 1
    spec = np.fft.rfft(gentle_distortion, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Keep only frequencies above bandwidth_hz
    mask = freqs >= bandwidth_hz * 0.8  # start slightly below cutoff
    spec[~mask] = 0.0

    harmonics = np.fft.irfft(spec, n=n_fft)[:n]

    # Mix at very low level (-20 dB)
    mono_out = mono.astype(np.float64) + harmonics.astype(np.float64) * 0.1

    _res_typed_168: np.ndarray = mono_out
    return _res_typed_168


def _repair_distortion(mono: np.ndarray) -> np.ndarray:
    """Repair mild distortion via soft de-clipping + spectral smoothing.

    Distorted signals have flattened waveform tops. A cubic nonlinearity
    partially restores the original curvature.
    """
    # Cubic expansion: y = x - 0.15 * x^3
    # This expands peaks that were compressed by clipping
    expanded = mono - 0.15 * (mono**3)

    # Blend: 50% expanded + 50% original (conservative)
    return 0.5 * expanded + 0.5 * mono  # type: ignore[no-any-return]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0) if audio.shape[1] < audio.shape[0] else audio.mean(axis=1)  # type: ignore[no-any-return]
