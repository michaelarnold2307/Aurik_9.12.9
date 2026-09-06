"""ReferenceMatcher — §INCREMENTAL #9.

Nutzer gibt Referenz-Track → Aurik matched EQ, Dynamics, Stereo-Width.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass  # type: ignore[name-defined]
class MatchProfile:
    target_eq_curve: np.ndarray = None  # type: ignore[assignment]
    target_rms_db: float = -18.0
    target_stereo_width: float = 0.5
    target_spectral_centroid: float = 2000.0


def analyze_reference(audio: np.ndarray, sr: int) -> MatchProfile:
    """Extrahiert EQ/Dynamics/Stereo-Profil aus Referenz-Track."""
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    n_fft = 4096
    spec = np.abs(np.fft.rfft(mono[: n_fft * 16], n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # EQ-Kurve (langfristiges Spektrum)
    n_frames = max(1, len(mono) // (n_fft // 2))
    long_spec = np.zeros(n_fft // 2 + 1)
    for i in range(n_frames):
        start = i * n_fft // 2
        chunk = mono[start : start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        long_spec += np.abs(np.fft.rfft(chunk * np.hanning(n_fft)))
    long_spec /= max(n_frames, 1)

    rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
    centroid = float(np.sum(freqs * long_spec) / max(np.sum(long_spec), 1e-10))
    stereo = 0.5
    if audio.ndim == 2:
        l, r = audio[:, 0], audio[:, 1]
        stereo = float(np.clip(1.0 - abs(np.corrcoef(l, r)[0, 1]), 0.0, 1.0))

    return MatchProfile(
        target_eq_curve=long_spec.astype(np.float32),
        target_rms_db=20 * np.log10(rms),
        target_stereo_width=stereo,
        target_spectral_centroid=centroid,
    )


def apply_match(audio: np.ndarray, sr: int, target: MatchProfile) -> np.ndarray:
    """Wendet EQ-Matching an."""
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    n_fft = 4096

    # Quell-EQ-Kurve
    src_long = np.zeros(n_fft // 2 + 1)
    n_frames = max(1, len(mono) // (n_fft // 2))
    for i in range(n_frames):
        start = i * n_fft // 2
        chunk = mono[start : start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        src_long += np.abs(np.fft.rfft(chunk * np.hanning(n_fft)))
    src_long /= max(n_frames, 1)

    # EQ-Korrektur: target / source (frequenz-abhängiger Gain)
    eq_gain = target.target_eq_curve / (src_long + 1e-10)
    eq_gain = np.clip(eq_gain, 0.1, 10.0)

    # Anwenden via FFT
    result = np.zeros_like(mono)
    hop = n_fft // 4
    for i in range(0, len(mono) - n_fft, hop):
        chunk = mono[i : i + n_fft] * np.hanning(n_fft)
        spec = np.fft.rfft(chunk)
        spec *= eq_gain[: len(spec)]
        result[i : i + n_fft] += np.fft.irfft(spec)[:n_fft]

    # RMS-Matching
    src_rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
    target_rms = 10 ** (target.target_rms_db / 20)
    gain = target_rms / src_rms
    result = np.clip(result * gain, -1.0, 1.0)

    logger.info("ReferenceMatch: EQ+dynamic angewendet")
    return cast(np.ndarray, result.astype(np.float32))
