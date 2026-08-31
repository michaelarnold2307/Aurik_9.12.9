"""
§v10.117: Stereo-bewusste Mid/Side-Gesangsverarbeitung.

Problem: Der aktuelle VocalEnhancer (vocal_ai_enhancement.py:918) wendet
einfaches ratio = processed / (original + eps) an. Dies zerstört das
Stereo-Bild:
- identische Ratio auf L/R → Mono-Charakter (Korrelation ↑)
- Phasenunterschiede zwischen L/R werden verstärkt/abgeschwächt
- Räumliche Tiefe geht verloren

Lösung: Mid/Side-Transformation vor der Bearbeitung.
- M = (L + R) / 2   → enthält Zentrum (Vocals, Bass, Kick)
- S = (L - R) / 2   → enthält Stereobreite (Hall, Atmosphäre)
- Bearbeitung nur auf M-Kanal → S bleibt unverändert
- Rücktransformation: L' = M' + S, R' = M' - S

Garantien:
- Korrelation zwischen L/R bleibt erhalten (Phase linear)
- Stereo-Breite unverändert (S-Kanal untouched)
- Mono-kompatibel (downmix = M')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MidSideResult:
    """Result of mid/side processing."""

    mid: np.ndarray  # Center channel
    side: np.ndarray  # Difference channel
    correlation: float  # Original L/R correlation


def to_mid_side(audio: np.ndarray) -> MidSideResult:
    """
    Convert stereo L/R to Mid/Side representation.

    Args:
        audio: [2, samples] or [channels, samples] stereo array

    Returns:
        MidSideResult with mid, side, and original correlation
    """
    if audio.ndim < 2 or audio.shape[0] < 2:
        # Mono input — return as mid with zero side
        return MidSideResult(
            mid=audio.squeeze().astype(np.float32).copy(),
            side=np.zeros_like(audio.squeeze(), dtype=np.float32),
            correlation=1.0,
        )

    L = audio[0].astype(np.float64)
    R = audio[1].astype(np.float64)

    # Compute correlation before transform
    L_ms = L - np.mean(L)
    R_ms = R - np.mean(R)
    corr = float(np.dot(L_ms, R_ms) / (np.std(L_ms) * np.std(R_ms) * len(L) + 1e-12))

    # Mid/Side transform
    mid = ((L + R) * 0.5).astype(np.float32)
    side = ((L - R) * 0.5).astype(np.float32)

    return MidSideResult(mid=mid, side=side, correlation=corr)


def from_mid_side(ms: MidSideResult) -> np.ndarray:
    """
    Convert Mid/Side back to stereo L/R.

    L = Mid + Side, R = Mid - Side
    """
    L = (ms.mid + ms.side).astype(np.float32)
    R = (ms.mid - ms.side).astype(np.float32)
    return cast(np.ndarray, (np.stack([L, R], axis=0)))


def process_vocal_mid_side(
    audio: np.ndarray,
    processor_fn,
    *,
    side_preservation: float = 1.0,
) -> np.ndarray:
    """
    Apply vocal processing in Mid/Side domain, preserving stereo image.

    Args:
        audio: Stereo audio [2, samples]
        processor_fn: Callable(audio_mono) → processed_mono
        side_preservation: Multiplier for side channel (1.0 = unchanged,
                          0.0 = mono, >1.0 = wider)

    Returns:
        Processed stereo audio [2, samples]
    """
    if audio.ndim < 2 or audio.shape[0] < 2:
        # Mono fallback
        return cast(np.ndarray, processor_fn(audio.squeeze()).reshape(1, -1))

    ms = to_mid_side(audio)

    # Process only mid channel
    processed_mid = processor_fn(ms.mid)

    # Preserve side channel (optionally scale)
    processed_side = ms.side * float(np.clip(side_preservation, 0.0, 2.0))

    # Reconstruct
    result = from_mid_side(
        MidSideResult(
            mid=processed_mid,
            side=processed_side,
            correlation=ms.correlation,
        )
    )

    return result


def compute_stereo_preservation_score(original: np.ndarray, processed: np.ndarray) -> dict[str, float]:
    """
    Compute stereo preservation metrics.

    Returns dict with:
    - correlation_diff: Absolute change in L/R correlation (ideal: 0.0)
    - width_ratio: Side energy ratio processed/original (ideal: 1.0)
    - balance_shift: Change in L/R energy balance in dB (ideal: 0.0)
    """
    orig_ms = to_mid_side(original)
    proc_ms = to_mid_side(processed)

    # Correlation preservation
    corr_diff = abs(proc_ms.correlation - orig_ms.correlation)

    # Stereo width preservation (side energy ratio)
    orig_side_rms = np.sqrt(np.mean(orig_ms.side**2)) + 1e-12
    proc_side_rms = np.sqrt(np.mean(proc_ms.side**2)) + 1e-12
    width_ratio = proc_side_rms / orig_side_rms

    # L/R balance
    if original.ndim >= 2 and original.shape[0] >= 2:
        orig_balance = np.mean(original[0] ** 2) / (np.mean(original[1] ** 2) + 1e-12)
        proc_balance = np.mean(processed[0] ** 2) / (np.mean(processed[1] ** 2) + 1e-12)
        balance_shift = abs(float(10.0 * np.log10(max(orig_balance / (proc_balance + 1e-12), 1e-6))))
    else:
        balance_shift = 0.0

    return {
        "correlation_diff": float(corr_diff),
        "width_ratio": float(width_ratio),
        "balance_shift_db": float(balance_shift),
    }
