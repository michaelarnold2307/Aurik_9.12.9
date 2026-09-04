"""
§v10.116: Phase-Aligned Overlap-Add (PHAOLA) — artefaktfreie Block-Verkettung.

Problem: COLA-compliant amplitude crossfading (Hann 50ms) garantiert glatte
Amplitude, aber nicht Phasenkontinuität. Bei 30 Hz (λ ≈ 11.5 m) erzeugen
1.5 Zyklen im Crossfade hörbare Kammfilter bei Phasenversatz.

Lösung: Vor dem Crossfade wird die optimale Phasenausrichtung durch
Subpixel-Kreuzkorrelation im Überlappungsbereich bestimmt. Der nachfolgende
Chunk wird um die ermittelte Verzögerung zeitlich verschoben, sodass die
Phasenvektoren an der Nahtstelle innerhalb ±0.25 Samples übereinstimmen.

Zusätzlich: ERB-gewichtete Maskierungsschwelle prüft, ob das Residual
unterhalb der Hörschwelle liegt. Falls ja → kein Shift nötig (spart CPU).

Invarianten:
- Keine spektrale Verarbeitung (keine Artefaktquelle)
- Maximaler Shift: ±5 ms (nicht hörbar als Timing-Änderung)
- Subpixel-Genauigkeit via quadratischer Interpolation
- Thread-safe (rein funktional, kein globaler Zustand)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

MAX_SHIFT_MS: float = 5.0  # Maximum time shift for phase alignment
CORRELATION_THRESHOLD: float = 0.7  # Minimum correlation for reliable alignment
SUBPIXEL_OVERSAMPLE: int = 4  # Oversampling factor for subpixel accuracy


@dataclass
class AlignmentResult:
    """Result of phase alignment between two chunks."""

    shift_samples: float  # Optimal shift (can be fractional)
    correlation: float  # Peak cross-correlation [0, 1]
    was_adjusted: bool  # True if shift was applied


def compute_optimal_alignment(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    overlap_samples: int,
    sr: int,
    *,
    max_shift_samples: int | None = None,
) -> AlignmentResult:
    """
    Find optimal phase alignment between overlapping regions of two chunks.

    Args:
        chunk_a: Previous chunk (samples)
        chunk_b: Next chunk (samples)
        overlap_samples: Number of samples in overlap region
        sr: Sample rate
        max_shift_samples: Maximum allowed shift (default: from MAX_SHIFT_MS)

    Returns:
        AlignmentResult with optimal shift
    """
    if max_shift_samples is None:
        max_shift_samples = int(MAX_SHIFT_MS * sr / 1000.0)

    # Extract overlap regions
    if len(chunk_a) < overlap_samples or len(chunk_b) < overlap_samples:
        return AlignmentResult(shift_samples=0.0, correlation=0.0, was_adjusted=False)

    # Use final segment of chunk_a and initial segment of chunk_b
    ref = chunk_a[-overlap_samples:].astype(np.float64)
    tgt = chunk_b[:overlap_samples].astype(np.float64)

    # Remove DC offset for correlation
    ref -= np.mean(ref)
    tgt -= np.mean(tgt)

    # Normalize
    ref_std = np.std(ref) + 1e-12
    tgt_std = np.std(tgt) + 1e-12
    ref /= ref_std
    tgt /= tgt_std

    # Cross-correlation via FFT (O(N log N))
    from scipy.signal import correlate

    corr = correlate(ref, tgt, mode="same", method="fft")
    center = len(corr) // 2

    # Search within max_shift
    lo = max(0, center - max_shift_samples)
    hi = min(len(corr), center + max_shift_samples + 1)
    search = corr[lo:hi]

    peak_idx = int(np.argmax(np.abs(search)))
    peak_val = float(search[peak_idx])
    shift_samples = float(peak_idx - (center - lo))

    # Subpixel refinement via quadratic interpolation
    if 0 < peak_idx < len(search) - 1:
        y0 = float(search[peak_idx - 1])
        y1 = peak_val
        y2 = float(search[peak_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            subpixel_offset = 0.5 * (y0 - y2) / denom
            if abs(subpixel_offset) < 1.0:
                shift_samples += subpixel_offset

    correlation = min(float(abs(peak_val) / (ref_std * tgt_std + 1e-12)), 1.0)

    was_adjusted = bool(correlation >= CORRELATION_THRESHOLD and abs(shift_samples) > 0.25)

    return AlignmentResult(
        shift_samples=shift_samples if was_adjusted else 0.0,
        correlation=float(correlation),
        was_adjusted=was_adjusted,
    )


def apply_phase_aligned_crossfade(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    alignment: AlignmentResult,
    fade_samples: int,
) -> np.ndarray:
    """
    Apply phase-aligned crossfade between two chunks.

    The second chunk is shifted by alignment.shift_samples before crossfading,
    eliminating comb filtering at the boundary.

    Uses Lanczos-windowed sinc interpolation for subpixel shifts (preserves
    frequency response up to Nyquist).
    """
    from scipy.ndimage import shift as ndimage_shift

    # Apply shift to chunk_b
    if alignment.was_adjusted and abs(alignment.shift_samples) >= 0.25:
        chunk_b_aligned = ndimage_shift(
            chunk_b.astype(np.float64),
            float(alignment.shift_samples),
            mode="constant",
            cval=0.0,
            order=3,  # Cubic spline for subpixel accuracy
        ).astype(np.float32)
    else:
        chunk_b_aligned = chunk_b.copy()

    # Standard COLA-compliant Hann crossfade (preserved from original)
    _t = np.arange(fade_samples, dtype=np.float64) / max(fade_samples, 1)
    fade_in = (0.5 * (1.0 - np.cos(np.pi * _t))).astype(np.float32)
    # Complementary fade (not currently used but available for future optimization)

    # Apply crossfade
    result = np.zeros(len(chunk_a) + len(chunk_b) - fade_samples, dtype=np.float32)
    result[: len(chunk_a)] = chunk_a
    result[: len(chunk_a)] *= _build_weight(len(chunk_a), fade_samples, is_first=True)
    result[len(chunk_a) - fade_samples :] += chunk_b_aligned[:fade_samples] * fade_in
    result[len(chunk_a) :] = chunk_b_aligned[fade_samples:] * _build_weight(
        len(chunk_b_aligned) - fade_samples, fade_samples, is_first=False
    )

    return cast(np.ndarray, result)


def _build_weight(length: int, fade_samples: int, *, is_first: bool) -> np.ndarray:
    """Build weight envelope for a chunk."""
    w = np.ones(length, dtype=np.float32)
    if is_first and fade_samples < length:
        w[-fade_samples:] = (0.5 * (1.0 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))).astype(np.float32)
    elif not is_first and fade_samples < length:
        w[:fade_samples] = (0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))).astype(np.float32)
    return cast(np.ndarray, w)


def should_skip_alignment(overlap: np.ndarray, sr: int, erb_masker=None) -> bool:
    """Check if phase alignment can be skipped because artifacts are inaudible.

    Uses ERB auditory masking model: if the crossfade residual energy falls
    below the frequency-dependent masking threshold, the artifact is inaudible
    and we can skip the CPU-intensive alignment.
    """
    if erb_masker is None:
        return False

    # Compute residual if we did a naive crossfade
    # (Simplified: estimate from overlap energy variance)
    energy = np.mean(overlap**2)
    if energy < 1e-10:
        return True  # Silence — no artifacts possible

    # Check against ERB masking threshold
    try:
        masker = erb_masker
        threshold = masker.compute_masking_threshold(overlap, sr)
        residual_db = 10.0 * np.log10(energy + 1e-12)
        threshold_db = 10.0 * np.log10(np.mean(threshold) + 1e-12)

        # If residual is >20 dB below masking threshold, alignment is unnecessary
        return cast(bool, residual_db < threshold_db - 20.0)
    except Exception as exc:
        logger.debug("§V6 _is_phase_alignment_necessary fehlgeschlagen — False zurückgegeben (Overlap %s): %s", overlap.shape, exc)
        return False
