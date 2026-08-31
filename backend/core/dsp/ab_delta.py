"""
A/B Delta Computation — Difference signal for instant quality comparison.

§14.9: Computes the difference between original and restored audio,
enabling "delta mode" listening: hear exactly what Aurik changed.

The delta signal is loudness-normalized so it's audible even when
the changes are subtle.

Author: Aurik 10.0.1
"""

from typing import cast

import numpy as np


def compute_ab_delta(original: np.ndarray, restored: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute the difference signal between original and restored audio.

    Args:
        original: Original audio (pre-restoration), same shape as restored.
        restored: Restored audio, same shape as original.
        normalize: If True, normalize delta to -6 dBFS peak for audibility.

    Returns:
        Delta signal: restored - original. Positive = added, negative = removed.
    """
    # Align lengths
    min_len = min(
        original.shape[-1] if original.ndim > 1 else len(original),
        restored.shape[-1] if restored.ndim > 1 else len(restored),
    )

    if original.ndim > 1:
        orig = original[..., :min_len]
    else:
        orig = original[:min_len]

    if restored.ndim > 1:
        rest = restored[..., :min_len]
    else:
        rest = restored[:min_len]

    delta = rest.astype(np.float64) - orig.astype(np.float64)

    if normalize:
        peak = float(np.max(np.abs(delta))) + 1e-12
        target_peak = 10 ** (-6.0 / 20.0)  # -6 dBFS
        delta = delta * (target_peak / peak)

    return cast(np.ndarray, (np.clip(delta, -1.0, 1.0).astype(np.float32)))
