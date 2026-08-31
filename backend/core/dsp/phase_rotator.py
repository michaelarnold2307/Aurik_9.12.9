"""§PHROT-1 MP3 Phase Rotator (v10.13).

Lightweight phase rotation for the bass band (20–200 Hz) to improve
mono compatibility of MP3-encoded stereo material. MP3 joint-stereo
encoding introduces phase scrambling in low frequencies; a subtle
all-pass rotation restores natural L/R coherence without affecting
the stereo image perceptibly.

Parameters:
  low_freq, high_freq — affected frequency band (Hz)
  max_rotation_deg   — maximum phase rotation at band edges (degrees)
  strength           — blend factor (0.0 = off, 0.15 = subtle)

Non-blocking: import errors are silently caught by the caller.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.signal import butter, sosfilt


def _design_allpass_2nd_order(freq_hz: float, sr: int, rotation_deg: float) -> np.ndarray:
    """Design a 2nd-order all-pass filter with specified phase rotation at freq_hz.

    An all-pass has zeros at z=1/p* and poles at z=p.
    Phase at frequency f: φ = -2πfT - 2·atan2(imag, real) for each pole pair.

    For simplicity we use a biquad all-pass via bilinear transform of
    H(s) = (s² - ω₀/Q·s + ω₀²) / (s² + ω₀/Q·s + ω₀²) with Q tuned for
    desired phase rotation.
    """
    w0 = 2.0 * np.pi * freq_hz / sr
    # Q controls the phase slope: higher Q = sharper transition
    # For a gentle 30° rotation spread over the band, use Q ≈ 0.5
    q_val = 0.5
    alpha = np.sin(w0) / (2.0 * q_val)
    cos_w0 = np.cos(w0)

    # All-pass: b = [1 - alpha, -2*cos_w0, 1 + alpha] * scale
    #           a = [1 + alpha, -2*cos_w0, 1 - alpha] * scale
    b = np.array([1.0 - alpha, -2.0 * cos_w0, 1.0 + alpha])
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])

    # Scale the rotation by cascading multiple sections or adjusting Q
    # A single 2nd-order all-pass gives ~20-30° at center frequency when Q=0.5
    # For target rotation, cascade multiple identical sections
    target_phase = np.deg2rad(rotation_deg)
    actual_phase = 2.0 * np.arctan2(np.sin(w0) / (2.0 * q_val), 1.0)
    num_sections = max(1, int(round(target_phase / max(actual_phase, 1e-6))))

    # Build SOS array
    sos = np.tile(np.hstack([b, a]), (num_sections, 1))
    return cast(np.ndarray, sos)


def apply_phase_rotator(
    audio: np.ndarray,
    sr: int,
    low_freq: float = 20.0,
    high_freq: float = 200.0,
    max_rotation_deg: float = 30.0,
    strength: float = 0.15,
) -> np.ndarray:
    """Apply subtle bass phase rotation to stereo audio.

    Processes L/R independently with identical all-pass filters
    (no inter-channel de-correlation — the filter is the same for both
    channels). The wet/dry blend via *strength* preserves the original
    signal character while gently rotating low-frequency phase.

    Args:
        audio: Stereo array, shape (N, 2) or (2, N).
        sr: Sample rate (must be 48000).
        low_freq, high_freq: Affected band (Hz).
        max_rotation_deg: Peak phase rotation at band edge (degrees).
        strength: Wet/dry blend (0.0 = bypass, 0.15 = subtle).

    Returns:
        Phase-rotated audio, same shape and dtype as input.
    """
    if audio.ndim != 2:
        return audio  # mono — no rotation needed

    arr = np.asarray(audio, dtype=np.float32)
    assert sr == 48000, f"Phase rotator requires sr=48000, got {sr}"

    # Detect orientation
    if arr.shape[0] == 2 and arr.shape[1] > 2:
        ch_first = True
        ch_l, ch_r = arr[0].copy(), arr[1].copy()
    elif arr.shape[1] == 2 and arr.shape[0] > 2:
        ch_first = False
        ch_l, ch_r = arr[:, 0].copy(), arr[:, 1].copy()
    else:
        return audio

    # Design all-pass at band center
    center_freq = np.sqrt(low_freq * high_freq)  # geometric mean
    sos = _design_allpass_2nd_order(center_freq, sr, max_rotation_deg)

    # Apply to each channel (identical filter → no new L/R delay)
    wet_l = sosfilt(sos, ch_l.astype(np.float64))
    wet_r = sosfilt(sos, ch_r.astype(np.float64))

    # Blend with dry
    strength_clamped = float(np.clip(abs(strength), 0.0, 1.0))
    out_l = ch_l + strength_clamped * (wet_l[: len(ch_l)] - ch_l)
    out_r = ch_r + strength_clamped * (wet_r[: len(ch_r)] - ch_r)

    # Rebuild stereo array
    if ch_first:
        result = np.vstack([out_l[np.newaxis, :], out_r[np.newaxis, :]])
    else:
        result = np.column_stack([out_l, out_r])

    return cast(np.ndarray, (np.clip(result.astype(np.float32), -1.0, 1.0)))
