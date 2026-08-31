"""
Section Strength Envelope — Continuous, psychoacoustically transparent per-section scaling.

Converts discrete SectionTarget objects into a smooth, sample-accurate strength
envelope that phases can multiply with their base processing strength.  Cosine
crossfades between sections ensure no audible transitions.

Architecture:
    SectionTarget[] → build_strength_envelope() → ndarray[float32] (len = n_samples)
    Phases read: strength = base_strength * envelope[frame_start:frame_end].mean()

Key invariants:
    - Max change rate: 1 dB / 100 ms (below human perception threshold)
    - Cosine crossfade: 200 ms at all section boundaries
    - Frisson zones: strength clamped to ≤ 0.30 (preserve goosebump moments)
    - The envelope is precomputed ONCE and passed via restoration_context
    - Phases do NOT recompute — they simply read the precomputed array

Author: Aurik v10.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from backend.core.section_goal_adapter import SectionTarget  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# ── Psychoacoustic Constants ──────────────────────────────────────────
_CROSSFADE_S = 0.200  # 200 ms cosine crossfade
_MAX_DB_PER_100MS = 1.0  # max audible change rate (Zwicker & Fastl 1999)
_FRISSON_STRENGTH_CAP = 0.30  # preserve emotional peaks
_DEFAULT_STRENGTH = 0.75  # fallback when no section data


def build_strength_envelope(
    section_targets: list[SectionTarget],
    n_samples: int,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Build a continuous strength envelope from discrete section targets.

    Args:
        section_targets: List of SectionTarget from SectionGoalAdapter.
        n_samples:       Total number of audio samples.
        sample_rate:     Sample rate in Hz (default 48000).

    Returns:
        Float32 array of shape (n_samples,) in range [0.0, 1.0].
        All samples initialized to _DEFAULT_STRENGTH.
    """
    envelope = np.full(n_samples, _DEFAULT_STRENGTH, dtype=np.float32)

    if not section_targets:
        logger.debug("SectionStrengthEnvelope: no targets → uniform %.2f", _DEFAULT_STRENGTH)
        return cast(np.ndarray, envelope)

    # Sort by start time
    sorted_targets = sorted(section_targets, key=lambda s: s.start_s)

    # ── Build envelope: set per-section strength with cosine crossfades ──
    crossfade_samples = int(_CROSSFADE_S * sample_rate)
    max_step_samples = int(0.100 * sample_rate)  # 100ms for rate limiting

    for i, target in enumerate(sorted_targets):
        t_start = int(target.start_s * sample_rate)
        t_end = int(target.end_s * sample_rate)
        t_start = max(0, min(t_start, n_samples - 1))
        t_end = max(t_start + 1, min(t_end, n_samples))

        # ── Compute section strength from target fields ──
        # nr_strength_scale: base noise reduction intensity (0.5–2.0)
        # vq_weight: vocal quality emphasis (0.5–2.0)
        # frisson_protection: if True, cap at _FRISSON_STRENGTH_CAP
        section_strength = float(np.clip(target.nr_strength_scale * target.vq_weight, 0.30, 1.50))
        if target.frisson_protection:
            section_strength = min(section_strength, _FRISSON_STRENGTH_CAP)

        # Clamp to valid range
        section_strength = float(np.clip(section_strength, 0.10, 1.50))

        # ── Apply with cosine crossfade at boundaries ──
        # Leading crossfade (unless first segment)
        cf_start = t_start
        if i > 0:
            cf_start = max(0, t_start - crossfade_samples // 2)

        # Trailing crossfade (unless last segment)
        cf_end = t_end
        if i < len(sorted_targets) - 1:
            cf_end = min(n_samples, t_end + crossfade_samples // 2)

        # Build cosine ramp
        seg_len = cf_end - cf_start
        if seg_len <= 0:
            continue

        np.arange(cf_start, cf_end)
        seg_values = np.full(seg_len, section_strength, dtype=np.float32)

        # Cosine fade-in at start (first 200ms of this section's influence)
        fade_in_len = min(crossfade_samples, seg_len // 2)
        if fade_in_len > 1 and i > 0:
            fade_in = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_in_len) / fade_in_len)
            seg_values[:fade_in_len] = (
                envelope[cf_start : cf_start + fade_in_len] * (1.0 - fade_in) + section_strength * fade_in
            ).astype(np.float32)

        # Cosine fade-out at end (last 200ms of this section's influence)
        fade_out_len = min(crossfade_samples, seg_len - fade_in_len)
        if fade_out_len > 1 and i < len(sorted_targets) - 1:
            fade_out_start = seg_len - fade_out_len
            fade_out = 0.5 + 0.5 * np.cos(np.pi * np.arange(fade_out_len) / fade_out_len)
            seg_values[fade_out_start:] = (section_strength * fade_out + _DEFAULT_STRENGTH * (1.0 - fade_out)).astype(
                np.float32
            )

        # ── Rate-limit: ensure no step > _MAX_DB_PER_100MS ──
        if i > 0:
            prev_end = int(sorted_targets[i - 1].end_s * sample_rate)
            transition_zone = slice(max(0, prev_end - max_step_samples), cf_start)
            if transition_zone.stop > transition_zone.start:
                prev_val = np.mean(envelope[transition_zone])
                target_db = 20.0 * np.log10(max(section_strength, 1e-6))
                prev_db = 20.0 * np.log10(max(prev_val, 1e-6))
                db_diff = abs(target_db - prev_db)
                steps_needed = int(db_diff / _MAX_DB_PER_100MS)
                if steps_needed > 1:
                    # Extend the crossfade to respect rate limit
                    extended_fade = min(crossfade_samples * steps_needed, seg_len)
                    if extended_fade > fade_in_len:
                        ext_fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(extended_fade) / extended_fade)
                        seg_start = cf_start
                        seg_values[:extended_fade] = (
                            envelope[seg_start : seg_start + extended_fade] * (1.0 - ext_fade)
                            + section_strength * ext_fade
                        ).astype(np.float32)

        envelope[cf_start:cf_end] = seg_values

    # Final clamp
    envelope = np.clip(envelope, 0.10, 1.50).astype(np.float32)

    logger.debug(
        "SectionStrengthEnvelope: %d sections → %d samples, range [%.2f, %.2f], mean=%.3f",
        len(section_targets),
        n_samples,
        float(np.min(envelope)),
        float(np.max(envelope)),
        float(np.mean(envelope)),
    )

    return cast(np.ndarray, envelope)


def get_section_strength_at(
    envelope: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> float:
    """Read the envelope strength for a specific sample range.

    Phases call this for each processing frame:
        strength = base_strength * get_section_strength_at(envelope, frame_start, frame_end)

    Args:
        envelope:     Precomputed envelope from build_strength_envelope().
        start_sample: Start sample index (inclusive).
        end_sample:   End sample index (exclusive).

    Returns:
        Mean envelope value in [0.10, 1.50] for the given range.
    """
    if envelope is None or len(envelope) == 0:
        return _DEFAULT_STRENGTH
    start = max(0, min(start_sample, len(envelope) - 1))
    end = max(start + 1, min(end_sample, len(envelope)))
    return float(np.mean(envelope[start:end]))
