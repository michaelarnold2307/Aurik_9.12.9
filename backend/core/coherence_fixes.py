"""CoherenceFixes — §STRATEGIC: Kohärenz-Fixes für alle Risiko-Stellen.

PMGG Wet/Dry, STCG, Dropout-Repair, Noise Gate:
  Jede Stelle bekommt Cross-Fade/Smoothing wo nötig.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


def crossfade_blend(original: np.ndarray, processed: np.ndarray, sr: int, blend_s: float = 0.05) -> np.ndarray:
    """Wet/Dry-Blending mit Cross-Fade an den Übergängen.

    Verhindert hörbare Sprünge wenn PMGG die Stärke ändert.
    """
    blend_samples = int(blend_s * sr)
    if blend_samples < 8 or len(original) < blend_samples * 2:
        return processed

    mono_orig = np.mean(original, axis=-1) if original.ndim > 1 else np.asarray(original, dtype=np.float32)
    mono_proc = np.mean(processed, axis=-1) if processed.ndim > 1 else np.asarray(processed, dtype=np.float32)

    # Fade an den Rändern: processed → original → processed
    fade_in = np.linspace(0, 1, blend_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, blend_samples, dtype=np.float32)

    result = mono_proc.copy()
    result[:blend_samples] = mono_orig[:blend_samples] * fade_out + mono_proc[:blend_samples] * fade_in
    result[-blend_samples:] = mono_orig[-blend_samples:] * fade_in + mono_proc[-blend_samples:] * fade_out

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))


def smooth_stereo_correction(audio: np.ndarray, delay_samples: int, sr: int, smooth_s: float = 0.10) -> np.ndarray:
    """STCG: Zeitliche Glättung der Kanal-Korrektur.

    Statt abrupter sample-genauer Shifts → graduelle Verschiebung.
    """
    if abs(delay_samples) < 1 or audio.ndim < 2:
        return audio

    smooth_frames = int(smooth_s * sr)
    if smooth_frames < 16 or audio.shape[0] < smooth_frames * 2:
        return audio  # Audio too short for smooth transition

    l = audio[:, 0].copy()
    r = audio[:, 1].copy()

    # Graduelle Verschiebung über Smooth-Fenster
    ramp = np.linspace(0, 1, smooth_frames, dtype=np.float32)
    corrected_r = np.roll(r, delay_samples)

    # Nur im Smooth-Fenster blenden
    r[:smooth_frames] = r[:smooth_frames] * (1 - ramp) + corrected_r[:smooth_frames] * ramp
    r[-smooth_frames:] = r[-smooth_frames:] * ramp + corrected_r[-smooth_frames:] * (1 - ramp)

    result = np.stack([l, r], axis=-1).astype(np.float32)
    logger.debug("STCG smooth: delay=%d samples → %dms fade", delay_samples, int(smooth_s * 1000))
    return result  # type: ignore[no-any-return]


def dropout_fade(
    original: np.ndarray, repaired: np.ndarray, dropout_start: int, dropout_end: int, sr: int
) -> np.ndarray:
    """Dropout-Repair: Ein-/Ausblendung an reparierten Stellen."""
    mono = np.mean(original, axis=-1) if original.ndim > 1 else np.asarray(original, dtype=np.float32)
    rep = np.mean(repaired, axis=-1) if repaired.ndim > 1 else np.asarray(repaired, dtype=np.float32)

    fade_len = min(int(0.005 * sr), (dropout_end - dropout_start) // 4, 256)
    if fade_len < 4:
        return repaired

    fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)

    result = mono.copy()
    result[dropout_start:dropout_end] = rep[dropout_start:dropout_end]

    # Einblendung am Anfang des Dropouts
    s0 = max(0, dropout_start - fade_len)
    s1 = min(len(result), dropout_start + fade_len)
    if s1 > s0:
        result[s0:dropout_start] = mono[s0:dropout_start]
        blend_len = min(fade_len, dropout_end - dropout_start)
        for i in range(blend_len):
            idx = dropout_start + i
            if idx < len(result):
                w = fade_in[i]
                result[idx] = mono[idx] * (1 - w) + rep[idx] * w

    # Ausblendung am Ende des Dropouts
    e0 = max(0, dropout_end - fade_len)
    e1 = min(len(result), dropout_end + fade_len)
    if e1 > e0:
        for i in range(fade_len):
            idx = dropout_end - fade_len + i
            if idx >= 0 and idx < len(result):
                w = fade_out[i]
                result[idx] = rep[idx] * (1 - w) + mono[idx] * w

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))


def noise_gate_envelope(gate_mask: np.ndarray, sr: int, attack_ms: float = 2.0, release_ms: float = 50.0) -> np.ndarray:
    """Noise Gate: Attack/Release-Hüllkurve statt hartem Schalten."""
    attack_samples = int(attack_ms / 1000 * sr)
    release_samples = int(release_ms / 1000 * sr)

    if attack_samples < 1 or release_samples < 1:
        return gate_mask

    smoothed = gate_mask.copy().astype(np.float32)
    n = len(gate_mask)

    for i in range(1, n):
        if gate_mask[i] > smoothed[i - 1]:
            # Attack: schnelles Öffnen
            alpha = np.exp(-1.0 / max(attack_samples, 1))
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * gate_mask[i]
        else:
            # Release: langsames Schließen
            alpha = np.exp(-1.0 / max(release_samples, 1))
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * gate_mask[i]

    logger.debug("NoiseGate envelope: attack=%dms release=%dms", attack_ms, release_ms)
    return cast(np.ndarray, smoothed)
