"""PlatformExport — §INCREMENTAL #3: Plattform-Normalisierter Export.

Ein Aufruf: export_for(audio, sr, "spotify") → -14 LUFS, -1 dBTP.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

PLATFORMS = {
    "spotify": {"target_lufs": -14.0, "max_true_peak": -1.0, "name": "Spotify"},
    "youtube": {"target_lufs": -13.0, "max_true_peak": -1.0, "name": "YouTube"},
    "apple": {"target_lufs": -16.0, "max_true_peak": -1.0, "name": "Apple Music"},
    "cd": {"target_lufs": -9.0, "max_true_peak": -0.3, "name": "CD Master"},
    "broadcast": {"target_lufs": -23.0, "max_true_peak": -2.0, "name": "EBU R128 Broadcast"},
    "soundcloud": {"target_lufs": -12.0, "max_true_peak": -1.0, "name": "SoundCloud"},
    "tidal": {"target_lufs": -14.0, "max_true_peak": -1.0, "name": "TIDAL"},
    "deezer": {"target_lufs": -14.0, "max_true_peak": -1.0, "name": "Deezer"},
}


def export_for(audio: np.ndarray, sr: int, platform: str = "spotify") -> np.ndarray:
    preset = PLATFORMS.get(platform.lower(), PLATFORMS["spotify"])
    target_lufs = float(preset["target_lufs"])  # type: ignore[arg-type]
    max_tp = float(preset["max_true_peak"])  # type: ignore[arg-type]

    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / rms
    gain = float(np.clip(gain, 0.05, 20.0))

    result = np.clip(audio * gain, -1.0, 1.0)

    # True-Peak Limiter (simple)
    peak = float(np.max(np.abs(result)))
    if 20 * np.log10(max(peak, 1e-10)) > max_tp:
        ceiling = 10 ** (max_tp / 20)
        result = np.clip(result, -ceiling, ceiling)

    logger.info("PlatformExport [%s]: LUFS target=%.0f, gain=%.1fdB", preset["name"], target_lufs, 20 * np.log10(gain))
    return cast(np.ndarray, result.astype(np.float32))


def list_platforms() -> list[dict[str, Any]]:
    return [{"id": k, **v} for k, v in PLATFORMS.items()]
