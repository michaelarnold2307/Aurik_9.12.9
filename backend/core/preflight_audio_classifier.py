"""§v10.17 PreflightAudioClassifier — erkennt Silence/Mono/Ultra-Short VOR der Pipeline.

Vermeidet dass 64 Phasen auf 1-Sekunden-Stille oder Mono-Dateien laufen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_MIN_DURATION_S: float = 3.0  # < 3s → "ultra_short"
_SILENCE_RMS_THRESHOLD: float = 1e-5  # RMS unter diesem Wert = Stille
_MONO_CORR_THRESHOLD: float = 0.999  # L/R-Korrelation > 0.999 = Mono


@dataclass
class AudioClass:
    is_silence: bool = False
    is_mono: bool = False
    is_ultra_short: bool = False
    is_stereo: bool = True
    duration_s: float = 0.0
    rms: float = 0.0
    peak_db: float = -99.0
    lr_correlation: float = 1.0
    should_skip_phases: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.should_skip_phases is None:
            self.should_skip_phases = []


def classify_audio(audio: np.ndarray, sr: int) -> AudioClass:
    """Klassifiziert das Audio VOR der Pipeline."""
    try:
        arr = np.asarray(audio, dtype=np.float64)
        n = len(arr) if arr.ndim == 1 else (arr.shape[1] if arr.shape[0] <= 2 else arr.shape[0])
        dur = n / sr

        result = AudioClass(duration_s=dur)

        # Ultra-short
        if dur < _MIN_DURATION_S:
            result.is_ultra_short = True
            result.should_skip_phases = ["ALL_STEREO_PHASES", "ALL_DYNAMICS_PHASES"]

        # RMS / Silence
        mono = arr.mean(axis=0) if (arr.ndim > 1 and arr.shape[0] <= 2) else (arr.mean(axis=1) if arr.ndim > 1 else arr)
        rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
        result.rms = rms
        result.peak_db = 20.0 * np.log10(float(np.max(np.abs(mono))) + 1e-12)

        if rms < _SILENCE_RMS_THRESHOLD:
            result.is_silence = True
            result.should_skip_phases = ["ALL_PHASES"]  # Nichts zu tun

        # Mono/Stereo
        if arr.ndim == 2:
            ch0 = arr[0] if arr.shape[0] <= 2 else arr[:, 0]
            ch1 = arr[1] if arr.shape[0] <= 2 else arr[:, 1]
            seg = min(len(ch0), len(ch1), sr * 5)
            corr = float(np.corrcoef(ch0[:seg], ch1[:seg])[0, 1])
            result.lr_correlation = corr if np.isfinite(corr) else 1.0
            if result.lr_correlation > _MONO_CORR_THRESHOLD:
                result.is_mono = True
                result.is_stereo = False
                result.should_skip_phases = ["ALL_STEREO_PHASES"]  # Stereo-Breite etc. sinnlos
        else:
            result.is_stereo = False
            result.is_mono = True
            result.should_skip_phases = ["ALL_STEREO_PHASES"]

        return result

    except Exception as e:
        logger.debug("§V6 PreflightAudioClassifier fehlgeschlagen — leeres AudioClass zurückgegeben: %s", e)
        return AudioClass()
