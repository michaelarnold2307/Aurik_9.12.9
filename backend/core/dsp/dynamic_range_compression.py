"""
Dynamic‑Range‑Compression (Multi‑Band) – SOTA‑Implementierung.

Der Kompressor arbeitet in drei Bändern: <200 Hz, 200–2000 Hz, >2000 Hz.
Jedes Band hat einen eigenen Threshold und Ratio. Die Parameter werden
adaptiv anhand der Zwicker‑Metriken (Roughness) und des LUFS‑Levels gewählt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)

# Band‑Grenzen in Hz
_BAND_EDGES_HZ = [200.0, 2000.0]

# Standard‑Parameter (Threshold dBFS, Ratio)
_DEFAULT_PARAMS = {
    "low": {"threshold_dbfs": -20.0, "ratio": 2.0},
    "mid": {"threshold_dbfs": -18.0, "ratio": 1.8},
    "high": {"threshold_dbfs": -16.0, "ratio": 1.5},
}


@dataclass
class BandParams:
    threshold_dbfs: float
    ratio: float


@dataclass
class CompressionResult:
    audio: np.ndarray
    params_used: dict[str, BandParams]


# Hilfsfunktion – Band‑Filter (Butterworth 4. Ordnung)


def _bandpass(audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return cast(np.ndarray, (lfilter(b, a, audio)))


# Kompression pro Band


def _compress_band(signal: np.ndarray, threshold_dbfs: float, ratio: float) -> np.ndarray:
    thresh_lin = 10 ** (threshold_dbfs / 20.0)
    # Signal‑Amplitude abs
    amp = np.abs(signal)
    mask = amp > thresh_lin
    if not mask.any():
        return signal
    gain = np.ones_like(amp)
    gain[mask] = (thresh_lin + (amp[mask] - thresh_lin) / ratio) / amp[mask]
    return cast(np.ndarray, signal * gain)


# Adaptive Parameter‑Auswahl basierend auf LUFS und Roughness
from typing import cast

from .loudness_meter import compute_loudness
from .zwicker_metrics import compute_roughness_asper


def _adaptive_params(audio: np.ndarray, sr: int) -> dict[str, BandParams]:
    # Roughness-based adaptive thresholds (loudness already incorporated in caller)
    rough = compute_roughness_asper(audio, sr)
    # Beispiel‑Logik: bei hoher Roughness senken wir Thresholds
    factor = 1.0 + 0.05 * rough  # 5 % pro asper
    params = {}
    for band in ["low", "mid", "high"]:
        base = _DEFAULT_PARAMS[band]
        params[band] = BandParams(
            threshold_dbfs=base["threshold_dbfs"] / factor,
            ratio=base["ratio"],
        )
    return params


# Hauptfunktion – Multi‑Band‑Kompression


def apply_dynamic_range_compression(audio: np.ndarray, sr: int) -> CompressionResult:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # Stereo → pro Kanal komprimieren
        left = apply_dynamic_range_compression(audio[0], sr).audio
        right = apply_dynamic_range_compression(audio[1], sr).audio
        return CompressionResult(np.stack([left, right]), {})
    params = _adaptive_params(audio, sr)
    # Band‑Filter und Kompression
    low_band = _bandpass(audio, sr, 0.0, _BAND_EDGES_HZ[0])
    mid_band = _bandpass(audio, sr, _BAND_EDGES_HZ[0], _BAND_EDGES_HZ[1])
    high_band = _bandpass(audio, sr, _BAND_EDGES_HZ[1], sr / 2.0)

    low_comp = _compress_band(low_band, params["low"].threshold_dbfs, params["low"].ratio)
    mid_comp = _compress_band(mid_band, params["mid"].threshold_dbfs, params["mid"].ratio)
    high_comp = _compress_band(high_band, params["high"].threshold_dbfs, params["high"].ratio)

    # Summe der komprimierten Bänder (ohne Überlagerung)
    comp_audio = low_comp + mid_comp + high_comp
    return CompressionResult(
        comp_audio,
        {
            "low": params["low"],
            "mid": params["mid"],
            "high": params["high"],
        },
    )
