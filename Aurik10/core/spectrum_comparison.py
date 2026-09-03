"""Spektrum-Vergleich — Vorher/Nachher-Spektrogramm-Daten. Spec v10.206 §4.

Berechnet Spektrogramm-Daten für GUI-Visualisierung:
- Vorher-Spektrogramm (Original)
- Nachher-Spektrogramm (Restauriert)
- Delta-Spektrogramm (Differenz)
- Frequenzgang-Differenz (gemittelt über Zeit)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_spectrum_comparison(
    original: np.ndarray,
    restored: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:
    """Berechnet Spektrum-Vergleichsdaten für GUI.

    Args:
        original: Original-Audio (float32, mono oder stereo)
        restored: Restauriertes Audio
        sample_rate: Sample-Rate
        n_fft: FFT-Größe
        hop_length: Hop-Länge

    Returns:
        Dict mit spectrogram_before, spectrogram_after, spectrogram_delta,
        frequency_response_diff, metadata.
    """

    def to_mono(audio):
        arr = np.asarray(audio, dtype=np.float32)
        return arr.mean(axis=1) if arr.ndim == 2 else arr

    mono_orig = to_mono(original)
    mono_rest = to_mono(restored)
    min_len = min(len(mono_orig), len(mono_rest))
    mono_orig = mono_orig[:min_len]
    mono_rest = mono_rest[:min_len]

    # Spektrogramme
    try:
        from scipy.signal import stft

        f_orig, t_orig, Z_orig = stft(mono_orig, fs=sample_rate, nperseg=n_fft, noverlap=min(n_fft - hop_length, max(0, n_fft - 1)))  # §v10.103 noverlap-Clamp
        _, _, Z_rest = stft(mono_rest, fs=sample_rate, nperseg=n_fft, noverlap=min(n_fft - hop_length, max(0, n_fft - 1)))  # §v10.103 noverlap-Clamp
    except ImportError:
        # Fallback: numpy FFT
        n_frames = (min_len - n_fft) // hop_length + 1
        f_orig = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        Z_orig = np.zeros((len(f_orig), n_frames), dtype=np.complex64)
        Z_rest = np.zeros_like(Z_orig)
        for i in range(n_frames):
            start = i * hop_length
            Z_orig[:, i] = np.fft.rfft(mono_orig[start : start + n_fft] * np.hanning(n_fft))
            Z_rest[:, i] = np.fft.rfft(mono_rest[start : start + n_fft] * np.hanning(n_fft))

    mag_orig = 20.0 * np.log10(np.abs(Z_orig) + 1e-12)
    mag_rest = 20.0 * np.log10(np.abs(Z_rest) + 1e-12)
    mag_delta = mag_rest - mag_orig

    # Frequenzgang-Differenz (gemittelt über Zeit)
    freq_diff = np.mean(mag_delta, axis=1)

    return {
        "sample_rate": sample_rate,
        "duration_s": min_len / sample_rate,
        "frequencies_hz": f_orig[: len(f_orig) // 2].tolist(),
        "times_s": (np.arange(mag_orig.shape[1]) * hop_length / sample_rate).tolist(),
        "spectrogram_before_db": _downsample_2d(mag_orig[: len(f_orig) // 2]).tolist(),
        "spectrogram_after_db": _downsample_2d(mag_rest[: len(f_orig) // 2]).tolist(),
        "spectrogram_delta_db": _downsample_2d(mag_delta[: len(f_orig) // 2]).tolist(),
        "frequency_response_diff_db": freq_diff[: len(f_orig) // 2].tolist(),
        "max_delta_db": float(np.max(np.abs(mag_delta))),
        "mean_abs_delta_db": float(np.mean(np.abs(mag_delta))),
    }


def _downsample_2d(data: np.ndarray, max_freq_bins: int = 512, max_time_bins: int = 1024) -> np.ndarray:
    """Reduziert Spektrogramm-Auflösung für GUI-Übertragung."""
    result = data
    if result.shape[0] > max_freq_bins:
        factor = result.shape[0] // max_freq_bins
        result = result[::factor, :]
    if result.shape[1] > max_time_bins:
        factor = result.shape[1] // max_time_bins
        result = result[:, ::factor]
    return np.round(result, 1)
