"""SBR (Spectral Band Replication) — Adaptive bandwidth extension for Aurik.

Replaces the basic octave-copy with envelope-aware spectral shaping.
Natural-sounding HF extension without ML model dependency.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


def _sbr_extend(audio: np.ndarray, sr: int) -> np.ndarray:
    """Adaptive spectral band replication for bandwidth extension.

    Instead of per-bin octave-copy (which creates grainy comb-filtering),
    this analyses the spectral envelope of the source band and recreates
    a natural-sounding HF extension with matched timbre.

    Algorithm:
      1. Detect effective bandwidth → adaptive source/destination bands
      2. Compute smoothed spectral envelope of source band
      3. Apply envelope to destination band with soft crossover
      4. Preserve original phase — only enhance magnitude
      5. Adaptive gain based on source-to-destination energy ratio
    """
    try:
        n_fft = 2048
        hop = n_fft // 4
        D = _librosa_stft(audio, n_fft=n_fft, hop_length=hop)
        mag, phase = np.abs(D), np.angle(D)

        freq_per_bin = sr / n_fft
        nyq_bin = mag.shape[0] - 1

        # --- Step 1: Adaptive band detection ---
        # Find the effective upper bandwidth of the material
        # (frequency above which mean energy drops below -40 dB of peak)
        mean_mag = np.mean(mag, axis=1)
        peak_mag = float(np.max(mean_mag))
        if peak_mag < 1e-10:
            return audio  # silence

        # Find highest bin with energy > -40 dB of peak
        threshold = peak_mag * 0.01  # -40 dB
        above = np.where(mean_mag > threshold)[0]
        if len(above) == 0:
            return audio
        effective_bw_bin = int(above[-1])
        _effective_bw_hz = effective_bw_bin * freq_per_bin

        # Source: lower half of effective bandwidth (where real energy lives)
        src_lo = int(max(1, effective_bw_bin * 0.25))
        src_hi = int(max(src_lo + 4, effective_bw_bin * 0.55))
        src_len = src_hi - src_lo
        if src_len < 4:
            return audio

        # Destination: from upper source boundary to effective BW × 1.6
        dst_lo = src_hi
        dst_hi = int(min(nyq_bin, effective_bw_bin * 1.6))
        dst_len = dst_hi - dst_lo
        if dst_len < 4:
            return audio

        # --- Step 2: Smoothed spectral envelope ---
        # Compute per-frame envelope via moving-average in frequency
        smooth_win = max(3, src_len // 8)
        src_mag = mag[src_lo:src_hi, :]  # (src_len, n_frames)

        # Per-frame: apply gentle smoothing in frequency
        if smooth_win >= 3:
            kernel = np.hanning(smooth_win)
            kernel = kernel / kernel.sum()
            from scipy.ndimage import convolve1d

            src_smooth = np.zeros_like(src_mag)
            for t in range(src_mag.shape[1]):
                src_smooth[:, t] = convolve1d(src_mag[:, t], kernel, mode="nearest")
        else:
            src_smooth = src_mag

        # Per-frame spectral tilt of source band (dB/octave)
        src_tilt = np.zeros(src_mag.shape[1], dtype=np.float32)
        for t in range(src_mag.shape[1]):
            col = src_smooth[:, t]
            if np.max(col) > 1e-10:
                nonzero = col > 1e-10
                if nonzero.sum() >= 4:
                    x = np.arange(src_len, dtype=np.float64)[nonzero]
                    y = np.log(np.maximum(col[nonzero], 1e-10))
                    if len(x) >= 2:
                        coeffs = np.polyfit(x, y, 1)
                        src_tilt[t] = float(coeffs[0])

        # --- Step 3: Apply envelope to destination ---
        for t in range(mag.shape[1]):
            # Build destination envelope: resample source envelope to dst bins
            # with natural HF roll-off continuing the source tilt
            env_src = src_smooth[:, t]
            env_max = float(np.max(env_src))
            if env_max < 1e-10:
                continue

            # Normalize and resample
            env_norm = env_src / env_max
            # Map src indices [0, src_len) to dst indices [0, dst_len)
            src_indices = np.linspace(0, src_len - 1, dst_len)
            env_dst = np.interp(src_indices, np.arange(src_len), env_norm)

            # Apply natural HF decay: continue the spectral tilt from source
            tilt_db_per_bin = float(src_tilt[t])  # log-mag per bin
            decay = np.exp(tilt_db_per_bin * np.arange(dst_len, dtype=np.float64))
            env_dst = env_dst * decay

            # Adaptive gain: match perceived loudness with crossover fade
            crossfade = np.linspace(1.0, 0.3, dst_len)  # fade from 1.0 to 0.3
            gain = 0.35 * crossfade  # overall gentle gain

            # Apply to destination bins (max with existing to preserve)
            for i in range(dst_len):
                dst_bin = dst_lo + i
                if dst_bin < mag.shape[0]:
                    new_val = env_dst[i] * env_max * gain[i]
                    mag[dst_bin, t] = np.maximum(mag[dst_bin, t], new_val)

        # --- Step 4: Reconstruct ---
        D_new = mag * np.exp(1j * phase)
        y = _librosa_istft(D_new, hop_length=hop, length=len(audio))
        return cast(np.ndarray, (np.asarray(y, dtype=np.float32)))

    except Exception as e:
        logger.warning("sbr_extend.py::_sbr_extend Ersatzpfad: %s", e)
        return audio


# ── librosa wrappers (graceful degradation without librosa) ──────────


def _librosa_stft(audio, n_fft=2048, hop_length=512):
    """STFT with librosa or numpy fallback."""
    try:
        import librosa

        return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    except ImportError as exc:
        logger.debug("§V6 librosa.stft nicht verfügbar — NumPy-FFT Fallback aktiviert: %s", exc)
        n_frames = (len(audio) - n_fft) // hop_length + 1
        result = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
        window = np.hanning(n_fft)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft] * window
            result[:, i] = np.fft.rfft(frame, n=n_fft)
        return result


def _librosa_istft(D, hop_length=512, length=None):
    """ISTFT with librosa or numpy fallback."""
    try:
        import librosa

        return librosa.istft(D, hop_length=hop_length, length=length)
    except ImportError as exc:
        logger.debug("§V6 librosa.istft nicht verfügbar — NumPy-ISTFT Fallback aktiviert: %s", exc)
        n_fft = (D.shape[0] - 1) * 2
        n_frames = D.shape[1]
        result = np.zeros(n_fft + hop_length * (n_frames - 1), dtype=np.float64)
        window = np.hanning(n_fft)
        for i in range(n_frames):
            frame = np.fft.irfft(D[:, i], n=n_fft)
            start = i * hop_length
            result[start : start + n_fft] += frame * window
        if length is not None:
            result = result[:length]
        return result
