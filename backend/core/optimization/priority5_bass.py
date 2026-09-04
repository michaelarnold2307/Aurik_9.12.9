"""
optimization/priority5_bass.py – Phasenkohärente Bass-Verarbeitung.
===============================================================

Applies linear-phase low-end enhancement while preserving resonance
character of the source material.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


class ResonancePreserver:
    """Erkennt dominant resonance peaks in the low-frequency range.

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    """

    def __init__(self, sr: int = 48000) -> None:
        self.sr = sr

    def detect_resonances(
        self,
        audio: np.ndarray,
        sr: int,
        n_top: int = 10,
        freq_range: tuple[float, float] = (20.0, 500.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Erkennt resonant frequencies in *audio*.

        Parameters
        ----------
        audio:
            Input signal.
        sr:
            Sample rate.
        n_top:
            Maximum number of resonance peaks to return.
        freq_range:
            (low_hz, high_hz) search window.

        Returns
        -------
        (freqs, mags) — arrays of resonance frequencies and magnitudes.
        """
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        n_fft = min(8192, len(x))
        spec = np.abs(np.fft.rfft(x[:n_fft], n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        low, high = freq_range
        band_mask = (freqs >= low) & (freqs <= high)
        band_freqs = freqs[band_mask]
        band_mags = spec[band_mask]

        if len(band_mags) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        # Simple peak picking: bins with local maxima
        peak_indices = []
        for i in range(1, len(band_mags) - 1):
            if band_mags[i] > band_mags[i - 1] and band_mags[i] > band_mags[i + 1]:
                peak_indices.append(i)

        if not peak_indices:
            peak_indices = [int(np.argmax(band_mags))]

        # Sort by magnitude, take top-n
        peak_indices = sorted(peak_indices, key=lambda i: band_mags[i], reverse=True)[:n_top]

        out_freqs = band_freqs[peak_indices].astype(np.float32)
        out_mags = band_mags[peak_indices].astype(np.float32)
        return out_freqs, out_mags


class PhaseCoherentBassProcessor:
    """Linear-phase FIR bass enhancement.

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    """

    def __init__(self, sr: int = 48000) -> None:
        self.sr = sr
        self._preserver = ResonancePreserver(sr=sr)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Wendet an: linear-phase low-end enhancement.

        Returns an ndarray within ±1 s of the input length.
        """
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return x.copy()  # type: ignore[no-any-return]

        # Low-pass to extract sub-bass (< 250 Hz) — zero-phase (§2.51: sosfiltfilt statt
        # causalem lfilter, FIR lfilter(taps=127) erzeugte 63-Sample Gruppenversatz → Comb-Filter)
        nyq = sr / 2.0
        cutoff = min(250.0 / nyq, 0.99)
        try:
            _sos_bass = butter(4, cutoff, btype="low", output="sos")
            bass = sosfiltfilt(_sos_bass, x.astype(np.float64)) if len(x) >= 15 else x.astype(np.float64)
        except Exception:
            bass = x.astype(np.float64)

        # Gentle +2 dB enhancement to sub-bass
        enhanced = x.astype(np.float64) + 0.26 * bass  # 0.26 ≈ +2 dB mix-in

        out = np.clip(np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return out.astype(np.float32)  # type: ignore[no-any-return]
logger = logging.getLogger(__name__)
