"""
optimization/priority4_phase.py – Mehrband-Phasenkohärenz-Verbesserer.
=====================================================================

Applies linear-phase FIR filtering per frequency band to improve
inter-band phase alignment.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter


class MultibandPhaseCoherenceEnhancer:
    """Linear-phase multiband filtering for phase coherence improvement.

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    """

    _BANDS = [
        (20, 250),
        (250, 800),
        (800, 2000),
        (2000, 8000),
        (8000, 20000),
    ]

    def __init__(self, sr: int = 48000) -> None:
        self.sr = sr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance phase coherence across frequency bands.

        Returns an ndarray within ±1 s of the input length.
        """
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return x.copy()  # type: ignore[no-any-return]

        out = np.zeros(len(x), dtype=np.float64)
        for low_hz, high_hz in self._BANDS:
            band = self._extract_band_linear_phase(x, sr, low_hz, high_hz)
            # Trim / pad to original length
            if len(band) > len(x):
                band = band[: len(x)]
            elif len(band) < len(x):
                band = np.pad(band, (0, len(x) - len(band)))
            out += band

        out = np.clip(
            np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0) / len(self._BANDS),
            -1.0,
            1.0,
        )
        return out.astype(np.float32)  # type: ignore[no-any-return]

    def _extract_band_linear_phase(
        self,
        audio: np.ndarray,
        sr: int,
        low_hz: float,
        high_hz: float,
        numtaps: int = 101,
    ) -> np.ndarray:
        """Extrahiert a band using a linear-phase FIR bandpass filter.

        Parameters
        ----------
        audio:
            Input signal (float32 or float64).
        sr:
            Sample rate.
        low_hz, high_hz:
            Pass-band edges in Hz.
        numtaps:
            FIR filter order (must be odd for linear phase).

        Returns
        -------
        Filtered signal, same length as *audio*.
        """
        x = np.asarray(audio, dtype=np.float64)
        nyq = sr / 2.0
        low_norm = max(low_hz / nyq, 1e-4)
        high_norm = min(high_hz / nyq, 1.0 - 1e-4)

        if low_norm >= high_norm:
            return np.zeros(len(x))  # type: ignore[no-any-return]

        try:
            if low_norm < 1e-3:
                # Lowpass
                taps = firwin(numtaps, high_norm, window="hamming")
            else:
                taps = firwin(
                    numtaps,
                    [low_norm, high_norm],
                    pass_zero=False,
                    window="hamming",
                )
            filtered = lfilter(taps, [1.0], x)
        except Exception:
            filtered = x.copy()

        return filtered  # type: ignore[no-any-return]
logger = logging.getLogger(__name__)
