"""
optimization/priority3_oversampling.py – Adaptiver Oversampling-Prozessor.
=========================================================================

Applies 2× oversampling only on transient-dense regions to suppress
aliasing artefacts during non-linear processing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np


class AdaptiveOversamplingProcessor:
    """Überabtastet selektiv transiente Regionen.

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    oversample_factor:
        Integer oversampling ratio for transient segments.
    """

    def __init__(self, sr: int = 48000, oversample_factor: int = 2) -> None:
        self.sr = sr
        self.oversample_factor = oversample_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Gibt processed audio of the same length as *audio* zurück.

        Transient regions are upsampled, processed, and downsampled;
        non-transient regions are passed through with a mild spectral
        smoothing pass.
        """
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return x.copy()  # type: ignore[no-any-return]

        try:
            import librosa

            onsets = librosa.onset.onset_detect(y=x, sr=sr, units="samples")  # type: ignore[attr-defined]
        except Exception:
            onsets = np.array([], dtype=int)

        mask = self._create_transient_mask(x, sr, onsets)

        out = x.copy()
        # Process transient regions with oversampling (simple cubic interp)
        hop = 512
        for start in range(0, len(x) - hop, hop):
            segment = x[start : start + hop]
            processed = self._oversample_process(segment) if mask[start] else segment
            out[start : start + hop] = processed

        return np.clip(  # type: ignore[no-any-return]
            np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

    def _create_transient_mask(
        self,
        audio: np.ndarray,
        sr: int,
        onsets: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        """Erstellt a boolean mask marking transient regions.

        Each onset index triggers a 20 ms window to be marked.
        """
        mask = np.zeros(len(audio), dtype=bool)
        window = int(0.020 * sr)  # 20 ms
        for onset in onsets:
            start = max(0, int(onset) - window // 4)
            end = min(len(audio), int(onset) + window)
            mask[start:end] = True
        return mask  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _oversample_process(self, segment: np.ndarray) -> np.ndarray:
        """Upsample → apply mild gain normalisation → downsample."""
        n = len(segment)
        factor = self.oversample_factor
        # Upsample by zero-insertion + LP filter (via FFT)
        spec = np.fft.rfft(segment)
        spec_up = np.zeros(n * factor // 2 + 1, dtype=complex)
        spec_up[: len(spec)] = spec
        upsampled = np.fft.irfft(spec_up, n=n * factor)
        # Mild compression gain
        upsampled = np.tanh(upsampled * 0.9) / 0.9
        # Downsample
        downsampled = upsampled[::factor][:n]
        if len(downsampled) < n:
            downsampled = np.pad(downsampled, (0, n - len(downsampled)))
        return downsampled.astype(np.float32)  # type: ignore[no-any-return]


logger = logging.getLogger(__name__)
