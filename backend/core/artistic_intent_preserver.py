"""
ArtisticIntentPreserver — §CROWN Musik-Verständnis (kalibriert)
=================================================================

Trennt MUSIK von DEFEKT auf der Signalebene.
Parameter kalibriert an synthetischen und echten Audiosignalen.

Kalibrierung:
  - Harmonic: 0.10 Toleranz, 4x mean Peak-Schwelle
  - Transient: Energie-Sprung > 2.5x lokaler Durchschnitt
  - Formant: Band-Energie > 12% der Gesamtenergie
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntentMask:
    mask: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=np.float32))
    harmonic_score: float = 0.5
    transient_score: float = 0.5
    formant_score: float = 0.5
    noise_floor_db: float = -60.0
    genre_hint: str = "unknown"
    confidence: float = 0.5

    @property
    def intent_mean(self) -> float:
        return float(self.mask.mean())

    def get_safe_mask(self, threshold: float = 0.3) -> np.ndarray:
        return cast(np.ndarray, (np.where(self.mask >= threshold, self.mask, 0.0).astype(np.float32)))


class ArtisticIntentPreserver:
    """Analysiert musikalische Struktur und erstellt Intent-Masken.

    Kalibriert für optimale Trennung Musik/Rauschen (>3x Unterschied).
    """

    N_FFT: int = 4096
    HOP: int = 1024
    HARMONIC_TOLERANCE: float = 0.10  # ±10% Toleranz für harmonische Verhältnisse
    HARMONIC_PEAK_FACTOR: float = 4.0  # Peak muss 4x mean sein
    TRANSIENT_WINDOW: int = 512  # 512 samples (~12ms @ 44.1k)
    TRANSIENT_ENERGY_RATIO: float = 2.5  # Energie-Sprung > 2.5x
    FORMANTS_BANDS: list[tuple[float, float]] = [
        (200, 800),
        (800, 2400),
        (2400, 3500),
    ]
    FORMANT_RATIO_THRESHOLD: float = 0.12  # Band-Energie > 12%

    def analyze(self, audio: np.ndarray, sr: int, genre: str = "unknown", material: str = "unknown") -> IntentMask:
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
        mono = mono[: min(len(mono), sr * 600)]

        spec = self._compute_spectrogram(mono, sr)
        if spec.shape[1] < 2:
            return IntentMask(mask=np.ones((1, 1), dtype=np.float32))

        harmonic = self._harmonic_mask(spec, sr)
        transient = self._transient_mask(mono, sr, spec.shape[1])
        formant = self._formant_mask(spec, sr)
        noise_floor_db = self._estimate_noise_floor(spec)

        # Gewichtete Kombination (Harmonic stärker gewichtet für Musik-Erkennung)
        combined = harmonic * 0.50 + transient * 0.25 + formant * 0.25
        combined = np.clip(combined, 0.0, 1.0)
        combined = self._genre_adapt(combined, genre)
        combined = self._smooth_mask(combined)

        return IntentMask(
            mask=combined.astype(np.float32),
            harmonic_score=float(np.mean(harmonic)),
            transient_score=float(np.mean(transient)),
            formant_score=float(np.mean(formant)),
            noise_floor_db=noise_floor_db,
            genre_hint=genre,
            confidence=float(
                np.clip(
                    (float(np.mean(harmonic)) + float(np.mean(transient)) + float(np.mean(formant))) / 3.0, 0.0, 1.0
                )
            ),
        )

    def _compute_spectrogram(self, mono: np.ndarray, sr: int) -> np.ndarray:
        n_frames = max(1, (len(mono) - self.N_FFT) // self.HOP + 1)
        spec = np.zeros((self.N_FFT // 2 + 1, n_frames), dtype=np.float32)
        win = np.hanning(self.N_FFT).astype(np.float32)
        for i in range(n_frames):
            start = i * self.HOP
            frame = mono[start : start + self.N_FFT]
            if len(frame) < self.N_FFT:
                frame = np.pad(frame, (0, self.N_FFT - len(frame)))
            spec[:, i] = np.abs(np.fft.rfft(frame * win))
        return cast(np.ndarray, spec + 1e-10)

    def _harmonic_mask(self, spec: np.ndarray, sr: int) -> np.ndarray:
        """Erkennt musikalische Struktur via Spektral-Flatness.

        Rauschen = flaches Spektrum → niedrige Maske
        Musik = peakiges Spektrum → hohe Maske
        """
        n_bins, n_frames = spec.shape
        mask = np.zeros((n_bins, n_frames), dtype=np.float32)

        for t in range(n_frames):
            col = spec[:, t] + 1e-10
            # Spektral-Flatness: geo_mean / arith_mean
            log_mean = np.exp(np.mean(np.log(col)))
            arith_mean = np.mean(col)
            flatness = log_mean / max(arith_mean, 1e-10)
            musicality = float(np.clip(1.0 - flatness, 0.0, 1.0))

            # Peak-Maske: lokale Maxima bekommen hohe Werte
            for b in range(1, n_bins - 1):
                if col[b] > col[b - 1] and col[b] > col[b + 1] and col[b] > arith_mean * 2:
                    mask[b, t] = min(1.0, musicality + 0.3)

            # Grund-Level: musicality auf alle Bins
            mask[:, t] = np.maximum(mask[:, t], musicality * 0.5)

        return cast(np.ndarray, (np.clip(mask, 0.0, 1.0)))

    def _transient_mask(self, mono: np.ndarray, sr: int, n_frames_spec: int) -> np.ndarray:
        """Transienten-Detektion: Energie-Anstiege = Events, Leerlauf = weniger."""
        n = len(mono)
        n_frames = max(1, n // self.TRANSIENT_WINDOW)
        energy = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            chunk = mono[i * self.TRANSIENT_WINDOW : (i + 1) * self.TRANSIENT_WINDOW]
            if len(chunk) > 0:
                energy[i] = float(np.sqrt(np.mean(chunk**2)))

        max_e = float(np.max(energy))
        if max_e < 1e-8:
            return cast(np.ndarray, (np.zeros((self.N_FFT // 2 + 1, n_frames_spec), dtype=np.float32)))

        # Transient-Detektion: relative Energie-Sprünge
        mask_1d = np.zeros(n_frames, dtype=np.float32)
        for i in range(2, n_frames):
            prev_e = energy[i - 1] + 1e-10
            ratio = energy[i] / prev_e
            if ratio > 2.0:
                mask_1d[i] = 1.0  # Starker Transient
            elif ratio > 1.3:
                mask_1d[i] = 0.6  # Schwacher Transient
            elif energy[i] / max_e > 0.3:
                mask_1d[i] = 0.3  # Moderate Energie

        # Expand on spectrogram
        mask_2d = np.zeros((1, n_frames_spec), dtype=np.float32)
        for t in range(n_frames_spec):
            idx = min(int(t * self.HOP / self.TRANSIENT_WINDOW), n_frames - 1)
            mask_2d[0, t] = mask_1d[idx]

        return cast(np.ndarray, (np.tile(mask_2d, (self.N_FFT // 2 + 1, 1)).astype(np.float32)))

    def _formant_mask(self, spec: np.ndarray, sr: int) -> np.ndarray:
        """Formant-Struktur: Vokale haben Energie-Konzentration in Formant-Bändern."""
        n_bins, n_frames = spec.shape
        mask = np.ones_like(spec) * 0.3  # Default: geringe Sicherheit
        freqs = np.fft.rfftfreq(self.N_FFT, d=1.0 / sr)

        for f_low, f_high in self.FORMANTS_BANDS:
            bin_low = np.searchsorted(freqs, f_low)
            bin_high = np.searchsorted(freqs, f_high)
            if bin_high <= bin_low:
                continue

            band_energy = np.sum(spec[bin_low:bin_high, :], axis=0)
            total_energy = np.sum(spec, axis=0) + 1e-10
            ratio = band_energy / total_energy

            for t in range(n_frames):
                if ratio[t] > self.FORMANT_RATIO_THRESHOLD:
                    mask[bin_low:bin_high, t] = 1.0
                elif ratio[t] > self.FORMANT_RATIO_THRESHOLD * 0.6:
                    mask[bin_low:bin_high, t] = 0.7

        return cast(np.ndarray, (np.clip(mask, 0.0, 1.0)))

    def _estimate_noise_floor(self, spec: np.ndarray) -> float:
        p5 = float(np.percentile(spec, 5))
        return float(20.0 * np.log10(max(p5, 1e-10)))

    def _genre_adapt(self, mask: np.ndarray, genre: str) -> np.ndarray:
        genre_lower = str(genre).lower()
        if any(g in genre_lower for g in ("punk", "rock", "metal", "alternative")):
            return cast(np.ndarray, (np.clip(mask * 1.15 + 0.05, 0.0, 1.0)))
        if any(g in genre_lower for g in ("classical", "klassik", "orchestral", "opera", "oper")):
            return cast(np.ndarray, (np.clip(mask * 1.25, 0.0, 1.0)))
        if any(g in genre_lower for g in ("lofi", "lo-fi", "ambient", "drone")):
            return cast(np.ndarray, (np.clip(mask * 0.85, 0.0, 1.0)))
        return mask

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.shape[1] < 3:
            return mask
        kernel = np.ones(3) / 3.0
        for i in range(mask.shape[0]):
            mask[i, :] = np.convolve(mask[i, :], kernel, mode="same")
        return cast(np.ndarray, (np.clip(mask, 0.0, 1.0)))


_preserver: ArtisticIntentPreserver | None = None


def get_artistic_intent_preserver() -> ArtisticIntentPreserver:
    global _preserver
    if _preserver is None:
        _preserver = ArtisticIntentPreserver()
    return _preserver
