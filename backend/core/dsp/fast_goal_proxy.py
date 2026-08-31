"""backend/core/dsp/fast_goal_proxy.py — §v10.700 H2.

FastGoalProxy: DSP-only Pre-Screening der 15 Musical Goals in ≤200ms.
Kein ML, kein FFT>4096, keine externen Modelle.

Nutzt Bark-Band-Analyse, Crest-Faktor, Onset-Dichte und Chroma-Merkmale
für eine schnelle Abschätzung der Goal-Werte VOR dem Pipeline-Start.

Korrelation mit PMGG-Ergebnissen: ≥0.85 (Ziel).

Nutzung:
    proxy = FastGoalProxy()
    goals = proxy.measure_fast(audio, sample_rate=48000)
    # → {"brillanz": 0.62, "waerme": 0.73, ...}
"""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

# Bark-Band-Grenzen (Hz) für 48kHz
_BARK_EDGES = np.array(
    [
        0,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
        24000,
    ],
    dtype=np.float64,
)


class FastGoalProxy:
    """Schnelle DSP-basierte Goal-Abschätzung (≤200ms für 15 Goals)."""

    def measure_fast(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        """Misst alle 15 Musical Goals in einem Durchlauf.

        Returns:
            Dict mit Goal-Namen → Score [0.0, 1.0]
        """
        t0 = time.monotonic()

        # Mono-Konvertierung
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=-1)
        else:
            audio_mono = audio

        audio_f64 = np.asarray(audio_mono, dtype=np.float64).ravel()

        # ── Bark-Band-Energie ─────────────────────────────────
        bark_energy = _bark_band_energy(audio_f64, sample_rate)

        # ── Zeitbereichs-Merkmale ──────────────────────────────
        rms = float(np.sqrt(np.mean(audio_f64**2)))
        peak = float(np.abs(audio_f64).max())
        crest = peak / (rms + 1e-10)

        # ── Frequenzbereichs-Merkmale ──────────────────────────
        n_fft = min(4096, len(audio_f64))
        spec = np.abs(np.fft.rfft(audio_f64[:n_fft]))
        spec_db = 20 * np.log10(spec + 1e-10)

        spectral_centroid = float(np.sum(np.arange(len(spec)) * spec) / (np.sum(spec) + 1e-10))
        spectral_flatness = float(np.exp(np.mean(np.log(spec + 1e-10))) / (np.mean(spec) + 1e-10))

        # ── Goal-Berechnung ────────────────────────────────────
        goals: dict[str, float] = {}

        # Brillanz: Energie in hohen Bark-Bändern (Bark 20-24)
        goals["brillanz"] = _score(np.sum(bark_energy[19:24]) / (np.sum(bark_energy) + 1e-10), 0.02, 0.30)

        # Wärme: Energie in mittleren Bändern (Bark 8-14)
        goals["waerme"] = _score(np.sum(bark_energy[7:14]) / (np.sum(bark_energy) + 1e-10), 0.10, 0.45)

        # Natürlichkeit: Spectral Flatness (je flacher = rauschiger)
        goals["natuerlichkeit"] = 1.0 - _score(spectral_flatness, 0.1, 0.8)

        # Authentizität: Crest-Faktor (hoch = transientenreich = natürlich)
        goals["authentizitaet"] = _score(crest, 2.0, 8.0)

        # Bass-Kraft: Sub-Bass-Energie (Bark 0-3, <200Hz)
        goals["bass_kraft"] = _score(np.sum(bark_energy[:4]) / (np.sum(bark_energy) + 1e-10), 0.01, 0.25)

        # Transparenz: spektrale Klarheit (Centroid)
        goals["transparenz"] = _score(spectral_centroid / len(spec), 0.03, 0.20)

        # Emotionalität: Dynamik-Varianz
        frame_size = sample_rate // 50  # 20ms frames
        n_frames = len(audio_f64) // frame_size
        if n_frames > 1:
            frame_rms = np.sqrt(np.mean(audio_f64[: n_frames * frame_size].reshape(n_frames, frame_size) ** 2, axis=1))
            dyn_range = float(np.std(frame_rms) / (np.mean(frame_rms) + 1e-10))
        else:
            dyn_range = 0.0
        goals["emotionalitaet"] = _score(dyn_range, 0.05, 0.40)

        # Groove: Onset-Dichte
        onset_density = _onset_density(audio_f64, sample_rate)
        goals["groove"] = _score(onset_density, 0.5, 4.0)

        # Raumtiefe: Stereo-Korrelation (nur bei Stereo)
        if audio.ndim > 1 and audio.shape[-1] >= 2:
            corr = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
            goals["spatial_depth"] = _score(1.0 - abs(corr), 0.0, 0.6)
        else:
            goals["spatial_depth"] = 0.0

        # Timbre-Authentizität: Harmonische Dichte
        harm_density = _harmonic_density(spec)
        goals["timbre_authentizitaet"] = _score(harm_density, 0.01, 0.15)

        # Tonales Zentrum: Chroma-Peak
        chroma_peak = _chroma_peak(audio_f64, sample_rate)
        goals["tonales_zentrum"] = _score(chroma_peak, 1.5, 4.0)

        # Mikro-Dynamik: Frame-zu-Frame-Varianz
        goals["micro_dynamics"] = _score(float(np.std(np.diff(frame_rms))) if n_frames > 2 else 0.0, 0.001, 0.02)

        # Separation-Treue: Spektrale Schärfe
        goals["separation_fidelity"] = _score(spectral_centroid / len(spec), 0.05, 0.18)

        # Artikulation: Transienten-Anteil
        goals["artikulation"] = _score(crest, 3.0, 10.0)

        elapsed = time.monotonic() - t0
        goals["_elapsed_ms"] = round(elapsed * 1000, 1)

        return goals


# ── Hilfsfunktionen ─────────────────────────────────────────────


def _bark_band_energy(audio: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet Energie in 25 Bark-Bändern."""
    n_fft = 4096
    spec = np.abs(np.fft.rfft(audio[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    energy = np.zeros(len(_BARK_EDGES) - 1)
    for i in range(len(_BARK_EDGES) - 1):
        mask = (freqs >= _BARK_EDGES[i]) & (freqs < _BARK_EDGES[i + 1])
        energy[i] = np.sum(spec[mask] ** 2)
    return cast(np.ndarray, energy)


def _onset_density(audio: np.ndarray, sr: int) -> float:
    """Onset-Dichte (Onsets pro Sekunde)."""
    frame_size = 512
    hop_size = 256
    n_frames = (len(audio) - frame_size) // hop_size
    if n_frames < 2:
        return 0.0
    energy = np.array([np.sum(audio[i * hop_size : i * hop_size + frame_size] ** 2) for i in range(n_frames)])
    # Spectral flux approximation
    flux = np.diff(energy)
    threshold = np.mean(flux) + 0.5 * np.std(flux)
    onsets = int(np.sum(flux > threshold))
    duration_s = len(audio) / sr
    return onsets / max(duration_s, 0.01)  # type: ignore[no-any-return]


def _harmonic_density(spec: np.ndarray) -> float:
    """Harmonische Dichte: Anteil der Peaks an Gesamtenergie."""
    if len(spec) < 4:
        return 0.0
    # Finde lokale Peaks
    peaks = (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:])
    peak_energy = float(np.sum(spec[1:-1][peaks] ** 2))
    total_energy = np.sum(spec**2) + 1e-10
    return float(peak_energy / total_energy)


def _chroma_peak(audio: np.ndarray, sr: int) -> float:
    """Chroma-Peak: Stärke des dominantesten Pitch-Class."""
    n_fft = 4096
    spec = np.abs(np.fft.rfft(audio[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # 12 Chroma-Bins (C, C#, D, ...)
    chroma = np.zeros(12)
    for i, f in enumerate(freqs):
        if f > 80:  # ignoriere DC
            pitch_class = int(round(12 * np.log2(f / 261.63))) % 12
            chroma[pitch_class] += spec[i] ** 2

    chroma_sum = np.sum(chroma) + 1e-10
    return float(np.max(chroma) / (chroma_sum / 12))


def _score(value: float, low: float, high: float) -> float:
    """Normalisiert einen Wert in [low, high] auf Score [0, 1]."""
    normalized = (value - low) / (high - low + 1e-10)
    return float(np.clip(normalized, 0.0, 1.0))
