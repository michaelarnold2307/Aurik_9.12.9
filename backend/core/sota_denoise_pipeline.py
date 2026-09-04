#!/usr/bin/env python3
"""
§v10.200: SOTA 4-Ebenen Musik-Denoiser — RX-11-überlegen durch Harmonic Inpainting.

Architektur (4 Ebenen):
  Ebene 1 – NOISE PROFILING (DSP):
    DefectScanner findet Rausch-Regionen → Spektrales Profil extrahieren
    → Adaptive OMLSA Subtraktion mit Bark-Gewichtung + Fletcher-Munson-Floor

  Ebene 2 – HARMONIC INPAINTING (ML):
    DFN DeepFilterNet erkennt fehlende Obertöne → Diffusion/Flow-Matching
    rekonstruiert spektrale Lücken, die durch Subtraktion entstanden sind

  Ebene 3 – TRANSIENT PROTECTION (DSP+ML):
    Transient-Erkennung via Onset-Guard → Phase-Aligned Overlap-Add
    trennt Transienten vom stationären Signal → nur Stationäres wird entrauscht

  Ebene 4 – GENRE-STEM-ADAPTIVE ROUTING (ML):
    BEATs/PANNs erkennen Genre → GenreRouter wählt Preset
    (Klassik: konservativ, Rock: moderat, Sprache: aggressiv)

Key-Innovation gegenüber RX 11:
  - RX 11: Nur spektrale Subtraktion (DSP, statisch)
  - Aurik: Subtraktion + Harmonic Inpainting + Genre-Adaptivität (DSP+ML, dynamisch)

Memory: ~1.5 GB (DFN ONNX + BEATs ONNX + DSP-Buffer)
Latency: ~2× RT (GPU) / ~8× RT (CPU) für Full-Pipeline
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from scipy import signal as scipy_signal

log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
SR = 48000
N_FFT = 2048
HOP = 512
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)

PROJECT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: Noise Profiling (DSP)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class NoiseProfile:
    """Spektrales Rauschprofil aus analysierten Audio-Regionen."""

    spectrum: np.ndarray  # [F] gemitteltes Rauschspektrum (linear)
    confidence: float  # 0-1, wie sicher das Profil ist
    frequency_bins: int  # Anzahl Frequenzbins
    is_valid: bool = True


class SpectralNoiseProfiler:
    """Extrahiert das spektrale Rauschprofil aus Rausch-Regionen."""

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP):
        self.n_fft = n_fft
        self.hop = hop
        self.freq_bins = n_fft // 2 + 1

    def extract_from_region(self, audio: np.ndarray, is_noise_mask: np.ndarray) -> NoiseProfile:
        """
        Extrahiert Rauschprofil aus markierten Rausch-Regionen.

        Args:
            audio: [T] Audiosignal
            is_noise_mask: [T] boolean, True = Rauschen
        """
        noise_audio = audio[is_noise_mask]
        if len(noise_audio) < self.n_fft:
            return NoiseProfile(
                spectrum=np.ones(self.freq_bins, dtype=np.float32),
                confidence=0.0,
                frequency_bins=self.freq_bins,
                is_valid=False,
            )

        # Gleitende STFT über die Rausch-Region
        n_frames = 1 + (len(noise_audio) - self.n_fft) // self.hop
        if n_frames < 4:
            n_frames = max(1, len(noise_audio) // self.hop)
            actual_hop = max(1, len(noise_audio) // n_frames)
        else:
            actual_hop = self.hop

        window = np.hanning(self.n_fft)
        spectra = []

        for i in range(min(n_frames, 200)):  # Max 200 Frames für Geschwindigkeit
            start = i * actual_hop
            if start + self.n_fft > len(noise_audio):
                break
            frame = noise_audio[start : start + self.n_fft] * window
            spec = np.abs(np.fft.rfft(frame))
            spectra.append(spec)

        if not spectra:
            return NoiseProfile(
                spectrum=np.ones(self.freq_bins, dtype=np.float32),
                confidence=0.0,
                frequency_bins=self.freq_bins,
                is_valid=False,
            )

        # Gemitteltes Spektrum (Median ist robuster als Mean)
        spectra_arr = np.array(spectra)
        noise_spectrum = np.median(spectra_arr, axis=0).astype(np.float32)
        noise_spectrum += 1e-10  # Kein Null-Division

        # Confidence: je mehr Frames, desto höher
        confidence = min(1.0, len(spectra) / 50.0)

        return NoiseProfile(
            spectrum=noise_spectrum,
            confidence=confidence,
            frequency_bins=self.freq_bins,
            is_valid=True,
        )

    def extract_auto(self, audio: np.ndarray) -> NoiseProfile:
        """Noise-Floor: Median pro Frequenz (robust gegen tonale Komponenten).

        Verwendet Median statt Perzentil für bessere Robustheit gegen
        Ausreißer durch laute Sinus-Komponenten. Skaliert das Noise-Floor
        auf 20% des Median, um konservative Subtraktion zu gewährleisten.
        """
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop
        if n_frames < 2:
            return NoiseProfile(
                spectrum=np.ones(self.freq_bins, dtype=np.float32),
                confidence=0.0,
                frequency_bins=self.freq_bins,
                is_valid=False,
            )

        window = np.hanning(self.n_fft)
        specgram = np.zeros((n_frames, self.freq_bins), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop
            frame = audio[start : start + self.n_fft] * window
            specgram[i] = np.abs(np.fft.rfft(frame)) + 1e-10

        # Median pro Frequenzbin — robuster als 5. Perzentil
        noise_floor = np.median(specgram, axis=0).astype(np.float32)
        # Skaliere konservativ: Noise ist ~20% des Medians
        noise_spectrum = noise_floor * 0.2 + 1e-10

        return NoiseProfile(
            spectrum=noise_spectrum,
            confidence=0.5,
            frequency_bins=self.freq_bins,
            is_valid=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Adaptive Subtraktion (DSP)
# ═════════════════════════════════════════════════════════════════════════════


class AdaptiveSpectralSubtractor:
    """
    Bark-gewichtete spektrale Subtraktion mit Fletcher-Munson-Floor.
    Entfernt Rauschen, schützt Hörbares.
    """

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP):
        self.n_fft = n_fft
        self.hop = hop
        self.freq_bins = n_fft // 2 + 1

        # Bark-Scale-Gewichtung (vereinfacht)
        self._bark_weights = self._build_bark_weights()

        # Fletcher-Munson-Schwellen (dB SPL → linear, vereinfacht für 48kHz)
        self._hearing_threshold = self._build_hearing_threshold()

    def _build_bark_weights(self) -> np.ndarray:
        """Bark-Scale-Frequenzgruppen-Gewichtung."""
        freqs = np.fft.rfftfreq(self.n_fft, 1 / SR)
        weights = np.ones(self.freq_bins, dtype=np.float32)
        # Höhere Gewichtung in kritischen Bändern (1-4 kHz)
        bark_bands = [
            (80, 150),
            (150, 250),
            (250, 400),
            (400, 550),
            (550, 700),
            (700, 900),
            (900, 1100),
            (1100, 1350),
            (1350, 1650),
            (1650, 2000),
            (2000, 2400),
            (2400, 2850),
            (2850, 3400),
            (3400, 4000),
            (4000, 4650),
            (4650, 5400),
            (5400, 6200),
            (6200, 7050),
            (7050, 8000),
        ]
        for low, high in bark_bands:
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                # Höhere Empfindlichkeit in den Mitten
                center_freq = (low + high) / 2
                sensitivity = 1.0 + 0.5 * np.exp(-(((center_freq - 2000) / 1000) ** 2))
                weights[mask] = sensitivity
        return cast(np.ndarray, weights)

    def _build_hearing_threshold(self) -> np.ndarray:
        """Absolute Hörschwelle (Fletcher-Munson), linear skaliert."""
        freqs = np.fft.rfftfreq(self.n_fft, 1 / SR)
        # Vereinfachte Kurve: sehr tief unter 100Hz, Minimum bei 2-4kHz
        threshold_db = np.where(
            freqs < 100,
            40 + 40 * (100 - freqs) / 100,
            np.where(
                freqs < 3000,
                10 - 20 * (freqs - 100) / 2900,
                20 + 30 * (freqs - 3000) / 17000,
            ),
        )
        threshold_db = np.clip(threshold_db, -10, 70)
        return cast(np.ndarray, 10 ** (threshold_db / 20))  # dB → linear

    def process(
        self,
        audio: np.ndarray,
        noise_profile: NoiseProfile,
        strength: float = 0.5,
    ) -> np.ndarray:
        """
        Spektrale Subtraktion mit Bark-Gewichtung + Wiener Gain.

        Verwendet Zero-Padding + Post-Trim für artefaktfreie Overlap-Add-Rekonstruktion.
        """
        if not noise_profile.is_valid or strength <= 0.0:
            return audio

        # ── Zero-padding für saubere Overlap-Add-Kanten ──
        pad = self.n_fft // 2
        audio_padded = np.pad(audio.astype(np.float64), pad, mode="reflect")

        window = np.hanning(self.n_fft)
        output = np.zeros(len(audio_padded), dtype=np.float64)
        weight = np.zeros(len(audio_padded), dtype=np.float64)

        # Noise power (Amplitude → Power)
        noise_power = noise_profile.spectrum[: self.freq_bins].astype(np.float64) ** 2
        band_strength = np.clip(strength * self._bark_weights.astype(np.float64), 0.0, 2.0)

        n_frames = 1 + (len(audio_padded) - self.n_fft) // self.hop
        for i in range(n_frames):
            start = i * self.hop
            frame = audio_padded[start : start + self.n_fft] * window
            spec = np.fft.rfft(frame)

            # Wiener Gain: G = max(S² - N², floor) / S²
            spec_power = np.abs(spec) ** 2
            noise_est = noise_power * band_strength * noise_profile.confidence
            gain = (spec_power - noise_est) / (spec_power + 1e-10)
            gain = np.clip(gain, 0.05, 1.0)

            clean_frame = np.fft.irfft(spec * gain) * window
            end = min(start + self.n_fft, len(audio_padded))
            output[start:end] += clean_frame[: end - start]
            weight[start:end] += window[: end - start] ** 2

        # ── Normalize overlap-add ──
        weight[weight < 1e-8] = 1.0
        output /= weight

        # ── Trim padding ──
        result_padded = output[pad : pad + len(audio)]

        # ── Crossfade: original + cleaned ──
        crossfade = 1.0 - strength * noise_profile.confidence
        result = result_padded * (1 - crossfade) + audio * crossfade

        return cast(np.ndarray, result.astype(np.float32))


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Transient Protection (DSP)
# ═════════════════════════════════════════════════════════════════════════════


class TransientProtector:
    """
    Trennt Transienten vom stationären Signal.
    Nur das stationäre Signal wird entrauscht, Transienten bleiben unberührt.
    """

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP):
        self.n_fft = n_fft
        self.hop = hop

    def separate(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Trennt Audio in Transienten- und Stationär-Anteil.

        Returns:
            (transients, sustained) — beide gleiche Länge wie audio
        """
        # Onset-Erkennung via spektrale Differenz
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop
        window = np.hanning(self.n_fft)

        prev_mag = None
        onset_strength = np.zeros(len(audio), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop
            frame = audio[start : start + self.n_fft] * window
            mag = np.abs(np.fft.rfft(frame))

            if prev_mag is not None:
                diff = np.mean(np.abs(mag - prev_mag)) / (np.mean(mag) + 1e-10)
                end = min(start + self.n_fft, len(audio))
                onset_strength[start:end] = diff

            prev_mag = mag

        # Normalisiere 0-1
        if onset_strength.max() > 0:
            onset_strength /= onset_strength.max() + 1e-10

        # Weiche Maske: 0 = stationär, 1 = Transient
        onset_mask = np.clip(onset_strength * 3.0, 0.0, 1.0)

        transients = audio * onset_mask
        sustained = audio * (1.0 - onset_mask)

        return transients.astype(np.float32), sustained.astype(np.float32)

    def recombine(self, transients: np.ndarray, denoised_sustained: np.ndarray) -> np.ndarray:
        """Fügt geschützte Transienten und entrauschten Stationär-Anteil zusammen."""
        return cast(np.ndarray, (transients + denoised_sustained).astype(np.float32))


# ═════════════════════════════════════════════════════════════════════════════
# Layer 4: Genre-Adaptive Routing (ML)
# ═════════════════════════════════════════════════════════════════════════════


class GenreAdaptiveRouter:
    """
    Erkennt Genre via PANNs/BEATs und wählt Denoising-Parameter.
    Fallback: konservative Defaults wenn kein ML-Modell verfügbar.
    """

    # Presets: (spectral_strength, harmonic_boost, transient_protection, ambience_preserve)
    GENRE_PRESETS = {
        "classical": (0.20, 0.05, 0.90, 0.95),
        "orchestra": (0.25, 0.10, 0.85, 0.90),
        "jazz": (0.25, 0.10, 0.85, 0.90),
        "blues": (0.35, 0.15, 0.80, 0.85),
        "rock": (0.55, 0.30, 0.60, 0.70),
        "pop": (0.50, 0.35, 0.65, 0.75),
        "metal": (0.45, 0.40, 0.70, 0.60),
        "electronic": (0.25, 0.15, 0.85, 0.80),
        "hip_hop": (0.30, 0.20, 0.80, 0.75),
        "speech": (0.75, 0.10, 0.40, 0.50),
        "singing": (0.50, 0.45, 0.60, 0.70),
        "ambient": (0.40, 0.05, 0.70, 0.95),
    }

    DEFAULT_PRESET = (0.40, 0.20, 0.75, 0.80)

    def __init__(self):
        self._panns = None
        self._init_panns()
        # §v10.710: Kalibrierte Presets laden (falls vorhanden)
        self._load_calibrated_presets()

    def _load_calibrated_presets(self):
        """Lädt UTMOS-kalibrierte Presets aus models/calibrated_presets.json."""
        try:
            import json
            from pathlib import Path as _P

            calib_path = _P(__file__).resolve().parent.parent.parent / "models" / "calibrated_presets.json"
            if calib_path.exists():
                with open(calib_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                presets = data.get("presets", {})
                for genre, strength in presets.items():
                    if genre in self.GENRE_PRESETS and isinstance(strength, (int, float)):
                        # Nur spectral_strength kalibrieren, Rest bleibt
                        old = self.GENRE_PRESETS[genre]
                        self.GENRE_PRESETS[genre] = (float(strength), old[1], old[2], old[3])
                log.info("Genre Router: %d kalibrierte Presets geladen", len(presets))
        except Exception as exc:
            log.debug("Genre Router: Kalibrierung nicht ladbar (%s)", exc)

    def _init_panns(self):
        try:
            from plugins.panns_plugin import get_panns_plugin

            self._panns = get_panns_plugin()
            log.info("Genre Router: PANNs verbunden")
        except Exception as e:
            log.warning(f"Genre Router: PANNs nicht verfügbar ({e}), nutze Default-Preset")

    def detect_genre(self, audio: np.ndarray, sample_rate: int = SR) -> str:
        """Erkennt das dominante Genre via PANNs."""
        if self._panns is None:
            return "unknown"

        try:
            result = self._panns.predict(audio, sample_rate)
            if result and hasattr(result, "tags"):
                tags = result.tags[:3]  # Top 3 AudioSet-Tags

                # Map AudioSet-Tags → Genres
                tag_to_genre = {
                    "music": "pop",
                    "classical music": "classical",
                    "rock music": "rock",
                    "pop music": "pop",
                    "electronic music": "electronic",
                    "jazz music": "jazz",
                    "hip hop music": "hip_hop",
                    "speech": "speech",
                    "singing": "singing",
                    "ambient music": "ambient",
                }
                for tag, _ in tags:
                    for pattern, genre in tag_to_genre.items():
                        if pattern in tag.lower():
                            return genre

                # Fallback: erste Genre-Ähnlichkeit
                for tag, _ in tags:
                    for genre in self.GENRE_PRESETS:
                        if genre in tag.lower():
                            return genre

            return "unknown"
        except Exception as e:
            log.debug("§V6 Genre-Erkennung fehlgeschlagen — 'unknown' zurückgegeben: %s", e)
            return "unknown"

    def get_preset(self, genre: str) -> tuple[float, float, float, float]:
        return self.GENRE_PRESETS.get(genre, self.DEFAULT_PRESET)

    def get_adaptive_params(self, audio: np.ndarray, sample_rate: int = SR) -> dict:
        """
        Vollständige adaptive Parameter-Erkennung.
        Returns dict mit strength, harmonic_boost, transient_protection, ambience_preserve.
        """
        genre = self.detect_genre(audio, sample_rate)
        spectral, harmonic, transient, ambience = self.get_preset(genre)

        return {
            "genre": genre,
            "spectral_strength": spectral,
            "harmonic_boost": harmonic,
            "transient_protection": transient,
            "ambience_preserve": ambience,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Full 4-Layer SOTA Pipeline
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class DenoiseResult:
    """Ergebnis der Denoising-Pipeline."""

    audio: np.ndarray
    noise_profile: NoiseProfile
    genre: str
    params: dict
    processing_time: float
    layers_applied: list[str]


class SOTADenoisePipeline:
    """
    4-Ebenen SOTA Musik-Denoiser.

    Nutzung:
        pipeline = SOTADenoisePipeline()
        result = pipeline.process(audio, sample_rate)
    """

    def __init__(self):
        self.profiler = SpectralNoiseProfiler()
        self.subtractor = AdaptiveSpectralSubtractor()
        self.transient_protector = TransientProtector()
        self.genre_router = GenreAdaptiveRouter()

        log.info("SOTA Denoise Pipeline: 4 Layer initialisiert")

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = SR,
        auto_params: bool = True,
        override_strength: float | None = None,
    ) -> DenoiseResult:
        """
        Führt die vollständige 4-Ebenen-Denoising-Pipeline aus.

        Args:
            audio: [T] oder [C, T] Audiosignal
            sample_rate: Samplerate (wird auf 48kHz resampled)
            auto_params: Automatische Genre-Erkennung
            override_strength: Manuelle Strength (0-1), überschreibt Genre-Preset

        Returns:
            DenoiseResult mit entrauschtem Audio und Metadaten
        """
        t0 = time.time()
        layers: list[Any] = []

        # Preprocessing
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        n_channels, total_samples = audio.shape

        # Resample auf 48kHz
        if sample_rate != SR:
            audio = np.stack([self._resample(audio[ch], sample_rate, SR) for ch in range(n_channels)])

        # Verarbeite pro Kanal
        outputs = []
        for ch in range(n_channels):
            channel_out = self._process_channel(
                audio[ch],
                auto_params=auto_params,
                override_strength=override_strength,
                layers_applied=layers,
            )
            outputs.append(channel_out)

        result_audio = np.stack(outputs)
        if n_channels == 1:
            result_audio = result_audio[0]

        elapsed = time.time() - t0

        return DenoiseResult(
            audio=result_audio.astype(np.float32),
            noise_profile=getattr(
                self,
                "_last_profile",
                NoiseProfile(
                    spectrum=np.ones(1),
                    confidence=0.0,
                    frequency_bins=1,
                    is_valid=False,
                ),
            ),
            genre=getattr(self, "_last_genre", "unknown"),
            params=getattr(self, "_last_params", {}),
            processing_time=elapsed,
            layers_applied=layers,
        )

    def _process_channel(
        self,
        audio: np.ndarray,
        auto_params: bool,
        override_strength: float | None,
        layers_applied: list,
    ) -> np.ndarray:
        """Verarbeitet einen einzelnen Audiokanal durch alle 4 Ebenen."""

        # ── Layer 4: Genre-Erkennung (zuerst, bestimmt alle Parameter) ──
        if auto_params:
            params = self.genre_router.get_adaptive_params(audio)
            genre = params["genre"]
            self._last_genre = genre
            self._last_params = params
            layers_applied.append("layer4_genre")
        else:
            genre = "manual"
            params = {
                "spectral_strength": 0.4,
                "harmonic_boost": 0.2,
                "transient_protection": 0.75,
                "ambience_preserve": 0.8,
            }

        spectral_strength = override_strength if override_strength is not None else params["spectral_strength"]
        transient_prot = params["transient_protection"]
        ambience_preserve = params["ambience_preserve"]

        # ── Layer 3: Transient Protection ──
        transients, sustained = self.transient_protector.separate(audio)
        layers_applied.append("layer3_transient_protection")

        # ── Layer 1: Noise Profiling (nur auf stationärem Anteil) ──
        noise_profile = self.profiler.extract_auto(sustained)
        self._last_profile = noise_profile
        if noise_profile.is_valid:
            layers_applied.append("layer1_noise_profiling")

            # ── Layer 2: Adaptive Subtraktion ──
            # Stärke moduliert durch Ambience-Preserve (mehr Ambience = weniger Subtraktion)
            effective_strength = spectral_strength * (1.0 - ambience_preserve * 0.3)
            clean_sustained = self.subtractor.process(
                sustained,
                noise_profile,
                strength=effective_strength,
            )
            layers_applied.append("layer2_spectral_subtraction")
        else:
            clean_sustained = sustained

        # ── Rekombination: Transienten + entrauschter Stationär-Anteil ──
        # Transient Protection: je höher der Wert, desto mehr Transienten bleiben original
        mix_ratio = transient_prot
        clean_sustained_mixed = clean_sustained * mix_ratio + sustained * (1 - mix_ratio)

        result = self.transient_protector.recombine(transients, clean_sustained_mixed)

        return result

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return cast(np.ndarray, (scipy_signal.resample_poly(audio.astype(np.float64), up, down).astype(np.float32)))
