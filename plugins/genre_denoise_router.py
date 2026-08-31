#!/usr/bin/env python3
"""
§v10.131: Genre-Aware Denoising Router — adapts denoising strength to music genre.

Uses PANNs CNN14 (AudioSet-527) to detect the music genre, then selects
the optimal denoising preset. Different genres have different noise tolerance:
  - Classical/Jazz: minimal denoising (preserve dynamics)
  - Rock/Pop: moderate (balance clarity and energy)
  - Electronic/Hip-Hop: light (synthetic sounds already clean)
  - Speech/Podcast: aggressive (voice clarity priority)
  - Ambient/Field: moderate-heavy (noise floor reduction)

Dependencies: PANNs ONNX (already loaded by panns_plugin)
Memory: < 1 MB (genre mapping tables only, no model loading)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# AudioSet label indices for genre detection
AUDIOSET_GENRE_MAP = {
    # Classical family
    "classical": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
    ],
    "orchestra": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "opera": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "choir": [0, 1, 2, 3],
    # Rock/Pop family
    "rock": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "pop": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    "metal": [70, 71, 72, 73, 74, 75, 76, 77],
    "punk": [78, 79, 80, 81, 82],
    "indie": [83, 84, 85, 86, 87, 88, 89],
    # Electronic family
    "electronic": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    "techno": [100, 101, 102, 103, 104, 105],
    "house": [106, 107, 108, 109, 110, 111],
    "ambient": [112, 113, 114, 115, 116, 117],
    "drum_and_bass": [118, 119, 120, 121, 122],
    # Hip-Hop/R&B
    "hip_hop": [123, 124, 125, 126, 127, 128, 129, 130],
    "rnb": [131, 132, 133, 134, 135, 136, 137, 138],
    # Jazz/Blues
    "jazz": [139, 140, 141, 142, 143, 144, 145, 146, 147, 148],
    "blues": [149, 150, 151, 152, 153, 154, 155, 156],
    # Folk/World
    "folk": [157, 158, 159, 160, 161, 162, 163, 164, 165],
    "world": [166, 167, 168, 169, 170, 171, 172, 173, 174],
    "latin": [175, 176, 177, 178, 179, 180, 181, 182],
    "reggae": [183, 184, 185, 186, 187],
    # Speech/Vocal
    "speech": [188, 189, 190, 191, 192, 193, 194, 195, 196],
    "singing": [197, 198, 199, 200, 201, 202, 203, 204, 205],
    # Soundscape
    "field_recording": [206, 207, 208, 209, 210, 211, 212],
    "nature": [213, 214, 215, 216, 217, 218, 219, 220],
    "urban": [221, 222, 223, 224, 225, 226, 227, 228],
}

# Denoising presets per genre
# Format: (df_strength, erb_smoothing, transient_protection, harmonic_boost)
GENRE_DENOISE_PRESETS = {
    "classical": (0.20, 0.10, 0.90, 0.05),  # Very light, preserve dynamics
    "orchestra": (0.25, 0.15, 0.85, 0.10),
    "opera": (0.30, 0.20, 0.80, 0.15),
    "choir": (0.35, 0.25, 0.75, 0.10),
    "rock": (0.55, 0.40, 0.60, 0.30),
    "pop": (0.50, 0.35, 0.65, 0.35),
    "metal": (0.45, 0.30, 0.70, 0.40),
    "punk": (0.60, 0.45, 0.55, 0.25),
    "indie": (0.45, 0.30, 0.70, 0.20),
    "electronic": (0.25, 0.20, 0.85, 0.15),
    "techno": (0.20, 0.15, 0.90, 0.20),
    "house": (0.25, 0.20, 0.85, 0.25),
    "ambient": (0.40, 0.30, 0.70, 0.10),
    "drum_and_bass": (0.35, 0.25, 0.80, 0.30),
    "hip_hop": (0.30, 0.25, 0.80, 0.35),
    "rnb": (0.35, 0.25, 0.75, 0.40),
    "jazz": (0.25, 0.15, 0.85, 0.10),
    "blues": (0.35, 0.25, 0.75, 0.20),
    "folk": (0.40, 0.30, 0.70, 0.15),
    "world": (0.40, 0.30, 0.70, 0.20),
    "latin": (0.45, 0.35, 0.65, 0.30),
    "reggae": (0.40, 0.30, 0.70, 0.25),
    "speech": (0.75, 0.50, 0.40, 0.10),
    "singing": (0.50, 0.35, 0.65, 0.45),
    "field_recording": (0.65, 0.45, 0.50, 0.05),
    "nature": (0.55, 0.40, 0.55, 0.05),
    "urban": (0.60, 0.45, 0.50, 0.10),
}

DEFAULT_PRESET = (0.45, 0.30, 0.70, 0.20)


@dataclass
class GenreResult:
    """Result of genre classification and denoising recommendation."""

    primary_genre: str
    confidence: float
    top3_genres: list[tuple[str, float]]
    df_strength: float
    erb_smoothing: float
    transient_protection: float
    harmonic_boost: float


class GenreAwareDenoiseRouter:
    """Routes denoising decisions based on detected music genre."""

    def __init__(self):
        self._genre_labels = sorted(AUDIOSET_GENRE_MAP.keys())
        # Build a reverse mapping: AudioSet index → genre
        self._index_to_genre: dict[int, str] = {}
        for genre, indices in AUDIOSET_GENRE_MAP.items():
            for idx in indices:
                if idx not in self._index_to_genre:
                    self._index_to_genre[idx] = genre
                # Keep first assignment (most specific)

        log.info(f"Genre Router: {len(self._genre_labels)} genres, {len(self._index_to_genre)} AudioSet indices mapped")

    def classify(self, audioset_scores: np.ndarray) -> GenreResult:
        """
        Classify genre from AudioSet-527 scores and return denoising preset.

        Args:
            audioset_scores: [527] float32 AudioSet sigmoid scores.
        """
        if audioset_scores.shape[-1] != 527:
            raise ValueError(f"Expected 527 AudioSet scores, got {audioset_scores.shape}")

        scores = audioset_scores.reshape(-1)
        if scores.ndim > 1:
            scores = scores.mean(axis=0)

        # Aggregate scores by genre
        genre_scores: dict[str, float] = {}
        for genre, indices in AUDIOSET_GENRE_MAP.items():
            valid_indices = [i for i in indices if i < 527]
            if valid_indices:
                genre_scores[genre] = float(scores[valid_indices].mean())

        if not genre_scores:
            return self._build_result("unknown", 0.0, [], DEFAULT_PRESET)

        # Sort genres by score
        ranked = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        primary_genre, primary_score = ranked[0]
        top3 = [(g, float(s)) for g, s in ranked[:3]]

        # Get denoising preset
        preset = GENRE_DENOISE_PRESETS.get(primary_genre, DEFAULT_PRESET)

        return self._build_result(primary_genre, primary_score, top3, preset)

    def _build_result(
        self,
        genre: str,
        confidence: float,
        top3: list[tuple[str, float]],
        preset: tuple[float, float, float, float],
    ) -> GenreResult:
        return GenreResult(
            primary_genre=genre,
            confidence=round(confidence, 3),
            top3_genres=top3,
            df_strength=preset[0],
            erb_smoothing=preset[1],
            transient_protection=preset[2],
            harmonic_boost=preset[3],
        )

    def get_preset(self, genre: str) -> tuple[float, float, float, float]:
        """Get denoising preset for a specific genre."""
        return GENRE_DENOISE_PRESETS.get(genre, DEFAULT_PRESET)

    def interpolate_preset(
        self,
        genre1: str,
        genre2: str,
        alpha: float,
    ) -> tuple[float, float, float, float]:
        """Linearly interpolate between two genre presets."""
        p1 = self.get_preset(genre1)
        p2 = self.get_preset(genre2)
        return (
            round(p1[0] * (1 - alpha) + p2[0] * alpha, 3),
            round(p1[1] * (1 - alpha) + p2[1] * alpha, 3),
            round(p1[2] * (1 - alpha) + p2[2] * alpha, 3),
            round(p1[3] * (1 - alpha) + p2[3] * alpha, 3),
        )
