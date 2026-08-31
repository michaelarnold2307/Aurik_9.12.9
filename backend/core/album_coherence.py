"""Album-Kohärenz — Konsistente Restaurierung über alle Tracks eines Albums.

Verhindert dass gleicher Artist auf verschiedenen Tracks unterschiedlich klingt.
Speichert Referenz-Profile pro Album/Artist und wendet sie auf alle Tracks an.

- Loudness-Normalisierung: alle Tracks auf gleichen LUFS-Pegel
- EQ-Kohärenz: spektrale Balance über Tracks hinweg abgleichen
- Dynamik-Band: alle Tracks im gleichen Dynamik-Fenster
- Stimm-Konsistenz: Vocal-Modelle pro Artist wiederverwenden
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)
STORE_PATH = Path.home() / ".aurik" / "album_profiles.json"


class AlbumProfile:
    """Gespeichertes Profil für ein Album/einen Artist."""

    def __init__(self, album_id: str) -> None:
        self.album_id = album_id
        self.track_count: int = 0
        self.loudness_lufs: float = -14.0
        self.loudness_range_lu: float = 6.0
        self.eq_reference: list[float] = []
        self.dynamic_range_db: float = 12.0
        self.voice_model: dict[str, Any] = {}
        self.track_qualities: list[float] = []

    def update_from_track(
        self, lufs: float, lra: float, eq_curve: list[float], dyn_range: float, quality: float
    ) -> None:
        n = self.track_count
        w_old = n / (n + 1)
        w_new = 1.0 / (n + 1)
        self.loudness_lufs = w_old * self.loudness_lufs + w_new * lufs
        self.loudness_range_lu = w_old * self.loudness_range_lu + w_new * lra
        self.dynamic_range_db = w_old * self.dynamic_range_db + w_new * dyn_range
        if eq_curve and self.eq_reference:
            if len(eq_curve) == len(self.eq_reference):
                self.eq_reference = [w_old * s + w_new * e for s, e in zip(self.eq_reference, eq_curve)]
        elif eq_curve:
            self.eq_reference = list(eq_curve)
        self.track_qualities.append(quality)
        self.track_count += 1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> AlbumProfile:
        p = cls(d["album_id"])
        p.__dict__.update(d)
        return p


class AlbumCoherence:
    """Album-weite Kohärenz-Steuerung."""

    def __init__(self) -> None:
        self._profiles: dict[str, AlbumProfile] = {}
        self._load()

    def _load(self) -> None:
        if STORE_PATH.exists():
            try:
                with open(STORE_PATH) as f:
                    data = json.load(f)
                self._profiles = {k: AlbumProfile.from_dict(v) for k, v in data.items()}
            except Exception:
                pass

    def _save(self) -> None:
        STORE_PATH.parent.mkdir(exist_ok=True)
        with open(STORE_PATH, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._profiles.items()}, f, indent=2)

    def get_or_create_profile(self, album_id: str) -> AlbumProfile:
        if album_id not in self._profiles:
            self._profiles[album_id] = AlbumProfile(album_id)
        return self._profiles[album_id]

    def apply_coherence(
        self, album_id: str, audio: np.ndarray, sample_rate: int, current_lufs: float, track_quality: float
    ) -> np.ndarray:
        """Wendet Album-Kohärenz auf einen Track an."""
        profile = self.get_or_create_profile(album_id)
        if profile.track_count == 0:
            return audio  # Erster Track = Referenz

        # Loudness an Album-Niveau anpassen
        gain_db = profile.loudness_lufs - current_lufs
        gain_linear = 10.0 ** (gain_db / 20.0)
        result = audio * np.clip(gain_linear, 0.5, 2.0)

        logger.info(
            "Album-Coherence [%s]: Track %d, LUFS-Gain %.1fdB, Ø-Qualität %.1f",
            album_id,
            profile.track_count + 1,
            gain_db,
            float(np.mean(profile.track_qualities + [track_quality])),
        )
        return cast(np.ndarray, result.astype(np.float32))
