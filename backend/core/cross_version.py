"""CrossVersion — §INCREMENTAL #10.

Mehrere Versionen desselben Songs → Best-of-Kompilation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VersionScore:
    index: int = 0
    brightness: float = 0.5
    dynamic_range_db: float = 12.0
    clip_free: float = 1.0
    overall: float = 0.5


def score_versions(versions: list[np.ndarray], sr: int) -> list[VersionScore]:
    """Bewertet mehrere Versionen nach Qualitätskriterien."""
    scores = []
    for i, audio in enumerate(versions):
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
        n_fft = min(4096, len(mono))
        spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        hf = float(np.sum(spec[freqs >= 4000] ** 2))
        total = np.sum(spec**2) + 1e-10
        brightness = float(np.clip(hf / total, 0, 1))
        abs_m = np.abs(mono)
        p99 = float(np.percentile(abs_m, 99))
        p1 = float(np.percentile(abs_m, 1)) + 1e-10
        dr = 20 * np.log10(p99 / p1)
        clip_free = float(np.clip(1.0 - np.mean(abs_m > 0.98) * 100, 0, 1))
        overall = brightness * 0.3 + min(dr / 40, 1.0) * 0.4 + clip_free * 0.3
        scores.append(
            VersionScore(index=i, brightness=brightness, dynamic_range_db=dr, clip_free=clip_free, overall=overall)
        )
    return scores


def select_best(versions: list[np.ndarray], scores: list[VersionScore]) -> np.ndarray:
    """Wählt die beste Version aus."""
    if not scores:
        return versions[0] if versions else np.zeros(1)
    best = max(scores, key=lambda s: s.overall)
    logger.info("CrossVersion: selected v%d (Wert=%.2f)", best.index, best.overall)
    return versions[best.index]
