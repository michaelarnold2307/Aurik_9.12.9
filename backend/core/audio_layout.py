#!/usr/bin/env python3
"""backend/core/audio_layout.py — Kanonische Layout-Helfer (SOTA-Sweep 2026-08-22).

Kontext: Die interne Pipeline-Konvention ist CHANNELS-FIRST (C, N)
(AGENTS.md). Der Batch-/GUI-Pfad liefert SAMPLES-FIRST (N, C). Beide
Layouts sind durch reines Transponieren bit-identisch — die Bug-Klasse
dieser Session (RLP, dtw_groove, Consensus, Impuls-Detektor, TQC, GPO,
MDEM, transport_bump …) entstand ausschließlich aus mean(axis=0)-Mono-
Mixen, die ein Layout hart annahmen.

Diese Module sind die EINE Quelle der Wahrheit für Layout-Fragen:
  - mono_mix()         layout-sichere Mono-Konvertierung
  - sample_axis()      Index der Zeitachse
  - is_channels_first()
  - to_channels_first()  bit-identisches Transpose (nur wenn nötig)
  - to_samples_first()

Bit-Identitäts-Garantie: alle Operationen sind reine Achsen-Wahlen/
Transposes — keine Arithmetik, keine Resamples.
"""

from __future__ import annotations

import numpy as np

# Maximale plausible Kanalzahl eines Audiosignals (Guard gegen
# Verwechslung von Zeit- und Kanal-Achse bei (2, 2)-artigen Shapes).
MAX_CHANNELS: int = 8


def is_channels_first(arr: np.ndarray) -> bool:
    """True wenn 2-D und erste Achse die KANAL-Achse ist: (C, N) mit C ≤ 8."""
    return bool(
        arr.ndim == 2
        and arr.shape[0] <= MAX_CHANNELS
        and arr.shape[1] > MAX_CHANNELS
    )


def is_samples_first(arr: np.ndarray) -> bool:
    """True wenn 2-D und letzte Achse die KANAL-Achse ist: (N, C) mit C ≤ 8."""
    return bool(
        arr.ndim == 2
        and arr.shape[1] <= MAX_CHANNELS
        and arr.shape[0] > MAX_CHANNELS
    )


def sample_axis(arr: np.ndarray) -> int:
    """Index der ZEITACHSE. Mono → 0; (C, N) → 1; (N, C) → 0; sonst 0."""
    if arr.ndim != 2:
        return 0
    return 1 if is_channels_first(arr) else 0


def mono_mix(arr: np.ndarray) -> np.ndarray:
    """Layout-sichere Mono-Konvertierung (Kanal-Mittel, nie Zeit-Mittel).

    Mono bleibt unverändert. (C, N) → (N,). (N, C) → (N,).
    Kleine/degenerierte Arrays: Achse 0, wenn diese ≤ MAX_CHANNELS ist.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if is_channels_first(arr):
        return arr.mean(axis=0)
    if is_samples_first(arr):
        return arr.mean(axis=1)
    # Ambige/kleine Shapes (z. B. (2, 2)): Kanäle auf der kleineren Achse.
    if arr.shape[0] <= arr.shape[1]:
        return arr.mean(axis=0)
    return arr.mean(axis=1)


def to_channels_first(arr: np.ndarray) -> np.ndarray:
    """Bit-identisches Transpose auf (C, N); Mono und (C, N) bleiben unverändert."""
    arr = np.asarray(arr)
    if arr.ndim != 2 or is_channels_first(arr):
        return arr
    if is_samples_first(arr):
        return np.ascontiguousarray(arr.T)
    return arr


def to_samples_first(arr: np.ndarray) -> np.ndarray:
    """Bit-identisches Transpose auf (N, C); Mono und (N, C) bleiben unverändert."""
    arr = np.asarray(arr)
    if arr.ndim != 2 or is_samples_first(arr):
        return arr
    if is_channels_first(arr):
        return np.ascontiguousarray(arr.T)
    return arr


__all__ = [
    "MAX_CHANNELS",
    "is_channels_first",
    "is_samples_first",
    "sample_axis",
    "mono_mix",
    "to_channels_first",
    "to_samples_first",
]
