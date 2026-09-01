"""DefectHeatmap — §INCREMENTAL #6: Visuelle Defekt-Karte.

Erstellt eine Zeit×Frequenz-Matrix mit Defekt-Intensitäten.
Exportierbar als JSON für GUI-Visualisierung.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute(audio: np.ndarray, sr: int, time_resolution_s: float = 1.0) -> dict[str, Any]:
    """Erstellt Defekt-Heatmap aus Audio-Signal."""
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    n = len(mono)
    hop = int(sr * time_resolution_s)
    n_segments = max(1, n // hop)
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    heatmap = np.zeros((n_segments, 6), dtype=np.float32)  # 6 Defekt-Typen

    for seg in range(n_segments):
        start = seg * hop
        chunk = mono[start : start + hop]
        if len(chunk) < hop:
            chunk = np.pad(chunk, (0, hop - len(chunk)))
        spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))

        # 1. Clicks: hochfrequente Impulse
        hf_energy = float(np.sum(spec[freqs >= 6000] ** 2))
        total_energy = np.sum(spec**2) + 1e-10
        heatmap[seg, 0] = float(np.clip(hf_energy / total_energy * 2, 0, 1))

        # 2. Hum: 50/60Hz + Harmonische
        hum_bands = [(45, 65), (95, 125), (145, 165)]
        hum_e = sum(np.sum(spec[(freqs >= lo) & (freqs <= hi)] ** 2) for lo, hi in hum_bands)
        heatmap[seg, 1] = float(np.clip(hum_e / total_energy * 3, 0, 1))

        # 3. Hiss: Rauschflur über 8kHz
        hiss = spec[freqs >= 8000]
        _hiss_ratio = float(np.mean(hiss)) / max(float(np.mean(spec)), 1e-10) * 5.0
        heatmap[seg, 2] = max(0.0, min(1.0, _hiss_ratio))

        # 4. Clipping
        heatmap[seg, 3] = float(np.clip(np.mean(np.abs(chunk) > 0.95) * 50, 0, 1))

        # 5. Dropouts: plötzliche Stille
        rms = float(np.sqrt(np.mean(chunk**2)))
        heatmap[seg, 4] = 1.0 if rms < 1e-5 else 0.0

        # 6. Crackle: feine HF-Impulse (kürzer als Clicks)
        crackle = np.diff(chunk)
        _crackle_ratio = float(np.std(crackle)) / max(float(np.std(chunk)), 1e-10)
        heatmap[seg, 5] = max(0.0, min(1.0, _crackle_ratio))

    return {
        "time_resolution_s": time_resolution_s,
        "duration_s": n / sr,
        "n_segments": n_segments,
        "defect_types": ["clicks", "hum", "hiss", "clipping", "dropouts", "crackle"],
        "heatmap": heatmap.tolist(),
        "overall": {
            t: float(np.mean(heatmap[:, i]))
            for i, t in enumerate(["clicks", "hum", "hiss", "clipping", "dropouts", "crackle"])
        },
    }


def to_json(heatmap: dict[str, Any], path: str | None = None) -> str:
    data = json.dumps(heatmap, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(data)
    return data
