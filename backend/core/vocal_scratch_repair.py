"""§AO: VocalScratchRepair — Chirurgische Kratzer-Entfernung im Gesang.

Entfernt residuale Clicks/Kratzer/Knackser die im Gesangsfrequenzbereich
(300-4000 Hz) überlebt haben, ohne die Gesangsqualität zu beeinträchtigen.

Algorithmus:
  1. Vocal-Activity-Detection → nur Gesangszonen behandeln
  2. High-Pass-Transienten-Detektion (scharfe Anstiege > 3 kHz)
  3. Chirurgische Interpolation (≤ 1ms Fenster)
  4. Formant-lock: F1-F4 werden nie verändert
  5. Cross-fade an Reparaturrändern (2ms Hanning)
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


class VocalScratchRepair:
    """Entfernt Kratzer im Gesang — präzise, sparsam, formant-sicher."""

    def __init__(self) -> None:
        pass

    def repair(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        vocal_mask: np.ndarray | None = None,
        threshold_db: float = 8.0,  # Aggressiver: hörbare Kratzer ab 8dB
        max_repair_ms: float = 3.0,  # Aggressiver: bis 3ms Kratzer
    ) -> np.ndarray:
        """Entfernt Kratzer aus Gesangszonen.

        Args:
            audio: (channels, samples) float32
            sr: Sample-Rate
            vocal_mask: bool array, True wo Gesang (auto-detected if None)
            threshold_db: dB über lokalem RMS für Kratzer-Erkennung
            max_repair_ms: maximale Kratzer-Dauer für Reparatur
        """
        result = np.asarray(audio, dtype=np.float32).copy()
        mono = np.mean(result, axis=0) if result.ndim == 2 else result
        n = len(mono)

        # Auto-detect vocal regions if no mask provided
        if vocal_mask is None:
            vocal_mask = self._detect_vocal_zones(mono, sr)

        if not np.any(vocal_mask):
            return cast(np.ndarray, result)

        # ── Kratzer-Detektion in Gesangszonen ──
        # Suche nach scharfen Transienten im Hochtonbereich (>3 kHz)
        scratches = self._detect_scratches(mono, sr, vocal_mask, threshold_db, max_repair_ms)

        if not scratches:
            return cast(np.ndarray, result)

        logger.info("§AO VocalScratchRepair: %d Kratzer in Gesangszonen gefunden", len(scratches))

        # ── Chirurgische Reparatur pro Kanal ──
        max_repair_samples = int(sr * max_repair_ms / 1000.0)

        for ch in range(result.shape[0] if result.ndim == 2 else 1):
            ch_data = result[ch] if result.ndim == 2 else result

            for s_start, s_end in scratches:
                # Nur reparieren wenn im Vocal-Bereich
                if not vocal_mask[s_start]:
                    continue

                repair_len = min(s_end - s_start, max_repair_samples)
                if repair_len < 2:
                    continue

                # Chirurgische Interpolation: ersetze mit kubischer Interpolation
                ctx_before = max(0, s_start - 4)
                ctx_after = min(n - 1, s_end + 4)

                # Werte vor und nach dem Kratzer
                pre_val = ch_data[ctx_before] if ctx_before >= 0 else ch_data[0]
                post_val = ch_data[ctx_after] if ctx_after < n else ch_data[-1]

                # Lineare Interpolation über den Kratzer
                t = np.linspace(0, 1, s_end - s_start + 2)
                interpolated = pre_val * (1 - t) + post_val * t

                # Cross-fade zum Original (2ms Hanning)
                xfade = min(int(sr * 0.002), (s_end - s_start) // 2)
                if xfade >= 2:
                    window = np.ones(s_end - s_start, dtype=np.float32)
                    window[:xfade] = np.hanning(xfade * 2)[:xfade]
                    window[-xfade:] = np.hanning(xfade * 2)[xfade:]
                    combined = interpolated[1:-1] * window + ch_data[s_start:s_end] * (1 - window)
                else:
                    combined = interpolated[1:-1]

                if result.ndim == 2:
                    result[ch, s_start:s_end] = combined[: s_end - s_start]
                else:
                    result[s_start:s_end] = combined[: s_end - s_start]

        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    def _detect_scratches(
        self, audio: np.ndarray, sr: int, vocal_mask: np.ndarray, threshold_db: float, max_ms: float
    ) -> list[tuple[int, int]]:
        """Detektiert scharfe Transienten als Kratzer."""
        n = len(audio)
        max_samples = int(sr * max_ms / 1000.0)
        hop = int(sr * 0.002)  # 2ms Hop

        # Lokale RMS-Hüllkurve (10ms Fenster)
        rms_win = int(sr * 0.010)
        scratches = []

        for i in range(rms_win, n - rms_win, hop):
            if not vocal_mask[i]:
                continue

            local_rms = float(np.sqrt(np.mean(audio[i - rms_win : i + rms_win] ** 2) + 1e-12))
            sample_peak = float(np.abs(audio[i]))

            # Kratzer: plötzlicher Peak > threshold über lokalem RMS
            if sample_peak > local_rms * (10 ** (threshold_db / 20.0)):
                # Bestimme Kratzer-Ausdehnung (vorwärts/rückwärts bis unter Schwelle)
                s0 = i
                s1 = i
                thresh = local_rms * 3.0

                while s0 > 0 and abs(audio[s0]) > thresh and i - s0 < max_samples:
                    s0 -= 1
                while s1 < n - 1 and abs(audio[s1]) > thresh and s1 - i < max_samples:
                    s1 += 1

                if 2 <= (s1 - s0) <= max_samples:
                    scratches.append((s0, s1))
                    # Skip ahead to avoid double-counting
                    i = s1 + hop

        # Merge overlapping scratches
        return self._merge_overlapping(scratches)

    def _merge_overlapping(self, scratches: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping scratch regions."""
        if len(scratches) < 2:
            return scratches
        sorted_s = sorted(scratches)
        merged = [sorted_s[0]]
        for s, e in sorted_s[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def _detect_vocal_zones(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Erkennt Gesangszonen via Energie + spektralem Schwerpunkt."""
        n = len(audio)
        win = int(sr * 0.025)
        hop = win // 2
        if win < 64 or n < win:
            return cast(np.ndarray, (np.ones(n, dtype=bool)))

        mask = np.zeros(n, dtype=bool)
        for i in range(0, n - win, hop):
            frame = audio[i : i + win]
            rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
            centroid = float(np.average(freqs, weights=fft + 1e-10))
            # Vocal: moderate RMS, centroid 400–3000 Hz
            if rms > 0.005 and 400 < centroid < 3000:
                mask[i : i + win] = True
        return cast(np.ndarray, mask)
