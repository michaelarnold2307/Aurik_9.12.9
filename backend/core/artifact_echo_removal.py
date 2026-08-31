"""§AR: ArtifactEchoRemoval — Entfernt Echo-Artefakte und Onset-Shifts.

Behebt zwei häufige DSP-Artefakte die als Warnungen auftauchen:
  1. Echo-Artefakte (ECHO_ARTIFACT) — Autokorrelations-Echos <50ms Lag
  2. Onset-Shifts — Timing-Verschiebungen >3ms durch Phasen-Verarbeitung

Algorithmus:
  - Echo-Detektion via Autokorrelation (Lag 5-50ms, Peak > 0.7 Korrelation)
  - Echo-Unterdrückung via spektrale Subtraktion des verzögerten Signals
  - Onset-Re-Alignment via Kreuzkorrelation mit Original
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


class ArtifactEchoRemoval:
    """Entfernt Echo-Artefakte und korrigiert Onset-Shifts."""

    def __init__(self) -> None:
        pass

    def remove_echo(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        original_reference: np.ndarray | None = None,
        max_lag_ms: float = 50.0,
        min_correlation: float = 0.7,
        ast_musical_confidence: float = 0.0,  # §v10.304: senkt Aggressivität bei Musik
    ) -> np.ndarray:
        """Entfernt Echo-Artefakte aus dem Audio.

        Args:
            audio: Verarbeitetes Audio
            sr: Sample-Rate
            original_reference: Original-Audio als Referenz (optional)
            max_lag_ms: Maximaler Echo-Lag
            min_correlation: Minimale Korrelation für Echo-Erkennung
            ast_musical_confidence: 0-1, hebt min_correlation bei Musik an
        """
        # §v10.304 AST-Guard: Bei Musik-Instrument-Anteil Echo-Schwelle anheben
        _effective_corr = min_correlation + ast_musical_confidence * 0.25
        _effective_corr = min(0.95, _effective_corr)  # max 0.95
        result = np.asarray(audio, dtype=np.float32).copy()
        mono = np.mean(result, axis=0) if result.ndim == 2 else result

        # Autokorrelations-basierte Echo-Detektion
        max_lag = int(sr * max_lag_ms / 1000.0)
        min_lag = int(sr * 0.005)  # 5ms minimum (unterhalb = Direct-Sound)

        if len(mono) < max_lag * 4:
            return cast(np.ndarray, result)

        echoes_found = 0
        for ch in range(result.shape[0] if result.ndim == 2 else 1):
            ch_data = result[ch] if result.ndim == 2 else result

            # Autokorrelation in Blöcken (100ms)
            block_size = int(sr * 0.100)
            hop = block_size // 2

            for i in range(0, len(ch_data) - block_size, hop):
                block = ch_data[i : i + block_size]
                if len(block) < max_lag * 2:
                    continue

                # Autokorrelation
                acorr = np.correlate(block, block, mode="full")
                acorr = acorr[len(acorr) // 2 :]  # Nur positive Lags
                acorr = acorr / (acorr[0] + 1e-12)  # Normalisieren

                # Suche nach Echo-Peaks im Bereich 5-50ms
                echo_region = acorr[min_lag:max_lag]
                if len(echo_region) == 0:
                    continue

                peak_corr = float(np.max(echo_region))
                peak_lag = int(np.argmax(echo_region)) + min_lag

                if peak_corr > _effective_corr:
                    # Echo gefunden → unterdrücken via spektrale Subtraktion
                    echo_block = np.roll(block, -peak_lag) * peak_corr * 0.5
                    corrected = block - echo_block

                    # Cross-fade zum Original
                    xfade = min(int(sr * 0.005), block_size // 4)
                    if xfade >= 2:
                        win = np.ones(block_size, dtype=np.float32)
                        win[:xfade] = np.linspace(0, 1, xfade)
                        win[-xfade:] = np.linspace(1, 0, xfade)
                        ch_data[i : i + block_size] = corrected * win + block * (1 - win)
                    else:
                        ch_data[i : i + block_size] = corrected

                    echoes_found += 1

            if result.ndim == 2:
                result[ch] = ch_data[: len(result[ch])]
            else:
                result = ch_data[: len(result)].astype(np.float32)

        if echoes_found > 0:
            logger.info("§AR EchoRemoval: %d echo artifacts removed", echoes_found)

        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    def realign_onsets(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        original_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Korrigiert Onset-Shifts durch Kreuzkorrelation mit Original.

        Nur wenn original_reference gegeben ist — sonst kein Alignment möglich.
        """
        if original_reference is None:
            return audio

        result = np.asarray(audio, dtype=np.float32).copy()
        mono_orig = np.mean(original_reference, axis=0) if original_reference.ndim == 2 else original_reference
        mono_proc = np.mean(result, axis=0) if result.ndim == 2 else result

        # Kreuzkorrelation in 1s-Blöcken für lokales Alignment
        block_size = int(sr * 1.0)
        max_shift = int(sr * 0.010)  # Max 10ms Shift

        for i in range(0, min(len(mono_orig), len(mono_proc)) - block_size, block_size):
            orig_block = mono_orig[i : i + block_size]
            proc_block = mono_proc[i : i + block_size]

            if len(orig_block) < max_shift * 2 or len(proc_block) < max_shift * 2:
                continue

            # Kreuzkorrelation
            xcorr = np.correlate(proc_block, orig_block, mode="full")
            shift = int(np.argmax(xcorr)) - len(proc_block) + 1

            if abs(shift) > 2 and abs(shift) < max_shift:  # Nur korrigieren wenn >2 Samples
                for ch in range(result.shape[0] if result.ndim == 2 else 1):
                    if shift > 0:
                        # Proc ist verzögert → vorziehen
                        result[ch, i : i + block_size - shift] = result[ch, i + shift : i + block_size]
                    elif shift < 0:
                        # Proc ist vorgezogen → verzögern
                        shift_abs = abs(shift)
                        result[ch, i + shift_abs : i + block_size] = result[ch, i : i + block_size - shift_abs]

        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        original_reference: np.ndarray | None = None,
        ast_musical_confidence: float = 0.0,  # §v10.304
    ) -> np.ndarray:
        """Entfernt Echo + korrigiert Onsets in einem Durchlauf."""
        result = self.remove_echo(
            audio,
            sr,
            ast_musical_confidence=ast_musical_confidence,
        )
        if original_reference is not None:
            result = self.realign_onsets(result, sr, original_reference=original_reference)
        return result
