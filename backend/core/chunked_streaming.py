"""backend/core/chunked_streaming.py — §v10.350 Chunked-Streaming Architektur.

Grundstein für O(duration) → O(1) RAM-Transformation.
Verarbeitet lange Audio-Dateien in überlappenden Chunks statt als Ganzes.

Status: ✅ Produktion (§v10.450 UnifiedRestorerV3._restore_chunked)

Design:
- Audio wird in Chunks von chunk_duration_s (default 30s) zerlegt
- Jeder Chunk durchläuft die komplette Pipeline
- Crossfade-Overlap (default 2s) für nahtlose Übergänge
- Nur 2 Chunks gleichzeitig im RAM (aktueller + nächster)
- RAM-Bedarf: O(chunk_size) statt O(duration)
  - 224s Audio: von ~28 GB auf ~4 GB

Usage::

    from backend.core.chunked_streaming import ChunkedPipeline

    cp = ChunkedPipeline(restorer, chunk_duration_s=30.0, overlap_s=2.0)
    result = cp.process(audio, sample_rate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# §v10.350: Default-Parameter aus Benchmarking (224s Kassette, 42 Phasen).
DEFAULT_CHUNK_DURATION_S: float = 30.0
DEFAULT_OVERLAP_S: float = 2.0
DEFAULT_CROSSFADE_S: float = 0.05  # 50ms Crossfade


@dataclass
class ChunkConfig:
    """Konfiguration für Chunked-Streaming."""

    chunk_duration_s: float = DEFAULT_CHUNK_DURATION_S
    overlap_s: float = DEFAULT_OVERLAP_S
    crossfade_s: float = DEFAULT_CROSSFADE_S
    min_chunk_duration_s: float = 10.0  # Letzter Chunk mindestens so lang


@dataclass
class ChunkResult:
    """Ergebnis eines einzelnen Chunk-Durchlaufs."""

    audio: np.ndarray
    sample_rate: int
    chunk_index: int
    start_sample: int
    end_sample: int


class ChunkedPipeline:
    """Chunked-Streaming-Wrapper für die Restaurierungs-Pipeline.

    Zerlegt langes Audio in Chunks, verarbeitet jeden durch die Pipeline,
    und setzt sie mit Crossfade-Overlap wieder zusammen.

    Thread-sicher: verarbeitet Chunks sequentiell (kein Parallelismus).
    """

    def __init__(
        self,
        chunk_duration_s: float = DEFAULT_CHUNK_DURATION_S,
        overlap_s: float = DEFAULT_OVERLAP_S,
        crossfade_s: float = DEFAULT_CROSSFADE_S,
    ) -> None:
        self.config = ChunkConfig(
            chunk_duration_s=chunk_duration_s,
            overlap_s=overlap_s,
            crossfade_s=crossfade_s,
        )
        logger.info(
            "ChunkedPipeline: chunk=%.1fs overlap=%.1fs crossfade=%.0fms",
            chunk_duration_s,
            overlap_s,
            crossfade_s * 1000,
        )

    def compute_chunks(self, audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """Berechnet Chunk-Grenzen (start_sample, end_sample) für Audio.

        Returns:
            Liste von (start_sample, end_sample) Tupeln.
        """
        # §v10.451: audio.shape[0] für Sample-Zahl (nicht shape[-1]=Kanäle)
        n_samples = audio.shape[0]
        chunk_samples = int(self.config.chunk_duration_s * sample_rate)
        overlap_samples = int(self.config.overlap_s * sample_rate)
        # §v10.459: Guard gegen negative step (overlap > chunk_size)
        if overlap_samples >= chunk_samples:
            overlap_samples = max(1, chunk_samples // 4)
            logger.warning(
                "ChunkedPipeline: overlap (%.1fs) >= chunk (%.1fs) → clamped to %.1fs",
                self.config.overlap_s,
                self.config.chunk_duration_s,
                overlap_samples / sample_rate,
            )
        step = chunk_samples - overlap_samples

        chunks: list[tuple[int, int]] = []
        start = 0
        while start < n_samples:
            end = min(start + chunk_samples, n_samples)
            # Letzter Chunk: mindestens min_chunk_duration_s
            remaining = n_samples - start
            if remaining < chunk_samples and remaining < int(self.config.min_chunk_duration_s * sample_rate):
                # Zu kurz für eigenen Chunk → an vorherigen anhängen
                if chunks:
                    chunks[-1] = (chunks[-1][0], n_samples)
                else:
                    chunks.append((0, n_samples))
                break
            chunks.append((start, end))
            start += step

        logger.info("ChunkedPipeline: %d Chunks für %.1fs Audio", len(chunks), n_samples / sample_rate)
        return chunks

    def crossfade(self, chunk_a: np.ndarray, chunk_b: np.ndarray, overlap_samples: int) -> np.ndarray:
        """Hann-Fenster-Crossfade zwischen zwei Chunks.

        §v10.401: Hann-Fenster statt linear — null Ableitung an den
        Endpunkten verhindert Klick-Artefakte an Chunk-Grenzen.

        Args:
            chunk_a: Ende des vorherigen Chunks.
            chunk_b: Anfang des nächsten Chunks.
            overlap_samples: Anzahl Samples im Überlappungsbereich.

        Returns:
            np.ndarray mit Crossfade-Ergebnis (nur Überlappungsbereich).
        """
        if overlap_samples <= 0:
            return chunk_b

        # Hann-Fenster: fade-out + fade-in = 1.0 (perfekte Rekonstruktion)
        t = np.linspace(0, np.pi, overlap_samples, dtype=np.float32)
        fade_out = np.cos(t * 0.5) ** 2  # cos² fade-out
        fade_in = np.sin(t * 0.5) ** 2  # sin² fade-in
        # §v10.451: Stereo: (n,1) broadcast mit (n,ch) → (n,ch)
        if chunk_a.ndim == 2:
            fade_out = fade_out[:, np.newaxis]
            fade_in = fade_in[:, np.newaxis]
            a_tail = chunk_a[-overlap_samples:, :]
            b_head = chunk_b[:overlap_samples, :]
        else:
            a_tail = chunk_a[-overlap_samples:]
            b_head = chunk_b[:overlap_samples]

        return a_tail * fade_out + b_head * fade_in  # type: ignore[no-any-return]

    def collect_results(self, results: list[ChunkResult], sample_rate: int) -> np.ndarray:
        """Setzt Chunk-Ergebnisse mit Crossfade + RMS-Matching zusammen.

        §v10.712: RMS-Matching an Chunk-Grenzen vor Crossfade.
        Jeder Folge-Chunk wird auf den RMS-Pegel des vorherigen Chunks
        normalisiert (letzte 500ms des vorherigen vs erste 500ms des aktuellen).
        Verhindert die massiven Pegelsprünge (8–17 dB) die durch per-Chunk
        OneTakeExport und variable Post-Processing entstehen.

        Returns:
            np.ndarray — vollständiges restauriertes Audio.
        """
        if not results:
            raise ValueError("Keine Chunk-Ergebnisse")

        total_samples = results[-1].end_sample
        # §v10.451: Stereo: (total_samples, kanäle), nicht (chunk_samples, total_samples)
        shape = (total_samples,) if results[0].audio.ndim == 1 else (total_samples, results[0].audio.shape[1])
        output = np.zeros(shape, dtype=np.float32)

        overlap_samples = int(self.config.overlap_s * sample_rate)
        # §v10.712: RMS-Matching-Fenster — letzte 500ms für Pegelvergleich
        match_samples = int(0.5 * sample_rate)

        for i, cr in enumerate(results):
            chunk_audio = cr.audio.astype(np.float64)
            out_start = cr.start_sample
            out_end = cr.end_sample

            if i > 0 and match_samples > 0:
                # §v10.712: RMS-Match vor Crossfade
                _prev_chunk = results[i - 1]
                _prev_tail = _prev_chunk.audio[-min(match_samples, _prev_chunk.audio.shape[0]) :]
                _curr_head = chunk_audio[: min(match_samples, chunk_audio.shape[0])]

                _rms_prev = float(np.sqrt(np.mean(np.square(_prev_tail)) + 1e-12))
                _rms_curr = float(np.sqrt(np.mean(np.square(_curr_head)) + 1e-12))

                if _rms_prev > 1e-10 and _rms_curr > 1e-10:
                    _rms_ratio = _rms_prev / _rms_curr
                    # Clamp ratio to avoid extreme gains
                    _rms_ratio = float(np.clip(_rms_ratio, 0.25, 4.0))
                    if abs(_rms_ratio - 1.0) > 0.02:  # > 0.17 dB difference
                        chunk_audio = chunk_audio * _rms_ratio
                        _rms_db = 20.0 * np.log10(_rms_ratio)
                        logger.info(
                            "§v10.712 RMS-Match Chunk %d→%d: Verhaeltnis=%.3f (%.1f dB)",
                            i - 1,
                            i,
                            _rms_ratio,
                            _rms_db,
                        )

            chunk_audio = chunk_audio.astype(np.float32)

            if i == 0:
                # Erster Chunk: direkt kopieren
                chunk_len = out_end - out_start
                if output.ndim == 1:
                    output[out_start:out_end] = chunk_audio[:chunk_len]
                else:
                    output[out_start:out_end, :] = chunk_audio[:chunk_len, :]
            else:
                # Crossfade mit vorherigem Chunk
                crossfade_len = min(overlap_samples, out_end - out_start)
                if crossfade_len > 0:
                    if output.ndim == 1:
                        prev_tail = output[out_start : out_start + crossfade_len]
                        curr_head = chunk_audio[:crossfade_len]
                    else:
                        prev_tail = output[out_start : out_start + crossfade_len, :]
                        curr_head = chunk_audio[:crossfade_len, :]
                    faded = self.crossfade(prev_tail, curr_head, crossfade_len)
                    if output.ndim == 1:
                        output[out_start : out_start + crossfade_len] = faded
                    else:
                        output[out_start : out_start + crossfade_len, :] = faded

                # Rest kopieren
                remaining = out_end - out_start - crossfade_len
                if remaining > 0:
                    if output.ndim == 1:
                        output[out_start + crossfade_len : out_end] = chunk_audio[crossfade_len:]
                    else:
                        output[out_start + crossfade_len : out_end, :] = chunk_audio[crossfade_len:, :]

        return cast(np.ndarray, output)

    @staticmethod
    def verify_crossfade(results: list, sample_rate: int) -> list[str]:
        """§v10.440: Prüft Crossfade-Qualität an Chunk-Grenzen.

        Misst RMS-Sprung zwischen letzten 100ms von Chunk N und
        ersten 100ms von Chunk N+1. Loggt Warnings bei >0.5dB.

        Returns:
            Liste von Warnungen (leer = perfekte Übergänge).
        """
        warnings: list[str] = []
        check_ms = 0.100  # 100ms Fenster
        check_samples = int(check_ms * sample_rate)
        for i in range(1, len(results)):
            _a = results[i - 1]
            _b = results[i]
            if not hasattr(_a, "audio") or not hasattr(_b, "audio"):
                continue
            _a_end = _a.audio[-check_samples:, :] if _a.audio.ndim >= 2 else _a.audio[-check_samples:]
            _b_start = _b.audio[:check_samples, :] if _b.audio.ndim >= 2 else _b.audio[:check_samples]
            _rms_a = float(np.sqrt(np.mean(np.square(_a_end)) + 1e-12))
            _rms_b = float(np.sqrt(np.mean(np.square(_b_start)) + 1e-12))
            _rms_db = 20.0 * np.log10(max(_rms_a, _rms_b) / max(min(_rms_a, _rms_b), 1e-12))
            if _rms_db > 0.5:
                _w = f"Crossfade-Warnung Chunk {i - 1}→{i}: RMS-Sprung {_rms_db:.1f} dB"
                logger.warning(_w)
                warnings.append(_w)
        if not warnings:
            logger.debug(f"Crossfade-Qualität: {len(results) - 1} Übergänge OK")
        return warnings
