"""Echtzeit-Preview — Streaming-Architektur mit <500ms Latenz.

Spec 11 §ROADMAP-5 Erweiterung.
Verarbeitet Audio in kleinen Chunks und streamt direkt zum Audio-Ausgang.
Ermöglicht sofortiges Hören ohne auf vollständige Pipeline zu warten.

Architektur:
  Audio → Chunked-Reader (256ms) → Mini-Pipeline → Ring-Buffer → PortAudio

Latenz-Budget:
  Chunk-Größe:  256ms (12288 samples @ 48kHz)
  Processing:   <200ms (Mini-Pipeline: Phase 03 + ComfortGuard)
  Buffer:        50ms
  ─────────────────────────────
  Total:        <500ms
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_MS: int = 256
CHUNK_SAMPLES: dict[int, int] = {48000: 12288, 44100: 11290}
RING_BUFFER_SIZE: int = 4  # Chunks im Voraus


class StreamingPreview:
    """Echtzeit-Audio-Preview mit Chunk-basierter Verarbeitung."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sr = sample_rate
        self.chunk_samples = CHUNK_SAMPLES.get(sample_rate, int(sample_rate * CHUNK_MS / 1000))
        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_chunk_ready: Callable | None = None

    @property
    def latency_ms(self) -> float:
        return (self.chunk_samples / self.sr) * 1000

    def start(self, audio: np.ndarray, on_chunk_ready: Callable[[np.ndarray], None] | None = None) -> None:
        """Startet Streaming-Preview im Hintergrund-Thread."""
        self._on_chunk_ready = on_chunk_ready
        self._running = True
        self._thread = threading.Thread(target=self._process_chunks, args=(audio,), daemon=True)
        self._thread.start()
        logger.info("Streaming-Preview gestartet: %dms Chunks, %dHz", CHUNK_MS, self.sr)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._buffer.clear()
        logger.info("Streaming-Preview gestoppt")

    def get_next_chunk(self) -> np.ndarray | None:
        """Holt nächsten fertigen Chunk (non-blocking)."""
        with self._buffer_lock:
            if self._buffer:
                return self._buffer.pop(0)
        return None

    def _process_chunks(self, audio: np.ndarray) -> None:
        """Hintergrund-Thread: Chunk-weise Mini-Pipeline."""
        arr = np.asarray(audio, dtype=np.float32)
        total = len(arr)
        pos = 0
        chunks_processed = 0

        while self._running and pos < total:
            end = min(pos + self.chunk_samples, total)
            chunk = arr[pos:end]

            try:
                # Mini-Pipeline: Denoise + ComfortGuard
                processed = self._mini_pipeline(chunk)
            except Exception:
                processed = chunk

            with self._buffer_lock:
                self._buffer.append(processed)
                while len(self._buffer) > RING_BUFFER_SIZE:
                    self._buffer.pop(0)

            if self._on_chunk_ready:
                try:
                    self._on_chunk_ready(processed)
                except Exception:
                    logger.debug("streaming_preview: chunk processing failed, skipping chunk", exc_info=True)

            pos = end
            chunks_processed += 1

        self._running = False
        logger.info("Streaming-Preview: %d Chunks verarbeitet", chunks_processed)

    def _mini_pipeline(self, chunk: np.ndarray) -> np.ndarray:
        """Minimale Pipeline für Echtzeit (<200ms Budget)."""
        if len(chunk) < 256:
            return chunk

        try:
            from backend.core.comfort_guard import apply_comfort_guard
            from backend.core.phases.phase_03_denoise import DenoisePhase

            p3 = DenoisePhase(sample_rate=self.sr)
            r3 = p3.process(chunk, sample_rate=self.sr, material_type="unknown")
            result = apply_comfort_guard(r3.audio, self.sr)
            return cast(np.ndarray, result.astype(np.float32))
        except Exception:
            # Fallback: leichter Lowpass mit Längen-Guard (§v10.103)
            from scipy.signal import butter
            from backend.core.audio_utils import safe_filtfilt

            b, a_coeff = butter(4, 16000 / (self.sr / 2), btype="low")
            return cast(np.ndarray, (safe_filtfilt(b, a_coeff, chunk.astype(np.float64)).astype(np.float32)))


def create_streaming_preview(sample_rate: int = 48000) -> StreamingPreview:
    return StreamingPreview(sample_rate)
