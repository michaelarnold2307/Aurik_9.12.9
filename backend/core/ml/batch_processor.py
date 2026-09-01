"""Batch-Processor mit Session-Recycling.
Spec 15 par 9.4. Nach N Tracks Sessions freigeben und neu laden.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    recycle_interval: int = 5
    max_memory_mb: float = 2048.0
    gc_between_tracks: bool = True
    timeout_per_track_s: float = 1800.0


@dataclass
class BatchTrackResult:
    track_index: int
    track_path: str
    success: bool
    quality_score: float = 0.0
    processing_time_s: float = 0.0
    error_message: str = ""
    memory_mb_peak: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    total_tracks: int
    successful: int = 0
    failed: int = 0
    total_time_s: float = 0.0
    avg_quality: float = 0.0
    tracks: list[BatchTrackResult] = field(default_factory=list)
    session_recycles: int = 0


class BatchProcessor:
    def __init__(self, process_fn: Callable, config: BatchConfig | None = None):
        self._process = process_fn
        self._config = config or BatchConfig()
        self._lock = threading.Lock()
        self._track_count = 0

    def process_tracks(self, track_paths: list[str], sample_rate: int = 48000) -> BatchResult:
        result = BatchResult(total_tracks=len(track_paths))
        t0 = time.monotonic()
        for idx, path in enumerate(track_paths):
            if self._track_count > 0 and self._track_count % self._config.recycle_interval == 0:
                self._recycle_sessions()
                result.session_recycles += 1
            tr = self._process_single(idx, path, sample_rate)
            result.tracks.append(tr)
            if tr.success:
                result.successful += 1
            else:
                result.failed += 1
            self._track_count += 1
        result.total_time_s = time.monotonic() - t0
        if result.successful > 0:
            result.avg_quality = sum(t.quality_score for t in result.tracks if t.success) / result.successful
        return result

    def _process_single(self, idx: int, path: str, sr: int) -> BatchTrackResult:
        t0 = time.monotonic()
        try:
            self._process(path, sr)  # Process and validate; return status only
            return BatchTrackResult(
                track_index=idx, track_path=path, success=True, processing_time_s=time.monotonic() - t0
            )
        except Exception as e:
            logger.warning(f"Batch track {idx} failed: {e}")
            return BatchTrackResult(
                track_index=idx,
                track_path=path,
                success=False,
                error_message=str(e),
                processing_time_s=time.monotonic() - t0,
            )
        finally:
            if self._config.gc_between_tracks:
                gc.collect()

    def _recycle_sessions(self):
        try:
            from backend.core.ml.session_manager import get_session_manager

            mgr = get_session_manager()
            mgr.clear()
            gc.collect()
            logger.info("Batch: sessions recycled after %d tracks", self._track_count)
        except Exception as e:
            logger.debug("Session recycle skipped: %s", e)


def get_batch_processor(process_fn: Callable, **kwargs) -> BatchProcessor:
    return BatchProcessor(process_fn, BatchConfig(**kwargs))
