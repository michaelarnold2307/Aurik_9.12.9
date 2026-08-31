"""backend/core/memmap_pool.py — §v10.320 Tempfile-backed numpy.memmap pool.

Replaces in-memory np.zeros/np.empty for large (>= 64 MB) temporary arrays.
Arrays are backed by temp files in /tmp, paged in on demand via mmap,
and automatically cleaned up on del or pool destruction.

Usage::

    from backend.core.memmap_pool import get_memmap_pool

    pool = get_memmap_pool()
    arr = pool.allocate(shape=(2, 10765313), dtype=np.float32)
    # ... use arr like a normal numpy array ...
    pool.free(arr)  # or arr._mmap_path for manual cleanup

Design goals:
- Zero-copy: numpy operations work directly on mmap'd pages
- Auto-cleanup: temp files deleted on pool destruction
- Transparent: returns np.memmap, compatible with all numpy operations
- Bounded: max 4 temp files alive at once (LRU eviction)
"""

from __future__ import annotations

import atexit
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# §v10.320: Arrays >= 64 MB via memmap statt Heap-Allokation.
# Schwelle aus Benchmark: 64 MB entspricht ~8s Stereo 48kHz float32.
# Darunter lohnt sich der mmap-Overhead nicht.
_MEMMAP_THRESHOLD_BYTES: int = 64 * 1024 * 1024  # 64 MB


class MemmapPool:
    """LRU-bounded pool of tempfile-backed numpy.memmap arrays."""

    def __init__(self, max_files: int = 4, tmp_dir: str | None = None) -> None:
        self._lock = threading.Lock()
        self._max_files = max_files
        self._tmp_dir = tmp_dir or tempfile.gettempdir()
        self._files: OrderedDict[str, float] = OrderedDict()  # path → last_access_ts
        self._closed = False
        atexit.register(self.close)

    def allocate(self, shape: tuple[int, ...], dtype: type = np.float32) -> np.memmap:
        """Allocate a memory-mapped array backed by a temp file.

        Args:
            shape: Array shape (same as np.zeros).
            dtype: Data type (default float32).

        Returns:
            np.memmap array. Has an extra attribute _mmap_path for cleanup.
        """
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        with self._lock:
            if self._closed:
                raise RuntimeError("MemmapPool is closed")

            # Evict oldest if at capacity
            while len(self._files) >= self._max_files:
                _oldest = min(self._files, key=lambda k: self._files[k])
                self._evict_file(_oldest)

            _fd, _path = tempfile.mkstemp(suffix=".aurik_memmap", dir=self._tmp_dir)
            os.close(_fd)

        try:
            mm = np.memmap(_path, dtype=dtype, mode="w+", shape=shape)
            # Store path for cleanup
            mm._mmap_path = _path  # type: ignore[attr-defined]
            with self._lock:
                self._files[_path] = time.monotonic()
            logger.debug("MemmapPool: allocated %s (%.1f MB)", shape, n_bytes / 1e6)
            return mm
        except Exception:
            try:
                os.unlink(_path)
            except OSError:
                logger.debug("MemmapPool: Konnte %s nicht loeschen", _path, exc_info=True)
            raise

    def free(self, mm: np.memmap) -> None:
        """Free a memmap array and its backing temp file."""
        _path = getattr(mm, "_mmap_path", None)
        if _path is None:
            return
        try:
            mm._mmap_path = None  # type: ignore[attr-defined]
            del mm
        except Exception:
            logger.debug("memmap_pool.py:101: Silent exception absorbed", exc_info=True)
        with self._lock:
            if _path in self._files:
                self._evict_file(_path)

    def _evict_file(self, path: str) -> None:
        """Remove a temp file from disk and registry."""
        try:
            os.unlink(path)
        except OSError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        self._files.pop(path, None)

    def close(self) -> None:
        """Close pool: delete ALL temp files. Called at exit."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _path in list(self._files):
                self._evict_file(_path)
            logger.debug("MemmapPool: closed — all temp files cleaned up")

    @property
    def active_files(self) -> int:
        with self._lock:
            return len(self._files)


# ── Singleton ──────────────────────────────────────────────────────
_pool: MemmapPool | None = None
_pool_lock = threading.Lock()


def get_memmap_pool() -> MemmapPool:
    """Return the global MemmapPool singleton."""
    global _pool  # pylint: disable=global-statement
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = MemmapPool()
    return _pool


def allocate_if_large(shape: tuple[int, ...], dtype: type = np.float32) -> np.ndarray:
    """Allocate via memmap if size >= 64 MB, else via np.empty.

    Transparent drop-in replacement for np.empty() for large arrays.
    Small arrays use normal heap allocation.
    """
    n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if n_bytes >= _MEMMAP_THRESHOLD_BYTES:
        return get_memmap_pool().allocate(shape, dtype)
    return cast(np.ndarray, (np.empty(shape, dtype=dtype)))
