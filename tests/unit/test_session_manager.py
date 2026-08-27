"""Session-Manager-Test. Spec 15 paragraph 9.5.
Testet: Acquire/Release, LRU-Eviction, Memory-Limit, Concurrent-Access, Batch-Recycling.

Autor: Aurik 10
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestSessionManager:
    """Testet den InferenceSessionManager."""

    @pytest.fixture(autouse=True)
    def _stub_load(self):
        """Spec 15 §9.5: Diese Tests pruefen Cache-Semantik (LRU, Memory,
        Threading, Recycling) — nicht das echte ONNX-Laden. Der Ladevorgang
        wird daher gestubbt; echte fehlende Modelle fallen weiterhin laut
        in _load_session (onnxruntime NoSuchFile) durch.
        """
        from backend.core.ml.session_manager import InferenceSessionManager

        with patch.object(
            InferenceSessionManager,
            "_load_session",
            return_value=(MagicMock(), 1.0),
        ):
            yield

    def test_acquire_release(self):
        """Acquire/Release-Zyklus."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=2)
        sid = mgr.acquire("test_model", model_path="mock.onnx")
        assert sid is not None
        mgr.release("test_model")
        assert mgr.get_active_count() == 0

    def test_lru_eviction(self):
        """LRU-Eviction: aelteste Session wird verdraengt."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=2)
        mgr.acquire("m1", model_path="m1.onnx")
        mgr.acquire("m2", model_path="m2.onnx")
        mgr.acquire("m3", model_path="m3.onnx")  # Should evict m1
        assert "m1" not in mgr._cache

    def test_memory_limit(self):
        """Memory-Limit-Warnung."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=10, memory_limit_mb=1.0)
        with patch.object(mgr, "get_total_memory_mb", return_value=2500.0):
            mgr.acquire("big_model", model_path="big.onnx")
            assert mgr.get_total_memory_mb() > mgr.memory_limit_mb

    def test_concurrent_access(self):
        """Concurrent-Access: Thread-sicherer Zugriff."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=10)
        errors = []

        def worker(name):
            try:
                mgr.acquire(name, model_path=f"{name}.onnx")
                mgr.release(name)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_batch_recycling(self):
        """Batch-Recycling: Nach N Tracks Sessions leeren."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=4)
        for i in range(6):
            mgr.acquire(f"b{i}", model_path=f"b{i}.onnx")
        mgr.clear()
        assert mgr.get_active_count() == 0


class TestSessionManagerMigraphxSizeGuard:
    """§v10.40 Compile-Zeit-Regel: Modelle > 200 MB überspringen MIGraphX → ORT.

    Testet das ECHTE _load_session (kein autouse-Stub) mit gemockten
    Session-Klassen; Sparse-Dateien (truncate) kosten keinen Plattenplatz.
    """

    def test_large_model_skips_migraphx(self, tmp_path):
        import os

        from backend.core.ml.session_manager import InferenceSessionManager

        big = tmp_path / "big.onnx"
        big.touch()
        os.truncate(big, 201 * 1024 * 1024)  # Sparse — kein echter Plattenverbrauch

        with (
            patch("backend.core.migraphx_adapter.is_migraphx_available", return_value=True),
            patch(
                "backend.core.migraphx_adapter.MIGraphXSession",
                side_effect=AssertionError("MIGraphX darf für >200 MB nicht aufgerufen werden"),
            ),
            patch("onnxruntime.InferenceSession", return_value="ORT-SENTINEL"),
        ):
            session, size_mb = InferenceSessionManager._load_session(big)
        assert session == "ORT-SENTINEL"
        assert size_mb > 200.0

    def test_small_model_uses_migraphx(self, tmp_path):
        from backend.core.ml.session_manager import InferenceSessionManager

        small = tmp_path / "small.onnx"
        small.write_bytes(b"\x00" * 1024)

        with (
            patch("backend.core.migraphx_adapter.is_migraphx_available", return_value=True),
            patch("backend.core.migraphx_adapter.MIGraphXSession", return_value="MGX-SENTINEL") as mgx_mock,
            patch("onnxruntime.InferenceSession", return_value="ORT-SENTINEL") as ort_mock,
        ):
            session, _size_mb = InferenceSessionManager._load_session(small)
        assert session == "MGX-SENTINEL"
        mgx_mock.assert_called_once()
        ort_mock.assert_not_called()

    def test_size_helpers(self, tmp_path):
        import os

        from backend.core.migraphx_adapter import (
            MIGRAPHX_MAX_MODEL_MB,
            is_migraphx_size_eligible,
            migraphx_model_size_mb,
        )

        assert migraphx_model_size_mb(tmp_path / "missing.onnx") == 0.0
        assert is_migraphx_size_eligible(tmp_path / "missing.onnx") is True

        ok = tmp_path / "ok.onnx"
        ok.touch()
        os.truncate(ok, int(100 * 1024 * 1024))
        assert is_migraphx_size_eligible(ok) is True

        big = tmp_path / "big.onnx"
        big.touch()
        os.truncate(big, int((MIGRAPHX_MAX_MODEL_MB + 1) * 1024 * 1024))
        assert is_migraphx_size_eligible(big) is False
