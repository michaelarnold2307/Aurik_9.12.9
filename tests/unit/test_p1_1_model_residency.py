"""P1-1 — Modell-Residency (§15.9): zweiter Lauf ohne Modell-Nachladen.

Befund (Session-Matrix): Jede Matrix-Zelle lud ONNX-Modelle neu; die Pipeline
hielt Prozess-Eigen-Caches (source_aware_restorer._ort_session_cache) und
direkte ort.InferenceSession-Aufrufe (aurik_sota_pipeline MelBandRoformer)
neben dem zentralen InferenceSessionManager (§15.9).

Fix (2026-09-08): Beide Pipeline-Ladepfade laufen jetzt über den zentralen
InferenceSessionManager (Residency je Session, LRU, Memory-Limit, MIGraphX-
Adapter). Diese Tests sichern die Invariante:
  1. Zwei Aufrufe desselben Ladepfads in einer Session ⇒ identisches
     Session-Objekt, _load_session wird nur EINMAL aufgerufen.
  2. Kein direkter ort.InferenceSession-Aufruf in den Pipeline-Hot-Pfaden.
"""

from __future__ import annotations

import pathlib
import re
from unittest.mock import MagicMock, patch

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_manager():
    """Isolierter Manager je Test (kein Singleton-Leak zwischen Tests)."""
    from backend.core.ml.session_manager import InferenceSessionManager

    InferenceSessionManager.reset_instance()
    yield
    InferenceSessionManager.reset_instance()


@pytest.mark.unit
def test_source_aware_second_run_reuses_central_session() -> None:
    """Zweiter Lauf in derselben Session: kein Modell-Nachladen (§15.9/P1-1)."""
    from backend.core.ml.session_manager import InferenceSessionManager
    from backend.core.source_aware_restorer import _get_ort_session

    with patch.object(
        InferenceSessionManager,
        "_load_session",
        return_value=(MagicMock(), 1.0),
    ) as load:
        first = _get_ort_session("models/demucs/htdemucs_6s.onnx")
        second = _get_ort_session("models/demucs/htdemucs_6s.onnx")
        assert first is second, "Residency verletzt: zweiter Aufruf lud neu"
        assert load.call_count == 1, f"_load_session wurde {load.call_count}× aufgerufen"


@pytest.mark.unit
def test_sota_pipeline_enhance_music_residency() -> None:
    """MelBandRoformer (860M) wird nur einmal je Session geladen (P1-1)."""
    from backend.core.aurik_sota_pipeline import _get_session_manager
    from backend.core.ml.session_manager import InferenceSessionManager

    with patch.object(
        InferenceSessionManager,
        "_load_session",
        return_value=(MagicMock(), 860.0),
    ) as load:
        mgr = _get_session_manager()
        s1 = mgr.acquire("melbandroformer", "models/melbandroformer/melbandroformer_optimized.onnx")
        s2 = mgr.acquire("melbandroformer", "models/melbandroformer/melbandroformer_optimized.onnx")
        assert s1 is s2
        assert load.call_count == 1


@pytest.mark.unit
def test_no_direct_ort_session_in_pipeline_hot_paths() -> None:
    """Pipeline-Ladepfade nutzen den zentralen Manager — kein Direkt-Laden."""
    for rel in (
        "backend/core/source_aware_restorer.py",
        "backend/core/aurik_sota_pipeline.py",
    ):
        src = (_REPO / rel).read_text(encoding="utf-8")
        assert not re.search(r"(?:ort|onnxruntime)\.InferenceSession\(", src), (
            f"{rel}: direkter InferenceSession-Aufruf — §15.9 verlangt den zentralen Manager"
        )
        assert "get_session_manager" in src, f"{rel}: ohne zentralen SessionManager"


@pytest.mark.unit
def test_manager_eviction_keeps_residency_bounded() -> None:
    """LRU begrenzt die Residency auf max_sessions (§15.9 Speicherpolitik)."""
    from backend.core.ml.session_manager import InferenceSessionManager

    mgr = InferenceSessionManager(max_sessions=2)
    with patch.object(
        InferenceSessionManager,
        "_load_session",
        return_value=(MagicMock(), 1.0),
    ):
        mgr.acquire("a", "a.onnx")
        mgr.acquire("b", "b.onnx")
        mgr.acquire("a", "a.onnx")  # a ist jetzt MRU
        mgr.acquire("c", "c.onnx")  # verdrängt b (LRU)
    assert "a" in mgr and "c" in mgr and "b" not in mgr
    assert mgr.get_active_count() == 2
