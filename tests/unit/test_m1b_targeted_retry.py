"""§P1-7 — m1b: Gezielte Stufe-2-Nachbehandlung hörbarer Restdefekte.

Befund 2026-09-08: §v10.703 Defekt-Countdown 46 gefunden → 42 hörbar →
3 behoben → 42 über Schwelle. Die m1b-Queue wurde nur in die GUI-KMV
gestellt; im Headless-Flow konsumierte niemand sie. _run_m1b_targeted_retry()
führt die sicher zugeordneten Retry-Phasen intern einmalig aus.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


class _FakeRestorer(UnifiedRestorerV3):
    def __init__(self) -> None:
        self._restoration_context: dict = {}
        self._m1b_pass_active = False
        self._graceful_stop_event = None
        self._pipeline_calls: list[dict] = []

    def _execute_pipeline(self, *args, **kwargs) -> tuple:
        self._pipeline_calls.append(kwargs)
        sel = kwargs.get("selected_phases") or []
        if not sel:
            return np.zeros(10, dtype=np.float32), [], [], []
        return np.ones(10, dtype=np.float32), list(sel), [], []


def _call(r: _FakeRestorer, types: list[str]) -> np.ndarray | None:
    return r._run_m1b_targeted_retry(
        np.zeros(10, dtype=np.float32),
        48000,
        "vinyl",
        None,
        types,
        None,
        [],
        {},
        None,
        {},
        None,
        None,
        None,
        0.7,
    )


def test_retry_runs_mapped_phases_only() -> None:
    r = _FakeRestorer()
    out = _call(r, ["hum", "clicks", "hiss"])
    assert out is not None
    assert list(r._pipeline_calls[0]["selected_phases"]) == [
        "phase_02_hum_removal",
        "phase_01_click_removal",
        "phase_29_tape_hiss_reduction",
    ]
    # Chunk-Verschiebung wurde für den song-globalen Pass entfernt
    assert "chunk_start_sample" not in r._restoration_context
    assert r._m1b_pass_active is False  # finally hat zurückgesetzt


def test_forbidden_phases_excluded() -> None:
    """§0a: Verbotene Phasen dürfen nie über die Retry-Map laufen."""
    r = _FakeRestorer()
    out = _call(r, ["hum", "reverb_excess"])
    assert out is not None
    assert "phase_21_exciter" not in r._pipeline_calls[0]["selected_phases"]
    assert list(r._pipeline_calls[0]["selected_phases"]) == [
        "phase_02_hum_removal",
        "phase_49_advanced_dereverb",
    ]


def test_unmapped_types_do_not_run() -> None:
    r = _FakeRestorer()
    out = _call(r, ["bandwidth_loss"])  # Physical-Cap, keine Phase
    assert out is None
    assert r._pipeline_calls == []


def test_reentry_guard() -> None:
    r = _FakeRestorer()
    r._m1b_pass_active = True
    out = _call(r, ["hum"])
    assert out is None
    assert r._pipeline_calls == []


def test_no_execution_returns_none() -> None:
    r = _FakeRestorer()

    def _noop(*args, **kwargs):  # Pipeline führt nichts aus
        return np.zeros(10, dtype=np.float32), [], [], []

    r._execute_pipeline = _noop  # type: ignore[method-assign]
    out = _call(r, ["hum"])
    assert out is None


def test_chunk_shift_restored_after_pass() -> None:
    r = _FakeRestorer()
    r._restoration_context["chunk_start_sample"] = 48000
    _call(r, ["hum"])
    assert r._restoration_context["chunk_start_sample"] == 48000
