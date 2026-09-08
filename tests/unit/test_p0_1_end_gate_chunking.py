"""§P0-1 Song-Ebene-Analytik — Entscheidungsmatrix der End-Gate-Kaskade
+ Einladungs-Gate-Messung (Block b) + measure_all-Skip (Block a1).

Im Chunked-Pfad läuft die End-Gate-Recovery-Kaskade nur auf dem letzten
Chunk; die Stufe-2-Nachbehandlung (m1b) führt sie einmal auf dem
assemblierten Song aus. Außerhalb des Chunked-Pfads bleibt das Verhalten
bit-identisch zum bisherigen Stand.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3, _should_run_end_gate_cascade


class _FakeChecker:
    """Minimaler measure_all-Double — kein DSP, nur Zähl-Semantik."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def measure_all(self, audio, sample_rate, *, reference=None, material_type=None, panns_singing=0.0):
        self.calls.append((len(audio), sample_rate, reference, material_type, panns_singing))
        return {"brillanz": 0.5}


@pytest.mark.parametrize(
    ("violations", "tail_skip", "chunked_last", "expected"),
    [
        # Außerhalb des Chunked-Pfads: Verhalten unverändert
        ([], False, False, False),
        (["brillanz"], False, False, True),
        (["brillanz"], False, True, True),
        # Chunked-Pfad: Kaskade nur auf dem letzten Chunk
        ([], True, False, False),
        ([], True, True, False),
        (["brillanz"], True, False, False),  # mittlere Chunks: übersprungen
        (["brillanz"], True, True, True),  # letzter Chunk: Safety-Netz
        (["natuerlichkeit", "transient_energie"], True, False, False),
        (["natuerlichkeit", "transient_energie"], True, True, True),
    ],
)
def test_end_gate_cascade_decision_matrix(
    violations: list[str],
    tail_skip: bool,
    chunked_last: bool,
    expected: bool,
) -> None:
    assert (
        _should_run_end_gate_cascade(violations, tail_skip, chunked_last) is expected
    )


def test_measure_goals_for_tail_chunk_skip() -> None:
    """§P0-1 a1: Nicht-letzter Chunk überspringt measure_all; letzter misst."""
    checker = _FakeChecker()
    restorer = UnifiedRestorerV3.__new__(UnifiedRestorerV3)  # ohne schwere Init
    restorer._restoration_context = {}
    audio = [0.0] * 1000

    skipped = restorer._measure_goals_for_tail(
        checker, audio, 48000, None, "vinyl",
        chunked_tail_skip=True, chunked_last=False,
    )
    assert skipped == {}
    assert checker.calls == []

    measured = restorer._measure_goals_for_tail(
        checker, audio, 48000, None, "vinyl",
        chunked_tail_skip=True, chunked_last=True,
    )
    assert measured == {"brillanz": 0.5}
    assert len(checker.calls) == 1
    assert checker.calls[0][1] == 48000 and checker.calls[0][3] == "vinyl"


def test_measure_goals_for_tail_normal_path() -> None:
    """Außerhalb des Chunked-Pfads misst jeder Aufruf (Verhalten unverändert)."""
    checker = _FakeChecker()
    restorer = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
    restorer._restoration_context = {}
    result = restorer._measure_goals_for_tail(checker, [0.0] * 100, 48000, None, "cd_digital")
    assert result == {"brillanz": 0.5}
    assert len(checker.calls) == 1


def test_inviting_gate_skip_returns_none() -> None:
    """§P0-1 (b): skip=True → keine Messung, kein Import-Aufwand."""
    restorer = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
    restorer._restoration_context = {}
    assert restorer._run_inviting_gate_measure(np.zeros(100, dtype=np.float32), 48000, None, skip=True) is None


def test_inviting_gate_measure_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """§P0-1 (b): Messpfad liefert das Context-Dict (Modul-Stub)."""
    mod = types.ModuleType("backend.core.inviting_sound_gate")

    class _Res:
        passed = True
        max_asper_in_voice = 0.01
        sharpness_jump_max = 0.1
        details = {"sharpness_jump_raw_max": 0.2, "exempted_jumps": 3, "failures": []}
        fatigue_abort = False
        n_windows = 4

    mod.check_inviting_gate = lambda *a, **k: _Res()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "backend.core.inviting_sound_gate", mod)
    restorer = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
    restorer._restoration_context = {"singing_mask": None}
    ctx = restorer._run_inviting_gate_measure(np.zeros(1000, dtype=np.float32), 48000, None)
    assert ctx is not None
    assert ctx["passed"] is True
    assert ctx["n_windows"] == 4
    assert ctx["exempted_jumps"] == 3
