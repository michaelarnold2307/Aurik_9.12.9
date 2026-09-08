"""§P0-1 Song-Ebene-Analytik — Entscheidungsmatrix der End-Gate-Kaskade.

Im Chunked-Pfad läuft die End-Gate-Recovery-Kaskade nur auf dem letzten
Chunk; die Stufe-2-Nachbehandlung (m1b) führt sie einmal auf dem
assemblierten Song aus. Außerhalb des Chunked-Pfads bleibt das Verhalten
bit-identisch zum bisherigen Stand.
"""

from __future__ import annotations

import pytest

from backend.core.unified_restorer_v3 import _should_run_end_gate_cascade


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
