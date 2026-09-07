"""§v10.702 B3-Phase-2 Early-Merge Regressionstests (§G137).

Produktionsbefund (2026-09-07, Vinyl-Batch, Chunked-Streaming):
Der Early-Merge verglich STRING-Keys (``_b3_full_song_defect_types``) mit
ENUM-Keys (``defect_result.scores``) ohne Normalisierung. DefectType ist ein
Plain-Enum (kein str-Mixin) → Mengen-Differenz sah ALLE vorhandenen Typen als
„fehlend" an und überschrieb deren Scores (inkl. Locations) mit 0.06-Stubs
ohne Locations → Strength-Envelope degenerierte (μ=0.060 σ=0.000) → alle
Phasen liefen mit Floor-Stärke (No-Op-Kaskade, 42 Restdefekte unbehandelt).
"""

from __future__ import annotations

from backend.core.defect_scanner import (
    DefectAnalysisResult,
    DefectScore,
    DefectType,
    MaterialType,
)
from backend.core.unified_restorer_v3 import _b3_merge_full_song_defect_types


def _make_result(scores: dict) -> DefectAnalysisResult:
    return DefectAnalysisResult(
        material_type=MaterialType.VINYL,
        scores=scores,
        analysis_time_seconds=1.0,
        sample_rate=48_000,
        duration_seconds=224.3,
    )


def test_merge_does_not_overwrite_existing_scores_with_locations() -> None:
    """Regression: vorhandene ENUM-Key-Scores (mit Locations) bleiben unangetastet."""
    click_locations = [(1.0, 1.05), (5.0, 5.02), (90.0, 90.04)]
    result = _make_result(
        {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.708,
                confidence=0.99,
                locations=list(click_locations),
            ),
            DefectType.TRANSPORT_BUMP: DefectScore(
                defect_type=DefectType.TRANSPORT_BUMP,
                severity=0.638,
                confidence=0.99,
                locations=[(10.0, 10.25)],
            ),
        }
    )

    merged = _b3_merge_full_song_defect_types(
        result,
        {"clicks", "transport_bump", "hum"},
    )

    # Nur der wirklich fehlende Typ wird ergänzt
    assert merged == {"hum"}
    assert len(result.scores) == 3
    # Bestehende Scores behalten Severity, Konfidenz und Locations
    clicks = result.scores[DefectType.CLICKS]
    assert clicks.severity == 0.708
    assert clicks.confidence == 0.99
    assert clicks.locations == click_locations
    bump = result.scores[DefectType.TRANSPORT_BUMP]
    assert bump.locations == [(10.0, 10.25)]
    assert bump.severity == 0.638


def test_merge_adds_presence_stub_for_missing_type() -> None:
    result = _make_result({})

    merged = _b3_merge_full_song_defect_types(result, {"hum"})

    assert merged == {"hum"}
    stub = result.scores[DefectType.HUM]
    assert stub.severity == 0.06
    assert stub.confidence == 0.30
    assert stub.locations == []
    assert stub.metadata.get("source") == "b3_full_song_presence_scan"


def test_merge_noop_when_all_types_present() -> None:
    result = _make_result(
        {
            DefectType.WOW: DefectScore(
                defect_type=DefectType.WOW,
                severity=1.0,
                confidence=0.63,
                locations=[(3.0, 3.1)],
            ),
        }
    )

    merged = _b3_merge_full_song_defect_types(result, {"wow"})

    assert merged == set()
    assert len(result.scores) == 1
    assert result.scores[DefectType.WOW].severity == 1.0
    assert result.scores[DefectType.WOW].locations == [(3.0, 3.1)]


def test_merge_empty_input_is_noop() -> None:
    result = _make_result({DefectType.HUM: DefectScore(DefectType.HUM, 0.5, 0.5)})
    merged = _b3_merge_full_song_defect_types(result, set())
    assert merged == set()
    assert len(result.scores) == 1


def test_merge_unknown_type_uses_string_key_passthrough() -> None:
    result = _make_result({})
    merged = _b3_merge_full_song_defect_types(result, {"kein_echter_typ"})
    assert merged == {"kein_echter_typ"}
    assert "kein_echter_typ" in result.scores
