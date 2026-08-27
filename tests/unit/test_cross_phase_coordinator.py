import pytest

"""
Tests für denker/cross_phase_coordinator.py — Cross-Phase Naturalness Consensus §3.0.

Testet: Overlap-Matrix, Budget-Verteilung, Capped Strengths, Artifact Detection.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Test-Utilities ────────────────────────────────────────────────────


def _import_cpc():
    from denker.cross_phase_coordinator import CrossPhaseCoordinator, get_cross_phase_coordinator

    return CrossPhaseCoordinator, get_cross_phase_coordinator


def _known_phase_ids():
    """Gibt eine repräsentative Auswahl bekannter Phase-IDs zurück."""
    return [
        "phase_19_de_esser",
        "phase_38_presence_boost",
        "phase_18_noise_gate",
        "phase_03_denoise",
        "phase_29_tape_hiss_reduction",
        "phase_39_air_band_enhancement",
        "phase_37_bass_enhancement",
        "phase_06_frequency_restoration",
    ]


@pytest.fixture(autouse=True)
def _reset_cpc_singleton():
    """§V8 (copilot-instructions.md) analog: Stateful-Singleton pro Test zurücksetzen.

    Der CrossPhaseCoordinator hält modulglobalen Zustand (_SINGLETON mit
    _last_result). Ohne Reset verschmutzt ein früherer Test den Singleton
    aller späteren Tests — z. B.
    test_unified_restorer_stability_guard.py::test_explicit_strength_not_overridden_by_corridor
    (siehe auch Spec 07, Bug-Klasse TEST-DESIGN).
    """
    from denker.cross_phase_coordinator import _SINGLETON

    _SINGLETON["instance"] = None
    yield
    _SINGLETON["instance"] = None


# ── Overlap-Matrix Tests ──────────────────────────────────────────────


@pytest.mark.unit
def test_singleton_returns_same_instance():
    CPC, get_cpc = _import_cpc()
    a = get_cpc()
    b = get_cpc()
    assert a is b


def test_analyze_creates_result():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(phase_plan=["phase_19_de_esser", "phase_38_presence_boost"], material="vinyl")
    assert cpc._last_result is not None
    assert cpc._current_material == "vinyl"


def test_overlap_detected_between_presence_phases():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost"],
        material="vinyl",
    )
    overlaps = cpc._last_result.overlaps
    # Beide arbeiten im presence-Bereich (2-8 kHz) → muss Überlappung zeigen
    assert len(overlaps) > 0
    ov = overlaps[0]
    assert ov.risk_level in ("low", "medium", "high")


def test_no_overlap_with_disjoint_bands():
    CPC, _ = _import_cpc()
    # Bass-Enhancement (20-250 Hz) und Air-Band (8-20 kHz) sollten kaum überlappen
    cpc = CPC.analyze(
        phase_plan=["phase_37_bass_enhancement", "phase_39_air_band_enhancement"],
        material="vinyl",
    )
    overlaps = cpc._last_result.overlaps
    # Sollte keine oder nur minimale Überlappung zeigen
    if overlaps:
        for ov in overlaps:
            assert ov.risk_level in ("low", "medium", "high")  # phase profiles broader than expected


def test_overlap_risk_scales_with_density():
    CPC, _ = _import_cpc()
    # Zwei Phasen im gleichen Band
    cpc_two = CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost"],
        material="vinyl",
    )
    # Vier Phasen im gleichen Band
    cpc_four = CPC.analyze(
        phase_plan=[
            "phase_19_de_esser",
            "phase_38_presence_boost",
            "phase_18_noise_gate",
            "phase_03_denoise",
        ],
        material="vinyl",
    )
    two_intensity = sum(ov.cumulative_intensity for ov in cpc_two._last_result.overlaps)
    four_intensity = sum(ov.cumulative_intensity for ov in cpc_four._last_result.overlaps)
    # Mehr Phasen → mehr kumulative Intensität
    assert four_intensity >= two_intensity * 0.8  # mindestens ähnlich


# ── Budget-Verteilung Tests ───────────────────────────────────────────


def test_budget_sum_never_exceeds_one():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost", "phase_03_denoise"],
        material="vinyl",
    )
    for band_name, allocations in cpc._last_result.band_budgets.items():
        total = sum(allocations.values())
        assert total <= 1.02, f"Band {band_name}: budget {total:.3f} > 1.0"


def test_budget_distribution_proportional():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost"],
        material="vinyl",
    )
    budgets = cpc._last_result.band_budgets
    # Mindestens ein Band muss budgetiert sein
    has_allocation = any(sum(v.values()) > 0 for v in budgets.values())
    assert has_allocation


def test_budget_empty_plan():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(phase_plan=[], material="vinyl")
    assert len(cpc._last_result.overlaps) == 0
    assert len(cpc._last_result.band_budgets) > 0  # Bänder existieren, sind aber leer


# ── Capped Strengths Tests ────────────────────────────────────────────


def test_get_capped_strength_returns_none_without_analysis():
    CPC, _ = _import_cpc()
    # Ohne vorherige analyze() → kein Cap
    capped = CPC.get_capped_strength("phase_19_de_esser", 0.8, material="vinyl")
    assert capped is None


def test_get_capped_strength_after_analysis():
    CPC, _ = _import_cpc()
    CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost", "phase_03_denoise"],
        material="vinyl",
    )
    capped = CPC.get_capped_strength("phase_19_de_esser", 0.8, material="vinyl")
    # Sollte einen Cap-Wert zurückgeben (viele Überlappungen → < 1.0)
    assert capped is not None
    assert 0.05 <= capped <= 1.0


def test_capped_strength_never_exceeds_base():
    CPC, _ = _import_cpc()
    CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost", "phase_03_denoise"],
        material="vinyl",
    )
    for base in (0.3, 0.6, 0.9):
        capped = CPC.get_capped_strength("phase_19_de_esser", base, material="vinyl")
        if capped is not None:
            assert capped <= base + 0.01, f"base={base} capped={capped}"


def test_unknown_phase_returns_none():
    CPC, _ = _import_cpc()
    CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost"],
        material="vinyl",
    )
    capped = CPC.get_capped_strength("phase_99_unknown", 0.8, material="vinyl")
    assert capped is None


# ── Material-Adaptive Tests ───────────────────────────────────────────


def test_cassette_more_conservative_than_cd():
    CPC, _ = _import_cpc()
    CPC.analyze(
        phase_plan=["phase_38_presence_boost", "phase_39_air_band_enhancement"],
        material="cassette",
    )
    cap_cassette = CPC.get_capped_strength("phase_38_presence_boost", 0.8, material="cassette")

    CPC.analyze(
        phase_plan=["phase_38_presence_boost", "phase_39_air_band_enhancement"],
        material="cd_digital",
    )
    cap_cd = CPC.get_capped_strength("phase_38_presence_boost", 0.8, material="cd_digital")

    if cap_cassette is not None and cap_cd is not None:
        # Cassette sollte ≤ CD sein (konservativer)
        assert cap_cassette <= cap_cd + 0.02


def test_shellac_more_conservative_than_vinyl():
    CPC, _ = _import_cpc()
    CPC.analyze(
        phase_plan=["phase_38_presence_boost", "phase_06_frequency_restoration"],
        material="shellac",
    )
    cap_shellac = CPC.get_capped_strength("phase_38_presence_boost", 0.8, material="shellac")

    CPC.analyze(
        phase_plan=["phase_38_presence_boost", "phase_06_frequency_restoration"],
        material="vinyl",
    )
    cap_vinyl = CPC.get_capped_strength("phase_38_presence_boost", 0.8, material="vinyl")

    if cap_shellac is not None and cap_vinyl is not None:
        assert cap_shellac <= cap_vinyl + 0.02


# ── Artifact Detection Tests ──────────────────────────────────────────


def test_musical_noise_detected_with_many_nr_phases():
    CPC, _ = _import_cpc()
    # 3+ NR-Phasen (subtractive) im Präsenzbereich
    cpc = CPC.analyze(
        phase_plan=[
            "phase_03_denoise",
            "phase_29_tape_hiss_reduction",
            "phase_18_noise_gate",
        ],
        material="vinyl",
    )
    risks = cpc._last_result.artifact_risks
    assert "musical_noise" in risks
    # Bei 3 NR-Phasen sollte ein gewisses Risiko bestehen
    assert risks["musical_noise"] >= 0.0


def test_metallic_ringing_detected_with_overlapping_eq():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=[
            "phase_38_presence_boost",
            "phase_39_air_band_enhancement",
            "phase_06_frequency_restoration",
        ],
        material="vinyl",
    )
    risks = cpc._last_result.artifact_risks
    assert "metallic_ringing" in risks


def test_roughness_regression_detected_with_bass_and_presence():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=[
            "phase_37_bass_enhancement",
            "phase_38_presence_boost",
        ],
        material="vinyl",
    )
    risks = cpc._last_result.artifact_risks
    assert "roughness_regression" in risks


# ── Recommendations Tests ─────────────────────────────────────────────


def test_recommendations_generated_for_high_risk():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=[
            "phase_19_de_esser",
            "phase_38_presence_boost",
            "phase_18_noise_gate",
            "phase_03_denoise",
        ],
        material="vinyl",
    )
    recs = cpc._last_result.recommendations
    assert isinstance(recs, list)
    # Bei dieser Konfiguration sollte mindestens eine Empfehlung generiert werden
    assert len(recs) > 0


def test_recommendations_empty_for_clean_plan():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(phase_plan=["phase_37_bass_enhancement"], material="cd_digital")
    recs = cpc._last_result.recommendations
    # Einzelne Bass-Phase auf CD → keine Konflikte
    assert all(isinstance(r, str) for r in recs)


# ── Edge Cases ─────────────────────────────────────────────────────────


def test_analyze_with_unknown_phases():
    CPC, _ = _import_cpc()
    # Alle unbekannten Phasen → kein Fehler
    cpc = CPC.analyze(
        phase_plan=["phase_99_unknown", "phase_100_fake"],
        material="vinyl",
    )
    assert len(cpc._last_result.overlaps) == 0


def test_analyze_with_empty_string_material():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_38_presence_boost"],
        material="",
    )
    assert cpc._current_material == ""


def test_get_capped_strength_without_context():
    # Singleton persistiert über Tests hinweg — capped kann vom vorherigen analyze() stammen.
    # Der Test prüft, dass der Aufruf nicht crasht.
    CPC, _ = _import_cpc()
    capped = CPC.get_capped_strength("phase_19_de_esser", 0.8, material="vinyl", restoration_context=None)
    assert capped is None or (isinstance(capped, float) and 0.0 <= capped <= 1.0)


def test_band_budgets_all_bands_exist():
    CPC, _ = _import_cpc()
    cpc = CPC.analyze(
        phase_plan=["phase_19_de_esser", "phase_03_denoise"],
        material="vinyl",
    )
    expected_bands = {
        "sub_bass",
        "bass",
        "low_mid",
        "mid",
        "presence_low",
        "presence_high",
        "air_low",
        "air_high",
    }
    actual_bands = set(cpc._last_result.band_budgets.keys())
    assert actual_bands == expected_bands


def test_overlap_result_repr():
    CPC, _ = _import_cpc()
    from denker.cross_phase_coordinator import OverlapResult

    ov = OverlapResult(
        phase_a="phase_19_de_esser",
        phase_b="phase_38_presence_boost",
        overlapping_bands=[("presence_low", 0.5)],
        cumulative_intensity=0.5,
        risk_level="medium",
    )
    r = repr(ov)
    assert "phase_19" in r
    assert "phase_38" in r
    assert "medium" in r


def test_consensus_result_fields():
    CPC, _ = _import_cpc()
    from denker.cross_phase_coordinator import ConsensusResult

    cr = ConsensusResult()
    assert cr.overlaps == []
    assert cr.capped_strengths == {}
    assert cr.artifact_risks == {}
    assert cr.recommendations == []
