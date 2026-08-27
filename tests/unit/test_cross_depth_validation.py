"""§G86 Cross-Depth-Validierung: Alle Schwellwerte für depth 1–5.

Validiert dass JEDER depth-abhängige Schwellwert für alle Chain-Tiefen
physikalisch plausible Werte liefert. Verhindert Regressionen wenn
die chain_factor-Logik geändert wird.
"""

import numpy as np
import pytest

from backend.core.cumulative_interaction_guard import (
    CumulativeInteractionGuard,
    InteractionGuardState,
)
from backend.core.per_phase_musical_goals_gate import _get_adaptive_threshold
from backend.core.spec_constitution import get_constitution

# ── CIG GDD Threshold ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "depth,expected_factor",
    [
        (1, 1.00),
        (2, 1.00),
        (3, 1.25),
        (4, 1.50),
        (5, 1.75),
    ],
)
def test_gdd_chain_factor_scales_correctly(depth, expected_factor):
    """§v10.120: GDD chain_factor = 1.0 + max(0, depth-2) * 0.25."""
    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.transfer_chain_depth = depth
    state.restorability_score = 100.0
    state.material_type = "cd_digital"  # kein Analog-Faktor

    base = guard._compute_gdd_threshold(
        "phase_29_tape_hiss_reduction",
        InteractionGuardState(transfer_chain_depth=1, restorability_score=100.0, material_type="cd_digital"),
    )
    actual = guard._compute_gdd_threshold("phase_29_tape_hiss_reduction", state)
    factor = actual / base

    assert factor == pytest.approx(expected_factor, abs=0.01), (
        f"depth={depth}: expected {expected_factor}×, got {factor:.3f}×"
    )


@pytest.mark.parametrize(
    "depth,min_gdd_ms",
    [
        (1, 5.0),
        (2, 5.0),
        (3, 6.0),
        (4, 7.5),
        (5, 8.0),
    ],
)
def test_gdd_threshold_never_below_minimum(depth, min_gdd_ms):
    """GDD-Schwelle darf nie unter ein physikalisches Minimum fallen."""
    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.transfer_chain_depth = depth
    state.restorability_score = 100.0
    state.material_type = "cd_digital"

    gdd = guard._compute_gdd_threshold("phase_07_harmonic_restoration", state)
    assert abs(gdd) >= min_gdd_ms, f"depth={depth}: GDD={abs(gdd):.1f}ms < {min_gdd_ms}ms"


# ── Constitution artifact_freedom ───────────────────────────────────────────


@pytest.mark.parametrize(
    "depth,threshold",
    [
        (1, 0.95),
        (2, 0.88),
        (3, 0.80),
        (4, 0.70),
        (5, 0.70),
    ],
)
def test_constitution_artifact_freedom_threshold(depth, threshold):
    """§v10.119: artifact_freedom_min pro Depth."""
    const = get_constitution()
    # af knapp über threshold → kein Veto
    r_ok = const.check_paragraph_zero(None, 48000, artifact_freedom=threshold, hpi=0.6, chain_depth=depth)  # type: ignore[arg-type]
    veto_ok = [v for v in r_ok if "VETO" in v and "artifact_freedom" in v]
    assert not veto_ok, f"depth={depth}: af={threshold} sollte OK sein, bekam VETO: {veto_ok}"

    # af knapp unter threshold → Veto
    r_veto = const.check_paragraph_zero(None, 48000, artifact_freedom=threshold - 0.01, hpi=0.6, chain_depth=depth)  # type: ignore[arg-type]
    veto = [v for v in r_veto if "VETO" in v and "artifact_freedom" in v]
    assert veto, f"depth={depth}: af={threshold - 0.01:.2f} sollte VETO auslösen"


def test_constitution_artifact_freedom_restorability_modulation():
    """§v10.119: restorability < 50 senkt die Schwelle material-adaptiv (Spec 24)."""
    const = get_constitution()
    # depth=1: Basis 0.95. rs=40 → 0.90. af=0.92 → OK.
    r_ok = const.check_paragraph_zero(
        None,
        48000,
        artifact_freedom=0.92,
        hpi=0.6,
        chain_depth=1,
        restorability=40.0,  # type: ignore[arg-type]
    )
    veto_ok = [v for v in r_ok if "VETO" in v and "artifact_freedom" in v]
    assert not veto_ok, f"rs=40: af=0.92 sollte OK sein, bekam VETO: {veto_ok}"
    # rs=80 (gutes Material): 0.92 < 0.95 → VETO bleibt
    r_veto = const.check_paragraph_zero(
        None,
        48000,
        artifact_freedom=0.92,
        hpi=0.6,
        chain_depth=1,
        restorability=80.0,  # type: ignore[arg-type]
    )
    veto = [v for v in r_veto if "VETO" in v and "artifact_freedom" in v]
    assert veto, "rs=80: af=0.92 sollte VETO auslösen"
    # rs=30 (poor): 0.95−0.10=0.85. af=0.88 → OK.
    r_poor = const.check_paragraph_zero(
        None,
        48000,
        artifact_freedom=0.88,
        hpi=0.6,
        chain_depth=1,
        restorability=30.0,  # type: ignore[arg-type]
    )
    veto_poor = [v for v in r_poor if "VETO" in v and "artifact_freedom" in v]
    assert not veto_poor, f"rs=30: af=0.88 sollte OK sein, bekam VETO: {veto_poor}"


# ── PMGG Regression Threshold ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "depth,min_ratio_vs_depth1",
    [
        (1, 1.00),
        (2, 1.00),
        (3, 1.10),
        (4, 1.20),
        (5, 1.30),
    ],
)
def test_pmgg_threshold_scales_with_depth(depth, min_ratio_vs_depth1):
    """§v10.120: PMGG Regression-Toleranz skaliert mit Depth."""
    base = _get_adaptive_threshold(50.0, "cassette", 1)
    actual = _get_adaptive_threshold(50.0, "cassette", depth)
    ratio = actual / base
    assert ratio >= min_ratio_vs_depth1, f"depth={depth}: ratio={ratio:.3f} < {min_ratio_vs_depth1}"


def test_pmgg_threshold_monotonically_increasing():
    """PMGG-Schwelle muss monoton mit Depth steigen."""
    prev = 0.0
    for depth in range(1, 6):
        t = _get_adaptive_threshold(50.0, "cassette", depth)
        assert t >= prev, f"depth={depth}: {t:.4f} < prev={prev:.4f}"
        prev = t


# ── Material-Fairness ────────────────────────────────────────────────────────

ANALOG_MATERIALS = ["vinyl", "tape", "cassette", "reel_tape", "shellac", "wax_cylinder", "wire_recording"]
DIGITAL_MATERIALS = ["cd_digital", "streaming", "aac", "mp3_high", "dat"]


@pytest.mark.parametrize("mat", ANALOG_MATERIALS)
def test_analog_materials_get_gdd_boost(mat):
    """Analog-Materialien bekommen Material-Faktor ≥ 3.0 für spectral-subtraction."""
    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.transfer_chain_depth = 1
    state.restorability_score = 100.0
    state.material_type = mat

    gdd = guard._compute_gdd_threshold("phase_03_denoise", state)
    assert abs(gdd) >= 30.0, f"{mat}: GDD={abs(gdd):.1f}ms — Analog-Boost fehlt"


@pytest.mark.parametrize("mat", DIGITAL_MATERIALS)
def test_digital_materials_no_false_analog_boost(mat):
    """Digital-Materialien sollen KEINEN Analog-Boost bekommen."""
    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.transfer_chain_depth = 1
    state.restorability_score = 100.0
    state.material_type = mat

    gdd = guard._compute_gdd_threshold("phase_03_denoise", state)
    assert abs(gdd) < 30.0, f"{mat}: GDD={abs(gdd):.1f}ms — unerwarteter Analog-Boost"


# ── Chain-Depth Plausibilität ───────────────────────────────────────────────


def test_chain_depth_5_never_exceeds_3x_depth1():
    """Depth=5 sollte maximal 1.75× (GDD) bzw. 1.30× (PMGG) sein."""
    guard = CumulativeInteractionGuard()

    # GDD
    state5 = InteractionGuardState(transfer_chain_depth=5, restorability_score=65.0, material_type="cassette")
    state1 = InteractionGuardState(transfer_chain_depth=1, restorability_score=65.0, material_type="cassette")
    gdd_ratio = guard._compute_gdd_threshold("phase_29_tape_hiss_reduction", state5) / guard._compute_gdd_threshold(
        "phase_29_tape_hiss_reduction", state1
    )
    assert gdd_ratio <= 2.0, f"GDD depth=5 ist {gdd_ratio:.2f}× depth=1 — zu weit"

    # PMGG
    pmgg_ratio = _get_adaptive_threshold(50.0, "cassette", 5) / _get_adaptive_threshold(50.0, "cassette", 1)
    assert pmgg_ratio <= 2.0, f"PMGG depth=5 ist {pmgg_ratio:.2f}× depth=1 — zu weit"


def test_depth_1_is_identical_to_before_fix():
    """Depth=1 muss exakt die alten Werte liefern (keine Regression)."""
    guard = CumulativeInteractionGuard()
    state = InteractionGuardState(transfer_chain_depth=1, restorability_score=64.0, material_type="cassette")

    # GDD für phase_29 bei depth=1: Base 10ms × mat_factor 3.0 × rest_factor 1.3 = 39ms
    gdd = guard._compute_gdd_threshold("phase_29_tape_hiss_reduction", state)
    assert 38.0 <= abs(gdd) <= 40.0, f"GDD depth=1: {abs(gdd):.1f}ms, erwartet ~39ms"
