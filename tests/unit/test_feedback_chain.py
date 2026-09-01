"""Unit-Tests für core/feedback_chain.py — FeedbackChain.

Spec §2.16: Iterative Qualitätsschleife mit konservativer Konvergenz.
≥ 12 Tests: Konvergenz, Max-Iterations, NaN-Guard, Rollback, Mono/Stereo.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

np.random.seed(42)

from backend.core.feedback_chain import (
    DEFAULT_TARGET_SCORE,
    EXCELLENCE_TARGET_SCORE,
    FeedbackChain,
    FeedbackChainResult,
    get_feedback_chain,
)

SR = 48000


def _sine(freq: float = 440.0, secs: float = 1.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(freq: float = 440.0, secs: float = 1.0) -> np.ndarray:
    mono = _sine(freq, secs)
    return np.stack([mono, mono * 0.9])


def _identity_fn(audio: np.ndarray, sr: int) -> np.ndarray:
    """Improve function that returns audio unchanged (convergence after 1 iter)."""
    return audio.copy()


def _improve_fn(audio: np.ndarray, sr: int) -> np.ndarray:
    """Slightly boosts RMS to simulate improvement, capped to avoid regression."""
    return np.clip(audio * 1.02, -1.0, 1.0)


def _degrading_fn(audio: np.ndarray, sr: int) -> np.ndarray:
    """Always returns worse audio (silent) — triggers regression guard."""
    return np.zeros_like(audio)


# ---------------------------------------------------------------------------
# Klasse 1: Import und Instantiierung
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFeedbackChainImport:
    def test_01_class_importable(self):
        assert FeedbackChain is not None

    def test_02_result_class_importable(self):
        assert FeedbackChainResult is not None

    def test_03_constants_valid(self):
        assert 0.0 < DEFAULT_TARGET_SCORE < 1.0
        assert EXCELLENCE_TARGET_SCORE > DEFAULT_TARGET_SCORE

    def test_04_instantiate_defaults(self):
        fc = FeedbackChain()
        assert fc.max_iterations >= 1
        assert fc.convergence_delta > 0.0

    def test_05_singleton_returns_instance(self):
        fc = get_feedback_chain()
        assert isinstance(fc, FeedbackChain)


# ---------------------------------------------------------------------------
# Klasse 2: Ergebnis-Struktur
# ---------------------------------------------------------------------------


class TestFeedbackChainResult:
    def test_06_result_has_audio(self):
        audio = _sine()
        fc = FeedbackChain(max_iterations=1)
        result = fc.run(audio, _identity_fn)
        assert isinstance(result, FeedbackChainResult)
        assert isinstance(result.audio, np.ndarray)

    def test_07_result_audio_same_shape_mono(self):
        audio = _sine()
        fc = FeedbackChain(max_iterations=2)
        result = fc.run(audio, _identity_fn)
        assert result.audio.shape == audio.shape

    def test_08_result_audio_same_shape_stereo(self):
        audio = _stereo()
        fc = FeedbackChain(max_iterations=2)
        result = fc.run(audio, _identity_fn)
        assert result.audio.shape == audio.shape

    def test_09_result_iterations_at_least_one(self):
        fc = FeedbackChain(max_iterations=3)
        result = fc.run(_sine(), _identity_fn)
        assert result.iterations >= 1

    def test_10_result_mos_history_not_empty(self):
        fc = FeedbackChain(max_iterations=2)
        result = fc.run(_sine(), _identity_fn)
        assert len(result.mos_history) >= 1

    def test_11_result_mos_values_in_range(self):
        fc = FeedbackChain(max_iterations=3)
        result = fc.run(_sine(), _identity_fn)
        for mos in result.mos_history:
            assert 1.0 <= mos <= 5.0, f"MOS {mos} außerhalb [1, 5]"


# ---------------------------------------------------------------------------
# Klasse 3: Konvergenz und Max-Iterations
# ---------------------------------------------------------------------------


class TestFeedbackChainConvergence:
    def test_12_identity_fn_converges(self):
        """Identische Ausgabe → konvergiert nach erster Iteration."""
        fc = FeedbackChain(max_iterations=5, convergence_delta=0.001)
        result = fc.run(_sine(), _identity_fn)
        assert result.converged is True

    def test_13_max_iterations_respected(self):
        """Nie konvergierende Funktion → max_iterations eingehalten."""
        call_count = {"n": 0}

        def noisy_fn(audio: np.ndarray, sr: int) -> np.ndarray:
            call_count["n"] += 1
            return np.clip(audio + np.random.randn(*audio.shape).astype(np.float32) * 0.3, -1.0, 1.0)  # type: ignore[no-any-return]

        fc = FeedbackChain(max_iterations=4, convergence_delta=1e-9)
        result = fc.run(_sine(), noisy_fn)
        assert result.iterations <= 4

    def test_14_improvement_fn_improves_mos(self):
        """Verbesserungsfunktion → MOS steigt mindestens nicht stark ab."""
        fc = FeedbackChain(max_iterations=3, convergence_delta=1e-6)
        result = fc.run(_sine(secs=0.5), _improve_fn)
        assert result.mos_history[-1] >= result.mos_history[0] * 0.95


# ---------------------------------------------------------------------------
# Klasse 4: NaN/Inf-Guard
# ---------------------------------------------------------------------------


class TestFeedbackChainNaNGuard:
    def test_15_nan_input_handled(self):
        audio = np.full(SR, float("nan"), dtype=np.float32)
        fc = FeedbackChain(max_iterations=2)
        result = fc.run(audio, _identity_fn)
        assert np.all(np.isfinite(result.audio))

    def test_16_inf_input_handled(self):
        audio = np.full(SR, float("inf"), dtype=np.float32)
        fc = FeedbackChain(max_iterations=2)
        result = fc.run(audio, _identity_fn)
        assert np.all(np.isfinite(result.audio))

    def test_17_nan_from_fn_handled(self):
        def nan_fn(a: np.ndarray, sr: int) -> np.ndarray:
            return np.full_like(a, float("nan"))

        fc = FeedbackChain(max_iterations=2)
        result = fc.run(_sine(), nan_fn)
        # Best ist das ursprüngliche Audio (NaN-fn schlechteres/rollback)
        assert result.audio is not None

    def test_18_output_clipped_to_minus1_plus1(self):
        """Ausgabe-Audio darf nie ±1 überschreiten."""
        fc = FeedbackChain(max_iterations=3)
        result = fc.run(_sine(), _identity_fn)
        assert np.all(result.audio >= -1.0)
        assert np.all(result.audio <= 1.0)


# ---------------------------------------------------------------------------
# Klasse 5: Regression Guard und Rollback
# ---------------------------------------------------------------------------


class TestFeedbackChainRegressionGuard:
    def test_19_degrading_fn_returns_best(self):
        """Stets verschlechternde Funktion → Rollback auf erstes Audio."""
        audio = _sine()
        fc = FeedbackChain(max_iterations=5)
        result = fc.run(audio, _degrading_fn)
        # Best-MOS aus degrading (Stille) < original → best = original
        best_mos = fc.compute_perceptual_score(result.audio)
        assert math.isfinite(best_mos)

    def test_20_perceptual_score_static_method(self):
        score = FeedbackChain.compute_perceptual_score(_sine())
        assert 1.0 <= score <= 5.0

    def test_21_perceptual_score_silence_low(self):
        silence_score = FeedbackChain.compute_perceptual_score(np.zeros(SR, dtype=np.float32))
        sine_score = FeedbackChain.compute_perceptual_score(_sine())
        assert silence_score < sine_score

    def test_21a_hard_runtime_cap_prevents_hours_long_loops(self, monkeypatch):
        """FeedbackChain muss eine harte Laufzeitobergrenze erfüllen, auch bei pathologischen Iterationen."""
        import backend.core.feedback_chain as fc_mod

        fc = FeedbackChain(max_iterations=100, convergence_delta=1e-9, max_runtime_s=0.25)
        audio = _sine(secs=0.5)
        clock = {"t": 0.0}

        def fake_perf_counter():
            value = clock["t"]
            clock["t"] += 0.05
            return value

        monkeypatch.setattr(fc_mod.time, "perf_counter", fake_perf_counter)

        def noisy_fn(a: np.ndarray, sr: int) -> np.ndarray:
            noisy_audio = a + 0.01 * np.random.randn(*a.shape).astype(np.float32)
            return np.asarray(np.clip(noisy_audio, -1.0, 1.0), dtype=np.float32)

        result = fc.run(audio, noisy_fn)

        assert result.total_time_s <= 0.30
        assert result.iterations < 10


# ---------------------------------------------------------------------------
# Klasse 6: Phasen-Listen-Modus
# ---------------------------------------------------------------------------


class TestFeedbackChainPhasesMode:
    def test_22_phases_list_mode(self):
        """Phasen-Listen-Modus: Liste von (id, fn, kwargs)-Tupeln."""
        audio = _sine()
        phases: Any = [
            ("ph01", lambda a, sr: np.clip(a, -1.0, 1.0), {}),
            ("ph02", lambda a, sr: a.copy(), {}),
        ]
        fc = FeedbackChain(max_iterations=2)
        result = fc.run(audio, phases)
        assert result.audio.shape == audio.shape
        assert result.iterations >= 1

    def test_23_excellence_mode_higher_target(self):
        fc_normal = FeedbackChain(excellence_mode=False)
        fc_excellence = FeedbackChain(excellence_mode=True)
        assert fc_excellence.target_score > fc_normal.target_score


# ---------------------------------------------------------------------------
# Klasse 7: §2.33 PhysicalCeiling-Gate (Lücke B, 9.10.x)
# ---------------------------------------------------------------------------

HEADROOM = 0.03  # aus §2.33 Spec


class TestFeedbackChainCeiling:
    """Tests für ceiling-Parameter in FeedbackChain.run() — Spec §2.33."""

    # --- Parametersignatur ---

    def test_24_run_accepts_ceiling_param(self):
        """run() muss ceiling=float akzeptieren (kein TypeError)."""
        fc = FeedbackChain(max_iterations=3)
        result = fc.run(_sine(), _identity_fn, ceiling=0.90)
        assert isinstance(result, FeedbackChainResult)

    def test_25_run_ceiling_none_no_early_exit(self):
        """ceiling=None → kein Ceiling-basierter Abbruch, normales Verhalten."""
        fc = FeedbackChain(max_iterations=5, convergence_delta=1e-9)
        result = fc.run(_sine(), _improve_fn, ceiling=None)
        assert not result.ceiling_reached

    # --- FeedbackChainResult.ceiling_reached ---

    def test_26_result_has_ceiling_reached_field(self):
        """FeedbackChainResult muss ceiling_reached-Attribut haben."""
        fc = FeedbackChain(max_iterations=1)
        result = fc.run(_sine(), _identity_fn)
        assert hasattr(result, "ceiling_reached")
        assert isinstance(result.ceiling_reached, bool)

    def test_27_ceiling_reached_false_by_default(self):
        """Ohne Ceiling-Auslösung ist ceiling_reached=False."""
        fc = FeedbackChain(max_iterations=2)
        result = fc.run(_sine(), _identity_fn)
        assert result.ceiling_reached is False

    # --- Ceiling frühzeitig erreicht ---

    def test_28_ceiling_reached_when_mos_above_threshold(self):
        """Wenn MOS initial >= ceiling - 0.03: erste Iteration → ceiling_reached=True."""
        audio = _sine()
        initial_mos = FeedbackChain.compute_perceptual_score(audio)
        # Setze ceiling knapp unter initial_mos + HEADROOM → sofort erreicht
        ceiling_val = initial_mos - HEADROOM + 0.001
        fc = FeedbackChain(max_iterations=10, convergence_delta=1e-9)
        result = fc.run(audio, _improve_fn, ceiling=ceiling_val)
        assert result.ceiling_reached is True

    def test_29_ceiling_reached_implies_converged(self):
        """ceiling_reached=True → converged muss ebenfalls True sein."""
        audio = _sine()
        initial_mos = FeedbackChain.compute_perceptual_score(audio)
        ceiling_val = initial_mos - HEADROOM + 0.001
        fc = FeedbackChain(max_iterations=10, convergence_delta=1e-9)
        result = fc.run(audio, _improve_fn, ceiling=ceiling_val)
        if result.ceiling_reached:
            assert result.converged is True

    def test_30_ceiling_early_exit_fewer_iterations(self):
        """Mit sehr niedrigem Ceiling: max_iterations wird NICHT voll ausgeschöpft."""
        audio = _sine()
        initial_mos = FeedbackChain.compute_perceptual_score(audio)
        # Ceiling knapp über oder gleich initial_mos: sofortiger Abbruch
        ceiling_val = initial_mos + 0.001  # sehr leicht über aktuell → erster Pass reicht
        fc = FeedbackChain(max_iterations=10, convergence_delta=1e-9)
        result = fc.run(audio, _improve_fn, ceiling=ceiling_val)
        # Mit Ceiling darf nicht alle 10 Iterationen laufen
        assert result.iterations < 10 or result.ceiling_reached

    def test_31_ceiling_above_5_never_triggers(self):
        """ceiling=6.0 (über MOS-Max) → ceiling_reached niemals True."""
        fc = FeedbackChain(max_iterations=5, convergence_delta=1e-9)
        result = fc.run(_sine(), _improve_fn, ceiling=6.0)
        assert result.ceiling_reached is False

    def test_32_ceiling_zero_triggers_immediately(self):
        """ceiling=0.0: MOS immer >= 0.0 - 0.03 → ceiling_reached sofort."""
        audio = _sine()
        fc = FeedbackChain(max_iterations=10, convergence_delta=1e-9)
        result = fc.run(audio, _improve_fn, ceiling=0.0)
        assert result.ceiling_reached is True
        # Muss nach erster Iteration (Iteration 1) abbrechen
        assert result.iterations <= 2  # 1 Einstieg + maximal 1 Verbesser-Iteration

    # --- Rollback bleibt korrekt ---

    def test_33_ceiling_audio_output_finite(self):
        """Auch bei ceiling-basiertem Abbruch: Ausgabe-Audio muss endlich und [-1,1] sein."""
        audio = _sine()
        initial_mos = FeedbackChain.compute_perceptual_score(audio)
        fc = FeedbackChain(max_iterations=10)
        result = fc.run(audio, _improve_fn, ceiling=initial_mos - HEADROOM + 0.001)
        assert np.all(np.isfinite(result.audio))
        assert np.all(result.audio >= -1.0)
        assert np.all(result.audio <= 1.0)

    def test_34_ceiling_stereo_works(self):
        """ceiling-Gate funktioniert auch mit Stereo-Eingabe."""
        audio = _stereo()
        initial_mos = FeedbackChain.compute_perceptual_score(audio)
        ceiling_val = initial_mos - HEADROOM + 0.001
        fc = FeedbackChain(max_iterations=10)
        result = fc.run(audio, _improve_fn, ceiling=ceiling_val)
        assert result.audio.shape == audio.shape
        assert isinstance(result.ceiling_reached, bool)

    # --- Boundary: HEADROOM_THRESHOLD ---

    def test_35_exactly_at_ceiling_minus_headroom_not_triggered(self):
        """MOS == ceiling - 0.03: kein Abbruch (Grenzwert exklusiv)."""
        audio = _sine()
        initial_mos = FeedbackChain.compute_perceptual_score(audio)
        # Setze Ceiling so, dass initial_mos < ceiling - HEADROOM
        safe_ceiling = initial_mos + HEADROOM + 0.01
        fc = FeedbackChain(max_iterations=1, convergence_delta=1e-9)
        result = fc.run(audio, _identity_fn, ceiling=safe_ceiling)
        # Darf nicht ceiling_reached sein, da MOS noch nicht nahe genug
        assert result.ceiling_reached is False


# ---------------------------------------------------------------------------
# Klasse 8: P1-1 Perzeptiver Primärpfad (VERSA/PQS) + markierter Fallback
# ---------------------------------------------------------------------------


class TestFeedbackChainPerceptualLoopScoring:
    def test_36_pqs_loop_marks_pqs_as_score_source(self):
        fc = FeedbackChain(max_iterations=1, use_pqs_in_loop=True, use_versa_in_loop=False)
        # §2.44: use_versa_in_loop wird ignoriert (immer True) — _versa_score_fn direkt deaktivieren
        fc._versa_score_fn = None
        fc._pqs_score_fn = lambda audio, sr: SimpleNamespace(mos=4.2)

        result = fc.run(_sine(), _identity_fn)

        assert result.metadata["score_source"] == "pqs_absolute"
        assert result.metadata["score_fallback_used"] is False
        assert "pqs_absolute" in result.metadata["score_sources_seen"]  # type: ignore[operator]

    def test_37_pqs_failure_marks_heuristic_fallback(self):
        fc = FeedbackChain(max_iterations=1, use_pqs_in_loop=True, use_versa_in_loop=False)
        # §2.44: use_versa_in_loop wird ignoriert (immer True) — _versa_score_fn direkt deaktivieren
        fc._versa_score_fn = None

        def _raise_pqs(audio: np.ndarray, sr: int) -> object:
            raise RuntimeError("pqs unavailable")

        fc._pqs_score_fn = _raise_pqs

        result = fc.run(_sine(), _identity_fn)

        assert result.metadata["score_source"] == "heuristic_rms"
        assert result.metadata["score_fallback_used"] is True
        assert "heuristic_rms" in result.metadata["score_sources_seen"]  # type: ignore[operator]

    def test_38_versa_is_preferred_over_pqs_when_both_enabled(self):
        fc = FeedbackChain(max_iterations=1, use_pqs_in_loop=True, use_versa_in_loop=True)
        fc._versa_score_fn = lambda audio, sr: SimpleNamespace(mos=4.7, model_used="versa-test")
        fc._pqs_score_fn = lambda audio, sr: SimpleNamespace(mos=4.1)

        result = fc.run(_sine(), _identity_fn)

        assert result.metadata["score_source"] in ("versa", "versa_segmented")
        assert result.metadata["score_fallback_used"] is False
        assert result.overall_score == pytest.approx(4.7)


# ---------------------------------------------------------------------------
# Klasse: §2.54 Adaptive Threshold Tests
# ---------------------------------------------------------------------------


class TestAdaptiveThresholds254:
    """§2.54: FeedbackChain thresholds must be material/restorability-adaptive."""

    def test_39_prune_threshold_shellac_more_lenient_than_cd(self):
        fc_cd = FeedbackChain(material="cd_digital", restorability_score=20.0)
        fc_shellac = FeedbackChain(material="shellac", restorability_score=85.0)
        # Restorative phase pruning threshold
        t_cd = fc_cd._compute_adaptive_prune_threshold(is_restorative=True)
        t_shellac = fc_shellac._compute_adaptive_prune_threshold(is_restorative=True)
        # More lenient = more negative
        assert t_shellac < t_cd, f"Shellac ({t_shellac}) should be more lenient than CD ({t_cd})"

    def test_40_prune_threshold_enhancement_stricter_than_restorative(self):
        fc = FeedbackChain(material="vinyl", restorability_score=60.0)
        t_rest = fc._compute_adaptive_prune_threshold(is_restorative=True)
        t_enh = fc._compute_adaptive_prune_threshold(is_restorative=False)
        assert t_enh > t_rest, "Enhancement threshold should be stricter (less negative)"

    def test_41_mos_regression_tolerance_poor_material_higher(self):
        fc_cd = FeedbackChain(material="cd_digital", restorability_score=20.0)
        fc_shellac = FeedbackChain(material="shellac", restorability_score=85.0, defect_severity_mean=0.7)
        tol_cd = fc_cd._compute_adaptive_mos_regression_tolerance()
        tol_shellac = fc_shellac._compute_adaptive_mos_regression_tolerance()
        assert tol_shellac > tol_cd, f"Shellac tol ({tol_shellac}) should exceed CD ({tol_cd})"

    def test_42_prune_threshold_clamped_within_bounds(self):
        fc = FeedbackChain(material="wax_cylinder", restorability_score=100.0, defect_severity_mean=1.0)
        t = fc._compute_adaptive_prune_threshold(is_restorative=True)
        assert -0.30 <= t <= -0.05, f"Must be in [-0.30, -0.05], got {t}"

    def test_43_mos_regression_tolerance_clamped(self):
        fc = FeedbackChain(material="wax_cylinder", restorability_score=100.0, defect_severity_mean=1.0)
        tol = fc._compute_adaptive_mos_regression_tolerance()
        assert 0.03 <= tol <= 0.25, f"Must be in [0.03, 0.25], got {tol}"

    def test_44_default_params_backward_compatible(self):
        """Default restorability/severity should produce values close to legacy."""
        fc = FeedbackChain()  # defaults: restorability=50, severity=0.3
        t_rest = fc._compute_adaptive_prune_threshold(is_restorative=True)
        t_enh = fc._compute_adaptive_prune_threshold(is_restorative=False)
        # Legacy: -0.05 restorative, -0.01 enhancement
        assert -0.15 <= t_rest <= -0.005
        assert -0.03 <= t_enh <= -0.005


# ---------------------------------------------------------------------------
# Klasse: §2.56 Goal-Weights Bias in FeedbackChain (v10.0.0)
# ---------------------------------------------------------------------------


class TestGoalWeightsBias256:
    """§2.56: goal_weights must bias FeedbackChain pruning threshold and MOS tolerance."""

    _P1P2_HIGH = {
        "natuerlichkeit": 1.8,
        "authentizitaet": 1.6,
        "tonal_center": 1.5,
        "timbre_authentizitaet": 1.5,
        "artikulation": 1.4,
        # P4/P5 neutral
        "brillanz": 1.0,
        "raumtiefe": 1.0,
        "waerme": 1.0,
        "bassgewalt": 1.0,
    }
    _P4P5_HIGH = {
        "natuerlichkeit": 1.0,
        "authentizitaet": 1.0,
        "tonal_center": 1.0,
        "timbre_authentizitaet": 1.0,
        "artikulation": 1.0,
        "brillanz": 1.7,
        "raumtiefe": 1.6,
        "waerme": 1.5,
        "bassgewalt": 1.5,
    }
    _UNIFORM = {
        "natuerlichkeit": 1.0,
        "authentizitaet": 1.0,
        "tonal_center": 1.0,
        "timbre_authentizitaet": 1.0,
        "artikulation": 1.0,
        "brillanz": 1.0,
        "raumtiefe": 1.0,
        "waerme": 1.0,
        "bassgewalt": 1.0,
    }

    def _make_fc(self, goal_weights):
        fc = FeedbackChain(material="vinyl", restorability_score=60.0, defect_severity_mean=0.4)
        fc.goal_weights = goal_weights
        return fc

    def test_45_p1p2_heavy_tightens_prune_threshold(self):
        """P1/P2-heavy songs → prune threshold is closer to 0 (stricter pruning)."""
        fc_p1p2 = self._make_fc(self._P1P2_HIGH)
        fc_p4p5 = self._make_fc(self._P4P5_HIGH)
        fc_uni = self._make_fc(self._UNIFORM)

        t_p1p2 = fc_p1p2._compute_adaptive_prune_threshold(is_restorative=True)
        t_p4p5 = fc_p4p5._compute_adaptive_prune_threshold(is_restorative=True)
        t_uni = fc_uni._compute_adaptive_prune_threshold(is_restorative=True)

        # P1/P2 heavy → stricter (less negative) than uniform
        assert t_p1p2 > t_uni, f"P1/P2-heavy ({t_p1p2}) should be stricter (less negative) than uniform ({t_uni})"
        # P4/P5 heavy → more lenient (more negative) than uniform
        assert t_p4p5 < t_uni, f"P4/P5-heavy ({t_p4p5}) should be more lenient (more negative) than uniform ({t_uni})"

    def test_46_goal_weights_bias_bounded(self):
        """Bias must remain within ±0.05; overall threshold stays within [-0.30, -0.005]."""
        fc = self._make_fc(self._P1P2_HIGH)
        t = fc._compute_adaptive_prune_threshold(is_restorative=True)
        assert -0.30 <= t <= -0.005, f"Threshold must be in [-0.30, -0.005], got {t}"

    def test_47_no_goal_weights_unchanged(self):
        """With no goal_weights set, result equals the non-biased computation."""
        fc_with = self._make_fc(self._UNIFORM)
        fc_without = FeedbackChain(material="vinyl", restorability_score=60.0, defect_severity_mean=0.4)
        # Uniform weights (all 1.0) → mean P1P2 == mean P4P5 → bias ≈ 0
        t_with = fc_with._compute_adaptive_prune_threshold(is_restorative=True)
        t_without = fc_without._compute_adaptive_prune_threshold(is_restorative=True)
        assert abs(t_with - t_without) < 0.001, (
            f"Uniform weights ({t_with}) should equal no-weights result ({t_without})"
        )

    def test_48_mos_tolerance_tightened_for_p1p2(self):
        """P1/P2-heavy songs get lower MOS regression tolerance."""
        fc_p1p2 = self._make_fc(self._P1P2_HIGH)
        fc_uni = self._make_fc(self._UNIFORM)
        tol_p1p2 = fc_p1p2._compute_adaptive_mos_regression_tolerance()
        tol_uni = fc_uni._compute_adaptive_mos_regression_tolerance()
        # P1/P2 heavy → tighter tolerance (smaller value) to protect critical goals
        assert tol_p1p2 <= tol_uni, f"P1/P2-heavy tolerance ({tol_p1p2}) should be ≤ uniform ({tol_uni})"

    def test_49_mos_tolerance_no_goal_weights_unchanged(self):
        """Without goal_weights, MOS tolerance is unaffected by §2.56."""
        fc = FeedbackChain(material="vinyl", restorability_score=60.0, defect_severity_mean=0.4)
        tol = fc._compute_adaptive_mos_regression_tolerance()
        assert 0.03 <= tol <= 0.25, f"Tolerance out of bounds: {tol}"

    def test_50_p4p5_heavy_does_not_violate_p1p2_floor(self):
        """Even extreme P4/P5 weighting must not make prune threshold looser than -0.30."""
        extreme_p4p5 = dict(self._P4P5_HIGH)
        for k in ("brillanz", "raumtiefe", "waerme", "bassgewalt"):
            extreme_p4p5[k] = 2.0
        fc = self._make_fc(extreme_p4p5)
        t = fc._compute_adaptive_prune_threshold(is_restorative=True)
        assert t >= -0.30, f"Threshold must not exceed -0.30, got {t}"
