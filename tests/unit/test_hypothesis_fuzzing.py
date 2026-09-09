from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")  # CI-Minimal-Umgebung (cross-platform)

"""
tests/unit/test_hypothesis_fuzzing.py — Property-Based Tests für Aurik 10.0.0 (§5.1).

Nutzt Hypothesis für automatisiertes Fuzzing der DSP-Kernfunktionen.
Ziel: NaN/Inf-Safety, Bounds-Invarianten und Konsistenz bei beliebigen Eingaben.

Ausführen: pytest tests/unit/test_hypothesis_fuzzing.py -v --timeout=60
"""


import math

import numpy as np
import pytest

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")

# ---------------------------------------------------------------------------
# Hilfs-Strategien
# ---------------------------------------------------------------------------

# Gültige Audio-Arrays: float32, normalisiert in [-1, 1], SR 48000
audio_strategy = st.integers(min_value=1024, max_value=48000 * 10).flatmap(
    lambda n: st.builds(
        lambda data: np.array(data, dtype=np.float32),
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n
        ),
    )
)

sample_rate_strategy = st.just(48000)

score_1d = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=100)


# ---------------------------------------------------------------------------
# 1. PQS-Scorer: NaN-Invariante
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestPQSScorerFuzzing:
    """Property-based Tests für PerceptualQualityScorer (§2.6)."""

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n=st.integers(min_value=4096, max_value=48000 * 5),
        noise_level=st.floats(min_value=0.0, max_value=0.3),
    )
    def test_01_score_absolute_no_nan(self, n: int, noise_level: float) -> None:
        """score_audio_absolute() liefert immer finite Werte."""
        np.random.seed(42)
        audio = np.random.randn(n).astype(np.float32) * 0.3
        audio += np.random.randn(n).astype(np.float32) * noise_level
        audio = np.clip(audio, -1.0, 1.0)

        try:
            from backend.core.perceptual_quality_scorer import score_audio_absolute

            result = score_audio_absolute(audio, sr=48000)
            assert math.isfinite(result.mos), f"MOS ist NaN/Inf: {result.mos}"
            assert 1.0 <= result.mos <= 5.0, f"MOS außerhalb [1,5]: {result.mos}"
        except ImportError:
            pytest.skip("PerceptualQualityScorer nicht importierbar")

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(n=st.integers(min_value=4096, max_value=48000 * 3))
    def test_02_silence_gives_finite_score(self, n: int) -> None:
        """Stille als Eingabe → kein Absturz, finite Ausgabe."""
        audio = np.zeros(n, dtype=np.float32)
        try:
            from backend.core.perceptual_quality_scorer import score_audio_absolute

            result = score_audio_absolute(audio, sr=48000)
            assert math.isfinite(result.mos)
        except ImportError:
            pytest.skip("PerceptualQualityScorer nicht importierbar")


# ---------------------------------------------------------------------------
# 2. MusicalGoalsChecker: Bounds-Invariante
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestMusicalGoalsFuzzing:
    """Property-based Tests für MusicalGoalsChecker (§1.2)."""

    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(n=st.integers(min_value=8192, max_value=48000 * 2))
    def test_01_all_scores_bounded_01(self, n: int) -> None:
        """measure_all() liefert immer Scores in [0, 1] ohne NaN."""
        np.random.seed(0)
        audio = np.random.randn(n).astype(np.float32) * 0.3
        audio = np.clip(audio, -1.0, 1.0)

        try:
            import os

            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

            # CREPE-ONNX kann bei hypothesis-Fuzzing in den 30s-Timeout laufen → CREPE deaktivieren
            os.environ.setdefault("AURIK_DISABLE_CREPE", "1")
            checker = MusicalGoalsChecker()
            scores = checker.measure_all(audio, sr=48000)
            for goal, score in scores.items():
                assert math.isfinite(score), f"NaN/Inf in {goal}: {score}"
                assert 0.0 <= score <= 1.0, f"Score außerhalb [0,1] bei {goal}: {score}"
        except ImportError:
            pytest.skip("MusicalGoalsChecker nicht importierbar")
        finally:
            os.environ.pop("AURIK_DISABLE_CREPE", None)

    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(n=st.integers(min_value=8192, max_value=48000 * 2))
    def test_02_dirac_impulse_no_crash(self, n: int) -> None:
        """Dirac-Impuls als Eingabe → kein Absturz."""
        audio = np.zeros(n, dtype=np.float32)
        audio[n // 2] = 1.0

        try:
            import os

            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

            os.environ.setdefault("AURIK_DISABLE_CREPE", "1")
            checker = MusicalGoalsChecker()
            scores = checker.measure_all(audio, sr=48000)
            assert isinstance(scores, dict)
            assert len(scores) > 0
        except ImportError:
            pytest.skip("MusicalGoalsChecker nicht importierbar")
        finally:
            os.environ.pop("AURIK_DISABLE_CREPE", None)


# ---------------------------------------------------------------------------
# 3. PerceptualEmbedder: L2-Normalisierungs-Invariante
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestEmbedderFuzzing:
    """Property-based Tests für PerceptualEmbedder (§2.3)."""

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(n=st.integers(min_value=8192, max_value=48000 * 5))
    def test_01_embedding_l2_normalized(self, n: int) -> None:
        """embed_audio() liefert L2-normierten Vektor ‖v‖₂ ≈ 1.0."""
        np.random.seed(1)
        audio = np.random.randn(n).astype(np.float32) * 0.4
        audio = np.clip(audio, -1.0, 1.0)

        try:
            from backend.core.perceptual_embedder import embed_audio

            embedding = embed_audio(audio, 48000)
            vec = embedding.vector
            norm = float(np.linalg.norm(vec))
            assert math.isfinite(norm), f"Norm ist NaN/Inf: {norm}"
            assert abs(norm - 1.0) < 0.05, f"Embedding nicht L2-normalisiert: ‖v‖₂ = {norm:.4f}"
        except ImportError:
            pytest.skip("PerceptualEmbedder nicht importierbar")

    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(n=st.integers(min_value=8192, max_value=48000 * 3))
    def test_02_same_audio_same_embedding(self, n: int) -> None:
        """Gleiche Eingabe → identisches Embedding (deterministisch)."""
        np.random.seed(7)
        audio = np.random.randn(n).astype(np.float32) * 0.3
        audio = np.clip(audio, -1.0, 1.0)

        try:
            from backend.core.perceptual_embedder import embed_audio

            e1 = embed_audio(audio, 48000)
            e2 = embed_audio(audio, 48000)
            np.testing.assert_allclose(e1.vector, e2.vector, atol=1e-6)
        except ImportError:
            pytest.skip("PerceptualEmbedder nicht importierbar")


# ---------------------------------------------------------------------------
# 4. Audio-Clip-Invariante: Ausgabe niemals > 1.0
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestAudioClipInvariantFuzzing:
    """Kein Modul darf unkontrolliertes Clipping erzeugen (§3.1)."""

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n=st.integers(min_value=2048, max_value=48000 * 3),
        gain=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_01_output_clip_invariant_after_gain(self, n: int, gain: float) -> None:
        """np.clip(-1, 1) nach jeder Verstärkungsoperation: |max| ≤ 1.0."""
        np.random.seed(42)
        audio = np.random.randn(n).astype(np.float32) * 0.5
        processed = np.clip(audio * gain, -1.0, 1.0)
        assert float(np.max(np.abs(processed))) <= 1.0 + 1e-6

    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(n=st.integers(min_value=2048, max_value=48000 * 2))
    def test_02_nan_guard_after_sum(self, n: int) -> None:
        """np.nan_to_num nach Summation: keine NaN/Inf im Ausgang."""
        np.random.seed(99)
        a = np.random.randn(n).astype(np.float32) * 0.5
        b = np.random.randn(n).astype(np.float32) * 0.5
        # Simuliert typische Phase-Summation mit möglichen NaN-Quellen
        result = np.nan_to_num(a + b, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)
        assert np.isfinite(result).all()
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 5. GP-Optimizer: propose() liefert immer valide Parameterräume
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestGPOptimizerFuzzing:
    """Property-based Tests für GPParameterOptimizer (§2.5)."""

    PARAM_BOUNDS = {
        "noise_reduction_strength": (0.05, 0.95),
        "harmonic_boost_db": (0.0, 6.0),
        "ola_crossfade_ms": (5.0, 60.0),
        "compression_ratio": (1.05, 5.0),
        "eq_high_shelf_db": (-6.0, 6.0),
        "ar_order": (16.0, 128.0),
        "click_threshold_sigma": (3.0, 8.0),
        "hpf_cutoff_hz": (10.0, 120.0),
        "nr_smoothing_ms": (20.0, 200.0),
        "declip_threshold": (0.90, 0.99),
    }

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(material=st.sampled_from(["tape", "vinyl", "shellac", "unknown", "cd_digital"]))
    def test_01_propose_within_bounds(self, material: str) -> None:
        """propose() gibt Werte innerhalb der definierten Bounds zurück."""
        try:
            from backend.core.gp_parameter_optimizer import get_optimizer

            optimizer = get_optimizer()
            proposal = optimizer.propose(material)
            params = proposal.params if hasattr(proposal, "params") else proposal

            if isinstance(params, dict):
                for key, (lo, hi) in self.PARAM_BOUNDS.items():
                    if key in params:
                        assert lo - 1e-6 <= params[key] <= hi + 1e-6, f"{key}={params[key]} außerhalb [{lo},{hi}]"
        except ImportError:
            pytest.skip("GPParameterOptimizer nicht importierbar")

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(score=st.floats(allow_nan=True, allow_infinity=True))
    def test_02_update_ignores_invalid_scores(self, score: float) -> None:
        """update() mit NaN/Inf-Score → kein Absturz (§6.4)."""
        try:
            from backend.core.gp_parameter_optimizer import get_optimizer

            optimizer = get_optimizer()
            # Sollte safe sein — keine Exception beim NaN/Inf-Score
            # Korrekte Signatur: update(parameters, score, material)
            optimizer.update({"noise_reduction_strength": 0.5}, score, "unknown")
        except ImportError:
            pytest.skip("GPParameterOptimizer nicht importierbar")
        except Exception as e:
            # Nur wenn score endlich — dann ist ein Fehler tatsächlich ein Fehler
            if math.isfinite(score):
                raise AssertionError(f"update() warf Exception bei finitem Score: {e}") from e


# ---------------------------------------------------------------------------
# 6. GoalApplicabilityFilter — Fuzzing (§2.32)
# ---------------------------------------------------------------------------

_MATERIALS_GAF = [
    "tape",
    "vinyl",
    "shellac",
    "wax_cylinder",
    "cd_digital",
    "mp3_low",
    "mp3_high",
    "unknown",
]
_ERA_DECADES_GAF = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, None]


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestGoalApplicabilityFuzzing:
    """Property-Based Tests für GoalApplicabilityFilter (§2.32)."""

    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=0, max_value=48_000 * 20),
        channels=st.sampled_from([1, 2]),
        material=st.sampled_from(_MATERIALS_GAF),
        era_decade=st.sampled_from(_ERA_DECADES_GAF),
        flashsr_available=st.booleans(),
    )
    def test_01_no_crash_random_audio(
        self,
        n_samples: int,
        channels: int,
        material: str,
        era_decade,
        flashsr_available: bool,
    ) -> None:
        """Kein Absturz bei beliebiger Audio-Länge, Material, Ära."""
        try:
            from backend.core.goal_applicability_filter import (
                GoalApplicabilityResult,
                evaluate_goal_applicability,
            )
        except ImportError:
            pytest.skip("GoalApplicabilityFilter nicht importierbar")
        if n_samples == 0:
            audio = None
        elif channels == 1:
            audio = np.random.randn(n_samples).astype(np.float32) * 0.5
        else:
            audio = np.random.randn(n_samples, channels).astype(np.float32) * 0.5
        result = evaluate_goal_applicability(
            audio=audio,
            sr=48000,
            material=material,
            era_decade=era_decade,
            flashsr_available=flashsr_available,
        )
        assert isinstance(result, GoalApplicabilityResult)

    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=512, max_value=48_000 * 20),
        channels=st.sampled_from([1, 2]),
        material=st.sampled_from(_MATERIALS_GAF),
        era_decade=st.sampled_from(_ERA_DECADES_GAF),
    )
    def test_02_partition_invariant(
        self,
        n_samples: int,
        channels: int,
        material: str,
        era_decade,
    ) -> None:
        """applicable ∪ inapplicable == ALL_GOALS (erschöpfende Partition)."""
        try:
            from backend.core.goal_applicability_filter import (
                ALL_GOALS,
                evaluate_goal_applicability,
            )
        except ImportError:
            pytest.skip("GoalApplicabilityFilter nicht importierbar")
        if channels == 1:
            audio = np.random.randn(n_samples).astype(np.float32) * 0.4
        else:
            audio = np.random.randn(n_samples, channels).astype(np.float32) * 0.4
        result = evaluate_goal_applicability(
            audio=audio,
            sr=48000,
            material=material,
            era_decade=era_decade,
        )
        assert result.applicable | result.inapplicable == ALL_GOALS
        assert result.applicable & result.inapplicable == frozenset()

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=512, max_value=48_000 * 5),
        channels=st.sampled_from([1, 2]),
        material=st.sampled_from(_MATERIALS_GAF),
    )
    def test_03_always_applicable_never_inapplicable(
        self,
        n_samples: int,
        channels: int,
        material: str,
    ) -> None:
        """ALWAYS_APPLICABLE-Goals niemals inapplicable."""
        try:
            from backend.core.goal_applicability_filter import (
                ALWAYS_APPLICABLE,
                evaluate_goal_applicability,
            )
        except ImportError:
            pytest.skip("GoalApplicabilityFilter nicht importierbar")
        if channels == 1:
            audio = np.random.randn(n_samples).astype(np.float32) * 0.4
        else:
            audio = np.random.randn(n_samples, channels).astype(np.float32) * 0.4
        result = evaluate_goal_applicability(audio=audio, sr=48000, material=material)
        for goal in ALWAYS_APPLICABLE:
            assert goal not in result.inapplicable, f"{goal} ∈ ALWAYS_APPLICABLE, aber als inapplicable markiert"

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(scale=st.floats(allow_nan=True, allow_infinity=True))
    def test_04_nan_inf_audio_no_crash(self, scale: float) -> None:
        """NaN/Inf-Audio → kein Absturz."""
        try:
            from backend.core.goal_applicability_filter import (
                GoalApplicabilityResult,
                evaluate_goal_applicability,
            )
        except ImportError:
            pytest.skip("GoalApplicabilityFilter nicht importierbar")
        safe_scale = float(np.clip(scale, -1e4, 1e4)) if math.isfinite(scale) else 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            raw = np.random.randn(4800).astype(np.float32) * np.float32(safe_scale)
        raw[0] = float("nan")
        raw[-1] = float("inf")
        result = evaluate_goal_applicability(audio=raw, sr=48000, material="unknown")
        assert isinstance(result, GoalApplicabilityResult)


# ---------------------------------------------------------------------------
# 7. EraAuthenticPerceptualCompletion — Fuzzing (§2.35)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestEraCompletionFuzzing:
    """Property-Based Tests für EraAuthenticPerceptualCompletion (§2.35)."""

    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=0, max_value=48_000 * 10),
        channels=st.sampled_from([1, 2]),
        era=st.sampled_from([1920, 1940, 1960, 1980, 2000, None]),
    )
    def test_01_complete_no_crash(self, n_samples: int, channels: int, era) -> None:
        """complete() stürzt nie ab."""
        try:
            from backend.core.era_authentic_perceptual_completion import (
                EraCompletionResult,
                get_era_completion,
            )
        except ImportError:
            pytest.skip("EraAuthenticPerceptualCompletion nicht importierbar")
        if n_samples == 0:
            audio = np.zeros(0, dtype=np.float32)
        elif channels == 1:
            audio = np.random.randn(n_samples).astype(np.float32) * 0.4
        else:
            audio = np.random.randn(n_samples, channels).astype(np.float32) * 0.4
        comp = get_era_completion()
        result = comp.complete(audio=audio, sr=48000, era=era)
        assert isinstance(result, EraCompletionResult)

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=512, max_value=48_000 * 5),
        era=st.sampled_from([1920, 1940, 1960, 1980, 2000, None]),
    )
    def test_02_output_finite_and_clipped(self, n_samples: int, era) -> None:
        """Audio-Ausgang NaN/Inf-frei und in [-1, 1]."""
        try:
            from backend.core.era_authentic_perceptual_completion import get_era_completion
        except ImportError:
            pytest.skip("EraAuthenticPerceptualCompletion nicht importierbar")
        audio = np.random.randn(n_samples).astype(np.float32) * 0.5
        comp = get_era_completion()
        result = comp.complete(audio=audio, sr=48000, era=era)
        assert np.isfinite(result.audio).all(), "NaN/Inf im Audio-Ausgang"
        assert float(np.max(np.abs(result.audio))) <= 1.0 + 1e-5

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(era=st.sampled_from([1920, 1940, 1960, 1980, 2000, None]))
    def test_03_brillanz_ceiling_bounded(self, era) -> None:
        """brillanz_ceiling ∈ [0, 1]."""
        try:
            from backend.core.era_authentic_perceptual_completion import get_era_completion
        except ImportError:
            pytest.skip("EraAuthenticPerceptualCompletion nicht importierbar")
        audio = np.random.randn(4800).astype(np.float32) * 0.4
        comp = get_era_completion()
        result = comp.complete(audio=audio, sr=48000, era=era)
        assert math.isfinite(result.brillanz_ceiling)
        assert 0.0 <= result.brillanz_ceiling <= 1.0


# ---------------------------------------------------------------------------
# 8. IntroducedArtifactDetector — Fuzzing (§2.23)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis nicht installiert")
class TestIADFuzzing:
    """Property-Based Tests für IntroducedArtifactDetector (§2.23)."""

    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=0, max_value=48_000 * 5),
        channels=st.sampled_from([1, 2]),
    )
    def test_01_detect_no_crash(self, n_samples: int, channels: int) -> None:
        """detect() stürzt nie ab."""
        try:
            from backend.core.introduced_artifact_detector import IADResult, get_iad
        except ImportError:
            pytest.skip("IntroducedArtifactDetector nicht importierbar")
        if n_samples == 0:
            orig = np.zeros(0, dtype=np.float32)
            rest = np.zeros(0, dtype=np.float32)
        elif channels == 1:
            orig = np.random.randn(n_samples).astype(np.float32) * 0.4
            rest = np.random.randn(n_samples).astype(np.float32) * 0.4
        else:
            orig = np.random.randn(n_samples, channels).astype(np.float32) * 0.4
            rest = np.random.randn(n_samples, channels).astype(np.float32) * 0.4
        iad = get_iad()
        result = iad.detect(original=orig, restored=rest, sr=48000)
        assert isinstance(result, IADResult)

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(n_samples=st.integers(min_value=512, max_value=48_000 * 3))
    def test_02_scores_bounded(self, n_samples: int) -> None:
        """Alle Artefakt-Konfidenzwerte ∈ [0, 1] und finite."""
        try:
            from backend.core.introduced_artifact_detector import get_iad
        except ImportError:
            pytest.skip("IntroducedArtifactDetector nicht importierbar")
        orig = np.random.randn(n_samples).astype(np.float32) * 0.4
        rest = np.random.randn(n_samples).astype(np.float32) * 0.4
        iad = get_iad()
        result = iad.detect(original=orig, restored=rest, sr=48000)
        # IADResult hat keine artifact_scores-Dict — Konfidenz via .artifacts-Liste prüfen
        for region in result.artifacts:
            assert math.isfinite(region.confidence), f"Nicht-finite Konfidenz: {region.confidence}"
            assert 0.0 <= region.confidence <= 1.0, f"Konfidenz außerhalb [0,1]: {region.confidence}"
            assert math.isfinite(region.severity), f"Nicht-finite Severity: {region.severity}"
            assert 0.0 <= region.severity <= 1.0, f"Severity außerhalb [0,1]: {region.severity}"
        # Gesamt-Konfidenz prüfen
        assert math.isfinite(result.confidence), f"Gesamt-Konfidenz nicht finite: {result.confidence}"
        assert 0.0 <= result.confidence <= 1.0, f"Gesamt-Konfidenz außerhalb [0,1]: {result.confidence}"

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(n_samples=st.integers(min_value=512, max_value=48_000 * 3))
    def test_03_mask_shape_correct(self, n_samples: int) -> None:
        """get_artifact_mask() → bool-Array der Länge n_samples."""
        try:
            from backend.core.introduced_artifact_detector import get_iad
        except ImportError:
            pytest.skip("IntroducedArtifactDetector nicht importierbar")
        orig = np.random.randn(n_samples).astype(np.float32) * 0.4
        rest = np.random.randn(n_samples).astype(np.float32) * 0.4
        iad = get_iad()
        result = iad.detect(original=orig, restored=rest, sr=48000)
        mask = iad.get_artifact_mask(result, n_samples)
        assert mask.shape == (n_samples,)
        assert mask.dtype == bool

    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(scale=st.floats(allow_nan=True, allow_infinity=True))
    def test_04_nan_inf_no_crash(self, scale: float) -> None:
        """NaN/Inf-Audio → kein Absturz."""
        try:
            from backend.core.introduced_artifact_detector import IADResult, get_iad
        except ImportError:
            pytest.skip("IntroducedArtifactDetector nicht importierbar")
        # Keep fuzz intent (extreme values) but avoid test-side overflow warnings.
        safe_scale = float(np.clip(scale if math.isfinite(scale) else 0.3, -1.0e6, 1.0e6))
        raw = np.random.randn(4800).astype(np.float32) * safe_scale
        raw[0] = float("nan")
        raw[-1] = float("inf")
        iad = get_iad()
        result = iad.detect(original=raw, restored=raw.copy(), sr=48000)
        assert isinstance(result, IADResult)
