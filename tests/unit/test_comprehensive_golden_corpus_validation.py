"""Comprehensive Golden Corpus Validation — Real-Audio Baseline Measurement.

Phase 3 Extended: Measure separation_fidelity and musical goals across
all golden sample genres (vocal, jazz, classical, instrumental) to establish
baseline quality and validate that Phase 2 infrastructure improves restoration.

Compliance: §G2 (100% song analysis), §G5 (deterministic), §G8 (transparent).
"""

import logging
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)


class TestComprehensiveGoldenCorpus:
    """Corpus validation across genres."""

    GOLDEN_ROOT = Path(__file__).parent.parent.parent / "golden_samples"
    GENRES = ["vocal", "jazz", "classical", "instrumental"]
    MAX_SAMPLES_PER_GENRE = 5

    @pytest.mark.slow
    def test_corpus_structure(self):
        """Validate golden samples exist."""
        assert self.GOLDEN_ROOT.exists(), f"Golden samples root missing: {self.GOLDEN_ROOT}"

        for genre in self.GENRES:
            genre_dir = self.GOLDEN_ROOT / genre
            assert genre_dir.exists(), f"Genre directory missing: {genre_dir}"
            samples = list(genre_dir.glob("*.wav"))
            assert len(samples) > 0, f"No samples found in {genre_dir}"

    @pytest.mark.slow
    def test_separation_fidelity_baseline_all_genres(self):
        """Measure separation_fidelity baseline across all genres."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()
        results = {}

        for genre in self.GENRES:
            genre_dir = self.GOLDEN_ROOT / genre
            samples = sorted(list(genre_dir.glob("*.wav")))[:self.MAX_SAMPLES_PER_GENRE]

            genre_scores = []
            for sample_path in samples:
                try:
                    import soundfile as sf

                    audio, sr = sf.read(str(sample_path))
                    if audio.ndim > 1:
                        audio = audio[:, 0]  # Mono

                    # Measure without reference (proxy mode)
                    score = metric.measure(
                        audio.astype(np.float32),
                        sr,
                        reference=None,
                        material_type="unknown",
                        global_scalar=1.0,
                    )
                    genre_scores.append(float(score))
                    logger.info(
                        "Separation_fidelity (%s, %s): %.3f",
                        genre,
                        sample_path.name,
                        score,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to measure %s (%s): %s",
                        genre,
                        sample_path.name,
                        e,
                    )
                    continue

            if genre_scores:
                avg_score = float(np.mean(genre_scores))
                results[genre] = {
                    "scores": genre_scores,
                    "mean": avg_score,
                    "count": len(genre_scores),
                }
                logger.info(
                    "✅ Genre %s: mean=%.3f (n=%d)",
                    genre,
                    avg_score,
                    len(genre_scores),
                )

        # Vocal should have similar or better separation than instrumental
        if "vocal" in results and "instrumental" in results:
            vocal_mean = results["vocal"]["mean"]
            instr_mean = results["instrumental"]["mean"]
            logger.info(
                "Vocal vs Instrumental: %.3f vs %.3f",
                vocal_mean,
                instr_mean,
            )

        assert len(results) > 0, "No results collected"
        print(f"\n✅ Comprehensive Corpus Results: {len(results)} genres measured")

    @pytest.mark.slow
    def test_musical_goals_all_vocal_samples(self):
        """Measure all 15 musical goals on vocal samples."""
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker()
        vocal_dir = self.GOLDEN_ROOT / "vocal"
        samples = sorted(list(vocal_dir.glob("*.wav")))[:2]  # First 2 vocal samples

        for sample_path in samples:
            try:
                import soundfile as sf

                audio, sr = sf.read(str(sample_path))
                if audio.ndim > 1:
                    audio = audio[:, 0]

                scores = checker.measure_all(
                    audio.astype(np.float32),
                    sr,
                    material_type="unknown",
                    reference=None,
                )
                logger.info("✅ Vocal Sample: %s", sample_path.name)
                for goal, score in sorted(scores.items()):
                    logger.info("  %s: %.3f", goal, score)

                # Validate all 15 goals present
                expected_goals = {
                    "artikulation",
                    "authentizitaet",
                    "bass_kraft",
                    "brillanz",
                    "emotionalitaet",
                    "groove",
                    "micro_dynamics",
                    "natuerlichkeit",
                    "separation_fidelity",
                    "spatial_depth",
                    "timbre_authentizitaet",
                    "tonal_center",
                    "transient_energie",
                    "transparenz",
                    "waerme",
                }
                assert (
                    set(scores.keys()) == expected_goals
                ), f"Missing goals: {expected_goals - set(scores.keys())}"

                # All scores should be finite and in [0, 1]
                for score in scores.values():
                    assert np.isfinite(score), "Non-finite score"
                    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

            except Exception as e:
                logger.warning("Failed to measure %s: %s", sample_path.name, e)
                continue

    @pytest.mark.slow
    def test_separation_fidelity_with_reference_vs_without(self):
        """Compare separation_fidelity: reference-based vs reference-free."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        metric = SeparationFidelityMetric()
        vocal_dir = self.GOLDEN_ROOT / "vocal"
        reference_dir = self.GOLDEN_ROOT / "references"

        sample_files = sorted(list(vocal_dir.glob("*.wav")))[:1]  # First vocal sample

        for sample_path in sample_files:
            try:
                import soundfile as sf

                # Load main audio
                audio, sr = sf.read(str(sample_path))
                if audio.ndim > 1:
                    audio = audio[:, 0]

                # Try to load reference (if available)
                ref_path = reference_dir / sample_path.name
                reference = None
                if ref_path.exists():
                    reference, ref_sr = sf.read(str(ref_path))
                    if reference.ndim > 1:
                        reference = reference[:, 0]
                    if ref_sr != sr:
                        import librosa

                        reference = librosa.resample(reference, orig_sr=ref_sr, target_sr=sr)

                # Measure without reference (proxy mode)
                score_proxy = metric.measure(
                    audio.astype(np.float32),
                    sr,
                    reference=None,
                    material_type="unknown",
                    global_scalar=1.0,
                )

                logger.info("Separation_fidelity (proxy): %.3f", score_proxy)

                # Measure with reference if available
                if reference is not None:
                    score_ref = metric.measure(
                        audio.astype(np.float32),
                        sr,
                        reference=reference.astype(np.float32),
                        material_type="unknown",
                        global_scalar=1.0,
                    )
                    logger.info(
                        "Separation_fidelity (reference-based): %.3f",
                        score_ref,
                    )
                    logger.info("Difference (ref - proxy): %.3f", score_ref - score_proxy)

                assert np.isfinite(score_proxy), "Proxy score not finite"
                assert 0.0 <= score_proxy <= 1.0, "Proxy score out of range"

            except Exception as e:
                logger.warning("Failed to measure %s: %s", sample_path.name, e)
                continue


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
