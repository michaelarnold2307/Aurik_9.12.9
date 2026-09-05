"""§v10.101 SegmentQualityScorer — Unit-Tests für segment-weise Qualitätsbewertung.

Testet gleitende Fenster, Score-Berechnung, bad_segments-Erkennung und Edge-Cases.

Spec: .github/specs/18_non_plus_ultra_perceptual_fidelity.md §v10.101
"""

from __future__ import annotations

import numpy as np
import pytest


def _audio(sr: int = 48000, duration: float = 2.0, freq: float = 440.0) -> np.ndarray:
    """Erzeugt Test-Audio (Sinus)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noisy_audio(sr: int = 48000, duration: float = 2.0) -> np.ndarray:
    """Erzeugt Test-Audio mit Rauschen."""
    base = _audio(sr, duration)
    noise = np.random.randn(len(base)).astype(np.float32) * 0.15
    return (base + noise).astype(np.float32)


@pytest.mark.unit
class TestSegmentQualityScorer:
    """Segment-weise Qualitätsbewertung funktioniert korrekt."""

    def setup_method(self) -> None:
        """Reset Singleton vor jedem Test."""
        import backend.core.segment_quality_scorer as sqs_module
        sqs_module._scorer_instance = None  # pylint: disable=protected-access

    def test_singleton_factory(self):
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        s1 = get_segment_quality_scorer()
        s2 = get_segment_quality_scorer()
        assert s1 is s2, "Singleton-Factory sollte dieselbe Instanz zurückgeben"

    def test_score_returns_segments(self):
        """score() sollte SegmentScore-Liste zurückgeben."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        audio = _audio(48000, 5.0)  # 5s Audio → mehrere Segmente

        scores = scorer.score(audio, sr=48000)
        assert len(scores) >= 2, "5s Audio sollte mindestens 2 Segmente produzieren"

    def test_score_bounds(self):
        """Alle Scores liegen in [0, 1]."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        audio = _audio(48000, 5.0)

        scores = scorer.score(audio, sr=48000)
        for s in scores:
            assert 0.0 <= s.score <= 1.0, f"Score={s.score} für Segment {s.segment_id}"

    def test_bad_segments_below_threshold(self):
        """get_bad_segments() sollte Segmente < threshold zurückgeben."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        # Sehr rauschendes Audio → niedrige Scores
        audio = _noisy_audio(48000, 5.0) * 0.1  # Amplitude reduzieren

        bad = scorer.get_bad_segments(audio, sr=48000, threshold=0.7)
        # Alle zurückgegebenen Segmente sollten < 0.7 sein
        for s in bad:
            assert s.score < 0.7, f"Segment {s.segment_id} Score={s.score} sollte < 0.7 sein"

    def test_segment_overlap(self):
        """Segmente sollten sich überlappen (50% Hop)."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        audio = _audio(48000, 6.0)  # 6s Audio

        scores = scorer.score(audio, sr=48000)
        if len(scores) >= 2:
            # Zweites Segment sollte um 50% (1s) nach dem ersten beginnen
            expected_hop = int(1.0 * 48000)
            actual_hop = scores[1].start_sample - scores[0].start_sample
            assert abs(actual_hop - expected_hop) < 100, f"Hop sollte ~{expected_hop} sein, war {actual_hop}"

    def test_segment_attributes(self):
        """SegmentScore hat alle erwarteten Attribute."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        audio = _audio(48000, 5.0)

        scores = scorer.score(audio, sr=48000)
        if scores:
            s = scores[0]
            assert hasattr(s, "start_sample")
            assert hasattr(s, "end_sample")
            assert hasattr(s, "start_time_s")
            assert hasattr(s, "duration_s")
            assert hasattr(s, "score")
            assert hasattr(s, "rms_dbfs")
            assert hasattr(s, "peak_dbfs")
            assert hasattr(s, "snr_estimate_db")

    def test_window_seconds_clamped(self):
        """window_seconds sollte in [1, 30] geklammert sein."""
        from backend.core.segment_quality_scorer import SegmentQualityScorer

        # Zu klein → auf 1.0 clampen
        scorer = SegmentQualityScorer(window_seconds=0.5)
        assert scorer.window_seconds == 1.0

        # Zu groß → auf 30.0 clampen
        scorer = SegmentQualityScorer(window_seconds=60.0)
        assert scorer.window_seconds == 30.0


@pytest.mark.unit
class TestSegmentQualityScorerEdgeCases:
    """Edge-Cases für segment-weise Qualitätsbewertung."""

    def setup_method(self) -> None:
        """Reset Singleton vor jedem Test."""
        import backend.core.segment_quality_scorer as sqs_module
        sqs_module._scorer_instance = None  # pylint: disable=protected-access

    def test_audio_too_short(self):
        """Audio kürzer als Fenster sollte leere Liste zurückgeben."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=5.0)
        short = _audio(48000, 1.0)  # 1s < 5s Fenster

        scores = scorer.score(short, sr=48000)
        assert len(scores) == 0

    def test_silent_audio(self):
        """Stille sollte konservative Scores zurückgeben."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        silent = np.zeros(96000, dtype=np.float32)  # 2s Stille

        scores = scorer.score(silent, sr=48000)
        for s in scores:
            assert 0.0 <= s.score <= 1.0

    def test_nan_handling(self):
        """NaN/Inf-Werte sollten sicher behandelt werden."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        audio = _audio(48000, 5.0)
        audio[100] = np.nan
        audio[200] = np.inf

        scores = scorer.score(audio, sr=48000)
        for s in scores:
            assert np.isfinite(s.score), "Score sollte NaN/Inf-frei sein"

    def test_stereo_audio(self):
        """Stereo-Audio (2, N) sollte korrekt verarbeitet werden."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        t = np.linspace(0, 5.0, int(48000 * 5), endpoint=False)
        stereo = np.stack([
            0.3 * np.sin(2 * np.pi * 440.0 * t),
            0.25 * np.sin(2 * np.pi * 440.0 * t + 0.1),
        ]).astype(np.float32)  # (2, N) channel-first

        scores = scorer.score(stereo, sr=48000)
        assert len(scores) >= 2
        for s in scores:
            assert 0.0 <= s.score <= 1.0

    def test_bad_segments_sorted_by_score(self):
        """bad_segments sollte nach Score aufsteigend sortiert sein."""
        from backend.core.segment_quality_scorer import get_segment_quality_scorer

        scorer = get_segment_quality_scorer(window_seconds=2.0)
        audio = _noisy_audio(48000, 5.0) * 0.1

        bad = scorer.get_bad_segments(audio, sr=48000, threshold=0.7)
        if len(bad) >= 2:
            for i in range(len(bad) - 1):
                assert bad[i].score <= bad[i + 1].score, "bad_segments sollte aufsteigend sortiert sein"
