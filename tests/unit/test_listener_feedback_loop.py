"""ListenerFeedbackLoop — Unit-Tests für Listener-in-the-Loop Feedback-Schleife.

Testet Segment-Erstellung zur Review, Feedback-Speicherung, low_score_segments-Erkennung
und FeedbackResult-Gesamtbewertung.

Spec: AGENTS.md / backend/core/listener_feedback_loop.py
"""

from __future__ import annotations

import numpy as np
import pytest


def _audio(sr: int = 48000, duration: float = 2.0, freq: float = 440.0) -> np.ndarray:
    """Erzeugt Test-Audio (Sinus)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.unit
class TestListenerFeedbackLoop:
    """Listener-in-the-Loop Feedback-Schleife funktioniert korrekt."""

    def setup_method(self) -> None:
        """Reset Singleton vor jedem Test."""
        import backend.core.listener_feedback_loop as lfl_module
        lfl_module._feedback_instance = None  # pylint: disable=protected-access

    def test_singleton_factory(self):
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        f1 = get_listener_feedback_loop()
        f2 = get_listener_feedback_loop()
        assert f1 is f2, "Singleton-Factory sollte dieselbe Instanz zurückgeben"

    def test_get_segments_for_review(self):
        """get_segments_for_review() sollte ReviewSegment-Liste zurückgeben."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)  # 6s Audio → mehrere Segmente
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")
        assert len(segments) >= 3, "6s Audio sollte mindestens 3 Segmente (je 2s) produzieren"

    def test_segment_attributes(self):
        """ReviewSegment hat alle erwarteten Attribute."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")
        if segments:
            seg = segments[0]
            assert hasattr(seg, "segment_id")
            assert hasattr(seg, "start_sample")
            assert hasattr(seg, "end_sample")
            assert hasattr(seg, "start_time_s")
            assert hasattr(seg, "duration_s")
            assert hasattr(seg, "audio_restored")
            assert hasattr(seg, "audio_original")
            assert hasattr(seg, "user_score")
            assert seg.user_score is None  # Noch nicht bewertet

    def test_record_feedback(self):
        """record_feedback() sollte Benutzerbewertung speichern."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")
        seg = segments[0]

        feedback.record_feedback(seg.segment_id, 8.5)
        # Score sollte gespeichert sein
        assert feedback._segments[seg.segment_id].user_score == 8.5

    def test_low_score_segments(self):
        """get_low_score_segments() sollte Segmente < threshold zurückgeben."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")

        # Verschiedene Scores zuweisen
        for i, seg in enumerate(segments):
            score = 5.0 if i % 2 == 0 else 9.0  # Abwechselnd schlecht/gut
            feedback.record_feedback(seg.segment_id, score)

        bad = feedback.get_low_score_segments(threshold=6.0)
        # Alle zurückgegebenen Segmente sollten < 6.0 sein
        for s in bad:
            assert s.user_score is not None and s.user_score < 6.0

    def test_feedback_result(self):
        """get_feedback_result() sollte korrekte Gesamtbewertung zurückgeben."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop, FeedbackResult

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")

        # Alle Segmente mit Score 7.0 bewerten
        for seg in segments:
            feedback.record_feedback(seg.segment_id, 7.0)

        result = feedback.get_feedback_result(threshold=6.0)
        assert isinstance(result, FeedbackResult)
        assert result.mean_score == pytest.approx(7.0, abs=0.01)
        assert result.n_segments == len(segments)
        assert result.n_low_score == 0  # Alle >= 6.0
        assert not result.needs_rerestoration

    def test_needs_rerestoration_true(self):
        """needs_rerestoration sollte True sein wenn Segmente < threshold."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")

        # Alle Segmente mit Score < 6.0 bewerten
        for seg in segments:
            feedback.record_feedback(seg.segment_id, 4.0)

        result = feedback.get_feedback_result(threshold=6.0)
        assert result.needs_rerestoration


@pytest.mark.unit
class TestListenerFeedbackLoopEdgeCases:
    """Edge-Cases für Listener-Feedback-Schleife."""

    def setup_method(self) -> None:
        """Reset Singleton vor jedem Test."""
        import backend.core.listener_feedback_loop as lfl_module
        lfl_module._feedback_instance = None  # pylint: disable=protected-access

    def test_no_segments_scored(self):
        """Keine bewerteten Segmente sollte konservatives FeedbackResult zurückgeben."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop, FeedbackResult

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        result = feedback.get_feedback_result(threshold=6.0)

        assert isinstance(result, FeedbackResult)
        assert result.n_segments == 0
        assert not result.needs_rerestoration

    def test_unknown_segment_id(self):
        """Unbekannte segment_id sollte sicher behandelt werden."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        # Sollte keine Exception werfen
        feedback.record_feedback("unknown_seg_999", 5.0)

    def test_score_clamped_to_range(self):
        """Score sollte in [0, 10] geklammert sein."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")
        seg = segments[0]

        # Score > 10 → auf 10 clampen
        feedback.record_feedback(seg.segment_id, 15.0)
        assert feedback._segments[seg.segment_id].user_score == pytest.approx(10.0, abs=0.01)

        # Score < 0 → auf 0 clampen
        feedback.record_feedback(seg.segment_id, -3.0)
        assert feedback._segments[seg.segment_id].user_score == pytest.approx(0.0, abs=0.01)

    def test_nan_handling(self):
        """NaN/Inf-Werte in Audio sollten sicher behandelt werden."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        restored[100] = np.nan
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")
        assert len(segments) >= 3

    def test_segment_duration_clamped(self):
        """segment_duration_s sollte in [1, 30] geklammert sein."""
        from backend.core.listener_feedback_loop import ListenerFeedbackLoop

        # Zu klein → auf 1.0 clampen
        feedback = ListenerFeedbackLoop(segment_duration_s=0.5)
        assert feedback._segment_s == 1.0

        # Zu groß → auf 30.0 clampen
        feedback = ListenerFeedbackLoop(segment_duration_s=60.0)
        assert feedback._segment_s == 30.0

    def test_low_segments_sorted_by_score(self):
        """get_low_score_segments sollte nach Score aufsteigend sortiert sein."""
        from backend.core.listener_feedback_loop import get_listener_feedback_loop

        feedback = get_listener_feedback_loop(segment_duration_s=2.0)
        restored = _audio(48000, 6.0)
        original = _audio(48000, 6.0, freq=220.0)

        segments = feedback.get_segments_for_review(restored, original, sr=48000, song_id="test_001")

        # Verschiedene Scores < 6.0
        for i, seg in enumerate(segments):
            score = 3.0 + i * 1.5 if (3.0 + i * 1.5) < 6.0 else 2.0
            feedback.record_feedback(seg.segment_id, score)

        bad = feedback.get_low_score_segments(threshold=6.0)
        if len(bad) >= 2:
            for i in range(len(bad) - 1):
                assert bad[i].user_score is not None and bad[i + 1].user_score is not None
                assert bad[i].user_score <= bad[i + 1].user_score
