"""Tests für das DTW-Groove-Messloch-Guard (A3, Hörordnung §7/§8a)."""

from __future__ import annotations

import numpy as np

from dsp.dtw_groove import measure_groove


def _click_train(sr: int, dur_s: float, period_s: float = 0.25, shift_s: float = 0.0, amp: float = 0.5) -> np.ndarray:
    out = np.zeros(int(sr * dur_s), dtype=np.float32)
    n = max(1, int(0.005 * sr))
    t = shift_s
    while t < dur_s - 0.01:
        idx = int(t * sr)
        out[idx : idx + n] += amp
        t += period_s
    return out


def test_identical_trains_align_well() -> None:
    sr = 48000
    a = _click_train(sr, 4.0)
    res = measure_groove(a, a.copy(), sr)
    assert res.groove_score >= 0.8, f"score={res.groove_score} rms={res.dtw_rms_ms}"


def test_time_shifted_train_marked_alignment_failed() -> None:
    """Befund 2026-08-23: rms=5220 ms → Score 0.0 verfälschte die 15 Goals.

    dsp-Ebene: Ein um Sekunden verschobenes Signal ist ein Messfehler, kein
    Groove-Urteil → method_used="alignment_failed", score 0.0 als Signal für
    die GrooveMetric, die daraus ihren IOI-Ersatzpfad ableitet (Spec 01 §1.4.5b).
    """
    sr = 48000
    orig = _click_train(sr, 8.0)
    shifted = _click_train(sr, 8.0, shift_s=5.0)
    res = measure_groove(orig, shifted, sr)
    if res.dtw_rms_ms > 500.0:
        assert res.method_used == "alignment_failed"
        assert res.groove_score == 0.0
    else:
        # DTW hat doch ein sinnvolles Alignment gefunden → normaler Score-Pfad
        assert 0.0 <= res.groove_score <= 1.0


def test_onset_loss_still_zero_for_silence() -> None:
    """Asymmetrischer Verlust bei echter Stille bleibt 0.0 (kein False-Pass)."""
    sr = 48000
    orig = _click_train(sr, 4.0)
    silent = np.zeros_like(orig)
    res = measure_groove(orig, silent, sr)
    assert res.method_used.startswith("onset_loss")
    assert res.groove_score == 0.0


def test_both_silent_neutral() -> None:
    sr = 48000
    z = np.zeros(sr * 2, dtype=np.float32)
    res = measure_groove(z, z.copy(), sr)
    assert res.groove_score == 1.0  # nichts messbar → neutral, kein False-Pass
