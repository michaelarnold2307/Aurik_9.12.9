"""Tests für das Einladungs-Gate (Hörordnung Ebene 4, §6)."""

from __future__ import annotations

import numpy as np

from backend.core.inviting_sound_gate import (
    InvitingGateResult,
    check_inviting_gate,
    compute_sharpness_acum,
    get_inviting_gate,
)


def _sine(sr: int = 48000, dur_s: float = 12.0, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_smooth_sine_passes() -> None:
    sr = 48000
    res = check_inviting_gate(_sine(sr), sr, fatigue_index=0.1)
    assert res.passed is True
    assert res.max_asper_in_voice <= 0.5
    assert res.n_windows >= 2


def test_am_roughness_detected_vs_sine() -> None:
    sr = 48000
    t = np.arange(sr * 12) / sr
    am = (0.3 * (1 + 0.9 * np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    res_am = check_inviting_gate(am, sr, fatigue_index=0.0)
    res_sine = check_inviting_gate(_sine(sr), sr, fatigue_index=0.0)
    # AM (70 Hz Modulation = Rauigkeit) muss deutlich rauer sein als Sinus
    assert res_am.max_asper > res_sine.max_asper + 0.05


def test_fatigue_abort_flag() -> None:
    sr = 48000
    res = check_inviting_gate(_sine(sr, dur_s=8.0), sr, fatigue_index=0.55)
    assert res.fatigue_abort is True
    res_ok = check_inviting_gate(_sine(sr, dur_s=8.0), sr, fatigue_index=0.1)
    assert res_ok.fatigue_abort is False


def test_sharpness_plausibility() -> None:
    sr = 48000
    rng = np.random.default_rng(7)
    bright = np.diff(np.cumsum(rng.standard_normal(sr * 6 + 1))).astype(np.float32) * 0.2
    sine_sharp = compute_sharpness_acum(_sine(sr, dur_s=6.0), sr)
    bright_sharp = compute_sharpness_acum(bright, sr)
    assert 0.0 <= sine_sharp <= 4.0
    assert bright_sharp > sine_sharp  # helles Rauschen ist schärfer als ein 440-Hz-Sinus


def test_short_audio_skipped_gracefully() -> None:
    sr = 48000
    res = check_inviting_gate(np.zeros(sr // 4, dtype=np.float32), sr)
    assert res.details.get("skipped") is not None


def test_singleton_and_defaults() -> None:
    gate = get_inviting_gate()
    assert gate is get_inviting_gate()
    r = InvitingGateResult()
    assert r.passed is True
    assert r.fatigue_abort is False
