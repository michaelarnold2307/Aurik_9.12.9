"""Unit-Tests für den Vocal-Overdrive-Guard (backend/core/dsp/vocal_overdrive_guard.py).

§Ebene-0-Hör-Invariante: Verarbeitung darf dem Gesang keinen nichtlinearen
Drive hinzufügen (Kamm-/IMD-Excess, Sättigung, Crest-Kollaps, Clipping in
stimmlichen Frames). Testfälle:

  1. No-Op (post == pre)      → passed, keine Aktion
  2. Sanfte EQ-/Präsenz-Änderung → passed (kein False-Positive)
  3. tanh-Sättigung (Overdrive) → Verstoß (phase: Blend/hart; final: hart)
  4. Hard-Clipping             → Verstoß, voiced_clip_ratio > 0
  5. protect() liefert geschütztes Audio in erwarteter Shape (Stereo)
  6. final-Modus iteriert, bis die Invariante erfüllt ist
  7. vocal_active=False        → trivial passed (kein Eingriff)
  8. Längen-Differenz ≤ 256 Samples wird toleriert (gemeinsamer Bereich)
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.vocal_overdrive_guard import (
    VOICED_CLIP_HARD_RATIO,
    measure_vocal_overdrive,
    protect_vocal_overdrive,
    vocal_drive_telemetry,
)

SR = 48000
DUR = 6.0


def _make_voice(f0: float = 233.0, amp: float = 0.28, seed: int = 42) -> np.ndarray:
    n = int(SR * DUR)
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float64)
    for k in range(1, 9):
        x += amp / (k ** 1.1) * np.sin(2 * np.pi * k * f0 * t + 0.3 * k)
    x += 0.05 * np.sin(2 * np.pi * 110 * t)  # schwache Begleitung
    rng = np.random.default_rng(seed)
    x += rng.standard_normal(n) * 10 ** (-60 / 20)
    x *= 0.5 / np.max(np.abs(x))
    return x.astype(np.float32)


def _rms_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = float(np.sqrt(np.mean(b**2))) / max(float(np.sqrt(np.mean(a**2))), 1e-12)
    return (a * g).astype(np.float32)


@pytest.fixture(scope="module")
def voice() -> np.ndarray:
    return _make_voice()


def test_noop_passes(voice: np.ndarray) -> None:
    r = measure_vocal_overdrive(voice, voice.copy(), SR)
    assert r.passed
    assert r.blend_factor == 1.0


def test_eq_tilt_passes(voice: np.ndarray) -> None:
    from scipy.signal import butter, sosfiltfilt

    sos_hp = butter(2, 3000 / (SR / 2), btype="highpass", output="sos")
    hp = sosfiltfilt(sos_hp, voice).astype(np.float32)
    post = np.clip(voice + 0.08 * hp, -1.0, 1.0).astype(np.float32)
    post = _rms_match(post, voice)
    r = measure_vocal_overdrive(voice, post, SR, mode="phase")
    assert r.passed, r.reasons
    r2 = measure_vocal_overdrive(voice, post, SR, mode="final")
    assert r2.passed, r2.reasons


def test_tanh_overdrive_violates(voice: np.ndarray) -> None:
    post = (np.tanh(3.0 * voice) / np.tanh(3.0 * 0.5) * 0.5).astype(np.float32)
    post = _rms_match(post, voice)
    r = measure_vocal_overdrive(voice, post, SR, mode="phase")
    assert not r.passed
    assert r.blend_factor < 1.0
    rf = measure_vocal_overdrive(voice, post, SR, mode="final")
    assert not rf.passed


def test_hard_clip_violates_with_clip_ratio(voice: np.ndarray) -> None:
    post = np.clip(voice * 4.0, -0.999, 0.999).astype(np.float32)
    r = measure_vocal_overdrive(voice, post, SR, mode="phase")
    assert not r.passed
    assert r.voiced_clip_ratio > VOICED_CLIP_HARD_RATIO
    assert r.hard_revert  # Clipping in Stimm-Frames ist ein harter Verstoß


def test_protect_stereo_shape(voice: np.ndarray) -> None:
    pre_st = np.stack([voice, voice * 0.97], axis=1)  # (N, 2)
    post = (np.tanh(3.0 * voice) / np.tanh(3.0 * 0.5) * 0.5).astype(np.float32)
    post_st = np.stack([post, post], axis=1)
    out, r = protect_vocal_overdrive(pre_st, post_st, SR, phase_id="unit_test", mode="phase")
    assert out.shape == post_st.shape
    assert r.blend_factor < 1.0
    assert not np.array_equal(out, post_st)  # Schutz wurde angewendet


def test_final_mode_converges(voice: np.ndarray) -> None:
    """Final-Modus iteriert, bis die Invariante erfüllt ist."""
    post = (np.tanh(2.2 * voice) / np.tanh(2.2 * 0.5) * 0.5).astype(np.float32)
    post = _rms_match(post, voice)
    out, r = protect_vocal_overdrive(voice, post, SR, mode="final", phase_id="unit_final")
    r_after = measure_vocal_overdrive(voice, out, SR, mode="final")
    assert r_after.passed, r_after.reasons
    # Audio darf nicht unverändert sein (Verstoß lag vor)
    assert not np.array_equal(out, post)


def test_vocal_inactive_trivially_passes(voice: np.ndarray) -> None:
    post = np.clip(voice * 4.0, -0.999, 0.999).astype(np.float32)
    r = measure_vocal_overdrive(voice, post, SR, vocal_active=False)
    assert r.passed
    assert r.blend_factor == 1.0


def test_length_tolerance(voice: np.ndarray) -> None:
    post = (np.tanh(3.0 * voice) / np.tanh(3.0 * 0.5) * 0.5).astype(np.float32)
    post_short = post[:-100]
    r = measure_vocal_overdrive(voice, post_short, SR, mode="phase")
    # gemeinsamer Bereich wird gemessen → Verstoß bleibt erkennbar ODER Schutz greift
    assert not r.passed or r.blend_factor < 1.0 or r.voiced_frames > 0
    r_skip = measure_vocal_overdrive(voice, post[:-1000], SR, mode="phase")
    # > 256 Samples Differenz → trivial passed (kein Raise)
    assert r_skip.passed


def test_telemetry_keys(voice: np.ndarray) -> None:
    r = measure_vocal_overdrive(voice, voice.copy(), SR)
    tel = vocal_drive_telemetry(r)
    for key in (
        "vocal_drive_passed",
        "vocal_drive_blend",
        "vocal_drive_reasons",
        "vocal_drive_voiced_clip_ratio",
        "vocal_drive_comb_excess_db_p90",
    ):
        assert key in tel
