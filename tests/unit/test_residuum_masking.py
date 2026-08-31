"""Tests für das Residuum-basierte Bark-Masking (Hörordnung Ebene 2)."""

from __future__ import annotations

import numpy as np

from backend.core.residuum_masking import ResiduumMaskingResult, estimate_residuum_salience


def _noise(sr: int, dur_s: float, rms: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(sr * dur_s))
    return (x / (np.sqrt(np.mean(x**2)) + 1e-12) * rms).astype(np.float64)


def _click_at(audio: np.ndarray, sr: int, t: float, amp: float) -> np.ndarray:
    idx = int(t * sr)
    out = audio.copy()
    n = max(1, int(0.002 * sr))
    out[idx : idx + n] += amp
    return out


def test_loud_context_masks_quiet_click() -> None:
    sr = 48000
    base = _noise(sr, 6.0, rms=0.2, seed=1)  # lauter maskierender Kontext
    audio = _click_at(base, sr, 3.0, amp=0.05)  # leiser Click (Residuum klein)
    res = estimate_residuum_salience(audio, sr, 2.99, 3.01)
    assert 0.0 <= res.salience <= 1.0
    # Residuum klein gegenüber Kontext → überwiegend maskiert
    assert res.salience < 0.7, f"salience={res.salience}"


def test_silent_context_exposes_click() -> None:
    sr = 48000
    base = np.zeros(sr * 6, dtype=np.float64)  # stiller Kontext
    audio = _click_at(base, sr, 3.0, amp=0.4)
    res = estimate_residuum_salience(audio, sr, 2.99, 3.01)
    assert res.salience >= 0.5, f"salience={res.salience}"


def test_monotonicity_snr_decrease_raises_salience() -> None:
    sr = 48000
    base = _noise(sr, 6.0, rms=0.2, seed=2)
    quiet = _click_at(base, sr, 3.0, amp=0.05)
    loud = _click_at(base, sr, 3.0, amp=0.5)
    s_quiet = estimate_residuum_salience(quiet, sr, 2.99, 3.01).salience
    s_loud = estimate_residuum_salience(loud, sr, 2.99, 3.01).salience
    # Lauterer Defekt bei gleichem Kontext → höhere Hörbarkeit
    assert s_loud >= s_quiet, f"{s_quiet=} {s_loud=}"


def test_deterministic() -> None:
    sr = 48000
    audio = _click_at(_noise(sr, 6.0, rms=0.15, seed=3), sr, 3.0, amp=0.2)
    r1 = estimate_residuum_salience(audio, sr, 2.99, 3.01)
    r2 = estimate_residuum_salience(audio, sr, 2.99, 3.01)
    assert r1.salience == r2.salience
    assert np.array_equal(r1.residuum_db_per_band, r2.residuum_db_per_band)


def test_short_audio_conservative_exposed() -> None:
    sr = 48000
    short = np.zeros(sr // 10, dtype=np.float64)  # 0.1s: Kontext ums Event < N_FFT
    res = estimate_residuum_salience(short, sr, 0.04, 0.06)
    assert res.salience == 1.0  # konservativ: exponiert, kein Skip-Risiko


def test_result_struct() -> None:
    r = ResiduumMaskingResult(
        salience=0.4,
        residuum_db_per_band=np.zeros(24),
        threshold_db_per_band=np.zeros(24),
        audible_band_count=3,
        band_count=24,
    )
    assert 0.0 <= r.salience <= 1.0
    assert r.band_count == 24
