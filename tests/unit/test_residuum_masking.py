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


# ---------------------------------------------------------------------------
# §2.46h (2026-09-06): Vektorisierung + Batch-Pfad
# ---------------------------------------------------------------------------


def test_stft_vectorization_bit_identical() -> None:
    """§2.46h-1: Der strided-batch STFT liefert bit-identische Werte zum
    alten Per-Frame-rfft (gleiche Fenster, gleicher pocketfft-Algorithmus)."""
    from backend.core import residuum_masking as rm

    sr = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(sr * 3).astype(np.float64)

    hop = max(rm._N_FFT // 4, 1)
    xx = np.pad(x, (0, rm._N_FFT - len(x))) if len(x) < rm._N_FFT else x
    n_frames = 1 + (len(xx) - rm._N_FFT) // hop
    freqs = np.fft.rfftfreq(rm._N_FFT, 1.0 / sr)
    out = np.zeros((n_frames, len(freqs)), dtype=np.float64)
    win = np.hanning(rm._N_FFT)
    for f in range(n_frames):
        seg = xx[f * hop : f * hop + rm._N_FFT] * win
        out[f, :] = 20.0 * np.log10(np.abs(np.fft.rfft(seg)) + 1e-12)

    new_frames, new_freqs = rm._stft_magnitude_db(x, sr)
    assert np.array_equal(out, new_frames)
    assert np.array_equal(freqs, new_freqs)


def test_bark_bands_bit_identical() -> None:
    """§2.46h-2: Gecachte Bin-Indizes liefern bit-identische Bark-Mediane."""
    from backend.core import residuum_masking as rm

    sr = 48000
    rng = np.random.default_rng(43)
    spec = rng.standard_normal((40, 2049)).astype(np.float64) * 20.0 - 60.0
    freqs = np.fft.rfftfreq(rm._N_FFT, 1.0 / sr)

    old = np.zeros(len(rm._BARK_CENTERS), dtype=np.float64)
    for b in range(len(rm._BARK_CENTERS)):
        mask = (freqs >= rm._BARK_EDGES_HZ[b]) & (freqs < rm._BARK_EDGES_HZ[b + 1])
        old[b] = float(np.median(spec[:, mask])) if np.any(mask) else -120.0
    new = rm._to_bark_bands(spec, freqs)
    assert np.array_equal(old, new)


def test_batch_deterministic_and_same_semantics() -> None:
    """§2.46h-3: Der Batch-Pfad ist deterministisch, liefert [0,1]-Salienzen
    und für einen einzelnen Event exakt den Per-Event-Wert (Fallback)."""
    from backend.core.residuum_masking import estimate_residuum_salience_batch

    sr = 48000
    audio = _click_at(_noise(sr, 6.0, rms=0.2, seed=4), sr, 3.0, amp=0.2)
    loc = (2.99, 3.01)

    single = estimate_residuum_salience_batch(audio, sr, [loc])
    assert set(single.keys()) == {loc}
    assert single[loc].salience == estimate_residuum_salience(audio, sr, 2.99, 3.01).salience

    locs = [(1.0, 1.01), (2.0, 2.01), (3.0, 3.01), (4.0, 4.01), (5.0, 5.01)]
    r1 = estimate_residuum_salience_batch(audio, sr, locs)
    r2 = estimate_residuum_salience_batch(audio, sr, locs)
    assert set(r1.keys()) == set(locs)
    assert all(0.0 <= r.salience <= 1.0 for r in r1.values())
    assert all(np.array_equal(r1[loc].residuum_db_per_band, r2[loc].residuum_db_per_band) for loc in locs)
