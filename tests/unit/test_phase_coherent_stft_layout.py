"""Regressionstests für den §v10.303 2-Sample-Kollaps in PhaseCoherentSTFT.

Befund (Produktion 2026-08): Bei channels-first Stereo (2, N) kollabierte
restore_phase_coherence() das Signal auf 2 Samples, weil das Längen-Matching
_input_shape[0] (= Kanalzahl 2) statt der Audiolänge verwendete. Folgekette:
STCG spread=9999, FC in 0.04 s, measure_all auf length=2 → alle 15 Goals 0.000
→ MUSHRA 40.7 → QUALITY GATES FAILED → Wohlklang-Garantie-Re-Run.
"""

import numpy as np
import pytest

from backend.core.dsp.phase_coherent_stft import restore_phase_coherence


def _make_stereo(n_samples: int, sr: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, n_samples)) * 0.1).astype(np.float32)


@pytest.mark.unit
def test_restore_channels_first_keeps_full_length() -> None:
    """Regressionsfall: (2, N) → (2, N), niemals 2 Samples."""
    sr = 48000
    n = sr * 3
    orig = _make_stereo(n, sr)
    processed = orig + _make_stereo(n, sr, seed=8) * 0.02
    out = restore_phase_coherence(degraded_reference=orig, processed_audio=processed, sample_rate=sr)
    assert out.shape == (2, n), f"2-Sample-Kollaps: {out.shape}"


@pytest.mark.unit
def test_restore_channels_last_keeps_full_length() -> None:
    """(N, 2) → (N, 2)."""
    sr = 48000
    n = sr * 2
    orig = _make_stereo(n, sr).T
    processed = orig.copy()
    out = restore_phase_coherence(degraded_reference=orig, processed_audio=processed, sample_rate=sr)
    assert out.shape == (n, 2), f"Layout-/Längenfehler: {out.shape}"


@pytest.mark.unit
def test_restore_mono_keeps_full_length() -> None:
    """(N,) → (N,)."""
    sr = 48000
    n = sr * 2
    rng = np.random.default_rng(11)
    orig = (rng.standard_normal(n) * 0.1).astype(np.float32)
    processed = orig.copy()
    out = restore_phase_coherence(degraded_reference=orig, processed_audio=processed, sample_rate=sr)
    assert out.shape == (n,), f"Längenfehler mono: {out.shape}"


@pytest.mark.unit
def test_restore_output_is_finite() -> None:
    sr = 48000
    n = sr
    orig = _make_stereo(n, sr)
    out = restore_phase_coherence(degraded_reference=orig, processed_audio=orig, sample_rate=sr)
    assert np.isfinite(out).all()


@pytest.mark.unit
def test_restore_empty_input_passthrough() -> None:
    """Leeres/winziges Eingangssignal darf nicht crashen."""
    sr = 48000
    tiny = np.zeros((2, 64), dtype=np.float32)
    out = restore_phase_coherence(degraded_reference=tiny, processed_audio=tiny, sample_rate=sr)
    assert out.shape == (2, 64)
