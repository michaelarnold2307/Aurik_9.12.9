"""MCD-Golden-Tests: beide Implementierungen müssen identisch rechnen (Spec 24).

Root-Fix-Regression 2026-08-16: mushra_evaluator._compute_mcd (CMVN) und
mert_mushra_proxy._compute_mcd (vorher rohe MFCCs → 435.2 dB statt 26.4 dB)
divergierten. Golden-Werte: identisches Signal = 0 dB, komplett verschiedene
Klangfarbe ≤ 40 dB (Cap), und beide Implementierungen müssen konsistent sein.
"""

from __future__ import annotations

import numpy as np
import pytest


def _noise(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples).astype(np.float32)
    return x / (np.abs(x).max() + 1e-6)


def _lowpass(x: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = alpha * acc + (1.0 - alpha) * x[i]
        y[i] = acc
    return y


@pytest.fixture(scope="module")
def evaluator_mcd():
    from backend.core.mushra_evaluator import MushraEvaluator

    _ev = MushraEvaluator.__new__(MushraEvaluator)
    return lambda ref, test, sr: MushraEvaluator._compute_mcd(_ev, ref, test, sr)


@pytest.fixture(scope="module")
def proxy_mcd():
    from backend.core.mert_mushra_proxy import MertMushraProxy

    return lambda ref, test, sr: MertMushraProxy._compute_mcd(ref, test, sr)


SR = 48000


def test_mcd_identical_signal_is_zero(evaluator_mcd, proxy_mcd) -> None:
    x = _noise(SR, 42)
    for mcd_fn in (evaluator_mcd, proxy_mcd):
        mcd = mcd_fn(x, x, SR)
        assert mcd == pytest.approx(0.0, abs=0.01), f"identisch: MCD={mcd:.4f} ≠ 0"


def test_mcd_timbre_difference_in_physical_range(evaluator_mcd, proxy_mcd) -> None:
    x = _noise(SR, 42)
    y = _lowpass(x, alpha=0.97)  # starker Tiefpass → deutlicher Klangfarben-Unterschied
    for mcd_fn in (evaluator_mcd, proxy_mcd):
        mcd = mcd_fn(x, y, SR)
        assert 0.5 <= mcd <= 40.0, f"Klangfarben-Differenz außerhalb 0–40 dB: {mcd:.2f}"


def test_mcd_cap_for_completely_different_timbre(evaluator_mcd, proxy_mcd) -> None:
    x = _noise(SR, 42)
    t = np.arange(SR) / SR
    sine = (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    for mcd_fn in (evaluator_mcd, proxy_mcd):
        mcd = mcd_fn(x, sine, SR)
        assert mcd <= 40.0 + 1e-6, f"Cap verletzt: {mcd:.2f}"


def test_both_implementations_agree(evaluator_mcd, proxy_mcd) -> None:
    """Konsistenz-Korridor: identische CMVN-Formel → max. 1 dB Abweichung."""
    x = _noise(SR, 42)
    y = _lowpass(x, alpha=0.85)
    m1 = evaluator_mcd(x, y, SR)
    m2 = proxy_mcd(x, y, SR)
    assert abs(m1 - m2) <= 1.0, f"Implementierungen divergieren: {m1:.2f} vs {m2:.2f} dB"
