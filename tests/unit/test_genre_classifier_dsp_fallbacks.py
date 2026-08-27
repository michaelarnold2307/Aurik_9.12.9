"""tests/unit/test_genre_classifier_dsp_fallbacks.py

Tests für die §Spec-24-DSP-Ersatzpfade in backend/core/genre_classifier.py:
numba-defekte librosa-Aufrufe (get_call_template-AttributeError im ROCm-Venv)
müssen echte Messwerte liefern statt Konstanten (2.0 Onsets/s, „Unbekannt“).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from backend.core.genre_classifier import GermanSchlagerClassifier, _estimate_key_dsp, _onset_rate_dsp

SR = 48_000


def _clicks(n_bursts: int = 4, sr: int = SR) -> np.ndarray:
    """1 s Audio mit n_bursts rhythmischen Bursts (4 × 50 ms)."""
    x = np.zeros(sr, dtype=np.float32)
    rng = np.random.default_rng(7)
    for i in range(n_bursts):
        start = int(sr * i / n_bursts)
        x[start : start + sr // 20] = 0.5 * rng.standard_normal(sr // 20).astype(np.float32)
    return x.astype(np.float32)


def _sine(freq: float = 440.0, n: int = SR, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ─── DSP-Helfer: echte Messungen ────────────────────────────────────────────


def test_onset_rate_dsp_detects_rhythmic_bursts() -> None:
    rate = _onset_rate_dsp(_clicks(n_bursts=4), SR)
    assert 2.0 < rate < 8.0, f"Erwartet ~4 Onsets/s, erhalten {rate}"


def test_onset_rate_dsp_steady_sine_is_low() -> None:
    rate = _onset_rate_dsp(_sine(), SR)
    assert rate < 2.0


def test_estimate_key_dsp_a440_is_a() -> None:
    assert _estimate_key_dsp(_sine(440.0, n=SR), SR) == "A-Dur"


def test_estimate_key_dsp_short_signal_unknown() -> None:
    assert _estimate_key_dsp(_sine(440.0, n=2048), SR) == "Unbekannt"


# ─── Methoden-Fallbacks bei numba-Defekt ────────────────────────────────────


def test_onset_rate_numba_defect_uses_dsp_path() -> None:
    clicks = _clicks()
    with patch(
        "librosa.onset.onset_detect",
        side_effect=AttributeError("'function' object has no attribute 'get_call_template'"),
    ):
        rate = GermanSchlagerClassifier._onset_rate(clicks, SR)
    # DSP-Pfad liefert echten Messwert (nahe _onset_rate_dsp) statt Konstante 2.0.
    assert rate == pytest.approx(_onset_rate_dsp(clicks, SR), abs=1e-6)


def test_estimate_key_numba_defect_uses_dsp_path() -> None:
    sine = _sine(440.0, n=SR)
    clf = GermanSchlagerClassifier.__new__(GermanSchlagerClassifier)
    with patch(
        "librosa.feature.chroma_stft",
        side_effect=AttributeError("'function' object has no attribute 'get_call_template'"),
    ):
        key = clf._estimate_key(sine, SR)
    assert key == "A-Dur"
