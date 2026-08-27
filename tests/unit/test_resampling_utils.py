"""tests/unit/test_resampling_utils.py

Tests für backend/core/resampling_utils.py — numba-Kompatibilitäts-Guard.
Befund 2026-08-16: Im ROCm-Venv ist der numba-Dispatcher von librosa.resample
ein plain function ohne get_call_template → AttributeError. Der Guard muss auf
scipy.signal.resample_poly ausweichen und darf NIE pass-through bei falscher
Samplerate liefern (korrumpiert ML-Embeddings).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from backend.core.resampling_utils import resample_audio, resample_to_48k

SR = 48_000


def _tone(n: int = 8000, sr: int = 32_000) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_passthrough_same_rate() -> None:
    x = _tone()
    out = resample_audio(x, 32_000, 32_000)
    assert out is x or np.array_equal(out, x)


def test_librosa_path_length_ratio() -> None:
    x = _tone(n=8000, sr=32_000)
    out = resample_audio(x, 32_000, 48_000)
    assert out.shape[-1] == int(np.ceil(8000 * 48_000 / 32_000))
    assert np.all(np.isfinite(out))


def test_numba_defect_falls_back_to_scipy() -> None:
    x = _tone(n=8000, sr=32_000)
    with patch(
        "backend.core.resampling_utils.librosa.resample",
        side_effect=AttributeError("'function' object has no attribute 'get_call_template'"),
    ):
        out = resample_audio(x, 32_000, 48_000)
    assert out.shape[-1] == int(np.ceil(8000 * 48_000 / 32_000))
    assert np.all(np.isfinite(out))


def test_non_numba_attribute_error_raises() -> None:
    x = _tone()
    with patch(
        "backend.core.resampling_utils.librosa.resample",
        side_effect=AttributeError("anderer Fehler"),
    ):
        with pytest.raises(AttributeError):
            resample_audio(x, 32_000, 48_000)


def test_resample_to_48k_returns_tuple() -> None:
    x = _tone(n=4000, sr=22_050)
    out, sr = resample_to_48k(x, 22_050)
    assert sr == 48_000
    assert out.shape[-1] == int(np.ceil(4000 * 48_000 / 22_050))
    assert np.all(np.isfinite(out))
