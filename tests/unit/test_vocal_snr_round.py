"""Tests für scripts/prepare_vocal_snr_round.py — deterministische Task-Erzeugung."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import prepare_vocal_snr_round as vsr


def test_make_task_audio_snr_and_determinism() -> None:
    rng = np.random.RandomState(3)
    audio = (0.2 * rng.randn(48000)).astype(np.float32)
    task1, snr1 = vsr.make_task_audio(audio, 48000, snr_db=5.0, seed=11)
    task2, snr2 = vsr.make_task_audio(audio, 48000, snr_db=5.0, seed=11)
    np.testing.assert_array_equal(task1, task2)  # deterministisch (§G5 (GEBOTE.md))
    assert abs(snr1 - 5.0) < 0.2  # Ziel-SNR ungefähr getroffen
    assert snr2 == snr1
    assert np.all(np.isfinite(task1))
    assert np.max(np.abs(task1)) <= 1.0 + 1e-6
    # Task unterscheidet sich hörbar vom Original
    assert np.mean(np.abs(task1 - audio)) > 0.02


def test_make_task_audio_resamples_to_48k() -> None:
    rng = np.random.RandomState(5)
    audio = (0.1 * rng.randn(44100)).astype(np.float32)
    task, _ = vsr.make_task_audio(audio, 44100, snr_db=5.0, seed=12)
    assert len(task) == 48000
