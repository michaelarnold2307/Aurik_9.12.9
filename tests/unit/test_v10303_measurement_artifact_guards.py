"""Tests für die §v10.303 Messketten-Artefakt-Guards (P0/P1).

P0: measure_all darf degenerierte Signale (2-Sample-Kollaps) nicht als
    0.000-Score-Kaskade messen, sondern neutral (0.5) + CRITICAL-Warnung melden.
P1: MQA-Verdict muss bei degeneriertem Messsignal „MESSARTEFAKT-VERDACHT"
    statt „QUALITY GATES FAILED" lauten (Hörordnung §7 Konfliktregel).
"""

import logging

import numpy as np
import pytest

from backend.core.musical_goals.musical_goals_metrics import get_checker
from backend.core.musical_quality_assurance import (
    MediumType,
    MusicalQualityAssurance,
    ProcessingMode,
)


@pytest.mark.unit
def test_measure_all_degenerate_signal_returns_neutral_scores() -> None:
    sr = 48000
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((sr, 2)).astype(np.float32)
    collapsed = np.zeros((2, 2), dtype=np.float32)  # 2-Sample-Kollaps
    res = get_checker().measure_all(collapsed, sr, reference=ref)
    assert set(res.values()) == {0.5}, f"neutrale Scores erwartet: {set(res.values())}"


@pytest.mark.unit
def test_measure_all_intact_stereo_not_flagged() -> None:
    """Intaktes Stereo (samples-first UND channels-first) darf nicht gekippt werden."""
    sr = 48000
    rng = np.random.default_rng(2)
    audio = rng.standard_normal((sr // 4, 2)).astype(np.float32)
    checker = get_checker()
    res_sf = checker.measure_all(audio, sr, reference=audio)
    res_cf = checker.measure_all(audio.T, sr, reference=audio.T)
    assert set(res_sf.values()) != {0.5}
    assert set(res_cf.values()) != {0.5}


@pytest.mark.unit
def test_mqa_measurement_artefact_verdict() -> None:
    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    orig = np.repeat(tone[:, None], 2, axis=1)
    collapsed = np.zeros((2, 2), dtype=np.float32)
    mods = [f"phase_{i:02d}_x" for i in range(30)]
    rep = MusicalQualityAssurance().validate_final_quality(
        orig,
        collapsed,
        sr,
        MediumType.CD,
        ProcessingMode.RESTORATION,
        mods,
        mushra_score=40.7,
        hpi_score=0.1,
    )
    assert "MESSARTEFAKT-VERDACHT" in rep.verdict, rep.verdict


@pytest.mark.unit
def test_mqa_intact_signal_keeps_failed_verdict() -> None:
    """Gegenprobe: intaktes Signal + schlechte MUSHRA → normales FAILED."""
    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    orig = np.repeat(tone[:, None], 2, axis=1)
    mods = [f"phase_{i:02d}_x" for i in range(30)]
    rep = MusicalQualityAssurance().validate_final_quality(
        orig,
        orig.copy(),
        sr,
        MediumType.CD,
        ProcessingMode.RESTORATION,
        mods,
        mushra_score=40.7,
        hpi_score=0.1,
    )
    assert "MESSARTEFAKT-VERDACHT" not in rep.verdict, rep.verdict
