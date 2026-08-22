from __future__ import annotations

"""Regression tests for MusicalQualityAssurance processing intensity handling."""


import numpy as np
import pytest

from backend.core.musical_quality_assurance import (
    IntegrityCheckResult,
    MediumType,
    MusicalQualityAssurance,
    ProcessingMode,
)
from backend.core.quality_prediction import QualityEstimate, QualityLevel


def _quality(overall: float = 80.0) -> QualityEstimate:
    return QualityEstimate(
        overall_score=overall,
        quality_level=QualityLevel.GOOD,
        snr_db=65.0,
        dynamic_range_db=18.0,
        thd_percent=0.3,
        clarity=0.8,
        warmth=0.7,
        brightness=0.6,
        naturalness=0.8,
        authenticity=0.85,
        confidence=0.9,
        bandwidth_hz=(20.0, 18000.0),
        has_artifacts=False,
        artifact_types=[],
    )


def _quality_with_snr(snr_db: float, overall: float = 80.0) -> QualityEstimate:
    q = _quality(overall=overall)
    q.snr_db = float(snr_db)
    # Alle anderen Gates bewusst sicher über Schwellwerten halten,
    # damit nur das SNR-Gate bewertet wird.
    q.clarity = 0.92
    q.warmth = 0.78
    q.brightness = 0.72
    q.naturalness = 0.90
    q.authenticity = 0.88
    return q


def _integrity_ok() -> IntegrityCheckResult:
    return IntegrityCheckResult(
        passed=True,
        overall_integrity=0.95,
        violations=[],
        violation_details={},
        naturalness_change=0.0,
        character_preservation=1.0,
        overprocessing_risk=0.1,
        recommendations=[],
        should_rollback=False,
        should_stop_processing=False,
    )


@pytest.mark.unit
def test_validate_final_quality_uses_unique_modules_for_intensity(monkeypatch):
    """Duplicate module names must not inflate processing intensity."""
    mqa = MusicalQualityAssurance()

    monkeypatch.setattr(mqa.analyzer, "analyze_quality", lambda _audio, _sr, **_kw: _quality())
    monkeypatch.setattr(mqa, "check_quality_gate", lambda *_args, **_kwargs: (True, "ok"))
    monkeypatch.setattr(mqa, "check_musical_integrity", lambda *_args, **_kwargs: _integrity_ok())

    report = mqa.validate_final_quality(
        original_audio=np.zeros(2, dtype=np.float32),
        processed_audio=np.zeros(2, dtype=np.float32),
        sample_rate=48000,
        medium_type=MediumType.VINYL_33,
        processing_mode=ProcessingMode.RESTORATION,
        modules_applied=[
            "denoise",
            "denoise",
            "declip",
            "declip",
            "denoise",
            "declip",
            "denoise",
            "declip",
            "denoise",
            "declip",
        ],
    )

    # §2.54: Divisor ist jetzt 50 (Aurik-9-Pipeline-Max), nicht 8.
    # 2 unique modules / 50 = 0.04 — Deduplizierung ist weiterhin korrekt.
    assert report.processing_intensity == pytest.approx(2 / 50.0, abs=1e-6)
    assert report.overprocessed is False
    assert "OVERPROCESSED" not in report.verdict


def test_quality_gate_snr_borderline_low_baseline_passes_with_tolerance(monkeypatch):
    """Grenzfall um ~0.3 dB bei niedriger Baseline darf nicht als hard fail enden."""
    mqa = MusicalQualityAssurance()
    baseline = _quality_with_snr(28.2)

    # Grenzfall: früher false fail bei 28.2 < 28.5.
    monkeypatch.setattr(mqa.analyzer, "analyze_quality", lambda _audio, _sr: _quality_with_snr(28.2))

    gate_ok, reason = mqa.check_quality_gate(
        np.zeros(48000, dtype=np.float32),
        48000,
        baseline,
        MediumType.VINYL_33,
        ProcessingMode.RESTORATION,
    )

    assert gate_ok is True, f"Grenzfall muss passieren, erhielt: {reason}"


def test_quality_gate_does_not_blame_preexisting_low_warmth_and_naturalness(monkeypatch):
    """No-op auf historischem Proxy-schwachem Material ist kein Aurik-Wärmeschaden."""
    mqa = MusicalQualityAssurance()
    baseline = _quality_with_snr(24.0, overall=62.0)
    baseline.warmth = 0.02
    baseline.naturalness = 0.00
    baseline.authenticity = 0.95
    current = _quality_with_snr(24.1, overall=62.0)
    current.warmth = 0.02
    current.naturalness = 0.00
    current.authenticity = 0.95

    monkeypatch.setattr(mqa.analyzer, "analyze_quality", lambda _audio, _sr: current)

    gate_ok, reason = mqa.check_quality_gate(
        np.zeros(48000, dtype=np.float32),
        48000,
        baseline,
        MediumType.WAX_CYLINDER,
        ProcessingMode.RESTORATION,
    )

    assert gate_ok is True, f"No-op darf keinen Wärme-/Naturalness-Fail erzeugen: {reason}"


def test_quality_gate_snr_clear_drop_still_fails(monkeypatch):
    """Deutlicher SNR-Abfall muss weiterhin korrekt als fail erkannt werden."""
    mqa = MusicalQualityAssurance()
    baseline = _quality_with_snr(28.2)

    monkeypatch.setattr(mqa.analyzer, "analyze_quality", lambda _audio, _sr: _quality_with_snr(26.5))

    gate_ok, reason = mqa.check_quality_gate(
        np.zeros(48000, dtype=np.float32),
        48000,
        baseline,
        MediumType.VINYL_33,
        ProcessingMode.RESTORATION,
    )

    assert gate_ok is False
    assert "SNR too low" in reason


def test_quality_gate_snr_high_baseline_restoration_preserves(monkeypatch):
    """§v10.x Baseline-Relativierung: Bei hoher Baseline (≥38 dB) darf
    Restoration den SNR ERHALTEN — eine erzwungene +3-dB-Steigerung würde
    aggressives Denoising provozieren (§2.54, §0) und bestraft eine bewusst
    konservative Pipeline (Log: SNR 38.9→38.9, Verdikt ❌ trotz MUSHRA ✅)."""
    mqa = MusicalQualityAssurance()
    baseline = _quality_with_snr(50.0)

    # Restoration: Stagnation bei sauberer Quelle ist OK (keine Regression).
    monkeypatch.setattr(mqa.analyzer, "analyze_quality", lambda _audio, _sr: _quality_with_snr(50.0, overall=90.0))

    gate_ok, reason = mqa.check_quality_gate(
        np.zeros(48000, dtype=np.float32),
        48000,
        baseline,
        MediumType.CD,
        ProcessingMode.RESTORATION,
    )

    assert gate_ok is True, f"Restoration-Erhalt muss bestehen: {reason}"


def test_quality_gate_snr_high_baseline_studio_expects_improvement(monkeypatch):
    """§v10.x Studio 2026 bleibt anspruchsvoll: +1 dB SNR-Erwartung bei hoher Baseline."""
    mqa = MusicalQualityAssurance()
    baseline = _quality_with_snr(50.0)

    monkeypatch.setattr(mqa.analyzer, "analyze_quality", lambda _audio, _sr: _quality_with_snr(50.0, overall=90.0))

    gate_ok, reason = mqa.check_quality_gate(
        np.zeros(48000, dtype=np.float32),
        48000,
        baseline,
        MediumType.CD,
        ProcessingMode.STUDIO_2026,
    )

    assert gate_ok is False
    assert "SNR too low" in reason
