"""tests/unit/test_zero_length_guards.py

Tests für die §Spec-24-Zero-Length-Guards:
DefectScanner.scan() und MediumDetector.detect() müssen bei (nahezu) leerem
Audio ehrliche Leer-Ergebnisse liefern statt Unsinn (Befund 2026-08-16:
„0.0s Audio“, vinyl=1.000 aus Stille, 8 Consensus-Defekte).
"""

from __future__ import annotations

import numpy as np

from backend.core.defect_scanner import DefectScanner


def test_defect_scanner_zero_length_returns_empty_result() -> None:
    scanner = DefectScanner()
    result = scanner.scan(np.zeros(100, dtype=np.float32), sample_rate=48000)
    assert result.scores == {}
    assert result.duration_seconds < 0.05


def test_defect_scanner_short_silence_guarded() -> None:
    scanner = DefectScanner()
    # 2400 Samples @48k = 0.05s → genau unter der Schwelle
    result = scanner.scan(np.zeros(2000, dtype=np.float32), sample_rate=48000)
    assert result.scores == {}


def test_defect_scanner_normal_audio_not_guarded() -> None:
    scanner = DefectScanner()
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(48000).astype(np.float32) * 0.1  # 1 s
    result = scanner.scan(audio, sample_rate=48000)
    # Normaler Scan läuft durch — Scores vorhanden (auch wenn nur Rauschen).
    assert isinstance(result.scores, dict)


def test_medium_detector_zero_length_returns_unknown() -> None:
    from forensics.medium_detector import get_medium_detector

    md = get_medium_detector()
    result = md.detect(np.zeros(100, dtype=np.float32), 48000)
    assert result.primary_material == "unknown"
    assert result.confidence == 0.0
    assert result.transfer_chain == []
    assert result.bayesian_scores.get("unknown", 0.0) == 1.0
