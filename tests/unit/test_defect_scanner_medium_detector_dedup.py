from __future__ import annotations

import pytest

"""Regression tests for medium detector deduplication in DefectScanner.scan()."""


from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np

from backend.core.defect_scanner import DefectScanner, MaterialType


@dataclass
class _StubForensicMedium:
    transfer_chain: list[str] = field(default_factory=lambda: ["vinyl", "mp3_low"])
    primary_material: str = "vinyl"
    confidence: float = 0.81


def _silence(sr: int = 48_000, duration_s: float = 0.1) -> np.ndarray:
    n = int(sr * duration_s)
    return np.zeros(n, dtype=np.float32)


@pytest.mark.unit
def test_scan_uses_cached_forensic_medium_without_second_detect_call() -> None:
    """If forensic_medium_result is provided, no second MediumDetector call is allowed."""
    scanner = DefectScanner()
    cached_medium = _StubForensicMedium()

    with patch(
        "backend.core.forensics.medium_detector.MediumDetector",
        side_effect=AssertionError("MediumDetector must not be instantiated when cached forensic result is provided"),
    ):
        result = scanner.scan(
            _silence(),
            48_000,
            material_type=MaterialType.VINYL,
            file_ext=".mp3",
            forensic_medium_result=cached_medium,
        )

    assert result.transfer_chain_raw is cached_medium
    assert "vinyl" in str(result.transfer_chain_str)


def test_scan_publishes_chain_override_metadata_for_cacheable_forensic_medium() -> None:
    """Chain-adaptive threshold hints must be exposed on the scan result metadata."""
    scanner = DefectScanner()

    @dataclass
    class _ShellacTapeForensicMedium:
        transfer_chain: list[str] = field(default_factory=lambda: ["shellac", "tape"])
        primary_material: str = "shellac"
        confidence: float = 0.93

    cached_medium = _ShellacTapeForensicMedium()

    with patch(
        "backend.core.forensics.medium_detector.MediumDetector",
        side_effect=AssertionError("MediumDetector must not be instantiated when cached forensic result is provided"),
    ):
        result = scanner.scan(
            _silence(),
            48_000,
            material_type=MaterialType.SHELLAC,
            file_ext=".wav",
            forensic_medium_result=cached_medium,
        )

    assert result.transfer_chain_raw is cached_medium
    assert result.metadata["chain_stage_materials"] == ["tape"]
    assert result.metadata["material_confidence"] == 0.93
    assert result.metadata["chain_threshold_override_applied"] is True
    assert result.metadata["chain_threshold_override_count"] >= 1  # type: ignore[operator]
    assert result.metadata["chain_threshold_overrides"]


def test_scan_calls_medium_detector_once_when_no_cached_forensic_result() -> None:
    """Without cached forensic medium, exactly one MediumDetector.detect() call is expected."""
    scanner = DefectScanner()
    calls = {"init": 0, "detect": 0}

    class _FakeMediumDetector:
        def __init__(self) -> None:
            calls["init"] += 1

        def detect(self, audio: np.ndarray, sr: int, file_ext: str | None = None):
            calls["detect"] += 1
            return _StubForensicMedium(transfer_chain=["tape", "mp3_low"], primary_material="tape", confidence=0.77)

    with patch("backend.core.forensics.medium_detector.MediumDetector", _FakeMediumDetector):
        result = scanner.scan(
            _silence(),
            48_000,
            material_type=MaterialType.TAPE,
            file_ext=".mp3",
            forensic_medium_result=None,
        )

    assert calls == {"init": 1, "detect": 1}
    assert "tape" in str(result.transfer_chain_str)


# ============================================================
# §2.46f Forensic-Veto (2026-09-06)
# ============================================================
# Produktionsbefund: „§2.46f Auto-detected cassette with confidence 10.42“
# bei Physical-Gate-Vinyl (rotation=0.411). Die §v10.14-Baseline-Boni
# (vinyl+6/tape+4/cassette+5) waren Priors als Score getarnt und ließen die
# Feature-Heuristik den Forensic-MediumDetector überstimmen. Nach dem Fix:
#  1. Keine Baseline-Boni → schwache Evidenz liefert UNKNOWN statt Raterei.
#  2. Bekannter Forensic-Primary wird bei schwacher Evidenz übernommen
#     (Defer) und bei Widerspruch erzwungen (Veto) — MediumDetector ist
#     autoritativ, DefectScanner supplementär (pre_analysis §v10.14).
#  3. Log meldet „score“ statt „confidence“ (kein [0,1]-Maß).


def _silent_stereo(seconds: float = 1.5) -> np.ndarray:
    """Stille → alle Feature-Scores ~0 → Heuristik bleibt unter der 0.5-Schwelle."""
    n = int(48_000 * seconds)
    return np.zeros((2, n), dtype=np.float32)


def _hiss_stereo(seconds: float = 1.5) -> np.ndarray:
    """Weißes Rauschen = breitbandiges HF-Rauschen → Hiss-Score hoch.

    Das ist legitime „tape/cassette-like“-Evidenz der Feature-Heuristik —
    der Widerspruch zum Forensic-Primary muss via Veto aufgelöst werden.
    """
    rng = np.random.default_rng(42)
    n = int(48_000 * seconds)
    mono = rng.normal(0.0, 0.02, n).astype(np.float32)
    return np.stack([mono, mono.copy()], axis=0)


@pytest.mark.unit
class TestForensicVeto:
    def test_degenerate_signal_vetoes_to_forensic_primary(self) -> None:
        """§2.46f-1: Auch degenerierte Signale (Stille → hf_loss≈1 → mp3_low-Heuristik)
        dürfen den Forensic-Primary nicht überstimmen — Veto erzwingt Forensic."""
        scanner = DefectScanner(sample_rate=48_000)
        got = scanner._auto_detect_material(
            _silent_stereo(), era_decade=1970, forensic_material=MaterialType.VINYL
        )
        assert got == MaterialType.VINYL

    def test_hiss_contradiction_vetoes_to_forensic_primary(self) -> None:
        """§2.46f-2: Hiss-Evidenz (cassette-like) darf den Forensic-Primary
        NICHT überstimmen — Veto erzwingt das Forensic-Material."""
        scanner = DefectScanner(sample_rate=48_000)
        got = scanner._auto_detect_material(
            _hiss_stereo(), era_decade=1970, forensic_material=MaterialType.VINYL
        )
        assert got == MaterialType.VINYL

    def test_mono_path_defers_to_forensic_primary(self) -> None:
        """§2.46f-3: Defer gilt auch im Mono-Pfad."""
        scanner = DefectScanner(sample_rate=48_000)
        mono = _silent_stereo()[0]
        got = scanner._auto_detect_material(mono, era_decade=1985, forensic_material=MaterialType.TAPE)
        assert got == MaterialType.TAPE

    def test_scan_forensic_primary_drives_auto_material(self) -> None:
        """§2.46f-4: scan() ohne Caller-Material übernimmt den Forensic-Primary
        als Scan-Material statt eines geratenen oder UNKNOWN-Materials."""
        scanner = DefectScanner()
        cached = _StubForensicMedium()
        with patch(
            "backend.core.forensics.medium_detector.MediumDetector",
            side_effect=AssertionError(
                "MediumDetector must not be instantiated when cached forensic result is provided"
            ),
        ):
            result = scanner.scan(
                _hiss_stereo(0.3),
                48_000,
                material_type=None,
                file_ext=".wav",
                forensic_medium_result=cached,
            )
        # _auto_material folgt dem Forensic-Primary (vinyl) → kein Widerspruch.
        assert result.material_type == MaterialType.VINYL
        assert result.auto_detected_material is None
