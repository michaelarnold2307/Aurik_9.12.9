"""Tests für die H-Serie des Anti-Pattern-Scanners (Hörordnungs-/Exportqualität)."""

from __future__ import annotations

import sys
from pathlib import Path

# Scanner liegt unter .agents/skills/bug-prevention/ — für Import verfügbar machen
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".agents" / "skills" / "bug-prevention"))

from scan_anti_patterns import check_hoerordnung_export_patterns as _check


def _run(source: str, path: str = "backend/core/phases/test_phase.py") -> list[str]:
    return _check(path, source)


def test_h01_naked_int16_flagged_dithered_not() -> None:
    bad = "audio = signal.astype(np.int16)\n"
    assert any("H01" in m for m in _run(bad))
    good = "audio = dither_powr3(signal).astype(np.int16)\n"
    assert not any("H01" in m for m in _run(good))


def test_h02_griffinlim_flagged() -> None:
    assert any("H02" in m for m in _run("out = griffinlim(mag)\n"))
    assert not any("H02" in m for m in _run("out = pghi_reconstruct(mag, sr)\n"))


def test_h03_sosfilt_in_phase_flagged_analysis_exempt() -> None:
    # Phasen-Pfad → Meldung
    assert any("H03" in m for m in _run("y = sosfilt(sos, x)\n", "backend/core/phases/p.py"))
    # Analyse-Datei (kein Phasen/DSP-Pfad) → keine Meldung
    assert not any("H03" in m for m in _run("y = sosfilt(sos, x)\n", "backend/core/metrics.py"))
    # sosfiltfilt selbst ist kein Treffer
    assert not any("H03" in m for m in _run("y = sosfiltfilt(sos, x)\n", "backend/core/phases/p.py"))


def test_h04_time_in_decision_flagged_profiling_not() -> None:
    assert any("H04" in m for m in _run("if time.time() > deadline:\n    break\n"))
    assert not any("H04" in m for m in _run("t0 = time.time()\n"))


def test_h05_resample_without_guard_flagged() -> None:
    bad = "from scipy import signal\ny = signal.resample(x, 48000)\n"
    assert any("H05" in m for m in _run(bad, "backend/core/dsp/p.py"))
    guarded = "if abs(len(x) - target) / target > 0.001:\n    pass\ny = signal.resample(x, target)\n"
    assert not any("H05" in m for m in _run(guarded, "backend/core/dsp/p.py"))


def test_h06_hard_clamp_flagged_soft_knee_not() -> None:
    bad = "out = np.clip(audio, -1.0, 1.0)\n"
    assert any("H06" in m for m in _run(bad, "backend/core/phases/p.py"))
    soft = "out = soft_knee_limit(audio)\n"
    assert not any("H06" in m for m in _run(soft, "backend/core/phases/p.py"))


def test_h07_silent_except_flagged_logged_not() -> None:
    bad = "try:\n    x = ml()\nexcept Exception:\n    return 0.5\n"
    assert any("H07" in m for m in _run(bad))
    good = "try:\n    x = ml()\nexcept Exception:\n    logger.warning('fallback')\n    return 0.5\n"
    assert not any("H07" in m for m in _run(good))
