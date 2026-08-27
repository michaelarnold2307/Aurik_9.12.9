"""B6-Enforcement-Tests: Banquet nur für Vinyl(-Ketten) — Laufzeit-Verifikation.

Spec v10.900 B6: „Banquet auf nicht-Vinyl: −1,3 dB auf Digital“ ⇒
Material-Gate + Opt-In sind Pflicht. Diese Tests beweisen das Laufzeit-
Verhalten von phase_09 (nicht nur die Header-Deklaration).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import backend.core.phases.phase_09_crackle_removal as p09


@pytest.fixture
def _phase(monkeypatch):
    monkeypatch.setattr(p09, "QUALITY_MODE_AVAILABLE", True)
    monkeypatch.setattr(p09, "is_phase_ml_enabled", lambda _p: True)
    calls: list[str] = []

    def _recorder(self, audio, sr, params):
        calls.append("banquet_onnx")
        return audio

    monkeypatch.setattr(p09.CrackleRemovalPhase, "_remove_crackle_onnx_direct", _recorder)
    monkeypatch.setattr(p09.CrackleRemovalPhase, "_measure_crackle_reduction", lambda self, a, b: 0.0)
    monkeypatch.setattr(
        p09.CrackleRemovalPhase,
        "_apply_region_selective_strength_blend",
        lambda self, **kw: kw.get("dry_audio", kw.get("wet_audio")),
    )
    return p09.CrackleRemovalPhase(), calls


def _audio() -> np.ndarray:
    return (0.1 * np.random.RandomState(3).randn(24000)).astype(np.float32)


def test_digital_material_never_runs_banquet(_phase) -> None:
    phase, calls = _phase
    phase.process(_audio(), material_type="cd_digital")
    assert calls == [], "B6 verletzt: Banquet lief auf Digital-Material!"


def test_vinyl_material_runs_banquet(_phase) -> None:
    phase, calls = _phase
    phase.process(_audio(), material_type="vinyl")
    assert calls == ["banquet_onnx"], "Vinyl-Material hat den ML-Pfad nicht erreicht"


def test_chain_aware_vinyl_runs_banquet(_phase) -> None:
    """§Chain-Aware: Vinyl irgendwo in der Kette aktiviert Banquet (z. B. Vinyl→Cassette→MP3)."""
    phase, calls = _phase
    phase.process(_audio(), material_type="cd_digital", transfer_chain=["vinyl", "cassette", "mp3_low"])
    assert calls == ["banquet_onnx"]


# ── B11: HF-Rauschfloor-Check (§v10.900) ──────────────────────────────────────


def _hf_noise(sr: int, n: int, seed: int, rms: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.RandomState(seed)
    sos = butter(4, 8000.0, btype="high", fs=sr, output="sos")
    noise = sosfiltfilt(sos, rng.randn(n).astype(np.float32)).astype(np.float32)
    noise *= rms / (np.sqrt(np.mean(noise**2)) + 1e-12)
    return noise


def test_hf_noise_floor_deterministic_and_sensitive() -> None:
    sr = 48000
    t = np.arange(sr) / sr
    dry = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    assert p09._hf_noise_floor_db(dry, sr) == p09._hf_noise_floor_db(dry, sr)
    wet = dry + _hf_noise(sr, sr, seed=1, rms=1e-3)
    assert p09._hf_noise_floor_db(wet, sr) > p09._hf_noise_floor_db(dry, sr)


def test_b11_guard_contract_over_noise_levels() -> None:
    """Vertrag: Rollback genau dann, wenn delta > Rollback-Schwelle — bei allen Pegeln."""
    sr = 48000
    t = np.arange(sr) / sr
    rng = np.random.RandomState(0)
    # Dry mit vorhandenem HF-Floor (breitbandig), sonst ist jede HF-Zugabe realer Schaden
    dry = (0.3 * np.sin(2 * np.pi * 1000 * t) + 2e-3 * rng.randn(sr)).astype(np.float32)
    outcomes = set()
    for seed, rms in [(1, 1e-5), (2, 1e-4), (3, 3e-3), (4, 1e-1)]:
        wet = dry + _hf_noise(sr, sr, seed, rms)
        out, delta, rolled_back = p09._apply_b11_hf_guard(dry, wet, sr)
        assert rolled_back == (delta > p09._B11_HF_FLOOR_ROLLBACK_DB)
        if rolled_back:
            np.testing.assert_array_equal(out, dry)
        else:
            np.testing.assert_array_equal(out, wet)
        outcomes.add(rolled_back)
    assert outcomes == {True, False}, "Pegelbereich deckt beide Guard-Zweige ab"


def test_spec_decrackle_row_references_verified_model() -> None:
    """Rev. 2026-08-16: Keine Phantom-Zitate im Decrackle-Pfad (RBME-Net existiert nicht)."""
    spec = Path(__file__).resolve().parents[2] / ".github" / "specs" / "04_dsp_standards.md"
    row = next(
        l for l in spec.read_text(encoding="utf-8").splitlines() if l.strip().startswith("| Decrackle ")
    )
    assert "RBME-Net" not in row, "Phantom-Zitat RBME-Net wieder in der Spec!"
    assert "Banquet-Vinyl" in row
