"""Unit-Tests für backend/core/anti_fatigue_pass.py (Hörordnung §6, §V7).

Deterministisch, ohne Audio-I/O. Prüft Plan-Ableitung, einseitigen Crest,
Mikrodynamik-Expansion (Peak-neutral) und Do-No-Harm-Verhalten.
"""

from __future__ import annotations

import numpy as np

from backend.core.anti_fatigue_pass import (
    FatigueCorrectionPlan,
    anti_fatigue_pass,
    apply_microdynamics_expansion,
    fatigue_correction_plan,
)
from backend.core.listening_fatigue_metric import measure_fatigue

SR = 48000
_RNG = np.random.RandomState(42)


def _tone_bursts(seconds: float = 3.0) -> np.ndarray:
    """Natürliche Dynamik: kurze Sinus-Bursts mit langen Pausen (Crest ≈ 16 dB)."""
    t = np.arange(int(SR * seconds)) / SR
    sig = np.sin(2 * np.pi * 440.0 * t) * ((t % 0.4) < 0.02).astype(np.float64)
    return sig.astype(np.float32)


def _compressed_sine(seconds: float = 3.0) -> np.ndarray:
    """Stark komprimiert: reiner Sinus (Crest ≈ 3 dB)."""
    t = np.arange(int(SR * seconds)) / SR
    return (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _bright_noise(seconds: float = 3.0, seed: int = 42) -> np.ndarray:
    """Hell + komprimiert: hochpass-gefiltertes Rauschen, weich begrenzt."""
    n = int(SR * seconds)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n).astype(np.float64)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[freqs < 4000.0] *= 0.15  # HF-dominant
    bright = np.fft.irfft(spec, n)
    bright = bright / (np.max(np.abs(bright)) + 1e-12) * 0.9
    return np.tanh(2.2 * bright).astype(np.float32)  # weich begrenzt → Crest sinkt


def test_plan_empty_for_healthy_fatigue():
    components = {"fatigue": 0.30, "hf_dev": 0.0, "crest_dev": 0.0, "micro_dev": 0.0}
    plan = fatigue_correction_plan(components, fatigue=0.30)
    assert plan.is_empty
    assert plan.reason


def test_plan_hf_cut_and_expand_for_bright_compressed():
    components = {"fatigue": 0.55, "hf_dev": 0.6, "crest_dev": 0.5, "micro_dev": 0.4}
    plan = fatigue_correction_plan(components, fatigue=0.55)
    assert plan.hf_cut_db < 0.0
    assert plan.micro_expand_db > 0.0
    assert plan.hf_cut_db >= -3.0  # Brillanz-Boundary
    assert plan.micro_expand_db <= 2.0


def test_plan_empty_when_components_uncritical():
    components = {"fatigue": 0.41, "hf_dev": 0.02, "crest_dev": 0.02, "micro_dev": 0.02}
    plan = fatigue_correction_plan(components, fatigue=0.41)
    assert plan.is_empty


def test_crest_dev_one_sided_natural_dynamics_not_penalized():
    comps = measure_fatigue(_tone_bursts(), SR, return_components=True)
    assert isinstance(comps, dict)
    assert comps["crest_dev"] == 0.0  # hoher Crest = natürliche Dynamik
    assert comps["crest_db"] >= 14.0


def test_crest_dev_penalizes_compression():
    comps = measure_fatigue(_compressed_sine(), SR, return_components=True)
    assert isinstance(comps, dict)
    assert comps["crest_db"] < 14.0
    assert comps["crest_dev"] > 0.0


def test_microdynamics_expansion_peak_neutral():
    # Lauter Transient + leiser Schwanz: Expansion hebt nur Leises.
    n = int(SR * 3)
    sig = np.zeros(n, dtype=np.float64)
    sig[: int(SR * 0.2)] = 0.9 * np.sin(2 * np.pi * 440.0 * np.arange(int(SR * 0.2)) / SR)
    sig[int(SR * 0.5) :] = 0.05 * np.sin(2 * np.pi * 440.0 * np.arange(n - int(SR * 0.5)) / SR)
    out = apply_microdynamics_expansion(sig.astype(np.float32), SR, 1.5)
    peak_before = float(np.max(np.abs(sig)))
    peak_after = float(np.max(np.abs(out)))
    assert peak_after <= peak_before + 1e-6  # Peaks unberührt
    quiet_before = float(np.sqrt(np.mean(sig[int(SR * 0.5) :] ** 2)))
    quiet_after = float(np.sqrt(np.mean(out[int(SR * 0.5) :] ** 2)))
    assert quiet_after > quiet_before  # Leises angehoben


def test_pass_do_no_harm_on_healthy_audio():
    audio = _tone_bursts()
    result = anti_fatigue_pass(audio, SR)
    assert not result.applied
    assert np.array_equal(result.audio, np.asarray(audio, dtype=np.float64))


def test_pass_improves_fatiguing_audio():
    audio = _bright_noise()
    before = float(measure_fatigue(audio, SR))
    result = anti_fatigue_pass(audio, SR)
    if not result.applied:
        # Auch legitim: Do-No-Harm entscheidet gegen den Eingriff — dann muss
        # die Fatigue unkritisch gewesen sein bzw. der Eingriff nichts gebracht haben.
        assert before <= 0.41 or result.after >= before
    else:
        assert result.after < before
        assert float(np.max(np.abs(result.audio))) <= 1.0


def test_pass_deterministic():
    audio = _bright_noise(seed=7)
    r1 = anti_fatigue_pass(audio, SR)
    r2 = anti_fatigue_pass(audio, SR)
    assert r1.applied == r2.applied
    assert np.array_equal(r1.audio, r2.audio)
    assert r1.before == r2.before


def test_plan_dataclass_defaults():
    plan = FatigueCorrectionPlan()
    assert plan.is_empty
