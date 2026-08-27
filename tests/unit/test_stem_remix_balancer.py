"""StemRemixBalancer — Golden-Tests (§1.4/§2.8, Spec 02).

Invarianten:
    1. Summe: balance_remix(v, i, v+i, sr, 1.0) ≈ v+i (moderate Pegel)
    2. LUFS-Ziel = Quell-LUFS: Ausgabe-LUFS ≈ Referenz-LUFS (±1.5 LU)
    3. Gain-Cap: ±6 dB (kein Rausch-Boost bei leisen Mixen)
    4. Soft-Knee: kein Hard-Clip-Plateau; Peak ≤ 1.0
    5. NaN/Kollaps/Stille → Referenz unverändert
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.stem_remix_balancer import StemRemixBalancer, get_stem_remix_balancer


def _sine(freq: float, sr: int, n: int) -> np.ndarray:
    t = np.arange(n) / sr
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _lufs(arr: np.ndarray, sr: int) -> float:
    from backend.core.export_quality_gate import ExportQualityGate

    return float(ExportQualityGate._measure_lufs(arr, sr))


SR = 48000


def test_singleton() -> None:
    assert get_stem_remix_balancer() is get_stem_remix_balancer()


def test_sum_invariant_at_weight_one() -> None:
    bal = StemRemixBalancer()
    voc = _sine(440.0, SR, SR * 2)
    ins = _sine(220.0, SR, SR * 2)
    ref = voc + ins
    out = bal.balance_remix(voc, ins, ref, SR, vocal_weight=1.0)
    # Referenz-LUFS == Mix-LUFS → gain ≈ 0 → Summe bleibt erhalten (Toleranz 0.5 dB RMS)
    rms_ref = np.sqrt(np.mean(ref.astype(np.float64) ** 2))
    rms_out = np.sqrt(np.mean(out.astype(np.float64) ** 2))
    assert abs(20 * np.log10(rms_out / rms_ref)) < 0.5


def test_lufs_target_is_source() -> None:
    bal = StemRemixBalancer()
    voc = _sine(440.0, SR, SR * 2)
    ins = _sine(220.0, SR, SR * 2)
    ref = (0.6 * (voc + ins)).astype(np.float32)  # Referenz deutlich leiser als naive Summe
    out = bal.balance_remix(voc, ins, ref, SR, vocal_weight=1.0)
    assert abs(_lufs(out, SR) - _lufs(ref, SR)) < 1.5


def test_gain_cap_no_noise_boost() -> None:
    bal = StemRemixBalancer()
    voc = _sine(440.0, SR, SR * 2)
    ins = np.zeros_like(voc)
    ref = (0.05 * voc).astype(np.float32)  # sehr leise Referenz → +26 dB nötig
    out = bal.balance_remix(voc, ins, ref, SR, vocal_weight=1.0)
    gain_db = _lufs(out, SR) - _lufs(voc + ins, SR)
    assert gain_db <= 6.0 + 1e-6, f"Gain-Cap verletzt: {gain_db:.2f} dB"


def test_soft_knee_no_hard_clip_plateau() -> None:
    bal = StemRemixBalancer()
    rng = np.random.default_rng(3)
    voc = rng.standard_normal(SR).astype(np.float32) * 1.2
    ins = rng.standard_normal(SR).astype(np.float32) * 1.2
    ref = (voc + ins).astype(np.float32)  # LUFS identisch → kein Gain; Peak heiß
    out = bal.balance_remix(voc, ins, ref, SR, vocal_weight=1.0)
    assert float(np.max(np.abs(out))) <= 1.0
    # Kein hartes Plateau: weniger als 0.5 % der Samples exakt bei ±1.0
    _at_ceiling = np.mean(np.abs(out) >= 0.999)
    assert _at_ceiling < 0.005, f"Hard-Clip-Plateau: {_at_ceiling:.4f}"
    assert np.isfinite(out).all()


def test_nan_mix_falls_back_to_reference() -> None:
    bal = StemRemixBalancer()
    voc = _sine(440.0, SR, SR)
    ins = np.full_like(voc, np.nan)
    ref = voc + _sine(220.0, SR, SR)
    out = bal.balance_remix(voc, ins, ref, SR, 1.0)
    assert np.array_equal(out, ref)


def test_silent_reference_returns_peak_safe_mix() -> None:
    """Stille Referenz: kein LUFS-Zug möglich — Mix mit Peak-Schutz zurück."""
    bal = StemRemixBalancer()
    voc = _sine(440.0, SR, SR)
    ins = _sine(220.0, SR, SR)
    ref = np.zeros_like(voc)
    out = bal.balance_remix(voc, ins, ref, SR, 1.0)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) <= 1.0
    # Mix-Energie bleibt erhalten (kein Absenken auf Stille)
    rms_mix = np.sqrt(np.mean((voc + ins).astype(np.float64) ** 2))
    rms_out = np.sqrt(np.mean(out.astype(np.float64) ** 2))
    assert rms_out > 0.5 * rms_mix


def test_mono_stereo_layout_coercion() -> None:
    bal = StemRemixBalancer()
    voc_m = _sine(440.0, SR, SR)
    ins_m = _sine(220.0, SR, SR)
    ref_s = np.stack([voc_m + ins_m, voc_m + ins_m], axis=1).astype(np.float32)
    out = bal.balance_remix(voc_m, ins_m, ref_s, SR, 1.0)
    assert out.ndim == 2 and out.shape[1] == 2
    assert np.isfinite(out).all()
