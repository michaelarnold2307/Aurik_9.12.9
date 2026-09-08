"""§AUTH-P12 — Vibrato-Erhaltung der Flutter-Korrektur (phase_12).

Befund 2026-09-08: §v10.709 Quality-Degradation #1 nach
phase_12_wow_flutter_fix: ['authentizitaet']. Root-Cause: Die
Flutter-Korrektur (4–100-Hz-Band) flacht auch Vibrato/Intonations-Bends
der Performance ab. _preserve_musical_modulation() nimmt die Korrektur
proportional Richtung Identität zurück, wo musikalische Modulation die
Flutter-Korrektur dominiert — mechanischer Wow/Flutter bleibt voll
korrigiert. Deterministische Synthetik-Fälle ohne DSP.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix


def _make_frames(
    n_frames: int = 400,
    sr: int = 48000,
    vibrato_hz: float = 0.0,
    vibrato_cents: float = 0.0,
    wow_hz: float = 0.0,
    wow_cents: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetische Pitch-Trajektorie (440 Hz Grundton + Modulation)."""
    t = np.arange(n_frames) / (sr / (100 * sr / 1000 // 4))  # 40 fps @ 48 kHz
    ratio = 1.0
    if vibrato_cents > 0.0:
        ratio *= 2.0 ** ((vibrato_cents / 1200.0) * np.sin(2 * np.pi * vibrato_hz * t))
    if wow_cents > 0.0:
        ratio *= 2.0 ** ((wow_cents / 1200.0) * np.sin(2 * np.pi * wow_hz * t))
    pitch = 440.0 * ratio
    confidence = np.ones(n_frames, dtype=np.float64)
    return pitch, confidence


def _flutter_from_pitch(pitch: np.ndarray, strength: float = 0.7) -> np.ndarray:
    target = float(np.median(pitch))
    return 1.0 + strength * (pitch / target - 1.0)


@pytest.fixture()
def restorer() -> WowFlutterFix:
    return WowFlutterFix.__new__(WowFlutterFix)  # keine schwere Init nötig


def test_vibrato_dominant_gets_preserved(restorer: WowFlutterFix) -> None:
    """Vibrato (5 Hz, 30 Cents) dominiert → Korrektur wird stark zurückgenommen."""
    pitch, conf = _make_frames(vibrato_hz=5.0, vibrato_cents=30.0)
    sf = _flutter_from_pitch(pitch)
    out = restorer._preserve_musical_modulation(sf, pitch, conf, 48000)

    rms_in = float(np.sqrt(np.mean((sf - 1.0) ** 2)))
    rms_out = float(np.sqrt(np.mean((out - 1.0) ** 2)))
    # ≥60 % der Korrektur zurückgenommen (Envelope-Glättung dämpft das
    # theoretische 85-%-Limit; gemessen: ~69 %)
    assert rms_out < 0.40 * rms_in
    assert np.isfinite(out).all()
    assert float(np.min(out)) >= 0.9 and float(np.max(out)) <= 1.1


def test_wow_only_stays_corrected(restorer: WowFlutterFix) -> None:
    """Langsames Wow (0.5 Hz) ist mechanisch → Korrektur bleibt unverändert."""
    pitch, conf = _make_frames(wow_hz=0.5, wow_cents=40.0)
    sf = _flutter_from_pitch(pitch)
    out = restorer._preserve_musical_modulation(sf, pitch, conf, 48000)

    assert float(np.max(np.abs(out - sf))) < 1e-3


def test_identity_passthrough(restorer: WowFlutterFix) -> None:
    pitch, conf = _make_frames(vibrato_hz=5.0, vibrato_cents=20.0)
    sf = np.ones_like(pitch)
    out = restorer._preserve_musical_modulation(sf, pitch, conf, 48000)
    assert np.array_equal(out, sf)


def test_too_short_passthrough(restorer: WowFlutterFix) -> None:
    pitch = np.array([440.0, 441.0, 439.0], dtype=np.float64)
    conf = np.ones(3)
    sf = np.array([1.0, 1.01, 0.99])
    out = restorer._preserve_musical_modulation(sf, pitch, conf, 48000)
    assert np.array_equal(out, sf)


def test_nan_inf_guard(restorer: WowFlutterFix) -> None:
    """§0a: NaN/Inf in der Trajektorie darf nicht crashen."""
    pitch, conf = _make_frames(vibrato_hz=5.0, vibrato_cents=30.0)
    pitch = pitch.copy()
    pitch[10] = np.nan
    pitch[11] = np.inf
    sf = _flutter_from_pitch(np.nan_to_num(pitch, nan=440.0, posinf=440.0))
    out = restorer._preserve_musical_modulation(sf, pitch, conf, 48000)
    assert np.isfinite(out).all()
    assert float(np.min(out)) >= 0.9 and float(np.max(out)) <= 1.1


def test_no_valid_frames_passthrough(restorer: WowFlutterFix) -> None:
    """Alle Frames ungültig → Passthrough ohne Fehler."""
    pitch = np.zeros(200, dtype=np.float64)
    conf = np.zeros(200)
    sf = np.full(200, 1.05)
    out = restorer._preserve_musical_modulation(sf, pitch, conf, 48000)
    assert np.array_equal(out, sf)
