from __future__ import annotations

"""Regressionstests für reflective_listening_pass.py — Layout-Sicherheit (§v10.x).

Befund 2026-08-22: Die High-/Low-Shelf-Zweige filterten ``arr[:, ch]`` — das
setzt Samples-first-Layout voraus. Bei Channels-first (2, N) ist ``arr[:, ch]``
eine Spalte der Länge 2 → scipy.sosfiltfilt crasht mit
„length of input vector x must be greater than padlen, 9“. Der
Kurzsignal-Guard (≤16 Samples) schützt dagegen nicht, weil er die Zeitachse
korrekt erkennt — die Filterzweige taten es nicht.

Fix: ``_filter_time_axis`` wählt die Zeitachse explizit:
(N, C) → Achse 0, (C, N) → Achse 1, mono → Achse 0.
"""

import numpy as np

from backend.core.reflective_listening_pass import ReflectiveListeningPass, RLPIssue

SR = 48000


def _shelf_issue() -> RLPIssue:
    return RLPIssue(
        category="spectral_tilt",
        severity=0.6,
        detail="Test: High-Shelf +2 dB @ 8 kHz",
        correction={"eq_high_shelf_db": 2.0, "eq_freq_hz": 8000.0},
    )


def _active_stereo(secs: float = 1.0) -> np.ndarray:
    """Stereo-Signal mit aktivem Inhalt (Samples-first, (N, 2))."""
    n = int(SR * secs)
    t = np.arange(n) / SR
    left = 0.4 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * np.sin(2 * np.pi * 6000.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 554.0 * t) + 0.1 * np.sin(2 * np.pi * 9000.0 * t)
    return np.stack([left, right], axis=1).astype(np.float64)


class TestRLPLayoutSafety:
    def test_01_channels_first_high_shelf_no_crash(self):
        """(2, N) mit High-Shelf → kein padlen-Crash, finite Ausgabe."""
        rlp = ReflectiveListeningPass()
        stereo_sf = _active_stereo()
        stereo_cf = stereo_sf.T  # (2, N)
        out = rlp._apply_corrections(stereo_cf, SR, [_shelf_issue()], "vinyl")
        assert out.shape == stereo_cf.shape
        assert np.all(np.isfinite(out))

    def test_02_layout_equivalence_transposed(self):
        """(2, N) und (N, 2) liefern transponiert identische Ergebnisse."""
        rlp = ReflectiveListeningPass()
        stereo_sf = _active_stereo()
        out_sf = rlp._apply_corrections(stereo_sf, SR, [_shelf_issue()], "vinyl")
        out_cf = rlp._apply_corrections(stereo_sf.T, SR, [_shelf_issue()], "vinyl")
        assert np.allclose(out_cf, out_sf.T, atol=1e-9)

    def test_03_mono_equals_first_channel_of_layouts(self):
        """Mono-Ergebnis == erste Spalte des (N, 2)-Ergebnisses."""
        rlp = ReflectiveListeningPass()
        stereo_sf = _active_stereo()
        mono = stereo_sf[:, 0].copy()
        out_mono = rlp._apply_corrections(mono, SR, [_shelf_issue()], "vinyl")
        out_sf = rlp._apply_corrections(stereo_sf, SR, [_shelf_issue()], "vinyl")
        assert np.allclose(out_mono, out_sf[:, 0], atol=1e-9)

    def test_04_low_shelf_channels_first_no_crash(self):
        """(2, N) mit Low-Shelf → kein Crash, finite Ausgabe."""
        rlp = ReflectiveListeningPass()
        issue = RLPIssue(
            category="bass_loss",
            severity=0.5,
            detail="Test: Low-Shelf +1.5 dB @ 150 Hz",
            correction={"eq_low_shelf_db": 1.5, "eq_freq_hz": 150.0},
        )
        out = rlp._apply_corrections(_active_stereo().T, SR, [issue], "vinyl")
        assert np.all(np.isfinite(out))

    def test_05_short_signal_guard_unchanged(self):
        """Zeitachse ≤ 16 Samples → Identität (auch Channels-first)."""
        rlp = ReflectiveListeningPass()
        short = np.random.default_rng(0).standard_normal((2, 16))
        out = rlp._apply_corrections(short, SR, [_shelf_issue()], "vinyl")
        assert np.array_equal(out, short)

    def test_06_filter_time_axis_direct(self):
        """_filter_time_axis: (N,2) Achse 1 und (2,N) Achse 0 liefern gleiche Werte."""
        rlp = ReflectiveListeningPass()
        sos = rlp._make_high_shelf(SR, 8000.0, 2.0)
        stereo_sf = _active_stereo()
        a = rlp._filter_time_axis(sos, stereo_sf)
        b = rlp._filter_time_axis(sos, stereo_sf.T)
        assert np.allclose(a.T, b, atol=1e-9)
