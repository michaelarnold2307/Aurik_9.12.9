"""Tests für scripts/dsp_benchmark.py — synthetische DSP-Ground-Truth + NOLA-Regression."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import dsp_benchmark as db  # noqa: E402


def test_synth_degraded_deterministic() -> None:
    c1, d1 = db.synth_degraded(snr_db=5.0, seed=11)
    c2, d2 = db.synth_degraded(snr_db=5.0, seed=11)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(d1, d2)
    assert np.all(np.isfinite(d1))
    # degradiert ist lauter als clean (Rauschen + Crackles)
    assert np.sqrt(np.mean(d1**2)) > np.sqrt(np.mean(c1**2))


def test_metrics_clean_passthrough() -> None:
    clean, _ = db.synth_degraded(seed=7)
    ref_snr, lsd, edge = db.metrics(clean, clean, 48000)
    assert ref_snr > 60.0  # identisches Signal → Fehler ~0
    assert lsd < 1.0
    assert 0.5 < edge < 1.5


def test_reference_methods_have_bounded_edges() -> None:
    """Referenz-Implementierungen (korrektes boundary-Paar) erzeugen keine Kantenspikes."""
    clean, deg = db.synth_degraded(snr_db=15.0, seed=11)
    for fn in (db.spectral_gating_ref, db.wiener_ref):
        out = fn(deg, 48000)
        _, _, edge = db.metrics(clean, out, 48000)
        assert edge < 3.0, f"{fn.__name__}: edge_peak_ratio={edge:.1f}"


def test_aurik_omlsa_edge_artifact_bounded() -> None:
    """Regressionstest: §DSP-Fix Rev. 2026-08-16 (boundary-Paarung) — kein NOLA-Spike.

    Historie: vor dem Fix war edge_peak_ratio≈393 und ref_snr≈−3,4 dB;
    nach dem Fix: edge=1,0 und ref_snr≈+5,2 dB (gemessen).
    """
    clean, deg = db.synth_degraded(snr_db=15.0, seed=11)
    out = db.aurik_omlsa(deg, 48000)
    _, _, edge = db.metrics(clean, out, 48000)
    assert edge < 3.0
