from __future__ import annotations

"""Layout-Invarianten-Tests (§v10.x SOTA-Sweep 2026-08-22).

Kern-Invariante der Pipeline: (N, C) samples-first und (C, N)
channels-first sind durch reines Transponieren BIT-IDENTISCH. Jede
DSP-/Mess-Funktion muss diese Invariante erfüllen:
    f(x.T) == f(x).T   (bzw. f(mono(x)) für Mono-Ausgaben)

Diese Tests fangen die Bug-Klasse ab, die diese Session dominiert hat
(mean(axis=0)-Mono-Mixe: RLP, dtw_groove, Consensus, MDEM, TQC, GPO).
"""

import numpy as np
import pytest

from backend.core.audio_layout import (
    is_channels_first,
    is_samples_first,
    mono_mix,
    sample_axis,
    to_channels_first,
    to_samples_first,
)

SR = 48000
rng = np.random.default_rng(7)


def _stereo(secs: float = 1.0) -> np.ndarray:
    n = int(SR * secs)
    t = np.arange(n) / SR
    left = 0.4 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * np.sin(2 * np.pi * 6000.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 554.0 * t) + 0.1 * np.sin(2 * np.pi * 9000.0 * t)
    return np.stack([left, right], axis=1).astype(np.float32)  # (N, 2)


class TestLayoutHelpers:
    def test_01_detection(self):
        sf = _stereo()
        assert is_samples_first(sf)
        assert is_channels_first(sf.T)
        assert sample_axis(sf) == 0
        assert sample_axis(sf.T) == 1

    def test_02_mono_mix_layout_equivalent(self):
        sf = _stereo()
        m_sf = mono_mix(sf)
        m_cf = mono_mix(sf.T)
        assert m_sf.shape == m_cf.shape
        assert np.array_equal(m_sf, m_cf)

    def test_03_roundtrip_bit_identical(self):
        sf = _stereo()
        rt = to_samples_first(to_channels_first(sf))
        assert np.array_equal(rt, sf)  # bit-identisch, kein Copy-Artefakt

    def test_04_transpose_bit_identical(self):
        sf = _stereo()
        assert np.array_equal(np.ascontiguousarray(sf.T).T, sf)


class TestPipelineConsumers:
    """Kern-Konsumenten: Layout-Invariante f(x.T) == f(x).T / mono-identisch."""

    def test_01_detect_onsets(self):
        from dsp.dtw_groove import detect_onsets

        sf = _stereo()
        r_sf = detect_onsets(sf)
        r_cf = detect_onsets(sf.T)
        assert r_sf.n_onsets == r_cf.n_onsets

    def test_02_rlp_shelf(self):
        from backend.core.reflective_listening_pass import RLPIssue, ReflectiveListeningPass

        rlp = ReflectiveListeningPass()
        issue = RLPIssue(
            category="spectral_tilt",
            severity=0.6,
            detail="Invarianten-Test",
            correction={"eq_high_shelf_db": 2.0, "eq_freq_hz": 8000.0},
        )
        sf = _stereo()
        out_sf = rlp._apply_corrections(sf, SR, [issue], "vinyl")
        out_cf = rlp._apply_corrections(sf.T, SR, [issue], "vinyl")
        assert np.allclose(out_cf, out_sf.T, atol=1e-9)

    def test_03_mdem_lufs_profile(self):
        from backend.core.micro_dynamics_envelope_morphing import get_mdem

        mdem = get_mdem()
        sf = _stereo()
        p_sf = mdem.compute_lufs_profile(sf, SR)
        p_cf = mdem.compute_lufs_profile(sf.T, SR)
        assert p_sf.shape == p_cf.shape
        assert np.allclose(p_sf, p_cf, atol=1e-6)

    def test_04_tqc_measure(self):
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        sf = _stereo(26.0)
        r_sf = measure_temporal_coherence(sf, SR, material_key="vinyl")
        r_cf = measure_temporal_coherence(sf.T, SR, material_key="vinyl")
        assert abs(r_sf.max_span - r_cf.max_span) < 1e-6
        assert abs(r_sf.sigma - r_cf.sigma) < 1e-6

    def test_05_measure_all_layout(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker()
        sf = _stereo(5.0)
        s_sf = checker.measure_all(sf, SR)
        s_cf = checker.measure_all(sf.T, SR)
        assert set(s_sf.keys()) == set(s_cf.keys())
        for _k in s_sf:
            assert abs(s_sf[_k] - s_cf[_k]) < 1e-4, f"goal {_k} divergiert: {s_sf[_k]} vs {s_cf[_k]}"
