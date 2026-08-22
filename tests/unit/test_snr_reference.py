from __future__ import annotations

"""Referenz-SNR-Tests (§v10.x SNR-Kanonisierung 2026-08-22).

Die Kanonisierung definiert EINEN Referenz-Schätzer (aurik_snr_v1) mit
benannten Größen: source_snr_db / output_snr_db / bir_snr_proxy.
Gates vergleichen immer denselben Schätzer auf Quelle UND Ergebnis.
"""

import numpy as np
import pytest

from backend.core.snr_reference import (
    BIR_SNR_PROXY_KEY,
    OUTPUT_SNR_KEY,
    SILENCE_SNR_DB,
    SOURCE_SNR_KEY,
    SNR_DEFINITION_VERSION,
    estimate_snr_db,
    format_snr_label,
)

SR = 48000
rng = np.random.default_rng(11)


def _tone_snr(snr_target_db: float, secs: float = 2.0) -> np.ndarray:
    n = int(SR * secs)
    t = np.arange(n) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    noise = rng.standard_normal(n)
    noise_rms = 0.5 * 10 ** (-snr_target_db / 20.0)
    return (tone + noise_rms * noise).astype(np.float32)


class TestSnrReference:
    def test_01_definition_version(self):
        assert SNR_DEFINITION_VERSION == "aurik_snr_v1"
        assert {SOURCE_SNR_KEY, OUTPUT_SNR_KEY, BIR_SNR_PROXY_KEY} == {
            "source_snr_db",
            "output_snr_db",
            "bir_snr_proxy",
        }

    def test_02_silence_returns_guard(self):
        assert estimate_snr_db(np.zeros(SR, dtype=np.float32), SR) == SILENCE_SNR_DB

    def test_03_monotonic_with_injected_noise(self):
        """Höheres injiziertes SNR → höherer Schätzwert (streng monoton)."""
        vals = [estimate_snr_db(_tone_snr(s), SR) for s in (5.0, 15.0, 30.0, 45.0)]
        assert vals[0] < vals[1] < vals[2] < vals[3]

    def test_04_layout_invariant(self):
        sf = _tone_snr(20.0)
        stereo_sf = np.stack([sf, sf * 0.9], axis=1)
        stereo_cf = stereo_sf.T
        assert estimate_snr_db(stereo_sf, SR) == estimate_snr_db(stereo_cf, SR)

    def test_05_deterministic(self):
        a = _tone_snr(12.0)
        assert estimate_snr_db(a, SR) == estimate_snr_db(a.copy(), SR)

    def test_06_bounds(self):
        for s in (5.0, 40.0):
            v = estimate_snr_db(_tone_snr(s), SR)
            assert 0.0 <= v <= SILENCE_SNR_DB

    def test_07_label_format(self):
        assert format_snr_label(OUTPUT_SNR_KEY, 38.9) == "output_snr_db=38.9 dB"
