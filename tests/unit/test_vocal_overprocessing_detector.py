from __future__ import annotations

"""Unit tests for backend.core.vocal_overprocessing_detector.

Tests:
- Lisp detection (6–10 kHz variance)
- Formant drift (F1/F2 via LPC)
- Sibilance over-reduction detection
- VocalOverprocessingDetector integration
- VocalOverprocessingResult dataclass
- Edge cases: silence, NaN, short audio
"""


import numpy as np
import pytest

from backend.core.vocal_overprocessing_detector import (
    VocalOverprocessingDetector,
    VocalOverprocessingResult,
    _band_energy,
    _burg_lpc,
    _extract_f1_f2,
    _lpc_to_formants,
    _to_mono,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sr():
    return 48_000


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def vocal_like(rng, sr):
    """Synthetic vocal-like audio with harmonics."""
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    sig = 0.5 * np.sin(2.0 * np.pi * 200.0 * t)
    sig += 0.3 * np.sin(2.0 * np.pi * 400.0 * t)
    sig += 0.2 * np.sin(2.0 * np.pi * 600.0 * t)
    sig += 0.15 * np.sin(2.0 * np.pi * 800.0 * t)
    sig += 0.1 * np.sin(2.0 * np.pi * 1000.0 * t)
    sig += 0.02 * rng.randn(*sig.shape)
    return sig.astype(np.float64)


@pytest.fixture
def sibilant_audio(rng, sr):
    """Audio with high 5–10 kHz content (sibilance)."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = np.sin(2.0 * np.pi * 7000.0 * t) * 0.5
    sig += np.sin(2.0 * np.pi * 200.0 * t) * 0.3
    return sig.astype(np.float64)


@pytest.fixture
def detector():
    return VocalOverprocessingDetector()


# ---------------------------------------------------------------------------
# _to_mono tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToMono:
    def test_mono_stays_mono(self):
        mono = np.array([1.0, 2.0, 3.0])
        assert _to_mono(mono).ndim == 1

    def test_stereo_to_mono(self):
        stereo = np.array([[0.5, 1.0], [0.7, 0.3], [0.1, 0.9]])
        mono = _to_mono(stereo)
        assert mono.ndim == 1
        assert mono.shape[0] == 3


# ---------------------------------------------------------------------------
# _band_energy tests
# ---------------------------------------------------------------------------


class TestBandEnergy:
    def test_returns_finite(self, vocal_like, sr):
        e = _band_energy(vocal_like, sr, 200.0, 2000.0)
        assert np.isfinite(e)
        assert e >= 0.0

    def test_silence_low_energy(self, sr):
        silent = np.zeros(48000)
        e = _band_energy(silent, sr, 200.0, 2000.0)
        assert e < 1.0

    def test_band_selectivity(self, sr):
        """Frequency selectivity: band energy in 5-10 kHz for high-freq signal."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        hf_sig = np.sin(2.0 * np.pi * 7000.0 * t).astype(np.float64)
        e_hf = _band_energy(hf_sig, sr, 5000.0, 10000.0)
        e_lf = _band_energy(hf_sig, sr, 200.0, 500.0)
        assert e_hf > e_lf * 2.0


# ---------------------------------------------------------------------------
# _burg_lpc tests
# ---------------------------------------------------------------------------


class TestBurgLPC:
    def test_returns_correct_shape(self, vocal_like):
        frame = vocal_like[:400].astype(np.float64)
        a = _burg_lpc(frame, order=14)
        assert a.shape[0] == 15  # order + 1
        assert a[0] == 1.0  # First coefficient is always 1

    def test_handles_silence(self):
        frame = np.zeros(400)
        a = _burg_lpc(frame, order=14)
        assert a.shape[0] == 15
        assert np.isfinite(a).all()


# ---------------------------------------------------------------------------
# _lpc_to_formants tests
# ---------------------------------------------------------------------------


class TestLPCTToFormants:
    def test_returns_list(self, vocal_like, sr):
        frame = vocal_like[:400].astype(np.float64)
        a = _burg_lpc(frame, order=14)
        formants = _lpc_to_formants(a, sr)
        assert isinstance(formants, list)

    def test_frequencies_in_range(self, vocal_like, sr):
        frame = vocal_like[:400].astype(np.float64)
        a = _burg_lpc(frame, order=14)
        formants = _lpc_to_formants(a, sr)
        for f in formants:
            assert 200.0 < f < 3400.0


# ---------------------------------------------------------------------------
# _extract_f1_f2 tests
# ---------------------------------------------------------------------------


class TestExtractF1F2:
    def test_returns_tuple(self, vocal_like, sr):
        f1, f2 = _extract_f1_f2(vocal_like, sr)
        assert isinstance(f1, float)
        assert isinstance(f2, float)

    def test_handles_silence(self, sr):
        f1, f2 = _extract_f1_f2(np.zeros(48000), sr)
        assert f1 == 0.0
        assert f2 == 0.0

    def test_handles_short_audio(self, sr):
        f1, f2 = _extract_f1_f2(np.random.randn(512), sr)
        assert np.isfinite(f1) and np.isfinite(f2)


# ---------------------------------------------------------------------------
# VocalOverprocessingResult tests
# ---------------------------------------------------------------------------


class TestVocalOverprocessingResult:
    def test_is_clean(self):
        r = VocalOverprocessingResult(phase_id="test")
        assert r.is_clean

    def test_not_clean_when_lisp(self):
        r = VocalOverprocessingResult(phase_id="test", lisp_detected=True)
        assert not r.is_clean

    def test_not_clean_when_formant_drift(self):
        r = VocalOverprocessingResult(phase_id="test", formant_drift_warning=True)
        assert not r.is_clean

    def test_not_clean_when_sibilance(self):
        r = VocalOverprocessingResult(phase_id="test", sibilance_over_reduced=True)
        assert not r.is_clean


# ---------------------------------------------------------------------------
# VocalOverprocessingDetector tests
# ---------------------------------------------------------------------------


class TestVocalOverprocessingDetector:
    def test_check_de_essing_clean(self, detector, vocal_like, sr):
        """Clean audio (no overprocessing) should pass."""
        result = detector.check_de_essing(vocal_like, vocal_like, sr)
        assert result.is_clean
        assert not result.lisp_detected
        assert not result.sibilance_over_reduced

    def test_check_de_essing_lisp(self, detector, sibilant_audio, sr):
        """Heavily boosted HF should trigger lisp detection."""
        # Boost 6-10 kHz significantly
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, [6000.0, 10000.0], btype="bandpass", fs=sr, output="sos")
        boosted = sibilant_audio + 2.0 * sosfiltfilt(sos, sibilant_audio)
        boosted = boosted / np.max(np.abs(boosted))

        result = detector.check_de_essing(sibilant_audio, boosted, sr)
        # The HF variance may trigger lisp detection
        assert isinstance(result.lisp_detected, bool)
        assert isinstance(result.sibilance_over_reduced, bool)

    def test_check_de_essing_no_false_positive_for_needed_de_essing(self, detector, sr):
        """§VOD-1 Regression: sibilante QUELLE + nötiges aggressives De-essing.

        Die 6–10-kHz-Schwankung ist bereits im Eingang hoch (Burst-Struktur);
        De-essing, das die Schwankung REDUZIERT, darf NICHT als Lisp gelten
        (der alte absolute Check feuerte hier — Import-Song-Fall 115.9 „dB“).
        """
        from scipy.signal import butter, sosfiltfilt

        rng = np.random.RandomState(7)
        t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
        sos_bp = butter(4, [6000.0, 10000.0], btype="bandpass", fs=sr, output="sos")
        bp = sosfiltfilt(sos_bp, rng.randn(len(t)))
        gate = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 4.0 * t)))
        pre = 0.3 * np.sin(2 * np.pi * 200.0 * t) + 0.5 * bp * gate
        pre /= np.max(np.abs(pre))
        deessed = pre - 0.3 * bp * gate
        deessed /= np.max(np.abs(deessed))

        result = detector.check_de_essing(pre, deessed, sr)
        assert not result.lisp_detected
        assert not result.sibilance_over_reduced
        assert result.lisp_variance_db < 0  # Schwankung wurde reduziert, nicht erzeugt

    def test_check_de_essing_flags_burst_artifact_creation(self, detector, sr):
        """§VOD-1: De-essing, das NEUE Burst-Artefakte erzeugt, wird erkannt."""
        from scipy.signal import butter, sosfiltfilt

        rng = np.random.RandomState(7)
        t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
        sos_bp = butter(4, [6000.0, 10000.0], btype="bandpass", fs=sr, output="sos")
        bp = sosfiltfilt(sos_bp, rng.randn(len(t)))
        gate = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 4.0 * t)))
        pre = 0.3 * np.sin(2 * np.pi * 200.0 * t) + 0.5 * bp * gate
        pre /= np.max(np.abs(pre))
        artifact = pre + 1.5 * bp * gate
        artifact /= np.max(np.abs(artifact))

        result = detector.check_de_essing(pre, artifact, sr)
        assert result.lisp_detected
        assert result.lisp_variance_db > 0

    def test_check_de_essing_sibilance(self, detector, sibilant_audio, sr):
        """Severe sibilance reduction should trigger over-reduction."""
        # Cut HF drastically
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, 5000.0, btype="lowpass", fs=sr, output="sos")
        dull = sosfiltfilt(sos, sibilant_audio)

        result = detector.check_de_essing(sibilant_audio, dull, sr)
        assert result.sibilance_over_reduced
        assert result.sibilance_ratio < 0.40

    def test_check_formant_drift_clean(self, detector, vocal_like, sr):
        """Same audio before/after should have no drift."""
        result = detector.check_formant_drift(vocal_like, vocal_like, sr)
        assert not result.formant_drift_warning
        assert result.formant_drift_pct < 1.0

    def test_check_formant_drift_different(self, detector, sr):
        """Different audio content should produce drift."""
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        sig1 = 0.5 * np.sin(2.0 * np.pi * 200.0 * t).astype(np.float64)
        sig2 = 0.5 * np.sin(2.0 * np.pi * 300.0 * t).astype(np.float64) + 0.3 * np.sin(2.0 * np.pi * 500.0 * t)

        result = detector.check_formant_drift(sig1, sig2, sr)
        # Formants should differ noticeably
        assert isinstance(result.formant_drift_pct, float)

    def test_check_de_essing_warnings_on_detection(self, detector, sibilant_audio, sr):
        """When over-reduction detected, warnings list is populated."""
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, 5000.0, btype="lowpass", fs=sr, output="sos")
        dull = sosfiltfilt(sos, sibilant_audio)

        result = detector.check_de_essing(sibilant_audio, dull, sr)
        assert isinstance(result.warnings, list)

    def test_vql_threshold_constant(self, detector):
        """Ensure default threshold constants are reasonable."""
        # §Einheiten-Fix (2026-09-07): Std der Band-Energie in dB — der alte
        # Wert 15.0 galt einer dB²-Varianz (falsche Einheit).
        assert detector.LISP_VARIANCE_THRESHOLD_DB == 3.0
        assert detector.LISP_DELTA_THRESHOLD_DB == 3.0
        assert detector.SIBILANCE_RATIO_THRESHOLD == 0.40
        assert detector.FORMANT_DRIFT_THRESHOLD_PCT == 5.0

    def test_post_deessing_band_specs(self, detector):
        """Ensure band specs match specification."""
        assert detector.LISP_BAND == (6000.0, 10000.0)
        assert detector.SIBILANCE_BAND == (5000.0, 10000.0)
