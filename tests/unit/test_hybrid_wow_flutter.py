import numpy as np
import pytest


@pytest.mark.unit
def test_polyphonic_implausible_speed_curve_falls_back_to_pyin(monkeypatch):
    from backend.core.hybrid.hybrid_wow_flutter import PolyphonicSpeedCurveEstimator

    class _FakeBasicPitchResult:
        def __init__(self, t: int, k: int):
            # Create a huge pitch jump that leads to implausible cents deviation.
            # First half sets per-voice reference medians to ~100 Hz.
            # Second half pushes > +500 cents after clipping, thus >200 cents final range.
            self.pitches_hz = np.full((t, k), 100.0, dtype=np.float32)
            self.pitches_hz[t // 2 :, :] = 2000.0
            self.confidences = np.full((t, k), 0.95, dtype=np.float32)
            self.frame_times_s = np.arange(t, dtype=np.float32) * 0.01

    class _FakeBasicPitch:
        _model_loaded = True

        def analyze(self, _audio, _sr, max_polyphony=6):
            return _FakeBasicPitchResult(t=120, k=min(3, max_polyphony))

    est = PolyphonicSpeedCurveEstimator()
    est._bp = _FakeBasicPitch()  # type: ignore[assignment]

    fallback_pitch = np.full(120, 220.0, dtype=np.float32)
    fallback_conf = np.full(120, 0.42, dtype=np.float32)

    def _fake_pyin_fallback(_audio, _sr):
        return fallback_pitch, fallback_conf

    monkeypatch.setattr(est, "_pyin_fallback", _fake_pyin_fallback)

    audio = np.random.randn(48000).astype(np.float32) * 0.01
    pitch, conf = est.estimate(audio, 48000)

    assert np.allclose(pitch, fallback_pitch)
    assert np.allclose(conf, fallback_conf)


def _make_estimator_with_fake(monkeypatch, pitches, confidences, t, k):
    from backend.core.hybrid.hybrid_wow_flutter import PolyphonicSpeedCurveEstimator

    class _FakeResult:
        def __init__(self):
            self.pitches_hz = np.asarray(pitches, dtype=np.float32)
            self.confidences = np.asarray(confidences, dtype=np.float32)
            self.frame_times_s = np.arange(t, dtype=np.float32) * 0.01

    class _FakeBP:
        _model_loaded = True

        def analyze(self, _audio, _sr, max_polyphony=6):
            return _FakeResult()

    est = PolyphonicSpeedCurveEstimator()
    est._bp = _FakeBP()  # type: ignore[assignment]
    return est


@pytest.mark.unit
def test_single_voice_curve_used_instead_of_fallback(monkeypatch):
    """Stufe 2: exakt 1 Stimme → Single-Voice-Kurve statt pYIN-Fallback."""
    t, k = 120, 1
    pitches = np.full((t, k), 440.0, dtype=np.float32)
    confs = np.full((t, k), 0.95, dtype=np.float32)
    est = _make_estimator_with_fake(monkeypatch, pitches, confs, t, k)

    fallback_pitch = np.full(t, 220.0, dtype=np.float32)

    def _fake_pyin(_audio, _sr):
        return fallback_pitch, np.full(t, 0.42, dtype=np.float32)

    monkeypatch.setattr(est, "_pyin_fallback", _fake_pyin)
    pitch, conf = est.estimate(np.random.randn(48000).astype(np.float32) * 0.01, 48000)

    assert not np.allclose(pitch, fallback_pitch), "Single-Voice-Pfad darf nicht in pYIN fallen"
    assert np.allclose(pitch, 440.0, atol=2.0)  # deviation 0 → virtuelle Referenz
    assert float(np.median(conf[conf > 0])) == pytest.approx(0.6, abs=0.05)


@pytest.mark.unit
def test_single_voice_tracks_wow_deviation(monkeypatch):
    """Stufe 2: eine Stimme mit sinusförmigem Wow → Kurve folgt dem Drift."""
    t, k = 240, 1
    drift_cents = 25.0 * np.sin(2 * np.pi * np.arange(t) / 120.0)
    pitches = (440.0 * np.power(2.0, drift_cents / 1200.0)).astype(np.float32)[:, None]
    confs = np.full((t, k), 0.95, dtype=np.float32)
    est = _make_estimator_with_fake(monkeypatch, pitches, confs, t, k)

    pitch, conf = est.estimate(np.random.randn(48000).astype(np.float32) * 0.01, 48000)
    # Rückrechnung: speed_cents = 1200·log2(pitch/440)
    with np.errstate(divide="ignore"):
        speed_cents = 1200.0 * np.log2(np.clip(pitch, 20.0, None) / 440.0)
    speed_cents = np.nan_to_num(speed_cents, nan=0.0)
    # SG-geglättete Kurve korreliert mit dem Soll-Drift (Vorzeichen-konsistent)
    corr = float(np.corrcoef(speed_cents, drift_cents)[0, 1])
    assert corr > 0.9, f"Kurve folgt dem Wow-Drift nicht (corr={corr:.3f})"


@pytest.mark.unit
def test_no_voiced_frames_falls_back_to_pyin(monkeypatch):
    """Stufe 3: keine stimmhaften Frames (Konfidenz < Gate) → pYIN."""
    t, k = 120, 1
    pitches = np.full((t, k), 440.0, dtype=np.float32)
    confs = np.full((t, k), 0.05, dtype=np.float32)  # unter _MIN_CONF
    est = _make_estimator_with_fake(monkeypatch, pitches, confs, t, k)

    fallback_pitch = np.full(t, 220.0, dtype=np.float32)

    def _fake_pyin(_audio, _sr):
        return fallback_pitch, np.full(t, 0.42, dtype=np.float32)

    monkeypatch.setattr(est, "_pyin_fallback", _fake_pyin)
    pitch, conf = est.estimate(np.random.randn(48000).astype(np.float32) * 0.01, 48000)
    assert np.allclose(pitch, fallback_pitch)
