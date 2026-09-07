"""Unit-Tests für das gemeinsame psychoakustische Front-End.

§Muster 1+4: EIN Frame liefert Roughness/Sharpness/Loudness/Maskierung —
konsistente Werte statt dreier getrennter STFT-Welten.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.psychoacoustic_frame import (
    PsychoacousticFrame,
    build_psychoacoustic_frame,
)


def _sine_vibrato(sr: int = 48000, dur_s: float = 10.0) -> np.ndarray:
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
    freq = 440.0 + 5.0 * np.sin(2 * np.pi * 0.5 * t)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _am70(sr: int = 48000, dur_s: float = 10.0) -> np.ndarray:
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440.0 * t) * (1.0 + 0.8 * np.sin(2 * np.pi * 70.0 * t))).astype(np.float32)


def test_frame_builds_for_mono_and_stereo() -> None:
    sr = 48000
    mono = _sine_vibrato(sr)
    frame = build_psychoacoustic_frame(mono, sr)
    assert isinstance(frame, PsychoacousticFrame)
    assert frame.n_bark == 24
    assert frame.n_frames > 0
    assert len(frame.onset_env) > 0
    assert frame.masking_threshold_db.shape == (24,)

    stereo = np.stack([mono, mono * 0.9])
    frame_st = build_psychoacoustic_frame(stereo, sr)
    assert frame_st.n_frames == frame.n_frames


def test_clean_sine_roughness_below_threshold() -> None:
    """Sauberer 440-Hz-Sinus (Vibrato) darf keine Roughness-Spitze > 0.5 zeigen."""
    sr = 48000
    frame = build_psychoacoustic_frame(_sine_vibrato(sr), sr)
    windows = frame.roughness_windows(5.0, 2.5)
    assert windows, "Roughness-Fenster fehlen"
    assert float(np.max(windows)) < 0.5
    assert float(np.mean(windows)) < 0.2


def test_am70_is_rough() -> None:
    """70-Hz-Amplitudenmodulation liegt im Roughness-Band → hohe Werte."""
    sr = 48000
    frame = build_psychoacoustic_frame(_am70(sr), sr)
    windows = frame.roughness_windows(5.0, 2.5)
    assert windows
    assert float(np.max(windows)) > 0.8


def test_sharpness_and_loudness_bounded() -> None:
    sr = 48000
    frame = build_psychoacoustic_frame(_sine_vibrato(sr), sr)
    sharp = frame.sharpness_windows(5.0, 2.5)
    loud = frame.loudness_windows(5.0, 2.5)
    assert sharp and loud
    assert all(0.0 <= v <= 5.0 for v in sharp)
    assert all(0.0 <= v <= 1.0 for v in loud)


def test_window_count_matches_hop_semantics() -> None:
    """Fenster-Anzahl folgt der alten Audio-Chunk-Schleife (step = hop)."""
    sr = 48000
    dur_s = 10.0
    frame = build_psychoacoustic_frame(_sine_vibrato(sr, dur_s), sr)
    hop_s = 2.5
    expected = len(range(0, int(dur_s * sr), int(hop_s * sr)))
    assert len(frame.loudness_windows(5.0, hop_s)) == expected


def test_masking_threshold_is_signal_minus_spread() -> None:
    """Maskierungsschwelle = Band-Energie − 12 dB (ISO 11172-3 vereinfacht)."""
    sr = 48000
    frame = build_psychoacoustic_frame(_sine_vibrato(sr), sr)
    band = frame.band_energy_dbfs(400.0, 900.0)
    assert np.allclose(frame.masking_threshold_db, frame.bark_energy_db - 12.0, atol=1e-6)
    assert band <= 0.0  # dBFS nie positiv für 0.3-Amplitude
    assert frame.is_below_masking(-120.0, 40.0, 400.0)
    assert not frame.is_below_masking(0.0, 40.0, 400.0)


def test_line_energy_detects_hum_lines() -> None:
    """Hum-Linien-Energie misst nur die Linien, nicht das gesamte Band."""
    sr = 48000
    t = np.linspace(0.0, 2.0, int(sr * 2.0), endpoint=False)
    hum = (0.02 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    music = _sine_vibrato(sr, 2.0)
    frame_hum = build_psychoacoustic_frame(hum, sr)
    frame_mus = build_psychoacoustic_frame(music, sr)
    hum_dbfs = frame_hum.line_energy_dbfs([50.0, 100.0, 150.0])
    music_dbfs = frame_mus.line_energy_dbfs([50.0, 100.0, 150.0])
    assert hum_dbfs > -60.0  # Linien klar messbar
    assert hum_dbfs > music_dbfs  # Linien sind im Hum-Signal deutlich stärker


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
