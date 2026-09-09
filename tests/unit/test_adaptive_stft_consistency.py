import numpy as np
import pytest

pytest.importorskip("librosa")  # CI-Minimal-Umgebung (cross-platform)

from dsp.adaptive_stft import AdaptiveSTFT


@pytest.mark.unit
def test_adaptive_stft_istft_uses_last_stft_window_by_default() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float64)

    stft_mod = AdaptiveSTFT(n_fft=1024, hop_length=256, win_length=1024, window="hann")
    D = stft_mod.stft(audio, sr=sr, window="hamming")
    rec = stft_mod.istft(D, sr=sr, length=len(audio))

    assert rec.shape == audio.shape
    assert np.isfinite(rec).all()


def test_adaptive_stft_fallback_shape_is_complex(monkeypatch) -> None:
    stft_mod = AdaptiveSTFT(n_fft=1024)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("forced test fallback")

    monkeypatch.setattr(stft_mod, "_stft_classic", _raise)
    good = np.zeros(2048, dtype=np.float64)
    out = stft_mod.stft(good, sr=48000)

    assert out.ndim == 2
    assert np.iscomplexobj(out)
