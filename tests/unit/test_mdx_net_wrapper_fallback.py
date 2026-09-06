from __future__ import annotations

import numpy as np
import pytest

from backend.ml.inference_only.vocal_separation.mdx_net_wrapper import MDXNetSeparator


@pytest.mark.unit
def test_hpss_fallback_recombines_close_to_original(monkeypatch):
    separator = MDXNetSeparator(model_path="/does/not/exist.onnx", sample_rate=48000)
    audio = np.vstack(
        [
            np.linspace(-0.5, 0.5, 256, dtype=np.float32),
            np.linspace(0.5, -0.5, 256, dtype=np.float32),
        ]
    )

    monkeypatch.setattr("librosa.stft", lambda channel, n_fft, hop_length, center=True: channel)
    monkeypatch.setattr("librosa.decompose.hpss", lambda d, margin=2.0: (0.4 * d, 0.6 * d))
    monkeypatch.setattr("librosa.istft", lambda d, hop_length, length=None: np.asarray(d, dtype=np.float32))
    monkeypatch.setattr("librosa.util.fix_length", lambda x, size: np.asarray(x[:size], dtype=np.float32))

    vocals, instrumental = separator._fallback_separation(audio)
    recombined = vocals + instrumental

    np.testing.assert_allclose(recombined, audio, atol=1e-6)


def test_separate_returns_expected_stems_on_fallback(monkeypatch):
    separator = MDXNetSeparator(model_path="/does/not/exist.onnx", sample_rate=48000)
    audio = np.linspace(-0.25, 0.25, 512, dtype=np.float32)

    monkeypatch.setattr(
        separator,
        "_fallback_separation",
        lambda x: (np.stack([0.3 * x[0], 0.3 * x[1]]), np.stack([0.7 * x[0], 0.7 * x[1]])),
    )

    stems = separator.separate(audio, sr=48000, return_stems=True)

    assert set(stems.keys()) == {"vocals", "instrumental"}
    assert stems["vocals"].shape == (2, 512)
    assert stems["instrumental"].shape == (2, 512)
    np.testing.assert_allclose(stems["vocals"] + stems["instrumental"], np.stack([audio, audio]), atol=1e-6)
