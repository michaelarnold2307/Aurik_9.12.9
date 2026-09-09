import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")  # CI-Minimal-Umgebung (cross-platform)

from backend.api.rest import batch_endpoints


@pytest.fixture
def temp_batch_dirs(monkeypatch, tmp_path):
    in_dir = tmp_path / "input_audio"
    out_dir = tmp_path / "output_audio"
    in_dir.mkdir()
    out_dir.mkdir()
    monkeypatch.setattr(batch_endpoints, "AUDIO_IN_DIR", in_dir)
    monkeypatch.setattr(batch_endpoints, "AUDIO_OUT_DIR", out_dir)
    return in_dir, out_dir


def test_sanitize_batch_filename_rejects_path_traversal():
    assert batch_endpoints._sanitize_batch_filename("../../etc/passwd.wav") is None
    assert batch_endpoints._sanitize_batch_filename("subdir/evil.wav") is None
    assert batch_endpoints._sanitize_batch_filename("subdir\\evil.wav") is None
    assert batch_endpoints._sanitize_batch_filename("safe_track.wav") == "safe_track.wav"


def test_upload_storage_filename_never_uses_client_filename():
    storage_name = batch_endpoints._upload_storage_filename("safe_track.wav")

    assert storage_name is not None
    assert storage_name.endswith(".wav")
    assert "safe_track" not in storage_name
    assert batch_endpoints._upload_storage_filename("../../etc/passwd.wav") is None


def test_batch_file_path_is_contained_in_managed_directory(temp_batch_dirs):
    in_dir, _ = temp_batch_dirs

    assert batch_endpoints._batch_file_path(in_dir, "safe_track.wav") == in_dir / "safe_track.wav"
    with pytest.raises(ValueError, match="Unsicherer Batch-Dateiname"):
        batch_endpoints._batch_file_path(in_dir, "../outside.wav")


def test_htdemucs_resampling_uses_torch_tensors(monkeypatch):
    from plugins.htdemucs_plugin import HtdemucsPlugin

    plugin = HtdemucsPlugin()
    plugin._model_type = "pytorch"
    plugin._model = object()

    def fake_ensure_model():
        return None

    monkeypatch.setattr(plugin, "_ensure_model", fake_ensure_model)

    def fake_separate_pytorch(audio_2ch):
        return [audio_2ch, audio_2ch, audio_2ch, audio_2ch]

    monkeypatch.setattr(plugin, "_separate_pytorch", fake_separate_pytorch)

    fake_julius = types.SimpleNamespace(
        ResampleFrac=lambda in_sr, out_sr: types.SimpleNamespace(in_sr=in_sr, out_sr=out_sr),
        resample_frac=lambda frac, tensor: tensor,
    )
    monkeypatch.setitem(sys.modules, "julius", fake_julius)

    audio = np.linspace(-0.5, 0.5, 2048, dtype=np.float32)
    result = plugin.separate(audio, sr=22050)

    assert result.vocals.shape == audio.shape
    assert result.sr == 22050
