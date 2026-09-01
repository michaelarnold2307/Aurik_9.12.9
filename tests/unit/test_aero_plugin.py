"""Tests für plugins/aero_plugin.py — Challenger-Kandidat (12→48 kHz BWE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plugins.aero_plugin import _HR_SR, _LR_SR, AeroPlugin

_MODEL = Path(__file__).resolve().parents[2] / "models" / "aero" / "checkpoint_12-48_hl256.th"


def test_plugin_importable_without_model() -> None:
    plugin = AeroPlugin(checkpoint=Path("/nonexistent/checkpoint.th"))
    assert plugin.is_loaded is False
    assert plugin.enhance(np.zeros(12000, dtype=np.float32), 12000) is None


@pytest.mark.skipif(not _MODEL.exists(), reason="AERO-Checkpoint nicht vorhanden")
def test_plugin_loads_and_upsamples_deterministically() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    plugin = AeroPlugin(device="cpu")
    assert plugin.is_loaded, "AERO-Modell konnte nicht geladen werden"
    rng = np.random.RandomState(7)
    audio = (0.1 * rng.randn(12000)).astype(np.float32)
    out1 = plugin.enhance(audio, 12000)
    out2 = plugin.enhance(audio, 12000)
    assert out1 is not None and out2 is not None
    assert out1.shape == (12000 * (_HR_SR // _LR_SR),)
    assert np.all(np.isfinite(out1))
    assert np.max(np.abs(out1)) <= 1.0 + 1e-6
    np.testing.assert_array_equal(out1, out2)  # deterministisch (§G5 (GEBOTE.md))
