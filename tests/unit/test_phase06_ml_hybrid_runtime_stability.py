from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("psutil")  # CI-Minimal-Umgebung (cross-platform) ohne psutil
import psutil
import scipy.signal as signal

from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase


class _FreeMemVM:
    """Mocked psutil.virtual_memory — 32 GB free, bypasses phase_06 headroom guard."""

    available: int = 32 * 1024**3
    total: int = 32 * 1024**3
    percent: float = 10.0


class _FakeFlashSRPlugin:
    def process(self, audio: np.ndarray, sr: int, target_sr: int = 48_000) -> np.ndarray:
        # Deterministic HF boost-like behavior for testability.
        _ = (sr, target_sr)
        return np.clip(np.asarray(audio, dtype=np.float32) * 1.05, -1.0, 1.0)


def _make_rolloff_audio(sr: int = 48_000, duration_s: float = 0.35) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / float(sr)
    src = (
        0.45 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 1100.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 4500.0 * t)
        + 0.18 * np.sin(2.0 * np.pi * 9000.0 * t)
        + 0.12 * np.sin(2.0 * np.pi * 14000.0 * t)
    ).astype(np.float32)
    src += 0.05 * np.random.RandomState(42).standard_normal(n).astype(np.float32)
    sos = signal.butter(8, 4200.0 / (sr / 2.0), btype="low", output="sos")
    rolled = signal.sosfiltfilt(sos, src)
    return np.column_stack([rolled, rolled * 0.98]).astype(np.float32)


@pytest.mark.unit
def test_phase06_uses_ml_hybrid_when_flashsr_available(monkeypatch) -> None:
    # Mock psutil so the phase_06 headroom guard (requires 9.5 GB available) passes
    # on memory-constrained dev machines. Tests ML path logic, not RAM availability.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FreeMemVM())
    # §VERBOTEN: Budget-Tests ohne is_system_thrashing-Mock → flaky auf Hosts mit hoher Swap-Last
    monkeypatch.setattr("backend.core.ml_memory_budget.is_system_thrashing", lambda: False)
    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = _make_rolloff_audio()

    monkeypatch.setattr(
        "backend.core.phases.phase_06_frequency_restoration._get_flashsr_plugin",
        lambda: _FakeFlashSRPlugin(),
    )

    result = phase.process(
        audio,
        sample_rate=48_000,
        material_type="shellac",
        quality_mode="maximum",
        flashsr_min_duration_s=0.0,
    )

    assert result.success
    assert result.modifications.get("frequency_restored") is True
    assert result.metadata.get("strategy_used") == "ml_hybrid"
    assert float(result.metadata.get("ml_blend_alpha", 0.0)) > 0.0


def test_phase06_falls_back_to_dsp_on_ml_error(monkeypatch) -> None:
    # Mock psutil so the phase_06 headroom guard passes, allowing FlashSR to be
    # attempted (and fail with RuntimeError) — which sets ml_error in metadata.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FreeMemVM())

    class _BrokenAudioSRPlugin:
        def process(self, audio: np.ndarray, sr: int, target_sr: int = 48_000) -> np.ndarray:
            _ = (audio, sr, target_sr)
            raise RuntimeError("synthetic failure")

    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = _make_rolloff_audio()

    monkeypatch.setattr(
        "backend.core.phases.phase_06_frequency_restoration._get_flashsr_plugin",
        lambda: _BrokenAudioSRPlugin(),
    )

    result = phase.process(
        audio,
        sample_rate=48_000,
        material_type="shellac",
        quality_mode="maximum",
        flashsr_min_duration_s=0.0,
    )

    assert result.success
    assert result.modifications.get("frequency_restored") is True
    assert result.metadata.get("strategy_used") == "dsp_only"
    assert "ml_error" in result.metadata


def test_phase06_skips_flashsr_for_short_clip_guard(monkeypatch) -> None:
    called = {"plugin": False}

    class _ShouldNotBeCalledPlugin:
        def process(self, audio: np.ndarray, sr: int, target_sr: int = 48_000) -> np.ndarray:
            _ = (audio, sr, target_sr)
            called["plugin"] = True
            return np.asarray(audio, dtype=np.float32)

    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = _make_rolloff_audio(duration_s=0.35)

    monkeypatch.setattr(
        "backend.core.phases.phase_06_frequency_restoration._get_flashsr_plugin",
        lambda: _ShouldNotBeCalledPlugin(),
    )

    result = phase.process(
        audio,
        sample_rate=48_000,
        material_type="shellac",
        quality_mode="maximum",
    )

    assert result.success
    assert result.modifications.get("frequency_restored") is True
    assert result.metadata.get("strategy_used") == "dsp_only"
    assert "short_clip_guard" in str(result.metadata.get("ml_reason", ""))
    assert called["plugin"] is False


def test_phase06_skips_flashsr_on_thrashing_guard(monkeypatch) -> None:
    import backend.core.phases.phase_06_frequency_restoration as p06

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FreeMemVM())
    monkeypatch.setattr(p06, "_get_flashsr_plugin", lambda: _FakeFlashSRPlugin())
    monkeypatch.setattr("backend.core.ml_memory_budget.is_system_thrashing", lambda: True)

    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = np.zeros((2, 512), dtype=np.float32)
    monkeypatch.setattr(phase, "_restore_highs_professional", lambda a, _p, _sbr: a.copy())

    restored, meta = phase._restore_frequency_ml_hybrid(
        audio,
        params={"rolloff_hz": 6000.0, "restoration_strength": 0.7},
        material_type="shellac",
        quality_mode="maximum",
        enable_sbr=True,
        flashsr_min_duration_s=0.0,
    )

    assert restored.shape == audio.shape
    assert meta.get("strategy_used") == "dsp_only"
    assert meta.get("ml_reason") == "flashsr_thrashing_guard"


def test_phase06_releases_plm_active_flag_on_timeout(monkeypatch) -> None:
    import backend.core.phases.phase_06_frequency_restoration as p06

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FreeMemVM())
    monkeypatch.setattr("backend.core.ml_memory_budget.is_system_thrashing", lambda: False)

    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = np.zeros((2, 512), dtype=np.float32)
    monkeypatch.setattr(phase, "_restore_highs_professional", lambda a, _p, _sbr: a.copy())

    class _DummyPlugin:
        def process(self, audio: np.ndarray, sr: int, target_sr: int = 48_000) -> np.ndarray:
            _ = (audio, sr, target_sr)
            return np.asarray(audio, dtype=np.float32)

    monkeypatch.setattr(p06, "_get_flashsr_plugin", lambda: _DummyPlugin())

    class _FakePLM:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def set_active(self, name: str, active: bool) -> None:
            self.calls.append((name, active))

    fake_plm = _FakePLM()

    monkeypatch.setattr(
        "backend.core.plugin_lifecycle_manager.get_plugin_lifecycle_manager",
        lambda: fake_plm,
    )
    monkeypatch.setattr(
        "backend.core.plugin_lifecycle_manager.touch_plugin",
        lambda _name: None,
    )

    class _FastTimeoutThread:
        def __init__(self, target, daemon=True):
            _ = daemon
            self._target = target

        def start(self):
            return None

        def join(self, timeout=None):
            _ = timeout
            return None

    monkeypatch.setattr(threading, "Thread", _FastTimeoutThread)

    restored, meta = phase._restore_frequency_ml_hybrid(
        audio,
        params={"rolloff_hz": 6000.0, "restoration_strength": 0.7},
        material_type="shellac",
        quality_mode="maximum",
        enable_sbr=True,
        flashsr_min_duration_s=0.0,
    )

    assert restored.shape == audio.shape
    assert meta.get("strategy_used") == "dsp_only"
    assert meta.get("ml_error") == "timeout"
    assert ("FlashSR", True) in fake_plm.calls
    assert ("FlashSR", False) in fake_plm.calls
