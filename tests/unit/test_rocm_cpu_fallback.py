"""tests/unit/test_rocm_cpu_fallback.py

Tests für die §ROCm-Fallback-Fixes (2026-08-16, Spec 23-Kontext):
  1. LAION-CLAP ONNX-Inferenz: MIOpen-Kernel-Fehler → CPU-only Session-Retry
  2. MelBandRoformer: Nicht-Memory-ONNX-Fehler → CPU-Retry statt Chunk-Halbierung
  3. EraClassifier Tier-1: Plugin ohne geladenes Modell wird nicht aufgerufen
     (keine RuntimeError-Tracebacks im Log — §V6-konforme eine Warnung)

Nur synthetische Signale und Mocks; kein echtes Modell wird geladen.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.core.era_classifier import EraClassifier


class _FlakySession:
    """ONNX-Session-Mock: erster run() wirft (ROCm/MIOpen), danach OK."""

    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list:
        return [SimpleNamespace(name="input")]

    def run(self, output_names, feeds):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("MIOPEN failure 7: miopenStatusUnknownError")
        return [np.zeros((1, 512), dtype=np.float32)]


def _make_clap_plugin() -> object:
    """Erzeugt ein LAIONCLAPPlugin-Objekt ohne Konstruktor-Nebenwirkungen."""
    from plugins.laion_clap_plugin import LAIONCLAPPlugin

    plugin = LAIONCLAPPlugin.__new__(LAIONCLAPPlugin)
    plugin._model_loaded = True
    plugin._clap_model = None
    plugin._ensure_loaded = lambda: None  # type: ignore[attr-defined]
    return plugin


# ─── 1. LAION-CLAP: CPU-Fallback-Retry ──────────────────────────────────────


def test_laion_clap_embed_audio_cpu_fallback_retry() -> None:
    plugin = _make_clap_plugin()
    flaky = _FlakySession()
    cpu_session = _FlakySession()  # run() schlägt nie fehl (calls==1 → ok)
    cpu_session.calls = 1  # erster Aufruf auf CPU-Session ist bereits "2. Versuch"
    plugin._audio_session = flaky
    plugin._audio_session_model_path = "fake_audio_encoder.onnx"

    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    with patch("onnxruntime.InferenceSession", return_value=cpu_session):
        emb = plugin.embed_audio(audio, 48000)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    # Nach dem Fallback zeigt die Session auf die CPU-Instanz.
    assert plugin._audio_session is cpu_session


def test_laion_clap_embed_audio_raises_without_model_path() -> None:
    plugin = _make_clap_plugin()
    flaky = _FlakySession()
    plugin._audio_session = flaky
    plugin._audio_session_model_path = None

    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    with pytest.raises(RuntimeError):
        plugin.embed_audio(audio, 48000)


# ─── 2. MelBandRoformer: CPU-Retry-Session-Bau ──────────────────────────────


def test_bs_roformer_build_cpu_session_without_path_is_none() -> None:
    from plugins.bs_roformer_plugin import BSRoFormerPlugin

    plugin = BSRoFormerPlugin.__new__(BSRoFormerPlugin)
    plugin._session_model_path = None
    assert plugin._build_cpu_session() is None


def test_bs_roformer_build_cpu_session_with_path_uses_cpu_provider() -> None:
    from plugins.bs_roformer_plugin import BSRoFormerPlugin

    plugin = BSRoFormerPlugin.__new__(BSRoFormerPlugin)
    plugin._session_model_path = "fake_roformer.onnx"
    cpu_session = MagicMock()
    with patch("onnxruntime.InferenceSession", return_value=cpu_session) as ort_mock:
        result = plugin._build_cpu_session()
    assert result is cpu_session
    ort_mock.assert_called_once_with("fake_roformer.onnx", providers=["CPUExecutionProvider"])


# ─── 3. EraClassifier Tier-1: kein Aufruf ohne geladenes Modell ─────────────


def _make_era_classifier(plugin_model_loaded: bool) -> tuple[EraClassifier, MagicMock]:
    era = EraClassifier.__new__(EraClassifier)
    era._clap_lock = threading.Lock()
    era._clap_loaded = True
    clap = MagicMock()
    clap._model_loaded = plugin_model_loaded
    clap.embed_audio.return_value = np.zeros(512, dtype=np.float32)
    era._clap_plugin = clap
    era._clap_nearest_neighbor = lambda emb: (1977, 0.9)  # type: ignore[attr-defined]
    return era, clap


def test_era_tier1_lazy_loads_when_clap_model_not_loaded() -> None:
    """Contract 2026-08-22 (CLAP-Lazy-Load-Fix): `_model_loaded=False` darf Tier-1
    NICHT überspringen — der Plugin-Vertrag ist Lazy-Load IN `embed_audio()`
    (thread-sicher via `_load_lock`). Der frühere Vorcheck verhinderte Tier-1
    deterministisch beim ersten Aufruf (Befund: WARNUNG 14:16:03, CLAP erst
    14:16:24 geladen → Tier-2 lief trotz verfügbarem CLAP).
    """
    era, clap = _make_era_classifier(plugin_model_loaded=False)
    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    result = era._try_tier1(audio, 48000, bark=np.zeros(24), rolloff_hz=12000.0, _snr_db=30.0)
    # Lazy-Load: embed_audio() wird GERUFEN und lädt selbst — Tier-1 läuft.
    clap.embed_audio.assert_called_once()
    assert result is not None
    assert result.tier_used == 1


def test_era_tier1_falls_back_on_load_failure() -> None:
    """§G23/§V6: Totaler Ladefehler (RuntimeError aus embed_audio) → DSP-Ersatzpfad
    (None), kein Crash, kein stiller Erfolg.
    """
    era, clap = _make_era_classifier(plugin_model_loaded=False)
    clap.embed_audio.side_effect = RuntimeError("CLAP-Modell nicht ladbar")
    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    result = era._try_tier1(audio, 48000, bark=np.zeros(24), rolloff_hz=12000.0, _snr_db=30.0)
    assert result is None


def test_era_tier1_embeds_when_clap_model_loaded() -> None:
    era, clap = _make_era_classifier(plugin_model_loaded=True)
    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    result = era._try_tier1(audio, 48000, bark=np.zeros(24), rolloff_hz=12000.0, _snr_db=30.0)
    assert result is not None
    # 1977 wird durch __post_init__ auf VALID_DECADES geschnappt (1980) —
    # Label muss konsistent mit dem geschnappten decade sein (§Invariante).
    assert result.decade == 1980
    assert result.era_label == "1980er"
    assert result.tier_used == 1
    clap.embed_audio.assert_called_once()


# ─── 4. PANNs: CPU-Fallback-Retry (AveragePool-MIOpen-Fehler) ────────────────


def test_panns_get_tags_cpu_fallback_retry() -> None:
    from plugins.panns_plugin import PANNsPlugin

    plugin = PANNsPlugin.__new__(PANNsPlugin)
    flaky = _FlakySession()
    cpu_session = _FlakySession()
    cpu_session.calls = 1
    plugin._session = flaky
    plugin._use_fp16 = False
    plugin._device = "cuda"
    plugin._to_model_input = lambda audio, sr, position_ratio=0.5: np.zeros(  # type: ignore[attr-defined]
        (1, 1, 64, 101), dtype=np.float32
    )
    plugin._to_model_input_from_resampled = lambda mono_rs, pos: np.zeros(  # type: ignore[attr-defined]
        (1, 1, 64, 101), dtype=np.float32
    )

    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(32000) / 32000)).astype(np.float32)
    with patch("onnxruntime.InferenceSession", return_value=cpu_session):
        tags = plugin.get_tags(audio, 32000)

    assert isinstance(tags, dict)
    assert len(tags) > 0
    assert plugin._session is cpu_session
    assert plugin._device == "cpu"
    assert plugin._use_fp16 is False


# ─── 5. Zentrale ort_run_with_cpu_fallback-Helper ────────────────────────────


def test_ort_run_cpu_fallback_success_path() -> None:
    from backend.core.ml_device_manager import ort_run_with_cpu_fallback

    session = _FlakySession()
    session.calls = 1  # wirft nie
    factory = MagicMock(return_value=None)
    out = ort_run_with_cpu_fallback(session, {"input": np.zeros(1)}, rebuild_cpu_factory=factory)
    assert out[0].shape == (1, 512)
    factory.assert_not_called()


def test_ort_run_cpu_fallback_retries_once() -> None:
    from backend.core.ml_device_manager import ort_run_with_cpu_fallback

    flaky = _FlakySession()
    cpu_session = _FlakySession()
    cpu_session.calls = 1
    factory = MagicMock(return_value=cpu_session)
    out = ort_run_with_cpu_fallback(flaky, {"input": np.zeros(1)}, rebuild_cpu_factory=factory)
    assert out[0].shape == (1, 512)
    factory.assert_called_once()


def test_ort_run_cpu_fallback_propagates_without_factory() -> None:
    from backend.core.ml_device_manager import ort_run_with_cpu_fallback

    flaky = _FlakySession()
    with pytest.raises(RuntimeError):
        ort_run_with_cpu_fallback(flaky, {"input": np.zeros(1)})


def test_ort_run_cpu_fallback_propagates_when_factory_none() -> None:
    from backend.core.ml_device_manager import ort_run_with_cpu_fallback

    flaky = _FlakySession()
    with pytest.raises(RuntimeError):
        ort_run_with_cpu_fallback(flaky, {"input": np.zeros(1)}, rebuild_cpu_factory=lambda: None)
