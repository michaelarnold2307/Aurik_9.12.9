import pytest

"""Unit tests for §SMR-1 SotaVocalModelRouter."""

import types

import numpy as np


def _audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_stem_routing_policy_detects_live_like_ctx_and_chain():
    from backend.core.dsp.stem_routing_policy import prefer_demucs_native_from_ctx

    assert prefer_demucs_native_from_ctx({"material_type": "live"}) is True
    assert prefer_demucs_native_from_ctx({"transfer_chain": ["vinyl", "stage_capture"]}) is True
    assert prefer_demucs_native_from_ctx({"material_type": "vinyl", "transfer_chain": ["tape", "mp3"]}) is False


def test_stem_routing_policy_detects_live_like_material_values():
    from backend.core.dsp.stem_routing_policy import prefer_demucs_native_from_material

    assert prefer_demucs_native_from_material("crowd") is True
    assert prefer_demucs_native_from_material("cd_digital") is False


def test_router_preflight_skips_roformer_on_low_ram(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _VM:
        available = int(4.2 * 1024**3)

    class _FakeMdx:
        @staticmethod
        def separate_all_stems(audio: np.ndarray, sr: int, stems: list[str]):  # pylint: disable=unused-argument
            return {
                "vocals": np.full(48000, 0.08, dtype=np.float32),
                "inst": np.full(48000, 0.12, dtype=np.float32),
            }

    monkeypatch.setitem(__import__("sys").modules, "psutil", types.SimpleNamespace(virtual_memory=lambda: _VM()))
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.bs_roformer_plugin",
        types.SimpleNamespace(
            get_bs_roformer=lambda: (_ for _ in ()).throw(AssertionError("roformer must be skipped"))
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.mdx23c_plugin",
        types.SimpleNamespace(get_mdx23c_plugin=lambda: _FakeMdx()),
    )

    result = SotaVocalModelRouter().separate_vocal_instrumental(_audio(), 48000, panns_singing=0.8)
    assert result.success is True
    assert result.model_used == "mdx23c"
    assert any(x.startswith("bs_roformer:preflight_low_ram_4.2GB_req_") for x in result.fallback_chain)


def test_router_preflight_skips_demucs_on_low_ram(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeFallbackBsResult:
        stems = {"vocals": np.full(48000, 0.10, dtype=np.float32), "other": np.full(48000, 0.20, dtype=np.float32)}
        model_used = "nmf_dsp_fallback"
        confidence = 0.20
        sdri_db = 0.0

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeFallbackBsResult()

    class _VM:
        available = int(4.5 * 1024**3)

    class _FakeMdx:
        @staticmethod
        def separate_all_stems(audio: np.ndarray, sr: int, stems: list[str]):  # pylint: disable=unused-argument
            return {
                "vocals": np.full(48000, 0.08, dtype=np.float32),
                "inst": np.full(48000, 0.12, dtype=np.float32),
            }

    monkeypatch.setitem(__import__("sys").modules, "psutil", types.SimpleNamespace(virtual_memory=lambda: _VM()))
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.bs_roformer_plugin",
        types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.demucs_v4_plugin",
        types.SimpleNamespace(
            get_demucs_plugin=lambda: (_ for _ in ()).throw(AssertionError("demucs must be skipped"))
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.mdx23c_plugin",
        types.SimpleNamespace(get_mdx23c_plugin=lambda: _FakeMdx()),
    )

    result = SotaVocalModelRouter().separate_vocal_instrumental(
        _audio(),
        48000,
        panns_singing=0.8,
        ctx={"material_type": "live", "score_routing_enabled": False},
    )
    assert result.success is True
    assert result.model_used == "mdx23c"
    assert any(x.startswith("demucs_v4:preflight_low_ram_4.5GB_req_") for x in result.fallback_chain)


def test_router_required_memory_grows_with_duration_and_channels():
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    sr = 48000
    mono_short = np.zeros(sr * 10, dtype=np.float32)
    stereo_long = np.zeros((sr * 180, 2), dtype=np.float32)

    demucs_short = SotaVocalModelRouter._required_memory_gb("demucs_v4", mono_short, sr)  # pylint: disable=protected-access
    demucs_long = SotaVocalModelRouter._required_memory_gb("demucs_v4", stereo_long, sr)  # pylint: disable=protected-access
    roformer_short = SotaVocalModelRouter._required_memory_gb("bs_roformer", mono_short, sr)  # pylint: disable=protected-access

    assert demucs_long > demucs_short
    assert roformer_short > demucs_short


def test_router_required_memory_grows_with_pressure_factor():
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    sr = 48000
    mono = np.zeros(sr * 30, dtype=np.float32)
    base = SotaVocalModelRouter._required_memory_gb("mdx23c", mono, sr, pressure_factor=1.0)  # pylint: disable=protected-access
    pressured = SotaVocalModelRouter._required_memory_gb("mdx23c", mono, sr, pressure_factor=1.5)  # pylint: disable=protected-access
    assert pressured > base


def test_router_runtime_pressure_multiplier_uses_active_ml_plugins(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    router = SotaVocalModelRouter()

    class _VM:
        percent = 20.0

    class _SWAP:
        percent = 0.0

    monkeypatch.setitem(
        __import__("sys").modules,
        "psutil",
        types.SimpleNamespace(virtual_memory=lambda: _VM(), swap_memory=lambda: _SWAP()),
    )

    no_load = router._runtime_pressure_multiplier({"active_ml_plugins": 0})  # pylint: disable=protected-access
    high_load = router._runtime_pressure_multiplier({"active_ml_plugins": 4})  # pylint: disable=protected-access
    assert high_load > no_load


def test_router_prefers_bs_roformer_for_vocal_material(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeBsResult:
        stems = {
            "vocals": np.full(48000, 0.10, dtype=np.float32),
            "drums": np.full(48000, 0.20, dtype=np.float32),
            "bass": np.full(48000, 0.30, dtype=np.float32),
        }
        model_used = "bs_roformer"
        confidence = 0.88
        sdri_db = 8.5

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeBsResult()

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.bs_roformer_plugin",
        types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs()),
    )
    # RAM-Preflight mocken: genug RAM vortäuschen, damit BS-RoFormer nicht übersprungen wird
    monkeypatch.setattr(SotaVocalModelRouter, "_available_memory_gb", staticmethod(lambda: 32.0))

    result = SotaVocalModelRouter().separate_vocal_instrumental(
        _audio(),
        48000,
        panns_singing=0.8,
        ctx={"material_type": "live"},
    )
    assert result.success is True
    assert result.model_used == "bs_roformer"
    np.testing.assert_allclose(result.vocal, np.full(48000, 0.10, dtype=np.float32))
    np.testing.assert_allclose(result.instrumental, np.full(48000, 0.50, dtype=np.float32))
    assert result.metadata["confidence"] == 0.88
    assert "capability_status" in result.metadata


def test_router_skips_roformer_fallback_and_uses_demucs(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeFallbackBsResult:
        stems = {"vocals": np.full(48000, 0.10, dtype=np.float32), "other": np.full(48000, 0.20, dtype=np.float32)}
        model_used = "nmf_dsp_fallback"
        confidence = 0.20
        sdri_db = 0.0

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeFallbackBsResult()

    class _FakeDemucs:
        _session = object()

        @staticmethod
        def separate(
            audio: np.ndarray,
            sr: int,
            prefer_mdx23c: bool = True,
        ) -> dict[str, np.ndarray]:  # pylint: disable=unused-argument
            assert prefer_mdx23c is False
            return {
                "vocals": np.full(48000, 0.30, dtype=np.float32),
                "other": np.full(48000, 0.40, dtype=np.float32),
            }

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs())
    )
    monkeypatch.setitem(
        sys_modules, "plugins.demucs_v4_plugin", types.SimpleNamespace(get_demucs_plugin=lambda: _FakeDemucs())
    )
    monkeypatch.setitem(
        sys_modules,
        "plugins.mdx23c_plugin",
        types.SimpleNamespace(get_mdx23c_plugin=lambda: (_ for _ in ()).throw(RuntimeError("mdx unavailable"))),
    )
    # RAM-Preflight mocken: genug RAM vortäuschen, damit BS-RoFormer ausgeführt wird (nmf_dsp_fallback erwartet)
    monkeypatch.setattr(SotaVocalModelRouter, "_available_memory_gb", staticmethod(lambda: 32.0))

    result = SotaVocalModelRouter().separate_vocal_instrumental(
        _audio(),
        48000,
        panns_singing=0.8,
        ctx={"material_type": "live"},
    )
    assert result.success is True
    assert result.model_used == "demucs_v4_htdemucs"
    assert "bs_roformer:nmf_dsp_fallback" in result.fallback_chain
    assert "capability_status" in result.metadata
    np.testing.assert_allclose(result.vocal, np.full(48000, 0.30, dtype=np.float32))


def test_router_prefers_mdx23c_when_not_live_context(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeFallbackBsResult:
        stems = {"vocals": np.full(48000, 0.10, dtype=np.float32), "other": np.full(48000, 0.20, dtype=np.float32)}
        model_used = "nmf_dsp_fallback"
        confidence = 0.20
        sdri_db = 0.0

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeFallbackBsResult()

    class _FakeDemucs:
        _session = object()

        @staticmethod
        def separate(
            audio: np.ndarray,
            sr: int,
            prefer_mdx23c: bool = True,
        ) -> dict[str, np.ndarray]:  # pylint: disable=unused-argument
            raise AssertionError("demucs should not run without live/crowd context")

    class _FakeMdx:
        @staticmethod
        def separate_all_stems(audio: np.ndarray, sr: int, stems: list[str]):  # pylint: disable=unused-argument
            return {
                "vocals": np.full(48000, 0.08, dtype=np.float32),
                "inst": np.full(48000, 0.12, dtype=np.float32),
            }

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs())
    )
    monkeypatch.setitem(
        sys_modules, "plugins.demucs_v4_plugin", types.SimpleNamespace(get_demucs_plugin=lambda: _FakeDemucs())
    )
    monkeypatch.setitem(
        sys_modules, "plugins.mdx23c_plugin", types.SimpleNamespace(get_mdx23c_plugin=lambda: _FakeMdx())
    )

    result = SotaVocalModelRouter().separate_vocal_instrumental(_audio(), 48000, panns_singing=0.8)
    assert result.success is True
    assert result.model_used == "mdx23c"
    np.testing.assert_allclose(result.vocal, np.full(48000, 0.08, dtype=np.float32))


def test_router_score_routing_selects_better_candidate(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeFallbackBsResult:
        stems = {"vocals": np.full(48000, 0.10, dtype=np.float32), "other": np.full(48000, 0.20, dtype=np.float32)}
        model_used = "nmf_dsp_fallback"
        confidence = 0.20
        sdri_db = 0.0

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeFallbackBsResult()

    class _FakeDemucs:
        _session = object()

        @staticmethod
        def separate(audio: np.ndarray, sr: int, prefer_mdx23c: bool = True):  # pylint: disable=unused-argument
            return {
                "vocals": np.full(48000, 0.15, dtype=np.float32),
                "other": np.full(48000, 0.40, dtype=np.float32),
            }

    class _FakeMdx:
        @staticmethod
        def separate_all_stems(audio: np.ndarray, sr: int, stems: list[str]):  # pylint: disable=unused-argument
            return {
                "vocals": np.full(48000, 0.08, dtype=np.float32),
                "inst": np.full(48000, 0.12, dtype=np.float32),
            }

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs())
    )
    monkeypatch.setitem(
        sys_modules, "plugins.demucs_v4_plugin", types.SimpleNamespace(get_demucs_plugin=lambda: _FakeDemucs())
    )
    monkeypatch.setitem(
        sys_modules, "plugins.mdx23c_plugin", types.SimpleNamespace(get_mdx23c_plugin=lambda: _FakeMdx())
    )

    result = SotaVocalModelRouter().separate_vocal_instrumental(
        _audio(),
        48000,
        panns_singing=0.8,
        ctx={"material_type": "live", "score_routing_enabled": True},
    )
    assert result.success is True
    assert result.model_used == "mdx23c"
    assert float(result.metadata.get("route_score", 0.0)) > 0.0  # type: ignore[arg-type]
    assert result.metadata.get("route_score_selected") is True


def test_demucs_live_policy_contract_router_phase42():
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter
    from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

    router = SotaVocalModelRouter()
    phase42 = VocalEnhancement()

    live_like = [
        "live",
        "concert",
        "audience",
        "crowd",
        "bootleg",
        "stage",
        "live_recording",
        "crowd_tape",
    ]
    non_live = ["vinyl", "tape", "cd_digital", "mp3", "studio"]

    for tag in live_like:
        assert router._prefer_demucs_native({"material_type": tag}) is True  # pylint: disable=protected-access
        assert phase42._prefer_demucs_native(tag) is True  # pylint: disable=protected-access

    for tag in non_live:
        assert router._prefer_demucs_native({"material_type": tag}) is False  # pylint: disable=protected-access
        assert phase42._prefer_demucs_native(tag) is False  # pylint: disable=protected-access


def test_demucs_native_call_contract_router_phase42(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter
    from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

    calls: dict[str, list[bool]] = {"router": [], "phase42": []}

    class _FakeFallbackBsResult:
        stems = {"vocals": np.full(48000, 0.10, dtype=np.float32), "other": np.full(48000, 0.20, dtype=np.float32)}
        model_used = "nmf_dsp_fallback"
        confidence = 0.20
        sdri_db = 0.0

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeFallbackBsResult()

    class _FakeDemucs:
        _session = object()

        @staticmethod
        def separate(audio: np.ndarray, sr: int, prefer_mdx23c: bool = True):  # pylint: disable=unused-argument
            calls["router"].append(bool(prefer_mdx23c))
            return {
                "vocals": np.full(48000, 0.15, dtype=np.float32),
                "other": np.full(48000, 0.40, dtype=np.float32),
            }

        @staticmethod
        def separate_vocals(audio: np.ndarray, sr: int, prefer_mdx23c: bool = True):  # pylint: disable=unused-argument
            calls["phase42"].append(bool(prefer_mdx23c))
            a = np.asarray(audio, dtype=np.float32)
            return a * 0.55, a * 0.45

    class _VM:
        available = 6 * 1024**3

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(
        sys_modules,
        "plugins.bs_roformer_plugin",
        types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs()),
    )
    monkeypatch.setitem(
        sys_modules,
        "plugins.demucs_v4_plugin",
        types.SimpleNamespace(get_demucs_plugin=lambda: _FakeDemucs()),
    )
    monkeypatch.setitem(
        sys_modules,
        "plugins.mdx23c_plugin",
        types.SimpleNamespace(get_mdx23c_plugin=lambda: (_ for _ in ()).throw(RuntimeError("mdx unavailable"))),
    )

    router_result = SotaVocalModelRouter().separate_vocal_instrumental(
        _audio(),
        48000,
        panns_singing=0.8,
        ctx={"material_type": "live", "score_routing_enabled": False},
    )
    assert router_result.success is True
    assert router_result.model_used == "demucs_v4_htdemucs"

    import backend.core.phases.phase_42_vocal_enhancement as mod42

    monkeypatch.setattr(
        mod42.psutil if hasattr(mod42, "psutil") else __import__("psutil"),
        "virtual_memory",
        lambda: _VM(),
    )
    phase42_result = VocalEnhancement()._try_stem_separation(  # pylint: disable=protected-access
        np.tile(np.array([[0.1, 0.05]], dtype=np.float32), (48000, 1)),
        48000,
        material="live",  # type: ignore[arg-type]
    )
    assert phase42_result is not None
    assert phase42_result[3] == "demucs_v4_htdemucs"

    assert calls["router"] == [False]
    assert calls["phase42"] == [False]


def test_router_vocal_nr_skips_unloaded_miipher_for_sgmse(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeMiipher:
        _model_loaded = False

        @staticmethod
        def enhance(audio: np.ndarray, sr: int, noise_snr_db: float = 0.0) -> np.ndarray:  # pylint: disable=unused-argument
            raise AssertionError("MIIPHER stub must not run when no model is loaded")

    class _FakeSgmseResult:
        def __init__(self, audio: np.ndarray) -> None:
            self.audio = (audio * 0.5).astype(np.float32)
            self.model_used = "sgmse_plus_torchscript"

    class _FakeSgmse:
        _model_loaded = True

        @staticmethod
        def enhance(audio: np.ndarray, sr: int):  # pylint: disable=unused-argument
            return _FakeSgmseResult(audio)

    # §v10.14: MIIPHER-DiT als Primärmodell mocken (nicht verfügbar → Fallback zu SGMSE+)
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_dit_plugin",
        types.SimpleNamespace(get_miipher_dit=lambda: (_ for _ in ()).throw(ImportError("miipher_dit not available"))),
    )

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_plugin",
        types.SimpleNamespace(get_miipher_plugin=lambda: _FakeMiipher()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.sgmse_plugin",
        types.SimpleNamespace(get_sgmse_plugin=lambda: _FakeSgmse()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.deepfilternet_v3_ii_plugin",
        types.SimpleNamespace(get_deepfilternet_plugin=lambda: (_ for _ in ()).throw(RuntimeError("no dfn"))),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_vocal(audio, 48000, energy_bias_db=-6.0)
    assert result.success is True
    # MIIPHER-DiT nicht verfügbar → Fallback zu MIIPHER (nicht geladen) → SGMSE+
    assert "miipher_dit:import_error" in result.fallback_chain
    assert result.model_used == "sgmse_plus_torchscript"
    assert "miipher:not_loaded" in result.fallback_chain
    assert result.metadata["miipher_model_loaded"] is False
    assert result.metadata["sgmse_model_loaded"] is True
    assert result.metadata["miipher_compensation_active"] is True
    assert result.metadata["miipher_compensation_dfn_applied"] is False
    assert "capability_status" in result.metadata
    np.testing.assert_allclose(result.audio, audio * 0.5, atol=1e-7)


def test_router_vocal_nr_compensates_missing_miipher_with_dfn_and_hnr(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeMiipher:
        _model_loaded = False

    class _FakeSgmseResult:
        def __init__(self, audio: np.ndarray) -> None:
            self.audio = (audio * 0.5).astype(np.float32)
            self.model_used = "sgmse_plus_torchscript"

    class _FakeSgmse:
        _model_loaded = True

        @staticmethod
        def enhance(audio: np.ndarray, sr: int):  # pylint: disable=unused-argument
            return _FakeSgmseResult(audio)

    class _FakeDfn:
        @staticmethod
        def enhance(audio: np.ndarray, sr: int, energy_bias_db: float = -9.0):  # pylint: disable=unused-argument
            return (audio * 0.8).astype(np.float32)

    # §v10.14: MIIPHER-DiT als Primärmodell mocken (nicht verfügbar → Fallback zu SGMSE+ + DFN)
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_dit_plugin",
        types.SimpleNamespace(get_miipher_dit=lambda: (_ for _ in ()).throw(ImportError("miipher_dit not available"))),
    )

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_plugin",
        types.SimpleNamespace(get_miipher_plugin=lambda: _FakeMiipher()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.sgmse_plugin",
        types.SimpleNamespace(get_sgmse_plugin=lambda: _FakeSgmse()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.deepfilternet_v3_ii_plugin",
        types.SimpleNamespace(get_deepfilternet_plugin=lambda: _FakeDfn()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "backend.core.dsp.hnr_guard",
        types.SimpleNamespace(apply_hnr_blend=lambda pre, post, sr: (post * 0.9).astype(np.float32)),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_vocal(audio, 48000, energy_bias_db=-6.0)
    assert result.success is True
    # MIIPHER-DiT nicht verfügbar → Fallback zu MIIPHER (nicht geladen) → SGMSE+ + DFN + HNR
    assert "miipher_dit:import_error" in result.fallback_chain
    assert result.model_used.endswith("+deepfilternet_v3_ii+hnr_blend")
    assert result.model_used.startswith("sgmse_plus")
    assert result.metadata["miipher_compensation_active"] is True
    assert result.metadata["miipher_compensation_dfn_applied"] is True
    assert result.metadata["miipher_compensation_hnr_applied"] is True
    np.testing.assert_allclose(result.audio, audio * 0.36, atol=1e-7)


def test_router_accepts_productive_miipher_adapter_without_native_onnx(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeMiipherAdapter:
        _model_loaded = False
        route_metadata = {
            "model_used": "miipher_sgmse_plus",
            "capability_status": "sota_fallback",
            "native_miipher_loaded": False,
        }

        @staticmethod
        def is_productive() -> bool:
            return True

        @staticmethod
        def enhance(audio: np.ndarray, sr: int, noise_snr_db: float = 0.0) -> np.ndarray:  # pylint: disable=unused-argument
            return (audio * 0.25).astype(np.float32)

    # §v10.14: MIIPHER-DiT als Primärmodell mocken (nicht verfügbar → Fallback zu MIIPHER Adapter)
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_dit_plugin",
        types.SimpleNamespace(get_miipher_dit=lambda: (_ for _ in ()).throw(ImportError("miipher_dit not available"))),
    )

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_plugin",
        types.SimpleNamespace(get_miipher_plugin=lambda: _FakeMiipherAdapter()),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_vocal(audio, 48000, energy_bias_db=-6.0, noise_snr_db=4.0)
    assert result.success is True
    # MIIPHER-DiT nicht verfügbar → Fallback zu MIIPHER Adapter (productive)
    assert "miipher_dit:import_error" in result.fallback_chain
    assert result.model_used == "miipher_sgmse_plus"
    assert result.metadata["miipher_model_loaded"] is False
    assert result.metadata["miipher_adapter_productive"] is True
    assert result.metadata["capability_status"] == "sota_fallback"
    np.testing.assert_allclose(result.audio, audio * 0.25, atol=1e-7)


def test_router_instrumental_nr_falls_back_cleanly(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.deepfilternet_v3_ii_plugin",
        types.SimpleNamespace(get_deepfilternet_plugin=lambda: (_ for _ in ()).throw(RuntimeError("no model"))),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_instrumental(audio, 48000)
    assert result.success is False
    assert result.model_used == "none"
    assert result.fallback_chain
    assert result.metadata["capability_status"] == "dsp_fallback"
    np.testing.assert_allclose(result.audio, audio, atol=1e-7)
