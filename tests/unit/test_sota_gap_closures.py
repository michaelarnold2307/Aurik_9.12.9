"""Regressionstests: SOTA-Lückenschluss Rev. 2026-08-16.

Deckt die vier geschlossenen Lücken ab (siehe docs/guides/SOTA_MIGRATION_PLAN.md):
  1. RMVPE: ONNX-Session-Selbstheilung nach PLM-Unload (§V6 (copilot-instructions.md) — kein stiller DSP-Downgrade)
  2. phase_56: RMVPE-Stufe nutzt korrekte RmvpeResult-Attribute (f0/voiced_flag)
  3. vocal_harmonic_decomp: FCPE primär statt CREPE (Spec 04, Z. 1129: CREPE verboten)
  4. phase_66: Whisper-Denoiser nicht mehr in der NR-Fallback-Kette (deprecated)
  5. vocoder_chain: Spec-04-Kaskade (Vocos nur Studio-2026, §1.4-konform)
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

SR = 48000


def _sine(dur_s: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    _tone: np.ndarray = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return _tone


# ---------------------------------------------------------------------------
# 1. RMVPE: Session-Selbstheilung
# ---------------------------------------------------------------------------


class TestRmvpeSessionSelfHealing:
    def test_local_session_reference_survives_eviction(self) -> None:
        """Race-Fund Rev. 2026-08-16: Inferenz nutzt die lokale Session-Referenz —
        ein PLM-Evict (self._session = None) mitten im Call kann nichts mehr annullieren."""
        from plugins.rmvpe_plugin import RmvpePlugin

        p = RmvpePlugin.__new__(RmvpePlugin)  # kein __init__: kein ONNX-Load nötig
        p._session = None
        p._model_loaded = False

        class _StubSession:
            runs = 0

            def get_inputs(self):
                return [types.SimpleNamespace(name="mel", shape=[1, 128, -1])]

            def run(self, _outputs, feed):
                type(self).runs += 1
                t = feed["mel"].shape[2]
                return [np.zeros((1, t, 360), dtype=np.float32)]

        stub = _StubSession()
        result = p._analyze_onnx(np.zeros(48000, dtype=np.float32), SR, 0.5, session=stub)
        assert _StubSession.runs == 1
        assert result.model_used == "rmvpe_onnx"

    def test_reload_after_simulated_plm_unload(self) -> None:
        pytest.importorskip("onnxruntime")
        from backend.core.ml_memory_budget import release
        from plugins.rmvpe_plugin import RmvpePlugin

        release("RMVPE")
        p = RmvpePlugin()
        try:
            if not p._model_loaded:  # pragma: no cover — Umgebungsvoraussetzung
                pytest.skip("RMVPE ONNX konnte nicht geladen werden (Budget/Umgebung)")
            audio = _sine(0.5, freq=300.0)
            # PLM-Unload simulieren (unload_fn setzt genau diese beiden Felder)
            p._session = None
            p._model_loaded = False
            result = p.analyze(audio, SR)
            assert result.model_used == "rmvpe_onnx", (
                "Selbstheilung fehlgeschlagen: nach Session-Unload degradiert "
                f"der §4.4-Tracker still auf {result.model_used} (§V6 (copilot-instructions.md))."
            )
        finally:
            release("RMVPE")


# ---------------------------------------------------------------------------
# 2. phase_56: RMVPE-Tier nutzt korrekte RmvpeResult-API
# ---------------------------------------------------------------------------


class TestPhase56RmvpeTier:
    def test_rmvpe_tier_returns_median_f0(self, monkeypatch) -> None:
        from plugins.rmvpe_plugin import RmvpeResult

        class _FakeRmvpe:
            def analyze(self, mono: np.ndarray, sr: int) -> RmvpeResult:
                n = 30
                return RmvpeResult(
                    f0=np.full(n, 110.0, dtype=np.float32),
                    times=np.arange(n, dtype=np.float32) * 0.01,
                    confidence=np.ones(n, dtype=np.float32),
                    voiced_flag=np.ones(n, dtype=bool),
                    model_used="rmvpe_onnx",
                )

        def _fcpe_raise(*_a, **_k):
            raise RuntimeError("FCPE deaktiviert (Test: RMVPE-Tier erzwingen)")

        monkeypatch.setitem(sys.modules, "plugins.fcpe_plugin", types.SimpleNamespace(get_fcpe_plugin=_fcpe_raise))
        monkeypatch.setitem(
            sys.modules,
            "plugins.rmvpe_plugin",
            types.SimpleNamespace(get_rmvpe_plugin=lambda: _FakeRmvpe()),
        )

        from backend.core.phases.phase_56_spectral_band_gap_repair import _estimate_f0

        f0 = _estimate_f0(_sine(0.5, freq=220.0), SR)
        assert f0 is not None, "RMVPE-Tier lieferte None (Stufe fiel durch wie vor Rev. 2026-08-16)"
        assert abs(f0 - 110.0) < 1.0, f"Median f0 = {f0}, erwartet ~110 Hz (Fake-RMVPE)"


# ---------------------------------------------------------------------------
# 3. vocal_harmonic_decomp: FCPE primär
# ---------------------------------------------------------------------------


class TestVhmFcpeRouting:
    def test_mask_uses_fcpe_f0(self, monkeypatch) -> None:
        class _FakeFcpe:
            def analyze(self, audio: np.ndarray, sr: int):
                n = 100
                return types.SimpleNamespace(
                    f0_hz=np.full(n, 440.0, dtype=np.float32),
                    voiced_prob=np.ones(n, dtype=np.float32),
                )

        monkeypatch.setitem(
            sys.modules,
            "plugins.fcpe_plugin",
            types.SimpleNamespace(get_fcpe_plugin=lambda: _FakeFcpe()),
        )

        from backend.core.dsp.vocal_harmonic_decomp import build_vocal_harmonic_mask

        mask = build_vocal_harmonic_mask(_sine(1.0, freq=440.0), SR)
        assert mask is not None, "build_vocal_harmonic_mask gab None zurück (FCPE-Pfad defekt)"
        assert mask.voiced_fraction > 0.9
        voiced_f0 = mask.f0_contour[mask.f0_contour > 20.0]
        assert float(np.median(voiced_f0)) == pytest.approx(440.0, rel=0.01)


# ---------------------------------------------------------------------------
# 4. phase_66: kein Whisper-Denoiser in der Kette
# ---------------------------------------------------------------------------


class TestPhase66NoWhisperFallback:
    def test_get_dfn_does_not_touch_whisper(self, monkeypatch) -> None:
        class _WhisperSentinel:
            """Jeder Zugriff = Testfehler: Whisper darf nicht mehr importiert werden."""

            def __getattr__(self, name: str):
                raise AssertionError(f"Whisper-Denoiser wurde trotz Deprecation angefasst: {name}")

        monkeypatch.setitem(
            sys.modules,
            "plugins.deepfilternet_v3_ii_plugin",
            types.SimpleNamespace(DeepFilterNetV3IIPlugin=_WhisperSentinel.__init__),
        )
        monkeypatch.setitem(sys.modules, "plugins.whisper_denoiser_plugin", _WhisperSentinel())

        from backend.core.phases.phase_66_stem_targeted_nr import StemTargetedNRPhase

        dfn = StemTargetedNRPhase._get_dfn()
        assert dfn is None, "DFN nicht verfügbar → erwartet None (OMLSA-DSP-Fallback), kein Whisper"


# ---------------------------------------------------------------------------
# 5. vocoder_chain: Spec-04-Kaskade, Vocos nur Studio-2026
# ---------------------------------------------------------------------------


class TestVocoderChainSpecCascade:
    def test_studio_mode_prefers_vocos(self, monkeypatch) -> None:
        calls: list[str] = []

        class _FakeVocos:
            def vocode(self, audio: np.ndarray, sr: int, mode: str):
                calls.append(mode)
                return types.SimpleNamespace(audio=(audio * 0.5).astype(np.float32))

        monkeypatch.setitem(
            sys.modules,
            "plugins.vocos_plugin",
            types.SimpleNamespace(get_vocos_plugin=lambda: _FakeVocos()),
        )
        monkeypatch.setitem(sys.modules, "plugins.bigvgan_v2_plugin", _ImportErrorModule())
        monkeypatch.setitem(sys.modules, "plugins.hifigan_plugin", _ImportErrorModule())

        from backend.core.vocoder_chain import activate_vocoder_chain

        audio = _sine(0.2)
        out = activate_vocoder_chain(audio, SR, pqs_mos=3.0, studio_mode=True)
        assert calls == ["studio2026"], f"Vocos wurde nicht als Tier-1 gerufen: {calls}"
        assert out is not None and np.allclose(out, audio * 0.5, atol=1e-6)

    def test_restoration_mode_skips_vocos(self, monkeypatch) -> None:
        class _FakeBigVGAN:
            def synthesize(self, audio: np.ndarray, sr: int):
                return (audio * 0.75).astype(np.float32)

        def _vocos_violation(*_a, **_k):
            raise AssertionError("Vocos darf im Restoration-Modus nicht gerufen werden (§1.4)")

        monkeypatch.setitem(
            sys.modules,
            "plugins.vocos_plugin",
            types.SimpleNamespace(get_vocos_plugin=_vocos_violation),
        )
        monkeypatch.setitem(
            sys.modules,
            "plugins.bigvgan_v2_plugin",
            types.SimpleNamespace(BigVGANv2Plugin=lambda: _FakeBigVGAN()),
        )

        from backend.core.vocoder_chain import activate_vocoder_chain

        audio = _sine(0.2)
        out = activate_vocoder_chain(audio, SR, pqs_mos=3.0)
        assert out is not None and np.allclose(out, audio * 0.75, atol=1e-6)


class _ImportErrorModule:
    """Modul-Stub, der bei jedem Import-Attributzugriff ImportError wirft."""

    def __getattr__(self, name: str):
        raise ImportError(f"{name} nicht verfügbar (Test-Stub)")
