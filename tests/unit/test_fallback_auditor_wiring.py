"""Regressionstests: FallbackAuditor-Verdrahtung (§v10.17, Rev. 2026-08-16).

Deckt ab:
  1. FallbackAuditor.reset() (Song-Scope, §V8 (copilot-instructions.md)/§G1 (GEBOTE.md)) + Kaskaden-Block
  2. Alle sechs verdrahteten Sites registrieren Events im zentralen Auditor
  3. Denker-Reset und PreExport-Report sind verdrahtet (Quelltext-Vertrag)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

SR = 48000
_ROOT = Path(__file__).resolve().parents[2]


def _sine(dur_s: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    _tone: np.ndarray = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return _tone


class _ImportErrorModule:
    def __getattr__(self, name: str):
        raise ImportError(f"{name} nicht verfügbar (Test-Stub)")


def _fresh_auditor():
    from backend.core.fallback_auditor import get_fallback_auditor

    fa = get_fallback_auditor()
    fa.reset()
    return fa


# ---------------------------------------------------------------------------
# 1. Auditor-Kern
# ---------------------------------------------------------------------------


class TestFallbackAuditorCore:
    def test_reset_clears_events(self):
        fa = _fresh_auditor()
        fa.record("X", "gold", "fb", "reason")
        assert fa.degraded
        fa.reset()
        assert not fa.degraded
        assert fa.summary()["total_fallbacks"] == 0

    def test_cascade_block_after_eight_events(self):
        fa = _fresh_auditor()
        for i in range(8):
            fa.record(f"component_{i}", "gold", "fb", "reason")
        assert fa.should_block_pipeline, "Ab 8 Events muss der PreExportValidator blockieren (§v10.17)"

    def test_report_contains_component(self):
        fa = _fresh_auditor()
        fa.record("FeedbackChain", "versa_mos", "pqs_rms_dsp", "versa_load_failed")
        assert "FeedbackChain" in fa.report()
        assert "versa_mos" in fa.report()


# ---------------------------------------------------------------------------
# 2. Site-Verdrahtungen
# ---------------------------------------------------------------------------


class TestFallbackAuditorWiring:
    def test_aurik_restore_df_n_fallback_records(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules,
            "plugins.deepfilternet_v3_ii_plugin",
            types.SimpleNamespace(enhance_audio=_ImportErrorModule()),
        )
        import backend.aurik_restore as ar

        fa = _fresh_auditor()
        ar.restaurierung(_sine(), SR)
        events = fa.summary()["events"]
        assert any(e["component"] == "aurik_restore" and e["fallback"] == "dry_passthrough" for e in events), (
            f"aurik_restore-DFN-Fallback nicht registriert: {events}"
        )

    def test_stem_level_restorer_router_decline_records(self, monkeypatch):
        class _DecliningRouter:
            def separate_vocal_instrumental(self, audio, sample_rate, panns_singing=0.0, ctx=None):
                return types.SimpleNamespace(success=False, fallback_chain=["demucs_v4"])

        monkeypatch.setitem(
            sys.modules,
            "backend.core.dsp.sota_vocal_model_router",
            types.SimpleNamespace(get_sota_vocal_model_router=lambda: _DecliningRouter()),
        )
        from backend.core.dsp.stem_level_restorer import StemLevelRestorer

        fa = _fresh_auditor()
        StemLevelRestorer()._separate_stems(_sine(), SR)
        events = fa.summary()["events"]
        assert any(e["component"] == "StemLevelRestorer" and e["fallback"] == "dsp_bandpass" for e in events), (
            f"SLR-Router-Ablehnung nicht registriert: {events}"
        )

    def test_feedback_chain_versa_load_failure_records(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "plugins.versa_plugin", _ImportErrorModule())
        from backend.core.feedback_chain import FeedbackChain

        fa = _fresh_auditor()
        FeedbackChain(use_versa_in_loop=True)
        events = fa.summary()["events"]
        assert any(e["component"] == "FeedbackChain" and e["fallback"] == "pqs_rms_dsp" for e in events), (
            f"VERSA-Ladefehler nicht registriert: {events}"
        )

    @pytest.mark.parametrize(
        "module_path, class_name",
        [
            ("backend.core.hybrid.hybrid_wow_flutter", "HybridWowFlutter"),
            ("backend.core.hybrid.hybrid_speed_pitch_ml", "HybridSpeedPitch"),
        ],
    )
    def test_hybrid_pitch_cascade_records_pyin_fallback(self, monkeypatch, module_path, class_name):
        for mod in ("plugins.fcpe_plugin", "plugins.rmvpe_plugin", "plugins.pesto_plugin"):
            monkeypatch.setitem(sys.modules, mod, _ImportErrorModule())
        import importlib

        _mod = importlib.import_module(module_path)
        cls = getattr(_mod, class_name)

        fa = _fresh_auditor()
        inst = cls()
        inst._init_crepe()
        events = fa.summary()["events"]
        assert any(e["component"] == "PitchDetection" and e["fallback"] == "pyin_dsp" for e in events), (
            f"{class_name}: pYIN-DSP-Endfall nicht registriert: {events}"
        )


# ---------------------------------------------------------------------------
# 3. Denker-Reset + PreExport-Report (Quelltext-Vertrag)
# ---------------------------------------------------------------------------


class TestAuditorSongScopeContract:
    def test_denker_resets_auditor_per_song(self):
        src = (_ROOT / "denker" / "aurik_denker.py").read_text(encoding="utf-8")
        assert "get_fallback_auditor().reset()" in src, (
            "Denker muss den FallbackAuditor pro Song zurücksetzen (§V8 (copilot-instructions.md)/§G1 (GEBOTE.md), Rev. 2026-08-16)"
        )

    def test_pre_export_validator_reports_degraded(self):
        src = (_ROOT / "backend" / "core" / "pre_export_validator.py").read_text(encoding="utf-8")
        assert "fa.report()" in src, "PreExportValidator muss den konsolidierten Bericht ausgeben"
        assert "should_block_pipeline" in src
