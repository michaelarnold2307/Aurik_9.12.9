"""Regressionstests: aurik_restore UV3-Angleich + stille-Downgrade-Fixes (Rev. 2026-08-16).

Deckt ab:
  1. aurik_restore: ML→DSP-Fallbacks warnen (§V6) und Dry-Passthrough nie still
  2. Hybrid-Pitch-Kaskaden: CREPE als Produktions-Tier entfernt (Spec 04, Z. 1129)
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SR = 48000
_ROOT = Path(__file__).resolve().parents[2]


def _sine(dur_s: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _raising(*_a, **_k):
    raise RuntimeError("Test-Stub: Plugin nicht verfügbar")


class _ImportErrorModule:
    def __getattr__(self, name: str):
        raise ImportError(f"{name} nicht verfügbar (Test-Stub)")


# ---------------------------------------------------------------------------
# 1. aurik_restore: §V6-Dry-Passthrough
# ---------------------------------------------------------------------------


class TestAurikRestoreUv3Alignment:
    def test_restaurierung_warns_and_passes_dry_when_dfn_unavailable(self, monkeypatch, caplog):
        monkeypatch.setitem(
            sys.modules,
            "plugins.deepfilternet_v3_ii_plugin",
            types.SimpleNamespace(enhance_audio=_raising),
        )
        import backend.aurik_restore as ar

        audio = _sine()
        with caplog.at_level(logging.WARNING, logger="backend.aurik_restore"):
            out, sr = ar.restaurierung(audio, SR)
        assert sr == SR
        assert out.shape == audio.shape
        assert np.allclose(out, audio, atol=1e-6), "Dry-Passthrough muss das Eingangssignal unverändert lassen"
        assert "Dry-Passthrough (§V6)" in caplog.text, f"§V6-Warnung fehlt: {caplog.text}"

    def test_reparatur_warns_and_passes_dry_on_none(self, monkeypatch, caplog):
        class _FakeDiffwave:
            def inpaint(self, audio: np.ndarray, sr: int, mask=None):
                return None

        monkeypatch.setitem(
            sys.modules,
            "plugins.diffwave_plugin",
            types.SimpleNamespace(DiffwavePlugin=lambda: _FakeDiffwave()),
        )
        import backend.aurik_restore as ar

        audio = _sine()
        with caplog.at_level(logging.WARNING, logger="backend.aurik_restore"):
            out, sr = ar.reparatur(audio, SR)
        assert sr == SR
        assert np.allclose(out, audio, atol=1e-6)
        assert "DiffWave lieferte kein Ergebnis — Dry-Passthrough (§V6)" in caplog.text

    def test_rekonstruktion_warns_and_passes_dry_on_none(self, monkeypatch, caplog):
        class _FakeMdx:
            def process(self, audio: np.ndarray, sr: int, stem: str = "vocals"):
                return None

        monkeypatch.setitem(
            sys.modules,
            "plugins.mdx23c_plugin",
            types.SimpleNamespace(MDX23CPlugin=lambda: _FakeMdx()),
        )
        import backend.aurik_restore as ar

        audio = _sine()
        with caplog.at_level(logging.WARNING, logger="backend.aurik_restore"):
            out, sr = ar.rekonstruktion(audio, SR)
        assert sr == SR
        assert np.allclose(out, audio, atol=1e-6)
        assert "keinen Vocal-Stem — Dry-Passthrough (§V6)" in caplog.text

    def test_quality_gates_fail_closed_without_utmos(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "plugins.utmos_plugin", _ImportErrorModule())
        import backend.aurik_restore as ar

        audio = _sine()
        with caplog.at_level(logging.WARNING, logger="backend.aurik_restore"):
            passed = ar.quality_gates(audio, SR)
        assert passed is False, "Ohne UTMOS muss das Quality-Gate fail-closed sein (§V6)"
        assert "UTMOS nicht verfügbar" in caplog.text

    def test_no_module_level_plugin_imports(self):
        """UV3-Angleich: ML-Plugins lazy laden (kein Import-seitiger Modell-Load)."""
        src = (_ROOT / "backend" / "aurik_restore.py").read_text(encoding="utf-8")
        import ast

        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("plugins."):
                pytest.fail(f"aurik_restore.py importiert ML-Plugin auf Modulebene: {node.module}")


# ---------------------------------------------------------------------------
# 2. Hybrid-Pitch-Kaskaden: CREPE-Tier entfernt (Spec 04, Z. 1129)
# ---------------------------------------------------------------------------


class TestHybridPitchCascadesNoCrepeTier:
    @pytest.mark.parametrize(
        "rel_path",
        [
            "backend/core/hybrid/hybrid_wow_flutter.py",
            "backend/core/hybrid/hybrid_speed_pitch_ml.py",
        ],
    )
    def test_no_crepe_production_tier(self, rel_path: str):
        src = (_ROOT / rel_path).read_text(encoding="utf-8")
        assert "CREPE plugin geladen" not in src, (
            f"{rel_path}: CREPE darf kein Produktions-Tier mehr sein (Spec 04, Z. 1129)"
        )
        assert "pYIN-DSP-Ersatzpfad (§V6)" in src, f"{rel_path}: §V6-Warnung für DSP-Endfall fehlt"
        assert "Spec 04, Z. 1129" in src, f"{rel_path}: Spec-Referenz fehlt"
