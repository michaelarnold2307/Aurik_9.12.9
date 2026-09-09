"""§SMR-1 Demucs-Stufe — SOTA-Aktivierungsvertrag (Fix 2026-09-08).

Befund: models/manifest.json (gitignored, lokaler Zustand) markierte
htdemucs_6s als experimental=True → DemucsV4Plugin lud die ONNX-Session
NIE → die Demucs-Stufe der Router-Kette (BS-RoFormer → Demucs v4 → MDX23C)
lief permanent auf HPSS-DSP-Fallback — still (§V6-Verstoß).

SOTA-Lösung (Root-Cause statt Symptom, §V7):
- Produktions-Modelle laden standardmäßig, sobald die Datei existiert.
- Expliziter Opt-out statt stillem Gate: AURIK_DISABLE_HTDEMUCS_6S=1.
- Kein Lese-Zugriff mehr auf das gitignored Manifest im Ladepfad.

Diese Tests sichern den Vertrag regressionsfest.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from plugins.demucs_v4_plugin import DemucsV4Plugin

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DEMUCS_ONNX = _REPO / "models" / "demucs" / "htdemucs_6s.onnx"
_BS_ROFORMER_CKPT = _REPO / "models" / "bs_roformer" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


@pytest.fixture(autouse=True)
def _clear_optout(monkeypatch):
    monkeypatch.delenv("AURIK_DISABLE_HTDEMUCS_6S", raising=False)


def _construct_plugin(monkeypatch, tmp_path, *, env_disable: bool = False) -> "DemucsV4Plugin":
    from plugins.demucs_v4_plugin import DemucsV4Plugin

    dummy = tmp_path / "htdemucs_6s.onnx"
    dummy.write_bytes(b"dummy")
    if env_disable:
        monkeypatch.setenv("AURIK_DISABLE_HTDEMUCS_6S", "1")
    monkeypatch.setattr("onnxruntime.InferenceSession", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "backend.core.ml_memory_budget.try_allocate",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "backend.core.ml_device_manager.get_ort_providers",
        MagicMock(return_value=["CPUExecutionProvider"]),
    )
    return DemucsV4Plugin(model_path=str(dummy))


@pytest.mark.unit
def test_demucs_loads_session_by_default(monkeypatch, tmp_path) -> None:
    """Produktions-Modell lädt standardmäßig — kein stilles Gate mehr."""
    p = _construct_plugin(monkeypatch, tmp_path)
    assert p._session is not None, "Demucs-Stufe darf nicht stumm deaktiviert sein (§V6)"


@pytest.mark.unit
def test_demucs_env_optout_disables(monkeypatch, tmp_path) -> None:
    """Expliziter Opt-out deaktiviert die Stufe laut und nachvollziehbar."""
    p = _construct_plugin(monkeypatch, tmp_path, env_disable=True)
    assert p._session is None


@pytest.mark.unit
def test_plugin_no_longer_reads_experimental_manifest() -> None:
    """Der Ladepfad darf nicht mehr vom gitignored Manifest abhängen (§V7)."""
    src = (_REPO / "plugins" / "demucs_v4_plugin.py").read_text(encoding="utf-8")
    assert "_manifest_path" not in src, "Manifest-Lesezugriff im Ladepfad"
    assert "AURIK_DISABLE_HTDEMUCS_6S" in src


@pytest.mark.unit
def test_demucs_model_file_present() -> None:
    """Das gebündelte Meta-htdemucs_6s-ONNX (inkl. externer Gewichte) ist vorhanden."""
    if not _DEMUCS_ONNX.exists():
        pytest.skip("models/-Paket nicht installiert (gitignored)")
    total = _DEMUCS_ONNX.stat().st_size
    for _suffix in (".dat", ".data"):
        _dat = pathlib.Path(str(_DEMUCS_ONNX) + _suffix)
        if _dat.exists():
            total += _dat.stat().st_size
    assert total > 40 * 1024 * 1024, f"htdemucs_6s zu klein: {total} bytes"


@pytest.mark.unit
def test_bs_roformer_config_urls_are_live() -> None:
    """Keine toten Download-URLs im Plugin (BSRoFormer/bs_roformer war 404)."""
    src = (_REPO / "plugins" / "bs_roformer_plugin.py").read_text(encoding="utf-8")
    assert "BSRoFormer/bs_roformer/resolve" not in src, "tote HF-URL im Config"
    assert "TRvlvr/model_repo/releases/download" in src
    assert "MelBandRoformer.ckpt" in src  # korrekter Dateiname (war melbandroformer.onnx)


@pytest.mark.unit
def test_bs_roformer_ckpt_present_and_valid_size() -> None:
    """Der 317er-Checkpoint (609,7 MiB) ist lokal beschafft."""
    if not _BS_ROFORMER_CKPT.exists():
        pytest.skip("models/-Paket nicht installiert (gitignored)")
    size = _BS_ROFORMER_CKPT.stat().st_size
    assert abs(size - 639_331_213) < 1_048_576, f"unerwartete ckpt-Größe: {size}"
