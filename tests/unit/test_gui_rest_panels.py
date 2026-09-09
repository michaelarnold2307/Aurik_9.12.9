"""§GUI-T8 — GUI-Rest-Panels: i18n-Gap, PyQt5-Bindung, Headless-Verifikation.

Befund (2026-09-08): before_after.py und phase_report.py importierten PyQt6
in einer PyQt5-App (Doppelladen-Risiko/SIGABRT), 45 t()-Keys fehlten in der
i18n-Tabelle (Plugin-Manager & Ergebnis-Dialog zeigten rohe Keys), und die
Panels formatierten Dezimalzahlen mit Punkt statt Komma (§GUI-T5).

Diese Tests sichern die Invarianten:
  1. Jeder t("...")-Key in Aurik10 hat einen Eintrag in der i18n-Tabelle.
  2. Kein PyQt6-Import in Aurik10 (einzige GUI-Bindung: PyQt5).
  3. de_num() in ui_constants formatiert deutsch (Komma).
  4. Vorher/Nachher- & Phasen-Panels, Wellenform-Canvas, Splash und
     Plugin-Manager konstruieren und befüllen headless (offscreen) ohne
     Ausnahme und zeigen übersetzte, deutsch formatierte Texte.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import TYPE_CHECKING, Callable, cast

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]

# ── i18n-Gap-Regression (kein Qt nötig) ─────────────────────────────────────


def _used_i18n_keys() -> set[str]:
    used: set[str] = set()
    for p in (_REPO / "Aurik10").rglob("*.py"):
        if "i18n" in p.parts:
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'\bt\(\s*"([^"]+)"', src):
            used.add(m.group(1))
    return used


def _table_i18n_keys() -> set[str]:
    src = (_REPO / "Aurik10" / "i18n" / "__init__.py").read_text(encoding="utf-8")
    return set(re.findall(r'"([a-z0-9_]+(?:\.[a-z0-9_]+)+)"\s*:', src))


@pytest.mark.unit
def test_every_t_key_has_translation() -> None:
    """Kein Dialog darf rohe Keys anzeigen (t() fällt sonst auf den Key zurück)."""
    missing = sorted(_used_i18n_keys() - _table_i18n_keys())
    assert missing == [], f"Fehlende i18n-Keys: {missing}"


@pytest.mark.unit
def test_no_pyqt6_import_in_aurik10() -> None:
    """Nur PyQt5 als GUI-Bindung — Qt5/Qt6-Doppelladen crasht (SIGABRT)."""
    offenders: list[str] = []
    for p in (_REPO / "Aurik10").rglob("*.py"):
        if re.search(r"(?:from|import)\s+PyQt6\b", p.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(p.relative_to(_REPO)))
    assert offenders == [], f"PyQt6-Importe gefunden: {offenders}"


@pytest.mark.unit
def test_before_after_and_phase_report_have_no_untranslated_labels() -> None:
    """Sichtbare Texte der Panels laufen über t() — keine QLabel-Literale."""
    for name in ("before_after.py", "phase_report.py"):
        src = (_REPO / "Aurik10" / "ui" / name).read_text(encoding="utf-8")
        assert not re.search(r'QLabel\("[^"]*[A-Za-zÄÖÜäöüß]', src), f"{name}: hartkodierter QLabel-Text"
        assert 'setHorizontalHeaderLabels(["Phase"' not in src, f"{name}: hartkodierte Spaltenköpfe"


# ── de_num (pure, kein Qt) ──────────────────────────────────────────────────


def _run_de_num() -> Callable[..., str]:
    from Aurik10.ui.ui_constants import de_num

    return cast(Callable[..., str], de_num)


@pytest.mark.unit
def test_ui_constants_de_num_german_comma() -> None:
    de_num = _run_de_num()
    assert de_num(26.876) == "26,88"
    assert de_num(42.5, 1) == "42,5"
    assert de_num(-0.005, 3) == "-0,005"
    assert de_num(100.0) == "100,00"


# ── Headless-Qt-Verifikation ────────────────────────────────────────────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


@pytest.mark.unit
def test_before_after_widget_load_german_decimals() -> None:
    from Aurik10.ui.before_after import BeforeAfterWidget

    widget = BeforeAfterWidget()
    widget.load(
        before={"defects": 12, "bandwidth_hz": 8000, "lufs": -19.4, "quality": 0.58},
        after={"defects": 3, "bandwidth_hz": 15000, "lufs": -14.1, "quality": 0.86},
    )
    texts = [lab.text() for lab in widget.findChildren(__import__("PyQt5.QtWidgets", fromlist=["QLabel"]).QLabel)]
    assert any("Störungen" in t for t in texts), texts  # übersetzter Titel
    assert "12" in texts and any(t.startswith("3") for t in texts), texts  # Vorher/Nachher-Werte
    assert any("-19,4" in t for t in texts), texts  # deutsches Komma (§GUI-T5)
    assert any("-14,1" in t for t in texts), texts
    assert all("." not in t.split(" LUFS")[0] for t in texts if "LUFS" in t), texts


@pytest.mark.unit
def test_phase_report_widget_load_german_decimals() -> None:
    from Aurik10.ui.phase_report import PhaseReportWidget, _format_seconds

    widget = PhaseReportWidget()
    widget.load(
        phases_executed=["phase_01_detect", "phase_02_denoise"],
        phases_skipped=["phase_09_gate"],
        phase_deltas={
            "phase_01_detect": {"duration_s": 3.47, "quality_delta": 0.0042, "hpi_live": 0.812},
            "phase_02_denoise": {"duration_s": 122.0, "quality_delta": -0.0051, "hpi_live": 0.0},
        },
        total_time_s=125.5,
    )
    assert "2 Phasen ausgeführt" in widget._summary_label.text()
    assert "1 übersprungen" in widget._summary_label.text()
    headers = [widget._table.horizontalHeaderItem(i).text() for i in range(5)]
    assert headers[0] == "Phase" and headers[4] == "Status", headers
    assert widget._table.item(0, 1).text() == "3,5 s"  # deutsches Komma
    assert widget._table.item(1, 1).text() == "2:02"
    assert widget._table.item(0, 2).text() == "+0,004"
    assert widget._table.item(1, 2).text() == "-0,005"
    assert widget._table.item(0, 3).text() == "0,81"
    assert widget._table.item(1, 3).text() == "—"
    assert widget._table.item(2, 4).text() == "⏭ übersprungen"
    assert _format_seconds(0.4) == "0,4 s"


@pytest.mark.unit
def test_ab_preview_waveform_renders_offscreen() -> None:
    import numpy as np

    from Aurik10.ui.ab_preview import ABPreviewWidget

    widget = ABPreviewWidget()
    widget.resize(320, 120)
    widget.set_audio(np.random.default_rng(0).standard_normal(4800).astype(np.float32), 48000, "test.wav")
    widget._waveform.resize(320, 120)
    assert widget.grab() is not None  # paintEvent ohne Ausnahme


@pytest.mark.unit
def test_splash_screen_constructs_and_paints_offscreen() -> None:
    from Aurik10.ui.splash_screen import AurikSplashScreen

    splash = AurikSplashScreen()
    splash.set_status("Bereit.")
    splash.resize(760, 420)
    assert splash.grab() is not None  # paintEvent ohne Ausnahme


@pytest.mark.unit
def test_plugin_manager_title_translated() -> None:
    from Aurik10.ui.plugin_manager import PluginManagerWidget

    dialog = PluginManagerWidget()
    title = dialog.windowTitle()
    assert title != "plugin_manager.title", "Roher i18n-Key als Fenstertitel"
    assert title == "Plugin-Manager"
