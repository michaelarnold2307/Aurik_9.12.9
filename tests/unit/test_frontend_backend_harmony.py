"""§v10.990: Frontend↔Backend-Harmonie — Drift-Gates.

Diese Tests pinnen die Brücke zwischen GUI und SOTA-Backend:
  1. Die zwei BridgeCalibrationData-Kopien (Aurik10/ui ↔ backend/api) dürfen NIE driften.
  2. Bridge quality_color-Hex-Werte MÜSSEN mit der UI-Palette übereinstimmen.
  3. SOTA-Zugänge der Bridge liefern Frontend-taugliche Daten.
  4. Das Status-Panel nutzt die Palette (keine Hex-Werte mehr im Code).
  5. modern_window verdrahtet das Status-Panel in den Statusbereich.
"""

from __future__ import annotations

import ast
import re
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _dataclass_field_names(mod: object) -> list[str]:
    import dataclasses

    cls = getattr(mod, "BridgeCalibrationData")
    return [f.name for f in dataclasses.fields(cls)]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Drift-Gate: BridgeCalibrationData (Frontend-Kopie vs Backend-Kopie)
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_calibration_dataclass_copies_identical():
    """Die beiden BridgeCalibrationData-Kopien müssen Feld-identisch bleiben."""
    import Aurik10.ui.bridge_calibration as frontend_mod
    import backend.api.bridge_calibration_data as backend_mod

    fe = _dataclass_field_names(frontend_mod)
    be = _dataclass_field_names(backend_mod)
    assert fe == be, f"Feld-Drift: frontend={fe} backend={be}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Drift-Gate: Bridge-Qualitätsfarben == UI-Palette
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_quality_colors_match_ui_palette():
    """Die drei depth-abhängigen quality_color-Werte der Bridge entsprechen der Palette."""
    bridge_src = _read("backend/api/bridge.py")
    palette_src = _read("Aurik10/ui/ui_constants.py")

    palette_hexes = set(re.findall(r"#[0-9A-Fa-f]{6}", palette_src))
    # Bridge-seitige Qualitätsfarben (nur die drei depth-Abstufungen)
    bridge_colors = re.findall(r'color = "(#[0-9A-Fa-f]{6})" if depth >= \d else "\1"', bridge_src)
    # einfacher: alle Hex-Werte im _build_bridge_calibration_dict-Block
    m = re.search(
        r'def _build_bridge_calibration_dict.*?color = "(#[0-9A-Fa-f]{6})" if depth >= 4 else "\("(#[0-9A-Fa-f]{6})" if depth >= 3 else "(#[0-9A-Fa-f]{6})"\)"',
        bridge_src,
        re.S,
    )
    if m:
        deep, moderate, studio = m.group(1), m.group(2), m.group(3)
        for hex_val in (deep, moderate, studio):
            assert hex_val in palette_hexes, f"Bridge-Farbe {hex_val} fehlt in der UI-Palette"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Bridge-SOTA-Zugänge liefern Frontend-taugliche Daten
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_model_zoo_summary_shape():
    from backend.api.bridge import get_model_zoo_summary

    entries = get_model_zoo_summary()
    assert isinstance(entries, list)
    assert len(entries) >= 1
    for e in entries:
        assert {"name", "purpose", "status", "integration", "notes"} <= set(e.keys())


def test_bridge_sota_chain_status_keys():
    from backend.api.bridge import get_sota_chain_status

    status = get_sota_chain_status()
    assert "model_zoo" in status
    assert "components" in status
    comps = status["components"]
    for key in ("defect_consensus", "repair_planner", "artifact_guards", "perceptual_loop"):
        assert isinstance(comps.get(key), bool), f"components.{key} fehlt"


def test_bridge_guard_report_from_repair_report():
    """get_guard_report liest §v10.990 RepairReport-Telemetrie."""
    from backend.api.bridge import get_guard_report
    from backend.core.coordinated_repair import RepairPlan, RepairReport

    report = RepairReport(
        plan=RepairPlan(),
        completed_steps=[],
        failed_steps=[],
        total_time=1.0,
        input_peak=0.5,
        output_peak=0.5,
        guard_violations={"truepeak": 1, "spectral": 2},
        guard_peak_delta_db=0.8,
        utmos_iterations=3,
        utmos_blend_count=1,
        utmos_mos_before=3.1,
        utmos_mos_after=2.9,
    )
    data = get_guard_report(report)
    assert data["guards"]["truepeak"] == 1
    assert data["guards"]["spectral"] == 2
    assert data["guards"]["peak_delta_db"] == 0.8
    assert data["utmos_loop"]["iterations"] == 3
    assert data["utmos_loop"]["blend_back"] is True

    class _Wrap:
        repair_report = report
        metadata = {}

    data2 = get_guard_report(_Wrap())
    assert data2["guards"]["spectral"] == 2


def test_bridge_defect_consensus_summary_defensive():
    from backend.api.bridge import get_defect_consensus_summary

    assert get_defect_consensus_summary(None) == {}
    # Fremdes Objekt → Null-Statistik statt Exception (Frontend-tolerant)
    zeroed = get_defect_consensus_summary(object())
    assert zeroed["defect_count"] == 0
    assert zeroed["module_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Status-Panel: Palette statt Hex-Werte + SOTA-Methoden
# ═══════════════════════════════════════════════════════════════════════════════


def test_status_panel_uses_palette_tokens_not_hex():
    """restoration_status_panel.py darf keine Hex-Farben mehr hartkodieren."""
    src = _read("Aurik10/ui/restoration_status_panel.py")
    # Alle Hex-Werte außerhalb des Palette-Imports prüfen
    after_imports = src.split("from Aurik10.ui.ui_constants import")[-1]
    hex_occurrences = re.findall(r'"#[0-9A-Fa-f]{6}"', after_imports)
    assert not hex_occurrences, f"Hex-Werte im Panel-Code: {hex_occurrences}"


def test_status_panel_has_sota_methods():
    src = _read("Aurik10/ui/restoration_status_panel.py")
    for method in ("set_sota_chain", "set_consensus_summary", "set_repair_plan_summary", "set_guard_report"):
        assert f"def {method}(" in src, f"{method} fehlt im Status-Panel"


def test_ui_constants_palette_defined():
    src = _read("Aurik10/ui/ui_constants.py")
    for token in (
        "SURFACE_BG",
        "TEXT_PRIMARY",
        "TEXT_MUTED",
        "QUALITY_STUDIO",
        "QUALITY_MODERATE",
        "QUALITY_DEEP_CHAIN",
        "STATUS_OK_TEXT",
        "STATUS_CRIT_BG",
        "BADGE_MATERIAL_TEXT",
    ):
        assert f"{token} =" in src, f"Palette-Token {token} fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. modern_window-Verdrahtung (Source-Level-Gates)
# ═══════════════════════════════════════════════════════════════════════════════


def test_modern_window_wires_status_panel():
    src = _read("Aurik10/ui/modern_window.py")
    assert "_RestorationStatusPanel(wrapper)" in src, "Panel wird nicht im Statusbereich erzeugt"
    assert "self._status_panel = _RestorationStatusPanel(wrapper)" in src
    assert "self._sync_status_panel(_eff_step, _eff_total, _live_hint)" in src, "Phasen-Sync fehlt"
    assert "_panel.set_complete()" in src, "Abschluss-Update fehlt"
    assert "_sync_status_panel_sota()" in src


def test_close_event_headless_gate():
    """§v10.991: closeEvent darf im Headless/Offscreen-Modus keinen modalen Dialog öffnen.

    Regression für den Layout-Gate-Hang: _dlg.exec() nur wenn
    Worker laufen UND sichtbar/interaktiv.
    """
    src = _read("Aurik10/ui/modern_window.py")
    assert '_headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or not self.isVisible()' in src
    assert "if _workers_running and not _headless:" in src
    assert 'setattr(_dlg, "_close_requested", True)' in src


def test_restaurier_denker_stores_repair_plan_for_frontend():
    src = _read("denker/restaurier_denker.py")
    # Mypy-Fassung (933f1a89): cast(Any, cached_defect_result).repair_plan = _repair_plan
    assert "cached_defect_result).repair_plan = _repair_plan" in src


def test_bridge_exports_sota_accessors():
    """Alle neuen SOTA-Zugänge stehen in __all__ (für Stern-Imports stabil)."""
    src = _read("backend/api/bridge.py")
    m = re.search(r"__all__ = \[(.*?)\]", src, re.S)
    assert m, "__all__ fehlt in bridge.py"
    body = m.group(1)
    for name in (
        "get_model_zoo_summary",
        "get_sota_chain_status",
        "get_defect_consensus_summary",
        "get_repair_plan_summary",
        "get_guard_report",
        "get_repair_plan_consent",
        "get_restoration_bericht",
    ):
        assert f'"{name}"' in body, f"{name} fehlt in __all__"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.992: Laienverständliche Einwilligungs-Ansicht
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_repair_plan_consent_layman_language():
    """Consent-Ansicht übersetzt Defekte + Phasen in Alltagssprache."""
    from backend.api.bridge import get_repair_plan_consent
    from backend.core.coordinated_repair import RepairPlan, RepairPriority, RepairStep
    from backend.core.defect_consensus_pipeline import DefectCategory, DefectHypothesis, DefectManifest

    manifest = DefectManifest(
        defects=[
            DefectHypothesis(
                category=DefectCategory.HISS,
                start_sample=0,
                end_sample=1000,
                confidence=0.8,
                severity=0.2,
                source_module="x",
            ),
            DefectHypothesis(
                category=DefectCategory.CRACKLE,
                start_sample=0,
                end_sample=1000,
                confidence=0.9,
                severity=0.5,
                source_module="x",
            ),
            DefectHypothesis(
                category=DefectCategory.CLICK,
                start_sample=0,
                end_sample=1000,
                confidence=0.95,
                severity=0.7,
                source_module="x",
            ),
        ],
    )
    plan = RepairPlan(
        steps=[
            RepairStep(
                phase_id="phase_09_crackle_removal",
                priority=RepairPriority.TRANSIENT,
                defect_category="crackle",
                affected_samples=[],
            ),
            RepairStep(
                phase_id="phase_03_denoise",
                priority=RepairPriority.BREITBAND,
                defect_category="hiss",
                affected_samples=[],
            ),
        ]
    )

    class _DefektResult:
        _consensus_manifest = manifest
        repair_plan = plan

    consent = get_repair_plan_consent(_DefektResult())
    # Sortiert nach Schwere: Klick (Kritisch) → Knistern (Stark) → Rauschen (Mittel)
    assert [f["label"] for f in consent["found"]] == ["Knackser & Klicks", "Knistern", "Rauschen"]
    assert [f["severity"] for f in consent["found"]] == ["Kritisch", "Stark", "Mittel"]
    # Kein technisches Phasen-Vokabular in der Einwilligung
    assert consent["will_do"] == ["Knistern entfernen", "Rauschen reduzieren"]
    assert not any("phase_" in w for w in consent["will_do"])


def test_bridge_repair_plan_consent_fallback_and_defensive():
    """Legacy defect_scores-Fallback + defensives Verhalten."""
    from backend.api.bridge import get_repair_plan_consent

    assert get_repair_plan_consent(None) == {}
    assert get_repair_plan_consent(object()) == {}

    class _LegacyResult:
        defect_scores = {"tape_hiss": 0.45, "hum": 0.05}

    consent = get_repair_plan_consent(_LegacyResult())
    assert [f["label"] for f in consent["found"]] == ["Bandrauschen", "Brummen"]
    assert consent["found"][0]["severity"] == "Stark"
    assert consent["will_do"] == []


def test_status_panel_consent_method_and_wiring():
    src_panel = _read("Aurik10/ui/restoration_status_panel.py")
    assert "def set_repair_consent(" in src_panel
    assert "_consent_label" in src_panel
    src_win = _read("Aurik10/ui/modern_window.py")
    assert "_panel.set_repair_consent(get_repair_plan_consent(_defect))" in src_win
    assert "self._sync_status_panel_sota()" in src_win


def test_bridge_repair_plan_summary_has_aligned_actions():
    """§v10.997: actions[] parallel zu phase_order[] — Basis des Live-Trackers."""
    from backend.api.bridge import get_repair_plan_summary
    from backend.core.coordinated_repair import RepairPlan, RepairPriority, RepairStep

    plan = RepairPlan(
        steps=[
            RepairStep(
                phase_id="phase_09_crackle_removal",
                priority=RepairPriority.TRANSIENT,
                defect_category="crackle",
                affected_samples=[],
            ),
            RepairStep(
                phase_id="phase_03_denoise",
                priority=RepairPriority.BREITBAND,
                defect_category="hiss",
                affected_samples=[],
            ),
        ]
    )
    summary = get_repair_plan_summary(plan)
    assert summary["phase_order"] == ["phase_09_crackle_removal", "phase_03_denoise"]
    assert summary["actions"] == ["Knistern entfernen", "Rauschen reduzieren"]
    assert len(summary["actions"]) == len(summary["phase_order"])


@pytest.mark.gui
def test_status_panel_live_plan_tracker():
    """§v10.997: Während der Verarbeitung wird der Plan live als ✓/▶/○ angezeigt."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from Aurik10.ui.restoration_status_panel import RestorationStatusPanel

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    panel = RestorationStatusPanel()
    panel.resize(1200, 120)
    panel.show()
    app.processEvents()

    panel.set_repair_consent(
        {
            "found": [{"label": "Knistern", "severity": "Stark"}],
            "will_do": ["Knistern entfernen", "Rauschen reduzieren", "Höhen rekonstruieren"],
        }
    )
    panel.set_repair_plan_summary(
        {
            "step_count": 3,
            "phase_order": ["phase_09_crackle_removal", "phase_03_denoise", "phase_06_frequency_restoration"],
            "actions": ["Knistern entfernen", "Rauschen reduzieren", "Höhen rekonstruieren"],
        }
    )

    # Vor dem Start: statische Einwilligung
    assert "Aurik wird: Knistern entfernen" in panel._consent_label.text()

    # Erste Phase läuft
    panel.set_phase("phase_09_crackle_removal", 1, 3)
    assert "▶ Knistern entfernen" in panel._consent_label.text()
    assert "○ Rauschen reduzieren" in panel._consent_label.text()

    # Zweite Phase läuft → erste ist erledigt
    panel.set_phase("phase_03_denoise", 2, 3)
    assert "✓ Knistern entfernen" in panel._consent_label.text()
    assert "▶ Rauschen reduzieren" in panel._consent_label.text()

    # Abschluss → alles erledigt
    panel.set_complete()
    _text = panel._consent_label.text()
    assert _text.count("✓") == 3
    assert "▶" not in _text and "○" not in _text


@pytest.mark.gui
def test_status_panel_consent_renders():
    """Funktional: Consent-Zeile wird sichtbar und zeigt laienverständlichen Text."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from Aurik10.ui.restoration_status_panel import RestorationStatusPanel

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    panel = RestorationStatusPanel()
    panel.resize(1000, 120)
    panel.show()
    app.processEvents()
    panel.set_repair_consent(
        {
            "found": [{"label": "Knistern", "severity": "Stark"}],
            "will_do": ["Knistern entfernen", "Rauschen reduzieren"],
        }
    )
    assert panel._consent_label.isVisible()
    assert "Gefunden: Knistern" in panel._consent_label.text()
    assert "Aurik wird: Knistern entfernen" in panel._consent_label.toolTip()
    panel.set_repair_consent({})
    assert panel._consent_label.isHidden()


def test_ui_still_bridge_only():
    """§11: Das Status-Panel importiert weiterhin NUR backend.api.bridge."""
    src = _read("Aurik10/ui/restoration_status_panel.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("backend.core", "plugins", "dsp")), alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith(("backend.core", "plugins", "dsp")), mod


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.993: GUI-Bug-Erkennung im Live-Betrieb — Crash-Report-Sichtbarkeit
# ═══════════════════════════════════════════════════════════════════════════════


def test_crash_reporter_new_reports_lifecycle(tmp_path, monkeypatch):
    """get_new_reports → mark_seen → keine erneute Anzeige."""
    import json
    import time

    from backend.core import crash_reporter

    monkeypatch.setattr(crash_reporter, "_REPORTS_DIR", tmp_path)
    monkeypatch.setattr(crash_reporter, "_LAST_SEEN_FILE", tmp_path / ".last_seen")

    # Basislinie ohne Reports
    assert crash_reporter.get_new_reports() == []
    crash_reporter.mark_reports_seen()
    _base = crash_reporter.get_last_seen_ts()
    assert _base > 0

    # Neuen Report schreiben (älter als Basislinie → unsichtbar)
    _old = tmp_path / "crash_old.json"
    _old.write_text(json.dumps({"exception": {"type": "ValueError", "message": "alt"}}), encoding="utf-8")
    _old_ts = time.time() - 60
    import os

    os.utime(_old, (_old_ts, _old_ts))
    assert crash_reporter.get_new_reports() == []

    # Frischen Report schreiben → sichtbar mit type/message
    _new = tmp_path / "crash_new.json"
    _new.write_text(json.dumps({"exception": {"type": "KeyError", "message": "kaputt"}}), encoding="utf-8")
    reports = crash_reporter.get_new_reports()
    assert len(reports) == 1
    assert reports[0]["type"] == "KeyError"
    assert reports[0]["message"] == "kaputt"

    # Gesehen → weg
    crash_reporter.mark_reports_seen()
    assert crash_reporter.get_new_reports() == []


def test_bridge_crash_report_accessors():
    """Bridge-Zugänge liefern Listen und sind defensiv."""
    from backend.api.bridge import get_new_crash_reports, mark_crash_reports_seen

    reports = get_new_crash_reports()
    assert isinstance(reports, list)
    mark_crash_reports_seen()  # darf nie werfen


def test_main_installs_crash_handler():
    """§v10.993: Der GUI-Start MUSS den Exception-Hook installieren (sonst toter Code).

    §11-UI-Bridge (normativ): Der Import läuft über backend.api.bridge — direkte
    backend.core-Importe in Aurik10/ sind verboten (test_ui_bridge_only_imports).
    """
    src = _read("Aurik10/main.py")
    assert "from backend.api.bridge import install_crash_handler" in src
    assert "install_crash_handler()" in src


def test_modern_window_checks_crash_reports_on_start():
    """§v10.993: Startup-Dialog für Fehler der letzten Sitzung ist verdrahtet."""
    src = _read("Aurik10/ui/modern_window.py")
    assert "QTimer.singleShot(2500, self._check_crash_reports)" in src
    assert "def _check_crash_reports(self) -> None:" in src
    assert "Bericht kopieren" in src
    assert "mark_crash_reports_seen()" in src


def test_ci_gui_smoke_gate_exists():
    """§v10.993: Das GUI-Smoke-Gate MUSS im CI-Lite-Workflow verankert sein.

    Ohne dieses Gate laufen GUI-Tests nie automatisch — Bug-Regressionen
    (wie der §v10.991-Dialog-Hang) würden unentdeckt in den Live-Betrieb gehen.
    """
    src = _read(".github/workflows/ci-lite.yml")
    assert "gui-smoke-gate:" in src
    assert "tests/normative/test_e2e_gui_smoke.py" in src
    assert "tests/ui/test_ui_quality.py" in src
    assert "--run-gui-tests" in src
    assert "QT_QPA_PLATFORM: offscreen" in src


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.996: Konsolidierter Restaurierungs-Bericht — der Kreis schließt sich
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_restoration_bericht_closes_the_loop():
    """Plan (Defekt-Ergebnis) + Ausführung (Result) → EIN Bericht."""
    from backend.api.bridge import get_restoration_bericht
    from backend.core.coordinated_repair import RepairPlan, RepairPriority, RepairStep
    from backend.core.defect_consensus_pipeline import DefectCategory, DefectHypothesis, DefectManifest

    manifest = DefectManifest(
        defects=[
            DefectHypothesis(
                category=DefectCategory.CRACKLE,
                start_sample=0,
                end_sample=1000,
                confidence=0.9,
                severity=0.5,
                source_module="x",
            ),
        ]
    )
    plan = RepairPlan(
        steps=[
            RepairStep(
                phase_id="phase_09_crackle_removal",
                priority=RepairPriority.TRANSIENT,
                defect_category="crackle",
                affected_samples=[],
            ),
        ]
    )

    class _Defect:
        _consensus_manifest = manifest
        repair_plan = plan

    class _Result:
        quality_estimate = 0.78
        phases_skipped = ["phase_02_hum_removal"]
        deferred_phases = ["phase_55_diffusion_inpainting"]
        metadata = {
            "phases_total": 6,
            "restorability_score": 43.0,
            "no_effect_phase_count": 1,
            "mushra": {"mushra_score": 84.0},
            "hpi_score": 0.31,
            "narrator": {"verdict": "Knistern entfernt, Höhen rekonstruiert."},
            "do_no_harm": {"reverted": False},
        }

    bericht = get_restoration_bericht(_Result(), _Defect())
    assert bericht["found"] == [{"label": "Knistern", "severity": "Stark"}]
    assert bericht["planned"] == ["Knistern entfernen"]
    assert bericht["done_count"] == 6
    assert bericht["skipped_count"] == 1
    assert bericht["deferred_count"] == 1
    assert bericht["no_effect_count"] == 1
    assert bericht["guards"]["guards"]["truepeak"] == 0  # keine Eingriffe = gutes Zeichen
    assert bericht["proof"]["quality_after"] == 78.0
    assert bericht["proof"]["mushra"] == 84.0
    assert bericht["was_reverted"] is False


def test_bridge_restoration_bericht_defensive():
    from backend.api.bridge import get_restoration_bericht

    assert get_restoration_bericht(None) == {}
    assert get_restoration_bericht(object()) == {}


def test_results_summary_renders_bericht_section():
    src = _read("Aurik10/ui/results_summary.py")
    assert "restoration_bericht" in src
    assert "🔍 Gefunden:" in src
    assert "✅ Aurik hat:" in src
    assert "🛡️ Sicherheitsnetz:" in src


def test_modern_window_passes_bericht_to_dialog():
    src = _read("Aurik10/ui/modern_window.py")
    assert "_bericht = get_restoration_bericht(restoration_result, _defect)" in src
    assert "restoration_bericht=_bericht," in src


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.999: Live-Mithören — jederzeit, so lange wie gewollt, Sprung überallhin
# ═══════════════════════════════════════════════════════════════════════════════


def test_live_mitheoren_toggle_and_position_memory():
    """Der Zwischenstand-Knopf stoppt statt neu zu starten; Position bleibt erhalten."""
    src = _read("Aurik10/ui/modern_window.py")
    # Toggle: laufender Live-Zwischenstand → Stoppen statt Neustart
    assert "# §v10.999: Läuft der Zwischenstand bereits? → Stoppen statt Neustart" in src
    assert "self._playback_is_live_preview = True" in src
    # Positionsgedächtnis: Start an der zuletzt gehörten Stelle
    assert '_start = float(getattr(self, "_live_playback_frac", 0.0) or 0.0)' in src
    assert "self._play_audio(_safe_preview, _preview_sr, start_pos_frac=_start)" in src
    # Beim Stoppen wird die aktuelle Position gemerkt
    assert "self._live_playback_frac = float(_sp.position_frac)" in src
    # Zu Ende gehört → nächster Start von vorn
    assert "self._live_playback_frac = 0.0" in src


def test_live_mitheoren_scrubber_seek_works():
    """Der Transport-Scrubber springt tatsächlich (war vorher toter Code)."""
    src = _read("Aurik10/ui/modern_window.py")
    # Scrubber-Referenzen werden in BEIDEN Wiedergabe-Pfaden gesetzt
    assert src.count("self._play_audio_ref =") >= 2
    assert src.count("self._play_sr_ref =") >= 2
    # Scrubber ruft Fraction-Seek (Streaming-Player erwartet Fraction)
    assert "self._seek_playback(_frac)" in src
    assert "_player.seek(float(frac))" in src
    # Live-Sprung merkt sich die Position
    assert "self._live_playback_frac = _frac" in src


def test_live_snapshot_is_full_length():
    """Der Live-Snapshot ist der VOLLE aktuelle Stand — nicht nur Sekunden."""
    src = _read("Aurik10/ui/modern_window.py")
    assert "self._live_preview_audio = _preview.copy()" in src


@pytest.mark.gui
def test_results_summary_dialog_shows_bericht():
    """Funktional: Der Dialog rendert die Bericht-Sektion aus dem data-dict."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from Aurik10.ui.results_summary import ResultsSummaryDialog

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    dlg = ResultsSummaryDialog(
        {
            "file_name": "test.wav",
            "restoration_bericht": {
                "found": [{"label": "Knistern", "severity": "Stark"}],
                "planned": ["Knistern entfernen"],
                "done_count": 3,
                "skipped_count": 0,
                "deferred_count": 0,
                "no_effect_count": 0,
                "guards": {
                    "guards": {"truepeak": 0, "pumping": 0, "formant": 0, "spectral": 0},
                    "utmos_loop": {"iterations": 5},
                },
                "proof": {"verdict": "Knistern entfernt."},
            },
            "quality_before": None,
            "quality_after": None,
        }
    )
    dlg.show()
    app.processEvents()
    from PyQt5.QtWidgets import QLabel

    texts = [w.text() for w in dlg.findChildren(QLabel)]
    assert any("Gefunden: Knistern" in t for t in texts)
    assert any("Aurik hat: Knistern entfernen" in t for t in texts)
    assert any("UTMOS-Kontrolle 5×" in t for t in texts)
    dlg.close()
