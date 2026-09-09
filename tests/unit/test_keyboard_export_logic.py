"""§GUI-T4 — keyboard_shortcuts + export_presets: reine Logik (headless).

Die Tasten→Aktion-Entscheidung, die Seek-Klemmung und der Presets-Vertrag
sind pure Logik — sie entscheiden, welche Aktionen der Player ausführt und
welche Exportformate der Laien-Dialog anbietet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")  # CI-Minimal-Umgebung (cross-platform) ohne PyQt5

from PyQt5.QtCore import Qt

from Aurik10.ui.export_presets import ExportPresetDialog
from Aurik10.ui.keyboard_shortcuts import KeyboardShortcuts


def test_key_action_matrix() -> None:
    assert KeyboardShortcuts.key_action(Qt.Key_Space, 0) == ("play_pause", 0.0)
    assert KeyboardShortcuts.key_action(Qt.Key_Left, 0) == ("seek", -5.0)
    assert KeyboardShortcuts.key_action(Qt.Key_Left, Qt.ShiftModifier) == ("seek", -30.0)
    assert KeyboardShortcuts.key_action(Qt.Key_Right, 0) == ("seek", 5.0)
    assert KeyboardShortcuts.key_action(Qt.Key_Right, Qt.ShiftModifier) == ("seek", 30.0)
    assert KeyboardShortcuts.key_action(Qt.Key_Escape, 0) == ("stop", 0.0)
    assert KeyboardShortcuts.key_action(Qt.Key_E, 0) == ("expert", 0.0)
    assert KeyboardShortcuts.key_action(Qt.Key_A, 0) is None  # unbelegte Taste


def test_seek_frac_clamps() -> None:
    assert KeyboardShortcuts.seek_frac(100.0, -200.0, 200.0) == 0.0  # vor Anfang
    assert KeyboardShortcuts.seek_frac(190.0, 30.0, 200.0) == 1.0  # hinter Ende
    assert abs(KeyboardShortcuts.seek_frac(50.0, 5.0, 200.0) - 0.275) < 1e-9
    assert KeyboardShortcuts.seek_frac(10.0, 5.0, 0.0) <= 1.0  # duration 0 → kein Crash


def test_presets_contract() -> None:
    presets = ExportPresetDialog.PRESETS
    assert len(presets) == 7
    ids = [p["id"] for p in presets]
    assert len(set(ids)) == 7  # eindeutig
    for p in presets:
        assert p["title_key"] and p["desc_key"]
        if p["id"] == "custom":
            assert p["fmt"] is None  # Custom öffnet den technischen Dialog
        else:
            assert p["fmt"] in {"mp3", "wav", "aac", "flac"}
            assert p["sr"] in {44100, 48000}
            assert p["bits"] in {16, 24}
            if p["fmt"] == "mp3":
                assert p.get("bitrate")  # MP3 braucht Bitrate
    archive = next(p for p in presets if p["id"] == "archive")
    assert archive["fmt"] == "flac" and archive["bits"] == 24
    youtube = next(p for p in presets if p["id"] == "youtube")
    assert youtube["fmt"] == "wav" and youtube["bits"] == 24
