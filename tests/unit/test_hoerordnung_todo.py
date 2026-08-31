"""Tests für die To-Do-Generierung des H-Serie-Scanners (Session-Übergabe)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".agents" / "skills" / "bug-prevention"))

from scan_anti_patterns import _write_hoerordnung_todo


def _issue(path: str, line: int, hid: str) -> str:
    return f"{path}:{line}: {hid} Testbeschreibung fuer {hid}."


def test_prioritaet_und_stabile_ids(tmp_path: Path) -> None:
    dest = tmp_path / "todo.md"
    issues = [
        _issue("backend/core/phases/p1.py", 10, "H07"),
        _issue("backend/core/phases/p2.py", 20, "H02"),
        _issue("backend/core/phases/p3.py", 30, "H03"),
    ]
    _write_hoerordnung_todo(issues, str(dest))
    txt = dest.read_text(encoding="utf-8")
    lines = [ln for ln in txt.splitlines() if ln.startswith("- [")]
    # H02 (schwerste) muss zuerst stehen, H07 zuletzt
    assert "H02-" in lines[0]
    assert "H07-" in lines[-1]
    # IDs sind Hash-stabil (6 Hex-Zeichen), nicht fortlaufend
    for ln in lines:
        eid = ln.split("|")[0].strip().split()[3]
        assert len(eid.split("-")[1]) == 6


def test_status_erhalt_ueber_regeneration(tmp_path: Path) -> None:
    dest = tmp_path / "todo.md"
    issues = [_issue("backend/core/phases/p1.py", 10, "H02"), _issue("backend/core/phases/p2.py", 20, "H03")]
    _write_hoerordnung_todo(issues, str(dest))
    txt = dest.read_text(encoding="utf-8")
    # Erste offene Checkbox-Zeile als Ziel
    line = next(ln for ln in txt.splitlines() if ln.startswith("- [ ]"))
    eid = line.split("|")[0].strip().split()[3]
    dest.write_text(txt.replace(f"- [ ] {eid}", f"- [x] {eid}", 1), encoding="utf-8")
    _write_hoerordnung_todo(issues, str(dest))  # regenerieren
    txt2 = dest.read_text(encoding="utf-8")
    assert f"- [x] {eid}" in txt2  # bleibt erledigt
    assert f"- [ ] {eid}" not in txt2


def test_leere_funde_schreiben_leere_liste(tmp_path: Path) -> None:
    dest = tmp_path / "todo.md"
    _write_hoerordnung_todo([], str(dest))
    assert dest.exists()
    assert not any(ln.startswith("- [") for ln in dest.read_text(encoding="utf-8").splitlines())
