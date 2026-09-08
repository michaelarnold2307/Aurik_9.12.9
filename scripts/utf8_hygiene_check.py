#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UTF-8-Hygiene-Check (P2-1) — fail-closed im Pre-Commit.

Normative Grundlage: TODO-P2-1 (docs/TODOS_SOTA_ROADMAP.md). Der gesamte
getrackte Code-Bestand (3175 Textdateien, gemessen 2026-09-08) ist valides
UTF-8. Dieser Guard verhindert, dass UTF-16/UTF-16LE- oder Mischkodierungs-
Reste (BOM, NUL-Byte-Fenster, invalide UTF-8-Sequenzen) wieder unbemerkt
eindringen — sie brechen grep/read-Ansichten und Diff-Werkzeuge.

Regeln (fail-closed):
  R1  Kein UTF-16/UTF-32 BOM (FF FE / FE FF / 00 00 FE FF / FF FE 00 00).
  R2  Ganze Datei muss als UTF-8 dekodierbar sein (keine invaliden Sequenzen).
  R3  Keine UTF-16LE-NUL-Byte-Fenster in Textdateien (>10 % NUL in einem
      2-KiB-Fenster) — fängt BOM-lose UTF-16LE-Mischkodierung.

Verwendung: python3 scripts/utf8_hygiene_check.py [--check] [pfad ...]
Ohne Pfade werden alle getrackten Textdateien via `git ls-files` geprüft.
Exit-Code 0 = sauber, 1 = Verstoß (mit Datei/Byte-Offset-Diagnose).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Konsistent mit SKIP_DIRS aus scripts/aurik_verboten_linter.py
SKIP_DIRS = {
    ".venv",
    ".venv_aurik",
    "__pycache__",
    "node_modules",
    ".git",
    "models/",
    "temp_repro/",
    "plugins/_vendor_",
}

TEXT_EXTS = {
    ".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".cfg", ".ini",
    ".csv", ".tsv", ".sh", ".bat", ".ps1", ".html", ".css", ".js", ".ts",
    ".xml", ".rst", ".sql", ".ipynb", ".qss", ".ui",
}

BOMS = (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff", b"\xff\xfe", b"\xfe\xff")

WINDOW = 2048
NUL_RATIO = 0.10


def is_skipped(path: str) -> bool:
    return any(s in path for s in SKIP_DIRS)


def tracked_text_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [
        p
        for p in out.splitlines()
        if Path(p).suffix.lower() in TEXT_EXTS and not is_skipped(p)
    ]


def check_file(path: str) -> list[str]:
    """Gibt Verstoß-Meldungen zurück (leer = sauber)."""
    problems: list[str] = []
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return [f"{path}: nicht lesbar ({exc})"]
    if not raw:
        return problems
    if raw.startswith(BOMS):
        problems.append(f"{path}: UTF-16/UTF-32 BOM gefunden ({raw[:4].hex()})")
        return problems
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(
            f"{path}: invalide UTF-8-Sequenz bei Byte {exc.start} "
            f"(0x{raw[exc.start]:02x})"
        )
        return problems
    if b"\x00" in raw:
        # Kleine Dateien (< Fenster) als Ganzes prüfen, sonst gleitende Fenster.
        if len(raw) <= WINDOW:
            offsets = [0]
        else:
            offsets = list(range(0, len(raw) - WINDOW + 1, WINDOW))
        for off in offsets:
            window = raw[off : off + WINDOW]
            _nul_count = window.count(b"\x00")
            if _nul_count > len(window) * NUL_RATIO:
                problems.append(
                    f"{path}: UTF-16LE-NUL-Byte-Fenster bei Byte {off} "
                    f"({_nul_count}/{len(window)} NUL)"
                )
                break
    return problems


def main(argv: list[str]) -> int:
    explicit_paths = [a for a in argv[1:] if not a.startswith("-")]
    files = explicit_paths if explicit_paths else tracked_text_files()
    if not files:
        print("utf8_hygiene_check: keine Dateien gefunden (git ls-files leer?).")
        return 1
    violations: list[str] = []
    checked = 0
    for path in files:
        if is_skipped(path):
            continue
        checked += 1
        violations.extend(check_file(path))
    if violations:
        print(f"UTF-8-Hygiene: {len(violations)} Verstoß(e) in {checked} Dateien:")
        for v in violations:
            print("  " + v)
        return 1
    print(f"UTF-8-Hygiene: sauber — {checked} Dateien valides UTF-8 ohne BOM/NUL-Fenster.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
