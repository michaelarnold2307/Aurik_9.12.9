#!/usr/bin/env python3
"""Change-Ledger — strukturiertes Working-Memory aus Git (TASK_CHANGES.md).

Task-lokales Gegenstück zum PR-Evidenzblock: changed/deleted/renamed kommt
aus `git`, nicht aus dem Chat-Verlauf. CI (`ci-lite.yml`, pr-evidence-gate)
erzwingt Abdeckung: jede geänderte Code-Datei muss in TASK_CHANGES.md stehen.

Betriebsarten:
  python scripts/change_ledger.py snapshot [--base REF] [-- PFADE...]
  python scripts/change_ledger.py check --base REF [--head REF]

Exit-Codes:
  0 = Ledger vollständig / keine Code-Änderungen
  1 = TASK_CHANGES.md fehlt oder deckt Code-Änderungen nicht ab
  2 = Nutzungs-/Git-Fehler
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "TASK_CHANGES.md"
CODE_SCOPE_RE = re.compile(r"^(backend|plugins|denker|Aurik10|cli|scripts)/.*\.py$")

_STATUS_LABEL = {
    "A": "neu",
    "M": "modifiziert",
    "D": "gelöscht",
    "R": "umbenannt",
    "T": "Typänderung",
    "C": "kopiert",
    "??": "ungetrackt",
}


def _git(args: list[str]) -> str:
    """Führt git aus; beendet bei Fehlern mit Exit 2."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"FEHLER: git nicht ausführbar: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if proc.returncode != 0:
        print(f"FEHLER: git {' '.join(args)} → {proc.stderr.strip()}", file=sys.stderr)
        raise SystemExit(2)
    return proc.stdout


def _changed(base: str, head: str | None) -> list[tuple[str, str]]:
    """Liefert (status, pfad) für base..head bzw. base..Arbeitsbaum (+ ungetrackt)."""
    entries: list[tuple[str, str]] = []
    refs = [base, head] if head else [base]
    for line in _git(["diff", "--name-status", *refs]).splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        if status.startswith("R") and "\t" in path:
            _, _, path = path.partition("\t")
        entries.append((status[0], path))
    if not head:
        others = _git(["ls-files", "--others", "--exclude-standard"]).splitlines()
        entries.extend(("??", pfad.strip()) for pfad in others if pfad.strip())
    return entries


def _is_code_target(rel: str) -> bool:
    """True für Code-Dateien, die der Ledger abdecken muss."""
    if not CODE_SCOPE_RE.match(rel):
        return False
    return not any(part.startswith("_vendor") for part in rel.split("/"))


def snapshot(base: str, paths: list[str]) -> int:
    """Schreibt TASK_CHANGES.md aus dem Git-Diff (optional auf Pfade begrenzt)."""
    entries = _changed(base, None)
    rows: list[str] = []
    seen: set[str] = set()
    for status, path in entries:
        if path in seen:
            continue
        if paths and path not in paths:
            continue
        seen.add(path)
        rows.append(f"| {status} | {path} | {_STATUS_LABEL.get(status, status)} |")

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# TASK_CHANGES — Live-Ledger der aktuellen Aufgabe",
        "",
        f"> Generiert von `scripts/change_ledger.py snapshot` (Base: `{base}`, Stand: {now}).",
        "> CI (`ci-lite.yml` pr-evidence-gate) erzwingt Abdeckung: jede geänderte Code-Datei muss hier stehen.",
        "",
        "## Geänderte Dateien",
        "",
        "| Status | Pfad | Art |",
        "|---|---|---|",
        *rows,
        "",
        "## Entscheidungen",
        "",
        "- (Architektur-Entscheidungen, kanonische Symbole, Verbote — Aufgabe eintragen)",
        "",
    ]
    LEDGER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ledger geschrieben: {LEDGER.relative_to(ROOT)} ({len(rows)} Einträge)")
    return 0


def _ledger_paths() -> set[str] | None:
    """Pfade aus der Tabelle „## Geänderte Dateien"; None wenn Ledger fehlt."""
    if not LEDGER.exists():
        return None
    covered: set[str] = set()
    in_section = False
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_section = line[3:].strip().startswith("Geänderte Dateien")
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells or cells[0].lower() == "status":
            continue
        if set("".join(cells[0])) <= set("-: "):
            continue
        if len(cells) >= 2 and cells[1]:
            covered.add(cells[1])
    return covered


def check(base: str, head: str | None) -> int:
    """Prüft, ob TASK_CHANGES.md alle Code-Änderungen abdeckt."""
    entries = _changed(base, head)
    code_changed = [path for status, path in entries if status != "D" and _is_code_target(path)]
    if not code_changed:
        print("Keine Code-Änderungen — Ledger-Check übersprungen.")
        return 0

    covered = _ledger_paths()
    fehlend = [pfad for pfad in code_changed if covered is None or pfad not in covered]
    if fehlend:
        print("TASK_CHANGES.md deckt folgende Code-Änderungen nicht ab:")
        for pfad in fehlend:
            print(f"  - {pfad}")
        print(f"Fix: python scripts/change_ledger.py snapshot --base {base}")
        return 1
    print(f"Ledger deckt alle {len(code_changed)} Code-Änderung(en) ab.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Change-Ledger aus Git.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot", help="TASK_CHANGES.md aus git diff schreiben")
    snap.add_argument("--base", default="HEAD", help="Basis-Ref (Standard: HEAD)")
    snap.add_argument("paths", nargs="*", help="Nur diese Pfade aufnehmen")

    chk = sub.add_parser("check", help="Abdeckung von TASK_CHANGES.md prüfen")
    chk.add_argument("--base", default="HEAD", help="Basis-Ref (Standard: HEAD)")
    chk.add_argument("--head", default=None, help="Ziel-Ref (Standard: Arbeitsbaum)")

    args = parser.parse_args()
    if args.cmd == "snapshot":
        return snapshot(args.base, args.paths)
    return check(args.base, args.head)


if __name__ == "__main__":
    sys.exit(main())
