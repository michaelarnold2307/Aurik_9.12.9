#!/usr/bin/env python3
"""Ear-Critical-Paritäts-Check (fail-closed, Pre-Commit-Hook `aurik-ear-parity`).

Prüft die Registry `.github/EAR_CRITICAL_PARITY.md`:
- enforced: Implementierungs- und Test-Datei müssen existieren und das
  Prüf-Token enthalten (verhindert die §0j-Klasse von Fehlern: Regel
  spezifiziert, im Code aber wirkungslos oder ungetestet).
- deferred: nur mit expliziter Begründung erlaubt (kein stiller Skip).
- Mindestens 8 enforced-Zeilen (Schutz gegen versehentliches Leeren).

Exit 0 = OK, 1 = Verstoß.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = _ROOT / ".github" / "EAR_CRITICAL_PARITY.md"
MIN_ENFORCED_ROWS = 8


def _rows() -> list[list[str]]:
    if not REGISTRY.exists():
        return []
    rows: list[list[str]] = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells or cells[0] in ("Regel-ID", "---") or "Status" in cells:
            continue
        if len(cells) >= 7:
            rows.append(cells)
    return rows


def _contains(path: Path, token: str) -> bool:
    try:
        return token in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def check() -> tuple[int, list[str]]:
    problems: list[str] = []
    rows = _rows()
    if not rows:
        problems.append(f"Registry {REGISTRY} fehlt oder enthält keine Zeilen")
        return 1, problems
    enforced = 0
    for cells in rows:
        regel, token, impl, test, status = (
            cells[0],
            cells[3] if len(cells) > 3 else "",
            cells[4] if len(cells) > 4 else "",
            cells[5] if len(cells) > 5 else "",
            cells[6] if len(cells) > 6 else "",
        )
        reason = cells[7] if len(cells) > 7 else ""
        if status == "enforced":
            enforced += 1
            if token in ("", "—") or impl in ("", "—") or test in ("", "—"):
                problems.append(f"{regel}: enforced ohne Token/Implementierung/Test")
                continue
            impl_path = _ROOT / impl
            test_path = _ROOT / test
            if not impl_path.exists():
                problems.append(f"{regel}: Implementierungs-Datei fehlt: {impl}")
            elif not _contains(impl_path, token):
                problems.append(f"{regel}: Token {token!r} fehlt in {impl} (Regel ohne Code-Wirkung?)")
            if not test_path.exists():
                problems.append(f"{regel}: Test-Datei fehlt: {test}")
            elif not _contains(test_path, token):
                problems.append(f"{regel}: Token {token!r} fehlt in {test} (ungesicherte Regel)")
        elif status == "deferred":
            if reason in ("", "—"):
                problems.append(f"{regel}: deferred ohne Begründung — stiller Skip verboten")
        else:
            problems.append(f"{regel}: unbekannter Status {status!r} (erlaubt: enforced|deferred)")
    if enforced < MIN_ENFORCED_ROWS:
        problems.append(f"nur {enforced} enforced-Zeilen (< {MIN_ENFORCED_ROWS})")
    return (1 if problems else 0), problems


def main() -> int:
    code, problems = check()
    if problems:
        print(f"Ear-Critical-Parität: {len(problems)} Verstoß/Vorstöße:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Ear-Critical-Parität: OK (alle enforced-Regeln haben Code+Test).")
    return code


if __name__ == "__main__":
    sys.exit(main())
