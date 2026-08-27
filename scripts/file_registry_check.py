#!/usr/bin/env python3
"""File-Lifecycle-Check — Write-Gate für neue Code-Dateien + Registry-Validierung.

Fail-closed nach dem Muster von `scripts/id_registry_check.py` (R1/R2),
angewendet auf Dateien statt Regel-IDs. Verdrahtet als Pre-Commit-Hook
`aurik-file-lifecycle`.

Regeln:
  R1 neue Code-Datei (backend/ plugins/ denker/ Aurik10/ cli/ scripts/,
     *.py, gestaged, diff-filter=A) ohne Eintrag in
     `.github/FILE_REGISTRY.md` → ERROR. Das Pflichtfeld „Ersetzt“ erzwingt
     die Suche nach einer bestehenden Alternative VOR der Anlage
     („Search before Create“).
  R2 Status-Wert außerhalb ACTIVE/DEPRECATED/MIGRATING/GENERATED/TEST_ONLY/
     ARCHIVED/FORBIDDEN → ERROR.
  R3 DEPRECATED oder MIGRATING ohne „Ersetzt“-Ziel → ERROR.
  R4 Pfad doppelt in der Registry → ERROR.
  R5 Import einer FORBIDDEN/ARCHIVED-Datei → ERROR (via scripts/repo_graph.py).
  R6 (WARN) registrierte Datei existiert nicht mehr im Repo.
  R7 (WARN) „Ersetzt“-Ziel existiert weder auf Platte noch in der Registry.

Vendored-Drittanbieter-Code (`plugins/_vendor_*`) ist vom Write-Gate
ausgenommen — unverändert kopiert, wie bei den übrigen Linter-Gates.

Betriebsarten:
  python scripts/file_registry_check.py        # Commit-Modus (R1–R5)
  python scripts/file_registry_check.py --all  # Nur Registry-Validierung (R2–R7)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / ".github" / "FILE_REGISTRY.md"
CODE_SCOPE_RE = re.compile(r"^(backend|plugins|denker|Aurik10|cli|scripts)/.*\.py$")

logger = logging.getLogger(__name__)


def _load_repo_graph():
    """Lädt scripts/repo_graph.py als Modul (kein Package-Import nötig)."""
    path = Path(__file__).with_name("repo_graph.py")
    spec = importlib.util.spec_from_file_location("repo_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_code_target(rel: str) -> bool:
    """True für neue Code-Dateien, die dem Write-Gate unterliegen."""
    if not CODE_SCOPE_RE.match(rel):
        return False
    return not any(part.startswith("_vendor") for part in rel.split("/"))


def _staged_new_files() -> list[str]:
    """Liefert gestagte neue Dateien (diff-filter=A) relativ zum Repo-Root."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--diff-filter=A", "--name-only", "-z"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [pfad for pfad in proc.stdout.split("\0") if pfad]


def validate_registry(reg: dict[str, dict[str, str]], rg) -> tuple[list[str], list[str]]:
    """Prüft die Registry-Einträge; liefert (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    for rel in sorted(reg):
        entry = reg[rel]
        if entry["status"] not in rg.STATUS_ENUM:
            errors.append(f"R2 unbekannter Status {entry['status']!r} für {rel}")
        if entry["status"] in {"DEPRECATED", "MIGRATING"} and not entry["ersetzt"]:
            errors.append(f"R3 {entry['status']} ohne Ersetzt-Ziel: {rel}")
        if not (ROOT / rel).exists() and entry["status"] not in {"ARCHIVED", "FORBIDDEN"}:
            warnings.append(f"R6 registrierte Datei fehlt im Repo: {rel}")
        ziel = entry["ersetzt"]
        if ziel and ziel != "—":
            if not (ROOT / ziel).exists() and ziel not in reg:
                warnings.append(f"R7 Ersetzt-Ziel nicht auffindbar: {rel} → {ziel}")
    return errors, warnings


def duplicate_paths() -> list[str]:
    """R4: Pfade, die in der Dateien-Tabelle mehrfach vorkommen."""
    seen: dict[str, int] = {}
    in_section = False
    for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_section = line[3:].strip().startswith("Dateien")
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells or cells[0].lower() == "pfad":
            continue
        if set("".join(cells[0])) <= set("-: "):
            continue
        if cells[0]:
            seen[cells[0]] = seen.get(cells[0], 0) + 1
    return [pfad for pfad, count in seen.items() if count > 1]


def forbidden_import_errors(reg: dict[str, dict[str, str]], rg) -> list[str]:
    """R5: Importe auf FORBIDDEN/ARCHIVED-Dateien (nur wenn solche existieren)."""
    if not any(entry["status"] in rg.NO_IMPORT_STATUSES for entry in reg.values()):
        return []
    graph = rg.build_graph(reg)
    return [
        f"R5 {v['file']} importiert {v['import']} (Status FORBIDDEN/ARCHIVED)"
        for v in rg.find_forbidden_imports(graph, reg)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write-Gate für neue Code-Dateien + FILE_REGISTRY-Validierung.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Nur Registry-Validierung (R2–R7), kein Write-Gate auf gestagte Dateien",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rg = _load_repo_graph()
    registry = rg.parse_file_registry(REGISTRY_PATH)

    if not REGISTRY_PATH.exists():
        logger.error("Registry fehlt: %s", REGISTRY_PATH.relative_to(ROOT))
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(f"R4 doppelter Pfad-Eintrag: {pfad}" for pfad in duplicate_paths())
    errs, warns = validate_registry(registry, rg)
    errors.extend(errs)
    warnings.extend(warns)

    if not args.all:
        neue = [pfad for pfad in _staged_new_files() if _is_code_target(pfad)]
        fehlend = [pfad for pfad in neue if pfad not in registry]
        for pfad in fehlend:
            errors.append(
                f"R1 neue Code-Datei ohne Registry-Eintrag: {pfad} — Eintrag in "
                ".github/FILE_REGISTRY.md anlegen (Status, Domain, Canonical, "
                f"Ersetzt, Grund); vorher: scripts/repo_search.py --before-create {pfad}"
            )
        errors.extend(forbidden_import_errors(registry, rg))

    for warn in warnings:
        logger.warning("%s", warn)
    for error in errors:
        logger.error("%s", error)
    logger.info(
        "File-Lifecycle-Check: %d ERROR(s), %d WARNUNG(en), %d Datei(en) registriert.",
        len(errors),
        len(warnings),
        len(registry),
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
