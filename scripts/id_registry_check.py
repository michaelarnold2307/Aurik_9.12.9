#!/usr/bin/env python3
"""ID-Registry-Check — prüft §-Zitate gegen die kanonische ID-Registry.

Umsetzung der Phasen 2–4 des ID-Bereinigungsplans
(siehe docs/ID_COLLISION_MAP.md). Verdrahtet als Pre-Commit-Hook
`aurik-id-registry` (fail-closed für neue Verstöße).

Betriebsarten:
  - Report (Standard): Meldungen, Exit 0 — Analyse der Bestandszitate.
  - --strict: Exit 1 bei WARNUNGEN — die Pre-Commit-Variante.
  - --fix: qualifiziert nackte Ambiguitäts-Zitate mechanisch anhand des
    Qualifikations-Mappings in der Registry (idempotent, Kommentar-Ebene).

Regeln:
  R1 (unbekannte ID): Zitat passt in keinen registrierten Namensraum.
  R2 (nacktes Ambiguitäts-Zitat): ID aus dem Ambiguitäts-Set ohne
     Quellen-Qualifikation in derselben Zeile.
  R3 (Hinweis): Zitat aus einem veralteten, verifier-internen oder
     informellen Namensraum bzw. bare ID ohne Nummer.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / ".github" / "ID_REGISTRY.md"
DEFAULT_DIRS = [
    "backend",
    "denker",
    "forensics",
    "Aurik10",
    "tests",
    "plugins",
    "scripts",
]
SKIP_DIRS = {"__pycache__", ".venv", "venv", "build", "node_modules"}

R3_STATUSES = {"veraltet", "verifier-intern", "unspezifisch", "Referenz", "informell"}

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"§[A-Za-z0-9][A-Za-z0-9._-]*")

QUALIFIER_SCHLUESSEL = (
    "copilot-instructions",
    "GEBOTE",
    "GEBOTEN",
    "VERBOTEN",
    "VERBOTE",
    "Vintage",
    "Spec",
)


def _expand(ids: str) -> list[str]:
    """Erweitert '§G71 (GEBOTE.md)–§G75 (GEBOTE.md)' und Komma-Listen zu einzelnen IDs."""
    ergebnis: list[str] = []
    for teil in ids.split(","):
        teil = teil.strip()
        parts = [p.strip() for p in teil.split("–")]
        if len(parts) == 2:
            m1 = re.match(r"^(§[A-Za-z-]+)(\d+)$", parts[0])
            m2 = re.match(r"^(§[A-Za-z-]+)(\d+)$", parts[1])
            if m1 and m2 and m1.group(1) == m2.group(1):
                prefix = m1.group(1)
                start, end = int(m1.group(2)), int(m2.group(2))
                ergebnis.extend(f"{prefix}{n}" for n in range(start, end + 1))
                continue
        ergebnis.append(teil)
    return ergebnis


class Registry:
    """Geparste ID-Registry: Namensräume, Ambiguitäts-Set, Aliasse, Mapping."""

    def __init__(self) -> None:
        self.namespaces: list[tuple[re.Pattern, str, str]] = []
        self.ambiguous: set = set()
        self.aliases: dict = {}
        self.qualifikation: dict = {}

    def load(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        for row in self._tabelle(text, "Namensräume"):
            if row[0].lower() == "muster":
                continue
            try:
                muster_text = row[0].strip("`")
                self.namespaces.append((re.compile(muster_text), row[1], row[2]))
            except re.error:
                logger.error("Ungültiges Namensraum-Muster in Registry: %s", row[0])
        for row in self._tabelle(text, "Ambiguitäts-Set"):
            if row[0].lower() == "id":
                continue
            self.ambiguous.update(_expand(row[0]))
        for row in self._tabelle(text, "Aliasse"):
            if row[0].lower() == "alias-id":
                continue
            for einzel_id in _expand(row[0]):
                self.aliases[einzel_id] = row[1]
        for row in self._tabelle(text, "Qualifikations-Mapping"):
            if row[0].lower() == "muster":
                continue
            qual = row[1].strip() if len(row) > 1 else ""
            for einzel_id in _expand(row[0].replace("`", "")):
                self.qualifikation[einzel_id] = qual

    @staticmethod
    def _tabelle(text: str, titel: str) -> list[list[str]]:
        zeilen = text.splitlines()
        rows: list[list[str]] = []
        in_section = False
        for line in zeilen:
            if line.startswith("## "):
                in_section = line[3:].strip().startswith(titel)
                continue
            if not in_section:
                continue
            stripped = line.strip()
            if not stripped.startswith("|") or not stripped.endswith("|"):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not cells:
                continue
            if set("".join(cells[0])) <= set("-: "):
                continue  # Trenner-Zeile
            if cells[0]:
                rows.append(cells)
        return rows

    def klassifiziere(self, zitat: str) -> tuple[str | None, str | None]:
        """Liefert (Status, Quelle) oder (None, None) bei unbekannter ID."""
        if zitat in self.ambiguous:
            return ("ambig", None)
        for muster, quelle, status in self.namespaces:
            if muster.fullmatch(zitat):
                return (status, quelle)
        return (None, None)


def _qualifiziert(zeile: str) -> bool:
    return any(schluessel in zeile for schluessel in QUALIFIER_SCHLUESSEL)


def _sammle_dateien(root: Path, verzeichnisse: list[str]) -> list[Path]:
    ziele: list[Path] = []
    for name in verzeichnisse:
        start = root / name
        if not start.is_dir():
            continue
        for p in start.rglob("*.py"):
            teile = set(p.parts)
            if teile & SKIP_DIRS:
                continue
            ziele.append(p)
    return ziele


def _kuerze(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fixe_qualifikation(registry: Registry, ziele: list[Path]) -> None:
    """Qualifiziert nackte Ambiguitäts-Zitate mechanisch (idempotent)."""
    geaenderte_zeilen = 0
    geaenderte_dateien = 0
    for datei in ziele:
        try:
            inhalt = datei.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        neu: list[str] = []
        datei_geaendert = False
        for zeile in inhalt.splitlines():
            orig = zeile
            if not _qualifiziert(zeile):
                for zitat, qual in registry.qualifikation.items():
                    if zitat in zeile and zitat in registry.ambiguous:
                        zeile = re.sub(
                            rf"{re.escape(zitat)}(?![\w.-])",
                            f"{zitat} {qual}",
                            zeile,
                        )
            if zeile != orig:
                datei_geaendert = True
                geaenderte_zeilen += 1
            neu.append(zeile)
        if datei_geaendert:
            datei.write_text("\n".join(neu) + "\n", encoding="utf-8")
            geaenderte_dateien += 1
    logger.info(
        "Qualifikation: %d Zeile(n) in %d Datei(en) ergänzt.",
        geaenderte_zeilen,
        geaenderte_dateien,
    )


def scan(registry: Registry, ziele: list[Path]) -> tuple[int, int]:
    """Prüft alle Ziele; liefert (Anzahl WARNUNGEN, Anzahl HINWEISE)."""
    warnungen = 0
    hinweise = 0
    for datei in ziele:
        try:
            inhalt = datei.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for nr, zeile in enumerate(inhalt.splitlines(), start=1):
            for match in CITATION_RE.finditer(zeile):
                zitat = match.group(0).rstrip(".,;:)]}")
                status, quelle = registry.klassifiziere(zitat)
                if status is None:
                    warnungen += 1
                    logger.warning("R1 unbekannte ID: %s (%s:%d)", zitat, _kuerze(datei), nr)
                elif status == "ambig":
                    if not _qualifiziert(zeile):
                        warnungen += 1
                        logger.warning(
                            "R2 nacktes Ambiguitäts-Zitat: %s (%s:%d) — "
                            "Quelle in der Zeile angeben (Zitierdisziplin "
                            "AGENTS.md §5)",
                            zitat,
                            _kuerze(datei),
                            nr,
                        )
                elif status in R3_STATUSES:
                    hinweise += 1
                    alias_hinweis = registry.aliases.get(zitat)
                    if alias_hinweis:
                        logger.info(
                            "R3 %s: %s (%s:%d) Quelle: %s — Alias auf %s",
                            status,
                            zitat,
                            _kuerze(datei),
                            nr,
                            quelle,
                            alias_hinweis,
                        )
                    else:
                        logger.info(
                            "R3 %s: %s (%s:%d) Quelle: %s",
                            status,
                            zitat,
                            _kuerze(datei),
                            nr,
                            quelle,
                        )
    return warnungen, hinweise


def main() -> int:
    parser = argparse.ArgumentParser(description="Prüft §-Zitate gegen die ID-Registry.")
    parser.add_argument(
        "dateien",
        nargs="*",
        help="Zu prüfende Dateien; Standard: backend/ denker/ forensics/ Aurik10/ tests/ plugins/ scripts/",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--strict", action="store_true", help="Exit 1, sobald WARNUNGEN auftreten")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Nackte Ambiguitäts-Zitate gemäß Qualifikations-Mapping mechanisch qualifizieren (idempotent)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    registry = Registry()
    registry.load(args.registry)

    if args.dateien:
        ziele = [Path(f) for f in args.dateien]
    else:
        ziele = _sammle_dateien(ROOT, DEFAULT_DIRS)

    if args.fix:
        fixe_qualifikation(registry, ziele)

    warnungen, hinweise = scan(registry, ziele)
    logger.info(
        "ID-Registry-Check abgeschlossen: %d WARNUNG(en), %d HINWEIS(e), %d Datei(en) geprüft.",
        warnungen,
        hinweise,
        len(ziele),
    )
    return 1 if args.strict and warnungen > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
