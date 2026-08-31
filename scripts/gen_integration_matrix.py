"""gen_integration_matrix.py — Regeneriert die GEBOTE-/VERBOTEN-Integrations-Matrix.

Liest die normativen Kataloge (.github/GEBOTE.md, .github/VERBOTEN.md) und
bestimmt für jede ID den Integrations-Status:
  - ``integriert``            extern zitiert (Code/Tests/Instructions/Registry/Skripte/Specs)
  - ``integriert (Linter)``   V-ID ist im aurik_verboten_linter hartkodiert
  - ``integriert (zitiert)``  V-ID extern zitiert (nicht linter-enforced)
  - ``katalog``               nur im Definitionsdokument vorhanden (Referenzkatalog)

Das Definitionsdokument selbst und die Matrix zählen NICHT als Zitat
(sonst wäre jede ID trivial „integriert").

Ausgabe: .github/GEBOTE_INTEGRATION_MATRIX.md — maschinell geprüft durch
audit/spec_integration_scanner.py (Fehlerprotokoll, --fail-on error).

Nutzung:
    python scripts/gen_integration_matrix.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".github" / "GEBOTE_INTEGRATION_MATRIX.md"

_CITATION_DOCS: tuple[str, ...] = (
    "AGENTS.md",
    ".github/copilot-instructions.md",
    ".github/ID_REGISTRY.md",
    ".github/FILE_REGISTRY.md",
)
_CODE_DIRS: tuple[str, ...] = (
    "backend",
    "denker",
    "forensics",
    "cli",
    "Aurik10",
    "plugins",
    "tests",
    "scripts",
)
_SKIP_PARTS: tuple[str, ...] = (".git", "__pycache__", ".venv", "node_modules", "_vendor", "models", "temp_repro")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _gebote_rows() -> list[tuple[str, str, str]]:
    """(ID, Kategorie, Titel) aus GEBOTE.md-Tabellenzeilen."""
    rows: list[tuple[str, str, str]] = []
    kategorie = ""
    for ln in _read(ROOT / ".github" / "GEBOTE.md").splitlines():
        if ln.startswith("##"):
            kategorie = ln.lstrip("# ").strip()
            continue
        m = re.match(r"\|\s*(§G\d+)\s*\|\s*\*\*(.+?)\*\*", ln)
        if m:
            rows.append((m.group(1), kategorie, m.group(2).strip()))
    return rows


def _verboten_rows() -> list[tuple[str, str]]:
    """(ID, Titel) aus VERBOTEN.md — alle V-IDs, nicht nur Tabellenzeilen."""
    text = _read(ROOT / ".github" / "VERBOTEN.md")
    titles: dict[str, str] = {}
    for ln in text.splitlines():
        m = re.match(r"\|\s*(?:✅\s*|❌\s*)?(V\d{2})\s*[|:]\s*(.+?)\s*\|", ln)
        if m:
            titles.setdefault(m.group(1), m.group(2).strip()[:110])
    all_ids = sorted(set(re.findall(r"\bV\d{2}\b", text)))
    out: list[tuple[str, str]] = []
    for vid in all_ids:
        title = titles.get(vid)
        if title is None:
            # Fundstelle als Fallback-Titel
            title = next(
                (ln.strip()[:110] for ln in text.splitlines() if f" {vid} " in ln or f"({vid})" in ln),
                f"VERBOTEN {vid}",
            )
        out.append((vid, title))
    return out


def _citation_pools() -> list[str]:
    pools: list[str] = []
    for rel in _CITATION_DOCS:
        p = ROOT / rel
        if p.exists():
            pools.append(_read(p))
    for p in (ROOT / ".github" / "instructions").glob("*.md"):
        pools.append(_read(p))
    for p in (ROOT / ".github" / "specs").glob("*.md"):
        pools.append(_read(p))
    for rel_dir in _CODE_DIRS:
        rdir = ROOT / rel_dir
        if not rdir.exists():
            continue
        for p in rdir.rglob("*.py"):
            if any(part in _SKIP_PARTS for part in p.parts):
                continue
            try:
                pools.append(_read(p))
            except OSError:
                continue
    return pools


def main() -> int:
    joined = "\n".join(_citation_pools())
    linter_ids = set(re.findall(r"\bV\d{2}\b", _read(ROOT / "scripts" / "aurik_verboten_linter.py")))

    def cited(gid: str) -> bool:
        return bool(re.search(re.escape(gid) + r"(?!\d)", joined))

    rows: list[tuple[str, str, str, str]] = []
    for gid, kat, titel in _gebote_rows():
        status = "integriert" if cited(gid) else "katalog"
        rows.append((gid, kat, titel, status))
    for vid, titel in _verboten_rows():
        if vid in linter_ids:
            status = "integriert (Linter)"
        elif cited(vid):
            status = "integriert (zitiert)"
        else:
            status = "katalog"
        rows.append((vid, "VERBOTEN.md", titel, status))

    n_katalog = sum(1 for r in rows if r[3].startswith("katalog"))
    lines = [
        "# GEBOTE-/VERBOTEN-Integrations-Matrix",
        "",
        "> **Status: Aktiv — maschinell geprüft durch `audit/spec_integration_scanner.py`.**",
        "> Jede normative ID der GEBOTE.md (Referenzkatalog) und VERBOTEN.md muss entweder",
        "> extern zitiert sein (Code/Tests/Instructions/Registry/Skripte/Specs — ohne das",
        "> jeweilige Definitionsdokument und ohne diese Matrix selbst) oder hier einen",
        "> dokumentierten Status haben. Fehlt beides, meldet der Scanner einen ERROR im",
        "> Fehlerprotokoll. `katalog` = Referenzkatalog-Eintrag ohne externes Zitat;",
        "> `integriert` = extern zitiert; VERBOTEN-IDs zusätzlich mit Linter-Status.",
        "",
        f"Stand: {len(rows)} IDs ({len(rows) - n_katalog} integriert, {n_katalog} katalog).",
        "Regeneriert mit `python scripts/gen_integration_matrix.py`.",
        "",
        "| ID | Kategorie | Titel | Status |",
        "|---|---|---|---|",
    ]
    for gid, kat, titel, status in rows:
        lines.append(f"| {gid} | {kat} | {titel} | {status} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Matrix geschrieben: {OUT} ({len(rows)} IDs, {n_katalog} katalog)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
