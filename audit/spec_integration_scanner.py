"""spec_integration_scanner.py — Erkennung der vollständigen Integration aller Vorgaben und Specs.

Prüft die normative Kette (AGENTS.md §1) auf Integrations-Lücken und gibt
fehlende Punkte als Fehlerprotokoll aus (JSON + Markdown + Exit-Code):

  1. Normative Dokumente existieren (copilot-instructions, VERBOTEN, GEBOTE,
     ID_REGISTRY, FILE_REGISTRY, instructions/*, AGENTS.md).
  2. Spec-Datei-Referenzen lösen auf (keine toten Links, keine verwaisten Specs).
  3. GEBOTE-/VERBOTEN-Integrations-Matrix: jede normative ID muss entweder
     irgendwo zitiert sein (Code/Tests/Instructions/Registry/Skripte) ODER
     einen dokumentierten Eintrag im Integrations-Ledger haben
     (.github/GEBOTE_INTEGRATION_MATRIX.md). Fehlend → ERROR.
  4. copilot-instructions.md enthält die Pflicht-Abschnitte §I–§VI, §0a,
     [RELEASE_MUST].
  5. Enforce-Gates als Subprozesse integriert (Exit-Code = Protokoll-Error):
     release_must_coverage_check, id_registry_check, check_crossfire_phases.
  6. VERBOTEN-Linter-Abdeckung: jede V-ID der VERBOTEN.md muss im Linter
     hartkodiert sein oder im Ledger als dokumentiert ausgewiesen werden.

Nutzung:
    python audit/spec_integration_scanner.py --workspace . \
        --json-out audit/spec_integration_report.json \
        --md-out audit/spec_integration_report.md \
        --fail-on error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_SEVERITY_RANK: dict[str, int] = {"error": 3, "warning": 2, "info": 1}

_NORMATIVE_DOCS: tuple[str, ...] = (
    ".github/copilot-instructions.md",
    ".github/VERBOTEN.md",
    ".github/GEBOTE.md",
    ".github/ID_REGISTRY.md",
    ".github/FILE_REGISTRY.md",
    ".github/instructions/hoerordnung.instructions.md",
    ".github/instructions/pipeline.instructions.md",
    ".github/instructions/phases.instructions.md",
    ".github/instructions/dsp.instructions.md",
    ".github/instructions/musical_goals.instructions.md",
    ".github/instructions/tests.instructions.md",
    "AGENTS.md",
)

_INTEGRATION_LEDGER = ".github/GEBOTE_INTEGRATION_MATRIX.md"

_COPILOT_REQUIRED_SECTIONS: tuple[str, ...] = (
    "§I",  # GEBOTE G1–G9
    "§II",  # VERBOTE V1–V9
    "§III",  # DSP-Spezialregeln
    "§IV",  # Export-Reihenfolge
    "§V",  # CD-Rauschprofil-Modell
    "§VI",  # Startup-Vertrag
    "§0a",  # verbotene Phasen
)

_CODE_DIRS: tuple[str, ...] = ("backend", "denker", "forensics", "cli", "Aurik10", "plugins")
_SKIP_PARTS: tuple[str, ...] = (".git", "__pycache__", ".venv", "node_modules", "_vendor", "models", "temp_repro")


@dataclass
class IntegrationFinding:
    """Ein Integrations-Befund im Fehlerprotokoll."""

    check: str
    severity: str  # error | warning | info
    item: str
    detail: str
    recommendation: str


@dataclass
class IntegrationResult:
    findings: list[IntegrationFinding] = field(default_factory=list)
    duration_s: float = 0.0
    totals: dict[str, int] = field(default_factory=dict)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_code_texts(workspace: Path) -> tuple[list[str], str]:
    """Liest alle Produktions-/Test-Python-Dateien; gibt (Dateien, Gesamttext)."""
    files: list[str] = []
    parts: list[str] = []
    for rel_dir in (*_CODE_DIRS, "tests", "scripts"):
        root = workspace / rel_dir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(workspace).as_posix()
            if any(p in rel for p in _SKIP_PARTS):
                continue
            files.append(rel)
            parts.append(_read(path))
    return files, "\n".join(parts)


def _collect_ids(text: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, text))


def _gebote_ids(workspace: Path) -> set[str]:
    return _collect_ids(_read(workspace / ".github/GEBOTE.md"), r"§G\d+")


def _verboten_ids(workspace: Path) -> set[str]:
    return _collect_ids(_read(workspace / ".github/VERBOTEN.md"), r"\bV\d{2}\b")


def _gebote_heading(workspace: Path, gid: str) -> str:
    """Kategorie/Titel eines GEBOTE-IDs aus der Katalog-Datei (best effort)."""
    text = _read(workspace / ".github/GEBOTE.md")
    for line in text.splitlines():
        if gid in line and line.strip().startswith(("#", "|", "-", "*")):
            clean = line.strip().lstrip("#|*- ").strip()
            if len(clean) <= 140:
                return clean
    # Fallback: nächste Überschrift über der Fundstelle
    lines = text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if gid in ln), 0)
    for ln in reversed(lines[max(0, idx - 12) : idx]):
        if ln.startswith("#"):
            return ln.lstrip("# ").strip()[:140]
    return f"{gid} (Kategorie unbekannt)"


def _ledger_entries(workspace: Path) -> dict[str, str]:
    """Liest das Integrations-Ledger: ID → Status-Zeile."""
    ledger = workspace / _INTEGRATION_LEDGER
    if not ledger.exists():
        return {}
    entries: dict[str, str] = {}
    for line in _read(ledger).splitlines():
        m = re.match(r"\s*\|?\s*(§G\d+|V\d{2})\s*[|:]\s*(.+?)\s*\|?\s*$", line)
        if m:
            entries[m.group(1)] = m.group(2).strip()
    return entries


def check_normative_docs(workspace: Path) -> list[IntegrationFinding]:
    out: list[IntegrationFinding] = []
    for rel in _NORMATIVE_DOCS:
        if not (workspace / rel).exists():
            out.append(
                IntegrationFinding(
                    check="normative_docs",
                    severity="error",
                    item=rel,
                    detail="Normatives Dokument fehlt",
                    recommendation=f"Dokument {rel} wiederherstellen (AGENTS.md §1 normative Kette).",
                )
            )
    return out


def check_spec_references(workspace: Path) -> list[IntegrationFinding]:
    out: list[IntegrationFinding] = []
    specs_dir = workspace / ".github/specs"
    existing = {p.name for p in specs_dir.glob("*.md")}
    doc_pool = ""
    for rel in _NORMATIVE_DOCS:
        p = workspace / rel
        if p.exists():
            doc_pool += _read(p)
    # Index NICHT in den Referenz-Pool aufnehmen — er ist nur Existenz-Anker,
    # kein Integrations-Nachweis. Sonst wäre kein Spec je „verwaist“.
    for p in specs_dir.glob("*.md"):
        if p.name != "00_SPEC_INDEX.md":
            doc_pool += _read(p)
    # Nur echte Spec-Namensfamilien zählen (NN_…, v10.…), keine beliebigen
    # .md-Tokens aus Prosa/Code-Beispielen — verhindert Fragment-/False-Positives.
    _SPEC_PATH_RX = re.compile(r"(?:\.github/)?specs/([0-9v][A-Za-z0-9._-]*\.md)")
    _SPEC_BARE_RX = re.compile(
        r"(?<![A-Za-z0-9_/.])(?:v10(?:\.[0-9]+)?[A-Za-z0-9._-]*\.md|\d{2}[A-Za-z0-9][A-Za-z0-9._-]*\.md)"
    )
    refs: set[str] = set()
    for path_ref in _SPEC_PATH_RX.findall(doc_pool):
        refs.add(path_ref)
    for bare_ref in _SPEC_BARE_RX.findall(doc_pool):
        refs.add(bare_ref)
    # Tote Spec-Referenzen
    for ref in sorted(refs):
        if ref.endswith(".md") and ref not in existing and not (specs_dir / ref).exists():
            out.append(
                IntegrationFinding(
                    check="spec_references",
                    severity="error",
                    item=ref,
                    detail="Spec-Datei wird referenziert, existiert aber nicht",
                    recommendation="Spec-Datei anlegen oder Referenz korrigieren.",
                )
            )
    # Kanonischer Spec-Index: Source of Truth über den Spec-Bestand.
    index_path = specs_dir / "00_SPEC_INDEX.md"
    indexed: set[str] = set()
    if index_path.exists():
        for m in re.finditer(r"\|\s*([A-Za-z0-9][A-Za-z0-9._-]*\.md)\s*\|", _read(index_path)):
            indexed.add(m.group(1))
        dead = sorted(indexed - existing)
        missing = sorted({n for n in existing if n != "00_SPEC_INDEX.md"} - indexed)
        for name in dead:
            out.append(
                IntegrationFinding(
                    check="spec_index",
                    severity="error",
                    item=name,
                    detail="Spec-Index listet nicht existierende Datei",
                    recommendation="Index-Eintrag entfernen oder Datei wiederherstellen.",
                )
            )
        for name in missing:
            out.append(
                IntegrationFinding(
                    check="spec_index",
                    severity="error",
                    item=name,
                    detail="Spec-Datei fehlt im kanonischen Index",
                    recommendation="Eintrag in 00_SPEC_INDEX.md ergänzen.",
                )
            )
    else:
        out.append(
            IntegrationFinding(
                check="spec_index",
                severity="error",
                item="00_SPEC_INDEX.md",
                detail="Kanonischer Spec-Index fehlt",
                recommendation=".github/specs/00_SPEC_INDEX.md mit allen Specs anlegen.",
            )
        )
    # Verwaiste Specs: im Index verankert, aber nirgendwo sonst referenziert.
    # (Nicht indexierte Specs meldet bereits der spec_index-Check als ERROR.)
    # Severity info: Der Index ist der kanonische Existenz-Anker (analog zum
    # „katalog“-Status der GEBOTE-Matrix) — Transparenz ohne Gate-Blockade.
    _, code_pool = _iter_code_texts(workspace)
    for name in sorted(existing):
        if name == "00_SPEC_INDEX.md":
            continue
        if name in indexed and name not in doc_pool and name not in code_pool:
            out.append(
                IntegrationFinding(
                    check="spec_references",
                    severity="info",
                    item=name,
                    detail="Spec ist nur im Index verankert, sonst nirgendwo referenziert",
                    recommendation="Spec in der normativen Kette oder im Code verlinken, sonst als obsolet markieren.",
                )
            )
    return out


def check_gebote_verboten_matrix(workspace: Path) -> list[IntegrationFinding]:
    """Kern-Check: jede GEBOTE-/VERBOTEN-ID muss zitiert ODER im Ledger dokumentiert sein."""
    out: list[IntegrationFinding] = []
    ledger = _ledger_entries(workspace)
    if not ledger:
        out.append(
            IntegrationFinding(
                check="gebote_verboten_matrix",
                severity="error",
                item=_INTEGRATION_LEDGER,
                detail="Integrations-Ledger fehlt — keine dokumentierten Katalog-Status vorhanden",
                recommendation="Ledger anlegen (alle GEBOTE-/VERBOTEN-IDs mit Status).",
            )
        )
    pools: list[str] = []
    # Achtung: GEBOTE.md/VERBOTEN.md sind die Definitionsdokumente — sie
    # zitieren ihre eigenen IDs und dürfen NICHT als Integrations-Nachweis
    # zählen (sonst wäre jede ID trivial „zitiert“).
    for rel in _NORMATIVE_DOCS:
        p = workspace / rel
        if p.exists() and p.name not in {"GEBOTE.md", "VERBOTEN.md"}:
            pools.append(_read(p))
    _, code_pool = _iter_code_texts(workspace)
    pools.append(code_pool)
    joined = "\n".join(pools)

    for gid in sorted(_gebote_ids(workspace), key=lambda s: int(re.sub(r"\D", "", s) or 0)):
        # Exakte ID-Prüfung: §G10 darf nicht über §G100 als „zitiert“ gelten.
        if re.search(re.escape(gid) + r"(?!\d)", joined) or gid in ledger:
            continue
        out.append(
            IntegrationFinding(
                check="gebote_verboten_matrix",
                severity="error",
                item=gid,
                detail=f"Nicht integriert: {_gebote_heading(workspace, gid)}",
                recommendation="ID im Code zitieren (§G9 (copilot-instructions.md) Zitierdisziplin) oder Status im Integrations-Ledger dokumentieren.",
            )
        )
    for vid in sorted(_verboten_ids(workspace)):
        if vid in joined or vid in ledger:
            continue
        out.append(
            IntegrationFinding(
                check="gebote_verboten_matrix",
                severity="error",
                item=vid,
                detail="VERBOTEN-ID weder zitiert noch im Ledger dokumentiert",
                recommendation="Linter-Regel zuordnen oder Status im Integrations-Ledger dokumentieren.",
            )
        )
    # Ledger-Einträge ohne zugrunde liegende ID (veraltete Einträge)
    known_ids = _gebote_ids(workspace) | _verboten_ids(workspace)
    for lid in sorted(ledger):
        if lid not in known_ids:
            out.append(
                IntegrationFinding(
                    check="gebote_verboten_matrix",
                    severity="warning",
                    item=lid,
                    detail="Ledger-Eintrag ohne zugehörige normative ID (veraltet?)",
                    recommendation="Ledger-Eintrag entfernen oder ID-Quelle prüfen.",
                )
            )
    return out


def check_copilot_sections(workspace: Path) -> list[IntegrationFinding]:
    out: list[IntegrationFinding] = []
    text = _read(workspace / ".github/copilot-instructions.md")
    for section in _COPILOT_REQUIRED_SECTIONS:
        if section not in text:
            out.append(
                IntegrationFinding(
                    check="copilot_sections",
                    severity="error",
                    item=section,
                    detail="Pflicht-Abschnitt fehlt in copilot-instructions.md",
                    recommendation="Abschnitt wiederherstellen (CI parst diese Header).",
                )
            )
    if "[RELEASE_MUST]" not in text:
        out.append(
            IntegrationFinding(
                check="copilot_sections",
                severity="error",
                item="[RELEASE_MUST]",
                detail="Kein [RELEASE_MUST]-Header vorhanden",
                recommendation="Mindestens einen Release-Must-Punkt definieren.",
            )
        )
    return out


def check_verboten_linter_coverage(workspace: Path) -> list[IntegrationFinding]:
    out: list[IntegrationFinding] = []
    linter = _read(workspace / "scripts/aurik_verboten_linter.py")
    for vid in sorted(_verboten_ids(workspace)):
        if vid not in linter and vid not in _ledger_entries(workspace):
            out.append(
                IntegrationFinding(
                    check="verboten_linter_coverage",
                    severity="warning",
                    item=vid,
                    detail="VERBOTEN-ID ist weder im Linter hartkodiert noch im Ledger dokumentiert",
                    recommendation="Linter-Regel ergänzen (Regeländerung ⇒ Skript nachziehen) oder Ledger-Status setzen.",
                )
            )
    return out


def _run_gate(name: str, cmd: list[str], workspace: Path) -> IntegrationFinding | None:
    try:
        proc = subprocess.run(cmd, cwd=str(workspace), text=True, capture_output=True, check=False)
    except OSError as exc:
        return IntegrationFinding(
            check="enforce_gates",
            severity="error",
            item=name,
            detail=f"Gate nicht ausführbar: {exc}",
            recommendation="Skript-Abhängigkeiten prüfen.",
        )
    if proc.returncode != 0:
        detail = (proc.stdout or proc.stderr).strip().replace("\n", " | ")[:300]
        return IntegrationFinding(
            check="enforce_gates",
            severity="error",
            item=name,
            detail=f"Exit {proc.returncode}: {detail}",
            recommendation="Gate-Fehler beheben (Fehlerprotokoll sofort an Programmier-LLM).",
        )
    return None


def check_enforce_gates(workspace: Path) -> list[IntegrationFinding]:
    out: list[IntegrationFinding] = []
    py = str(workspace / ".venv_aurik/bin/python")
    gates = [
        ("release_must_coverage", [py, "scripts/release_must_coverage_check.py"]),
        ("id_registry", [py, "scripts/id_registry_check.py"]),
        ("crossfire_phases_0a", [py, "scripts/check_crossfire_phases.py"]),
    ]
    for name, cmd in gates:
        f = _run_gate(name, cmd, workspace)
        if f is not None:
            out.append(f)
    return out


def scan_spec_integration(workspace: Path) -> IntegrationResult:
    """Führt alle Integrations-Checks aus und liefert das Fehlerprotokoll."""
    workspace = Path(workspace).resolve()
    started = time.perf_counter()
    findings: list[IntegrationFinding] = []
    findings += check_normative_docs(workspace)
    findings += check_spec_references(workspace)
    findings += check_gebote_verboten_matrix(workspace)
    findings += check_copilot_sections(workspace)
    findings += check_verboten_linter_coverage(workspace)
    findings += check_enforce_gates(workspace)
    findings.sort(key=lambda f: (-_SEVERITY_RANK.get(f.severity, 0), f.check, f.item))
    totals = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        totals[f.severity] = totals.get(f.severity, 0) + 1
    return IntegrationResult(findings=findings, duration_s=round(time.perf_counter() - started, 3), totals=totals)


def write_json_report(path: Path, result: IntegrationResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "duration_s": result.duration_s,
        "totals": result.totals,
        "findings": [asdict(f) for f in result.findings],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_report(path: Path, result: IntegrationResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spec-Integrations-Fehlerprotokoll",
        "",
        f"- Erzeugt: {datetime.now().isoformat()}",
        f"- Fehler: **{result.totals.get('error', 0)}** · Warnungen: {result.totals.get('warning', 0)}",
        "",
    ]
    by_sev: dict[str, list[IntegrationFinding]] = {"error": [], "warning": [], "info": []}
    for f in result.findings:
        by_sev.setdefault(f.severity, []).append(f)
    for sev in ("error", "warning", "info"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        lines += [f"## {sev.upper()} ({len(items)})", ""]
        for f in items:
            lines.append(f"- **[{f.check}]** `{f.item}` — {f.detail}")
            lines.append(f"  - Behebung: {f.recommendation}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spec-Integrations-Scanner (Fehlerprotokoll)")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--json-out", default="audit/spec_integration_report.json")
    parser.add_argument("--md-out", default="audit/spec_integration_report.md")
    parser.add_argument("--fail-on", choices=("error", "warning", "none"), default="none")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    result = scan_spec_integration(workspace)

    json_out = workspace / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
    md_out = workspace / args.md_out if not Path(args.md_out).is_absolute() else Path(args.md_out)
    write_json_report(json_out, result)
    write_markdown_report(md_out, result)

    print(
        f"[spec-integration] errors={result.totals.get('error', 0)} "
        f"warnings={result.totals.get('warning', 0)} duration={result.duration_s}s"
    )
    for f in result.findings[:20]:
        print(f"  [{f.severity}] [{f.check}] {f.item} — {f.detail[:120]}")
    if len(result.findings) > 20:
        print(f"  … {len(result.findings) - 20} weitere Befunde in {md_out.name}")

    if args.fail_on == "error" and result.totals.get("error", 0) > 0:
        return 1
    if args.fail_on == "warning" and (result.totals.get("error", 0) + result.totals.get("warning", 0)) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
