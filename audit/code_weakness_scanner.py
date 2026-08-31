"""code_weakness_scanner.py — Statische Code-Schwachstellen-Prüfung (Watchdog-Erweiterung).

Scannt den Produktions-Code (backend/, denker/, Aurik10/, cli/) auf bekannte
Schwachstellen-Klassen und weist jeden Befund klar aus: Regel-ID, Schweregrad,
Datei:Zeile, Evidenz-Ausschnitt, Spec-Referenz und empfohlene Aktion.

Regelbasis (normative Kette, AGENTS.md §1/§3 — Zitate mit Quelle):
  - bridge_import_violation      §V4 (copilot-instructions.md)
  - dither_missing_int_conversion §V5 (copilot-instructions.md)
  - silent_fallback_no_log       §V6 (copilot-instructions.md)
  - bare_except                  Bug-Klassen (copilot-instructions.md, Silent Failure)
  - module_logger_missing        Logger-Pflicht (§III DSP, AGENTS.md §3)
  - nan_inf_guard_missing        §0a (copilot-instructions.md)
  - determinism_time_usage       §G5 (AGENTS.md §3 / copilot-instructions.md)
  - print_in_production          Logger-Pflicht (§III DSP, AGENTS.md §3)

Nutzung standalone:
    python audit/code_weakness_scanner.py --workspace . \
        --json-out audit/code_weakness_report.json \
        --md-out audit/code_weakness_report.md

Nutzung als Bibliothek (Live-Watchdog):
    from audit.code_weakness_scanner import scan_workspace, summarize_findings
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Iterable

_SEVERITY_RANK: dict[str, int] = {"critical": 3, "high": 2, "medium": 1, "low": 0}

_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", ".venv_aurik", "node_modules", "models", "temp_repro"}
)

# Aggregierte Regeln (Pro-Datei-Zählung statt pro Fund, um Logflut zu vermeiden).
_TIME_USAGE_MIN_COUNT = 2
_PRINT_MIN_COUNT = 3
_AGGREGATED_TOP_N = 10
_PER_FILE_AST_CAP = 3


@dataclass(frozen=True)
class WeaknessRule:
    """Definition einer Schwachstellen-Regel (klar ausweisbar)."""

    rule_id: str
    severity: str
    spec_ref: str
    title: str
    recommendation: str


RULES: dict[str, WeaknessRule] = {
    "bridge_import_violation": WeaknessRule(
        rule_id="bridge_import_violation",
        severity="critical",
        spec_ref="§V4 (copilot-instructions.md)",
        title="UI/Frontend importiert backend/core direkt statt über backend/api/bridge.py",
        recommendation=(
            "Import über backend.api.bridge umleiten; Denker-Schicht ist ausgenommen. (§V4 Bridge-Bypass-Verbot)"
        ),
    ),
    "dither_missing_int_conversion": WeaknessRule(
        rule_id="dither_missing_int_conversion",
        severity="high",
        spec_ref="§V5 (copilot-instructions.md)",
        title="Integer-Konversion ohne Dither (bit_depth < 32)",
        recommendation=(
            "POW-r Type 3 (primär) oder TPDF (Fallback) vor der Konversion anwenden; "
            "kein nacktes astype(np.int16). (§V5 Truncation-ohne-Dither-Verbot)"
        ),
    ),
    "silent_fallback_no_log": WeaknessRule(
        rule_id="silent_fallback_no_log",
        severity="high",
        spec_ref="§V6 (copilot-instructions.md)",
        title="Fallback/Return in except-Block ohne Logging",
        recommendation=(
            "logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie "
            "stumm bleiben. (§V6 Silent-Failure-Verbot)"
        ),
    ),
    "bare_except": WeaknessRule(
        rule_id="bare_except",
        severity="medium",
        spec_ref="Bug-Klassen (copilot-instructions.md) — Silent Failure",
        title="Bare except: fängt auch SystemExit/KeyboardInterrupt und verschluckt Fehler",
        recommendation="Exception-Typ explizit benennen und den Fehler loggen.",
    ),
    "module_logger_missing": WeaknessRule(
        rule_id="module_logger_missing",
        severity="medium",
        spec_ref="Logger-Pflicht (§III DSP, AGENTS.md §3)",
        title="Modul mit Fehlerpfaden hat keinen Logger (logging.getLogger(__name__))",
        recommendation=("logger = logging.getLogger(__name__) ergänzen; Fehlerpfade müssen logbar sein."),
    ),
    "nan_inf_guard_missing": WeaknessRule(
        rule_id="nan_inf_guard_missing",
        severity="medium",
        spec_ref="§0a (copilot-instructions.md)",
        title="Phase ohne NaN/Inf-Schutz (isfinite/nan_to_num/isnan fehlt komplett)",
        recommendation=("NaN/Inf-Guard in die Phase einbauen (§0a: Schutz in jeder Phase)."),
    ),
    "determinism_time_usage": WeaknessRule(
        rule_id="determinism_time_usage",
        severity="low",
        spec_ref="§G5 (AGENTS.md §3 / copilot-instructions.md)",
        title="time.time() im Produktions-Code — Determinismus-Risiko",
        recommendation=(
            "Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen "
            "time.monotonic()/perf_counter() verwenden. (§G5 Determinism)"
        ),
    ),
    "print_in_production": WeaknessRule(
        rule_id="print_in_production",
        severity="low",
        spec_ref="Logger-Pflicht (§III DSP, AGENTS.md §3)",
        title="print() in Produktions-Modul statt Logger",
        recommendation="Ausgaben über logging umleiten (Logger-Pflicht).",
    ),
}

_BRIDGE_IMPORT_RX = re.compile(
    r"^\s*(?:from\s+backend\.core\b.*\bimport\b|import\s+backend\.core\b|from\s+backend\s+import\s+core\b)",
    re.MULTILINE,
)
_INT_CAST_RX = re.compile(r"\bastype\(\s*(?:dtype\s*=\s*)?(?:(?:np|numpy)\.)?[\"']?(int8|int16|int32)[\"']?\)")
_DITHER_CONTEXT_TOKENS: tuple[str, ...] = ("dither", "tpdf", "powr", "pow-r")
_TIME_USAGE_RX = re.compile(r"\btime\.time\(\s*\)")
_PRINT_RX = re.compile(r"(?<![\w.])print\(")
_NAN_GUARD_TOKENS: tuple[str, ...] = ("isfinite", "isnan", "nan_to_num", "isinf")


@dataclass
class WeaknessFinding:
    """Ein klar ausgewiesener Schwachstellen-Befund."""

    rule_id: str
    severity: str
    file: str  # relativer POSIX-Pfad zur Workspace-Root
    line: int
    evidence: str
    spec_ref: str
    recommendation: str


@dataclass
class ScanResult:
    """Ergebnis eines kompletten Schwachstellen-Scans."""

    findings: list[WeaknessFinding] = field(default_factory=list)
    files_scanned: int = 0
    duration_s: float = 0.0
    per_rule_counts: dict[str, int] = field(default_factory=dict)
    # Transparenz: Befunde, die unter Meldeschwellen fielen oder durch
    # Kappungen (AST-Cap, Aggregations-Top-N, max_findings) nicht ausgewiesen
    # wurden. Schlüssel: rule_id bzw. "truncated_max_findings".
    suppressed: dict[str, int] = field(default_factory=dict)


def _is_test_path(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    return "tests" in parts or any(p.startswith("test_") for p in parts)


def _is_vendor_path(rel_posix: str) -> bool:
    return any(part.startswith("_vendor") for part in rel_posix.split("/"))


def iter_python_files(workspace: Path) -> Iterable[tuple[Path, str]]:
    """Yieldet (Pfad, relativer POSIX-Pfad) für alle zu prüfenden Python-Dateien."""
    for path in sorted(workspace.rglob("*.py")):
        rel = path.relative_to(workspace).as_posix()
        if any(part in _SKIP_DIRS or part.startswith(".venv") for part in Path(rel).parts):
            continue
        yield path, rel


def _evidence(lines: list[str], line_no: int) -> str:
    idx = max(0, min(line_no - 1, len(lines) - 1))
    snippet = lines[idx].strip()
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    return snippet or f"<Zeile {line_no}>"


def _has_logging_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            root = child.func.value
            name = ""
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {"logger", "logging", "log"}:
                return True
    return False


def _handler_returns_value(handler: ast.ExceptHandler) -> bool:
    for child in ast.walk(handler):
        if isinstance(child, ast.Return) and child.value is not None:
            return True
        if isinstance(child, ast.Raise):
            return False
    return False


def _check_bridge_imports(rel_posix: str, text: str, lines: list[str]) -> list[WeaknessFinding]:
    """§V4: UI/Frontend (Aurik10/, cli/) darf backend/core nie direkt importieren."""
    if not rel_posix.startswith(("Aurik10/", "cli/")):
        return []
    rule = RULES["bridge_import_violation"]
    findings: list[WeaknessFinding] = []
    for m in _BRIDGE_IMPORT_RX.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel_posix,
                line=line_no,
                evidence=_evidence(lines, line_no),
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
    return findings


def _check_dither(rel_posix: str, text: str, lines: list[str]) -> list[WeaknessFinding]:
    """§V5: Integer-Konversion (bit_depth < 32) ohne Dither im Kontext."""
    if not rel_posix.startswith(("backend/", "denker/")):
        return []
    rule = RULES["dither_missing_int_conversion"]
    findings: list[WeaknessFinding] = []
    for m in _INT_CAST_RX.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        window = " ".join(lines[max(0, line_no - 7) : min(len(lines), line_no + 6)]).lower()
        if any(token in window for token in _DITHER_CONTEXT_TOKENS):
            continue
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel_posix,
                line=line_no,
                evidence=_evidence(lines, line_no),
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
    return findings


def _check_ast_rules(
    rel_posix: str,
    tree: ast.AST,
    lines: list[str],
    suppressed: dict[str, int] | None = None,
) -> list[WeaknessFinding]:
    """§V6 Silent-Fallbacks + bare except (AST-basiert, pro Datei gedeckelt).

    Befunde jenseits des Pro-Datei-Caps werden nicht verschwiegen, sondern
    im ``suppressed``-Zähler ausgewiesen (Transparenz statt Logflut).
    """
    if not rel_posix.startswith(("backend/", "denker/")):
        return []
    findings: list[WeaknessFinding] = []
    per_file_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            rule = RULES["bare_except"]
        elif _has_logging_call(node) or not _handler_returns_value(node):
            continue
        else:
            rule = RULES["silent_fallback_no_log"]
        if per_file_count >= _PER_FILE_AST_CAP:
            if suppressed is not None:
                suppressed[rule.rule_id] = suppressed.get(rule.rule_id, 0) + 1
            continue
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel_posix,
                line=node.lineno,
                evidence=_evidence(lines, node.lineno),
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
        per_file_count += 1
    return findings


def _check_module_logger(rel_posix: str, text: str) -> WeaknessFinding | None:
    """Logger-Pflicht: Fehlerpfade ohne getLogger() im Modul."""
    if not rel_posix.startswith(("backend/", "denker/")):
        return None
    has_error_path = "except Exception" in text or "subprocess." in text
    if not has_error_path:
        return None
    if "getLogger(" in text:
        return None
    rule = RULES["module_logger_missing"]
    return WeaknessFinding(
        rule_id=rule.rule_id,
        severity=rule.severity,
        file=rel_posix,
        line=1,
        evidence="Modul enthält Fehlerpfade, aber kein logging.getLogger()",
        spec_ref=rule.spec_ref,
        recommendation=rule.recommendation,
    )


def _check_phase_nan_guard(rel_posix: str, text: str) -> WeaknessFinding | None:
    """§0a: Jede Phase braucht NaN/Inf-Schutz."""
    if not re.match(r"backend/core/phases/phase_[0-9]+_", rel_posix):
        return None
    if "numpy" not in text and "np." not in text:
        return None
    if any(token in text for token in _NAN_GUARD_TOKENS):
        return None
    rule = RULES["nan_inf_guard_missing"]
    return WeaknessFinding(
        rule_id=rule.rule_id,
        severity=rule.severity,
        file=rel_posix,
        line=1,
        evidence="Phase nutzt numpy, aber kein isfinite/nan_to_num/isnan/isinf",
        spec_ref=rule.spec_ref,
        recommendation=rule.recommendation,
    )


def scan_workspace(workspace: Path, *, max_findings: int = 200) -> ScanResult:
    """Führt den kompletten Schwachstellen-Scan über die Workspace aus.

    Die Sortierung erfolgt nach Schweregrad (critical zuerst), dann Datei/Zeile,
    damit bei Begrenzung auf ``max_findings`` niemals kritische Befunde verloren gehen.
    """
    workspace = Path(workspace).resolve()
    started = time.perf_counter()
    findings: list[WeaknessFinding] = []
    files_scanned = 0
    time_usage_counts: dict[str, int] = {}
    print_counts: dict[str, int] = {}
    # Alle Treffer (auch unter Schwelle), um die unterdrückte Anzahl ehrlich
    # auszuweisen statt still zu verschweigen (Transparenz).
    time_hits_all: dict[str, int] = {}
    print_hits_all: dict[str, int] = {}
    suppressed: dict[str, int] = {}

    for path, rel in iter_python_files(workspace):
        if _is_test_path(rel) or _is_vendor_path(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files_scanned += 1
        lines = text.splitlines()

        findings.extend(_check_bridge_imports(rel, text, lines))
        findings.extend(_check_dither(rel, text, lines))

        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            findings.extend(_check_ast_rules(rel, tree, lines, suppressed))

        finding = _check_module_logger(rel, text)
        if finding is not None:
            findings.append(finding)
        finding = _check_phase_nan_guard(rel, text)
        if finding is not None:
            findings.append(finding)

        if rel.startswith(("backend/core/", "denker/")):
            time_hits = len(_TIME_USAGE_RX.findall(text))
            if time_hits >= 1:
                time_hits_all[rel] = time_hits
            if time_hits >= _TIME_USAGE_MIN_COUNT:
                time_usage_counts[rel] = time_hits
            print_hits = len(_PRINT_RX.findall(text))
            if rel.startswith("backend/core/"):
                if print_hits >= 1:
                    print_hits_all[rel] = print_hits
                if print_hits >= _PRINT_MIN_COUNT:
                    print_counts[rel] = print_hits

    # Aggregierte Befunde (Pro-Datei-Zählung, Top-N — keine Logflut).
    for rule_id in ("determinism_time_usage", "print_in_production"):
        counts = time_usage_counts if rule_id == "determinism_time_usage" else print_counts
        hits_all = time_hits_all if rule_id == "determinism_time_usage" else print_hits_all
        rule = RULES[rule_id]
        top_files = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_AGGREGATED_TOP_N]
        reported_occurrences = 0
        for rel, count in top_files:
            reported_occurrences += count
            findings.append(
                WeaknessFinding(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    file=rel,
                    line=1,
                    evidence=f"{count} Vorkommen von {'time.time()' if rule_id == 'determinism_time_usage' else 'print()'}",
                    spec_ref=rule.spec_ref,
                    recommendation=rule.recommendation,
                )
            )
        # Transparenz: Treffer unter Schwelle + Treffer jenseits Top-N ausweisen.
        _suppressed_occurrences = sum(hits_all.values()) - reported_occurrences
        if _suppressed_occurrences > 0:
            suppressed[rule_id] = suppressed.get(rule_id, 0) + _suppressed_occurrences

    findings.sort(key=lambda f: (-_SEVERITY_RANK.get(f.severity, 0), f.rule_id, f.file, f.line))
    if len(findings) > max_findings:
        suppressed["truncated_max_findings"] = len(findings) - max_findings
        findings = findings[:max_findings]

    per_rule_counts: dict[str, int] = {}
    for finding in findings:
        per_rule_counts[finding.rule_id] = per_rule_counts.get(finding.rule_id, 0) + 1

    return ScanResult(
        findings=findings,
        files_scanned=files_scanned,
        duration_s=round(time.perf_counter() - started, 3),
        per_rule_counts=dict(sorted(per_rule_counts.items())),
        suppressed=dict(sorted(suppressed.items())),
    )


def summarize_findings(result: ScanResult, *, top_n: int = 5) -> dict[str, Any]:
    """Kompakte Zusammenfassung für Live-Snapshots (Watchdog-Payload)."""
    by_severity = dict.fromkeys(("critical", "high", "medium", "low"), 0)
    for finding in result.findings:
        if finding.severity in by_severity:
            by_severity[finding.severity] += 1
    return {
        "total": len(result.findings),
        **by_severity,
        "files_scanned": result.files_scanned,
        "top_findings": [
            {"rule_id": f.rule_id, "severity": f.severity, "file": f.file, "line": f.line}
            for f in result.findings[:top_n]
        ],
    }


def write_json_report(path: Path, result: ScanResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "files_scanned": result.files_scanned,
        "duration_s": result.duration_s,
        "per_rule_counts": result.per_rule_counts,
        "suppressed": dict(result.suppressed),
        "total_findings": len(result.findings),
        "findings": [asdict(f) for f in result.findings],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_report(path: Path, result: ScanResult) -> None:
    """Menschlich klarer Report: Befunde nach Schweregrad gruppiert."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_severity: dict[str, list[WeaknessFinding]] = {"critical": [], "high": [], "medium": [], "low": []}
    for finding in result.findings:
        by_severity.setdefault(finding.severity, []).append(finding)

    lines: list[str] = [
        "# Code-Schwachstellen-Report (Watchdog)",
        "",
        f"- Erzeugt: {datetime.now().isoformat()}",
        f"- Geprüfte Dateien: {result.files_scanned} (Dauer: {result.duration_s}s)",
        f"- Befunde gesamt: **{len(result.findings)}**",
    ]
    for sev in ("critical", "high", "medium", "low"):
        lines.append(f"  - {sev}: {len(by_severity.get(sev, []))}")
    if result.per_rule_counts:
        lines.append("- Pro Regel: " + ", ".join(f"{k}={v}" for k, v in result.per_rule_counts.items()))
    if result.suppressed:
        lines.append(
            "- Unterdrückte Befunde (unter Schwelle/Kappung, bewusst sichtbar): "
            + ", ".join(f"{k}={v}" for k, v in sorted(result.suppressed.items()))
        )
        lines.append(
            "  (Schwellen: AST-Cap 3/Datei, time.time ≥ 2, print ≥ 3, Top-N 10, max_findings — "
            "unterdrückt heißt nicht: nicht vorhanden.)"
        )

    for sev in ("critical", "high", "medium", "low"):
        items = by_severity.get(sev, [])
        if not items:
            continue
        lines += ["", f"## {sev.upper()} ({len(items)})", ""]
        for finding in items:
            rule = RULES.get(finding.rule_id)
            title = rule.title if rule else ""
            lines.append(f"- `{finding.file}:{finding.line}` — **{finding.rule_id}** ({finding.spec_ref})")
            if title:
                lines.append(f"  - {title}")
            lines.append(f"  - Evidenz: `{finding.evidence}`")
            lines.append(f"  - Empfehlung: {finding.recommendation}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Statischer Code-Schwachstellen-Scan (Watchdog)")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--json-out", default="audit/code_weakness_report.json")
    parser.add_argument("--md-out", default="audit/code_weakness_report.md")
    parser.add_argument("--max-findings", type=int, default=200)
    parser.add_argument(
        "--fail-on",
        choices=("critical", "high", "none"),
        default="none",
        help="Exit 1, wenn Befunde der angegebenen Schwere existieren (CI-Modus).",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    result = scan_workspace(workspace, max_findings=args.max_findings)

    json_out = workspace / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
    md_out = workspace / args.md_out if not Path(args.md_out).is_absolute() else Path(args.md_out)
    write_json_report(json_out, result)
    write_markdown_report(md_out, result)

    summary = summarize_findings(result)
    _supp_total = sum(result.suppressed.values())
    print(
        f"[code-weakness-scan] files={result.files_scanned} "
        f"findings={summary['total']} critical={summary['critical']} high={summary['high']} "
        f"medium={summary['medium']} low={summary['low']} "
        f"suppressed={_supp_total} duration={result.duration_s}s"
    )
    for finding in result.findings[:15]:
        print(f"  [{finding.severity}] {finding.file}:{finding.line} {finding.rule_id}")
    if len(result.findings) > 15:
        print(f"  … {len(result.findings) - 15} weitere Befunde in {json_out.name}/{md_out.name}")

    if args.fail_on == "critical" and summary["critical"] > 0:
        return 1
    if args.fail_on == "high" and (summary["critical"] > 0 or summary["high"] > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
