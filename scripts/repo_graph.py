#!/usr/bin/env python3
"""Repo-Graph — Import-Graph, Top-Level-Symbole und Entry-Points über `ast`.

Maschinenlesbare Repo-Karte für Agenten: „Repo → Karte → relevante Dateien"
statt „alles ins Kontextfenster". Konsolidiert die statischen Audits
`scripts/audit_bridge_coverage.py` (Bridge-Verbot §V4 (copilot-instructions.md))
und `scripts/audit_silent_dead_imports.py` (tote Import-Namen in
try/except-ImportError-Blöcken) und ergänzt Importer-Analyse,
Symbol-Index und Registry-Status-Prüfung: FORBIDDEN/ARCHIVED-Dateien
dürfen nicht importiert werden. Nur Standardbibliothek, Python ≥ 3.10.

Betriebsarten:
  python scripts/repo_graph.py                 # Report (Karte + Orphans)
  python scripts/repo_graph.py --check         # Bridge/Dead-Import/Status (Exit 1)
  python scripts/repo_graph.py --duplicates    # Symbol-Kollisionen (Hook D, fail-closed)
  python scripts/repo_graph.py --write-json    # .github/repo_graph.json erzeugen
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / ".github" / "FILE_REGISTRY.md"
OUT_JSON = ROOT / ".github" / "repo_graph.json"

CODE_DIRS = ["backend", "plugins", "denker", "Aurik10", "cli", "scripts"]
SKIP_DIR_PARTS = {
    ".venv_aurik",
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}
FRONTEND_DIRS = ["Aurik10", "cli"]
ALLOWED_DIRECT_IMPORTS = {
    # Startup-pflicht: vor Bridge-Initialisierung
    "Aurik10/main.py": ["backend.core.ml_device_manager"],
    # UI-spezifische Business-Logik (kein DSP-Core, nur UI-Helfer)
    "Aurik10/ui/modern_window.py": ["backend.core.donation_reminder"],
    # CLI-spezifisch: kein GUI-Kontext
    "cli/aurik_debug.py": ["backend.core.unified_restorer_v3", "backend.core.pipeline_trace"],
    "cli/aurik_cli.py": ["backend.core.cd_noise_profile"],
}
_CATCH_NAMES = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
STATUS_ENUM = frozenset({"ACTIVE", "DEPRECATED", "MIGRATING", "GENERATED", "TEST_ONLY", "ARCHIVED", "FORBIDDEN"})
NO_IMPORT_STATUSES = frozenset({"ARCHIVED", "FORBIDDEN"})
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:[-_ ]?(?:old|new|legacy|backup|final|deprecated|copy|alt|v?\d+))?$")


def _skip_path(path: Path) -> bool:
    """True für Verzeichnisse, die nicht indiziert werden (Vendor-Code, Caches)."""
    return any(part in SKIP_DIR_PARTS or part.startswith("_vendor") for part in path.parts)


def collect_py_files() -> list[Path]:
    """Sammelt alle indizierbaren Python-Dateien in CODE_DIRS (sortiert)."""
    files: list[Path] = []
    for name in CODE_DIRS:
        start = ROOT / name
        if not start.is_dir():
            continue
        for py in start.rglob("*.py"):
            if _skip_path(py):
                continue
            files.append(py)
    return sorted(files)


def _module_file(module_name: str) -> Path | None:
    """Löst einen absoluten Modulnamen auf eine repo-interne Datei auf."""
    parts = module_name.split(".")
    as_file = ROOT.joinpath(*parts).with_suffix(".py")
    if as_file.is_file() and not _skip_path(as_file):
        return as_file
    as_pkg = ROOT.joinpath(*parts, "__init__.py")
    if as_pkg.is_file() and not _skip_path(as_pkg):
        return as_pkg
    return None


def _relative_target(importer: Path, module: str | None, level: int) -> Path | None:
    """Löst einen relativen Import (level ≥ 1) relativ zur importer-Datei auf."""
    base = importer.parent
    for _ in range(level - 1):
        base = base.parent
    target = base.joinpath(*module.split(".")) if module else base
    for cand in (target.with_suffix(".py"), target / "__init__.py"):
        if cand.is_file() and not _skip_path(cand):
            return cand
    return None


def _is_main_block(tree: ast.Module) -> bool:
    """True, wenn das Modul einen `if __name__ == "__main__":`-Block hat."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            if node.left.id == "__name__":
                return True
    return False


def parse_py(path: Path) -> tuple[list[str], list[str], list[str], list[str], bool]:
    """Liefert (import_ziele, klassen, funktionen, symbole, is_entry) für eine Datei.

    import_ziele sind repo-interne Zielpfade (relativ). klassen/funktionen/
    symbole enthalten nur öffentliche (nicht `_`-präfixierte) Top-Level-Namen;
    symbole ist die Vereinigung inkl. Modul-Zuweisungen.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return [], [], [], [], False
    import_targets: list[str] = []
    classes: list[str] = []
    functions: list[str] = []
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                classes.append(node.name)
                symbols.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                functions.append(node.name)
                symbols.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    symbols.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                symbols.append(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                resolved = _module_file(node.module or "")
            else:
                resolved = _relative_target(path, node.module, node.level)
            if resolved is not None:
                import_targets.append(str(resolved.relative_to(ROOT)))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _module_file(alias.name)
                if resolved is not None:
                    import_targets.append(str(resolved.relative_to(ROOT)))
    return (
        sorted(set(import_targets)),
        sorted(set(classes)),
        sorted(set(functions)),
        sorted(set(symbols)),
        _is_main_block(tree),
    )


def parse_file_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, str]]:
    """Liest die `## Dateien`-Tabelle der FILE_REGISTRY (Pfad → Felder).

    Felder: status, domain, canonical (bool), ersetzt, grund.
    """
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    in_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
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
        if len(cells) < 5:
            continue
        rows[cells[0]] = {
            "status": cells[1].upper(),
            "domain": cells[2],
            "canonical": cells[3].lower().startswith("ja"),
            "ersetzt": "" if cells[4] == "—" else cells[4],
            "grund": cells[5] if len(cells) > 5 else "",
        }
    return rows


def build_graph(registry: dict[str, dict[str, str]]) -> dict[str, dict]:
    """Baut die Repo-Karte: Datei → imports/imported_by/klassen/symbole/…"""
    graph: dict[str, dict] = {}
    for py in collect_py_files():
        rel = str(py.relative_to(ROOT))
        imports, classes, functions, symbols, is_entry = parse_py(py)
        reg = registry.get(rel, {})
        graph[rel] = {
            "imports": imports,
            "imported_by": [],
            "classes": classes,
            "functions": functions,
            "symbols": symbols,
            "is_entry": is_entry,
            "domain": reg.get("domain", ""),
            "status": reg.get("status", ""),
        }
    for rel, info in graph.items():
        for target in info["imports"]:
            if target in graph:
                graph[target]["imported_by"].append(rel)
    return graph


def find_bridge_violations() -> list[dict]:
    """Direkte Core-Imports aus Frontend/CLI — identische Regeln wie audit_bridge_coverage."""
    violations: list[dict] = []
    pattern = re.compile(r"from (backend\.core\.\S+) import")
    for frontend_dir in FRONTEND_DIRS:
        for py_file in (ROOT / frontend_dir).rglob("*.py"):
            if _skip_path(py_file):
                continue
            rel = str(py_file.relative_to(ROOT))
            content = py_file.read_text(encoding="utf-8", errors="replace")
            for match in pattern.finditer(content):
                import_path = match.group(1)
                if "bridge" in import_path:
                    continue
                if import_path in ALLOWED_DIRECT_IMPORTS.get(rel, []):
                    continue
                violations.append(
                    {
                        "file": rel,
                        "import": import_path,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )
    return violations


def _top_level_names(path: Path) -> set[str] | None:
    """Sammelt importierbare Top-Level-Namen einer Datei (wie audit_silent_dead_imports)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None
    names: set[str] = set()

    def _collect(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.If):
                _collect(node.body)
                _collect(node.orelse)
            elif isinstance(node, ast.Try):
                _collect(node.body)
                for handler in node.handlers:
                    _collect(handler.body)
                _collect(node.orelse)
                _collect(node.finalbody)

    _collect(tree.body)
    return names


def _catches_import_error(handlers: list[ast.ExceptHandler]) -> bool:
    for handler in handlers:
        if handler.type is None:
            return True
        if isinstance(handler.type, ast.Name) and handler.type.id in _CATCH_NAMES:
            return True
        if isinstance(handler.type, ast.Tuple) and any(
            isinstance(e, ast.Name) and e.id in _CATCH_NAMES for e in handler.type.elts
        ):
            return True
    return False


def find_dead_imports() -> list[dict]:
    """Tote repo-interne Import-Namen in try/except-ImportError-Blöcken."""
    findings: list[dict] = []
    for path in collect_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if not _catches_import_error(node.handlers):
                continue
            for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if not (isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0):
                    continue
                mod_path = _module_file(stmt.module)
                if mod_path is None:
                    continue
                target_names = _top_level_names(mod_path)
                if target_names is None:
                    continue
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    if alias.name in target_names:
                        continue
                    # §v10.x Submodul-Import: `from <pkg> import <modul>` — wenn
                    # <pkg>.<modul>.py existiert, ist das KEIN toter Import
                    # (Befund 2026-08-22: False Positives für calibrated_constants,
                    # perceptual_validator, mert_plugin — Module existieren).
                    if _module_file(f"{stmt.module}.{alias.name}") is not None:
                        continue
                    findings.append(
                        {
                            "file": str(path.relative_to(ROOT)),
                            "line": stmt.lineno,
                            "module": stmt.module,
                            "name": alias.name,
                        }
                    )
    return findings


def find_forbidden_imports(graph: dict[str, dict], registry: dict[str, dict[str, str]]) -> list[dict]:
    """Verstöße: eine Datei importiert eine FORBIDDEN/ARCHIVED-Datei."""
    forbidden = {rel for rel, entry in registry.items() if entry["status"] in NO_IMPORT_STATUSES}
    if not forbidden:
        return []
    violations: list[dict] = []
    for rel, info in graph.items():
        for target in info["imports"]:
            if target in forbidden:
                violations.append({"file": rel, "import": target})
    return violations


def find_duplicate_symbols(graph: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Liefert (errors, warnings) für parallele Implementierungen.

    ERROR: gleicher öffentlicher Klassen-Name in zwei ACTIVE-Dateien
    derselben Domain. WARN: Funktions-Kollisionen sowie Suffix-Varianten
    (payment.py / payment_v2.py), bei denen die Basis-Datei existiert.
    """
    errors: list[str] = []
    warnings: list[str] = []

    by_domain: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for rel, info in graph.items():
        if info["status"] == "ACTIVE" and info["domain"]:
            by_domain[info["domain"]].append((rel, info["classes"]))

    for domain, files in sorted(by_domain.items()):
        seen_classes: dict[str, str] = {}
        seen_functions: dict[str, str] = {}
        for rel, classes in sorted(files):
            info = graph[rel]
            for name in classes:
                if name in seen_classes:
                    errors.append(
                        f"Klassen-Duplikat: `{name}` in {seen_classes[name]} und {rel} "
                        f"(Domain {domain}) — eine Implementierung muss DEPRECATED/replaces werden"
                    )
                else:
                    seen_classes[name] = rel
            functions = set(info["functions"]) - {"main"}
            for name in sorted(functions):
                if name in seen_functions and name not in seen_classes:
                    warnings.append(
                        f"Funktions-Kollision: `{name}` in {seen_functions[name]} und {rel} (Domain {domain})"
                    )
                else:
                    seen_functions[name] = rel

    stem_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel in graph:
        stem = Path(rel).stem
        match = _SUFFIX_RE.fullmatch(stem)
        base = match.group("base") if match else stem
        if base != stem:
            stem_map[(str(Path(rel).parent), base)].append(rel)
    for (parent, base), rels in sorted(stem_map.items()):
        base_rel = f"{parent}/{base}.py"
        if base_rel not in graph:
            continue  # ohne Basis-Datei kein Hinweis (vermeidet phase_01/…-Rauschen)
        warnings.append(
            f"Suffix-Variante(n) zu {base_rel}: {', '.join(sorted(rels))} — "
            f"parallele Implementierung oder legitime Variante? Prüfen."
        )
    return errors, warnings


def _print_check() -> int:
    issues = 0
    for v in find_bridge_violations():
        issues += 1
        print(f"BRIDGE {v['file']}:{v['line']} → {v['import']}")
    for d in find_dead_imports():
        issues += 1
        print(f"DEAD   {d['file']}:{d['line']}: `{d['name']}` fehlt in `{d['module']}`")
    registry = parse_file_registry()
    graph = build_graph(registry)
    for f in find_forbidden_imports(graph, registry):
        issues += 1
        print(f"STATUS {f['file']} importiert {f['import']} (FORBIDDEN/ARCHIVED)")
    if issues:
        print(f"{issues} Verstoß/Vorstöße gefunden (Bridge/Dead-Import/Status).")
        return 1
    print("Keine Verstöße (Bridge/Dead-Import/Status).")
    return 0


def _print_duplicates() -> int:
    registry = parse_file_registry()
    graph = build_graph(registry)
    errors, warnings = find_duplicate_symbols(graph)
    for w in warnings:
        print(f"WARNUNG: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print(f"{len(errors)} Symbol-Duplikat(e) — fail-closed.")
        return 1
    print("Keine Klassen-Duplikate zwischen ACTIVE-Dateien gefunden.")
    return 0


def _print_report() -> int:
    registry = parse_file_registry()
    graph = build_graph(registry)
    total = len(graph)
    orphans = sorted(rel for rel, info in graph.items() if not info["imported_by"] and not info["is_entry"])
    registered = len(registry)
    print(f"Repo-Graph: {total} Dateien indiziert ({', '.join(CODE_DIRS)}), {registered} registriert.")
    print(f"Dateien ohne Importer und ohne Entry-Point: {len(orphans)} (advisory)")
    for rel in orphans[:20]:
        print(f"  - {rel}")
    if len(orphans) > 20:
        print(f"  … und {len(orphans) - 20} weitere")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Repo-Graph: Import-Graph, Symbole, Checks.")
    parser.add_argument("--check", action="store_true", help="Bridge/Dead-Import/Status (Exit 1 bei Verstößen)")
    parser.add_argument(
        "--duplicates",
        action="store_true",
        help="Symbol-Duplikate prüfen (Exit 1 bei Klassen-Kollisionen)",
    )
    parser.add_argument("--write-json", action="store_true", help=".github/repo_graph.json schreiben")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()

    if args.write_json:
        registry = parse_file_registry(args.registry)
        graph = build_graph(registry)
        OUT_JSON.write_text(
            json.dumps({"generated_by": "scripts/repo_graph.py", "files": graph}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Graphen geschrieben: {OUT_JSON.relative_to(ROOT)} ({len(graph)} Dateien)")
        return 0
    if args.check:
        return _print_check()
    if args.duplicates:
        return _print_duplicates()
    return _print_report()


if __name__ == "__main__":
    sys.exit(main())
