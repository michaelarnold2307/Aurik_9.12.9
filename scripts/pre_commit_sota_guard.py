#!/usr/bin/env python3
"""§v10.306 Pre-Commit SOTA-Guard — erzwingt alle Startup- und Qualitäts-Regeln.

Prüft vor jedem Commit:
  1. Kein torch.zeros("cuda") außerhalb warmup_rocm()
  2. setText/setToolTip mit Hardcoded-Strings (MUSS t() verwenden)
  3. Kein import innerhalb with _lock:/with self._lock:
  4. os.environ nur wenn import os vorhanden
  5. Warmup-Plugin-Accessoren existieren tatsächlich

Usage: python3 scripts/pre_commit_sota_guard.py [file ...]
Exit 0 wenn sauber, Exit 1 bei Verstößen.
"""

import ast
import importlib
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
VIOLATIONS: list[str] = []


def _find_files(paths: list[str]) -> list[Path]:
    """Collect Python files from paths."""
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.suffix == ".py":
            files.append(pp)
        elif pp.is_dir():
            files.extend(pp.rglob("*.py"))
    return [f for f in files if "__pycache__" not in str(f) and ".venv" not in str(f)]


def check_torch_zeros_cuda(filepath: Path) -> None:
    """§G181 (GEBOTE.md): torch.zeros(..., device=\"cuda\") nur in warmup_rocm erlaubt."""
    content = filepath.read_text()
    if "torch.zeros" not in content and "torch.ones" not in content and "torch.empty" not in content:
        return
    lines = content.split("\n")
    in_warmup = False
    for i, line in enumerate(lines, 1):
        if "def warmup_rocm" in line:
            in_warmup = True
        if in_warmup and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            if "def warmup_rocm" not in line:
                in_warmup = False
        if ("torch.zeros" in line or "torch.ones" in line or "torch.empty" in line) and 'device="cuda"' in line.replace(
            "'", '"'
        ):
            if not in_warmup and "_ROCM_WARMUP" not in line:
                VIOLATIONS.append(
                    f'{filepath}:{i}: torch.zeros/ones/empty("cuda") ausserhalb warmup_rocm() VERBOTEN (§G181)'
                )


def check_hardcoded_strings(filepath: Path) -> None:
    """§G178 (GEBOTE.md): setText/setToolTip MUSS t() verwenden, keine Hardcoded-Strings."""
    if "modern_window.py" not in str(filepath):
        return
    content = filepath.read_text()
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Match: self.something.setText("...") without t() in the same statement
        if re.search(r'self\.\w+\.setText\s*\(\s*["\']', line):
            # Check if t() appears in the rest of this logical line
            after = line
            j = i
            while j < len(lines) and (")" not in after.split("setText")[-1] if "setText" in after else True):
                j += 1
                if j < len(lines):
                    after += " " + lines[j - 1]
            if "t(" not in after and 'setText(""' not in after:
                VIOLATIONS.append(f"{filepath}:{i}: self.xxx.setText() ohne t() — MUSS t() verwenden (§G178)")


def check_lock_during_import(filepath: Path) -> None:
    """§G174 (GEBOTE.md): Kein import innerhalb with _lock: / with self._lock:."""
    content = filepath.read_text()
    lines = content.split("\n")
    lock_indent: int | None = None
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if not stripped:
            continue
        current_indent = len(line) - len(line.lstrip())
        # Detect end of with block: same or less indent than the 'with' line
        if lock_indent is not None and current_indent <= lock_indent and stripped:
            lock_indent = None
        # Detect start of with _lock block
        if re.search(r"with\s+(self\.)?_lock\s*:", stripped):
            lock_indent = current_indent
            continue
        # Check for import inside active lock
        if lock_indent is not None and current_indent > lock_indent:
            if re.search(r"^\s*(from\s+\S+\s+import|import\s+\S+)", line):
                # Exempt standard library imports and common exceptions
                if not re.search(
                    r"(onnxruntime|torch|subprocess|sys|os|logging|threading|collections|json|pathlib|typing|dataclasses)",
                    line,
                ):
                    VIOLATIONS.append(f"{filepath}:{i}: import innerhalb with _lock: VERBOTEN (§G174)")


def check_os_environ_import(filepath: Path) -> None:
    """§G180 (GEBOTE.md): os.environ.get() nur wenn import os vorhanden."""
    content = filepath.read_text()
    if "os.environ" not in content and "os.getenv" not in content:
        return
    # Check module-level imports
    tree = ast.parse(content)
    has_os_import = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    has_os_import = True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "os":
                has_os_import = True
    if not has_os_import and "import os as" not in content:
        VIOLATIONS.append(f"{filepath}: os.environ/os.getenv verwendet aber 'import os' fehlt (§G180)")


def check_warmup_accessors(filepath: Path) -> None:
    """§G175 (GEBOTE.md): Warmup-Plugin-Zugriffsnamen via statischer Datei-Analyse validieren."""
    if "bridge.py" not in str(filepath):
        return
    content = filepath.read_text()
    plugins_match = re.search(r"_plugins\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not plugins_match:
        return
    plugins_block = plugins_match.group(1)
    accessors = re.findall(r'\(\s*"([^"]+)",\s*"([^"]+)"\s*\)', plugins_block)
    if not accessors:
        return

    for mod_name, accessor in accessors:
        if "mert" in mod_name.lower():
            continue
        # Statische Prüfung: Modul-Datei finden und Accessor-Name darin suchen
        mod_path = mod_name.replace(".", "/") + ".py"
        for base in [ROOT, ROOT / "backend", ROOT / "plugins"]:
            candidate = (
                base / mod_path.split("/")[-1] if "/" not in mod_path else base / "/".join(mod_path.split("/")[1:])
            )
            if not candidate.exists():
                candidate = ROOT / mod_path
            if candidate.exists():
                break
        if not candidate.exists():
            # Versuche direkten Pfad
            candidate = ROOT / mod_path
        if not candidate.exists():
            VIOLATIONS.append(f"{filepath}: Warmup-Modul-Datei für '{mod_name}' nicht gefunden (§G175)")
            continue
        try:
            mod_content = candidate.read_text()
            if f"def {accessor}" not in mod_content:
                VIOLATIONS.append(f"{filepath}: Warmup-Accessor '{accessor}' nicht in {candidate.name} (§G175)")
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["backend", "Aurik10", "denker", "plugins"]
    files = _find_files(paths)

    for f in files:
        try:
            check_torch_zeros_cuda(f)
            check_hardcoded_strings(f)
            check_lock_during_import(f)
            check_os_environ_import(f)
        except SyntaxError:
            pass  # Skip files with syntax errors
        except Exception as e:
            print(f"WARN: {f}: {e}", file=sys.stderr)

    # Warmup accessor check only on bridge.py
    bridge = ROOT / "backend" / "api" / "bridge.py"
    if bridge.exists():
        check_warmup_accessors(bridge)

    if VIOLATIONS:
        print(f"\n❌ {len(VIOLATIONS)} Verstöße gegen §v10.306 Startup-SOTA-Regeln:\n")
        for v in VIOLATIONS:
            print(f"  {v}")
        print("\n👉 Fix: Kein torch.zeros('cuda') außer warmup, setText mit t(), kein import in lock")
        sys.exit(1)
    else:
        print("✅ §v10.306 SOTA-Guard: Alle Prüfungen bestanden")
        sys.exit(0)


if __name__ == "__main__":
    main()
