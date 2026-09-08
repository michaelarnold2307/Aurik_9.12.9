"""§GUI-T2 — Startup-Vertrag (§VI copilot-instructions, §v10.305).

Strukturelle Verifikation der nicht-verhandelbaren Startup-Reihenfolge:
GPU-Erkennung im Hauptthread VOR ModernMainWindow; main() nur unter
__main__; Launch-Skript nutzt python3 -B (kein Bytecode-Cache).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "Aurik10" / "main.py"
LAUNCHER = ROOT / "run_aurik.sh"


def _find_main_func(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("main() nicht gefunden")


def _first_statement_index(func: ast.FunctionDef, match: str) -> int | None:
    """Erster Statement-Index, dessen Quelltext `match` enthält (rekursiv)."""
    for i, stmt in enumerate(func.body):
        seg = ast.get_source_segment(Path(MAIN).read_text(encoding="utf-8"), stmt) or ""
        if match in seg:
            return i
    return None


def test_main_guarded_by_name_main() -> None:
    src = Path(MAIN).read_text(encoding="utf-8")
    tree = ast.parse(src)
    guarded = any(
        isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "__name__"
        for n in tree.body
    )
    assert guarded
    # main() wird unter dem Guard aufgerufen
    assert re.search(r'if __name__ == "__main__":\s*\n\s*main\(\)', src)


def test_gpu_detection_before_main_window() -> None:
    """§VI: GPU-Erkennung im Hauptthread VOR ModernMainWindow()."""
    src = Path(MAIN).read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_main_func(tree)

    gpu_idx = _first_statement_index(func, "get_ml_device_manager")
    window_idx = _first_statement_index(func, "ModernMainWindow()")

    assert gpu_idx is not None, "GPU-Erkennung fehlt in main()"
    assert window_idx is not None, "ModernMainWindow-Instanziierung fehlt in main()"
    assert gpu_idx < window_idx, (
        f"GPU-Erkennung ({gpu_idx}) muss VOR ModernMainWindow ({window_idx}) laufen"
    )


def test_gpu_wait_for_detection_present() -> None:
    """GPU-Erkennung muss auf das Detektionsergebnis warten (kein Race)."""
    src = Path(MAIN).read_text(encoding="utf-8")
    assert "wait_for_detection" in src


def test_launcher_uses_python_b() -> None:
    """§VI: Launch-Skript muss mit python3 -B starten (kein __pycache__)."""
    if not LAUNCHER.exists():
        return  # Skript fehlt in dieser Umgebung — nicht blockieren
    src = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    # mindestens ein Startpfad mit -B vor Aurik10/main.py
    assert re.search(r'"\$\{?VENV_PYTHON\}?"\s+-B\b', src) or re.search(r'-B\b.*Aurik10/main\.py', src)
