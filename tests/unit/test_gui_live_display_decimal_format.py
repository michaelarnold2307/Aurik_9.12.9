"""§GUI-T5 — Nutzersichtbare Dezimalformate der Live-Anzeige: einheitlich deutsch.

Während der Restaurierung sieht der Nutzer Prozentwerte an mehreren Stellen:
Hauptbalken, Queue-Liste, Smooth-Bar-Fallback, Heartbeat-Prognose und
Defekt-Chip-Schweregrade. Befund (§GUI-T5): Queue-Liste, Smooth-Bar-Fallback
und ein Heartbeat-Pfad formatierten mit Punkt („26.88 %"), der Rest mit Komma
(„26,88 %") — je nach Code-Pfad flackert/wechselt das Dezimaltrennzeichen.

Fix: Eine pure Modul-Funktion `_de_num()` ist die einzige Quelle der Wahrheit
für nutzersichtbare Dezimalzahlen; alle Live-Pfade nutzen sie. Diese Tests
sichern die Invariante headless (AST-basiert, ohne Qt-Import).
"""

from __future__ import annotations

import ast
import pathlib

MODERN_WINDOW = pathlib.Path("Aurik10/ui/modern_window.py")
_SRC = MODERN_WINDOW.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _extract_de_num() -> str:
    """Extrahiert den Quelltext der Modul-Funktion _de_num für isolierte Ausführung."""
    for node in _TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_de_num":
            return ast.get_source_segment(_SRC, node) or ""
    raise AssertionError("_de_num Funktion fehlt in modern_window.py")


def _run_de_num(value: float, digits: int = 2) -> str:
    """Führt die echte _de_num-Implementierung isoliert aus (kein Qt nötig)."""
    src = _extract_de_num()
    ns: dict = {}
    exec(compile(src, "<_de_num>", "exec"), ns)
    return ns["_de_num"](value, digits)


# ── Verhalten der puren Funktion ────────────────────────────────────────────


def test_de_num_uses_german_comma() -> None:
    assert _run_de_num(26.876) == "26,88"
    assert _run_de_num(42.5, 1) == "42,5"
    assert _run_de_num(0.02) == "0,02"
    assert _run_de_num(100.0) == "100,00"


def test_de_num_no_dot_remainder() -> None:
    for v in (0.005, 1.0, 26.876, 99.999, 100.0):
        assert "." not in _run_de_num(v)


# ── Quelltext-Invarianten: Live-Pfade nutzen _de_num ───────────────────────


def test_de_num_defined_at_module_level() -> None:
    assert "def _de_num(value: float, digits: int = 2) -> str:" in _SRC


def test_smooth_bar_fallback_uses_de_num() -> None:
    """_set_value_immediately: Punkt-Format f\"{pct:.2f} %\" ist verboten."""
    assert 'super().setFormat(f"{_de_num(pct)} %")' in _SRC, (
        "ModernProgressBar._set_value_immediately muss _de_num (Komma) nutzen"
    )
    assert 'f"{pct:.2f} %"' not in _SRC


def test_heartbeat_forecast_uses_de_num() -> None:
    """Heartbeat-Prognose: f\"{_overall:.1f} %\" mit Punkt ist verboten."""
    assert 'setFormat(f"{_de_num(_overall, 1)} %")' in _SRC, (
        "Heartbeat-Prognose muss _de_num (Komma) nutzen"
    )


def test_queue_list_item_uses_de_num() -> None:
    """Queue-Liste (⏳ datei (x %)): Punkt-Format ({progress / 100:.2f}%) verboten."""
    assert "({_de_num(progress / 100)} %)" in _SRC, (
        "Queue-Listen-Eintrag muss _de_num (Komma) nutzen"
    )
    assert "({progress / 100:.2f}%)" not in _SRC


def test_chip_severity_uses_de_num() -> None:
    """Defekt-Chip-Schweregrade: f\"{sev_f:.2f}%\" und \"0.00%\" sind verboten."""
    assert 'sev_txt = f"{_de_num(sev_f)}%"' in _SRC
    assert 'sev_txt = "0,00%"' in _SRC
    assert 'f"{sev_f:.2f}%"' not in _SRC
    assert '"0.00%"' not in _SRC


def test_no_duplicate_replace_outside_de_num() -> None:
    """Eine Quelle der Wahrheit: `.replace(\".\", \",\")` nur in _de_num selbst."""
    lines = _SRC.splitlines()
    hits = [(i, ln.strip()) for i, ln in enumerate(lines, 1) if '.replace(".", ",")' in ln]
    assert len(hits) == 1, f"replace-Duplikate gefunden (nur _de_num erlaubt): {hits}"
    func = next(n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_de_num")
    func_src = ast.get_source_segment(_SRC, func) or ""
    outside = _SRC.replace(func_src, "")
    lines_out = [ln for ln in outside.splitlines() if '.replace(".", ",")' in ln]
    assert not lines_out, f"replace-Duplikate außerhalb _de_num: {lines_out}"
