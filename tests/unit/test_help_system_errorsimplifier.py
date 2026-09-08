"""§GUI-T3 — help_system.ErrorSimplifier: Laien-Fehlertexte (headless).

Die Fehler→Laien-Meldung-Zuordnung ist reine Logik; sie entscheidet, was
Nutzer bei jedem Fehler sehen. i18n.t() fällt in Testumgebungen auf den
Schlüssel zurück — deshalb wird gegen die Schlüssel assertiert.
"""

from __future__ import annotations

from Aurik10.ui.help_system import ErrorSimplifier


def test_memory_patterns() -> None:
    assert ErrorSimplifier.simplify(MemoryError("x")) == "help.error.memory"
    assert ErrorSimplifier.simplify("cannot allocate memory") == "help.error.memory"


def test_priority_memory_before_gpu() -> None:
    """'CUDA out of memory' ist zuerst ein Speicherfehler (Reihenfolge!)."""
    assert ErrorSimplifier.simplify("CUDA out of memory") == "help.error.memory"


def test_file_and_permission() -> None:
    assert ErrorSimplifier.simplify(FileNotFoundError("No such file: x")) == "help.error.file_not_found"
    assert ErrorSimplifier.simplify("Permission denied") == "help.error.permission"


def test_gpu_pattern() -> None:
    assert ErrorSimplifier.simplify("CUDA error: illegal memory access") == "help.error.gpu"


def test_generic_error_fallback() -> None:
    out = ErrorSimplifier.simplify("WeirdError: something broke")
    assert "help.error.generic" in out


def test_non_error_passthrough() -> None:
    assert ErrorSimplifier.simplify("nur ein Hinweis") == "nur ein Hinweis"


def test_get_all_messages_nonempty() -> None:
    msgs = ErrorSimplifier.get_all_messages()
    assert isinstance(msgs, dict) and len(msgs) >= 15  # 16 eindeutige Keys (dedupliziert)
    assert all(isinstance(v, str) and v for v in msgs.values())
    assert "help.error.memory" in msgs
