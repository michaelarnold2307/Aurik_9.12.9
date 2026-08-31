"""Tests für den Spec-Integrations-Scanner (audit/spec_integration_scanner.py).

Golden-Tests: Jeder Check muss echte Integrations-Lücken als ERROR/WARNING
melden und saubere Zustände NICHT melden (Fehlerprotokoll-Disziplin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "audit"))

from spec_integration_scanner import (
    check_copilot_sections,
    check_gebote_verboten_matrix,
    check_normative_docs,
    check_spec_references,
    check_verboten_linter_coverage,
)

_INSTRUCTION_FILES = (
    "hoerordnung",
    "pipeline",
    "phases",
    "dsp",
    "musical_goals",
    "tests",
)


def _build_fixture(tmp_path: Path) -> Path:
    """Minimale normative Workspace-Struktur (alle Pflicht-Dokumente vorhanden)."""
    gh = tmp_path / ".github"
    (gh / "instructions").mkdir(parents=True)
    (gh / "specs").mkdir(parents=True)
    (tmp_path / "backend" / "core").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "tests").mkdir()

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (gh / "copilot-instructions.md").write_text(
        "§I GEBOTE\n§II VERBOTE\n§III DSP\n§IV Export\n§V CD-Rauschprofil\n§VI Startup\n§0a Verbotene Phasen\n[RELEASE_MUST] Qualität\n",
        encoding="utf-8",
    )
    (gh / "ID_REGISTRY.md").write_text("# Registry\n", encoding="utf-8")
    (gh / "FILE_REGISTRY.md").write_text("# Files\n", encoding="utf-8")
    for name in _INSTRUCTION_FILES:
        (gh / "instructions" / f"{name}.instructions.md").write_text("# x\n", encoding="utf-8")
    (gh / "GEBOTE.md").write_text(
        "## Kategorie I\n| §G10 | **Titel Zehn** | Beschreibung |\n| §G11 | **Titel Elf** | Beschreibung |\n",
        encoding="utf-8",
    )
    (gh / "VERBOTEN.md").write_text(
        "| V01 | Titel Eins | Beschreibung |\n| V02 | Titel Zwei | Beschreibung |\n",
        encoding="utf-8",
    )
    (gh / "specs" / "01_musical_goals.md").write_text("# Musikziele\n", encoding="utf-8")
    (gh / "specs" / "00_SPEC_INDEX.md").write_text(
        "# Index\n| 01_musical_goals.md | Musikziele |\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "aurik_verboten_linter.py").write_text(
        "# Linter mit V01-Regel\nV01\n",
        encoding="utf-8",
    )
    (tmp_path / "backend" / "core" / "mod.py").write_text(
        "# §G10 (GEBOTE.md) zitiert\n",
        encoding="utf-8",
    )
    return tmp_path


def _severities(findings) -> set[str]:
    return {f.severity for f in findings}


def test_normative_docs_ok_and_missing(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    assert check_normative_docs(ws) == []
    (ws / ".github" / "GEBOTE.md").unlink()
    errs = check_normative_docs(ws)
    assert any(f.item == ".github/GEBOTE.md" and f.severity == "error" for f in errs)


def test_matrix_uncited_id_without_ledger_is_error(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    # §G11 wird nirgendwo zitiert und hat keinen Ledger-Eintrag → ERROR
    errs = check_gebote_verboten_matrix(ws)
    assert any(f.item == "§G11" and f.severity == "error" for f in errs)
    # §G10 ist im Code zitiert → kein Befund
    assert not any(f.item == "§G10" for f in errs)


def test_matrix_uncited_id_with_ledger_is_ok(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    (ws / ".github" / "GEBOTE_INTEGRATION_MATRIX.md").write_text(
        "# Matrix\n| §G11 | Kategorie I | Titel Elf | katalog |\n",
        encoding="utf-8",
    )
    errs = check_gebote_verboten_matrix(ws)
    assert not any(f.item == "§G11" for f in errs)


def test_matrix_exact_id_no_prefix_collision(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    (ws / ".github" / "GEBOTE.md").write_text(
        "## Kategorie I\n| §G10 | **Titel Zehn** | … |\n| §G100 | **Titel Hundert** | … |\n",
        encoding="utf-8",
    )
    # §G100 ist zitiert → §G10 darf NICHT über §G100 als zitiert gelten
    (ws / "backend" / "core" / "mod.py").write_text("# §G100 (GEBOTE.md)\n", encoding="utf-8")
    errs = check_gebote_verboten_matrix(ws)
    assert any(f.item == "§G10" and f.severity == "error" for f in errs)
    assert not any(f.item == "§G100" for f in errs)


def test_dead_spec_reference_is_error(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    (ws / ".github" / "instructions" / "pipeline.instructions.md").write_text(
        "Siehe specs/99_missing.md\n",
        encoding="utf-8",
    )
    errs = check_spec_references(ws)
    assert any(f.item == "99_missing.md" and f.severity == "error" for f in errs)


def test_spec_missing_from_index_is_error(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    (ws / ".github" / "specs" / "02_pipeline_architecture.md").write_text("# P\n", encoding="utf-8")
    errs = check_spec_references(ws)
    assert any(f.item == "02_pipeline_architecture.md" and f.severity == "error" for f in errs)


def test_indexed_unreferenced_spec_is_info(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    errs = check_spec_references(ws)
    # 01_musical_goals.md ist im Index, aber nirgendwo zitiert → info
    assert any(f.item == "01_musical_goals.md" and f.severity == "info" for f in errs)
    assert "error" not in _severities(errs)


def test_index_missing_is_error(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    (ws / ".github" / "specs" / "00_SPEC_INDEX.md").unlink()
    errs = check_spec_references(ws)
    assert any(f.item == "00_SPEC_INDEX.md" and f.severity == "error" for f in errs)


def test_copilot_sections_ok_and_missing(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    assert check_copilot_sections(ws) == []
    (ws / ".github" / "copilot-instructions.md").write_text("§I\n", encoding="utf-8")
    errs = check_copilot_sections(ws)
    assert any(f.item == "§VI" and f.severity == "error" for f in errs)
    assert any(f.item == "[RELEASE_MUST]" and f.severity == "error" for f in errs)


def test_verboten_linter_coverage(tmp_path: Path) -> None:
    ws = _build_fixture(tmp_path)
    # V01 im Linter → ok; V02 weder im Linter noch im Ledger → warning
    warns = check_verboten_linter_coverage(ws)
    assert any(f.item == "V02" and f.severity == "warning" for f in warns)
    assert not any(f.item == "V01" for f in warns)
    # Mit Ledger-Eintrag → kein Befund
    (ws / ".github" / "GEBOTE_INTEGRATION_MATRIX.md").write_text(
        "# Matrix\n| V02 | VERBOTEN.md | Titel Zwei | katalog |\n",
        encoding="utf-8",
    )
    assert check_verboten_linter_coverage(ws) == []
