"""Tests für scripts/ear_critical_parity_check.py — fail-closed-Registry-Prüfung."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import ear_critical_parity_check as epc

_HEADER = (
    "| Regel-ID | Quelle | Ohr-Grund | Prüf-Token | Implementierung | Test | Status | Defer-Begründung |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def test_real_registry_passes() -> None:
    code, problems = epc.check()
    assert code == 0, problems
    enforced = sum(1 for cells in epc._rows() if len(cells) > 6 and cells[6] == "enforced")
    assert enforced >= epc.MIN_ENFORCED_ROWS


def test_enforced_row_with_missing_impl_fails(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "reg.md"
    registry.write_text(
        _HEADER + "| §X1 | quelle | grund | token_xy | fehlt/datei.py | tests/fehlt.py | enforced | — |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(epc, "REGISTRY", registry)
    code, problems = epc.check()
    assert code == 1
    assert any("fehlt" in p for p in problems)


def test_deferred_without_reason_fails(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "reg.md"
    registry.write_text(
        _HEADER + "| §X2 | quelle | grund | — | — | — | deferred | — |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(epc, "REGISTRY", registry)
    code, problems = epc.check()
    assert code == 1
    assert any("ohne Begründung" in p for p in problems)


def test_empty_registry_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(epc, "REGISTRY", tmp_path / "fehlt.md")
    code, problems = epc.check()
    assert code == 1
    assert problems
