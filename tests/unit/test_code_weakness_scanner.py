"""Tests für den statischen Code-Schwachstellen-Scanner (audit/code_weakness_scanner.py).

Golden-Tests pro Regel: Jede Regel muss ihre Ziel-Schwachstelle zuverlässig
melden (True Positive) und saubere Gegenbeispiele NICHT melden (kein
False Positive). Zusätzlich: Unterdrückungs-Transparenz (suppressed-Zähler).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Scanner liegt unter audit/ — für Import verfügbar machen
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "audit"))

from code_weakness_scanner import scan_workspace


def _write(tmp: Path, rel: str, content: str) -> None:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _rules(workspace: Path) -> dict[str, list]:
    result = scan_workspace(workspace)
    out: dict[str, list] = {}
    for f in result.findings:
        out.setdefault(f.rule_id, []).append(f)
    return out


def test_bridge_import_flagged_in_cli(tmp_path: Path) -> None:
    _write(tmp_path, "cli/tool.py", "from backend.core.unified_restorer_v3 import UnifiedRestorerV3\n")
    rules = _rules(tmp_path)
    assert rules.get("bridge_import_violation"), "CLI-Direktimport muss gemeldet werden"
    assert rules["bridge_import_violation"][0].severity == "critical"


def test_bridge_import_not_flagged_via_bridge_or_elsewhere(tmp_path: Path) -> None:
    _write(tmp_path, "cli/tool.py", "from backend.api.bridge import get_restorer_classes\n")
    _write(tmp_path, "backend/core/mod.py", "from backend.core.foo import Bar\n")
    _write(tmp_path, "denker/plan.py", "from backend.core.foo import Bar\n")
    assert "bridge_import_violation" not in _rules(tmp_path)


def test_dither_missing_flagged_without_context(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/dsp/x.py", "audio = audio.astype(np.int16)\n")
    rules = _rules(tmp_path)
    assert rules.get("dither_missing_int_conversion"), "nacktes astype(int16) muss gemeldet werden"


def test_dither_not_flagged_with_context(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/dsp/x.py", "audio = dither_powr3(audio).astype(np.int16)\n")
    assert "dither_missing_int_conversion" not in _rules(tmp_path)


def test_silent_fallback_flagged_and_logged_not(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "backend/core/ml_router.py",
        "try:\n    x = ml()\nexcept Exception:\n    return 0.5\n",
    )
    assert "silent_fallback_no_log" in _rules(tmp_path)
    _write(
        tmp_path,
        "backend/core/ml_router.py",
        "try:\n    x = ml()\nexcept Exception:\n    logger.warning('fallback')\n    return 0.5\n",
    )
    assert "silent_fallback_no_log" not in _rules(tmp_path)


def test_bare_except_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/mod.py", "try:\n    x = f()\nexcept:\n    return 1\n")
    rules = _rules(tmp_path)
    assert rules.get("bare_except"), "bare except muss gemeldet werden"
    assert rules["bare_except"][0].severity == "medium"


def test_module_logger_missing_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "backend/core/mod.py",
        "def f():\n    try:\n        x = g()\n    except Exception:\n        return None\n",
    )
    assert "module_logger_missing" in _rules(tmp_path)
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import logging\nlogger = logging.getLogger(__name__)\ndef f():\n    try:\n        x = g()\n    except Exception:\n        return None\n",
    )
    assert "module_logger_missing" not in _rules(tmp_path)


def test_nan_inf_guard_missing_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "backend/core/phases/phase_99_test.py",
        "import numpy as np\ndef process(audio):\n    return audio * 2.0\n",
    )
    assert "nan_inf_guard_missing" in _rules(tmp_path)
    _write(
        tmp_path,
        "backend/core/phases/phase_99_test.py",
        "import numpy as np\ndef process(audio):\n    return np.nan_to_num(audio * 2.0)\n",
    )
    assert "nan_inf_guard_missing" not in _rules(tmp_path)


def test_determinism_time_usage_aggregated(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/mod.py", "import time\ntime.time()\ntime.time()\n")
    rules = _rules(tmp_path)
    assert rules.get("determinism_time_usage"), "≥2 time.time() müssen aggregiert gemeldet werden"
    assert "2 Vorkommen" in rules["determinism_time_usage"][0].evidence
    # 1 Treffer liegt unter der Meldeschwelle → nur suppressed, kein Befund
    _write(tmp_path, "backend/core/mod.py", "import time\ntime.time()\n")
    result = scan_workspace(tmp_path)
    assert "determinism_time_usage" not in {f.rule_id for f in result.findings}
    assert result.suppressed.get("determinism_time_usage", 0) >= 1


def test_print_in_production_aggregated(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/mod.py", "print(1)\nprint(2)\nprint(3)\n")
    rules = _rules(tmp_path)
    assert rules.get("print_in_production"), "≥3 print() müssen aggregiert gemeldet werden"
    # 2 Treffer unter Schwelle → nur suppressed
    _write(tmp_path, "backend/core/mod.py", "print(1)\nprint(2)\n")
    result = scan_workspace(tmp_path)
    assert "print_in_production" not in {f.rule_id for f in result.findings}
    assert result.suppressed.get("print_in_production", 0) >= 1


def test_ast_cap_suppressed_counted(tmp_path: Path) -> None:
    # 5 bare excepts: 3 gemeldet (AST-Cap), 2 im suppressed-Zähler sichtbar
    body = "".join("try:\n    x = f()\nexcept:\n    return 1\n" for _ in range(5))
    _write(tmp_path, "backend/core/mod.py", body)
    result = scan_workspace(tmp_path)
    bare = [f for f in result.findings if f.rule_id == "bare_except"]
    assert len(bare) == 3, f"AST-Cap greift bei 3: {len(bare)}"
    assert result.suppressed.get("bare_except", 0) >= 2, "unterdrückte AST-Befunde müssen gezählt werden"


def test_critical_sorted_first_despite_max_findings(tmp_path: Path) -> None:
    # Viele Low-Befunde + ein CRITICAL: bei max_findings=1 darf nur das CRITICAL bleiben
    for i in range(5):
        _write(tmp_path, f"backend/core/mod_{i}.py", "import time\ntime.time()\ntime.time()\n")
    _write(tmp_path, "cli/tool.py", "from backend.core.unified_restorer_v3 import UnifiedRestorerV3\n")
    result = scan_workspace(tmp_path, max_findings=1)
    assert len(result.findings) == 1
    assert result.findings[0].severity == "critical"
    assert result.suppressed.get("truncated_max_findings", 0) >= 1
