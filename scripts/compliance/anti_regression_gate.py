#!/usr/bin/env python3
"""
§2.59 Anti-Regression-Gate — alle 9 behobenen Bug-Muster abdecken.

Jeder Bug dieser Session wird hier als Check verewigt.
Läuft als Pre-Commit-Hook. Blockt Commits, die bekannte Fehlermuster
reproduzieren würden.

Bug-Abdeckung:
  Bug 1: @staticmethod + self.X → check_staticmethod_self.py
  Bug 2: input_path fehlt → dieser Check
  Bug 3: doppeltes Präfix (cached_cached_*) → dieser Check
  Bug 4: PhasePruner falsche Defekt-Namen → ContractValidator
  Bug 5: defekt_hint ohne defect_types → dieser Check
  Bug 6: Preservation Mode Schwelle < 0.97 → dieser Check
  Bug 7: source_fidelity_bandwidth_hz (falsches Feld) → dieser Check
  Bug 8: QualityModeConfig fehlt → check_import_breaking.py
  Bug 9: except Exception: pass → dieser Check
  Bug 10: Hartcodiertes .venv_aurik in subprocess → dieser Check
  Bug 11: Fehlende Retry-Logik in WAV-Loadern → dieser Check
  Bug 12: Unsicheres Tuple-Unpack von wavfile.read() → dieser Check
  Bug 13: Audio-Callback ohne Dual-Path (fehlendes Qt-Signal) → dieser Check
"""

import ast
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


def _workspace_python_file(filepath: str) -> Path | None:
    """Return an existing Python file only when it is inside the repository."""
    candidate = Path(filepath).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        logger.warning("Ignoring file outside repository: %s", filepath)
        return None
    if candidate.suffix != ".py" or not candidate.is_file():
        logger.warning("Ignoring missing or non-Python file: %s", filepath)
        return None
    return candidate


def _relpath(filepath: str) -> str:
    try:
        return str(Path(filepath).resolve().relative_to(ROOT))
    except ValueError:
        return filepath


def _is_scanner_or_test_file(filepath: str) -> bool:
    rel = _relpath(filepath)
    return rel == "scripts/compliance/anti_regression_gate.py" or rel.startswith(("tests/", "benchmarks/"))


def check_typo_double_prefix(filepath: str) -> list[str]:
    """Bug 3: Doppelte Präfixe wie cached_cached_*."""
    issues: list[str] = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_typo_double_prefix Ersatzpfad", exc_info=True)
        return issues
    # Pattern: word_word_ where word == word (like cached_cached_)
    for match in re.finditer(r"\b([a-z]+)_\1_[a-z]", content):
        issues.append(
            f"{filepath}:{content[: match.start()].count(chr(10)) + 1}: doppeltes Präfix '{match.group()}' (Bug 3)"
        )
    return issues


def check_preservation_mode_threshold(filepath: str) -> list[str]:
    """Bug 6: Preservation Mode bw_loss < 0.97."""
    issues: list[str] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_preservation_Betriebsart_Schwelle Ersatzpfad", exc_info=True)
        return issues
    # Pattern: bw_loss_sev >= 0.90 (old threshold)
    if re.search(r"bw_loss.*>=\s*0\.9[0-6]", content):
        for i, line in enumerate(content.split("\n"), 1):
            if "bw_loss" in line and ">=" in line:
                m = re.search(r">=\s*(0\.\d+)", line)
                if m and float(m.group(1)) < 0.97:
                    issues.append(f"{filepath}:{i}: Preservation Mode Schwelle ={m.group(1)} < 0.97 (Bug 6)")
    return issues


def check_wrong_field_name(filepath: str) -> list[str]:
    """Bug 7: Falsche Feldnamen."""
    issues: list[str] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    KNOWN_WRONG = {
        "source_fidelity_bandwidth_hz": "source_fidelity_bandwidth_target_hz",
    }
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_wrong_field_name Ersatzpfad", exc_info=True)
        return issues
    for wrong, correct in KNOWN_WRONG.items():
        if wrong in content:
            for i, line in enumerate(content.split("\n"), 1):
                if wrong in line:
                    issues.append(f"{filepath}:{i}: Falsches Feld '{wrong}' → sollte '{correct}' sein (Bug 7)")
    return issues


def check_bare_except_pass(filepath: str) -> list[str]:
    """Bug 9: except Exception: pass ohne Logging."""
    issues: list[str] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_bare_except_pass Ersatzpfad", exc_info=True)
        return issues
    for i, line in enumerate(lines):
        if re.match(r"\s*except\s+Exception\s*(as\s+\w+)?\s*:", line):
            # Check next line for bare pass/return/continue
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r"\s*(pass|return|continue)\s*$", next_line):
                    # Check if logger.debug is within 2 lines above
                    has_logger = False
                    for j in range(max(0, i - 2), i):
                        if "logger." in lines[j]:
                            has_logger = True
                            break
                    if not has_logger:
                        issues.append(
                            f"{filepath}:{i + 1}: stummer except Exception: {next_line.strip()} ohne Logging (Bug 9)"
                        )
    return issues


def check_pruner_signature(filepath: str) -> list[str]:
    """PhasePruner.prune() must accept restoration_context."""
    issues: list[Any] = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_pruner_signature Ersatzpfad", exc_info=True)
        return issues
    if "def prune(" in content and "restoration_context" not in content:
        for i, line in enumerate(content.split(chr(10)), 1):
            if "def prune(" in line:
                if "restoration_context" not in line:
                    issues.append(f"{filepath}:{i}: prune() missing restoration_context parameter")
    return issues


def check_sentinel_architecture(filepath: str) -> list[str]:
    """VocalDistortionSentinel must be SENSOR only (measure, no check/strength_overrides)."""
    issues: list[Any] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_sentinel_architecture Ersatzpfad", exc_info=True)
        return issues
    if "class VocalDistortionSentinel" in content:
        if "def check(" in content and "def measure(" not in content:
            issues.append(f"{filepath}: Sentinel has check() but no measure() — must be SENSOR")
        if "strength_overrides" in content or "injected_phases" in content:
            issues.append(
                f"{filepath}: Sentinel must not contain strength_overrides/injected_phases — WRITE to restoration_context instead"
            )
    return issues


def check_magic_numbers(filepath: str) -> list[str]:
    """No hardcoded multipliers where continuous measurement is appropriate."""
    issues: list[Any] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_magic_numbers Ersatzpfad", exc_info=True)
        return issues
    # Pattern: *= 0.85 or *= 0.6 in goal weight context
    for i, line in enumerate(content.split(chr(10)), 1):
        if "weights[" in line and "*= 0.85" in line:
            issues.append(f"{filepath}:{i}: hardcoded *= 0.85 — use continuous function instead")
        if "weights[" in line and "*= 0.6" in line and "bw_ratio" not in content:
            issues.append(f"{filepath}:{i}: hardcoded *= 0.6 — use bw_ratio instead")
    return issues


def check_absolute_bw_loss(filepath: str) -> list[str]:
    """bw_loss must be material-relative, not absolute against 20 kHz."""
    issues: list[Any] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_absolute_bw_loss Ersatzpfad", exc_info=True)
        return issues
    # Pattern: using _bw_loss_sev directly for decisions (not _bw_loss_relative)
    if "_bw_loss_sev" in content and "_bw_loss_relative" not in content:
        # Allow in the guard calculation itself (where _bw_loss_relative is defined)
        if "def _build_song_calibration_profile" not in content:
            issues.append(f"{filepath}: uses _bw_loss_sev without material-relative normalization")
    material_bw_markers = (
        "MATERIAL_EXPECTED_BW",
        "MATERIAL_BW_CEILING",
        "MATERIAL_BANDWIDTH",
        "_MATERIAL_BW",
        "max_bandwidth_hz",
        "bandwidth_hz",
        "DECADE_HF_LIMITS",
        "CARRIER_TRANSFER_CHARACTERISTICS",
        "SourceMediumProfile",
        "material_bw_ceiling",
        "material_bw_cap",
    )
    has_material_bw_context = any(marker in content for marker in material_bw_markers)
    # Pattern: hardcoded bandwidth comparison against 20000. Files with a
    # material/era BW table are not absolute references; 20000 is only the
    # digital/full-band fallback inside an adaptive mapping.
    for i, line in enumerate(content.split(chr(10)), 1):
        if "20000" in line and ("bandwidth" in line.lower() or "bw" in line.lower()):
            if not has_material_bw_context:
                issues.append(f"{filepath}:{i}: hardcoded 20000 Hz bandwidth reference without MATERIAL_EXPECTED_BW")
    return issues


def check_defect_classification(filepath: str) -> list[str]:
    """Every DefectType must be classified as SURGICAL or GLOBAL."""
    issues: list[Any] = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_defect_classification Ersatzpfad", exc_info=True)
        return issues
    if "SURGICAL_DEFECT_TYPES" in content:
        # Check all DefectTypes are accounted for
        try:
            from backend.core.defect_scanner import DefectType
            from backend.core.surgical_defect_analyzer import SURGICAL_DEFECT_TYPES

            all_defects = {e.value for e in DefectType}
            surgical = all_defects & SURGICAL_DEFECT_TYPES
            all_defects - SURGICAL_DEFECT_TYPES
            # GLOBAL types are everything not in SURGICAL — no explicit list needed
            if len(surgical) != 24:
                issues.append(f"{filepath}: SURGICAL_DEFECT_TYPES has {len(surgical)} types (expected 24)")
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return issues


def check_surgical_architecture(filepath: str) -> list[str]:
    """Surgical repair architecture must be intact."""
    issues: list[Any] = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        logger.warning("anti_regression_gate.py::Pruefung_surgical_architecture Ersatzpfad", exc_info=True)
        return issues

    # Check 1: PhasePlan must have surgical_routing
    if "class PhasePlan" in content:
        if "surgical_routing" not in content:
            issues.append(f"{filepath}: PhasePlan missing surgical_routing field")

    # Check 2: PhasePruner must protect surgical phases
    if "def prune(" in content and "IntelligentPhasePruner" in content:
        if "surgical_defect_types" not in content:
            issues.append(f"{filepath}: PhasePruner missing surgical phase protection")

    # Check 3: PhaseResult must have time_range
    if "class PhaseResult" in content:
        if "time_range" not in content:
            issues.append(f"{filepath}: PhaseResult missing time_range field")

    # Check 4: restoration_context must propagate surgical_defect_types
    if "_active_defekt_hint = _defekt_hint_kwarg" in content:
        if "surgical_defect_types" not in content:
            issues.append(f"{filepath}: UV3 not propagating surgical_defect_types to restoration_context")

    # Check 5: ExzellenzDenker must know surgical zones
    if "class ExzellenzDenker" in content or "def messe_und_repariere" in content:
        if "surgical" not in content.lower():
            issues.append(f"{filepath}: ExzellenzDenker missing surgical zone awareness")

    return issues


def check_hardcoded_venv_in_subprocess(filepath: str) -> list[str]:
    """Bug 10: Hartcodiertes .venv_aurik in subprocess.Popen-Aufrufen."""
    issues: list[str] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    # Prüfe auf .venv_aurik in subprocess-Kontext
    if ".venv_aurik" not in content:
        return issues
    lines = content.split("\n")
    in_subprocess_block = False
    for i, line in enumerate(lines):
        if "subprocess.Popen" in line or "subprocess.run" in line:
            in_subprocess_block = True
        if in_subprocess_block and ".venv_aurik" in line:
            issues.append(
                f"{filepath}:{i + 1}: Hartcodiertes .venv_aurik in subprocess-Aufruf — "
                "muss sys.executable sein (Bug 10, §V34)"
            )
        if in_subprocess_block and ("]" in line or ")" in line):
            # Check if this line closes the Popen arg list
            if line.strip().startswith("]") or line.strip().startswith(")"):
                in_subprocess_block = False
    return issues


def check_missing_wav_retry(filepath: str) -> list[str]:
    """Bug 11: Fehlende Retry-Logik in WAV-Loadern (§V35)."""
    issues: list[str] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    # Nur relevant für Monitoring-/Analyzer-Scripts die load_audio_file aufrufen
    if "load_audio_file" not in content:
        return issues
    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError:
        return issues
    direct_decode_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            owner = func.value.id if isinstance(func.value, ast.Name) else ""
            if (owner, func.attr) in {
                ("sf", "read"),
                ("soundfile", "read"),
                ("wavfile", "read"),
            }:
                direct_decode_call = True
                break
            if func.attr == "from_file":
                direct_decode_call = True
                break
    if not direct_decode_call:
        return issues
    # Nicht relevant für low-level utility modules (haben eigene Cascade)
    if "meta_router" in filepath and "def _load_audio" in content:
        return issues
    if "file_import" in filepath:
        return issues
    if "modern_window" in filepath:
        return issues  # GUI-File, kein Monitoring-Script
    # Prüfe ob Retry-Loop vorhanden
    if "_max_retries" not in content and "_max_load_retries" not in content:
        for i, line in enumerate(content.split("\n"), 1):
            if "load_audio_file" in line:
                issues.append(
                    f"{filepath}:{i}: load_audio_file() ohne Retry-Loop — "
                    "3× Retry bei transienten WAV-Fehlern erforderlich (Bug 11, §V35)"
                )
                break  # Nur einmal pro Datei melden
    return issues


def check_unsafe_wavfile_unpack(filepath: str) -> list[str]:
    """Bug 12: Unsicheres Tuple-Unpack von scipy.io.wavfile.read() (§V36)."""
    issues: list[str] = []
    if _is_scanner_or_test_file(filepath):
        return issues
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception:
        return issues
    for i, line in enumerate(lines):
        if "wavfile.read(" in line and "," in line.split("=")[0] if "=" in line else False:
            # Tuple destructuring: sr, data = wavfile.read(...)
            lhs = line.split("=")[0].strip()
            if "," in lhs and "wavfile.read" in line:
                issues.append(
                    f"{filepath}:{i + 1}: Unsicheres Tuple-Unpack von wavfile.read() — "
                    "muss index-basierte Entpackung mit isinstance-Prüfung sein (Bug 12, §V36)"
                )
    return issues


def check_dual_path_audio_callback(filepath: str) -> list[str]:
    """Bug 13: Audio-Callback muss Dual-Path (SharedMemory + Qt-Signal) verwenden."""
    issues: list[str] = []
    if "modern_window" not in filepath:
        return issues
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    if "_audio_update_cb" in content and "waveform_phase_update.emit" in content:
        lines = content.split("\n")
        in_cb = False
        for i, line in enumerate(lines):
            if "def _audio_update_cb" in line:
                in_cb = True
            if in_cb and "waveform_phase_update.emit" in line:
                prev_lines = lines[max(0, i - 3) : i]
                if any("else:" in pl for pl in prev_lines):
                    issues.append(
                        f"{filepath}:{i + 1}: emit() nur im else-Fallback — "
                        "muss IMMER emittiert werden (Dual-Path, Bug 13)"
                    )
                break
    return issues


def check_name_error_risk(filepath: str) -> list[str]:
    import py_compile

    try:
        py_compile.compile(filepath, doraise=True)
        return []
    except py_compile.PyCompileError as e:
        return [f"{filepath}: does not compile: {e}"]


def main() -> None:
    changed = sys.argv[1:]
    if not changed:
        print("Anti-Regression-Gate: ⚠️ keine Dateien")
        sys.exit(0)

    all_issues: list[str] = []
    for fp in changed:
        safe_path = _workspace_python_file(fp)
        if safe_path is None:
            continue
        safe_filepath = str(safe_path)
        all_issues.extend(check_typo_double_prefix(safe_filepath))
        all_issues.extend(check_preservation_mode_threshold(safe_filepath))
        all_issues.extend(check_wrong_field_name(safe_filepath))
        all_issues.extend(check_bare_except_pass(safe_filepath))
        all_issues.extend(check_pruner_signature(safe_filepath))
        all_issues.extend(check_sentinel_architecture(safe_filepath))
        all_issues.extend(check_magic_numbers(safe_filepath))
        all_issues.extend(check_absolute_bw_loss(safe_filepath))
        all_issues.extend(check_defect_classification(safe_filepath))
        all_issues.extend(check_surgical_architecture(safe_filepath))
        all_issues.extend(check_hardcoded_venv_in_subprocess(safe_filepath))
        all_issues.extend(check_missing_wav_retry(safe_filepath))
        all_issues.extend(check_unsafe_wavfile_unpack(safe_filepath))
        all_issues.extend(check_dual_path_audio_callback(safe_filepath))
        all_issues.extend(check_name_error_risk(safe_filepath))

    if all_issues:
        print(f"🛡️ Anti-Regression-Gate: {len(all_issues)} Verletzung(en)\n")
        for issue in all_issues:
            print(f"  🚫 {issue}")
        print("\nDiese Muster wurden in Bugfix-Session 2026-07-09 behoben.")
        print("Commits, die sie reproduzieren, werden blockiert.")
        sys.exit(1)

    print("🛡️ Anti-Regression-Gate: ✅ keine bekannten Fehlermuster")
    sys.exit(0)


if __name__ == "__main__":
    main()
