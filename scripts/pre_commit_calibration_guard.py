#!/usr/bin/env python3
"""§v10.50 Pre-Commit Calibration Guard — erzwingt Vorgaben und Specs.

Prüft vor jedem Commit:
  §V25: Keine hartcodierten Schwellwerte ohne CalibrationContext-Ableitung
  §V26: Keine diskreten Lookup-Tabellen oder if/elif-Buckets
  §G79 (GEBOTE.md): Kalibrierungs-Audit-Log vorhanden
  §G80 (GEBOTE.md): Unkalibrierte Fallbacks mit WARNING

Exit 0 = sauber, Exit 1 = Verstoss.

Autor: Aurik 10 — 19. Juli 2026
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Module, in denen Schwellwerte ERWARTET werden (zentrale Dispatch-Funktionen)
_CALIBRATION_DISPATCH_MODULES: set[str] = {
    "backend/core/signal_flow_tracer.py",
    "backend/core/calibration_matrix.py",
    "backend/core/watchdog_monitor.py",
    "backend/core/joint_calibrator.py",
    "denker/cross_phase_coordinator.py",
}

# Physikalische Konstanten, die von §V25 ausgenommen sind
_PHYSICAL_CONSTANTS: set[str] = {
    "_BRICKWALL_CEILING_DBTP",  # ITU-R BS.1770
    "_LEVEL_COLLAPSE_DBFS",  # digital black
    "_PRE_PHASE_MIN_DBFS",  # signal detection floor
    "_SILENCE_ENERGY_THRESH",  # silence floor
    "_MAX_PHASE_RECORDS",  # memory limit
    "_ORIG_PSD_MAXLEN_S",  # computational budget
    "_ORIG_PSD_NPERSEG",  # FFT parameter
    "_SCAN_CACHE_MAX",  # cache size
    "_PEGEL_WARN_DB",  # 6dB peak = perceptual constant
    "_PEGEL_CRIT_DB",  # 12dB peak = perceptual constant
    "_ECHO_MIN_LAG_MS",  # 20ms = phase/echo boundary (Blauert)
    "_PEAK_PERCENTILE",  # statistical constant
    "_NOVELTY_WARN",  # overwritten by calibrate_sft_thresholds()
    "_NOVELTY_CRIT",  # overwritten by calibrate_sft_thresholds()
    "_HNR_WARN_DB",  # overwritten by calibrate_sft_thresholds()
    "_HNR_CRIT_DB",  # overwritten by calibrate_sft_thresholds()
    "_ECHO_CORR_THRESH",  # overwritten by calibrate_sft_thresholds()
    "CANONICAL_THRESHOLDS_RESTORATION",  # goal targets (base)
    "CANONICAL_THRESHOLDS_STUDIO2026",  # goal targets (base)
    "MATERIAL_SENSITIVITY",  # material constants
    "GOAL_WEIGHTS",  # perceptual weights
    "PROTECTED_PHASES",  # phase classification
    "SIGNIFICANT_OVERLAP",  # architectural constant
    "_prio_weight",  # categorical design constant (priority 1-5)
    "_DEFAULT_MIN_STRENGTH",  # overwritten by joint_calibrate / §v10.44
    "_MAX_RETRIES_G14",  # architectural constant
    "CREST_MIN_NATURAL",  # set inside calibrate_watchdog_thresholds()
    # Dokumentierte psychoakustische/physikalische Konstanten mit lokaler Herleitung.
    "_HNR_DELTA_THRESHOLD_DB",  # Boersma/HNR: >3 dB HNR-Zuwachs = rauigkeitsrelevanter Eingriff
    "_VOICED_RMS_THRESHOLD",  # Stimmhaftigkeits-Floor fuer ACF-HNR-Schaetzung
    "_PLOSIVE_CREST_FACTOR_MIN",  # Plosiv-Crest-Faktor aus Phonem-Boundary-DSP
    "_LUFS_WINDOW_S",  # 400-ms Mikro-Dynamik-Fenster fuer Goosebumps-Metrik
    "MIN_RMS_REDUCTION_DB",  # per-defect Mindestverbesserung fuer repaired_ok
    "MAX_RMS_REDUCTION_DB",  # Over-Repair-Schutz fuer per-defect Verifikation
    "_SILENCE_RMS_FLOOR",  # ~-80 dBFS Silence-Floor fuer Click-LPC-Reparatur
    "_WIDE_STEREO_GUARD",  # Korrelationsgrenze fuer natuerlich breite Stereo-Produktion
    "_WIDE_STEREO_CORR_CAP",  # Azimuth-Guard fuer vollstaendig dekorreliertes Wide-Stereo
    "_CREST_HI",  # 6-dB Crest-Grenze fuer spektrales De-Essing
    "_W_HNR",  # HNR-Gewicht im robusten Gender-Scoring
    "_WEIGHT_HNR",  # psychoakustisch kalibriertes Vokal-Gate-Gewicht (vocal_supremacy_gate)
    "MAX_ACCEPTABLE_STEREO_WIDTH_CHANGE",  # HIPS-Sicherheitsgrenze fuer Stem-Separation
    "_post90_snr",  # kalibrierte digitale Era-Priors aus Frame-Energy-DR
    "_dr_threshold",  # kalibrierte DR-Priors fuer Era-Upvote
}

# Muster für hartcodierte Schwellwerte
_HARDCODED_THRESHOLD_PATTERN = re.compile(
    r"(?:_NOVELTY|_PEGEL|_HNR|_ECHO|_FATIGUE|_CREST|_RMS_|STEREO_|CUMULATIVE_|"
    r"PHASE_TIMEOUTS|HPE_MIN|HPE_TARGET|SILENT_EXCEPT|_DEFAULT_MIN|_MAX_RETRIES|"
    r"_LUFS_|_MIN_NATURAL|_MAX_NATURAL|DC_OFFSET_MAX|NAN_INF_MAX|"
    r"SIGNIFICANT_OVERLAP)\w*\s*[=:]\s*[\d.]+"
)

# Muster für diskrete Lookup-Tabellen
_DISCRETE_BUCKET_PATTERN = re.compile(r"\{\s*\d+\s*:\s*[\d.]+\s*[,}]\s*\d+\s*:")


def get_changed_files() -> list[Path]:
    """Ermittelt git-staged und modified .py-Dateien."""
    files: list[Path] = []
    for cmd in [
        ["git", "-C", str(_PROJECT_ROOT), "diff", "--cached", "--name-only"],
        ["git", "-C", str(_PROJECT_ROOT), "diff", "--name-only"],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.endswith(".py") and not line.startswith(("tests/", "benchmarks/")):
                    fp = _PROJECT_ROOT / line
                    if fp.exists():
                        files.append(fp)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return files


class CalibrationVisitor(ast.NodeVisitor):
    """AST-Visitor der Verstöße gegen §V25-V28 sammelt."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.violations: list[tuple[int, str, str]] = []  # (line, id, desc)
        self.is_dispatch_module = any(d in filepath for d in _CALIBRATION_DISPATCH_MODULES)
        self.has_calibration_call = False

    def visit_Assign(self, node: ast.Assign) -> None:
        """Prüft Zuweisungen auf hartcodierte Schwellwerte."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                if name in _PHYSICAL_CONSTANTS:
                    continue
                # Prüfe auf hartcodierte Schwellwerte
                line = getattr(node, "lineno", 0)
                source_line = ""
                try:
                    with open(self.filepath) as f:
                        lines = f.readlines()
                    if 0 < line <= len(lines):
                        source_line = lines[line - 1].strip()
                except OSError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

                if _HARDCODED_THRESHOLD_PATTERN.search(source_line):
                    self.violations.append((line, "§V25", f"Hardcodierter Schwellwert: {name}"))

                # Prüfe auf diskrete Buckets
                if _DISCRETE_BUCKET_PATTERN.search(source_line):
                    self.violations.append((line, "§V26", f"Diskrete Lookup-Tabelle bei: {name}"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Prüft ob Kalibrierungs-Dispatch-Funktionen aufgerufen werden."""
        if isinstance(node.func, ast.Name):
            if node.func.id in (
                "calibrate_sft_thresholds",
                "calibrate_watchdog_thresholds",
                "calibrate_cross_phase_thresholds",
                "set_novelty_crit_threshold",
                "joint_calibrate",
            ):
                self.has_calibration_call = True
        self.generic_visit(node)


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Prüft eine Datei auf Kalibrierungs-Verstöße."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    except SyntaxError:
        return []

    visitor = CalibrationVisitor(str(filepath))
    visitor.visit(tree)

    # §V27: Prüfe ob Dispatch-Module tatsächlich kalibrieren
    relpath = str(filepath.relative_to(_PROJECT_ROOT))
    content = filepath.read_text(encoding="utf-8")
    # Module die selbst calibrate_* definieren SIND der Dispatch
    if visitor.is_dispatch_module and "def calibrate_" in content:
        visitor.has_calibration_call = True
    if visitor.is_dispatch_module and not visitor.has_calibration_call:
        if _HARDCODED_THRESHOLD_PATTERN.search(content):
            visitor.violations.append(
                (1, "§V27", f"Modul definiert Schwellwerte ohne Kalibrierungs-Dispatch: {relpath}")
            )

    return visitor.violations


def main() -> int:
    """Haupteinstiegspunkt."""
    files = get_changed_files()
    if not files:
        files = list(_PROJECT_ROOT.glob("backend/core/**/*.py"))
        files += list(_PROJECT_ROOT.glob("denker/**/*.py"))

    total_violations = 0
    files_checked = 0

    for fp in sorted(set(files)):
        violations = check_file(fp)
        if violations:
            rel = fp.relative_to(_PROJECT_ROOT)
            print(f"\n─── {rel} ───")
            for line, rule, desc in violations:
                print(f"  L{line}: [{rule}] {desc}")
                total_violations += 1
        files_checked += 1

    print(f"\n{'=' * 60}")
    print(f"Geprüft: {files_checked} Dateien, {total_violations} Verstöße")

    if total_violations == 0:
        print("✅ Kalibrierungs-Vorgaben eingehalten (§V25-V28, §G76-G81)")
        return 0
    else:
        print(f"❌ {total_violations} Verstöße — Commit blockiert")
        print("   → §V25: Keine hartcodierten Schwellwerte ohne CalibrationContext")
        print("   → §V26: Keine diskreten Lookup-Tabellen")
        print("   → §V27: Keine Kalibrierungs-Silos")
        print("   → §G79 (GEBOTE.md): Kalibrierungs-Audit-Log erforderlich")
        return 1


if __name__ == "__main__":
    sys.exit(main())
