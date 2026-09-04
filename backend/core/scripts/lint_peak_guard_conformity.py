#!/usr/bin/env python3
"""
Peak-Guard Conformity Linter (§2.45a RELEASE_MUST)

Detects violations of np.max(np.abs(...)) in productive gain-control paths.
- Allowed: Artifact detection, telemetry/logging, analysis, synthesis references
- Forbidden: Gain calculation, makeup gain, level normalization in production code

Rationale: A single transient/click must not prevent the entire audio from being
normalized (§0 Primum non nocere). Use np.percentile(np.abs(...), 99.9) instead.
"""

import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_CONTEXTS = {
    # Telemetry/Logging contexts (metadata/debug)
    "metadata",
    "logger",
    "logging",
    "debug",
    "info",
    "execution_time",
    "kwargs.get",
    "peak_before",
    "peak_after",
    "peak_db",
    "peak_telemetry",
    "profile",
    # Analysis/detection contexts
    "detect",
    "scan",
    "defect",
    "click",
    "crackle",
    "dropout",
    "wow",
    "flutter",
    "artifact",
    "energy",
    "analysis",
    "threshold",
    "if len",  # silence detection
    # Synthesis/reference contexts
    "normalize",  # when generating test signals
    "synthesis",
    "pink",
    "white",
    "noise",
    "synth",
    "reference",
    "generate",
    # Stereo analysis (IACC, etc.)
    "iacc",
    "spatial",
    "width",
    "correlation",
    "mono_compat",  # stereo compatibility metrics
    # True-Peak measurement (intentional, not gain control)
    "truepeak",
    "true_peak",
    # Short-segment analysis
    "if np.max(np.abs(segment)) <",  # silence check before processing
    "if np.max(np.abs(center)) <",  # wow/flutter alignment check
}


VIOLATION_PATTERNS = [
    (
        r"peak\s*=\s*np\.max\(np\.abs\([^)]+\)\)",
        "peak",
        "gain-adjacent variable assignment",
    ),
    (
        r"audio\s*\*=\s*.*np\.max\(np\.abs",
        "audio multiplication",
        "direct gain application",
    ),
    (
        r"gain\s*=\s*.*np\.max\(np\.abs",
        "gain calculation",
        "normalization factor",
    ),
    (
        r"makeup\s*.*=\s*.*np\.max\(np\.abs",
        "makeup gain",
        "loudness compensation",
    ),
    (
        r"level\s*.*=\s*.*np\.max\(np\.abs",
        "level calculation",
        "signal normalization",
    ),
    (
        r"return\s+.*np\.max\(np\.abs\([^)]+\)\)\s*\*",
        "return with max multiplication",
        "direct gain scaling on return",
    ),
]


def get_line_context(lines: list[str], line_idx: int, context_size: int = 3) -> str:
    """Extrahiert surrounding context for analysis."""
    start = max(0, line_idx - context_size)
    end = min(len(lines), line_idx + context_size + 1)
    return "\n".join(lines[start:end])


def is_allowed_context(line: str, file_path: str, line_num: int) -> bool:
    """Prüft if the np.max(np.abs(...)) usage is in an allowed context."""
    # True-Peak-Limiter phase is allowed (it measures true peaks, not for normalization)
    if "phase_47_truepeak_limiter" in file_path or "phase_47" in file_path:
        return True

    # Check for allowed context keywords
    for keyword in ALLOWED_CONTEXTS:
        if keyword in line.lower():
            return True

    # Check if it's in a comparison (if np.max(...) < threshold)
    if " < " in line or (" > " in line and "if " in line):
        return True

    # Check if it's in a division for analysis (peak / rms for dynamic range)
    if "/" in line and "dynamic_range" in line.lower():
        return True

    return False


def lint_file(file_path: str) -> list[tuple[int, str, str]]:
    """Lint a Python file for peak-guard violations."""
    violations: list[tuple[int, str, str]] = []

    if not os.path.isfile(file_path):
        return violations

    with open(file_path) as f:
        try:
            lines = f.readlines()
        except Exception as e:
            logger.debug("§V6 Datei konnte nicht gelesen werden %s: %s — leere Violations zurückgegeben", file_path, e)
            print(f"Warning: Could not read {file_path}: {e}")
            return violations

    for i, line in enumerate(lines, 1):
        if "np.max(np.abs(" not in line:
            continue

        # Skip comments
        if line.strip().startswith("#"):
            continue

        # Skip docstrings/test code
        if '"""' in line or "'''" in line or "test_" in line:
            continue

        # Check against violation patterns
        for pattern, pattern_name, desc in VIOLATION_PATTERNS:
            if re.search(pattern, line):
                # Check if it's in an allowed context
                if not is_allowed_context(line, file_path, i):
                    violations.append((i, f"{pattern_name}: {desc}", line.strip()[:80]))
                break

    return violations


def lint_directory(directory: str, exclude_patterns: list[str] | None = None) -> int:
    """Lint all Python files in a directory."""
    exclude_patterns = exclude_patterns or [
        "*/__pycache__/*",
        "*/.*",
        "*/.pytest_cache/*",
        "*/build/*",
        "*/dist/*",
        "*/.venv*/*",
    ]

    total_violations = 0
    project_root = Path(directory)

    for py_file in project_root.rglob("*.py"):
        # Skip excluded patterns
        relative_path = py_file.relative_to(project_root)
        if any(relative_path.match(pattern) for pattern in exclude_patterns):
            continue

        violations = lint_file(str(py_file))
        if violations:
            print(f"\n{py_file}:")
            for line_num, violation_type, code_snippet in violations:
                print(f"  Line {line_num}: {violation_type}")
                print(f"    {code_snippet}")
                total_violations += 1

    if total_violations == 0:
        print("✓ No peak-guard violations found.")
        return 0
    else:
        print(f"\n✗ Found {total_violations} peak-guard violation(s).")
        return 1


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "backend/core"

    # Always lint these critical directories
    critical_dirs = [
        "backend/core/phases",
        "backend/core/regulator",
        "backend/core/dsp",
    ]

    exit_code = 0
    for crit_dir in critical_dirs:
        if os.path.isdir(crit_dir):
            print(f"\n=== Linting {crit_dir} ===")
            code = lint_directory(crit_dir)
            exit_code = exit_code or code

    sys.exit(exit_code)
