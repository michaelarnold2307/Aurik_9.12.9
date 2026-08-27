#!/usr/bin/env python3
"""Detect drift between normative specs, tests, and key gate scripts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "spec_drift_report.json"

WATCHED_FILES = [
    ".github/copilot-instructions.md",
    ".github/instructions/hoerordnung.instructions.md",
    ".github/specs/01_musical_goals.md",
    ".github/specs/02_pipeline_architecture.md",
    ".github/specs/03_cognitive_modules.md",
    ".github/specs/04_dsp_standards.md",
    ".github/specs/05_material_system.md",
    ".github/specs/06_phases_system.md",
    ".github/specs/07_quality_and_tests.md",
    ".github/specs/08_architecture_and_distribution.md",
    "tests/normative/test_amrb_ci_gate.py",
    "tests/normative/test_competitive_ci_gate.py",
    "tests/normative/test_magic_button_autopilot_ci_gate.py",
    "tests/normative/test_material_priority_phases_gate.py",
    "scripts/release_must_coverage_check.py",
    "scripts/compliance_check.py",
    "scripts/check_musical_goals.py",
    ".github/ID_REGISTRY.md",
    ".github/FILE_REGISTRY.md",
    "docs/ID_COLLISION_MAP.md",
    "AGENTS.md",
]

BASELINE = ROOT / "reports" / "spec_drift_baseline.json"


@dataclass(frozen=True)
class FileDigest:
    """Speichert Pfad und SHA-256-Hash einer überwachten Datei."""

    path: str
    sha256: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect() -> dict[str, str]:
    digests: dict[str, str] = {}
    for rel in WATCHED_FILES:
        file_path = ROOT / rel
        if not file_path.exists():
            digests[rel] = "<missing>"
            continue
        digests[rel] = _hash_file(file_path)
    return digests


def _load_baseline() -> dict[str, str] | None:
    if not BASELINE.exists():
        return None

    raw = json.loads(BASELINE.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None

    baseline: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            baseline[k] = v
    return baseline


def _save_baseline(digests: dict[str, str]) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(digests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    """Führt den Drift-Check aus und liefert einen Exit-Code für CI-Gates."""

    digests = _collect()
    baseline = _load_baseline()

    report = {
        "watched": WATCHED_FILES,
        "current": digests,
        "drifted": [],
        "baseline_exists": baseline is not None,
    }

    if baseline is None:
        _save_baseline(digests)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("No spec drift baseline existed. Baseline initialized.")
        return 0

    drifted = [rel for rel, digest in digests.items() if baseline.get(rel) != digest]
    report["drifted"] = drifted

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if drifted:
        print("Spec drift detected in:")
        for rel in drifted:
            print(f"- {rel}")
        print("Run scripts/release_must_coverage_check.py and corresponding normative tests.")
        return 3

    print("No spec drift detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
