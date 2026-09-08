#!/usr/bin/env python3
"""Generate RELEASE_MUST coverage report for spec-to-test traceability.

This script links RELEASE_MUST clauses in .github/copilot-instructions.md
against test gates in tests/normative/ and tests/unit/.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / ".github" / "copilot-instructions.md"
NORMATIVE_TESTS = ROOT / "tests" / "normative"
UNIT_TESTS = ROOT / "tests" / "unit"
TESTS_ROOT = ROOT / "tests"
REPORT_PATH = ROOT / "reports" / "release_must_coverage.json"

# Hard traceability anchors for headings that are semantically broad and cannot
# be matched reliably by keyword heuristics alone.
FORCED_TRACEABILITY: dict[str, list[str]] = {
    "## [RELEASE_MUST] Strength-Envelope-Nichtdegeneration (v10.0.x)": [
        "tests/unit/test_strength_envelope_non_degenerate.py",
        "tests/unit/test_b3_full_song_defect_merge.py",
    ],
    "### §0h [RELEASE_MUST] Music-Death-Shield — absolute Schutzregel (v10.0.0)": [
        "tests/unit/test_artifact_freedom_gate.py",
        "tests/unit/test_edge_taper_no_intro_outro_artifacts.py",
        "tests/unit/test_silence_mask.py",
    ],
    "### §0j [RELEASE_MUST] KI-Modell-Limitation-Awareness": [
        "tests/test_ml_policy_engine.py",
        "tests/unit/test_ml_fallback_cascades.py",
        "tests/unit/test_ml_device_manager.py",
    ],
    "### §0k [RELEASE_MUST] Maximum-Achievable-Score-Prinzip": [
        "tests/normative/test_mas_convergence.py",
        "tests/unit/test_goal_baseline_check_optimizations.py",
    ],
    "### §0p [RELEASE_MUST] Vocal-Supremacy-Doktrin (v10.0.0)": [
        "tests/normative/test_vocal_excellence_contract.py",
        "tests/unit/test_vocal_chain_invariants.py",
        "tests/unit/test_vocal_quality_index.py",
    ],
    "### §0d [RELEASE_MUST] Carrier-Recovery-Referenzmodell": [
        "tests/normative/test_carrier_recovery_reference_model_contract.py",
        "tests/unit/test_carrier_chain_recovery.py",
    ],
    "### §0f [RELEASE_MUST] KI-Agenten-Vorgehensweise: Systemisch vs. Punktuell": [
        "tests/normative/test_spec_consistency.py",
        "tests/unit/test_verboten_linter_compliance.py",
    ],
    "### [RELEASE_MUST] AMD-GPU-Beschleunigung (v10.0.0)": [
        "tests/unit/test_ml_device_manager_amd.py",
        "tests/unit/test_ml_device_manager.py",
    ],
    "### §0c [RELEASE_MUST] Universalitäts-Invariante — alle Importsongs": [
        "tests/test_uat_acceptance_criteria.py",
        "tests/normative/test_competitive_stratified_gate.py",
    ],
    "### [RELEASE_MUST] No-Competing-Instances-Protokoll": [
        "tests/normative/test_magic_button_autopilot_ci_gate.py",
        "tests/unit/test_frontend_ux_spec_compliance.py",
    ],
    "### [RELEASE_MUST] §2.31a Ganzheitliche Song-Selbstkalibrierung (v10.0.0)": [
        "tests/unit/test_unified_restorer_v3.py",
        "tests/integration/test_pipeline_integration.py",
    ],
    "### [RELEASE_MUST] §2.31b PMGG Song-Kalibrierungs-Integration (v10.0.0)": [
        "tests/unit/test_per_phase_musical_goals_gate.py",
    ],
    "### [RELEASE_MUST] Hybrid-Release-Mode für Kernmodelle": [
        "tests/normative/test_hybrid_release_mode.py",
    ],
    "**[RELEASE_MUST] §2.29 PMGG Phase-Skip-Verbot (v10.0.0)**:": [
        "tests/unit/test_per_phase_musical_goals_gate.py",
        "tests/test_uat_acceptance_criteria.py",
    ],
    "**[RELEASE_MUST] §2.29a PMGG Inference-Caching bei Retries (v10.0.0)**:": [
        "tests/normative/test_full_pipeline_determinism.py",
    ],
    "**[RELEASE_MUST] §2.29b PMGG Stable-Metric-Invariante (v10.0.0)**:": [
        "tests/unit/test_per_phase_musical_goals_gate.py",
    ],
    "**[RELEASE_MUST] Vocal-Intimitäts-Gate (Phase 42)**:": [
        "tests/unit/test_vocal_chain_integration.py",
    ],
    "### [RELEASE_MUST] PerformanceGuard — RT-Budget-System (v10.0.0)": [
        "tests/normative/test_performance_budget_ci_gate.py",
        "tests/unit/test_performance_guard_spec_compliance.py",
    ],
    "### [RELEASE_MUST] Quality-First Hauptlauf (v10.0.0)": [
        "tests/unit/test_unified_restorer_v3.py",
        "tests/unit/test_denker/test_aurik_denker.py",
    ],
    "### [RELEASE_MUST] §2.38 Kontinuierliche ML-Veredelung (KMV) — Vollqualitäts-Garantie": [
        "tests/normative/test_kmv_stufe2.py",
    ],
    "### [RELEASE_MUST] §2.40 Vollpipeline-Determinismus + Konkurrenten-Stratifizierung (v10.0.0)": [
        "tests/normative/test_full_pipeline_determinism.py",
        "tests/normative/test_competitive_stratified_gate.py",
    ],
    "### [RELEASE_MUST] §2.39 OOM-Recovery-Checkpoint-System — Nahtlose Wiederaufnahme": [
        "tests/unit/test_recovery_checkpoint.py",
    ],
    "### [RELEASE_MUST] `ml_memory_budget` — Zentrale OOM-Schutzschicht": [
        "tests/normative/test_combined_ml_memory_budget.py",
        "tests/unit/test_ml_plugin_load_and_cleanup.py",
    ],
    "## §2.19 Genre-Classifier-Härtung (17 Genres, [RELEASE_MUST])": [
        "tests/unit/test_genre_classifier.py",
        "tests/unit/test_v100_genre_classifier_profiles.py",
    ],
    "**§2.36a Phonem-spezifische DSP-Algorithmen ([RELEASE_MUST], v10.0.0):**": [
        "tests/normative/test_lyrics_guided_enhancement_gate.py",
        "tests/unit/test_lyrics_guided_enhancement.py",
    ],
    "### [RELEASE_MUST] §2.45 Minimal-Intervention-Prinzip (v10.0.0)": [
        "tests/test_phase_skipping_integration.py",
        "tests/test_phase_skipping.py",
        "tests/unit/test_unified_restorer_v3.py",
    ],
    "### [RELEASE_MUST] §2.45a Mid-Pipeline-Loudness-Drift-Guard (v10.0.0, erweitert v10.0.0)": [
        "tests/unit/test_loudness_cascade_guard.py",
        "tests/unit/test_unified_restorer_v3.py",
    ],
    "### [RELEASE_MUST] §2.46 Carrier-Chain-Inversion (v10.0.0)": [
        "tests/unit/test_denker/test_tontraegerkette_denker.py",
        "tests/unit/test_era_classifier.py",
        "tests/integration/test_pipeline_integration.py",
    ],
    "### [RELEASE_MUST] §2.46a Deep-Transfer-Chain-Pflicht (v10.0.0)": [
        "tests/unit/test_denker_intelligence_trio.py",
        "tests/integration/test_pipeline_integration.py",
    ],
    "### [RELEASE_MUST] §2.47 Adaptive-Intelligence-Prinzip (v10.0.0)": [
        "tests/unit/test_gp_parameter_optimizer.py",
        "tests/unit/test_per_phase_musical_goals_gate.py",
        "tests/unit/test_era_classifier.py",
        "tests/test_defect_scanner_comprehensive.py",
    ],
    "### [RELEASE_MUST] §2.47a Frontend-Backend-PreAnalysis-Handover (v10.0.0)": [
        "tests/unit/test_pre_analysis_handover_no_double_detect.py",
        "tests/unit/test_pre_analysis_path_normalization.py",
    ],
    "### [RELEASE_MUST] §2.48 Kumulative-Phasen-Interaktions-Guard (v10.0.0)": [
        "tests/unit/test_cumulative_interaction_guard.py",
    ],
    "### [RELEASE_MUST] §2.48 Kumulative-Phasen-Interaktions-Guard (v10.0.0, aktualisiert v10.0.0)": [
        "tests/unit/test_cumulative_interaction_guard.py",
    ],
    "### [RELEASE_MUST] §2.49 Artefakt-Freiheits-Gate (v10.0.0)": [
        "tests/unit/test_artifact_freedom_gate.py",
    ],
    "**§2.49b [RELEASE_MUST] Post-Pipeline Kumulativer Stereo-Collapse-Guard (v10.0.0)**": [
        "tests/unit/test_post_pipeline_stereo_guard.py",
        "tests/unit/test_unified_restorer_v3.py",
    ],
    "**[RELEASE_MUST] §2.51 Stereo-Kohärenz-Invariante für Phasen (v10.0.0)**": [
        "tests/unit/test_phases_mid_late.py",
        "tests/unit/test_dolby_nr_detector_and_phase04_headbump.py",
    ],
    "### [RELEASE_MUST] §2.53a Exzellenz-API-Kompatibilitätsvertrag (v10.0.0)": [
        "tests/unit/test_goal_repair_workflow.py",
        "tests/unit/test_denker/test_exzellenz_denker.py",
    ],
    "### [RELEASE_MUST] §2.53b Denker-Plan-Determinismus in UV3 (v10.0.0)": [
        "tests/unit/test_precomputed_phase_plan_determinism.py",
    ],
    "### [RELEASE_MUST] §2.54 Adaptives Phasen-Optimum — Messen-Handeln-Validieren (v10.0.0)": [
        "tests/unit/test_material_adaptive_phase_strength.py",
        "tests/unit/test_cumulative_interaction_guard.py",
    ],
    "### [RELEASE_MUST] §2.55 PMGG-CIG-Synchronisations-Invariante (v10.0.0)": [
        "tests/unit/test_pmgg_cig_sync.py",
    ],
    "### [RELEASE_MUST] §6.2a Material-Pflicht-Phasen (v10.0.0)": [
        "tests/normative/test_material_priority_phases_gate.py",
    ],
    "### [RELEASE_MUST] §2.29c PMGG Restorative-Baseline-Capping (v10.0.0)": [
        "tests/unit/test_per_phase_musical_goals_gate.py",
    ],
    "### §0l [RELEASE_MUST] Per-Phase-Strength-Orakel und 15-Ziele-Teamarbeit (v10.0.0)": [
        "tests/unit/test_phase_strength_oracle.py",
        "tests/unit/test_unified_restorer_v3.py",
    ],
    "### §0m [RELEASE_MUST] Maximal-Ausbaustufe Defektintelligenz (beide Modi)": [
        "tests/normative/test_release_must_0m_0q_contract.py",
        "tests/normative/test_spec_consistency.py",
        "tests/unit/test_defect_scanner_gold_standard.py",
    ],
    "### §0q [RELEASE_MUST] Bug-Gap-Erkennungs-Strategie (v10.0.0)": [
        "tests/normative/test_release_must_0m_0q_contract.py",
        "tests/normative/test_worldclass_kpi_release_gate_contract.py",
        "tests/normative/test_trusted_vocal_restoration_report.py",
    ],
    "## [RELEASE_MUST] Frontend-Version-Anzeige-Invariante": [
        "tests/unit/test_version_checker_and_ux.py",
        "tests/normative/test_modern_window_gui_contract.py",
    ],
    "## [RELEASE_MUST] ROCm-TorchAudio-ABI-Invariante": [
        "tests/unit/test_ml_device_manager_amd.py",
        "tests/unit/test_ml_device_manager.py",
    ],
}


@dataclass(frozen=True)
class CoverageItem:
    """Traceability result for one RELEASE_MUST instruction line."""

    release_must: str
    matched_tests: list[str]
    covered: bool


def _extract_release_must_lines(text: str) -> list[str]:
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if "[RELEASE_MUST]" not in line:
            continue
        if len(line) < 18:
            continue
        items.append(line)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _keywords(line: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_\-]+", line.lower())
    stop = {
        "release_must",
        "der",
        "die",
        "das",
        "und",
        "mit",
        "für",
        "von",
        "auf",
        "in",
        "zu",
        "no",
        "mode",
    }
    return {w for w in words if len(w) >= 5 and w not in stop}


def _iter_test_files() -> Iterable[Path]:
    files: list[Path] = []
    if TESTS_ROOT.exists():
        files.extend(sorted(TESTS_ROOT.rglob("test_*.py")))
    return files


def _extract_explicit_paths(release_line: str) -> list[str]:
    """Extract explicit test/config paths from a release line.

    If copilot instructions reference concrete files (e.g. tests/unit/test_x.py,
    conftest.py, benchmarks/...py), we can directly map those paths and avoid
    keyword-only false negatives.
    """
    return re.findall(r"[A-Za-z0-9_./-]+\.py", release_line)


def _match_tests(release_line: str) -> list[str]:
    forced_paths = FORCED_TRACEABILITY.get(release_line, [])
    forced_matches = [p for p in forced_paths if (ROOT / p).exists()]

    explicit_matches: list[str] = []
    for raw in _extract_explicit_paths(release_line):
        path = ROOT / raw
        if path.exists():
            explicit_matches.append(raw)

    line_keys = _keywords(release_line)
    if not line_keys:
        # Keep explicit path matches even if keyword extraction yields no tokens.
        return sorted(set(forced_matches + explicit_matches))

    matches: list[str] = list(forced_matches + explicit_matches)
    for path in _iter_test_files():
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        hit_count = sum(1 for key in line_keys if key in text)
        if hit_count >= 2:
            matches.append(str(path.relative_to(ROOT)))
    return sorted(set(matches))


def build_report() -> dict:
    """Build the RELEASE_MUST-to-test coverage report payload."""

    spec_text = SPEC_PATH.read_text(encoding="utf-8", errors="replace")
    release_must_items = _extract_release_must_lines(spec_text)

    coverage_items: list[CoverageItem] = []
    for item in release_must_items:
        tests = _match_tests(item)
        coverage_items.append(CoverageItem(release_must=item, matched_tests=tests, covered=bool(tests)))

    total = len(coverage_items)
    covered = sum(1 for item in coverage_items if item.covered)
    pct = (covered / total * 100.0) if total else 0.0

    return {
        "source": str(SPEC_PATH.relative_to(ROOT)),
        "test_dirs": [
            str(NORMATIVE_TESTS.relative_to(ROOT)),
            str(UNIT_TESTS.relative_to(ROOT)),
        ],
        "total_release_must_items": total,
        "covered_items": covered,
        "coverage_percent": round(pct, 2),
        "items": [asdict(item) for item in coverage_items],
    }


def main() -> int:
    """Write the coverage report and return the CI-style exit code."""

    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Hard gate for CI: every RELEASE_MUST line should map to at least one test/config path.
    uncovered = report["total_release_must_items"] - report["covered_items"]
    if uncovered > 0:
        print(
            f"RELEASE_MUST coverage incomplete: {report['covered_items']}/{report['total_release_must_items']} "
            f"({report['coverage_percent']}%)."
        )
        print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
        return 2

    print(
        f"RELEASE_MUST coverage OK: {report['covered_items']}/{report['total_release_must_items']} "
        f"({report['coverage_percent']}%)."
    )
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
