#!/usr/bin/env python3
"""V01–V75 VERBOTEN-Linter v5 — vollständig an VERBOTEN.md angepasst.

Abdeckung: 27 von 44 regex-detectable Regeln (AST/Runtime-Regeln separat).
Referenz: .github/VERBOTEN.md — Linter-Referenz-Tabelle.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

_PROJECT_ROOT = Path(__file__).parent.parent


class Violation(NamedTuple):
    """Ein einzelner VERBOTEN-Linter-Fund."""

    rule: str
    severity: str
    description: str
    file: str

    def __str__(self) -> str:
        return f"{self.rule}[{self.severity}]: {self.description} — {self.file}"


# ═══════════════════════════════════════════════════════════════════════════
# Rule definitions: (pattern, description, skip-patterns, severity)
# ═══════════════════════════════════════════════════════════════════════════
RULES: dict[str, dict] = {
    # ── V01: print() in production code (§Teil-A Logging) ──────────────
    "V01": {
        "p": r"\bprint\s*\(",
        "d": "print() statt logger — Produktionscode",
        "skip": {
            "test_",
            "conftest",
            "__init__",
            "scripts/",
            "setup.py",
            "examples/",
            "audit/",
            "Aurik10/",
            "benchmarks/",
            "cli/",
            "golden_samples/",
            "usability/",
            "policy/",
            "tests/",
        },
        "sev": "ERROR",
    },
    # ── V-BRIDGE (Bridge-Bypass): from backend.* import ─────────────────
    # §v10.70: Regex erfasst JEDEN direkten Import aus backend.* (außer api.bridge)
    # vom Frontend oder anderen Modulen. Aurik10/ ist NICHT im skip — der Linter
    # MUSS Frontend-Dateien auf Bridge-Bypasses scannen.
    "V-BRIDGE": {
        "p": r"from backend\.(core|file_import|dsp|plugins|config|ml_inference|phases)\.|import backend\.(core|file_import|dsp|plugins|config|ml_inference|phases)\.",
        "d": "Bridge-Bypass-Verbot",
        "skip": {
            ".agents/",
            "audit/",
            "backend/",
            "benchmarks/",
            "bridge",
            "__init__",
            "conftest",
            "denker/",
            "dsp/",
            "api/",
            "forensics/",
            "golden_samples/",
            "policy/",
            "tests/",
            "test_",
            "plugins/",
            "scripts/",
            "usability/",
            "workflow/",
        },
        "sev": "ERROR",
    },
    # ── V02: sf.read() / librosa.load() statt load_audio_file() ────────
    "V02": {
        "p": r"\b(sf\.read\s*\(|librosa\.load\s*\()",
        "d": "sf.read/librosa.load statt load_audio_file() – export‑Sicherheit",
        "skip": {
            "test_",
            "exporter.py",
            "audio_exporter.py",
            "file_import",
            "generate_dummy",
            "conftest",
            "audit/",
            "benchmarks/",
            "scripts/",
            "Aurik10/",
            "golden_samples",
            "tests/",
        },
        "sev": "ERROR",
    },
    # ── VXX: sf.read() / librosa.load() statt load_audio_file() ────────
    "VXX": {
        "p": r"\b(sf\.read\s*\(|librosa\.load\s*\()",
        "d": "sf.read/librosa.load statt load_audio_file() – export‑Sicherheit",
        "skip": {
            "test_",
            "exporter.py",
            "audio_exporter.py",
            "file_import",
            "generate_dummy",
            "conftest",
            "audit/",
            "benchmarks/",
            "scripts/",
            "Aurik10/",
            "golden_samples",
            "tests/",
        },
        "sev": "ERROR",
    },
    # ── V03: map_location="cuda" ohne ml_device_manager ────────────────
    "V03": {
        "p": r'map_location\s*=\s*["\']cuda["\']',
        "d": "map_location=cuda ohne ml_device_manager",
        "skip": {"ml_device_manager", "test_"},
        "sev": "ERROR",
    },
    # ── V05: griffinlim() als Endschritt ───────────────────────────────
    "V05": {
        "p": r"\bgriffinlim\s*\(",
        "d": "griffinlim() — PGHI/Vocos verwenden",
        "skip": {"test_", "VERBOTEN", "docs", "spec", "scripts/"},
        "sev": "ERROR",
    },
    # ── V08: np.max(np.abs(audio)) in Gain-Pfad ────────────────────────
    "V08": {
        "p": r"np\.max\s*\(\s*np\.abs\s*\(\s*audio\s*\)\s*\)",
        "d": "np.max(abs(audio)) — np.percentile(99.9) verwenden",
        "skip": {
            "test_",
            "exporter.py",
            "export_guard",
            "audio_exporter.py",
            "peak_guard",
            "dsp/",
            "forensics/",
            "safety_wrappers/",
            "golden_samples",
            "scripts/",
            "metrics",
            "classifier",
            "detector",
            "monitor",
            "generator",
            "persistence/",
            "quality_prediction",
            "naturalness",
            "emotional_resonance",
        },
        "sev": "WARNING",
    },
    # ── V09: consecutive_rollbacks += in Carrier-Repair ────────────────
    "V09": {
        "p": r"consecutive_rollbacks\s*\+=",
        "d": "consecutive_rollbacks in Carrier-Repair-Phase",
        "skip": {"test_", "feedback_chain", "cumulative_interaction_guard"},
        "sev": "ERROR",
    },
    # ── V04: gate_dbfs=-36.0 ohne reference_for_gate (§2.45a) ─────────
    "V04": {
        "p": r"gate_dbfs\s*=\s*-?3[0-9]\.\d+",
        "negate": r"reference_for_gate",
        "d": "gate_dbfs ohne reference_for_gate — compute_signal_relative_gate_dbfs() verwenden (§2.45a)",
        "skip": {"test_", "exporter.py", "dsp/", "scripts/"},
        "sev": "ERROR",
    },
    # ── V12: CausalDefectReasoner einseitige Tabellen (AST-Check) ──────
    # Wird in _scan_v32_v33_ast() als AST-Check implementiert — Regex hier nur für
    # schnelle Vorabprüfung auf CAUSE_TO_PHASES ohne CAUSES-Konsistenz
    "V12": {
        "p": r"CAUSE_TO_PHASES\s*=\s*\{",
        "negate": r"\bCAUSES\b.*\{[^}]*Cause\.|CAUSES\s*=\s*\[",
        "d": "CAUSE_TO_PHASES ohne bidirektionale CAUSES-Konsistenz — CausalDefectReasoner (§2.59)",
        "skip": {"test_", "causal_defect_reasoner.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
    # ── V13: _MATERIAL_PRIORITY_PHASES Duplikat-Schlüssel (AST-Check) ──
    # Wird in _scan_v32_v33_ast() als AST-Check implementiert — Regex hier nur für
    # schnelle Vorabprüfung auf Duplikate im Dict-Literal
    "V13": {
        "p": r"_MATERIAL_PRIORITY_PHASES\s*=\s*\{",
        "negate": r"#.*no-dup-check|#.*F601-ignore",
        "d": "_MATERIAL_PRIORITY_PHASES Duplikat-Schlüssel prüfen — F601-Scan in unified_restorer_v3.py",
        "skip": {"test_", "unified_restorer_v3.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
    # ── V11: sosfilt() ohne sosfiltfilt() im selben File ───────────────
    "V11": {
        "p": r"\bsosfilt\s*\(",
        "negate": r"\bsosfiltfilt\s*\(",
        "d": "sosfilt ohne sosfiltfilt — destruktive Interferenz möglich",
        "skip": {
            "test_",
            "exporter.py",
            "_export_nuance",
            "dsp/",
            "forensics/",
            "safety_wrappers/",
            "denker/",
            "metrics",
            "classifier",
            "detector",
            "feature_",
            "genre_",
            "mushra",
            "bark_",
            "allpass",
            "psychoacoustics",
            "benchmarks",
            "processing/",
            "adaptive_plugins",
            "consonant_enhancement",
            "defect_analysis",
            "defect_scanner",
            "emergency_restoration",
            "ki_hearing_model",
            "perceptual_export_optimizer",
            "perceptual_intensity",
            "production_enhancements",
            "sub_stem_processor",
            "phase_03_denoise",
            "phase_09_crackle",
            "phase_34_mid_side",
            "phase_40_loudness",
            "run_amrb",
            "scripts/",
            "verify_all",
        },
        "sev": "WARNING",
    },
    # ── V14: Speech-Metrik (PESQ/STOI/DNSMOS/NISQA) ───────────────────
    "V14": {
        "p": r"(?:^|\s)(?:PESQ|pesq|SI[.-]SDR|si_sdr|STOI|stoi|DNSMOS|NISQA|VISQOL.*Speech)\b",
        "d": "Speech-Metrik — PQS-MOS/VERSA verwenden",
        "skip": {
            "test_",
            "VERBOTEN",
            "forbidden",
            "benchmark",
            "sota_eval",
            "spec",
            "docs",
            "quality_",
            "dsp/",
            "config/",
            "policy/",
            "audit/",
            "multi_pass",
            "musical_goals",
            "hyperparameter",
        },
        "sev": "ERROR",
    },
    # ── V21: Truncation ohne Dither ────────────────────────────────────
    "V21": {
        "p": r"int\s*\(\s*audio.*\)\s*(?:#.*(?:ohne|without|kein).*(?:dither|noise.shape))",
        "d": "Truncation ohne Dither",
        "skip": {"exporter.py", "test_", "dither"},
        "sev": "ERROR",
    },
    # ── V27: JITTER_ARTIFACTS → phase_12 (falsch) ─────────────────────
    # Bounded, boundary-aware span instead of unbounded .*: spans a single
    # dict-value block (e.g. a PhaseAssignment(...) call or a phases-list)
    # across lines but stops before a new top-level dict entry starts
    # (§false-positive guard — a naive bounded span still crossed into the
    # NEXT unrelated entry in real code, see causal_defect_reasoner.py).
    "V27": {
        "p": r'JITTER_ARTIFACTS(?s:(?:(?!\n\s{0,8}(?:"[a-zA-Z_]+"|DefectType\.\w+)\s*:).){0,200}?)phase_12_wow_flutter',
        "d": "JITTER mit phase_12 — phase_14+23 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V28: NR_BREATHING_ARTIFACT → phase_03/29 (falsch) ─────────────
    "V28": {
        "p": r'NR_BREATHING_ARTIFACT(?s:(?:(?!\n\s{0,8}(?:"[a-zA-Z_]+"|DefectType\.\w+)\s*:).){0,200}?)phase_(?:0[3]|29)',
        "d": "NR_ATEMARTEFAKT mit NR-Phase — phase_54 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V29: OVERLOAD_DISTORTION → phase_63 (falsch) ───────────────────
    "V29": {
        "p": r'OVERLOAD_DISTORTION(?s:(?:(?!\n\s{0,8}(?:"[a-zA-Z_]+"|DefectType\.\w+)\s*:).){0,200}?)phase_63',
        "d": "OVERLOAD mit phase_63 — phase_09+23 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V30: ALIASING → phase_03 (falsch) ─────────────────────────────
    "V30": {
        "p": r'ALIASING(?s:(?:(?!\n\s{0,8}(?:"[a-zA-Z_]+"|DefectType\.\w+)\s*:).){0,200}?)phase_03',
        "d": "ALIASING mit phase_03 — phase_23+50 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V31: ROOM_MODE_RESONANCE mit phase_05 als einzigem Primary ohne
    # Notch-EQ (phase_04) — Breitband-Rolloff trifft resonante Peaks nicht
    # chirurgisch (§4.11, VERBOTEN.md V31). ──────────────────────────────
    "V31": {
        "p": r'room_mode_resonance(?s:(?:(?!\n\s{0,8}(?:"[a-zA-Z_]+"|DefectType\.\w+)\s*:).){0,200}?)phase_05_rumble_filter',
        "negate": r'room_mode_resonance(?s:(?:(?!\n\s{0,8}(?:"[a-zA-Z_]+"|DefectType\.\w+)\s*:).){0,200}?)phase_04',
        "d": "ROOM_MODE_RESONANCE mit phase_05 ohne Notch-EQ (phase_04) — phase_04 als Primary ergänzen",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "WARNING",
    },
    # ── V39: §0a-forbidden Phasen für Restoration ─────────────────────
    "V39": {
        "p": r"Restoration.*(?:phase_21_exciter|phase_35_multiband|phase_42_vocal)",
        "d": "§0a-verbotene Phase für Restoration-Cause",
        "skip": {
            "test_",
            "VERBOTEN",
            "docs",
            "causal_defect_reasoner",
            "scripts/",
            "cumulative_interaction_guard",
            "unified_restorer_v3",
        },
        "sev": "ERROR",
    },
    # ── V44: IACC ohne Stereo-Guard ────────────────────────────────────
    "V44": {
        "p": r"IACC\s*[<>]\s*0\.[0-9]+",
        "d": "IACC ohne Stereo-Guard",
        "skip": {"stereo_guard", "test_", "musical_goals_metrics", "perceptual_salience"},
        "sev": "INFO",
    },
    # ── V46: dBFS mit linearem Skalierungsfaktor ───────────────────────
    "V46": {
        "p": r"level_db\s*\*\s*strength|dbFS\s*\*\s*(?:0\.\d+|strength)",
        "d": "dBFS linear skaliert — 10^(dB/20) verwenden",
        "skip": {"test_", "dsp/"},
        "sev": "ERROR",
    },
    # ── V47: DR-Expansion ohne Ceiling UND Noise-Floor-Guard (§6.2b, §v10.61) ──
    "V47": {
        "p": r"(?:phase_26|dynamic_range_expansion).*\n.*(?:strength|intensity)\s*[+\-*\/]",
        "negate": r"_MATERIAL_DR_CEILING_DB|Per-Band-Noise-Floor-Guard|noise_floor_guard",
        "d": "DR-Expansion ohne Material-Ceiling und Noise-Floor-Guard (§6.2b, §v10.61)",
        "skip": {"test_", "phase_26_dynamic_range_expansion.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
    # ── V48: BW-Extension ohne Ceiling (§6.2c) ────────────────────────
    "V48": {
        "p": r"(?:phase_0[67]|phase_23|phase_39).*\n.*(?:bw_extension|bandwidth_extend|harm_ext)",
        "negate": r"_MATERIAL_BW_CEILING_HZ|material_bw_ceiling",
        "d": "BW-Extension ohne Material-Ceiling (§6.2c) — Shellac≤8kHz, Vinyl≤16kHz, Cassette≤12kHz",
        "skip": {"test_", "phase_06.py", "phase_07.py", "phase_23.py", "phase_39.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
    # ── V49: Rauschtextur-Check fehlt (§4.7) ──────────────────────────
    "V49": {
        "p": r"(?:phase_03|denoise|noise_reduction).*\n.*(?:result|output)\s*=",
        "negate": r"NoiseTextureCoherenceGuard|texture_coherence",
        "d": "Denoising ohne NoiseTextureCoherenceGuard (§4.7) — Kohärenz≥0.80 in Restoration",
        "skip": {"test_", "phase_03_denoise.py", "scripts/", "docs/"},
        "sev": "WARNING",
    },
    # ── V53: Pitch-Kaskade ohne RMVPE (§4.4) ──────────────────────────
    "V53": {
        "p": r"(?:FCPE|CREPE|PESTO|pYIN).*pitch.*(?:fallback|cascade)",
        "negate": r"RMVPE|get_rmvpe_plugin|rmvpe",
        "d": "Pitch-Kaskade ohne RMVPE als Tier-2 (§4.4) — FCPE→RMVPE→PESTO→pYIN",
        "skip": {"test_", "pitch_detector.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
    # ── V58: Phase-50 Spike-Detection ohne HF-Guard (§2.57) ───────────
    "V58": {
        "p": r"phase_50.*spike|spike_detection.*repair_channel",
        "negate": r"_hf_protected_bin_start|material_rolloff.*bin_hz",
        "d": "Phase-50 Spike-Detection ohne HF-Guard (§2.57) — analoge Harmoniken schützen",
        "skip": {"test_", "phase_50.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
    # ── V73: Phase-23 Inpainting ohne POCS (§4.7c) ────────────────────
    "V73": {
        "p": r"phase_23.*inpaint|spectral_inpaint.*repair_channel",
        "negate": r"POCS|pocs_iter|consistency_projection",
        "d": "Phase-23 Inpainting ohne POCS-Schleife (§4.7c) — material-adaptiv n_iter=2–5 vor PGHI",
        "skip": {"test_", "phase_23.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
    # ── V74: signal.lfilter in Vocal Bell-EQ (Phase 42) ───────────────
    "V74": {
        "p": r"signal\.lfilter\s*\([^)]*(_boost_presence|_enhance_chest)",
        "negate": r"signal\.filtfilt",
        "d": "signal.lfilter in Vocal Bell-EQ — filtfilt verwenden (zero-phase, §V74)",
        "skip": {"test_", "phase_42_vocal_enhancement.py", "scripts/", "docs/"},
        "sev": "ERROR",
    },
}

SKIP_DIRS = {
    ".venv",
    ".venv_aurik",
    "__pycache__",
    "node_modules",
    ".git",
    "models/",
    "temp_repro/",
    "plugins/_vendor_",
}  # _vendor_* = unverändert kopierter Drittanbieter-Code (MIT, LICENSE beiliegend) — nie umstylen

# ═══════════════════════════════════════════════════════════════════════════
# File-scope skip: certain file patterns are globally excluded per rule
# ═══════════════════════════════════════════════════════════════════════════
GLOBAL_FILE_SKIP: dict[str, set[str]] = {
    # Skip production-only rules for test/script/doc files globally
    "prod_only": {"test_", "/tests/", "setup.py", "conftest.py", ".github/", "docs/", "scripts/"},
    # Skip for plugins/sdk abstract base classes
    "sdk": {"plugins/sdk/"},
}


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        base = _literal_str(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _string_values(node: ast.AST) -> set[str]:
    values: set[str] = set()
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            text = _literal_str(elt)
            if text:
                values.add(text)
    elif isinstance(node, ast.Call) and _literal_str(node.func) == "frozenset" and node.args:
        values.update(_string_values(node.args[0]))
    else:
        text = _literal_str(node)
        if text:
            values.add(text)
    return values


def _assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
            if isinstance(target, ast.Attribute):
                return target.attr
    if isinstance(node, ast.AnnAssign):
        target = node.target
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
    return None


def _assigned_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    return None


def _scan_v32_v33_ast(fp: Path, source: str, rel: Path) -> list[Violation]:
    issues: list[Violation] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues

    exclusions: dict[str, set[str]] = {}
    critical_transparency_phases: set[str] = set()
    for node in ast.walk(tree):
        name = _assigned_name(node)
        value = _assigned_value(node)
        if name == "_PHASE_SPECIFIC_DRIFT_EXCLUSIONS" and isinstance(value, ast.Dict):
            for key, val in zip(value.keys, value.values, strict=False):
                if isinstance(key, ast.AST):
                    key_text = _literal_str(key)
                else:
                    key_text = None
                if key_text:
                    exclusions[key_text] = _string_values(val)
        if name == "CRITICAL_PAIRS" and isinstance(value, (ast.List, ast.Tuple)):
            for entry in value.elts:
                if not isinstance(entry, ast.Tuple) or len(entry.elts) < 2:
                    continue
                goal_text = _literal_str(entry.elts[1])
                if goal_text != "transparenz":
                    continue
                for phase_text in _string_values(entry.elts[0]):
                    if phase_text:
                        critical_transparency_phases.add(phase_text)

    if critical_transparency_phases:
        for phase_id in sorted(critical_transparency_phases):
            phase_prefix = "_".join(phase_id.split("_")[:2]) if phase_id.startswith("phase_") else phase_id
            excluded = exclusions.get(phase_id) or exclusions.get(phase_prefix) or set()
            if "transparenz" not in excluded:
                issues.append(
                    Violation(
                        rule="V32",
                        severity="ERROR",
                        description="Carrier-NR-Phase in CRITICAL_PAIRS ohne transparenz-Exclusion",
                        file=str(rel),
                    )
                )
                break
    # --- AST checks for V12 and V13 ---
    # V12: CAUSE_TO_PHASES without bidirectional CAUSES consistency
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        name = _assigned_name(node)
        value = _assigned_value(node)
        if name == "CAUSE_TO_PHASES" and isinstance(value, ast.Dict):
            for key_node in value.keys:
                key_text = _literal_str(key_node)
                if key_text and not re.search(r"\\bCAUSES\\b", key_text):
                    issues.append(Violation(rule="V12", severity="ERROR", description=RULES["V12"]["d"], file=str(rel)))
    # V13: _MATERIAL_PRIORITY_PHASES duplicate keys
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        name = _assigned_name(node)
        value = _assigned_value(node)
        if name == "_MATERIAL_PRIORITY_PHASES" and isinstance(value, ast.Dict):
            seen: set[str] = set()
            for key_node in value.keys:
                key_text = _literal_str(key_node)
                if key_text:
                    if key_text in seen:
                        issues.append(
                            Violation(rule="V13", severity="ERROR", description=RULES["V13"]["d"], file=str(rel))
                        )
                    else:
                        seen.add(key_text)

    phase_path = str(fp).replace("\\", "/")
    if "/phases/phase_" in phase_path or fp.name.startswith("phase_"):
        tracked_names = {"DETECTION_THRESHOLD", "CORRECTION_STRENGTH"}
        carrier_keys = {"MaterialType.TAPE", "MaterialType.REEL_TAPE", "MaterialType.VINYL", "MaterialType.SHELLAC"}
        cassette_keys = {"MaterialType.CASSETTE", "CASSETTE", "cassette"}
        for node in ast.walk(tree):
            name = _assigned_name(node)
            value = _assigned_value(node)
            if name not in tracked_names or not isinstance(value, ast.Dict):
                continue
            keys = {_literal_str(key) for key in value.keys if key is not None}
            keys = {key for key in keys if key}
            if keys & carrier_keys and not (keys & cassette_keys):
                issues.append(
                    Violation(
                        rule="V33",
                        severity="ERROR",
                        description=f"{name} fehlt MaterialType.CASSETTE für materialindizierte Phase-Parameter",
                        file=str(rel),
                    )
                )
                break
    return issues


def _should_skip_rule(fp: Path, rid: str) -> bool:
    """Check if file should be skipped for a given rule."""
    r = str(fp)
    if any(s in r for s in SKIP_DIRS):
        return True
    rule = RULES.get(rid)
    if rule is None:
        return True
    for s in rule.get("skip", set()):
        if s == "test_":
            # §Fix: match the FILENAME, not the full path — pytest's tmp_path
            # fixture always embeds the calling test function's name (which
            # starts with "test_") in the temp directory, so a full-path
            # substring check would wrongly skip every fixture file created
            # by a unit test that exercises the linter itself.
            if fp.name.startswith("test_"):
                return True
        elif s in r:
            return True
    return False


def scan(fp: Path) -> list[Violation]:
    """Scan one Python file for VERBOTEN violations."""
    if fp.suffix != ".py":
        return []
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    # Strip docstrings and comments (keep code only for regex matching)
    code_lines: list[str] = []
    in_ds = False
    for line in lines:
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_ds = not in_ds
            continue
        if in_ds:
            continue
        if s.startswith("#"):
            continue
        code_lines.append(line)

    code_text = "\n".join(code_lines)
    try:
        rel = fp.relative_to(_PROJECT_ROOT)
    except ValueError:
        # File outside _PROJECT_ROOT (e.g. a pytest tmp_path fixture) — report
        # the absolute path instead of failing the scan.
        rel = fp

    issues: list[Violation] = []
    for rid, rule in RULES.items():
        if _should_skip_rule(fp, rid):
            continue
        if re.search(rule["p"], code_text, re.IGNORECASE | re.MULTILINE):
            # Check negate pattern: if present, only flag when negate does NOT match
            negate_pat = rule.get("negate")
            if negate_pat and re.search(negate_pat, code_text, re.IGNORECASE | re.MULTILINE):
                continue  # negate pattern found → file uses correct form, skip
            sev = rule.get("sev", "WARNING")
            issues.append(Violation(rule=rid, severity=sev, description=rule["d"], file=str(rel)))

    issues.extend(_scan_v32_v33_ast(fp, "\n".join(lines), rel))
    return issues


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VERBOTEN-Linter v4")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional files or directories to scan (defaults to the whole project)",
    )
    parser.add_argument("--ci", action="store_true", help="CI mode: exit 1 on issues")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--errors-only", action="store_true", help="Show only ERROR severity")
    args = parser.parse_args()

    _scan_roots = [Path(p).resolve() for p in args.paths] if args.paths else [_PROJECT_ROOT]
    _py_file_map: dict[str, Path] = {}
    for _scan_root in _scan_roots:
        if _scan_root.is_file():
            if _scan_root.suffix == ".py":
                _py_file_map[str(_scan_root)] = _scan_root
        elif _scan_root.exists():
            for _py_file in _scan_root.rglob("*.py"):
                _py_file_map[str(_py_file)] = _py_file
    py_files = list(_py_file_map.values())

    all_issues: dict[str, list[Violation]] = {}
    for pf in py_files:
        iss = scan(pf)
        if iss:
            try:
                _rel = str(pf.relative_to(_PROJECT_ROOT))
            except ValueError:
                _rel = str(pf)
            all_issues[_rel] = iss

    total = sum(len(v) for v in all_issues.values())

    if args.json:
        errors = sum(1 for v in all_issues.values() for i in v if i.severity == "ERROR")
        warnings = total - errors
        print(
            json.dumps(
                {
                    "clean": total == 0,
                    "issues": total,
                    "errors": errors,
                    "warnings": warnings,
                    "files_scanned": len(py_files),
                    "rules_active": len(RULES),
                }
            )
        )
        return 0

    if total:
        if args.errors_only:
            err_issues = {f: [i for i in iss if i.severity == "ERROR"] for f, iss in all_issues.items()}
            err_issues = {f: iss for f, iss in err_issues.items() if iss}
            if err_issues:
                err_total = sum(len(v) for v in err_issues.values())
                print(f"\n{err_total} ERRORS in {len(err_issues)} files:")
                for f, iss in sorted(err_issues.items()):
                    for i in iss:
                        print(f"  {i}")
            else:
                print("VERBOTEN-Linter: no errors")
            return 1 if args.ci and err_issues else 0
        print(f"\n{total} issues in {len(all_issues)} files:")
        for f, iss in sorted(all_issues.items()):
            for i in iss:
                print(f"  {i}")
        return 1 if args.ci else 0
    else:
        print("VERBOTEN-Linter: clean")
        return 0


if __name__ == "__main__":
    sys.exit(main())
