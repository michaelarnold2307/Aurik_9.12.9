"""tests/unit/test_aurik_orchestrator_wiring.py

Tests für den Zero-Touch-Orchestrierungsvertrag (Spec 23_zero_touch_orchestration_contract.md):
  - P1 gatekeep (Material-Tiers)
  - P2 StreamingDoNoHarm (Harm-Erkennung, Stopp nach 3 konsekutiven Phasen)
  - P4 resolve_assessment / resolve (Widerspruchs-Auflösung, Warnings-Passthrough)
  - Singleton + Preflight-Lifecycle (Watchdog-Aktivierung, Session-Persistenz)
  - Verdrahtungsvertrag in denker/aurik_denker.py (AST-Kontrakt)

Verwendet ausschließlich synthetische Signale; np.random.seed(42) für Reproduzierbarkeit.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.core.aurik_orchestrator import (  # Bootstrap vor Import (siehe oben)
    AurikOrchestrator,
    SessionLearner,
    StreamingDoNoHarm,
    gatekeep,
    get_orchestrator,
    resolve_assessment,
    surgery_first_prune,
)

SR = 48_000


def _tone(duration_s: float = 1.0, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    np.random.seed(42)
    t = np.linspace(0, duration_s, int(SR * duration_s), dtype=np.float32, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _hard_limited(x: np.ndarray) -> np.ndarray:
    """Stark komprimiertes Signal → Crest-Crash + HPI-Einbruch im Watchdog."""
    return (np.sign(x) * 0.5).astype(np.float32)


# ─── P1: Gatekeeper ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rs", "mode", "max_phases"),
    [
        (20.0, "passthrough", 0),
        (40.0, "conservative", 8),
        (55.0, "repair_only", 20),
        (80.0, "full", 50),
    ],
)
def test_gatekeep_tiers(rs: float, mode: str, max_phases: int) -> None:
    decision = gatekeep(
        restorability_score=rs,
        transfer_chain_depth=1,
        bandwidth_loss=0.0,
        snr_db=40.0,
        terminal_codec=None,
        material_type="tape",
        is_restoration_mode=True,
    )
    assert decision.mode == mode
    assert decision.max_phases == max_phases


def test_gatekeep_conservative_excludes_enhancement() -> None:
    decision = gatekeep(
        restorability_score=40.0,
        transfer_chain_depth=1,
        bandwidth_loss=0.0,
        snr_db=40.0,
        terminal_codec=None,
        material_type="tape",
        is_restoration_mode=True,
    )
    assert decision.mode == "conservative"
    assert "denoise" in decision.allowed_phase_families
    assert "air_band" not in decision.allowed_phase_families
    assert "diffusion" not in decision.allowed_phase_families


# ─── P5: Surgery-First ───────────────────────────────────────────────────────


def test_surgery_first_prune_passthrough_removes_all() -> None:
    decision = gatekeep(
        restorability_score=10.0,
        transfer_chain_depth=1,
        bandwidth_loss=0.0,
        snr_db=30.0,
        terminal_codec=None,
        material_type="shellac",
        is_restoration_mode=True,
    )
    pruned, removed = surgery_first_prune(["phase_03_denoise", "phase_39_air_band"], decision)
    assert pruned == []
    assert len(removed) == 2


def test_surgery_first_prune_repair_priority_under_cap() -> None:
    decision = gatekeep(
        restorability_score=80.0,
        transfer_chain_depth=1,
        bandwidth_loss=0.0,
        snr_db=45.0,
        terminal_codec=None,
        material_type="vinyl",
        is_restoration_mode=True,
    )
    decision.max_phases = 3
    pruned, removed = surgery_first_prune(
        [
            "phase_39_air_band",
            "phase_03_denoise",
            "phase_23_spectral_repair",
            "phase_38_stereo_enhance",
        ],
        decision,
    )
    # Repair zuerst, dann Enhancement; risky (spectral_repair) fällt raus.
    assert pruned == ["phase_03_denoise", "phase_39_air_band", "phase_38_stereo_enhance"]
    assert "phase_23_spectral_repair" in removed


# ─── P2: StreamingDoNoHarm ───────────────────────────────────────────────────


def test_watchdog_stops_after_three_consecutive_harmful() -> None:
    tone = _tone()
    watchdog = StreamingDoNoHarm(tone, SR)
    for i in range(2):
        result = watchdog.watch(f"phase_{i}", tone, _hard_limited(tone))
        assert result.phase_was_harmful
        assert result.continue_pipeline
    third = watchdog.watch("phase_2", tone, _hard_limited(tone))
    assert third.phase_was_harmful
    assert not third.continue_pipeline
    assert "STOP" in third.reason


def test_watchdog_neutral_phase_resets_counter() -> None:
    tone = _tone()
    watchdog = StreamingDoNoHarm(tone, SR)
    watchdog.watch("p1", tone, _hard_limited(tone))
    watchdog.watch("p2", tone, _hard_limited(tone))
    neutral = watchdog.watch("p3", tone, tone)
    assert not neutral.phase_was_harmful
    assert neutral.continue_pipeline
    watchdog.watch("p4", tone, _hard_limited(tone))
    watchdog.watch("p5", tone, _hard_limited(tone))
    # Erst die dritte konsekutive schädliche Phase stoppt.
    stopped = watchdog.watch("p6", tone, _hard_limited(tone))
    assert not stopped.continue_pipeline


def test_after_phase_without_preflight_is_documented_noop() -> None:
    orchestrator = AurikOrchestrator()
    result = orchestrator.after_phase("phase_03_denoise", _tone(), _tone())
    assert result.continue_pipeline
    assert not result.phase_was_harmful
    assert orchestrator.watchdog is None


# ─── P3/P4: Lifecycle & Assessment ───────────────────────────────────────────


def test_resolve_reverted_verdict_degraded() -> None:
    assessment = resolve_assessment(
        mushra_score=50.0,
        quality_gate_delta=0.0,
        hpi_score=0.5,
        artifact_freedom=0.5,
        naturalness=-0.3,
        restorability_score=60.0,
        was_reverted=True,
        phases_run=4,
        time_s=10.0,
        warnings=[],
    )
    assert assessment.overall_verdict == "degraded"
    assert assessment.confidence == 0.95


def test_resolve_mushra_hpi_contradiction_prefers_hpi() -> None:
    assessment = resolve_assessment(
        mushra_score=85.0,
        quality_gate_delta=6.0,
        hpi_score=0.5,
        artifact_freedom=0.8,
        naturalness=0.6,
        restorability_score=60.0,
        was_reverted=False,
        phases_run=3,
        time_s=10.0,
        warnings=[],
    )
    assert assessment.overall_verdict == "unchanged"
    assert any("false-positive" in w for w in assessment.warnings)


def test_orchestrator_full_lifecycle(tmp_path: Path) -> None:
    orchestrator = AurikOrchestrator()
    # Isolation: SessionLearner lädt beim Konstruieren die reale Memory-Datei
    # (~/.aurik/session_memory.json) mit echten Produktions-Einträgen — für den
    # Test explizit auf leeren Zustand setzen.
    orchestrator.learner._memory = {"songs": []}
    tone = _tone()
    pruned, decision = orchestrator.preflight(
        original_audio=tone,
        sample_rate=SR,
        restorability_score=80.0,
        transfer_chain_depth=1,
        bandwidth_loss=0.0,
        snr_db=45.0,
        terminal_codec="flac",
        material_type="vinyl",
        is_restoration_mode=True,
        selected_phases=["phase_03_denoise", "phase_39_air_band"],
    )
    assert decision.mode == "full"
    assert len(pruned) == 2
    assert orchestrator.watchdog is not None

    watch = orchestrator.after_phase("phase_03_denoise", tone, _hard_limited(tone))
    assert watch.phase_was_harmful

    assessment = orchestrator.resolve(
        mushra_score=85.0,
        quality_gate_delta=15.0,
        hpi_score=0.8,
        artifact_freedom=0.9,
        naturalness=0.8,
        restorability_score=80.0,
        was_reverted=False,
        phases_run=2,
        warnings=["hinweis-1"],
    )
    assert assessment.overall_verdict == "improved"
    assert "hinweis-1" in assessment.warnings

    with patch.object(SessionLearner, "_MEMORY_PATH", Path(tmp_path) / "session_memory.json"):
        orchestrator.close_session()
    memory_path = Path(tmp_path) / "session_memory.json"
    assert memory_path.exists()
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    assert len(memory["songs"]) == 1
    assert memory["songs"][0]["verdict"] == "improved"
    assert memory["songs"][0]["material"] == "vinyl"


def test_get_orchestrator_singleton() -> None:
    assert get_orchestrator() is get_orchestrator()


# ─── Verdrahtungsvertrag (Spec 23): Denker ruft preflight + close_session ────


def _collect_call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def test_denker_orchestrator_wiring_contract() -> None:
    src_path = _REPO_ROOT / "denker" / "aurik_denker.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    orchestriere = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_orchestriere"
        ),
        None,
    )
    assert orchestriere is not None, "_orchestriere() nicht gefunden"
    calls = _collect_call_names(orchestriere)
    assert "preflight" in calls, "Orchestrator-Preflight fehlt in _orchestriere (Spec 23)"
    assert "close_session" in calls, "Orchestrator-close_session fehlt in _orchestriere (Spec 23)"
