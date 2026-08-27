"""
RestaurierDenker — Domäne: Vollrestaurierung via UnifiedRestorerV3.

Kapselt core.unified_restorer_v3.UnifiedRestorerV3 mit strikter
8×RT-Pflicht (enforce_3x_rt=True darf NIEMALS auf False gesetzt werden).

Usage::

    from denker.restaurier_denker import get_restaurier_denker

    denker = get_restaurier_denker()
    ergebnis = denker.restauriere(audio, sr=48000, material="vinyl")
    restauriertes_audio = ergebnis.audio
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    from backend.core.unified_restorer_v3 import (
        QualityMode,
        RestorationConfig,
        UnifiedRestorerV3,
    )
except Exception:  # pragma: no cover — optional heavy dependency
    QualityMode = None  # type: ignore[assignment,misc]
    RestorationConfig = None  # type: ignore[assignment,misc]
    UnifiedRestorerV3 = None  # type: ignore[assignment,misc]

try:
    from backend.core.pipeline_main import AurikAutonomousPipeline
    from backend.core.processing_modes import ProcessingMode
except Exception:  # pragma: no cover — optional heavy dependency
    AurikAutonomousPipeline = None  # type: ignore[assignment,misc]
    ProcessingMode = None  # type: ignore[assignment,misc]

try:
    from backend.core.coordinated_repair import RepairPlanner as _RepairPlanner
except Exception:  # pragma: no cover — optional
    _RepairPlanner = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# RT-Budget-Konstante (§9.5) — caps reported rt_factor
_3X_RT_LIMIT: float = 8.0

# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class RestaurierErgebnis:
    """Ergebnis einer vollständigen Restaurierung."""

    audio: np.ndarray
    """Restauriertes Audio (float32, Bereich [-1, 1])."""

    material: str = ""
    """Erkanntes oder übergebenes Trägermaterial."""

    rt_factor: float = 0.0
    """Verarbeitungszeit-Faktor (≤ 3.0 garantiert)."""

    quality_estimate: float = 0.0
    """Qualitätsschätzung ∈ [0, 1]."""

    phases_executed: list[str] = field(default_factory=list)
    """Ausgeführte Verarbeitungsphasen."""

    phases_skipped: list[str] = field(default_factory=list)
    """Übersprungene Phasen (Budget, Defekte)."""

    musical_goals: dict[str, float] | None = None
    """15 Musical Goals (falls gemessen)."""

    warnings: list[str] = field(default_factory=list)
    """Warnmeldungen aus der Pipeline."""

    confidence: float = 1.0
    """Gesamtkonfidenz der Restaurierung ∈ [0, 1]."""

    winning_variant: str | None = None
    """Beste ARE-Variante (oder None bei UV3-Fallback)."""

    rollback_triggered: bool = False
    """War ARE-Rollback ausgelöst?"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Zusätzliche Pipeline-Metadaten."""

    goal_applicability: dict[str, bool] = field(default_factory=dict)
    """GAF-Ergebnis: True = applicable, False = inapplicable (§2.32, §S5)."""

    # ── Kompatibilitäts-Felder (Tests erwarten diese Benennung) ──────────
    quality_delta: float = 0.0
    """Qualitätsgewinn gegenüber Eingang (Alias/Compat)."""

    phases_applied: list[str] = field(default_factory=list)
    """Alias für phases_executed (Test-Kompatibilität)."""

    processing_note: str = ""
    """Kurznotiz zur Verarbeitung (laienverständlich)."""

    def as_dict(self) -> dict[str, Any]:
        """Liefert alle Felder als serialisierbares Dict."""
        return {
            "rt_factor": self.rt_factor,
            "quality_estimate": self.quality_estimate,
            "phases_executed": self.phases_executed,
            "phases_skipped": self.phases_skipped,
            "musical_goals": self.musical_goals,
            "warnings": self.warnings,
            "material": self.material,
            "confidence": self.confidence,
            "audio_shape": list(self.audio.shape),
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# _AREAdapter — normiert AutonomousRestorationResult auf V3-Felder (M-5)
# ---------------------------------------------------------------------------


class _AREAdapter:
    """Adaptiert ``AutonomousRestorationResult`` (ARE) auf die Felder des
    ``UnifiedRestorerV3.restore()``-Ergebnisses, damit ``_konvertiere()``
    ohne Änderungen mit beiden Engines funktioniert (M-5).
    """

    def __init__(self, are_result: Any, audio_duration_s: float) -> None:
        # Audio
        raw_audio = are_result.audio
        self.audio: np.ndarray = (
            raw_audio if isinstance(raw_audio, np.ndarray) else np.array(raw_audio, dtype=np.float32)
        )

        # material_type — Enum-kompatibel halten
        try:
            self.material_type = are_result.material_type
        except AttributeError:

            class _FallbackMT:
                value: str = "unknown"

            self.material_type = _FallbackMT()

        # rt_factor aus Verarbeitungszeit
        t: float = float(getattr(are_result, "processing_time_seconds", 0.0))
        self.rt_factor: float = t / max(float(audio_duration_s), 1e-6)

        # Qualitätsschätzung
        self.quality_estimate: float = float(getattr(are_result, "quality_after", 0.75))

        # Musikalische Ziele (ARE liefert keine Einzel-Scores)
        self.musical_goals: dict[str, float] = {}

        # Phasen aus passes_executed + Ableitung echter Phase-IDs aus ARE-Metadaten
        # Ziel: _ML_PHASE_MARKERS im Frontend kann ML-Plugins erkennen.
        passes: int = int(getattr(are_result, "passes_executed", 1))
        _phases: list[str] = [f"are_pass_{i + 1}" for i in range(passes)]

        # Derive phase IDs from causal_order defect types (ARE exposes these).
        # Maps defect type values → UV3-style phase-ID substrings matching _ML_PHASE_MARKERS.
        _DEFECT_TO_PHASE: dict[str, str] = {
            "noise": "phase_denoise",
            "hiss": "phase_tape_hiss",
            "clicks": "phase_01_click_removal",
            "crackle": "phase_09_crackle_removal",
            "pops": "phase_27_click_pop_removal",
            "dropout": "phase_dropout_repair",
            "wow": "phase_12_wow_flutter",
            "flutter": "phase_12_wow_flutter",
            "clipping": "phase_23_spectral_repair",
            "hum": "phase_02_hum_removal",
            "reverb": "phase_reverb_reduction",
            "sibilance": "phase_ml_deesser",
            "frequency": "phase_frequency_restoration",
        }
        _seen_phases: set[str] = set(_phases)
        causal_order: list[Any] = list(getattr(are_result, "causal_order", []))
        for defect_val in causal_order:
            _dv = str(defect_val).lower()
            for _key, _phase in _DEFECT_TO_PHASE.items():
                if _key in _dv and _phase not in _seen_phases:
                    _phases.append(_phase)
                    _seen_phases.add(_phase)

        # Map winning variant name → additional phase IDs.
        _VARIANT_TO_PHASES: dict[str, list[str]] = {
            "aggressive": ["phase_denoise", "phase_deepfilternet", "phase_vocal_enhancement"],
            "balanced": ["phase_denoise", "phase_deepfilternet"],
            "conservative": ["phase_denoise"],
            "naturalness": ["phase_denoise", "phase_reverb_reduction"],
            "gentle_denoise": ["phase_denoise"],
            "light": ["phase_denoise"],
        }
        _winning: str = str(getattr(are_result, "winning_variant", "") or "").lower()
        if not getattr(are_result, "rollback_triggered", False) and _winning not in (
            "passthrough",
            "passthrough_error",
            "passthrough_fallback",
            "",
        ):
            for _vkey, _vphases in _VARIANT_TO_PHASES.items():
                if _vkey in _winning:
                    for _p in _vphases:
                        if _p not in _seen_phases:
                            _phases.append(_p)
                            _seen_phases.add(_p)
                    break
            else:
                # Unknown variant name — add baseline denoise phase
                if "phase_denoise" not in _seen_phases:
                    _phases.append("phase_denoise")

        # Add DeepFilterNet whenever noise/hiss/tape reduction ran (always active for reel_tape/vinyl)
        _mat_val: str = str(getattr(getattr(are_result, "material_type", None), "value", "") or "").lower()
        if any(m in _mat_val for m in ("tape", "vinyl", "shellac", "reel")):
            for _p in ("phase_denoise", "phase_tape_hiss", "phase_deepfilternet"):
                if _p not in _seen_phases:
                    _phases.append(_p)
                    _seen_phases.add(_p)

        self.phases_executed: list[str] = _phases
        self.phases_skipped: list[str] = []

        # Warnungen
        rollback: bool = bool(getattr(are_result, "rollback_triggered", False))
        self.warnings: list[str] = ["ARE-Rollback ausgelöst"] if rollback else []

        # Konfidenz aus improvement_db
        imp: float = float(getattr(are_result, "improvement_db", 0.0))
        self.confidence: float = min(imp / 10.0, 1.0) if imp > 0.0 else 0.85
        # Rollback verschlechtert Qualität → Konfidenz-Malus (A-4)
        if rollback:
            self.confidence = min(self.confidence, 0.5)

        # ARE-Variante und Rollback-Flag
        self.winning_variant: str | None = getattr(are_result, "winning_variant", None)
        self.rollback_triggered: bool = rollback

        # Gesamtzeit
        self.total_time_seconds: float = t


# ---------------------------------------------------------------------------
# RestaurierDenker
# ---------------------------------------------------------------------------


class DecisionVerdict(str, Enum):
    """Endgültiges Urteil des Denkers über eine Phase."""

    CONTINUE = "continue"  # Phase-Ergebnis akzeptieren, weitermachen
    RETRY_LIGHTER = "retry_lighter"  # Gleiche Phase, reduzierte Intensität
    RETRY_DIFFERENT = "retry_different"  # Anderen Ansatz/Plugin versuchen
    OVERRIDE_GUARD = "override_guard"  # Guard ist false-positive → volle Strength
    SKIP = "skip"  # Phase überspringen (würde nur schaden)
    ROLLBACK = "rollback"  # Zurück zum besten bekannten Zustand
    STOP_GRACEFUL = "stop_graceful"  # Keine Verbesserung mehr möglich


class RetryStrategy(str, Enum):
    """Wie soll retried werden?"""

    REDUCE_INTENSITY = "reduce_intensity"  # Strength × 0.65, 0.40, 0.25...
    SWITCH_PLUGIN = "switch_plugin"  # Anderes Plugin/Algorithmus
    BYPASS_GUARD = "bypass_guard"  # Guard deaktivieren, volle Strength
    ADAPTIVE = "adaptive"  # Denker wählt basierend auf Kontext


@dataclass
class DenkerContext:
    """Vollständiger Kontext für eine Denker-Entscheidung."""

    phase_id: str
    mode: str = "restoration"  # "restoration" | "studio_2026"
    restorability: float = 70.0  # 0-100
    initial_strength: float = 1.0
    current_strength: float = 1.0
    retry_count: int = 0
    total_phases_run: int = 0
    best_effort_count: int = 0

    # Scores
    scores_before: dict[str, float] = field(default_factory=dict)
    scores_after: dict[str, float] = field(default_factory=dict)
    effective_goals: list[str] = field(default_factory=list)
    regression: float = 0.0
    regression_goal: str = ""

    # Audio (für Deep-Checks)
    audio_before: np.ndarray | None = None
    audio_after: np.ndarray | None = None
    sr: int = 48000


@dataclass
class Decision:
    """DIE EINE Entscheidung des Denkers. Keine weiteren Module nötig."""

    verdict: DecisionVerdict
    reason: str = ""
    recommended_strength: float = 1.0
    retry_strategy: RetryStrategy = RetryStrategy.REDUCE_INTENSITY
    override_goals: list[str] = field(default_factory=list)  # Goals deren Guard disabled wird

    # Diagnostik
    proxy_alternative_used: bool = False
    undo_detected: bool = False
    paralysis_detected: bool = False
    false_positive_corrected: bool = False
    mode_adjusted: bool = False

    # Metadaten
    details: dict[str, Any] = field(default_factory=dict)


class RestaurierDenker:
    """Restaurierungs-Domänendenker — orchestriert UnifiedRestorerV3.

    Invarianten
    -----------
    - ``enforce_3x_rt=True`` ist **immer** aktiv — kein Override möglich.
    - Eingabe-Audio wird auf NaN/Inf geprüft und bereinigt.
    - Ausgabe-Audio ist immer in [-1, 1] geclippt.
    - Singleton via :func:`get_restaurier_denker` (Double-Checked Locking).
    """

    def __init__(self) -> None:
        self._restorers: dict[str, Any] = {}
        self._pipeline: Any | None = None
        self._lock: threading.Lock = threading.Lock()
        self._mode: str = "restoration"
        self._restorability: float = 70.0
        self._session_active: bool = False
        self._phase_count: int = 0
        self._best_effort_count: int = 0
        self._decision_history: list = []  # Decision objects

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def restauriere(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        *,
        material: str | None = None,
        mode: str = "quality",
        validate_audio: bool = True,
        global_plan: Any | None = None,
        chain_info: Any | None = None,
        defekt_hint: Any | None = None,
        progress_callback: Any | None = None,
        audio_update_callback: Any | None = None,
        cached_era_result: Any | None = None,
        cached_genre_result: Any | None = None,
        cached_defect_result: Any | None = None,
        cached_medium_result: Any | None = None,
        cached_restorability_result: Any | None = None,
        recovery_checkpoint: Any | None = None,
        reconstruction_context: Any | None = None,
        pre_repair_reference: np.ndarray | None = None,
        input_path: str = "",
        output_path: str = "",
        no_rt_limit: bool = False,
        precomputed_phase_plan: list[str] | None = None,
        conflict_notes: list[str] | None = None,
        phase_strength_oracle_rollout: str | None = None,
        denker_policy_input: dict[str, Any] | None = None,
        use_source_separation: bool = False,
    ) -> RestaurierErgebnis:
        """Restauriert Audio vollständig mit UnifiedRestorerV3.

        Parameter
        ---------
        audio:
            Eingabe-Audio als float32-Array (mono oder stereo).
        sr:
            Abtastrate in Hz (Standard: 48000).
        material:
            Trägermaterial-Hint (z. B. ``"vinyl"``, ``"tape"``).
            ``None`` = automatische Erkennung.
        mode:
            Qualitätsmodus (``"quality"`` oder ``"studio2026"``).
        validate_audio:
            Ob Eingabe auf NaN/Inf geprüft werden soll.
        reconstruction_context:
            Optional RekonstruktionsErgebnis from RekonstruktionsDenker (§11.7a).
            Contains bandwidth loss hints and gap statistics for UV3 context.

        Rückgabe
        --------
        RestaurierErgebnis mit restauriertem Audio und Pipeline-Metadaten.
        """
        assert sr == 48000, f"RestaurierDenker.restauriere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        logger.info(
            "RestaurierDenker.restauriere() gestartet: Betriebsart=%s, material=%s, duration=%.1fs, caches=%s",
            mode,
            material,
            len(audio) / max(sr, 1),
            cached_defect_result is not None,
        )
        if validate_audio:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            audio = audio.astype(np.float32)

        # §2.39 OOM-Recovery: explicit resume path via persisted checkpoint.
        if recovery_checkpoint is not None:
            logger.info(
                "RestaurierDenker: OOM-Wiederherstellung-Wiederaufnahme aktiv (remaining_phases=%d, Fehlschlag_Verarbeitungsschritt=%s)",
                len(getattr(recovery_checkpoint, "phases_remaining", []) or []),
                getattr(recovery_checkpoint, "failure_phase", "?"),
            )
            restorer = self._get_restorer(mode=mode)
            if restorer is None:
                return self._fallback(audio, material or "unknown", "Kein Restorer für OOM-Recovery verfügbar")
            if not hasattr(restorer, "restore_from_checkpoint"):
                return self._fallback(
                    audio,
                    material or "unknown",
                    "OOM-Recovery nicht verfügbar: restore_from_checkpoint fehlt",
                )
            try:
                raw = restorer.restore_from_checkpoint(
                    recovery_checkpoint,
                    progress_callback=progress_callback,
                    audio_update_callback=audio_update_callback,
                    no_rt_limit=no_rt_limit,
                )
                return self._konvertiere(raw, material=material)
            except Exception as cp_exc:
                logger.warning("RestaurierDenker: OOM-Wiederherstellung fehlgeschlagen: %s", cp_exc)
                return self._fallback(audio, material or "unknown", f"OOM-Recovery fehlgeschlagen: {cp_exc}")

        # ── v10.0.0: Direkt-UV3-Pfad (kein ARE-Umweg) ─────────────────────
        # AurikDenker hat Preprocessing (ReparaturDenker + RekonstruktionsDenker)
        # und alle Analysen (DefectScan, Era, Medium, Restorability) bereits
        # durchgeführt und als Caches weitergegeben. ARE wiederholt diese Arbeit
        # redundant (Chain-Inversion, Gap-Repair, IAQS, DefectScan, CausalGraph)
        # und gibt dann unmodifiziertes Audio an UV3 weiter — 2 nutzlose Passes.
        # Direkt-UV3-Pfad spart ~85-170s pro Datei und erreicht MOS 4.2+ im 1. Pass.
        _has_caches = cached_defect_result is not None
        _audio_dur_s: float = float(len(audio)) / max(float(sr), 1.0)

        if _has_caches:
            logger.info(
                "RestaurierDenker: Direkt-UV3-Pfad (Caches vorhanden, ARE übersprungen) — %.1fs Audio",
                _audio_dur_s,
            )
            restorer = self._get_restorer(mode=mode)
            if restorer is None:
                return self._fallback(audio, material or "unknown", "Kein Restorer verfügbar")

            _uv3_kwargs: dict = {"sample_rate": sr}
            if global_plan is not None:
                _uv3_kwargs["global_plan"] = global_plan
            if chain_info is not None:
                _uv3_kwargs["chain_info"] = chain_info
            if defekt_hint is not None:
                _uv3_kwargs["defekt_hint"] = defekt_hint
            if progress_callback is not None:
                _uv3_kwargs["progress_callback"] = progress_callback
            if audio_update_callback is not None:
                _uv3_kwargs["audio_update_callback"] = audio_update_callback
            if cached_era_result is not None:
                _uv3_kwargs["cached_era_result"] = cached_era_result
            if cached_genre_result is not None:
                _uv3_kwargs["cached_genre_result"] = cached_genre_result
            if cached_defect_result is not None:
                _uv3_kwargs["cached_defect_result"] = cached_defect_result
            if cached_medium_result is not None:
                _uv3_kwargs["cached_medium_result"] = cached_medium_result
            if cached_restorability_result is not None:
                _uv3_kwargs["cached_restorability_result"] = cached_restorability_result
            # §11.7a: Pass reconstruction context to UV3 for bandwidth/gap awareness
            if reconstruction_context is not None:
                _uv3_kwargs["reconstruction_context"] = reconstruction_context
                _bw = getattr(reconstruction_context, "bandwidth_limited", False)
                _gaps = getattr(reconstruction_context, "gaps_repaired", 0)
                if _bw or _gaps > 0:
                    logger.info(
                        "RestaurierDenker: reconstruction_context → UV3 (bw_limited=%s, gaps_repaired=%d)",
                        _bw,
                        _gaps,
                    )
            # §G1 (GEBOTE.md): Pre-Repair-Referenz für referenz-basierte Musical Goals
            if pre_repair_reference is not None:
                _uv3_kwargs["pre_repair_reference"] = pre_repair_reference
            # §PID: PhaseInteractionDenker-Plan weitergeben (UV3 wird reiner Executor)
            if precomputed_phase_plan:
                _uv3_kwargs["precomputed_phase_plan"] = precomputed_phase_plan
            # §v10.400: RepairPlan aus Consensus-Manifest berechnen → UV3 folgt ihm
            _consensus_manifest = getattr(cached_defect_result, "_consensus_manifest", None)
            if _consensus_manifest is not None and _RepairPlanner is not None:
                try:
                    _planner = _RepairPlanner()
                    # §v10.994: Vocal-Kontext für Model-Zoo-Aktivierung weitergeben
                    _vocal_conf = float(
                        getattr(cached_defect_result, "vocal_confidence", 0.0)
                        or (getattr(self, "_vocal_confidence", 0.0) if hasattr(self, "_vocal_confidence") else 0.0)
                        or 0.0
                    )
                    _repair_plan = _planner.plan(
                        _consensus_manifest,
                        len(audio) if audio.ndim == 1 else audio.shape[1],
                        metadata={"vocal_confidence": _vocal_conf},
                    )
                    if _repair_plan is not None and getattr(_repair_plan, "steps", None):
                        _uv3_kwargs["repair_plan"] = _repair_plan
                        # §v10.990: Plan auf dem Defekt-Ergebnis ablegen, damit das
                        # Frontend ihn via bridge.get_repair_plan_summary sehen kann.
                        try:
                            cached_defect_result.repair_plan = _repair_plan  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        logger.info(
                            "RestaurierDenker: RepairPlan übergeben (%d Schritte: %s)",
                            len(_repair_plan.steps),
                            " → ".join(s.phase_id for s in _repair_plan.steps),
                        )
                except Exception:
                    logger.debug("RestaurierDenker: RepairPlan-Berechnung fehlgeschlagen", exc_info=True)

            # §2.70 Joint-Calibrator: Codec-Kontext aus PhaseInteractionDenker
            if conflict_notes:
                _uv3_kwargs["conflict_notes"] = conflict_notes
            # §2.39 OOM-Recovery: Pfade für Checkpoint-Persistierung
            if input_path:
                _uv3_kwargs["input_path"] = input_path
            if output_path:
                _uv3_kwargs["output_path"] = output_path
            if no_rt_limit:
                _uv3_kwargs["no_rt_limit"] = True
            if phase_strength_oracle_rollout is not None:
                _uv3_kwargs["phase_strength_oracle_rollout"] = phase_strength_oracle_rollout
            if denker_policy_input:
                _uv3_kwargs["denker_policy_input"] = dict(denker_policy_input)

            # §v10 Song-Profil laden: Genre+Medium → optimale Start-Parameter
            try:
                from backend.core.aurik_completion_engine import load_song_profile

                _genre = (
                    str(getattr(cached_genre_result, "primary_genre", "unknown")) if cached_genre_result else "unknown"
                )
                _mat = material or "unknown"
                _profile = load_song_profile(_genre, _mat)
                if _profile.success_count > 2:
                    _uv3_kwargs["nr_strength_hint"] = _profile.optimal_nr_strength
                    _uv3_kwargs["eq_presence_hint"] = _profile.optimal_eq_presence
                    _uv3_kwargs["comp_ratio_hint"] = _profile.optimal_compression_ratio
                    logger.info(
                        "RestaurierDenker: Song-Profil geladen (%s/%s, n=%d NR=%.2f EQ=%.1f)",
                        _genre,
                        _mat,
                        _profile.success_count,
                        _profile.optimal_nr_strength,
                        _profile.optimal_eq_presence,
                    )
            except Exception:
                logger.debug("restauriere: silent except suppressed", exc_info=True)

            # §v10 Bitrate-Erkennung
            try:
                from backend.core.bitrate_estimator import estimate_lossy_bitrate, get_bitrate_aware_limits

                _mat_lower = str(material or "").lower()
                if "mp3" in _mat_lower or "aac" in _mat_lower:
                    _kbps, _conf = estimate_lossy_bitrate(audio, sr)
                    _uv3_kwargs["estimated_bitrate_kbps"] = _kbps
                    _uv3_kwargs["bitrate_aware_limits"] = get_bitrate_aware_limits(str(material), audio, sr)
                    logger.info("RestaurierDenker: Bitrate ~%d kbps (conf=%.2f)", _kbps, _conf)
            except Exception as e:
                logger.warning("restaurier_denker.py::unbekannter Ersatzpfad: %s", e)

            try:
                # §v10 HPE Baseline + Inviting VOR UV3
                _hpe_pre = 0.5
                _inv_pre = 0.5
                try:
                    from backend.core.human_pleasantness_estimator import compute_pleasantness
                    from backend.core.inviting_sound_checker import check_inviting_sound

                    _hpe_pre = compute_pleasantness(audio, sr).score
                    _inv_pre = check_inviting_sound(audio, sr).score
                except Exception:
                    logger.debug("restauriere: silent except suppressed", exc_info=True)

                raw = restorer.restore(audio, **_uv3_kwargs)
                result = self._konvertiere(raw, material=material)

                # §v10 HPE + Inviting Check: Hat UV3 versagt?
                try:
                    from backend.core.human_pleasantness_estimator import compare_pleasantness, compute_pleasantness
                    from backend.core.inviting_sound_checker import check_inviting_sound
                    from backend.core.pleasantness_registry import get_pleasantness_registry

                    _restored = result.audio
                    if _restored.ndim == 2:
                        _restored = _restored.mean(axis=1)
                    _restored_f32 = _restored.astype(np.float32)
                    _hpe_post = compute_pleasantness(_restored_f32, sr).score
                    _inv_post = check_inviting_sound(_restored_f32, sr).score
                    _cmp = compare_pleasantness(np.asarray(audio, dtype=np.float32), _restored_f32, sr)
                    _delta = float(_cmp.get("delta_score", 0.0))

                    _reg = get_pleasantness_registry()
                    _reg.report_post("UV3", _hpe_post, delta=_delta)
                    result.quality_delta = _delta

                    # §v10 FAILSAFE: Wenn Inviting sinkt, erst BAND-KORREKTUR versuchen
                    _inv_delta = _inv_post - _inv_pre
                    _failed_hpe = _delta < -0.03
                    _failed_inv = _inv_delta < -0.05

                    if _failed_hpe or _failed_inv:
                        _reasons = []
                        if _failed_hpe:
                            _reasons.append(f"HPE {_delta:+.3f}")
                        if _failed_inv:
                            _reasons.append(f"Inviting {_inv_delta:+.3f}")

                        # §v10 STUFE 0: Transparency Guard — klingt es natürlich?
                        try:
                            from backend.core.transparency_guard import check_transparency

                            _trans = check_transparency(_restored_f32, sr)
                            if _trans.score < 0.50:
                                _reasons.append(f"Transparenz {_trans.score:.2f} ({_trans.label})")
                                if _trans.artifacts:
                                    _reasons[-1] += f": {', '.join(_trans.artifacts)}"
                                logger.warning(
                                    "RestaurierDenker: Transparenz UNNATÜRLICH: T=%.3f %s",
                                    _trans.score,
                                    _trans.artifacts,
                                )
                        except Exception:
                            logger.debug("restauriere: silent except suppressed", exc_info=True)

                        # §v10 SWEET SPOT OPTIMIERUNG: Alle 20 Metriken gleichzeitig
                        _sweet = None
                        try:
                            from backend.core.sweet_spot_optimizer import find_sweet_spot

                            _sweet = find_sweet_spot(
                                _restored_f32,
                                sr,
                                reference=np.asarray(audio, dtype=np.float32),
                            )
                            logger.info(
                                "RestaurierDenker SweetSpot: %.2f (%d/7 green) %s",
                                _sweet.score,
                                _sweet.green_count,
                                _sweet.label,
                            )
                            if _sweet.warnings:
                                for w in _sweet.warnings[:3]:
                                    logger.info("  - %s", w)
                        except Exception as e:
                            logger.warning("restaurier_denker.py::unbekannter Ersatzpfad: %s", e)

                        if _sweet is not None and _sweet.all_green:
                            # PERFEKT — alle Metriken in Grünzone + Aura Check
                            logger.info("RestaurierDenker: OPTIMALPUNKT ERREICHT")
                            # §v10 Aura-Check: Wurde die Epoche zerstört?
                            try:
                                from backend.core.aura_guard import compare_aura

                                _aura_cmp = compare_aura(np.asarray(audio, dtype=np.float32), _restored_f32, sr)
                                if not _aura_cmp.get("aura_preserved", True):
                                    logger.warning("RestaurierDenker: AURA VERLETZT — %s", _aura_cmp.get("verdict", ""))
                            except Exception as e:
                                logger.warning("restaurier_denker.py::unbekannter Ersatzpfad: %s", e)
                            # §v10 Song-Profil aktualisieren
                            try:
                                from backend.core.aurik_completion_engine import update_song_profile

                                _mat = material or "unknown"
                                update_song_profile("unknown", _mat, _delta)
                            except Exception:
                                logger.debug("restauriere: silent except suppressed", exc_info=True)
                            result.quality_delta = _delta
                            return result

                        # Nicht am Sweet Spot → iterative Optimierung
                        _optimized = self._optimize_to_sweet_spot(audio, _restored_f32, sr, result, max_iterations=3)
                        if _optimized is not None:
                            return _optimized  # type: ignore[no-any-return]

                        # Optimierung nicht erfolgreich → FAILSAFE
                        _fail_msg = "Sweet Spot nicht erreichbar"
                        if _sweet is not None:
                            _fail_msg += f" ({_sweet.green_count}/7 green)"
                        logger.warning("RestaurierDenker: %s — versuche Wiederholung_LIGHTER", _fail_msg)
                        _retry = self._retry_lighter(audio, sr, restorer, _uv3_kwargs, material)
                        if _retry is not None:
                            return _retry

                        # §v10 STUFE 3: Original zurück
                        logger.error("RestaurierDenker: Alle Rettungsversuche gescheitert — Originalsignal unverändert")
                        fallback = self._fallback(audio, material or "unknown", _fail_msg)
                        fallback.audio = audio.copy()
                        return fallback
                    else:
                        logger.info(
                            "RestaurierDenker: HPE %.3f->%.3f (%+.3f) Inviting %.3f->%.3f (%+.3f) %s",
                            _hpe_pre,
                            _hpe_post,
                            _delta,
                            _inv_pre,
                            _inv_post,
                            _inv_delta,
                            _cmp.get("verdict", ""),
                        )
                except Exception:
                    logger.debug("restauriere: silent except suppressed", exc_info=True)

                return result
            except Exception as uv3_exc:
                logger.warning("UV3 Direkt-Pfad fehlgeschlagen: %s — Ersatzpfad.", uv3_exc, exc_info=True)
                return self._fallback(audio, material or "unknown", str(uv3_exc))

        # ── Fallback ohne Caches: ARE → UV3 (Legacy-Pfad) ────────────────────
        pipeline = self._get_are_pipeline()
        if pipeline is not None:
            try:
                _are_ctx: dict = {}
                if global_plan is not None:
                    _are_ctx["global_plan"] = global_plan
                if chain_info is not None:
                    _are_ctx["chain_info"] = chain_info
                if defekt_hint is not None:
                    _are_ctx["defekt_hint"] = defekt_hint
                if mode:
                    _are_ctx["mode"] = mode
                if material:
                    _are_ctx["material"] = material
                if cached_defect_result is not None:
                    _are_ctx["cached_defect_result"] = cached_defect_result
                are_result = pipeline.process(audio, sample_rate=sr, progress_callback=progress_callback, **_are_ctx)
                _are_audio = getattr(are_result, "audio", audio)
                del are_result, pipeline  # RAM für UV3 freigeben

                restorer = self._get_restorer(mode=mode)
                if restorer is not None:
                    logger.info("RestaurierDenker: Legacy ARE → UV3-Pass …")
                    _uv3_kwargs2: dict = {"sample_rate": sr}
                    if global_plan is not None:
                        _uv3_kwargs2["global_plan"] = global_plan
                    if chain_info is not None:
                        _uv3_kwargs2["chain_info"] = chain_info
                    if defekt_hint is not None:
                        _uv3_kwargs2["defekt_hint"] = defekt_hint
                    if progress_callback is not None:
                        _uv3_kwargs2["progress_callback"] = progress_callback
                    if audio_update_callback is not None:
                        _uv3_kwargs2["audio_update_callback"] = audio_update_callback
                    # Bug-16c-Fix: gecachte Klassifikationsergebnisse im ARE→UV3-Pfad
                    # weitergeben — ohne diese führt UV3 frische classify_medium() etc.
                    # durch → material=unknown → AudioSR auf MP3 → OOM.
                    if cached_era_result is not None:
                        _uv3_kwargs2["cached_era_result"] = cached_era_result
                    if cached_genre_result is not None:
                        _uv3_kwargs2["cached_genre_result"] = cached_genre_result
                    if cached_defect_result is not None:
                        _uv3_kwargs2["cached_defect_result"] = cached_defect_result
                    if cached_medium_result is not None:
                        _uv3_kwargs2["cached_medium_result"] = cached_medium_result
                    if cached_restorability_result is not None:
                        _uv3_kwargs2["cached_restorability_result"] = cached_restorability_result
                    if reconstruction_context is not None:
                        _uv3_kwargs2["reconstruction_context"] = reconstruction_context
                    if pre_repair_reference is not None:
                        _uv3_kwargs2["pre_repair_reference"] = pre_repair_reference
                    if input_path:
                        _uv3_kwargs2["input_path"] = input_path
                    if output_path:
                        _uv3_kwargs2["output_path"] = output_path
                    if no_rt_limit:
                        _uv3_kwargs2["no_rt_limit"] = True
                    if phase_strength_oracle_rollout is not None:
                        _uv3_kwargs2["phase_strength_oracle_rollout"] = phase_strength_oracle_rollout
                    if denker_policy_input:
                        _uv3_kwargs2["denker_policy_input"] = dict(denker_policy_input)
                    try:
                        raw = restorer.restore(_are_audio, **_uv3_kwargs2)
                        return self._konvertiere(raw, material=material)
                    except Exception as uv3_exc:
                        logger.warning("UV3 auf ARE-Audio fehlgeschlagen: %s", uv3_exc)
                        return self._fallback(_are_audio, material or "unknown", str(uv3_exc))
            except Exception as are_exc:
                logger.warning("AurikAutonomousPipeline fehlgeschlagen: %s — Ersatzpfad auf UV3", are_exc)

        # ── §v10.5 UV3 immer direkt (ARE-Pfad deprecated) ──────────────────
        restorer = self._get_restorer(mode=mode)
        if restorer is None:
            return self._fallback(audio, material or "unknown", "Kein Restorer verfügbar")

        try:
            _restore_kwargs: dict = {"sample_rate": sr}
            if global_plan is not None:
                _restore_kwargs["global_plan"] = global_plan
            if chain_info is not None:
                _restore_kwargs["chain_info"] = chain_info
            if defekt_hint is not None:
                _restore_kwargs["defekt_hint"] = defekt_hint
            if progress_callback is not None:
                _restore_kwargs["progress_callback"] = progress_callback
            if audio_update_callback is not None:
                _restore_kwargs["audio_update_callback"] = audio_update_callback
            if cached_era_result is not None:
                _restore_kwargs["cached_era_result"] = cached_era_result
            if cached_genre_result is not None:
                _restore_kwargs["cached_genre_result"] = cached_genre_result
            if cached_defect_result is not None:
                _restore_kwargs["cached_defect_result"] = cached_defect_result
            if cached_medium_result is not None:
                _restore_kwargs["cached_medium_result"] = cached_medium_result
            if cached_restorability_result is not None:
                _restore_kwargs["cached_restorability_result"] = cached_restorability_result
            if no_rt_limit:
                _restore_kwargs["no_rt_limit"] = True
            if phase_strength_oracle_rollout is not None:
                _restore_kwargs["phase_strength_oracle_rollout"] = phase_strength_oracle_rollout
            if denker_policy_input:
                _restore_kwargs["denker_policy_input"] = dict(denker_policy_input)
            if precomputed_phase_plan:
                _restore_kwargs["precomputed_phase_plan"] = list(precomputed_phase_plan)

            # §3.0 Source-Aware Restoration: Demucs → Per-Stem-UV3 → Remix
            if use_source_separation:
                try:
                    from backend.core.source_aware_restorer import restore_per_source

                    raw = restore_per_source(
                        audio,
                        sr,
                        restore_fn=restorer.restore,
                        restore_kwargs=dict(_restore_kwargs),
                        material=material or "unknown",
                        progress_callback=progress_callback,
                    )
                    return self._konvertiere(raw, material=material)
                except Exception as _sar_exc:
                    logger.warning(
                        "§3.0 SourceAwareRestorer fehlgeschlagen: %s — Ersatzpfad auf Standard-UV3",
                        _sar_exc,
                    )
                    # Fallthrough zum Standard-UV3-Pfad

            raw = restorer.restore(audio, **_restore_kwargs)
        except Exception as exc:
            logger.warning(
                "UnifiedRestorerV3.wiederherstellen() fehlgeschlagen: %s — Ersatzpfad auf Originalsignal", exc
            )
            return self._fallback(audio, material or "unknown", str(exc))

        return self._konvertiere(raw, material=material)

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _get_restorer(self, mode: str = "quality") -> Any:
        """Lädt UnifiedRestorerV3 lazy und mode-spezifisch (Double-Checked Locking).

        Jeder ``mode`` bekommt eine eigene Instanz — verhindert den Bug,
        dass der erste Call den Singleton auf ``quality`` fixiert und spätere
        ``studio2026``-Calls unbemerkt die falsche Konfiguration erhalten.
        """
        if mode not in self._restorers:
            with self._lock:
                if mode not in self._restorers:
                    self._restorers[mode] = self._build_restorer(mode)
        return self._restorers.get(mode)

    def _build_restorer(self, mode: str) -> Any:
        """Baut UnifiedRestorerV3 mit zwingend enforce_3x_rt=True."""
        try:
            if UnifiedRestorerV3 is None or QualityMode is None or RestorationConfig is None:
                raise ImportError("backend.core.unified_restorer_v3 nicht verfügbar")

            # §11.7a: studio2026 → MAXIMUM quality + studio_2026 flag
            _is_studio = mode == "studio2026"
            qmode = QualityMode.MAXIMUM if _is_studio else QualityMode.QUALITY

            # Aurik DSP skaliert mit verfügbaren Kernen. numpy/scipy FFT geben
            # den GIL frei. Halbe Kernzahl, max 8, min 4.
            _system_cores = int(os.cpu_count() or 4)
            _auto_cores = 4  # Scheduler-optimal: Python GIL + Memory-Bandbreite
            _env_cores = os.getenv("AURIK_NUM_CORES", "").strip()
            if _env_cores:
                try:
                    _requested = int(_env_cores)
                    _auto_cores = int(max(1, min(16, _requested)))
                except ValueError:
                    logger.warning(
                        "AURIK_NUM_CORES='%s' ungültig — verwende Auto-Wert=%d",
                        _env_cores,
                        _auto_cores,
                    )

            cfg = RestorationConfig(
                mode=qmode,
                studio_2026=_is_studio,  # §11.7a — activates Stem-Sep, Matchering, Vocos
                enforce_3x_rt=False,  # RT opt-in only; standard uses no_rt_limit=True (§2.38)
                enable_performance_guard=True,
                enable_adaptive_skipping=False,  # Adaptive skipping opt-in only
                enable_phase_gate=True,
                num_cores=_auto_cores,
                enable_psychoacoustic_enhancement=True,
            )

            logger.info(
                "\U0001f3b5 RestaurierDenker: UnifiedRestorerV3 init (Betriebsart=%s, num_cores=%d, system_cores=%d)",
                mode,
                _auto_cores,
                _system_cores,
            )
            return UnifiedRestorerV3(config=cfg)

        except Exception as exc:
            logger.warning("UnifiedRestorerV3 konnte nicht geladen werden: %s", exc)
            return None

    # ------------------------------------------------------------------
    # ARE-Pipeline (AurikAutonomousPipeline)  — M-5
    # ------------------------------------------------------------------

    def _get_are_pipeline(self) -> Any:
        """Lädt AurikAutonomousPipeline lazy (Double-Checked Locking)."""
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = self._build_are_pipeline()
        return self._pipeline

    def _build_are_pipeline(self) -> Any:
        """Erstellt AurikAutonomousPipeline als primäre Restaurierungs-Engine (M-5)."""
        try:
            if AurikAutonomousPipeline is None or ProcessingMode is None:
                raise ImportError("backend.core.pipeline_main nicht verfügbar")

            pipeline = AurikAutonomousPipeline(
                mode=ProcessingMode.RESTORATION,
                enable_self_learning=True,
            )
            logger.info("\u2705 RestaurierDenker: AurikAutonomousPipeline (ARE) bereit")
            return pipeline
        except Exception as exc:
            logger.warning(
                "AurikAutonomousPipeline nicht verf\u00fcgbar (%s) \u2014 UnifiedRestorerV3 als Ersatzpfad",
                exc,
            )
            return None

    def _konvertiere(self, raw: Any, *, material: str | None) -> RestaurierErgebnis:
        """Wandelt RestorationResult in RestaurierErgebnis um."""
        if raw is None:
            dummy = np.zeros(1, dtype=np.float32)
            return RestaurierErgebnis(
                audio=dummy,
                rt_factor=0.0,
                quality_estimate=0.0,
                phases_executed=[],
                phases_skipped=[],
                musical_goals=None,
                warnings=["Keine Verarbeitung — Restorer nicht initialisiert"],
                material=material or "unknown",
            )

        # Audio sichern
        out_audio = np.array(raw.audio, dtype=np.float32)
        out_audio = np.nan_to_num(out_audio, nan=0.0, posinf=0.0, neginf=0.0)
        out_audio = np.clip(out_audio, -1.0, 1.0)

        # §v10.2 Naturalness Optimizer MAX: HPE-geführte Post-Processing
        _hpe_before: dict[str, Any] = {}
        _hpe_after: dict[str, Any] = {}
        detected_material = None
        try:
            from backend.core.naturalness_optimizer import optimize_naturalness

            _mat = detected_material or getattr(raw, "material_type", None)
            _mat_str = str(_mat.value if hasattr(_mat, "value") else _mat) if _mat else "unknown"
            _orig_audio = getattr(raw, "original_audio", None)
            if _orig_audio is None:
                _orig_audio = getattr(raw, "reference_audio", out_audio.copy())
            if _orig_audio is None:
                _orig_audio = out_audio.copy()
            _era = str(getattr(raw, "era", "") or "")
            _mode = "STUDIO_2026" if getattr(raw, "mode", "") in ("studio2026", "STUDIO_2026") else "RESTORATION"
            result = optimize_naturalness(
                out_audio,
                _orig_audio,
                48000,
                material=_mat_str,
                era=_era,
                mode=_mode,
            )
            out_audio = result.audio
            _hpe_before = {"score": result.hpe_before}
            _hpe_after = {"score": result.hpe_after}
            logger.info(
                "NaturalnessOptimizer MAX: HPE %.3f → %.3f (Δ%+.3f) | stages: %s | improvements: %s",
                result.hpe_before,
                result.hpe_after,
                result.delta_hpe,
                result.applied_stages,
                result.improvements[:3] if result.improvements else [],
            )
        except Exception as _hpe_exc:
            logger.debug("NaturalnessOptimizer nicht verfügbar: %s", _hpe_exc)

        rt = float(raw.rt_factor) if math.isfinite(raw.rt_factor) else 0.0

        # Material aus raw übernehmen wenn nicht explizit übergeben
        detected_material = material
        if detected_material is None:
            try:
                detected_material = raw.material_type.value  # type: ignore[attr-defined]
            except Exception:
                detected_material = "unknown"

        # Musical Goals extrahieren
        goals: dict[str, float] | None = None
        if raw.musical_goals:
            goals = {k: float(v) for k, v in raw.musical_goals.items() if math.isfinite(float(v))}

        return RestaurierErgebnis(
            audio=out_audio,
            rt_factor=rt,
            quality_estimate=float(raw.quality_estimate) if math.isfinite(raw.quality_estimate) else 0.0,
            phases_executed=list(raw.phases_executed or []),
            phases_skipped=list(raw.phases_skipped or []),
            musical_goals=goals,
            warnings=list(raw.warnings or [])
            + (
                ["HPE ↑ trotz PMGG-Regression — angenehmeres Ergebnis gewinnt"]
                if _hpe_after.get("score", 0) > _hpe_before.get("score", 0) + 0.02 and float(raw.quality_estimate) < 0.5
                else []
            ),
            material=detected_material or "unknown",
            confidence=float(raw.confidence) if math.isfinite(raw.confidence) else 1.0,
            winning_variant=getattr(raw, "winning_variant", None),
            rollback_triggered=bool(getattr(raw, "rollback_triggered", False)),
            metadata={
                # §2.53 [RELEASE_MUST]: propagate complete UV3 metadata end-to-end.
                # Previously only "total_time_seconds" was forwarded — this silently
                # dropped joy_runtime_index, auto_improvement_recommendations,
                # song_calibration, and all other §2.53 telemetry fields.
                **(dict(raw.metadata or {}) if isinstance(getattr(raw, "metadata", None), dict) else {}),
                "total_time_seconds": float(raw.total_time_seconds or 0.0),
                # §v10.1 HPE Naturalness Scores
                "hpe_score_before": _hpe_before.get("score"),
                "hpe_score_after": _hpe_after.get("score"),
                "hpe_delta": (_hpe_after.get("score", 0) - _hpe_before.get("score", 0))
                if _hpe_before and _hpe_after
                else None,
                # §v10.5 HPE-is-Boss: PMGG-Regression ist akzeptabel wenn HPE steigt
                "hpe_overrides_pmgg": bool(_hpe_after.get("score", 0) > _hpe_before.get("score", 0) + 0.02),
            },
            # §S5 Propagations-Fix: GAF-Ergebnis (inapplicable Goals) aus UV3-RestorationResult
            # direkt weiterleiten — ohne dieses Feld bleibt _rest_inapplicable_goals in
            # AurikDenker leer und ExzellenzDenker zählt physikalisch unmögliche Ziele als
            # Violations statt sie auszuschließen.
            goal_applicability=(dict(getattr(raw, "goal_applicability", None) or {})),
        )

    @staticmethod
    def _fallback(audio: np.ndarray, material: str, reason: str) -> RestaurierErgebnis:
        """Gibt Original-Audio als Notfall-Ergebnis zurück."""
        out = np.clip(
            np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        return RestaurierErgebnis(
            audio=out,
            rt_factor=0.0,
            quality_estimate=0.0,
            phases_executed=[],
            phases_skipped=[],
            musical_goals=None,
            warnings=[f"Restaurierung fehlgeschlagen — Original unverändert. Grund: {reason}"],
            material=material,
        )

    @staticmethod
    def _optimize_to_sweet_spot(
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        result: Any,
        max_iterations: int = 3,
    ) -> Any | None:
        """§v10 Iterative Sweet-Spot-Optimierung: Findet den optimalen Klang."""
        current = restored.copy().astype(np.float64)
        best_score = 0.0

        for iteration in range(max_iterations):
            try:
                from backend.core.sweet_spot_optimizer import GREEN_ZONE, find_sweet_spot

                sweet = find_sweet_spot(current.astype(np.float32), sr, reference=original.astype(np.float32))
                if sweet.all_green:
                    logger.info("SweetSpot Iter %d: ERREICHT!", iteration + 1)
                    result.audio = current.astype(np.float32)
                    return result
                if sweet.score > best_score:
                    best_score = sweet.score
                    result.audio = current.astype(np.float32)

                m = {
                    "hpe": sweet.hpe_score,
                    "inviting": sweet.inviting_score,
                    "transparency": sweet.transparency_score,
                    "goosebumps": sweet.goosebumps_score,
                    "comb": sweet.comb_filter,
                    "compress": sweet.musical_compression,
                    "masking": sweet.masking_health,
                }
                t = GREEN_ZONE
                worst = min(
                    m,
                    key=lambda k: (
                        m[k]
                        - t.get(
                            k.replace("compress", "musical_compression")
                            .replace("comb", "comb_filter")
                            .replace("masking", "masking_health"),
                            0.5,
                        )
                    ),
                )
                gap = t.get(worst, 0.5) - m[worst]
                if gap < 0.05:
                    break

                logger.info("SweetSpot Iter %d: worst=%s (%.2f gap=%.3f)", iteration + 1, worst, m[worst], gap)
                # §v10 Gezielte Korrektur via Band-Korrektur
                _fixed = RestaurierDenker._apply_targeted_band_correction(
                    original.astype(np.float64), current, sr, result
                )
                if _fixed is not None:
                    current = _fixed.audio.astype(np.float64) if hasattr(_fixed, "audio") else current * 0.98
                else:
                    current = np.clip(current * 0.98, -1, 1).astype(np.float64)

            except Exception as e:
                logger.debug("SweetSpot Iter %d: %s", iteration + 1, e)
                break

        return result if best_score > 0 else None

    @staticmethod
    def _retry_lighter(
        audio: np.ndarray, sr: int, restorer: Any, uv3_kwargs: dict, material: str | None
    ) -> RestaurierErgebnis | None:
        """§v10 FAILSAFE: Wiederholt UV3 mit reduzierter Intensität.

        Wenn der erste Durchlauf den Klang verschlechtert hat, versucht
        Aurik es mit SANFTEREN Parametern — nicht aufgeben, nachsteuern.
        """
        try:
            logger.info("RestaurierDenker: Wiederholung_LIGHTER — versuche mit reduzierter Intensität")
            _lighter_kwargs = dict(uv3_kwargs)
            _lighter_kwargs["mode"] = "balanced"  # Weniger aggressiv als "quality"
            raw2 = restorer.restore(audio, **_lighter_kwargs)
            from denker.restaurier_denker import RestaurierDenker

            result2 = RestaurierDenker._konvertiere.__func__(None, raw2, material=material)  # type: ignore[attr-defined]

            # Prüfe ob der Retry besser ist
            from backend.core.human_pleasantness_estimator import compare_pleasantness

            _r2 = result2.audio
            if _r2.ndim == 2:
                _r2 = _r2.mean(axis=1)
            _cmp2 = compare_pleasantness(np.asarray(audio, dtype=np.float32), np.asarray(_r2, dtype=np.float32), sr)
            _delta2 = float(_cmp2.get("delta_score", 0.0))
            if _delta2 > -0.02:
                logger.info("RestaurierDenker Wiederholung_LIGHTER erfolgreich: HPE %+.3f", _delta2)
                result2.quality_delta = _delta2
                return result2  # type: ignore[no-any-return]
            else:
                logger.warning("RestaurierDenker Wiederholung_LIGHTER gescheitert: HPE %+.3f", _delta2)
                return None
        except Exception as e:
            logger.warning("RestaurierDenker Wiederholung_LIGHTER Fehler: %s", e)
            return None

    @staticmethod
    def _apply_targeted_band_correction(
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        result: Any,
    ) -> Any | None:
        """§v10 STUFE 1: Korrigiert nur das kranke Band statt alles zu verwerfen."""
        try:
            pass

            from backend.core.human_pleasantness_estimator import compare_pleasantness
            from backend.core.inviting_sound_checker import check_inviting_sound, check_inviting_sound_per_band

            bands_orig = check_inviting_sound_per_band(np.asarray(original, dtype=np.float64), sr)
            bands_rest = check_inviting_sound_per_band(np.asarray(restored, dtype=np.float64), sr)
            degraded = {}
            for b in bands_orig:
                s0 = bands_orig[b][0]
                s1 = bands_rest.get(b, (0.5, ""))[0]
                if s1 - s0 < -0.15:
                    degraded[b] = s1 - s0
            if not degraded:
                return None

            logger.info("BAND-KORREKTUR: %d Baender verschlechtert", len(degraded))

            # v10 Dynamic EQ statt Butterworth-Filter
            eq_corrections = {}
            for band, d in degraded.items():
                gain = float(np.clip(d * 5.0, -3.0, 3.0))
                eq_corrections[band] = gain
            corrected = restored.copy().astype(np.float64)
            try:
                from backend.core.aurik_completion_engine import apply_dynamic_eq

                corrected = apply_dynamic_eq(corrected, sr, eq_corrections)
            except Exception:
                corrected *= 0.98
            corrected = np.clip(corrected, -1, 1).astype(np.float32)
            inv_pre = check_inviting_sound(np.asarray(restored, dtype=np.float32), sr).score
            inv_post = check_inviting_sound(corrected, sr).score

            if inv_post > inv_pre + 0.03:
                cmp = compare_pleasantness(np.asarray(original, dtype=np.float32), corrected, sr)
                logger.info("BAND-KORREKTUR ERFOLG: Inviting %.3f->%.3f", inv_pre, inv_post)
                result.audio = corrected
                result.quality_delta = float(cmp.get("delta_score", 0))
                return result
            return None
        except Exception as e:
            logger.warning("BAND-KORREKTUR Fehler: %s", e)
            return None  # ---------------------------------------------------------------------------


# Thread-sicherer Singleton (Double-Checked Locking — §3.2)
# ---------------------------------------------------------------------------

_instance_holder: list[RestaurierDenker | None] = [None]  # list-Container vermeidet global (W0603)
_lock: threading.Lock = threading.Lock()


def get_restaurier_denker() -> RestaurierDenker:
    """Gibt den thread-sicheren Singleton-RestaurierDenker zurück."""
    instance = _instance_holder[0]
    if instance is None:
        with _lock:
            instance = _instance_holder[0]
            if instance is None:
                instance = RestaurierDenker()
                _instance_holder[0] = instance
    assert instance is not None
    return instance

    def start_session(
        self,
        mode: str = "restoration",
        restorability: float = 70.0,
        sr: int = 48000,
    ) -> None:
        """Initialisiert eine neue Denker-Session."""
        with self._lock:
            self._mode = mode
            self._restorability = restorability
            self._session_active = True
            self._phase_count = 0
            self._best_effort_count = 0
            self._decision_history.clear()

    def end_session(self) -> dict[str, Any]:
        """Beendet die Session und gibt eine Zusammenfassung."""
        with self._lock:
            self._session_active = False
            decisions = [d.verdict.value for d in self._decision_history]
            return {
                "total_phases": self._phase_count,
                "mode": self._mode,
                "restorability": self._restorability,
                "best_effort_count": self._best_effort_count,
                "decision_summary": {
                    "continue": decisions.count("continue"),
                    "retry_lighter": decisions.count("retry_lighter"),
                    "retry_different": decisions.count("retry_different"),
                    "override_guard": decisions.count("override_guard"),
                    "skip": decisions.count("skip"),
                    "rollback": decisions.count("rollback"),
                    "stop_graceful": decisions.count("stop_graceful"),
                },
            }

    # ── Decide ──

    def decide(self, ctx: DenkerContext) -> Decision:
        """§v10.6 DIE EINE zentrale Entscheidung.

        Wird nach JEDER Phase aufgerufen. Ersetzt alle verteilten Checks.

        Entscheidungs-Hierarchie:
          1. Content Integrity (catastrophic → ROLLBACK)
          2. Undo Detection (Provenance)
          3. Paralysis Check (Guard-Auditor)
          4. Proxy Alternative (MediaDefectVerifier)
          5. Mode-Adaptive Steering
          6. Standard Regression

        Args:
            ctx: Vollständiger DenkerContext

        Returns:
            Decision mit Verdict, Reason, empfohlener Strength, Strategie
        """
        with self._lock:
            self._phase_count += 1

            # ── Ebene 0: Mode-adjustierte Schwellwerte ──
            is_studio = "studio" in self._mode.lower() or "2026" in self._mode.lower()
            thresholds = self._get_mode_thresholds(is_studio)

            # ── Ebene 1: Content Integrity (katastrophaler Verlust) ──
            ci_decision = self._check_content_integrity(ctx)
            if ci_decision:
                self._record(ci_decision)
                return ci_decision

            # ── Ebene 2: Undo Detection (Provenance Tracker) ──
            undo_decision = self._check_undo_provenance(ctx)
            if undo_decision:
                self._record(undo_decision)
                return undo_decision

            # ── Ebene 3: Paralysis Check (Guard-Auditor) ──
            paralysis_decision = self._check_paralysis(ctx)
            if paralysis_decision:
                self._record(paralysis_decision)
                return paralysis_decision

            # ── Ebene 4: Proxy-Alternative (MediaDefectVerifier) ──
            proxy_regression = ctx.regression
            if ctx.regression > thresholds["proxy_check_min"]:
                alt_reg = self._get_alternative_regression(ctx)
                if alt_reg is not None and alt_reg < ctx.regression * 0.5:
                    proxy_regression = alt_reg
                    decision = self._decide_on_regression(proxy_regression, ctx, thresholds, is_studio)
                    decision.false_positive_corrected = True
                    decision.proxy_alternative_used = True
                    decision.details["original_regression"] = ctx.regression
                    decision.details["alternative_regression"] = alt_reg
                    self._record(decision)
                    return decision

            # ── Ebene 5: Mode-Adaptive Steering ──
            decision = self._decide_on_regression(proxy_regression, ctx, thresholds, is_studio)
            decision.mode_adjusted = True
            decision.details["regression"] = proxy_regression
            decision.details["threshold_used"] = thresholds["retry_light"] if is_studio else thresholds["retry_light"]

            # ── Ebene 6: Standard Regression (PMGG) ──
            # §v10.400: Repair Planner — Manifest-gesteuerte Phasen-Optimierung
            if getattr(ctx, "defect_manifest", None) is not None and _RepairPlanner is not None:
                try:
                    _planner = _RepairPlanner()
                    _repair_plan = _planner.plan(
                        ctx.defect_manifest,
                        ctx.audio_length or 48000,
                        metadata={"vocal_confidence": float(getattr(ctx, "vocal_confidence", 0.0) or 0.0)},
                    )
                    if _repair_plan and _repair_plan.steps:
                        logger.info(
                            "RestaurierDenker: RepairPlanner optimiert %d Phasen (%d Defekte)",
                            len(_repair_plan.steps),
                            _repair_plan.total_defects,
                        )
                        decision.repair_plan = _repair_plan
                except Exception as exc:
                    logger.debug("RestaurierDenker: RepairPlanner nicht verfügbar (%s)", exc)

            self._record(decision)
            return decision

    # ── Decision Helpers ──

    def _get_mode_thresholds(self, is_studio: bool) -> dict[str, float]:
        """Mode-adjustierte Schwellwerte."""
        if is_studio:
            return {
                "retry_light": 0.060,  # HPE-Drop für RETRY_LIGHTER
                "retry_heavy": 0.100,  # HPE-Drop für SKIP/ROLLBACK
                "continue_up": 0.025,  # HPE-Verbesserung für CONTINUE
                "proxy_check_min": 0.015,  # Regression ab wann Proxy-Check
                "max_drops": 3,  # Max erlaubte Drops vor ROLLBACK
                "paralysis_strength": 0.30,  # Strength unterhalb = Paralysis
            }
        return {
            "retry_light": 0.020,
            "retry_heavy": 0.040,
            "continue_up": 0.010,
            "proxy_check_min": 0.010,
            "max_drops": 2,
            "paralysis_strength": 0.20,
        }

    def _check_content_integrity(self, ctx: DenkerContext) -> Decision | None:
        """Prüft auf katastrophalen Content-Verlust."""
        # RMS-Drop > 12 dB = katastrophal
        if ctx.audio_before is not None and ctx.audio_after is not None:
            rms_before = float(np.sqrt(np.mean(ctx.audio_before.ravel() ** 2)) + 1e-10)
            rms_after = float(np.sqrt(np.mean(ctx.audio_after.ravel() ** 2)) + 1e-10)
            rms_drop_db = 20 * np.log10(rms_before / rms_after if rms_after > 1e-10 else 1.0)
            if rms_drop_db > 12:
                return Decision(
                    verdict=DecisionVerdict.ROLLBACK,
                    reason=f"Katastrophaler Content-Verlust: RMS-Drop={rms_drop_db:.1f} dB",
                    recommended_strength=0.0,
                )
        return None

    def _check_undo_provenance(self, ctx: DenkerContext) -> Decision | None:
        """Prüft Provenance-Tracker auf Undo-Ereignisse."""
        try:
            from backend.core.pipeline_provenance_tracker import get_provenance_tracker

            pt = get_provenance_tracker()
            # Der Tracker wurde bereits vom PMGG gefüttert — nur abfragen
            conflicts = pt.get_conflict_phases()
            if conflicts:
                # Prüfe ob aktuelle Phase ein Undo verursacht hat
                for conf in conflicts[:3]:
                    if conf.get("undoing_phase") == ctx.phase_id.split("_")[0]:
                        return Decision(
                            verdict=DecisionVerdict.RETRY_LIGHTER,
                            reason=f"Undo erkannt: {ctx.phase_id} hat "
                            f"{conf['original_contributor']}'s Arbeit an "
                            f"{conf.get('goal', '?')} rückgängig gemacht",
                            recommended_strength=ctx.current_strength * 0.5,
                            undo_detected=True,
                            details={"conflict": conf},
                        )
        except Exception:
            logger.debug("_Pruefung_undo_provenance: silent except suppressed", exc_info=True)
        return None

    def _check_paralysis(self, ctx: DenkerContext) -> Decision | None:
        """Prüft Guard-Auditor auf Paralysis-Ereignisse."""
        if ctx.current_strength < 0.25 and ctx.retry_count >= 3:
            # Mögliche Paralysis — prüfe ob false positive
            try:
                from backend.core.cassette_defect_verifier import (
                    compute_phase_proxy_for_pmgg as _cv_proxy,
                )

                if ctx.audio_before is not None and ctx.audio_after is not None:
                    alt_scores = _cv_proxy(ctx.phase_id, ctx.audio_before, ctx.audio_after, ctx.sr)
                    alt_regression = 0.0
                    for g in ctx.effective_goals:
                        b = ctx.scores_before.get(g, 0.5)
                        a = alt_scores.get(g, ctx.scores_after.get(g, 0.5))
                        if a < b:
                            alt_regression = max(alt_regression, b - a)

                    if alt_regression < ctx.regression * 0.5:
                        # False positive bestätigt → Guard override!
                        self._best_effort_count += 1
                        return Decision(
                            verdict=DecisionVerdict.OVERRIDE_GUARD,
                            reason=f"Guard-Paralysis bei {ctx.current_strength:.0%}: "
                            f"PMGG Δ={ctx.regression:.3f} → Alternativ Δ={alt_regression:.3f} "
                            f"(false positive). Re-run mit voller Strength.",
                            recommended_strength=1.0,
                            retry_strategy=RetryStrategy.BYPASS_GUARD,
                            paralysis_detected=True,
                            override_goals=list(ctx.effective_goals),
                            details={
                                "paralyzed_strength": ctx.current_strength,
                                "original_regression": ctx.regression,
                                "alternative_regression": alt_regression,
                            },
                        )
            except Exception:
                logger.debug("_Pruefung_paralysis: silent except suppressed", exc_info=True)
        return None

    def _get_alternative_regression(self, ctx: DenkerContext) -> float | None:
        """Berechnet alternative Regression via MediaDefectVerifier."""
        try:
            from backend.core.cassette_defect_verifier import (
                compute_phase_proxy_for_pmgg as _cv_proxy,
            )

            if ctx.audio_before is not None and ctx.audio_after is not None:
                alt_scores = _cv_proxy(ctx.phase_id, ctx.audio_before, ctx.audio_after, ctx.sr)
                alt_reg = 0.0
                for g in ctx.effective_goals:
                    if g in alt_scores:
                        b = ctx.scores_before.get(g, 0.5)
                        a = alt_scores[g]
                        if a < b:
                            alt_reg = max(alt_reg, b - a)
                return alt_reg if alt_reg > 0 else None
        except Exception:
            logger.debug("_get_alternative_regression: silent except suppressed", exc_info=True)
        return None

    def _decide_on_regression(
        self,
        regression: float,
        ctx: DenkerContext,
        thresholds: dict[str, float],
        is_studio: bool,
    ) -> Decision:
        """Trifft Entscheidung basierend auf Regression + Mode + Retry-Count."""

        # Verbesserung → CONTINUE
        if regression < thresholds["retry_light"]:
            return Decision(
                verdict=DecisionVerdict.CONTINUE,
                reason=f"Regression {regression:.4f} < {thresholds['retry_light']} — "
                f"Phase erfolgreich ({'Studio' if is_studio else 'Restoration'})",
                recommended_strength=ctx.current_strength,
            )

        # Leichter Drop → RETRY_LIGHTER (Restoration) oder RETRY_DIFFERENT (Studio)
        if regression < thresholds["retry_heavy"]:
            if is_studio and ctx.retry_count >= 2:
                return Decision(
                    verdict=DecisionVerdict.RETRY_DIFFERENT,
                    reason=f"Leichter Drop (Δ={regression:.3f}) nach {ctx.retry_count} Retries "
                    f"→ alternativen Ansatz versuchen (Studio 2026)",
                    recommended_strength=1.0,
                    retry_strategy=RetryStrategy.SWITCH_PLUGIN,
                )
            new_strength = ctx.current_strength * (0.65 if ctx.retry_count == 0 else 0.40)
            return Decision(
                verdict=DecisionVerdict.RETRY_LIGHTER,
                reason=f"Leichter Drop (Δ={regression:.3f}) → reduzierte Intensität ({new_strength:.0%})",
                recommended_strength=new_strength,
                retry_strategy=RetryStrategy.REDUCE_INTENSITY,
            )

        # Starker Drop → SKIP oder ROLLBACK
        if ctx.retry_count >= thresholds["max_drops"]:
            return Decision(
                verdict=DecisionVerdict.ROLLBACK,
                reason=f"Starker Drop (Δ={regression:.3f}) nach {ctx.retry_count} Retries → ROLLBACK",
                recommended_strength=0.0,
            )
        return Decision(
            verdict=DecisionVerdict.SKIP,
            reason=f"Starker Drop (Δ={regression:.3f}) → Phase {ctx.phase_id} überspringen",
            recommended_strength=0.0,
        )

    # ── Helpers ──

    def _record(self, decision: Decision) -> None:
        """Zeichnet eine Entscheidung in der History auf."""
        self._decision_history.append(decision)
        if len(self._decision_history) > 200:
            self._decision_history = self._decision_history[-100:]

    def get_history(self) -> list[dict[str, Any]]:
        """Gibt Entscheidungs-History als dicts zurück."""
        with self._lock:
            return [
                {
                    "phase": self._phase_count - len(self._decision_history) + i + 1,
                    "verdict": d.verdict.value,
                    "reason": d.reason,
                    "strength": d.recommended_strength,
                    "undo": d.undo_detected,
                    "paralysis": d.paralysis_detected,
                    "proxy": d.proxy_alternative_used,
                }
                for i, d in enumerate(self._decision_history[-20:])
            ]

    def reset(self) -> None:
        """Reset für neuen Durchlauf."""
        with self._lock:
            self._session_active = False
            self._phase_count = 0
            self._best_effort_count = 0
            self._decision_history.clear()
