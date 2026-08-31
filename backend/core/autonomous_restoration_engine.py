"""
core/autonomous_restoration_engine.py
Autonomous Restoration Engine (ARE)
=====================================

Vollautomatische Steuerung des gesamten Aurik-Processings.
Der Nutzer gibt NUR den Modus an: RESTORATION oder STUDIO_2026.
Alle weiteren Entscheidungen trifft die Engine autonom:

  1. Forensische Materialerkennung  (Vinyl, Shellac, Tape, CD, Streaming …)
  2. Defekt-Profiling               (11 Defekttypen mit Severity/Confidence)
  3. Automatische Zielformulierung  (AutoMusicalGoalSetter)
  4. Ketten-Auswahl & -Optimierung  (AdaptiveChainBuilder)
  5. Vorverarbeitung abschließen     (Analyse → REST-Denker für UV3-Full-Pass)
  6. Quality-Gate-Prüfung           (musikalische + technische Schwellwerte)
  7. Rollback bei Verschlechterung  (Overprocessing-Schutz)
  8. Self-Learning-Update           (Ergebnis fließt in zukünftige Sessions)

Einzige öffentliche API:
    engine = AutonomousRestorationEngine(mode=ProcessingMode.RESTORATION)
    result = engine.process(audio, sample_rate)

Author: Aurik Development Team
Version: 1.0.0 "Zero-Intervention Excellence"
Date: 2026-02-17
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from backend.core.auto_musical_goal_setter import AutoMusicalGoalSetter, MusicalGoalProfile
from backend.core.causal_defect_graph import CausalDefectGraph
from backend.core.defect_phase_mapper import DefectPhaseMapper
from backend.core.defect_quality_report import DefectQualityReport, DefectQualityReporter
from backend.core.defect_scanner import DefectAnalysisResult, DefectScanner, DefectType, MaterialType
from backend.core.gap_reconstructor import GapReconstructor
from backend.core.medium_chain_model import PhysicalMediumChainModel
from backend.core.multi_pass_strategy import (
    IntrinsicAudioQualityScorer,
    ProcessingVariant,
    VariantStrategy,
)
from backend.core.processing_modes import ProcessingMode, get_processing_config
from backend.core.provenance_audit import ProvenanceAudit
from backend.core.quality_prediction import QualityAnalyzer, QualityEstimate
from backend.core.self_learning_optimizer import SelfLearningOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class AutonomousRestorationResult:
    """Vollständiges Ergebnis der autonomen Restaurierung."""

    audio: np.ndarray
    """Restauriertes Audio (float32, −1…+1)."""

    sample_rate: int
    """Abtastrate des Ausgangssignals."""

    mode: ProcessingMode
    """Gewählter Nutzer-Modus."""

    material_type: MaterialType
    """Automatisch erkanntes Quellmaterial."""

    defect_profile: DefectAnalysisResult
    """Vollständiges Defekt-Profil des Eingangssignals."""

    goal_profile: MusicalGoalProfile
    """Automatisch formuliertes musikalisches Zielprofil."""

    winning_variant: str
    """Name der Best-Performing-Variante."""

    quality_before: float
    """Geschätzte Qualität vor der Verarbeitung (0–100)."""

    quality_after: float
    """Gemessene Qualität nach der Verarbeitung (0–100)."""

    improvement_db: float
    """SNR-Verbesserung in dB."""

    passes_executed: int
    """Anzahl durchgeführter Processing-Varianten."""

    rollback_triggered: bool
    """True wenn Overprocessing-Schutz eingriff."""

    processing_time_seconds: float
    """Gesamte Verarbeitungszeit."""

    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    """Vollständiges Audit-Log aller Entscheidungen."""

    # --- Weltspitzen-Differenzierer ---
    causal_order: list[str] = field(default_factory=list)
    """Kausal geordnete Reparaturreihenfolge (Root causes zuerst)."""
    causal_explanation: str = ""
    """Prosaerklärung der kausalen Defektabhängigkeiten."""

    # --- ARE-Transparenz (§v10.4): Alle Varianten-Scores für PerceptualQualityCouncil ---
    variant_scores: dict[str, float] = field(default_factory=dict)
    """IAQS-Scores aller evaluierten Varianten {name → score}.
    Ermöglicht dem AurikDenker/PerceptualQualityCouncil die perzeptive
    Nachbewertung — nicht nur den technischen Gewinner zu sehen."""

    chain_corrections: list[str] = field(default_factory=list)
    """Angewendete physikalische Ketteninversions-Korrekturen."""
    chain_spectral_change_db: float = 0.0
    """Mittlere spektrale Änderung durch Ketteninversion (dB)."""
    defect_quality_report: dict[str, Any] | None = None
    """Defektspezifisches Qualitätsprotokoll (per-Defekt SNR, Konfidenz, Kontext)."""
    provenance: dict[str, Any] | None = None
    """Vollständiges Provenanz-Audit (JSONL-exportierbar, archivtauglich)."""
    gaps_found: int = 0
    """Anzahl erkannter Dropout-/Stille-Lücken."""
    gaps_repaired: int = 0
    """Anzahl erfolgreich reparierter Lücken (semantische Rekonstruktion)."""
    gap_total_repaired_ms: float = 0.0
    """Gesamt-Reparaturzeit aller reparierten Lücken (ms)."""


# ---------------------------------------------------------------------------
# Autonomous Restoration Engine
# ---------------------------------------------------------------------------


class AutonomousRestorationEngine:
    """
    Vollautomatische Audio-Restaurierungs-Engine.

    Einziger Nutzer-Parameter: `mode` (RESTORATION | STUDIO_2026).
    Alles andere ist intern und transparent für den Nutzer.
    """

    # Minimale Qualitätsverbesserung, ab der ein Pass als Gewinner gilt
    MIN_IMPROVEMENT_THRESHOLD = 0.5  # Punkte (0–100)

    # Maximale akzeptierte Qualitätsverschlechterung vor Rollback
    ROLLBACK_THRESHOLD = -5.0  # Punkte (0–100) — IAQS-Skala; nur bei echter Verschlechterung

    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        enable_self_learning: bool = True,
    ):
        # Nur RESTORATION und STUDIO_2026 sind gültige Nutzer-Modi
        if mode not in (ProcessingMode.RESTORATION, ProcessingMode.STUDIO_2026):
            raise ValueError(
                f"mode muss ProcessingMode.RESTORATION oder ProcessingMode.STUDIO_2026 sein, erhalten: {mode!r}."
            )

        self.mode = mode
        self.enable_self_learning = enable_self_learning

        # Sub-Systeme
        self._defect_scanner = DefectScanner()
        self._goal_setter = AutoMusicalGoalSetter(mode=mode)
        self._phase_mapper = DefectPhaseMapper()
        self._quality_analyzer = QualityAnalyzer()
        # IAQS für Rollback-Vergleich (stabiler als QualityAnalyzer bei Vorher/Nachher)
        self._iaqs = IntrinsicAudioQualityScorer()
        self._optimizer = SelfLearningOptimizer(mode=mode) if enable_self_learning else None
        # Weltspitzen-Differenzierer
        self._causal_graph = CausalDefectGraph()
        self._chain_model = PhysicalMediumChainModel()
        self._quality_reporter = DefectQualityReporter()
        self._gap_reconstructor = GapReconstructor()
        self._denker_context: dict = {}

        logger.info(
            "AutonomousRestorationEngine initialisiert | Modus: %s | Self-Learning: %s",
            mode.value,
            enable_self_learning,
        )

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def process(
        self, audio: np.ndarray, sample_rate: int, progress_callback=None, **kwargs
    ) -> AutonomousRestorationResult:
        """
        Vollautomatische Restaurierung. Einzige Eingabe: Audio + Abtastrate.

        Args:
            audio: Eingabe-Audio als float32 numpy-Array (mono oder stereo).
            sample_rate: Abtastrate in Hz.
            progress_callback: Optional callable(pct:int, msg:str, elapsed_s:float)
            **kwargs: Denker context (global_plan, chain_info, defekt_hint, mode, material)
                      forwarded to UV3 restore() in full-processing phase.

        Returns:
            AutonomousRestorationResult mit restauriertem Audio und vollständigem Protokoll.
        """
        # Store Denker context for full-processing UV3 call (§Dach: context propagation)
        _ctx_keys = ("global_plan", "chain_info", "defekt_hint", "mode", "material", "cached_defect_result")
        self._denker_context: dict = {k: v for k, v in kwargs.items() if k in _ctx_keys and v is not None}  # type: ignore[no-redef]
        start_time = time.perf_counter()
        audit: list[dict[str, Any]] = []

        def _p(pct: int, msg: str) -> None:
            """Emittiert progress to the caller — pct is 0-100 within ARE's own scale."""
            if progress_callback is not None:
                try:
                    progress_callback(pct, msg, time.perf_counter() - start_time)
                except Exception as _cb_exc:
                    logger.debug("ARE progress_callback fehlgeschlagen: %s", _cb_exc)

        # ----------------------------------------------------------------
        # Phase 0: Validierung & Normalisierung
        # ----------------------------------------------------------------
        _p(2, "Eingabe wird geprüft …")
        audio = self._validate_and_normalize_input(audio)
        audit.append({"phase": "input_validation", "shape": audio.shape, "sr": sample_rate})

        # ----------------------------------------------------------------
        # Phase 1: Qualität des Eingangssignals bestimmen (Baseline)
        # ----------------------------------------------------------------
        logger.debug("[ENGINE] Verarbeitungsschritt 1: QualityAnalyzer …")
        _p(7, "Eingangsqualität wird analysiert …")
        _t1 = time.perf_counter()
        # Analyse-Clips: max. 30 s (kein Audio-Output, nur Score)
        _clip30 = sample_rate * 30
        _audio_clip = audio[:_clip30] if len(audio) > _clip30 else audio
        quality_before_estimate: QualityEstimate = self._quality_analyzer.analyze_quality(_audio_clip, sample_rate)
        logger.debug(
            "[ENGINE] Verarbeitungsschritt 1a fertig (%.1fs): level=%s",
            time.perf_counter() - _t1,
            quality_before_estimate.quality_level.value,
        )
        # IAQS-Score für Rollback-Vergleich (0–1 × 100 = 0–100)
        logger.debug("[ENGINE] Verarbeitungsschritt 1b: IAQS.Wert_as_float …")
        _t1b = time.perf_counter()
        quality_before = self._iaqs.score_as_float(_audio_clip, sample_rate) * 100  # type: ignore[attr-defined]
        logger.debug(
            "[ENGINE] Verarbeitungsschritt 1b fertig (%.1fs): Wert=%.1f", time.perf_counter() - _t1b, quality_before
        )
        audit.append(
            {
                "phase": "baseline_quality",
                "score": quality_before,
                "level": quality_before_estimate.quality_level.value,
            }
        )
        logger.info("Eingangsqualität: %.1f/100 (%s)", quality_before, quality_before_estimate.quality_level.value)

        # ----------------------------------------------------------------
        # Phase 2: Forensische Defekt- und Material-Analyse (vollautomatisch)
        # ----------------------------------------------------------------
        logger.debug("[ENGINE] Verarbeitungsschritt 2: Starte DefectScanner.scan() …")
        _p(18, "Defekte und Material werden erkannt …")
        _cached_defect = self._denker_context.get("cached_defect_result")
        # §9.7.5b: Propagate material hint from Denker context (MediumDetector)
        # to avoid DefectScanner auto-detecting wrong material (e.g. vinyl for tape).
        _material_hint: MaterialType | None = None
        _mat_ctx = self._denker_context.get("material")
        if isinstance(_mat_ctx, MaterialType):
            _material_hint = _mat_ctx
        elif isinstance(_mat_ctx, str) and _mat_ctx:
            try:
                _material_hint = MaterialType(_mat_ctx)
            except (ValueError, KeyError) as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        if _cached_defect is not None:
            defect_result: DefectAnalysisResult = _cached_defect
            logger.info("[ENGINE] Verarbeitungsschritt 2: Verwende gecachten DefectScan (kein Triple-Scan).")
        else:
            defect_result = self._defect_scanner.scan(audio, sample_rate, _material_hint)
        logger.debug("[ENGINE] Verarbeitungsschritt 2 fertig: material=%s", defect_result.material_type.value)
        material = defect_result.material_type
        top_defects = defect_result.get_top_defects(n=5)
        audit.append(
            {
                "phase": "defect_analysis",
                "material": material.value,
                "top_defects": [{"type": d.defect_type.value, "severity": round(d.severity, 3)} for d in top_defects],
            }
        )
        logger.info(
            "Material erkannt: %s | Top-Defekt: %s (Severity %.2f)",
            material.value,
            top_defects[0].defect_type.value if top_defects else "none",
            top_defects[0].severity if top_defects else 0.0,
        )

        # ----------------------------------------------------------------
        # Differenzierer #1: Kausale Defektgraph-Analyse
        # ----------------------------------------------------------------
        logger.debug("[ENGINE] Diff#1: Kausale Defektgraph-Analyse …")
        _p(26, "Kausale Defektreihenfolge wird analysiert …")
        _td1 = time.perf_counter()
        all_defects = defect_result.get_top_defects(n=15)
        causal_ordered = self._causal_graph.resolve_causal_order(all_defects)
        causal_explanation = self._causal_graph.explain(all_defects)
        phantom_defects = self._causal_graph.get_phantom_defects(all_defects)
        logger.debug("[ENGINE] Diff#1 fertig (%.1fs)", time.perf_counter() - _td1)
        audit.append(
            {
                "phase": "causal_defect_graph",
                "causal_order": [d.defect_type.value for d in causal_ordered],
                "phantom_defects": [d.value for d in phantom_defects],
                "explanation_lines": len(causal_explanation.splitlines()),
            }
        )
        logger.info(
            "Kausale Reparaturreihenfolge: %s",
            " → ".join(d.defect_type.value for d in causal_ordered[:5]),
        )

        # ----------------------------------------------------------------
        # Differenzierer #2: Physikalische Ketteninversion
        # ----------------------------------------------------------------
        logger.debug("[ENGINE] Diff#2: Ketteninversion …")
        _p(30, f"Signalkette wird invertiert [{material.value}] …")
        _td2 = time.perf_counter()
        if getattr(defect_result, "is_multi_generation", False):
            chain_result = self._chain_model.invert_chain_sequence(
                audio,
                sample_rate,
                defect_result.transfer_chain_raw,
                all_defects,
            )
        else:
            chain_result = self._chain_model.invert_chain(audio, sample_rate, material, all_defects)
        logger.debug(
            "[ENGINE] Diff#2 fertig (%.1fs): corrections=%d, delta=%.2f dB",
            time.perf_counter() - _td2,
            len(chain_result.corrections_applied),
            chain_result.spectral_change_db,
        )
        audio = chain_result.audio  # Ab hier: kettenentzerrtes Audio
        audit.append(
            {
                "phase": "medium_chain_inversion",
                "material": material.value,
                "corrections": chain_result.corrections_applied,
                "spectral_change_db": chain_result.spectral_change_db,
            }
        )
        logger.info(
            "Ketteninversion [%s]: %d Korrekturen, spektr. Δ=%.2f dB",
            material.value,
            len(chain_result.corrections_applied),
            chain_result.spectral_change_db,
        )

        # ----------------------------------------------------------------
        # Differenzierer #3: Semantische Lückenfüllung (GapReconstructor)
        # ----------------------------------------------------------------
        logger.debug("[ENGINE] Diff#3: GapReconstructor …")
        _p(34, "Lücken und Aussetzer werden erkannt …")
        _td3 = time.perf_counter()
        gap_result = self._gap_reconstructor.reconstruct(
            audio,
            sample_rate,
            material_hint=material.value.split("_")[0].lower(),
        )
        logger.debug(
            "[ENGINE] Diff#3 fertig (%.1fs): gaps_found=%d, repaired=%d",
            time.perf_counter() - _td3,
            gap_result.gaps_found,
            gap_result.gaps_repaired,
        )
        audio = gap_result.audio  # Ab hier: Lücken-bereinigtes Audio
        audit.append(
            {
                "phase": "gap_reconstruction",
                "gaps_found": gap_result.gaps_found,
                "gaps_repaired": gap_result.gaps_repaired,
                "gap_total_repaired_ms": gap_result.total_repaired_ms,
            }
        )
        if gap_result.gaps_found > 0:
            logger.info(
                "Lückenfüllung: %d gefunden, %d repariert, %.1f ms gesamt",
                gap_result.gaps_found,
                gap_result.gaps_repaired,
                gap_result.total_repaired_ms,
            )

        # ----------------------------------------------------------------
        # Phase 3: Automatische musikalische Zielformulierung
        # ----------------------------------------------------------------
        logger.debug("[ENGINE] Verarbeitungsschritt 3: GoalSetter …")
        _p(37, "Musikalische Restaurierungsziele werden berechnet …")
        _t3 = time.perf_counter()
        goal_profile: MusicalGoalProfile = self._goal_setter.compute_goals(
            defect_result=defect_result,
            quality_estimate=quality_before_estimate,
        )
        logger.debug("[ENGINE] Verarbeitungsschritt 3 fertig (%.1fs)", time.perf_counter() - _t3)
        audit.append(
            {
                "phase": "goal_setting",
                "goals": goal_profile.to_dict(),
            }
        )
        logger.info(
            "Musikalische Ziele: SNR_target=%.1f dB, Authenticity=%.2f, Naturalness=%.2f, Clarity=%.2f",
            goal_profile.target_snr_db,
            goal_profile.target_authenticity,
            goal_profile.target_naturalness,
            goal_profile.target_clarity,
        )

        # ----------------------------------------------------------------
        # Phase 4: Processing-Varianten aufbauen (material- & defekt-adaptiv)
        # ----------------------------------------------------------------
        logger.debug("[ENGINE] Verarbeitungsschritt 4: _build_variants …")
        _p(40, "Restaurierungs-Varianten werden geplant …")
        _t4 = time.perf_counter()
        _audio_dur_s: float = len(audio) / max(float(sample_rate), 1.0)
        variants = self._build_variants(defect_result, goal_profile, audio_duration_s=_audio_dur_s)
        logger.debug(
            "[ENGINE] Verarbeitungsschritt 4 fertig (%.1fs): %s Variante(n)", time.perf_counter() - _t4, len(variants)
        )
        audit.append(
            {
                "phase": "variant_selection",
                "variants": [v.name for v in variants],
            }
        )
        logger.info("Varianten: %s", [v.name for v in variants])

        # ----------------------------------------------------------------
        # Phase 5: Multi-Pass – alle Varianten ausführen, beste wählen
        # ----------------------------------------------------------------
        _top_defect_str = ", ".join(d.defect_type.value for d in causal_ordered[:3]) if causal_ordered else "unbekannt"
        _p(42, f"Multi-Pass-Restaurierung: {len(variants)} Varianten · Defekte: {_top_defect_str} …")
        logger.debug(
            "[ENGINE] Verarbeitungsschritt 5: Starte _multi_pass() mit %d Variante(n): %s …",
            len(variants),
            [v.name for v in variants],
        )
        best_audio, best_variant_name, pass_scores, _variant_all_scores = self._multi_pass(  # type: ignore[misc]
            audio=audio,
            sample_rate=sample_rate,
            variants=variants,
            goal_profile=goal_profile,
            progress_callback=progress_callback,
        )
        logger.debug("[ENGINE] Verarbeitungsschritt 5 fertig: winner=%s", best_variant_name)
        _p(87, f"Qualitäts-Gate: Ergebnis '{best_variant_name}' wird geprüft …")
        audit.append(
            {
                "phase": "multi_pass",
                "winner": best_variant_name,
                "scores": {k: round(v, 3) for k, v in pass_scores.items()},
            }
        )
        logger.info("Gewinner-Variante: %s (Wert %.3f)", best_variant_name, pass_scores.get(best_variant_name, 0.0))

        # ----------------------------------------------------------------
        # Phase 6: Quality-Gate & Rollback-Schutz
        # ----------------------------------------------------------------
        # IAQS-Vergleich: quality_before wurde auf einem max. 30-s-Clip gemessen (Phase 1).
        # quality_after MUSS auf dem gleichen kurzen Clip gemessen werden —
        # IAQS-Metriken (SNR, Bark-Balance, Spektralregularität) sind längenabhängig.
        # Ohne diese Symmetrie erzeugt eine 225-s-Datei systematisch Falsch-Rollbacks.
        _qa_clip_samples = sample_rate * 30
        _best_clip = best_audio[:_qa_clip_samples] if len(best_audio) > _qa_clip_samples else best_audio
        self._quality_analyzer.analyze_quality(_best_clip, sample_rate)
        # IAQS für Rollback-Vergleich (beide auf gleichem 30-s-Clip — identische Länge wie quality_before)
        quality_after = self._iaqs.score_as_float(_best_clip, sample_rate) * 100  # type: ignore[attr-defined]
        improvement = quality_after - quality_before
        rollback_triggered = False

        if improvement < self.ROLLBACK_THRESHOLD:
            logger.warning(
                "Qualitätsverschlechterung (Δ=%.2f) — Rollback auf Eingangssignal.",
                improvement,
            )
            best_audio = audio
            quality_after = quality_before
            improvement = 0.0
            rollback_triggered = True
            audit.append({"phase": "quality_gate", "result": "ROLLBACK", "delta": improvement})
        else:
            audit.append(
                {
                    "phase": "quality_gate",
                    "result": "PASS",
                    "before": round(quality_before, 2),
                    "after": round(quality_after, 2),
                    "delta": round(improvement, 2),
                }
            )
            logger.info("Quality-Gate: PASS | Δ=+%.2f (%.1f → %.1f)", improvement, quality_before, quality_after)

        _p(92, f"Restaurierung abgeschlossen: '{best_variant_name}' …")

        # ----------------------------------------------------------------
        # Phase 7: Self-Learning-Update
        # ----------------------------------------------------------------
        if self.enable_self_learning and self._optimizer is not None:
            self._optimizer.record_result(
                material=material,
                variant=best_variant_name,
                defect_profile=defect_result,
                quality_delta=improvement,
            )
            audit.append({"phase": "self_learning", "updated": True})

        # ----------------------------------------------------------------
        # Phase 8: SNR-Differenz berechnen
        # ----------------------------------------------------------------
        improvement_db = self._estimate_snr_improvement(audio, best_audio)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Verarbeitung abgeschlossen in %.1f s | SNR Δ=%.2f dB",
            total_time,
            improvement_db,
        )

        # ----------------------------------------------------------------
        # Differenzierer #7: Provenanz-Vollaudit (archivtauglich)
        # ----------------------------------------------------------------
        provenance = ProvenanceAudit(
            material=material.value,
            mode=self.mode.value,
        )
        for step in audit:
            provenance.record_from_dict(
                step=step.get("phase", "unknown"),
                are_audit_entry=step,
            )
        # Finale Entscheidung dokumentieren
        provenance.record_decision(
            step="restoration_complete",
            rationale=(
                f"Restaurierung abgeschlossen: Material={material.value}, "
                f"Variante={best_variant_name}, "
                f"Δ={improvement_db:+.2f} dB SNR, "
                f"Rollback={'Ja' if rollback_triggered else 'Nein'}"
            ),
            confidence=1.0 - (0.5 if rollback_triggered else 0.0),
            parameters={
                "winning_variant": best_variant_name,
                "quality_delta": round(improvement, 2),
                "snr_improvement_db": round(improvement_db, 2),
                "causal_order": [d.defect_type.value for d in causal_ordered],
                "chain_corrections": chain_result.corrections_applied,
            },
        )

        # ----------------------------------------------------------------
        # Differenzierer #5: Defektspezifisches Qualitätsprotokoll
        # ----------------------------------------------------------------
        dq_report = DefectQualityReport(
            material_type=material.value,
            mode=self.mode.value,
            total_audio_duration_seconds=round(len(best_audio) / sample_rate, 3),
        )
        # Globalen Reparaturbericht für alle erkannten Defekte erstellen
        for defect_score in causal_ordered[:8]:  # Top-8 für Performance
            entry = self._quality_reporter.measure_repair(
                audio_before=audio,
                audio_after=best_audio,
                sample_rate=sample_rate,
                defect_type=defect_score.defect_type,
                severity_before=defect_score.severity,
                confidence=defect_score.confidence,
                phase_id=0,
                repair_method=best_variant_name,
                processing_time_ms=round(total_time * 1000 / max(len(causal_ordered), 1), 2),
            )
            dq_report.add_entry(entry)

        return AutonomousRestorationResult(
            audio=np.clip(best_audio, -1.0, 1.0),  # Clip am Ausgang (§3.1)
            sample_rate=sample_rate,
            mode=self.mode,
            material_type=material,
            defect_profile=defect_result,
            goal_profile=goal_profile,
            winning_variant=best_variant_name,
            variant_scores=_variant_all_scores,
            quality_before=round(quality_before, 2),
            quality_after=round(quality_after, 2),
            improvement_db=float(np.nan_to_num(round(improvement_db, 2), nan=0.0)),
            passes_executed=len(variants),
            rollback_triggered=rollback_triggered,
            processing_time_seconds=round(total_time, 3),
            audit_trail=audit,
            # Weltspitzen-Differenzierer
            causal_order=[d.defect_type.value for d in causal_ordered],
            causal_explanation=causal_explanation,
            chain_corrections=chain_result.corrections_applied,
            chain_spectral_change_db=chain_result.spectral_change_db,
            defect_quality_report=dq_report.to_dict(),
            provenance=provenance.to_dict(),
            gaps_found=gap_result.gaps_found,
            gaps_repaired=gap_result.gaps_repaired,
            gap_total_repaired_ms=gap_result.total_repaired_ms,
        )

    # ------------------------------------------------------------------
    # Interne Methoden (vollautomatisch, kein Nutzereingriff)
    # ------------------------------------------------------------------

    def _validate_and_normalize_input(self, audio: np.ndarray) -> np.ndarray:
        """Stellt sicher, dass audio float32 im Bereich [−1, +1] ist."""
        if not hasattr(audio, "astype"):
            raise TypeError(f"audio muss np.ndarray sein, nicht {type(audio).__name__!r}.")
        audio = audio.astype(np.float32)
        # §DSP-Invariante: np.percentile(99.9) statt np.max — Impuls-Artefakt
        # darf Normalisierung nicht blockieren (copilot-instructions.md VERBOTEN).
        peak = float(np.percentile(np.abs(audio), 99.9))
        if peak > 1.0:
            logger.info("Audio-Normalisierung: Peak %.4f → 1.0", peak)
            audio = audio / peak
        # NaN/Inf-Guard + Clip (§3.1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = np.clip(audio, -1.0, 1.0)
        # Mono-Sicherung: sicherstellen dass max 2 Kanäle
        if audio.ndim > 2:
            audio = audio[:, :2]
        elif audio.ndim == 1:
            pass  # mono OK
        return audio

    def _build_variants(
        self,
        defect_result: DefectAnalysisResult,
        goal_profile: MusicalGoalProfile,  # pylint: disable=unused-argument
        audio_duration_s: float = 60.0,
    ) -> list[ProcessingVariant]:
        """
        Baut vollautomatisch 3–5 Processing-Varianten basierend auf
        Defekt-Profil, Material und Zielprofil.

        Keine Nutzereingabe erforderlich.
        """
        # Alle signifikanten Defekte abrufen (bis zu 20 Typen — 11 Kern + 9 Weltklasse-Erweiterung).
        # Severity-Schwellwert und das Varianten-Limit sind die eigentlichen Filter —
        # nicht ein willkürliches n=11, das neue Typen verdecken würde.
        all_defects = defect_result.get_top_defects(n=20)
        primary_severity = all_defects[0].severity if all_defects else 0.0
        material = defect_result.material_type

        base_mode = self.mode

        # MAX_VARIANTS: Performance-Grenze (jede Variante = 1 restore()-Aufruf).
        # Jede Variante löst einen vollständigen UV3-Pipeline-Durchlauf aus
        # (DefectScan + EraClassify + CausalDefect + Phasen + FeedbackChain + PQS),
        # selbst auf 10s-Excerpts ~85s Overhead. Daher konservativ begrenzen.
        if audio_duration_s < 10.0:
            MAX_VARIANTS = 2  # Kurze Test-Clips / Snippets: minimal
        elif audio_duration_s < 60.0:
            MAX_VARIANTS = 2  # Kurze bis mittlere Passagen
        else:
            MAX_VARIANTS = 3  # Standard-Stücke und lange Aufnahmen

        # Basis-Varianten: immer dabei
        variants: list[ProcessingVariant] = [
            ProcessingVariant.create_conservative(base_mode=base_mode),
            ProcessingVariant.create_balanced(base_mode=base_mode),
        ]

        # Natürlichkeit-Priorisierung: Bei RESTORATION als Alternative (nur wenn Platz)
        if self.mode == ProcessingMode.RESTORATION and len(variants) < MAX_VARIANTS:
            variants.append(ProcessingVariant.create_naturalness_first(base_mode=base_mode))

        # Adaptiv: Bei starken Defekten auch aggressive Variante hinzufügen (nur wenn Platz)
        if primary_severity > 0.4 and len(variants) < MAX_VARIANTS:
            variants.append(ProcessingVariant.create_aggressive(base_mode=base_mode))

        # Spezialisten für ALLE Defekte mit Severity > 0.2 — nach Severity sortiert.
        # Primärer Defekt hat höchste Severity und wird damit automatisch zuerst abgedeckt.
        # Sobald MAX_VARIANTS erreicht, stoppen — Performance vor Vollständigkeit.
        existing_names = {v.name for v in variants}
        for defect_entry in all_defects:
            if len(variants) >= MAX_VARIANTS:
                break
            if defect_entry.severity < 0.2:
                # Defekte ab hier zu schwach für einen eigenen Spezialisten
                break
            specialist = self._build_specialist_variant(defect_entry.defect_type, defect_entry.severity, base_mode)
            if specialist is not None and specialist.name not in existing_names:
                variants.append(specialist)
                existing_names.add(specialist.name)

        # Bei analogem Material: Authentizitäts-optimierte Variante (wenn noch Platz)
        if (
            len(variants) < MAX_VARIANTS
            and material
            in (
                MaterialType.SHELLAC,
                MaterialType.VINYL,
                MaterialType.TAPE,
                MaterialType.REEL_TAPE,
            )
            and self.mode == ProcessingMode.RESTORATION
        ):
            gd = ProcessingVariant.create_gentle_denoise(base_mode=base_mode)
            if gd.name not in existing_names:
                variants.append(gd)
                existing_names.add(gd.name)

        # Self-Learning-Empfehlung integrieren (wenn noch Platz)
        if self.enable_self_learning and self._optimizer is not None and len(variants) < MAX_VARIANTS:
            recommended = self._optimizer.recommend_variant(
                material=defect_result.material_type,
                defect_profile=defect_result,
            )
            if recommended and recommended not in existing_names:
                learned_variant = ProcessingVariant.create_balanced(base_mode=base_mode)
                learned_variant.name = f"learned_{recommended}"
                learned_variant.description = "Self-Learning empfohlene Variante"
                variants.append(learned_variant)

        logger.info("Varianten gebaut (%d/%d): %s", len(variants), MAX_VARIANTS, [v.name for v in variants])
        return variants

    def _build_specialist_variant(
        self,
        defect_type: DefectType | None,
        severity: float,
        base_mode: ProcessingMode,
    ) -> ProcessingVariant | None:
        """Erstellt eine defektspezifische Spezialist-Variante via DefectPhaseMapper."""
        if defect_type is None or severity < 0.2:
            return None

        base_config = get_processing_config(base_mode)

        config, variant_name = self._phase_mapper.build_specialist_config(
            base_config=base_config,
            defect_type=defect_type,
            severity=severity,
            is_restoration_mode=(base_mode == ProcessingMode.RESTORATION),
        )

        # Primary Phase-IDs für Logging (§0a: Restoration-verbotene Phasen gefiltert)
        primary_phases = self._phase_mapper.get_primary_phases(defect_type, mode=base_mode.value)
        description = (
            f"Defekt-Spezialist für {defect_type.value} (Severity={severity:.2f}, Phasen={primary_phases[:2]})"
        )

        return ProcessingVariant(
            name=variant_name,
            strategy=VariantStrategy.BALANCED,
            config=config,  # type: ignore[arg-type]
            description=description,
            weight=1.2,  # leicht bevorzugt beim Scoring
        )

    def _multi_pass(
        self,
        audio: np.ndarray,
        sample_rate: int,
        variants: list[ProcessingVariant],
        goal_profile: MusicalGoalProfile,
        progress_callback=None,
    ) -> tuple[np.ndarray, str, dict[str, float]]:
        """§v10 Multi-Pass: Evaluiert Varianten auf einem 10s-Exzerpt und wählt die beste.

        Seit v10 wird die Multi-Pass-Strategie REAKTIVIERT:
        - Extrahiert 10s Exzerpt aus der Song-Mitte
        - Führt JEDE Variante als Mini-UV3-Pass auf dem Exzerpt aus
        - Scored mit IAQS (Intrinsic Audio Quality Scorer)
        - Gewinner-Variante gibt ihre Parameter an den Full-Pass weiter

        Args:
            audio: Vollständiges Input-Audio
            sample_rate: Sample-Rate
            variants: Liste von ProcessingVariant-Objekten
            goal_profile: MusicalGoalProfile
            progress_callback: Optionaler Progress-Callback

        Returns:
            (audio, winning_variant_name, variant_params_dict)
        """
        if not variants or len(variants) <= 1:
            _name = variants[0].name if variants else "adaptive"
            logger.info("Multi-Pass: ≤1 Variante — direkt %s", _name)
            return audio, _name, {}

        import time as _time

        from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer

        # Extrahiere 10s Exzerpt aus der Song-Mitte (repräsentativ)
        _n = len(audio)
        _excerpt_len = int(10.0 * sample_rate)
        _excerpt_start = max(0, (_n - _excerpt_len) // 2)
        _excerpt_end = min(_n, _excerpt_start + _excerpt_len)

        if audio.ndim == 2:
            _excerpt = audio[:, _excerpt_start:_excerpt_end]
        else:
            _excerpt = audio[_excerpt_start:_excerpt_end]

        if _excerpt.shape[-1] < sample_rate:
            logger.warning(
                "Multi-Pass: Exzerpt zu kurz (%ds) — Ersatzpfad auf Voll-Audio", _excerpt.shape[-1] / sample_rate
            )
            _excerpt = audio

        # Evaluiere jede Variante
        scorer = IntrinsicAudioQualityScorer()
        best_variant = variants[0]
        best_score = -999.0
        results: list[tuple[str, float]] = []

        for _i, _variant in enumerate(variants):
            _t0 = _time.perf_counter()
            try:
                # Baue Phasen-Parameter aus Variante
                _variant.to_phase_params() if hasattr(_variant, "to_phase_params") else {}

                # Rufe UV3 auf dem Exzerpt mit Varianten-Parametern auf
                # (nutzt self._uv3_restore falls verfügbar, sonst Mini-Pass)
                _restored = self._run_mini_restore(
                    _excerpt, sample_rate, _variant, goal_profile, progress_callback=progress_callback
                )

                # Score das Ergebnis
                _score = scorer.score(_excerpt, _restored, sample_rate)
                _elapsed = _time.perf_counter() - _t0
                results.append((_variant.name, _score))
                logger.debug(
                    "Multi-Pass [%d/%d]: %s → Wert=%.3f (%.1fs)",
                    _i + 1,
                    len(variants),
                    _variant.name,
                    _score,
                    _elapsed,
                )

                if _score > best_score:
                    best_score = _score
                    best_variant = _variant
            except Exception as _exc:
                logger.warning("Multi-Pass: Variante %s fehlgeschlagen: %s", _variant.name, _exc)
                results.append((_variant.name, -999.0))

        # Beste Variante auswählen
        _winner_params = {}
        if hasattr(best_variant, "to_phase_params"):
            _winner_params = best_variant.to_phase_params()
        elif hasattr(best_variant, "parameters"):
            _winner_params = dict(best_variant.parameters)

        logger.info(
            "Multi-Pass: %d Varianten evaluiert → Winner: %s (Wert=%.3f, params=%d)",
            len(results),
            best_variant.name,
            best_score,
            len(_winner_params),
        )

        if progress_callback is not None:
            try:
                progress_callback(88, f"Beste Strategie: {best_variant.name}", 0.0)
            except Exception as e:
                logger.warning("autonomous_restoration_engine.py::unbekannter Ersatzpfad: %s", e)

        # Audio unverändert zurückgeben (Parameter werden downstream angewandt)
        _all_scores: dict[str, float] = dict(results)
        return audio, best_variant.name, _winner_params, _all_scores  # type: ignore[return-value]

    def _run_mini_restore(
        self,
        audio: np.ndarray,
        sample_rate: int,
        variant: Any,
        goal_profile: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Führt einen Mini-Restore auf einem Exzerpt mit Varianten-Parametern aus."""
        try:
            from backend.core.unified_restorer_v3 import UnifiedRestorerV3

            _uv3 = UnifiedRestorerV3()
            _mode = getattr(variant, "quality_mode", "balanced")
            _kwargs: dict[str, Any] = {
                "quality_mode": str(_mode),
                "material_type": getattr(self, "_detected_material", "unknown"),
                "progress_callback": kwargs.get("progress_callback"),
            }
            if hasattr(variant, "parameters") and variant.parameters:
                _kwargs["variant_params"] = dict(variant.parameters)
            return _uv3.restore(audio, sample_rate, **_kwargs)  # type: ignore[return-value]
        except Exception:
            # Fallback: einfache Gain-Anpassung als Baseline
            return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))

    @staticmethod
    def _estimate_snr_improvement(original: np.ndarray, processed: np.ndarray) -> float:
        """Schätzt SNR-Verbesserung in dB (Signal = processed, Noise = Differenz)."""
        try:
            # Shape normalisieren: beide auf Mono reduzieren
            orig_mono = np.mean(original, axis=0) if original.ndim == 2 else original
            proc_mono = np.mean(processed, axis=0) if processed.ndim == 2 else processed
            # Ggf. Längen angleichen (restore() kann resampling verursachen)
            min_len = min(len(orig_mono), len(proc_mono))
            if min_len == 0:
                return 0.0
            orig_mono = orig_mono[:min_len]
            proc_mono = proc_mono[:min_len]
            diff = orig_mono - proc_mono
            signal_power = float(np.mean(proc_mono**2))
            noise_power = float(np.mean(diff**2))
            if noise_power < 1e-12 or signal_power < 1e-12:
                return 0.0
            return float(10.0 * np.log10(signal_power / noise_power))
        except Exception as e:
            logger.warning("autonomous_restoration_engine.py::_estimate_snr_improvement Ersatzpfad: %s", e)
            return 0.0
