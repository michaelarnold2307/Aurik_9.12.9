"""
DefektDenker — Domäne: Defekt-Scan + Kausale Ursachenanalyse
=============================================================

Kapselt:
  * `core.defect_scanner.DefectScanner`         (23 DefectTypes, §6.3)
  * `core.causal_defect_reasoner.CausalDefectReasoner` (Bayesianische Inferenz)

Liefert einen strukturierten `DefektBericht` mit erkannten Defekten,
ihrer Ursache und einem Restaurierungsplan.

Singleton-Pattern nach §3.2 (Double-Checked Locking).
NaN/Inf-Schutz nach §3.1.
Type-Annotations nach §3.7.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class DefektBericht:
    """Kombinierter Bericht aus Defekt-Scan und Kausal-Analyse."""

    defect_scores: dict[str, float]
    """Name → Schwere des Defekts ∈ [0, 1] für alle erkannten DefectTypes."""

    primary_cause: str
    """Wahrscheinlichste Ursache (z. B. 'tape_hiss', 'vinyl_crackle')."""

    cause_confidence: float
    """Konfidenz der Ursachen-Aussage ∈ [0, 1]."""

    recommended_phases: list[str]
    """Empfohlene Restaurierungs-Phasen (Phase-Dateinamen ohne .py)."""

    phase_parameters: dict[str, Any]
    """Phasen-spezifische Parameter aus dem RestorationPlan."""

    material: str
    """Materialtyp, auf den die Analyse angewendet wurde."""

    reasoning: str
    """Kausal-Begründung in Deutsch (laienverständlich)."""

    overall_severity: float = 0.0
    """Mittlere Defekt-Schwere über alle Defekte ∈ [0, 1]."""

    def as_dict(self) -> dict[str, Any]:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "primary_cause": self.primary_cause,
            "cause_confidence": self.cause_confidence,
            "material": self.material,
            "overall_severity": self.overall_severity,
            "recommended_phases": self.recommended_phases,
            "defect_scores": self.defect_scores,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# DefektErgebnis — Kompatibilitäts-Alias (Tests importieren diesen Namen)
# ---------------------------------------------------------------------------


@dataclass
class DefektErgebnis:
    """Vereinfachtes Defekt-Ergebnis (Test-Interface / Kompatibilitäts-Alias).

    Felder korrespondieren mit DefektBericht; primär_defect = primary_cause.
    """

    defect_scores: dict[str, float]
    """Name → Schwere des Defekts ∈ [0, 1]."""

    primary_defect: str
    """Wahrscheinlichste Ursache (z. B. 'tape_hiss')."""

    confidence: float
    """Konfidenz der Ursachen-Aussage ∈ [0, 1]."""

    material_context: str
    """Materialtyp, auf den die Analyse angewendet wurde."""

    recommended_phases: list[str]
    """Empfohlene Restaurierungs-Phasen."""

    reasoning: str
    """Kausal-Begründung (laienverständlich)."""

    # §v10.220: Consensus-Manifest der DefectConsensusPipeline (30 Module)
    _consensus_manifest: Any | None = None

    overall_severity: float = 0.0
    """Mittlere Defekt-Schwere ∈ [0, 1]."""

    raw_scan_result: Any | None = None
    """Originaler DefectAnalysisResult aus DefectScanner.scan() — wird als
    cached_defect_result an RestaurierDenker/UV3 weitergegeben, damit UV3
    keinen zweiten internen Scan (mit falscher Material-Erkennung) starten muss.
    Bug-17-Fix: materialkorrektes Scan-Ergebnis fließt durch die gesamte Kette."""

    @property
    def primary_cause(self) -> str:
        """Alias for primary_defect — required by UV3._select_phases() causal_plan protocol."""
        return self.primary_defect

    @classmethod
    def from_bericht(cls, bericht: DefektBericht, raw_scan_result: Any | None = None) -> DefektErgebnis:
        """Konvertiert DefektBericht → DefektErgebnis."""
        return cls(
            defect_scores=bericht.defect_scores,
            primary_defect=bericht.primary_cause,
            confidence=bericht.cause_confidence,
            material_context=bericht.material,
            recommended_phases=bericht.recommended_phases,
            reasoning=bericht.reasoning,
            overall_severity=bericht.overall_severity,
            raw_scan_result=raw_scan_result,
        )


# ---------------------------------------------------------------------------
# DefektDenker
# ---------------------------------------------------------------------------


class DefektDenker:
    """Analysiert Defekte und schlussfolgert auf deren Ursachen.

    Zwei-Schritt-Prozess:
        1. DefectScanner.scan()           → DefectAnalysisResult (23 DefectTypes)
        2. CausalDefectReasoner.reason()  → RestorationPlan (Bayesianisch)

    Verwendung::

        denker = get_defekt_denker()
        bericht = denker.analysiere(audio, sr=44100, material="vinyl")
        logger.debug(bericht.primary_cause, bericht.recommended_phases)
    """

    def __init__(self) -> None:
        self._scanner: Any | None = None
        self._reasoner: Any | None = None
        self._init_lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy-Init
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lädt DefectScanner and CausalDefectReasoner lazily."""
        if self._loaded:
            return
        with self._init_lock:
            if self._loaded:
                return
            # --- DefectScanner (Original, immer vorhanden) ---
            try:
                from backend.core.defect_scanner import DefectScanner

                self._scanner = DefectScanner()
                logger.info("DefektDenker: DefectScanner geladen.")
            except Exception as exc:
                logger.warning("DefektDenker: DefectScanner nicht verfügbar (%s).", exc)

            # --- §v10.220: Defect Consensus Pipeline (30 Module) ---
            try:
                from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

                self._consensus = DefectConsensusPipeline()
                logger.info("DefektDenker: DefectConsensusPipeline geladen (30 Module).")
            except Exception as exc:
                self._consensus = cast(Any, None)
                logger.debug("DefektDenker: Consensus nicht verfügbar (%s).", exc)

            # --- CausalDefectReasoner ---
            try:
                from backend.core.causal_defect_reasoner import CausalDefectReasoner

                self._reasoner = CausalDefectReasoner()
                logger.info("DefektDenker: CausalDefectReasoner geladen.")
            except Exception as exc:
                logger.warning("DefektDenker: CausalDefectReasoner nicht verfügbar (%s).", exc)

            self._loaded = True

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def analysiere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        material: str = "unknown",
        validate_audio: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
        cached_defect_result: Any | None = None,
        file_ext: str = "",
    ) -> DefektErgebnis:
        """Erkennt und klassifiziert Defekte im Audio-Signal.

        Algorithmus:
            1. NaN/Inf-Bereinigung der Eingabe (§3.1)
            2. DefectScanner.scan() → DefectAnalysisResult (23 DefectTypes)
            3. CausalDefectReasoner.reason() → RestorationPlan
            4. Strukturierung zu DefektBericht → Konvertierung in DefektErgebnis

        Args:
            audio:          Eingabe-Audio, float32/64.
            sr:             Sample-Rate in Hz.
            material:       Materialtyp als String (z. B. 'vinyl', 'tape').
            validate_audio: NaN/Inf-Bereinigung durchführen.
            file_ext:       Dateiendung (z. B. '.mp3') für MediumDetector-Prior.

        Returns:
            DefektErgebnis (Kompatibilitäts-Alias) mit Defekten, Ursachen und
            Restaurierungsplan.  Interne Verarbeitung erfolgt via DefektBericht;
            aurik_denker.py unterstützt beide Feldnamen über getattr-Fallback.
        """
        assert sr == 48000, f"DefektDenker.analysiere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        if validate_audio:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        self._ensure_loaded()

        # Step 1: Defect Scan (Cache-First — kein Doppelscan §9.4)
        defect_scores: dict[str, float] = {}
        scan_result: Any | None = None
        if cached_defect_result is not None:
            scan_result = cached_defect_result
            defect_scores = self._extract_scores(scan_result)
            logger.info("DefektDenker: Verwende gecachten DefectScan (kein Doppelscan).")
        elif self._scanner is not None:
            try:
                # §6.4c: Pass resolved material_type to DefectScanner so TRANSPORT_BUMP
                # and other analog-only detectors are skipped for digital materials.
                _mat_type_arg = None
                if material and material != "unknown":
                    try:
                        from backend.core.defect_scanner import MaterialType as _MaterialType

                        # MediumDetector returns "cassette" which has no enum value;
                        # cassette tape maps to MaterialType.TAPE ("tape") in this codebase.
                        _mat_str = "tape" if material == "cassette" else material
                        _mat_type_arg = _MaterialType(_mat_str)
                    except (ValueError, KeyError, ImportError):
                        _mat_type_arg = None  # unknown value → let Scanner auto-detect
                logger.info(
                    "DefektDenker: scan() material=%s → material_type=%s",
                    material,
                    _mat_type_arg,
                )
                scan_result = self._scanner.scan(
                    audio,
                    sample_rate=sr,
                    material_type=_mat_type_arg,
                    progress_callback=progress_callback,
                    file_ext=file_ext,
                )
                defect_scores = self._extract_scores(scan_result)
            except Exception as exc:
                logger.warning("DefektDenker: scan() fehlgeschlagen (%s).", exc)

        # §v10.220: Run Consensus Pipeline for enriched defect manifest.
        # Denker-eigenen Scan als Seed übergeben — der defect_scanner-Detektor
        # scannt dann NICHT erneut („kein Doppelscan“ gilt auch für die
        # Consensus-Pipeline).
        _consensus_manifest = None
        if self._consensus is not None and scan_result is not None:
            try:
                _consensus_manifest = self._consensus.analyze(
                    audio,
                    sr,
                    precomputed_results={"defect_scanner": scan_result},
                )
                logger.info(
                    "DefektDenker: Consensus Manifest erstellt (%d Defekte, %d Konflikte gelöst)",
                    len(_consensus_manifest.defects) if _consensus_manifest else 0,
                    _consensus_manifest.conflicts_resolved if _consensus_manifest else 0,
                )
            except Exception as exc:
                logger.debug("DefektDenker: Consensus fehlgeschlagen (%s)", exc)

        # Step 2: Causal Reasoning
        plan: Any | None = None
        if self._reasoner is not None and scan_result is not None:
            try:
                plan = self._reasoner.reason(
                    defect_scores=getattr(scan_result, "scores", {}),
                    material=material,
                    audio=audio,
                    sr=sr,
                )
            except Exception as exc:
                logger.warning("DefektDenker: reason() fehlgeschlagen (%s).", exc)

        result = DefektErgebnis.from_bericht(
            self._to_bericht(defect_scores, plan, material), raw_scan_result=scan_result
        )
        if _consensus_manifest is not None:
            result._consensus_manifest = _consensus_manifest
        return result

    def scan_nur(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        validate_audio: bool = True,
    ) -> dict[str, float]:
        """Führt nur den Defekte-Scan ohne Kausal-Analyse durch.

        Returns:
            Dict[DefectType-Name → Schwere ∈ [0, 1]].
        """
        if validate_audio:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        self._ensure_loaded()
        if self._scanner is None:
            return {}
        try:
            result = self._scanner.scan(audio, sample_rate=sr)
            return self._extract_scores(result)
        except Exception as exc:
            logger.warning("DefektDenker: scan_nur() fehlgeschlagen (%s).", exc)
            return {}

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_scores(scan_result: Any) -> dict[str, float]:
        """Extrahiert DefectType → severity float dict from DefectAnalysisResult."""
        scores: dict[str, float] = {}
        raw = getattr(scan_result, "scores", {})
        if not isinstance(raw, dict):
            return scores
        for defect_type, defect_score in raw.items():
            name = defect_type.name if hasattr(defect_type, "name") else str(defect_type)
            sev = getattr(defect_score, "severity", defect_score)
            try:
                sev_float = float(sev)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if math.isfinite(sev_float):
                scores[name.lower()] = max(0.0, min(1.0, sev_float))
        return scores

    @staticmethod
    def _overall_severity(scores: dict[str, float]) -> float:
        """Berechnet mean severity, guarded against NaN/empty."""
        vals = [v for v in scores.values() if math.isfinite(v) and v > 0]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _to_bericht(
        self,
        defect_scores: dict[str, float],
        plan: Any | None,
        material: str,
    ) -> DefektBericht:
        """Konvertiert raw scan/plan results to DefektBericht."""
        if plan is not None:
            primary = str(getattr(plan, "primary_cause", "unknown"))
            confidence = float(getattr(plan, "confidence", 0.5))
            phases: list[str] = list(getattr(plan, "recommended_phases", []))
            params: dict[str, Any] = dict(getattr(plan, "phase_parameters", {}))
            reasoning_raw = str(getattr(plan, "reasoning", ""))
        else:
            # Fallback: most severe defect becomes primary cause
            primary = max(defect_scores, key=lambda k: defect_scores[k]) if defect_scores else "unknown"
            confidence = 0.3
            phases = [
                "phase_01_click_removal",
                "phase_02_hum_removal",
                "phase_03_denoise",
                "phase_06_frequency_restoration",
                "phase_09_crackle_removal",
                "phase_23_spectral_repair",
            ]
            params = {}
            reasoning_raw = ""

        if not math.isfinite(confidence):
            confidence = 0.3
        confidence = max(0.0, min(1.0, confidence))

        overall = self._overall_severity(defect_scores)

        reasoning = reasoning_raw or self._begründung(primary, confidence, overall)

        logger.info(
            "DefektDenker: primary=%s confidence=%.2f severity=%.2f phases=%s",
            primary,
            confidence,
            overall,
            phases[:3],
        )

        return DefektBericht(
            defect_scores=defect_scores,
            primary_cause=primary,
            cause_confidence=confidence,
            recommended_phases=phases,
            phase_parameters=params,
            material=material,
            reasoning=reasoning,
            overall_severity=overall,
        )

    @staticmethod
    def _begründung(cause: str, confidence: float, severity: float) -> str:
        """Erzeugt laienverständlichen Begründungstext (Deutsch)."""
        cause_labels: dict[str, str] = {
            "tape_hiss": "Bandrauschen",
            "tape_dropout": "Banddropouts (kurze Stille-Einbrüche)",
            "vinyl_crackle": "Vinyl-Knistern",
            "vinyl_warp": "Verwellte Schallplatte",
            "electrical_hum": "Elektrisches Brummen (50/60 Hz)",
            "head_misalignment": "Fehlausrichtung des Tonkopfs",
            "dc_offset": "Gleichspannungs-Versatz",
            "digital_clip": "Digitale Übersteuerung",
            "soft_saturation": "Analoge Sättigung (normal, wird bewahrt)",
            "compression_artifacts": "Kompressions-Artefakte (z. B. MP3)",
            "print_through": "Tape-Echo (magnetisches Übersprechen)",
        }
        label = cause_labels.get(cause.lower(), cause)
        pct = int(confidence * 100)
        sev_label = "stark" if severity > 0.6 else ("mittel" if severity > 0.3 else "leicht")
        return (
            f"Hauptursache erkannt: {label} ({pct} % Konfidenz). Defekt-Schwere: {sev_label} ({int(severity * 100)} %)."
        )


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: DefektDenker | None = None
_lock = threading.Lock()


def get_defekt_denker() -> DefektDenker:
    """Thread-sicherer Singleton-Accessor für DefektDenker."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = DefektDenker()
    return _instance
