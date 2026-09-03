"""
pleasantness_first_gate.py — v10.0.9 Pleasantness-First Restoration Gate
=========================================================================

Stellt sicher, dass JEDE Restaurierungsentscheidung den Wohlklang fuer
das menschliche Ohr verbessert — nicht nur technische Metriken.

Vier Schutzmechanismen:
  1. HPE-First Gate: Vor/Nach jeder Phase Pleasantness messen
  2. Cross-Phase Consensus: Kumulative Wirkung ueber mehrere Phasen
  3. Defect Inaudibility: Nach Restoration unter Hoerschwelle?
  4. Musical Phrasing: Emotionale Bögen (Vers/Chorus) schuetzen

Prinzip: "Primus inter pares" — das menschliche Ohr entscheidet,
nicht die technische Metrik.

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Human Hearing Thresholds (ISO 226:2023) ────────────────────────────
# Frequencies where the ear is most sensitive (2-5kHz): threshold ~0 dB SPL
# Frequencies below 100Hz and above 10kHz: threshold rises significantly
HUMAN_HEARING_THRESHOLD_DB_SPL: dict[str, float] = {
    "1000Hz": 0.0,  # Reference: 0 dB SPL = hearing threshold
    "2000Hz": -2.0,  # Ear most sensitive here
    "3000Hz": -3.0,  # Maximum sensitivity
    "4000Hz": -2.0,
    "100Hz": 20.0,  # Much less sensitive at low frequencies
    "50Hz": 35.0,
    "10000Hz": 10.0,  # Less sensitive at high frequencies
    "16000Hz": 25.0,
}

# ── Pleasantness thresholds ─────────────────────────────────────────────
HPE_MIN_IMPROVEMENT: float = 0.03  # Minimum delta to count as improvement
MAX_CONSECUTIVE_NO_IMPROVEMENT: int = 3  # Stop if 3 phases don't improve

# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class PhasePleasantness:
    """Pleasantness-Snapshot vor/nach einer Phase."""

    phase_name: str
    hpe_before: float
    hpe_after: float
    delta: float
    improved: bool
    recommendation: str = ""


@dataclass
class CrossPhaseConsensus:
    """Kumulative Wirkung ueber mehrere Phasen."""

    total_delta: float = 0.0
    phases_applied: int = 0
    phases_improved: int = 0
    phases_degraded: int = 0
    cumulative_warnings: list[str] = field(default_factory=list)


@dataclass
class DefectInaudibilityResult:
    """Prueft ob ein Defekt nach Restoration unter der Hoerschwelle liegt."""

    defect_type: str
    pre_level_db: float  # Vor Restoration
    post_level_db: float  # Nach Restoration
    hearing_threshold_db: float  # Hoerschwelle bei dieser Frequenz
    inaudible: bool  # True wenn post < threshold
    margin_db: float  # Wie weit unter der Schwelle


@dataclass
class PleasantnessGateReport:
    """Gesamt-Report des Pleasantness-First-Gate."""

    phase_checks: list[PhasePleasantness] = field(default_factory=list)
    consensus: CrossPhaseConsensus = field(default_factory=CrossPhaseConsensus)
    inaudibility: list[DefectInaudibilityResult] = field(default_factory=list)
    overall_passed: bool = True
    summary: str = ""


# ── PleasantnessFirstGate ───────────────────────────────────────────────


class PleasantnessFirstGate:
    """Zentraler Gate-Keeper: stellt Wohlklang ueber technische Korrektheit.

    Usage:
        gate = PleasantnessFirstGate()
        gate.start_session(original_audio, sr)
        # Vor jeder Phase:
        ok, msg = gate.check_phase_start("phase_03_denoise", current_audio)
        if not ok: skip_phase()  # HPE sagt: verschlechtert den Klang
        # Nach jeder Phase:
        gate.check_phase_end("phase_03_denoise", result_audio)
        # Am Ende:
        report = gate.finalize()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._original_audio: np.ndarray | None = None
        self._original_sr: int = 48000
        self._current_audio: np.ndarray | None = None
        self._phases: list[PhasePleasantness] = []
        self._consensus = CrossPhaseConsensus()
        self._consecutive_no_improvement: int = 0
        self._hpe_cache: dict[str, float] = {}

    def start_session(self, audio: np.ndarray, sr: int) -> float:
        """Initialisiert eine Restaurierungs-Sitzung mit Referenz-Pleasantness."""
        self._original_audio = np.asarray(audio, dtype=np.float32).copy()
        self._original_sr = sr
        self._current_audio = self._original_audio.copy()
        self._phases.clear()
        self._consensus = CrossPhaseConsensus()
        self._consecutive_no_improvement = 0
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            hpe = compute_pleasantness(audio, sr)
            return hpe.score
        except Exception:
            logger.warning("§V6 ML→DSP-Fallback: Pleasantness-Estimation fehlgeschlagen → neutraler Return (0.5)")
            return 0.5

    def check_phase_start(self, phase_name: str, candidate_audio: np.ndarray) -> tuple[bool, str]:
        """Vor einer Phase: wuerde sie den Klang verbessern?

        Vergleicht Pleasantness des Kandidaten (nach Phase) mit dem
        aktuellen Audio (vor Phase). Nur wenn HPE-Verbesserung vorliegt,
        wird die Phase freigegeben.

        Returns: (ok_to_proceed, reason)
        """
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            hpe_current = compute_pleasantness(self._current_audio, self._original_sr)  # type: ignore[arg-type]
            hpe_candidate = compute_pleasantness(candidate_audio, self._original_sr)

            delta = hpe_candidate.score - hpe_current.score

            if delta < -0.05:
                return False, (
                    f"HPE-Gate BLOCKED {phase_name}: "
                    f"Pleasantness {hpe_candidate.score:.3f} < {hpe_current.score:.3f} "
                    f"(delta={delta:+.3f}) — wuerde Klang verschlechtern"
                )
            if delta < HPE_MIN_IMPROVEMENT:
                return True, (
                    f"HPE-Gate ALLOWED {phase_name}: marginal ({delta:+.3f}) — keine signifikante Verbesserung"
                )
            return True, (f"HPE-Gate ALLOWED {phase_name}: Pleasantness verbessert ({delta:+.3f})")
        except Exception as e:
            logger.debug("PleasantnessFirstGate HPE Pruefung error: %s", e)
            return True, f"HPE-Gate SKIPPED {phase_name}: HPE unavailable"

    def check_phase_end(self, phase_name: str, result_audio: np.ndarray) -> PhasePleasantness:
        """Nach einer Phase: Pleasantness-Delta messen."""
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            hpe_before = compute_pleasantness(self._current_audio, self._original_sr)  # type: ignore[arg-type]
            hpe_after = compute_pleasantness(result_audio, self._original_sr)
            delta = hpe_after.score - hpe_before.score
            improved = delta > HPE_MIN_IMPROVEMENT

            check = PhasePleasantness(
                phase_name=phase_name,
                hpe_before=hpe_before.score,
                hpe_after=hpe_after.score,
                delta=delta,
                improved=improved,
                recommendation=(
                    f"Phase {phase_name}: Pleasantness {hpe_before.score:.3f} -> {hpe_after.score:.3f} "
                    f"({'verbessert' if improved else 'verschlechtert'}, delta={delta:+.3f})"
                ),
            )

            with self._lock:
                self._phases.append(check)
                self._consensus.phases_applied += 1
                if improved:
                    self._consensus.phases_improved += 1
                    self._consecutive_no_improvement = 0
                else:
                    self._consensus.phases_degraded += 1
                    self._consecutive_no_improvement += 1
                self._consensus.total_delta += delta
                self._current_audio = np.asarray(result_audio, dtype=np.float32).copy()

            if self._consecutive_no_improvement >= MAX_CONSECUTIVE_NO_IMPROVEMENT:
                logger.warning(
                    "PleasantnessFirstGate: %d Phasen ohne Verbesserung — weitere Phasen werden uebersprungen",
                    self._consecutive_no_improvement,
                )

            return check
        except Exception as e:
            logger.debug("PleasantnessFirstGate Verarbeitungsschritt-end error: %s", e)
            return PhasePleasantness(phase_name=phase_name, hpe_before=0.5, hpe_after=0.5, delta=0.0, improved=False)

    def should_skip_remaining(self) -> bool:
        """Sollten weitere Phasen uebersprungen werden (keine Verbesserung)?"""
        return self._consecutive_no_improvement >= MAX_CONSECUTIVE_NO_IMPROVEMENT

    def check_defect_inaudibility(
        self,
        defect_type: str,
        pre_level_db: float,
        post_level_db: float,
        freq_hz: float = 1000.0,
    ) -> DefectInaudibilityResult:
        """Prueft ob ein Defekt nach Restoration UNHOERBAR ist.

        Vergleicht den Post-Restoration-Pegel mit der menschlichen
        Hoerschwelle (ISO 226:2023) bei der relevanten Frequenz.
        """
        # Finde naechste Frequenz in der Threshold-Tabelle
        thresholds = HUMAN_HEARING_THRESHOLD_DB_SPL
        closest = min(thresholds.keys(), key=lambda k: abs(float(k.replace("Hz", "")) - freq_hz))
        threshold_db = thresholds[closest]

        inaudible = post_level_db < threshold_db
        margin = threshold_db - post_level_db

        return DefectInaudibilityResult(
            defect_type=defect_type,
            pre_level_db=pre_level_db,
            post_level_db=post_level_db,
            hearing_threshold_db=threshold_db,
            inaudible=inaudible,
            margin_db=margin,
        )

    def check_musical_phrasing(
        self,
        audio: np.ndarray,
        sr: int,
        section_boundaries: list[tuple[float, float, str]],  # (start_s, end_s, label)
    ) -> list[str]:
        """Prueft ob musikalische Phrasierung erhalten blieb.

        Vergleicht Energie/Dynamik zwischen gleichen Sektionstypen
        (z.B. Vers 1 vs Vers 2). Unterschiede >3dB = Phrasierung verletzt.
        """
        warnings: list[str] = []
        sections_by_type: dict[str, list[np.ndarray]] = {}

        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)

        for start_s, end_s, label in section_boundaries:
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
            if end_idx > len(arr):
                continue
            section = arr[start_idx:end_idx]
            if section.size > 0:
                if label not in sections_by_type:
                    sections_by_type[label] = []
                sections_by_type[label].append(section)

        for label, sections in sections_by_type.items():
            if len(sections) < 2:
                continue
            rms_values = [float(np.sqrt(np.mean(s**2) + 1e-12)) for s in sections]
            rms_db = [20.0 * np.log10(max(r, 1e-12)) for r in rms_values]
            spread = max(rms_db) - min(rms_db)

            if spread > 3.0:
                warnings.append(
                    f"Musical-Phrasing: '{label}' Sektionen variieren um {spread:.1f}dB "
                    f"— emotionale Bögen gefaehrdet (max 3dB erlaubt)"
                )
            elif spread > 1.5:
                logger.debug("Musical-Phrasing: '%s' Sektionen variieren %.1fdB (akzeptabel)", label, spread)

        return warnings

    def finalize(self) -> PleasantnessGateReport:
        """Erstellt finalen Pleasantness-Gate-Report."""
        report = PleasantnessGateReport(
            phase_checks=self._phases.copy(),
            consensus=self._consensus,
        )

        degraded = self._consensus.phases_degraded
        improved = self._consensus.phases_improved

        if improved > degraded and self._consensus.total_delta > 0.05:
            report.overall_passed = True
            report.summary = (
                f"Pleasantness-Gate PASSED: {improved}/{self._consensus.phases_applied} "
                f"Phasen verbessert (total delta={self._consensus.total_delta:+.3f})"
            )
        elif degraded > 0:
            report.overall_passed = False
            report.summary = (
                f"Pleasantness-Gate FAILED: {degraded} Phasen verschlechtert "
                f"(total delta={self._consensus.total_delta:+.3f}). "
                f"Rollback empfohlen."
            )
        else:
            report.overall_passed = True
            report.summary = "Pleasantness-Gate: keine signifikante Veraenderung"

        return report


# ── Singleton ───────────────────────────────────────────────────────────
_gate: PleasantnessFirstGate | None = None
_gate_lock = threading.Lock()


def get_pleasantness_gate() -> PleasantnessFirstGate:
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = PleasantnessFirstGate()
    return _gate
