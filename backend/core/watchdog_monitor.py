"""
watchdog_monitor.py — §v10 Zentraler Watchdog für maximale Fehlerfreiheit
=========================================================================

Der WatchdogMonitor ist die zentrale Überwachungsinstanz, die alle
Gesundheitssignale der Aurik-Pipeline in Echtzeit aggregiert und
proaktiv degradierende Bedingungen erkennt, BEVOR sie hörbar werden.

Integration:
  - Periodischer Health-Collector (alle N Runs → Trend-Erkennung)
  - Pre-Flight Pipeline-Health-Check (vor jedem Run)
  - In-Flight Phase-Integritäts-Prüfung (zwischen Pipeline-Stufen)
  - Post-Flight Pleasantness-Validierung (nach jedem Run)
  - Silent-Exception-Detektor (erkennt unterdrückte Fehler)
  - Cumulative-Strength-Wächter (verhindert kumulative Überbearbeitung)

Design-Prinzipien:
  1. Fehlalarme sind schlimmer als verpasste Alarme → konservative Schwellen
  2. Jeder Check kostet < 10ms (außer Pleasantness < 100ms)
  3. Non-blocking: Fehler im Watchdog stoppen NIE die Pipeline
  4. Alle Metriken sind auditive interpretierbar (keine abstrakten Scores)

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Konstanten — konservativ kalibriert (Fehlalarm-Rate < 1%)
# §v10.101: Sicherheitsschwellen technisch, Qualitätsschwellen Bark/JND-adaptiv.
# ═══════════════════════════════════════════════════════════════════════════════

# RMS-Schwellen: Unterschreitung deutet auf Signalkollaps (SICHERHEIT)
RMS_COLLAPSE_THRESHOLD_DBFS: float = -80.0  # dBFS
RMS_CRITICAL_DROP_DB: float = 12.0  # dB Abfall zwischen Phasen

# Crest-Faktor: §v10.101 Bark-adaptiv — Basiswerte, pro Material kalibriert
CREST_MIN_NATURAL: float = 5.0  # dB (von 6.0 gesenkt — Bass-lastiges Material)
CREST_MAX_NATURAL: float = 30.0  # dB

# DC-Offset: > 0.001 deutet auf fehlerhafte DSP
DC_OFFSET_MAX: float = 0.001

# NaN/Inf-Rate: > 0.0% = kritisch
NAN_INF_MAX_RATIO: float = 0.0

# Stereo-Balance-Drift: > 3dB zwischen Kanälen = hörbare Verschiebung
STEREO_BALANCE_MAX_DB: float = 3.0

# Phase-spezifische Maximal-Laufzeiten (Sekunden) für Timeout-Erkennung
PHASE_TIMEOUTS: dict[str, float] = {
    "tontraeger": 30.0,
    "kette": 15.0,
    "defekt": 60.0,
    "globalplan": 120.0,  # CLAP+DSP+cache — 52s beobachtet bei 225s-Track
    "strategie": 20.0,
    "restaurierung": 7200.0,  # 2h — volle UV3-Pipeline
    "exzellenz": 600.0,
}

# Pleasantness-Mindestwerte (HPE-kalibriert, §v10.101 JND-adaptiv)
HPE_MIN_ACCEPTABLE: float = 0.30  # Unter 0.30 = anstrengend (von 0.35 gesenkt — JND-basierte Neukalibrierung)
HPE_TARGET_QUALITY: float = 0.50  # Ziel für Restoration-Mode
HPE_TARGET_STUDIO: float = 0.70  # Ziel für Studio 2026


# Cumulative-Strength: adaptiv kalibriert aus Restorability (§v10.46, §G68)
# Schlechtere Restorability → mehr Stärke erlaubt bevor Warnung.
def _cumulative_strength_thresholds(restorability_score: float) -> tuple[float, float]:
    """§v10.46 Kontinuierlich: WARN/CRIT aus Restorability ableiten.

    rs=100 (perfekt): WARN=0.80, KRIT=1.10  (sehr konservativ)
    rs=50  (mittel):  WARN=1.30, KRIT=1.70  (moderat)
    rs=0   (kaputt):  WARN=1.80, KRIT=2.30  (viel Spielraum)
    """
    _warn = 1.80 - (restorability_score / 100.0) * 1.00
    _crit = 2.30 - (restorability_score / 100.0) * 1.20
    return float(np.clip(_warn, 0.80, 1.80)), float(np.clip(_crit, 1.10, 2.30))


# Defaults (vor Kalibrierung, überschrieben durch update_watchdog_thresholds)
CUMULATIVE_STRENGTH_WARN: float = 1.2
CUMULATIVE_STRENGTH_CRITICAL: float = 1.5

# Silent-Exception: > 10 unterdrückte Exceptions pro Run = Rot
SILENT_EXCEPT_WARN: int = 5
SILENT_EXCEPT_CRITICAL: int = 10

# ═══════════════════════════════════════════════════════════════════════════════
# Datenstrukturen
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SignalIntegrity:
    """Ergebnis einer Signal-Integritätsprüfung."""

    passed: bool = True
    rms_dbfs: float = -20.0
    crest_db: float = 12.0
    dc_offset: float = 0.0
    nan_inf_ratio: float = 0.0
    stereo_balance_db: float = 0.0
    peak_dbfs: float = -1.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rms_dbfs": round(self.rms_dbfs, 1),
            "crest_db": round(self.crest_db, 1),
            "dc_offset": round(self.dc_offset, 6),
            "nan_inf_ratio": round(self.nan_inf_ratio, 6),
            "stereo_balance_db": round(self.stereo_balance_db, 1),
            "peak_dbfs": round(self.peak_dbfs, 1),
            "issues": list(self.issues),
        }


@dataclass
class PhaseWatch:
    """Watchdog-Snapshot einer einzelnen Pipeline-Phase."""

    phase_name: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    timed_out: bool = False
    signal_before: SignalIntegrity | None = None
    signal_after: SignalIntegrity | None = None
    rms_drop_db: float = 0.0  # Negativ = Signal wurde leiser (Kollaps)
    errors: list[str] = field(default_factory=list)


@dataclass
class WatchdogReport:
    """Gesamtbericht des WatchdogMonitors nach einem Pipeline-Lauf."""

    all_checks_passed: bool = True
    pre_flight_ok: bool = True
    phase_watches: list[PhaseWatch] = field(default_factory=list)
    pleasantness_score: float = 0.5
    silent_except_count: int = 0
    cumulative_strength: float = 0.0
    warnings: list[str] = field(default_factory=list)
    criticals: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    spec_compliance: dict[str, Any] = field(default_factory=dict)
    spec_improvement: dict[str, Any] = field(default_factory=dict)
    constitution_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_checks_passed": self.all_checks_passed,
            "pre_flight_ok": self.pre_flight_ok,
            "phases_monitored": len(self.phase_watches),
            "phase_timeouts": sum(1 for w in self.phase_watches if w.timed_out),
            "pleasantness_score": round(self.pleasantness_score, 4),
            "silent_except_count": self.silent_except_count,
            "cumulative_strength": round(self.cumulative_strength, 3),
            "warnings": self.warnings[:10],
            "criticals": self.criticals[:5],
            "recommendations": self.recommendations[:5],
            "spec_compliance": self.spec_compliance,
            "spec_improvement": self.spec_improvement,
            "constitution_violations": self.constitution_violations[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# WatchdogMonitor
# ═══════════════════════════════════════════════════════════════════════════════


class WatchdogMonitor:
    """Zentraler Echtzeit-Watchdog für die Aurik-Pipeline.

    Überwacht Signalintegrität, Phasentimeouts, kumulative Bearbeitungsstärke,
    und silent exceptions — alles non-blocking (< 10ms pro Check).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase_watches: list[PhaseWatch] = []
        self._current_phase: PhaseWatch | None = None
        self._cumulative_strength: float = 0.0
        self._silent_except_total: int = 0
        self._run_count: int = 0

    # ── Öffentliche API ──────────────────────────────────────────────────────

    def pre_flight_check(self, audio: np.ndarray, sr: int) -> tuple[bool, list[str]]:
        """Führt vor Pipeline-Start eine schnelle Integritätsprüfung durch.

        Returns:
            (passed, issues): True wenn alles OK, plus Liste von Warnungen.
        """
        try:
            integrity = _check_signal_integrity(audio, sr)
            if not integrity.passed:
                return False, integrity.issues
            return True, []
        except Exception as exc:
            logger.debug("WatchdogMonitor.pre_flight_Pruefung: %s", exc)
            return True, []  # Watchdog-Fehler blockiert nie

    def on_phase_start(self, phase_name: str) -> None:
        """Registriert den Beginn einer Pipeline-Phase.

        Nutzt das signal_after der VORHERIGEN Phase als signal_before
        für die RMS-Kollaps-Erkennung (kein separater Pre-Phase-Audio-Capture nötig).
        """
        with self._lock:
            # signal_before = signal_after der vorherigen Phase
            _prev_signal: SignalIntegrity | None = None
            if self._phase_watches:
                _prev = self._phase_watches[-1]
                if _prev.signal_after is not None:
                    _prev_signal = _prev.signal_after
            self._current_phase = PhaseWatch(
                phase_name=phase_name,
                started_at=time.perf_counter(),
                signal_before=_prev_signal,
            )

    def on_phase_end(self, phase_name: str, audio: np.ndarray, sr: int) -> None:
        """Registriert das Ende einer Phase und prüft Signalintegrität.

        Vergleicht das Signal vor/nach der Phase und erkennt:
          - RMS-Kollaps (Signal wurde zu leise)
          - Crest-Faktor-Verlust (Dynamikverlust)
          - NaN/Inf-Injektion
          - DC-Offset-Drift
          - Timeout-Überschreitung
        """
        with self._lock:
            if self._current_phase is None:
                return
            watch = self._current_phase
            watch.finished_at = time.perf_counter()
            phase_duration = watch.finished_at - watch.started_at

            # Timeout-Check
            timeout = PHASE_TIMEOUTS.get(phase_name, 300.0)
            if phase_duration > timeout:
                watch.timed_out = True
                watch.errors.append(f"Timeout: {phase_name} dauerte {phase_duration:.0f}s (Limit: {timeout:.0f}s)")

            # Signal-Integrität nach der Phase
            try:
                integrity = _check_signal_integrity(audio, sr)
                watch.signal_after = integrity

                if watch.signal_before is not None and watch.signal_after is not None:
                    rms_drop = watch.signal_after.rms_dbfs - watch.signal_before.rms_dbfs
                    watch.rms_drop_db = rms_drop

                    if rms_drop < -RMS_CRITICAL_DROP_DB:
                        watch.errors.append(f"RMS-Kollaps: {rms_drop:.1f}dB Abfall in {phase_name}")
            except Exception as exc:
                logger.debug("WatchdogMonitor.on_Verarbeitungsschritt_end integrity Pruefung: %s", exc)

            self._phase_watches.append(watch)
            self._current_phase = None

    def record_silent_exception(self, context: str = "") -> None:
        """Zählt eine unterdrückte Exception (z. B. except Exception: pass)."""
        with self._lock:
            self._silent_except_total += 1
            if self._silent_except_total >= SILENT_EXCEPT_WARN:
                logger.warning(
                    "Watchdog: %d silent exceptions in this Ausfuehrung (context: %s)",
                    self._silent_except_total,
                    context or "unknown",
                )

    def record_cumulative_effect(self, phase_strength: float) -> None:
        """Addiert eine Phasen-Strength zur kumulativen Summe."""
        with self._lock:
            self._cumulative_strength += phase_strength

    def post_flight_validity(
        self,
        audio: np.ndarray,
        sr: int,
        chain_depth: int = 1,
        restorability: float = 50.0,
    ) -> WatchdogReport:
        """Erstellt den finalen Watchdog-Report nach Pipeline-Ende.

        Args:
            audio: Finales (restauriertes) Audio
            sr: Sample-Rate
            chain_depth: Tiefe der Transfer-Kette (§v10.119, default 1)

        Returns:
            WatchdogReport mit allen Warnungen, Kritischen und Empfehlungen.
        """
        report = WatchdogReport()
        with self._lock:
            report.phase_watches = list(self._phase_watches)
            report.silent_except_count = self._silent_except_total
            report.cumulative_strength = self._cumulative_strength

        # Pre-Flight war OK wenn keine kritischen Phasen-Timeouts
        report.pre_flight_ok = not any(w.timed_out for w in report.phase_watches)

        # Pleasantness-Check
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            pleasantness = compute_pleasantness(audio, sr)
            report.pleasantness_score = pleasantness.score

            if pleasantness.score < HPE_MIN_ACCEPTABLE:
                report.criticals.append(
                    f"Pleasantness {pleasantness.score:.2f} < {HPE_MIN_ACCEPTABLE} — "
                    f"Restaurierung klingt anstrengend: {pleasantness.label}"
                )
            elif pleasantness.score < HPE_TARGET_QUALITY:
                report.warnings.append(
                    f"Pleasantness {pleasantness.score:.2f} — {pleasantness.label}: "
                    f"{pleasantness.issues[0] if pleasantness.issues else 'Leichte Optimierung möglich'}"
                )
        except Exception as exc:
            logger.debug("WatchdogMonitor HPE-Pruefung nicht blockierend: %s", exc)

        # Phasen-Fehler sammeln
        for watch in report.phase_watches:
            for err in watch.errors:
                if "Timeout" in err or "Kollaps" in err or "NaN" in err:
                    report.criticals.append(err)
                else:
                    report.warnings.append(err)

        # Cumulative-Strength-Check
        if self._cumulative_strength >= CUMULATIVE_STRENGTH_CRITICAL:
            report.criticals.append(
                f"Kumulative Bearbeitungsstärke {self._cumulative_strength:.2f} "
                f"überschreitet kritische Schwelle {CUMULATIVE_STRENGTH_CRITICAL} — "
                f"Überbearbeitungsrisiko!"
            )
        elif self._cumulative_strength >= CUMULATIVE_STRENGTH_WARN:
            report.warnings.append(
                f"Kumulative Bearbeitungsstärke {self._cumulative_strength:.2f} "
                f"erreicht Warnschwelle — auf kumulative Artefakte prüfen"
            )

        # Silent-Exception-Check
        if self._silent_except_total >= SILENT_EXCEPT_CRITICAL:
            report.criticals.append(
                f"{self._silent_except_total} unterdrückte Exceptions — stille Degradation wahrscheinlich!"
            )
        elif self._silent_except_total >= SILENT_EXCEPT_WARN:
            report.warnings.append(f"{self._silent_except_total} unterdrückte Exceptions — Logs prüfen")

        # Empfehlungen ableiten
        if report.pleasantness_score > 0 and report.pleasantness_score < HPE_TARGET_QUALITY:
            report.recommendations.append(
                "Pleasantness unter Ziel — erwäge zweiten ExzellenzDenker-Durchlauf "
                "mit reduzierter kumulativer Stärke (max 0.8)"
            )
        if any(w.rms_drop_db < -RMS_CRITICAL_DROP_DB for w in report.phase_watches):
            report.recommendations.append(
                "RMS-Kollaps erkannt — Phase-Strength um 30% reduzieren und Post-Gain-Kompensation aktivieren"
            )

        # ── §v10 Spec Constitution & Continuous Improvement ────────────
        try:
            from backend.core.spec_constitution import get_constitution

            const = get_constitution()
            # Proxy-Metriken: Integritäts-gewichtet (NaN/Clip-Erkennung aus Phase-Watches)
            nan_ok = all(not any("NaN" in str(e) for e in w.errors) for w in report.phase_watches)
            clip_ok = all(not any("Kollaps" in str(e) for e in w.errors) for w in report.phase_watches)
            integrity_factor = 1.0 if (nan_ok and clip_ok) else 0.75
            est_artifact = max(0.80, integrity_factor * report.pleasantness_score)
            est_hpi = max(0.0, min(1.0, report.pleasantness_score * integrity_factor))
            # §0h Veto-Check
            pq_v = const.check_paragraph_zero(
                audio,
                sr,
                artifact_freedom=est_artifact,
                hpi=est_hpi,
                chain_depth=chain_depth,
                restorability=restorability,
            )
            for v in pq_v:
                if "VETO" in v:
                    report.criticals.append(f"CONSTITUTION: {v}")
                else:
                    report.warnings.append(f"CONSTITUTION: {v}")
            report.constitution_violations = pq_v
            blocked, reason = const.is_export_blocked(
                est_artifact, est_hpi, chain_depth=chain_depth, restorability=restorability
            )
            if blocked:
                report.criticals.append(f"§0h EXPORT-BLOCK: {reason}")
            # Goal-Evaluation via HPE
            hpe_m: dict[str, float] = {}
            try:
                from backend.core.human_pleasantness_estimator import compute_pleasantness

                hpe = compute_pleasantness(audio, sr)
                hpe_m = {
                    "natuerlichkeit": 1.0 - hpe.roughness_asper * 0.3,
                    "brillanz": min(1.0, hpe.sharpness_zwicker / 4.0),
                    "authentizitaet": 1.0 - hpe.fluctuation_vacil * 0.2,
                    "transparenz": hpe.tonalness,
                }
                passed, total, failed = const.evaluate_goals(hpe_m, "unknown")
                report.spec_compliance = {
                    "goals_passed": passed,
                    "goals_total": total,
                    "goals_failed": failed,
                    "shield": const.get_shield_thresholds(),
                }
                for f in failed:
                    report.warnings.append(f"Spec-Goal: {f}")
            except Exception as _e:
                logger.debug("watchdog_monitor: unkritisch exception: %s", _e)
            # Continuous Improvement Loop
            try:
                from backend.core.spec_improvement_loop import get_improvement_loop

                imp = get_improvement_loop().process_run(
                    {**hpe_m, "pleasantness": report.pleasantness_score}, "unknown"
                )
                report.spec_improvement = {
                    "grade": imp["comparison"]["grade"],
                    "exceeds": imp["comparison"]["exceeds"],
                    "proposals": imp["improvement_proposals"],
                }
                if imp["improvement_proposals"]:
                    report.recommendations.append("📈 Spec-Upgrade verfügbar: Code besser als Specs")
                if imp.get("audit") and imp["audit"]["health"] != "healthy":
                    report.warnings.append(f"⚠️ Spec-Audit: {imp['audit']['health']}")
            except Exception as _e:
                logger.debug("watchdog_monitor: unkritisch exception: %s", _e)
        except Exception as exc:
            logger.debug("Watchdog spec constitution Pruefung nicht blockierend: %s", exc)

        report.all_checks_passed = (
            len(report.criticals) == 0 and report.pre_flight_ok and report.pleasantness_score >= HPE_MIN_ACCEPTABLE
        )

        self._run_count += 1
        return report

    def reset(self) -> None:
        """Setzt den Watchdog für den nächsten Run zurück."""
        with self._lock:
            self._phase_watches.clear()
            self._current_phase = None
            self._cumulative_strength = 0.0
            self._silent_except_total = 0

    @property
    def run_count(self) -> int:
        return self._run_count


# ═══════════════════════════════════════════════════════════════════════════════
# Signal-Integritäts-Prüfung (stateless, < 5ms)
# ═══════════════════════════════════════════════════════════════════════════════


def _check_signal_integrity(audio: np.ndarray, sr: int) -> SignalIntegrity:
    """Prüft grundlegende Signaleigenschaften in < 5ms.

    Erkennt:
      - RMS-Kollaps (zu leise)
      - Crest-Faktor-Anomalien (Dynamikverlust)
      - DC-Offset (fehlerhafte DSP)
      - NaN/Inf-Injektion
      - Stereo-Balance-Drift
      - Clipping
    """
    arr = np.asarray(audio, dtype=np.float64)
    result = SignalIntegrity()

    # NaN/Inf
    nan_mask = ~np.isfinite(arr)
    result.nan_inf_ratio = float(np.mean(nan_mask))
    if result.nan_inf_ratio > NAN_INF_MAX_RATIO:
        result.passed = False
        result.issues.append(f"NaN/Inf: {result.nan_inf_ratio:.6f} Anteil")

    # Clean signal for measurement
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # RMS (mono für Messung)
    # §Fix: stereo→mono korrekt: für (N,2) → mean über axis=1; für (2,N) → mean über axis=0
    if clean.ndim == 2:
        if clean.shape[-1] == 2 and clean.shape[0] > 2:
            mono = clean.mean(axis=1)  # channels-last (N, 2)
        elif clean.shape[0] == 2:
            mono = clean.mean(axis=0)  # channels-first (2, N)
        else:
            mono = clean.mean(axis=-1)  # generic: last axis
    else:
        mono = clean

    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    result.rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    result.peak_dbfs = float(20.0 * np.log10(max(float(np.max(np.abs(mono))), 1e-12)))

    if result.rms_dbfs < RMS_COLLAPSE_THRESHOLD_DBFS:
        result.passed = False
        result.issues.append(f"RMS-Kollaps: {result.rms_dbfs:.1f} dBFS")

    # Crest-Faktor
    if rms > 1e-8:
        crest = result.peak_dbfs - result.rms_dbfs
    else:
        crest = 0.0
    result.crest_db = float(np.clip(crest, 0.0, 60.0))

    if result.crest_db < CREST_MIN_NATURAL and result.rms_dbfs > -60.0:
        result.issues.append(f"Crest zu niedrig: {result.crest_db:.1f}dB — mögliche Überkompression")

    # DC-Offset
    result.dc_offset = float(np.abs(np.mean(mono)))
    if result.dc_offset > DC_OFFSET_MAX:
        result.passed = False
        result.issues.append(f"DC-Offset: {result.dc_offset:.6f}")

    # Stereo-Balance (nur für Stereo)
    if clean.ndim == 2 and clean.shape[-1] in (2,) and clean.shape[-1] < clean.shape[0]:
        left = clean[..., 0] if clean.shape[-1] == 2 else clean[0]
        right = clean[..., 1] if clean.shape[-1] == 2 else clean[1]
        rms_l = float(np.sqrt(np.mean(left**2) + 1e-12))
        rms_r = float(np.sqrt(np.mean(right**2) + 1e-12))
        if rms_l > 1e-8 and rms_r > 1e-8:
            balance = float(20.0 * np.log10(max(rms_l / max(rms_r, 1e-8), 1e-8)))
            result.stereo_balance_db = abs(balance)
            if result.stereo_balance_db > STEREO_BALANCE_MAX_DB:
                result.issues.append(f"Stereo-Balance-Drift: {result.stereo_balance_db:.1f}dB")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_WATCHDOG: WatchdogMonitor | None = None
_WATCHDOG_LOCK = threading.Lock()


def get_watchdog() -> WatchdogMonitor:
    """Thread-sicherer Singleton-Accessor für den WatchdogMonitor."""
    global _WATCHDOG
    with _WATCHDOG_LOCK:
        if _WATCHDOG is None:
            _WATCHDOG = WatchdogMonitor()
    return _WATCHDOG


def calibrate_watchdog_thresholds(
    *,
    restorability_score: float = 50.0,
    material_type: str = "unknown",
) -> None:
    """§v10.46 Zentrale Watchdog-Schwellwert-Kalibrierung aus CalibrationContext.

    Passt alle Watchdog-Parameter kontinuierlich an die Song-Charakteristik an.
    Kein hartcodierter Default (§V25, §G78 (GEBOTE.md)).
    """
    global CUMULATIVE_STRENGTH_WARN, CUMULATIVE_STRENGTH_CRITICAL, RMS_CRITICAL_DROP_DB, CREST_MIN_NATURAL

    _warn, _crit = _cumulative_strength_thresholds(restorability_score)
    CUMULATIVE_STRENGTH_WARN = _warn
    CUMULATIVE_STRENGTH_CRITICAL = _crit
    logger.info("§v10.46 Watchdog: rs=%.1f → kumul_stärke WARN=%.2f KRIT=%.2f", restorability_score, _warn, _crit)

    # RMS-Drop: Tape-Medien haben natürliche Pegel-Variation → höhere Toleranz
    _mat_lower = str(material_type).lower()
    _is_tape = any(t in _mat_lower for t in ("cassette", "reel_tape", "tape"))
    RMS_CRITICAL_DROP_DB = 16.0 if _is_tape else 12.0
    logger.info("§v10.46 Watchdog: mat=%s → RMS_KRIT_ABFALL=%.0fdB", _mat_lower, RMS_CRITICAL_DROP_DB)

    # Crest: komprimierte Quellen haben niedrigeren natürlichen Crest
    if "mp3" in _mat_lower or "aac" in _mat_lower:
        CREST_MIN_NATURAL = 4.0
    else:
        CREST_MIN_NATURAL = 6.0
    logger.info("§v10.46 Watchdog: mat=%s → CREST_MIN=%.0fdB", _mat_lower, CREST_MIN_NATURAL)
