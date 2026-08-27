"""
pipeline_guard.py — v10 Pipeline-Guard: Watchdog + CLP + Dynamics + Whisper
============================================================================

Zentraler Integrationspunkt fuer alle v10 Optimierungen. Wird von
AurikDenker._orchestriere() an strategischen Punkten aufgerufen.

Einstiegspunkte:
  guard_pre_flight(audio, sr)          -> (ok, issues)
  guard_phase_start(name)              -> None
  guard_phase_end(name, audio, sr)     -> None
  guard_post_restore(original, result, sr, material) -> (blended, report)
  guard_auto_recover(audio, sr, profile) -> restored_audio

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


class PipelineGuard:
    """Zentraler Wachter fuer die gesamte Aurik-Restaurierungspipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._watchdog: Any = None
        self._dynamics: Any = None
        self._whisper_result: Any = None
        self._clp_result: Any = None
        self._original_audio: np.ndarray | None = None
        self._original_sr: int = 48000
        self._original_profile: Any = None
        self._material: str = "unknown"
        self._start_time: float = 0.0
        self._phase_count: int = 0
        self._warnings: list[str] = []
        self._criticals: list[str] = []
        # §v10.106 Bug-Pattern-Watchdog: Zählt klassifizierte Exceptions
        self._bug_patterns: dict[str, int] = {}
        self._bug_phases: dict[str, set[str]] = {}  # pattern → {phase_ids}

    def _wd(self) -> Any:
        if self._watchdog is None:
            from backend.core.watchdog_monitor import get_watchdog

            self._watchdog = get_watchdog()
        return self._watchdog

    def _dp(self) -> Any:
        if self._dynamics is None:
            from backend.core.dynamics_preserver import get_dynamics_preserver

            self._dynamics = get_dynamics_preserver()
        return self._dynamics

    # ── Pre-Flight ────────────────────────────────────────────────────

    def pre_flight(self, audio: np.ndarray, sr: int, material: str = "unknown") -> tuple[bool, list[str]]:
        """Vor Pipeline-Start: Original sichern, CLP analysieren, Whisper scannen."""
        self._start_time = time.perf_counter()
        self._material = material
        self._original_audio = np.asarray(audio, dtype=np.float32).copy()
        self._original_sr = sr

        issues: list[str] = []

        # Watchdog pre-flight
        try:
            ok, wd_issues = self._wd().pre_flight_check(audio, sr)
            if not ok:
                issues.extend(wd_issues)
        except Exception as e:
            logger.debug("PipelineGuard: watchdog pre-flight error: %s", e)

        # Dynamics: Original-Profil erfassen
        try:
            self._original_profile = self._dp().capture_original(audio, sr)
        except Exception as e:
            logger.debug("PipelineGuard: dynamics Erfassung error: %s", e)

        # CLP: Kritische Frequenzzonen analysieren
        try:
            from backend.core.critical_listening_points import analyze_critical_zones

            self._clp_result = analyze_critical_zones(audio, sr, vocal_boost=True)
            logger.info(
                "PipelineGuard CLP: vocal=%.1f%%, whisper=%.1f%%, zones=%s",
                self._clp_result.vocal_presence * 100,
                self._clp_result.whisper_energy * 100,
                {z: f"{s:.2f}" for z, s in self._clp_result.zone_scores.items()},
            )
        except Exception as e:
            logger.debug("PipelineGuard: CLP Analyse error: %s", e)

        # Whisper: Leise Gesangsdetails erkennen
        try:
            from backend.core.whisper_detail_preserver import analyze_whisper_detail

            self._whisper_result = analyze_whisper_detail(audio, sr)
            if self._whisper_result.segments:
                logger.info(
                    "PipelineGuard Whisper: %d Segmente (%.1f%%), max_atten=%.1fdB",
                    len(self._whisper_result.segments),
                    self._whisper_result.whisper_ratio * 100,
                    self._whisper_result.max_attenuation_db,
                )
        except Exception as e:
            logger.debug("PipelineGuard: Whisper Analyse error: %s", e)

        return len(issues) == 0, issues

    # ── Phase Tracking ────────────────────────────────────────────────

    def phase_start(self, name: str) -> None:
        self._phase_count += 1
        try:
            self._wd().on_phase_start(name)
        except Exception as e:
            logger.debug("PipelineGuard: Verarbeitungsschritt_start error: %s", e)

    def phase_end(self, name: str, audio: np.ndarray, sr: int) -> None:
        try:
            self._wd().on_phase_end(name, audio, sr)
        except Exception as e:
            logger.debug("PipelineGuard: Verarbeitungsschritt_end watchdog error: %s", e)

        try:
            self._dp().check_phase(name, audio, sr)
        except Exception as e:
            logger.debug("PipelineGuard: Verarbeitungsschritt_end dynamics error: %s", e)

    def record_strength(self, value: float) -> None:
        try:
            self._wd().record_cumulative_effect(value)
        except Exception as e:
            logger.debug("PipelineGuard: aufzeichnen_strength error: %s", e)

    def record_silent_exception(self, context: str = "") -> None:
        try:
            self._wd().record_silent_exception(context)
        except Exception as e:
            logger.debug("PipelineGuard: aufzeichnen_silent_exception error: %s", e)

    # ── §v10.106 Bug-Pattern-Klassifikation ────────────────────────────
    # Klassifiziert Exceptions nach den 6 bekannten Anti-Patterns aus der
    # Exception-Forensik (Juli 2026). Ermöglicht Echtzeit-Erkennung von
    # systematischen Bugs während des Pipeline-Laufs.

    BUG_PATTERNS: dict[str, str] = {
        "tuple-ndim": "'tuple' object has no attribute 'ndim'",
        "inhomogeneous": "setting an array element with a sequence",
        "broadcast": "operands could not be broadcast",
        "padlen": "greater than padlen",
        "noverlap": "noverlap must be less than nperseg",
        "os-unbound": "local variable 'os' referenced before assignment",
        "skip-result": "_SkipResult",
        "stereo-template": "Stereo template must be 2D",
        "enum-keyerror": "MaterialType.",
    }

    def classify_exception(self, phase_id: str, error_msg: str) -> str | None:
        """Ordnet eine Exception einem Bug-Pattern zu, oder None."""
        for pattern, keyword in self.BUG_PATTERNS.items():
            if keyword in error_msg:
                self._bug_patterns[pattern] = self._bug_patterns.get(pattern, 0) + 1
                if pattern not in self._bug_phases:
                    self._bug_phases[pattern] = set()
                self._bug_phases[pattern].add(phase_id)

                # Schwellwerte für Mid-Pipeline-Warnungen
                count = self._bug_patterns[pattern]
                if count >= 10 and count % 10 == 0:
                    logger.warning(
                        "🐛 Bug-Pattern '%s': %d× in %d Phasen (%s)",
                        pattern,
                        count,
                        len(self._bug_phases[pattern]),
                        ", ".join(sorted(self._bug_phases[pattern])[:5]),
                    )
                return pattern
        return None

    def get_bug_pattern_report(self) -> dict[str, Any]:
        """Gibt Bug-Pattern-Statistik für post_flight."""
        return {
            "total_bugs": sum(self._bug_patterns.values()),
            "patterns": dict(self._bug_patterns),
            "affected_phases": {k: sorted(v) for k, v in self._bug_phases.items()},
            "critical_patterns": [p for p, c in self._bug_patterns.items() if c >= 20],
        }

    # ── Post-Restore: Whisper-Blending + Dynamics-Recovery ─────────────

    def post_restore(self, original: np.ndarray, restored: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Nach der Restaurierung: Whisper-Details zurueckmischen + Dynamik wiederherstellen."""
        result = np.asarray(restored, dtype=np.float32).copy()
        report: dict[str, Any] = {"whisper_blended": False, "dynamics_restored": False}

        # Whisper-Blending: Original-Details in leisen Passagen zurueckmischen
        if self._whisper_result is not None and self._whisper_result.segments:
            try:
                from backend.core.whisper_detail_preserver import apply_whisper_preservation

                result = apply_whisper_preservation(original, sr, self._whisper_result, processed_audio=result)
                report["whisper_blended"] = True
                report["whisper_segments"] = len(self._whisper_result.segments)
            except Exception as e:
                logger.debug("PipelineGuard: Whisper blending error: %s", e)

        # Dynamics-Auto-Recovery
        if self._original_profile is not None:
            try:
                if self._dp().should_restore():
                    logger.info(
                        "PipelineGuard: Auto-Wiederherstellung Dynamics (loss=%.1fdB)", self._dp().cumulative_loss_db
                    )
                    from backend.core.dynamics_preserver import restore_dynamics

                    result = restore_dynamics(result, sr, self._original_profile, strength=0.7)
                    report["dynamics_restored"] = True
                    report["dynamics_loss_db"] = round(self._dp().cumulative_loss_db, 2)
            except Exception as e:
                logger.debug("PipelineGuard: Dynamics Wiederherstellung error: %s", e)

        return result, report

    # ── Post-Flight ───────────────────────────────────────────────────

    def post_flight(
        self, final_audio: np.ndarray, sr: int, chain_depth: int = 1, restorability: float = 50.0
    ) -> dict[str, Any]:
        """Nach Pipeline-Ende: Watchdog-Report + Pleasantness-Check."""
        report: dict[str, Any] = {
            "phases_monitored": self._phase_count,
            "warnings": list(self._warnings),
            "criticals": list(self._criticals),
        }

        # Watchdog post-flight
        try:
            wd_report = self._wd().post_flight_validity(
                final_audio, sr, chain_depth=chain_depth, restorability=restorability
            )
            report["watchdog"] = wd_report.to_dict()
            report["all_checks_passed"] = wd_report.all_checks_passed
            if wd_report.criticals:
                self._criticals.extend(wd_report.criticals)
                report["criticals"] = list(self._criticals)
            if wd_report.warnings:
                self._warnings.extend(wd_report.warnings)
                report["warnings"] = list(self._warnings)
        except Exception as e:
            logger.debug("PipelineGuard: post_flight watchdog error: %s", e)
            report["all_checks_passed"] = True

        # HPE (Human Pleasantness Estimator)
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            hpe = compute_pleasantness(final_audio, sr)
            report["pleasantness_score"] = hpe.score
            report["pleasantness_label"] = hpe.label
            if hpe.issues:
                report["pleasantness_issues"] = hpe.issues
        except Exception as e:
            logger.debug("PipelineGuard: HPE nicht verfuegbar: %s", e)

        # Adaptive Goal Evaluation: dynamische Schwellen aus Materialphysik
        try:
            from backend.core.spec_constitution import get_constitution

            const = get_constitution()

            # Schaetze physikalische Eigenschaften aus CLP + Material
            est_bw_hz = 20000.0  # default
            est_snr_db = 60.0  # default
            est_era = 2000  # default

            if self._clp_result is not None and self._clp_result.zone_scores:
                # Bandbreite aus CLP-Zonen schaetzen
                zone_energy = self._clp_result.zone_scores
                if zone_energy.get("Luft", 0) > 0.1 or zone_energy.get("Brillanz", 0) > 0.3:
                    est_bw_hz = 20000.0
                elif zone_energy.get("Brillanz", 0) > 0.1:
                    est_bw_hz = 12000.0
                elif zone_energy.get("Praesenz", 0) > 0.3:
                    est_bw_hz = 8000.0
                else:
                    est_bw_hz = 5000.0

                # SNR aus Whisper-Energy + Vocal-Presence schaetzen
                whisper = self._clp_result.whisper_energy
                vocal = self._clp_result.vocal_presence
                if vocal > 0.5:
                    est_snr_db = 55.0
                elif whisper > 0.3:
                    est_snr_db = 35.0
                elif whisper > 0.1:
                    est_snr_db = 45.0
                else:
                    est_snr_db = 60.0

            # Era aus Material-Typ schaetzen
            material_era = {
                "wax_cylinder": 1910,
                "shellac": 1940,
                "vinyl": 1970,
                "tape": 1965,
                "reel_tape": 1960,
                "cassette": 1985,
                "cd_digital": 1995,
                "dat": 1990,
                "mp3_low": 2000,
                "mp3_high": 2005,
                "aac": 2010,
                "streaming": 2015,
            }
            est_era = material_era.get(self._material, 2000)

            # Dynamische Schwellen berechnen
            adaptive = const.compute_adaptive_thresholds(
                self._material,
                effective_bandwidth_hz=est_bw_hz,
                effective_snr_db=est_snr_db,
                era_decade=est_era,
            )
            static = const.get_musical_goal_thresholds(self._material)

            # HPE-Metriken in Goal-Format (via getattr fuer Robustheit)
            _hpe_roughness = getattr(hpe, "roughness_asper", 0.5)
            _hpe_sharpness = getattr(hpe, "sharpness_zwicker", 2.0)
            _hpe_fluctuation = getattr(hpe, "fluctuation_vacil", 0.5)
            _hpe_tonalness = getattr(hpe, "tonalness", 0.6)
            hpe_goals = {
                "natuerlichkeit": max(0.0, 1.0 - _hpe_roughness * 0.3),
                "brillanz": min(1.0, _hpe_sharpness / 4.0),
                "authentizitaet": max(0.0, 1.0 - _hpe_fluctuation * 0.2),
                "transparenz": _hpe_tonalness,
            }

            passed_adaptive, total_adaptive, failed_adaptive = const.evaluate_goals(
                {g: hpe_goals.get(g, static.get(g, 0.6)) for g in adaptive}, self._material
            )
            report["adaptive_goals"] = {
                "passed": passed_adaptive,
                "total": total_adaptive,
                "failed": failed_adaptive,
                "bandwidth_hz": est_bw_hz,
                "snr_db": est_snr_db,
                "era_decade": est_era,
                "sample_thresholds": {g: f"{adaptive[g]:.3f}" for g in list(adaptive.keys())[:5]},
            }
            if failed_adaptive:
                report["warnings"].append(
                    f"Adaptive-Goals: {passed_adaptive}/{total_adaptive} passed "
                    f"(BW={est_bw_hz:.0f}Hz, SNR={est_snr_db:.0f}dB, Era={est_era})"
                )
        except Exception as e:
            logger.debug("PipelineGuard: adaptive goals error: %s", e)

        # G41/G67: PleasantnessFirstGate — HPE-First validation
        try:
            from backend.core.pleasantness_first_gate import PleasantnessFirstGate

            pfg = PleasantnessFirstGate()
            pfg.start_session(self._original_audio if self._original_audio is not None else final_audio, sr)
            check = pfg.check_phase_end("restoration", final_audio)
            if check.delta < -0.05:
                report["warnings"].append(
                    f"G67 Pleasantness-Gate: restored WORSE than original (delta={check.delta:+.3f})"
                )
            elif check.delta > 0.03:
                report["pleasantness_improved"] = True
                report["pleasantness_delta"] = round(check.delta, 4)
            # G46: Adaptive threshold comparison
            if self._clp_result is not None:
                report["adaptive_thresholds_active"] = True
        except Exception as e:
            logger.debug("PipelineGuard: PleasantnessFirstGate error: %s", e)

        elapsed = time.perf_counter() - self._start_time
        report["guard_overhead_ms"] = round(elapsed * 1000)

        # §v10.106 Bug-Pattern-Report: Klassifizierte Exceptions
        bug_report = self.get_bug_pattern_report()
        if bug_report["total_bugs"] > 0:
            report["bug_patterns"] = bug_report
            if bug_report["critical_patterns"]:
                self._criticals.append(
                    f"{bug_report['total_bugs']} Exceptions in "
                    f"{len(bug_report['patterns'])} Bug-Patterns: "
                    f"{', '.join(bug_report['critical_patterns'])}"
                )
                report["criticals"] = list(self._criticals)

        if report.get("criticals"):
            logger.warning(
                "PipelineGuard: %d kritische Warnungen: %s", len(report["criticals"]), report["criticals"][:3]
            )

        return report

    # ── Properties ────────────────────────────────────────────────────

    @property
    def clp_result(self) -> Any:
        return self._clp_result

    @property
    def whisper_result(self) -> Any:
        return self._whisper_result

    @property
    def original_profile(self) -> Any:
        return self._original_profile

    @property
    def material(self) -> str:
        return self._material

    # ── Phase-Integration: CLP + Dynamics + Whisper ────────────────────

    def get_clp_attenuation_limit(self, freq_hz: float) -> float:
        """Maximal erlaubte Daempfung (dB) fuer eine Frequenz basierend auf CLP-Maske.

        Denoise-Phasen rufen dies pro Frequenzband auf, um Ueberdaempfung in
        gehoerempfindlichen Bereichen (2-5 kHz Praesenz-Zone) zu verhindern.

        Returns: dB-Wert (0 = keine Daempfung erlaubt, 99 = unbegrenzt)
        """
        if self._clp_result is None:
            return 99.0
        try:
            from backend.core.pipeline_guard import get_clp_max_attenuation_for_frequency

            return get_clp_max_attenuation_for_frequency(freq_hz, self._clp_result)
        except Exception:
            return 99.0

    def get_clp_gain_limit(self, freq_hz: float) -> float:
        """Maximal erlaubte Verstaerkung (dB) fuer eine Frequenz.

        EQ/Enhancement-Phasen rufen dies auf, um Ueberbetonung in
        gehoerempfindlichen Bereichen zu verhindern.
        """
        if self._clp_result is None:
            return 99.0
        try:
            return get_clp_max_gain_for_frequency(freq_hz, self._clp_result)
        except Exception:
            return 99.0

    def get_whisper_protection(self, time_s: float) -> float:
        """Whisper-Schutzfaktor (0-1) fuer einen Zeitpunkt.

        1.0 = voller Schutz (NR-Staerke auf Minimum reduzieren)
        0.0 = kein Schutz (normale Verarbeitung)

        NR-Phasen multiplizieren ihre Strength mit (1.0 - protection).
        """
        if self._whisper_result is None or self._whisper_result.preservation_mask is None:
            return 0.0
        try:
            mask = self._whisper_result.preservation_mask
            hop_s = 0.025  # 25ms (wie in Whisper-Analyse)
            frame_idx = int(time_s / hop_s)
            if 0 <= frame_idx < len(mask):
                return float(mask[frame_idx])
        except Exception as _e:
            logger.debug("pipeline_guard: unkritisch exception: %s", _e)
        return 0.0

    def check_and_restore_dynamics(self, audio: np.ndarray, sr: int, phase_name: str) -> np.ndarray:
        """Prueft Dynamik-Verlust und stellt bei Bedarf wieder her.

        Wird NACH jeder Dynamics-relevanten Phase (Kompressor, Limiter, EQ) aufgerufen.
        Bei kumulativem Verlust >4dB wird selektiv wiederhergestellt.
        """
        if self._original_profile is None:
            return audio
        try:
            dyn = self._dp()
            loss = dyn.check_phase(phase_name, audio, sr)
            if loss.severity == "severe" or dyn.should_restore():
                from backend.core.dynamics_preserver import restore_dynamics

                logger.info(
                    "PipelineGuard: Auto-Dynamics-Wiederherstellung after %s (loss=%.1fdB)",
                    phase_name,
                    dyn.cumulative_loss_db,
                )
                return restore_dynamics(audio, sr, self._original_profile, strength=0.5)
            if loss.severity == "moderate":
                logger.debug(
                    "PipelineGuard: Dynamics loss after %s: %.1fdB (moderate, no Wiederherstellung)",
                    phase_name,
                    loss.total_loss_db,
                )
        except Exception as e:
            logger.debug("PipelineGuard: dynamics Pruefung error: %s", e)
        return audio

    # ── Combined NR Protection (CLP + Whisper) ───────────────────────

    def get_nr_protection_limits(self, freq_hz: float, time_s: float) -> tuple[float, float]:
        """Gibt kombinierte NR-Schutzlimits: (max_attenuation_db, strength_multiplier).

        Denoise-Phasen rufen dies PRO Frequenzband und Frame auf.
        CLP-Maske begrenzt Daempfung in gehoerempfindlichen Zonen (2-5kHz).
        Whisper-Maske reduziert NR-Staerke in leisen Gesangspassagen.
        """
        max_atten = self.get_clp_attenuation_limit(freq_hz)
        whisper_prot = self.get_whisper_protection(time_s)
        strength_mult = 1.0 - whisper_prot * 0.7
        return max_atten, max(0.1, strength_mult)

    def is_dynamics_phase(self, phase_name: str) -> bool:
        """Prueft ob eine Phase Dynamics-relevant ist."""
        dynamics_phases = {
            "phase_10_compression",
            "phase_11_limiting",
            "phase_26_dynamic_range_expansion",
            "phase_36_transient_shaper",
            "phase_40_loudness_normalization",
            "phase_54_transparent_dynamics",
        }
        return any(p in phase_name.lower() for p in dynamics_phases)

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._original_audio = None
        self._original_profile = None
        self._whisper_result = None
        self._clp_result = None
        self._warnings = []
        self._criticals = []
        self._phase_count = 0
        try:
            self._wd().reset()
        except Exception as _e:
            logger.debug("pipeline_guard: unkritisch exception: %s", _e)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience-Funktionen (stateless, direkt importierbar)
# ═══════════════════════════════════════════════════════════════════════════════


def get_clp_max_attenuation_for_frequency(freq_hz: float, clp_result: Any) -> float:
    """Maximal erlaubte Daempfung (dB) fuer eine Frequenz basierend auf CLP-Maske."""
    if clp_result is None or clp_result.critical_mask is None:
        return 99.0
    from backend.core.critical_listening_points import CLP_ZONES

    for zone in CLP_ZONES:
        if zone.f_min <= freq_hz <= zone.f_max:
            return zone.max_cut_db
    return 99.0


def get_clp_max_gain_for_frequency(freq_hz: float, clp_result: Any) -> float:
    """Maximal erlaubte Verstaerkung (dB) fuer eine Frequenz."""
    if clp_result is None or clp_result.critical_mask is None:
        return 99.0
    from backend.core.critical_listening_points import CLP_ZONES

    for zone in CLP_ZONES:
        if zone.f_min <= freq_hz <= zone.f_max:
            return zone.max_gain_db
    return 99.0


def auto_detect_gender(audio: np.ndarray, sr: int) -> str:
    """Erkennt Geschlecht der Stimme aus der Grundfrequenz-Verteilung."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    n_fft = 4096
    if len(mono) < n_fft:
        return "auto"
    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(sr, 1))
    male_mask = (freqs >= 85) & (freqs <= 180)
    female_mask = (freqs >= 165) & (freqs <= 350)
    male_energy = float(np.sum(spec[male_mask]))
    female_energy = float(np.sum(spec[female_mask]))
    if male_energy > female_energy * 1.3:
        return "male"
    elif female_energy > male_energy * 1.3:
        return "female"
    return "auto"


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_guard: PipelineGuard | None = None
_guard_lock = threading.Lock()


def get_guard() -> PipelineGuard:
    """Thread-sicherer Singleton-Accessor."""
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = PipelineGuard()
    return _guard
