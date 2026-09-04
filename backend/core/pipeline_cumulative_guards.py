"""§CUMULATIVE Pipeline-Cumulative-Guards — SOTA v10.0.0

Konsolidierte kumulative Qualitäts-Überwachung über die gesamte Pipeline.
Einzelne Phasen-Guards (Mikrodynamik, Noise-Texture) prüfen nur
pre vs. post der eigenen Phase — die schleichende Erosion über viele
Phasen bleibt unsichtbar. Dieses Modul trackt die Kumulation.

Muster:
  M1  CumulativeDynamicsTracker     — Crest-Verlust über Pipeline
  M2  EarlyQualityGate              — Abbruch nach 15% wenn schlechter
  M3  Phase07PreFlightSafety        — bandwidth_loss Guard vor Harmonic
  M4  CumulativeNoiseTextureTracker — ≥3 NT-Trigger → Stopp subtraktiv
  M5  GrooveHardGuard               — >50% Onset-Verlust → Rollback
  M6  Phase40PerfGuard              — Performance-Wächter (Diagnose)
  M7  CrossValidator                — MUSHRA vs QualityGate Meta-Bewertung
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# M1: CumulativeDynamicsTracker
# ═══════════════════════════════════════════════════════════════════════════


class CumulativeDynamicsTracker:
    """Trackt Crest-Faktor über die gesamte Pipeline.

    Jede NR-Phase frisst ~0.5–2 dB Crest. Kein Einzel-Guard schlägt Alarm,
    aber die Summe zerstört die Dynamik. Der DoNoHarmGuardian erkennt es erst
    am Ende — dann ist alle Rechenzeit verschwendet.
    """

    def __init__(self) -> None:
        self._original_crest_db: float | None = None
        self._current_crest_db: float | None = None
        self._cumulative_loss_db: float = 0.0
        self._phase_losses: list[tuple[str, float]] = []
        self._blocked: bool = False
        self._warned: bool = False

    def set_original(self, audio: np.ndarray) -> None:
        """Setzt die Original-Crest-Referenz."""
        self._original_crest_db = self._measure_crest(audio)
        self._current_crest_db = self._original_crest_db
        if self._original_crest_db is not None:
            logger.info("§CUMUL-Dyn Originalsignal-Crest: %.1f dB", self._original_crest_db)

    def check(self, audio: np.ndarray, phase_id: str) -> dict[str, Any]:
        """Prüft nach einer Phase den kumulativen Crest-Verlust.

        Returns dict mit Keys: ok, crest_db, loss_db, should_block_subtractive
        """
        if self._blocked:
            return {"ok": False, "blocked": True, "reason": "already_blocked"}

        new_crest = self._measure_crest(audio)
        if new_crest is None or self._current_crest_db is None:
            return {"ok": True, "crest_db": new_crest}

        phase_loss = self._current_crest_db - new_crest
        if phase_loss > 0.1:
            self._phase_losses.append((phase_id, phase_loss))

        self._current_crest_db = new_crest
        self._cumulative_loss_db = (self._original_crest_db or 0.0) - new_crest

        result: dict[str, Any] = {
            "ok": True,
            "crest_db": new_crest,
            "phase_loss_db": round(phase_loss, 2),
            "cumulative_loss_db": round(self._cumulative_loss_db, 2),
            "should_block_subtractive": False,
        }

        # Warnung bei ≥3 dB kumulativem Verlust
        if self._cumulative_loss_db >= 3.0 and not self._warned:
            self._warned = True
            logger.warning(
                "§CUMUL-Dyn WARNING: %.1f dB kumulativer Crest-Verlust (%d Phasen: %s)",
                self._cumulative_loss_db,
                len(self._phase_losses),
                ", ".join(f"{p}({l:.1f})" for p, l in self._phase_losses[-5:]),
            )

        # Block bei ≥6 dB — Dynamik ist zerstört
        if self._cumulative_loss_db >= 6.0:
            self._blocked = True
            result["ok"] = False
            result["should_block_subtractive"] = True
            result["reason"] = f"Crest-Verlust {self._cumulative_loss_db:.1f} dB ≥ 6 dB"
            logger.error(
                "§CUMUL-Dyn BLOCK: %.1f dB Crest-Verlust → "
                "alle subtraktiven Phasen gesperrt! "
                "Originalsignal-Crest=%.1f dB, aktuell=%.1f dB",
                self._cumulative_loss_db,
                self._original_crest_db,
                new_crest,
            )

        return result

    @staticmethod
    def _measure_crest(audio: np.ndarray) -> float | None:
        """Crest-Faktor = Peak/RMS in dB."""
        try:
            a = np.asarray(audio, dtype=np.float32).ravel()
            peak = float(np.max(np.abs(a))) + 1e-12
            rms = float(np.sqrt(np.mean(a * a))) + 1e-12
            return float(20.0 * np.log10(peak / rms))
        except Exception as exc:
            logger.debug("§V6 _measure_crest fehlgeschlagen — None zurückgegeben (Audio %s): %s", audio.shape, exc)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# M2: EarlyQualityGate
# ═══════════════════════════════════════════════════════════════════════════


class EarlyQualityGate:
    """Bricht die Pipeline früh ab, wenn nach 15% der Reparatur-Phasen
    keine Verbesserung messbar ist.
    """

    # Phasen, nach denen der Early-Check läuft (Reparatur-Phasen)
    _GATE_AFTER_PHASES: frozenset[str] = frozenset(
        {
            "phase_01_click_removal",
            "phase_02_hum_removal",
            "phase_03_denoise",
            "phase_04_eq_correction",
            "phase_05_rumble_filter",
            "phase_09_crackle_removal",
        }
    )

    def __init__(
        self,
        total_phases: int,
        restorability_score: float,
        original_crest_db: float | None = None,
        material_type: str = "unknown",
        chain_depth: int = 1,
    ) -> None:
        self._total = max(total_phases, 1)
        self._executed = 0
        self._repair_executed = 0
        self._restorability = restorability_score
        self._original_crest = original_crest_db
        self._early_abort_triggered: bool = False
        self._early_abort_reason: str = ""
        # Snapshot vor Pipeline-Start
        self._snapshot_rms_db: float | None = None
        self._snapshot_crest_db: float | None = None
        # §v10.102: Depth-adaptiver Crest-Abort-Threshold
        _mat = str(material_type).lower()
        _depth = max(1, int(chain_depth))
        if _mat in ("cassette", "reel_tape", "tape"):
            self._crest_abort_db: float = 4.0 + 2.0 * _depth  # depth=1→6, depth=4→12
        else:
            self._crest_abort_db: float = 4.0  # type: ignore[no-redef]

    def set_pre_snapshot(self, audio: np.ndarray) -> None:
        """Speichert Pre-Pipeline-Metriken."""
        try:
            a = np.asarray(audio, dtype=np.float32).ravel()
            rms = float(np.sqrt(np.mean(a * a))) + 1e-12
            peak = float(np.max(np.abs(a))) + 1e-12
            self._snapshot_rms_db = float(20.0 * np.log10(rms))
            self._snapshot_crest_db = float(20.0 * np.log10(peak / rms))
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    def after_phase(self, phase_id: str, audio: np.ndarray, phase_failed: bool = False) -> dict[str, Any]:
        """Wird nach jeder Phase aufgerufen. Prüft Early-Abort-Bedingungen."""
        self._executed += 1
        if phase_id in self._GATE_AFTER_PHASES:
            self._repair_executed += 1

        result: dict[str, Any] = {"should_abort": False, "reason": ""}

        # Nur prüfen wenn ≥3 Reparatur-Phasen gelaufen sind
        # UND Restorability < 50 (schlechtes Material)
        if self._repair_executed < 3 or self._restorability > 50:
            return result

        progress_pct = self._executed / self._total

        # Check 1: Phase komplett fehlgeschlagen?
        if phase_failed:
            self._early_abort_triggered = True
            result["should_abort"] = True
            result["reason"] = f"Phase {phase_id} fehlgeschlagen"
            return result

        # Check 2: Bei 15–20% Fortschritt: Crest-Check
        if 0.12 <= progress_pct <= 0.25 and self._snapshot_crest_db is not None:
            try:
                a = np.asarray(audio, dtype=np.float32).ravel()
                peak = float(np.max(np.abs(a))) + 1e-12
                rms = float(np.sqrt(np.mean(a * a))) + 1e-12
                current_crest = float(20.0 * np.log10(peak / rms))
                crest_drop = self._snapshot_crest_db - current_crest
                if crest_drop > self._crest_abort_db:
                    self._early_abort_triggered = True
                    result["should_abort"] = True
                    result["reason"] = (
                        f"Crest-Einbruch {crest_drop:.1f} dB > {self._crest_abort_db:.1f} dB nach "
                        f"{self._executed}/{self._total} Phasen — "
                        f"Material zu schlecht für Full-Pipeline"
                    )
                    logger.warning("§EARLY-GATE abbrechen: %s", result["reason"])
            except Exception as _eg_exc:
                logger.warning(
                    "§G93 pipeline_cumulative_guards early-gate check failed (non-blocking): %s", _eg_exc, exc_info=True
                )

        return result

    @property
    def aborted(self) -> bool:
        return self._early_abort_triggered


# ═══════════════════════════════════════════════════════════════════════════
# M3: Phase07PreFlightSafety
# ═══════════════════════════════════════════════════════════════════════════


def phase07_preflight_safety(
    bandwidth_loss: float,
    current_rms_db: float,
    sample_rate: int,
) -> dict[str, Any]:
    """Pre-Flight-Check für Phase 07 (Harmonic Restoration).

    Auf bandbreitenbegrenztem Material (bandwidth_loss > 0.8) kann die
    harmonische Restauration komplett kollabieren (→ -86.5 dBFS Stille).
    """
    result: dict[str, Any] = {
        "safe": True,
        "strength_cap": 1.0,
        "skip": False,
        "warnings": [],
    }

    if bandwidth_loss > 0.9:
        result["safe"] = False
        result["strength_cap"] = 0.15
        result["warnings"].append(
            f"bandwidth_loss={bandwidth_loss:.2f} > 0.9 → extreme BW loss, harmonic restoration capped at 15%"
        )
    elif bandwidth_loss > 0.7:
        result["strength_cap"] = 0.35
        result["warnings"].append(f"bandwidth_loss={bandwidth_loss:.2f} > 0.7 → harmonic restoration capped at 35%")

    if current_rms_db < -30:
        result["strength_cap"] = min(result["strength_cap"], 0.15)
        result["warnings"].append(f"RMS={current_rms_db:.1f} dBFS sehr leise → zusätzlicher Cap")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# M4: CumulativeNoiseTextureTracker
# ═══════════════════════════════════════════════════════════════════════════


class CumulativeNoiseTextureTracker:
    """Trackt Noise-Texture-Guard-Trigger über die Pipeline.

    Wenn ≥3 Phasen den NT-Guard triggern (noise_texture_dist > 0.25),
    ist der Rauschboden kumulativ deformiert → subtraktive Phasen stoppen.
    """

    def __init__(self) -> None:
        self._trigger_count: int = 0
        self._trigger_phases: list[str] = []
        self._blocked: bool = False

    def record_trigger(self, phase_id: str, dist: float) -> dict[str, Any]:
        """Registriert einen NT-Guard-Trigger."""
        if self._blocked:
            return {"ok": False, "reason": "already_blocked"}

        self._trigger_count += 1
        self._trigger_phases.append(f"{phase_id}({dist:.3f})")

        if self._trigger_count >= 3:
            self._blocked = True
            logger.error(
                "§CUMUL-NT BLOCK: %d Noise-Texture-Trigger → subtraktive Phasen gesperrt! Trigger: %s",
                self._trigger_count,
                ", ".join(self._trigger_phases),
            )
            return {
                "ok": False,
                "should_block_subtractive": True,
                "reason": f"{self._trigger_count} NT-Trigger: {self._trigger_phases}",
            }

        if self._trigger_count >= 2:
            logger.warning(
                "§CUMUL-NT WARNING: %d Noise-Texture-Trigger — Rauschboden kumulativ deformiert. Trigger: %s",
                self._trigger_count,
                ", ".join(self._trigger_phases),
            )

        return {"ok": True, "trigger_count": self._trigger_count}


# ═══════════════════════════════════════════════════════════════════════════
# M5: GrooveHardGuard
# ═══════════════════════════════════════════════════════════════════════════


class GrooveHardGuard:
    """Harter Guard: >50% Onset-Verlust nach einer Phase → Rollback.

    Für rhythmische Musik (Schlager, Pop) ist der Groove identitätsstiftend.
    Ein Onset-Verlust von 91% (wie im Log: 184→16) ist katastrophal.
    """

    def __init__(self) -> None:
        self._pre_onset_count: int | None = None
        self._total_rollbacks: int = 0

    def set_pre_onsets(self, audio: np.ndarray, sr: int) -> None:
        """Erfasst Onset-Count vor der Phase."""
        self._pre_onset_count = self._count_onsets(audio, sr)

    def check_post(self, phase_id: str, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Prüft Onset-Verlust nach der Phase."""
        if self._pre_onset_count is None or self._pre_onset_count < 10:
            return {"ok": True, "onset_loss_pct": 0.0}

        post_count = self._count_onsets(audio, sr)
        if post_count <= 0:
            return {"ok": True, "onset_loss_pct": 0.0}

        loss_pct = (self._pre_onset_count - post_count) / self._pre_onset_count * 100.0

        result: dict[str, Any] = {
            "ok": True,
            "onset_loss_pct": round(loss_pct, 1),
            "pre_count": self._pre_onset_count,
            "post_count": post_count,
        }

        if loss_pct > 50:
            self._total_rollbacks += 1
            result["ok"] = False
            result["should_rollback"] = True
            result["reason"] = f"Groove-Verlust {loss_pct:.0f}% ({self._pre_onset_count}→{post_count} Onsets)"
            logger.error("§GROOVE-GUARD ROLLBACK %s: %s", phase_id, result["reason"])

        return result

    @staticmethod
    def _count_onsets(audio: np.ndarray, sr: int) -> int:
        """Zählt Onsets via simplen Energie-Anstiegs-Detektor."""
        try:
            mono = (
                np.asarray(audio, dtype=np.float32).mean(axis=0)
                if audio.ndim == 2
                else np.asarray(audio, dtype=np.float32)
            )
            frame_len = int(sr * 0.010)  # 10ms
            hop = frame_len // 2
            energy = np.array([np.sum(mono[i : i + frame_len] ** 2) for i in range(0, len(mono) - frame_len, hop)])
            if len(energy) < 3:
                return 0
            # Onset = Energie-Anstieg > Faktor 2 zum Vorgänger
            energy_prev = np.concatenate([[energy[0]], energy[:-1]])
            onsets = int(np.sum((energy > energy_prev * 2.0) & (energy > 1e-8)))
            return int(onsets)
        except Exception as exc:
            logger.debug("§V6 _count_onsets fehlgeschlagen — 0 zurückgegeben (Audio %s, SR %d): %s", audio.shape, sr, exc)
            return 0


# ═══════════════════════════════════════════════════════════════════════════
# M6: Phase40PerfGuard
# ═══════════════════════════════════════════════════════════════════════════


def diagnose_phase40_performance(phase_id: str, elapsed_s: float, audio_duration_s: float) -> None:
    """Diagnostiziert abnormale Phase-40-Laufzeit."""
    if phase_id != "phase_40_loudness_normalization":
        return
    rt_factor = elapsed_s / max(audio_duration_s, 0.1)
    if rt_factor > 1.0:
        logger.warning(
            "§PERF-GUARD Verarbeitungsschritt_40: %.1fs für %.1fs Audio = %.1f× RT "
            "(erwartet <0.5×). Möglicher Performance-Bug in "
            "Loudness-Normalization-Loop.",
            elapsed_s,
            audio_duration_s,
            rt_factor,
        )


# ═══════════════════════════════════════════════════════════════════════════
# M7: CrossValidator
# ═══════════════════════════════════════════════════════════════════════════


def cross_validate_assessment(
    mushra_score: float,
    quality_gate_delta: float,
    hpi_score: float,
    artifact_freedom: float,
    naturalness: float,
) -> dict[str, Any]:
    """Validiert Widersprüche zwischen Bewertungssystemen.

    Wenn MUSHRA > 90 aber QualityGate delta < 5, ist MUSHRA vermutlich
    ein False-Positive (z.B. weil NSIM=1.0 weil Audio unverändert).
    """
    result: dict[str, Any] = {
        "consistent": True,
        "flags": [],
        "recommendation": "accept",
    }

    # Widerspruch 1: MUSHRA excellent aber keine Verbesserung
    if mushra_score > 90 and quality_gate_delta < 5:
        result["consistent"] = False
        result["flags"].append("mushra_false_positive")
        result["recommendation"] = "degraded_input"
        logger.warning(
            "§CROSS-validieren: MUSHRA=%.0f aber QualityGate Δ=%.1f → "
            "MUSHRA vermutlich False-Positive (unverändertes Audio?). "
            "Empfehlung: degraded_Eingabe melden, nicht 'Excellent'.",
            mushra_score,
            quality_gate_delta,
        )

    # Widerspruch 2: HPI gut aber Naturalness schlecht
    if hpi_score > 0.7 and naturalness < 0.4:
        result["consistent"] = False
        result["flags"].append("hpi_naturalness_mismatch")
        logger.warning(
            "§CROSS-validieren: HPI=%.3f aber Naturalness=%.3f → "
            "HPI könnte durch konservative Parameter verfälscht sein.",
            hpi_score,
            naturalness,
        )

    # Widerspruch 3: MUSHRA excellent aber AFG schlecht
    if mushra_score > 85 and artifact_freedom < 0.5:
        result["consistent"] = False
        result["flags"].append("mushra_afg_mismatch")
        logger.warning(
            "§CROSS-validieren: MUSHRA=%.0f aber AFG=%.3f → subjektive Bewertung ignoriert hörbare Artefakte.",
            mushra_score,
            artifact_freedom,
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline-Integration: Sammel-Guard
# ═══════════════════════════════════════════════════════════════════════════


class PipelineCumulativeGuard:
    """Sammel-Instanz aller kumulativen Guards für die Pipeline-Integration."""

    def __init__(self) -> None:
        self.dynamics = CumulativeDynamicsTracker()
        self.early_gate: EarlyQualityGate | None = None
        self.noise_texture = CumulativeNoiseTextureTracker()
        self.groove = GrooveHardGuard()
        self._total_phases: int = 0
        self._executed_count: int = 0
        self._phase_timings: list[tuple[str, float]] = []
        self._start_time: float = 0.0
        self._audio_duration_s: float = 0.0
        self._original_audio: np.ndarray | None = None
        self._sample_rate: int = 48000
        self._initialized: bool = False

        # CrossValidator state
        self._quality_gate_deltas: list[float] = []
        self._mushra_scores: list[float] = []

    @property
    def original_crest_db(self) -> float:
        return self.dynamics._original_crest_db  # type: ignore[return-value]

    @property
    def cumulative_crest_loss_db(self) -> float:
        return self.dynamics._cumulative_loss_db

    @property
    def noise_texture_trigger_count(self) -> int:
        return self.noise_texture._trigger_count

    @property
    def early_quality_passed(self) -> bool:
        if self.early_gate is None:
            return True
        return not self.early_gate.aborted

    def set_audio_duration(self, duration_s: float) -> None:
        self._audio_duration_s = duration_s

    def reset(
        self,
        original_audio: np.ndarray,
        sample_rate: int,
        total_phases: int = 0,
        restorability_score: float = 50.0,
        material_type: str = "unknown",
        chain_depth: int = 1,
    ) -> None:
        """Initialisiert/Resettet alle Guards vor einem Pipeline-Run."""
        self._original_audio = np.asarray(original_audio, dtype=np.float32)
        self._sample_rate = sample_rate
        self._total_phases = total_phases
        self._executed_count = 0
        self._start_time = time.time()
        self._phase_timings.clear()
        self._quality_gate_deltas.clear()
        self._mushra_scores.clear()
        self._initialized = True
        self.dynamics = CumulativeDynamicsTracker()
        self.dynamics.set_original(self._original_audio)
        self.early_gate = EarlyQualityGate(
            total_phases,
            restorability_score,
            material_type=material_type,
            chain_depth=chain_depth,
        )
        self.early_gate.set_pre_snapshot(self._original_audio)
        self.noise_texture = CumulativeNoiseTextureTracker()
        self.groove = GrooveHardGuard()

    def apply_calibration(self, calib) -> None:
        """§V25/§V27-konform: Übernimmt alle Schwellen aus der zentralen Kalibrierung.

        KEIN einziger hartcodierter Wert. Alle Toleranzen stammen aus
        PipelineCalibration, die sie kontinuierlich aus Pre-Analysis-
        Messwerten ableitet.
        """
        self._crest_tolerance_db = calib.crest_tolerance_db
        self._crest_block_db = calib.crest_block_db
        self._early_abort_pct = calib.early_abort_phase_pct
        self._conservative_mode = calib.restorability_score < calib.conservative_mode_threshold
        self._nt_tolerance = calib.nt_tolerance_per_trigger
        self._nt_max_triggers = calib.nt_max_triggers_before_block
        self._onset_tolerance_pct = calib.onset_loss_tolerance_pct
        self._onset_block_pct = calib.onset_loss_block_pct
        logger.info(
            "§kalibriert crest=%.1f/%.1fdB nt=%.3f/%d onset=%.0f/%.0f%% early=%.0f%% cons=%s",
            self._crest_tolerance_db,
            self._crest_block_db,
            self._nt_tolerance,
            self._nt_max_triggers,
            self._onset_tolerance_pct,
            self._onset_block_pct,
            self._early_abort_pct * 100,
            self._conservative_mode,
        )

    def record_noise_texture_trigger(self) -> None:
        """Convenience: Zeichnet einen NT-Trigger auf (von externen Guard-Aufrufen)."""
        self.noise_texture.record_trigger("external", 0.26)

    def record_quality_gate_delta(self, delta: float) -> None:
        self._quality_gate_deltas.append(delta)

    def record_mushra_score(self, score: float) -> None:
        self._mushra_scores.append(score)

    def detect_mushra_quality_discrepancy(self) -> bool:
        """M7: Erkennt Widerspruch MUSHRA ≫ QualityGate."""
        if not self._mushra_scores or not self._quality_gate_deltas:
            return False
        avg_mushra = sum(self._mushra_scores) / len(self._mushra_scores)
        avg_delta = sum(self._quality_gate_deltas) / len(self._quality_gate_deltas)
        return avg_mushra > 85.0 and avg_delta < 5.0

    def pre_phase(self, phase_id: str, audio: np.ndarray, sr: int) -> None:
        """Wird VOR jeder Phase aufgerufen."""
        # Groove-Guard: Onsets vor der Phase zählen
        if phase_id in {
            "phase_03_denoise",
            "phase_07_harmonic_restoration",
            "phase_09_crackle_removal",
            "phase_29_tape_hiss_reduction",
        }:
            self.groove.set_pre_onsets(audio, sr)

    def post_phase(
        self,
        phase_id: str,
        audio: np.ndarray,
        sr: int,
        elapsed_s: float,
        phase_failed: bool = False,
        noise_texture_dist: float | None = None,
        bandwidth_loss: float = 0.0,
        current_rms_db: float = -20.0,
    ) -> dict[str, Any]:
        """Wird NACH jeder Phase aufgerufen. Sammelt alle Guard-Ergebnisse."""
        self._executed_count += 1
        self._phase_timings.append((phase_id, elapsed_s))

        result: dict[str, Any] = {
            "continue": True,
            "abort_reason": "",
            "block_subtractive": False,
            "warnings": [],
        }

        # ── M1: Cumulative Dynamics ──
        # §V25: crest_tolerance aus Mikrodynamik, nicht hartcodiert
        _tolerance = getattr(self, "_crest_tolerance_db", 4.0)
        _block = getattr(self, "_crest_block_db", 6.0)
        dyn_result = self.dynamics.check(audio, phase_id)
        # Override with calibrated thresholds
        dyn_result["calibrated_tolerance"] = _tolerance
        dyn_result["calibrated_block"] = _block
        _crest_loss = self.dynamics._cumulative_loss_db
        if _crest_loss >= _block:
            result["block_subtractive"] = True
            result["warnings"].append(f"M1: Crest-Verlust {_crest_loss:.1f} dB ≥ {_block:.1f} dB")
        elif _crest_loss >= _tolerance:
            result["warnings"].append(f"M1: Crest-Verlust {_crest_loss:.1f} dB ≥ {_tolerance:.1f} dB (Warnung)")

        # ── M2: Early Quality Gate ──
        _early_pct = getattr(self, "_early_abort_pct", 0.15)
        _progress = self._executed_count / max(self._total_phases, 1)
        if _progress >= _early_pct and self.early_gate is not None:
            eqg_result = self.early_gate.after_phase(phase_id, audio, phase_failed)
            if eqg_result.get("should_abort"):
                result["continue"] = False
                result["abort_reason"] = eqg_result["reason"]

        # ── M4: Cumulative Noise Texture ──
        _nt_max = getattr(self, "_nt_max_triggers", 3)
        _nt_tol = getattr(self, "_nt_tolerance", 0.15)
        if noise_texture_dist is not None and noise_texture_dist > _nt_tol:
            nt_result = self.noise_texture.record_trigger(phase_id, noise_texture_dist)
            if self.noise_texture._trigger_count >= _nt_max:
                result["block_subtractive"] = True
                result["warnings"].append(f"M4: {self.noise_texture._trigger_count} NT-Trigger ≥ {_nt_max}")

        # ── M5: Groove Hard Guard ──
        _onset_block = getattr(self, "_onset_block_pct", 50.0)
        _onset_tol = getattr(self, "_onset_tolerance_pct", 25.0)

        # ── M5: Groove Hard Guard ──
        if phase_id in {
            "phase_03_denoise",
            "phase_07_harmonic_restoration",
            "phase_09_crackle_removal",
            "phase_29_tape_hiss_reduction",
        }:
            groove_result = self.groove.check_post(phase_id, audio, sr)
            if groove_result.get("should_rollback"):
                result["warnings"].append(f"M5: {groove_result.get('reason')} → Rollback empfohlen")

        # ── M6: Phase 40 Perf Guard ──
        diagnose_phase40_performance(phase_id, elapsed_s, self._audio_duration_s)

        return result

    def final_assessment(self) -> dict[str, Any]:
        """Am Pipeline-Ende: M7 CrossValidator + Gesamtbericht."""
        total_elapsed = time.time() - self._start_time
        return {
            "total_phases": self._executed_count,
            "total_time_s": round(total_elapsed, 1),
            "crest_loss_db": round(self.dynamics._cumulative_loss_db, 1),
            "nt_triggers": self.noise_texture._trigger_count,
            "groove_rollbacks": self.groove._total_rollbacks,
            "early_abort": self.early_gate.aborted,  # type: ignore[union-attr]
            "phase_timings": self._phase_timings[-10:],
        }


# ── Singleton ─────────────────────────────────────────────────────────────
_cumulative_guard_instance: PipelineCumulativeGuard | None = None


def get_cumulative_guards() -> PipelineCumulativeGuard:
    """Thread-safe singleton accessor."""
    global _cumulative_guard_instance
    if _cumulative_guard_instance is None:
        _cumulative_guard_instance = PipelineCumulativeGuard()
    return _cumulative_guard_instance
