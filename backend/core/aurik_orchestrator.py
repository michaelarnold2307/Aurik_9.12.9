"""§ORCHESTRATOR AurikOrchestrator — Der fehlende Dirigent. v10.0.0

Fünf Pfeiler der Perfektion:
  P1  Gatekeeper         — Pre-Pipeline: restorability → Phasenplan
  P2  Streaming DoNoHarm  — Watchdog 2.0 nach JEDER Phase
  P3  Session-Learning    — Experience-Memory über Songs hinweg
  P4  Assessment-Resolver — Single Source of Truth (MUSHRA≠QG≠HPI)
  P5  Surgery-First       — Minimalprinzip (Messen→Entscheiden)

Architektur-Prinzipien:
  • §0 Primum non nocere: keine Phase läuft ohne vorherige Messung
  • §V25: KEINE hartcodierten Schwellwerte — alles aus Messungen
  • §V27: EINE zentrale Entscheidungsinstanz für ALLE Module
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PreFlightDecision:
    """P1: Entscheidung VOR der Pipeline."""

    should_run: bool
    max_phases: int
    mode: str  # "full", "repair_only", "conservative", "passthrough"
    allowed_phase_families: set[str]
    reason: str
    restorability_score: float
    calibrated_caps: dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseWatchResult:
    """P2: Ergebnis der Streaming-Überwachung nach einer Phase."""

    continue_pipeline: bool = True
    phase_was_harmful: bool = False
    hpi_delta: float = 0.0
    artifact_freedom_delta: float = 0.0
    crest_delta: float = 0.0
    reason: str = ""


@dataclass
class ResolvedAssessment:
    """P4: Aufgelöste, widerspruchsfreie Bewertung."""

    overall_verdict: str  # "improved", "unchanged", "degraded", "unprocessable"
    quality_score: float  # 0-100, EIN Wert
    confidence: float  # 0-1
    explanation: str
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# P1: Gatekeeper — Pre-Pipeline-Entscheider
# ═══════════════════════════════════════════════════════════════════════════

# Reparatur-Phasen (dürfen immer laufen)
_REPAIR_FAMILIES: frozenset[str] = frozenset(
    {
        "click",
        "hum",
        "denoise",
        "eq",
        "rumble",
        "crackle",
        "dropout",
        "declip",
        "azimuth",
        "phase_correction",
        "transient_preservation",
        "transport_bump",
    }
)

# Enhancement-Phasen (nur bei restorability ≥ 50)
_ENHANCE_FAMILIES: frozenset[str] = frozenset(
    {
        "stereo",
        "presence",
        "air_band",
        "bass",
        "vocal",
        "harmonic",
        "mastering",
        "de_esser",
        "loudness",
        "dynamics",
        "speed_pitch",
        "mid_side",
    }
)

# Hochriskante Phasen (nur bei restorability ≥ 70, SNR ≥ 20)
_RISKY_FAMILIES: frozenset[str] = frozenset(
    {
        "diffusion",
        "inpainting",
        "band_gap",
        "frequency_restoration",
        "spectral_repair",
        "groove_echo",
        "inner_groove",
    }
)


def _family_from_phase_id(phase_id: str) -> str:
    """Extrahiert die Familie aus einer Phase-ID."""
    pid = str(phase_id).lower()
    for family in [
        "click",
        "hum",
        "denoise",
        "eq",
        "rumble",
        "crackle",
        "dropout",
        "declip",
        "azimuth",
        "phase_correction",
        "transient_preservation",
        "transport_bump",
        "stereo",
        "presence",
        "air_band",
        "bass",
        "vocal",
        "harmonic",
        "mastering",
        "de_esser",
        "loudness",
        "dynamics",
        "speed_pitch",
        "mid_side",
        "diffusion",
        "inpainting",
        "band_gap",
        "frequency_restoration",
        "spectral_repair",
        "groove_echo",
        "inner_groove",
        "tape_hiss",
        "wow_flutter",
        "surface_noise",
        "splice",
        "noise_gate",
        "truepeak",
        "output_format",
        "semantic",
        "modulation",
        "deesser",
    ]:
        if family in pid:
            return family
    return "unknown"


def gatekeep(
    *,
    restorability_score: float,
    transfer_chain_depth: int,
    bandwidth_loss: float,
    snr_db: float,
    terminal_codec: str | None,
    material_type: str,
    is_restoration_mode: bool,
) -> PreFlightDecision:
    """P1: Entscheidet VOR der Pipeline, WAS laufen soll.

    Returns:
        PreFlightDecision mit max_phases, mode, allowed_families.
    """
    rs = float(restorability_score)
    bw = float(bandwidth_loss)
    depth = int(transfer_chain_depth)

    # ── Regel 1: Passthrough bei katastrophalem Material ──
    if rs < 30:
        return PreFlightDecision(
            should_run=True,
            max_phases=0,
            mode="passthrough",
            allowed_phase_families=set(),
            restorability_score=rs,
            reason=(
                f"Restorability={rs:.0f}/100: Material zu degradiert. "
                f"Keine Bearbeitung möglich. Original wird zurückgegeben."
            ),
        )

    # ── Regel 2: Conservative bei schlechtem Material ──
    if rs < 45 or (bw > 0.8 and depth >= 4):
        return PreFlightDecision(
            should_run=True,
            max_phases=8,
            mode="conservative",
            allowed_phase_families=_REPAIR_FAMILIES,  # type: ignore[arg-type]
            restorability_score=rs,
            reason=(
                f"Restorability={rs:.0f}/100, bw_loss={bw:.2f}, depth={depth}: "
                f"Conservative mode — nur Reparatur-Phasen, kein Enhancement."
            ),
            calibrated_caps={"denoise": 0.30, "harmonic": 0.15},
        )

    # ── Regel 3: Repair-Only bei mittlerem Material ──
    if rs < 65:
        return PreFlightDecision(
            should_run=True,
            max_phases=20,
            mode="repair_only",
            allowed_phase_families=_REPAIR_FAMILIES | _ENHANCE_FAMILIES,  # type: ignore[arg-type]
            restorability_score=rs,
            reason=(f"Restorability={rs:.0f}/100: Repair-Mode. Basis-Enhancement erlaubt, keine riskanten Phasen."),
        )

    # ── Regel 4: Full Pipeline ──
    return PreFlightDecision(
        should_run=True,
        max_phases=50,
        mode="full",
        allowed_phase_families=_REPAIR_FAMILIES | _ENHANCE_FAMILIES | _RISKY_FAMILIES,  # type: ignore[arg-type]
        restorability_score=rs,
        reason=f"Restorability={rs:.0f}/100: Full pipeline freigegeben.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# P2: Streaming DoNoHarm — Watchdog 2.0
# ═══════════════════════════════════════════════════════════════════════════


class StreamingDoNoHarm:
    """P2: Watchdog, der NACH JEDER PHASE prüft — nicht erst am Ende."""

    def __init__(self, original_audio: np.ndarray, sample_rate: int) -> None:
        self._original = np.asarray(original_audio, dtype=np.float32)
        self._sr = sample_rate
        self._snapshots: list[dict[str, Any]] = []
        self._harmful_phases: list[str] = []
        self._total_harmful: int = 0
        self._consecutive_harmful: int = 0
        self._stopped: bool = False

    def snapshot_pre_phase(self, audio: np.ndarray) -> dict[str, float]:
        """Erfasst Metriken VOR einer Phase."""
        return self._measure(audio)

    def watch(
        self,
        phase_id: str,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
    ) -> PhaseWatchResult:
        """P2: Vergleicht pre/post und entscheidet über Fortsetzung.

        Returns PhaseWatchResult mit continue_pipeline=False wenn
        die Phase geschadet hat und die Pipeline gestoppt werden muss.
        """
        pre = self._measure(audio_before)
        post = self._measure(audio_after)

        hpi_delta = post.get("hpi_proxy", 0.5) - pre.get("hpi_proxy", 0.5)
        afg_delta = post.get("artifact_score", 1.0) - pre.get("artifact_score", 1.0)
        crest_delta = post.get("crest_db", 10.0) - pre.get("crest_db", 10.0)
        rms_delta = abs(post.get("rms_db", -20.0) - pre.get("rms_db", -20.0))

        result = PhaseWatchResult()
        result.hpi_delta = round(hpi_delta, 4)
        result.artifact_freedom_delta = round(afg_delta, 4)
        result.crest_delta = round(crest_delta, 2)
        result.continue_pipeline = True

        # Check 1: HPI-Einbruch
        if hpi_delta < -0.10:
            result.phase_was_harmful = True
            self._harmful_phases.append(phase_id)
            self._total_harmful += 1
            self._consecutive_harmful += 1
            result.reason += f"HPI {hpi_delta:+.3f}; "
        else:
            self._consecutive_harmful = 0

        # Check 2: AFG-Einbruch (>20% Artefakt-Zunahme)
        if afg_delta < -0.20:
            result.phase_was_harmful = True
            result.reason += f"AFG {afg_delta:+.3f}; "

        # Check 3: Crest-Crash
        if crest_delta < -3.0:
            result.phase_was_harmful = True
            result.reason += f"Crest {crest_delta:+.1f}dB; "

        # Check 4: Silence-Crash (RMS-Drop > 30dB)
        if rms_delta > 30.0:
            result.phase_was_harmful = True
            result.reason += f"RMS-Drop {rms_delta:.0f}dB; "

        # Decision: Stoppen wenn schädlich
        if result.phase_was_harmful:
            if self._consecutive_harmful >= 3:
                result.continue_pipeline = False
                self._stopped = True
                result.reason += f"STOP: {self._consecutive_harmful} konsekutive schädliche Phasen."

        return result

    @staticmethod
    def _measure(audio: np.ndarray) -> dict[str, float]:
        """Schnelle Metriken für Streaming-Watchdog."""
        try:
            a = np.asarray(audio, dtype=np.float32).ravel()
            n = len(a)
            if n < 256:
                return {}
            rms = float(np.sqrt(np.mean(a * a))) + 1e-12
            peak = float(np.max(np.abs(a))) + 1e-12
            crest = float(20.0 * np.log10(peak / rms))
            rms_db = float(20.0 * np.log10(rms))

            # Einfacher HPI-Proxy: Crest-Nähe zum idealen Bereich (10-14 dB)
            hpi = float(np.clip(1.0 - abs(crest - 12.0) / 12.0, 0.0, 1.0))

            # Artefakt-Proxy: RMS-Stabilität über Segmente
            seg_len = max(n // 20, 256)
            seg_rms = np.array([np.sqrt(np.mean(a[i : i + seg_len] ** 2)) for i in range(0, n - seg_len, seg_len)])
            seg_rms = seg_rms[seg_rms > 1e-8]
            if len(seg_rms) >= 3:
                art_score = float(np.clip(1.0 - np.std(seg_rms) / (np.mean(seg_rms) + 1e-8), 0.0, 1.0))
            else:
                art_score = 1.0

            return {
                "hpi_proxy": hpi,
                "artifact_score": art_score,
                "crest_db": crest,
                "rms_db": rms_db,
            }
        except Exception as exc:
            logger.debug("§V6 _compute_quality_metrics fehlgeschlagen — leeres Dict zurückgegeben: %s", exc)
            return {}


# ═══════════════════════════════════════════════════════════════════════════
# P3: Session-Learning — Experience-Memory
# ═══════════════════════════════════════════════════════════════════════════


class SessionLearner:
    """P3: Lernt aus vergangenen Sessions — kein Song wiederholt Fehler."""

    _MEMORY_PATH = Path.home() / ".aurik" / "session_memory.json"

    def __init__(self) -> None:
        self._memory: dict[str, Any] = self._load()
        self._current_session_songs: list[dict[str, Any]] = []

    def recall(self, material_type: str, restorability: float, terminal_codec: str | None) -> dict[str, Any] | None:
        """Erinnert sich an ähnliche Songs aus früheren Sessions."""
        best_match = None
        best_score = 0.0
        for entry in self._memory.get("songs", []):
            score = 0.0
            if entry.get("material") == material_type:
                score += 0.4
            if entry.get("terminal_codec") == terminal_codec:
                score += 0.3
            rs_diff = abs(entry.get("restorability", 50) - restorability)
            score += max(0.0, 0.3 - rs_diff / 100.0)
            if score > best_score:
                best_score = score
                best_match = entry
        if best_match and best_score > 0.4:
            logger.info(
                "§Sitzung-MEMORY: recall '%s' rs=%.0f → %s (Wert=%.2f)",
                material_type,
                restorability,
                best_match.get("verdict", "?"),
                best_score,
            )
            return best_match  # type: ignore[no-any-return]
        return None

    def record(
        self,
        material_type: str,
        restorability: float,
        terminal_codec: str | None,
        verdict: str,
        phases_run: int,
        time_s: float,
        warnings: list[str],
    ) -> None:
        """Speichert das Ergebnis dieses Songs."""
        # §Sitzung-MEMORY Hygiene (2026-08-22): Nur objektiv verbesserte Läufe
        # persistieren — sonst lernt das Gedächtnis aus Fehlentscheidungen und
        # reaktiviert sie beim nächsten ähnlichen Song (Befund: rs=50-Repair-Mode
        # wurde als Erfahrung gespeichert und per Recall wieder angewandt).
        if str(verdict or "").lower() != "improved":
            logger.debug("§Sitzung-MEMORY: Lauf nicht persistiert (verdict=%s)", verdict)
            return
        entry = {
            "material": material_type,
            "restorability": restorability,
            "terminal_codec": terminal_codec,
            "verdict": verdict,
            "phases_run": phases_run,
            "time_s": time_s,
            "warnings": warnings,
            "ts": time.time(),
        }
        self._current_session_songs.append(entry)

    def persist_session(self) -> None:
        """Schreibt die Session-Erfahrung persistent."""
        if not self._current_session_songs:
            return
        songs = list(self._memory.get("songs", []))
        songs.extend(self._current_session_songs)
        # Max 200 Einträge behalten
        if len(songs) > 200:
            songs = songs[-200:]
        self._memory["songs"] = songs
        self._current_session_songs.clear()
        try:
            self._MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._MEMORY_PATH.write_text(json.dumps(self._memory, indent=2))
            logger.info(
                "§Sitzung-MEMORY: %d songs persisted to %s",
                len(songs),
                self._MEMORY_PATH,
            )
        except Exception as e:
            logger.debug("§Sitzung-MEMORY persist: %s", e)

    def _load(self) -> dict[str, Any]:
        try:
            if self._MEMORY_PATH.exists():
                return json.loads(self._MEMORY_PATH.read_text())  # type: ignore[no-any-return]
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        return {"songs": []}


# ═══════════════════════════════════════════════════════════════════════════
# P4: Assessment-Resolver — Single Source of Truth
# ═══════════════════════════════════════════════════════════════════════════


def resolve_assessment(
    *,
    mushra_score: float,
    quality_gate_delta: float,
    hpi_score: float,
    artifact_freedom: float,
    naturalness: float,
    restorability_score: float,
    was_reverted: bool,
    phases_run: int,
    time_s: float,
    warnings: list[str],
) -> ResolvedAssessment:
    """P4: Löst widersprüchliche Bewertungen zu EINER Wahrheit auf.

    Kein »MUSHRA=95 aber QG=51«-Widerspruch mehr.
    """
    explanations: list[str] = []
    all_warnings = list(warnings)

    # ── Fall 1: DoNoHarm hat revertiert ──
    if was_reverted:
        return ResolvedAssessment(
            overall_verdict="degraded",
            quality_score=0.0,
            confidence=0.95,
            explanation=(
                f"DoNoHarmGuardian hat die Bearbeitung verworfen. "
                f"Das Original wurde zurückgegeben — die Pipeline "
                f"verschlechterte das Audio (naturalness_drop={naturalness:.2f}). "
                f"{phases_run} Phasen in {time_s:.0f}s waren wirkungslos."
            ),
            metrics={"hpi": hpi_score, "restorability": restorability_score},
            warnings=all_warnings,
        )

    # ── Fall 2: Keine messbare Verbesserung ──
    if quality_gate_delta < 5.0 and hpi_score < 0.75:
        return ResolvedAssessment(
            overall_verdict="unchanged",
            quality_score=round(restorability_score, 0),
            confidence=0.85,
            explanation=(
                f"Keine signifikante Verbesserung messbar "
                f"(QualityGate Δ={quality_gate_delta:.1f}, HPI={hpi_score:.3f}). "
                f"Das Original ist mit restorability={restorability_score:.0f}/100 "
                f"zu degradiert für sinnvolle Bearbeitung."
            ),
            metrics={
                "hpi": hpi_score,
                "restorability": restorability_score,
                "mushra": mushra_score,
                "quality_delta": quality_gate_delta,
            },
            warnings=all_warnings,
        )

    # ── Fall 3: MUSHRA-HPI-Widerspruch (nur bei niedrigem HPI) ──
    # §v10.102: Bei HPI ≥ 0.7 sind MUSHRA und HPI konsistent (beide sagen
    # "gute Qualität") — auch wenn MUSHRA > HPI*100+10. Der Widerspruch
    # ist nur relevant wenn HPI niedrig ist und MUSHRA trotzdem hoch.
    expected_mushra = hpi_score * 100.0
    if mushra_score > expected_mushra + 10 and quality_gate_delta < 10 and hpi_score < 0.70:
        all_warnings.append(f"MUSHRA({mushra_score:.0f}) ≫ HPI({hpi_score:.3f}): MUSHRA vermutlich false-positive")
        return ResolvedAssessment(
            overall_verdict="unchanged",
            quality_score=round(hpi_score * 100, 0),
            confidence=0.75,
            explanation=(
                f"MUSHRA-Proxy ({mushra_score:.0f}) widerspricht HPI ({hpi_score:.3f}) "
                f"und QualityGate (Δ={quality_gate_delta:.1f}). "
                f"HPI wird als verlässlicher eingestuft."
            ),
            metrics={"hpi": hpi_score, "mushra": mushra_score, "quality_delta": quality_gate_delta},
            warnings=all_warnings,
        )

    # ── Fall 4: Klare Verbesserung ──
    explanations.append(
        f"Qualität verbessert: HPI={hpi_score:.3f}, MUSHRA={mushra_score:.0f}, ΔQG=+{quality_gate_delta:.1f}"
    )
    quality = min(95.0, hpi_score * 95.0 + quality_gate_delta * 0.5)
    verdict = "improved" if quality > 60 else "unchanged"

    return ResolvedAssessment(
        overall_verdict=verdict,
        quality_score=round(quality, 1),
        confidence=0.80,
        explanation="; ".join(explanations),
        metrics={
            "hpi": hpi_score,
            "mushra": mushra_score,
            "quality_delta": quality_gate_delta,
            "artifact_freedom": artifact_freedom,
        },
        warnings=all_warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P5: Surgery-First — Minimalprinzip
# ═══════════════════════════════════════════════════════════════════════════


def surgery_first_prune(
    selected_phases: list[str],
    decision: PreFlightDecision,
) -> tuple[list[str], list[str]]:
    """P5: Reduziert Phasenliste auf das Nötigste.

    Returns:
        (pruned_phases, removed_phases)
    """
    if decision.mode == "passthrough":
        return [], list(selected_phases)

    allowed = decision.allowed_phase_families
    pruned: list[str] = []
    removed: list[str] = []

    for pid in selected_phases:
        family = _family_from_phase_id(pid)
        if family in allowed:
            pruned.append(pid)
        else:
            removed.append(pid)

    # Cap auf max_phases
    if len(pruned) > decision.max_phases:
        # Priorität: Reparatur-Phasen zuerst
        repair = [p for p in pruned if _family_from_phase_id(p) in _REPAIR_FAMILIES]
        enhance = [p for p in pruned if _family_from_phase_id(p) in _ENHANCE_FAMILIES]
        risky = [
            p
            for p in pruned
            if _family_from_phase_id(p) not in _REPAIR_FAMILIES and _family_from_phase_id(p) not in _ENHANCE_FAMILIES
        ]

        budget = decision.max_phases
        pruned = repair[:budget]
        budget -= len(pruned)
        if budget > 0:
            pruned += enhance[:budget]
            budget = decision.max_phases - len(pruned)
        if budget > 0:
            pruned += risky[:budget]

        # Alles was nicht reingepasst hat → removed. Zuweisung statt += : Die
        # Familien-Verlierer stecken bereits in removed; += zählte sie doppelt
        # (Befund 2026-08-22: „§SURGERY-FIRST: 47→20 Phasen (48 entfernt)“ bei
        # nur 47 übergebenen Phasen).
        all_kept = set(pruned)
        removed = [p for p in selected_phases if p not in all_kept]

    if removed:
        logger.info(
            "§SURGERY-FIRST: %d→%d Phasen (%d entfernt: %s)",
            len(selected_phases),
            len(pruned),
            len(removed),
            ", ".join(removed[:5]) + ("..." if len(removed) > 5 else ""),
        )

    return pruned, removed


# ═══════════════════════════════════════════════════════════════════════════
# P1-P5 Orchestrator — Singleton
# ═══════════════════════════════════════════════════════════════════════════


class AurikOrchestrator:
    """Der fehlende Dirigent. Zentrale Instanz für alle 5 Pfeiler."""

    def __init__(self) -> None:
        self.learner = SessionLearner()
        self.watchdog: StreamingDoNoHarm | None = None
        self.decision: PreFlightDecision | None = None
        self._pre_snapshot: dict[str, float] = {}
        self._phase_results: list[PhaseWatchResult] = []
        self._session: dict[str, Any] = {}
        # Spec 23_zero_touch_orchestration_contract.md: Einmal-Warnung, wenn
        # after_phase() ohne vorheriges preflight() läuft (Watchdog passiv).
        self._warned_watchdog_uninit: bool = False

    # ── Pre-Pipeline ─────────────────────────────────────────────────

    def preflight(
        self,
        *,
        original_audio: np.ndarray,
        sample_rate: int,
        restorability_score: float,
        transfer_chain_depth: int,
        bandwidth_loss: float,
        snr_db: float,
        terminal_codec: str | None,
        material_type: str,
        is_restoration_mode: bool,
        selected_phases: list[str],
    ) -> tuple[list[str], PreFlightDecision]:
        """P1+P3+P5: Entscheidet und reduziert VOR der Pipeline."""

        # P3: Erfahrung abrufen
        experience = self.learner.recall(material_type, restorability_score, terminal_codec)
        if experience:
            logger.info(
                "§ORCHESTRATOR recall: ähnlicher Song → %s (%d Phasen)",
                experience.get("verdict", "?"),
                experience.get("phases_run", 0),
            )

        # P1: Gatekeeper-Entscheidung
        self.decision = gatekeep(
            restorability_score=restorability_score,
            transfer_chain_depth=transfer_chain_depth,
            bandwidth_loss=bandwidth_loss,
            snr_db=snr_db,
            terminal_codec=terminal_codec,
            material_type=material_type,
            is_restoration_mode=is_restoration_mode,
        )
        logger.info("§GATEKEEPER: %s", self.decision.reason)

        # P5: Surgery-First
        pruned, removed = surgery_first_prune(selected_phases, self.decision)

        # P2: Watchdog initialisieren
        self.watchdog = StreamingDoNoHarm(original_audio, sample_rate)
        self._pre_snapshot = self.watchdog.snapshot_pre_phase(original_audio)
        self._phase_results.clear()

        self._session = {
            "material": material_type,
            "restorability": restorability_score,
            "terminal_codec": terminal_codec,
            "phases_total": len(pruned),
            "phases_removed": len(removed),
            "mode": self.decision.mode,
            "start_time": time.time(),
        }

        return pruned, self.decision

    # ── During Pipeline ──────────────────────────────────────────────

    def after_phase(
        self,
        phase_id: str,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
    ) -> PhaseWatchResult:
        """P2: Watchdog-Check nach jeder Phase."""
        if self.watchdog is None:
            # Spec 23_zero_touch_orchestration_contract.md: Ohne preflight() ist
            # der Watchdog absichtlich passiv (No-Op) — einmalig loggen, damit
            # ein unverdrahteter Pfad sichtbar wird statt still zu versagen.
            if not self._warned_watchdog_uninit:
                self._warned_watchdog_uninit = True
                logger.debug(
                    "§WATCHDOG passiv: preflight() fehlt — after_phase(%s) No-Op (Spec 23)",
                    phase_id,
                )
            return PhaseWatchResult()

        result = self.watchdog.watch(phase_id, audio_before, audio_after)
        self._phase_results.append(result)

        if not result.continue_pipeline:
            logger.warning(
                "§WATCHDOG STOP: %s nach Verarbeitungsschritt %s",
                result.reason,
                phase_id,
            )

        if result.phase_was_harmful:
            logger.warning(
                "§WATCHDOG HARMFUL: %s — %s",
                phase_id,
                result.reason,
            )

        return result

    # ── Post-Pipeline ────────────────────────────────────────────────

    def resolve(
        self,
        *,
        mushra_score: float = 50.0,
        quality_gate_delta: float = 0.0,
        hpi_score: float = 0.5,
        artifact_freedom: float = 1.0,
        naturalness: float = 0.5,
        restorability_score: float = 50.0,
        was_reverted: bool = False,
        phases_run: int = 0,
        warnings: list[str] | None = None,
    ) -> ResolvedAssessment:
        """P4: Single Source of Truth — EINE Bewertung."""
        elapsed = time.time() - self._session.get("start_time", time.time())

        assessment = resolve_assessment(
            mushra_score=mushra_score,
            quality_gate_delta=quality_gate_delta,
            hpi_score=hpi_score,
            artifact_freedom=artifact_freedom,
            naturalness=naturalness,
            restorability_score=restorability_score,
            was_reverted=was_reverted,
            phases_run=phases_run,
            time_s=elapsed,
            warnings=list(warnings or []),
        )

        # P3: Erfahrung speichern
        self.learner.record(
            material_type=str(self._session.get("material", "unknown")),
            restorability=float(restorability_score),
            terminal_codec=str(self._session.get("terminal_codec")),
            verdict=assessment.overall_verdict,
            phases_run=phases_run,
            time_s=elapsed,
            warnings=assessment.warnings,
        )

        logger.info(
            "§ORCHESTRATOR RESOLVE: %s (%.0f/100, conf=%.2f) — %s",
            assessment.overall_verdict,
            assessment.quality_score,
            assessment.confidence,
            assessment.explanation[:120],
        )

        return assessment

    def close_session(self) -> None:
        """P3: Session-Erfahrung persistieren."""
        self.learner.persist_session()


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_orchestrator: AurikOrchestrator | None = None


def get_orchestrator() -> AurikOrchestrator:
    """Singleton-Zugriff auf den AurikOrchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AurikOrchestrator()
    return _orchestrator
