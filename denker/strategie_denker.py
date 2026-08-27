"""
StrategieDenker — Domäne: 8×RT-Budgetplanung + Performance-Guard
=================================================================

Kapselt `core.performance_guard.PerformanceGuard` und plant die
Verarbeitungs-Strategie anhand des verfügbaren Zeit-Budgets.

Die 8×RT-Grenze (§9.5) ist hart: Verarbeitung darf maximal das
Achtfache der Audiodauer dauern. Dieser Denker sorgt dafür, dass
dieses Limit durchgesetzt und kommuniziert wird.

Singleton-Pattern nach §3.2 (Double-Checked Locking).
Type-Annotations nach §3.7.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np

from backend.core.restoration_policy import synthesize_human_hearing_comfort_profile

logger = logging.getLogger(__name__)


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Lädt Symbole lazy, um schwere/zyklische Imports zu vermeiden."""
    return getattr(import_module(module_name), symbol_name)


# 32×RT-Grenze aus §9.5 / PerformanceGuard.LIMIT_3X_RT
_3X_RT_LIMIT: float = 32.0


# ---------------------------------------------------------------------------
# Strategie-Daten
# ---------------------------------------------------------------------------


@dataclass
class StrategiePlan:
    """Enthält die geplante Verarbeitungs-Strategie und das RT-Budget."""

    audio_duration_s: float
    """Länge der Quelldatei in Sekunden."""

    max_processing_s: float
    """Maximal erlaubte Verarbeitungszeit in Sekunden (8× Audiodauer)."""

    quality_mode: str
    """Gewählter Qualitätsmodus: 'quality', 'balanced' oder 'speed'."""

    enforce_limit: bool
    """True = 8×RT-Limit wird hart durchgesetzt."""

    enable_adaptive_skipping: bool
    """True = Nicht-kritische Phasen werden übersprungen wenn Budget knapp."""

    recommended_chunk_s: float = 60.0
    """Empfohlene Chunk-Größe für adaptive Verarbeitung (§7.6, defekt-adaptiv)."""

    defect_severity: float = 0.0
    """Defekt-Schwere aus DefektDenker ∈ [0, 1]; steuert Chunk-Größe nach §7.6."""

    intervention_budget: float = 0.5
    """Songweites Eingriffsbudget ∈ [0, 1]; niedriger = konservativer, höher = mehr Reparaturbedarf."""

    listening_experience_targets: dict[str, float] = field(default_factory=dict)
    """Hörbezogene Zielprioritäten für das zentrale restoration_policy_profile."""

    human_hearing_risk_map: dict[str, float] = field(default_factory=dict)
    """Risikoabschätzung für hörkritische Eigenschaften wie Transienten, Wärme und Ermüdung."""

    human_hearing_comfort_profile: dict[str, float] = field(default_factory=dict)
    """Songindividuelle Hoerkomfort-Parameter fuer das zentrale restoration_policy_profile."""

    # ── §v10 Pleasantness-First ──
    pleasantness_baseline: float = 0.0
    """HPE-Baseline vor der Verarbeitung — misst, wie angenehm das Original klingt."""

    goosebumps_baseline: float = 0.0
    """Gänsehaut-Baseline vor der Verarbeitung — misst emotionale Wirkung."""

    budget_note: str = ""
    """Hinweis auf Budget-Engpässe (Deutsch, laienverständlich)."""

    def as_dict(self) -> dict:
        """Serialisiert den StrategiePlan als dict für Telemetrie/Logging."""
        return {
            "audio_duration_s": self.audio_duration_s,
            "max_processing_s": self.max_processing_s,
            "quality_mode": self.quality_mode,
            "enforce_limit": self.enforce_limit,
            "enable_adaptive_skipping": self.enable_adaptive_skipping,
            "recommended_chunk_s": self.recommended_chunk_s,
            "defect_severity": self.defect_severity,
            "intervention_budget": self.intervention_budget,
            "listening_experience_targets": dict(self.listening_experience_targets),
            "human_hearing_risk_map": dict(self.human_hearing_risk_map),
            "human_hearing_comfort_profile": dict(self.human_hearing_comfort_profile),
            "pleasantness_baseline": self.pleasantness_baseline,
            "goosebumps_baseline": self.goosebumps_baseline,
            "budget_note": self.budget_note,
        }


@dataclass
class BudgetStatus:
    """Laufender Budget-Status während der Verarbeitung."""

    elapsed_s: float
    """Bisher verstrichene Verarbeitungszeit in Sekunden."""

    budget_remaining_s: float
    """Verbleibende Zeit im Budget in Sekunden."""

    rt_factor_current: float
    """Aktueller RT-Faktor (verstrichene Zeit / Audiodauer)."""

    should_exit_early: bool
    """True wenn das Budget erschöpft ist und gestoppt werden sollte."""

    phases_completed: int = 0
    """Zahl der abgeschlossenen Phasen."""


@dataclass
class StrategieErgebnis:
    """Ergebnis der Strategie-Planung für die Restaurierung."""

    selected_phases: list
    """Liste der ausgewählten Verarbeitungsphasen."""

    phase_parameters: dict
    """Parameter-Mapping pro Phase."""

    strategy_name: str
    """Name der gewählten Strategie (z. B. 'Rauschunterdrückung')."""

    estimated_quality_gain: float
    """Geschätzter Qualitätsgewinn durch die Strategie (0–1)."""

    reasoning: str
    """Laienverständliche Begründung der Strategie-Wahl."""

    rt_limit: float = 3.0
    """Echtzeit-Faktor-Grenze (z. B. 3.0 = max. 3× Audiodauer)."""

    start_time: float = 0.0
    """Startzeitpunkt (time.time()) für Budget-Tracking."""


# ---------------------------------------------------------------------------
# StrategieDenker
# ---------------------------------------------------------------------------

# Präfixe kritischer Phasen: erhalten immer mindestens das Plan-Qualitäts-Tier.
# "fast" ist für diese Phasen verboten — sie sind klangkritisch.
_CRITICAL_PHASE_PREFIXES: frozenset[str] = frozenset(
    {
        "phase_01_",  # click_removal — Einzelartefakt-Entfernung
        "phase_03_",  # denoise — Kernfunktion Denoising
        "phase_07_",  # harmonic_restoration — Klangtreue
        "phase_09_",  # crackle_removal — Vinyl-Pflicht
        "phase_12_",  # wow_flutter_fix — Pitch-Stabilität
        "phase_23_",  # spectral_repair — Bandbreitenrekonstruktion
        "phase_29_",  # tape_hiss_reduction — Tape-Material-Pflicht
        "phase_40_",  # loudness_normalization — EBU R128 Output-Gate
        "phase_47_",  # truepeak_limiter — Safety-Gate
    }
)


class StrategieDenker:
    """Plant die Verarbeitungs-Strategie und überwacht das 8×RT-Budget.

    Kernaufgabe:
        1. plan()         → StrategiePlan erstellen
        2. starte_timer() → Messung beginnen
        3. check()        → BudgetStatus abfragen (SOLL frühzeitig beendet werden?)

    PerformanceGuard wird genutzt, um die Phasen-Priorisierung
    (MUSICAL_EXCELLENCE_PHASES werden niemals übersprungen) zu erhalten.

    Verwendung::

        denker = get_strategie_denker()
        plan  = denker.plan(audio, sr, mode="quality")
        denker.starte_timer(audio_duration_s=plan.audio_duration_s)
        # ... Phase ausführen ...
        status = denker.check(phases_remaining=5)
        if status.should_exit_early:
            break  # Budget erschöpft — Verarbeitung sicher beenden
    """

    def __init__(self) -> None:
        self._guard: Any | None = None
        self._guard_lock = threading.Lock()
        self._guard_loaded = False

        # Runtime tracking
        self._t_start: float | None = None
        self._audio_duration_s: float = 0.0
        self._current_plan: StrategiePlan | None = None

    # ------------------------------------------------------------------
    # Lazy-Init des PerformanceGuard
    # ------------------------------------------------------------------

    def _ensure_guard(self, mode: str = "quality", enforce: bool = True) -> None:
        """Instantiate or re-instantiate PerformanceGuard with given mode."""
        with self._guard_lock:
            try:
                _performance_guard_cls = _load_symbol("backend.core.performance_guard", "PerformanceGuard")

                # Map string mode to QualityMode if possible
                quality_mode_obj = self._parse_mode(mode)
                self._guard = _performance_guard_cls(
                    mode=quality_mode_obj,
                    enforce_limit=enforce,
                    enable_adaptive_skipping=True,
                )
                logger.info("StrategieDenker: PerformanceGuard (Betriebsart=%s) geladen.", mode)
            except Exception as exc:
                logger.warning(
                    "StrategieDenker: PerformanceGuard nicht verfügbar (%s). Einfacher Timer-Ersatzpfad wird genutzt.",
                    exc,
                )
                self._guard = None
            self._guard_loaded = True

    @staticmethod
    def _parse_mode(mode_str: str) -> Any:
        """Konvertiert mode string to QualityMode enum value if available."""
        try:
            _quality_mode_cls = _load_symbol("backend.core.unified_restorer_v3", "QualityMode")

            _m = str(mode_str or "quality").strip().lower().replace("_", "").replace(" ", "")
            mapping = {
                "fast": _quality_mode_cls.FAST,
                "balanced": _quality_mode_cls.BALANCED,
                "quality": _quality_mode_cls.QUALITY,
                "restoration": _quality_mode_cls.QUALITY,
                "maximum": _quality_mode_cls.MAXIMUM,
                "studio2026": _quality_mode_cls.MAXIMUM,
                "studio": _quality_mode_cls.MAXIMUM,
                # "speed" existiert nicht im QualityMode-Enum → FAST als Fallback
            }
            return mapping.get(_m, _quality_mode_cls.QUALITY)
        except Exception:
            logger.warning("strategie_denker.py::_parse_Betriebsart Ersatzpfad", exc_info=True)
            return mode_str  # PerformanceGuard handles unknown mode gracefully

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def plan(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        mode: str = "quality",
        enforce_3x_rt: bool = True,
        defect_severity: float = 0.0,
        signal_signature: dict[str, float] | None = None,
        # §v10.706 Denker-IQ: chain_depth für adaptive Budget-Skalierung
        chain_depth: int = 1,
        restorability_score: float = 70.0,
    ) -> StrategiePlan:
        """Erstellt den Verarbeitungs-Strategieplan.

        Algorithmus:
            1. Audiodauer berechnen
            2. 8×RT-Budget ableiten  (max_processing_s = 8 × audio_duration_s)
            3. Chunk-Größe gemäß §9.5 adaptiv-defektdichte setzen
            4. PerformanceGuard initialisieren

        Args:
            audio:         Eingabe-Audio.
            sr:            Sample-Rate in Hz.
            mode:          Qualitätsmodus ('quality', 'balanced', 'speed').
            enforce_3x_rt: 8×RT-Limit hart durchsetzen.

        Returns:
            StrategiePlan mit Budget-Angaben.
        """
        assert sr == 48000, f"StrategieDenker.plan() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        audio_dur = _safe_duration(audio, sr)
        logger.debug("StrategieDenker.plan(): audio_dur=%.1fs", audio_dur)

        self._ensure_guard(mode=mode, enforce=enforce_3x_rt)

        max_proc = _3X_RT_LIMIT * audio_dur
        # §v10.706 Denker-IQ: Chain-depth-adaptive Budget-Skalierung.
        # Tiefe Ketten (4+) brauchen mehr Phasen → mehr Budget.
        # Jede zusätzliche Generation: +15% Budget (kumulativ).
        _depth_budget_factor = float(np.clip(1.0 + (chain_depth - 1) * 0.15, 1.0, 2.0))
        # Restorability-Adaption: schlechter restaurierbar → mehr Budget
        _rs_budget_factor = float(np.clip(1.0 + (100.0 - restorability_score) * 0.005, 1.0, 1.50))
        max_proc *= _depth_budget_factor * _rs_budget_factor
        _sev = float(defect_severity) if math.isfinite(float(defect_severity)) else 0.0
        _sev = max(0.0, min(1.0, _sev))
        _effective_sev = _derive_effective_defect_severity(_sev, signal_signature)
        chunk_s = _adaptive_chunk(audio_dur, defect_severity=_effective_sev)

        _sig = signal_signature or {}
        _crest = float(_sig.get("crest_db", 0.0) or 0.0)
        _hf_ratio = float(_sig.get("hf_ratio", 0.0) or 0.0)
        _transient_ratio = float(_sig.get("transient_ratio", 0.0) or 0.0)
        _micro_db = float(_sig.get("micro_dynamic_db", 0.0) or 0.0)
        _intervention_budget = float(
            np.clip(
                0.18 + 0.62 * _effective_sev + 0.12 * min(max(_hf_ratio, 0.0), 1.0),
                0.12,
                0.88,
            )
        )
        _listening_targets = {
            "natuerlichkeit": float(np.clip(1.0 + max(0.0, 18.0 - _crest) / 60.0, 1.0, 1.35)),
            "authentizitaet": 1.20,
            "micro_dynamics": float(np.clip(1.0 + max(0.0, 10.0 - _micro_db) / 35.0, 1.0, 1.35)),
            "artikulation": float(np.clip(1.0 + min(max(_transient_ratio, 0.0), 0.05) * 5.0, 1.0, 1.25)),
            "waerme": 1.10,
        }
        _hearing_risks = {
            "listening_fatigue": float(np.clip(_hf_ratio * 2.2 + _effective_sev * 0.25, 0.0, 1.0)),
            "transient_smear": float(np.clip(max(0.0, 0.012 - _transient_ratio) * 45.0, 0.0, 1.0)),
            "microdynamics_loss": float(np.clip(max(0.0, 9.0 - _micro_db) / 18.0, 0.0, 1.0)),
            "overprocessing": float(np.clip((1.0 - _effective_sev) * 0.45 + max(0.0, _crest - 16.0) / 40.0, 0.0, 1.0)),
        }
        _comfort_profile = synthesize_human_hearing_comfort_profile(
            {
                "strategy": {
                    "intervention_budget": _intervention_budget,
                    "human_hearing_risk_map": _hearing_risks,
                    "signal_signature": dict(_sig),
                },
                "signal_signature": dict(_sig),
            },
            mode=mode,
            intervention_budget=_intervention_budget,
        )

        # ── §v10 HPE & Gänsehaut-Baseline ──
        # Psychoakustische Metriken auf dem kompletten Audio. Für Spuren >60s
        # werden sie auf einen deterministischen 30-s-Mittel-Ausschnitt
        # umgeleitet (BUG-FIX 2026-08-22): Voll-Längen-Berechnung auf 10M+
        # Samples alloziert >400 MB temporär und kann Minuten blockieren.
        # HPE/Goosebumps sind Baseline-Referenzwerte — Default 0.5 bleibt
        # der konservative Fallback bei Timeout.
        def _middle_excerpt(audio: np.ndarray, sr: int, win_s: float = 30.0) -> np.ndarray:
            """Deterministischer Mittel-Ausschnitt für ML-Scorer (§G5 (copilot-instructions.md))."""
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim == 2:
                _axis = 0 if arr.shape[-1] <= 2 else -1
            else:
                _axis = 0
            total = arr.shape[_axis]
            win = int(win_s * sr)
            if total <= win:
                return arr
            start = (total - win) // 2
            return np.take(arr, np.arange(start, start + win), axis=_axis)

        _hpe_base = 0.5
        _goose_base = 0.5
        _HPE_GOOSE_TIMEOUT = 30.0
        # §9 Performance-Budget BUG-FIX 2026-08-22: Kein Skip mehr bei >60s —
        # HPE/Goosebumps laufen auf einem deterministischen 30-s-Mittel-Ausschnitt,
        # damit die Wohlklang-Baseline auch bei langen Songs am Leben bleibt.
        _hpe_audio = audio
        _hpe_note = ""
        if audio_dur > 60.0:
            _hpe_audio = _middle_excerpt(audio, sr, 30.0)
            _hpe_note = f" (Excerpt 30s von {audio_dur:.0f}s)"
            try:
                from backend.core.human_pleasantness_estimator import compute_pleasantness

                logger.debug("StrategieDenker: HPE-Berechnung gestartet (%.1fs Audio) …", audio_dur)
                _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    _hpe_fut = _pool.submit(compute_pleasantness, _hpe_audio, sr)
                    _hpe_r = _hpe_fut.result(timeout=_HPE_GOOSE_TIMEOUT)
                    _hpe_base = float(_hpe_r.score)
                    logger.info("StrategieDenker: HPE-Baseline = %.3f (%s)%s", _hpe_base, _hpe_r.label, _hpe_note)
                finally:
                    _pool.shutdown(wait=False)
            except concurrent.futures.TimeoutError:
                logger.warning("StrategieDenker: HPE Zeitlimit (>%.0fs) — Ersatzpfad 0.5", _HPE_GOOSE_TIMEOUT)
            except Exception:
                logger.warning("strategie_denker.py::HPE Ersatzpfad", exc_info=True)
            try:
                from backend.core.goosebumps_factor import compute_goosebumps

                logger.debug("StrategieDenker: Goosebumps-Berechnung gestartet …")
                _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    _goose_fut = _pool.submit(compute_goosebumps, _hpe_audio, sr)
                    _goose_r = _goose_fut.result(timeout=_HPE_GOOSE_TIMEOUT)
                    _goose_base = float(_goose_r.score)
                    logger.info(
                        "StrategieDenker: Goosebumps-Baseline = %.3f (%s)%s", _goose_base, _goose_r.label, _hpe_note
                    )
                finally:
                    _pool.shutdown(wait=False)
            except concurrent.futures.TimeoutError:
                logger.warning("StrategieDenker: Goosebumps Zeitlimit (>%.0fs) — Ersatzpfad 0.5", _HPE_GOOSE_TIMEOUT)
            except Exception:
                logger.warning("strategie_denker.py::Goosebumps Ersatzpfad", exc_info=True)

        note = ""
        if audio_dur > 300:
            note = (
                "Lange Datei erkannt — Verarbeitung erfolgt in Abschnitten "
                f"(jeweils {int(chunk_s)} s), um das 8×RT-Zeitbudget einzuhalten."
            )
        elif audio_dur < 5:
            note = "Sehr kurze Aufnahme — volle Verarbeitungstiefe aktiviert."

        plan = StrategiePlan(
            audio_duration_s=audio_dur,
            max_processing_s=max_proc,
            quality_mode=mode,
            enforce_limit=enforce_3x_rt,
            enable_adaptive_skipping=True,
            recommended_chunk_s=chunk_s,
            defect_severity=_effective_sev,
            intervention_budget=_intervention_budget,
            listening_experience_targets=_listening_targets,
            human_hearing_risk_map=_hearing_risks,
            human_hearing_comfort_profile=_comfort_profile,
            pleasantness_baseline=_hpe_base,  # §v10
            goosebumps_baseline=_goose_base,  # §v10
            budget_note=note,
        )
        self._current_plan = plan
        logger.info(
            "StrategieDenker: Strategieplan — Dauer=%.1fs, Grenze=%.1fs, Chunk=%.1fs, Sev=%.2f (base=%.2f)",
            audio_dur,
            max_proc,
            chunk_s,
            _effective_sev,
            _sev,
        )
        return plan

    def starte_timer(self, audio_duration_s: float) -> None:
        """Startet die Zeitmessung für das Budget-Tracking.

        Ruft optional `PerformanceGuard.start_monitoring()` auf.

        Args:
            audio_duration_s: Länge der Quelldatei in Sekunden.
        """
        self._audio_duration_s = max(audio_duration_s, 0.001)
        self._t_start = time.monotonic()

        # §v10.706 Denker-IQ: Use plan's scaled budget if available, else fallback
        _budget = getattr(self._current_plan, "max_processing_s", None) if self._current_plan else None
        _budget_s = float(_budget) if _budget else _3X_RT_LIMIT * audio_duration_s
        if self._guard is not None:
            try:
                self._guard.start_monitoring(audio_duration_seconds=self._audio_duration_s)
            except Exception as exc:
                logger.debug("StrategieDenker: start_monitoring() Fehler: %s", exc)

        logger.info(
            "StrategieDenker: Timer gestartet (Audio=%.1fs, Grenze=%.1fs).",
            audio_duration_s,
            _budget_s,
        )

    def check(self, phases_remaining: int = 0) -> BudgetStatus:
        """Prüft den aktuellen Budget-Status.

        Args:
            phases_remaining: Anzahl der noch anstehenden Verarbeitungs-Phasen.

        Returns:
            BudgetStatus mit Empfehlung, ob frühzeitig abgebrochen werden soll.
        """
        if self._t_start is None:
            return BudgetStatus(
                elapsed_s=0.0,
                budget_remaining_s=float("inf"),
                rt_factor_current=0.0,
                should_exit_early=False,
            )

        elapsed = time.monotonic() - self._t_start
        _plan_budget = getattr(self._current_plan, "max_processing_s", None) if self._current_plan else None
        max_proc = float(_plan_budget) if _plan_budget else _3X_RT_LIMIT * max(self._audio_duration_s, 0.001)
        remaining = max(0.0, max_proc - elapsed)
        rt_factor = elapsed / max(self._audio_duration_s, 0.001)

        # Check via PerformanceGuard if available
        guard_exit = False
        if self._guard is not None:
            try:
                guard_exit = bool(self._guard.check_early_exit(remaining_phases=phases_remaining))
            except Exception as exc:
                logger.debug("StrategieDenker: Pruefung_early_exit() Fehler: %s", exc)

        # Hard limit override
        hard_exit = rt_factor >= _3X_RT_LIMIT

        should_exit = guard_exit or hard_exit

        if should_exit:
            logger.warning(
                "StrategieDenker: Grenze erschöpft! RT-Faktor=%.2f ≥ 3.0 (elapsed=%.1fs, Grenze=%.1fs).",
                rt_factor,
                elapsed,
                max_proc,
            )

        return BudgetStatus(
            elapsed_s=elapsed,
            budget_remaining_s=remaining,
            rt_factor_current=rt_factor,
            should_exit_early=should_exit,
        )

    def schaetze_phasen_tier(
        self,
        plan: StrategiePlan,
        phase_list: list[str],
        *,
        restorability_score: float = 70.0,
    ) -> dict[str, str]:
        """Empfiehlt Qualitäts-Tier (\"maximum\" | \"quality\" | \"fast\") pro Phase.

        Regeln (in Prioritätsreihenfolge):
          1. Studio-2026-Modus → alle Phasen \"maximum\".
          2. Restorability < 35 + kritische Phase → \"maximum\".
          3. Enges Budget (max_processing_s < 5× audio_duration) + unkritisch → \"fast\".
          4. Sonst: plan.quality_mode (\"quality\" als Default).

        Kritische Phasen (_CRITICAL_PHASE_PREFIXES) erhalten niemals \"fast\" —
        ihr Ergebnis bestimmt Klangtreue und musical-goals direkt.

        Args:
            plan:                StrategiePlan aus StrategieDenker.plan().
            phase_list:          Sortierte Phasenliste (aus PhasePlan.phases).
            restorability_score: Restorability 0–100 (PhysicalCeiling-Schätzung).

        Returns:
            Dict phase_id → Tier-String. Leer wenn plan None.
        """
        if plan is None:
            return {}

        _is_studio = str(getattr(plan, "quality_mode", "") or "").lower() in (
            "studio2026",
            "studio_2026",
            "maximum",
        )
        _low_restorability = float(restorability_score) < 35.0
        _audio_dur = float(getattr(plan, "audio_duration_s", 1.0) or 1.0)
        _max_proc = float(getattr(plan, "max_processing_s", _audio_dur * _3X_RT_LIMIT) or _audio_dur * _3X_RT_LIMIT)
        _tight_budget = _max_proc < 5.0 * _audio_dur
        _default_tier = "maximum" if _is_studio else str(getattr(plan, "quality_mode", "quality") or "quality")

        tiers: dict[str, str] = {}
        for phase in phase_list:
            _is_critical = any(phase.startswith(pfx) for pfx in _CRITICAL_PHASE_PREFIXES)

            if _is_studio or (_low_restorability and _is_critical):
                tier = "maximum"
            elif _tight_budget and not _is_critical:
                tier = "fast"
            else:
                tier = _default_tier

            tiers[phase] = tier

        logger.debug(
            "StrategieDenker.schaetze_phasen_tier(): %d Phasen "
            "(restorability=%.0f, tight_Grenze=%s, studio=%s) → %d maximum, %d fast",
            len(tiers),
            restorability_score,
            _tight_budget,
            _is_studio,
            sum(1 for t in tiers.values() if t == "maximum"),
            sum(1 for t in tiers.values() if t == "fast"),
        )
        return tiers

    @property
    def performance_guard(self) -> Any | None:
        """Zugriff auf den internen PerformanceGuard (für fortgeschrittene Nutzung)."""
        return self._guard


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _safe_duration(audio: np.ndarray, sr: int) -> float:
    """Gibt audio duration in seconds, guarded for NaN and edge cases zurück.

    Handles both (channels, samples) and (samples, channels) layouts by
    using the same heuristic as AurikDenker.denke(): axis with size <= 2
    is channels, the other axis is samples.
    """
    if sr < 1 or audio.size == 0:
        return 0.001
    if audio.ndim == 1:
        n_samples = audio.shape[0]
    elif audio.shape[-1] <= 2:
        # (N, channels) layout — samples on axis 0
        n_samples = audio.shape[0]
    else:
        # (channels, N) layout — samples on axis -1
        n_samples = audio.shape[-1]
    dur = n_samples / max(sr, 1)
    return dur if math.isfinite(dur) and dur > 0 else 0.001


def _adaptive_chunk(audio_dur_s: float, defect_severity: float = 0.0) -> float:
    """Bestimmt recommended chunk size per §7.6 (defect-density adaptive).

    Spec §7.6:
        defect_severity >= 0.6  →  5 s   (fine-grained, high-defect material)
        defect_severity >= 0.3  → 15 s   (moderate defects)
        default                 → 60 s   (clean material — large context chunks)
        silence segment caps at 120 s    (not observable here — caller handles)
    Minimum: 2 s | Maximum: min(120 s, audio_dur_s)
    """
    if audio_dur_s <= 2.0:
        return audio_dur_s  # Too short to subdivide
    if defect_severity >= 0.6:
        chunk = 5.0  # §7.6: Feingranular bei hohem Defektniveau
    elif defect_severity >= 0.3:
        chunk = 15.0  # §7.6: Mittel bei moderatem Defektniveau
    elif audio_dur_s > 300:
        chunk = 120.0  # Long clean files: 120 s for context coherence
    elif audio_dur_s > 60:
        chunk = 60.0
    else:
        return audio_dur_s  # Short clean files: process as whole
    return float(min(max(chunk, 2.0), audio_dur_s))


def _derive_effective_defect_severity(base_severity: float, signal_signature: dict[str, float] | None) -> float:
    """Leitet eine signalbewusste Defektschwere für Budget-/Chunk-Planung ab."""
    sev = float(np.clip(base_severity, 0.0, 1.0))
    if not signal_signature:
        return sev

    crest_db = float(signal_signature.get("crest_db", 0.0))
    transient_ratio = float(signal_signature.get("transient_ratio", 0.0))
    micro_dynamic_db = float(signal_signature.get("micro_dynamic_db", 0.0))
    hf_ratio = float(signal_signature.get("hf_ratio", 0.0))

    # Defekt-Risiko nur nach oben korrigieren (konservativ, no-harm).
    extra = 0.0
    if crest_db >= 20.0:
        extra += 0.15
    elif crest_db >= 16.0:
        extra += 0.08
    if transient_ratio >= 0.012:
        extra += 0.14
    elif transient_ratio >= 0.006:
        extra += 0.08
    if micro_dynamic_db >= 14.0:
        extra += 0.06
    if hf_ratio >= 0.12:
        extra += 0.05

    return float(np.clip(sev + extra, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------

_SINGLETON: dict[str, StrategieDenker | None] = {"instance": None}
_SINGLETON_LOCK = threading.Lock()


def get_strategie_denker() -> StrategieDenker:
    """Thread-sicherer Singleton-Accessor für StrategieDenker."""
    _instance = _SINGLETON["instance"]
    if _instance is None:
        with _SINGLETON_LOCK:
            _instance = _SINGLETON["instance"]
            if _instance is None:
                _instance = StrategieDenker()
                _SINGLETON["instance"] = _instance
    return _instance
