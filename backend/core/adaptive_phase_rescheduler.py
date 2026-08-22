"""
AdaptivePhaseRescheduler — Geschlossener Regelkreis für UV3-Pipeline (§2.78)
=============================================================================

Nach jeder Phase wird geprüft, ob verbleibende Musical-Goal-Lücken
Recovery-Phasen erfordern, die noch nicht im Ausführungsplan stehen.
Solche Phasen werden dynamisch an das Ende von `selected_phases` angehängt
(Python for-Loop besucht neu angehängte Listenelemente — kein Loop-Umbau nötig).

Invarianten:
- §0a: phase_21_exciter / phase_35_multiband_compression / phase_42_vocal_enhancement
        niemals in Restoration injiziert (get_goal_recovery_phases garantiert es;
        Rescheduler prüft zusätzlich als Sicherheitsnetz)
- §2.45 Minimal-Intervention: Injektion nur wenn gap > GAP_THRESHOLD (0.05)
- §2.52 _NEVER_SKIP: Diese Phasen sind immer im Plan — keine Injektion nötig
- §2.65 MAS: Aufrufer prüft _mas_fully_achieved vor reschedule()-Aufruf (non-blocking)
- §0c: Kein song-spezifischer Hardcode; alle Entscheidungen via get_goal_recovery_phases()
- §2.67 Phase-Koalitionen: keine Injektion von Einzel-Phasen aus Koalitionen
- Max-Injektion: MAX_INJECTIONS_PER_SESSION (3) pro Song-Session
- Singleton (thread-safe, double-checked locking)

Author: Aurik Development Team
Version: 1.0.0 (§2.78 Closed-Loop Rescheduler v10.0.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from backend.core.calibration_matrix import (
    compute_tcci as _compute_tcci,
)
from backend.core.calibration_matrix import (
    get_goal_recovery_phases as _get_goal_recovery_phases,
)

logger = logging.getLogger(__name__)

# ── Limits (statische Defaults — adaptiver Wert per Song via Hilfsfunktionen) ─────────────
MAX_INJECTIONS_PER_SESSION: int = 3  # Basis-Limit; adaptiv erhöht bei niedriger Restorability
GAP_THRESHOLD: float = 0.05  # Basis-Schwelle; adaptiv gesenkt bei niedriger Restorability


def _adaptive_gap_threshold(restorability_score: float) -> float:
    """Berechnet einen restorability-adaptiven GAP_THRESHOLD.

    Logik (§2.45 Minimal-Intervention + §0k Maximum-Achievable-Score):
    - Gut restaurierbare Songs (rest ≈ 100): 0.05 — nur klare Lücken auslösen Injektion.
    - Schwer restaurierbare Songs (rest ≈ 0):  0.025 — kleinere Lücken schon ansprechen,
      weil jede Recovery-Chance zählt und falsch-negative teurer sind als falsch-positive.

    Lineare Interpolation zwischen den Polen; Ergebnis auf [0.020, 0.05] geclippt.
    v10.0.0: Basis-Schwelle 0.035→0.030 für sensitivere Recovery bei mittlerer Qualität.
    """
    _rest = float(np.clip(restorability_score, 0.0, 100.0))
    return float(np.clip(0.020 + 0.030 * (_rest / 100.0), 0.020, 0.050))


def _adaptive_max_injections(restorability_score: float) -> int:
    """Berechnet das restorability-adaptive Session-Injektionslimit.

    Logik:
    - rest > 50: Limit 3 (Standard — gut restaurierbare Songs brauchen wenig Recovery).
    - rest <= 50: Limit 4 — mehr Recovery-Versuche erlaubt.
    - rest <= 25: Limit 5 — maximale Recovery-Versuche für sehr schwierige Songs.

    §0h/§0c: Kein Hardcode song-spezifisch; nur restorability-basiert.
    """
    _rest = float(np.clip(restorability_score, 0.0, 100.0))
    if _rest <= 25.0:
        return 5
    if _rest <= 50.0:
        return 4
    return 3


def _sanitize_confidence(value: object | None, default: float = 0.65) -> float:
    """Normalisiert Goal-Confidence robust auf [0, 1]."""
    try:
        if value is None:
            return float(default)
        v = float(value)  # type: ignore[arg-type]
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return float(np.clip(v, 0.0, 1.0))
    except Exception as e:
        logger.warning("adaptive_Verarbeitungsschritt_rescheduler.py::_sanitize_confidence Ersatzpfad: %s", e)
        return float(default)


def _goal_recovery_priority(
    goal: str,
    gap: float,
    confidence: float,
    transfer_chain: list[str],
    material_type: str,
    coalition_bonus: float = 0.0,
) -> float:
    """Berechnet eine prädiktive Priorität für Recovery-Ziele.

    Größere Lücken bleiben wichtig, werden aber durch Confidence, Transfer-Chain-
    Komplexität und goal-spezifische Risiko-Sensitivität ergänzt. Dadurch
    priorisiert der Rescheduler nicht nur den größten numerischen Gap, sondern
    den wahrscheinlichsten und am stärksten zielwirksamen Recovery-Schritt.
    """
    _gap = float(np.clip(gap, 0.0, 1.0))
    if _gap <= 0.0:
        return 0.0

    _conf = _sanitize_confidence(confidence, default=0.65)
    _tcci = float(np.clip(_compute_tcci(transfer_chain), 0.0, 1.0))
    _goal = str(goal or "").strip().lower()
    _material = str(material_type or "unknown").strip().lower()

    _hf_detail_goals = frozenset({"brillanz", "transparenz", "transient_energie", "artikulation"})
    _analog_stability_goals = frozenset(
        {"natuerlichkeit", "authentizitaet", "emotionalitaet", "waerme", "timbre_authentizitaet"}
    )
    _spatial_goals = frozenset({"spatial_depth", "separation_fidelity"})
    _analog_materials = frozenset(
        {"cassette", "tape", "vinyl", "shellac", "reel_tape", "wire_recording", "wax_cylinder"}
    )

    _confidence_weight = 0.60 + 0.40 * _conf

    if _goal in _hf_detail_goals:
        _chain_weight = 1.0 + 0.22 * _tcci
    elif _goal in _spatial_goals:
        _chain_weight = 1.0 + 0.08 * _tcci
    else:
        _chain_weight = 1.0 + 0.05 * _tcci

    _material_weight = 1.0
    if _material in _analog_materials:
        if _goal in _analog_stability_goals:
            _material_weight = 1.08
        elif _goal in _hf_detail_goals:
            _material_weight = 1.05
        elif _goal in _spatial_goals:
            _material_weight = 0.98
    elif _material in {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}:
        if _goal in _hf_detail_goals:
            _material_weight = 1.10
        elif _goal in _spatial_goals:
            _material_weight = 0.95

    _coalition_weight = 1.0 + float(np.clip(coalition_bonus, 0.0, 1.0)) * 0.18
    _priority = _gap * _confidence_weight * _chain_weight * _material_weight * _coalition_weight
    return float(np.clip(_priority, 0.0, 1.0))


def _goal_recovery_coalition_bonus(
    goal: str,
    primary_phase: str,
    recovery_phase_map: dict[str, list[str]],
    goal_gaps: dict[str, float],
    goal_confidence: dict[str, float] | None,
) -> float:
    """Schätzt, wie viele weitere offene Goals dieselbe Recovery-Phase mittragen.

    Ein Bonus entsteht, wenn die Primär-Phase eines Goals auch in den Recovery-Listen
    anderer noch offener Goals auftaucht. Dadurch priorisiert der Rescheduler Phasen,
    die mehrere Defizite in einem Schritt mit adressieren können.
    """
    _primary = str(primary_phase or "").strip()
    if not _primary:
        return 0.0

    _support = 0.0
    for _other_goal, _other_gap in goal_gaps.items():
        if _other_goal == goal:
            continue
        _recovery = recovery_phase_map.get(_other_goal, [])
        if not _recovery:
            continue
        if _primary not in _recovery[:2]:
            continue
        _other_conf = _sanitize_confidence((goal_confidence or {}).get(_other_goal), default=0.65)
        # Höhere, verlässlichere und größere Lücken zählen stärker.
        _support += float(np.clip(_other_gap, 0.0, 1.0)) * (0.55 + 0.45 * _other_conf)

    return float(np.clip(_support, 0.0, 1.0))


def _estimate_chain_rarity(transfer_chain: list[str]) -> float:
    """Schätzt die Seltenheit/Komplexität einer Transfer-Chain in [0,1]."""
    _chain = [str(v).strip().lower() for v in (transfer_chain or []) if str(v).strip()]
    if not _chain:
        return 0.0
    _len_term = float(np.clip((len(_chain) - 1) / 5.0, 0.0, 1.0))
    _uniq_term = float(np.clip((len(set(_chain)) - 1) / 4.0, 0.0, 1.0))
    _analog = {"vinyl", "shellac", "tape", "reel_tape", "cassette", "wire_recording", "wax_cylinder"}
    _lossy = {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
    _mixed_chain = 1.0 if any(c in _analog for c in _chain) and any(c in _lossy for c in _chain) else 0.0
    return float(np.clip(0.40 * _len_term + 0.35 * _uniq_term + 0.25 * _mixed_chain, 0.0, 1.0))


def _phase_intervention_risk_price(
    phase_id: str,
    *,
    material_type: str,
    transfer_chain: list[str],
    is_studio_2026: bool,
) -> float:
    """Schätzt Interventionsrisiko einer Recovery-Phase in [0,1]."""
    _phase = str(phase_id or "").strip().lower()
    _material = str(material_type or "unknown").strip().lower()
    _tcci = float(np.clip(_compute_tcci(transfer_chain), 0.0, 1.0))

    _base_risk = 0.20
    if _phase.startswith(("phase_24", "phase_55", "phase_50")):
        _base_risk = 0.30
    elif _phase.startswith(("phase_06", "phase_07", "phase_23")):
        _base_risk = 0.36
    elif _phase.startswith(("phase_42", "phase_46", "phase_48")):
        _base_risk = 0.40
    elif _phase.startswith(("phase_35", "phase_21", "phase_26")):
        _base_risk = 0.45

    _analog = {"vinyl", "shellac", "tape", "reel_tape", "cassette", "wire_recording", "wax_cylinder"}
    _material_penalty = (
        0.08 if _material in _analog and _phase.startswith(("phase_06", "phase_07", "phase_23")) else 0.0
    )
    _mode_penalty = 0.05 if (not is_studio_2026 and _phase.startswith(("phase_06", "phase_07", "phase_23"))) else 0.0
    _chain_penalty = 0.14 * _tcci

    return float(np.clip(_base_risk + _material_penalty + _mode_penalty + _chain_penalty, 0.0, 0.95))


def _is_counterfactual_safe_phase(phase_id: str) -> bool:
    """Phasen mit vorhandenem localized-counterfactual Guard in PMGG."""
    _phase = str(phase_id or "")
    return _phase.startswith(("phase_24", "phase_50", "phase_55"))


# §0a-verbotene Phasen — dürfen in Restoration nie injiziert werden
_RESTORATION_FORBIDDEN: frozenset[str] = frozenset(
    {
        "phase_21_exciter",
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
    }
)

# §2.52 _NEVER_SKIP-Phasen laufen immer — keine Injektion nötig (redundant / harmlos, aber klar)
_NEVER_SKIP: frozenset[str] = frozenset(
    {
        "phase_01_click_removal",
        "phase_09_crackle_removal",
        "phase_12_wow_flutter_fix",
        "phase_14_phase_correction",
        "phase_15_stereo_balance",
        "phase_30_dc_offset_removal",
        "phase_47_truepeak_limiter",
        "phase_65_vocal_naturalness_restoration",
    }
)

# Goals die per VQI-Gate / HPI-Gate eigene Recovery-Pfade haben
# → kein Rescheduler-Override nötig (würde nur duplizieren)
_SELF_MANAGED_GOALS: frozenset[str] = frozenset(
    {
        "vocal_quality",  # VQI-Gate §0p / _recovery_cascade()
        "formant_fidelity",  # Formant-Guard §0p / phase_04-Rollback
    }
)

# Prioritäts-Reihenfolge der Goals (P0 > P1 > P2 > P3 > P4 > P5)
# Der Rescheduler arbeitet Goals von höchster zu niedrigster Prio ab.
_GOAL_PRIORITY: list[str] = [
    # P1
    "natuerlichkeit",
    "authentizitaet",
    # P2
    "tonal_center",
    "timbre_authentizitaet",
    "timbre",
    "artikulation",
    "transient_energie",
    # P3
    "emotionalitaet",
    "micro_dynamics",
    "groove",
    # P4
    "transparenz",
    "waerme",
    "bass_kraft",
    "separation_fidelity",
    # P5
    "brillanz",
    "spatial_depth",
]


@dataclass
class RescheduleResult:
    """Ergebnis eines reschedule()-Aufrufs."""

    new_phases_to_append: list[str] = field(default_factory=list)
    """Phasen-IDs die ans Ende von selected_phases angehängt werden sollen."""

    goal_gaps_found: dict[str, float] = field(default_factory=dict)
    """Goal-Lücken die Injektion ausgelöst haben (für Logging/Telemetry)."""


class AdaptivePhaseRescheduler:
    """Closed-Loop Rescheduler für die UV3-Pipeline.

    Analysiert nach jeder Phase die verbleibenden Goal-Lücken und
    injiziert Recovery-Phasen in selected_phases (Append).

    Singleton-Pattern: nie direkt instanziieren — `get_adaptive_phase_rescheduler()` verwenden.
    """

    def __init__(self) -> None:
        self._op_lock = threading.Lock()
        self._session_injected: set[str] = set()  # Phasen die in dieser Session injiziert wurden

    # ─── Public API ────────────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """Setzt Session-State zurück (aufrufen: je Song-Session, nach _conductor.reset())."""
        with self._op_lock:
            self._session_injected.clear()

    def reschedule(
        self,
        current_goal_scores: dict[str, float],
        song_goal_targets: dict[str, float],
        selected_phases: list[str],
        executed: set[str],
        is_studio_2026: bool = False,
        transfer_chain: list[str] | None = None,
        material_type: str = "unknown",
        restorability_score: float | None = None,
        goal_confidence: dict[str, float] | None = None,
        uncertainty_budget: float | None = None,
    ) -> RescheduleResult:
        """Berechnet Phasen-Injektionen basierend auf aktuellen Goal-Lücken.

        Args:
            current_goal_scores: Goal-Proxy-Scores nach letzter Phase (aus _fast_goal_snapshot).
            song_goal_targets: Per-Song Studio-Day-Targets (aus estimate_song_goal_targets).
            selected_phases: Aktuelle (veränderliche) Phase-Liste — wird NICHT verändert hier;
                             Caller fügt `new_phases_to_append` ans Ende.
            executed: Menge bereits ausgeführter Phase-IDs (inkl. skipped).
            is_studio_2026: Modus-Flag (§0a-Guard für Restoration).
            transfer_chain: Transfer-Chain aus _restoration_context (für chain-aware Recovery).
            material_type: Materialklasse für material-sensitive Gap-Bewertung.
            restorability_score: RestorabilityEstimator-Score [0–100] — steuert adaptive
                                 GAP_THRESHOLD und MAX_INJECTIONS (§0k, §2.45).

        Returns:
            RescheduleResult mit new_phases_to_append-Liste (leer bei keinem Bedarf).

        Non-blocking: Exceptions → leeres RescheduleResult (Pipeline läuft weiter).
        """
        # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 70.0).
        from backend.core.calibration_context import resolve_restorability_score

        restorability_score = resolve_restorability_score(restorability_score)
        try:
            return self._reschedule_internal(
                current_goal_scores=current_goal_scores,
                song_goal_targets=song_goal_targets,
                selected_phases=selected_phases,
                executed=executed,
                is_studio_2026=is_studio_2026,
                transfer_chain=list(transfer_chain or []),
                material_type=str(material_type or "unknown"),
                restorability_score=float(restorability_score),
                goal_confidence=dict(goal_confidence or {}),
                uncertainty_budget=(
                    None if uncertainty_budget is None else float(np.clip(float(uncertainty_budget), 0.0, 1.0))
                ),
            )
        except Exception as exc:
            logger.debug("§2.78 AdaptivePhaseRescheduler.reschedule() nicht blockierend: %s", exc)
            return RescheduleResult()

    # ─── Private Implementierung ────────────────────────────────────────────────

    def _reschedule_internal(
        self,
        current_goal_scores: dict[str, float],
        song_goal_targets: dict[str, float],
        selected_phases: list[str],
        executed: set[str],
        is_studio_2026: bool,
        transfer_chain: list[str],
        material_type: str,
        restorability_score: float | None = None,
        goal_confidence: dict[str, float] | None = None,
        uncertainty_budget: float | None = None,
    ) -> RescheduleResult:
        # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 70.0).
        from backend.core.calibration_context import resolve_restorability_score

        restorability_score = resolve_restorability_score(restorability_score)
        with self._op_lock:
            _injected_this_call: list[str] = []
            _gaps_triggered: dict[str, float] = {}

            # Restorability-adaptive Limits (§0k, §2.45 Minimal-Intervention)
            _max_inj = _adaptive_max_injections(restorability_score)
            _gap_thr = _adaptive_gap_threshold(restorability_score)
            _uq_budget = float(np.clip(float(uncertainty_budget), 0.0, 1.0)) if uncertainty_budget is not None else 0.35
            _chain_rarity = _estimate_chain_rarity(transfer_chain)

            _goal_conf_values: list[float] = []
            for _g in _GOAL_PRIORITY:
                if _g in _SELF_MANAGED_GOALS:
                    continue
                if float(song_goal_targets.get(_g, 0.0)) <= 0.0:
                    continue
                _goal_conf_values.append(_sanitize_confidence((goal_confidence or {}).get(_g), default=0.65))

            if _goal_conf_values:
                _avg_goal_conf = float(np.mean(_goal_conf_values))
                # Unsicherheits-Budget: bei global niedriger Konfidenz konservativer agieren.
                if _avg_goal_conf < 0.50:
                    _max_inj = max(1, _max_inj - 1)
                    _gap_thr = float(np.clip(_gap_thr * 1.15, _gap_thr, 0.45))

            # Globale Unsicherheitsdrossel (geschlossener Regelkreis):
            # hohe Unsicherheit -> weniger Injektionen, höhere Gap-Hürde.
            if _uq_budget >= 0.60:
                _max_inj = max(1, _max_inj - 1)
                _gap_thr = float(np.clip(_gap_thr * (1.0 + 0.22 * (_uq_budget - 0.60) / 0.40), _gap_thr, 0.45))
            # Rare-chain-Aufwertung: bei sehr seltenen/komplexen Chains erlauben wir
            # ein kleines zusätzliches Recovery-Fenster, um Generalisierungslücken zu schließen.
            if _chain_rarity >= 0.68 and _uq_budget <= 0.70:
                _max_inj = min(_max_inj + 1, 5)

            # §0l Material-adaptiver Gap-Threshold: analoge Träger (cassette, tape, vinyl, shellac)
            # haben stärkere physikalische Degradierung → strengere Lückenschwelle (0.90-Faktor),
            # damit Recovery-Phasen früher greifen (V38: material_type-aware Oracle).
            _ANALOG_MATERIALS: frozenset[str] = frozenset(
                {"cassette", "tape", "vinyl", "shellac", "reel_tape", "wire_recording", "wax_cylinder"}
            )
            if str(material_type).lower() in _ANALOG_MATERIALS:
                _gap_thr = float(np.clip(_gap_thr * 0.90, 0.022, _gap_thr))

            # Frühausstieg: Session-Limit bereits erreicht
            if len(self._session_injected) >= _max_inj:
                return RescheduleResult()

            # Aktueller Plan als Set für schnelle Lookups
            _plan_set: set[str] = set(selected_phases)

            # Goal-Lücken berechnen (normiert: gap = target − score)
            _goal_gaps: dict[str, float] = {}
            for goal in _GOAL_PRIORITY:
                if goal in _SELF_MANAGED_GOALS:
                    continue
                _target = float(song_goal_targets.get(goal, 0.0))
                _score = float(current_goal_scores.get(goal, 0.0))
                if _target <= 0.0:
                    continue
                _gap = float(np.clip(_target - _score, 0.0, 1.0))
                _conf = _sanitize_confidence((goal_confidence or {}).get(goal), default=0.65)
                _goal_gap_thr = _gap_thr
                # Bei niedriger Goal-Konfidenz konservativer injizieren, um
                # Fehlreaktionen auf unsichere Messwerte zu vermeiden.
                if _conf < 0.40:
                    _goal_gap_thr = float(np.clip(_gap_thr * 1.35, _gap_thr, 0.40))
                elif _conf < 0.70:
                    _goal_gap_thr = float(np.clip(_gap_thr * 1.15, _gap_thr, 0.30))

                if _gap > _goal_gap_thr:
                    _goal_gaps[goal] = _gap

            if not _goal_gaps:
                return RescheduleResult()

            # Recovery-Phasen für Goals mit signifikanten Lücken bestimmen
            _get_recovery = _get_goal_recovery_phases
            _recovery_phase_map: dict[str, list[str]] = {
                goal_name: _get_recovery(
                    goal=goal_name,
                    is_studio_2026=is_studio_2026,
                    transfer_chain=transfer_chain if transfer_chain else None,
                )
                for goal_name in _goal_gaps
            }

            # Goals nach prädiktiver Recovery-Priorität sortieren:
            # große Lücken bleiben wichtig, aber Confidence, Chain- und Material-Risiko
            # bestimmen die Reihenfolge mit. Zusätzlich wird eine Koalitions-Heuristik
            # berücksichtigt: Phasen, die mehrere offene Goals mit abdecken können,
            # werden früher angelegt.
            _sorted_goals = sorted(
                _goal_gaps.items(),
                key=lambda it: _goal_recovery_priority(
                    it[0],
                    it[1],
                    _sanitize_confidence((goal_confidence or {}).get(it[0]), default=0.65),
                    transfer_chain,
                    material_type,
                    coalition_bonus=_goal_recovery_coalition_bonus(
                        it[0],
                        _recovery_phase_map.get(it[0], [""])[0] if _recovery_phase_map.get(it[0]) else "",
                        _recovery_phase_map,
                        _goal_gaps,
                        goal_confidence,
                    ),
                ),
                reverse=True,
            )

            for _goal_name, _gap_val in _sorted_goals:
                if len(self._session_injected) + len(_injected_this_call) >= _max_inj:
                    break

                _recovery_phases = _recovery_phase_map.get(_goal_name) or _get_recovery(
                    goal=_goal_name,
                    is_studio_2026=is_studio_2026,
                    transfer_chain=transfer_chain if transfer_chain else None,
                )
                if not _recovery_phases:
                    continue

                # Nur Primär-Phase (Index 0) injizieren — §2.45 Minimal-Intervention
                _primary_phase = _recovery_phases[0]

                _risk_price = _phase_intervention_risk_price(
                    _primary_phase,
                    material_type=material_type,
                    transfer_chain=transfer_chain,
                    is_studio_2026=is_studio_2026,
                )
                _counterfactual_safe = _is_counterfactual_safe_phase(_primary_phase)
                if _counterfactual_safe:
                    _risk_price = float(np.clip(_risk_price * 0.86, 0.0, 1.0))
                _risk_weight = float(np.clip(0.55 + 0.45 * _uq_budget, 0.55, 1.0))
                _expected_utility = float(_gap_val) * (1.0 - _risk_price * _risk_weight)

                # High-risk low-utility Injektionen unter Unsicherheit vermeiden.
                if _expected_utility <= max(0.015, _gap_thr * 0.45):
                    logger.debug(
                        "§2.78 risk-pricing: %s für goal=%s übersprungen (gap=%.3f risk=%.3f Grenze=%.3f utility=%.3f)",
                        _primary_phase,
                        _goal_name,
                        _gap_val,
                        _risk_price,
                        _uq_budget,
                        _expected_utility,
                    )
                    continue

                # Guards
                if _primary_phase in executed:
                    continue  # bereits ausgeführt → kein Gewinn durch Wiederholung
                if _primary_phase in _plan_set:
                    continue  # bereits im Plan
                if _primary_phase in self._session_injected:
                    continue  # bereits in dieser Session injiziert
                if _primary_phase in _injected_this_call:
                    continue  # Guard gegen Duplikate innerhalb dieses Aufrufs
                if _primary_phase in _NEVER_SKIP:
                    continue  # _NEVER_SKIP laufen immer — Injektion unnötig
                # §0a-Guard: in Restoration keine verbotenen Phasen
                if not is_studio_2026 and _primary_phase in _RESTORATION_FORBIDDEN:
                    logger.debug(
                        "§2.78 §0a-Guard: %s nicht injiziert (Restoration verboten)",
                        _primary_phase,
                    )
                    continue

                _injected_this_call.append(_primary_phase)
                _gaps_triggered[_goal_name] = round(_gap_val, 4)
                logger.info(
                    "§2.78 Rescheduler: %s für goal=%s (gap=%.3f risk=%.2f Grenze=%.2f utility=%.3f) vorgemerkt",
                    _primary_phase,
                    _goal_name,
                    _gap_val,
                    _risk_price,
                    _uq_budget,
                    _expected_utility,
                )

            # Session-State nach erfolgreicher Berechnung aktualisieren
            for _p in _injected_this_call:
                self._session_injected.add(_p)

            return RescheduleResult(
                new_phases_to_append=_injected_this_call,
                goal_gaps_found=_gaps_triggered,
            )


# ── Singleton ──────────────────────────────────────────────────────────────────
_instance: AdaptivePhaseRescheduler | None = None
_lock = threading.Lock()


def get_adaptive_phase_rescheduler() -> AdaptivePhaseRescheduler:
    """Thread-safe Singleton-Factory für AdaptivePhaseRescheduler."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AdaptivePhaseRescheduler()
    return _instance
