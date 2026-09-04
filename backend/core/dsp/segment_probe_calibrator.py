"""§2.82 Segment-Probe-Kalibrierung — empirische Stärkenverifikation für schnelle DSP-Phasen.

Für DSP-Phasen mit Ausführungszeit < ~500 ms auf 3 s-Segmenten: Testet 3 Kandidaten-Stärken
auf einem repräsentativen 3 s-Ausschnitt, wählt die beste gegen den 15-Ziele-Teamvektor.
Nur für Phasen in SEGMENT_PROBE_ELIGIBLE_PHASES. Non-blocking: Exception → Oracle-Stärke.

Teamwork-Score: gewichtete Summe der geschlossenen Goal-Lücken.
Overprocessing-Penalty: Rückschritt bei natuerlichkeit/timbre/waerme/mikrodynamik/emotionalitaet.
Bester Kandidat = höchster (team_score − 0.5 × penalty).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SegmentProbeResult", "run_segment_probe", "SEGMENT_PROBE_ELIGIBLE_PHASES"]

# Phasen, die für Segment-Probe qualifizieren:
# - DSP-only (kein ML-Modell), schnell (< ~500 ms auf 3 s-Segment)
# - Haben einen `strength`-Parameter
# §0a-verbotene Phasen (phase_21, phase_35, phase_42) sind NICHT enthalten.
SEGMENT_PROBE_ELIGIBLE_PHASES: frozenset[str] = frozenset(
    {
        "phase_04_eq_correction",  # tonal_restoration / O3 — DSP-EQ
        "phase_05_rumble_filter",  # subtractive_cleanup — DSP-HPF/Filter
        "phase_10_compression",  # dynamics_repair / O6 — DSP-Kompressor
        "phase_11_limiting",  # dynamics_repair / O6 — DSP-Limiter
        "phase_16_final_eq",  # tonal_mastering / O3 — DSP-EQ
        "phase_18_noise_gate",  # subtractive_cleanup — DSP-Gate
        "phase_26_dynamic_range_expansion",  # dynamics_repair / O6 — DSP-Expander
        "phase_36_transient_shaper",  # transient_shaping — DSP-Transient-Shaper
        "phase_54_transparent_dynamics",  # dynamics_control / O6 — Envelope-Smoother
    }
)

# Ziele, die Überprocessing direkt anzeigen (Schutz vor Over-Processing)
_OVERPROCESSING_SENSITIVE_GOALS: frozenset[str] = frozenset(
    {
        "natuerlichkeit",
        "timbre_authentizitaet",
        "waerme",
        "mikrodynamik",
        "emotionalitaet",
    }
)

# Maximale Wall-Zeit für die gesamte Probe (alle Kandidaten zusammen)
_MAX_PROBE_WALL_TIME_S: float = 2.0
# Mindestlänge des Audio-Signals für Probe-Aktivierung
_MIN_AUDIO_DURATION_S: float = 10.0
# Länge des Probe-Segments
_PROBE_SEGMENT_S: float = 3.0

# kwargs-Keys die beim Probe-Aufruf entfernt werden (große Arrays / Callbacks)
_STRIP_KWARGS_KEYS: frozenset[str] = frozenset(
    {
        "audio",
        "audio_ref",
        "reference_audio",
        "reference",
        "noise_profile",
        "vocal_mask",
        "stems",
        "progress_sub_callback",
        "progress_callback",
        "_probe_mode",
        "skip_telemetry",
        "phase_strength_oracle_profile",
        "segment_probe_result",
    }
)


@dataclass(frozen=True)
class SegmentProbeResult:
    """Ergebnis einer Segment-Probe-Kalibrierung (§2.82)."""

    confirmed_strength: float
    """Empirisch bestätigte optimale Stärke."""
    oracle_strength: float
    """Ursprüngliche Oracle-Stärke (Ankerpunkt)."""
    best_candidate_idx: int
    """Index des besten Kandidaten in `candidates`."""
    candidates: list[float]
    """Getestete Kandidaten-Stärken."""
    team_scores: list[float]
    """Gewichteter Goal-Gap-Schließungs-Score je Kandidat."""
    overprocessing_penalty: list[float]
    """Über-Dämpfungs-Strafe je Kandidat."""
    probe_duration_s: float
    """Tatsächliche Wall-Zeit der Probe in Sekunden."""
    segment_start_s: float
    """Start des Probe-Segments im Audio in Sekunden."""
    skipped: bool = False
    """True wenn Probe übersprungen wurde."""
    skip_reason: str = ""
    """Grund für das Überspringen."""

    def to_dict(self) -> dict[str, Any]:
        """Serialisierung als flaches Dict (JSON-kompatibel)."""
        return {
            "confirmed_strength": float(self.confirmed_strength),
            "oracle_strength": float(self.oracle_strength),
            "best_candidate_idx": int(self.best_candidate_idx),
            "candidates": [float(c) for c in self.candidates],
            "team_scores": [float(s) for s in self.team_scores],
            "overprocessing_penalty": [float(p) for p in self.overprocessing_penalty],
            "probe_duration_s": float(self.probe_duration_s),
            "segment_start_s": float(self.segment_start_s),
            "skipped": bool(self.skipped),
            "skip_reason": str(self.skip_reason),
        }


def _safe_probe_kwargs(base_kwargs: dict[str, Any], strength: float) -> dict[str, Any]:
    """Filtert kwargs für sicheren isolierten Probe-Aufruf.

    Entfernt alle numpy-Arrays und bekannte Callback-/Metadaten-Keys.
    Setzt strength auf den Kandidaten-Wert.
    """
    safe: dict[str, Any] = {}
    for k, v in base_kwargs.items():
        if k in _STRIP_KWARGS_KEYS:
            continue
        if isinstance(v, np.ndarray):
            continue
        safe[k] = v
    safe["strength"] = float(strength)
    safe["_probe_mode"] = True  # Signal an Phase: kein Tracking, keine Callbacks
    return safe


def _extract_probe_segment(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Extrahiert ein repräsentatives 3 s-Segment (25 %-Marker, min. 10 s vom Start).

    Returns:
        (segment, start_time_s)
    """
    total_s = audio.shape[-1] / max(sr, 1)
    # 25 %-Marker, mindestens 10 s (Intro-Schutz), mindestens 1 s vor Segment-Ende
    start_s = max(10.0, total_s * 0.25)
    start_s = min(start_s, total_s - _PROBE_SEGMENT_S - 1.0)
    start_s = max(0.0, start_s)

    start_smp = int(start_s * sr)
    end_smp = start_smp + int(_PROBE_SEGMENT_S * sr)
    if audio.ndim == 1:
        seg = audio[start_smp:end_smp]
    else:
        seg = audio[:, start_smp:end_smp]

    # Fallback: Anfang nehmen wenn Segment zu leer
    if seg.shape[-1] < sr:
        seg = audio[..., : min(int(_PROBE_SEGMENT_S * sr), audio.shape[-1])]
        start_s = 0.0

    return np.array(seg, dtype=np.float32, copy=True), start_s


def _compute_team_score(
    pre_snap: dict[str, float],
    post_snap: dict[str, float],
    goal_gaps: dict[str, float],
    goal_weights: dict[str, float] | None,
) -> tuple[float, float]:
    """Berechnet Team-Score und Überprocessing-Strafe für einen Kandidaten.

    Team-Score: gewichtete Summe der geschlossenen Goal-Lücken (normiert auf Gap-Größe).
    Overprocessing-Penalty: Schaden an schutzwürdigen Goals (natuerlichkeit etc.).

    Returns:
        (team_score, overprocessing_penalty)
    """
    weights = goal_weights if isinstance(goal_weights, dict) else {}
    team_score = 0.0
    overprocessing_penalty = 0.0

    for goal, gap in goal_gaps.items():
        if gap <= 0.0 or not math.isfinite(gap):
            continue
        pre_val = float(pre_snap.get(goal, 0.0))
        post_val = float(post_snap.get(goal, 0.0))
        if not (math.isfinite(pre_val) and math.isfinite(post_val)):
            continue
        delta = post_val - pre_val
        w = float(weights.get(goal, 1.0))
        # Wie viel der Lücke geschlossen: +1.0 = perfekt; -1.5 = massiver Rückschritt
        gap_closed_ratio = float(np.clip(delta / max(gap, 1e-6), -1.5, 1.5))
        team_score += w * gap_closed_ratio
        # Schutzwürdige Goals: Rückschritte extra bestrafen
        if goal in _OVERPROCESSING_SENSITIVE_GOALS and delta < -0.01:
            overprocessing_penalty += w * abs(delta)

    return team_score, overprocessing_penalty


def run_segment_probe(
    *,
    audio: np.ndarray,
    sr: int,
    phase_process_fn: Callable[..., np.ndarray],
    base_kwargs: dict[str, Any],
    oracle_strength: float,
    goal_gaps: dict[str, float],
    goal_weights: dict[str, float] | None = None,
    material_key: str = "unknown",
) -> SegmentProbeResult:
    """Testet 3 Kandidaten-Stärken auf einem 3 s-Segment, gibt die beste zurück.

    Nicht-blockierend: jede Exception führt zu skipped=True, oracle_strength bleibt.
    Läuft nur bei Audio >= 10 s und Wall-Zeit <= 2 s Gesamtbudget.

    Args:
        audio: Vollständiges Eingangssignal (1D oder 2D, Samples zuletzt), float32/64.
        sr: Sample-Rate (erwartet 48000).
        phase_process_fn: Aufrufbar mit (segment, **kwargs) → np.ndarray.
        base_kwargs: Vollständige Phase-kwargs (werden kopiert, nicht mutiert).
        oracle_strength: Analytisch bestimmte Oracle-Stärke (Ankerpunkt).
        goal_gaps: Ziel-Lücken {goal_name: gap_value}.
        goal_weights: Optionale Ziel-Gewichtungen {goal_name: weight}.
        material_key: Materialschlüssel für _fast_goal_snapshot.

    Returns:
        SegmentProbeResult mit confirmed_strength als Empfehlung.
    """
    _t0 = time.monotonic()
    total_s = audio.shape[-1] / max(sr, 1)

    # Zu kurz für sinnvolle Probe
    if total_s < _MIN_AUDIO_DURATION_S:
        return SegmentProbeResult(
            confirmed_strength=oracle_strength,
            oracle_strength=oracle_strength,
            best_candidate_idx=0,
            candidates=[oracle_strength],
            team_scores=[0.0],
            overprocessing_penalty=[0.0],
            probe_duration_s=0.0,
            segment_start_s=0.0,
            skipped=True,
            skip_reason="audio_zu_kurz",
        )

    # Keine Goal-Lücken → Probe bringt keinen Informationsgewinn
    if not goal_gaps:
        return SegmentProbeResult(
            confirmed_strength=oracle_strength,
            oracle_strength=oracle_strength,
            best_candidate_idx=0,
            candidates=[oracle_strength],
            team_scores=[0.0],
            overprocessing_penalty=[0.0],
            probe_duration_s=0.0,
            segment_start_s=0.0,
            skipped=True,
            skip_reason="keine_goal_gaps",
        )

    # _fast_goal_snapshot importieren (zirkulärer Import: lazy)
    try:
        from backend.core.unified_restorer_v3 import (
            UnifiedRestorerV3 as _UV3,  # pylint: disable=import-outside-toplevel
        )

        _snapshot_fn = _UV3._fast_goal_snapshot  # pylint: disable=protected-access
    except Exception as _imp_err:
        logger.debug("§V6 UV3._fast_goal_snapshot Import fehlgeschlagen — Probe mit Oracle-Stärke zurückgegeben (skipped): %s", _imp_err)
        return SegmentProbeResult(
            confirmed_strength=oracle_strength,
            oracle_strength=oracle_strength,
            best_candidate_idx=0,
            candidates=[oracle_strength],
            team_scores=[0.0],
            overprocessing_penalty=[0.0],
            probe_duration_s=time.monotonic() - _t0,
            segment_start_s=0.0,
            skipped=True,
            skip_reason=f"import_fehler:{type(_imp_err).__name__}",
        )

    try:
        segment, start_s = _extract_probe_segment(audio, sr)
        pre_snap = _snapshot_fn(segment, sr, material_key)

        # Drei Kandidaten: 60 %, 100 %, 140 % der Oracle-Stärke (geclampt auf [0.02, 1.0])
        base = float(np.clip(oracle_strength, 0.05, 1.0))
        candidates = sorted({round(float(np.clip(base * f, 0.02, 1.0)), 4) for f in (0.60, 1.00, 1.40)})

        team_scores: list[float] = []
        overprocessing_penalties: list[float] = []

        for c_strength in candidates:
            # Zeitbudget-Check vor jedem Kandidaten
            if time.monotonic() - _t0 > _MAX_PROBE_WALL_TIME_S:
                if not team_scores:
                    return SegmentProbeResult(
                        confirmed_strength=oracle_strength,
                        oracle_strength=oracle_strength,
                        best_candidate_idx=0,
                        candidates=candidates,
                        team_scores=[],
                        overprocessing_penalty=[],
                        probe_duration_s=time.monotonic() - _t0,
                        segment_start_s=start_s,
                        skipped=True,
                        skip_reason="timeout_vor_erstem_kandidaten",
                    )
                # Bereits mindestens ein Score → mit bisherigen Ergebnissen weiterfahren
                break

            try:
                _kw = _safe_probe_kwargs(base_kwargs, c_strength)
                result_seg = phase_process_fn(segment, **_kw)
                if not isinstance(result_seg, np.ndarray):
                    result_seg = segment
                result_seg = np.nan_to_num(result_seg, nan=0.0, posinf=0.0, neginf=0.0)
                result_seg = np.clip(result_seg, -1.0, 1.0)
            except Exception as _phase_err:
                logger.debug("§2.82 Probe-Kandidat %.3f Fehler: %s", c_strength, _phase_err)
                team_scores.append(-999.0)
                overprocessing_penalties.append(999.0)
                continue

            post_snap = _snapshot_fn(result_seg, sr, material_key)
            t_score, op_pen = _compute_team_score(pre_snap, post_snap, goal_gaps, goal_weights)
            team_scores.append(t_score)
            overprocessing_penalties.append(op_pen)

        if not team_scores or all(s <= -999.0 for s in team_scores):
            return SegmentProbeResult(
                confirmed_strength=oracle_strength,
                oracle_strength=oracle_strength,
                best_candidate_idx=0,
                candidates=candidates,
                team_scores=team_scores,
                overprocessing_penalty=overprocessing_penalties,
                probe_duration_s=time.monotonic() - _t0,
                segment_start_s=start_s,
                skipped=True,
                skip_reason="alle_kandidaten_fehlgeschlagen",
            )

        # Bester Kandidat: höchster kombinierter Score (team − 0.5 × penalty)
        combined = [ts - 0.5 * op for ts, op in zip(team_scores, overprocessing_penalties)]
        # Bei Gleichstand: niedrigere Stärke bevorzugen (Minimal-Intervention §2.45)
        best_idx = int(np.argmax(combined))
        confirmed = float(candidates[best_idx])
        duration = time.monotonic() - _t0

        logger.info(
            "§2.82 SegmentProbe: oracle=%.3f → confirmed=%.3f (Δ=%+.3f) idx=%d combined=%s t=%.2fs",
            oracle_strength,
            confirmed,
            confirmed - oracle_strength,
            best_idx,
            [f"{s:.2f}" for s in combined],
            duration,
        )
        return SegmentProbeResult(
            confirmed_strength=confirmed,
            oracle_strength=oracle_strength,
            best_candidate_idx=best_idx,
            candidates=candidates,
            team_scores=team_scores,
            overprocessing_penalty=overprocessing_penalties,
            probe_duration_s=duration,
            segment_start_s=start_s,
            skipped=False,
        )

    except Exception as _exc:
        logger.debug("§2.82 SegmentProbe Exception (nicht blockierend): %s", _exc)
        return SegmentProbeResult(
            confirmed_strength=oracle_strength,
            oracle_strength=oracle_strength,
            best_candidate_idx=0,
            candidates=[oracle_strength],
            team_scores=[0.0],
            overprocessing_penalty=[0.0],
            probe_duration_s=time.monotonic() - _t0,
            segment_start_s=0.0,
            skipped=True,
            skip_reason=f"exception:{type(_exc).__name__}",
        )
