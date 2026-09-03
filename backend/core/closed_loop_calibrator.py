"""backend/core/closed_loop_calibrator.py — §v10.600 Closed-Loop Strength Regelkreis.

Regelkreis-Architektur:
  1. Vor Phase N: strength aus Regelkreis-State
  2. Phase N ausführen
  3. Nach Phase N: quality_delta messen (pre vs post audio)
  4. Regelkreis entscheidet: stärker / schwächer / halten
  5. Strength für Phase N+1 anpassen

Status: ✅ Produktion (§v10.600 UnifiedRestorerV3._execute_pipeline Integration)
"""

from __future__ import annotations

import logging

import numpy as np

from backend.core.calibration_context import get_calibration_context

logger = logging.getLogger(__name__)


def measure_phase_quality_delta(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    is_repair: bool = False,
) -> float:
    """Misst das Qualitäts-Delta einer Phase.

    Kombiniert Crest-Änderung, RMS-Stabilität und spektrale Korrelation
    zu einem einzigen Wert. Positiv = Verbesserung, negativ = Verschlechterung.

    §v10.650 W5: Repair-Phasen erhalten einen Korrelations-Bonus von +0.08.
    Reparatur ersetzt PLANGEMÄSS defekte durch neue Samples — die niedrige
    Korrelation ist ERWARTET und kein Qualitätsverlust.

    Returns:
        float in [-1.0, 1.0]: >0.05 = klare Verbesserung, ~0 = neutral, <−0.05 = Regression
    """
    try:
        pre = np.asarray(audio_before, dtype=np.float32).ravel()
        post = np.asarray(audio_after, dtype=np.float32).ravel()
        n = min(len(pre), len(post))
        if n < 256:
            return 0.0
        pre = pre[:n]
        post = post[:n]

        # 1. Crest-Stabilität (30% Gewicht)
        pre_rms = float(np.sqrt(np.mean(pre**2))) + 1e-12
        post_rms = float(np.sqrt(np.mean(post**2))) + 1e-12
        pre_peak = float(np.max(np.abs(pre))) + 1e-12
        post_peak = float(np.max(np.abs(post))) + 1e-12
        pre_crest = float(20.0 * np.log10(pre_peak / pre_rms))
        post_crest = float(20.0 * np.log10(post_peak / post_rms))
        crest_delta = post_crest - pre_crest
        crest_score = float(np.clip(1.0 + crest_delta / 6.0, 0.0, 1.0))

        # 2. RMS-Stabilität (30% Gewicht)
        rms_ratio = min(post_rms, pre_rms) / max(post_rms, pre_rms)
        rms_score = float(rms_ratio)

        # 3. Korrelation (40% Gewicht)
        step = max(1, n // 8192)
        pre_ds = pre[::step]
        post_ds = post[::step]
        corr = float(np.corrcoef(pre_ds, post_ds)[0, 1]) if len(pre_ds) > 2 else 1.0
        corr = max(0.0, min(1.0, corr)) if not np.isnan(corr) else 1.0

        quality = 0.30 * crest_score + 0.30 * rms_score + 0.40 * corr
        delta = quality - 0.95  # 0.95 = Referenz „keine Änderung"

        # §v10.650 W5: Repair-Bonus — Dropout-Füllung/Inpainting ersetzt
        # defekte Samples → niedrige Korrelation ist ERWARTET, nicht Bestrafung.
        # Bonus begrenzt auf max 0.0, so dass neutrale Reparatur als "kein Effekt"
        # (Δ≈0.00) statt als "Regression" (Δ≈−0.08) gewertet wird.
        if is_repair:
            delta = min(delta + 0.08, 0.0)

        return float(np.clip(delta, -1.0, 1.0))
    except Exception:
        logger.warning("§V6 ML→DSP-Fallback: measure_phase_quality_delta fehlgeschlagen → neutraler Return (0.0)")
        return 0.0


def closed_loop_calibrate(
    state: ClosedLoopState,
    phase_id: str,
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    strength_used: float,
) -> float:
    """Regelkreis-Entscheidung nach einer Phase: Strength anpassen?

    Misst quality_delta, entscheidet über Anpassung, aktualisiert den State.

    Returns:
        float: Empfohlene Strength-Änderung (0 = keine Änderung)
    """
    # 1. Klassifiziere Phase (wird für Messung und Adaption benötigt)
    _is_repair = any(
        kw in phase_id.lower()
        for kw in (
            "click",
            "crackle",
            "hum",
            "dropout",
            "declip",
            "repair",
            "spectral_repair",
            "inpainting",
            "wow",
            "flutter",
        )
    )

    # 2. Messe Qualitäts-Delta (mit Repair-Bonus §v10.650 W5)
    _delta = measure_phase_quality_delta(audio_before, audio_after, is_repair=_is_repair)

    # 3. Regelkreis-Entscheidung
    _current = float(np.clip(strength_used, 0.0, 1.0))

    # §v10.702 R3: Empfindlichere Schwellwerte — erkennen subtile Verbesserungen
    # Vorher: IMPROVEMENT_THRESHOLD=0.04, REGRESSION_THRESHOLD=-0.06
    # 43/44 Phasen liefen als "kein Effekt" → kumulativer Schaden unerkannt
    REGRESSION_THRESHOLD = -0.03  # vorher -0.06 — Regressionen früher erkennen
    IMPROVEMENT_THRESHOLD = 0.015  # vorher 0.04 — subtile Verbesserungen erfassen
    STRONG_IMPROVEMENT = 0.08  # vorher 0.12 — realistischere Schwelle
    ADAPT_STEP = 0.04  # vorher 0.08 — feinere Anpassung
    REPAIR_ADAPT_STEP = 0.08  # vorher 0.12 — Repair-Phasen behalten gröberen Schritt

    _step = REPAIR_ADAPT_STEP if _is_repair else ADAPT_STEP
    _min_str = max(0.05, state.current_strength * 0.3)
    _max_str = min(1.0, state.current_strength * 2.5)

    # §v10.702 R3: Kumulativen Δ-Tracker aktualisieren
    state.cumulative_delta += _delta
    state.cumulative_delta_count += 1

    # ── Entscheidungsmatrix ────────────────────────────────────────
    _new = _current
    _decision = "hold"
    _reason = ""
    _confidence = 0.5

    # §v10.702 R3: Kumulativer Boost — nach 5 Phasen mit Δ>0.01
    # signalisiert der kumulative Trend eine echte Verbesserung,
    # auch wenn jede einzelne Phase unter der Schwelle liegt.
    _cumulative_boost = False
    if state.cumulative_delta_count >= 5 and state.cumulative_delta > 0.05:
        _cumulative_boost = True
        state.cumulative_delta = 0.0  # Reset nach Boost
        state.cumulative_delta_count = 0

    if _delta < REGRESSION_THRESHOLD:
        _new = max(_min_str, _current - _step * 1.5)
        _decision = "decrease"
        _reason = f"Regression Δ={_delta:+.3f}: reduziere {_current:.3f}→{_new:.3f}"
        _confidence = min(0.95, abs(_delta) * 4.0)
        state.consecutive_regressions += 1
        state.consecutive_improvements = 0
        state.consecutive_no_effect = 0
    elif _delta > STRONG_IMPROVEMENT or _cumulative_boost:
        if _cumulative_boost and _delta <= STRONG_IMPROVEMENT:
            _boost_reason = f"kumulativ Δ={state.cumulative_delta:+.3f} über {state.cumulative_delta_count} Phasen"
        else:
            _boost_reason = f"Δ={_delta:+.3f}"
        _new = min(_max_str, _current + _step * 0.5)
        _decision = "increase"
        _reason = f"Verbesserung ({_boost_reason}): erhöhe {_current:.3f}→{_new:.3f}"
        _confidence = 0.75
        state.consecutive_improvements += 1
        state.consecutive_regressions = 0
        state.consecutive_no_effect = 0
    elif _delta > IMPROVEMENT_THRESHOLD:
        _new = _current
        _decision = "hold"
        _reason = f"Verbesserung Δ={_delta:+.3f}: halte {_current:.3f}"
        _confidence = 0.85
        state.consecutive_improvements += 1
        state.consecutive_regressions = 0
        state.consecutive_no_effect = 0
    elif _delta < -0.015:  # §v10.702 R3: engere Schwelle für leichte Verschlechterung
        _new = max(_min_str, _current - _step * 0.5)
        _decision = "decrease"
        _reason = f"Leichte Verschlechterung Δ={_delta:+.3f}: reduziere {_current:.3f}→{_new:.3f}"
        _confidence = 0.55
        state.consecutive_regressions += 1
        state.consecutive_improvements = 0
        state.consecutive_no_effect = 0
    elif abs(_delta) < 0.01:
        # Kein messbarer Effekt
        state.consecutive_no_effect += 1
        if state.consecutive_no_effect >= 3:
            _new = 0.0
            _decision = "skip"
            _reason = f"3× kein Effekt: Phase {phase_id} wird übersprungen"
            _confidence = 0.90
            state.skip_count += 1
        else:
            _new = _current
            _decision = "hold"
            _reason = f"Kein Effekt ({state.consecutive_no_effect}/3): halte {_current:.3f}"
            _confidence = 0.60
        state.consecutive_improvements = 0
        state.consecutive_regressions = 0
    else:
        _new = _current
        _decision = "hold"
        _reason = f"Neutral Δ={_delta:+.3f}: halte {_current:.3f}"
        _confidence = 0.70

    # 3. Aktualisiere State
    _adjust = _new - _current
    state.current_strength = round(float(np.clip(_new, _min_str, _max_str)), 3)
    state.state = _decision

    # 4. Protokolliere
    state.record_phase(phase_id, _delta, _decision, state.current_strength, _reason)

    logger.info(
        "§v10.600 ClosedLoop %s: Δ=%+.4f %s→%s strength=%.3f decision=%s (%s)",
        phase_id,
        _delta,
        _current,
        state.current_strength,
        state.current_strength,
        _decision,
        _reason[:60],
    )

    return round(_adjust, 3)


class ClosedLoopState:
    """Hält den Zustand des Closed-Loop-Regelkreises über Phasen hinweg.

    Thread-sicher: eine Instanz pro Pipeline-Lauf (nicht global).
    §v10.600: Integriert in UnifiedRestorerV3._execute_pipeline.
    """

    def __init__(
        self,
        restorability_score: float = 65.0,
        transfer_chain_depth: int | None = None,
        material: str = "vinyl",
    ):
        # Initial-Konfiguration aus Pre-Analysis
        self.restorability_score = float(np.clip(restorability_score, 10.0, 100.0))
        if transfer_chain_depth is None:
            _ctx = get_calibration_context()
            transfer_chain_depth = _ctx.transfer_chain_depth if _ctx is not None else 1
        self.transfer_chain_depth = max(1, int(transfer_chain_depth))
        self.material = str(material or "vinyl")

        # Aktuelle Strength (startet mit joint_calibrator-kompatiblem Wert)
        self.current_strength = self._compute_initial_strength()
        self._last_strength = self.current_strength

        # Regelkreis-Status
        self.state = ClosedLoopState._State.HOLD
        self.consecutive_improvements: int = 0
        self.consecutive_no_effect: int = 0
        self.consecutive_regressions: int = 0
        self.total_phases_measured: int = 0
        self.cumulative_improvement: float = 0.0
        self.regression_count: int = 0
        self.skip_count: int = 0
        self.cumulative_delta: float = 0.0  # §v10.702 R3: kumulativer Δ-Tracker
        self.cumulative_delta_count: int = 0  # §v10.702 R3: Zähler für kumulativen Boost
        self.phase_history: list[dict] = []

    class _State:
        HOLD = "hold"
        STRONGER = "stronger"
        WEAKER = "weaker"
        SKIP = "skip"

    def _compute_initial_strength(self) -> float:
        """Berechnet die initiale Strength aus Restorability (wie joint_calibrator)."""
        rs = self.restorability_score
        if rs >= 90:
            s = 0.20
        elif rs >= 60:
            s = 0.35
        elif rs >= 30:
            s = 0.40
        else:
            s = 0.45
        # Chain depth boost
        depth_boost = 1.0 + (self.transfer_chain_depth - 1) * 0.08
        return round(float(np.clip(s * depth_boost, 0.10, 0.60)), 3)

    def record_phase(
        self,
        phase_id: str,
        quality_delta: float,
        decision: str,
        next_strength: float,
        reason: str,
    ) -> None:
        """Protokolliert eine Phasen-Entscheidung im Regelkreis."""
        self.total_phases_measured += 1
        self.last_quality_delta = quality_delta
        self.cumulative_improvement += max(0.0, quality_delta)
        if quality_delta < -0.06:
            self.regression_count += 1
        if decision == "skip":
            self.skip_count += 1

        self.phase_history.append(
            {
                "phase": phase_id,
                "delta": round(quality_delta, 4),
                "decision": decision,
                "next_strength": round(next_strength, 4),
                "reason": reason,
            }
        )

        # Nur letzte 20 Phasen behalten
        if len(self.phase_history) > 20:
            self.phase_history = self.phase_history[-20:]

    def summary(self) -> dict:
        """Gibt eine Zusammenfassung des Regelkreis-Zustands."""
        return {
            "phases_measured": self.total_phases_measured,
            "cumulative_improvement": round(self.cumulative_improvement, 4),
            "regression_count": self.regression_count,
            "skip_count": self.skip_count,
            "last_delta": round(self.last_quality_delta, 4),
            "health": (
                "excellent"
                if self.regression_count == 0 and self.cumulative_improvement > 0.5
                else "good"
                if self.regression_count <= 2
                else "unstable"
                if self.regression_count <= 5
                else "critical"
            ),
        }
