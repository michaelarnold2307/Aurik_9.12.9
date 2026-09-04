"""
§v10.15 PostProcessingGate — Selbstkalibrierung für die Post-Processing-Kette.

Analog zu PMGG (PerPhaseMusicalGoalsGate) aber für die finale Politur NACH
den 64 DSP-Phasen.  Weniger Ziele (5 statt 15), keine Retry-Schleife
(Post-Processing-Komponenten haben meist keine Stärke-Parameter), aber
dieselbe Verify→Adopt/Rollback-Logik.

Prinzip:
  1. 5-s-Stichprobe VOR der Komponente messen (5 Ziele, ≤ 80 ms)
  2. Komponente ausführen
  3. 5 Ziele NACH der Komponente messen
  4. Delta < −REGRESSION_THRESHOLD → Komponente überspringen, Original zurück
  5. Sonst → Ergebnis übernehmen

Ziele (Subset von PMGG, fokussiert auf finale Klangqualität):
  - brillanz / warmth / natreblichkeit / transparenz / spatial_depth

Ausnahme: HumanizationPass hat adaptive Stärke und wird separat behandelt.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Konstanten ──────────────────────────────────────────────────────────

# Regression threshold: wie viel Verschlechterung (0–1) wir tolerieren
# Post-Processing ist finale Politur — wir sind STRENGER als PMGG (0.025)
_REGRESSION_THRESHOLD: float = 0.015

# Post-Processing-Ziele (Subset der PMGG-Goals, DSP-only, ≤ 80 ms)
_POST_GOALS: tuple[str, ...] = (
    "brillanz",
    "warmth",
    "natreblichkeit",
    "transparenz",
    "spatial_depth",
)


# ── Gate Result ─────────────────────────────────────────────────────────


@dataclass
class PostGateResult:
    """Ergebnis einer Post-Processing-Gate-Prüfung."""

    audio: np.ndarray
    adopted: bool  # True = Komponente übernommen, False = Original zurück
    scores_before: dict[str, float] = field(default_factory=dict)
    scores_after: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    skip_reason: str = ""


# ── Gate ────────────────────────────────────────────────────────────────


class PostProcessingGate:
    """Führt eine Post-Processing-Komponente aus und verifiziert sie.

    Usage:
        gate = PostProcessingGate()
        result = gate.apply(
            component_fn=lambda audio, sr: humanize.apply(audio, sr, strength=0.12),
            audio=restored_audio,
            sr=48000,
            label="HumanizationPass",
        )
        restored_audio = result.audio  # entweder verarbeitet oder original
    """

    def __init__(self) -> None:
        self._total_skipped: int = 0
        self._total_adopted: int = 0
        self._total_checks: int = 0

    # ── §v10.16 Binäre Suche für PostGate-Stärke ──────────────────────

    def _binary_search_strength(
        self,
        component_fn: Callable[[np.ndarray, int, float | None], np.ndarray],
        audio: np.ndarray,
        sr: int,
        label: str,
        goals: tuple[str, ...],
        threshold: float,
        initial: float = 1.0,
        iterations: int = 12,
        precision: float = 0.00025,
    ) -> float:
        """Binäre Suche für optimale PostGate-Stärke (12 Iter., ±0.025%).

        Algorithmus:
        1. Starte mit initialer Stärke (default 1.0).
        2. Bisektion: lo=0.0, hi=initial.
        3. Pro Iteration: mid = (lo+hi)/2. Teste mid via component_fn.
        4. Wenn Regression → hi = mid (Stärke zu hoch).
           Wenn keine Regression → lo = mid (könnte höher gehen).
        5. Nach 12 Iterationen: lo ist die maximale regressionsfreie Stärke.

        Returns optimal strength value (float).
        """
        _lo, _hi = 0.0, float(initial)
        _best = 0.0

        for _i in range(iterations):
            _mid = (_lo + _hi) / 2.0
            if _hi - _lo < precision:
                break
            try:
                _processed = component_fn(audio, sr, _mid)
                if _processed.shape != audio.shape:
                    _hi = _mid
                    continue
                _scores_after = self._measure(_processed, sr, goals)
                _regression = any(
                    _scores_after.get(g, 0.5) - self._measure(audio, sr, (g,)).get(g, 0.5) < -threshold for g in goals
                )
                if _regression:
                    _hi = _mid
                else:
                    _lo = _mid
                    _best = _mid
            except Exception:
                _hi = _mid

        logger.debug(
            "PostGate [%s] binary search: optimal=%.4f (iter=%d, range=[%.4f,%.4f])",
            label,
            _best,
            iterations,
            _lo,
            _hi,
        )
        return float(_best)

    # ── Öffentliche API ────────────────────────────────────────────────

    def apply(
        self,
        component_fn: Callable[[np.ndarray, int, float | None], np.ndarray],
        audio: np.ndarray,
        sr: int,
        label: str = "post",
        goals: tuple[str, ...] = _POST_GOALS,
        threshold: float | None = None,
        *,
        strength: float | None = None,
        binary_search_strength: bool = False,
    ) -> PostGateResult:
        """Führt *component_fn* aus und verifiziert das Ergebnis.

        Args:
            component_fn: Funktion (audio, sr, strength=None) → processed_audio
            strength: Wenn gesetzt → an component_fn übergeben
            binary_search_strength: Wenn True → binäre Suche für optimale Stärke (12 Iter.)
            audio: Input float32 stereo
            sr: 48000 Hz
            label: Komponentenname für Logging
            goals: Zu prüfende Ziele (default: _POST_GOALS)
            threshold: Überschreibt _REGRESSION_THRESHOLD

        Returns:
            PostGateResult mit .audio (entweder processed oder original)
        """
        t0 = time.time()
        self._total_checks += 1
        _thresh = threshold if threshold is not None else _REGRESSION_THRESHOLD

        # §v10.16: Binäre Suche für optimale Stärke
        if binary_search_strength and strength is not None:
            _optimal = self._binary_search_strength(
                component_fn,
                audio,
                sr,
                label,
                goals,
                _thresh,
                initial=float(strength),
            )
            if _optimal > 0.001:
                strength = _optimal
                logger.debug("PostGate [%s]: binäre Suche → optimal=%.5f", label, strength)
        # akzeptieren. Fängt 2-arg-Lambdas wie ``lambda a, sr: ...`` sofort ab,
        # statt zur Laufzeit mit kryptischem TypeError zu crashen.
        PostProcessingGate._validate_lambda(label, component_fn)

        # ── 1. Messen VORHER ──────────────────────────────────────────
        scores_before = self._measure(audio, sr, goals)

        # ── 2. Komponente ausführen ───────────────────────────────────
        try:
            processed = component_fn(audio, sr, strength)
        except Exception as exc:
            logger.warning(
                "PostGate [%s]: Komponente fehlgeschlagen (%s) — Originalsignal zurück",
                label,
                exc,
            )
            self._total_skipped += 1
            return PostGateResult(
                audio=audio,
                adopted=False,
                scores_before=scores_before,
                skip_reason=f"exception: {exc}",
                duration_ms=(time.time() - t0) * 1000.0,
            )

        # Sanity: processed muss gleiche Shape haben
        if processed.shape != audio.shape:
            logger.warning(
                "PostGate [%s]: Shape-Änderung %s → %s — Originalsignal zurück",
                label,
                audio.shape,
                processed.shape,
            )
            self._total_skipped += 1
            return PostGateResult(
                audio=audio,
                adopted=False,
                scores_before=scores_before,
                skip_reason=f"shape mismatch: {audio.shape} → {processed.shape}",
                duration_ms=(time.time() - t0) * 1000.0,
            )

        # ── 3. Messen NACHHER ─────────────────────────────────────────
        scores_after = self._measure(processed, sr, goals)

        # ── 4. Delta-Check ────────────────────────────────────────────
        deltas: dict[str, float] = {}
        regressions: list[str] = []
        for goal in goals:
            before = scores_before.get(goal, 0.5)
            after = scores_after.get(goal, 0.5)
            delta = after - before
            deltas[goal] = delta
            if delta < -_thresh:
                regressions.append(f"{goal}({delta:+.3f})")

        # ── 5. Entscheidung ───────────────────────────────────────────
        if regressions:
            logger.info(
                "PostGate [%s]: Regression in %d/%d Zielen (%s) — Komponente ÜBERSPRUNGEN",
                label,
                len(regressions),
                len(goals),
                ", ".join(regressions[:3]),
            )
            self._total_skipped += 1
            return PostGateResult(
                audio=audio,
                adopted=False,
                scores_before=scores_before,
                scores_after=scores_after,
                deltas=deltas,
                skip_reason=f"regression: {regressions}",
                duration_ms=(time.time() - t0) * 1000.0,
            )

        logger.debug(
            "PostGate [%s]: Alle %d Ziele OK — Komponente übernommen (%.1f ms)",
            label,
            len(goals),
            (time.time() - t0) * 1000.0,
        )
        self._total_adopted += 1
        return PostGateResult(
            audio=processed,
            adopted=True,
            scores_before=scores_before,
            scores_after=scores_after,
            deltas=deltas,
            duration_ms=(time.time() - t0) * 1000.0,
        )

    # ── Interne Messung ───────────────────────────────────────────────

    @staticmethod
    def _measure(audio: np.ndarray, sr: int, goals: tuple[str, ...]) -> dict[str, float]:
        """Misst die angegebenen Ziele auf einer 5-s-Stichprobe.

        Verwendet PMGGs _measure_quick (15 Ziele, DSP-only) und filtert
        auf die gewünschten Ziele.
        """
        try:
            from backend.core.per_phase_musical_goals_gate import _measure_quick

            full_scores = _measure_quick(audio, sr, precise_override=True, enable_vocal_guard=False)
            return {g: full_scores.get(g, 0.5) for g in goals}
        except Exception as e:
            logger.warning("§V6 _measure fehlgeschlagen — alle Goals auf 0.5 gesetzt (Gate passiert immer): %s", e)
            return dict.fromkeys(goals, 0.5)

    # ── Signatur-Validierung ────────────────────────────────────────

    @staticmethod
    def _validate_lambda(label: str, component_fn: Callable) -> None:
        """§v10.0.5: Prüft dass component_fn mindestens 3 positional args akzeptiert.

        Verhindert, dass 2-arg-Lambdas (``lambda a, sr: ...``) zur Laufzeit
        mit ``TypeError: takes 2 positional arguments but 3 were given`` crashen.
        """
        try:
            sig = inspect.signature(component_fn)
            params = list(sig.parameters.values())
            required = sum(
                1
                for p in params
                if p.default is inspect.Parameter.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            positional = sum(
                1
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            has_catch_all = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL or p.kind == inspect.Parameter.VAR_KEYWORD for p in params
            )
            # Erlaubt: 3+ positional (mit/ohne default) OR catch-all
            if not has_catch_all and positional < 3:
                raise AssertionError(
                    f"PostGate [{label}]: component_fn hat nur {positional} positional"
                    f" args, braucht aber 3 (audio, sr, strength)."
                    f" Signatur={sig}"
                )
        except (ValueError, TypeError):
            # C-builtins oder inspect-inkompatible Callables —
            # können wir nicht prüfen, also durchlassen.
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # ── Statistik ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_checks": self._total_checks,
            "total_adopted": self._total_adopted,
            "total_skipped": self._total_skipped,
        }


# ── Singleton ─────────────────────────────────────────────────────────

import threading as _threading_singleton

_instance: PostProcessingGate | None = None
_lock = _threading_singleton.Lock()


def get_post_processing_gate() -> PostProcessingGate:
    """Gibt die process-weite PostProcessingGate-Singleton zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PostProcessingGate()
    return _instance
