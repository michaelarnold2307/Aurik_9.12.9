"""§G76 (GEBOTE.md) Zentraler Kalibrierungs-Kontext — Single Source of Truth.

CALIBRATION_CONTEXT ist die EINZIGE Quelle für alle kalibrierten Schwellwerte,
Caps, Floors und Blend-Faktoren in Aurik.

JEDES Modul, das einen Parameter aus Pre-Analysis-Messwerten benötigt, MUSS
diesen Kontext abrufen — NIE eine eigene Konstante pflegen, NIE einen
stillschweigenden Default verwenden.

§G86 Maschinelle Durchsetzung:
  - Der Sentinel _UNSET verhindert stillschweigende Defaults.
  - test_verbotene_defaults_linter.py scannt auf `= 1` Defaults.
  - test_cross_depth_validation.py validiert ALLE depth-Stufen.
"""

from __future__ import annotations

__all__ = [
    "CalibrationContext",
    "UNSET",
    "require_calibration_context",
    "get_calibration_context",
    "set_calibration_context",
]

import threading
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Sentinel: verhindert stillschweigende Defaults (§G86)
# ═══════════════════════════════════════════════════════════════════════════════


class _UnsetType:
    """Sentinel-Wert für nicht gesetzte Kalibrierungs-Parameter.

    Jeder Parameter, der aus dem CalibrationContext stammen MUSS,
    verwendet UNSET als Default. Bei Zugriff auf einen UNSET-Wert
    wird eine laute Exception geworfen — kein stiller Fallback.
    """

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False

    def __int__(self) -> int:
        raise RuntimeError(
            "UNSET CalibrationContext-Wert wurde als int verwendet. "
            "Der CalibrationContext wurde nicht korrekt injiziert. "
            "§G76 (GEBOTE.md): Jedes Modul MUSS den CalibrationContext explizit erhalten."
        )

    def __float__(self) -> float:
        raise RuntimeError(
            "UNSET CalibrationContext-Wert wurde als float verwendet. "
            "Der CalibrationContext wurde nicht korrekt injiziert. "
            "§G76 (GEBOTE.md): Jedes Modul MUSS den CalibrationContext explizit erhalten."
        )

    def __eq__(self, other: object) -> bool:
        return other is self

    def __hash__(self) -> int:
        return id(self)


UNSET: Any = _UnsetType()
"""Sentinel: Verwende UNSET statt `= 1` für Kalibrierungs-Parameter."""


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationContext — die zentrale Kalibrierungs-Datenstruktur
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True, frozen=True)
class CalibrationContext:
    """§G76 (GEBOTE.md) Zentraler Kalibrierungs-Kontext.

    Bündelt ALLE Pre-Analysis-Messwerte in EINEM Objekt.
    JEDES Modul, das einen Schwellwert benötigt, MUSS diesen
    Kontext als Quelle verwenden.

    Felder (alle immutable — nach der Pre-Analysis fixiert):
        restorability_score: 0–100 (RestorabilityEstimator)
        transfer_chain_depth: Anzahl Transfer-Stufen (1 = Studio-Master)
        material_type: Tonträger-Typ (vinyl, cassette, shellac, …)
        snr_db: Geschätztes Signal-Rausch-Verhältnis in dB
        bandwidth_hz: Effektive Bandbreite in Hz
        era_decade: Geschätzte Aufnahme-Dekade (1920–2020)
        genre: Geschätztes Genre
        vocal_confidence: PANNs Gesangs-Konfidenz (0–1)

    Verwendung:
        ctx = CalibrationContext(
            restorability_score=64.0,
            transfer_chain_depth=4,
            material_type="cassette",
        )
        threshold = compute_threshold(ctx)  # NICHT: compute_threshold(rs=64, depth=1)
    """

    # Kern-Messwerte (MÜSSEN gesetzt sein)
    restorability_score: float
    transfer_chain_depth: int
    material_type: str

    # Optionale Messwerte (mit Defaults, die physikalisch neutral sind)
    snr_db: float = 30.0
    bandwidth_hz: float = 20000.0
    era_decade: int = 1980
    genre: str = "unknown"
    vocal_confidence: float = 0.0

    # Abgeleitete Werte (werden im __post_init__ berechnet)
    _depth: int = field(init=False, repr=False, default=1)
    _is_analog: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Validiert und leitet Hilfswerte ab."""
        # Validierung
        if self.transfer_chain_depth is UNSET:
            raise ValueError(
                "CalibrationContext.transfer_chain_depth ist UNSET. "
                "§G76 (GEBOTE.md): transfer_chain_depth MUSS aus MediumDetector stammen."
            )
        if self.restorability_score is UNSET:
            raise ValueError(
                "CalibrationContext.restorability_score ist UNSET. "
                "§G76 (GEBOTE.md): restorability_score MUSS aus RestorabilityEstimator stammen."
            )

        # Abgeleitete Werte (via object.__setattr__ wegen frozen=True)
        _depth = max(1, int(self.transfer_chain_depth))
        object.__setattr__(self, "_depth", _depth)

        _ANALOG_MATERIALS = frozenset(
            {
                "vinyl",
                "shellac",
                "tape",
                "cassette",
                "reel_tape",
                "wax_cylinder",
                "wire_recording",
                "lacquer_disc",
            }
        )
        _is_analog = str(self.material_type).lower() in _ANALOG_MATERIALS
        object.__setattr__(self, "_is_analog", _is_analog)

    # ── Convenience-Properties ──────────────────────────────────────────

    @property
    def depth(self) -> int:
        """Transfer-Chain-Tiefe (1–n). Garantiert ≥ 1."""
        return self._depth

    @property
    def is_analog(self) -> bool:
        """True wenn physischer Analog-Träger (Vinyl, Tape, Shellac, …)."""
        return self._is_analog

    @property
    def is_deep_chain(self) -> bool:
        """True bei 4+ Transfer-Stufen (Kassette, extreme Ketten)."""
        return self._depth >= 4

    @property
    def chain_factor(self) -> float:
        """§v10.120 Multiplikativer Chain-Faktor für GDD/Regression.

        depth=1–2: 1.00  (kein Boost)
        depth=3:   1.25
        depth=4:   1.50
        depth=5+:  1.50 + 0.25×(depth−4)
        """
        return float(np.clip(1.0 + max(0, self._depth - 2) * 0.25, 1.0, 10.0))

    @property
    def artifact_freedom_min(self) -> float:
        """§v10.119 Depth-adaptiver artifact_freedom-Mindestwert.

        depth=1 (Studio):   0.95
        depth=2 (Vinyl):    0.88
        depth=3 (Shellac):  0.80
        depth≥4 (Kassette): 0.70
        """
        if self._depth >= 4:
            return 0.70
        elif self._depth == 3:
            return 0.80
        elif self._depth == 2:
            return 0.88
        return 0.95

    @property
    def regression_threshold_base(self) -> float:
        """§2.29/§2.54 Material- und Restorability-adaptiver Basis-Schwellwert."""
        _rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        if _rs >= 70.0:
            base = 0.035  # REGRESSION_THRESHOLD_GOOD
        elif _rs >= 40.0:
            base = 0.050  # REGRESSION_THRESHOLD_FAIR
        else:
            base = 0.065  # REGRESSION_THRESHOLD_POOR
        # Material-Bonus
        _ANALOG_BONUS = {
            "vinyl": 0.005,
            "shellac": 0.008,
            "tape": 0.004,
            "cassette": 0.006,
            "reel_tape": 0.004,
            "wax_cylinder": 0.010,
            "wire_recording": 0.006,
        }
        bonus = _ANALOG_BONUS.get(str(self.material_type).lower(), 0.003)
        return float(np.clip(base + bonus, 0.012, 0.070))

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert für Logging/Telemetrie."""
        return {
            "restorability_score": round(self.restorability_score, 1),
            "transfer_chain_depth": self._depth,
            "material_type": self.material_type,
            "snr_db": round(self.snr_db, 1),
            "bandwidth_hz": round(self.bandwidth_hz, 0),
            "era_decade": self.era_decade,
            "genre": self.genre,
            "vocal_confidence": round(self.vocal_confidence, 3),
        }

    # ── Convenience: CIG State befüllen ──────────────────────────────────

    def apply_to_cig_state(self, state: Any) -> None:
        """Überträgt Kalibrierungswerte auf einen InteractionGuardState.

        Usage:
            ctx = require_calibration_context()
            ctx.apply_to_cig_state(state)
        """
        state.transfer_chain_depth = self._depth
        state.restorability_score = float(np.clip(self.restorability_score, 0.0, 100.0))
        state.material_type = str(self.material_type)

    # ── Convenience: Threshold-Factories ─────────────────────────────────

    def gdd_threshold(self, phase_id: str) -> float:
        """Berechnet die adaptive GDD-Schwelle für eine Phase.

        Nutzt CIG's _compute_gdd_threshold, befüllt den State aus diesem Context.
        """
        from backend.core.cumulative_interaction_guard import (
            CumulativeInteractionGuard,
            InteractionGuardState,
        )

        guard = CumulativeInteractionGuard()
        state = InteractionGuardState()
        self.apply_to_cig_state(state)
        return guard._compute_gdd_threshold(phase_id, state)

    def regression_threshold(self) -> float:
        """§2.29/§2.54 Material- und Restorability-adaptiver REGRESSION_THRESHOLD."""
        from backend.core.per_phase_musical_goals_gate import _get_adaptive_threshold

        return _get_adaptive_threshold(self.restorability_score, str(self.material_type), self._depth)


# ═══════════════════════════════════════════════════════════════════════════════
# Thread-lokaler CalibrationContext für Modul-übergreifenden Zugriff
# ═══════════════════════════════════════════════════════════════════════════════

_calibration_context: threading.local = threading.local()


def set_calibration_context(ctx: CalibrationContext) -> None:
    """Setzt den CalibrationContext für den aktuellen Thread.

    Wird EINMAL pro Pipeline-Lauf aufgerufen, VOR der ersten Phase.
    """
    _calibration_context.value = ctx


def reset_calibration_context() -> None:
    """Setzt den thread-lokalen CalibrationContext zurück.

    §G1 (copilot-instructions.md)/V8 Song-Isolation: Aufruf am Song-Ende und
    in Test-Teardowns — verhindert, dass ein Pipeline-Lauf (oder Test) den
    Kontext des nächsten Songs/Tests kontaminiert.
    """
    if hasattr(_calibration_context, "value"):
        del _calibration_context.value


def get_calibration_context() -> CalibrationContext | None:
    """Gibt den aktuellen CalibrationContext zurück (oder None)."""
    return getattr(_calibration_context, "value", None)


def resolve_restorability_score(
    restorability_score: float | None = None,
    *,
    default: float = 70.0,
) -> float:
    """Kanonische rs-Quelle (§G76 (GEBOTE.md), §v10.x).

    Befund 2026-08-22 (rs-Inkonsistenz): CALIB-Pfade hatten verstreute
    Hardcode-Defaults (70.0 / 65.0 / 64.0) — Module, die den kalibrierten
    Wert nicht explizit bekamen, arbeiteten still mit 70.0 weiter, während
    andere denselben Song mit dem echten Estimator-Wert (z. B. 64) verarbeiteten.
    Ergebnis: inkonsistente Schwellwerte innerhalb eines Laufs.

    Auflösungs-Reihenfolge:
        1. Explizit übergebener Wert (höchste Autorität).
        2. Thread-lokaler CalibrationContext (von der Pre-Analysis gesetzt).
        3. Modul-Default (konservativ, nur für Tests ohne Pipeline-Kontext).
    """
    if restorability_score is not None:
        return float(np.clip(restorability_score, 0.0, 100.0))
    _ctx = get_calibration_context()
    if _ctx is not None and _ctx.restorability_score is not UNSET:
        return float(np.clip(_ctx.restorability_score, 0.0, 100.0))
    return float(np.clip(default, 0.0, 100.0))


def require_calibration_context() -> CalibrationContext:
    """Gibt den CalibrationContext zurück oder wirft eine laute Exception.

    Verwendung in Modulen, die OHNE CalibrationContext nicht sinnvoll
    arbeiten können:
        ctx = require_calibration_context()
        threshold = compute_threshold(ctx)

    Raises:
        RuntimeError: Wenn kein CalibrationContext gesetzt wurde.
    """
    ctx = get_calibration_context()
    if ctx is None:
        raise RuntimeError(
            "CalibrationContext ist nicht gesetzt. "
            "§G76 (GEBOTE.md): set_calibration_context() MUSS vor Pipeline-Start aufgerufen werden. "
            "Module, die require_calibration_context() verwenden, "
            "dürfen NUR innerhalb einer laufenden Pipeline aufgerufen werden."
        )
    return ctx


# Lazy import für numpy (vermeidet Zirkelschluss)
import numpy as np
