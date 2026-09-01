"""§v10 Shared output guard helpers — HPE-zentriert.

Nicht nur RMS und Stereo prüfen, sondern auch:
„Klingt das Ergebnis für menschliche Ohren angenehmer als das Original?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputGuardDecision:
    """Decision payload used by high-quality output guards."""

    fallback: bool
    reason: str
    rms_delta_db: float
    stereo_side_ratio: float
    pleasantness_delta: float = 0.0  # §v10 HPE


def rms(audio: np.ndarray) -> float:
    """Gibt RMS for mono/stereo audio zurück."""
    x = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def side_rms(audio: np.ndarray) -> float:
    """Gibt side-channel RMS for stereo audio (handles both (N,2) and (2,N)) zurück."""
    if audio.ndim != 2:
        return 0.0
    if audio.shape[1] == 2 and audio.shape[0] > 2:
        side = 0.5 * (audio[:, 0].astype(np.float32) - audio[:, 1].astype(np.float32))
    elif audio.shape[0] == 2 and audio.shape[1] > 2:
        side = 0.5 * (audio[0].astype(np.float32) - audio[1].astype(np.float32))
    else:
        return 0.0
    return float(np.sqrt(np.mean(side**2) + 1e-12))


def evaluate_output_guard(
    *,
    original: np.ndarray,
    candidate: np.ndarray,
    enabled: bool,
    max_abs_rms_delta_db: float,
    stereo_side_ratio_min: float,
    stereo_side_ratio_max: float,
    sr: int = 48000,
    pleasantness_min_delta: float = -0.05,  # §v10: max tolerable pleasantness drop
) -> OutputGuardDecision:
    """§v10 Bewertet output guard — HPE als zentrale Metrik.

    Drei Prüfungen (HPE zuerst — das Ohr entscheidet):
    1. HPE: Klingt candidate angenehmer als original? (ΔP ≥ min_delta)
    2. RMS: Pegel im erlaubten Bereich?
    3. Stereo: Side-Ratio im erlaubten Bereich?
    """
    rms_delta_db = float(20.0 * np.log10((rms(candidate) + 1e-12) / (rms(original) + 1e-12)))
    side_ratio = 1.0
    pleasantness_delta = 0.0

    def _is_stereo_2d(arr: np.ndarray) -> bool:
        if arr.ndim != 2:
            return False
        _res_typed_73: bool = (arr.shape[1] == 2 and arr.shape[0] > 2) or (arr.shape[0] == 2 and arr.shape[1] > 2)
        return _res_typed_73

    is_stereo = _is_stereo_2d(original) and _is_stereo_2d(candidate)
    if is_stereo:
        side_ratio = float((side_rms(candidate) + 1e-12) / (side_rms(original) + 1e-12))

    if not enabled:
        return OutputGuardDecision(False, "disabled", rms_delta_db, side_ratio)

    # §v10 Prüfung 1: HPE — das WICHTIGSTE Kriterium
    try:
        from backend.core.human_pleasantness_estimator import compare_pleasantness

        hpe_cmp = compare_pleasantness(
            np.asarray(original, dtype=np.float32),
            np.asarray(candidate, dtype=np.float32),
            sr,
        )
        pleasantness_delta = float(hpe_cmp.get("delta_score", 0.0))

        if pleasantness_delta < pleasantness_min_delta:
            return OutputGuardDecision(
                True,
                f"pleasantness_drop (ΔP={pleasantness_delta:+.3f} < {pleasantness_min_delta:+.3f})",
                rms_delta_db,
                side_ratio,
                pleasantness_delta,
            )
    except Exception as e:
        logger.warning("Ausgabe_guard.py::_is_stereo_2d Ersatzpfad: %s", e)
        pass  # HPE nicht verfügbar → nur technische Prüfung

    # Prüfung 2: RMS
    if abs(rms_delta_db) > float(max_abs_rms_delta_db):
        return OutputGuardDecision(True, "rms_shift", rms_delta_db, side_ratio, pleasantness_delta)

    # Prüfung 3: Stereo
    if is_stereo and not (float(stereo_side_ratio_min) <= side_ratio <= float(stereo_side_ratio_max)):
        return OutputGuardDecision(True, "stereo_side_ratio", rms_delta_db, side_ratio, pleasantness_delta)

    return OutputGuardDecision(False, "ok", rms_delta_db, side_ratio, pleasantness_delta)
