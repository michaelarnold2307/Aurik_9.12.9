"""
Denoise Helpers — Aurik 10.0.0
==============================

Module-level helper functions and constants extracted from phase_03_denoise.py:
- Era-adaptive NR routing (§4.4, §2.14+)
- Decade strength multiplier (piecewise-linear interpolation)
- SOTA ML-NR Routing decision logic

Diese Funktionen sind stateless und benötigen keine Phase-Instanz.
Sie werden von phase_03_denoise.py importiert.

Author: Aurik 10.0.0 Development Team
Version: 2.0.0 (Professional Upgrade)
Date: 15. Februar 2026
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# §4.4 Era-Aware NR-Routing constants
_OMLSA_ONLY_MATERIALS_P03 = frozenset({"wax_cylinder", "wire_recording", "acoustic_recording"})
_ERA_ACOUSTIC_CUTOFF = 1930  # Phonograph era: character noise, no ML NR
_ERA_EARLY_ELECTRIC_CUTOFF = 1945  # Shellac electrical: restricted DFN only
_MIIPHER_SNR_CUTOFF_DB = 10.0  # MIIPHER primary when SNR below this threshold
_MIIPHER_SINGING_MIN = 0.35  # Minimum PANNs confidence for MIIPHER activation

# §2.14+ Era-adaptive NR: piecewise-linear era→strength multiplier (kalibriert):
#   1890–1930: ×1.15 (aggressiv — hohes intrinsisches Rauschen)
#   1940:      ×1.10 (frühe elektrische Ära)
#   1950:      ×1.05 (verbessertes Tape/Vinyl)
#   1960:      ×1.00 (neutrale Baseline)
#   1970:      ×0.95 (bessere Produktion)
#   1980:      ×0.90 (digitale Transition)
#   1990+:     ×0.80 (saubere digitale Quellen)
ERA_DECADE_KNOTS: tuple[tuple[int, float], ...] = (
    (1890, 1.15),
    (1930, 1.15),
    (1940, 1.10),
    (1950, 1.05),
    (1960, 1.00),
    (1970, 0.95),
    (1980, 0.90),
    (1990, 0.80),
    (2025, 0.80),
)
_ERA_DECADE_MIN = ERA_DECADE_KNOTS[0][0]
_ERA_DECADE_MAX = ERA_DECADE_KNOTS[-1][0]


def decade_strength_multiplier(decade: int) -> float:
    """Stärke-Multiplikator für eine Aufnahme-Dekade (§2.14+ Era-adaptive NR).

    Piecewise-lineare Interpolation über ``ERA_DECADE_KNOTS``; außerhalb des
    Knotenbereichs wird auf den Randwert geklemmt (1890/2025).
    """
    _dec = float(max(_ERA_DECADE_MIN, min(_ERA_DECADE_MAX, int(decade))))
    _era_decades = [k[0] for k in ERA_DECADE_KNOTS]
    _era_mults = [k[1] for k in ERA_DECADE_KNOTS]
    return float(np.interp(_dec, _era_decades, _era_mults))


def _determine_era_nr_routing(
    era_decade: int,
    material_type: str,
    est_snr_db: "float | None",
    panns_singing: float,
    is_vocal_material: bool,
    is_non_digital: bool,
) -> str:
    """
    §4.4 SOTA Era-Aware ML-NR Routing decision (v10.0.0.x).

    Returns one of:
      "miipher_primary"  — MIIPHER → DFN fallback (deep SNR, post-1950, vocal)
      "sota_4layer"      — §v10.200 SOTA 4-Ebenen Denoiser (post-1950, music, high-quality)
      "dfn_primary"      — DFN primary, current SOTA behavior
      "dfn_restricted"   — DFN capped at 30 %% wet (early electrical 1930-1945, shellac)
      "omlsa_only"       — No ML NR (acoustic era, wax/wire, digital material)

    §0a Carrier-Chain compliance: Pre-1945 phonograph surface noise IS carrier
    character (SOFT_SATURATION = BEWAHREN). DFN/MIIPHER are speech-trained; applied
    to 1930s shellac they remove harmonic texture → timbral corruption. OMLSA with
    conservative g_floor is correct for those eras (§2.46 Carrier-Chain-Stufen).
    For post-1950 deep-noise vocal (SNR < 10 dB), MIIPHER delivers highest vocal
    quality (Zhang et al. 2023, Google; §4.4 SOTA Matrix 2026).
    """
    mat = str(getattr(material_type, "value", material_type) or "unknown").lower()
    if mat in _OMLSA_ONLY_MATERIALS_P03 or not is_non_digital:
        return "omlsa_only"
    if era_decade <= _ERA_ACOUSTIC_CUTOFF:
        return "omlsa_only"
    if era_decade <= _ERA_EARLY_ELECTRIC_CUTOFF and mat in ("shellac", "shellac_early"):
        return "dfn_restricted"
    if (
        is_vocal_material
        and panns_singing >= _MIIPHER_SINGING_MIN
        and est_snr_db is not None
        and est_snr_db < _MIIPHER_SNR_CUTOFF_DB
    ):
        return "miipher_primary"
    # §v10.200: Use SOTA 4-Layer for post-1950 music (non-vocal) material
    if not is_vocal_material and era_decade > _ERA_EARLY_ELECTRIC_CUTOFF:
        return "sota_4layer"
    return "dfn_primary"
