"""
RLP‑Entscheidung – objektiver Vergleich vor/nach RLP.

Die Funktion `should_keep_rlp` vergleicht Zwicker‑Metriken (Roughness) und LUFS. Wenn die Roughness nach RLP um ≥ 10 % sinkt oder der Integrated LUFS um ≥ 0.5 dB steigt, wird RLP beibehalten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .loudness_meter import compute_loudness
from .zwicker_metrics import compute_roughness_asper

logger = logging.getLogger(__name__)

@dataclass
class RLPDecisionResult:
    keep_rlp: bool
    roughness_improvement_db: float
    lufs_improvement_db: float


def should_keep_rlp(audio_pre: np.ndarray, audio_post: np.ndarray, sr: int) -> RLPDecisionResult:
    """
    Entscheidet, ob ein RLP‑Schritt beibehalten werden soll.

    Kriterien (SOTA):
      * Roughness sinkt um ≥ 10 % → keep
      * Integrated LUFS steigt um ≥ 0.5 dB → keep
    """
    pre_rough = compute_roughness_asper(audio_pre, sr)
    post_rough = compute_roughness_asper(audio_post, sr)
    rough_impr = (pre_rough - post_rough) / max(pre_rough, 1e-6)

    pre_lufs = compute_loudness(audio_pre, sr).integrated_lufs
    post_lufs = compute_loudness(audio_post, sr).integrated_lufs
    lufs_impr = post_lufs - pre_lufs

    keep = rough_impr >= 0.10 or lufs_impr >= 0.5
    logger.debug(
        "RLP‑Entscheidung: Rough=%+.2f%%, LUFS=%+.2fdB → %s",
        rough_impr * 100,
        lufs_impr,
        keep,
    )
    return RLPDecisionResult(keep_rlp=keep,
                             roughness_improvement_db=(pre_rough - post_rough),
                             lufs_improvement_db=lufs_impr)
