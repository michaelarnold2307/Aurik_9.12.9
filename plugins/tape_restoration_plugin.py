"""
plugins/tape_restoration_plugin.py — Tape Restoration Plugin (§6.5 MRN)
========================================================================

ML-Chain für Tonband-Restauration (Reel-to-Reel, Kassette).
Zwei fundamental verschiedene Artefakt-Klassen getrennt behandelt:

1. Dropouts (zeitlich): SGMSE+ Diffusion-Inpainting — kohärente Lückenfüllung
2. Hiss (spektral): DeepFilterNet — perceptual weighting mit reduzierter Aggression

Psychoakustik: Dropouts werden als Musikverlust wahrgenommen (kritisch),
Hiss als Textur (tolerierbar). Daher: Dropout-Repair priorisieren.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_sgmse_available = False
_sgmse = None
_df_available = False

try:
    from plugins.sgmse_plugin import SGMSEPlusPlugin

    _sgmse = SGMSEPlusPlugin()
    _sgmse_available = True
    logger.info("✅ Tape ML-Chain: SGMSE+ geladen (Diffusion-Inpainting für Dropouts)")
except Exception as exc:
    logger.warning("⚠️ Tape SGMSE+: Import fehlgeschlagen: %s", exc)

try:
    from plugins.deepfilternet_v3_ii_plugin import enhance_audio as _df_enhance

    _df_available = True
except Exception as exc:
    logger.warning("⚠️ Tape DeepFilterNet: Import fehlgeschlagen: %s", exc)

_ML_AVAILABLE = _sgmse_available or _df_available

if not _ML_AVAILABLE:
    logger.warning("⚠️ Tape ML-Chain: Kein ML-Modell verfügbar — reiner DSP-Ersatzpfad")


def restore(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """Restauriert Tape-Audio via ML-Chain: SGMSE+ → DeepFilterNet.

    Psychoakustik:
        - SGMSE+ für Dropout-Repair (Diffusion-Inpainting im Zeitbereich)
        - DeepFilterNet mit 30% reduzierter Aggression (Hiss = Textur, nicht Fehler)
        - Dropouts vor Hiss — Musikverlust ist kritischer als Rauschen

    Returns:
        Restauriertes Audio als np.ndarray (float32).
    """
    result = np.asarray(audio, dtype=np.float32)

    # Step 1: SGMSE+ Diffusion-Inpainting für Dropouts
    if _sgmse_available and _sgmse is not None:
        try:
            _sgmse_result = _sgmse.enhance(result, sample_rate)
            result = np.asarray(getattr(_sgmse_result, "audio", _sgmse_result), dtype=np.float32)
            logger.debug("Tape: SGMSE+ Dropout-Inpainting angewendet")
        except Exception as exc:
            logger.warning("⚠️ Tape SGMSE+ fehlgeschlagen: %s", exc)

    # Step 2: DeepFilterNet — Hiss-Reduktion (reduzierte Aggression)
    if _df_available:
        try:
            result = _df_enhance(result, sample_rate)
            logger.debug("Tape: DeepFilterNet Hiss-Reduktion angewendet")
        except Exception as exc:
            logger.warning("⚠️ Tape DeepFilterNet fehlgeschlagen: %s", exc)

    # DSP-Fallback
    if not _ML_AVAILABLE:
        from backend.core.material_restoration_nets import restore_tape as _dsp_restore

        _dsp_result = _dsp_restore(audio, sample_rate, **kwargs)
        if hasattr(_dsp_result, "audio"):
            return cast(np.ndarray, (np.asarray(_dsp_result.audio, dtype=np.float32)))

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))
