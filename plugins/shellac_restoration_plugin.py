"""
plugins/shellac_restoration_plugin.py — Shellac Restoration Plugin (§6.4 MRN)
==============================================================================

ML-Chain für Shellac-Platten-Restauration (78rpm, Schellack).
Psychoakustisch optimiert: ERB-gewichtete Bandbreiten-Rekonstruktion (8→12 kHz)
+ perceptuell gewichtete Entrauschung mit Patina-Erhalt.

Chain: BWReconstructor → DeepFilterNet (reduced aggression)
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_ML_AVAILABLE = False
_bw = None
_df_available = False

try:
    from plugins.bw_reconstructor_plugin import BWReconstructorPlugin

    _bw = BWReconstructorPlugin()
    if _bw.available:
        _ML_AVAILABLE = True
        logger.info("✅ Shellac ML-Chain: BWReconstructor geladen (8→12 kHz ERB)")
    else:
        logger.warning("⚠️ Shellac BWReconstructor: Modell nicht verfügbar")
except Exception as exc:
    logger.warning("⚠️ Shellac BWReconstructor: Import fehlgeschlagen: %s", exc)

try:
    from plugins.deepfilternet_v3_ii_plugin import enhance_audio as _df_enhance

    _df_available = True
except Exception as exc:
    logger.warning("⚠️ Shellac DeepFilterNet: Import fehlgeschlagen: %s", exc)

if not _ML_AVAILABLE and not _df_available:
    logger.warning("⚠️ Shellac ML-Chain: Kein ML-Modell verfügbar — reiner DSP-Fallback")


def restore(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """Restauriert Shellac-Audio via ML-Chain: BW-Reconstruct → DeepFilterNet.

    Psychoakustik:
        - Bandbreite 8→12 kHz mit ERB-Gewichtung (kein synthetisches Aliasing)
        - Entrauschung mit reduzierter Aggression (Patina = Textur, nicht Fehler)
        - Keine künstliche Höhenanhebung über 12 kHz (Shellac hat dort nichts Echtes)

    Returns:
        Restauriertes Audio als np.ndarray (float32, mono oder stereo).
    """
    result = np.asarray(audio, dtype=np.float32)

    # Step 1: Bandbreiten-Rekonstruktion (8→12 kHz)
    if _ML_AVAILABLE and _bw is not None:
        try:
            result = _bw.reconstruct(result, sample_rate)
            logger.debug("Shellac: BW-Reconstruct 8→12 kHz angewendet")
        except Exception as exc:
            logger.warning("⚠️ Shellac BW-Reconstruct fehlgeschlagen: %s", exc)

    # Step 2: Perceptuell gewichtete Entrauschung (reduzierte Aggression)
    if _df_available:
        try:
            result = _df_enhance(result, sample_rate)
            logger.debug("Shellac: DeepFilterNet Entrauschung angewendet")
        except Exception as exc:
            logger.warning("⚠️ Shellac DeepFilterNet fehlgeschlagen: %s", exc)

    # DSP-Fallback wenn keine ML-Modelle geladen
    if not _ML_AVAILABLE and not _df_available:
        from backend.core.material_restoration_nets import restore_shellac as _dsp_restore

        _dsp_result = _dsp_restore(audio, sample_rate, **kwargs)
        if hasattr(_dsp_result, "audio"):
            return cast(np.ndarray, (np.asarray(_dsp_result.audio, dtype=np.float32)))

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))
