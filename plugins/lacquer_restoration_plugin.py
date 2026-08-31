"""
plugins/lacquer_restoration_plugin.py — Lacquer Disc Restoration Plugin (§6.7 MRN)
===================================================================================

ML-Chain für Lacquer-Disc-Restauration (Acetat, Transcription Disc).
Zwischen Shellac und Vinyl: bessere Bandbreite (12→14 kHz) aber charakteristisches
Palmitinsäure-Knistern als "Patina" erhalten.

Chain: BWReconstructor → DeepFilterNet (40% Wet — Patina-Erhalt)
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
        logger.info("✅ Lacquer ML-Chain: BWReconstructor geladen (12→14 kHz ERB)")
    else:
        logger.warning("⚠️ Lacquer BWReconstructor: Modell nicht verfügbar")
except Exception as exc:
    logger.warning("⚠️ Lacquer BWReconstructor: Import fehlgeschlagen: %s", exc)

try:
    from plugins.deepfilternet_v3_ii_plugin import enhance_audio as _df_enhance

    _df_available = True
except Exception as exc:
    logger.warning("⚠️ Lacquer DeepFilterNet: Import fehlgeschlagen: %s", exc)

if not _ML_AVAILABLE and not _df_available:
    logger.warning("⚠️ Lacquer ML-Chain: Kein ML-Modell verfügbar — reiner DSP-Fallback")


def restore(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """Restauriert Lacquer-Disc-Audio via ML-Chain: BW-Reconstruct → DeepFilterNet.

    Psychoakustik:
        - Bandbreite 12→14 kHz konservativ (kein 16k — klingt synthetisch)
        - DeepFilterNet mit 40% Wet — Palmitinsäure-Knistern als Patina erhalten
        - Charakteristisches Rauschen ist Textur, nicht Fehler

    Returns:
        Restauriertes Audio als np.ndarray (float32).
    """
    result = np.asarray(audio, dtype=np.float32)

    # Step 1: Bandbreiten-Rekonstruktion (12→14 kHz)
    if _ML_AVAILABLE and _bw is not None:
        try:
            result = _bw.reconstruct(result, sample_rate)
            logger.debug("Lacquer: BW-Reconstruct 12→14 kHz angewendet")
        except Exception as exc:
            logger.warning("⚠️ Lacquer BW-Reconstruct fehlgeschlagen: %s", exc)

    # Step 2: Perceptuell gewichtete Entrauschung (40% Wet — Patina-Erhalt)
    if _df_available:
        try:
            # Blende 40% Wet: 60% Original + 40% entrauscht = Patina bleibt hörbar
            _denoised = _df_enhance(result, sample_rate)
            _wet = 0.40
            result = _wet * _denoised + (1.0 - _wet) * result
            logger.debug("Lacquer: DeepFilterNet 40%% Wet (Patina-Erhalt) angewendet")
        except Exception as exc:
            logger.warning("⚠️ Lacquer DeepFilterNet fehlgeschlagen: %s", exc)

    # DSP-Fallback
    if not _ML_AVAILABLE and not _df_available:
        from backend.core.material_restoration_nets import restore_lacquer as _dsp_restore

        _dsp_result = _dsp_restore(audio, sample_rate, **kwargs)
        if hasattr(_dsp_result, "audio"):
            return cast(np.ndarray, (np.asarray(_dsp_result.audio, dtype=np.float32)))

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))
