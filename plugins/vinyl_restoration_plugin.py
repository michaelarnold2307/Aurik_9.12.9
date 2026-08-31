"""
plugins/vinyl_restoration_plugin.py — Vinyl Restoration Plugin (§6.6 MRN)
==========================================================================

ML-Chain für Vinyl-Platten-Restauration (LP, 33/45rpm).
Primär: BanquetVinylPlugin (ONNX, 92 MB) — speziell für Vinyl-Crackle/Noise trainiert.
Sekundär: DeepFilterNet für Restsignale.

Psychoakustik: Stereo-Phase-Aware — keine Mono-Zwangskonvertierung.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_ML_AVAILABLE = False
_banquet = None
_df_available = False

try:
    from plugins.banquet_vinyl_plugin import BanquetVinylPlugin

    _banquet = BanquetVinylPlugin()
    _ML_AVAILABLE = True
    logger.info("✅ Vinyl ML-Chain: BanquetVinylPlugin geladen (92 MB ONNX)")
except Exception as exc:
    logger.warning("⚠️ Vinyl BanquetVinylPlugin: Import fehlgeschlagen: %s", exc)

try:
    from plugins.deepfilternet_v3_ii_plugin import enhance_audio as _df_enhance

    _df_available = True
except Exception as exc:
    logger.warning("⚠️ Vinyl DeepFilterNet: Import fehlgeschlagen: %s", exc)

if not _ML_AVAILABLE:
    logger.warning("⚠️ Vinyl ML-Chain: Banquet nicht verfügbar — DSP-Fallback aktiv")


def restore(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """Restauriert Vinyl-Audio via ML-Chain: Banquet → DeepFilterNet.

    Psychoakustik:
        - BanquetVinyl ist speziell für Vinyl-Crackle/Noise trainiert
        - Stereo-Erhalt: keine Mono-Zwangskonvertierung
        - DeepFilterNet als Feinpolitur mit niedriger Aggression

    Returns:
        Restauriertes Audio als np.ndarray (float32).
    """
    result = np.asarray(audio, dtype=np.float32)
    strength = float(kwargs.get("strength", 0.8))

    # Step 1: BanquetVinyl — primäre Vinyl-Restauration
    if _ML_AVAILABLE and _banquet is not None:
        try:
            result = _banquet.process(result, sample_rate, strength=strength)
            logger.debug("Vinyl: BanquetVinyl angewendet (strength=%.2f)", strength)
        except Exception as exc:
            logger.warning("⚠️ Vinyl BanquetVinyl fehlgeschlagen: %s → DSP-Fallback", exc)
            _dsp_fallback = True
        else:
            _dsp_fallback = False
    else:
        _dsp_fallback = True

    # Step 2: DeepFilterNet — Restsignale entfernen
    if _df_available and not _dsp_fallback:
        try:
            result = _df_enhance(result, sample_rate)
            logger.debug("Vinyl: DeepFilterNet Feinpolitur angewendet")
        except Exception as exc:
            logger.warning("⚠️ Vinyl DeepFilterNet fehlgeschlagen: %s", exc)

    # DSP-Fallback
    if _dsp_fallback:
        from backend.core.material_restoration_nets import restore_vinyl as _dsp_restore

        apply_riaa = bool(kwargs.get("apply_riaa", False))
        _dsp_result = _dsp_restore(audio, sample_rate, apply_riaa=apply_riaa, **kwargs)
        if hasattr(_dsp_result, "audio"):
            return cast(np.ndarray, (np.asarray(_dsp_result.audio, dtype=np.float32)))

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))
