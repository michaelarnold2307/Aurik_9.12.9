"""Vocoder Chain — Spec-04-[RELEASE_MUST]-Kaskade (Rev. 2026-08-16).

Spec 02 §1.5 Schritt 13. Aktiviert wenn PQS-MOS < 4.3 nach Phase-Pipeline.

Kaskade (Spec 04 §[RELEASE_MUST] „Neuronale Synthese / Vocoder-Kaskade“):
    Studio-2026:  Vocos 48 kHz nativ → BigVGAN-v2 → HiFi-GAN → PGHI-ISTFT
    Restoration:  BigVGAN-v2 → HiFi-GAN → PGHI-ISTFT (Vocos verboten, §1.4)

VERBOTEN (Spec 04):
    - vocos_mel_spec_24khz.onnx als primäres Modell (SR-Mismatch zu 48 kHz)
    - Griffin-Lim als Endschritt in Studio-2026
DiffWave ist kein Vocoder-Tier (Inpainting-Aufgabe, phase_55) und hier bewusst
nicht verdrahtet.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


def _ok(result) -> bool:
    """True wenn ein synthetisiertes Audio-Array mit Inhalt vorliegt."""
    return result is not None and isinstance(result, np.ndarray) and result.size > 0


def activate_vocoder_chain(
    audio: np.ndarray,
    sample_rate: int = 48000,
    pqs_mos: float = 4.5,
    *,
    studio_mode: bool = False,
) -> np.ndarray | None:
    """Aktiviert die Vocoder-Kette wenn PQS-MOS unter Schwellwert.

    Args:
        audio:       Restauriertes Audio
        sample_rate: Sample-Rate
        pqs_mos:     Aktueller PQS-MOS-Wert
        studio_mode: True = Studio-2026-Kaskade inkl. Vocos-Tier-1 (§1.4:
                     Vocos ist im Restoration-Modus verboten).

    Returns:
        Vocoder-verarbeitetes Audio, oder None wenn nicht aktiviert.
    """
    if pqs_mos >= 4.3:
        return None  # Keine Vocoder-Kette nötig

    logger.info("Vocoder-Kette aktiviert (PQS-MOS %.1f < 4.3)", pqs_mos)
    arr = np.asarray(audio, dtype=np.float32)

    # Stufe 1 (nur Studio-2026): Vocos 48 kHz nativ — Spec 04 [RELEASE_MUST]
    if studio_mode:
        try:
            from plugins.vocos_plugin import get_vocos_plugin

            voc = get_vocos_plugin()
            res = voc.vocode(arr, sample_rate, mode="studio2026")
            out = getattr(res, "audio", None)
            if _ok(out):
                logger.info("Vocoder-Kette: Vocos 48 kHz erfolgreich")
                return cast(np.ndarray | None, (np.asarray(out, dtype=np.float32)))
        except Exception as e:
            logger.warning("Vocos 48 kHz fehlgeschlagen: %s — Rückfall zu BigVGAN-v2", e)

    # Stufe 2: BigVGAN-v2 (primär im Restoration-Modus)
    try:
        from plugins.bigvgan_v2_plugin import BigVGANv2Plugin

        result = BigVGANv2Plugin().synthesize(arr, sample_rate)
        if _ok(result):
            logger.info("Vocoder-Kette: BigVGAN-v2 erfolgreich")
            return cast(np.ndarray | None, (np.asarray(result, dtype=np.float32)))
    except Exception as e:
        logger.warning("BigVGAN-v2 fehlgeschlagen: %s — Rückfall zu HiFi-GAN", e)

    # Stufe 3: HiFi-GAN (Tertiär-Notfallstufe, Spec 04)
    try:
        from plugins.hifigan_plugin import HiFiGANPlugin

        _hf_result = HiFiGANPlugin().reconstruct(arr, sample_rate)
        if _ok(_hf_result):
            logger.info("Vocoder-Kette: HiFi-GAN Notfallstufe erfolgreich")
            return cast(np.ndarray | None, (np.asarray(_hf_result, dtype=np.float32)))
    except Exception as e:
        logger.warning("HiFi-GAN fehlgeschlagen: %s — Rückfall zu PGHI-ISTFT", e)

    # Stufe 4: PGHI (deterministischer DSP-Endfall, Spec 04; §4.5 pghi_reconstruct)
    try:
        from scipy.signal import stft as scipy_stft

        from dsp.pghi import pghi_reconstruct

        _, _, z_stft = scipy_stft(arr, fs=sample_rate, nperseg=2048, noverlap=2048 - 256)
        mag = np.abs(z_stft).astype(np.float32)
        _pghi_raw = pghi_reconstruct(mag, sr=sample_rate, win_size=2048, hop=256)
        out = np.asarray(_pghi_raw, dtype=np.float32)
        # Längen-Invariante: Resynthese auf Eingangslänge trimmen/paden (§G5 (GEBOTE.md))
        if out.shape[0] < arr.shape[0]:
            out = np.pad(out, (0, arr.shape[0] - out.shape[0]))
        else:
            out = out[: arr.shape[0]]
        logger.info("Vocoder-Kette: PGHI-DSP-Endfall erfolgreich")
        return cast(np.ndarray | None, out)
    except Exception as e:
        logger.error("Vocoder-Kette: ALLE Stufen fehlgeschlagen — Ursprung zurück: %s", e)
        return audio


def is_vocoder_available() -> bool:
    """Prueft ob mindestens ein Vocoder-Backend verfuegbar ist."""
    try:
        from plugins.bigvgan_v2_plugin import BigVGANv2Plugin

        return True
    except ImportError as _vocos_exc:
        logger.debug("Vocos nicht verfügbar: %s", _vocos_exc)
    try:
        from plugins.hifigan_plugin import HiFiGANPlugin

        return True
    except ImportError as _bgv_exc:
        logger.debug("BigVGAN-v2 nicht verfügbar: %s", _bgv_exc)
    return False
