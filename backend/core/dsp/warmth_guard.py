"""§WBG (V25) Wärmeband-Guard.

Misst den kumulativen Energieverlust im 200–800 Hz Band (Wärmeband) nach jeder
Phase. Überschreitet der Verlust 2.5 dB kumulativ, werden weitere Phasen mit
einem Warmth-Blend-Faktor skaliert.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.warmth_guard import measure_warmth_band_delta, WarmthBandResult
    result = measure_warmth_band_delta(pre, post, sr)
    # UV3 akkumuliert result.loss_db in _restoration_context["warmth_band_loss_db"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

from backend.core.residuum_masking import estimate_delta_masking_jnd_db

logger = logging.getLogger(__name__)

# Wärmeband: 200–800 Hz
_WARMTH_LOW_HZ = 200.0
_WARMTH_HIGH_HZ = 800.0
# Kumulativer Verlust-Schwellwert → alle weiteren Phasen skalieren
WARMTH_LOSS_THRESHOLD_DB = 2.5
# Maximaler Gesamtverlust für Blend-Berechnung
WARMTH_MAX_LOSS_DB = 5.0


@dataclass
class WarmthBandResult:
    """Ergebnis der Wärmeband-Messung.

    Attributes:
        loss_db: Energieverlust im 200–800 Hz Band (positiv = Verlust in dB).
        gain_db: Energiegewinn (positiv = Gewinn; bei subtr. Phases meist 0).
        ok: True wenn loss_db <= 0 (kein Wärmeverlust).
        warmth_blend_factor: Empfohlener Blend-Faktor für nachfolgende Phasen
            (1.0 = kein Eingriff, < 1.0 = Strength reduzieren).
            Wird aus kumulativem Verlust berechnet, nicht aus diesem einzelnen Delta.
    """

    loss_db: float
    gain_db: float
    ok: bool
    warmth_blend_factor: float = 1.0


def _bandpass_energy_db(audio: np.ndarray, sr: int) -> float:
    """Berechnet die RMS-Energie im 200–800 Hz Band in dBFS."""
    try:
        mono = audio.mean(axis=0).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
        nyq = sr / 2.0
        sos = butter(4, [_WARMTH_LOW_HZ / nyq, _WARMTH_HIGH_HZ / nyq], btype="band", output="sos")
        filtered = sosfiltfilt(sos, mono).astype(np.float32)
        rms = float(np.sqrt(np.mean(filtered**2) + 1e-12))
        return float(20.0 * np.log10(rms + 1e-12))
    except Exception as e:
        logger.warning("warmth_guard.py::_bandpass_energy_db Ersatzpfad: %s", e)
        return -120.0


def measure_warmth_band_delta(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    *,
    cumulative_loss_db: float = 0.0,
) -> WarmthBandResult:
    """Misst den Energieverlust im Wärmeband (200–800 Hz) dieser Phase.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase.
        sr: Sample-Rate (muss 48000 sein).
        cumulative_loss_db: Bisher akkumulierter Wärmeverlust (für Blend-Berechnung).

    Returns:
        WarmthBandResult mit loss_db und warmth_blend_factor.
    """
    assert sr == 48000
    _fallback = WarmthBandResult(loss_db=0.0, gain_db=0.0, ok=True, warmth_blend_factor=1.0)

    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if pre.shape != post.shape or pre.size < 256:
            return _fallback

        pre_db = _bandpass_energy_db(pre, sr)
        post_db = _bandpass_energy_db(post, sr)

        delta_db = post_db - pre_db  # negativ = Verlust
        loss_db = max(0.0, -delta_db)  # positiv = Verlust
        gain_db = max(0.0, delta_db)  # positiv = Gewinn

        # §P1-3 (Hörordnung Ebene 2): Der maskierte Anteil des aktuellen
        # Phasen-Verlusts ist unhörbar und zählt NICHT zum kumulativen Verlust
        # (weniger falsche Rollbacks). Vorherige Verluste bleiben voll wirksam.
        _jnd = estimate_delta_masking_jnd_db(pre, post, sr, freq_range_hz=(_WARMTH_LOW_HZ, _WARMTH_HIGH_HZ))
        _audible_loss_db = max(0.0, loss_db - float(_jnd.jnd_db))

        # Warmth-Blend basierend auf kumulativem (hörbaren) Verlust
        total_loss = cumulative_loss_db + _audible_loss_db
        if total_loss > WARMTH_LOSS_THRESHOLD_DB:
            blend = float(np.clip(1.0 - total_loss / WARMTH_MAX_LOSS_DB, 0.1, 1.0))
        else:
            blend = 1.0

        ok = loss_db <= 0.01  # Toleranz 0.01 dB

        if loss_db > 0.5:
            logger.info(
                "§V25 Wärmeband-Guard: loss=%.2f dB (200–800 Hz) kumul=%.2f dB → blend=%.2f | §P1-3 Maskierungs-JND=%.2f dB → hörbarer Verlust=%.2f dB",
                loss_db,
                total_loss,
                blend,
                _jnd.jnd_db,
                _audible_loss_db,
            )

        return WarmthBandResult(
            loss_db=round(loss_db, 3),
            gain_db=round(gain_db, 3),
            ok=ok,
            warmth_blend_factor=round(blend, 3),
        )

    except Exception as exc:
        logger.debug("measure_warmth_band_delta nicht blockierend: %s", exc)
        return _fallback
