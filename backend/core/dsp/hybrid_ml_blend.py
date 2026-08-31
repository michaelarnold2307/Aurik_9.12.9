"""Hybrid-ML-Blend (§v10.101) — kanonische Naht für ML/DSP-Hybride.

Jede Stelle, an der ein ML-Modell-Ausgang (wet) auf DSP-/Original-Audio (dry)
trifft, MUSS diese Funktion verwenden — außerhalb UND innerhalb des
UV3-Phasen-Executors (der sie zentral bereits anwendet, §G104/§G101).

Garantien (alle GEBOTE-Kanäle in EINER Naht):
  1. §G104 JND-Gate: unhörbare Änderung → dry zurück (kein Artefakt-Risiko)
  2. §G101 perceptual_blend: Bark-Band-Wet, nur oberhalb der Maskierungsschwelle
  3. §8.2 Energie-Guard: ML-Stille (RMS < 20 % des Inputs) → dry zurück
  4. NaN/Inf + Clip: §3.1 Pflicht
  5. Deterministisch (§G136): keine Zufallszahlen, reine Funktion

Dadurch verhalten sich alle Hybride identisch — als wäre der Kanal seit
Beginn von Aurik vorhanden gewesen.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_ENERGY_FLOOR_RATIO = 0.20  # ML-Output-RMS darf nicht unter 20% des Inputs fallen
_DEFAULT_WET = 1.0  # perceptual_blend maskiert unhörbare Bänder ohnehin


def hybrid_ml_apply(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    *,
    scalar_wet: float = _DEFAULT_WET,
    min_audible_bands: int = 2,
    material_type: str = "unknown",
    genre: str = "unknown",
) -> np.ndarray:
    """Blendet ML-Output perzeptuell auf Dry — mit allen Compliance-Gates.

    Args:
        dry:  Original-Audio (float32, mono [T] oder stereo [N, C]/[2, N]).
        wet:  ML-verarbeitetes Audio, gleiche Shape wie dry.
        sr:   Sample-Rate (48000).
        scalar_wet:          Maximaler Global-Wet [0, 1].
        min_audible_bands:   JND-Gate-Schwelle (§G104, Standard 2).
        material_type/genre: Adaptivität für JND-Faktoren (§v10.116).

    Returns:
        Perzeptuell geblendetes Audio in der Shape von dry.
    """
    dry = np.asarray(dry, dtype=np.float32)
    wet = np.asarray(wet, dtype=np.float32)

    # Shape-/Finite-Guards (§3.1)
    if dry.shape != wet.shape:
        logger.debug("hybrid_ml_apply: Shape-Mismatch dry %s vs wet %s — dry zurück", dry.shape, wet.shape)
        return dry.copy()
    dry = np.nan_to_num(dry, nan=0.0, posinf=0.0, neginf=0.0)
    wet = np.nan_to_num(wet, nan=0.0, posinf=0.0, neginf=0.0)

    # §V53 ML-NAHT-GUARDS (Vorschlag 03, docs/proposals): Polarity + Lag —
    # deterministisch, vor allen übrigen Gates. Konstantenquelle:
    # CalibratedConstants (getattr-Fallback bis zur Übernahme).
    try:
        from backend.core import calibrated_constants as _cc

        _max_lag = int(getattr(_cc, "MAX_ML_LAG_SAMPLES", 128))
    except Exception:  # nicht blockierend
        _max_lag = 128

    # 1. Polarity-Guard: Pearson-Korrelation (zentriert) < 0 ⇒ invertiert.
    _flat_dry = dry.ravel().astype(np.float64)
    _flat_wet = wet.ravel().astype(np.float64)
    _d_c = _flat_dry - np.mean(_flat_dry)
    _w_c = _flat_wet - np.mean(_flat_wet)
    _dot = float(np.dot(_d_c, _w_c))
    if _dot < 0.0:
        logger.warning(
            "hybrid_ml_apply: Polarity-Guard — invertierte ML-Ausgabe erkannt (dot=%.3g) — Vorzeichen korrigiert",
            _dot,
        )
        wet = -wet

    # 2. Lag-Guard: Kreuzkorrelationsmaximum; |lag| > MAX_ML_LAG_SAMPLES ⇒ dry.
    _n = _flat_dry.size
    if _n >= 1024:
        _step = max(1, _n // 16384)
        _d_ds = _d_c[::_step]
        _w_ds = _w_c[::_step]
        _corr = np.correlate(_w_ds, _d_ds, mode="full")
        _lag = int(np.argmax(_corr) - (_d_ds.size - 1)) * _step
        if abs(_lag) > _max_lag:
            logger.warning(
                "hybrid_ml_apply: Lag-Guard — ML-Ausgabe um %d Samples versetzt (> %d) — dry zurück",
                _lag,
                _max_lag,
            )
            return dry.copy()

    # §G104 Perceptual-JND-Gate: unhörbare Änderung → Rollback auf dry.
    try:
        from backend.core.dsp.perceptual_gate import should_skip_phase

        _shape_gate = dry if dry.ndim == 2 else dry
        if should_skip_phase(
            _shape_gate,
            wet,
            sr,
            min_audible_bands=min_audible_bands,
            material_type=material_type,
            genre=genre,
        ):
            logger.debug("hybrid_ml_apply: JND-Gate — Änderung unhörbar → dry")
            return dry.copy()
    except Exception as _jnd_exc:  # nicht blockierend
        logger.debug("hybrid_ml_apply: JND-Gate nicht verfügbar: %s", _jnd_exc)

    # §8.2 Energie-Guard: ML darf das Signal nicht in Stille verwandeln.
    _rms_dry = float(np.sqrt(np.mean(dry**2)) + 1e-12)
    _rms_wet = float(np.sqrt(np.mean(wet**2)) + 1e-12)
    if _rms_wet < _ENERGY_FLOOR_RATIO * _rms_dry:
        logger.debug(
            "hybrid_ml_apply: Energie-Guard (wet-RMS %.4f < %.2f x dry-RMS %.4f) — dry",
            _rms_wet,
            _ENERGY_FLOOR_RATIO,
            _rms_dry,
        )
        return dry.copy()

    if scalar_wet <= 0.0:
        return dry.copy()
    if scalar_wet >= 1.0 and np.array_equal(dry, wet):
        return dry.copy()

    # §G101 Perceptual-Blend: Bark-Band-Wet oberhalb der Maskierungsschwelle.
    try:
        from backend.core.dsp.perceptual_blend import perceptual_blend

        out = perceptual_blend(dry, wet, sr, scalar_wet=float(scalar_wet))
    except Exception as _pb_exc:
        logger.debug("hybrid_ml_apply: perceptual_blend nicht verfügbar (%s) — skalarer Fallback", _pb_exc)
        out = dry + float(np.clip(scalar_wet, 0.0, 1.0)) * (wet - dry)

    return cast(
        np.ndarray, (np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32))
    )
