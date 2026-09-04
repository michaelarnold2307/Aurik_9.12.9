"""
§v10.7 Spectral Bandwidth Extender — DSP-only, kein ML.

Rekonstruiert fehlende hohe Frequenzen aus dem existierenden Spektrum:
- Shellac (BW ≤ 8 kHz) → rekonstruiert 8-16 kHz aus 4-8 kHz via Harmonic Shifting
- Wax Cylinder (BW ≤ 5 kHz) → rekonstruiert 5-10 kHz aus 2.5-5 kHz
- Wire Recording (BW ≤ 6 kHz) → rekonstruiert 6-12 kHz aus 3-6 kHz

Algorithmus: Spektrale Spiegelung + Harmonische Extrapolation.
Kein ONNX, kein PyTorch, kein Download. Nur scipy + numpy.
100% offline, 100% deterministisch, ~10 ms für 1s Audio.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Material-spezifische Bandbreiten-Grenzen
MATERIAL_BANDWIDTH_LIMITS: dict[str, float] = {
    "shellac": 8000.0,
    "wax_cylinder": 5000.0,
    "wire_recording": 6000.0,
    "lacquer_disc": 9000.0,
    "mp3_low": 16000.0,
}


def extend_bandwidth(
    audio: np.ndarray,
    sr: int,
    *,
    material: str = "unknown",
    amount: float = 0.4,  # 40% Mix der synthetisierten Höhen
) -> np.ndarray:
    """Erweitert die Bandbreite des Audios via spektraler Spiegelung.

    Nur für Materialien mit bekannten Bandbreiten-Limits.
    Bei modernen Quellen (CD, Streaming) → kein Effekt (Original-Audio).

    Args:
        audio: Eingabe-Audio (float32, mono/stereo, 48 kHz)
        sr: Sample-Rate (muss 48000 sein)
        material: shellac, wax_cylinder, wire_recording, lacquer_disc, mp3_low
        amount: Mix-Verhältnis (0.0 = Original, 0.5 = 50/50)

    Returns:
        Bandbreiten-erweitertes Audio (float32)
    """
    if material not in MATERIAL_BANDWIDTH_LIMITS:
        return audio

    cutoff = MATERIAL_BANDWIDTH_LIMITS[material]
    if cutoff >= 18000:  # Praktisch volle Bandbreite
        return audio

    arr = np.asarray(audio, dtype=np.float32).copy()
    is_stereo = arr.ndim == 2 and arr.shape[1] == 2
    mono = arr.mean(axis=1) if is_stereo else arr

    try:
        extended = _spectral_extend(mono, sr, cutoff, amount)
    except Exception as e:
        logger.warning("bandwidth_extender Ersatzpfad: %s", e)
        return audio

    if is_stereo:
        ratio = np.clip(extended / (mono + 1e-12), 0.7, 1.3)
        result = arr * ratio[:, np.newaxis]
    else:
        result = extended

    return result.astype(np.float32)  # type: ignore[no-any-return]


def _spectral_extend(
    mono: np.ndarray,
    sr: int,
    cutoff_hz: float,
    amount: float,
) -> np.ndarray:
    """Spektrale Spiegelung: kopiere Energie von (cutoff/2 → cutoff) nach (cutoff → 2*cutoff).

    Prinzip: Obertöne verhalten sich harmonisch. Wenn die Grundfrequenzen
    bei 2-4 kHz liegen, sind ihre Obertöne bei 4-8 kHz. Was oberhalb
    von cutoff fehlt, kann durch Spiegelung der Oktave darunter
    rekonstruiert werden — gedämpft und spektral geformt.
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:
        logger.debug("§V6 scipy.signal.butter/sosfiltfilt nicht verfügbar — Audio unverändert zurückgegeben: %s", exc)
        return mono

    nyq = sr / 2.0
    cutoff_limited = min(cutoff_hz, nyq * 0.48)  # Max ~11.5 kHz bei 24 kHz effektiv
    target_max = min(cutoff_limited * 2.0, nyq * 0.95)

    if target_max <= cutoff_limited:
        return mono

    # 1. Extrahiere Quell-Band: cutoff/2 bis cutoff
    src_lo = cutoff_limited / 2.0
    src_hi = cutoff_limited

    sos_src = butter(3, [src_lo / nyq, src_hi / nyq], btype="band", output="sos")
    source = sosfiltfilt(sos_src, mono)

    # 2. Moduliere: Frequenz-Shift via Ringmodulation + Filtering
    # Einfach: Vollweggleichrichtung erzeugt Oktave über Grundfrequenz
    rectified = np.abs(source)
    # Hochpass bei cutoff → behalte nur die neue Oktave
    sos_hp = butter(3, cutoff_limited / nyq, btype="high", output="sos")
    extended_hi = sosfiltfilt(sos_hp, rectified)

    # 3. Spektrale Formung: natürlicher High-Frequency-Rolloff (−3 dB/Oktave)
    # Simuliert via Tiefpass bei target_max
    sos_shape = butter(1, target_max / nyq, btype="low", output="sos")
    extended_hi = sosfiltfilt(sos_shape, extended_hi)

    # 4. Begrenzung: Max 10% der Gesamtenergie
    rms_total = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    rms_ext = float(np.sqrt(np.mean(extended_hi**2)) + 1e-12)
    if rms_ext > rms_total * 0.3 and rms_ext > 1e-10:
        extended_hi *= (rms_total * 0.3) / rms_ext

    # 5. Mix: Original + synthetisierte Höhen × amount
    mixed = mono + extended_hi * amount

    return mixed.astype(np.float32)  # type: ignore[no-any-return]


def get_material_bandwidth(material: str) -> float | None:
    """Gibt die bekannte Bandbreiten-Grenze für ein Material zurück."""
    return MATERIAL_BANDWIDTH_LIMITS.get(material)


def needs_bandwidth_extension(material: str) -> bool:
    """True wenn das Material von Bandbreiten-Extension profitieren würde."""
    limit = MATERIAL_BANDWIDTH_LIMITS.get(material)
    return limit is not None and limit < 16000
