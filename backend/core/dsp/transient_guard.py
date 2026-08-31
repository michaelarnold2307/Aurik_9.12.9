"""§PEP (V22) Pre-Echo-Prevention — Transient-Shift-Detektor.

Prüft nach additiven ML-Phasen (phase_06, phase_07, phase_23), ob Transient-
Onsets zeitlich verschoben wurden (Pre-Echo). Shift > ±3.5 ms → blend_reduction
als Metadata-Flag; kein Rollback (non-blocking WARNING).

Messmethode (§v10.53): Cross-Correlation an Onset-Positionen.
- Onsets werden via Spektralfluss in pre detektiert (nur zur Lokalisierung).
- Für jeden Onset wird ein ±10.7 ms Fenster aus pre und post extrahiert.
- Normalisierte Cross-Correlation zwischen den Fenstern liefert den echten
  Zeitversatz (Lag des XCorr-Peaks).
- XCorr ist unempfindlich gegenüber spektralen Änderungen (EQ, Presence-Boost)
  und misst ausschließlich Zeitbereichs-Verschiebungen.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.transient_guard import detect_transient_shifts, TransientShiftResult
    result = detect_transient_shifts(pre, post, sr)
    # result.max_shift_ms > 3.5 → metadata["onset_shift_ms"] setzen
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

try:
    import librosa  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optionale Abhängigkeit
    librosa = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Toleranzgrenzwert für Onset-Verschiebung
# §01 (Spec 01/07): Attack-Zeiten ≤ ±2 ms Änderung — normative Grenze.
TRANSIENT_SHIFT_THRESHOLD_MS = 2.0
# §v10.52 Pre-Echo-Calibration: Blend-Divisor 2.0→5.0 + Max-Cap 0.60.
# 21ms Shift: 21/(2.0×5.0)=2.10→0.60 (Cap)
# 5ms Shift: 5/(2.0×5.0)=0.50→50%
_BLEND_DIVISOR = 5.0
_MAX_BLEND_REDUCTION = 0.60
# §v10.53 XCorr-Fenster: ±512 Samples ≈ ±10.7 ms bei 48 kHz.
# Groß genug für robuste Korrelation, klein genug um einzelne Transienten zu isolieren.
_XCORR_HALF_WINDOW: int = 512
# Minimale RMS-Energie im Fenster für valide Cross-Correlation.
_MIN_WINDOW_RMS: float = 1e-6


@dataclass
class TransientShiftResult:
    """Ergebnis der Transient-Shift-Detektion.

    Attributes:
        max_shift_ms: Maximale Onset-Verschiebung in ms (positiv = nach vorne = Pre-Echo).
        onset_count: Anzahl erkannter Onsets.
        ok: True wenn max_shift_ms <= 2.0 ms.
        blend_reduction: Empfohlene Wet-Reduktion (0.0–1.0, 0 = kein Eingriff).
    """

    max_shift_ms: float
    onset_count: int
    ok: bool
    blend_reduction: float = 0.0
    shifts_ms: list[float] = field(default_factory=list)


def _detect_onsets_simple(audio_mono: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    """Onset-Detektion via Spektralfluss — nur zur Lokalisierung von Messpunkten.

    Die erkannten Positionen dienen als Kandidaten für die Cross-Correlation-Messung.
    Falsch-positive oder falsch-negative Onsets sind unkritisch:
    - Falsch-positive: XCorr zeigt trotzdem ≈0 ms Shift (kein Schaden).
    - Falsch-negative: Weniger Messpunkte, aber die echten Transienten werden erfasst.
    """
    try:
        if librosa is None:
            raise RuntimeError("librosa nicht verfügbar")

        onsets = librosa.onset.onset_detect(y=audio_mono, sr=sr, hop_length=hop, units="samples", backtrack=True)  # type: ignore[attr-defined]
        return np.asarray(onsets, dtype=np.int64)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("transient_guard.py::_erkennen_onsets_simple Ersatzpfad: %s", e)

    # Fallback: Differenz der Frame-Energie
    frame_len = hop
    n = len(audio_mono)
    energies = []
    for i in range(0, n - frame_len, frame_len):
        energies.append(float(np.sum(audio_mono[i : i + frame_len] ** 2)))
    energies = np.array(energies, dtype=np.float32)  # type: ignore[assignment]
    diff = np.diff(energies, prepend=energies[:1])
    threshold = float(np.mean(diff) + 1.5 * np.std(diff))
    onset_frames = np.where(diff > threshold)[0]
    return (onset_frames * frame_len).astype(np.int64)  # type: ignore[no-any-return]


def _xcorr_shift_at(
    pre_mono: np.ndarray,
    post_mono: np.ndarray,
    center_sample: int,
    half_window: int,
    sr: int,
) -> float | None:
    """Cross-Correlation-Shift an einer Onset-Position.

    Extrahiert ein Fenster (±half_window) um center_sample aus pre und post,
    normalisiert beide und berechnet die Cross-Correlation. Der Lag des
    XCorr-Peaks ist der echte Zeitversatz in ms.

    Returns:
        Shift in ms (positiv = post später als pre), oder None bei ungültigem Fenster.
    """
    start = center_sample - half_window
    end = center_sample + half_window
    n = len(pre_mono)
    if start < 0 or end > n:
        return None

    pre_win = pre_mono[start:end].astype(np.float64)
    post_win = post_mono[start:end].astype(np.float64)

    # Minimum-Energie-Check: stille Fenster liefern keine sinnvolle Korrelation
    rms = float(np.sqrt(np.mean(pre_win**2) + 1e-12))
    if rms < _MIN_WINDOW_RMS:
        return None

    # Normalisierung (zero-mean, unit-variance)
    eps = 1e-12
    pre_norm = (pre_win - pre_win.mean()) / (pre_win.std() + eps)
    post_norm = (post_win - post_win.mean()) / (post_win.std() + eps)

    # Cross-Correlation: np.correlate(post, pre, 'full') → Peak-Position relativ zur Mitte
    xcorr = np.correlate(post_norm, pre_norm, mode="full")
    lag_samples = int(np.argmax(xcorr)) - (len(pre_win) - 1)

    return float(lag_samples) / sr * 1000.0


def detect_transient_shifts(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
) -> TransientShiftResult:
    """Erkennt zeitliche Verschiebungen von Transient-Onsets via Cross-Correlation.

    Methode (§v10.53):
    1. Onsets in pre via Spektralfluss detektieren (nur zur Positionsbestimmung).
    2. Für jeden Onset: ±10.7 ms Fenster aus pre und post extrahieren.
    3. Normalisierte Cross-Correlation → Lag des Peaks = echter Zeitversatz.
    4. Maximalen |Shift| über alle Onsets reporten.

    XCorr ist unempfindlich gegenüber spektralen Änderungen (EQ, Presence-Boost,
    Harmonic-Restauration) und misst ausschließlich Zeitbereichs-Verschiebungen.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase (same shape as pre).
        sr: Sample-Rate (muss 48000 sein).

    Returns:
        TransientShiftResult. ok=False wenn max_shift_ms > 3.5 ms.
    """
    assert sr == 48000
    _fallback = TransientShiftResult(max_shift_ms=0.0, onset_count=0, ok=True, blend_reduction=0.0)

    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        pre_mono = pre.mean(axis=0) if pre.ndim == 2 else pre
        post_mono = post.mean(axis=0) if post.ndim == 2 else post

        if len(pre_mono) < 512:
            return _fallback

        hop = 256
        # Onsets nur in pre — sie dienen als Messpunkte für die XCorr
        pre_onsets = _detect_onsets_simple(pre_mono, sr, hop)

        if len(pre_onsets) == 0:
            return _fallback

        shifts_ms: list[float] = []

        for onset_pre in pre_onsets:
            shift = _xcorr_shift_at(pre_mono, post_mono, int(onset_pre), _XCORR_HALF_WINDOW, sr)
            if shift is not None:
                shifts_ms.append(shift)

        if not shifts_ms:
            return _fallback

        max_shift = float(np.max(np.abs(shifts_ms)))
        ok = max_shift <= TRANSIENT_SHIFT_THRESHOLD_MS

        # Blend-Reduktion: proportional zur Überschreitung
        blend_reduction = 0.0
        if not ok:
            blend_reduction = float(
                np.clip(max_shift / (TRANSIENT_SHIFT_THRESHOLD_MS * _BLEND_DIVISOR), 0.0, _MAX_BLEND_REDUCTION)
            )
            logger.info(
                "§V22 Pre-Echo (XCorr): max_shift=%.2f ms > %.0f ms → blend_reduction=%.2f",
                max_shift,
                TRANSIENT_SHIFT_THRESHOLD_MS,
                blend_reduction,
            )

        return TransientShiftResult(
            max_shift_ms=round(max_shift, 3),
            onset_count=len(pre_onsets),
            ok=ok,
            blend_reduction=round(blend_reduction, 3),
            shifts_ms=[round(s, 3) for s in shifts_ms[:10]],  # max 10 für Metadata
        )

    except Exception as exc:
        logger.debug("erkennen_transient_shifts nicht blockierend: %s", exc)
        return _fallback


def compute_transient_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """§v10.303.13: Energie-Delta-basierte Transienten-Maske.

    Detektiert Frames mit starkem Energie-Anstieg (+3dB) als Transienten
    (Onsets, Attacks, Klicks). Diese Frames sollten in subtraktiven Phasen
    (Denoise, Dehiss, Dereverb) weniger aggressiv bearbeitet werden, um
    Groove und Mikrodynamik zu erhalten.

    Returns:
        Float-Array [0.0, 1.0] mit Länge = Anzahl Frames.
        1.0 = sicherer Transient, 0.0 = kein Transient.
    """
    # STFT-Parameter
    _n_fft = 2048
    _hop = _n_fft // 4  # 512 samples = 10.7ms @48kHz

    _mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
    _n_frames = 1 + (len(_mono) - _n_fft) // _hop
    if _n_frames < 4:
        return cast(np.ndarray, (np.zeros(max(1, _n_frames), dtype=np.float32)))

    # Energie pro Frame
    _energy = np.array([float(np.mean(_mono[i * _hop : i * _hop + _n_fft] ** 2)) for i in range(_n_frames)])
    _energy_db = 10.0 * np.log10(_energy + 1e-12)
    # Energie-Delta: +3dB Anstieg = Onset
    _delta = np.diff(_energy_db, prepend=_energy_db[0])
    _mask = (_delta > 3.0).astype(np.float32)
    # Smooth über 3 Frames (32ms) für natürliche Übergänge
    _mask = np.convolve(_mask, np.ones(3) / 3, mode="same")
    return cast(np.ndarray, (np.clip(_mask, 0.0, 1.0).astype(np.float32)))
