"""
§0p Adaptiver Vibrato-Detektor — Aurik 10

Zweck: Robuste Erkennung von Vibrato-Raten (3–7 Hz) für historische Gesangsaufnahmen.
Historische Vibrato-Raten variieren je nach Ära und Stil (3 Hz barock, 5–7 Hz modern).

Nutzt adaptive Frequenzbandbreite basierend auf Material-Typ und Era-Decade.

Usage:
    from backend.core.vibrato_detector import detect_vibrato_rate, VibratoDetectionResult

    result = detect_vibrato_rate(audio, sr=48000, era_decade=1920)
    if result.rate_hz > 0:
        # Vibrato erkannt — Rate für Guard verwenden
        ...
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Historische Vibrato-Raten nach Ära (psychoakustisch kalibriert) ──────

_VIBRATO_RATES_BY_ERA: dict[str, tuple[float, float]] = {
    "baroque": (3.0, 4.5),     # Barock: langsames Vibrato
    "classical": (4.0, 5.5),   # Klassik: moderates Vibrato
    "romantic": (5.0, 6.5),    # Romantik: etwas schneller
    "modern": (5.5, 7.0),      # Modern: volles Vibrato
}


@dataclass
class VibratoDetectionResult:
    """Ergebnis der Vibrato-Erkennung.

    Attributes:
        rate_hz: Geschätzte Vibrato-Rate in Hz (0 wenn kein Vibrato).
        depth_hz: F0-Modulationstiefe in Hz (max-min F0).
        confidence: Erkennungs-Konfidenz [0, 1].
        is_vibrato: True wenn rate_hz im erwarteten Bereich liegt.
    """

    rate_hz: float
    depth_hz: float
    confidence: float
    is_vibrato: bool


def _estimate_f0_temporal(
    mono: np.ndarray,
    sr: int,
    window_ms: float = 150.0,
    hop_ms: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Schätzt frame-weise F0 via Autokorrelation (temporal).

    Args:
        mono: Mono-Audio (float64).
        sr: Sample-Rate in Hz.
        window_ms: Fensterlänge in ms.
        hop_ms: Hop-Schrittweite in ms.

    Returns:
        (f0_frames, times) — F0-Werte und Zeitstempel in Sekunden.
    """
    win_len = int(window_ms / 1000.0 * sr)
    hop_len = int(hop_ms / 1000.0 * sr)

    n = len(mono)
    n_frames = max(1, (n - win_len) // hop_len + 1)

    f0s = np.zeros(n_frames, dtype=np.float32)
    times = np.linspace(0.0, n / sr, n_frames)

    for i in range(n_frames):
        start = i * hop_len
        frame = mono[start : start + win_len]

        if len(frame) < 64:
            continue

        # Energie-Check
        if float(np.abs(frame).max()) < 1e-4:
            continue

        # FFT-basierte Autokorrelation (O(N log N))
        frame_f64 = frame.astype(np.float64)
        fft_len = 2 * len(frame_f64)
        spectrum = np.fft.rfft(frame_f64, n=fft_len)
        acf = np.fft.irfft(np.abs(spectrum) ** 2).real
        acf_norm = acf / (acf[0] + 1e-10)

        # Pitch-Lag-Bereich: F0 80–500 Hz
        lag_min = max(1, int(sr / 500.0))
        lag_max = min(len(acf) - 1, int(sr / 80.0))

        if lag_min >= lag_max:
            continue

        acf_window = acf_norm[lag_min:lag_max]
        peak_idx = int(np.argmax(acf_window))
        r_t0 = float(acf_window[peak_idx])

        if r_t0 > 0.3:
            period = peak_idx + lag_min
            f0s[i] = float(sr) / period

    return f0s, times


def _estimate_vibrato_rate(
    f0_frames: np.ndarray,
    sr: int,
    expected_lo_hz: float = 3.0,
    expected_hi_hz: float = 7.0,
) -> tuple[float, float, float]:
    """Schätzt Vibrato-Rate aus F0-Modulation via spektrale Analyse.

    Nutzt FFT der F0-Kurve im erwarteten Vibrato-Bereich (3–7 Hz).

    Returns:
        (rate_hz, depth_hz, confidence)
    """
    # Nur stimmhafte Frames (F0 > 50 Hz)
    voiced = f0_frames[f0_frames > 50.0]

    if len(voiced) < 20:
        return 0.0, 0.0, 0.0

    # FFT der F0-Kurve
    n_fft = min(len(voiced), 1024)
    f0_seg = voiced[:n_fft].astype(np.float64)

    # DC-Komponente entfernen (mittlere F0)
    f0_detrended = f0_seg - np.mean(f0_seg)

    # Hanning-Fenster
    window = np.hanning(len(f0_detrended))
    f0_windowed = f0_detrended * window

    # FFT — Frequenzachse in Hz (F0-Sample-Rate ≈ sr / hop_len)
    f0_sr = sr / int(50.0 / 1000.0 * sr)  # ≈ 20 Hz bei 50ms Hop
    spectrum = np.abs(np.fft.rfft(f0_windowed))
    freqs = np.fft.rfftfreq(len(f0_windowed), d=1.0 / f0_sr)

    # Suche Peak im Vibrato-Bereich (3–7 Hz)
    vibrato_mask = (freqs >= expected_lo_hz) & (freqs <= expected_hi_hz)
    if not np.any(vibrato_mask):
        return 0.0, 0.0, 0.0

    vibrato_spectrum = spectrum[vibrato_mask]
    vibrato_freqs = freqs[vibrato_mask]

    peak_idx = int(np.argmax(vibrato_spectrum))
    rate_hz = float(vibrato_freqs[peak_idx])
    peak_magnitude = float(vibrato_spectrum[peak_idx])

    # Tiefe: max-min F0 in Hz
    depth_hz = float(np.ptp(f0_seg))  # peak-to-peak

    # Konfidenz: normalisierte Peak-Magnitude vs. Gesamte Energie
    total_energy = float(np.sum(spectrum**2))
    confidence = float(np.clip(peak_magnitude**2 / (total_energy + 1e-10), 0.0, 1.0))

    return rate_hz, depth_hz, confidence


def detect_vibrato_rate(
    audio: np.ndarray,
    sr: int,
    era_decade: int | None = None,
) -> VibratoDetectionResult:
    """Detektiert Vibrato-Rate mit adaptiver Frequenzbandbreite.

    Args:
        audio: Audio-Signal (float32/float64, Mono oder Stereo).
        sr: Sample-Rate in Hz.
        era_decade: Ära in Jahrzehnten (z. B. 1920). None → Default 3–7 Hz.

    Returns:
        VibratoDetectionResult mit Rate, Tiefe und Konfidenz.
    """
    # NaN/Inf-Schutz (§0a)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Mono
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    # Adaptive Frequenzbandbreite basierend auf Era-Decade
    if era_decade is not None:
        if era_decade < 1750:
            expected_lo, expected_hi = _VIBRATO_RATES_BY_ERA["baroque"]
        elif era_decade < 1820:
            expected_lo, expected_hi = _VIBRATO_RATES_BY_ERA["classical"]
        elif era_decade < 1900:
            expected_lo, expected_hi = _VIBRATO_RATES_BY_ERA["romantic"]
        else:
            expected_lo, expected_hi = _VIBRATO_RATES_BY_ERA["modern"]
    else:
        expected_lo, expected_hi = 3.0, 7.0

    # F0-Schätzung
    f0_frames, _times = _estimate_f0_temporal(mono, sr)

    # Vibrato-Rate schätzen
    rate_hz, depth_hz, confidence = _estimate_vibrato_rate(
        f0_frames, sr, expected_lo, expected_hi
    )

    is_vibrato = (expected_lo <= rate_hz <= expected_hi) and (depth_hz > 0.3) and (confidence > 0.15)

    result = VibratoDetectionResult(
        rate_hz=round(rate_hz, 2),
        depth_hz=round(depth_hz, 2),
        confidence=round(confidence, 3),
        is_vibrato=is_vibrato,
    )

    if is_vibrato:
        logger.debug(
            "§0p Vibrato-Detektor: Rate=%.1f Hz Tiefe=%.2f Hz Konfidenz=%.3f (era=%d)",
            rate_hz, depth_hz, confidence, era_decade or -1,
        )

    return result


# ── Thread-safe Singleton ────────────────────────────────────────────────

_detector_lock = threading.Lock()


def get_vibrato_detector():
    """Gibt die Vibrato-Detektor-Funktion zurück (stateless, §0p).

    Returns:
        Callable[[np.ndarray, int | None, int | None], VibratoDetectionResult]:
            detect_vibrato_rate-Funktion.
    """
    return detect_vibrato_rate
