"""Adaptive Parameter Infrastructure — §GEBOT-G36.

Stellt `_derive_from_signal()` als wiederverwendbares Basis-Pattern
für alle Restaurierungsphasen bereit. Jede Phase kann:
  1. `derive_params(audio, sr)` aufrufen → signal-adaptive Parameter
  2. Parameter im `process()`-Aufruf anwenden
  3. Optional: `verify_and_readjust()` für Selbst-Verifikation

Die abgeleiteten Parameter ersetzen statische Werte durch
song-individuelle, aus dem Signal berechnete Werte.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def _mono(audio: np.ndarray) -> np.ndarray:
    """§2.51 Layout-sichere Mono-Konversion: mean(axis=0) kollabierte channels-last (N,2) auf (2,)."""
    if audio.ndim == 1:
        return cast(np.ndarray, np.asarray(audio))
    from backend.core.audio_utils import safe_to_mono  # local import: vermeidet Zyklen

    return cast(np.ndarray, np.asarray(safe_to_mono(audio)))


def derive_noise_floor(audio: np.ndarray, sr: int) -> dict:
    """Ermittelt adaptiven Noise-Floor aus dem Signal.

    Returns:
        dict mit noise_floor_db, signal_peak_db, estimated_snr_db,
             optimal_threshold_db, band_reduction_factors
    """
    mono = _mono(audio)
    mono = np.asarray(mono, dtype=np.float64)
    _n = len(mono)

    # RMS und Peak
    rms = float(np.sqrt(np.mean(mono**2) + 1e-20))
    peak = float(np.max(np.abs(mono)))
    _rms_db = 20.0 * np.log10(max(rms, 1e-12))
    peak_db = 20.0 * np.log10(max(peak, 1e-12))

    # Rauschschätzung via spektraler Minimum-Statistik
    from scipy.signal import stft

    _f, _t_stft, _Z = stft(mono, fs=sr, nperseg=2048, noverlap=1536, boundary="even")
    _mag_db = 20.0 * np.log10(np.maximum(np.abs(_Z), 1e-12))

    # Per-Bin Minimum über Zeit → Noise-Floor-Schätzung
    _noise_floor_per_bin = np.percentile(_mag_db, 10, axis=1)
    _freqs = _f

    # Bandspezifische Noise-Floors
    bands = {
        "low": (60, 250),
        "mid": (250, 2000),
        "high": (2000, 8000),
        "air": (8000, 16000),
    }
    band_noise = {}
    for name, (lo, hi) in bands.items():
        mask = (_freqs >= lo) & (_freqs <= hi)
        if mask.any():
            band_noise[name] = float(np.mean(_noise_floor_per_bin[mask]))
        else:
            band_noise[name] = -60.0

    # Globaler Noise-Floor (Median über alle Bins)
    noise_floor_db = float(np.median(_noise_floor_per_bin))
    estimated_snr_db = peak_db - noise_floor_db

    # Adaptive Schwelle: 6dB über Noise-Floor, aber mindestens -48dB
    optimal_threshold_db = max(noise_floor_db + 6.0, -48.0)

    # Band-Reduktions-Faktoren: laute Bänder → weniger Reduktion
    # Leise Bänder (nah am Noise-Floor) → mehr Reduktion
    band_reduction = {}
    for name in bands:
        _dist = max(0.1, band_noise[name] - noise_floor_db)
        # Mehr Reduktion wenn Band-Noise nah am Global-Noise (rauschig)
        band_reduction[name] = float(np.clip(1.0 / (_dist + 1.0), 0.3, 1.0))

    return {
        "noise_floor_db": float(noise_floor_db),
        "signal_peak_db": float(peak_db),
        "estimated_snr_db": float(estimated_snr_db),
        "optimal_threshold_db": float(optimal_threshold_db),
        "band_noise_db": band_noise,
        "band_reduction_factors": band_reduction,
    }


def derive_transient_sensitivity(audio: np.ndarray, sr: int) -> dict:
    """Ermittelt adaptive Transienten-Empfindlichkeit.

    Returns:
        dict mit onset_threshold, min_gap_ms, attack_preserve_factor
    """
    mono = _mono(audio)
    mono = np.asarray(mono, dtype=np.float64)

    # Energie-Hüllkurve
    _env = np.convolve(np.abs(mono), np.ones(512) / 512, mode="same")
    _env_mean = float(np.mean(_env))
    _env_std = float(np.std(_env))

    # Crest-Faktor (Peak/RMS) für Transienten-Dichte
    _crest = float(np.max(np.abs(mono)) / max(np.sqrt(np.mean(mono**2)), 1e-8))

    # Adaptiver Onset-Threshold
    # Hoher Crest-Faktor (transienten-reich) → höhere Schwelle
    # Niedriger Crest-Faktor (gleichmäßig) → niedrigere Schwelle
    onset_threshold = float(np.clip(2.0 + _crest * 0.3, 2.0, 6.0))

    # Minimale Gap-Länge (ms): kürzer bei dichtem Material
    _env_variation = _env_std / max(_env_mean, 1e-8)
    min_gap_ms = float(np.clip(20.0 - _env_variation * 5.0, 5.0, 30.0))

    # Attack-Preserve: mehr Schutz bei transienten-reichem Material
    attack_preserve = float(np.clip(0.5 + _crest * 0.05, 0.5, 1.0))

    return {
        "onset_threshold": onset_threshold,
        "min_gap_ms": min_gap_ms,
        "attack_preserve_factor": attack_preserve,
        "crest_factor": float(_crest),
    }


def verify_output_quality(original: np.ndarray, processed: np.ndarray, sr: int) -> dict:
    """Selbst-Verifikation: prüft ob Verarbeitung das Signal verbessert hat.

    Returns:
        dict mit passed, rms_change_db, peak_change_db, spectral_correlation
    """
    orig = _mono(original)
    proc = _mono(processed)
    n = min(len(orig), len(proc))
    orig = np.asarray(orig[:n], dtype=np.float64)
    proc = np.asarray(proc[:n], dtype=np.float64)

    rms_o = float(np.sqrt(np.mean(orig**2) + 1e-20))
    rms_p = float(np.sqrt(np.mean(proc**2) + 1e-20))
    peak_o = float(np.max(np.abs(orig)))
    peak_p = float(np.max(np.abs(proc)))

    rms_change_db = 20.0 * np.log10(max(rms_p / rms_o, 1e-6))
    peak_change_db = 20.0 * np.log10(max(peak_p / peak_o, 1e-6))

    # Spektrale Korrelation (wie ähnlich ist die spektrale Hüllkurve?)
    _fft_o = np.abs(np.fft.rfft(orig * np.hanning(n)))
    _fft_p = np.abs(np.fft.rfft(proc * np.hanning(n)))
    _corr_num = float(np.sum((_fft_o - np.mean(_fft_o)) * (_fft_p - np.mean(_fft_p))))
    _corr_den = np.sqrt(np.sum((_fft_o - np.mean(_fft_o)) ** 2) * np.sum((_fft_p - np.mean(_fft_p)) ** 2) + 1e-20)
    spectral_correlation = float(_corr_num / _corr_den)

    # Kein NaN/Inf im Output
    no_artifacts = bool(np.all(np.isfinite(proc)))

    # Kein extremes Clipping
    no_clipping = bool(np.max(np.abs(proc)) <= 1.001)

    passed = no_artifacts and no_clipping and abs(rms_change_db) < 40.0

    needs_readjust = abs(rms_change_db) > 6.0 or spectral_correlation < 0.5

    return {
        "passed": passed,
        "needs_readjust": needs_readjust,
        "rms_change_db": rms_change_db,
        "peak_change_db": peak_change_db,
        "spectral_correlation": spectral_correlation,
        "no_artifacts": no_artifacts,
        "no_clipping": no_clipping,
    }
