"""§v10 PIM-Utility — Einheitlicher Hook für alle Phasen.

Import: from backend.core.pim_phase_hook import apply_pim_intensity
"""

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


def apply_pim_intensity(
    kwargs: dict,
    phase_name: str,
    *,
    default_nr: float = 0.5,
    default_de_ess: float = 0.5,
    default_comp: float = 1.5,
) -> dict[str, float]:
    """§v10 Liest PIM-Intensitäts-Map aus kwargs und gibt kalibrierte Parameter zurück.

    Jede Phase ruft diese Funktion als ERSTES in ihrem process() auf.
    Wenn keine PIM-Map vorhanden ist, werden die Defaults zurückgegeben.

    Args:
        kwargs:     Die **kwargs der Phase (enthält ggf. pim_intensity_map)
        phase_name: Name der Phase für Logging (z.B. "phase_09_crackle")
        default_nr: Default NR-Stärke (wenn keine PIM-Map)
        default_de_ess: Default De-Ess-Stärke
        default_comp: Default Compression-Ratio

    Returns:
        dict mit 'nr_strength', 'de_ess_strength', 'compression_ratio',
        'nr_global', 'transient_protection'
    """
    pim = kwargs.get("pim_intensity_map")
    if pim is None:
        return {
            "nr_strength": default_nr,
            "de_ess_strength": default_de_ess,
            "compression_ratio": default_comp,
            "nr_global": 1.0,
            "transient_protection": 1.0,
        }

    # Per-Band-Intensitäten für die musikalisch kritischen Bänder abrufen
    nr_presence = pim.get_nr_strength("presence", "verse")
    nr_air = pim.get_nr_strength("air", "verse")
    nr_mid = pim.get_nr_strength("mid", "verse")
    nr_ultra = pim.get_nr_strength("ultra", "verse")
    nr_global = pim.global_modifiers.get("nr_global", 1.0)

    # Kalibrierte Werte (gewichteter Durchschnitt der kritischen Bänder)
    # Höhere Gewichtung für Presence (Gesang) und Mid (Gitarren/Keys)
    nr_strength = float(
        np.clip((nr_presence * 0.35 + nr_mid * 0.25 + nr_air * 0.25 + nr_ultra * 0.15) * nr_global, 0.05, 0.95)
    )

    # Transientenschutz aus dem Presence-Band ableiten
    transient_prot = float(
        pim.per_band.get("presence", type("X", (), {"transient_preserve": 1.0})()).transient_preserve
        if hasattr(pim.per_band.get("presence", None), "transient_preserve")
        else 1.0
    )

    logger.debug(
        "PIM→%s: nr=%.2f (presence=%.2f mid=%.2f air=%.2f global=%.2f)",
        phase_name,
        nr_strength,
        nr_presence,
        nr_mid,
        nr_air,
        nr_global,
    )

    return {
        "nr_strength": nr_strength,
        "de_ess_strength": default_de_ess,
        "compression_ratio": default_comp,
        "nr_global": nr_global,
        "transient_protection": transient_prot,
    }


def compute_per_band_nr_mask(
    pim_map,
    sr: int,
    n_fft: int = 2048,
    *,
    section: str = "verse",
    max_attenuation_db: float = -40.0,
) -> np.ndarray:
    """§v10 Echte Per-Band-Gain-Maske aus der PIM-Intensity-Map.

    Erzeugt eine Frequenz-Gain-Kurve (dB), die pro FFT-Bin die PIM-
    kalibrierte NR-Stärke abbildet. Diese Maske wird NACH der Denoising-
    Stufe als spektrale Gewichtung angewandt.

    Bänder mit HOHER PIM-Intensität (z.B. ultra=0.65) bekommen STÄRKERE
    Dämpfung. Bänder mit NIEDRIGER PIM-Intensität (z.B. presence=0.20)
    werden GESCHONT — die Gain-Maske liegt nahe 0 dB.

    Args:
        pim_map:            PIM IntensityMap-Objekt
        sr:                 Sample-Rate
        n_fft:              FFT-Größe
        section:            Song-Sektion ("verse", "chorus", etc.)
        max_attenuation_db: Maximale Dämpfung in dB (Default -40 dB)

    Returns:
        gain_db: np.ndarray der Länge n_fft//2+1 mit Gain in dB pro FFT-Bin
    """
    from backend.core.perceptual_intensity_mapper import CRITICAL_BANDS

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    gain_db = np.zeros(len(freqs), dtype=np.float32)

    for band_name, (lo, hi) in CRITICAL_BANDS.items():
        if hi >= sr / 2:
            continue
        nr_strength = pim_map.get_nr_strength(band_name, section)
        # NR-Stärke → Gain-Dämpfung: nr=0 → 0dB, nr=1 → max_attenuation_db
        band_gain_db = float(nr_strength * max_attenuation_db)
        mask = (freqs >= lo) & (freqs < hi)
        gain_db[mask] = band_gain_db

    # Smoothing: Gauss-Filter über die Band-Grenzen (3 FFT-Bins)
    from scipy.ndimage import gaussian_filter1d

    gain_db = gaussian_filter1d(gain_db.astype(np.float64), sigma=1.5).astype(np.float32)

    return gain_db  # type: ignore[no-any-return]


def apply_per_band_mask(
    audio: np.ndarray,
    gain_db: np.ndarray,
    sr: int,
    *,
    mix: float = 0.7,
) -> np.ndarray:
    """§v10 Wendet die Per-Band-Gain-Maske auf Audio an (STFT-Domain).

    Args:
        audio:   Mono oder Stereo Audio (float32/64)
        gain_db: Gain-Maske in dB pro FFT-Bin
        sr:      Sample-Rate
        mix:     Wet/Dry-Mix (0.0 = Original, 1.0 = voll maskiert)

    Returns:
        Audio mit per-Band kalibrierter spektraler Gewichtung
    """
    from scipy import signal as scipy_signal

    arr = np.asarray(audio, dtype=np.float64)
    n_fft = (len(gain_db) - 1) * 2
    gain_linear = 10.0 ** (gain_db / 20.0)

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        # STFT
        f, t, Zxx = scipy_signal.stft(ch, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        # Apply per-band gain
        gain_broadcast = gain_linear[: Zxx.shape[0], np.newaxis]
        Zxx_masked = Zxx * gain_broadcast
        # Inverse STFT
        _, reconstructed = scipy_signal.istft(Zxx_masked, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        # Trim to original length
        if len(reconstructed) > len(ch):
            reconstructed = reconstructed[: len(ch)]
        elif len(reconstructed) < len(ch):
            reconstructed = np.pad(reconstructed, (0, len(ch) - len(reconstructed)))
        return reconstructed  # type: ignore[no-any-return]

    if arr.ndim == 2 and arr.shape[1] <= 2:
        processed = np.zeros_like(arr)
        for c in range(arr.shape[1]):
            ch_dry = arr[:, c]
            ch_wet = _process_channel(ch_dry)
            processed[:, c] = ch_dry * (1.0 - mix) + ch_wet * mix
        return cast(np.ndarray, processed.astype(np.float32))
    elif arr.ndim == 2 and arr.shape[0] <= 2:
        processed = np.zeros_like(arr)
        for c in range(arr.shape[0]):
            ch_dry = arr[c, :]
            ch_wet = _process_channel(ch_dry)
            processed[c, :] = ch_dry * (1.0 - mix) + ch_wet * mix
        return cast(np.ndarray, processed.astype(np.float32))
    else:
        ch_dry = arr.ravel()
        ch_wet = _process_channel(ch_dry)
        return cast(np.ndarray, (ch_dry * (1.0 - mix) + ch_wet * mix).astype(np.float32))


def apply_pre_emphasis(audio, sr, freq_hz=2000, boost_db=6.0):
    """§v10 #5: Pre-Emphasis — Höhen vor NR anheben."""
    from scipy import signal

    sos = signal.butter(2, freq_hz, "highshelf", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)


def apply_de_emphasis(audio, sr, freq_hz=2000, cut_db=6.0):
    """§v10 #5: De-Emphasis — Höhen nach NR absenken."""
    from scipy import signal

    sos = signal.butter(2, freq_hz, "lowshelf", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)


def compute_loudness_adaptive_nr(audio, sr, base_nr=0.5):
    """§v10 #7: Loudness-adaptive NR — leise Passagen sanfter."""
    mono = (
        audio.mean(axis=-1)
        if audio.ndim > 1 and audio.shape[-1] <= 2
        else (audio.mean(axis=0) if audio.ndim > 1 else audio)
    )
    rms = float(np.sqrt(np.mean(np.asarray(mono) ** 2)) + 1e-12)
    rms_db = 20.0 * np.log10(rms)
    # Leise (<-30dBFS): NR auf 60% reduzieren. Laut (>-15dBFS): volle NR.
    loudness_factor = float(np.clip((rms_db + 30.0) / 15.0, 0.6, 1.0))
    return base_nr * loudness_factor
