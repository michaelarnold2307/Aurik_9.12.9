"""
Aurik 10.0.0 - Test Utilities
===========================

Zentrale Test-Utilities und Audio-Generatoren für die Aurik 10.0.0 Testsuite.
Wird von conftest.py und anderen Testmodulen importiert.

Author: Aurik Testing Team
Date: 2026
Version: 10.0.0
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

# ═══════════════════════════════════════════════════════════════════════════
# MATERIAL QUALITY SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

MATERIAL_QUALITY_SPECS: dict[str, dict] = {
    "PRISTINE": {
        "snr_min": 80.0,
        "snr_target": 90.0,
        "noise_level": 0.0001,
        "distortion_level": 0.0,
        "frequency_response_flat": True,
        "dynamic_range_db": 90,
        "description": "Studio-quality, pristine audio",
    },
    "EXCELLENT": {
        "snr_min": 60.0,
        "snr_target": 70.0,
        "noise_level": 0.001,
        "distortion_level": 0.001,
        "frequency_response_flat": True,
        "dynamic_range_db": 75,
        "description": "Excellent quality, minimal degradation",
    },
    "GOOD": {
        "snr_min": 40.0,
        "snr_target": 50.0,
        "noise_level": 0.005,
        "distortion_level": 0.005,
        "frequency_response_flat": False,
        "dynamic_range_db": 55,
        "description": "Good quality, some noise",
    },
    "FAIR": {
        "snr_min": 25.0,
        "snr_target": 35.0,
        "noise_level": 0.02,
        "distortion_level": 0.01,
        "frequency_response_flat": False,
        "dynamic_range_db": 40,
        "description": "Fair quality, noticeable degradation",
    },
    "POOR": {
        "snr_min": 15.0,
        "snr_target": 22.0,
        "noise_level": 0.05,
        "distortion_level": 0.03,
        "frequency_response_flat": False,
        "dynamic_range_db": 25,
        "description": "Poor quality, significant noise and distortion",
    },
    "VERY_POOR": {
        "snr_min": 5.0,
        "snr_target": 10.0,
        "noise_level": 0.15,
        "distortion_level": 0.08,
        "frequency_response_flat": False,
        "dynamic_range_db": 15,
        "description": "Very poor quality, severe degradation",
    },
    "EXTREME": {
        "snr_min": 0.0,
        "snr_target": 3.0,
        "noise_level": 0.40,
        "distortion_level": 0.20,
        "frequency_response_flat": False,
        "dynamic_range_db": 8,
        "description": "Extreme degradation, barely recoverable",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# MEDIUM-SPECIFIC THRESHOLDS
# SNR: Signal-to-Noise Ratio (dB) - expected minimum after restoration
# THD: Total Harmonic Distortion (%) - expected after restoration
# ═══════════════════════════════════════════════════════════════════════════

MEDIUM_SPECIFIC_THRESHOLDS: dict[str, dict] = {
    # Vinyl formats
    "VINYL_LP_STEREO": {
        "snr_min": 50.0,
        "thd_max": 1.0,
        "freq_low": 20,
        "freq_high": 20000,
        "noise_type": "crackling",
        "wow_flutter": True,
    },
    "VINYL_LP_MONO": {
        "snr_min": 45.0,
        "thd_max": 1.5,
        "freq_low": 30,
        "freq_high": 15000,
        "noise_type": "crackling",
        "wow_flutter": True,
    },
    "VINYL_45RPM": {
        "snr_min": 48.0,
        "thd_max": 1.2,
        "freq_low": 30,
        "freq_high": 18000,
        "noise_type": "crackling",
        "wow_flutter": True,
    },
    "VINYL_78RPM": {
        "snr_min": 35.0,
        "thd_max": 3.0,
        "freq_low": 100,
        "freq_high": 8000,
        "noise_type": "hiss_crackle",
        "wow_flutter": True,
    },
    # Cassette formats
    "CASSETTE_TYPE_I": {
        "snr_min": 52.0,
        "thd_max": 1.0,
        "freq_low": 40,
        "freq_high": 16000,
        "noise_type": "hiss",
        "wow_flutter": True,
    },
    "CASSETTE_TYPE_II": {
        "snr_min": 57.0,
        "thd_max": 0.8,
        "freq_low": 30,
        "freq_high": 17000,
        "noise_type": "hiss",
        "wow_flutter": True,
    },
    "CASSETTE_TYPE_IV": {
        "snr_min": 62.0,
        "thd_max": 0.5,
        "freq_low": 20,
        "freq_high": 18000,
        "noise_type": "low_hiss",
        "wow_flutter": True,
    },
    # Digital formats
    "CD_STANDARD": {
        "snr_min": 96.0,
        "thd_max": 0.001,
        "freq_low": 20,
        "freq_high": 22050,
        "noise_type": "quantization",
        "wow_flutter": False,
    },
    "CD_DAMAGED": {
        "snr_min": 70.0,
        "thd_max": 0.1,
        "freq_low": 20,
        "freq_high": 22050,
        "noise_type": "skip_dropout",
        "wow_flutter": False,
    },
    "MP3_320": {
        "snr_min": 90.0,
        "thd_max": 0.01,
        "freq_low": 20,
        "freq_high": 20000,
        "noise_type": "compression",
        "wow_flutter": False,
    },
    "MP3_128": {
        "snr_min": 75.0,
        "thd_max": 0.05,
        "freq_low": 20,
        "freq_high": 16000,
        "noise_type": "compression",
        "wow_flutter": False,
    },
    "MP3_64": {
        "snr_min": 55.0,
        "thd_max": 0.2,
        "freq_low": 20,
        "freq_high": 12000,
        "noise_type": "heavy_compression",
        "wow_flutter": False,
    },
    # High-resolution formats
    "SACD_DSD": {
        "snr_min": 120.0,
        "thd_max": 0.0001,
        "freq_low": 5,
        "freq_high": 100000,
        "noise_type": "none",
        "wow_flutter": False,
    },
    "DAT_DIGITAL": {
        "snr_min": 92.0,
        "thd_max": 0.002,
        "freq_low": 20,
        "freq_high": 22000,
        "noise_type": "quantization",
        "wow_flutter": False,
    },
    "LOSSLESS_FLAC": {
        "snr_min": 96.0,
        "thd_max": 0.001,
        "freq_low": 20,
        "freq_high": 24000,
        "noise_type": "none",
        "wow_flutter": False,
    },
    # Historic/cylinder formats
    "CYLINDER_EDISON": {
        "snr_min": 15.0,
        "thd_max": 10.0,
        "freq_low": 300,
        "freq_high": 3000,
        "noise_type": "extreme_hiss_scratch",
        "wow_flutter": True,
    },
    "CYLINDER_COLUMBIA": {
        "snr_min": 12.0,
        "thd_max": 12.0,
        "freq_low": 400,
        "freq_high": 2500,
        "noise_type": "extreme_hiss_scratch",
        "wow_flutter": True,
    },
    # Tape formats
    "TAPE_15IPS": {
        "snr_min": 65.0,
        "thd_max": 0.3,
        "freq_low": 20,
        "freq_high": 20000,
        "noise_type": "low_hiss",
        "wow_flutter": False,
    },
    "TAPE_7_5IPS": {
        "snr_min": 58.0,
        "thd_max": 0.6,
        "freq_low": 30,
        "freq_high": 16000,
        "noise_type": "hiss",
        "wow_flutter": True,
    },
    "TAPE_3_75IPS": {
        "snr_min": 48.0,
        "thd_max": 1.5,
        "freq_low": 60,
        "freq_high": 10000,
        "noise_type": "heavy_hiss",
        "wow_flutter": True,
    },
    "REEL_TO_REEL": {
        "snr_min": 62.0,
        "thd_max": 0.4,
        "freq_low": 20,
        "freq_high": 18000,
        "noise_type": "low_hiss",
        "wow_flutter": False,
    },
    # Broadcast formats
    "FM_STEREO": {
        "snr_min": 60.0,
        "thd_max": 0.5,
        "freq_low": 50,
        "freq_high": 15000,
        "noise_type": "rf_noise",
        "wow_flutter": False,
    },
    "AM_RADIO": {
        "snr_min": 40.0,
        "thd_max": 2.0,
        "freq_low": 100,
        "freq_high": 5000,
        "noise_type": "rf_noise_heavy",
        "wow_flutter": False,
    },
    "SHORTWAVE": {
        "snr_min": 20.0,
        "thd_max": 5.0,
        "freq_low": 200,
        "freq_high": 4000,
        "noise_type": "rf_interference",
        "wow_flutter": False,
    },
    # Streaming formats
    "STREAMING_HIGH": {
        "snr_min": 85.0,
        "thd_max": 0.02,
        "freq_low": 20,
        "freq_high": 20000,
        "noise_type": "compression",
        "wow_flutter": False,
    },
    "STREAMING_MEDIUM": {
        "snr_min": 70.0,
        "thd_max": 0.1,
        "freq_low": 20,
        "freq_high": 18000,
        "noise_type": "compression",
        "wow_flutter": False,
    },
    "STREAMING_LOW": {
        "snr_min": 55.0,
        "thd_max": 0.5,
        "freq_low": 20,
        "freq_high": 12000,
        "noise_type": "heavy_compression",
        "wow_flutter": False,
    },
    # Studio/production
    "STUDIO_24BIT": {
        "snr_min": 120.0,
        "thd_max": 0.0001,
        "freq_low": 10,
        "freq_high": 48000,
        "noise_type": "none",
        "wow_flutter": False,
    },
    "STUDIO_32BIT": {
        "snr_min": 144.0,
        "thd_max": 0.00001,
        "freq_low": 5,
        "freq_high": 96000,
        "noise_type": "none",
        "wow_flutter": False,
    },
    # Telefon/VoIP
    "TELEPHONE_PSTN": {
        "snr_min": 30.0,
        "thd_max": 3.0,
        "freq_low": 300,
        "freq_high": 3400,
        "noise_type": "telephone_noise",
        "wow_flutter": False,
    },
    "VOIP_COMPRESSED": {
        "snr_min": 40.0,
        "thd_max": 1.0,
        "freq_low": 100,
        "freq_high": 7000,
        "noise_type": "packet_loss",
        "wow_flutter": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# AUDIO GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def generate_audio_by_quality(
    quality_level: str,
    sr: int = 48000,
    duration: float = 2.0,
) -> np.ndarray:
    """
    Generiert synthetisches Audio mit charakteristischen Eigenschaften
    für ein bestimmtes Material-Qualitätslevel.

    Args:
        quality_level: Eines der MATERIAL_QUALITY_SPECS-Keys
        sr: Sample-Rate in Hz
        duration: Dauer in Sekunden

    Returns:
        Mono-Audio als np.ndarray, float32, Wertebereich [-1, 1]
    """
    if quality_level not in MATERIAL_QUALITY_SPECS:
        raise ValueError(
            f"Unbekanntes Quality-Level: {quality_level!r}. Verfügbar: {list(MATERIAL_QUALITY_SPECS.keys())}"
        )

    spec = MATERIAL_QUALITY_SPECS[quality_level]
    n_samples = int(sr * duration)
    # §AMRB-Seeding-Invariante: stabiler Seed, kein prozessabhängiges hash().
    import hashlib as _hl

    _seed = int.from_bytes(_hl.sha256(str(quality_level).encode()).digest()[:4], byteorder="big")
    rng = np.random.default_rng(seed=_seed)

    # Basiston: Mehrere Sinus-Komponenten (vereinfachtes Musiksignal)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = (
        0.5 * np.sin(2 * np.pi * 440.0 * t)  # A4
        + 0.3 * np.sin(2 * np.pi * 880.0 * t)  # A5
        + 0.2 * np.sin(2 * np.pi * 261.6 * t)  # C4
    )

    # Rauschen entsprechend dem Qualitätslevel
    noise_level = spec["noise_level"]
    noise = rng.normal(0.0, noise_level, n_samples)
    signal = signal + noise

    # Verzerrung hinzufügen
    distortion = spec["distortion_level"]
    if distortion > 0:
        signal = np.tanh(signal * (1.0 + distortion * 5)) / (1.0 + distortion * 2)

    # Normalisieren auf Peak ≤ 0.95
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.95

    return signal.astype(np.float32)  # type: ignore[no-any-return]


def generate_medium_specific_audio(
    medium_type: str,
    sr: int = 48000,
    duration: float = 2.0,
) -> np.ndarray:
    """
    Generiert synthetisches Audio mit mediumspezifischen Eigenschaften
    (Rauschen, Bandbegrenzung, Wow/Flutter etc.).

    Args:
        medium_type: Eines der MEDIUM_SPECIFIC_THRESHOLDS-Keys
        sr: Sample-Rate in Hz
        duration: Dauer in Sekunden

    Returns:
        Mono-Audio als np.ndarray, float32, Wertebereich [-1, 1]
    """
    if medium_type not in MEDIUM_SPECIFIC_THRESHOLDS:
        raise ValueError(
            f"Unbekannter Medium-Typ: {medium_type!r}. Verfügbar: {list(MEDIUM_SPECIFIC_THRESHOLDS.keys())}"
        )

    spec = MEDIUM_SPECIFIC_THRESHOLDS[medium_type]
    n_samples = int(sr * duration)
    # §AMRB-Seeding-Invariante: MD5-basierter Seed — kein hash() (prozessabhängig ohne PYTHONHASHSEED)
    import hashlib as _hl

    _seed = int(_hl.md5(str(medium_type).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed=_seed)

    # Basiston
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = (
        0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t) + 0.2 * np.sin(2 * np.pi * 261.6 * t)
    )

    # Bandbegrenzung (Tiefpass + Hochpass)
    freq_low = spec["freq_low"]
    freq_high = min(spec["freq_high"], sr // 2 - 100)

    if freq_low > 20:
        normalized_low = max(0.001, min(0.999, freq_low / (sr / 2)))
        sos_high = butter(4, normalized_low, btype="high", output="sos")
        signal = sosfiltfilt(sos_high, signal)

    if freq_high < sr // 2 - 100:
        normalized_high = max(0.001, min(0.999, freq_high / (sr / 2)))
        sos_low = butter(4, normalized_high, btype="low", output="sos")
        signal = sosfiltfilt(sos_low, signal)

    # Rauschen entsprechend dem Typ
    snr_min = spec["snr_min"]
    # Rauschstärke aus SNR berechnen (niedrigere SNR → mehr Rauschen)
    noise_power = 10 ** (-snr_min / 20) * 0.1
    noise = rng.normal(0.0, float(noise_power), n_samples)
    signal = signal + noise

    # Wow/Flutter (Pitch-Modulation) für analoge Medien
    if spec.get("wow_flutter"):
        flutter_rate = 3.0 + rng.uniform(0, 2)  # Hz
        flutter_depth = 0.003
        phase_mod = flutter_depth * np.sin(2 * np.pi * flutter_rate * t)
        # Vereinfachte Zeitstreckung via Winkel-Modulation (Näherung)
        t_mod = t + np.cumsum(phase_mod) / sr
        signal = np.interp(t_mod, t, signal)

    # Normalisieren
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.90

    return signal.astype(np.float32)  # type: ignore[no-any-return]


def get_expected_thresholds(medium_type: str) -> dict:
    """Gibt die erwarteten Schwellenwerte für einen Medium-Typ zurück."""
    if medium_type not in MEDIUM_SPECIFIC_THRESHOLDS:
        raise ValueError(f"Unbekannter Medium-Typ: {medium_type!r}")
    return MEDIUM_SPECIFIC_THRESHOLDS[medium_type]


def validate_musical_goals(
    audio: np.ndarray,
    sr: int,
    quality_level: str,
) -> bool:
    """
    Validiert ob ein Audio-Signal die Mindestanforderungen
    für ein bestimmtes Qualitätslevel erfüllt.

    Args:
        audio: Audio-Signal als np.ndarray
        sr: Sample-Rate
        quality_level: Material-Qualitätslevel

    Returns:
        True wenn die Anforderungen erfüllt sind
    """
    if quality_level not in MATERIAL_QUALITY_SPECS:
        raise ValueError(f"Unbekanntes Quality-Level: {quality_level!r}")

    MATERIAL_QUALITY_SPECS[quality_level]

    # Einfache SNR-Schätzung (RMS signal / RMS noise näherungsweise)
    signal_rms = float(np.sqrt(np.mean(audio**2)))
    if signal_rms < 1e-10:
        return False

    # Dynamikbereich prüfen
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return False

    # Signal hat Inhalt
    return True
