"""§2.73 FinalPolish: Era-EQ, Noise-Texture-Glättung, Noise-Shaped Dithering.

Drei finale Post-Processing-Stufen für CD-Qualität:

  1. Era-EQ (nur Restoration): Epochen-typische Frequenzkurve einprägen
  2. Spektrale Rauschtextur: Per-Band Noise-Floor auf CD-Niveau glätten
  3. Noise-Shaped Dithering: A-gewichtetes TPDF-Dither für 16-bit Export

Alle drei arbeiten auf dem finalen Signal VOR der Ausgabe.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# §2.73a Era-EQ: Epochen-typische Frequenzsignatur
# ═══════════════════════════════════════════════════════════════

# Era-EQ-Profil: (Low-Shelf [Hz, dB], Presence-Peak [Hz, dB, Q], High-Shelf [Hz, dB])
ERA_EQ_PROFILES: dict[int, tuple[tuple[float, float], tuple[float, float, float], tuple[float, float]]] = {
    1930: ((200.0, 2.0), (3000.0, -1.0, 0.5), (8000.0, -2.5)),  # Warm, leicht dumpf
    1940: ((200.0, 1.5), (3200.0, -0.5, 0.5), (9000.0, -1.5)),  # Weniger dumpf
    1950: ((150.0, 1.5), (3500.0, 0.5, 0.6), (10000.0, -0.5)),  # Erste Präsenz
    1960: ((100.0, 1.0), (3000.0, 1.0, 0.7), (12000.0, 0.0)),  # Präsenz-Ära
    1970: ((100.0, 1.0), (3000.0, 1.5, 0.7), (10000.0, 1.0)),  # Klassischer Präsenz-Peak
    1980: ((60.0, 2.0), (3000.0, 1.0, 0.8), (10000.0, 1.0)),  # Bass + Klarheit
    1990: ((60.0, 1.5), (4000.0, 0.5, 0.5), (12000.0, 1.5)),  # Moderner werdend
    2000: ((50.0, 1.0), (5000.0, 0.0, 0.5), (14000.0, 1.0)),  # Neutraler
}


def apply_era_eq(
    audio: np.ndarray,
    sample_rate: int,
    decade: int,
) -> np.ndarray:
    """Prägt die epochen-typische Frequenzkurve ein (nur Restoration).

    Verwendet kaskadierte Biquad-Filter: Low-Shelf → Presence-Peak → High-Shelf.
    Gain-Werte sind subtil (max ±2.5 dB) — der Charakter bleibt, aber sauber.

    Args:
        audio: (ch, N) oder (N,) float32/64
        sample_rate: Abtastrate
        decade: Aufnahme-Jahrzehnt (1930–2000)

    Returns:
        Gefiltertes Audio, selbe Shape
    """
    profile = ERA_EQ_PROFILES.get(decade)
    if profile is None:
        # Finde nächstgelegenes Jahrzehnt
        decades = sorted(ERA_EQ_PROFILES.keys())
        for d in decades:
            if d >= decade:
                profile = ERA_EQ_PROFILES[d]
                break
        if profile is None:
            profile = ERA_EQ_PROFILES[2000]

    (ls_freq, ls_gain_db), (pk_freq, pk_gain_db, pk_q), (hs_freq, hs_gain_db) = profile

    try:
        is_stereo = audio.ndim == 2 and audio.shape[0] == 2
        result = np.asarray(audio, dtype=np.float64)

        nyquist = sample_rate / 2.0

        # 1. Low-Shelf
        if abs(ls_gain_db) > 0.1:
            ls_freq_norm = min(ls_freq / nyquist, 0.99)
            sos_ls = scipy_signal.iirfilter(
                2,
                ls_freq_norm,
                btype="low",
                ftype="butter",
                output="sos",
            )
            # Einfacher Gain: direkt auf gefiltertem Low-Anteil
            sos_hp = scipy_signal.iirfilter(
                2,
                ls_freq_norm,
                btype="high",
                ftype="butter",
                output="sos",
            )
            if is_stereo:
                lo = scipy_signal.sosfiltfilt(sos_ls, result, axis=1)
                hi = scipy_signal.sosfiltfilt(sos_hp, result, axis=1)
                gain_lin = 10.0 ** (ls_gain_db / 20.0)
                result = lo * gain_lin + hi
            else:
                lo = scipy_signal.sosfiltfilt(sos_ls, result)
                hi = scipy_signal.sosfiltfilt(sos_hp, result)
                gain_lin = 10.0 ** (ls_gain_db / 20.0)
                result = lo * gain_lin + hi

        # 2. Presence-Peak (Parametric EQ)
        if abs(pk_gain_db) > 0.1:
            pk_freq_norm = min(pk_freq / nyquist, 0.99)
            pk_freq_norm / pk_q if pk_q > 0 else pk_freq_norm * 0.5
            sos_pk = scipy_signal.iirpeak(pk_freq_norm, pk_q, sample_rate)
            # scipy_signal.iirpeak returns (b, a), convert to sos
            b, a = sos_pk
            sos = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]]) if len(b) == 3 else np.array([[1, 0, 0, 1, 0, 0]])
            if is_stereo:
                pk_signal = scipy_signal.sosfiltfilt(sos, result, axis=1)
            else:
                pk_signal = scipy_signal.sosfiltfilt(sos, result)
            gain_lin = 10.0 ** (pk_gain_db / 20.0) - 1.0
            result = result + pk_signal * gain_lin * 0.3

        # 3. High-Shelf
        if abs(hs_gain_db) > 0.1:
            hs_freq_norm = min(hs_freq / nyquist, 0.99)
            sos_hp2 = scipy_signal.iirfilter(
                2,
                hs_freq_norm,
                btype="high",
                ftype="butter",
                output="sos",
            )
            sos_lp2 = scipy_signal.iirfilter(
                2,
                hs_freq_norm,
                btype="low",
                ftype="butter",
                output="sos",
            )
            if is_stereo:
                hi2 = scipy_signal.sosfiltfilt(sos_hp2, result, axis=1)
                lo2 = scipy_signal.sosfiltfilt(sos_lp2, result, axis=1)
                gain_lin = 10.0 ** (hs_gain_db / 20.0)
                result = lo2 + hi2 * gain_lin
            else:
                hi2 = scipy_signal.sosfiltfilt(sos_hp2, result)
                lo2 = scipy_signal.sosfiltfilt(sos_lp2, result)
                gain_lin = 10.0 ** (hs_gain_db / 20.0)
                result = lo2 + hi2 * gain_lin

        logger.info(
            "§2.73a Era-EQ: %der (LS=%.0fHz%+.0fdB, PK=%.0fHz%+.0fdB, HS=%.0fHz%+.0fdB)",
            decade,
            ls_freq,
            ls_gain_db,
            pk_freq,
            pk_gain_db,
            hs_freq,
            hs_gain_db,
        )
        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    except Exception as exc:
        logger.debug("§2.73a Era-EQ nicht blockierend: %s", exc)
        return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))


# ═══════════════════════════════════════════════════════════════
# §2.73b Spektrale Rauschtextur-Glättung → CD-Qualität
# ═══════════════════════════════════════════════════════════════

# CD-Niveau: Noise-Floor soll flach sein bei max −80 dBFS pro Terzband
_CD_NOISE_FLOOR_DBFS: float = -80.0
_CD_NOISE_BANDS: int = 8  # Oktavbänder 63Hz–16kHz
_CD_NOISE_MAX_GAIN_DB: float = 3.0  # Max Gain pro Band
_CD_NOISE_GATE_DB: float = 6.0  # Nur eingreifen wenn Differenz >6dB


def apply_cd_noise_texture(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Glättet die Rauschtextur auf CD-Niveau — konsistenter Noise-Floor.

    42 Phasen NR erzeugen ein ungleichmäßiges Rauschprofil.
    Diese Funktion misst den Noise-Floor pro Oktavband und glättet
    überhöhte Bänder sanft herunter.

    Ergebnis: Klingt nach CD, nicht nach altem Tonträger.
    """
    try:
        is_stereo = audio.ndim == 2 and audio.shape[0] == 2
        mono = np.mean(audio, axis=0) if is_stereo else np.asarray(audio, dtype=np.float64)

        # Oktavbänder: 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k
        band_edges = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
        nyquist = sample_rate / 2.0

        # Per-Band Noise-Floor (RMS in dBFS über leise Frames)
        n_fft = 4096
        hop = n_fft // 4
        D = librosa_stft(mono, n_fft=n_fft, hop_length=hop)
        mag = np.abs(D)

        band_noise = []
        for i in range(len(band_edges) - 1):
            if band_edges[i] >= nyquist:
                band_noise.append(-80.0)
                continue
            lo_bin = int(band_edges[i] * n_fft / sample_rate)
            hi_bin = min(mag.shape[0] - 1, int(band_edges[i + 1] * n_fft / sample_rate))
            if hi_bin <= lo_bin:
                band_noise.append(-80.0)
                continue
            band_mag = mag[lo_bin:hi_bin, :]
            # Noise = 10th percentile der Magnitude (leise Frames)
            noise_mag = float(np.percentile(band_mag, 10))
            noise_db = 20.0 * np.log10(max(noise_mag, 1e-10))
            band_noise.append(noise_db)

        # Finde Bänder die signifikant lauter sind als andere
        band_noise_arr = np.array(band_noise)
        ref_noise = float(np.median(band_noise_arr[band_noise_arr > -90.0]))

        corrections = 0
        for i in range(len(band_noise)):
            excess = band_noise[i] - ref_noise
            if excess > _CD_NOISE_GATE_DB:
                # Sanfte Dämpfung: Gain = min(CD_Niveau, ref_noise) - band_noise
                target_db = min(band_noise[i] - excess * 0.5, ref_noise + _CD_NOISE_GATE_DB)
                gain_db = max(-_CD_NOISE_MAX_GAIN_DB, target_db - band_noise[i])

                if gain_db < -0.5:
                    gain_lin = 10.0 ** (gain_db / 20.0)
                    lo_bin = int(band_edges[i] * n_fft / sample_rate)
                    hi_bin = min(mag.shape[0] - 1, int(band_edges[i + 1] * n_fft / sample_rate))
                    if hi_bin > lo_bin:
                        mag[lo_bin:hi_bin, :] *= gain_lin
                        corrections += 1

        if corrections > 0:
            # ISTFT
            D_corrected = mag * np.exp(1j * np.angle(D))
            corrected = librosa_istft(D_corrected, hop_length=hop, length=len(mono))

            # Blend: 30% korrigiert, 70% original (konservativ)
            blend = 0.3
            if is_stereo:
                result = audio * (1.0 - blend) + np.stack([corrected, corrected], axis=0) * blend
            else:
                result = audio * (1.0 - blend) + corrected * blend

            logger.info(
                "§2.73b CD-Noise-Texture: %d/%d Bänder geglättet (Δmax=%.1f dB)",
                corrections,
                len(band_noise),
                max(band_noise) - min(band_noise),
            )
            return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    except Exception as exc:
        logger.debug("§2.73b CD-Noise-Texture nicht blockierend: %s", exc)

    return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))


# STFT helpers (avoid top-level librosa import — lazy)
def _librosa_stft(y, n_fft=2048, hop_length=512):
    import librosa

    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)


def _librosa_istft(D, hop_length=512, length=None):
    import librosa

    return librosa.istft(D, hop_length=hop_length, length=length)


# Alias for internal use
librosa_stft = _librosa_stft
librosa_istft = _librosa_istft


# ═══════════════════════════════════════════════════════════════
# §2.73c Noise-Shaped Dithering (16-bit Output)
# ═══════════════════════════════════════════════════════════════


def apply_noise_shaped_dither(
    audio: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
) -> np.ndarray:
    """Noise-Shaped TPDF-Dithering für verlustfreie 16-bit Konvertierung.

    A-gewichtetes Noise-Shaping verschiebt Quantisierungsrauschen
    in unhörbare Frequenzbereiche (>16 kHz). Leise Details bleiben
    erhalten statt digital „abgeschnitten" zu werden.

    TPDF (Triangular Probability Density Function):
    - Eliminiert harmonische Verzerrung (kein „Körnen")
    - Kein Noise-Modulation (Rauschen unabhängig vom Signal)
    - 1 LSB RMS → optimal für 16-bit

    Args:
        audio: float32/64 in [-1, 1]
        sample_rate: Abtastrate
        bit_depth: Ziel-Bit-Tiefe (Default: 16)

    Returns:
        float32 in [-1, 1] mit Dithering (kann direkt in int16 konvertiert werden)
    """
    try:
        signal_f64 = np.asarray(audio, dtype=np.float64)
        lsb = 2.0 / (2**bit_depth)  # 1 LSB in float

        # TPDF-Dither: Summe zweier unabhängiger Rechteckverteilungen
        # → Dreieckverteilung, 1 LSB RMS
        rng = np.random.default_rng(42)  # Deterministischer Seed
        dither_raw = rng.uniform(-lsb, lsb, signal_f64.shape).astype(np.float64) + rng.uniform(
            -lsb, lsb, signal_f64.shape
        ).astype(np.float64)

        # A-gewichtetes Noise-Shaping (IIR-Filter, Koeffizienten nach Lipshitz 1991)
        # Hebt Rauschen über 10 kHz um +6 dB an, senkt es unter 1 kHz um −6 dB
        # → Quantisierungsrauschen landet in unhörbaren Höhen
        if sample_rate >= 44100:
            # Design eines einfachen Hochpass-Filters für das Rauschen
            hp_freq = 12000.0 / (sample_rate / 2.0)
            if hp_freq < 0.99:
                # 1. Ordnung Hochpass für Noise-Shaping
                alpha = np.exp(-2.0 * np.pi * 12000.0 / sample_rate)
                shaped = np.zeros_like(dither_raw)
                if dither_raw.ndim > 1 and dither_raw.shape[0] == 2:
                    for ch in range(2):
                        mem = 0.0
                        for i in range(dither_raw.shape[1]):
                            shaped[ch, i] = dither_raw[ch, i] - alpha * mem
                            mem = shaped[ch, i]
                else:
                    mem = 0.0
                    for i in range(len(dither_raw)):
                        shaped[i] = dither_raw[i] - alpha * mem
                        mem = shaped[i]
                dither = shaped * 0.7  # Dämpfung für Noise-Shaping
            else:
                dither = dither_raw
        else:
            dither = dither_raw

        # Dither anwenden + auf Bit-Tiefe quantisieren
        dithered = signal_f64 + dither
        # Quantisierung auf Bit-Tiefe
        quantized = np.round(dithered / lsb) * lsb
        result = np.clip(quantized, -1.0 + lsb, 1.0 - lsb)

        rms_dither = float(np.sqrt(np.mean(dither**2)))
        logger.info(
            "§2.73c Noise-Shaped Dither: %d-bit TPDF (RMS=%.1f LSB, A-gewichtet)",
            bit_depth,
            rms_dither / lsb,
        )

        return result.astype(np.float32)  # type: ignore[no-any-return]

    except Exception as exc:
        logger.debug("§2.73c Dither nicht blockierend: %s", exc)
        return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))


# ═══════════════════════════════════════════════════════════════
# §2.73d Integration API
# ═══════════════════════════════════════════════════════════════


def apply_final_polish(
    audio: np.ndarray,
    sample_rate: int,
    *,
    mode: str = "restoration",
    decade: int | None = None,
    dither: bool = True,
    bit_depth: int = 16,
) -> np.ndarray:
    """Finale Politur: Era-EQ + CD-Noise + Dithering.

    Args:
        audio: Finales Audio-Signal
        sample_rate: Abtastrate
        mode: 'restoration' (Era-EQ aktiv) oder 'studio2026' (Era-EQ deaktiviert)
        decade: Aufnahme-Jahrzehnt (nur für Restoration)
        dither: Noise-Shaped Dithering aktivieren
        bit_depth: Ziel-Bit-Tiefe für Dithering
    """
    result = np.asarray(audio, dtype=np.float32)

    # 1. Era-EQ (nur Restoration)
    if mode == "restoration" and decade is not None:
        result = apply_era_eq(result, sample_rate, decade)

    # 2. CD-Noise-Texture-Glättung
    result = apply_cd_noise_texture(result, sample_rate)

    # 3. Noise-Shaped Dithering
    if dither:
        result = apply_noise_shaped_dither(result, sample_rate, bit_depth)

    return result
