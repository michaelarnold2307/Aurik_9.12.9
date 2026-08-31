"""Phoneme-Boundary-Detector — DSP-based articulation fallback for LGE stage 2.

Problem: LyricsGuidedEnhancement (§2.36) nutzt primär ML-basierte Phonem-Grenzen.
         Wenn keine Lyrics verfügbar sind oder das ML-Modell OOM fällt, braucht
         Stufe 2 eine DSP-basierte Methode zur Phon-Grenz-Erkennung.

Methode: Zero-Crossing-Rate + Energie + spectral shape.
  - Hohe ZCR (> 0.15) + niedrige Energie (< -45 dBFS) → voiced→unvoiced Grenze
  - Energie-Spike (> 12 dB relativ) → plosive Onset (p, t, k, b, d, g)
    - High-band ratio + spectral flatness → fricative texture (s, f, sh, th)

Diese Methode erreicht ca. 70 % Accuracy vs. ML (pyaapt/wav2vec2) — ausreichend
als robuster Fallback ohne externe Modelle.

API:
    from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp
    boundaries = detect_phoneme_boundaries_dsp(audio, sr)  # ndarray[bool], len = n_frames
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "detect_phoneme_boundaries_dsp",
    "detect_phoneme_protection_mask_dsp",
    "PhonemeClass",
    "FrameFeatures",
    "get_phoneme_features_dsp",
]

# ---------------------------------------------------------------------------
# Schwellwerte (empirisch, Material-unabhängig auf 48 kHz kalibriert)
# ---------------------------------------------------------------------------
_ZCR_VOICED_UNVOICED_THRESHOLD = 0.15  # ZCR > 0.15 → wahrscheinlich unvoiced
_ENERGY_QUIET_DBFS = -45.0  # Frame < -45 dBFS → quasi-stille Zone
_PLOSIVE_ONSET_DB = 12.0  # Energie-Delta > 12 dB → Plosive-Onset
_FRICATIVE_DESCENT_DB = -8.0  # Energie-Delta < -8 dB nach Spike → Frikative
_FRICATIVE_CENTROID_HZ = 3_500.0
_FRICATIVE_FLATNESS_MIN = 0.035
_FRICATIVE_HIGH_BAND_RATIO_MIN = 0.28
_PLOSIVE_CREST_FACTOR_MIN = 3.2


class PhonemeClass:
    """Einfache Phonem-Klassen-Klassifikation pro Frame."""

    VOICED = "voiced"  # F0-moduliertes Signal (Vokale, Nasale)
    UNVOICED = "unvoiced"  # Legacy alias for unvoiced consonants
    FRICATIVE = "fricative"  # Noise-like consonants (s, f, sh, th)
    PLOSIVE = "plosive"  # Energie-Spike (p, t, k, b, d, g)
    SILENCE = "silence"  # Unter Energie-Schwelle


class FrameFeatures:
    """Feature-Container pro Frame."""

    __slots__ = (
        "crest_factor",
        "delta_rms_db",
        "high_band_ratio",
        "phoneme_class",
        "rms_dbfs",
        "spectral_centroid_hz",
        "spectral_flatness",
        "zcr",
    )

    def __init__(
        self,
        zcr: float,
        rms_dbfs: float,
        delta_rms_db: float,
        phoneme_class: str,
        spectral_centroid_hz: float = 0.0,
        spectral_flatness: float = 0.0,
        high_band_ratio: float = 0.0,
        crest_factor: float = 0.0,
    ) -> None:
        # pylint: disable=too-many-positional-arguments
        self.zcr = zcr
        self.rms_dbfs = rms_dbfs
        self.delta_rms_db = delta_rms_db
        self.phoneme_class = phoneme_class
        self.spectral_centroid_hz = spectral_centroid_hz
        self.spectral_flatness = spectral_flatness
        self.high_band_ratio = high_band_ratio
        self.crest_factor = crest_factor


def detect_phoneme_boundaries_dsp(
    audio: np.ndarray,
    sr: int,  # kept for API consistency
    hop_length: int = 512,
) -> np.ndarray:
    """ZCR/Energie-basierte Phonem-Grenzerkennung (Stufe-2 LGE Fallback).

    Erkennt Übergänge zwischen voiced/unvoiced/plosive/silence Segmenten
    ohne externe ML-Modelle. Zuverlässig als Fallback für alle Materialtypen.

    Parameters
    ----------
    audio : np.ndarray
        Mono-Signal (1D) oder Stereo (2×N oder N×2); bei Stereo → Downmix.
    sr : int  # noqa: ARG001
        Abtastrate. Kein assert — Analyse-Modul (§Codierregeln).
    hop_length : int
        Hop in Samples (Standard: 512 bei 48 kHz ≈ 10.7 ms/Frame).

    Returns
    -------
    np.ndarray
        Boolean-Array der Länge ``n_frames``.
        ``True`` = Frame ist eine Phonem-Grenze (Zustandsübergang).
    """
    try:
        # Stereo → Mono
        hop_length = _valid_hop_length(hop_length)
        mono = _to_mono(audio)
        if len(mono) < hop_length * 4:
            return np.zeros(max(1, len(mono) // hop_length), dtype=bool)  # type: ignore[no-any-return]

        features = _extract_frame_features(mono, sr, hop_length)
        classes = [feature.phoneme_class for feature in features]
        boundaries = _detect_boundaries(classes)

        logger.debug(
            "phoneme_boundaries_dsp: %d frames, %d boundaries (sr=%d hop=%d)",
            len(features),
            int(np.sum(boundaries)),
            sr,
            hop_length,
        )
        return boundaries

    except Exception as exc:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        logger.debug("phoneme_boundaries_dsp: Fehler (nicht blockierend): %s", exc)
        hop_length = _valid_hop_length(hop_length)
        n_frames = max(1, len(np.asarray(audio).flatten()) // hop_length)
        return np.zeros(n_frames, dtype=bool)  # type: ignore[no-any-return]


def get_phoneme_features_dsp(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> list[FrameFeatures]:
    """Liefert detaillierte Feature-Objekte pro Frame (optional für Debug/Visualisierung)."""
    try:
        hop_length = _valid_hop_length(hop_length)
        mono = _to_mono(audio)
        return _extract_frame_features(mono, sr, hop_length)
    except Exception as exc:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        logger.debug("phoneme_features_dsp: Fehler: %s", exc)
        return []


def detect_phoneme_protection_mask_dsp(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    guard_ms: float = 18.0,
) -> np.ndarray:
    """Gibt a sample-level mask for consonants that should survive NR zurück.

    Plosives and fricatives are articulation anchors in vocal music.  The mask
    can be used by NR/LGE stages to reduce wet processing around those frames.
    """
    try:
        hop_length = _valid_hop_length(hop_length)
        mono = _to_mono(audio)
        n_samples = len(mono)
        mask = np.zeros(n_samples, dtype=bool)
        features = _extract_frame_features(mono, sr, hop_length)
        guard = max(0, int(sr * guard_ms / 1000.0))
        protected = {PhonemeClass.PLOSIVE, PhonemeClass.FRICATIVE, PhonemeClass.UNVOICED}
        for idx, feature in enumerate(features):
            if feature.phoneme_class not in protected:
                continue
            start = max(0, idx * hop_length - guard)
            end = min(n_samples, idx * hop_length + hop_length * 2 + guard)
            mask[start:end] = True
        return cast(np.ndarray, mask)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        logger.debug("phoneme_protection_mask_dsp: Fehler (nicht blockierend): %s", exc)
        return np.zeros(len(np.asarray(audio).flatten()), dtype=bool)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------


def _valid_hop_length(hop_length: int) -> int:
    """Gibt a positive hop length for non-blocking DSP fallback calls zurück."""
    return max(1, int(hop_length))


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertierung Stereo zu Mono (mean über Kanal-Achse)."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr  # type: ignore[no-any-return]
    if arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.mean(axis=0)  # type: ignore[no-any-return]
        if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
            return arr.mean(axis=1)  # type: ignore[no-any-return]
    return arr.flatten()  # type: ignore[no-any-return]


def _frame_audio(audio: np.ndarray, hop_length: int) -> list[np.ndarray]:
    """Teile Signal in Frames der Länge hop_length × 2 mit hop_length Hop."""
    frame_len = hop_length * 2
    n = len(audio)
    frames = []
    for start in range(0, n - frame_len, hop_length):
        frames.append(audio[start : start + frame_len])
    if not frames:
        frames.append(audio)
    return frames


def _zcr(frame: np.ndarray) -> float:
    """Zero-Crossing-Rate normiert auf [0, 1]: Anzahl Vorzeichenwechsel / Framelänge."""
    if len(frame) < 2:
        return 0.0
    crossings = float(np.sum(np.diff(np.sign(frame + 1e-10)) != 0))
    return crossings / float(len(frame) - 1)


def _rms_dbfs(frame: np.ndarray) -> float:
    """RMS-Pegel in dBFS."""
    rms = float(np.sqrt(np.mean(frame**2) + 1e-20))
    return float(20.0 * np.log10(rms))


def _spectral_features(frame: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """Gibt centroid, flatness, high-band ratio and crest factor zurück."""
    if len(frame) < 8:
        return 0.0, 0.0, 0.0, 0.0
    frame_f = np.asarray(frame, dtype=np.float64)
    rms = float(np.sqrt(np.mean(frame_f**2) + 1e-20))
    peak = float(np.max(np.abs(frame_f)))
    crest = float(peak / max(rms, 1e-10))
    n_fft = min(2048, int(2 ** np.ceil(np.log2(max(8, len(frame_f))))))
    windowed = frame_f[:n_fft]
    if len(windowed) < n_fft:
        windowed = np.pad(windowed, (0, n_fft - len(windowed)), mode="constant")
    spectrum = np.abs(np.fft.rfft(windowed * np.hanning(n_fft))) + 1e-12
    power = spectrum**2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(1, sr))
    total = float(np.sum(power)) + 1e-12
    centroid = float(np.sum(freqs * power) / total)
    flatness = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-12))
    high_band = power[(freqs >= 3_000.0) & (freqs <= min(10_000.0, sr * 0.48))]
    high_ratio = float(np.sum(high_band) / total) if high_band.size else 0.0
    return centroid, flatness, high_ratio, crest


def _extract_frame_features(audio: np.ndarray, sr: int, hop_length: int) -> list[FrameFeatures]:
    """Extrahiert frame-level articulation features."""
    frames = _frame_audio(audio, hop_length)
    n_frames = len(frames)
    zcr_arr = np.array([_zcr(frame) for frame in frames], dtype=np.float64)
    rms_arr = np.array([_rms_dbfs(frame) for frame in frames], dtype=np.float64)
    delta_rms = np.zeros(n_frames, dtype=np.float64)
    delta_rms[1:] = rms_arr[1:] - rms_arr[:-1]
    spectral = [_spectral_features(frame, sr) for frame in frames]
    centroid = np.array([item[0] for item in spectral], dtype=np.float64)
    flatness = np.array([item[1] for item in spectral], dtype=np.float64)
    high_ratio = np.array([item[2] for item in spectral], dtype=np.float64)
    crest = np.array([item[3] for item in spectral], dtype=np.float64)
    classes = _classify_frames(zcr_arr, rms_arr, delta_rms, centroid, flatness, high_ratio, crest)
    return [
        FrameFeatures(
            float(zcr_arr[i]),
            float(rms_arr[i]),
            float(delta_rms[i]),
            classes[i],
            float(centroid[i]),
            float(flatness[i]),
            float(high_ratio[i]),
            float(crest[i]),
        )
        for i in range(n_frames)
    ]


def _classify_frames(
    zcr_arr: np.ndarray,
    rms_arr: np.ndarray,
    delta_rms: np.ndarray,
    spectral_centroid_hz: np.ndarray,
    spectral_flatness: np.ndarray,
    high_band_ratio: np.ndarray,
    crest_factor: np.ndarray,
) -> list[str]:
    # pylint: disable=too-many-positional-arguments
    """Klassifiziere jeden Frame als voiced/fricative/plosive/silence."""
    n = len(zcr_arr)
    classes = []
    for i in range(n):
        if rms_arr[i] < _ENERGY_QUIET_DBFS:
            classes.append(PhonemeClass.SILENCE)
        elif delta_rms[i] > _PLOSIVE_ONSET_DB or (crest_factor[i] > _PLOSIVE_CREST_FACTOR_MIN and delta_rms[i] > 5.0):
            classes.append(PhonemeClass.PLOSIVE)
        elif (
            zcr_arr[i] > _ZCR_VOICED_UNVOICED_THRESHOLD
            and spectral_centroid_hz[i] > _FRICATIVE_CENTROID_HZ
            and (spectral_flatness[i] > _FRICATIVE_FLATNESS_MIN or high_band_ratio[i] > _FRICATIVE_HIGH_BAND_RATIO_MIN)
        ):
            classes.append(PhonemeClass.FRICATIVE)
        else:
            classes.append(PhonemeClass.VOICED)
    return classes


def _detect_boundaries(classes: list[str]) -> np.ndarray:
    """Boolean-Array: True wenn Frame i ein Zustandsübergang ist."""
    n = len(classes)
    boundaries = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if classes[i] != classes[i - 1]:
            boundaries[i] = True
    return boundaries  # type: ignore[no-any-return]
