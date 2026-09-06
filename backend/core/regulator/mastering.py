"""
SOTA-Mastering-Chain für AURIK: Modular, erweiterbar, produktiv.
Enthält: LUFS-Normalisierung, Multiband-Kompression, adaptives EQing, Limiter, Stereo-Enhancement.
"""

import logging
from collections.abc import Callable
from typing import Any, cast

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from backend.core.audio_utils import apply_musical_gain_envelope, limit_quiet_edge_boost

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

pghi_reconstruct: Callable[..., np.ndarray] | None
try:
    from dsp.pghi import pghi_reconstruct as _pghi_reconstruct

    pghi_reconstruct = _pghi_reconstruct
except ImportError:
    pghi_reconstruct = None


def lufs_normalize(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalisiert program loudness while protecting intentionally quiet edges."""
    if pyln is None:
        raise RuntimeError("pyloudnorm muss installiert sein für LUFS-Normalisierung.")
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    gain = target_lufs - loudness
    gain_linear = 10 ** (gain / 20)
    if gain_linear > 1.0005:
        # §2.45a-II v10: Soft-knee defaults (200 ms crossfade, 6 dB knee).
        normalized = apply_musical_gain_envelope(
            audio,
            gain_linear,
            gate_dbfs=-36.0,
            sr=sr,
            reference_for_gate=audio,
        )
        return limit_quiet_edge_boost(audio, normalized, sr=sr)
    out = cast(np.ndarray, np.asarray(audio * gain_linear, dtype=np.float32))
    return out


def multiband_compress(
    audio: np.ndarray, sr: int, bands=((20, 250), (250, 4000), (4000, 20000)), ratio=2.0
) -> np.ndarray:
    """Wendet einen einfachen Drei-Band-Abwärts-Kompressionspass an."""
    # Einfache Multiband-Kompression (SOTA: für Produktion durch spezialisierte Module ersetzen)
    out = np.zeros_like(audio)
    for low, high in bands:
        sos = butter(2, [low / (sr / 2), high / (sr / 2)], btype="band", output="sos")
        band = sosfilt(sos, audio)
        # Soft-Knee Kompression (vereinfachtes Modell)
        threshold = np.percentile(np.abs(band), 90)
        band = np.where(
            np.abs(band) > threshold,
            np.sign(band) * (threshold + (np.abs(band) - threshold) / ratio),
            band,
        )
        out += band
    return out  # type: ignore[no-any-return]


def adaptive_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Wendet an: spectral-balancing EQ with phase-preserving reconstruction."""
    # SOTA: spectral smoothing EQ — phase reconstruction via PGHI (§4.5 RELEASE_MUST)
    n_fft = 2048
    hop = n_fft // 4
    stft_full = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    S_orig = np.abs(stft_full)
    mean_spectrum = np.mean(S_orig, axis=1)
    target = np.median(mean_spectrum)
    gain = target / (mean_spectrum + 1e-6)
    S_eq = S_orig * gain[:, None]
    # PGHI phase reconstruction (§4.5 — Griffin-Lim als Fallback verboten: zerstört Phasenkohärenz)
    try:
        if pghi_reconstruct is None:
            raise ImportError("dsp.pghi.pghi_reconstruct unavailable")
        audio_eq = pghi_reconstruct(S_eq, sr=sr, win_size=n_fft, hop=hop, initial_phase=np.angle(stft_full))
    except Exception:
        # Phase-preserving iSTFT fallback — original phases aus stft_full beibehalten
        Zxx_eq = S_eq * np.exp(1j * np.angle(stft_full))
        audio_eq = librosa.istft(Zxx_eq, hop_length=hop, win_length=n_fft)
    return librosa.util.fix_length(audio_eq, size=len(audio))  # type: ignore[no-any-return]


def limiter(audio: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    """Wendet ausschließlich dämpfendes Peak-Limiting gegen das 99,9%-Perzentil an."""
    # True Peak Limiter (vereinfachte Version) — §2.45a Peak-Guard Conformity
    # Use percentile(99.9) to prevent single transient from blocking limiting
    # Only reduce gain if needed (never amplify - that would be compression, not limiting)
    peak = float(np.percentile(np.abs(audio), 99.9)) if len(audio) > 0 else 1e-8
    if peak > threshold:
        # Only apply reduction, never amplification
        audio = audio * (threshold / peak)
    return audio


def stereo_enhance(audio: np.ndarray, width: float = 1.1) -> np.ndarray:
    """Widen stereo side content while leaving mono material untouched."""
    # Stereo-Enhancement (nur für Stereo)
    if audio.ndim == 1:
        return audio
    mid = (audio[0] + audio[1]) / 2
    side = (audio[0] - audio[1]) / 2 * width
    left = mid + side
    right = mid - side
    return np.vstack([left, right])  # type: ignore[no-any-return]


def mastering_chain(audio: np.ndarray, sr: int, config: dict[str, Any] | None = None) -> np.ndarray:
    """
    Vollständige SOTA-Mastering-Chain.
    Args:
        audio: np.ndarray (mono oder stereo)
        sr: Sample-Rate
        config: optionale Parameter für einzelne Stufen
    Returns:
        gemastertes Audio
    """
    config = config or {}
    audio = lufs_normalize(audio, sr, target_lufs=config.get("target_lufs", -14.0))
    audio = multiband_compress(audio, sr)
    audio = adaptive_eq(audio, sr)
    audio = limiter(audio)
    if audio.ndim == 2:
        audio = stereo_enhance(audio)
    return audio


# ---------------------------------------------------------------------------
# Legacy-Funktionen (portiert aus backend/mastering.py) — Rückwärtskompatibilität
# ---------------------------------------------------------------------------


def dither(audio: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """
    TPDF-Dithering (Triangular Probability Density Function) for quantization
    noise shaping before bit-depth reduction.

    16-bit: ±1 LSB TPDF (two independent uniform distributions summed)
    24-bit: ±1 LSB TPDF at 24-bit resolution (1/8388608)
    32-bit float: no dithering required (no quantization noise)
    """
    if bit_depth == 16:
        # Echtes TPDF: Summe zweier unabhängiger Gleichverteilungen → Dreiecksverteilung
        lsb_16 = 1.0 / 32768.0
        noise = np.random.uniform(-lsb_16, lsb_16, size=audio.shape) + np.random.uniform(
            -lsb_16, lsb_16, size=audio.shape
        )
        return audio + noise  # type: ignore[no-any-return]
    elif bit_depth == 24:
        # TPDF für 24-Bit: 1 LSB = 1/2^23
        lsb_24 = 1.0 / 8_388_608.0
        noise = np.random.uniform(-lsb_24, lsb_24, size=audio.shape) + np.random.uniform(
            -lsb_24, lsb_24, size=audio.shape
        )
        return audio + noise  # type: ignore[no-any-return]
    # 32-bit float: kein Quantisierungsrauschen, kein Dithering nötig
    return audio


def simple_eq(
    audio: np.ndarray,
    sr: int,
    bass_gain_db: float = -2.0,
    treble_gain_db: float = 2.0,
) -> np.ndarray:
    """Einfacher EQ: Bass absenken, Höhen anheben (Butterworth-Shelf-Filter)."""
    # §2.51 zero-phase: sosfiltfilt statt sosfilt — Filterbänder werden zu audio addiert
    _n = audio.shape[-1] if hasattr(audio, "shape") else len(audio)
    sos_low = butter(2, 200 / (sr / 2), btype="low", output="sos")
    bass = sosfiltfilt(sos_low, audio) if _n >= 15 else sosfilt(sos_low, audio)
    audio = audio + (10 ** (bass_gain_db / 20) - 1) * bass
    sos_high = butter(2, 4000 / (sr / 2), btype="high", output="sos")
    treble = sosfiltfilt(sos_high, audio) if _n >= 15 else sosfilt(sos_high, audio)
    audio = audio + (10 ** (treble_gain_db / 20) - 1) * treble
    return audio


def simple_compressor(
    audio: np.ndarray,
    threshold: float = 0.6,
    ratio: float = 4.0,
    makeup_gain: float = 1.0,
) -> np.ndarray:
    """Einfacher Downward-Kompressor (Legacy; bevorzuge multiband_compress)."""
    abs_audio = np.abs(audio)
    over = abs_audio > threshold
    audio[over] = np.sign(audio[over]) * (threshold + (abs_audio[over] - threshold) / ratio)
    return audio * makeup_gain  # type: ignore[no-any-return]
logger = logging.getLogger(__name__)
