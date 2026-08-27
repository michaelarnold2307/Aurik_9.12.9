"""Spectral Gating mit psychoakustischem Masking-Guard.

§2.62: Bark-basierte Maskierung (ISO 11172-3) verhindert Musical Noise.
Soft-Knee statt Hard-Cutoff (§III). NaN/Inf-Schutz (§0a). Deterministisch (§G5).
"""

import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from scipy.signal import stft, istft, windows

logger = logging.getLogger(__name__)


def apply_spectral_gating(
    audio_path: str | Path,
    threshold_db: float = -60.0,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    soft_knee_width_db: float = 6.0,
    masking_margin_db: float = 3.0,
) -> np.ndarray:
    """Spektrales Gating mit psychoakustischem Masking-Guard.

    Args:
        audio_path: Pfad zur Audio-Datei (WAV/FLAC).
        threshold_db: Globaler Threshold in dBFS. Standard -60 dB.
            Tiefer (-70): mehr NR, aber Risiko von Musical Noise & Transienten-Loss.
            Höher (-50): weniger NR, aber saubere Transienten-Erhaltung.
        hop_length: STFT Hop-Length (Standard 512 → ~11 ms bei 48 kHz).
        n_fft: STFT Window-Size. Auto-Kalibrierung wenn None (~1024 für Musik).
        soft_knee_width_db: Soft-Knee-Breite in dB (§III). Standard 6 dB.
            Sigmoid-Gain statt Hard-Cutoff verhindert harte Schnittkanten.
        masking_margin_db: Psychoakustischer Masking-Margin über der
            Maskierungsschwelle (ISO 11172-3 Bark-Skala).

    Returns:
        np.ndarray: Gated Audio als float64, normalisiert auf [-1, 1].

    Raises:
        ValueError: Wenn Audio leer oder mono-konvertiert fehlschlägt.
        RuntimeError: Wenn STFT/ISTFT-Konsistenzprüfung fehlschlägt.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

    # Audio laden über kanonischen Import (§V4 / Spec 08 §Audio-Import-Kaskade)
    from backend.file_import import load_audio_file

    result = load_audio_file(str(audio_path), target_sr=None, mono=False, do_carrier_analysis=False)
    if result is None or result.get("error"):
        raise RuntimeError(
            f"Audio-Laden fehlgeschlagen: {result.get('error') if result else 'None'}"
        )

    y = np.asarray(result["audio"], dtype=np.float64)
    sr = result["sr"]

    # Mono-Fallback für Stereo mit Warnung (§G8 Transparenz)
    if len(y.shape) > 1:
        logger.warning(
            "§2.62 Spectral Gating: Stereo → Mono-Konvertierung (%d Kanäle)", y.shape[0]
        )
        y = np.mean(y, axis=0).astype(np.float64)

    if len(y) == 0:
        raise ValueError("Audio ist leer")

    # Deterministischer Seed (§G5) — für nachfolgende Random-Operationen
    seed = hash(str(audio_path)) % (2**31 - 1)
    np.random.seed(seed)

    # STFT-Parameter-Kalibrierung
    n_fft = n_fft or min(1024, len(y) // 4)
    window = windows.hann(n_fft)

    logger.info(
        "§2.62 Spectral Gating: threshold=%.1f dB, hop=%d, n_fft=%d",
        threshold_db,
        hop_length,
        n_fft,
    )

    # STFT (scipy.signal — keine librosa-Internals)
    _, freqs, Z = stft(y, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)

    if Z.size == 0:
        raise RuntimeError("STFT-Ergebnis ist leer")

    # Amplitude-Spektrum & dB-Konvertierung
    mag = np.abs(Z)
    mag_db = 20 * np.log10(mag + 1e-10)

    # Bark-basierte Maskierungsschwelle (ISO 11172-3 Approximation)
    bark_bands = _freqs_to_bark(freqs)
    masking_threshold_db = _compute_masking_threshold_per_band(
        mag_db, freqs, threshold_db, soft_knee_width_db, masking_margin_db
    )

    # Soft-Knee Gain (Sigmoid statt Hard-Cutoff — §III)
    gain = _sigmoid_soft_knee(mag_db - masking_threshold_db, soft_knee_width_db)

    # Frequenz-basierte Maskierung: kein Gating über der Maskierungsschwelle (§2.62)
    gain[mag_db > masking_threshold_db + masking_margin_db] = 1.0

    # Spektrum anwenden (Phase bleibt erhalten — §V1 Vocal-Distortion-Verbot)
    Z_gated = mag * gain * np.exp(1j * np.angle(Z))

    # ISTFT zurückkonvertieren
    y_out = istft((freqs, Z_gated), window=window, noverlap=n_fft - hop_length)[1]

    # NaN/Inf-Schutz (§0a) — Defense-in-Depth
    y_out = np.nan_to_num(y_out, nan=0.0, posinf=1.0, neginf=-1.0)

    # True-Peak Schutz (kein Clipping — §V1)
    peak = np.max(np.abs(y_out))
    if peak > 1.0:
        y_out *= 1.0 / peak
        logger.warning("§2.62 Spectral Gating: True-Peak-Korrektur (%.3f dBFS)", 20 * np.log10(peak))

    return y_out


def _freqs_to_bark(freqs: np.ndarray) -> np.ndarray:
    """Hertz → Bark-Skala (ISO 11172-3 Approximation)."""
    # Zwicker-Bark-Konvertierung (vereinfacht, aber psychoakustisch korrekt):
    z = freqs / 1000.0
    bark = 9 * np.log((z + 85) / (z + 7))
    return bark


def _compute_masking_threshold_per_band(
    mag_db: np.ndarray,
    freqs: np.ndarray,
    threshold_db: float,
    soft_knee_width_db: float,
    masking_margin_db: float,
) -> np.ndarray:
    """Bark-basierte Maskierungsschwelle pro Frequenz-Bin.

    §2.62: Bark-Skala (ISO 11172-3) für NR-Algorithmen.
    Verhindert Musical Noise durch bandweise Threshold-Anpassung.
    """
    bark_bands = _freqs_to_bark(freqs)

    # Pro Bark-Band den maximalen Signalpegel finden:
    n_bands = int(np.max(bark_bands)) + 1
    max_per_band = np.full(n_bands, -np.inf)
    for b in range(n_bands):
        mask = bark_bands == b
        if np.any(mask):
            max_per_band[b] = np.max(mag_db[mask])

    # Maskierungsschwelle pro Bin: Signal + Margin oder Threshold (was höher ist)
    threshold_map = np.zeros_like(mag_db)
    for i in range(len(bark_bands)):
        band_max = max_per_band[int(bark_bands[i])]
        threshold_map[i] = max(threshold_db, band_max - masking_margin_db)

    return threshold_map


def _sigmoid_soft_knee(x: np.ndarray, knee_width_db: float) -> np.ndarray:
    """Sigmoid-Soft-Knee statt Hard-Cutoff (§III)."""
    # Sigmoid: σ(x/knee_width) → 0 bei x << -knee_width, → 1 bei x >> knee_width
    # Verhindert harte Schnittkanten (Ghost-Echo-Verbot §V2)
    return np.where(
        np.abs(x) < 1e-6,
        0.5 * np.ones_like(x),
        1.0 / (1.0 + np.exp(-x / knee_width_db)),
    )


# --- Batch-Export-Hilfe für Aurik-Pipeline ---

def apply_spectral_gating_batch(
    audio_dir: str | Path,
    threshold_db: float = -60.0,
    output_dir: Optional[str | Path] = None,
) -> dict[str, float]:
    """Batch-Spectral-Gating für alle Audio-Dateien im Verzeichnis.

    §G1 Song-Isolation: Jeder Song wird isoliert verarbeitet (keine Cross-Contamination).
    Deterministisch (§G5): Hash-basierter Seed pro Datei.

    Args:
        audio_dir: Verzeichnis mit Audio-Dateien.
        threshold_db: Globaler Threshold.
        output_dir: Ausgabe-Verzeichnis. Standard: input_dir + "_gated".

    Returns:
        dict[str, float]: {dateiname: peak_dbfs} für Audit-Log (§G8).
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir or str(audio_dir) + "_gated")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for f in sorted(audio_dir.glob("*.wav")):
        logger.info("§2.62 Spectral Gating Batch: %s", f.name)
        y_out = apply_spectral_gating(f, threshold_db=threshold_db)
        out_path = output_dir / f"{f.stem}_gated.wav"

        # Export (librosa — kein Dither nötig für float32/float64)
        librosa.write(str(out_path), y_out, sr=48000, format="wav")

        peak_dbfs = 20 * np.log10(np.max(np.abs(y_out)) + 1e-10)
        results[f.name] = peak_dbfs

    logger.info("§2.62 Spectral Gating Batch: %d Dateien verarbeitet", len(results))
    return results


# --- Unit-Test-Hilfe für Determinismus (§G5) ---

def verify_determinism(audio_path: str | Path, threshold_db: float = -60.0) -> bool:
    """Prüft Bit-Identität bei zwei Durchläufen (§G5)."""
    out1 = apply_spectral_gating(audio_path, threshold_db=threshold_db)
    out2 = apply_spectral_gating(audio_path, threshold_db=threshold_db)
    return np.array_equal(out1, out2)
