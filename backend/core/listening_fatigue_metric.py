"""
§v10.15 Listening Fatigue Metric — misst Hörermüdung.

Hohe Fatigue = zu hell, zu komprimiert, zu steril — das Gehirn
ermüdet nach 15 min Hören.  Niedrige Fatigue = natürliche Dynamik,
ausgewogene Spektralbalance → Langzeit-Hörgenuss.

Wertebereich: 0.0 (keine Ermüdung, optimal) … 1.0 (stark ermüdend).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── Konstanten ──────────────────────────────────────────────────────────

# Optimaler Crest-Faktor (Peak/RMS) für natürlichen, nicht-ermüdenden Klang
_OPTIMAL_CREST_DB: float = 14.0  # ~14 dB = natürliche Dynamik
_MIN_CREST_DB: float = 6.0  # < 6 dB = stark komprimiert → ermüdend
_MAX_CREST_DB: float = 22.0  # > 22 dB = Mikrofonaufnahme, selten

# Optimale Spektralbalance (Bass/Mitten/Höhen-Verhältnis)
# Perceptual target: nicht zu hell (zu viel > 4 kHz), nicht zu dumpf
_OPTIMAL_HF_RATIO: float = 0.25  # 25% der Energie > 4 kHz
_MAX_HF_RATIO: float = 0.55  # > 55% → zu hell → ermüdend

# Optimale Mikrodynamik (Frame-zu-Frame Energie-Varianz)
_OPTIMAL_MICRO_DYN: float = 0.15  # 15% relative Varianz = lebendig
_MIN_MICRO_DYN: float = 0.03  # < 3% = flach/steril → ermüdend


def measure_fatigue(
    audio: np.ndarray,
    sr: int,
    *,
    return_components: bool = False,
) -> float | dict[str, float]:
    """Misst den Listening-Fatigue-Score.

    Args:
        audio: float32, mono oder stereo, beliebige Länge.
        sr: 48000 Hz.
        return_components: Wenn True, dict mit Einzelkomponenten zurück.

    Returns:
        fatigue ∈ [0.0, 1.0] — 0 = optimal, 1 = stark ermüdend.
        Oder dict wenn return_components=True.
    """
    try:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim > 1:
            arr = arr.mean(axis=0) if arr.shape[0] <= arr.shape[1] else arr.mean(axis=1)

        n = max(len(arr), 1)
        if n < sr:  # < 1 s → zu kurz für zuverlässige Messung
            return _fallback(return_components)

        # ── 1. Crest-Faktor (Peak/RMS) ──────────────────────────────
        rms = float(np.sqrt(np.mean(arr**2)) + 1e-12)
        peak = float(np.max(np.abs(arr))) if rms > 1e-10 else 1e-10
        crest_db = 20.0 * np.log10(peak / rms + 1e-12)
        crest_db = float(np.clip(crest_db, _MIN_CREST_DB, _MAX_CREST_DB))

        # Deviation from optimal: 0 = optimal, 1 = worst.
        # Einseitig (§Hörordnung Ebene 1): Nur KOMPRESSION (Crest unter Optimum)
        # ist ermüdend — natürliche Dynamik (Crest über Optimum) wird nicht bestraft.
        if crest_db >= _OPTIMAL_CREST_DB:
            crest_dev = 0.0
        else:
            crest_dev = (_OPTIMAL_CREST_DB - crest_db) / (_OPTIMAL_CREST_DB - _MIN_CREST_DB)
        crest_dev = float(np.clip(crest_dev, 0.0, 1.0))

        # ── 2. Spektralbalance (HF-Anteil) ──────────────────────────
        fft_n = min(8192, n)
        spec = np.abs(np.fft.rfft(arr[:fft_n] * np.hamming(fft_n)))
        total_energy = float(np.sum(spec**2)) + 1e-12
        # Energie > 4 kHz (Bark 17+, wo das Ohr empfindlich auf Überbetonung reagiert)
        hf_bin = int(4000.0 / (sr / 2) * (len(spec) - 1))
        hf_energy = float(np.sum(spec[hf_bin:] ** 2))
        hf_ratio = hf_energy / total_energy
        hf_ratio = float(np.clip(hf_ratio, 0.0, _MAX_HF_RATIO))

        hf_dev = abs(hf_ratio - _OPTIMAL_HF_RATIO) / (_MAX_HF_RATIO - _OPTIMAL_HF_RATIO)
        hf_dev = float(np.clip(hf_dev, 0.0, 1.0))

        # ── 3. Mikrodynamik (Frame-Energie-Varianz) ─────────────────
        frame_s = max(1, int(sr * 0.05))  # 50 ms
        n_frames = min(80, max(2, n // frame_s))
        frame_energy = np.array([float(np.mean(arr[i * frame_s : (i + 1) * frame_s] ** 2)) for i in range(n_frames)])
        frame_energy += 1e-12
        # Relative standard deviation of frame energy
        micro_dyn = float(np.std(frame_energy) / (np.mean(frame_energy) + 1e-12))
        micro_dyn = float(np.clip(micro_dyn, _MIN_MICRO_DYN, _OPTIMAL_MICRO_DYN * 2))

        if micro_dyn >= _OPTIMAL_MICRO_DYN:
            micro_dev = 0.0  # Optimal or better
        else:
            micro_dev = 1.0 - (micro_dyn - _MIN_MICRO_DYN) / (_OPTIMAL_MICRO_DYN - _MIN_MICRO_DYN)
        micro_dev = float(np.clip(micro_dev, 0.0, 1.0))

        # ── 4. Gewichtete Kombination ───────────────────────────────
        # Spektralbalance wiegt am stärksten (zu hell = Haupt-Ermüdungsfaktor)
        # Crest-Faktor und Mikrodynamik ergänzen
        fatigue = 0.45 * hf_dev + 0.30 * crest_dev + 0.25 * micro_dev
        fatigue = float(np.clip(fatigue, 0.0, 1.0))

        if return_components:
            return {
                "fatigue": fatigue,
                "crest_dev": crest_dev,
                "hf_dev": hf_dev,
                "micro_dev": micro_dev,
                "crest_db": crest_db,
                "hf_ratio": hf_ratio,
                "micro_dyn": micro_dyn,
            }
        return fatigue

    except Exception:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        return _fallback(return_components)


def _fallback(return_components: bool) -> float | dict[str, float]:
    if return_components:
        return {
            "fatigue": 0.0,
            "crest_dev": 0.0,
            "hf_dev": 0.0,
            "micro_dev": 0.0,
            "crest_db": 14.0,
            "hf_ratio": 0.25,
            "micro_dyn": 0.15,
        }
    return 0.0


def fatigue_as_pmgg_goal(audio: np.ndarray, sr: int) -> float:
    """Konvertiert Fatigue zu PMGG-Goal-Score [0, 1].

    PMGG-Ziele sind 0 = schlecht, 1 = optimal.
    Fatigue ist 0 = optimal, 1 = schlecht.
    → Goal-Score = 1.0 - fatigue.
    """
    fatigue = measure_fatigue(audio, sr)
    if isinstance(fatigue, dict):
        fatigue = fatigue["fatigue"]
    return float(np.clip(1.0 - float(fatigue), 0.0, 1.0))
