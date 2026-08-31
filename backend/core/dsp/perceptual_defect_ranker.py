"""
=============================================================================

Re-rankt technisch erkannte Defekte nach ihrer tatsächlichen Hörbarkeit.
Nur Defekte, die das menschliche Ohr wahrnehmen kann, werden priorisiert.

Prinzip: Ein -40dB Click unter einer Snare-Drum ist technisch da,
aber unhörbar. Ihn zu reparieren erzeugt nur Artefakt-Risiko ohne
akustischen Nutzen.

Algorithmus:
  1. Bark-Gewichtung: Frequenzabhängige Sensitivität des Ohrs
  2. Temporale Maskierung: Pre-Masking 20ms, Post-Masking 100ms
  3. Spektrale Maskierung: Signal maskiert Defekt im gleichen Bark-Band
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

from backend.core.dsp.bark_lufs_util import (
    BARK_EDGES_HZ,
    BARK_HEARING_THRESHOLD_DB,
    N_GAMMATONE,
    measure_lufs_per_bark,
    split_into_gammatone_bands,
)

# ---------------------------------------------------------------------------
# Bark-Gewichtung: Relative Sensitivität des Ohrs pro Frequenzband
# Normiert auf 1.0 bei maximaler Sensitivität (3-4 kHz, Bark 17-18)
# ---------------------------------------------------------------------------
_BARK_SENSITIVITY = np.array(
    [
        0.05,
        0.08,
        0.12,
        0.18,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.82,
        0.88,
        0.92,
        0.95,
        0.98,
        1.00,
        1.00,
        0.98,
        0.95,
        0.90,
        0.82,
        0.72,
        0.58,
        0.42,
        0.30,
        0.20,
        0.14,
        0.10,
        0.07,
        0.05,
        0.03,
        0.02,
    ],
    dtype=np.float32,
)


def _freq_to_bark_band(freq_hz: float) -> int:
    """Findet den Bark-Band-Index für eine Frequenz."""
    for b in range(len(BARK_EDGES_HZ) - 1):
        if BARK_EDGES_HZ[b] <= freq_hz < BARK_EDGES_HZ[b + 1]:
            return b
    return len(BARK_EDGES_HZ) - 2


def _bark_sensitivity_weight(freq_hz: float) -> float:
    """Bark-gewichtete Sensitivität für eine Frequenz."""
    band = _freq_to_bark_band(freq_hz)
    band = min(band, len(_BARK_SENSITIVITY) - 1)
    return float(_BARK_SENSITIVITY[band])


def compute_perceptual_severity(
    technical_severity: float,
    defect_freq_hz: float = 1000.0,
    defect_position_s: float = 0.0,
    audio: np.ndarray | None = None,
    sr: int = 48000,
    *,
    onset_positions_s: list[float] | None = None,
    signal_lufs_per_band: np.ndarray | None = None,
    defect_level_db: float = -40.0,
) -> float:
    """Berechnet perzeptuelle Severity aus technischer Detection.

    Args:
        technical_severity: Technische Severity [0, 1]
        defect_freq_hz: Mittenfrequenz des Defekts
        defect_position_s: Zeitposition in Sekunden
        audio: Audio-Signal für Maskierungsberechnung (optional)
        sr: Sample-Rate
        onset_positions_s: Positionen von Transienten für zeitliche Maskierung
        signal_lufs_per_band: LUFS pro Bark-Band (optional, beschleunigt)
        defect_level_db: Pegel des Defekts in dBFS

    Returns:
        Perzeptuelle Severity [0, 1] — 0 = unhörbar, 1 = maximal hörbar
    """
    severity = float(technical_severity)
    if severity <= 0.01:
        return 0.0

    # ── 1. Bark-Gewichtung ──────────────────────────────────────────
    bark_weight = _bark_sensitivity_weight(defect_freq_hz)
    severity *= 0.5 + 0.5 * bark_weight  # 50% Basis + 50% frequenzabhängig

    # ── 2. Hörschwellen-Prüfung ─────────────────────────────────────
    band = _freq_to_bark_band(defect_freq_hz)
    band = min(band, len(BARK_HEARING_THRESHOLD_DB) - 1)
    hearing_thresh_dbfs = float(BARK_HEARING_THRESHOLD_DB[band]) - 100.0  # SPL→dBFS
    if defect_level_db < hearing_thresh_dbfs:
        severity *= 0.1  # Defekt unter absoluter Hörschwelle

    # ── 3. Spektrale Maskierung ─────────────────────────────────────
    if signal_lufs_per_band is not None and band < len(signal_lufs_per_band):
        signal_level = float(signal_lufs_per_band[band])
        if signal_level > -60:
            # Signal maskiert: je lauter das Signal, desto mehr Maskierung
            level_diff = signal_level - defect_level_db
            if level_diff > 20:
                severity *= 0.05  # Signal 20dB+ lauter → Defekt komplett maskiert
            elif level_diff > 10:
                severity *= 0.2  # Signal 10-20dB lauter → stark maskiert
            elif level_diff > 5:
                severity *= 0.5  # Signal 5-10dB lauter → moderat maskiert
            elif level_diff < -10:
                severity *= 1.0  # Defekt lauter als Signal → voll hörbar

    # ── 4. Temporale Maskierung ─────────────────────────────────────
    if onset_positions_s and defect_position_s > 0:
        for onset_s in onset_positions_s:
            dt = defect_position_s - onset_s
            if -0.020 <= dt < 0:  # Pre-masking: 20ms vor Transient
                severity *= 0.3
            elif 0 <= dt <= 0.100:  # Post-masking: 100ms nach Transient
                decay = np.exp(-dt / 0.030)  # Exponentieller Decay
                severity *= 0.1 + 0.9 * (1.0 - decay)
            elif 0.100 < dt <= 0.200:  # Übergangszone
                severity *= 0.5

    return float(np.clip(severity, 0.0, 1.0))


def rerank_defects_perceptual(
    defect_scores: list[tuple[str, float, float, float]],
    audio: np.ndarray,
    sr: int,
    *,
    min_perceptual_severity: float = 0.05,
) -> list[tuple[str, float, float]]:
    """Re-rankt Defekte nach perzeptueller Hörbarkeit.

    Args:
        defect_scores: Liste von (name, technical_severity, freq_hz, position_s)
        audio: Audio-Signal für Maskierungsberechnung
        sr: Sample-Rate
        min_perceptual_severity: Minimale perzeptuelle Severity zum Behalten

    Returns:
        Liste von (name, perceptual_severity, technical_severity) sortiert
    """
    # Bark-Band LUFS für Maskierungsberechnung
    try:
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
        bands = split_into_gammatone_bands(mono[: min(len(mono), sr * 30)].astype(np.float32), sr)
        signal_lufs = measure_lufs_per_bark(bands, sr)
    except Exception:
        signal_lufs = None

    # Onset-Detektion für temporale Maskierung
    try:
        import librosa

        onset_frames = librosa.onset.onset_detect(  # type: ignore[attr-defined]  # librosa-Stubs exportieren onset nicht
            y=mono[: min(len(mono), sr * 30)],
            sr=sr,
            hop_length=512,
            backtrack=False,
        )
        onset_positions = [float(f * 512 / sr) for f in onset_frames]
    except Exception:
        onset_positions = []

    reranked: list[tuple[str, float, float]] = []
    for name, tech_sev, freq_hz, pos_s in defect_scores:
        # Schätze Defektpegel aus technischer Severity
        estimated_level_db = -20.0 - 40.0 * (1.0 - tech_sev)

        percep_sev = compute_perceptual_severity(
            technical_severity=tech_sev,
            defect_freq_hz=freq_hz,
            defect_position_s=pos_s,
            audio=audio,
            sr=sr,
            onset_positions_s=onset_positions,
            signal_lufs_per_band=signal_lufs,
            defect_level_db=estimated_level_db,
        )

        if percep_sev >= min_perceptual_severity:
            reranked.append((name, percep_sev, tech_sev))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked
