"""Residuum-basiertes Bark-Masking — Hörordnung Ebene 2 (hoerordnung.instructions.md §4).

Der bestehende Zeitbereichs-Salience-Vergleich (Spitze-gegen-Spitze) hat eine
strukturelle Schwäche: Die Defekt-Spitze IST das lokale Maximum, daher kann der
±400 ms-Kontext sie praktisch nie „verdecken" (Befund 2026-08-23: 12969/12969
salient). Psychoakustisch korrekt ist die Frage:

    **Wird der DEFEKT-ANTEIL (Residuum über dem maskierenden Inhalt) durch das
    umgebende Signal verdeckt?**

Dieses Modul schätzt dafür pro Defekt-Event:
- Kontext-Spektrum: robustes (Median-) Spektrum des maskierenden Inhalts (±400 ms
  um das Event, Event selbst ausgenommen).
- Event-Spektrum: Spektrum am Defekt-Ort (Signal + Defekt).
- Residuum-Spektrum: max(0, Event − Kontext) pro Bark-Band.
- Maskierungsschwelle: Spread-Funktion (ISO 11172-3, dreieckige Slopes 27 dB/Bark)
  auf das Kontext-Spektrum.
- Salience: gewichteter Anteil der Residuum-Energie, der über der Schwelle liegt
  (0 = vollständig maskiert, 1 = vollständig exponiert).

Deterministisch, numpy-only, robust gegen NaN/Inf (Hörordnung: keine
Audibility-Entscheidung aus Müll-Daten).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Bark-Grenzen (ISO 11172-3), 24 Bänder bis 15.5 kHz Näherung
_BARK_EDGES_HZ = np.array(
    [
        0,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ],
    dtype=np.float64,
)
_BARK_CENTERS = (_BARK_EDGES_HZ[:-1] + _BARK_EDGES_HZ[1:]) / 2.0

_SPREAD_SLOPE_DB_PER_BARK = 27.0  # ISO 11172-3, dreieckige Spread-Funktion
_MASK_OFFSET_DB = 3.0  # konservativer Offset: Residuum muss deutlich über Schwelle
_CONTEXT_S = 0.4  # ±400 ms Kontext
_N_FFT = 4096


@dataclass
class ResiduumMaskingResult:
    salience: float
    residuum_db_per_band: np.ndarray
    threshold_db_per_band: np.ndarray
    audible_band_count: int
    band_count: int


def _stft_magnitude_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Kurzzeit-Magnituden-Spektren (dB) eines Signals → (times, freqs_db)."""
    hop = max(_N_FFT // 4, 1)
    if len(x) < _N_FFT:
        x = np.pad(x, (0, _N_FFT - len(x)))
    n_frames = 1 + (len(x) - _N_FFT) // hop
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    frames_db = np.zeros((n_frames, len(freqs)), dtype=np.float64)
    win = np.hanning(_N_FFT)
    for f in range(n_frames):
        seg = x[f * hop : f * hop + _N_FFT] * win
        mag = np.abs(np.fft.rfft(seg))
        frames_db[f, :] = 20.0 * np.log10(mag + 1e-12)
    return frames_db, freqs


def _to_bark_bands(spec_db: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Mittelt ein Spektrum (dB) über Bark-Bänder → pro Band der Median (dB)."""
    bands = np.zeros(len(_BARK_CENTERS), dtype=np.float64)
    for b in range(len(_BARK_CENTERS)):
        mask = (freqs >= _BARK_EDGES_HZ[b]) & (freqs < _BARK_EDGES_HZ[b + 1])
        if np.any(mask):
            if spec_db.ndim == 2:
                # Median über Zeit-Frames und Frequenz-Bins des Bands
                bands[b] = float(np.median(spec_db[:, mask]))
            else:
                bands[b] = float(np.median(spec_db[mask]))
        else:
            bands[b] = -120.0
    return bands


def _spread_mask_threshold(masker_db: np.ndarray) -> np.ndarray:
    """Maskierungsschwelle pro Bark-Band aus dem maskierenden Spektrum.

    ISO 11172-3 Spread-Funktion: dreieckig mit 27 dB/Bark (jede Masker-Komponente
    verdeckt mit −27 dB pro Bark Abstand). Schwelle = Maximum über alle Beiträge.
    """
    n = len(masker_db)
    # Abstandsmatrix in Bark
    idx = np.arange(n, dtype=np.float64)
    dist = np.abs(idx[:, None] - idx[None, :])
    contributions = masker_db[None, :] - _SPREAD_SLOPE_DB_PER_BARK * dist
    thr = np.max(contributions, axis=1)
    # Sehr leise Maskierer erzeugen keine nennenswerte Schwelle — Floor bei Ruhehörschwelle-Proxy
    thr = np.maximum(thr, -80.0)
    return thr + _MASK_OFFSET_DB


def estimate_residuum_salience(
    audio: np.ndarray,
    sr: int,
    loc_start: float,
    loc_end: float,
    context_s: float = _CONTEXT_S,
) -> ResiduumMaskingResult:
    """Schätzt die Hörbarkeit des Defekt-Residuums an einem Event.

    Returns:
        ResiduumMaskingResult mit salience in [0, 1] (1 = exponiert).
    """
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=-1)
    dur_s = len(mono) / max(sr, 1)
    ev_start = int(np.clip(loc_start * sr, 0, len(mono) - 1))
    ev_end = int(np.clip(loc_end * sr, ev_start + 1, len(mono)))
    ctx_start = int(np.clip(loc_start - context_s, 0, dur_s) * sr)
    ctx_end = int(np.clip(loc_end + context_s, 0, dur_s) * sr)

    # Kontext-Segmente (links + rechts, Event ausgenommen)
    left = mono[ctx_start:ev_start]
    right = mono[ev_end:ctx_end]
    ctx = np.concatenate([left, right])
    if len(ctx) < _N_FFT:
        # Zu wenig Kontext (sehr kurze Datei) → konservativ: exponiert
        return ResiduumMaskingResult(
            salience=1.0,
            residuum_db_per_band=np.zeros(len(_BARK_CENTERS)),
            threshold_db_per_band=np.zeros(len(_BARK_CENTERS)),
            audible_band_count=len(_BARK_CENTERS),
            band_count=len(_BARK_CENTERS),
        )

    ctx_frames, freqs = _stft_magnitude_db(ctx, sr)
    ctx_bands = _to_bark_bands(ctx_frames, freqs)  # Median über Zeit = robustes maskierendes Spektrum

    ev = mono[ev_start:ev_end]
    if len(ev) < 1:
        ev = mono[ev_start : ev_start + 1]
    ev_frames, freqs2 = _stft_magnitude_db(ev, sr)
    ev_bands = _to_bark_bands(ev_frames, freqs2)

    residuum_db = np.maximum(0.0, ev_bands - ctx_bands)
    threshold_db = _spread_mask_threshold(ctx_bands)
    audible = residuum_db > np.maximum(threshold_db - ctx_bands, 0.0)  # über Schwelle relativ zum Kontext

    # Salience: energie-gewichteter Anteil hörbarer Bänder
    residuum_energy = 10.0 ** (residuum_db / 10.0)
    total_energy = float(np.sum(residuum_energy)) + 1e-12
    audible_energy = float(np.sum(residuum_energy * audible.astype(np.float64)))
    salience = float(np.clip(audible_energy / total_energy, 0.0, 1.0))

    # Ruhe-Event-Schutz: kein Residuum, aber Defekt möglicherweise als absolute
    # Störung hörbar (Stille-Kontext) → konservative Exponiertheit.
    if total_energy <= 1e-8 and float(np.max(ev_bands)) > -60.0:
        salience = max(salience, 0.5)

    return ResiduumMaskingResult(
        salience=float(np.nan_to_num(salience, nan=0.5)),
        residuum_db_per_band=residuum_db,
        threshold_db_per_band=threshold_db,
        audible_band_count=int(np.sum(audible)),
        band_count=len(_BARK_CENTERS),
    )


__all__ = [
    "ResiduumMaskingResult",
    "estimate_residuum_salience",
]
