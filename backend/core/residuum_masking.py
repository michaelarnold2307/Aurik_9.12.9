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
from typing import cast

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
    win = np.hanning(_N_FFT)
    # §2.46h (2026-09-06) Vektorisierung: Strided-Batch statt Python-Frame-Loop.
    # pocketfft transformiert jede Zeile unabhängig mit identischem Algorithmus
    # wie der frühere Per-Frame-rfft → bit-identisches Ergebnis, ~N× schneller.
    # (Produktionsbefund: ~50 rfft-Aufrufe pro Event-Kontext, GIL-gebunden.)
    _x8 = np.ascontiguousarray(x, dtype=np.float64)
    frames = np.lib.stride_tricks.as_strided(_x8, shape=(n_frames, _N_FFT), strides=(hop * _x8.itemsize, _x8.itemsize))
    mag = np.abs(np.fft.rfft(frames * win, axis=1))
    frames_db = 20.0 * np.log10(mag + 1e-12)
    return frames_db, freqs


# §2.46h (2026-09-06): Bark-Bin-Indizes pro Band einmalig vorberechnen —
# ersetzt 28× Frequenz-Vergleichs-Masken pro Aufruf (1708 Aufrufe im
# 20s-Scan). Die Median-Berechnung bleibt numerisch identisch.
_BARK_BIN_INDEX: list[np.ndarray] = []
_BARK_BIN_INDEX_FREQS: np.ndarray | None = None


def _to_bark_bands(spec_db: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Mittelt ein Spektrum (dB) über Bark-Bänder → pro Band der Median (dB)."""
    global _BARK_BIN_INDEX, _BARK_BIN_INDEX_FREQS
    bands = np.zeros(len(_BARK_CENTERS), dtype=np.float64)
    if _BARK_BIN_INDEX_FREQS is None or not np.array_equal(_BARK_BIN_INDEX_FREQS, freqs):
        _BARK_BIN_INDEX = [
            np.where((freqs >= _BARK_EDGES_HZ[b]) & (freqs < _BARK_EDGES_HZ[b + 1]))[0]
            for b in range(len(_BARK_CENTERS))
        ]
        _BARK_BIN_INDEX_FREQS = np.asarray(freqs, dtype=np.float64)
    for b in range(len(_BARK_CENTERS)):
        _bin_idx = _BARK_BIN_INDEX[b]
        if _bin_idx.size > 0:
            if spec_db.ndim == 2:
                # Median über Zeit-Frames und Frequenz-Bins des Bands
                bands[b] = float(np.median(spec_db[:, _bin_idx]))
            else:
                bands[b] = float(np.median(spec_db[_bin_idx]))
        else:
            bands[b] = -120.0
    return cast(np.ndarray, bands)


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
    return cast(np.ndarray, thr + _MASK_OFFSET_DB)


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


def estimate_residuum_salience_batch(
    audio: np.ndarray,
    sr: int,
    locations: list[tuple[float, float]],
    context_s: float = _CONTEXT_S,
) -> dict[tuple[float, float], ResiduumMaskingResult]:
    """§2.46h (2026-09-06) Batch-Variante: EIN Full-Audio-STFT statt 2 STFTs je Event.

    Der Per-Event-Pfad berechnet pro Event einen STFT über das konkatenierte
    Kontext-Segment (n_fft=4096, ~50 Frames) + einen Event-STFT — im 20s-Scan
    ~40.000 rfft(4096)-Aufrufe, die den GIL nur kurz freigeben und dadurch
    weder sequentiell noch per Threading skalieren. Hier wird der STFT EINMAL
    über das Gesamt-Audio berechnet und pro Event werden nur die passenden
    Frames ausgewählt (Kontext-Frames außerhalb des Events, Event-Frames
    innerhalb).

    Numerik: deterministisch (§G5), aber NICHT bit-identisch zum Per-Event-
    Pfad — bewusst: die alte Version vermischte über die Konkatenation nicht
    benachbarte Audio-Teile in „Junction-Frames“ und zero-paddete kurze
    Events; die Batch-Version nutzt ausschließlich echte Audio-Fenster
    (psychoakustisch korrekter, gleiche Bark-/Spread-/Salienz-Formeln).

    Returns:
        Mapping location → ResiduumMaskingResult (gleiche Semantik wie der
        Per-Event-Pfad).
    """
    if not locations:
        return {}
    if len(locations) <= 1:
        return {loc: estimate_residuum_salience(audio, sr, loc[0], loc[1], context_s=context_s) for loc in locations}

    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=-1)
    dur_s = len(mono) / max(sr, 1)
    if len(mono) < _N_FFT:
        return {loc: estimate_residuum_salience(audio, sr, loc[0], loc[1], context_s=context_s) for loc in locations}

    hop = max(_N_FFT // 4, 1)
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    full_frames_db, _ = _stft_magnitude_db(mono, sr)
    n_frames = full_frames_db.shape[0]
    # Frame-Zeitfenster: Frame f deckt [f*hop, f*hop + _N_FFT) Samples ab.
    frame_start_t = (np.arange(n_frames, dtype=np.float64) * hop) / sr
    frame_end_t = (np.arange(n_frames, dtype=np.float64) * hop + _N_FFT) / sr

    # Bark-Bin-Indizes einmalig aufbauen (identisch zu _to_bark_bands)
    _bin_idx_per_band = [
        np.where((freqs >= _BARK_EDGES_HZ[b]) & (freqs < _BARK_EDGES_HZ[b + 1]))[0] for b in range(len(_BARK_CENTERS))
    ]

    def _bands_of_frames(sel_frames: np.ndarray) -> np.ndarray:
        if sel_frames.size == 0:
            _empty: np.ndarray = np.full(len(_BARK_CENTERS), -120.0, dtype=np.float64)
            return _empty
        _sub = full_frames_db[sel_frames, :]
        _bands: np.ndarray = np.zeros(len(_BARK_CENTERS), dtype=np.float64)
        for b in range(len(_BARK_CENTERS)):
            _bins = _bin_idx_per_band[b]
            if _bins.size > 0:
                _bands[b] = float(np.median(_sub[:, _bins]))
            else:
                _bands[b] = -120.0
        return _bands

    out: dict[tuple[float, float], ResiduumMaskingResult] = {}
    for loc in locations:
        loc_start, loc_end = float(loc[0]), float(loc[1])
        ev_start = int(np.clip(loc_start * sr, 0, len(mono) - 1))
        ev_end = int(np.clip(loc_end * sr, ev_start + 1, len(mono)))
        ctx_start = int(np.clip(loc_start - context_s, 0, dur_s) * sr)
        ctx_end = int(np.clip(loc_end + context_s, 0, dur_s) * sr)
        ev_start_t = ev_start / sr
        ev_end_t = ev_end / sr

        _ctx_mask = (frame_end_t <= float(ctx_end) / sr) & (frame_start_t >= float(ctx_start) / sr)
        _ctx_mask &= (frame_end_t <= ev_start_t) | (frame_start_t >= ev_end_t)
        _ev_mask = (frame_end_t > ev_start_t) & (frame_start_t < ev_end_t)

        ctx_bands = _bands_of_frames(np.where(_ctx_mask)[0])
        ev_bands = _bands_of_frames(np.where(_ev_mask)[0])

        if float(np.max(ctx_bands)) <= -119.0:
            # Kein auswertbarer Kontext → konservativ exponiert (wie Per-Event-Pfad)
            out[loc] = ResiduumMaskingResult(
                salience=1.0,
                residuum_db_per_band=np.zeros(len(_BARK_CENTERS)),
                threshold_db_per_band=np.zeros(len(_BARK_CENTERS)),
                audible_band_count=len(_BARK_CENTERS),
                band_count=len(_BARK_CENTERS),
            )
            continue
        if float(np.max(ev_bands)) <= -119.0:
            # Kein Event-Frame (extrem kurzer Defekt) → ein Frame um die Event-Mitte
            _ev_mid = int((ev_start + ev_end) // 2)
            _ev_t = _ev_mid / sr
            _ev_mask = (frame_end_t > _ev_t) & (frame_start_t < _ev_t)
            ev_bands = _bands_of_frames(np.where(_ev_mask)[0])

        residuum_db = np.maximum(0.0, ev_bands - ctx_bands)
        threshold_db = _spread_mask_threshold(ctx_bands)
        audible = residuum_db > np.maximum(threshold_db - ctx_bands, 0.0)
        residuum_energy = 10.0 ** (residuum_db / 10.0)
        total_energy = float(np.sum(residuum_energy)) + 1e-12
        audible_energy = float(np.sum(residuum_energy * audible.astype(np.float64)))
        salience = float(np.clip(audible_energy / total_energy, 0.0, 1.0))
        if total_energy <= 1e-8 and float(np.max(ev_bands)) > -60.0:
            salience = max(salience, 0.5)
        out[loc] = ResiduumMaskingResult(
            salience=float(np.nan_to_num(salience, nan=0.5)),
            residuum_db_per_band=residuum_db,
            threshold_db_per_band=threshold_db,
            audible_band_count=int(np.sum(audible)),
            band_count=len(_BARK_CENTERS),
        )
    return out


# ── P1-3: Lokale Maskierungs-JND für Phasen-Deltas ────────────────────────

_DELTA_JND_CAP_DB = 6.0  # Obergrenze der maskierten Toleranz-Erhöhung (1 Bark-Spread-Schritt)


@dataclass
class DeltaMaskingJNDResult:
    """Ergebnis der P1-3-Maskierungs-JND-Schätzung für einen Phasen-Delta.

    Attributes:
        jnd_db: Maskierte Delta-Marge in dB [0, _DELTA_JND_CAP_DB].
            Guards setzen ihre effektive Toleranz auf max(fest, jnd_db).
        delta_above_db: Wie weit das lauteste Delta-Band ÜBER der Schwelle
            liegt (0 = vollständig maskiert, >0 = exponiert).
        threshold_db: Maximale Maskierungsschwelle über die Bänder (Log-Kontext).
    """

    jnd_db: float
    delta_above_db: float
    threshold_db: float


def delta_masking_margin_db_per_band(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
) -> np.ndarray:
    """P1-3: Maskierungs-Marge des Phasen-Deltas je Bark-Band (dB).

    margin[b] = Schwelle[b] − Delta[b] — positiv heißt: der Delta-Anteil in
    Band b ist lokal maskiert. Rückgabewerte NaN/Inf-geschützt; bei nicht
    auswertbarem Input wird ein Null-Array (keine Maskierung) zurückgegeben.
    """
    _zeros: np.ndarray = np.zeros(len(_BARK_CENTERS), dtype=np.float64)
    try:
        if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
            return _zeros
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        if pre.shape != post.shape or pre.size < 256:
            return _zeros

        def _mono(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2:
                _ax = 0 if x.shape[0] <= 2 else 1
                return x.mean(axis=_ax)
            return x

        pre_mono = _mono(pre)
        delta = _mono(post) - pre_mono

        pre_frames, freqs = _stft_magnitude_db(pre_mono, sr)
        pre_bands = _to_bark_bands(pre_frames, freqs)
        thr = _spread_mask_threshold(pre_bands)

        d_frames, _ = _stft_magnitude_db(delta, sr)
        d_bands = _to_bark_bands(d_frames, freqs)
        return thr - d_bands
    except Exception as exc:
        logger.debug("delta_masking_margin_db_per_band nicht blockierend: %s", exc)
        return _zeros


def bark_band_index_of_freq(f_hz: float) -> int:
    """P1-3: Bark-Band-Index (ISO-11172-3-Grenzen) für eine Frequenz.

    Nächster Bark-Mittenfrequenz-Index — für Guard-Integration
    (z. B. Formant-Bänder in lpc_formant_tracker).
    """
    return int(np.clip(np.argmin(np.abs(_BARK_CENTERS - float(f_hz))), 0, len(_BARK_CENTERS) - 1))


def estimate_delta_masking_jnd_db(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    *,
    freq_range_hz: tuple[float, float] | None = None,
) -> DeltaMaskingJNDResult:
    """P1-3 (Hörordnung Ebene 2): Schätzt, wie viele dB des Phasen-Deltas
    ``d = post − pre`` durch das Signal selbst verdeckt werden.

    Guard-Kontrakt (dsp.instructions.md §WBG/§ATI/§SCK):
        effektive_Toleranz = max(feste_Toleranz, jnd_db)
    → Wenn der Phasen-Eingriff lokal maskiert ist, lösen die Guards keine
      Rollbacks/Rücknahmen für unhörbare Abweichungen aus (weniger falsche
      Rollbacks, weniger End-Gate-Recovery-Runden).

    Methode: Bark-Spektrum des Signals (Median über Frames) → ISO-11172-3-
    Spread-Schwelle (27 dB/Bark, +3 dB Offset) → Abstand des Delta-Spektrums
    unterhalb der Schwelle. Konservativ: Bei nicht auswertbarem Input wird
    jnd_db=0 angenommen (keine Maskierung unterstellt).

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N] / [N, 2].
        post: Audio nach der Phase (gleiche Shape).
        sr: Sample-Rate.
        freq_range_hz: Optionales Frequenzfenster (f_low, f_high) — nur Bark-
            Bänder mit Mittenfrequenz in diesem Bereich zählen.

    Returns:
        DeltaMaskingJNDResult (deterministisch, NaN/Inf-geschützt).
    """
    _cons = DeltaMaskingJNDResult(jnd_db=0.0, delta_above_db=0.0, threshold_db=-120.0)
    try:
        margins = delta_masking_margin_db_per_band(pre, post, sr)
        if freq_range_hz is not None:
            _sel = (_BARK_CENTERS >= freq_range_hz[0]) & (_BARK_CENTERS <= freq_range_hz[1])
            if not np.any(_sel):
                return _cons
            margins = margins[_sel]
        jnd_db = float(np.clip(np.max(margins), 0.0, _DELTA_JND_CAP_DB))
        above = float(np.clip(np.max(-margins), 0.0, None))
        return DeltaMaskingJNDResult(
            jnd_db=round(jnd_db, 3),
            delta_above_db=round(above, 3),
            # Log-Kontext-Feld: Die operative Größe sind die per-Band-Margins
            # (delta_masking_margin_db_per_band); hier bewusst ohne Verbraucher.
            threshold_db=-120.0,
        )
    except Exception as exc:
        logger.debug("estimate_delta_masking_jnd_db nicht blockierend: %s", exc)
        return _cons


__all__ = [
    "DeltaMaskingJNDResult",
    "ResiduumMaskingResult",
    "bark_band_index_of_freq",
    "delta_masking_margin_db_per_band",
    "estimate_delta_masking_jnd_db",
    "estimate_residuum_salience",
    "estimate_residuum_salience_batch",
]
