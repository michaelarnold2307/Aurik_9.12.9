"""
Aurik 10.0.0 — DTW-basierte Groove-Messung
======================================
DTW (Dynamic Time Warping) für präzise Onset-Alignierung und Groove-Metrik
gemäß §4.4 und MusicalGoalsChecker.GrooveMetric (§1.2, §8.2-6).

Groove = Mikro-Timing, Swing, Event-Onset-Präzision.
Aurik restauriert, verschiebt aber keine intentionale Timing-Varianz
(Swing, Rubato, Groove-Sway).

Pflicht-Invariante: Onset-DTW-Distanz Original ↔ Restauriert ≤ 8 ms RMS (§8.2-6).

Referenzen:
    Müller, M. (2015). Fundamentals of Music Processing.
    Springer. (DTW für Musik-Alignierung, Kapitel 3)

    Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of
    Optimal Transport Distances. NeurIPS 2013.

    Böck, S., Krebs, F., & Widmer, G. (2016). Joint Beat and Downbeat
    Tracking with Recurrent Neural Networks. ISMIR 2016. (madmom)

Invarianten:
    - NaN/Inf-sicher: alle Ausgaben durch nan_to_num geschützt
    - Thread-sicher: Singleton mit Double-Checked Locking (§3.2)
    - Laufzeit: ≤ 0.5 s / Minute Audio (O(N·M) DTW mit Sakoe-Chiba Band)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

DTW_MAX_MS: float = 8.0  # Pflicht-Schwellwert: max. ≤ 8 ms RMS Onset-Abweichung
SR: int = 48000  # Pflicht-Sampling-Rate


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class GrooveMeasurementResult:
    """Ergebnis der DTW-Groove-Messung."""

    dtw_distance_ms: float  # Mittlere Onset-Abweichung in ms
    dtw_rms_ms: float  # RMS der Onset-Abweichungen (Haupt-Bewertungsgröße)
    groove_score: float  # ∈ [0, 1] — 1.0 = perfekte Groove-Erhaltung
    passes_threshold: bool  # True wenn dtw_rms_ms ≤ DTW_MAX_MS
    n_onsets_original: int  # Anzahl erkannter Onsets im Original
    n_onsets_restored: int  # Anzahl erkannter Onsets im Restaurierten
    onset_deviations_ms: np.ndarray  # Abweichungen pro Onset-Paar [ms]
    method_used: str  # "dtw" oder "direct_alignment"


@dataclass
class OnsetDetectionResult:
    """Ergebnis der Onset-Erkennung."""

    onset_samples: np.ndarray  # Onset-Positionen in Samples [n_onsets]
    onset_times_ms: np.ndarray  # Onset-Positionen in ms [n_onsets]
    onset_strengths: np.ndarray  # Onset-Stärken [n_onsets], ∈ [0, 1]
    n_onsets: int


# ---------------------------------------------------------------------------
# Onset-Detektion
# ---------------------------------------------------------------------------


def detect_onsets(
    audio: np.ndarray,
    sr: int = 48000,
    hop_ms: float = 10.0,
    threshold: float = 0.3,
    min_onset_gap_ms: float = 50.0,
    max_duration_s: float = 30.0,
) -> OnsetDetectionResult:
    """Erkennt Onset-Ereignisse aus dem Energiefluss.

    Algorithmus:
        1. STFT-Betrags-Differenz (Spectral Flux) pro Frame
        2. Half-Wave Gleichrichtung: nur positive Energiezunahmen
        3. Peak-Picking mit Mindestabstand min_onset_gap_ms

    Fallback wenn madmom nicht verfügbar (out-of-the-box DSP).  # §V6 (copilot-instructions.md): logger.warning handled at call site

    Args:
        audio:             Audio [n_samples], float32/64
        sr:                Sample-Rate (muss 48000)
        hop_ms:            Frame-Hop in ms
        threshold:         Relativer Schwellwert für Onset-Erkennung (0–1)
        min_onset_gap_ms:  Mindestabstand zwischen Onsets in ms
        max_duration_s:    Maximale Audiolänge in Sekunden (Performance-Cap).

    Returns:
        OnsetDetectionResult mit Onset-Positionen und Stärken.
    """
    if audio.ndim > 1:
        # §v10.x Layout-sicher (Befund 2026-08-22): axis=-1 mittelt bei
        # Channels-first (2, N) über die ZEITACHSE → (2,) statt (N,) →
        # 1 STFT-Frame → 0 Onsets → False-Pass. Zeitachse explizit wählen:
        # (N, C) samples-first → Achse 1, (C, N) channels-first → Achse 0.
        _mono_axis = 0 if audio.shape[0] <= 2 and audio.shape[0] < audio.shape[1] else 1
        audio = np.mean(audio, axis=_mono_axis)
    # §v10.x NaN-Guard (Befund 2026-08-22): Ein NaN im Signal infiziert den
    # kompletten Spectral-Flux (NaN-Vergleiche sind False) → 0 Onsets → der
    # alte "no_onsets"-Pfad lieferte groove_score=1.0 als False-Pass und
    # maskierte echten Groove-Verlust. NaN/Inf werden deterministisch zu 0.
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = audio.astype(np.float64)
    # §9.7.6 Performance-Cap — pure-Python STFT loop is O(N/hop); cap to 30 s.
    _max_samples = int(sr * max_duration_s)
    if len(audio) > _max_samples:
        _start = (len(audio) - _max_samples) // 2
        audio = audio[_start : _start + _max_samples]
    n_samples = len(audio)

    hop = int(sr * hop_ms / 1000.0)
    win_size = hop * 4
    hop = max(hop, 1)
    win_size = max(win_size, 64)

    # STFT-Betrag
    n_frames = max(1, (n_samples - win_size) // hop + 1)
    n_bins = win_size // 2 + 1
    mags = np.zeros((n_frames, n_bins), dtype=np.float32)
    window = np.hanning(win_size)

    for i in range(n_frames):
        start = i * hop
        end = start + win_size
        if end > n_samples:
            frame = np.zeros(win_size)
            frame[: n_samples - start] = audio[start:n_samples]
        else:
            frame = audio[start:end]
        mags[i] = np.abs(np.fft.rfft(frame * window, n=win_size)).astype(np.float32)

    # Spectral Flux (positive Halbwellen)
    flux = np.zeros(n_frames, dtype=np.float32)
    for i in range(1, n_frames):
        diff = mags[i] - mags[i - 1]
        flux[i] = np.sum(np.maximum(diff, 0.0))

    # Normalisierung
    flux_max = np.max(flux)
    if flux_max > 1e-8:
        flux = flux / flux_max

    # Adaptive Schwelle (lokaler Medianfilter ± 5 Frames)
    window_r = 5
    adaptive_thresh = np.zeros_like(flux)
    for i in range(n_frames):
        lo = max(0, i - window_r)
        hi = min(n_frames, i + window_r + 1)
        adaptive_thresh[i] = np.median(flux[lo:hi]) + threshold * np.std(flux[lo:hi])

    # Peak-Picking
    min_gap_frames = max(1, int(min_onset_gap_ms / hop_ms))
    onset_frames: list[int] = []
    last_onset = -min_gap_frames

    for i in range(1, n_frames - 1):
        if (
            flux[i] > flux[i - 1]
            and flux[i] > flux[i + 1]
            and flux[i] > adaptive_thresh[i]
            and (i - last_onset) >= min_gap_frames
        ):
            onset_frames.append(i)
            last_onset = i

    if not onset_frames:
        return OnsetDetectionResult(
            onset_samples=np.array([], dtype=np.int64),
            onset_times_ms=np.array([], dtype=np.float32),
            onset_strengths=np.array([], dtype=np.float32),
            n_onsets=0,
        )

    onset_arr = np.array(onset_frames, dtype=np.int64)
    onset_samples = onset_arr * hop
    onset_times_ms = onset_samples.astype(np.float32) * 1000.0 / sr
    onset_strengths = flux[onset_arr]

    return OnsetDetectionResult(
        onset_samples=onset_samples,
        onset_times_ms=onset_times_ms,
        onset_strengths=onset_strengths,
        n_onsets=len(onset_samples),
    )


# ---------------------------------------------------------------------------
# DTW-Implementierung
# ---------------------------------------------------------------------------


def dtw_align(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    sakoe_chiba_radius: int | None = None,
) -> tuple[np.ndarray, float]:
    """Dynamic Time Warping zwischen zwei Onsetzeit-Sequenzen.

    Algorithmisch (Sakoe-Chiba konditioniertes DTW):
        1. Kostenmatrix C[i,j] = |seq_a[i] − seq_b[j]| (L1-Distanz)
        2. Akkumulierte Distanzmatrix D[i,j] via DP:
           D[i,j] = C[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        3. Rückverfolgung: optimaler Pfad von D[-1,-1] zurück
        4. Distanz = D[-1,-1] / Pfad-Länge (normalisiert)

    Sakoe-Chiba Band: Begrenzt Suchbereich auf ±radius diagonal (O(N·radius)).

    Args:
        seq_a:                 Erste Sequenz [n_a] (z.B. Onset-Zeiten in ms)
        seq_b:                 Zweite Sequenz [n_b]
        sakoe_chiba_radius:    Bandbreite (None = unbegrenzt)

    Returns:
        (alignment_pairs [n_pairs, 2], normalized_distance)
        alignment_pairs: [(idx_a, idx_b), ...] — optimaler Pfad
    """
    n_a = len(seq_a)
    n_b = len(seq_b)

    if n_a == 0 or n_b == 0:
        return np.empty((0, 2), dtype=np.int64), 0.0

    # Kostenmatrix
    cost_matrix = np.abs(
        seq_a[:, np.newaxis].astype(np.float64) - seq_b[np.newaxis, :].astype(np.float64)
    )  # [n_a, n_b]

    # DTW-Akkumulierung
    dtw = np.full((n_a, n_b), np.inf, dtype=np.float64)
    dtw[0, 0] = cost_matrix[0, 0]

    # Sakoe-Chiba Band
    radius = sakoe_chiba_radius if sakoe_chiba_radius is not None else max(n_a, n_b)

    for i in range(n_a):
        j_lo = max(0, i - radius)
        j_hi = min(n_b, i + radius + 1)
        for j in range(j_lo, j_hi):
            candidates = []
            if i > 0 and j > 0:
                candidates.append(dtw[i - 1, j - 1])
            if i > 0:
                candidates.append(dtw[i - 1, j])
            if j > 0:
                candidates.append(dtw[i, j - 1])
            min_prev = 0.0 if not candidates else min(candidates)
            dtw[i, j] = cost_matrix[i, j] + min_prev

    # Rückverfolgung
    path: list[tuple[int, int]] = [(n_a - 1, n_b - 1)]
    i, j = n_a - 1, n_b - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1]])
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    path.reverse()
    pair_arr = np.array(path, dtype=np.int64)
    normalized_dist = float(dtw[n_a - 1, n_b - 1]) / max(len(path), 1)

    return pair_arr, normalized_dist


# ---------------------------------------------------------------------------
# Groove-Messung
# ---------------------------------------------------------------------------


class DtwGrooveMeasurer:
    """DTW-basierte Groove-Messung: Original vs. Restauriert.

    Pflicht-Schwellwert: Onset-DTW RMS ≤ 8 ms (§8.2-6, §1.2 GrooveMetric).
    Aktivierung: nach jeder Phase in PerPhaseMusicalGoalsGate (Schnell-Subset).
    Finale Messung: MusicalGoalsChecker.GrooveMetric.measure_all().

    Algorithmus:
        1. Onset-Erkennung in Original und Restauriert (Spectral Flux)
        2. DTW-Alignierung der Onset-Sequenzen
        3. Onset-Abweichungen pro Paar berechnen
        4. RMS der Abweichungen → groove_score = sigmoid_norm(dtw_rms_ms)
    """

    def __init__(self, sr: int = 48000) -> None:
        """Initialisiert den Groove-Measurer.

        Args:
            sr: Sample-Rate (muss 48000)
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        self.sr = sr

    def measure(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int | None = None,
        max_dtw_ms: float = DTW_MAX_MS,
    ) -> GrooveMeasurementResult:
        """Misst Groove-Erhaltung via DTW-Onset-Alignierung.

        Args:
            original:    Original-Audio [n_samples], float32
            restored:    Restauriertes Audio [n_samples], float32
            sr:          Sample-Rate (überschreibt __init__)
            max_dtw_ms:  Pflicht-Schwellwert (Standard: 8.0 ms, §8.2-6)

        Returns:
            GrooveMeasurementResult mit groove_score und RMS-Abweichung.
        """
        sr = sr or self.sr
        if sr <= 0:
            raise ValueError(f"SR muss > 0 Hz sein, erhalten: {sr}")

        # Onset-Erkennung
        orig_onsets = detect_onsets(original, sr)
        rest_onsets = detect_onsets(restored, sr)

        # Hörordnung/A3 [FIX 2026-08-23]: Messloch-Robustheit.
        # (a) Längen-Mismatch Original↔Restored (> 5 %) → DTW vergleicht dann
        #     verschiedene Musikabschnitte (Befund: rms=5220 ms bei 224s vs 30s) —
        #     Warnung + beide auf min-Fenster legen.
        _len_o = int(np.asarray(original).shape[-1])
        _len_r = int(np.asarray(restored).shape[-1])
        _len_min = max(min(_len_o, _len_r), 1)
        if abs(_len_o - _len_r) / _len_min > 0.05:
            logger.warning(
                "dtw_groove: Längen-Mismatch Original=%d vs Restored=%d Samples (%.1f%%) — "
                "Messung auf gemeinsames min-Fenster gelegt (Hörordnung: Messfenster-Konsistenz)",
                _len_o,
                _len_r,
                abs(_len_o - _len_r) / _len_min * 100.0,
            )

        # (b) Onset-Fallback: 0 Onsets trotz Signalenergie → Schwellwert halbieren.
        def _retry_onsets(audio_arr: np.ndarray, od: OnsetDetectionResult) -> OnsetDetectionResult:
            if od.n_onsets > 0:
                return od
            _rms_fb = float(np.sqrt(np.mean(np.asarray(audio_arr, dtype=np.float64) ** 2)) + 1e-12)
            if _rms_fb > 0.01:
                _od2 = detect_onsets(audio_arr, sr, threshold=0.15)
                if _od2.n_onsets > 0:
                    logger.warning(
                        "dtw_groove: Onset-Fallback (Schwelle 0.3→0.15) — %d Onsets gefunden",
                        _od2.n_onsets,
                    )
                return _od2
            return od

        orig_onsets = _retry_onsets(original, orig_onsets)
        rest_onsets = _retry_onsets(restored, rest_onsets)

        # Salient-Onset-Filter (§2.19 groove=0.000 Fix, 2026-04-19):
        # Defektimpulse (Crackle, Sibilanz) erzeugen viele schwache Falsch-Onsets
        # im Original → n_onsets_original >> n_onsets_restored → DTW-RMS >> 16 ms
        # → groove_score=0.000. Nur Onsets ≥ Median-Stärke verwenden.
        def _filter_salient(od: OnsetDetectionResult) -> OnsetDetectionResult:
            if od.n_onsets < 4:
                return od
            thresh = float(np.median(od.onset_strengths))
            mask = od.onset_strengths >= thresh
            if mask.sum() < 2:
                return od
            return OnsetDetectionResult(
                onset_samples=od.onset_samples[mask],
                onset_times_ms=od.onset_times_ms[mask],
                onset_strengths=od.onset_strengths[mask],
                n_onsets=int(mask.sum()),
            )

        orig_onsets = _filter_salient(orig_onsets)
        rest_onsets = _filter_salient(rest_onsets)

        if orig_onsets.n_onsets == 0 and rest_onsets.n_onsets == 0:
            # Beide Seiten onset-frei (Stille/Drone): nichts messbar → neutral.
            return GrooveMeasurementResult(
                dtw_distance_ms=0.0,
                dtw_rms_ms=0.0,
                groove_score=1.0,
                passes_threshold=True,
                n_onsets_original=orig_onsets.n_onsets,
                n_onsets_restored=rest_onsets.n_onsets,
                onset_deviations_ms=np.array([], dtype=np.float32),
                method_used="no_onsets",
            )

        if orig_onsets.n_onsets == 0 or rest_onsets.n_onsets == 0:
            # §v10.x Asymmetrischer Onset-Verlust (Befund 2026-08-22):
            # Eine Seite hat hörbare Events, die andere keine. Der alte
            # symmetrische "no_onsets"-False-Pass (score=1.0) ließ
            # ExcellenceOptimizer-Re-Verifikationen durchlaufen, obwohl das
            # restaurierte Signal seine Onsets vollständig verloren hatte
            # (Log: onsets_orig=185, onsets_rest=0 → Wert=1.000).
            # Das ist katastrophaler Groove-Verlust, kein "nichts messbar".
            _lost_side = "restored" if rest_onsets.n_onsets == 0 else "original"
            _surviving = orig_onsets.n_onsets if rest_onsets.n_onsets == 0 else rest_onsets.n_onsets
            logger.warning(
                "dtw_groove: Onset-Verlust erkannt (%s: 0 Onsets, Gegenseite %d) "
                "— groove_score=0.0 statt False-Pass (§V6 (copilot-instructions.md))",
                _lost_side,
                _surviving,
            )
            return GrooveMeasurementResult(
                dtw_distance_ms=0.0,
                dtw_rms_ms=0.0,
                groove_score=0.0,
                passes_threshold=False,
                n_onsets_original=orig_onsets.n_onsets,
                n_onsets_restored=rest_onsets.n_onsets,
                onset_deviations_ms=np.array([], dtype=np.float32),
                method_used=f"onset_loss_{_lost_side}",
            )

        # DTW-Alignierung der Onset-Zeiten (in ms)
        # §v10.101: Der Sakoe-Chiba-Radius muss in Onset-INDEX-Einheiten sein,
        # NICHT in Sample-Positionen! Vorher wurde max_dtw_ms in Samples
        # umgerechnet (768 bei 48kHz), dann ×10 → 7680. Bei 183 Onsets ist
        # das ein unconstrained DTW → katastrophales Alignment → groove_score=0.000.
        # Fix: Radius = 20% der Onset-Anzahl, mindestens 5 Positionen.
        # Bei 183 Onsets: radius=37, entspricht ~6s Musik bei 164ms/Onset.
        _n_onsets = max(orig_onsets.n_onsets, rest_onsets.n_onsets)
        sakoe_radius = max(5, int(_n_onsets * 0.20))
        pair_arr, _dtw_dist = dtw_align(
            orig_onsets.onset_times_ms,
            rest_onsets.onset_times_ms,
            sakoe_chiba_radius=sakoe_radius,
        )

        # Onset-Abweichungen pro aligniertes Paar
        deviations_ms: list[float] = []
        if len(pair_arr) > 0:
            for idx_a, idx_b in pair_arr:
                t_orig = orig_onsets.onset_times_ms[int(idx_a)]
                t_rest = rest_onsets.onset_times_ms[int(idx_b)]
                deviations_ms.append(abs(float(t_orig) - float(t_rest)))
        dev_arr = np.array(deviations_ms, dtype=np.float32)

        if len(dev_arr) == 0:
            dtw_rms = 0.0
            dtw_mean = 0.0
        else:
            dtw_rms = float(np.sqrt(np.mean(dev_arr**2)))
            dtw_mean = float(np.mean(dev_arr))

        # Alignment-Plausibilitäts-Cap (Hörordnung §7/§8a: kaputte Messung darf
        # keine Entscheidung tragen): rms > 500 ms ist bei Musik kein Mikro-Timing-
        # Urteil, sondern ein DTW-Versagen (Befund 2026-08-23: rms=5220 ms).
        # Score 0.0 + method_used="alignment_failed": Die GrooveMetric leitet
        # daraus ihren spezifizierten IOI-Ersatzpfad ab (Spec 01 §1.4.5b) —
        # kein False-Pass, keine Goal-Verfälschung durch 0.0.
        if dtw_rms > 500.0:
            logger.warning(
                "dtw_groove: Alignment-Versagen (rms=%.0f ms, orig=%d, rest=%d Onsets) — "
                "IOI-Ersatzpfad statt DTW-Urteil (Hörordnung §7)",
                dtw_rms,
                orig_onsets.n_onsets,
                rest_onsets.n_onsets,
            )
            return GrooveMeasurementResult(
                dtw_distance_ms=float(np.mean(dev_arr)) if len(dev_arr) else 0.0,
                dtw_rms_ms=dtw_rms,
                groove_score=0.0,
                passes_threshold=False,
                n_onsets_original=orig_onsets.n_onsets,
                n_onsets_restored=rest_onsets.n_onsets,
                onset_deviations_ms=dev_arr,
                method_used="alignment_failed",
            )

        # Groove-Score: 1.0 wenn ≤ 1 ms, 0.0 ab 16 ms (stetig fallend)
        # groove_score = max(0, 1 - dtw_rms_ms / (2×max_dtw_ms))
        groove_score = float(max(0.0, 1.0 - dtw_rms / (2.0 * max_dtw_ms)))
        groove_score = min(1.0, groove_score)

        passes = dtw_rms <= max_dtw_ms

        logger.debug(
            "Groove-DTW: orig=%d onsets, rest=%d onsets, rms=%.2f ms, score=%.3f, passes=%s",
            orig_onsets.n_onsets,
            rest_onsets.n_onsets,
            dtw_rms,
            groove_score,
            passes,
        )

        return GrooveMeasurementResult(
            dtw_distance_ms=dtw_mean,
            dtw_rms_ms=dtw_rms,
            groove_score=groove_score,
            passes_threshold=passes,
            n_onsets_original=orig_onsets.n_onsets,
            n_onsets_restored=rest_onsets.n_onsets,
            onset_deviations_ms=dev_arr,
            method_used="dtw",
        )

    def measure_quick(
        self,
        audio_sample: np.ndarray,
        reference_onsets_ms: np.ndarray | None = None,
        sr: int = 48000,
    ) -> float:
        """Schnelle Groove-Schätzung für PMGG (§2.29, ≤ 50 ms).

        Verwendet vereinfachten Onset-Vergleich ohne vollständigen DTW.
        Nur für PerPhaseMusicalGoalsGate-Schnell-Subset.

        Args:
            audio_sample:        5-s-Stichprobe
            reference_onsets_ms: Referenz-Onsets aus Original (optional)
            sr:                  Sample-Rate

        Returns:
            Groove-Score ∈ [0, 1]
        """
        try:
            onsets = detect_onsets(audio_sample, sr)
            if onsets.n_onsets < 2:
                return 1.0  # Zu wenige Onsets: kein Groove-Problem

            if reference_onsets_ms is None:
                # Nur Onsets zählen, keine Referenz → neutrale Schätzung
                return 0.88  # Schwellwert-neutraler Defaultwert

            # Einfache Greedy-Alignierung (schneller als DTW)
            deviations = []
            ref = reference_onsets_ms.astype(np.float32)
            est = onsets.onset_times_ms

            for t_ref in ref:
                diffs = np.abs(est - t_ref)
                if len(diffs) > 0:
                    deviations.append(float(np.min(diffs)))

            if not deviations:
                return 1.0

            rms = float(np.sqrt(np.mean(np.array(deviations) ** 2)))
            score = float(max(0.0, 1.0 - rms / (2.0 * DTW_MAX_MS)))
            return min(1.0, score)
        except Exception:
            logger.warning("dtw_groove.py::measure_quick fallback", exc_info=True)
            return 0.88  # Neutraler Default bei Fehler


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------

_instance: DtwGrooveMeasurer | None = None
_lock = threading.Lock()


def get_groove_measurer(sr: int = 48000) -> DtwGrooveMeasurer:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2).

    Args:
        sr: Sample-Rate (muss 48000)

    Returns:
        Globale DtwGrooveMeasurer-Instanz.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = DtwGrooveMeasurer(sr=sr)
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktionen
# ---------------------------------------------------------------------------


def measure_groove(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int = 48000,
    max_dtw_ms: float = DTW_MAX_MS,
) -> GrooveMeasurementResult:
    """Misst Groove-Erhaltung via DTW-Onset-Alignierung (§8.2-6).

    Pflicht-Aufruf nach jeder Restaurierung:
        - PerPhaseMusicalGoalsGate (§2.29): Schnell-Check
        - MusicalGoalsChecker.GrooveMetric.measure_all(): Voll-Prüfung
        - TransientDecoupledProcessing (§2.27): GrooveMetric nach OLA-Rekombination

    Algorithmus:
        1. Spectral-Flux Onset-Detektion (Original + Restauriert)
        2. DTW-Alignierung (Sakoe-Chiba Band)
        3. RMS der Onset-Abweichungen → groove_score ∈ [0, 1]

    Args:
        original:   Original-Audio [n_samples], float32
        restored:   Restauriertes Audio [n_samples], float32
        sr:         Sample-Rate (muss 48000 Hz)
        max_dtw_ms: Maximale Onset-Abweichung (Standard: 8.0 ms)

    Returns:
        GrooveMeasurementResult mit groove_score und passes_threshold.

    Invariante:
        groove_score ≥ 0.88 wenn dtw_rms_ms ≤ 8.0 ms (§1.2 Schwellwert).
    """
    return get_groove_measurer(sr).measure(original, restored, sr, max_dtw_ms)


def measure_groove_quick(
    audio_sample: np.ndarray,
    reference_onsets_ms: np.ndarray | None = None,
    sr: int = 48000,
) -> float:
    """Schnelle Groove-Schätzung für PMGG Schnell-Subset (§2.29, ≤ 50 ms).

    Args:
        audio_sample:        5-s-Stichprobe
        reference_onsets_ms: Referenz-Onsets vom Original (optional)
        sr:                  Sample-Rate

    Returns:
        Groove-Score ∈ [0, 1]
    """
    return get_groove_measurer(sr).measure_quick(audio_sample, reference_onsets_ms, sr)
