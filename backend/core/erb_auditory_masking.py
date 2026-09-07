"""ERB-skaliertes Auditory-Masking-Modell für frequenzabhängige Salienz-Schätzung.

Ersetzt feste Breitband-Masking-Schwellen durch ein psychoakustisch korrektes
frequenzabhängiges Modell auf Basis von Equivalent Rectangular Bandwidth (ERB)
Kritischband-Filtern und dem Power-Spektrum-Modell des Maskierens.

Wissenschaftliche Grundlage:
- Glasberg, B.R. & Moore, B.C.J. (1990). "Derivation of auditory filter
  shapes from notched-noise data". *Hearing Research* 47, 103–138.
- Moore, B.C.J. & Glasberg, B.R. (1983). "Suggested formulae for calculating
  auditory-filter bandwidths and excitation patterns". *JASA* 74(3), 750–753.
- Moore, B.C.J., Glasberg, B.R. & Baer, T. (1997). "A model for the prediction
  of thresholds, loudness, and partial loudness". *JAES* 45(4), 224–240.
- Brungart, D.S. (2001). "Informational and energetic masking effects in the
  perception of two simultaneous talkers". *JASA* 109(3), 1101–1109.

Verbesserungen gegenüber dem Festschwellen-Modell:
1. **Frequenzabhängige Masking-Ausbreitung** über ERB-Erregungsmuster
   (tiefe Frequenzen maskieren breiter als hohe — asymmetrische Critical-Ratio)
2. **Exponentieller Zeitzerfall** (kein Stufenmodell) für Forward-Masking
   mit 3:1 Forward/Backward-Asymmetrie (Jesteadt, Bacon & Lehman 1982)
3. **Informational-Masking-Bonus** für harmonisch strukturierte Inhalte
   (Brungart 2001; senkt Salienz für Defekte in tonalen Passagen)

Modul-Invarianten (§3.x konform):
- Thread-sicheres Singleton via Double-Checked-Locking
- NaN/Inf-Guard auf allen numerischen Ausgaben
- Keine Audio-Veränderung — liefert nur Masking-Schwellen
- Kein Sample-Rate-Assert (Analyse-Modul — läuft bei nativem Import-SR)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ERB auditory filter model (Glasberg & Moore 1990)
# ---------------------------------------------------------------------------


def erb_hz(f_hz: float) -> float:
    """Equivalent Rectangular Bandwidth at frequency *f_hz*.

    ERB(f) = 24.7 * (4.37 * f/1000 + 1)   [Glasberg & Moore 1990, Eq. 3]
    """
    return 24.7 * (4.37 * f_hz / 1000.0 + 1.0)


def erb_rate(f_hz: float) -> float:
    """ERB-rate number (Cams) for frequency *f_hz*.

    N(f) = 21.4 * log10(0.00437*f + 1)     [Glasberg & Moore 1990, Eq. 4]
    """
    return float(21.4 * np.log10(0.00437 * f_hz + 1.0))


def erb_rate_to_hz(n_cams: float) -> float:
    """Konvertiert ERB-rate (Cams) back to Hz."""
    return float((10.0 ** (n_cams / 21.4) - 1.0) / 0.00437)


# ---------------------------------------------------------------------------
# Excitation spreading function
# ---------------------------------------------------------------------------


def _spreading_function_db(
    fc_masker: float,
    fc_signal: float,
) -> float:
    """Masking spread in dB as a function of frequency distance.

    Simplified roex(p) spreading function (Moore & Glasberg 1997):
    - Upper skirt (signal above masker): −24 dB/ERB
    - Lower skirt (signal below masker): −10 dB/ERB (upward masking is asymmetric)

    Returns the attenuation in dB of masking effect at *fc_signal*
    due to a masker at *fc_masker*.  A return of 0 means full masking;
    large negative values mean no masking.
    """
    delta_erb = erb_rate(fc_signal) - erb_rate(fc_masker)

    if abs(delta_erb) < 0.01:
        return 0.0  # same band — full masking
    elif delta_erb > 0:
        # Signal above masker (upward masking): shallower spread
        # Upward masking is stronger — excitation spreads UP (Moore 1997)
        return -10.0 * delta_erb
    else:
        # Signal below masker (downward masking): steeper drop
        return -24.0 * abs(delta_erb)


# ---------------------------------------------------------------------------
# Temporal masking decay
# ---------------------------------------------------------------------------


def _forward_masking_decay_db(dt_ms: float) -> float:
    """Forward masking decay in dB as a function of time after masker offset.

    Exponential decay model (Jesteadt, Bacon & Lehman 1982):
    ΔL = -20 * log10(1 + dt/τ)  where τ ≈ 10 ms for loud maskers.

    Effective range: ~200 ms.  Beyond that, returns -inf (no masking).
    """
    if dt_ms <= 0:
        return 0.0
    if dt_ms > 200.0:
        return -100.0
    tau_ms = 10.0
    return float(-20.0 * np.log10(1.0 + dt_ms / tau_ms))


def _backward_masking_decay_db(dt_ms: float) -> float:
    """Backward masking decay in dB.

    Much shorter than forward: effective range ~20 ms, roughly 1/3
    the strength of forward masking (Moore 2003).
    """
    if dt_ms <= 0:
        return 0.0
    if dt_ms > 20.0:
        return -100.0
    tau_ms = 3.3  # ~1/3 of forward masking τ
    return float(-20.0 * np.log10(1.0 + dt_ms / tau_ms))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ERBMaskingThreshold:
    """Masking threshold at a specific frequency band and time."""

    centre_freq_hz: float
    erb_width_hz: float
    threshold_db: float  # defect level must exceed this to be audible
    masking_type: str  # "simultaneous" | "temporal_forward" | "temporal_backward" | "combined"
    informational_bonus_db: float = 0.0  # extra masking for tonal content


@dataclass
class ERBMaskingResult:
    """Frequency-dependent masking analysis for a single defect event."""

    band_thresholds: list[ERBMaskingThreshold] = field(default_factory=list)
    mean_threshold_db: float = -100.0
    max_threshold_db: float = -100.0
    salience: float = 1.0  # 0.0 = fully masked, 1.0 = fully exposed
    dominant_masking_type: str = "none"


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


class ERBAuditoryMaskingModel:
    """Frequenzabhängiges Masking-Modell auf Basis von ERB-Kritischband-Filtern.

    Ersetzt die festen Breitband-Schwellen (−12/−8/−6 dB) im
    PerceptualSalienceEstimator durch ein psychoakustisch korrektes Modell
    das berücksichtigt:

    1.  Frequenzabhängige Kritische Bandbreite (eng bei tiefen, weit bei hohen Frequenzen)
    2.  Asymmetrische Ausbreitung (Aufwärts-Masking stärker als Abwärts-Masking)
    3.  Exponentieller Zeitzerfall (kein Stufenmodell)
    4.  Informational-Masking für harmonisch strukturierte Inhalte
    """

    _N_BANDS = 24  # ERB-Bänder von 50 Hz bis 16 kHz
    _F_LOW = 50.0
    _F_HIGH = 16000.0
    _CONTEXT_WINDOW_S = 0.4  # ±400 ms Kontext für simultanes Masking

    # Signal-zu-Masker-Abstand für Schwelle (Moore & Glasberg 1997)
    _SMR_ABSOLUTE_DB = 5.0  # defect must be ≥5 dB above masked threshold to be salient

    def __init__(self) -> None:
        # §Perf: Caches für centres und Spreading-Matrix — identisch pro SR, nur einmal berechnen.
        self._centres_cache: dict[float, np.ndarray] = {}
        self._spreading_matrix_cache: dict[bytes, np.ndarray] = {}
        # §Perf: Mono-Cache — _to_mono_f64 auf demselben Audio-Objekt 500× aufgerufen.
        # Jeder perceptual_salience-Durchlauf übergibt dasselbe Array → einmalige Konversion reicht.
        # Key: (id(audio), shape) — ausreichend eindeutig für eine Pipeline-Session.
        self._mono_cache: dict[tuple, np.ndarray] = {}
        self._mono_cache_lock = threading.Lock()  # §2.46h: Parallel-ERB teilt den Singleton
        # §Perf: ERB-Band-Masken-Cache — pro (n_fft, sr) vorberechnet.
        # Ersetzt 24× Python-Schleife mit mask/mean durch einen einzigen Matmul.
        # Key: (n_fft, sr, centres_bytes) → (mask_matrix (n_bands, n_rfft), band_sizes (n_bands,))
        self._band_mask_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def compute_masking_threshold(
        self,
        audio: np.ndarray,
        sr: int,
        defect_start_s: float,
        defect_end_s: float,
        defect_freq_range: tuple[float, float] | None = None,
    ) -> ERBMaskingResult:
        """Berechnet frequenzabhängige Masking-Schwelle an einer Defektstelle.

        Parameter
        ----------
        audio : np.ndarray
            Mono- oder Stereo-Audio bei nativem Sample-Rate.
        sr : int
            Sample-Rate in Hz.
        defect_start_s, defect_end_s : float
            Zeitliche Position des Defekts in Sekunden.
        defect_freq_range : tuple[float, float] | None
            Falls bekannt, der Frequenzbereich des Defekts (Hz).
            Bei None werden alle ERB-Bänder ausgewertet.

        Rückgabe
        -------
        ERBMaskingResult mit bandweisen Schwellen und aggregierter Salienz.
        """
        # §Perf: Mono-Cache — teures nan_to_num auf vollem Array nur einmal pro Audio-Objekt.
        _mono_key = (id(audio), audio.shape)
        if _mono_key not in self._mono_cache:
            # §2.46h (2026-09-06): Der Singleton wird jetzt von Parallel-ERB-Threads
            # geteilt — Cache-Mutation unter Lock (Wert ist deterministisch; ohne
            # Lock drohen doppelte Berechnung und 80-MB-Doppelhaltung).
            with self._mono_cache_lock:
                if _mono_key not in self._mono_cache:
                    if len(self._mono_cache) >= 2:
                        # Cache auf max 2 Einträge begrenzen — große Audio-Arrays (≥80 MB) nicht unbegrenzt halten
                        del self._mono_cache[next(iter(self._mono_cache))]
                    self._mono_cache[_mono_key] = self._to_mono_f64(audio)
        mono = self._mono_cache[_mono_key]
        n_samples = len(mono)
        duration_s = n_samples / sr
        nyquist = sr / 2.0

        # ERB centre frequencies
        centres = self._erb_centres(min(self._F_HIGH, nyquist - 1.0))

        # Compute short-time power spectrum around defect and context
        defect_power = self._band_power_at_time(
            mono,
            sr,
            centres,
            defect_start_s,
            defect_end_s,
        )

        # Context: ±400 ms around defect (excluding defect itself)
        ctx_before_start = max(0.0, defect_start_s - self._CONTEXT_WINDOW_S)
        ctx_after_end = min(duration_s, defect_end_s + self._CONTEXT_WINDOW_S)

        ctx_power_before = self._band_power_at_time(
            mono,
            sr,
            centres,
            ctx_before_start,
            defect_start_s,
        )
        ctx_power_after = self._band_power_at_time(
            mono,
            sr,
            centres,
            defect_end_s,
            ctx_after_end,
        )

        # Temporal distances for masking decay
        dt_forward_ms = max(0.0, (defect_start_s - ctx_before_start) * 1000.0 * 0.5)
        dt_backward_ms = max(0.0, (ctx_after_end - defect_end_s) * 1000.0 * 0.5)

        # Skalare Decay-Werte (identisch für alle Bänder — einmal berechnen)
        fwd_decay = _forward_masking_decay_db(dt_forward_ms)
        bwd_decay = _backward_masking_decay_db(dt_backward_ms)

        # Tonality estimate for informational masking
        tonality = self._estimate_tonality(mono, sr, defect_start_s, defect_end_s)
        info_bonus = 3.0 * tonality if tonality > 0.5 else 0.0

        band_thresholds: list[ERBMaskingThreshold] = []
        band_indices: list[int] = []  # §Perf: Band-Index für vektorisierten defect_levels-Block
        max_masking_db = -100.0
        dominant_type = "none"

        # §Perf: Vektorisierter Simultaneous-Masking-Block.
        # Spreading-Matrix (N×N) nur einmal pro SR berechnet/gecacht.
        # Ersetzt O(N²) Schleife mit 288.000 skalaren Aufrufen durch einen Numpy-Broadcast.
        spread_mat = self._get_spreading_matrix(centres)  # (n_masker, n_signal)
        # §Perf: Vektorisiertes _power_to_db — ersetzt 48+48 skalare np.log10-Aufrufe.
        ctx_before_db = np.where(
            ctx_power_before > 0,
            10.0 * np.log10(np.maximum(ctx_power_before, 1e-15)),
            -150.0,
        )  # (n,)
        ctx_after_db = np.where(
            ctx_power_after > 0,
            10.0 * np.log10(np.maximum(ctx_power_after, 1e-15)),
            -150.0,
        )  # (n,)
        ctx_levels_db = np.maximum(ctx_before_db, ctx_after_db)  # (n,)
        # simul_mask_db[i] = max_j( ctx_levels_db[j] + spread_mat[j,i] )
        simul_mask_db_vec = (ctx_levels_db[:, np.newaxis] + spread_mat).max(axis=0)  # (n,)
        fwd_mask_db_vec = ctx_before_db + fwd_decay  # (n,)
        bwd_mask_db_vec = ctx_after_db + bwd_decay  # (n,)
        threshold_db_vec = np.maximum(simul_mask_db_vec, np.maximum(fwd_mask_db_vec, bwd_mask_db_vec))
        if info_bonus > 0.0:
            threshold_db_vec = threshold_db_vec + info_bonus

        # §Perf: nan_to_num einmalig vektorisiert statt 24× im Per-Band-Loop (13501 Calls/0.349s).
        threshold_db_vec_clean = np.where(np.isnan(threshold_db_vec), -100.0, threshold_db_vec)

        for i, fc in enumerate(centres):
            if defect_freq_range is not None:
                f_lo, f_hi = defect_freq_range
                ew = erb_hz(fc)
                if fc + 0.5 * ew < f_lo or fc - 0.5 * ew > f_hi:
                    continue  # Defekt belegt dieses Band nicht

            ew = erb_hz(fc)
            thr_db = float(threshold_db_vec_clean[i])
            simul_i = float(simul_mask_db_vec[i])
            fwd_i = float(fwd_mask_db_vec[i])
            bwd_i = float(bwd_mask_db_vec[i])

            if simul_i >= max(fwd_i, bwd_i):
                m_type = "simultaneous"
            elif fwd_i >= bwd_i:
                m_type = "temporal_forward"
            else:
                m_type = "temporal_backward"

            if thr_db > max_masking_db:
                max_masking_db = thr_db
                dominant_type = m_type

            band_thresholds.append(
                ERBMaskingThreshold(
                    centre_freq_hz=float(fc),
                    erb_width_hz=float(ew),
                    threshold_db=thr_db,  # §Perf: bereits NaN-bereinigt via threshold_db_vec_clean
                    masking_type=m_type,
                    informational_bonus_db=float(info_bonus),
                )
            )
            band_indices.append(i)

        if not band_thresholds:
            return ERBMaskingResult(salience=1.0, dominant_masking_type="none")

        # Compute aggregate salience
        # §Perf: Vektorisiert — kein O(n) list.index()-Lookup, kein skalarer _power_to_db-Loop
        if band_thresholds:
            valid_idx = np.array(band_indices, dtype=np.intp)
            dp_valid = defect_power[valid_idx]
            defect_power_db = np.where(
                dp_valid > 0,
                10.0 * np.log10(np.maximum(dp_valid, 1e-15)),
                -150.0,
            )
            thresholds_for_salience = threshold_db_vec[valid_idx]
            excess = float(np.mean(defect_power_db - thresholds_for_salience))
            salience = float(
                np.clip(
                    (excess + self._SMR_ABSOLUTE_DB) / (2.0 * self._SMR_ABSOLUTE_DB),
                    0.0,
                    1.0,
                )
            )
            mean_thresh = float(np.mean(thresholds_for_salience))
            max_thresh = float(np.max(thresholds_for_salience))
        else:
            salience = 1.0
            mean_thresh = -100.0
            max_thresh = -100.0

        salience = float(np.nan_to_num(salience, nan=0.5))

        result = ERBMaskingResult(
            band_thresholds=band_thresholds,
            mean_threshold_db=float(np.nan_to_num(mean_thresh, nan=-100.0)),
            max_threshold_db=float(np.nan_to_num(max_thresh, nan=-100.0)),
            salience=salience,
            dominant_masking_type=dominant_type,
        )

        logger.debug(
            "ERB-Masking: %.0f–%.0f Hz, %d Bänder, Schwelle=%.1f dB, Salienz=%.3f, dominant=%s, Tonalität=%.2f",
            centres[0] if len(centres) > 0 else 0,
            centres[-1] if len(centres) > 0 else 0,
            len(band_thresholds),
            mean_thresh,
            salience,
            dominant_type,
            tonality,
        )

        return result

    # ------------------------------------------------------------------
    # Convenience: salience for broadband defect
    # ------------------------------------------------------------------

    def clear_session_caches(self) -> None:
        """Leert Sitzungs-Caches nach DefectScanner-Phase (Speicher-Freigabe).

        Sollte vom PerceptualSalienceEstimator nach Abschluss des Scans aufgerufen
        werden, um große Audio-Arrays (≥80 MB) aus dem Singleton-Speicher zu entfernen.
        """
        self._mono_cache.clear()

    def estimate_salience(
        self,
        audio: np.ndarray,
        sr: int,
        defect_start_s: float,
        defect_end_s: float,
    ) -> float:
        """Schnelle Salienz-Schätzung (0.0–1.0) für einen breitbandigen Defekt.

        Vereinfachter Wrapper um compute_masking_threshold.
        """
        result = self.compute_masking_threshold(
            audio,
            sr,
            defect_start_s,
            defect_end_s,
        )
        return result.salience

    # ------------------------------------------------------------------
    # Internal: spectral analysis
    # ------------------------------------------------------------------

    def _erb_centres(self, f_max: float) -> np.ndarray:
        """Generiert ERB-Mittenfrequenzen bis *f_max*. Gecacht pro f_max."""
        key = round(f_max, 1)
        if key not in self._centres_cache:
            f_high = min(self._F_HIGH, f_max)
            n_low = erb_rate(self._F_LOW)
            n_high = erb_rate(f_high)
            n_vals = np.linspace(n_low, n_high, self._N_BANDS)
            self._centres_cache[key] = (10.0 ** (n_vals / 21.4) - 1.0) / 0.00437
        return self._centres_cache[key]

    def _get_spreading_matrix(self, centres: np.ndarray) -> np.ndarray:
        """Precomputed N×N Spreading-Matrix. spread_mat[j, i] = spreading von Masker j zu Signal i.

        Ergebnis gecacht per centres-Fingerprint — wird nur einmal pro SR berechnet.
        Reduziert 288.000 skalare _spreading_function_db-Aufrufe auf einen einzigen
        Numpy-Broadcast (roex-Formel vektorisiert nach Moore & Glasberg 1997).
        """
        key = centres.tobytes()
        if key not in self._spreading_matrix_cache:
            erb_rates = np.array([erb_rate(fc) for fc in centres])  # shape (n,)
            # delta_erb[j, i] = erb_rate(signal_i) - erb_rate(masker_j)
            delta_erb = erb_rates[np.newaxis, :] - erb_rates[:, np.newaxis]  # (n_masker, n_signal)
            mat = np.where(
                np.abs(delta_erb) < 0.01,
                0.0,
                np.where(
                    delta_erb > 0,
                    -10.0 * delta_erb,  # Signal über Masker: flache Flanke
                    24.0 * delta_erb,  # Signal unter Masker: steile Flanke (-24*|delta|)
                ),
            )
            self._spreading_matrix_cache[key] = mat
        return self._spreading_matrix_cache[key]

    def _get_band_masks(
        self,
        n_fft: int,
        sr: int,
        centres: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vorberechnete ERB-Band-Masken als dichte Matrix. Gecacht per (n_fft, sr).

        Returns
        -------
        mask_mat : np.ndarray, shape (n_bands, n_rfft)
            Boolean-Matrix: mask_mat[i, k] = True wenn FFT-Bin k in Band i liegt.
        band_sizes : np.ndarray, shape (n_bands,)
            Anzahl Bins pro Band (für Mittelwert-Normierung). Mindestens 1.
        """
        key = (n_fft, sr, centres.tobytes())
        if key not in self._band_mask_cache:
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)  # (n_rfft,)
            n_rfft = len(freqs)
            mask_mat = np.zeros((len(centres), n_rfft), dtype=bool)
            for i, fc in enumerate(centres):
                ew = erb_hz(fc)
                f_lo = max(0.0, fc - 0.5 * ew)
                f_hi = fc + 0.5 * ew
                mask_mat[i] = (freqs >= f_lo) & (freqs <= f_hi)
            band_sizes = np.maximum(1, mask_mat.sum(axis=1)).astype(np.float64)  # (n_bands,)
            self._band_mask_cache[key] = (mask_mat.astype(np.float64), band_sizes)
        return self._band_mask_cache[key]

    def _band_power_at_time(
        self,
        mono: np.ndarray,
        sr: int,
        centres: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> np.ndarray:
        """Berechnet power in each ERB band for the given time range.

        Returns array of shape (n_bands,) with mean power per band.
        """
        s = max(0, int(t_start * sr))
        e = min(len(mono), int(t_end * sr))
        if e <= s:
            return np.full(len(centres), 1e-15, dtype=np.float64)  # type: ignore[no-any-return]

        segment = mono[s:e]

        # FFT
        n_fft = max(256, min(8192, len(segment)))
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))

        # §Perf: rfft.real²+rfft.imag² vermeidet sqrt in np.abs() (Leistungsspektrum).
        _rfft_out = np.fft.rfft(segment[:n_fft])
        spectrum = _rfft_out.real**2 + _rfft_out.imag**2

        # §Perf: Vektorisierte Band-Power-Berechnung via vorberechneter Masken-Matrix.
        # Ersetzt 24× Python-Schleife (mask + mean) durch einen einzigen Matrix-Vektor-Produkt.
        mask_mat, band_sizes = self._get_band_masks(n_fft, sr, centres)
        # mask_mat @ spectrum = Summe der Spektral-Energie pro Band (n_bands,)
        powers = np.maximum(1e-15, (mask_mat @ spectrum) / band_sizes)

        return np.asarray(powers, dtype=np.float64)  # type: ignore[no-any-return]

    def _estimate_tonality(
        self,
        mono: np.ndarray,
        sr: int,
        t_start: float,
        t_end: float,
    ) -> float:
        """Schätzt tonality of audio segment (0.0 = noise, 1.0 = pure tone).

        Uses spectral flatness (Wiener entropy) as a proxy.
        Low flatness = tonal, high flatness = noise-like.
        """
        s = max(0, int(t_start * sr))
        e = min(len(mono), int(t_end * sr))
        if e - s < 256:
            return 0.5

        segment = mono[s:e]
        n_fft = min(4096, len(segment))
        spectrum = np.abs(np.fft.rfft(segment[:n_fft])) ** 2 + 1e-15

        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness = geo_mean / (arith_mean + 1e-15)

        # Convert flatness to tonality: flat=1 → tonal=0, flat=0 → tonal=1
        tonality = float(np.clip(1.0 - flatness, 0.0, 1.0))
        return float(np.nan_to_num(tonality, nan=0.5))

    @staticmethod
    def _power_to_db(power: float) -> float:
        """Konvertiert Leistung nach dB mit unterem Grenzwert."""
        return float(10.0 * np.log10(max(power, 1e-15)))

    @staticmethod
    def _to_mono_f64(audio: np.ndarray) -> np.ndarray:
        """Konvertiert nach Mono float64 mit NaN/Inf-Guard.

        Kanonisches Aurik-Format: (N, Kanäle) — Achse 1 ist die Kanal-Dimension.
        Für (N, 2): shape[0]=N >> shape[1]=2 → mean(axis=1) → N-elementiger Mono-Vektor.
        Für (2, N): shape[0]=2 <  shape[1]=N  → mean(axis=0) → N-elementiger Mono-Vektor.
        FALSCH: mean(axis=0) auf (N,2) liefert einen 2-elementigen Vektor → alle nachgelagerten
        FFT/Band-Power-Berechnungen kollabieren → salience=0.000 bei jedem Aufruf.
        """
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            # Detect Aurik (N, channels) vs. legacy (channels, N):
            # The channel count is always ≤ 2; the sample count is always >> 2.
            arr = arr.mean(axis=1) if arr.shape[0] > arr.shape[1] else arr.mean(axis=0)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Thread-safe singleton (double-checked locking — §3.2)
# ---------------------------------------------------------------------------

_instance: ERBAuditoryMaskingModel | None = None
_lock = threading.Lock()


def get_erb_auditory_masking_model() -> ERBAuditoryMaskingModel:
    """Gibt thread-sicheres Singleton-ERBAuditoryMaskingModel zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ERBAuditoryMaskingModel()
    return _instance
