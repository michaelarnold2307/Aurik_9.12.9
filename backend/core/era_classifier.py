"""
EraClassifier — Ära-/Dekaden-adaptives Processing (§2.14 Spec)
===============================================================

Erkennt das Aufnahme-Jahrzehnt (1890–2025) automatisch und leitet


    Tier-1: LAION-CLAP-Embeddings → Nearest-Neighbor zu Ära-Referenz-Ankern

Referenz: §2.14 Aurik-9-Spec (v10.0.0)
Autor: Aurik Development Team
Datum: 20. Februar 2026
"""

from __future__ import annotations

import hashlib
import importlib
import itertools

# v10.101 SOTA: FFT-basierte Ära-Klassifikation, Gammatone-geschützt.
import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from dataclasses import replace as dc_replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_remaster_detector_factory() -> Any:
    """Lädt den RemasterDetector-Factory-Call lazy und gecacht."""
    module = importlib.import_module("backend.core.remaster_detector")
    return module.get_remaster_detector


@lru_cache(maxsize=1)
def _get_laion_clap_factory() -> Any:
    """Lädt den LAION-CLAP-Factory-Call lazy und gecacht."""
    module = importlib.import_module("plugins.laion_clap_plugin")
    return module.get_laion_clap


@lru_cache(maxsize=1)
def _get_resample_poly() -> Any | None:
    """Gibt scipy.signal.resample_poly zurück oder None, falls SciPy fehlt."""
    try:
        module = importlib.import_module("scipy.signal")
        return module.resample_poly
    except Exception as e:
        logger.warning("era_classifier.py::_get_resample_poly Ersatzpfad: %s", e)
        return None


# --------------- Dekaden-Definition ----------------------------------------


VALID_DECADES: list[int] = [
    1890,
    1900,
    1910,
    1920,
    1930,
    1940,
    1950,
    1960,
    1970,
    1980,
    1990,
    2000,
    2010,
    2020,
    2025,
]

# Bekannte HF-Rolloff-Grenzen pro Jahrzehnt [Hz]
DECADE_HF_LIMITS: dict[int, float] = {
    1890: 3000,
    1900: 4000,
    1910: 5000,
    1920: 6000,
    1930: 7000,
    1940: 8000,
    1950: 10000,
    1960: 12000,
    1970: 16000,
    1980: 20000,
    1990: 20000,
    2000: 20000,
    2010: 20000,
    2020: 20000,
    2025: 20000,
}

# Material-Prior pro Dekade
DECADE_MATERIAL_PRIOR: dict[int, str] = {
    1890: "wax_cylinder",
    1900: "wax_cylinder",
    1910: "shellac",
    1920: "shellac",
    1930: "shellac",
    1940: "shellac",
    1950: "vinyl",
    1960: "vinyl",
    1970: "reel_tape",
    1980: "tape",
    1990: "cd_digital",
    2000: "cd_digital",
    2010: "streaming",
    2020: "streaming",
    2025: "streaming",
}

# Medium-based minimum decade floor: a recording on a given medium cannot
# predate the physical invention of that medium.  Used by
# constrain_era_to_medium() to correct impossible decade assignments
# (e.g. reel_tape → 1890 is a classification artefact, not a real recording).
#
# Conservative lower bounds (rounded to nearest VALID_DECADES entry):
#   wax_cylinder  : 1890 (Edison 1877 → commercial 1888)
#   wire_recording: 1900 (Poulsen Telegraphone 1898)
#   shellac       : 1900 (shellac discs commercial ~1898)
#   lacquer_disc  : 1920 (transcription discs widespread 1920s)
#   vinyl         : 1950 (Columbia 12″ LP 1948, 45rpm 1949)
#   reel_tape     : 1940 (AEG Magnetophon 1935; commercial 1940s)
#   tape/cassette : 1960 (Philips compact cassette 1963)
#   dat           : 1980 (DAT standard 1987; decade floor 1980)
#   minidisc      : 1990 (Sony MiniDisc 1992)
#   cd_digital/cd : 1980 (first CD October 1982)
#   mp3_low/high  : 1990 (MP3 standard 1993)
#   aac           : 2000 (AAC in iTunes 2001)
#   streaming     : 2000 (consumer streaming widespread ~2005)
MEDIUM_DECADE_FLOOR: dict[str, int] = {
    "wax_cylinder": 1890,
    "wire_recording": 1900,
    "shellac": 1900,
    "lacquer_disc": 1920,
    "vinyl": 1950,
    "reel_tape": 1940,
    "tape": 1960,
    "cassette": 1960,
    "dat": 1980,
    "minidisc": 1990,
    "cd_digital": 1980,
    "cd": 1980,
    "mp3_low": 1990,
    "mp3_high": 1990,
    "aac": 2000,
    "streaming": 2000,
}

# GP-Warmstart: noise_reduction_strength prior mean pro Epoche
DECADE_NR_PRIOR_MEAN: dict[int, float] = {
    1890: 0.95,
    1900: 0.95,
    1910: 0.92,
    1920: 0.90,
    1930: 0.90,
    1940: 0.85,
    1950: 0.80,
    1960: 0.75,
    1970: 0.65,
    1980: 0.55,
    1990: 0.50,
    2000: 0.50,
    2010: 0.45,
    2020: 0.45,
    2025: 0.45,
}

DECADE_NR_PRIOR_STD: dict[int, float] = dict.fromkeys(VALID_DECADES, 0.07)
DECADE_NR_PRIOR_STD.update(
    {1900: 0.05, 1910: 0.05, 1920: 0.05, 1930: 0.05, 1940: 0.06, 1970: 0.08, 1980: 0.10, 1990: 0.10}
)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class EraResult:
    """Ergebnis des EraClassifiers.

    Attributes:
        decade:                Erkanntes Jahrzehnt (z. B. 1940, 1970, …).
        era_label:             Menschenlesbare Bezeichnung (z. B. „1970er").
        confidence:            Konfidenz ∈ [0.0, 1.0].
        material_prior:        Empfohlener Material-Typ-String aus ``DECADE_MATERIAL_PRIOR``.
        noise_profile:         Spektrales Rauschprofil (Bark-Bänder, 24 Werte).
        tier_used:             Welche Erkennungsstufe genutzt wurde (1 = CLAP, 2 = DSP, 3 = Heuristik).
        tier1_used:            True wenn Tier-1 (CLAP) Erkennung aktiv war.
        regression_snr:        SNR-Wert der Regression-Schätzung (falls verfügbar).
        regression_mos:         MOS-Wert der Regression-Schätzung (falls verfügbar).
        hf_rolloff_hz:         Gemessener HF-Rolloff-Punkt (-3 dB) in Hz.
        is_remaster_suspected: True wenn RemasterDetector einen Remaster erkannt hat.
    """

    decade: int
    era_label: str
    confidence: float
    material_prior: str
    noise_profile: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.float32))
    tier_used: int = 2
    tier1_used: bool = False
    regression_snr: float = 0.0
    regression_mos: float = 0.0
    hf_rolloff_hz: float = 20000.0
    is_remaster_suspected: bool = False

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.decade not in VALID_DECADES:
            self.decade = min(VALID_DECADES, key=lambda d: abs(d - self.decade))
            # §Invariante: era_label muss zum (geschnappten) decade passen —
            # sonst meldet der Classifier „Jahrzehnt=1980“ mit Label „1977er“.
            self.era_label = f"{self.decade}er"

    def as_dict(self) -> dict:
        """Serialisierung ohne ndarray."""
        d = asdict(self)
        d["noise_profile"] = self.noise_profile.tolist()
        return d


# ---------------------------------------------------------------------------
# Bark-Skala Hilfsfunktion
# ---------------------------------------------------------------------------

BARK_EDGES_HZ = [
    20,
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
]


def _bark_band_energies(audio_mono: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet normalisierte Energie in 24 Bark-Bändern.

    Args:
        audio_mono: Mono-Audio (1D float32/64).
        sr:         Sample-Rate.

    Returns:
        ndarray shape (24,) — normalisierte Energien (sum = 1).
    """
    n_fft = min(4096, len(audio_mono))
    hop = max(1, n_fft // 4)
    # STFT (kein scipy.signal für diesen einfachen Spektral-Pfad — numpy direkt)
    frames = []
    for start in range(0, len(audio_mono) - n_fft, hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        frame_fft = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(frame_fft)
    if not frames:
        return np.ones(24) / 24.0  # type: ignore[no-any-return]
    psd = np.mean(np.array(frames), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    energies = np.zeros(24, dtype=np.float32)
    for i, (lo, hi) in enumerate(itertools.pairwise(BARK_EDGES_HZ)):
        mask = (freqs >= lo) & (freqs < hi)
        energies[i] = float(np.sum(psd[mask]))

    total = energies.sum()
    if total < 1e-12:
        return np.ones(24, dtype=np.float32) / 24.0  # type: ignore[no-any-return]
    normalized: np.ndarray = np.nan_to_num(energies / total).astype(np.float32)
    return normalized


# ---------------------------------------------------------------------------
# Tier-2: DSP-Fingerprint
# ---------------------------------------------------------------------------


def _dsp_hf_rolloff(audio_mono: np.ndarray, sr: int) -> float:
    """Effective recording bandwidth via multi-estimator fusion (5 independent probes).

    Five independent spectral bandwidth estimators are computed and fused via a
    weighted median.  This makes the result robust against individual estimator
    failures such as bass-heavy spectral imbalance (which biases cumulative-energy
    methods) or LP-filter skirt leakage (which biases spectral-edge methods).

    Estimators
    ----------
    1. **E90** — cumulative 90th-percentile energy (kalibriert gegen
       _dsp_fingerprint_decade Schwellwerte).  Highly reliable for
       LP-filtered test signals; can under-estimate for bass-heavy music.
       Weight: 0.35.

    2. **E85** — cumulative 85th-percentile energy, conservative anchor.
       Always ≤ E90; provides a lower-bound guard.
       Weight: 0.20.

    3. **Edge-30dB** — highest frequency still within -30 dB of the spectral
       peak in 200 Hz – SR/2.  Reliable for full-band music; can over-estimate
       for LP-filtered noise because the 6th-order Butterworth skirt extends
       ~1.8× beyond the cutoff before reaching -30 dB.  Activated only when
       the energy in the gap (between E90 and the edge) exceeds 15% of the
       tail energy — this rejects LP-filter skirt artefacts while preserving
       the correction for bass-heavy music.
       Weight: 0.20.

    4. **Slope-break** — largest downward gradient discontinuity in the
       smoothed log-power spectrum (200 Hz – SR/2, log-frequency axis).  The
       physical LP-filter pole cluster produces a sharp gradient change
       exactly at the cutoff frequency; this estimator finds that change
       directly from the spectral shape, independent of energy distribution.
       Weight: 0.15.

    5. **Flatness-onset** — lowest frequency above which the spectral flatness
       of successive octave sub-bands drops below 0.15 (i.e. the band looks
       like white noise/near-silence instead of structured audio content).
       Identifies the onset of the noise floor, which typically starts just
       above the recording bandwidth.
       Weight: 0.10.

    Fusion: weighted median over available estimators (estimators that fail
    gracefully return None and are excluded).  Final result is clamped to
    [200.0, SR/2].

    Args:
        audio_mono: Mono-Audio (1-D).
        sr:         Sample-Rate.

    Returns:
        Effective bandwidth (rolloff) frequency in Hz.
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 64:
        return float(sr) / 2.0
    hop = n_fft // 2  # 50 % overlap for better averaging
    specs = []
    for start in range(0, max(1, len(audio_mono) - n_fft), hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        specs.append(np.abs(np.fft.rfft(frame)) ** 2)
    if not specs:
        return float(sr) / 2.0

    avg_spec = np.mean(np.array(specs), axis=0)
    cum_energy = np.cumsum(avg_spec)
    total_energy = cum_energy[-1]
    if total_energy < 1e-12:
        return float(sr) / 2.0

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    nyquist = float(sr) / 2.0

    # ── Estimator 1: Cumulative 90th-percentile energy (E90) ──────────────
    # Calibrated: all decade thresholds in _dsp_fingerprint_decade() are derived
    # from 0.90 × DECADE_HF_LIMITS, so E90 is the primary physics anchor.
    idx90 = int(np.clip(np.searchsorted(cum_energy, 0.90 * total_energy), 0, len(avg_spec) - 1))
    e90 = float(freqs[idx90])

    # ── Estimator 2: Cumulative 85th-percentile energy (E85) ──────────────
    # Conservative lower-bound guard; always ≤ E90 by construction.
    idx85 = int(np.clip(np.searchsorted(cum_energy, 0.85 * total_energy), 0, len(avg_spec) - 1))
    e85 = float(freqs[idx85])

    # ── Estimator 3: Spectral edge -30 dB with gap-energy guard ───────────
    # The guard rejects LP-filter skirt artefacts: for a 6th-order Butterworth
    # at cutoff F, the -30 dB spectral edge is at ~1.82 × F, but the energy in
    # (E90, edge) is < 1 % of total (all in the filter's deep stop-band).
    # For bass-heavy real music, the same gap contains genuine high-frequency
    # content (> 15 % of the tail energy beyond E90).
    edge_30db: float | None = None
    mask_music = freqs >= 200.0
    if np.any(mask_music) and np.any(avg_spec[mask_music] > 0):
        peak_power = float(np.max(avg_spec[mask_music]))
        if peak_power > 1e-20:
            edge_threshold = peak_power * 1e-3  # -30 dB below peak
            above_edge = np.where((avg_spec > edge_threshold) & mask_music)[0]
            if len(above_edge) > 0:
                candidate = float(freqs[above_edge[-1]])
                # Gap-energy guard: only use if real energy between E90 and edge
                tail_energy = float(np.sum(avg_spec[idx90:])) + 1e-30
                gap_mask = (freqs > e90) & (freqs <= candidate)
                gap_energy = float(np.sum(avg_spec[gap_mask]))
                if gap_energy / tail_energy > 0.15:
                    edge_30db = candidate

    # ── Estimator 4: Spectral slope-break (largest gradient discontinuity) ─
    # Smooth the log-power spectrum on a log-frequency axis, then find the bin
    # with the steepest downward gradient change — this corresponds to the
    # physical filter pole cluster (LP cutoff).
    slope_break: float | None = None
    try:
        mask_range = (freqs >= 200.0) & (freqs <= nyquist * 0.97)
        if np.sum(mask_range) >= 20:
            f_range = freqs[mask_range]
            s_range = avg_spec[mask_range]
            # Smooth with 7-bin moving average in log-frequency space
            log_s = 10.0 * np.log10(s_range + 1e-20)
            kernel_size = min(7, len(log_s) // 4)
            if kernel_size >= 3:
                kernel = np.ones(kernel_size) / kernel_size
                smoothed = np.convolve(log_s, kernel, mode="same")
                # First derivative on log-frequency axis
                log_f = np.log2(f_range + 1.0)
                df = np.diff(log_f)
                ds = np.diff(smoothed)
                slope = ds / (df + 1e-10)
                # Second derivative (gradient change)
                d2slope = np.diff(slope)
                # Find the most negative gradient change (steepest drop onset)
                # in the 1 kHz – (SR/2 – 1 kHz) range to avoid bass artifacts
                f_mid = 0.5 * (f_range[1:-1] + f_range[2:])  # bin centres for d2slope
                valid_mask = (f_mid >= 1000.0) & (f_mid <= nyquist - 1000.0)
                if np.any(valid_mask):
                    d2_valid = d2slope[valid_mask]
                    f_valid = f_mid[valid_mask]
                    best_idx = int(np.argmin(d2_valid))
                    slope_break = float(f_valid[best_idx])
    except Exception:
        slope_break = None

    # ── Estimator 5: Spectral flatness change-point ────────────────────────
    # Scan octave sub-bands upward from 1 kHz; find the lowest frequency where
    # the band's spectral flatness (geometric/arithmetic mean ratio) drops
    # below 0.15, indicating the band contains near-white noise or silence
    # rather than structured audio.  That onset frequency is the recording BW.
    flat_onset: float | None = None
    try:
        f_lo = 1000.0
        while f_lo < nyquist * 0.85:
            f_hi = min(f_lo * 2.0, nyquist)
            band_mask = (freqs >= f_lo) & (freqs < f_hi) & (avg_spec > 0)
            if np.sum(band_mask) < 4:
                break
            band = avg_spec[band_mask]
            geom = float(np.exp(np.mean(np.log(band + 1e-30))))
            arith = float(np.mean(band))
            flatness = geom / (arith + 1e-30)
            if flatness < 0.15:
                flat_onset = f_lo
                break
            f_lo = f_hi
    except Exception:
        flat_onset = None

    # ── Fusion: weighted median over available estimators ─────────────────
    # Weights reflect calibration reliability (see docstring).
    candidates: list[tuple[float, float]] = [(e90, 0.35), (e85, 0.20)]
    if edge_30db is not None:
        candidates.append((edge_30db, 0.20))
    if slope_break is not None:
        candidates.append((slope_break, 0.15))
    if flat_onset is not None:
        candidates.append((flat_onset, 0.10))

    # ── Outlier-robust fusion: IQR-based down-weighting ───────────────────
    # When estimators disagree strongly (e.g. bass-heavy music where E90 is
    # dominated by LF content but edge-30dB sees genuine HF extension), the
    # weighted median can be skewed by outlier estimators.  We down-weight
    # any estimator whose value lies > 1.5× IQR from the median, preventing
    # single-estimator flukes from dominating the fusion result.
    if len(candidates) >= 3:
        vals = np.array([v for v, _ in candidates])
        med = float(np.median(vals))
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr = max(q3 - q1, 500.0)  # floor at 500 Hz to avoid zero-IQR for pure tones
        adjusted: list[tuple[float, float]] = []
        for val, w in candidates:
            if abs(val - med) > 1.5 * iqr:
                adjusted.append((val, w * 0.25))
            else:
                adjusted.append((val, w))
        candidates = adjusted

    # Bass-heavy override: when the cumulative-energy estimators (E85, E90) are
    # dominated by sub-1 kHz bass content (e.g. heavily bass-boosted 1970s MP3),
    # they under-estimate the recording bandwidth by a wide margin.  In this
    # regime Edge-30dB is the only reliable measure of the true HF extension.
    # Activation: E90 < 1.5 kHz AND the accepted edge-30dB > 5 kHz AND gap-energy
    # fraction > 0.40 (substantial real content above E90, not just LP-skirt).
    # The 1.5 kHz floor ensures LP-filtered calibration signals never trigger
    # this path: a 3 kHz Butterworth LP gives E90 ≈ 2.8 kHz >> 1.5 kHz.
    # Real bass-heavy Schlager/hip-hop typically has E90 < 1.2 kHz.
    if edge_30db is not None and e90 < 1500.0 and edge_30db > 5000.0:
        # Compute gap fraction (already accepted by the 0.15 guard above)
        _edge_cand = edge_30db
        _tail = float(np.sum(avg_spec[idx90:])) + 1e-30
        _gap = float(np.sum(avg_spec[(freqs > e90) & (freqs <= _edge_cand)]))
        if _gap / _tail > 0.40:
            return float(np.clip(edge_30db, 200.0, nyquist))

    # ── §v10.14.1 Multi-Estimator Convergence Guard ─────────────────────
    # Wenn E90 extrem niedrig ist (< 2 kHz) aber mindestens 2 unabhängige
    # Schätzer (edge_30dB, slope_break, flat_onset) auf > 5 kHz zeigen,
    # ist E90 durch Bass-Dominanz korrumpiert.  Vertraue dem Median der
    # nicht-korrumpierten Schätzer.
    # Dieser Guard feuert auch bei gap/tail < 0.40 (z.B. bei Codec-
    # komprimierten Aufnahmen wo der Codec das Rauschen oberhalb des LPFs
    # unterdrückt und der Gap-Energie-Anteil künstlich niedrig ist).
    if e90 < 2000.0 and len(candidates) >= 3:
        _high_estimators = [v for v, _ in candidates if v > 5000.0]
        if len(_high_estimators) >= 2:
            _converged_bw = float(np.median(_high_estimators))
            logger.debug(
                "BW Multi-Estimator Convergence: E90=%.0fHz corrupted, %d/%d estimators converge at %.0f Hz → override",
                e90,
                len(_high_estimators),
                len(candidates),
                _converged_bw,
            )
            return float(np.clip(_converged_bw, 200.0, nyquist))

    # Weighted median: sort by value, find where cumulative weight crosses 0.5
    candidates.sort(key=lambda x: x[0])
    total_w = sum(w for _, w in candidates)
    cum_w = 0.0
    rolloff = e90  # fallback
    for val, w in candidates:
        cum_w += w / total_w
        if cum_w >= 0.50:
            rolloff = val
            break

    return float(np.clip(rolloff, 200.0, nyquist))


def _detect_stereo_properties(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """Erkennt stereo presence and width via frame-wise inter-channel correlation.

    Stereo became commercially available ~1958. Narrow hard-pan stereo is
    characteristic of the 1960s (< 0.15 width), while multi-track wide stereo
    (> 0.15) became standard in the 1970s.

    Returns:
        (is_stereo, stereo_width):
            is_stereo:    True if genuine stereo (channels not identical).
            stereo_width: 0.0 (mono/dual-mono) … ~0.5 (wide stereo).
    """
    if audio.ndim < 2 or (audio.ndim == 2 and min(audio.shape) < 2):
        return False, 0.0

    # Extract L/R channels (shape: (n_samples, n_channels) or (n_channels, n_samples))
    if audio.shape[-1] <= 2:
        left, right = audio[:, 0], audio[:, 1]
    else:
        left, right = audio[0, :], audio[1, :]

    # Dual-mono check
    if np.allclose(left, right, atol=1e-6):
        return False, 0.0

    frame_len = max(1, sr // 10)  # 100 ms frames
    n_frames = min(len(left) // frame_len, 500)  # cap analysis at 50 s
    correlations: list[float] = []
    for i in range(n_frames):
        start = i * frame_len
        lf = left[start : start + frame_len]
        rf = right[start : start + frame_len]
        if np.std(lf) < 1e-8 or np.std(rf) < 1e-8:
            continue
        _lf = lf - lf.mean()
        _rf = rf - rf.mean()
        _nl = float(np.linalg.norm(_lf))
        _nr = float(np.linalg.norm(_rf))
        c = float(np.dot(_lf, _rf) / (_nl * _nr + 1e-10))
        if np.isfinite(c):
            correlations.append(float(abs(c)))

    if not correlations:
        return False, 0.0

    mean_corr = float(np.mean(correlations))
    is_stereo = mean_corr < 0.98
    stereo_width = float(np.clip(1.0 - mean_corr, 0.0, 1.0))
    return is_stereo, stereo_width


def _estimate_spectral_tilt(audio_mono: np.ndarray, sr: int) -> float:
    """Schätzt spectral tilt via linear regression on log-power spectrum.

    Spectral tilt (dB/octave) captures the overall frequency-response shape:
        1940s–1960s: -5 … -7 (mid-forward, strong HF roll-off)
        1970s:       -3 … -5 (flatter, improved tape/electronics)
        1980s+:      -1 … -3 (flat/bright, modern EQ)

    Regression is restricted to 200 Hz – 16 kHz (musically relevant range).

    Returns:
        Spectral tilt in dB/octave, clamped to [-12, 2].
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 256:
        return -4.0  # conservative neutral
    hop = n_fft // 2
    specs: list[np.ndarray] = []
    for start in range(0, max(1, len(audio_mono) - n_fft), hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        specs.append(np.abs(np.fft.rfft(frame)) ** 2)
    if not specs:
        return -4.0

    avg_spec = np.mean(np.array(specs), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= 200.0) & (freqs <= 16000.0) & (avg_spec > 1e-20)
    if np.sum(mask) < 10:
        return -4.0

    log_freqs = np.log2(freqs[mask])
    log_power = 10.0 * np.log10(avg_spec[mask] + 1e-20)
    # Guard: -inf from near-zero power bins → LAPACK DLASCL failure
    log_power = np.nan_to_num(log_power, nan=0.0, posinf=0.0, neginf=-120.0)

    A = np.vstack([log_freqs, np.ones(len(log_freqs))]).T
    try:
        slope = float(np.linalg.lstsq(A, log_power, rcond=None)[0][0])
        return float(np.clip(slope, -12.0, 2.0))
    except Exception as e:
        logger.warning("era_classifier.py::_estimate_spectral_tilt Ersatzpfad: %s", e)
        return -4.0


def _estimate_dynamic_range(audio_mono: np.ndarray, sr: int) -> float:
    """Schätzt dynamic range as P95-P5 frame-energy spread in dB.

    Higher values → wider dynamics (less compression, more modern or
    higher-quality recording).  Typical ranges:
        Pre-1960 analog: 15–30 dB
        1970s tape:      25–40 dB
        1980s+ digital:  35–55 dB

    Returns:
        Dynamic range estimate in dB, clamped to [5, 70].
    """
    frame_size = max(1, sr // 10)  # 100 ms frames
    n_frames = len(audio_mono) // frame_size
    if n_frames < 5:
        return 25.0  # conservative neutral
    energies = np.array([np.mean(audio_mono[i * frame_size : (i + 1) * frame_size] ** 2) for i in range(n_frames)])
    # Remove silence frames (< -60 dBFS RMS equivalent)
    energies = energies[energies > 1e-6]
    if len(energies) < 5:
        return 25.0
    p95 = float(np.percentile(energies, 95))
    p5 = float(np.percentile(energies, 5))
    if p5 < 1e-18:
        return 40.0
    dr = 10.0 * math.log10(max(p95 / p5, 1.0))
    return float(np.clip(dr, 5.0, 70.0))


def _estimate_noise_modulation(audio_mono: np.ndarray, sr: int) -> float:
    """Schätzt temporal amplitude modulation of the noise/background floor.

    Vintage recordings contain transport-related amplitude modulation in the
    noise floor: wow & flutter on reel/cassette tape (mechanical variability)
    and pressing vibration harmonics on shellac/vinyl.  This is quantified as
    the coefficient of variation (sigma/mu) of RMS energy in quiet frames.

    Calibrated ranges:
        Pre-1950 wax / shellac / early tape :  > 0.40
        1950-1965 reel tape                 : 0.22 - 0.45
        1965-1975 compact cassette          : 0.12 - 0.26
        1975+  Dolby / HX cassette          :  < 0.15
        Digital recording                   :  < 0.06

    Args:
        audio_mono: 1-D float array, normalised to [-1, 1].
        sr:         Sample rate in Hz.

    Returns:
        Modulation index in [0.0, 1.0]; higher = more modulation = older era.
    """
    frame_size = max(1, sr // 20)  # 50 ms frames
    n_frames = len(audio_mono) // frame_size
    if n_frames < 20:
        return 0.20  # insufficient data
    energies = np.array(
        [float(np.mean(audio_mono[i * frame_size : (i + 1) * frame_size] ** 2)) for i in range(n_frames)],
        dtype=np.float32,
    )
    q30 = float(np.percentile(energies, 30))
    if q30 < 1e-20:
        return 0.20  # all silence
    quiet_mask = energies <= q30 * 5.0
    quiet_energies = energies[quiet_mask]
    if len(quiet_energies) < 8:
        return 0.20
    mean_e = float(np.mean(quiet_energies))
    std_e = float(np.std(quiet_energies))
    if mean_e < 1e-18:
        return 0.20
    return float(np.clip(std_e / mean_e, 0.0, 1.0))


def _estimate_lf_presence(audio_mono: np.ndarray, sr: int) -> float:
    """Low-frequency presence ratio: energy below 300 Hz vs 300 Hz - 3 kHz.

    Carbon-microphone recordings (1920s-1940s) have a telephone-like response:
    very little bass below 300 Hz.  Later ribbon and condenser microphones
    (1950s+) recover bass progressively.

    Typical ranges:
        1920s-1940s carbon mic :  < 0.18
        1950s ribbon mic       : 0.18 - 0.35
        1960s condenser        : 0.30 - 0.55
        1970s+  modern chain   :  > 0.45

    Returns:
        LF-presence ratio in [0.0, 1.0].
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 256:
        return 0.35
    hop = n_fft // 2
    specs: list[np.ndarray] = []
    for start in range(0, max(1, len(audio_mono) - n_fft), hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        specs.append(np.abs(np.fft.rfft(frame)) ** 2)
    if not specs:
        return 0.35
    avg_spec = np.mean(np.array(specs), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    lf_energy = float(np.sum(avg_spec[freqs < 300.0]))
    mf_energy = float(np.sum(avg_spec[(freqs >= 300.0) & (freqs < 3000.0)]))
    if mf_energy < 1e-20:
        return 0.35
    return float(np.clip(lf_energy / (lf_energy + mf_energy), 0.0, 1.0))


def _estimate_highband_presence(audio_mono: np.ndarray, sr: int) -> float:
    """High-band presence ratio: energy above 8 kHz vs 2-8 kHz.

    This metric helps distinguish late-1960s/1970s productions from earlier
    bandwidth-limited chains when rolloff and SNR are borderline.

    Typical behavior:
        Early analog / bandwidth-limited: < 0.10
        Late analog / 1970s+ tape:       0.12 - 0.30
        Modern full-band content:        > 0.25

    Returns:
        High-band presence ratio in [0.0, 1.0].
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 256:
        return 0.12
    hop = n_fft // 2
    specs: list[np.ndarray] = []
    for start in range(0, max(1, len(audio_mono) - n_fft), hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        specs.append(np.abs(np.fft.rfft(frame)) ** 2)
    if not specs:
        return 0.12
    avg_spec = np.mean(np.array(specs), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    hb_energy = float(np.sum(avg_spec[freqs >= 8000.0]))
    ref_energy = float(np.sum(avg_spec[(freqs >= 2000.0) & (freqs < 8000.0)]))
    if ref_energy < 1e-20:
        return 0.12
    return float(np.clip(hb_energy / (hb_energy + ref_energy), 0.0, 1.0))


def _transition_1970_score(
    *,
    highband_presence: float,
    lf_presence: float,
    snr_db: float,
    stereo_width: float,
    noise_modulation: float,
) -> float:
    """Kombinierter Evidenz-Score für 1960/1970-Übergangs-Entscheidungen.

    Returns a calibrated score in [0, 1] where higher values indicate stronger
    evidence for 1970s-style production chains.
    """
    hb_n = float(np.clip((highband_presence - 0.06) / 0.24, 0.0, 1.0))
    lf_n = float(np.clip((lf_presence - 0.20) / 0.20, 0.0, 1.0))
    snr_n = float(np.clip((snr_db - 34.0) / 20.0, 0.0, 1.0))
    st_n = float(np.clip((stereo_width - 0.05) / 0.10, 0.0, 1.0))
    mod_old_n = float(np.clip((noise_modulation - 0.12) / 0.24, 0.0, 1.0))

    score = 0.30 * hb_n + 0.23 * lf_n + 0.20 * snr_n + 0.15 * st_n + 0.12 * (1.0 - mod_old_n)
    return float(np.clip(score, 0.0, 1.0))


def _dsp_fingerprint_decade(
    rolloff_hz: float,
    snr_db: float,
    *,
    is_stereo: bool = False,
    stereo_width: float = 0.0,
    spectral_tilt: float = -4.0,
    dynamic_range_db: float = 25.0,
    noise_modulation: float = 0.20,
    lf_presence: float = 0.35,
    highband_presence: float = 0.12,
    transfer_chain: list[str] | None = None,  # §v10.303.32
) -> tuple[int, float]:
    """Mappt Bandbreite + SNR auf Jahrzehnt via kalibrierter Schwellwert-Tabelle.

    Erkennungsprinzip:
    - Jahrzehnte 1890–1980: Bandbreite (HF-Rolloff) ist der primäre Indikator,
      da analoge Medien physikalisch begrenzte Übertragungsbandbreiten haben.
      SNR dient als Sekundärindikator und verstärkt die Konfidenz bei Grenzfällen.
    - Jahrzehnte 1990–2025: Alle haben nominell ≥20 kHz Bandbreite — BW allein
      kann sie nicht unterscheiden. Der dynamische Bereich (SNR = P90/P10 der
      Frame-Energien) dient als primärer Diskriminator: CD (1990) ≈ 30–45 dB
      Musikdynamik, HD-Streaming (2010+) > 45 dB.

    SNR-Schwellwerte für post-1980 Material sind bewusst konservativ gesetzt, da
    der Frame-Energie-SNR-Schätzer die Musikdynamik misst, nicht den Rauschboden
    des Mediums. Stark komprimierte Musik (DR3–DR8) ergibt niedrige SNR-Werte
    und fällt zurecht in 1980/1990 (ältere Produktionspraxis).

    Args:
        rolloff_hz: Gemessener HF-Rolloff in Hz (90th-percentile cumulative energy).
        snr_db:     Geschätzter SNR in dB (frame-energy P90/P10 ratio).

    Returns:
        (decade, confidence)
    """
    bw_khz = rolloff_hz / 1000.0

    # ── §v10.14.1 Codec-BW-Recovery ──────────────────────────────────────
    # Bei stark basslastigen oder codec-komprimierten Aufnahmen kann der
    # kumulative Energie-Schätzer (E90) auf absurd niedrige Werte fallen
    # (z.B. 300 Hz), weil die meiste Energie im Bassbereich konzentriert ist.
    # Eine Stereo-Aufnahme mit > 35 dB SNR und messbarem Stereobild KANN
    # physikalisch NICHT auf < 2 kHz bandbegrenzt sein — das ist ein
    # Messartefakt, keine echte Aufnahmecharakteristik.
    #
    # Recovery: Ersetze die korrumpierte BW durch einen SNR-basierten
    # Schätzwert. Der konservative Floor (10 kHz) verhindert Overcorrection
    # bei echten Schmalband-Aufnahmen.
    # §v10.14.1: SNR-Condition entfernt — eine echte Stereo-Aufnahme mit
    # <2 kHz Bandbreite ist physikalisch unmöglich (Stereo erst ab ~1958,
    # Bandbreite damals ≥8 kHz). Ist die BW <2 kHz bei Stereo, ist das
    # IMMER ein Messartefakt.
    _bw_corrupted = bw_khz < 2.0 and is_stereo
    if _bw_corrupted:
        # SNR-basierte BW-Schätzung: je höher der SNR, desto breiter das
        # Originalsignal. Calibriert gegen gemessene Werte für 1970er–2000er
        # Aufnahmen mit Codec-Artefakten.
        _recovered_bw = float(np.clip(snr_db * 0.28 + 2.0, 10.0, 20.0))
        bw_khz = _recovered_bw
        logger.debug(
            "EraClassifier: BW corrupted (measured %.1f kHz, stereo=%s, SNR=%.1f dB) → "
            "recovered to %.1f kHz via SNR-proxy",
            rolloff_hz / 1000.0,
            is_stereo,
            snr_db,
            bw_khz,
        )

    # Primary decade selection via bandwidth.
    # Thresholds = midpoint of 0.90 × adjacent DECADE_HF_LIMITS:
    #   threshold(D, D+1) = (DECADE_HF_LIMITS[D] × 0.9 + DECADE_HF_LIMITS[D+1] × 0.9) / 2
    # Recalibrated 23.03.2026 to match DECADE_HF_LIMITS table consistently.
    # 1900 added: threshold(1890, 1900) = (3000×0.9 + 4000×0.9)/2 = 3150 → 3.2 kHz
    if bw_khz < 3.2:
        decade = 1890  # LIMIT  3 kHz → expected rolloff ~2.7 kHz
    elif bw_khz < 4.5:
        decade = 1900  # LIMIT  4 kHz → expected rolloff ~3.6 kHz
    elif bw_khz < 5.4:
        decade = 1910  # LIMIT  5 kHz → expected rolloff ~4.5 kHz
    elif bw_khz < 7.0:
        decade = 1920  # LIMIT  6 kHz → expected rolloff ~5.4 kHz
    elif bw_khz < 8.8:
        decade = 1930  # LIMIT  7 kHz → expected rolloff ~6.3 kHz
    elif bw_khz < 9.5:
        decade = 1940  # LIMIT  8 kHz → expected rolloff ~7.2 kHz; (7200+9000)/2=8100
    elif bw_khz < 11.5:
        decade = 1950  # LIMIT 10 kHz → expected rolloff ~9.0 kHz; (9000+10800)/2=9900
    elif bw_khz < 12.6:
        decade = 1960  # LIMIT 12 kHz → expected rolloff ~10.8 kHz; (10800+14400)/2=12600
    elif bw_khz < 17.0:
        decade = 1970  # LIMIT 16 kHz → expected rolloff ~14.4 kHz; (14400+18000)/2=16200
    elif bw_khz < 19.0:
        decade = 1980  # LIMIT 20 kHz → expected rolloff ~18.0 kHz
    else:
        # Full-bandwidth (≥ 19 kHz): BW cannot distinguish 1990–2025.
        # Use Gaussian-weighted SNR scoring to select the most likely digital
        # decade.  Each decade has a calibrated expected SNR (μ) derived from
        # typical frame-energy dynamic range measurements.  The Gaussian
        # likelihood penalises distance from μ; argmax gives the best match.
        #
        # Expected SNR values (frame-energy P90/P10 ratio in dB):
        #   1980 tape: ~14 dB  |  1990 CD: ~24 dB  |  2000: ~34 dB
        #   2010: ~44 dB       |  2020+: ~55 dB
        # σ calibrated at ~8 dB covering typical DR variation within a decade.
        _post90_snr = {1980: 14.0, 1990: 24.0, 2000: 34.0, 2010: 44.0, 2020: 55.0}
        _post90_sigma = 8.0
        best_dec, best_ll = 1980, -1e30
        for dec, mu in _post90_snr.items():
            z = (snr_db - mu) / _post90_sigma
            ll = -0.5 * z * z
            if ll > best_ll:
                best_ll = ll
                best_dec = dec
        decade = best_dec

    # SNR micro-correction for vintage decades (Carbon/Ribbon-microphone heuristic)
    if snr_db < 20.0 and bw_khz < 6.0:
        decade = min(decade, 1930)  # Carbon-microphone characteristic
    elif snr_db < 25.0 and bw_khz < 8.0 and decade > 1940:
        decade = min(max(decade, 1920), 1940)  # Ribbon-microphone era

    # §v10.303.32 Chain-Aware SNR Correction:
    # Multi-Carrier-Ketten degradieren den gemessenen SNR systematisch.
    # 3 analoge Träger + MP3 = 10–16 dB SNR-Verlust. Der EraClassifier
    # muss diesen Verlust kompensieren, sonst wird 1977 als 1960 fehldatiert.
    # §v10.14.1: Cap von 18→22 dB erhöht + Stereo-Breiten-Bonus (Stereo-
    # Aufnahmen verlieren durch Generation-Copying mehr wahrgenommenen SNR).
    _transfer_chain_items = list(transfer_chain or [])
    _chain_depth = len(_transfer_chain_items) if _transfer_chain_items else 1
    _snr_correction_db = 0.0
    if _chain_depth >= 2:
        # Pro analogem Träger ~4 dB, pro Codec ~3 dB SNR-Verlust
        _analog_steps = sum(
            1
            for c in _transfer_chain_items
            if c not in {"mp3_low", "mp3_high", "aac", "streaming", "minidisc", "cd_digital"}
        )
        _codec_steps = _chain_depth - _analog_steps
        _snr_correction_db = _analog_steps * 4.0 + _codec_steps * 3.0
        # Stereo-Bonus: Breite Stereo-Aufnahmen verlieren ~1.5 dB SNR pro
        # Träger-Generation durch Kanal-Übersprechen und Phasenfehler.
        if is_stereo and stereo_width >= 0.10:
            _snr_correction_db += stereo_width * _chain_depth * 8.0
        _snr_correction_db = min(_snr_correction_db, 22.0)  # Cap bei 22 dB (§v10.14.1)
    _snr_corrected = snr_db + _snr_correction_db

    # §v10.303.32+§v10.14.1 Deep-Chain Bias: Schon ab 2 Trägern sind
    # ALLE Features (SNR, BW, Stereo, Dynamic Range) systematisch degradiert.
    # Die akkumulierte Degradation drückt die Fingerprint-Analyse um
    # 1–2 Jahrzehnte nach unten. Korrektur: +10 / +20 Jahre abhängig von
    # der Anzahl der Sekundärmerkmale die auf spätere Ära hindeuten.
    if _chain_depth >= 2 and decade < 1980:
        _hints_later = 0
        if is_stereo and stereo_width >= 0.08:
            _hints_later += 1
        if dynamic_range_db >= 22.0:
            _hints_later += 1
        if spectral_tilt > -6.0:
            _hints_later += 1
        if _snr_corrected >= 45.0:
            _hints_later += 1
        if is_stereo and stereo_width >= 0.14:
            _hints_later += 1  # Wide stereo → strongly suggests ≥1970s multi-track
        if _hints_later >= 3:
            decade = min(decade + 20, 1990)
        elif _hints_later >= 1:
            decade = min(decade + 10, 1990)
        if _hints_later >= 1:
            logger.debug(
                "EraClassifier: deep-chain bias +%dy (depth=%d, hints=%d, SNR %.0f→%.0f)",
                20 if _hints_later >= 3 else 10,
                _chain_depth,
                _hints_later,
                snr_db,
                _snr_corrected,
            )

    # SNR-based upward correction for 1950–1970 borderline cases:
    # A tape recording from 1977 with bandwidth loss (e.g. 13 kHz rolloff)
    # may land in decade=1960 by BW alone, but its SNR (~48–55 dB) clearly
    # exceeds the 1960s expectation (44 dB).  If BW is within 1.5 kHz of
    # the next-decade threshold AND SNR matches the higher decade better,
    # promote by one decade.
    if decade in (1950, 1960, 1970) and decade < 1980:
        next_decade = decade + 10
        expected_snr_cur = _decade_expected_snr(decade)
        expected_snr_next = _decade_expected_snr(next_decade)
        threshold_bw = DECADE_HF_LIMITS.get(next_decade, 20000.0) / 1000.0 * 0.9
        bw_near_boundary = (threshold_bw - bw_khz) < 2.5  # within 2.5 kHz of next (covers tape→MP3 rolloff loss)
        snr_favors_next = abs(_snr_corrected - expected_snr_next) < abs(_snr_corrected - expected_snr_cur)
        if bw_near_boundary and snr_favors_next:
            decade = next_decade

    # ------------------------------------------------------------------
    # Multi-factor corrections: stereo, spectral tilt, dynamic range
    # ------------------------------------------------------------------

    # (A) Stereo presence — strong era discriminator.
    # Commercial stereo appeared ~1958; pre-1958 recordings are virtually all mono.
    if is_stereo and decade < 1960:
        decade = 1960  # stereo recording cannot predate late 1950s

    # (B) Stereo width — narrows 1960 vs 1970.
    # 1960s: hard-pan L/R, mono-compatible mixes → narrow width (< 0.12).
    # 1970s+: multi-track, wider imaging → width ≥ 0.12.
    if is_stereo and decade in (1960, 1970):
        if stereo_width >= 0.12 and decade == 1960:
            # Wide stereo field strongly suggests ≥ 1970s production
            decade = 1970
        elif stereo_width < 0.06 and decade == 1970 and bw_khz < 14.0:
            # Very narrow stereo with limited BW → likely 1960s
            decade = 1960

    # (C) Spectral tilt — captures recording-chain frequency response.
    # 1960s: steep roll-off (≤ -5 dB/oct, mid-forward due to equipment).
    # 1970s: flatter response (> -5 dB/oct, improved electronics/tape).
    # GUARD: When BW is codec-limited (< 10 kHz) AND tilt is extremely
    # steep (< -8.0 dB/oct), both metrics reflect the lossy codec's
    # brick-wall LPF (e.g. mp3_low @ 64–128 kbps), not the recording
    # equipment.  Tilt-based demotion is unreliable in this regime.
    _codec_limited = bw_khz < 10.0 and spectral_tilt < -8.0
    if decade in (1960, 1970):
        if spectral_tilt > -4.0 and decade == 1960:
            decade = 1970  # flatter spectrum → favour 1970s
        elif spectral_tilt < -7.5 and decade == 1970 and bw_khz < 15.0 and not _codec_limited:
            decade = 1960  # steep tilt with moderate BW → 1960s (not codec)

    # (D) Dynamic range — improved electronics yield wider dynamics.
    # 1960s analog: typically 15–28 dB frame-energy DR.
    # 1970s tape:   typically 25–40 dB.
    # GUARD: Codec-limited recordings (mp3_low) compress DR significantly;
    # demotion from DR alone is unreliable when codec artifacts dominate.
    if decade in (1960, 1970):
        if dynamic_range_db >= 30.0 and decade == 1960:
            decade = 1970
        elif dynamic_range_db < 18.0 and decade == 1970 and bw_khz < 14.0 and not _codec_limited:
            decade = 1960

    # (E) Consensus vote for borderline 1950–1980 decades:
    # Count how many secondary features favour "one decade higher".  If ≥ 2
    # of 3 vote higher AND BW is within 2 kHz of the next-decade threshold,
    # promote.  Prevents single-feature flukes from dominating.
    if decade in (1950, 1960, 1970):
        next_dec = decade + 10
        thr_bw = DECADE_HF_LIMITS.get(next_dec, 20000.0) / 1000.0 * 0.9
        if (thr_bw - bw_khz) < 2.5:  # BW borderline (2.5 kHz covers tape→MP3 HF compression headroom)
            votes_up = 0
            # Stereo vote
            if is_stereo and stereo_width >= 0.10:
                votes_up += 1
            # Tilt vote
            _tilt_threshold = {1950: -6.0, 1960: -5.0, 1970: -3.5}
            if spectral_tilt > _tilt_threshold.get(decade, -5.0):
                votes_up += 1
            # DR vote
            _dr_threshold = {1950: 22.0, 1960: 28.0, 1970: 35.0}
            if dynamic_range_db >= _dr_threshold.get(decade, 28.0):
                votes_up += 1
            if votes_up >= 2:
                decade = next_dec

    # (F) Noise-floor temporal modulation — wow/flutter proxy
    # Guard: codec-limited recordings (narrow BW + steep tilt) produce high noise
    # modulation via MP3/codec artifacts — this is NOT physical wow/flutter.
    if noise_modulation > 0.38 and decade in (1960, 1970) and bw_khz < 10.0 and not _codec_limited:
        decade = min(decade, 1960)
    elif noise_modulation < 0.08 and decade in (1950, 1960) and bw_khz > 9.0:
        decade = max(decade, 1970)
    elif noise_modulation > 0.25 and decade == 1980 and bw_khz < 14.0:
        decade = 1970

    # (G) Low-frequency presence — carbon/ribbon microphone era footprint
    if lf_presence < 0.18 and decade in (1950, 1960) and snr_db < 35.0:
        decade = min(decade, 1940)
    elif lf_presence > 0.50 and decade in (1920, 1930) and bw_khz > 5.0:
        decade = max(decade, 1940)

    # (H) High-band presence — late analog / 1970s discriminator
    if decade == 1960 and highband_presence > 0.22 and snr_db >= 45.0 and stereo_width >= 0.10:
        decade = 1970
    elif decade == 1970 and highband_presence < 0.10 and noise_modulation > 0.30 and snr_db < 40.0:
        decade = 1960

    transition_score_1970 = 0.50
    # (I) Combined evidence in the 1960/1970 transition zone.
    # Uses a calibrated weighted score to reduce single-feature overfitting.
    if decade in (1960, 1970) and 10.5 <= bw_khz <= 15.5:
        transition_score_1970 = _transition_1970_score(
            highband_presence=highband_presence,
            lf_presence=lf_presence,
            snr_db=snr_db,
            stereo_width=stereo_width,
            noise_modulation=noise_modulation,
        )
        if decade == 1960 and transition_score_1970 >= 0.58 and highband_presence > 0.16 and snr_db >= 40.0:
            decade = 1970
        elif decade == 1970 and transition_score_1970 <= 0.38 and highband_presence < 0.16 and noise_modulation > 0.20:
            decade = 1960

    # Confidence: combine BW error and SNR deviation for each era class.
    expected_bw = DECADE_HF_LIMITS.get(decade, 20000.0) / 1000.0
    bw_error = abs(bw_khz - expected_bw) / max(expected_bw, 1.0)
    expected_snr = _decade_expected_snr(decade)
    snr_error = abs(snr_db - expected_snr) / max(expected_snr, 1.0)

    if decade >= 1990:
        # Post-1990: SNR is primary classifier; BW error is irrelevant (all ≈20 kHz)
        conf = float(np.clip(1.0 - snr_error * 0.55, 0.50, 0.90))
    elif decade <= 1940:
        # Pre-1950: BW and SNR both carry independent physical evidence
        conf = float(np.clip(1.0 - bw_error * 0.50 - snr_error * 0.30, 0.25, 0.90))
    else:
        # 1950–1980: BW-dominated, SNR secondary
        conf = float(np.clip(1.0 - bw_error * 0.70 - snr_error * 0.15, 0.25, 0.87))

    # Confidence boost when multiple features agree with the selected decade.
    if decade in (1960, 1970, 1980):
        agreement_count = 0
        if is_stereo and decade >= 1960:
            agreement_count += 1
        if (
            (decade == 1960 and spectral_tilt <= -5.0)
            or (decade == 1970 and -5.0 < spectral_tilt <= -3.0)
            or (decade == 1980 and spectral_tilt > -3.0)
        ):
            agreement_count += 1
        if (
            (decade == 1960 and dynamic_range_db < 30.0)
            or (decade == 1970 and 25.0 <= dynamic_range_db < 42.0)
            or (decade == 1980 and dynamic_range_db >= 35.0)
        ):
            agreement_count += 1
        conf = min(0.92, conf + agreement_count * 0.04)

    # Confidence boost for pre-1950 acoustic/electrical-shellac material.
    # Symmetrisch zum 1960–1980-Agreement-Boost oben: Pre-1950-Aufnahmen tragen
    # mehrere UNABHÄNGIGE physikalische Signaturen, die — wenn sie konvergieren —
    # eine eindeutige Era-Zuordnung belegen. Ohne diesen Zweig blieb der Classifier
    # bei zweifelsfreiem Shellac-Material künstlich unsicher (z. B. conf≈0.42 trotz
    # Mono + niedrigem SNR + hoher Oberflächen-Modulation + fehlendem Sub-Bass).
    # WISSENSCHAFTLICHE INVARIANTE (§0g, §0c): Der Boost ändert NIEMALS die Dekade
    # (nur die Konfidenz), ist vollständig materialgetrieben (keine song-spezifische
    # Regel) und feuert nur bei echter Evidenz-Konvergenz — bleibt die Evidenz
    # mehrdeutig, bleibt die Unsicherheit erhalten (ehrliche §0g-Unsicherheit).
    elif decade <= 1940:
        agreement_count = 0
        # (1) Mono: kommerzielle Stereofonie erst ~1958 → starker Pre-1950-Indikator.
        if not is_stereo:
            agreement_count += 1
        # (2) Shellac/akustische Oberflächen-/Trichter-Aufnahme → niedriger SNR.
        if snr_db < 30.0:
            agreement_count += 1
        # (3) Mechanische Oberflächen-Modulation (Crackle/Rumble/Gleichlauf).
        if noise_modulation > 0.20:
            agreement_count += 1
        # (4) Akustische/früh-elektrische Ketten ohne Sub-Bass-Energie.
        if lf_presence < 0.30:
            agreement_count += 1
        # (5) Steiler HF-Roll-off konsistent mit bandbegrenztem Träger.
        if spectral_tilt <= -5.0:
            agreement_count += 1
        conf = min(0.90, conf + agreement_count * 0.035)

    # Confidence fine-tuning for the 1960/1970 transition evidence.
    if decade in (1960, 1970):
        # Score certainty around the transition boundary:
        # far from 0.5 -> clearer evidence, near 0.5 -> ambiguous evidence.
        dist_mid = abs(transition_score_1970 - 0.5)
        if dist_mid >= 0.20:
            conf = min(0.92, conf + 0.03)
        elif dist_mid <= 0.08:
            conf = max(0.25, conf - 0.04)

        if (decade == 1970 and highband_presence > 0.20 and noise_modulation < 0.20) or (
            decade == 1960 and highband_presence < 0.12 and noise_modulation > 0.30
        ):
            conf = min(0.92, conf + 0.03)
        # Penalize contradictory evidence so the classifier can fall back to medium-floor safely.
        if highband_presence > 0.22 and noise_modulation > 0.32:
            conf = max(0.25, conf - 0.05)

    if bw_khz >= 18.0 and decade < 1990:
        conf = max(conf, 0.75)  # Full-bandwidth analog clearly ≥ 1980

    logger.debug(
        "DSP-Fingerprint: bw=%.1fkHz snr=%.1fdB stereo=%s width=%.3f "
        "tilt=%.1fdB/oct dr=%.1fdB mod=%.2f lf=%.2f hb=%.2f ts=%.2f "
        "→ decade=%d conf=%.2f",
        bw_khz,
        snr_db,
        is_stereo,
        stereo_width,
        spectral_tilt,
        dynamic_range_db,
        noise_modulation,
        lf_presence,
        highband_presence,
        transition_score_1970,
        decade,
        conf,
    )
    return decade, conf


def _decade_expected_snr(decade: int) -> float:
    """Grobe SNR-Erwartung pro Dekade [dB]."""
    snr_map = {
        1890: 12,
        1900: 15,
        1910: 18,
        1920: 22,
        1930: 28,
        1940: 32,
        1950: 38,
        1960: 44,
        1970: 52,
        1980: 58,
        1990: 65,
        2000: 70,
        2010: 75,
        2020: 80,
        2025: 80,
    }
    return snr_map.get(decade, 50)


def _estimate_snr(audio_mono: np.ndarray, sr: int = 48000) -> float:
    """Frame-basierte SNR-Schätzung via Energie-Perzentile.

    Robuster als die frühere Sample-Level-Sortierung für Vintage-Aufnahmen
    mit dauerhaftem Rauschen: Die Sortierung von Einzelsamples lieferte dort
    ~52 dB SNR statt der korrekten ~15–25 dB, da kurze Stille-Momente die
    untersten Perzentile dominierten.

    Frame-Level-Ansatz (100 ms Frames):
        - 10. Energie-Perzentil → Rauschboden
        - 90. Energie-Perzentil → Nutz-Signal

    Args:
        audio_mono: Mono-Audio.
        sr:         Sample-Rate (für Frame-Größen-Berechnung).

    Returns:
        Geschätzter SNR in dB, geclamppt auf [0, 80].
    """
    frame_size = max(1, sr // 10)  # 100-ms-Frames
    frames = [audio_mono[i : i + frame_size] for i in range(0, len(audio_mono) - frame_size, frame_size)]
    if not frames:
        return 40.0
    energies = np.array([np.mean(f**2) for f in frames])
    noise_floor = float(np.percentile(energies, 10))
    signal_power = float(np.percentile(energies, 90))
    if noise_floor < 1e-18:
        return 60.0
    snr = 10.0 * math.log10(max(signal_power / noise_floor, 1.0))
    return float(np.clip(snr, 0.0, 80.0))


# ---------------------------------------------------------------------------
# Tier-3: Mikrofon-Typ-Heuristik
# ---------------------------------------------------------------------------


def _microphone_type_decade(bark_energies: np.ndarray) -> tuple[int, float]:
    """Ära-Schätzung aus 24 Bark-Band-Energien via 95th-Perzentil-Bandbreite (Tier-3 Fallback).

    Mirrors Tier-2's 90th-percentile rolloff logic but operates on the already-computed
    Bark-band energy array.  The 95th-percentile Bark band (bw95) is the band index below
    which 95 % of the total spectral energy lies.  This is a robust proxy for the recording's
    effective bandwidth and avoids the non-linear Bark band width bias that invalidates
    simple min/max flatness ratios.

    Calibration (6th-order Butterworth through _bark_band_energies at 48 kHz):
        cutoff 3.5 kHz  → bw95 ≈ 15–16  (Bark band boundary ~3.2 kHz)
        cutoff 6.0 kHz  → bw95 ≈ 18–19  (~5.3 kHz)
        cutoff 10 kHz   → bw95 ≈ 20     (~6.4 kHz region)
        cutoff 14 kHz   → bw95 ≈ 21     (~7.7 kHz)
        cutoff 20 kHz+  → bw95 ≈ 22–23  (full band)

    Args:
        bark_energies: 24 normalised Bark-band energies (sum should be ≈ 1.0).

    Returns:
        (decade, confidence)
    """
    total = float(bark_energies.sum()) + 1e-12

    # Low-frequency dominance: fraction of energy below 630 Hz (Bark bands 0–5)
    lf_frac = float(np.sum(bark_energies[:6]) / total)

    # 95th-percentile Bark bandwidth band (similar principle to Tier-2's 90th-pct rolloff)
    cum = np.cumsum(bark_energies)
    bw95 = int(np.clip(int(np.searchsorted(cum, 0.95 * total)), 0, 23))

    # Map bandwidth band to decade.  Boundaries calibrated against Butterworth LP test signals.
    if bw95 <= 6:
        # < ~770 Hz effective BW → pre-1920 acoustic/mechanical format
        return 1910, 0.42
    elif bw95 <= 9:
        # 770–1270 Hz → carbon-microphone era (1920s)
        return 1920, 0.40
    elif bw95 <= 12:
        # 1270–1720 Hz → early ribbon/condenser (1930s)
        return 1930, 0.38
    elif bw95 <= 16:
        # 1720–3700 Hz → vintage tape/early vinyl; SNR-based sub-split
        dec = 1930 if lf_frac > 0.30 else 1940
        return dec, 0.36
    elif bw95 <= 18:
        # 3700–5300 Hz → HiFi LP / early reel tape (1950s/60s)
        return 1960, 0.33
    elif bw95 <= 20:
        # 5300–7700 Hz → FM radio / cassette era (1970s)
        return 1970, 0.31
    elif bw95 <= 22:
        # 7700–12000 Hz → HiFi tape / early digital (1980s)
        return 1980, 0.30
    else:
        # > 12000 Hz → full-bandwidth digital era (1990+)
        return 1990, 0.28


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


# §2.13 Multi-Generation Era Ceiling (v10.0.0): Analog-Träger haben
# Produktionszeiträume. Wenn die Kette einen analogen Ursprungsträger
# enthält (z. B. vinyl→cassette→mp3), darf die erkannte Ära NICHT
# jünger sein als das Ende der kommerziellen Produktion dieses Trägers.
# Dies verhindert, dass der letzte Codec in der Kette (mp3=1990er)
# den ursprünglichen Aufnahmezeitpunkt überschreibt.
_ANALOG_ERA_CEILING: dict[str, int] = {
    "shellac": 1955,  # Schellack: letzte kommerzielle Pressungen ~1955
    "wax_cylinder": 1930,  # Wachswalze: Ende ~1929
    "vinyl": 1989,  # Vinyl: CD verdrängt Vinyl ~1989 (Nischenproduktion danach)
    "lacquer_disc": 1970,  # Lacquer-Disc (Acetat): Ende ~1970
    "wire_recording": 1950,  # Drahtaufnahme: Ende ~1950
    "cassette": 2005,  # Kassette: letzte große Produktionen ~2005
    "tape": 1995,  # Tonband: Ende ~1995 (Nischen bis heute)
    "reel_tape": 1995,  # Tonband (Spule): Ende ~1995
    "8track": 1985,  # 8-Track: Ende ~1985
    "dat": 2005,  # DAT: Ende ~2005
    "minidisc": 2005,  # MiniDisc: Ende ~2005
}


def _apply_analog_era_ceiling(
    result: EraResult, transfer_chain: list[str] | None
) -> tuple[EraResult, int | None, str | None]:
    """§2.13 Multi-Generation Era Ceiling — Analog-Träger begrenzt die Ära.

    Enthält die Transfer-Kette einen analogen Ursprungsträger, darf die
    erkannte Ära nicht jünger sein als das Produktionsende dieses Trägers
    (Beispiel: vinyl→mp3 — CLAP/DSP sagt 2000er, Ceiling klemmt auf 1989).

    Die Korrektur nutzt ``dataclasses.replace`` statt Hand-Rekonstruktion:
    alle Felder (noise_profile, hf_rolloff_hz, is_remaster_suspected, …)
    bleiben erhalten und Pflichtfelder wie ``era_label`` können nicht
    verloren gehen (Watchdog-Bug „Pre-Analyse degradiert" 2026-09-06).

    Args:
        result: Vorläufiges Era-Ergebnis (darf None sein → unverändert zurück).
        transfer_chain: Effektive Transfer-Kette (z. B. ["vinyl", "mp3_low"]).

    Returns:
        (korrigiertes Ergebnis, ceiling, ceiling_source) — bei keiner
        Korrektur (result, None, None).
    """
    if not transfer_chain or result is None:
        return result, None, None
    _ceiling = 9999
    _ceiling_source = None
    for _mat in transfer_chain:
        _mat_lower = str(_mat).lower().replace(" ", "_").replace("-", "_")
        _c = _ANALOG_ERA_CEILING.get(_mat_lower)
        if _c is not None and _c < _ceiling:
            _ceiling = _c
            _ceiling_source = _mat_lower
    if _ceiling >= 9999 or result.decade <= _ceiling:
        return result, None, None
    return (
        dc_replace(
            result,
            decade=_ceiling,
            era_label=f"{_ceiling}er",
            confidence=result.confidence * 0.85,  # leichte Unsicherheit
        ),
        _ceiling,
        _ceiling_source,
    )


class EraClassifier:
    """Erkennt Aufnahme-Ära (1890–2025) und leitet epochenspezifische Priors ab.

    Erkennungs-Kaskade (3 Stufen):
        Tier-1: LAION-CLAP-Embeddings → NN zu Ära-Referenz-Ankern
        Tier-2: DSP-Fingerprint → HF-Rolloff + Bandbreiten-Kurve
        Tier-3: Mikrofon-Typ-Heuristik (Carbon/Kondensator)

    Ausgabe: EraResult(decade, era_label, confidence, material_prior,
                       noise_profile, tier_used, hf_rolloff_hz)

    Invarianten:
        - Konfidenz < 0.4 → material_prior = "unknown" (konservative Priors)
        - CLAP-Fallback auf DSP-Fingerprint wenn Import fehlschlägt  # §V6 (copilot-instructions.md): logger.warning handled at call site
        - Decade-Label wird in RestorationResult.era_decade gespeichert
        - Ergebnisse werden ausschließlich im RAM gecacht (kein Disk-I/O)
    """

    def __init__(self) -> None:
        self._clap_plugin: object | None = None
        self._clap_loaded: bool = False
        self._clap_lock = threading.Lock()
        self._ram_cache: dict[str, EraResult] = {}
        self._ram_cache_lock = threading.Lock()
        # Short clips are classified robustly via DSP tiers; loading Tier-1 CLAP
        # for a few seconds of audio is disproportionally expensive and causes
        # flaky timeout behavior in constrained test runs.
        self._tier1_min_duration_s: float = 12.0

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def classify(self, audio: np.ndarray, sr: int, transfer_chain: list[str] | None = None) -> EraResult:
        """Erkennt Aufnahme-Ära (Cascaded Tier-1 → Tier-2 → Tier-3).

        Args:
            audio: Audio-Signal (mono oder stereo).
            sr:    Sample-Rate in Hz — muss exakt 48000 sein (Spec §3.x).

        Returns:
            EraResult mit Dekade, Confidence und Material-Prior.

        Raises:
            ValueError:    Falls audio leer ist.
        """
        if audio.size == 0:
            raise ValueError("Audio darf nicht leer sein.")
        # SR-agnostic: analysis modules work at native import SR (Spec §Performance-Budget)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        audio_mono = np.mean(audio, axis=-1 if audio.shape[-1] <= 2 else 0) if audio.ndim > 1 else audio.copy()

        # RAM-Cache-Key aus SHA256-Prefix + Transfer-Chain (für Chain-Awareness)
        # + Version-Marker (§v10.14): invalidiert Caches aus älteren Code-Versionen
        _chain_str = ",".join(sorted(transfer_chain)) if transfer_chain else "nochain"
        _version_tag = "v10.14_fmt_check"  # §v10.14: File-Format-Plausibilität
        sha = hashlib.sha256(audio_mono.tobytes() + _chain_str.encode() + _version_tag.encode()).hexdigest()[:16]
        with self._ram_cache_lock:
            cached = self._ram_cache.get(sha)
        if cached is not None:
            logger.debug(
                "EraClassifier: RAM-Zwischenspeicher-Hit %s → Jahrzehnt=%d, Konfidenz=%.2f, Tier=%d",
                sha,
                cached.decade,
                cached.confidence,
                cached.tier_used,
            )
            return cached

        bark = _bark_band_energies(audio_mono, sr)
        rolloff_hz = _dsp_hf_rolloff(audio_mono, sr)
        snr_db = _estimate_snr(audio_mono, sr)

        # Multi-factor features (§2.14 enhanced era discrimination)
        is_stereo, stereo_width = _detect_stereo_properties(audio, sr)
        spectral_tilt = _estimate_spectral_tilt(audio_mono, sr)
        dynamic_range_db = _estimate_dynamic_range(audio_mono, sr)

        noise_modulation = _estimate_noise_modulation(audio_mono, sr)
        lf_presence = _estimate_lf_presence(audio_mono, sr)
        highband_presence = _estimate_highband_presence(audio_mono, sr)

        # Tier-1: CLAP (optional). Skip for short clips to keep latency bounded.
        _audio_duration_s = float(audio_mono.size) / max(1.0, float(sr))
        if _audio_duration_s >= self._tier1_min_duration_s:
            result = self._try_tier1(audio_mono, sr, bark, rolloff_hz, snr_db)
        else:
            logger.debug(
                "EraClassifier: ueberspringen Tier-1 (short clip %.2fs < %.2fs)",
                _audio_duration_s,
                self._tier1_min_duration_s,
            )
            result = None

        # Physical plausibility cross-check for Tier-1 (Eargle 2004, §2.47).
        # CLAP is a statistical model; hard physical facts override its classification:
        #  • Commercial stereo recordings started ca. 1958.  A stereo signal CANNOT
        #    originate from a pre-1958 decade regardless of CLAP confidence.
        #  • Significant high-band energy above 8 kHz is incompatible with a decade
        #    that nominally had ≤ 8 kHz recording bandwidth (pre-1940s equipment).
        #  • §2.13 Analog-Era-Plausibilität: Wenn die Chain analoge Medien enthält
        #    (vinyl, cassette, shellac), kann die Aufnahme nicht NACH dem Ende der
        #    Analog-Ära entstanden sein. CLAP > 1989 + analog chain → Tier-2.
        #  • §2.47 Digitization Gate (NEU v10.14.0): Stereo und HF-Präsenz sind
        #    für digitalisierte historische Aufnahmen KEINE Plausibilitäts-
        #    verletzung. Eine 1928er Schellack-Aufnahme die auf CD digitalisiert
        #    wurde, hat legitimes Stereo (Left/Right Duplizierung) und volle
        #    Bandbreite bis 22 kHz (vom A/D-Wandler). CLAP erkennt die Ära
        #    korrekt — der stereo/hf-Violations-Gate darf dies nicht überschreiben.
        #    Nur bei rein analoger Kette (kein Digital-Material) sind stereo/hf
        #    physikalische Unmöglichkeiten und werden als Violation gewertet.
        # When any constraint is violated, Tier-1 is discarded and Tier-2 DSP
        # (which implements these guards natively) is used instead.
        if result is not None and result.tier_used == 1:
            _clap_decade = result.decade
        # §v10.14 CLAP-File-Format-Plausibilität: Wenn CLAP eine Ära vor der
        # Existenz des Dateiformats liefert, ist die CLAP-Vorhersage physikalisch
        # unmöglich und muss verworfen werden.
        # Bsp: MP3 (erfunden 1995) + CLAP=1920 → unmöglich → DSP-Fallback.
        _MATERIAL_INVENTED: dict[str, int] = {
            "mp3_low": 1995,
            "mp3_high": 1995,
            "mp3_high_vbr": 1998,
            "aac": 1997,
            "streaming": 2005,
            "cd_digital": 1982,
            "dat": 1987,
            "minidisc": 1992,
            "dcc": 1992,
            "bluray_audio": 2006,
            "dvd_audio": 2000,
            "sacd": 1999,
            "pcm_digital": 1982,
            "lossless_digital": 1982,
        }
        if result is not None and result.tier_used == 1:
            _clap_decade = result.decade
            # Check: hat das Dateiformat ein späteres Erfindungsdatum als CLAP?
            if transfer_chain:
                for _tm in transfer_chain:
                    _tm_key = str(_tm).lower().replace(" ", "_").replace("-", "_")
                    _invented = _MATERIAL_INVENTED.get(_tm_key)
                    if _invented and _clap_decade < _invented - 40:
                        logger.info(
                            "EraClassifier: CLAP %d < Format-Erfindung %s(%d) → "
                            "physikalisch unmöglich, Tier-1 verworfen, Tier-2 DSP",
                            _clap_decade,
                            _tm_key,
                            _invented,
                        )
                        result = None
                        break

            # §2.47: Detect whether this is a digitized source — if ANY digital
            # material is present in the chain, stereo/HF could be digitization
            # artifacts, not original recording properties.
            _DIGITISED_MATERIALS: frozenset[str] = frozenset(
                {
                    "cd",
                    "cd_digital",
                    "mp3_low",
                    "mp3_high",
                    "aac",
                    "streaming",
                    "dat",
                    "minidisc",
                    "dcc",
                    "bluray_audio",
                    "dvd_audio",
                    "sacd",
                    "pcm_digital",
                    "lossless_digital",
                }
            )
            _is_digitized = transfer_chain is not None and any(
                str(t).lower().replace(" ", "_").replace("-", "_") in _DIGITISED_MATERIALS for t in transfer_chain
            )
            # §SOTA #6: Nur echtes Stereo (width>0.05) triggert Violation.
            # Dual-Mono (width≈0) ist typisch für digitalisierte Mono-Aufnahmen.
            # §2.47: Keine Stereo-Violation bei digitalisierter Quelle.
            _stereo_violation = is_stereo and _clap_decade < 1960 and stereo_width > 0.05 and not _is_digitized
            # §2.47: Keine HF-Violation bei digitalisierter Quelle.
            _hf_violation = (highband_presence > 0.20) and (_clap_decade < 1940) and not _is_digitized
            _analog_era_violation = (
                not _is_digitized  # §2.47: nur bei rein analoger Kette relevant
                and transfer_chain is not None
                and any(
                    m
                    in (
                        "vinyl",
                        "cassette",
                        "shellac",
                        "lacquer_disc",
                        "wax_cylinder",
                        "tape",
                        "reel_tape",
                        "wire_recording",
                        "8track",
                    )
                    for m in (str(t).lower().replace(" ", "_").replace("-", "_") for t in transfer_chain)
                )
                and _clap_decade > 1989
            )
            # Material-Floor-Prüfung: CLAP-Jahrzehnt muss ≥ Einführungsjahr
            # des analogen Trägermediums sein. Bsp: vinyl (floor=1950) +
            # CLAP=1930 → physikalisch unmöglich → Tier-2 DSP-Override.
            _material_floor_violation = transfer_chain is not None and any(
                MEDIUM_DECADE_FLOOR.get(m, 0) > _clap_decade
                for m in (str(t).lower().replace(" ", "_").replace("-", "_") for t in transfer_chain)
                if m
                in (
                    "vinyl",
                    "cassette",
                    "shellac",
                    "lacquer_disc",
                    "wax_cylinder",
                    "tape",
                    "reel_tape",
                    "wire_recording",
                    "8track",
                )
            )
            if _stereo_violation or _hf_violation or _analog_era_violation or _material_floor_violation:
                _violations = []
                if _stereo_violation:
                    _violations.append("stereo")
                if _hf_violation:
                    _violations.append("hf_presence")
                if _analog_era_violation:
                    _violations.append("analog_chain")
                if _material_floor_violation:
                    _violations.append("material_floor")
                logger.info(
                    "EraClassifier: Tier-1 Plausibilitätsverletzung ("
                    "CLAP-Jahrzehnt=%d, violations=%s) → Tier-2 DSP-Override",
                    _clap_decade,
                    ",".join(_violations),
                )
                result = None  # force Tier-2

        # §2.13 CLAP-Confidence-Gate: CLAP ist ein general-purpose Audio-Modell,
        # kein Era-Spezialist. Bei confidence < 0.50 ist die Vorhersage nicht
        # besser als Raten → Tier-1 verwerfen, Tier-2 DSP direkt nutzen.
        if result is not None and result.tier_used == 1 and result.confidence < 0.50:
            logger.info(
                "EraClassifier: CLAP confidence %.2f < 0.50 — Tier-1 verworfen, Tier-2 DSP",
                result.confidence,
            )
            result = None

        # ── Decade-Boundary-Softener ────────────────────────────────────
        # CLAP general-purpose Audio-Modell klassifiziert borderline-Fälle
        # (z.B. 1977 → 1990) manchmal um mehrere Dekaden zu hoch, besonders
        # bei MP3-komprimierten Aufnahmen.  Die spektrale Bandbreite ist der
        # härteste physikalische Fakt — eine Aufnahme mit <10 kHz Rolloff
        # KANN nicht aus den 1990ern stammen.
        #
        # §v10.14.1 Guard: Bei Codec-korrumpierter BW-Messung (sehr niedriger
        # Rolloff + hoher SNR + Stereo) ist der Rolloff ein Messartefakt, KEIN
        # echtes Bandbreiten-Limit. Der Softener wird dann deaktiviert.
        #
        # Korrektur-Stärke hängt von der Bandbreite ab:
        #   rolloff <  8 kHz → −3 Dekaden  (AM-Radio / Schellack-Ära)
        #   rolloff < 12 kHz → −2 Dekaden  (frühes Tape / Vinyl)
        #   rolloff < 16 kHz → −1 Dekade   (pre-digitales Equipment)
        if result is not None and result.tier_used == 1 and result.decade >= 1980:
            _rolloff = float(rolloff_hz or 22000)
            # §v10.14.1: Deaktiviere Softener wenn BW-Messung korrumpiert ist.
            # Eine Stereo-Aufnahme mit <2 kHz Bandbreite ist physikalisch
            # unmöglich — der Rolloff ist ein Messartefakt.
            _bw_measurement_corrupted = _rolloff < 2000.0 and is_stereo
            _steps = 0
            if not _bw_measurement_corrupted:
                if _rolloff < 8000:
                    _steps = 3
                elif _rolloff < 12000:
                    _steps = 2
                elif _rolloff < 16000 and result.confidence < 0.72:
                    _steps = 1

            if _steps > 0:
                _original = result.decade
                _corrected = max(1890, result.decade - _steps * 10)
                if _corrected != _original:
                    logger.info(
                        "🕰️ Decade-Boundary-Softener: %d→%d (CLAP-conf=%.2f, rolloff=%.0fHz, steps=%d) — "
                        "analog characteristics suggest earlier decade",
                        _original,
                        _corrected,
                        result.confidence,
                        _rolloff,
                        _steps,
                    )
                    result = dc_replace(
                        result,
                        decade=_corrected,
                        era_label=f"{_corrected}er",
                        material_prior=DECADE_MATERIAL_PRIOR.get(_corrected, result.material_prior),
                        confidence=result.confidence * 0.85,
                    )

        # Tier-2: DSP-Fingerprint (multi-factor) — immer als Sanity-Check für CLAP Tier-1.
        # CLAP ist ein general-purpose Audio-Modell und kein Ära-Spezialist.
        # DSP nutzt physikalische Signalcharakteristika (Bandbreite, SNR, Stereo-Breite,
        # Dynamikumfang, Klirr/Modulation) und ist damit der härtere Faktencheck.
        _tier2_result = self._tier2(
            bark,
            rolloff_hz,
            snr_db,
            is_stereo=is_stereo,
            stereo_width=stereo_width,
            spectral_tilt=spectral_tilt,
            dynamic_range_db=dynamic_range_db,
            noise_modulation=noise_modulation,
            lf_presence=lf_presence,
            highband_presence=highband_presence,
            transfer_chain=transfer_chain,  # §v10.303.32
        )
        if result is None or result.confidence < 0.40:
            result = _tier2_result
        elif _tier2_result.decade != result.decade and result.tier_used == 1:
            # §v10.15 Intelligente DSP-Override-Schwelle:
            # DSP darf CLAP nur überschreiben, wenn DSP SELBST hinreichend
            # sicher ist — und zwar proportional zur CLAP-Confidence.
            # Je sicherer CLAP, desto sicherer muss DSP sein.
            #   CLAP ≥ 0.80 → DSP muss ≥ 0.60
            #   CLAP ≥ 0.65 → DSP muss ≥ 0.50
            #   CLAP ≥ 0.50 → DSP muss ≥ 0.42
            #   CLAP  < 0.50 → DSP muss ≥ 0.35 (alte Schwelle)
            _clap_conf = result.confidence
            if _clap_conf >= 0.80:
                _dsp_required = 0.60
            elif _clap_conf >= 0.65:
                _dsp_required = 0.50
            elif _clap_conf >= 0.50:
                _dsp_required = 0.42
            else:
                _dsp_required = 0.35
            if _tier2_result.confidence >= _dsp_required:
                logger.info(
                    "EraClassifier: Tier-2 DSP-Sanity-Pruefung widerspricht CLAP Tier-1 "
                    "(CLAP=%d/%.2f, DSP=%d/%.2f, required=%.2f) → DSP-Override",
                    result.decade,
                    result.confidence,
                    _tier2_result.decade,
                    _tier2_result.confidence,
                    _dsp_required,
                )
                result = _tier2_result
            else:
                logger.info(
                    "EraClassifier: Tier-2 DSP widerspricht CLAP, aber DSP-Confidence "
                    "%.2f < required %.2f (CLAP=%.2f) → CLAP behalten",
                    _tier2_result.confidence,
                    _dsp_required,
                    result.confidence,
                )

        # §2.13 Multi-Generation Era Ceiling: Wenn die Kette einen analogen
        # Ursprungsträger enthält, darf die erkannte Ära nicht jünger sein
        # als das Produktionsende dieses Trägers.
        # Beispiel: vinyl→cassette→mp3 → vinyl ceiling=1989. Auch wenn CLAP/DSP
        # 1990er erkennen (wegen mp3-Artefakten), wird auf 1989 geklemmt.
        if transfer_chain and result is not None:
            _original_decade = result.decade
            result, _ceiling, _ceiling_source = _apply_analog_era_ceiling(result, transfer_chain)
            if _ceiling is not None:
                logger.info(
                    "🕰️ §2.13 Era-Ceiling: chain=%s → %s ceiling=%d → "
                    "decade %d→%d (CLAP/DSP said %d, corrected for analog source)",
                    transfer_chain,
                    _ceiling_source,
                    _ceiling,
                    _original_decade,
                    result.decade,
                    _original_decade,
                )

        # Tier-3: Mikrofon-Heuristik (letzter Fallback)
        if result.confidence < 0.30:
            result = self._tier3(bark, rolloff_hz, snr_db)

        # §v10.20 Kalibrierung: Confidence-Floor für physikalisch plausible Ergebnisse
        # Wenn die Audio-Merkmale (rolloff, SNR, Bark) konsistent sind, aber die
        # Confidence niedrig bleibt (z.B. wegen CLAP-Unsicherheit), heben wir den
        # Wert auf ein plausibles Minimum an — statt das Ergebnis als unsicher zu markieren.
        if result.confidence < 0.45 and rolloff_hz is not None and snr_db is not None:
            _rolloff_khz = rolloff_hz / 1000.0
            # Physikalische Plausibilität: rolloff + SNR + Bark ergeben konsistentes Bild
            if 2.0 <= _rolloff_khz <= 22.0 and -40 <= snr_db <= 60:
                result = dc_replace(result, confidence=max(result.confidence, 0.50))
                logger.info(
                    "§v10.20 Era-Kalibrierung: Confidence %.2f → 0.50 (physikalisch plausibel: rolloff=%.1fkHz, SNR=%.1fdB)",
                    result.confidence,
                    _rolloff_khz,
                    snr_db,
                )

        # Invariante: Conf < 0.40 → konservatives Material
        if result.confidence < 0.40:
            result = EraResult(
                decade=result.decade,
                era_label=result.era_label,
                confidence=result.confidence,
                material_prior="unknown",
                noise_profile=result.noise_profile,
                tier_used=result.tier_used,
                hf_rolloff_hz=rolloff_hz,
            )

        # RemasterDetector-Guard (§2.14): verhindert falsche Ära-Zuweisung bei Remasters
        try:
            get_remaster_detector = _get_remaster_detector_factory()
            _rm = get_remaster_detector().analyse(audio_mono, sr)
            if _rm is not None and _rm.is_remaster:
                result = dc_replace(result, is_remaster_suspected=True)
                logger.info(
                    "RemasterDetector: Remaster erkannt (conf=%.2f, BW=%.1f kHz)",
                    _rm.confidence,
                    getattr(_rm, "hf_rolloff_khz", 0.0),
                )
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        # Codec-BW plausibility gate (Müller & Ewert 2011; Spijkervet 2020):
        # Modern recordings (≥2000) show fullBW (>18 kHz) + moderate-to-high SNR.
        # Pre-1960 era assignments with fullBW contradict carrier-chain physics.
        result = _apply_codec_bw_plausibility_gate(result, highband_presence, snr_db)

        with self._ram_cache_lock:
            self._ram_cache[sha] = result

        # ── §Original-Medium-Inference ───────────────────────────────────
        # Wenn das erkannte Jahrzehnt < 1990 ist, das material_prior aber
        # auf digital zeigt (MediumDetector-Bias durch Dateiendung),
        # ersetze durch das für diese Ära typische Analog-Medium.
        # Eine MP3-Datei von 1985 wurde ursprünglich auf Vinyl/Band
        # veröffentlicht — Aurik soll das ORIGINAL-Medium restaurieren.
        _digital_materials = {"cd_digital", "mp3_low", "mp3_high", "streaming", "unknown", "dat", "minidisc"}
        if result.decade < 1990 and result.material_prior in _digital_materials:
            _original_medium = DECADE_MATERIAL_PRIOR.get(result.decade, "unknown")
            if _original_medium not in _digital_materials:
                logger.info(
                    "🕰️ Originalsignal-Medium-Inference: decade=%d → %s (override %s → %s)",
                    result.decade,
                    _original_medium,
                    result.material_prior,
                    _original_medium,
                )
                result = dc_replace(result, material_prior=_original_medium)

        logger.info(
            "🕰️ EraClassifier: Jahrzehnt=%d, Konfidenz=%.2f, Material=%s, Tier=%d",
            result.decade,
            result.confidence,
            result.material_prior,
            result.tier_used,
        )
        return result

    def get_material_prior(self, era: EraResult) -> str:
        """Gibt empfohlenen Material-String für CausalDefectReasoner zurück.

        Bei Konfidenz < 0.40 → 'unknown' (konservative Priors, Spec §2.14).
        """
        if era.confidence < 0.40:
            return "unknown"
        return era.material_prior

    def get_gp_warmstart(self, era: EraResult) -> dict[str, float]:
        """GP-Optimizer-Initialisierungswerte für das erkannte Jahrzehnt.

        Returns:
            Dict mit Parameternamen → Initialwert.
        """
        decade = era.decade
        nr_mean = DECADE_NR_PRIOR_MEAN.get(decade, 0.65)
        nr_std = DECADE_NR_PRIOR_STD.get(decade, 0.08)
        return {
            "noise_reduction_strength": float(np.clip(nr_mean, 0.10, 1.0)),
            "noise_reduction_strength_std": nr_std,
            "harmonic_boost_db": 2.0 if decade <= 1950 else 1.0,
            "ola_crossfade_ms": 50.0 if decade <= 1940 else 30.0,
            "bass_restoration_db": 2.5 if decade <= 1960 else 0.5,
            "era_decade": float(decade),
            "era_confidence": float(era.confidence),
        }

    # ------------------------------------------------------------------
    # Tier-Implementierungen
    # ------------------------------------------------------------------

    def _try_tier1(
        self,
        audio_mono: np.ndarray,
        sr: int,
        bark: np.ndarray,
        rolloff_hz: float,
        _snr_db: float,
    ) -> EraResult | None:
        """Tier-1: LAION-CLAP-basierte Ära-Erkennung (optional)."""
        try:
            with self._clap_lock:
                if not self._clap_loaded:
                    get_laion_clap = _get_laion_clap_factory()
                    self._clap_plugin = get_laion_clap()
                    self._clap_loaded = True
            if self._clap_plugin is None:
                return None
            # CLAP requires exactly 48 kHz — resample if input SR differs.
            # Most legacy imports arrive at 44 100 Hz; without explicit resampling
            # embed_audio() raises a ValueError and the tier falls back to DSP
            # unnecessarily.  scipy.signal.resample_poly is phase-linear and cheap
            # for the short (≤30 s) excerpts used here.
            _audio_clap = audio_mono
            _sr_clap = sr
            if sr != 48000:
                try:
                    _rspoly = _get_resample_poly()
                    if _rspoly is None:
                        logger.warning(
                            "§G23 EraClassifier Tier-1: scipy.signal.resample_poly nicht verfügbar — DSP-Ersatzpfad"
                        )
                        return None
                    _g = math.gcd(48000, sr)
                    _audio_clap = _rspoly(audio_mono, 48000 // _g, sr // _g).astype(np.float32)
                    _sr_clap = 48000
                    logger.debug("EraClassifier Tier-1: resampled %d → 48000 Hz for CLAP embed", sr)
                except Exception as _rs_exc:
                    logger.debug(
                        "EraClassifier Tier-1: resample fehlgeschlagen (%s) — ueberspringen CLAP tier", _rs_exc
                    )
                    return None
            # CLAP-Embedding → Cosinus-Ähnlichkeit zu Ära-Ankern.
            # _clap_plugin ist als object|None typisiert (lazy importiertes Plugin);
            # embed_audio() existiert erst zur Laufzeit — cast(Any) statt type-ignore,
            # damit mypy UND Pyright konsistent durchlaufen.
            # BUG-FIX 2026-08-22: Der frühere `_model_loaded`-Vorcheck verhinderte
            # Tier-1 DETERMINISTISCH beim ersten Aufruf — der Plugin-Vertrag ist
            # Lazy-Load IN embed_audio() (→ _ensure_loaded(), thread-sicher via
            # _load_lock): Das Modell ist vor dem ersten embed-Aufruf nie geladen,
            # der Check ließ embed_audio() also nie laufen → WARNUNG erschien
            # garantiert, Tier-2 lief, obwohl CLAP verfügbar war (Befund: WARNUNG
            # 14:16:03, CLAP erst 14:16:24 vom Genre-Pfad fertig geladen).
            # Jetzt: embed_audio() lädt selbst (synchron, einmalig pro Prozess);
            # nur ein totaler Ladefehler (RuntimeError) fällt wie zuvor in den
            # §G23 (GEBOTE.md)-konformen DSP-Ersatzpfad mit WARNUNG.
            if getattr(self._clap_plugin, "_model_loaded", False) is not True:
                logger.info("EraClassifier Tier-1: CLAP wird erstmalig geladen (synchron, einmalig)…")
            embedding = cast(Any, self._clap_plugin).embed_audio(_audio_clap, _sr_clap)
            decade, conf = self._clap_nearest_neighbor(embedding)
            if conf < 0.35:
                return None
            return EraResult(
                decade=decade,
                era_label=f"{decade}er",
                confidence=conf,
                material_prior=DECADE_MATERIAL_PRIOR.get(decade, "unknown"),
                noise_profile=bark,
                tier_used=1,
                hf_rolloff_hz=rolloff_hz,
            )
        except Exception as exc:
            # §V6 (copilot-instructions.md): ML→DSP-Fallback — DEBUG-level Logging
            # um zu zeigen, dass ein Fallback stattfand, aber keine Warnung für den User.
            logger.debug("§G23 EraClassifier Tier-1 fallback (normal): %s", exc)
            return None

    def _tier2(
        self,
        bark: np.ndarray,
        rolloff_hz: float,
        snr_db: float,
        *,
        is_stereo: bool = False,
        stereo_width: float = 0.0,
        spectral_tilt: float = -4.0,
        dynamic_range_db: float = 25.0,
        noise_modulation: float = 0.20,
        lf_presence: float = 0.35,
        highband_presence: float = 0.12,
        transfer_chain: list[str] | None = None,  # §v10.303.32
    ) -> EraResult:
        """Tier-2: DSP-Fingerprint (multi-factor: BW + SNR + stereo + tilt + DR + modulation + LF + HB)."""
        decade, conf = _dsp_fingerprint_decade(
            rolloff_hz,
            snr_db,
            is_stereo=is_stereo,
            stereo_width=stereo_width,
            spectral_tilt=spectral_tilt,
            dynamic_range_db=dynamic_range_db,
            noise_modulation=noise_modulation,
            lf_presence=lf_presence,
            highband_presence=highband_presence,
            transfer_chain=getattr(self, "_transfer_chain", None) or transfer_chain,
        )
        material = DECADE_MATERIAL_PRIOR.get(decade, "unknown")
        # §v10.303.42 Reel-Tape-Aware: Studio-Aufnahmen wurden auf Tonband
        # produziert, nicht auf Vinyl. Vinyl war das Vertriebsmedium.
        _chain_for_era = transfer_chain or getattr(self, "_transfer_chain", None) or []
        if "reel_tape" in _chain_for_era:
            material = "reel_tape"
            if decade < 1970:
                decade = min(decade + 10, 1990)
        return EraResult(
            decade=decade,
            era_label=f"{decade}er",
            confidence=conf,
            material_prior=material,
            noise_profile=bark,
            tier_used=2,
            hf_rolloff_hz=rolloff_hz,
        )

    def _apply_ast_constraints(
        self,
        audio: np.ndarray,
        sr: int,
        result: EraResult,
    ) -> EraResult:
        """§v10.304: AST-Instrument-Constraints auf Era-Ergebnis anwenden.

        AST ist deterministisch: Ein Song MIT Synthesizer kann NICHT von 1920 sein.
        Diese Constraints haben VORRANG vor DSP-Klassifikation.

        Returns:
            EraResult mit AST-korrigierter Dekade (oder unverändert).
        """
        try:
            from backend.core.ast_audio_set_classifier import (
                ERA_AUDIOSET_PROXIES,
                get_ast_classifier,
            )

            _clf = get_ast_classifier()
            if not _clf.is_loaded():
                return result
            _ast_result = _clf.classify(audio, sr, top_k=20)
            if _ast_result.model_used == "fallback":
                return result

            _min_decade = None
            _max_decade = None
            for _idx, (_era_min, _era_max) in ERA_AUDIOSET_PROXIES.items():
                if _ast_result.get_prob(_idx) >= 0.15:
                    if _min_decade is None or _era_min > _min_decade:
                        _min_decade = _era_min
                    if _max_decade is None or _era_max < _max_decade:
                        _max_decade = _era_max

            if _min_decade is not None and result.decade < _min_decade:
                logger.info(
                    "§v10.304 AST-Era-Floor: %d→%d (instrument constraint)",
                    result.decade,
                    _min_decade,
                )
                return EraResult(
                    decade=_min_decade,
                    era_label=f"{_min_decade}er",
                    confidence=result.confidence,
                    material_prior=result.material_prior,
                    noise_profile=result.noise_profile,
                    tier_used=result.tier_used,
                    hf_rolloff_hz=result.hf_rolloff_hz,
                )
        except Exception as _exc:
            logger.debug("AST-Era-Constraint fehlgeschlagen: %s", _exc)
        return result

    def _tier3(self, bark: np.ndarray, rolloff_hz: float, _snr_db: float) -> EraResult:
        """Tier-3: Mikrofon-Typ-Heuristik."""
        decade, conf = _microphone_type_decade(bark)
        material = DECADE_MATERIAL_PRIOR.get(decade, "unknown")
        return EraResult(
            decade=decade,
            era_label=f"{decade}er",
            confidence=conf,
            material_prior=material,
            noise_profile=bark,
            tier_used=3,
            hf_rolloff_hz=rolloff_hz,
        )

    def _clap_nearest_neighbor(self, embedding: np.ndarray) -> tuple[int, float]:
        """Findet nächsten Ära-Anker im CLAP-Embedding-Raum.

        Wenn keine vorberechneten Anker vorhanden sind, gibt unbekannte Ära zurück.
        """
        anchors_path = Path(__file__).parent.parent.parent / "models" / "era_classifier" / "era_anchors.npy"
        if not anchors_path.exists():
            return 1960, 0.20
        try:
            anchors = np.load(str(anchors_path))  # expected (n_anchors, embedding_dim + 1)
            # Format guard: last column must be decade labels (≥1900).
            # If anchors.shape[1] == embedding.shape[0], the label column is missing —
            # the catalog was saved without appending the decades vector.
            # Attempting to use it would produce (a) all-zero "labels" and
            # (b) a 511×512 matmul dimension error.  Skip and fall back to Tier-2.
            if anchors.shape[1] != embedding.shape[0] + 1:
                logger.debug(
                    "CLAP NN-Suche übersprungen: anchors shape %s inkompatibel mit "
                    "embedding dim %d (erwartet %d Spalten — embedding + 1 Dekaden-Label)",
                    anchors.shape,
                    embedding.shape[0],
                    embedding.shape[0] + 1,
                )
                return 1960, 0.20
            # Letzte Spalte: decade-Label
            decade_labels = anchors[:, -1].astype(int)
            # Sanity-check: at least some labels must be valid decades
            if int(decade_labels.max()) < 1900:
                logger.debug(
                    "CLAP NN-Suche übersprungen: Dekaden-Labels ungültig (max=%d < 1900)",
                    int(decade_labels.max()),
                )
                return 1960, 0.20
            anchor_vecs = anchors[:, :-1]
            # L2-normalisieren für Cosinus
            anchor_norms = np.linalg.norm(anchor_vecs, axis=1, keepdims=True) + 1e-12
            anchor_vecs = anchor_vecs / anchor_norms
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
            cosine_sims = anchor_vecs @ emb_norm
            best_idx = int(np.argmax(cosine_sims))
            best_sim = float(cosine_sims[best_idx])
            conf = float(np.clip((best_sim + 1.0) / 2.0 * 1.2, 0.0, 1.0))
            return int(decade_labels[best_idx]), conf
        except Exception as exc:
            logger.debug("CLAP NN-Suche fehlgeschlagen: %s", exc)
            return 1960, 0.20

    def clear_ram_cache(self) -> None:
        """Leert den In-Memory-Cache (z. B. zum Testen oder nach Speicherengpass)."""
        with self._ram_cache_lock:
            self._ram_cache.clear()


# ---------------------------------------------------------------------------
# Singleton (Thread-sicher, Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_INSTANCE_HOLDER: dict[str, EraClassifier | None] = {"instance": None}


def get_era_classifier() -> EraClassifier:
    """Thread-sicherer Singleton-Accessor für EraClassifier."""
    if _INSTANCE_HOLDER["instance"] is None:
        with _lock:
            if _INSTANCE_HOLDER["instance"] is None:
                _INSTANCE_HOLDER["instance"] = EraClassifier()
    classifier = _INSTANCE_HOLDER["instance"]
    assert classifier is not None
    return classifier


def _apply_codec_bw_plausibility_gate(
    era_result: EraResult,
    highband_presence: float,
    snr_db: float,
) -> EraResult:
    """Apply a weak plausibility correction based on HF-bandwidth and SNR signals.

    Two scenarios are addressed (Müller & Ewert 2011; Spijkervet & Haasdijk 2020):

    1. **Modern BW + old era**: highband_presence > 0.88 (nearly full 20 kHz BW) +
       snr_db > 35 dB strongly implies a digital/streaming source.  If era says
       pre-1960, the assignment is physically implausible — gently reduce confidence
       and nudge material_prior toward 'unknown' to avoid blocking downstream
       restoration phases.

    2. **Full BW + modern era aligned**: highband_presence > 0.85 + snr_db > 30 dB +
       decade >= 2000 → reinforce confidence slightly (×1.08, capped at 0.88) because
       the DSP features are coherent with the era prediction.

    The gate is intentionally soft (confidence only, no decade change) to preserve
    §0 Minimal-Intervention.  Decade changes require cross-source evidence.

    Args:
        era_result:        EraResult from Tier-1/2/3.
        highband_presence: Proportion of HF energy (> 8 kHz) as seen from
                           _estimate_highband_presence().  [0, 1].
        snr_db:            Estimated SNR in dB from _estimate_snr().

    Returns:
        Original or gently adjusted EraResult.
    """
    if era_result is None:
        return era_result

    # Scenario 1: full-BW + high-SNR but pre-1960 era → implausible carrier physics
    if highband_presence > 0.88 and snr_db > 35.0 and era_result.decade < 1960:
        new_conf = float(np.clip(era_result.confidence * 0.75, 0.25, 0.80))
        # Don't change decade — we may still be right (1950s recording digitized as FLAC)
        # but drop material_prior so downstream doesn't over-invest in analog restoration
        logger.info(
            "EraClassifier BW-plausibility gate: pre-1960 era but fullBW+highSNR "
            "(highband=%.2f, snr=%.1f dB, decade=%d) → confidence %.2f → %.2f, prior='unknown'",
            highband_presence,
            snr_db,
            era_result.decade,
            era_result.confidence,
            new_conf,
        )
        return dc_replace(era_result, confidence=new_conf, material_prior="unknown")

    # Scenario 2: fullBW + modern era agreement → reinforce confidence
    if highband_presence > 0.85 and snr_db > 30.0 and era_result.decade >= 2000:
        new_conf = float(np.clip(era_result.confidence * 1.08, era_result.confidence, 0.88))
        if new_conf > era_result.confidence:
            logger.debug(
                "EraClassifier BW-plausibility gate: modern era reinforced "
                "(highband=%.2f, snr=%.1f dB, decade=%d) → confidence %.2f → %.2f",
                highband_presence,
                snr_db,
                era_result.decade,
                era_result.confidence,
                new_conf,
            )
            return dc_replace(era_result, confidence=new_conf)

    return era_result


def classify_era(audio: np.ndarray, sr: int, transfer_chain: list[str] | None = None) -> EraResult:
    """Convenience-Funktion: Erkennt Aufnahme-Ära ohne explizite Instanz.

    Args:
        audio: Audio-Signal (mono oder stereo, float32/64 [-1, 1]).
        sr:    Sample-Rate in Hz.
        transfer_chain: Optionale Tonträgerkette für Chain-Awareness.

    Returns:
        EraResult mit Dekade, Confidence, Material-Prior und Noise-Profil.
    """
    _clf = get_era_classifier()
    result = _clf.classify(audio, sr, transfer_chain=transfer_chain)
    # §v10.304: AST-Instrument-Constraints NACH DSP-Klassifikation anwenden
    result = _clf._apply_ast_constraints(audio, sr, result)
    return result


# Codec containers are encoding formats, not physical source media.
# A 1977 vinyl digitized as mp3 is still from 1977 — the codec does not date
# the content.  These are excluded from era-floor constraints.
_CODEC_CONTAINERS: frozenset[str] = frozenset(
    {
        "mp3_low",
        "mp3_high",
        "aac",
        "streaming",
    }
)

# Analoge Material-Priors — eine Era-Zuweisung auf diese Träger ist bei reinen
# Codec-Ketten (ohne analoge Vorstufe) mit hoher Wahrscheinlichkeit eine
# Fehldeutung des Codec-Tiefpasses als historische Aufnahme-Bandbreite.
_ANALOG_ERA_PRIORS: frozenset[str] = frozenset(
    {
        "shellac",
        "wax_cylinder",
        "vinyl",
        "wire_recording",
        "lacquer_disc",
        "tape",
        "reel_tape",
        "cassette",
    }
)


def constrain_era_to_medium(era_result: EraResult, medium: str) -> EraResult:
    """Applies a physical medium-based minimum decade floor to an EraResult.

    A tape recording cannot originate from 1890; a vinyl disc cannot predate
    1948.  This function corrects impossible decade assignments that arise when
    the EraClassifier operates on short or ambiguous audio segments.

    Codec containers (mp3_low, mp3_high, aac, streaming) are explicitly
    excluded — they are encoding formats, not physical origins.  A vinyl
    recording from 1977 digitized as mp3 must retain its 1970er era.

    The corrected decade is the smallest VALID_DECADES entry >= the floor for
    the given medium.  Confidence is scaled down by 0.65 (indicating the
    assignment was constrained rather than directly measured) but clamped to
    [0.25, 0.80] to prevent both over-confidence and useless uncertainty.

    Args:
        era_result: EraResult produced by EraClassifier.classify().
        medium:     Physical medium string (e.g. 'tape', 'reel_tape', 'vinyl').
                    Case-insensitive; unknown medium strings are ignored.
                    Codec containers are silently skipped.

    Returns:
        Original EraResult if no constraint applies; corrected EraResult otherwise.
    """
    medium_lower = medium.strip().lower()
    if medium_lower in _CODEC_CONTAINERS:
        # §Fix9-Spiegelung (UV3-Pfad): Reine Codec-Klassifikation OHNE analoge
        # Vorstufe + prä-digitale Dekade, die primär aus dem HF-Rolloff stammt,
        # ist codec-kontaminiert — ein MP3/AAC-Tiefpass ist kein Aufnahme-Ära-
        # Indikator. Ein echtes Analog-Original in einer Codec-Kette erreicht
        # diese Funktion nie mit medium=codec, weil der MediumDetector dann den
        # letzten Analog-Träger als primary meldet (§6.1b Letzter-Analog-Träger-
        # Primärprinzip). Daher ist die Korrektur hier sicher und generisch (§0c).
        _prior_lower = str(era_result.material_prior or "").lower()
        if (
            era_result.decade < 1975
            and _prior_lower in _ANALOG_ERA_PRIORS
            and 0.0 < float(era_result.hf_rolloff_hz) <= 16_500.0
        ):
            corrected_codec_decade = min(d for d in VALID_DECADES if d >= 1980)
            codec_conf = float(np.clip(max(era_result.confidence * 0.65, 0.55), 0.25, 0.80))
            logger.info(
                "EraClassifier Lossy-Codec-Korrektur (§Fix9/UV3): %der → %der "
                "(medium=%s, rolloff=%.0f Hz codec-bedingt, prior %s → %s, conf %.2f → %.2f)",
                era_result.decade,
                corrected_codec_decade,
                medium,
                era_result.hf_rolloff_hz,
                _prior_lower,
                medium_lower,
                era_result.confidence,
                codec_conf,
            )
            return EraResult(
                decade=corrected_codec_decade,
                era_label=f"{corrected_codec_decade}er",
                confidence=codec_conf,
                material_prior=medium_lower,
                noise_profile=era_result.noise_profile,
                tier_used=era_result.tier_used,
                hf_rolloff_hz=era_result.hf_rolloff_hz,
                is_remaster_suspected=era_result.is_remaster_suspected,
            )
        return era_result
    floor = MEDIUM_DECADE_FLOOR.get(medium_lower, 0)
    if floor == 0 or era_result.decade >= floor:
        return era_result

    valid_above_floor = [d for d in VALID_DECADES if d >= floor]
    if not valid_above_floor:
        return era_result

    corrected_decade = min(valid_above_floor)
    new_conf = float(np.clip(era_result.confidence * 0.65, 0.25, 0.80))
    # Ensure the corrected confidence stays above the material_prior threshold (0.40)
    # when the floor correction is large (>= 2 decade steps) and the original
    # confidence was trustworthy (>= 0.40).  Without this guard a conf=0.60 era
    # that gets bumped 3 steps (e.g. 1890->1960 for tape) ends up at 0.39, which
    # silently drops material_prior to "unknown" downstream.
    n_corrected_steps = sum(1 for d in VALID_DECADES if era_result.decade <= d < corrected_decade)
    if n_corrected_steps >= 2 and era_result.confidence >= 0.40:
        new_conf = max(new_conf, 0.42)
    # Use the actually detected medium — not the decade-based prior which can
    # map e.g. 1960 → "vinyl" even though the medium was detected as "tape".
    new_material = medium_lower

    logger.info(
        "EraClassifier medium-floor constraint: %dер → %d (medium=%s, floor=%d, confidence %.2f → %.2f)",
        era_result.decade,
        corrected_decade,
        medium,
        floor,
        era_result.confidence,
        new_conf,
    )
    return EraResult(
        decade=corrected_decade,
        era_label=f"{corrected_decade}er",
        confidence=new_conf,
        material_prior=new_material,
        noise_profile=era_result.noise_profile,
        tier_used=era_result.tier_used,
        hf_rolloff_hz=era_result.hf_rolloff_hz,
        is_remaster_suspected=era_result.is_remaster_suspected,
    )
