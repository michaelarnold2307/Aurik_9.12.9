"""
Preservation Metrics — SOTA Verification for Blind Tests (§G46–§G48)

Three standalone metrics for verifying preservation quality:
  §G46  Harmonic Preservation Score — overtone structure intact?
  §G47  Transient Preservation Score — attacks survive processing?
  §G48  Vocal Formant Preservation Score — voice character unchanged?

Each function: (original, processed, sr) → score in [0, 1]
  1.0 = perfectly preserved  0.0 = completely destroyed

Scientific basis:
  Harmonic:   F0-tracking → harmonic peak comparison (Fletcher, 1964)
  Transient:  Spectral-flux onset detection (Bello et al., 2005)
  Formant:    LPC-based spectral envelope comparison (Markel & Gray, 1976)

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


# ── §G46 Harmonic Preservation ──────────────────────────────────────────


def compute_harmonic_preservation_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    f0_min: float = 80.0,
    f0_max: float = 1200.0,
    n_harmonics: int = 8,
) -> float:
    """§G46: Harmonic Preservation via Harmonic-to-Noise Ratio (HNR).

    Compares HNR before/after processing across voiced frames.
    HNR = energy at harmonic frequencies / energy between harmonics.
    A significant HNR drop indicates harmonic structure damage.

    Algorithm:
    1. Detect F0 per frame via autocorrelation
    2. For voiced frames: measure harmonic energy (at n·F0 ± 5% bandwidth)
       and inter-harmonic noise energy (between harmonics)
    3. Compute HNR ratio: HNR_processed / HNR_original per frame
    4. Weight by frame energy, average across frames
    5. Score = weighted average of min(ratio, 1.0)

    Sensitive to: over-smoothing, harmonic loss, excessive denoising,
    over-compression, lowpass filtering, spectral flattening.
    """
    orig = _to_mono(original)
    proc = _to_mono(processed)
    n = min(len(orig), len(proc))
    if n < 4096:
        return 1.0
    orig = orig[:n].astype(np.float64)
    proc = proc[:n].astype(np.float64)

    # Frame-based analysis
    frame_len = int(0.050 * sr)  # 50ms frames
    hop = frame_len // 2
    n_frames = (n - frame_len) // hop + 1
    if n_frames < 3:
        return 1.0

    win = np.hanning(frame_len)
    hnr_ratios = []
    weights = []

    for i in range(n_frames):
        start = i * hop
        fo = orig[start : start + frame_len] * win
        fp = proc[start : start + frame_len] * win

        # Frame energy
        e_o = float(np.sum(fo**2))
        if e_o < 1e-10:
            continue

        # Detect F0
        f0 = _detect_f0_autocorr(fo, sr, f0_min, f0_max)
        if f0 is None:
            # Unvoiced: skip (harmonic preservation only matters in voiced frames)
            continue

        # Compute HNR for original and processed
        hnr_o = _compute_hnr(fo, sr, f0, n_harmonics)
        hnr_p = _compute_hnr(fp, sr, f0, n_harmonics)

        if hnr_o is None or hnr_p is None or hnr_o < 1e-3:
            continue

        # HNR ratio: processed / original
        # > 1.0 = processing ADDED harmonics (unlikely but possible)
        # < 1.0 = processing LOST harmonics
        ratio = hnr_p / hnr_o

        hnr_ratios.append(min(ratio, 1.0))  # Cap at 1.0 (no bonus for adding harmonics)
        weights.append(e_o)

    if not hnr_ratios:
        # No voiced frames found: fall back to spectral flatness
        return _harmonic_fallback_flatness(orig, proc, sr)

    weights = np.array(weights, dtype=np.float64)  # type: ignore[assignment]
    weights /= np.sum(weights) + 1e-10
    return float(np.dot(hnr_ratios, weights))


def _detect_f0_autocorr(signal: np.ndarray, sr: int, f0_min: float, f0_max: float) -> float | None:
    """Detect F0 via autocorrelation (robust, no FFT dependency)."""
    max_lag = int(sr / f0_min)
    min_lag = int(sr / f0_max)
    if max_lag >= len(signal) or min_lag < 2:
        return None
    # Autocorrelation
    s = signal - np.mean(signal)
    corr = np.correlate(s, s, mode="full")
    corr = corr[len(corr) // 2 :]  # Keep positive lags
    corr[:min_lag] = 0.0
    if len(corr) <= max_lag:
        return None
    peak_idx = int(np.argmax(corr[min_lag:max_lag])) + min_lag
    # Confidence: peak height relative to zero-lag
    confidence = corr[peak_idx] / max(corr[0], 1e-15)
    if confidence < 0.15:
        return None
    return sr / peak_idx


def _compute_hnr(signal: np.ndarray, sr: int, f0: float, n_harmonics: int) -> float | None:
    """Compute Harmonic-to-Noise Ratio in dB."""
    n_fft = 4096
    if n_fft >= len(signal):
        n_fft = 1
        while n_fft < len(signal):
            n_fft <<= 1
    win = np.hanning(min(n_fft, len(signal)))
    spec = np.abs(np.fft.rfft(signal[:n_fft] * win, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Harmonic energy: narrow bands around n·F0 (±5% bandwidth)
    harm_energy = 0.0
    harm_mask = np.zeros(len(freqs), dtype=bool)
    for k in range(1, n_harmonics + 1):
        hz = f0 * k
        if hz >= sr / 2:
            break
        bw = hz * 0.05  # ±5%
        lo = int(np.searchsorted(freqs, hz - bw))
        hi = int(np.searchsorted(freqs, hz + bw, side="right"))
        lo = max(0, lo)
        hi = min(len(freqs) - 1, hi)
        if hi > lo:
            harm_energy += float(np.sum(spec[lo:hi] ** 2))
            harm_mask[lo:hi] = True

    if harm_energy < 1e-20:
        return None

    # Noise energy: everything outside harmonic bands (in voiced range)
    voice_range = (freqs >= 80.0) & (freqs <= min(f0 * n_harmonics * 1.1, sr / 2))
    noise_mask = voice_range & (~harm_mask)
    if not np.any(noise_mask):
        return 100.0  # Pure harmonics, no noise

    noise_energy = float(np.sum(spec[noise_mask] ** 2))
    if noise_energy < 1e-20:
        return 100.0

    hnr = harm_energy / noise_energy
    return float(hnr)


def _harmonic_fallback_flatness(orig: np.ndarray, proc: np.ndarray, sr: int) -> float:
    """Fallback: compare spectral flatness in midrange."""
    n_fft = 4096
    if len(orig) < n_fft:
        n_fft = 1
        while n_fft < len(orig):
            n_fft <<= 1
    win = np.hanning(min(n_fft, len(orig)))
    so = np.abs(np.fft.rfft(orig[:n_fft] * win, n=n_fft))
    sp = np.abs(np.fft.rfft(proc[:n_fft] * win, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mask = (freqs >= 300) & (freqs <= 4000)
    if not np.any(mask):
        return 1.0

    # Spectral flatness: geometric mean / arithmetic mean
    def _flatness(x):
        x = np.maximum(x, 1e-15)
        return float(np.exp(np.mean(np.log(x))) / np.mean(x))

    fo = _flatness(so[mask])
    fp = _flatness(sp[mask])
    # Flatter spectrum = less harmonic structure = damaged
    # Ratio: fp/fo > 1 means processing flattened the spectrum
    if fo < 1e-10:
        return 1.0
    ratio = min(fo, fp) / max(fo, fp)
    return float(ratio)


# ── §G47 Transient Preservation ─────────────────────────────────────────


def compute_transient_preservation_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    onset_threshold: float = 0.3,
    freq_lo: float = 2000.0,
    freq_hi: float = 8000.0,
) -> float:
    """§G47: Vergleicht Onset-Struktur vor/nach Processing.

    Algorithmus (Bello et al., 2005):
    1. Berechne spektrale Energie in Transienten-Band (2–8 kHz)
    2. Onset Detection Function = positive Halbwelle der Energie-Differenz
    3. Vergleiche Onset-Peaks: Position + relative Stärke
    4. Score = recall (wie viele Original-Onsets sind erhalten?)

    Returns: float [0,1] — 1.0 = alle Onsets erhalten.
    """
    mono_orig = _to_mono(original)
    mono_proc = _to_mono(processed)
    n_min = min(len(mono_orig), len(mono_proc))
    if n_min < 2048:
        return 1.0

    mono_orig = mono_orig[:n_min].astype(np.float64)
    mono_proc = mono_proc[:n_min].astype(np.float64)

    n_fft = 1024
    hop = 256
    n_frames = (n_min - n_fft) // hop + 1
    if n_frames < 4:
        return 1.0

    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lo = np.searchsorted(freqs, freq_lo)
    hi = np.searchsorted(freqs, freq_hi)

    def onset_func(audio: np.ndarray) -> np.ndarray:
        energy = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop
            frame = audio[s : s + n_fft] * win
            spec = np.abs(np.fft.rfft(frame))
            energy[i] = float(np.sum(spec[lo:hi] ** 2))
        if np.max(energy) < 1e-15:
            return cast(np.ndarray, (np.zeros(n_frames, dtype=np.float64)))
        energy_db = 10.0 * np.log10(energy + 1e-15)
        # Onset = positive half-wave of first difference
        diff = np.diff(energy_db)
        onset = np.maximum(diff, 0.0)
        onset /= np.max(onset) + 1e-15
        return cast(np.ndarray, onset)

    onset_o = onset_func(mono_orig)
    onset_p = onset_func(mono_proc)

    if np.max(onset_o) < onset_threshold:
        return 1.0  # No significant onsets in original — nothing to preserve

    # Find onset peaks in original
    orig_peaks = _find_peaks(onset_o, threshold=onset_threshold * np.max(onset_o))

    if len(orig_peaks) == 0:
        return 1.0

    # Check each original onset: does processed have a peak at the same position?
    preserved = 0
    for idx in orig_peaks:
        # Look in ±2 frame window
        lo_w = max(0, idx - 2)
        hi_w = min(len(onset_p) - 1, idx + 3)
        if np.max(onset_p[lo_w:hi_w]) >= onset_threshold * np.max(onset_p) * 0.5:
            preserved += 1

    score = float(preserved) / float(len(orig_peaks))
    return score


def _find_peaks(signal: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Finde lokale Maxima über threshold."""
    if len(signal) < 3:
        return cast(np.ndarray, (np.array([], dtype=int)))
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    return cast(np.ndarray, (np.array(peaks, dtype=int)))


# ── §G48 Vocal Formant Preservation ─────────────────────────────────────


def compute_formant_preservation_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    formant_range: tuple = (200.0, 4000.0),
    n_formants: int = 4,
) -> float:
    """§G48: Vergleicht Vokal-Formanten (F1–F4) vor/nach Processing.

    Algorithmus:
    1. Bandpass 200–4000 Hz (Gesangs-Formantbereich)
    2. LPC (Linear Predictive Coding) Ordnung = sr/1000 + 4
    3. Formanten = Peaks im LPC-Spektrum
    4. Vergleiche Formant-Positionen: relative Abweichung < 5% = erhalten
    5. Zusätzlich: spektrale Energie-Erhaltung im Formantbereich

    Returns: float [0,1] — 1.0 = Formanten identisch.
    """
    mono_orig = _to_mono(original)
    mono_proc = _to_mono(processed)
    n_min = min(len(mono_orig), len(mono_proc))
    if n_min < 4096:
        return 1.0

    mono_orig = mono_orig[:n_min].astype(np.float64)
    mono_proc = mono_proc[:n_min].astype(np.float64)

    # Process in 100ms windows, take best-scoring window
    win_len = int(0.100 * sr)
    hop = win_len // 2
    n_frames = (n_min - win_len) // hop + 1
    if n_frames < 3:
        return 1.0

    scores = []
    weights = []

    for i in range(n_frames):
        start = i * hop
        fo = mono_orig[start : start + win_len]
        fp = mono_proc[start : start + win_len]

        rms_o = float(np.sqrt(np.mean(fo**2)))
        if rms_o < 1e-6:
            continue  # Silence — skip

        formants_o = _extract_formants(fo, sr, n_formants)
        formants_p = _extract_formants(fp, sr, n_formants)

        if not formants_o or not formants_p:
            continue

        # Compare formant positions
        n_common = min(len(formants_o), len(formants_p))
        deviations = []
        for j in range(n_common):
            f_o = formants_o[j]
            f_p = formants_p[j]
            if f_o > 0:
                rel_dev = abs(f_p - f_o) / f_o
                deviations.append(rel_dev)

        if not deviations:
            continue

        # Score based on mean relative deviation
        mean_dev = float(np.mean(deviations))
        # <5% deviation → score 1.0, >20% → score 0.0
        frame_score = float(max(0.0, min(1.0, 1.0 - (mean_dev - 0.05) / 0.15)))

        scores.append(frame_score)
        weights.append(rms_o)

    if not scores:
        # Fallback: spectral energy preservation in formant range
        return _formant_energy_fallback(mono_orig, mono_proc, sr, formant_range)

    weights = np.array(weights, dtype=np.float64)  # type: ignore[assignment]
    weights /= np.sum(weights) + 1e-10
    return float(np.dot(scores, weights))


def _extract_formants(audio: np.ndarray, sr: int, n_formants: int = 4) -> list[float]:
    """Extrahiere Formant-Frequenzen via LPC."""
    try:
        # LPC order: ~sr/1000 for formant resolution
        order = min(int(sr / 1000.0) + 4, 50)
        n = len(audio)

        # Autocorrelation
        r = np.correlate(audio, audio, mode="full")[n - 1 : n + order]
        r = r.astype(np.float64)

        if abs(r[0]) < 1e-15:
            return []

        # Levinson-Durbin
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        e = float(r[0])
        for k in range(1, order + 1):
            lam = float(r[k])
            for j in range(1, k):
                lam += a[j] * r[k - j]
            lam = -lam / max(e, 1e-15)
            a_prev = a.copy()
            for j in range(1, k):
                a[j] = a_prev[j] + lam * a_prev[k - j]
            a[k] = lam
            e *= 1.0 - lam**2

        if e <= 0:
            return []

        # Frequency response of LPC filter → find peaks
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        # H(z) = 1 / A(z), A(z) = 1 + a₁z⁻¹ + ...
        # |A(e^(jω))|² → peaks in |H|² = 1/|A|²
        A = np.fft.rfft(a, n=n_fft)
        H = 1.0 / (np.abs(A) + 1e-10)
        H_db = 20.0 * np.log10(H)

        # Find peaks in 200-4000 Hz range
        lo = np.searchsorted(freqs, 200.0)
        hi = np.searchsorted(freqs, 4000.0)
        if hi <= lo:
            return []

        peaks = _find_peaks(H_db[lo:hi], threshold=float(np.mean(H_db[lo:hi])))
        peak_values = [(freqs[lo + p], H_db[lo + p]) for p in peaks if lo + p < len(freqs)]
        peak_values.sort(key=lambda x: -x[1])  # Sort by magnitude
        formants = [f for f, _ in peak_values[:n_formants]]
        formants.sort()  # Sort by frequency
        return formants
    except Exception:
        return []


def _formant_energy_fallback(orig: np.ndarray, proc: np.ndarray, sr: int, formant_range: tuple) -> float:
    """Fallback: Energie-Erhaltung im Formantbereich."""
    n_fft = 2048
    n_min = min(len(orig), len(proc))
    if n_min < n_fft:
        return 1.0

    fo = orig[:n_min]
    fp = proc[:n_min]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lo = np.searchsorted(freqs, formant_range[0])
    hi = np.searchsorted(freqs, formant_range[1])

    # Average over blocks
    hop = n_fft // 2
    n_blocks = (n_min - n_fft) // hop + 1
    ratios = []
    for i in range(n_blocks):
        s = i * hop
        spec_o = np.abs(np.fft.rfft(fo[s : s + n_fft] * np.hanning(n_fft)))
        spec_p = np.abs(np.fft.rfft(fp[s : s + n_fft] * np.hanning(n_fft)))
        e_o = float(np.sum(spec_o[lo:hi] ** 2))
        e_p = float(np.sum(spec_p[lo:hi] ** 2))
        if e_o > 1e-10:
            ratios.append(min(e_p / e_o, e_o / e_p))  # Symmetric ratio
    if not ratios:
        return 1.0
    return float(np.mean(ratios))


# ── §G52 Micro-Dynamics Preservation ───────────────────────────────────


def compute_micro_dynamics_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    window_ms: float = 200.0,
) -> float:
    """§G52: Micro-dynamics preservation via crest factor distribution.

    Compares short-term crest factor (peak/RMS) distributions.
    Over-compression narrows the distribution → lower score.

    Algorithm:
    1. Compute crest factor in `window_ms` windows (50% overlap)
    2. Build histograms of crest values for original and processed
    3. Compare histogram correlation (Earth Mover's Distance-like)
    4. Score = overlap of distributions

    Returns: float [0,1] — 1.0 = identical micro-dynamics.
    """
    orig = _to_mono(original)
    proc = _to_mono(processed)
    n = min(len(orig), len(proc))
    if n < 4096:
        return 1.0
    orig = orig[:n].astype(np.float64)
    proc = proc[:n].astype(np.float64)

    win_s = int(window_ms * sr / 1000.0)
    if win_s < 64:
        return 1.0
    hop = win_s // 2
    n_frames = (n - win_s) // hop + 1
    if n_frames < 5:
        return 1.0

    crest_o = np.zeros(n_frames, dtype=np.float64)
    crest_p = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        s = i * hop
        fo = orig[s : s + win_s]
        fp = proc[s : s + win_s]
        rms_o = float(np.sqrt(np.mean(fo**2)))
        rms_p = float(np.sqrt(np.mean(fp**2)))
        peak_o = float(np.max(np.abs(fo)))
        peak_p = float(np.max(np.abs(fp)))
        crest_o[i] = peak_o / max(rms_o, 1e-10)
        crest_p[i] = peak_p / max(rms_p, 1e-10)

    # Skip near-silent windows
    energy_o = np.array(
        [float(np.sqrt(np.mean(orig[i * hop : i * hop + win_s] ** 2))) for i in range(n_frames)],
        dtype=np.float64,
    )
    active = energy_o > np.percentile(energy_o, 10)
    if np.sum(active) < 3:
        return 1.0

    crest_o = crest_o[active]
    crest_p = crest_p[active]

    # Compare distributions: mean + std of crest factor
    def _dist_stats(c):
        return float(np.mean(c)), float(np.std(c))

    mu_o, std_o = _dist_stats(crest_o)
    mu_p, std_p = _dist_stats(crest_p)

    # Mean shift: center of distribution moved (compression lowers mean crest)
    if mu_o > 0:
        mean_ratio = min(mu_o, mu_p) / max(mu_o, mu_p)
    else:
        mean_ratio = 1.0

    # Std narrowing: distribution compressed (compression reduces std)
    if std_o > 1e-6:
        std_ratio = min(std_o, std_p) / max(std_o, std_p)
    else:
        std_ratio = 1.0

    # Combined: mean shift is more important than std narrowing
    return float(np.clip(0.6 * mean_ratio + 0.4 * std_ratio, 0.0, 1.0))


# ── §G54 Emotional Arc Preservation ────────────────────────────────────


def compute_emotional_arc_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    window_s: float = 3.0,
) -> float:
    """§G54: Emotional Arc preservation via loudness contour + contrast.

    Measures how well the musical narrative (tension/release structure)
    is preserved. Processing that flattens dynamics, removes section
    contrast, or fills silence gaps damages the emotional arc.

    Physikalische Grenzen (Kendall & Carterette, 1996):
      - Loudness contour: short-term LUFS correlation (40% weight)
      - Section contrast: inter-section LUFS ratio preservation (30%)
      - Spectral movement: centroid trajectory correlation (20%)
      - Silence preservation: gap count + duration matching (10%)

    Returns: float [0,1] — 1.0 = identical emotional arc.
    """
    orig = _to_mono(original)
    proc = _to_mono(processed)
    n = min(len(orig), len(proc))
    if n < int(10.0 * sr):
        return 1.0  # Too short for arc analysis
    orig = orig[:n].astype(np.float64)
    proc = proc[:n].astype(np.float64)

    # ── 1. Loudness contour (40%) ──
    win_s = int(window_s * sr)
    hop_s = win_s // 2
    n_frames = (n - win_s) // hop_s + 1
    if n_frames < 5:
        return 1.0

    lufs_o = np.zeros(n_frames, dtype=np.float64)
    lufs_p = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop_s
        fo = orig[s : s + win_s]
        fp = proc[s : s + win_s]
        rms_o = float(np.sqrt(np.mean(fo**2)))
        rms_p = float(np.sqrt(np.mean(fp**2)))
        lufs_o[i] = 20.0 * np.log10(max(rms_o, 1e-15))
        lufs_p[i] = 20.0 * np.log10(max(rms_p, 1e-15))

    # Pearson correlation of loudness contour
    so = float(np.std(lufs_o))
    sp = float(np.std(lufs_p))
    if so < 1e-6 or sp < 1e-6:
        loudness_corr = 1.0
    else:
        loudness_corr = float(
            np.dot(lufs_o - np.mean(lufs_o), lufs_p - np.mean(lufs_p)) / (len(lufs_o) * so * sp + 1e-12)
        )
        loudness_corr = max(0.0, min(1.0, (loudness_corr + 1.0) / 2.0))
    loudness_score = float(loudness_corr)

    # ── 2. Section contrast (30%) ──
    # Divide into ~10-second sections and compare dynamic range
    section_s = int(10.0 * sr)
    n_sections = max(1, n // section_s)
    contrast_ratios = []
    for i in range(n_sections):
        s_start = i * section_s
        s_end = min(s_start + section_s, n)
        if s_end - s_start < win_s:
            continue
        # Dynamic range per section (P95 - P5 of RMS)
        frames_in_section = max(1, (s_end - s_start - win_s) // hop_s + 1)
        rms_vals_o = np.zeros(frames_in_section, dtype=np.float64)
        rms_vals_p = np.zeros(frames_in_section, dtype=np.float64)
        for j in range(frames_in_section):
            s = s_start + j * hop_s
            rms_vals_o[j] = float(np.sqrt(np.mean(orig[s : s + win_s] ** 2)))
            rms_vals_p[j] = float(np.sqrt(np.mean(proc[s : s + win_s] ** 2)))
        if len(rms_vals_o) < 3:
            continue
        dr_o = np.percentile(rms_vals_o, 95) / max(np.percentile(rms_vals_o, 5), 1e-15)  # type: ignore
        dr_p = np.percentile(rms_vals_p, 95) / max(np.percentile(rms_vals_p, 5), 1e-15)  # type: ignore
        ratio = min(dr_o, dr_p) / max(dr_o, dr_p)
        contrast_ratios.append(ratio)
    contrast_score = float(np.mean(contrast_ratios)) if contrast_ratios else 1.0

    # ── 3. Spectral movement (20%) ──
    n_fft = 2048
    spec_hop = n_fft // 2
    n_spec_frames = (n - n_fft) // spec_hop + 1
    if n_spec_frames >= 3:
        win = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        centroids_o = np.zeros(n_spec_frames, dtype=np.float64)
        centroids_p = np.zeros(n_spec_frames, dtype=np.float64)
        for i in range(n_spec_frames):
            s = i * spec_hop
            so_spec = np.abs(np.fft.rfft(orig[s : s + n_fft] * win))
            sp_spec = np.abs(np.fft.rfft(proc[s : s + n_fft] * win))
            centroids_o[i] = float(np.sum(freqs * so_spec) / max(np.sum(so_spec), 1e-10))
            centroids_p[i] = float(np.sum(freqs * sp_spec) / max(np.sum(sp_spec), 1e-10))
        sco = float(np.std(centroids_o))
        scp = float(np.std(centroids_p))
        if sco > 1e-6 and scp > 1e-6:
            centroid_corr = float(
                np.dot(
                    centroids_o - np.mean(centroids_o),
                    centroids_p - np.mean(centroids_p),
                )
                / (len(centroids_o) * sco * scp + 1e-12)
            )
            spectral_score = max(0.0, (centroid_corr + 1.0) / 2.0)
        else:
            spectral_score = 1.0
    else:
        spectral_score = 1.0

    # ── 4. Silence/gap preservation (10%) ──
    silence_thresh = -60.0  # dBFS
    lufs_o_db = 20.0 * np.log10(
        np.array(
            [max(float(np.sqrt(np.mean(orig[i * hop_s : i * hop_s + win_s] ** 2))), 1e-15) for i in range(n_frames)],
            dtype=np.float64,
        )
        + 1e-15
    )
    lufs_p_db = 20.0 * np.log10(
        np.array(
            [max(float(np.sqrt(np.mean(proc[i * hop_s : i * hop_s + win_s] ** 2))), 1e-15) for i in range(n_frames)],
            dtype=np.float64,
        )
        + 1e-15
    )
    gaps_o = int(np.sum(lufs_o_db < silence_thresh))
    gaps_p = int(np.sum(lufs_p_db < silence_thresh))
    if gaps_o == 0 and gaps_p == 0:
        silence_score = 1.0
    elif gaps_o == 0 or gaps_p == 0:
        silence_score = 0.5
    else:
        silence_score = min(gaps_o, gaps_p) / max(gaps_o, gaps_p)

    # ── Aggregate ──
    return float(
        np.clip(
            0.40 * loudness_score + 0.30 * contrast_score + 0.20 * spectral_score + 0.10 * silence_score,
            0.0,
            1.0,
        )
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0) if audio.shape[1] < audio.shape[0] else audio.mean(axis=1)  # type: ignore[no-any-return]
