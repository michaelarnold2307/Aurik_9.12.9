"""
Denoise Algorithms — Aurik 10.0.0
==================================

Core denoising algorithms extracted from phase_03_denoise.py:
- IMCRA noise estimation (Cohen & Berdugo 2002)
- OMLSA gain function (Cohen 2003)
- Salience-adaptive G_floor computation
- Adaptive guard profile calculation
- Gain-gradient phase correction (Prusa & Holighaus 2017)
- ERB-rate band mapping (Glasberg & Moore 1990)
- Multi-band gate, masking gate, musical noise suppression
- Transient preservation

Diese Funktionen sind stateless und benötigen keine Phase-Instanz.
Sie werden von phase_03_denoise.py importiert.

Author: Aurik 10.0.0 Development Team
Version: 2.0.0 (Professional Upgrade)
Date: 15. Februar 2026
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)

# ── Constants extracted from DenoisePhase class ───────────────────────────────

BAND_BOUNDARIES: dict[str, tuple[int, int]] = {
    "low": (20, 500),  # Bass/Low-Mid
    "mid": (500, 5000),  # Midrange
    "high": (5000, 20000),  # High frequencies (hiss region)
}

# MRSA Multi-Resolution Spectral Analysis zones (mandatory, §DSP-Spezialregeln)
MRSA_ZONES: tuple = (
    # (name,       win_size, hop_size, f_low_hz, f_high_hz)
    ("sub_bass", 65536, 16384, 0, 250),
    ("mid_low", 16384, 4096, 250, 2500),
    ("mid", 8192, 2048, 2500, 8000),
    ("presence", 1024, 256, 8000, 16000),
    ("air", 128, 32, 16000, 24000),
)

MRSA_CROSSFADE_BW_HZ: float = 100.0


# ── Static algorithm functions (originally staticmethod on DenoisePhase) ──────


def compute_salience_g_floor(
    audio: np.ndarray,
    sr: int,
    g_floor_base: float,
    n_t: int,
    hop: int,
) -> np.ndarray:
    """Berechnet a time-varying G_floor curve based on momentary loudness.

    Scientific basis:
        Moore (2003) "Psychology of Hearing" §9: the simultaneous masking
        threshold is loudness-relative.  In loud passages the residual noise
        after NR is inaudible → we can afford a lower G_floor (more aggressive
        noise removal).  In quiet/exposed passages the signal is fragile and
        OMLSA may mis-classify signal bins as noise → higher G_floor protects
        musical content (e.g. pianissimo transitions before a chorus).

    Mapping (linear interpolation):
        LUFS > -12 dBFS  (loud):   G_floor = 0.50 × g_floor_base  (aggressive)
        LUFS < -30 dBFS  (quiet):  G_floor = min(3.0 × base, 0.40) (conservative)

    A 500 ms smoothing kernel prevents pumping artefacts at loudness transitions.

    Args:
        audio:        Mono float32 audio at native SR (48 kHz in processing path).
        sr:           Sample rate.
        g_floor_base: Material-specific scalar G_floor (from params).
        n_t:          Number of STFT frames to produce.
        hop:          STFT hop size in samples (reference grid).

    Returns:
        g_floor_vec: np.ndarray shape (n_t,), dtype float32.
    """
    WIN_S = 0.4  # ITU-R BS.1770-5 momentary loudness window (400 ms)
    HOP_S = 0.1  # 100 ms hop
    win_n = max(1, int(WIN_S * sr))
    hop_n = max(1, int(HOP_S * sr))
    n = audio.shape[-1] if audio.ndim > 1 else len(audio)
    mono = audio[0] if audio.ndim == 2 else audio  # channel-first safe
    mono = np.asarray(mono, dtype=np.float64)

    n_lufs_frames = max(1, (n - win_n) // hop_n + 1)
    lufs_db = np.full(n_lufs_frames, -60.0, dtype=np.float32)
    for i in range(n_lufs_frames):
        start = i * hop_n
        frame = mono[start : start + win_n]
        rms = float(np.sqrt(np.mean(frame**2) + 1e-20))
        lufs_db[i] = float(np.clip(20.0 * np.log10(rms + 1e-10), -80.0, 0.0))

    # G_floor bounds: loud → aggressive (0.5×), quiet → conservative (3× capped at 0.40)
    g_lo = float(np.clip(0.50 * g_floor_base, 0.03, 0.10))
    g_hi = float(np.clip(3.0 * g_floor_base, g_floor_base + 1e-6, 0.40))
    # np.interp: x < xp[0] → fp[0], x > xp[-1] → fp[-1] (automatic clamping)
    g_floor_lufs = np.interp(lufs_db.astype(np.float64), [-30.0, -12.0], [g_hi, g_lo]).astype(np.float32)

    # 500 ms smoothing to avoid pumping at loudness transitions.
    smooth_frames = max(3, round(0.5 / max(HOP_S, 1e-6)))
    kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
    g_floor_smooth = np.convolve(g_floor_lufs, kernel, mode="same")[:n_lufs_frames]

    # Interpolate from LUFS time grid to STFT time grid
    t_lufs = np.arange(n_lufs_frames, dtype=np.float32) * HOP_S
    t_stft = np.arange(n_t, dtype=np.float32) * (hop / float(sr))
    g_floor_vec = np.interp(t_stft, t_lufs, g_floor_smooth).astype(np.float32)
    # Clamp to [g_lo, g_hi] as defensive guard against convolution edge artefacts.
    g_floor_vec = np.clip(g_floor_vec, g_lo, g_hi).astype(np.float32)
    return np.nan_to_num(g_floor_vec, nan=float(g_floor_base))  # type: ignore[no-any-return]


def compute_adaptive_guard_profile(
    material_type: str,
    quality_mode: str,
    restorability_score: float,
) -> dict[str, float]:
    """Berechnet adaptive denoise guard targets from song context.

    Returns thresholds for quality warnings and minimum/target energy preservation.
    """
    _mat = str(material_type or "unknown").lower().replace("-", "_").replace(" ", "_")
    _qm = str(quality_mode or "balanced").lower().replace("-", "_")
    _rest = float(np.clip(restorability_score, 0.0, 100.0))

    _digital_mats = {"cd_digital", "digital", "dat", "streaming", "aac", "mp3_high"}
    _is_digital = _mat in _digital_mats

    _base_quality_warn = 0.76 if _is_digital else 0.70
    _base_energy_min = 0.24 if _is_digital else 0.20

    _mode_quality_adj = {
        "fast": 0.00,
        "balanced": 0.01,
        "quality": 0.03,
        "maximum": 0.05,
        "restoration": 0.03,
        "studio_2026": 0.05,
    }.get(_qm, 0.01)
    _mode_energy_adj = {
        "fast": 0.05,
        "balanced": 0.02,
        "quality": 0.00,
        "maximum": -0.01,
        "restoration": 0.00,
        "studio_2026": -0.01,
    }.get(_qm, 0.02)

    _rest_quality_adj = ((_rest - 50.0) / 50.0) * 0.08
    _rest_energy_adj = ((_rest - 50.0) / 50.0) * 0.04

    quality_warning_threshold = float(
        np.clip(_base_quality_warn + _mode_quality_adj + _rest_quality_adj, 0.55, 0.85)
    )
    energy_min_ratio = float(np.clip(_base_energy_min + _mode_energy_adj + _rest_energy_adj, 0.14, 0.32))
    _target_margin = 0.06 if _qm in {"quality", "maximum", "restoration", "studio_2026"} else 0.04
    energy_target_ratio = float(np.clip(energy_min_ratio + _target_margin, 0.20, 0.45))

    if energy_target_ratio < energy_min_ratio + 0.02:
        energy_target_ratio = float(np.clip(energy_min_ratio + 0.02, 0.20, 0.45))

    return {
        "quality_warning_threshold": quality_warning_threshold,
        "energy_min_ratio": energy_min_ratio,
        "energy_target_ratio": energy_target_ratio,
    }


def apply_gain_gradient_phase_correction(
    Zxx_ref: np.ndarray,
    G_combined: np.ndarray,
    hop: int,
    sr: int,
) -> np.ndarray:
    """Gain-gradient phase correction before PGHI/iSTFT (Prusa & Holighaus 2017, §3.4).

    Time-varying gain G(k,t) introduces an instantaneous-frequency (IF) error of
    ∂log(G)/∂t per STFT frame.  PGHI estimates IF from the log-magnitude gradient and
    therefore inherits this artefact as phase chirps on transient attacks and gain-ramp
    edges, degrading TimbralAuthenticityMetric and SpatialDepthMetric.

    Correction:
        Δφ(k,t) = -(hop/sr) × cumsum_t( ∂log G(k,t)/∂t )

    The corrected STFT is a better PGHI initialisation and provides phase-correct
    reconstruction in the iSTFT fallback path.

    Scientific reference:
        Prusa & Holighaus (2017) "Phase-Vocoder Done Right", §3.4 "Enhancement".

    Args:
        Zxx_ref:    Reference STFT (n_bins × n_t), complex.
        G_combined: MRSA gain matrix (n_bins × n_t), float32, ∈ [0, 1].
        hop:        STFT hop size in samples.
        sr:         Processing sample rate (48 000 Hz).

    Returns:
        Zxx_corrected: complex64 STFT with gain applied and phase corrected.
    """
    log_G = np.log(np.maximum(G_combined.astype(np.float64), 1e-8))  # (n_bins, n_t)
    # ∂log(G)/∂t — forward difference; prepend first col to preserve shape
    dlogG_dt = np.diff(log_G, axis=1, prepend=log_G[:, :1])  # (n_bins, n_t)
    # Cumulative phase offset: Δφ = -(hop/sr) × ∫ ∂logG/∂τ dτ
    delta_phi = -np.cumsum(dlogG_dt, axis=1) * (hop / float(sr))  # (n_bins, n_t)
    mag_out = G_combined.astype(np.float64) * np.abs(Zxx_ref)
    phase_out = np.angle(Zxx_ref) + delta_phi
    Zxx_corrected = mag_out * np.exp(1j * phase_out)
    _result = np.nan_to_num(Zxx_corrected, nan=0.0, posinf=0.0, neginf=0.0)
    return _result.astype(np.complex64)  # type: ignore[no-any-return]


def compute_erb_bands(n_bins: int, sr: int) -> np.ndarray:
    """Map STFT frequency bins to ERB-rate band indices (Glasberg & Moore 1990).

    ERB-rate: E(f) = 21.4 × log10(4.37 × f/1000 + 1) [Cams].
    38 uniformly-spaced bands from 100 Hz to sr/2 give perceptually uniform
    coverage.  Multiple linear STFT bins that fall within one ERB band are
    auditorily unresolvable; pooling their minimum statistics prevents a
    single isolated low-energy bin from driving over-suppression of the
    entire fricative range (/s/, /f/, /ʃ/ at 4–8 kHz).

    Args:
        n_bins: Number of STFT frequency bins (n_fft//2 + 1).
        sr:     Sample rate (Hz).

    Returns:
        band_idx: np.ndarray shape (n_bins,) dtype int32 — ERB band per bin,
                  values ∈ [0, 37].
    """
    freqs = np.linspace(0.0, float(sr) / 2.0, n_bins, endpoint=True)

    def _hz_to_cam(f: np.ndarray) -> np.ndarray:
        return 21.4 * np.log10(4.37 * np.maximum(f, 1.0) / 1000.0 + 1.0)  # type: ignore[no-any-return]

    N_ERB = 38
    e_min = float(_hz_to_cam(np.array([100.0]))[0])
    e_max = float(_hz_to_cam(np.array([float(sr) / 2.0]))[0])
    erb_edges = np.linspace(e_min, e_max, N_ERB + 1)
    band_idx = np.clip(
        np.searchsorted(erb_edges[1:], _hz_to_cam(freqs)),  # type: ignore[arg-type]
        0,
        N_ERB - 1,
    ).astype(np.int32)  # Index-Array (ERB-Band), kein Audio-Dither nötig (§V5)
    return band_idx  # type: ignore[no-any-return]


def apply_masking_gate(gain: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Musical-noise post-filter via psychoacoustic simultaneous masking (§D).

    Musical noise = isolated high-gain STFT bins whose output power is below
    the simultaneous masking threshold set by their spectral neighbours.  The
    auditory system cannot separately resolve such isolated tones, yet they
    produce clearly audible chirping artefacts (Cappé 1994).

    Gate formula (Gustafsson et al. 2001, adapted; Scalart & Filho 1996):
        E_out(k,t)  = (G(k,t) × |Y(k,t)|)²
        M(t)        = α × P₂₄(E_out(:,t))   [α = 10^(−16/10) ≈ 0.025]
        gate(k,t)   = √( min(1, E_out(k,t) / M(t)) )
        G_out(k,t)  = clip( G(k,t) × gate(k,t), 0.1, 1.0 )

    Args:
        gain:      OMLSA gain matrix (n_freq × n_t), float, ∈ [0, 1].
        magnitude: |STFT| at zone resolution, same shape as gain.

    Returns:
        G_out: gain matrix, dtype float32, shape (n_freq × n_t), ∈ [0.1, 1].
    """
    output_power = (gain.astype(np.float64) * magnitude.astype(np.float64)) ** 2

    # 75th-percentile per frame: robust dominant spectral level.
    frame_p75 = np.percentile(output_power, 75, axis=0, keepdims=True) + 1e-20

    # α = 10^(-16/10) ≈ 0.025  — simultaneous masking offset (Fastl & Zwicker 2007 §4.2)
    ALPHA = 10.0 ** (-16.0 / 10.0)
    masking_threshold = ALPHA * frame_p75  # broadcast (1, n_t) → (n_freq, n_t)

    # Soft-knee gate: √(min(1, E_out / M)) preserves loud bins, attenuates chirps.
    gate = np.sqrt(np.minimum(1.0, output_power / (masking_threshold + 1e-20)))
    gate = np.clip(gate, 0.1, 1.0)  # floor -20 dB, never mute

    return np.clip(gain.astype(np.float64) * gate, 0.1, 1.0).astype(np.float32)  # type: ignore[no-any-return]


# ── Instance algorithm functions (originally methods on DenoisePhase) ─────────


def estimate_noise_imcra(
    magnitude: np.ndarray,
    times: np.ndarray,
    onset_frames: "np.ndarray | None" = None,
    sr: int = 48_000,
) -> np.ndarray:
    """IMCRA Noise PSD Estimation with ERB-rate grouping + adaptive smoothing.

    Cohen & Berdugo (2002): "Noise Estimation by Minima Controlled
    Recursive Averaging" (IMCRA).

    Algorithmus:
        - Gleitendes Minimum über M Frames (≈1.5 s)
        - Bias-Korrektur: b_min = 1.66 (Gauß'sches Rauschen)
        - ERB-rate Grouping: Glasberg & Moore (1990) — Verbesserung B
        - Exponentielle Glättung: α_n adaptiv (Loizou 2013, §7.3)

    Args:
        magnitude:    |STFT| (F×T)
        times:        STFT-Zeitachse
        onset_frames: Optional 1-D array of frame indices for detected onsets.
                      If None, auto-detected from positive spectral flux.

    Returns:
        noise_mag: Rausch-Amplitude (F×T), immer positiv
    """
    n_freq, n_frames = magnitude.shape
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.01
    M = max(3, int(1.5 / (dt + 1e-12)))  # Fensterbreite ≈ 1.5 s

    pow_spec = magnitude**2  # Leistungsspektrum

    # Minimum-Statistik pro Frequenzband
    sigma2 = np.zeros_like(pow_spec)
    window_buf = np.full((n_freq, M), np.inf)
    buf_ptr = 0

    for t in range(n_frames):
        window_buf[:, buf_ptr % M] = pow_spec[:, t]
        buf_ptr += 1
        valid = min(t + 1, M)
        local_min = np.min(window_buf[:, :valid], axis=1)
        sigma2[:, t] = local_min

    # Bias-Korrektur (IMCRA: b_min ≈ 1.66 für stationäres Gaußrauschen)
    b_min = 1.66
    sigma2 *= b_min

    # §B ERB-rate grouping: pool minimum statistics within auditory critical bands.
    if n_freq > 1 and sr > 0:
        erb_idx = compute_erb_bands(n_freq, sr)
        n_erb = int(erb_idx.max()) + 1
        sigma2_grouped = np.empty_like(sigma2)
        for b in range(n_erb):
            mask = erb_idx == b
            if np.any(mask):
                sigma2_grouped[mask, :] = np.mean(sigma2[mask, :], axis=0)
        sigma2 = sigma2_grouped

    # Stationarity-adaptive α: fast tracking at onsets, slow elsewhere.
    ALPHA_STAT = 0.85  # standard stationary noise tracking
    ALPHA_ONSET = 0.50  # fast update: transient onsets need fresh estimate
    ONSET_RADIUS = 2  # frames around each onset to apply fast α

    if onset_frames is None:
        energy = np.sum(pow_spec, axis=0)  # (n_frames,)
        flux = np.maximum(0.0, np.diff(energy, prepend=energy[:1]))  # positive only
        threshold = np.percentile(flux, 88) if n_frames > 10 else float(np.max(flux))
        onset_frames = np.where(flux > threshold)[0]

    alpha_t = np.full(n_frames, ALPHA_STAT, dtype=np.float64)
    for of in onset_frames:
        lo = max(0, int(of) - ONSET_RADIUS)
        hi = min(n_frames, int(of) + ONSET_RADIUS + 1)
        alpha_t[lo:hi] = ALPHA_ONSET

    # Exponentielle Glättung über die Zeit — per-frame alpha
    smoothed = np.zeros_like(sigma2)
    smoothed[:, 0] = sigma2[:, 0]
    for t in range(1, n_frames):
        a = alpha_t[t]
        smoothed[:, t] = a * smoothed[:, t - 1] + (1 - a) * sigma2[:, t]

    noise_mag = np.sqrt(np.maximum(smoothed, 1e-10))
    return np.nan_to_num(noise_mag, nan=1e-6, posinf=1.0, neginf=1e-6)  # type: ignore[no-any-return]


def compute_omlsa_gain(
    magnitude: np.ndarray,
    noise_mag: np.ndarray,
    params: dict[str, Any],
    g_floor_vec: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """OMLSA Gain Function (Cohen 2003).

    Cohen (2003): "Noise Spectrum Estimation in Adverse Environments:
    Improved Minima Controlled Recursive Averaging" (OMLSA).

    Formeln:
        γ(t,f) = |Y|² / σ²_n          (a-posteriori SNR)
        ξ(t,f) = max(γ − 1, 0)        (a-priori SNR, Decision-Directed-Approx.)
        Λ(t,f) = 1/(1+ξ) · exp(ξγ/(1+ξ))  (Likelihood-Ratio)
        p(t,f) = 1 / (1 + q/(1−q) / Λ)  (Präsenzwahrscheinlichkeit)
        G(t,f) = G_floor(t)^(1−p) · (ξ/(1+ξ))^p
        G(t,f) ∈ [G_floor(t), 1.0]

    Args:
        magnitude: |STFT| (F×T)
        noise_mag: Rausch-Amplitude (F×T)
        params: Enthält 'strength' (0..1) und optionales 'g_floor'
        g_floor_vec: Optional time-varying G_floor curve shape (n_t,).

    Returns:
        (G_omlsa, p_speech): Gain-Matrix und Signal-Präsenz-Wahrsch. (je F×T)
    """
    # G_FLOOR_BASE: material-spezifisch überschreibbar (z.B. shellac g_floor=0.30
    # verhindert Signal-Vernichtung bei SNR ≈ 6 dB — Pflicht-Invariante ≥0.10)
    G_FLOOR_BASE = float(params.get("g_floor", 0.1))  # Standard: −20 dB
    Q_NOISE = 0.5  # A-priori Wahrsch. für Rausch-only Frame
    STRENGTH = float(params.get("strength", 0.7))

    if (
        g_floor_vec is not None
        and isinstance(g_floor_vec, np.ndarray)
        and g_floor_vec.ndim == 1
        and g_floor_vec.shape[0] == magnitude.shape[1]
    ):
        G_FLOOR: np.ndarray | float = g_floor_vec[np.newaxis, :].astype(np.float64)  # (1, n_t)
    else:
        G_FLOOR = G_FLOOR_BASE  # scalar fallback

    sigma_n2 = noise_mag**2 + 1e-10
    Y2 = magnitude**2

    # A-posteriori SNR γ
    gamma = Y2 / sigma_n2

    # A-priori SNR ξ (einfache ML-Schätzung als robuster Startpunkt)
    xi = np.maximum(gamma - 1.0, 0.0)
    xi = np.maximum(xi, 1e-8)

    # v = ξγ/(1+ξ)  (MMSE-LSA Variable)
    v = xi * gamma / (1.0 + xi)
    v = np.clip(v, 0.0, 500.0)  # exp-Schranke

    # Likelihood-Ratio Λ = 1/(1+ξ) · exp(v)
    log_lambda = -np.log1p(xi) + v
    log_lambda = np.clip(log_lambda, -50.0, 50.0)
    Lambda = np.exp(log_lambda)
    Lambda = np.nan_to_num(Lambda, nan=1.0, posinf=1e6)

    # Signal-Präsenzwahrscheinlichkeit p(speech | Y)
    q_ratio = Q_NOISE / (1.0 - Q_NOISE)  # = 1.0 für Q_NOISE=0.5
    p_speech = 1.0 / (1.0 + q_ratio / (Lambda + 1e-10))
    p_speech = np.clip(p_speech, 0.0, 1.0)
    p_speech = np.nan_to_num(p_speech, nan=0.5)

    # Wiener Gain G_H1 = ξ/(1+ξ) (unter Signal-Präsenz H1)
    G_H1 = xi / (1.0 + xi)
    G_H1 = np.clip(G_H1, G_FLOOR, 1.0)

    # OMLSA: G = G_floor^(1-p) · G_H1^p
    log_G = (1.0 - p_speech) * np.log(G_FLOOR + 1e-10) + p_speech * np.log(G_H1 + 1e-10)
    G_omlsa = np.exp(np.clip(log_G, -20.0, 0.0))

    # Stärke skalieren (Nutzerpräferenz)
    G_omlsa = G_FLOOR + (G_omlsa - G_FLOOR) * STRENGTH
    G_omlsa = np.clip(G_omlsa, G_FLOOR, 1.0)
    G_omlsa = np.nan_to_num(G_omlsa, nan=G_FLOOR_BASE)  # nan= requires scalar

    # §2.28 HPG: Harmonic Preservation Guard — bin-genaue Oberton-Schutz-Maske.
    _hpg_mask = params.get("_hpg_protected_mask")
    if _hpg_mask is not None and isinstance(_hpg_mask, np.ndarray):
        try:
            _hpg_floor = 0.85  # §2.28 G_FLOOR_HARMONIC
            if _hpg_mask.shape[0] == G_omlsa.shape[0] and _hpg_mask.shape[1] == G_omlsa.shape[1]:
                _mask_aligned = _hpg_mask.astype(np.float64)
            elif _hpg_mask.ndim == 2:
                _n_orig = _hpg_mask.shape[1]
                _n_targ = G_omlsa.shape[1]
                _t_orig = np.linspace(0, 1, _n_orig)
                _t_targ = np.linspace(0, 1, _n_targ)
                _mask_aligned = np.zeros((G_omlsa.shape[0], _n_targ), dtype=np.float64)
                for _f in range(min(_hpg_mask.shape[0], G_omlsa.shape[0])):
                    _mask_aligned[_f, :] = np.interp(
                        _t_targ, _t_orig, _hpg_mask[_f, :].astype(np.float64)
                    )
            else:
                _mask_aligned = None
            if _mask_aligned is not None:
                _hpg_gain = np.where(_mask_aligned > 0.5, _hpg_floor, G_omlsa)
                _edge_kernel = np.array([0.25, 0.50, 0.75], dtype=np.float64)
                _edge_weight = np.zeros_like(_mask_aligned, dtype=np.float64)
                for _k in range(3):
                    _shifted = np.roll(_mask_aligned, _k - 1, axis=0)
                    _edge_weight += _edge_kernel[_k] * (_mask_aligned != _shifted).astype(np.float64)
                _edge_weight = np.clip(_edge_weight, 0.0, 0.5)
                _blend = np.clip(_mask_aligned + _edge_weight, 0.0, 1.0)
                G_omlsa = _blend * _hpg_gain + (1.0 - _blend) * G_omlsa
                logger.debug(
                    "§2.28 HPG: OMLSA gain protected — protected_bins=%.1f%%, μ_G=%.3f (w/ HPG) vs %.3f (raw)",
                    100.0 * float(np.mean(_mask_aligned > 0.5)),
                    float(np.mean(G_omlsa)),
                    float(np.mean(_hpg_gain)),
                )
        except Exception as _hpg_gain_exc:
            logger.debug("§2.28 HPG gain integration: nicht blockierend — %s", _hpg_gain_exc)

    logger.debug(
        "OMLSA: μ_G=%.3f σ_G=%.3f μ_p=%.3f (salience_adaptive=%s)",
        float(np.mean(G_omlsa)),
        float(np.std(G_omlsa)),
        float(np.mean(p_speech)),
        g_floor_vec is not None,
    )
    return G_omlsa, p_speech


def suppress_musical_noise(
    gain: np.ndarray, suppression_strength: float, smoothing_time: int, smoothing_freq: int
) -> np.ndarray:
    """Suppress musical noise via spectral smoothing (Cappé 1994).

    Cappé (1994): "Elimination of the Musical Noise Phenomenon with the
    Ephraim and Malah Noise Suppressor" — zeitliche und Frequenz-Glättung
    des OMLSA-Gains verhindert isolierte Gain-Spitzen (musical noise).

    Args:
        gain:               OMLSA gain matrix (n_freq × n_t)
        suppression_strength: Blend factor [0, 1] between original and smoothed
        smoothing_time:     Number of frames for temporal smoothing kernel
        smoothing_freq:     Number of bins for frequency smoothing kernel

    Returns:
        Smoothed gain
    """
    gain_smoothed = gain.copy()

    # Time smoothing (moving average over frames)
    if smoothing_time > 0:
        kernel_time = np.ones(smoothing_time) / smoothing_time
        for i in range(gain.shape[0]):
            gain_smoothed[i, :] = np.convolve(gain[i, :], kernel_time, mode="same")

    # Frequency smoothing (moving average over bins)
    if smoothing_freq > 0:
        kernel_freq = np.ones(smoothing_freq) / smoothing_freq
        for j in range(gain.shape[1]):
            gain_smoothed[:, j] = np.convolve(gain_smoothed[:, j], kernel_freq, mode="same")

    # Blend original and smoothed (based on suppression strength)
    gain_final = (1 - suppression_strength) * gain + suppression_strength * gain_smoothed

    # Gain floor (minimum reduction)
    gain_floor = 0.1  # Never reduce more than -20 dB
    gain_final = np.maximum(gain_final, gain_floor)

    return gain_final  # type: ignore[no-any-return]


def preserve_transients(
    magnitude: np.ndarray, gain: np.ndarray, preserve_strength: float
) -> np.ndarray:
    """Preserve transients by detecting attacks and reducing gain.

    Args:
        magnitude:          |STFT| (n_freq × n_t)
        gain:               OMLSA gain matrix (same shape)
        preserve_strength:  Blend factor [0, 1] toward full gain at transient positions

    Returns:
        Modified gain (less reduction on transients)
    """
    # Detect transients via temporal derivative
    magnitude_diff = np.diff(magnitude, axis=1, prepend=magnitude[:, [0]])

    # Normalize per frequency bin
    transient_score = np.abs(magnitude_diff) / (magnitude + 1e-10)

    # High score = transient detected
    transient_mask = transient_score > 0.5  # Threshold for transient detection

    gain_modified = gain.copy()
    gain_modified[transient_mask] = (1 - preserve_strength) * gain[transient_mask] + preserve_strength * 1.0

    return gain_modified


def apply_multiband_gate(
    gain: np.ndarray, freqs: np.ndarray, band_params: dict[str, dict[str, float]]
) -> np.ndarray:
    """Apply frequency-dependent gain modifications per band.

    Args:
        gain:       OMLSA gain matrix (n_freq × n_t)
        freqs:      Frequency axis (n_freq,)
        band_params: Band-specific reduction factors

    Returns:
        Modified gain (same shape as input)
    """
    gain_modified = gain.copy()

    for band_name, (f_low, f_high) in BAND_BOUNDARIES.items():
        # Find frequency bins in this band
        mask = (freqs >= f_low) & (freqs <= f_high)

        if band_name in band_params:
            # Get band-specific reduction factor
            reduction = band_params[band_name]["reduction"]

            # Scale gain in this band
            gain_modified[mask, :] *= reduction

    return gain_modified


def estimate_noise_profile_adaptive(
    Zxx: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    noise_start: float,
    noise_end: float,
) -> np.ndarray:
    """Statische Rauschprofil-Schätzung aus nutzer-definiertem Segment.

    Wird nur aufgerufen wenn noise_start/noise_end gesetzt sind.
    Gibt ein 1D Profil (F,) zurück — wird in _denoise_mono_professional
    auf (F,T) aufgeblasen.

    Args:
        Zxx: Komplexes STFT (F×T)
        freqs: Frequenzachse
        times: Zeitachse
        noise_start: Rauschbereich-Start (s)
        noise_end:   Rauschbereich-Ende (s)

    Returns:
        noise_profile: (F,) Rausch-Amplitude
    """
    magnitude = np.abs(Zxx)
    t_max = float(times[-1]) if len(times) > 0 else 1.0
    start_frame = int(noise_start * magnitude.shape[1] / (t_max + 1e-10))
    end_frame = int(noise_end * magnitude.shape[1] / (t_max + 1e-10))
    start_frame = max(0, min(start_frame, magnitude.shape[1] - 1))
    end_frame = max(start_frame + 1, min(end_frame, magnitude.shape[1]))
    noise_frames = magnitude[:, start_frame:end_frame]
    noise_profile = np.median(noise_frames, axis=1)
    return np.nan_to_num(noise_profile, nan=1e-6)  # type: ignore[no-any-return]
