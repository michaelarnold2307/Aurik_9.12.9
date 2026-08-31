"""§TLS Tape Level Stabilizer — STFT-based envelope repair for tape head contact dips.

Extracted from Phase 12 (Wow/Flutter Fix) for standalone reuse in Phase 24
(Dropout Repair) and other phases that need tape-head-contact-level repair.

Algorithm (Camras 1988; Wallace spacing-loss model):
    1. Compute RMS envelope in short windows
    2. Detect dips: envelope < median - threshold_db
    3. For each dip region: compute compensating STFT spectral gain
    4. Apply HF spectral-tilt correction (Wallace spacing-loss inversion)
    5. Limit max gain to avoid noise amplification
    6. Apply gain with smooth interpolation

Primary entry point:
    stabilize_tape_level(audio, sample_rate, ...) -> (repaired_audio, n_dips)

Usage from Phase 12:
    from backend.core.dsp.tape_level_stabilizer import stabilize_tape_level
    result, n = stabilize_tape_level(mono, sr, material=mat, ...)

Usage from Phase 24:
    from backend.core.dsp.tape_level_stabilizer import stabilize_tape_level
    result, n = stabilize_tape_level(chunk, sr, material=mat, max_gain_db=6.0)
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_DIP_THRESHOLD_DB: float = 8.0  # dB below reference = detected as dip
_MIN_DIP_DURATION_S: float = 0.060  # minimum 60 ms
_MAX_GAIN_DB_BY_MATERIAL: dict[MaterialType, float] = {
    MaterialType.TAPE: 14.0,
    MaterialType.CASSETTE: 14.0,
    MaterialType.REEL_TAPE: 14.0,
    MaterialType.VINYL: 10.0,
    MaterialType.SHELLAC: 8.0,
    MaterialType.WAX_CYLINDER: 6.0,
}


def stabilize_tape_level(
    audio: np.ndarray,
    sample_rate: int,
    *,
    material: MaterialType = MaterialType.TAPE,
    strength: float = 0.80,
    dip_threshold_db: float | None = None,
    max_gain_db: float | None = None,
    protected_zones: list[tuple[float, float, float]] | None = None,
    **_kwargs: Any,
) -> tuple[np.ndarray, int]:
    """Repair tape-head-contact envelope dips using STFT spectral gain.

    Args:
        audio: Mono or stereo audio. Stereo → linked-stereo gain applied.
        sample_rate: Sample rate (expected 48000).
        material: Material type for adaptive max gain.
        strength: Correction strength 0.0–1.0.
        dip_threshold_db: Override default dip threshold.
        max_gain_db: Override material-adaptive max gain.
        protected_zones: List of (start_s, end_s, cap) zones where
                         correction strength is capped.

    Returns:
        (repaired_audio, n_dips_repaired)
    """
    if audio.size == 0 or sample_rate <= 0:
        return audio.copy(), 0

    threshold_db = dip_threshold_db if dip_threshold_db is not None else _DIP_THRESHOLD_DB
    _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
    gain_cap_db = max_gain_db if max_gain_db is not None else _MAX_GAIN_DB_BY_MATERIAL.get(_mk, 10.0)  # type: ignore[call-overload]

    arr = np.asarray(audio, dtype=np.float64)
    is_stereo = arr.ndim == 2 and arr.shape[1] == 2
    mono = np.mean(arr, axis=1) if is_stereo else arr
    n_samples = len(mono)

    # ── Step 1: RMS envelope ──────────────────────────────────────────
    env_win = max(32, int(0.020 * sample_rate))  # 20 ms
    env_hop = max(16, env_win // 2)
    env_rms = np.array(
        [
            float(np.sqrt(np.mean(mono[i : i + env_win].astype(np.float64) ** 2) + 1e-12))
            for i in range(0, max(1, n_samples - env_win), env_hop)
        ]
    )

    if len(env_rms) < 8:
        return audio.copy(), 0

    # ── Step 2: Detect dips ───────────────────────────────────────────
    ref_rms = np.percentile(env_rms, 75)
    if ref_rms < 1e-9:
        return audio.copy(), 0

    env_db = 20.0 * np.log10(env_rms + 1e-12)
    ref_db = 20.0 * np.log10(ref_rms + 1e-12)
    dip_mask = env_db < (ref_db - threshold_db)

    # Merge adjacent dips
    dip_regions: list[tuple[int, int]] = []
    in_dip = False
    dip_start = 0
    for i, is_dip in enumerate(dip_mask):
        if is_dip and not in_dip:
            dip_start = i
            in_dip = True
        elif not is_dip and in_dip:
            # Check minimum duration
            dur_s = (i - dip_start) * env_hop / sample_rate
            if dur_s >= _MIN_DIP_DURATION_S:
                dip_regions.append((dip_start, i))
            in_dip = False
    if in_dip:
        dur_s = (len(dip_mask) - dip_start) * env_hop / sample_rate
        if dur_s >= _MIN_DIP_DURATION_S:
            dip_regions.append((dip_start, len(dip_mask)))

    if not dip_regions:
        return audio.copy(), 0

    # ── Step 3: Build spectral gain mask ──────────────────────────────
    fft_size = 2048
    hop_stft = fft_size // 2
    np.hanning(fft_size)

    # STFT of mono for analysis
    _, _, X_mono = signal.stft(
        mono.astype(np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size - hop_stft,
        boundary="even",
        padded=True,
    )
    n_freqs, n_frames_stft = X_mono.shape
    stft_centres = np.arange(n_frames_stft) * hop_stft + fft_size // 2
    rms_centres = np.arange(len(env_rms)) * env_hop + env_win // 2
    mag_db = 20.0 * np.log10(np.abs(X_mono) + 1e-12)
    spectral_gain = np.ones((n_freqs, n_frames_stft), dtype=np.float64)
    ctx_n = 64

    n_repaired = 0
    for dip_start_idx, dip_end_idx in dip_regions:
        dip_frames = np.arange(dip_start_idx, dip_end_idx)
        if len(dip_frames) == 0:
            continue

        deficit = env_db[dip_frames] - ref_db
        deficit = np.clip(deficit, -gain_cap_db, 0.0)
        max_deficit = float(np.max(np.abs(deficit)))

        # Per-dip strength (capped by protected zones)
        event_strength = _compute_dip_strength(
            mono,
            dip_start_idx * env_hop,
            dip_end_idx * env_hop,
            sample_rate,
            strength,
            max_deficit,
            gain_cap_db,
            protected_zones,
        )

        # Map to STFT frames
        rms_ctrs = rms_centres[dip_frames]
        stft_idx = np.searchsorted(stft_centres, rms_ctrs)
        stft_idx = np.unique(np.clip(stft_idx, 0, n_frames_stft - 1))
        if len(stft_idx) == 0:
            continue

        # Broadband gain
        bb_gain = np.interp(
            stft_centres[stft_idx].astype(float),
            rms_ctrs.astype(float),
            np.clip(np.abs(deficit) * event_strength, 0.0, gain_cap_db),
        )

        # HF tilt correction (Wallace spacing-loss)
        hf_tilt = _compute_hf_tilt(mag_db, stft_idx, ctx_n, event_strength)

        # Asymmetric fade envelope
        n_sf = len(stft_idx)
        onset_n = max(1, int(n_sf * 0.30))
        recovery_n = max(1, int(n_sf * 0.10))
        fade = np.ones(n_sf)
        fade[:onset_n] = np.linspace(0.0, 1.0, onset_n)
        if n_sf > onset_n + recovery_n:
            fade[-recovery_n:] = np.linspace(1.0, 0.0, recovery_n)

        # Combine
        bb_lin = 10.0 ** (bb_gain / 20.0)
        tilt_lin = 10.0 ** (hf_tilt / 20.0)
        max_lin = 10.0 ** (gain_cap_db / 20.0)
        comb = 1.0 + (bb_lin[:, None] * tilt_lin[None, :] - 1.0) * fade[:, None]
        comb = np.clip(comb, 1.0, max_lin)
        spectral_gain[:, stft_idx] = comb.T

        n_repaired += 1

    if n_repaired == 0:
        return audio.copy(), 0

    # ── Step 4: Apply to audio ─────────────────────────────────────────
    def _apply(ch: np.ndarray) -> np.ndarray:
        _, _, X = signal.stft(
            ch.astype(np.float64),
            fs=sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size - hop_stft,
            boundary="even",
            padded=True,
        )
        n_apply = min(X.shape[1], spectral_gain.shape[1])
        X[:, :n_apply] *= spectral_gain[:, :n_apply]
        _, y = signal.istft(
            X,
            fs=sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size - hop_stft,
            boundary="even",
        )
        out = np.zeros(n_samples, dtype=np.float64)
        n_trim = min(len(y), n_samples)
        out[:n_trim] = y[:n_trim]
        return cast(np.ndarray, out)

    if is_stereo:
        L_out = _apply(arr[:, 0])
        R_out = _apply(arr[:, 1])
        result = np.column_stack([L_out, R_out])
    else:
        result = _apply(mono)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result = np.clip(result, -1.0, 1.0).astype(np.float32)

    logger.debug("TapeLevelStabilizer: %d dips repaired, strength=%.2f", n_repaired, strength)
    return result, n_repaired


# ── Helpers ────────────────────────────────────────────────────────────


def _compute_dip_strength(
    mono: np.ndarray,
    event_start: int,
    event_end: int,
    sample_rate: int,
    base_strength: float,
    max_deficit_db: float,
    max_gain_db: float,
    protected_zones: list[tuple[float, float, float]] | None,
) -> float:
    """Compute per-dip correction strength."""
    n = len(mono)
    event_start = int(np.clip(event_start, 0, max(0, n - 1)))
    event_end = int(np.clip(event_end, event_start + 1, n))

    depth_factor = float(np.clip(max_deficit_db / max(1.0, max_gain_db - 3.0), 0.0, 1.0))
    duration_s = (event_end - event_start) / float(max(sample_rate, 1))
    duration_factor = float(np.clip((duration_s - 0.030) / 0.270, 0.0, 1.0))

    local_s = max(0.58, base_strength) + 0.26 * depth_factor + 0.08 * duration_factor
    local_s = float(np.clip(local_s, 0.0, 0.94))

    if protected_zones:
        center_s = float(event_start + event_end) * 0.5 / float(max(sample_rate, 1))
        for zone_start, zone_end, zone_cap in protected_zones:
            if float(zone_start) <= center_s <= float(zone_end):
                local_s = min(local_s, float(zone_cap))
                break

    return float(np.clip(local_s, 0.0, 1.0))


def _compute_hf_tilt(
    mag_db: np.ndarray,
    stft_idx: np.ndarray,
    ctx_n: int,
    event_strength: float,
) -> np.ndarray:
    """Compute HF spectral-tilt correction per frequency bin."""
    n_freqs = mag_db.shape[0]
    # Use a fixed frequency scale (0–24 kHz for 48 kHz sr, n_fft=2048)
    freqs_hz = np.linspace(0, 24000, n_freqs)
    hf_tilt = np.zeros(n_freqs)

    first_stft = int(stft_idx[0])
    ctx_start = max(0, first_stft - ctx_n)
    ctx_end = max(0, first_stft - 2)
    if ctx_end - ctx_start < 4:
        return cast(np.ndarray, hf_tilt)

    ctx_mag = mag_db[:, ctx_start:ctx_end]
    dip_mag = mag_db[:, stft_idx]
    ref_spec = np.percentile(ctx_mag, 75, axis=1)
    dip_spec = np.mean(dip_mag, axis=1)
    spectral_loss = ref_spec - dip_spec

    lf_mask = freqs_hz < 4000.0
    if lf_mask.any():
        broadband_loss = float(np.median(spectral_loss[lf_mask]))
    else:
        broadband_loss = float(np.median(spectral_loss))

    tilt_raw = spectral_loss - broadband_loss
    noise_floor = np.percentile(ctx_mag, 10, axis=1)
    snr_in_dip = dip_spec - noise_floor
    tilt_raw = np.where(snr_in_dip > 6.0, tilt_raw, 0.0)
    hf_tilt = np.clip(tilt_raw, 0.0, 10.0) * event_strength
    return cast(np.ndarray, hf_tilt)
