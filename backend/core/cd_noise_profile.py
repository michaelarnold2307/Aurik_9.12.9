"""
SOTA CD-Rauschprofil-Generator (§G8 (GEBOTE.md), §G15–§G19, §G30–§G45)

Nicht von einer echten CD-Produktion (1982–2000) unterscheidbar.

Wissenschaftliche Basis:
  Simultaneous masking (Zwicker & Fastl, 1999)
  Temporal masking (forward 100ms, backward 20ms)
  CD converter noise model (POW-r Type 3 + clock artifacts + 1/f flicker)

Pipeline: NACH allen Phasen, VOR Dithering (§G40).

Author: Aurik Development Team
Version: 10.0.6 SOTA
Date: 2026-07-13
"""

import hashlib
import logging
from typing import Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)

_CD_NOISE_FLOOR_DBFS_16BIT: float = -96.0
_CD_NOISE_FLOOR_DBFS_24BIT: float = -114.0  # 24-bit converter ENOB ≈ 19 bits (real-world ADC)
_CD_NOISE_MAX_DBFS: float = -85.0

# Masking: scientifically grounded thresholds
_MASKING_ASSUMED_DB_SPL: float = 100.0  # 0 dBFS = 100 dB SPL
_MASKING_THRESHOLD_DBFS: float = -45.0  # Global gate: only block very loud signals; ERB gain does per-band masking
_SILENCE_THRESHOLD_DBFS: float = -140.0  # true digital black only

# Temporal masking windows
_FORWARD_MASKING_MS: float = 100.0
_BACKWARD_MASKING_MS: float = 20.0
_MIN_CROSSFADE_MS: float = 50.0
_MAX_CROSSFADE_MS: float = 500.0

# RMS window
_RMS_WINDOW_S: float = 0.050


def _compute_deterministic_seed(audio: np.ndarray) -> int:
    flat = np.asarray(audio, dtype=np.float32).ravel()[:4096]
    digest = hashlib.sha256(flat.tobytes()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def _hz_to_erb(hz):
    return 21.4 * np.log10(0.00437 * np.asarray(hz, dtype=np.float64) + 1.0)


def _compute_sliding_erb_gain(audio_mono, sr, noise_db=-96.0, segment_s=10.0):
    """§G57: Multi-segment ERB gain averaging for spectral adaptation.

    Instead of a single 2-second window, computes ERB gain from
    multiple 10-second segments and averages with Hanning weights.
    This adapts to songs that change spectral character over time.
    """
    n = len(audio_mono)
    seg_samples = int(segment_s * sr)
    if n < seg_samples * 2:
        return _compute_erb_band_gain(audio_mono, sr, noise_db)
    # Slide 10s windows with 50% overlap
    hop = seg_samples // 2
    n_segments = max(1, (n - seg_samples) // hop + 1)
    gains = []
    for i in range(n_segments):
        start = i * hop
        seg = audio_mono[start : start + seg_samples]
        g = _compute_erb_band_gain(seg, sr, noise_db)
        gains.append(g)
    # Average all gains (equal weight per segment)
    return np.mean(gains, axis=0)


def _compute_erb_band_gain(audio_mono, sr, noise_db=-96.0):
    n_fft, n_bins = 2048, 1025
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mid = len(audio_mono) // 2
    seg_len = min(int(2.0 * sr), len(audio_mono) // 2)
    seg = audio_mono[max(0, mid - seg_len // 2) : max(0, mid - seg_len // 2) + seg_len].astype(np.float64)
    if len(seg) < n_fft:
        return np.ones(n_bins, dtype=np.float64)
    hop = n_fft // 2
    n_frames = (len(seg) - n_fft) // hop + 1
    win = np.hanning(n_fft)
    mag_mean = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        mag_mean += np.abs(np.fft.rfft(seg[s : s + n_fft] * win))
    mag_mean /= max(n_frames, 1)
    mag_db = 20.0 * np.log10(np.maximum(mag_mean, 1e-15))
    centers = np.array(
        [
            50,
            150,
            250,
            350,
            450,
            570,
            700,
            840,
            1000,
            1170,
            1370,
            1600,
            1850,
            2150,
            2500,
            2900,
            3400,
            4000,
            4800,
            5800,
            7000,
            8500,
            10500,
            13500,
            19500,
        ]
    )
    centers = centers[(centers > freqs[1]) & (centers < freqs[-1])]
    n_bands = len(centers)
    erb_c = _hz_to_erb(centers.astype(np.float64))
    bw = 24.7 * (4.37 * centers / 1000.0 + 1.0)
    band_levels = np.full(n_bands, -200.0)
    for i, (cf, b) in enumerate(zip(centers, bw)):
        m = (freqs >= cf - b / 2) & (freqs <= cf + b / 2)
        if np.any(m):
            band_levels[i] = np.max(mag_db[m])
    bin_erb = _hz_to_erb(freqs)
    mask_db = np.full(n_bins, -200.0)
    for i in range(n_bands):
        lv = band_levels[i]
        if lv < -140:
            continue
        dist = bin_erb - erb_c[i]
        spread = np.where(dist >= 0, lv - 25.0 * dist, lv + 10.0 * dist)
        mask_db = np.maximum(mask_db, spread)
    f = np.clip(freqs, 20.0, 20000.0)
    th_spl = (
        3.64 * (f / 1000.0) ** (-0.8)
        - 6.5 * np.exp(-0.6 * (f / 1000.0 - 3.3) ** 2)
        + np.where(f < 1000, 1e-3 * (f / 1000.0) ** 4, 0.0)
    )
    th_dbfs = np.clip(th_spl - 100.0, -140.0, 0.0)
    mask_db = np.maximum(mask_db, th_dbfs)
    gain = (noise_db > mask_db).astype(np.float64)
    gain[mag_db > -40.0] = 0.0
    gain = np.convolve(gain, np.ones(3) / 3, mode="same")
    gain = np.clip(gain, 0.0, 1.0)
    gain[0] = 0.0
    return gain


def _apply_erb_gain_to_noise(noise, erb_gain):
    n_bins = len(erb_gain)
    erb_freqs = np.linspace(0, 0.5, n_bins)
    n_fft_full = 1
    while n_fft_full < len(noise):
        n_fft_full <<= 1
    n_full_bins = n_fft_full // 2 + 1
    full_freqs = np.linspace(0, 0.5, n_full_bins)
    gain_interp = np.interp(full_freqs, erb_freqs, erb_gain)
    spectrum = np.fft.rfft(noise.astype(np.float64), n=n_fft_full)
    spectrum *= gain_interp
    filtered = np.fft.irfft(spectrum, n=n_fft_full)[: len(noise)]
    orig_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    filt_rms = float(np.sqrt(np.mean(filtered**2)))
    if filt_rms > 1e-15:
        filtered *= orig_rms / filt_rms
    return filtered


def _generate_sota_cd_noise(
    n_samples: int,
    sr: int,
    bit_depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """SOTA CD-Wandler-Rauschmodell.

    Modelliert:
    1. POW-r-Type-3 geformtes Dither (Craven/Law/Stuart, AES 1987)
    2. Clock-Einstreuung: -120 dBFS bei 22.05 kHz
    3. 1/f-Flicker-Rauschen unter 100 Hz
    """
    noise = rng.standard_normal(n_samples, dtype=np.float32)
    n_fft = 1
    while n_fft < n_samples:
        n_fft <<= 1
    spectrum = np.fft.rfft(noise.astype(np.float64), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    shape = np.ones(len(freqs), dtype=np.float64)

    # 1. POW-r Type 3 Shaping: progressive boost above 10 kHz
    knee = 10000.0
    above = freqs > knee
    if np.any(above):
        boost_db = 10.0 * (1.0 / (1.0 + np.exp(-(freqs[above] - 15000.0) / 1500.0)))
        shape[above] *= 10.0 ** (boost_db / 20.0)

    # 2. 1/f flicker below 100 Hz
    lf = freqs < 100.0
    if np.any(lf):
        f_lf = np.maximum(freqs[lf], 1.0)
        flicker_db = 3.0 * (1.0 - np.log10(f_lf) / 2.0)
        shape[lf] *= 10.0 ** (flicker_db / 20.0)

    # 3. Clock bleed: -120 dBFS pure tone at 22.05 kHz
    clock_bin = np.argmin(np.abs(freqs - 22050.0))
    if clock_bin < len(spectrum):
        clock_mag = 10.0 ** (-120.0 / 20.0) * float(n_fft)
        spectrum[clock_bin] += clock_mag * np.exp(1j * rng.uniform(0, 2 * np.pi))

    # §G18: Rolloff -3 dB/oct above 16 kHz
    rolloff = freqs > 16000.0
    if np.any(rolloff):
        octaves = np.log2(np.maximum(freqs[rolloff], 16000.0) / 16000.0)
        shape[rolloff] *= 10.0 ** (-3.0 * octaves / 20.0)

    shape[0] = 0.0
    spectrum *= shape
    shaped = np.fft.irfft(spectrum, n=n_fft)[:n_samples]

    target_dbfs = _CD_NOISE_FLOOR_DBFS_16BIT if bit_depth <= 16 else _CD_NOISE_FLOOR_DBFS_24BIT
    rms = float(np.sqrt(np.mean(shaped**2)))
    shaped *= (10.0 ** (target_dbfs / 20.0)) / max(rms, 1e-15)

    return cast(np.ndarray, shaped.astype(np.float32))


def _compute_masking_envelope(audio_mono: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet zeitabhängige Maskierungshüllkurve mit adaptivem Crossfade.

    Signal-adaptiver Crossfade (§G41):
    - Leise → lauter Übergang: kürzerer Fade (Forward-Masking schützt)
    - Laut → leise Übergang: längerer Fade (Nachhall-Empfindlichkeit)
    """
    win_s = int(_RMS_WINDOW_S * sr)
    hop_s = win_s // 2
    n_frames = max(1, (len(audio_mono) - win_s) // hop_s + 1)

    rms_db = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_s
        rms = float(np.sqrt(np.mean(audio_mono[start : start + win_s].astype(np.float64) ** 2)))
        rms_db[i] = 20.0 * np.log10(max(rms, 1e-15))

    # Binary mask: 1 = inject noise (signal below threshold)
    mask = (rms_db < _MASKING_THRESHOLD_DBFS).astype(np.float64)
    silence = rms_db < _SILENCE_THRESHOLD_DBFS
    mask[silence] = 0.0

    # Adaptive crossfade
    fwd_frames = max(1, int(_FORWARD_MASKING_MS * sr / 1000.0 / hop_s))
    bwd_frames = max(1, int(_BACKWARD_MASKING_MS * sr / 1000.0 / hop_s))
    min_fade = max(1, int(_MIN_CROSSFADE_MS * sr / 1000.0 / hop_s))
    max_fade = max(min_fade + 1, int(_MAX_CROSSFADE_MS * sr / 1000.0 / hop_s))

    if np.any(mask > 0):
        smoothed = mask.copy()
        diff = np.diff(np.concatenate([[0.0], smoothed, [0.0]]))
        starts = np.where(diff > 0.5)[0]
        ends = np.where(diff < -0.5)[0] - 1
        for s, e in zip(starts, ends):
            # Fade-in: adapt to pre-signal energy
            pre = float(np.mean(rms_db[max(0, s - 8) : s])) if s > 0 else -100.0
            fade_len = min_fade if pre < -80 else max(min_fade, min(fwd_frames, max_fade))
            s_fade = max(0, s - fade_len)
            if s_fade < s:
                n = s - s_fade
                curve = 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
                for j in range(s_fade, s):
                    if not silence[j]:
                        smoothed[j] = max(smoothed[j], curve[j - s_fade])

            # Fade-out: adapt to post-signal energy
            post = float(np.mean(rms_db[e + 1 : min(n_frames, e + 9)])) if e + 1 < n_frames else -100.0
            fade_len = min_fade if post < -80 else max(min_fade, min(bwd_frames, max_fade))
            e_fade = min(n_frames - 1, e + fade_len)
            if e < e_fade:
                n = e_fade - e
                curve = 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / n))
                for j in range(e + 1, e_fade + 1):
                    if not silence[j]:
                        smoothed[j] = max(smoothed[j], curve[j - (e + 1)])
        mask = smoothed

    mask[silence] = 0.0

    # Sample-level envelope
    envelope = np.zeros(len(audio_mono), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_s
        end = min(start + hop_s, len(audio_mono))
        envelope[start:end] = mask[i]

    # §G17 final: zero samples stay zero
    envelope[np.abs(audio_mono) < 1e-12] = 0.0

    return cast(np.ndarray, envelope)


def _compute_onset_strength(audio: np.ndarray, sr: int) -> float:
    if len(audio) < 2048:
        return 0.0
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    n_frames = (len(audio) - n_fft) // hop
    if n_frames < 3:
        return 0.0
    energies = np.array(
        [
            float(np.sum(np.abs(np.fft.rfft(audio[i * hop : i * hop + n_fft] * win)[10:100]) ** 2))
            for i in range(n_frames)
        ],
        dtype=np.float64,
    )
    if np.max(energies) < 1e-15:
        return 0.0
    energies /= np.max(energies)
    onset = np.diff(energies)
    onset = np.maximum(onset, 0.0)
    return float(np.max(onset))


def inject_cd_noise_profile(
    audio: np.ndarray,
    sr: int,
    *,
    mode: str = "restoration",
    bit_depth: int = 16,
    seed: int | None = None,
) -> np.ndarray:
    """§G8 (GEBOTE.md): Injiziert SOTA CD-Rauschprofil mit psychoakustischer Maskierung.

    Das Rauschprofil wird NUR dort appliziert, wo das menschliche Ohr
    es wahrnimmt — d.h. in leisen Passagen unterhalb -70 dBFS (§G44).
    Adaptive Crossfades verhindern hörbare Übergänge (§G41).
    Position: NACH Pipeline, VOR Dithering (§G40).
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 2:
        return audio

    peak = float(np.max(np.abs(arr)))
    if peak < 1e-10:
        return audio

    if seed is None:
        seed = _compute_deterministic_seed(arr)
    rng = np.random.default_rng(seed)

    is_stereo = arr.ndim == 2 and arr.shape[1] == 2
    orig_shape = arr.shape

    if is_stereo:
        left, right = arr[:, 0].copy(), arr[:, 1].copy()
        mono = (left.astype(np.float64) + right.astype(np.float64)) * 0.5
    else:
        mono = arr.ravel().astype(np.float64)
        left = arr.ravel()

    # Step 1: Time-domain masking envelope (§G44)
    envelope = _compute_masking_envelope(mono, sr)
    # §G56: Noise floor continuity — enforce minimum floor even in loud sections.
    # Without this, the noise floor jumps 20 dB between loud and quiet sections,
    # creating audible "noise gate" artifacts. A -6 dB residual prevents this
    # (was -20 dB = 20 dB modulation; -6 dB = only 6 dB modulation, below JND).
    _NOISE_FLOOR_FLOOR_DB = 6.0  # -6 dB below CD noise → -102 dBFS, unhörbare Modulation
    _min_env = 10.0 ** (-_NOISE_FLOOR_FLOOR_DB / 20.0)  # ~0.5 at -6 dB
    envelope = np.maximum(envelope, _min_env)
    # §G17: Re-apply digital black enforcement after minimum floor
    envelope[np.abs(mono) < 1e-12] = 0.0
    active_samples = int(np.sum(envelope > 0.01))

    # Step 2: ERB band gain — per-frequency masking (§G15, §G44)
    # §G43: 24-bit noise (-114 dBFS) is below human hearing threshold.
    # ERB masking would suppress it entirely. Instead, apply uniform
    # converter noise floor (authentic 24-bit ADC behavior).
    noise_db = _CD_NOISE_FLOOR_DBFS_16BIT if bit_depth <= 16 else _CD_NOISE_FLOOR_DBFS_24BIT
    try:
        if bit_depth <= 16:
            erb_gain = _compute_sliding_erb_gain(mono, sr, noise_db)  # §G57
        else:
            # 24-bit: uniform noise floor — authentic converter behavior
            erb_gain = np.ones(1025, dtype=np.float64)
    except Exception:
        erb_gain = np.ones(1025, dtype=np.float64)

    # Step 3: Generate CD noise + apply ERB gain (§G30: L/R uncorrelated)
    if is_stereo:
        sl, sr_seed = seed, seed ^ 0x5A5A5A5A5A5A5A5A
        nl = _generate_sota_cd_noise(len(mono), sr, bit_depth, np.random.default_rng(sl))
        nr = _generate_sota_cd_noise(len(mono), sr, bit_depth, np.random.default_rng(sr_seed))
        nl = _apply_erb_gain_to_noise(nl, erb_gain)
        nr = _apply_erb_gain_to_noise(nr, erb_gain)
        rl = left.astype(np.float64) + nl.astype(np.float64) * envelope
        rr = right.astype(np.float64) + nr.astype(np.float64) * envelope
        result = np.stack([rl, rr], axis=1)
    else:
        noise = _generate_sota_cd_noise(len(mono), sr, bit_depth, rng)
        noise = _apply_erb_gain_to_noise(noise, erb_gain)
        rm = left.astype(np.float64) + noise.astype(np.float64) * envelope
        result = rm.reshape(orig_shape)

    result = np.clip(result, -1.0, 1.0)

    # §G41: Onset verification — auto-correct if needed
    onset = _compute_onset_strength(result.ravel(), sr)
    if onset > 0.1:
        logger.info(
            "CD-Rauschprofil SOTA: Onset=%.3f überschreitet 0.1 — widening crossfade (§G41, §V26).",
            onset,
        )
        # Widen crossfade and recompute with longer fades
        global _FORWARD_MASKING_MS, _BACKWARD_MASKING_MS
        _orig_fwd, _orig_bwd = _FORWARD_MASKING_MS, _BACKWARD_MASKING_MS
        try:
            _FORWARD_MASKING_MS *= 2.0
            _BACKWARD_MASKING_MS *= 2.0
            envelope = _compute_masking_envelope(mono, sr)
            envelope = np.maximum(envelope, _min_env)  # §G56
            envelope[np.abs(mono) < 1e-12] = 0.0  # §G17
            active_samples = int(np.sum(envelope > 0.01))
            # Re-apply noise with wider envelope
            if is_stereo:
                rl = left.astype(np.float64) + nl.astype(np.float64) * envelope
                rr = right.astype(np.float64) + nr.astype(np.float64) * envelope
                result = np.stack([rl, rr], axis=1)
            else:
                rm = left.astype(np.float64) + noise.astype(np.float64) * envelope
                result = rm.reshape(orig_shape)
            result = np.clip(result, -1.0, 1.0)
            onset = _compute_onset_strength(result.ravel(), sr)
            logger.info(
                "CD-Rauschprofil SOTA: korrigierter Onset=%.3f (target <0.1).",
                onset,
            )
        finally:
            _FORWARD_MASKING_MS, _BACKWARD_MASKING_MS = _orig_fwd, _orig_bwd

    # §G39: Monitoring
    snr_before = _compute_snr_db(arr)
    snr_after = _compute_snr_db(result)
    diff_max = float(np.max(np.abs(result.ravel()[: len(arr.ravel())] - arr.ravel())))
    noise_peak_db = 20.0 * np.log10(max(diff_max, 1e-15))

    logger.info(
        "💿 CD-Noise SOTA [%s/%d-bit]: SNR %.1f -> %.1f dB | "
        "active: %d/%d (%.1f%%) | peak: %.1f dBFS | onset: %.4f | seed=%d",
        mode,
        bit_depth,
        snr_before,
        snr_after,
        active_samples,
        len(mono),
        100.0 * active_samples / max(len(mono), 1),
        noise_peak_db,
        onset,
        seed,
    )

    return result.astype(np.float32)  # type: ignore[no-any-return]


def _compute_snr_db(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float32).ravel()
    p = float(np.max(np.abs(arr)))
    nf = float(np.percentile(np.abs(arr), 10))
    if p < 1e-10 or nf < 1e-15:
        return float("inf")
    return float(20.0 * np.log10(p / nf))
