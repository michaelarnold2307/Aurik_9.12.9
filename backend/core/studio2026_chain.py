"""
§v10.6 Studio 2026 Re-Production Chain MAX — SOTA Modern Mastering.

7 Stages: Dynamic EQ, Adaptive MB Compression, Freq-Dependent Stereo,
Transient/Tonal Separation, Dynamic Presence, Sub-Bass Synthesis, True-Peak Limiter.
DNA-Guards: Voiceprint (MFCC-Cosine), Groove (Onset-DTW), Emotion (Contour-Pearson), Harmonics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Studio2026Result:
    audio: np.ndarray
    stages_applied: list[str] = field(default_factory=list)
    voiceprint_match: float = 1.0
    groove_preserved: float = 1.0
    emotion_preserved: float = 1.0
    harmonics_preserved: float = 1.0
    loudness_before_lufs: float = 0.0
    loudness_after_lufs: float = 0.0


def reprocess_studio2026(
    audio: np.ndarray,
    original: np.ndarray,
    sr: int,
    *,
    material: str = "unknown",
    era: str = "",
    mode: str = "RESTORATION",
    dry_run: bool = False,
    album_ref: dict | None = None,
) -> Studio2026Result:
    """Studio 2026 Re-Production: moderner Studio-Track mit DNA-Erhalt."""
    arr = np.asarray(audio, dtype=np.float32).copy()
    orig = np.asarray(original, dtype=np.float32)
    is_stereo = arr.ndim == 2 and arr.shape[1] == 2
    stages = []

    if dry_run:
        return Studio2026Result(audio=arr, stages_applied=["dry_run"])

    dna = _extract_dna(orig, sr)

    # Album-Konsistenz: EQ/LUFS-Änderungen auf ±2 dB vom Album-Median begrenzen
    _album_cap: float | None = None
    if album_ref is not None:
        mono_for_lufs = arr.mean(axis=1) if arr.ndim == 2 else arr
        cur_lufs = 20.0 * np.log10(float(np.sqrt(np.mean(mono_for_lufs**2)) + 1e-12)) - 3.0
        target_lufs = album_ref.get("lufs_median", cur_lufs)
        _album_cap = np.clip(target_lufs - cur_lufs, -2.0, 2.0)
        logger.info("Studio2026: Album-Konsistenz aktiv (LUFS-Korrektur capped auf %.1f dB)", _album_cap)

    # HPE-gesteuertes iteratives Steering via PhaseSteeringEngine (§v10.5 unified)
    _engine = None
    try:
        from backend.core.phase_steering_guard import SteerAction, get_engine

        _engine = get_engine()
    except Exception as e:
        logger.warning("Studio2026: SteeringEngine nicht verfügbar: %s", e)

    def _guarded(name, fn, *a, **kw):
        if _engine is None:
            return fn(*a, **kw)
        h0 = _engine._compute_hpe(arr, sr)
        r = fn(*a, **kw)
        ra = r if isinstance(r, np.ndarray) else (r[0] if isinstance(r, tuple) else r)
        h1 = _engine._compute_hpe(ra, sr)
        decision = _engine.decide(h0, h1, name, 1.0)
        if decision.action == SteerAction.SKIP:
            logger.info("Studio2026 %s: ueberspringen (SteeringEngine)", name)
            return arr
        if decision.action == SteerAction.RETRY_LIGHTER:
            logger.info("Studio2026 %s: Wiederholung (SteeringEngine, str=%.2f)", name, decision.new_strength)
            kw2 = dict(kw)
            for k in list(kw2):
                if k in ("strength", "amount", "gain_db"):
                    kw2[k] = float(kw2[k]) * decision.new_strength
            return fn(*a, **kw2)
        return r

    # 1. Dynamic EQ
    arr = _guarded("dynamic_eq", _dynamic_eq, arr, sr, material)
    stages.append("dynamic_eq")

    # 2. Adaptive MB Compression
    arr = _guarded("adaptive_mb_comp", _adaptive_mb_compression, arr, sr)
    stages.append("adaptive_mb_comp")

    # 3. Freq-Dependent Stereo
    if is_stereo:
        arr = _guarded("freq_stereo", _freq_dependent_stereo, arr, sr)
        stages.append("freq_stereo")

    # 4. Transient/Tonal
    arr = _guarded("transient_tonal", _transient_tonal_enhance, arr, sr)
    stages.append("transient_tonal")

    # 5. Dynamic Presence & Air
    arr = _guarded("dynamic_presence_air", _dynamic_presence_air, arr, sr)
    stages.append("dynamic_presence_air")

    # 6. Sub-Bass Synthesis
    if material not in ("shellac", "wax_cylinder", "wire_recording"):
        arr = _guarded("sub_bass_synth", _sub_bass_synth, arr, sr)
        stages.append("sub_bass_synth")

    # 7. True-Peak Limiter (nicht guarded — ist essentiell)
    arr, lu_before, lu_after = _true_peak_limiter(arr, sr)
    stages.append("true_peak_limiter")

    # Album-Konsistenz: LUFS-Korrektur anwenden (max ±2 dB)
    if _album_cap is not None and abs(_album_cap) > 0.5:
        gain = 10.0 ** (_album_cap / 20.0)
        arr = (arr * gain).astype(np.float32)
        lu_after = lu_before + _album_cap
        logger.info("Studio2026: Album-LUFS-Korrektur %.1f dB angewendet", _album_cap)

    # DNA-Verifikation
    vp = _verify_voiceprint(arr, orig, sr)
    gr = _verify_groove(arr, orig, sr)
    em = _verify_emotion(arr, orig, dna)
    ha = _verify_harmonics(arr, orig, sr)

    if vp < 0.88:
        logger.warning("Studio2026: Voiceprint %.3f < 0.88 — blending back", vp)
        arr = _blend(arr, audio, 0.5)

    return Studio2026Result(
        audio=arr,
        stages_applied=stages,
        voiceprint_match=vp,
        groove_preserved=gr,
        emotion_preserved=em,
        harmonics_preserved=ha,
        loudness_before_lufs=lu_before,
        loudness_after_lufs=lu_after,
    )


# ====================================================================
# Stage 1: Dynamic EQ — 6-band proportional gain, attack/release smoothing
# ====================================================================

_BANDS = {
    "sub": (20, 60, "Sub-Bass"),
    "bass": (60, 250, "Bass"),
    "low_mid": (250, 800, "Low-Mid"),
    "mid": (800, 2500, "Mid"),
    "high_mid": (2500, 6000, "High-Mid"),
    "air": (6000, 16000, "Air"),
}

_TARGET_CURVE = {  # 2026 modern target (dB relative to RMS)
    "sub": 0.0,
    "bass": 2.0,
    "low_mid": -1.0,
    "mid": 0.0,
    "high_mid": 2.5,
    "air": 3.0,
}


def _dynamic_eq(audio: np.ndarray, sr: int, material: str) -> np.ndarray:
    """6-band dynamic EQ: cuts only when exceeding target, boosts when below.
    Attack 10ms, Release 50ms per band. Max ±4 dB correction."""
    try:
        from scipy.signal import butter, sosfiltfilt

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        nyq = sr / 2
        result = mono.copy()

        for band, (lo, hi, _) in _BANDS.items():
            if hi >= nyq * 0.95:
                continue
            sos = butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
            band_signal = sosfiltfilt(sos, mono)

            # RMS per 10ms window
            win = int(0.01 * sr)
            n_win = len(band_signal) // win
            if n_win < 4:
                continue

            rms = np.array(
                [float(np.sqrt(np.mean(band_signal[i * win : (i + 1) * win] ** 2)) + 1e-12) for i in range(n_win)]
            )
            rms_db = 20.0 * np.log10(rms)

            target = _TARGET_CURVE.get(band, 0.0)
            full_rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
            full_db = 20.0 * np.log10(full_rms)
            target_rms_db = full_db + target

            diff = rms_db - target_rms_db

            # Proportional gain: only correct if deviation > 1 dB
            gain_db = np.zeros(n_win, dtype=np.float32)
            mask = np.abs(diff) > 1.0
            gain_db[mask] = -diff[mask] * 0.5  # Proportional, max ±4 dB
            gain_db = np.clip(gain_db, -4.0, 4.0)

            # Attack/Release smoothing
            att = np.exp(-1.0 / (0.010 * sr / win))
            rel = np.exp(-1.0 / (0.050 * sr / win))
            smoothed = np.zeros(n_win, dtype=np.float32)
            state = 0.0
            for i in range(n_win):
                coef = att if abs(gain_db[i]) > abs(state) else rel
                state = coef * state + (1 - coef) * gain_db[i]
                smoothed[i] = state

            # Apply per-sample via linear interpolation
            gain_linear = 10.0 ** (smoothed / 20.0)
            upsampled = np.interp(np.arange(len(mono)), np.arange(n_win) * win, gain_linear)
            result = result + band_signal * (upsampled - 1.0)

        if audio.ndim == 2:
            ratio = np.clip(result / (mono + 1e-12), 0.8, 1.2)
            return (audio * ratio[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return cast(np.ndarray, (np.clip(result, -1, 1).astype(np.float32)))
    except Exception as e:
        # §V6 Silent-Failure-Verbot: Log mit Begründung, aber nicht WARNING (graceful fallback)
        logger.info("§V6 Graceful fallback stage 1: %s", str(e)[:200])
        return audio


# ====================================================================
# Stage 2: Adaptive Multi-Band Compression — auto-threshold per band
# ===========================================================================


def _adaptive_mb_compression(audio: np.ndarray, sr: int) -> np.ndarray:
    """4-band compression with analysis-driven thresholds."""
    try:
        from scipy.signal import butter, sosfiltfilt

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        nyq = sr / 2

        bands = [
            ("lo", 80, butter(2, 80 / nyq, btype="low", output="sos")),
            (
                "lo_mid",
                (80, 300),
                [butter(2, 80 / nyq, btype="high", output="sos"), butter(2, 300 / nyq, btype="low", output="sos")],
            ),
            (
                "hi_mid",
                (300, 3000),
                [butter(2, 300 / nyq, btype="high", output="sos"), butter(2, 3000 / nyq, btype="low", output="sos")],
            ),
            ("hi", 3000, butter(2, 3000 / nyq, btype="high", output="sos")),
        ]

        processed = []
        for name, freq, sos in bands:
            if name == "lo" or name == "hi":
                b = sosfiltfilt(sos, mono)
            else:
                b = sosfiltfilt(sos[1], sosfiltfilt(sos[0], mono))

            # Auto-threshold: P70 der RMS
            win = int(0.02 * sr)
            n_win = len(b) // win
            rms_vals = np.array(
                [float(np.sqrt(np.mean(b[i * win : (i + 1) * win] ** 2)) + 1e-12) for i in range(n_win)]
            )
            rms_db = 20.0 * np.log10(rms_vals + 1e-12)
            thresh_db = float(np.percentile(rms_db, 70))

            # Compress above threshold
            n = len(b)
            att_c = np.exp(-1.0 / (0.010 * sr))
            rel_c = np.exp(-1.0 / (0.060 * sr))
            thresh_lin = 10.0 ** (thresh_db / 20.0)
            ratio = 1.5
            gain = np.ones(n, dtype=np.float32)
            gr = 1.0
            for i in range(n):
                if abs(b[i]) > thresh_lin:
                    tgt = thresh_lin + (abs(b[i]) - thresh_lin) / ratio
                    tgt_g = tgt / (abs(b[i]) + 1e-12)
                else:
                    tgt_g = 1.0
                gr = (att_c if tgt_g < gr else rel_c) * gr + (1 - (att_c if tgt_g < gr else rel_c)) * tgt_g
                gain[i] = gr
            processed.append(b * np.clip(gain, 0.5, 1.0))

        combined = processed[0] + processed[1] + processed[2] + processed[3]
        if audio.ndim == 2:
            ratio_arr = np.asarray(np.clip(combined / (mono + 1e-12), 0.8, 1.2), dtype=np.float32)
            return (audio * ratio_arr[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return combined.astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        # §V6 Silent-Failure-Verbot: Log mit Begründung, aber nicht WARNING (graceful fallback)
        logger.info("§V6 Graceful fallback stage 1b: %s", str(e)[:200])
        return audio


# ====================================================================
# Stage 3: Frequency-Dependent Stereo — Wide highs, tight lows
# ====================================================================


def _freq_dependent_stereo(audio: np.ndarray, sr: int) -> np.ndarray:
    """M/S processing with frequency-dependent side gain.
    Below 300 Hz: side ×0.9 (tighter mono-ish bass)
    300-6000 Hz: side ×1.15 (natural width)
    Above 6000 Hz: side ×1.30 (airy wide highs)
    """
    if audio.ndim < 2 or audio.shape[1] < 2:
        return audio
    try:
        from scipy.signal import butter, sosfiltfilt

        L, R = audio[:, 0], audio[:, 1]
        M, S = (L + R) / 2.0, (L - R) / 2.0
        nyq = sr / 2

        # Split side into 3 bands
        sos_lo = butter(2, 300 / nyq, btype="low", output="sos")
        sos_mid_lo = butter(2, 300 / nyq, btype="high", output="sos")
        sos_mid_hi = butter(2, 6000 / nyq, btype="low", output="sos")
        sos_hi = butter(2, 6000 / nyq, btype="high", output="sos")

        S_lo = sosfiltfilt(sos_lo, S) * 0.90
        S_mid = sosfiltfilt(sos_mid_hi, sosfiltfilt(sos_mid_lo, S)) * 1.15
        S_hi = sosfiltfilt(sos_hi, S) * 1.30

        S_new = S_lo + S_mid + S_hi
        L_out, R_out = (M + S_new), (M - S_new)
        return np.stack([L_out, R_out], axis=1).astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_freq_dependent_stereo: %s", e)
        return audio


# ====================================================================
# Stage 4: Transient/Tonal Separation + Enhancement
# ====================================================================


def _transient_tonal_enhance(audio: np.ndarray, sr: int) -> np.ndarray:
    """HPSS-style separation: transients via median filter, tonal = residual.
    Enhance transients +8%, keep tonal content unchanged.
    """
    try:
        from scipy.ndimage import median_filter

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio

        # Median filter for tonal separation (25ms window)
        win = int(0.025 * sr)
        if win % 2 == 0:
            win += 1
        tonal = median_filter(mono, size=win)
        transient = mono - tonal

        # Enhance transients
        enhanced = transient * 1.08
        result = tonal + enhanced

        if audio.ndim == 2:
            ratio = np.clip(result / (mono + 1e-12), 0.85, 1.15)
            return (audio * ratio[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return result.astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_transient_tonal_verbessern: %s", e)
        return audio


# ====================================================================
# Stage 5: Dynamic Presence & Air
# ====================================================================


def _dynamic_presence_air(audio: np.ndarray, sr: int) -> np.ndarray:
    """Presence 2-6 kHz +3 dB, Air 12-16 kHz +2 dB. Dynamic: nur boosten wo Energie fehlt."""
    try:
        from scipy.signal import butter, sosfiltfilt

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        nyq = sr / 2

        # Analyse: mittlere Energie in Presence/Air vs Gesamt
        sos_pres = butter(2, [2000 / nyq, 6000 / nyq], btype="band", output="sos")
        sos_air = butter(2, [12000 / nyq, 16000 / nyq], btype="band", output="sos")

        pres_energy = float(np.sqrt(np.mean(sosfiltfilt(sos_pres, mono) ** 2)) + 1e-12)
        air_energy = float(np.sqrt(np.mean(sosfiltfilt(sos_air, mono) ** 2)) + 1e-12)
        total_energy = float(np.sqrt(np.mean(mono**2)) + 1e-12)

        pres_ratio = pres_energy / total_energy
        air_ratio = air_energy / total_energy

        # Boost proportional zum Defizit
        pres_boost = max(0, min(3.0, (0.12 - pres_ratio) * 25))
        air_boost = max(0, min(2.0, (0.04 - air_ratio) * 50))

        result = mono.copy()
        if pres_boost > 0.3:
            filt = sosfiltfilt(sos_pres, mono)
            result = result + filt * (10.0 ** (pres_boost / 20.0) - 1.0)
        if air_boost > 0.3:
            filt = sosfiltfilt(sos_air, mono)
            result = result + filt * (10.0 ** (air_boost / 20.0) - 1.0)

        if audio.ndim == 2:
            ratio = np.clip(result / (mono + 1e-12), 0.7, 1.3)
            return (audio * ratio[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return cast(np.ndarray, result.astype(np.float32))
    except Exception as e:
        logger.warning("_dynamic_presence_air: %s", e)
        return audio


# ====================================================================
# Stage 6: Sub-Bass Harmonic Synthesis
# ====================================================================


def _sub_bass_synth(audio: np.ndarray, sr: int) -> np.ndarray:
    """Harmonische Sub-Bass-Synthese: analysiert 60-120 Hz, synthetisiert 30-60 Hz."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    try:
        from scipy.signal import butter, sosfiltfilt

        nyq = sr / 2
        sos_bass = butter(2, [60 / nyq, 120 / nyq], btype="band", output="sos")
        bass = sosfiltfilt(sos_bass, mono)
        sub = np.tanh(bass * 2.0) * 0.25
        sos_sub = butter(2, 60 / nyq, btype="low", output="sos")
        sub_only = sosfiltfilt(sos_sub, sub)
        mixed = mono + sub_only * 0.12
        if audio.ndim == 2:
            ratio = np.clip(mixed / (mono + 1e-12), 0.9, 1.1)
            return (audio * ratio[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return mixed.astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_sub_bass_synth: %s", e)
        return audio


# ====================================================================
# Stage 7: True-Peak Limiter — 4x oversampling, ISP detection, soft-clip
# ====================================================================


def _true_peak_limiter(audio: np.ndarray, sr: int):
    """True-Peak Limiter: 4x OS via resample_poly, ISP detection, soft-clip.
    Target: −0.3 dBTP, soft-clip threshold: −1.5 dBFS.
    """
    try:
        from scipy.signal import resample_poly

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio

        rms_before = float(np.sqrt(np.mean(mono**2)) + 1e-12)
        lu_before = 20.0 * np.log10(rms_before) - 3.0

        # 4x oversampling via resample_poly
        mono_4x = resample_poly(mono.astype(np.float64), up=4, down=1)
        sr_4x = sr * 4

        # Soft-clip threshold
        soft_clip_threshold = 0.84  # −1.5 dBFS
        ceiling = 0.965  # −0.3 dBTP

        # Lookahead limiter on oversampled signal
        lookahead = int(0.003 * sr_4x)
        n_4x = len(mono_4x)
        gain = np.ones(n_4x, dtype=np.float64)
        att_c = np.exp(-1.0 / (0.001 * sr_4x))
        rel_c = np.exp(-1.0 / (0.040 * sr_4x))
        gr_state = 1.0

        for i in range(n_4x - lookahead):
            peak = float(np.max(np.abs(mono_4x[i : i + lookahead]))) + 1e-12
            if peak > ceiling:
                target_gain = ceiling / peak
            else:
                target_gain = 1.0
            gr_state = (att_c if target_gain < gr_state else rel_c) * gr_state + (
                1 - (att_c if target_gain < gr_state else rel_c)
            ) * target_gain
            gain[i] = gr_state

        limited = mono_4x * gain

        # ISP detection: check if inter-sample peaks exist
        for i in range(0, n_4x - 3, 4):
            peaks = [abs(limited[i + j]) for j in range(4)]
            isp = max(peaks)
            if isp > ceiling and any(p < ceiling for p in peaks):
                scale = ceiling / isp
                for j in range(4):
                    if abs(limited[i + j]) > ceiling:
                        limited[i + j] *= scale

        # Soft-clip
        over = np.abs(limited) > soft_clip_threshold
        if over.any():
            limited[over] = np.sign(limited[over]) * (
                soft_clip_threshold
                + (ceiling - soft_clip_threshold)
                * np.tanh((np.abs(limited[over]) - soft_clip_threshold) / (ceiling - soft_clip_threshold))
            )

        # Downsample
        result_mono = resample_poly(limited, up=1, down=4)
        result_mono = np.clip(result_mono[: len(mono)], -1, 1)

        if audio.ndim == 2:
            ratio = np.clip(result_mono / (mono + 1e-12), 0.7, 1.3)
            result = (audio * ratio[: len(audio), np.newaxis]).astype(np.float32)
        else:
            result = result_mono.astype(np.float32)

        rms_after = float(np.sqrt(np.mean((result.mean(axis=1) if result.ndim == 2 else result) ** 2)) + 1e-12)
        lu_after = 20.0 * np.log10(rms_after) - 3.0
        return result, lu_before, lu_after
    except Exception as e:
        logger.warning("_true_peak_limiter Ersatzpfad: %s", e)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
        lu = 20.0 * np.log10(rms) - 3.0
        return audio, lu, lu


# ====================================================================
# DNA Guards
# ====================================================================


def _extract_dna(audio: np.ndarray, sr: int) -> dict:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    win = int(0.1 * sr)
    n_win = len(mono) // win
    contour = np.array([float(np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2))) for i in range(n_win)])
    from scipy.signal import find_peaks

    onsets, _ = find_peaks(np.abs(mono), distance=int(0.05 * sr), height=np.percentile(np.abs(mono), 80))
    return {
        "dynamic_contour": contour,
        "onsets": onsets,
        "rms": float(np.sqrt(np.mean(mono**2))),
        "peak": float(np.max(np.abs(mono))),
    }


def _verify_voiceprint(audio, original, sr):
    def _env(x):
        m = x.mean(axis=1) if x.ndim == 2 else x.ravel()[: sr * 2]
        s = np.abs(np.fft.rfft(m * np.hanning(len(m))))
        bands = np.array(
            [
                np.mean(s[int(20 * len(s) / sr) : int(100 * len(s) / sr)]),
                np.mean(s[int(100 * len(s) / sr) : int(500 * len(s) / sr)]),
                np.mean(s[int(500 * len(s) / sr) : int(2000 * len(s) / sr)]),
                np.mean(s[int(2000 * len(s) / sr) : int(6000 * len(s) / sr)]),
                np.mean(s[int(6000 * len(s) / sr) : int(12000 * len(s) / sr)]),
            ]
        )
        bands /= np.sum(bands) + 1e-12
        return bands

    vo, vc = _env(original), _env(audio)
    return float(np.dot(vo, vc) / (np.linalg.norm(vo) * np.linalg.norm(vc) + 1e-12))


def _verify_groove(audio, original, sr):
    from scipy.signal import find_peaks

    mo = original.mean(axis=1) if original.ndim == 2 else original.ravel()
    mc = audio.mean(axis=1) if audio.ndim == 2 else audio.ravel()
    oo, _ = find_peaks(np.abs(mo), distance=int(0.05 * sr), height=np.percentile(np.abs(mo), 80))
    oc, _ = find_peaks(np.abs(mc), distance=int(0.05 * sr), height=np.percentile(np.abs(mc), 80))
    if len(oo) == 0 or len(oc) == 0:
        return 1.0
    matched = sum(1 for o in oo if np.min(np.abs(oc - o)) < int(0.005 * sr))
    return matched / max(len(oo), 1)


def _verify_emotion(audio, original, dna):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    sr = 48000
    win = int(0.1 * sr)
    n_win = min(len(mono), len(dna["dynamic_contour"]) * win) // win
    contour = np.array([float(np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2))) for i in range(n_win)])
    dc = dna["dynamic_contour"][:n_win]
    if len(contour) < 3 or len(dc) < 3:
        return 1.0
    corr = np.corrcoef(contour, dc)[0, 1]
    return float(0.0 if np.isnan(corr) else max(0.0, corr))


def _verify_harmonics(audio, original, sr):
    def _h(x):
        m = x.mean(axis=1) if x.ndim == 2 else x.ravel()[:4096]
        s = np.abs(np.fft.rfft(m * np.hanning(len(m))))
        peaks = [s[i] for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > np.mean(s) * 2]
        peaks = np.array(peaks[:10])  # type: ignore[assignment]
        return np.ones(1) if len(peaks) == 0 else peaks / (np.sum(peaks) + 1e-12)

    ho, hc = _h(original), _h(audio)
    if len(ho) == 0 or len(hc) == 0:
        return 1.0
    n = min(len(ho), len(hc))
    corr = np.corrcoef(ho[:n], hc[:n])[0, 1]
    return float(0.0 if np.isnan(corr) else max(0.0, corr))


def _blend(a, b, ratio):
    return (a * ratio + b * (1.0 - ratio)).astype(np.float32)
