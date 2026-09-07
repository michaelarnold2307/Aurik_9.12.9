"""§2.71 Strength-Envelope v2: SOTA zeitvariante Phasen-Stärke.

Production-grade, inaudible surgical processing via:
  • Defect-type-specific Gaussian spread (clicks=30ms, noise=200ms, …)
  • Asymmetric attack/release smoothing (5ms/50ms) — zero pumping
  • Transient-aware gating — preserves drum/plosive punch
  • Psychoacoustic floor modulation — skip inaudible regions
  • Energy-compensated blending — consistent loudness through crossfades
  • Cubic-spline resampling — no stairstep artifacts
  • Per-frame vocal attenuation via VocalFocusAnalyzer timeline
  • Crossfade windowing at envelope transitions — no clicks

Reference:  ITU-R BS.1387 (PEAQ), Zwicker/Fastl psychoacoustics,
            Giannoulis et al. (2012) "Dynamic Range Compression"
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# §2.71a Constants
# ═══════════════════════════════════════════════════════════════════════

_ENVELOPE_HOP: int = 256  # ~5.3 ms @ 48 kHz (increased temporal resolution)

_MIN_STRENGTH_FLOOR: float = 0.06  # Never completely dry — noise texture continuity
_MAX_STRENGTH: float = 1.0

# Asymmetric smoothing time constants (seconds)
_ATTACK_TAU_S: float = 0.005  # 5 ms — instant onset
_RELEASE_TAU_S: float = 0.050  # 50 ms — gradual release (no pumping)

# Defect-type-specific Gaussian spread (seconds)
_DEFECT_SIGMA: dict[str, float] = {
    "click": 0.030,
    "crackle": 0.060,
    "impulse": 0.025,
    "pop": 0.030,
    "dropout": 0.100,
    "hiss": 0.200,
    "noise": 0.180,
    "hum": 0.250,
    "buzz": 0.200,
    "wow": 0.150,
    "flutter": 0.120,
    "clipping": 0.080,
    "saturation": 0.100,
    "distortion": 0.120,
    "rumble": 0.300,
    "default": 0.100,
}

# Transient detection — energy ratio threshold
_TRANSIENT_ENERGY_RATIO: float = 3.0
_TRANSIENT_WINDOW_MS: float = 8.0  # Short window for transient detection

# Psychoacoustic — floor boost for masked content
_PSY_MASKED_FLOOR_BOOST: float = 0.15  # Raise floor to 0.21 for inaudible content

# Energy compensation — window size for RMS matching
_ENERGY_WINDOW_MS: float = 50.0  # 50ms sliding RMS window

# Crossfade — Hann window duration at envelope edges
_CROSSFADE_MS: float = 2.5  # 2.5ms fade at state transitions


# ═══════════════════════════════════════════════════════════════════════
# §2.71b Core Envelope Computation
# ═══════════════════════════════════════════════════════════════════════


def compute_strength_envelope(
    defect_locations: dict[str, list[tuple[float, float]]] | None,
    defect_severity_map: dict[str, float] | None = None,
    defect_saliency_map: dict[str, float] | None = None,
    audio_duration_s: float = 1.0,
    sample_rate: int = 48_000,
    base_strength: float = 1.0,
    panns_singing: float = 0.0,
    vocal_timeline: np.ndarray | None = None,
    audio: np.ndarray | None = None,
    psychoacoustic_mask: np.ndarray | None = None,
    envelope_hop: int = _ENVELOPE_HOP,
    min_strength: float = _MIN_STRENGTH_FLOOR,
) -> np.ndarray:
    """SOTA Strength-Envelope — chirurgisch, unhörbar, produktionsreif.

    Args:
        defect_locations: {defekttyp: [(t_start_s, t_end_s), …]}
        defect_severity_map: {defekttyp: severity [0,1]}
        defect_saliency_map: {defekttyp: perceptual_salience [0,1]}
        audio_duration_s: Audiolänge in Sekunden
        sample_rate: Abtastrate (Hz)
        base_strength: Basis-Stärke vom Joint-Calibrator [0,1]
        panns_singing: PANNs Gesangs-Konfidenz [0,1]
        vocal_timeline: [n_frames] Vocal-Activity pro Frame (optional)
        audio: [n_samples] Mono-Audio für Transient Detection (optional)
        psychoacoustic_mask: [n_frames] Maskierungs-Schwelle (optional)
        envelope_hop: Hop-Länge (Samples)
        min_strength: Minimale Stärke

    Returns:
        np.ndarray [n_frames] float32 in [min_strength, 1.0]
    """
    n_frames = max(1, int(audio_duration_s * sample_rate / envelope_hop) + 1)
    envelope = np.full(n_frames, min_strength, dtype=np.float64)

    if not defect_locations:
        # §2.46g (2026-09-06): WARNING statt DEBUG — ein uniformes Floor-Envelope
        # bei leerem Locations-Set macht die Phasen-Strecke global zum No-Op
        # (Produktionsbefund: μ=0.060 σ=0.000, ActiveIntervention lehnte jede
        # Phase ab, 42 Restdefekte blieben unbehandelt). Ursache ist meist ein
        # verlorener Locations-/Salienz-Transport aus dem Cache — sichtbar machen.
        logger.warning(
            "§2.71: Keine Defekt-Locations → uniform floor=%.3f (Envelope degeneriert; "
            "Phasen laufen mit Minimal-Stärke — bitte Cached-Scan-Salienz/Locations prüfen)",
            min_strength,
        )
        return cast(np.ndarray, (np.asarray(envelope, dtype=np.float32)))

    sev_map = dict(defect_severity_map or {})
    sal_map = dict(defect_saliency_map or {})

    # ── Stage 1: Defect-density accumulation with type-specific sigma ─
    for defect_type, locs in defect_locations.items():
        severity = float(sev_map.get(defect_type, 0.5))
        salience = float(sal_map.get(defect_type, 1.0))
        # Salience-weighted severity — audible defects get full weight
        weight = float(np.clip(severity * (0.25 + 0.75 * salience), 0.0, 1.0))

        # Defect-type-specific sigma
        sigma_s = _DEFECT_SIGMA.get(defect_type.lower(), _DEFECT_SIGMA["default"])
        sigma_frames = max(1.0, sigma_s * sample_rate / envelope_hop)

        for t_start, t_end in locs:
            if t_end <= t_start:
                continue

            f_start = max(0, int(t_start * sample_rate / envelope_hop))
            f_end = min(n_frames, int(t_end * sample_rate / envelope_hop) + 1)
            if f_end <= f_start:
                continue

            center = (f_start + f_end) / 2.0
            # Defect core width + Gaussian tails
            core_half = max(1.0, (f_end - f_start) / 2.0)
            total_sigma = max(1.0, np.sqrt(core_half**2 + sigma_frames**2))

            f_min = max(0, int(center - 3.5 * total_sigma))
            f_max = min(n_frames, int(center + 3.5 * total_sigma) + 1)

            # Vectorized Gaussian (much faster than per-frame loop)
            f_indices = np.arange(f_min, f_max, dtype=np.float64)
            dist = np.abs(f_indices - center)
            gauss = np.exp(-0.5 * (dist / total_sigma) ** 2)
            envelope[f_min:f_max] = np.maximum(envelope[f_min:f_max], weight * gauss)

    # ── Stage 1.5: Temporal masking (forward 200ms, backward 20ms) ──
    # Laute Transienten maskieren leise Defekte in ihrer zeitlichen Nähe.
    # Psychoakustisch korrekt: Vorwärtsmaskierung (laut→leise) >> Rückwärts.
    if audio is not None and audio.size > 0:
        envelope = _apply_temporal_masking(envelope, audio, sample_rate, envelope_hop)

    # ── Stage 2: Asymmetric attack/release smoothing ─────────────────
    envelope = _apply_asymmetric_smoothing(envelope, sample_rate, envelope_hop)

    # ── Stage 3: Transient-aware gating ──────────────────────────────
    if audio is not None and audio.size > 0:
        transient_mask = _detect_transients(audio, sample_rate, envelope_hop, n_frames)
        if transient_mask is not None and np.any(transient_mask):
            # Reduce envelope on transients — preserve punch
            transient_attenuation = 0.35  # Reduce to 35% on transients
            envelope = np.where(
                transient_mask > 0.5,
                envelope * transient_attenuation + min_strength * (1.0 - transient_attenuation),
                envelope,
            )
            n_trans = int(np.sum(transient_mask > 0.5))
            logger.debug("§2.71: %d transient frames attenuated (×%.2f)", n_trans, transient_attenuation)

    # ── Stage 4: Psychoacoustic floor modulation ─────────────────────
    if psychoacoustic_mask is not None and len(psychoacoustic_mask) > 0:
        psy_mask = _resample_1d(psychoacoustic_mask, n_frames)
        # Higher masking threshold → raise floor (don't waste NR on inaudible)
        # Lower threshold → lower floor (audible content, protect it)
        psy_floor = min_strength + _PSY_MASKED_FLOOR_BOOST * np.clip(psy_mask, 0.0, 1.0)
        envelope = np.maximum(envelope, psy_floor)
        logger.debug("§2.71: Psychoacoustic floor angewendet (μ=%.3f)", float(np.mean(psy_floor)))

    # ── Stage 5: Vocal-aware per-frame modulation ────────────────────
    if vocal_timeline is not None and len(vocal_timeline) > 0:
        vocal_frames = _resample_1d(vocal_timeline, n_frames)
        # Vocal presence → attenuate envelope
        # Formula: attenuation = 1.0 - vocal_presence * 0.6
        #   vocal=1.0 → envelope ×0.40 (strong protection)
        #   vocal=0.0 → envelope ×1.00 (no attenuation)
        vocal_attenuation = np.clip(1.0 - vocal_frames * 0.60, 0.30, 1.0)
        envelope = envelope * vocal_attenuation
        logger.debug("§2.71: Per-frame vocal attenuation (μ=%.2f)", float(np.mean(vocal_attenuation)))
    elif panns_singing > 0.25:
        # Fallback: global vocal attenuation
        vocal_factor = float(np.clip(1.0 - (panns_singing - 0.25) * 1.2, 0.35, 1.0))
        envelope = envelope * vocal_factor
        logger.debug("§2.71: Global vocal attenuation ×%.2f (panns=%.2f)", vocal_factor, panns_singing)

    # ── Stage 6: Base-strength scaling ───────────────────────────────
    if base_strength < 0.95:
        envelope = envelope * base_strength

    # ── Stage 7: Floor + ceiling ─────────────────────────────────────
    envelope = np.clip(envelope, min_strength, _MAX_STRENGTH)

    # ── Stage 8: Crossfade windowing at edges ────────────────────────
    envelope = _apply_crossfade_windows(envelope, sample_rate, envelope_hop)

    # ── Logging ──────────────────────────────────────────────────────
    n_active = int(np.sum(envelope > min_strength + 0.02))
    mean_val = float(np.mean(envelope))
    std_val = float(np.std(envelope))
    logger.info(
        "§2.71 StrengthEnvelope v2: %d/%d frames (%.1f%%), μ=%.3f σ=%.3f, base=%.2f panns=%.2f dur=%.1fs",
        n_active,
        n_frames,
        100.0 * n_active / max(1, n_frames),
        mean_val,
        std_val,
        base_strength,
        panns_singing,
        audio_duration_s,
    )

    return cast(np.ndarray, (np.asarray(envelope, dtype=np.float32)))


# ═══════════════════════════════════════════════════════════════════════
# §2.71c Smoothing, Transient Detection, Windowing
# ═══════════════════════════════════════════════════════════════════════


def _apply_asymmetric_smoothing(
    envelope: np.ndarray,
    sample_rate: int,
    hop: int,
) -> np.ndarray:
    """Asymmetric attack/release smoothing via one-pole filter per frame.

    Attack:  fast (5ms)  — envelope rises immediately when defect starts
    Release: slow (50ms) — envelope falls gradually, no pumping artifact

    This is the key innovation that makes the envelope *inaudible*.
    Without it, abrupt wet→dry transitions cause audible "modulation".
    """
    attack_coeff = float(np.exp(-1.0 / max(1, (_ATTACK_TAU_S * sample_rate / hop))))
    release_coeff = float(np.exp(-1.0 / max(1, (_RELEASE_TAU_S * sample_rate / hop))))

    smoothed = np.copy(envelope)
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i - 1]:
            # Attack: envelope rising → fast follow
            smoothed[i] = attack_coeff * smoothed[i - 1] + (1.0 - attack_coeff) * envelope[i]
        else:
            # Release: envelope falling → slow decay
            smoothed[i] = release_coeff * smoothed[i - 1] + (1.0 - release_coeff) * envelope[i]

    return cast(np.ndarray, (np.asarray(smoothed, dtype=np.float64)))


def _detect_transients(
    audio: np.ndarray,
    sample_rate: int,
    hop: int,
    n_frames: int,
) -> np.ndarray | None:
    """Detect transients for punch preservation.

    Uses energy ratio between short and long windows (Duxbury et al. 2003).
    Returns a boolean mask where transients are present.
    """
    try:
        # Mono
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0) if audio.shape[0] <= 2 else audio[0]
        else:
            audio_mono = audio

        win_short = int(_TRANSIENT_WINDOW_MS * sample_rate / 1000)
        win_long = win_short * 4

        if win_short < 4 or len(audio_mono) < win_long:
            return None

        # Frame-wise energy
        energy = audio_mono.astype(np.float64) ** 2

        # Short-term energy (fast)
        kernel_short = np.ones(win_short) / win_short
        e_short = np.convolve(energy, kernel_short, mode="same")

        # Long-term energy (slow)
        kernel_long = np.ones(win_long) / win_long
        e_long = np.convolve(energy, kernel_long, mode="same")

        # Ratio — where short-term energy spikes above long-term
        eps = 1e-10
        ratio = e_short / (e_long + eps)

        # Decimate to envelope frame rate
        frame_indices = np.arange(0, len(audio_mono), hop)[:n_frames]
        ratio_frames = ratio[frame_indices]

        # Threshold
        return (ratio_frames > _TRANSIENT_ENERGY_RATIO).astype(np.float64)  # type: ignore[no-any-return]

    except Exception:
        logger.debug("§2.71: Transient detection uebersprungen", exc_info=True)
        return None


def _apply_crossfade_windows(
    envelope: np.ndarray,
    sample_rate: int,
    hop: int,
) -> np.ndarray:
    """Apply Hann crossfade windows at envelope transition edges.

    Prevents clicks when the wet/dry ratio changes rapidly.
    Uses a 2.5ms fade window at each transition.
    """
    n_frames = len(envelope)
    if n_frames < 3:
        return envelope

    fade_frames = max(1, int(_CROSSFADE_MS * sample_rate / (1000 * hop)))

    # Detect transitions (significant changes in envelope)
    diff = np.abs(np.diff(envelope))
    threshold = 0.15  # Only crossfade significant changes
    transitions = np.where(diff > threshold)[0]

    if len(transitions) == 0:
        return envelope

    result = np.copy(envelope)
    for t in transitions:
        # Apply Hann window around the transition
        f0 = max(0, t - fade_frames)
        f1 = min(n_frames, t + fade_frames + 1)
        if f1 - f0 < 2:
            continue

        window = np.hanning(f1 - f0)
        # Blend: original envelope at edges, smoothed in center
        blend = result[f0:f1] * (1.0 - window) + _smooth_segment(result[f0:f1]) * window
        result[f0:f1] = blend

    return cast(np.ndarray, result)


def _smooth_segment(segment: np.ndarray) -> np.ndarray:
    """3-point median filter for click-free smoothing."""
    if len(segment) < 3:
        return segment
    return ndimage.median_filter(segment, size=3)  # type: ignore[no-any-return]


def _apply_temporal_masking(
    envelope: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    hop: int,
) -> np.ndarray:
    """Psychoakustische Temporal Masking: Transienten maskieren nahe Defekte.

    Forward masking (200ms, exponentielle Dämpfung τ=50ms):
      Nach einem lauten Transienten sind leise Defekte bis zu 200ms
      lang weniger hörbar → Envelope wird reduziert.

    Backward masking (20ms, τ=5ms):
      Kurz VOR einem Transienten sind Defekte ebenfalls maskiert.

    Referenz: Moore (2003), Zwicker & Fastl (1999), ISO 532-1.
    """
    try:
        n_frames = len(envelope)
        if n_frames < 3:
            return envelope

        # 1. Transient-Detektion (Duxbury-Energie-Ratio)
        if audio.ndim > 1:
            mono = np.mean(audio, axis=0) if audio.shape[0] <= 2 else audio[0]
        else:
            mono = audio

        win_short_s = 0.008
        win_short = max(4, int(win_short_s * sample_rate))
        win_long = win_short * 4

        if len(mono) < win_long:
            return envelope

        energy = mono.astype(np.float64) ** 2
        e_short = np.convolve(energy, np.ones(win_short) / win_short, mode="same")
        e_long = np.convolve(energy, np.ones(win_long) / win_long, mode="same")
        ratio = e_short / (e_long + 1e-10)

        frame_indices = np.arange(0, len(mono), hop)[:n_frames]
        ratio_frames = ratio[frame_indices]
        transients = ratio_frames > 2.5

        if not np.any(transients):
            return envelope

        # 2. Forward-Masking-Profil (exponentiell, 200ms)
        tau_fwd_frames = max(1.0, 0.050 * sample_rate / hop)  # τ=50ms
        fwd_frames = int(0.200 * sample_rate / hop)  # 200ms max
        fwd_mask = np.zeros(fwd_frames, dtype=np.float64)
        for i in range(fwd_frames):
            fwd_mask[i] = np.exp(-i / tau_fwd_frames)
        fwd_mask = 1.0 - fwd_mask * 0.6  # Max 60% Reduktion

        # 3. Backward-Masking-Profil (exponentiell, 20ms)
        tau_bwd_frames = max(1.0, 0.005 * sample_rate / hop)  # τ=5ms
        bwd_frames = int(0.020 * sample_rate / hop)  # 20ms max
        bwd_mask = np.zeros(bwd_frames, dtype=np.float64)
        for i in range(bwd_frames):
            bwd_mask[bwd_frames - 1 - i] = np.exp(-i / tau_bwd_frames)
        bwd_mask = 1.0 - bwd_mask * 0.4  # Max 40% Reduktion

        # 4. Anwenden: Für jeden Transienten Maske auf Envelope
        mask_accum = np.ones(n_frames, dtype=np.float64)
        transient_positions = np.where(transients)[0]

        for pos in transient_positions:
            # Forward (nach dem Transienten)
            end_fwd = min(n_frames, pos + fwd_frames)
            length = end_fwd - pos
            if length > 0:
                mask_accum[pos:end_fwd] = np.minimum(mask_accum[pos:end_fwd], fwd_mask[:length])
            # Backward (vor dem Transienten)
            start_bwd = max(0, pos - bwd_frames)
            length = pos - start_bwd
            if length > 0:
                mask_accum[start_bwd:pos] = np.minimum(mask_accum[start_bwd:pos], bwd_mask[bwd_frames - length :])

        envelope_masked = envelope * mask_accum
        n_reduced = int(np.sum(mask_accum < 0.95))
        if n_reduced > 0:
            logger.debug(
                "§2.71 TemporalMasking: %d/%d Frames reduziert (μ=%.3f)",
                n_reduced,
                n_frames,
                float(np.mean(mask_accum)),
            )

        return cast(np.ndarray, (np.asarray(envelope_masked, dtype=np.float64)))

    except Exception as e:
        logger.warning("strength_envelope.py::unbekannter Ersatzpfad: %s", e)
        return envelope


def _resample_1d(data: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D array to target_len via linear interpolation."""
    if len(data) == target_len:
        return cast(np.ndarray, (np.asarray(data, dtype=np.float64)))
    if len(data) <= 1:
        return cast(np.ndarray, (np.full(target_len, float(data[0]) if len(data) else 0.0, dtype=np.float64)))

    x_orig = np.linspace(0.0, 1.0, len(data))
    x_target = np.linspace(0.0, 1.0, target_len)
    return interp1d(x_orig, data, kind="linear", fill_value="extrapolate")(x_target)  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════
# §2.71d Resampling & Blending
# ═══════════════════════════════════════════════════════════════════════


def resample_envelope_to_sr(
    envelope: np.ndarray,
    target_samples: int,
    envelope_hop: int = _ENVELOPE_HOP,
    cubic: bool = True,
) -> np.ndarray:
    """Resample envelope to audio sample rate.

    Uses cubic spline (default) for smooth, artifact-free interpolation.
    Falls back to linear for very short envelopes.

    Args:
        envelope: [n_frames] envelope values
        target_samples: number of audio samples
        envelope_hop: hop size in samples
        cubic: use cubic spline (True) or linear (False)

    Returns:
        [target_samples] float32 envelope
    """
    n_frames = len(envelope)
    if n_frames <= 1:
        return cast(np.ndarray, (np.full(target_samples, float(envelope[0]) if n_frames else 0.1, dtype=np.float32)))

    env_t = np.arange(n_frames, dtype=np.float64) * envelope_hop
    target_t = np.arange(target_samples, dtype=np.float64)

    if cubic and n_frames >= 4:
        # Cubic spline — smoother than linear, no stairstep
        try:
            spline = interp1d(
                env_t,
                envelope.astype(np.float64),
                kind="cubic",
                bounds_error=False,
                fill_value=(float(envelope[0]), float(envelope[-1])),
            )
            return spline(target_t).astype(np.float32)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("strength_envelope.py::resample_envelope_to_sr Ersatzpfad: %s", e)
            pass  # Fallback to linear

    return cast(np.ndarray, (np.interp(target_t, env_t, envelope.astype(np.float64)).astype(np.float32)))


def apply_strength_envelope_wet_dry(
    original: np.ndarray,
    processed: np.ndarray,
    envelope_sample: np.ndarray,
    min_blend: float = 0.05,
    energy_compensate: bool = True,
) -> np.ndarray:
    """Energy-compensated wet/dry blending.

    Standard blending can cause audible level fluctuations when the
    wet/dry ratio changes — processed audio might be quieter or louder
    than the original. Energy compensation normalizes per-window RMS.

    Args:
        original: original audio (any channel layout)
        processed: processed audio (same shape as original)
        envelope_sample: [n_samples] strength envelope (0=dry, 1=wet)
        min_blend: minimum dry component (avoids total silence)
        energy_compensate: match RMS between orig/proc per window

    Returns:
        blended audio, same shape as input
    """
    orig = np.asarray(original, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)
    env = np.asarray(envelope_sample, dtype=np.float64)

    is_multich = orig.ndim > 1
    n_samples = proc.shape[-1] if is_multich else proc.shape[0]

    # Resample envelope if needed
    if len(env) != n_samples:
        env = resample_envelope_to_sr(
            np.asarray([env], dtype=np.float64).flatten() if env.ndim > 1 else env,
            n_samples,
            cubic=True,
        ).astype(np.float64)

    wet = np.clip(env, 0.0, 1.0)
    dry = np.clip(1.0 - wet, min_blend, 1.0)

    # Energy compensation — match RMS in 50ms sliding windows
    if energy_compensate:
        win_samples = max(16, int(_ENERGY_WINDOW_MS * 48_000 / 1000))
        if n_samples > win_samples * 2:
            if is_multich:
                # Mono-downmix for RMS calculation
                orig_mono = np.mean(orig, axis=0)
                proc_mono = np.mean(proc, axis=0)
            else:
                orig_mono = orig
                proc_mono = proc

            # Sliding window RMS
            kernel = np.hanning(win_samples)
            kernel = kernel / np.sum(kernel)

            orig_rms = np.sqrt(np.convolve(orig_mono**2, kernel, mode="same") + 1e-12)
            proc_rms = np.sqrt(np.convolve(proc_mono**2, kernel, mode="same") + 1e-12)

            # Gain factor: match processed RMS to original RMS
            gain = np.ones(n_samples, dtype=np.float64)
            mask = proc_rms > 1e-10
            gain[mask] = np.clip(orig_rms[mask] / proc_rms[mask], 0.50, 2.0)

            # Apply gain to wet component only
            if is_multich:
                wet_bc = wet[np.newaxis, :]
                gain_bc = gain[np.newaxis, :]
            else:
                wet_bc = wet
                gain_bc = gain

            result = (dry * orig + wet_bc * proc * gain_bc).astype(np.float32)
            return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    # Standard blend (no energy compensation)
    if is_multich:
        while dry.ndim < orig.ndim:
            dry = dry[np.newaxis, :]
        while wet.ndim < orig.ndim:
            wet = wet[np.newaxis, :]

    result = dry * orig + wet * proc
    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))


# ═══════════════════════════════════════════════════════════════════════
# §2.71e Phase Integration API
# ═══════════════════════════════════════════════════════════════════════


def apply_strength_envelope(
    processed: np.ndarray,
    original: np.ndarray,
    envelope: np.ndarray,
    sample_rate: int = 48_000,
    base_strength: float = 1.0,
    envelope_hop: int = _ENVELOPE_HOP,
    min_blend: float = 0.05,
    energy_compensate: bool = True,
) -> np.ndarray:
    """Phase-callable convenience: resample + energy-compensated blend.

    This is the primary API for phases. Handles:
      1. Cubic-spline resampling to audio rate
      2. Base-strength scaling (Joint-Calibrator)
      3. Energy-compensated wet/dry blending
      4. Anti-clipping protection

    Args:
        processed: NR-processed audio
        original: pre-phase reference audio
        envelope: [n_frames] strength envelope
        sample_rate: audio sample rate
        base_strength: Joint-Calibrator strength [0,1]
        envelope_hop: envelope hop size
        min_blend: minimum dry component
        energy_compensate: enable RMS matching

    Returns:
        Blended audio (same shape as processed)
    """
    n_target = processed.shape[-1] if processed.ndim > 1 else processed.shape[0]
    env_sample = resample_envelope_to_sr(envelope, n_target, envelope_hop, cubic=True)

    # Joint-Calibrator scaling
    if base_strength < 0.95:
        env_sample = env_sample * base_strength

    return apply_strength_envelope_wet_dry(
        original=original,
        processed=processed,
        envelope_sample=env_sample,
        min_blend=min_blend,
        energy_compensate=energy_compensate,
    )


def inject_envelope_into_kwargs(
    kwargs: dict[str, Any],
    envelope: np.ndarray | None,
    envelope_hop: int = _ENVELOPE_HOP,
) -> None:
    """Inject strength envelope into phase kwargs (idempotent).

    Phases read ``kwargs.get("strength_envelope")`` for frame-wise blending.
    """
    if envelope is not None and "strength_envelope" not in kwargs:
        kwargs["strength_envelope"] = np.asarray(envelope, dtype=np.float32)
        kwargs["strength_envelope_hop"] = envelope_hop
