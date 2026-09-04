import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


def safe_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono, handling both (N, 2) and (2, N) layouts safely.

    Respects §2.51 Stereo-Kohärenz-Invariante: Convert to mono without loss
    of phase information or spectral coherence.

    Args:
        audio: Input audio, 1D (mono) or 2D (stereo in any orientation)

    Returns:
        Mono audio as 1D numpy array (or scalar for degenerate inputs)
    """
    if audio.ndim == 1:
        return audio

    # Ensure float64 for precision
    audio = audio.astype(np.float64)

    # Determine orientation and convert safely
    if audio.shape[0] == 2 and audio.shape[1] > 2:
        # (2, N) channels-first → mean over channels (axis=0)
        return np.asarray(np.mean(audio, axis=0))  # type: ignore[no-any-return]
    if audio.shape[0] == 2 and audio.shape[1] == 2:
        # Edge case: exactly (2, 2) — ambiguous, but treat as (2, N) channels-first
        # This gives a (2,) output
        return np.asarray(np.mean(audio, axis=0))  # type: ignore[no-any-return]
    if audio.shape[1] == 2:
        # (N, 2) channels-last → mean over channels (axis=1)
        return np.asarray(np.mean(audio, axis=1))  # type: ignore[no-any-return]
    # Ambiguous: use heuristic based on which dimension is smaller
    # (channels are typically 2, samples >> 2)
    axis = 0 if audio.shape[0] < audio.shape[1] else 1
    return np.asarray(np.mean(audio, axis=axis))  # type: ignore[no-any-return]


def stereo_channel_view(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gibt stereo channels as 1D arrays for either (2, N) or (N, 2) layout zurück."""
    if audio.ndim == 1:
        return audio, audio.copy()
    if audio.ndim != 2:
        raise ValueError(f"Stereo audio must be 1D or 2D, got shape {audio.shape}")
    if audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio[0], audio[1]
    if audio.shape[1] == 2:
        return audio[:, 0], audio[:, 1]
    if audio.shape[0] == 2 and audio.shape[1] == 2:
        return audio[0], audio[1]
    raise ValueError(f"Unsupported stereo layout: {audio.shape}")


def stereo_like(left: np.ndarray, right: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Rebuild stereo audio while preserving the template orientation."""
    if template.ndim == 1:
        return np.column_stack([left, right])  # type: ignore[no-any-return]
    if template.ndim != 2:
        raise ValueError(f"Stereo template must be 1D or 2D, got shape {template.shape}")
    if template.shape[0] == 2 and template.shape[1] > 2:
        return np.vstack([left, right])  # type: ignore[no-any-return]
    if template.shape[1] == 2:
        return np.column_stack([left, right])  # type: ignore[no-any-return]
    if template.shape[0] == 2 and template.shape[1] == 2:
        return np.vstack([left, right])  # type: ignore[no-any-return]
    raise ValueError(f"Unsupported stereo template layout: {template.shape}")


def to_channels_last(audio: np.ndarray) -> tuple["np.ndarray", bool]:
    """Normalisiert stereo audio to (N, 2) channels-last layout.

    Returns (normalized_audio, was_transposed) so the caller can restore the
    original orientation with ``restore_layout``.
    """
    if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio.T, True
    return audio, False


def restore_layout(audio: np.ndarray, was_transposed: bool) -> np.ndarray:
    """Undo a ``to_channels_last`` transposition if it was applied."""
    if was_transposed and audio.ndim == 2:
        return audio.T
    return audio


def safe_filtfilt(b, a, x, axis=-1, padtype="odd", padlen=None):
    """Zero-phase filter with automatic short-signal fallback.

    §v10.101: scipy.signal.filtfilt crasht mit "The length of the input
    vector x must be greater than padlen" wenn das Signal kürzer als
    die Filter-Padlänge ist. safe_filtfilt prüft die Länge vorab und
    fällt auf lfilter (minimum-phase) zurück, wenn filtfilt nicht möglich.
    """
    from scipy.signal import filtfilt, lfilter

    n = x.shape[axis] if hasattr(x, "shape") and x.ndim > 0 else len(x)
    # padlen = 3 * max(len(b), len(a)) für b/a; 3 * order für SOS
    if padlen is None:
        padlen = 3 * max(len(b), len(a))
    # Identity filter — no filtering needed
    if len(b) == 1 and len(a) == 1 and np.allclose(b[0], 1.0) and np.allclose(a[0], 1.0):
        return np.asarray(x)
    if n > padlen:
        return filtfilt(b, a, x, axis=axis, padtype=padtype, padlen=padlen)
    if n > max(len(b), len(a)):
        return lfilter(b, a, x, axis=axis)
    return np.asarray(x)


def safe_sosfiltfilt(
    sos: np.ndarray,
    x: np.ndarray,
    axis: int = -1,
    *,
    chain_depth: int = 1,
) -> np.ndarray:
    """Zero-phase SOS filter with depth-adaptive minimum-phase fallback.

    §v10.131: ``scipy.signal.sosfiltfilt`` (zero-phase) erzeugt Pre-Ringing
    durch den Rückwärts-Durchlauf. Auf degradiertem HF-Material (Kassette,
    chain_depth ≥ 4) wird dieses Pre-Ringing hörbar als metallisches Echo
    im 6–12 kHz-Bereich. ``sosfilt`` (kausal, minimum-phase) vermeidet das
    Pre-Ringing vollständig; der Gruppenlaufzeit-Unterschied (< 0.1 ms bei
    48 kHz) ist akustisch irrelevant, solange alle parallelen Bänder
    denselben Filter-Typ verwenden.

    Args:
        sos: Second-order sections (SOS) filter coefficients
        x: Input signal
        axis: Filter axis
        chain_depth: Transfer-Chain-Tiefe (§v10.131). Default 1 = zero-phase.
                     Bei ≥4 wird minimum-phase (sosfilt) verwendet.

    Returns:
        Gefiltertes Signal (float64 oder wie Input).
    """
    from scipy.signal import sosfilt, sosfiltfilt

    if chain_depth >= 4:
        return sosfilt(sos, x, axis=axis)  # type: ignore[no-any-return]
    return sosfiltfilt(sos, x, axis=axis)  # type: ignore[no-any-return]


def safe_stft(
    x: np.ndarray,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§v10.115 Zero-crash STFT with automatic short-signal fallback.

    scipy.signal.stft crasht mit "noverlap must be less than nperseg"
    wenn das Signal kürzer als nperseg ist oder noverlap >= nperseg.
    safe_stft clamt noverlap und skaliert nperseg automatisch.

    Returns (f, t, Zxx) wie scipy.signal.stft.
    """
    from scipy.signal import stft as _scipy_stft

    # §v10.119: Normalisiere boundary=True → 'zeros' (scipy-kompatibel)
    if kwargs.get("boundary") is True:
        kwargs["boundary"] = "zeros"
    n = x.shape[-1] if x.ndim > 0 else len(x)
    # Clamp nperseg to signal length
    _nperseg = min(nperseg, max(2, n))
    # Clamp noverlap: 0 <= noverlap < nperseg
    if noverlap is None:
        _noverlap = _nperseg // 2
    else:
        _noverlap = min(noverlap, max(0, _nperseg - 1))
    try:
        return _scipy_stft(x, fs=fs, window=window, nperseg=_nperseg, noverlap=_noverlap, **kwargs)  # type: ignore[no-any-return]
    except ValueError as exc:
        logger.debug("§V6 scipy STFT fehlgeschlagen — Minimal-Fallback aktiviert (ValueError): %s", exc)
        # Last resort: minimum viable STFT
        _nperseg = max(2, n)
        _noverlap = _nperseg // 4
        return _scipy_stft(x, fs=fs, window="hann", nperseg=_nperseg, noverlap=_noverlap)  # type: ignore[no-any-return]


def safe_istft(
    Zxx: np.ndarray,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """§v10.115 Zero-crash ISTFT with automatic short-signal fallback.

    Companion to safe_stft. Clamps parameters identically.
    Returns (t, x) wie scipy.signal.istft.
    """
    from scipy.signal import istft as _scipy_istft

    # §v10.119: Normalisiere boundary=True → 'zeros' (scipy-kompatibel)
    if kwargs.get("boundary") is True:
        kwargs["boundary"] = "zeros"
    # Clamp noverlap: 0 <= noverlap < nperseg
    if noverlap is None:
        _noverlap = nperseg // 2
    else:
        _noverlap = min(noverlap, max(0, nperseg - 1))
    try:
        return _scipy_istft(Zxx, fs=fs, window=window, nperseg=nperseg, noverlap=_noverlap, **kwargs)  # type: ignore[no-any-return]
    except ValueError as exc:
        logger.debug("§V6 scipy ISTFT fehlgeschlagen — Minimal-Fallback aktiviert (ValueError): %s", exc)
        _noverlap = nperseg // 4
        return _scipy_istft(Zxx, fs=fs, window="hann", nperseg=nperseg, noverlap=_noverlap)  # type: ignore[no-any-return]


def audio_sample_count(audio: np.ndarray) -> int:
    """Gibt the time-axis sample count for mono or stereo audio zurück."""
    if audio.ndim == 1:
        return int(audio.shape[0])
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            return int(audio.shape[1])
        return int(audio.shape[0])
    raise ValueError(f"Unsupported audio rank for sample count: {audio.shape}")


def compute_gated_rms_linear(sig: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Berechnet frame-gated RMS in linear scale (stereo-safe via mono energy).

    §2.45a v10.0.0: Adaptive gate (same as _rms_dbfs_gated in UV3).
    effective_gate = max(gate_dbfs, P5+10) — excludes vinyl/shellac surface-noise
    frames (-35 to -45 dBFS) so that noise removal is not misread as a music-level drop.
    Old fixed gate (-50 dBFS) included all noise frames → false drop → Pegelexplosion.
    """
    x = np.asarray(sig, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if x.ndim == 2:
        if x.shape[0] <= 2 and x.shape[1] > x.shape[0]:
            x = np.mean(x, axis=0)
        else:
            x = np.mean(x, axis=1)
    frame = 480
    n = int(x.shape[0])
    if n < frame:
        return float(np.sqrt(np.mean(x * x)) + 1e-12)

    # Collect all frame energies first (for adaptive gate computation)
    all_frame_power: list[float] = []
    for i in range(0, n - frame + 1, frame):
        f = x[i : i + frame]
        all_frame_power.append(float(np.mean(f * f)))

    # §2.45a Adaptive gate: P5+10 dB above noise floor
    effective_gate_dbfs = gate_dbfs
    if len(all_frame_power) >= 10:
        _p5_power = float(np.percentile(all_frame_power, 5))
        if _p5_power > 0.0:
            _p5_db = 10.0 * float(np.log10(_p5_power + 1e-12))
            _adaptive = _p5_db + 10.0
            if _adaptive > gate_dbfs:  # True whenever P5 > gate-10 (all real audio)
                effective_gate_dbfs = min(_adaptive, gate_dbfs + 25.0)

    gate_lin2 = 10.0 ** (effective_gate_dbfs / 10.0)
    vals: list[float] = [p for p in all_frame_power if p > gate_lin2]
    if not vals:
        return float(np.sqrt(np.mean(x * x)) + 1e-12)
    return float(np.sqrt(float(np.mean(vals))) + 1e-12)


def compute_gated_rms_dbfs(sig: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Berechnet frame-gated RMS in dBFS."""
    rms = compute_gated_rms_linear(sig, gate_dbfs=gate_dbfs)
    return float(20.0 * np.log10(rms + 1e-12))


# §2.45a: Per-material noise floor gate used as hard minimum in compute_signal_relative_gate_dbfs.
# Values = typical noise floor + 6 dB margin (AES/iZotope RX practice).
# Vinyl ≈ -33 dBFS noise → gate -27 dBFS; shellac ≈ -20 dBFS → gate -14 dBFS.
_MATERIAL_GATE_DBFS: dict[str, float] = {
    "shellac": -14.0,
    "wax_cylinder": -10.0,
    "lacquer_disc": -20.0,
    "wire_recording": -20.0,
    "acoustic_78": -14.0,
    "vinyl": -27.0,
    "reel_tape": -32.0,
    "cassette": -38.0,
    "tape": -32.0,
    "mp3_low": -44.0,
    "mp3_medium": -46.0,
    "cd_digital": -48.0,
    "streaming": -48.0,
    "dat": -48.0,
    "minidisc": -44.0,
    "unknown": -36.0,
}


def compute_signal_relative_gate_dbfs(  # pylint: disable=too-many-positional-arguments
    reference_audio: np.ndarray,
    margin_db: float = 9.0,
    percentile: float = 15.0,
    fallback_gate_dbfs: float = -36.0,
    frame_len: int = 480,
    material_key: str | None = None,
) -> float:
    """§2.45a Material-adaptive gate: signal-relative threshold (CEDAR/iZotope RX approach).

    Professional tools (CEDAR, iZotope RX 11, Waves Z-Noise) measure the noise floor of
    the actual source signal and set the gate = noise_floor + margin (6–10 dB).
    This avoids the failure mode of fixed absolute thresholds (e.g. -36.0 dBFS) when
    the source noise floor is higher than the threshold (vinyl: -33 dBFS > -36 dBFS).

    Uses the P15 percentile of frame RMS values (not P5) to get a robust noise floor
    estimate that stays in the actual noise region even for loud pop/rock content where
    P5 can fall into the music region.

    The computed gate can only be equal to or HIGHER than the material floor from
    _MATERIAL_GATE_DBFS — the material floor acts as a hard minimum (same design as
    CEDAR minimum-statistics: measured floor + margin, bounded by known carrier floor).

    Args:
        reference_audio: Pre-phase source audio (the signal whose noise floor to estimate).
        margin_db:        dB margin above noise floor (default 9 dB, AES/iZotope practice).
        percentile:       Percentile of frame RMS to use as noise floor (default 15).
        fallback_gate_dbfs: Used when reference is too short to estimate (< 10 frames).
        frame_len:        Frame length in samples (default 480 = 10 ms @ 48 kHz).
        material_key:     Optional material type (e.g. "vinyl", "shellac"). If provided,
                          _MATERIAL_GATE_DBFS[material_key] acts as the minimum gate.

    Returns:
        Gate threshold in dBFS. Frames above this threshold receive makeup gain.
    """
    _mat_floor = _MATERIAL_GATE_DBFS.get(str(material_key or "unknown").lower(), fallback_gate_dbfs)
    _floor = max(_mat_floor, fallback_gate_dbfs)
    try:
        arr = np.asarray(reference_audio, dtype=np.float32)
        if arr.ndim == 2:
            ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
            mono = np.mean(arr, axis=0) if ch_first else np.mean(arr, axis=1)
        else:
            mono = arr
        n = len(mono)
        n_full = max(1, n // frame_len)
        if n_full < 10:
            return _floor
        rms_db_vals: list[float] = []
        for fi in range(n_full):
            s, e = fi * frame_len, min((fi + 1) * frame_len, n)
            chunk = mono[s:e].astype(np.float64)
            rms_db_vals.append(float(20.0 * np.log10(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)) + 1e-12)))
        tail_s = n_full * frame_len
        if tail_s < n:
            tail = mono[tail_s:].astype(np.float64)
            rms_db_vals.append(float(20.0 * np.log10(float(np.sqrt(np.mean(tail * tail) + 1e-12)) + 1e-12)))
        if len(rms_db_vals) < 10:
            return _floor
        noise_floor_db = float(np.percentile(rms_db_vals, percentile))
        gate = float(np.clip(noise_floor_db + margin_db, -60.0, -10.0))
        return max(_floor, gate)  # material floor as minimum; signal can only raise it
    except Exception as e:
        logger.warning("audio_utils.py::berechnen_signal_relative_gate_dbfs Ersatzpfad: %s", e)
        return _floor


def _edge_channel_views(audio: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return [arr]
    try:
        left, right = stereo_channel_view(arr)
        return [
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
        ]
    except ValueError as exc:
        logger.debug("§V6 stereo_channel_view fehlgeschlagen — Mono-Fallback aktiviert (ValueError): %s", exc)
        return [safe_to_mono(arr)]


def _match_edge_channel_views(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ref_channels = _edge_channel_views(reference_audio)
    cand_channels = _edge_channel_views(candidate_audio)
    n_channels = max(len(ref_channels), len(cand_channels))
    if len(ref_channels) == 1 and n_channels > 1:
        ref_channels = ref_channels * n_channels
    if len(cand_channels) == 1 and n_channels > 1:
        cand_channels = cand_channels * n_channels
    return ref_channels[:n_channels], cand_channels[:n_channels]


def _profile_channel_flags(
    profile: dict[str, float | int | bool] | None,
    key: str,
    fallback: bool,
    channel_count: int,
) -> list[bool]:
    if profile is None:
        return [fallback] * channel_count
    raw_flags = profile.get(key)
    flags: list[bool]
    if isinstance(raw_flags, (list, tuple)):
        flags = [bool(v) for v in raw_flags]
    else:
        flags = []
    if not flags:
        return [fallback] * channel_count
    if len(flags) < channel_count:
        flags.extend([flags[-1]] * (channel_count - len(flags)))
    return flags[:channel_count]


def _quiet_edge_guard_profile(
    reference_audio: np.ndarray,
    sr: int,
    *,
    material_key: str | None = None,
) -> dict[str, float | int | bool] | None:
    """Misst whether original intro/outro should be treated as quiet edges."""
    ref_arr = np.asarray(reference_audio, dtype=np.float32)
    ref = safe_to_mono(ref_arr)
    n = len(ref)
    if n < max(int(sr * 2.0), 4_800):
        return None

    edge_len = min(int(sr * 4.0), max(int(sr * 1.0), int(n * 0.10)))
    centre_len = min(int(sr * 4.0), max(int(sr * 1.0), int(n * 0.20)))
    centre_start = max(0, (n - centre_len) // 2)
    gate_dbfs = compute_signal_relative_gate_dbfs(
        ref,
        fallback_gate_dbfs=-36.0,
        material_key=material_key,
    )
    centre_ref_db = compute_gated_rms_dbfs(ref[centre_start : centre_start + centre_len], gate_dbfs=gate_dbfs)

    intro_ref_db = compute_gated_rms_dbfs(ref[:edge_len], gate_dbfs=gate_dbfs)
    outro_ref_db = compute_gated_rms_dbfs(ref[-edge_len:], gate_dbfs=gate_dbfs)
    intro_quiet = bool((intro_ref_db <= gate_dbfs + 3.0) or (intro_ref_db <= centre_ref_db - 6.0))
    outro_quiet = bool((outro_ref_db <= gate_dbfs + 3.0) or (outro_ref_db <= centre_ref_db - 6.0))

    intro_quiet_channels: list[bool] = []
    outro_quiet_channels: list[bool] = []
    for channel in _edge_channel_views(ref_arr):
        channel = channel[:n]
        centre_ch_db = compute_gated_rms_dbfs(channel[centre_start : centre_start + centre_len], gate_dbfs=gate_dbfs)
        intro_ch_db = compute_gated_rms_dbfs(channel[:edge_len], gate_dbfs=gate_dbfs)
        outro_ch_db = compute_gated_rms_dbfs(channel[-edge_len:], gate_dbfs=gate_dbfs)
        intro_quiet_channels.append(bool((intro_ch_db <= gate_dbfs + 3.0) or (intro_ch_db <= centre_ch_db - 6.0)))
        outro_quiet_channels.append(bool((outro_ch_db <= gate_dbfs + 3.0) or (outro_ch_db <= centre_ch_db - 6.0)))

    return {
        "n": n,
        "edge_len": edge_len,
        "gate_dbfs": gate_dbfs,
        "channel_count": len(intro_quiet_channels),
        "intro_quiet": bool(intro_quiet or any(intro_quiet_channels)),
        "outro_quiet": bool(outro_quiet or any(outro_quiet_channels)),
        "intro_quiet_channels": tuple(intro_quiet_channels),  # type: ignore[dict-item]
        "outro_quiet_channels": tuple(outro_quiet_channels),  # type: ignore[dict-item]
    }


def quiet_edge_boost_ok(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sr: int,
    *,
    material_key: str | None = None,
    max_edge_boost_db: float = 2.0,
) -> bool:
    """Reject candidates that inflate intentionally quiet song edges."""
    profile = _quiet_edge_guard_profile(reference_audio, sr, material_key=material_key)
    if profile is None:
        return True

    ref_channels, cand_channels = _match_edge_channel_views(reference_audio, candidate_audio)
    n = min(
        int(profile["n"]),
        *(len(ch) for ch in ref_channels),
        *(len(ch) for ch in cand_channels),
    )
    if n < max(int(sr * 2.0), 4_800):
        return True

    edge_len = int(profile["edge_len"])
    gate_dbfs = float(profile["gate_dbfs"])
    intro_flags = _profile_channel_flags(
        profile,
        "intro_quiet_channels",
        bool(profile["intro_quiet"]),
        len(ref_channels),
    )
    outro_flags = _profile_channel_flags(
        profile,
        "outro_quiet_channels",
        bool(profile["outro_quiet"]),
        len(ref_channels),
    )

    def _p999_dbfs(x: np.ndarray) -> float:
        return float(20.0 * np.log10(float(np.percentile(np.abs(x.astype(np.float64)), 99.9)) + 1e-12))

    for start, end, channel_flags in (
        (0, edge_len, intro_flags),
        (n - edge_len, n, outro_flags),
    ):
        if not any(channel_flags):
            continue
        for channel_index, (ref_channel, cand_channel) in enumerate(zip(ref_channels, cand_channels)):
            if not channel_flags[channel_index]:
                continue
            ref_edge = ref_channel[:n][start:end]
            cand_edge = cand_channel[:n][start:end]
            ref_edge_db = compute_gated_rms_dbfs(ref_edge, gate_dbfs=gate_dbfs)
            cand_edge_db = compute_gated_rms_dbfs(cand_edge, gate_dbfs=gate_dbfs)
            if cand_edge_db > ref_edge_db + max_edge_boost_db:
                return False

            ref_edge_peak_db = _p999_dbfs(ref_edge)
            cand_edge_peak_db = _p999_dbfs(cand_edge)
            if cand_edge_peak_db > ref_edge_peak_db + max_edge_boost_db + 1.0:
                return False
    return True


def _scale_audio_region(
    audio: np.ndarray,
    start: int,
    end: int,
    scale: float,
    channel_index: int | None = None,
    *,
    crossfade_samples: int = 480,
    taper_in: bool = True,
    taper_out: bool = True,
) -> np.ndarray:
    if scale >= 0.9999 or end <= start:
        return audio
    out = np.array(audio, dtype=np.float32, copy=True)
    # §2.45a-II v10: Crossfade at region boundaries prevents hard clicks
    # when the gain change creates a discontinuity between the scaled
    # edge region and the unscaled music body.
    # taper_in/taper_out=False: hold flat at `scale` instead of ramping
    # to/from 1.0 at that boundary — needed for quiet-edge clamping at a
    # genuine file start/end, where a ramp toward 1.0 would reintroduce the
    # very peak the clamp is meant to suppress right inside the measurement
    # window (§0h Music-Death-Shield). Only the boundary that transitions
    # into/out of unclamped song content (not a file edge) needs a real ramp.
    cf = min(crossfade_samples, (end - start) // 4, 4800)  # max 100 ms @ 48 kHz
    if cf >= 2:
        ramp = (
            np.linspace(1.0, float(scale), cf, dtype=np.float32)
            if taper_in
            else np.full(cf, float(scale), dtype=np.float32)
        )
        iramp = (
            np.linspace(float(scale), 1.0, cf, dtype=np.float32)
            if taper_out
            else np.full(cf, float(scale), dtype=np.float32)
        )
        ramp = np.clip(ramp, 0.0, 1.0)
        iramp = np.clip(iramp, 0.0, 1.0)

        def _apply(ch: np.ndarray) -> None:
            ch[start : start + cf] *= ramp
            ch[start + cf : end - cf] *= np.float32(scale)
            ch[end - cf : end] *= iramp

        if out.ndim == 1:
            _apply(out)
            return cast(np.ndarray, out)
        ch_first = out.shape[0] <= 2 and out.shape[1] > out.shape[0]
        if ch_first:
            if channel_index is None:
                for c in range(out.shape[0]):
                    ch = out[c]
                    ch[start : start + cf] *= ramp
                    ch[start + cf : end - cf] *= np.float32(scale)
                    ch[end - cf : end] *= iramp
            else:
                _apply(out[channel_index])
            return cast(np.ndarray, out)
        if channel_index is None:
            for c in range(out.shape[1]):
                ch = out[:, c]
                ch[start : start + cf] *= ramp
                ch[start + cf : end - cf] *= np.float32(scale)
                ch[end - cf : end] *= iramp
        else:
            _apply(out[:, channel_index])
        return cast(np.ndarray, out)

    # Fallback: no crossfade (region too short)
    if out.ndim == 1:
        out[start:end] *= np.float32(scale)
        return out  # type: ignore[no-any-return]
    ch_first = out.shape[0] <= 2 and out.shape[1] > out.shape[0]
    if ch_first:
        if channel_index is None:
            out[:, start:end] *= np.float32(scale)
        else:
            out[channel_index, start:end] *= np.float32(scale)
        return out  # type: ignore[no-any-return]
    if channel_index is None:
        out[start:end, :] *= np.float32(scale)
    else:
        out[start:end, channel_index] *= np.float32(scale)
    return out  # type: ignore[no-any-return]


def limit_quiet_edge_boost(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sr: int,
    *,
    material_key: str | None = None,
    max_edge_boost_db: float = 2.0,
) -> np.ndarray:
    """Skaliert quiet intro/outro regions back toward the original edge level."""
    profile = _quiet_edge_guard_profile(reference_audio, sr, material_key=material_key)
    if profile is None:
        return np.asarray(candidate_audio, dtype=np.float32)  # type: ignore[no-any-return]

    out = np.asarray(candidate_audio, dtype=np.float32)
    ref_channels, cand_channels = _match_edge_channel_views(reference_audio, out)
    n = min(
        int(profile["n"]),
        *(len(ch) for ch in ref_channels),
        *(len(ch) for ch in cand_channels),
    )
    if n < max(int(sr * 2.0), 4_800):
        return out  # type: ignore[no-any-return]

    edge_len = int(profile["edge_len"])
    gate_dbfs = float(profile["gate_dbfs"])
    intro_flags = _profile_channel_flags(
        profile,
        "intro_quiet_channels",
        bool(profile["intro_quiet"]),
        len(ref_channels),
    )
    outro_flags = _profile_channel_flags(
        profile,
        "outro_quiet_channels",
        bool(profile["outro_quiet"]),
        len(ref_channels),
    )

    def _p999_dbfs(x: np.ndarray) -> float:
        return float(20.0 * np.log10(float(np.percentile(np.abs(x.astype(np.float64)), 99.9)) + 1e-12))

    for start, end, channel_flags in (
        (0, edge_len, intro_flags),
        (n - edge_len, n, outro_flags),
    ):
        if not any(channel_flags):
            continue
        # Only ramp toward 1.0 at a boundary that hands off to unclamped song
        # content — a genuine file start/end needs no such ramp (nothing to
        # blend with) and ramping there would reintroduce the clamped peak
        # right inside the measurement window (§0h Music-Death-Shield).
        _taper_in = start != 0
        _taper_out = end != n
        for channel_index, (ref_channel, cand_channel) in enumerate(zip(ref_channels, cand_channels)):
            if not channel_flags[channel_index]:
                continue
            ref_edge = ref_channel[:n][start:end]
            cand_edge = cand_channel[:n][start:end]
            ref_edge_db = compute_gated_rms_dbfs(ref_edge, gate_dbfs=gate_dbfs)
            cand_edge_db = compute_gated_rms_dbfs(cand_edge, gate_dbfs=gate_dbfs)
            ref_edge_peak_db = _p999_dbfs(ref_edge)
            cand_edge_peak_db = _p999_dbfs(cand_edge)

            scale = 1.0
            if cand_edge_db > ref_edge_db + max_edge_boost_db:
                scale = min(scale, float(10.0 ** ((ref_edge_db + max_edge_boost_db - cand_edge_db) / 20.0)))
            if cand_edge_peak_db > ref_edge_peak_db + max_edge_boost_db + 0.02:
                scale = min(
                    scale,
                    float(10.0 ** ((ref_edge_peak_db + max_edge_boost_db + 0.02 - cand_edge_peak_db) / 20.0)),
                )
            out = _scale_audio_region(
                out,
                start,
                end,
                max(scale, 0.0),
                channel_index=channel_index,
                taper_in=_taper_in,
                taper_out=_taper_out,
                # §Fix: a 10ms crossfade (default) partially retains near-1.0
                # amplitude for long enough to dominate a percentile(99.9)
                # peak measurement over the edge region for large corrections
                # — a short ~0.3ms click-avoidance fade avoids this while
                # still preventing an audible discontinuity at the boundary.
                crossfade_samples=min(16, max(sr // 3000, 4)),
            )
            _, cand_channels = _match_edge_channel_views(reference_audio, out)
    return out  # type: ignore[no-any-return]


def apply_musical_gain_envelope(  # pylint: disable=too-many-positional-arguments
    audio: np.ndarray,
    gain: float,
    gate_dbfs: float = -36.0,
    crossfade_ms: float = 200.0,
    sr: int = 48000,
    reference_for_gate: np.ndarray | None = None,
    material_key: str | None = None,
    *,
    knee_width_db: float = 6.0,
    small_gain_bypass_db: float = 2.0,
) -> np.ndarray:
    """§2.45a-II v10: Apply makeup gain with soft-knee continuous envelope.

    Uses a sigmoid soft-knee instead of a binary gate to create musically
    transparent gain transitions.  Frames near the noise-floor threshold
    receive partial gain, eliminating the audible pumping/jumping caused
    by hard on/off gate switching in earlier versions.

    Architecture (§2.45a-II v10):
        1. Compute per-frame RMS (10 ms frames @ 48 kHz).
        2. Determine adaptive gate threshold via CEDAR/iZotope RX approach
           (P15 + margin from reference signal, bounded by material floor).
        3. Build soft-knee envelope: sigmoid((rms_db - effective_gate) / knee_width_db)
           → continuous gain factor between 0.0 (silence) and 1.0 (music).
        4. Apply long crossfade (default 200 ms) for additional temporal smoothing.
        5. Small-gain bypass: when gain ≤ small_gain_bypass_db, apply uniform gain
           — the risk of amplifying noise by ≤ 2 dB is negligible compared to
           the risk of gate-induced audible artefacts.

    Design rationale:
        - Soft knee (6 dB default) = industry-standard compressor knee width.
          Frames 6 dB above effective_gate → ~88 % of target gain.
          Frames at effective_gate → 50 % of target gain.
          Frames 6 dB below effective_gate → ~12 % of target gain.
        - 200 ms crossfade = musically meaningful transition (≈ 1/16 note @ 120 BPM),
          replacing the old 10 ms click-avoidance window that was acoustically
          transparent for singular transients but created audible pumping when the
          gate repeatedly opened/closed at musical phrase boundaries.
        - No hard clamp: the soft knee naturally handles quiet-zone protection;
          the old §2.30b hard clamp created sharp gain discontinuities at the
          boundary between quiet and musical frames.

    Adaptive gate (CEDAR/iZotope RX approach, unchanged from v10.0.0):
        effective_gate = max(gate_dbfs, compute_signal_relative_gate_dbfs(reference, material_key))
        When reference_for_gate is provided, the gate is estimated from the
        ORIGINAL signal's noise floor.  When None, audio itself is used.

    Args:
        audio:                Input audio (1D or 2D float32).
        gain:                 Linear gain factor (>= 1.0; values <= 1.0005 are skipped).
        gate_dbfs:            Floor threshold — effective gate can only be equal to or HIGHER.
        crossfade_ms:         Width of the temporal smoothing window (default 200 ms).
        sr:                   Sample rate used to convert crossfade_ms to samples.
        reference_for_gate:   Optional pre-phase audio for noise-floor estimation.
        material_key:         Optional material type (e.g. "vinyl", "shellac").
        knee_width_db:        Soft-knee transition width in dB (default 6.0).
        small_gain_bypass_db: Gains ≤ this value skip the gate entirely (default 2.0 dB).

    Returns:
        Audio with gain applied via soft-knee envelope, same shape and dtype.
    """
    # Scalar early-exit only — array gain passes through (broadcast in per_sample_gain)
    if np.ndim(gain) == 0 and float(gain) <= 1.0005:
        return audio

    # §2.45a-II v10: Small-gain bypass — for gains ≤ small_gain_bypass_db, apply
    # uniform gain.  The risk of amplifying noise by ≤ 2 dB is negligible, and
    # avoiding the gate entirely eliminates any risk of audible artefacts.
    _gain_db = float(20.0 * np.log10(float(gain)))
    if np.ndim(gain) == 0 and _gain_db <= small_gain_bypass_db:
        arr = np.asarray(audio, dtype=np.float32)
        return cast(np.ndarray, (arr * gain).astype(np.float32))

    arr = np.asarray(audio, dtype=np.float32)
    was_2d = arr.ndim == 2
    # Build mono energy signal for gate detection (from audio being amplified)
    if was_2d:
        ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
        mono = np.mean(arr, axis=0) if ch_first else np.mean(arr, axis=1)
    else:
        mono = arr
    n = len(mono)
    frame_len = 480  # 10 ms @ 48 kHz
    n_full = max(1, n // frame_len)

    # --- Pass 1: collect per-frame RMS values (for gate-envelope construction) ---
    frame_rms_db: list[float] = []
    for fi in range(n_full):
        s = fi * frame_len
        e = min(s + frame_len, n)
        chunk = mono[s:e].astype(np.float64)
        frame_rms_db.append(float(20.0 * np.log10(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)) + 1e-12)))
    tail_rms_db: float | None = None
    tail_s = n_full * frame_len
    if tail_s < n:
        tail = mono[tail_s:].astype(np.float64)
        tail_rms_db = float(20.0 * np.log10(float(np.sqrt(np.mean(tail * tail) + 1e-12)) + 1e-12))
        frame_rms_db.append(tail_rms_db)

    # --- Adaptive gate: compute signal-relative threshold (CEDAR/iZotope RX approach) ---
    _gate_ref = reference_for_gate if reference_for_gate is not None else audio
    effective_gate = compute_signal_relative_gate_dbfs(
        _gate_ref,
        material_key=material_key,
        fallback_gate_dbfs=gate_dbfs,
    )
    # gate_dbfs is the hard floor — signal-relative gate can only raise it, never lower.
    effective_gate = max(gate_dbfs, effective_gate)

    # --- Pass 2: build soft-knee envelope using sigmoid ---
    # §2.45a-II v10: Replace binary gate (0 or 1) with continuous sigmoid soft knee.
    # soft_gate = 1 / (1 + exp(-(rms_db - effective_gate) / knee_width_db))
    # This creates a smooth, musical transition: frames well above the threshold
    # get near-full gain, frames well below get near-unity, and frames near the
    # threshold get proportional partial gain.
    _knee = max(knee_width_db, 0.5)  # prevent division by zero / near-zero
    gate_env = np.zeros(n, dtype=np.float32)
    full_rms = frame_rms_db[:n_full] if tail_rms_db is not None else frame_rms_db
    for fi, rms_db in enumerate(full_rms):
        # Sigmoid: maps (-inf, +inf) → (0, 1), centered at effective_gate
        _z = (rms_db - effective_gate) / _knee
        # Clamp _z for numerical stability; exp(±15) is already saturated
        _z = float(np.clip(_z, -15.0, 15.0))
        _soft = float(1.0 / (1.0 + np.exp(-_z)))
        s = fi * frame_len
        e = min(s + frame_len, n)
        gate_env[s:e] = _soft
    if tail_rms_db is not None:
        _z_tail = (tail_rms_db - effective_gate) / _knee
        _z_tail = float(np.clip(_z_tail, -15.0, 15.0))
        _soft_tail = float(1.0 / (1.0 + np.exp(-_z_tail)))
        gate_env[tail_s:] = _soft_tail

    # Smooth transitions with longer crossfade (default 200 ms)
    cf_samples = max(1, int(crossfade_ms * sr / 1000.0))
    if cf_samples > 1:
        # Use Hanning window for smoother temporal response than box-blur
        _kernel = np.hanning(min(cf_samples, n)).astype(np.float32)
        _kernel /= float(np.sum(_kernel) + 1e-12)
        gate_env = np.convolve(gate_env, _kernel, mode="same")
        gate_env = np.clip(gate_env, 0.0, 1.0)
    per_sample_gain = (1.0 + (gain - 1.0) * gate_env).astype(np.float32)

    # §2.30b removed in v10: the old hard clamp destroyed the smooth transition
    # created by the crossfade, reintroducing sharp gain discontinuities at
    # quiet/music boundaries.  The soft-knee sigmoid naturally handles quiet-zone
    # protection — frames well below the gate get near-unity gain, and the
    # transition is continuous.

    def _render(gain_env: np.ndarray) -> np.ndarray:
        if was_2d:
            ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
            if ch_first:
                return cast(np.ndarray, np.asarray(arr * gain_env[np.newaxis, :], dtype=np.float32))
            return cast(np.ndarray, np.asarray(arr * gain_env[:, np.newaxis], dtype=np.float32))
        return cast(np.ndarray, np.asarray(arr * gain_env, dtype=np.float32))

    out = _render(per_sample_gain)
    edge_reference = reference_for_gate if reference_for_gate is not None else arr
    edge_profile = _quiet_edge_guard_profile(edge_reference, sr, material_key=material_key)
    if edge_profile is not None and not quiet_edge_boost_ok(
        edge_reference,
        out,
        sr,
        material_key=material_key,
    ):
        edge_len = int(edge_profile["edge_len"])
        # §v10.128 FIX: Statt Hard-Clamp auf max 1.0 (erzeugt Stufenfunktion an
        # der Edge-Grenze → hörbarer Lautstärkesprung in den ersten Sekunden)
        # verwenden wir einen Smooth-Crossfade: linearer Übergang von 1.0 (bei t=0)
        # zum originalen Gain-Wert (bei t=edge_len). Das bewahrt die musikalische
        # Kontinuität und verhindert den abrupten Pegelsprung.
        # Crossfade-Länge: 200 ms (§2.45a-II) oder edge_len/4, je nachdem was kürzer ist.
        _cf_samples = min(max(int(0.200 * sr), 480), max(edge_len // 4, 1))
        if bool(edge_profile["intro_quiet"]):
            _clamped = np.minimum(per_sample_gain[:edge_len], 1.0)
            if _cf_samples < edge_len:
                # Smooth blend: ramp from clamped (t=0) to original (t=edge_len)
                _ramp = np.linspace(1.0, 0.0, _cf_samples, dtype=np.float32)
                _iramp = 1.0 - _ramp
                per_sample_gain[:_cf_samples] = _clamped[:_cf_samples] * _ramp + per_sample_gain[:_cf_samples] * _iramp
                per_sample_gain[_cf_samples:edge_len] = _clamped[_cf_samples:edge_len]
            else:
                per_sample_gain[:edge_len] = _clamped
        if bool(edge_profile["outro_quiet"]):
            _clamped_outro = np.minimum(per_sample_gain[-edge_len:], 1.0)
            if _cf_samples < edge_len:
                # Smooth blend: ramp from original (t=-edge_len) to clamped (t=-0)
                _ramp_out = np.linspace(0.0, 1.0, _cf_samples, dtype=np.float32)
                _iramp_out = 1.0 - _ramp_out
                per_sample_gain[-edge_len : -edge_len + _cf_samples] = (
                    _clamped_outro[:_cf_samples] * _ramp_out
                    + per_sample_gain[-edge_len : -edge_len + _cf_samples] * _iramp_out
                )
                per_sample_gain[-edge_len + _cf_samples :] = _clamped_outro[_cf_samples:]
            else:
                per_sample_gain[-edge_len:] = _clamped_outro
        out = _render(per_sample_gain)
        out = limit_quiet_edge_boost(
            edge_reference,
            out,
            sr,
            material_key=material_key,
        )
    return out


def check_gain_safety(
    audio: np.ndarray,
    requested_gain: float,
    max_peak_dbfs: float = -1.0,
) -> tuple[float, bool]:
    """Pre-flight: compute the maximum gain that won't clip the audio.

    §2.51a / §2.45a preventive approach: calculate max safe gain BEFORE
    applying it, so Pegelexplosion can never happen in the first place.

    Uses 99.9th-percentile peak (§DSP-invariant) to avoid impulse artefacts
    (crackle, clicks) blocking normalisation of the musical content.

    Args:
        audio:          Input audio (any shape float32).
        requested_gain: Desired linear gain factor.
        max_peak_dbfs:  Hard ceiling in dBFS (default -1.0 dBTP, broadcast-safe).

    Returns:
        (safe_gain, was_clamped) where safe_gain ≤ requested_gain and
        was_clamped=True iff the gain was reduced to stay under the ceiling.
    """
    if requested_gain <= 1.0005:
        return float(requested_gain), False
    arr = np.asarray(audio, dtype=np.float32)
    peak99 = float(np.percentile(np.abs(arr), 99.9))
    if peak99 < 1e-9:
        return 1.0, True  # Silent — no positive gain allowed
    max_peak_linear = float(10.0 ** (max_peak_dbfs / 20.0))
    max_safe = max_peak_linear / peak99
    if requested_gain <= max_safe:
        return float(requested_gain), False
    return float(max(1.0, max_safe)), True


def apply_soft_clip(
    audio: np.ndarray,
    ceiling: float = 1.0,
    knee_db: float = 0.6,
    material: str | None = None,
) -> np.ndarray:
    """§v10.62 Sanftes Soft-Clipping via tanh — KEIN Hard-Clamp auf ±1.0.

    Ersetzt ALLE `np.clip(audio, -1.0, 1.0)` Aufrufe im gesamten Projekt.
    Hard-Clipping erzeugt hörbare Obertöne (Rechteck-Fenster → sinc-Spektrum).
    Tanh-basiertes Soft-Clipping erzeugt nur ungerade Harmonische, die das
    menschliche Ohr als „analoge Sättigung" statt als „digitalen Clip" wahrnimmt.

    Material-adaptive Knee:
    - Shellac/Vinyl: 1.2 dB (weicher, bewahrt Charakter)
    - Tape/Cassette: 0.8 dB (mittel)
    - Digital/CD:    0.4 dB (transparent, fast unsichtbar)
    - Default:       0.6 dB

    Args:
        audio:   float32/64, mono oder stereo
        ceiling: Clipping-Schwelle (Default 1.0 = 0 dBFS)
        knee_db: Weiche der Übergangszone in dB
        material: Material-Typ für adaptive Knee

    Returns:
        float32 Array, sanft auf ±ceiling begrenzt.
    """
    import numpy as np

    # Material-adaptive knee
    if material is not None:
        _mat = str(material).lower()
        if any(t in _mat for t in ("shellac", "vinyl", "wax", "lacquer")):
            knee_db = 1.2
        elif any(t in _mat for t in ("tape", "cassette", "reel")):
            knee_db = 0.8
        elif any(t in _mat for t in ("digital", "cd", "streaming", "mp3")):
            knee_db = 0.4

    arr = np.asarray(audio, dtype=np.float64)
    # NaN/Inf vorbehandlung
    arr = np.nan_to_num(arr, nan=0.0, posinf=ceiling, neginf=-ceiling)

    if ceiling <= 0.0:
        return cast(np.ndarray, (np.zeros_like(audio, dtype=np.float32)))

    # Normalisiere auf ceiling → tanh → skaliere zurück
    knee_linear = float(10.0 ** (-knee_db / 20.0))  # z.B. 0.6 dB → 0.933
    soft_threshold = ceiling * knee_linear  # Einsatzpunkt des Soft-Clips

    # Nur Werte über soft_threshold werden weich begrenzt
    abs_arr = np.abs(arr)
    mask = abs_arr > soft_threshold
    if np.any(mask):
        # Normalisiere den Überschuss auf [0, 1]
        excess = (abs_arr[mask] - soft_threshold) / (ceiling - soft_threshold + 1e-15)
        # Tanh-Softclip: asymptotische Annäherung an ceiling
        soft_clipped = soft_threshold + (ceiling - soft_threshold) * np.tanh(excess)
        arr[mask] = np.sign(arr[mask]) * soft_clipped

    return cast(np.ndarray, arr.astype(np.float32))


def crossfade_to_bypass(
    processed: np.ndarray,
    original: np.ndarray,
    fade_ms: float = 5.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """§v10.62 Sanfter Übergang von bearbeitetem zu unbearbeitetem Audio.

    Verhindert hörbare Sprünge wenn eine Phase effective_strength=0 erreicht
    und das unbearbeitete Original zurückgibt. Erzeugt einen kurzen Crossfade
    (Default 5 ms) der für das menschliche Ohr nicht als „Klick" wahrnehmbar ist.

    Args:
        processed:  Letzter bearbeiteter Frame (kann stereo sein)
        original:   Unbearbeiteter Frame, auf den übergeblendet wird
        fade_ms:    Dauer des Crossfades in ms
        sample_rate: Abtastrate

    Returns:
        Sanft übergeblendetes Audio.
    """
    import numpy as np

    fade_len = max(1, int(fade_ms * sample_rate / 1000))
    fade_len = min(fade_len, processed.shape[-1], original.shape[-1])

    if fade_len < 2:
        return cast(np.ndarray, original.astype(np.float32))

    result = original.copy().astype(np.float64)

    # Cosine-Fade: smooth, keine hörbaren Diskontinuitäten
    t = np.linspace(0, np.pi / 2, fade_len)
    fade_in = np.sin(t)  # 0 → 1
    fade_out = np.cos(t)  # 1 → 0

    if processed.ndim == 2 and original.ndim == 2:
        for ch in range(processed.shape[0]):
            result[ch, :fade_len] = fade_out * processed[ch, -fade_len:] + fade_in * original[ch, :fade_len]
    elif processed.ndim == 1 and original.ndim == 1:
        result[:fade_len] = fade_out * processed[-fade_len:] + fade_in * original[:fade_len]
    else:
        # Dimension mismatch — fallback to original
        return cast(np.ndarray, original.astype(np.float32))

    return cast(np.ndarray, result.astype(np.float32))


# ── §v10.99 Edge Taper: Filter-Ringing am Audio-Ende eliminieren ──


def apply_edge_taper(
    audio: np.ndarray,
    sr: int,
    *,
    taper_ms: float = 12.0,
    fade_in: bool = True,
    fade_out: bool = True,
) -> np.ndarray:
    """Wendet kurze Fade-In/Fade-Out-Hanning-Fenster an den Audio-Rändern an.

    Eliminiert Filter-Ringing-Artefakte (sosfiltfilt, STFT-Überlappung)
    an den Audiogrenzen. 12 ms sind unterhalb der Wahrnehmbarkeitsschwelle
    (Haas-Effekt: < 30 ms), aber ausreichend für Filterausschwingen.

    Args:
        audio: Audio-Array (mono oder stereo, float32).
        sr: Sample-Rate.
        taper_ms: Dauer des Fades in ms (default 12 ms).
        fade_in: Fade-In am Anfang anwenden.
        fade_out: Fade-Out am Ende anwenden.

    Returns:
        Audio mit sanften Rändern (gleiche Shape).
    """
    result = np.asarray(audio, dtype=np.float64).copy()
    n_taper = max(2, int(sr * taper_ms / 1000.0))
    n_total = result.shape[-1] if result.ndim > 1 else len(result)

    if n_taper * 2 >= n_total:
        return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))  # too short, skip

    if fade_in:
        win_in = np.hanning(n_taper * 2)[:n_taper].astype(np.float64)
        if result.ndim == 2:
            win_in = win_in[:, np.newaxis]
            result[:, :n_taper] *= win_in.T if result.shape[0] <= 2 else win_in
        elif result.ndim == 1:
            result[:n_taper] *= win_in
        else:
            # Channels-first (C, N)
            result[:, :n_taper] *= win_in[np.newaxis, :]

    if fade_out:
        win_out = np.hanning(n_taper * 2)[n_taper:].astype(np.float64)
        if result.ndim == 2:
            win_out = win_out[:, np.newaxis]
            result[:, -n_taper:] *= win_out.T if result.shape[0] <= 2 else win_out
        elif result.ndim == 1:
            result[-n_taper:] *= win_out
        else:
            result[:, -n_taper:] *= win_out[np.newaxis, :]

    return cast(np.ndarray, (np.asarray(result, dtype=np.float32)))


# ── §v10.305 Safe Resampling mit Längen-Differenz-Guard ────────────────


def safe_resample_poly(
    audio: np.ndarray,
    up: int,
    down: int,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Resampelt mit polyphase-Filter und Längen-Differenz-Guard (§H05).

    Verhindert Zeitkompression bei Längen-Mismatch (>0.1% Differenz → trim/pad statt resample).

    Args:
        audio: Input-Audio (1D oder 2D)
        up: Upsampling-Faktor
        down: Downsampling-Faktor
        axis: Achse für Resampling (default=-1 = Zeitachse)

    Returns:
        Resampeltes Audio mit korrekter Länge
    """
    # Erwartete Länge berechnen
    n_samples = audio.shape[axis]
    expected_len = int(n_samples * up / down)

    try:
        from scipy.signal import resample_poly as _resample_poly

        result = _resample_poly(audio, up, down, axis=axis)
    except (ValueError, TypeError):
        logger.warning("safe_resample_poly: Resampling fehlgeschlagen → Kopie zurück")
        return audio.copy()

    # Längen-Differenz-Guard (§H05)
    actual_len = result.shape[axis]
    diff_ratio = abs(actual_len - expected_len) / max(expected_len, 1)

    if diff_ratio > 0.001:  # >0.1% Differenz
        logger.warning(
            "safe_resample_poly: Längen-Mismatch (%d vs %d, %.2f%%) → trim/pad statt resample",
            actual_len,
            expected_len,
            diff_ratio * 100,
        )
        # Trim oder pad auf erwartete Länge
        if result.shape[axis] > expected_len:
            # Trimmen
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(0, expected_len)
            return cast(np.ndarray, (np.asarray(result[tuple(slices)])))
        else:
            # Pad mit Nullen
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (0, expected_len - actual_len)
            padded = np.pad(result, pad_width, mode="constant", constant_values=0.0)
            return cast(np.ndarray, (np.asarray(padded)))

    return cast(np.ndarray, (np.asarray(result)))


# ── §v10.304 Safe Array Construction ────────────────────────────────────


def safe_asarray(
    obj: object,
    dtype: Any = None,
    fallback_shape: tuple = (0,),
) -> np.ndarray:
    """Numpy-Array-Konvertierung ohne P5-Crash.

    Fängt 'setting an array element with a sequence' (P5, 204×/Run)
    und broadcast-errors (P11, 59×/Run) ab. Liefert leeres Array
    mit fallback_shape statt Exception.

    Usage:
        from backend.core.audio_utils import safe_asarray
        arr = safe_asarray(some_data, dtype=np.float32)
    """
    try:
        if isinstance(obj, np.ndarray):
            return np.asarray(obj, dtype=dtype) if dtype else obj
        _arr = np.asarray(obj, dtype=dtype)
        return cast(np.ndarray, _arr)
    except (ValueError, TypeError):
        logger.debug("safe_asarray: inhomogeneous data, returning zeros")
        return cast(np.ndarray, (np.zeros(fallback_shape, dtype=dtype or np.float32)))
