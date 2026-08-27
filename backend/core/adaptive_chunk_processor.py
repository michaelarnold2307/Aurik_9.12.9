"""
backend/core/adaptive_chunk_processor.py — Aurik 10.0.0 §7.6: Severity-adaptive Chunk-Verarbeitung

Provides chunk-size computation and a generic chunked-processing wrapper
that phases can opt into.  Chunk size is driven by defect severity:

  - silence  → 120 s
  - sev ≥ 0.6 →  5 s  (fine-grained surgical repair)
  - sev ≥ 0.3 → 15 s
  - else      → 60 s  (min 2 s / max 120 s)

Crossfade between chunks uses Hanning window (10 ms) to prevent
boundary artefacts (§4.5 MRSA-Zonen-Spec).

Reference: copilot-instructions.md §7.6 (Chunk-Größe).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §7.6 Chunk-size constants
# ---------------------------------------------------------------------------

CHUNK_SILENCE_S: float = 120.0
CHUNK_HIGH_SEV_S: float = 5.0  # severity >= 0.6
CHUNK_MED_SEV_S: float = 15.0  # severity >= 0.3
CHUNK_LOW_SEV_S: float = 60.0  # default
CHUNK_MIN_S: float = 2.0
CHUNK_MAX_S: float = 120.0
CROSSFADE_S: float = 0.050  # 50 ms Hann crossfade (unhörbar auch bei 30 Hz Subbass)

# Maximum tolerance for beat-snapping chunk boundaries (opt-in feature).
# A boundary is only moved if the nearest beat is within this tolerance.
_BEAT_SNAP_TOL_S: float = 0.50  # ±500 ms


def compute_chunk_size_s(max_severity: float, is_silence: bool = False) -> float:
    """Gibt adaptive chunk size in seconds per §7.6 zurück.

    Args:
        max_severity: Highest defect severity relevant for the current phase (0.0–1.0).
        is_silence:   True if audio is (near-)silence.

    Returns:
        Chunk duration in seconds, clamped to [CHUNK_MIN_S, CHUNK_MAX_S].
    """
    if is_silence:
        return CHUNK_SILENCE_S
    if max_severity >= 0.6:
        return CHUNK_HIGH_SEV_S
    if max_severity >= 0.3:
        return CHUNK_MED_SEV_S
    return CHUNK_LOW_SEV_S


def _is_near_silence(audio: np.ndarray, threshold_db: float = -55.0) -> bool:
    """Prüft whether *audio* is near-silent (RMS below threshold)."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-15))
    db = 20.0 * np.log10(rms + 1e-15)
    return db < threshold_db  # type: ignore[no-any-return]


def _estimate_beat_times(audio_mono: np.ndarray, sr: int) -> list[float]:
    """Estimate beat positions in seconds using madmom RNNBeatProcessor.

    Uses the RNNBeatProcessor + DBNBeatTrackingProcessor pipeline from madmom
    (Böck et al. 2016 — canonically referenced in Spec §4.1).
    Falls back to librosa beat tracking when madmom is unavailable.
    Returns empty list on failure (caller falls back to regular boundaries).

    Args:
        audio_mono: 1-D float32 audio at *sr* Hz.
        sr:         Sample rate.

    Returns:
        Sorted list of beat timestamps in seconds.
    """
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # type: ignore[import]

        _proc = RNNBeatProcessor()(audio_mono)
        _beats = DBNBeatTrackingProcessor(fps=100)(_proc)
        return sorted(float(b) for b in _beats)
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    try:
        import librosa

        _, beat_frames = librosa.beat.beat_track(y=audio_mono, sr=sr, units="time")  # type: ignore[attr-defined]
        return sorted(float(b) for b in beat_frames)
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    return []


def _snap_to_beat(
    pos_samples: int,
    sr: int,
    beat_times_s: list[float],
    tol_s: float = _BEAT_SNAP_TOL_S,
    min_samples: int = 0,
    max_samples: int | None = None,
) -> int:
    """Snap a sample-domain position to the nearest beat within *tol_s*.

    Args:
        pos_samples:  Original boundary position (samples).
        sr:           Sample rate.
        beat_times_s: Sorted beat timestamps in seconds.
        tol_s:        Maximum allowed deviation (default 0.5 s).
        min_samples:  Lower bound on returned position.
        max_samples:  Upper bound on returned position (or None).

    Returns:
        Snapped position in samples, or *pos_samples* when no beat is close.
    """
    if not beat_times_s:
        return pos_samples
    pos_s = pos_samples / sr
    best_beat_s = min(beat_times_s, key=lambda b: abs(b - pos_s))
    if abs(best_beat_s - pos_s) > tol_s:
        return pos_samples  # no beat within tolerance
    snapped = int(best_beat_s * sr)
    snapped = max(snapped, min_samples)
    if max_samples is not None:
        snapped = min(snapped, max_samples)
    return snapped


def _find_safe_boundary(
    pos_samples: int,
    audio: np.ndarray,
    sr: int,
    tol_ms: float = 20.0,
    shift_ms: float = 25.0,
    onset_threshold: float = 0.35,
) -> int:
    """Verschiebt chunk boundary away from transients (§7.6a, spec Y3).

    If onset_strength > *onset_threshold* within ±*tol_ms* ms of *pos_samples*,
    the boundary is shifted +*shift_ms* ms forward to avoid cutting mid-transient.
    Falls back to the original position if onset detection fails.

    Args:
        pos_samples:      Candidate boundary position (samples).
        audio:            Full audio array (mono or stereo, shape [..., samples]).
        sr:               Sample rate.
        tol_ms:           Half-window length in ms for onset detection (±20 ms).
        shift_ms:         Forward shift applied when transient is detected (25 ms).
        onset_threshold:  onset_strength threshold above which a transient is assumed.

    Returns:
        Adjusted boundary position (samples); original position if no transient found.
    """
    tol_samples = max(1, int(tol_ms * sr / 1000.0))
    shift_samples = int(shift_ms * sr / 1000.0)
    n_total = audio.shape[-1] if audio.ndim >= 2 else len(audio)

    window_start = max(0, pos_samples - tol_samples)
    window_end = min(n_total, pos_samples + tol_samples)
    if window_end <= window_start:
        return pos_samples

    try:
        import librosa  # Available in .venv_aurik; lightweight import after first use

        # Downmix to mono for onset detection
        if audio.ndim == 2:
            audio_window = audio[:, window_start:window_end].mean(axis=0).astype(np.float32)
        else:
            audio_window = audio[window_start:window_end].astype(np.float32)

        hop_length = max(1, int(sr * 0.010))  # 10 ms hop
        oenv = librosa.onset.onset_strength(y=audio_window, sr=sr, hop_length=hop_length)  # type: ignore[attr-defined]
        if oenv.size > 0 and float(np.max(oenv)) > onset_threshold:
            candidate = pos_samples + shift_samples
            if candidate < n_total:
                logger.debug(
                    "§7.6a TransientGuard: onset %.3f > %.2f within ±%dms → boundary +%dms",
                    float(np.max(oenv)),
                    onset_threshold,
                    int(tol_ms),
                    int(shift_ms),
                )
                return candidate
    except Exception as _exc:
        logger.debug("_find_safe_boundary: onset detection uebersprungen (%s)", _exc)

    return pos_samples


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChunkProcessingResult:
    """Result of chunked phase processing."""

    audio: np.ndarray
    chunk_size_s: float
    n_chunks: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_in_adaptive_chunks(
    phase_fn,
    audio: np.ndarray,
    sr: int,
    max_severity: float,
    *,
    phase_kwargs: dict | None = None,
    crossfade_s: float = CROSSFADE_S,
    beat_sync_chunks: bool = False,
) -> ChunkProcessingResult:
    """Run *phase_fn* on severity-adaptive chunks with overlap-add crossfade.

    This is an OPT-IN utility.  Phases that benefit from fine-grained
    chunk processing (NR, enhancement, spectral repair) can delegate
    their main loop here.

    Args:
        phase_fn:          Callable(audio_chunk, **phase_kwargs) → np.ndarray
        audio:             Full audio array (1D or 2D [channels, samples]).
        sr:                Sample rate (must be 48000 for processing phases).
        max_severity:      Highest relevant defect severity (0.0–1.0).
        phase_kwargs:      Extra keyword arguments forwarded to *phase_fn*.
        crossfade_s:       Crossfade duration in seconds (default 10 ms).
        beat_sync_chunks:  If True, snap chunk boundaries to detected beat
                           positions (±500 ms tolerance). Uses madmom
                           RNNBeatProcessor (Böck et al. 2016) with librosa
                           fallback. Default False (backward-compatible).

    Returns:
        ChunkProcessingResult with stitched audio.
    """
    if phase_kwargs is None:
        phase_kwargs = {}

    is_stereo = audio.ndim == 2
    n_samples = audio.shape[-1]
    duration_s = n_samples / sr

    is_silence = _is_near_silence(audio)
    chunk_s = compute_chunk_size_s(max_severity, is_silence=is_silence)

    # Beat detection for boundary snapping (opt-in, only for full tracks > 10 s)
    beat_times_s: list[float] = []
    if beat_sync_chunks and duration_s >= 10.0 and not is_silence:
        audio_mono = audio.mean(axis=0).astype(np.float32) if is_stereo else audio.astype(np.float32)
        beat_times_s = _estimate_beat_times(audio_mono, sr)
        if beat_times_s:
            logger.debug(
                "AdaptiveChunk: beat-sync aktiviert, %d beats erkannt (%.1f–%.1f s)",
                len(beat_times_s),
                beat_times_s[0],
                beat_times_s[-1],
            )
        else:
            logger.debug("AdaptiveChunk: beat-sync requested but no beats erkannt — using fixed boundaries")

    # If audio fits in a single chunk, skip chunking overhead
    if duration_s <= chunk_s + crossfade_s:
        result_audio = phase_fn(audio, **phase_kwargs)
        return ChunkProcessingResult(
            audio=np.asarray(result_audio, dtype=np.float32),
            chunk_size_s=chunk_s,
            n_chunks=1,
        )

    chunk_samples = int(chunk_s * sr)
    fade_samples = max(1, int(crossfade_s * sr))
    hop_samples = max(1, chunk_samples - fade_samples)  # overlap = fade_samples

    # Half-Hanning COLA-compliant crossfade windows (Lücke-E-Fix v10.0.0).
    # w_in[i] = 0.5*(1 - cos(π*i/N))  rising Hanning half → smooth C¹ boundary
    # w_out = 1 - w_in  → w_in + w_out = 1.0 for all i (COLA at 50 % overlap)
    # §9.10.119: float64 intermediate precision eliminates float32 accumulation
    # error (±2–5% amplitude drift at chunk boundaries → audible pumping).
    _t = np.arange(fade_samples, dtype=np.float64) / max(fade_samples, 1)
    fade_in = (0.5 * (1.0 - np.cos(np.pi * _t))).astype(np.float32)
    fade_out = (1.0 - 0.5 * (1.0 - np.cos(np.pi * _t))).astype(np.float32)

    # Output buffer
    if is_stereo:
        out = np.zeros_like(audio, dtype=np.float32)
    else:
        out = np.zeros(n_samples, dtype=np.float32)
    weight = np.zeros(n_samples, dtype=np.float32)

    n_chunks = 0
    pos = 0
    _prev_tail = None  # §v10.116 PHAOLA: previous chunk tail for phase alignment
    while pos < n_samples:
        end = min(pos + chunk_samples, n_samples)
        if is_stereo:
            chunk = audio[:, pos:end].copy()
        else:
            chunk = audio[pos:end].copy()

        # Process chunk
        processed = phase_fn(chunk, **phase_kwargs)
        processed = np.asarray(processed, dtype=np.float32)

        # §GEBOT-G42: Stereo-Lag-Integrität nach jedem Chunk
        # Chunked-Phase-Prozessoren können L/R-Versatz einführen.
        # §v10.13 SOTA: Normalized time-domain cross-correlation on the
        # full chunk (reliable even on 5 s segments, unlike GCC-PHAT).
        # Peak value is Pearson's r [0,1]; threshold 0.1 cleanly
        # separates correlated L/R from uncorrelated noise.
        if is_stereo and processed.ndim == 2 and processed.shape[0] == 2:
            try:
                _ch_l = processed[0].astype(np.float64)
                _ch_r = processed[1].astype(np.float64)
                _ch_n = len(_ch_l)
                if _ch_n >= 1024:
                    # Normalized cross-correlation via scipy (FFT-accelerated)
                    from scipy.signal import correlate as _sp_corr

                    _l_ms = _ch_l - float(np.mean(_ch_l))
                    _r_ms = _ch_r - float(np.mean(_ch_r))
                    _l_std = float(np.std(_l_ms)) + 1e-12
                    _r_std = float(np.std(_r_ms)) + 1e-12
                    _corr = _sp_corr(_l_ms / (_l_std * _ch_n), _r_ms / _r_std, method="fft")
                    _center = _ch_n - 1
                    _max_lag = min(int(sr * 0.2), _center)  # ±200 ms search
                    _lo = max(0, _center - _max_lag)
                    _hi = min(len(_corr), _center + _max_lag + 1)
                    _search = _corr[_lo:_hi]
                    _peak = float(np.max(np.abs(_search)))
                    if _peak >= 0.1:
                        _lag = int(np.argmax(np.abs(_search))) - _max_lag
                        if abs(_lag) > 1:  # ≥1 sample correction threshold
                            from scipy.ndimage import shift as _nd_shift

                            _r_corrected = _nd_shift(
                                _ch_r.astype(np.float64), float(_lag), mode="constant", cval=0.0, order=3
                            )
                            processed = np.vstack([_ch_l[np.newaxis, :], _r_corrected[np.newaxis, :]]).astype(
                                np.float32
                            )
            except Exception as _acp_xcorr_exc:
                logger.debug(
                    "adaptive_chunk_processor: chunk cross-correlation fehlgeschlagen (unkritisch): %s", _acp_xcorr_exc
                )

        # §v10.116 PHAOLA: Phase-aligned overlap-add for mono sources.
        # Align phases between consecutive chunks before crossfading to
        # eliminate comb filtering at low frequencies (30 Hz audible modulation).
        # Skips if ERB masking threshold renders alignment unnecessary.
        _this_chunk_len = processed.shape[-1] if is_stereo else len(processed)
        if pos > 0 and not is_stereo and fade_samples < _this_chunk_len and _prev_tail is not None:
            try:
                from backend.core.phase_aligned_overlap_add import (
                    compute_optimal_alignment,
                    should_skip_alignment,
                )

                _overlap_region = processed[:fade_samples]
                if not should_skip_alignment(_overlap_region, sr):
                    _align = compute_optimal_alignment(_prev_tail, processed, fade_samples, sr)
                    if _align.was_adjusted:
                        from scipy.ndimage import shift as _nd_shift

                        processed = _nd_shift(
                            processed.astype(np.float64),
                            float(_align.shift_samples),
                            mode="constant",
                            cval=0.0,
                            order=3,
                        ).astype(np.float32)
                        logger.debug(
                            "PHAOLA: phase-aligned by %.2f samples (corr=%.3f)",
                            _align.shift_samples,
                            _align.correlation,
                        )
            except Exception as _phaola_exc:
                logger.debug("PHAOLA: nicht verfügbar (%s)", _phaola_exc)

        # Store current chunk tail for next iteration's alignment
        if not is_stereo and len(processed) >= fade_samples:
            _prev_tail = processed[-fade_samples:].copy()

        # Build weight envelope for this chunk
        chunk_len = end - pos
        w = np.ones(chunk_len, dtype=np.float32)
        # Fade-in (except first chunk)
        if pos > 0 and fade_samples < chunk_len:
            w[:fade_samples] = fade_in[:fade_samples]
        # Fade-out (except last chunk)
        if end < n_samples and fade_samples < chunk_len:
            w[-fade_samples:] = fade_out[:fade_samples]

        # Accumulate
        if is_stereo:
            for ch in range(processed.shape[0]):
                p_len = min(processed.shape[1], chunk_len)
                out[ch, pos : pos + p_len] += processed[ch, :p_len] * w[:p_len]
        else:
            p_len = min(len(processed), chunk_len)
            out[pos : pos + p_len] += processed[:p_len] * w[:p_len]
        weight[pos : pos + chunk_len] += w

        n_chunks += 1
        next_pos = pos + hop_samples
        # §7.6a Chunk-Boundary-Transient-Guard (spec Y3): shift boundary away from
        # transient onsets detected within ±20 ms of the candidate position.
        next_pos = _find_safe_boundary(next_pos, audio, sr)
        if beat_times_s:
            # Snap next boundary to nearest beat within tolerance.
            # min_samples ensures we always advance by at least 1 sample.
            next_pos = _snap_to_beat(
                next_pos,
                sr,
                beat_times_s,
                tol_s=_BEAT_SNAP_TOL_S,
                min_samples=pos + 1,
                max_samples=n_samples,
            )
        pos = next_pos

    # Normalize by accumulated weight (avoid division by zero)
    weight = np.maximum(weight, 1e-8)
    if is_stereo:
        out /= weight[np.newaxis, :]
    else:
        out /= weight

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)

    logger.debug(
        "§7.6 AdaptiveChunk: severity=%.2f → chunk=%.1fs, n_chunks=%d, crossfade=%.0fms",
        max_severity,
        chunk_s,
        n_chunks,
        crossfade_s * 1000,
    )

    return ChunkProcessingResult(audio=out, chunk_size_s=chunk_s, n_chunks=n_chunks)


# ---------------------------------------------------------------------------
# Thread-safe singleton (§3.2)
# ---------------------------------------------------------------------------

_instance: AdaptiveChunkProcessor | None = None
_lock = threading.Lock()


class AdaptiveChunkProcessor:
    """Singleton-Wrapper für adaptive Chunk-Verarbeitung."""

    def compute_chunk_size(self, max_severity: float, is_silence: bool = False) -> float:
        return compute_chunk_size_s(max_severity, is_silence=is_silence)

    def process(self, phase_fn, audio, sr, max_severity, **kwargs):
        return process_in_adaptive_chunks(phase_fn, audio, sr, max_severity, **kwargs)


def get_adaptive_chunk_processor() -> AdaptiveChunkProcessor:
    """Thread-safe singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AdaptiveChunkProcessor()
    return _instance
