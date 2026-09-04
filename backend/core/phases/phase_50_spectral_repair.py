"""
Phase 50: Spectral Repair v2.0 — STFT Inpainting
==================================================

Vollständige DSP-Implementierung ohne aurik_ml.
Ersetzt den kaputten ML-Stub.

ALGORITHMUS — Spektrales Inpainting:
  1. STFT: Hann-Fenster 2048, Hop 512
  2. Beschädigte Bin-Erkennung:
     - Vertikale Dimension (Zeit-Transient): |X[f,t]| > α × median(|X[f, t±K]|)
       → Transiente Peaks, kein Reparatur-Bedarf
     - Horizontale Dimension (Frequenz-Spitze): |X[f,t]| > β × median(|X[f±B, t]|)
       → Isolierter Bin-Spike → Reparatur notwendig
  3. Reparatur: Lineare Interpolation aus ±2 Nachbar-Bins
  4. ISTFT mit Overlap-Add, Normierung

METRIKEN:
  - n_repaired_bins: Anzahl reparierter Bins
  - repair_ratio: n_repaired / gesamt
  - snr_improvement_db: Approximierter SNR-Gewinn

Author: Aurik Development Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import to_channels_last
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# STFT-Parameter
_FFT_SIZE = 2048
_HOP = 512
_WIN = "hann"

# Detektions-Schwellwerte
_NEIGHBOR_BINS = 5  # ±B Bins für Frequenz-Nachbarschaft
_THRESHOLD_FACTOR = 4.0  # Bin > FACTOR × Median der Nachbarn → verdächtig
_INTERP_BINS = 2  # ±K Bins für Interpolation


def _repair_channel(
    channel: np.ndarray,
    sample_rate: int,
    threshold_factor: float,
    hf_protected_bin_start: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Spektrale Reparatur eines Mono-Kanals: Frequenz-Achse (Spikes) + Zeit-Achse (Scratches).

    Gibt (repaired, n_repaired_bins) zurück.

    v2.1 (upgraded):
      Pass 1 — Frequency-axis spike detection (unchanged from v2.0):
        Horizontal spikes: |X[f,t]| > FACTOR × median(|X[f±B, t]|) → interpolate

      Pass 2 — Time-axis damage detection (NEW):
        Vertical energy drops: frames where mean(|X[:, t]|) < DROP_FACTOR × median_frame_energy
        → Time-axis linear interpolation from neighbouring frames
        These correspond to scratches (short-duration broadband damage) and dropouts
        (complete signal absence in a frame).

    Args:
        channel:               Mono audio (1D float array).
        sample_rate:           Sample rate (Hz).
        threshold_factor:      Spike threshold — magnitude ratio to smooth envelope.
        hf_protected_bin_start: Frequency-bin index above which Pass 1 spike detection is
                                 disabled.  0 = no protection (default).

    §PriorPhase-Guard (phase_07/phase_06 harmonics):
        When a prior phase (phase_07 harmonic restoration, phase_06 SBR) adds content
        at frequencies above the material's natural rolloff, those bins look like
        "isolated spikes" to Pass 1 because the 11-bin smooth envelope is still near
        the noise floor (only 1 bin elevated out of 11).  Pass 1 would flag and remove
        them — reverting the prior restoration.

        Fix: the caller passes hf_protected_bin_start = bin_index(material_rolloff × 0.85).
        Bins above that index are excluded from Pass 1.  Pass 2 (frame energy dropout)
        remains active everywhere: it works on whole-frame RMS, not per-bin ratios, and
        is insensitive to isolated harmonic additions.
    """
    from scipy.ndimage import uniform_filter1d  # pylint: disable=import-outside-toplevel

    _f, _t, Zxx = sig.stft(
        channel,
        fs=sample_rate,
        window=_WIN,
        nperseg=_FFT_SIZE,
        noverlap=_FFT_SIZE - _HOP,
    )

    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    _n_freq, _n_time = mag.shape

    # ── PASS 1: Frequency-axis spike repair ────────────────────────────────────
    _filter_size = 2 * _NEIGHBOR_BINS + 1
    mag_smooth = uniform_filter1d(mag, size=_filter_size, axis=0, mode="nearest")
    safe_smooth = np.where(mag_smooth < 1e-10, 1e-10, mag_smooth)
    spike_mask = (mag > threshold_factor * safe_smooth) & (mag >= 1e-10)

    # §PriorPhase-Guard: Exclude HF bins above material rolloff from Pass 1.
    # Phase_07 (harmonic restoration) synthesises content at frequencies that were
    # previously near noise floor.  Those isolated peaks trigger the spike_mask
    # (1 bin elevated / 11-bin window → ratio ≈ 11 >> threshold_factor).
    # Pass 2 (frame dropout) is NOT masked — it is based on whole-frame RMS,
    # not per-bin ratio, and does not produce false positives on HF harmonics.
    if hf_protected_bin_start > 0:
        hf_start = min(hf_protected_bin_start, _n_freq)
        spike_mask[hf_start:, :] = False

    mag_lo = np.roll(mag, _INTERP_BINS, axis=0)
    mag_hi = np.roll(mag, -_INTERP_BINS, axis=0)
    mag_interp = (mag_lo + mag_hi) * 0.5
    mag_rep = np.where(spike_mask, mag_interp, mag)
    n_freq_repaired = int(np.count_nonzero(spike_mask))

    # ── PASS 2: Time-axis damage detection (scratch/dropout frames) ────────────
    # Measure per-frame mean energy
    frame_energy = np.mean(mag_rep, axis=0)  # shape: (n_time,)
    # Robust median via sorted percentile (avoid scipy.ndimage for 1D)
    median_energy = float(np.median(frame_energy))
    _DROP_FACTOR = 0.15  # Frame energy < 15% of median → damaged (dropout/scratch)

    damaged_frames = frame_energy < (_DROP_FACTOR * median_energy + 1e-14)
    n_time_repaired = int(np.sum(damaged_frames))

    if n_time_repaired > 0 and n_time_repaired < _n_time * 0.5:
        # Only repair if <50% of frames damaged (full silence passes through intact)
        # ── Initial seed: linear interpolation between nearest undamaged frames ──
        for t_idx in np.where(damaged_frames)[0]:
            # Find nearest undamaged neighbours on each side
            lo = t_idx - 1
            hi = t_idx + 1
            while lo >= 0 and damaged_frames[lo]:
                lo -= 1
            while hi < _n_time and damaged_frames[hi]:
                hi += 1
            if lo >= 0 and hi < _n_time:
                # Linear interpolation weight based on distance
                d_total = hi - lo
                w_hi = (t_idx - lo) / d_total
                w_lo = (hi - t_idx) / d_total
                mag_rep[:, t_idx] = w_lo * mag_rep[:, lo] + w_hi * mag_rep[:, hi]
            elif lo >= 0:
                mag_rep[:, t_idx] = mag_rep[:, lo]
            elif hi < _n_time:
                mag_rep[:, t_idx] = mag_rep[:, hi]

        # ── STFT-Consistency projection (Siedenburg & Dörfler 2013, JASA) ────────
        # Iteratively enforce time-frequency consistency for the inpainted frames
        # while keeping undamaged frames anchored to their original values.
        # This propagates spectral structure from known frames into gaps, exploiting
        # the redundancy of the STFT frame (each sample contributes to multiple frames).
        _N_CONSISTENCY_ITER = 5
        _known_mask = ~damaged_frames  # (n_time,) — True = undamaged
        _mag_anchor = mag[:, _known_mask].copy()  # original magnitudes for known frames
        _ph_anchor = phase[:, _known_mask].copy()  # original phases for known frames

        for _ci in range(_N_CONSISTENCY_ITER):
            # Reconstruct time-domain signal from current spectrogram estimate
            _Zxx_ci = mag_rep * np.exp(1j * phase)
            try:
                _, _td_ci = sig.istft(
                    _Zxx_ci,
                    fs=sample_rate,
                    window=_WIN,
                    nperseg=_FFT_SIZE,
                    noverlap=_FFT_SIZE - _HOP,
                )
                _n_need = len(channel)
                if len(_td_ci) >= _n_need:
                    _td_ci = _td_ci[:_n_need]
                else:
                    _td_ci = np.pad(_td_ci, (0, _n_need - len(_td_ci)))
                # Re-STFT — updated phase arises naturally from consistent TD signal
                _, _, _Zxx_new = sig.stft(
                    _td_ci,
                    fs=sample_rate,
                    window=_WIN,
                    nperseg=_FFT_SIZE,
                    noverlap=_FFT_SIZE - _HOP,
                )
                _mag_new = np.abs(_Zxx_new)
                _ph_new = np.angle(_Zxx_new)
                # Trim / pad to original STFT dimensions
                if _mag_new.shape[1] < _n_time:
                    _mag_new = np.pad(_mag_new, ((0, 0), (0, _n_time - _mag_new.shape[1])), mode="edge")
                    _ph_new = np.pad(_ph_new, ((0, 0), (0, _n_time - _ph_new.shape[1])), mode="edge")
                mag_rep = _mag_new[:_n_freq, :_n_time]
                phase = _ph_new[:_n_freq, :_n_time]
                # Projection step: re-impose known-frame constraint (POCS)
                mag_rep[:, _known_mask] = _mag_anchor
                phase[:, _known_mask] = _ph_anchor
            except Exception:
                logger.warning(
                    "§V6 Phase-50 POCS consistency iteration failed → break (keep current estimate): %s",
                    str(Exception),
                )
                break  # Consistency iteration failed; keep current estimate

    n_repaired = n_freq_repaired + n_time_repaired

    # ── Reconstruct ────────────────────────────────────────────────────────────
    Zxx_rep = mag_rep * np.exp(1j * phase)
    _, repaired = sig.istft(
        Zxx_rep,
        fs=sample_rate,
        window=_WIN,
        nperseg=_FFT_SIZE,
        noverlap=_FFT_SIZE - _HOP,
    )
    out_len = len(channel)
    repaired = repaired[:out_len] if len(repaired) >= out_len else np.pad(repaired, (0, out_len - len(repaired)))

    return repaired.astype(channel.dtype), n_repaired


class SpectralRepairPhase(PhaseInterface):
    """STFT-basiertes Spectral Inpainting ohne ML-Dependency."""

    _PHASE_ID = "phase_50_spectral_repair"
    _NAME = "Spectral Repair (STFT Inpainting)"
    description = (
        "Erkennt und repariert isolierte spektrale Bin-Spikes (Artefakte, Kratzer, "
        "Datenfehler) mittels STFT-Analyse. "
        "Pass 1: Frequenz-Achsen-Spike-Reparatur per Nachbar-Bin-Interpolation. "
        "Pass 2: Zeit-Achsen-Dropout-Reparatur via iterativer STFT-Konsistenz-Projektion "
        "(Siedenburg & Dörfler 2013). "
        "Kein ML, volle DSP-Transparenz."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=7,
            version="2.1.0",
            dependencies=[],
            estimated_time_factor=0.10,
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.88,
            description=self.description,
        )

    @staticmethod
    def _compute_threshold_runtime_profile(
        material_key: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive threshold profile for spectral repair."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("shellac", "wax_cylinder", "wire_recording", "lacquer_disc")):
            strength_floor = 0.08
            side_multiplier = 1.72
        elif any(token in _material for token in ("vinyl", "tape", "reel_tape", "cassette")):
            strength_floor = 0.10
            side_multiplier = 1.90
        else:
            strength_floor = 0.14
            side_multiplier = 2.18

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        strength_floor += (_rest_norm - 0.5) * 0.08
        side_multiplier += (_rest_norm - 0.5) * 0.40

        _mode_offsets = {
            "fast": (0.03, 0.18),
            "balanced": (0.00, 0.00),
            "quality": (0.00, 0.00),
            "maximum": (-0.02, -0.12),
        }
        _strength_off, _side_off = _mode_offsets.get(_mode, (0.0, 0.0))
        strength_floor += _strength_off
        side_multiplier += _side_off

        return {
            "strength_floor": float(np.clip(strength_floor, 0.06, 0.18)),
            "side_multiplier": float(np.clip(side_multiplier, 1.60, 2.40)),
        }

    @staticmethod
    def _local_event_strength(key: str, loc: tuple[float, float], event_metadata: dict[str, dict] | None) -> float:
        duration_s = max(0.0, float(loc[1]) - float(loc[0]))
        duration_factor = float(np.clip(duration_s / 0.42, 0.30, 1.0))
        key_factor = {
            "pre_echo": 0.90,
            "aliasing": 0.86,
            "codec_artifact": 0.78,
            "mp3_artifact": 0.78,
            "spectral_spike": 1.0,
            "spectral_gap": 0.84,
            "spectral_hole": 0.84,
            "digital_glitch": 0.92,
            "dropout": 0.70,
            "jitter_artifacts": 0.74,
        }.get(key, 0.66)
        severity = 0.55
        confidence = 0.78
        meta_obj = (event_metadata or {}).get(key)
        if isinstance(meta_obj, dict):
            severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
            confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
        return float(np.clip(key_factor * (0.34 + 0.46 * severity + 0.20 * confidence) * duration_factor, 0.12, 1.0))

    @staticmethod
    def _collect_protected_zones(kwargs: dict) -> list[tuple[float, float, float]]:
        zones: list[tuple[float, float, float]] = []
        for key, cap in (
            ("vibrato_zones", 0.20),
            ("frisson_zones", 0.30),
            ("whisper_zones", 0.25),
            ("passaggio_zones", 0.35),
        ):
            for zone in kwargs.get(key) or []:
                try:
                    start_s = float(getattr(zone, "start_s", None) or zone[0])
                    end_s = float(getattr(zone, "end_s", None) or zone[1])
                    if end_s > start_s:
                        zones.append((start_s, end_s, cap))
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
        return zones

    @staticmethod
    def _build_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
        event_metadata: dict[str, dict] | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, float]:
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0

        keys = (
            "pre_echo",
            "aliasing",
            "codec_artifact",
            "mp3_artifact",
            "spectral_spike",
            "spectral_gap",
            "spectral_hole",
            "digital_glitch",
            "dropout",
            "jitter_artifacts",
        )
        pad_by_key = {
            "pre_echo": 0.020,
            "aliasing": 0.030,
            "codec_artifact": 0.045,
            "mp3_artifact": 0.045,
            "spectral_spike": 0.018,
            "spectral_gap": 0.055,
            "spectral_hole": 0.055,
            "digital_glitch": 0.020,
            "dropout": 0.060,
            "jitter_artifacts": 0.030,
        }
        mask = np.zeros(n_samples, dtype=np.float32)
        for key in keys:
            pad = int(pad_by_key.get(key, 0.035) * sample_rate)
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start = int(max(0.0, float(loc[0])) * sample_rate)
                    end = int(max(0.0, float(loc[1])) * sample_rate)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
                if end <= start:
                    continue
                start = max(0, start - pad)
                end = min(n_samples, end + pad)
                if end > start:
                    strength = SpectralRepairPhase._local_event_strength(key, loc, event_metadata)
                    mask[start:end] = np.maximum(mask[start:end], strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0

        smooth = max(16, int(0.018 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if protected_zones:
            for start_s, end_s, cap in protected_zones:
                start = int(max(0.0, float(start_s)) * sample_rate)
                end = int(max(0.0, float(end_s)) * sample_rate)
                if end > start:
                    mask[start : min(n_samples, end)] = np.minimum(mask[start : min(n_samples, end)], float(cap))
        return mask, float(np.mean(mask))

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:  # type: ignore[override]  # pylint: disable=arguments-differ
        """
        Repariert spektrale Artefakte via STFT Inpainting.

        Args:
            audio:        Mono oder Stereo
            sample_rate:  Abtastrate Hz
            **kwargs:     threshold_factor (float, default 4.0)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        # ── §v10 PIM: Per-Band-Intensität kalibrieren ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim = apply_pim_intensity(
                kwargs, "spectral_repair2", default_nr=0.45, default_de_ess=0.25, default_comp=1.0
            )
            if kwargs.get("pim_intensity_map") is not None:
                for _key in ("noise_reduction_strength", "nr_strength", "strength", "wet"):
                    if _key in kwargs:
                        kwargs[_key] = _pim["nr_strength"]
        except Exception as e:
            logger.warning("Verarbeitungsschritt_50_spectral_repair.py::verarbeiten Ersatzpfad: %s", e)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p50_transposed = to_channels_last(audio)
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_50 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_50 >= 0.25 and effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_50,
                )

                _fmz_50 = kwargs.get("forward_masking_zones") or _fmg_fn_50().compute_zones(audio, sample_rate)
                if _fmz_50:
                    _n_s_50 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_50 = sum(z.end_sample - z.start_sample for z in _fmz_50)
                    _zone_frac_50 = float(np.clip(_zone_s_50 / max(1, _n_s_50), 0.0, 1.0))
                    effective_strength = float(np.clip(effective_strength + _zone_frac_50 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_50:
                logger.debug("Verarbeitungsschritt50 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_50)

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        _mat_50 = kwargs.get("material_type") or kwargs.get("material")
        _mat_str_50 = str(_mat_50).lower() if _mat_50 is not None else ""
        threshold_factor = float(kwargs.get("threshold_factor", _THRESHOLD_FACTOR))
        _runtime_profile_50 = self._compute_threshold_runtime_profile(
            _mat_str_50,
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        # §2.54 Material-adaptive threshold: degraded analog needs more aggressive repair
        # (lower threshold_factor = more bins detected as holes = more repairs).
        _MATERIAL_THRESHOLD_CAPS_50: dict[str, float] = {
            "wax_cylinder": 2.0,
            "shellac": 2.5,
            "optical_film": 2.5,
            "wire_recording": 2.5,
            "vinyl": 3.0,
            "reel_tape": 3.0,
            "tape": 3.0,
            "cassette": 3.0,
            "radio_broadcast": 3.5,
            "mp3_low": 3.5,
            "minidisc": 3.5,
            "mp3_high": 4.5,
            "cd_digital": 4.5,
            "dat": 4.5,
        }
        for _mk50, _mv50 in _MATERIAL_THRESHOLD_CAPS_50.items():
            if _mk50 in _mat_str_50:
                threshold_factor = min(threshold_factor, _mv50)
                break
        # Lower effective strength should make detection more conservative.
        threshold_factor_eff = threshold_factor / max(effective_strength, _runtime_profile_50["strength_floor"])

        # §PriorPhase-Guard: Protect HF bins from Pass-1 false-positive spike detection.
        # Phase_07 (harmonic restoration) and Phase_06 (SBR) synthesise content at
        # frequencies that were near noise floor before they ran.  The 11-bin local-average
        # smooth envelope is still near noise floor at those frequencies → ratio ≫ 4.0 →
        # Pass 1 would flag and inpaint Phase_07/06 output, reverting the restoration.
        # Fix: exclude bins above 85 % of the material's natural rolloff from Pass 1.
        # (All codec-artifact spikes occur WITHIN the natural bandwidth, so Pass 1 still
        # catches them.  Phase_07/06 harmonics are exclusively ABOVE the rolloff.)
        _ANALOG_ROLLOFF_PROTECTION_HZ: dict[str, float] = {
            "wax_cylinder": 5_000.0,
            "wire_recording": 6_000.0,
            "shellac": 7_000.0,
            "lacquer_disc": 8_000.0,
            "cassette": 13_000.0,
            "vinyl": 14_000.0,
            "tape": 14_000.0,
            "reel_tape": 16_000.0,
        }
        _bin_hz = sample_rate / _FFT_SIZE  # Hz per STFT bin
        _rolloff_protection_hz = _ANALOG_ROLLOFF_PROTECTION_HZ.get(_mat_str_50, 0.0)
        # Also check prefix matching for compound material strings
        if _rolloff_protection_hz == 0.0:
            for _k, _v in _ANALOG_ROLLOFF_PROTECTION_HZ.items():
                if _k in _mat_str_50:
                    _rolloff_protection_hz = _v
                    break
        _hf_protected_bin_start: int = 0
        if _rolloff_protection_hz > 0.0:
            _hf_protected_bin_start = max(0, int(_rolloff_protection_hz * 0.85 / _bin_hz))

        # §4.11 Pre-Echo Repair: MUSS vor generischer STFT-Reparatur laufen.
        # Pre-Echo ist kein stationäres Rauschen — frame-selektive Spektral-Dämpfung
        # im Prä-Masking-Fenster (Spec 04 §4.11, Fastl & Zwicker 2007 §7.2).
        _pre_echo_events_50 = list(kwargs.get("pre_echo_events", []))
        if _pre_echo_events_50:
            try:
                from backend.core.dsp.pre_echo_detector import (  # pylint: disable=import-outside-toplevel
                    get_pre_echo_detector as _get_ped50,
                )

                _ped50 = _get_ped50()
                _audio_pre_pre_echo = audio.copy()
                # §V38 VFA-Schutzzonen für per-Event-Blend-Cap (Vibrato 0.20, Frisson 0.30 etc.)
                _p50_vfa_zones: list[tuple[float, float, float]] = []

                def _append_vfa_zone_50(_zone: object, _cap: float) -> None:
                    try:
                        if isinstance(_zone, dict):
                            _start = float(_zone.get("start_s", _zone.get("start", 0.0)) or 0.0)
                            _end = float(_zone.get("end_s", _zone.get("end", 0.0)) or 0.0)
                        else:
                            _start = float(_zone[0])  # type: ignore[index]
                            _end = float(_zone[1])  # type: ignore[index]
                        if _end > _start:
                            _p50_vfa_zones.append((_start, _end, _cap))
                    except (TypeError, ValueError, IndexError):
                        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

                _p50_vfa = kwargs.get("vfa_result") or {}
                if isinstance(_p50_vfa, dict):
                    for _z in _p50_vfa.get("vibrato_zones", []):
                        _append_vfa_zone_50(_z, 0.20)
                    for _z in _p50_vfa.get("frisson_zones", []):
                        _append_vfa_zone_50(_z, 0.30)
                    for _z in _p50_vfa.get("whisper_zones", []):
                        _append_vfa_zone_50(_z, 0.25)
                    for _z in _p50_vfa.get("passaggio_zones", []):
                        _append_vfa_zone_50(_z, 0.35)
                for _z in kwargs.get("vibrato_zones", []) or []:
                    _append_vfa_zone_50(_z, 0.20)
                for _z in kwargs.get("frisson_zones", []) or []:
                    _append_vfa_zone_50(_z, 0.30)
                for _z in kwargs.get("whisper_zones", []) or []:
                    _append_vfa_zone_50(_z, 0.25)
                for _z in kwargs.get("passaggio_zones", []) or []:
                    _append_vfa_zone_50(_z, 0.35)
                for _evt50 in _pre_echo_events_50:
                    _audio_evt_pre50 = audio.copy() if _p50_vfa_zones else None
                    audio = _ped50.repair_region(audio, _evt50, sample_rate)
                    # §V38 In VFA-Schutzzone: Repair-Stärke auf Zone-Cap blend-limitieren
                    if _p50_vfa_zones and _audio_evt_pre50 is not None:
                        _pec_s_t50 = float(_evt50.get("pre_echo_start", 0)) / sample_rate
                        _pec_e_t50 = float(_evt50.get("pre_echo_end", _evt50.get("onset_sample", 0))) / sample_rate
                        for _pz_s50, _pz_e50, _pz_cap50 in _p50_vfa_zones:
                            if _pec_s_t50 < _pz_e50 and _pec_e_t50 > _pz_s50:
                                audio = _audio_evt_pre50 * (1.0 - _pz_cap50) + audio * _pz_cap50
                                break
                logger.info(
                    "Verarbeitungsschritt50 §4.11 pre_echo_repair: n_events=%d, material=%s",
                    len(_pre_echo_events_50),
                    _mat_str_50,
                )
                # §2.46e Sicherheits-Guard: kein Over-Processing durch Pre-Echo-Repair
                _pre_echo_diff_rms = float(
                    np.sqrt(np.mean((audio.astype(np.float64) - _audio_pre_pre_echo.astype(np.float64)) ** 2) + 1e-12)
                )
                _pre_echo_sig_rms = float(np.sqrt(np.mean(_audio_pre_pre_echo.astype(np.float64) ** 2) + 1e-12))
                if _pre_echo_diff_rms > _pre_echo_sig_rms * 0.25:
                    logger.warning(
                        "Verarbeitungsschritt50 pre_echo_repair over-processing guard: diff_rms/sig_rms=%.3f > 0.25, rollback",
                        _pre_echo_diff_rms / _pre_echo_sig_rms,
                    )
                    audio = _audio_pre_pre_echo
            except Exception as _ped50_exc:
                logger.debug("Verarbeitungsschritt50 §4.11 pre_echo_repair nicht blockierend: %s", _ped50_exc)

        n_channels = 1 if audio.ndim == 1 else audio.shape[1]
        is_stereo = audio.ndim == 2
        total_bins = 0

        if not is_stereo:
            repaired_c, n_rep = _repair_channel(
                audio.astype(np.float64), sample_rate, threshold_factor_eff, _hf_protected_bin_start
            )
            repaired_audio = repaired_c.astype(audio.dtype)
            total_bins = n_rep
        else:
            # §2.51: M/S domain — repair Mid fully, Side conservatively
            _inv_sqrt2 = 1.0 / np.sqrt(2.0)
            mid = (audio[:, 0] + audio[:, 1]).astype(np.float64) * _inv_sqrt2
            side = (audio[:, 0] - audio[:, 1]).astype(np.float64) * _inv_sqrt2
            repaired_mid, n_mid = _repair_channel(mid, sample_rate, threshold_factor_eff, _hf_protected_bin_start)
            repaired_side, n_side = _repair_channel(
                side,
                sample_rate,
                threshold_factor_eff * _runtime_profile_50["side_multiplier"],
                _hf_protected_bin_start,
            )
            total_bins = n_mid + n_side
            left = (repaired_mid + repaired_side) * _inv_sqrt2
            right = (repaired_mid - repaired_side) * _inv_sqrt2
            repaired_audio = np.column_stack([left, right]).astype(audio.dtype)

        if 0.0 < effective_strength < 1.0:
            repaired_audio = audio + effective_strength * (repaired_audio - audio)

        _local_profile50, _local_coverage50 = self._build_locality_profile(
            int(audio.shape[0]),
            sample_rate,
            kwargs.get("defect_locations"),
            kwargs.get("defect_event_metadata"),
            self._collect_protected_zones(kwargs),
        )
        if _local_coverage50 > 0.0:
            _local_wet50 = _local_profile50[:, np.newaxis] if repaired_audio.ndim == 2 else _local_profile50
            repaired_audio = audio + _local_wet50 * (repaired_audio - audio)

        total_bins_possible = int((_FFT_SIZE // 2 + 1) * (len(audio) // _HOP + 1) * n_channels)
        repair_ratio = total_bins / max(1, total_bins_possible)

        # Approximierter SNR-Gewinn
        diff = audio.astype(np.float64) - repaired_audio.astype(np.float64)
        sig_power = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
        noise_power = float(np.mean(diff**2)) + 1e-12
        snr_before = 10.0 * np.log10(sig_power / noise_power)

        logger.info(
            "Verarbeitungsschritt 50 SpectralRepair: n_repaired=%d, repair_Verhaeltnis=%.5f",
            total_bins,
            repair_ratio,
        )

        # §4.5 Psychoacoustic Masking Clamp — only repair where audible (§0 Primum non nocere)
        try:
            # pylint: disable-next=import-outside-toplevel
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp

            repaired_audio = apply_psychoacoustic_masking_clamp(
                audio,
                repaired_audio,
                sample_rate,
                strength=effective_strength,
                mode="additive",
            )
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt50 masking clamp nicht blockierend: %s", _pm_exc)

        repaired_audio = np.nan_to_num(repaired_audio, nan=0.0, posinf=0.0, neginf=0.0)
        repaired_audio = np.clip(repaired_audio, -1.0, 1.0)

        # §2.46e Hallucination-Guard: Additive Spectral-Reparatur kann Energie über Ceiling hinzufügen
        try:
            # pylint: disable-next=import-outside-toplevel
            from backend.core.dsp.hallucination_guard import check_hallucination as _check_hg50

            _material_50 = kwargs.get("material_type", "unknown")
            _mono_50 = (
                repaired_audio.mean(axis=0)
                if (repaired_audio.ndim == 2 and repaired_audio.shape[0] == 2 and repaired_audio.shape[1] > 2)
                else (repaired_audio.mean(axis=1) if repaired_audio.ndim == 2 else repaired_audio)
            )
            _audio_mono_50 = (
                audio.mean(axis=0)
                if (audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2)
                else (audio.mean(axis=1) if audio.ndim == 2 else audio)
            )
            _bw_ceiling_50 = {
                "shellac": 8000.0,
                "wax_cylinder": 5000.0,
                "vinyl": 16000.0,
                "reel_tape": 18000.0,
                "cassette": 15000.0,
            }.get(str(_material_50).lower().replace(" ", "_"))
            _hg_result50 = _check_hg50(
                _audio_mono_50.astype(np.float32),
                _mono_50.astype(np.float32),
                sr=sample_rate,
                material_bw_ceiling_hz=_bw_ceiling_50,
                mode="restoration",
                bw_extension_context=True,
            )
            if _hg_result50.requires_rollback:
                logger.warning(
                    "§2.46e Verarbeitungsschritt_50 Hallucination-Guard rollback: spectral_novelty=%.3f",
                    _hg_result50.spectral_novelty,
                )
                repaired_audio = audio.copy()
            if _hg_result50.score_penalty > 0:
                logger.info(
                    "§2.46e Verarbeitungsschritt_50 Wert_penalty=%.1f (spectral_novelty=%.3f)",
                    _hg_result50.score_penalty,
                    _hg_result50.spectral_novelty,
                )
        except Exception as _hg50_exc:
            logger.debug("§2.46e Verarbeitungsschritt_50 Hallucination-Guard (nicht blockierend): %s", _hg50_exc)

        # §0p HNR-Blend nach STFT-Inpainting (RELEASE_MUST §0p): ΔHNR > 3 dB → Dry-Wet-Blend
        _p50_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if _p50_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import apply_hnr_blend as _apply_hnr_p50  # pylint: disable=import-outside-toplevel  # noqa: I001

                _hnr_blended_p50, _hnr_diag_p50 = _apply_hnr_p50(
                    audio.astype(np.float32), repaired_audio.astype(np.float32), sample_rate
                )
                if _hnr_diag_p50.get("over_cleaned"):
                    repaired_audio = _hnr_blended_p50
            except Exception as _hnr_exc_p50:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_50 (nicht blockierend): %s", _hnr_exc_p50)

        # §V19 (Spec-Vintage-Guard)/V20/V21/V26/§2.72 Vokal- + Textur-Guards nach STFT-Inpainting (RELEASE_MUST §0p V19-V26)
        _mat50_guards = str(kwargs.get("material_type", _material_50) or "unknown").lower()
        _nt50_residual = audio - repaired_audio
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt50_dist_fn,
            )

            if _nt50_residual.shape == audio.shape:
                _nt50_d = _nt50_dist_fn(_nt50_residual, _mat50_guards, sr=sample_rate)
                if _nt50_d > 0.25:
                    repaired_audio = (0.5 * repaired_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_50: noise_texture_dist=%.3f > 0.25 → 50%% dry-blend",
                        _nt50_d,
                    )
        except Exception as _nt50_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_50 noise_texture nicht blockierend: %s", _nt50_exc
            )

        if _p50_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (  # pylint: disable=import-outside-toplevel
                    frame_energy_correlation as _fec50,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr50 = _fec50(audio, repaired_audio, sample_rate, frame_ms=10.0)
                if _corr50 < 0.97:
                    _need50 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    mikrodynamik_wet_scalar: float = float(_recommend_mkk_wet(_corr50, _p50_panns, global_need=_need50))
                    repaired_audio = (
                        mikrodynamik_wet_scalar * repaired_audio + (1.0 - mikrodynamik_wet_scalar) * audio
                    ).astype(np.float32)
                    logger.warning(
                        "§V20 Verarbeitungsschritt_50: mikrodynamik_corr=%.4f < 0.97 → wet=%.3f",
                        _corr50,
                        mikrodynamik_wet_scalar,
                    )
            except Exception as _v20_50_exc:
                logger.debug("§V20 Verarbeitungsschritt_50 mikrodynamik nicht blockierend: %s", _v20_50_exc)

        if any(x in _mat50_guards for x in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (  # pylint: disable=import-outside-toplevel
                    apply_noise_floor_minimum as _nfmin50,
                )

                repaired_audio = _nfmin50(repaired_audio, sample_rate, _mat50_guards, original_audio=audio)
            except Exception as _v21_50_exc:
                logger.debug("§V21 Verarbeitungsschritt_50 noise_floor nicht blockierend: %s", _v21_50_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_50,
            )

            _sc_result_50 = _scg_50(audio, repaired_audio, sample_rate)
            if not _sc_result_50.ok:
                _sc_wet_50 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                repaired_audio = (_sc_wet_50 * repaired_audio + (1.0 - _sc_wet_50) * audio).astype(np.float32)
        except Exception as _sc_exc_50:  # pylint: disable=broad-except
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_50 spectral_color nicht blockierend: %s", _sc_exc_50
            )

        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opm50,
            )

            repaired_audio = _opm50(audio, repaired_audio, None, max_delta_db=1.5)
        except Exception as _v26_50_exc:
            logger.debug("§V26 Verarbeitungsschritt_50 onset_guard nicht blockierend: %s", _v26_50_exc)

        if _p50_panns >= 0.25:
            try:
                from backend.core.dsp.vibrato_guard import (  # pylint: disable=import-outside-toplevel
                    check_vibrato_depth_preservation as _vib50_fn,
                )

                _vibr50 = _vib50_fn(audio, repaired_audio, sample_rate)
                if not _vibr50.ok:
                    repaired_audio = (0.5 * repaired_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§2.72 Verarbeitungsschritt_50: vibrato_reduction=%.1f%% → 50%% dry-blend",
                        _vibr50.depth_reduction_pct,
                    )
            except Exception as _vib50_exc:
                logger.debug("§2.72 Verarbeitungsschritt_50 vibrato nicht blockierend: %s", _vib50_exc)

        _rms_in_50 = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-12))
        _rms_out_50 = float(np.sqrt(np.mean(np.asarray(repaired_audio, dtype=np.float64) ** 2) + 1e-12))
        _rms_drop_50 = 20.0 * np.log10(max(_rms_out_50 / _rms_in_50, 1e-30)) if _rms_in_50 > 1e-8 else 0.0
        # §v10.101: Garantiere ndarray — verhindert tuple-ndim im PMGG/Steering-Pfad
        _safe_audio = np.asarray(repaired_audio, dtype=np.float32)
        return PhaseResult(
            success=True,
            audio=_safe_audio,
            execution_time_seconds=time.time() - t0,
            metadata={
                "threshold_factor": threshold_factor,
                "threshold_factor_effective": threshold_factor_eff,
                "threshold_runtime_profile": dict(_runtime_profile_50),
                "strength_floor": float(_runtime_profile_50["strength_floor"]),
                "side_multiplier": float(_runtime_profile_50["side_multiplier"]),
                "n_channels": n_channels,
                "stereo_mode": "ms_domain" if is_stereo else "mono",
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "repair_locality_coverage": round(float(_local_coverage50), 6),
                "hf_protected_bin_start": _hf_protected_bin_start,
                "hf_protection_rolloff_hz": round(_rolloff_protection_hz, 1),
                "rms_drop_db": round(float(min(0.0, _rms_drop_50)), 3),
                "loudness_makeup_db": 0.0,
            },
            metrics={
                "n_repaired_bins": total_bins,
                "repair_ratio": repair_ratio,
                "snr_improvement_db": max(0.0, snr_before - 30.0),
                "effective_strength": effective_strength,
            },
        )
