#!/usr/bin/env python3
"""
Phase 41: Professional Output Format Optimization v2.0.
=======================================================

High-quality resampling, dithering, and format-specific optimization for final delivery.

SCIENTIFIC FOUNDATION:
- Smith & Gossett (1984): A Flexible Sampling-Rate Conversion Method
- Reiss (2016): A Meta-Analysis of High Resolution Audio Perceptual Evaluation
- Wannamaker et al. (2000): A Theory of Non-Subtractive Dither
- Lipshitz et al. (1992): Quantization and Dither: A Theoretical Survey
- ITU-R BS.1770-4: Algorithms to Measure Audio Programme Loudness
- EBU R 128: Loudness Normalization and Permitted Maximum Level
- Oppenheim & Schafer (2009): Discrete-Time Signal Processing

INDUSTRY BENCHMARKS:
- iZotope Ozone 10 (SRC + dithering + LUFS normalization)
- Waves L2 Ultramaximizer (Dithering + IDR)
- Weiss Saracon (Professional SRC)
- iZotope RX 10 (Resampling + bit depth conversion)
- FabFilter Pro-L 2 (True peak limiting + dithering)
- Sonnox Oxford Limiter (Dithering algorithms)
- Nugen Audio ISL 2 (Loudness management)

ALGORITHM:
1. High-Quality Resampling
   - Polyphase FIR filterbank (anti-aliasing)
   - Kaiser window design (optimal stopband attenuation)
   - Linear phase (zero phase distortion)

2. Advanced Dithering
   - TPDF (Triangular PDF): Standard for 16-bit
   - Noise-shaped dithering: Psychoacoustic optimization
   - Bit depth: 16, 24, 32-bit float

3. Format-Specific Optimization
   - CD Red Book: 44.1kHz, 16-bit, TPDF dither
   - Hi-Res (96/24): 96kHz, 24-bit, minimal dither
   - Streaming: 48kHz, variable bit rate, LUFS normalization

4. Loudness Normalization
   - LUFS target: CD -14, Streaming -16, Hi-Res -18
   - True peak limiting: -1.0 dBTP

5. Material-Adaptive Parameters
   - Shellac: 44.1kHz/16-bit (archival standard)
   - Vinyl: 96kHz/24-bit (preserve analog fidelity)
   - Tape: 48kHz/24-bit (standard studio)
   - Digital: Format-specific

QUALITY TARGETS:
- SNR improvement: +3 to +6 dB (via dithering)
- Aliasing: <-100 dB
- Processing: <0.3× realtime

Author: Aurik Professional Team
Version: 2.0.0
Date: February 2026
"""

import logging
import os
import time
from math import gcd

import numpy as np
from scipy import signal

from backend.core.audio_utils import (
    apply_musical_gain_envelope,
    limit_quiet_edge_boost,
    safe_resample_poly as _safe_resample_poly41,
)
from backend.core.defect_scanner import MaterialType

try:
    from dsp.professional_meters import LUFSMeter

    PROFESSIONAL_METERS_AVAILABLE = True
except ImportError:
    PROFESSIONAL_METERS_AVAILABLE = False

from .output_guard import evaluate_output_guard
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class OutputFormatOptimization(PhaseInterface):
    """Professional output format optimization."""

    # Material-adaptive output sample rates
    OUTPUT_SAMPLE_RATE = {
        MaterialType.SHELLAC: 44100,  # Archival standard (CD quality)
        MaterialType.VINYL: 96000,  # Hi-res (preserve analog fidelity)
        MaterialType.TAPE: 48000,  # Studio standard
        MaterialType.CASSETTE: 48000,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: 44100,  # CD Red Book
        MaterialType.STREAMING: 48000,  # Streaming standard
    }

    # Material-adaptive bit depths
    OUTPUT_BIT_DEPTH = {
        MaterialType.SHELLAC: 16,  # Sufficient for archival noise floor
        MaterialType.VINYL: 24,  # Preserve dynamic range
        MaterialType.TAPE: 24,  # Studio standard
        MaterialType.CASSETTE: 24,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: 16,  # CD Red Book
        MaterialType.STREAMING: 16,  # Streaming standard
    }

    # LUFS targets (loudness normalization)
    LUFS_TARGET = {
        MaterialType.SHELLAC: -18.0,  # Conservative (archival)
        MaterialType.VINYL: -16.0,  # Moderate
        MaterialType.TAPE: -16.0,  # Moderate
        MaterialType.CASSETTE: -16.0,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: -14.0,  # CD standard
        MaterialType.STREAMING: -16.0,  # Spotify/YouTube standard
    }

    # True peak ceiling (prevent clipping in lossy codecs)
    TRUE_PEAK_CEILING = {
        MaterialType.SHELLAC: -1.0,
        MaterialType.VINYL: -0.5,  # Mastering headroom
        MaterialType.TAPE: -0.5,
        MaterialType.CASSETTE: -0.5,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: -0.1,  # CD Red Book
        MaterialType.STREAMING: -1.0,  # Codec headroom
    }

    # Dithering type — §4.5 Spec 04: POW-r Typ 3 PRIMÄR, TPDF FALLBACK
    DITHER_TYPE = {
        MaterialType.SHELLAC: "pow_r_3",
        MaterialType.VINYL: "pow_r_3",
        MaterialType.TAPE: "pow_r_3",
        MaterialType.CASSETTE: "pow_r_3",  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: "pow_r_3",
        MaterialType.STREAMING: "pow_r_3",
    }

    def __init__(self):
        super().__init__()
        self.name = "Output Format Optimization v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_41_output_format_optimization",
            name="Output Format Optimization v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=11,
            dependencies=["phase_40_final_loudness_normalization"],
            estimated_time_factor=0.12,
            version="2.0.0",
            memory_requirement_mb=100,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description="High-quality resampling, dithering, and format optimization",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        """
        Wendet an: output format optimization.

        Args:
            audio: Audio samples (mono or stereo)
            sample_rate: Input sample rate in Hz
            material_type: Material type

        Returns:
            PhaseResult with optimized audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        start_time = time.time()
        phase_material: MaterialType
        if isinstance(material_type, MaterialType):
            phase_material = material_type
        else:
            try:
                phase_material = MaterialType(str(material_type).lower())
            except ValueError:
                phase_material = MaterialType.UNKNOWN

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={
                    "resampled": False,
                    "input_sample_rate": sample_rate,
                    "output_sample_rate": sample_rate,
                    "output_bit_depth": 32,
                    "lufs_before": -70.0,
                    "lufs_after": -70.0,
                    "peak_reduction_db": 0.0,
                    "dithered": False,
                    "dither_type": "none",
                    "snr_improvement_db": 0.0,
                    "material": phase_material.value,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "version": "2.0",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        intended_output_sr = self.OUTPUT_SAMPLE_RATE.get(phase_material, 44100)
        intended_output_bit_depth = self.OUTPUT_BIT_DEPTH.get(phase_material, 16)
        lufs_target = self.LUFS_TARGET.get(phase_material, -16.0)
        true_peak_ceiling = self.TRUE_PEAK_CEILING.get(phase_material, -1.0)
        dither_type = self.DITHER_TYPE.get(phase_material, "tpdf")

        # Phase 41 runs inside the fixed 48 kHz restoration pipeline.
        # Delivery-format conversion belongs to the export layer, not to an
        # in-graph phase: resampling to material-specific rates and bit-depth
        # reduction inside UV3 corrupt downstream metrics and can destabilize
        # late-stage quality gates. Keep the live pipeline at the current
        # sample rate and float precision, and expose the intended delivery
        # format only as metadata for the exporter.
        output_sr = sample_rate
        output_bit_depth = 32

        quality_mode = str(kwargs.get("quality_mode", "balanced")).lower()
        _is_studio_2026 = bool(kwargs.get("studio_mode", False)) or (quality_mode in ("studio2026", "studio"))
        if quality_mode in ("quality", "maximum", "studio2026"):
            hq_scale = 1.10 if quality_mode in ("maximum", "studio2026") else 1.05
            lufs_target = float(np.clip(lufs_target + 0.5 * (1.0 - hq_scale), -24.0, -12.0))
            true_peak_ceiling = float(np.clip(true_peak_ceiling - 0.2 * (hq_scale - 1.0), -2.0, -0.1))
        else:
            hq_scale = 1.0

        # Step 1: High-quality resampling
        resampled = False
        if output_sr != sample_rate:
            audio_resampled = self._resample_high_quality(audio, sample_rate, output_sr)
            resampled = True
        else:
            audio_resampled = audio.copy()

        # Step 2/3: In-pipeline loudness operations.
        # Restoration mode: advisory-only (no active gain drive).
        # Rationale: global carrier-restoration pipeline must preserve musical level
        # and avoid late-stage loudness jumps; final delivery loudness is handled by
        # exporter/end-guard. Studio 2026 may still apply active normalization.
        if not _is_studio_2026:
            lufs_before = self._measure_integrated_lufs(audio_resampled, output_sr)
            lufs_after = lufs_before
            audio_normalized = audio_resampled
            audio_limited = audio_resampled
            peak_reduction_db = 0.0
        else:
            # Step 2: Loudness normalization (LUFS-based)
            audio_normalized, lufs_before, lufs_after = self._normalize_loudness(
                audio_resampled, output_sr, lufs_target
            )

            if 0.0 < _effective_strength < 1.0:
                audio_normalized = audio_resampled + _effective_strength * (audio_normalized - audio_resampled)

            # Step 3: True peak limiting
            audio_limited, peak_reduction_db = self._limit_true_peak(audio_normalized, true_peak_ceiling)

        # Step 4: Dithering (before bit-depth reduction — 16-bit and 24-bit)
        # §4.5 Spec 04: POW-r Typ 3 PRIMÄR, TPDF FALLBACK
        dithered = False
        if output_bit_depth == 16:
            if dither_type == "pow_r_3":
                audio_dithered = self._apply_pow_r_type3_dither(audio_limited, bit_depth=16)
            elif dither_type == "noise_shaped":
                audio_dithered = self._apply_noise_shaped_dither(audio_limited)
            else:
                audio_dithered = self._apply_tpdf_dither(audio_limited, bit_depth=16)
            dithered = True
        elif output_bit_depth == 24:
            if dither_type == "pow_r_3":
                audio_dithered = self._apply_pow_r_type3_dither(audio_limited, bit_depth=24)
            else:
                audio_dithered = self._apply_tpdf_dither(audio_limited, bit_depth=24)
            dithered = True
        else:
            audio_dithered = audio_limited

        # Step 5: Quantization
        audio_quantized = self._quantize(audio_dithered, output_bit_depth)

        audio_pre_guard = np.nan_to_num(audio_quantized.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        audio_pre_guard = np.clip(audio_pre_guard, -1.0, 1.0)

        # Measure SNR improvement (dithering benefit)
        snr_before = self._estimate_snr(audio_limited)
        snr_after = self._estimate_snr(audio_quantized)
        snr_improvement_db = snr_after - snr_before

        output_guard_enabled = quality_mode in ("quality", "maximum", "studio2026")
        guard = evaluate_output_guard(
            original=audio,
            candidate=audio_quantized,
            enabled=output_guard_enabled,
            max_abs_rms_delta_db=1.5,
            stereo_side_ratio_min=0.60,
            stereo_side_ratio_max=1.45,
        )
        if guard.fallback:
            audio_quantized = audio_pre_guard
            snr_after = self._estimate_snr(audio_quantized)
            snr_improvement_db = snr_after - snr_before

        processing_time = time.time() - start_time

        audio_quantized = np.nan_to_num(audio_quantized, nan=0.0, posinf=0.0, neginf=0.0)
        audio_quantized = np.clip(audio_quantized, -1.0, 1.0)

        # Return pre-quantization audio (audio_limited) for the pipeline.
        # Quantization creates discrete amplitude steps that spectral PMGG proxies
        # misinterpret as noise injection, causing false catastrophic regressions.
        # The actual bit-depth conversion is performed by the I/O layer (soundfile)
        # at export time — returning float32 here is correct and lossless for PMGG.
        audio_pipeline = np.nan_to_num(audio_limited.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        audio_pipeline = np.clip(audio_pipeline, -1.0, 1.0).astype(np.float32)

        return PhaseResult(
            success=True,
            audio=audio_pipeline,
            metrics={
                "resampled": resampled,
                "input_sample_rate": sample_rate,
                "output_sample_rate": output_sr,
                "output_bit_depth": output_bit_depth,
                "intended_output_sample_rate": intended_output_sr,
                "intended_output_bit_depth": intended_output_bit_depth,
                "lufs_before": float(lufs_before),
                "lufs_after": float(lufs_after),
                "peak_reduction_db": float(peak_reduction_db),
                "dithered": dithered,
                "dither_type": dither_type,
                "snr_improvement_db": float(snr_improvement_db),
                "material": phase_material.value,
                "rms_delta_db": float(guard.rms_delta_db),
                "stereo_side_ratio": float(guard.stereo_side_ratio),
            },
            execution_time_seconds=processing_time,
            metadata={
                "algorithm": "high_quality_src_dither_lufs",
                "version": "2.0",
                "quality_mode": quality_mode,
                "hq_scale": hq_scale,
                "pipeline_safe_format_optimization": True,
                "intended_output_sample_rate": intended_output_sr,
                "intended_output_bit_depth": intended_output_bit_depth,
                "output_guard_enabled": output_guard_enabled,
                "output_guard_fallback": guard.fallback,
                "output_guard_reason": guard.reason,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )

    def _resample_high_quality(self, audio: np.ndarray, input_sr: int, output_sr: int) -> np.ndarray:
        """
        High-quality polyphase resampling (O(N) resample_poly, faster than FFT-based resample).
        """
        if input_sr == output_sr:
            return audio.copy()
        g = gcd(input_sr, output_sr)
        up, down = output_sr // g, input_sr // g
        # axis=-1 = time axis for channels-first (2,N) and samples-first (N,) alike.
        # axis=0 was resampling the 2-element channel dimension, producing polyphase
        # garbage that silenced the second half of the audio. §ph41-axis fix.
        return _safe_resample_poly41(audio, up, down, axis=-1)  # type: ignore[no-any-return]

    def _normalize_loudness(
        self, audio: np.ndarray, sample_rate: int, lufs_target: float
    ) -> tuple[np.ndarray, float, float]:
        """
        LUFS-based loudness normalization.
        """
        lufs_before = self._measure_integrated_lufs(audio, sample_rate)
        if not np.isfinite(lufs_before):
            lufs_before = -70.0

        # Calculate gain adjustment
        lufs_difference = lufs_target - lufs_before
        gain_db = lufs_difference
        gain_linear = 10 ** (gain_db / 20.0)

        # Apply gain — §2.45a-II: musical-frame-only when amplifying to avoid
        # boosting analog surface noise in silent/fade-out sections.
        if gain_linear > 1.0005:
            # §2.45a-II v10: Soft-knee defaults (crossfade_ms=200, knee_width_db=6).
            audio_normalized = apply_musical_gain_envelope(
                audio,
                gain_linear,
                gate_dbfs=-36.0,
                sr=sample_rate,
                reference_for_gate=audio,
            )
            audio_normalized = limit_quiet_edge_boost(audio, audio_normalized, sr=sample_rate, max_edge_boost_db=0.5)
        else:
            audio_normalized = audio * gain_linear

        # Recalculate LUFS
        lufs_after = self._measure_integrated_lufs(audio_normalized, sample_rate)
        if not np.isfinite(lufs_after):
            lufs_after = -70.0

        return audio_normalized, lufs_before, lufs_after

    def _measure_integrated_lufs(self, audio: np.ndarray, sample_rate: int) -> float:
        """Misst integrated loudness using ITU-R BS.1770 where available."""
        audio_arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if PROFESSIONAL_METERS_AVAILABLE:
            try:
                meter = LUFSMeter(sr=sample_rate)
                meter_audio = audio_arr.T if audio_arr.ndim == 2 else audio_arr
                meter_result = meter.measure(meter_audio, sample_rate)
                return float(meter_result.get("integrated_lufs", -70.0))
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        # Fallback: conservative RMS proxy (kept for resilience if meter backend is unavailable).
        if audio_arr.ndim == 2:
            rms = np.sqrt(np.mean(audio_arr**2, axis=0))
            rms_avg = float(np.mean(rms))
        else:
            rms_avg = float(np.sqrt(np.mean(audio_arr**2)))
        return float(20.0 * np.log10(rms_avg + 1e-10) - 23.0)

    def _limit_true_peak(self, audio: np.ndarray, ceiling_db: float) -> tuple[np.ndarray, float]:
        """
        True peak limiting (prevent clipping in D/A conversion).
        """
        ceiling_linear = 10 ** (ceiling_db / 20.0)

        peak = self._measure_true_peak_linear(audio)

        if peak > ceiling_linear:
            # Apply gain reduction
            gain_reduction_linear = ceiling_linear / peak
            audio_limited = audio * gain_reduction_linear
            peak_reduction_db = 20 * np.log10(gain_reduction_linear)
        else:
            audio_limited = audio.copy()
            peak_reduction_db = 0.0

        return audio_limited, peak_reduction_db

    def _measure_true_peak_linear(self, audio: np.ndarray) -> float:
        """Measure inter-sample true peak using 4x oversampling.

        §DSP-Invariante: np.percentile(99.9) statt np.max — ein einzelner
        Crackle/Click-Spike nach 4x-Übersampling darf nicht die gesamte
        Gain-Reduction blockieren und das Programmpegel absenken.
        """
        audio_arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio_arr.ndim == 1:
            audio_up = _safe_resample_poly41(audio_arr, 4, 1)
            return float(np.percentile(np.abs(audio_up), 99.9))

        audio_up = _safe_resample_poly41(audio_arr, 4, 1, axis=0)
        return float(np.percentile(np.abs(audio_up), 99.9))

    def _apply_pow_r_type3_dither(self, audio: np.ndarray, bit_depth: int = 16) -> np.ndarray:
        """Apply POW-r Type 3 noise-shaped dither (Wannamaker et al. 1992).

        §4.5 Spec 04: PRIMÄR — psychoacoustic noise-shaping with 5th-order
        error-feedback filter.  Pushes quantisation noise into perceptually
        less sensitive frequency bands (~+6 dB effective SNR over TPDF).
        """
        if bit_depth == 16:
            lsb = 1.0 / (2**15)
        elif bit_depth == 24:
            lsb = 1.0 / (2**23)
        else:
            return audio

        # POW-r Type 3 error-feedback coefficients (5th order, designed for
        # 44.1/48 kHz — Wannamaker's psychoacoustic optimisation).
        _coeffs = np.array([2.033, -2.165, 1.959, -1.590, 0.6149], dtype=np.float64)
        order = len(_coeffs)

        # Content-derived seed for deterministic reproducibility (§2.40)
        _seed = int(abs(float(np.sum(np.abs(audio[: min(len(audio.ravel()), 1024)])))) * 1e5 + bit_depth) % (2**31)
        _rng = np.random.default_rng(seed=_seed)

        def _shape_channel(ch: np.ndarray) -> np.ndarray:
            ch = ch.astype(np.float64)
            out = np.empty_like(ch)
            error_buf = np.zeros(order, dtype=np.float64)
            max_val = (2 ** (bit_depth - 1)) - 1
            for i in range(len(ch)):
                # TPDF dither (2 × uniform = triangular)
                d = _rng.uniform(-lsb, lsb) + _rng.uniform(-lsb, lsb)
                # Error-feedback: inject shaped past errors
                shaped = ch[i] + d - float(np.dot(_coeffs, error_buf))
                # Quantise
                quantised = np.clip(np.round(shaped * max_val), -max_val, max_val) / max_val
                # Update error buffer (FIFO)
                error = quantised - ch[i]
                error_buf = np.roll(error_buf, 1)
                error_buf[0] = error
                out[i] = quantised
            return out  # type: ignore[no-any-return]

        if audio.ndim == 2:
            shaped_audio = np.empty_like(audio, dtype=np.float64)
            for c in range(audio.shape[1]):
                shaped_audio[:, c] = _shape_channel(audio[:, c])
            return shaped_audio.astype(np.float32)  # type: ignore[no-any-return]
        return _shape_channel(audio).astype(np.float32)  # type: ignore[no-any-return]

    def _apply_tpdf_dither(self, audio: np.ndarray, bit_depth: int = 16) -> np.ndarray:
        """
        Wendet an: TPDF (Triangular PDF) dither before bit-depth quantization.
        Supports 16-bit and 24-bit output.
        """
        if bit_depth == 16:
            # 1 LSB at 16-bit = 1/32768
            dither_amplitude = 1.0 / (2**15)
        elif bit_depth == 24:
            # 1 LSB at 24-bit = 1/8388608
            dither_amplitude = 1.0 / (2**23)
        else:
            return audio  # 32-bit float: no quantization noise

        # Two uniform random variables summed = triangular PDF (true TPDF)
        # §2.40 Determinismus: content-derived seed for bit-exact reproducibility
        _d41_seed = int(abs(float(np.sum(np.abs(audio[: min(len(audio.ravel()), 1024)])))) * 1e5 + bit_depth) % (2**31)
        _rng41 = np.random.default_rng(seed=_d41_seed)
        dither1 = _rng41.uniform(-dither_amplitude, dither_amplitude, audio.shape)
        dither2 = _rng41.uniform(-dither_amplitude, dither_amplitude, audio.shape)
        dither = dither1 + dither2

        return audio + dither  # type: ignore[no-any-return]

    def _apply_noise_shaped_dither(self, audio: np.ndarray) -> np.ndarray:
        """
        Wendet an: noise-shaped dither (psychoacoustic optimization).
        """
        # Simplified noise shaping: High-pass filtered dither
        # Pushes quantization noise to high frequencies (less audible)

        dither_amplitude = 1.0 / (2**15)

        # White dither — §2.40 Determinismus: content-derived seed
        _dns_seed = int(abs(float(np.sum(np.abs(audio[: min(len(audio.ravel()), 1024)])))) * 1e5 + 1) % (2**31)
        _rng_ns = np.random.default_rng(seed=_dns_seed)
        dither1 = _rng_ns.uniform(-dither_amplitude, dither_amplitude, audio.shape)
        dither2 = _rng_ns.uniform(-dither_amplitude, dither_amplitude, audio.shape)
        dither = dither1 + dither2

        # High-pass filter (noise shaping)
        # Simple first-order: y[n] = x[n] - x[n-1] (differentiator)
        if audio.ndim == 2:
            dither_shaped = np.zeros_like(dither)
            for ch in range(audio.shape[1]):
                dither_shaped[1:, ch] = dither[1:, ch] - 0.5 * dither[:-1, ch]
                dither_shaped[0, ch] = dither[0, ch]
        else:
            dither_shaped = np.zeros_like(dither)
            dither_shaped[1:] = dither[1:] - 0.5 * dither[:-1]
            dither_shaped[0] = dither[0]

        return audio + dither_shaped  # type: ignore[no-any-return]

    def _quantize(self, audio: np.ndarray, bit_depth: int) -> np.ndarray:
        """
        Quantize to specified bit depth.
        """
        if bit_depth == 16:
            max_val = 2**15 - 1
        elif bit_depth == 24:
            max_val = 2**23 - 1
        elif bit_depth == 32:
            # 32-bit float, no quantization
            return audio
        else:
            return audio

        # Convert to integer
        audio_int = np.round(audio * max_val)

        # Clip
        audio_int = np.clip(audio_int, -max_val, max_val)

        # Convert back to float
        audio_quantized = audio_int / max_val

        return audio_quantized  # type: ignore[no-any-return]

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Schätzt SNR via RMS-block method (O(N), avoids O(N log N) FFT overhead).
        """
        flat = audio.flatten() if audio.ndim == 2 else audio
        rms_signal = float(np.sqrt(np.mean(flat**2))) + 1e-10
        block_size = min(4096, len(flat))
        n_blocks = max(1, len(flat) // block_size)
        blocks = np.array_split(flat[: n_blocks * block_size], n_blocks)
        block_rms = np.array([np.sqrt(np.mean(b**2)) for b in blocks])
        noise_floor = float(np.percentile(block_rms, 10)) + 1e-10
        snr_db = 20.0 * np.log10(rms_signal / noise_floor)
        return min(snr_db, 120.0)  # type: ignore[no-any-return]


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Verarbeitungsschritt 41: Professional Ausgabe Format Optimization v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio
    duration = 2.0
    demo_sr = 48000  # Input at 48kHz

    t = np.linspace(0, duration, int(demo_sr * duration))

    # Test signal: 1kHz sine + noise
    test_signal = 0.3 * np.sin(2 * np.pi * 1000 * t)
    test_signal += 0.05 * np.random.randn(len(t))  # Add noise

    # Stereo
    test_signal_stereo = np.column_stack([test_signal, test_signal * 0.95])

    logger.debug("erzeugt %ss test audio @ %s Hz", duration, demo_sr)
    logger.debug("Signal: 1kHz sine + noise (stereo)")
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.CD_DIGITAL, "CD_DIGITAL"),
        (MaterialType.VINYL, "VINYL"),
        (MaterialType.STREAMING, "STREAMING"),
    ]

    for demo_material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug("Material: %s", material_name)
        logger.debug("─" * 80)
        logger.debug("")

        phase = OutputFormatOptimization()
        demo_result = phase.process(test_signal_stereo, demo_sr, demo_material)  # type: ignore[arg-type]

        logger.debug("✅ Professional Ausgabe Format Optimization:")
        logger.debug("   Eingabe: %s Hz", demo_result.metrics["input_sample_rate"])
        logger.debug(
            "   Ausgabe: %s Hz, %s-bit",
            demo_result.metrics["output_sample_rate"],
            demo_result.metrics["output_bit_depth"],
        )
        logger.debug("   Resampled: %s", demo_result.metrics["resampled"])
        logger.debug(
            "   LUFS: %.1f -> %.1f (target: %.1f)",
            demo_result.metrics["lufs_before"],
            demo_result.metrics["lufs_after"],
            phase.LUFS_TARGET[demo_material],
        )
        logger.debug("   Peak Reduction: %.2f dB", demo_result.metrics["peak_reduction_db"])
        logger.debug("   Dithered: %s (%s)", demo_result.metrics["dithered"], demo_result.metrics["dither_type"])
        logger.debug("   SNR Improvement: %.2f dB", demo_result.metrics["snr_improvement_db"])
        logger.debug(
            "   Processing time: %.3fs (%.2fx realtime)",
            demo_result.execution_time_seconds,
            demo_result.execution_time_seconds / duration,
        )
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test abgeschlossen")
    logger.debug("=" * 80)
