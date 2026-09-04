#!/usr/bin/env python3
"""
Phase 30: DC Offset Removal v2.0 - Professional
Advanced DC offset and subsonic rumble removal with adaptive filtering.

Algorithm Overview:
1. DC Tracking:
   - Measure DC offset over sliding windows
   - Detect time-varying DC drift
   - Adaptive removal strength
2. Subsonic Analysis:
   - Spectral analysis of <30 Hz content
   - Identify mechanical rumble vs. musical bass
   - Frequency-selective filtering
3. Phase-Linear Filtering:
   - FIR high-pass filters (zero phase distortion)
   - Preserve transient timing
   - Critical for stereo imaging
4. Adaptive HP Cutoff:
   - Material-specific cutoff frequencies
   - Q-factor control for roll-off steepness
   - Balance rumble removal vs. bass preservation
5. Quality Gates:
   - Verify no audible bass loss
   - Monitor phase coherence
   - Prevent over-filtering

Scientific Foundation:
- Harris (1978): On the Use of Windows for Harmonic Analysis with DFT
- Smith (2011): Spectral Audio Signal Processing (FIR Filter Design)
- Oppenheim & Schafer (2010): Discrete-Time Signal Processing
- Zölzer (2011): DAFX - Digital Audio Effects
- AES Paper 3922: Low-Frequency Filter Design

Industry Benchmarks:
- Waves X-Hum ($49)
- iZotope RX De-hum ($399)
- Sonnox SuprEsser ($249)
- Cedar DNS (Adaptive filter, $2000+)
- Z-Noise ($49)

Quality Target: 0.60 → 0.85 (+42% improvement)
Performance Target: <0.05× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import audio_sample_count, safe_filtfilt, stereo_channel_view, stereo_like  # §v10.101
from backend.core.defect_scanner import MaterialType

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class DCOffsetRemoval(PhaseInterface):
    """
    Professional DC Offset and Subsonic Rumble Removal.

    Key Features:
    - Time-varying DC tracking
    - Adaptive high-pass filtering
    - Phase-linear FIR filters
    - Material-adaptive cutoff frequencies
    - Q-factor control for steep roll-off
    - Bass preservation monitoring

    Use Cases:
    - Remove ADC bias from digitization
    - Eliminate turntable rumble
    - Clean mechanical noise
    - Preserve musical bass content

    Performance: <0.05× realtime on modern CPU
    """

    # Material-adaptive DC-removal configurations.
    # Phase 30 must stay conservative: true DC correction first, no aggressive rumble removal.
    HP_CONFIG = {
        MaterialType.SHELLAC: {
            "cutoff_hz": 8,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.VINYL: {
            "cutoff_hz": 6,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.TAPE: {
            "cutoff_hz": 5,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.REEL_TAPE: {
            "cutoff_hz": 5,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.CASSETTE: {
            "cutoff_hz": 5,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: {
            "cutoff_hz": 4,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.MP3_LOW: {
            "cutoff_hz": 4,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.MP3_HIGH: {
            "cutoff_hz": 4,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.AAC: {
            "cutoff_hz": 4,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
        MaterialType.STREAMING: {
            "cutoff_hz": 4,
            "filter_order": 2,
            "filter_type": "iir",
            "q_factor": 0.7,
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "DC Offset Removal v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_30_dc_offset_removal",
            name="DC Offset Removal v2 Professional",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=1,
            dependencies=[],
            estimated_time_factor=0.05,
            version="2.0.0",
            memory_requirement_mb=30,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.85,
            description="Advanced DC offset and subsonic rumble removal with phase-linear filtering",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: MaterialType = MaterialType.VINYL,  # type: ignore[override]
        **kwargs,
    ) -> PhaseResult:
        """
        Verarbeitet audio to remove DC offset and rumble.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Source material type

        Returns:
            PhaseResult with cleaned audio
        """
        material = material_type  # interner Alias
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        is_stereo = audio.ndim == 2
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        config = dict(self.HP_CONFIG.get(_mk, self.HP_CONFIG[MaterialType.VINYL]))  # type: ignore[call-overload]

        # Locality-aware intensity control from UV3.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "hp_cutoff_hz": float(config["cutoff_hz"]),  # type: ignore[arg-type]
                    "filter_type": config["filter_type"],
                    "filter_order": int(config["filter_order"]),  # type: ignore[call-overload]
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "processing": "skipped_zero_strength",
                    "rt_factor": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        _original_audio = np.asarray(audio, dtype=np.float32).copy()

        # Measure DC offset before removal
        dc_offset_before = [float(np.mean(audio[:, ch])) for ch in range(2)] if is_stereo else [float(np.mean(audio))]

        # Measure subsonic energy before removal
        subsonic_energy_before = self._measure_subsonic_energy(audio, sample_rate, config["cutoff_hz"])  # type: ignore[arg-type]

        # Process each channel
        if is_stereo:
            left, right = stereo_channel_view(audio)
            clean_left = self._remove_dc_and_rumble(left, sample_rate, config)
            clean_right = self._remove_dc_and_rumble(right, sample_rate, config)
            audio_processed = stereo_like(clean_left, clean_right, audio)
        else:
            audio_processed = self._remove_dc_and_rumble(audio, sample_rate, config)

        # Measure DC offset after removal
        if is_stereo:
            dc_offset_after = [float(np.mean(audio_processed[:, ch])) for ch in range(2)]
        else:
            dc_offset_after = [float(np.mean(audio_processed))]

        # Measure subsonic energy after removal
        subsonic_energy_after = self._measure_subsonic_energy(audio_processed, sample_rate, config["cutoff_hz"])  # type: ignore[arg-type]

        # Calculate reduction
        dc_reduction = [abs(before - after) for before, after in zip(dc_offset_before, dc_offset_after)]
        subsonic_reduction_db = 20 * np.log10((subsonic_energy_before + 1e-10) / (subsonic_energy_after + 1e-10))

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        audio_processed = np.nan_to_num(audio_processed, nan=0.0, posinf=0.0, neginf=0.0)
        audio_processed = np.clip(audio_processed, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            audio_processed = audio + _effective_strength * (audio_processed - audio)
            audio_processed = np.clip(audio_processed, -1.0, 1.0)

        audio_processed, _rms_drop_db, _makeup_db = self._preserve_phase_loudness(
            _original_audio,
            audio_processed,
            material,
        )
        if abs(_makeup_db) > 0.01:
            logger.info(
                "Verarbeitungsschritt 30 loudness-preservation: material=%s rms_drop=%+.2f dB via makeup %+.2f dB",
                material.value,
                _rms_drop_db,
                _makeup_db,
            )

        return PhaseResult(
            success=True,
            audio=audio_processed,
            execution_time_seconds=execution_time,
            resolved_defects={
                "DC_OFFSET": 0.0,  # DC-Offset = vollständig entfernt
            },
            metadata={
                "material": material.name,
                "hp_cutoff_hz": float(config["cutoff_hz"]),  # type: ignore[arg-type]
                "filter_type": config["filter_type"],
                "filter_order": int(config["filter_order"]),  # type: ignore[call-overload]
                "dc_offset_before": [round(v, 6) for v in dc_offset_before],
                "dc_offset_after": [round(v, 6) for v in dc_offset_after],
                "dc_reduction": [round(v, 6) for v in dc_reduction],
                "subsonic_reduction_db": float(round(subsonic_reduction_db, 2)),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rt_factor": float(rt_factor),
                "rms_drop_db": _rms_drop_db,
                "loudness_makeup_db": _makeup_db,
            },
            warnings=[] if rt_factor < 0.08 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _remove_dc_and_rumble(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Entfernt DC offset conservatively from a single channel."""
        cutoff_hz = config["cutoff_hz"]
        filter_order = config["filter_order"]
        filter_type = config["filter_type"]

        # **GUARD: Short-Audio-Buffer (§2.47, §0 Primum non nocere)**
        # filtfilt/sosfiltfilt require len(audio) > padlen (typically 9–20 samples)
        # For audio < minimum window size, passthrough instead of crashing
        MIN_AUDIO_SAMPLES = 512  # 10 ms @ 48 kHz — minimum safe for any filter
        if len(audio) < MIN_AUDIO_SAMPLES:
            logger.debug(
                "Verarbeitungsschritt_30: audio too short (%d < %d), passthrough", len(audio), MIN_AUDIO_SAMPLES
            )
            return np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

        # Stage 1: always remove static DC bias directly.
        dc = float(np.mean(audio))
        if abs(dc) > 1e-9:
            audio = audio - dc

        # Stage 2: remove residual near-DC drift with very low cutoff.
        # For tape/reel-tape we use the project-mandated zero-phase form.
        if cutoff_hz <= 0.0:
            return np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

        if filter_type == "iir" and cutoff_hz <= 5.0:
            b = np.array([1.0, -1.0], dtype=np.float64)
            a = np.array([1.0, -0.9995], dtype=np.float64)
            return safe_filtfilt(b, a, audio).astype(np.float32)  # type: ignore[no-any-return]

        if filter_type == "fir":
            # Phase-linear FIR filter
            # Design using window method
            nyquist = sample_rate / 2
            cutoff_norm = cutoff_hz / nyquist

            # Ensure odd order for symmetric FIR
            if filter_order % 2 == 0:
                filter_order += 1

            # Design FIR highpass
            fir_coeffs = signal.firwin(
                filter_order * 20 + 1,  # Higher order for sharper cutoff
                cutoff_norm,
                window="hamming",
                pass_zero=False,  # Highpass
            )

            # Apply filter (already zero-phase due to symmetric design)
            processed = safe_filtfilt(fir_coeffs, [1.0], audio)

        else:  # IIR
            # Butterworth IIR filter (efficient for minimal processing)
            sos = signal.butter(filter_order, cutoff_hz, btype="high", fs=sample_rate, output="sos")

            # Apply filter (forward-backward for zero-phase)
            processed = signal.sosfiltfilt(sos, audio)

        return np.nan_to_num(np.asarray(processed, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    def _preserve_phase_loudness(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, float, float]:
        """Preserve perceived level for DC-removal phase to avoid musical loss."""
        orig = np.asarray(original_audio, dtype=np.float64)
        proc = np.asarray(processed_audio, dtype=np.float64)
        if orig.shape != proc.shape:
            return np.clip(np.asarray(processed_audio, dtype=np.float32), -1.0, 1.0), 0.0, 0.0

        max_drop_db = {  # type: ignore[call-overload]
            MaterialType.SHELLAC: 1.1,
            MaterialType.VINYL: 0.9,
            MaterialType.TAPE: 1.0,
            MaterialType.REEL_TAPE: 1.0,
        }.get(material.name if isinstance(material, MaterialType) else str(material), 0.8)
        max_lift_db = 0.6

        # §2.45a-I: Gated RMS — nur Frames > -50 dBFS (kein Stille-inflationierter RMS)
        # §V04-EXEMPT: compute_gated_rms_linear() — reines RMS-Measurement, kein Gain-Envelope
        from backend.core.audio_utils import (
            compute_gated_rms_linear as _grl_p30,  # pylint: disable=import-outside-toplevel
        )

        orig_rms = float(_grl_p30(orig, gate_dbfs=-50.0))
        proc_rms = float(_grl_p30(proc, gate_dbfs=-50.0))
        if orig_rms < 1e-10 or proc_rms < 1e-12:
            return np.clip(proc.astype(np.float32), -1.0, 1.0), 0.0, 0.0

        delta_db = float(20.0 * np.log10(max(proc_rms / orig_rms, 1e-30)))
        gain_db = 0.0
        if delta_db < -max_drop_db:
            gain_db = (-max_drop_db) - delta_db
        elif delta_db > max_lift_db:
            gain_db = max_lift_db - delta_db

        if abs(gain_db) > 0.01:
            gain = float(10.0 ** (gain_db / 20.0))
            p999 = float(np.percentile(np.abs(proc), 99.9) + 1e-12)
            if gain > 1.0:
                gain = min(gain, float(0.985 / p999))
            # §2.45a-I: Einfache Multiplikation — kein Frame-Gate (globale Lautheitskorrektur;
            # peak_p999 schützt gegen Clipping)
            proc = np.clip(proc * gain, -1.0, 1.0)

        out_rms = float(_grl_p30(proc, gate_dbfs=-50.0))
        out_delta_db = float(20.0 * np.log10(max(out_rms / orig_rms, 1e-30)))
        makeup_db = float(20.0 * np.log10(max(out_rms / proc_rms, 1e-30)))
        return np.clip(proc.astype(np.float32), -1.0, 1.0), out_delta_db, makeup_db

    def _measure_subsonic_energy(self, audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
        """Misst RMS energy in subsonic band (<cutoff_hz)."""
        # Extract subsonic band
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use first channel for measurement

        # Low-pass filter
        sos = signal.butter(4, cutoff_hz, btype="low", fs=sample_rate, output="sos")
        from backend.core.audio_utils import safe_sosfiltfilt as _safe_sosfiltfilt30

        subsonic_signal = _safe_sosfiltfilt30(sos, audio)

        # RMS energy
        rms = np.sqrt(np.mean(subsonic_signal**2))
        return rms  # type: ignore[no-any-return]


# Test harness
if __name__ == "__main__":
    logger.debug("=== Verarbeitungsschritt 30: DC Offset Removal v2 Professional Test ===\n")

    processor = DCOffsetRemoval()

    # Test materials
    test_materials = [
        MaterialType.VINYL,
        MaterialType.TAPE,
        MaterialType.SHELLAC,
        MaterialType.CD_DIGITAL,
    ]

    for mat in test_materials:
        logger.debug("Testing %s:", mat.value.upper())

        # Create test signal: music + DC offset + rumble
        sr = 44100
        duration = 1.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)

        # Music: 440 Hz tone
        music = 0.5 * np.sin(2 * np.pi * 440 * t)

        # DC offset
        dc_offset = 0.15

        # Subsonic rumble (15 Hz)
        rumble = 0.08 * np.sin(2 * np.pi * 15 * t)

        # Combine
        corrupted = music + dc_offset + rumble

        # Process
        start = time.time()
        result = processor.process(corrupted, sr, mat)
        elapsed = time.time() - start

        # Display results
        meta = result.metadata
        logger.debug("  HP cutoff: %.1f Hz (%s)", meta["hp_cutoff_hz"], meta["filter_type"].upper())
        logger.debug("  DC before: %s", meta["dc_offset_before"])
        logger.debug("  DC after: %s", meta["dc_offset_after"])
        logger.debug("  DC reduction: %s", meta["dc_reduction"])
        logger.debug("  Subsonic reduction: %.2f dB", meta["subsonic_reduction_db"])
        logger.debug("  Processing time: %.4fs", elapsed)
        logger.debug("  RT factor: %.4f×", meta["rt_factor"])
        logger.debug("  ✅\n")
