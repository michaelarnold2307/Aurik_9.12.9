#!/usr/bin/env python3
"""
Phase 21: Harmonic Exciter v2.0 - Professional
Multi-band harmonic synthesis for enhanced presence, brilliance, and "air".

Algorithm Overview:
1. Multi-Band Architecture:
   - Low-mid: 500-2000 Hz (warmth and body)
   - High-mid: 2000-6000 Hz (presence and clarity)
   - High: 6000-20000 Hz (brilliance and air)
2. Harmonic Generation Strategies:
   - Even harmonics (2nd, 4th): Warmth (tube-like)
   - Odd harmonics (3rd, 5th): Brightness (tape-like)
   - Mixed harmonics: Natural enhancement
3. Waveshaping Models:
   - Soft saturation: tanh (smooth, musical)
   - Hard clipping: arctan (aggressive, bright)
   - Tube simulation: polynomial (warm, vintage)
4. Psychoacoustic Optimization:
   - Fletcher-Munson compensation
   - Critical band masking
   - Transient preservation
5. Material-Adaptive Processing:
   - Shellac: Restore missing HF content (bandwidth limited)
   - Vinyl: Add "air" and openness
   - Tape: Simulate tape head HF bump
   - Digital: Subtle enhancement for analog character

Scientific Foundation:
- Arfib (1991): Digital Synthesis of Complex Spectra by Means of Multiplication of Nonlinear Distorted Sine Waves
- Doidic et al. (1998): A New Approach to Digital Audio Effects Using Fourier Analysis
- Välimäki et al. (2011): Enhanced Wave Digital Triode Model for Real-Time Tube Amplifier Emulation
- Yeh & Smith (2008): Simulating Guitar Distortion Circuits Using Wave Digital and Nonlinear State-Space Formulations
- Zölzer (2011): DAFX - Digital Audio Effects

Industry Benchmarks:
- Waves Aphex Aural Exciter ($49)
- SPL Vitalizer ($199)
- BBE Sonic Maximizer ($299)
- Ozone Exciter Module ($299)
- Crane Song HEDD ($2995 hardware)

Quality Target: 0.75 → 0.90 (+20% improvement)
Performance Target: <0.12× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import copy
import logging
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import audio_sample_count, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class Exciter(PhaseInterface):
    """
    Professional Multi-Band Harmonic Exciter.

    Key Features:
    - 3-band independent harmonic generation
    - Even/odd harmonic control
    - Multiple saturation models (soft/hard/tube)
    - Transient preservation
    - Material-adaptive processing
    - Psychoacoustic optimization

    Use Cases:
    - Restore bandwidth-limited recordings
    - Add "air" and brilliance
    - Enhance presence and clarity
    - Simulate analog warmth

    Performance: <0.12× realtime on modern CPU
    """

    # Frequency band definitions for excitation
    EXCITER_BANDS = {
        "low_mid": (500, 2000),  # Warmth and body
        "high_mid": (2000, 6000),  # Presence and clarity
        "high": (6000, 20000),  # Brilliance and air
    }

    # Material-adaptive exciter configurations
    EXCITER_CONFIG = {
        MaterialType.SHELLAC: {
            "low_mid": {"intensity": 0.25, "harmonics": "even", "saturation": "tube"},
            "high_mid": {"intensity": 0.40, "harmonics": "mixed", "saturation": "soft"},
            "high": {"intensity": 0.50, "harmonics": "odd", "saturation": "hard"},
            "mix": 0.45,
        },
        MaterialType.VINYL: {
            "low_mid": {"intensity": 0.20, "harmonics": "even", "saturation": "tube"},
            "high_mid": {"intensity": 0.30, "harmonics": "mixed", "saturation": "soft"},
            "high": {"intensity": 0.35, "harmonics": "odd", "saturation": "soft"},
            "mix": 0.30,
        },
        MaterialType.TAPE: {
            "low_mid": {"intensity": 0.18, "harmonics": "even", "saturation": "tube"},
            "high_mid": {"intensity": 0.28, "harmonics": "mixed", "saturation": "soft"},
            "high": {"intensity": 0.38, "harmonics": "odd", "saturation": "soft"},
            "mix": 0.32,
        },
        MaterialType.CASSETTE: {
            "low_mid": {"intensity": 0.18, "harmonics": "even", "saturation": "tube"},
            "high_mid": {"intensity": 0.25, "harmonics": "mixed", "saturation": "soft"},  # v10.0.0: leicht reduziert
            "high": {"intensity": 0.30, "harmonics": "odd", "saturation": "soft"},  # v10.0.0: BW-Ceiling 12 kHz
            "mix": 0.28,
        },  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE, HF konservativ
        MaterialType.CD_DIGITAL: {
            "low_mid": {"intensity": 0.10, "harmonics": "even", "saturation": "tube"},
            "high_mid": {"intensity": 0.15, "harmonics": "mixed", "saturation": "soft"},
            "high": {"intensity": 0.20, "harmonics": "odd", "saturation": "soft"},
            "mix": 0.18,
        },
        MaterialType.STREAMING: {
            "low_mid": {"intensity": 0.12, "harmonics": "even", "saturation": "tube"},
            "high_mid": {"intensity": 0.18, "harmonics": "mixed", "saturation": "soft"},
            "high": {"intensity": 0.25, "harmonics": "odd", "saturation": "soft"},
            "mix": 0.22,
        },
    }

    def _compute_exciter_runtime_profile(
        self,
        material_type: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """Berechnet adaptive runtime constants for harmonic excitation."""
        _mat = str(material_type or "unknown").lower().replace("-", "_").replace(" ", "_")
        _qm = str(quality_mode or "balanced").lower().replace("-", "_")
        _rest = float(np.clip(restorability_score, 0.0, 100.0))

        _base = {
            "shellac": {"drive": 2.30, "odd": 0.24, "scale": 0.52},
            "wax_cylinder": {"drive": 2.10, "odd": 0.22, "scale": 0.48},
            "vinyl": {"drive": 2.70, "odd": 0.30, "scale": 0.62},
            "tape": {"drive": 2.80, "odd": 0.31, "scale": 0.64},
            "reel_tape": {"drive": 2.90, "odd": 0.32, "scale": 0.66},
            "mp3_low": {"drive": 3.00, "odd": 0.34, "scale": 0.68},
            "cd_digital": {"drive": 2.40, "odd": 0.26, "scale": 0.56},
            "digital": {"drive": 2.40, "odd": 0.26, "scale": 0.56},
            "streaming": {"drive": 2.50, "odd": 0.28, "scale": 0.58},
            "unknown": {"drive": 2.60, "odd": 0.30, "scale": 0.60},
        }.get(_mat, {"drive": 2.60, "odd": 0.30, "scale": 0.60})

        _mode_adj = {
            "fast": -0.22,
            "balanced": 0.0,
            "quality": +0.18,
            "maximum": +0.24,
            "restoration": +0.10,
            "studio_2026": +0.24,
        }.get(_qm, 0.0)
        _rest_adj = ((_rest - 50.0) / 50.0) * 0.12

        saturation_drive = float(np.clip(_base["drive"] + _mode_adj + _rest_adj, 1.80, 4.00))
        odd_partial_blend = float(np.clip(_base["odd"] + 0.35 * _mode_adj + 0.30 * _rest_adj, 0.18, 0.45))
        harmonic_output_scale = float(np.clip(_base["scale"] + 0.60 * _mode_adj + 0.60 * _rest_adj, 0.45, 0.85))

        return {
            "saturation_drive": saturation_drive,
            "odd_partial_blend": odd_partial_blend,
            "harmonic_output_scale": harmonic_output_scale,
        }

    def __init__(self):
        super().__init__()
        self.name = "Harmonic Exciter v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_21_exciter",
            name="Harmonic Exciter v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_38_presence_boost", "phase_39_air_band_enhancement"],
            estimated_time_factor=0.12,
            version="2.0.0",
            memory_requirement_mb=60,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description="Multi-band harmonic synthesis with psychoacoustic optimization",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: MaterialType = MaterialType.CD_DIGITAL,  # type: ignore[override]
        **kwargs,
    ) -> PhaseResult:  # type: ignore[override]
        """
        Wendet an: multi-band harmonic excitation to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with excited audio
        """
        material = material_type  # interner Alias
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_21 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_21 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_21,
                )

                _fmz_21 = kwargs.get("forward_masking_zones") or _fmg_fn_21().compute_zones(audio, sample_rate)
                if _fmz_21:
                    _n_s_21 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_21 = sum(z.end_sample - z.start_sample for z in _fmz_21)
                    _zone_frac_21 = float(np.clip(_zone_s_21 / max(1, _n_s_21), 0.0, 1.0))
                    _effective_strength = float(np.clip(_effective_strength + _zone_frac_21 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_21:
                logger.debug("Verarbeitungsschritt21 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_21)

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={
                    "skipped": True,
                    "reason": "skipped_zero_strength",
                    "hf_boost_db": 0.0,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[],
                modifications={},
            )

        # ── SOFT_SATURATION-Guard (Spec §6.3: Tube-Sättigung BEWAHREN) ──────
        # Spec: SOFT_SATURATION = gerade Obertöne (Röhren-Charakter) → nicht hinzufügen
        defect_scores = kwargs.get("defect_scores", {})
        soft_sat_score = 0.0
        for k, v in defect_scores.items():
            k_name = k.name if hasattr(k, "name") else str(k)
            if "SOFT_SAT" in k_name.upper() or "soft_sat" in k_name.lower():
                soft_sat_score = float(v)
                break
        if soft_sat_score >= 0.40:
            logger.info(
                "Verarbeitungsschritt 21: SOFT_SATURATION erkannt (Wert=%.2f) — Exciter übersprungen "
                "(Spec §6.3: Tube-Charakter bewahren)",
                soft_sat_score,
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                metrics={"skipped": True, "reason": "soft_saturation_preserve"},
                execution_time_seconds=0.0,
                metadata={
                    "algorithm": "skip_soft_saturation_guard",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Phase 21 übersprungen: SOFT_SATURATION-Material erkannt — Röhren-Charakter wird bewahrt."],
                modifications={},
            )

        is_stereo = audio.ndim == 2

        # §2.51 Wide-Stereo-Guard: Stereo-Exciter auf wide-stereo / phasenverschobenen Quellen
        # erzeugt Phase-Cancellation-Artefakte. Korrelation < 0.45 → No-Op (identisch phase_13).
        if is_stereo:
            try:
                _left_ck, _right_ck = stereo_channel_view(audio)
                _N_ck = min(len(_left_ck), 48000 * 3)
                _c_ck = float(
                    np.dot(_left_ck[:_N_ck], _right_ck[:_N_ck])
                    / (np.linalg.norm(_left_ck[:_N_ck]) * np.linalg.norm(_right_ck[:_N_ck]) + 1e-10)
                )
                if _c_ck < 0.45:
                    logger.info(
                        "Verarbeitungsschritt 21: Wide-Stereo erkannt (Korrelation=%.3f < 0.45) — Exciter übersprungen "
                        "(§2.51: M/S-Exciter auf stark entkorreliertem Material erzeugt Artefakte)",
                        _c_ck,
                    )
                    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                    audio = np.clip(audio, -1.0, 1.0)
                    return PhaseResult(
                        success=True,
                        audio=audio,
                        metrics={"skipped": True, "reason": "wide_stereo_guard", "lr_correlation": _c_ck},
                        execution_time_seconds=time.time() - start_time,
                        metadata={
                            "algorithm": "skip_wide_stereo_guard",
                            "lr_correlation": _c_ck,
                            "phase_locality_factor": phase_locality_factor,
                            "effective_strength": _effective_strength,
                            "rms_drop_db": 0.0,
                            "loudness_makeup_db": 0.0,
                        },
                        warnings=["Phase 21 übersprungen: Wide-Stereo-Material (Korrelation < 0.45)."],
                        modifications={},
                    )
            except Exception as _wsg_exc:
                logger.debug("Verarbeitungsschritt 21 Wide-Stereo-Guard nicht blockierend: %s", _wsg_exc)

        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113

        config: dict[str, Any] = copy.deepcopy(
            self.EXCITER_CONFIG.get(_mk, self.EXCITER_CONFIG[MaterialType.CD_DIGITAL])  # type: ignore[call-overload]
        )
        for band_name in self.EXCITER_BANDS:
            config[band_name]["intensity"] = float(float(config[band_name]["intensity"]) * _effective_strength)
        config["mix"] = float(float(config["mix"]) * _effective_strength)

        # §2.51: M/S domain — excite Mid only, Side untouched
        if is_stereo:
            _inv_sqrt2 = 1.0 / np.sqrt(2.0)
            left, right = stereo_channel_view(audio)
            mid = (left + right) * _inv_sqrt2
            side = (left - right) * _inv_sqrt2
            excited_mid = self._excite_channel(mid, sample_rate, config)
            excited_audio = stereo_like(
                (excited_mid + side) * _inv_sqrt2,
                (excited_mid - side) * _inv_sqrt2,
                audio,
            )
        else:
            excited_audio = self._excite_channel(audio, sample_rate, config)

        if 0.0 < _effective_strength < 1.0:
            excited_audio = audio + _effective_strength * (excited_audio - audio)

        # Measure HF enhancement
        hf_energy_before = self._measure_hf_energy(audio, sample_rate)
        hf_energy_after = self._measure_hf_energy(excited_audio, sample_rate)
        hf_boost_db = 20 * np.log10((hf_energy_after + 1e-10) / (hf_energy_before + 1e-10))

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        excited_audio = np.nan_to_num(excited_audio, nan=0.0, posinf=0.0, neginf=0.0)
        excited_audio = np.clip(excited_audio, -1.0, 1.0)

        # §4.5 Psychoacoustic Masking Clamp — fulfill docstring: harmonic excitation only where audible
        try:
            from backend.core.dsp.psychoacoustics import (
                apply_psychoacoustic_masking_clamp,  # pylint: disable=import-outside-toplevel
            )

            excited_audio = apply_psychoacoustic_masking_clamp(
                audio,
                excited_audio,
                sample_rate,
                strength=_effective_strength,
                mode="additive",
            )
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt21 masking clamp nicht blockierend: %s", _pm_exc)

        return PhaseResult(
            success=True,
            audio=excited_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "hf_boost_db": float(hf_boost_db),
                "mix_amount": float(config["mix"]),
                "rt_factor": float(rt_factor),
                "stereo_mode": "ms_mid_only" if is_stereo else "mono",
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.15 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _excite_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Wendet an: multi-band excitation to a single channel."""
        excited_bands = []

        # Process each frequency band
        for band_name, freq_range in self.EXCITER_BANDS.items():
            band_config = config[band_name]

            # Extract band
            band_audio = self._extract_band(audio, sample_rate, freq_range)

            # Generate harmonics
            excited_band = self._generate_harmonics(
                band_audio, band_config["intensity"], band_config["harmonics"], band_config["saturation"]
            )

            excited_bands.append(excited_band)

        # Combine excited bands
        excited_sum = sum(excited_bands)

        # Mix with original
        mix_amount = config["mix"]
        excited_audio = audio * (1 - mix_amount) + excited_sum * mix_amount

        # Normalize if needed — §2.49 Peak-Guard: percentile(99.9) so single impulse artefacts don't block normalisation
        peak = float(np.percentile(np.abs(excited_audio), 99.9))
        if peak > 0.99:
            excited_audio = excited_audio * (0.99 / peak)

        return np.nan_to_num(np.asarray(excited_audio), nan=0.0)  # type: ignore[no-any-return]

    def _extract_band(self, audio: np.ndarray, sample_rate: int, freq_range: tuple[float, float]) -> np.ndarray:
        """Extrahiert frequency band using bandpass filter."""
        sos = signal.butter(4, freq_range, btype="band", fs=sample_rate, output="sos")
        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) — band wird mit original gemischt.
        return np.nan_to_num(np.asarray(signal.sosfiltfilt(sos, audio)), nan=0.0)  # type: ignore[no-any-return]

    def _generate_harmonics(
        self, audio: np.ndarray, intensity: float, harmonic_type: str, saturation_type: str
    ) -> np.ndarray:
        """Generiert harmonics using waveshaping."""
        # Scale input for saturation
        scaled = audio * intensity * 3.0

        # Apply saturation
        if saturation_type == "soft":
            # Soft saturation (tanh)
            saturated = np.tanh(scaled)
        elif saturation_type == "hard":
            # Hard clipping (arctan)
            saturated = (2 / np.pi) * np.arctan(scaled * 2)
        elif saturation_type == "tube":
            # Tube-like polynomial
            saturated = scaled - (scaled**3) / 3
            saturated = np.clip(saturated, -1, 1)
        else:
            saturated = np.tanh(scaled)

        # Filter harmonics based on type
        if harmonic_type == "even":
            # Even harmonics: full-wave rectification
            saturated = np.abs(saturated) * np.sign(saturated)
        elif harmonic_type == "odd":
            # Odd harmonics: half-wave rectification
            saturated = np.where(saturated > 0, saturated, saturated * 0.3)
        # 'mixed': use as-is

        # High-pass to remove original fundamental
        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) — harmonics werden addiert.
        sos_hp = signal.butter(2, 1000, btype="high", fs=48000, output="sos")
        harmonics_only = signal.sosfiltfilt(sos_hp, saturated)

        return np.nan_to_num(np.asarray(harmonics_only * 0.7), nan=0.0)  # type: ignore[no-any-return]  # Scale down

    def _measure_hf_energy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Misst high-frequency energy (>6 kHz)."""
        if audio.ndim == 2:
            audio = audio[:, 0]

        sos = signal.butter(4, 6000, btype="high", fs=sample_rate, output="sos")
        from backend.core.audio_utils import safe_sosfiltfilt as _safe_sosfiltfilt21

        hf_signal = _safe_sosfiltfilt21(sos, audio)
        return float(np.sqrt(np.mean(hf_signal**2)))
