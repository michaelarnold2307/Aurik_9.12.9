#!/usr/bin/env python3
"""
Phase 22: Professional Tape Saturation v2.0
=============================================

Multi-band analog tape saturation emulation with harmonic series modeling.

SCIENTIFIC FOUNDATION:
- Parker et al. (2014): Wave Digital Filters for Vacuum Tube Emulation
- Huovilainen (2004): Design of a Scalable Polyphonic Synthesizer
- Stilson & Smith (1996): Alias-Free Digital Synthesis of Classic Analog Waveforms
- Välimäki & Reiss (2008): All About Audio Equalization: Solutions and Frontiers
- Zölzer (2011): DAFX - Digital Audio Effects
- McNally (1984): Dynamic Range Control of Digital Audio Signals
- Hamada & Koizumi (1981): Analysis of Distortion in Tape Recording

INDUSTRY BENCHMARKS:
- Universal Audio Ampex ATR-102 (Industry standard tape emulation)
- Slate Digital Virtual Tape Machines (VTM)
- Softube Tape (Multi-track tape simulator)
- IK Multimedia Tape Machine Collection
- Waves J37 Tape (Abbey Road)
- McDSP Analog Channel (AC101/AC202)
- Acustica Audio Taupe (Tape suite)

ALGORITHM:
1. Multi-Band Processing (3 bands: Bass <300Hz, Mid 300-4k, High >4kHz)
   - Independent saturation per band
   - Different harmonic characteristics per band

2. Tape Speed Emulation
   - 15 IPS (high fidelity): Minimal saturation, extended HF
   - 7.5 IPS (standard): Moderate saturation, HF roll-off @ 18kHz
   - 3.75 IPS (vintage): Strong saturation, HF roll-off @ 12kHz

3. Harmonic Series Modeling
   - 2nd harmonic (even): Warmth, fullness
   - 3rd harmonic (odd): Bite, presence
   - 4th+ harmonics: Tape coloration

4. Tape Hysteresis
   - Asymmetric transfer function (positive > negative)
   - Frequency-dependent (more effect on LF)

5. Material-Adaptive Parameters
   - Tape: Strong saturation (authentic tape character)
   - Vinyl: Moderate (analog warmth without tape artifacts)
   - Digital: Subtle (analogizing digital sources)

QUALITY TARGETS:
- THD: 0.5-3% (tape-typical)
- Harmonic increase: +2 to +6 dB
- Processing: <0.2× realtime

Author: Aurik Professional Team
Version: 2.0.0
Date: February 2026
"""

import logging
import time

import numpy as np
from scipy import signal

from backend.core.audio_utils import to_channels_last
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class TapeSaturation(PhaseInterface):
    """Professional multi-band tape saturation emulation."""

    @staticmethod
    def _compute_tape_saturation_profile(
        material_type: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """Berechnet adaptive runtime profile for tape saturation."""
        mat = str(material_type or "unknown").lower().replace("-", "_").replace(" ", "_")
        qm = str(quality_mode or "balanced").lower().replace("-", "_")
        rest = float(np.clip(restorability_score, 0.0, 100.0))

        base = {
            "shellac": {"drive": 7.2, "h2": 0.078, "h3": 0.042, "h4": 0.018, "side": 0.23},
            "vinyl": {"drive": 9.4, "h2": 0.112, "h3": 0.057, "h4": 0.024, "side": 0.30},
            "tape": {"drive": 10.8, "h2": 0.134, "h3": 0.067, "h4": 0.030, "side": 0.36},
            "reel_tape": {"drive": 11.3, "h2": 0.141, "h3": 0.071, "h4": 0.032, "side": 0.38},
            "cassette": {"drive": 10.4, "h2": 0.128, "h3": 0.064, "h4": 0.029, "side": 0.35},
            "cd_digital": {"drive": 7.9, "h2": 0.088, "h3": 0.046, "h4": 0.020, "side": 0.24},
            "streaming": {"drive": 8.3, "h2": 0.095, "h3": 0.050, "h4": 0.022, "side": 0.26},
            "mp3_low": {"drive": 8.6, "h2": 0.100, "h3": 0.053, "h4": 0.023, "side": 0.28},
            "mp3_medium": {"drive": 8.9, "h2": 0.104, "h3": 0.055, "h4": 0.024, "side": 0.29},
            "unknown": {"drive": 8.8, "h2": 0.102, "h3": 0.054, "h4": 0.024, "side": 0.28},
        }.get(mat, {"drive": 8.8, "h2": 0.102, "h3": 0.054, "h4": 0.024, "side": 0.28})

        mode_adj = {
            "fast": -0.55,
            "balanced": 0.0,
            "quality": 0.40,
            "maximum": 0.62,
            "restoration": 0.25,
            "studio_2026": 0.62,
        }.get(qm, 0.0)
        rest_adj = ((rest - 50.0) / 50.0) * 0.32

        drive_gain_scalar = float(np.clip(base["drive"] + mode_adj + rest_adj, 4.0, 14.0))
        h2_scale = float(np.clip(base["h2"] + 0.015 * mode_adj + 0.012 * rest_adj, 0.050, 0.200))
        h3_scale = float(np.clip(base["h3"] + 0.010 * mode_adj + 0.008 * rest_adj, 0.025, 0.100))
        h4_scale = float(np.clip(base["h4"] + 0.005 * mode_adj + 0.004 * rest_adj, 0.010, 0.050))
        side_drive_fraction = float(np.clip(base["side"] + 0.030 * mode_adj + 0.020 * rest_adj, 0.15, 0.50))

        return {
            "drive_gain_scalar": drive_gain_scalar,
            "h2_scale": h2_scale,
            "h3_scale": h3_scale,
            "h4_scale": h4_scale,
            "side_drive_fraction": side_drive_fraction,
        }

    # Material-adaptive saturation drive (§v10.113: Lookup per material.value-String)
    SATURATION_DRIVE = {
        "shellac": 0.0,  # No tape (era mismatch)
        "vinyl": 0.30,  # Moderate analog warmth
        "tape": 0.55,  # Strong (authentic cassette tape)
        "reel_tape": 0.45,  # Professional reel-to-reel (slightly less than cassette)
        "cassette": 0.55,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        "cd_digital": 0.20,  # Subtle analogizing
        "streaming": 0.25,  # Light warmth
    }

    # Mix amount (wet/dry)
    SATURATION_MIX = {
        "shellac": 0.0,
        "vinyl": 0.40,  # 40% saturated
        "tape": 0.60,  # 60% saturated (cassette)
        "reel_tape": 0.50,  # 50% -- pro tape, fuller sound
        "cassette": 0.60,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        "cd_digital": 0.25,  # 25%
        "streaming": 0.30,  # 30%
    }

    # Tape speed (affects HF response and saturation)
    TAPE_SPEED = {
        "shellac": None,
        "vinyl": "7.5_ips",  # Standard
        "tape": "3.75_ips",  # Cassette tape: significant HF roll-off
        "reel_tape": "15_ips",  # Professional studio reel-to-reel
        "cassette": "3.75_ips",  # v10.0.0: IEC 60094-1 — Kassette läuft auf 3,75 ips (4,75 cm/s)
        "cd_digital": "15_ips",
        "streaming": "7.5_ips",
    }

    # Tape speed HF roll-off frequencies
    TAPE_SPEED_HF_ROLLOFF = {
        "15_ips": 20000,  # Minimal roll-off
        "7.5_ips": 18000,  # Standard roll-off
        "3.75_ips": 12000,  # Vintage roll-off
    }

    # Band split frequencies (3 bands: Bass, Mid, High)
    BAND_SPLIT_LOW = 300  # Bass < 300 Hz
    BAND_SPLIT_HIGH = 4000  # High > 4 kHz

    # Per-band saturation scaling
    BAND_DRIVE_SCALE = {
        "bass": 1.2,  # More saturation on bass (tape characteristic)
        "mid": 1.0,  # Standard
        "high": 0.7,  # Less saturation on highs (preserve clarity)
    }

    # Harmonic series weights (2nd, 3rd, 4th+)
    HARMONIC_WEIGHTS = {
        "bass": [0.6, 0.3, 0.1],  # 2nd dominant (warmth)
        "mid": [0.5, 0.4, 0.1],  # Balanced
        "high": [0.4, 0.5, 0.1],  # 3rd dominant (presence)
    }

    # Hysteresis amount (asymmetric saturation)
    HYSTERESIS_AMOUNT = {
        "shellac": 0.0,
        "vinyl": 0.12,
        "tape": 0.25,  # Tape-typical (cassette)
        "reel_tape": 0.20,  # Professional reel: wider tape, less hysteresis
        "cassette": 0.25,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        "cd_digital": 0.08,
        "streaming": 0.10,
    }

    def __init__(self):
        super().__init__()
        self.name = "Tape Saturation v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_22_tape_saturation",
            name="Tape Saturation v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_21_exciter"],
            estimated_time_factor=0.10,
            version="2.0.0",
            memory_requirement_mb=60,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.93,
            description="Multi-band analog tape saturation with harmonic modeling",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: MaterialType = MaterialType.VINYL,  # type: ignore[override]
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="22")
        """
        Wendet an: tape saturation.

        Args:
            audio: Audio samples (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type

        Returns:
            PhaseResult with saturated audio
        """
        # Interner Alias; unterstützt auch legacy-kwarg "material=" (UV3 nutzt material=)
        material = kwargs.get("material", material_type) or material_type
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        audio, _p22_transposed = to_channels_last(audio)
        start_time = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard: Stärke in post-transienten Masking-Fenstern erhöhen.
        _panns_s_22 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_22 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_22,
                )

                _fmg_22 = _fmg_fn_22()
                _fmz_22 = _fmg_22.compute_zones(audio, sample_rate)
                if _fmz_22:
                    _n_s_22 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_samples_22 = sum(z.end_sample - z.start_sample for z in _fmz_22)
                    _zone_frac_22 = float(np.clip(_zone_samples_22 / max(1, _n_s_22), 0.0, 1.0))
                    _boost_22 = _zone_frac_22 * 0.15
                    _effective_strength = float(np.clip(_effective_strength + _boost_22, 0.0, 1.0))
                    logger.debug(
                        "Verarbeitungsschritt22 §V41 ForwardMasking: zone_frac=%.2f boost=%.3f → eff_str=%.3f",
                        _zone_frac_22,
                        _boost_22,
                        _effective_strength,
                    )
            except Exception as _fmg_exc_22:
                logger.debug("Verarbeitungsschritt22 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_22)

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={
                    "saturation_applied": False,
                    "material": material.value,
                    "effective_strength": _effective_strength,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        quality_mode = str(kwargs.get("quality_mode", "balanced") or "balanced")
        restorability_score = float(kwargs.get("restorability_score", 75.0))
        material_key = material.value if isinstance(material, MaterialType) else str(material)
        tape_saturation_profile = self._compute_tape_saturation_profile(
            material_key,
            quality_mode,
            restorability_score,
        )

        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        drive = self.SATURATION_DRIVE.get(_mk, 0.25)  # type: ignore[arg-type]
        drive *= tape_saturation_profile["drive_gain_scalar"] / 10.0
        mix_amount = self.SATURATION_MIX.get(_mk, 0.30)  # type: ignore[arg-type]
        tape_speed = self.TAPE_SPEED.get(_mk, "7.5_ips")  # type: ignore[arg-type]
        hysteresis = self.HYSTERESIS_AMOUNT.get(_mk, 0.10)  # type: ignore[arg-type]

        # §2.54 tape_speed_ips kwarg override (from UV3 _restoration_context → _infer_tape_speed_ips)
        _injected_ips = kwargs.get("tape_speed_ips")
        if _injected_ips is not None:
            _ips_map = {1.875: "3.75_ips", 3.75: "3.75_ips", 7.5: "7.5_ips", 15.0: "15_ips"}
            _key = min(_ips_map.keys(), key=lambda k: abs(k - float(_injected_ips)))
            tape_speed = _ips_map[_key]

        # §5 Vintage Aesthetics + §2.14+ Era-adaptive saturation:
        # Pre-1960 recordings → preserve tube warmth (more drive, higher mix).
        # Post-1980 → reduce saturation to avoid coloring clean recordings.
        decade = kwargs.get("decade")
        if decade is not None:
            if decade <= 1950:
                drive = min(1.0, drive * 1.20)
                mix_amount = min(0.60, mix_amount * 1.15)
            elif decade >= 1980:
                drive = max(0.01, drive * 0.70)
                mix_amount = max(0.01, mix_amount * 0.70)

        # §2.20 Genre: if soft_saturation_preserve is set (Schlager, Rock),
        # boost mix to better preserve the original character.
        if kwargs.get("soft_saturation_preserve", False):
            mix_amount = min(0.60, mix_amount * 1.15)

        drive = float(drive * _effective_strength)
        mix_amount = float(mix_amount * _effective_strength)

        if drive < 0.01 or mix_amount < 0.01:
            # Skip processing
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={
                    "saturation_applied": False,
                    "material": material.value,
                    "effective_strength": _effective_strength,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_low_drive_or_mix",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        is_stereo = audio.ndim == 2

        if is_stereo:
            # §2.51: M/S-Domain — tape saturation (nichtlinear) nur auf Mid anwenden
            # Analog zu phase_21_exciter: "Excitation nur auf Mid, Side unverändert"
            _sqrt2_inv = 1.0 / np.sqrt(2.0)
            _mid = (audio[:, 0] + audio[:, 1]) * _sqrt2_inv
            _side = (audio[:, 0] - audio[:, 1]) * _sqrt2_inv

            # Full saturation on Mid (carries the musical content, tape intermodulation)
            _mid_sat = self._saturate_multi_band(
                _mid,
                sample_rate,
                drive,
                tape_speed,  # type: ignore[arg-type]
                hysteresis,
                tape_saturation_profile["h2_scale"],
                tape_saturation_profile["h3_scale"],
                tape_saturation_profile["h4_scale"],
            )

            # Side: reduced saturation to avoid L/R spatial artifacts
            _side_drive = drive * tape_saturation_profile["side_drive_fraction"]
            if _side_drive > 0.01:
                _side_sat = self._saturate_multi_band(
                    _side,
                    sample_rate,
                    _side_drive,
                    tape_speed,  # type: ignore[arg-type]
                    hysteresis * 0.5,
                    tape_saturation_profile["h2_scale"],
                    tape_saturation_profile["h3_scale"],
                    tape_saturation_profile["h4_scale"],
                )
            else:
                _side_sat = _side

            # Ensure length coherence
            _min_len = min(len(_mid_sat), len(_side_sat), audio.shape[0])
            _mid_sat = _mid_sat[:_min_len]
            _side_sat = _side_sat[:_min_len]

            # Reconstruct L/R from M/S
            _left = (_mid_sat + _side_sat) / np.sqrt(2.0)
            _right = (_mid_sat - _side_sat) / np.sqrt(2.0)
            saturated = np.column_stack([_left, _right])
        else:
            saturated = self._saturate_multi_band(
                audio,
                sample_rate,
                drive,
                tape_speed,  # type: ignore[arg-type]
                hysteresis,
                tape_saturation_profile["h2_scale"],
                tape_saturation_profile["h3_scale"],
                tape_saturation_profile["h4_scale"],
            )

        # Wet/dry mix
        if len(saturated) != len(audio):
            # Length mismatch: trim or pad
            if len(saturated) > len(audio):
                saturated = saturated[: len(audio)]
            else:
                if is_stereo:
                    saturated = np.pad(saturated, ((0, len(audio) - len(saturated)), (0, 0)), mode="edge")
                else:
                    saturated = np.pad(saturated, (0, len(audio) - len(saturated)), mode="edge")

        mixed = (1.0 - mix_amount) * audio + mix_amount * saturated

        if 0.0 < _effective_strength < 1.0:
            mixed = audio + _effective_strength * (mixed - audio)

        # Measure THD (Total Harmonic Distortion)
        thd_percent = self._estimate_thd(audio, mixed)

        # Measure harmonic increase
        harmonic_before = self._measure_harmonic_content(audio, sample_rate)
        harmonic_after = self._measure_harmonic_content(mixed, sample_rate)
        harmonic_increase_db = 20 * np.log10((harmonic_after + 1e-10) / (harmonic_before + 1e-10))

        processing_time = time.time() - start_time

        mixed = np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)
        mixed = np.clip(mixed, -1.0, 1.0)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung (§2.74, non-blocking): EQ-ähnliche Sättigungs-Änderung muss Spektralfarbe bewahren
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg22,
            )

            _sc22 = _scg22(audio, mixed, sample_rate)
            if not _sc22.ok:
                _sc22_wet = 0.70
                mixed = (_sc22_wet * mixed + (1.0 - _sc22_wet) * audio).astype(np.float32)
                logger.warning(
                    "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_22 spectral_color non-ok → strength −30%%"
                )
        except Exception as _sc22_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_22 spectral_color (nicht blockierend): %s", _sc22_exc
            )

        return PhaseResult(
            success=True,
            audio=mixed,
            metrics={
                "saturation_applied": True,
                "thd_percent": float(thd_percent),
                "harmonic_increase_db": float(harmonic_increase_db),
                "drive": drive,
                "mix_amount": mix_amount,
                "tape_speed": tape_speed,
                "hysteresis": hysteresis,
                "material": material.value,
            },
            execution_time_seconds=processing_time,
            metadata={
                "algorithm": "multi_band_tape_saturation",
                "version": "2.0",
                "bands": 3,
                "tape_saturation_profile": tape_saturation_profile,
                "quality_mode": quality_mode,
                "restorability_score": restorability_score,
            },
            modifications={
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
        )

    @staticmethod
    def _peak_envelope(audio: np.ndarray, attack_ms: float, release_ms: float, sr: int) -> np.ndarray:
        """Berechnet peak-follower envelope with separate attack/release time constants.

        Real magnetic tape does not saturate uniformly: loud transients compress
        harder than quiet passages because the oxide approaches saturation faster
        at higher flux levels.  This function tracks the signal envelope with
        analogue-like attack/release dynamics so that the drive can be modulated
        sample-by-sample, faithfully reproducing that level-dependent character.

        Scientific basis:
            McNally (1984). "Dynamic Range Control of Digital Audio Signals."
            Journal of the Audio Engineering Society, 32(5), 316-327.
            Attack < release mirrors tape transport mechanics: flux ramps up
            quickly but decays more slowly due to remanence.

        Args:
            audio:      Mono audio signal (1-D).
            attack_ms:  Envelope attack time constant in milliseconds.
            release_ms: Envelope release time constant in milliseconds.
            sr:         Sample rate in Hz.

        Returns:
            Peak envelope array, same length as audio, values in [0, +inf).
        """
        a_att = np.exp(-1.0 / max(sr * attack_ms * 1e-3, 1.0))
        a_rel = np.exp(-1.0 / max(sr * release_ms * 1e-3, 1.0))
        env = np.empty(len(audio), dtype=np.float64)
        env[0] = abs(float(audio[0]))
        for i in range(1, len(audio)):
            e = abs(float(audio[i]))
            if e > env[i - 1]:
                env[i] = a_att * env[i - 1] + (1.0 - a_att) * e
            else:
                env[i] = a_rel * env[i - 1]
        return env  # type: ignore[no-any-return]

    def _saturate_multi_band(
        self,
        audio: np.ndarray,
        sample_rate: int,
        drive: float,
        tape_speed: str,
        hysteresis: float,
        h2_scale: float,
        h3_scale: float,
        h4_scale: float,
    ) -> np.ndarray:
        """Multi-band tape saturation with envelope-following dynamic drive.

        Applies a peak-follower (McNally 1984) to the full-band signal to derive
        a time-varying drive_vec.  Each band then receives a sample-wise drive
        instead of a fixed scalar, so loud transients saturate proportionally
        harder than quiet passages -- matching the physical behaviour of magnetic
        tape oxide under varying flux levels.
        """
        nyquist = sample_rate / 2.0

        # Envelope-following dynamic drive (McNally 1984)
        # attack 3 ms / release 80 ms mirrors tape flux rise/decay physics.
        mono = np.mean(audio, axis=1).astype(np.float64) if audio.ndim == 2 else audio.astype(np.float64)
        env = self._peak_envelope(mono, attack_ms=3.0, release_ms=80.0, sr=sample_rate)
        p95 = np.percentile(env, 95) + 1e-8
        # Scale: quiet zones → ~40 % drive, loud zones → 100 % drive
        drive_vec = drive * np.clip(0.40 + 0.60 * env / p95, 0.40, 1.0).astype(np.float32)
        nyquist = sample_rate / 2.0

        # Design crossover filters (Linkwitz-Riley 4th order)
        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) statt sosfilt (kausal, Pegelexplosion).
        # Bass: < 300 Hz
        sos_bass_lp = signal.butter(4, self.BAND_SPLIT_LOW / nyquist, btype="lowpass", output="sos")
        bass = signal.sosfiltfilt(sos_bass_lp, audio)

        # High: > 4000 Hz
        sos_high_hp = signal.butter(4, self.BAND_SPLIT_HIGH / nyquist, btype="highpass", output="sos")
        high = signal.sosfiltfilt(sos_high_hp, audio)

        # Mid: 300-4000 Hz (residual — korrekt nur bei zero-phase LR)
        mid = audio - bass - high

        # Per-band saturation (pass drive_vec so each band is level-adaptive)
        bass_saturated = self._saturate_band(
            bass,
            drive_vec * self.BAND_DRIVE_SCALE["bass"],
            hysteresis,
            self.HARMONIC_WEIGHTS["bass"],
            h2_scale,
            h3_scale,
            h4_scale,
        )
        mid_saturated = self._saturate_band(
            mid,
            drive_vec * self.BAND_DRIVE_SCALE["mid"],
            hysteresis,
            self.HARMONIC_WEIGHTS["mid"],
            h2_scale,
            h3_scale,
            h4_scale,
        )
        high_saturated = self._saturate_band(
            high,
            drive_vec * self.BAND_DRIVE_SCALE["high"],
            hysteresis,
            self.HARMONIC_WEIGHTS["high"],
            h2_scale,
            h3_scale,
            h4_scale,
        )

        # Recombine bands
        saturated = bass_saturated + mid_saturated + high_saturated

        # Apply tape speed HF roll-off
        if tape_speed and tape_speed in self.TAPE_SPEED_HF_ROLLOFF:
            hf_rolloff = self.TAPE_SPEED_HF_ROLLOFF[tape_speed]
            if hf_rolloff < nyquist * 0.95:
                sos_tape_hf = signal.butter(2, hf_rolloff / nyquist, btype="lowpass", output="sos")
                from backend.core.audio_utils import safe_sosfiltfilt as _safe_sosfiltfilt22

                saturated = _safe_sosfiltfilt22(sos_tape_hf, saturated)

        # Soft limiter (prevent clipping) — §2.49 Peak-Guard: percentile(99.9) so single
        # crackle/click impulses do not block normalization of the musical signal.
        peak = float(np.percentile(np.abs(saturated), 99.9))
        if peak > 0.95:
            saturated = saturated * (0.95 / peak)

        return np.nan_to_num(np.asarray(saturated, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    @staticmethod
    def _tanh_adaa(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """1st-order Antiderivative Antialiasing for tanh.

        Computes (F(x0) - F(x1)) / (x0 - x1) where F(x) = log(cosh(x)) is
        the antiderivative of tanh.  Midpoint fallback when |x0 - x1| < 1e-7.

        Scientific basis:
            Parker, Esqueda & Bergner (2019). "Antiderivative Antialiasing for
            Stateless and Stateful Nonlinearities." IEEE SPL 26(3), 357-361.
        """
        dX = x0 - x1
        close = np.abs(dX) < 1e-7

        def _log_cosh(x: np.ndarray) -> np.ndarray:
            ax = np.abs(x)
            return np.nan_to_num(np.asarray(ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0), dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

        midpoint = np.tanh(0.5 * (x0 + x1))
        adaa = (_log_cosh(x0) - _log_cosh(x1)) / np.where(close, 1.0, dX)
        return np.nan_to_num(np.where(close, midpoint, adaa), nan=0.0)  # type: ignore[no-any-return]

    def _saturate_band(
        self,
        audio: np.ndarray,
        drive: "float | np.ndarray",
        hysteresis: float,
        harmonic_weights: list,
        h2_scale: float = 0.10,
        h3_scale: float = 0.05,
        h4_scale: float = 0.02,
    ) -> np.ndarray:
        """Saturate a single band using ADAA-processed tanh with hysteresis.

        Accepts `drive` as either a scalar float or a 1-D numpy array of the
        same length as `audio` (envelope-following dynamic drive, McNally 1984).
        1st-order Antiderivative Antialiasing (Parker et al. 2019) replaces
        direct np.tanh calls to suppress aliased harmonics that would otherwise
        fold back from above Nyquist into the audio band at high drive levels.
        The asymmetric positive/negative processing (hysteresis) models the
        asymmetric flux response of real tape oxide (Hamada & Koizumi 1981).
        """
        # Drive stage — works for scalar and vector drive
        driven = audio * (1.0 + drive * 8.0)

        # Previous-sample reference for ADAA (causal, 1-sample look-back)
        prev = np.roll(driven, 1)
        prev[0] = 0.0

        prev_neg = prev * (1.0 - hysteresis)

        # ADAA tanh: positive half — standard, negative half — hysteresis-scaled
        saturated = np.where(
            driven >= 0,
            self._tanh_adaa(driven, prev),
            self._tanh_adaa(driven * (1.0 - hysteresis), prev_neg),
        )

        # Add harmonics (2nd, 3rd, 4th)
        # 2nd harmonic (even): saturated^2 (scaled)
        h2 = saturated**2 * np.sign(saturated) * harmonic_weights[0] * h2_scale

        # 3rd harmonic (odd): saturated^3
        h3 = saturated**3 * harmonic_weights[1] * h3_scale

        # 4th+ harmonics (subtle)
        h4 = saturated**4 * np.sign(saturated) * harmonic_weights[2] * h4_scale

        # Mix harmonics
        saturated_with_harmonics = saturated + h2 + h3 + h4

        # Normalize to prevent clipping — §2.49 Peak-Guard: percentile(99.9)
        peak = float(np.percentile(np.abs(saturated_with_harmonics), 99.9))
        if peak > 1.0:
            saturated_with_harmonics /= peak

        return np.nan_to_num(np.asarray(saturated_with_harmonics, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    def _estimate_thd(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Schätzt THD (Total Harmonic Distortion) as RMS difference.
        """
        if original.ndim == 2:
            original = np.mean(original, axis=1)
        if processed.ndim == 2:
            processed = np.mean(processed, axis=1)

        # Ensure same length
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        # Difference = added harmonics/distortion
        difference = processed - original

        rms_original = np.sqrt(np.mean(original**2))
        rms_difference = np.sqrt(np.mean(difference**2))

        thd_percent = rms_difference / rms_original * 100.0 if rms_original > 1e-10 else 0.0

        return min(thd_percent, 100.0)

    def _measure_harmonic_content(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Misst harmonic content (energy in harmonic band).
        """
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # FFT
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)

        # Harmonic band (200 Hz - 5 kHz)
        harmonic_band = (freqs >= 200) & (freqs <= 5000)

        harmonic_energy = np.mean(spectrum[harmonic_band]) if np.any(harmonic_band) else 0.0

        return harmonic_energy  # type: ignore[return-value]


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Verarbeitungsschritt 22: Professional Tape Saturation v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio (pure sine for THD measurement)
    duration = 2.0
    _sr = 48000

    t = np.linspace(0, duration, int(_sr * duration))

    # Pure 440 Hz sine (A4) at moderate level
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    logger.debug("erzeugt %ss test audio @ %s Hz", duration, _sr)
    logger.debug("Signal: Pure 440 Hz sine (A4)")
    logger.debug("Purpose: Measure THD and harmonic addition")
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE"),
        (MaterialType.VINYL, "VINYL"),
        (MaterialType.CD_DIGITAL, "CD_DIGITAL"),
    ]

    for _test_mat, material_name in materials:
        logger.debug("─" * 80)
        logger.debug("Material: %s", material_name)
        logger.debug("─" * 80)
        logger.debug("")

        phase = TapeSaturation()
        result = phase.process(test_signal, _sr, _test_mat)

        logger.debug("✅ Professional Tape Saturation:")
        logger.debug("   THD: %.2f%%", result.metrics["thd_percent"])
        logger.debug("   Harmonic Increase: %.2f dB", result.metrics["harmonic_increase_db"])
        logger.debug("   Drive: %.2f", result.metrics["drive"])
        logger.debug("   Mix Amount: %s", format(result.metrics["mix_amount"], ".0%"))
        logger.debug("   Tape Speed: %s", result.metrics["tape_speed"])
        logger.debug("   Hysteresis: %.2f", result.metrics["hysteresis"])
        logger.debug(
            "   Processing time: %.3fs (%.2f\u00d7 realtime)",
            result.execution_time_seconds,
            result.execution_time_seconds / duration,
        )
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test abgeschlossen")
    logger.debug("=" * 80)
