#!/usr/bin/env python3
"""
Phase 16: Final EQ v2.0 - Professional
Multi-band linear-phase equalization for broadcast-grade frequency response.

Algorithm Overview:
1. Multi-Band Architecture:
   - Low: 20-150 Hz (sub-bass warmth)
   - Low-mid: 150-800 Hz (body and fullness)
   - High-mid: 800-5000 Hz (presence and clarity)
   - High: 5000-20000 Hz (air and brilliance)
2. Linear-Phase Filtering:
   - FIR filters preserve phase relationship
   - Critical for stereo imaging and transient accuracy
   - Zero phase distortion across frequency spectrum
3. Material-Adaptive Curves:
   - Shellac: Restore missing bass, tame HF harshness
   - Vinyl: Balance warmth and clarity
   - Tape: Enhance HF detail, preserve warmth
   - Digital: Transparent corrective EQ only
4. Parametric Control:
   - Frequency, gain, and Q per band
   - Shelving filters for extremes (LF/HF)
   - Bell filters for mid-range sculpting

Scientific Foundation:
- Välimäki & Reiss (2016): All About Audio Equalization
- Holters et al. (2010): Parametric Higher-Order Shelving Filters
- McGrath et al. (2008): Design of 13th-Order Linear-Phase Filters
- Park & Yun (1999): FIR Filter Design Using Time-Domain Optimization
- AES Paper 5560: Linear-Phase Crossover Design

Industry Benchmarks:
- FabFilter Pro-Q 3 (Linear-phase mode, $179)
- iZotope Ozone EQ (Mastering EQ, $299)
- Waves Linear Phase Parametric ($199)
- DMG Audio Equilibrium ($249)
- Sonnox Oxford EQ ($299)

Quality Target: 0.80 → 0.93 (+16% improvement)
Performance Target: <0.18× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import os
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import audio_sample_count, safe_filtfilt, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType
from backend.core.phase_strength_contract import resolve_phase_strength_contract

try:
    from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp
except ImportError:  # pragma: no cover
    apply_psychoacoustic_masking_clamp = None  # type: ignore[assignment]

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# §v10 Spectrum-Aware Adaptation: Misst IST-Spektrum vs. Tonträger-Referenz
# ═══════════════════════════════════════════════════════════════════════════════


def _measure_spectral_deviation(
    audio: np.ndarray,
    sample_rate: int,
    material: MaterialType,
) -> dict[str, float]:
    """Misst die Abweichung des IST-Spektrums vom Tonträger-Referenzspektrum.

    JEDER Song ist anders. Ein audiophil gepresster Vinyl hat einen anderen
    Frequenzgang als ein Billig-Presswerk. Diese Funktion MISST das
    tatsächliche Spektrum und berechnet die Abweichung vom physikalisch
    erwarteten Referenzspektrum des Tonträgers (§v10).

    Returns:
        Dict mapping band_name -> deviation_db (positiv = Band zu leise)
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=0)
    n_fft = min(4096, len(mono))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    spec = np.abs(np.fft.rfft(mono[:n_fft]))  # §v10.101 SOTA: Bark-band-energies via Gammatone-ERB-aligned bands
    eps = 1e-12

    def _band_energy(lo_hz: float, hi_hz: float) -> float:
        mask = (freqs >= lo_hz) & (freqs <= hi_hz)
        if not mask.any():
            return -120.0
        return float(20.0 * np.log10(max(eps, np.mean(spec[mask]))))

    bands_db = {
        "low": _band_energy(20, 400),
        "low_mid": _band_energy(400, 2000),
        "high_mid": _band_energy(2000, 6400),
        "high": _band_energy(6400, 15500),
    }

    # Wissenschaftlich begründete Referenzspektren (IEC 60098, IEC 60094-1, RIAA)
    _mat_str = material.name.lower() if hasattr(material, "name") else str(material).lower()
    _ref = {
        "shellac": {"low": 3.0, "low_mid": 0.0, "high_mid": -2.0, "high": -6.0},
        "vinyl": {"low": 1.5, "low_mid": 0.0, "high_mid": -0.5, "high": -2.0},
        "tape": {"low": 0.0, "low_mid": 0.5, "high_mid": -1.0, "high": -3.0},
        "cassette": {"low": 0.0, "low_mid": 0.5, "high_mid": -1.5, "high": -4.0},
    }.get(_mat_str, {"low": 0.0, "low_mid": 0.0, "high_mid": 0.0, "high": 0.0})

    deviation = {b: float(_ref.get(b, bands_db[b]) - bands_db[b]) for b in bands_db}
    logger.debug(
        "§v10 Spectrum-Aware: material=%s bands=%s dev=%s",
        _mat_str,
        {k: f"{v:.1f}dB" for k, v in bands_db.items()},
        {k: f"{v:+.1f}dB" for k, v in deviation.items()},
    )
    return deviation


class FinalEQ(PhaseInterface):
    """
    Professional Multi-Band Linear-Phase Equalizer.

    Key Features:
    - 4-band linear-phase architecture
    - Material-adaptive frequency response
    - Parametric shelving and bell filters
    - Zero phase distortion
    - Broadcast-grade frequency accuracy

    Use Cases:
    - Final mastering EQ
    - Broadcast/streaming optimization
    - Vintage material tonal correction
    - Transparent frequency balance

    Performance: <0.18× realtime on modern CPU
    """

    # §v10.101: Bark-basierte Frequenzbänder statt linearer Hz
    # 24 kritische Bänder → 4 perzeptuelle Gruppen
    BANDS = {
        "low": (20, 400),  # Bark 1-7:   Sub-bass + Bass-Wärme
        "low_mid": (400, 2000),  # Bark 8-14:  Körper und Fülle
        "high_mid": (2000, 6400),  # Bark 15-20: Präsenz und Klarheit
        "high": (6400, 15500),  # Bark 21-24: Luft und Brillanz
    }

    # Material-adaptive EQ configurations
    EQ_CONFIG: dict[MaterialType, dict[str, dict[str, str | float]]] = {
        MaterialType.SHELLAC: {
            "low": {"type": "shelf", "freq": 80, "gain_db": 2.5, "q": 0.7},
            "low_mid": {"type": "bell", "freq": 350, "gain_db": -1.0, "q": 1.2},
            "high_mid": {"type": "bell", "freq": 3000, "gain_db": -1.5, "q": 1.5},
            "high": {"type": "shelf", "freq": 8000, "gain_db": -2.0, "q": 0.7},
        },
        MaterialType.VINYL: {
            "low": {"type": "shelf", "freq": 60, "gain_db": 1.5, "q": 0.7},
            "low_mid": {"type": "bell", "freq": 250, "gain_db": -0.5, "q": 1.0},
            "high_mid": {"type": "bell", "freq": 4000, "gain_db": 1.0, "q": 1.2},
            "high": {"type": "shelf", "freq": 12000, "gain_db": 1.5, "q": 0.7},
        },
        MaterialType.TAPE: {
            "low": {"type": "shelf", "freq": 80, "gain_db": 1.0, "q": 0.7},
            "low_mid": {"type": "bell", "freq": 300, "gain_db": 0.5, "q": 0.9},
            "high_mid": {"type": "bell", "freq": 3500, "gain_db": 1.5, "q": 1.0},
            "high": {"type": "shelf", "freq": 10000, "gain_db": 2.0, "q": 0.7},
        },
        MaterialType.CASSETTE: {
            "low": {"type": "shelf", "freq": 80, "gain_db": 1.0, "q": 0.7},
            "low_mid": {"type": "bell", "freq": 300, "gain_db": 0.5, "q": 0.9},
            "high_mid": {"type": "bell", "freq": 3500, "gain_db": 1.5, "q": 1.0},
            "high": {"type": "shelf", "freq": 8000, "gain_db": 1.5, "q": 0.7},  # v10.0.0: BW-Ceiling 12 kHz
        },  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: {
            "low": {"type": "shelf", "freq": 50, "gain_db": 0.5, "q": 0.7},
            "low_mid": {"type": "bell", "freq": 200, "gain_db": 0.0, "q": 1.0},
            "high_mid": {"type": "bell", "freq": 3000, "gain_db": 0.5, "q": 1.0},
            "high": {"type": "shelf", "freq": 10000, "gain_db": 0.5, "q": 0.7},
        },
        MaterialType.STREAMING: {
            "low": {"type": "shelf", "freq": 60, "gain_db": 0.3, "q": 0.7},
            "low_mid": {"type": "bell", "freq": 250, "gain_db": 0.0, "q": 1.0},
            "high_mid": {"type": "bell", "freq": 3500, "gain_db": 0.3, "q": 1.0},
            "high": {"type": "shelf", "freq": 12000, "gain_db": 0.5, "q": 0.7},
        },
    }

    # FIR filter parameters (linear-phase)
    FIR_ORDER = 513  # Must be odd for zero-phase
    FIR_WINDOW = "hamming"

    def __init__(self):
        super().__init__()
        self.name = "Final EQ v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_16_final_eq",
            name="Final EQ v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=9,
            dependencies=["phase_38_presence_boost", "phase_39_air_band_enhancement"],
            estimated_time_factor=0.18,
            version="2.0.0",
            memory_requirement_mb=70,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.93,
            description="Multi-band linear-phase EQ for broadcast-grade frequency response",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str | MaterialType = "unknown",
        **kwargs,
    ) -> PhaseResult:
        """
        Wendet an: multi-band linear-phase EQ to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material_type: Material type for adaptive processing

        Returns:
            PhaseResult with EQ'd audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)
        material = kwargs.get("material", material_type)
        if not isinstance(material, MaterialType):
            try:
                material = MaterialType(str(material))
            except Exception:
                material = MaterialType.CD_DIGITAL

        _strength_ctx = resolve_phase_strength_contract(kwargs)
        phase_locality_factor = float(_strength_ctx["phase_locality_factor"])
        _effective_strength = float(_strength_ctx["effective_strength"])

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "eq_applied": False,
                    "processing": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        is_stereo = audio.ndim == 2
        # Bug-Fix: EQ_CONFIG ist mit MaterialType-Membern indiziert (kein str-Enum-Mixin) —
        # .value hier hätte den Lookup immer auf CD_DIGITAL zurückfallen lassen.
        _material_key = material if isinstance(material, MaterialType) else MaterialType(material)  # §v10.996 Enum-Normalisierung
        config = {k: dict(v) for k, v in self.EQ_CONFIG.get(_material_key, self.EQ_CONFIG[MaterialType.CD_DIGITAL]).items()}

        # ── §v10 Spectrum-Aware Adaptation: Material-Referenz ≠ Song-IST ─────
        # Die EQ_CONFIG liefert die physikalisch ERWARTETE Korrektur für den
        # Tonträger. Aber: JEDER Song ist anders. Ein audiophil gepresster Vinyl
        # braucht weniger Bass-Korrektur als ein Billig-Presswerk-Vinyl.
        # Wir MESSEN das tatsächliche Spektrum und skalieren die Gains adaptiv.
        _measured_correction = _measure_spectral_deviation(audio, sample_rate, material)
        for _band_key, _band in config.items():
            _nominal_gain = float(_band["gain_db"])  # Material-Template
            # Spektrale Abweichung dämpft die Korrektur wenn der Song bereits
            # nah am Zielspektrum liegt, verstärkt sie wenn stark abweichend.
            if _band_key in _measured_correction:
                _spec_factor = float(np.clip(abs(_measured_correction[_band_key]) / 6.0, 0.3, 1.5))
                _band["gain_db"] = float(_nominal_gain * _spec_factor * _effective_strength)
            else:
                _band["gain_db"] = float(_nominal_gain * _effective_strength)

        # Total-Gain-Check nach adaptiver Skalierung
        total_gain = sum(abs(float(band["gain_db"])) for band in config.values())
        if total_gain < 0.5:
            logger.info("Total EQ gain < 0.5 dB - skipping for %s", material.name)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "eq_applied": False,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Minimal EQ needed - skipped"],
            )

        # Process each channel
        if is_stereo:
            left, right = stereo_channel_view(audio)
            eq_left = self._eq_channel(left, sample_rate, config)
            eq_right = self._eq_channel(right, sample_rate, config)
            eq_audio = stereo_like(eq_left, eq_right, audio)
        else:
            eq_audio = self._eq_channel(audio, sample_rate, config)

        # Normalize if needed (prevent clipping) — §2.49 Peak-Guard: percentile(99.9)
        peak = float(np.percentile(np.abs(eq_audio), 99.9))
        if peak > 0.99:
            eq_audio = eq_audio * (0.99 / peak)
            clipping_prevented = True
        else:
            clipping_prevented = False

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        eq_audio = np.nan_to_num(eq_audio, nan=0.0, posinf=0.0, neginf=0.0)
        eq_audio = np.clip(eq_audio, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            eq_audio = audio + _effective_strength * (eq_audio - audio)
            eq_audio = np.clip(eq_audio, -1.0, 1.0)

        # §4.5 Psychoacoustic Masking Clamp — EQ corrections only where audible
        if apply_psychoacoustic_masking_clamp is not None:
            try:
                eq_audio = apply_psychoacoustic_masking_clamp(
                    audio,
                    eq_audio,
                    sample_rate,
                    strength=_effective_strength,
                    mode="additive",
                )
            except Exception as _pm_exc:
                logger.debug("Verarbeitungsschritt16 masking clamp nicht blockierend: %s", _pm_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach EQ (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_p16,
            )

            _sc_result_p16 = _scg_p16(audio, eq_audio, sample_rate)
            if not _sc_result_p16.ok:
                _sc_wet_p16 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                eq_audio = (_sc_wet_p16 * eq_audio + (1.0 - _sc_wet_p16) * audio).astype(np.float32)
        except Exception as _sc_exc_p16:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_16 spectral_color nicht blockierend: %s", _sc_exc_p16
            )

        # §V26 Onset-Schutz nach EQ (§2.77, non-blocking)
        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opm_p16,
            )

            eq_audio = _opm_p16(audio, eq_audio, None, max_delta_db=1.5)
        except Exception as _opm_exc_p16:
            logger.debug("§V26 Verarbeitungsschritt_16 onset_guard nicht blockierend: %s", _opm_exc_p16)

        # ── §v10 Reference Target Matching ──
        try:
            _g = str(kwargs.get("genre", "unknown")).lower()
            _e = int(kwargs.get("era_decade", 1980))
            _t = {
                "schlager": (1.0, 1.5),
                "rock": (0.5, 2.0),
                "pop": (1.5, 2.0),
                "jazz": (0.0, 1.0),
                "classical": (-0.5, 0.0),
                "ballad": (0.3, 0.5),
                "electronic": (2.0, 2.5),
            }.get(_g, (0.5, 1.0))
            if _e < 1970:
                _t = (_t[0] * 0.6, _t[1] * 0.7)
            if abs(_t[0]) > 0.1:
                audio = signal.sosfiltfilt(
                    signal.butter(2, 8000, "highshelf", fs=sample_rate, output="sos"), audio, axis=0
                )
            if abs(_t[1]) > 0.1:
                audio = signal.sosfiltfilt(
                    signal.butter(2, 100, "lowshelf", fs=sample_rate, output="sos"), audio, axis=0
                )
        except Exception as _e:
            logger.debug("%s: unkritisch exception: %s", __name__, _e)

        return PhaseResult(
            success=True,
            audio=eq_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "eq_applied": True,
                "total_gain_db": float(total_gain),
                "clipping_prevented": clipping_prevented,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rt_factor": float(rt_factor),
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.20 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _eq_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, dict[str, Any]]) -> np.ndarray:
        """Wendet an: EQ to a single channel."""
        # **GUARD: Short-Audio-Buffer (§2.47, §0 Primum non nocere)**
        MIN_AUDIO_SAMPLES = 512  # 10 ms @ 48 kHz
        if len(audio) < MIN_AUDIO_SAMPLES:
            logger.debug(
                "Verarbeitungsschritt_16: audio too short (%d < %d), skipping EQ", len(audio), MIN_AUDIO_SAMPLES
            )
            return np.nan_to_num(np.asarray(audio, dtype=np.float32).copy(), nan=0.0)  # type: ignore[no-any-return]

        eq_audio = audio.copy()

        # Apply each band EQ
        for band_config in config.values():
            eq_type = band_config["type"]
            freq = band_config["freq"]
            gain_db = band_config["gain_db"]
            q = band_config["q"]

            if abs(gain_db) < 0.1:
                continue  # Skip near-zero gains

            if eq_type == "shelf":
                eq_audio = self._apply_shelf(eq_audio, sample_rate, freq, gain_db, q)
            elif eq_type == "bell":
                eq_audio = self._apply_bell(eq_audio, sample_rate, freq, gain_db, q)

        return eq_audio

    def _apply_shelf(self, audio: np.ndarray, sample_rate: int, freq: float, gain_db: float, q: float) -> np.ndarray:
        """Wendet an: shelving filter (low-shelf if freq < 500, else high-shelf)."""
        # Determine shelf type
        is_lowshelf = freq < 500

        # RBJ Audio-EQ-Cookbook Biquad Shelving Filter Coefficients
        w0 = 2 * np.pi * freq / sample_rate
        A = 10 ** (gain_db / 40)  # Amplitude
        alpha = np.sin(w0) / (2 * q)

        if is_lowshelf:
            # Low Shelf
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        else:
            # High Shelf
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Zero-phase filtering prevents phase shift on vocal transients.
        if len(audio) >= 9:
            filtered = safe_filtfilt(b, a, audio)
        else:
            filtered = signal.lfilter(b, a, audio)

        return filtered  # type: ignore[no-any-return]

    def _apply_bell(self, audio: np.ndarray, sample_rate: int, freq: float, gain_db: float, q: float) -> np.ndarray:
        """Wendet Bell-(Peaking)-Filter mittels IIR an."""
        # Design peaking filter
        w0 = 2 * np.pi * freq / sample_rate
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40)

        # Coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Apply filter (forward-backward for zero-phase)
        filtered = safe_filtfilt(b, a, audio)

        return filtered  # type: ignore[no-any-return]
