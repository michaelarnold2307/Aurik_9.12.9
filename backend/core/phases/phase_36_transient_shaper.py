#!/usr/bin/env python3
"""
Phase 36: Transient Shaper v2.1 - Professional
Multi-band transient enhancement and sustain control.

Algorithm Overview:
1. Multi-Band Split: 4 bands (Bass/Low-Mid/Mid-High/High @ 150/800/5k Hz)
2. Transient Detection:
   - Envelope follower (attack/release) — LOG-DOMAIN ballistics (v2.1)
   - Onset detection (spectral flux)
   - Peak detection (adaptive threshold)
3. Per-Band Shaping:
   - Attack enhancement: Boost transient peaks (0-20ms window)
   - Sustain control: Adjust decay/sustain portion
   - Independent attack/sustain ratios per band
4. Material Adaptation:
   - Shellac/Vinyl: Conservative (preserve vintage character)
   - Tape: Moderate (restore punch from tape compression)
   - Digital: Aggressive (add punch to quantized drums)
5. Safety Limiting: Prevent clipping from attack boost

Scientific Foundation:
- Zölzer (2011): DAFX - Digital Audio Effects §6.1 — log-domain ballistics
- Giannoulis et al. (2012): "Digital Dynamic Range Compressor Design — A Tutorial
  and Analysis", JAES 60(6), pp. 399–408 — log-domain attack/release detector
- Arfib et al. (2011): Time-Frequency Processing of Musical Signals
- Bello et al. (2005): A Tutorial on Onset Detection in Music Signals
- Dixon (2006): Onset Detection Revisited - Beat Tracking
- Massberg & Tan (2006): Asymmetric FIR Filters for Transient Shaping

Industry Benchmarks:
- SPL Transient Designer (Analog Classic)
- Native Instruments Transient Master (Digital Standard)
- iZotope Neutron Transient Shaper (AI-powered)
- Waves Trans-X (Professional)
- Sonnox Oxford TransMod (High-end)
- FabFilter Pro-MB (Multi-band transient control)

Quality Target: 0.75 → 0.90 (+20% improvement)
Performance Target: <0.20× realtime

Author: Aurik Development Team
Version: 2.1.0 Professional
"""

import logging
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import audio_sample_count, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class TransientShaper(PhaseInterface):
    """
    Professional Multi-Band Transient Shaper.

    Key Features:
    - 4-band processing for frequency-specific control
    - Attack enhancement (boost transients 0-20ms)
    - Sustain control (adjust decay/tail)
    - Material-adaptive parameters
    - Onset detection for precise timing
    - Safety limiting to prevent clipping

    Use Cases:
    - Enhance drum punch (kick, snare)
    - Restore transient detail lost in compression
    - Tighten bass (reduce sustain)
    - Brighten percussion (enhance high-frequency attacks)

    Performance: <0.20× realtime on modern CPU
    """

    # Crossover frequencies for 4-band split (Hz)
    CROSSOVER_FREQS = [400, 2000, 6400]  # §v10.101: Bark-Crossover

    # Shaping parameters (material-adaptive)
    SHAPING_CONFIG = {
        MaterialType.SHELLAC: {
            "attack_gain_db": [2.0, 1.5, 1.0, 0.5],  # Per band (Bass/Low-Mid/Mid-High/High)
            "sustain_gain_db": [0.0, -0.5, -0.5, 0.0],
            "attack_window_ms": 15,
            "release_window_ms": 100,
        },
        MaterialType.VINYL: {
            "attack_gain_db": [3.0, 2.5, 2.0, 1.5],
            "sustain_gain_db": [-1.0, -1.5, -1.0, 0.0],
            "attack_window_ms": 12,
            "release_window_ms": 80,
        },
        MaterialType.TAPE: {
            "attack_gain_db": [4.0, 3.5, 3.0, 2.0],
            "sustain_gain_db": [-2.0, -2.5, -2.0, -0.5],
            "attack_window_ms": 10,
            "release_window_ms": 70,
        },
        MaterialType.CASSETTE: {  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
            "attack_gain_db": [4.0, 3.5, 3.0, 2.0],
            "sustain_gain_db": [-2.0, -2.5, -2.0, -0.5],
            "attack_window_ms": 10,
            "release_window_ms": 70,
        },
        MaterialType.CD_DIGITAL: {
            "attack_gain_db": [5.0, 4.5, 4.0, 3.0],  # Aggressive (restore punch)
            "sustain_gain_db": [-3.0, -3.5, -3.0, -1.0],
            "attack_window_ms": 8,
            "release_window_ms": 60,
        },
        MaterialType.STREAMING: {
            "attack_gain_db": [4.5, 4.0, 3.5, 2.5],
            "sustain_gain_db": [-2.5, -3.0, -2.5, -1.0],
            "attack_window_ms": 10,
            "release_window_ms": 65,
        },
    }

    # Transient detection threshold (relative to RMS)
    ONSET_THRESHOLD_DB = 6.0

    def __init__(self):
        super().__init__()
        self.name = "Transient Shaper v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_36_transient_shaper",
            name="Transient Shaper v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            dependencies=["phase_08_transient_preservation"],
            estimated_time_factor=0.20,
            version="2.1.0",
            memory_requirement_mb=70,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description="Multi-band transient enhancement and sustain control",
        )

    # pylint: disable-next=arguments-renamed
    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int,
        material: MaterialType | None = None,
        **kwargs: Any,
    ) -> PhaseResult:
        # §v10.0.0 Fragile-Guard: Kein Transient Shaping auf bandbreiten-begrenztem Material.
        # bw_loss > 0.8 → keine nutzbaren Transienten im HF-Bereich → skip.
        _bw = float(kwargs.get("bandwidth_loss", 0.0))
        if _bw > 0.80:
            logger.info("Verarbeitungsschritt 36: uebersprungen — bandwidth_loss=%.2f > 0.80 (fragile material)", _bw)
            return PhaseResult(success=True, audio=audio, metadata={"skipped": "fragile"})
        # §v10.706 B16: SMP has_soft_saturation → Tape-Sättigung weicht Transienten auf
        _mat_val = str(getattr(material, "value", material)).lower() if material else "unknown"
        try:
            from backend.core.source_medium_profile import get_medium_profile

            if getattr(get_medium_profile(_mat_val), "has_soft_saturation", False):
                kwargs = dict(kwargs)
                kwargs["strength"] = float(kwargs.get("strength", 1.0)) * 0.80
        except Exception:
            logger.debug(
                "Verarbeitungsschritt_36: SourceMediumProfile nicht verfügbar — has_soft_saturation-Pruefung übersprungen"
            )
        # §Fix: material=None must not shadow _process_impl's own sensible
        # default (MaterialType.CD_DIGITAL) — forwarding None unconditionally
        # crashed later at `material.name` when callers omit `material`.
        _material = material if material is not None else MaterialType.CD_DIGITAL
        return self._process_impl(audio, sample_rate, _material, **kwargs)

    def _process_impl(  # type: ignore[override]
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Wendet an: transient shaping to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with shaped audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength < 0.01:  # §v10.0.5: < 1% = keine sinnvolle Wirkung
            logger.info(
                "Verarbeitungsschritt 36: uebersprungen — effective_strength=%.3f (no transient shaping angewendet)",
                _effective_strength,
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            # §5/5: Echte Peak-Messung auch bei Skip
            _p_peak = float(20.0 * np.log10(np.percentile(np.abs(audio), 99.9) + 1e-10))
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[],
            )

        # §2.46g soft_saturation-Guard: Transient-Shaping auf gesättigtem Material
        # verstärkt die Hörbarkeit von Sättigungsverzerrungen auf Transienten.
        # Hard-Cap bei preserve=True: 50 %.
        _p36_soft_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        _p36_soft_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        if _p36_soft_sat_preserve or _p36_soft_sat_sev > 0.35:
            _p36_sat_scale = 1.0
            if _p36_soft_sat_sev > 0.35:
                _p36_sat_scale = float(np.clip(1.0 - (_p36_soft_sat_sev - 0.35) * 0.8, 0.30, 1.0))
            if _p36_soft_sat_preserve and _p36_sat_scale > 0.50:
                _p36_sat_scale = 0.50
            _effective_strength = float(_effective_strength * _p36_sat_scale)
            logger.debug(
                "Verarbeitungsschritt 36 soft_saturation guard: severity=%.2f preserve=%s → scale=%.2f (strength=%.3f)",
                _p36_soft_sat_sev,
                _p36_soft_sat_preserve,
                _p36_sat_scale,
                _effective_strength,
            )

        is_stereo = audio.ndim == 2
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        config_raw = self.SHAPING_CONFIG.get(_mk, self.SHAPING_CONFIG[MaterialType.CD_DIGITAL])  # type: ignore[call-overload]
        config = {
            "attack_gain_db": [  # type: ignore[attr-defined]
                float(v * _effective_strength)
                for v in config_raw["attack_gain_db"]  # type: ignore[attr-defined]
            ],
            "sustain_gain_db": [  # type: ignore[attr-defined]
                float(v * _effective_strength)
                for v in config_raw["sustain_gain_db"]  # type: ignore[attr-defined]
            ],
            "attack_window_ms": config_raw["attack_window_ms"],
            "release_window_ms": config_raw["release_window_ms"],
        }

        # Measure initial transient energy
        transient_energy_before = self._measure_transient_energy(audio, sample_rate)

        # Process each channel
        if is_stereo:
            # §2.51 Linked Stereo: derive transient detection from mono sidechain (√(L²+R²)/√2)
            # so that both channels receive the identical gain curve — prevents stereo-field divergence.
            _ch0, _ch1 = stereo_channel_view(audio)
            mono_sc = np.sqrt(_ch0**2 + _ch1**2) * (1.0 / np.sqrt(2))
            shaped_left = self._shape_channel(_ch0, sample_rate, config, sidechain=mono_sc)
            shaped_right = self._shape_channel(_ch1, sample_rate, config, sidechain=mono_sc)
            shaped_audio = stereo_like(shaped_left, shaped_right, audio)
        else:
            shaped_audio = self._shape_channel(audio, sample_rate, config)

        if 0.0 < _effective_strength < 1.0:
            shaped_audio = audio + _effective_strength * (shaped_audio - audio)

        # Measure final transient energy
        transient_energy_after = self._measure_transient_energy(shaped_audio, sample_rate)
        transient_boost_db = 20 * np.log10((transient_energy_after + 1e-10) / (transient_energy_before + 1e-10))

        # Safety limiting — §2.49 Peak-Guard: percentile(99.9) so single impulse artefacts don't block normalisation
        peak = float(np.percentile(np.abs(shaped_audio), 99.9))
        if peak > 0.99:
            shaped_audio = shaped_audio * (0.99 / peak)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        shaped_audio = np.nan_to_num(shaped_audio, nan=0.0, posinf=0.0, neginf=0.0)
        shaped_audio = np.clip(shaped_audio, -1.0, 1.0)

        # §2.46e HallucinationGuard: Additive Phase darf kein halluziniertes Material einführen (VERBOTEN)
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _chk_hg36,
            )

            _hg_mode_36 = str(kwargs.get("mode", "restoration"))
            _pre36_mono = (
                audio.mean(axis=0)
                if (audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2)
                else (audio.mean(axis=1) if audio.ndim == 2 else audio)
            )
            _post36_mono = (
                shaped_audio.mean(axis=0)
                if (shaped_audio.ndim == 2 and shaped_audio.shape[0] == 2 and shaped_audio.shape[1] > 2)
                else (shaped_audio.mean(axis=1) if shaped_audio.ndim == 2 else shaped_audio)
            )
            _hg36 = _chk_hg36(
                _pre36_mono.astype(np.float32),
                _post36_mono.astype(np.float32),
                sr=sample_rate,
                mode=_hg_mode_36,
                bw_extension_context=True,
            )
            if _hg36.requires_rollback:
                shaped_audio = audio.copy()
                logger.warning("§2.46e Verarbeitungsschritt_36 HallucinationGuard: rollback (spectral_novelty > 0.15)")
        except Exception as _hg36_exc:
            logger.debug("§2.46e Verarbeitungsschritt_36 HallucinationGuard (nicht blockierend): %s", _hg36_exc)

        return PhaseResult(
            success=True,
            audio=shaped_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "transient_boost_db": float(transient_boost_db),
                "peak_before": float(np.percentile(np.abs(audio), 99.9)),  # V08: percentile not np.max
                "peak_after": float(np.percentile(np.abs(shaped_audio), 99.9)),  # V08: percentile not np.max
                "rt_factor": float(rt_factor),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.25 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _shape_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: dict[str, Any],
        sidechain: np.ndarray | None = None,
    ) -> np.ndarray:
        """Shape transients in a single audio channel.

        Args:
            sidechain: Optional mono sidechain for envelope/transient detection.
                When provided (§2.51 Linked Stereo), gain curves are derived from
                the sidechain rather than from ``audio`` itself.  The sidechain
                must have the same length as ``audio``.
        """
        # **GUARD: Short-Audio-Buffer (§2.47, §0 Primum non nocere)**
        MIN_AUDIO_SAMPLES = 512  # 10 ms @ 48 kHz
        if len(audio) < MIN_AUDIO_SAMPLES:
            logger.debug(
                "Verarbeitungsschritt_36: audio too short (%d < %d), passthrough", len(audio), MIN_AUDIO_SAMPLES
            )
            return np.nan_to_num(np.asarray(audio, dtype=np.float32).copy(), nan=0.0)  # type: ignore[no-any-return]

        # Split into bands
        bands = self._split_into_bands(audio, sample_rate)
        sc_bands: list[np.ndarray | None]
        if sidechain is not None:
            sc_bands = self._split_into_bands(sidechain, sample_rate)
        else:
            sc_bands = [None] * len(bands)

        # Shape each band
        shaped_bands = []
        for i, (band, sc_band) in enumerate(zip(bands, sc_bands)):
            attack_gain = config["attack_gain_db"][i]
            sustain_gain = config["sustain_gain_db"][i]

            shaped_band = self._shape_band(
                band,
                sample_rate,
                attack_gain,
                sustain_gain,
                config["attack_window_ms"],
                config["release_window_ms"],
                sidechain_band=sc_band,
            )
            shaped_bands.append(shaped_band)

        # Combine bands
        shaped_audio = self._combine_bands(shaped_bands)

        return shaped_audio[: len(audio)]

    def _split_into_bands(self, audio: np.ndarray, sample_rate: int) -> list:
        """Split audio into 4 frequency bands."""
        bands = []

        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) statt sosfilt (kausal, Pegelexplosion).
        # Band 1: Bass (0 - 150 Hz)
        sos_low = signal.butter(4, self.CROSSOVER_FREQS[0], btype="low", fs=sample_rate, output="sos")
        bands.append(signal.sosfiltfilt(sos_low, audio))

        # Band 2: Low-Mid (150 - 800 Hz)
        sos_mid1 = signal.butter(
            4, [self.CROSSOVER_FREQS[0], self.CROSSOVER_FREQS[1]], btype="band", fs=sample_rate, output="sos"
        )
        bands.append(signal.sosfiltfilt(sos_mid1, audio))

        # Band 3: Mid-High (800 - 5000 Hz)
        sos_mid2 = signal.butter(
            4, [self.CROSSOVER_FREQS[1], self.CROSSOVER_FREQS[2]], btype="band", fs=sample_rate, output="sos"
        )
        bands.append(signal.sosfiltfilt(sos_mid2, audio))

        # Band 4: High (5000+ Hz)
        sos_high = signal.butter(4, self.CROSSOVER_FREQS[2], btype="high", fs=sample_rate, output="sos")
        bands.append(signal.sosfiltfilt(sos_high, audio))

        return bands

    def _shape_band(
        self,
        band: np.ndarray,
        sample_rate: int,
        attack_gain_db: float,
        sustain_gain_db: float,
        attack_window_ms: float,
        release_window_ms: float,
        sidechain_band: np.ndarray | None = None,
    ) -> np.ndarray:
        """Shape transients in a single frequency band.

        Args:
            sidechain_band: When provided (§2.51 Linked Stereo), envelope detection
                and transient masking are derived from this signal rather than from
                ``band`` itself.  The resulting gain curve is then applied to ``band``.
        """
        # Compute envelope (fast attack, slow release for transient detection)
        attack_samples = int(attack_window_ms * sample_rate / 1000)
        release_samples = int(release_window_ms * sample_rate / 1000)

        # §2.51: use sidechain signal for detection if provided (Linked Stereo)
        detection_signal = sidechain_band if sidechain_band is not None else band
        envelope = self._compute_envelope(detection_signal, attack_samples, release_samples)

        # Detect transients (steep rises in envelope)
        transient_mask = self._detect_transients(envelope, attack_samples)

        # Create gain curve
        gain_db = np.where(transient_mask, attack_gain_db, sustain_gain_db)

        # Smooth gain transitions
        # **GUARD**: window_length must be ≤ len(gain_db) and odd
        min_window = 5  # Minimum for polyorder=3
        max_window = min(51, len(gain_db) // 10 * 2 + 1)
        window_length = max(min_window, min(max_window, len(gain_db)))
        if window_length % 2 == 0:
            window_length -= 1  # Ensure odd
        window_length = max(3, min(window_length, len(gain_db)))  # Final safeguard

        if window_length >= 5 and len(gain_db) >= window_length:
            gain_db_smooth = signal.savgol_filter(gain_db, window_length=window_length, polyorder=3)
        else:
            gain_db_smooth = gain_db  # Passthrough if too short

        # Apply gain
        gain_linear = 10 ** (gain_db_smooth / 20)
        shaped_band = band * gain_linear

        return shaped_band  # type: ignore[no-any-return]

    def _compute_envelope(self, audio: np.ndarray, attack_samples: int, release_samples: int) -> np.ndarray:
        """Compute envelope with asymmetric attack/release in the *log domain*.

        Processing in dBFS ensures perceptually uniform ballistics: a transient
        of equal loudness difference (dB) relative to its local floor is captured
        consistently regardless of absolute signal level (Weber–Fechner law).  In
        contrast, a linear-domain follower over-responds to loud transients and
        misses quiet ones by the same dB amount.

        Algorithm (Giannoulis et al. 2012, JAES 60(6); Zölzer 2011 DAFX §6.1):
            1. Convert |x| to dBFS:  x_db = 20·log₁₀(max(|x|, ε))
            2. Asymmetric smoothing in dBFS:
                   e_db[n] = α_a · x_db[n] + (1−α_a) · e_db[n−1]  if x_db > e_db[n−1]
                           = α_r · x_db[n] + (1−α_r) · e_db[n−1]  otherwise
            3. Convert back to linear:  e[n] = 10^(e_db[n] / 20)

        Optimized: Downsample by factor 16 for the sequential loop,
        then upsample back.  Preserves asymmetric attack/release behavior
        while reducing 10.8 M iterations to ~675 K.

        Args:
            audio:           1-D float64 signal (a single sub-band channel)
            attack_samples:  Time constant in samples for fast detector
            release_samples: Time constant in samples for slow release

        Returns:
            Envelope as non-negative linear-domain float64 array, same length as
            ``audio``.
        """
        _DS = 16  # Downsample factor — 0.33 ms at 48 kHz, well below attack window
        _FLOOR_DB = -120.0  # dBFS floor: prevents log(0) and pins silence to a finite value

        abs_audio = np.abs(audio)

        # Downsample via block-max (preserves transient peaks)
        n = len(abs_audio)
        n_trim = (n // _DS) * _DS
        abs_ds = abs_audio[:n_trim].reshape(-1, _DS).max(axis=1)
        # Handle remainder
        if n_trim < n:
            abs_ds = np.append(abs_ds, abs_audio[n_trim:].max())

        # --- Convert to dBFS for perceptually uniform ballistics ---
        floor_lin = 10.0 ** (_FLOOR_DB / 20.0)  # ≈ 1e-6
        abs_ds_db = 20.0 * np.log10(np.maximum(abs_ds, floor_lin))

        # Scale time constants for the downsampled rate
        att_ds = max(1, attack_samples // _DS)
        rel_ds = max(1, release_samples // _DS)
        attack_coeff = 1.0 - np.exp(-1.0 / att_ds)
        release_coeff = 1.0 - np.exp(-1.0 / rel_ds)

        # Vectorized asymmetric envelope follower (replaces slow Python for-loop).
        # Strategy: Attack pass = causal IIR with attack_coeff; Release pass = causal IIR
        # with release_coeff.  Take element-wise max of both so that fast rises use
        # attack_coeff while slow decays use release_coeff — asymmetric behavior preserved.
        b_att = np.array([attack_coeff])
        a_att = np.array([1.0, -(1.0 - attack_coeff)])
        b_rel = np.array([release_coeff])
        a_rel = np.array([1.0, -(1.0 - release_coeff)])
        # Initialize at first-sample steady state to prevent IC=0 transient artifact.
        # Without zi, the slow release filter starts at 0 dBFS and takes hundreds of
        # samples to converge, dominating max-envelope and corrupting quiet-transient
        # detection (Giannoulis 2012, §2.4).
        _zi_att = signal.lfilter_zi(b_att, a_att) * abs_ds_db[0]
        _zi_rel = signal.lfilter_zi(b_rel, a_rel) * abs_ds_db[0]
        att_pass, _ = signal.lfilter(b_att, a_att, abs_ds_db, zi=_zi_att)
        rel_pass, _ = signal.lfilter(b_rel, a_rel, abs_ds_db, zi=_zi_rel)
        envelope_db = np.maximum(att_pass, rel_pass)

        # Convert back to linear domain for downstream compatibility
        envelope_ds = 10.0 ** (envelope_db / 20.0)

        # Upsample back via linear interpolation
        x_ds = np.linspace(0, n - 1, len(envelope_ds))
        x_full = np.arange(n)
        envelope = np.interp(x_full, x_ds, envelope_ds)

        return envelope  # type: ignore[no-any-return]

    def _detect_transients(self, envelope: np.ndarray, attack_samples: int) -> np.ndarray:
        """Erkennt transients based on envelope slope."""
        # Compute derivative (rate of change)
        slope = np.diff(envelope, prepend=envelope[0])

        # Threshold based on local statistics
        window_size = attack_samples * 4
        # **GUARD**: window_length must be ≤ len(slope) and odd
        max_window = min(window_size * 2 + 1, len(slope) // 5 * 2 + 1)
        window_length = max(3, min(max_window, len(slope)))
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(3, min(window_length, len(slope)))

        if window_length >= 3 and len(slope) >= window_length:
            local_mean = signal.savgol_filter(slope, window_length=window_length, polyorder=1)
        else:
            local_mean = slope  # Fallback if too short

        # Guard: savgol_filter auf quadrierten Werten kann durch Float-Rundung minimal
        # negative Ergebnisse liefern => sqrt(negativ) = NaN => RuntimeWarning; clamp >= 0
        if window_length >= 3 and len(slope) >= window_length:
            local_std = np.sqrt(
                np.maximum(
                    signal.savgol_filter(
                        (slope - local_mean) ** 2,
                        window_length=window_length,
                        polyorder=1,
                    ),
                    0.0,
                )
            )
        else:
            local_std = np.sqrt(np.maximum((slope - local_mean) ** 2, 0.0))

        # Transient: slope > mean + 2*std
        transient_mask = slope > (local_mean + 2 * local_std)

        # Extend transient mask forward (attack window) — vectorized via convolution
        if attack_samples > 1:
            kernel = np.ones(attack_samples)
            extended_mask = (
                np.convolve(transient_mask.astype(np.float32), kernel, mode="full")[: len(transient_mask)] > 0
            )
        else:
            extended_mask = transient_mask

        return extended_mask  # type: ignore[no-any-return]

    def _combine_bands(self, bands: list) -> np.ndarray:
        """Kombiniert frequency bands."""
        combined = sum(bands)
        return combined  # type: ignore[no-any-return]

    def _measure_transient_energy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Misst transient energy (high-frequency content in first 20ms)."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel

        # High-pass filter (removes bass, focuses on transients)
        sos = signal.butter(4, 2000, btype="high", fs=sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # Compute envelope
        audio_hp_1d = np.asarray(audio_hp, dtype=np.float64).reshape(-1)
        n = audio_hp_1d.shape[0]
        spectrum = np.fft.fft(audio_hp_1d)
        h = np.zeros(n, dtype=np.float64)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1 : n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (n + 1) // 2] = 2.0
        analytic = np.fft.ifft(spectrum * h)
        envelope = np.abs(analytic)

        # Measure peak envelope energy
        transient_energy: float = float(np.max(envelope))

        return float(transient_energy)
