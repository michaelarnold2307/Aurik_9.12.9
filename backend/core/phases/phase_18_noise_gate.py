#!/usr/bin/env python3
"""
Phase 18: Noise Gate v2.0 - Professional
Multi-band frequency-dependent noise gate with adaptive side-chain processing.

Algorithm Overview:
1. Multi-Band Architecture:
   - Split signal into 4 frequency bands
   - Independent gate control per band
   - Frequency-dependent thresholds and attack/release
2. Advanced Envelope Detection:
   - RMS envelope with adaptive window
   - Peak detection for transient preservation
   - Side-chain filtering for accurate triggering
3. Soft-Knee Gate Curves:
   - Gradual transition (3-12 dB knee)
   - Natural-sounding attenuation
   - Prevents abrupt on/off artifacts
4. Material-Adaptive Parameters:
   - Shellac: Gentle gating (preserve character)
   - Vinyl: Moderate (reduce surface noise in quiet passages)
   - Tape: Aggressive (clean noise floor)
   - Digital: Ultra-precise (minimal artifacts)
5. Look-Ahead Processing:
   - 5-10ms look-ahead prevents transient clipping
   - Predictive gate opening for natural attacks

Scientific Foundation:
- McNally (1984): Dynamic Range Control of Digital Audio Signals
- Giannoulis et al. (2012): Digital Dynamic Range Compressor Design Tutorial
- Reiss & McPherson (2015): Audio Effects: Theory, Implementation and Application
- Zölzer (2011): DAFX - Digital Audio Effects
- AES Paper 3466: Dynamics Processing for High-Quality Audio

Industry Benchmarks:
- Waves NS1 (Intelligent noise suppressor, $79)
- FabFilter Pro-G (Multi-band gate, $129)
- Sonnox SuprEsser (Dynamic EQ/Gate, $249)
- iZotope RX Spectral Gate ($399)
- Cedar DNS (Broadcast noise suppressor, $2000+)

Quality Target: 0.75 → 0.90 (+20% improvement)
Performance Target: <0.15× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import os
import time
from typing import Any

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from backend.core.audio_utils import (
    apply_musical_gain_envelope,
    audio_sample_count,
    compute_signal_relative_gate_dbfs,
    safe_to_mono,
    stereo_channel_view,
    stereo_like,
)
from backend.core.defect_scanner import MaterialType
from backend.core.quality_mode import QualityModeConfig, is_phase_ml_enabled, log_mode_decision

try:
    from backend.core.dsp.psychoacoustics import (
        apply_psychoacoustic_masking_clamp,
        compute_masking_threshold_iso11172,
    )
except ImportError:  # pragma: no cover
    apply_psychoacoustic_masking_clamp = None  # type: ignore[assignment]
    compute_masking_threshold_iso11172 = None  # type: ignore[assignment]

try:
    from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_phoneme_mask_18
except ImportError:  # pragma: no cover
    _get_phoneme_mask_18 = None  # type: ignore[assignment]

try:
    from backend.core.ml_memory_budget import release as _release_ml_budget_18
    from backend.core.ml_memory_budget import try_allocate as _try_allocate_ml_budget_18
except ImportError:  # pragma: no cover
    _release_ml_budget_18 = None  # type: ignore[assignment]
    _try_allocate_ml_budget_18 = None  # type: ignore[assignment]

try:
    from backend.core.natural_performance_detector import (
        get_natural_performance_detector as _get_natural_performance_detector_18,
    )
except ImportError:  # pragma: no cover
    _get_natural_performance_detector_18 = None  # type: ignore[assignment]

try:
    from plugins.silero_plugin import get_silero_plugin as _get_silero_plugin_18
except ImportError:  # pragma: no cover
    _get_silero_plugin_18 = None  # type: ignore[assignment]

from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


def _compute_band_masking_gain_floors(audio: np.ndarray, sample_rate: int) -> list[float] | None:
    """Derive one psychoacoustic gain floor per gate band for §2.62."""
    try:
        if compute_masking_threshold_iso11172 is None:
            raise RuntimeError("psychoacoustics unavailable")

        _mono = safe_to_mono(audio) if audio.ndim == 2 else audio
        _mask_ratio = compute_masking_threshold_iso11172(_mono, sample_rate, n_fft=2048, hop_length=512)
        _mask_floor = np.mean(_mask_ratio, axis=1).astype(np.float32)
        _freqs = np.linspace(0.0, sample_rate / 2.0, _mask_floor.shape[0], dtype=np.float32)
        _bands_hz = (
            (0.0, float(NoiseGate.CROSSOVER_FREQS[0])),
            (float(NoiseGate.CROSSOVER_FREQS[0]), float(NoiseGate.CROSSOVER_FREQS[1])),
            (float(NoiseGate.CROSSOVER_FREQS[1]), float(NoiseGate.CROSSOVER_FREQS[2])),
            (float(NoiseGate.CROSSOVER_FREQS[2]), float(sample_rate / 2.0)),
        )
        _floors: list[float] = []
        for _f_lo, _f_hi in _bands_hz:
            _mask = (_freqs >= _f_lo) & (_freqs <= _f_hi)
            if not np.any(_mask):
                _floors.append(0.10)
                continue
            _band_floor = float(np.clip(np.mean(_mask_floor[_mask]), 0.10, 1.0))
            _floors.append(_band_floor)
        return _floors
    except Exception as exc:
        logger.debug("§2.62 Verarbeitungsschritt_18 Masking-Guard nicht verfügbar (nicht blockierend): %s", exc)
        return None


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    _mono = safe_to_mono(sig).astype(np.float64) if sig.ndim == 2 else sig.astype(np.float64)
    _frame = 480  # 10 ms @ 48 kHz
    _active = [
        _mono[i : i + _frame]
        for i in range(0, len(_mono) - _frame, _frame)
        if 20.0 * np.log10(np.sqrt(np.mean(_mono[i : i + _frame] ** 2)) + 1e-10) > -50.0
    ]
    if not _active:
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean([np.mean(r**2) for r in _active])) + 1e-10))


class NoiseGate(PhaseInterface):
    """
    Professional Multi-Band Frequency-Dependent Noise Gate.

    Key Features:
    - 4-band independent gating (150/800/5000 Hz crossovers)
    - Frequency-dependent thresholds and timing
    - Soft-knee transitions (3-12 dB)
    - RMS + peak envelope detection
    - Look-ahead transient preservation
    - Material-adaptive parameters

    Use Cases:
    - Reduce noise floor in quiet passages
    - Clean up between musical phrases
    - Minimize background hiss/hum
    - Preserve dynamic range and transients

    Performance: <0.15× realtime on modern CPU
    """

    # Frequency band crossover points
    CROSSOVER_FREQS = [400, 2000, 6400]  # §v10.101: Bark-Crossover  # Hz

    # Material-adaptive gate configurations
    GATE_CONFIG = {
        MaterialType.SHELLAC: {
            "thresholds_db": [-42, -40, -37, -35],  # Gentler gate to preserve vocal micro-detail
            "reductions_db": [-8, -10, -12, -14],  # Avoid over-attenuation in short shellac vocals
            "attack_ms": [24, 18, 12, 10],
            "release_ms": [180, 150, 120, 100],
            "knee_db": 6,
            "look_ahead_ms": 8,
        },
        MaterialType.VINYL: {
            "thresholds_db": [-40, -38, -35, -33],
            "reductions_db": [-15, -18, -22, -25],
            "attack_ms": [15, 12, 8, 6],
            "release_ms": [120, 100, 80, 60],
            "knee_db": 9,
            "look_ahead_ms": 10,
        },
        MaterialType.TAPE: {
            "thresholds_db": [
                -38,
                -38,
                -35,
                -30,
            ],  # Band 1+2 angehoben (war -45/-43) — schützt Bass/Grundton bei Tape-Material
            "reductions_db": [
                -12,
                -15,
                -20,
                -18,
            ],  # Band 1+2 sanfter (war -20/-25) — verhindert Low-Freq-Energie­verlust
            "attack_ms": [10, 8, 8, 6],
            "release_ms": [120, 100, 100, 80],  # Band 1+2 langsamer (war 100/80) — pumping-frei
            "knee_db": 12,
            "look_ahead_ms": 10,
        },
        MaterialType.CASSETTE: {
            "thresholds_db": [-36, -36, -33, -28],  # v10.0.0: etwas höher als TAPE (Cassette-Hiss)
            "reductions_db": [-10, -13, -18, -16],  # v10.0.0: etwas sanfter — erhält Hiss-Textur
            "attack_ms": [10, 8, 8, 6],
            "release_ms": [130, 110, 105, 85],  # v10.0.0: minimal länger (pumping-Schutz)
            "knee_db": 12,
            "look_ahead_ms": 10,
        },  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE, Hiss-Profil angepasst
        MaterialType.CD_DIGITAL: {
            "thresholds_db": [-55, -53, -50, -48],
            "reductions_db": [-30, -35, -40, -50],
            "attack_ms": [5, 4, 3, 2],
            "release_ms": [80, 60, 50, 40],
            "knee_db": 12,
            "look_ahead_ms": 12,  # §v10.64: 5→12, ≥ 2× max(attack)=5ms
        },
        MaterialType.STREAMING: {
            "thresholds_db": [-60, -58, -55, -53],
            "reductions_db": [-35, -40, -50, -60],
            "attack_ms": [3, 2, 2, 1],
            "release_ms": [60, 50, 40, 30],
            "knee_db": 12,
            "look_ahead_ms": 8,  # §v10.64: 5→8, ≥ 2× max(attack)=3ms
        },
    }

    _MAX_RMS_DROP_DB = {
        "tape": 2.0,
        "reel_tape": 1.8,
        "cassette": 2.2,
        "vinyl": 1.5,
        "shellac": 1.2,
        "wax_cylinder": 1.0,
        "cd_digital": 1.2,
        "dat": 1.0,
        "mp3_low": 1.4,
        "mp3_high": 1.4,
        "aac": 1.4,
        "unknown": 1.5,
    }

    def __init__(self):
        super().__init__()
        self.name = "Noise Gate v2 Professional"
        self._silero_vad = None  # Lazy loading

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_18_noise_gate",
            name="Noise Gate v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=8,
            dependencies=["phase_03_denoise", "phase_28_surface_noise_profiling"],
            estimated_time_factor=0.15,
            version="2.0.0",
            memory_requirement_mb=80,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description="Multi-band frequency-dependent noise gate with adaptive side-chain",
        )

    def _get_silero_vad(self):
        """Lädt beim ersten Zugriff: Silero VAD plugin for ML-based voice activity detection."""
        if self._silero_vad is None:
            try:
                if _get_silero_plugin_18 is None:
                    raise RuntimeError("Silero plugin unavailable")

                self._silero_vad = _get_silero_plugin_18()
                logger.info("Silero VAD plugin geladen erfolgreich")
            except Exception as e:
                logger.warning("konnte nicht laden Silero VAD plugin: %s", e)
                self._silero_vad = False  # Mark as unavailable

        return self._silero_vad if self._silero_vad is not False else None

    def _detect_voice_activity(
        self,
        audio: np.ndarray,
        sample_rate: int,
        silero_plugin,
    ) -> np.ndarray:
        """Detect voice/music activity using SileroPlugin.get_speech_mask().

        Nutzt die korrekte SileroPlugin-API (§11.3 plugins/silero_plugin.py):
        get_speech_mask(audio, sr) → bool-Array [n_samples].
        Gibt ein float32-Array [0,1] pro Sample zurück (Gate-Steuerkurve).

        Returns:
            Probability array float32 ∈ [0,1] für jeden Sample (1 = aktiv).
        """
        try:
            # SileroPlugin.get_speech_mask() → bool-Maske [n_samples]
            bool_mask = silero_plugin.get_speech_mask(audio, sample_rate)
            # bool → float32 Wahrscheinlichkeitskurve (NaN/Inf-sicher)
            vad_probabilities = bool_mask.astype(np.float32)
            vad_probabilities = np.nan_to_num(vad_probabilities, nan=1.0, posinf=1.0, neginf=0.0)
            return np.asarray(np.clip(vad_probabilities, 0.0, 1.0), dtype=np.float32)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("Voice activity detection fehlgeschlagen (DSP-only Modus): %s", e)
            # Fallback: Gate komplett offen (kein Signalverlust)
            return np.ones(len(audio), dtype=np.float32)  # type: ignore[no-any-return]

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="18")
        check_ml_model_ready("SileroVAD", phase_name="18")
        """
        Wendet an: multi-band noise gate to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material_type: Material type for adaptive processing

        Returns:
            PhaseResult with gated audio
        """

        # ── §v10 PIM: Per-Band-Intensität lesen ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim_params = apply_pim_intensity(
                kwargs, "noise_gate", default_nr=0.3, default_de_ess=0.2, default_comp=1.0
            )
            if kwargs.get("pim_intensity_map") is not None and "noise_reduction_strength" in kwargs:
                kwargs["noise_reduction_strength"] = _pim_params["nr_strength"]
        except Exception as e:
            logger.warning("Verarbeitungsschritt_18_noise_gate.py::verarbeiten Ersatzpfad: %s", e)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)
        if isinstance(material_type, MaterialType):
            material = material_type
        else:
            try:
                material = MaterialType(str(material_type).upper())  # type: ignore[assignment]
            except ValueError:
                try:
                    material = MaterialType[str(material_type).upper()]  # type: ignore[assignment]
                except (KeyError, AttributeError):
                    material = MaterialType.UNKNOWN  # type: ignore[assignment]

        # §v10.303.16 Continuous-Noise-Guard: Noise-Gate auf Material mit
        # kontinuierlichem Rauschen (Tape/Cassette) erzeugt Pumpen und
        # Echoeffekte. Der Gate öffnet/schließt auf der Hiss-Grenze →
        # hörbare Atmung. Phase 29 (Tape Hiss) ist die korrekte Behandlung.
        _CONTINUOUS_NOISE_MATERIALS = {
            MaterialType.CASSETTE,
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
        }
        if material in _CONTINUOUS_NOISE_MATERIALS:
            logger.info(
                "§v10.303.16 Continuous-Noise-Guard: %s → Noise Gate uebersprungen "
                "(kontinuierliches Rauschen, Gate würde pumpen. Verarbeitungsschritt 29 übernimmt)",
                material.name,
            )
            return PhaseResult(
                success=True,
                audio=audio.copy() if isinstance(audio, np.ndarray) else np.asarray(audio, dtype=np.float32),
                execution_time_seconds=time.time() - start_time,
                metadata={"algorithm": "skipped_continuous_noise", "material": material.name},
            )

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V40 NMR-Feedback: NR-Stärke adaptiv anpassen (FeedbackChain-aware).
        try:
            from backend.core.dsp.nmr_feedback import (
                compute_nmr_score as _nmr_fn_18,
            )

            _nmr_result_18 = _nmr_fn_18(audio, sample_rate)
            if not _nmr_result_18.ok:
                logger.warning(
                    "Verarbeitungsschritt18 §V40 NMR: nmr_above_masking → §2.45 Minimal-Intervention prüfen",
                )
            _effective_strength = float(
                np.clip(
                    _effective_strength + _nmr_result_18.recommended_nr_strength_delta,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "Verarbeitungsschritt18 §V40 NMR: delta=%.3f → eff_str=%.3f",
                _nmr_result_18.recommended_nr_strength_delta,
                _effective_strength,
            )
        except Exception as _nmr_exc_18:
            logger.debug("Verarbeitungsschritt18 §V40 NMR nicht blockierend: %s", _nmr_exc_18)

        # §2.17 SectionStrengthEnvelope: Kontinuierliche per-Segment-Modulation.
        # Reduziert Noise-Gate-Intensität in leisen Strophen (wo das Gate
        # natürlichen Rauschflor erhalten soll), verstärkt in dichten Refrains
        # (wo Rauschen störender wirkt). Fließende Cosine-Crossfades zwischen
        # allen Sektionen — kein hörbares Pumpen.
        _envelope = kwargs.get("strength_envelope")
        if _envelope is not None and len(_envelope) > 0:
            try:
                from backend.core.dsp.section_strength_envelope import get_section_strength_at

                _n_total = audio.shape[-1] if audio.ndim > 1 else len(audio)
                _env_val = get_section_strength_at(_envelope, 0, _n_total)
                _effective_strength = float(np.clip(_effective_strength * _env_val, 0.0, 1.0))
                logger.debug(
                    "Verarbeitungsschritt18 §2.17 SectionStrengthEnvelope: env=%.3f → eff_str=%.3f",
                    _env_val,
                    _effective_strength,
                )
            except Exception as _env_exc_18:
                logger.debug("Verarbeitungsschritt18 §2.17 SectionStrengthEnvelope nicht blockierend: %s", _env_exc_18)

        is_stereo = audio.ndim == 2
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        config = dict(self.GATE_CONFIG.get(_mk, self.GATE_CONFIG[MaterialType.CD_DIGITAL]))
        config["reductions_db"] = [float(r * _effective_strength) for r in config["reductions_db"]]  # type: ignore[attr-defined]

        # §GEBOT-G55: Adaptive Gate-Thresholds via Noise-Floor-Analyse
        # Zwei Kassetten mit unterschiedlichem Rauschboden brauchen verschiedene Schwellen.
        try:
            from backend.core.adaptive_parameter_infrastructure import derive_noise_floor

            _nf18 = derive_noise_floor(audio, sample_rate)
            # Adaptiere Schwellen: +6dB über Noise-Floor, aber innerhalb [-55, -25] dB
            _base_threshold = float(np.clip(_nf18["noise_floor_db"] + 8.0, -55.0, -25.0))
            _n_bands = len(config["thresholds_db"])  # type: ignore[arg-type]
            config["thresholds_db"] = [_base_threshold + i * 3.0 for i in range(_n_bands)]
            logger.debug(
                "Verarbeitungsschritt 18 adaptive: noise_floor=%.1fdB → gate_thresholds=%s",
                _nf18["noise_floor_db"],
                [f"{t:.0f}" for t in config["thresholds_db"]],  # type: ignore[attr-defined]
            )
        except Exception as _e:
            logger.debug("%s: unkritisch exception: %s", __name__, _e)

        config["masking_gain_floors"] = _compute_band_masking_gain_floors(audio, sample_rate)

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "noise_reduction_db": 0.0,
                    "bands": len(config["thresholds_db"]),  # type: ignore[arg-type]
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "processing": "skipped_zero_strength",
                    "rt_factor": 0.0,
                },
            )

        # Process each channel
        if is_stereo:
            # §2.51 Linked-Stereo: gate decision based on max(L_rms, R_rms).
            # Both channels open/close together to preserve stereo coherence.
            # RMS-linked sidechain √(L²+R²)/√2 represents the combined signal level.
            left, right = stereo_channel_view(audio)
            _linked_sc = np.sqrt(left**2 + right**2) * (1.0 / np.sqrt(2))
            _gated_sc = self._gate_channel(_linked_sc, sample_rate, config)
            # Derive per-sample linked gain (ratio of gated to original sidechain level).
            _sc_mag = np.abs(_linked_sc)
            _linked_gain = np.where(_sc_mag > 1e-9, np.abs(_gated_sc) / np.maximum(_sc_mag, 1e-9), 1.0)
            _linked_gain = np.clip(_linked_gain, 0.0, 1.0)
            gated_audio = stereo_like(_linked_gain * left, _linked_gain * right, audio)
        else:
            gated_audio = self._gate_channel(audio, sample_rate, config)

        # §2.45a Pre-Makeup-Guard: Erkennt Gates die Musik stärker dämpfen als Stille
        # (d. h. Gate-Schwellen zu weit offen für Musik-Pegel).
        # Geprüft VOR Makeup: silence_ratio ≈ 1.0 = Gate berührt Stille nicht → Musik wird gegated.
        _orig_mono_pre = safe_to_mono(audio) if is_stereo else audio
        _gated_mono_pre = safe_to_mono(gated_audio) if is_stereo else gated_audio
        _music_silence_alert_pre, _music_silence_wet_pre = self._check_music_silence_inversion(
            _orig_mono_pre, _gated_mono_pre, sample_rate
        )
        if _music_silence_alert_pre:
            gated_audio = _music_silence_wet_pre * gated_audio + (1.0 - _music_silence_wet_pre) * audio
            gated_audio = np.clip(gated_audio, -1.0, 1.0)

        # Low-frequency energy guard: prevent excessive bass/fundamental removal
        # especially critical for tape/vinyl where gate can eat musical low end
        _orig_mono = safe_to_mono(audio) if is_stereo else audio
        _gated_mono = safe_to_mono(gated_audio) if is_stereo else gated_audio
        sos_lf = signal.butter(4, 800, btype="low", fs=sample_rate, output="sos")
        from backend.core.audio_utils import safe_sosfiltfilt as _safe_sosfiltfilt18

        _lf_orig = np.sqrt(np.mean(_safe_sosfiltfilt18(sos_lf, _orig_mono) ** 2) + 1e-20)
        _lf_gated = np.sqrt(np.mean(_safe_sosfiltfilt18(sos_lf, _gated_mono) ** 2) + 1e-20)
        _lf_loss_ratio = _lf_gated / _lf_orig
        if _lf_loss_ratio < 0.60:
            # >40% low-freq energy lost — blend with original to limit damage
            _blend = 0.60 / (_lf_loss_ratio + 1e-10)
            _blend = min(_blend, 3.0)  # safety cap
            _wet = max(0.30, 1.0 / _blend)  # how much gated signal to keep
            gated_audio = _wet * gated_audio + (1.0 - _wet) * audio
            logger.warning(
                "Verarbeitungsschritt 18 low-freq guard: LF Verhaeltnis %.2f < 0.60 — blended wet=%.2f to protect bass",
                _lf_loss_ratio,
                _wet,
            )

        # §2.45a / §2.53: Apply wet/dry blend BEFORE loudness preservation so that
        # rms_drop_db / makeup metadata accurately reflect the ACTUAL output level change
        # (not the 100% wet level used for makeup calibration at low PMGG strengths).
        gated_audio = np.nan_to_num(gated_audio, nan=0.0, posinf=0.0, neginf=0.0)
        gated_audio = np.clip(gated_audio, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            gated_audio = audio + _effective_strength * (gated_audio - audio)
            gated_audio = np.clip(gated_audio, -1.0, 1.0)

        # §2.36 Phonem-Schutz: Konsonanten-Burst-Frames (/p/,/t/,/k/) können vom
        # Noise-Gate als Stille fehlklassifiziert werden → Original-Signal für diese
        # Frames wiederherstellen (Gate öffnen). Non-blocking.
        try:
            if _get_phoneme_mask_18 is None:
                raise RuntimeError("lyrics-guided enhancement unavailable")

            _hop_18 = 512
            _mono_18 = (safe_to_mono(audio) if is_stereo else audio).astype(np.float32)
            _pmask_18 = _get_phoneme_mask_18(_mono_18, sample_rate, hop_length=_hop_18)
            if np.any(_pmask_18):
                _n18 = len(_mono_18)
                _smask_18 = np.zeros(_n18, dtype=bool)
                for _fi18, _fp18 in enumerate(_pmask_18):
                    if _fp18:
                        _fs18 = _fi18 * _hop_18
                        _fe18 = min(_n18, _fs18 + _hop_18)
                        _smask_18[_fs18:_fe18] = True
                if is_stereo and gated_audio.ndim == 2:
                    if gated_audio.shape[0] == 2 and gated_audio.shape[1] > 2:
                        gated_audio[:, _smask_18] = audio[:, _smask_18]
                    else:
                        gated_audio[_smask_18, :] = audio[_smask_18, :]
                else:
                    gated_audio[_smask_18] = audio[_smask_18]
                logger.debug(
                    "§2.36 Verarbeitungsschritt_18 Phonem-Schutz: %d/%d Frames restauriert",
                    int(np.sum(_pmask_18)),
                    len(_pmask_18),
                )
        except Exception as _pmask18_exc:
            logger.debug("§2.36 Verarbeitungsschritt_18 Phonem-Mask (nicht blockierend): %s", _pmask18_exc)

        gated_audio, loudness_stats = self._apply_material_loudness_preservation(audio, gated_audio, material)

        # §2.45a Post-Makeup-Guard: Stille-Anstieg nach Makeup prüfen.
        # Wenn Makeup-Gain die Stille ÜBER den Original-Rausch-Boden hebt und Musik
        # gleichzeitig deutlich unterdrückt wurde → sofortiger Rescue.
        _orig_mono_post = safe_to_mono(audio) if is_stereo else audio
        _gated_mono_post = safe_to_mono(gated_audio) if is_stereo else gated_audio
        _music_silence_alert_post, _music_silence_wet_post = self._check_music_silence_inversion(
            _orig_mono_post, _gated_mono_post, sample_rate
        )
        if _music_silence_alert_post and not _music_silence_alert_pre:
            # Nur eingreifen wenn Pre-Makeup-Guard NICHT bereits korrigiert hat
            # (Doppel-Rescue würde Dry-Signal dominieren)
            _rescued = _music_silence_wet_post * gated_audio + (1.0 - _music_silence_wet_post) * audio
            gated_audio = np.clip(_rescued, -1.0, 1.0)

        # §4.5 Psychoacoustic Masking Clamp — protect musically masked gate regions
        try:
            if apply_psychoacoustic_masking_clamp is None:
                raise RuntimeError("psychoacoustic masking clamp unavailable")

            gated_audio = apply_psychoacoustic_masking_clamp(
                audio,
                gated_audio,
                sample_rate,
                strength=_effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt18 masking clamp nicht blockierend: %s", _pm_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — restore breath zones gated as silence.
        # Atemgeräusche (−55 bis −40 dBFS, 50–500 ms) werden vom Gate als Stille klassifiziert
        # und weggeschnitten. NPA-Detektor schützt diese Zonen. Non-blocking.
        try:
            if _get_natural_performance_detector_18 is None:
                raise RuntimeError("natural performance detector unavailable")

            _npa_audio_18 = audio
            if _npa_audio_18.ndim == 2 and _npa_audio_18.shape[0] == 2 and _npa_audio_18.shape[1] > 2:
                _npa_audio_18 = _npa_audio_18.T  # channels-first → channels-last
            _npa_result_18 = _get_natural_performance_detector_18().detect(_npa_audio_18, sample_rate)
            _npa_n_18 = (
                gated_audio.shape[1]
                if (gated_audio.ndim == 2 and gated_audio.shape[0] == 2 and gated_audio.shape[1] > gated_audio.shape[0])
                else (gated_audio.shape[0] if gated_audio.ndim <= 2 else len(gated_audio))
            )
            _npa_mask_18 = _npa_result_18.get_protected_mask(_npa_n_18, sample_rate)
            if np.any(_npa_mask_18):
                if is_stereo and gated_audio.ndim == 2:
                    if gated_audio.shape[0] == 2 and gated_audio.shape[1] > 2:
                        gated_audio[:, _npa_mask_18] = (
                            audio[:, _npa_mask_18] if audio.ndim == 2 else gated_audio[:, _npa_mask_18]
                        )
                    else:
                        _a18_ref = audio if (audio.ndim == 2 and audio.shape == gated_audio.shape) else gated_audio
                        gated_audio[_npa_mask_18, :] = _a18_ref[_npa_mask_18, :]
                elif gated_audio.ndim == 1 and audio.ndim == 1:
                    gated_audio[_npa_mask_18] = audio[_npa_mask_18]
                logger.debug(
                    "§2.46f Verarbeitungsschritt_18 NPA: %d protected samples restauriert (Atemgeräusche/Vibrato)",
                    int(np.sum(_npa_mask_18)),
                )
        except Exception as _npa18_exc:
            logger.debug("§2.46f Verarbeitungsschritt_18 NPA-Guard (nicht blockierend): %s", _npa18_exc)

        # §0p/V19/V20/V21/V26/§2.72 Vokal- + Textur-Guards nach Noise-Gate (RELEASE_MUST §0p V19-V26)
        _p18_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        _mat18_guards = str(getattr(material, "name", str(material)) or "unknown").lower()
        if _p18_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import apply_hnr_blend as _apply_hnr_18

                _hnr_blended_18, _hnr_diag_18 = _apply_hnr_18(
                    audio.astype(np.float32), gated_audio.astype(np.float32), sample_rate
                )
                if _hnr_diag_18.get("over_cleaned"):
                    gated_audio = _hnr_blended_18
            except Exception as _hnr_18_exc:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_18 (nicht blockierend): %s", _hnr_18_exc)

        _nt18_residual = audio - gated_audio
        try:
            from backend.core.dsp.noise_texture_guard import (
                compute_noise_texture_distance as _nt18_dist_fn,
            )

            if _nt18_residual.shape == audio.shape:
                _nt18_d = _nt18_dist_fn(_nt18_residual, _mat18_guards, sr=sample_rate)
                if _nt18_d > 0.10:
                    # Progressiver Blend statt binär 50/50:
                    # _nt18_d 0.10→0% dry, 0.25→20% dry, 0.40→50% dry, 0.60→80% dry
                    _dry_ratio = float(np.clip((_nt18_d - 0.10) / 0.50, 0.0, 0.80))
                    _wet_ratio = 1.0 - _dry_ratio
                    gated_audio = (_wet_ratio * gated_audio + _dry_ratio * audio).astype(np.float32)
                    logger.info("§V19 (Spec-Vintage-Guard) phase_18: noise_texture_dist=%.3f", _nt18_d)
        except Exception as _nt18_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_18 noise_texture nicht blockierend: %s", _nt18_exc
            )

        if _p18_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (
                    frame_energy_correlation as _fec18,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr18 = _fec18(audio, gated_audio, sample_rate, frame_ms=10.0)
                if _corr18 < 0.97:
                    _need18 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet18 = _recommend_mkk_wet(_corr18, _p18_panns, global_need=_need18)
                    gated_audio = (_wet18 * gated_audio + (1.0 - _wet18) * audio).astype(np.float32)
                    logger.warning(
                        "§V20 Verarbeitungsschritt_18: mikrodynamik_corr=%.4f < 0.97 → wet=%.3f", _corr18, _wet18
                    )
            except Exception as _v20_18_exc:
                logger.debug("§V20 Verarbeitungsschritt_18 mikrodynamik nicht blockierend: %s", _v20_18_exc)

        if any(x in _mat18_guards for x in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (
                    apply_noise_floor_minimum as _nfmin18,
                )

                gated_audio = _nfmin18(gated_audio, sample_rate, _mat18_guards, original_audio=audio)
            except Exception as _v21_18_exc:
                logger.debug("§V21 Verarbeitungsschritt_18 noise_floor nicht blockierend: %s", _v21_18_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (
                check_spectral_color_preservation as _scg_18,
            )

            _sc_result_18 = _scg_18(audio, gated_audio, sample_rate)
            if not _sc_result_18.ok:
                _sc_wet_18 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                gated_audio = (_sc_wet_18 * gated_audio + (1.0 - _sc_wet_18) * audio).astype(np.float32)
        except Exception as _sc_exc_18:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_18 spectral_color nicht blockierend: %s", _sc_exc_18
            )

        try:
            from backend.core.dsp.onset_guard import (
                apply_onset_protection_mask as _opm18,
            )

            gated_audio = _opm18(audio, gated_audio, None, max_delta_db=1.5)
        except Exception as _v26_18_exc:
            logger.debug("§V26 Verarbeitungsschritt_18 onset_guard nicht blockierend: %s", _v26_18_exc)

        if _p18_panns >= 0.25:
            try:
                from backend.core.dsp.vibrato_guard import (
                    check_vibrato_depth_preservation as _vib18_fn,
                )

                _vibr18 = _vib18_fn(audio, gated_audio, sample_rate)
                if not _vibr18.ok:
                    gated_audio = (0.5 * gated_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§2.72 Verarbeitungsschritt_18: vibrato_reduction=%.1f%% → 50%% dry-blend",
                        _vibr18.depth_reduction_pct,
                    )
            except Exception as _vib18_exc:
                logger.debug("§2.72 Verarbeitungsschritt_18 vibrato nicht blockierend: %s", _vib18_exc)

        # Metrics — §2.45a-I: gated RMS, ignoriert Stille-Frames
        _rms_orig_db = _rms_dbfs_gated(audio)
        _rms_gated_db = _rms_dbfs_gated(gated_audio)
        noise_reduction_db = (_rms_gated_db - _rms_orig_db) if _rms_orig_db > -80.0 else 0.0

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)
        return PhaseResult(
            success=True,
            audio=gated_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "noise_reduction_db": float(noise_reduction_db),
                "bands": len(config["thresholds_db"]),  # type: ignore[arg-type]
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": loudness_stats["rms_drop_db"],
                "loudness_makeup_db": loudness_stats["makeup_gain_db"],
                "rt_factor": float(rt_factor),
            },
            warnings=[] if rt_factor < 0.18 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    # ------------------------------------------------------------------
    # §2.45a Music/Silence inversion guard
    # ------------------------------------------------------------------
    def _check_music_silence_inversion(
        self,
        orig_mono: np.ndarray,
        gated_mono: np.ndarray,
        sr: int,
    ) -> tuple[bool, float]:
        """Erkennt inverses Gate-/Makeup-Verhalten: Musik stärker verändert als Stille.

        Wird zweimal aufgerufen:
          A) PRE-Makeup  → orig/gated sind Pegel VOR Makeup; silence_ratio ≈ 1.0 wenn Gate Stille
             nicht berührt; Alert wenn Musik aber stark gedämpft (Gate-Schwelle-Fehler).
          B) POST-Makeup → Alert wenn Stille ÜBER Original (Makeup überkompensiert Stille-Abschnitte)
             und Musik gleichzeitig unter Original liegt.

        Algorithmus (frame-basiert, 50 ms Frames):
          1. Frames klassifizieren: music   (RMS orig ≥ P40-Perzentil)
                                    silence (RMS orig <  P20-Perzentil)
          2. ratio = rms_gated / rms_orig pro Frame
          3. Median music_ratio vs. silence_ratio vergleichen
          4. Inversion A (pre-makeup):  music_ratio < 0.65, silence_ratio > 0.88
               → Gate trifft Musik, lässt Stille unberührt
          4. Inversion B (post-makeup): music_ratio < 0.90, silence_ratio > 1.04
               → Makeup hebt Stille über Original, Musik bleibt unter Original

        Returns:
            (alert: bool, wet_factor: float [0..1])
        """
        frame_len = max(1, int(sr * 0.05))  # 50 ms frames
        n_frames = len(orig_mono) // frame_len
        if n_frames < 4:
            return False, 1.0

        orig_frames = orig_mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        gated_frames = gated_mono[: n_frames * frame_len].reshape(n_frames, frame_len)

        rms_orig = np.sqrt(np.mean(orig_frames**2, axis=1) + 1e-20)
        rms_gated = np.sqrt(np.mean(gated_frames**2, axis=1) + 1e-20)
        ratio = rms_gated / rms_orig  # 1.0 = unverändert; < 1.0 = gedämpft; > 1.0 = angehoben

        p20 = float(np.percentile(rms_orig, 20))
        p40 = float(np.percentile(rms_orig, 40))

        music_mask = rms_orig >= p40
        silence_mask = rms_orig < p20

        n_music = int(music_mask.sum())
        n_silence = int(silence_mask.sum())

        if n_music < 2 or n_silence < 2:
            return False, 1.0

        music_ratio = float(np.median(ratio[music_mask]))  # 1.0 = musik unverändert
        silence_ratio = float(np.median(ratio[silence_mask]))  # 1.0 = stille unverändert

        # Inversion A (typisch pre-makeup): Gate-Schwelle zu weit offen →
        # Musik wird relativ stärker gedämpft als Stille.
        MUSIC_DROP_A = 0.65  # Musik < −3.7 dB
        SILENCE_OK_A = 0.88  # Stille kaum berührt (< −1.1 dB)
        alert_a = music_ratio < MUSIC_DROP_A and silence_ratio > SILENCE_OK_A

        # Inversion B (typisch post-makeup): Makeup hebt Stille über Original
        # (global Gain auf ein Signal das hauptsächlich aus Stille-Frames besteht)
        MUSIC_DROP_B = 0.90  # Musik < −0.9 dB (netto unter Original)
        SILENCE_RISE_B = 1.04  # Stille > +0.3 dB über Original
        alert_b = music_ratio < MUSIC_DROP_B and silence_ratio > SILENCE_RISE_B

        alert = alert_a or alert_b

        if alert:
            mode = "pre-makeup" if alert_a else "post-makeup"
            wet = float(np.clip(music_ratio / max(MUSIC_DROP_A, MUSIC_DROP_B) * 0.55, 0.25, 0.55))
            logger.warning(
                "Verarbeitungsschritt 18 §2.45a MUSIC/SILENCE INVERSION (%s): "
                "music_Verhaeltnis=%.3f silence_Verhaeltnis=%.3f → Rescue wet=%.2f "
                "(n_music=%d, n_silence=%d)",
                mode,
                music_ratio,
                silence_ratio,
                wet,
                n_music,
                n_silence,
            )
            return True, wet

        if music_ratio < 0.80 or silence_ratio > 1.02:
            logger.info(
                "Verarbeitungsschritt 18 §2.45a level Pruefung: music_Verhaeltnis=%.3f silence_Verhaeltnis=%.3f — OK (no rescue)",
                music_ratio,
                silence_ratio,
            )
        return False, 1.0

    def _apply_material_loudness_preservation(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, dict[str, float]]:
        material_key = getattr(material, "name", str(material)).lower()
        max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(material_key, self._MAX_RMS_DROP_DB["unknown"]))

        # §2.45a-I Gated-RMS: ignoriert Stille-Frames < −50 dBFS
        _rms_in_db = _rms_dbfs_gated(np.asarray(original_audio, dtype=np.float32))
        _rms_out_db = _rms_dbfs_gated(np.asarray(processed_audio, dtype=np.float32))
        rms_in = float(10.0 ** (_rms_in_db / 20.0))
        rms_drop_db = (_rms_out_db - _rms_in_db) if _rms_in_db > -90.0 else 0.0
        makeup_gain_db = 0.0

        if rms_in > 1e-8 and rms_drop_db < -max_rms_drop_db:
            target_rms_drop_db = -max_rms_drop_db
            required_gain_db = target_rms_drop_db - rms_drop_db
            makeup_gain_db = float(np.clip(required_gain_db, 0.0, 6.0))
            if makeup_gain_db > 0.0:
                _gain_lin = float(10.0 ** (makeup_gain_db / 20.0))
                # §2.45a-II: signal-relative gate — CEDAR/iZotope RX approach (v10.0.0)
                # §v10.0.0 Genre-adaptive: Schlager/Pop brauchen weicheren Gate
                # (dichter Mix, keine echten Stille-Passagen). Klassik/Jazz
                # vertragen tieferen Gate (tatsächliche Pausen zwischen Phrasen).
                _genre_18 = ""
                _gate_dbfs_18 = compute_signal_relative_gate_dbfs(original_audio, material_key=material_key)
                if _genre_18 in ("schlager", "pop", "rock", "metal"):
                    _gate_dbfs_18 = float(_gate_dbfs_18) + 6.0  # 6 dB softer gate
                    logger.debug("Verarbeitungsschritt 18: genre=%s → gate softened by +6dB", _genre_18)
                elif _genre_18 in ("klassik", "oper", "jazz"):
                    _gate_dbfs_18 = float(_gate_dbfs_18) - 3.0  # 3 dB deeper gate
                    logger.debug("Verarbeitungsschritt 18: genre=%s → gate deepened by -3dB", _genre_18)
                processed_audio = apply_musical_gain_envelope(
                    processed_audio,
                    _gain_lin,
                    gate_dbfs=_gate_dbfs_18,
                    crossfade_ms=10.0,
                    sr=48000,
                    reference_for_gate=original_audio,
                )
                processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
                # §2.45a-III: soft-limiter only when peak99 > 0.98
                current_peak = float(np.percentile(np.abs(processed_audio), 99.9))
                if current_peak > 0.98:
                    _abs_18 = np.abs(processed_audio)
                    _over_18 = _abs_18 > 0.92
                    if np.any(_over_18):
                        processed_audio = np.where(
                            _over_18,
                            np.sign(processed_audio) * (0.92 + 0.08 * np.tanh((_abs_18 - 0.92) / 0.08)),
                            processed_audio,
                        )
                processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
                _rms_out_db = _rms_dbfs_gated(np.asarray(processed_audio, dtype=np.float32))
                rms_drop_db = (_rms_out_db - _rms_in_db) if _rms_in_db > -90.0 else 0.0
                logger.info(
                    "Verarbeitungsschritt 18 loudness-preservation: material=%s rms_drop=%.2f dB via makeup %.2f dB (envelope-gated)",
                    material_key,
                    rms_drop_db,
                    makeup_gain_db,
                )

        return processed_audio, {
            "rms_drop_db": round(float(rms_drop_db), 3),
            "makeup_gain_db": round(float(makeup_gain_db), 3),
        }

    def _gate_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Wendet an: multi-band gating to a single channel with optional ML VAD."""
        # Check if ML VAD should be used
        use_vad = is_phase_ml_enabled(18)
        vad_probabilities = None

        if use_vad:
            silero = self._get_silero_vad()
            if silero is not None:
                # §2.47 ml_memory_budget guard (80 MB for Silero VAD)
                _vad_budget_ok = False
                _vad_release = None
                try:
                    if _try_allocate_ml_budget_18 is None:
                        raise ImportError

                    if _try_allocate_ml_budget_18("SileroVAD_phase18", 0.08):
                        _vad_budget_ok = True
                        _vad_release = _release_ml_budget_18
                    else:
                        logger.debug("SileroVAD_Verarbeitungsschritt18: ml_memory_Grenze insufficient — DSP-Ersatzpfad")
                except ImportError:
                    _vad_budget_ok = True  # budget tracking unavailable — allow inference
                if _vad_budget_ok:
                    try:
                        log_mode_decision("phase_18", True, "Using Silero VAD for intelligent gating")
                        # Get voice activity probabilities (0-1 for each frame)
                        vad_probabilities = self._detect_voice_activity(audio, sample_rate, silero)
                    except Exception as e:
                        logger.warning("Silero VAD fehlgeschlagen: %s, using DSP only", e)
                    finally:
                        if _vad_release is not None:
                            _vad_release("SileroVAD_phase18")
            else:
                log_mode_decision("phase_18", False, "Silero VAD unavailable")
        else:
            log_mode_decision("phase_18", False, f"Mode: {QualityModeConfig.get_mode().value}")

        # Split into frequency bands
        bands = self._split_bands(audio, sample_rate)

        # Apply gate to each band
        gated_bands = []
        for i, band_audio in enumerate(bands):
            threshold_db = config["thresholds_db"][i]
            reduction_db = config["reductions_db"][i]
            attack_ms = config["attack_ms"][i]
            release_ms = config["release_ms"][i]
            knee_db = config["knee_db"]
            # §v10.65: LF-Raumton bewahren — Band 0 (<150 Hz) maximal halbe Reduktion.
            # Natürliche Räume haben ihre stärkste Raumtönung im Bass. Komplette
            # Stille unter 150 Hz klingt sofort "künstlich" — das Gehirn erwartet
            # kontinuierliche tieffrequente Rauminformation.
            if i == 0:
                reduction_db = reduction_db * 0.5
            masking_gain_floor = 0.10
            _masking_floors = config.get("masking_gain_floors")
            if isinstance(_masking_floors, list) and i < len(_masking_floors):
                masking_gain_floor = float(np.clip(_masking_floors[i], 0.10, 1.0))

            gated_band = self._apply_gate(
                band_audio,
                sample_rate,
                threshold_db,
                reduction_db,
                attack_ms,
                release_ms,
                knee_db,
                masking_gain_floor,
                vad_probabilities,  # Pass VAD info to gate
            )
            gated_bands.append(gated_band)

        # Recombine bands
        gated_audio = self._combine_bands(gated_bands)

        return gated_audio

    def _split_bands(self, audio: np.ndarray, sample_rate: int) -> list:
        """Split audio into frequency bands using Linkwitz-Riley filters."""
        bands = []

        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) statt sosfilt (kausal, Pegelexplosion).
        # Band 1: Low (< 150 Hz)
        sos_low = signal.butter(4, self.CROSSOVER_FREQS[0], btype="low", fs=sample_rate, output="sos")
        bands.append(signal.sosfiltfilt(sos_low, audio))

        # Band 2: Low-mid (150-800 Hz)
        sos_mid1 = signal.butter(
            4, [self.CROSSOVER_FREQS[0], self.CROSSOVER_FREQS[1]], btype="band", fs=sample_rate, output="sos"
        )
        bands.append(signal.sosfiltfilt(sos_mid1, audio))

        # Band 3: High-mid (800-5000 Hz)
        sos_mid2 = signal.butter(
            4, [self.CROSSOVER_FREQS[1], self.CROSSOVER_FREQS[2]], btype="band", fs=sample_rate, output="sos"
        )
        bands.append(signal.sosfiltfilt(sos_mid2, audio))

        # Band 4: High (> 5000 Hz)
        sos_high = signal.butter(4, self.CROSSOVER_FREQS[2], btype="high", fs=sample_rate, output="sos")
        bands.append(signal.sosfiltfilt(sos_high, audio))

        return bands

    def _combine_bands(self, bands: list) -> np.ndarray:
        """Kombiniert frequency bands back into full-bandwidth signal."""
        # Simple summation (Linkwitz-Riley filters sum to flat response)
        return np.asarray(sum(bands), dtype=np.float32)  # type: ignore[no-any-return]

    def _apply_gate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold_db: float,
        reduction_db: float,
        attack_ms: float,
        release_ms: float,
        knee_db: float,
        masking_gain_floor: float,
        vad_probabilities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Wendet an: gating to a single frequency band with optional VAD guidance.

        Fully vectorised — no per-sample Python loops.
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        # Compute RMS envelope — causal IIR to prevent predictive gate opening.
        # mode="same" in convolve is non-causal (uses window_samples//2 ≈ 480 future
        # samples), causing the gate to start opening before actual note onsets.
        # This introduces audible pre-echo (energy in the 5–10 ms pre-window before
        # attacks), triggering §2.49 rollbacks on every transient-rich segment.
        # lfilter is causal by definition: y[n] depends only on x[n..n-M+1]. §2.49
        window_samples = int(0.020 * sample_rate)  # 20ms RMS window
        _rms_b = np.ones(window_samples, dtype=np.float32) / window_samples
        rms_power = signal.lfilter(_rms_b, [1.0], audio.astype(np.float32) ** 2)
        rms_power = np.maximum(rms_power, 0.0)
        rms = np.sqrt(rms_power)
        rms_db = 20 * np.log10(rms + 1e-10)

        # Adjust threshold based on VAD (if available)
        if vad_probabilities is not None:
            # Resample VAD to match audio length
            if len(vad_probabilities) != len(audio):
                x_vad = np.linspace(0, 1, len(vad_probabilities))
                x_audio = np.linspace(0, 1, len(audio))
                f = interp1d(x_vad, vad_probabilities, kind="linear", fill_value="extrapolate")
                vad_probabilities = np.asarray(f(x_audio), dtype=np.float32)
                vad_probabilities = np.clip(vad_probabilities, 0.0, 1.0)

            # Adapt threshold: Lower when voice/music is present (keep gate open)
            threshold_db_adapted = threshold_db - 15 * vad_probabilities
        else:
            threshold_db_adapted = np.full_like(rms_db, threshold_db)

        # ---- Vectorised gain computation (soft knee) ----
        knee_half = knee_db / 2.0
        thresh_lo = threshold_db_adapted - knee_half
        thresh_hi = threshold_db_adapted + knee_half

        # Full reduction zone
        gain_db = np.full_like(rms_db, reduction_db)
        # Soft knee transition zone
        knee_mask = (rms_db >= thresh_lo) & (rms_db <= thresh_hi)
        ratio = np.where(knee_mask, (rms_db - thresh_lo) / max(knee_db, 1e-6), 0.0)
        gain_db = np.where(knee_mask, reduction_db * (1 - ratio), gain_db)
        # No reduction zone
        gain_db = np.where(rms_db > thresh_hi, 0.0, gain_db)

        # ---- Vectorised attack/release smoothing (IIR via scipy) ----
        attack_coeff = 1 - np.exp(-1 / (sample_rate * attack_ms / 1000))
        release_coeff = 1 - np.exp(-1 / (sample_rate * release_ms / 1000))

        # Two-pass smoothing: attack pass (fast decrease) then release pass (slow increase)
        # This avoids the per-sample Python loop entirely.
        # Approximate IIR envelope follower with two exponential filters:
        #   - Attack: fast-responding low-pass on gain_db
        #   - Release: slow-responding low-pass on the attack output
        # Use lfilter for exact IIR behaviour (still vectorised C-level).
        gain_db_smooth = np.empty_like(gain_db)
        gain_db_smooth[0] = gain_db[0]
        # Determine per-sample coefficient: attack when going down, release up
        going_down = np.diff(gain_db, prepend=gain_db[0]) < 0
        np.where(going_down, attack_coeff, release_coeff)
        # Apply IIR via lfilter: y[n] = coeff*x[n] + (1-coeff)*y[n-1]
        # Rewrite as: y[n] = coeff[n]*x[n] + (1-coeff[n])*y[n-1]
        # For uniform coeff we could use lfilter directly. For varying coeff
        # we use a fast C-level loop via numba or fallback to a tight loop.
        # Since attack/release coefficients are uniform, we approximate with
        # the dominant (slower) coefficient via lfilter, then clamp.
        dominant_coeff = min(attack_coeff, release_coeff)
        b = np.array([dominant_coeff])
        a = np.array([1.0, -(1.0 - dominant_coeff)])
        gain_db_smooth = signal.lfilter(b, a, gain_db).astype(np.float32)

        # Convert to linear gain and apply
        gain_linear = 10 ** (gain_db_smooth / 20)
        gain_linear = np.maximum(gain_linear, float(np.clip(masking_gain_floor, 0.10, 1.0)))
        gated = audio * gain_linear

        return np.asarray(gated, dtype=np.float32)  # type: ignore[no-any-return]
