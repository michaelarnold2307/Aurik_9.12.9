#!/usr/bin/env python3
"""
Phase 42: Vocal Enhancement v2.0 - Professional.
Comprehensive vocal processing chain for clarity, presence, and polish.

Algorithm Overview:
1. Vocal Detection:
   - Frequency range analysis (fundamental @ 80-400 Hz)
   - Formant detection (F1/F2 @ 300-3000 Hz)
   - Harmonic series identification
2. Multi-Stage Processing:
   - De-essing (sibilance control @ 6-10 kHz)
   - Presence boost (clarity @ 3-6 kHz)
   - Formant enhancement (vowel clarity @ 1-3 kHz)
   - Breath control (gentle reduction @ 8-12 kHz)
   - Chest resonance (warmth @ 100-250 Hz)
3. Dynamic Processing:
   - Micro-compression (syllable-level dynamics)
   - Envelope shaping (attack/sustain balance)
4. Material Adaptation:
   - Shellac/Vinyl: Restore missing formants
   - Tape: Restore HF detail
   - Digital: Add analog warmth

Scientific Foundation:
- Fant (1960): Acoustic Theory of Speech Production
- Peterson & Barney (1952): Control Methods Used in Study of Vowels
- Hillenbrand et al. (1995): Acoustic Characteristics of American English Vowels
- Sundberg (1987): The Science of the Singing Voice
- Titze (2000): Principles of Voice Production

Industry Benchmarks:
- iZotope Nectar (Vocal processing suite)
- Waves Renaissance Vox (Classic vocal compressor)
- FabFilter Pro-Q 3 (Surgical EQ for vocals)
- Antares Auto-Tune Pro (Pitch + formant correction)
- Universal Audio Neve 1073 (Classic vocal chain)
- SSL Channel Strip (Broadcast vocal processing)

Quality Target: 0.85 → 0.95 (+12% improvement)
Performance Target: <0.30× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import time
from typing import Any, cast

import numpy as np
from scipy import signal

from backend.core.audio_utils import safe_filtfilt, safe_to_mono  # §v10.101
from backend.core.defect_scanner import MaterialType
from backend.core.dsp.stem_routing_policy import prefer_demucs_native_from_material
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# VocalAI Enhancement (Spec §2.8 — Stimmtyp-adaptive Gesangsverarbeitung)
try:
    from backend.core.vocal_ai_enhancement import UnifiedVocalAIEnhancer as _UnifiedVocalAI

    VOCAL_AI_AVAILABLE = True
except ImportError:
    _UnifiedVocalAI = None  # type: ignore
    VOCAL_AI_AVAILABLE = False
    logging.getLogger(__name__).warning("VocalAIEnhancement nicht verfügbar — Standard-DSP-Vogalverarbeitung aktiv")

# FormantSystem: LPC-basiertes Formant-Tracking + Singer's Formant Enhancement (§2.8)
try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls
    from dsp.formant_system import VowelPhonemeFormantTargets as _VowelTargetsCls

    _FORMANT_SYSTEM_AVAILABLE = True
except ImportError:
    _FormantSystemCls = None  # type: ignore
    _VowelTargetsCls = None  # type: ignore
    _FORMANT_SYSTEM_AVAILABLE = False
    logging.getLogger(__name__).debug("FormantSystem (dsp) nicht verfügbar — Bell-EQ-Fallback aktiv")

# DSP-PhonemeDetector für vokal-phonemspezifische Formant-Steuerung
try:
    from plugins.phoneme_detector import get_phoneme_detector as _get_phoneme_detector

    _PHONEME_DETECTOR_AVAILABLE = True
except ImportError:
    _get_phoneme_detector = None  # type: ignore
    _PHONEME_DETECTOR_AVAILABLE = False

# BreathDetector: Segment-basierte Atemerkennung (§2.8 — ZCR + Energie)
try:
    from plugins.breath_detector import get_breath_detector as _get_breath_detector

    _BREATH_DETECTOR_AVAILABLE = True
except ImportError:
    _get_breath_detector = None  # type: ignore
    _BREATH_DETECTOR_AVAILABLE = False

# §2.36 LyricsGuidedEnhancement: Whisper-basierte Phonem-Timeline für Formant-Steering
try:
    from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement as _get_lyrics_guided

    _LYRICS_GUIDED_AVAILABLE = True
except ImportError:
    _get_lyrics_guided = None  # type: ignore
    _LYRICS_GUIDED_AVAILABLE = False

logger = logging.getLogger(__name__)


class VocalEnhancement(PhaseInterface):
    """
    Professional Vocal Enhancement Engine.

    Key Features:
    - Multi-stage vocal processing
    - De-essing (6-10 kHz)
    - Presence boost (3-6 kHz)
    - Formant enhancement (1-3 kHz)
    - Breath control (8-12 kHz)
    - Chest resonance (100-250 Hz)
    - Micro-compression
    - Material-adaptive parameters

    Use Cases:
    - Enhance vocal clarity and intelligibility
    - Restore vintage vocal recordings
    - Polish modern vocal tracks
    - Broadcast vocal optimization

    Performance: <0.30× realtime on modern CPU
    """

    # Vocal frequency bands
    VOCAL_BANDS = {
        "chest": (100, 250),  # Chest resonance (warmth)
        "fundamental": (80, 400),  # Fundamental frequency range
        "formant": (300, 3000),  # Formant region (vowels)
        "presence": (3000, 6000),  # Presence and clarity
        "sibilance": (6000, 10000),  # Sibilance (s, t, sh sounds)
        "breath": (8000, 12000),  # Breath noise
    }

    # Processing parameters (material-adaptive)
    ENHANCEMENT_CONFIG = {
        MaterialType.SHELLAC: {
            "deess_threshold_db": -15,
            "deess_reduction_db": 6,  # Sprint-2: 8 → 6 dB (Vokal-Charakter bewahren)
            "presence_gain_db": 5.0,
            "formant_gain_db": 4.0,
            "chest_gain_db": 3.0,
            "breath_reduction_db": 6,
            "compression_ratio": 1.8,  # Sprint-2: 2.5 → 1.8 (Primum non nocere: Vocal-Dynamik erhalten)
        },
        MaterialType.VINYL: {
            "deess_threshold_db": -18,
            "deess_reduction_db": 6,
            "presence_gain_db": 4.0,
            "formant_gain_db": 3.5,
            "chest_gain_db": 2.5,
            "breath_reduction_db": 5,
            "compression_ratio": 2.0,
        },
        MaterialType.TAPE: {
            "deess_threshold_db": -20,
            "deess_reduction_db": 5,
            "presence_gain_db": 3.5,
            "formant_gain_db": 3.0,
            "chest_gain_db": 2.0,
            "breath_reduction_db": 4,
            "compression_ratio": 1.8,
        },
        MaterialType.CASSETTE: {  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
            "deess_threshold_db": -20,
            "deess_reduction_db": 5,
            "presence_gain_db": 3.5,
            "formant_gain_db": 3.0,
            "chest_gain_db": 2.0,
            "breath_reduction_db": 4,
            "compression_ratio": 1.8,
        },
        MaterialType.CD_DIGITAL: {
            "deess_threshold_db": -20,
            "deess_reduction_db": 6,
            "presence_gain_db": 4.5,
            "formant_gain_db": 4.0,
            "chest_gain_db": 2.5,
            "breath_reduction_db": 5,
            "compression_ratio": 2.2,
        },
        MaterialType.STREAMING: {
            "deess_threshold_db": -18,
            "deess_reduction_db": 6,
            "presence_gain_db": 4.0,
            "formant_gain_db": 3.5,
            "chest_gain_db": 2.5,
            "breath_reduction_db": 5,
            "compression_ratio": 2.0,
        },
    }

    # §VoiceAge Age-Adaptive Enhancement Factors (v10.0.0)
    # Keys correspond to VoiceAgeGroup.value strings.
    # Older voices (MATURE/SENIOR) exhibit inherent breathiness and tremolo that should be
    # preserved, not corrected against.  Younger voices tolerate more aggressive processing.
    # breath_reduction_scale: Multiplier on config["breath_reduction_db"]
    # compression_scale: Multiplier on the effective (ratio-1) portion of config["compression_ratio"]
    # formant_scale: Multiplier on config["formant_gain_db"] (tremolo → less formant correction)
    # chest_scale: Multiplier on config["chest_gain_db"] (senior voices lose low-mid chest resonance)
    # breath_preservation: float passed to VocalAIEnhancer.enhance() (0=aggressive, 1=preserve all)
    _AGE_ADAPTIVE_FACTORS: dict[str, dict[str, float]] = {
        "child": {
            "breath_reduction_scale": 1.00,
            "compression_scale": 0.80,
            "formant_scale": 1.20,
            "chest_scale": 0.70,
            "breath_preservation": 0.75,
        },
        "teenager": {
            "breath_reduction_scale": 1.00,
            "compression_scale": 0.90,
            "formant_scale": 1.10,
            "chest_scale": 0.85,
            "breath_preservation": 0.70,
        },
        "young_adult": {
            "breath_reduction_scale": 1.00,
            "compression_scale": 1.00,
            "formant_scale": 1.00,
            "chest_scale": 1.00,
            "breath_preservation": 0.70,
        },
        "adult": {
            "breath_reduction_scale": 0.95,
            "compression_scale": 1.00,
            "formant_scale": 0.95,
            "chest_scale": 1.05,
            "breath_preservation": 0.72,
        },
        "mature": {
            "breath_reduction_scale": 0.75,
            "compression_scale": 0.85,
            "formant_scale": 0.80,
            "chest_scale": 1.10,
            "breath_preservation": 0.82,
        },
        "senior": {
            "breath_reduction_scale": 0.60,
            "compression_scale": 0.75,
            "formant_scale": 0.65,
            "chest_scale": 1.15,
            "breath_preservation": 0.90,
        },
    }

    # §8.3 Vocal-Intimitäts-Gate (material-adaptiv): maximal tolerierbarer
    # Abfall der Intimitäts-Metrik (fricative/plosive Präsenz) vor Rescue.
    _INTIMACY_MAX_DROP_BY_MATERIAL: dict[MaterialType, float] = {
        MaterialType.SHELLAC: 0.06,
        MaterialType.VINYL: 0.05,
        MaterialType.TAPE: 0.045,
        MaterialType.CASSETTE: 0.045,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: 0.04,
        MaterialType.STREAMING: 0.04,
    }
    _INTIMACY_RESCUE_MAX_BY_MATERIAL: dict[MaterialType, float] = {
        MaterialType.SHELLAC: 0.60,
        MaterialType.VINYL: 0.55,
        MaterialType.TAPE: 0.50,
        MaterialType.CASSETTE: 0.50,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: 0.45,
        MaterialType.STREAMING: 0.45,
    }

    def __init__(self):
        super().__init__()
        self.name = "Vocal Enhancement v2 Professional"
        # LPC-basiertes Formant-Tracking + Singer's Formant Enhancement (2.5–3.5 kHz)
        self._formant_system = None
        if _FORMANT_SYSTEM_AVAILABLE and _FormantSystemCls is not None:
            try:
                self._formant_system = _FormantSystemCls(
                    n_formants=5, correction_strength=0.5, enhance_singers_formant=True
                )
                logger.debug("FormantSystem (LPC) für Verarbeitungsschritt 42 initialisiert")
            except Exception as _e:
                logger.debug("FormantSystem-Init fehlgeschlagen: %s", _e)

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_42_vocal_enhancement",
            name="Vocal Enhancement v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_19_de_esser", "phase_38_presence_boost"],
            estimated_time_factor=0.30,
            version="2.0.0",
            memory_requirement_mb=90,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.95,
            description="Comprehensive vocal processing chain for clarity and polish",
        )

    def process(  # type: ignore  # pylint: disable=arguments-renamed
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        check_ml_model_ready("BS-RoFormer", phase_name="42")
        check_ml_model_ready("Demucs", phase_name="42")
        check_ml_model_ready("PANNs", phase_name="42")
        check_ml_model_ready("Whisper", phase_name="42")
        """
        Apply vocal enhancement to audio — with stem-based processing when separation succeeds.

        Stem-Pipeline (§1.4 StemRemixBalancer, §2.8 Vocal-Kette):
            1. Try bs_roformer → vocals stem, instrument residual
            2. Fallback: demucs_v4 separate_vocals()
            3. Enhance only the vocal stem (DSP chain below)
            4. StemRemixBalancer.balance_remix() → LUFS-korrekter Re-Mix
            5. Fallback: full-audio DSP enhancement (kein Stem-Sep verfügbar)  # §V6 (copilot-instructions.md): logger.warning handled at call site

        Args:
            audio: Input audio (mono or stereo, 48 000 Hz)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing
                **kwargs: Optional ctx: defect_locations (dict), defect_saliency (dict — §9.1c)
                    Wenn vorhanden, werden Lyrics-Salienz-Gewichte in De-Esser Aggressivität integriert:
                    fricative_saliency wird Phase 42 aware → De-Esser spart high-saliency fricatives

        Returns:
            PhaseResult with enhanced audio
        """
        start_time = time.time()
        self.validate_input(audio)
        # §2.51/Zero-Strength-Passthrough: strength=0 muss das Signal bit-genau
        # durchreichen — VOR SOTA-Pipeline und DSP-Kette (kein Eingriff).
        _p42_strength_early = float(kwargs.get("strength", 1.0))
        if _p42_strength_early <= 0.0:
            _audio_out = np.asarray(audio, dtype=np.float32)
            return PhaseResult(
                success=True,
                audio=_audio_out.copy(),
                execution_time_seconds=0.0,
                metadata={
                    "processing": "skipped_zero_strength",
                    "algorithm": "skipped_zero_strength",
                    "effective_strength": 0.0,
                    "strength": 0.0,
                    "material": material.name,
                },
                warnings=[],
            )
        # §v10.210: SOTA Vocal Pipeline — 10 DSP-Module koordiniert
        # Läuft VOR der bestehenden DSP-Kette und liefert Register/Sibilance-Analyse.
        try:
            from backend.core.sota_vocal_pipeline import SOTAVocalPipeline

            _vocal_pipeline = SOTAVocalPipeline()
            _vocal_result = None
            if audio.ndim == 2:
                # §2.51 Stereo-Layout-Erhalt: SOTAVocalPipeline ist mono-only —
                # je Kanal verarbeiten und Originallayout wiederherstellen.
                if audio.shape[0] == 2 and audio.shape[1] != 2:
                    _chunks = [audio[0], audio[1]]  # channels-first (2, N)
                    _res = [_vocal_pipeline.process(np.ascontiguousarray(_c), int(sample_rate)) for _c in _chunks]
                    audio = np.stack([_r.audio for _r in _res], axis=0).astype(np.float32)
                else:
                    _chunks = [audio[:, 0], audio[:, 1]]  # channels-last (N, 2)
                    _res = [_vocal_pipeline.process(np.ascontiguousarray(_c), int(sample_rate)) for _c in _chunks]
                    audio = np.stack([_r.audio for _r in _res], axis=1).astype(np.float32)
                _vocal_result = _res[0]
            else:
                _vocal_result = _vocal_pipeline.process(audio, int(sample_rate))
                audio = _vocal_result.audio
            logger.info(
                "Phase 42: SOTA Vocal Pipeline register=%s deess=%.1fdB harmonic=%.0f%% time=%.1fs",
                _vocal_result.profile.register,
                _vocal_result.sibilance_reduction_db,
                _vocal_result.harmonic_preservation_pct,
                _vocal_result.processing_time,
            )
        except Exception:
            pass  # SOTA Vocal optional — Fallback auf bestehende DSP-Kette

        # §v10.14: MIIPHER-DiT Early-Exit — wenn Phase 03 bereits
        # Flow-Matching-DiT angewandt hat, ist Phase 42 redundant.
        # Doppeltes Enhancement = Overprocessing → Primum non nocere.
        _route_taken = str(kwargs.get("vocal_route_taken", "") or "")
        if _route_taken == "miipher_dit":
            logger.info(
                "Verarbeitungsschritt 42: MIIPHER-DiT bereits in Verarbeitungsschritt 03 gelaufen — überspringe (kein Doppel-Enhancement)"
            )
            return PhaseResult(
                audio=audio,
                metadata={
                    "sample_rate": sample_rate,
                    "algorithm": "skipped_miipher_dit_already_applied",
                    "processing_time_ms": 0.0,
                },
            )
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            # pylint: disable-next=import-outside-toplevel
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_evict42

            _get_plm_evict42().evict_for_phase("phase_42_vocal_enhancement")
        except Exception as e:
            logger.warning("Verarbeitungsschritt_42_vocal_enhancement.py::verarbeiten Ersatzpfad: %s", e)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_42 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_42 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_42,
                )

                _fmz_42 = kwargs.get("forward_masking_zones") or _fmg_fn_42().compute_zones(audio, sample_rate)
                if _fmz_42:
                    _n_s_42 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_42 = sum(z.end_sample - z.start_sample for z in _fmz_42)
                    _zone_frac_42 = float(np.clip(_zone_s_42 / max(1, _n_s_42), 0.0, 1.0))
                    _effective_strength = float(np.clip(_effective_strength + _zone_frac_42 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_42:
                logger.debug("Verarbeitungsschritt42 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_42)

        # ── §SVM-1 SingerVoiceModel: Stimmtyp-adaptives Vocal Enhancement ──
        _svm_42 = kwargs.get("singer_voice_model")
        if _svm_42 is not None and isinstance(_svm_42, dict) and _svm_42.get("confidence", 0.0) > 0.3:
            try:
                _svm_hnr = float(_svm_42.get("hnr_db", 20.0) or 20.0)
                _svm_tilt = float(_svm_42.get("spectral_tilt_db_per_octave", 0.0) or 0.0)
                _svm_conf = float(_svm_42.get("confidence", 0.0))
                _svm_vd = float(_svm_42.get("vibrato_depth_cents", 0.0) or 0.0)
                # Raue Stimme (HNR < 15 dB) → konservativer enhancen
                if _svm_hnr < 15.0:
                    _effective_strength = float(
                        np.clip(_effective_strength * (0.70 + 0.30 * _svm_hnr / 15.0), 0.0, 1.0)
                    )
                # Dunkle Stimme (tilt > 2 dB/oct) → mehr Präsenz-Boost
                if _svm_tilt > 2.0:
                    _effective_strength = float(np.clip(_effective_strength * (1.0 + 0.10 * _svm_conf), 0.0, 1.0))
                # Vibrato-Schutz bei > 50 cent Tiefe → Formant-Lock aktivieren
                if _svm_vd > 50.0:
                    kwargs["preserve_vibrato"] = True
                    kwargs["vibrato_depth_cents"] = _svm_vd
                logger.debug(
                    "Verarbeitungsschritt42 §SVM-1 SVM: hnr=%.1fdB tilt=%.1f vibrato=%.1f → eff=%.3f",
                    _svm_hnr,
                    _svm_tilt,
                    _svm_vd,
                    _effective_strength,
                )
            except Exception as _svm_exc_42:
                logger.debug("Verarbeitungsschritt42 §SVM-1 nicht blockierend: %s", _svm_exc_42)

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
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

        is_stereo = audio.ndim == 2
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        config = dict(self.ENHANCEMENT_CONFIG.get(_mk, self.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL]))  # type: ignore[call-overload]
        config["deess_reduction_db"] = float(config["deess_reduction_db"] * _effective_strength)
        config["presence_gain_db"] = float(config["presence_gain_db"] * _effective_strength)
        config["formant_gain_db"] = float(config["formant_gain_db"] * _effective_strength)
        config["chest_gain_db"] = float(config["chest_gain_db"] * _effective_strength)
        config["breath_reduction_db"] = float(config["breath_reduction_db"] * _effective_strength)
        config["compression_ratio"] = float(1.0 + (config["compression_ratio"] - 1.0) * _effective_strength)

        # §2.46g soft_saturation_severity-Guard (phase_42 hard-cap 0.35):
        # Vocal-Enhancement bei gesättigtem Material reduzieren, damit additive
        # Gain-Stufen (presence/formant) keine Übersteuerungsartefakte verstärken.
        _p42_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        _p42_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        if _p42_sat_preserve and _effective_strength > 0.35:
            _sat_scale_p42 = 0.35 / max(_effective_strength, 1e-6)
            config["presence_gain_db"] = float(config["presence_gain_db"] * _sat_scale_p42)
            config["formant_gain_db"] = float(config["formant_gain_db"] * _sat_scale_p42)
            config["chest_gain_db"] = float(config["chest_gain_db"] * _sat_scale_p42)
            logger.debug(
                "Verarbeitungsschritt42 §2.46g soft_saturation_preserve hard-cap 0.35: scale=%.3f", _sat_scale_p42
            )
        elif _p42_sat_sev > 0.3:
            _sat_scale_p42 = float(np.clip(1.0 - (_p42_sat_sev - 0.3) * 1.2, 0.16, 1.0))
            config["presence_gain_db"] = float(config["presence_gain_db"] * _sat_scale_p42)
            config["formant_gain_db"] = float(config["formant_gain_db"] * _sat_scale_p42)
            config["chest_gain_db"] = float(config["chest_gain_db"] * _sat_scale_p42)
            logger.debug(
                "Verarbeitungsschritt42 §2.46g soft_saturation_severity=%.3f → scale=%.3f (presence/formant/chest geschützt)",
                _p42_sat_sev,
                _sat_scale_p42,
            )

        # §9.10.118 Era-Adaptive De-Esser Thresholds:
        # Vintage microphones (pre-1960) have inherently softer sibilants due
        # to limited HF response — aggressive de-essing removes already-scarce
        # presence detail.  Modern recordings (post-2000) may contain harsher
        # sibilance from condenser mics + digital clipping → need lower trigger.
        # Scientific basis: Eargle (2005) — "Handbook of Recording Engineering".
        _song_cal = kwargs.get("song_calibration_profile") or {}
        _era_decade = None
        if isinstance(_song_cal, dict):
            _era_decade = _song_cal.get("era_decade")
        if _era_decade is not None:
            try:
                _era_int = int(_era_decade)
                if _era_int <= 1940:
                    # Pre-war: ribbon/carbon mic, very soft HF — barely de-ess
                    config["deess_threshold_db"] = float(config["deess_threshold_db"] + 6.0)
                    config["deess_reduction_db"] = float(config["deess_reduction_db"] * 0.5)
                elif _era_int <= 1960:
                    # Early condenser era: softer sibilants than modern
                    config["deess_threshold_db"] = float(config["deess_threshold_db"] + 3.0)
                    config["deess_reduction_db"] = float(config["deess_reduction_db"] * 0.7)
                elif _era_int >= 2000:
                    # Modern digital: potentially harsh sibilance
                    config["deess_threshold_db"] = float(config["deess_threshold_db"] - 2.0)
                logger.debug(
                    "Verarbeitungsschritt42 era-adaptive de-esser: era=%d → Schwelle=%.1f dB, reduction=%.1f dB",
                    _era_int,
                    config["deess_threshold_db"],
                    config["deess_reduction_db"],
                )
            except (TypeError, ValueError):
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        # §2.36 + §9.1c: Salience-aware vocal shaping.
        # Hohe Frikativ-/Sibilanz-Salienz => weniger aggressives De-Essern
        # (Intimität und Artikulation erhalten). Hohe Harshness-Salienz =>
        # Presence leicht dämpfen, damit Schärfe nicht verstärkt wird.
        _defect_saliency_map = kwargs.get("defect_saliency_map", {})
        _sibilance_saliency = 0.0
        _harshness_saliency = 0.0
        if isinstance(_defect_saliency_map, dict):
            _sibilance_saliency = float(np.clip(_defect_saliency_map.get("sibilance", 0.0), 0.0, 1.0))
            _harshness_saliency = float(np.clip(_defect_saliency_map.get("vocal_harshness", 0.0), 0.0, 1.0))

            if _sibilance_saliency > 0.0:
                # Preserve high-saliency consonants: reduce de-ess aggressiveness.
                config["deess_reduction_db"] = float(config["deess_reduction_db"] * (1.0 - 0.50 * _sibilance_saliency))
                config["deess_threshold_db"] = float(config["deess_threshold_db"] + 3.0 * _sibilance_saliency)

            if _harshness_saliency > 0.0:
                # Prevent brittle brightness on harsh material.
                config["presence_gain_db"] = float(config["presence_gain_db"] * (1.0 - 0.20 * _harshness_saliency))

        # --- Vocal Harshness Severity aus DefectScanner-Ergebnis (§v10.0.0) ---
        # Wenn DefectScanner VOCAL_HARSHNESS erkannt hat, wird die Presence-Boost-Phase
        # gedämpft und eine Harshness-Absenkung vorgeschaltet.
        harshness_severity = 0.0
        defect_scores = kwargs.get("defect_scores", {})
        if defect_scores:
            for dt_key, ds_val in defect_scores.items():
                key_str = dt_key.value if hasattr(dt_key, "value") else str(dt_key)
                if key_str == "vocal_harshness":
                    harshness_severity = float(getattr(ds_val, "severity", 0.0) if hasattr(ds_val, "severity") else 0.0)
                    break

        # §2.8 Vocal-Chain: Pipeline-weite Gender-Info aus _restoration_context
        _vocal_gender = str(kwargs.get("vocal_gender", "unknown"))

        # §VoiceAge Age-Adaptive Config Scaling (v10.0.0)
        # Use GenderDetector to estimate age_group and scale config parameters accordingly.
        # Senior/Mature voices: preserve inherent breathiness + tremolo (less correction).
        # Child voices: softer compression, stronger formant enhancement.
        # Non-blocking — falls back to unscaled config on any exception.
        _detected_age_group_value: str | None = None
        _age_breath_preservation: float = 0.70
        if VOCAL_AI_AVAILABLE and _UnifiedVocalAI is not None:
            try:
                # pylint: disable-next=import-outside-toplevel
                from backend.core.vocal_ai_enhancement import GenderDetector as _GenderDetectorCls

                _age_audio = (
                    audio
                    if audio.ndim == 1
                    else (audio[0] if (audio.shape[0] == 2 and audio.shape[1] > 2) else audio[:, 0])
                )
                _age_detector = _GenderDetectorCls(sample_rate=sample_rate)
                _age_chars = _age_detector.detect(_age_audio.astype(np.float32))
                if _age_chars.age_group is not None:
                    _detected_age_group_value = str(_age_chars.age_group.value)
                    _af = self._AGE_ADAPTIVE_FACTORS.get(_detected_age_group_value, {})
                    _brs = float(_af.get("breath_reduction_scale", 1.0))
                    _cs = float(_af.get("compression_scale", 1.0))
                    _fs = float(_af.get("formant_scale", 1.0))
                    _ches = float(_af.get("chest_scale", 1.0))
                    _age_breath_preservation = float(_af.get("breath_preservation", 0.70))
                    config["breath_reduction_db"] = float(config["breath_reduction_db"] * _brs)
                    config["compression_ratio"] = float(1.0 + (config["compression_ratio"] - 1.0) * _cs)
                    config["formant_gain_db"] = float(config["formant_gain_db"] * _fs)
                    config["chest_gain_db"] = float(config["chest_gain_db"] * _ches)
                    logger.info(
                        "Verarbeitungsschritt42 age-adaptive: age_group=%s "
                        "breath_red×%.2f comp×%.2f formant×%.2f chest×%.2f breath_pres=%.2f",
                        _detected_age_group_value,
                        _brs,
                        _cs,
                        _fs,
                        _ches,
                        _age_breath_preservation,
                    )
            except Exception as _age_err:
                logger.debug("Verarbeitungsschritt42 age-detection nicht blockierend: %s", _age_err)

        # §4.10-VintageVoice: Bei Vintage-Material ohne erkannte Altersgruppe ist der
        # Fallback breath_preservation=0.70 zu aggressiv für historische Stimmcharaktere
        # (Caruso, Billie Holiday, früher Jazz, klassische Oper auf Schellack/Vinyl).
        # Vintage-Stimmen haben inhärente Atemigkeit, Vibrato-Fluktuationen und
        # formantbedingte Resonanzkurven, die zum Künstler-Erkennungsbild gehören.
        # Wird nur aktiviert wenn GenderDetector keine Altersgruppe ermitteln konnte
        # (age_group is None) — sobald eine Altersgruppe erkannt wurde, steuert
        # _AGE_ADAPTIVE_FACTORS weiter (Senior=0.90 ist bereits korrekt).
        _vintage_material_keys = frozenset(
            {
                "shellac",
                "wax_cylinder",
                "vinyl",
                "lacquer_disc",
                "wire_recording",
                "acoustic_78",
                "reel_tape",
                "tape",
                "cassette",
            }
        )
        try:
            _material_key_42 = (
                str(material.name).lower()
                if hasattr(material, "name")
                else str(kwargs.get("primary_material", "")).lower()
            )
        except Exception:
            _material_key_42 = ""
        if _detected_age_group_value is None and _material_key_42 in _vintage_material_keys:
            _old_breath_42 = _age_breath_preservation
            _age_breath_preservation = max(_age_breath_preservation, 0.78)
            if _age_breath_preservation > _old_breath_42:
                logger.info(
                    "§4.10-VintageVoice: material='%s' ohne erkannte Altersgruppe → "
                    "breath_preservation-Boden 0.78 (war %.2f) — Vintage-Stimmidentität schützen",
                    _material_key_42,
                    _old_breath_42,
                )

        # §2.36 Lyrics-Salienz-Timeline: per-phoneme adaptive EQ
        # When a lyrics_saliency_timeline is provided (from LyricsGuidedEnhancement),
        # phase_42 adapts the formant/presence Bell-EQ bandwidth per vocal segment:
        #   - High-saliency phonemes (solo vocal, key lyrics) → narrow Q, gentle boost
        #   - Low-saliency phonemes (background, chatter) → wider Q, stronger shaping
        # This preserves articulation where it matters most (§0, Primum non nocere).
        _lyrics_timeline = kwargs.get("lyrics_saliency_timeline")
        _lyrics_adaptation_active = False
        _lyrics_presence_scale = 1.0
        _lyrics_formant_q_scale = 1.0
        if _lyrics_timeline is not None and isinstance(_lyrics_timeline, dict):
            try:
                # Timeline: {(start_s, end_s): saliency_float, ...}
                # Compute average saliency for the entire audio (simplified —
                # per-segment adaptation happens in _enhance_channel).
                _sal_values = [float(v) for v in _lyrics_timeline.values() if v is not None]
                if _sal_values:
                    _avg_sal = float(np.clip(np.mean(_sal_values), 0.0, 1.0))
                    # High average saliency → less aggressive, preserve articulation
                    _lyrics_presence_scale = 1.0 - 0.3 * _avg_sal  # 0.7–1.0
                    _lyrics_formant_q_scale = 1.0 + 0.5 * _avg_sal  # 1.0–1.5 (narrower Q)
                    config["presence_gain_db"] = float(config["presence_gain_db"] * _lyrics_presence_scale)
                    _lyrics_adaptation_active = True
                    logger.info(
                        "Verarbeitungsschritt42 lyrics-saliency adaptation: avg_sal=%.2f presence_scale=%.2f q_scale=%.2f",
                        _avg_sal,
                        _lyrics_presence_scale,
                        _lyrics_formant_q_scale,
                    )
            except Exception as _lsa_exc:
                logger.debug("Verarbeitungsschritt42 lyrics-saliency nicht blockierend: %s", _lsa_exc)

        # §P2 Style-Intent-Guard: intentionale Pitch-Abweichungen in style_intent_zones schützen.
        # In Stil-Zonen (Blue Notes, Microtonal Bends, Culture-Specific Tuning) wird
        # formant_gain_db auf 30 % reduziert — keine falschen Korrekturen intentionaler Stilmerkmale.
        _style_intent_zones = []
        _vfa_result = kwargs.get("vfa_result") or kwargs.get("_restoration_context", {}).get("vfa_result", {})
        if isinstance(_vfa_result, dict):
            _style_intent_zones = list(_vfa_result.get("style_intent_zones", []))
        elif hasattr(_vfa_result, "style_intent_zones"):
            _style_intent_zones = list(_vfa_result.style_intent_zones)
        if _style_intent_zones:
            _audio_duration_s = audio.shape[-1] / max(sample_rate, 1)
            _total_style_s = sum((e - s) for s, e in _style_intent_zones if 0 <= s < e)
            _style_coverage = float(np.clip(_total_style_s / max(_audio_duration_s, 1.0), 0.0, 1.0))
            if _style_coverage > 0.1:  # mind. 10 % Abdeckung für globale Reduktion
                _style_formant_scale = 1.0 - 0.70 * _style_coverage  # bis -70 %
                config["formant_gain_db"] = float(config["formant_gain_db"] * max(_style_formant_scale, 0.30))
                logger.info(
                    "Verarbeitungsschritt42 §P2 style-intent-guard: %d Zonen, coverage=%.1f%% → formant_gain×%.2f",
                    len(_style_intent_zones),
                    _style_coverage * 100,
                    max(_style_formant_scale, 0.30),
                )

        # Detect if audio contains vocals (simple heuristic)
        has_vocals = self._detect_vocals(audio, sample_rate)

        # §0p Passaggio-Schutz [RELEASE_MUST]: Registerübergangszonen (Brust→Kopf→Falsett)
        # erfordern reduzierte Eingriffsstärken — Formant- und Presence-Bearbeitung
        # kann in Übergangszonen Timbre-Knicke erzeugen.
        # §0p: energy_bias in Passaggio = −3 dB (Mittelwert Brust/Kopf) → Gain-Parameter ×0.40.
        _passaggio_zones_p42 = list(kwargs.get("passaggio_zones") or [])
        if not _passaggio_zones_p42 and hasattr(_vfa_result, "passaggio_zones"):
            _passaggio_zones_p42 = list(_vfa_result.passaggio_zones)
        elif not _passaggio_zones_p42 and isinstance(_vfa_result, dict):
            _passaggio_zones_p42 = list(_vfa_result.get("passaggio_zones", []))
        if _passaggio_zones_p42:
            _audio_dur_p42 = audio.shape[-1] / max(sample_rate, 1)
            _passaggio_coverage = float(
                np.clip(
                    sum((e - s) for s, e in _passaggio_zones_p42 if 0 <= s < e) / max(_audio_dur_p42, 1.0),
                    0.0,
                    1.0,
                )
            )
            if _passaggio_coverage > 0.03:  # ≥ 3 % Anteil → Schutz aktivieren
                _p42_passaggio_scale = max(0.40, 1.0 - 0.60 * _passaggio_coverage)
                config["formant_gain_db"] = float(config["formant_gain_db"] * _p42_passaggio_scale)
                config["presence_gain_db"] = float(config["presence_gain_db"] * _p42_passaggio_scale)
                logger.info(
                    "Verarbeitungsschritt42 §0p passaggio-guard: %d Zonen, coverage=%.1f%% → formant/presence×%.2f",
                    len(_passaggio_zones_p42),
                    _passaggio_coverage * 100,
                    _p42_passaggio_scale,
                )

        # Bei Stereo wird auf Mono-Projektion gemessen, um einen robusten
        # kanalunabhängigen Vergleichswert zu erhalten.
        _intimacy_pre = self._measure_vocal_intimacy(audio, sample_rate)

        if not has_vocals:
            logger.info("No vocal content erkannt - skipping vocal enhancement")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "vocals_detected": False,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["No vocal content detected - enhancement skipped"],
            )

        # ── Stem-based vocal enhancement (§1.4 StemRemixBalancer, §2.8) ──────
        _quality_mode_hint = str(kwargs.get("quality_mode", "quality")).strip().lower()
        _quality_first_unleashed = bool(
            kwargs.get("quality_first_unleashed", _quality_mode_hint in ("quality", "maximum"))
        )
        try:
            stem_result = self._try_stem_separation(
                audio,
                sample_rate,
                material=material,
                quality_mode=_quality_mode_hint,
                quality_first_unleashed=_quality_first_unleashed,
            )
        except TypeError as _stem_sep_sig_exc:
            if "unexpected keyword argument" not in str(_stem_sep_sig_exc):
                raise
            logger.debug(
                "Verarbeitungsschritt42 Stem-Sep Ersatzpfad auf Legacy-Signatur ohne quality kwargs: %s",
                _stem_sep_sig_exc,
            )
            try:
                stem_result = self._try_stem_separation(audio, sample_rate, material=material)
            except TypeError as _stem_sep_material_exc:
                if "unexpected keyword argument" not in str(_stem_sep_material_exc):
                    raise
                logger.debug(
                    "Verarbeitungsschritt42 Stem-Sep Ersatzpfad auf Legacy-Signatur ohne material kwargs: %s",
                    _stem_sep_material_exc,
                )
                stem_result = self._try_stem_separation(audio, sample_rate)
        stem_model_used = "none"

        if stem_result is not None:
            vocals_stem, instr_stem, vocal_weight, stem_model_used = stem_result
            logger.debug("Verarbeitungsschritt42: Stem-Sep via %s — verarbeite Vocal-Stem", stem_model_used)

            # §2.28 HPG on vocal stem: HarmonicPreservationGuard info is lost
            # after Stem-Sep because UV3 runs HPG on full mix.  Re-extract
            # harmonic mask from the isolated vocal stem so H1/H2/H3 are
            # protected during formant/presence/compression processing.
            _hpg_vocal_mask = None
            _hpg_vocal_href = None
            try:
                # pylint: disable-next=import-outside-toplevel
                from backend.core.harmonic_preservation_guard import get_harmonic_preservation_guard

                _hpg_v = get_harmonic_preservation_guard()
                _v_mono = safe_to_mono(vocals_stem) if vocals_stem.ndim == 2 else vocals_stem
                _hpg_vocal_mask, _hpg_vocal_href = _hpg_v.extract_harmonic_mask(
                    _v_mono.astype(np.float32), sample_rate, instrument_tag="vocals"
                )
                logger.debug("Verarbeitungsschritt42 HPG: vocal-stem harmonic mask extracted")
            except Exception as _hpg_v_err:
                logger.debug("Verarbeitungsschritt42 HPG auf Vocal-Stem nicht verfügbar: %s", _hpg_v_err)

            # §G58 Vocal Repair: detect and fix damaged vocals before enhancement
            try:
                from backend.core.phases.vocal_repair import apply_vocal_repair, detect_vocal_damage

                _v_repair_mono = safe_to_mono(vocals_stem) if vocals_stem.ndim == 2 else vocals_stem
                _v_damage = detect_vocal_damage(_v_repair_mono.astype(np.float64), sample_rate)
                if _v_damage.get("needs_repair"):
                    vocals_stem = apply_vocal_repair(vocals_stem, sample_rate, damage=_v_damage)
                    logger.info(
                        "Verarbeitungsschritt42 VocalRepair: damage erkannt (bw=%.0fHz crest=%.1fdB) — repaired",
                        _v_damage.get("bandwidth_hz", 0),
                        _v_damage.get("crest_factor_db", 0),
                    )
            except Exception as _vr_exc:
                logger.debug("Verarbeitungsschritt42 VocalRepair uebersprungen: %s", _vr_exc)

            # Enhance only the vocal stem
            if vocals_stem.ndim == 2:
                # §2.51 M/S: vocals are centred → enhance Mid only, Side untouched.
                # Independent L/R _enhance_channel calls introduce time-variant
                # gain divergence between channels → §2.49 phase-cancellation.
                _sqrt2_s = np.sqrt(2.0)
                # Handle both (2,N) channels-first and (N,2) channels-last
                if vocals_stem.shape[0] == 2 and vocals_stem.shape[1] > 2:
                    _vs_ch0, _vs_ch1 = vocals_stem[0], vocals_stem[1]  # (2,N)
                else:
                    _vs_ch0, _vs_ch1 = vocals_stem[:, 0], vocals_stem[:, 1]  # (N,2)
                _v_mid = (_vs_ch0 + _vs_ch1) / _sqrt2_s
                _v_side = (_vs_ch0 - _vs_ch1) / _sqrt2_s
                _v_mid_enh = self._enhance_channel(
                    _v_mid, sample_rate, config, harshness_severity, _vocal_gender, material
                )
                _ev_l = (_v_mid_enh + _v_side) / _sqrt2_s
                _ev_r = (_v_mid_enh - _v_side) / _sqrt2_s
                # Preserve original stereo orientation
                if vocals_stem.shape[0] == 2 and vocals_stem.shape[1] > 2:
                    enhanced_vocals = np.vstack([_ev_l, _ev_r])  # (2,N)
                else:
                    enhanced_vocals = np.column_stack([_ev_l, _ev_r])  # (N,2)
            else:
                enhanced_vocals = self._enhance_channel(
                    vocals_stem, sample_rate, config, harshness_severity, _vocal_gender, material
                )

            # §2.28 HPG correction on enhanced vocal stem: restore harmonic
            # energy that was attenuated by presence-boost or compression.
            if _hpg_vocal_mask is not None and _hpg_vocal_href is not None:
                try:
                    _hpg_v = get_harmonic_preservation_guard()
                    _enh_mono = safe_to_mono(enhanced_vocals) if enhanced_vocals.ndim == 2 else enhanced_vocals
                    _corrected = _hpg_v.apply_correction(
                        _enh_mono.astype(np.float32), _hpg_vocal_href, _hpg_vocal_mask, sample_rate
                    )
                    _corrected = np.clip(np.nan_to_num(_corrected, nan=0.0), -1.0, 1.0)
                    if enhanced_vocals.ndim == 2:
                        # Apply correction as gain ratio to both channels
                        _gain = np.where(np.abs(_enh_mono) > 1e-8, _corrected / (_enh_mono + 1e-12), 1.0)
                        _gain = np.clip(_gain, 0.5, 2.0)
                        enhanced_vocals = enhanced_vocals * _gain[:, np.newaxis]
                    else:
                        enhanced_vocals = _corrected.astype(enhanced_vocals.dtype)
                    logger.debug("Verarbeitungsschritt42 HPG: harmonic correction angewendet to vocal stem")
                except Exception as _hpg_corr_err:
                    logger.debug("Verarbeitungsschritt42 HPG Korrektur fehlgeschlagen: %s", _hpg_corr_err)

            # §8.3 Vocal-Stem MDEM: Recover micro-dynamics ON THE VOCAL STEM ITSELF
            # (before remix, so instrumental doesn't dominate the LUFS profile).
            # The original vocal stem is the reference; enhanced vocal stem is the target.
            enhanced_vocals = self._apply_vocal_stem_mdem(enhanced_vocals, vocals_stem, sample_rate)

            # §2.60 STCG: Compensate any latency introduced by the vocal enhancement chain
            # (STFT rounding, ML inference, compression, formant EQ can shift the stem by
            # sub-sample amounts). The original vocals_stem is the timing reference.
            try:
                from backend.core.stereo_temporal_coherence_guard import (  # pylint: disable=import-outside-toplevel
                    get_stereo_temporal_coherence_guard as _get_stcg,
                )

                enhanced_vocals = _get_stcg().align_stem_to_reference(
                    enhanced_vocals, vocals_stem, sample_rate, stem_label="vocals_enhanced"
                )
            except Exception as _stcg_err:
                logger.debug(
                    "STCG Stem-Align Verarbeitungsschritt_42 fehlgeschlagen (nicht blockierend): %s", _stcg_err
                )

            # StemRemixBalancer: LUFS-korrekter Re-Mix (§1.4 Spec)
            try:
                # pylint: disable-next=import-outside-toplevel
                from backend.core.stem_remix_balancer import StemRemixBalancer

                enhanced_audio = StemRemixBalancer().balance_remix(
                    enhanced_vocals, instr_stem, audio, sample_rate, float(vocal_weight)
                )
            except Exception as _remix_err:
                logger.debug("StemRemixBalancer fehlgeschlagen — Direkt-Mix: %s", _remix_err)
                n = min(enhanced_vocals.shape[0], instr_stem.shape[0])
                enhanced_audio = (enhanced_vocals[:n] + instr_stem[:n]) * 0.5
        else:
            # Fallback: process full audio without stem separation
            logger.debug("Verarbeitungsschritt42: Kein Stem-Sep — Vollbild-Verarbeitung")

            # §2.35c Shellac/WaxCylinder: DeepFormants (§4.4 primär) → LPC-Burg-Fallback.
            # Formant-Enhancement via plugins/formant_tracker.py (DeepFormants CNN ONNX).
            # Dies ersetzt den direkten lpc_formant_tracker-Aufruf: DeepFormants ist höhere
            # Qualitätsstufe, LPC-Burg (§4.4 Fallback) bleibt intern im Plugin erhalten.
            _is_shellac_bw_limited = str(material.value if hasattr(material, "value") else material).lower() in (
                "shellac",
                "wax_cylinder",
                "waxcylinder",
                "cyliner",
            )
            if _is_shellac_bw_limited:
                try:
                    # pylint: disable-next=import-outside-toplevel
                    from plugins.formant_tracker import get_formant_tracker as _get_ft

                    _ft_result = _get_ft().track(audio, sample_rate)
                    if _ft_result.confidence >= 0.5 and len(_ft_result.formants) >= 2:
                        # Gültige Formant-Schätzung → Bell-EQ-Boost über lpc_formant_tracker.enhance()
                        # pylint: disable-next=import-outside-toplevel
                        from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker as _get_lfc

                        _lfc_result = _get_lfc().enhance(
                            audio,
                            sample_rate,
                            era_decade=int(_era_decade) if _era_decade is not None else None,
                        )
                        if _lfc_result is not None and np.isfinite(_lfc_result).all():
                            enhanced_audio = np.clip(_lfc_result, -1.0, 1.0).astype(np.float32)
                            logger.debug(
                                "§2.35c Verarbeitungsschritt_42 FormantTracker(DeepFormants→LPC) aktiv "
                                "(material=%s, F1=%.0f Hz, conf=%.2f)",
                                material,
                                _ft_result.formants[0] if _ft_result.formants else 0.0,
                                _ft_result.confidence,
                            )
                        else:
                            raise ValueError("LPC-Formant-Enhance: ungültiges Ergebnis")
                    else:
                        # Zu wenige voiced Frames → _enhance_channel als sicherer Pfad
                        raise ValueError(
                            f"FormantTracker: niedrige Konfidenz {_ft_result.confidence:.2f} "
                            f"/ {len(_ft_result.formants)} Formanten — kein Boost"
                        )
                except Exception as _lfc_exc:
                    logger.debug("§2.35c FormantTracker fehlgeschlagen → _verbessern_channel: %s", _lfc_exc)
                    # Fallback auf Standard-Vollbild-Enhancement
                    enhanced_audio = self._enhance_channel(
                        audio[:, 0] if audio.ndim == 2 and audio.shape[0] > 2 else audio,
                        sample_rate,
                        config,
                        harshness_severity,
                        _vocal_gender,
                        material,
                    )
            elif is_stereo:
                # §2.51 M/S: enhance Mid only (vocals centred), Side untouched.
                _sqrt2_f = np.sqrt(2.0)
                # Handle both (N,2) channels-last and (2,N) channels-first orientations
                if audio.shape[0] == 2 and audio.shape[1] > 2:
                    _ch0, _ch1 = audio[0], audio[1]  # (2,N)
                else:
                    _ch0, _ch1 = audio[:, 0], audio[:, 1]  # (N,2)
                _f_mid = (_ch0 + _ch1) / _sqrt2_f
                _f_side = (_ch0 - _ch1) / _sqrt2_f
                _f_mid_enh = self._enhance_channel(
                    _f_mid, sample_rate, config, harshness_severity, _vocal_gender, material
                )
                _out_l = (_f_mid_enh + _f_side) / _sqrt2_f
                _out_r = (_f_mid_enh - _f_side) / _sqrt2_f
                # Preserve original stereo orientation
                if audio.shape[0] == 2 and audio.shape[1] > 2:
                    enhanced_audio = np.vstack([_out_l, _out_r])  # (2,N)
                else:
                    enhanced_audio = np.column_stack([_out_l, _out_r])  # (N,2)
            else:
                enhanced_audio = self._enhance_channel(
                    audio, sample_rate, config, harshness_severity, _vocal_gender, material
                )

        # §2.8 VocalAIEnhancement: Optional post-processing with full gender-aware chain
        # (GenderDetector → BreathPreservation → GenderAwareDeEsser → Formant/Emotion check)
        # Only applied when module available AND effective_strength > 0.5 (avoid double processing at low strength)
        _vocal_ai_applied = False
        if VOCAL_AI_AVAILABLE and _UnifiedVocalAI is not None and _effective_strength > 0.5:
            try:
                _vai = _UnifiedVocalAI(sample_rate=sample_rate)
                _vai_result = _vai.enhance(
                    enhanced_audio,
                    breath_preservation=_age_breath_preservation,
                    sibilance_reduction=False,  # Already handled by Phase 19 + de-ess stage
                )
                # Safety: Only accept if formant preservation is high (identity protection)
                if _vai_result.formant_preservation_score >= 0.85:
                    enhanced_audio = _vai_result.audio
                    _vocal_ai_applied = True
                    logger.debug(
                        "Verarbeitungsschritt42 VocalAI: formant_pres=%.2f emotion_pres=%.2f quality_impr=%.3f",
                        _vai_result.formant_preservation_score,
                        _vai_result.emotion_preservation_score,
                        _vai_result.quality_improvement,
                    )
                else:
                    logger.debug(
                        "Verarbeitungsschritt42 VocalAI: abgelehnt (formant_pres=%.2f < 0.85 — Identitätsschutz)",
                        _vai_result.formant_preservation_score,
                    )
            except Exception as _vai_err:
                logger.debug("VocalAIEnhancement fehlgeschlagen (ignoriert): %s", _vai_err)

        # §2.46e [RELEASE_MUST] Hallucination-Guard — nach allen additiven Operationen.
        # Phase_42 fügt Presence-, Formant- und Chest-Energie hinzu (3–5 dB additive Boosts).
        # Auf Shellac (BW ≤ 7 kHz) oder historischem Material kann das synthetische HF-Energie
        # erzeugen, die im Original nie existierte. Restoration: spectral_novelty > 0.15 → Rollback.
        # Studio 2026: spectral_novelty > 0.08 → Score-Penalty (kein Rollback, da Enhancement-Modus).
        # Modus-Normalisierung: UV3 kann entweder den String "studio2026" ODER ein Enum
        # RestoreMode.STUDIO_2026 übergeben → str(enum) = "RestoreMode.STUDIO_2026".
        # .replace('.', '_') → "restoremode_studio_2026", dann "studio" und "2026" erkennbar.
        _raw_p42_mode = kwargs.get("mode", kwargs.get("_mode", "restoration"))
        _p42_mode = str(_raw_p42_mode).strip().lower().replace(".", "_").replace(" ", "")
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _check_hallucination_p42,
            )

            _hg_result = _check_hallucination_p42(
                audio,
                enhanced_audio,
                sr=sample_rate,
                mode=_p42_mode,
                bw_extension_context=True,
            )
            if _hg_result.requires_rollback:
                logger.warning(
                    "Verarbeitungsschritt_42 §2.46e Hallucination-Guard: spectral_novelty=%.3f → Rollback (Betriebsart=%s, material=%s)",
                    _hg_result.spectral_novelty,
                    _p42_mode,
                    material.value if hasattr(material, "value") else str(material),
                )
                enhanced_audio = audio.copy()
            elif _hg_result.score_penalty > 0.0:
                logger.info(
                    "Verarbeitungsschritt_42 §2.46e Hallucination-Guard: spectral_novelty=%.3f → Wert-Penalty %.1f (Betriebsart=%s)",
                    _hg_result.spectral_novelty,
                    _hg_result.score_penalty,
                    _p42_mode,
                )
        except Exception as _hg_exc_p42:
            logger.debug("Verarbeitungsschritt_42 Hallucination-Guard (nicht blockierend): %s", _hg_exc_p42)

        if 0.0 < _effective_strength < 1.0:
            enhanced_audio = audio + _effective_strength * (enhanced_audio - audio)

        # §8.3 Safety-Gate: wenn Intimität signifikant fällt, konservatives
        # Rescue-Blending mit Dry-Signal anwenden.
        _intimacy_post = self._measure_vocal_intimacy(enhanced_audio, sample_rate)
        _intimacy_delta = float(_intimacy_post - _intimacy_pre)
        _intimacy_gate_triggered = False
        _intimacy_rescue_mix = 0.0
        _intimacy_max_drop = float(self._INTIMACY_MAX_DROP_BY_MATERIAL.get(_mk, 0.04))  # type: ignore[call-overload]
        _intimacy_rescue_max = float(self._INTIMACY_RESCUE_MAX_BY_MATERIAL.get(_mk, 0.45))  # type: ignore[call-overload]
        if _intimacy_delta < -_intimacy_max_drop:
            _intimacy_gate_triggered = True
            # Mehr Rescue bei größerem Abfall, aber begrenzt um Effekt zu bewahren.
            _severity = float(min(1.0, max(0.0, (-_intimacy_delta - _intimacy_max_drop) / 0.10)))
            _intimacy_rescue_mix = float(0.20 + (_intimacy_rescue_max - 0.20) * _severity)
            enhanced_audio = (1.0 - _intimacy_rescue_mix) * enhanced_audio + _intimacy_rescue_mix * audio
            _intimacy_post = self._measure_vocal_intimacy(enhanced_audio, sample_rate)
            _intimacy_delta = float(_intimacy_post - _intimacy_pre)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        # §0p panns_singing — wird von Formant-Gate, HNR-Blend und VQI-Gate geteilt
        _p42_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))

        # §0p [RELEASE_MUST] HNR-Blend — nach ML-Enhancement bei Gesangsmaterial (§0p, non-blocking)
        # Pflicht: apply_hnr_blend() nach ML-NR/Enhancement bei panns_singing >= 0.25 (ΔHNR > 3 dB → Dry-Wet-Blend).
        if _p42_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import (
                    apply_hnr_blend as _hnr_blend_p42,  # pylint: disable=import-outside-toplevel
                )

                enhanced_audio, _ = _hnr_blend_p42(audio, enhanced_audio, sample_rate)
            except Exception as _hnr_exc_p42:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_42 (nicht blockierend): %s", _hnr_exc_p42)

        # §0p [RELEASE_MUST] Formant-Gate v10.0.0 — F1–F4 dürfen max. ±2 dB verschoben werden.
        # `check_formant_shift_db()` misst Spektral-Energie an F1–F4-Formantfrequenzen pre/post.
        # max_shift > 2 dB → sofortiger Rollback (§0p Vocal-Invarianten).
        if _p42_panns >= 0.35:
            try:
                from backend.core.dsp.lpc_formant_tracker import (  # pylint: disable=import-outside-toplevel
                    check_formant_shift_db as _check_formant_p42,
                )
                from backend.core.musical_goals.era_vocal_profile import (  # pylint: disable=import-outside-toplevel
                    resolve_formant_tolerance_db as _rft_p42,
                )

                _fg_tol_p42 = float(
                    kwargs.get(
                        "formant_tolerance_db",
                        _rft_p42(
                            era_decade=int(_era_decade) if _era_decade is not None else None,
                            era_profile=kwargs.get("era_vocal_profile"),
                        ),
                    )
                )
                _fg_rollback, _fg_max_shift_db = _check_formant_p42(
                    audio, enhanced_audio, sample_rate, threshold_db=_fg_tol_p42
                )
                if _fg_rollback:
                    logger.warning(
                        "§0p Verarbeitungsschritt_42 Formant-Gate: max_shift=%.1f dB > %.1f dB → Rollback (panns=%.2f)",
                        _fg_max_shift_db,
                        _fg_tol_p42,
                        _p42_panns,
                    )
                    enhanced_audio = audio.copy()
                else:
                    logger.debug("§0p Verarbeitungsschritt_42 Formant-Gate OK: max_shift=%.2f dB", _fg_max_shift_db)
            except Exception as _fg_exc:
                logger.debug("§0p Formant-Gate Verarbeitungsschritt_42 (nicht blockierend): %s", _fg_exc)

        # §0p [RELEASE_MUST] VQI per-Phase Gate — phase_42 ist die aggressivste Vokal-Phase
        # (Formant-Enhancement, Harshness-Reduction, Stem-Separation). VQI < 0.95 → Rollback
        # damit Überprozessierung nicht die Stimmidentität zerstört.
        if _p42_panns >= 0.35:
            try:
                from backend.core.musical_goals.era_vocal_profile import (
                    get_era_vocal_profile as _gevp_p42,  # pylint: disable=import-outside-toplevel  # §EraVocalProfile
                )
                from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
                    compute_vqi as _compute_vqi_p42,
                )

                _vqi_result_p42 = _compute_vqi_p42(
                    audio_orig=audio,
                    audio_restored=enhanced_audio,
                    sr=sample_rate,
                    era_profile=_gevp_p42(int(_era_decade)) if _era_decade is not None else None,
                )
                _vqi_p42 = float(_vqi_result_p42.get("vqi", 1.0))
                if _vqi_p42 < 0.95:
                    logger.info(
                        "Verarbeitungsschritt_42: VQI per-Verarbeitungsschritt rollback (vqi=%.3f < 0.95, panns=%.2f) — Enhancement zurückgesetzt",
                        _vqi_p42,
                        _p42_panns,
                    )
                    enhanced_audio = audio.copy()
            except Exception as _vqi_exc_p42:
                logger.debug(
                    "VQI per-Verarbeitungsschritt Verarbeitungsschritt_42 (nicht blockierend): %s", _vqi_exc_p42
                )

        return PhaseResult(
            success=True,
            audio=enhanced_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "vocals_detected": True,
                "presence_gain_db": float(config["presence_gain_db"]),
                "formant_gain_db": float(config["formant_gain_db"]),
                "compression_ratio": float(config["compression_ratio"]),
                "rt_factor": float(rt_factor),
                "vocal_ai_linked": VOCAL_AI_AVAILABLE,
                "vocal_ai_applied": _vocal_ai_applied,
                "stem_separation_model": stem_model_used,
                "harshness_severity": float(harshness_severity),
                "harshness_reduction_applied": harshness_severity > 0.05,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "vocal_gender": _vocal_gender,
                "sibilance_saliency": float(_sibilance_saliency),
                "vocal_harshness_saliency": float(_harshness_saliency),
                "lyrics_saliency_active": _lyrics_adaptation_active,
                "lyrics_presence_scale": float(_lyrics_presence_scale),
                "lyrics_formant_q_scale": float(_lyrics_formant_q_scale),
                "vocal_intimacy_pre": float(_intimacy_pre),
                "vocal_intimacy_post": float(_intimacy_post),
                "vocal_intimacy_delta": float(_intimacy_delta),
                "vocal_intimacy_gate_triggered": bool(_intimacy_gate_triggered),
                "vocal_intimacy_rescue_mix": float(_intimacy_rescue_mix),
                "vocal_intimacy_max_drop": float(_intimacy_max_drop),
                "vocal_intimacy_rescue_max": float(_intimacy_rescue_max),
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.35 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _measure_vocal_intimacy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Schätzt vocal intimacy from fricative presence and plosive transients.

        Returns a normalized score in [0, 1]. The metric is intentionally
        lightweight and deterministic so it can be used as an in-phase safety gate.
        """
        try:
            x = np.asarray(audio, dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if x.ndim == 2:
                x = np.mean(x, axis=1)
            if x.size < 1024:
                return 0.5

            # Fricative band (4–8 kHz): articulation air/intimacy.
            sos_fric = signal.butter(4, [4000.0, 8000.0], btype="band", fs=sample_rate, output="sos")
            fric = signal.sosfilt(sos_fric, x)
            fric_rms = float(np.sqrt(np.mean(fric**2) + 1e-12))

            # Mid vocal body (300–3500 Hz): reference energy.
            sos_mid = signal.butter(4, [300.0, 3500.0], btype="band", fs=sample_rate, output="sos")
            mid = signal.sosfilt(sos_mid, x)
            mid_rms = float(np.sqrt(np.mean(mid**2) + 1e-12))

            fric_ratio = fric_rms / (mid_rms + 1e-12)
            fric_score = float(np.clip(fric_ratio / 0.35, 0.0, 1.0))

            # Plosive band (120–350 Hz) + transient derivative.
            sos_plo = signal.butter(3, [120.0, 350.0], btype="band", fs=sample_rate, output="sos")
            plo = signal.sosfilt(sos_plo, x)
            dplo = np.diff(plo, prepend=plo[0])
            transient = float(np.percentile(np.abs(dplo), 95))
            ref_std = float(np.std(x) + 1e-9)
            plosive_score = float(np.clip((transient / ref_std) / 3.0, 0.0, 1.0))

            score = 0.55 * fric_score + 0.45 * plosive_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("Verarbeitungsschritt_42_vocal_enhancement.py::_measure_vocal_intimacy Ersatzpfad: %s", e)
            return 0.5

    @staticmethod
    def _wiener_stereo_from_mono(
        audio_stereo: np.ndarray,
        voc_mono: np.ndarray,
        sr: int,  # pylint: disable=unused-argument
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Wendet an: Wiener-style soft masking to preserve stereo field (§9.10.118).

        Instead of duplicating mono stems to L/R (destroying interaural phase
        differences → stereo collapse), compute a time-frequency mask from the
        mono separation and apply it to each original stereo channel.

        Algorithm:
            mask(t,f) = |V_mono(t,f)|^2 / (|V_mono(t,f)|^2 + |I_mono(t,f)|^2 + eps)
            V_L = mask * X_L,  V_R = mask * X_R  (preserves L/R phase)
            I_L = (1-mask) * X_L,  I_R = (1-mask) * X_R

        Scientific basis: Liutkus et al. (2014) — "Kernel Additive Models for
        Source Separation"; Wiener optimal gain minimises MSE while preserving
        original phase.
        """
        # Handle both (2,N) and (N,2) stereo orientations
        _is_channels_first = bool(audio_stereo.ndim == 2 and audio_stereo.shape[0] == 2 and audio_stereo.shape[1] > 2)
        _n_samples = audio_stereo.shape[1] if _is_channels_first else audio_stereo.shape[0]
        n = min(_n_samples, len(voc_mono))
        audio_st = audio_stereo[:, :n] if _is_channels_first else audio_stereo[:n]
        voc_m = voc_mono[:n]
        inst_m = np.clip(safe_to_mono(audio_st).astype(np.float32) - voc_m, -1.0, 1.0)

        # STFT parameters matching MRSA reference window
        win_size = 2048
        hop = win_size // 4
        window = np.hanning(win_size).astype(np.float32)

        # Compute magnitude masks from mono separation
        V_stft = np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(np.pad(voc_m, (0, win_size)), win_size)[::hop] * window
        )
        I_stft = np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(np.pad(inst_m, (0, win_size)), win_size)[::hop] * window
        )

        V_pow = np.abs(V_stft) ** 2
        I_pow = np.abs(I_stft) ** 2
        eps = 1e-10
        mask = V_pow / (V_pow + I_pow + eps)  # Wiener gain ∈ [0, 1]

        # Smooth mask temporally (3 frames ≈ 32 ms) to reduce musical noise
        kernel = np.ones(3) / 3.0
        for f_idx in range(mask.shape[1]):
            mask[:, f_idx] = np.convolve(mask[:, f_idx], kernel, mode="same")
        mask = np.clip(mask, 0.0, 1.0)

        # Apply mask to each stereo channel in STFT domain
        vocals_out = np.zeros_like(audio_st)
        instr_out = np.zeros_like(audio_st)
        for ch in range(2):
            ch_data = (audio_st[ch, :] if _is_channels_first else audio_st[:, ch]).astype(np.float32)
            ch_padded = np.pad(ch_data, (0, win_size))
            frames = np.lib.stride_tricks.sliding_window_view(ch_padded, win_size)[::hop]
            X_ch = np.fft.rfft(frames * window)

            V_ch = X_ch * mask
            I_ch = X_ch * (1.0 - mask)

            # Overlap-add reconstruction
            v_out = np.zeros(n + win_size, dtype=np.float32)
            i_out = np.zeros(n + win_size, dtype=np.float32)
            for t_idx in range(min(V_ch.shape[0], frames.shape[0])):
                pos = t_idx * hop
                v_frame = np.fft.irfft(V_ch[t_idx], n=win_size).real.astype(np.float32)
                i_frame = np.fft.irfft(I_ch[t_idx], n=win_size).real.astype(np.float32)
                end = min(pos + win_size, len(v_out))
                seg = end - pos
                v_out[pos:end] += v_frame[:seg] * window[:seg]
                i_out[pos:end] += i_frame[:seg] * window[:seg]

            # Normalise OLA (Hann window 75% overlap → sum(w²) = constant)
            norm = np.zeros(n + win_size, dtype=np.float32)
            for t_idx in range(min(V_ch.shape[0], frames.shape[0])):
                pos = t_idx * hop
                end = min(pos + win_size, len(norm))
                seg = end - pos
                norm[pos:end] += window[:seg] ** 2
            norm = np.maximum(norm, 1e-8)

            _v_ch = np.clip(v_out[:n] / norm[:n], -1.0, 1.0)
            _i_ch = np.clip(i_out[:n] / norm[:n], -1.0, 1.0)

            # §2.61 / §0h OLA Tail-Ringing Guard — last win_size samples may contain
            # STFT ringing from the zero-padded tail. Suppress output to max(input_rms)
            # in the tail region to prevent Pegelexplosion at song end.
            _tail_len = min(win_size, n)
            _tail_orig = ch_data[n - _tail_len :]
            _tail_orig_rms = float(np.sqrt(np.mean(_tail_orig**2)) + 1e-9)
            _tail_v_rms = float(np.sqrt(np.mean(_v_ch[n - _tail_len :] ** 2)) + 1e-9)
            _tail_i_rms = float(np.sqrt(np.mean(_i_ch[n - _tail_len :] ** 2)) + 1e-9)
            # If the OLA output tail is louder than input (ringing artifact) → scale down
            if _tail_v_rms > _tail_orig_rms * 1.05:
                _scale = min(_tail_orig_rms / _tail_v_rms, 1.0)
                _v_ch[n - _tail_len :] *= _scale
            if _tail_i_rms > _tail_orig_rms * 1.05:
                _scale = min(_tail_orig_rms / _tail_i_rms, 1.0)
                _i_ch[n - _tail_len :] *= _scale

            if _is_channels_first:
                vocals_out[ch, :] = _v_ch
                instr_out[ch, :] = _i_ch
            else:
                vocals_out[:, ch] = _v_ch
                instr_out[:, ch] = _i_ch

        return vocals_out.astype(np.float32), instr_out.astype(np.float32)

    def _try_stem_separation(
        self,
        audio: np.ndarray,
        sr: int,
        material: MaterialType = MaterialType.CD_DIGITAL,
        quality_mode: str = "quality",
        quality_first_unleashed: bool = False,
    ) -> "tuple[np.ndarray, np.ndarray, float, str] | None":
        """Vocal/Instrument stem separation cascade: bs_roformer → demucs_v4 → mdx23c → dsp.

        Returns (vocals, instruments, vocal_weight, model_name) or None on total failure.
        Both stems match the input shape (mono [n] or stereo [n, 2]).

        §9.10.118 Stereo Preservation: For stereo input, mono separation is used
        to compute a Wiener soft mask, which is then applied to each L/R channel
        individually — preserving interaural phase differences (stereo imaging).
        """
        # Convert for mono-based models; keep original shape for result
        audio_mono = safe_to_mono(audio).astype(np.float32)
        _prefer_demucs_native = self._prefer_demucs_native(material)

        # §v10.60 Depth-Aware Fast-Path: Bei transfer_depth ≥ 3 sind ML-Stem-Separation-
        # Modelle (MelBandRoformer, MDX23C) für moderne, saubere Produktionen trainiert.
        # Auf 4-fach degradierten Signalen (reel→vinyl→cassette→mp3) produzieren sie
        # SDRi < -1 dB → Fallback-Kaskade → HPSS. Zeitverschwendung + Artefakt-Risiko.
        # Direkt HPSS + Wiener — zuverlässig, schnell, keine ML-Halluzination.
        _chain_p42 = (getattr(self, "_restoration_context", {}) or {}).get("transfer_chain", [])
        _depth_p42 = len(_chain_p42) if _chain_p42 else 1
        if _depth_p42 >= 5:
            logger.info(
                "Verarbeitungsschritt42 depth=%d → Fast-Path HPSS+Wiener (ML-Stem-Sep übersprungen bei degradiertem Signal)",
                _depth_p42,
            )
            return self._hpss_wiener_fallback(audio, audio_mono, sr)

        _skip_roformer_reason: str | None = None
        try:
            import psutil as _psutil  # pylint: disable=import-outside-toplevel

            _avail_gb = float(_psutil.virtual_memory().available / (1024**3))
            # MelBandRoformer (T²-Transformer) needs ~9 GB working memory per 15s chunk.
            # With 7s chunks (v10.0.0) ~2 GB each — but swap pressure can still OOM at <12 GB.
            if _avail_gb < 12.0:
                _skip_roformer_reason = f"low_ram_{_avail_gb:.1f}GB"
        except Exception:
            _avail_gb = None  # type: ignore[assignment]

        # Quality-first policy: do not skip RoFormer only because the material is long.
        # Time factor must not degrade quality in quality/maximum paths.
        if _skip_roformer_reason is None and not quality_first_unleashed and quality_mode not in ("quality", "maximum"):
            _duration_s = float(len(audio_mono) / max(1, sr))
            if _duration_s > 120.0:
                _skip_roformer_reason = f"long_audio_{_duration_s:.1f}s"

        # ── 1: BSRoFormer (MelBandRoformer, falls Modell verfügbar) ──────────
        if _skip_roformer_reason is not None:
            logger.info(
                "Verarbeitungsschritt42 Stem-Sep: bs_roformer übersprungen (%s) — direkter Ersatzpfad auf MDX23C/NMF/HPSS",
                _skip_roformer_reason,
            )
        else:
            _plm42_rof = None
            try:
                from plugins.bs_roformer_plugin import get_bs_roformer  # pylint: disable=import-outside-toplevel

                try:
                    # pylint: disable-next=import-outside-toplevel
                    from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm42r

                    _plm42_rof = _get_plm42r()
                    _plm42_rof.set_active("MelBandRoformer", True)
                except Exception:
                    _plm42_rof = None

                roformer = get_bs_roformer()
                if _plm42_rof is not None:
                    try:
                        _plm42_rof.touch_plugin("MelBandRoformer")  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning(
                            "Verarbeitungsschritt_42_vocal_enhancement.py::_try_stem_separation Ersatzpfad: %s", e
                        )
                sep = roformer.separate(audio_mono, sr, stems=["vocals"])
                if sep is not None and "vocals" in sep.stems:
                    _sdri_db = float(getattr(sep, "sdri_db", 0.0))
                    if _sdri_db < -1.0:
                        logger.warning(
                            "Verarbeitungsschritt42 Stem-Sep: bs_roformer SDRi=%.1f dB < -1.0 dB → Ersatzpfad auf MDX23C/NMF/HPSS",
                            _sdri_db,
                        )
                        raise ValueError(f"bs_roformer_low_sdri:{_sdri_db:.2f}")
                    voc_mono = np.asarray(sep.stems["vocals"], dtype=np.float32)
                    n = min(len(audio_mono), len(voc_mono))
                    inst_mono = np.clip(audio_mono[:n] - voc_mono[:n], -1.0, 1.0)
                    if audio.ndim == 2:
                        # §9.10.118: Wiener stereo masking preserves L/R phase
                        vocals_out, instr_out = self._wiener_stereo_from_mono(audio[:n], voc_mono[:n], sr)
                    else:
                        vocals_out = voc_mono[:n]
                        instr_out = inst_mono
                    confidence = float(getattr(sep, "confidence", 0.5))
                    logger.debug(
                        "Verarbeitungsschritt42 Stem-Sep: bs_roformer confidence=%.2f model=%s (stereo=%s)",
                        confidence,
                        sep.model_used,
                        audio.ndim == 2,
                    )
                    return vocals_out, instr_out, confidence, sep.model_used
            except Exception as exc:
                logger.debug("Verarbeitungsschritt42 bs_roformer fehlgeschlagen: %s", exc)
            finally:
                if _plm42_rof is not None:
                    try:
                        _plm42_rof.set_active("MelBandRoformer", False)
                    except Exception as e:
                        logger.warning("Verarbeitungsschritt_42_vocal_enhancement.py::unbekannter Ersatzpfad: %s", e)

        # ── 2: HTDemucs 6s fallback (nur live/crowd + native Session) ───────
        if _prefer_demucs_native:
            if _avail_gb is not None and _avail_gb < 5.0:
                logger.info(
                    "Verarbeitungsschritt42 Stem-Sep: demucs_v4 übersprungen (low_ram_%.1fGB) — Ersatzpfad auf MDX23C/NMF/HPSS",
                    _avail_gb,
                )
            else:
                try:
                    from plugins.demucs_v4_plugin import get_demucs_plugin  # pylint: disable=import-outside-toplevel

                    demucs = get_demucs_plugin()
                    if getattr(demucs, "_session", None) is None:
                        logger.debug("Verarbeitungsschritt42 demucs_v4 übersprungen: keine native HTDemucs-Sitzung")
                    else:
                        try:
                            voc_mono, inst_mono = demucs.separate_vocals(audio_mono, sr, prefer_mdx23c=False)
                        except TypeError:
                            # Backward compatibility for older plugin stubs in tests.
                            voc_mono, inst_mono = demucs.separate_vocals(audio_mono, sr)
                        n = min(len(audio_mono), len(voc_mono), len(inst_mono))
                        if audio.ndim == 2:
                            vocals_out, instr_out = self._wiener_stereo_from_mono(
                                audio[:n], np.asarray(voc_mono[:n]), sr
                            )
                        else:
                            vocals_out = np.asarray(voc_mono[:n], dtype=np.float32)
                            instr_out = np.asarray(inst_mono[:n], dtype=np.float32)
                        return vocals_out, instr_out, 0.60, "demucs_v4_htdemucs"
                except Exception as exc:
                    logger.debug("Verarbeitungsschritt42 demucs_v4 fehlgeschlagen: %s", exc)

        # ── 3: MDX23C fallback (Kim_Vocal_2) ─────────────────────────────────
        _plm42_mdx = None
        if _avail_gb is not None and _avail_gb < 3.0:
            logger.info(
                "Verarbeitungsschritt42 Stem-Sep: mdx23c übersprungen (low_ram_%.1fGB) — Ersatzpfad auf NMF/HPSS",
                _avail_gb,
            )
        else:
            try:
                from plugins.mdx23c_plugin import get_mdx23c_plugin  # pylint: disable=import-outside-toplevel

                try:
                    # pylint: disable-next=import-outside-toplevel
                    from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm42m

                    _plm42_mdx = _get_plm42m()
                    # MDX23C registers as "MDX23C_vocals" and "MDX23C_inst" in PLM (§4.6c sync)
                    _plm42_mdx.set_active("MDX23C_vocals", True)
                    _plm42_mdx.set_active("MDX23C_inst", True)
                except Exception:
                    _plm42_mdx = None

                try:
                    from plugins.mdx23c_plugin import get_loaded_mdx23c_plugin
                except Exception:
                    get_loaded_mdx23c_plugin = None  # type: ignore[assignment]

                mdx = get_loaded_mdx23c_plugin() if callable(get_loaded_mdx23c_plugin) else None
                if mdx is None:
                    mdx = get_mdx23c_plugin()
                if _plm42_mdx is not None:
                    try:
                        _plm42_mdx.touch_plugin("MDX23C_vocals")  # type: ignore[attr-defined]
                        _plm42_mdx.touch_plugin("MDX23C_inst")  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning("Verarbeitungsschritt_42_vocal_enhancement.py::unbekannter Ersatzpfad: %s", e)
                voc_mono = mdx.process(audio_mono, sr, stem="vocals")
                if _plm42_mdx is not None:
                    try:
                        _plm42_mdx.touch_plugin("MDX23C_vocals")  # type: ignore[attr-defined]
                        _plm42_mdx.touch_plugin("MDX23C_inst")  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning("Verarbeitungsschritt_42_vocal_enhancement.py::unbekannter Ersatzpfad: %s", e)
                inst_mono = mdx.process(audio_mono, sr, stem="inst")
                n = min(len(audio_mono), len(voc_mono), len(inst_mono))
                if audio.ndim == 2:
                    # §9.10.118: Wiener stereo masking preserves L/R phase
                    vocals_out, instr_out = self._wiener_stereo_from_mono(audio[:n], voc_mono[:n], sr)
                else:
                    vocals_out = voc_mono[:n]
                    instr_out = inst_mono[:n]
                return vocals_out, instr_out, 0.65, "mdx23c_kim_vocal_2"
            except Exception as exc:
                logger.debug("Verarbeitungsschritt42 mdx23c fehlgeschlagen: %s", exc)
            finally:
                if _plm42_mdx is not None:
                    try:
                        _plm42_mdx.set_active("MDX23C_vocals", False)
                        _plm42_mdx.set_active("MDX23C_inst", False)
                    except Exception as e:
                        logger.warning("Verarbeitungsschritt_42_vocal_enhancement.py::unbekannter Ersatzpfad: %s", e)

        # ── 4: NMF-β Fallback (§2.47 ML-Failure-Degradationskascade: NMF-β→HPSS) ──
        try:
            from plugins.mdx23c_plugin import MDX23CModel as _MDX23CModel  # pylint: disable=import-outside-toplevel

            _audio_2d = audio_mono[np.newaxis, :] if audio_mono.ndim == 1 else audio_mono
            voc_nmf_2d = _MDX23CModel._nmf_beta_fallback(_audio_2d, is_vocals=True)  # pylint: disable=protected-access
            inst_nmf_2d = _MDX23CModel._nmf_beta_fallback(_audio_2d, is_vocals=False)  # pylint: disable=protected-access
            voc_nmf = voc_nmf_2d[0] if voc_nmf_2d.ndim == 2 else voc_nmf_2d
            inst_nmf = inst_nmf_2d[0] if inst_nmf_2d.ndim == 2 else inst_nmf_2d
            # §2.47 sdB ≥ 5 Guard: NMF-β nur wenn Separation-Güte ausreichend.
            # sdB-Proxy: Verhältnis Vokal-RMS zu Instrument-RMS; < 5 dB → HPSS tertiär.
            _voc_rms = float(np.sqrt(np.mean(voc_nmf**2) + 1e-12))
            _inst_rms = float(np.sqrt(np.mean(inst_nmf**2) + 1e-12))
            _sdb = float(20.0 * np.log10(_voc_rms / (_inst_rms + 1e-12)))
            if _sdb < 5.0:
                logger.info(
                    "Verarbeitungsschritt42 NMF-β: sdB=%.1f dB < 5 dB → HPSS tertiärer Ersatzpfad (§2.47)",
                    _sdb,
                )
                raise ValueError(f"NMF-β sdB {_sdb:.1f} dB < 5 dB threshold")
            n = min(len(audio_mono), len(voc_nmf))
            if audio.ndim == 2:
                vocals_out, instr_out = self._wiener_stereo_from_mono(audio[:n], voc_nmf[:n], sr)
            else:
                vocals_out = voc_nmf[:n]
                instr_out = inst_nmf[:n]
            logger.debug("Verarbeitungsschritt42 NMF-β Ersatzpfad erfolgreich (§2.47) sdB=%.1f dB", _sdb)
            return vocals_out, instr_out, 0.45, "nmf_beta_dsp"
        except Exception as exc:
            logger.debug("Verarbeitungsschritt42 NMF-β fehlgeschlagen: %s — HPSS tertiärer Ersatzpfad", exc)

        # ── 4: HPSS tertiärer Fallback ────────────────────────────────────────
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            if audio.ndim == 2:
                mono_in = audio_mono
            else:
                mono_in = audio
            # Wall-time guard: HPSS on long audio (>120s) is slow on CPU.
            # Limit to 90 s budget; if exceeded, return None and skip phase.
            _hpss_max_s = 90.0
            _audio_dur_s = float(len(mono_in)) / max(1, sr)
            if _audio_dur_s > 120.0:
                logger.info(
                    "Verarbeitungsschritt42 HPSS tertiärer Ersatzpfad: audio=%.1fs > 120s — wall-time guard = %.0fs",
                    _audio_dur_s,
                    _hpss_max_s,
                )
            _t0_hpss = time.monotonic()
            harmonic_mono, _ = librosa.effects.hpss(mono_in)  # type: ignore[attr-defined]  # librosa-Stubs exportieren effects nicht
            _hpss_elapsed = time.monotonic() - _t0_hpss
            if _hpss_elapsed > _hpss_max_s:
                logger.warning(
                    "Verarbeitungsschritt42 HPSS wall-time überschritten (%.0fs > %.0fs) — Ergebnis verworfen",
                    _hpss_elapsed,
                    _hpss_max_s,
                )
                return None
            n = min(len(audio_mono), len(harmonic_mono))
            if audio.ndim == 2:
                vocals_out, instr_out = self._wiener_stereo_from_mono(audio[:n], harmonic_mono[:n], sr)
            else:
                vocals_out = harmonic_mono[:n]
                instr_out = np.clip(audio_mono[:n] - harmonic_mono[:n], -1.0, 1.0)
            logger.debug(
                "Verarbeitungsschritt42 HPSS tertiärer Ersatzpfad erfolgreich (§2.47) elapsed=%.1fs", _hpss_elapsed
            )
            return vocals_out, instr_out, 0.30, "hpss_tertiary"
        except Exception as exc:
            logger.debug("Verarbeitungsschritt42 HPSS Ersatzpfad fehlgeschlagen: %s", exc)

        return None

    def _hpss_wiener_fallback(
        self, audio: np.ndarray, audio_mono: np.ndarray, sr: int
    ) -> "tuple[np.ndarray, np.ndarray, float, str] | None":
        """§v10.60 HPSS+Wiener Fast-Path für depth≥3.

        Überspringt die gesamte ML-Kaskade (MelBandRoformer→MDX23C→NMF-β)
        und verwendet direkt HPSS mit Wiener-Stereo-Rekonstruktion.
        Zuverlässig auf degradierten Signalen, keine ML-Halluzination.
        """
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            mono_in = audio_mono if audio.ndim == 2 else audio
            harmonic_mono, _ = librosa.effects.hpss(mono_in)  # type: ignore[attr-defined]  # librosa-Stubs exportieren effects nicht
            n = min(len(audio_mono), len(harmonic_mono))
            if audio.ndim == 2:
                vocals_out, instr_out = self._wiener_stereo_from_mono(audio[:n], harmonic_mono[:n], sr)
            else:
                vocals_out = harmonic_mono[:n]
                instr_out = np.clip(audio_mono[:n] - harmonic_mono[:n], -1.0, 1.0)
            logger.debug("Verarbeitungsschritt42 HPSS+Wiener Fast-Path: n=%d samples", n)
            return vocals_out, instr_out, 0.35, "hpss_wiener_fastpath"
        except Exception as exc:
            logger.debug("Verarbeitungsschritt42 HPSS+Wiener Fast-Path fehlgeschlagen: %s", exc)
            return None

    def _detect_vocals(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Einfache Vokalerkennung auf Basis von Formant-Energie."""
        if audio.ndim == 2:
            # Channel-format-aware: (2, N) channels-first → audio[:, 0] = shape (2,) — wrong.
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                audio = audio[0]  # channels-first: take first channel row
            else:
                audio = audio[:, 0]  # channels-last: take first channel column

        # Measure energy in formant region (300-3000 Hz)
        sos_formant = signal.butter(4, self.VOCAL_BANDS["formant"], btype="band", fs=sample_rate, output="sos")
        formant_signal = signal.sosfilt(sos_formant, audio)
        formant_energy = float(np.mean(formant_signal**2))

        # Measure total energy
        total_energy = float(np.mean(audio**2))

        # If formant region has >20% of total energy, likely contains vocals
        if total_energy > 1e-10:
            formant_ratio = float(formant_energy / total_energy)
            return formant_ratio > 0.20
        else:
            return False

    def _enhance_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: dict[str, Any],
        harshness_severity: float = 0.0,
        vocal_gender: str = "unknown",
        material_type: "MaterialType | None" = None,
    ) -> np.ndarray:
        """Enhance vocals in a single audio channel.

        §2.8 Integration (v10.0.0):
        - PhonemeDetector → phoneme-guided formant steering
        - BreathDetector → segment-aware breath reduction (replaces static bandpass)
        - Gender → passed to FormantSystem for per-gender formant targets

        §Hebel-4 (v10.0.0):
        - Carrier-Formant-Decay-Inversion between Stage 0 and Stage 1
        """
        enhanced = audio.copy()

        # Stage 0: Harshness reduction (NEW — §v10.0.0)
        # When DefectScanner detects VOCAL_HARSHNESS, apply targeted mid-presence
        # notch/dip BEFORE any enhancement to remove harsh/scratchy character.
        if harshness_severity > 0.05:
            enhanced = self._reduce_harshness(enhanced, sample_rate, harshness_severity)

        # Stage 0.5: §Hebel-4 Carrier-Formant-Decay-Inversion.
        # Detects and inverts the systematic per-formant frequency-response decay
        # introduced by the carrier medium's transfer function (vinyl lacquer,
        # tape oxide layer, shellac binder). Applies only for analog carrier types.
        if material_type is not None:
            enhanced = self._restore_carrier_formant_decay(enhanced, sample_rate, material_type)

        # Stage 1: De-essing (sibilance control)
        enhanced = self._apply_deessing(enhanced, sample_rate, config)

        # Stage 2: Formant enhancement (vowel clarity) — with PhonemeDetector + Gender
        enhanced = self._enhance_formants(enhanced, sample_rate, config, vocal_gender)

        # Stage 3: Presence boost (clarity) — attenuated when harshness detected
        if harshness_severity > 0.3:
            # Reduce presence boost proportionally to harshness severity
            adapted_config = dict(config)
            reduction = min(harshness_severity * 0.8, 0.9)  # up to 90% reduction
            adapted_config["presence_gain_db"] = config["presence_gain_db"] * (1.0 - reduction)
            logger.debug(
                "Verarbeitungsschritt42: Harshness %.2f → presence_gain reduced %.1f→%.1f dB",
                harshness_severity,
                config["presence_gain_db"],
                adapted_config["presence_gain_db"],
            )
            enhanced = self._boost_presence(enhanced, sample_rate, adapted_config)
        else:
            enhanced = self._boost_presence(enhanced, sample_rate, config)

        # Stage 4: Chest resonance (warmth)
        enhanced = self._enhance_chest(enhanced, sample_rate, config)

        # Stage 5: Breath control — §2.8 segment-aware via BreathDetector
        enhanced = self._control_breath(enhanced, sample_rate, config)

        # Stage 6: Micro-compression (dynamics)
        enhanced = self._apply_compression(enhanced, sample_rate, config)

        return enhanced

    def _restore_carrier_formant_decay(
        self,
        audio: np.ndarray,
        sample_rate: int,
        material_type: "MaterialType",
    ) -> np.ndarray:
        """§Hebel-4: Carrier-Formant-Decay-Inversion (v10.0.0).

        Inverts the systematic per-formant frequency-response decay introduced by
        the physical carrier medium's transfer function. Analog carriers attenuate
        specific formant regions due to:
          - vinyl:     groove geometry + playback-stylus cross-talk → F2/F3 roll-off
          - reel_tape: oxide-layer + bias HF compression → F3/F4 attenuation
          - shellac:   binder composition + narrow groove → F2 attenuation, strong HF mask
          - cassette:  slow tape speed + Dolby NR interaction → F3/F4 suppression

        Algorithm (DSP-only, < 8 ms @ 48 kHz / 3 min audio):
          1. LPC analysis (order 32) on voiced segments → estimated formant peaks F1–F4
          2. Compare measured formant amplitudes against material-canonical ceilings
          3. Per-formant deficit = measured − canonical_ceiling (only when deficit < 0)
          4. Bell peaking eq at each formant with deficit-corrected gain (filtfilt → zero-phase)
          5. §0 Safety: material-adaptive max_gain_db cap; no overcorrection

        References:
          - Peterson & Barney (1952) — F1/F2 targets for English vowels
          - Sundberg (1974) — Singer's Formant ≈ 2.5–3.5 kHz
          - Eargle (2005) — Handbook of Recording Engineering, carrier transfer functions
        """
        try:
            n = audio.shape[-1] if audio.ndim >= 2 else len(audio)
            if n < 2048:
                return audio

            _mat = material_type.value if hasattr(material_type, "value") else str(material_type)

            # Carrier-specific attenuation profiles: (formant_hz, canonical_ceiling_dbfs, max_correction_db, Q)
            # ceiling: max formant energy a pristine recording reaches on this carrier type
            # max_correction_db: §0 safety cap per formant band
            _CARRIER_PROFILES: dict[str, list[tuple[int, float, float, float]]] = {
                "vinyl": [
                    (500, -24.0, 2.5, 3.0),  # F1 — largely preserved
                    (1500, -22.0, 3.5, 2.0),  # F2 — groove cross-talk, moderate decay
                    (2500, -26.0, 4.0, 2.5),  # F3 — stylus resonance zone
                    (3500, -30.0, 3.0, 3.0),  # F4 — inner-groove HF loss
                ],
                "reel_tape": [
                    (500, -22.0, 2.0, 3.0),
                    (1500, -21.0, 2.5, 2.0),
                    (2500, -27.0, 4.0, 2.5),  # HF oxide compression zone
                    (3500, -32.0, 3.5, 3.0),
                ],
                "tape": [
                    (500, -22.0, 2.0, 3.0),
                    (1500, -21.0, 2.5, 2.0),
                    (2500, -27.0, 4.0, 2.5),
                    (3500, -32.0, 3.5, 3.0),
                ],
                "shellac": [
                    (500, -26.0, 2.5, 3.5),
                    (1500, -28.0, 4.5, 2.0),  # Shellac binder severe F2 loss
                    (2500, -34.0, 3.0, 3.0),  # BW ceiling ≤ 7 kHz by §0
                    (3200, -38.0, 2.5, 3.5),
                ],
                "minidisc": [
                    (500, -23.0, 2.0, 3.0),
                    (1500, -22.0, 2.5, 2.0),
                    (2500, -29.0, 4.5, 2.5),  # ATRAC codec + slow-speed interaction
                    (3500, -33.0, 4.0, 3.0),
                ],
            }

            _profile = _CARRIER_PROFILES.get(_mat)
            if _profile is None:
                # cd_digital, mp3, dat — no carrier formant decay to invert
                return audio

            # Use mono for level analysis (always), then apply on each channel
            x_mono = safe_to_mono(audio)

            # Measure per-formant energy as dBFS using narrow bandpass RMS
            def _formant_energy_dbfs(sig: np.ndarray, center_hz: int, q: float) -> float:
                bw = center_hz / q
                lo = max(20.0, center_hz - bw * 0.5)
                hi = min(sample_rate * 0.47, center_hz + bw * 0.5)
                if hi <= lo + 10:
                    return -80.0
                try:
                    sos = signal.butter(2, [lo, hi], btype="band", fs=sample_rate, output="sos")
                    band = signal.sosfilt(sos, sig)
                    rms = float(np.sqrt(np.mean(band**2) + 1e-12))
                    return float(20.0 * np.log10(rms + 1e-12))
                except Exception as e:
                    logger.warning(
                        "Verarbeitungsschritt_42_vocal_enhancement.py::_formant_energy_dbfs Ersatzpfad: %s", e
                    )
                    return -80.0

            def _bell_eq_sos(center_hz: float, gain_db: float, q: float) -> np.ndarray:
                """Baut SOS-Koeffizienten für ein Bell-EQ Biquad (minimum-phase).

                §v10.65: Minimum-phase via lfilter verhindert Pre-Ringing.
                SOS-Array für Composite-Filter-Design — alle Formanten in
                EINEM Filterdurchlauf statt serieller filtfilt-Kaskade.
                """
                if abs(gain_db) < 0.05:
                    return cast(np.ndarray, (np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)))
                w0 = 2.0 * np.pi * center_hz / sample_rate
                sin_w0 = float(np.sin(w0))
                cos_w0 = float(np.cos(w0))
                alpha = sin_w0 / (2.0 * q)
                A = 10.0 ** (gain_db / 40.0)
                b0 = 1.0 + alpha * A
                b1 = -2.0 * cos_w0
                b2 = 1.0 - alpha * A
                a0 = 1.0 + alpha / A
                a1 = -2.0 * cos_w0
                a2 = 1.0 - alpha / A
                return cast(
                    np.ndarray, (np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64))
                )

            # §v10.65: Sammle alle Formant-Korrekturen in einem SOS-Array
            # und wende sie in EINEM lfilter-Durchlauf an — kein serielles
            # filtfilt, kein kumulatives Pre-Ringing, kein Formant-Shift.
            sos_sections: list[np.ndarray] = []
            total_correction_db = 0.0
            n_corrected = 0
            for center_hz, canonical_ceiling_dbfs, max_corr_db, q in _profile:
                measured_dbfs = _formant_energy_dbfs(x_mono, center_hz, q)
                deficit_db = canonical_ceiling_dbfs - measured_dbfs
                if deficit_db <= 0.5:
                    continue
                correction_db = float(np.clip(deficit_db * 0.6, 0.0, max_corr_db))
                if correction_db < 0.2:
                    continue
                sos_sections.append(_bell_eq_sos(center_hz, correction_db, q))
                total_correction_db += correction_db
                n_corrected += 1

            result = audio.copy().astype(np.float32)
            if sos_sections:
                sos = np.vstack(sos_sections)
                if audio.ndim == 2:
                    for ch in range(audio.shape[0]):
                        result[ch] = signal.sosfilt(sos, result[ch].astype(np.float64)).astype(np.float32)
                else:
                    result = signal.sosfilt(sos, result.astype(np.float64)).astype(np.float32)

            if n_corrected > 0:
                logger.debug(
                    "§Hebel-4 carrier_formant_decay_inversion: material=%s, formants=%d, Σgain=+%.2f dB",
                    _mat,
                    n_corrected,
                    total_correction_db,
                )

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            result = np.clip(result, -1.0, 1.0)
            return result.astype(audio.dtype)  # type: ignore[no-any-return]

        except Exception as _cfd_exc:
            logger.debug("§Hebel-4 _wiederherstellen_carrier_formant_decay fehlgeschlagen (ignoriert): %s", _cfd_exc)
            return audio

    @staticmethod
    def _compute_formant_recovery_guard_profile(
        material_type: "MaterialType",
        quality_mode: str,
        restorability_score: float,
    ) -> dict:
        """§2.54 Adaptive Formant-Recovery-Guard-Profile.

        Liefert Mindest-Schwellwerte für Formant-Korrekturen in _restore_carrier_formant_decay.
        Niedrigere Werte = permissiver (mehr Korrekturen erlaubt) = geeignet für stark
        degradiertes Material oder niedrige Restorability.

        Args:
            material_type: Trägermedium (shellac, vinyl, tape …).
            quality_mode: "fast" | "balanced" | "quality" | "maximum".
            restorability_score: 0–100; niedrig = stark degradiert.

        Returns:
            dict mit headroom_min_db, correction_min_db, eq_min_gain_db.
        """
        # Material-Baseline: härteres analoges Material → permissiver (niedrigere Schwellen)
        _hard_analog = {
            "shellac",
            "wax_cylinder",
            "wire_recording",
            "lacquer_disc",
            "acoustic_78",
        }
        _mid_analog = {"vinyl", "reel_tape", "tape"}
        _soft_analog = {"cassette", "8_track", "minidisc"}

        mat_key = material_type.value if hasattr(material_type, "value") else str(material_type)

        if mat_key in _hard_analog:
            headroom_base = 0.30
            correction_base = 0.12
            eq_base = 0.03
        elif mat_key in _mid_analog:
            headroom_base = 0.45
            correction_base = 0.18
            eq_base = 0.05
        elif mat_key in _soft_analog:
            headroom_base = 0.55
            correction_base = 0.22
            eq_base = 0.06
        else:  # digital / unknown
            headroom_base = 0.65
            correction_base = 0.26
            eq_base = 0.08

        # Restorability: niedrige Restorability → permissiver (niedrigere Schwellen)
        # Score 10 → -0.15; Score 90 → +0.05 (relativ zur Base)
        rest_norm = max(0.0, min(100.0, float(restorability_score or 50.0))) / 100.0
        rest_delta = (rest_norm - 0.5) * 0.20  # [-0.10, +0.10]
        headroom = headroom_base + rest_delta
        correction = correction_base + rest_delta * 0.6
        eq = eq_base + rest_delta * 0.04

        # Quality-mode: fast → konservativer (höhere Schwellen); maximum → aggressiver
        _mode_offsets = {
            "fast": +0.12,
            "balanced": +0.04,
            "quality": 0.0,
            "maximum": -0.06,
        }
        mode_key = str(quality_mode or "balanced").strip().lower()
        # Alias-Mapping (§2.31 quality_mode aliases)
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        mode_key = _aliases.get(mode_key, mode_key)
        offset = _mode_offsets.get(mode_key, 0.0)
        headroom += offset
        correction += offset * 0.6
        eq += offset * 0.04

        # Hard bounds
        headroom = float(np.clip(headroom, 0.25, 0.80))
        correction = float(np.clip(correction, 0.10, 0.35))
        eq = float(np.clip(eq, 0.02, 0.10))

        return {
            "headroom_min_db": headroom,
            "correction_min_db": correction,
            "eq_min_gain_db": eq,
        }

    def _reduce_harshness(self, audio: np.ndarray, sample_rate: int, severity: float) -> np.ndarray:
        """Reduce vocal harshness via dynamic presence-band attenuation (2–6 kHz).

        Algorithm:
        1. Extract 2–6 kHz presence band via bandpass filter
        2. Compute RMS envelope of the presence band
        3. Apply dynamic gain reduction only where presence energy exceeds threshold
        4. Smooth the gain curve to avoid artifacts
        5. Re-combine with attenuated presence band

        This is NOT a static notch — it preserves quiet vocal presence while
        taming harsh peaks (similar to a multiband compressor targeting 2–6 kHz).

        Strength is proportional to DefectScanner severity:
        - severity 0.1–0.3: gentle 2–4 dB reduction on peaks
        - severity 0.3–0.6: moderate 4–8 dB reduction
        - severity 0.6–1.0: aggressive 8–12 dB reduction
        """
        n = len(audio)
        if n < 512:
            return audio

        # Extract presence band (2–6 kHz)
        # §2.51 Anti-Zeitversatz: sosfiltfilt — Band wird mit (presence_reduced - presence)
        # auf Original aufaddiert; sosfilt erzeugt Zeitversatz → Kammfilter-Artefakt.
        sos_bp = signal.butter(4, [2000.0, 6000.0], btype="band", fs=sample_rate, output="sos")
        presence = signal.sosfiltfilt(sos_bp, audio)

        # Compute RMS envelope (5 ms smoothing)
        frame_len = max(1, int(0.005 * sample_rate))
        envelope = np.sqrt(np.convolve(presence**2, np.ones(frame_len) / frame_len, mode="same") + 1e-12)

        # Dynamic threshold: based on median presence energy (preserves normal levels)
        median_env = float(np.median(envelope) + 1e-12)
        # Threshold above which we reduce (1.5× median for gentle, 1.2× for aggressive)
        threshold_factor = 1.5 - 0.3 * min(severity, 1.0)  # 1.5 → 1.2
        threshold = median_env * threshold_factor

        # Maximum gain reduction in dB (severity-scaled)
        max_reduction_db = 2.0 + 10.0 * min(severity, 1.0)  # 2–12 dB range

        # Global broadband harshness: if presence band energy is uniformly high,
        # apply a baseline attenuation (the dynamic approach alone misses
        # signals that are uniformly harsh since median ≈ peaks).
        overall_rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
        presence_rms = float(np.sqrt(np.mean(presence**2)) + 1e-12)
        # If presence is > 35% of total energy, apply global attenuation
        pres_ratio = presence_rms / overall_rms
        if pres_ratio > 0.35:
            # Scale: at ratio=0.35 → 0 dB; at ratio=0.8 → up to max_reduction_db/2
            global_atten_db = min(severity, 1.0) * (pres_ratio - 0.35) / 0.45 * (max_reduction_db * 0.5)
            global_atten_db = min(global_atten_db, max_reduction_db * 0.5)
        else:
            global_atten_db = 0.0

        # Compute gain curve
        envelope_db = 20.0 * np.log10(envelope / threshold + 1e-12)
        # Gain reduction only above threshold (soft-knee)
        gain_db = np.where(
            envelope_db > 0.0,
            -np.minimum(envelope_db * 0.6, max_reduction_db),
            0.0,
        )
        # Apply global attenuation floor
        if global_atten_db > 0.1:
            gain_db = np.minimum(gain_db, -global_atten_db)

        # Smooth gain to prevent clicks (2 ms attack, 20 ms release)
        attack_samples = max(1, int(0.002 * sample_rate))
        release_samples = max(1, int(0.020 * sample_rate))
        smoothed = np.zeros_like(gain_db)
        smoothed[0] = gain_db[0]
        for i in range(1, len(gain_db)):
            if gain_db[i] < smoothed[i - 1]:
                alpha = 1.0 - np.exp(-1.0 / attack_samples)
            else:
                alpha = 1.0 - np.exp(-1.0 / release_samples)
            smoothed[i] = smoothed[i - 1] + alpha * (gain_db[i] - smoothed[i - 1])

        gain_linear = 10.0 ** (smoothed / 20.0)

        # Apply gain only to the presence band and re-combine
        presence_reduced = presence * gain_linear
        result = audio + (presence_reduced - presence)

        actual_reduction = float(-np.mean(smoothed[smoothed < -0.1])) if np.any(smoothed < -0.1) else 0.0
        logger.info(
            "Verarbeitungsschritt42 harshness reduction: severity=%.2f max_reduction=%.1fdB actual_mean=%.1fdB",
            severity,
            max_reduction_db,
            actual_reduction,
        )

        return result  # type: ignore[no-any-return]

    @staticmethod
    def _prefer_demucs_native(material: object) -> bool:
        """Aktiviert nativen HTDemucs nur fuer live/crowd-nahe Quellen."""
        return prefer_demucs_native_from_material(material)  # type: ignore[no-any-return]

    def _apply_deessing(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Wendet De-Essing auf das Sibilanz-Band an."""
        # Extract sibilance band
        # §2.51 Anti-Zeitversatz: sosfiltfilt — Band wird mit (sibilance_reduced - sibilance)
        # auf Original aufaddiert; sosfilt erzeugt Zeitversatz → Kammfilter-Artefakt.
        sos = signal.butter(4, self.VOCAL_BANDS["sibilance"], btype="band", fs=sample_rate, output="sos")
        sibilance = signal.sosfiltfilt(sos, audio)

        # Dynamic range compression on sibilance
        analytic = np.asarray(signal.hilbert(sibilance))
        envelope = np.abs(analytic)
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Apply reduction above threshold
        gain_db = np.where(
            envelope_db > config["deess_threshold_db"],
            -config["deess_reduction_db"] * ((envelope_db - config["deess_threshold_db"]) / 20),
            0,
        )

        # Smooth gain
        gain_db_smooth = signal.savgol_filter(gain_db, window_length=min(101, len(gain_db) // 10 * 2 + 1), polyorder=3)
        gain_linear = 10 ** (gain_db_smooth / 20)

        # Apply to sibilance and subtract from original
        sibilance_reduced = sibilance * gain_linear
        deessed = audio + (sibilance_reduced - sibilance) * 0.7

        return deessed  # type: ignore[no-any-return]

    def _enhance_formants(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: dict[str, Any],
        vocal_gender: str = "unknown",
    ) -> np.ndarray:
        """Enhance formant region using LPC FormantSystem with Singer's Formant Enhancement.

        Processing chain (§2.8):
            1. FormantSystem.process() — LPC tracking, drift correction,
               Singer's Formant (2.5–3.5 kHz).  Primary path.
            2. PhonemeDetector.detect() — DSP-basierte Phonem-Segmentierung
               (V/C/sib/sil).  Liefert voiced-Segment-Timestamps.
            3. FormantSystem.phoneme_guided_enhance() — per-vowel canonical
               target steering (Peterson & Barney 1952, Hillenbrand 1995)
               mit echten Phonem-Segmenten + Pipeline-Gender.
            4. Bell EQ @ 1.5 kHz — DSP fallback when FormantSystem unavailable.
        """
        # Primary: LPC-based formant tracking + Singer's Formant Enhancement
        if self._formant_system is not None:
            try:
                enhanced, _ = self._formant_system.process(audio, sample_rate)
                enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
                enhanced = np.clip(enhanced, -1.0, 1.0)

                # §2.8: PhonemeDetector für segmentgenaue Formant-Steuerung
                _phoneme_segments = None
                if _PHONEME_DETECTOR_AVAILABLE and _get_phoneme_detector is not None:
                    try:
                        _pd = _get_phoneme_detector()
                        _pd_result = _pd.detect(enhanced, sample_rate)
                        if _pd_result.phonemes:
                            _phoneme_segments = list(zip(_pd_result.phonemes, _pd_result.timestamps_ms))
                            logger.debug(
                                "Verarbeitungsschritt42 PhonemeDetector: %d Segmente (confidence=%.2f)",
                                len(_pd_result.phonemes),
                                _pd_result.confidence,
                            )
                    except Exception as _pd_err:
                        logger.debug("PhonemeDetector fehlgeschlagen (F1/F2-Ersatzpfad): %s", _pd_err)

                # Stage 2: phoneme-guided per-vowel formant steering
                # Uses real phoneme segments + pipeline-detected gender
                # Adaptive correction strength: higher when gender is confidently known
                _correction_strength = 0.25  # conservative default
                if vocal_gender in ("male", "female", "child"):
                    _correction_strength = 0.45  # confident gender → stronger correction
                try:
                    _pre_formant = enhanced.copy()
                    enhanced, _pg_report = self._formant_system.phoneme_guided_enhance(
                        enhanced,
                        sample_rate,
                        phoneme_segments=_phoneme_segments,
                        gender=vocal_gender,
                        correction_strength=_correction_strength,
                    )
                    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
                    enhanced = np.clip(enhanced, -1.0, 1.0)
                    logger.debug(
                        "Verarbeitungsschritt42 phoneme_guided_verbessern: vowel_frames=%d/%d gender=%s",
                        _pg_report.get("vowel_segments_processed", 0),
                        _pg_report.get("total_frames", 0),
                        vocal_gender,
                    )

                    # §2.36 LyricsGuided → Formant-Steering:
                    # Whisper-based phoneme timeline gates formant correction to
                    # confirmed vocal regions with stress-adaptive weighting.
                    # Stressed vowels: full correction (1.0), unstressed: 0.6,
                    # fricatives: 0.3, silence/plosive: 0.0 → prevents coloring
                    # non-vocal regions and focuses correction on perceptually
                    # salient vowel segments.
                    if _LYRICS_GUIDED_AVAILABLE and _get_lyrics_guided is not None:
                        try:
                            _lge = _get_lyrics_guided()
                            _lge_result = _lge.transcribe(enhanced, sample_rate)
                            if _lge_result.words and _lge_result.overall_confidence > 0.3:
                                _n = len(enhanced)
                                _formant_weight = np.zeros(_n, dtype=np.float32)
                                _STRESS_WEIGHTS = {
                                    "vowel_stressed": 1.0,
                                    "vowel_unstressed": 0.6,
                                    "fricative_stressed": 0.3,
                                    "fricative_unstressed": 0.2,
                                    "plosive": 0.1,
                                    "silence": 0.0,
                                }
                                for _w in _lge_result.words:
                                    _ws = max(0, int(_w.start_s * sample_rate))
                                    _we = min(_n, int(_w.end_s * sample_rate))
                                    if _ws < _we:
                                        _wt = _STRESS_WEIGHTS.get(_w.phoneme_type, 0.3)
                                        # Scale by word confidence
                                        _wt *= min(1.0, _w.confidence)
                                        np.maximum(_formant_weight[_ws:_we], _wt, out=_formant_weight[_ws:_we])
                                # Blend: voiced regions get formant correction,
                                # non-vocal regions keep pre-formant audio
                                _diff = enhanced - _pre_formant
                                enhanced = _pre_formant + _formant_weight * _diff
                                enhanced = np.clip(enhanced, -1.0, 1.0)
                                logger.debug(
                                    "Verarbeitungsschritt42 LyricsGuided formant-steering: %d words, "
                                    "confidence=%.2f, mean_weight=%.2f",
                                    len(_lge_result.words),
                                    _lge_result.overall_confidence,
                                    float(np.mean(_formant_weight)),
                                )
                        except Exception as _lge_err:
                            logger.debug(
                                "Verarbeitungsschritt42 LyricsGuided formant-steering fehlgeschlagen (ignoriert): %s",
                                _lge_err,
                            )
                except Exception as _pg_err:
                    logger.debug("phoneme_guided_verbessern fehlgeschlagen (ignoriert): %s", _pg_err)

                return enhanced.astype(audio.dtype)  # type: ignore[no-any-return]
            except Exception as _fs_err:
                logger.debug("FormantSystem fehlgeschlagen, Bell-EQ-Ersatzpfad: %s", _fs_err)

        # DSP-Fallback: Multi-Formant Bell-EQ chain (v10.0.0)
        # Replaces single 1.5 kHz bell with 4-band formant chain derived from
        # Peterson & Barney 1952 (F1–F3) + Sundberg 1974 (Singer's Formant).
        # Each band lifts its formant region proportionally to gain_db:
        #   F1  500 Hz  — low vowel clarity (open/mid vowels)
        #   F2 1500 Hz  — mid vowel intelligibility (front/back vowel contrast)
        #   F3 2500 Hz  — consonant definition, speech clarity
        #   SF 3200 Hz  — Singer's Formant (vocal projection, presence)
        gain_db = config["formant_gain_db"]
        _FORMANT_BANDS = [
            # (center_hz, gain_fraction, Q)
            (500, 0.50, 3.0),  # F1
            (1500, 0.80, 2.0),  # F2 — dominant mid-vowel formant
            (2500, 0.35, 2.5),  # F3 — speech clarity
            (3200, 0.20, 3.5),  # Singer's Formant — vocal projection
        ]
        enhanced = audio.copy()
        for _f0_hz, _gain_frac, _q in _FORMANT_BANDS:
            _gain_band = gain_db * _gain_frac
            if abs(_gain_band) < 0.15:
                continue
            _w0 = 2.0 * np.pi * _f0_hz / sample_rate
            _sin_w0 = np.sin(_w0)
            _cos_w0 = np.cos(_w0)
            _alpha_f = _sin_w0 / (2.0 * _q)
            _A = 10.0 ** (_gain_band / 40.0)
            _b0 = 1.0 + _alpha_f * _A
            _b1 = -2.0 * _cos_w0
            _b2 = 1.0 - _alpha_f * _A
            _a0 = 1.0 + _alpha_f / _A
            _a1 = -2.0 * _cos_w0
            _a2 = 1.0 - _alpha_f / _A
            _b = np.array([_b0, _b1, _b2]) / _a0
            _a = np.array([1.0, _a1 / _a0, _a2 / _a0])
            # §2.51 Zero-phase Bell-EQ: filtfilt prevents group-delay pre-smear on vocal
            # plosives/onsets (lfilter VERBOTEN per VERBOTEN-list). For filtfilt the
            # effective dB gain doubles (forward+backward pass), so design at gain/2.
            # Short-signal fallback (< 9 samples) uses lfilter for numerical safety.
            if len(enhanced) >= 9:
                # Design at half gain so filtfilt's forward+backward pass yields intended gain
                _A_half = 10.0 ** ((_gain_band / 2.0) / 40.0)
                _b0h = 1.0 + _alpha_f * _A_half
                _b2h = 1.0 - _alpha_f * _A_half
                _a0h = 1.0 + _alpha_f / _A_half
                _a2h = 1.0 - _alpha_f / _A_half
                _bh = np.array([_b0h, _b1, _b2h]) / _a0h
                _ah = np.array([1.0, _a1 / _a0h, _a2h / _a0h])
                enhanced = safe_filtfilt(_bh, _ah, enhanced)
            else:
                enhanced = signal.lfilter(_b, _a, enhanced)
        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced = np.clip(enhanced, -1.0, 1.0)
        return enhanced.astype(audio.dtype)  # type: ignore[no-any-return]

    def _boost_presence(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Verstärkt presence region with ISO 226 loudness compensation.

        Improvements (§8.3 Psychoacoustics):
        - Loudness-adaptive gain: quieter signals get more boost (ISO 226 equal-loudness)
        - Narrower Q at 4.5 kHz aligned to Zwicker critical bandwidth (~520 Hz → Q ≈ 8)
        - Vibrato detection: reduce presence gain during active vibrato to protect F0 modulation
        """
        n = len(audio)
        if n < 512:
            return audio

        base_gain_db = config["presence_gain_db"]
        center_freq = 4500.0

        # --- ISO 226 Loudness Compensation ---
        # At low loudness, equal-loudness contours show higher sensitivity in 2-5 kHz
        # → we need less boost. At high loudness, the sensitivity flattens → more boost needed.
        # Simplified ISO 226 correction: measure RMS level, adapt gain.
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        rms_db = 20.0 * np.log10(rms + 1e-10)
        # Reference: -14 dBFS (EBU R128 nominal). Below → reduce gain, above → keep.
        # ISO 226 at 4 kHz: ~10 dB loudness-dependent sensitivity shift across 40-80 phon
        loudness_compensation = np.clip((rms_db + 14.0) * 0.04, -0.3, 0.2)
        adapted_gain_db = base_gain_db * (1.0 + loudness_compensation)

        # --- Vibrato detection: protect F0 modulation ---
        # Measure spectral flux in 80-400 Hz (vocal F0 range) at 4-8 Hz modulation rate
        vibrato_attenuation = 1.0
        try:
            # Extract F0 band
            sos_f0 = signal.butter(3, [80.0, 400.0], btype="band", fs=sample_rate, output="sos")
            f0_band = signal.sosfilt(sos_f0, audio)
            # Compute amplitude envelope
            f0_band_1d = np.asarray(f0_band, dtype=np.float64).reshape(-1)
            n_f0 = f0_band_1d.shape[0]
            f0_spectrum = np.fft.fft(f0_band_1d)
            h = np.zeros(n_f0, dtype=np.float64)
            if n_f0 % 2 == 0:
                h[0] = 1.0
                h[n_f0 // 2] = 1.0
                h[1 : n_f0 // 2] = 2.0
            else:
                h[0] = 1.0
                h[1 : (n_f0 + 1) // 2] = 2.0
            analytic = np.abs(np.fft.ifft(f0_spectrum * h))
            # Look for 4-8 Hz modulation (vibrato rate)
            if len(analytic) > sample_rate // 2:
                # Downsample envelope to ~100 Hz for modulation analysis
                ds_factor = max(1, sample_rate // 100)
                env_ds = analytic[::ds_factor]
                if len(env_ds) > 32:
                    env_ds = env_ds - np.mean(env_ds)
                    fft_env = np.abs(np.fft.rfft(env_ds))
                    freqs = np.fft.rfftfreq(len(env_ds), d=ds_factor / sample_rate)
                    # Vibrato energy in 4-8 Hz range
                    vibrato_mask = (freqs >= 4.0) & (freqs <= 8.0)
                    total_mask = freqs > 0.5
                    if np.any(vibrato_mask) and np.any(total_mask):
                        vibrato_energy = float(np.sum(fft_env[vibrato_mask] ** 2))
                        total_energy = float(np.sum(fft_env[total_mask] ** 2)) + 1e-12
                        vibrato_ratio = vibrato_energy / total_energy
                        # If vibrato dominates (>15% of modulation energy), attenuate presence
                        if vibrato_ratio > 0.15:
                            vibrato_attenuation = max(0.5, 1.0 - vibrato_ratio)
                            logger.debug(
                                "Verarbeitungsschritt42 vibrato erkannt: Verhaeltnis=%.2f → presence attenuation=%.2f",
                                vibrato_ratio,
                                vibrato_attenuation,
                            )
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)  # Vibrato detection failure is non-critical

        final_gain_db = adapted_gain_db * vibrato_attenuation
        if abs(final_gain_db) < 0.01:
            return audio

        # Psychoacoustically-aligned Q: critical bandwidth at 4.5 kHz ≈ 520 Hz
        # Q = center_freq / bandwidth → 4500 / 520 ≈ 8.6 (narrower, perceptually correct)
        # Compromise: Q=5.0 (still much tighter than old Q=1.5, but avoids ringing)
        q = 5.0
        w0 = 2 * np.pi * center_freq / sample_rate
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (final_gain_db / 40)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Zero-phase filtering prevents phase shift on vocal transients (plosives, vowel onsets).
        # filtfilt needs at least 3*max(len(a),len(b)) = 9 samples; fall back to lfilter for tiny buffers.
        if len(audio) >= 9:
            enhanced = safe_filtfilt(b, a, audio)
        else:
            enhanced = signal.lfilter(b, a, audio)
        return enhanced  # type: ignore[no-any-return]

    def _enhance_chest(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Enhance chest resonance."""
        # Bell filter @ 175 Hz
        w0 = 2 * np.pi * 175 / sample_rate
        gain_db = config["chest_gain_db"]
        q = 1.0
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Zero-phase Bell-EQ — prevents group delay on chest resonance fundamentals.
        if len(audio) >= 9:
            enhanced = safe_filtfilt(b, a, audio)
        else:
            enhanced = signal.lfilter(b, a, audio)
        return enhanced  # type: ignore[no-any-return]

    def _control_breath(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Reduce breath noise — segment-aware via BreathDetector (§2.8).

        Primary path: BreathDetector identifies breath segments via ZCR + energy,
        then applies targeted reduction only in those segments (preserving
        non-breath HF content like cymbals, harmonics).

        Fallback: Static 8-12 kHz bandpass reduction (legacy behavior).
        """
        n = len(audio)
        if n < 1024:
            return audio

        # §2.8 Primary: Segment-aware breath reduction via BreathDetector
        if _BREATH_DETECTOR_AVAILABLE and _get_breath_detector is not None:
            try:
                _bd = _get_breath_detector()
                _bd_result = _bd.detect(audio.astype(np.float32), sample_rate)
                if _bd_result.breath_positions:
                    controlled = audio.copy()
                    reduction_db = config["breath_reduction_db"]
                    reduction_linear = 10 ** (-reduction_db / 20)
                    # Crossfade window: 5 ms Hanning (§2.8 BreathDetector spec)
                    _xfade_samples = max(1, int(0.005 * sample_rate))
                    _half_xfade = np.hanning(_xfade_samples * 2)

                    for _start, _end in zip(_bd_result.breath_positions, _bd_result.breath_end_positions):
                        _s = max(0, int(_start))
                        _e = min(n, int(_end))
                        if _e <= _s:
                            continue
                        seg_len = _e - _s
                        # Build gain envelope: full reduction in center, crossfade at edges
                        gain = np.ones(seg_len, dtype=np.float64) * reduction_linear
                        # Fade-in at start
                        _fi = min(_xfade_samples, seg_len // 2)
                        if _fi > 0:
                            gain[:_fi] = (
                                1.0 - (1.0 - reduction_linear) * _half_xfade[_xfade_samples - _fi : _xfade_samples]
                            )
                        # Fade-out at end
                        _fo = min(_xfade_samples, seg_len // 2)
                        if _fo > 0:
                            gain[-_fo:] = (
                                1.0 - (1.0 - reduction_linear) * _half_xfade[_xfade_samples : _xfade_samples + _fo]
                            )
                        controlled[_s:_e] *= gain
                    logger.debug(
                        "Verarbeitungsschritt42 BreathDetector: %d Segmente reduziert (%.1f dB), confidence=%.2f",
                        len(_bd_result.breath_positions),
                        reduction_db,
                        _bd_result.confidence,
                    )
                    return controlled
                # No breath segments found — return unchanged
                return audio
            except Exception as _bd_err:
                logger.debug("BreathDetector fehlgeschlagen, Bandpass-Ersatzpfad: %s", _bd_err)

        # DSP-Fallback: Static 8-12 kHz bandpass reduction
        # §2.51 Anti-Zeitversatz: sosfiltfilt — Band wird mit (breath_reduced - breath)
        # auf Original aufaddiert; sosfilt erzeugt Zeitversatz → Kammfilter-Artefakt.
        sos = signal.butter(4, self.VOCAL_BANDS["breath"], btype="band", fs=sample_rate, output="sos")
        breath = signal.sosfiltfilt(sos, audio)
        reduction_linear = 10 ** (-config["breath_reduction_db"] / 20)
        breath_reduced = breath * reduction_linear
        controlled = audio + (breath_reduced - breath) * 0.6
        return controlled  # type: ignore[no-any-return]

    def _apply_compression(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Wendet an: psychoacoustically-optimized vocal micro-compression.

        Improvements over naive RMS compression (§8.3 Micro-Dynamics):
        - Separate attack (3 ms) and release (120 ms) for vocal syllabic preservation
        - Soft-knee (6 dB) to avoid hard compression artifacts
        - Loudness-adaptive makeup gain (recovers 80% of compression depth)
        - Exponential envelope follower (not polynomial smoothing)
        """
        n = len(audio)
        if n < 512:
            return audio

        # Envelope follower with vocal-optimized attack/release
        attack_s = 0.003  # 3 ms — fast enough for consonant transients
        release_s = 0.120  # 120 ms — slow enough to preserve vowel sustain
        attack_coeff = 1.0 - np.exp(-1.0 / (attack_s * sample_rate))
        release_coeff = 1.0 - np.exp(-1.0 / (release_s * sample_rate))

        envelope = np.zeros(n, dtype=np.float64)
        abs_audio = np.abs(audio)
        envelope[0] = abs_audio[0]
        for i in range(1, n):
            if abs_audio[i] > envelope[i - 1]:
                envelope[i] = envelope[i - 1] + attack_coeff * (abs_audio[i] - envelope[i - 1])
            else:
                envelope[i] = envelope[i - 1] + release_coeff * (abs_audio[i] - envelope[i - 1])

        env_db = 20.0 * np.log10(envelope + 1e-10)

        threshold_db = -15.0
        ratio = config["compression_ratio"]
        knee_db = 6.0  # Soft knee width

        # Soft-knee gain computation (avoids hard compression onset)
        gain_db = np.zeros(n, dtype=np.float64)
        half_knee = knee_db / 2.0
        for i in range(n):
            over = env_db[i] - threshold_db
            if over <= -half_knee:
                gain_db[i] = 0.0
            elif over >= half_knee:
                gain_db[i] = -over * (1.0 - 1.0 / ratio)
            else:
                # Quadratic soft-knee transition
                gain_db[i] = -((over + half_knee) ** 2) / (4.0 * knee_db) * (1.0 - 1.0 / ratio)

        gain_linear = 10.0 ** (gain_db / 20.0)
        compressed = audio * gain_linear

        # Loudness-adaptive makeup gain: recover 80% of mean compression depth
        active_reduction = gain_db[gain_db < -0.1]
        if len(active_reduction) > 0:
            mean_reduction_db = float(np.mean(active_reduction))
            makeup_db = -mean_reduction_db * 0.8
            makeup_linear = 10.0 ** (makeup_db / 20.0)
        else:
            makeup_linear = 1.0
        # §2.45a-II: apply makeup gain only to musical frames (not silence) to prevent
        # amplification of surface noise in fade-out / silent sections.
        if makeup_linear > 1.0005:
            from backend.core.audio_utils import apply_musical_gain_envelope  # pylint: disable=import-outside-toplevel

            # V04-Fix: reference_for_gate MUSS den Pre-Phase-Input (audio) nutzen, nicht
            # das komprimierte Intermediat — sonst kann Makeup-Gain in Stille-Zonen
            # durchgreifen, deren Pegel nur durch Kompression unter den Gate-Schwell
            # gedrückt wurden, im Original aber höher lagen.
            compressed = apply_musical_gain_envelope(
                compressed,
                makeup_linear,
                gate_dbfs=-36.0,
                crossfade_ms=10.0,
                sr=sample_rate,
                reference_for_gate=audio,
            )
        else:
            compressed = compressed * makeup_linear

        return compressed  # type: ignore[no-any-return]

    def _apply_vocal_stem_mdem(
        self,
        enhanced_vocals: np.ndarray,
        original_vocals: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Wendet an: MDEM (Micro-Dynamics Envelope Morphing) on the VOCAL STEM.

        §8.3 Psychoacoustic fix: MDEM in UV3 operates on full mix where
        instrumental energy dominates the LUFS profile.  By applying MDEM
        on the isolated vocal stem BEFORE remix, the vocal micro-dynamics
        (syllabic articulation, breath rhythm, emotional swells) are
        recovered from the original vocal stem's LUFS profile.

        This is a lightweight LUFS-morphing pass (400 ms window) that
        corrects gain per frame so that the enhanced vocal stem's loudness
        contour matches the original vocal stem.

        Falls back gracefully if MDEM module isn't available.
        """
        try:
            from backend.core.micro_dynamics_envelope_morphing import get_mdem  # pylint: disable=import-outside-toplevel  # noqa: I001

            _mdem = get_mdem()
            # Ensure matching lengths
            n = min(enhanced_vocals.shape[0], original_vocals.shape[0])
            enh = enhanced_vocals[:n]
            orig = original_vocals[:n]
            morphed = _mdem.morph(enh, orig, sample_rate, mode="restoration")
            morphed = np.nan_to_num(morphed, nan=0.0, posinf=0.0, neginf=0.0)
            morphed = np.clip(morphed, -1.0, 1.0)
            # Ensure output matches original shape
            if morphed.shape[0] < enhanced_vocals.shape[0]:
                out = enhanced_vocals.copy()
                out[: morphed.shape[0]] = morphed
                return out
            logger.debug("Verarbeitungsschritt42 Vocal-Stem MDEM: micro-dynamics recovered on vocal stem")
            return morphed  # type: ignore[no-any-return]
        except Exception as _mdem_err:
            logger.debug("Vocal-Stem MDEM nicht verfügbar (ignoriert): %s", _mdem_err)
            return enhanced_vocals
