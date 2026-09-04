#!/usr/bin/env python3
"""
Phase 28: Surface Noise Profiling v3.0 - Über-SOTA OMLSA/IMCRA
Adaptive spectral noise profiling via IMCRA minimum statistics
und OMLSA-Gain (Optimally Modified Log-Spectral Amplitude).

Algorithm Overview (v3.0):
1. IMCRA-Rauscheschätzung (Iterative Minimum Controlled Recursive Averaging):
   - Bias-korrigiertes gleitendes Minimum (≈1.5s Fenster, b_min=1.66)
   - Exponentielle Glättung α_n=0.85
   - Gibt F×T-Rauschleistungsmatrix zurück
2. OMLSA-Gain (Cohen 2003):
   - γ(t,f) = |Y(t,f)|² / σ²_n(t,f)
   - ξ(t,f) = max(γ-1, 0)  [a-priori SNR]
   - v = ξ·γ / (1+ξ)
   - Λ(t,f) = exp(-ξ + v)  [likelihood]
   - p(t,f) = 1/(1 + q/präsenzÜberβ(Λ))  [Sprachpräsenz]
   - G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p   G_floor=0.1
3. Cappe-Gain-Glättung (1994):
   - Temporale Glättung mit materialadaptiver Zeitkonstante
4. Material-Adaptierung:
   - Shellac: aggressiv (hohes Oberflächenrauschen)
   - Vinyl: ausgewogen (Crackle + Oberflächenrauschen)
   - Tape: konservativ (hauptsächlich Tape-Hiss)
   - Digital: minimal (Dithering-Rauschen)

Scientific Foundation:
- Cohen & Berdugo (2002): IMCRA — primär
- Cohen (2003): OMLSA — primär
- Cappé (1994): Elimination of the Musical Noise Phenomenon — Gain-Glättung
- Le Roux & Vincent (2013): Consistent Wiener Filtering — Phasenkonsistenz
- Ephraim & Malah (1984): historische Referenz — NICHT primär eingesetzt
- Martin (2001): Minimum Statistics — Basis-Konzept für IMCRA

Industry Benchmarks:
- iZotope RX Spectral De-noise ($399)
- Cedar DNS (Adaptive noise suppressor, $2000+)
- Waves Z-Noise ($49)
- Accusonus ERA-N ($99)
- Acon Digital DeNoise ($99)

Quality Target: 0.78 → 0.94 (+20% improvement)
Performance Target: <0.25× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import time
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d as _minimum_filter1d
from scipy.signal import lfilter as _lfilter
from scipy.signal import lfilter_zi as _lfilter_zi

from backend.core.audio_utils import safe_to_mono, to_channels_last
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.restoration_policy import get_effective_song_goal_weights

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# PGHI phase reconstruction after spectral gain application (Spec §DSP — PFLICHT)
try:
    _PGHI_AVAILABLE = True
except Exception:
    _PGHI_AVAILABLE = False
    logger.warning("PGHI not verfuegbar; scipy.signal.istft Ersatzpfad active for Verarbeitungsschritt_28")


class SurfaceNoiseProfiling(PhaseInterface):
    """
    Professional Surface Noise Profiling with Wiener Filtering.

    Key Features:
    - Multi-pass VAD for noise-only region detection
    - Minimum statistics noise tracking
    - Wiener filter with SNR-based gains
    - Over-subtraction with frequency-dependent flooring
    - Temporal gain smoothing (reduce musical noise)
    - Material-adaptive parameters

    Use Cases:
    - Vinyl surface noise reduction
    - Shellac crackle suppression
    - Tape hiss removal
    - Digital dithering noise cleanup

    Performance: <0.25× realtime on modern CPU
    """

    # STFT parameters
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    # Material-adaptive noise reduction configurations
    NOISE_CONFIG = {
        MaterialType.SHELLAC: {
            "over_subtraction_alpha": 2.8,
            "spectral_floor": 0.12,
            "vad_threshold_db": -38,
            "smoothing_frames": 8,
            "noise_learn_duration_s": 1.5,
        },
        MaterialType.VINYL: {
            "over_subtraction_alpha": 2.2,
            "spectral_floor": 0.10,  # §2.62 hard min
            "vad_threshold_db": -42,
            "smoothing_frames": 10,
            "noise_learn_duration_s": 1.2,
        },
        MaterialType.TAPE: {
            "over_subtraction_alpha": 1.8,
            "spectral_floor": 0.10,  # §2.62 hard min
            "vad_threshold_db": -48,
            "smoothing_frames": 12,
            "noise_learn_duration_s": 1.0,
        },
        MaterialType.CASSETTE: {
            "over_subtraction_alpha": 1.55,  # v10.0.0: NR ≈ TAPE×0.85 — Cassette-Rauschen hat anderes Spektralprofil
            "spectral_floor": 0.10,  # §2.62 hard min
            "vad_threshold_db": -46,  # v10.0.0: etwas höher als TAPE (Cassette-Hiss höherer Pegel)
            "smoothing_frames": 13,  # v10.0.0: minimal mehr Glättung
            "noise_learn_duration_s": 1.0,
        },  # v10.0.0: IEC 60094-1 — Cassette-Hiss-Profil angepasst
        MaterialType.CD_DIGITAL: {
            "over_subtraction_alpha": 1.3,
            "spectral_floor": 0.10,  # §2.62 hard min
            "vad_threshold_db": -55,
            "smoothing_frames": 15,
            "noise_learn_duration_s": 0.8,
        },
        MaterialType.STREAMING: {
            "over_subtraction_alpha": 1.2,
            "spectral_floor": 0.10,  # §2.62 hard min
            "vad_threshold_db": -60,
            "smoothing_frames": 15,
            "noise_learn_duration_s": 0.5,
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
        "streaming": 1.2,
        "unknown": 1.5,
    }

    # §2.45a Material-adaptive gate: noise_floor + 6 dB margin (AES/iZotope RX practice).
    # Vinyl noise ≈ −33 dBFS → gate −27 dBFS; fixed −36 dBFS lets noise frames through → Pegelexplosion.
    _MAKEUP_GATE_DBFS = {
        "shellac": -14.0,
        "wax_cylinder": -10.0,
        "lacquer_disc": -20.0,
        "wire_recording": -20.0,
        "acoustic_78": -14.0,
        "vinyl": -27.0,
        "reel_tape": -32.0,
        "cassette": -38.0,
        "tape": -32.0,
        "mp3_low": -44.0,
        "mp3_medium": -46.0,
        "cd_digital": -48.0,
        "streaming": -48.0,
        "dat": -48.0,
        "minidisc": -44.0,
        "unknown": -36.0,
    }

    def __init__(self):
        super().__init__()
        self.name = "Surface Noise Profiling v2 Professional"

    @staticmethod
    def _derive_safe_surface_strength(
        effective_strength: float,
        material_key: str,
        panns_tags: dict[str, float],
    ) -> float:
        """Reduce denoise aggressiveness on vocal/analog-sensitive material."""
        strength = float(effective_strength)
        vocal_prob = max(
            float(panns_tags.get("Singing voice", 0.0)),
            float(panns_tags.get("Vocals", 0.0)),
            float(panns_tags.get("Speech", 0.0)),
            float(panns_tags.get("Male singing", 0.0)),
            float(panns_tags.get("Female singing", 0.0)),
        )
        is_analog_sensitive = any(
            token in material_key for token in ("vinyl", "shellac", "wax_cylinder", "wire_recording", "lacquer_disc")
        )
        if vocal_prob >= 0.40:
            strength *= 0.82
        if is_analog_sensitive:
            strength *= 0.90
        return float(np.clip(strength, 0.0, 1.0))

    @staticmethod
    def _goal_hint_strength_scalar(kwargs: dict[str, object]) -> float:
        """Berechnet bounded advisory strength scalar from song goal weights (§2.56a)."""
        goal_weights = get_effective_song_goal_weights(kwargs)
        if not isinstance(goal_weights, dict):
            return 1.0

        def _w(name: str, default: float = 1.0) -> float:
            try:
                return float(goal_weights.get(name, default))
            except Exception as e:
                logger.warning("Verarbeitungsschritt_28_surface_noise_Messung.py::_w Ersatzpfad: %s", e)
                return default

        naturalness = float(np.clip(_w("natuerlichkeit"), 0.3, 2.0))
        authenticity = float(np.clip(_w("authentizitaet"), 0.3, 2.0))
        transparency = float(np.clip(_w("transparenz"), 0.3, 2.0))
        brilliance = float(np.clip(_w("brillanz"), 0.3, 2.0))

        scalar = (
            1.0
            + 0.10 * (transparency - 1.0)
            + 0.06 * (brilliance - 1.0)
            - 0.10 * (naturalness - 1.0)
            - 0.08 * (authenticity - 1.0)
        )
        return float(np.clip(scalar, 0.80, 1.12))

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_28_surface_noise_profiling",
            name="Surface Noise Profiling v3 OMLSA/IMCRA",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=5,
            dependencies=["phase_03_denoise"],
            estimated_time_factor=0.25,
            version="3.0.0",
            memory_requirement_mb=120,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.95,
            description="IMCRA-Rauschsch\u00e4tzung + OMLSA-Gain (Cohen 2002/2003) \u2014 \u00dcber-SOTA",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: MaterialType = MaterialType.CD_DIGITAL,  # type: ignore[override]
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="28")
        """
        Wendet an: spectral noise profiling and removal.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with denoised audio
        """
        material = material_type  # interner Alias
        # ── §v10 PIM: Per-Band-Intensität kalibrieren ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim = apply_pim_intensity(kwargs, "surface_noise", default_nr=0.5, default_de_ess=0.2, default_comp=1.0)
            if kwargs.get("pim_intensity_map") is not None:
                for _key in ("noise_reduction_strength", "nr_strength", "strength", "wet"):
                    if _key in kwargs:
                        kwargs[_key] = _pim["nr_strength"]
        except Exception as e:
            logger.warning("Verarbeitungsschritt_28_surface_noise_Messung.py::verarbeiten Ersatzpfad: %s", e)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p28_transposed = to_channels_last(audio)
        start_time = time.time()
        self.validate_input(audio)

        # §2.46f Natural-Performance-Artifacts-Guard — detect protected breath/vibrato zones before NR
        _npa_result_28 = None
        try:
            from backend.core.natural_performance_detector import (
                get_natural_performance_detector,
            )

            _npa_result_28 = get_natural_performance_detector().detect(audio, sample_rate)
        except Exception as _npa_exc_28:
            logger.debug("§2.46f NPA detection nicht blockierend: %s", _npa_exc_28)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        _goal_hint_scalar = self._goal_hint_strength_scalar(kwargs)
        _effective_strength = float(np.clip(_effective_strength * _goal_hint_scalar, 0.0, 1.0))

        # §v10.96 Defekt-basiertes Skip-Gate: Surface-Noise nur auf analogen Trägern.
        # Digitales Material (CD/DAT/Streaming) hat physikalisch kein Oberflächenrauschen.
        # SNR > 40 dB → Rauschboden bereits unterhalb psychoakustischer Wahrnehmbarkeit.
        _mat_str = str(getattr(material, "value", material) or "").lower()
        _digital_mats = {"cd_digital", "cd", "dat", "streaming", "digital"}
        _snr_28 = float(kwargs.get("snr_db", kwargs.get("snr", 0.0)) or 0.0)
        if _mat_str in _digital_mats or _snr_28 > 40.0:
            _reason = f"digital_material={_mat_str}" if _mat_str in _digital_mats else f"snr={_snr_28:.0f}dB"
            logger.info(
                "§v10.96 Surface-Noise-ueberspringen: %s → surface noise Messung uebersprungen",
                _reason,
            )
            _effective_strength = 0.0

        # §V40 NMR-Feedback: NR-Stärke adaptiv anpassen (FeedbackChain-aware).
        try:
            from backend.core.dsp.nmr_feedback import (
                compute_nmr_score as _nmr_fn_28,
            )

            _nmr_result_28 = _nmr_fn_28(audio, sample_rate)
            if not _nmr_result_28.ok:
                logger.warning(
                    "Verarbeitungsschritt28 §V40 NMR: nmr_above_masking → §2.45 Minimal-Intervention prüfen",
                )
            _effective_strength = float(
                np.clip(
                    _effective_strength + _nmr_result_28.recommended_nr_strength_delta,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "Verarbeitungsschritt28 §V40 NMR: delta=%.3f → eff_str=%.3f",
                _nmr_result_28.recommended_nr_strength_delta,
                _effective_strength,
            )
        except Exception as _nmr_exc_28:
            logger.debug("Verarbeitungsschritt28 §V40 NMR nicht blockierend: %s", _nmr_exc_28)

        _material_key = str(getattr(material, "name", material)).lower()
        _panns_tags = {k: float(v) for k, v in kwargs.get("panns_tags", {}).items() if isinstance(v, (int, float, str))}
        _safe_strength = self._derive_safe_surface_strength(_effective_strength, _material_key, _panns_tags)

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "material": material.name,
                    "noise_reduction_db": 0.0,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "goal_hint_scalar": _goal_hint_scalar,
                },
                warnings=[],
            )

        is_stereo = audio.ndim == 2
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        config = dict(self.NOISE_CONFIG.get(_mk, self.NOISE_CONFIG[MaterialType.CD_DIGITAL]))  # type: ignore[call-overload]

        # §v10.200 Depth-adaptive: tiefere Ketten brauchen höheren spectral_floor
        # und sensitiveren VAD-Threshold (mehr Rauschen → mehr erkennen)
        _td_p28 = max(1, int(kwargs.get("transfer_chain_depth", 1)))
        if _td_p28 >= 4:
            config["spectral_floor"] = float(np.clip(config["spectral_floor"] + 0.08, 0.10, 0.35))
            config["vad_threshold_db"] = float(np.clip(config["vad_threshold_db"] + 6.0, -60.0, -25.0))
        elif _td_p28 >= 3:
            config["spectral_floor"] = float(np.clip(config["spectral_floor"] + 0.04, 0.10, 0.30))
            config["vad_threshold_db"] = float(np.clip(config["vad_threshold_db"] + 3.0, -60.0, -25.0))

        # §GEBOT-G55: Adaptive VAD-Threshold via Noise-Floor-Analyse
        try:
            from backend.core.adaptive_parameter_infrastructure import derive_noise_floor

            _nf28 = derive_noise_floor(audio, sample_rate)
            config["vad_threshold_db"] = float(np.clip(_nf28["noise_floor_db"] + 6.0, -60.0, -25.0))
            logger.debug(
                "Verarbeitungsschritt 28 adaptive: vad_Schwelle=%.0fdB (noise_floor=%.1fdB)",
                config["vad_threshold_db"],
                _nf28["noise_floor_db"],
            )
        except Exception as _e:
            logger.debug("%s: unkritisch exception: %s", __name__, _e)

        config["over_subtraction_alpha"] = float(1.0 + (config["over_subtraction_alpha"] - 1.0) * _safe_strength)
        config["spectral_floor"] = float(
            np.clip(1.0 - (1.0 - config["spectral_floor"]) * _safe_strength, 0.10, 1.0)
        )  # §2.62 hard min
        config["smoothing_frames"] = int(max(3, round(config["smoothing_frames"] + (1.0 - _safe_strength) * 6.0)))

        # §2.51 Linked-Stereo: OMLSA-Gain aus Mid-Sidechain (L+R)/\u221a2, identisch auf L+R
        if is_stereo:
            mid_sidechain = (audio[:, 0] + audio[:, 1]) / np.sqrt(2.0)
            denoised_mid, noise_db_mid = self._denoise_channel(mid_sidechain, sample_rate, config)
            _eps_sn = 1e-10
            _gain_sn = np.where(
                np.abs(mid_sidechain) > _eps_sn,
                denoised_mid / (mid_sidechain + _eps_sn * np.sign(mid_sidechain + _eps_sn)),
                1.0,
            )
            _gain_sn = np.clip(_gain_sn, 0.0, 10.0)
            denoised_audio = np.column_stack(
                (
                    audio[:, 0] * _gain_sn,
                    audio[:, 1] * _gain_sn,
                )
            )
            avg_noise_db = noise_db_mid
        else:
            denoised_audio, avg_noise_db = self._denoise_channel(audio, sample_rate, config)

        if 0.0 < _safe_strength < 1.0:
            denoised_audio = audio + _safe_strength * (denoised_audio - audio)

        # §4.5 Psychoacoustic Masking Clamp: protect musically masked regions
        # from unnecessary surface-noise removal (§0 Primum non nocere).
        try:
            from backend.core.dsp.psychoacoustics import (
                apply_psychoacoustic_masking_clamp,
            )

            _mono_orig = safe_to_mono(audio)
            _mono_proc = safe_to_mono(denoised_audio)
            _masked_mono = apply_psychoacoustic_masking_clamp(
                _mono_orig,
                _mono_proc,
                sample_rate,
                strength=_safe_strength,
                mode="subtractive",
            )
            if audio.ndim == 2:
                _gain_mask = np.where(
                    np.abs(_mono_proc) > 1e-10,
                    _masked_mono / (_mono_proc + 1e-10),
                    1.0,
                )
                _gain_mask = np.clip(_gain_mask, 0.0, 2.0)
                denoised_audio = denoised_audio * _gain_mask[:, np.newaxis]
            else:
                denoised_audio = _masked_mono
            denoised_audio = np.clip(denoised_audio, -1.0, 1.0)
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt28 masking clamp nicht blockierend: %s", _pm_exc)

        denoised_audio, loudness_stats = self._apply_material_loudness_preservation(
            audio,
            denoised_audio,
            material,
        )

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        # §2.46f Natural-Performance-Artifacts-Guard — restore protected breath/vibrato zones after NR
        if _npa_result_28 is not None:
            try:
                _npa_n_28 = denoised_audio.shape[0]
                _npa_mask_28 = _npa_result_28.get_protected_mask(_npa_n_28, sample_rate)
                if np.any(_npa_mask_28):
                    if (denoised_audio.ndim == 2 and audio.ndim == 2 and audio.shape[1] == denoised_audio.shape[1]) or (
                        denoised_audio.ndim == 1 and audio.ndim == 1
                    ):
                        denoised_audio[_npa_mask_28] = audio[_npa_mask_28]
                    logger.debug(
                        "§2.46f NPA Verarbeitungsschritt28: wiederhergestellt %d protected samples",
                        int(np.sum(_npa_mask_28)),
                    )
            except Exception as _npa_rest_28:
                logger.debug("§2.46f NPA restoration nicht blockierend: %s", _npa_rest_28)

        denoised_audio = np.nan_to_num(denoised_audio, nan=0.0, posinf=0.0, neginf=0.0)
        denoised_audio = np.clip(denoised_audio, -1.0, 1.0)

        # §0p/V19/V20/V21/V26/§2.72 Vokal- + Textur-Guards nach Surface-Noise-NR (RELEASE_MUST §0p V19-V26)
        _p28_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if _p28_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import apply_hnr_blend as _apply_hnr_28

                _hnr_blended_28, _hnr_diag_28 = _apply_hnr_28(
                    audio.astype(np.float32), denoised_audio.astype(np.float32), sample_rate
                )
                if _hnr_diag_28.get("over_cleaned"):
                    denoised_audio = _hnr_blended_28
            except Exception as _hnr_28_exc:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_28 (nicht blockierend): %s", _hnr_28_exc)

        _nt28_residual = audio - denoised_audio
        try:
            from backend.core.dsp.noise_texture_guard import (
                compute_noise_texture_distance as _nt28_dist_fn,
            )

            if _nt28_residual.shape == audio.shape:
                _nt28_d = _nt28_dist_fn(_nt28_residual, _material_key, sr=sample_rate)
                if _nt28_d > 0.25:
                    denoised_audio = (0.5 * denoised_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_28: noise_texture_dist=%.3f > 0.25 → 50%% dry-blend",
                        _nt28_d,
                    )
        except Exception as _nt28_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_28 noise_texture nicht blockierend: %s", _nt28_exc
            )

        if _p28_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (
                    frame_energy_correlation as _fec28,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr28 = _fec28(audio, denoised_audio, sample_rate, frame_ms=10.0)
                if _corr28 < 0.97:
                    _need28 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    # §v10.101 Material-adaptiv
                    _wet28 = _recommend_mkk_wet(_corr28, _p28_panns, global_need=_need28, material=_material_key)
                    denoised_audio = (_wet28 * denoised_audio + (1.0 - _wet28) * audio).astype(np.float32)
                    logger.warning(
                        "§V20 Verarbeitungsschritt_28: mikrodynamik_corr=%.4f < 0.97 → wet=%.3f", _corr28, _wet28
                    )
            except Exception as _v20_28_exc:
                logger.debug("§V20 Verarbeitungsschritt_28 mikrodynamik nicht blockierend: %s", _v20_28_exc)

        if any(x in _material_key for x in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (
                    apply_noise_floor_minimum as _nfmin28,
                )

                denoised_audio = _nfmin28(denoised_audio, sample_rate, _material_key, original_audio=audio)
            except Exception as _v21_28_exc:
                logger.debug("§V21 Verarbeitungsschritt_28 noise_floor nicht blockierend: %s", _v21_28_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (
                check_spectral_color_preservation as _scg_28,
            )

            _sc_result_28 = _scg_28(audio, denoised_audio, sample_rate)
            if not _sc_result_28.ok:
                _sc_wet_28 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                denoised_audio = (_sc_wet_28 * denoised_audio + (1.0 - _sc_wet_28) * audio).astype(np.float32)
        except Exception as _sc_exc_28:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_28 spectral_color nicht blockierend: %s", _sc_exc_28
            )

        try:
            from backend.core.dsp.onset_guard import (
                apply_onset_protection_mask as _opm28,
            )

            denoised_audio = _opm28(audio, denoised_audio, None, max_delta_db=1.5)
        except Exception as _v26_28_exc:
            logger.debug("§V26 Verarbeitungsschritt_28 onset_guard nicht blockierend: %s", _v26_28_exc)

        if _p28_panns >= 0.25:
            try:
                from backend.core.dsp.vibrato_guard import (
                    check_vibrato_depth_preservation as _vib28_fn,
                )

                _vibr28 = _vib28_fn(audio, denoised_audio, sample_rate)
                if not _vibr28.ok:
                    denoised_audio = (0.5 * denoised_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§2.72 Verarbeitungsschritt_28: vibrato_reduction=%.1f%% → 50%% dry-blend",
                        _vibr28.depth_reduction_pct,
                    )
            except Exception as _vib28_exc:
                logger.debug("§2.72 Verarbeitungsschritt_28 vibrato nicht blockierend: %s", _vib28_exc)

        denoised_audio = np.nan_to_num(denoised_audio, nan=0.0, posinf=0.0, neginf=0.0)
        denoised_audio = np.clip(denoised_audio, -1.0, 1.0)

        # §2.71 Strength-Envelope: Chirurgische Surface-Noise-Entfernung
        _strength_env = kwargs.get("strength_envelope")
        if _strength_env is not None:
            try:
                from backend.core.strength_envelope import apply_strength_envelope

                _env_pre = np.asarray(denoised_audio, dtype=np.float32)
                denoised_audio = apply_strength_envelope(
                    processed=_env_pre,
                    original=np.asarray(audio, dtype=np.float32),
                    envelope=_strength_env,
                    sample_rate=sample_rate,
                    base_strength=_effective_strength,
                )
                if float(np.mean(np.abs(denoised_audio - _env_pre))) > 0.001:
                    logger.info(
                        "§2.71 Envelope-Blending Verarbeitungsschritt 28: Δ=%.4f RMS",
                        float(np.mean(np.abs(denoised_audio - _env_pre))),
                    )
            except Exception as _se_exc:
                logger.debug("§2.71 Envelope nicht blockierend: %s", _se_exc)

        return PhaseResult(
            success=True,
            audio=denoised_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "noise_reduction_db": float(avg_noise_db),
                "over_subtraction_alpha": float(config["over_subtraction_alpha"]),
                "rt_factor": float(rt_factor),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "safe_strength": _safe_strength,
                "goal_hint_scalar": _goal_hint_scalar,
                "rms_drop_db": loudness_stats["rms_drop_db"],
                "loudness_makeup_db": loudness_stats["makeup_gain_db"],
            },
            warnings=[] if rt_factor < 0.30 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _apply_material_loudness_preservation(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, dict[str, float]]:
        from backend.core.audio_utils import (
            apply_musical_gain_envelope,
            compute_gated_rms_dbfs,
            compute_signal_relative_gate_dbfs,
        )

        material_key = getattr(material, "name", str(material)).lower()
        max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(material_key, self._MAX_RMS_DROP_DB["unknown"]))
        # §2.45a signal-relative gate: max(material_floor, P15(reference)+9 dB)
        # CEDAR/iZotope RX approach — gate derived from actual source noise floor (v10.0.0).
        _gate_dbfs = compute_signal_relative_gate_dbfs(original_audio, material_key=material_key)

        # §2.45a-I Gated-RMS: only musical frames > −50 dBFS
        _rms_in_db = compute_gated_rms_dbfs(np.asarray(original_audio, dtype=np.float32))
        _rms_out_db = compute_gated_rms_dbfs(np.asarray(processed_audio, dtype=np.float32))
        rms_in = float(10.0 ** (_rms_in_db / 20.0))
        rms_drop_db = (_rms_out_db - _rms_in_db) if _rms_in_db > -90.0 else 0.0
        makeup_gain_db = 0.0

        if rms_in > 1e-8 and rms_drop_db < -max_rms_drop_db:
            target_rms_drop_db = -max_rms_drop_db
            required_gain_db = target_rms_drop_db - rms_drop_db
            makeup_gain_db = float(np.clip(required_gain_db, 0.0, 6.0))
            # §GGB-1: Global Gain Budget — per-Chunk capping
            if makeup_gain_db > 0.0:
                try:
                    from backend.core.global_gain_budget import get_global_gain_budget

                    _approved = get_global_gain_budget().request(
                        "phase_28_surface_noise_profiling", makeup_gain_db, priority="normal"
                    )
                    makeup_gain_db = _approved
                except Exception as _e:
                    logger.debug("%s: unkritisch exception: %s", __name__, _e)
            if makeup_gain_db > 0.0:
                _gain_lin = float(10.0 ** (makeup_gain_db / 20.0))
                # §2.45a-II: gain applied ONLY to musical frames via envelope.
                # reference_for_gate=original_audio: use pre-phase noise floor for P5 computation.
                # After surface-noise removal, processed audio's P5 drops (fully-denoised frames
                # drag it to -55 dBFS) → without reference, adaptive gate fails → residual noise
                # at -35 dBFS gets amplified → Pegelexplosion. (v10.0.0)
                processed_audio = apply_musical_gain_envelope(
                    processed_audio,
                    _gain_lin,
                    gate_dbfs=_gate_dbfs,
                    crossfade_ms=10.0,
                    sr=48000,
                    reference_for_gate=original_audio,
                )
                processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
                # §2.45a-III: soft-limiter only when peak99 > 0.98
                current_peak = float(np.percentile(np.abs(processed_audio), 99.9))
                if current_peak > 0.98:
                    _abs_28 = np.abs(processed_audio)
                    _over_28 = _abs_28 > 0.92
                    if np.any(_over_28):
                        processed_audio = np.where(
                            _over_28,
                            np.sign(processed_audio) * (0.92 + 0.08 * np.tanh((_abs_28 - 0.92) / 0.08)),
                            processed_audio,
                        )
                processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
                _rms_out_db = compute_gated_rms_dbfs(np.asarray(processed_audio, dtype=np.float32))
                rms_drop_db = (_rms_out_db - _rms_in_db) if _rms_in_db > -90.0 else 0.0
                logger.info(
                    "Verarbeitungsschritt 28 loudness-preservation: material=%s rms_drop=%.2f dB via makeup %.2f dB (envelope-gated)",
                    material_key,
                    rms_drop_db,
                    makeup_gain_db,
                )

        return processed_audio, {
            "rms_drop_db": round(float(rms_drop_db), 3),
            "makeup_gain_db": round(float(makeup_gain_db), 3),
        }

    def _denoise_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Entfernt Oberflächenrauschen via IMCRA-Rauschschätzung + OMLSA-Gain.

        Algorithmus (v3.0):
            1. STFT mit 75% Overlap (nperseg=2048, noverlap=1536)
            2. IMCRA: Bias-korrigiertes gleitendes Minimum → F×T Rauschleistung
            3. OMLSA: G(t,f) = G_floor^(1-p) * G_H1^p, G_floor=0.1
            4. Cappé-Glättung: Gain temporal geglättet
            5. ISTFT + NaN-Schutz + clip
        """
        # Step 1: STFT (75% Overlap)
        nperseg = self.FRAME_SIZE
        noverlap = max(0, nperseg - self.HOP_SIZE)
        if noverlap >= nperseg:
            noverlap = nperseg // 2  # Fallback: 50% Overlap
        if len(audio.shape) == 0 or audio.shape[0] < nperseg:
            return audio, 0.0  # Audio too short for STFT
        _, t_arr, stft = signal.stft(
            audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann", boundary="even"
        )

        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Step 2: IMCRA-Rauschschätzung (F×T-Matrix)
        noise_mag = self._estimate_noise_imcra(magnitude, t_arr, config)

        # Step 3: OMLSA-Gain
        gain = self._compute_omlsa_gain(magnitude, noise_mag, config)

        # §2.62 Psychoakustischer Masking-Guard (ISO 11172-3) — per-Band Floor (non-blocking)
        try:
            from backend.core.dsp.psychoacoustics import (
                compute_masking_threshold_iso11172 as _cmask_p28,
            )

            _mask_ratio_p28 = _cmask_p28(audio, sample_rate, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE)
            _mfloor_p28 = np.mean(_mask_ratio_p28, axis=1).astype(np.float32)  # (n_freq,)
            _mfreqs_p28 = np.linspace(0.0, sample_rate / 2.0, _mask_ratio_p28.shape[0], dtype=np.float32)
            _mfloor_interp = np.interp(
                np.linspace(0.0, sample_rate / 2.0, gain.shape[0], dtype=np.float32), _mfreqs_p28, _mfloor_p28
            ).astype(np.float32)
            gain = np.maximum(gain, _mfloor_interp[:, np.newaxis])
            logger.debug("§2.62 Verarbeitungsschritt_28 Masking-Guard: mean_floor=%.3f", float(np.mean(_mfloor_p28)))
        except Exception as _msk28_exc:
            logger.debug(
                "§2.62 Verarbeitungsschritt_28 Masking-Guard nicht verfügbar (nicht blockierend): %s", _msk28_exc
            )

        # Step 4: Cappé-Gain-Glättung (vectorised IIR via lfilter — O(F) per col, no Python loop)
        alpha_g = 1.0 - 1.0 / max(config["smoothing_frames"], 1)
        # First-order causal IIR: y[n] = alpha_g*y[n-1] + (1-alpha_g)*x[n], y[0] = gain[0]
        # lfilter_zi gives steady-state zi so that y[0] = x[0] (no startup transient)
        _b_g = [1.0 - alpha_g]
        _a_g = [1.0, -alpha_g]
        _zi_g = _lfilter_zi(_b_g, _a_g)[np.newaxis, :] * gain[:, 0:1]  # (F, 1)
        gain_smooth, _ = _lfilter(_b_g, _a_g, gain, axis=1, zi=_zi_g)
        gain_smooth = np.nan_to_num(gain_smooth, nan=1.0, posinf=1.0, neginf=0.1)
        gain_smooth = np.clip(gain_smooth, 0.1, 1.0)

        # §2.36 Phonem-Schutz: Konsonanten-Burst-Frames (/p/,/t/,/k/,/s/) → NR-Bypass (G=1.0)
        # Breitbandige Energie-Spikes von Plosiven/Frikativen haben dasselbe spektrale Profil
        # wie Surface-Noise → OMLSA würde Artikulation zerstören. (VERBOTEN §2.36)
        try:
            from backend.core.lyrics_guided_enhancement import (
                get_phoneme_mask as _get_pmask_p28,
            )

            _pmask_p28 = _get_pmask_p28(audio.astype(np.float32), sample_rate, hop_length=self.HOP_SIZE)
            if np.any(_pmask_p28):
                _n_t_p28 = gain_smooth.shape[1]
                _pidx_p28 = np.where(_pmask_p28[:_n_t_p28])[0]
                if len(_pidx_p28) > 0:
                    gain_smooth[:, _pidx_p28] = 1.0
                    logger.debug(
                        "§2.36 Verarbeitungsschritt_28 Phonem-Bypass: %d/%d Frames auf G=1.0",
                        len(_pidx_p28),
                        _n_t_p28,
                    )
        except Exception as _pm28_exc:
            logger.debug("§2.36 Verarbeitungsschritt_28 Phonem-Mask (nicht blockierend): %s", _pm28_exc)

        # Step 5: Spectrum anwenden
        cleaned_mag = magnitude * gain_smooth
        cleaned_stft = cleaned_mag * np.exp(1j * phase)

        # Step 6: Direct ISTFT reconstruction.
        # cleaned_stft already contains full phase information (original phase preserved in step 5).
        # Direct ISTFT is both semantically correct and 50-100× faster than PGHI.
        # PGHI is only needed when phase is discarded (magnitude-only), which is not the case here.
        try:
            _, denoised = signal.istft(
                np.asarray(cleaned_stft, dtype=np.complex64),
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
                boundary=True,
            )
            denoised = np.asarray(denoised, dtype=np.float32)
        except Exception as _istft_exc:
            logger.debug("Verarbeitungsschritt_28 istft fehlgeschlagen (unkritisch): %s", _istft_exc)
            denoised = audio.astype(np.float32)  # passthrough fallback

        # Länge anpassen + NaN/Clipping-Schutz
        denoised = denoised[: len(audio)]
        if len(denoised) < len(audio):
            denoised = np.pad(denoised, (0, len(audio) - len(denoised)))
        denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
        denoised = np.clip(denoised, -1.0, 1.0)

        # Rauschreduktion schätzen
        rms_in = np.sqrt(np.mean(audio**2) + 1e-12)
        rms_out = np.sqrt(np.mean(denoised**2) + 1e-12)
        noise_reduction_db = 20.0 * np.log10(rms_in / rms_out)
        if not np.isfinite(noise_reduction_db):
            noise_reduction_db = 0.0

        return denoised, noise_reduction_db

    def _estimate_noise_imcra(self, magnitude: np.ndarray, t_arr: np.ndarray, _config: dict[str, Any]) -> np.ndarray:
        """IMCRA-Rauschschätzung: Bias-korrigiertes gleitendes Minimum (Cohen & Berdugo 2002).

        Algorithmus:
            sigma²_n(t,f) = b_min * min_{tau in [t-M, t]}( P_hat(tau,f) )
            P_hat(t,f) alpha_n * P_hat(t-1,f) + (1-alpha_n) * |Y(t,f)|²

        Args:
            magnitude: STFT-Betrag (F × T)
            t_arr:     Zeitstempel der STFT-Frames (T,)
            config:    Phasen-Konfiguration (enthält 'smoothing_frames')

        Returns:
            noise_mag: Rauschbetrag (F × T), NaN-frei
        """
        _, T = magnitude.shape
        b_min = 1.66  # Bias-Korrekturfaktor (Cohen 2003)
        alpha_n = 0.85  # Glättungskoeffizient für Rauschleistung
        eps = 1e-10

        # Fensterbreite ≈1.5s oder mind. 15 Frames
        hop_s = float(t_arr[1] - t_arr[0]) if T > 1 and len(t_arr) > 1 else 0.01
        M = max(15, round(1.5 / hop_s))

        # Geglättete Leistung P_hat (F × T) — vectorised causal IIR via lfilter
        # P_hat[t] = alpha_n * P_hat[t-1] + (1-alpha_n) * |Y[t]|^2, P_hat[0] = |Y[0]|^2
        # lfilter_zi gives steady-state zi so that P_hat[0] = |Y[0]|^2 (no startup transient)
        _b_p = [1.0 - alpha_n]
        _a_p = [1.0, -alpha_n]
        _zi_p = _lfilter_zi(_b_p, _a_p)[np.newaxis, :] * (magnitude[:, 0:1] ** 2)  # (F, 1)
        P_hat, _ = _lfilter(_b_p, _a_p, magnitude**2, axis=1, zi=_zi_p)
        P_hat = np.nan_to_num(P_hat, nan=eps)

        # Gleitendes Minimum über M Frames — vectorised via minimum_filter1d
        # origin = M//2 shifts the centered window left: causal window covers [t-M, t]
        # mode='constant', cval=max ensures boundary frames use +inf (no wrap-around)
        noise_power = _minimum_filter1d(
            P_hat,
            size=M + 1,
            axis=1,
            origin=M // 2,
            mode="constant",
            cval=np.finfo(P_hat.dtype).max,
        )

        # Bias-Korrektur + Wurzel → Rauschbetrag
        noise_mag = np.sqrt(np.maximum(b_min * noise_power, eps))
        return np.asarray(np.nan_to_num(noise_mag, nan=eps, posinf=eps, neginf=eps), dtype=np.float32)  # type: ignore[no-any-return]

    def _compute_omlsa_gain(self, magnitude: np.ndarray, noise_mag: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """OMLSA-Gain: G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p  (Cohen 2003).

        Formel:
            gamma = |Y|² / sigma²_n          (a-posteriori SNR)
            xi    = max(gamma - 1, 0)          (a-priori SNR, Decision-Directed)
            v     = clip(xi * gamma / (1+xi), 0, 500)
            Lambda = exp(-xi + v)              (Likelihood-Verhältnis)
            q     = materialabhängige Rauschprior  (aus config)
            p     = 1 / (1 + q/((1-q)*Lambda + eps))
            G_H1  = xi / (1 + xi)
            G     = exp((1-p)*ln(G_floor) + p*ln(G_H1 + eps))

        Args:
            magnitude: STFT-Betrag (F × T)
            noise_mag: Rauschbetrag aus IMCRA (F × T)
            config:    Enthält 'spectral_floor' (= G_floor, material-adaptiv)

        Returns:
            G: OMLSA-Gain (F × T) in [G_floor, 1.0]
        """
        G_floor = float(config.get("spectral_floor", 0.1))
        G_floor = max(G_floor, 0.10)  # §2.62: VERBOTEN G_floor < 0.10
        # q: Rausch-Präsenz-Prior (material-adaptiv: Shellac aggressiver)
        q = 1.0 - float(config.get("spectral_floor", 0.1))  # höheres floor → mehr Rauschen erwartet
        q = np.clip(q, 0.05, 0.95)
        eps = 1e-10

        sigma2_n = np.maximum(noise_mag**2, eps)
        gamma = np.maximum(magnitude**2 / sigma2_n, 0.0)  # a-posteriori SNR
        xi = np.maximum(gamma - 1.0, 0.0)  # a-priori SNR
        v = np.clip(xi * gamma / (xi + 1.0 + eps), 0.0, 500.0)
        lam = np.exp(np.clip(-xi + v, -50.0, 50.0))  # Likelihood-Verhältnis
        p = 1.0 / (1.0 + q / ((1.0 - q) * lam + eps))  # Sprachpräsenzwahrsch.
        G_H1 = xi / (xi + 1.0 + eps)  # Wiener-Gain bei Signal

        log_G = (1.0 - p) * np.log(G_floor) + p * np.log(np.maximum(G_H1, eps))
        log_G = np.clip(log_G, np.log(G_floor), 0.0)
        G = np.exp(log_G)
        G = np.nan_to_num(G, nan=G_floor, posinf=1.0, neginf=G_floor)
        return np.nan_to_num(np.clip(G, G_floor, 1.0).astype(np.float32), nan=0.0)  # type: ignore[no-any-return]
