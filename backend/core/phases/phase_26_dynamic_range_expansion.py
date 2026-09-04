#!/usr/bin/env python3
"""
Phase 26: Dynamic Range Expansion v2.0 - Professional
Multi-band upward/downward expansion for dynamic restoration.

Algorithm Overview:
1. Multi-Band Split: 4 bands (Bass/Low-Mid/Mid-High/High @ 150/800/5k Hz)
2. Per-Band Expansion:
   - RMS envelope detection (adaptive window 10-50ms)
   - Upward Expansion: Boost quiet passages (restore micro-dynamics)
   - Downward Expansion: Attenuate very quiet passages (noise floor reduction)
   - Soft-knee transition (3-9 dB per band)
   - Attack/Release envelopes (material-adaptive)
3. Material Adaptation:
   - Shellac/Vinyl: Conservative (preserve character, heavy compression)
   - Tape: Moderate (restore some dynamics)
   - Digital: Aggressive (restore full dynamics from over-compression)
4. Safety Limits: Prevent over-expansion (max 12 dB boost)
5. Multi-Band Combine: Reconstruct with preserved phase

Scientific Foundation:
- Reiss & McPherson (2015): Audio Effects - Theory and Implementation
- McNally (1984): Dynamic Range Control - Expansion fundamentals
- Giannoulis et al. (2012): Digital Dynamic Range Compressor Design
- Zölzer (2011): DAFX - Digital Audio Effects
- AES Convention Paper 5939 (2003): Multiband Dynamics Processing
- Katz (2015): Mastering Audio - The Art and Science

Industry Benchmarks:
- Waves C1 Compressor/Gate (Multi-band dynamics)
- FabFilter Pro-MB (Multiband processing)
- iZotope Ozone Dynamics (Mastering expansion)
- Oxford Dynamics (Professional expander/gate)
- DMG Audio Expurgate (Expansion specialist)

Quality Target: 0.70 → 0.88 (+26% improvement)
Performance Target: <0.25× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import os
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import audio_sample_count, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.phase_strength_contract import resolve_phase_strength_contract

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class DynamicRangeExpansion(PhaseInterface):
    """
    Professional Multi-Band Dynamic Range Expander.

    Key Features:
    - 4-band processing for frequency-specific control
    - Upward expansion (restore micro-dynamics)
    - Downward expansion (noise floor reduction)
    - Material-adaptive parameters
    - Soft-knee transitions
    - Look-ahead for transient preservation

    Use Cases:
    - Restore dynamics from over-compressed masters
    - Enhance micro-dynamics (breathing, room ambience)
    - Reduce noise floor in quiet passages
    - Material-specific dynamic restoration

    Performance: <0.25× realtime on modern CPU
    """

    # §v10.101: Bark-basierte Crossover-Frequenzen statt linearer Hz.
    # Abgeleitet aus Zwicker 1961: 24 kritische Bänder → 4 Gruppen.
    # Bass (Bark 1-7): 0-400 Hz | Low-Mid (Bark 8-14): 400-2000 Hz
    # Mid-High (Bark 15-20): 2000-6400 Hz | High (Bark 21-24): 6400-15500 Hz
    CROSSOVER_FREQS = [400, 2000, 6400]

    # Expansion parameters (material-adaptive)
    EXPANSION_CONFIG = {
        MaterialType.SHELLAC: {
            "upward_ratio": 1.15,  # 1:1.15 (conservative)
            "upward_threshold_db": -20,
            "downward_ratio": 1.5,  # 1:1.5 (gate-like)
            "downward_threshold_db": -40,
            "knee_width_db": 9,
            "attack_ms": 30,
            "release_ms": 150,
        },
        MaterialType.VINYL: {
            "upward_ratio": 1.2,
            "upward_threshold_db": -18,
            "downward_ratio": 2.0,
            "downward_threshold_db": -45,
            "knee_width_db": 6,
            "attack_ms": 25,
            "release_ms": 120,
        },
        MaterialType.TAPE: {
            "upward_ratio": 1.3,
            "upward_threshold_db": -15,
            "downward_ratio": 2.5,
            "downward_threshold_db": -50,
            "knee_width_db": 6,
            "attack_ms": 20,
            "release_ms": 100,
        },
        MaterialType.CASSETTE: {
            "upward_ratio": 1.3,
            "upward_threshold_db": -15,
            "downward_ratio": 2.5,
            "downward_threshold_db": -50,
            "knee_width_db": 6,
            "attack_ms": 20,
            "release_ms": 100,
        },  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: {
            "upward_ratio": 1.5,  # Aggressive (restore from brick-wall limiting)
            "upward_threshold_db": -12,
            "downward_ratio": 3.0,
            "downward_threshold_db": -55,
            "knee_width_db": 3,
            "attack_ms": 10,
            "release_ms": 80,
        },
        MaterialType.STREAMING: {
            "upward_ratio": 1.4,
            "upward_threshold_db": -14,
            "downward_ratio": 2.5,
            "downward_threshold_db": -52,
            "knee_width_db": 4,
            "attack_ms": 15,
            "release_ms": 90,
        },
    }

    # Max expansion (safety limit)
    MAX_EXPANSION_DB = 12.0

    # §v10.101 SOTA Per-Band CD Noise-Floor Targets [dBFS] — Bark-basiert
    # Jedes Frequenzband hat seinen eigenen Studio-Raumton.
    _CD_PER_BAND_FLOOR_DB: tuple[float, float, float, float] = (
        -65.0,  # 0-400 Hz (Bark 1-7): Bass — sauber aber mit Raumresonanz
        -72.0,  # 400-2000 Hz (Bark 8-14): Low-Mid — warm aber rein
        -76.0,  # 2000-6400 Hz (Bark 15-20): Mid-High — Präsenz, höchste Sauberkeit
        -70.0,  # 6400-15500 Hz (Bark 21-24): High — Luft ohne Hiss
    )

    _BAND_NAMES: tuple[str, str, str, str] = (
        "bass",
        "low_mid",
        "mid_high",
        "high",
    )

    def __init__(self):
        super().__init__()
        self.name = "Dynamic Range Expansion v2 Professional"
        self._max_expansion_db_current = self.MAX_EXPANSION_DB
        # §v10.61 Temporal smoothing state: dict[band_idx] = smoothed_floor_dbfs
        self._floor_ema_state: dict[int, float] = {}

    @staticmethod
    def _compute_expansion_profile(
        material: str,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """Berechnet adaptive expansion profile (§2.56, §6.2b)."""
        mat = str(material or "unknown").lower().replace("-", "_").replace(" ", "_")
        qm = str(quality_mode or "balanced").lower().replace("-", "_")
        if restorability_score is None:
            restorability_score = 50.0
        rest = float(np.clip(restorability_score, 0.0, 100.0))

        base = {
            "wax_cylinder": 3.0,
            "shellac": 3.6,
            "vinyl": 5.2,
            "tape": 6.0,
            "reel_tape": 6.2,
            "cassette": 5.5,
            "cd_digital": 8.6,
            "dat": 8.3,
            "streaming": 7.4,
            "unknown": 6.0,
        }.get(mat, 6.0)

        mode_adj = {
            "fast": -1.2,
            "balanced": 0.0,
            "restoration": 0.4,
            "quality": 1.2,
            "maximum": 2.0,
            "studio_2026": 2.0,
        }.get(qm, 0.0)

        # Low restorability => conservative expansion to avoid artifact amplification.
        rest_adj = ((rest - 50.0) / 50.0) * 1.0
        max_expansion_db = float(np.clip(base + mode_adj + rest_adj, 2.0, 12.0))
        return {"max_expansion_db": max_expansion_db}

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_26_dynamic_range_expansion",
            name="Dynamic Range Expansion v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=4,
            dependencies=["phase_10_compression", "phase_11_limiting"],
            estimated_time_factor=0.25,
            version="2.0.0",
            memory_requirement_mb=80,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.88,
            description="Multi-band upward/downward expansion for dynamic restoration",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="26")
        """
        Wendet an: dynamic range expansion to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material_type: Material type for adaptive processing

        Returns:
            PhaseResult with expanded audio
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

        # §V41 ForwardMaskingGuard: Stärke in post-transienten Masking-Fenstern erhöhen.
        _panns_s_26 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_26 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_26,
                )

                _fmg_26 = _fmg_fn_26()
                _fmz_26 = _fmg_26.compute_zones(audio, sample_rate)
                if _fmz_26:
                    _n_s_26 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_samples_26 = sum(z.end_sample - z.start_sample for z in _fmz_26)
                    _zone_frac_26 = float(np.clip(_zone_samples_26 / max(1, _n_s_26), 0.0, 1.0))
                    _boost_26 = _zone_frac_26 * 0.15
                    _effective_strength = float(np.clip(_effective_strength + _boost_26, 0.0, 1.0))
                    logger.debug(
                        "Verarbeitungsschritt26 §V41 ForwardMasking: zone_frac=%.2f boost=%.3f → eff_str=%.3f",
                        _zone_frac_26,
                        _boost_26,
                        _effective_strength,
                    )
            except Exception as _fmg_exc_26:
                logger.debug("Verarbeitungsschritt26 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_26)

        quality_mode = kwargs.get("quality_mode")
        restorability_score = kwargs.get("restorability_score", 50.0)
        material_key = str(getattr(material, "value", material) or "unknown")
        expansion_profile = self._compute_expansion_profile(material_key, quality_mode, restorability_score)
        self._max_expansion_db_current = float(expansion_profile["max_expansion_db"])

        # §v10.706 B14: SMP is_compressed → bereits komprimierte Medien nicht expandieren
        try:
            from backend.core.source_medium_profile import get_medium_profile

            _smp26 = get_medium_profile(material_key)
            if getattr(_smp26, "is_compressed", False):
                self._max_expansion_db_current *= 0.75
                logger.info(
                    "Verarbeitungsschritt26 SMP: is_compressed → max_expansion_db reduziert auf %.1f",
                    self._max_expansion_db_current,
                )
        except Exception:
            logger.debug(
                "Verarbeitungsschritt_26: SourceMediumProfile nicht verfügbar — is_compressed-Pruefung übersprungen"
            )

        # §v10.94 Non-Plus-Ultra: P10 Kompressions-Gain lesen und Expansion anpassen.
        # P10 (Compression) läuft VOR P26 (Expansion) mit denselben Crossover-Bändern
        # (150/800/5k Hz). Wenn P10 bereits −3 dB in "bass" komprimiert hat,
        # reduziert P26 die Expansion im Bass-Band um denselben Betrag → kein Pumping.
        _p10_gain: dict[str, float] = dict(kwargs.get("p10_per_band_gain_db", {}) or {})
        if _p10_gain:
            _total_p10 = sum(abs(float(v)) for v in _p10_gain.values())
            if _total_p10 > 0.5:
                # Reduziere max_expansion_db proportional zur P10-Kompression
                _p10_factor = float(np.clip(1.0 - _total_p10 / 12.0, 0.50, 1.0))
                _old_max = self._max_expansion_db_current
                self._max_expansion_db_current = float(max(2.0, _old_max * _p10_factor))
                logger.debug(
                    "P26: p10_per_band_gain_db=%s (total=%.1f dB) → max_expansion_db %.1f→%.1f",
                    _p10_gain,
                    _total_p10,
                    _old_max,
                    self._max_expansion_db_current,
                )

        is_stereo = audio.ndim == 2
        _mk = material
        config: dict[str, Any] = dict(self.EXPANSION_CONFIG.get(_mk, self.EXPANSION_CONFIG[MaterialType.CD_DIGITAL]))

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "dynamic_range_before_db": 0.0,
                    "dynamic_range_after_db": 0.0,
                    "dr_increase_db": 0.0,
                    "upward_ratio": 1.0,
                    "downward_ratio": 1.0,
                    "expansion_runtime_profile": expansion_profile,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "processing": "skipped_zero_strength",
                    "rt_factor": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # §v10.0.5: Wenn effektive Expansions-Ratio ≈ 1.0 (strength → 0),
        # ist die Wirkung vernachlässigbar. Expansion würde nur NOVELTY_CRIT
        # ohne hörbare Dynamik-Verbesserung einführen → Phase skippen.
        _eff_up = float(1.0 + (config["upward_ratio"] - 1.0) * _effective_strength)
        _eff_down = float(1.0 + (config["downward_ratio"] - 1.0) * _effective_strength)
        if _eff_up <= 1.02 and _eff_down <= 1.02:
            logger.info(
                "§v10.0.5 DR-Expansion-ueberspringen: eff_Verhaeltnis up/down=%.3f/%.3f (str=%.3f) → uebersprungen",
                _eff_up,
                _eff_down,
                _effective_strength,
            )
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "dr_increase_db": 0.0,
                    "effective_strength": _effective_strength,
                    "processing": "skipped_negligible_ratio",
                },
            )

        # Scale expansion aggressiveness toward neutral ratios for sparse locality.
        config["upward_ratio"] = float(1.0 + (config["upward_ratio"] - 1.0) * _effective_strength)
        config["downward_ratio"] = float(1.0 + (config["downward_ratio"] - 1.0) * _effective_strength)

        # §soft_saturation-Guard: DR-Expansion bei saturiertem Material konservativer.
        # Upward-Expansion macht Sättigungs-Transienten an Peaks sichtbar/hörbar (Pumpen).
        # soft_saturation_preserve=True → Expansion auf max. 50 % reduzieren.
        _p26_soft_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        _p26_soft_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        if _p26_soft_sat_preserve or _p26_soft_sat_sev > 0.4:
            _p26_sat_scale = 1.0
            if _p26_soft_sat_sev > 0.4:
                _p26_sat_scale = float(np.clip(1.0 - (_p26_soft_sat_sev - 0.4) * 0.8, 0.35, 1.0))
            if _p26_soft_sat_preserve and _p26_sat_scale > 0.50:
                _p26_sat_scale = 0.50
            # Ratios linear Richtung neutral (1.0) skalieren
            config["upward_ratio"] = float(1.0 + (config["upward_ratio"] - 1.0) * _p26_sat_scale)
            config["downward_ratio"] = float(1.0 + (config["downward_ratio"] - 1.0) * _p26_sat_scale)
            logger.debug(
                "Verarbeitungsschritt 26 soft_saturation guard: severity=%.2f preserve=%s → scale=%.2f "
                "(up_Verhaeltnis=%.3f down_Verhaeltnis=%.3f)",
                _p26_soft_sat_sev,
                _p26_soft_sat_preserve,
                _p26_sat_scale,
                config["upward_ratio"],
                config["downward_ratio"],
            )

        # Measure initial dynamic range
        dr_before = self._measure_dynamic_range(audio)

        # §v10.61 SOTA CD-Noise-Floor (organisch, per-band, psychoakustisch,
        # temporal geglättet): Der Export soll nach Neuaufnahme klingen.
        #
        # Drei SOTA-Dimensionen:
        #  1. Per-Band: Jedes der 4 Bänder hat eigenen spektralen Floor
        #     (Bass -65, Low-Mid -72, Mid-High -76, High -70 dBFS)
        #  2. Psychoakustische Maskierung: Laute Bänder maskieren ihren
        #     Rauschboden → Floor darf höher sein. Leise, exponierte
        #     Bänder → Floor muss strenger sein.
        #  3. Temporale Glättung: Floor-Parameter gleiten mit EMA
        #     (Attack 50ms, Release 200ms) — kein Frame-Pumpen.
        config["downward_floor_db_per_band"] = list(self._CD_PER_BAND_FLOOR_DB)
        config["downward_floor_knee_db"] = 4.0
        try:
            _mono_for_nf = audio if audio.ndim == 1 else audio.mean(axis=0).astype(np.float64)
            _current_nf_db = self._estimate_noise_floor_mono(_mono_for_nf, sample_rate)
            logger.debug(
                "Verarbeitungsschritt 26: NF_current=%.1fdB → per-band CD floors=%s, downward_Schwelle=%.1fdB (Betriebsart)",
                _current_nf_db,
                [f"{f:.0f}" for f in self._CD_PER_BAND_FLOOR_DB],
                config.get("downward_threshold_db", -50.0),
            )
        except Exception:
            logger.debug(
                "Verarbeitungsschritt_26_dynamic_range_expansion.py:420: Silent exception absorbed", exc_info=True
            )

        # §2.51 Linked-Stereo: Gain-Envelope aus \u221a(L\u00b2+R\u00b2)/\u221a2, identisch auf L+R
        if is_stereo:
            left, right = stereo_channel_view(audio)
            mono_sidechain = np.sqrt((left**2 + right**2) / 2.0)
            expanded_mono = self._expand_channel(mono_sidechain, sample_rate, config)
            _eps_exp = 1e-10
            _gain_exp = np.where(
                np.abs(mono_sidechain) > _eps_exp,
                expanded_mono / (mono_sidechain + _eps_exp),
                1.0,
            )
            _gain_exp = np.clip(_gain_exp, 0.0, 10.0)
            expanded_audio = stereo_like(left * _gain_exp, right * _gain_exp, audio)
        else:
            expanded_audio = self._expand_channel(audio, sample_rate, config)

        # Measure final dynamic range
        dr_after = self._measure_dynamic_range(expanded_audio)
        dr_increase_db = dr_after - dr_before

        # §6.2b DR-Material-Ceiling — Expansion darf physikalisches Medium-Maximum nicht überschreiten
        _dr_ceiling_capped = False
        try:
            from backend.core.carrier_transfer_characteristics import get_dr_ceiling_db

            _mat_key = material.value if hasattr(material, "value") else str(material)
            _quality_mode = kwargs.get("quality_mode", kwargs.get("mode", "restoration"))
            _is_studio = "studio" in str(_quality_mode).lower()

            _dr_ceil = get_dr_ceiling_db(_mat_key)
            if _is_studio:
                _dr_ceil = _dr_ceil * 1.5  # type: ignore[assignment]  # Studio 2026: Soft-Cap at 1.5×

            if dr_after > _dr_ceil:
                _dr_after_uncapped = dr_after  # save before blend-back for logging
                if dr_before >= _dr_ceil:
                    # Input already exceeds ceiling — expansion would violate §6.2b.
                    # Skip expansion entirely (cap_ratio=0 = full rollback to original).
                    expanded_audio = audio.copy()
                    dr_after = dr_before
                    dr_increase_db = 0.0
                    _dr_ceiling_capped = True
                    logger.info(
                        "§6.2b DR-Ceiling: source already exceeds ceiling "
                        "(dr_before=%.1f >= ceil=%.1f, material=%s) → expansion uebersprungen",
                        dr_before,
                        _dr_ceil,
                        _mat_key,
                    )
                else:
                    # Blend back toward input to cap DR at ceiling
                    _cap_ratio = max(0.0, min(1.0, (_dr_ceil - dr_before) / max(_dr_after_uncapped - dr_before, 0.01)))
                    expanded_audio = audio + _cap_ratio * (expanded_audio - audio)
                    expanded_audio = np.clip(expanded_audio, -1.0, 1.0)
                    dr_after = self._measure_dynamic_range(expanded_audio)
                    dr_increase_db = dr_after - dr_before
                    _dr_ceiling_capped = True
                    logger.info(
                        "§6.2b DR-Ceiling: dr_before=%.1f, uncapped=%.1f > ceil=%.1f (material=%s) → capped to %.1f dB",
                        dr_before,
                        _dr_after_uncapped,
                        _dr_ceil,
                        _mat_key,
                        dr_after,
                    )
        except Exception as _dr_ceil_exc:
            logger.debug("DR-Ceiling Pruefung fehlgeschlagen (nicht blockierend): %s", _dr_ceil_exc)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        expanded_audio = np.nan_to_num(expanded_audio, nan=0.0, posinf=0.0, neginf=0.0)
        expanded_audio = np.clip(expanded_audio, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            expanded_audio = audio + _effective_strength * (expanded_audio - audio)
            expanded_audio = np.clip(expanded_audio, -1.0, 1.0)

        # §2.46e Hallucination-Guard: DR-Expansion kann neue spektrale Energie einführen
        try:
            from backend.core.dsp.hallucination_guard import check_hallucination as _check_hg26

            _mono_26 = (
                expanded_audio.mean(axis=0)
                if (expanded_audio.ndim == 2 and expanded_audio.shape[0] == 2 and expanded_audio.shape[1] > 2)
                else (expanded_audio.mean(axis=1) if expanded_audio.ndim == 2 else expanded_audio)
            )
            _audio_mono_26 = (
                audio.mean(axis=0)
                if (audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2)
                else (audio.mean(axis=1) if audio.ndim == 2 else audio)
            )
            _mode_26 = str(kwargs.get("processing_mode", kwargs.get("mode", "restoration"))).lower()
            _hg_result26 = _check_hg26(
                _audio_mono_26.astype(np.float32),
                _mono_26.astype(np.float32),
                sr=sample_rate,
                mode="restoration" if "studio" not in _mode_26 else "studio_2026",
                bw_extension_context=True,
            )
            if _hg_result26.requires_rollback:
                logger.warning(
                    "§2.46e Verarbeitungsschritt_26 Hallucination-Guard rollback: spectral_novelty=%.3f",
                    _hg_result26.spectral_novelty,
                )
                expanded_audio = audio.copy()
            if _hg_result26.score_penalty > 0:
                logger.info(
                    "§2.46e Verarbeitungsschritt_26 Wert_penalty=%.1f (spectral_novelty=%.3f)",
                    _hg_result26.score_penalty,
                    _hg_result26.spectral_novelty,
                )
        except Exception as _hg26_exc:
            logger.debug("§2.46e Verarbeitungsschritt_26 Hallucination-Guard (nicht blockierend): %s", _hg26_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung (§2.74, non-blocking): Dynamic-Expansion darf Spektralfarbe nicht verändern
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg26,
            )

            _sc26 = _scg26(audio, expanded_audio, sample_rate)
            if not _sc26.ok:
                _sc26_wet = 0.70
                expanded_audio = (_sc26_wet * expanded_audio + (1.0 - _sc26_wet) * audio).astype(np.float32)
                logger.warning(
                    "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_26 spectral_color non-ok → strength −30%%"
                )
        except Exception as _sc26_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_26 spectral_color (nicht blockierend): %s", _sc26_exc
            )

        return PhaseResult(
            success=True,
            audio=expanded_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "dynamic_range_before_db": float(dr_before),
                "dynamic_range_after_db": float(dr_after),
                "dr_increase_db": float(dr_increase_db),
                "upward_ratio": float(config["upward_ratio"]),
                "downward_ratio": float(config["downward_ratio"]),
                "expansion_runtime_profile": expansion_profile,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rt_factor": float(rt_factor),
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
                "dr_ceiling_capped": _dr_ceiling_capped,
                "hallucination_decision": (
                    "rollback" if ("_hg_result26" in dir() and _hg_result26.requires_rollback) else "ok"
                ),
            },
            warnings=[] if rt_factor < 0.3 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _expand_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """Erweitert a single audio channel using multi-band processing."""
        # Create filter bank
        bands = self._split_into_bands(audio, sample_rate)

        # Expand each band
        expanded_bands = []
        for band_idx, band in enumerate(bands):
            expanded_band = self._expand_band(band, sample_rate, config, band_idx)
            expanded_bands.append(expanded_band)

        # Combine bands
        expanded_audio = self._combine_bands(expanded_bands)

        return expanded_audio[: audio_sample_count(audio)]

    def _split_into_bands(self, audio: np.ndarray, sample_rate: int) -> list:
        """
        Split audio into 4 frequency bands using Linkwitz-Riley 4th-order crossovers.

        LR4 = cascaded 2nd-order Butterworth applied twice.  Complementary
        subtraction guarantees perfect reconstruction: sum(bands) == audio.

        Previous implementation used independent Butterworth bandpass filters
        which introduced ±1.5 dB ripple at crossovers and phase cancellation.

        Scientific basis: Linkwitz & Riley 1976, JAES 24(1).
        """
        # Crossover 1: 150 Hz
        # sosfiltfilt (zero-phase) required: low is subtracted from audio to get rest_1;
        # causal sosfilt would introduce group delay → timing skew in complementary bands → Pegelexplosion (§2.51, V11)
        sos1 = signal.butter(2, self.CROSSOVER_FREQS[0], btype="low", fs=sample_rate, output="sos")
        from backend.core.audio_utils import (
            safe_sosfiltfilt as _safe_sosfiltfilt26,  # pylint: disable=import-outside-toplevel
        )

        low = _safe_sosfiltfilt26(sos1, audio)  # LR4-equivalent zero-phase low-pass
        rest_1 = audio - low  # Complementary high (>150 Hz)

        # Crossover 2: 800 Hz (applied to rest_1)
        sos2 = signal.butter(2, self.CROSSOVER_FREQS[1], btype="low", fs=sample_rate, output="sos")
        mid_low = _safe_sosfiltfilt26(sos2, rest_1)  # LR4-equivalent zero-phase
        rest_2 = rest_1 - mid_low  # >800 Hz

        # Crossover 3: 5000 Hz (applied to rest_2)
        sos3 = signal.butter(2, self.CROSSOVER_FREQS[2], btype="low", fs=sample_rate, output="sos")
        mid_high = _safe_sosfiltfilt26(sos3, rest_2)  # LR4-equivalent zero-phase
        high = rest_2 - mid_high  # >5000 Hz

        return [low, mid_low, mid_high, high]

    def _expand_band(
        self,
        band: np.ndarray,
        sample_rate: int,
        config: dict[str, float],
        band_idx: int = 0,
    ) -> np.ndarray:
        """
        Wendet an: expansion to a single band — fully vectorized.

        Replaces O(N) Python for-loop with numpy vectorized operations
        for ~100× speedup on typical audio (10M+ samples).

        §v10.61 SOTA: Per-Band spectral floor + psychoacoustic masking
        adaptation + temporal EMA smoothing.

        Scientific basis: Giannoulis et al. 2012 JAES 60(6) §3.
        """
        # Compute RMS envelope
        window_samples = max(1, int(config["attack_ms"] * sample_rate / 1000))
        envelope = self._compute_rms_envelope(band, window_samples)

        # Convert to dB
        envelope_db = 20.0 * np.log10(envelope + 1e-10)

        upward_thresh = config["upward_threshold_db"]
        downward_thresh = config["downward_threshold_db"]
        upward_ratio = config["upward_ratio"]
        downward_ratio = config["downward_ratio"]
        knee = config["knee_width_db"]
        half_knee = knee / 2.0

        # Vectorized gain computation (replaces per-sample for-loop)
        # Zones are mutually exclusive (preserving elif semantics)
        mask_up_full = envelope_db > (upward_thresh + half_knee)
        mask_up_knee = ~mask_up_full & (envelope_db > (upward_thresh - half_knee))
        mask_dn_full = ~mask_up_full & ~mask_up_knee & (envelope_db < (downward_thresh - half_knee))
        mask_dn_knee = ~mask_up_full & ~mask_up_knee & ~mask_dn_full & (envelope_db < (downward_thresh + half_knee))

        gain_db = np.zeros_like(envelope_db)

        # Upward expansion: above knee
        gain_db[mask_up_full] = (envelope_db[mask_up_full] - upward_thresh) * (upward_ratio - 1.0)

        # Upward expansion: in knee (soft transition)
        excess_k = envelope_db[mask_up_knee] - (upward_thresh - half_knee)
        gain_db[mask_up_knee] = (excess_k / knee) ** 2 * (upward_ratio - 1.0) * knee

        # Downward expansion: below knee
        gain_db[mask_dn_full] = -(downward_thresh - envelope_db[mask_dn_full]) * (downward_ratio - 1.0)

        # ═══════════════════════════════════════════════════════════════
        # §v10.61 SOTA CD-Noise-Floor-Guard — drei Dimensionen:
        #
        #  D1. Per-Band spektraler Floor: jedes der 4 Bänder hat eigenen
        #      Ziel-Raumton (Bass -65, Low-Mid -72, Mid-High -76, High -70)
        #  D2. Psychoakustische Maskierungs-Adaptation:
        #      - Band-Energie > -20 dBFS → Floor +8 dB (maskiert)
        #      - Band-Energie > -30 dBFS → Floor +5 dB
        #      - Band-Energie > -40 dBFS → Floor +2 dB
        #      - Band-Energie ≤ -40 dBFS → Floor +0 dB (exponiert, streng)
        #  D3. Temporale EMA-Glättung: Attack α=0.15 (50ms), Release α=0.05 (200ms)
        #      → verhindert Frame-Pumpen, Floor gleitet organisch
        # ═══════════════════════════════════════════════════════════════

        # D1: Per-band base floor
        _per_band_floors = config.get("downward_floor_db_per_band")
        if _per_band_floors is not None and band_idx < len(_per_band_floors):  # type: ignore[arg-type]
            _base_floor_db = float(_per_band_floors[band_idx])  # type: ignore[index]
        else:
            _base_floor_db = config.get("downward_floor_db", -72.0)

        # D2: Psychoacoustic masking — loud bands mask their noise floor
        _band_mean_db = float(np.mean(envelope_db))
        if _band_mean_db > -20.0:
            _mask_relax_db = 8.0
        elif _band_mean_db > -30.0:
            _mask_relax_db = 5.0
        elif _band_mean_db > -40.0:
            _mask_relax_db = 2.0
        else:
            _mask_relax_db = 0.0

        _adaptive_floor_db = _base_floor_db + _mask_relax_db

        # D3: Temporal EMA smoothing — glide, don't jump
        _ema_prev = self._floor_ema_state.get(band_idx, _adaptive_floor_db)
        if _adaptive_floor_db > _ema_prev:
            # Floor steigt (entspannt, mehr Maskierung) → schnell folgen
            _alpha = 0.15  # Attack: ~50 ms
        else:
            # Floor fällt (strenger, exponiert) → langsam folgen, kein Pumpen
            _alpha = 0.05  # Release: ~200 ms
        _smoothed_floor_db = _alpha * _adaptive_floor_db + (1.0 - _alpha) * _ema_prev
        self._floor_ema_state[band_idx] = _smoothed_floor_db

        # Soft asymptotic approach toward the smoothed, psychoacoustic floor
        _floor_knee_db = config.get("downward_floor_knee_db", 4.0)
        _envelope_result_db = envelope_db + gain_db
        _below_floor = _envelope_result_db < _smoothed_floor_db
        if np.any(_below_floor):
            _deficit_db = _smoothed_floor_db - _envelope_result_db[_below_floor]
            _deficit_db = np.asarray(_deficit_db, dtype=np.float64)
            _soft_correction = _deficit_db * np.exp(-_deficit_db / _floor_knee_db)
            gain_db[_below_floor] += _soft_correction

        # Downward expansion: in knee (soft transition)
        deficit_k = (downward_thresh + half_knee) - envelope_db[mask_dn_knee]
        gain_db[mask_dn_knee] = -((deficit_k / knee) ** 2) * (downward_ratio - 1.0) * knee

        # Limit expansion
        max_expansion_db = float(getattr(self, "_max_expansion_db_current", self.MAX_EXPANSION_DB))
        gain_db = np.clip(gain_db, -max_expansion_db, max_expansion_db)

        # Smooth gain (attack/release) — 16× downsampled for performance
        gain_db_smooth = self._smooth_gain(gain_db, sample_rate, config["attack_ms"], config["release_ms"])

        # Apply gain
        gain_linear = 10.0 ** (gain_db_smooth / 20.0)
        expanded_band = band * gain_linear

        return np.nan_to_num(np.asarray(expanded_band, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    def _compute_rms_envelope(self, audio: np.ndarray, window_samples: int) -> np.ndarray:
        """Berechnet RMS envelope."""
        audio_squared = audio**2
        # Use uniform filter for efficiency
        from scipy.ndimage import uniform_filter1d

        rms = np.sqrt(uniform_filter1d(audio_squared, window_samples, mode="nearest"))
        return np.nan_to_num(np.asarray(rms, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    def _smooth_gain(self, gain_db: np.ndarray, sample_rate: int, attack_ms: float, release_ms: float) -> np.ndarray:
        """
        Wendet Attack/Release-Glättung auf Gain an – 16-fach unterabgetastet.

        Uses block-max downsampling to preserve peak gain values while
        reducing the sequential IIR loop from N to N/16 iterations.
        Linear interpolation restores full-rate resolution.

        Scientific basis: Giannoulis et al. 2012, JAES 60(6) — log-domain
        ballistics with asymmetric attack/release.
        """
        DS = 16
        n = len(gain_db)

        if n < DS * 4:
            # Short signal — full-rate processing
            attack_coeff = np.exp(-1000.0 / (attack_ms * sample_rate))
            release_coeff = np.exp(-1000.0 / (release_ms * sample_rate))
            smoothed = np.zeros_like(gain_db)
            smoothed[0] = gain_db[0]
            for i in range(1, n):
                if gain_db[i] > smoothed[i - 1]:
                    smoothed[i] = attack_coeff * smoothed[i - 1] + (1 - attack_coeff) * gain_db[i]
                else:
                    smoothed[i] = release_coeff * smoothed[i - 1] + (1 - release_coeff) * gain_db[i]
            return smoothed  # type: ignore[no-any-return]

        # Downsample: preserve dominant gain per block (max |gain|)
        n_blocks = n // DS
        blocks = gain_db[: n_blocks * DS].reshape(n_blocks, DS)
        block_idx = np.argmax(np.abs(blocks), axis=1)
        ds_gain = blocks[np.arange(n_blocks), block_idx]

        # Adjusted coefficients for downsampled rate
        ds_sr = sample_rate / DS
        attack_coeff = np.exp(-1000.0 / (attack_ms * ds_sr))
        release_coeff = np.exp(-1000.0 / (release_ms * ds_sr))

        # Sequential IIR at 1/16th rate
        smoothed_ds = np.empty(n_blocks)
        smoothed_ds[0] = ds_gain[0]
        for i in range(1, n_blocks):
            if ds_gain[i] > smoothed_ds[i - 1]:
                smoothed_ds[i] = attack_coeff * smoothed_ds[i - 1] + (1 - attack_coeff) * ds_gain[i]
            else:
                smoothed_ds[i] = release_coeff * smoothed_ds[i - 1] + (1 - release_coeff) * ds_gain[i]

        # Interpolate back to full rate
        x_ds = np.arange(n_blocks) * DS + DS // 2
        x_full = np.arange(n)
        smoothed = np.interp(x_full, x_ds, smoothed_ds)

        return smoothed  # type: ignore[no-any-return]

    def _combine_bands(self, bands: list) -> np.ndarray:
        """Kombiniert frequency bands."""
        # Simple sum (Linkwitz-Riley crossovers maintain flat magnitude response)
        combined = sum(bands)
        return np.nan_to_num(np.asarray(combined, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    def _measure_dynamic_range(self, audio: np.ndarray) -> float:
        """Misst dynamic range (dB)."""
        if audio.ndim == 2:
            # Linked-Stereo-konforme Messung auf Mid-Kanal statt nur linker Spur.
            left, right = stereo_channel_view(audio)
            audio = ((left + right) / np.sqrt(2.0)).astype(np.float64)

        # Use percentile-based measurement (more robust than peak/RMS)
        audio_abs = np.abs(audio)
        p95 = np.percentile(audio_abs, 97)  # Loud passages
        p5 = np.percentile(audio_abs, 3)  # Quiet passages

        if p5 > 1e-10:
            dr_db = 20 * np.log10(p95 / p5)
        else:
            dr_db = 60.0  # Default high DR

        return float(dr_db)

    def _estimate_noise_floor_mono(self, mono: np.ndarray, sr: int) -> float:
        """§v10.61 Schätzt Rauschboden aus 5. Perzentil der Frame-RMS [dBFS].

        Liefert den aktuellen Noise-Floor für die CD-Target-Koordination
        zwischen Phase_03 (Denoise-Output) und Phase_26 (Downward-Expansion).
        """
        try:
            import numpy as np

            frame_len = max(256, sr // 100)  # ~10 ms
            n_frames = max(1, len(mono) // frame_len)
            if n_frames < 8:
                return -40.0  # zu kurz für Schätzung
            frames = np.reshape(mono[: n_frames * frame_len], (n_frames, frame_len))
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-15)
            noise_floor_rms = float(np.percentile(rms, 5))
            return float(20.0 * np.log10(max(noise_floor_rms, 1e-12)))
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            return -40.0  # safe fallback
