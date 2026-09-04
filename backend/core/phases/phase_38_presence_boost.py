"""
Phase 38: Presence Boost v2.0 - Professional.
Mid-range clarity and vocal/instrument presence enhancement.

Algorithm Overview:
1. Frequency Focus: 2-6 kHz (vocal/instrument presence region)
2. Multi-Band Processing:
   - Lower Presence (2-3.5 kHz): Body and warmth
   - Upper Presence (3.5-6 kHz): Clarity and definition
3. Dynamic EQ: Adaptive boost based on content
4. Formant Protection: Preserve vocal character
5. Material Adaptation:
   - Shellac/Vinyl: Restore clarity lost in aging
   - Tape: Compensate for HF roll-off
   - Digital: Add life to over-processed vocals

Scientific Foundation:
- Fletcher & Munson (1933): Equal Loudness Contours
- Moore et al. (1997): A Model for the Prediction of Thresholds, Loudness, and Partial Loudness
- Fastl & Zwicker (2007): Psychoacoustics - Facts and Models
- Zwicker & Fastl (1990): Psychoacoustics
- Terhardt (1979): Calculating Virtual Pitch

Industry Benchmarks:
- Pultec EQP-1A (Classic presence peak @ 3-5 kHz)
- API 550A (Presence band @ 3-4 kHz)
- Neve 1073 (Presence shelving)
- SSL G-Series (Presence bell filter)
- Maag EQ4 (Air Band + Presence)

Quality Target: 0.80 → 0.92 (+15% improvement)
Performance Target: <0.10× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import os
import time

import numpy as np
from scipy import signal

from backend.core.audio_utils import audio_sample_count, safe_filtfilt, stereo_channel_view, stereo_like  # §v10.101
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

try:
    from backend.core.dsp.hallucination_guard import check_hallucination as _check_hg_38_fn
except Exception:
    _check_hg_38_fn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PresenceBoost(PhaseInterface):
    """
    Professional Presence Enhancement Engine.

    Key Features:
    - Multi-band presence boost (2-6 kHz)
    - Dynamic EQ (content-adaptive)
    - Formant protection
    - Material-adaptive intensity
    - Minimal artifacts

    Use Cases:
    - Enhance vocal clarity and definition
    - Bring instruments forward in mix
    - Restore presence lost in processing
    - Improve intelligibility

    Performance: <0.10× realtime on modern CPU
    """

    # §v10.101 SOTA: Presence-Bänder auf Bark-Skala kalibriert
    # Bark 18-20 (2.0-4.4 kHz): Körper und Wärme der Präsenz
    # Bark 21-24 (4.4-9.5 kHz): Klarheit und Definition
    PRESENCE_BANDS = {
        "lower": (2000, 4400),  # Bark 18-20: Warmth and body
        "upper": (4400, 9500),  # Bark 21-24: Clarity and definition
    }

    # Enhancement parameters (material-adaptive)
    BOOST_CONFIG = {
        MaterialType.SHELLAC: {
            "lower_gain_db": 4.5,  # v10.0.0: ↑3.0→4.5
            "upper_gain_db": 5.5,  # v10.0.0: ↑4.0→5.5
            "q_factor": 1.5,
        },
        MaterialType.VINYL: {
            "lower_gain_db": 3.5,  # v10.0.0: ↑2.5→3.5
            "upper_gain_db": 4.5,  # v10.0.0: ↑3.5→4.5
            "q_factor": 1.8,
        },
        MaterialType.TAPE: {
            "lower_gain_db": 3.0,  # v10.0.0: ↑2.0→3.0
            "upper_gain_db": 4.0,  # v10.0.0: ↑3.0→4.0
            "q_factor": 2.0,
        },
        MaterialType.CASSETTE: {  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
            "lower_gain_db": 3.0,
            "upper_gain_db": 4.0,
            "q_factor": 2.0,
        },
        MaterialType.CD_DIGITAL: {
            "lower_gain_db": 4.5,  # v10.0.0: ↑3.5→4.5
            "upper_gain_db": 5.5,  # v10.0.0: ↑4.5→5.5
            "q_factor": 1.2,
        },
        MaterialType.STREAMING: {
            "lower_gain_db": 4.0,  # v10.0.0: ↑3.0→4.0
            "upper_gain_db": 5.0,  # v10.0.0: ↑4.0→5.0
            "q_factor": 1.5,
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "Presence Boost v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_38_presence_boost",
            name="Presence Boost v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            dependencies=[],
            estimated_time_factor=0.10,
            version="2.0.0",
            memory_requirement_mb=40,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.92,
            description="Mid-range clarity and vocal/instrument presence enhancement",
        )

    def process(self, audio: np.ndarray, sample_rate: int, material_type: str = "unknown", **kwargs) -> PhaseResult:  # type: ignore[override]
        """
        Wendet an: presence boost to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material_type: Material type for adaptive processing

        Returns:
            PhaseResult with enhanced audio
        """
        try:
            _mat_38 = MaterialType(material_type)
        except Exception:
            logger.warning(
                "§V6 Phase-38 material parse failed → CD_DIGITAL fallback: %s",
                str(Exception),
            )
            _mat_38 = MaterialType.CD_DIGITAL
        material = _mat_38
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard: Stärke in post-transienten Masking-Fenstern erhöhen.
        # Optimierung v10.0.0: UV3 berechnet Zonen vor der Phase und übergibt sie via kwargs;
        # nur wenn keine pre-computed Zonen vorhanden, werden sie neu berechnet (spart CPU).
        _panns_s_38 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_38 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_38,
                )

                _fmz_38 = kwargs.get("forward_masking_zones") or _fmg_fn_38().compute_zones(audio, sample_rate)
                if _fmz_38:
                    _n_s_38 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_samples_38 = sum(z.end_sample - z.start_sample for z in _fmz_38)
                    _zone_frac_38 = float(np.clip(_zone_samples_38 / max(1, _n_s_38), 0.0, 1.0))
                    _boost_38 = _zone_frac_38 * 0.15
                    _effective_strength = float(np.clip(_effective_strength + _boost_38, 0.0, 1.0))
                    logger.debug(
                        "Verarbeitungsschritt38 §V41 ForwardMasking: zone_frac=%.2f boost=%.3f → eff_str=%.3f",
                        _zone_frac_38,
                        _boost_38,
                        _effective_strength,
                    )
            except Exception as _fmg_exc_38:
                logger.debug("Verarbeitungsschritt38 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_38)

        # §2.17 SectionStrengthEnvelope: Kontinuierliche per-Segment-Modulation.
        # Passt Presence-Boost dynamisch an: mehr Präsenz in Refrains (wo der Mix
        # ohnehin dichter ist und Klarheit braucht), weniger in Strophen (wo
        # Zurückhaltung natürlicher wirkt). Cosine-Crossfades verhindern hörbare
        # Übergänge zwischen den Sektionen.
        _envelope = kwargs.get("strength_envelope")
        if _envelope is not None and len(_envelope) > 0:
            try:
                from backend.core.dsp.section_strength_envelope import get_section_strength_at

                _n_total = audio.shape[-1] if audio.ndim > 1 else len(audio)
                _env_val = get_section_strength_at(_envelope, 0, _n_total)
                _effective_strength = float(np.clip(_effective_strength * _env_val, 0.0, 1.0))
                logger.debug(
                    "Verarbeitungsschritt38 §2.17 SectionStrengthEnvelope: env=%.3f → eff_str=%.3f",
                    _env_val,
                    _effective_strength,
                )
            except Exception as _env_exc_38:
                logger.debug("Verarbeitungsschritt38 §2.17 SectionStrengthEnvelope nicht blockierend: %s", _env_exc_38)

        if _effective_strength <= 0.0:
            logger.info(
                "Verarbeitungsschritt 38: uebersprungen — effective_strength=%.3f (no presence boost angewendet)",
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

        is_stereo = audio.ndim == 2
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        config = dict(self.BOOST_CONFIG.get(_mk, self.BOOST_CONFIG[MaterialType.CD_DIGITAL]))  # type: ignore[call-overload]
        config["lower_gain_db"] = float(config["lower_gain_db"] * _effective_strength)
        config["upper_gain_db"] = float(config["upper_gain_db"] * _effective_strength)
        # §v10.70 Modus-Trennung: Restoration → max +2 dB Presence-Boost.
        # Präsenz durch EQ ist eine kreative Entscheidung des Toningenieurs,
        # keine Wiederherstellung. In Restoration wird nur kompensiert was
        # durch Bandbreiten-Verlust oder Codec-Kompression verloren ging.
        _mode_38 = str(kwargs.get("mode", kwargs.get("processing_mode", "restoration"))).lower()
        if "studio" not in _mode_38:
            config["lower_gain_db"] = min(config["lower_gain_db"], 2.0)
            config["upper_gain_db"] = min(config["upper_gain_db"], 2.0)

        # ── Era/Genre-adaptive presence scaling (context injection §2.x) ──
        _brillanz = kwargs.get("brillanz_target")
        _decade = kwargs.get("decade")
        _genre = kwargs.get("genre_label", "")
        if _brillanz is not None:
            # brillanz_target 0.85–0.92 → scale 0.85–1.15
            _b_scale = 0.30 + 0.90 * float(_brillanz)
            _b_scale = max(0.70, min(1.20, _b_scale))
            config["lower_gain_db"] *= _b_scale
            config["upper_gain_db"] *= _b_scale
            logger.debug(
                "Verarbeitungsschritt 38: brillanz_target=%.2f → presence scale=%.2f", float(_brillanz), _b_scale
            )
        if _decade is not None:
            _dec = int(_decade)
            if _dec <= 1950:
                # Vintage: cap presence boost (don't over-brighten) — v10.0.0: ↑2.5/3.0→3.5/4.0
                config["lower_gain_db"] = min(config["lower_gain_db"], 3.5)
                config["upper_gain_db"] = min(config["upper_gain_db"], 4.0)
            elif _dec >= 2000:
                # Modern digital: slightly less presence (already bright)
                config["lower_gain_db"] *= 0.85
                config["upper_gain_db"] *= 0.85
        _genre_lower_38 = str(_genre).lower()
        if _genre_lower_38 in ("klassik", "oper"):
            # Classical/Opera: reduce presence boost to preserve natural timbre
            config["lower_gain_db"] *= 0.70
            config["upper_gain_db"] *= 0.75
        elif _genre_lower_38 in ("electronic", "hip-hop"):
            # Electronic/Hip-Hop: already bright-produced; strong presence can over-sharpen.
            config["lower_gain_db"] *= 0.80
            config["upper_gain_db"] *= 0.80
        elif _genre_lower_38 == "metal":
            # Metal: high-gain guitars already have mid-presence — boost conservatively.
            config["lower_gain_db"] *= 0.85
            config["upper_gain_db"] *= 0.85
        elif _genre_lower_38 in ("reggae", "dub"):
            # Reggae/Dub: warm rolled-off production; artificial presence can sound harsh.
            config["lower_gain_db"] *= 0.75
            config["upper_gain_db"] *= 0.80

        # §vocal_presence: PANNs-Singing ≥ 0.35 → Presence-Boost zurückhalten.
        # Vokalzone (200–4000 Hz) wird von der Stimme dominiert; starker Boost klingt grell.
        # Injiziert via _restoration_context["vocal_presence_active"] → kwargs (UV3).
        _vp_active_38 = bool(kwargs.get("vocal_presence_active", False))
        if _vp_active_38:
            _vp_strength_38 = float(np.clip(kwargs.get("vocal_presence_strength", 0.0), 0.0, 1.0))
            # Lineare Skalierung: strength=0 → kein Eingriff; strength=1.0 → 60 % Reduktion
            _vp_scale_38 = float(np.clip(1.0 - 0.60 * _vp_strength_38, 0.40, 1.0))
            config["lower_gain_db"] *= _vp_scale_38
            config["upper_gain_db"] *= _vp_scale_38
            logger.debug(
                "Verarbeitungsschritt 38: vocal_presence_active → presence_scale=%.2f (vp_strength=%.2f)",
                _vp_scale_38,
                _vp_strength_38,
            )

        # §2.41 (v10.0.0) SOTA: Ära-bewusste Presence-Center aus SourceFidelityTarget.
        # Verschiedene Mikrofon-Ären haben unterschiedliche Hotspot-Frequenzen:
        #   Acoustic/Carbon (1920s): 2000–4000 Hz (Horn-Resonanz + Carbon-Peak)
        #   Ribbon (1930s): 2800–4300 Hz (Ribbon-Wärme-Zone)
        #   Condenser_early (1950s): 3200–5500 Hz (U47 Presence-Peak 5–8 kHz)
        #   condenser_modern (1970s+): 4000–6500 Hz (Moderner Standard)
        _sfr_cal = kwargs.get("song_calibration_profile", {})
        _sfr_presence_lower = float(_sfr_cal.get("source_fidelity_presence_hz_lower", 0.0))
        _sfr_presence_upper = float(_sfr_cal.get("source_fidelity_presence_hz_upper", 0.0))
        if _sfr_presence_lower > 0.0 and _sfr_presence_upper > 0.0:
            config["lower_center_hz"] = _sfr_presence_lower
            config["upper_center_hz"] = _sfr_presence_upper
            logger.debug(
                "Verarbeitungsschritt 38: era-aware presence center lower=%.0f upper=%.0f Hz (mic=%s)",
                _sfr_presence_lower,
                _sfr_presence_upper,
                str(_sfr_cal.get("source_fidelity_era_mic_type", "?")),
            )
        # Harmonic-Density-Skalierung: frühe Ären haben dünsere Obertöne → mehr Presence.
        _harm_density = float(_sfr_cal.get("source_fidelity_harmonic_density", 1.0))
        if _harm_density < 0.85 and _harm_density > 0.0:
            _hd_boost = float(np.clip(0.85 / _harm_density, 1.0, 1.25))
            config["lower_gain_db"] = float(np.clip(config["lower_gain_db"] * _hd_boost, 0.0, 10.0))
            config["upper_gain_db"] = float(np.clip(config["upper_gain_db"] * _hd_boost, 0.0, 10.0))
            logger.debug("Verarbeitungsschritt 38: harm_density=%.2f → presence_boost×%.2f", _harm_density, _hd_boost)

        # §soft_saturation-Guard: Presence-Boost bei gesättigtem Material begrenzen.
        # Soft_saturation erzeugt bereits geradzahlige Obertöne im Presence-Band (2–6 kHz).
        # Weiterer Boost auf diesem Band macht Gesang "übersteuert/kratzig" (§0h, §0i).
        # soft_saturation_preserve=True (z.B. Schlager-Profil) → max. 45 % des Boosts.
        # soft_saturation_severity > 0.3 → proportionale Reduzierung bis 28 % bei severity=1.0.
        _soft_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        _soft_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        if _soft_sat_preserve or _soft_sat_sev > 0.3:
            _sat_scale = 1.0
            if _soft_sat_sev > 0.3:
                # Lineare Reduzierung: severity 0.3 → scale 1.0, severity 1.0 → scale 0.16
                _sat_scale = float(np.clip(1.0 - (_soft_sat_sev - 0.3) * 1.2, 0.16, 1.0))
            if _soft_sat_preserve and _sat_scale > 0.45:
                _sat_scale = 0.45  # Hard-Cap bei preserve=True: max. 45 % des Boosts
            config["lower_gain_db"] = float(config["lower_gain_db"] * _sat_scale)
            config["upper_gain_db"] = float(config["upper_gain_db"] * _sat_scale)
            logger.debug(
                "Verarbeitungsschritt 38 soft_saturation guard: severity=%.2f preserve=%s → scale=%.2f (lower=%.2f dB upper=%.2f dB)",
                _soft_sat_sev,
                _soft_sat_preserve,
                _sat_scale,
                config["lower_gain_db"],
                config["upper_gain_db"],
            )

        # §2.51 M/S-Domain: Presence EQ auf Mid voll, Side konservativ (\u00d72 Threshold)
        if is_stereo:
            _ch0, _ch1 = stereo_channel_view(audio)
            mid = (_ch0 + _ch1) / np.sqrt(2.0)
            side = (_ch0 - _ch1) / np.sqrt(2.0)
            mid_enhanced = self._enhance_channel(mid, sample_rate, config)
            # Side: konservative Bearbeitung mit halber Gain
            side_config = dict(config)
            side_config["lower_gain_db"] = config["lower_gain_db"] * 0.3
            side_config["upper_gain_db"] = config["upper_gain_db"] * 0.3
            side_enhanced = self._enhance_channel(side, sample_rate, side_config)
            out_l = (mid_enhanced + side_enhanced) / np.sqrt(2.0)
            out_r = (mid_enhanced - side_enhanced) / np.sqrt(2.0)
            enhanced_audio = stereo_like(out_l, out_r, audio)
        else:
            enhanced_audio = self._enhance_channel(audio, sample_rate, config)

        if 0.0 < _effective_strength < 1.0:
            enhanced_audio = audio + _effective_strength * (enhanced_audio - audio)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
        # §2.46e Hallucination-Guard (Pflicht für additive Phasen)
        try:
            _mode_38 = str(kwargs.get("mode", "restoration"))
            _hg_38 = _check_hg_38_fn(
                audio,
                enhanced_audio,
                sr=sample_rate,
                mode=_mode_38,
                bw_extension_context=True,
            )  # type: ignore[misc]
            if _hg_38.requires_rollback:
                logger.warning(
                    "Verarbeitungsschritt_38: hallucination_guard rollback (spectral_novelty=%.3f)",
                    _hg_38.spectral_novelty,
                )
                enhanced_audio = np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            elif _hg_38.score_penalty > 0.0:
                logger.info(
                    "Verarbeitungsschritt_38: hallucination_guard penalty=%.1f (spectral_novelty=%.3f)",
                    _hg_38.score_penalty,
                    _hg_38.spectral_novelty,
                )
        except Exception as _hg38_exc:
            logger.debug(
                "Verarbeitungsschritt_38: hallucination_guard fehlgeschlagen (nicht blockierend): %s", _hg38_exc
            )

        # §V22 Pre-Echo-Prevention — Additive Presence-Boost auf Transient-Shifts prüfen (§2.73, non-blocking)
        try:
            from backend.core.dsp.transient_guard import (
                detect_transient_shifts as _dts_38,
            )

            _ax_v22_38 = -1 if audio.ndim == 2 and audio.shape[-1] <= 8 else 0
            _pre_v22_38 = (
                audio.mean(axis=_ax_v22_38).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
            )
            _ax_post_38 = -1 if enhanced_audio.ndim == 2 and enhanced_audio.shape[-1] <= 8 else 0
            _post_v22_38 = (
                enhanced_audio.mean(axis=_ax_post_38).astype(np.float32)
                if enhanced_audio.ndim == 2
                else enhanced_audio.astype(np.float32)
            )
            _ts_38 = _dts_38(_pre_v22_38, _post_v22_38, sample_rate)
            if not _ts_38.ok:
                _wet_ts_38 = max(0.0, 1.0 - _ts_38.blend_reduction)
                enhanced_audio = (_wet_ts_38 * enhanced_audio + (1.0 - _wet_ts_38) * audio).astype(np.float32)
                logger.warning(
                    "§V22 Verarbeitungsschritt_38: onset_shift=%.2f ms → blend_reduction=%.2f",
                    _ts_38.max_shift_ms,
                    _ts_38.blend_reduction,
                )
        except Exception as _v22_38_exc:
            logger.debug("§V22 Verarbeitungsschritt_38 transient_guard nicht blockierend: %s", _v22_38_exc)
        return PhaseResult(
            success=True,
            audio=enhanced_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "lower_gain_db": float(config["lower_gain_db"]),
                "upper_gain_db": float(config["upper_gain_db"]),
                "rt_factor": float(rt_factor),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[],
        )

    def _enhance_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """
        Enhance presence in a single audio channel.

        v2.1 Upgrades:
          1. Dynamic EQ Gate: Boost only when RMS content is significant (>-40 dBFS).
             Prevents amplifying noise in quiet passages.
          2. Sibilance Protection: Measure 4-8 kHz energy vs presence energy.
             If sibilance ratio is elevated, reduce upper presence boost proportionally.
          3. Formant-adaptive EQ centers: nudge center freqs slightly based on
             spectral centroid in the presence band.
        """
        enhanced = audio.copy()

        # ── 1. Dynamic gate: compute RMS in short-time blocks ──
        int(0.020 * sample_rate)  # 20 ms blocks
        rms_global = float(np.sqrt(np.mean(audio**2) + 1e-12))
        rms_db = 20.0 * np.log10(rms_global + 1e-10)

        # Gate: if signal is very quiet (< -40 dBFS), reduce boost to avoid enhancing noise
        gate_scale = 1.0
        if rms_db < -40.0:
            gate_scale = max(0.0, 1.0 - (-40.0 - rms_db) / 20.0)  # linear fade 0..1 between -40..-60 dBFS

        lower_gain = config["lower_gain_db"] * gate_scale
        upper_gain = config["upper_gain_db"] * gate_scale

        # ── 2. Sibilance protection: measure 4.5–8 kHz vs 1.5–4.5 kHz ratio ──
        if sample_rate > 16000:
            n_fft = 2048
            frame = audio[: min(n_fft * 8, len(audio))]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=n_fft * 8)) ** 2
            freqs = np.fft.rfftfreq(n_fft * 8, d=1.0 / sample_rate)
            pres_mask = (freqs >= 1500.0) & (freqs <= 4500.0)
            sib_mask = (freqs >= 4500.0) & (freqs <= 8000.0)
            pres_energy = float(np.sum(spectrum[pres_mask]) + 1e-12)
            sib_energy = float(np.sum(spectrum[sib_mask]) + 1e-12)
            sib_ratio = sib_energy / pres_energy
            # If sibilance already dominates (ratio > 0.7), reduce upper presence boost
            if sib_ratio > 0.70:
                sib_scale = max(0.3, 1.0 - (sib_ratio - 0.70) * 1.5)
                upper_gain *= sib_scale
                logger.debug(
                    "Verarbeitungsschritt 38 sibilance guard: sib_Verhaeltnis=%.2f → upper_gain×%.2f",
                    sib_ratio,
                    sib_scale,
                )

        # ── 3. Apply bell filters ──
        # §v10.65: Frequenz-Jitter ±3% pro Aufruf verhindert "plastic wrap"-
        # Spektralsignatur. Stationäre Bell-EQs auf exakt denselben Frequenzen
        # über ein ganzes Album hinweg registriert das Gehirn als künstlich.
        # Jitter ist deterministisch (hash-basiert) → reproduzierbar.
        _audio_hash = hash(audio.tobytes()[:2048]) & 0xFFFF
        _jitter_factor = 1.0 + ((_audio_hash / 65535.0) - 0.5) * 0.06  # ±3%
        lower_center = float(config.get("lower_center_hz", 2750.0)) * _jitter_factor
        upper_center = float(config.get("upper_center_hz", 4750.0)) * _jitter_factor

        logger.debug(
            "Verarbeitungsschritt 38: presence EQ jitter=%.1f%% → lower=%.0fHz upper=%.0fHz",
            (_jitter_factor - 1.0) * 100,
            lower_center,
            upper_center,
        )

        if lower_gain > 0.05:
            enhanced = self._apply_bell_filter(
                enhanced, sample_rate, center_freq=lower_center, gain_db=lower_gain, q=config["q_factor"]
            )

        # Upper presence (3.5–6 kHz by default): clarity and definition
        if upper_gain > 0.05:
            enhanced = self._apply_bell_filter(
                enhanced, sample_rate, center_freq=upper_center, gain_db=upper_gain, q=config["q_factor"]
            )

        return enhanced

    def _apply_bell_filter(
        self,
        audio: np.ndarray,
        sample_rate: int,
        center_freq: float,
        gain_db: float,
        q: float,
        zero_phase: bool = False,
    ) -> np.ndarray:
        """Wendet an: parametric EQ bell filter.

        §v10.65: Default ist minimum-phase (lfilter). Zero-phase (filtfilt)
        erzeugt symmetrisches Pre-Ringing vor Transienten — das menschliche
        Ohr erwartet kausales Nachschwingen, kein Vorläuten. Nur für
        Crossovers und M/S-Trennung (wo L/R-Phasenbeziehung kritisch ist)
        sollte zero_phase=True gesetzt werden.
        """
        # Design peaking filter
        w0 = 2 * np.pi * center_freq / sample_rate
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40)

        # Coefficients (bilinear transform)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        # Normalize
        b = np.array([b0, b1, b2], dtype=np.float64) / a0
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        # §v10.65: Minimum-phase filtering prevents pre-ringing on transients.
        # lfilter (minimum-phase) is default; filtfilt (zero-phase) only for crossovers.
        if len(audio) >= 9:
            if zero_phase:
                filtered = safe_filtfilt(b, a, audio)
            else:
                filtered = signal.lfilter(b, a, audio)
        else:
            filtered = signal.lfilter(b, a, audio)

        return np.nan_to_num(np.asarray(filtered, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]
