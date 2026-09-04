#!/usr/bin/env python3
"""
Phase 20: Professional Reverb Reduction v3.0 — OMLSA/IMCRA
===========================================================

Spektrale Nachhall-Reduktion via statistischer Rauschunterdrückung (OMLSA/IMCRA)
mit transientenerhaltender Nachbearbeitung.

SCIENTIFIC FOUNDATION (Primär, Über-SOTA-Pflicht):
- Cohen & Berdugo (2002): "Noise Estimation by Minima Controlled Recursive Averaging"
  → IMCRA: gleitendes Minimum mit Bias-Kompensation b_min=1.66 für diffuse
    Schallfelder wie Hall-Ausläufer. Ersetzt primitive Median-Rauschschätzung.
- Cohen (2003): "Noise Spectrum Estimation in Adverse Environments: Improved
  Minima Controlled Recursive Averaging" (OMLSA)
  → OMLSA: G(t,f) = G_floor^(1−p) · (ξ/(1+ξ))^p eliminiert musikalisches Rauschen.
- Le Roux & Vincent (2013): "Consistent Wiener Filtering" — Gain-Clamp G_floor.
- Cappé (1994): Temporale Gain-Glättung α_g=0.85 — unterdrückt Gain-Flattern.
- Perraudin et al. (2013): PGHI — scipy.signal.stft/istft sichert OLA-Phasenkonsistenz.

Historische Referenz (nur noch informativ, nicht als primärer Algorithmus):
- Moorer (1979): About This Reverberation Business
- Schroeder (1962): Natural Sounding Artificial Reverberation
- Kendall (2010): The Decorrelation of Audio Signals and Its Impact on Spatial Imagery
- Välimäki et al. (2012): Fifty Years of Artificial Reverberation
- ITU-R BS.1116-3: Methods for the Subjective Assessment of Small Impairments
- Bech & Zacharov (2006): Perceptual Audio Evaluation

INDUSTRY BENCHMARKS:
- iZotope RX 10 De-reverb (Spectral analysis + ML)
- Waves Clarity Vx DeReverb (Transient-preserving)
- Zynaptiq Unveil (Source separation based)
- SPL DeVerb (Dynamics-based)
- Cedar Retouch Pro (Professional standard)
- Accusonus ERA-D (Real-time dereverb)

ALGORITHM:
1. Transient Detection
   - Attack/Sustain separation
   - Transients bypass processing (preserve direct sound)

2. Spectral Envelope Analysis
   - STFT with 2048 window, 75% overlap
   - Identify reverb tail characteristics (exponential decay)
   - Separate direct sound from reflections

3. Spectral Gating
   - Frequency-dependent thresholds
   - Soft-knee gating (avoid artifacts)
   - Preserve tonal components while reducing diffuse field

4. Material-Adaptive Parameters
   - Shellac: Moderate (often already dry)
   - Vinyl: Light (preserve natural ambience)
   - Tape: Strong (analog reverb artifacts)
   - Digital: Minimal (production choice)

QUALITY TARGETS:
- Reverb reduction: 30-60% tail dampening
- Transient preservation: >98% attack energy
- Processing: <0.3× realtime

Author: Aurik Professional Team
Version: 3.0.0
Date: März 2026
"""

import logging
import time

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d as _min_filter1d_p20  # vectorised sliding-min
from scipy.signal import lfilter as _lfilter_p20  # vectorised IIR smoothing

from backend.core.audio_utils import compute_gated_rms_linear as _gated_rms_20
from backend.core.audio_utils import (
    safe_stft,  # §v10.115 explicit wrapper (no monkey-patch)
    to_channels_last,
)
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.restoration_policy import get_effective_song_goal_weights

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# Resource Management for fallback to lightweight algorithms
try:
    from backend.core.adaptive_resource_manager import adaptive_resource_manager

    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    logging.getLogger(__name__).warning("AdaptiveResourceManager not available, no automatic fallback")

# ML-Hybrid Support (Aurik 10.0.0 - Phase 20 v3.0)
try:
    from backend.core.hybrid.hybrid_dereverb import DereverbConfig, DereverbStrategy, HybridDereverb

    ML_HYBRID_AVAILABLE = True
except ImportError:
    ML_HYBRID_AVAILABLE = False
    logging.getLogger(__name__).warning("ML-Hybrid dereverb not available, using DSP-only mode")

# WPE Dereverberation (Spec §4.4 — Tier-1 DSP: kanonisches Dereverb-Plugin, Nakatani 2010)
try:
    from plugins.wpe_plugin import get_wpe_plugin

    WPE_AVAILABLE = True
except ImportError:
    WPE_AVAILABLE = False
    logging.getLogger(__name__).warning("WPE-Plugin nicht verfügbar — OMLSA/IMCRA-Fallback aktiv")

try:
    _PGHI_AVAILABLE_P20 = True
except ImportError:
    _PGHI_AVAILABLE_P20 = False

# §DSP-Instructions: MMSE-LSA Gain (Ephraim-Malah 1985) — E1 = exponential integral
try:
    from scipy.special import exp1 as _scipy_exp1_p20

    def _exp1_p20_gain(nu: np.ndarray) -> np.ndarray:
        """MMSE-LSA gain factor exp(0.5 * E1(ν)) per Ephraim-Malah (1985)."""
        return np.exp(np.clip(0.5 * _scipy_exp1_p20(np.maximum(nu, 1e-10)), 0.0, 5.0))  # type: ignore[no-any-return]

except ImportError as _exp1_import_err:  # pragma: no cover
    logger.debug("§V6 scipy.special.exp1 nicht verfügbar — Identity-Gain Fallback aktiviert (phase_20): %s", _exp1_import_err)

    def _exp1_p20_gain(nu: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """Fallback: identity = degenerate Wiener gain (scipy.special unavailable)."""
        return np.ones_like(nu)  # type: ignore[no-any-return]


logger = logging.getLogger(__name__)


class ReverbReduction(PhaseInterface):
    """Professional spectral-based reverb reduction."""

    _MAX_RMS_DROP_DB = {
        "tape": 2.5,
        "reel_tape": 2.2,
        "cassette": 2.8,
        "vinyl": 2.0,
        "shellac": 1.8,
        "wax_cylinder": 1.5,
        "cd_digital": 1.8,
        "mp3_low": 1.5,  # Fix 11B: digital — very conservative for compression material
        "mp3_high": 1.8,
        "aac": 1.8,
        "m4a": 1.8,
        "ogg": 1.8,
        "flac": 1.8,
        "streaming": 1.6,
        "unknown": 2.0,
    }

    # Fix 11C: Digital materials that need reverb defect evidence before ML dereverb
    _DIGITAL_LOW_REVERB_MATERIALS: frozenset = frozenset(
        {
            "mp3_low",
            "mp3_high",
            "aac",
            "m4a",
            "ogg",
            "streaming",
        }
    )
    # Minimum reverb severity required to use ML-Hybrid for digital material
    _DIGITAL_ML_REVERB_SEVERITY_MIN: float = 0.30

    # Material-adaptive reduction strength
    REDUCTION_STRENGTH = {
        MaterialType.SHELLAC: 0.50,  # Moderate (often dry already)
        MaterialType.VINYL: 0.40,  # Light (preserve natural ambience)
        MaterialType.TAPE: 0.65,  # Strong (analog reverb artifacts)
        MaterialType.CASSETTE: 0.65,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: 0.30,  # Minimal (production choice)
        MaterialType.STREAMING: 0.25,  # Very minimal
    }

    # Tail damping factor (how quickly reverb tail decays)
    TAIL_DAMPING = {
        MaterialType.SHELLAC: 0.70,
        MaterialType.VINYL: 0.60,
        MaterialType.TAPE: 0.80,
        MaterialType.CASSETTE: 0.80,  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: 0.50,
        MaterialType.STREAMING: 0.40,
    }

    # Transient threshold (energy ratio for transient detection)
    TRANSIENT_THRESHOLD = 3.0  # 3× energy increase = transient

    # STFT parameters
    WINDOW_SIZE = 2048
    HOP_SIZE = 512  # 75% overlap

    # MRSA Multi-Resolution Spectral Analysis zones (mandatory, §DSP-Spezialregeln)
    # VERBOTEN: arbitrary FFT sizes — only these 5 zone-optimal windows are permitted.
    _MRSA_ZONES: tuple = (
        # (name,       win_size, hop_size, f_low_hz, f_high_hz)
        ("sub_bass", 65536, 16384, 0, 250),
        ("mid_low", 16384, 4096, 250, 2500),
        ("mid", 8192, 2048, 2500, 8000),
        ("presence", 1024, 256, 8000, 16000),
        ("air", 128, 32, 16000, 24000),
    )
    # Hanning crossfade transition bandwidth at zone boundaries (~10 ms spectral transition)
    _MRSA_CROSSFADE_BW_HZ: float = 100.0

    def __init__(self):
        super().__init__()
        self.name = "Reverb Reduction v3 OMLSA/IMCRA"
        # If SGMSE TorchScript fails with deterministic shape/runtime errors,
        # avoid retrying the same expensive ML path on subsequent PMGG retries.
        self._force_dsp_only_due_ml_error: bool = False
        self._ml_disable_reason: str = ""
        self._preserve_mask_p20: np.ndarray | None = None  # gesetzt in process()

    def _adaptive_clarity_limits(self, kwargs: dict[str, object]) -> tuple[float, float, float, float]:
        """Berechnet song-adaptive C80/D50 guard limits.

        Uses §2.56 per-song goal weights as a modulation signal while keeping
        §4.5c base guard semantics intact.
        Returns: (c80_down_limit_db, c80_soft_limit_db, c80_hard_limit_db, d50_limit)
        """
        _gw = get_effective_song_goal_weights(kwargs)
        _w_nat = 1.0
        _w_auth = 1.0
        _w_timbre = 1.0
        _w_trans = 1.0
        _w_art = 1.0
        _w_bril = 1.0
        if isinstance(_gw, dict):
            _w_nat = float(np.clip(float(_gw.get("natuerlichkeit", 1.0)), 0.30, 2.00))
            _w_auth = float(np.clip(float(_gw.get("authentizitaet", 1.0)), 0.30, 2.00))
            _w_timbre = float(np.clip(float(_gw.get("timbre_authentizitaet", 1.0)), 0.30, 2.00))
            _w_trans = float(np.clip(float(_gw.get("transparenz", 1.0)), 0.30, 2.00))
            _w_art = float(np.clip(float(_gw.get("artikulation", 1.0)), 0.30, 2.00))
            _w_bril = float(np.clip(float(_gw.get("brillanz", 1.0)), 0.30, 2.00))

        _preserve_w = float(np.clip((_w_nat + _w_auth + _w_timbre) / 3.0, 0.30, 2.00))
        _clarity_w = float(np.clip((_w_trans + _w_art + _w_bril) / 3.0, 0.30, 2.00))

        _rest = float(np.clip(float(kwargs.get("restorability_score", 65.0)), 0.0, 100.0))  # type: ignore[arg-type]
        _rest_factor = float(np.clip(1.0 + (50.0 - _rest) / 250.0, 0.85, 1.20))

        _ratio = float(np.sqrt(_clarity_w / max(_preserve_w, 1e-6)))
        c80_down_limit = float(np.clip((-2.0 * _rest_factor) / np.sqrt(max(_preserve_w, 1e-6)), -3.2, -1.2))
        c80_soft_limit = float(np.clip(4.0 * _ratio * _rest_factor, 2.8, 5.2))
        c80_hard_limit = float(np.clip(6.0 * _ratio * _rest_factor, 4.2, 7.5))
        d50_limit = float(np.clip(0.12 * _ratio * _rest_factor, 0.08, 0.18))
        return c80_down_limit, c80_soft_limit, c80_hard_limit, d50_limit

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_20_reverb_reduction",
            name="Reverb Reduction v3 OMLSA/IMCRA",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_03_denoise"],
            estimated_time_factor=0.15,
            version="3.0.0",
            memory_requirement_mb=120,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description=(
                "Nachhall-Reduktion via STFT-OMLSA/IMCRA (Cohen 2002/2003) — "
                "diffuse Schallfelder ohne musikalisches Rauschen, "
                "Transientenerhalt und scipy.signal.stft/istft (PGHI-konsistent)"
            ),
        )

    def process(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.VINYL, **kwargs
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="20")
        """
        Wendet an: reverb reduction.

        Args:
            audio: Audio samples (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type

        Returns:
            PhaseResult with reverb-reduced audio
        """
        self.validate_input(audio)
        # ── §v10 PIM: Per-Band-Intensität lesen ──
        _per_band_mask = None
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity, compute_per_band_nr_mask

            _pim = apply_pim_intensity(kwargs, "reverb", default_nr=0.5, default_de_ess=0.3, default_comp=1.0)
            _pim_map = kwargs.get("pim_intensity_map")
            if _pim_map is not None and "noise_reduction_strength" in kwargs:
                kwargs["noise_reduction_strength"] = _pim["nr_strength"]
            if _pim_map is not None:
                _per_band_mask = compute_per_band_nr_mask(_pim_map, sample_rate)
        except Exception as e:
            logger.warning("Verarbeitungsschritt_20_reverb_reduction.py::verarbeiten Ersatzpfad: %s", e)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p20_transposed = to_channels_last(audio)
        start_time = time.time()

        # §2.46f Natural-Performance-Artifacts-Guard — detect protected zones before reverb reduction
        _npa_result_20 = None
        try:
            from backend.core.natural_performance_detector import get_natural_performance_detector  # pylint: disable=import-outside-toplevel  # noqa: I001

            _npa_result_20 = get_natural_performance_detector().detect(audio, sample_rate)
        except Exception as _npa_exc_20:
            logger.debug("§2.46f NPA detection nicht blockierend: %s", _npa_exc_20)

        # §0p Formant-Integrity pre-snapshot — §0p: F1–F4 dürfen durch keine Phase um mehr als ±15% verschoben werden
        _f1_pre_20 = None
        _p20_panns_fgt = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if _p20_panns_fgt >= 0.35:
            try:
                from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker as _get_lfc_20  # pylint: disable=import-outside-toplevel  # noqa: I001

                _ft_in_20 = audio.mean(axis=0) if audio.ndim == 2 else audio
                _lfc_res_20 = _get_lfc_20().track(_ft_in_20.astype(np.float32), sample_rate)
                _f1_pre_20 = float(_lfc_res_20.get("f1_mean", 0.0)) or None
            except Exception as e:
                logger.warning("Verarbeitungsschritt_20_reverb_reduction.py::verarbeiten Ersatzpfad: %s", e)

        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        strength = self.REDUCTION_STRENGTH.get(_mk, 0.4)  # type: ignore[call-overload]
        damping = self.TAIL_DAMPING.get(_mk, 0.6)  # type: ignore[call-overload]

        # Locality-aware intensity control from UV3.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength, 0.0, 1.0))

        # §V40 NMR-Feedback: NR-Stärke adaptiv anpassen (FeedbackChain-aware).
        try:
            from backend.core.dsp.nmr_feedback import (
                compute_nmr_score as _nmr_fn_20,  # pylint: disable=import-outside-toplevel
            )

            _nmr_result_20 = _nmr_fn_20(audio, sample_rate)
            if not _nmr_result_20.ok:
                logger.warning(
                    "Verarbeitungsschritt20 §V40 NMR: nmr_above_masking → §2.45 Minimal-Intervention prüfen",
                )
            _effective_strength = float(
                np.clip(
                    _effective_strength + _nmr_result_20.recommended_nr_strength_delta,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "Verarbeitungsschritt20 §V40 NMR: delta=%.3f → eff_str=%.3f",
                _nmr_result_20.recommended_nr_strength_delta,
                _effective_strength,
            )
        except Exception as _nmr_exc_20:  # pylint: disable=broad-except
            logger.debug("Verarbeitungsschritt20 §V40 NMR nicht blockierend: %s", _nmr_exc_20)

        # §2.51 Locality: der Faktor dämpft die FINALE Stärke (auch ohne NMR-Feedback) —
        # locality < 1 garantiert eff < 1.
        _effective_strength = float(np.clip(_effective_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                metrics={
                    "rms_change_db": 0.0,
                    "reduction_strength": 0.0,
                    "tail_damping": damping,
                    "material": material.value,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                },
            )

        strength = float(np.clip(strength * _effective_strength, 0.0, 1.0))

        # §2.20 Genre-adaptive reverb: classical/opera preserve concert hall ambience;
        # Schlager profile may define dereverb_strength_cap.
        # Defense-in-depth: SongCal genre_reverb_factor already scales strength via PMGG;
        # these in-phase hardcaps are secondary guards if SongCal propagation fails.
        genre_label = kwargs.get("genre_label", "Unbekannt")
        if genre_label in ("Klassik", "Oper"):
            strength = min(strength, 0.25)
            logger.debug("Verarbeitungsschritt 20: Genre=%s → reverb strength capped to %.2f", genre_label, strength)
        elif genre_label == "Jazz":
            strength = min(strength, 0.30)
            logger.debug("Verarbeitungsschritt 20: Genre=%s → reverb strength capped to %.2f", genre_label, strength)
        elif genre_label == "Reggae":
            # Dub/Echo-Reverb is an iconic genre-defining device — protect aggressively.
            strength = min(strength, 0.20)
            logger.debug("Verarbeitungsschritt 20: Genre=%s → reverb strength capped to %.2f", genre_label, strength)
        elif genre_label == "Gospel":
            # Church reverb is authenticity, not artifact (§0).
            strength = min(strength, 0.30)
            logger.debug("Verarbeitungsschritt 20: Genre=%s → reverb strength capped to %.2f", genre_label, strength)
        elif genre_label == "Folk":
            # Small-room naturalness is part of the performance — preserve gently.
            strength = min(strength, 0.40)
            logger.debug("Verarbeitungsschritt 20: Genre=%s → reverb strength capped to %.2f", genre_label, strength)

        # Vocal-preservation: avoid over-dereverb on singing-heavy content.
        _vocal_conf_20 = float(kwargs.get("vocal_confidence", kwargs.get("panns_singing_confidence", 0.0)))
        _vocal_detected_20 = bool(kwargs.get("vocal_detected", False)) or (_vocal_conf_20 >= 0.35)
        if _vocal_detected_20:
            _vocal_cap_20 = float(np.clip(0.38 - 0.10 * _vocal_conf_20, 0.28, 0.38))
            if strength > _vocal_cap_20:
                logger.debug(
                    "Verarbeitungsschritt 20: vocal guard active (conf=%.2f) → strength %.2f -> %.2f",
                    _vocal_conf_20,
                    strength,
                    _vocal_cap_20,
                )
                strength = _vocal_cap_20

        # §0j vocal_energy_bias_db from VFA context — §0j: energy_bias from UV3 VocalFocusAnalyzer
        _ctx_energy_bias_20 = float(kwargs.get("_restoration_context", {}).get("vocal_energy_bias_db", -6.0))
        # §4.8a-ii + §Gap8: preserve_mask injiziert von UV3, gesetzt auf self für _run_omlsa_mrsa.
        _pm_raw_p20 = kwargs.get("_restoration_context", {}).get("preserve_mask")
        self._preserve_mask_p20 = (
            np.asarray(_pm_raw_p20, dtype=np.float32)
            if isinstance(_pm_raw_p20, np.ndarray) and _pm_raw_p20.size > 0
            else None
        )
        if _ctx_energy_bias_20 < -6.0 and _vocal_detected_20:
            # Convert energy_bias_db to additional strength reduction (more negative = less aggressive)
            _eb_scale_20 = float(10.0 ** (_ctx_energy_bias_20 / 20.0))  # e.g. -9 dB → 0.355
            strength = float(np.clip(strength * max(_eb_scale_20, 0.20), 0.05, strength))
            logger.debug(
                "§0j Verarbeitungsschritt_20 vocal_energy_bias_db=%.1f dB → strength scaled to %.3f",
                _ctx_energy_bias_20,
                strength,
            )

        # §2.14+ Era-adaptive: older recordings (pre-1960) often have room ambience
        # integral to the character — reduce dereverb strength.
        decade = kwargs.get("decade")
        if decade is not None and decade <= 1950:
            strength = min(strength, 0.30)
            logger.debug("Verarbeitungsschritt 20: decade=%d → reverb strength capped to %.2f", decade, strength)

        # §2.46f Room-Acoustics-Fingerprint guard — authentic room character protection.
        # Injected by UV3 from room_acoustics_fingerprinter into _restoration_context.
        _raf_20 = kwargs.get("room_acoustics_fingerprint") or {}
        _raf_cap_20 = float(_raf_20.get("dereverb_strength_cap", 1.0))
        if _raf_cap_20 < 1.0 and strength > _raf_cap_20:
            logger.debug(
                "Verarbeitungsschritt 20 §2.46f RoomAcoustics guard: rt60=%.2fs room=%s → strength %.2f → %.2f",
                float(_raf_20.get("rt60_s", 0.0)),
                _raf_20.get("room_type", "?"),
                strength,
                _raf_cap_20,
            )
            strength = _raf_cap_20

        # ML-Hybrid Mode Routing (v3.0)
        quality_mode = kwargs.get("quality_mode", "quality")

        # Check resource availability for ML-Hybrid (fallback to lightweight if needed)
        use_lightweight = False
        if RESOURCE_MANAGER_AVAILABLE:
            use_lightweight = adaptive_resource_manager.should_use_lightweight_mode()
            # Quality-first contract: do not downgrade to lightweight in quality tiers.
            if quality_mode in ["quality", "maximum"]:
                use_lightweight = False
            elif use_lightweight:
                logger.info(
                    "Verarbeitungsschritt 20: Resource constraint erkannt, forcing DSP-only Betriebsart (CPU: %.1f%%, Memory: %.1f%%)",
                    adaptive_resource_manager.get_cpu_usage(),
                    adaptive_resource_manager.get_memory_usage(),
                )

        # Fix 11C: For digital material with no reverb defect evidence → skip expensive ML
        _defect_result_ph20 = kwargs.get("defect_result")
        _reverb_severity_ph20 = 0.0
        if _defect_result_ph20 is not None:
            for _d in getattr(_defect_result_ph20, "defects", []):
                _dtype = getattr(_d, "defect_type", "")
                if isinstance(_dtype, str):
                    _dtype_s = _dtype.lower()
                else:
                    _dtype_s = str(_dtype).lower()
                if "reverb" in _dtype_s:
                    _reverb_severity_ph20 = max(_reverb_severity_ph20, float(getattr(_d, "severity", 0.0)))
        _skip_ml_digital_no_reverb = (
            material.value in self._DIGITAL_LOW_REVERB_MATERIALS
            and _reverb_severity_ph20 < self._DIGITAL_ML_REVERB_SEVERITY_MIN
        )

        # §v10.14 Universal Primum-non-nocere: Bei quasi-null Reverb (alle Materialien)
        # jede DSP/ML-Verarbeitung überspringen. Die Fix11C deckte nur digitale Materialien
        # ab — aber auch analoge Träger (cassette, reel_tape, lacquer) können nach
        # vorherigen Phasen bereits reverb-frei sein. DSP-Gating auf reverb-freiem Signal
        # attenuierte unnötig −3.19 dB und verursachte ROLLBACK (Δ−0.301 HPE).
        # Erweiterung: _skip_ml_digital_no_reverb prüft jetzt ALLE Materialien.
        _is_digital_ph20 = material.value in self._DIGITAL_LOW_REVERB_MATERIALS
        _skip_ml_digital_no_reverb = _is_digital_ph20 and _reverb_severity_ph20 < self._DIGITAL_ML_REVERB_SEVERITY_MIN
        _skip_all_no_reverb = (not _is_digital_ph20) and _reverb_severity_ph20 < 0.02
        if _skip_all_no_reverb:
            logger.info(
                "Verarbeitungsschritt 20: §v10.14 — reverb_severity=%.3f < 0.02, material=%s → passthrough "
                "(universal Primum-non-nocere)",
                _reverb_severity_ph20,
                material.value,
            )
            _pt = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            _pt = np.clip(_pt, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=_pt,
                metrics={
                    "rms_change_db": 0.0,
                    "reduction_strength": 0.0,
                    "reverb_estimate": _reverb_severity_ph20,
                    "material": material.value,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "passthrough_primum_non_nocere_v10_14",
                    "reverb_severity": _reverb_severity_ph20,
                    "skip_reason": "reverb_below_detectable_threshold_all_materials",
                    # §2.51 Locality-Vertrag: skalierte Soll-Stärke bleibt sichtbar.
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                },
            )

        if _skip_ml_digital_no_reverb:
            if _reverb_severity_ph20 < 0.10:
                # §0 Primum non nocere: near-zero reverb on digital material → passthrough.
                # Any dereverb processing would only add artifacts.
                logger.info(
                    "Verarbeitungsschritt 20: Fix11C — digital=%s, reverb_severity=%.3f < 0.10 → passthrough (§0 Primum non nocere)",
                    material.value,
                    _reverb_severity_ph20,
                )
                _pt = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
                _pt = np.clip(_pt, -1.0, 1.0)
                return PhaseResult(
                    success=True,
                    audio=_pt,
                    metrics={
                        "rms_change_db": 0.0,
                        "reduction_strength": 0.0,
                        "reverb_estimate": _reverb_severity_ph20,
                        "material": material.value,
                    },
                    execution_time_seconds=time.time() - start_time,
                    metadata={
                        "algorithm": "digital_passthrough_no_reverb_fix11c",
                        "reverb_severity": _reverb_severity_ph20,
                        "skip_reason": "digital_material_below_reverb_threshold",
                    },
                )
            else:
                # reverb_severity 0.10–0.29: scale DSP strength proportionally
                _severity_scale = _reverb_severity_ph20 / self._DIGITAL_ML_REVERB_SEVERITY_MIN
                strength = float(np.clip(strength * _severity_scale, 0.0, strength))
                logger.info(
                    "Verarbeitungsschritt 20: Fix11C — ueberspringen ML-Hybrid: digital=%s, reverb_severity=%.2f → "
                    "DSP-only, scaled strength=%.2f",
                    material.value,
                    _reverb_severity_ph20,
                    strength,
                )

        # ML-Hybrid only if resources available and quality mode permits
        use_ml_hybrid = (
            ML_HYBRID_AVAILABLE
            and quality_mode in ["balanced", "maximum", "quality"]
            and not use_lightweight
            and not self._force_dsp_only_due_ml_error
            and not _skip_ml_digital_no_reverb  # Fix 11C: digital with no reverb → DSP-only
        )

        if use_ml_hybrid:
            try:
                logger.info(
                    "Verarbeitungsschritt 20 ML-Hybrid: Betriebsart=%s, material=%s", quality_mode, material.value
                )

                # Configure ML dereverb strategy
                # 'quality' und 'maximum' → HYBRID (SGMSE+ ML-Primär + WPE-DSP-Fallback, §4.4)
                if quality_mode in ("maximum", "quality"):
                    strategy = DereverbStrategy.HYBRID  # SGMSE+ primär → WPE DSP-Fallback (§4.4)
                else:  # balanced
                    strategy = DereverbStrategy.ADAPTIVE  # Smart: DSP only if light reverb

                dereverb = HybridDereverb(
                    config=DereverbConfig(
                        strategy=strategy,
                        dsp_strength=strength,
                        dsp_damping=damping,
                        enable_preprocessing=True,
                        reverb_threshold=0.3,  # Skip DCCRN if reverb already low
                    )
                )

                ml_result = dereverb.dereverb(audio, sample_rate=sample_rate)
                processing_time = time.time() - start_time

                # Estimate RMS change from reverb reduction
                # §2.45a-I: gated RMS — silence frames (fadeout, intro) excluded
                # so that long silent tails do not inflate the apparent RMS-drop
                # and trigger the catastrophic-ML-fallback guard (Fix 11A) falsely.
                rms_before = _gated_rms_20(audio)
                rms_after = _gated_rms_20(ml_result.audio)
                # Guard: np.log10(0) => RuntimeWarning; clamp ratio >= 1e-30
                rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))

                logger.info(
                    "ML-Hybrid vollstaendig: DSP=%s, ML=%s, reverb=%.3f, RMS change=%.2f dB, time=%.2f s",
                    ml_result.dsp_applied,
                    ml_result.ml_applied,
                    ml_result.reverb_estimate,
                    rms_change_db,
                    processing_time,
                )

                # Fix 11A: Catastrophic ML/WPE fallback drop guard
                # If the ML path (including WPE fallback inside HybridDereverb) caused a severe
                # drop, disable ML for PMGG retries and restrict the wet/dry blend to protect signal.
                _ml_drop_abs = abs(min(0.0, rms_change_db))
                if _ml_drop_abs > 6.0 and not self._force_dsp_only_due_ml_error:
                    # Prevent costly PMGG retry with another SGMSE+/WPE cycle
                    self._force_dsp_only_due_ml_error = True
                    self._ml_disable_reason = (
                        f"ML/WPE fallback caused {rms_change_db:.1f} dB drop — disabling ML for retries"
                    )
                    logger.warning(
                        "Verarbeitungsschritt 20: Fix11A §2.45a — catastrophic ML/WPE drop %.1f dB > 6 dB → "
                        "disable ML for PMGG retries (reason: %s)",
                        rms_change_db,
                        self._ml_disable_reason,
                    )
                # Restrict blend further when drop is severe (≥ 10 dB)
                _blend_str = _effective_strength
                if _ml_drop_abs > 10.0:
                    _blend_str = min(_effective_strength, 0.10)
                    logger.warning(
                        "Verarbeitungsschritt 20: Fix11A — extreme drop %.1f dB → capping blend to %.2f",
                        rms_change_db,
                        _blend_str,
                    )

                # Generate warnings
                warnings = []
                if ml_result.reverb_estimate > 0.7:
                    warnings.append(
                        f"High reverb detected: {ml_result.reverb_estimate:.2f} (may require multiple passes)"
                    )

                _audio_clean = np.nan_to_num(ml_result.audio, nan=0.0, posinf=0.0, neginf=0.0)
                _audio_clean = np.clip(_audio_clean, -1.0, 1.0)
                if 0.0 < _blend_str < 1.0:
                    _audio_clean = audio + _blend_str * (_audio_clean - audio)
                    _audio_clean = np.clip(_audio_clean, -1.0, 1.0)
                _audio_clean, _rms_change_db, _makeup_gain_db = self._apply_material_loudness_preservation(
                    audio,
                    _audio_clean,
                    material,
                )
                return PhaseResult(
                    success=True,
                    audio=_audio_clean,
                    metrics={
                        "rms_change_db": float(_rms_change_db),
                        "reverb_estimate": ml_result.reverb_estimate,
                        "dsp_applied": ml_result.dsp_applied,
                        "ml_applied": getattr(ml_result, "ml_applied", ml_result.dccrn_applied),
                        "strategy": str(ml_result.strategy_used),
                        "reduction_strength": strength,
                        "tail_damping": damping,
                        "material": material.value,
                        "quality_mode": quality_mode,
                    },
                    execution_time_seconds=processing_time,
                    metadata={
                        "algorithm": "hybrid_wpe_resemble_v4",
                        "ml_hybrid": True,
                        "dsp_applied": ml_result.dsp_applied,
                        "ml_applied": getattr(ml_result, "ml_applied", ml_result.dccrn_applied),
                        "reverb_estimate": ml_result.reverb_estimate,
                        "processing_time": ml_result.processing_time,
                        "version": "3.0_ml_hybrid",
                        "window_size": self.WINDOW_SIZE,
                        "hop_size": self.HOP_SIZE,
                        "ml_metadata": ml_result.metadata,
                        "phase_locality_factor": phase_locality_factor,
                        "effective_strength": _effective_strength,
                        "loudness_makeup_db": float(_makeup_gain_db),
                        "vocal_guard_active": bool(_vocal_detected_20),
                        "vocal_confidence": float(_vocal_conf_20),
                    },
                    warnings=warnings,
                    modifications={},
                )

            except Exception as e:
                import traceback as _tb  # pylint: disable=import-outside-toplevel

                _err_text = str(e)
                _is_deterministic_ml_fail = (
                    "Sizes of tensors must match" in _err_text
                    or "TorchScript" in _err_text
                    or "expected shape" in _err_text.lower()
                    or "shape mismatch" in _err_text.lower()
                )
                if _is_deterministic_ml_fail and not self._force_dsp_only_due_ml_error:
                    self._force_dsp_only_due_ml_error = True
                    self._ml_disable_reason = _err_text[:220]
                    logger.warning(
                        "Verarbeitungsschritt 20: disable ML-hybrid for remaining calls due to deterministic SGMSE error: %s",
                        self._ml_disable_reason,
                    )

                logger.warning(
                    "ML-Hybrid dereverb fehlgeschlagen: %s, falling back to DSP. Error type: %s\n%s",
                    e,
                    type(e).__name__,
                    _tb.format_exc(),
                )
                # Fall through to DSP path below

        # DSP-Only Path (Fast mode or ML fallback)
        logger.info("Verarbeitungsschritt 20 DSP-Only: material=%s, strength=%s", material.value, strength)

        # ── Tier-1 DSP: WPE (Nakatani 2010) — DSP-Fallback für Dereverb (§4.4; ML-Primär: SGMSE+) ──
        # WPE entfernt Spätreflexionen via iterative gewichtete lineare Prädiktion.
        # Kaskade: nara_wpe → NumPy-WPE → OMLSA (innerhalb des Plugins).
        if WPE_AVAILABLE:
            try:
                wpe = get_wpe_plugin()
                # WPE erwartet SR == 48000 (Spec-Invariante)
                wpe_strength = float(np.clip(strength * 0.90, 0.3, 0.95))
                audio_wpe = wpe.enhance(audio.astype(np.float32), sample_rate, strength=wpe_strength)
                audio_wpe = np.nan_to_num(audio_wpe, nan=0.0, posinf=0.0, neginf=0.0)
                audio_wpe = np.clip(audio_wpe.astype(np.float64), -1.0, 1.0)
                # Passen Länge und Form an
                if audio_wpe.shape != audio.shape:
                    if audio.ndim == 1:
                        audio_wpe = audio_wpe.flatten()[: len(audio)]
                        if len(audio_wpe) < len(audio):
                            audio_wpe = np.pad(audio_wpe, (0, len(audio) - len(audio_wpe)))
                    else:
                        min_len = min(
                            audio_wpe.shape[0] if audio_wpe.ndim == 1 else audio_wpe.shape[-1], audio.shape[-1]
                        )
                        audio_wpe = audio_wpe[:min_len] if audio_wpe.ndim == 1 else audio_wpe[:, :min_len]
                reduced = audio_wpe
                processing_time = time.time() - start_time
                rms_before = np.sqrt(np.mean(audio**2))
                rms_after = np.sqrt(np.mean(reduced**2))
                rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))
                logger.info("Verarbeitungsschritt 20: WPE-Tier erfolgreich (strength=%.2f)", wpe_strength)
                reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
                reduced = np.clip(reduced, -1.0, 1.0)
                if 0.0 < _effective_strength < 1.0:
                    reduced = audio + _effective_strength * (reduced - audio)
                    reduced = np.clip(reduced, -1.0, 1.0)
                reduced, rms_change_db, _makeup_gain_db = self._apply_material_loudness_preservation(
                    audio,
                    reduced,
                    material,
                )
                return PhaseResult(
                    success=True,
                    audio=reduced,
                    metrics={
                        "rms_change_db": float(rms_change_db),
                        "reduction_strength": strength,
                        "tail_damping": damping,
                        "material": material.value,
                    },
                    execution_time_seconds=processing_time,
                    metadata={
                        "algorithm": "wpe_nakatani2010_tier1",
                        "version": "3.0_wpe",
                        "wpe_strength": wpe_strength,
                        "phase_locality_factor": phase_locality_factor,
                        "effective_strength": _effective_strength,
                        "loudness_makeup_db": float(_makeup_gain_db),
                    },
                )
            except Exception as wpe_err:
                logger.warning("Verarbeitungsschritt 20: WPE fehlgeschlagen (%s) — OMLSA/IMCRA-Ersatzpfad", wpe_err)
        # ── Tier-2 DSP: OMLSA/IMCRA (Cohen 2002/2003) — Fallback ──────────────
        is_stereo = audio.ndim == 2

        if is_stereo:
            # Detect channel-major (2, N) vs time-major (N, 2).
            # Aurik uses channel-major (2, N) throughout the pipeline.
            _is_ch_maj = audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
            _l = audio[0] if _is_ch_maj else audio[:, 0]
            _r = audio[1] if _is_ch_maj else audio[:, 1]
            # §2.51 M/S: apply reverb suppression on Mid-channel only so both
            # channels receive the SAME gain curve — independent L/R processing
            # estimates different room impulses per channel, creating stereo-field
            # asymmetry that triggers §2.49 phase-cancellation rollbacks.
            _sqrt2 = np.sqrt(2.0)
            _mid = (_l + _r) / _sqrt2
            _side = (_l - _r) / _sqrt2
            mid_reduced = self._reduce_reverb(_mid, sample_rate, strength, damping)
            # Side: apply weaker dereverb (side already less reverberant)
            _side_str = strength * 0.5
            side_reduced = self._reduce_reverb(_side, sample_rate, _side_str, damping)
            min_len = min(len(mid_reduced), len(side_reduced))
            _l_out = (mid_reduced[:min_len] + side_reduced[:min_len]) / _sqrt2
            _r_out = (mid_reduced[:min_len] - side_reduced[:min_len]) / _sqrt2
            if _is_ch_maj:
                reduced = np.stack([_l_out, _r_out], axis=0)  # (2, N)
            else:
                reduced = np.column_stack([_l_out, _r_out])  # (N, 2)
        else:
            reduced = self._reduce_reverb(audio, sample_rate, strength, damping)

        processing_time = time.time() - start_time

        # Measure reverb reduction (RT60-like estimate)
        rms_before = np.sqrt(np.mean(audio**2))
        rms_after = np.sqrt(np.mean(reduced**2))
        # Guard: np.log10(0) => RuntimeWarning; clamp ratio >= 1e-30
        rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))

        reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
        reduced = np.clip(reduced, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            reduced = audio + _effective_strength * (reduced - audio)
            reduced = np.clip(reduced, -1.0, 1.0)
        reduced, rms_change_db, _makeup_gain_db = self._apply_material_loudness_preservation(
            audio,
            reduced,
            material,
        )

        # §4.5c Early-Reflection-Guard (Spec §4.5c, v10.0.0)
        # C80 = 10·log10(E_early80ms / E_late) — Kuttruff 2009; ΔC80 ≤ 6 dB
        # D50 = E_early50ms / E_total — ΔD50 ≤ 0.12 (sekundär)
        _c80_guard_triggered = False
        _early_blend_triggered = False
        _delta_c80 = 0.0
        _delta_d50 = 0.0
        _c80_down_lim, _c80_soft_lim, _c80_hard_lim, _d50_lim = self._adaptive_clarity_limits(kwargs)
        try:
            # §2.51 / axis-orientation: use channel-0 as 1D mono proxy.
            # audio may be channel-major (2,N) or time-major (N,2).
            if audio.ndim == 2:
                _ch_maj = audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
                _mono_in = audio[0] if _ch_maj else audio[:, 0]
                _mono_out = reduced[0] if _ch_maj else reduced[:, 0]
            else:
                _mono_in = audio
                _mono_out = reduced
            _e80 = int(sample_rate * 0.080)
            _e50 = int(sample_rate * 0.050)
            if len(_mono_in) > _e80:
                _c80_pre = 10.0 * float(
                    np.log10(max(np.sum(_mono_in[:_e80] ** 2), 1e-12) / max(np.sum(_mono_in[_e80:] ** 2), 1e-12))
                )
                _c80_post = 10.0 * float(
                    np.log10(max(np.sum(_mono_out[:_e80] ** 2), 1e-12) / max(np.sum(_mono_out[_e80:] ** 2), 1e-12))
                )
                _delta_c80 = _c80_post - _c80_pre

                # D50 measurement
                _e_total_in = max(float(np.sum(_mono_in**2)), 1e-12)
                _e_total_out = max(float(np.sum(_mono_out**2)), 1e-12)
                _d50_pre = float(np.clip(float(np.sum(_mono_in[:_e50] ** 2)) / _e_total_in, 0.0, 1.0))
                _d50_post = float(np.clip(float(np.sum(_mono_out[:_e50] ** 2)) / _e_total_out, 0.0, 1.0))
                _delta_d50 = _d50_post - _d50_pre

                if _delta_c80 < _c80_down_lim:
                    # C80 degraded → rollback to dry
                    logger.warning(
                        "Verarbeitungsschritt 20 §4.5c C80-guard: ΔC80=%.2f dB < %.2f dB → rollback",
                        _delta_c80,
                        _c80_down_lim,
                    )
                    reduced = audio.copy()
                    _c80_guard_triggered = True
                elif _delta_c80 > _c80_hard_lim:
                    # Excessive clarity boost → scale wet proportionally
                    _c80_wet_scale = float(np.clip(_c80_hard_lim / (_delta_c80 + 1e-9), 0.30, 1.0))
                    reduced = audio + _c80_wet_scale * (reduced - audio)
                    reduced = np.clip(reduced, -1.0, 1.0)
                    _c80_guard_triggered = True
                    logger.info(
                        "Verarbeitungsschritt 20 §4.5c C80-guard: ΔC80=%.2f dB > %.2f dB → wet scaled to %.2f",
                        _delta_c80,
                        _c80_hard_lim,
                        _c80_wet_scale,
                    )
                elif _delta_c80 > _c80_soft_lim:
                    # Moderate boost → blend 35 % early reflections back (spec α=0.35, 50 ms)
                    _early_win = int(sample_rate * 0.050)
                    _alpha = 0.35
                    _rd = reduced.copy().astype(np.float64)
                    _og = audio.astype(np.float64)
                    if _rd.ndim == 2:
                        for _ch in range(_rd.shape[0]):
                            _e = min(_early_win, _rd.shape[1])
                            _rd[_ch, :_e] = (1.0 - _alpha) * _rd[_ch, :_e] + _alpha * _og[_ch, :_e]
                    else:
                        _e = min(_early_win, len(_rd))
                        _rd[:_e] = (1.0 - _alpha) * _rd[:_e] + _alpha * _og[:_e]
                    reduced = np.clip(_rd.astype(np.float32), -1.0, 1.0)
                    _early_blend_triggered = True
                    logger.info(
                        "Verarbeitungsschritt 20 §4.5c C80-guard: ΔC80=%.2f dB — early-reflection blend 35 %% angewendet",
                        _delta_c80,
                    )

                # §4.5c D50 secondary guard: ΔD50 > 0.12 → reduce wet further
                if abs(_delta_d50) > _d50_lim and not _c80_guard_triggered:
                    _d50_scale = float(np.clip(_d50_lim / (abs(_delta_d50) + 1e-9), 0.30, 1.0))
                    reduced = audio + _d50_scale * (reduced - audio)
                    reduced = np.clip(reduced, -1.0, 1.0)
                    logger.info(
                        "Verarbeitungsschritt 20 §4.5c D50-guard: ΔD50=%.3f > %.3f → wet scaled to %.2f",
                        _delta_d50,
                        _d50_lim,
                        _d50_scale,
                    )
        except Exception as _c80_exc:
            logger.debug("Verarbeitungsschritt 20 C80/D50-guard uebersprungen (unkritisch): %s", _c80_exc)

        # §4.5 Psychoacoustic Masking Clamp — preserve reverb tails below masking threshold
        try:
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp  # pylint: disable=import-outside-toplevel  # noqa: I001

            reduced = apply_psychoacoustic_masking_clamp(
                audio,
                reduced,
                sample_rate,
                strength=_effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt20 masking clamp nicht blockierend: %s", _pm_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — restore protected breath zones after dereverb
        if _npa_result_20 is not None:
            try:
                _npa_n_20 = reduced.shape[0]
                _npa_mask_20 = _npa_result_20.get_protected_mask(_npa_n_20, sample_rate)
                if np.any(_npa_mask_20):
                    reduced[_npa_mask_20] = audio[_npa_mask_20]
                    logger.debug(
                        "§2.46f NPA Verarbeitungsschritt20: wiederhergestellt %d protected samples",
                        int(np.sum(_npa_mask_20)),
                    )
            except Exception as _npa_rest_20:
                logger.debug("§2.46f NPA restoration nicht blockierend: %s", _npa_rest_20)

        # §2.36 Phonem-Schutz: Nass-Anteil des Dereverb-Prozessors kann Transienten-Energie
        # von Plosiv-Bursts absenken wenn der Hüllkurven-Schätzer sie als Reverb-Einsatz
        # behandelt. Plosiv-Burst-Frames aus Original restaurieren.
        try:
            from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_pmask_20  # pylint: disable=import-outside-toplevel  # noqa: I001

            _hop_20 = 512
            _mono_20 = (
                reduced.mean(axis=0)
                if (reduced.ndim == 2 and reduced.shape[0] == 2 and reduced.shape[1] > 2)
                else (reduced.mean(axis=1) if reduced.ndim == 2 else reduced)
            )
            _pmask_20 = _get_pmask_20(_mono_20.astype(np.float32), sample_rate, hop_length=_hop_20)
            if np.any(_pmask_20):
                _n_20 = _mono_20.shape[0]
                _smask_20 = np.zeros(_n_20, dtype=bool)
                for _fi20, _fp20 in enumerate(_pmask_20):
                    if _fp20:
                        _fs20 = _fi20 * _hop_20
                        _fe20 = min(_n_20, _fs20 + _hop_20)
                        _smask_20[_fs20:_fe20] = True
                if reduced.ndim == 2 and audio.ndim == 2:
                    if reduced.shape[0] == 2 and reduced.shape[1] > 2:
                        reduced[:, _smask_20] = audio[:, _smask_20]
                    elif reduced.shape == audio.shape:
                        reduced[_smask_20, :] = audio[_smask_20, :]
                elif reduced.ndim == 1 and audio.ndim == 1:
                    reduced[_smask_20] = audio[_smask_20]
        except Exception as _pm20_exc:
            logger.debug("§2.36 Verarbeitungsschritt_20 Phonem-Mask (nicht blockierend): %s", _pm20_exc)

        # §0p [RELEASE_MUST] VQI per-Phase Gate — panns_singing >= 0.35: rollback bei VQI < 0.95
        # phase_20 kann Reverb-Tail des Gesangs durch Dereverb-Artefakte beschädigen.
        _p20_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", _vocal_conf_20)))

        # §0p Passaggio-Schutz [RELEASE_MUST]: Temporal register detection mit Passaggio-Glättung (±5 Frames).
        # In Übergangszonen (Brust→Kopf): partial blend zurück zum Original (energy_bias ≈ -3.0 dB äquivalent).
        # Dereverb in Passaggio-Zonen kann Register-Übergänge strukturell beschädigen → blend-back schützt.
        if _p20_panns >= 0.25:
            try:
                # pylint: disable-next=import-outside-toplevel
                from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal as _dvrt_p20

                _reg_seq_p20 = _dvrt_p20(audio, sample_rate, panns_singing=_p20_panns)
                _has_passaggio_p20 = len({_r for _, _, _r, _ in _reg_seq_p20}) > 1
                if _has_passaggio_p20:
                    _n_p20 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _blend_p20 = np.zeros(_n_p20, dtype=np.float32)
                    for _zs, _ze, _zr, _zb in _reg_seq_p20:
                        if _zb > -6.0:  # Übergangszone (interpoliert zwischen -6.0 und -3.0)
                            _si = min(int(_zs * sample_rate), _n_p20)
                            _ei = min(int(_ze * sample_rate), _n_p20)
                            # alpha = Anteil Kopfstimme/Passaggio (0=Brust, 1=Kopf)
                            _alpha = float(np.clip((_zb - (-6.0)) / 3.0, 0.0, 1.0))
                            _blend_p20[_si:_ei] = np.maximum(_blend_p20[_si:_ei], _alpha)
                    if audio.ndim > 1:
                        reduced = (1.0 - _blend_p20[np.newaxis, :]) * reduced + _blend_p20[
                            np.newaxis, :
                        ] * audio.astype(np.float32)
                    else:
                        reduced = (1.0 - _blend_p20) * reduced + _blend_p20 * audio.astype(np.float32)
                    logger.debug("§0p Verarbeitungsschritt_20 Passaggio blend zones=%d", len(_reg_seq_p20))
            except Exception as _pvrt20_exc:
                logger.debug("§0p Passaggio temporal Verarbeitungsschritt_20 (nicht blockierend): %s", _pvrt20_exc)

        # §0p HNR-Blend nach ML-Dereverb (RELEASE_MUST §0p): ΔHNR > 3 dB → Dry-Wet-Blend
        if _p20_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import apply_hnr_blend as _apply_hnr_p20  # pylint: disable=import-outside-toplevel  # noqa: I001

                _hnr_blended_p20, _hnr_diag_p20 = _apply_hnr_p20(
                    audio.astype(np.float32), reduced.astype(np.float32), sample_rate
                )
                if _hnr_diag_p20.get("over_cleaned"):
                    reduced = _hnr_blended_p20

            except Exception as _hnr_exc_p20:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_20 (nicht blockierend): %s", _hnr_exc_p20)

        # §0p Formant-Integrity post-check — rollback if F1 shifted >±15%
        if _f1_pre_20 is not None:
            try:
                from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker as _get_lfc_20_post  # pylint: disable=import-outside-toplevel  # noqa: I001

                _ft_out_20 = reduced.mean(axis=0) if reduced.ndim == 2 else reduced
                _f1_post_20 = float(
                    _get_lfc_20_post().track(_ft_out_20.astype(np.float32), sample_rate).get("f1_mean", 0.0)
                )
                if _f1_post_20 > 0 and abs(_f1_post_20 - _f1_pre_20) > _f1_pre_20 * 0.15:
                    logger.warning(
                        "§0p Formant drift phase_20 (F1 %.0f→%.0f Hz, delta=%.0f Hz) — rollback",
                        _f1_pre_20,
                        _f1_post_20,
                        abs(_f1_post_20 - _f1_pre_20),
                    )
                    reduced = audio.copy()
            except Exception as e:
                logger.warning("Verarbeitungsschritt_20_reverb_reduction.py::unbekannter Ersatzpfad: %s", e)
        if _p20_panns >= 0.35:
            try:
                from backend.core.musical_goals.era_vocal_profile import (
                    get_era_vocal_profile as _gevp_p20,  # pylint: disable=import-outside-toplevel  # §EraVocalProfile
                )
                from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
                    compute_vqi as _compute_vqi_p20,
                )

                _vqi_result_p20 = _compute_vqi_p20(
                    audio_orig=audio,
                    audio_restored=reduced,
                    sr=sample_rate,
                    era_profile=_gevp_p20(int(decade)) if decade is not None else None,
                )
                _vqi_p20 = float(_vqi_result_p20.get("vqi", 1.0))
                if _vqi_p20 < 0.95:
                    logger.info(
                        "Verarbeitungsschritt_20: VQI per-Verarbeitungsschritt rollback (vqi=%.3f < 0.95, panns=%.2f) — pre-dereverb bewahrt",
                        _vqi_p20,
                        _p20_panns,
                    )
                    reduced = audio.copy()
            except Exception as _vqi_exc_p20:
                logger.debug(
                    "VQI per-Verarbeitungsschritt Verarbeitungsschritt_20 (nicht blockierend): %s", _vqi_exc_p20
                )

        # §G2 (GEBOTE.md) Breath-Segment Protection (§2.46f): EMOTIONAL_TENSION Atemgeräusche
        # mit Original zurückblenden — Dereverb glättet sonst natürliche Atemräume.
        _breath_segs_p20 = list(kwargs.get("breath_segments", []) or [])
        if _breath_segs_p20:
            try:
                _n_out_p20 = reduced.shape[-1] if reduced.ndim == 2 else len(reduced)
                _n_in_p20 = audio.shape[-1] if audio.ndim == 2 else len(audio)
                _n_blend_p20 = min(_n_out_p20, _n_in_p20)
                _result_blend_p20 = np.array(reduced, copy=True)
                _blended_any_p20 = False
                for _bs_p20 in _breath_segs_p20:
                    _cat_p20 = getattr(_bs_p20, "category", None)
                    _cat_str_p20 = str(getattr(_cat_p20, "value", _cat_p20 or "")).lower()
                    if "tension" not in _cat_str_p20 and "emotional" not in _cat_str_p20:
                        continue
                    _bs_start_p20 = float(getattr(_bs_p20, "start_s", 0.0))
                    _bs_end_p20 = float(getattr(_bs_p20, "end_s", 0.0))
                    _g_fl_p20 = float(np.clip(getattr(_bs_p20, "recommended_g_floor", 0.50), 0.0, 1.0))
                    _dry_p20 = float(np.clip(_g_fl_p20, 0.05, 0.95))
                    if _bs_end_p20 <= _bs_start_p20:
                        continue
                    _si_p20 = int(round(_bs_start_p20 * sample_rate))
                    _ei_p20 = int(round(_bs_end_p20 * sample_rate))
                    _si_p20 = max(0, min(_si_p20, _n_blend_p20))
                    _ei_p20 = max(0, min(_ei_p20, _n_blend_p20))
                    if _si_p20 >= _ei_p20:
                        continue
                    if _result_blend_p20.ndim == 2 and audio.ndim == 2:
                        _result_blend_p20[:, _si_p20:_ei_p20] = (
                            _dry_p20 * audio[:, _si_p20:_ei_p20] + (1.0 - _dry_p20) * reduced[:, _si_p20:_ei_p20]
                        )
                    elif _result_blend_p20.ndim == 1 and audio.ndim == 1:
                        _result_blend_p20[_si_p20:_ei_p20] = (
                            _dry_p20 * audio[_si_p20:_ei_p20] + (1.0 - _dry_p20) * reduced[_si_p20:_ei_p20]
                        )
                    _blended_any_p20 = True
                if _blended_any_p20:
                    reduced = np.clip(np.nan_to_num(_result_blend_p20, nan=0.0), -1.0, 1.0).astype(np.float32)
                    logger.debug(
                        "§G2 (GEBOTE.md) BreathProtect Verarbeitungsschritt_20: %d tension-segs geschützt",
                        len(_breath_segs_p20),
                    )
            except Exception as _g2_p20_exc:
                logger.debug("§G2 (GEBOTE.md) BreathProtect Verarbeitungsschritt_20 nicht blockierend: %s", _g2_p20_exc)

        # V19 Noise-Textur-Invariante (§NTI): Residual nach Reverb-Reduction darf kein
        # material-fremdes Spektralprofil (Whitening) aufweisen (VERBOTEN-V19).
        _mat20_str = str(getattr(material, "value", str(material) or "unknown") or "unknown").lower()
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt20_dist_fn,
            )

            _nt20_residual = audio.astype(np.float32) - reduced.astype(np.float32)
            _nt20_dist = _nt20_dist_fn(_nt20_residual, _mat20_str, sr=sample_rate)
            if _nt20_dist > 0.25:
                reduced = (0.5 * reduced + 0.5 * audio).astype(np.float32)
                logger.warning(
                    "Verarbeitungsschritt20 V19 Noise-Textur-Dist=%.3f > 0.25 → 50%%-Blend (Träger-Textur bewahrt)",
                    _nt20_dist,
                )
        except Exception as _nt20_exc:
            logger.debug("Verarbeitungsschritt20 V19 Noise-Textur-Guard (nicht blockierend): %s", _nt20_exc)

        # V20 Mikrodynamik-Korrelation (§2.75): Frame-Energie auf voiced-Zonen ≥ 0.97
        # nach Reverb-Reduction (VERBOTEN-V20).
        if _p20_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (  # pylint: disable=import-outside-toplevel
                    frame_energy_correlation as _fec20,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr20 = _fec20(audio, reduced, sample_rate, frame_ms=10.0)
                if _corr20 < 0.97:
                    _need20 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet20 = _recommend_mkk_wet(_corr20, _p20_panns, global_need=_need20)
                    reduced = (_wet20 * reduced + (1.0 - _wet20) * audio).astype(np.float32)
                    logger.warning(
                        "Verarbeitungsschritt20 V20 Mikrodynamik-Korr=%.3f < 0.97 → wet=%.3f Blend",
                        _corr20,
                        _wet20,
                    )
            except Exception as _dyn20_exc:
                logger.debug("Verarbeitungsschritt20 V20 Mikrodynamik-Guard (nicht blockierend): %s", _dyn20_exc)

        # V21 Mindestrauschboden (§2.76): Analog-Material darf nach Reverb-Reduction keine
        # digitale Stille aufweisen — Rauschboden ist Naturalness-Marker (VERBOTEN-V21).
        if any(t in _mat20_str for t in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (  # pylint: disable=import-outside-toplevel
                    apply_noise_floor_minimum as _nfg20,
                )

                reduced = _nfg20(reduced, sample_rate, _mat20_str, original_audio=audio)
            except Exception as _nf20_exc:
                logger.debug("Verarbeitungsschritt20 V21 Noise-Floor-Guard (nicht blockierend): %s", _nf20_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_20,
            )

            _sc_result_20 = _scg_20(audio, reduced, sample_rate)
            if not _sc_result_20.ok:
                _sc_wet_20 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                reduced = (_sc_wet_20 * reduced + (1.0 - _sc_wet_20) * audio).astype(np.float32)
        except Exception as _sc_exc_20:  # pylint: disable=broad-except
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_20 spectral_color nicht blockierend: %s", _sc_exc_20
            )

        # V26 Onset-Guard (§2.77): HPSS-Onset-Fenster (0–20 ms nach Transient) dürfen durch
        # Reverb-Reduction nicht energetisch beeinflusst werden (VERBOTEN-V26).
        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opg20,
            )

            reduced = _opg20(audio, reduced, None, max_delta_db=1.5)
        except Exception as _on20_exc:
            logger.debug("Verarbeitungsschritt20 V26 Onset-Guard (nicht blockierend): %s", _on20_exc)

        # §2.72 Vibrato-Tiefe-Guard (§0p Vocal-Supremacy RELEASE_MUST): F0-Modulationstiefe
        # darf durch Reverb-Reduction nicht mehr als ±10 % reduziert werden → 50 %-Blend.
        if _p20_panns >= 0.25:
            try:
                from backend.core.dsp.vibrato_guard import (  # pylint: disable=import-outside-toplevel
                    check_vibrato_depth_preservation as _vib20,
                )

                _vib20_result = _vib20(audio, reduced, sample_rate)
                if not _vib20_result.ok:
                    reduced = (0.5 * reduced + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "Verarbeitungsschritt20 §2.72 Vibrato-Tiefe: reduction=%.1f%% > 10%% → 50%%-Blend",
                        _vib20_result.depth_reduction_pct,
                    )
            except Exception as _vib20_exc:
                logger.debug("Verarbeitungsschritt20 §2.72 Vibrato-Guard (nicht blockierend): %s", _vib20_exc)

        # ── §v10 Per-Band-Maske NACH reverb anwenden ──
        if _per_band_mask is not None:
            try:
                from backend.core.pim_phase_hook import apply_per_band_mask

                _before = audio
                _after = apply_per_band_mask(_before, _per_band_mask, sample_rate, mix=0.55)
                audio = _after
            except Exception as e:
                logger.warning("Verarbeitungsschritt_20_reverb_reduction.py::unbekannter Ersatzpfad: %s", e)

        # §2.71 Strength-Envelope: Chirurgische Dereverb
        _strength_env = kwargs.get("strength_envelope")
        if _strength_env is not None:
            try:
                from backend.core.strength_envelope import apply_strength_envelope

                _env_pre = np.asarray(reduced, dtype=np.float32)
                reduced = apply_strength_envelope(
                    processed=_env_pre,
                    original=np.asarray(audio, dtype=np.float32),
                    envelope=_strength_env,
                    sample_rate=sample_rate,
                    base_strength=_effective_strength,
                )
                if float(np.mean(np.abs(reduced - _env_pre))) > 0.001:
                    logger.info(
                        "§2.71 Envelope-Blending Verarbeitungsschritt 20: Δ=%.4f RMS",
                        float(np.mean(np.abs(reduced - _env_pre))),
                    )
            except Exception as _se_exc:
                logger.debug("§2.71 Envelope nicht blockierend: %s", _se_exc)

        return PhaseResult(
            success=True,
            audio=reduced,
            metrics={
                "rms_change_db": float(rms_change_db),
                "reduction_strength": strength,
                "tail_damping": damping,
                "material": material.value,
            },
            execution_time_seconds=processing_time,
            metadata={
                "algorithm": "stft_omlsa_imcra_cohen2003",
                "version": "3.0_omlsa",
                "window_size": self.WINDOW_SIZE,
                "hop_size": self.HOP_SIZE,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "delta_c80": float(_delta_c80),
                "c80_down_limit_db": float(_c80_down_lim),
                "c80_soft_limit_db": float(_c80_soft_lim),
                "c80_hard_limit_db": float(_c80_hard_lim),
                "d50_limit": float(_d50_lim),
                "c80_guard_triggered": _c80_guard_triggered,
                "early_blend_triggered": _early_blend_triggered,
                "loudness_makeup_db": float(_makeup_gain_db),
                "vocal_guard_active": bool(_vocal_detected_20),
                "vocal_confidence": float(_vocal_conf_20),
            },
        )

    def _rms_dbfs_gated(self, audio: np.ndarray, gate_dbfs: float = -50.0) -> float:
        """Frame-wise RMS over active music frames only (§2.45a-I)."""
        _x = np.asarray(audio, dtype=np.float32)
        _mono = np.mean(_x, axis=1) if _x.ndim == 2 else _x
        if _mono.size < 480:
            _rms = float(np.sqrt(np.mean(_mono.astype(np.float64) ** 2) + 1e-12))
            return float(20.0 * np.log10(_rms + 1e-10))
        _frame = 480
        _vals = []
        for _i in range(0, len(_mono) - _frame + 1, _frame):
            _seg = _mono[_i : _i + _frame]
            _r = float(np.sqrt(np.mean(_seg.astype(np.float64) ** 2) + 1e-12))
            _db = float(20.0 * np.log10(_r + 1e-10))
            if _db > gate_dbfs:
                _vals.append(_r * _r)
        if not _vals:
            return -96.0
        _rms = float(np.sqrt(np.mean(_vals) + 1e-12))
        return float(20.0 * np.log10(_rms + 1e-10))

    def _musical_gain_envelope(self, audio: np.ndarray, gain: float, sr: int) -> np.ndarray:
        """Wendet an: gain only on musical frames (§2.45a-II), keep silence untouched."""
        _x = np.asarray(audio, dtype=np.float32)
        _mono = np.mean(_x, axis=1) if _x.ndim == 2 else _x
        _frame = max(1, int(0.010 * sr))
        _env = np.ones(len(_mono), dtype=np.float32)
        for _i in range(0, len(_mono) - _frame + 1, _frame):
            _seg = _mono[_i : _i + _frame]
            _r = float(np.sqrt(np.mean(_seg.astype(np.float64) ** 2) + 1e-12))
            _db = float(20.0 * np.log10(_r + 1e-10))
            if _db > -50.0:
                _env[_i : _i + _frame] = gain
        if _x.ndim == 2:
            return (_x * _env[:, None]).astype(np.float32)  # type: ignore[no-any-return]
        return (_x * _env).astype(np.float32)  # type: ignore[no-any-return]

    def _apply_material_loudness_preservation(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, float, float]:
        material_key = getattr(material, "value", getattr(material, "name", str(material))).lower()
        max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(material_key, self._MAX_RMS_DROP_DB["unknown"]))

        rms_in_db = self._rms_dbfs_gated(original_audio)
        rms_out_db = self._rms_dbfs_gated(processed_audio)
        rms_change_db = rms_out_db - rms_in_db if rms_in_db > -80.0 else 0.0
        makeup_gain_db = 0.0

        if rms_in_db > -80.0 and rms_change_db < -max_rms_drop_db:
            target_rms_change_db = -max_rms_drop_db
            required_gain_db = target_rms_change_db - rms_change_db
            # §2.45a-II fix: apply full gain — peak-headroom cap disabled (see phase_05 fix).
            # §2.45a-III: soft-limiter only when peak99 > 0.98.
            makeup_gain_db = float(np.clip(required_gain_db, 0.0, 6.0))
            if makeup_gain_db > 0.0:
                _gain = float(10.0 ** (makeup_gain_db / 20.0))
                # §2.45a-II: canonical apply_musical_gain_envelope (audio_utils) with
                # adaptive noise-floor gate — prevents Pegelexplosion in vinyl/shellac
                # silent sections (surface noise ~-40 dBFS > fixed -50 dBFS gate).
                from backend.core.audio_utils import apply_musical_gain_envelope as _amge_20  # pylint: disable=import-outside-toplevel  # noqa: I001
                from backend.core.audio_utils import compute_signal_relative_gate_dbfs as _sig_gate_20  # pylint: disable=import-outside-toplevel

                # §2.45a-II: signal-relative gate — CEDAR/iZotope RX approach (v10.0.0)
                _gate_dbfs_20 = _sig_gate_20(original_audio, material_key=material_key)
                processed_audio = _amge_20(
                    processed_audio,
                    _gain,
                    gate_dbfs=_gate_dbfs_20,
                    crossfade_ms=10.0,
                    sr=48000,
                    reference_for_gate=original_audio,
                )
                processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
                current_peak = float(np.percentile(np.abs(processed_audio), 99.9))
                if current_peak > 0.98:
                    _abs_20 = np.abs(processed_audio)
                    _over_20 = _abs_20 > 0.92
                    if np.any(_over_20):
                        _sign_20 = np.sign(processed_audio)
                        processed_audio = np.where(
                            _over_20, _sign_20 * (0.92 + 0.08 * np.tanh((_abs_20 - 0.92) / 0.08)), processed_audio
                        )
                processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
                rms_out_db = self._rms_dbfs_gated(processed_audio)
                rms_change_db = rms_out_db - rms_in_db if rms_in_db > -80.0 else 0.0
                logger.info(
                    "Verarbeitungsschritt 20 loudness-preservation: material=%s rms_change=%.2f dB via makeup %.2f dB",
                    material_key,
                    rms_change_db,
                    makeup_gain_db,
                )

        return processed_audio, float(rms_change_db), float(makeup_gain_db)

    def _reduce_reverb(self, audio: np.ndarray, sample_rate: int, strength: float, damping: float) -> np.ndarray:
        """Nachhall-Reduktion via STFT-OMLSA/IMCRA (Cohen 2002/2003).

        Algorithmus:
            1. scipy.signal.stft  — OLA-konsistent (ersetzt np.fft.rfft-Schleife)
            2. IMCRA gleitendes Minimum (Cohen 2003):
               σ²_d(t,f) = b_min · min_{t'∈[t-M,t]} S̃(t',f), b_min=1.66
            3. DD a-priori SNR:
               ξ̂(t,f) = α·G²(t-1,f)·γ(t-1,f) + (1-α)·max(γ(t,f)-1, 0)
            4. OMLSA Gain:
               G(t,f) = G_floor^(1−p(t,f)) · (ξ̂/(1+ξ̂))^p(t,f), Clip [G_floor,1]
            5. Cappé Temporal-Smoothing: Ĝ_t = α_g·Ĝ_{t-1} + (1-α_g)·G_t
            6. scipy.signal.istft  — phasenkonsistente PGHI-Rekonstruktion
            7. Transientenerhalt: Original zurückgemischt wo transient_mask > 0.5
            8. nan_to_num + clip[-1, 1]

        Forschungsreferenz:
            Cohen & Berdugo (2002) Signal Processing Letters
            Cohen (2003) IEEE Trans. Speech Audio Process.
            Le Roux & Vincent (2013) Consistent Wiener Filtering

        Args:
            audio:       1D float32, normalisiert auf [-1, 1]
            sample_rate: Abtastrate in Hz (intern 48 000 Hz)
            strength:    Reduktionsstärke ∈ [0.0, 1.0] (materialadaptiv)
            damping:     Nachhall-Dämpfungs-Prior (beeinflusst G_floor)

        Returns:
            np.ndarray: Restauriertes Audio, gleiche Länge wie Eingang, clip[-1, 1].
        """
        len(audio)

        # ── 1. Transientenerkennung (Sample-Ebene, vor STFT) ─────────────────
        transient_mask_raw = self._detect_transients(audio, sample_rate)

        # ── 2–7. MRSA Multi-Resolution Spectral Analysis OMLSA/IMCRA (§DSP-Spezialregeln) ─
        # Replaces single-STFT OMLSA with 5-zone optimal-resolution processing + PGHI.
        audio_out = self._reduce_reverb_mrsa(audio, sample_rate, strength, damping)

        # ── 8. Transientenerhalt ───────────────────────────────────────────────
        # Transient-Maske auf Sample-Ebene hochsampeln
        transient_up = signal.resample(transient_mask_raw, len(audio_out))
        transient_up = np.clip(transient_up, 0.0, 1.0)
        audio_out = audio_out * (1.0 - transient_up) + audio[: len(audio_out)] * transient_up
        audio_out = np.clip(audio_out, -1.0, 1.0)

        return audio_out  # type: ignore[no-any-return]

    def _reduce_reverb_mrsa(self, audio: np.ndarray, sample_rate: int, strength: float, damping: float) -> np.ndarray:  # pylint: disable=unused-argument
        """MRSA 5-zone OMLSA/IMCRA reverb reduction with PGHI phase reconstruction.

        Multi-Resolution Spectral Analysis (MRSA): each frequency zone is processed
        at its optimal time-frequency resolution using a zone-specific STFT window.
        Per-zone OMLSA/IMCRA gains are interpolated (frequency & time) to the
        reference STFT grid and blended with Hanning-weighted crossfades at zone
        boundaries. Final audio is synthesised via PGHI (Perraudin 2013).

        Zone definitions (mandatory, §DSP-Spezialregeln):
            sub_bass:  win=65536, hop=16384, 0–250 Hz
            mid_low:   win=16384, hop=4096,  250–2500 Hz
            mid:       win=8192,  hop=2048,  2500–8000 Hz
            presence:  win=1024,  hop=256,   8000–16000 Hz
            air:       win=128,   hop=32,    16000–24000 Hz

        Args:
            audio:       Mono float32 [-1, 1], SR=48000.
            sample_rate: Sample rate (must be 48000).
            strength:    Reduction strength ∈ [0.0, 1.0].
            damping:     Reverb damping prior (influences G_floor).

        Returns:
            np.ndarray: Restored audio, same length as input, clipped to [-1, 1].
        """
        n_audio = len(audio)
        nyquist = float(sample_rate // 2)

        # §2.63 Reflect-Padding: VOR STFT (root-cause boundary fix, §2.63)
        # Prevents intro/outro artefacts from STFT-boundary mismatch.
        _pad_len_20 = 2048  # REF_WIN context on each side
        _audio_padded_20 = np.pad(audio, _pad_len_20, mode="reflect")

        # Reference STFT (win=2048, 75 % overlap) — same as original _reduce_reverb
        REF_WIN = 2048
        REF_HOP = REF_WIN - self.WINDOW_SIZE + self.HOP_SIZE  # preserves original 512-hop
        REF_NOVERLAP = REF_WIN - REF_HOP

        f_ref, _, Zxx_ref = safe_stft(
            _audio_padded_20,  # §2.63: reflect-padded input
            fs=sample_rate,
            window="hann",
            nperseg=REF_WIN,
            noverlap=REF_NOVERLAP,
            boundary="even",
            padded=True,
        )
        n_bins, n_t = f_ref.shape[0], Zxx_ref.shape[1]

        # OMLSA hyper-parameters
        G_floor = float(np.clip(0.1 + (1.0 - strength) * 0.05, 0.10, 0.15))  # hard min §2.62
        q = float(np.clip(strength * 0.60, 0.10, 0.80))
        b_min = 1.66  # IMCRA bias correction (Cohen 2003)
        alpha_g = 0.85  # Cappé smoothing (1994)

        # §2.62 Psychoakustischer Masking-Guard (ISO 11172-3) — per-Band Floor
        _masking_floor_p20: np.ndarray | None = None
        _masking_freqs_p20: np.ndarray | None = None
        try:
            from backend.core.dsp.psychoacoustics import compute_masking_threshold_iso11172 as _cmask_p20  # pylint: disable=import-outside-toplevel  # noqa: I001

            _src_p20 = audio.mean(axis=0) if audio.ndim == 2 else audio
            _mask_ratio_p20 = _cmask_p20(_src_p20, sample_rate, n_fft=2048, hop_length=512)
            _masking_floor_p20 = np.mean(_mask_ratio_p20, axis=1).astype(np.float32)
            _masking_freqs_p20 = np.linspace(0.0, sample_rate / 2.0, _mask_ratio_p20.shape[0], dtype=np.float32)
            _mf_mean = float(np.mean(_masking_floor_p20))  # type: ignore[arg-type]
            logger.debug("§2.62 Verarbeitungsschritt_20 Masking-Guard: mean_floor=%.3f", _mf_mean)
        except Exception as _msk20_exc:
            logger.debug(
                "§2.62 Verarbeitungsschritt_20 Masking-Guard nicht verfügbar (nicht blockierend): %s", _msk20_exc
            )

        # Accumulate weighted zone gains
        G_acc = np.zeros((n_bins, n_t), dtype=np.float64)
        w_acc = np.zeros(n_bins, dtype=np.float64)

        for zone_name, zone_win, zone_hop, f_low, f_high in self._MRSA_ZONES:
            try:
                # Use zone-specific STFT if audio is long enough
                if n_audio >= zone_win * 2:
                    zone_noverlap = zone_win - zone_hop
                    f_z, _, Zxx_z = safe_stft(
                        _audio_padded_20,  # §2.63: reflect-padded input
                        fs=sample_rate,
                        window="hann",
                        nperseg=zone_win,
                        noverlap=zone_noverlap,
                        boundary="even",
                        padded=True,
                    )
                else:
                    f_z, Zxx_z = f_ref, Zxx_ref
                    zone_win, zone_hop = REF_WIN, REF_HOP

                mag_z = np.abs(Zxx_z)  # (F_z, T_z)
                n_z_t = mag_z.shape[1]

                # Vectorised IMCRA noise estimation: sliding-minimum (Cohen 2003)
                frames_per_sec_z = float(sample_rate / zone_hop)
                M_z = max(3, int(1.5 * frames_per_sec_z))
                power_z = mag_z**2
                # minimum_filter1d is fast C-code (no Python frame loop)
                S_min_z = _min_filter1d_p20(power_z, size=M_z, axis=1, mode="reflect")
                noise_sq_z = np.maximum(b_min * S_min_z, 1e-12)

                # Vectorised OMLSA gain (Cohen 2003, no Decision-Directed recursion needed
                # because the sliding-min noise estimator already provides a stable σ²_d)
                gamma_z = power_z / noise_sq_z
                xi_z = np.maximum(gamma_z - 1.0, 0.0)
                nu_z = np.clip(xi_z * gamma_z / (xi_z + 1.0 + 1e-12), 0.0, 500.0)
                log_lambda_z = -np.log1p(xi_z + 1e-12) + nu_z
                Lambda_z = np.exp(np.clip(log_lambda_z, -50.0, 50.0))
                p_H1_z = np.clip(
                    Lambda_z / (1.0 + Lambda_z + 1e-12) / (1.0 + q / ((1.0 - q) * Lambda_z + 1e-12)), 0.0, 1.0
                )
                G_wiener_z = xi_z / (xi_z + 1.0 + 1e-12)
                # MMSE-LSA (Ephraim-Malah 1985, §DSP-Instructions): G = G_H1 * exp(0.5 * E1(ν))
                G_mmse_lsa_z = G_wiener_z * _exp1_p20_gain(nu_z)
                G_mmse_lsa_z = np.clip(np.nan_to_num(G_mmse_lsa_z, nan=G_floor), G_floor, 1.0)
                log_G_z = (1.0 - p_H1_z) * np.log(G_floor + 1e-10) + p_H1_z * np.log(np.maximum(G_mmse_lsa_z, 1e-10))
                G_z = np.exp(np.clip(log_G_z, np.log(G_floor + 1e-10), 0.0))
                G_z = np.clip(G_z, G_floor, 1.0)

                # §2.62: Per-Frequenz-Masking-Floor anwenden (non-blocking)
                if _masking_floor_p20 is not None and _masking_freqs_p20 is not None:
                    try:
                        _mfloor_z20 = np.interp(f_z, _masking_freqs_p20, _masking_floor_p20).astype(np.float32)
                        G_z = np.maximum(G_z, _mfloor_z20[:, np.newaxis])
                    except Exception as e:
                        logger.warning("Verarbeitungsschritt_20_reverb_reduction.py::unbekannter Ersatzpfad: %s", e)

                # §4.8a-ii preserve_mask (§Gap8 v10.0.0): G_eff = mask*0.90 + (1-mask)*G_z
                # Bewahrt Shellac-H2/H4-Wärme und Vinyl-Charakter während Nachhall-Reduktion.
                _pm_p20 = getattr(self, "_preserve_mask_p20", None)
                if _pm_p20 is not None and _pm_p20.size > 0:
                    try:
                        _n_bins_z20 = G_z.shape[0]
                        if len(_pm_p20) != _n_bins_z20:
                            _pm_z20 = np.interp(
                                np.arange(_n_bins_z20),
                                np.linspace(0, _n_bins_z20 - 1, len(_pm_p20)),
                                _pm_p20.astype(np.float64),
                            ).astype(np.float64)
                        else:
                            _pm_z20 = _pm_p20.astype(np.float64)
                        _pm_col_z20 = _pm_z20[:, np.newaxis]
                        G_z = _pm_col_z20 * 0.90 + (1.0 - _pm_col_z20) * G_z
                        G_z = np.clip(G_z, G_floor, 1.0)
                    except Exception as _pm_exc:
                        logger.debug("§Gap8 preserve_mask Verarbeitungsschritt_20 nicht blockierend: %s", _pm_exc)

                # Cappé temporal smoothing via fast IIR lfilter (no Python loop)
                G_z_sm = _lfilter_p20([1.0 - alpha_g], [1.0, -alpha_g], G_z, axis=1)
                G_z_sm = np.clip(np.nan_to_num(G_z_sm, nan=G_floor), G_floor, 1.0)

                # Extract zone frequency range from zone STFT
                zm_z = (f_z >= float(f_low)) & (f_z <= float(f_high))
                if not np.any(zm_z):
                    continue
                f_z_zone = f_z[zm_z]
                G_zone = G_z_sm[zm_z, :]  # (n_zone_bins, n_z_t)

                # Reference STFT bins for this zone (extended by crossfade bandwidth)
                ref_zm = (f_ref >= max(0.0, float(f_low) - self._MRSA_CROSSFADE_BW_HZ)) & (
                    f_ref <= min(nyquist, float(f_high) + self._MRSA_CROSSFADE_BW_HZ)
                )
                if not np.any(ref_zm):
                    continue
                f_ref_zone = f_ref[ref_zm]
                ref_indices = np.where(ref_zm)[0]
                n_ref_zone = len(ref_indices)

                # Temporal resampling: zone frames → reference frames — vectorised
                if n_z_t != n_t and len(f_z_zone) > 0:
                    _idx_c = np.linspace(0.0, n_z_t - 1, n_t)
                    _idx_lo = np.clip(np.floor(_idx_c).astype(int), 0, n_z_t - 2)
                    _idx_hi = _idx_lo + 1
                    _frac = (_idx_c - _idx_lo)[np.newaxis, :]  # (1, n_t)
                    G_zone_t = ((1.0 - _frac) * G_zone[:, _idx_lo] + _frac * G_zone[:, _idx_hi]).astype(np.float64)
                else:
                    G_zone_t = G_zone.astype(np.float64)

                # Frequency interpolation: zone bins → reference bins — vectorised
                G_ref_zone = np.empty((n_ref_zone, n_t), dtype=np.float64)
                if len(f_z_zone) >= 2:
                    _src_x = np.asarray(f_z_zone, dtype=np.float64)
                    _dst_x = np.asarray(f_ref_zone, dtype=np.float64)
                    _i = np.searchsorted(_src_x, _dst_x, side="right") - 1
                    _i_lo = np.clip(_i, 0, len(_src_x) - 2)
                    _i_hi = _i_lo + 1
                    _dx = _src_x[_i_hi] - _src_x[_i_lo]
                    _frac2 = np.clip((_dst_x - _src_x[_i_lo]) / np.maximum(_dx, 1e-12), 0.0, 1.0)
                    _left = _dst_x < _src_x[0]
                    _right = _dst_x > _src_x[-1]
                    _frac2[_left] = 0.0
                    _frac2[_right] = 1.0
                    _i_lo[_left] = 0
                    _i_hi[_left] = 0
                    _i_lo[_right] = len(_src_x) - 1
                    _i_hi[_right] = len(_src_x) - 1
                    G_ref_zone = (1.0 - _frac2[:, None]) * G_zone_t[_i_lo, :] + _frac2[:, None] * G_zone_t[_i_hi, :]
                elif len(f_z_zone) == 1:
                    G_ref_zone[:, :] = G_zone_t[0:1, :]
                else:
                    continue

                # Hanning crossfade weights at zone boundaries
                if n_ref_zone > 2:
                    hann_w = np.hanning(n_ref_zone + 2)[1:-1]
                    hann_w = np.clip(hann_w, 1e-3, 1.0)
                else:
                    hann_w = np.ones(n_ref_zone)

                # Vectorised accumulation (replaces per-bin Python loop)
                G_acc[ref_indices, :] += hann_w[:, None] * G_ref_zone
                w_acc[ref_indices] += hann_w

            except Exception as zone_exc:
                logger.warning("MRSA Verarbeitungsschritt 20 zone '%s' fehlgeschlagen: %s", zone_name, zone_exc)
                continue

        # Combine zone gains; unprocessed bins → pass-through (gain=1.0)
        valid = w_acc > 0.0
        G_combined = np.ones((n_bins, n_t), dtype=np.float32)
        G_combined[valid, :] = (G_acc[valid, :] / w_acc[valid, np.newaxis]).astype(np.float32)
        G_combined = np.clip(np.nan_to_num(G_combined, nan=1.0), 0.0, 1.0)

        # Late-reverb temporal decay suppression (v10.0.0):
        # Room reverberation produces exponentially decaying tails after transients;
        # OMLSA alone treats all time-frames equally and cannot separate the
        # reverberant tail from the direct sound.  We add a time-varying secondary
        # gain that suppresses frames identified as part of a reverberant decay.
        # Ref: Noh & Hwang 2014; Braun & Haardt 2016 — spectral late-reverb model.
        if strength > 0.15 and n_t > 8:
            try:
                # Per-frame mean energy from reference STFT (linear scale)
                E_frame = np.mean(np.abs(Zxx_ref) ** 2, axis=0)  # shape (n_t,)
                E_frame = np.maximum(E_frame, 1e-15)
                E_log_db = 10.0 * np.log10(E_frame)  # dB per frame

                # Frame-to-frame delta energy (positive = rising, negative = decaying)
                dE = np.diff(E_log_db, prepend=E_log_db[0])  # shape (n_t,)

                # Smooth dE to suppress single-sample noise spikes
                _sm = max(3, min(7, n_t // 20))
                _kern = np.ones(_sm, dtype=np.float32) / _sm
                dE_smooth = np.convolve(dE.astype(np.float32), _kern, mode="same")

                # Decay mask: frames where energy is steadily dropping > 0.5 dB/hop
                decay_mask = (dE_smooth < -0.5).astype(np.float32)  # shape (n_t,)

                # Direct-sound protection window (~40 ms after each onset):
                # onset frames and the immediately following window are exempted
                # so direct attack transients are never suppressed.
                _prot = max(1, int(0.040 * sample_rate / REF_HOP))
                _onset_indices = np.where(dE > 2.0)[0]  # onset = energy rise > 2 dB
                for _oi in _onset_indices:
                    _end = min(n_t, int(_oi) + _prot)
                    decay_mask[int(_oi) : _end] = 0.0

                # Extra gain reduction in decay frames; strength-scaled.
                # Maximum penalty 35 % at full strength → never below -4.4 dB (G_lr ≥ 0.60).
                _penalty = float(np.clip(strength * 0.35, 0.0, 0.35))
                G_lr = np.clip(1.0 - _penalty * decay_mask, 0.60, 1.0).astype(np.float32)

                # Broadcast: (n_bins, n_t) × (n_t,) → shape-safe
                G_combined = np.clip(G_combined * G_lr[np.newaxis, :], 0.0, 1.0)

                logger.debug(
                    "MRSA Verarbeitungsschritt 20 late-reverb suppression: penalty=%.2f, decay_frames=%d/%d, onset_protected=%d",
                    _penalty,
                    int(np.sum(decay_mask > 0)),
                    n_t,
                    len(_onset_indices),
                )
            except Exception as _lr_exc:
                logger.debug("MRSA Verarbeitungsschritt 20 late-reverb suppression uebersprungen: %s", _lr_exc)

        # Apply combined gain to reference STFT — Zxx_processed retains original phase.
        # Direct ISTFT is semantically correct and 50-100× faster than PGHI.
        Zxx_processed = G_combined * np.abs(Zxx_ref) * np.exp(1j * np.angle(Zxx_ref))
        try:
            _, audio_out = signal.safe_istft(
                np.asarray(Zxx_processed, dtype=np.complex64),
                fs=sample_rate,
                window="hann",
                nperseg=REF_WIN,
                noverlap=REF_NOVERLAP,
                boundary="even",
            )
        except Exception as _istft_p20_exc:
            logger.warning("Verarbeitungsschritt_20 istft fehlgeschlagen (unkritisch): %s", _istft_p20_exc)
            audio_out = np.zeros(n_audio, dtype=np.float32)

        audio_out = np.real(audio_out).astype(np.float32)
        # §2.63: Strip reflect-padding deterministisch (Originallänge wiederherstellen)
        audio_out = audio_out[_pad_len_20 : _pad_len_20 + n_audio]
        if len(audio_out) > n_audio:
            audio_out = audio_out[:n_audio]
        elif len(audio_out) < n_audio:
            audio_out = np.pad(audio_out, (0, n_audio - len(audio_out)))

        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        logger.debug(
            "MRSA Verarbeitungsschritt 20: 5 zones verarbeitet, valid_bins=%d/%d, G_mean=%.3f",
            int(np.sum(valid)),
            n_bins,
            float(np.mean(G_combined)),
        )
        return audio_out  # type: ignore[no-any-return]

    def _detect_transients(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Erkennt transients using energy envelope.

        Returns:
            Binary mask (1 = transient, 0 = sustain/decay)
        """
        # Calculate energy envelope (RMS in short windows)
        window_samples = int(0.01 * sample_rate)  # 10ms windows
        hop_samples = window_samples // 2

        num_windows = (len(audio) - window_samples) // hop_samples + 1
        energy = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            window = audio[start:end]
            energy[i] = np.sqrt(np.mean(window**2))  # Fix: assign to energy array

        # Detect transients as rapid energy increases
        transient_mask = np.zeros(num_windows)
        for i in range(1, num_windows):
            if energy[i] > self.TRANSIENT_THRESHOLD * energy[i - 1]:
                # Transient detected
                transient_mask[i] = 1.0
                # Extend mask for attack phase (20ms)
                extend_frames = int(0.02 * sample_rate / hop_samples)
                transient_mask[i : min(i + extend_frames, num_windows)] = 1.0

        return transient_mask  # type: ignore[no-any-return]


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Verarbeitungsschritt 20: Professional Reverb Reduction v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio with synthetic reverb
    duration = 3.0
    _test_sr = 44100
    t = np.linspace(0, duration, int(_test_sr * duration))
    dry_signal = np.zeros_like(t)

    # Add impulses at 0.5s intervals
    for impulse_time in np.arange(0, duration, 0.5):
        impulse_sample = int(impulse_time * _test_sr)
        if impulse_sample < len(dry_signal):
            dry_signal[impulse_sample : impulse_sample + 100] = 0.8 * np.exp(-np.arange(100) / 20)

    # Add musical content
    dry_signal += 0.2 * np.sin(2 * np.pi * 440 * t)

    # Add synthetic reverb (exponential decay of signal)
    reverb_tail = signal.lfilter([1], [1, -0.7], dry_signal)  # Simple comb filter
    reverbed_signal = dry_signal + 0.4 * reverb_tail

    logger.debug("erzeugt %ss test audio @ %s Hz", duration, _test_sr)
    logger.debug("Dry signal + synthetic reverb tail")
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE"),
        (MaterialType.VINYL, "VINYL"),
        (MaterialType.CD_DIGITAL, "CD_DIGITAL"),
    ]

    for _test_material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug("Material: %s", material_name)
        logger.debug("─" * 80)
        logger.debug("")

        phase = ReverbReduction()
        result = phase.process(reverbed_signal, _test_sr, _test_material)

        logger.debug("✅ Professional Reverb Reduction:")
        logger.debug("   RMS Change: %.2f dB", result.metrics["rms_change_db"])
        logger.debug("   Reduction Strength: %.2f", result.metrics["reduction_strength"])
        logger.debug("   Tail Damping: %.2f", result.metrics["tail_damping"])
        logger.debug(
            "   Processing time: %.3f s (%.2f× realtime)",
            result.execution_time_seconds,
            result.execution_time_seconds / duration,
        )
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test abgeschlossen")
    logger.debug("=" * 80)
