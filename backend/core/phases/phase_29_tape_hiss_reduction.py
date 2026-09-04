#!/usr/bin/env python3
"""
Phase 29: Tape Hiss Reduction v3.1 - Über-SOTA OMLSA/IMCRA + MMSE-LSA
Adaptive HF-Rauschunterdrückung für Tape-Aufnahmen via spektraler OMLSA/IMCRA-Verarbeitung.

Algorithmus (v3.1):
1. STFT (nperseg=2048, 75% Overlap) des gesamten Signals
2. IMCRA-Rauschschätzung (Cohen & Berdugo 2002):
   - Bias-korrigiertes gleitendes Minimum im HF-Bereich
   - b_min=1.66, alpha_n=0.85, Fenster ~1.5s
3. MMSE-LSA-Gain (Ephraim-Malah 1985, §DSP-Instructions):
   - G_H1(t,f) = xi/(1+xi)  [Wiener-Estimate]
   - G_mmse_lsa(t,f) = G_H1 * exp(0.5 * E1(nu))  [MMSE-LSA — verhindert Musical Noise]
   - G(t,f) = G_floor^(1-p) * G_mmse_lsa^p  [OMLSA-Wrapper für Speech-Presence]
   - HF-selektiv: Bins < hf_low erhalten G=1.0 (unangetastet)
   - Bins >= hf_low: MMSE-LSA-Gain mit materialadaptivem G_floor
4. Cappé-Gain-Glättung (1994): temporal geglättet
5. ISTFT + NaN-Schutz + clip[-1, 1]
6. ML-Hybrid: DeepFilterNet v3 II für Residual-Hiss >2kHz (optional)

Scientific Foundation:
- Cohen & Berdugo (2002): IMCRA — primär
- Cohen (2003): OMLSA — primär
- Cappé (1994): Elimination of the Musical Noise Phenomenon — Gain-Glättung
- Le Roux & Vincent (2013): Consistent Wiener Filtering — Phasenkonsistenz
- Überholt (NICHT primär): einfacher Percentile-Gate, Bandpass-Expander-Kette

Author: Aurik Development Team
Version: 2.0.0 Professional ML-Hybrid
"""

import logging
import os
import tempfile
import time
from typing import cast

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d as _min_filter1d_p29  # vectorised sliding-min
from scipy.signal import lfilter as _lfilter_p29  # vectorised IIR smoothing (Cappé 1994)

# §DSP-Instructions: MMSE-LSA Gain (Ephraim-Malah 1985) — E1 = exponential integral
try:
    from scipy.special import exp1 as _scipy_exp1_p29

    def _exp1_p29_gain(nu: np.ndarray) -> np.ndarray:
        """MMSE-LSA gain factor exp(0.5 * E1(ν)) per Ephraim-Malah (1985).

        Reduces Musical Noise by preventing over-suppression of low-SNR bins:
        - For ν → 0  (low SNR):  E1(ν) → large → gain > Wiener (less suppression)
        - For ν → ∞  (high SNR): E1(ν) → 0    → gain ≈ Wiener
        All gains are subsequently clamped to [G_floor, 1.0].
        """
        return cast(
            np.ndarray,
            np.asarray(np.exp(np.clip(0.5 * _scipy_exp1_p29(np.maximum(nu, 1e-10)), 0.0, 5.0)), dtype=np.float64),
        )

except ImportError as _exp1_import_err:  # pragma: no cover
    logger.debug("§V6 scipy.special.exp1 nicht verfügbar — Identity-Gain Fallback aktiviert (phase_29): %s", _exp1_import_err)

    def _exp1_p29_gain(nu: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """Fallback: identity = degenerate Wiener gain (scipy.special unavailable)."""
        fallback_gain: np.ndarray = np.ones_like(nu)
        return fallback_gain


from backend.core.audio_utils import (
    apply_musical_gain_envelope,
    compute_signal_relative_gate_dbfs,
    restore_layout,
    to_channels_last,
)
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.restoration_policy import get_effective_song_goal_weights

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# ML-Hybrid Support
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from backend.core.quality_mode import QualityMode

    QUALITY_MODE_AVAILABLE = True
except ImportError:
    QUALITY_MODE_AVAILABLE = False

try:
    _PGHI_AVAILABLE_P29 = True
except ImportError:  # pragma: no cover
    _PGHI_AVAILABLE_P29 = False

logger = logging.getLogger(__name__)


def _get_phase29_npd():
    """Stabiler Resolver fuer den NPA-Singleton in Phase 29."""
    from backend.core.natural_performance_detector import get_natural_performance_detector

    return get_natural_performance_detector()


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    if sig.ndim == 2:
        _mono = (
            sig.mean(axis=0).astype(np.float64)
            if (sig.shape[0] <= 2 and sig.shape[1] > 2)
            else sig.mean(axis=1).astype(np.float64)
        )
    else:
        _mono = sig.astype(np.float64)
    _frame = 480  # 10 ms @ 48 kHz
    _active = [
        _mono[i : i + _frame]
        for i in range(0, len(_mono) - _frame, _frame)
        if 20.0 * np.log10(np.sqrt(np.mean(_mono[i : i + _frame] ** 2)) + 1e-10) > -50.0
    ]
    if not _active:
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean([np.mean(r**2) for r in _active])) + 1e-10))


class TapeHissReductionPhase(PhaseInterface):
    """
    Enhanced tape hiss reduction with adaptive gates and ML-Hybrid Support.

    Tape hiss is characterized by:
    - High-frequency noise (primarily >8 kHz)
    - Stationary (constant noise floor)
    - Gaussian distribution

    Strategy:
    1. Split into frequency bands (8 bands above 4 kHz)
    2. Estimate noise floor per band
    3. Apply adaptive expander gate per band
    4. Smooth gate action (attack/release)
    5. Reconstruct with preserved phase
    6. ML-Hybrid: <2kHz DSP → >2kHz ML DeepFilterNet refinement

    Material Adaptation:
    - Tape: Moderate reduction (primary target)
    - Shellac/Vinyl: Light (mainly surface noise, handled by phase_28)
    - CD/Streaming: Disabled
    """

    # ML frequency band threshold (Hz)
    ML_FREQUENCY_THRESHOLD_HZ = 2000  # <2kHz: DSP, >2kHz: ML optional

    # MRSA Multi-Resolution Spectral Analysis zones (mandatory, §DSP-Spezialregeln)
    _MRSA_ZONES: tuple = (
        # (name,       win_size, hop_size, f_low_hz, f_high_hz)
        ("sub_bass", 65536, 16384, 0, 250),
        ("mid_low", 16384, 4096, 250, 2500),
        ("mid", 8192, 2048, 2500, 8000),
        ("presence", 1024, 256, 8000, 16000),
        ("air", 128, 32, 16000, 24000),
    )
    _MRSA_CROSSFADE_BW_HZ: float = 100.0

    # Hiss reduction threshold (dB above noise floor to start gating)
    GATE_THRESHOLD_DB = {
        MaterialType.SHELLAC: -3,  # Extra conservative for vocal transparency
        MaterialType.VINYL: -8,
        MaterialType.TAPE: -10,  # More aggressive
        MaterialType.CASSETTE: -8,  # §SibilantProtect: konservativer als TAPE; Sibilantenbereich schonen
        MaterialType.CD_DIGITAL: -999,  # Disabled
        MaterialType.STREAMING: -999,
    }

    # Reduction depth (dB to attenuate below threshold)
    REDUCTION_DEPTH_DB = {
        MaterialType.SHELLAC: 4,
        MaterialType.VINYL: 8,
        MaterialType.TAPE: 12,  # Aggressive for tape
        MaterialType.CD_DIGITAL: 0,
        MaterialType.STREAMING: 0,
    }

    # HF focus range (Hz) - where to apply reduction most aggressively
    HF_FOCUS_RANGE = {
        MaterialType.SHELLAC: (7500, 12000),
        MaterialType.VINYL: (8000, 15000),
        MaterialType.TAPE: (8000, 18000),  # Tape hiss dominates 8-18 kHz
        MaterialType.CASSETTE: (9000, 16000),  # §SibilantProtect: hf_low=9 kHz schützt Sibilantenzone (4-9 kHz)
        MaterialType.CD_DIGITAL: (0, 0),
        MaterialType.STREAMING: (0, 0),
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

    @staticmethod
    def _limit_quiet_zone_boost(
        reference_audio: np.ndarray,
        candidate_audio: np.ndarray,
        sample_rate: int,
        material_key: str,
        max_quiet_boost_db: float = 2.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Limit energy added by hiss reduction in quiet structural regions."""
        ref = np.asarray(reference_audio, dtype=np.float32)
        cand = np.asarray(candidate_audio, dtype=np.float32).copy()
        if ref.shape != cand.shape or ref.size == 0:
            return cand, {"quiet_zone_limited_frames": 0.0, "quiet_zone_max_delta_db": 0.0}

        if ref.ndim == 2:
            ref_mono = ref.mean(axis=1) if ref.shape[1] <= 8 else ref.mean(axis=0)
            cand_mono = cand.mean(axis=1) if ref.shape[1] <= 8 else cand.mean(axis=0)
        else:
            ref_mono = ref
            cand_mono = cand

        n_samples = int(min(ref_mono.size, cand_mono.size))
        frame = max(256, int(sample_rate * 0.05))
        hop = frame
        if n_samples < frame:
            return cand, {"quiet_zone_limited_frames": 0.0, "quiet_zone_max_delta_db": 0.0}

        gate_dbfs = compute_signal_relative_gate_dbfs(ref, material_key=str(material_key).lower())
        gate_dbfs = min(float(gate_dbfs), -30.0)
        limited_frames = 0
        max_delta_db = 0.0

        for start in range(0, n_samples - frame + 1, hop):
            end = start + frame
            ref_seg = ref_mono[start:end].astype(np.float64)
            cand_seg = cand_mono[start:end].astype(np.float64)
            ref_rms = float(np.sqrt(np.mean(ref_seg * ref_seg)) + 1e-12)
            cand_rms = float(np.sqrt(np.mean(cand_seg * cand_seg)) + 1e-12)
            ref_db = float(20.0 * np.log10(ref_rms))
            cand_db = float(20.0 * np.log10(cand_rms))
            # v10.0.0 Fix C Korrektur: Digitale Stille (ref_db < -80 dBFS) würde
            # unphysikalische delta_db-Werte (~215 dB) erzeugen → scale ≈ 2e-11 →
            # klangliche Auslöschung. Aber §0h verlangt, dass Stille-Zonen sakrosankt
            # sind: Wenn der Kandidat in einer digitalen Stille-Zone laut ist (Pegelexplosion),
            # muss er auf 0 (Stille) zurückgesetzt werden. Nur wenn der Kandidat ebenfalls
            # leise ist, kann der Frame sicher übersprungen werden.
            if ref_db < -80.0:
                if cand_db > gate_dbfs:
                    # §0h Pegelexplosion in Stille-Zone → Hard-Reset auf 0
                    if cand.ndim == 2 and cand.shape[0] == n_samples:
                        cand[start:end, :] = 0.0
                    elif cand.ndim == 2:
                        cand[:, start:end] = 0.0
                    else:
                        cand[start:end] = 0.0
                    limited_frames += 1
                continue
            if ref_db > gate_dbfs:
                continue
            delta_db = cand_db - ref_db
            max_delta_db = max(max_delta_db, float(delta_db))
            if delta_db <= max_quiet_boost_db:
                continue
            scale = float(10.0 ** ((max_quiet_boost_db - delta_db) / 20.0))
            if cand.ndim == 2 and cand.shape[0] == n_samples:
                cand[start:end, :] *= scale
            elif cand.ndim == 2:
                cand[:, start:end] *= scale
            else:
                cand[start:end] *= scale
            limited_frames += 1

        clipped_candidate: np.ndarray = np.clip(np.nan_to_num(cand, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
            np.float32
        )
        return clipped_candidate, {
            "quiet_zone_limited_frames": float(limited_frames),
            "quiet_zone_max_delta_db": round(float(max_delta_db), 3),
        }

    # §v10.101 SOTA: Bark-basierte Bänder für Hiss-Detektion (>4 kHz = Bark 18–24)
    # Statt 8 linearer Bänder: 7 Gammatone-optimierte kritische Bänder.
    NUM_BANDS = 7  # Bark 18-24: 3.7-15.5 kHz

    def __init__(self, sample_rate: int = 48000, **_kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self._deepfilternet_plugin = None
        self._era_nr_g_floor: float = 0.10  # §EraTarget: era-adaptive NR G_floor (initialized in process())
        self._restoration_context_p29: dict = {}  # §0j: injected from UV3, read in _refine_hf_with_ml
        self._omlsa_panns_singing: float = 0.0  # §SibilantProtect: set in process() before channel calls
        self._preserve_mask_p29: np.ndarray | None = None  # gesetzt in process()
        self._omlsa_runtime_profile_current = {
            "imcra_b_min": 1.66,
            "imcra_alpha_g": 0.85,
            "omlsa_q": 0.50,
            "hf_floor_scale": 0.45,
        }

    @staticmethod
    def _compute_omlsa_runtime_profile(
        material: str,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """Berechnet adaptive OMLSA/IMCRA runtime profile (§2.54)."""
        mat = str(material or "unknown").lower().replace("-", "_").replace(" ", "_")
        qm = str(quality_mode or "balanced").lower().replace("-", "_")
        if restorability_score is None:
            restorability_score = 50.0
        rest = float(np.clip(restorability_score, 0.0, 100.0))

        base = {
            "shellac": {"b_min": 1.72, "alpha_g": 0.87, "q": 0.60, "hf_floor_scale": 0.58},
            "wax_cylinder": {"b_min": 1.74, "alpha_g": 0.88, "q": 0.61, "hf_floor_scale": 0.60},
            "vinyl": {"b_min": 1.67, "alpha_g": 0.85, "q": 0.53, "hf_floor_scale": 0.50},
            "tape": {"b_min": 1.66, "alpha_g": 0.84, "q": 0.50, "hf_floor_scale": 0.45},
            "reel_tape": {"b_min": 1.65, "alpha_g": 0.84, "q": 0.49, "hf_floor_scale": 0.44},
            "cassette": {"b_min": 1.68, "alpha_g": 0.85, "q": 0.52, "hf_floor_scale": 0.48},
            "cd_digital": {"b_min": 1.52, "alpha_g": 0.80, "q": 0.39, "hf_floor_scale": 0.36},
            "streaming": {"b_min": 1.50, "alpha_g": 0.79, "q": 0.38, "hf_floor_scale": 0.35},
            "mp3_low": {"b_min": 1.58, "alpha_g": 0.82, "q": 0.45, "hf_floor_scale": 0.40},
            "unknown": {"b_min": 1.60, "alpha_g": 0.83, "q": 0.46, "hf_floor_scale": 0.42},
        }.get(mat, {"b_min": 1.60, "alpha_g": 0.83, "q": 0.46, "hf_floor_scale": 0.42})

        mode_adj = {
            "fast": -1.0,
            "balanced": 0.0,
            "restoration": 0.3,
            "quality": 0.8,
            "maximum": 1.2,
            "studio_2026": 1.2,
        }.get(qm, 0.0)

        rest_smoothing = (50.0 - rest) / 50.0  # low restorability => higher smoothing

        imcra_b_min = float(np.clip(base["b_min"] + 0.05 * mode_adj + 0.07 * rest_smoothing, 1.40, 1.90))
        imcra_alpha_g = float(np.clip(base["alpha_g"] + 0.03 * mode_adj + 0.04 * rest_smoothing, 0.75, 0.92))
        omlsa_q = float(np.clip(base["q"], 0.35, 0.65))
        hf_floor_scale = float(np.clip(base["hf_floor_scale"], 0.30, 0.65))

        return {
            "imcra_b_min": imcra_b_min,
            "imcra_alpha_g": imcra_alpha_g,
            "omlsa_q": omlsa_q,
            "hf_floor_scale": hf_floor_scale,
        }

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
                logger.warning("Verarbeitungsschritt_29_tape_hiss_reduction.py::_w Ersatzpfad: %s", e)
                return default

        naturalness = float(np.clip(_w("natuerlichkeit"), 0.3, 2.0))
        authenticity = float(np.clip(_w("authentizitaet"), 0.3, 2.0))
        articulation = float(np.clip(_w("artikulation"), 0.3, 2.0))
        brilliance = float(np.clip(_w("brillanz"), 0.3, 2.0))

        scalar = (
            1.0
            + 0.10 * (brilliance - 1.0)
            - 0.10 * (naturalness - 1.0)
            - 0.08 * (authenticity - 1.0)
            - 0.04 * (articulation - 1.0)
        )
        return float(np.clip(scalar, 0.82, 1.12))

    def _get_deepfilternet_plugin(self):
        """
        Lädt DeepFilterNet v3 II Plugin beim ersten Zugriff.

        Returns:
            DeepFilterNet plugin or None if unavailable
        """
        if self._deepfilternet_plugin is not None:
            return self._deepfilternet_plugin

        try:
            from plugins.deepfilternet_v3_ii_plugin import get_deepfilternet_plugin

            self._deepfilternet_plugin = get_deepfilternet_plugin()  # type: ignore[assignment]
            logger.info("✅ DeepFilterNet v3 II Plugin geladen for Tape Hiss Reduction")
            return self._deepfilternet_plugin
        except Exception as e:
            logger.warning("⚠️  DeepFilterNet Plugin not verfuegbar: %s", e)
            logger.info("    Falling back to DSP-only hiss reduction")
            return None

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_29_tape_hiss_reduction",
            name="Tape Hiss Reduction v3 OMLSA/IMCRA",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=6,
            dependencies=["phase_03_denoise", "phase_28_surface_noise_profiling"],
            estimated_time_factor=0.10,
            version="3.0.0",
            memory_requirement_mb=60,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description="HF-OMLSA-Rauschunterdrückung (Cohen 2002/2003) — Über-SOTA",
        )

    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int,
        material: MaterialType = MaterialType.TAPE,
        quality_mode: str | None = None,
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("DeepFilterNetV3", phase_name="29")
        check_ml_model_ready("PANNs", phase_name="29")
        check_ml_model_ready("DeepFilterNetV3", phase_name="29")
        # §v10.15 Type-Guard
        if not isinstance(audio, np.ndarray):
            if isinstance(audio, (tuple, list)):
                audio = audio[0] if len(audio) > 0 else np.zeros(1, dtype=np.float32)
            audio = np.asarray(audio, dtype=np.float32)
        """
        # §v10.15 Type-Guard: ensure audio is ndarray, not a tuple
        if not isinstance(audio, np.ndarray):
            if isinstance(audio, (tuple, list)):
                logger.error("phase_29 received %s instead of ndarray — extracting first element", type(audio).__name__)
                audio = audio[0] if len(audio) > 0 else np.zeros(1, dtype=np.float32)
            audio = np.asarray(audio, dtype=np.float32)
        # Normalize orientation
        _p29_was_channels_first = False
        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = np.ascontiguousarray(audio.T)
            _p29_was_channels_first = True

        Verarbeitet audio to reduce tape hiss with ML-Hybrid support.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Source material type
            quality_mode: Quality mode (FAST/BALANCED/MAXIMUM), None=auto

        Returns:
            PhaseResult with denoised audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        # ── §v10 PIM: Per-Band-Intensität lesen ──
        _per_band_mask = None
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity, compute_per_band_nr_mask

            _pim = apply_pim_intensity(kwargs, "tape_hiss", default_nr=0.6, default_de_ess=0.3, default_comp=1.0)
            _pim_map = kwargs.get("pim_intensity_map")
            if _pim_map is not None and "noise_reduction_strength" in kwargs:
                kwargs["noise_reduction_strength"] = _pim["nr_strength"]
            if _pim_map is not None:
                _per_band_mask = compute_per_band_nr_mask(_pim_map, sample_rate)
        except Exception as e:
            logger.warning("Verarbeitungsschritt_29_tape_hiss_reduction.py::verarbeiten Ersatzpfad: %s", e)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.sample_rate = sample_rate
        self.validate_input(audio)
        audio, _p29_transposed = to_channels_last(audio)

        # §EraTarget: Read era-adaptive G_floor from restoration context (v10.0.0).
        # Stored on self for access in _process_channel_omlsa_mrsa without signature change.
        _era_ctx_p29 = kwargs.get("_restoration_context", {}).get("era_carrier_target", {})
        self._era_nr_g_floor = float(np.clip(_era_ctx_p29.get("nr_g_floor", 0.10), 0.10, 0.50))
        # §4.8a-ii preserve_mask: Stored on self for _process_channel_omlsa_mrsa.
        # NR-Gain in PRESERVE-Bins wird auf G_PRESERVE_FLOOR=0.90 gefloort.
        _ctx_pm_p29 = dict(kwargs.get("_restoration_context", {}) or {})
        _pm_raw_p29 = _ctx_pm_p29.get("preserve_mask")
        self._preserve_mask_p29 = (
            np.asarray(_pm_raw_p29, dtype=np.float32)
            if isinstance(_pm_raw_p29, np.ndarray) and _pm_raw_p29.size > 0
            else None
        )
        # §0j: Store full restoration context on self for sub-methods (_refine_hf_with_ml)
        self._restoration_context_p29 = dict(kwargs.get("_restoration_context", {}) or {})

        # §2.46f Natural-Performance-Artifacts-Guard — detect protected zones before tape hiss reduction
        _npa_result_29 = None
        try:
            _npa_result_29 = _get_phase29_npd().detect(audio, sample_rate)
        except Exception as _npa_exc_29:
            logger.debug("§2.46f NPA detection nicht blockierend: %s", _npa_exc_29)

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_evict29

            _get_plm_evict29().evict_for_phase("phase_29_tape_hiss_reduction")
        except Exception as e:
            logger.warning("Verarbeitungsschritt_29_tape_hiss_reduction.py::verarbeiten Ersatzpfad: %s", e)

        # Determine if ML should be used
        use_ml = False
        if QUALITY_MODE_AVAILABLE and quality_mode:
            try:
                qm = QualityMode[quality_mode.upper()]
                use_ml = qm.is_ml_enabled  # Phase 29
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        restorability_score = kwargs.get("restorability_score", 50.0)
        material_key = str(getattr(material, "value", material) or "unknown")
        omlsa_runtime_profile = self._compute_omlsa_runtime_profile(
            material_key,
            quality_mode,
            restorability_score,
        )
        self._omlsa_runtime_profile_current = omlsa_runtime_profile

        # Skip for digital sources (MaterialType enum)
        if material in [MaterialType.CD_DIGITAL, MaterialType.STREAMING]:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=restore_layout(audio.copy(), _p29_transposed),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "processing": "skipped",
                    "omlsa_runtime_profile": omlsa_runtime_profile,
                },
                warnings=["Digital source - no tape hiss expected"],
            )

        # Skip for digital sources identified by string key (mp3, aac, dat, cd variants).
        # These sources never have tape hiss; OMLSA processing would only add artefacts (§0).
        # EXCEPTION §6.2a: If transfer_chain contains an analog tape/reel_tape/cassette stage
        # (e.g. vinyl→tape→mp3_low), tape hiss reduction MUST run — the mp3 is only the
        # digital container; the actual source has tape hiss.
        _DIGITAL_KEYS = frozenset(
            {
                "mp3_low",
                "mp3_medium",
                "mp3_high",
                "aac",
                "aac_low",
                "aac_high",
                "cd_digital",
                "dat",
                "digital",
                "streaming",
                "mp3",
            }
        )
        _ANALOG_TAPE_KEYS = frozenset({"tape", "reel_tape", "cassette"})
        _transfer_chain_raw = kwargs.get("transfer_chain", [])
        _transfer_chain = [str(c).lower() for c in _transfer_chain_raw] if isinstance(_transfer_chain_raw, list) else []
        _chain_has_tape = any(c in _ANALOG_TAPE_KEYS for c in _transfer_chain)

        _mat_key_norm = material_key.lower().replace("-", "_").replace(" ", "_")
        if _mat_key_norm in _DIGITAL_KEYS and not _chain_has_tape:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            logger.info(
                "Verarbeitungsschritt 29: Digital-Material '%s' — No-Op (no tape hiss expected, §0 Primum non nocere)",
                material_key,
            )
            return PhaseResult(
                success=True,
                audio=restore_layout(audio.copy(), _p29_transposed),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material_key,
                    "processing": "skipped_digital",
                    "reason": "digital_source_no_tape_hiss",
                    "omlsa_runtime_profile": omlsa_runtime_profile,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[f"Digital source '{material_key}' - tape hiss reduction skipped"],
            )
        if _mat_key_norm in _DIGITAL_KEYS and _chain_has_tape:
            logger.info(
                "Verarbeitungsschritt 29: Digital-Material '%s' aber transfer_chain=%s enthält Tape-Stufe "
                "— OMLSA läuft (§6.2a Carrier-Chain-Invariante, §2.46a)",
                material_key,
                _transfer_chain,
            )
            # Für die weiteren Verarbeitungsschritte material_key auf "tape" setzen,
            # damit OMLSA-Profile und Gate-Berechnungen auf Tape-Standards basieren.
            material_key = "tape"

        # §2.47 [RELEASE_MUST] SNR > 35 dB Dry-Signal Bypass
        # If the signal is essentially clean, tape hiss reduction is unnecessary
        # and risks introducing OMLSA musical-noise artifacts (§0 Primum non nocere).
        #
        # Tape hiss is a HF phenomenon concentrated between ~2 kHz and Nyquist.
        # We estimate a spectral SNR: ratio of signal-band energy (200–5 kHz) to
        # hiss-band energy (8 kHz–Nyquist).  A ratio > 35 dB means the hiss is
        # negligible and processing would add more artefacts than it removes.
        _p29_snr_bypass = False
        try:
            _p29_seg = audio[:, 0] if audio.ndim == 2 else audio
            _p29_seg = _p29_seg.astype(np.float64)
            _p29_n = len(_p29_seg)
            # Use at most the central 3 s
            _p29_win = min(_p29_n, 3 * sample_rate)
            _p29_start = max(0, (_p29_n - _p29_win) // 2)
            _p29_chunk = _p29_seg[_p29_start : _p29_start + _p29_win]
            if len(_p29_chunk) >= sample_rate // 2:  # at least 500 ms
                # FFT magnitude spectrum
                _p29_fft = np.abs(np.fft.rfft(_p29_chunk))
                _p29_freqs = np.fft.rfftfreq(len(_p29_chunk), d=1.0 / sample_rate)
                # §2.31 Materialadaptive Hiss-Band-Definition:
                # Kassette+MP3: MP3 entfernt Energie >8 kHz → Kassetten-Hiss liegt in
                # 4–8 kHz; Signalband 200–4 kHz. Reines Kassettenband: 5–12 kHz Hiss.
                # Digital: 8 kHz–Nyquist (klassische Hiss-Region).
                _p29_transfer_chain = kwargs.get("transfer_chain") or []
                _p29_mat_key = getattr(material, "value", str(material)).lower()
                _p29_chain_has_mp3 = any("mp3" in str(c).lower() for c in _p29_transfer_chain)
                _p29_is_tape_mat = _p29_mat_key in ("cassette", "tape", "reel_tape")
                if _p29_is_tape_mat and _p29_chain_has_mp3:
                    # Kassette+MP3: Hiss in 4–8 kHz (MP3 schneidet 8kHz+ weg)
                    _p29_sig_mask = (_p29_freqs >= 200.0) & (_p29_freqs <= 4000.0)
                    _p29_hiss_mask = (_p29_freqs >= 4000.0) & (_p29_freqs <= 8000.0)
                    _p29_snr_threshold = 36.0  # Kassetten-Hiss in 4–8 kHz ist immer hörbar
                elif _p29_is_tape_mat:
                    # Reines Kassettenband: Hiss 5–12 kHz
                    _p29_sig_mask = (_p29_freqs >= 200.0) & (_p29_freqs <= 5000.0)
                    _p29_hiss_mask = (_p29_freqs >= 5000.0) & (_p29_freqs <= 12000.0)
                    _p29_snr_threshold = 35.0
                else:
                    # Digital/Vinyl: klassisch 8 kHz–Nyquist
                    _p29_sig_mask = (_p29_freqs >= 200.0) & (_p29_freqs <= 5000.0)
                    _p29_hiss_mask = _p29_freqs >= 8000.0
                    _p29_snr_threshold = 35.0
                if _p29_sig_mask.any() and _p29_hiss_mask.any():
                    _p29_sig_pwr = float(np.mean(_p29_fft[_p29_sig_mask] ** 2))
                    _p29_hiss_pwr = float(np.mean(_p29_fft[_p29_hiss_mask] ** 2))
                    if _p29_hiss_pwr > 1e-30 and _p29_sig_pwr > 1e-30:
                        _p29_est_snr = 10.0 * np.log10(_p29_sig_pwr / _p29_hiss_pwr)
                        if _p29_est_snr > _p29_snr_threshold:
                            _p29_snr_bypass = True
                            logger.info(
                                "§2.47 Verarbeitungsschritt 29: spectral HF-SNR=%.1f dB > %.1f dB → "
                                "Dry-Signal bypass (hiss band negligible, OMLSA uebersprungen)",
                                _p29_est_snr,
                                _p29_snr_threshold,
                            )
        except Exception as _p29_snr_exc:
            logger.debug(
                "Verarbeitungsschritt 29 SNR bypass estimation fehlgeschlagen (nicht blockierend): %s", _p29_snr_exc
            )

        if _p29_snr_bypass:
            _pass = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            _pass = np.clip(_pass, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=restore_layout(_pass, _p29_transposed),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "processing": "snr_bypass",
                    "snr_bypass": True,
                    "omlsa_runtime_profile": omlsa_runtime_profile,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[f"SNR > {_p29_snr_threshold:.0f} dB: clean tape signal, hiss reduction bypassed"],
            )

        # Get material-specific parameters
        # Fallback via .value-Vergleich loest Doppel-Import-Problem
        # (core.defect_scanner vs. backend.core.defect_scanner erzeugen
        # verschiedene Enum-Klassen-Objekte, obwohl der Wert identisch ist)
        _mat_val = getattr(material, "value", str(material))
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        gate_threshold_db = self.GATE_THRESHOLD_DB.get(_mk) or next(  # type: ignore[call-overload]
            (v for k, v in self.GATE_THRESHOLD_DB.items() if getattr(k, "value", None) == _mat_val),
            -10,
        )
        reduction_depth_db = self.REDUCTION_DEPTH_DB.get(_mk) or next(  # type: ignore[call-overload]
            (v for k, v in self.REDUCTION_DEPTH_DB.items() if getattr(k, "value", None) == _mat_val),
            8,
        )
        _hf = self.HF_FOCUS_RANGE.get(_mk) or next(  # type: ignore[call-overload]
            (v for k, v in self.HF_FOCUS_RANGE.items() if getattr(k, "value", None) == _mat_val),
            (8000, 18000),
        )
        hf_low, hf_high = _hf  # type: ignore[misc]

        # Locality-aware modulation from UV3.
        # Sparse hiss-related defect coverage -> conservative denoising outside affected regions.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        _goal_hint_scalar = self._goal_hint_strength_scalar(kwargs)
        _effective_strength = float(np.clip(_effective_strength * _goal_hint_scalar, 0.0, 1.0))

        # §G78 (GEBOTE.md) CalibrationContext: Kalibrierter Cap aus Messwerten.
        _calib_cap = kwargs.get("phase29_strength_cap")
        if _calib_cap is not None:
            _effective_strength = min(_effective_strength, float(_calib_cap))

        # §K-2 Pre-Phase ACF-Authentizitaet-Cap (§2.45b Minimal-Intervention):
        # Wenn das Input-Audio bereits hohe ACF-Periodizität zeigt (hochwertiges Signal),
        # wird _effective_strength gekappt — verhindert Over-Processing.
        # Non-blocking; Variablen werden vom Post-Delta-Guard weiter unten wiederverwendet.
        _p29_auth_proxy: float = 0.5
        _p29_acf_start: int = 0
        _p29_acf_len: int = 0
        _p29_lag_min: int = 1
        _p29_lag_max: int = 2
        try:
            _p29_mono_in = audio.mean(axis=1) if audio.ndim == 2 else audio
            _p29_n = int(len(_p29_mono_in))
            _p29_acf_start = max(0, _p29_n // 3)
            _p29_acf_len = min(8192, _p29_n - _p29_acf_start)
            _p29_lag_min = max(1, int(sample_rate / 1000))
            _p29_lag_max = min(int(sample_rate / 50), _p29_acf_len)
            if _p29_acf_len >= 512 and _p29_lag_max > _p29_lag_min:
                _p29_seg_in = _p29_mono_in[_p29_acf_start : _p29_acf_start + _p29_acf_len].astype(np.float64)
                # V08-konform: FFT-basierte Autokorrelation statt O(n²) np.correlate
                _p29_acf_in = signal.fftconvolve(_p29_seg_in, _p29_seg_in[::-1], mode="full")
                _p29_acf_in = _p29_acf_in[len(_p29_acf_in) // 2 :]
                _p29_acf_in /= float(_p29_acf_in[0]) + 1e-12
                _p29_auth_proxy = float(
                    np.clip(
                        (float(np.max(_p29_acf_in[_p29_lag_min:_p29_lag_max])) + 1.0) / 2.0,
                        0.0,
                        1.0,
                    )
                )
                if _p29_auth_proxy >= 0.90:
                    _p29_acf_cap_f = 0.25
                elif _p29_auth_proxy >= 0.80:
                    _p29_acf_cap_f = 0.45
                elif _p29_auth_proxy >= 0.70:
                    _p29_acf_cap_f = 0.65
                else:
                    _p29_acf_cap_f = 1.0
                if _p29_acf_cap_f < 1.0:
                    _effective_strength = float(np.clip(_effective_strength * _p29_acf_cap_f, 0.0, 1.0))
                    logger.info(
                        "Verarbeitungsschritt29 §2.45b ACF-Pre-Cap: auth_proxy=%.3f → cap=%.2f → eff_str=%.3f",
                        _p29_auth_proxy,
                        _p29_acf_cap_f,
                        _effective_strength,
                    )
        except Exception as _p29_cap_exc:
            logger.debug("Verarbeitungsschritt29 ACF-Pre-Cap (nicht blockierend): %s", _p29_cap_exc)

        # §V40 NMR-Feedback: NR-Stärke adaptiv anpassen (FeedbackChain-aware).
        try:
            from backend.core.dsp.nmr_feedback import (
                compute_nmr_score as _nmr_fn_29,
            )

            _nmr_result_29 = _nmr_fn_29(audio, sample_rate)
            if not _nmr_result_29.ok:
                logger.warning(
                    "Verarbeitungsschritt29 §V40 NMR: nmr_above_masking → §2.45 Minimal-Intervention prüfen",
                )
            _effective_strength = float(
                np.clip(
                    _effective_strength + _nmr_result_29.recommended_nr_strength_delta,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "Verarbeitungsschritt29 §V40 NMR: delta=%.3f → eff_str=%.3f",
                _nmr_result_29.recommended_nr_strength_delta,
                _effective_strength,
            )
        except Exception as _nmr_exc_29:
            logger.debug("Verarbeitungsschritt29 §V40 NMR nicht blockierend: %s", _nmr_exc_29)

        if _effective_strength < 0.01:  # §v10.0.5: < 1% = keine sinnvolle Wirkung
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=restore_layout(passthrough, _p29_transposed),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "processing": "skipped_zero_strength",
                    "omlsa_runtime_profile": omlsa_runtime_profile,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "goal_hint_scalar": _goal_hint_scalar,
                },
                warnings=["Tape hiss reduction skipped due to zero effective strength"],
            )

        # §v10.58 Depth-Aware Tape-Hiss: Bei transfer_depth ≥ 3 DSP-only + max strength 0.40.
        # DeepFilterNet auf stark degradierten Signalen produziert Musical Noise und Dropouts
        # (identisches Problem wie Phase_03). OMLSA ist sicherer.
        _chain_p29 = kwargs.get("transfer_chain") or (kwargs.get("_restoration_context", {}) or {}).get(
            "transfer_chain", []
        )
        _transfer_depth_p29 = len(_chain_p29) if _chain_p29 else 1
        # §v10.706 B12: Medium-adaptiver Cap aus SourceMediumProfile
        _smp_cap_p29 = 1.0
        try:
            from backend.core.source_medium_profile import get_medium_profile

            _mat_raw = kwargs.get("material_type", kwargs.get("material", "unknown"))
            _mat_p29 = str(getattr(_mat_raw, "value", _mat_raw)).lower()
            _smp = get_medium_profile(_mat_p29)
            _smp_cap_p29 = float(getattr(_smp, "hiss_reduction_max_strength", 1.0))
        except Exception:
            logger.debug("Verarbeitungsschritt_29: SourceMediumProfile nicht verfügbar — verwende Default hiss_cap=1.0")
        if _transfer_depth_p29 >= 5:
            _effective_strength = float(np.clip(_effective_strength, 0.0, min(0.30, _smp_cap_p29)))
            logger.info(
                "§v10.58 Verarbeitungsschritt 29 depth=%d → max strength %.2f + DSP-only (degradiertes Signal, §v10.705 B41/B12)",
                _transfer_depth_p29,
                min(0.30, _smp_cap_p29),
            )
        elif _transfer_depth_p29 >= 4:
            _effective_strength = float(np.clip(_effective_strength, 0.0, min(0.42, _smp_cap_p29)))
            logger.info(
                "§v10.58 Verarbeitungsschritt 29 depth=%d → max strength 0.42 + DSP-only (§v10.705 B41)",
                _transfer_depth_p29,
            )

        # Create frequency bands (logarithmic spacing)
        _nyquist = sample_rate / 2  # used by helper methods referencing sample_rate

        is_stereo = audio.ndim == 2
        _stereo_lag_stats = {
            "lag_input_samples": 0,
            "lag_output_samples": 0,
            "lag_corrected": False,
            "lag_output_corrected_samples": 0,
        }

        # §SibilantProtect [RELEASE_MUST] (v10.0.0.x): PANNs-Score vor OMLSA-Aufruf auf self
        # speichern, damit _process_channel_omlsa_mrsa das Sibilantenband (Presence-Zone)
        # transient-aware glätten kann (schneller Onset-Anstieg bei Gesang).
        self._omlsa_panns_singing = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))

        if is_stereo:
            # §2.51 Linked-Sidechain OMLSA: Gain-Maske aus Mid-Kanal berechnen,
            # identisch auf L und R anwenden. Verhindert stereo-inkohärente HF-Dämpfung
            # bei kanalasymmetrischem Tape-Rauschen (Phantom-Mitte-Instabilität).
            mid_channel = (audio[:, 0] + audio[:, 1]) * (1.0 / np.sqrt(2.0))
            mid_channel = mid_channel.astype(np.float32)

            # Psychoacoustic masking from L channel (dominant sidechain for mid)
            kwargs.get("masking_result")

            # Compute OMLSA gain on Mid sidechain — applied identically to both channels
            audio_processed = np.zeros_like(audio)
            for ch in range(2):
                channel = audio[:, ch]

                # §Psychoacoustic: select pre-computed masking result for this channel.
                if ch == 1:
                    _ch_masking = kwargs.get("masking_result_r") or kwargs.get("masking_result")
                else:
                    _ch_masking = kwargs.get("masking_result")

                # STFT-OMLSA on this channel but with Mid-linked gain sidechain
                audio_processed[:, ch] = self._process_channel_omlsa(
                    channel,
                    sample_rate,
                    hf_low,
                    hf_high,
                    material,
                    intensity_scale=_effective_strength,
                    masking_result=_ch_masking,
                    linked_sidechain=mid_channel,
                )
        else:
            # Mono: standard processing
            _ch_masking = kwargs.get("masking_result")
            audio_processed = self._process_channel_omlsa(
                audio,
                sample_rate,
                hf_low,
                hf_high,
                material,
                intensity_scale=_effective_strength,
                masking_result=_ch_masking,
            )

        # Calculate overall HF noise reduction
        audio_ch0 = audio[:, 0] if is_stereo else audio
        proc_ch0 = audio_processed[:, 0] if is_stereo else audio_processed
        hf_band_orig = self._extract_band(audio_ch0, sample_rate, hf_low, hf_high)
        hf_band_proc = self._extract_band(proc_ch0, sample_rate, hf_low, hf_high)

        # Guard: log10(0) when both bands are silent -> RuntimeWarning; clamp >= 1e-30
        hf_reduction_db = 20 * np.log10(np.maximum(np.std(hf_band_orig) / (np.std(hf_band_proc) + 1e-10), 1e-30))

        # HF over-suppression guard (Restoration): avoid excessive brilliance loss.
        # If tape hiss attenuation exceeds a material-adaptive ceiling, blend back
        # only the HF residual from original audio.
        hf_detail_blend = 0.0
        _mat_name = getattr(material, "name", str(material)).upper()
        _hf_ceiling_db = {
            "TAPE": 10.0,
            "REEL_TAPE": 10.5,
            "VINYL": 9.5,
            "SHELLAC": 8.5,
            # §SibilantProtect: Kassette+mp3_low-Kette → HF durch MP3 bereits reduziert;
            # 9.5 dB Ceiling ließ OMLSA Formanten abtragen (authentizitaet 0.98→0.48)
            "CASSETTE": 7.0,
        }.get(_mat_name, 10.0)
        if hf_reduction_db > _hf_ceiling_db and _effective_strength > 0.0:
            _excess_db = float(hf_reduction_db - _hf_ceiling_db)
            # Weniger HF-Rückblendung: sonst werden Bandfehler als Luftigkeit reingemischt.
            hf_detail_blend = float(np.clip((_excess_db / 16.0) * 0.18 * _effective_strength, 0.0, 0.18))
            if hf_detail_blend > 0.0:
                if is_stereo:
                    for ch in range(2):
                        _orig_hf = self._extract_band(audio[:, ch], sample_rate, hf_low, hf_high)
                        _proc_hf = self._extract_band(audio_processed[:, ch], sample_rate, hf_low, hf_high)
                        audio_processed[:, ch] = np.clip(
                            audio_processed[:, ch] + hf_detail_blend * (_orig_hf - _proc_hf),
                            -1.0,
                            1.0,
                        )
                else:
                    _orig_hf = self._extract_band(audio, sample_rate, hf_low, hf_high)
                    _proc_hf = self._extract_band(audio_processed, sample_rate, hf_low, hf_high)
                    audio_processed = np.clip(
                        audio_processed + hf_detail_blend * (_orig_hf - _proc_hf),
                        -1.0,
                        1.0,
                    )

                # Recompute HF reduction after guard blend.
                proc_ch0 = audio_processed[:, 0] if is_stereo else audio_processed
                hf_band_proc = self._extract_band(proc_ch0, sample_rate, hf_low, hf_high)
                hf_reduction_db = 20 * np.log10(
                    np.maximum(np.std(hf_band_orig) / (np.std(hf_band_proc) + 1e-10), 1e-30)
                )

        # ML Refinement for HF (>2kHz) - if enabled and significant hiss present
        # §0j [RELEASE_MUST]: PANNs-Singing-Score für energy_bias-äquivalente Anpassung
        _p29_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        ml_refined = False
        if use_ml and _effective_strength > 0.0 and hf_reduction_db > 3:  # Only refine if significant hiss was removed
            ml_success = self._refine_hf_with_ml(audio_processed, sample_rate, panns_singing=_p29_panns)
            if ml_success:
                ml_refined = True
                logger.info("✅ ML HF refinement angewendet (DeepFilterNet): residual hiss removal >2kHz")

        # Preserve PMGG strength control via wet/dry blending.
        if 0.0 < _effective_strength < 1.0:
            audio_processed = audio + _effective_strength * (audio_processed - audio)

        audio_processed, loudness_stats = self._apply_material_loudness_preservation(
            audio,
            audio_processed,
            material,
        )

        if is_stereo:
            audio_processed, _stereo_lag_stats = self._enforce_stereo_lag_safety(
                audio,
                audio_processed,
                sample_rate,
            )

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        audio_processed = np.nan_to_num(audio_processed, nan=0.0, posinf=0.0, neginf=0.0)
        audio_processed = np.clip(audio_processed, -1.0, 1.0)

        # §2.46f Natural-Performance-Artifacts-Guard — restore protected breath zones after hiss reduction
        if _npa_result_29 is not None:
            try:
                _npa_n_29 = audio_processed.shape[0]
                _npa_mask_29 = _npa_result_29.get_protected_mask(_npa_n_29, sample_rate)
                if np.any(_npa_mask_29):
                    audio_processed[_npa_mask_29] = audio[_npa_mask_29]
                    logger.debug(
                        "§2.46f NPA Verarbeitungsschritt29: wiederhergestellt %d protected samples",
                        int(np.sum(_npa_mask_29)),
                    )
            except Exception as _npa_rest_29:
                logger.debug("§2.46f NPA restoration nicht blockierend: %s", _npa_rest_29)

        # §TimbralCoherence: Carrier-Rauchtextur nach Over-NR wiederherstellen.
        # Wenn OMLSA/DFN den Rauschboden zu stark abgetragen hat (deviation > 3 dB),
        # wird die material-typische Rauchtextur in Stille-Passagen reiniziert.
        try:
            from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture as _restore_ntr_p29

            _ntr_strength_p29 = float(np.clip(_effective_strength * 0.25, 0.0, 0.45))
            _mat_str_p29 = str(getattr(material, "value", material)).lower()
            audio_processed = _restore_ntr_p29(
                audio,
                audio_processed,
                sample_rate,
                material_type=_mat_str_p29,
                strength=_ntr_strength_p29,
            )
            audio_processed = np.clip(np.nan_to_num(audio_processed, nan=0.0), -1.0, 1.0)
        except Exception as _ntr_exc_p29:
            logger.debug(
                "§TimbralCoherence noise_texture_resynth Verarbeitungsschritt29 (nicht blockierend): %s", _ntr_exc_p29
            )

        # §0p HNR-Blend nach OMLSA-NR (RELEASE_MUST §0p): ΔHNR > 3 dB → Dry-Wet-Blend
        if _p29_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import apply_hnr_blend as _apply_hnr_p29

                _hnr_blended_p29, _hnr_diag_p29 = _apply_hnr_p29(
                    audio.astype(np.float32), audio_processed.astype(np.float32), sample_rate
                )
                if _hnr_diag_p29.get("over_cleaned"):
                    audio_processed = _hnr_blended_p29
            except Exception as _hnr_exc_p29:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_29 (nicht blockierend): %s", _hnr_exc_p29)

        # §0p VQI per-Phase Gate: Stimmqualität nach Tape-NR messen.
        # Over-aggressive OMLSA kann Formanten beschädigen → Rollback auf Original.
        if _p29_panns >= 0.35:
            try:
                from backend.core.musical_goals.era_vocal_profile import (
                    get_era_vocal_profile as _gevp_p29,  # §EraVocalProfile
                )
                from backend.core.musical_goals.vocal_quality_index import (
                    compute_vqi as _compute_vqi_p29,
                )
                from backend.core.musical_goals.vocal_quality_index import (
                    get_vqi_material_floor as _gvmf_p29,
                )

                _era_p29_dec = (
                    kwargs.get("decade")
                    or kwargs.get("era_decade")
                    or (kwargs.get("_restoration_context") or {}).get("decade")
                )
                _vqi_result_p29 = _compute_vqi_p29(
                    audio_orig=audio,
                    audio_restored=audio_processed,
                    sr=sample_rate,
                    era_profile=_gevp_p29(int(_era_p29_dec)) if _era_p29_dec else None,
                )
                _vqi_p29 = float(_vqi_result_p29.get("vqi", 1.0))
                # §v10.57 Material-adaptiver VQI-Threshold (identisch zu phase_03)
                _mat_p29 = str(kwargs.get("material_type", "") or kwargs.get("material", "")).strip()
                _vqi_floor_p29 = float(_gvmf_p29(_mat_p29))
                _vqi_thr_p29 = float(np.clip(_vqi_floor_p29 + 0.10, 0.82, 0.92))
                if _vqi_p29 < _vqi_thr_p29:
                    _vqi_blend_p29 = float(np.clip(_vqi_p29 / max(_vqi_thr_p29, 0.01), 0.15, 0.85))
                    audio_processed = (_vqi_blend_p29 * audio_processed + (1.0 - _vqi_blend_p29) * audio).astype(
                        np.float32
                    )
                    logger.info(
                        "Verarbeitungsschritt_29: VQI-Blend vqi=%.3f < thr=%.2f (floor=%.2f) blend=%.2f panns=%.2f",
                        _vqi_p29,
                        _vqi_thr_p29,
                        _vqi_floor_p29,
                        _vqi_blend_p29,
                        _p29_panns,
                    )
            except Exception as _vqi_exc_p29:
                logger.debug(
                    "VQI per-Verarbeitungsschritt Verarbeitungsschritt29 (nicht blockierend): %s", _vqi_exc_p29
                )

        # §G5 (GEBOTE.md) Formant ±2 dB Guard (§0p RELEASE_MUST): F1–F4 via LPC post-Tape-NR.
        # Tape-NR (OMLSA) kann bei starker Glättung Formant-Regionen beschädigen;
        # Spektralenergie-Shift an F1–F4 direkt messen.
        if _p29_panns >= 0.25:
            try:
                from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db as _cfs_p29
                from backend.core.musical_goals.era_vocal_profile import (
                    resolve_formant_tolerance_db as _rft_p29,
                )

                _ctx_p29 = kwargs.get("_restoration_context", {}) if hasattr(kwargs, "get") else {}
                _era_p29 = kwargs.get("decade") or kwargs.get("era_decade") or _ctx_p29.get("decade")
                _fg_tol_p29 = float(
                    kwargs.get(
                        "formant_tolerance_db",
                        _rft_p29(
                            era_decade=int(_era_p29) if _era_p29 is not None else None,
                            era_profile=kwargs.get("era_vocal_profile"),
                        ),
                    )
                )
                _fg_rollback_p29, _fg_shift_p29 = _cfs_p29(
                    audio, audio_processed, sample_rate, threshold_db=_fg_tol_p29
                )
                if _fg_rollback_p29:
                    audio_processed = audio.copy()
                    logger.warning(
                        "§G5 (GEBOTE.md) FormantGuard Verarbeitungsschritt_29: max F-shift %.2f dB > %.1f dB → Rollback",
                        _fg_shift_p29,
                        _fg_tol_p29,
                    )
                else:
                    logger.debug(
                        "§G5 (GEBOTE.md) FormantGuard Verarbeitungsschritt_29: max F-shift %.2f dB — OK", _fg_shift_p29
                    )
            except Exception as _fg_p29_exc:
                logger.debug("§G5 (GEBOTE.md) FormantGuard Verarbeitungsschritt_29 nicht blockierend: %s", _fg_p29_exc)

        # §G2 (GEBOTE.md) Breath-Segment Protection (§2.46f): EMOTIONAL_TENSION Atemgeräusche
        # mit Original zurückblenden — Tape-NR glättet sonst Natur-Artefakte weg.
        _breath_segs_p29 = list(kwargs.get("breath_segments", []) or []) if hasattr(kwargs, "get") else []
        if _breath_segs_p29:
            try:
                _n_out_p29 = audio_processed.shape[-1] if audio_processed.ndim == 2 else len(audio_processed)
                _n_in_p29 = audio.shape[-1] if audio.ndim == 2 else len(audio)
                _n_blend_p29 = min(_n_out_p29, _n_in_p29)
                _result_blend_p29 = np.array(audio_processed, copy=True)
                _blended_any_p29 = False
                for _bs_p29 in _breath_segs_p29:
                    _cat_p29 = getattr(_bs_p29, "category", None)
                    _cat_str_p29 = str(getattr(_cat_p29, "value", _cat_p29 or "")).lower()
                    if "tension" not in _cat_str_p29 and "emotional" not in _cat_str_p29:
                        continue
                    _bs_start_p29 = float(getattr(_bs_p29, "start_s", 0.0))
                    _bs_end_p29 = float(getattr(_bs_p29, "end_s", 0.0))
                    _g_fl_p29 = float(np.clip(getattr(_bs_p29, "recommended_g_floor", 0.50), 0.0, 1.0))
                    _dry_p29 = float(np.clip(_g_fl_p29, 0.05, 0.95))
                    if _bs_end_p29 <= _bs_start_p29:
                        continue
                    _si_p29 = int(round(_bs_start_p29 * sample_rate))
                    _ei_p29 = int(round(_bs_end_p29 * sample_rate))
                    _si_p29 = max(0, min(_si_p29, _n_blend_p29))
                    _ei_p29 = max(0, min(_ei_p29, _n_blend_p29))
                    if _si_p29 >= _ei_p29:
                        continue
                    if _result_blend_p29.ndim == 2 and audio.ndim == 2:
                        _result_blend_p29[:, _si_p29:_ei_p29] = (
                            _dry_p29 * audio[:, _si_p29:_ei_p29]
                            + (1.0 - _dry_p29) * audio_processed[:, _si_p29:_ei_p29]
                        )
                    elif _result_blend_p29.ndim == 1 and audio.ndim == 1:
                        _result_blend_p29[_si_p29:_ei_p29] = (
                            _dry_p29 * audio[_si_p29:_ei_p29] + (1.0 - _dry_p29) * audio_processed[_si_p29:_ei_p29]
                        )
                    _blended_any_p29 = True
                if _blended_any_p29:
                    audio_processed = np.clip(np.nan_to_num(_result_blend_p29, nan=0.0), -1.0, 1.0).astype(np.float32)
                    logger.debug(
                        "§G2 (GEBOTE.md) BreathProtect Verarbeitungsschritt_29: %d tension-segs geschützt",
                        len(_breath_segs_p29),
                    )
            except Exception as _g2_p29_exc:
                logger.debug("§G2 (GEBOTE.md) BreathProtect Verarbeitungsschritt_29 nicht blockierend: %s", _g2_p29_exc)

        # §Gap3 PhraseBoundaryGuard — taper artifacts at phrase transitions (§0p Vocal-Supremacy)
        try:
            from backend.core.dsp.phrase_boundary_guard import (  # noqa: I001
                detect_phrase_boundaries as _detect_pbg_29,
                apply_phrase_boundary_taper as _apply_pbg_29,
            )

            _pbg_bounds_29 = _detect_pbg_29(audio, sample_rate)
            if _pbg_bounds_29:
                _pbg_env_29 = _apply_pbg_29(audio, _pbg_bounds_29, sample_rate, taper_ms=20.0).astype(np.float32)
                _pbg_aud_29 = audio if audio.ndim == audio_processed.ndim else np.asarray(audio, dtype=np.float32)
                if audio_processed.ndim == 1:
                    audio_processed = _pbg_aud_29 + (audio_processed - _pbg_aud_29) * _pbg_env_29
                elif audio_processed.ndim == 2 and audio_processed.shape[0] == 2 and audio_processed.shape[1] > 2:
                    audio_processed = _pbg_aud_29 + (audio_processed - _pbg_aud_29) * _pbg_env_29[np.newaxis, :]
                else:
                    audio_processed = _pbg_aud_29 + (audio_processed - _pbg_aud_29) * _pbg_env_29[:, np.newaxis]
                audio_processed = np.clip(np.nan_to_num(audio_processed, nan=0.0), -1.0, 1.0).astype(np.float32)
                logger.debug("§Gap3 PhraseBoundaryGuard Verarbeitungsschritt_29: %d boundaries", len(_pbg_bounds_29))
        except Exception as _pbg_exc_29:
            logger.debug("PhraseBoundaryGuard Verarbeitungsschritt_29 (nicht blockierend): %s", _pbg_exc_29)

        # V19 Noise-Textur-Invariante (§NTI): Residual nach Tape-Hiss-NR darf kein
        # material-fremdes Spektralprofil (Whitening) aufweisen (VERBOTEN-V19).
        try:
            from backend.core.dsp.noise_texture_guard import (
                compute_noise_texture_distance as _nt29_dist_fn,
            )

            _nt29_residual = audio.astype(np.float32) - audio_processed.astype(np.float32)
            _nt29_dist = _nt29_dist_fn(_nt29_residual, str(material_key or "unknown"), sr=sample_rate)
            if _nt29_dist > 0.25:
                audio_processed = (0.5 * audio_processed + 0.5 * audio).astype(np.float32)
                logger.warning(
                    "Verarbeitungsschritt29 V19 Noise-Textur-Dist=%.3f > 0.25 → 50%%-Blend (Träger-Textur bewahrt)",
                    _nt29_dist,
                )
        except Exception as _nt29_exc:
            logger.debug("Verarbeitungsschritt29 V19 Noise-Textur-Guard (nicht blockierend): %s", _nt29_exc)

        # V20 Mikrodynamik-Korrelation (§2.75): Frame-Energie auf voiced-Zonen ≥ 0.97
        # nach Tape-Hiss-NR (VERBOTEN-V20).
        if _p29_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (
                    frame_energy_correlation as _fec29,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr29 = _fec29(audio, audio_processed, sample_rate, frame_ms=10.0)
                if _corr29 < 0.97:
                    _need29 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    # §v10.101 Material-adaptiv
                    _wet29 = _recommend_mkk_wet(_corr29, _p29_panns, global_need=_need29, material=material_key)
                    audio_processed = (_wet29 * audio_processed + (1.0 - _wet29) * audio).astype(np.float32)
                    logger.warning(
                        "Verarbeitungsschritt29 V20 Mikrodynamik-Korr=%.3f < 0.97 → wet=%.3f Blend",
                        _corr29,
                        _wet29,
                    )
            except Exception as _dyn29_exc:
                logger.debug("Verarbeitungsschritt29 V20 Mikrodynamik-Guard (nicht blockierend): %s", _dyn29_exc)

        # V21 Mindestrauschboden (§2.76): Analog-Material darf nach Tape-Hiss-NR keine
        # digitale Stille aufweisen — Rauschboden ist Naturalness-Marker (VERBOTEN-V21).
        _mat29_str = str(material_key or "unknown").lower()
        if any(t in _mat29_str for t in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (
                    apply_noise_floor_minimum as _nfg29,
                )

                audio_processed = _nfg29(audio_processed, sample_rate, _mat29_str, original_audio=audio)
            except Exception as _nf29_exc:
                logger.debug("Verarbeitungsschritt29 V21 Noise-Floor-Guard (nicht blockierend): %s", _nf29_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (
                check_spectral_color_preservation as _scg_29,
            )

            _sc_result_29 = _scg_29(audio, audio_processed, sample_rate)
            if not _sc_result_29.ok:
                _sc_wet_29 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                audio_processed = (_sc_wet_29 * audio_processed + (1.0 - _sc_wet_29) * audio).astype(np.float32)
        except Exception as _sc_exc_29:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_29 spectral_color nicht blockierend: %s", _sc_exc_29
            )

        # V26 Onset-Guard (§2.77): HPSS-Onset-Fenster (0–20 ms nach Transient) dürfen durch
        # Tape-Hiss-NR nicht energetisch beeinflusst werden (VERBOTEN-V26).
        try:
            from backend.core.dsp.onset_guard import (
                apply_onset_protection_mask as _opg29,
            )

            audio_processed = _opg29(audio, audio_processed, None, max_delta_db=1.5)
        except Exception as _on29_exc:
            logger.debug("Verarbeitungsschritt29 V26 Onset-Guard (nicht blockierend): %s", _on29_exc)

        # §2.72 Vibrato-Tiefe-Guard (§0p Vocal-Supremacy RELEASE_MUST): F0-Modulationstiefe
        # darf durch Tape-Hiss-NR nicht mehr als ±10 % reduziert werden → 50 %-Blend.
        if _p29_panns >= 0.25:
            try:
                from backend.core.dsp.vibrato_guard import (
                    check_vibrato_depth_preservation as _vib29,
                )

                _vib29_result = _vib29(audio, audio_processed, sample_rate)
                if not _vib29_result.ok:
                    audio_processed = (0.5 * audio_processed + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "Verarbeitungsschritt29 §2.72 Vibrato-Tiefe: reduction=%.1f%% > 10%% → 50%%-Blend",
                        _vib29_result.depth_reduction_pct,
                    )
            except Exception as _vib29_exc:
                logger.debug("Verarbeitungsschritt29 §2.72 Vibrato-Guard (nicht blockierend): %s", _vib29_exc)

        # §K-2 Post-Processing ACF-Delta-Guard (§Primum-non-nocere):
        # Vergleicht Authentizitaet-Proxy vor und nach Verarbeitung. Wenn die
        # Periodizität um mehr als 0.20 sinkt → proportionaler Dry-Wet-Blend-Rollback.
        # Non-blocking; nutzt Messvariablen aus dem Pre-Cap-Block.
        try:
            if _p29_acf_len >= 512 and _p29_lag_max > _p29_lag_min:
                _p29_mono_out = audio_processed.mean(axis=1) if audio_processed.ndim == 2 else audio_processed
                _p29_seg_out = _p29_mono_out[_p29_acf_start : _p29_acf_start + _p29_acf_len].astype(np.float64)
                # V08-konform: FFT-basierte Autokorrelation statt O(n²) np.correlate
                _p29_acf_out = signal.fftconvolve(_p29_seg_out, _p29_seg_out[::-1], mode="full")
                _p29_acf_out = _p29_acf_out[len(_p29_acf_out) // 2 :]
                _p29_acf_out /= float(_p29_acf_out[0]) + 1e-12
                _p29_auth_out = float(
                    np.clip(
                        (float(np.max(_p29_acf_out[_p29_lag_min:_p29_lag_max])) + 1.0) / 2.0,
                        0.0,
                        1.0,
                    )
                )
                _p29_auth_delta = _p29_auth_out - _p29_auth_proxy
                if _p29_auth_delta < -0.20:
                    # 0.20-Einbruch → 50% dry; 0.40-Einbruch → vollständig dry (max 0.75 blend)
                    _p29_post_blend = float(np.clip(1.0 + _p29_auth_delta / 0.40, 0.0, 0.75))
                    audio_processed = (_p29_post_blend * audio_processed + (1.0 - _p29_post_blend) * audio).astype(
                        np.float32
                    )
                    logger.warning(
                        "Verarbeitungsschritt29 §K-2 ACF-Delta-Guard: auth_delta=%.3f (%.3f→%.3f) → blend=%.2f",
                        _p29_auth_delta,
                        _p29_auth_proxy,
                        _p29_auth_out,
                        _p29_post_blend,
                    )
        except Exception as _p29_delta_exc:
            logger.debug("Verarbeitungsschritt29 ACF-Delta-Guard (nicht blockierend): %s", _p29_delta_exc)

        audio_processed, _quiet_zone_stats_p29 = self._limit_quiet_zone_boost(
            audio,
            audio_processed,
            sample_rate,
            material_key,
        )
        if _quiet_zone_stats_p29["quiet_zone_limited_frames"] > 0:
            logger.warning(
                "§0h Verarbeitungsschritt_29 Quiet-Zone-Guard: limited %.0f frame(s), maxΔ=%.2f dB",
                _quiet_zone_stats_p29["quiet_zone_limited_frames"],
                _quiet_zone_stats_p29["quiet_zone_max_delta_db"],
            )

        # §V42 Rauigkeits-Regression nach Tape-Hiss-NR (non-blocking, §2.62): VERBOTEN-V42
        try:
            from backend.core.dsp.zwicker_metrics import (
                check_roughness_regression as _crr29,
            )

            _zr29 = _crr29(audio, audio_processed, sample_rate)
            if _zr29.roughness_regression:
                audio_processed = (0.90 * audio_processed + 0.10 * audio).astype(np.float32)
                logger.warning("Verarbeitungsschritt29 §V42 Rauigkeits-Regression → Blend ×0.90")
            if _zr29.pumping_detected:
                audio_processed = (0.80 * audio_processed + 0.20 * audio).astype(np.float32)
                logger.warning("Verarbeitungsschritt29 §V42 NR-Pumpen → Blend ×0.80")
        except Exception as _zr29_exc:
            logger.debug("Verarbeitungsschritt29 §V42 Roughness-Pruefung nicht blockierend: %s", _zr29_exc)

        # §BandAnchor: spektrale Grundbalance bei zu heller/zu dünner Ausgabe
        # sanft zurück auf den Originalträger ziehen, ohne den Hiss-Fix zu verlieren.
        _band_anchor_lowmid_ratio_29 = 1.0
        _band_anchor_presence_ratio_29 = 1.0
        _band_anchor_air_ratio_29 = 1.0
        _band_anchor_mix_29 = 0.0
        try:

            def _anchor_mono_29(sig: np.ndarray) -> np.ndarray:
                _arr = np.asarray(sig, dtype=np.float32)
                if _arr.ndim == 1:
                    return cast(np.ndarray, _arr)
                if _arr.ndim == 2:
                    if _arr.shape[0] <= 8 and _arr.shape[1] > _arr.shape[0]:
                        return cast(np.ndarray, np.asarray(np.mean(_arr, axis=0), dtype=np.float32))
                    if _arr.shape[1] <= 8:
                        return cast(np.ndarray, np.asarray(np.mean(_arr, axis=1), dtype=np.float32))
                    _axis = 0 if _arr.shape[0] < _arr.shape[1] else 1
                    return cast(np.ndarray, np.asarray(np.mean(_arr, axis=_axis), dtype=np.float32))
                return cast(np.ndarray, np.asarray(np.ravel(_arr), dtype=np.float32))

            _anchor_src_29 = _anchor_mono_29(audio)
            _anchor_proc_29 = _anchor_mono_29(audio_processed)
            if len(_anchor_src_29) >= 2048 and len(_anchor_proc_29) >= 2048:

                def _band_energy_29(sig: np.ndarray, lo: float, hi: float) -> float:
                    _freqs, _pxx = signal.welch(sig.astype(np.float64), sample_rate, nperseg=min(8192, len(sig)))
                    _mask = (_freqs >= lo) & (_freqs < hi)
                    if not np.any(_mask):
                        return 0.0
                    return float(np.trapz(_pxx[_mask], _freqs[_mask]))

                _src_lowmid_29 = _band_energy_29(_anchor_src_29, 120.0, 400.0)
                _proc_lowmid_29 = _band_energy_29(_anchor_proc_29, 120.0, 400.0)
                _src_presence_29 = _band_energy_29(_anchor_src_29, 400.0, 2000.0)
                _proc_presence_29 = _band_energy_29(_anchor_proc_29, 400.0, 2000.0)
                _src_air_29 = _band_energy_29(_anchor_src_29, 8000.0, 16000.0)
                _proc_air_29 = _band_energy_29(_anchor_proc_29, 8000.0, 16000.0)

                _lowmid_ratio_29 = _proc_lowmid_29 / (_src_lowmid_29 + 1e-18)
                _presence_ratio_29 = _proc_presence_29 / (_src_presence_29 + 1e-18)
                _air_ratio_29 = _proc_air_29 / (_src_air_29 + 1e-18)
                _band_anchor_lowmid_ratio_29 = float(_lowmid_ratio_29)
                _band_anchor_presence_ratio_29 = float(_presence_ratio_29)
                _band_anchor_air_ratio_29 = float(_air_ratio_29)

                _anchor_need_29 = 0.0
                if _lowmid_ratio_29 < 0.86:
                    _anchor_need_29 = max(_anchor_need_29, float(np.clip((0.86 - _lowmid_ratio_29) / 0.36, 0.0, 1.0)))
                if _presence_ratio_29 < 0.90:
                    _anchor_need_29 = max(_anchor_need_29, float(np.clip((0.90 - _presence_ratio_29) / 0.30, 0.0, 1.0)))
                if _air_ratio_29 > 1.8:
                    _anchor_need_29 = max(_anchor_need_29, float(np.clip((_air_ratio_29 - 1.8) / 3.2, 0.0, 1.0)))

                if _anchor_need_29 > 0.0:
                    _anchor_mix_29 = float(np.clip(_anchor_need_29 * 0.65, 0.0, 0.65))
                    _band_anchor_mix_29 = float(_anchor_mix_29)
                    audio_processed = ((1.0 - _anchor_mix_29) * audio_processed + _anchor_mix_29 * audio).astype(
                        np.float32
                    )
                    logger.warning(
                        "Verarbeitungsschritt29 BandAnchor: lowmid=%.2f presence=%.2f air=%.2f → Originalsignal-blend=%.2f",
                        _lowmid_ratio_29,
                        _presence_ratio_29,
                        _air_ratio_29,
                        _anchor_mix_29,
                    )
        except Exception as _band_anchor_exc_29:
            logger.debug("Verarbeitungsschritt29 BandAnchor (nicht blockierend): %s", _band_anchor_exc_29)

        # ── §v10 Per-Band-Maske NACH tape_hiss anwenden ──
        if _per_band_mask is not None:
            try:
                from backend.core.pim_phase_hook import apply_per_band_mask

                _before = audio
                _after = apply_per_band_mask(_before, _per_band_mask, sample_rate, mix=0.55)
                audio = _after
            except Exception as e:
                logger.warning("Verarbeitungsschritt_29_tape_hiss_reduction.py::_band_energy_29 Ersatzpfad: %s", e)

        # §2.71 Strength-Envelope: Chirurgische Tape-Hiss-Reduktion
        _strength_env = kwargs.get("strength_envelope")
        if _strength_env is not None:
            try:
                from backend.core.strength_envelope import apply_strength_envelope

                _env_pre = np.asarray(audio_processed, dtype=np.float32)
                audio_processed = apply_strength_envelope(
                    processed=_env_pre,
                    original=np.asarray(audio, dtype=np.float32),
                    envelope=_strength_env,
                    sample_rate=sample_rate,
                    base_strength=_effective_strength,
                )
                if float(np.mean(np.abs(audio_processed - _env_pre))) > 0.001:
                    logger.info(
                        "§2.71 Envelope-Blending Verarbeitungsschritt 29: Δ=%.4f RMS",
                        float(np.mean(np.abs(audio_processed - _env_pre))),
                    )
            except Exception as _se_exc:
                logger.debug("§2.71 Envelope nicht blockierend: %s", _se_exc)

        return PhaseResult(
            success=True,
            audio=restore_layout(audio_processed, _p29_transposed),
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "gate_threshold_db": float(gate_threshold_db),  # type: ignore[arg-type]
                "reduction_depth_db": float(reduction_depth_db),  # type: ignore[arg-type]
                "hf_focus_range_hz": [int(hf_low), int(hf_high)],
                "omlsa_runtime_profile": omlsa_runtime_profile,
                "hf_reduction_db": round(float(hf_reduction_db), 2),
                "hf_detail_blend": round(float(hf_detail_blend), 4),
                "ml_refined": ml_refined,
                "algorithm_version": "3.0_omlsa_ml_hybrid" if ml_refined else "3.0_omlsa",
                "algorithm": "IMCRA+OMLSA (Cohen 2002/2003)",
                "stereo_mode": "linked_mid_sidechain" if is_stereo else "mono",
                "ml_model": "DeepFilterNet v3 II" if ml_refined else None,
                "rt_factor": float(rt_factor),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "goal_hint_scalar": _goal_hint_scalar,
                "acf_auth_proxy_pre": round(float(_p29_auth_proxy), 4),
                "rms_drop_db": loudness_stats["rms_drop_db"],
                "loudness_makeup_db": loudness_stats["makeup_gain_db"],
                "lag_input_samples": int(_stereo_lag_stats["lag_input_samples"]),
                "lag_output_samples": int(_stereo_lag_stats["lag_output_samples"]),
                "lag_corrected": bool(_stereo_lag_stats["lag_corrected"]),
                "lag_output_corrected_samples": int(_stereo_lag_stats["lag_output_corrected_samples"]),
                "quiet_zone_limited_frames": int(_quiet_zone_stats_p29["quiet_zone_limited_frames"]),
                "quiet_zone_max_delta_db": float(_quiet_zone_stats_p29["quiet_zone_max_delta_db"]),
                "band_anchor_lowmid_ratio": round(float(_band_anchor_lowmid_ratio_29), 4),
                "band_anchor_presence_ratio": round(float(_band_anchor_presence_ratio_29), 4),
                "band_anchor_air_ratio": round(float(_band_anchor_air_ratio_29), 4),
                "band_anchor_original_blend": round(float(_band_anchor_mix_29), 4),
            },
            warnings=[] if rt_factor < 0.12 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _apply_material_loudness_preservation(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, dict[str, float]]:
        material_key = getattr(material, "name", str(material)).lower()
        max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(material_key, self._MAX_RMS_DROP_DB["unknown"]))

        # §2.45a-I Gated-RMS: ignoriert Stille-Frames < −50 dBFS
        # CRITICAL FIX: _rms_dbfs_gated gibt -96.0 zurück wenn nach OMLSA alle Frames
        # unter -50 dBFS liegen (echtes Signal ~-20 dB gedämpft, NICHT wirklich stumm).
        # Fallback auf globalen RMS verhindert fälschliche -62 dB "Abfall"-Berechnung.
        _orig_arr = np.asarray(original_audio, dtype=np.float32)
        _proc_arr = np.asarray(processed_audio, dtype=np.float32)
        _rms_in_db = _rms_dbfs_gated(_orig_arr)

        _rms_out_gated = _rms_dbfs_gated(_proc_arr)
        if _rms_out_gated <= -90.0:
            # Kein aktiver Frame → globaler RMS als Fallback (verhindert False-Negative)
            _proc_mono = (
                _proc_arr.mean(axis=0)
                if _proc_arr.ndim == 2 and _proc_arr.shape[0] <= 2
                else (_proc_arr.mean(axis=1) if _proc_arr.ndim == 2 else _proc_arr)
            )
            _rms_out_db = float(20.0 * np.log10(np.sqrt(np.mean(_proc_mono.astype(np.float64) ** 2)) + 1e-12))
            logger.debug(
                "Verarbeitungsschritt 29: gated-RMS kein aktiver Frame → globaler RMS Ersatzpfad %.1f dBFS", _rms_out_db
            )
        else:
            _rms_out_db = _rms_out_gated

        rms_in = float(10.0 ** (_rms_in_db / 20.0))
        rms_drop_db = (_rms_out_db - _rms_in_db) if _rms_in_db > -90.0 else 0.0
        makeup_gain_db = 0.0

        # Material-adaptives Makeup-Cap (§2.45a): max. 30 dB (verhindert Explosion),
        # mindestens so viel dass der Abfall auf max_rms_drop_db begrenzt wird.
        _MAKEUP_CAP_DB = 30.0

        if rms_in > 1e-8 and rms_drop_db < -max_rms_drop_db:
            target_rms_drop_db = -max_rms_drop_db
            required_gain_db = target_rms_drop_db - rms_drop_db
            makeup_gain_db = float(np.clip(required_gain_db, 0.0, _MAKEUP_CAP_DB))
            if makeup_gain_db > 0.0:
                _gain_lin = float(10.0 ** (makeup_gain_db / 20.0))
                _peak_99 = float(np.percentile(np.abs(_proc_arr), 99.9))
                if _peak_99 > 1e-8:
                    # Peak-Guard: Gain nur soweit wie Clipping-frei möglich
                    _safe_gain = min(_gain_lin, 0.999 / _peak_99)
                    # §2.45a-II: signal-relative gate — CEDAR/iZotope RX approach (v10.0.0)
                    _gate_dbfs_29 = compute_signal_relative_gate_dbfs(_orig_arr, material_key=material_key)
                    processed_audio = apply_musical_gain_envelope(
                        _proc_arr,
                        _safe_gain,
                        gate_dbfs=_gate_dbfs_29,
                        crossfade_ms=10.0,
                        sr=48000,
                        reference_for_gate=_orig_arr,
                    )
                    _peak_after = float(np.percentile(np.abs(np.asarray(processed_audio, dtype=np.float32)), 99.9))
                    if _peak_after > 0.999 and _peak_after > 1e-8:
                        _trim_gain = 0.999 / _peak_after
                        processed_audio = np.clip(processed_audio * _trim_gain, -1.0, 1.0).astype(np.float32)
                        _safe_gain *= _trim_gain
                    makeup_gain_db = float(20.0 * np.log10(_safe_gain + 1e-12))
                else:
                    processed_audio = _proc_arr  # Signal zu klein — kein Makeup
                    makeup_gain_db = 0.0

                # §2.45a-I: re-measure after makeup-gain
                _rms_out_db = _rms_dbfs_gated(np.asarray(processed_audio, dtype=np.float32))
                if _rms_out_db <= -90.0:
                    _pm = (
                        processed_audio.mean(axis=0)
                        if processed_audio.ndim == 2 and processed_audio.shape[0] <= 2
                        else (processed_audio.mean(axis=1) if processed_audio.ndim == 2 else processed_audio)
                    )
                    _rms_out_db = float(20.0 * np.log10(np.sqrt(np.mean(_pm.astype(np.float64) ** 2)) + 1e-12))
                rms_drop_db = (_rms_out_db - _rms_in_db) if _rms_in_db > -90.0 else 0.0
                logger.info(
                    "Verarbeitungsschritt 29 loudness-preservation: material=%s rms_drop=%.2f dB via makeup %.2f dB (envelope-aware)",
                    material_key,
                    rms_drop_db,
                    makeup_gain_db,
                )

        return processed_audio, {
            "rms_drop_db": round(float(rms_drop_db), 3),
            "makeup_gain_db": round(float(makeup_gain_db), 3),
        }

    @staticmethod
    def _estimate_stereo_lag_samples(stereo_audio: np.ndarray, max_lag_samples: int = 960) -> int:
        """Schätzt inter-channel lag (R relative to L) for channels-last stereo."""
        if stereo_audio.ndim != 2 or stereo_audio.shape[1] != 2:
            return 0
        left = np.asarray(stereo_audio[:, 0], dtype=np.float64)
        right = np.asarray(stereo_audio[:, 1], dtype=np.float64)
        n = min(len(left), len(right))
        if n < 4096:
            return 0
        win = min(n, 48000)
        start = max(0, (n - win) // 2)
        left = left[start : start + win]
        right = right[start : start + win]
        left -= np.mean(left)
        right -= np.mean(right)
        denom = float(np.sqrt(np.mean(left**2) * np.mean(right**2)) + 1e-12)
        if denom < 1e-10:
            return 0
        corr = signal.correlate(left, right, mode="full", method="fft")
        lags = signal.correlation_lags(len(left), len(right), mode="full")
        mask = np.abs(lags) <= int(max_lag_samples)
        if not np.any(mask):
            return 0
        idx = int(np.argmax(corr[mask]))
        return int(lags[mask][idx])

    @staticmethod
    def _shift_channel_no_wrap(channel: np.ndarray, shift_samples: int) -> np.ndarray:
        """Verschiebt channel with edge fill, never wrap samples."""
        x: np.ndarray = np.asarray(channel, dtype=np.float32)
        n = len(x)
        if n == 0 or shift_samples == 0:
            x_copy: np.ndarray = x.copy()
            return x_copy
        out = np.empty_like(x)
        if shift_samples > 0:
            shift = min(shift_samples, n - 1)
            out[:shift] = x[0]
            out[shift:] = x[:-shift]
            out_result: np.ndarray = out
            return out_result
        shift = min(-shift_samples, n - 1)
        out[:-shift] = x[shift:]
        out[-shift:] = x[-1]
        out_result_neg: np.ndarray = out
        return out_result_neg

    def _enforce_stereo_lag_safety(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, dict[str, int | bool]]:
        """Keep inter-channel lag close to input to prevent audible L/R desync."""
        stats = {
            "lag_input_samples": 0,
            "lag_output_samples": 0,
            "lag_corrected": False,
            "lag_output_corrected_samples": 0,
        }
        if original_audio.ndim != 2 or processed_audio.ndim != 2:
            return processed_audio, stats
        if original_audio.shape[1] != 2 or processed_audio.shape[1] != 2:
            return processed_audio, stats

        max_lag_samples = int(min(960, max(48, sample_rate // 50)))
        lag_in = self._estimate_stereo_lag_samples(original_audio, max_lag_samples=max_lag_samples)
        lag_out = self._estimate_stereo_lag_samples(processed_audio, max_lag_samples=max_lag_samples)
        stats["lag_input_samples"] = int(lag_in)
        stats["lag_output_samples"] = int(lag_out)

        # Allow up to 1 ms introduced lag; beyond that we align output back to input lag.
        max_introduced = int(max(1, round(sample_rate * 0.001)))
        lag_delta = int(lag_out - lag_in)
        if abs(lag_delta) <= max_introduced:
            stats["lag_output_corrected_samples"] = int(lag_out)
            return processed_audio, stats

        corrected = np.asarray(processed_audio, dtype=np.float32).copy()
        corrected[:, 1] = self._shift_channel_no_wrap(corrected[:, 1], int(lag_delta))
        lag_corr = self._estimate_stereo_lag_samples(corrected, max_lag_samples=max_lag_samples)
        stats["lag_corrected"] = True
        stats["lag_output_corrected_samples"] = int(lag_corr)
        logger.warning(
            "Verarbeitungsschritt 29 stereo-lag safety: corrected introduced lag delta=%d samples (in=%d out=%d corrected=%d)",
            lag_delta,
            lag_in,
            lag_out,
            lag_corr,
        )
        clipped_corrected: np.ndarray = np.clip(corrected, -1.0, 1.0).astype(np.float32)
        return clipped_corrected, stats

    def _process_channel_omlsa(
        self,
        channel: np.ndarray,
        sample_rate: int,
        hf_low: float,
        hf_high: float,
        material: "MaterialType",
        intensity_scale: float = 1.0,
        masking_result=None,  # §Psychoacoustic: pre-computed MaskingResult for this channel
        linked_sidechain: np.ndarray | None = None,  # §2.51: Mid-channel for linked gain computation
    ) -> np.ndarray:
        """STFT-OMLSA-Verarbeitung: HF-selektive Rauschunterdrückung (Cohen 2002/2003).

        Algorithmus:
            1. STFT (nperseg=2048, noverlap=1536)
            2. IMCRA-Rauschschätzung im HF-Bereich [hf_low, hf_high]
            3. OMLSA-Gain: G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p
            4. Bins < hf_low: G=1.0 (unangetastet — Tieftonschutz)
            5. Cappé-Glättung: alpha_g = 0.85
            6. ISTFT + NaN/Clip-Schutz

        §2.51 Linked-Sidechain: when ``linked_sidechain`` (Mid) is provided, the
        IMCRA noise estimation and OMLSA gain are computed from the sidechain signal
        so that L and R receive the **identical** gain mask — stereo-coherent.

        Args:
            channel:           Mono-Audio (1D float32)
            sample_rate:       Abtastrate in Hz
            hf_low:            Untere HF-Grenze (Hz), z.B. 8000
            hf_high:           Obere HF-Grenze (Hz), z.B. 18000
            material:          MaterialType für G_floor
            linked_sidechain:  Optional Mid-channel for linked stereo gain computation

        Returns:
            processed: Restauriertes Mono-Audio (gleiche Länge wie channel)
        """
        # Material-adaptiver G_floor
        G_floor_map = {
            "SHELLAC": 0.12,
            "VINYL": 0.10,
            "TAPE": 0.08,
            "REEL_TAPE": 0.07,
            # §SibilantProtect: Kassette benötigt höheren G_floor — für Formant-Schutz
            # (mp3_low-Kette: HF bereits komprimiert, OMLSA überätz Formanten bei G_floor=0.10)
            "CASSETTE": 0.20,
            "DAT": 0.06,
        }
        mat_name = getattr(material, "name", str(material)).upper()
        G_floor = G_floor_map.get(mat_name, 0.10)
        intensity_scale = float(np.clip(intensity_scale, 0.0, 1.0))
        # Raise floor towards 1.0 for conservative locality handling.
        G_floor = float(np.clip(1.0 - intensity_scale * (1.0 - G_floor), 0.0, 1.0))

        # STFT + OMLSA via MRSA 5-zone processing (§DSP-Spezialregeln)
        # §2.51: When linked_sidechain is provided, IMCRA/OMLSA gain is computed on
        # the sidechain (Mid) but applied to this channel's STFT magnitudes.
        # §2.63: Reflect-Padding VOR OMLSA-STFT (root-cause boundary fix §2.63)
        # Provides context for the IMCRA noise estimator at signal boundaries.
        _pad_len_29 = 2048
        _channel_padded_29 = np.pad(channel, _pad_len_29, mode="reflect")
        _sidechain_padded_29 = (
            np.pad(linked_sidechain, _pad_len_29, mode="reflect") if linked_sidechain is not None else None
        )
        processed = self._process_channel_omlsa_mrsa(
            _channel_padded_29,
            sample_rate,
            hf_low,
            hf_high,
            material,
            intensity_scale,
            linked_sidechain=_sidechain_padded_29,
        )
        # §2.63: Strip reflect-padding deterministisch (Originallänge wiederherstellen)
        processed = processed[_pad_len_29 : _pad_len_29 + len(channel)]
        if len(processed) < len(channel):
            processed = np.pad(processed, (0, len(channel) - len(processed)))
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # §4.5 Psychoakustischer Masking-Gain-Clamp (ISO 11172-3, Painter & Spanias 2000)
        # Berechnet auf Input-Audio → Schutzmaske für Stille / ungemaskierte Bereiche
        try:
            # §2.62 Canonical iso11172 masking guard (replaces legacy psychoacoustic_masking_model).
            # masking_result (if pre-computed upstream) may be a legacy MaskingResult object with
            # .gain_modifier or a raw ndarray from compute_masking_threshold_iso11172.
            # Always recompute via the canonical iso11172 API for consistency.
            _ = masking_result  # unused — iso11172 recomputes from channel directly
            from backend.core.dsp.psychoacoustics import compute_masking_threshold_iso11172 as _cmask_p29_td

            # §2.51 Stereo-Kohärenz: Masking aus Mid-Sidechain berechnen,
            # damit L und R identische Gain-Maske erhalten (kanalasymmetrisches Rauschen).
            _pmm_source = linked_sidechain if linked_sidechain is not None else channel
            _pmm_arr = _cmask_p29_td(_pmm_source.astype(np.float32), sample_rate, n_fft=2048, hop_length=512)
            # _pmm_arr: (n_freq_bins, n_frames), values ∈ [0,1]: 1.0 = full masking
            # max over freq axis → (n_frames,): if ANY band has signal, frame is unmasked (gain=1).
            _pmm_gain_t = np.max(_pmm_arr, axis=0).astype(np.float32)
            _hop = 512  # entspricht nperseg=2048, noverlap=1536
            _pmm_centers = np.arange(len(_pmm_gain_t)) * float(_hop) + _hop * 0.5
            _pmm_x = np.arange(len(processed), dtype=np.float32)
            _gain_samples = np.interp(_pmm_x, _pmm_centers, _pmm_gain_t).astype(np.float32)
            # §2.45a / §2.54: Scale masking suppression toward 1.0 by intensity_scale.
            # At low PMGG strength (e.g. 0.14) the masking clamp must be near-transparent
            # — otherwise full suppression runs regardless of strength causing unexpected
            # RMS drops, makeup-gain overshoot and TFS coherence degradation.
            _gain_samples_scaled = (1.0 + intensity_scale * (_gain_samples - 1.0)).astype(np.float32)
            processed = np.clip((processed * _gain_samples_scaled).astype(np.float32), -1.0, 1.0)
            logger.debug(
                "🎭 PsychoacousticMasking [Verarbeitungsschritt29]: mean_floor=%.3f mean_gain=%.3f scaled_mean=%.3f (scale=%.2f)",
                float(np.mean(_pmm_arr)),
                float(np.mean(_pmm_gain_t)),
                float(np.mean(_gain_samples_scaled)),
                intensity_scale,
            )
        except Exception as _pmm_exc:
            logger.debug("PsychoacousticMaskingModel nicht verfügbar: %s", _pmm_exc)

        # §2.36 Phonem-Schutz: Plosiv-Burst-Frames aus Original restaurieren (sample-level).
        try:
            from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_pmask_p29o

            _hop_29o = 512
            _pmask_29o = _get_pmask_p29o(channel.astype(np.float32), sample_rate, hop_length=_hop_29o)
            if np.any(_pmask_29o):
                _n_29o = len(channel)
                for _fi_29o, _fp_29o in enumerate(_pmask_29o):
                    if _fp_29o:
                        _fs_29o = _fi_29o * _hop_29o
                        _fe_29o = min(_n_29o, _fs_29o + _hop_29o)
                        processed[_fs_29o:_fe_29o] = channel[_fs_29o:_fe_29o]
        except Exception as _pm_29o_exc:
            logger.debug("§2.36 Verarbeitungsschritt_29 _omlsa Phonem-Mask (nicht blockierend): %s", _pm_29o_exc)

        processed_result: np.ndarray = processed
        # §v10.304: STFT/ISTFT produces frame-boundary length drift — normalize to input.
        if len(processed_result) != len(channel):
            _min_len = min(len(processed_result), len(channel))
            processed_result = processed_result[:_min_len]
            if _min_len < len(channel):
                processed_result = np.pad(processed_result, (0, len(channel) - _min_len))
        return processed_result

    def _process_channel_omlsa_mrsa(
        self,
        channel: np.ndarray,
        sample_rate: int,
        hf_low: float,
        hf_high: float,
        material: "MaterialType",
        intensity_scale: float = 1.0,
        linked_sidechain: np.ndarray | None = None,
    ) -> np.ndarray:
        """MRSA 5-zone OMLSA/IMCRA tape-hiss reduction with PGHI phase reconstruction.

        Multi-Resolution Spectral Analysis (MRSA): each frequency zone is processed
        at its optimal time-frequency resolution. Zones below hf_low receive pass-through
        (gain=1.0), protecting low-frequency content. PGHI replaces plain iSTFT.

        §2.51 Linked-Sidechain: when ``linked_sidechain`` (Mid) is provided, IMCRA noise
        estimation runs on the sidechain signal, producing a stereo-coherent gain mask
        that is applied to the actual channel's STFT. This prevents L/R asymmetric
        gain modulation that causes phantom-center instability on tape material.

        Args:
            channel:           Mono audio [1D float32].
            sample_rate:       Must be 48000.
            hf_low:            Lower HF gate boundary (Hz), e.g. 8000.
            hf_high:           Upper HF gate boundary (Hz), e.g. 18000.
            material:          MaterialType for G_floor selection.
            intensity_scale:   Locality factor ∈ [0, 1].
            linked_sidechain:  Optional Mid-channel for linked stereo gain computation.

        Returns:
            Processed mono audio, same length as input.
        """
        # **GUARD: Short-Audio-Buffer (§2.47, §0 Primum non nocere)**
        MIN_AUDIO_SAMPLES = 512  # 10 ms @ 48 kHz
        if len(channel) < MIN_AUDIO_SAMPLES:
            logger.debug(
                "Verarbeitungsschritt_29: audio too short (%d < %d), passthrough", len(channel), MIN_AUDIO_SAMPLES
            )
            passthrough_result: np.ndarray = np.asarray(channel, dtype=np.float32).copy()
            return passthrough_result

        n = len(channel)
        nyquist = float(sample_rate // 2)
        eps = 1e-10

        # Material-adaptive G_floor — §2.62: absolute minimum 0.10 (VERBOTEN: G_floor < 0.10)
        G_floor_map = {"SHELLAC": 0.12, "VINYL": 0.10, "TAPE": 0.10, "REEL_TAPE": 0.10, "CASSETTE": 0.20, "DAT": 0.10}
        mat_name = getattr(material, "name", str(material)).upper()
        G_floor = G_floor_map.get(mat_name, 0.10)
        intensity_scale = float(np.clip(intensity_scale, 0.0, 1.0))
        G_floor = float(np.clip(1.0 - intensity_scale * (1.0 - G_floor), 0.10, 1.0))  # hard min §2.62
        # §EraTarget: era-adaptive G_floor lift (v10.0.0) — preserves authentic carrier noise texture.
        # E.g. 1935 shellac → _era_nr_g_floor=0.35 ensures vintage ambience survives OMLSA.
        G_floor = float(max(G_floor, getattr(self, "_era_nr_g_floor", 0.10)))
        runtime_profile = getattr(self, "_omlsa_runtime_profile_current", {})
        q = float(np.clip(runtime_profile.get("omlsa_q", 0.5), 0.35, 0.65))
        b_min = float(np.clip(runtime_profile.get("imcra_b_min", 1.66), 1.40, 1.90))
        alpha_g = float(np.clip(runtime_profile.get("imcra_alpha_g", 0.85), 0.75, 0.92))
        hf_floor_scale = float(np.clip(runtime_profile.get("hf_floor_scale", 0.45), 0.30, 0.65))

        # Reference STFT (win=2048, 75 % overlap) — on channel (for magnitude application)
        REF_WIN = min(2048, len(channel) // 2) if len(channel) >= 128 else 64
        REF_HOP = 512
        REF_NOVERLAP = REF_WIN - REF_HOP
        f_ref, _, Zxx_ref = signal.stft(
            channel, fs=sample_rate, nperseg=REF_WIN, noverlap=REF_NOVERLAP, window="hann", boundary="even"
        )
        n_bins, n_t = f_ref.shape[0], Zxx_ref.shape[1]

        # §2.51 Linked-Sidechain: compute gain from Mid sidechain for stereo coherence.
        # If no sidechain is provided, gain is computed from the channel itself (mono path).
        _gain_source = linked_sidechain if linked_sidechain is not None else channel

        # §2.51 Linked-Sidechain: HF-Salienz-Guard und Masking-Floor MÜSSEN ebenfalls aus dem
        # Mid-Sidechain berechnet werden, damit L und R identische Gain-Modifikationen erhalten.
        # Nur der Rekonstruktions-STFT (Zxx_ref) bleibt kanal-spezifisch (Phase-Information).
        if linked_sidechain is not None:
            _, _, _Zxx_ref_for_guard = signal.stft(
                linked_sidechain,
                fs=sample_rate,
                nperseg=REF_WIN,
                noverlap=REF_NOVERLAP,
                window="hann",
                boundary="even",
            )
            _channel_for_masking = linked_sidechain
        else:
            _Zxx_ref_for_guard = Zxx_ref
            _channel_for_masking = channel

        G_acc = np.zeros((n_bins, n_t), dtype=np.float64)
        w_acc = np.zeros(n_bins, dtype=np.float64)

        # §2.62 Psychoakustischer Masking-Guard (ISO 11172-3):
        # OMLSA-Gain darf pro Frequenzbin nicht unter die Maskierungsschwelle fallen.
        # Rauschen das vom Musiksignal maskiert wird erzeugt bei Entfernung klinisches Klangbild.
        _masking_floor_p29: np.ndarray | None = None
        _masking_freqs_p29: np.ndarray | None = None
        try:
            from backend.core.dsp.psychoacoustics import compute_masking_threshold_iso11172 as _cmask_p29

            _mask_ratio_p29 = _cmask_p29(_channel_for_masking, sample_rate, n_fft=2048, hop_length=512)
            _mask_arr_p29 = np.asarray(_mask_ratio_p29, dtype=np.float32)
            _masking_floor_arr_p29 = _mask_arr_p29.mean(axis=1).astype(np.float32)  # (n_freq_2048,)
            _masking_floor_p29 = _masking_floor_arr_p29
            _masking_freqs_p29 = np.linspace(0.0, sample_rate / 2.0, _mask_arr_p29.shape[0], dtype=np.float32)
            logger.debug(
                "§2.62 Verarbeitungsschritt_29 Masking-Guard: mean_floor=%.3f", float(_masking_floor_arr_p29.mean())
            )
        except Exception as _msk_exc_p29:
            logger.debug(
                "§2.62 Verarbeitungsschritt_29 Masking-Guard nicht verfügbar (nicht blockierend): %s", _msk_exc_p29
            )

        for zone_name, zone_win, zone_hop, f_low, f_high in self._MRSA_ZONES:
            try:
                # §2.51: STFT for gain computation uses _gain_source (Mid sidechain
                # for stereo, or channel itself for mono).
                if n >= zone_win * 2:
                    zone_noverlap = zone_win - zone_hop
                    f_z, _, Zxx_z = signal.stft(
                        _gain_source,
                        fs=sample_rate,
                        nperseg=zone_win,
                        noverlap=zone_noverlap,
                        window="hann",
                        boundary="even",
                    )
                else:
                    # Fallback to reference STFT — recompute from gain source if linked
                    if linked_sidechain is not None:
                        f_z, _, Zxx_z = signal.stft(
                            _gain_source,
                            fs=sample_rate,
                            nperseg=REF_WIN,
                            noverlap=REF_NOVERLAP,
                            window="hann",
                            boundary="even",
                        )
                    else:
                        f_z, Zxx_z = f_ref, Zxx_ref
                    zone_win, zone_hop = REF_WIN, REF_HOP

                mag_z = np.abs(Zxx_z)
                n_z_t = mag_z.shape[1]
                frames_per_sec_z = float(sample_rate / zone_hop)
                M_z = max(3, int(1.5 * frames_per_sec_z))

                # Vectorised IMCRA: sliding minimum as noise estimate (Cohen 2003)
                power_z = mag_z**2
                S_min_z = _min_filter1d_p29(power_z, size=M_z, axis=1, mode="reflect")
                noise_sq_z = np.maximum(b_min * S_min_z, eps)

                # Vectorised OMLSA gain + MMSE-LSA (Ephraim-Malah 1985, §DSP-Instructions)
                gamma_z = power_z / noise_sq_z
                xi_z = np.maximum(gamma_z - 1.0, 0.0)
                nu_z = np.clip(xi_z * gamma_z / (xi_z + 1.0 + eps), 0.0, 500.0)
                lam_z = np.exp(np.clip(-xi_z + nu_z, -50.0, 50.0))
                p_z = 1.0 / (1.0 + q / ((1.0 - q) * lam_z + eps))
                G_H1_z = xi_z / (xi_z + 1.0 + eps)
                # MMSE-LSA: G = G_H1 * exp(0.5 * E1(ν))  where ν = G_H1 * γ = nu_z
                # Prevents Musical Noise: low-SNR bins get less suppression than Wiener.
                G_mmse_lsa_z = G_H1_z * _exp1_p29_gain(nu_z)
                G_mmse_lsa_z = np.clip(np.nan_to_num(G_mmse_lsa_z, nan=G_floor), G_floor, 1.0)
                log_G_z = (1.0 - p_z) * np.log(G_floor + eps) + p_z * np.log(np.maximum(G_mmse_lsa_z, eps))
                G_z = np.exp(np.clip(log_G_z, np.log(G_floor + eps), 0.0))
                G_z = np.clip(np.nan_to_num(G_z, nan=G_floor), G_floor, 1.0)

                # §4.8a-ii PRESERVE-Mask: G_eff = mask * G_PRESERVE_FLOOR + (1 - mask) * G_z
                # Floort OMLSA/MMSE-LSA-Gain in PRESERVE-Bins (Shellac H2/H4, Tape Bias etc.) auf 0.90.
                _pm_p29 = getattr(self, "_preserve_mask_p29", None)
                if _pm_p29 is not None and _pm_p29.size > 0:
                    _G_PRES_P29 = 0.90
                    _n_bins_z = G_z.shape[0]
                    if len(_pm_p29) != _n_bins_z:
                        _pm_interp_z = np.interp(
                            np.arange(_n_bins_z),
                            np.linspace(0, _n_bins_z - 1, len(_pm_p29)),
                            _pm_p29.astype(np.float64),
                        ).astype(np.float64)
                    else:
                        _pm_interp_z = _pm_p29.astype(np.float64)
                    _pm_col_z = _pm_interp_z[:, np.newaxis]  # (n_bins, 1)
                    G_z = _pm_col_z * _G_PRES_P29 + (1.0 - _pm_col_z) * G_z
                    G_z = np.clip(np.nan_to_num(G_z, nan=G_floor), G_floor, 1.0)

                # §Gap5 EmotionalArc FrissonZone Schutz — §0p v10.0.0
                # ArcPlan aus _restoration_context: geschützte Zonen bekommen weniger NR.
                _arc_plan_p29 = getattr(self, "_arc_plan_p29", None)
                if _arc_plan_p29 is None:
                    _arc_plan_p29 = getattr(self, "_restoration_context_p29", {}).get("arc_protection_weights")
                if _arc_plan_p29 is not None and hasattr(_arc_plan_p29, "weight_at"):
                    try:
                        _n_frames_z = G_z.shape[1]
                        _hop_s_z = zone_hop / max(1, sample_rate)
                        _frame_times_z = np.arange(_n_frames_z, dtype=np.float64) * _hop_s_z
                        _arc_w_z = np.array(
                            [_arc_plan_p29.weight_at(float(t), float(t + _hop_s_z)) for t in _frame_times_z],
                            dtype=np.float64,
                        )
                        _protect_z = np.clip((_arc_w_z - 1.0) * 0.5, 0.0, 0.50)
                        G_z = G_z + _protect_z[np.newaxis, :] * (1.0 - G_z)
                        G_z = np.clip(G_z, G_floor, 1.0)
                    except Exception as _arc_z_exc:
                        logger.debug("§Gap5 Arc-Schutz Verarbeitungsschritt_29 nicht blockierend: %s", _arc_z_exc)

                # §2.62: Per-Frequenz-Masking-Floor — Signal-konditioniert (non-blocking).
                # Schützt signalpräsente Bins vor Überunterdrückung (§2.62).
                # Signal-konditioniert: Floor nur auf Bins mit G_z > _P29_SIGNAL_GATE anwenden.
                # Das verhindert, dass rausch-dominierte Bins (G_z klein) durch die Masking-Floor
                # "gerettet" werden — was andernfalls NR blockiert und bass_kraft/HF-Energie verfälscht.
                # Kalibration: _P29_SIGNAL_GATE=0.5 trennt Signal- von Rausch-Bins empirisch optimal.
                _P29_SIGNAL_GATE = 0.50  # Bins > 0.5 gelten als signalpräsent
                if _masking_floor_p29 is not None and _masking_freqs_p29 is not None:
                    try:
                        _mfloor_zone = np.interp(f_z, _masking_freqs_p29, _masking_floor_p29).astype(np.float32)
                        _mfloor_zone = np.minimum(_mfloor_zone, 0.65)  # Absoluter Cap: immer ≥ 35 % NR möglich
                        _signal_mask = G_z > _P29_SIGNAL_GATE  # (F, T) bool
                        G_z = np.where(
                            _signal_mask,
                            np.maximum(G_z, _mfloor_zone[:, np.newaxis]),
                            G_z,
                        )
                    except Exception as e:
                        logger.warning("Verarbeitungsschritt_29_tape_hiss_reduction.py::unbekannter Ersatzpfad: %s", e)
                        pass  # nie pipeline-blockierend

                # §v10.0.0: Stronger HF suppression in presence/air zones when DeepFilterNet absent.
                # DeepFilterNet removes residual hiss 2–16 kHz; without it, G_floor must be lower.
                # TAPE: 0.08 → 0.036, VINYL: 0.10 → 0.045, SHELLAC: 0.12 → 0.054 in these zones.
                if zone_name in ("presence", "air") and intensity_scale > 0.40:
                    _hf_floor = float(np.clip(G_floor * hf_floor_scale, 0.020, G_floor))
                    G_z = np.clip(G_z, _hf_floor, 1.0)

                # Zones below hf_low: pass-through (protect low frequencies)
                lf_mask_z = f_z < float(hf_low)
                G_z[lf_mask_z, :] = 1.0
                # Zones above hf_high (Nyquist region): pass-through
                if float(hf_high) < nyquist:
                    hf_mask_z = f_z > float(hf_high)
                    G_z[hf_mask_z, :] = 1.0

                # Cappé temporal smoothing via fast IIR
                G_z_sm = _lfilter_p29([1.0 - alpha_g], [1.0, -alpha_g], G_z, axis=1)
                G_z_sm = np.clip(np.nan_to_num(G_z_sm, nan=G_floor), G_floor, 1.0)

                # §SibilantProtect [RELEASE_MUST] (v10.0.0.x): Transient-aware Gain für
                # Presence-Zone bei Gesangsmaterial. Symmetrische Cappé-Glättung
                # (tau≈32 ms) verschluckt Sibilanten-Onsets (50-150 ms) → progressive
                # Sibilantenunterdrückung in dichten Vokalpassagen. Fix: paralleler
                # Fast-IIR (alpha_fast=alpha_g-0.30, tau≈8 ms) → Max beider Glättungen
                # → schneller Anstieg bei Sibilantenonset, sicherer Abfall danach.
                _panns_s_p29 = float(getattr(self, "_omlsa_panns_singing", 0.0))
                if zone_name == "presence" and _panns_s_p29 >= 0.25:
                    _alpha_fast_p29 = float(np.clip(alpha_g - 0.30, 0.20, 0.70))
                    _G_z_fast_p29 = _lfilter_p29([1.0 - _alpha_fast_p29], [1.0, -_alpha_fast_p29], G_z, axis=1)
                    _G_z_fast_p29 = np.clip(np.nan_to_num(_G_z_fast_p29, nan=G_floor), G_floor, 1.0)
                    G_z_sm = np.maximum(G_z_sm, _G_z_fast_p29)
                    G_z_sm = np.clip(G_z_sm, G_floor, 1.0)

                # Extract zone frequency range
                zm_z = (f_z >= float(f_low)) & (f_z <= float(f_high))
                if not np.any(zm_z):
                    continue
                f_z_zone = f_z[zm_z]
                G_zone = G_z_sm[zm_z, :]

                # Reference bins for this zone (with crossfade bandwidth)
                ref_zm = (f_ref >= max(0.0, float(f_low) - self._MRSA_CROSSFADE_BW_HZ)) & (
                    f_ref <= min(nyquist, float(f_high) + self._MRSA_CROSSFADE_BW_HZ)
                )
                if not np.any(ref_zm):
                    continue
                f_ref_zone = f_ref[ref_zm]
                ref_indices = np.where(ref_zm)[0]
                n_ref_zone = len(ref_indices)

                # Temporal resampling — vectorised (replaces per-bin Python loop)
                if n_z_t != n_t and len(f_z_zone) > 0:
                    # Both src and dst are regularly spaced → integer-index lerp
                    _idx_c = np.linspace(0.0, n_z_t - 1, n_t)
                    _idx_lo = np.clip(np.floor(_idx_c).astype(int), 0, n_z_t - 2)
                    _idx_hi = _idx_lo + 1
                    _frac = (_idx_c - _idx_lo)[np.newaxis, :]  # (1, n_t)
                    G_zone_t = ((1.0 - _frac) * G_zone[:, _idx_lo] + _frac * G_zone[:, _idx_hi]).astype(np.float64)
                else:
                    G_zone_t = G_zone.astype(np.float64)

                # Frequency interpolation — vectorised (replaces n_t Python loop iterations)
                G_ref_zone = np.empty((n_ref_zone, n_t), dtype=np.float64)
                if len(f_z_zone) >= 2:
                    # Precompute interpolation weights (fixed for all time frames)
                    _src_x = np.asarray(f_z_zone, dtype=np.float64)
                    _dst_x = np.asarray(f_ref_zone, dtype=np.float64)
                    _i = np.searchsorted(_src_x, _dst_x, side="right") - 1
                    _i_lo = np.clip(_i, 0, len(_src_x) - 2)
                    _i_hi = _i_lo + 1
                    _dx = _src_x[_i_hi] - _src_x[_i_lo]
                    _frac2 = np.clip((_dst_x - _src_x[_i_lo]) / np.maximum(_dx, 1e-12), 0.0, 1.0)
                    # Handle left / right extrapolation
                    _left = _dst_x < _src_x[0]
                    _right = _dst_x > _src_x[-1]
                    _frac2[_left] = 0.0
                    _frac2[_right] = 1.0
                    _i_lo[_left] = 0
                    _i_hi[_left] = 0
                    _i_lo[_right] = len(_src_x) - 1
                    _i_hi[_right] = len(_src_x) - 1
                    # Matrix lerp: (n_ref, n_t)
                    G_ref_zone = (1.0 - _frac2[:, None]) * G_zone_t[_i_lo, :] + _frac2[:, None] * G_zone_t[_i_hi, :]
                elif len(f_z_zone) == 1:
                    G_ref_zone[:, :] = G_zone_t[0:1, :]
                else:
                    continue

                # Hanning crossfade weights
                if n_ref_zone > 2:
                    hann_w = np.hanning(n_ref_zone + 2)[1:-1]
                    hann_w = np.clip(hann_w, 1e-3, 1.0)
                else:
                    hann_w = np.ones(n_ref_zone)

                # Vectorised accumulation (replaces per-bin Python loop)
                G_acc[ref_indices, :] += hann_w[:, None] * G_ref_zone
                w_acc[ref_indices] += hann_w

            except Exception as zone_exc:
                logger.warning("MRSA Verarbeitungsschritt 29 zone '%s' fehlgeschlagen: %s", zone_name, zone_exc)
                continue

        # Combine zone gains; unprocessed bins → pass-through
        valid = w_acc > 0.0
        G_combined = np.ones((n_bins, n_t), dtype=np.float32)
        G_combined[valid, :] = (G_acc[valid, :] / w_acc[valid, np.newaxis]).astype(np.float32)
        G_combined = np.clip(np.nan_to_num(G_combined, nan=1.0), 0.0, 1.0)

        # HF detail protection: preserve salient tape harmonics/transients in 6-18 kHz
        # while still reducing stationary hiss in low-salience bins.
        _mat = getattr(material, "name", str(material)).upper()
        _base_floor_by_mat = {
            "TAPE": 0.11,
            "REEL_TAPE": 0.10,
            "VINYL": 0.09,
            "SHELLAC": 0.09,
            "DAT": 0.08,
        }
        _hf_guard_low = max(float(hf_low), 6000.0)
        _hf_guard_high = min(float(hf_high), 18000.0)
        _hf_guard_mask = (f_ref >= _hf_guard_low) & (f_ref <= _hf_guard_high)
        if np.any(_hf_guard_mask):
            _mag_hf = np.abs(_Zxx_ref_for_guard[_hf_guard_mask, :]).astype(np.float64)
            _bin_sal = np.median(_mag_hf, axis=1)
            _bin_den = float(np.percentile(_bin_sal, 95) + eps)
            _bin_sal_n = np.clip(_bin_sal / _bin_den, 0.0, 1.0)

            _frame_den = np.percentile(_mag_hf, 90, axis=1, keepdims=True) + eps
            _frame_sal_n = np.clip(_mag_hf / _frame_den, 0.0, 1.0)

            _base_floor = float(_base_floor_by_mat.get(_mat, 0.09))
            # With higher intensity we still preserve enough HF detail to avoid dullness.
            _floor_min = float(np.clip(_base_floor - 0.03 * intensity_scale, 0.07, 0.16))
            _bin_floor = np.clip(_floor_min + 0.12 * _bin_sal_n, _floor_min, 0.30)
            _dyn_floor = _bin_floor[:, None] + 0.08 * _frame_sal_n
            _dyn_floor = np.clip(_dyn_floor, _bin_floor[:, None], 0.36).astype(np.float32)
            G_combined[_hf_guard_mask, :] = np.maximum(G_combined[_hf_guard_mask, :], _dyn_floor)

        # §2.36 Phonem-Schutz: Plosiv-Burst-Frames (/p/,/t/,/k/) via get_phoneme_mask()
        # schützen — AR-Residual-Spikes dieser Konsonanten-Bursts weisen dasselbe
        # spektrale Profil auf wie Tape-Hiss → OMLSA reduziert Artikulation.
        # Bypass: G_combined[:, phoneme_frames] = 1.0 (kein Gain-Eingriff).
        try:
            from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_pmask_29

            _pmask_29 = _get_pmask_29(channel.astype(np.float32), sample_rate, hop_length=REF_HOP)
            if np.any(_pmask_29):
                _n_t_29 = G_combined.shape[1]
                _pidx_29 = np.where(_pmask_29[:_n_t_29])[0]
                if len(_pidx_29) > 0:
                    G_combined[:, _pidx_29] = 1.0
                    logger.debug(
                        "§2.36 Verarbeitungsschritt_29 Phonem-Bypass: %d/%d Frames auf G=1.0",
                        len(_pidx_29),
                        _n_t_29,
                    )
        except Exception as _pm29_exc:
            logger.debug("§2.36 Verarbeitungsschritt_29 Phonem-Mask (nicht blockierend): %s", _pm29_exc)

        # §v10.303.13 Transient-Guard: Schützt Onsets vor Überdämpfung
        try:
            from backend.core.dsp.transient_guard import compute_transient_mask

            _tm29 = compute_transient_mask(channel.astype(np.float32), sample_rate)
            if len(_tm29) > 0:
                if len(_tm29) != G_combined.shape[1]:
                    _t_src = np.linspace(0.0, 1.0, len(_tm29))
                    _t_dst = np.linspace(0.0, 1.0, G_combined.shape[1])
                    _tm29 = np.interp(_t_dst, _t_src, _tm29)
                _n_tm29 = int(np.sum(_tm29 > 0.3))
                if _n_tm29 > 0:
                    _tm2 = np.clip(_tm29, 0.0, 1.0)[np.newaxis, :]
                    G_combined = G_combined * (1.0 - _tm2 * 0.5) + _tm2 * 0.5
                    logger.info(
                        "§v10.303.13 Transient-Guard Verarbeitungsschritt29: %d Frames vor Hiss-Reduktion geschützt",
                        _n_tm29,
                    )
        except Exception as _tm29_exc:
            logger.debug("§v10.303.13 Verarbeitungsschritt29 Transient-Guard (nicht blockierend): %s", _tm29_exc)

        # Apply gain + iSTFT reconstruction.
        # NOTE: Zxx_proc preserves the original phase from Zxx_ref (G_combined is real positive,
        # so angle(G*Zxx_ref) == angle(Zxx_ref)).  PGHI is designed for magnitude-only STFTs
        # where phase is unknown — using it here would DISCARD the correct phase and generate
        # a new estimate, introducing ~38 ms STFT group-delay deviation → CIG rollback
        # (confirmed production 2026-04-25).  Use scipy.signal.istft directly.
        Zxx_proc = G_combined * np.abs(Zxx_ref) * np.exp(1j * np.angle(Zxx_ref))
        _, audio_out = signal.istft(
            Zxx_proc, fs=sample_rate, nperseg=REF_WIN, noverlap=REF_NOVERLAP, window="hann", boundary=True
        )

        audio_out = np.real(audio_out)
        audio_out = audio_out[:n]
        if len(audio_out) < n:
            audio_out = np.pad(audio_out, (0, n - len(audio_out)))
        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0).astype(np.float32)

        logger.debug(
            "MRSA Verarbeitungsschritt 29: 5 zones verarbeitet, valid_bins=%d/%d, G_mean=%.3f, linked_sidechain=%s",
            int(np.sum(valid)),
            n_bins,
            float(np.mean(G_combined)),
            linked_sidechain is not None,
        )
        audio_out_result: np.ndarray = np.asarray(audio_out, dtype=np.float32)
        return audio_out_result

    def _extract_band(self, signal_in: np.ndarray, sample_rate: int, low_freq: float, high_freq: float) -> np.ndarray:
        """Bandpass-Filterung f\u00fcr Metrik-Berechnung (Hilfsmethode)."""
        nyquist = sample_rate / 2
        low_norm = max(low_freq, 20.0)
        high_norm = min(high_freq, nyquist * 0.98)
        if low_norm >= high_norm:
            return signal_in.copy()
        sos = signal.butter(4, [low_norm, high_norm], btype="band", fs=sample_rate, output="sos")
        band_result: np.ndarray = np.asarray(signal.sosfilt(sos, signal_in), dtype=np.float32)
        return band_result

    def _estimate_noise_floor(self, band_signal: np.ndarray) -> float:
        """
        Legacy-Methode (10th-Percentile RMS) \u2014 nur als R\u00fcckwärtskompatibilitäts-Alias.
        Primitivere Schätzung; STFT-OMLSA via _process_channel_omlsa ist primär.
        """
        # Compute short-term RMS (10ms windows)
        window_samples = int(0.01 * self.sample_rate)
        num_windows = len(band_signal) // window_samples

        rms_vals = []
        for i in range(num_windows):
            start = i * window_samples
            end = start + window_samples
            window = band_signal[start:end]
            rms = np.sqrt(np.mean(window**2))
            rms_vals.append(rms)

        # 10th percentile as noise floor estimate
        noise_floor = np.percentile(rms_vals, 10) if rms_vals else 1e-10
        noise_floor_db = 20 * np.log10(noise_floor + 1e-10)

        return float(noise_floor_db)

    def _refine_hf_with_ml(self, audio: np.ndarray, sample_rate: int, panns_singing: float = 0.0) -> bool:
        """
        Refine HF hiss reduction (>2kHz) using DeepFilterNet v3 II.

        Band-Specific Strategy:
        1. DSP handles full spectrum with multi-band gates
        2. ML refines >2kHz region to remove residual hiss without artifacts
        3. <2kHz left untouched to preserve warmth and bass

        Args:
            audio: Audio array (mono or stereo, will be modified in-place)
            sample_rate: Sample rate

        Returns:
            True if successful, False otherwise
        """
        if not SOUNDFILE_AVAILABLE:
            logger.warning("soundfile not verfuegbar for ML HF refinement")
            return False

        plugin = self._get_deepfilternet_plugin()
        if plugin is None:
            return False

        # §2.47 ml_memory_budget guard (400 MB for DeepFilterNet v3 II)
        _dfn_release = None
        try:
            from backend.core.ml_memory_budget import release as _rel_29
            from backend.core.ml_memory_budget import try_allocate as _try_alloc_29

            if not _try_alloc_29("DeepFilterNet_phase29", 0.40):
                logger.debug("DeepFilterNet_Verarbeitungsschritt29: ml_memory_Grenze insufficient — DSP-Ersatzpfad")
                return False
            _dfn_release = _rel_29
        except ImportError:
            pass  # budget tracking unavailable — allow inference

        # §4.6b: PLM active-guard — prevents emergency-eviction during DeepFilterNet inference
        _plm29_dfn = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm29

            _plm29_dfn = _get_plm29()
            _plm29_dfn.set_active("DeepFilterNetV3", True)
        except Exception as e:
            logger.warning("Verarbeitungsschritt_29_tape_hiss_reduction.py::_refine_hf_with_ml Ersatzpfad: %s", e)

        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_temp:
                input_path = input_temp.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_temp:
                output_path = output_temp.name

            # Write audio to temp file
            sf.write(input_path, audio, sample_rate)

            # §0j [RELEASE_MUST] Energy-Bias PANNs-adaptiv:
            # Vokal (panns_singing >= 0.40) → leichtere HF-Unterdrückung (post_filter=False),
            # äquivalent zu energy_bias=-6 dB (Harmonik-Schutz §0j Spec 04).
            # Instrumental → aggressivere Unterdrückung (post_filter=True), äquivalent energy_bias=-9 dB.
            # §2.35c Register-adaptiver energy_bias: Kopfstimme -3 dB, Brust -6 dB, Fry/Flüstern -9 dB
            _is_vocal_content_p29 = float(panns_singing) >= 0.40
            _energy_bias_equiv_db = -6.0 if _is_vocal_content_p29 else -9.0
            if _is_vocal_content_p29:
                try:
                    from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal as _dvrt_p29

                    # §0p Passaggio-Schutz [RELEASE_MUST]: Temporal register detection mit Passaggio-Glättung.
                    # Bei Registerübergang (Brust→Kopf): energy_bias = max der Zonen (-3.0 dB) → schützt Übergänge.
                    _reg_seq_p29 = _dvrt_p29(audio, sample_rate, panns_singing=float(panns_singing))
                    _zone_biases_p29 = [_b for _, _, _, _b in _reg_seq_p29]
                    _energy_bias_equiv_db = max(_zone_biases_p29) if _zone_biases_p29 else -6.0
                    _has_passaggio_p29 = len({_r for _, _, _r, _ in _reg_seq_p29}) > 1
                    logger.debug(
                        "§0p Verarbeitungsschritt_29 Passaggio=%s energy_bias=%.1f dB zones=%d",
                        _has_passaggio_p29,
                        _energy_bias_equiv_db,
                        len(_reg_seq_p29),
                    )
                except Exception as _reg29_exc:
                    logger.debug("§0p Passaggio temporal Verarbeitungsschritt_29 (nicht blockierend): %s", _reg29_exc)
            _post_filter_p29 = not _is_vocal_content_p29  # True für Instrumental
            logger.debug(
                "§0j Verarbeitungsschritt_29 energy_bias_equiv=%.1f dB (panns_singing=%.2f vocal=%s post_filter=%s)",
                _energy_bias_equiv_db,
                float(panns_singing),
                _is_vocal_content_p29,
                _post_filter_p29,
            )

            # §0j vocal_energy_bias_db from VFA context — override local estimate
            _ctx_energy_bias_29 = float(self._restoration_context_p29.get("vocal_energy_bias_db", 0.0))
            if _ctx_energy_bias_29 < -6.0 and _is_vocal_content_p29:
                _energy_bias_equiv_db = _ctx_energy_bias_29
                logger.debug("§0j Verarbeitungsschritt_29 energy_bias from context=%.1f dB", _ctx_energy_bias_29)

            # Process with DeepFilterNet
            returncode, _stdout, _stderr = plugin.process(
                input_path,
                output_path,
                post_filter=_post_filter_p29,
            )

            if returncode == 0 and os.path.exists(output_path):
                # Read refined audio
                from backend.file_import import load_audio_file

                _res = load_audio_file(output_path, do_carrier_analysis=False)
                if not _res or "audio" not in _res:
                    return False
                refined = np.asarray(_res["audio"], dtype=np.float32)

                # Blend strategy: Keep <2kHz from original, use ML for >2kHz
                if refined.shape == audio.shape:
                    # Extract HF bands
                    sos_lp = signal.butter(4, self.ML_FREQUENCY_THRESHOLD_HZ, btype="low", fs=sample_rate, output="sos")
                    sos_hp = signal.butter(
                        4, self.ML_FREQUENCY_THRESHOLD_HZ, btype="high", fs=sample_rate, output="sos"
                    )

                    # Apply filters
                    is_stereo = audio.ndim == 2
                    # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) — LP+HP werden
                    # rekombiniert; sosfilt würde Zeitversatz + Filtereinschalttransiente erzeugen.
                    if is_stereo:
                        for ch in range(2):
                            lf_original = signal.sosfiltfilt(sos_lp, audio[:, ch])
                            hf_refined = signal.sosfiltfilt(sos_hp, refined[:, ch])
                            audio[:, ch] = lf_original + hf_refined
                    else:
                        lf_original = signal.sosfiltfilt(sos_lp, audio)
                        hf_refined = signal.sosfiltfilt(sos_hp, refined)
                        audio[:] = lf_original + hf_refined

                    logger.info("✅ ML HF refinement erfolgreich (>2kHz band)")
                    return True
                else:
                    logger.warning("Shape mismatch: %s vs %s", refined.shape, audio.shape)
                    return False
            else:
                logger.warning("DeepFilterNet fehlgeschlagen (returncode=%s)", returncode)
                return False

        except Exception as e:
            logger.error("ML HF refinement error: %s", e)
            return False

        finally:
            # Cleanup temp files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            if _dfn_release is not None:
                _dfn_release("DeepFilterNet_phase29")
            # §4.6b: release PLM active-guard
            if _plm29_dfn is not None:
                try:
                    _plm29_dfn.set_active("DeepFilterNetV3", False)
                except Exception as e:
                    logger.warning("Verarbeitungsschritt_29_tape_hiss_reduction.py::unbekannter Ersatzpfad: %s", e)

    def _apply_adaptive_gate(
        self, band_signal: np.ndarray, noise_floor_db: float, threshold_db: float, reduction_db: float, sample_rate: int
    ) -> np.ndarray:
        """
        Wendet adaptives Expander-Gate auf das Bandsignal an.

        Gate formula:
            gain = 1.0 if level > threshold
            gain = 10^(reduction_db / 20) if level < threshold
            Smooth transition in between
        """
        # Compute envelope (RMS with attack/release)
        envelope = self._compute_envelope(band_signal, sample_rate)

        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Compute gate threshold
        gate_threshold = noise_floor_db + threshold_db

        # Compute gains
        reduction_factor = 10 ** (reduction_db / 20)
        gains = np.ones_like(envelope)

        # Below threshold: apply reduction
        below_mask = envelope_db < gate_threshold
        gains[below_mask] = 1.0 / reduction_factor

        # Smooth gains (attack/release)
        gains_smoothed = self._smooth_gains(gains, sample_rate)

        # Apply gains
        processed = band_signal * gains_smoothed

        processed_result: np.ndarray = np.asarray(processed, dtype=np.float32)
        return processed_result

    def _compute_envelope(
        self, signal_in: np.ndarray, sample_rate: int, attack_ms: float = 5.0, release_ms: float = 50.0
    ) -> np.ndarray:
        """
        Berechnet envelope with attack/release smoothing.
        """
        # Rectify
        rectified = np.abs(signal_in)

        # Attack/release coefficients
        attack_coeff = np.exp(-1 / (attack_ms * 0.001 * sample_rate))
        release_coeff = np.exp(-1 / (release_ms * 0.001 * sample_rate))

        # Envelope follower
        envelope = np.zeros_like(rectified)
        envelope[0] = rectified[0]

        for i in range(1, len(rectified)):
            if rectified[i] > envelope[i - 1]:
                # Attack
                envelope[i] = attack_coeff * envelope[i - 1] + (1 - attack_coeff) * rectified[i]
            else:
                # Release
                envelope[i] = release_coeff * envelope[i - 1] + (1 - release_coeff) * rectified[i]

        envelope_result: np.ndarray = np.asarray(envelope, dtype=np.float32)
        return envelope_result

    def _smooth_gains(self, gains: np.ndarray, sample_rate: int, smooth_ms: float = 10.0) -> np.ndarray:
        """
        Glättet gain curve to prevent artifacts.
        """
        # Lowpass filter gains
        cutoff = 1000.0 / smooth_ms  # Lower cutoff for longer smooth_ms
        sos = signal.butter(2, cutoff, "low", fs=sample_rate, output="sos")
        gains_smoothed = signal.sosfilt(sos, gains)

        gains_result: np.ndarray = np.asarray(gains_smoothed, dtype=np.float32)
        return gains_result


# Test harness
if __name__ == "__main__":
    logger.debug("=== Verarbeitungsschritt 29: Tape Hiss Reduction v2 Test ===\n")

    processor = TapeHissReductionPhase(sample_rate=44100)

    # Test materials
    test_materials = [
        MaterialType.VINYL,
        MaterialType.TAPE,
        MaterialType.SHELLAC,
    ]

    for _test_material in test_materials:
        logger.debug("Testing %s:", _test_material.value.upper())

        # Create test signal: music + tape hiss
        sr = 44100
        duration = 2.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)

        # Music: 440 Hz tone with modulation
        np.random.seed(42)
        music = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))

        # Tape hiss: High-frequency noise (8-18 kHz dominant)
        hiss = 0.12 * np.random.randn(samples)
        sos_hiss = signal.butter(4, [8000, 18000], "band", fs=sr, output="sos")
        hiss = signal.sosfilt(sos_hiss, hiss)

        # Combine
        noisy = music + hiss

        # Create stereo
        _test_audio = np.column_stack([noisy, noisy])

        # Process
        _test_start = time.time()
        result = processor.process(_test_audio, sr, _test_material)
        _test_processed = result.audio
        meta = result.metadata or {}
        elapsed = time.time() - _test_start

        # Calculate HF noise reduction
        sos_hf = signal.butter(4, 8000, "high", fs=sr, output="sos")
        hf_orig = signal.sosfilt(sos_hf, _test_audio[:, 0])
        hf_proc = signal.sosfilt(sos_hf, _test_processed[:, 0])

        hf_reduction = 20 * np.log10(np.std(hf_orig) / (np.std(hf_proc) + 1e-10))

        # Display results
        logger.debug("  Gate Schwelle: %.1f dB", meta.get("gate_threshold_db", 0))
        logger.debug("  Reduction depth: %.1f dB", meta.get("reduction_depth_db", 0))
        logger.debug("  HF focus range: %s Hz", meta.get("hf_focus_range_hz", []))
        logger.debug("  Num bands: %s", meta.get("num_bands", 0))
        logger.debug("  HF reduction: %.2f dB", meta.get("hf_reduction_db", 0))
        logger.debug("  Per-band reduction: %s... (first 3)", meta.get("reduction_per_band_db", [])[:3])
        logger.debug("  Processing time: %.3fs", elapsed)
        logger.debug("  ✅\n")
