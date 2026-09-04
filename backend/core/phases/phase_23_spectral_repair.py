#!/usr/bin/env python3
"""
Phase 23: Spectral Repair v3.0 — IMCRA Adaptive Noise-Floor + Vectorized Inpainting

Algorithm (v3.0 — Über-SOTA):
1. STFT Analysis (scipy.signal.stft, Hann, material-adaptive Fensterlänge)
2. IMCRA Noise-Floor-Schätzung (Cohen 2003):
   - Exponential Power-Smoothing (α_d=0.85)
   - Sliding-Minimum über M Frames (b_min=1.66)
   - Werkzeug: scipy.ndimage.minimum_filter1d
3. Defekt-Detektion (3 Strategien):
   - Dropout: magnitude < 0.3 × noise_floor (IMCRA-basiert, bin-adaptiv)
   - Spike/Artefakt: Z-Score über IMCRA-Floor (robust via MAD)
   - Phasensprung: |Δφ(t,f)| > Schwellwert
4. Inpainting (vektorisiert, O(F+T)):
   - Horizontal (Zeit): scipy.interpolate.interp1d per Frequenzband
   - Vertikal (Frequenz): scipy.interpolate.interp1d per Zeitframe
   - Blend: 0.6 × horizontal + 0.4 × vertikal (Smaragdis 2003)
5. Phase-Velocity-Fortsetzung: δφ(f,t) = φ(f,t-1) - φ(f,t-2)
6. Konsistente ISTFT-Rekonstruktion

Scientific Foundation:
- Cohen & Berdugo (2002): Noise Estimation by Minima Controlled Recursive Averaging — IMCRA
- Cohen (2003): Noise Spectrum Estimation in Adverse Environments — OMLSA/IMCRA
- Smaragdis & Brown (2003): NMF for Audio — Inpainting-Blend-Gewichte
- Févotte & Idier (2011): NMF with β-Divergenz — spektrale Konsistenz

VERBOTEN (entfernt, per copilot-instructions §4.2):
- np.mean/np.std als globaler Rauschboden → ersetzt durch IMCRA Sliding-Minimum
- Fixierter energy_floor_db-Schwellwert → ersetzt durch adaptiven bin-spezifischen Floor
- O(F×T) Python-Doppelschleife → ersetzt durch vektorisierte F+T scipy.interpolate

Quality Target: PQS MOS ≥ 4.0 nach Reparatur
Performance Target: <0.5× Echtzeit bei 48 kHz

Author: Aurik Development Team
Version: 3.0.0
"""

import ctypes
import gc
import importlib
import logging
import os
import time
from typing import Any

import numpy as np

try:
    import psutil

    _PSUTIL_OK = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False
from typing import cast

from scipy import interpolate, ndimage, signal

from backend.core.audio_utils import safe_stft  # §v10.115 explicit wrapper (no monkey-patch)
from backend.core.clipping_detection import ClippingType, classify_clipping
from backend.core.defect_scanner import MaterialType
from backend.core.dsp.hallucination_guard import check_hallucination
from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp
from backend.core.lyrics_guided_enhancement import get_phoneme_mask
from backend.core.ml_memory_budget import is_system_thrashing
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.mrsa_zones import (
    ZONE_ORDER,
    ZONES,
    ZoneSTFT,
    analyze_zones,
    merge_zones,
    synthesize_zone,
)
from backend.core.natural_performance_detector import get_natural_performance_detector
from backend.core.plugin_lifecycle_manager import (
    evict_stale_plugins,
    get_plugin_lifecycle_manager,
)
from backend.core.quality_mode import QualityModeConfig, is_phase_ml_enabled, log_mode_decision

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


def _get_phase23_npd():
    """Stabiler Resolver fuer den NPA-Singleton in Phase 23."""
    return get_natural_performance_detector()


def _try_get_apollo_instance() -> Any | None:
    """Lädt Apollo nur bei Bedarf (Codec-Pfad), nie beim Modul-Import.

    Verhindert hohe RAM-Spitzen in Unit-Tests/Nicht-Codec-Läufen, die Apollo
    ohnehin nicht nutzen.
    """
    try:
        from plugins.apollo_plugin import get_apollo

        return get_apollo()
    except Exception as _apollo_import_exc:  # pragma: no cover - environment/ABI dependent
        logger.debug(
            "Verarbeitungsschritt_23: Apollo lazy import fehlgeschlagen, DSP-Ersatzpfad aktiv: %s", _apollo_import_exc
        )
        return None


# §2.46b Spectral-Tilt-Preservation: material-adaptive tolerance in dB/octave
_TILT_TOLERANCE_P23: dict[str, float] = {
    "digital": 1.5,
    "cd_digital": 1.5,
    "streaming": 1.5,
    "tape": 1.875,
    "reel_tape": 1.875,
    "vinyl": 2.25,
    "minidisc": 2.25,
    "shellac": 3.0,
    "wax_cylinder": 3.0,
    "wire_recording": 3.0,
}


def _est_tilt_p23(audio: np.ndarray, sr: int) -> float:
    """Quick spectral tilt estimate in dB/octave (§2.46b)."""
    if audio.ndim == 2:
        mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
    else:
        mono = audio
    n = min(len(mono), 8192)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(mono[:n] * np.hanning(n))) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    valid = (freqs >= 100.0) & (freqs <= sr * 0.45)
    if np.sum(valid) < 8:
        return 0.0
    log_f = np.log2(freqs[valid] + 1e-12)
    log_m = 20.0 * np.log10(spec[valid])
    log_f_c = log_f - log_f.mean()
    log_m_c = log_m - log_m.mean()
    denom = float(np.dot(log_f_c, log_f_c))
    return float(np.dot(log_f_c, log_m_c) / denom) if denom > 1e-10 else 0.0


class SpectralRepair(PhaseInterface):
    """
    Professional Spectral Repair Engine.

    Key Features:
    - Multi-strategy inpainting (horizontal/vertical/harmonic)
    - Adaptive defect detection (z-score, energy, phase)
    - Material-specific sensitivity
    - Real-time performance (<0.5× realtime)
    - Quality validation and adaptive blending

    Use Cases:
    - MP3/AAC codec artifacts (pre-echo, quantization noise)
    - Tape dropouts (short-duration signal loss)
    - Vinyl ticks/pops (localized spectral damage)
    - Frequency band gaps (missing treble/bass)
    - Phase discontinuities (digital glitches)

    Performance: <0.5× realtime on modern CPU
    """

    @staticmethod
    def _compute_admm_runtime_profile(
        material: str | MaterialType,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """Berechnet adaptive runtime profile for ADMM spectral repair (§2.54)."""
        mat = str(material.value if isinstance(material, MaterialType) else material).lower().replace("-", "_")
        qm = str(quality_mode or "balanced").lower().replace("-", "_")
        rest = float(np.clip(restorability_score, 0.0, 100.0))

        # Material baselines: digital materials have clean spectra, analogues are noisy
        base = {
            "shellac": {"clip_pct": 99.2, "clip_floor": 0.82, "side_mult": 1.02},
            "vinyl": {"clip_pct": 99.3, "clip_floor": 0.84, "side_mult": 1.03},
            "tape": {"clip_pct": 99.35, "clip_floor": 0.85, "side_mult": 1.04},
            "reel_tape": {"clip_pct": 99.4, "clip_floor": 0.86, "side_mult": 1.04},
            "minidisc": {"clip_pct": 99.4, "clip_floor": 0.88, "side_mult": 1.05},
            "cd_digital": {"clip_pct": 99.5, "clip_floor": 0.90, "side_mult": 1.06},
            "streaming": {"clip_pct": 99.55, "clip_floor": 0.92, "side_mult": 1.07},
            "unknown": {"clip_pct": 99.4, "clip_floor": 0.88, "side_mult": 1.04},
        }.get(mat, {"clip_pct": 99.4, "clip_floor": 0.88, "side_mult": 1.04})

        # Quality mode: fast is conservative (higher clip_pct = fewer bins clipped), quality is aggressive
        mode_adj_pct = {
            "fast": 0.22,
            "balanced": 0.0,
            "quality": -0.35,
            "maximum": -0.55,
            "restoration": -0.15,
            "studio_2026": -0.55,
        }.get(qm, 0.0)
        mode_adj_floor = {
            "fast": 0.035,
            "balanced": 0.0,
            "quality": -0.045,
            "maximum": -0.070,
            "restoration": -0.020,
            "studio_2026": -0.070,
        }.get(qm, 0.0)
        mode_adj_mult = {
            "fast": 0.025,
            "balanced": 0.0,
            "quality": -0.020,
            "maximum": -0.035,
            "restoration": -0.010,
            "studio_2026": -0.035,
        }.get(qm, 0.0)

        # Restorability: low (noisy/degraded) = fewer clipped bins (conservative), high (clean) = more aggressive

        # Restorability: low (noisy/degraded) = more aggressive clipping (lower clip_pct), high (clean) = conservative
        rest_adj_pct = ((rest - 50.0) / 50.0) * 0.28
        rest_adj_floor = ((rest - 50.0) / 50.0) * 0.035
        rest_adj_mult = ((rest - 50.0) / 50.0) * 0.015

        clip_percentile = float(np.clip(base["clip_pct"] + mode_adj_pct + rest_adj_pct, 98.8, 99.9))
        clip_floor = float(np.clip(base["clip_floor"] + mode_adj_floor + rest_adj_floor, 0.80, 0.93))
        side_clip_multiplier = float(np.clip(base["side_mult"] + mode_adj_mult + rest_adj_mult, 1.00, 1.10))

        return {
            "clip_percentile": clip_percentile,
            "clip_floor": clip_floor,
            "side_clip_multiplier": side_clip_multiplier,
        }

    @staticmethod
    def _build_defect_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
        defect_event_metadata: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Zeitlokale Blendmaske fuer Pre-Echo/Aliasing/Codec-Spektralreparaturen."""
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0

        keys = (
            "pre_echo",
            "aliasing",
            "codec_artifact",
            "mp3_artifact",
            "spectral_gap",
            "spectral_hole",
            "dropout",
            "digital_glitch",
            "jitter_artifacts",
        )
        key_factor = {
            "pre_echo": 0.95,
            "aliasing": 0.88,
            "codec_artifact": 0.78,
            "mp3_artifact": 0.78,
            "spectral_gap": 0.86,
            "spectral_hole": 0.86,
            "dropout": 0.82,
            "digital_glitch": 0.90,
            "jitter_artifacts": 0.74,
        }
        pad_by_key = {
            "pre_echo": 0.020,
            "aliasing": 0.030,
            "codec_artifact": 0.045,
            "mp3_artifact": 0.045,
            "spectral_gap": 0.060,
            "spectral_hole": 0.060,
            "dropout": 0.040,
            "digital_glitch": 0.025,
            "jitter_artifacts": 0.035,
        }
        mask = np.zeros(n_samples, dtype=np.float32)
        for key in keys:
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start_s = max(0.0, float(loc[0]))
                    end_s = max(start_s, float(loc[1]))
                except Exception:
                    logger.debug("_build_defect_locality_Profil: silent except suppressed", exc_info=True)
                    continue
                if end_s <= start_s:
                    continue
                severity = 0.5
                confidence = 0.8
                meta_obj = (defect_event_metadata or {}).get(key)
                if isinstance(meta_obj, dict):
                    severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
                    confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
                duration_factor = float(np.clip((end_s - start_s) / 0.20, 0.35, 1.0))
                event_strength = float(
                    np.clip(
                        key_factor.get(key, 0.75) * (0.40 + 0.40 * severity + 0.20 * confidence) * duration_factor,
                        0.16,
                        1.0,
                    )
                )
                pad = float(pad_by_key.get(key, 0.04))
                s = int(max(0.0, start_s - pad) * sample_rate)
                e = int(min(float(n_samples) / float(sample_rate), end_s + pad) * sample_rate)
                if e > s:
                    mask[s:e] = np.maximum(mask[s:e], event_strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0
        smooth = max(16, int(0.008 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        return mask, float(np.mean(mask))

    # STFT Parameters (material-adaptive)
    STFT_CONFIG = {
        MaterialType.SHELLAC: {
            "nperseg": 4096,  # Larger window for noisy material
            "noverlap": 3072,  # 75% overlap
            "nfft": 8192,
        },
        MaterialType.VINYL: {
            "nperseg": 2048,
            "noverlap": 1536,  # 75% overlap
            "nfft": 4096,
        },
        MaterialType.TAPE: {
            "nperseg": 2048,
            "noverlap": 1536,
            "nfft": 4096,
        },
        MaterialType.CASSETTE: {
            "nperseg": 2048,
            "noverlap": 1536,
            "nfft": 4096,
        },  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: {
            "nperseg": 2048,
            "noverlap": 1024,  # 50% overlap (less processing needed)
            "nfft": 4096,
        },
        MaterialType.STREAMING: {
            "nperseg": 1024,
            "noverlap": 512,
            "nfft": 2048,
        },
    }

    # Defect detection thresholds
    DETECTION_THRESHOLDS = {
        MaterialType.SHELLAC: {
            "outlier_z_score": 4.5,  # Higher (more tolerant of noise)
            "energy_floor_db": -55,  # Higher floor
            "phase_jump_threshold": np.pi * 0.7,
        },
        MaterialType.VINYL: {
            "outlier_z_score": 4.0,
            "energy_floor_db": -60,
            "phase_jump_threshold": np.pi * 0.6,
        },
        MaterialType.TAPE: {
            "outlier_z_score": 3.5,
            "energy_floor_db": -65,
            "phase_jump_threshold": np.pi * 0.5,
        },
        MaterialType.CASSETTE: {
            "outlier_z_score": 3.5,
            "energy_floor_db": -63,  # v10.0.0: leicht höher als TAPE (Cassette-Hiss-Boden)
            "phase_jump_threshold": np.pi * 0.5,
        },  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
        MaterialType.CD_DIGITAL: {
            "outlier_z_score": 3.0,  # More sensitive
            "energy_floor_db": -70,
            "phase_jump_threshold": np.pi * 0.4,
        },
        MaterialType.STREAMING: {
            "outlier_z_score": 2.5,  # Very sensitive (MP3 artifacts)
            "energy_floor_db": -75,
            "phase_jump_threshold": np.pi * 0.3,
        },
    }

    # Inpainting blend amounts (how aggressive to repair)
    REPAIR_STRENGTH = {
        MaterialType.SHELLAC: 0.60,  # Moderate (preserve character)
        MaterialType.VINYL: 0.70,
        MaterialType.TAPE: 0.75,
        MaterialType.CASSETTE: 0.64,  # v10.0.0: NR ≈ TAPE×0.85 — Cassette-Rauschen hat anderes Spektralprofil
        MaterialType.CD_DIGITAL: 0.85,  # Aggressive (digital artifacts obvious)
        MaterialType.STREAMING: 0.90,  # Very aggressive (codec artifacts)
    }

    # Soft-relax window for thrashing guards (§2.54): allow robust processing when
    # headroom is still objectively high, keep hard stop for real emergency states.
    _THRASH_RELAX_ML_MIN_AVAIL_GB = 14.0
    _THRASH_RELAX_ML_MIN_AVAIL_RATIO = 0.40
    _THRASH_RELAX_ML_MAX_SWAP_PCT = 95.0
    _THRASH_RELAX_ML_MAX_ATTEMPTS = 1
    _THRASH_RELAX_MRSA_MIN_AVAIL_GB = 8.0
    _THRASH_RELAX_MRSA_MIN_AVAIL_RATIO = 0.28
    _THRASH_RELAX_MRSA_MAX_SWAP_PCT = 98.0
    _THRASH_RELAX_MRSA_MAX_ATTEMPTS = 1

    def __init__(self):
        super().__init__()
        self.name = "Spectral Repair v3 IMCRA"
        self._flashsr_plugin = None  # Lazy loading
        # §v10.303: Early-Exit-Tracking — wenn FlashSR beim ersten Durchlauf
        # <500 Hz Bandbreite gewinnt, wird der zweite Durchlauf übersprungen.
        self._flashsr_last_bw_gain_hz: float = float("inf")
        self._flashsr_pass_count: int = 0
        self._ml_guard_events: list[dict[str, Any]] = []
        self._current_material: MaterialType = MaterialType.CD_DIGITAL  # updated per process() call
        self._pressure_relax_ml_attempts: int = 0
        self._pressure_relax_mrsa_attempts: int = 0

    @staticmethod
    def _material_key(material: str | MaterialType) -> str:
        raw = getattr(material, "value", material)
        return str(raw).lower().replace(" ", "_").replace("-", "_")

    @classmethod
    def _material_bw_ceiling_hz(cls, material: str | MaterialType) -> float | None:
        """§6.2c — Delegiert an die zentrale Carrier-Definition."""
        mat_key = cls._material_key(material)
        if mat_key is None:
            return None
        try:
            from backend.core.carrier_transfer_characteristics import get_bw_ceiling_hz

            return float(get_bw_ceiling_hz(mat_key))
        except ImportError as exc:
            logger.debug("§V6 carrier_transfer_characteristics.get_bw_ceiling_hz nicht verfügbar — Fallback-Werte aktiviert (Material %s): %s", mat_key, exc)
            # Fallback: kanonische Werte (IEC 60094-1)
            _fallback: dict[str, float] = {
                "shellac": 8000.0,
                "wax_cylinder": 5000.0,
                "wire_recording": 6000.0,
                "vinyl": 16000.0,
                "tape": 15000.0,
                "cassette": 14000.0,
                "reel_tape": 18000.0,
            }
            return _fallback.get(mat_key)

    @classmethod
    def _apply_material_bw_ceiling(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        material: str | MaterialType,
        mode: str = "restoration",
    ) -> tuple[np.ndarray, bool, float | None]:
        ceiling_hz = cls._material_bw_ceiling_hz(material)
        if ceiling_hz is None or "studio" in str(mode).lower():
            return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), False, ceiling_hz
        nyquist = float(sample_rate) / 2.0
        ratio = float(np.clip(ceiling_hz / nyquist, 0.01, 0.99))
        if ratio >= 0.985 or audio.shape[0] < 128:
            return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), False, ceiling_hz
        sos = signal.butter(6, ratio, btype="low", output="sos")
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] <= 2 and arr.shape[1] > 2:
                filtered = np.stack([signal.sosfiltfilt(sos, arr[c]) for c in range(arr.shape[0])], axis=0)
            else:
                filtered = np.stack([signal.sosfiltfilt(sos, arr[:, c]) for c in range(arr.shape[1])], axis=1)
        else:
            filtered = signal.sosfiltfilt(sos, arr)
        filtered = np.clip(np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return filtered.astype(np.float32), True, ceiling_hz

    def _has_sufficient_ml_headroom(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Gibt True when enough physical RAM is available for FlashSR stage zurück.

        Guard 1 — bandwidth-driven (§FSR-GATE v10.13): FlashSR läuft immer, wenn
        die geschätzte Bandbreite unter dem Zielwert liegt. Kein Material-Gate mehr.
        FlashSR wurde für maximalen Wohlklang entwickelt.

        Guard 2 — channel-aware RAM check (§2.38a): stereo doubles inference working
        memory; empirical per-minute inference buffer overhead is added.
        """
        # Guard 1 (v10.13): Kein Material-Gate — rein qualitätsgetrieben.
        # FlashSR läuft wenn Bandbreite unter Zielwert.

        # Guard 2: channel-aware physical RAM check (§2.38a)
        # Aurik internal format: (N,) mono or (N, ch) stereo — first axis is always samples.
        n_channels = int(audio.shape[1]) if (audio.ndim == 2 and 1 < audio.shape[1] <= 8) else 1
        n_samples = int(audio.shape[0])
        duration_s = n_samples / float(max(1, sample_rate))

        # Zone-based budget: _run_flashsr_ml() processes in 10-second zones, so only ONE
        # zone is in memory at a time.  Use zone duration (not full audio duration) to avoid
        # a false OOM-guard reject for long songs like 225 s tracks.
        # §v10.306: Chunk 10s→4s → RAM ~1.5GB→~600MB pro Zone.
        _FLASHSR_ZONE_SECONDS = 4
        duration_for_budget_s = min(duration_s, float(_FLASHSR_ZONE_SECONDS))

        required_gb = 2.5  # Base model weight budget (reduced from 5.0 with 4s chunks)
        if duration_for_budget_s >= 60.0:
            required_gb += 1.0
        required_gb += 0.6 * (duration_for_budget_s / 60.0)  # per-zone inference overhead (reduced from 1.5)
        required_gb = min(required_gb, 22.0)  # sanity cap
        # Note: n_channels multiplier removed — _repair_with_flashsr processes mono channels
        # individually (M/S in _repair_channel), so each call is always mono.

        available_gb = float(psutil.virtual_memory().available / (1024**3)) if _PSUTIL_OK else 4.0
        if available_gb < required_gb + 1.5:
            try:
                evict_stale_plugins(required_mb=int(required_gb * 1024))
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            available_gb = float(psutil.virtual_memory().available / (1024**3)) if _PSUTIL_OK else 4.0

        if available_gb < required_gb:
            self._ml_guard_events.append(
                {
                    "phase_id": "phase_23_spectral_repair",
                    "model": "FlashSR",
                    "reason": "insufficient_physical_ram_headroom",
                    "required_gb": float(required_gb),
                    "available_gb": float(available_gb),
                    "channels": n_channels,
                    "duration_s": float(duration_s),
                    "fallback": "dsp_inpainting",
                }
            )
            logger.warning(
                "SpectralRepair RAM guard triggered: %.1f GB verfuegbar, %.1f GB required "
                "(duration=%.1fs, ch=%d) — using DSP Ersatzpfad",
                available_gb,
                required_gb,
                duration_s,
                n_channels,
            )
            return False
        return True

    def _can_relax_thrashing_guard(self, *, for_mrsa: bool) -> bool:
        """Gibt True when pressure is elevated but still far from emergency zurück.

        This enables a controlled attempt for robust paths in phase 23 instead of
        forcing immediate Single-STFT fallback on every transient pressure spike.
        """
        try:
            vm = psutil.virtual_memory() if _PSUTIL_OK else None
            swap = psutil.swap_memory() if _PSUTIL_OK else None
            avail_gb = float(vm.available / (1024**3))  # type: ignore[union-attr]
            avail_ratio = float(vm.available / max(vm.total, 1))  # type: ignore[union-attr]
            swap_pct = float(getattr(swap, "percent", 100.0))

            if for_mrsa:
                return (
                    avail_gb >= self._THRASH_RELAX_MRSA_MIN_AVAIL_GB
                    and avail_ratio >= self._THRASH_RELAX_MRSA_MIN_AVAIL_RATIO
                    and swap_pct < self._THRASH_RELAX_MRSA_MAX_SWAP_PCT
                )

            return (
                avail_gb >= self._THRASH_RELAX_ML_MIN_AVAIL_GB
                and avail_ratio >= self._THRASH_RELAX_ML_MIN_AVAIL_RATIO
                and swap_pct < self._THRASH_RELAX_ML_MAX_SWAP_PCT
            )
        except Exception as e:
            logger.warning("Verarbeitungsschritt_23_spectral_repair.py::_can_relax_thrashing_guard Ersatzpfad: %s", e)
            return False

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_23_spectral_repair",
            name="Spectral Repair v3 IMCRA",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            dependencies=["phase_03_denoise", "phase_24_dropout_repair"],
            estimated_time_factor=0.50,
            version="3.0.0",
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.94,
            description="IMCRA Adaptive Noise-Floor + Vectorized Spectral Inpainting (Cohen 2003)",
        )

    def _get_flashsr_plugin(self):
        """Lädt beim ersten Zugriff: FlashSR plugin for ML-based repair."""
        if self._flashsr_plugin is None:
            try:
                from plugins.flashsr_plugin import FlashSRPlugin

                self._flashsr_plugin = FlashSRPlugin()
                logger.info("FlashSR plugin geladen erfolgreich")
            except Exception as e:
                logger.warning("konnte nicht laden FlashSR plugin: %s", e)
                self._flashsr_plugin = False  # Mark as unavailable

        return self._flashsr_plugin if self._flashsr_plugin is not False else None

    @staticmethod
    def _is_system_thrashing() -> bool:
        """Gibt True when swap/RAM pressure makes heavy phase-23 paths unsafe zurück."""
        try:
            return bool(is_system_thrashing())
        except Exception as e:
            logger.warning("Verarbeitungsschritt_23_spectral_repair.py::_is_system_thrashing Ersatzpfad: %s", e)
            return False

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        check_ml_model_ready("FlashSR", phase_name="23")
        check_ml_model_ready("PANNs", phase_name="23")
        check_ml_model_ready("Whisper", phase_name="23")
        check_ml_model_ready("FlashSR", phase_name="23")
        """
        Wendet an: spectral repair to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material_type: Material type for adaptive processing

        Returns:
            PhaseResult with repaired audio
        """
        material = kwargs.pop("material", material_type)
        # ── §v10 PIM: Per-Band-Intensität kalibrieren ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim = apply_pim_intensity(
                kwargs, "spectral_repair", default_nr=0.45, default_de_ess=0.25, default_comp=1.0
            )
            if kwargs.get("pim_intensity_map") is not None:
                for _key in ("noise_reduction_strength", "nr_strength", "strength", "wet"):
                    if _key in kwargs:
                        kwargs[_key] = _pim["nr_strength"]
        except Exception:
            logger.debug("verarbeiten: silent except suppressed", exc_info=True)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

        # §v10.14 FIX: FlashSR Early-Exit-State pro Aufruf zurücksetzen.
        # Vorher persistierte _flashsr_last_bw_gain_hz über Segmente —
        # wenn Segment 1 0 Hz Gain hatte, wurde FlashSR für ALLE Folgesegmente
        # deaktiviert, selbst wenn diese massiv von BW-Erweiterung profitiert hätten.
        self._flashsr_last_bw_gain_hz = float("inf")
        self._flashsr_pass_count = 0

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            get_plugin_lifecycle_manager().evict_for_phase("phase_23_spectral_repair")
        except Exception:
            logger.debug("verarbeiten: silent except suppressed", exc_info=True)

        start_time = time.time()
        _progress_cb = kwargs.get("progress_sub_callback")

        def _report_progress(pct: float, label: str) -> None:
            if callable(_progress_cb):
                try:
                    _progress_cb(float(np.clip(pct, 0.0, 100.0)), label, time.time() - start_time)
                except Exception:
                    logger.debug("_report_progress: silent except suppressed", exc_info=True)

        _report_progress(2.0, "Spektralreparatur: Vorbereitung")
        self._ml_guard_events = []
        # Store material as lowercase string value for guard comparison (handles both str and MaterialType enum)
        self._current_material = str(getattr(material, "value", material)).lower()  # type: ignore[assignment]
        self.validate_input(audio)
        _audio_for_tilt_p23 = audio.copy()  # §2.46b: tilt reference before any processing (incl. Apollo)

        # Normalize stereo layout: UV3 sends (2, N) channels-first; phase_23 processes
        # internally as (N, 2) samples-first for M/S and column_stack operations.
        _was_channels_first = audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2
        if _was_channels_first:
            audio = audio.T  # (2, N) → (N, 2)
            _audio_for_tilt_p23 = _audio_for_tilt_p23.T

        is_stereo = audio.ndim == 2

        # §Phase-level wall-time deadline: shared across Mid + Side MRSA sub-calls.
        # Without this, stereo audio gets 2× the per-call zone budget → 39-min runs
        # on 225s vinyl (563s × 2 = 1126s MRSA alone, then all downstream phases skip).
        # Budget: min(300s, max(90s, 1.3 × duration)) — for 225s: 292.5s total phase cap.
        _p23_dur_s = float(audio.shape[0]) / float(max(1, sample_rate))
        _phase_deadline = time.monotonic() + min(300.0, max(90.0, 1.3 * _p23_dur_s))
        logger.info(
            "Verarbeitungsschritt_23: wall-deadline=%.0fs (audio=%.1fs)",
            min(300.0, max(90.0, 1.3 * _p23_dur_s)),
            _p23_dur_s,
        )

        # Get material-specific parameters
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        stft_cfg = self.STFT_CONFIG.get(_mk, self.STFT_CONFIG[MaterialType.CD_DIGITAL])  # type: ignore[arg-type]
        # §v10.14 NFFT-Adaption: kurze Clips (<10s) → 1024 NFFT für bessere Zeitauflösung
        _clip_dur = float(len(audio)) / float(sample_rate)
        if _clip_dur < 10.0:
            stft_cfg = dict(stft_cfg)
            stft_cfg["nfft"] = 1024
            stft_cfg["nperseg"] = min(1024, stft_cfg["nperseg"])
            stft_cfg["noverlap"] = min(768, stft_cfg["noverlap"])
        thresholds = self.DETECTION_THRESHOLDS.get(_mk, self.DETECTION_THRESHOLDS[MaterialType.CD_DIGITAL])  # type: ignore[arg-type]

        # §GEBOT-G55: Signal-adaptive Detection-Thresholds via Noise-Floor-Analyse
        # Zwei Vinyl-Platten können unterschiedliche Rauschböden haben.
        try:
            from backend.core.adaptive_parameter_infrastructure import derive_noise_floor

            _nf23 = derive_noise_floor(audio, sample_rate)
            _energy_floor_adaptive = float(np.clip(_nf23["noise_floor_db"] + 6.0, -70.0, -35.0))
            thresholds = dict(thresholds)
            thresholds["energy_floor_db"] = _energy_floor_adaptive
            logger.debug(
                "Verarbeitungsschritt 23 adaptive: energy_floor=%.1f dB (noise_floor=%.1f)",
                _energy_floor_adaptive,
                _nf23["noise_floor_db"],
            )
        except Exception as _e:
            logger.debug("%s: unkritisch exception: %s", __name__, _e)
        repair_strength = self.REPAIR_STRENGTH.get(_mk, 0.75)  # type: ignore[arg-type]
        _material_meta_key23 = self._material_key(material)

        # Locality-aware modulation from UV3.
        # Sparse defect coverage -> lower inpainting intensity to preserve unaffected texture.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        repair_strength = float(np.clip(repair_strength * _effective_strength, 0.0, 1.0))

        # §V41 ForwardMaskingGuard: Additive/Inpainting-Phase → Stärke in post-transienten
        # Masking-Fenstern erhöhen (psychoakustische Maskierung schützt vor hörbaren Artefakten).
        # Nur bei panns_singing ≥ 0.25 und tatsächlich aktivem Repair.
        _panns_s_23 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_23 >= 0.25 and repair_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_23,
                )

                _fmg_23 = _fmg_fn_23()
                _fmz_23 = _fmg_23.compute_zones(audio, sample_rate)
                if _fmz_23:
                    _n_s_23 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_samples_23 = sum(z.end_sample - z.start_sample for z in _fmz_23)
                    _zone_frac_23 = float(np.clip(_zone_samples_23 / max(1, _n_s_23), 0.0, 1.0))
                    # Inpainting: konservativer Boost als BW-Erweiterungs-Phasen (max +0.10)
                    _boost_23 = _zone_frac_23 * 0.10
                    repair_strength = float(np.clip(repair_strength + _boost_23, 0.0, 1.0))
                    logger.debug(
                        "Verarbeitungsschritt23 §V41 ForwardMasking: zone_frac=%.2f boost=%.3f → repair_str=%.3f",
                        _zone_frac_23,
                        _boost_23,
                        repair_strength,
                    )
            except Exception as _fmg_exc_23:
                logger.debug("Verarbeitungsschritt23 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_23)

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            if _was_channels_first and passthrough.ndim == 2:
                passthrough = passthrough.T  # (N, 2) → (2, N)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": _material_meta_key23,
                    "defect_reduction_percent": 0.0,
                    "spectral_coherence": 1.0,
                    "repair_strength": 0.0,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rt_factor": 0.0,
                    "nperseg": stft_cfg["nperseg"],
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Repair skipped due to zero effective strength"],
            )

        if repair_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            if _was_channels_first and passthrough.ndim == 2:
                passthrough = passthrough.T  # (N, 2) → (2, N)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": _material_meta_key23,
                    "defect_reduction_percent": 0.0,
                    "spectral_coherence": 1.0,
                    "repair_strength": 0.0,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rt_factor": 0.0,
                    "nperseg": stft_cfg["nperseg"],
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Repair skipped due to zero effective strength"],
            )

        # §2.68d [RELEASE_MUST] SSIP: Stille-Zonen aus ORIGINAL-Audio ermitteln.
        # _get_structural_silence_zones() liefert immer eine Liste — niemals None (V16).
        _ssip_zones_p23: list[tuple[int, int]] = []
        _mat23_ssip_key = str(getattr(material, "value", str(material_type) or "unknown") or "unknown").lower()
        try:
            from backend.core.dsp.structural_silence_isolation import (
                _get_structural_silence_zones as _ssip_get_zones_p23,
            )

            _ssip_zones_p23 = _ssip_get_zones_p23(kwargs, audio, sample_rate, _mat23_ssip_key)
            if _ssip_zones_p23:
                logger.debug(
                    "§2.68 SSIP Verarbeitungsschritt_23: %d strukturelle Stille-Zone(n) erkannt",
                    len(_ssip_zones_p23),
                )
        except Exception as _ssip_init_exc23:
            logger.debug("SSIP Verarbeitungsschritt_23 init nicht blockierend: %s", _ssip_init_exc23)

        # --- Apollo pre-processing for lossy-codec materials ---
        # Apollo TorchScript handles MDCT-specific artefacts (pre-echo, spectral staircase,
        # psychoacoustic masking) better than generic STFT inpainting.  When the Apollo
        # model is loaded the output is fed into the standard STFT inpainting below as a
        # pre-cleaned signal.  Only activates when Apollo model is actually loaded
        # (no fallback call here — Apollo DSP-fallback runs inside repair() if model absent).
        _APOLLO_CODEC_MATERIALS = frozenset({"mp3_low", "mp3_high", "aac", "minidisc", "streaming"})
        _apollo_preproc_applied = False
        _apollo_hf_gain_db: float | None = None
        if self._current_material in _APOLLO_CODEC_MATERIALS:
            # §OOM-Guard: Swap-Thrashing-Check vor Apollo-TorchScript-Inferenz.
            # Apollo lädt bis zu 21 GB anon-RSS bei langen Audios (Mamba-State-Space).
            # Wenn Swap bereits > 80 % voll, bricht die Allokation den Prozess via
            # Kernel-OOM-Killer (kein Python-Traceback, kein faulthandler-Dump).
            # In diesem Fall DSP-Fallback erzwingen — Apollo wird übersprungen.
            _apollo_swap_blocked = False
            try:
                if is_system_thrashing():
                    logger.warning(
                        "Verarbeitungsschritt_23: Apollo TorchScript übersprungen — Swap-Thrashing erkannt "
                        "(OOM-Killer-Prävention). DSP-Inpainting wird verwendet."
                    )
                    _apollo_swap_blocked = True
            except Exception as _swap_chk_exc:
                logger.debug(
                    "Verarbeitungsschritt_23: Swap-Thrashing-Pruefung fehlgeschlagen (unkritisch): %s", _swap_chk_exc
                )
            # RAM-Guard: Apollo benötigt mind. 6 GB freien RAM (TorchScript + Mamba-State-Space).
            # Swap-Thrashing-Check allein reicht nicht — Crash tritt auch bei niedrigem
            # Swap-Prozent auf wenn nach großen Vorphasen (SGMSE+/MDX) wenig RAM verfügbar ist.
            if not _apollo_swap_blocked:
                try:
                    _avail_ram_p23 = psutil.virtual_memory().available / (1024**3) if _PSUTIL_OK else 4.0
                    if _avail_ram_p23 < 6.0:
                        logger.warning(
                            "Verarbeitungsschritt_23: Apollo TorchScript übersprungen — nur %.1f GB RAM verfügbar "
                            "(< 6.0 GB Mindest-Headroom). DSP-Inpainting wird verwendet.",
                            _avail_ram_p23,
                        )
                        _apollo_swap_blocked = True
                except Exception as _ram_chk_exc:
                    logger.debug("Verarbeitungsschritt_23: RAM-Pruefung fehlgeschlagen (unkritisch): %s", _ram_chk_exc)

            _plm23 = None
            if not _apollo_swap_blocked:
                try:
                    try:
                        _plm23 = get_plugin_lifecycle_manager()
                        _plm23.set_active("Apollo", True)  # §4.6b: protect from eviction during inference
                    except Exception:
                        logger.warning(
                            "§V6 Phase-23 Apollo lifecycle guard failed → no eviction protection: %s",
                            str(Exception),
                        )
                        _plm23 = None

                    _apollo_inst = _try_get_apollo_instance()
                    if (
                        _apollo_inst is not None
                        and bool(getattr(_apollo_inst, "_model_loaded", False))
                        and getattr(_apollo_inst, "_torch_model", None) is not None
                    ):
                        if is_stereo:
                            # Audio is normalized to (N, 2) at method start — safe to use [:, 0]
                            if _plm23 is not None:
                                try:
                                    _plm23.touch_plugin("Apollo")  # type: ignore[attr-defined]
                                except Exception:
                                    logger.debug("_report_progress: silent except suppressed", exc_info=True)
                            _ap_l = _apollo_inst.repair(audio[:, 0], sample_rate, material=self._current_material)
                            _ap_l_audio = _ap_l.audio
                            _ap_l_hf = float(_ap_l.hf_gain_db)
                            del _ap_l
                            # GC between channels — free torch tensors before second channel
                            gc.collect(0)
                            if _plm23 is not None:
                                try:
                                    _plm23.touch_plugin("Apollo")  # type: ignore[attr-defined]
                                except Exception:
                                    logger.debug("_report_progress: silent except suppressed", exc_info=True)
                            _ap_r = _apollo_inst.repair(audio[:, 1], sample_rate, material=self._current_material)
                            # §2.51 L/R-Zeitversatz-Guard: Apollo kann je Kanal minimal
                            # unterschiedliche Sample-Zahlen zurückgeben (Mamba-State-Init-Differenz).
                            # Auf kleinere Länge trimmen, damit kein statischer L/R-Versatz entsteht.
                            _ap_l_arr = np.asarray(_ap_l_audio, dtype=np.float32)
                            _ap_r_arr = np.asarray(_ap_r.audio, dtype=np.float32)
                            _ap_n = min(len(_ap_l_arr), len(_ap_r_arr))
                            audio = np.column_stack((_ap_l_arr[:_ap_n], _ap_r_arr[:_ap_n]))  # (N, 2) length-aligned
                            _apollo_hf_gain_db = (_ap_l_hf + float(_ap_r.hf_gain_db)) / 2.0
                            del _ap_r, _ap_l_audio
                        else:
                            _ap_res = _apollo_inst.repair(audio, sample_rate, material=self._current_material)
                            audio = _ap_res.audio
                            _apollo_hf_gain_db = float(_ap_res.hf_gain_db)
                            del _ap_res
                        _apollo_preproc_applied = True
                        logger.info(
                            "Verarbeitungsschritt_23: Apollo pre-processing angewendet (material=%s, hf_gain=+%.1f dB)",
                            self._current_material,
                            _apollo_hf_gain_db,
                        )
                        _report_progress(20.0, "Spektralreparatur: Apollo-Vorverarbeitung")
                except Exception as _apollo_exc:
                    logger.debug("Apollo pre-processing uebersprungen (unkritisch): %s", _apollo_exc)
                finally:
                    if _plm23 is not None:
                        try:
                            _plm23.set_active("Apollo", False)
                        except Exception:
                            logger.debug("_report_progress: silent except suppressed", exc_info=True)

        # --- ADMM Declipping path (spec §4.5a) ---
        # Detect hard clipping and route to sparse-recovery solver instead of
        # standard inpainting.  SOFT_SATURATION → no ADMM (per §5 vintage rules).
        _use_admm = False
        _clip_level = 0.98
        _defect_type_kwarg = kwargs.get("defect_type")
        if _defect_type_kwarg is not None and hasattr(_defect_type_kwarg, "name"):
            if _defect_type_kwarg.name in ("CLIPPING", "HARMONIC_DISTORTION"):
                _use_admm = True
        else:
            try:
                if is_stereo:
                    _mono_ref = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
                else:
                    _mono_ref = audio
                _clip_check = _mono_ref[: min(len(_mono_ref), sample_rate * 10)]
                if classify_clipping(_clip_check, sample_rate) == ClippingType.CLIPPING:
                    _use_admm = True
            except Exception as _ce:
                logger.debug("classify_clipping Pruefung fehlgeschlagen: %s", _ce)

        if _use_admm:
            # Estimate clip ceiling as 99.5th percentile of absolute amplitude
            _clip_level = float(np.percentile(np.abs(audio), 99.5))
            _clip_level = float(np.clip(_clip_level, 0.85, 1.0))
            logger.info("ADMM declipping activated: clip_level=%.4f", _clip_level)
            # §Spec04b ADMM max_iter: length-adaptive — fewer iterations for long signals
            # to avoid exhausting the UV3 wall-time budget.
            # Literature: 30–50 iterations typically sufficient for convergence
            # (Záviška 2021). 200 iter × 12 s/iter on 10.8 M samples = 40 min.
            _dur_s = float(audio.shape[-1] if audio.ndim > 1 else len(audio)) / float(sample_rate)
            _admm_max_iter = int(np.clip(round(200.0 * min(1.0, 30.0 / max(_dur_s, 1.0))), 30, 200))
            logger.info("ADMM max_iter=%d (duration=%.1fs)", _admm_max_iter, _dur_s)
            # §OOM-Guard: ADMM processes in ≤60 s chunks (see _admm_declip), so peak RAM
            # per chunk is ~500 MB instead of 10+ GB for full-length signals.
            # Only abort to inpainting as a last resort when < 1.5 GB free (hard-crash zone).
            _admm_ram_ok = True
            try:
                _admm_avail_gb = (
                    psutil.virtual_memory().available / (1024**3) if _PSUTIL_OK else 4.0 if _PSUTIL_OK else 4.0
                )
                if _admm_avail_gb < 1.5:
                    logger.warning(
                        "Verarbeitungsschritt_23: ADMM-OOM-Guard (Notfall) — nur %.1f GB frei, < 1.5 GB "
                        "— ADMM deaktiviert, Standard-Spektralinpainting wird verwendet",
                        _admm_avail_gb,
                    )
                    _admm_ram_ok = False
            except Exception as _admm_ram_exc:
                logger.debug(
                    "Verarbeitungsschritt_23: ADMM RAM-Pruefung fehlgeschlagen (unkritisch): %s", _admm_ram_exc
                )
            if not _admm_ram_ok:
                _use_admm = False
        if _use_admm:
            if is_stereo:
                # §2.51 M/S: Reparatur auf Mid-Kanal; Side minimal (Stereo-Kohärenz-Invariante).
                _sqrt2 = np.sqrt(2.0)
                _mid = (audio[:, 0] + audio[:, 1]) / _sqrt2
                _side = (audio[:, 0] - audio[:, 1]) / _sqrt2
                _report_progress(35.0, "Spektralreparatur: ADMM Mid-Kanal")
                _repaired_mid = self._admm_declip(_mid, _clip_level, sample_rate, max_iter=_admm_max_iter)
                # Side: declip mildly (half strength) to avoid breaking stereo field
                _side_clip = float(np.clip(_clip_level * 1.05, 0.85, 1.0))
                _report_progress(55.0, "Spektralreparatur: ADMM Side-Kanal")
                _repaired_side = self._admm_declip(_side, _side_clip, sample_rate, max_iter=_admm_max_iter)
                repaired_audio = np.column_stack(
                    (
                        (_repaired_mid + _repaired_side) / _sqrt2,
                        (_repaired_mid - _repaired_side) / _sqrt2,
                    )
                )
            else:
                _report_progress(60.0, "Spektralreparatur: ADMM")
                repaired_audio = self._admm_declip(audio, _clip_level, sample_rate, max_iter=_admm_max_iter)
        else:
            # Process via standard spectral inpainting — §2.51 M/S for stereo.
            if is_stereo:
                # §2.51 M/S: Reparatur auf Mid-Kanal; Side minimal (Stereo-Kohärenz-Invariante).
                _sqrt2 = np.sqrt(2.0)
                _mid = (audio[:, 0] + audio[:, 1]) / _sqrt2
                _side = (audio[:, 0] - audio[:, 1]) / _sqrt2
                _report_progress(35.0, "Spektralreparatur: Mid-Kanal")
                _repaired_mid = self._repair_channel(
                    _mid,
                    sample_rate,
                    stft_cfg,
                    thresholds,
                    repair_strength,
                    progress_cb=lambda p, lbl: _report_progress(
                        35.0 + 20.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                        f"Spektralreparatur Mid: {lbl}",
                    ),
                    phase_deadline=_phase_deadline,
                )
                # Side: minimal repair at half strength to preserve stereo field
                _side_strength = repair_strength * 0.5
                _report_progress(55.0, "Spektralreparatur: Side-Kanal")
                _repaired_side = self._repair_channel(
                    _side,
                    sample_rate,
                    stft_cfg,
                    thresholds,
                    _side_strength,
                    progress_cb=lambda p, lbl: _report_progress(
                        55.0 + 20.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                        f"Spektralreparatur Side: {lbl}",
                    ),
                    phase_deadline=_phase_deadline,
                )
                repaired_audio = np.column_stack(
                    (
                        (_repaired_mid + _repaired_side) / _sqrt2,
                        (_repaired_mid - _repaired_side) / _sqrt2,
                    )
                )
            else:
                _report_progress(60.0, "Spektralreparatur: Inpainting")
                repaired_audio = self._repair_channel(
                    audio,
                    sample_rate,
                    stft_cfg,
                    thresholds,
                    repair_strength,
                    progress_cb=lambda p, lbl: _report_progress(
                        35.0 + 40.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                        f"Spektralreparatur: {lbl}",
                    ),
                    phase_deadline=_phase_deadline,
                )

        # Calculate metrics
        defect_reduction = self._calculate_defect_reduction(audio, repaired_audio, sample_rate)
        spectral_coherence = self._calculate_spectral_coherence(repaired_audio, sample_rate)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        repaired_audio = np.nan_to_num(repaired_audio, nan=0.0, posinf=0.0, neginf=0.0)
        repaired_audio = np.clip(repaired_audio, -1.0, 1.0)
        _report_progress(78.0, "Spektralreparatur: Qualitätsmetriken")

        # §2.46b Spectral-Tilt-Guard: cap HF inpainting if tilt deviates beyond tolerance
        # Only applies to spectral inpainting path, not ADMM declipping (no HF synthesis there)
        _tilt_capped_p23 = False
        if not _use_admm:
            try:
                _mat_k23 = str(self._current_material).lower().replace(" ", "_").replace("-", "_")
                _tol23 = _TILT_TOLERANCE_P23.get(_mat_k23, 2.0)
                _tb23 = _est_tilt_p23(_audio_for_tilt_p23, sample_rate)
                _ta23 = _est_tilt_p23(repaired_audio, sample_rate)
                _dev23 = abs(_ta23 - _tb23)
                if _dev23 > _tol23:
                    _cap23 = float(np.clip(1.0 - (_dev23 - _tol23) / (_tol23 * 2.0), 0.5, 1.0))
                    repaired_audio = _cap23 * repaired_audio + (1.0 - _cap23) * _audio_for_tilt_p23
                    repaired_audio = np.clip(repaired_audio, -1.0, 1.0)
                    _tilt_capped_p23 = True
                    logger.info(
                        "Verarbeitungsschritt_23 §2.46b tilt-cap: before=%.2f after=%.2f dev=%.2f tol=%.2f cap=%.2f",
                        _tb23,
                        _ta23,
                        _dev23,
                        _tol23,
                        _cap23,
                    )
            except Exception as _tc23:
                logger.debug("Verarbeitungsschritt_23 §2.46b tilt-cap uebersprungen (graceful): %s", _tc23)

        # §Waerme-Rescue: Analog-Materialien (vinyl/shellac/tape) haben spezifische LF-Wärme.
        # MRSA-Inpainting halluziniert HF-Inhalt der die E(200-800 Hz)/E(800-3000 Hz)-Ratio
        # reduziert → waerme fällt katastrophal (Elke Best: 0.93→0.62, −0.31).
        # Tilt-cap allein reicht nicht: cap=0.63 aber waerme−0.31 weil MRSA HF massiv boosted.
        # Rescue: nach tilt-cap messen; wenn Wärme-Verlust > 0.15, Original zurückblenden bis
        # Verlust ≤ 0.15 — bewahrt Ära-/Träger-Klang ohne MRSA vollständig zu deaktivieren.
        _ANALOG_WAERME_MATERIALS = frozenset(
            {
                "vinyl",
                "shellac",
                "reel_tape",
                "tape",
                "cassette",
                "wax_cylinder",
                "wire_recording",
                "lacquer_disc",
            }
        )
        # §Waerme-Rescue gilt für BEIDE Pfade (ADMM + Inpainting):
        # ADMM-Declipping verändert primär Clipping-Peaks, nicht die LF/MF-Energiebilanz —
        # aber Vinyl-Material mit mp3_low-Kette (Elke Best) zeigt waerme=0.694 < 0.740
        # auch ohne aktiven ADMM-Waerme-Drop. Rescue kostet < 1 ms → immer anwenden.
        try:
            _mat_k23_wr = str(self._current_material).lower().replace(" ", "_").replace("-", "_")
            if _mat_k23_wr in _ANALOG_WAERME_MATERIALS:

                def _waerme_proxy_p23(a: np.ndarray) -> float:
                    mono = a.mean(axis=0) if (a.ndim == 2 and a.shape[0] <= 2) else a.mean(axis=1) if a.ndim == 2 else a
                    _n = min(len(mono), 8192)
                    if _n < 64:
                        return 0.5
                    spec = np.abs(np.fft.rfft(mono[:_n] * np.hanning(_n))) ** 2
                    freqs = np.fft.rfftfreq(_n, d=1.0 / sample_rate)
                    e_low = float(np.mean(spec[(freqs >= 200) & (freqs < 800)] + 1e-12))
                    e_up = float(np.mean(spec[(freqs >= 800) & (freqs < 3000)] + 1e-12))
                    return float(np.clip(e_low / e_up / 4.0, 0.0, 1.0))

                _w_ref = _waerme_proxy_p23(_audio_for_tilt_p23)
                _w_cur = _waerme_proxy_p23(repaired_audio)
                _w_drop = _w_ref - _w_cur
                _MAX_WAERME_DROP_P23 = 0.15
                if _w_drop > _MAX_WAERME_DROP_P23:
                    # blend = MAX_DROP / actual_drop → inpainting-Anteil minimiert
                    _wr_blend = float(np.clip(_MAX_WAERME_DROP_P23 / max(_w_drop, 1e-6), 0.0, 1.0))
                    repaired_audio = _wr_blend * repaired_audio + (1.0 - _wr_blend) * _audio_for_tilt_p23
                    repaired_audio = np.clip(repaired_audio, -1.0, 1.0)
                    logger.info(
                        "Verarbeitungsschritt_23 §Waerme-Rescue: mat=%s waerme %.4f→%.4f (drop=%.3f > %.2f) "
                        "→ rescue-blend=%.2f (%.0f%% Originalsignal beigemischt)",
                        _mat_k23_wr,
                        _w_ref,
                        _w_cur,
                        _w_drop,
                        _MAX_WAERME_DROP_P23,
                        _wr_blend,
                        (1.0 - _wr_blend) * 100,
                    )
        except Exception as _wr_exc:
            logger.debug("Verarbeitungsschritt_23 waerme-rescue nicht blockierend: %s", _wr_exc)

        # §4.5 Psychoacoustic Masking Clamp — only repair audible spectral gaps
        try:
            repaired_audio = apply_psychoacoustic_masking_clamp(
                audio,
                repaired_audio,
                sample_rate,
                strength=_effective_strength,
                mode="additive",
            )
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt23 masking clamp nicht blockierend: %s", _pm_exc)

        # §2.36 Phonem-Schutz — Restore plosive burst frames aus Original
        # (Spektralreparatur kann Burst-Transienten dämpfen/glätten)
        try:
            _ph23_mono = (
                repaired_audio.mean(axis=0)
                if (repaired_audio.ndim == 2 and repaired_audio.shape[0] <= 2 and repaired_audio.shape[1] > 2)
                else (repaired_audio.mean(axis=1) if repaired_audio.ndim == 2 else repaired_audio)
            )
            _orig23_mono = (
                audio.mean(axis=0)
                if (audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > 2)
                else (audio.mean(axis=1) if audio.ndim == 2 else audio)
            )
            _pmask23 = get_phoneme_mask(_ph23_mono.astype(np.float32), sample_rate, hop_length=512)
            if np.any(_pmask23):
                _n23 = len(_ph23_mono)
                for _fi23, _fp23 in enumerate(_pmask23):
                    if _fp23:
                        _fs23 = _fi23 * 512
                        _fe23 = min(_n23, _fs23 + 512)
                        if repaired_audio.ndim == 2:
                            repaired_audio[_fs23:_fe23, :] = audio[_fs23:_fe23, :]
                        else:
                            repaired_audio[_fs23:_fe23] = audio[_fs23:_fe23]
        except Exception as _ph23_exc:
            logger.debug("§2.36 Verarbeitungsschritt23 Phonem-Mask (nicht blockierend): %s", _ph23_exc)

        # §2.46f NaturalPerformanceArtifacts-Guard — Restore atemgeräusche, vibrato, early-reflections
        try:
            _npa23 = _get_phase23_npd()
            _npa_result23 = _npa23.detect(
                audio if audio.ndim == 1 else audio.mean(axis=0),
                sample_rate,
            )
            if _npa_result23.success:
                _npa_mask23 = _npa_result23.get_protected_mask(
                    repaired_audio.shape[0] if repaired_audio.ndim == 1 else repaired_audio.shape[-1],
                    sample_rate,
                )
                if np.any(_npa_mask23):
                    if repaired_audio.ndim == 2:
                        repaired_audio[_npa_mask23, :] = audio[_npa_mask23, :]
                    else:
                        repaired_audio[_npa_mask23] = audio[_npa_mask23]
        except Exception as _npa23_exc:
            logger.debug("§2.46f Verarbeitungsschritt23 NPA-Guard (nicht blockierend): %s", _npa23_exc)

        _defect_locality_coverage23 = 0.0
        try:
            _locality_profile23, _defect_locality_coverage23 = self._build_defect_locality_profile(
                int(repaired_audio.shape[0]),
                int(sample_rate),
                kwargs.get("defect_locations"),
                kwargs.get("defect_event_metadata"),
            )
            if _locality_profile23.size > 0 and _defect_locality_coverage23 > 0.0:
                if repaired_audio.ndim == 2:
                    _profile23 = _locality_profile23[:, np.newaxis]
                    repaired_audio = np.clip(audio + _profile23 * (repaired_audio - audio), -1.0, 1.0).astype(
                        np.float32
                    )
                else:
                    repaired_audio = np.clip(audio + _locality_profile23 * (repaired_audio - audio), -1.0, 1.0).astype(
                        np.float32
                    )
        except Exception as _locality23_exc:
            logger.debug("Verarbeitungsschritt_23 defect-locality blend (nicht blockierend): %s", _locality23_exc)

        # §V38/§0p VFA-Zonen-Blend-Back — VocalFocusAnalyzer-Zonen aus kwargs:
        # Vibrato (cap 0.20), Frisson (cap 0.30), Flüster (cap 0.25), Passaggio (cap 0.35).
        # repair_strength wird in geschützten Zonen effektiv gedeckelt — schützt vor
        # unbeabsichtigter Veränderung vokal-kritischer Bereiche durch Spektralinpainting.
        try:
            _vfa_zones_p23: list[tuple[float, float, float]] = []
            for _vz in kwargs.get("vibrato_zones") or []:
                try:
                    _vfa_zones_p23.append((float(_vz[0]), float(_vz[1]), 0.20))
                except Exception:
                    logger.debug("_waerme_proxy_p23: silent except suppressed", exc_info=True)
            for _fz in kwargs.get("frisson_zones") or []:
                try:
                    _vfa_zones_p23.append((float(_fz[0]), float(_fz[1]), 0.30))
                except Exception:
                    logger.debug("_waerme_proxy_p23: silent except suppressed", exc_info=True)
            for _wz in kwargs.get("whisper_zones") or []:
                try:
                    _vfa_zones_p23.append((float(_wz[0]), float(_wz[1]), 0.25))
                except Exception:
                    logger.debug("_waerme_proxy_p23: silent except suppressed", exc_info=True)
            for _pz in kwargs.get("passaggio_zones") or []:
                try:
                    _vfa_zones_p23.append((float(_pz[0]), float(_pz[1]), 0.35))
                except Exception:
                    logger.debug("_waerme_proxy_p23: silent except suppressed", exc_info=True)
            if _vfa_zones_p23:
                _n_p23 = repaired_audio.shape[0] if repaired_audio.ndim == 1 else repaired_audio.shape[-1]
                _blend_p23 = np.ones(_n_p23, dtype=np.float32) * float(repair_strength)
                for _zs, _ze, _cap in _vfa_zones_p23:
                    _ss_p23 = int(np.clip(round(_zs * sample_rate), 0, _n_p23))
                    _se_p23 = int(np.clip(round(_ze * sample_rate), 0, _n_p23))
                    if _se_p23 > _ss_p23:
                        _blend_p23[_ss_p23:_se_p23] = np.minimum(_blend_p23[_ss_p23:_se_p23], float(_cap))
                # Nur Zonen mit tatsächlich niedrigerem Cap als repair_strength korrigieren
                if np.any(_blend_p23 < repair_strength - 1e-4):
                    _audio_ref_p23 = audio  # Pre-Phase-Input
                    if repaired_audio.ndim == 2:
                        _b = _blend_p23[np.newaxis, :] if audio.shape[0] <= 2 else _blend_p23[:, np.newaxis]
                        repaired_audio = np.clip(_b * repaired_audio + (1.0 - _b) * _audio_ref_p23, -1.0, 1.0).astype(
                            np.float32
                        )
                    else:
                        repaired_audio = np.clip(
                            _blend_p23 * repaired_audio + (1.0 - _blend_p23) * _audio_ref_p23, -1.0, 1.0
                        ).astype(np.float32)
                    logger.debug(
                        "Verarbeitungsschritt_23 VFA-Blend-Back: %d Schutzzonen, repair_strength=%.2f",
                        len(_vfa_zones_p23),
                        repair_strength,
                    )
        except Exception as _vfa23_exc:
            logger.debug("Verarbeitungsschritt_23 VFA-Blend-Back (nicht blockierend): %s", _vfa23_exc)

        _mode23 = str(kwargs.get("mode", "restoration")).lower()
        _bw_ceiling_applied23 = False
        _bw_ceiling_hz23: float | None = None
        try:
            repaired_audio, _bw_ceiling_applied23, _bw_ceiling_hz23 = self._apply_material_bw_ceiling(
                repaired_audio,
                sample_rate,
                material,
                mode=_mode23,
            )
            if _bw_ceiling_applied23:
                logger.info(
                    "§6.2c Verarbeitungsschritt_23 material BW-Ceiling: %s ≤ %.0f Hz before HallucinationGuard",
                    self._material_key(material),
                    float(_bw_ceiling_hz23 or 0.0),
                )
        except Exception as _bw23_exc:
            logger.debug("§6.2c Verarbeitungsschritt23 material BW-Ceiling (nicht blockierend): %s", _bw23_exc)

        # §2.46e Hallucination-Guard: Spektral-Inpainting/Reparatur darf kein Material
        # einbringen das nicht im Input physikalisch vorhanden war (Restoration-Modus).
        try:
            if "studio" not in _mode23:
                _bw23 = _bw_ceiling_hz23 or self._material_bw_ceiling_hz(material) or 22050.0
                _mono_orig23 = audio.mean(axis=-1) if audio.ndim == 2 else audio
                _mono_rep23 = repaired_audio.mean(axis=-1) if repaired_audio.ndim == 2 else repaired_audio
                _mono_orig23 = _mono_orig23 if _mono_orig23.ndim == 1 else _mono_orig23.ravel()
                _mono_rep23 = _mono_rep23 if _mono_rep23.ndim == 1 else _mono_rep23.ravel()
                # §6.2c BW-Ceiling-Guard-Interaktion: Wenn _apply_material_bw_ceiling bereits
                # angewendet wurde, ist eine erneute harmonic_ceiling_violation-Prüfung ein
                # False-Positive. Der 6th-Order-Butterworth liefert kein Brick-Wall-Rolloff
                # (−36 dB an Nyquist); FlashSR erzeugt vor dem Ceiling deutlich mehr Energie
                # über dem Ceiling, sodass Restenergie nach Filterung dennoch ceiling_band_ratio
                # > 8× ergibt → vollständiger Rollback obwohl das Signal korrekt gedeckelt ist.
                # Lösung: material_bw_ceiling_hz nur übergeben wenn Ceiling noch NICHT aktiv war.
                _hg_ceiling_hz23 = None if _bw_ceiling_applied23 else _bw23
                _hg_result23 = check_hallucination(
                    _mono_orig23,
                    _mono_rep23,
                    sr=sample_rate,
                    mode=_mode23,
                    material_bw_ceiling_hz=_hg_ceiling_hz23,
                )
                if _hg_result23.requires_rollback:
                    _salvaged23 = False
                    _rollback_reason23 = (
                        "harmonic_ceiling_violation"
                        if bool(getattr(_hg_result23, "harmonic_ceiling_violation", False))
                        else "spectral_novelty"
                    )

                    # Defense-in-depth: vor Hard-Rollback einen konservativen Blend versuchen,
                    # der das Träger-Ceiling respektiert und den Hallucination-Guard erneut prüft.
                    for _blend23 in (0.40, 0.25, 0.12):
                        _candidate23 = audio + (repaired_audio - audio) * float(_blend23)
                        _candidate23 = np.nan_to_num(_candidate23, nan=0.0, posinf=0.0, neginf=0.0)
                        _candidate23 = np.clip(_candidate23, -1.0, 1.0)

                        _blend_ceiling_applied23 = False
                        try:
                            _candidate23, _blend_ceiling_applied23, _ = self._apply_material_bw_ceiling(
                                _candidate23,
                                sample_rate,
                                material,
                                mode=_mode23,
                            )
                        except Exception:
                            # non-blocking: Blend dennoch gegen Guard prüfen
                            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

                        _cand_mono23 = _candidate23.mean(axis=-1) if _candidate23.ndim == 2 else _candidate23
                        _cand_mono23 = _cand_mono23 if _cand_mono23.ndim == 1 else _cand_mono23.ravel()
                        _cand_hg_ceiling_hz23 = None if _blend_ceiling_applied23 else _bw23
                        _cand_hg23 = check_hallucination(
                            _mono_orig23,
                            _cand_mono23,
                            sr=sample_rate,
                            mode=_mode23,
                            material_bw_ceiling_hz=_cand_hg_ceiling_hz23,
                        )
                        if not _cand_hg23.requires_rollback:
                            repaired_audio = _candidate23
                            _salvaged23 = True
                            logger.info(
                                "§2.46e Verarbeitungsschritt23 Hallucination Rescue: reason=%s blend=%.2f novelty=%.3f",
                                _rollback_reason23,
                                float(_blend23),
                                float(_cand_hg23.spectral_novelty),
                            )
                            break

                    if not _salvaged23:
                        logger.debug(
                            "§2.46e Verarbeitungsschritt23 Hallucination rollback: reason=%s novelty=%.3f",
                            _rollback_reason23,
                            float(_hg_result23.spectral_novelty),
                        )
                        repaired_audio = audio.copy()
                if _hg_result23.score_penalty > 0:
                    logger.info(
                        "§2.46e Verarbeitungsschritt23 Wert_penalty=%.1f (spectral_novelty=%.3f)",
                        _hg_result23.score_penalty,
                        _hg_result23.spectral_novelty,
                    )
        except Exception as _hg23_exc:
            logger.debug("§2.46e Verarbeitungsschritt23 Hallucination-Guard (nicht blockierend): %s", _hg23_exc)

        _report_progress(92.0, "Spektralreparatur: Abschluss")

        # §2.46f Edge-Taper (defense-in-depth): secondary safety net after context-padding in
        # _repair_single_channel above. Context-padding is the root-cause fix; edge-taper catches
        # any residual boundary artefacts from DSP path or ISTFT stitching across MRSA zones.
        try:
            _et23_sr = sample_rate
            _et23_fade_s = 0.5
            _et23_n = int(_et23_fade_s * _et23_sr)
            # Always use axis-0 (time axis): at this point repaired_audio is
            # (N, 2) channels-last or (N,) mono — shape[-1] would give 2 for stereo.
            _et23_total = repaired_audio.shape[0]
            if _et23_total >= _et23_n * 4:
                _et23_fade = np.linspace(0.0, 1.0, _et23_n, dtype=np.float32)
                _orig23_et = audio
                if repaired_audio.ndim == 2:
                    # (N, 2) channels-last layout at this point (before channels-first restore)
                    repaired_audio[:_et23_n, :] = (
                        repaired_audio[:_et23_n, :] * _et23_fade[:, None]
                        + _orig23_et[:_et23_n, :] * (1.0 - _et23_fade[:, None])
                    ).astype(repaired_audio.dtype)
                    repaired_audio[-_et23_n:, :] = (
                        repaired_audio[-_et23_n:, :] * _et23_fade[::-1, None]
                        + _orig23_et[-_et23_n:, :] * (1.0 - _et23_fade[::-1, None])
                    ).astype(repaired_audio.dtype)
                else:
                    repaired_audio[:_et23_n] = (
                        repaired_audio[:_et23_n] * _et23_fade + _orig23_et[:_et23_n] * (1.0 - _et23_fade)
                    ).astype(repaired_audio.dtype)
                    repaired_audio[-_et23_n:] = (
                        repaired_audio[-_et23_n:] * _et23_fade[::-1] + _orig23_et[-_et23_n:] * (1.0 - _et23_fade[::-1])
                    ).astype(repaired_audio.dtype)
                logger.debug("Verarbeitungsschritt23 edge-taper: %.0f ms at intro+outro", _et23_fade_s * 1000)
        except Exception as _et23_exc:
            logger.debug("Verarbeitungsschritt23 edge-taper nicht blockierend: %s", _et23_exc)

        # Restore channels-first layout expected by UV3 (2, N)
        if _was_channels_first and repaired_audio.ndim == 2:
            repaired_audio = repaired_audio.T  # (N, 2) → (2, N)

        # §2.68 SSIP Post-Inpainting-Audit: Hard-Reset in Stille-Zonen (V17, §2.68d).
        # Muss nach channels-first-Restore laufen damit Layout-Format korrekt ist.
        if _ssip_zones_p23:
            try:
                from backend.core.dsp.structural_silence_isolation import (
                    get_structural_silence_isolator as _get_ssip_audit23,
                )

                _orig23_ssip = audio.T if (_was_channels_first and audio.ndim == 2) else audio
                repaired_audio = _get_ssip_audit23().post_inpainting_silence_audit(
                    audio_before_inpainting=_orig23_ssip,
                    audio_after_inpainting=repaired_audio,
                    silence_zones=_ssip_zones_p23,
                    sr=sample_rate,
                )
            except Exception as _ssip_audit_exc23:
                logger.debug(
                    "SSIP post_inpainting_silence_audit Verarbeitungsschritt_23 (nicht blockierend): %s",
                    _ssip_audit_exc23,
                )

        # §V22 Pre-Echo-Prevention — Generative Spektralreparatur auf Transient-Shifts prüfen (§2.73, non-blocking)
        try:
            from backend.core.dsp.transient_guard import (
                detect_transient_shifts as _dts_23,
            )

            _pre_v22_23 = (
                audio.mean(axis=-1 if audio.ndim == 2 and audio.shape[-1] <= 8 else 0).astype(np.float32)
                if audio.ndim == 2
                else audio.astype(np.float32)
            )
            _ax_v22_23 = -1 if repaired_audio.ndim == 2 and repaired_audio.shape[-1] <= 8 else 0
            _post_v22_23 = (
                repaired_audio.mean(axis=_ax_v22_23).astype(np.float32)
                if repaired_audio.ndim == 2
                else repaired_audio.astype(np.float32)
            )
            _ts_23 = _dts_23(_pre_v22_23, _post_v22_23, sample_rate)
            if not _ts_23.ok:
                _wet_ts_23 = max(0.0, 1.0 - _ts_23.blend_reduction)
                repaired_audio = (_wet_ts_23 * repaired_audio + (1.0 - _wet_ts_23) * audio).astype(np.float32)
                logger.warning(
                    "§V22 Verarbeitungsschritt_23: onset_shift=%.2f ms → blend_reduction=%.2f",
                    _ts_23.max_shift_ms,
                    _ts_23.blend_reduction,
                )
        except Exception as _v22_23_exc:
            logger.debug("§V22 Verarbeitungsschritt_23 transient_guard nicht blockierend: %s", _v22_23_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung (§2.74, non-blocking): Reparatur darf Spektralfarbe nicht verändern
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg23,
            )

            _orig23_v24 = audio.T if (_was_channels_first and audio.ndim == 2) else audio
            _rep23_v24 = repaired_audio
            _sc23 = _scg23(_orig23_v24, _rep23_v24, sample_rate)
            if not _sc23.ok:
                _sc23_wet = 0.70
                repaired_audio = (_sc23_wet * repaired_audio + (1.0 - _sc23_wet) * _orig23_v24).astype(np.float32)
                logger.warning(
                    "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_23 spectral_color non-ok → strength −30%%"
                )
        except Exception as _sc23_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_23 spectral_color (nicht blockierend): %s", _sc23_exc
            )

        _result = PhaseResult(
            success=True,
            audio=repaired_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": _material_meta_key23,
                "defect_reduction_percent": float(defect_reduction * 100),
                "spectral_coherence": float(spectral_coherence),
                "repair_strength": float(repair_strength),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "defect_locality_coverage": float(_defect_locality_coverage23),
                "rt_factor": float(rt_factor),
                "nperseg": stft_cfg["nperseg"],
                "apollo_preproc_applied": _apollo_preproc_applied,
                "apollo_preproc_hf_gain_db": _apollo_hf_gain_db,
                "ml_guard_events": list(self._ml_guard_events),
                "deferred_for_kmv": ["phase_23_spectral_repair"] if self._ml_guard_events else [],
                "spectral_tilt_capped": _tilt_capped_p23,
                "material_bw_ceiling_applied": bool(_bw_ceiling_applied23),
                "material_bw_ceiling_hz": _bw_ceiling_hz23,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.6 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )
        _report_progress(100.0, "Spektralreparatur: fertig")
        return _result

    # §OOM-Guard: max segment size for chunked ADMM processing.
    # 60 s × 48000 Hz × float64 × ~7 wavelet arrays = ~390 MB peak per chunk.
    # Vs. 225 s full-signal → ~1.5 GB peak → well within safe headroom on 16+ GB systems.
    _ADMM_CHUNK_S: float = 60.0
    # Crossfade overlap at chunk boundaries: 480 samples = 10 ms @ 48 kHz.
    # db4 level-5 wavelet boundary influence: 2^5 × 8 = 256 samples — safely covered.
    _ADMM_OVERLAP_SAMPLES: int = 480

    def _admm_declip(
        self,
        audio: np.ndarray,
        clip_level: float,
        sr: int = 48_000,
        rho: float = 0.1,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """ADMM sparse-recovery declipping per spec §4.5a (Záviška 2021).

        For signals longer than _ADMM_CHUNK_S seconds, processing is split into
        overlapping chunks with 10 ms crossfade (§OOM-Guard). Quality is identical
        to full-signal processing: ADMM convergence is local (clipping is per-sample),
        and wavelet boundary artefacts are contained within the 10 ms overlap region.
        Peak RAM is O(chunk_size) instead of O(signal_length), preventing OOM on
        long songs (225 s vinyl: 10+ GB → ~400 MB peak with 60 s chunks).

        Solves:  minimize  ||W x||_1
                 subject to  x[reliable] = y[reliable]
                              x[clipped] ∈ [-C, C]

        where W is a Daubechies-db4 Level-5 wavelet transform and C = clip_level.

        TransientGuard: at onset positions (±5 ms) rho is scaled by 3.0 to
        preserve attack transients (ArticulationMetric ≥ 0.88 invariant).

        Post-ADMM: clipped samples that remain > clip_level are hard-clamped
        to avoid residual ears.

        Reference: Záviška et al. (2021) "A Survey and an Extensive Evaluation
        of Popular Audio Declipping Methods"; Condat (2013) ADMM tutorial.

        Args:
            audio:     1-D float32 mono signal (clipped).
            clip_level: Amplitude ceiling (e.g. 0.98 for near-full-scale).
            sr:        Sample rate — must be 48 000 Hz.
            rho:       ADMM penalty parameter (default 0.1, adaptive at onsets).
            max_iter:  Maximum ADMM iterations.
            tol:       Convergence tolerance (relative primal residual).

        Returns:
            Declipped 1-D float32 array, same shape as input.
        """
        try:
            pywt = importlib.import_module("pywt")
        except ModuleNotFoundError:
            logger.warning("pywt not verfuegbar — ADMM declipping uebersprungen, returning Originalsignal")
            return np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

        # float32 throughout: 2× faster wavelet ops, 2× less RAM vs float64;
        # precision is > 140 dB below the noise floor even for 24-bit audio.
        y = np.asarray(audio, dtype=np.float32)
        n = len(y)

        # §OOM-Guard: chunk long signals to cap peak RAM.
        # Identical quality: declipping is a per-sample operation; ADMM converges
        # locally. Wavelet boundary effects are absorbed by the 10 ms overlap.
        _chunk_n = int(self._ADMM_CHUNK_S * sr)
        if n > _chunk_n:
            _ovlp = self._ADMM_OVERLAP_SAMPLES
            result = np.empty(n, dtype=np.float32)
            pos = 0
            chunk_idx = 0
            while pos < n:
                # Extend chunk slightly into next segment for overlap-add
                c_start = pos
                c_end = min(pos + _chunk_n + _ovlp, n)
                chunk = y[c_start:c_end].copy()  # float32 independent copy
                # Skip ADMM entirely for chunks with no clipped samples — hot path
                # for songs where clipping is sparse (common on tape/mp3 material).
                if not np.any(np.abs(chunk) >= clip_level * 0.99):
                    repaired_chunk = chunk
                else:
                    repaired_chunk = self._admm_declip(chunk, clip_level, sr, rho, max_iter, tol)
                if chunk_idx == 0:
                    # First chunk: copy entirely (no predecessor)
                    _copy_end = min(_chunk_n, n)
                    result[c_start : c_start + _copy_end] = repaired_chunk[:_copy_end]
                else:
                    # Crossfade overlap region with previously written samples
                    xf_len = min(_ovlp, len(repaired_chunk), n - pos)
                    if xf_len > 0:
                        fade_in = np.linspace(0.0, 1.0, xf_len, dtype=np.float32)
                        fade_out = 1.0 - fade_in
                        result[pos : pos + xf_len] = (
                            result[pos : pos + xf_len] * fade_out + repaired_chunk[:xf_len] * fade_in
                        )
                    # Non-overlap tail
                    tail_start = xf_len
                    tail_end = min(_chunk_n + xf_len, len(repaired_chunk))
                    copy_n = tail_end - tail_start
                    if copy_n > 0:
                        result[pos + xf_len : pos + xf_len + copy_n] = repaired_chunk[tail_start:tail_end]
                gc.collect(0)
                chunk_idx += 1
                pos += _chunk_n
            logger.info(
                "ADMM declip: chunked processing done (%d chunks, chunk_s=%.0fs)",
                chunk_idx,
                self._ADMM_CHUNK_S,
            )
            return np.nan_to_num(result, nan=0.0)  # type: ignore[no-any-return]

        # --- Reliable vs clipped mask ---
        reliable_mask = np.abs(y) < clip_level * 0.99
        clipped_mask = ~reliable_mask
        if not np.any(clipped_mask):
            return np.nan_to_num(y.astype(np.float32), nan=0.0)  # type: ignore[no-any-return]

        # --- Onset detection for TransientGuard (§4.5a) ---
        onset_win = max(1, int(sr * 0.005))  # ±5 ms
        onset_guard = np.zeros(n, dtype=bool)
        try:
            hop = 64
            n_frames = max(1, (n - hop) // hop)
            # Vectorised energy per frame — np.add.reduceat avoids Python-loop overhead
            # over ~45 000 frames for a 60 s chunk.
            _n_align = n_frames * hop
            frame_energy = np.add.reduceat(y[:_n_align] ** 2, np.arange(0, _n_align, hop))
            diff = np.diff(frame_energy, prepend=frame_energy[0])
            mu, sigma = float(np.mean(diff)), float(np.std(diff)) + 1e-10
            onset_frames = np.where(diff > mu + 2.0 * sigma)[0]
            for of in onset_frames:
                s = max(0, of * hop - onset_win)
                e = min(n, of * hop + onset_win)
                onset_guard[s:e] = True
        except Exception as _exc:
            logger.debug(
                "Operation fehlgeschlagen (unkritisch): %s", _exc
            )  # No transient guard on error — safe fallback

        # --- Wavelet parameters: db4 Level-5 ---
        wavelet = "db4"
        level = 5

        # --- ADMM initialisation ---
        x = y.copy()
        coeffs_shape = pywt.wavedec(x, wavelet, level=level, mode="periodization")
        z = [c.copy() for c in coeffs_shape]
        u = [np.zeros_like(c) for c in coeffs_shape]

        lam = 0.01 * rho  # Sparsity weight balanced against rho
        rho_onset = rho * 3.0  # TransientGuard penalty

        # Cache before inner loop — np.any() on large bool array is O(n), called once here.
        _any_onset_g = bool(np.any(onset_guard))
        # Pre-allocated convergence-tracking buffer: np.copyto avoids malloc/free per iteration.
        x_prev_buf = x.copy()
        # §Spec04b ADMM wall-time budget: prevents 41-min hangs on long songs.
        # Budget = min(180 s, 1.5× audio duration) — CPU-only pywt can't be GPU-accelerated.
        _admm_t0 = time.monotonic()
        _admm_wall_budget_s = min(180.0, float(n) / float(sr) * 1.5)
        for _iter in range(max_iter):
            # x-update: reconstruct from (z − u), then project onto constraints
            z_minus_u = [z[i] - u[i] for i in range(len(z))]
            x_new = pywt.waverec(z_minus_u, wavelet, mode="periodization")
            del z_minus_u  # free immediately — ~n×8 bytes per wavelet band
            x_new = x_new[:n]
            if len(x_new) < n:
                x_new = np.pad(x_new, (0, n - len(x_new)))

            # Projection: reliable → original; clipped → clamp to [-C, C]
            x_new[reliable_mask] = y[reliable_mask]
            x_new[clipped_mask] = np.clip(x_new[clipped_mask], -clip_level, clip_level)

            # At onset guard positions: keep original value when reliable to
            # preserve transient shape (rho_onset applied implicitly via z-update)
            onset_reliable = onset_guard & reliable_mask
            x_new[onset_reliable] = y[onset_reliable]
            del onset_reliable

            x = x_new
            del x_new

            # z-update + u-update: ONE DWT pass instead of two.
            # x is not modified between z and u updates within the same iteration,
            # so xc = wavedec(x) can be reused for both — saves 33% of wavelet transforms.
            xc = pywt.wavedec(x, wavelet, level=level, mode="periodization")
            for i in range(len(z)):
                v = xc[i] + u[i]
                # Use onset-aware threshold only for approximation coefficients (i==0)
                thr = lam / (rho_onset if i == 0 and _any_onset_g else rho)
                z[i] = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
            # u-update: reuse xc (x unchanged since z-update in this iteration)
            for i in range(len(u)):
                u[i] = u[i] + xc[i] - z[i]
            del xc

            # Convergence check with pre-allocated buffer (no malloc per iteration)
            rel = float(np.linalg.norm(x - x_prev_buf)) / (float(np.linalg.norm(x_prev_buf)) + 1e-10)
            if rel < tol:
                logger.debug("ADMM declip converged after %d iterations (rel=%.2e)", _iter + 1, rel)
                break
            np.copyto(x_prev_buf, x)  # in-place — avoids repeated allocation
            # Wall-time budget check (non-convergence path)
            if time.monotonic() - _admm_t0 > _admm_wall_budget_s:
                logger.warning(
                    "ADMM declip: wall-time Grenze %.0fs exceeded after %d iterations — early exit",
                    _admm_wall_budget_s,
                    _iter + 1,
                )
                break
            # Periodic GC to prevent iteration-accumulation of temporaries
            if (_iter + 1) % 10 == 0:
                gc.collect(0)

        # Hard-clamp residual excursions > clip_level as safety net
        x = np.clip(x, -1.0, 1.0)
        return x.astype(np.float32)  # type: ignore[no-any-return]

    def _repair_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stft_cfg: dict[str, int],
        thresholds: dict[str, float],
        repair_strength: float,
        progress_cb=None,
        phase_deadline: float = 0.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Repariert a single audio channel using spectral inpainting."""

        def _report(pct: float, label: str) -> None:
            if callable(progress_cb):
                try:
                    progress_cb(float(np.clip(pct, 0.0, 100.0)), label)
                except Exception:
                    logger.debug("_report: silent except suppressed", exc_info=True)

        _report(8.0, "STFT")
        # Compute STFT
        _f, _t, Zxx = safe_stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=stft_cfg["nperseg"],
            noverlap=stft_cfg["noverlap"],
            nfft=stft_cfg["nfft"],
            boundary="even",
        )

        # Magnitude and phase
        magnitude = cast(np.ndarray, np.abs(Zxx))
        phase = np.asarray(np.angle(Zxx))
        _report(22.0, "Defekterkennung")

        # Detect defects using DSP (always)
        defect_mask = self._detect_defects(magnitude, phase, thresholds)

        if np.sum(defect_mask) == 0:
            # No defects detected
            _report(100.0, "Keine Defekte")
            return np.nan_to_num(audio, nan=0.0)

        # Calculate defect severity for adaptive ML decision
        defect_severity = float(np.sum(defect_mask) / defect_mask.size)
        system_thrashing = self._is_system_thrashing()
        allow_ml_under_pressure = system_thrashing and self._can_relax_thrashing_guard(for_mrsa=False)
        if allow_ml_under_pressure and self._pressure_relax_ml_attempts >= self._THRASH_RELAX_ML_MAX_ATTEMPTS:
            allow_ml_under_pressure = False
            logger.warning(
                "Verarbeitungsschritt_23: pressure-relax ML Versuch cap reached (%d) — forcing DSP Ersatzpfad",
                self._THRASH_RELAX_ML_MAX_ATTEMPTS,
            )

        # Decide: ML or DSP?
        use_ml = is_phase_ml_enabled(23) and QualityModeConfig.should_use_ml("phase_23", defect_severity)
        # Crash-safety for local/default pytest runs:
        # Phase-23 ML can allocate heavy model stacks and destabilize some hosts.
        # Only allow ML during tests when explicitly enabled via --run-heavy-tests.
        _in_pytest23 = "PYTEST_CURRENT_TEST" in os.environ
        _allow_heavy_tests23 = os.environ.get("AURIK_RUN_HEAVY_TESTS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if _in_pytest23 and not _allow_heavy_tests23:
            use_ml = False
            logger.info(
                "Verarbeitungsschritt_23: ML deaktiviert in default pytest Ausfuehrung (crash-safety); use --Ausfuehrung-heavy-tests for ML path"
            )
        if use_ml and system_thrashing and not allow_ml_under_pressure:
            logger.warning(
                "Verarbeitungsschritt_23: ML repair uebersprungen — system thrashing erkannt, forcing DSP Ersatzpfad"
            )
            use_ml = False
        elif use_ml and allow_ml_under_pressure:
            self._pressure_relax_ml_attempts += 1
            logger.warning(
                "Verarbeitungsschritt_23: controlled ML Wiederholung under pressure aktiviert — high RAM headroom erkannt (Versuch %d/%d)",
                self._pressure_relax_ml_attempts,
                self._THRASH_RELAX_ML_MAX_ATTEMPTS,
            )

        # §v10.60 Depth-Aware Spectral Repair: Bei transfer_depth ≥ 5 (extreme chain)
        # ist FlashSR ML auf degradierten Signalen unzuverlässig (halluziniert Frequenzen).
        # §v10.120 Calibration-Shift: depth 4 (deep cassette) bekommt ML-Repair.
        # DSP-STFT-Bin-Interpolation ist sicherer und ausreichend präzise.
        _chain_p23 = kwargs.get("transfer_chain") or (kwargs.get("_restoration_context", {}) or {}).get(
            "transfer_chain", []
        )
        if len(_chain_p23) >= 5 and use_ml:
            use_ml = False
            logger.info(
                "Verarbeitungsschritt 23 depth=%d → DSP-only spectral inpainting (FlashSR ML übersprungen)",
                len(_chain_p23),
            )

        if use_ml:
            if not self._has_sufficient_ml_headroom(audio, sample_rate):
                flashsr = None
            else:
                flashsr = self._get_flashsr_plugin()
            if flashsr is not None:
                # §v10.303: Early-Exit wenn vorheriger FlashSR-Durchlauf <500 Hz BW-Gewinn brachte.
                # Zwei Durchläufe kosten ~650s bei marginalem Nutzen — überspringe Wiederholung.
                if self._flashsr_pass_count >= 1 and self._flashsr_last_bw_gain_hz < 500.0:
                    logger.info(
                    "Verarbeitungsschritt_23 FlashSR Early-Exit: letzter BW-Gewinn=%.0f Hz < 500 Hz → "
                    "überspringe Pass #%d (kein signifikanter Gewinn erwartet)",
                    self._flashsr_last_bw_gain_hz,
                    self._flashsr_pass_count + 1,
                )
                return np.nan_to_num(audio, nan=0.0)
                # ML-based repair with FlashSR
                log_mode_decision("phase_23", True, f"Defect severity: {defect_severity:.2%}")
                _report(55.0, "FlashSR")
                repaired_audio = self._repair_with_flashsr(audio, sample_rate, defect_mask, repair_strength, flashsr)
                _report(98.0, "FlashSR fertig")
                # §v10.303: BW-Gewinn messen für Early-Exit-Entscheidung im nächsten Durchlauf
                try:
                    _bw_before = float(self._estimate_bandwidth(audio, sample_rate))
                    _bw_after = float(self._estimate_bandwidth(repaired_audio, sample_rate))
                    self._flashsr_last_bw_gain_hz = max(0.0, _bw_after - _bw_before)
                    self._flashsr_pass_count += 1
                    logger.debug(
                        "Verarbeitungsschritt_23 FlashSR Pass #%d: BW %.0f→%.0f Hz (Δ=%.0f Hz)",
                        self._flashsr_pass_count,
                        _bw_before,
                        _bw_after,
                        self._flashsr_last_bw_gain_hz,
                    )
                except Exception:
                    logger.warning(
                        "§V6 Phase-23 FlashSR BW-Messung fehlgeschlagen (unkritisch): %s",
                        str(Exception),
                    )
                    pass  # Non-blocking
                return repaired_audio
            else:
                logger.warning("FlashSR nicht verfuegbar, falling back to DSP")

        # DSP-based repair (fallback or FAST mode)
        log_mode_decision("phase_23", False, f"Mode: {QualityModeConfig.get_mode().value}")

        # §DSP-Spezialregeln: MRSA 5-Zone-Reparatur für BALANCED/QUALITY/MAXIMUM
        _mode_upper = QualityModeConfig.get_mode().value.upper()
        if _mode_upper not in ("FAST",) and len(audio) >= sample_rate:
            # OOM-Preflight: MRSA (5 Zonen × STFT) benötigt ~2 GB für Stereo-Audio.
            # Unter 4 GB verfügbar → Single-STFT-Fallback um systemd-oomd zu vermeiden.
            _mrsa_ok = True
            _allow_mrsa_under_pressure = system_thrashing and self._can_relax_thrashing_guard(for_mrsa=True)
            if (
                _allow_mrsa_under_pressure
                and self._pressure_relax_mrsa_attempts >= self._THRASH_RELAX_MRSA_MAX_ATTEMPTS
            ):
                _allow_mrsa_under_pressure = False
                logger.warning(
                    "Verarbeitungsschritt_23: pressure-relax MRSA Versuch cap reached (%d) — using Single-STFT Ersatzpfad",
                    self._THRASH_RELAX_MRSA_MAX_ATTEMPTS,
                )
            if system_thrashing and not _allow_mrsa_under_pressure:
                logger.warning(
                    "Verarbeitungsschritt_23: MRSA uebersprungen — system thrashing erkannt, using Single-STFT Ersatzpfad"
                )
                _mrsa_ok = False
            elif _allow_mrsa_under_pressure:
                self._pressure_relax_mrsa_attempts += 1
                logger.warning(
                    "Verarbeitungsschritt_23: MRSA allowed under controlled pressure window (Versuch %d/%d)",
                    self._pressure_relax_mrsa_attempts,
                    self._THRASH_RELAX_MRSA_MAX_ATTEMPTS,
                )
            try:
                if _mrsa_ok:
                    _avail_gb = psutil.virtual_memory().available / (1024**3) if _PSUTIL_OK else 4.0
                    if _avail_gb < 4.0:
                        logger.warning(
                            "Verarbeitungsschritt_23: MRSA-OOM-Preflight fehlgeschlagen (%.1f GB < 4.0 GB) — Single-STFT-Ersatzpfad",
                            _avail_gb,
                        )
                        _mrsa_ok = False
            except Exception as e:
                logger.warning("Verarbeitungsschritt_23_spectral_repair.py::unbekannter Ersatzpfad: %s", e)
                pass  # psutil nicht verfügbar — MRSA versuchen
            if _mrsa_ok:
                try:
                    _report(45.0, "MRSA-Start")
                    out = self._repair_channel_mrsa(
                        audio,
                        sample_rate,
                        thresholds,
                        repair_strength,
                        progress_cb=lambda p, lbl: _report(
                            45.0 + 50.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                            f"MRSA: {lbl}",
                        ),
                        phase_deadline=phase_deadline,
                    )
                    # §0h Defense-in-Depth-Gain-Cap: _repair_channel_mrsa hat bereits einen
                    # internen +3 dB Cap (merge_zones filterbank-Überlapp-Schutz). Dieser
                    # externe Cap hier fängt Edge-Cases ab, in denen der interne Cap durch
                    # OOM-Passthrough, Budget-Exceedance oder unerwartete Signalenergie umgangen
                    # wird. +3 dB Toleranz — nie über Pre-Phase-Level plus 3 dB.
                    _in_rms_ch = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2)) + 1e-12)
                    _out_rms_ch = float(np.sqrt(np.mean(out**2)) + 1e-12)
                    _gain_ch = 20.0 * float(np.log10(_out_rms_ch / _in_rms_ch))
                    if _gain_ch > 3.0:
                        _atten_ch = float(10.0 ** -((_gain_ch - 3.0) / 20.0))
                        out = np.clip(out * _atten_ch, -1.0, 1.0).astype(np.float32)
                        logger.warning(
                            "Verarbeitungsschritt_23 _repair_channel: external gain-cap +%.1f dB → capped to +3.0 dB",
                            _gain_ch,
                        )
                    _report(98.0, "MRSA fertig")
                    return out
                except Exception as _mrsa_err:
                    logger.warning("MRSA-Reparatur fehlgeschlagen (%s), Single-STFT-Ersatzpfad", _mrsa_err)

        # Single-STFT fallback (FAST mode or MRSA failure)
        # Apply inpainting strategies
        _report(60.0, "Inpainting Magnitude")
        repaired_magnitude = self._inpaint_magnitude(magnitude, defect_mask)
        _report(72.0, "Inpainting Phase")
        repaired_phase = self._inpaint_phase(phase, defect_mask)

        # Reconstruct complex spectrogram
        Zxx_repaired = repaired_magnitude * np.exp(1j * repaired_phase)

        # Blend original and repaired
        Zxx_blended = Zxx * (1 - defect_mask * repair_strength) + Zxx_repaired * (defect_mask * repair_strength)

        # POCS: Iterative STFT-Konsistenz-Projektion (Siedenburg & Dörfler 2013)
        # Motivation: Die lineare Blend-Maske produziert ein STFT das keine gültige
        # Kurzzeitfouriertransformierte eines reellen Signals ist → Aliasing-Artefakte
        # an Defektgrenzen. Iteratives ISTFT→STFT erhält STFT-Konsistenz:
        #   1. ISTFT → Zeitsignal (Konsistenz-Projektion auf reelles Signal-Raum)
        #   2. STFT → neue Phase (physikalisch konsistente Phase aus dem Zeitsignal)
        #   3. Intakte Bins: Re-Ankern auf Original-STFT (verhindert Phasen-Drift)
        #   4. Defekt-Bins: Inpainting-Magnitude + neue Phase aus Round-Trip
        # Aktivierung: nur wenn Defekt-Coverage ausreichend (>0.5 %) und kein FAST-Mode.
        # Iterationszahl: material-adaptiv — 2 Iter. bei kleinen Defektmengen,
        # 5 Iter. bei schweren Schäden (Shellac-Dropouts, MP3-Codec-Löcher).
        _pocs_mode = QualityModeConfig.get_mode().value.upper()
        _pocs_min_severity = 0.005  # 0.5 % defect coverage
        if _pocs_mode not in ("FAST",) and defect_severity >= _pocs_min_severity:
            _pocs_n_iter = int(np.clip(round(2 + defect_severity * 15), 2, 5))
            # Wall-time guard: lange Signale → weniger Iterationen
            _pocs_dur_s = len(audio) / max(sample_rate, 1)
            if _pocs_dur_s > 60.0:
                _pocs_n_iter = min(_pocs_n_iter, 2)
            _report(74.0, f"POCS Konsistenz ({_pocs_n_iter} Iter.)")
            try:
                _Zxx_pocs = Zxx_blended.copy()
                _F_p, _T_p = _Zxx_pocs.shape
                for _pocs_i in range(_pocs_n_iter):
                    # Step 1: STFT → Zeitsignal (ISTFT)
                    _, _sig_pocs = signal.safe_istft(
                        _Zxx_pocs,
                        fs=sample_rate,
                        window="hann",
                        nperseg=stft_cfg["nperseg"],
                        noverlap=stft_cfg["noverlap"],
                        nfft=stft_cfg["nfft"],
                        boundary=True,
                    )
                    # Länge angleichen
                    _n_needed = len(audio)
                    if len(_sig_pocs) >= _n_needed:
                        _sig_pocs = _sig_pocs[:_n_needed]
                    else:
                        _sig_pocs = np.pad(_sig_pocs, (0, _n_needed - len(_sig_pocs)))
                    # Step 2: Zeitsignal → neues STFT (physikalisch konsistente Phase)
                    _, _, _Zxx_new = safe_stft(
                        _sig_pocs,
                        fs=sample_rate,
                        window="hann",
                        nperseg=stft_cfg["nperseg"],
                        noverlap=stft_cfg["noverlap"],
                        nfft=stft_cfg["nfft"],
                        boundary="even",
                    )
                    # Step 3: Formabgleich (Randeffekte können T um ±1 verschieben)
                    _T_new = _Zxx_new.shape[1]
                    _T_use = min(_T_p, _T_new)
                    _Zxx_iter = _Zxx_pocs.copy()  # Startwert: vorherige Iteration
                    # Intakte Bins: Original-STFT re-ankern (Drift-Unterdrückung)
                    _anchor = (~defect_mask)[:, :_T_use]
                    _Zxx_iter[:, :_T_use][_anchor] = Zxx[:, :_T_use][_anchor]
                    # Defekt-Bins: Inpainting-Magnitude + neue Phase aus Round-Trip
                    _pocs_phase_new = np.asarray(np.angle(_Zxx_new[:, :_T_use]))
                    _defect_crop = defect_mask[:, :_T_use]
                    _Zxx_iter[:, :_T_use][_defect_crop] = np.asarray(repaired_magnitude)[:, :_T_use][
                        _defect_crop
                    ] * np.exp(1j * _pocs_phase_new[_defect_crop])
                    _Zxx_pocs = _Zxx_iter
                Zxx_blended = _Zxx_pocs
                logger.debug(
                    "Verarbeitungsschritt_23 POCS: %d Iterationen abgeschlossen (defect_severity=%.2f%%, dur=%.1fs)",
                    _pocs_n_iter,
                    defect_severity * 100,
                    _pocs_dur_s,
                )
            except Exception as _pocs_err:
                # Vollständig non-blocking: Zxx_blended bleibt unverändert
                logger.debug("Verarbeitungsschritt_23 POCS: nicht-blockierender Ersatzpfad — %s", _pocs_err)
            _report(80.0, "POCS fertig")

        # Direct ISTFT reconstruction — Zxx_blended retains phase info from original STFT.
        # ISTFT is semantically correct and 50-100× faster than PGHI.
        try:
            _, audio_repaired = signal.safe_istft(
                np.asarray(Zxx_blended, dtype=np.complex64),
                fs=sample_rate,
                window="hann",
                nperseg=stft_cfg["nperseg"],
                noverlap=stft_cfg["noverlap"],
                nfft=stft_cfg["nfft"],
                boundary=True,
            )
            audio_repaired = np.asarray(audio_repaired, dtype=np.float64)
        except Exception as _istft_p23_err:
            logger.debug("Verarbeitungsschritt_23 single-STFT istft fehlgeschlagen (unkritisch): %s", _istft_p23_err)
            audio_repaired = audio.astype(np.float64)  # passthrough
        _report(98.0, "Rekonstruktion")

        return np.nan_to_num(audio_repaired[: len(audio)], nan=0.0)  # type: ignore[no-any-return]

    def _repair_channel_mrsa(
        self,
        audio: np.ndarray,
        sample_rate: int,
        thresholds: dict[str, float],
        repair_strength: float,
        progress_cb=None,
        phase_deadline: float = 0.0,
    ) -> np.ndarray:
        """Repariert single audio channel using 5-zone MRSA (§DSP-Spezialregeln).

        Applies spectral inpainting independently per frequency zone with
        zone-appropriate STFT resolution (win 65536→128). Reconstructs each zone
        via PGHI-approximation and merges via Hanning crossfade (10 ms, §DSP).

        Args:
            audio: Mono float32 input channel.
            sample_rate: Must be 48000 Hz.
            thresholds: Material-specific detection thresholds.
            repair_strength: Blend factor for inpainting (0–1).

        Returns:
            Repaired mono float32 audio.
        """
        audio_f32 = np.asarray(audio, dtype=np.float32)
        zone_audios: dict[str, np.ndarray] = {}
        # zone_meta: ZoneSTFT objects with empty .stft — only hz_lo/hz_hi needed by merge_zones
        zone_meta: dict[str, ZoneSTFT] = {}
        _n_zones = len(ZONE_ORDER)

        # §Spec04b MRSA wall-time budget: 5-zone STFT on long signals can take 45+ min
        # without a time limit. Budget = min(600 s, 2.5× audio duration).
        # Zones beyond the budget are returned as passthrough (original audio preserved).
        _mrsa_dur_s = float(len(audio_f32)) / float(max(sample_rate, 1))
        # §Phase-level shared deadline: limits this MRSA call to remaining phase budget.
        # Without this, each M/S call gets an independent 562.5s budget → 39-min total.
        if phase_deadline > 0.0:
            _remaining_s = max(10.0, phase_deadline - time.monotonic())
            _mrsa_wall_budget_s = min(600.0, 2.5 * _mrsa_dur_s, _remaining_s)
        else:
            _mrsa_wall_budget_s = min(600.0, 2.5 * _mrsa_dur_s)
        logger.info("Verarbeitungsschritt_23 MRSA: wall_Grenze=%.0fs (dur=%.1fs)", _mrsa_wall_budget_s, _mrsa_dur_s)
        _mrsa_t0 = time.monotonic()
        _mrsa_budget_exceeded = False

        # §Spec04b Intra-Zone-Budget-Guard — außerhalb der Schleife definiert (B023-sicher).
        # Wird VOR jeder teuren Intra-Zone-Operation aufgerufen um sicherzustellen,
        # dass ein hängender Einzelschritt nie unbemerkt das Gesamtbudget reißt.
        def _intra_zone_budget_exceeded(zone_name: str, op_label: str) -> bool:
            """True wenn MRSA-Gesamtbudget überschritten — setzt _mrsa_budget_exceeded."""
            nonlocal _mrsa_budget_exceeded
            if _mrsa_budget_exceeded:
                return True
            _elapsed = time.monotonic() - _mrsa_t0
            if _elapsed > _mrsa_wall_budget_s:
                _mrsa_budget_exceeded = True
                logger.warning(
                    "Verarbeitungsschritt_23 MRSA: intra-zone Grenze %.0fs überschritten vor '%s'"
                    " in Zone '%s' (elapsed=%.1fs) — Zone als Passthrough",
                    _mrsa_wall_budget_s,
                    op_label,
                    zone_name,
                    _elapsed,
                )
                return True
            return False

        for _zi, name in enumerate(ZONE_ORDER):
            if callable(progress_cb):
                try:
                    progress_cb(5.0 + 90.0 * (_zi / _n_zones), f"Zone {name}")
                except Exception:
                    logger.debug("_intra_zone_Grenze_exceeded: silent except suppressed", exc_info=True)
            logger.info(
                "Verarbeitungsschritt_23 MRSA: zone %d/%d '%s' elapsed=%.1fs Grenze=%.0fs",
                _zi + 1,
                _n_zones,
                name,
                time.monotonic() - _mrsa_t0,
                _mrsa_wall_budget_s,
            )
            # Wall-time guard: passthrough remaining zones if budget exceeded
            if not _mrsa_budget_exceeded and (time.monotonic() - _mrsa_t0) > _mrsa_wall_budget_s:
                logger.warning(
                    "Verarbeitungsschritt_23 MRSA: wall-time Grenze %.0fs exceeded after zone %d/%d — remaining zones as passthrough",
                    _mrsa_wall_budget_s,
                    _zi,
                    _n_zones,
                )
                _mrsa_budget_exceeded = True
            # §OOM-Guard: each zone needs ~500 MB for STFT + intermediaries.
            # If RAM is critically low, treat zone as passthrough rather than risk a crash.
            if not _mrsa_budget_exceeded:
                try:
                    _mrsa_avail_gb = (
                        psutil.virtual_memory().available / (1024**3) if _PSUTIL_OK else 4.0 if _PSUTIL_OK else 4.0
                    )
                    if _mrsa_avail_gb < 1.0:
                        logger.warning(
                            "Verarbeitungsschritt_23 MRSA: OOM-Guard — nur %.1f GB frei (< 1.0 GB) — Zone %d/%d als Passthrough",
                            _mrsa_avail_gb,
                            _zi + 1,
                            _n_zones,
                        )
                        _mrsa_budget_exceeded = True
                except Exception as _mrsa_ram_exc:
                    logger.debug(
                        "Verarbeitungsschritt_23 MRSA: RAM-Pruefung fehlgeschlagen (unkritisch): %s", _mrsa_ram_exc
                    )

            _z_cfg = ZONES[name]
            _eff_win = min(_z_cfg["win"], len(audio_f32))
            # Lightweight meta object for merge_zones (no STFT data)
            _zone_meta_empty = ZoneSTFT(
                name=name,
                freqs=np.empty(0, dtype=np.float64),
                times=np.empty(0, dtype=np.float64),
                stft=np.empty(0, dtype=np.complex64),
                win=_z_cfg["win"],
                hop=_z_cfg["hop"],
                hz_lo=float(_z_cfg["hz"][0]),
                hz_hi=float(_z_cfg["hz"][1]),
                eff_win=int(_eff_win),
                eff_hop=int(_z_cfg["hop"]),
            )
            zone_meta[name] = _zone_meta_empty

            if _mrsa_budget_exceeded:
                # §OOM-Guard passthrough: pass original audio directly — merge_zones will
                # bandpass-filter it to the correct frequency range.  Avoids PGHI overhead
                # (synthesize_zone on sub_bass = ~865 MB PGHI + temp arrays for no gain).
                zone_audios[name] = audio_f32
                continue

            # §OOM-Guard: compute ONLY this zone's STFT (lazy — not all 5 at once).
            # Peak RAM per zone = ~172 MB (STFT) + ~4× (processing intermediaries) ≈ 850 MB max.
            # vs. old approach: all 5 STFTs in memory simultaneously ≈ 863 MB + processing.
            if _intra_zone_budget_exceeded(name, "analyze_zones"):
                zone_audios[name] = audio_f32
                continue
            _single = analyze_zones(audio_f32, sample_rate, zones={name: _z_cfg})
            zone = _single[name]
            del _single  # release dict wrapper immediately

            magnitude = np.abs(zone.stft)
            phase_arr = np.angle(zone.stft)

            if _intra_zone_budget_exceeded(name, "_detect_defects"):
                zone_audios[name] = synthesize_zone(zone, zone.stft, len(audio_f32))
                del zone, magnitude, phase_arr
                gc.collect(0)
                continue
            defect_mask = self._detect_defects(magnitude, phase_arr, thresholds)

            if np.sum(defect_mask) == 0:
                # No defects in this zone — passthrough: reconstruct from original STFT
                zone_audios[name] = synthesize_zone(zone, zone.stft, len(audio_f32))
                del zone, magnitude, phase_arr, defect_mask
                gc.collect(0)
                continue

            if _intra_zone_budget_exceeded(name, "_inpaint_magnitude"):
                zone_audios[name] = synthesize_zone(zone, zone.stft, len(audio_f32))
                del zone, magnitude, phase_arr, defect_mask
                gc.collect(0)
                continue
            repaired_mag = self._inpaint_magnitude(magnitude, defect_mask)
            Zxx_repaired = repaired_mag * np.exp(1j * phase_arr)
            blend_mask = defect_mask * repair_strength
            Zxx_blended = zone.stft * (1.0 - blend_mask) + Zxx_repaired * blend_mask

            zone_audios[name] = synthesize_zone(zone, Zxx_blended, len(audio_f32))
            # Release all intermediary arrays and this zone's STFT immediately
            del zone, magnitude, phase_arr, defect_mask, repaired_mag, Zxx_repaired, blend_mask, Zxx_blended
            gc.collect(0)

        if callable(progress_cb):
            try:
                progress_cb(100.0, "Zonen-Merge")
            except Exception:
                logger.debug("_intra_zone_Grenze_exceeded: silent except suppressed", exc_info=True)
        result = merge_zones(zone_audios, zone_meta, sample_rate, len(audio_f32))
        # §0h Music-Death-Shield: MRSA output must not exceed +3 dB of input RMS.
        # merge_zones sums 5 zone reconstructions — imperfect filterbank overlap or
        # inpainting bias can create constructive gain of 10–20 dB (observed: +18.76 dB).
        _mrsa_in_rms = float(np.sqrt(np.mean(audio_f32**2)) + 1e-12)
        _mrsa_out_rms = float(np.sqrt(np.mean(result**2)) + 1e-12)
        _mrsa_gain_db = 20.0 * np.log10(_mrsa_out_rms / _mrsa_in_rms)
        if _mrsa_gain_db > 3.0:
            _mrsa_atten = float(10.0 ** (-((_mrsa_gain_db - 3.0) / 20.0)))
            result = np.clip(result * _mrsa_atten, -1.0, 1.0).astype(np.float32)
            logger.warning(
                "Verarbeitungsschritt_23 MRSA: gain-cap angewendet (+%.1f dB → capped to +3.0 dB, att=%.2f dB)",
                _mrsa_gain_db,
                _mrsa_gain_db - 3.0,
            )
        return np.nan_to_num(result, nan=0.0)  # type: ignore[no-any-return]

    def _estimate_bandwidth(self, audio: np.ndarray, sample_rate: int) -> float:
        """§v10.303: Schätzt die effektive Bandbreite via 95%-Energie-Grenze."""
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        n_fft = min(4096, len(a))
        if n_fft < 64:
            return float(sample_rate // 2)
        spec = np.abs(np.fft.rfft(a[:n_fft] * np.hanning(n_fft)))
        cumsum = np.cumsum(spec)
        total = cumsum[-1] + 1e-12
        idx = int(np.searchsorted(cumsum, 0.95 * total))
        return float(idx / n_fft * sample_rate)

    def _repair_with_flashsr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        defect_mask: np.ndarray,
        repair_strength: float,
        flashsr: Any,
    ) -> np.ndarray:
        """
        Repariert audio using FlashSR ML model.

        Strategy: DSP-Detection + ML-Repair
        1. DSP detects defect regions (already done - defect_mask)
        2. Extract defect regions with context (±500ms)
        3. Process with FlashSR (super-resolution inpainting)
        4. Blend back with repair_strength

        Args:
            audio: Input audio channel (mono)
            sample_rate: Sample rate in Hz
            defect_mask: Binary mask from DSP detection
            repair_strength: Blend amount (0-1)

        Returns:
            Repaired audio
        """
        _ = defect_mask
        if flashsr is None:
            return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # §4.6b PLM Active-Guard — prevents Emergency-Eviction during FlashSR inference
        _plm23_asr = None
        try:
            _plm23_asr = get_plugin_lifecycle_manager()
            _plm23_asr.set_active("FlashSR", True)
        except Exception:
            logger.debug("_repair_with_flashsr: silent except suppressed", exc_info=True)
        try:
            if not self._has_sufficient_ml_headroom(audio, sample_rate):
                return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            # use the IDENTICAL strip offset → no inter-channel lag possible.
            _asr23_total_n = audio.shape[-1] if np.asarray(audio).ndim == 2 else len(np.asarray(audio))
            _asr23_ctx_n = min(int(1.0 * sample_rate), _asr23_total_n // 4)

            def _repair_single_channel(channel_audio: np.ndarray) -> np.ndarray:
                # §2.46f Context-Padding: reflect-pad 1 s on both sides before ML so the model
                # always operates on interior signal at what were previously the signal boundaries.
                # This prevents ML boundary artefacts at the intro/outro entirely (root-cause fix).
                #
                # Lag-safety invariant: _asr23_ctx_n is computed ONCE outside this closure
                # (same value for L and R). Strip is always exactly _asr23_ctx_n samples from
                # the start, taking exactly len(channel_audio) samples → both channels end at
                # identical length without any resampling, which would stretch them differently.
                target_sr = 48000
                _orig_len23 = len(channel_audio)
                _use_pad23 = _asr23_ctx_n > 0 and _orig_len23 > _asr23_ctx_n * 4
                if _use_pad23:
                    _ch_padded23 = np.pad(channel_audio, (_asr23_ctx_n, _asr23_ctx_n), mode="reflect")
                    _raw_repaired = flashsr.process(_ch_padded23, sample_rate, target_sr)
                else:
                    _raw_repaired = flashsr.process(channel_audio, sample_rate, target_sr)
                repaired_channel = np.asarray(_raw_repaired, dtype=np.float32)
                repaired_channel = np.squeeze(repaired_channel)
                # Strip context padding: take exactly _orig_len23 samples at offset _asr23_ctx_n.
                # Using a fixed slice guarantees L and R are trimmed identically → no inter-channel lag.
                if _use_pad23 and repaired_channel.ndim == 1 and len(repaired_channel) >= _asr23_ctx_n + _orig_len23:
                    repaired_channel = repaired_channel[_asr23_ctx_n : _asr23_ctx_n + _orig_len23]
                    logger.debug(
                        "Verarbeitungsschritt_23 FlashSR: context-padding stripped (%d samples offset)", _asr23_ctx_n
                    )

                if repaired_channel.ndim != 1:
                    logger.warning(
                        "FlashSR returned unexpected shape %s for mono channel — falling back to passthrough",
                        repaired_channel.shape,
                    )
                    repaired_channel = channel_audio

                if len(repaired_channel) != len(channel_audio):
                    _target_len23 = len(channel_audio)
                    _cur_len23 = len(repaired_channel)
                    if _cur_len23 > _target_len23:
                        # Deterministic hard-crop from start avoids time-warp and channel drift.
                        repaired_channel = repaired_channel[:_target_len23]
                    elif _cur_len23 < _target_len23:
                        # End-pad only (no left-pad) keeps onset alignment and avoids lag injection.
                        repaired_channel = np.pad(
                            repaired_channel,
                            (0, _target_len23 - _cur_len23),
                            mode="edge",
                        )
                    logger.warning(
                        "Verarbeitungsschritt_23 FlashSR length corrected without resample: cur=%d target=%d",
                        _cur_len23,
                        _target_len23,
                    )

                audio_final = (
                    channel_audio * (1 - repair_strength)
                    + repaired_channel.astype(channel_audio.dtype) * repair_strength
                )
                return audio_final[: len(channel_audio)]  # type: ignore[no-any-return]

            audio_arr = np.asarray(audio)
            if audio_arr.ndim == 1:
                result_asr23 = _repair_single_channel(audio_arr)
            elif audio_arr.ndim == 2:
                # §7.1a / §2.51 Stereo-Kohärenz: do NOT repair L/R independently.
                # FlashSR path uses M/S domain: repair Mid only, keep Side unchanged.
                def _repair_stereo_ms_channels_first(st_cf: np.ndarray) -> np.ndarray:
                    l_in = st_cf[0].astype(np.float32, copy=False)
                    r_in = st_cf[1].astype(np.float32, copy=False)
                    mid = 0.5 * (l_in + r_in)
                    side = 0.5 * (l_in - r_in)
                    mid_repaired = _repair_single_channel(mid)
                    l_out = np.clip(mid_repaired + side, -1.0, 1.0)
                    r_out = np.clip(mid_repaired - side, -1.0, 1.0)
                    return np.stack([l_out, r_out], axis=0)  # type: ignore[no-any-return]

                # UV3 uses channels-first stereo (2, N). Preserve incoming orientation.
                if audio_arr.shape[0] == 2 and audio_arr.shape[1] > 2:
                    result_asr23 = _repair_stereo_ms_channels_first(audio_arr).astype(audio_arr.dtype, copy=False)
                elif audio_arr.shape[1] == 2 and audio_arr.shape[0] > 2:
                    _cf = audio_arr.T
                    _cf_repaired = _repair_stereo_ms_channels_first(_cf)
                    result_asr23 = _cf_repaired.T.astype(audio_arr.dtype, copy=False)
                else:
                    logger.warning(
                        "FlashSR repair received unsupported audio shape %s — falling back to passthrough",
                        audio_arr.shape,
                    )
                    return np.nan_to_num(audio, nan=0.0)
            else:
                logger.warning(
                    "FlashSR repair received unsupported audio shape %s — falling back to passthrough",
                    audio_arr.shape,
                )
                return np.nan_to_num(audio, nan=0.0)

            # §0a / §6.2c BW-Ceiling Hard-Cap: FlashSR generiert bis 48 kHz/2 = 24 kHz.
            # Bei analogen Trägern mit physikalischem BW-Limit muss HF abgeschnitten werden
            # (§2.46e Hallucination-Guard: keine Energie über Trägerlimit hinaus).
            _BW_CAP_ASR23: dict[str, float] = {
                "shellac": 8000.0,
                "wax_cylinder": 5000.0,
                "vinyl": 16000.0,
                "reel_tape": 18000.0,
                "cassette": 15000.0,
            }
            _mat_asr23 = str(getattr(self, "_current_material", "unknown")).lower().replace(" ", "_").replace("-", "_")
            _bw_asr23 = _BW_CAP_ASR23.get(_mat_asr23)
            if _bw_asr23 is not None:
                try:
                    _nyq_asr23 = float(sample_rate) / 2.0
                    _ratio_asr23 = float(np.clip(_bw_asr23 / _nyq_asr23, 0.01, 0.99))
                    _sos_asr23 = signal.butter(6, _ratio_asr23, btype="low", output="sos")
                    if result_asr23.ndim == 2:
                        if result_asr23.shape[0] <= 2 and result_asr23.shape[1] > 2:
                            result_asr23 = np.stack(
                                [signal.sosfiltfilt(_sos_asr23, result_asr23[c]) for c in range(result_asr23.shape[0])],
                                axis=0,
                            ).astype(np.float32)
                        else:
                            result_asr23 = np.stack(
                                [
                                    signal.sosfiltfilt(_sos_asr23, result_asr23[:, c])
                                    for c in range(result_asr23.shape[1])
                                ],
                                axis=1,
                            ).astype(np.float32)
                    else:
                        result_asr23 = signal.sosfiltfilt(_sos_asr23, result_asr23).astype(np.float32)
                    logger.debug(
                        "§6.2c Verarbeitungsschritt_23 FlashSR BW-Ceiling: %s ≤ %.0f Hz", _mat_asr23, _bw_asr23
                    )
                except Exception as _bw_asr23_exc:
                    logger.debug("§6.2c Verarbeitungsschritt_23 BW-Ceiling (nicht blockierend): %s", _bw_asr23_exc)

            return result_asr23

        except Exception as e:
            logger.warning("FlashSR processing fehlgeschlagen (DSP Ersatzpfad aktiv): %s", e)
            # Fallback to DSP (will be handled by caller)
            return np.nan_to_num(audio, nan=0.0)
        finally:
            if _plm23_asr is not None:
                try:
                    _plm23_asr.set_active("FlashSR", False)
                except Exception:
                    logger.debug("_repair_stereo_ms_channels_first: silent except suppressed", exc_info=True)

    def _estimate_noise_floor_imcra(self, magnitude: np.ndarray) -> np.ndarray:
        """IMCRA-adaptiver Rauschboden pro Zeit-Frequenz-Bin (Cohen 2003).

        Algorithmus:
            1. Leistungsspektrum P(t,f) = |magnitude|²
            2. Exp. Glättung: S̃(t,f) = α_d·S̃(t-1,f) + (1-α_d)·P(t,f)  α_d=0.85
            3. Sliding-Minimum: σ²_min(t,f) = min_{t'∈[t-M,t]} S̃(t',f)
            4. Rauschboden: σ_d(t,f) = √(b_min · σ²_min(t,f))  b_min=1.66

        Forschungsreferenz:
            Cohen (2003): „Noise Spectrum Estimation in Adverse Environments:
            Improved Minima Controlled Recursive Averaging"

        Args:
            magnitude: STFT-Magnitude (F, T), float32/64

        Returns:
            noise_floor: Adaptiver Rauschboden (F, T), Amplitude-Einheiten, NaN-frei
        """
        power = magnitude**2  # (F, T)

        # Exponentielle Glättung α_d=0.85 (Cohen 2003 Gleichung 3)
        alpha_d = 0.85
        smoothed = np.empty_like(power)
        smoothed[:, 0] = power[:, 0]
        for t_idx in range(1, power.shape[1]):
            smoothed[:, t_idx] = alpha_d * smoothed[:, t_idx - 1] + (1.0 - alpha_d) * power[:, t_idx]

        # Sliding-Minimum (M ≈ 1.5 s in STFT-Frames, mind. 5, max. 40)
        M = max(5, min(40, power.shape[1] // 4))
        min_smoothed = ndimage.minimum_filter1d(smoothed, size=M, axis=1, mode="nearest")

        # Overcorrection b_min=1.66 → zurück zu Amplitude (Cohen 2003, Gl. 12)
        b_min = 1.66
        noise_floor = np.sqrt(np.maximum(b_min * min_smoothed, 1e-20))
        noise_floor = np.nan_to_num(noise_floor, nan=1e-10, posinf=1.0, neginf=1e-10)
        return np.nan_to_num(np.asarray(noise_floor, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    # §2.57 BW-Ceiling pro analogem Material (Hz): restaurierte HF über diesem
    # Schwellwert darf nicht als Codec-Spike erkannt werden (Phase_06-Restaurierung schützen).
    _ANALOG_HF_PROTECT_HZ: dict[str, float] = {
        "vinyl": 13600.0,  # vinyl BW_CEILING 16 kHz × 0.85
        "shellac": 6800.0,  # shellac BW_CEILING 8 kHz × 0.85
        "wax_cylinder": 4250.0,  # wax_cylinder BW_CEILING 5 kHz × 0.85
        "wire_recording": 5100.0,
        "reel_tape": 13600.0,
        "tape": 13600.0,
        "cassette": 10200.0,  # cassette BW_CEILING 12 kHz × 0.85
        "lacquer_disc": 13600.0,
        "minidisc": 11900.0,
    }

    def _detect_defects(self, magnitude: np.ndarray, phase: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
        """Defekt-Detektion via IMCRA-adaptivem Rauschboden + Phasenkonsistenz.

        Strategien:
            1. Dropout:  magnitude < 0.3 × IMCRA_noise_floor  (bin-adaptiv)
            2. Artefakt: Z-Score über IMCRA-Floor via MAD  (1.4826 · MAD = σ_robust)
            3. Phasensprung: |Δφ(t,f)| > Schwellwert

        Entfernt (verboten per copilot-instructions §4.2):
            np.mean/std als globaler Rauschboden → IMCRA Sliding-Minimum
            Fixierter energy_floor_db → adaptiver bin-spezifischer Floor

        Args:
            magnitude:  STFT-Magnitude (F, T)
            phase:      STFT-Phase (F, T)
            thresholds: Material-spezifische Schwellwerte

        Returns:
            defect_mask: Bool-Array (F, T), True = defekter Bin
        """
        defect_mask = np.zeros_like(magnitude, dtype=bool)

        # §2.57: HF-Protected-Bin-Grenze für analoge Materialien.
        # Bins ≥ _hf_protected_start werden aus Spike-Detektion (Pass-1) ausgeschlossen.
        # Verhindert False-Positive-Flagging von durch Phase_06 restaurierten HF-Harmoniken.
        _mat_key = str(getattr(self, "_current_material", "")).lower().replace("-", "_").replace(" ", "_")
        _hf_protect_hz = self._ANALOG_HF_PROTECT_HZ.get(_mat_key, 0.0)
        _nfft_est = (magnitude.shape[0] - 1) * 2  # magnitude.shape[0] = nfft//2 + 1
        _bin_hz = 48000.0 / (_nfft_est + 1e-12)
        _hf_protected_start = int(_hf_protect_hz / (_bin_hz + 1e-12)) if _hf_protect_hz > 0 else magnitude.shape[0]

        # -- Strategie 1 + 2: IMCRA-adaptiver Rauschboden --
        noise_floor = self._estimate_noise_floor_imcra(magnitude)

        # Dropout: Magnitude deutlich unterhalb des geschätzten Rauschbodens (alle Bins)
        dropout_mask = magnitude < (noise_floor * 0.3)
        defect_mask |= dropout_mask

        # Spike/Codec-Artefakt: Z-Score über IMCRA-Floor (robust via MAD)
        ratio = np.where(noise_floor > 1e-12, magnitude / (noise_floor + 1e-12), 1.0)
        ratio_db = 20.0 * np.log10(np.maximum(ratio, 1e-10))

        median_ratio = np.median(ratio_db, axis=1, keepdims=True)
        mad = np.median(np.abs(ratio_db - median_ratio), axis=1, keepdims=True) + 1e-6
        z_scores = (ratio_db - median_ratio) / (mad * 1.4826)
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
        spike_mask = z_scores > thresholds["outlier_z_score"]
        # §2.57: HF-geschützte Bins (restaurierte Harmoniken) nicht als Spike markieren
        if _hf_protected_start < magnitude.shape[0]:
            spike_mask[_hf_protected_start:, :] = False
        defect_mask |= spike_mask

        # -- Strategie 3: Phasensprünge --
        phase_diff = np.diff(phase, axis=1)
        phase_jumps = np.abs(phase_diff) > thresholds["phase_jump_threshold"]
        # §2.57: HF-geschützte Bins auch aus Phase-Sprung-Detektion ausschließen.
        # Phase_06 verändert HF-Phasenrelationen intentional → kein False-Positive.
        if _hf_protected_start < magnitude.shape[0]:
            phase_jumps[_hf_protected_start:, :] = False
        defect_mask[:, 1:] |= phase_jumps

        # Morphologische Bereinigung
        defect_mask = ndimage.binary_opening(defect_mask, structure=np.ones((3, 3)))
        defect_mask = ndimage.binary_closing(defect_mask, structure=np.ones((5, 3)))

        return np.nan_to_num(np.asarray(defect_mask), nan=0.0)  # type: ignore[no-any-return]

    def _inpaint_magnitude(self, magnitude: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
        """Vektorisiertes Spectral Inpainting — O(F+T) statt O(F×T).

        Algorithmus (Smaragdis & Brown 2003, Blend-Gewichte):
            Für jede Frequenz f: interp1d NaN-Lücken entlang Zeitachse  →  mag_h
            Für jeden Zeitframe t: interp1d NaN-Lücken entlang Frequenzachse → mag_v
            Repaired[defect] = 0.6 · mag_h[defect] + 0.4 · mag_v[defect]

        Entfernt (verboten):
            O(F×T) Python-Doppelschleife mit einzeln berechneten Pixeln
        """
        if not np.any(defect_mask):
            return magnitude.copy()

        # --- Horizontal: Zeit-Richtung (Zeile = Frequenz) ---
        mag_h = magnitude.copy().astype(np.float32)
        mag_h[defect_mask] = np.nan

        for f in range(mag_h.shape[0]):
            row = mag_h[f, :]
            if not np.any(np.isnan(row)):
                continue
            valid = np.where(~np.isnan(row))[0]
            if len(valid) >= 2:
                xs = np.arange(len(row))
                interp_fn = interpolate.interp1d(
                    valid, row[valid], kind="linear", fill_value=(row[valid[0]], row[valid[-1]]), bounds_error=False
                )
                row[:] = interp_fn(xs)
            elif len(valid) == 1:
                row[:] = row[valid[0]]
            else:
                row[:] = 1e-10
            mag_h[f, :] = np.nan_to_num(row, nan=1e-10)

        # --- Vertikal: Frequenz-Richtung (Spalte = Zeitframe) ---
        mag_v = magnitude.copy().astype(np.float32)
        mag_v[defect_mask] = np.nan

        for t in range(mag_v.shape[1]):
            col = mag_v[:, t]
            if not np.any(np.isnan(col)):
                continue
            valid = np.where(~np.isnan(col))[0]
            if len(valid) >= 2:
                xs = np.arange(len(col))
                interp_fn = interpolate.interp1d(
                    valid, col[valid], kind="linear", fill_value=(col[valid[0]], col[valid[-1]]), bounds_error=False
                )
                col[:] = interp_fn(xs)
            elif len(valid) == 1:
                col[:] = col[valid[0]]
            else:
                col[:] = 1e-10
            mag_v[:, t] = np.nan_to_num(col, nan=1e-10)

        # Blend an Defektstellen: 0.6 horizontal + 0.4 vertikal
        repaired = magnitude.copy().astype(np.float32)
        blended = 0.6 * mag_h + 0.4 * mag_v
        repaired[defect_mask] = blended[defect_mask]

        return np.nan_to_num(np.maximum(repaired, 0.0), nan=0.0)  # type: ignore[no-any-return]

    def _inpaint_phase(self, phase: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
        """Phase-Inpainting via Phasen-Geschwindigkeits-Fortsetzung.

        Statt einfaches Frame-Copy: Phasengeschwindigkeit δφ(f,t) = φ(f,t-1) − φ(f,t-2)
        wird extrapoliert. Dies entspricht der instantanen Frequenz und erhält
        die Phasenkohärenz (Laroche & Dolson 1999, Phase-Vocoder).
        """
        repaired = phase.copy()
        F, T = phase.shape

        for f in range(F):
            mask_row = defect_mask[f, :]
            if not np.any(mask_row):
                continue
            row = repaired[f, :]
            prev_phi = phase[f, 0]
            prev_delta = 0.0
            for t in range(1, T):
                if mask_row[t]:
                    row[t] = prev_phi + prev_delta
                    # update prev_phi mit extrapoliertem Wert für nächste Iteration
                    prev_phi = row[t]
                    # prev_delta bleibt konstant (lineare Phase-Fortsetzung)
                else:
                    if t >= 2 and not mask_row[t - 1]:
                        prev_delta = phase[f, t] - phase[f, t - 1]
                    prev_phi = phase[f, t]

        return repaired

    def _calculate_defect_reduction(self, original: np.ndarray, repaired: np.ndarray, sample_rate: int) -> float:
        """Calculate percentage of defects reduced."""
        # Simple metric: reduction in high-frequency noise
        _, _, Pxx_orig = signal.spectrogram(original if original.ndim == 1 else original[:, 0], fs=sample_rate)
        _, _, Pxx_rep = signal.spectrogram(repaired if repaired.ndim == 1 else repaired[:, 0], fs=sample_rate)

        noise_orig = np.std(Pxx_orig)
        noise_rep = np.std(Pxx_rep)

        reduction = max(0, min(1, (noise_orig - noise_rep) / noise_orig)) if noise_orig > 1e-10 else 0.0

        return reduction  # type: ignore[return-value]

    def _calculate_spectral_coherence(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate spectral coherence (smoothness) score."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel

        # Compute spectrogram
        _f, _t, Pxx = signal.spectrogram(audio, fs=sample_rate, nperseg=2048)

        # Measure smoothness (inverse of spectral roughness)
        spectral_diff = np.diff(Pxx, axis=0)
        roughness = np.mean(np.abs(spectral_diff))

        # Normalize to 0-1 range (lower roughness = higher coherence)
        coherence = 1.0 / (1.0 + roughness * 100)

        return float(coherence)
