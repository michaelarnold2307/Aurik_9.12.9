"""
Phase 1: Professional Click Removal - Aurik 10.0.0
================================================

Professional-grade click and pop removal competing with iZotope RX De-click.

ALGORITHM (Professional-Level):
--------------------------------
1. **Multi-Scale Detection**
   - Short clicks (1-3 samples): Digital errors, vinyl ticks
   - Medium clicks (4-10 samples): Vinyl pops, digital glitches
   - Long clicks (11-50 samples): Scratches, handling noise

2. **Click-Type Classification**
   - Digital clicks: Sharp edges, full-scale excursions
   - Analog clicks: Softer attacks, vinyl/tape artifacts
   - Musical transients: Preserve legitimate attacks (drums, etc.)

3. **Adaptive Interpolation**
   - Linear: Short clicks, simple waveforms
   - Cubic Spline: Medium clicks, smooth transitions
   - Spectral: Long clicks, complex harmonic content
   - ARX-based: Tonal content with phase coherence

4. **Material-Adaptive Processing**
   - Shellac: Aggressive (threshold=0.05, many clicks expected)
   - Vinyl: Moderate (threshold=0.10, typical wear)
   - Tape: Gentle (threshold=0.20, preserve dynamics)
   - CD/Digital: Conservative (threshold=0.30, rare clicks)

SCIENTIFIC FOUNDATION:
---------------------
- **Godsill & Rayner (1998)**: "Digital Audio Restoration"
  → Bayesian click detection and interpolation
- **Välimäki et al. (2007)**: "Enhanced Pitch-Synchronous Click Removal"
  → Preserve harmonic structure during interpolation
- **Crochiere & Rabiner (1983)**: "Multirate Digital Signal Processing"
  → Multi-scale analysis for click detection

PERFORMANCE TARGET:
------------------
- <1.0× Realtime (professional standard)
- Memory: <80 MB for 10min audio
- Quality Impact: 0.95 (was 0.90 in v1.0)

BENCHMARK COMPARISON:
--------------------
- iZotope RX De-click: Industry standard, ~0.8× realtime
- Audacity Click Removal: Basic, threshold-based
- Aurik v2.0: Professional, multi-scale, <1.0× realtime ✅

Author: Aurik 10.0.0 Development Team
Version: 2.0.0 (Professional Upgrade)
Date: 15. Februar 2026
"""

import logging
import os
import time
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import lfilter

from backend.core.audio_utils import limit_quiet_edge_boost, restore_layout, safe_to_mono, to_channels_last
from backend.core.defect_scanner import MaterialType  # §v10.113
from backend.core.dsp.silence_mask import apply_silence_preservation
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult, create_phase_result

try:
    import librosa as _librosa_lpc

    LIBROSA_AVAILABLE = True
except ImportError:
    _librosa_lpc = None  # type: ignore[assignment]
    LIBROSA_AVAILABLE = False

try:
    from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin

    DEEPFILTERNET_PLUGIN_AVAILABLE = True
except ImportError:
    DeepFilterNetV3IIPlugin = None  # type: ignore[assignment, misc]
    DEEPFILTERNET_PLUGIN_AVAILABLE = False

# ML-Hybrid Support: DFN Plugin (enhance() API, kein subprocess)
# SOUNDFILE_AVAILABLE wird nach Umstieg auf enhance() nicht mehr benötigt.

try:
    from backend.core.quality_mode import QualityMode

    QUALITY_MODE_AVAILABLE = True
except ImportError:
    QUALITY_MODE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClickRemovalPhase(PhaseInterface):
    """
    Professional Click Removal Phase v2.0 with ML-Hybrid Support

    Multi-scale detection with adaptive interpolation for
    professional-grade click and pop removal.

    Features:
    - 3-scale click detection (short/medium/long)
    - Click-type classification (digital/analog/transient)
    - Adaptive interpolation (linear/cubic/spectral)
    - Musical transient preservation
    - Material-adaptive processing
    - ML-Hybrid: DeepFilterNet v3 II for severe clicks (BALANCED/MAXIMUM modes)

    Comparable to: iZotope RX De-click (basic mode)
    """

    # Material-adaptive Sensitivity (Professional-tuned)
    MATERIAL_THRESHOLDS = {
        "shellac": {
            "short": 0.04,  # Very sensitive
            "medium": 0.06,
            "long": 0.08,
            "transient_preserve": 0.7,  # Moderate preservation
        },
        "vinyl": {
            "short": 0.08,  # Moderate
            "medium": 0.12,
            "long": 0.15,
            "transient_preserve": 0.8,  # Good preservation
        },
        "tape": {
            "short": 0.15,  # Gentle
            "medium": 0.20,
            "long": 0.25,
            "transient_preserve": 0.9,  # Strong preservation
        },
        # §V33 MaterialType-Vollständigkeit: Kassette → explizit statt Unknown-Fallback
        # Kassetten-Klicks sind kürzer als Tape-Klicks (Bandrisse, Aussetzer);
        # transient_preserve höher als Tape zum Schutz vokal-adjacenter Transienten
        "cassette": {
            "short": 0.18,
            "medium": 0.22,
            "long": 0.28,
            "transient_preserve": 0.92,
        },
        "unknown": {"short": 0.10, "medium": 0.15, "long": 0.20, "transient_preserve": 0.85},  # Balanced default
    }

    # Click duration thresholds (samples)
    SHORT_CLICK_MAX = 3  # 1-3 samples
    MEDIUM_CLICK_MAX = 10  # 4-10 samples
    LONG_CLICK_MAX = 50  # 11-50 samples

    # ML severity threshold (clicks above this use ML in BALANCED mode)
    ML_SEVERITY_THRESHOLD = 0.6

    def __init__(self):
        """Initialisiert Phase 1 Click Removal."""
        super().__init__()
        self._deepfilternet_plugin = None

    def _get_deepfilternet_plugin(self):
        """
        Lädt DeepFilterNet v3 II Plugin beim ersten Zugriff.

        Returns:
            DeepFilterNet plugin or None if unavailable
        """
        if self._deepfilternet_plugin is not None:
            return self._deepfilternet_plugin

        try:
            if not DEEPFILTERNET_PLUGIN_AVAILABLE or DeepFilterNetV3IIPlugin is None:
                raise ImportError("DeepFilterNet v3 II plugin unavailable")
            self._deepfilternet_plugin = DeepFilterNetV3IIPlugin()
            logger.info("✅ DeepFilterNet v3 II Plugin geladen for Click Removal")
            return self._deepfilternet_plugin
        except Exception as e:
            logger.warning("⚠️  DeepFilterNet Plugin not verfuegbar: %s", e)
            logger.info("    Falling back to DSP-only click removal")
            return None

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_01_click_removal",
            name="Professional Click Removal v2.0",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,  # HIGH - Clicks sind sehr störend
            version="2.0.0",
            dependencies=[],  # First phase
            estimated_time_factor=0.025,  # 2.5% (was 2%)
            memory_requirement_mb=80,  # Increased for multi-scale
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.95,  # Professional (was 0.90)
            description=(
                "Professional multi-scale click removal with adaptive interpolation (comparable to iZotope RX De-click)"
            ),
        )

    @staticmethod
    def _compute_click_removal_profile(
        material: str = "vinyl",
        quality_mode: str | None = "balanced",
        restorability: float = 50.0,
    ) -> dict:
        """Berechnet material- and quality-adaptive profile for click removal (§2.56).

        Returns a dict with keys:
          ml_severity_threshold  [0.35, 0.80]
          cubic_ctx              [6, 20]   — cubic interpolation context (samples)
          spectral_ctx           [64, 256] — spectral repair context (FFT bins)
        """
        # Base thresholds per material (lower = more aggressive)
        _BASE_THRESHOLD: dict[str, float] = {
            "shellac": 0.42,
            "wax_cylinder": 0.40,
            "vinyl": 0.55,
            "tape": 0.62,
            "reel_tape": 0.60,
            "cassette": 0.50,  # §ML-Lowering: cassette clicks benefit from ML even at moderate severity
            "cd_digital": 0.70,
            "mp3_low": 0.65,
        }
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        base_thr = _BASE_THRESHOLD.get(_mk, 0.57)

        # Quality-mode adjustment
        qm = (quality_mode or "balanced").lower()
        qm_delta = {"fast": +0.08, "balanced": 0.0, "quality": -0.06, "maximum": -0.10}.get(qm, 0.0)

        # Restorability adjustment: low restorability → more aggressive (lower threshold)
        rest_delta = (float(restorability) - 50.0) * 0.001  # rest=20 → -0.03; rest=80 → +0.03

        ml_threshold = float(base_thr + qm_delta + rest_delta)
        ml_threshold = max(0.35, min(0.80, ml_threshold))

        # Cubic context: shellac/wax need more context; quality mode → more
        _BASE_CUBIC: dict[str, int] = {
            "shellac": 14,
            "wax_cylinder": 16,
            "vinyl": 12,
            "tape": 10,
            "reel_tape": 10,
            "cassette": 10,
            "cd_digital": 8,
        }
        cubic_ctx = _BASE_CUBIC.get(_mk, 12)
        if qm in ("quality", "maximum"):
            cubic_ctx = min(20, cubic_ctx + 4)
        elif qm == "fast":
            cubic_ctx = max(6, cubic_ctx - 2)

        # Spectral context: shellac/wax → more bins; quality → more
        _BASE_SPEC: dict[str, int] = {
            "shellac": 192,
            "wax_cylinder": 224,
            "vinyl": 160,
            "tape": 128,
            "reel_tape": 128,
            "cassette": 128,
            "cd_digital": 96,
        }
        spectral_ctx = _BASE_SPEC.get(_mk, 160)
        if qm in ("quality", "maximum"):
            spectral_ctx = min(256, spectral_ctx + 64)
        elif qm == "fast":
            spectral_ctx = max(64, spectral_ctx - 64)

        return {
            "ml_severity_threshold": ml_threshold,
            "cubic_ctx": cubic_ctx,
            "spectral_ctx": spectral_ctx,
        }

    @staticmethod
    def _sample_count(input_audio: np.ndarray) -> int:
        """Return sample count for mono, channels-first, or channels-last audio."""
        if input_audio.ndim == 2 and input_audio.shape[0] <= 2 and input_audio.shape[1] > input_audio.shape[0]:
            return int(input_audio.shape[1])
        return int(input_audio.shape[0])

    @staticmethod
    def _resolve_silence_mask(kwargs: dict[str, Any], input_audio: np.ndarray) -> np.ndarray | None:
        """Resolve sample-level silence protection mask from kwargs/restoration context."""
        mask = kwargs.get("silence_mask")
        if mask is None:
            ctx = kwargs.get("restoration_context")
            if isinstance(ctx, dict):
                mask = ctx.get("silence_mask")
        n_samples = ClickRemovalPhase._sample_count(input_audio)
        if isinstance(mask, np.ndarray) and mask.size > 1:
            resolved = np.asarray(mask, dtype=np.float32).ravel()
            if resolved.size < n_samples:
                resolved = np.pad(resolved, (0, n_samples - resolved.size), mode="edge")
            return resolved[:n_samples]  # type: ignore[no-any-return]

        zones = kwargs.get("structural_silence_zones")
        if zones is None:
            ctx = kwargs.get("restoration_context")
            if isinstance(ctx, dict):
                zones = ctx.get("structural_silence_zones")
        if isinstance(zones, list) and zones:
            resolved = np.ones(n_samples, dtype=np.float32)
            for zone in zones:
                try:
                    start, end = int(zone[0]), int(zone[1])
                except Exception:
                    logger.debug("_resolve_silence_mask: silent except suppressed", exc_info=True)
                    continue
                start = max(0, min(n_samples, start))
                end = max(start, min(n_samples, end))
                resolved[start:end] = 0.0
            return resolved  # type: ignore[no-any-return]
        return None

    @staticmethod
    def _click_overlaps_protected_silence(
        start: int,
        end: int,
        silence_mask: np.ndarray | None,
        sample_rate: int,
    ) -> bool:
        """True if a click candidate touches structural silence/fade protection."""
        if silence_mask is None or silence_mask.size <= 1:
            return False
        guard = max(1, int(sample_rate * 0.02))
        left = max(0, int(start) - guard)
        right = min(int(silence_mask.size), int(end) + guard + 1)
        return bool(np.any(silence_mask[left:right] < 0.5))

    @staticmethod
    def _apply_silence_and_edge_guards(
        original: np.ndarray,
        processed: np.ndarray,
        silence_mask: np.ndarray | None,
        sample_rate: int,
        material_type: str,
    ) -> np.ndarray:
        """Restore protected silence and prevent boosted quiet song edges."""
        guarded = np.asarray(processed, dtype=np.float32)
        if silence_mask is not None and silence_mask.size > 1:
            try:
                guarded = apply_silence_preservation(original, guarded, silence_mask)
            except Exception as exc:
                logger.debug("§silence-guarantee Verarbeitungsschritt_01: wiederherstellen uebersprungen: %s", exc)
        try:
            guarded = limit_quiet_edge_boost(
                original,
                guarded,
                sr=sample_rate,
                material_key=str(material_type).lower(),
                max_edge_boost_db=0.5,
            )
        except Exception as exc:
            logger.debug("§0h quiet-edge guard Verarbeitungsschritt_01 uebersprungen: %s", exc)
        return np.clip(np.nan_to_num(guarded, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _emit_progress(progress_callback: Any, pct: float, label: str, elapsed_s: float) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(float(pct), label, float(elapsed_s))
        except TypeError:
            try:
                progress_callback(float(pct), label)
            except Exception:
                logger.debug("_emit_progress: silent except suppressed", exc_info=True)
        except Exception:
            logger.debug("_emit_progress: silent except suppressed", exc_info=True)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        preserve_transients: bool = True,
        quality_mode: str | None = None,
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("DeepFilterNetV3", phase_name="01")
        check_ml_model_ready("PANNs", phase_name="01")
        check_ml_model_ready("Whisper", phase_name="01")
        check_ml_model_ready("DeepFilterNetV3", phase_name="01")
        """
        Professional click removal with multi-scale detection and ML-Hybrid support.

        Args:
            audio: Input audio
            sample_rate: Sample rate (Hz)
            material_type: Material type for adaptive processing
            preserve_transients: Protect musical attacks (drums, etc.)
            quality_mode: Quality mode (FAST/BALANCED/MAXIMUM), None=auto
            **kwargs: Additional parameters

        Returns:
            PhaseResult with click-free audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        # ── §v10 PIM: Per-Band-Intensität kalibrieren ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim = apply_pim_intensity(kwargs, "click_removal", default_nr=0.3, default_de_ess=0.1, default_comp=1.0)
            if kwargs.get("pim_intensity_map") is not None:
                for _key in ("noise_reduction_strength", "nr_strength", "strength", "wet"):
                    if _key in kwargs:
                        kwargs[_key] = _pim["nr_strength"]
        except Exception:
            logger.debug("verarbeiten: silent except suppressed", exc_info=True)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        progress_sub_callback = kwargs.get("progress_sub_callback")
        silence_mask = self._resolve_silence_mask(kwargs, audio)
        self._emit_progress(progress_sub_callback, 3.0, "Knackser-Schutzbereiche werden geprüft", 0.0)

        # Determine if ML should be used
        use_ml = False
        if QUALITY_MODE_AVAILABLE and quality_mode:
            try:
                qm = QualityMode[quality_mode.upper()]
                use_ml = qm.is_ml_enabled  # Phase 1
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        # §0j energy_bias: panns_singing für DFN-Vokal-Schutz
        _panns_singing = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))

        # §V38 Per-Event-Strength-Oracle: VFA-Schutzzonen aufbauen (start_s, end_s, max_severity_cap).
        _p01_protected_zones: list[tuple[float, float, float]] = []
        _vfa_p01 = kwargs.get("vfa_result") or {}
        if isinstance(_vfa_p01, dict):
            for _z in _vfa_p01.get("vibrato_zones", []):
                try:
                    _p01_protected_zones.append((float(_z[0]), float(_z[1]), 0.20))
                except (TypeError, IndexError):
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            for _z in _vfa_p01.get("frisson_zones", []):
                try:
                    _p01_protected_zones.append((float(_z[0]), float(_z[1]), 0.30))
                except (TypeError, IndexError):
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            for _z in _vfa_p01.get("whisper_zones", []):
                try:
                    _p01_protected_zones.append((float(_z[0]), float(_z[1]), 0.25))
                except (TypeError, IndexError):
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            for _z in _vfa_p01.get("passaggio_zones", []):
                try:
                    _p01_protected_zones.append((float(_z[0]), float(_z[1]), 0.35))
                except (TypeError, IndexError):
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        # Get material-specific thresholds
        thresholds = dict(self.MATERIAL_THRESHOLDS.get(material_type, self.MATERIAL_THRESHOLDS["unknown"]))

        # Locality-aware intensity control from UV3.
        # Sparse event defects should be repaired more locally/gently to avoid
        # global timbre changes outside defect regions.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        if phase_locality_factor < 0.999:
            # Higher thresholds => fewer candidates when locality is sparse.
            inv = 1.0 / max(phase_locality_factor, 1e-6)
            thresholds["short"] = float(np.clip(thresholds["short"] * inv, 0.005, 2.0))
            thresholds["medium"] = float(np.clip(thresholds["medium"] * inv, 0.005, 2.0))
            thresholds["long"] = float(np.clip(thresholds["long"] * inv, 0.005, 2.0))
            # Preserve more transients in sparse mode.
            thresholds["transient_preserve"] = float(
                np.clip(thresholds["transient_preserve"] + 0.10 * (1.0 - phase_locality_factor), 0.0, 0.99)
            )

        # §V38 Strength-Orakel: muss VOR den Stereo/Mono-Pfaden berechnet werden,
        # damit _compute_click_local_strength() mit korrektem base_strength aufgerufen wird.
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §2.51 Linked-Stereo: Click-Detektion auf Mono-Mix, Repair synchron auf L+R
        # aber OHNE globalen Gain-Transfer. Stattdessen werden nur die erkannten
        # Ereignisfenster kanalweise gepatcht. Saubere Gegenkanäle bleiben unverändert.
        is_stereo = audio.ndim == 2
        if is_stereo:
            stereo_audio, was_transposed = to_channels_last(audio)
            mono_mix = safe_to_mono(stereo_audio)
            result_audio, stats_mono = self._remove_clicks_linked_stereo(
                stereo_audio,
                mono_mix,
                sample_rate,
                thresholds,
                preserve_transients,
                use_ml,
                progress_callback=progress_sub_callback,
                silence_mask=silence_mask,
                start_time=start_time,
                panns_singing=_panns_singing,
                protected_zones=_p01_protected_zones or None,
                base_strength=_effective_strength,
            )
            result_audio = restore_layout(result_audio, was_transposed)
            total_clicks = stats_mono["total"]
            ml_repaired_count = stats_mono.get("ml_repaired", 0)
            click_types = {
                "short": stats_mono["short"],
                "medium": stats_mono["medium"],
                "long": stats_mono["long"],
                "transients_preserved": stats_mono["transients_preserved"],
                "ml_repaired": ml_repaired_count,
            }
        else:
            result_audio, stats = self._remove_clicks_professional(
                audio,
                sample_rate,
                thresholds,
                preserve_transients,
                use_ml,
                progress_callback=progress_sub_callback,
                silence_mask=silence_mask,
                start_time=start_time,
                panns_singing=_panns_singing,
                protected_zones=_p01_protected_zones or None,
                base_strength=_effective_strength,
            )
            total_clicks = stats["total"]
            ml_repaired_count = stats.get("ml_repaired", 0)
            click_types = {
                "short": stats["short"],
                "medium": stats["medium"],
                "long": stats["long"],
                "transients_preserved": stats["transients_preserved"],
                "ml_repaired": ml_repaired_count,
            }

        execution_time = time.time() - start_time

        # Generate warnings
        warnings = []
        if total_clicks > 1000:
            warnings.append(f"High click count: {total_clicks} (severe degradation)")
        if click_types["long"] > 100:
            warnings.append(f"Many long clicks ({click_types['long']}): possible scratches")

        # Calculate preservation ratio
        preservation_ratio = 0.0
        if total_clicks > 0:
            preservation_ratio = click_types["transients_preserved"] / (
                total_clicks + click_types["transients_preserved"]
            )

        # Calculate ML usage ratio
        ml_ratio = 0.0
        if total_clicks > 0 and ml_repaired_count > 0:
            ml_ratio = ml_repaired_count / total_clicks

        result_audio = np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0)

        result_audio = np.clip(result_audio, -1.0, 1.0)

        # Strength-aware Wet/Dry-Blend (PMGG-Retry-Kompatibilität):
        # PMGG übergibt strength < 1.0 bei Retries.  Blend VOR Return,
        # damit reduzierte Strength die Verarbeitungsintensität senkt.
        # _pmgg_strength / _effective_strength bereits oben berechnet (§V38 — vor Stereo/Mono-Pfaden).
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return create_phase_result(
                audio=passthrough,
                modifications={
                    "total_clicks_removed": 0,
                    "reason": "zero effective strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                },
                resolved_defects={},
                warnings=["Click removal skipped due to zero effective strength"],
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "execution_time_seconds": time.time() - start_time,
                },
            )
        if 0.0 < _effective_strength < 1.0:
            result_audio = (audio + _effective_strength * (result_audio - audio)).astype(audio.dtype)
            result_audio = np.clip(result_audio, -1.0, 1.0)

        result_audio = self._apply_silence_and_edge_guards(
            audio,
            result_audio,
            silence_mask,
            sample_rate,
            material_type,
        )
        self._emit_progress(progress_sub_callback, 98.0, "Knackser-Reparatur gesichert", time.time() - start_time)

        return create_phase_result(
            audio=result_audio,
            modifications={
                "total_clicks_removed": total_clicks,
                "short_clicks": click_types["short"],
                "medium_clicks": click_types["medium"],
                "long_clicks": click_types["long"],
                "transients_preserved": click_types["transients_preserved"],
                "ml_repaired": ml_repaired_count,
                "ml_usage_ratio": ml_ratio,
                "preservation_ratio": preservation_ratio,
                "material_type": material_type,
                "algorithm_version": "2.0_ml_hybrid" if use_ml else "2.0_professional",
            },
            warnings=warnings,
            metadata={
                "algorithm": "multi_scale_adaptive_interpolation",
                "ml_model": "DeepFilterNet v3 II" if use_ml else None,
                "interpolation_methods": (
                    ["linear", "cubic", "spectral", "ml_deepfilternet"] if use_ml else ["linear", "cubic", "spectral"]
                ),
                "scientific_ref": "Godsill & Rayner (1998), Välimäki et al. (2007)",
                "benchmark": "iZotope RX De-click (basic)",
                "execution_time_seconds": execution_time,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
                "stereo_strategy": "linked_sparse_patch_transfer" if is_stereo else "mono_direct_repair",
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
            resolved_defects={
                "CLICKS": float(np.clip(1.0 - min(total_clicks / max(audio.shape[-1] * 0.001, 1), 1.0), 0.0, 0.3)),
            },
        )

    def _build_click_repair_plan(
        self,
        audio: np.ndarray,
        sample_rate: int,
        thresholds: dict[str, float],
        preserve_transients: bool,
        use_ml: bool,
        silence_mask: np.ndarray | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        """Erstellt einen deterministischen Reparaturplan für Click-Events."""
        stats = {"short": 0, "medium": 0, "long": 0, "transients_preserved": 0, "ml_repaired": 0, "total": 0}
        click_candidates = self._detect_clicks_multiscale(audio, thresholds)
        classified_clicks = self._classify_clicks(audio, click_candidates, preserve_transients, thresholds)

        severe_clicks: list[dict[str, Any]] = []
        normal_clicks: list[dict[str, Any]] = []
        for click in classified_clicks:
            if self._click_overlaps_protected_silence(click["start"], click["end"], silence_mask, sample_rate):
                stats["transients_preserved"] += 1
                continue
            if click["type"] == "transient":
                stats["transients_preserved"] += 1
                continue

            severity = float(click.get("severity", 0.5))
            if protected_zones:
                _ck_s = click["start"] / sample_rate
                _ck_e = (click["end"] + 1) / sample_rate
                for _pz_s, _pz_e, _pz_cap in protected_zones:
                    if _ck_s < _pz_e and _ck_e > _pz_s:
                        severity = min(severity, _pz_cap)
                        break

            click_plan = dict(click)
            click_plan["severity"] = severity
            if use_ml and severity > self.ML_SEVERITY_THRESHOLD:
                severe_clicks.append(click_plan)
            else:
                normal_clicks.append(click_plan)

        return severe_clicks, normal_clicks, stats

    def _channel_click_requires_repair(
        self,
        channel_audio: np.ndarray,
        click: dict[str, Any],
        thresholds: dict[str, float],
    ) -> bool:
        """Prüft, ob der konkrete Kanal im gekoppelten Stereo-Fenster wirklich einen Click trägt."""
        start = int(click["start"])
        end = int(click["end"])
        duration = end - start + 1
        if duration <= self.SHORT_CLICK_MAX:
            base_threshold = float(thresholds["short"])
        elif duration <= self.MEDIUM_CLICK_MAX:
            base_threshold = float(thresholds["medium"])
        else:
            base_threshold = float(thresholds["long"])

        local_start = max(1, start - 4)
        local_end = min(len(channel_audio) - 1, end + 5)
        if local_end <= local_start:
            return False

        local_diff = np.abs(np.diff(channel_audio[local_start - 1 : local_end + 1]))
        local_peak = float(np.max(local_diff)) if local_diff.size else 0.0

        ctx_radius = 48
        before = channel_audio[max(0, start - ctx_radius) : start]
        after = channel_audio[end + 1 : min(len(channel_audio), end + 1 + ctx_radius)]
        context = (
            np.concatenate([before, after]) if len(before) or len(after) else np.array([], dtype=channel_audio.dtype)
        )
        if context.size >= 4:
            context_diff = np.abs(np.diff(context))
            context_median = float(np.median(context_diff)) if context_diff.size else 0.0
            context_mad = float(np.median(np.abs(context_diff - context_median))) if context_diff.size else 0.0
            adaptive_gate = context_median + 3.5 * max(1.4826 * context_mad, 1e-8)
        else:
            adaptive_gate = base_threshold

        absolute_region_peak = float(np.max(np.abs(channel_audio[start : end + 1]))) if end >= start else 0.0
        context_peak = float(np.percentile(np.abs(context), 95)) if context.size else 0.0
        return bool(
            local_peak > max(base_threshold * 0.75, adaptive_gate)
            or absolute_region_peak > context_peak + base_threshold * 0.5
        )

    def _repair_click_patch_ml(
        self,
        audio: np.ndarray,
        sample_rate: int,
        click: dict[str, Any],
        *,
        panns_singing: float = 0.0,
    ) -> bool:
        """Repariert einen einzelnen starken Click als lokales ML-Patch statt Vollsignal-Transfer."""
        plugin = self._get_deepfilternet_plugin()
        if plugin is None:
            return False

        start = int(click["start"])
        end = int(click["end"])
        duration = max(1, end - start + 1)
        context = int(np.clip(duration * 32, 256, 2048))
        patch_start = max(0, start - context)
        patch_end = min(len(audio), end + context + 1)
        if patch_end - patch_start < duration + 8:
            return False

        patch = audio[patch_start:patch_end].copy()
        energy_bias = -6.0 if float(panns_singing) >= 0.4 else -9.0

        _plm01_dfn = None
        try:
            _plm01_dfn = get_plugin_lifecycle_manager()
            _plm01_dfn.set_active("DeepFilterNetV3", True)
        except Exception:
            _plm01_dfn = None

        try:
            repaired_patch = np.asarray(
                plugin.enhance(patch, sr=sample_rate, energy_bias_db=energy_bias), dtype=np.float32
            )
            if len(repaired_patch) != len(patch):
                return False
            replace_start = max(patch_start, start - min(8, duration))
            replace_end = min(patch_end, end + min(8, duration) + 1)
            local_start = replace_start - patch_start
            local_end = replace_end - patch_start
            replacement = repaired_patch[local_start:local_end].copy()
            fade = min(8, len(replacement) // 4)
            if fade >= 2:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                original = audio[replace_start:replace_end].copy()
                replacement[:fade] = (1.0 - ramp) * original[:fade] + ramp * replacement[:fade]
                replacement[-fade:] = ramp[::-1] * original[-fade:] + (1.0 - ramp[::-1]) * replacement[-fade:]
            audio[replace_start:replace_end] = np.clip(replacement, -1.0, 1.0)
            return True
        except Exception as e:
            logger.debug("Lokales ML-Click-Patch fehlgeschlagen: %s", e)
            return False
        finally:
            if _plm01_dfn is not None:
                try:
                    _plm01_dfn.set_active("DeepFilterNetV3", False)
                except Exception:
                    logger.debug("_repair_click_patch_ml: silent except suppressed", exc_info=True)

    @staticmethod
    def _compute_click_local_strength(
        mono_ref: np.ndarray,
        start: int,
        end: int,
        sr: int,
        base_strength: float,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> float:
        """§V38 Per-Event-Strength-Oracle für Click-Reparatur (v10.0.0).

        Berechnet eine event-lokale Reparaturstärke basierend auf:
        1) 250 ms Kontext-RMS-Proxy: stille Zonen (Flüster, Pausen) → weniger Blend
        2) VFA-Schutzzonen-Cap: Vibrato 0.20, Frisson 0.30, Flüster 0.25, Passaggio 0.35

        Das Oracle verhindert, dass leichte Events überprozessiert werden, und schützt
        expressionstransiente Passagen (§0p Vocal-Supremacy-Doktrin).

        Args:
            mono_ref:       Mono-Audio-Referenz (zum Kontext-RMS-Computing)
            start, end:     Click-Grenzen in Samples
            sr:             Sample-Rate
            base_strength:  Globale Reparaturstärke aus PMGG/Locality-Faktor
            protected_zones: [(start_s, end_s, max_cap), ...] aus VFA

        Returns:
            Lokale Stärke in [0, base_strength] — niemals höher als base_strength.
        """
        if base_strength <= 0.0:
            return 0.0

        n = len(mono_ref)
        # click_mid: Zeitreferenzpunkt für künftige Zeitstempel-Checks (noch nicht genutzt).
        # VFA-Cap basiert auf Sample-Indizes, nicht auf Zeitstempel.

        # 1) VFA-Schutzzonen-Cap (harte Obergrenze)
        zone_cap = base_strength  # default: kein Cap
        if protected_zones:
            for pz_start, pz_end, pz_cap in protected_zones:
                click_start_s = start / max(sr, 1)
                click_end_s = (end + 1) / max(sr, 1)
                if click_start_s < pz_end and click_end_s > pz_start:
                    zone_cap = min(zone_cap, pz_cap)
                    break  # strengster Cap zuerst (single zone check)

        # 2) 250 ms Kontext-RMS-Proxy: Amplitude um den Click herum
        ctx_half = max(1, int(0.125 * sr))  # 125 ms beiderseits = 250 ms Fenster
        ctx_start = max(0, start - ctx_half)
        ctx_end = min(n, end + ctx_half + 1)
        # Kontext OHNE den Click selbst (um Click-Amplitude nicht zu messen)
        ctx_before = mono_ref[ctx_start:start] if start > ctx_start else np.array([], dtype=np.float32)
        ctx_after = mono_ref[end + 1 : ctx_end] if end + 1 < ctx_end else np.array([], dtype=np.float32)
        ctx_arr = np.concatenate([ctx_before, ctx_after])
        if len(ctx_arr) >= 16:
            ctx_rms = float(np.sqrt(np.mean(ctx_arr.astype(np.float64) ** 2) + 1e-12))
        else:
            ctx_rms = 0.1  # Standardwert wenn kein Kontext

        # Stille/Flüster → vorsichtiger: RMS < 0.01 → lokale Stärke × 0.5
        # Normale Musik  → volle Stärke: RMS ≥ 0.05
        # Klang-adaptiver Blend zwischen diesen Grenzen (linear)
        if ctx_rms >= 0.05:
            amplitude_scale = 1.0
        elif ctx_rms <= 0.01:
            amplitude_scale = 0.5
        else:
            amplitude_scale = 0.5 + 0.5 * (ctx_rms - 0.01) / (0.05 - 0.01)

        local_strength = float(np.clip(base_strength * amplitude_scale, 0.0, zone_cap))
        return local_strength

    def _apply_click_plan_to_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        severe_clicks: list[dict[str, Any]],
        normal_clicks: list[dict[str, Any]],
        thresholds: dict[str, float],
        use_ml: bool,
        *,
        panns_singing: float = 0.0,
        base_strength: float = 1.0,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Wendet einen gekoppelten Reparaturplan kanalweise und ereignislokal an.

        §V38 v10.0.0: Per-Event-Strength-Oracle — jedes Click-Event erhält eine
        lokale Reparaturstärke via `_compute_click_local_strength()`.
        Schützt Vibrato/Frisson/Flüster/Passaggio-Zonen automatisch.
        """
        repaired = audio.copy()
        ml_repaired = 0
        mono_ref = audio  # Mono-Referenz für Kontext-RMS

        for click in severe_clicks:
            if not self._channel_click_requires_repair(repaired, click, thresholds):
                continue
            start_idx = int(click["start"])
            end_idx = int(click["end"])
            # §V38 per-event local strength
            local_s = self._compute_click_local_strength(
                mono_ref, start_idx, end_idx, sample_rate, base_strength, protected_zones
            )
            if local_s <= 0.0:
                continue
            if use_ml and self._repair_click_patch_ml(repaired, sample_rate, click, panns_singing=panns_singing):
                ml_repaired += 1
                if local_s < 1.0:
                    # Wet/Dry-Blend für ML-Reparatur im Event-Fenster
                    ctx = slice(max(0, start_idx), min(len(repaired), end_idx + 1))
                    repaired[ctx] = audio[ctx] + local_s * (repaired[ctx] - audio[ctx])
                continue
            repaired_copy = repaired.copy()
            repaired_copy = self._apply_click_repair_to_channel(repaired_copy, click)
            if local_s < 1.0:
                ctx = slice(max(0, start_idx), min(len(repaired), end_idx + 1))
                repaired[ctx] = repaired[ctx] + local_s * (repaired_copy[ctx] - repaired[ctx])
            else:
                repaired = repaired_copy

        for click in normal_clicks:
            if not self._channel_click_requires_repair(repaired, click, thresholds):
                continue
            start_idx = int(click["start"])
            end_idx = int(click["end"])
            local_s = self._compute_click_local_strength(
                mono_ref, start_idx, end_idx, sample_rate, base_strength, protected_zones
            )
            if local_s <= 0.0:
                continue
            repaired_copy = repaired.copy()
            repaired_copy = self._apply_click_repair_to_channel(repaired_copy, click)
            if local_s < 1.0:
                ctx = slice(max(0, start_idx), min(len(repaired), end_idx + 1))
                repaired[ctx] = repaired[ctx] + local_s * (repaired_copy[ctx] - repaired[ctx])
            else:
                repaired = repaired_copy

        return repaired, ml_repaired

    def _apply_click_repair_to_channel(self, audio: np.ndarray, click: dict[str, Any]) -> np.ndarray:
        """Wendet die passende lokale DSP-Reparatur für ein einzelnes Click-Event an."""
        start_idx = int(click["start"])
        end_idx = int(click["end"])
        duration = end_idx - start_idx + 1
        if duration <= self.SHORT_CLICK_MAX:
            return self._interpolate_linear(audio, start_idx, end_idx)
        if duration <= self.MEDIUM_CLICK_MAX:
            _mask_rbme = np.zeros(len(audio), dtype=bool)
            _mask_rbme[start_idx : end_idx + 1] = True
            _rbme_out = self._rbme_interpolate(audio, _mask_rbme)
            _gap_changed = not np.allclose(
                _rbme_out[start_idx : end_idx + 1],
                audio[start_idx : end_idx + 1],
                atol=1e-7,
            )
            if _gap_changed:
                return _rbme_out
            return self._interpolate_cubic(audio, start_idx, end_idx)
        return self._interpolate_spectral(audio, start_idx, end_idx)

    def _remove_clicks_linked_stereo(
        self,
        stereo_audio: np.ndarray,
        mono_mix: np.ndarray,
        sample_rate: int,
        thresholds: dict[str, float],
        preserve_transients: bool,
        use_ml: bool,
        panns_singing: float = 0.0,
        progress_callback: Any = None,
        silence_mask: np.ndarray | None = None,
        start_time: float | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
        base_strength: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Linked-Stereo-Reparatur mit mono-gekoppelter Detektion und kanal-lokalen Patches."""
        severe_clicks, normal_clicks, stats = self._build_click_repair_plan(
            mono_mix,
            sample_rate,
            thresholds,
            preserve_transients,
            use_ml,
            silence_mask=silence_mask,
            protected_zones=protected_zones,
        )
        self._emit_progress(
            progress_callback,
            18.0,
            "Knackser werden lokalisiert",
            time.time() - (start_time or time.time()),
        )
        self._emit_progress(
            progress_callback,
            32.0,
            "Knackser werden klassifiziert",
            time.time() - (start_time or time.time()),
        )

        repaired = stereo_audio.copy()
        total_channels = max(1, repaired.shape[1])
        ml_used_any = False
        for channel_idx in range(total_channels):
            if channel_idx == 0:
                self._emit_progress(
                    progress_callback,
                    45.0,
                    "Knackser werden kanal-lokal repariert",
                    time.time() - (start_time or time.time()),
                )
            # §V38: per-event local strength via mono_mix als Kontext-Referenz
            channel_repaired, channel_ml = self._apply_click_plan_to_channel(
                repaired[:, channel_idx],
                sample_rate,
                severe_clicks,
                normal_clicks,
                thresholds,
                use_ml,
                panns_singing=panns_singing,
                base_strength=base_strength,
                protected_zones=protected_zones,
            )
            repaired[:, channel_idx] = channel_repaired
            ml_used_any = ml_used_any or channel_ml > 0

        for click in severe_clicks + normal_clicks:
            duration = int(click["end"]) - int(click["start"]) + 1
            if duration <= self.SHORT_CLICK_MAX:
                stats["short"] += 1
            elif duration <= self.MEDIUM_CLICK_MAX:
                stats["medium"] += 1
            else:
                stats["long"] += 1
            stats["total"] += 1
        if ml_used_any:
            stats["ml_repaired"] = len(severe_clicks)
        return repaired, stats

    def _remove_clicks_professional(
        self,
        audio: np.ndarray,
        sample_rate: int,
        thresholds: dict[str, float],
        preserve_transients: bool,
        use_ml: bool,
        panns_singing: float = 0.0,
        progress_callback: Any = None,
        silence_mask: np.ndarray | None = None,
        start_time: float | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
        base_strength: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Professional click removal with multi-scale detection and ML-Hybrid support.

        §V38 v10.0.0: Per-Event-Strength-Oracle — lokale Reparaturstärke
        via `_compute_click_local_strength()` für jedes Click-Event.

        Returns:
            (cleaned_audio, statistics_dict)
        """
        audio_cleaned = audio.copy()

        severe_clicks, normal_clicks, stats = self._build_click_repair_plan(
            audio,
            sample_rate,
            thresholds,
            preserve_transients,
            use_ml,
            silence_mask=silence_mask,
            protected_zones=protected_zones,
        )
        self._emit_progress(
            progress_callback,
            18.0,
            "Knackser werden lokalisiert",
            time.time() - (start_time or time.time()),
        )
        self._emit_progress(
            progress_callback,
            32.0,
            "Knackser werden klassifiziert",
            time.time() - (start_time or time.time()),
        )

        # Step 4: Process severe clicks with ML (if available and enabled)
        if severe_clicks and use_ml:
            self._emit_progress(
                progress_callback,
                45.0,
                "Starke Knackser werden repariert",
                time.time() - (start_time or time.time()),
            )
            ml_success = self._repair_clicks_ml(audio_cleaned, sample_rate, severe_clicks, panns_singing=panns_singing)
            if ml_success:
                stats["ml_repaired"] = len(severe_clicks)
                # Count by duration for stats
                for click in severe_clicks:
                    duration = click["end"] - click["start"] + 1
                    if duration <= self.SHORT_CLICK_MAX:
                        stats["short"] += 1
                    elif duration <= self.MEDIUM_CLICK_MAX:
                        stats["medium"] += 1
                    else:
                        stats["long"] += 1
                    stats["total"] += 1
            else:
                # ML failed, add back to normal clicks for DSP fallback
                logger.warning("ML click repair fehlgeschlagen, falling back to DSP")
                normal_clicks.extend(severe_clicks)
        else:
            # No ML available/enabled, process all with DSP
            normal_clicks.extend(severe_clicks)

        # Step 5: Process normal clicks with DSP interpolation
        total_normal = max(1, len(normal_clicks))
        for idx, click in enumerate(normal_clicks):
            if click["type"] == "transient":
                stats["transients_preserved"] += 1
                continue  # Skip musical transients
            if self._click_overlaps_protected_silence(click["start"], click["end"], silence_mask, sample_rate):
                stats["transients_preserved"] += 1
                continue

            if idx == 0 or idx == total_normal - 1 or idx % max(1, total_normal // 20) == 0:
                self._emit_progress(
                    progress_callback,
                    45.0 + 45.0 * (idx / total_normal),
                    "Knackser werden repariert",
                    time.time() - (start_time or time.time()),
                )

            start_idx = click["start"]
            end_idx = click["end"]
            duration = end_idx - start_idx + 1
            # §V38 per-event local strength
            local_s = self._compute_click_local_strength(
                audio_cleaned, start_idx, end_idx, sample_rate, base_strength, protected_zones
            )
            if local_s <= 0.0:
                stats["total"] += 1
                continue
            pre_event = audio_cleaned.copy()
            audio_cleaned = self._apply_click_repair_to_channel(audio_cleaned, click)
            if local_s < 1.0:
                ctx = slice(max(0, start_idx), min(len(audio_cleaned), end_idx + 1))
                audio_cleaned[ctx] = pre_event[ctx] + local_s * (audio_cleaned[ctx] - pre_event[ctx])
            if duration <= self.SHORT_CLICK_MAX:
                stats["short"] += 1
            elif duration <= self.MEDIUM_CLICK_MAX:
                stats["medium"] += 1
            else:
                stats["long"] += 1

            stats["total"] += 1

        return audio_cleaned, stats

    def _repair_clicks_ml(
        self,
        audio: np.ndarray,
        sample_rate: int,
        clicks: list[dict[str, Any]],
        *,
        panns_singing: float = 0.0,
    ) -> bool:
        """Repariert severe clicks via DeepFilterNet v3 II enhance() (in-memory, §0j energy_bias).

        Args:
            audio: Audio-Array (mono, wird in-place modifiziert)
            sample_rate: Sample-Rate
            clicks: Liste der Click-Dicts mit 'start', 'end', 'severity'
            panns_singing: PANNs-Singing-Score für energy_bias-Wahl (§0j)

        Returns:
            True wenn erfolgreich, False sonst
        """
        plugin = self._get_deepfilternet_plugin()
        if plugin is None:
            return False

        # §0j energy_bias: -6 dB für Vokal (panns_singing >= 0.4), -9 dB für Instrumental
        _dfn_energy_bias = -6.0 if float(panns_singing) >= 0.4 else -9.0

        # §4.6b: PLM active-guard — prevents emergency-eviction during DeepFilterNet inference
        _plm01_dfn = None
        try:
            _plm01_dfn = get_plugin_lifecycle_manager()
            _plm01_dfn.set_active("DeepFilterNetV3", True)
        except Exception:
            logger.debug("_repair_clicks_ml: silent except suppressed", exc_info=True)

        try:
            # In-memory enhance() statt Subprocess-Datei-API (§V05: kein griffinlim, kein process())
            repaired = plugin.enhance(audio, sr=sample_rate, energy_bias_db=_dfn_energy_bias)
            repaired = np.asarray(repaired, dtype=np.float32)
            n = min(len(repaired), len(audio))
            if n == len(audio):
                audio[:] = repaired[:n]
                logger.info(
                    "ML click repair erfolgreich (%s Knackser, energy_bias=%.1f dB)",
                    len(clicks),
                    _dfn_energy_bias,
                )
                return True
            else:
                logger.warning("Längen-Mismatch: %s vs %s", len(repaired), len(audio))
                return False

        except Exception as e:
            logger.error("ML click repair Fehler: %s", e)
            return False

        finally:
            # §4.6b: PLM active-guard freigeben
            if _plm01_dfn is not None:
                try:
                    _plm01_dfn.set_active("DeepFilterNetV3", False)
                except Exception:
                    logger.debug("_repair_clicks_ml: silent except suppressed", exc_info=True)

    def _detect_clicks_multiscale(self, audio: np.ndarray, thresholds: dict[str, float]) -> list[tuple[int, int]]:
        """
        Multi-scale click detection using MAD-based adaptive thresholds.

        SOTA upgrade (v2.1): Replaces fixed ``median_diff * 10`` multiplier
        with per-sample adaptive thresholds derived from the Median Absolute
        Deviation (MAD).  MAD is a robust dispersion estimator that remains
        accurate even when > 40 % of data are outliers (clicks) — unlike
        standard deviation, which is inflated by the very events we want to
        detect.

        Algorithm:
            1. Compute |Δx| = |x[n] − x[n−1]| (first-order difference)
            2. Sliding-window median of |Δx| over W = 4801 samples (~100 ms @ 48 kHz)
            3. MAD = 1.4826 × median(||Δx| − median(|Δx|)||)  per window
               (1.4826 = consistency factor for Gaussian equivalence; Hampel 1974)
            4. Adaptive threshold = local_median + k × MAD
               k = 4.0 (≈ 99.994 % of Gaussian, catches 3-sigma clicks)
            5. Material sensitivity further scales k: shellac → k=3.5, tape → k=5.0

        Advantages over fixed-multiplier approach:
            - Catches clicks in high-noise regions (tape hiss, vinyl surface noise)
              where global median is elevated and fixed multiplier misses them
            - Avoids false positives in quiet passages where fixed multiplier
              triggers on normal musical transients
            - Scientific: Picard (1992), Huber (1981) "Robust Statistics"

        Returns:
            List of (start_idx, end_idx) tuples
        """
        diff = np.abs(np.diff(audio))

        # Sliding-window size: ~100 ms @ 48 kHz (must be odd for median_filter)
        _W = 4801

        # Robust local statistics via MAD (Median Absolute Deviation)
        local_median = median_filter(diff, size=min(_W, len(diff) | 1), mode="reflect")
        local_deviation = np.abs(diff - local_median)
        local_mad = 1.4826 * median_filter(local_deviation, size=min(_W, len(diff) | 1), mode="reflect")

        # Material-adaptive multiplier k (base from threshold config)
        # Lower threshold → more sensitive → lower k
        base_thresh = thresholds["short"]
        if base_thresh <= 0.06:  # shellac: very sensitive
            k = 3.5
        elif base_thresh <= 0.12:  # vinyl: moderate
            k = 4.0
        elif base_thresh <= 0.20:  # tape: gentle
            k = 5.0
        else:  # digital: conservative
            k = 6.0

        # Per-sample adaptive threshold
        adaptive_threshold = local_median + k * np.maximum(local_mad, 1e-8)

        # Also enforce a minimum floor from the material threshold
        # to prevent detecting micro-noise as clicks
        global_floor = thresholds["short"] * 0.5
        adaptive_threshold = np.maximum(adaptive_threshold, global_floor)

        # Detect clicks: diff exceeds local adaptive threshold
        click_mask = diff > adaptive_threshold

        # Group consecutive samples into click regions
        click_regions: list[tuple[int, int]] = []
        in_click = False
        start_idx = 0

        for i, is_click in enumerate(click_mask):
            if is_click and not in_click:
                start_idx = i
                in_click = True
            elif not is_click and in_click:
                click_regions.append((start_idx, i - 1))
                in_click = False

        if in_click:
            click_regions.append((start_idx, len(click_mask) - 1))

        return click_regions

    def _classify_clicks(
        self,
        audio: np.ndarray,
        click_candidates: list[tuple[int, int]],
        preserve_transients: bool,
        thresholds: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Classify clicks as: digital, analog, or musical transient.
        Also calculates severity score (0-1) for ML routing.

        Returns:
            List of click dictionaries with 'type', 'start', 'end', 'severity'
        """
        classified = []
        transient_threshold = thresholds["transient_preserve"]

        for start, end in click_candidates:
            duration = end - start + 1

            # Feature extraction
            click_region = audio[start : end + 1]
            click_energy: float = float(np.sum(click_region**2))
            click_amplitude: float = float(np.max(np.abs(click_region)))

            # Calculate severity (0-1):
            # - Amplitude contribution: 50%
            # - Duration contribution: 50%
            amplitude_severity = min(1.0, click_amplitude / 0.8)  # Normalize to [0,1], 0.8=severe
            duration_severity = min(1.0, duration / self.LONG_CLICK_MAX)  # Normalize by max duration
            severity = 0.5 * amplitude_severity + 0.5 * duration_severity

            # Check if this is a musical transient (legitimate attack)
            if preserve_transients and duration < 20:
                # Analyze surrounding context
                before = audio[max(0, start - 100) : start]
                after = audio[end + 1 : min(len(audio), end + 101)]

                # Musical transients have coherent energy distribution
                if len(before) > 10 and len(after) > 10:
                    before_energy = np.mean(before**2)
                    after_energy = np.mean(after**2)

                    # High energy before/after suggests musical content
                    if before_energy > 0.001 and after_energy > 0.001:
                        energy_ratio = click_energy / (before_energy + after_energy + 1e-10)

                        # If energy is proportional (not spike), it's transient
                        if energy_ratio < transient_threshold * 100:
                            classified.append(
                                {
                                    "type": "transient",
                                    "start": start,
                                    "end": end,
                                    "severity": 0.0,  # Transients are not defects
                                }
                            )
                            continue

            # Classify as digital or analog click
            # Digital clicks: Sharp edges, abrupt changes
            # Analog clicks: Softer, more gradual
            click_type = "digital" if duration <= 5 and click_amplitude > 0.7 else "analog"

            classified.append({"type": click_type, "start": start, "end": end, "severity": severity})

        return classified

    def _interpolate_linear(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Linear interpolation for short clicks (1-3 samples)."""
        if start == 0 or end >= len(audio) - 1:
            return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)  # Can't interpolate at edges

        # Interpolate between neighbors
        left_val = audio[start - 1]
        right_val = audio[end + 1]

        # Linear spacing
        num_samples = end - start + 1
        interpolated = np.linspace(left_val, right_val, num_samples + 2)[1:-1]

        audio[start : end + 1] = interpolated
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    def _interpolate_cubic(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Cubic spline interpolation for medium clicks (4-10 samples)."""
        # Need at least 4 points for cubic spline
        ctx_size = 10
        ctx_start = max(0, start - ctx_size)
        ctx_end = min(len(audio), end + ctx_size + 1)

        if ctx_end - ctx_start < 10:
            return self._interpolate_linear(audio, start, end)

        # Extract context (samples before and after click)
        context_x = []
        context_y = []

        for i in range(ctx_start, start):
            context_x.append(i)
            context_y.append(audio[i])

        for i in range(end + 1, ctx_end):
            context_x.append(i)
            context_y.append(audio[i])

        if len(context_x) < 4:
            return self._interpolate_linear(audio, start, end)

        # Cubic spline interpolation
        cs = CubicSpline(context_x, context_y)

        # Generate interpolated values
        interpolated_x = np.arange(start, end + 1)
        interpolated_y = cs(interpolated_x)

        # Clip to audio range
        interpolated_y = np.clip(interpolated_y, -1.0, 1.0)

        audio[start : end + 1] = interpolated_y
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    def _interpolate_spectral(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Spektrale Interpolation für lange Clicks (11–50 Samples).

        Algorithmus (High-Order LPC + Hann-gewichtete Spektral-Blend):
            1. Kontext-Fenster: 128 Samples vor und nach der Lücke
            2. Vorwärtsvorhersage: High-Order LPC via Levinson-Durbin (order ≥ 20)
               scipy.signal.lpc (Levinson-Durbin) mit order = min(48, ctx//3)
            3. Rückwärtsvorhersage: Gleiche Methode auf umgekehrtem After-Segment
            4. Spektraler Energieausgleich: DFT-Magnitude der Vorhersagen angleichen
               → vermeidet Amplitudensprünge an den Kanten
            5. Cosinus-Blending (Hann-Gewichte) statt linear — weicherer Übergang
            6. Kantenglätte: 8-Sample Crossfade mit Originalrand
            7. clip[-1, 1] + nan_to_num

        Referenz:
            Levinson (1947) / Durbin (1960) Rekurrenz — über scipy.signal.lpc
            Lagrange & Marchand (2007): Long Interpolation Using AR Sinusoidal Modeling
              (Inspirationsquelle für beiderseitige Kontextnutzung)

        Args:
            audio: 1D-Audio-Array (wird in-place modifiziert).
            start: Beginn-Index der Lücke (inklusiv).
            end:   End-Index der Lücke (inklusiv).

        Returns:
            np.ndarray: Audio mit interpolierter Lücke.
        """
        # High-Order AR via librosa.lpc (Levinson-Durbin, Ordnung ≥ 20)
        # Pflicht: scipy.signal hat kein lpc ab 1.15 — librosa.lpc ist die
        # normkonforme Lösung (Aurik-Standard: AR-Ordnung ≥ 20).
        if not LIBROSA_AVAILABLE or _librosa_lpc is None:
            raise ImportError("librosa is required for spectral click interpolation")

        ctx_size = 128
        ctx_start = max(0, start - ctx_size)
        ctx_end = min(len(audio), end + ctx_size + 1)
        click_len = end - start + 1

        before = audio[ctx_start:start].astype(np.float64)
        after = audio[end + 1 : ctx_end].astype(np.float64)

        if len(before) < 24 or len(after) < 24:
            return self._interpolate_cubic(audio, start, end)

        # High-Order AR — Pflicht: Ordnung ≥ 30 @ 48 kHz (copilot-instructions §LPC-Pattern)
        # Minimum 30 bindet mehrere Stimmperioden + Formant-Struktur (Rabiner & Schafer 1978).
        order = max(30, min(48, len(before) // 3, len(after) // 3))

        try:
            # Levinson-Durbin via librosa.lpc (scipy ≥ 1.15 hat kein lpc mehr)
            a_fwd = _librosa_lpc.lpc(before.astype(np.float32), order=order).astype(np.float64)
            if not np.isfinite(a_fwd).all():
                return self._interpolate_cubic(audio, start, end)
            a_bwd = _librosa_lpc.lpc(after[::-1].astype(np.float32), order=order).astype(np.float64)
            if not np.isfinite(a_bwd).all():
                return self._interpolate_cubic(audio, start, end)

            # Vorwärtsvorhersage in die Lücke
            zi_fwd = before[-order:].copy()
            pred_fwd, _ = lfilter([1.0], a_fwd, np.zeros(click_len), zi=zi_fwd)

            # Rückwärtsvorhersage (nach links) in die Lücke
            zi_bwd = after[:order][::-1].copy()
            pred_bwd_r, _ = lfilter([1.0], a_bwd, np.zeros(click_len), zi=zi_bwd)
            pred_bwd = pred_bwd_r[::-1]

            # LPC instability guard: AR poles near/outside the unit circle cause
            # exponential growth in predictions (especially in short contexts).
            # Clip to 2× context peak before normalization to prevent silence-region
            # artifacts from exploding forward/backward predictions.
            _ctx_n = min(32, len(before), len(after))
            ctx_peak = max(
                float(np.max(np.abs(before[-_ctx_n:]))),
                float(np.max(np.abs(after[:_ctx_n]))),
                1e-6,
            )
            pred_fwd = np.clip(pred_fwd, -ctx_peak * 2.0, ctx_peak * 2.0)
            pred_bwd = np.clip(pred_bwd, -ctx_peak * 2.0, ctx_peak * 2.0)

            # Spektraler Energieausgleich mit Silence-Gate (~-80 dBFS).
            # Bug-Fix: 1/rms_pred_fwd explodiert wenn LPC fast null ist → rms_pred≈1e-10
            # → scale = rms_ctx/1e-10 = 1e4+ → minimalstes Rauschen auf hörbaren Pegel.
            # Lösung: bei Stille-Kontext direkt auf Null setzen; bei Audio→Stille-Übergang
            # linear ausblenden, damit kein Vorwärts-LPC-Artefakt in die Stille gejagt wird.
            _SILENCE_RMS_FLOOR = 1e-4  # ~-80 dBFS
            if click_len >= 8:
                rms_fwd = np.sqrt(np.mean(before[-8:] ** 2))
                rms_bwd = np.sqrt(np.mean(after[:8] ** 2))
                fwd_silent = rms_fwd <= _SILENCE_RMS_FLOOR
                bwd_silent = rms_bwd <= _SILENCE_RMS_FLOOR

                if not fwd_silent:
                    rms_pred_fwd = np.sqrt(np.mean(pred_fwd**2)) + 1e-10
                    pred_fwd = pred_fwd * (rms_fwd / rms_pred_fwd)
                    if bwd_silent:
                        # Audio→Stille: Vorwärtsvorhersage über Gap ausblenden
                        pred_fwd = pred_fwd * np.linspace(1.0, 0.0, click_len)
                else:
                    pred_fwd = np.zeros_like(pred_fwd)

                if not bwd_silent:
                    rms_pred_bwd = np.sqrt(np.mean(pred_bwd**2)) + 1e-10
                    pred_bwd = pred_bwd * (rms_bwd / rms_pred_bwd)
                    if fwd_silent:
                        # Stille→Audio: Rückwärtsvorhersage über Gap einblenden
                        pred_bwd = pred_bwd * np.linspace(0.0, 1.0, click_len)
                else:
                    pred_bwd = np.zeros_like(pred_bwd)

            # Cosinus-Blend (Hann-Form) statt linearer Gewichtung
            alpha = 0.5 * (1.0 - np.cos(np.pi * np.arange(click_len) / click_len))
            interpolated = (1.0 - alpha) * pred_fwd + alpha * pred_bwd

            # 8-Sample Crossfade an den Kanten
            fade_n = min(8, click_len // 4)
            if fade_n >= 2:
                ramp = np.linspace(0.0, 1.0, fade_n)
                # Einblenden aus Originalrand (vor Lücke)
                edge_pre = audio[max(0, start - fade_n) : start].astype(np.float64)
                if len(edge_pre) == fade_n:
                    interpolated[:fade_n] = ramp * interpolated[:fade_n] + (1.0 - ramp) * edge_pre
                # Ausblenden in Originalrand (nach Lücke)
                edge_post = audio[end + 1 : end + 1 + fade_n].astype(np.float64)
                if len(edge_post) == fade_n:
                    interpolated[-fade_n:] = ramp[::-1] * interpolated[-fade_n:] + (1.0 - ramp[::-1]) * edge_post

            # NaN/Inf-Schutz + Clip
            interpolated = np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0)
            interpolated = np.clip(interpolated, -1.0, 1.0)
            audio[start : end + 1] = interpolated

        except Exception as exc:
            logger.debug("§V6 _interpolate_hermite fehlgeschlagen — Cubic-Spline Fallback aktiviert (Range %d-%d): %s", start, end, exc)
            # Graceful Degradation auf Cubic-Spline
            return self._interpolate_cubic(audio, start, end)

        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        """RBME (Roux & Bimbot 2014): Iterative sparse Bayesian click inpainting.

        Minimizes: ||x - y||^2 + lambda * ||D x||^2 over missing samples,
        where D is a local gradient operator (AR prior).
        Better than pure AR: stabilises prediction via bidirectional pass and
        respects musical transients through energy-bounded blend weights.

        Scientific reference:
            Roux & Bimbot (2014). "Consistent and Repetition-Robust Audio Inpainting
            via Generalized Sparse Bayesian Estimation of Audio Spectra." ICASSP 2014.

        Args:
            signal:  1-D audio array (float32/64).
            mask:    Boolean array, True = missing sample (click position).
            n_iter:  Refinement iterations (default 5 — higher = better boundary fidelity).

        Returns:
            1-D audio array with missing samples inpainted. Returns unchanged copy if
            the reliable-sample budget is insufficient for robust AR estimation.
        """
        result = signal.copy()
        missing_idx = np.where(mask)[0]
        if len(missing_idx) == 0:
            return result

        # AR model on reliable samples (LPC order 16 @ 48 kHz — Aurik spec minimum ≥ 16)
        # PERFORMANCE FIX (§2.54): Use local context window instead of full signal.
        # LPC captures short-range temporal correlation; using the full signal (10M+ samples)
        # causes O(n_clicks × n_signal) = hours on vinyl/shellac with many clicks.
        # Context window ±4096 samples (~85ms @ 48kHz) is sufficient for AR order 16.
        _gap_center = int(missing_idx[len(missing_idx) // 2])
        _CTX_RADIUS = 4096  # 85ms @ 48kHz, >> LPC order 16 context requirement
        _local_start = max(0, _gap_center - _CTX_RADIUS)
        _local_end = min(len(signal), _gap_center + _CTX_RADIUS)
        _local_mask = mask[_local_start:_local_end]
        reliable_signal = signal[_local_start:_local_end][~_local_mask]

        if len(reliable_signal) < 40:
            return result  # insufficient context — caller falls back to cubic spline

        _ar_order = min(16, len(reliable_signal) // 4)
        if _ar_order < 2:
            return result

        # Autocorrelation (positive lags 0 … order) — O(n_local) not O(n_signal)
        _r = np.array(
            [
                np.sum(reliable_signal[: len(reliable_signal) - k] * reliable_signal[k:]) / len(reliable_signal)
                for k in range(_ar_order + 1)
            ]
        )

        # Toeplitz solve (Levinson-Durbin approx) with Tikhonov regularisation (λ=1e-6)
        try:
            _R = np.array([[_r[abs(i - j)] for j in range(_ar_order)] for i in range(_ar_order)])
            _ar_coeffs = np.linalg.solve(_R + 1e-6 * np.eye(_ar_order), _r[1 : _ar_order + 1])
        except np.linalg.LinAlgError as exc:
            logger.debug("§V6 AR-Toeplitz-Solve fehlgeschlagen — Ergebnis unverändert zurückgegeben (LinAlgError): %s", exc)
            return result  # AR estimation failed → caller falls back to cubic spline

        if not np.isfinite(_ar_coeffs).all():
            return result

        # Iterative refinement: bidirectional AR prediction (forward + backward)
        for _iter in range(n_iter):
            # Forward pass: propagate AR prediction left→right into missing region
            for idx in missing_idx:
                if idx >= _ar_order:
                    _pred = float(np.dot(_ar_coeffs, result[idx - _ar_order : idx][::-1]))
                    result[idx] = 0.7 * result[idx] + 0.3 * _pred
            # Backward pass: stabilise via reverse AR prediction right→left
            for idx in missing_idx[::-1]:
                if idx < len(result) - _ar_order:
                    _pred_b = float(np.dot(_ar_coeffs, result[idx + 1 : idx + _ar_order + 1]))
                    result[idx] = 0.6 * result[idx] + 0.4 * _pred_b

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)
        return result  # type: ignore[no-any-return]

    def supports_material(self, _material_type: str) -> bool:
        """All materials supported."""
        return True


__all__ = ["ClickRemovalPhase"]
