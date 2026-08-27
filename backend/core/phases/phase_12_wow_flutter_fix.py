#!/usr/bin/env python3
"""
Phase 12: Professional Wow & Flutter Correction v2.0
=====================================================

Corrects pitch and speed instability caused by mechanical variations in analog playback equipment.

SCIENTIFIC FOUNDATION (Über-SOTA):
- **Mauch & Dixon (2014)**: pYIN — probabilistischer Multi-Threshold-Pitch-Estimator
  → Primärer Algorithmus (ersetzt simples YIN als primäre Methode)

# §G-PYIN-CACHE: Prevent redundant ML pitch detection when phase_12 is called
# per-channel (stereo) or per-chunk (surgical). Cached by audio hash.
_PYIN_CACHE: dict[int, Any] = {}
_PYIN_CACHE_MAX = 8
- De Cheveigné & Kawahara (2002): YIN — nur noch als Legacy-Fallback referenziert
- Laroche & Dolson (1999): Improved Phase Vocoder Time-Scale Modification of Audio
- Driedger & Müller (2016): TSM Toolbox

INDUSTRY BENCHMARKS:
- iZotope RX 10 De-flutter (Phase Vocoder + Spectral Continuity)
- Waves X-Speed (Time & Pitch Manipulation with Formant Preservation)
- Cedar Retouch Pro (Professional Wow/Flutter Removal)
- Capstan by Plangent Processes (Reference Standard for Analog Transfer)
- WOW Control by Magix (Spectral Repair based)
- Steinberg SpectraLayers (Visual Wow/Flutter Correction)
- iZotope Neutron (Pitch Tracking + Time Correction)

ALGORITHM OVERVIEW:

1. Multi-Resolution Pitch Detection (**pYIN — Mauch & Dixon 2014**)
   - 50ms windows with 75% overlap (high temporal resolution)
   - pYIN: Probabilistic Multi-Threshold CMND (Beta-verteilte Gewichte)
   - Sub-harmonic rejection via gewichtetes Kandidaten-Medioid
   - Temporal smoothing (exponentiell, α=0.7)

2. Wow vs Flutter Separation
   - Wow: <4 Hz speed variations (slow pitch drift)
   - Flutter: 4-100 Hz speed variations (fast mechanical vibrations)
   - Separate low-pass (<4 Hz) and band-pass (4-100 Hz) filters
   - Different correction strategies per component

3. Spectral Continuity Preservation
   - Phase Vocoder with vertical phase coherence
   - Formant-preserving time-stretching
   - Harmonic locking (prevent harmonic smearing)
   - Transient detection and preservation

4. Time-Stretching via Phase Vocoder
   - STFT with Hann window (2048 samples, 75% overlap)
   - Phase unwrapping and instantaneous frequency estimation
   - Time-varying stretch factors from pitch deviation curve
   - Overlap-add synthesis with phase coherence

5. Material-Adaptive Correction
   - Tape: Aggressive correction (0.9), high sensitivity (capstan flutter)
   - Vinyl: Moderate correction (0.7), turntable speed variations
   - Shellac: Conservative (0.6), hand-crank artifacts
   - CD_Digital: Minimal (0.2), rare digital artifacts

QUALITY TARGETS:
- Pitch stability: <0.3% residual deviation (Professional standard)
- Spectral artifacts: <-40 dB (imperceptible)
- Transient preservation: >95% (no smearing)
- Realtime factor: <0.8× (faster than playback)

Author: Aurik Professional Team
Version: 2.0.0 (Professional)
Date: February 2026
"""
# pylint: disable=import-outside-toplevel

import logging
import os
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import (
    audio_sample_count,
    restore_layout,
    safe_to_mono,
    stereo_channel_view,
    stereo_like,
    to_channels_last,
)
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# Resource Management for fallback to lightweight algorithms
try:
    from backend.core.adaptive_resource_manager import adaptive_resource_manager

    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    logging.getLogger(__name__).warning("AdaptiveResourceManager not available, no automatic fallback")

# ML-Hybrid Support (Aurik 10.0.0 - Phase 12 v3.0)
try:
    from backend.core.hybrid.hybrid_wow_flutter import HybridWowFlutter, PitchDetectionStrategy, WowFlutterConfig

    ML_HYBRID_AVAILABLE = True
except ImportError:
    ML_HYBRID_AVAILABLE = False
    logging.getLogger(__name__).warning("ML-Hybrid wow/flutter detector not available, using DSP-only mode")

logger = logging.getLogger(__name__)

# §G-PYIN-CACHE: Per-Session pYIN-Resultat-Cache
_PYIN_CACHE: dict[tuple[int, ...], Any] = {}
_PYIN_CACHE_MAX: int = 8

# ── §C3 Circuit-Breaker: Polyphonic consensus degradation ────────────
# When polyphonic estimation fails with consensus=0 on N consecutive
# chunks, skip the expensive polyphonic estimator for the remainder of
# the run.  Cassette/tape material with insufficient polyphonic content
# (T=1–14, no voiced voices) triggers identical pYIN fallback every time,
# wasting CPU on a known-lost cause.
# §v10.18: Increased from 3→8. Narrow-bandwidth MP3-encoded material
# often produces zero-consensus on initial chunks (coding artifacts
# confuse BasicPitch) but recovers on later passages with more harmonic
# density. 3 chunks was too aggressive — legitimate polyphonic content
# was permanently disabled before it had a chance to succeed.
_POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS: int = 0
_POLYPHONIC_CB_MAX_FAILURES: int = 8  # Max consecutive zero-consensus chunks before skipping
_POLYPHONIC_CB_ACTIVE: bool = False  # True = polyphonic estimator permanently skipped for this run


def _reset_polyphonic_circuit_breaker() -> None:
    """Reset circuit breaker state at the start of each restoration run."""
    global _POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS, _POLYPHONIC_CB_ACTIVE
    _POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS = 0
    _POLYPHONIC_CB_ACTIVE = False
    _PYIN_CACHE.clear()  # §G-PYIN-CACHE: reset per-run


class WowFlutterFix(PhaseInterface):
    """Professional Wow & Flutter Correction with YIN pitch detection and Phase Vocoder time-stretching."""

    # Material-adaptive correction strength (0.0-1.0)
    CORRECTION_STRENGTH = {
        MaterialType.TAPE: 0.80,  # v10.0.0: raised from 0.65 — tonal_center PMGG-excluded (§2.29b);
        #   cassette head-settling wow/flutter requires stronger correction.
        #   Was reduced in v10.0.0 due to tonal_center regression, but K-S proxy
        #   (§9.7.11) now excludes tonal_center from PMGG delta-checks for phase_12.
        MaterialType.CASSETTE: 0.80,  # v10.0.0: same as TAPE — compact cassette uses identical
        #   capstan/pinch-roller transport (IEC 60094-1); head-settling wow/flutter same physics.
        #   Previous fallback to 0.7 (default) was too conservative for cassette transport bumps.
        MaterialType.VINYL: 0.70,  # Moderate (turntable speed variations, belt/motor issues)
        MaterialType.SHELLAC: 0.60,  # Conservative (hand-crank artifacts, worn mechanisms)
        MaterialType.CD_DIGITAL: 0.20,  # Minimal (rare digital artifacts)
        MaterialType.STREAMING: 0.10,  # Very minimal (usually none)
    }

    # Detection sensitivity (minimum pitch deviation to correct, in %)
    DETECTION_THRESHOLD = {
        MaterialType.TAPE: 0.3,  # 0.3% pitch deviation (high sensitivity)
        MaterialType.CASSETTE: 0.3,  # v10.0.0: same as TAPE — compact cassette transport bumps
        #   (Bandhopser) cause local pitch deviations of 0.3-0.8%; the previous 0.5% default
        #   missed borderline transport bumps. IEC 60094-1 cassette flutter spec: ≤ 0.2% WRMS
        #   at 4.75 cm/s → any detected deviation ≥ 0.3% is above spec → correct it.
        MaterialType.VINYL: 0.5,  # 0.5% pitch deviation
        MaterialType.SHELLAC: 0.8,  # 0.8% pitch deviation
        MaterialType.CD_DIGITAL: 2.0,  # 2.0% pitch deviation (low sensitivity)
        MaterialType.STREAMING: 3.0,  # 3.0% pitch deviation (rarely triggered)
    }

    # YIN algorithm parameters
    YIN_THRESHOLD = 0.15  # Confidence threshold for pitch detection
    PITCH_WINDOW_MS = 100  # Larger window for stability (was 50ms)
    # v10.0.0: Restored to 75 % overlap (factor=4, was 2=50 %) for 4× better
    # temporal resolution in pYIN flutter detection. At 48 kHz and 100 ms window:
    #   factor=2 → hop=50 ms — insufficient for 4–20 Hz flutter (20–250 ms periods)
    #   factor=4 → hop=25 ms — 2+ samples per flutter cycle (Nyquist-safe)
    # librosa.pyin fast-path ≪ 200 ms even for 20-min files. §9.10.80
    # Quality-first: no RT sacrifice in main path.
    PITCH_HOP_FACTOR = 4  # 75 % overlap — restored from 2 (v10.0.0)

    # Phase Vocoder / STFT parameters
    STFT_WINDOW_SIZE = 2048  # 23 Hz/bin @ 48 kHz — restored from 1024 (v10.0.0)
    STFT_HOP_SIZE = 512  # 75 % overlap with 2048 window — restored from 256

    # Formant preservation (prevent "chipmunk" effect)
    PRESERVE_FORMANTS = True
    _MAX_PERCENTILE_PEAK = 0.985

    def __init__(self):
        super().__init__()
        self.name = "Wow & Flutter Correction v2 Professional"
        self._quality_mode_hint: str = "quality"
        self._quality_first_unleashed: bool = False
        # §P3 Chunk-Overlap-State: Hält die letzten 1 s Audio des vorherigen
        # Chunks für Crossfade an Chunk-Grenzen, um Artefakte durch UV3-
        # Chunking (typisch 5 s) zu eliminieren.
        self._chunk_overlap_tail: np.ndarray | None = None
        self._chunk_overlap_sr: int = 48000
        self._chunk_overlap_dur_s: float = 1.0  # 1 s overlap = 20 % bei 5 s Chunks

    @staticmethod
    def _compute_adaptive_threshold_profile(
        material: MaterialType,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """Berechnet adaptive detection/confidence thresholds for phase_12.

        Values are bounded to stable ranges and tuned to keep fast mode more
        conservative than quality while allowing low-restorability material to
        pass with a slightly lower correction-confidence floor.
        """
        _qm = str(quality_mode or "balanced").lower().replace("-", "_")
        _rest = float(np.clip(restorability_score, 0.0, 100.0))
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113

        _det_base = WowFlutterFix.DETECTION_THRESHOLD.get(_mk, 0.5)  # type: ignore[call-overload]
        _yin_base = float(WowFlutterFix.YIN_THRESHOLD)

        _ml_conf_base_by_mat = {
            "shellac": 0.62,
            "wax_cylinder": 0.64,
            "vinyl": 0.60,
            "tape": 0.56,
            "reel_tape": 0.56,
            "cd_digital": 0.58,
            "streaming": 0.60,
        }
        _ml_conf_base = float(_ml_conf_base_by_mat.get(_mk, 0.60))

        _min_conf_base_by_mat = {
            "shellac": 0.30,
            "wax_cylinder": 0.28,
            "vinyl": 0.26,
            "tape": 0.22,
            "reel_tape": 0.22,
            "cd_digital": 0.30,
            "streaming": 0.32,
        }
        _min_conf_base = float(_min_conf_base_by_mat.get(_mk, 0.26))

        _mode_det_adj = {
            "fast": +0.25,
            "balanced": 0.0,
            "quality": -0.10,
            "maximum": -0.15,
            "restoration": -0.05,
            "studio_2026": -0.15,
        }.get(_qm, 0.0)
        _mode_yin_adj = {
            "fast": +0.02,
            "balanced": 0.0,
            "quality": -0.01,
            "maximum": -0.015,
            "restoration": -0.005,
            "studio_2026": -0.015,
        }.get(_qm, 0.0)
        _mode_ml_adj = {
            "fast": +0.06,
            "balanced": 0.0,
            "quality": -0.03,
            "maximum": -0.05,
            "restoration": -0.02,
            "studio_2026": -0.05,
        }.get(_qm, 0.0)

        # Lower restorability: allow slightly lower correction-confidence floor.
        _rest_min_conf_adj = ((_rest - 50.0) / 50.0) * 0.06

        detection_threshold = float(np.clip(_det_base + _mode_det_adj, 0.20, 3.50))
        yin_threshold = float(np.clip(_yin_base + _mode_yin_adj, 0.08, 0.25))
        ml_confidence_threshold = float(np.clip(_ml_conf_base + _mode_ml_adj, 0.45, 0.85))
        min_confidence_for_correction = float(np.clip(_min_conf_base + _rest_min_conf_adj, 0.18, 0.60))

        return {
            "detection_threshold": detection_threshold,
            "yin_threshold": yin_threshold,
            "ml_confidence_threshold": ml_confidence_threshold,
            "min_confidence_for_correction": min_confidence_for_correction,
        }

    @staticmethod
    def _derive_safe_timing_profile(
        material: MaterialType,
        mean_confidence: float,
        vocals_confidence: float,
        *,
        polyphonic_fallback: bool = False,
    ) -> tuple[float, float]:
        """Reduce timing aggression for content that is easy to damage.

        Vocal-heavy vintage transfers are especially sensitive to articulation and
        authenticity loss from even correct-but-strong time warping. Keep the
        internal timing remap narrower than the outer PMGG strength alone would
        allow.
        """
        strength_scale = 1.0
        max_stretch_delta = 0.05

        if vocals_confidence >= 0.40:
            strength_scale *= 0.82
            max_stretch_delta = min(max_stretch_delta, 0.035)

        if (
            material
            in {
                MaterialType.VINYL,
                MaterialType.SHELLAC,
                MaterialType.WAX_CYLINDER,
                MaterialType.WIRE_RECORDING,
                MaterialType.LACQUER_DISC,
            }
            and mean_confidence < 0.75
        ):
            strength_scale *= float(np.clip(0.82 + 0.20 * mean_confidence, 0.82, 0.97))
            max_stretch_delta = min(max_stretch_delta, 0.03)
        elif mean_confidence < 0.60:
            strength_scale *= 0.90
            max_stretch_delta = min(max_stretch_delta, 0.04)

        if polyphonic_fallback:
            # If the polyphonic speed estimator rejected the signal as implausible,
            # stay in a narrow DSP-safe correction band. Continuing with the broader
            # quality-mode path can drift tonal center on difficult analog vocals.
            strength_scale *= 0.78
            max_stretch_delta = min(max_stretch_delta, 0.02)

        return float(strength_scale), float(max_stretch_delta)

    @staticmethod
    def _should_bypass_unsafe_polyphonic_fallback(
        material: MaterialType,
        mean_confidence: float,
        vocals_confidence: float,
        *,
        polyphonic_fallback: bool,
    ) -> bool:
        """Bypass correction entirely when the estimator already proved unstable.

        If the polyphonic speed estimator rejects the input as implausible, a second,
        still-global pitch correction pass on vocal-heavy analog material often harms
        tonal center more than it helps. In that case, minimal intervention is safer
        than a likely rollback.
        """
        if not polyphonic_fallback:
            return False
        if vocals_confidence < 0.40:
            return False
        if mean_confidence >= 0.80:
            return False
        return material in {
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
        }

    @staticmethod
    def _polyphonic_estimate_is_insufficient(pitch_trajectory: np.ndarray, confidence: np.ndarray) -> bool:
        """Bewertet polyphone Schätzung auf minimale Evidenz.

        Ein einzelner Pitch-Frame (z. B. T=1) oder nahezu keine validen Frames
        ist für eine stabile Wow/Flutter-Korrektur nicht belastbar.
        In diesem Fall erzwingen wir Re-Estimate via pYIN statt Komplett-Skip.
        """
        if pitch_trajectory.size < 4:
            return True
        if confidence.size == 0:
            return True
        valid_pitch = int(np.sum(np.asarray(pitch_trajectory) > 0.0))
        valid_conf = int(np.sum(np.asarray(confidence) > 0.15))
        return valid_pitch < 4 or valid_conf < 4

    def get_metadata(self) -> PhaseMetadata:
        """Gibt zurück: phase metadata."""
        return PhaseMetadata(
            phase_id="phase_12_wow_flutter_fix",
            name="Wow & Flutter Correction v2 Professional",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,  # HIGH priority (wow/flutter very audible)
            dependencies=["phase_01_click_removal", "phase_09_crackle_removal"],
            estimated_time_factor=0.15,  # 15% of audio duration
            version="2.0.0",
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.92,  # Professional-grade wow/flutter correction
            description=(
                "Professional wow & flutter correction with YIN pitch detection and Phase Vocoder time-stretching"
            ),
        )

    def process(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, audio: np.ndarray, sample_rate: int = 48000, material_type: str = "unknown", **kwargs: Any
    ) -> PhaseResult:
        check_ml_model_ready("BasicPitch", phase_name="12")
        check_ml_model_ready("CREPE", phase_name="12")
        check_ml_model_ready("FCPE", phase_name="12")
        check_ml_model_ready("PANNs", phase_name="12")
        check_ml_model_ready("Whisper", phase_name="12")
        """
        Correct wow & flutter using professional pitch detection + Phase Vocoder.

        Args:
            audio: Audio samples (mono: [samples], stereo: [samples, 2])
            sample_rate: Sample rate in Hz
            material: Material type for adaptive parameters

        Returns:
            PhaseResult with wow/flutter corrected audio
        """
        _material_raw = kwargs.get("material", material_type) if material_type == "unknown" else material_type
        material: MaterialType = (
            _material_raw
            if isinstance(_material_raw, MaterialType)
            else MaterialType(_material_raw)
            if isinstance(_material_raw, str) and _material_raw in {m.value for m in MaterialType}
            else MaterialType.VINYL
        )
        self.validate_input(audio)
        assert sample_rate == 48000, f"Interne SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        _progress_cb = kwargs.get("progress_sub_callback")

        def _report_progress(pct: float, label: str) -> None:
            if callable(_progress_cb):
                try:
                    _progress_cb(float(np.clip(pct, 0.0, 100.0)), label, time.time() - start_time)
                except Exception:
                    logger.debug("_report_progress: silent except suppressed", exc_info=True)

        _original_audio = np.asarray(audio, dtype=np.float32).copy()

        # §2.51: Stereo-Layout-Normalisierung — akzeptiere (2,N) und (N,2),
        # verarbeite kanonisch channels-last, stelle am Ausgang wieder her
        # (Rev. 2026-08-16: test_stereo_axis_matrix[phase_12] schlug fehl, weil
        # process() channels-first-Input unnormalisiert weitergab).
        audio, _was_channels_first = to_channels_last(_original_audio)
        _original_audio = np.asarray(audio, dtype=np.float32).copy()

        # §P3 Chunk-Overlap: Wenn ein vorheriger Chunk existiert, prepend den
        # 1s-Überlapp mit Crossfade, um Artefakte an Chunk-Grenzen zu eliminieren.
        _overlap_samples = int(self._chunk_overlap_dur_s * sample_rate)
        _chunk_has_overlap = False
        if self._chunk_overlap_tail is not None and self._chunk_overlap_tail.size > 0:
            _tail = self._chunk_overlap_tail
            if _tail.shape == audio.shape or (_tail.ndim == audio.ndim):
                _tail_len = min(_overlap_samples, _tail.shape[0])
                _crossfade_len = min(_tail_len, _original_audio.shape[0])
                if _crossfade_len > 0:
                    _tail_use = _tail[-_tail_len:] if _tail.shape[0] >= _tail_len else _tail
                    # Hanning-crossfade an der Chunk-Grenze
                    _fade_in = np.hanning(_crossfade_len * 2)[:_crossfade_len]
                    _fade_out = 1.0 - _fade_in
                    if _original_audio.ndim == 2 and _tail_use.ndim == 1:
                        _tail_use = np.column_stack([_tail_use, _tail_use])
                    if _original_audio.ndim == 2:
                        _fade_in = _fade_in[:, np.newaxis]
                        _fade_out = _fade_out[:, np.newaxis]
                    _overlap_region = _tail_use[:_crossfade_len] * _fade_out.astype(np.float32) + _original_audio[
                        :_crossfade_len
                    ] * _fade_in.astype(np.float32)
                    _original_audio[:_crossfade_len] = _overlap_region
                    _chunk_has_overlap = True
                    logger.debug(
                        "Verarbeitungsschritt 12 chunk-overlap: crossfaded %d samples from previous chunk",
                        _crossfade_len,
                    )
            self._chunk_overlap_tail = None  # consumed

        # §2.60/§2.51 STCG pre-phase: L/R-Alignment VOR Chunk-Verarbeitung.
        # Phase 12's M/S-domain processing (§2.51) prevents NEW L/R time drift,
        # but it must START with aligned channels. A residual lag from earlier
        # pipeline stages (DC-offset removal, TDP) will cause the M/S matrix to
        # mix temporally offset channels, creating phantom spatial artefacts.
        # Sub-sample correction via GCC-PHAT + scipy.ndimage.shift.
        # §v10.13: STCG NUR auf vollständigem Audio (≥15s), nicht auf Chunks.
        # MP3-Joint-Stereo-Artefakte auf 5s-Chunks werden von XCorr als
        # Zeitversatz fehlinterpretiert → unterschiedliche Korrekturen pro
        # Chunk erzeugen echten L/R-Drift. Der M/S-Gleichlauf-Fix (Side =
        # selber Algorithmus wie Mid) verhindert Lag-Entstehung ohne STCG.
        _p12_dur_s = audio.shape[0] / sample_rate if audio.ndim >= 1 else 0.0
        if audio.ndim == 2 and _p12_dur_s >= 15.0:
            try:
                from backend.core.stereo_temporal_coherence_guard import (
                    get_stereo_temporal_coherence_guard,
                )

                audio = get_stereo_temporal_coherence_guard().correct_interchannel_delay(
                    audio, sample_rate, phase_id="phase_12_pre_chunking"
                )
            except Exception as _stcg_p12_exc:
                logger.debug("Verarbeitungsschritt 12 STCG pre-chunking uebersprungen: %s", _stcg_p12_exc)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §v10.96 Defekt-basiertes Skip-Gate: Wow/Flutter-Detektion vor Ausführung.
        # Wenn weder Wow noch Flutter vom DefectScanner detektiert wurden,
        # ist die teure ML-Hybrid-Pitch-Detektion (221 s) unnötig.
        # Transport-Bump-Repair (135 Bumps, ~5 s) läuft trotzdem — ist schnell.
        _wow_kw = kwargs.get("wow_severity", kwargs.get("wow"))
        _flutter_kw = kwargs.get("flutter_severity", kwargs.get("flutter"))
        _explicit_defect_data = _wow_kw is not None or _flutter_kw is not None
        _wow_sev_defect = float(_wow_kw or 0.0)
        _flutter_sev_defect = float(_flutter_kw or 0.0)
        # §v10.200 Fallback: DefectScanner-Daten aus _restoration_context abrufen,
        # da kwargs nach Resampling (44100→48000) die Werte verlieren können.
        if not _explicit_defect_data:
            _rctx_fb = kwargs.get("_restoration_context", {}) or {}
            _defect_scores_fb = _rctx_fb.get("defect_scores", _rctx_fb.get("defect_focus_scores", {})) or {}
            if isinstance(_defect_scores_fb, dict) and any(
                _k in _defect_scores_fb for _k in ("wow", "flutter", "wow_severity", "flutter_severity")
            ):
                _explicit_defect_data = True
            _wow_sev_defect = float(
                _defect_scores_fb.get("wow", _defect_scores_fb.get("wow_severity", _wow_sev_defect)) or 0.0
            )
            _flutter_sev_defect = float(
                _defect_scores_fb.get("flutter", _defect_scores_fb.get("flutter_severity", _flutter_sev_defect)) or 0.0
            )
        # §v10.96: Nur skippen, wenn der DefectScanner WOW/FLUTTER EXPLIZIT als
        # abwesend (< 0.15) gemeldet hat. Ohne Defect-Daten (Direktaufruf/Unit)
        # fail-open — die Phase läuft normal.
        _wow_flutter_skip = _explicit_defect_data and _wow_sev_defect < 0.15 and _flutter_sev_defect < 0.15
        if _wow_flutter_skip:
            # §v10.303: Erste Meldung als INFO, Wiederholungen als DEBUG (Log-Spam-Prävention)
            _log_fn = logger.info if not getattr(self, "_wow_skip_logged", False) else logger.debug
            self._wow_skip_logged = True
            _log_fn(
                "§v10.96 Wow/Flutter-Skip: wow=%.3f flutter=%.3f < 0.15 → ML-Hybrid skipped, transport-bump only",
                _wow_sev_defect,
                _flutter_sev_defect,
            )
            # Setze wow/flutter-spezifische Parameter auf Minimalwerte
            _effective_strength = 0.0  # Keine Pitch-Detektion, nur Transport-Repair

        if _effective_strength <= 0.0:
            # §2.74: Transport-Defekte (tape_head_level_dip/transport_bump) werden auch
            # bei Wow/Flutter-Skip repariert — kein Pitch-Tracking nötig.
            _TAPE_MATS_Z = {MaterialType.TAPE, MaterialType.REEL_TAPE, MaterialType.CASSETTE}
            _mat_z = material if isinstance(material, MaterialType) else None
            _dl_z = kwargs.get("defect_locations") or {}
            _has_dip_z = bool(_dl_z.get("tape_head_level_dip") or _dl_z.get("transport_bump"))
            _n_dips_z = 0
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            if _mat_z in _TAPE_MATS_Z or _has_dip_z:
                passthrough, _n_dips_z = self._stabilize_tape_level(
                    passthrough,
                    sample_rate,
                    1.0,
                    is_primary_tape=(_mat_z in _TAPE_MATS_Z),
                    confirmed_tape_dip=_has_dip_z,
                    protected_zones=self._collect_vfa_protected_zones(kwargs),
                )
                passthrough = np.clip(np.nan_to_num(passthrough, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=restore_layout(passthrough, _was_channels_first),
                resolved_defects={},
                metrics={
                    "wow_flutter_detected": False,
                    "max_deviation_percent": 0.0,
                    "correction_applied": 1.0 if _n_dips_z > 0 else 0.0,
                    "material": material.value,
                    "mean_confidence": 0.0,
                    "quality_mode": kwargs.get("quality_mode", "balanced"),
                    "tape_level_dips_repaired": _n_dips_z,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "version": "4.1_locality",
                    "phase_locality_factor": phase_locality_factor,
                    # §2.51 Locality-Vertrag: die skalierte Soll-Stärke bleibt sichtbar,
                    # auch wenn der §v10.96-Skip-Gate die Verarbeitung überspringt.
                    "effective_strength": float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0)),
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # Get material-specific parameters
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        strength = float(self.CORRECTION_STRENGTH.get(_mk, 0.7) * _effective_strength)  # type: ignore[call-overload]
        threshold = self.DETECTION_THRESHOLD.get(_mk, 0.5)  # type: ignore[call-overload]

        is_stereo = audio.ndim == 2
        mono = safe_to_mono(audio).astype(np.float32) if is_stereo else audio.copy()

        _report_progress(5.0, "Wow/Flutter: Tonhöhen-Analyse startet")
        # ML-Hybrid Mode Routing (v3.0)
        quality_mode = kwargs.get("quality_mode", "quality")
        self._quality_mode_hint = str(quality_mode).strip().lower()
        self._quality_first_unleashed = bool(
            kwargs.get("quality_first_unleashed", self._quality_mode_hint in {"quality", "maximum"})
        )
        use_lightweight = False
        if RESOURCE_MANAGER_AVAILABLE:
            use_lightweight = adaptive_resource_manager.should_use_lightweight_mode()
            # Quality-first contract: do not downgrade to lightweight in quality tiers.
            if quality_mode in ["quality", "maximum"]:
                use_lightweight = False
            elif use_lightweight:
                logger.info(
                    "Verarbeitungsschritt 12: Resource constraint erkannt, forcing DSP-only Betriebsart (CPU: %.1f%%, Memory: %.1f%%)",
                    adaptive_resource_manager.get_cpu_usage(),
                    adaptive_resource_manager.get_memory_usage(),
                )

        # Strategy routing v4.0 — Capstan-kompetitiv:
        # maximum  → PolyphonicSpeedCurveEstimator (multi-F0 consensus, §2.12)
        # quality  → HybridWowFlutter pYIN+FCPE/CREPE (ML-Hybrid, HYBRID strategy)
        # balanced → HybridWowFlutter pYIN+FCPE/CREPE (ML-Hybrid, ADAPTIVE strategy)
        # fast / lightweight → pYIN DSP
        _poly_applied = False
        _poly_fallback = False
        pitch_trajectory: np.ndarray = np.zeros(1)
        confidence: np.ndarray = np.zeros(1)

        if quality_mode in ["quality", "maximum"] and not use_lightweight:
            # §C3 Circuit-Breaker: skip polyphonic estimator after N consecutive
            # zero-consensus failures (e.g. cassette with no polyphonic content).
            global _POLYPHONIC_CB_ACTIVE, _POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS
            if _POLYPHONIC_CB_ACTIVE:
                logger.debug(
                    "Verarbeitungsschritt 12: Polyphoner Pfad per Circuit-Breaker deaktiviert "
                    "(nach %d konsekutiven Zero-Consensus-Chunks) — pYIN direkt",
                    _POLYPHONIC_CB_MAX_FAILURES,
                )
            else:
                try:
                    from backend.core.hybrid.hybrid_wow_flutter import (
                        PolyphonicSpeedCurveEstimator,
                    )

                    _poly_est = PolyphonicSpeedCurveEstimator()
                    pitch_trajectory, confidence = _poly_est.estimate(mono, sample_rate)
                    _poly_applied = True
                    logger.info(
                        "Verarbeitungsschritt 12 polyphoner Konsensus: T=%d Frames, material=%s",
                        len(pitch_trajectory),
                        material.value,
                    )
                except Exception as _poly_exc:
                    logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
                    _poly_fallback = True
                    logger.warning(
                        "PolyphonicSpeedCurveEstimator fehlgeschlagen (%s) — ML-Hybrid-Ersatzpfad",
                        _poly_exc,
                    )

        # ML-Hybrid only if polyphonic path did not succeed and resources permit
        use_ml_hybrid = (
            not _poly_applied
            and ML_HYBRID_AVAILABLE
            and quality_mode in ["balanced", "quality", "maximum"]
            and not use_lightweight
        )

        if _poly_fallback and quality_mode in ["quality", "maximum"]:
            # Conservative restoration fallback: keep Phase 12 in narrow DSP mode
            # after an implausible polyphonic estimate instead of escalating again.
            use_ml_hybrid = False

        if use_ml_hybrid:
            try:
                # §G-PYIN-CACHE: Avoid re-running ML detection for same audio (per-channel calls)
                _audio_hash = hash(mono.tobytes())
                if _audio_hash in _PYIN_CACHE:
                    ml_result = _PYIN_CACHE[_audio_hash]  # type: ignore[index]
                    logger.debug(
                        "Verarbeitungsschritt 12 ML-Hybrid: zwischengespeichert pitch Ergebnis reused (confidence=%.3f)",
                        ml_result.mean_confidence,
                    )
                else:
                    logger.info(
                        "Verarbeitungsschritt 12 ML-Hybrid: Betriebsart=%s, material=%s", quality_mode, material.value
                    )

                    # Configure ML pitch detector strategy
                    if quality_mode in ["quality", "maximum"]:
                        strategy = PitchDetectionStrategy.HYBRID
                    else:
                        strategy = PitchDetectionStrategy.ADAPTIVE

                    # §v10.14 PLM-Guard: FCPE+CREPE vor Eviction schützen
                    _plm12 = None
                    try:
                        _plm12 = get_plugin_lifecycle_manager()
                        _plm12.set_active("FCPE", True)
                        _plm12.set_active("CREPE", True)
                    except Exception:
                        _plm12 = None
                    try:
                        detector = HybridWowFlutter(
                            config=WowFlutterConfig(
                                strategy=strategy,
                                crepe_model="full" if quality_mode in ["quality", "maximum"] else "medium",
                                confidence_threshold=0.7,
                                enable_preprocessing=True,
                            )
                        )

                        ml_result = detector.detect_pitch(mono, sample_rate=sample_rate)
                    finally:
                        if _plm12 is not None:
                            try:
                                _plm12.set_active("CREPE", False)
                                _plm12.set_active("FCPE", False)
                            except Exception:
                                pass
                    if len(_PYIN_CACHE) >= _PYIN_CACHE_MAX:
                        _PYIN_CACHE.pop(next(iter(_PYIN_CACHE)))
                    _PYIN_CACHE[_audio_hash] = ml_result  # type: ignore[index]

                logger.info(
                    "ML-Hybrid Pitch-Detektion abgeschlossen: pYIN=%s, CREPE=%s, confidence=%.3f",
                    ml_result.pyin_applied,
                    ml_result.crepe_applied,
                    ml_result.mean_confidence,
                )

                # Use ML-detected pitch trajectory
                pitch_trajectory = ml_result.pitch_trajectory
                confidence = ml_result.confidence

            except Exception as e:
                logger.warning(
                    "ML-Hybrid Pitch-Detektion fehlgeschlagen: %s, Ersatzpfad auf pYIN DSP. Fehlertyp: %s",
                    e,
                    type(e).__name__,
                )
                # Fall through to DSP path below
                use_ml_hybrid = False

        # DSP-Only (Fast-Modus oder Fallback): pYIN
        if not _poly_applied and not use_ml_hybrid:
            logger.info("Verarbeitungsschritt 12 pYIN DSP: material=%s", material.value)
            pitch_trajectory, confidence = self._estimate_pitch_yin(mono, sample_rate)

        # Polyphonie-Guard: Wenn der Konsensus nur minimale Evidenz liefert
        # (typisch T=1 oder fast keine validen Frames), nicht sofort skippen.
        # Stattdessen robust auf pYIN neu schätzen, damit transportbedingte
        # Instabilitäten (Bandhopser/Wow) nicht unentdeckt bleiben.
        if _poly_applied and self._polyphonic_estimate_is_insufficient(pitch_trajectory, confidence):
            logger.info(
                "Verarbeitungsschritt 12: Polyphoner Konsensus unzureichend (T=%d, valid_pitch=%d, valid_conf=%d) — "
                "Re-Estimate via pYIN",
                int(pitch_trajectory.size),
                int(np.sum(np.asarray(pitch_trajectory) > 0.0)),
                int(np.sum(np.asarray(confidence) > 0.15)),
            )
            # §C3: Track consecutive zero-consensus — circuit-breaker.
            _consensus_frames = (
                int(np.sum(~np.isnan(np.full(pitch_trajectory.size, np.nan)))) if pitch_trajectory.size > 0 else 0
            )
            if _consensus_frames == 0:
                _POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS += 1
                if _POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS >= _POLYPHONIC_CB_MAX_FAILURES:
                    _POLYPHONIC_CB_ACTIVE = True
                    # §C3 SOTA: Circuit-Breaker ist KEIN Fehler, sondern eine
                    # normale Fallback-Strategie für Material ohne polyphone
                    # Inhalte (Solo-Gesang, Sprache, Mono-Aufnahmen). Der
                    # pYIN-Schätzer (Mauch & Dixon 2014) übernimmt zuverlässig.
                    logger.info(
                        "Verarbeitungsschritt 12 Circuit-Breaker: %d konsekutive Zero-Consensus-Chunks — "
                        "polyphoner Pfad für diesen Lauf deaktiviert",
                        _POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS,
                    )
            else:
                _POLYPHONIC_CONSECUTIVE_ZERO_CONSENSUS = 0
            pitch_trajectory, confidence = self._estimate_pitch_yin(mono, sample_rate)
            _poly_applied = False
            _poly_fallback = True

        _report_progress(52.0, "Wow/Flutter: Tonhöhen-Analyse abgeschlossen")
        # Continue with standard wow/flutter correction pipeline
        # (regardless of detection method)

        # Confidence-Guard: Bei sehr niedriger mittlerer Konfidenz die Phase gar nicht
        # erst anwenden — Phase-Vocoder-Timestretch auf Basis unzuverlässiger Pitch-Daten
        # erzeugt Artefakte (0.09 PMGG-Regression im E2E) ohne tatsächlichen Nutzen.
        # Timing-Phasen haben keine Wet/Dry-Retries, daher frühes Bail-out.
        #
        # v10.0.0: Tape-Start Confidence Adaptation.
        # Cassette motor startup (0–20 s) produces low-confidence pitch regions because
        # the signal is genuinely unstable (speed ramp-up).  The confidence guard must
        # NOT skip the ENTIRE phase just because the first few seconds have low confidence.
        # For tape material: lower the threshold to 0.25 (from 0.40) to prevent skipping
        # precisely where correction is most needed.
        # Scientific basis: Mauch & Dixon (2014) §3.2 "Voiced probability in noisy signals";
        # pYIN confidence degrades with speed instability but pitch estimates remain usable
        # above 0.20 threshold in the low-Hz (<200 Hz) tape flutter range.
        _valid_conf = confidence[confidence > 0]
        _mean_conf = float(np.mean(_valid_conf)) if len(_valid_conf) > 0 else 0.0
        _MIN_CONFIDENCE_FOR_CORRECTION = 0.40
        if material in (MaterialType.TAPE, MaterialType.CASSETTE):
            _MIN_CONFIDENCE_FOR_CORRECTION = 0.25  # tape/cassette: transport-start-aware lower threshold
        if _mean_conf < _MIN_CONFIDENCE_FOR_CORRECTION:
            # -------------------------------------------------------------------
            # Tier-3-Fallback: Instantan-Frequenz-Tracking (IEC 60386, SNR-robust)
            # Wird aktiviert wenn pYIN/BasicPitch für rauschiges Kassetten-/
            # Bandmaterial völlig scheitert (Konfidenz ≈ 0.000).
            # Funktioniert ohne Voiced-Detection; bei Pitch-Unsicherheit bleibt
            # die nicht-destruktive Transport-Level-Stabilisierung aktiv.
            # -------------------------------------------------------------------
            _TAPE_LEVEL_MATERIALS = {MaterialType.TAPE, MaterialType.REEL_TAPE, MaterialType.CASSETTE}
            _mat_enum = material if isinstance(material, MaterialType) else None
            _is_primary_tape = _mat_enum in _TAPE_LEVEL_MATERIALS
            _centroid_succeeded = False

            if _is_primary_tape and _effective_strength > 0.0:
                try:
                    _c_pitch, _c_conf = self._estimate_speed_curve_from_instantaneous_frequency(mono, sample_rate)
                    _c_valid = _c_conf[_c_conf > 0]
                    _c_mean_conf = float(np.mean(_c_valid)) if len(_c_valid) > 0 else 0.0
                    if _c_pitch.size >= 4 and _c_mean_conf >= 0.15:
                        pitch_trajectory = _c_pitch
                        confidence = _c_conf
                        _mean_conf = _c_mean_conf
                        _centroid_succeeded = True
                        logger.info(
                            "Verarbeitungsschritt 12: Instantan-Frequenz-Ersatzpfad aktiviert — "
                            "Konfidenz %.3f, T=%d Frames, material=%s",
                            _mean_conf,
                            len(pitch_trajectory),
                            material.value,
                        )
                except Exception as _c_exc:
                    logger.warning(
                        "Verarbeitungsschritt 12 IF-Ersatzpfad fehlgeschlagen (%s) — Transportstabilisierung", _c_exc
                    )

            if not _centroid_succeeded:
                logger.info(
                    "Verarbeitungsschritt 12: konservativer Ersatzpfad aktiv (Konfidenz %.3f < %.2f) — "
                    "Transportstabilisierung statt Vollkorrektur",
                    _mean_conf,
                    _MIN_CONFIDENCE_FOR_CORRECTION,
                )
                # Transport-Level-Stabilisierung auch im Low-Confidence-Pfad,
                # damit kein Komplett-Skip bei Kassetten-/Bandmaterial entsteht.
                n_level_dips_repaired = 0
                # §2.74: BOTH tape_head_level_dip AND transport_bump trigger
                # the tape level stabilizer. These are genuine analog defects that
                # MP3 cannot produce — transport_bump was previously excluded by
                # the MP3 guard but is now properly passed through.
                _dl = kwargs.get("defect_locations") or {}
                _has_tape_dip_defect = bool(_dl.get("tape_head_level_dip") or _dl.get("transport_bump"))
                if (_is_primary_tape or _has_tape_dip_defect) and _effective_strength > 0.0:
                    audio, n_level_dips_repaired = self._stabilize_tape_level(
                        audio,
                        sample_rate,
                        _effective_strength,
                        is_primary_tape=_is_primary_tape,
                        confirmed_tape_dip=_has_tape_dip_defect,
                        protected_zones=self._collect_vfa_protected_zones(kwargs),
                    )
                    if n_level_dips_repaired > 0:
                        logger.info(
                            "Verarbeitungsschritt 12 tape level stabilizer (low confidence path): %d dips repaired",
                            n_level_dips_repaired,
                        )
                # §v10.60 Modulation-Noise-Reduction: Wenn Pitch-Confidence zu niedrig
                # für Vollkorrektur ist, trotzdem konservative Envelope-Stabilisierung
                # anwenden. Reduziert das "Flattern" (10-100Hz Amplitudenmodulation)
                # das bei Kassetten/Tapes auch ohne messbare Tonhöhenschwankung auftritt.
                if _effective_strength > 0.0:
                    try:
                        from scipy.signal import butter, filtfilt

                        # §v10.14.1 Transient-preserving Envelope Stabilizer:
                        # Die ursprüngliche Implementierung (8 Hz Cutoff, 60% Blend)
                        # zerstörte Transienten: Kick-Drums (50-200 Hz Envelope-Mod)
                        # wurden als "Modulationsrauschen" fehlinterpretiert und
                        # um bis zu 30% gedämpft → onsets_orig=180, onsets_rest=15.
                        #
                        # Fix: 2 Hz Cutoff (nur sehr langsame Modulation wie
                        # Capstan-Wow), 15% Blend-Stärke, Gain-Floor 0.85 (max
                        # -15% statt -30%), und Transienten-Gate: Frames mit
                        # onset energy > 3σ des lokalen Kontexts werden preserved.
                        _mod_cutoff = 2.0  # Hz — nur Capstan-Wow (<2 Hz)
                        _mod_strength = float(np.clip(_effective_strength * 0.15, 0.05, 0.20))
                        b_mod, a_mod = butter(2, _mod_cutoff / (sample_rate / 2), btype="low")

                        def _stabilize_envelope(sig: np.ndarray) -> np.ndarray:
                            env = np.abs(sig.astype(np.float64))
                            env_smooth = filtfilt(b_mod, a_mod, env)
                            env_out = _mod_strength * env_smooth + (1.0 - _mod_strength) * env
                            gain = np.divide(env_out, env, out=np.ones_like(env), where=env > 1e-10)
                            # §v10.14.1 Transient-Gate: Onsets > 3σ → gain=1.0 (preserve)
                            _onset = np.abs(np.diff(env, prepend=env[:1]))
                            _onset_sigma = float(np.std(_onset) + 1e-10)
                            _onset_mask = _onset > 3.0 * _onset_sigma
                            # Dilate mask by 5ms to protect the full attack
                            _dilate_n = int(0.005 * sample_rate)
                            if _dilate_n > 1:
                                _kernel = np.ones(_dilate_n) / _dilate_n
                                _onset_mask = np.convolve(_onset_mask.astype(float), _kernel, mode="same") > 0.3
                            gain[_onset_mask] = 1.0
                            gain = np.clip(gain, 0.85, 1.30)  # Floor 0.85 statt 0.70
                            return (sig.astype(np.float64) * gain).astype(np.float32)

                        if audio.ndim == 2:
                            ch_first = audio.shape[0] == 2 and audio.shape[1] > 2
                            if ch_first:
                                for ch in range(audio.shape[0]):
                                    audio[ch] = _stabilize_envelope(audio[ch])
                            else:
                                for ch in range(audio.shape[1]):
                                    audio[:, ch] = _stabilize_envelope(audio[:, ch])
                        else:
                            audio = _stabilize_envelope(audio)
                        logger.debug(
                            "Verarbeitungsschritt 12: Envelope-Stabilisierung (mod=%.2f, cutoff=%.0f Hz)",
                            _mod_strength,
                            _mod_cutoff,
                        )
                    except Exception as _mod_exc:
                        logger.debug("Verarbeitungsschritt 12 Envelope nicht blockierend: %s", _mod_exc)
                audio, _rms_drop_db, _makeup_db = self._preserve_phase_loudness(
                    _original_audio,
                    audio,
                    material,
                )
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                audio = np.clip(audio, -1.0, 1.0)
                return PhaseResult(
                    success=True,
                    audio=restore_layout(audio, _was_channels_first),
                    metrics={
                        "wow_flutter_detected": False,
                        "max_deviation_percent": 0.0,
                        "correction_applied": _effective_strength if n_level_dips_repaired > 0 else 0.0,
                        "material": material.value,
                        "mean_confidence": _mean_conf,
                        "quality_mode": quality_mode,
                        "skipped_reason": "low_confidence_fallback",
                        "tape_level_dips_repaired": n_level_dips_repaired,
                    },
                    execution_time_seconds=time.time() - start_time,
                    metadata={
                        "algorithm": "confidence_guard_conservative_fallback",
                        "version": "4.2_confidence_guard_fallback",
                        "ml_hybrid": use_ml_hybrid,
                        "polyphonic": _poly_applied,
                        "phase_locality_factor": phase_locality_factor,
                        "effective_strength": _effective_strength,
                        "rms_drop_db": _rms_drop_db,
                        "loudness_makeup_db": _makeup_db,
                    },
                )
            # IF-Fallback erfolgreich → weiter mit normaler Korrekturpipeline
            # (_fit_sinusoidal_wow_curve → _separate_wow_flutter → _calculate_stretch_factors)

        pitch_trajectory, _sinusoidal_wow_profile = self._fit_sinusoidal_wow_curve(
            pitch_trajectory,
            confidence,
            sample_rate,
        )
        if _sinusoidal_wow_profile.get("applied", False):
            logger.info(
                "Verarbeitungsschritt 12 sinusförmiger Wow-Fit aktiv: freq=%.2f Hz amp=%.1f cents r2=%.2f",
                float(_sinusoidal_wow_profile.get("frequency_hz", 0.0)),
                float(_sinusoidal_wow_profile.get("amplitude_cents", 0.0)),
                float(_sinusoidal_wow_profile.get("r2", 0.0)),
            )

        # Step 1: Separate wow (<4 Hz) and flutter (4-100 Hz) components
        wow_component, flutter_component = self._separate_wow_flutter(pitch_trajectory, sample_rate)

        # Step 3: Detect significant wow/flutter (check if correction needed)
        wow_flutter_detected, max_deviation = self._detect_wow_flutter(pitch_trajectory, confidence, threshold)

        if not wow_flutter_detected:
            # No significant wow/flutter detected
            metadata: dict[str, Any] = {
                "algorithm": (
                    "polyphonic_multi_f0_consensus_v1"
                    if _poly_applied
                    else "hybrid_ml_pyin_crepe_v3"
                    if use_ml_hybrid
                    else "pyin_phase_vocoder"
                ),
                "version": "4.0_polyphonic" if _poly_applied else "3.0_ml_hybrid" if use_ml_hybrid else "3.0_pyin",
                "ml_hybrid": use_ml_hybrid,
                "polyphonic": _poly_applied,
                "sinusoidal_wow_profile": _sinusoidal_wow_profile,
            }

            if use_ml_hybrid:
                metadata["pyin_applied"] = ml_result.pyin_applied
                metadata["crepe_applied"] = ml_result.crepe_applied
                metadata["strategy_used"] = str(ml_result.strategy_used)
                metadata["ml_metadata"] = ml_result.metadata

            # Step 6c (also in no-wow/flutter path): Tape level stabilization
            n_level_dips_repaired = 0
            _TAPE_LEVEL_MATERIALS = {MaterialType.TAPE, MaterialType.REEL_TAPE, MaterialType.CASSETTE}
            _mat_enum = material if isinstance(material, MaterialType) else None
            _dl_high = kwargs.get("defect_locations") or {}
            _has_tape_dip_defect = bool(_dl_high.get("tape_head_level_dip") or _dl_high.get("transport_bump"))
            _is_primary_tape = _mat_enum in _TAPE_LEVEL_MATERIALS
            if (_is_primary_tape or _has_tape_dip_defect) and _effective_strength > 0.0:
                audio, n_level_dips_repaired = self._stabilize_tape_level(
                    audio,
                    sample_rate,
                    _effective_strength,
                    is_primary_tape=_is_primary_tape,
                    confirmed_tape_dip=_has_tape_dip_defect,
                    protected_zones=self._collect_vfa_protected_zones(kwargs),
                )
                if n_level_dips_repaired > 0:
                    logger.info(
                        "Verarbeitungsschritt 12 tape level stabilizer (no wow/flutter): %d dips repaired",
                        n_level_dips_repaired,
                    )

            audio, _rms_drop_db, _makeup_db = self._preserve_phase_loudness(
                _original_audio,
                audio,
                material,
            )

            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=restore_layout(audio, _was_channels_first),
                metrics={
                    "wow_flutter_detected": False,
                    "max_deviation_percent": max_deviation,
                    "correction_applied": _effective_strength if n_level_dips_repaired > 0 else 0.0,
                    "material": material.value,
                    "mean_confidence": float(np.mean(confidence[confidence > 0])),
                    "quality_mode": quality_mode,
                    "tape_level_dips_repaired": n_level_dips_repaired,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    **metadata,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": _rms_drop_db,
                    "loudness_makeup_db": _makeup_db,
                },
            )

        vocals_conf = float(kwargs.get("panns_vocals_confidence", 0.0))
        if vocals_conf == 0.0:  # Fallback: direct callers may use panns_singing key
            vocals_conf = float(kwargs.get("panns_singing", 0.0))
        _timing_safe_strength_scale, _max_stretch_delta = self._derive_safe_timing_profile(
            material,
            _mean_conf,
            vocals_conf,
            polyphonic_fallback=_poly_fallback,
        )
        _timing_safe_strength = float(np.clip(strength * _timing_safe_strength_scale, 0.0, max(0.15, strength)))

        if self._should_bypass_unsafe_polyphonic_fallback(
            material,
            _mean_conf,
            vocals_conf,
            polyphonic_fallback=_poly_fallback,
        ):
            restored = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            restored = np.clip(restored, -1.0, 1.0)
            processing_time = time.time() - start_time
            return PhaseResult(
                success=True,
                audio=restore_layout(restored, _was_channels_first),
                metrics={
                    "wow_flutter_detected": True,
                    "max_deviation_percent": max_deviation,
                    "residual_deviation_percent": max_deviation,
                    "wow_magnitude_percent": 0.0,
                    "flutter_magnitude_percent": 0.0,
                    "correction_strength": 0.0,
                    "mean_confidence": float(_mean_conf),
                    "material": material.value,
                    "quality_mode": quality_mode,
                    "transport_bumps_repaired": 0,
                    "tape_level_dips_repaired": 0,
                },
                execution_time_seconds=processing_time,
                metadata={
                    "algorithm": "unsafe_polyphonic_fallback_bypass",
                    "version": "4.1_locality",
                    "ml_hybrid": False,
                    "psola_active": vocals_conf >= 0.4,
                    "panns_vocals_confidence": vocals_conf,
                    "threshold": threshold,
                    "polyphonic_fallback": _poly_fallback,
                    "timing_safe_strength": 0.0,
                    "timing_safe_strength_scale": _timing_safe_strength_scale,
                    "max_stretch_delta": _max_stretch_delta,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "bypassed_due_to_unsafe_polyphonic_fallback": True,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[
                    "Wow/flutter correction bypassed: polyphonic estimator was"
                    " implausible and fallback was unsafe for vocal analog material"
                ],
            )

        # Step 4: Calculate time-stretching factors from pitch deviation
        # §2.46b: Computed wow severity for adaptive pitch-span threshold
        self._mat_type_for_stretch = str(material_type)
        _wow_sev = float(kwargs.get("wow_severity", kwargs.get("flutter_severity", 0.0)) or 0.0)
        # Fallback: restoration_context → conservative default for analog tape
        if _wow_sev == 0.0:
            _ctx = getattr(self, "_restoration_context", {}) or {}
            _wow_sev = float(_ctx.get("wow_severity", _ctx.get("flutter_severity", 0.0)) or 0.0)
        if _wow_sev == 0.0 and str(material_type).lower() in ("cassette", "reel_tape", "tape", "vinyl"):
            _wow_sev = 0.30  # Conservative minimum for analog tape/disc
        self._wow_sev_for_stretch = float(_wow_sev)
        stretch_factors = self._calculate_stretch_factors(
            pitch_trajectory,
            confidence,
            _timing_safe_strength,
            max_stretch_delta=_max_stretch_delta,
        )
        # §P5 Wow/Flutter getrennt korrigieren: Wow (<4 Hz, langsam) und Flutter
        # (4-100 Hz, schnell) haben verschiedene physikalische Ursachen und brauchen
        # unterschiedliche Korrektur-Strategien. Wow (Capstan-Unwucht) → stärkere
        # und breitere Korrektur. Flutter (mechanische Vibration) → konservativere
        # Korrektur mit engerem Stretch-Delta.
        _wow_stretch = self._calculate_stretch_factors(
            pitch_trajectory,
            confidence,
            _timing_safe_strength * 1.10,  # wow: 10 % more aggressive
            max_stretch_delta=_max_stretch_delta,
        )
        _flutter_stretch = self._calculate_stretch_factors(
            pitch_trajectory,
            confidence,
            _timing_safe_strength * 0.70,  # flutter: 30 % more conservative
            max_stretch_delta=_max_stretch_delta * 0.60,  # narrower delta for fast variations
        )
        # Blend: 55 % wow + 45 % flutter (wow is the dominant perceptual component)
        stretch_factors = _wow_stretch * 0.55 + _flutter_stretch * 0.45
        _locality_coverage = 0.0
        stretch_factors, _locality_coverage = self._apply_defect_locality_to_stretch_factors(
            stretch_factors,
            audio_length_samples=audio_sample_count(audio),
            sample_rate=sample_rate,
            defect_locations=kwargs.get("defect_locations"),
        )

        # Step 5: Apply time-stretching – PSOLA für Vokal-Segmente, WSOLA sonst
        # Moulines & Charpentier (1990): PSOLA ist formanterhaltend bei Gesangsmaterial;
        # Phase-Vocoder (hier: WSOLA/resample) für Instrumental-/Nicht-Vokal-Material.
        _report_progress(65.0, "Wow/Flutter: Zeitstreckung (PSOLA/Phase-Vocoder) läuft...")
        _stretch_fn = self._psola_timestretch if vocals_conf >= 0.4 else self._phase_vocoder_timestretch
        # §DDSP-Harmonic: HPSS-Isolation aktiviert für Vokal + hochrauschige Medien.
        # Zeitstreckung nur auf harmonischen Anteil → Rausch-Textur (Tape-Hiss, Crackle)
        # bleibt zeitlich unverändert (physikalisch authentisch). Non-blocking — Fallback
        # auf standard Phase-Vocoder wenn HPSS < Qualitäts-Gate.
        _use_harm_iso = (
            vocals_conf >= 0.25
            and material in {MaterialType.TAPE, MaterialType.CASSETTE}
            and len(stretch_factors) > 0
            and float(np.max(np.abs(np.asarray(stretch_factors, dtype=np.float32) - 1.0))) < 0.08
        )
        if vocals_conf >= 0.4:
            logger.debug(
                "Verarbeitungsschritt 12: PSOLA aktiviert (PANNs Vocals-Konfidenz=%.2f ≥ 0.40)",
                vocals_conf,
            )
        if is_stereo:
            # §2.51 M/S-Domain Stereo Processing — verhindert L/R Zeitversatz.
            #
            # ROOT CAUSE des Zeitversatzes (und der daraus folgenden Pegelexplosion):
            # _psola_timestretch() schätzt Pitch-Perioden (pYIN) UNABHÄNGIG pro Kanal.
            # L und R haben leicht unterschiedlichen Inhalt → verschiedene period_samps →
            # OLA-Grain-Grenzen weichen pro Frame ab → kumulativer Zeitversatz über den Song
            # → L/R Korrelation sinkt von +0.9 auf ca. -0.10 (anti-phasig).
            #
            # Folge-Kaskade (Pegelexplosion Intro/Outro):
            # 1. Anti-Phasen-L/R → Mono-Downmix (L+R)/2 zeigt deutlich weniger Pegel
            # 2. MDEM / correct_arc / AQI messen "Pegelabfall" via Mono-Downmix
            # 3. Makeup-Gain wird auf alle Frames ausgelöst, inkl. Intro-Vinyl-Rauschen
            #    und Outro-Fadeout → Pegelexplosion in Nicht-Musik-Bereichen
            #
            # FIX (v10.0.0): Phase-Vocoder für BEIDE M/S-Kanäle erzwingen.
            #   Mid = (L+R)/2 → Phase-Vocoder (sample-genaue Zeitkorrektur)
            #   Side = (L-R)/2 → Phase-Vocoder (identisches src_pos-Mapping wie Mid!)
            #   L_out = Mid_out + Side_out = L_in[src_pos[t]]  (mathematisch exakt)
            #   R_out = Mid_out - Side_out = R_in[src_pos[t]]  (mathematisch exakt)
            # Beide Kanäle erhalten dasselbe src_pos-Mapping → ZERO L/R Zeitversatz.
            #
            # §v10.12 PSOLA-Side: Wenn PSOLA für Mid aktiv ist (vocals_conf >= 0.40),
            # wird PSOLA auch für Side verwendet — mit denselben Pitch-Marken vom Mid-
            # Kanal (§P4). Das erhält die Formant-Treue im räumlichen Anteil (Hall,
            # Sibilanten, Stereobreite) und verhindert den Kohärenzverlust zwischen
            # Direktsignal und Raum.
            _left_ch, _right_ch = stereo_channel_view(audio)
            _mid_ch = (_left_ch.astype(np.float32) + _right_ch.astype(np.float32)) * 0.5
            _side_ch = (_left_ch.astype(np.float32) - _right_ch.astype(np.float32)) * 0.5
            # §2.51 L/R-Timing-Invariante (v10.0.0): BEIDE M/S-Kanäle MÜSSEN denselben
            # Algorithmus verwenden.  PSOLA (OLA-Grain-Grenzen, pYIN-Perioden) und
            # Phase-Vocoder (np.interp-Sample-Remapping) haben verschiedene effektive
            # Zeitauflösungen → L = Mid+Side und R = Mid-Side erhalten zeitlich inkohärente
            # Summanden → sichtbarer L/R Zeitversatz im Wellenformbild.
            # Fix: Phase-Vocoder für Mid UND Side im Stereo-M/S-Pfad.
            # §P4: Bei Vokal-Aktivierung PSOLA für BEIDE M/S-Kanäle (gleiche Pitch-Marken).
            _mid_stretched = (
                self._harmonic_isolated_timestretch(_mid_ch, stretch_factors, sample_rate)
                if _use_harm_iso
                else (_stretch_fn(_mid_ch, stretch_factors, sample_rate))
            )
            # Side: SELBER Algorithmus wie Mid (§2.51 Timing-Invariante, v10.13 fix).
            # Unterschiedliche Algorithmen (PSOLA vs Phase-Vocoder) erzeugen
            # verschiedene effektive Zeitauflösungen → L/R-Zeitversatz beim
            # M/S-Rekomponieren. Der Algorithmus wird allein durch _use_harm_iso
            # bzw. _stretch_fn bestimmt, identisch für Mid und Side.
            _side_stretched = (
                self._harmonic_isolated_timestretch(_side_ch, stretch_factors, sample_rate)
                if _use_harm_iso
                else _stretch_fn(_side_ch, stretch_factors, sample_rate)
            )
            # §2.51 Amplitudenkorrektur: PSOLA ist NICHT amplitudenerhaltend.
            # OLA-Windowing dämpft das Mid-Signal typisch um 5–8 dB → MDEM/correct_arc
            # messen diesen Drop im Mono-Downmix und triggern Makeup-Gain auf ALLE Frames
            # inkl. Intro-Rauschen/Outro-Fade → Pegelexplosion.
            # Fix: Mid-RMS nach PSOLA auf Eingabe-RMS normalisieren (max ±6 dB).
            _mid_rms_in = float(np.sqrt(np.mean(_mid_ch**2) + 1e-12))
            _n_ms = min(len(_mid_stretched), len(_side_stretched))
            _mid_rms_out = float(np.sqrt(np.mean(_mid_stretched[:_n_ms] ** 2) + 1e-12))
            if _mid_rms_in > 1e-9 and _mid_rms_out > 1e-9:
                _mid_norm_gain = float(np.clip(_mid_rms_in / _mid_rms_out, 0.5, 2.0))  # ±6 dB
                _mid_stretched = np.clip(_mid_stretched * _mid_norm_gain, -1.0, 1.0)
                logger.debug(
                    "Verarbeitungsschritt_12: M/S Mid-RMS-Normalisierung: in=%.1f dBFS out=%.1f dBFS gain=%.1f dB",
                    20.0 * np.log10(_mid_rms_in + 1e-12),
                    20.0 * np.log10(_mid_rms_out + 1e-12),
                    20.0 * np.log10(_mid_norm_gain + 1e-12),
                )
            restored_left = (_mid_stretched[:_n_ms] + _side_stretched[:_n_ms]).astype(audio.dtype)
            restored_right = (_mid_stretched[:_n_ms] - _side_stretched[:_n_ms]).astype(audio.dtype)
            _p12_n = min(len(restored_left), len(restored_right), audio_sample_count(audio))
            if _p12_n < audio_sample_count(audio):
                logger.debug(
                    "Verarbeitungsschritt_12: M/S-Längenangleichung: orig=%d → %d",
                    audio_sample_count(audio),
                    _p12_n,
                )
            restored = stereo_like(restored_left[:_p12_n], restored_right[:_p12_n], audio)
        else:
            if _use_harm_iso:
                # §DDSP-Harmonic: Zeitstreckung nur auf harmonischen Anteil.
                # Rausch-Residuum (Tape-Hiss, Crackle) bleibt zeitlich unverändert.
                restored = self._harmonic_isolated_timestretch(audio, stretch_factors, sample_rate)
            else:
                restored = _stretch_fn(audio, stretch_factors, sample_rate)

        # §C3 Neural Phase Vocoder — post-stretch phase coherence restoration.
        # PSOLA/Phase-Vocoder time-stretching can introduce phase incoherence in
        # voiced regions (harmonic misalignment). Apply PGHI-consistent phase
        # regularisation to restore natural harmonic phase relationships.
        try:
            _orig_mono = safe_to_mono(audio).astype(np.float64) if is_stereo else np.asarray(audio, dtype=np.float64)
            if is_stereo:
                _res_mid = safe_to_mono(restored).astype(np.float64)
                _res_mid_coh = self._apply_neural_phase_coherence(_res_mid, sample_rate, reference=_orig_mono)
                _coh_diff = _res_mid_coh - _res_mid
                _res_left, _res_right = stereo_channel_view(restored)
                _n_coh = min(len(_res_left), len(_res_right), len(_coh_diff))
                _left_coh = _res_left.copy()
                _right_coh = _res_right.copy()
                _left_coh[:_n_coh] += _coh_diff[:_n_coh] * 0.5
                _right_coh[:_n_coh] += _coh_diff[:_n_coh] * 0.5
                restored = stereo_like(_left_coh, _right_coh, restored)
                restored = np.clip(restored, -1.0, 1.0)
            else:
                _rest_ref = np.asarray(audio, dtype=np.float64) if len(audio) == len(restored) else None
                restored = self._apply_neural_phase_coherence(restored, sample_rate, reference=_rest_ref)
        except Exception as _c3_proc_exc:
            logger.debug("§C3 Verarbeitungsschritt coherence integration nicht blockierend: %s", _c3_proc_exc)

        _report_progress(82.0, "Wow/Flutter: Zeitstreckung abgeschlossen")
        # Step 6: Verify correction (measure residual deviation)
        restored_mono = safe_to_mono(restored) if is_stereo else restored

        # Step 6b: Targeted transport bump repair (impulsive micro-speed jumps 50–300 ms)
        bump_locations = kwargs.get("transport_bump_locations")
        if not bump_locations:
            # Fallback: extract from defect_locations dict (UV3 passes defect_locations kwarg)
            _dl = kwargs.get("defect_locations") or {}
            bump_locations = _dl.get("transport_bump", []) if isinstance(_dl, dict) else []
        n_bumps_repaired = 0
        if bump_locations and len(bump_locations) > 0:
            # Confidence-aware bump repair: keep defect removal active while avoiding
            # over-processing when pitch certainty is marginal.
            _bump_strength = float(
                np.clip(
                    _timing_safe_strength * (0.65 + 0.35 * np.clip(_mean_conf, 0.0, 1.0)),
                    0.15,
                    max(0.15, _timing_safe_strength),
                )
            )
            if len(bump_locations) > 120:
                # Bei vielen Bumps: Kassette/Tape-Material eher stärker reparieren
                # (viele echte Laufwerks-Events), nicht abschwächen.
                _material_for_bump = kwargs.get("material_type")
                _mat_val_bump = getattr(_material_for_bump, "value", str(_material_for_bump)).lower()
                if "cassette" in _mat_val_bump or "tape" in _mat_val_bump:
                    _bump_strength = min(_bump_strength * 1.20, max(0.15, _timing_safe_strength))
                else:
                    _bump_strength *= 0.85
            # Schutzzonen für per-Bump individuelle Stärke zusammenstellen (§0p Vocal-Supremacy)
            _p12_protected_zones: list[tuple[float, float, float]] = []
            for _z in kwargs.get("vibrato_zones") or []:
                try:
                    _p12_protected_zones.append((float(_z[0]), float(_z[1]), 0.20))  # §0p Vibrato-Schutz
                except Exception:
                    logger.debug("_report_progress: silent except suppressed", exc_info=True)
            for _z in kwargs.get("frisson_zones") or []:
                try:
                    _fz_s = float(getattr(_z, "start_s", None) or _z[0])
                    _fz_e = float(getattr(_z, "end_s", None) or _z[1])
                    _p12_protected_zones.append((_fz_s, _fz_e, 0.30))  # Frisson sakrosankt
                except Exception:
                    logger.debug("_report_progress: silent except suppressed", exc_info=True)
            for _z in kwargs.get("whisper_zones") or []:
                try:
                    _p12_protected_zones.append((float(_z[0]), float(_z[1]), 0.25))  # Flüsterpassagen
                except Exception:
                    logger.debug("_report_progress: silent except suppressed", exc_info=True)
            for _z in kwargs.get("passaggio_zones") or []:
                try:
                    _p12_protected_zones.append((float(_z[0]), float(_z[1]), 0.35))  # Passaggio-Übergänge
                except Exception:
                    logger.debug("_report_progress: silent except suppressed", exc_info=True)
            restored, n_bumps_repaired = self._repair_transport_bumps(
                restored,
                sample_rate,
                bump_locations,
                _bump_strength,
                protected_zones=_p12_protected_zones or None,
            )
            restored_mono = safe_to_mono(restored) if is_stereo else restored
            logger.info(
                "Verarbeitungsschritt 12 transport_bump repair: %d/%d bumps repaired (strength=%.3f, conf=%.3f)",
                n_bumps_repaired,
                len(bump_locations),
                _bump_strength,
                _mean_conf,
            )

        # Step 6c: Tape head contact level stabilization (autonomous detection + repair)
        # Repairs gradual level dips caused by tape-head pressure variation / capstan
        # irregularity in cassette recordings.  These dips fall through Phase 24 dropout
        # repair (threshold too aggressive for gradual 60-100 ms fades) and are not
        # covered by transport_bump repair (which requires DefectScanner locations).
        # Also triggers for multi-chain material (e.g. vinyl→tape→mp3) when
        # TAPE_HEAD_LEVEL_DIP was detected via defect_locations (§2.46a).
        n_level_dips_repaired = 0
        _TAPE_LEVEL_MATERIALS = {MaterialType.TAPE, MaterialType.REEL_TAPE, MaterialType.CASSETTE}
        _mat_enum = material if isinstance(material, MaterialType) else None
        _report_progress(90.0, "Wow/Flutter: Impuls-Reparaturen abgeschlossen")
        _dl_post = kwargs.get("defect_locations") or {}
        _has_tape_dip_defect = bool(_dl_post.get("tape_head_level_dip") or _dl_post.get("transport_bump"))
        _is_primary_tape = _mat_enum in _TAPE_LEVEL_MATERIALS
        if (_is_primary_tape or _has_tape_dip_defect) and _effective_strength > 0.0:
            restored, n_level_dips_repaired = self._stabilize_tape_level(
                restored,
                sample_rate,
                _effective_strength,
                is_primary_tape=_is_primary_tape,
                confirmed_tape_dip=_has_tape_dip_defect,
                protected_zones=self._collect_vfa_protected_zones(kwargs),
            )
            if n_level_dips_repaired > 0:
                restored_mono = safe_to_mono(restored) if is_stereo else restored
                logger.info(
                    "Verarbeitungsschritt 12 tape level stabilizer: %d dips repaired",
                    n_level_dips_repaired,
                )

        residual_pitch, residual_conf = self._estimate_pitch_yin(restored_mono, sample_rate)
        residual_deviation = self._calculate_max_deviation(residual_pitch, residual_conf)

        # Chroma Pearson rollback guard: if tonal center drifts, revert to original
        # (Phase Vocoder / PSOLA can introduce pitch shifts that destroy tonal center)
        try:
            _orig_mono = safe_to_mono(audio) if is_stereo else audio
            _n_chroma = min(len(_orig_mono), len(restored_mono))
            _hop_chroma = 512
            _n_frames = max(1, _n_chroma // _hop_chroma)
            _chroma_orig = np.zeros(12, dtype=np.float64)
            _chroma_rest = np.zeros(12, dtype=np.float64)
            for _ci in range(min(_n_frames, 200)):  # sample up to 200 frames
                _s = _ci * _hop_chroma
                _e = _s + _hop_chroma
                if _e > _n_chroma:
                    break
                _sp_o = np.abs(np.fft.rfft(_orig_mono[_s:_e]))
                _sp_r = np.abs(np.fft.rfft(restored_mono[_s:_e]))
                _freqs_c = np.fft.rfftfreq(_hop_chroma, 1.0 / sample_rate)
                for _b in range(12):
                    _f_lo = 65.41 * (2 ** (_b / 12.0))
                    _f_hi = 65.41 * (2 ** ((_b + 1) / 12.0))
                    _mask_c = (_freqs_c >= _f_lo) & (_freqs_c < _f_hi)
                    _chroma_orig[_b] += np.sum(_sp_o[_mask_c] ** 2)
                    _chroma_rest[_b] += np.sum(_sp_r[_mask_c] ** 2)
            _norm_o = np.sqrt(np.sum(_chroma_orig**2)) + 1e-10
            _norm_r = np.sqrt(np.sum(_chroma_rest**2)) + 1e-10
            # §2.44: Pearson-Korrelation der 12 Chroma-Bins (tonal-center guard).
            _co_mean = float(_chroma_orig.mean())
            _cr_mean = float(_chroma_rest.mean())
            _pearson_num = float(np.sum((_chroma_orig - _co_mean) * (_chroma_rest - _cr_mean)))
            _pearson_den = float(
                np.sqrt(np.sum((_chroma_orig - _co_mean) ** 2) * np.sum((_chroma_rest - _cr_mean) ** 2))
            )
            _chroma_pearson = _pearson_num / _pearson_den if _pearson_den > 1e-12 else 1.0
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            _chroma_pearson = 1.0  # fallback: assume OK

        if _chroma_pearson < 0.95:
            logger.warning(
                "Verarbeitungsschritt 12 chroma guard: Pearson %.3f < 0.95 — reverting to Originalsignal audio "
                "(wow/flutter correction caused tonal center drift)",
                _chroma_pearson,
            )
            restored = audio.copy()
            residual_deviation = max_deviation  # unchanged

        processing_time = time.time() - start_time

        # Calculate wow/flutter statistics
        wow_magnitude = np.std(wow_component[wow_component != 0])
        flutter_magnitude = np.std(flutter_component[flutter_component != 0])

        # Build metadata (wow/flutter detected path — avoid redefinition of 'metadata' from line 589)
        _meta_detected: dict[str, Any] = {
            "algorithm": (
                "polyphonic_multi_f0_consensus_v1"
                if _poly_applied
                else (
                    "hybrid_ml_pyin_crepe_psola_v3"
                    if (use_ml_hybrid and vocals_conf >= 0.4)
                    else (
                        "hybrid_ml_pyin_crepe_v3"
                        if use_ml_hybrid
                        else "pyin_psola"
                        if vocals_conf >= 0.4
                        else "pyin_phase_vocoder"
                    )
                )
            ),
            "version": "4.0_polyphonic" if _poly_applied else "3.0_ml_hybrid" if use_ml_hybrid else "3.0_pyin",
            "ml_hybrid": use_ml_hybrid,
            "psola_active": vocals_conf >= 0.4,
            "panns_vocals_confidence": vocals_conf,
            "threshold": threshold,
            "stft_window": self.STFT_WINDOW_SIZE,
            "stft_hop": self.STFT_HOP_SIZE,
            "polyphonic_fallback": _poly_fallback,
            "timing_safe_strength": _timing_safe_strength,
            "timing_safe_strength_scale": _timing_safe_strength_scale,
            "max_stretch_delta": _max_stretch_delta,
            "sinusoidal_wow_profile": _sinusoidal_wow_profile,
        }

        if use_ml_hybrid:
            _meta_detected["pyin_applied"] = ml_result.pyin_applied
            _meta_detected["crepe_applied"] = ml_result.crepe_applied
            _meta_detected["strategy_used"] = str(ml_result.strategy_used)
            _meta_detected["pitch_detection_time"] = ml_result.processing_time
            _meta_detected["ml_metadata"] = ml_result.metadata

        restored = np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.clip(restored, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            restored = audio + _effective_strength * (restored - audio)
            restored = np.clip(restored, -1.0, 1.0)

        # §2.46f NPA-Guard: Natürliches Vibrato/Portamento (F0-Mod 4–7 Hz, ≤±50 Cent)
        # darf nicht durch Wow/Flutter-Korrektur geglättet werden — Pitch-Segmente restaurieren.
        try:
            from backend.core.natural_performance_detector import get_natural_performance_detector

            _mono12 = safe_to_mono(_original_audio) if _original_audio.ndim == 2 else _original_audio
            _npa_mask12 = (
                get_natural_performance_detector()
                .detect(_mono12, sample_rate)
                .get_protected_mask(len(_mono12), sample_rate)
            )
            if _npa_mask12 is not None and _npa_mask12.any():
                if restored.ndim == 2:
                    _rest_left, _rest_right = stereo_channel_view(restored)
                    _orig_left, _orig_right = stereo_channel_view(_original_audio)
                    _n_mask = min(
                        len(_npa_mask12), len(_rest_left), len(_rest_right), len(_orig_left), len(_orig_right)
                    )
                    _mask = _npa_mask12[:_n_mask]
                    _left_guarded = _rest_left.copy()
                    _right_guarded = _rest_right.copy()
                    _mask_idx = np.flatnonzero(_mask)
                    _left_guarded[_mask_idx] = _orig_left[_mask_idx]
                    _right_guarded[_mask_idx] = _orig_right[_mask_idx]
                    restored = stereo_like(_left_guarded, _right_guarded, restored)
                else:
                    restored[_npa_mask12] = _original_audio[_npa_mask12]
        except Exception as _npa12_exc:
            logger.debug("§2.46f Verarbeitungsschritt12 NPA-Guard (nicht blockierend): %s", _npa12_exc)

        restored, _rms_drop_db, _makeup_db = self._preserve_phase_loudness(
            _original_audio,
            restored,
            material,
        )
        # §GGB-1: Global Gain Budget — cap cumulative makeup gain
        if _makeup_db > 0.0:
            try:
                from backend.core.global_gain_budget import get_global_gain_budget

                _approved = get_global_gain_budget().request("phase_12_wow_flutter_fix", _makeup_db, priority="normal")
                if _approved < _makeup_db:
                    _scale = float(10.0 ** ((_approved - _makeup_db) / 20.0))
                    restored = restored * _scale + _original_audio * (1.0 - _scale)
                    restored = np.clip(restored, -1.0, 1.0)
                    _makeup_db = _approved
            except Exception as _e:
                logger.debug("%s: unkritisch exception: %s", __name__, _e)
        if abs(_makeup_db) > 0.01:
            logger.info(
                "Verarbeitungsschritt 12 loudness-preservation: material=%s rms_drop=%+.2f dB via makeup %+.2f dB",
                material.value,
                _rms_drop_db,
                _makeup_db,
            )
        _report_progress(96.0, "Wow/Flutter: Abschluss-Validierung")
        return PhaseResult(
            success=True,
            audio=restore_layout(restored, _was_channels_first),
            resolved_defects={
                "WOW": float(np.clip(residual_deviation / 100.0, 0.0, 0.3)),
                "FLUTTER": float(np.clip(residual_deviation / 100.0, 0.0, 0.3)),
            },
            metrics={
                "wow_flutter_detected": True,
                "max_deviation_percent": max_deviation,
                "residual_deviation_percent": residual_deviation,
                "wow_magnitude_percent": float(wow_magnitude) if not np.isnan(wow_magnitude) else 0.0,
                "flutter_magnitude_percent": float(flutter_magnitude) if not np.isnan(flutter_magnitude) else 0.0,
                "correction_strength": _timing_safe_strength,
                "mean_confidence": float(np.mean(confidence[confidence > 0])),
                "material": material.value,
                "quality_mode": quality_mode,
                "transport_bumps_repaired": n_bumps_repaired,
                "tape_level_dips_repaired": n_level_dips_repaired,
            },
            execution_time_seconds=processing_time,
            metadata={
                **_meta_detected,
                "phase_locality_factor": phase_locality_factor,
                "stretch_locality_coverage": _locality_coverage,
                "effective_strength": _effective_strength,
                "rms_drop_db": _rms_drop_db,
                "loudness_makeup_db": _makeup_db,
            },
        )

    def _fit_sinusoidal_wow_curve(
        self,
        pitch_trajectory: np.ndarray,
        confidence: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Glättet dominante sinusförmige Wow-Modulationen in der Pitch-Kurve.

        Der Fit ist absichtlich konservativ: Er greift nur bei kleiner Pitch-Spanne
        (Transportfehler, keine Melodie) und einem klaren 0.1-3 Hz Peak.
        """
        profile: dict[str, Any] = {
            "applied": False,
            "frequency_hz": 0.0,
            "amplitude_cents": 0.0,
            "r2": 0.0,
        }
        pitch = np.asarray(pitch_trajectory, dtype=np.float64)
        conf = np.asarray(confidence, dtype=np.float64)
        if pitch.size < 32 or conf.size != pitch.size:
            return pitch_trajectory, profile

        confident_mask = (pitch > 0.0) & (conf > 0.5)
        if int(np.sum(confident_mask)) < 24:
            return pitch_trajectory, profile

        confident_pitches = pitch[confident_mask]
        p5 = float(np.percentile(confident_pitches, 5))
        p95 = float(np.percentile(confident_pitches, 95))
        if p5 <= 0.0 or p95 <= p5:
            return pitch_trajectory, profile
        span_cents = float(1200.0 * np.log2(p95 / p5))
        if span_cents > 100.0:
            return pitch_trajectory, profile

        median_pitch = float(np.median(confident_pitches))
        if median_pitch <= 0.0:
            return pitch_trajectory, profile

        residual_cents = np.zeros_like(pitch, dtype=np.float64)
        valid_idx = np.flatnonzero(pitch > 0.0)
        residual_cents[valid_idx] = 1200.0 * np.log2(np.maximum(pitch[valid_idx], 1e-8) / median_pitch)
        conf_idx = np.flatnonzero(confident_mask)
        if conf_idx.size < 24:
            return pitch_trajectory, profile
        residual_interp = np.interp(np.arange(pitch.size), conf_idx, residual_cents[conf_idx])
        residual_interp = residual_interp - float(np.median(residual_interp))

        window_samples = int(self.PITCH_WINDOW_MS * sample_rate / 1000)
        hop_samples = max(1, window_samples // self.PITCH_HOP_FACTOR)
        frame_rate = float(sample_rate) / float(hop_samples)
        freqs = np.fft.rfftfreq(pitch.size, d=1.0 / frame_rate)
        window = np.hanning(pitch.size)
        spectrum = np.fft.rfft(residual_interp * window)
        band_mask = (freqs >= 0.10) & (freqs <= 3.0)
        if not np.any(band_mask):
            return pitch_trajectory, profile
        band_indices = np.flatnonzero(band_mask)
        band_power = np.abs(spectrum[band_indices]) ** 2
        if band_power.size == 0 or float(np.sum(band_power)) <= 1e-12:
            return pitch_trajectory, profile
        peak_local = int(np.argmax(band_power))
        peak_index = int(band_indices[peak_local])
        peak_freq = float(freqs[peak_index])
        dominance = float(band_power[peak_local] / (np.sum(band_power) + 1e-12))
        if dominance < 0.45:
            return pitch_trajectory, profile

        t = np.arange(pitch.size, dtype=np.float64) / frame_rate
        omega_t = 2.0 * np.pi * peak_freq * t[confident_mask]
        design = np.column_stack([np.sin(omega_t), np.cos(omega_t), np.ones(int(np.sum(confident_mask)))])
        y = residual_cents[confident_mask]
        try:
            coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        except Exception as e:
            logger.warning("Verarbeitungsschritt_12_wow_flutter_fix.py::_fit_sinusoidal_wow_curve Ersatzpfad: %s", e)
            return pitch_trajectory, profile
        amp_cents = float(np.hypot(coeffs[0], coeffs[1]))
        if amp_cents < 4.0 or amp_cents > 60.0:
            return pitch_trajectory, profile

        fitted_conf = design @ coeffs
        ss_res = float(np.sum((y - fitted_conf) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2) + 1e-12)
        r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
        if r2 < 0.45:
            return pitch_trajectory, profile

        full_design = np.column_stack(
            [
                np.sin(2.0 * np.pi * peak_freq * t),
                np.cos(2.0 * np.pi * peak_freq * t),
                np.ones(pitch.size),
            ]
        )
        fitted_cents = full_design @ coeffs
        fitted_pitch = median_pitch * np.power(2.0, fitted_cents / 1200.0)
        smoothed = pitch.copy()
        smoothed[pitch > 0.0] = fitted_pitch[pitch > 0.0]
        profile.update(
            {
                "applied": True,
                "frequency_hz": peak_freq,
                "amplitude_cents": amp_cents,
                "r2": r2,
                "dominance": dominance,
                "span_cents": span_cents,
            }
        )
        return smoothed.astype(pitch_trajectory.dtype, copy=False), profile

    def _preserve_phase_loudness(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, float, float]:
        """Keep Phase-12 loudness close to input while preserving defect repairs.

        Returns:
            (audio_out, rms_delta_db, applied_makeup_db)
        """
        orig = np.asarray(original_audio, dtype=np.float64)
        proc = np.asarray(processed_audio, dtype=np.float64)
        if orig.shape != proc.shape:
            return np.clip(np.asarray(processed_audio, dtype=np.float32), -1.0, 1.0), 0.0, 0.0

        # §v10.13 SOTA: Normierte XCorr, kein ≥15s-Guard nötig.
        _stcg_applied = False
        if proc.ndim == 2:
            try:
                from backend.core.stereo_temporal_coherence_guard import get_stereo_temporal_coherence_guard

                proc_aligned = get_stereo_temporal_coherence_guard().correct_interchannel_delay(
                    proc.astype(np.float32),
                    48000,
                    phase_id="phase_12_wow_flutter_fix",
                )
                proc = np.asarray(proc_aligned, dtype=np.float64)
                _stcg_applied = True
            except Exception as exc:
                logger.debug("Verarbeitungsschritt 12 loudness-preservation: STCG uebersprungen (%s)", exc)

        # §2.51 Stereo-lag xcorr fallback — NUR wenn STCG fehlgeschlagen ist.
        # GCC-PHAT (used by STCG) fails for narrow-band (near-sinusoidal) audio.
        # Der Fallback wird NUR als Notlösung verwendet, NICHT als konkurrierende
        # Zweitkorrektur nach erfolgreichem STCG-Lauf.
        if proc.ndim == 2 and proc.shape[0] >= 1024 and not _stcg_applied:
            try:
                _n_onset = min(proc.shape[0], 4096)
                _cl = proc[:_n_onset, 0].astype(np.float64)
                _cr = proc[:_n_onset, 1].astype(np.float64)
                # Per-channel RMS envelope (4-sample blocks) to find first active block.
                _block = 4
                _env_l = np.array(
                    [float(np.sqrt(np.mean(_cl[i : i + _block] ** 2))) for i in range(0, _n_onset - _block, _block)]
                )
                _env_r = np.array(
                    [float(np.sqrt(np.mean(_cr[i : i + _block] ** 2))) for i in range(0, _n_onset - _block, _block)]
                )
                _global_rms = max(float(np.sqrt(np.mean(_cl**2 + _cr**2))), 1e-10)
                _thresh = _global_rms * 0.05  # 5 % of global RMS = "active"
                # Find first active block in each channel.
                _act_l = int(np.argmax(_env_l > _thresh)) if np.any(_env_l > _thresh) else len(_env_l)
                _act_r = int(np.argmax(_env_r > _thresh)) if np.any(_env_r > _thresh) else len(_env_r)
                _lag_samples = (_act_r - _act_l) * _block  # positive = R is delayed
                _max_corr_lag = 2048
                if abs(_lag_samples) > 2 and abs(_lag_samples) <= _max_corr_lag:
                    # _lag_samples > 0 → R delayed → advance R (remove leading samples)
                    # _lag_samples < 0 → L delayed → delay R (add leading zeros)
                    _N_proc = proc.shape[0]
                    if _lag_samples > 0:
                        _new_r = np.concatenate([proc[_lag_samples:, 1], np.zeros(_lag_samples)])
                    else:
                        _new_r = np.concatenate([np.zeros(-_lag_samples), proc[: _N_proc + _lag_samples, 1]])
                    proc = np.column_stack([proc[:, 0], _new_r])
                    logger.debug(
                        "§2.51 Verarbeitungsschritt-12 xcorr Ersatzpfad: L/R lag=%d → correction=%+d samples",
                        _lag_samples,
                        _lag_samples,
                    )
            except Exception as _xc_exc:
                logger.debug("§2.51 Verarbeitungsschritt-12 xcorr Ersatzpfad nicht blockierend: %s", _xc_exc)

        _max_rms_drop_db = {  # type: ignore[call-overload]
            MaterialType.SHELLAC: 1.2,
            MaterialType.WAX_CYLINDER: 1.2,
            MaterialType.WIRE_RECORDING: 1.3,
            MaterialType.VINYL: 1.5,
            MaterialType.TAPE: 1.7,
            MaterialType.REEL_TAPE: 1.6,
            MaterialType.CD_DIGITAL: 2.0,
            MaterialType.DAT: 2.0,
            MaterialType.MP3_LOW: 2.2,
            MaterialType.MP3_HIGH: 2.0,
            MaterialType.AAC: 2.0,
            MaterialType.STREAMING: 2.0,
        }.get(material if isinstance(material, MaterialType) else material, 1.8)
        _max_rms_lift_db = 1.0

        # §2.45a-I: Gated RMS — only frames > -50 dBFS (kein Stille-inflationierter RMS)
        # §V04-EXEMPT: compute_gated_rms_linear() RMS measurement,
        # NOT apply_musical_gain_envelope() — no reference_for_gate needed
        from backend.core.audio_utils import compute_gated_rms_linear as _grl_p12  # pylint: disable=import-outside-toplevel # noqa: I001

        _orig_rms = float(_grl_p12(orig, gate_dbfs=-50.0))
        _proc_rms = float(_grl_p12(proc, gate_dbfs=-50.0))
        if _orig_rms < 1e-10 or _proc_rms < 1e-12:
            return np.clip(proc.astype(np.float32), -1.0, 1.0), 0.0, 0.0

        _delta_db = float(20.0 * np.log10(max(_proc_rms / _orig_rms, 1e-30)))
        _gain_db = 0.0

        if _delta_db < -_max_rms_drop_db:
            _gain_db = (-_max_rms_drop_db) - _delta_db
        elif _delta_db > _max_rms_lift_db:
            _gain_db = _max_rms_lift_db - _delta_db

        if abs(_gain_db) > 0.01:
            _gain = float(10.0 ** (_gain_db / 20.0))
            _peak_p999 = float(np.percentile(np.abs(proc), 99.9) + 1e-12)
            if _gain > 1.0:
                _gain = min(_gain, float(0.985 / _peak_p999))
            # §2.45a-I: Einfache Multiplikation — kein Frame-Gate (hier globale Lautheitskorrektur,
            # nicht UV3-Mid-Pipeline-Drift-Guard; peak_p999 schützt gegen Clipping)
            proc = np.clip(proc * _gain, -1.0, 1.0)

        # Final preventive cap: no phase output above percentile ceiling.
        _peak_p999_out = float(np.percentile(np.abs(proc), 99.9) + 1e-12)
        if _peak_p999_out > self._MAX_PERCENTILE_PEAK:
            proc = np.clip(proc * (self._MAX_PERCENTILE_PEAK / _peak_p999_out), -1.0, 1.0)

        _out_rms = float(_grl_p12(proc, gate_dbfs=-50.0))
        _out_delta_db = float(20.0 * np.log10(max(_out_rms / _orig_rms, 1e-30)))
        _applied_makeup_db = float(20.0 * np.log10(max(_out_rms / _proc_rms, 1e-30)))

        return np.clip(proc.astype(np.float32), -1.0, 1.0), _out_delta_db, _applied_makeup_db

    def _estimate_pitch_yin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Rückwärts-kompatibles Alias auf pYIN (Mauch & Dixon 2014).

        Aufruf-Schnittstelle identisch mit YIN, aber probabilistische
        Multilevel-Threshold-Auswertung für höhere Robustheit.
        """
        return self._estimate_pitch_pyin(audio, sample_rate)

    def _estimate_pitch_pyin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Probabilistic YIN (pYIN) nach Mauch & Dixon (2014).

        Primary path: librosa.pyin (vectorised C++ backend, ~100× faster than
        the pure-Python triple-loop fallback). Falls back to the Python
        implementation capped at 30 s centre if librosa is unavailable.

        Args:
            audio: Mono float32 [-1,1]
            sample_rate: Sample rate (expected: 48000 Hz)

        Returns:
            (pitch_trajectory, confidence): Pitch Hz and confidence [0,1] per frame
        """
        # Kurzsignal-Guard: pYIN ist fuer sehr kleine Fenster numerisch instabil und
        # erzeugt sonst irrefuehrende n_fft-Warnungen aus librosa.
        _n_samples = int(len(audio))
        if _n_samples < 256:
            _hop = max(1, int(self.PITCH_WINDOW_MS * sample_rate / 1000) // self.PITCH_HOP_FACTOR)
            _frames = max(1, int(np.ceil(_n_samples / max(1, _hop))))
            return np.zeros(_frames, dtype=np.float64), np.zeros(_frames, dtype=np.float64)

        # -----------------------------------------------------------------
        # High-quality path: librosa.pyin (C-accelerated)
        # Env override semantics:
        #   AURIK_ENABLE_LIBROSA_PYIN=1  -> force enable
        #   AURIK_ENABLE_LIBROSA_PYIN=0  -> force disable
        #   unset/auto                   -> enable in quality/maximum
        # -----------------------------------------------------------------
        _qm_hint = str(getattr(self, "_quality_mode_hint", "quality")).strip().lower()
        _quality_first_unleashed = bool(getattr(self, "_quality_first_unleashed", _qm_hint in {"quality", "maximum"}))
        _pyin_env = os.environ.get("AURIK_ENABLE_LIBROSA_PYIN", "auto").strip().lower()
        if _pyin_env in {"1", "true", "yes", "on"}:
            _enable_librosa_pyin = True
        elif _pyin_env in {"0", "false", "no", "off"}:
            _enable_librosa_pyin = False
        else:
            _enable_librosa_pyin = _quality_first_unleashed
        if _enable_librosa_pyin:
            try:
                import librosa  # always available in .venv_aurik

                hop_samples = max(1, int(self.PITCH_WINDOW_MS * sample_rate / 1000) // self.PITCH_HOP_FACTOR)
                _safe_frame_length = 1 << int(np.floor(np.log2(min(2048, _n_samples))))
                _safe_frame_length = max(256, _safe_frame_length)
                hop_samples = min(hop_samples, max(1, _safe_frame_length // 4))
                f0, voiced_flag, voiced_prob = librosa.pyin(
                    audio.astype(np.float32),
                    fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
                    fmax=float(librosa.note_to_hz("C7")),  # ~2093 Hz
                    sr=sample_rate,
                    hop_length=hop_samples,
                    frame_length=_safe_frame_length,
                    fill_na=0.0,
                )
                # voiced_prob gives per-frame confidence; unvoiced → 0
                f0 = np.nan_to_num(f0, nan=0.0)
                confidence = np.where(voiced_flag, voiced_prob, 0.0).astype(np.float64)
                pitch_trajectory = f0.astype(np.float64)

                logger.debug(
                    "pYIN (librosa): %d frames, μ_pitch=%.1f Hz, μ_conf=%.3f",
                    len(pitch_trajectory),
                    float(np.mean(pitch_trajectory[pitch_trajectory > 0]) or 0),
                    float(np.mean(confidence)),
                )
                return pitch_trajectory, confidence

            except Exception as _lib_exc:
                logger.debug("librosa.pyin nicht verfuegbar (%s) — falling back to Python pYIN (30 s cap)", _lib_exc)
        else:
            logger.debug(
                "librosa.pyin deaktiviert by policy/env (quality_Betriebsart=%s, env=%s) — using Python pYIN Ersatzpfad",
                _qm_hint,
                _pyin_env,
            )

        # -----------------------------------------------------------------
        # Fallback: pure-Python pYIN.
        # In quality/maximum, do not cap to center window; in balanced/fast keep 30 s cap.
        # -----------------------------------------------------------------
        _PYIN_CAP_S = 0 if _quality_first_unleashed else 30
        _cap_samples = int(_PYIN_CAP_S * sample_rate) if _PYIN_CAP_S > 0 else 0
        _analysis_offset_samples = 0  # sample index where the analysed window starts in full audio
        if _PYIN_CAP_S > 0 and len(audio) > _cap_samples:
            _mid = len(audio) // 2
            _half = _cap_samples // 2
            audio_pyin = audio[_mid - _half : _mid + _half]
            _analysis_offset_samples = _mid - _half
            logger.debug(
                "pYIN Python Ersatzpfad: %.0f s audio capped to %d s centre", len(audio) / sample_rate, _PYIN_CAP_S
            )
        else:
            audio_pyin = audio

        window_samples = int(self.PITCH_WINDOW_MS * sample_rate / 1000)
        hop_samples = window_samples // self.PITCH_HOP_FACTOR

        min_period = int(sample_rate / 1000)  # max 1000 Hz
        max_period = int(sample_rate / 50)  # min 50 Hz
        max_period = min(max_period, window_samples // 2)

        num_windows = max(1, (len(audio_pyin) - window_samples) // hop_samples + 1)
        pitch_trajectory = np.zeros(num_windows, dtype=np.float64)
        confidence = np.zeros(num_windows, dtype=np.float64)

        # pYIN: Multi-Threshold weights via Beta(2,18)-like distribution
        N_thr = 20
        thresholds = np.linspace(0.01, 0.30, N_thr)
        beta_weights = (1 - thresholds) ** 17 * thresholds
        beta_weights /= beta_weights.sum() + 1e-10

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            if end > len(audio_pyin):
                break

            window = audio_pyin[start:end] * np.hanning(window_samples)

            # CMND function (YIN) — FFT-based autocorrelation O(N log N)
            from backend.core.core_utils import fft_autocorr

            autocorr = fft_autocorr(window)
            diff = 2.0 * (autocorr[0] - autocorr[:max_period])
            cmnd = np.ones(max_period)
            cumsum = np.cumsum(diff[1:])
            tau_range = np.arange(1, max_period)
            cmnd[1:] = diff[1:] * tau_range / (cumsum + 1e-10)

            # Multi-threshold pYIN
            cand_pitches: list = []
            cand_weights: list = []

            for thr, w in zip(thresholds, beta_weights):
                tau_est = 0
                for tau in range(min_period, max_period):
                    if cmnd[tau] < thr and 0 < tau < max_period - 1:
                        if cmnd[tau] <= cmnd[tau - 1] and cmnd[tau] <= cmnd[tau + 1]:
                            tau_est = tau
                            break
                if tau_est == 0:
                    tau_est = min_period + int(np.argmin(cmnd[min_period:max_period]))

                # Parabolic interpolation
                if 0 < tau_est < max_period - 1:
                    s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
                    denom = s0 - 2 * s1 + s2
                    if abs(denom) > 1e-10:
                        delta = 0.5 * (s0 - s2) / denom
                        tau_est = tau_est + delta

                if tau_est > 0:
                    cand_pitches.append(float(sample_rate) / float(tau_est))
                    cand_weights.append(w)

            if cand_pitches:
                cand_arr = np.array(cand_pitches)
                wgt_arr = np.array(cand_weights)
                wgt_arr /= wgt_arr.sum() + 1e-10
                mu = float(np.dot(cand_arr, wgt_arr))
                mask = np.abs(cand_arr - mu) < 0.10 * mu
                if mask.any():
                    pitch_trajectory[i] = float(np.dot(cand_arr[mask], wgt_arr[mask]) / (wgt_arr[mask].sum() + 1e-10))
                    confidence[i] = float(wgt_arr[mask].sum())
                else:
                    pitch_trajectory[i] = mu
                    confidence[i] = 0.3
            else:
                pitch_trajectory[i] = 0.0
                confidence[i] = 0.0

        # Temporal smoothing (simplified HMM tracking via exp smoothing)
        alpha_smooth = 0.7
        for i in range(1, num_windows):
            if pitch_trajectory[i] > 0 and pitch_trajectory[i - 1] > 0:
                pitch_trajectory[i] = alpha_smooth * pitch_trajectory[i - 1] + (1 - alpha_smooth) * pitch_trajectory[i]

        # §2.54 Temporal alignment — when pYIN analysed only a center window of a longer
        # audio, expand the trajectory to the full audio length.
        # Without this, N frames from a 30 s window get linspace-interpolated across
        # the full audio in _phase_vocoder_timestretch, applying the center-window
        # pitch-variation pattern to completely different temporal positions (intro/outro).
        # Frames outside the analysed window receive: pitch = center median (neutral),
        # confidence = 0.05 → below the >0.3 / >0.5 thresholds → stretch_factor = 1.0.
        if _analysis_offset_samples > 0 and len(audio) > len(audio_pyin):
            _full_windows = max(1, (len(audio) - window_samples) // hop_samples + 1)
            _center_first = _analysis_offset_samples // hop_samples
            _center_last = _center_first + num_windows
            _neutral_hz = (
                float(np.median(pitch_trajectory[pitch_trajectory > 0])) if np.any(pitch_trajectory > 0) else 0.0
            )
            _full_pitch = np.full(_full_windows, _neutral_hz, dtype=np.float64)
            _full_conf = np.full(_full_windows, 0.05, dtype=np.float64)
            _s = min(_center_first, _full_windows)
            _e = min(_center_last, _full_windows)
            _n = min(_e - _s, len(pitch_trajectory))
            _full_pitch[_s : _s + _n] = pitch_trajectory[:_n]
            _full_conf[_s : _s + _n] = confidence[:_n]
            pitch_trajectory = _full_pitch
            confidence = _full_conf
            num_windows = _full_windows
            logger.debug(
                "pYIN Python Ersatzpfad: trajectory expanded to %d frames (full audio),"
                " center window at frames %d..%d, neutral pitch=%.1f Hz outside window",
                _full_windows,
                _s,
                _s + _n,
                _neutral_hz,
            )

        logger.debug(
            "pYIN (Python): %d frames, μ_pitch=%.1f Hz, μ_conf=%.3f",
            num_windows,
            float(np.mean(pitch_trajectory[pitch_trajectory > 0]) or 0),
            float(np.mean(confidence)),
        )
        return pitch_trajectory, confidence

    def _yin_algorithm(
        self, window: np.ndarray, sample_rate: int, min_period: int, max_period: int
    ) -> tuple[float, float]:
        """Legacy YIN — nur noch als Fallback, primär wird pYIN verwendet.

        De Cheveigné & Kawahara (2002): \"YIN, a fundamental frequency
        estimator for speech and music\".
        ACHTUNG: Nicht als primärer Algorithmus verwenden \u2014 pYIN bevorzugen.
        """
        from backend.core.core_utils import fft_autocorr

        autocorr = fft_autocorr(window)
        diff = 2.0 * (autocorr[0] - autocorr[:max_period])
        cmnd = np.ones(max_period)
        cumsum = np.cumsum(diff[1:])
        tau_range = np.arange(1, max_period)
        cmnd[1:] = diff[1:] * tau_range / (cumsum + 1e-10)

        tau_estimate = 0
        min_cmnd = 1.0
        for tau in range(min_period, max_period):
            if cmnd[tau] < self.YIN_THRESHOLD and 0 < tau < max_period - 1:
                if cmnd[tau] < cmnd[tau - 1] and cmnd[tau] < cmnd[tau + 1]:
                    tau_estimate = tau
                    min_cmnd = cmnd[tau]
                    break
        if tau_estimate == 0:
            tau_estimate = min_period + int(np.argmin(cmnd[min_period:max_period]))
            min_cmnd = cmnd[tau_estimate]

        if 0 < tau_estimate < max_period - 1:
            s0, s1, s2 = cmnd[tau_estimate - 1], cmnd[tau_estimate], cmnd[tau_estimate + 1]
            denom = s0 - 2 * s1 + s2
            if abs(denom) > 1e-10:
                delta = 0.5 * (s0 - s2) / denom
                tau_estimate = tau_estimate + delta

        pitch_hz = float(sample_rate) / float(tau_estimate) if tau_estimate > 0 else 0.0
        conf = max(0.0, 1.0 - min_cmnd)
        return pitch_hz, conf

    @staticmethod
    def _compute_bump_local_strength(
        mono_audio: np.ndarray,
        bump_start: int,
        bump_end: int,
        sample_rate: int,
        base_strength: float,
        protected_zones: list,
    ) -> float:
        """Berechnet die individuelle Korrekturstärke für einen einzelnen Transport-Bump.

        Je größer die lokale Energieanomalie, desto stärker die Korrektur.
        In Vibrato- (§0p), Frisson-, Flüster- und Passaggio-Schutzzonen wird
        die Stärke auf ein Zone-spezifisches Cap begrenzt.

        Args:
            mono_audio:      Mono-Signal für Kontext-RMS-Messung
            bump_start:      Start-Sample des Bumps
            bump_end:        End-Sample des Bumps
            sample_rate:     Sample-Rate in Hz
            base_strength:   Basis-Stärke (Material/Confidence-adaptiv)
            protected_zones: [(start_s, end_s, max_strength), ...] — Schutzzonen

        Returns:
            Individuelle Korrekturstärke ∈ [0.10, 1.0]
        """
        ctx_pad = int(0.250 * max(sample_rate, 1))
        n = len(mono_audio)
        bump_region = mono_audio[bump_start:bump_end]
        if len(bump_region) < 4:
            return float(np.clip(base_strength, 0.10, 1.0))
        ctx_pre = mono_audio[max(0, bump_start - ctx_pad) : bump_start]
        ctx_post = mono_audio[bump_end : min(n, bump_end + ctx_pad)]
        ctx_parts = [a for a in (ctx_pre, ctx_post) if len(a) >= 4]
        if not ctx_parts:
            return float(np.clip(base_strength, 0.10, 1.0))
        ctx_audio = np.concatenate(ctx_parts)
        bump_rms = float(np.sqrt(np.mean(bump_region**2) + 1e-12))
        ctx_rms = float(np.sqrt(np.mean(ctx_audio**2) + 1e-12))
        energy_ratio = bump_rms / (ctx_rms + 1e-12)
        if energy_ratio < 1.0:
            # Energie-Einbruch: typischer Kassetten-/Bandhopper (tape-head-lift)
            local_severity = float(np.clip(1.0 - energy_ratio, 0.0, 1.0))
        else:
            # Energie-Spike: mechanischer Schlag (selten)
            local_severity = float(np.clip((energy_ratio - 1.0) / 1.5, 0.0, 1.0))
        # Magnitude-Faktor: Mindest 0.40 — auch leichte Bumps werden sicher repariert
        mag_factor = float(np.clip(0.40 + 0.60 * local_severity, 0.40, 1.0))
        local_strength = base_strength * mag_factor
        # Schutzzone: Stärke auf Zone-spezifisches Cap begrenzen
        if protected_zones:
            bump_center_s = float(bump_start + bump_end) * 0.5 / float(max(sample_rate, 1))
            for _zone in protected_zones:
                try:
                    pz_start, pz_end, pz_cap = float(_zone[0]), float(_zone[1]), float(_zone[2])
                    if pz_start <= bump_center_s <= pz_end:
                        local_strength = min(local_strength, pz_cap)
                        break
                except Exception:
                    logger.debug("_berechnen_bump_local_strength: silent except suppressed", exc_info=True)
                    continue
        if base_strength < 1e-6:
            return 0.0
        return float(np.clip(local_strength, 0.10, 1.0))

    @staticmethod
    def _collect_vfa_protected_zones(phase_kwargs: dict[str, Any]) -> list[tuple[float, float, float]]:
        """Sammelt Vokal-/Performance-Schutzzonen fuer per-Event-Staerkecaps."""
        protected_zones: list[tuple[float, float, float]] = []
        for zone in phase_kwargs.get("vibrato_zones") or []:
            try:
                protected_zones.append((float(zone[0]), float(zone[1]), 0.20))
            except Exception:
                logger.debug("_collect_vfa_protected_zones: silent except suppressed", exc_info=True)
                continue
        for zone in phase_kwargs.get("frisson_zones") or []:
            try:
                start_s = float(getattr(zone, "start_s", None) or zone[0])
                end_s = float(getattr(zone, "end_s", None) or zone[1])
                protected_zones.append((start_s, end_s, 0.30))
            except Exception:
                logger.debug("_collect_vfa_protected_zones: silent except suppressed", exc_info=True)
                continue
        for zone in phase_kwargs.get("whisper_zones") or []:
            try:
                protected_zones.append((float(zone[0]), float(zone[1]), 0.25))
            except Exception:
                logger.debug("_collect_vfa_protected_zones: silent except suppressed", exc_info=True)
                continue
        for zone in phase_kwargs.get("passaggio_zones") or []:
            try:
                protected_zones.append((float(zone[0]), float(zone[1]), 0.35))
            except Exception:
                logger.debug("_collect_vfa_protected_zones: silent except suppressed", exc_info=True)
                continue
        return protected_zones

    @staticmethod
    def _compute_tape_level_local_strength(
        mono_audio: np.ndarray,
        event_start: int,
        event_end: int,
        sample_rate: int,
        base_strength: float,
        max_deficit_db: float,
        max_gain_db: float,
        *,
        is_primary_tape: bool,
        confirmed_tape_dip: bool,
        protected_zones: list[tuple[float, float, float]],
    ) -> float:
        """Berechnet per Tape-Level-Dip die kleinste plausible, aber wirksame Korrekturstaerke."""
        if base_strength < 1e-6:
            return 0.0
        n = len(mono_audio)
        event_start = int(np.clip(event_start, 0, max(0, n - 1)))
        event_end = int(np.clip(event_end, event_start + 1, n))
        event_len = max(1, event_end - event_start)

        ctx_pad = int(0.250 * max(sample_rate, 1))
        ctx_pre = mono_audio[max(0, event_start - ctx_pad) : event_start]
        ctx_post = mono_audio[event_end : min(n, event_end + ctx_pad)]
        ctx_parts = [part for part in (ctx_pre, ctx_post) if len(part) >= 8]
        context_loss_db = 0.0
        if ctx_parts:
            ctx_audio = np.concatenate(ctx_parts)
            event_audio = mono_audio[event_start:event_end]
            event_rms = float(np.sqrt(np.mean(event_audio**2) + 1e-12))
            ctx_rms = float(np.sqrt(np.mean(ctx_audio**2) + 1e-12))
            context_loss_db = float(np.clip(20.0 * np.log10((ctx_rms + 1e-12) / (event_rms + 1e-12)), 0.0, max_gain_db))

        depth_ref_db = max(1.0, max_gain_db - 3.0)
        depth_factor = float(np.clip(max_deficit_db / depth_ref_db, 0.0, 1.0))
        context_factor = float(np.clip(context_loss_db / max(1.0, max_gain_db), 0.0, 1.0))
        duration_s = event_len / float(max(sample_rate, 1))
        duration_factor = float(np.clip((duration_s - 0.030) / 0.270, 0.0, 1.0))

        if is_primary_tape:
            floor = 0.58
            cap = 0.94
        elif confirmed_tape_dip:
            floor = 0.45
            cap = 0.88
        else:
            floor = 0.10
            cap = 0.45

        local_strength = max(
            float(base_strength),
            floor + 0.26 * depth_factor + 0.08 * context_factor + 0.08 * duration_factor,
        )
        local_strength = float(np.clip(local_strength, 0.0, cap))

        if protected_zones:
            center_s = float(event_start + event_end) * 0.5 / float(max(sample_rate, 1))
            for zone_start, zone_end, zone_cap in protected_zones:
                if float(zone_start) <= center_s <= float(zone_end):
                    local_strength = min(local_strength, float(zone_cap))
                    break

        return float(np.clip(local_strength, 0.0, 1.0))

    def _repair_transport_bumps(
        self,
        audio: np.ndarray,
        sample_rate: int,
        bump_locations: list[tuple[float, float]],
        strength: float = 0.85,
        protected_zones: list | None = None,
    ) -> tuple[np.ndarray, int]:
        """Repariert impulsive transport bumps at known locations.

        Multi-stage repair strategy (v2):
            1. Amplitude envelope smoothing toward context RMS level
            2. Local pitch correction via context-guided resampling
            3. Spectral context interpolation: blend bump spectrum toward
               weighted average of pre/post context spectra (removes LF thump
               and spectral centroid disruption that characterize transport bumps)
            4. Hanning-crossfade at margins for seamless integration

        Args:
            audio:          Audio signal (mono [N] or stereo [N, 2])
            sample_rate:    Sample rate in Hz
            bump_locations: List of (start_s, end_s) time pairs from DefectScanner
            strength:       Correction strength 0.0–1.0 (material-adaptive)

        Returns:
            (repaired_audio, n_bumps_repaired)
        """
        result, _was_channels_first = to_channels_last(audio.copy())
        is_stereo = audio.ndim == 2
        n_samples = audio_sample_count(result)
        margin_s = 0.030  # 30 ms crossfade margin
        margin_samples = int(margin_s * sample_rate)
        n_repaired = 0

        _mono_ref = safe_to_mono(result) if is_stereo else result
        for bump_start_s, bump_end_s in bump_locations:
            bump_start = int(bump_start_s * sample_rate)
            bump_end = int(bump_end_s * sample_rate)

            if bump_start < 0 or bump_end > n_samples or bump_end <= bump_start:
                continue
            # Per-Bump individuelle Stärke: lokale Energieanomalie + Schutzzonen-Check (§0l/§0p)
            _per_bump_s = self._compute_bump_local_strength(
                _mono_ref, bump_start, bump_end, sample_rate, strength, protected_zones or []
            )

            region_start = max(0, bump_start - margin_samples)
            region_end = min(n_samples, bump_end + margin_samples)
            region_len = region_end - region_start

            if region_len < sample_rate // 100:  # minimum 10 ms
                continue

            ctx_before_start = max(0, region_start - sample_rate // 4)
            ctx_after_end = min(n_samples, region_end + sample_rate // 4)

            if is_stereo:
                mono_ctx_before = np.mean(result[ctx_before_start:region_start], axis=1)
                mono_ctx_after = np.mean(result[region_end:ctx_after_end], axis=1)
            else:
                mono_ctx_before = result[ctx_before_start:region_start]
                mono_ctx_after = result[region_end:ctx_after_end]

            # 1. Reference RMS from surrounding context
            ctx_audio = np.concatenate([mono_ctx_before, mono_ctx_after])
            ref_rms = float(np.sqrt(np.mean(ctx_audio**2) + 1e-12)) if len(ctx_audio) > 0 else 1.0

            # 2. Amplitude envelope smoothing
            if is_stereo:
                for ch in range(result.shape[1]):
                    result[region_start:region_end, ch] = self._smooth_bump_envelope(
                        result[region_start:region_end, ch],
                        ref_rms,
                        margin_samples,
                        _per_bump_s,
                    )
            else:
                result[region_start:region_end] = self._smooth_bump_envelope(
                    result[region_start:region_end],
                    ref_rms,
                    margin_samples,
                    _per_bump_s,
                )

            # 3. Local pitch correction
            bump_len = bump_end - bump_start
            if bump_len >= sample_rate // 50:
                if is_stereo:
                    for ch in range(result.shape[1]):
                        result[bump_start:bump_end, ch] = self._local_pitch_flatten(
                            result[bump_start:bump_end, ch],
                            mono_ctx_before,
                            mono_ctx_after,
                            sample_rate,
                            _per_bump_s,
                        )
                else:
                    result[bump_start:bump_end] = self._local_pitch_flatten(
                        result[bump_start:bump_end],
                        mono_ctx_before,
                        mono_ctx_after,
                        sample_rate,
                        _per_bump_s,
                    )

            # 4. Spectral context interpolation — remove LF thump and timbral disruption
            #    by blending the bump's magnitude spectrum toward the surrounding context
            if bump_len >= 256:
                if is_stereo:
                    for ch in range(result.shape[1]):
                        result[bump_start:bump_end, ch] = self._spectral_context_blend(
                            result[bump_start:bump_end, ch],
                            (
                                result[ctx_before_start:region_start, ch]
                                if ctx_before_start < region_start
                                else np.zeros(1)
                            ),
                            result[region_end:ctx_after_end, ch] if region_end < ctx_after_end else np.zeros(1),
                            _per_bump_s,
                        )
                else:
                    result[bump_start:bump_end] = self._spectral_context_blend(
                        result[bump_start:bump_end],
                        result[ctx_before_start:region_start] if ctx_before_start < region_start else np.zeros(1),
                        result[region_end:ctx_after_end] if region_end < ctx_after_end else np.zeros(1),
                        _per_bump_s,
                    )

            n_repaired += 1

        # Safety: NaN/Inf guard + clip
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)
        return restore_layout(result, _was_channels_first), n_repaired

    @staticmethod
    def _smooth_bump_envelope(
        segment: np.ndarray,
        ref_rms: float,
        margin_samples: int,
        strength: float,
    ) -> np.ndarray:
        """Glättet the amplitude envelope of a bump region toward the reference RMS level.

        Uses a Hanning-weighted crossfade at margins and local RMS gain correction
        in the bump interior to eliminate sudden amplitude jumps.
        """
        result = segment.copy()
        n = len(result)
        if n < 4 or ref_rms < 1e-10:
            return result

        # Compute local RMS in short windows (5 ms)
        win = max(1, min(n, 240))  # ~5 ms at 48 kHz
        local_rms = np.array(
            [float(np.sqrt(np.mean(result[i : i + win] ** 2) + 1e-12)) for i in range(0, n - win + 1, max(1, win // 2))]
        )

        if len(local_rms) == 0:
            return result

        # Gain correction: target is ref_rms, blend with strength
        for i, lr in enumerate(local_rms):
            if lr > 1e-10:
                gain = 1.0 + strength * (ref_rms / lr - 1.0)
                gain = np.clip(gain, 0.5, 2.0)  # Safety clamp
                start_idx = i * max(1, win // 2)
                end_idx = min(n, start_idx + win)
                result[start_idx:end_idx] *= gain

        # Apply Hanning crossfade at margins
        fade_len = min(margin_samples, n // 2)
        if fade_len > 1:
            fade_in = np.hanning(fade_len * 2)[:fade_len]
            fade_out = np.hanning(fade_len * 2)[fade_len:]
            # Blend: corrected × fade + original × (1 - fade)
            result[:fade_len] = segment[:fade_len] * (1.0 - fade_in) + result[:fade_len] * fade_in
            result[-fade_len:] = segment[-fade_len:] * (1.0 - fade_out) + result[-fade_len:] * fade_out

        return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

    @staticmethod
    def _spectral_context_blend(
        bump_audio: np.ndarray,
        ctx_before: np.ndarray,
        ctx_after: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Mischt the magnitude spectrum of a bump region toward surrounding context.

        Transport bumps inject spurious low-frequency energy (mechanical thump) and
        cause abrupt spectral centroid shifts. This method:
          1. Computes average magnitude spectrum of pre/post context
          2. Computes magnitude spectrum of the bump region
          3. Blends bump magnitudes toward context magnitudes (strength-weighted)
          4. Preserves original phase (no phase distortion)
          5. RMS-normalizes output to prevent amplitude drift

        Uses raw FFT (no windowing) to avoid edge-amplification artifacts.
        """
        n = len(bump_audio)
        if n < 128 or strength < 0.01:
            return bump_audio

        fft_size = 1
        while fft_size < n:
            fft_size *= 2

        # Compute reference spectrum from context (weighted average)
        ref_mag = None
        n_ref = 0
        for ctx in (ctx_before, ctx_after):
            if len(ctx) < 64:
                continue
            chunk = ctx[-n:] if ctx is ctx_before else ctx[:n]
            if len(chunk) < 64:
                continue
            padded = np.zeros(fft_size, dtype=np.float64)
            padded[: len(chunk)] = chunk.astype(np.float64)
            mag = np.abs(np.fft.rfft(padded))
            if ref_mag is None:
                ref_mag = mag
            else:
                ref_mag = ref_mag + mag
            n_ref += 1

        if ref_mag is None or n_ref == 0:
            return bump_audio

        ref_mag /= n_ref

        # Compute bump spectrum (no windowing — avoids edge amplification)
        bump_padded = np.zeros(fft_size, dtype=np.float64)
        bump_f64 = bump_audio.astype(np.float64)
        bump_padded[:n] = bump_f64
        bump_fft = np.fft.rfft(bump_padded)
        bump_mag = np.abs(bump_fft)
        bump_phase = np.angle(bump_fft)

        # Only suppress frequencies where bump exceeds context (remove thump),
        # don't boost missing frequencies (that would add artifacts)
        blended_mag = bump_mag.copy()
        excess = bump_mag > ref_mag * 1.2
        blended_mag[excess] = bump_mag[excess] * (1.0 - strength * 0.7) + ref_mag[excess] * strength * 0.7

        # Reconstruct with original phase
        blended_fft = blended_mag * np.exp(1j * bump_phase)
        result_full = np.fft.irfft(blended_fft, n=fft_size)
        result = result_full[:n].astype(np.float32)

        # RMS-normalize to match original bump level (prevent amplitude drift)
        orig_rms = float(np.sqrt(np.mean(bump_f64**2) + 1e-12))
        result_rms = float(np.sqrt(np.mean(result.astype(np.float64) ** 2) + 1e-12))
        if result_rms > 1e-10:
            result *= np.float32(orig_rms / result_rms)

        # Blend with original using strength
        result = bump_audio * (1.0 - strength * 0.5) + result * (strength * 0.5)

        return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Step 6c: Tape Head Contact Level Stabilizer
    # ------------------------------------------------------------------
    # Repairs gradual envelope dips caused by tape-head pressure
    # variation or capstan irregularity in cassette/reel recordings.
    #
    # Defect morphology (from real-world cassette analysis):
    #   - Gradual fade-down: 60-100 ms onset
    #   - Minimum depth: 10-25 dB below local context level
    #   - Sharp recovery: < 25 ms back to normal level
    #   - Distributed across entire song, not just intro
    #   - Average rate: ~0.5-1.0 per second
    #
    # These dips are NOT caught by Phase 24 dropout repair:
    #   - Phase 24 requires > 75 % energy drop (these are 10-20 dB = 50-90 %)
    #   - Phase 24 max_dropout_ms = 200 ms (many dips last 150-300 ms)
    #   - Phase 24 detects sudden drops, not gradual fades
    #
    # Algorithm (Tape Head Contact Level Stabilizer v1):
    #   1. Compute RMS envelope in 20 ms windows, 10 ms hop
    #   2. Compute slow-moving reference via percentile filter (500 ms)
    #   3. Detect dips: envelope < reference - dip_threshold_db
    #   4. For each dip region: compute compensating gain, smooth edges
    #   5. Limit max gain to avoid noise amplification
    #   6. Apply gain with smooth interpolation
    #
    # Scientific basis:
    #   - Camras (1988): Magnetic Recording Handbook - head contact mechanics
    #   - Fastl & Zwicker (2007): equal-loudness perception of level dips
    # ------------------------------------------------------------------

    def _estimate_speed_curve_from_instantaneous_frequency(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Geschwindigkeitskurve via Hilbert Instantan-Frequenz (IEC 60386, SNR-robust).

        Sicherheitsdesign (IEC 60386 §5.2):
          - Dieselbe Methode wie DefectScanner._detect_wow() — funktioniert bei
            SNR = -1 dB (bestätigt: DefectScanner erkennt wow=1.00 für Elke Best)
          - Keine Voiced-Detection nötig — arbeitet auf beliebigem Audiosignal
          - Bandpass 80–4000 Hz isoliert stimmhafte/Instruments-Frequenzen
          - Hilbert-Transform → Instantan-Phase → Instantan-Frequenz pro Sample
          - Normierung auf Median-IF → relative Geschwindigkeitsabweichung
          - Wow-Isolierung: Bandpassfilter 0.2–4 Hz auf IF-Zeitreihe
          - Plausibilitätsprüfung: Amplitude 3–80 Cents + Dominanz ≥ 0.10

        Rückgabe kompatibel mit _fit_sinusoidal_wow_curve / _calculate_stretch_factors:
          virtual_pitch[t] = 440 * relative_IF_deviation[t]
        """
        from scipy.signal import butter, sosfilt, sosfiltfilt
        from scipy.signal import hilbert as _hilbert

        # --- Konstanten ---
        BP_LOW = 80.0  # Hz — Untergrenze Bandpass (Stimminhalt / Instrumente)
        BP_HIGH = 4000.0  # Hz — Obergrenze
        IF_WIN_MS = 50.0  # ms — Glättungsfenster für Instantan-Frequenz
        WOW_LOW_HZ = 0.2  # Hz — Untergrenze Wow-Band (IEC 60386 Slow-Wow)
        WOW_HIGH_HZ = 4.0  # Hz — Obergrenze Wow-Band
        AMP_MIN_CENTS = 3.0
        AMP_MAX_CENTS = 80.0
        VALID_FRAC_MIN = 0.05  # mind. 5 % valide IF-Frames

        mono = (safe_to_mono(audio) if audio.ndim == 2 else audio).astype(np.float32)

        # Kurzsignal-Guard: mindestens 4 s für Sub-Hz Auflösung
        if len(mono) < sample_rate * 4:
            logger.debug("Verarbeitungsschritt 12 IF-Ersatzpfad: Audio zu kurz (%d Samples < 4s)", len(mono))
            return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

        try:
            # Segment-Limit (120 s) — Hilbert auf dem vollständigen Segment zu teuer
            _max_samples = min(len(mono), sample_rate * 120)
            seg = mono[:_max_samples].astype(np.float64)

            # Bandpass 80–4000 Hz
            _nyq = sample_rate / 2.0
            bp_sos = butter(
                3,
                [max(BP_LOW / _nyq, 0.001), min(BP_HIGH / _nyq, 0.999)],
                btype="band",
                output="sos",
            )
            bp_audio = sosfilt(bp_sos, seg)

            # Hilbert-Transform → Instantan-Phase → Instantan-Frequenz (Hz)
            analytic = np.asarray(_hilbert(bp_audio), dtype=np.complex128)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) * float(sample_rate) / (2.0 * np.pi)  # [N-1]

            # Valide IF-Frames: nur pitched-Frequenzen
            valid_mask = (inst_freq > 50.0) & (inst_freq < 5000.0)
            valid_fraction = float(np.sum(valid_mask)) / float(max(1, len(valid_mask)))
            if valid_fraction < VALID_FRAC_MIN:
                logger.debug(
                    "Verarbeitungsschritt 12 IF-Ersatzpfad: zu wenig valide IF-Frames (%.1f %% < %.0f %%)",
                    valid_fraction * 100,
                    VALID_FRAC_MIN * 100,
                )
                return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

            # Normalisierung: IF / Median-IF → relative Abweichung (dimensionslos)
            median_if = float(np.median(inst_freq[valid_mask]))
            if median_if < 50.0:
                return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

            if_norm = inst_freq / (median_if + 1e-6)
            if_norm = np.nan_to_num(if_norm, nan=1.0)
            if_norm = np.clip(if_norm, 0.80, 1.20)  # max ±20 % (physikalisch plausibel)

            # Glätten: 50-ms-Fester (entspricht Capstan-Meter-Bandbreite)
            if_win = max(1, int(IF_WIN_MS * sample_rate / 1000))
            if_smooth = np.convolve(if_norm, np.ones(if_win) / if_win, mode="valid")
            if len(if_smooth) < 8:
                return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

            # Wow-Band isolieren: Bandpass 0.2–4 Hz auf der IF-Zeitreihe
            if_frame_rate = float(sample_rate) / float(if_win)
            _if_nyq = if_frame_rate / 2.0
            _wow_lo = max(WOW_LOW_HZ / _if_nyq, 0.001)
            _wow_hi = min(WOW_HIGH_HZ / _if_nyq, 0.999)
            if _wow_lo >= _wow_hi:
                return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

            sos_wow = butter(4, [_wow_lo, _wow_hi], btype="band", output="sos")
            wow_component = sosfiltfilt(sos_wow, if_smooth - 1.0)
            wow_component = np.nan_to_num(wow_component, nan=0.0)

            # Amplitude prüfen (RMS → Peak-Schätzung via sqrt(2))
            wow_rms = float(np.sqrt(np.mean(wow_component**2)))
            amplitude_cents = float(abs(1200.0 * np.log2(1.0 + wow_rms * np.sqrt(2.0) + 1e-10)))
            if amplitude_cents < AMP_MIN_CENTS or amplitude_cents > AMP_MAX_CENTS:
                logger.debug(
                    "Verarbeitungsschritt 12 IF-Ersatzpfad: Amplitude außerhalb Bereich (%.2f Cents)", amplitude_cents
                )
                return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

            # Dominanz: Wow-Peak vs. Breitband-IF-Energie (0.2–4 Hz)
            _freqs_fft = np.fft.rfftfreq(len(wow_component), d=1.0 / if_frame_rate)
            _spectrum = np.abs(np.fft.rfft(wow_component)) ** 2
            _peak_band = (_freqs_fft >= 0.5) & (_freqs_fft <= 3.0)
            _total_wow = float(np.sum(_spectrum[(_freqs_fft >= WOW_LOW_HZ) & (_freqs_fft <= WOW_HIGH_HZ)]) + 1e-12)
            if _peak_band.any():
                _bp_power = _spectrum[_peak_band]
                _peak_local = int(np.argmax(_bp_power))
                _peak_freq = float(_freqs_fft[_peak_band][_peak_local])
                _dominance = float(_bp_power[_peak_local]) / _total_wow
            else:
                _peak_freq, _dominance = 0.0, 0.0

            if _dominance < 0.10:
                logger.debug(
                    "Verarbeitungsschritt 12 IF-Ersatzpfad: keine dominante Wow-Komponente (peak=%.2f Hz, dominance=%.4f < 0.10)",
                    _peak_freq,
                    _dominance,
                )
                return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

            # IF (Wow-Anteil) auf Phase-12-Frame-Rate resamplen
            _hop_samples = int(self.PITCH_WINDOW_MS * sample_rate / 1000) // self.PITCH_HOP_FACTOR  # typisch 1200
            _target_len = max(4, len(mono) // _hop_samples)
            if_resampled = np.interp(
                np.linspace(0, len(wow_component) - 1, _target_len),
                np.arange(len(wow_component)),
                wow_component,
            )

            # Virtueller Pitch: 440 * (1 + wow_component)
            virtual_pitch = (440.0 * (1.0 + if_resampled)).astype(np.float64)
            virtual_pitch = np.clip(virtual_pitch, 20.0, 4000.0)

            # Konfidenz: abhängig von IF-Valid-Fraction und Wow-Dominanz
            _conf_level = float(np.clip(valid_fraction * _dominance * 8.0, 0.25, 0.75))
            confidence = np.full(len(virtual_pitch), _conf_level, dtype=np.float64)

            logger.info(
                "Verarbeitungsschritt 12 IF-Ersatzpfad (IEC 60386): T=%d Frames, "
                "Wow-Peak=%.2f Hz, Amplitude=%.1f Cents, Dominanz=%.3f, valid_frac=%.2f",
                len(virtual_pitch),
                _peak_freq,
                amplitude_cents,
                _dominance,
                valid_fraction,
            )
            return virtual_pitch, confidence

        except Exception as _exc:
            logger.warning("Verarbeitungsschritt 12 IF-Ersatzpfad fehlgeschlagen: %s", _exc)
            return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    def _stabilize_tape_level(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float,
        *,
        is_primary_tape: bool = True,
        confirmed_tape_dip: bool = False,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Erkennt and repair tape head contact level dips — STFT-domain SOTA v2.

        Physics basis (Wallace 1951):
            When head-tape contact pressure varies, effective head-tape spacing
            Δd increases → spacing loss L(f) = e^(-2π·Δd·f/v).
            HF loss is exponentially greater than LF loss: a 1 µm spacing
            increase at cassette speed (v=47.6 mm/s) causes ~11 dB loss at
            10 kHz but only ~1 dB at 1 kHz.  Wideband-only gain therefore
            restores loudness but leaves a residual HF spectral tilt.

        v2 SOTA upgrades (2026-04-09):
          1. STFT-domain frequency-dependent gain (HF boosted proportional to
             estimated spacing loss derived from per-dip context comparison)
          2. Per-dip HF tilt estimated adaptively from spectral centroid shift
             in context vs. dip frames — no fixed tape-speed assumption needed
          3. SNR guard per frequency bin: bins at noise floor are not boosted
             (prevents amplification of residual tape hiss during deep dips)
          4. Asymmetric gain envelope: slow onset (~30 % of dip duration) /
             fast recovery (~10 %) matches the physical capstan-irregularity
             dip morphology
          5. §2.51 linked stereo: spectral gain derived from mono, applied
             identically to L and R channels

        Args:
            audio:       Audio signal (mono [N] or stereo [N, 2] — float32/64)
            sample_rate: Sample rate in Hz (expected: 48000)
            strength:    Correction strength [0.0, 1.0]

        Returns:
            (stabilized_audio, n_dips_repaired)

        References:
            Wallace, R.L. (1951) The Reproduction of Magnetically Recorded
            Signals. Bell System Technical Journal 30(4):1145–1173.
            Camras, M. (1988) Magnetic Recording Handbook. §8.3.
            McKnight, J.G. (1969) Tape Reproducer Response Measurements.
        """
        if strength < 0.01:
            return audio, 0

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        is_stereo = audio.ndim == 2
        n_samples = audio_sample_count(audio)

        if is_stereo:
            mono = safe_to_mono(audio).astype(np.float64)
        else:
            mono = audio.astype(np.float64)

        # ── Step 1: RMS envelope (20 ms / 10 ms) + percentile-75 reference ──
        env_win_s = 0.020
        env_hop_s = 0.010
        ref_win_s = 0.500
        # Material-adaptive thresholds:
        # Primary tape (reel_tape/cassette): standard tape-head contact physics → tight params.
        # Chain material (vinyl→tape→mp3 etc.): stabilizer runs because tape is in chain but the
        # primary material is NOT tape → use conservative params to avoid false positives on
        # natural musical dynamics (reverb tails, note decays, breath pauses = 30–80 ms).
        if is_primary_tape:
            dip_thresh_db = 3.0
            min_dip_frames = 3  # 30 ms — tight for cassette capstan bumps
            max_gain_db = 15.0
        elif confirmed_tape_dip:
            dip_thresh_db = 4.5
            min_dip_frames = 6  # 60 ms — detected tape-stage dips in transferred material
            max_gain_db = 13.0
        else:
            dip_thresh_db = 6.0  # only repair severe dips (> 6 dB below rolling p75)
            min_dip_frames = 20  # 200 ms minimum — real tape contact dips last 200–500 ms
            max_gain_db = 6.0  # max 6 dB boost for chain material (§0 Primum non nocere)
        # Intro/outro protection: ignore only true edge zones. Primary tape/cassette can contain
        # audible head-contact dips immediately after start; blanket 5-s protection masks exactly
        # the defect class phase_12 is supposed to repair. Chain-material stays conservative.
        _protect_s = 0.5 if is_primary_tape else 5.0
        _protect_frames = int(_protect_s / env_hop_s)

        env_win = max(1, int(env_win_s * sample_rate))
        env_hop = max(1, int(env_hop_s * sample_rate))
        n_frames = max(0, (n_samples - env_win) // env_hop)

        if n_frames < 10:
            return audio, 0

        # rms_env — vectorised via stride_tricks (replaces Python list comprehension)
        _n_needed = (n_frames - 1) * env_hop + env_win
        if _n_needed <= len(mono):
            _rms_frames = np.lib.stride_tricks.sliding_window_view(mono[:_n_needed], env_win)[::env_hop][:n_frames]
            rms_env = np.sqrt(np.mean(_rms_frames**2, axis=1) + 1e-15)
        else:
            rms_env = np.array(
                [np.sqrt(np.mean(mono[i * env_hop : i * env_hop + env_win] ** 2) + 1e-15) for i in range(n_frames)],
                dtype=np.float64,
            )
        rms_db = 20.0 * np.log10(rms_env + 1e-15)

        from scipy.ndimage import percentile_filter

        ref_n = max(3, int(ref_win_s / env_hop_s))
        ref_n = ref_n + (1 - ref_n % 2)  # ensure odd
        # mode='nearest' repeats the edge value at signal boundaries.
        # mode='reflect' (former setting) mirrors the signal: for a quiet intro the reflected
        # window contained the louder song body → ref_db[0] = loud → intro detected as dip
        # → up-to-15 dB boost at beginning/end of song (root cause of level surge artifact).
        ref_db = percentile_filter(rms_db, percentile=75, size=ref_n, mode="nearest")

        dip_mask_rms = rms_db < (ref_db - dip_thresh_db)
        # Intro/outro protection: force dip_mask=False in first/last _protect_frames
        if _protect_frames > 0 and n_frames > 2 * _protect_frames:
            dip_mask_rms[:_protect_frames] = False
            dip_mask_rms[-_protect_frames:] = False

        from scipy.ndimage import label as nd_label

        labeled, n_dips_raw = nd_label(dip_mask_rms)  # type: ignore[misc]  # type: ignore[misc]

        if n_dips_raw == 0:
            return audio, 0

        # ── Step 2: Collect valid dip events ─────────────────────────────
        rms_centres = np.arange(n_frames) * env_hop + env_win // 2  # sample pos

        dip_events: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(1, n_dips_raw + 1):
            frames = np.where(labeled == i)[0]
            if len(frames) < min_dip_frames:
                continue
            deficit = ref_db[frames] - rms_db[frames]
            if np.max(deficit) > max_gain_db + 5.0:
                continue  # likely genuine silence gap
            if np.mean(rms_db[frames]) < -55.0:
                continue  # noise floor — nothing to restore
            dip_events.append((frames, deficit))

        if not dip_events:
            return audio, 0

        # ── Step 3: STFT setup ────────────────────────────────────────────
        # fft_size = 2048 → 42.7 ms @ 48 kHz; hop = 512 → 10.7 ms (75 % OV)
        fft_size = 2048
        hop_stft = fft_size // 4
        n_freqs = fft_size // 2 + 1
        freqs_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

        _, _, X_mono = signal.stft(
            mono,
            fs=sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size - hop_stft,
            boundary="even",
            padded=True,
        )
        n_stft_frames = X_mono.shape[1]
        stft_centres = np.arange(n_stft_frames) * hop_stft + fft_size // 2  # samples

        X_mag_db = 20.0 * np.log10(np.abs(X_mono) + 1e-15)  # [n_freqs, n_stft_frames]

        # Spectral gain mask: identity (1.0) everywhere — modified per dip below
        spectral_gain = np.ones((n_freqs, n_stft_frames), dtype=np.float64)

        # ── Step 4: Build per-dip spectral gain (broadband + HF tilt) ────
        n_repaired = 0
        ctx_n = 64  # context window ≈ 680 ms before dip onset

        for rms_frames, deficit in dip_events:
            _event_max_deficit_db = float(np.max(deficit))
            _event_start = int(rms_frames[0] * env_hop)
            _event_end = int(min(n_samples, (rms_frames[-1] + 1) * env_hop + env_win))
            event_strength = self._compute_tape_level_local_strength(
                mono,
                _event_start,
                _event_end,
                sample_rate,
                strength,
                _event_max_deficit_db,
                max_gain_db,
                is_primary_tape=is_primary_tape,
                confirmed_tape_dip=confirmed_tape_dip,
                protected_zones=protected_zones or [],
            )

            # Map RMS dip frames → nearest STFT frame indices
            rms_ctrs = rms_centres[rms_frames]
            stft_idx = np.searchsorted(stft_centres, rms_ctrs)
            stft_idx = np.unique(np.clip(stft_idx, 0, n_stft_frames - 1))
            if len(stft_idx) == 0:
                continue

            # Per-STFT-frame broadband gain: interpolate from per-RMS-frame deficit
            bb_gain_db_stft = np.interp(
                stft_centres[stft_idx].astype(float),
                rms_centres[rms_frames].astype(float),
                np.clip(deficit * event_strength, 0.0, max_gain_db),
            )  # dB, shape [n_stft_dip]

            # ── HF spectral-tilt correction (Wallace spacing-loss inversion) ──
            first_stft = int(stft_idx[0])
            ctx_start = max(0, first_stft - ctx_n)
            ctx_end = max(0, first_stft - 2)
            hf_tilt_db = np.zeros(n_freqs, dtype=np.float64)  # per-bin

            if ctx_end - ctx_start >= 4:
                ctx_mag = X_mag_db[:, ctx_start:ctx_end]  # [n_freqs, ctx]
                dip_mag = X_mag_db[:, stft_idx]  # [n_freqs, n_dip_stft]

                # Reference: p75 of context per bin (robust to note-onset transients)
                ref_spec = np.percentile(ctx_mag, 75, axis=1)  # [n_freqs]
                dip_spec = np.mean(dip_mag, axis=1)  # [n_freqs]

                spectral_loss = ref_spec - dip_spec  # positive = dropped in dip

                # Remove broadband component (median of LF bins < 4 kHz)
                lf_mask = freqs_hz < 4000.0
                if lf_mask.any():
                    broadband_loss = float(np.median(spectral_loss[lf_mask]))
                else:
                    broadband_loss = float(np.median(spectral_loss))

                tilt_raw = spectral_loss - broadband_loss  # frequency-dep. residual

                # SNR guard: bins where signal ≤ noise floor + 6 dB → no HF boost
                noise_floor = np.percentile(ctx_mag, 10, axis=1)  # p10 ≈ noise floor
                snr_in_dip = dip_spec - noise_floor
                tilt_raw = np.where(snr_in_dip > 6.0, tilt_raw, 0.0)

                # Cap HF tilt at 10 dB; only positive values (loss → boost)
                hf_tilt_db = np.clip(tilt_raw, 0.0, 10.0) * event_strength

            # ── Asymmetric gain envelope (slow onset / fast recovery) ────
            n_sf = len(stft_idx)
            onset_n = max(1, int(n_sf * 0.30))  # ~30 % slow fade-in
            recovery_n = max(1, int(n_sf * 0.10))  # ~10 % fast fade-out
            fade_env = np.ones(n_sf, dtype=np.float64)
            fade_env[:onset_n] = np.linspace(0.0, 1.0, onset_n)
            if n_sf > onset_n + recovery_n:
                fade_env[-recovery_n:] = np.linspace(1.0, 0.0, recovery_n)

            # ── Combine broadband + HF-tilt into spectral_gain mask — vectorised ──
            tilt_lin = 10.0 ** (hf_tilt_db / 20.0)  # [n_freqs] — per-bin extra boost
            max_lin = 10.0 ** (max_gain_db / 20.0)

            _active_mask = bb_gain_db_stft >= 0.3
            if np.any(_active_mask):
                _active_idx = stft_idx[_active_mask]
                _bb_lin = 10.0 ** (bb_gain_db_stft[_active_mask] / 20.0)  # (m,)
                _fe = fade_env[_active_mask]  # (m,)
                # combined[frame, freq] = 1 + (bb_lin[frame] * tilt_lin[freq] - 1) * fade_env[frame]
                _comb = _bb_lin[:, None] * tilt_lin[None, :]  # (m, n_freqs)
                _comb = 1.0 + (_comb - 1.0) * _fe[:, None]  # (m, n_freqs)
                _comb = np.clip(_comb, 1.0, max_lin)  # (m, n_freqs)
                spectral_gain[:, _active_idx] = _comb.T  # (n_freqs, m)

            n_repaired += 1

        if n_repaired == 0:
            return audio, 0

        # ── Step 5: Apply spectral gain to each channel (§2.51 linked) ──
        def _apply_gain_to_channel(sig_ch: np.ndarray) -> np.ndarray:
            """Wendet an: the spectral_gain mask to one audio channel via STFT."""
            _, _, X_ch = signal.stft(
                sig_ch.astype(np.float64),
                fs=sample_rate,
                window="hann",
                nperseg=fft_size,
                noverlap=fft_size - hop_stft,
                boundary="even",
                padded=True,
            )
            n_apply = min(X_ch.shape[1], spectral_gain.shape[1])
            X_ch[:, :n_apply] *= spectral_gain[:, :n_apply]
            _, y = signal.istft(
                X_ch,
                fs=sample_rate,
                window="hann",
                nperseg=fft_size,
                noverlap=fft_size - hop_stft,
                boundary="even",
            )
            # Trim / zero-pad to original length
            y_out = np.zeros(n_samples, dtype=np.float64)
            n_trim = min(len(y), n_samples)
            y_out[:n_trim] = y[:n_trim]
            return y_out  # type: ignore[no-any-return]

        if is_stereo:
            left, right = stereo_channel_view(audio)
            L_out = _apply_gain_to_channel(left)
            R_out = _apply_gain_to_channel(right)
            result = stereo_like(L_out, R_out, audio)
        else:
            result = _apply_gain_to_channel(mono)

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0).astype(np.float32)

        logger.debug(
            "Tape level stabilizer v2 (STFT): %d dips repaired, HF-tilt correction active, strength=%.2f",
            n_repaired,
            strength,
        )

        return result, n_repaired

    def _local_pitch_flatten(
        self,
        bump_audio: np.ndarray,
        ctx_before: np.ndarray,
        ctx_after: np.ndarray,
        sample_rate: int,
        strength: float,
    ) -> np.ndarray:
        """Flatten pitch excursion in a bump region using context-guided resampling.

        Algorithm:
            1. Estimate reference pitch from context (before + after bump)
            2. Estimate pitch within the bump
            3. Compute per-sample resampling ratio to flatten toward reference
            4. Apply via linear interpolation (fast, no phase vocoder overhead)
        """
        n = len(bump_audio)
        if n < 64:
            return bump_audio

        # Estimate reference pitch from context
        ref_pitch_before = self._quick_pitch_estimate(ctx_before, sample_rate) if len(ctx_before) > 256 else 0.0
        ref_pitch_after = self._quick_pitch_estimate(ctx_after, sample_rate) if len(ctx_after) > 256 else 0.0

        if ref_pitch_before > 0 and ref_pitch_after > 0:
            ref_pitch = (ref_pitch_before + ref_pitch_after) / 2.0
        elif ref_pitch_before > 0:
            ref_pitch = ref_pitch_before
        elif ref_pitch_after > 0:
            ref_pitch = ref_pitch_after
        else:
            return bump_audio  # Cannot determine reference → skip

        # Estimate pitch in bump
        bump_pitch = self._quick_pitch_estimate(bump_audio, sample_rate)
        if bump_pitch <= 0 or ref_pitch <= 0:
            return bump_audio

        # Compute pitch ratio
        ratio = ref_pitch / bump_pitch
        # Only correct if there's a meaningful deviation (> 0.5 %)
        if abs(ratio - 1.0) < 0.005:
            return bump_audio

        # Clamp correction to avoid extreme warping
        correction = 1.0 + strength * (ratio - 1.0)
        correction = np.clip(correction, 0.9, 1.1)  # max ±10 % correction

        # Resample via linear interpolation
        new_len = int(n * correction)
        if new_len < 4 or new_len > n * 3:
            return bump_audio

        indices = np.linspace(0, n - 1, new_len)
        resampled = np.interp(indices, np.arange(n), bump_audio).astype(np.float32)

        # Fit back to original length via truncation or zero-pad + crossfade
        if len(resampled) >= n:
            result = resampled[:n]
        else:
            result = np.zeros(n, dtype=np.float32)
            result[: len(resampled)] = resampled
            # Fade out tail
            fade = max(1, n - len(resampled))
            result[len(resampled) :] = bump_audio[len(resampled) :] * np.linspace(1.0, 0.0, fade)[: n - len(resampled)]

        return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

    def _quick_pitch_estimate(self, audio: np.ndarray, sample_rate: int) -> float:
        """Fast autocorrelation-based pitch estimate for short segments.

        Returns estimated pitch in Hz, or 0.0 if unvoiced/unreliable.
        """
        n = len(audio)
        if n < 128:
            return 0.0

        # Use center portion
        center = audio[n // 4 : 3 * n // 4]
        center = center - np.mean(center)
        if np.max(np.abs(center)) < 1e-6:
            return 0.0

        # Autocorrelation via FFT
        n_fft = 1
        while n_fft < len(center) * 2:
            n_fft *= 2
        fft_x = np.fft.rfft(center, n=n_fft)
        acf = np.fft.irfft(fft_x * np.conj(fft_x))[: len(center)]

        if acf[0] < 1e-10:
            return 0.0
        acf = acf / acf[0]

        # Find first peak after initial decline
        min_lag = max(2, int(sample_rate / 1000.0))  # 1000 Hz max
        max_lag = min(len(acf) - 1, int(sample_rate / 50.0))  # 50 Hz min

        if max_lag <= min_lag:
            return 0.0

        search = acf[min_lag : max_lag + 1]
        if len(search) < 3:
            return 0.0

        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]

        if peak_val < 0.3:  # low confidence
            return 0.0

        lag = min_lag + peak_idx
        return float(sample_rate) / float(lag) if lag > 0 else 0.0

    def _separate_wow_flutter(self, pitch_trajectory: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate wow (<4 Hz) and flutter (4-100 Hz) components.

        Wow: Slow speed variations (<4 Hz) - from motor/belt issues
        Flutter: Fast speed variations (4-100 Hz) - from mechanical vibrations

        Args:
            pitch_trajectory: Pitch estimates (Hz) over time
            sample_rate: Original audio sample rate (used to determine analysis frame rate)

        Returns:
            (wow_component, flutter_component): Separated pitch deviation components
        """
        # Remove zero values (unvoiced frames)
        valid_mask = pitch_trajectory > 0
        valid_pitches = pitch_trajectory[valid_mask]

        if len(valid_pitches) < 10:
            # Too few valid estimates
            return np.zeros_like(pitch_trajectory), np.zeros_like(pitch_trajectory)

        # Calculate median pitch (stable reference)
        median_pitch = np.median(valid_pitches)

        # Pitch deviation from median (as percentage)
        deviation = np.zeros_like(pitch_trajectory)
        deviation[valid_mask] = (valid_pitches - median_pitch) / median_pitch * 100

        # Calculate frame rate of pitch trajectory
        window_samples = int(self.PITCH_WINDOW_MS * sample_rate / 1000)
        hop_samples = window_samples // self.PITCH_HOP_FACTOR
        frame_rate = sample_rate / hop_samples

        # Low-pass filter for wow (<4 Hz)
        wow_cutoff = 4.0  # Hz
        nyquist = frame_rate / 2
        if wow_cutoff < nyquist:
            sos_wow = signal.butter(4, wow_cutoff / nyquist, btype="low", output="sos")
            wow_component = signal.sosfiltfilt(sos_wow, deviation)
        else:
            wow_component = deviation  # Frame rate too low, treat all as wow

        # Band-pass filter for flutter (4-100 Hz)
        flutter_low = 4.0  # Hz
        flutter_high = 100.0  # Hz
        if flutter_high < nyquist:
            sos_flutter = signal.butter(4, [flutter_low / nyquist, flutter_high / nyquist], btype="band", output="sos")
            flutter_component = signal.sosfiltfilt(sos_flutter, deviation)
        else:
            flutter_component = deviation - wow_component

        return wow_component, flutter_component

    def _detect_wow_flutter(
        self, pitch_trajectory: np.ndarray, confidence: np.ndarray, threshold: float
    ) -> tuple[bool, float]:
        """
        Erkennt wow & flutter by analyzing pitch deviations with confidence weighting.

        Args:
            pitch_trajectory: Pitch estimates (Hz)
            confidence: Confidence scores (0.0-1.0)
            threshold: Detection threshold (percent deviation)

        Returns:
            (detected, max_deviation_percent)
        """
        max_deviation = self._calculate_max_deviation(pitch_trajectory, confidence)

        # Detect if max deviation exceeds threshold
        detected = max_deviation > threshold

        return detected, max_deviation

    def _calculate_max_deviation(self, pitch_trajectory: np.ndarray, confidence: np.ndarray) -> float:
        """Calculate max pitch deviation weighted by confidence."""
        # Only use confident pitch estimates (confidence > 0.5)
        confident_mask = (pitch_trajectory > 0) & (confidence > 0.5)

        if np.sum(confident_mask) < 10:
            return 0.0

        confident_pitches = pitch_trajectory[confident_mask]
        median_pitch = np.median(confident_pitches)

        # Deviation from median (percentage)
        deviations = np.abs((confident_pitches - median_pitch) / median_pitch) * 100

        # Max deviation
        max_dev = np.max(deviations) if len(deviations) > 0 else 0.0

        return max_dev

    def _calculate_stretch_factors(
        self,
        pitch_trajectory: np.ndarray,
        confidence: np.ndarray,
        strength: float,
        *,
        max_stretch_delta: float = 0.05,
    ) -> np.ndarray:
        """
        Calculate time-stretching factors to correct pitch deviations.

        Stretch factor > 1.0: slow down (pitch was too high)
        Stretch factor < 1.0: speed up (pitch was too low)

        Args:
            pitch_trajectory: Pitch estimates (Hz)
            confidence: Confidence scores (0.0-1.0)
            strength: Correction strength (0.0-1.0)

        Returns:
            Array of stretch factors (one per pitch estimate)
        """
        # Use confident estimates to determine target pitch
        confident_mask = (pitch_trajectory > 0) & (confidence > 0.5)

        if np.sum(confident_mask) < 10:
            # Not enough confident estimates, no correction
            return np.ones_like(pitch_trajectory)  # type: ignore[no-any-return]

        confident_pitches = pitch_trajectory[confident_mask]

        # §0 Primum non nocere — Melody-Detection Guard:
        # pYIN measures ALL pitch variation, including melodic note changes.
        # A span > 100 cents (1 semitone P5..P95) cannot be wow/flutter:
        # real wow/flutter is < 50 cents; melody spans multiple semitones.
        # Applying melody-derived stretch_factors destroys the recording.
        if len(confident_pitches) >= 10:
            _cp_p5 = np.percentile(confident_pitches, 5)
            _cp_p95 = np.percentile(confident_pitches, 95)
            if _cp_p5 > 0.0 and _cp_p95 > _cp_p5:
                _span_cents = 1200.0 * np.log2(_cp_p95 / _cp_p5)
                _mat_key = str(getattr(self, "_mat_type_for_stretch", "") or "").lower()
                _wow_sev = float(getattr(self, "_wow_sev_for_stretch", 0.0) or 0.0)
                # Smooth: limit = 100 + 300 × severity, clamped to [100, 400]
                _adaptive_limit = 100.0 + 300.0 * min(_wow_sev, 1.0)
                _adaptive_limit = max(100.0, min(_adaptive_limit, 400.0))
                if _span_cents > _adaptive_limit:
                    logger.info(
                        "Verarbeitungsschritt 12 pitch span %.0f cents > %.0f (mat=%s wow=%.2f)",
                        _span_cents,
                        _adaptive_limit,
                        _mat_key,
                        _wow_sev,
                    )
                    return np.ones_like(pitch_trajectory)
                else:
                    logger.info(
                        "Verarbeitungsschritt 12 pitch span %.0f cents within limit %.0f (mat=%s wow=%.2f)",
                        _span_cents,
                        _adaptive_limit,
                        _mat_key,
                        _wow_sev,
                    )

        target_pitch = np.median(confident_pitches)

        # Calculate stretch factors for each frame — vectorised (replaces Python for-loop)
        _valid_mask = (pitch_trajectory > 0) & (confidence > 0.3)
        _raw_stretch = np.where(_valid_mask, pitch_trajectory / max(float(target_pitch), 1e-8), 1.0)
        _unclamped = 1.0 + strength * (_raw_stretch - 1.0)
        stretch_factors = np.where(
            _valid_mask,
            np.clip(_unclamped, 1.0 - max_stretch_delta, 1.0 + max_stretch_delta),
            1.0,
        ).astype(pitch_trajectory.dtype)

        # Stretch-Faktoren glätten mit Savitzky-Golay (polynomialer Least-Squares-Smoother)
        # Ersetzt signal.medfilt: Savitzky-Golay erhält Peaks und liefert glatteren Verlauf
        try:
            from scipy.signal import savgol_filter

            stretch_factors = savgol_filter(stretch_factors, window_length=5, polyorder=2)
        except Exception:
            # Notfall-Fallback: Uniform-Smoothing
            from scipy.ndimage import uniform_filter1d

            stretch_factors = uniform_filter1d(stretch_factors, size=5, mode="nearest")
        stretch_factors = np.clip(
            stretch_factors,
            1.0 - max_stretch_delta,
            1.0 + max_stretch_delta,
        )

        return stretch_factors  # type: ignore[no-any-return]

    @staticmethod
    def _apply_defect_locality_to_stretch_factors(
        stretch_factors: np.ndarray,
        *,
        audio_length_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
    ) -> tuple[np.ndarray, float]:
        """Begrenzt Stretch-Faktoren auf erkannte Wow/Flutter-nahe Defektsegmente."""
        sf = np.asarray(stretch_factors, dtype=np.float32)
        if sf.size == 0 or audio_length_samples <= 0 or sample_rate <= 0:
            return sf, 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return sf, 0.0

        location_keys = (
            "wow",
            "flutter",
            "wow_flutter",
            "multiband_wow_flutter",
            "scrape_flutter",
            "flutter_spectral_sidebands",
        )
        merged_locations: list[tuple[float, float]] = []
        for key in location_keys:
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start_s = max(0.0, float(loc[0]))
                    end_s = max(start_s, float(loc[1]))
                except Exception:
                    logger.debug(
                        "_anwenden_defect_locality_to_stretch_factors: silent except suppressed", exc_info=True
                    )
                    continue
                merged_locations.append((start_s, end_s))

        if not merged_locations:
            return sf, 0.0

        n_frames = int(sf.size)
        duration_s = float(audio_length_samples) / float(sample_rate)
        frame_centers_s = ((np.arange(n_frames, dtype=np.float32) + 0.5) / max(float(n_frames), 1.0)) * duration_s
        active_mask = np.zeros(n_frames, dtype=bool)
        pad_s = 0.05
        for start_s, end_s in merged_locations:
            active_mask |= (frame_centers_s >= max(0.0, start_s - pad_s)) & (frame_centers_s <= end_s + pad_s)

        if not np.any(active_mask):
            return np.ones_like(sf), 0.0

        masked = np.where(active_mask, sf, 1.0).astype(np.float32)
        return masked, float(np.mean(active_mask.astype(np.float32)))

    def _harmonic_isolated_timestretch(
        self, audio: np.ndarray, stretch_factors: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """DDSP-inspirierte Zeitstreckung mit Harmonic-Residual-Trennung (§DDSP-Harmonic).

        Trennt Signal in harmonische Komponente (Obertöne, Vokal) und Rausch-Residuum
        (Tape-Hiss, Vinyl-Crackle). Zeitstreckung nur auf harmonische Komponente anwenden —
        Rausch-Residuum bleibt zeitlich unverändert (physikalisch authentische Trägertextur).

        Algorithmus:
            1. HPSS (librosa.effects.hpss, margin=3.0): H=Harmonisch, R=Residuum
            2. Qualitäts-Gate: harmonic_ratio < 0.15 → Fallback Phase-Vocoder
            3. _phase_vocoder_timestretch NUR auf H → H_korrigiert
            4. Rekombination: H_korrigiert + R (Längenangleichung)

        Fallback auf standard Phase-Vocoder wenn:
            - librosa nicht verfügbar
            - HPSS-Trennqualität < threshold (harmonic_ratio < 0.15)
            - beliebige Exception (non-blocking §0c)

        Wissenschaftliche Grundlage: Engel et al. (2020) DDSP-Konzept der harmonischen
        Sinusoid-Synthese. Hier rein DSP (HPSS = Medianfilter-basiert), kein ML.
        """
        try:
            import librosa  # type: ignore[import-untyped]

            audio_f = np.asarray(audio, dtype=np.float32)
            _harmonic, _residual = librosa.effects.hpss(audio_f, margin=3.0)  # type: ignore[attr-defined]

            # Trennungsqualität: harmonische Energie-Ratio
            _total_rms = float(np.sqrt(np.mean(audio_f**2) + 1e-12))
            if _total_rms < 1e-9:
                return self._phase_vocoder_timestretch(audio, stretch_factors, sample_rate)
            _harmonic_ratio = float(np.sqrt(np.mean(_harmonic**2) + 1e-12)) / _total_rms
            if _harmonic_ratio < 0.15:
                # Kaum harmonischer Inhalt → HPSS macht keinen Sinn
                return self._phase_vocoder_timestretch(audio, stretch_factors, sample_rate)

            # Zeitstreckung NUR auf harmonische Komponente
            _h_corrected = self._phase_vocoder_timestretch(_harmonic, stretch_factors, sample_rate)

            # Rekombination (Längenangleichung)
            _n = min(len(_h_corrected), len(_residual), len(audio_f))
            _result = _h_corrected[:_n] + _residual[:_n]
            _result = np.nan_to_num(_result, nan=0.0, posinf=0.0, neginf=0.0)
            _result = np.clip(_result, -1.0, 1.0)

            logger.debug(
                "Verarbeitungsschritt_12: DDSP-Harmonic-Isolation: harmonic_Verhaeltnis=%.2f → Trägertextur erhalten",
                _harmonic_ratio,
            )
            _out: np.ndarray = np.asarray(_result, dtype=audio.dtype)
            return _out
        except Exception as _hpss_exc:
            logger.debug(
                "Verarbeitungsschritt_12: HPSS-Isolation Ersatzpfad auf Verarbeitungsschritt-Vocoder: %s", _hpss_exc
            )
            return self._phase_vocoder_timestretch(audio, stretch_factors, sample_rate)

    def _phase_vocoder_timestretch(
        self, audio: np.ndarray, stretch_factors: np.ndarray, _sample_rate: int
    ) -> np.ndarray:
        """Wendet an: time-varying phase-coherent time-stretching via STFT phase vocoder.

        §PV1 replaces the old WSOLA-like np.interp resample with a professional
        STFT-based phase vocoder (Laroche & Dolson 1999; Driedger & Müller 2016).
        Uses instantaneous frequency estimation, phase propagation across frames,
        and overlap-add reconstruction — the same class of algorithm as iZotope RX
        and Capstan.

        Falls back to WSOLA (np.interp) if the phase vocoder module is unavailable.

        Args:
            audio: Mono audio samples
            stretch_factors: Time-varying stretch factors (one per pitch window)
            _sample_rate: Sample rate

        Returns:
            Time-stretched audio, same length as input
        """
        if len(audio) < 8 or len(stretch_factors) == 0:
            return audio.copy()

        try:
            from backend.core.dsp.phase_vocoder import phase_vocoder_timestretch

            return phase_vocoder_timestretch(
                audio,
                stretch_factors,
                _sample_rate,
                n_fft=self.STFT_WINDOW_SIZE,
                hop_ratio=self.PITCH_HOP_FACTOR,
            )
        except Exception as _pv_exc:
            logger.debug("§PV1 Verarbeitungsschritt vocoder nicht verfuegbar (%s) — Ersatzpfad to WSOLA", _pv_exc)
            return self._phase_vocoder_wsola_fallback(audio, stretch_factors)

    def _phase_vocoder_wsola_fallback(self, audio: np.ndarray, stretch_factors: np.ndarray) -> np.ndarray:
        """WSOLA-style fallback via np.interp — used when STFT phase vocoder is unavailable."""
        audio_f = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n_samples = len(audio_f)

        sf = np.asarray(stretch_factors, dtype=np.float32)
        sf = np.clip(sf, 0.90, 1.10)
        if len(sf) == 1:
            sf_samples = np.full(n_samples, sf[0], dtype=np.float32)
        else:
            src_idx = np.linspace(0, n_samples - 1, len(sf), dtype=np.float32)
            dst_idx = np.arange(n_samples, dtype=np.float32)
            sf_samples = np.interp(dst_idx, src_idx, sf).astype(np.float32)

        try:
            from scipy.signal import savgol_filter

            win = max(5, (n_samples // 400) | 1)
            win = min(win, n_samples if n_samples % 2 == 1 else n_samples - 1)
            if win >= 5:
                sf_samples = savgol_filter(sf_samples, window_length=win, polyorder=2, mode="interp")
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        sf_samples = np.clip(sf_samples, 0.90, 1.10)
        if np.max(np.abs(sf_samples - 1.0)) < 0.002:
            return audio.copy()

        src_step = 1.0 / np.clip(sf_samples, 0.85, 1.15)
        src_pos = np.cumsum(src_step)
        src_pos -= src_pos[0]
        max_pos = float(src_pos[-1]) + 1e-12
        src_pos *= (n_samples - 1) / max_pos
        src_pos = np.clip(src_pos, 0.0, n_samples - 1)

        corrected = np.interp(src_pos, np.arange(n_samples, dtype=np.float32), audio_f)
        corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        return corrected.astype(audio.dtype, copy=False)  # type: ignore[no-any-return]

    def _apply_neural_phase_coherence(
        self, audio: np.ndarray, sample_rate: int, reference: np.ndarray | None = None
    ) -> np.ndarray:
        """§C3 Neural Phase Vocoder post-processing — PGHI-consistent phase coherence.

        After wow/flutter time-stretching via Phase Vocoder the STFT phases can
        become incoherent in voiced regions (harmonic misalignment, phasiness).
        This method applies spectral-coherence-guided phase propagation (Marafioti
        et al. 2019 PGHI-based approach) to restore natural phase relationships.

        Algorithm:
        1. Compute STFT of corrected signal and (optional) reference signal.
        2. Measure per-bin group-delay coherence: C_k = |H_corr·H_ref*|/(|H_corr||H_ref|)
           where H is the complex STFT coefficient (or autocorrelation proxy without ref).
        3. Build a frequency-smoothed coherence mask: bins with C ≥ 0.85 are well-aligned;
           bins with C < 0.50 are candidates for phase regularisation.
        4. Phase regularisation: for low-coherence bins propagate phase from the
           already-coherent neighbouring bin via instantaneous frequency estimation
           (≈PGHI derivative of log-magnitude spectrogram):
               φ_new[k, t] = φ[k, t-1] + 2π × f_instantaneous × hop/sr
        5. Reconstruct via iSTFT with regularised phases.
        6. Blend blend_weight of regularised output with original (conservative default 0.30).

        References:
            Marafioti et al. (2019) ICASSP — Phase Gradient Heap Integration (PGHI).
            Laroche & Dolson (1999) IEEE TSAP — instantaneous frequency estimation.

        Returns phase-coherence-improved audio (same shape as input).
        """
        if len(audio) < 1024:
            return audio.copy()

        try:
            assert sample_rate == 48000, "Phase 12 neural coherence: sr must be 48000"
            n_fft = 2048
            hop = n_fft // 4
            win = np.hanning(n_fft).astype(np.float32)
            audio_f = np.asarray(audio, dtype=np.float32)
            orig_len = len(audio_f)

            # Step 1: STFT of corrected audio — vectorised via stride_tricks + batched rfft
            n_frames = 1 + (orig_len - n_fft) // hop
            if n_frames < 2:
                return audio.copy()

            # Framing: zero-copy view, shape (n_frames, n_fft)
            _frames = np.lib.stride_tricks.sliding_window_view(audio_f, n_fft)[::hop][:n_frames]
            stft = np.fft.rfft(_frames * win, axis=1)  # (T, F) — 1 batched call

            mag = np.abs(stft)
            phase = np.angle(stft)

            # Step 2: Coherence proxy — if reference available, use it; else use autocorrelation
            if reference is not None and len(reference) == orig_len:
                ref_f = np.asarray(reference, dtype=np.float32)
                _frames_ref = np.lib.stride_tricks.sliding_window_view(ref_f, n_fft)[::hop][:n_frames]
                stft_ref = np.fft.rfft(_frames_ref * win, axis=1)
                # Per-bin Pearson-like correlation magnitude
                mag_ref = np.abs(stft_ref)
                coherence = np.abs(stft * np.conj(stft_ref)) / (mag * mag_ref + 1e-14)
            else:
                # Autocorrelation-based coherence: temporal consistency of phase increments
                phase_diff = np.diff(phase, axis=0)  # (T-1, F)
                # Coherence ≈ circular std of phase increments (low spread = high coherence)
                phase_std = np.std(phase_diff, axis=0)  # (F,)
                coherence = np.clip(1.0 - phase_std / np.pi, 0.0, 1.0)
                coherence = np.tile(coherence, (n_frames, 1))

            # Step 3: Identify incoherent bins (C < 0.5)
            incoherent_mask = coherence < 0.50
            n_incoherent = int(np.sum(incoherent_mask))
            if n_incoherent == 0 or n_incoherent > 0.80 * coherence.size:
                # All coherent (already OK) or all incoherent (PGHI propagation becomes unstable).
                # Use a conservative temporal STFT coherence smoothing fallback instead of passthrough.
                return self._phase_coherence_emergency_smoothing(audio, sample_rate)

            # Step 4: PGHI-inspired phase propagation — vectorised inner loop
            # Instantaneous frequency estimation via log-magnitude gradient
            log_mag = np.log(mag.astype(np.float64) + 1e-14)
            # Horizontal gradient (time direction) → IF estimate
            d_log_mag_dt = np.gradient(log_mag, axis=0)
            f_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
            omega_bins = 2.0 * np.pi * f_bins  # (F,)

            phase_reg = phase.astype(np.float64)
            _hop_sr = hop / sample_rate
            for t in range(1, n_frames):
                _mask_t = incoherent_mask[t]
                if np.any(_mask_t):
                    _if_est = omega_bins + d_log_mag_dt[t]  # (F,) — vectorised
                    _propagated = phase_reg[t - 1] + _if_est * _hop_sr  # (F,)
                    phase_reg[t] = np.where(_mask_t, _propagated, phase_reg[t])

            # Step 5: Reconstruct with regularised phases — vectorised irfft + OLA
            stft_reg = mag.astype(np.float64) * np.exp(1j * phase_reg)
            # Batch irfft: (T, n_fft) — much faster than individual calls
            frames_out = np.fft.irfft(stft_reg, axis=1)[:, :n_fft] * win.astype(np.float64)
            win_sq = (win**2).astype(np.float64)
            output = np.zeros(orig_len, dtype=np.float64)
            norm = np.zeros(orig_len, dtype=np.float64)
            for t in range(n_frames):
                s = t * hop
                output[s : s + n_fft] += frames_out[t]
                norm[s : s + n_fft] += win_sq
            norm = np.where(norm > 1e-10, norm, 1.0)
            output = output / norm

            output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

            # Step 6: Conservative blend (0.30 weight to avoid over-smoothing attack transients)
            blend = 0.30
            result = (1.0 - blend) * audio_f + blend * output
            result = np.clip(result, -1.0, 1.0)
            logger.debug(
                "§C3 Neural Verarbeitungsschritt Coherence: repaired %d incoherent STFT bins (blend=%.2f)",
                n_incoherent,
                blend,
            )
            return result.astype(audio.dtype)  # type: ignore[no-any-return]

        except Exception as _c3_exc:
            logger.debug("§C3 Neural Verarbeitungsschritt Coherence nicht blockierend: %s", _c3_exc)
            return self._phase_coherence_emergency_smoothing(audio, sample_rate)

    def _phase_coherence_emergency_smoothing(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Konservativer Notfallpfad fuer C3: leichte STFT-Temporalglättung statt Pass-Through."""
        try:
            if sample_rate <= 0:
                return audio.copy()

            if len(audio) < 1024:
                # Kurzsignal-Notfallpfad: leichte zeitliche Kohärenzglättung statt No-op.
                audio_f = np.asarray(audio, dtype=np.float64)
                if len(audio_f) < 3:
                    return np.nan_to_num(audio_f, nan=0.0, posinf=0.0, neginf=0.0).astype(audio.dtype, copy=False)  # type: ignore[no-any-return]
                kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
                smooth = np.convolve(audio_f, kernel, mode="same")
                blend = 0.12
                result = (1.0 - blend) * audio_f + blend * smooth
                result = np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                return result.astype(audio.dtype, copy=False)  # type: ignore[no-any-return]

            n_fft = 1024
            hop = n_fft // 4
            win = np.hanning(n_fft).astype(np.float64)
            audio_f = np.asarray(audio, dtype=np.float64)
            n = len(audio_f)

            n_frames = 1 + max(0, (n - n_fft) // hop)
            if n_frames < 3:
                kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
                smooth = np.convolve(audio_f, kernel, mode="same")
                blend = 0.12
                result = (1.0 - blend) * audio_f + blend * smooth
                result = np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                return result.astype(audio.dtype, copy=False)  # type: ignore[no-any-return]

            frames = np.lib.stride_tricks.sliding_window_view(audio_f, n_fft)[::hop][:n_frames]
            stft = np.fft.rfft(frames * win, axis=1)

            # Leichte zeitliche Kohärenzglättung im komplexen Spektrum.
            stft_smooth = stft.copy()
            stft_smooth[1:-1] = 0.70 * stft[1:-1] + 0.15 * stft[:-2] + 0.15 * stft[2:]

            frames_out = np.fft.irfft(stft_smooth, n=n_fft, axis=1) * win[np.newaxis, :]
            out = np.zeros(n, dtype=np.float64)
            norm = np.zeros(n, dtype=np.float64)
            win_sq = win**2
            for i in range(n_frames):
                s = i * hop
                out[s : s + n_fft] += frames_out[i]
                norm[s : s + n_fft] += win_sq

            norm = np.where(norm > 1e-10, norm, 1.0)
            out = out / norm
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

            # Sehr konservativer Blend, um Transienten zu schonen.
            blend = 0.18
            result = (1.0 - blend) * audio_f + blend * out
            result = np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            return result.astype(audio.dtype, copy=False)  # type: ignore[no-any-return]
        except Exception as _c3_fallback_exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            if "result" in locals():
                return result.astype(audio.dtype, copy=False)  # type: ignore[no-any-return]
            logger.debug("§C3 emergency smoothing Ersatzpfad fehlgeschlagen: %s", _c3_fallback_exc)
            return audio.copy()

    def _psola_timestretch(
        self,
        audio: np.ndarray,
        stretch_factors: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Pitch-Synchronous Overlap-Add (PSOLA) für Gesangsmaterial.

        Erhält Formanten bei Wow/Flutter-Korrektur (Moulines & Charpentier 1990;
        Macon & Clements 1997). Aktiviert wenn PANNs Vocals confidence >= 0.40.
        Fallback auf _phase_vocoder_timestretch() bei unbekanntem Grundton.

        Args:
            audio:           Mono-Audio [samples], beliebige float-Dtype.
            stretch_factors: Stretch-Faktoren pro pYIN-Frame, Ratios [~0.95–1.05].
            sample_rate:     Sample-Rate in Hz (assert == 48000).

        Returns:
            Zeitgedehntes Audio gleicher Länge wie Eingabe, NaN/Inf-frei, gleiche Dtype.
        """
        if len(audio) == 0:
            return audio.copy()

        dtype = audio.dtype
        audio_f = audio.astype(np.float32)

        # Grundfrequenz-Schätzung für Pitch-Marken (pYIN — Mauch & Dixon 2014)
        pitch_hz, confidence = self._estimate_pitch_yin(audio_f, sample_rate)
        n_frames = len(pitch_hz)
        if n_frames == 0:
            return self._phase_vocoder_timestretch(audio, stretch_factors, sample_rate)

        hop = self.STFT_HOP_SIZE

        # Pitch-Perioden in Samples (Fallback 440 Hz für nicht-stimmhafte Segmente)
        voiced = (pitch_hz > 50) & (confidence > 0.40)
        f0_safe = np.where(voiced & (pitch_hz > 0), pitch_hz, 440.0)
        period_samps = np.round(sample_rate / np.maximum(f0_safe, 1.0)).astype(int)
        period_samps = np.clip(period_samps, 20, sample_rate // 50)  # >= 50 Hz Untergrenze

        # Stretch-Faktoren auf n_frames interpolieren
        if len(stretch_factors) != n_frames:
            x_src = np.linspace(0, n_frames - 1, max(len(stretch_factors), 2))
            sf_per_frame = np.interp(np.arange(n_frames), x_src, stretch_factors)
        else:
            sf_per_frame = stretch_factors.astype(np.float32)
        sf_per_frame = np.clip(sf_per_frame, 0.9, 1.1)

        # OLA-Ausgangspuffer (großzügig dimensioniert, am Ende getrimmt)
        n_input = len(audio_f)
        max_period = int(np.max(period_samps))
        out_buf = np.zeros(n_input + max_period * 4, dtype=np.float32)
        weight_buf = np.zeros_like(out_buf)

        # §2.54 PSOLA-Safety: hop=512 with high f0 (>187 Hz) → grain size (2*period) < hop
        # → consecutive grains don't overlap → zero-weight gaps → silence artefacts.
        # Guard: fall back to Phase Vocoder when median period < hop/2 (f0 > 187 Hz).
        if int(np.median(period_samps)) < hop // 2:
            logger.debug(
                "Verarbeitungsschritt_12 PSOLA safety: median f0=%.0f Hz > %.0f Hz Schwelle → Verarbeitungsschritt-Vocoder Ersatzpfad",
                float(sample_rate) / max(float(np.median(period_samps)), 1.0),
                float(sample_rate) / max(hop // 2, 1),
            )
            return self._phase_vocoder_timestretch(audio, stretch_factors, sample_rate)

        out_write = 0
        for i in range(n_frames):
            in_center = i * hop
            if in_center >= n_input:
                break
            period = int(period_samps[i])
            sf = float(sf_per_frame[i])

            # Grain: symmetrisch ±1 Periode um in_center (Hanning-gewichtet)
            i_s = max(0, in_center - period)
            i_e = min(n_input, in_center + period)
            grain = audio_f[i_s:i_e].copy()
            if len(grain) == 0:
                out_write += round(hop * sf)
                continue

            win = np.hanning(len(grain))
            grain *= win

            # Ausgabe-Fensterposition (OLA)
            out_center = out_write
            o_s = max(0, out_center - period)
            o_e = min(len(out_buf), out_center + period)
            g_len = o_e - o_s
            if g_len <= 0:
                out_write += round(hop * sf)
                continue

            # Grain auf Fensterlänge anpassen (Trimm oder Zero-Pad)
            if g_len < len(grain):
                grain = grain[:g_len]
                win = win[:g_len]
            elif g_len > len(grain):
                pad = g_len - len(grain)
                grain = np.pad(grain, (0, pad))
                win = np.pad(win, (0, pad))

            out_buf[o_s:o_e] += grain
            weight_buf[o_s:o_e] += win
            out_write += round(hop * sf)

        # OLA-Normierung; Ausgabe auf Originallänge trimmen + NaN-Schutz
        safe_w = np.maximum(weight_buf[:n_input], 1e-8)
        result = out_buf[:n_input] / safe_w
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, -1.0, 1.0).astype(dtype)  # type: ignore[no-any-return]


# Standalone test
def _run_test() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Verarbeitungsschritt 12: Professional Wow & Flutter Correction v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio with synthetic wow/flutter
    duration = 5.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate 440 Hz tone with combined wow (1 Hz) + flutter (20 Hz)
    wow_freq = 1.0  # Hz (slow speed variation)
    wow_depth = 0.015  # 1.5% pitch variation

    flutter_freq = 20.0  # Hz (fast mechanical vibration)
    flutter_depth = 0.005  # 0.5% pitch variation

    # Combined instantaneous pitch variation
    pitch_variation = (
        1.0 + wow_depth * np.sin(2 * np.pi * wow_freq * t) + flutter_depth * np.sin(2 * np.pi * flutter_freq * t)
    )
    instantaneous_freq = 440 * pitch_variation

    # Generate audio with varying pitch
    phase = np.cumsum(2 * np.pi * instantaneous_freq / sample_rate)
    audio = 0.3 * np.sin(phase)

    # Add harmonics for more realistic tone
    audio += 0.15 * np.sin(2 * phase)  # 2nd harmonic
    audio += 0.10 * np.sin(3 * phase)  # 3rd harmonic

    logger.debug("erzeugt %ss test audio @ %s Hz", duration, sample_rate)
    logger.debug("Base frequency: 440 Hz with harmonics (2nd, 3rd)")
    logger.debug("Wow: %s Hz, Depth: %.2f%%", wow_freq, wow_depth * 100)
    logger.debug("Flutter: %s Hz, Depth: %.2f%%", flutter_freq, flutter_depth * 100)
    logger.debug("Total pitch variation: %.2f%%", (wow_depth + flutter_depth) * 100)
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE (Aggressive correction)"),
        (MaterialType.VINYL, "VINYL (Moderate correction)"),
        (MaterialType.SHELLAC, "SHELLAC (Conservative correction)"),
    ]

    for material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug("Material: %s", material_name)
        logger.debug("─" * 80)
        logger.debug("")

        phase = WowFlutterFix()
        result = phase.process(audio, sample_rate, material.value)

        if result.metrics["wow_flutter_detected"]:
            logger.debug("✅ Professional Wow & Flutter Correction:")
            logger.debug("   erkannt: YES")
            logger.debug("   Max Deviation: %.3f%%", result.metrics["max_deviation_percent"])
            logger.debug("   Wow Magnitude: %.3f%%", result.metrics["wow_magnitude_percent"])
            logger.debug("   Flutter Magnitude: %.3f%%", result.metrics["flutter_magnitude_percent"])
            logger.debug("   Residual Deviation: %.3f%% (target <0.3%%)", result.metrics["residual_deviation_percent"])
            logger.debug("   Correction Strength: %s", result.metrics["correction_strength"])
            logger.debug("   Mean Confidence: %.2f", result.metrics["mean_confidence"])
            logger.debug(
                "   Processing time: %.3fs (%.2f\u00d7 realtime)",
                result.execution_time_seconds,
                result.execution_time_seconds / duration,
            )
            logger.debug("")
        else:
            logger.debug("⚠️  No significant wow/flutter erkannt")
            logger.debug("   Max Deviation: %.3f%%", result.metrics["max_deviation_percent"])
            logger.debug("   Schwelle: %s%%", phase.DETECTION_THRESHOLD.get(material, 0.5))  # _mat_enum_
            logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test abgeschlossen")
    logger.debug("=" * 80)


if __name__ == "__main__":
    _run_test()
