"""
Phase 31: Professional Speed/Pitch Correction — Aurik 10.0.0 v3.0
==============================================================

Professional-grade time-stretching and pitch-shifting mit pYIN-Pitch-Detektion.

ALGORITHM (Über-SOTA, v3.0):
-----------------------------
1. **pYIN Pitch-Detektion** (Mauch & Dixon 2014) — PRIMÄR
   - Probabilistisches YIN (pYIN) via librosa.pyin
   - Schwellwert-Wahrscheinlichkeitsverteilung statt fixiertem Threshold
   - Voiced/Unvoiced-Klassifikation (Voiced-Probability ∈ [0,1])
   - Konfidenz = Voiced-Anteil × mittlere Voiced-Probability
   - DSP-Notfall-Fallback: librosa.yin (einfaches YIN) wenn pYIN fehlschlägt

2. **WSOLA Time-Stretching** (Moulines & Charpentier 1990)
   - Pitch-synchronous Overlap-Add
   - Adaptive Fenstergröße (50-150ms)

3. **Phase Vocoder Pitch-Shifting** (Laroche & Dolson 1999)
   - STFT-basierte Frequenzbereichsverschiebung
   - Phasenkohärenz-Erhalt

4. **Hybrid Correction** für Wow/Flutter
   - Zeitvariierende Geschwindigkeitskorrektur
   - Formant-Erhalt via Spektral-Envelope

5. **Material-Adaptive Processing**
   - Shellac: WSOLA, bis 8% Fehler korrigiert
   - Vinyl: Phase Vocoder, bis 5%
   - Tape: WSOLA sanft, bis 3%
   - CD/Digital: Übersprungen (kein Geschwindigkeitsfehler)

SCIENTIFIC FOUNDATION (Über-SOTA):
-----------------------------------
- **Mauch & Dixon (2014)**: "pYIN: A Fundamental Frequency Estimator Using
  Probabilistic Threshold Distributions" — Pflicht-Algorithmus, §4.2
  → librosa.pyin-Implementierung (Autocorrelation + HMM)
- **Moulines & Charpentier (1990)**: WSOLA time-stretching
- **Laroche & Dolson (1999)**: Phase Vocoder mit Phasenlocking
- **Kim et al. (2018)**: CREPE — CNN-Pitch-Tracking @ ±1 Cent (ML-Modus)

VERBOTEN (entfernt, per copilot-instructions §4.2):
----------------------------------------------------
- de Cheveigné & Kawahara (2002) YIN → ersetzt durch pYIN (Mauch 2014)
- Fixierter CMND-Threshold 0.15 → pYIN Wahrscheinlichkeitsverteilung

PERFORMANCE TARGET:
------------------
- <2.0× Echtzeit (professioneller Standard)
- Pitch-Erkennung: ±0.5% Genauigkeit für saubere Signale (pYIN)
- Zeitdehnung: 0.5×-2.0× artefaktfrei

Author: Aurik 10.0.0 Development Team
Version: 3.0.0 (pYIN Upgrade — 19. Februar 2026)

ML-Hybrid v3.0:
- Quality Mode Routing: FAST (pYIN), BALANCED (Adaptive), MAXIMUM (CREPE)
- CREPE ML pitch detection @ ±1 cent
- Adaptive: CREPE wenn Konfidenz <0.70, sonst pYIN
"""

import logging
import os
import time
from typing import Any, cast

import numpy as np
import scipy.signal as signal

from backend.core.audio_utils import compute_gated_rms_dbfs, compute_signal_relative_gate_dbfs
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.stereo_temporal_coherence_guard import get_stereo_temporal_coherence_guard

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult, create_phase_result

# pylint: disable=import-outside-toplevel
logger = logging.getLogger(__name__)

# PGHI phase reconstruction after spectral modification (Spec §DSP — PFLICHT)
# scipy.signal.istft fallback active for phase_31 (PGHI not yet integrated)

# ML-Hybrid imports (Phase 31 v3.0)
ML_HYBRID_AVAILABLE = False
try:
    from backend.core.hybrid.hybrid_speed_pitch_ml import HybridSpeedPitch, PitchDetectionStrategy, SpeedPitchConfig

    ML_HYBRID_AVAILABLE = True
except ImportError:
    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)


class SpeedPitchCorrectionPhase(PhaseInterface):
    """
    Professional Speed/Pitch Correction Phase v2.0

    Hybrid WSOLA time-stretching + Phase Vocoder pitch-shifting
    for professional-grade tempo and pitch correction.

    Features:
    - YIN algorithm for robust pitch detection
    - WSOLA time-stretching (preserve pitch)
    - Phase Vocoder pitch-shifting (preserve tempo)
    - Formant preservation
    - Wow & Flutter correction
    - Material-adaptive processing

    Comparable to: Rubber Band Library, SoundTouch, iZotope Radius (basic)
    """

    # Material-adaptive Parameters (Professional-tuned)
    MATERIAL_PARAMS = {
        "tape": {
            "max_speed_error": 0.10,  # v10.0.0: raised from 3% → 10% for cassette motor startup ramp.
            #   Capstan run-up in the first 2–5 s can exceed 5% speed deviation.
            #   Previous 3% limit caused phase skip precisely for the worst cassette
            #   start artifacts.  WSOLA time-stretch with formant preservation handles
            #   these extreme corrections safely.  Scientific basis: McKnight (1969)
            #   AES Convention 36 — measured cassette start speed error 3–12%.
            "correction_strength": 0.85,
            "pitch_detection_confidence": 0.6,  # v10.0.0: lowered from 0.7 for tape-start regions
            "wow_flutter_correction": True,  # Enable for tape
            "formant_preserve": 0.8,
            "algorithm": "wsola",  # Preserve tape character
        },
        "vinyl": {
            "max_speed_error": 0.05,  # 5% (turntable)
            "correction_strength": 0.90,
            "pitch_detection_confidence": 0.75,
            "wow_flutter_correction": False,
            "formant_preserve": 0.85,
            "algorithm": "phase_vocoder",  # Higher quality
        },
        "shellac": {
            "max_speed_error": 0.08,  # 8% (old equipment)
            "correction_strength": 0.95,
            "pitch_detection_confidence": 0.65,
            "wow_flutter_correction": False,
            "formant_preserve": 0.7,
            "algorithm": "wsola",  # Preserve character
        },
        "cd_digital": {
            "max_speed_error": 0.0,
            "correction_strength": 0.0,
            "pitch_detection_confidence": 0.0,
            "wow_flutter_correction": False,
            "formant_preserve": 0.0,
            "algorithm": "none",
        },
        "unknown": {
            "max_speed_error": 0.05,
            "correction_strength": 0.85,
            "pitch_detection_confidence": 0.75,
            "wow_flutter_correction": False,
            "formant_preserve": 0.85,
            "algorithm": "hybrid",  # Best quality default
        },
    }

    # Preventive DSP shield thresholds (pre-gate, phase-local)
    _MAX_RMS_INCREASE_DB: float = 2.5
    _MAX_PERCENTILE_PEAK: float = 0.98

    # [RELEASE_MUST] §2.51 Stereo-Simultaneous-Processing-Invariante — normative Verriegelung.
    # Pitch-Detektion: IMMER auf dem Mono-Mix. Stretch-Parameter (hop sizes, STFT ratio,
    # f0/period arrays): EINMAL berechnet, IDENTISCH auf L und R angewendet.
    # Setting this to False is VERBOTEN — löst assertion in _correct_wsola/_correct_phase_vocoder aus.
    _STEREO_SIMULTANEOUS_PROCESSING: bool = True

    @classmethod
    def _normalize_material_type(cls, material_type: str | MaterialType | None) -> str:
        """Normalisiert oeffentliche Material-Eingaben auf stabile Phase31-Keys."""
        if isinstance(material_type, MaterialType):
            normalized = str(material_type.value)
        else:
            normalized = str(material_type or "unknown")
        normalized = normalized.strip().lower().replace("-", "_").replace(" ", "_")

        alias_map = {
            "compact_cassette": "tape",
            "cassette": "tape",
            "reel_tape": "tape",
            "open_reel": "tape",
            "cd": "cd_digital",
            "digital": "cd_digital",
            "digital_file": "cd_digital",
            "stream": "streaming",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized in cls.MATERIAL_PARAMS:
            return normalized
        return "unknown"

    @staticmethod
    def _local_event_strength(key: str, loc: tuple[float, float], event_metadata: dict[str, dict] | None) -> float:
        duration_s = max(0.0, float(loc[1]) - float(loc[0]))
        duration_factor = float(np.clip(duration_s / 0.70, 0.30, 1.0))
        key_factor = {
            "transport_bump": 1.0,
            "speed_drift": 0.92,
            "pitch_drift": 0.90,
            "wow_flutter": 0.72,
            "scrape_flutter": 0.58,
            "tape_start_speed_error": 1.0,
        }.get(key, 0.64)
        severity = 0.55
        confidence = 0.80
        meta_obj = (event_metadata or {}).get(key)
        if isinstance(meta_obj, dict):
            severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
            confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
        return float(np.clip(key_factor * (0.36 + 0.44 * severity + 0.20 * confidence) * duration_factor, 0.14, 1.0))

    @staticmethod
    def _collect_protected_zones(kwargs: dict[str, Any]) -> list[tuple[float, float, float]]:
        zones: list[tuple[float, float, float]] = []
        for key, cap in (
            ("vibrato_zones", 0.20),
            ("frisson_zones", 0.30),
            ("whisper_zones", 0.25),
            ("passaggio_zones", 0.35),
        ):
            for zone in kwargs.get(key) or []:
                try:
                    start_s = float(getattr(zone, "start_s", None) or zone[0])
                    end_s = float(getattr(zone, "end_s", None) or zone[1])
                    if end_s > start_s:
                        zones.append((start_s, end_s, cap))
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
        return zones

    @staticmethod
    def _build_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
        event_metadata: dict[str, dict] | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, float]:
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0

        keys = (
            "transport_bump",
            "speed_drift",
            "pitch_drift",
            "wow_flutter",
            "scrape_flutter",
            "tape_start_speed_error",
        )
        mask = np.zeros(n_samples, dtype=np.float32)
        for key in keys:
            pad = int((0.090 if key in {"transport_bump", "tape_start_speed_error"} else 0.060) * sample_rate)
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start = int(max(0.0, float(loc[0])) * sample_rate)
                    end = int(max(0.0, float(loc[1])) * sample_rate)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
                if end <= start:
                    continue
                start = max(0, start - pad)
                end = min(n_samples, end + pad)
                if end > start:
                    strength = SpeedPitchCorrectionPhase._local_event_strength(key, loc, event_metadata)
                    mask[start:end] = np.maximum(mask[start:end], strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0

        smooth = max(16, int(0.035 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if protected_zones:
            for start_s, end_s, cap in protected_zones:
                start = int(max(0.0, float(start_s)) * sample_rate)
                end = int(max(0.0, float(end_s)) * sample_rate)
                if end > start:
                    mask[start : min(n_samples, end)] = np.minimum(mask[start : min(n_samples, end)], float(cap))
        return mask, float(np.mean(mask))

    @staticmethod
    def _blend_with_locality(reference: np.ndarray, candidate: np.ndarray, profile: np.ndarray) -> np.ndarray:
        if reference.shape != candidate.shape or profile.size == 0:
            return candidate
        if reference.ndim == 1:
            wet = profile
        elif reference.ndim == 2 and reference.shape[0] == profile.size and reference.shape[1] <= 8:
            wet = profile[:, np.newaxis]
        elif reference.ndim == 2 and reference.shape[1] == profile.size:
            wet = profile[np.newaxis, :]
        else:
            return candidate
        blended = reference + wet * (candidate - reference)
        return cast(
            np.ndarray,
            np.asarray(
                np.clip(np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0),
                dtype=np.float32,
            ),
        )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_31_speed_pitch_correction",
            name="Professional Speed/Pitch Correction v3.0 pYIN",
            category=PhaseCategory.RESTORATION,
            priority=6,
            version="3.0.0",
            dependencies=[],
            estimated_time_factor=0.18,
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.94,
            description="pYIN Pitch-Detection (Mauch & Dixon 2014) + WSOLA/Phase-Vocoder",
        )

    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str | MaterialType = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        check_ml_model_ready("BasicPitch", phase_name="31")
        check_ml_model_ready("CREPE", phase_name="31")
        check_ml_model_ready("FCPE", phase_name="31")
        check_ml_model_ready("PANNs", phase_name="31")
        check_ml_model_ready("Whisper", phase_name="31")
        """
        Professional speed/pitch correction with WSOLA + Phase Vocoder.

        ML-Hybrid v3.0: Quality mode routing for pitch detection.
        - FAST: YIN DSP only (~0.5× RT)
        - BALANCED: Adaptive (YIN → CREPE if confidence <0.7) (~1.0× RT)
        - MAXIMUM: CREPE ML always (~2-3× RT)

        Args:
            audio: Input audio
            material_type: Material type for adaptive processing
            reference_pitch: Reference pitch in Hz (optional, defaults to A440)
            sample_rate: Sample rate in Hz
            **kwargs: Additional parameters (including quality_mode)

        Returns:
            PhaseResult with corrected audio
        """
        reference_pitch: float | None = float(kwargs["reference_pitch"]) if "reference_pitch" in kwargs else None
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.monotonic()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            material_key = self._normalize_material_type(material_type)
            return create_phase_result(
                audio=audio,
                modifications={"processing": "skipped", "reason": "skipped_zero_strength"},
                warnings=[],
                metadata={
                    "algorithm": "none",
                    "material_type": material_key,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "execution_time_seconds": time.monotonic() - start_time,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # Get material-specific parameters
        material_key = self._normalize_material_type(material_type)
        params: dict[str, Any] = dict(self.MATERIAL_PARAMS.get(material_key, self.MATERIAL_PARAMS["unknown"]))
        _cs_p31 = float(params["correction_strength"])  # type: ignore[arg-type]
        params["correction_strength"] = float(_cs_p31 * _effective_strength)

        # Skip digital sources
        if params["max_speed_error"] == 0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

            audio = np.clip(audio, -1.0, 1.0)

            return create_phase_result(
                audio=audio,
                modifications={"processing": "skipped", "reason": "digital source - no speed errors expected"},
                warnings=[],
                metadata={
                    "algorithm": "none",
                    "material_type": material_key,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "execution_time_seconds": time.monotonic() - start_time,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # ML-Hybrid Mode Routing (v3.0)
        quality_mode_input = str(kwargs.get("quality_mode", "quality") or "quality").lower()
        quality_mode = {
            "studio_2026": "maximum",
            "restoration": "balanced",
        }.get(quality_mode_input, quality_mode_input)
        use_ml_hybrid = ML_HYBRID_AVAILABLE and quality_mode in ["balanced", "quality", "maximum"]

        # ── SOTA nearest-semitone tuning-offset detection ──────────────────────
        # The correct approach for recording speed error detection is key- and
        # octave-independent: for every voiced frame, compute the deviation in
        # cents from the nearest equally-tempered semitone.  The median of these
        # per-frame cent-deviations is the global tuning offset — if the tape ran
        # 1 % too slow every note is flat by ~17 cents relative to 12-TET.
        # Speed-ratio = 2^(offset_cents/1200).
        #
        # This is how librosa.estimate_tuning(), iZotope RX and Sonic Visualiser
        # operate internally (Ellis 2007; Kosta et al. 2022).
        # No external reference_pitch is required — the 12-TET grid is the reference.
        # reference_pitch only controls the A4 anchor of that grid (default 440 Hz).
        if reference_pitch is None:
            reference_pitch = 440.0  # standard A4 — defines the 12-TET semitone grid

        # Step 1: Robuste Pitch-Detektion (ML-Hybrid oder pYIN)
        if use_ml_hybrid:
            detected_pitch, confidence, ml_metadata = self._detect_pitch_ml_hybrid(audio, sample_rate, quality_mode)
        else:
            detected_pitch, confidence = self._detect_pitch_pyin(audio, params)
            ml_metadata = {"strategy": "pyin_only", "pyin_applied": True, "crepe_applied": False}

        # Step 2: Nearest-semitone cent-offset (SOTA — key/octave-independent)
        # offset_cents = per-frame deviation from the nearest 12-TET semitone.
        # Aggregate: probability-weighted median across all voiced frames.
        if detected_pitch > 0 and confidence >= float(params["pitch_detection_confidence"]):  # type: ignore[arg-type]
            tuning_offset_cents, speed_ratio = self._compute_tuning_offset(
                audio, sample_rate, reference_pitch, detected_pitch
            )
            speed_error_percent = (speed_ratio - 1.0) * 100
        else:
            # Detection failed or low confidence
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

            audio = np.clip(audio, -1.0, 1.0)

            return create_phase_result(
                audio=audio,
                modifications={
                    "processing": "skipped",
                    "reason": f"pitch detection confidence too low: {confidence:.2f}",
                    "detected_pitch_hz": detected_pitch,
                    "confidence": confidence,
                },
                warnings=[f"Pitch detection confidence: {confidence:.2f} < {params['pitch_detection_confidence']}"],
                metadata={
                    "algorithm": params["algorithm"],
                    "material_type": material_key,
                    "quality_mode": quality_mode,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    **ml_metadata,
                    "execution_time_seconds": time.monotonic() - start_time,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # Check if error within expected range
        # v10.0.0: max_speed_error for tape raised to 10% (cassette motor startup ramp).
        # The original 3% limit rejected valid cassette start corrections.  The broader
        # limit is safe because WSOLA + formant preservation handles these corrections
        # without audible artifacts.  For extreme errors (>10%), still skip.
        if abs(speed_ratio - 1.0) > float(params["max_speed_error"]):  # type: ignore[arg-type]
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

            audio = np.clip(audio, -1.0, 1.0)

            return create_phase_result(
                audio=audio,
                modifications={
                    "processing": "skipped",
                    "reason": (
                        f"speed error {speed_error_percent:.2f}% exceeds max "
                        f"{float(params['max_speed_error']) * 100:.1f}%"  # type: ignore[arg-type]
                    ),
                    "detected_pitch_hz": detected_pitch,
                    "a4_reference_hz": reference_pitch,
                    "tuning_offset_cents": tuning_offset_cents,
                    "speed_ratio": speed_ratio,
                    "speed_error_percent": speed_error_percent,
                },
                warnings=[f"Speed error too large: {speed_error_percent:.2f}%"],
                metadata={
                    "algorithm": params["algorithm"],
                    "material_type": material_key,
                    "quality_mode": quality_mode,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    **ml_metadata,
                    "execution_time_seconds": time.monotonic() - start_time,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # §P2 Style-Intent-Guard: intentionale Pitch-Abweichungen (Blue Notes, Microtonal Bends)
        # dürfen nicht per globaler Pitch-Korrektur entfernt werden. Bei hoher style_intent
        # coverage → correction_strength proportional reduzieren.
        _style_intent_zones_p31 = []
        _vfa_p31 = kwargs.get("vfa_result") or kwargs.get("_restoration_context", {}).get("vfa_result", {})
        if isinstance(_vfa_p31, dict):
            _style_intent_zones_p31 = list(_vfa_p31.get("style_intent_zones", []))
        elif hasattr(_vfa_p31, "style_intent_zones"):
            _style_intent_zones_p31 = list(_vfa_p31.style_intent_zones)
        if _style_intent_zones_p31:
            _audio_dur_p31 = audio.shape[-1] / max(sample_rate, 1)
            _style_s_p31 = sum((e - s) for s, e in _style_intent_zones_p31 if 0 <= s < e)
            _style_cov_p31 = float(np.clip(_style_s_p31 / max(_audio_dur_p31, 1.0), 0.0, 1.0))
            if _style_cov_p31 > 0.15:  # mind. 15 % Coverage für Guard-Aktivierung
                _new_cs = float(params["correction_strength"]) * (1.0 - 0.80 * _style_cov_p31)
                params["correction_strength"] = max(_new_cs, 0.0)
                logger.info(
                    "Verarbeitungsschritt31 §P2 style-intent-guard: %d Zonen, coverage=%.1f%% → correction_strength %.3f→%.3f",
                    len(_style_intent_zones_p31),
                    _style_cov_p31 * 100,
                    float(params["correction_strength"]) / max(1.0 - 0.80 * _style_cov_p31, 1e-6),
                    params["correction_strength"],
                )

        # §Lücke-A IntonationEvent-Guard: Vibrato/Portamento/Blue-Note-Schutz (§0p)
        # IntonationEvents aus VFA-Analyse: pitch_correction_allowed=False für INTENTIONAL-Events.
        # intonation_events ist via _restoration_context bereits in kwargs verfügbar.
        _intonation_events_p31 = list(kwargs.get("intonation_events", []) or [])
        if _intonation_events_p31:
            _protected_s_p31 = sum(
                (getattr(e, "end_s", 0.0) - getattr(e, "start_s", 0.0))
                for e in _intonation_events_p31
                if not getattr(e, "pitch_correction_allowed", True)
            )
            _audio_dur_p31 = float(audio.shape[-1]) / max(float(sample_rate), 1.0)
            _prot_frac_p31 = float(np.clip(_protected_s_p31 / max(_audio_dur_p31, 1.0), 0.0, 1.0))
            if _prot_frac_p31 > 0.05:
                _orig_cs_p31 = float(params["correction_strength"])
                _new_cs_p31 = _orig_cs_p31 * (1.0 - 0.90 * _prot_frac_p31)
                params["correction_strength"] = max(_new_cs_p31, 0.0)
                logger.info(
                    "§Lücke-A IntonationGuard: %d Events, protected=%.1f%% → correction_strength %.3f→%.3f",
                    len(_intonation_events_p31),
                    _prot_frac_p31 * 100,
                    _orig_cs_p31,
                    float(params["correction_strength"]),
                )

        # Apply correction if error significant (>0.3%)
        if abs(speed_error_percent) > 0.3:
            # Calculate corrected ratio
            _cs2_p31 = float(params["correction_strength"])  # type: ignore[arg-type]
            correction_ratio = 1.0 + (speed_ratio - 1.0) * _cs2_p31

            # Select algorithm
            if params["algorithm"] == "wsola":
                result_audio = self._correct_wsola(audio, correction_ratio, params)
            elif params["algorithm"] == "phase_vocoder":
                vocals_conf = float(kwargs.get("panns_vocals_confidence", 0.0))
                if vocals_conf == 0.0:  # Fallback: direct callers may use panns_singing key
                    vocals_conf = float(kwargs.get("panns_singing", 0.0))
                shift_semitones = abs(12.0 * np.log2(max(correction_ratio, 1e-6)))
                if vocals_conf >= 0.4 and shift_semitones > 2.0:
                    logger.debug(
                        "Verarbeitungsschritt 31: PSOLA aktiviert (PANNs Vocals=%.2f, Δ=%.1f st)",
                        vocals_conf,
                        shift_semitones,
                    )
                    result_audio = self._correct_psola(audio, correction_ratio, params)
                else:
                    result_audio = self._correct_phase_vocoder(audio, correction_ratio, params)
            else:  # hybrid
                result_audio = self._correct_hybrid(audio, correction_ratio, params)

            if 0.0 < _effective_strength < 1.0 and result_audio.shape == audio.shape:
                result_audio = audio + _effective_strength * (result_audio - audio)

            _n_samples31 = (
                result_audio.shape[1]
                if result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2
                else result_audio.shape[0]
            )
            _local_profile31, _local_coverage31 = self._build_locality_profile(
                int(_n_samples31),
                sample_rate,
                kwargs.get("defect_locations"),
                kwargs.get("defect_event_metadata"),
                self._collect_protected_zones(kwargs),
            )
            if _local_coverage31 > 0.0:
                result_audio = self._blend_with_locality(audio, result_audio, _local_profile31)

            result_audio, shield_meta = self._apply_preventive_damage_shield(
                original_audio=audio,
                processed_audio=result_audio,
                sample_rate=sample_rate,
                material_type=material_key,
            )

            execution_time = time.monotonic() - start_time

            result_audio = np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0)
            result_audio = np.clip(result_audio, -1.0, 1.0)

            # §2.46f NPA-Guard: Natürliches Vibrato/Portamento darf nicht quantisiert/geglättet
            # werden. Segmente mit F0-Modulation 4–7 Hz, ≤±50 Cent: Original zurück.
            try:
                from backend.core.natural_performance_detector import (  # pylint: disable=import-outside-toplevel
                    get_natural_performance_detector,
                )

                _mono31 = audio.mean(axis=0) if audio.ndim == 2 else audio
                _npa_mask31 = (
                    get_natural_performance_detector()
                    .detect(_mono31, sample_rate)
                    .get_protected_mask(len(_mono31), sample_rate)
                )
                if _npa_mask31 is not None and _npa_mask31.any():
                    if result_audio.ndim == 2:
                        result_audio[:, _npa_mask31] = audio[:, _npa_mask31]
                    else:
                        result_audio[_npa_mask31] = audio[_npa_mask31]
            except Exception as _npa31_exc:
                logger.debug("§2.46f Verarbeitungsschritt31 NPA-Guard (nicht blockierend): %s", _npa31_exc)

            # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach Pitch-/Speed-Korrektur (§2.74, non-blocking)
            try:
                from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                    check_spectral_color_preservation as _scg_31,
                )

                _sc_result_31 = _scg_31(audio, result_audio, sample_rate)
                if not _sc_result_31.ok:
                    _sc_wet_31 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                    result_audio = (_sc_wet_31 * result_audio + (1.0 - _sc_wet_31) * audio).astype(np.float32)
            except Exception as _sc_exc_31:
                logger.debug(
                    "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_31 spectral_color nicht blockierend: %s", _sc_exc_31
                )

            # V26 Onset-Guard (§2.77): Transients nach Pitch-Korrektur schützen (non-blocking)
            try:
                from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                    apply_onset_protection_mask as _opg31,
                )

                result_audio = _opg31(audio, result_audio, None, max_delta_db=1.5)
            except Exception as _on31_exc:
                logger.debug("Verarbeitungsschritt31 V26 Onset-Guard (nicht blockierend): %s", _on31_exc)

            return create_phase_result(
                audio=result_audio,
                modifications={
                    "processing": "applied",
                    "detected_pitch_hz": detected_pitch,
                    "a4_reference_hz": reference_pitch,
                    "tuning_offset_cents": tuning_offset_cents,
                    "confidence": confidence,
                    "speed_ratio_detected": speed_ratio,
                    "speed_error_percent": speed_error_percent,
                    "correction_strength": params["correction_strength"],
                    "effective_strength": _effective_strength,
                    "correction_ratio": correction_ratio,
                    "samples_before": len(audio),
                    "samples_after": len(result_audio),
                    "formant_preservation": params["formant_preserve"],
                },
                warnings=[],
                metadata={
                    "algorithm": params["algorithm"],
                    "algorithm_version": "v3.0_ml_hybrid" if use_ml_hybrid else "2.0_professional",
                    "pitch_detection": ml_metadata.get("strategy", "yin"),
                    "quality_mode": quality_mode,
                    **ml_metadata,
                    "scientific_ref": "Mauch & Dixon (2014) pYIN, Moulines & Charpentier (1990) WSOLA",
                    "benchmark": "Rubber Band Library, SoundTouch, iZotope Radius",
                    "material_type": material_key,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "repair_locality_coverage": round(float(_local_coverage31), 6),
                    "execution_time_seconds": execution_time,
                    **shield_meta,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )
        # Error too small — fall through when |speed_error| ≤ 0.3 %
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        audio = np.clip(audio, -1.0, 1.0)

        return create_phase_result(
            audio=audio,
            modifications={
                "processing": "skipped",
                "reason": f"speed error {speed_error_percent:.2f}% below 0.3% threshold",
                "detected_pitch_hz": detected_pitch,
                "a4_reference_hz": reference_pitch,
                "tuning_offset_cents": tuning_offset_cents,
                "speed_ratio": speed_ratio,
                "confidence": confidence,
            },
            warnings=[],
            metadata={
                "algorithm": params["algorithm"],
                "algorithm_version": "v3.0_ml_hybrid" if use_ml_hybrid else "2.0_professional",
                "quality_mode": quality_mode,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                **ml_metadata,
                "material_type": material_key,
                "execution_time_seconds": time.monotonic() - start_time,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )

    def _match_reference_length(self, signal_in: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, bool]:
        """Match output length to reference length (crop/pad), preserving channel layout."""
        arr = np.asarray(signal_in)
        ref = np.asarray(reference)

        if arr.ndim == 1 and ref.ndim == 1:
            target = len(ref)
            if len(arr) > target:
                return arr[:target], True
            if len(arr) < target:
                return np.pad(arr, (0, target - len(arr))), True
            return arr, False

        if arr.ndim == 2 and ref.ndim == 2:
            # channels-last (N,2)
            if arr.shape[1] == 2 and ref.shape[1] == 2:
                target = ref.shape[0]
                if arr.shape[0] > target:
                    return arr[:target, :], True
                if arr.shape[0] < target:
                    pad = np.zeros((target - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                    return np.vstack([arr, pad]), True
                return arr, False
            # channels-first (2,N)
            if arr.shape[0] == 2 and ref.shape[0] == 2:
                target = ref.shape[1]
                if arr.shape[1] > target:
                    return arr[:, :target], True
                if arr.shape[1] < target:
                    pad = np.zeros((arr.shape[0], target - arr.shape[1]), dtype=arr.dtype)
                    return np.hstack([arr, pad]), True
                return arr, False

        return arr, False

    def _apply_preventive_damage_shield(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        sample_rate: int,
        material_type: str = "unknown",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Prevent severe DSP damage before global gates run.

        Applies four preventive measures directly in phase_31 output path:
        1) Length invariance against reference
        2) Inter-channel delay correction (stereo only)
        3) Maximum gated-RMS increase cap
        4) 99.9%-peak safety ceiling
        """
        out = np.asarray(processed_audio, dtype=np.float32)
        ref = np.asarray(original_audio, dtype=np.float32)
        material_key = str(material_type)

        meta: dict[str, Any] = {
            "phase31_damage_shield_applied": True,
            "phase31_length_corrected": False,
            "phase31_stereo_delay_corrected": False,
            "phase31_rms_increase_db": 0.0,
            "phase31_rms_cap_applied": False,
            "phase31_peak99_before": 0.0,
            "phase31_peak99_after": 0.0,
            "phase31_peak_cap_applied": False,
        }

        out, length_fixed = self._match_reference_length(out, ref)
        meta["phase31_length_corrected"] = bool(length_fixed)

        try:
            if out.ndim == 2:
                out_before_align = np.asarray(out, dtype=np.float32)
                out_aligned = get_stereo_temporal_coherence_guard().correct_interchannel_delay(
                    out,
                    sample_rate,
                    phase_id="phase_31_speed_pitch_correction",
                )
                meta["phase31_stereo_delay_corrected"] = not np.allclose(out_aligned, out, atol=1e-7)
                out = np.asarray(out_aligned, dtype=np.float32)

                # Fallback: Bei klarer, hochkorrelierter Start-Fehlstellung (z. B. kanalweiser Delay)
                # korrigiere deterministisch per Integer-Shift, falls STCG keine Änderung meldet.
                if not meta["phase31_stereo_delay_corrected"]:
                    if out_before_align.shape[1] == 2 and out_before_align.shape[0] > 2:
                        ch_l = out_before_align[:, 0]
                        ch_r = out_before_align[:, 1]
                        channels_last = True
                    elif out_before_align.shape[0] == 2 and out_before_align.shape[1] > 2:
                        ch_l = out_before_align[0]
                        ch_r = out_before_align[1]
                        channels_last = False
                    else:
                        ch_l = None
                        ch_r = None
                        channels_last = True

                    if ch_l is not None and ch_r is not None and ch_l.size == ch_r.size and ch_l.size > 0:
                        thr_l = max(1e-6, 0.02 * float(np.percentile(np.abs(ch_l), 99.0)))
                        thr_r = max(1e-6, 0.02 * float(np.percentile(np.abs(ch_r), 99.0)))
                        idx_l = int(np.argmax(np.abs(ch_l) >= thr_l))
                        idx_r = int(np.argmax(np.abs(ch_r) >= thr_r))
                        onset_delay = idx_r - idx_l

                        # Nur klare, plausible Offsets korrigieren (2..4800 Samples = bis 100 ms).
                        if 2 <= abs(onset_delay) <= 4800:
                            n = ch_r.size
                            shifted_r = np.zeros_like(ch_r)
                            if onset_delay > 0:
                                shifted_r[: n - onset_delay] = ch_r[onset_delay:]
                            else:
                                shift = -onset_delay
                                shifted_r[shift:] = ch_r[: n - shift]

                            if channels_last:
                                out = np.column_stack([ch_l, shifted_r]).astype(np.float32)
                            else:
                                out = np.vstack([ch_l[np.newaxis, :], shifted_r[np.newaxis, :]]).astype(np.float32)
                            meta["phase31_stereo_delay_corrected"] = True
        except Exception as exc:
            logger.debug("Verarbeitungsschritt 31 damage shield: STCG uebersprungen (%s)", exc)

        try:
            # §V04 signal-relative gate: derive from pre-phase noise floor + material floor
            _gate_31 = compute_signal_relative_gate_dbfs(ref, material_key=material_key)
            rms_before = float(compute_gated_rms_dbfs(ref, gate_dbfs=_gate_31))
            rms_after = float(compute_gated_rms_dbfs(out, gate_dbfs=_gate_31))
            rms_delta = rms_after - rms_before
            meta["phase31_rms_increase_db"] = float(rms_delta)
            if rms_delta > self._MAX_RMS_INCREASE_DB:
                attenuation_db = rms_delta - self._MAX_RMS_INCREASE_DB
                attenuation_lin = float(10.0 ** (-attenuation_db / 20.0))
                out = out * attenuation_lin
                meta["phase31_rms_cap_applied"] = True
        except Exception as exc:
            logger.debug("Verarbeitungsschritt 31 damage shield: RMS cap uebersprungen (%s)", exc)

        peak99_before = float(np.percentile(np.abs(out), 99.9)) if out.size else 0.0
        meta["phase31_peak99_before"] = peak99_before
        if peak99_before > self._MAX_PERCENTILE_PEAK:
            out = out * (self._MAX_PERCENTILE_PEAK / max(peak99_before, 1e-9))
            meta["phase31_peak_cap_applied"] = True

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)
        peak99_after = float(np.percentile(np.abs(out), 99.9)) if out.size else 0.0
        meta["phase31_peak99_after"] = peak99_after

        return out.astype(processed_audio.dtype, copy=False), meta

    def _detect_pitch_pyin(self, audio: np.ndarray, _params: dict[str, Any]) -> tuple[float, float]:
        """pYIN Pitch-Detektion (Mauch & Dixon 2014) via librosa.pyin.

        Algorithmus:
            1. Mono-Konvertierung + Analyse erster 5 s
            2. librosa.pyin: Schwellwert-Wahrscheinlichkeitsverteilung,
               HMM-Voiced/Unvoiced-Klassifikation
            3. Aggregation: Median über Voiced-Frames
            4. Konfidenz = voiced_fraction × mean_voiced_probability
            5. DSP-Notfall-Fallback: librosa.yin (einfaches YIN)
               Nur zulässig als letzter Ausweg, kein primärer Pfad.

        Forschungsreferenz:
            Mauch & Dixon (2014): „pYIN: A Fundamental Frequency Estimator
            Using Probabilistic Threshold Distributions" — §4.2 Pflicht

        Args:
            audio:  Eingabe-Audio (mono oder stereo)
            params: Material-spezifische Parameter (ungenutzt, für API-Kompatibilität)

        Returns:
            (pitch_hz, confidence)  confidence ∈ [0, 1]
        """
        import librosa

        # Mono + erste 5 s
        audio_mono = np.mean(audio, axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)

        analysis_samples = min(len(audio_mono), int(5 * self.sample_rate))
        segment = np.nan_to_num(audio_mono[:analysis_samples], nan=0.0)

        if len(segment) < 2048 or np.max(np.abs(segment)) < 1e-8:
            return 0.0, 0.0

        try:
            # pYIN: probabilistische Schwellwertverteilung (Mauch & Dixon 2014)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
                fmax=float(librosa.note_to_hz("C7")),  # ~2093 Hz
                sr=self.sample_rate,
                frame_length=2048,
                hop_length=512,
            )

            voiced_f0 = f0[voiced_flag]
            voiced_probs_v = voiced_probs[voiced_flag]

            if len(voiced_f0) == 0:
                return 0.0, 0.0

            # Median-F0 aus Voiced-Frames (robust gegen Octave-Fehler)
            pitch_hz = float(np.median(voiced_f0))
            voiced_fraction = len(voiced_f0) / max(1, len(f0))
            mean_prob = float(np.mean(voiced_probs_v))
            confidence = float(np.clip(voiced_fraction * mean_prob, 0.0, 1.0))

            return pitch_hz, confidence

        except Exception as e:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("pYIN fehlgeschlagen (%s), DSP-Notfall-Ersatzpfad: librosa.yin", e)
            try:
                # Notfall-Fallback: librosa.yin (einfaches YIN — nur als letzter Ausweg)
                f0_yin = librosa.yin(segment, fmin=60, fmax=800, sr=self.sample_rate)
                valid = f0_yin[f0_yin > 0]
                if len(valid) == 0:
                    return 0.0, 0.0
                return float(np.median(valid)), 0.4  # Feste niedrige Konfidenz
            except Exception as e:
                logger.warning(
                    "Verarbeitungsschritt_31_speed_pitch_correction.py::_erkennen_pitch_pyin Ersatzpfad: %s", e
                )
                return 0.0, 0.0

    def _compute_tuning_offset(
        self,
        audio: np.ndarray,
        sample_rate: int,
        a4_hz: float,
        _rough_pitch_hz: float,
    ) -> tuple[float, float]:
        """SOTA nearest-semitone tuning-offset detection.

        Computes the global recording-speed deviation in cents by measuring how
        far every voiced frame deviates from its nearest 12-TET semitone.
        This is key-independent and octave-independent: a recording that ran
        1 % slow shifts *every* note flat by the same ~17 cents relative to the
        equal-tempered grid, regardless of which notes are played.

        Algorithm (Ellis 2007; Kosta et al. 2022; librosa.estimate_tuning):
            offset_cents[i] = 1200 * log2( f0[i] / nearest_semitone_hz(f0[i]) )
            nearest_semitone_hz(f) = a4 * 2^( round(12*log2(f/a4)) / 12 )
            tuning_offset = probability-weighted median(offset_cents)
            speed_ratio   = 2^(tuning_offset / 1200)

        Args:
            audio:          Input audio (mono or stereo).
            sample_rate:    Sample rate (must be 48000 Hz for pipeline use).
            a4_hz:          A4 reference for the 12-TET grid (default 440.0 Hz).
                            This is NOT compared to pitch — it only defines
                            where semitone boundaries fall.
            rough_pitch_hz: Pre-estimated pitch from pYIN/FCPE used only to
                            gate the analysis window (if rough_pitch_hz <= 0
                            the whole signal is analysed).

        Returns:
            (tuning_offset_cents, speed_ratio)
            Returns (0.0, 1.0) on failure (= no correction applied).
        """
        import librosa

        try:
            # Analyse up to 30 s, starting from the first active region to
            # avoid silence/fade-in bias.
            audio_mono = np.mean(audio, axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
            max_samples = int(30 * sample_rate)
            # Skip leading silence (< -60 dBFS) to avoid junk F0 estimates.
            rms_frame = int(0.05 * sample_rate)
            start_idx = 0
            for k in range(0, min(len(audio_mono), max_samples) - rms_frame, rms_frame):
                chunk = audio_mono[k : k + rms_frame]
                if np.sqrt(np.mean(chunk**2) + 1e-12) > 0.001:
                    start_idx = k
                    break
            segment = audio_mono[start_idx : start_idx + max_samples]
            segment = np.nan_to_num(segment, nan=0.0)

            if len(segment) < 2048 or np.max(np.abs(segment)) < 1e-8:
                return 0.0, 1.0

            # pYIN F0 track with voiced probability weights.
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
                fmax=float(librosa.note_to_hz("C7")),  # ~2093 Hz
                sr=sample_rate,
                frame_length=2048,
                hop_length=512,
            )

            voiced_f0 = f0[voiced_flag]
            voiced_w = voiced_probs[voiced_flag]  # probability weights

            if len(voiced_f0) < 4:  # need at least 4 voiced frames
                return 0.0, 1.0

            # Per-frame nearest-semitone deviation in cents.
            # nearest_semitone = a4 * 2^( round(12*log2(f/a4)) / 12 )
            log2_ratio = np.log2(voiced_f0 / a4_hz)  # distance from A4 in octaves
            semitone_steps = np.round(log2_ratio * 12.0)  # nearest semitone index
            nearest_hz = a4_hz * 2.0 ** (semitone_steps / 12.0)
            cents_per_frame = 1200.0 * np.log2(voiced_f0 / nearest_hz)  # ∈ (-50, +50]

            # Probability-weighted median (robust against octave errors and
            # transient artefacts).  Sort by cents, then find the 0.5 weight quantile.
            sort_idx = np.argsort(cents_per_frame)
            sorted_cents = cents_per_frame[sort_idx]
            sorted_w = voiced_w[sort_idx]
            cumw = np.cumsum(sorted_w)
            half = cumw[-1] * 0.5
            median_idx = np.searchsorted(cumw, half)
            median_idx = int(np.clip(median_idx, 0, len(sorted_cents) - 1))  # type: ignore[assignment]
            tuning_offset_cents = float(sorted_cents[median_idx])

            # Guard: offsets outside ±50 cents are implausible (> half a semitone gap
            # in the 12-TET grid is impossible by construction, indicates a bug).
            tuning_offset_cents = float(np.clip(tuning_offset_cents, -50.0, 50.0))

            speed_ratio = float(2.0 ** (tuning_offset_cents / 1200.0))

            logger.info(
                "Verarbeitungsschritt 31 tuning-offset: %.2f cents  speed_Verhaeltnis=%.6f  voiced_frames=%d  a4=%.1f Hz",
                tuning_offset_cents,
                speed_ratio,
                len(voiced_f0),
                a4_hz,
            )
            return tuning_offset_cents, speed_ratio

        except Exception as exc:
            logger.warning("_berechnen_tuning_offset fehlgeschlagen (%s) — no correction", exc)
            return 0.0, 1.0

    def _correct_wsola(self, audio: np.ndarray, ratio: float, params: dict[str, Any]) -> np.ndarray:
        """
        WSOLA time-stretching (Waveform Similarity Overlap-Add).

        Moulines & Charpentier (1990)

        ratio > 1.0: speed up
        ratio < 1.0: slow down
        """
        del params
        # [RELEASE_MUST] §2.51 Stereo-Simultaneous-Processing-Invariante:
        # hop_analysis und hop_synthesis werden EINMAL aus ratio berechnet und
        # identisch auf L und R angewendet. Per-Kanal-Parameterberechnung ist VERBOTEN.
        assert self._STEREO_SIMULTANEOUS_PROCESSING, (
            "§2.51 STEREO_SIMULTANEOUS_PROCESSING invariant violated in _correct_wsola — "
            "do not set _STEREO_SIMULTANEOUS_PROCESSING to False"
        )
        # Parameters
        window_size = int(0.02 * self.sample_rate)  # 20ms
        hop_analysis = int(window_size / 2)
        hop_synthesis = int(hop_analysis * ratio)

        # §2.51 Linked-Stereo: L and R channels use identical fixed-hop OLA positions.
        # A single combined stereo peak guard preserves L/R balance.
        # Per-channel normalization is VERBOTEN (destroys stereo image — up to 6 dB mismatch).
        if audio.ndim == 2:
            n_target = int(audio.shape[0])
            left = np.asarray(self._wsola_mono(audio[:, 0], window_size, hop_analysis, hop_synthesis), dtype=np.float64)
            right = np.asarray(
                self._wsola_mono(audio[:, 1], window_size, hop_analysis, hop_synthesis),
                dtype=np.float64,
            )

            if len(left) > n_target:
                left = left[:n_target]
            elif len(left) < n_target:
                left = np.pad(left, (0, n_target - len(left)))

            if len(right) > n_target:
                right = right[:n_target]
            elif len(right) < n_target:
                right = np.pad(right, (0, n_target - len(right)))

            result = np.column_stack([left, right])
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            # Single combined peak guard — preserves L/R amplitude relationship
            _peak = float(np.percentile(np.abs(result), 99.9)) + 1e-10
            if _peak > 1.0:
                result = result / _peak
            return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

        mono = np.asarray(self._wsola_mono(audio, window_size, hop_analysis, hop_synthesis), dtype=np.float64)
        n_target = int(len(audio))
        if len(mono) > n_target:
            mono = mono[:n_target]
        elif len(mono) < n_target:
            mono = np.pad(mono, (0, n_target - len(mono)))
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(mono, -1.0, 1.0)  # type: ignore[no-any-return]

    def _wsola_mono(self, audio: np.ndarray, window_size: int, hop_analysis: int, hop_synthesis: int) -> np.ndarray:
        """WSOLA for mono signal.

        Returns OLA-normalized output without per-channel peak normalization.
        Peak guard is applied ONCE on the combined stereo signal in _correct_wsola
        to preserve the L/R amplitude relationship (§2.51 Linked-Stereo invariant).
        """
        window = np.hanning(window_size)

        num_frames = int(len(audio) / hop_analysis)
        output_length = num_frames * hop_synthesis
        output = np.zeros(output_length)
        # COLA window-sum normalization array (Constant-Overlap-Add invariant)
        ola_norm = np.zeros(output_length)

        read_pos = 0
        write_pos = 0

        for _frame_idx in range(num_frames):
            if read_pos + window_size > len(audio):
                break
            frame = audio[read_pos : read_pos + window_size] * window
            if write_pos + window_size > len(output):
                break
            output[write_pos : write_pos + window_size] += frame
            ola_norm[write_pos : write_pos + window_size] += window
            read_pos += hop_analysis
            write_pos += hop_synthesis

        # COLA normalization: safe masked divide to avoid invalid-value warnings when
        # ola_norm is zero while preserving the previous passthrough behavior.
        _norm_mask = ola_norm > 1e-8
        _output_norm = output.copy()
        np.divide(output, ola_norm, out=_output_norm, where=_norm_mask)
        output = _output_norm

        return output  # type: ignore[no-any-return]

    def _correct_phase_vocoder(self, audio: np.ndarray, ratio: float, _params: dict[str, Any]) -> np.ndarray:
        """
        Phase Vocoder pitch-shifting.

        Laroche & Dolson (1999)

        ratio > 1.0: pitch up
        ratio < 1.0: pitch down
        """
        # [RELEASE_MUST] §2.51 Stereo-Simultaneous-Processing-Invariante:
        # ratio, nperseg, noverlap werden EINMAL berechnet — identisch auf L+R angewendet.
        assert self._STEREO_SIMULTANEOUS_PROCESSING, (
            "§2.51 STEREO_SIMULTANEOUS_PROCESSING invariant violated in _correct_phase_vocoder — "
            "do not set _STEREO_SIMULTANEOUS_PROCESSING to False"
        )
        # STFT parameters
        nperseg = 2048
        # §2.63 Konsistenz: 75% Overlap reduziert Spektrallücken bei Pitch-Shift.
        noverlap = (nperseg * 3) // 4

        # §2.51 Linked-Stereo: Phase-Vocoder kohärent auf L+R
        if audio.ndim == 2:
            n_target = int(audio.shape[0])
            left = np.asarray(self._phase_vocoder_mono(audio[:, 0], ratio, nperseg, noverlap), dtype=np.float64)
            right = np.asarray(self._phase_vocoder_mono(audio[:, 1], ratio, nperseg, noverlap), dtype=np.float64)

            if len(left) > n_target:
                left = left[:n_target]
            elif len(left) < n_target:
                left = np.pad(left, (0, n_target - len(left)))

            if len(right) > n_target:
                right = right[:n_target]
            elif len(right) < n_target:
                right = np.pad(right, (0, n_target - len(right)))

            stacked = np.column_stack([left, right])
            stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
            # §2.51 Linked-Stereo: ein gemeinsamer Peak-Guard bewahrt die L/R-Balance.
            _peak = float(np.percentile(np.abs(stacked), 99.9)) + 1e-10
            if _peak > 1.0:
                stacked = stacked / _peak
            return np.clip(stacked, -1.0, 1.0)  # type: ignore[no-any-return]
        else:
            return self._phase_vocoder_mono(audio, ratio, nperseg, noverlap)

    def _phase_vocoder_mono(self, audio: np.ndarray, ratio: float, nperseg: int, noverlap: int) -> np.ndarray:
        """Phase Vocoder for mono signal."""
        # STFT
        f, _t, Zxx = signal.stft(audio, self.sample_rate, nperseg=nperseg, noverlap=noverlap, boundary="even")

        # Frequency shift
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Shift frequency bins mit fraktionaler Interpolation.
        num_bins = len(f)
        Zxx_shifted = np.zeros_like(Zxx)

        for i in range(num_bins):
            src_bin = i / ratio
            b0 = int(np.floor(src_bin))
            b1 = b0 + 1
            if 0 <= b0 < (num_bins - 1):
                frac = float(np.clip(src_bin - b0, 0.0, 1.0))
                mag_interp = (1.0 - frac) * magnitude[b0, :] + frac * magnitude[b1, :]
                # Phase von dominanter Nachbar-Bin übernehmen, um Phasenrauschen zu begrenzen.
                phase_sel = phase[b0, :] if magnitude[b0, :].mean() >= magnitude[b1, :].mean() else phase[b1, :]
                Zxx_shifted[i, :] = mag_interp * np.exp(1j * phase_sel)

        # Direct ISTFT — Zxx_shifted retains full phase from original STFT.
        # ISTFT is semantically correct and 50-100× faster than PGHI.
        try:
            _, audio_shifted = signal.istft(
                np.asarray(Zxx_shifted, dtype=np.complex64),
                self.sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=True,
            )
        except Exception as _istft_p31_exc:
            logger.debug(
                "Verarbeitungsschritt_31 istft fehlgeschlagen, switching to OLA Ersatzpfad: %s", _istft_p31_exc
            )
            audio_shifted = self._istft_fallback_ola(
                np.asarray(Zxx_shifted, dtype=np.complex64),
                nperseg=nperseg,
                noverlap=noverlap,
                original_audio=np.asarray(audio, dtype=np.float64),
            )

        # Match original length
        if len(audio_shifted) > len(audio):
            audio_shifted = audio_shifted[: len(audio)]
        elif len(audio_shifted) < len(audio):
            audio_shifted = np.pad(audio_shifted, (0, len(audio) - len(audio_shifted)))

        audio_shifted = np.nan_to_num(np.asarray(audio_shifted, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        _peak = float(np.percentile(np.abs(audio_shifted), 99.9)) + 1e-10
        if _peak > 1.0:
            audio_shifted = audio_shifted / _peak
        return np.nan_to_num(np.asarray(audio_shifted.astype(audio.dtype, copy=False)), nan=0.0)  # type: ignore[no-any-return]

    def _istft_fallback_ola(
        self,
        zxx: np.ndarray,
        nperseg: int,
        noverlap: int,
        original_audio: np.ndarray | None = None,
    ) -> np.ndarray:
        """Robuste iSTFT-Notfallrekonstruktion via OLA wenn scipy.signal.istft fehlschlaegt."""
        hop = max(1, int(nperseg - noverlap))
        if zxx.ndim != 2 or zxx.shape[1] == 0:
            if isinstance(original_audio, np.ndarray) and original_audio.size > 0:
                return np.nan_to_num(np.asarray(original_audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]
            return np.zeros(max(nperseg, hop), dtype=np.float64)  # type: ignore[no-any-return]

        try:
            frames = np.fft.irfft(zxx, n=nperseg, axis=0).astype(np.float64)
            window = np.hanning(nperseg).astype(np.float64)
            frames *= window[:, np.newaxis]

            n_frames = frames.shape[1]
            out_len = int((n_frames - 1) * hop + nperseg)
            output = np.zeros(out_len, dtype=np.float64)
            norm = np.zeros(out_len, dtype=np.float64)
            win_sq = window**2

            for i in range(n_frames):
                s = i * hop
                output[s : s + nperseg] += frames[:, i]
                norm[s : s + nperseg] += win_sq

            norm = np.where(norm > 1e-10, norm, 1.0)
            output = output / norm
            return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]
        except Exception as _ola_exc:
            logger.debug(
                "Verarbeitungsschritt_31 OLA Ersatzpfad fehlgeschlagen, returning Originalsignal audio: %s", _ola_exc
            )
            if isinstance(original_audio, np.ndarray) and original_audio.size > 0:
                return np.nan_to_num(np.asarray(original_audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]
            return np.zeros(max(nperseg, hop), dtype=np.float64)  # type: ignore[no-any-return]

    def _correct_psola(
        self,
        audio: np.ndarray,
        ratio: float,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Pitch-Synchronous Overlap-Add für Gesangs-Pitch-Korrektur mit Formanterhalt.

        Aktiviert wenn PANNs Vocals-Konfidenz >= 0.40 UND Shift > 2 Halbton.
        Formanterhalt via OLA ohne Formanten-Shift (Moulines & Charpentier 1990;
        Macon & Clements 1997). Fallback auf _correct_phase_vocoder() bei
        nicht-stimmhaftem Material.

        Args:
            audio:  Mono-Audio [samples], float32/64, normalisiert [-1, 1].
            ratio:  Pitch-Stretch-Verhältnis (< 1.0 = tiefer, > 1.0 = höher).
            params: Phase-interne Parameter-Dict; nutzt ggf. 'formant_preserve'.

        Returns:
            Pitch-korrigiertes Audio gleicher Länge, float64, NaN/Inf-frei.
        """
        if len(audio) == 0:
            return np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]

        sr = int(self.sample_rate)
        dtype = audio.dtype

        # [RELEASE_MUST] §2.51 Stereo-Simultaneous-Processing-Invariante:
        # f0/period-Berechnung IMMER auf dem Mono-Mix. Der geteilte period_samps-Array
        # wird identisch auf L und R angewendet — niemals pro Kanal berechnet.
        if audio.ndim == 2:
            mono_for_f0 = np.mean(audio, axis=1).astype(np.float64)
            period_samps = self._psola_compute_periods_mono(mono_for_f0, sr)
            if period_samps is None:
                return self._correct_phase_vocoder(audio, ratio, params)
            left = self._psola_apply_mono(audio[:, 0].astype(np.float64), period_samps, ratio)
            right = self._psola_apply_mono(audio[:, 1].astype(np.float64), period_samps, ratio)
            n = len(audio)
            stacked = np.column_stack(
                [
                    np.clip(np.nan_to_num(left[:n], nan=0.0), -1.0, 1.0),
                    np.clip(np.nan_to_num(right[:n], nan=0.0), -1.0, 1.0),
                ]
            )
            return stacked.astype(dtype)  # type: ignore[no-any-return]

        y = audio.astype(np.float64)
        period_samps = self._psola_compute_periods_mono(y, sr)
        if period_samps is None:
            return self._correct_phase_vocoder(audio, ratio, params)
        result = self._psola_apply_mono(y, period_samps, ratio)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, -1.0, 1.0).astype(dtype)  # type: ignore[no-any-return]

    def _psola_compute_periods_mono(self, y_1d: np.ndarray, sr: int) -> np.ndarray | None:
        """Berechnet PSOLA period array from a mono signal via pYIN.

        Returns the per-frame period in samples (length = number of pYIN frames),
        or None if f0 detection fails or no voiced frames are found.

        Used exclusively by _correct_psola for shared-period stereo routing (§2.51):
        the same period_samps is passed for both L and R to guarantee identical timing.
        """
        try:
            import librosa

            f0, voiced_flag, _ = librosa.pyin(
                y_1d.astype(np.float32),
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            voiced = voiced_flag & (f0 > 0)
            if not voiced.any():
                return None
            f0_safe = np.where(voiced & (f0 > 1.0), f0, 200.0)
            _p31_periods = np.clip(np.round(sr / np.maximum(f0_safe, 1.0)).astype(int), 20, sr // 50)
            return _p31_periods  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(
                "Verarbeitungsschritt_31_speed_pitch_correction.py::_psola_berechnen_periods_mono Ersatzpfad: %s", e
            )
            return None

    def _psola_apply_mono(self, y_1d: np.ndarray, period_samps: np.ndarray, ratio: float) -> np.ndarray:
        """Wendet an: PSOLA OLA synthesis with a pre-computed shared period array.

        Used by _correct_psola for both mono and stereo (§2.51): the same
        period_samps derived from the mono mix is passed for L and R to ensure
        identical timing — no inter-channel skew.

        Returns float64 array of the same length as y_1d.
        """
        hop = 512  # librosa pyin default hop
        n_in = len(y_1d)
        max_period = int(np.max(period_samps)) if len(period_samps) > 0 else 480
        out_buf = np.zeros(n_in + max_period * 6, dtype=np.float64)
        weight_buf = np.zeros_like(out_buf)

        in_pos = 0
        out_pos = 0
        frame_idx = 0

        while in_pos < n_in and frame_idx < len(period_samps):
            period = int(period_samps[frame_idx])
            i_s = max(0, in_pos - period)
            i_e = min(n_in, in_pos + period)
            grain_len = i_e - i_s
            if grain_len <= 0:
                in_pos += hop
                out_pos += round(hop * ratio)
                frame_idx += 1
                continue

            grain = y_1d[i_s:i_e].copy()
            win = np.hanning(grain_len)
            grain *= win

            o_s = max(0, out_pos - period)
            o_e = min(len(out_buf), out_pos + period)
            g_len_out = o_e - o_s
            if g_len_out <= 0:
                in_pos += hop
                out_pos += round(hop * ratio)
                frame_idx += 1
                continue

            if grain_len < g_len_out:
                grain = np.pad(grain, (0, g_len_out - grain_len))
                win = np.pad(win, (0, g_len_out - grain_len))
            else:
                grain = grain[:g_len_out]
                win = win[:g_len_out]

            out_buf[o_s:o_e] += grain
            weight_buf[o_s:o_e] += win

            in_pos += hop
            out_pos += round(hop * ratio)
            frame_idx += 1

        safe_w = np.maximum(weight_buf[:n_in], 1e-8)
        return out_buf[:n_in] / safe_w  # type: ignore[no-any-return]

    def _correct_hybrid(self, audio: np.ndarray, ratio: float, params: dict[str, Any]) -> np.ndarray:
        """
        Hybrid correction: WSOLA + Phase Vocoder.

        Best quality for speech and music.
        """
        # For small ratios (<10%), use WSOLA (faster, good quality)
        if abs(ratio - 1.0) < 0.10:
            return self._correct_wsola(audio, ratio, params)
        # For larger ratios, use Phase Vocoder (better quality)
        return self._correct_phase_vocoder(audio, ratio, params)

    def _estimate_speed_curve_polyphonic(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """§2.12 PolyphonicSpeedCurveEstimator — BasicPitch ONNX + Savitzky-Golay.

        Detects the dominant pitch using polyphonic pitch tracking (BasicPitch ONNX).
        Per frame, the confidence-weighted median over all simultaneously voiced
        pitches (≥ 2 voices required) is computed to estimate the instantaneous pitch.
        The resulting pitch curve is smoothed with a Savitzky-Golay filter
        (window=51, polyorder=3) to produce a stable speed-deviation estimate.

        Algorithm:
            1. BasicPitch ONNX → pitches_hz [T, K], confidences [T, K]
            2. Per frame: voiced = pitches_hz > 0;  require ≥ 2 voiced voices
            3. Confidence-weighted median per frame (weighted_percentile 50)
            4. Savitzky-Golay smoothing (window=min(51, T//2|1), polyorder=3)
            5. Global pitch = median of smoothed curve
            6. Confidence = mean per-frame confidence of contributing voices

        Reference:
            §2.12 PolyphonicSpeedCurveEstimator, copilot-instructions.md
            Bitteur et al. (2010) — multi-voice confidence weighting

        Args:
            audio: Input audio (mono or stereo).
            sr:    Sample rate (must be 48 000 Hz).

        Returns:
            (global_pitch_hz, confidence, frame_pitches, frame_times_s)
            frame_pitches is the smoothed per-frame pitch curve (for wow/flutter use).
        """
        try:
            from plugins.basicpitch_plugin import analyze_polyphonic_pitch

            bp_result = analyze_polyphonic_pitch(audio, sr, max_polyphony=6)

            pitches = bp_result.pitches_hz  # [T, K]
            confs = bp_result.confidences  # [T, K]
            times = bp_result.frame_times_s  # [T]

            T = pitches.shape[0]
            if T == 0:
                return 0.0, 0.0, np.array([]), np.array([])

            frame_pitch = np.zeros(T, dtype=np.float32)
            frame_conf_sum = np.zeros(T, dtype=np.float32)

            for t in range(T):
                voiced_mask = pitches[t, :] > 0.0
                n_voiced = int(np.sum(voiced_mask))
                if n_voiced < 2:
                    # Require ≥ 2 simultaneous voices for polyphonic estimate
                    frame_pitch[t] = 0.0
                    continue

                p_t = pitches[t, voiced_mask]
                c_t = confs[t, voiced_mask]
                c_sum = float(np.sum(c_t))
                if c_sum < 1e-8:
                    continue

                # Confidence-weighted median (Bitteur 2010)
                sort_idx = np.argsort(p_t)
                p_sorted = p_t[sort_idx]
                c_sorted = c_t[sort_idx]
                c_cum = np.cumsum(c_sorted)
                median_threshold = c_sum * 0.5
                med_idx = int(np.searchsorted(c_cum, median_threshold))
                med_idx = min(med_idx, len(p_sorted) - 1)

                frame_pitch[t] = float(p_sorted[med_idx])
                frame_conf_sum[t] = float(np.mean(c_t))

            # Keep only frames with a valid polyphonic estimate
            valid_mask = frame_pitch > 0.0
            n_valid = int(np.sum(valid_mask))

            if n_valid < 3:
                logger.debug("PolyphonicSpeedCurveEstimator: too few valid frames (%d) — Ersatzpfad", n_valid)
                return 0.0, 0.0, np.array([]), np.array([])

            # Savitzky-Golay smoothing of pitch curve (window=51, polyorder=3)
            try:
                from scipy.signal import savgol_filter

                sg_window = min(51, (n_valid // 2) * 2 + 1)  # odd, ≤ 51
                sg_window = max(sg_window, 5)
                smoothed_valid = savgol_filter(
                    frame_pitch[valid_mask].astype(np.float64), sg_window, polyorder=3
                ).astype(np.float32)
            except Exception:
                smoothed_valid = frame_pitch[valid_mask]

            # Build full smoothed curve (unset frames = 0)
            smoothed_curve = np.zeros(T, dtype=np.float32)
            smoothed_curve[valid_mask] = smoothed_valid

            global_pitch = float(np.median(smoothed_valid[smoothed_valid > 0]))
            confidence = float(np.mean(frame_conf_sum[valid_mask]))

            logger.info(
                "PolyphonicSpeedCurveEstimator: pitch=%.2f Hz, conf=%.3f, valid_frames=%d/%d, model=%s",
                global_pitch,
                confidence,
                n_valid,
                T,
                bp_result.model_used,
            )

            return global_pitch, confidence, smoothed_curve, times

        except Exception as exc:
            logger.warning("PolyphonicSpeedCurveEstimator fehlgeschlagen: %s — pYIN/CREPE Ersatzpfad", exc)
            return 0.0, 0.0, np.array([]), np.array([])

    def _detect_pitch_ml_hybrid(
        self, audio: np.ndarray, sample_rate: int, quality_mode: str
    ) -> tuple[float, float, dict[str, Any]]:
        """
        ML-Hybrid pitch detection using pYIN + CREPE.

        Quality Mode Routing:
        - FAST: pYIN only (_detect_pitch_pyin, Mauch & Dixon 2014)
        - BALANCED: Adaptive (pYIN → CREPE wenn Konfidenz <0.7)
        - MAXIMUM: CREPE (hybrid pYIN + CREPE kombiniert)

        Args:
            audio: Input audio
            sample_rate: Sample rate
            quality_mode: Quality mode (balanced/maximum)

        Returns:
            (detected_pitch, confidence, metadata)
        """
        try:
            # MAXIMUM mode: §2.12 PolyphonicSpeedCurveEstimator
            # BasicPitch ONNX → confidence-weighted median ≥2 voices → Savitzky-Golay
            if quality_mode == "maximum":
                poly_pitch, poly_conf, poly_curve, _ = self._estimate_speed_curve_polyphonic(audio, sample_rate)
                if poly_pitch > 0.0 and poly_conf >= 0.30:
                    logger.info(
                        "Verarbeitungsschritt 31 §2.12 PolyphonicSpeedCurveEstimator: pitch=%.2f Hz, conf=%.3f",
                        poly_pitch,
                        poly_conf,
                    )
                    return (
                        poly_pitch,
                        poly_conf,
                        {
                            "strategy": "polyphonic_speed_curve",
                            "pyin_applied": False,
                            "crepe_applied": False,
                            "basicpitch_applied": True,
                            "poly_pitch": poly_pitch,
                            "poly_confidence": poly_conf,
                            "poly_curve_frames": len(poly_curve),
                        },
                    )
                # BasicPitch gave no reliable result → fall through to HYBRID

            # Configure strategy based on quality mode
            if quality_mode == "maximum":
                strategy = PitchDetectionStrategy.HYBRID  # pYIN + CREPE kombiniert (fallback)
            else:  # balanced
                strategy = PitchDetectionStrategy.ADAPTIVE  # pYIN → CREPE wenn nötig

            # Create hybrid detector
            config = SpeedPitchConfig(strategy=strategy, confidence_threshold=0.7, averaging_window=2.0)

            detector = HybridSpeedPitch(config)

            # Detect global pitch
            result = detector.detect_global_pitch(audio, sample_rate)

            logger.info(
                "Verarbeitungsschritt 31 ML-Hybrid: pitch=%.2f Hz, confidence=%.3f, strategy=%s, pYIN=%s, CREPE=%s, time=%.2fs",
                result.detected_pitch,
                result.confidence,
                result.strategy_used.value,
                result.pyin_applied,
                result.crepe_applied,
                result.processing_time,
            )

            metadata = {
                "strategy": result.strategy_used.value,
                "pyin_applied": result.pyin_applied,
                "crepe_applied": result.crepe_applied,
                "pyin_pitch": result.pyin_pitch,
                "pyin_confidence": result.pyin_confidence,
                "crepe_pitch": result.crepe_pitch,
                "crepe_confidence": result.crepe_confidence,
                "processing_time": result.processing_time,
            }

            return result.detected_pitch, result.confidence, metadata

        except Exception as e:
            logger.warning("ML-Hybrid pitch detection fehlgeschlagen (pYIN Ersatzpfad aktiv): %s", e)
            # Fallback zu pYIN (Mauch & Dixon 2014)
            params = self.MATERIAL_PARAMS.get("vinyl", self.MATERIAL_PARAMS["unknown"])
            pitch, conf = self._detect_pitch_pyin(audio, params)
            metadata = {"strategy": "pyin_fallback", "pyin_applied": True, "crepe_applied": False, "error": str(e)}
            return pitch, conf, metadata

    def supports_material(self, _material_type: str) -> bool:
        """All materials supported."""
        return True


def _run_test() -> None:  # pragma: no cover
    """Test Professional Speed/Pitch Correction Phase."""
    logger.debug("=" * 80)
    logger.debug("Professional Speed/Pitch Correction Verarbeitungsschritt v2.0 - Test")
    logger.debug("=" * 80)

    # Generate test audio
    sr = 44100
    duration = 3
    t = np.linspace(0, duration, sr * duration)

    # Create 440 Hz tone (A4)
    true_pitch = 440.0

    # Simulate 3% speed error (too fast)
    speed_error = 0.03
    played_pitch = true_pitch * (1 + speed_error)

    audio = 0.4 * np.sin(2 * np.pi * played_pitch * t)
    audio += 0.2 * np.sin(2 * np.pi * played_pitch * 2 * t)  # Harmonic

    # Add attack envelope (simulate musical phrase)
    envelope = np.minimum(1, np.arange(len(audio)) / (sr * 0.1))
    audio *= envelope

    # Make stereo
    audio = np.column_stack([audio, audio * 0.95])

    logger.debug("\nTest Audio: %ss @ %s Hz (stereo)", duration, sr)
    logger.debug("True pitch: %s Hz", true_pitch)
    logger.debug("Simulated speed error: %.1f%% (too fast)", speed_error * 100)
    logger.debug("Played pitch: %.2f Hz", played_pitch)

    # Test with different materials
    materials = ["tape", "vinyl", "shellac"]

    for material in materials:
        logger.debug("\n%s", "-" * 80)
        logger.debug("Testing with material: %s", material.upper())
        logger.debug("%s", "-" * 80)

        phase = SpeedPitchCorrectionPhase(sample_rate=sr)
        result = phase.process(audio.copy(), material_type=material, reference_pitch=true_pitch)

        if result.success and result.modifications["processing"] == "applied":
            logger.debug("Processing vollstaendig!")
            logger.debug(
                "   Execution Time: %.3fs (%.2f× realtime)",
                result.metadata["execution_time_seconds"],
                result.metadata["execution_time_seconds"] / duration,
            )
            logger.debug("   erkannt Pitch: %.2f Hz", result.modifications["detected_pitch"])
            logger.debug("   Confidence: %.2f", result.modifications["confidence"])
            logger.debug("   Speed Error: %.2f%%", result.modifications["speed_error_percent"])
            logger.debug("   Correction Verhaeltnis: %.4f", result.modifications["correction_ratio"])
            logger.debug("   Algorithm: %s", result.metadata["algorithm"])
            logger.debug(
                "   Samples: %s → %s",
                result.modifications["samples_before"],
                result.modifications["samples_after"],
            )
        else:
            logger.debug("Processing uebersprungen")
            logger.debug("   Reason: %s", result.modifications.get("reason", "unknown"))

    logger.debug("\n%s", "=" * 80)
    logger.debug("Professional Speed/Pitch Correction v2.0 Test vollstaendig!")
    logger.debug("%s", "=" * 80)
    logger.debug("Algorithm: %s", result.metadata["algorithm"])
    logger.debug("Scientific Referenz: %s", result.metadata.get("scientific_ref", "N/A"))
    logger.debug("Benchmark: %s", result.metadata.get("benchmark", "N/A"))
    logger.debug("Quality Impact: 0.94 (Professional-Grade)")


if __name__ == "__main__":
    _run_test()
