"""
IAD-Detektion durch Pipeline-Gates (JND + Perceptual-Blend) ergänzt.
Aurik 10.0.0 — ArtifaktFreiheitsGate §2.49 [RELEASE_MUST]
======================================================
Dediziertes Gate für Artefakt-Erkennung — unabhängig von den 15 Musical Goals.
5 Artefakttypen mit material-adaptiven Schwellwerten und perzeptueller Salienz-Gewichtung.

Referenz: Spec 02 §2.49, Spec 02 §2.44 (HPI-Integration)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert

from backend.core.defect_scanner import MaterialType  # §v10.113
from backend.core.phase_ontology import (
    NOISE_TEXTURE_VALID_TYPES,
    PRE_ECHO_VALID_TYPES,
    PhaseOperationType,
    get_phase_type,
)
from backend.core.pipeline_health_state import make_fail_reason

logger = logging.getLogger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────────
_INSTANCE_HOLDER: dict[str, ArtifactFreedomGate | None] = {"gate": None}
_lock = threading.Lock()


def get_artifact_freedom_gate() -> ArtifactFreedomGate:
    """Thread-safe Singleton accessor (§ Pflicht-Pattern)."""
    if _INSTANCE_HOLDER["gate"] is None:
        with _lock:
            if _INSTANCE_HOLDER["gate"] is None:
                _INSTANCE_HOLDER["gate"] = ArtifactFreedomGate()
    gate = _INSTANCE_HOLDER["gate"]
    assert gate is not None
    return gate


# ── Data classes ───────────────────────────────────────────────────────────


@dataclass
class DetectedArtifact:
    """Single detected artifact event."""

    artifact_type: (
        str  # musical_noise | pre_echo | spectral_hole | phase_cancellation | metallic_ringing | crackle_impulse
    )
    start_sample: int
    end_sample: int
    severity_db: float  # raw severity in dB or correlation
    frequency_hz: float  # centre frequency of artifact (0.0 if broadband)
    context_rms_dbfs: float  # RMS of surrounding context segment
    salience_weighted_score: float = 0.0  # after freq * context * duration weighting
    temporal_masking_weight: float = 1.0  # §B4 temporal masking factor (1.0 = no masking)


@dataclass
class ArtifactFreedomResult:
    """Result of artifact freedom evaluation."""

    artifact_freedom: float  # ∈ [0.0, 1.0] — < 0.95 → veto
    detected_artifacts: list[DetectedArtifact] = field(default_factory=list)
    noise_texture_deviation_db_oct: float = 0.0  # spectral tilt deviation
    noise_texture_penalty: float = 0.0  # 0.0 or -0.05
    material_type: str = "digital"
    detail_report: dict = field(default_factory=dict)
    fail_reason: object | None = None  # §1.4a FailReason when artifact_freedom < 0.95
    # §2.49c Roughness/Sharpness psychoacoustic annoyance metrics
    roughness_delta_asper: float = 0.0  # Δrauheit (output - input) in asper
    sharpness_delta_acum: float = 0.0  # Δschärfe (output - input) in acum
    roughness_sharpness_penalty: float = 0.0  # ≤ 0.0, applied to artifact_freedom


@dataclass
class SourceMaterialBaseline:
    """§2.50 Pre-pipeline source material artifact characteristics.

    Measured once on the degraded input before any phase runs.
    Used by all subsequent gate evaluations to distinguish pre-existing
    carrier-chain artifacts from pipeline-introduced artifacts.
    """

    # Stereo field health
    phase_cancellation_ratio: float = 0.0  # fraction of 100ms frames already mono-incompatible
    stereo_mono_compat_mean: float = 1.0  # mean mono_compat across all frames in source
    stereo_lr_corr_mean: float = 1.0  # mean L/R Pearson correlation in source
    interchannel_lag_samples: int = 0  # signed L/R delay estimate from source material
    has_critical_stereo_issue: bool = False  # True when > 20 % frames mono-incompatible
    has_anti_phase_region: bool = False  # True when any frame has lr_corr < 0
    # Spectral health
    hf_loss_db: float = 0.0  # estimated HF loss vs. broadband reference (0 = none)
    material_type: str = "digital"


# ── Material-adaptive Thresholds (§2.49 Table) ────────────────────────────

# Base thresholds (CD/Digital)
_BASE_THRESHOLDS = {
    "musical_noise_peak_db": 12.0,
    "pre_echo_rel_attack_db": -40.0,
    "spectral_hole_hz": 200.0,
    "phase_cancellation_corr": 0.3,
    "metallic_ringing_peak_db": 6.0,
    # §2.49 Crackle-Impuls-Wiedereinfuehrung: kurtosis-Schwelle fuer Delta-Signal
    # und Peak-zu-RMS-Verhaeltnis. PGHI-Ringing / spektrale Subtraktions-Artefakte
    # liegen typisch bei kurtosis 15-40 und Peak/RMS > 18 dB.
    # (Godsill & Rayner 1998; Mazur & Blok 2011)
    "crackle_impulse_kurtosis": 10.0,  # basis: kurtosis im Delta-Signal (20ms-Fenster)
    "crackle_peak_rms_db": 18.0,  # basis: Peak/RMS-Verhaeltnis im Delta (dB)
}

# Tolerance factors per material (multiplied with base)
_MATERIAL_FACTORS = {
    "digital": {
        "musical_noise_peak_db": 1.0,
        "pre_echo_rel_attack_db": 1.0,
        "spectral_hole_hz": 1.0,
        "phase_cancellation_corr": 1.0,
        "metallic_ringing_peak_db": 1.0,
        "crackle_impulse_kurtosis": 1.0,  # strikte Toleranz fuer digitale Quellen
        "crackle_peak_rms_db": 1.0,
    },
    "cd": {
        "musical_noise_peak_db": 1.0,
        "pre_echo_rel_attack_db": 1.0,
        "spectral_hole_hz": 1.0,
        "phase_cancellation_corr": 1.0,
        "metallic_ringing_peak_db": 1.0,
        "crackle_impulse_kurtosis": 1.0,
        "crackle_peak_rms_db": 1.0,
    },
    "tape": {
        "musical_noise_peak_db": 1.25,
        "pre_echo_rel_attack_db": 0.875,
        "spectral_hole_hz": 1.5,
        "phase_cancellation_corr": 0.667,
        "metallic_ringing_peak_db": 1.333,
        "crackle_impulse_kurtosis": 1.2,  # Tape hat leichten Impuls-Rauschboden
        "crackle_peak_rms_db": 1.0,
    },
    "vinyl": {
        "musical_noise_peak_db": 1.5,
        "pre_echo_rel_attack_db": 0.75,
        "spectral_hole_hz": 2.0,
        "phase_cancellation_corr": 0.667,
        "metallic_ringing_peak_db": 1.667,
        "crackle_impulse_kurtosis": 1.4,  # Vinyl-Crackle-Rauschboden erhoehe Toleranz
        "crackle_peak_rms_db": 1.1,  # etwas hoeheres Peak/RMS noetig
    },
    "shellac": {
        "musical_noise_peak_db": 1.833,
        "pre_echo_rel_attack_db": 0.625,
        "spectral_hole_hz": 3.0,
        "phase_cancellation_corr": 0.5,
        "metallic_ringing_peak_db": 2.333,
        "crackle_impulse_kurtosis": 1.7,  # Shellac hat starken Crackle-Rauschboden
        "crackle_peak_rms_db": 1.2,
    },
    "wax": {
        "musical_noise_peak_db": 1.833,
        "pre_echo_rel_attack_db": 0.625,
        "spectral_hole_hz": 3.0,
        "phase_cancellation_corr": 0.5,
        "metallic_ringing_peak_db": 2.333,
        "crackle_impulse_kurtosis": 2.0,  # Wachswalze: sehr hoher Impuls-Rauschboden
        "crackle_peak_rms_db": 1.3,
    },
    # v10.0.0: cassette fehlte in _MATERIAL_FACTORS — _normalize_material() gab fälschlicherweise
    # "digital" zurück (kein substring-Match für "cassette" in den anderen Keys).
    # Kassette (IEC 60094-1 Type I): SNR ~52 dB, Flutter 0.2 % WRMS, BW 50–12.5 kHz.
    # Werte: zwischen tape (1.25×) und vinyl (1.5×) — mehr Hiss als Tape, weniger Crackle als Vinyl.
    "cassette": {
        "musical_noise_peak_db": 1.3,  # Kassettenrauschen etwas höher als Tape
        "pre_echo_rel_attack_db": 0.875,  # analog (wie tape)
        "spectral_hole_hz": 1.5,  # Dropout-Toleranz wie tape
        "phase_cancellation_corr": 0.667,  # Kassetten-Stereo analog zu Tape
        "metallic_ringing_peak_db": 1.333,  # wie tape
        "crackle_impulse_kurtosis": 1.2,  # leichter Impuls-Rauschboden
        "crackle_peak_rms_db": 1.0,
    },
}

# Artifact type weights (§2.49)
_TYPE_WEIGHTS = {
    "musical_noise": 1.0,
    "pre_echo": 0.8,
    "spectral_hole": 0.6,
    "phase_cancellation": 1.0,
    "metallic_ringing": 0.9,
    "crackle_impulse": 1.1,  # hoeher als musical_noise: Crackle sehr salient
}

# Max tolerance (normalisation denominator for artifact_freedom)
_MAX_TOLERANCE = 5.0
_MAX_TOLERANCE_BY_MATERIAL: dict[str, float] = {
    "digital": 5.0,
    "cd": 5.0,
    "cd_digital": 5.0,
    "streaming": 5.5,
    "mp3_high": 5.5,
    "mp3_low": 6.0,
    "aac": 5.8,
    "minidisc": 5.8,
    "tape": 6.2,
    "reel_tape": 6.2,
    "cassette": 6.4,
    "vinyl": 6.8,
    "shellac": 7.5,
    "wax": 7.8,
    "wax_cylinder": 7.8,
    "wire_recording": 7.8,
}

# §2.49c Roughness/Sharpness Guard thresholds
_ROUGHNESS_FLAG_ASPER: float = 0.15  # Δrauheit per phase in asper
_SHARPNESS_FLAG_ACUM: float = 0.30  # Δschärfe cumulative in acum
_ROUGHNESS_MATERIAL_TOLERANCE: dict[str, float] = {
    "digital": 1.0,
    "cd_digital": 1.0,
    "streaming": 1.0,
    "tape": 1.25,
    "reel_tape": 1.25,
    "cassette": 1.25,  # v10.0.0: analog wie tape
    "vinyl": 1.5,
    "minidisc": 1.5,
    "shellac": 2.0,
    "wax_cylinder": 2.0,
    "wire_recording": 2.0,
}
# Phase types for which Roughness/Sharpness guard applies (§2.49c §2.48a)
_ROUGHNESS_APPLICABLE_TYPES: frozenset = frozenset(
    {
        PhaseOperationType.DYNAMICS,
        PhaseOperationType.ADDITIVE,
        PhaseOperationType.ENHANCEMENT,
    }
)


def _as_membership_tuple(values: object) -> tuple[object, ...]:
    """Normalisiert membership sources so runtime checks survive scalar misconfiguration.

    Accepts scalars, iterables, or None and returns a tuple for safe membership checks.
    """
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        return (values,)
    if isinstance(values, Iterable):
        return tuple(values)
    return (values,)


def _safe_contains(values: object, candidate: object) -> bool:
    return candidate in _as_membership_tuple(values)


def _safe_startswith_any(prefixes: object, text: str) -> bool:
    if not text:
        return False
    return any(isinstance(prefix, str) and text.startswith(prefix) for prefix in _as_membership_tuple(prefixes))


class ArtifactFreedomGate:
    """§2.49 Artefakt-Freiheits-Gate — 5 artifact types, material-adaptive, salienz-weighted."""

    # Minimum silence segment length for noise texture analysis
    SILENCE_MIN_MS: float = 200.0
    SILENCE_THRESHOLD_DBFS: float = -50.0

    # STFT parameters
    N_FFT: int = 2048
    HOP_LENGTH: int = 512

    # §2.49 — Restorative / corrective phases.  These phases either SUBTRACT signal
    # components (noise, reverb, transients) or CORRECT frequency response (EQ, rumble,
    # azimuth) or ADJUST timing / levels.  Their per-phase residual is the
    # removed/corrected content, which by definition may look like spectral peaks
    # (removed noise structure), frequency drops (EQ roll-off), or pre-attack energy
    # changes (timing correction, level normalisation).  All of these mimic artefact
    # signatures numerically but are musically correct and intentional.
    # For these phases, the only check that remains active is phase-cancellation and
    # metallic-ringing (the artefact types that genuine processing errors introduce
    # even in corrective phases).  Pre-echo, musical-noise, and spectral-hole
    # detectors are all DISABLED for restorative phases.
    _RESTORATIVE_PHASE_IDS: frozenset[str] = frozenset(
        {
            # Noise / artefact removal
            "phase_02",
            "phase_03",
            "phase_05",
            "phase_09",
            "phase_18",
            "phase_20",
            "phase_23",
            "phase_24",
            "phase_29",
            "phase_49",
            # EQ / frequency correction (intentional spectrum changes)
            "phase_04",
            "phase_06",
            # Timing / tape-path correction (residual = corrected drift)
            "phase_12",
            "phase_25",
            "phase_30",
            "phase_31",
            # Click / pop / DC removal
            "phase_01",
            "phase_27",
            # Loudness / format / dither (intentional level shifts and LSB-noise — no real artefacts)
            "phase_40",
            "phase_41",
            "phase_17",
            # Additive harmonic synthesis: no transient delay structure; pre-echo check
            # is semantically undefined for sine-based overtone generators.
            "phase_07",
            # Multiband compression with gain makeup: uniform gain increase is applied to
            # all frames including pre-attack periods, which inflates pre_energy
            # numerically beyond the 1.5× threshold even when no temporal smearing occurs.
            # Pre-echo is causally impossible in a strictly feed-forward compressor.
            "phase_35",
        }
    )

    # Crackle detector exclusion for dedicated impulse-removal phases.
    # Their corrective delta is dominated by removed impulses and tends to
    # look high-kurtosis by construction, which can cause false positives.
    _CRACKLE_IMPULSE_EXCLUDE_PREFIXES: tuple[str, ...] = (
        "phase_01",
        "phase_09",
        "phase_27",
    )

    def evaluate(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        material_type: str = "digital",
        phase_id: str = "",
        goal_weights: dict[str, float] | None = None,
        restorability_score: float | None = None,
        transfer_chain_depth: int = 1,
    ) -> ArtifactFreedomResult:
        """Bewertet artifact freedom of restored audio vs original.

        Args:
            original: input audio (float32, [-1, 1])
            restored: processed audio (float32, [-1, 1])
            sr: sample rate
            material_type: carrier type for adaptive thresholds
            phase_id: phase identifier — used to select detector set (restorative vs enhancement)
            goal_weights: optional §2.56 per-song goal-weights for adaptive annoyance guard
            restorability_score: optional restorability context (0-100) for adaptive tolerance
            transfer_chain_depth: chain depth for depth-adaptive artifact tolerance

        Returns:
            ArtifactFreedomResult with artifact_freedom score and details
        """
        original = np.nan_to_num(np.asarray(original, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.nan_to_num(np.asarray(restored, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        orig_mono = original if original.ndim == 1 else np.mean(original, axis=0)
        rest_mono = restored if restored.ndim == 1 else np.mean(restored, axis=0)
        min_len = min(len(orig_mono), len(rest_mono))
        if min_len < sr // 10:  # < 100 ms — too short
            return ArtifactFreedomResult(artifact_freedom=1.0, material_type=material_type)

        orig_mono = orig_mono[:min_len]
        rest_mono = rest_mono[:min_len]

        mat_key = self._normalize_material(material_type)
        thresholds = self._get_thresholds(mat_key)

        # §v10.121 Depth-adaptive artifact detection: deeper transfer chains accumulate
        # more temporal smear (tape flutter, vinyl pre-ringing, multi-generation phase
        # rotation). This carrier-inherent smear triggers false pre-echo detections.
        # Scale the pre-echo threshold proportionally to chain depth so only genuine
        # restoration-induced pre-echo is flagged.
        _depth_pe = int(max(1, transfer_chain_depth))
        if _depth_pe >= 4:
            _depth_pe_factor = 1.0 + (_depth_pe - 3) * 0.3  # depth 4→1.3×, depth 5→1.6×
        elif _depth_pe == 3:
            _depth_pe_factor = 1.15
        elif _depth_pe == 2:
            _depth_pe_factor = 1.08
        else:
            _depth_pe_factor = 1.0
        thresholds["pre_echo_rel_attack_db"] *= _depth_pe_factor

        # §2.48a Architektur-Inversion: Guards feuern nur wenn ihre Voraussetzung
        # strukturell erfüllt ist — abgeleitet aus dem intrinsischen Phase-Typ
        # (phase_ontology.py), nicht aus Ausnahmelisten.
        #
        # Kompatibilität: _RESTORATIVE_PHASE_IDS bleibt als Legacy-Fallback für Phasen,
        # die noch nicht im Ontologie-Register eingetragen sind.
        _phase_type = get_phase_type(phase_id) if phase_id else PhaseOperationType.ENHANCEMENT
        _is_restorative = (
            _phase_type in (PhaseOperationType.SUBTRACTIVE, PhaseOperationType.CORRECTIVE)
            or _safe_contains(self._RESTORATIVE_PHASE_IDS, phase_id)
            or _safe_startswith_any(self._RESTORATIVE_PHASE_IDS, phase_id)
        )
        _per_phase_mode = bool(phase_id)

        all_artifacts: list[DetectedArtifact] = []

        # 1. Musical Noise — disabled in per-phase mode (residual = processing delta,
        #    not silence-domain noise artefact)
        if not _per_phase_mode:
            all_artifacts.extend(self._detect_musical_noise(orig_mono, rest_mono, sr, thresholds))

        # 2. Pre-Echo — §2.48a: Nur valide für Typen in PRE_ECHO_VALID_TYPES.
        #    SUBTRACTIVE: Residual = entferntes Rauschen ≠ Prä-Transient-Energie.
        #    CORRECTIVE: Timing-Korrektur ändert Prä-Attack-Energie intentional.
        #    ML_GENERATIVE: Diffusionsausgang hat keine MDCT-Quantisierungsstruktur.
        #    ADDITIVE: Harmonik-Synthese erzeugt keine kausale Prä-Echo-Struktur.
        #    (Richter et al. 2022; Brandenburg & Johnston 1994)
        _pre_echo_valid = _safe_contains(PRE_ECHO_VALID_TYPES, _phase_type)
        _level_scale = 1.0
        if _pre_echo_valid:
            if _per_phase_mode:
                _rms_orig = float(np.sqrt(np.mean(orig_mono**2) + 1e-12))
                _rms_rest = float(np.sqrt(np.mean(rest_mono**2) + 1e-12))
                if _rms_orig > 1e-10:
                    _level_scale = _rms_rest / _rms_orig
            all_artifacts.extend(self._detect_pre_echo(orig_mono, rest_mono, sr, thresholds, _level_scale))

        # 3. Spectral Holes — skip für restorative/corrective Phasen (intentionaler Rauschboden-Abfall)
        if not _is_restorative:
            all_artifacts.extend(self._detect_spectral_holes(orig_mono, rest_mono, sr, thresholds))

        # 4. Phase-Cancellation (stereo only) — always run.
        # Pass original in all modes so the detector can:
        #   a) detect stereo collapse (channel disappeared > 40 dB drop) at any eval stage
        #   b) in per-phase mode: skip frames already broken before this phase ran
        # Without original: detector falls back to absolute corr/mono-compat check.
        _orig_stereo_for_pc = original if (original.ndim == 2 and original.shape[0] >= 2) else None
        if restored.ndim == 2 and restored.shape[0] >= 2:
            # §2.49 [RELEASE_MUST] Phase-Typ-adaptiver PC-Delta-Threshold (§2.54).
            # SUBTRACTIVE (Dereverb, Denoising): STFT-M/S-Verarbeitung verändert L/R-Korrelation
            #   strukturell — 0.10 erzeugt systematisch False Positives auf allen Dereverb-Phasen.
            #   Schwellwert aus Physik: Dereverb-Gain < 6 dB Kanalasymmetrie ist inaudibel (ITU-R BS.775).
            # CORRECTIVE/ML_GENERATIVE: moderate Toleranz — intentionale Signalform-Änderungen.
            # ADDITIVE/ENHANCEMENT/DYNAMICS: enge Toleranz — parallele Kanalverarbeitung erwartet.
            # Invariante: Schwellwerte aus Phase-Typ-Semantik abgeleitet, NICHT song-spezifisch.
            _PC_DELTA_BY_TYPE: dict = {
                PhaseOperationType.SUBTRACTIVE: 0.22,  # Dereverb, Denoising — freq.-domain L/R shift
                PhaseOperationType.CORRECTIVE: 0.18,  # Phase-Korrektur, EQ — gezielte Änderungen
                PhaseOperationType.ML_GENERATIVE: 0.20,  # Neuronale Verarbeitung — L/R unvorhersehbar
                PhaseOperationType.ADDITIVE: 0.12,  # Harmonik-Synthese — sollte L/R-parallel sein
                PhaseOperationType.ENHANCEMENT: 0.12,  # Loudness, EQ — parallele Kanalverarbeitung
                PhaseOperationType.DYNAMICS: 0.12,  # Kompression — Gain-Kopplung erwartet
            }
            _pc_delta_threshold = _PC_DELTA_BY_TYPE.get(_phase_type, 0.12)
            all_artifacts.extend(
                self._detect_phase_cancellation(restored, sr, thresholds, _orig_stereo_for_pc, _pc_delta_threshold)
            )

        # 5. Metallic Ringing — residual-based (persistent peaks in restored-original).
        #    In per-phase mode the residual = processing delta (added harmonics, EQ boosts, etc.)
        #    which trivially produces "peaks persisting > 50 ms" = real musical content, not artefacts.
        #    Disable in per-phase mode; only run on final pipeline output evaluation.
        if not _per_phase_mode:
            all_artifacts.extend(self._detect_metallic_ringing(orig_mono, rest_mono, sr, thresholds))

        # 6. Crackle-Impuls-Wiedereinfuehrung — per-Phase, SUBTRACTIVE/CORRECTIVE/ML_GENERATIVE.
        # STFT/spektrale Phasen (phase_23 STFT-Diskontinuitaeten, phase_29 PGHI-Ringing,
        # phase_14 Bandgrenzen-FIR-Transienten, phase_20 OMLSA-Ripple) koennen Impuls-
        # Artefakte einfuehren, obwohl ihr Zweck restorativ ist. Dieses Gate ist
        # NICHT durch _is_restorative blockiert — Crackle-Wiedereinfuehrung durch
        # eine spektrale Phase ist ein genuines Artefakt.
        # Kurtosis des Delta-Signals: Crackle kurtosis >> 10, normales Spektrum ~3-6.
        _crackle_check_types = (
            PhaseOperationType.SUBTRACTIVE,
            PhaseOperationType.CORRECTIVE,
            PhaseOperationType.ML_GENERATIVE,
        )
        if (
            _per_phase_mode
            and _phase_type in _crackle_check_types
            and not _safe_startswith_any(self._CRACKLE_IMPULSE_EXCLUDE_PREFIXES, phase_id)
        ):
            # §0d Phase-type-adaptive crackle thresholds: CORRECTIVE phases
            # (click-pop-removal, dropout-repair) remove impulses by design — the delta
            # signal IS the removed impulse content, which trivially exceeds kurtosis=10.
            # The Dropout-Zone-Guard and Click-Removal-Guard handle the main false-positive
            # cases in _detect_crackle_impulse. For boundary interpolation residuals that
            # pass those guards, raise thresholds so only genuinely extreme re-introduced
            # spikes (kurtosis > 15 = Godsill & Rayner 1998: real crackle regime) are flagged.
            if _phase_type == PhaseOperationType.CORRECTIVE:
                _crackle_thresholds = dict(thresholds)  # shallow copy — do not mutate shared dict
                _crackle_thresholds["crackle_impulse_kurtosis"] = (
                    thresholds.get("crackle_impulse_kurtosis", 10.0) * 1.5
                )  # 15.0
                _crackle_thresholds["crackle_peak_rms_db"] = (
                    thresholds.get("crackle_peak_rms_db", 18.0) + 6.0
                )  # 24.0 dB
            else:
                _crackle_thresholds = thresholds
            all_artifacts.extend(self._detect_crackle_impulse(orig_mono, rest_mono, sr, _crackle_thresholds))

        # Apply salience weighting
        for art in all_artifacts:
            art.salience_weighted_score = self._compute_salience_weight(art, sr)

        # Noise texture coherence — §2.48a: Nur valide für SUBTRACTIVE Phasen.
        # (Schwarz & Grill 2004: BW-Erweiterung ändert Spektral-Tilt intentional.)
        # ADDITIVE, CORRECTIVE, ML_GENERATIVE: Tilt-Änderung ist kein Artefakt.
        _noise_texture_valid = _safe_contains(NOISE_TEXTURE_VALID_TYPES, _phase_type)
        if _noise_texture_valid:
            noise_dev, noise_penalty = self._check_noise_texture(orig_mono, rest_mono, sr)
        else:
            noise_dev, noise_penalty = 0.0, 0.0

        # §2.49c Roughness/Sharpness Guard — DYNAMICS, ADDITIVE, ENHANCEMENT only
        _roughness_delta = 0.0
        _sharpness_delta = 0.0
        _rs_penalty = 0.0
        if _safe_contains(_ROUGHNESS_APPLICABLE_TYPES, _phase_type):
            try:
                rough_orig = self._compute_roughness_zwicker(orig_mono, sr)
                rough_rest = self._compute_roughness_zwicker(rest_mono, sr)
                sharp_orig = self._compute_sharpness_bismarck(orig_mono, sr)
                sharp_rest = self._compute_sharpness_bismarck(rest_mono, sr)
                mat_tol = _ROUGHNESS_MATERIAL_TOLERANCE.get(mat_key, 1.0)
                # §2.56 harmonization: adaptive tolerance based on song priorities.
                # Preserve-heavy songs (nat/auth/timbre important) => stricter annoyance guard.
                # Clarity-heavy songs (trans/art/brillanz important) => slightly more tolerance.
                _preserve_w = 1.0
                _clarity_w = 1.0
                if isinstance(goal_weights, dict) and goal_weights:
                    _preserve_w = float(
                        np.clip(
                            (
                                float(goal_weights.get("natuerlichkeit", 1.0))
                                + float(goal_weights.get("authentizitaet", 1.0))
                                + float(goal_weights.get("timbre_authentizitaet", 1.0))
                            )
                            / 3.0,
                            0.30,
                            2.00,
                        )
                    )
                    _clarity_w = float(
                        np.clip(
                            (
                                float(goal_weights.get("transparenz", 1.0))
                                + float(goal_weights.get("artikulation", 1.0))
                                + float(goal_weights.get("brillanz", 1.0))
                            )
                            / 3.0,
                            0.30,
                            2.00,
                        )
                    )

                _rest = 65.0 if restorability_score is None else float(np.clip(restorability_score, 0.0, 100.0))
                # Harder material (low restorability) gets slightly more tolerance.
                _rest_tol = float(np.clip(1.0 + (55.0 - _rest) / 250.0, 0.85, 1.20))
                # If preserve-weight is high, tolerance should shrink; if clarity-weight is high, expand.
                _goal_tol = float(np.clip(np.sqrt(_clarity_w / max(_preserve_w, 1e-6)), 0.80, 1.25))
                _adaptive_tol = mat_tol * _rest_tol * _goal_tol

                _roughness_delta = max(0.0, rough_rest - rough_orig)
                _sharpness_delta = max(0.0, sharp_rest - sharp_orig)
                if _roughness_delta > _ROUGHNESS_FLAG_ASPER * _adaptive_tol:
                    _rs_penalty -= 0.05
                if _sharpness_delta > _SHARPNESS_FLAG_ACUM * _adaptive_tol:
                    _rs_penalty -= 0.10
            except Exception as _ex:
                logger.debug("roughness/sharpness guard fehlgeschlagen: %s", _ex)

        # Score calculation — §B4 temporal_masking_weight is kept as separate field so
        # that _compute_salience_weight() does not silently overwrite it (bug fix).
        weighted_sum = sum(
            _TYPE_WEIGHTS.get(a.artifact_type, 1.0) * a.salience_weighted_score * a.temporal_masking_weight
            for a in all_artifacts
        )
        # §2.54 material-adaptive cumulative tolerance: older/noisier carriers contain
        # more benign residual micro-artifacts after restorative phases. A single global
        # denominator over-penalizes shellac/wax/tape vs digital material.
        _max_tolerance = float(_MAX_TOLERANCE_BY_MATERIAL.get(mat_key, _MAX_TOLERANCE))

        # §0d/§2.54 Carrier-Recovery-Baseline-Capping (analog to §2.29c for PMGG):
        # Restorative phases (denoise, dropout-repair, click-removal, dereverb) work on
        # defect-contaminated audio. The noise/defect floor itself is impulsive and
        # phase-shifted → artifact detectors fire on carrier-noise content that is
        # EXPECTED to change, not on newly introduced artefacts.
        # Scale max_tolerance proportionally to degradation: heavily degraded (rest≈0)
        # gets ×1.8, near-pristine (rest≈100) gets ×1.0. Applies only to restorative
        # phases so enhancement phases retain their full guard sensitivity.
        if _is_restorative and restorability_score is not None:
            _rest_clamped = float(np.clip(restorability_score, 0.0, 100.0))
            _restorative_tol_factor = 1.0 + max(0.0, (80.0 - _rest_clamped) / 80.0) * 0.8
            _max_tolerance *= _restorative_tol_factor

        # §v10.121 Depth-adaptive max_tolerance: deeper chains have inherently more
        # carrier-born temporal anomalies (flutter, multi-gen smear, phase rotation)
        # that artifact detectors mistake for restoration damage. Scale tolerance so
        # scoring reflects genuine new artifacts only.
        _depth_tol_factor = float(np.clip(1.0 + (_depth_pe - 1) * 0.22, 1.0, 2.50))
        _max_tolerance *= _depth_tol_factor

        artifact_freedom = float(np.clip(1.0 - (weighted_sum / _max_tolerance), 0.0, 1.0))
        artifact_freedom = artifact_freedom + noise_penalty  # penalty is negative or 0
        artifact_freedom = artifact_freedom + _rs_penalty  # §2.49c roughness/sharpness penalty
        artifact_freedom = float(np.clip(artifact_freedom, 0.0, 1.0))

        # Per-phase rollback should be driven by concrete artifact detections.
        # Soft penalties (noise texture / roughness-sharpness) are useful for
        # ranking and telemetry, but can over-penalize isolated phases.
        if _per_phase_mode and not all_artifacts and artifact_freedom < 0.95:
            artifact_freedom = 0.95

        detail: dict[str, object] = {}
        detail["n_musical_noise"] = sum(1 for a in all_artifacts if a.artifact_type == "musical_noise")
        detail["n_pre_echo"] = sum(1 for a in all_artifacts if a.artifact_type == "pre_echo")
        detail["n_spectral_holes"] = sum(1 for a in all_artifacts if a.artifact_type == "spectral_hole")
        detail["n_phase_cancellation"] = sum(1 for a in all_artifacts if a.artifact_type == "phase_cancellation")
        detail["n_metallic_ringing"] = sum(1 for a in all_artifacts if a.artifact_type == "metallic_ringing")
        detail["n_crackle_impulse"] = sum(1 for a in all_artifacts if a.artifact_type == "crackle_impulse")
        detail["weighted_artifact_sum"] = round(weighted_sum, 4)
        detail["noise_texture_deviation_db_oct"] = round(noise_dev, 3)
        # §2.49c: populate roughness/sharpness detail when guard fired
        if _rs_penalty < 0.0:
            if _roughness_delta > _ROUGHNESS_FLAG_ASPER:
                detail["roughness_flag"] = {"delta_asper": round(_roughness_delta, 4)}
            if _sharpness_delta > _SHARPNESS_FLAG_ACUM:
                detail["sharpness_flag"] = {"delta_acum": round(_sharpness_delta, 4)}
        if _per_phase_mode and not all_artifacts and (noise_penalty < 0.0 or _rs_penalty < 0.0):
            detail["soft_penalty_only"] = True

        # §v10.17: Material-abhängiger Veto-Schwellwert.
        # Kassette/Vinyl/Schellack DÜRFEN Transport-Bumps und Knistern haben.
        _MATERIAL_VETO = {
            "cd_digital": 0.95,
            "digital": 0.95,
            "mp3_high": 0.90,
            "aac": 0.90,
            "mp3_low": 0.85,
            "vinyl": 0.80,
            "cassette": 0.75,
            "tape": 0.80,
            "reel_tape": 0.80,
            "shellac": 0.65,
            "wax_cylinder": 0.60,
        }
        _veto_base = float(_MATERIAL_VETO.get(mat_key, 0.90))

        # §G100 Adaptive Veto-Margin: Statt hartem Binary-Gate wird der
        # Veto-Threshold aus Restorability-Score und Transfer-Chain-Depth
        # kontinuierlich moduliert. Tief degradiertes Material (niedrige RS,
        # tiefe Chain) bekommt eine niedrigere Schwelle — es wäre unfair,
        # eine 4.-Gen-Kassette mit denselben Artefakt-Limits zu messen wie
        # eine pristine CD.
        # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 70.0).
        from backend.core.calibration_context import resolve_restorability_score

        _rs_veto = float(
            resolve_restorability_score(
                restorability_score if isinstance(restorability_score, (int, float)) else None
            )
        )
        _depth_veto = max(1, int(transfer_chain_depth if isinstance(transfer_chain_depth, (int, float)) else 1))
        # Restorability-Faktor: rs 0→0.82, rs 50→0.91, rs 100→1.00
        _rs_factor = float(np.clip(0.82 + (_rs_veto / 100.0) * 0.18, 0.80, 1.0))
        # Depth-Faktor: depth 1→1.00, depth 4→0.91, depth 6→0.85
        _depth_factor = float(np.clip(1.0 - max(0, _depth_veto - 1) * 0.03, 0.82, 1.0))
        _veto = float(np.clip(_veto_base * _rs_factor * _depth_factor, 0.35, 0.97))

        # §G79 (GEBOTE.md) Calibration-Audit: dokumentiere adaptiven Veto-Threshold
        logger.info(
            "§CALIB af_veto: rs=%.0f depth=%d mat=%s → veto=%.4f (base=%.2f)",
            _rs_veto,
            _depth_veto,
            mat_key,
            _veto,
            _veto_base,
        )

        _afg_fr = None
        if artifact_freedom < _veto:
            _afg_fr = make_fail_reason(
                "ArtifactFreedomGate",
                "ARTIFACT_VETO",
                severity="failed",
                action="rollback",
                details=(
                    f"artifact_freedom={artifact_freedom:.4f} < {_veto:.3f} "
                    f"(base={_veto_base:.2f} rs={_rs_veto:.0f} depth={_depth_veto} "
                    f"material={mat_key}), {len(all_artifacts)} artifacts detected"
                ),
            )

        return ArtifactFreedomResult(
            artifact_freedom=round(artifact_freedom, 4),
            detected_artifacts=all_artifacts,
            noise_texture_deviation_db_oct=noise_dev,
            noise_texture_penalty=noise_penalty,
            material_type=mat_key,
            detail_report=detail,
            fail_reason=_afg_fr,
            roughness_delta_asper=round(_roughness_delta, 4),
            sharpness_delta_acum=round(_sharpness_delta, 4),
            roughness_sharpness_penalty=round(_rs_penalty, 4),
        )

    # ── §B4 Temporal Masking Helper ────────────────────────────────────────

    @staticmethod
    def _compute_temporal_masking_weights(
        audio: np.ndarray,
        sr: int,
        artifacts: list[DetectedArtifact],
    ) -> list[float]:
        """Psychoacoustic temporal masking weight per artifact (B4, ISO 532-1 §3.1).

        Forward masking: loud transient masks artefacts up to 200 ms AFTER it.
        Backward masking: 5 ms BEFORE a loud transient, artefacts are weakly masked.

        Returns a weight ∈ (0, 1] per artifact — multiply artifact salience_weighted_score
        by this weight before accumulating the penalty.

        Design:
          - Forward mask decay: w = 0.15 + 0.85 * (lag_ms / 200)           for lag ≤ 200 ms
          - Backward mask:      w = 0.10                                     for lag ≤ 5 ms
          - Outside any masking window: w = 1.0 (full penalty)
        """
        if not artifacts:
            return []

        # Detect loud transients: 10 ms frames, peaks ≥ -20 dBFS relative to signal peak
        frame_len = max(1, int(0.010 * sr))
        hop = max(1, frame_len // 2)
        n_frames = max(1, (len(audio) - frame_len) // hop)

        frame_rms: list[float] = []
        for i in range(n_frames):
            s = i * hop
            seg = audio[s : s + frame_len]
            frame_rms.append(float(np.sqrt(np.mean(seg**2) + 1e-12)))

        if not frame_rms:
            return [1.0] * len(artifacts)

        peak_rms = float(np.percentile(frame_rms, 99))
        threshold_rms = peak_rms * 10.0 ** (-20.0 / 20.0)  # -20 dBFS relative to peak

        transient_positions_s: list[float] = []
        for i, rms_val in enumerate(frame_rms):
            if rms_val >= threshold_rms:
                transient_positions_s.append(float(i * hop) / sr)

        if not transient_positions_s:
            return [1.0] * len(artifacts)

        fwd_mask_s = 0.200  # 200 ms forward masking window
        bwd_mask_s = 0.005  # 5 ms backward masking window

        weights: list[float] = []
        for art in artifacts:
            art_center_s = float(art.start_sample + (art.end_sample - art.start_sample) / 2.0) / sr
            best_w = 1.0  # default: no masking

            for t_s in transient_positions_s:
                lag = art_center_s - t_s  # positive = after transient

                if 0.0 <= lag <= fwd_mask_s:
                    # Forward masking: linearly decays from 0.15 (at transient) to 1.0 (at 200 ms)
                    w = 0.15 + 0.85 * (lag / fwd_mask_s)
                    best_w = min(best_w, w)
                elif -bwd_mask_s <= lag < 0.0:
                    # Backward masking: strong suppression (artefact before loud event)
                    best_w = min(best_w, 0.10)

            weights.append(best_w)

        return weights

    # ── Detector: Musical Noise ────────────────────────────────────────────

    def _detect_musical_noise(
        self,
        orig: np.ndarray,
        restored: np.ndarray,
        sr: int,
        thresholds: dict,
    ) -> list[DetectedArtifact]:
        """Erkennt isolated tonal peaks in silence segments of the residual."""
        artifacts: list[DetectedArtifact] = []
        residual = restored - orig
        threshold_db = thresholds["musical_noise_peak_db"]

        # Find silence segments in the restored audio
        frame_len = int(0.03 * sr)  # 30 ms frames
        hop = frame_len // 2
        n_frames = max(1, (len(restored) - frame_len) // hop)

        for i in range(n_frames):
            start = i * hop
            end = start + frame_len
            frame = restored[start:end]
            rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
            rms_db = 20.0 * np.log10(rms + 1e-12)

            if rms_db > -40.0:  # not a silence segment
                continue

            # Check residual spectrum for isolated tonal peaks
            res_frame = residual[start:end]
            if len(res_frame) < 64:
                continue

            # §2.49 Residual minimum-energy guard: skip frames whose residual is
            # at or below the quantisation-noise floor (~−100 dBFS).  Minimal-
            # change phases (DC-offset removal, surface-noise profiling) produce
            # residuals ≈ −100 dBFS; their spectra contain random peaks that
            # statistically exceed the musical-noise threshold, causing thousands
            # of false positives.  Real DSP artefacts are ≥ −60 dBFS.
            _res_rms = float(np.sqrt(np.mean(res_frame**2) + 1e-12))
            _res_rms_db = 20.0 * np.log10(_res_rms + 1e-12)
            if _res_rms_db < -70.0:
                continue  # quantisation-noise only — no genuine artefact

            win = np.hanning(len(res_frame))
            spectrum = np.abs(np.fft.rfft(res_frame * win))
            mag_db = 20.0 * np.log10(spectrum + 1e-12)

            # §2.49 Directional spectra: musical noise = ADDED energy, not removed.
            # Subtractive phases (surface-noise profiling, denoise) produce a residual
            # whose spectrum mirrors the removed noise — residual peaks are CORRECT
            # removals, not introduced artefacts.  Only flag bins where restored
            # energy > original energy (additive action, not subtractive).
            rest_spectrum = np.abs(np.fft.rfft(restored[start:end] * win))
            orig_spectrum = np.abs(np.fft.rfft(orig[start:end] * win))

            # §2.49c ERB-weighted simultaneous masking threshold (ISO 11172-3):
            # Spectral peaks that are below the psychoacoustic masking threshold
            # of the surrounding signal are inaudible and must not be flagged.
            # Masking model: spreading function spans ±1.5 ERB from each bin;
            # masking threshold ≈ signal_level − (14.5 + bark_distance × 7.5) dB.
            _rest_mag_db = 20.0 * np.log10(rest_spectrum + 1e-12)
            _n_bins = len(rest_spectrum)
            _freq_axis = np.arange(_n_bins) * sr / (2.0 * _n_bins)
            # ERB approximation: ERB(f) ≈ 24.7 * (4.37 * f/1000 + 1) (Glasberg & Moore 1990)
            _erb_widths = 24.7 * (4.37 * _freq_axis / 1000.0 + 1.0)
            _masking_threshold = np.full(_n_bins, -120.0, dtype=np.float64)
            # Simplified spreading: for each bin, masking from ±2 ERB neighbours
            for _mb in range(_n_bins):
                if _rest_mag_db[_mb] < -60.0:
                    continue
                _erb_w = max(1.0, _erb_widths[_mb])
                _spread_hz = 2.0 * _erb_w
                _spread_bins = max(1, int(_spread_hz / max(1.0, sr / (2.0 * _n_bins))))
                _lo = max(0, _mb - _spread_bins)
                _hi = min(_n_bins, _mb + _spread_bins + 1)
                _mask_level = _rest_mag_db[_mb] - 14.5  # simultaneous masking offset
                _masking_threshold[_lo:_hi] = np.maximum(_masking_threshold[_lo:_hi], _mask_level)

            # Median neighbor comparison: peak must exceed neighbors by threshold
            for j in range(2, len(mag_db) - 2):
                neighbors = np.median(mag_db[max(0, j - 5) : j + 6])
                excess = mag_db[j] - neighbors
                if excess > threshold_db:
                    # Directional guard: skip if restored energy ≤ original energy
                    # at this bin — the phase removed a peak (correct), not added one.
                    if rest_spectrum[j] <= orig_spectrum[j] * 1.05:
                        continue
                    # ERB masking guard: skip if the artifact peak is below
                    # the psychoacoustic masking threshold — it is inaudible.
                    if j < _n_bins and mag_db[j] < _masking_threshold[j]:
                        continue
                    freq_hz = float(j * sr / (2 * len(spectrum)))
                    artifacts.append(
                        DetectedArtifact(
                            artifact_type="musical_noise",
                            start_sample=start,
                            end_sample=end,
                            severity_db=float(excess),
                            frequency_hz=freq_hz,
                            context_rms_dbfs=rms_db,
                        )
                    )
                    break  # one per frame

        # §B4: Apply temporal masking — artefacts near loud transients are less perceptible.
        # Store as temporal_masking_weight (separate field) so it is not overwritten by
        # the outer _compute_salience_weight() loop in evaluate().
        if artifacts:
            try:
                _rest_mono_mn = restored if restored.ndim == 1 else np.mean(restored, axis=0)
                mask_weights = self._compute_temporal_masking_weights(_rest_mono_mn, sr, artifacts)
                for art, mw in zip(artifacts, mask_weights):
                    art.temporal_masking_weight = float(mw)
            except Exception as _tm_exc:
                logger.debug("§B4 temporal-masking (musical_noise) nicht blockierend: %s", _tm_exc)

        return artifacts

    # ── Detector: Pre-Echo ─────────────────────────────────────────────────

    def _detect_pre_echo(
        self,
        orig: np.ndarray,
        restored: np.ndarray,
        sr: int,
        thresholds: dict,
        level_scale: float = 1.0,
    ) -> list[DetectedArtifact]:
        """Erkennt energy before transient attacks that wasn't in the original.

        level_scale: global RMS ratio (restored_rms / orig_rms).  Used in
            per-phase mode to avoid treating uniform gain changes (loudness
            normalisation, mild compression) as spurious pre-echo.  The
            orig_pre_energy reference is multiplied by this factor before the
            1.5× newness test, so level-only changes no longer fire the detector.
        """
        artifacts: list[DetectedArtifact] = []
        threshold_db = thresholds["pre_echo_rel_attack_db"]
        pre_window_samples = int(0.005 * sr)  # 5 ms window before attack

        # Find transients (onsets) in original
        frame_len = int(0.01 * sr)  # 10 ms
        hop = frame_len // 2
        n_frames = max(1, (len(orig) - frame_len) // hop)
        energies = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            s = i * hop
            e = s + frame_len
            energies[i] = float(np.mean(orig[s:e] ** 2) + 1e-12)

        # Detect sharp onsets (energy ratio > 10 dB jump)
        for i in range(2, n_frames):
            if energies[i] < 1e-10:
                continue
            ratio_db = 10.0 * np.log10(energies[i] / (energies[i - 1] + 1e-12))
            if ratio_db < 10.0:
                continue

            # Found a transient — check pre-echo in restored
            attack_sample = i * hop
            pre_start = max(0, attack_sample - pre_window_samples)
            if pre_start >= attack_sample:
                continue

            attack_peak = float(np.max(np.abs(restored[attack_sample : attack_sample + frame_len])) + 1e-12)
            pre_energy = float(np.max(np.abs(restored[pre_start:attack_sample])) + 1e-12)
            orig_pre_energy = float(np.max(np.abs(orig[pre_start:attack_sample])) + 1e-12)

            # Level-normalised newness test: compensate for uniform gain changes
            # (loudness normalisation, compression) that shift ALL frames by level_scale.
            # Real pre-echo = NEW temporal energy not explained by the level change.
            adjusted_orig_pre = orig_pre_energy * level_scale
            # Absolute-floor guard: don't flag if the absolute new energy is below
            # −60 dBFS of the attack peak.  Harmonic synthesis and spectral repair add
            # a small continuous signal (natural harmonic tail), which inflates
            # pre_energy when orig_pre_energy was near-zero (Tape silence ~-70 dBFS).
            # The NEW energy must be:
            #   (a) > 1.5× the level-adjusted original pre energy  AND
            #   (b) > 0.001 × attack_peak (i.e., not below −60 dB relative to attack)
            # Without condition (b), near-zero orig_pre_energy causes false positives
            # when even tiny restoration additions (< −60 dBFS) numerically exceed 1.5×.
            _new_energy_min = attack_peak * 0.001  # −60 dBFS relative to attack
            if pre_energy <= adjusted_orig_pre * 1.5 or pre_energy < _new_energy_min:
                continue  # no significant new pre-echo beyond level adjustment

            rel_db = 20.0 * np.log10(pre_energy / attack_peak)
            if rel_db > threshold_db:  # threshold is negative, e.g. -40 dB
                rms = float(np.sqrt(np.mean(restored[pre_start:attack_sample] ** 2) + 1e-12))
                rms_db = 20.0 * np.log10(rms + 1e-12)
                artifacts.append(
                    DetectedArtifact(
                        artifact_type="pre_echo",
                        start_sample=pre_start,
                        end_sample=attack_sample,
                        severity_db=float(rel_db),
                        frequency_hz=0.0,  # broadband
                        context_rms_dbfs=rms_db,
                    )
                )

        return artifacts

    # ── Detector: Spectral Holes ───────────────────────────────────────────

    def _detect_spectral_holes(
        self,
        orig: np.ndarray,
        restored: np.ndarray,
        sr: int,
        thresholds: dict,
    ) -> list[DetectedArtifact]:
        """Erkennt frequency gaps in the restored passband that weren't in original."""
        artifacts: list[DetectedArtifact] = []
        hole_threshold_hz = thresholds["spectral_hole_hz"]

        # Compute average spectrum of both
        n_fft = min(self.N_FFT, len(orig))
        if n_fft < 256:
            return artifacts

        # Use multiple frames for robust estimate
        hop = n_fft // 2
        n_frames = max(1, (len(orig) - n_fft) // hop)
        n_frames = min(n_frames, 50)  # cap for performance

        orig_mag_acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        rest_mag_acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        win = np.hanning(n_fft).astype(np.float32)

        for i in range(n_frames):
            s = i * hop
            e = s + n_fft
            if e > len(orig):
                break
            orig_mag_acc += np.abs(np.fft.rfft(orig[s:e] * win))
            rest_mag_acc += np.abs(np.fft.rfft(restored[s:e] * win))

        orig_mag_acc /= max(n_frames, 1)
        rest_mag_acc /= max(n_frames, 1)

        orig_db = 20.0 * np.log10(orig_mag_acc + 1e-12)
        rest_db = 20.0 * np.log10(rest_mag_acc + 1e-12)

        # Passband: bins where original has energy above its own noise floor + 6 dB.
        # Use a generous threshold so flat-spectrum signals (white noise, broadband music)
        # are not excluded — the 10th-percentile noise floor + 10 dB is too strict for
        # flat signals where the entire spectrum IS the signal.
        noise_floor = np.percentile(orig_db, 10)
        passband_floor = min(noise_floor + 6.0, np.percentile(orig_db, 25))
        passband_mask = orig_db > passband_floor

        # Check for holes: bins where restored dropped > 15 dB below original while
        # original had content there.  15 dB threshold catches real spectral holes
        # without false-triggering on natural HF roll-off after restoration.
        drop_db = orig_db - rest_db
        hole_mask = passband_mask & (drop_db > 15.0)

        # §2.49c ERB-weighted masking: spectral holes that are below the
        # simultaneous masking threshold of the *restored* signal are
        # inaudible — don't flag them.  Same spreading model as musical noise.
        _n_bins_sh = len(rest_db)
        _freq_axis_sh = np.arange(_n_bins_sh) * sr / float(n_fft)
        _erb_w_sh = 24.7 * (4.37 * _freq_axis_sh / 1000.0 + 1.0)
        _mask_th_sh = np.full(_n_bins_sh, -120.0, dtype=np.float64)
        for _mb_sh in range(_n_bins_sh):
            if rest_db[_mb_sh] < -60.0:
                continue
            _spread_hz_sh = 2.0 * max(1.0, _erb_w_sh[_mb_sh])
            _spread_bins_sh = max(1, int(_spread_hz_sh / max(1.0, float(sr) / n_fft)))
            _lo_sh = max(0, _mb_sh - _spread_bins_sh)
            _hi_sh = min(_n_bins_sh, _mb_sh + _spread_bins_sh + 1)
            _ml_sh = rest_db[_mb_sh] - 14.5
            _mask_th_sh[_lo_sh:_hi_sh] = np.maximum(_mask_th_sh[_lo_sh:_hi_sh], _ml_sh)
        # Holes where the dropped energy was below masking threshold are masked
        _audible_hole = np.copy(hole_mask)
        for _hb in np.where(hole_mask)[0]:
            if rest_db[_hb] > _mask_th_sh[_hb]:
                pass  # audible — keep in hole_mask
            else:
                _audible_hole[_hb] = False
        hole_mask = _audible_hole

        if not np.any(hole_mask):
            return artifacts

        # Find contiguous hole regions
        freq_res = sr / n_fft
        hole_bins = np.where(hole_mask)[0]
        if len(hole_bins) == 0:
            return artifacts

        # Group consecutive bins
        groups = np.split(hole_bins, np.where(np.diff(hole_bins) > 1)[0] + 1)
        for group in groups:
            if len(group) == 0:
                continue
            hole_width_hz = len(group) * freq_res
            if hole_width_hz >= hole_threshold_hz:
                centre_bin = group[len(group) // 2]
                centre_hz = float(centre_bin * freq_res)
                avg_drop = float(np.mean(drop_db[group]))
                artifacts.append(
                    DetectedArtifact(
                        artifact_type="spectral_hole",
                        start_sample=0,
                        end_sample=len(restored),
                        severity_db=avg_drop,
                        frequency_hz=centre_hz,
                        context_rms_dbfs=-30.0,  # global artefact
                    )
                )

        return artifacts

    # ── Detector: Phase-Cancellation ───────────────────────────────────────

    @staticmethod
    def _lr_corr_and_compat(
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[float, float]:
        """Gibt (lr_corr, mono_compat) for a pair of channel frames zurück."""
        l_rms = float(np.sqrt(np.mean(left**2) + 1e-12))
        r_rms = float(np.sqrt(np.mean(right**2) + 1e-12))

        l_norm = left - np.mean(left)
        r_norm = right - np.mean(right)
        denom = float(np.sqrt(np.sum(l_norm**2) * np.sum(r_norm**2)) + 1e-12)
        lr_corr = float(np.sum(l_norm * r_norm) / denom) if denom > 1e-12 else 0.0

        mid_rms = float(np.sqrt(np.mean(((left + right) / 2.0) ** 2) + 1e-12))
        lr_energy = float(np.sqrt(l_rms**2 + r_rms**2) + 1e-12)
        mono_compat = mid_rms / lr_energy
        return lr_corr, mono_compat

    def _detect_phase_cancellation(
        self,
        restored_stereo: np.ndarray,
        sr: int,
        thresholds: dict,
        original_stereo: np.ndarray | None = None,
        delta_threshold: float = 0.12,
    ) -> list[DetectedArtifact]:
        """Erkennt mono-incompatible phase cancellation in stereo output.

        In per-phase mode (``original_stereo`` provided) only frames where THIS
        phase *introduced or worsened* phase cancellation are counted.  Frames
        that were already mono-incompatible in the source material are ignored so
        that pre-existing cassette / vinyl stereo problems do not trigger the gate
        on every single phase evaluation.
        """
        artifacts: list[DetectedArtifact] = []
        corr_threshold = thresholds["phase_cancellation_corr"]
        # Delta guard: only flag if this phase degraded mono-compat beyond the
        # phase-type-adaptive threshold (passed by caller from §2.54 logic).
        # SUBTRACTIVE (dereverb): 0.22 — STFT M/S processing shifts L/R correlation structurally.
        # CORRECTIVE: 0.18 | ML_GENERATIVE: 0.20 | ADDITIVE/ENHANCEMENT/DYNAMICS: 0.12 (default).
        _DELTA_THRESHOLD = float(delta_threshold)

        left = restored_stereo[0]
        right = restored_stereo[1] if restored_stereo.shape[0] >= 2 else left

        orig_left: np.ndarray | None = None
        orig_right: np.ndarray | None = None
        if original_stereo is not None and original_stereo.ndim == 2 and original_stereo.shape[0] >= 2:
            orig_left = original_stereo[0]
            orig_right = original_stereo[1]

        # ── Stereo-Collapse Guard ─────────────────────────────────────────
        # If original had a meaningful channel but restored lost it entirely
        # (> 40 dB RMS drop), flag as one high-severity artifact and return early.
        # The frame-loop below cannot catch this: a silent R-channel has l_rms+r_rms
        # dominated by L, so rms-silence-filter skips and mono-compat = 0.5 passes.
        if orig_left is not None and orig_right is not None:
            _orig_l_rms = float(np.sqrt(np.mean(orig_left**2) + 1e-12))
            _orig_r_rms = float(np.sqrt(np.mean(orig_right**2) + 1e-12))
            _rest_l_rms = float(np.sqrt(np.mean(left**2) + 1e-12))
            _rest_r_rms = float(np.sqrt(np.mean(right**2) + 1e-12))
            _r_collapsed = _orig_r_rms > 1e-4 and _rest_r_rms < _orig_r_rms * 0.01
            _l_collapsed = _orig_l_rms > 1e-4 and _rest_l_rms < _orig_l_rms * 0.01
            # Absolute stereo imbalance guard: catches cumulative drift where each single-phase
            # drop is < 40 dB but total collapse is catastrophic (Bug B fix).
            # Triggers if: output has >20 dB L/R imbalance AND input was balanced (< 6 dB).
            if not (_r_collapsed or _l_collapsed) and _orig_l_rms > 1e-4 and _orig_r_rms > 1e-4:
                _abs_out_imb = abs(20.0 * np.log10((_rest_l_rms + 1e-12) / (_rest_r_rms + 1e-12)))
                _abs_in_imb = abs(20.0 * np.log10((_orig_l_rms + 1e-12) / (_orig_r_rms + 1e-12)))
                if _abs_out_imb > 20.0 and _abs_in_imb < 6.0:
                    _weaker_out = _rest_r_rms if _rest_l_rms >= _rest_r_rms else _rest_l_rms
                    _weaker_orig = _orig_r_rms if _rest_l_rms >= _rest_r_rms else _orig_l_rms
                    artifacts.append(
                        DetectedArtifact(
                            artifact_type="phase_cancellation",
                            start_sample=0,
                            end_sample=len(left),
                            severity_db=float(20.0 * np.log10(_weaker_out / (_weaker_orig + 1e-12) + 1e-12)),
                            frequency_hz=0.0,
                            context_rms_dbfs=20.0 * np.log10(max(_orig_l_rms, _orig_r_rms) + 1e-12),
                        )
                    )
                    return artifacts  # cumulative stereo collapse — skip frame loop
            if _r_collapsed or _l_collapsed:
                _lost = _rest_r_rms if _r_collapsed else _rest_l_rms
                _orig = _orig_r_rms if _r_collapsed else _orig_l_rms
                artifacts.append(
                    DetectedArtifact(
                        artifact_type="phase_cancellation",
                        start_sample=0,
                        end_sample=len(left),
                        severity_db=float(20.0 * np.log10(_lost / (_orig + 1e-12) + 1e-12)),
                        frequency_hz=0.0,
                        context_rms_dbfs=20.0 * np.log10(max(_orig_l_rms, _orig_r_rms) + 1e-12),
                    )
                )
                return artifacts  # whole-channel loss dominates — skip frame loop

        # ── Frame-level L/R cross-correlation analysis ────────────────────
        # Phase cancellation = L and R strongly anti-correlated (r < 0) OR
        # mid-signal energy significantly lower than the individual channel energies
        # would imply (relative mono compatibility loss).
        frame_len = int(0.1 * sr)  # 100 ms frames
        hop = frame_len // 2
        n_frames = max(1, (len(left) - frame_len) // hop)

        for i in range(n_frames):
            s = i * hop
            e = s + frame_len
            l_frame = left[s:e]
            r_frame = right[s:e]

            l_rms = float(np.sqrt(np.mean(l_frame**2) + 1e-12))
            r_rms = float(np.sqrt(np.mean(r_frame**2) + 1e-12))

            if l_rms < 1e-6 and r_rms < 1e-6:
                continue  # silence

            lr_corr, mono_compat = self._lr_corr_and_compat(l_frame, r_frame)

            # Anti-correlation threshold: lr_corr must be < -0.20 to be considered
            # genuinely anti-phase. Values between 0 and -0.20 arise from normal
            # per-phase processing differences (gate attack/release transients, STFT
            # window misalignment, slight L/R processing asymmetry) and are NOT
            # audibly problematic — strongly anti-correlated signals cause the
            # "hollow center" / mono-incompatibility that this check targets.
            is_anti_corr = lr_corr < -0.20
            is_mono_incompat = mono_compat < corr_threshold

            if not (is_anti_corr or is_mono_incompat):
                continue  # no cancellation in restored output

            # Per-phase delta check: only count if THIS phase worsened the stereo field.
            # If original was already anti-correlated / mono-incompatible, skip (source material).
            if orig_left is not None and orig_right is not None:
                _oe = min(e, len(orig_left))
                _orig_l = orig_left[s:_oe]
                _orig_r = orig_right[s:_oe]
                if len(_orig_l) < 2 or len(_orig_r) < 2:
                    pass  # too short — use absolute check
                else:
                    _orig_corr, _orig_compat = self._lr_corr_and_compat(_orig_l, _orig_r)
                    _orig_had_cancellation = (_orig_corr < 0.0) or (_orig_compat < corr_threshold)
                    if _orig_had_cancellation:
                        # Pre-existing stereo problem — skip this frame
                        continue
                    # Original was OK; only flag if this phase degraded it noticeably
                    if mono_compat >= (_orig_compat - _DELTA_THRESHOLD):
                        continue  # phase did not make it worse beyond threshold

                    # Near-mono input guard: quasi-mono sources (orig_compat > 0.65, e.g. mono
                    # tape recording) can produce minor processing-induced L/R asymmetry after
                    # independent-channel phases (noise gate transients, dropout fill, harmonic
                    # synthesis). If output is still moderately mono-compatible (> 0.40), the
                    # degradation is inaudible on the original near-mono material — skip it.
                    # Output below 0.40 is still caught as a genuine stereo collapse.
                    if _orig_compat > 0.65 and mono_compat > 0.40:
                        continue  # near-mono source; processing artifact is inaudible

                    logger.debug(
                        "_erkennen_Verarbeitungsschritt_cancellation frame=%d orig_compat=%.3f "
                        "mono_compat=%.3f is_anti=%s delta=%.3f FLAGGED",
                        i,
                        _orig_compat,
                        mono_compat,
                        str(is_anti_corr),
                        _orig_compat - mono_compat,
                    )

            rms_db = 20.0 * np.log10((l_rms + r_rms) / 2.0 + 1e-12)
            artifacts.append(
                DetectedArtifact(
                    artifact_type="phase_cancellation",
                    start_sample=s,
                    end_sample=e,
                    severity_db=float(mono_compat),  # mono compatibility ratio
                    frequency_hz=0.0,
                    context_rms_dbfs=rms_db,
                )
            )

        return artifacts

    # ── Detector: Metallic Ringing ─────────────────────────────────────────

    def _detect_metallic_ringing(
        self,
        orig: np.ndarray,
        restored: np.ndarray,
        sr: int,
        thresholds: dict,
    ) -> list[DetectedArtifact]:
        """Erkennt resonant peaks in CQT that persist > 50 ms."""
        artifacts: list[DetectedArtifact] = []
        threshold_db = thresholds["metallic_ringing_peak_db"]
        min_duration_samples = int(0.05 * sr)  # 50 ms

        residual = restored - orig
        frame_len = int(0.02 * sr)  # 20 ms frames
        hop = frame_len // 2
        n_frames = max(1, (len(residual) - frame_len) // hop)
        n_fft = max(256, frame_len)

        # Track persistent spectral peaks across frames
        peak_tracker: dict[int, int] = {}  # bin -> consecutive frames count
        peak_severity: dict[int, float] = {}  # bin -> max severity

        win = np.hanning(frame_len).astype(np.float32)

        for i in range(n_frames):
            s = i * hop
            e = s + frame_len
            frame = residual[s:e]
            if len(frame) < frame_len:
                break

            padded = np.zeros(n_fft, dtype=np.float32)
            padded[:frame_len] = frame * win
            spectrum = np.abs(np.fft.rfft(padded))
            mag_db = 20.0 * np.log10(spectrum + 1e-12)

            # Find peaks exceeding neighbors by threshold
            frame_peaks: set[int] = set()
            for j in range(3, len(mag_db) - 3):
                local_med = float(np.median(mag_db[max(0, j - 5) : j + 6]))
                excess = mag_db[j] - local_med
                if excess > threshold_db:
                    frame_peaks.add(j)
                    peak_severity[j] = max(peak_severity.get(j, 0.0), float(excess))

            # Update tracker
            new_tracker: dict[int, int] = {}
            for b in frame_peaks:
                new_tracker[b] = peak_tracker.get(b, 0) + 1
            peak_tracker = new_tracker

            # Check for persistent peaks (> 50 ms)
            min_frames = max(1, min_duration_samples // hop)
            for b, count in peak_tracker.items():
                if count >= min_frames:
                    freq_hz = float(b * sr / n_fft)
                    rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
                    rms_db = 20.0 * np.log10(rms + 1e-12)
                    artifacts.append(
                        DetectedArtifact(
                            artifact_type="metallic_ringing",
                            start_sample=max(0, s - count * hop),
                            end_sample=e,
                            severity_db=peak_severity.get(b, float(threshold_db)),
                            frequency_hz=freq_hz,
                            context_rms_dbfs=rms_db,
                        )
                    )
                    # Reset to avoid duplicate reports
                    peak_tracker[b] = 0

        return artifacts

    # ── Crackle Impulse Re-introduction Detector (§2.49 #6) ───────────────

    def _detect_crackle_impulse(
        self,
        orig: np.ndarray,
        restored: np.ndarray,
        sr: int,
        thresholds: dict,
    ) -> list[DetectedArtifact]:
        """Erkennt impulse artifacts (crackle-like) introduced by this phase.

        STFT/spectral phases can introduce impulse artifacts even when their
        primary purpose is restorative:
        - phase_29 PGHI reconstruction: ringing at phase boundaries
        - phase_23 spectral repair: STFT-frame discontinuities
        - phase_14 band recombination: FIR startup transients at crossovers
        - phase_20 OMLSA: gain ripple transients in spectral bins

        Method: kurtosis of the delta signal (restored - orig) in 20ms windows.
        Crackle/impulse artifacts have kurtosis >> 10; normal spectral processing
        produces delta kurtosis ~3–6.

        Delta-based (directional guard): only fires when restored has larger peak
        amplitude than original — new spike was added, not merely removed content.

        Scientific basis:
          Godsill & Rayner 1998: "Digital Audio Restoration" — impulse noise
            identification via AR-residual excess kurtosis (Cap. 3).
          Mazur & Blok 2011: kurtosis as impulsiveness indicator in audio.
        """
        artifacts: list[DetectedArtifact] = []

        kurtosis_threshold = thresholds.get("crackle_impulse_kurtosis", 10.0)
        peak_rms_threshold = thresholds.get("crackle_peak_rms_db", 18.0)

        delta = restored - orig  # processing delta — new impulses appear here

        win_samples = max(32, int(0.020 * sr))  # 20 ms
        hop_samples = max(16, win_samples // 2)
        if len(delta) < win_samples:
            return artifacts
        n_windows = (len(delta) - win_samples) // hop_samples

        for i in range(n_windows):
            start = i * hop_samples
            end = start + win_samples
            delta_win = delta[start:end]
            rest_win = restored[start:end]
            orig_win = orig[start:end]

            # Skip if delta is near-zero (phase changed nothing here)
            delta_rms = float(np.sqrt(np.mean(delta_win**2) + 1e-12))
            if delta_rms < 1e-8:
                continue

            # Kurtosis of normalized delta — high kurtosis = impulsive content
            delta_norm = delta_win / (delta_rms + 1e-12)
            kurtosis = float(np.mean(delta_norm**4))
            if kurtosis <= kurtosis_threshold:
                continue

            # Peak-to-RMS ratio of delta (crackle: very high ratio)
            peak_delta = float(np.max(np.abs(delta_win)))
            peak_rms_db = 20.0 * np.log10((peak_delta + 1e-12) / (delta_rms + 1e-12))
            if peak_rms_db <= peak_rms_threshold:
                continue

            # Dropout/silence-zone guard: if original frame is near-silent, the
            # restored frame contains interpolated gap-fill content (dropout repair)
            # or click-suppressed silence.  In either case the delta is the correct
            # repair output, not a new impulse artifact.  Applies to CORRECTIVE phases
            # (dropout_repair, click_pop_removal) as well as general restorative phases.
            rms_orig_win = float(np.sqrt(np.mean(orig_win**2) + 1e-12))
            rms_orig_dbfs_w = 20.0 * np.log10(rms_orig_win + 1e-12)
            if rms_orig_dbfs_w < -25.0:
                continue  # dropout gap or near-silence: interpolated content is not an artifact

            # Directional guard: peak in restored must be larger than in orig
            # (phase added a spike, not just removed something)
            peak_rest = float(np.max(np.abs(rest_win)))
            peak_orig = float(np.max(np.abs(orig_win)))

            # Click-removal guard: if orig had a significantly larger peak than restored,
            # the phase REMOVED an impulse (the click/pop).  The delta is the removed click —
            # by definition impulsive and high-kurtosis.  This is correct restoration, not
            # an introduced artifact.  Threshold 1.30 allows for interpolation boundary effects
            # without letting genuine click re-introduction through.
            if peak_orig > peak_rest * 1.30:
                continue  # phase removed a large impulse (click-repair): delta is correct

            if peak_rest <= peak_orig * 1.15:  # 15 % tolerance — new spike must be meaningfully larger
                continue

            rms_rest = float(np.sqrt(np.mean(rest_win**2) + 1e-12))
            rms_dbfs = 20.0 * np.log10(rms_rest + 1e-12)

            artifacts.append(
                DetectedArtifact(
                    artifact_type="crackle_impulse",
                    start_sample=start,
                    end_sample=end,
                    severity_db=float(peak_rms_db),
                    frequency_hz=0.0,  # broadband impulse
                    context_rms_dbfs=rms_dbfs,
                )
            )

        if artifacts:
            logger.debug(
                "AFG crackle_impulse: %d windows flagged (kurtosis_thr=%.1f, peak_rms_thr=%.1f dB)",
                len(artifacts),
                kurtosis_threshold,
                peak_rms_threshold,
            )
        return artifacts

    # ── Noise Texture Coherence (§2.49) ────────────────────────────────────

    def _check_noise_texture(
        self,
        orig: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> tuple[float, float]:
        """Compare spectral tilt of noise floor in silence segments.

        Returns:
            (tilt_deviation_db_oct, penalty): deviation and score penalty (0 or -0.05)
        """
        int(self.SILENCE_MIN_MS / 1000.0 * sr)

        # Find silence segments in original
        frame_len = int(0.03 * sr)
        hop = frame_len // 2
        n_frames = max(1, (len(orig) - frame_len) // hop)

        silence_frames_orig: list[int] = []

        for i in range(n_frames):
            s = i * hop
            e = s + frame_len
            rms = float(np.sqrt(np.mean(orig[s:e] ** 2) + 1e-12))
            rms_db = 20.0 * np.log10(rms + 1e-12)
            if rms_db < self.SILENCE_THRESHOLD_DBFS:
                silence_frames_orig.append(i)

        if len(silence_frames_orig) < 3:
            return 0.0, 0.0  # not enough silence to measure

        # Compute spectral tilt in silence segments for both
        orig_tilt = self._compute_spectral_tilt(orig, sr, silence_frames_orig, frame_len, hop)
        rest_tilt = self._compute_spectral_tilt(restored, sr, silence_frames_orig, frame_len, hop)

        if orig_tilt is None or rest_tilt is None:
            return 0.0, 0.0

        deviation = abs(rest_tilt - orig_tilt)

        if deviation <= 3.0:
            return deviation, 0.0  # OK
        elif deviation <= 6.0:
            return deviation, -0.05  # warning penalty
        else:
            return deviation, -0.05  # rollback signaled (caller handles)

    def _compute_spectral_tilt(
        self,
        audio: np.ndarray,
        sr: int,
        silence_frame_indices: list[int],
        frame_len: int,
        hop: int,
    ) -> float | None:
        """Berechnet average spectral tilt (dB/octave) in silence frames."""
        n_fft = max(256, frame_len)
        win = np.hanning(frame_len).astype(np.float32)
        acc_mag = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        count = 0

        for idx in silence_frame_indices:
            s = idx * hop
            e = s + frame_len
            if e > len(audio):
                break
            frame = audio[s:e]
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[:frame_len] = frame * win
            acc_mag += np.abs(np.fft.rfft(padded))
            count += 1

        if count == 0:
            return None

        avg_mag = acc_mag / count
        mag_db = 20.0 * np.log10(avg_mag + 1e-12)

        # Linear regression of mag_db vs log2(frequency)
        freqs = np.arange(1, len(mag_db)) * sr / n_fft
        freqs = freqs[freqs > 20.0]  # skip DC and sub-bass
        if len(freqs) < 10:
            return None

        log2_freq = np.log2(freqs + 1e-12)
        mag_slice = mag_db[1 : len(freqs) + 1]

        # Linear regression: tilt in dB per octave
        A = np.vstack([log2_freq, np.ones(len(log2_freq))]).T
        try:
            result = np.linalg.lstsq(A, mag_slice, rcond=None)
            tilt = float(result[0][0])  # dB/octave
        except Exception:
            tilt = 0.0

        return tilt

    # ── §2.49c Roughness/Sharpness (Zwicker/Bismarck) ─────────────────────

    def _compute_roughness_zwicker(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt roughness in asper (Zwicker 1991 approximation).

        Uses temporal envelope modulation energy in 15–300 Hz range.
        Reference: 1 asper ≈ 1 kHz tone, 60 dB SPL, 100 % AM at 70 Hz.
        """
        if len(audio) < int(0.1 * sr):
            return 0.0

        try:
            # Temporal envelope via Hilbert transform
            audio_arr = np.asarray(audio, dtype=np.float64)
            analytic = np.asarray(hilbert(audio_arr), dtype=np.complex128)
            envelope = np.asarray(np.abs(analytic), dtype=np.float64)

            # Normalise envelope to remove DC (absolute level)
            envelope -= np.mean(envelope)

            # FFT of envelope to get modulation spectrum
            env_fft = np.abs(np.fft.rfft(envelope))
            env_freqs = np.fft.rfftfreq(len(envelope), d=1.0 / sr)

            # Integrate energy in 15–300 Hz band (roughness band)
            mask = (env_freqs >= 15.0) & (env_freqs <= 300.0)
            if not np.any(mask):
                return 0.0

            am_energy = float(np.sum(env_fft[mask] ** 2)) / max(len(audio), 1)

            # Calibrate: empirical scaling → ~1.0 asper for typical steady-state 70 Hz AM
            # am_energy_ref at 60 dB SPL, f_AM=70 Hz, 100% AM, sr=48000 ≈ 1.5e-3
            _ROUGHNESS_CALIB = 1.5e-3
            roughness = float(am_energy / (_ROUGHNESS_CALIB + 1e-12))
            return max(0.0, min(roughness, 10.0))  # cap at 10 asper
        except Exception as e:
            logger.warning("artifact_freedom_gate.py::_berechnen_roughness_zwicker Ersatzpfad: %s", e)
            return 0.0

    def _compute_sharpness_bismarck(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt sharpness in acum (Bismarck 1974 / DIN 45692 approximation).

        Uses Bark-scale spectral centroid with g(z) psychoacoustic weighting.
        Reference: 1 acum = 1 kHz narrow-band noise at 60 dB SPL.
        """
        if len(audio) < int(0.05 * sr):
            return 0.0

        try:
            # Bark-scale centre frequencies for 24 critical bands
            bark_centers_hz = np.array(
                [
                    50,
                    150,
                    250,
                    350,
                    450,
                    570,
                    700,
                    840,
                    1000,
                    1170,
                    1370,
                    1600,
                    1850,
                    2150,
                    2500,
                    2900,
                    3400,
                    4000,
                    4800,
                    5800,
                    7000,
                    8500,
                    10500,
                    13500,
                ],
                dtype=np.float64,
            )
            # Corresponding Bark values (Traunmüller 1990)
            bark_values = np.array(
                [
                    0.5,
                    1.5,
                    2.5,
                    3.5,
                    4.5,
                    5.5,
                    6.5,
                    7.5,
                    8.5,
                    9.5,
                    10.5,
                    11.5,
                    12.5,
                    13.5,
                    14.5,
                    15.5,
                    16.5,
                    17.5,
                    18.5,
                    19.5,
                    20.5,
                    21.5,
                    22.5,
                    23.5,
                ],
                dtype=np.float64,
            )

            # Bismarck weighting function
            def _g(z: float) -> float:
                return 1.0 if z <= 16.0 else 0.066 * np.exp(0.171 * z)

            # Compute band energies via FFT
            n_fft = min(4096, len(audio))
            win = np.hanning(n_fft).astype(np.float32)
            mag = np.abs(np.fft.rfft(audio[:n_fft] * win))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

            # Specific loudness proxy: power per Bark band (unnormalized)
            N_prime = np.zeros(len(bark_centers_hz))
            for i, (f_c, bw_hz) in enumerate(zip(bark_centers_hz, np.diff(np.append(bark_centers_hz, 16000.0)))):
                mask = (freqs >= f_c - bw_hz / 2) & (freqs < f_c + bw_hz / 2)
                N_prime[i] = float(np.sum(mag[mask] ** 2))

            total_N = float(np.sum(N_prime))
            if total_N < 1e-12:
                return 0.0

            g_weights = np.array([_g(z) for z in bark_values])
            weighted_sum = float(np.sum(N_prime * g_weights * bark_values))

            # Bismarck formula: 0.11 × ∫ N'(z)·g(z)·z dz / ∫ N'(z) dz
            sharpness = 0.11 * weighted_sum / total_N

            return max(0.0, min(sharpness, 10.0))  # cap at 10 acum
        except Exception as e:
            logger.warning("artifact_freedom_gate.py::_g Ersatzpfad: %s", e)
            return 0.0

    # ── Salienz-Gewichtung (§2.49) ─────────────────────────────────────────

    def _compute_salience_weight(self, artifact: DetectedArtifact, sr: int) -> float:
        """Berechnet perceptual salience weight for an artifact."""
        # Frequency factor
        freq = artifact.frequency_hz
        if 200.0 <= freq <= 5000.0:
            freq_factor = 1.0
        elif freq > 12000.0:
            freq_factor = 0.2
        elif freq > 5000.0 or freq < 200.0:
            freq_factor = 0.5
        else:
            freq_factor = 1.0  # broadband (freq=0)

        if freq == 0.0:
            freq_factor = 0.8  # broadband — moderate salience

        # Context factor (silence = more audible)
        rms_db = artifact.context_rms_dbfs
        if rms_db < -40.0:
            context_factor = 1.5  # silence — very audible
        elif rms_db > -20.0:
            context_factor = 0.5  # loud passage — masked
        else:
            context_factor = 1.0

        # Duration factor
        duration_samples = artifact.end_sample - artifact.start_sample
        duration_ms = 1000.0 * duration_samples / sr if sr > 0 else 0.0
        if duration_ms > 100.0:
            duration_factor = 1.5
        elif duration_ms < 20.0:
            duration_factor = 0.5
        else:
            duration_factor = 1.0

        # §2.49 Spec: salience_weighted_score must be ∈ [0, 1] — clamp to prevent
        # wsum > n_artifacts (false-positive rollback cascade)
        return float(np.clip(freq_factor * context_factor * duration_factor, 0.0, 1.0))

    # ── Threshold helpers ──────────────────────────────────────────────────

    def _get_thresholds(self, material: str) -> dict:
        """Gibt zurück: material-adaptive thresholds."""
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        factors = _MATERIAL_FACTORS.get(_mk, _MATERIAL_FACTORS["digital"])
        thresholds = {}
        for key, base_val in _BASE_THRESHOLDS.items():
            factor = factors.get(key, 1.0)
            if key == "pre_echo_rel_attack_db":
                # For negative threshold: multiply factor relaxes it (less negative)
                thresholds[key] = base_val * factor
            else:
                thresholds[key] = base_val * factor
        return thresholds

    @staticmethod
    def _normalize_material(material_type: str) -> str:
        """Map MaterialType string to normalized key."""
        mt = str(material_type).lower().strip()
        # Exact-prefix / substring check — spezifischere Begriffe zuerst prüfen.
        # "cassette" enthält keinen der anderen Keys (tape/vinyl/shellac/wax),
        # daher muss es vor dem generischen Schleifendurchlauf explizit abgefangen werden.
        if "cassette" in mt:
            return "cassette"
        for key in _MATERIAL_FACTORS:
            if key in mt:
                return key
        if "cd" in mt or "wav" in mt or "mp3" in mt or "aac" in mt or "flac" in mt or "ogg" in mt:
            return "cd"
        return "digital"

    @staticmethod
    def _estimate_interchannel_lag_samples(audio: np.ndarray, sr: int, max_seconds: float = 5.0) -> int:
        """Schätzt signed L/R lag using GCC-PHAT on a bounded source window."""
        try:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim != 2:
                return 0
            if arr.shape[0] >= 2:
                left = arr[0]
                right = arr[1]
            elif arr.shape[1] >= 2:
                left = arr[:, 0]
                right = arr[:, 1]
            else:
                return 0

            n = min(len(left), len(right), int(float(sr) * max_seconds))
            if n < max(1024, sr // 10):
                return 0

            x = left[:n].astype(np.float64)
            y = right[:n].astype(np.float64)
            if max(float(np.sqrt(np.mean(x * x))), float(np.sqrt(np.mean(y * y)))) < 1e-6:
                return 0
            n_fft = 1
            while n_fft < 2 * n:
                n_fft <<= 1

            X = np.fft.rfft(x, n=n_fft)
            Y = np.fft.rfft(y, n=n_fft)
            cross = X * np.conj(Y)
            gcc = np.fft.irfft(cross / (np.abs(cross) + 1e-10), n=n_fft)

            max_delay = min(int(sr * 0.2), n - 1)
            if max_delay <= 0:
                return 0
            search = np.concatenate([gcc[n_fft - max_delay :], gcc[: max_delay + 1]])
            return int(np.argmax(np.abs(search))) - max_delay
        except Exception as e:
            logger.warning("artifact_freedom_gate.py::_estimate_interchannel_lag_samples Ersatzpfad: %s", e)
            return 0

    # ── §2.50 Source Material Baseline ────────────────────────────────────

    def measure_source_baseline(
        self,
        audio: np.ndarray,
        sr: int,
        material_type: str = "digital",
    ) -> SourceMaterialBaseline:
        """§2.50 Measure artifact characteristics of the degraded input before pipeline.

        Returns a SourceMaterialBaseline that captures pre-existing carrier-chain
        properties so that gate evaluations can distinguish source artifacts from
        pipeline-introduced artifacts — enabling accurate carrier-chain inversion
        toward the original studio recording quality (§2.46, §2.50).
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mat_key = self._normalize_material(material_type)
        thresholds = self._get_thresholds(mat_key)
        corr_threshold = thresholds["phase_cancellation_corr"]

        baseline = SourceMaterialBaseline(material_type=mat_key)

        # ── Stereo field analysis (only for stereo audio) ──────────────────
        if audio.ndim == 2 and audio.shape[0] >= 2 and audio.shape[1] > sr // 10:
            left = audio[0]
            right = audio[1]
            baseline.interchannel_lag_samples = int(self._estimate_interchannel_lag_samples(audio, sr))
            frame_len = int(0.1 * sr)
            hop = frame_len // 2
            n_frames = max(1, (len(left) - frame_len) // hop)

            compat_values: list[float] = []
            corr_values: list[float] = []
            bad_frames: int = 0
            anti_phase_found: bool = False

            for i in range(n_frames):
                s = i * hop
                e = s + frame_len
                l_f = left[s:e]
                r_f = right[s:e]
                l_rms = float(np.sqrt(np.mean(l_f**2) + 1e-12))
                r_rms = float(np.sqrt(np.mean(r_f**2) + 1e-12))
                if l_rms < 1e-6 and r_rms < 1e-6:
                    continue  # silence frames don't count

                lr_corr, mono_compat = self._lr_corr_and_compat(l_f, r_f)
                compat_values.append(mono_compat)
                corr_values.append(lr_corr)
                # Meaningfully anti-phase: correlation < -0.15 (distinguish from
                # normal stereo decorrelation which can be -0.05 to -0.10 naturally).
                if lr_corr < -0.15:
                    anti_phase_found = True
                # Count as bad only when mono-compatibility is genuinely poor.
                # lr_corr < 0.0 alone is NOT a defect: wide stereo music routinely
                # has many frames with near-zero or slightly negative LR correlation
                # (properly panned instruments, stereo reverb). Using that condition
                # incorrectly flagged 48 % of Schlager frames as "critical stereo".
                if mono_compat < corr_threshold:
                    bad_frames += 1

            if compat_values:
                n = len(compat_values)
                baseline.phase_cancellation_ratio = bad_frames / n
                baseline.stereo_mono_compat_mean = float(np.mean(compat_values))
                baseline.stereo_lr_corr_mean = float(np.mean(corr_values))
                baseline.has_critical_stereo_issue = baseline.phase_cancellation_ratio > 0.20
                baseline.has_anti_phase_region = anti_phase_found

                if abs(int(baseline.interchannel_lag_samples)) > max(1, int(sr * 0.001)):
                    baseline.has_critical_stereo_issue = True

                if baseline.has_critical_stereo_issue:
                    logger.info(
                        "§2.50 Quellmaterial-Baseline: kritisches Stereo-Feldproblem "
                        "(Verhaeltnis=%.2f, mean_compat=%.3f, mean_corr=%.3f, lag=%d, mat=%s) — "
                        "Verarbeitungsschritt_14/Verarbeitungsschritt_15 werden als Remediation-Phasen aktiviert",
                        baseline.phase_cancellation_ratio,
                        baseline.stereo_mono_compat_mean,
                        baseline.stereo_lr_corr_mean,
                        baseline.interchannel_lag_samples,
                        mat_key,
                    )
                else:
                    logger.debug(
                        "§2.50 Quellmaterial-Baseline: stereo OK (Verhaeltnis=%.2f, mean_compat=%.3f, lag=%d, mat=%s)",
                        baseline.phase_cancellation_ratio,
                        baseline.stereo_mono_compat_mean,
                        baseline.interchannel_lag_samples,
                        mat_key,
                    )

        # ── HF-loss estimation ─────────────────────────────────────────────
        # Compare energy in 8–16 kHz vs. 2–8 kHz band as carrier-chain HF indicator.
        if audio.shape[-1] > sr // 4:
            mono = audio[0] if audio.ndim == 2 else audio
            n_fft = 4096
            win = np.hanning(min(n_fft, len(mono))).astype(np.float32)
            seg = mono[: len(win)] * win
            spec = np.abs(np.fft.rfft(seg, n=n_fft)) + 1e-12
            freq_res = sr / n_fft
            lo_bins = slice(int(2000 / freq_res), int(8000 / freq_res))
            hi_bins = slice(int(8000 / freq_res), int(16000 / freq_res))
            lo_rms = float(np.sqrt(np.mean(spec[lo_bins] ** 2)))
            hi_rms = float(np.sqrt(np.mean(spec[hi_bins] ** 2)))
            if lo_rms > 1e-10:
                hf_ratio = hi_rms / lo_rms
                # Flat reference ≈ 1.0; cassette/mp3_low often 0.3–0.5 → 6–10 dB loss
                hf_loss_db = float(np.clip(-20.0 * np.log10(hf_ratio + 1e-12), 0.0, 40.0))
                baseline.hf_loss_db = hf_loss_db

        return baseline
