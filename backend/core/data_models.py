"""
AURIK Core Data Models

Formal data structures implementing the AURIK Architecture Specification (Chapter 4).
These models replace informal dict/list structures with typed Pydantic models for:
- Type safety and validation
- JSON serialization/deserialization
- API compatibility
- Audit trail completeness

Based on AURIK Architecture Specification:
- Section 3.1: Analysis Engine
- Section 3.2: Aesthetic Judgment Model
- Section 3.5: Quality Assurance
- Section 4.1: Core Data Structures
- Section 4.2: Archiving Strategy
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Enumerations
# ============================================================================


class Genre(str, Enum):
    """Musical genres with specific aesthetic weighting profiles (Spec 3.2.2)"""

    CLASSICAL = "classical"
    JAZZ = "jazz"
    ROCK_METAL = "rock_metal"
    ELECTRONIC = "electronic"
    VOCAL_POP = "vocal_pop"
    SCHLAGER = "schlager"  # German Schlager / Folk Pop (priority recognition)
    VINTAGE_ANALOG = "vintage_analog"
    UNKNOWN = "unknown"


class MediaType(str, Enum):
    """Erkannte Quell-Medientypen (Spec 3.1.2)."""

    VINYL = "vinyl"
    TAPE = "tape"
    CASSETTE = "cassette"
    CD = "cd"
    DIGITAL_NATIVE = "digital_native"
    RADIO_BROADCAST = "radio_broadcast"
    UNKNOWN = "unknown"


# §DefectType-Konsolidierung: kanonische Quelle ist defect_scanner.py (54+ Defekttypen)
from backend.core.defect_scanner import DefectType  # type: ignore[import]

# ============================================================================
# Audio File Model
# ============================================================================


class AudioFile(BaseModel):
    """
    Audio file metadata with integrity tracking.

    Spec Reference: Section 4.1 - ResturationJob structure
    """

    file_path: str = Field(description="Absolute path to audio file")
    file_hash: str = Field(description="SHA256 hash for integrity verification")
    format: str = Field(description="Audio format (WAV, FLAC, MP3, etc.)")
    sample_rate: int = Field(description="Sample rate in Hz", gt=0)
    bit_depth: int | None = Field(None, description="Bit depth (16, 24, 32)")
    channels: int = Field(description="Number of audio channels", ge=1, le=8)
    duration: float = Field(description="Duration in seconds", ge=0.0)
    file_size: int = Field(description="File size in bytes", ge=0)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic model configuration with JSON schema examples."""

        json_schema_extra = {
            "example": {
                "file_path": "/archive/originals/abc123/input.wav",
                "file_hash": "9f86d081884c7d659a2feaa0c55ad015...",
                "format": "WAV",
                "sample_rate": 48000,
                "bit_depth": 24,
                "channels": 2,
                "duration": 180.5,
                "file_size": 51840000,
                "created_at": "2024-01-15T10:30:00Z",
            }
        }


# ============================================================================
# Analysis Profile (Spec 3.1)
# ============================================================================


class FormatInfo(BaseModel):
    """Formatiert recognition and validation (Spec 3.1.1)."""

    container_format: str = Field(description="Container format (WAV, FLAC, etc.)")
    codec: str = Field(description="Audio codec")
    sample_rate: int = Field(gt=0)
    bit_depth: int | None = None
    channels: int = Field(ge=1)
    dc_offset: float = Field(description="DC offset in dB")
    has_clipping: bool = Field(description="Clipping flags detected")


class MaterialChainAnalysis(BaseModel):
    """Material and conversion chain analysis (Spec 3.1.2)"""

    detected_medium: MediaType = Field(description="Primary source medium")
    medium_confidence: float = Field(ge=0.0, le=1.0)

    # Vinyl-specific
    vinyl_rpm: int | None = Field(None, description="33/45/78 RPM if vinyl detected")

    # Tape-specific
    tape_type: str | None = Field(None, description="Cassette, Reel-to-Reel, 8-Track")

    # Conversion chain
    adc_type: str | None = Field(None, description="Detected ADC type")
    resampling_artifacts: bool = Field(default=False)
    lossy_codec_history: list[str] = Field(default_factory=list, description="MP3 encoder fingerprints")
    generation_count: int = Field(default=1, description="Estimated copy generation", ge=1)


class SpectralAnalysis(BaseModel):
    """Frequency spectrum analysis (Spec 3.1.2)"""

    spectral_centroid: float = Field(description="Spectral centroid in Hz", ge=0.0)
    spectral_rolloff: float = Field(description="Spectral rolloff frequency in Hz", ge=0.0)
    spectral_flux: float = Field(description="Spectral flux (change rate)")
    bandwidth: float = Field(description="Effective bandwidth in Hz", ge=0.0)
    has_aliasing: bool = Field(default=False, description="Aliasing artifacts detected")
    frequency_gaps: list[tuple[float, float]] = Field(default_factory=list, description="Frequency gaps (Hz ranges)")


class DynamicsAnalysis(BaseModel):
    """Dynamics and level analysis (Spec 3.1.2)"""

    lufs_integrated: float = Field(description="Integrated loudness (LUFS)")
    lufs_short_term: float = Field(description="Short-term loudness (LUFS)")
    lufs_momentary: float = Field(description="Momentary loudness (LUFS)")
    dynamic_range_db: float = Field(description="Dynamic range (DR)", ge=0.0)
    crest_factor_db: float = Field(description="Crest factor", ge=0.0)
    true_peak_dbfs: float = Field(description="True peak level (dBFS)")
    rms_db: float = Field(description="RMS level (dB)")
    loudness_range_lu: float = Field(description="Loudness range (LRA) in LU", ge=0.0)


class StereoAnalysis(BaseModel):
    """Stereo field analysis (Spec 3.1.2)"""

    mid_side_balance: float = Field(description="Mid/Side balance ratio", ge=0.0)
    stereo_width: float = Field(description="Stereo width (0=mono, 1=normal, >1=wide)", ge=0.0)
    phase_coherence: float = Field(description="Phase coherence (0-1)", ge=0.0, le=1.0)
    iacc: float = Field(description="Interaural Cross-Correlation coefficient", ge=-1.0, le=1.0)
    panning_distribution: dict[str, float] = Field(default_factory=dict, description="Panning histogram")
    mono_compatibility_score: float = Field(description="Mono compatibility (0-1)", ge=0.0, le=1.0)


class DefectDetection(BaseModel):
    """Individual defect detection result (Spec 3.1.3)"""

    defect_type: DefectType
    severity: float = Field(description="Severity score (0-1)", ge=0.0, le=1.0)
    confidence: float = Field(description="Detection confidence (0-1)", ge=0.0, le=1.0)
    affected_frequency_range: tuple[float, float] | None = Field(None, description="Frequency range in Hz")
    temporal_locations: list[float] = Field(default_factory=list, description="Time locations in seconds")
    classification_details: dict[str, Any] = Field(default_factory=dict, description="Type-specific details")


class MusicalContext(BaseModel):
    """Musical context analysis (Spec 3.1.4)"""

    genre: Genre = Field(description="Detected genre")
    genre_confidence: float = Field(ge=0.0, le=1.0)

    # Instrumentation
    dominant_instruments: list[str] = Field(default_factory=list, description="Detected instruments")

    # Structure
    tempo_bpm: float | None = Field(None, description="Tempo in BPM", ge=0.0)
    time_signature: str | None = Field(None, description="Time signature (4/4, 3/4, etc.)")
    key_signature: str | None = Field(None, description="Musical key (C major, A minor, etc.)")

    # Arrangement
    structure_segments: list[dict[str, Any]] = Field(
        default_factory=list, description="Song structure (intro, verse, chorus)"
    )

    # Dynamics
    dynamic_contour: list[float] = Field(default_factory=list, description="Loudness contour over time")
    harmonic_complexity: float = Field(description="Harmonic complexity score (0-1)", ge=0.0, le=1.0, default=0.5)


class VocalAnalysis(BaseModel):
    """Vocal and speech analysis (Spec 3.1.5)"""

    has_vocals: bool = Field(description="Vocals detected")
    vocal_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    # Speaker characteristics
    num_speakers: int = Field(ge=0, default=0)
    language: str | None = Field(None, description="Detected language (ISO 639-1)")
    language_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    # Emotional content
    valence: float | None = Field(None, description="Emotional valence (-1 to 1)", ge=-1.0, le=1.0)
    arousal: float | None = Field(None, description="Emotional arousal (0-1)", ge=0.0, le=1.0)


class FeatureVectors(BaseModel):
    """Extrahierte Feature-Vektoren (Spec 3.1.6)."""

    # Temporal features
    onset_times: list[float] = Field(default_factory=list, description="Onset times in seconds")
    beat_times: list[float] = Field(default_factory=list, description="Beat times in seconds")
    tempo_bpm: float | None = None

    # Spectral features
    mfccs: list[list[float]] | None = Field(None, description="MFCCs (40 coefficients per frame)")
    spectral_contrast: list[list[float]] | None = Field(None, description="Spectral contrast per frame")
    chroma_features: list[list[float]] | None = Field(None, description="Chroma features (12 bins per frame)")

    # Tonal features
    pitch_contour: list[float] | None = Field(None, description="F0 contour in Hz")
    pitch_confidence: list[float] | None = Field(None, description="Pitch confidence per frame")
    harmonicity: float | None = Field(None, description="Harmonic-to-Noise Ratio", ge=0.0)

    # Rhythmic features
    rhythm_patterns: dict[str, Any] = Field(default_factory=dict, description="Rhythm pattern analysis")
    syncopation_index: float | None = Field(None, description="Syncopation level", ge=0.0, le=1.0)


class AnalysisProfile(BaseModel):
    """
    Comprehensive analysis profile from Analysis Engine.

    Spec Reference: Section 3.1 - Analysis Engine
    """

    # Metadata
    analyzed_at: datetime = Field(default_factory=datetime.now)
    analysis_version: str = Field(default="1.0.0", description="Analysis engine version")

    # Core analysis results
    format_info: FormatInfo
    material_chain: MaterialChainAnalysis
    spectral: SpectralAnalysis
    dynamics: DynamicsAnalysis
    stereo: StereoAnalysis

    # Defects and issues
    detected_defects: list[DefectDetection] = Field(default_factory=list)
    overall_quality_score: float = Field(description="Overall quality assessment (0-1)", ge=0.0, le=1.0)

    # Musical content
    musical_context: MusicalContext
    vocal_analysis: VocalAnalysis
    feature_vectors: FeatureVectors

    # Raw feature dictionary (backward compatibility)
    raw_features: dict[str, Any] = Field(default_factory=dict, description="Additional raw features")

    @field_validator("detected_defects")
    @classmethod
    def sort_defects_by_severity(cls, v: list[DefectDetection]) -> list[DefectDetection]:
        """Sortiert defects by severity (highest first)."""
        return sorted(v, key=lambda d: d.severity, reverse=True)


# ============================================================================
# Aesthetic Judgment Model (Spec 3.2)
# ============================================================================


class GenreWeights(BaseModel):
    """
    Genre-adaptive weighting for aesthetic dimensions.

    Spec Reference: Section 3.2.2 - Genre-adaptive Gewichtung
    """

    brilliance: float = Field(ge=0.0, le=1.0)
    transparency: float = Field(ge=0.0, le=1.0)
    naturalness: float = Field(ge=0.0, le=1.0)
    authenticity: float = Field(ge=0.0, le=1.0)
    emotionality: float = Field(ge=0.0, le=1.0)
    warmth: float = Field(ge=0.0, le=1.0)
    spatiality: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "brilliance",
        "transparency",
        "naturalness",
        "authenticity",
        "emotionality",
        "warmth",
        "spatiality",
    )
    @classmethod
    def check_weights_sum(cls, v: float, _info) -> float:
        """Weights should sum to approximately 1.0 (validated at model level)"""
        return v

    def validate_sum(self) -> bool:
        """Prüft that all weights sum to 1.0 (±0.01 tolerance)."""
        total = (
            self.brilliance
            + self.transparency
            + self.naturalness
            + self.authenticity
            + self.emotionality
            + self.warmth
            + self.spatiality
        )
        return abs(total - 1.0) < 0.01


class AestheticScores(BaseModel):
    """
    Individual aesthetic dimension scores (proxy metrics).

    Spec Reference: Section 1.2 - Musikalische Zielgrößen
    """

    # Brilliance (Spec 1.2: Brillanz)
    brilliance: float = Field(description="Clarity and presence in HF range", ge=0.0, le=1.0)

    # Transparency (Spec 1.2: Transparenz)
    transparency: float = Field(description="Instrument separation and clarity", ge=0.0, le=1.0)

    # Naturalness (Spec 1.2: Natürlichkeit)
    naturalness: float = Field(description="Organic sound without artifacts", ge=0.0, le=1.0)

    # Authenticity (Spec 1.2: Authentizität)
    authenticity: float = Field(description="Faithfulness to original", ge=0.0, le=1.0)

    # Emotionality (Spec 1.2: Emotionalität)
    emotionality: float = Field(description="Preservation of musical expression", ge=0.0, le=1.0)

    # Warmth (Spec 1.2: Wärme)
    warmth: float = Field(description="Pleasant low-mids and harmonic saturation", ge=0.0, le=1.0)

    # Spatiality (Spec 1.2: Räumlichkeit)
    spatiality: float = Field(description="3D depth and natural stereo image", ge=0.0, le=1.0)

    # Supplementary metrics
    proxy_details: dict[str, Any] = Field(default_factory=dict, description="Detailed proxy metric values")


class ConstraintCheckResult(BaseModel):
    """Result of constraint system validation (Spec 3.2.3)"""

    constraint_name: str = Field(description="Name of the constraint")
    passed: bool = Field(description="Whether constraint was satisfied")
    measured_value: float = Field(description="Measured value")
    threshold_value: float = Field(description="Required threshold")
    severity: str = Field(description="Violation severity: info, warning, error")
    message: str = Field(description="Human-readable description")


class QualityReport(BaseModel):
    """
    Complete quality assessment report.

    Spec Reference: Section 3.5 - Quality Assurance & Audit
    """

    # Timestamp
    evaluated_at: datetime = Field(default_factory=datetime.now)

    # Composite Aesthetic Score (Spec 3.2.1)
    cas_before: float = Field(description="CAS before processing", ge=0.0, le=1.0)
    cas_after: float = Field(description="CAS after processing", ge=0.0, le=1.0)
    cas_improvement: float = Field(description="CAS improvement delta", ge=-1.0, le=1.0)

    # Individual aesthetic scores
    aesthetic_scores_before: AestheticScores
    aesthetic_scores_after: AestheticScores

    # Objective metrics (Spec 3.5.1)
    dnsmos_score: float | None = Field(
        None, description="Legacy compat field (forbidden for music in runtime)", ge=0.0, le=5.0
    )
    cdpam_score: float | None = Field(None, description="Legacy compat field (forbidden for music in runtime)", ge=0.0)

    # Constraint system results (Spec 3.2.3)
    constraints_satisfied: bool = Field(description="All constraints passed")
    constraint_checks: list[ConstraintCheckResult] = Field(default_factory=list)

    # Issues and warnings
    warnings: list[str] = Field(default_factory=list, description="Quality warnings")
    errors: list[str] = Field(default_factory=list, description="Quality errors")

    # Supplementary data
    additional_metrics: dict[str, Any] = Field(default_factory=dict, description="Additional metric values")


# ============================================================================
# Processing Step (Spec 4.1)
# ============================================================================


class ProcessingStep(BaseModel):
    """
    Individual processing step with full auditability.

    Spec Reference: Section 4.1 - ProcessingStep structure
    """

    # Identity
    step_id: int = Field(description="Sequential step number", ge=0)
    operation: str = Field(description="Operation type (e.g., 'denoise', 'declip')")

    # Model information
    model_name: str = Field(description="Model name (e.g., 'DeepFilterNet')")
    model_version: str = Field(default="unknown", description="Model version for reproducibility")
    parameters: dict[str, Any] = Field(description="Processing parameters")

    # Integrity tracking
    input_hash: str = Field(description="SHA256 hash of input audio")
    output_hash: str = Field(description="SHA256 hash of output audio")

    # Quality tracking
    cas_before: float = Field(description="CAS score before this step", ge=0.0, le=1.0)
    cas_after: float = Field(description="CAS score after this step", ge=0.0, le=1.0)
    cas_delta: float = Field(description="CAS change from this step", ge=-1.0, le=1.0)

    # Decision audit trail
    decision_reason: str = Field(description="Why this step was chosen/applied")
    skipped: bool = Field(default=False, description="Whether step was skipped")
    skip_reason: str | None = Field(None, description="Reason for skipping if skipped=True")

    # Execution metadata
    executed_at: datetime = Field(default_factory=datetime.now)
    duration_seconds: float = Field(description="Processing duration", ge=0.0)

    # Supplementary
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")


# ============================================================================
# Restoration Job (Spec 4.1)
# ============================================================================


class ResturationJob(BaseModel):
    """
    Central tracking object for complete restoration workflow.

    Spec Reference: Section 4.1 - ResturationJob structure
    """

    # Identity
    job_id: UUID = Field(default_factory=uuid4, description="Unique job identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(None, description="Completion timestamp")

    # Files
    input_file: AudioFile = Field(description="Original input file")
    output_file: AudioFile | None = Field(None, description="Final output file")
    intermediate_files: list[AudioFile] = Field(default_factory=list, description="Intermediate processing files")

    # Analysis and processing
    analysis_profile: AnalysisProfile = Field(description="Comprehensive audio analysis")
    processing_chain: list[ProcessingStep] = Field(default_factory=list, description="Ordered list of processing steps")

    # Quality assessment
    quality_report: QualityReport | None = Field(None, description="Final quality assessment")

    # Model versions (for reproducibility)
    model_versions: dict[str, str] = Field(default_factory=dict, description="Model name -> version mapping")

    # Archiving metadata
    archived: bool = Field(default=False, description="Whether job has been archived")
    archive_path: str | None = Field(None, description="Path to archived data")

    # User and configuration
    user_id: str | None = Field(None, description="User who submitted job")
    config: dict[str, Any] = Field(default_factory=dict, description="Job configuration")

    # Status and logging
    status: str = Field(default="created", description="Job status: created, running, completed, failed")
    log_entries: list[str] = Field(default_factory=list, description="Execution log entries")
    error_message: str | None = Field(None, description="Error message if failed")

    class Config:
        """Pydantic model configuration with JSON schema examples."""

        json_schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "created_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:05:30Z",
            }
        }

    def add_processing_step(self, step: ProcessingStep) -> None:
        """Fügt hinzu: a processing step to the chain."""
        self.processing_chain.append(step)
        self.log_entries.append(f"Step {step.step_id}: {step.operation} ({step.model_name})")

    def mark_completed(self) -> None:
        """Mark job as completed"""
        self.completed_at = datetime.now()
        self.status = "completed"

    def mark_failed(self, error: str) -> None:
        """Mark job as failed with error message"""
        self.status = "failed"
        self.error_message = error
        self.completed_at = datetime.now()

    def get_total_duration(self) -> float:
        """Calculate total processing duration in seconds"""
        if not self.completed_at or not self.created_at:
            return 0.0
        return (self.completed_at - self.created_at).total_seconds()

    def get_total_cas_improvement(self) -> float:
        """Calculate total CAS improvement across all steps"""
        if not self.quality_report:
            return 0.0
        return self.quality_report.cas_improvement


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "AestheticScores",
    "AnalysisProfile",
    # Audio file
    "AudioFile",
    "ConstraintCheckResult",
    "DefectDetection",
    "DefectType",
    "DynamicsAnalysis",
    "FeatureVectors",
    # Analysis components
    "FormatInfo",
    # Enums
    "Genre",
    # Aesthetic judgment
    "GenreWeights",
    "MaterialChainAnalysis",
    "MediaType",
    "MusicalContext",
    # Processing tracking
    "ProcessingStep",
    "QualityReport",
    "ResturationJob",
    "SpectralAnalysis",
    "StereoAnalysis",
    "VocalAnalysis",
]
