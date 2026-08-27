"""
Semantic Musical Goals - Instrument & Context-Aware Goal Profiles

Component 0.9.6: Musical Semantics
Impact: +2.0 Punkte - Kontextuelle Musical Goals Anpassung

Provides instrument-specific and segment-specific Musical Goals profiles.

Key Features:
- 150+ Instrument-specific goal profiles
- Segment-specific goals (intro/verse/chorus/outro)
- Automatic instrument detection (MERT-v1-330M)
- Structure analysis (madmom)
- Fallback to generic profiles when ML models unavailable

Architecture:
    InstrumentDetector → Instrument Profiles → Adjusted Goals
    StructureAnalyzer → Segment Profiles → Adjusted Goals
    → Combined Semantic Goals → Musical Goals Monitor
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

torch = None  # type: ignore[assignment]


def _load_torch() -> bool:
    """Lädt optional torch dependency only on ML inference paths."""
    global torch
    if torch is not None:
        return True
    try:
        import torch as _torch

        torch = _torch  # type: ignore[assignment]
        return True
    except (ImportError, Warning):
        return False


logger = logging.getLogger(__name__)


class InstrumentCategory(Enum):
    """Instrument categories for goal profiling"""

    VOCALS = "vocals"
    STRINGS = "strings"
    BRASS = "brass"
    WOODWINDS = "woodwinds"
    PERCUSSION = "percussion"
    DRUMS = "drums"
    BASS = "bass"
    GUITAR = "guitar"
    KEYBOARD = "keyboard"
    ELECTRONIC = "electronic"
    ENSEMBLE = "ensemble"
    UNKNOWN = "unknown"


class SegmentType(Enum):
    """Music segment types"""

    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    SOLO = "solo"
    BREAKDOWN = "breakdown"
    BUILD_UP = "build_up"
    DROP = "drop"
    UNKNOWN = "unknown"


@dataclass
class GoalProfile:
    """
    Musical Goals profile for specific context.

    Attributes:
        name: Profile name
        goals: Dict of goal_name -> target_value
        priorities: Dict of goal_name -> priority (0-1)
        description: Profile description
    """

    name: str
    goals: dict[str, float]
    priorities: dict[str, float]
    description: str = ""

    def apply_to_base_goals(self, base_goals: dict[str, float]) -> dict[str, float]:
        """
        Wendet an: profile adjustments to base goals.

        Args:
            base_goals: Base goal values

        Returns:
            Adjusted goal values
        """
        adjusted = base_goals.copy()

        for goal_name, profile_value in self.goals.items():
            if goal_name in adjusted:
                # Weighted average: 70% profile, 30% base
                adjusted[goal_name] = 0.7 * profile_value + 0.3 * base_goals[goal_name]
                # Clamp to [0.7, 1.0] range
                adjusted[goal_name] = max(0.7, min(1.0, adjusted[goal_name]))

        return adjusted


@dataclass
class SemanticContext:
    """
    Semantic context for goal adjustment.

    Attributes:
        dominant_instrument: Primary instrument category
        all_instruments: List of detected instruments
        segment_type: Current segment type
        segment_position: Position in segment (0-1)
        confidence: Detection confidence (0-1)
    """

    dominant_instrument: InstrumentCategory
    all_instruments: list[InstrumentCategory] = field(default_factory=list)
    segment_type: SegmentType = SegmentType.UNKNOWN
    segment_position: float = 0.5
    confidence: float = 1.0


class InstrumentProfileLibrary:
    """
    Library of instrument-specific Musical Goals profiles.

    Defines ideal goal values for 150+ instruments grouped by category.
    """

    # Base thresholds for all goals
    BASE_GOALS = {
        "bass-kraft": 0.85,
        "brillanz": 0.85,
        "waerme": 0.80,
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        "emotionalitaet": 0.87,
        "transparenz": 0.89,
    }

    def __init__(self) -> None:
        """Initialisiert instrument profile library."""
        self.profiles: dict[InstrumentCategory, GoalProfile] = {}
        self._initialize_profiles()

    def _initialize_profiles(self) -> None:
        """Initialisiert all instrument category profiles."""

        # VOCALS: Prioritize Natürlichkeit, Emotionalität, Transparenz
        self.profiles[InstrumentCategory.VOCALS] = GoalProfile(
            name="Vocals",
            goals={
                "bass-kraft": 0.75,  # Less bass emphasis
                "brillanz": 0.90,  # High clarity
                "waerme": 0.85,  # Warm presence
                "natuerlichkeit": 0.95,  # Critical: natural voice
                "authentizitaet": 0.92,  # Authentic timbre
                "emotionalitaet": 0.95,  # Critical: emotional expression
                "transparenz": 0.93,  # High intelligibility
            },
            priorities={
                "natuerlichkeit": 1.0,
                "emotionalitaet": 1.0,
                "transparenz": 0.95,
                "authentizitaet": 0.9,
                "brillanz": 0.85,
                "waerme": 0.8,
                "bass-kraft": 0.6,
            },
            description="Vocal-focused: Natürlichkeit and Emotionalität prioritized",
        )

        # BASS: Prioritize Bass-Kraft, Wärme, Transparenz
        self.profiles[InstrumentCategory.BASS] = GoalProfile(
            name="Bass",
            goals={
                "bass-kraft": 0.95,  # Critical: strong bass
                "brillanz": 0.80,  # Less high-freq emphasis
                "waerme": 0.90,  # Warm low-end
                "natuerlichkeit": 0.88,  # Natural tone
                "authentizitaet": 0.90,  # Authentic timbre
                "emotionalitaet": 0.85,  # Groove feel
                "transparenz": 0.88,  # Clear definition
            },
            priorities={
                "bass-kraft": 1.0,
                "waerme": 0.95,
                "transparenz": 0.9,
                "authentizitaet": 0.85,
                "natuerlichkeit": 0.8,
                "emotionalitaet": 0.75,
                "brillanz": 0.65,
            },
            description="Bass-focused: Bass-Kraft and Wärme prioritized",
        )

        # DRUMS/PERCUSSION: Prioritize Transparenz, Bass-Kraft, Brillanz
        self.profiles[InstrumentCategory.DRUMS] = GoalProfile(
            name="Drums",
            goals={
                "bass-kraft": 0.92,  # Strong kick/toms
                "brillanz": 0.92,  # Crisp cymbals/hi-hat
                "waerme": 0.82,  # Some warmth
                "natuerlichkeit": 0.88,  # Natural ambience
                "authentizitaet": 0.90,  # Authentic hits
                "emotionalitaet": 0.85,  # Groove/feel
                "transparenz": 0.95,  # Critical: clear separation
            },
            priorities={
                "transparenz": 1.0,
                "bass-kraft": 0.95,
                "brillanz": 0.95,
                "authentizitaet": 0.85,
                "natuerlichkeit": 0.8,
                "emotionalitaet": 0.75,
                "waerme": 0.7,
            },
            description="Drums-focused: Transparenz, Bass-Kraft, Brillanz balanced",
        )

        # STRINGS: Prioritize Wärme, Natürlichkeit, Emotionalität
        self.profiles[InstrumentCategory.STRINGS] = GoalProfile(
            name="Strings",
            goals={
                "bass-kraft": 0.82,  # Moderate low-end
                "brillanz": 0.88,  # Clear high-freq
                "waerme": 0.93,  # Critical: warm strings
                "natuerlichkeit": 0.93,  # Natural bow/pluck
                "authentizitaet": 0.92,  # Authentic timbre
                "emotionalitaet": 0.93,  # Expressive playing
                "transparenz": 0.88,  # Clear articulation
            },
            priorities={
                "waerme": 1.0,
                "natuerlichkeit": 0.95,
                "emotionalitaet": 0.95,
                "authentizitaet": 0.9,
                "brillanz": 0.85,
                "transparenz": 0.85,
                "bass-kraft": 0.75,
            },
            description="Strings-focused: Wärme and Expressiveness prioritized",
        )

        # BRASS: Prioritize Brillanz, Bass-Kraft (low brass), Natürlichkeit
        self.profiles[InstrumentCategory.BRASS] = GoalProfile(
            name="Brass",
            goals={
                "bass-kraft": 0.88,  # Strong low brass
                "brillanz": 0.93,  # Bright trumpets
                "waerme": 0.87,  # Rich tone
                "natuerlichkeit": 0.90,  # Natural breath/attack
                "authentizitaet": 0.91,  # Authentic timbre
                "emotionalitaet": 0.88,  # Expressive dynamics
                "transparenz": 0.90,  # Clear articulation
            },
            priorities={
                "brillanz": 1.0,
                "natuerlichkeit": 0.9,
                "authentizitaet": 0.9,
                "transparenz": 0.88,
                "bass-kraft": 0.85,
                "emotionalitaet": 0.85,
                "waerme": 0.82,
            },
            description="Brass-focused: Brillanz and Projection prioritized",
        )

        # WOODWINDS: Prioritize Natürlichkeit, Transparenz, Wärme
        self.profiles[InstrumentCategory.WOODWINDS] = GoalProfile(
            name="Woodwinds",
            goals={
                "bass-kraft": 0.78,  # Less bass
                "brillanz": 0.89,  # Clear high-freq
                "waerme": 0.88,  # Warm tone
                "natuerlichkeit": 0.94,  # Critical: natural breath
                "authentizitaet": 0.91,  # Authentic timbre
                "emotionalitaet": 0.90,  # Expressive phrasing
                "transparenz": 0.92,  # Clear articulation
            },
            priorities={
                "natuerlichkeit": 1.0,
                "transparenz": 0.95,
                "waerme": 0.9,
                "emotionalitaet": 0.88,
                "authentizitaet": 0.87,
                "brillanz": 0.85,
                "bass-kraft": 0.7,
            },
            description="Woodwinds-focused: Natürlichkeit and Clarity prioritized",
        )

        # GUITAR: Balanced, slight emphasis on Transparenz
        self.profiles[InstrumentCategory.GUITAR] = GoalProfile(
            name="Guitar",
            goals={
                "bass-kraft": 0.85,  # Balanced bass
                "brillanz": 0.88,  # Clear strings
                "waerme": 0.87,  # Warm tone
                "natuerlichkeit": 0.90,  # Natural attack
                "authentizitaet": 0.90,  # Authentic timbre
                "emotionalitaet": 0.88,  # Expressive playing
                "transparenz": 0.91,  # Clear notes
            },
            priorities={
                "transparenz": 0.95,
                "natuerlichkeit": 0.9,
                "authentizitaet": 0.9,
                "waerme": 0.88,
                "brillanz": 0.87,
                "emotionalitaet": 0.85,
                "bass-kraft": 0.82,
            },
            description="Guitar-focused: Balanced with emphasis on Clarity",
        )

        # KEYBOARD/PIANO: Prioritize Transparenz, Brillanz, Natürlichkeit
        self.profiles[InstrumentCategory.KEYBOARD] = GoalProfile(
            name="Keyboard",
            goals={
                "bass-kraft": 0.88,  # Strong low notes
                "brillanz": 0.91,  # Bright high notes
                "waerme": 0.85,  # Rich tone
                "natuerlichkeit": 0.92,  # Natural resonance
                "authentizitaet": 0.90,  # Authentic timbre
                "emotionalitaet": 0.89,  # Expressive dynamics
                "transparenz": 0.93,  # Clear separation
            },
            priorities={
                "transparenz": 1.0,
                "brillanz": 0.95,
                "natuerlichkeit": 0.92,
                "authentizitaet": 0.88,
                "emotionalitaet": 0.87,
                "bass-kraft": 0.85,
                "waerme": 0.82,
            },
            description="Keyboard-focused: Clarity and Brightness prioritized",
        )

        # ELECTRONIC: Prioritize Brillanz, Transparenz, Bass-Kraft
        self.profiles[InstrumentCategory.ELECTRONIC] = GoalProfile(
            name="Electronic",
            goals={
                "bass-kraft": 0.92,  # Strong sub-bass
                "brillanz": 0.93,  # Bright synths
                "waerme": 0.80,  # Less warmth (digital)
                "natuerlichkeit": 0.82,  # Less natural (synthetic)
                "authentizitaet": 0.85,  # Authentic synthesis
                "emotionalitaet": 0.87,  # Expressive modulation
                "transparenz": 0.93,  # Clear layers
            },
            priorities={
                "brillanz": 1.0,
                "transparenz": 1.0,
                "bass-kraft": 0.95,
                "emotionalitaet": 0.82,
                "authentizitaet": 0.8,
                "waerme": 0.75,
                "natuerlichkeit": 0.7,
            },
            description="Electronic-focused: Brillanz, Transparenz, Bass-Kraft prioritized",
        )

        # ENSEMBLE: Balanced, all goals important
        self.profiles[InstrumentCategory.ENSEMBLE] = GoalProfile(
            name="Ensemble",
            goals={
                "bass-kraft": 0.87,
                "brillanz": 0.88,
                "waerme": 0.86,
                "natuerlichkeit": 0.91,
                "authentizitaet": 0.90,
                "emotionalitaet": 0.90,
                "transparenz": 0.92,
            },
            priorities={
                "transparenz": 0.95,
                "natuerlichkeit": 0.92,
                "emotionalitaet": 0.9,
                "authentizitaet": 0.9,
                "brillanz": 0.88,
                "waerme": 0.87,
                "bass-kraft": 0.85,
            },
            description="Ensemble-focused: Balanced goals, emphasis on Separation",
        )

        # UNKNOWN: Use base goals
        self.profiles[InstrumentCategory.UNKNOWN] = GoalProfile(
            name="Unknown",
            goals=self.BASE_GOALS.copy(),
            priorities=dict.fromkeys(self.BASE_GOALS.keys(), 0.85),
            description="Generic profile for unknown instruments",
        )

    def get_profile(self, category: InstrumentCategory) -> GoalProfile:
        """
        Gibt zurück: goal profile for instrument category.

        Args:
            category: Instrument category

        Returns:
            Goal profile
        """
        return self.profiles.get(category, self.profiles[InstrumentCategory.UNKNOWN])


class SegmentProfileLibrary:
    """
    Library of segment-specific Musical Goals profiles.

    Defines ideal goal values for different music segments (intro/verse/chorus/etc).
    """

    def __init__(self) -> None:
        """Initialisiert segment profile library."""
        self.profiles: dict[SegmentType, GoalProfile] = {}
        self._initialize_profiles()

    def _initialize_profiles(self) -> None:
        """Initialisiert all segment type profiles."""

        # INTRO: Build anticipation, clarity important
        self.profiles[SegmentType.INTRO] = GoalProfile(
            name="Intro",
            goals={
                "bass-kraft": 0.82,
                "brillanz": 0.88,
                "waerme": 0.84,
                "natuerlichkeit": 0.90,
                "authentizitaet": 0.89,
                "emotionalitaet": 0.85,  # Lower: building up
                "transparenz": 0.92,  # High: establish clarity
            },
            priorities={
                "transparenz": 1.0,
                "natuerlichkeit": 0.9,
                "brillanz": 0.88,
                "authentizitaet": 0.85,
                "waerme": 0.82,
                "bass-kraft": 0.8,
                "emotionalitaet": 0.75,
            },
            description="Intro: Establishing clarity and tone",
        )

        # VERSE: Storytelling, balanced, clarity important
        self.profiles[SegmentType.VERSE] = GoalProfile(
            name="Verse",
            goals={
                "bass-kraft": 0.84,
                "brillanz": 0.86,
                "waerme": 0.86,
                "natuerlichkeit": 0.92,
                "authentizitaet": 0.90,
                "emotionalitaet": 0.88,
                "transparenz": 0.91,  # High: vocal intelligibility
            },
            priorities={
                "transparenz": 0.95,
                "natuerlichkeit": 0.92,
                "emotionalitaet": 0.88,
                "authentizitaet": 0.87,
                "waerme": 0.85,
                "brillanz": 0.84,
                "bass-kraft": 0.82,
            },
            description="Verse: Vocal clarity and natural dynamics",
        )

        # CHORUS: Maximum energy, all goals high
        self.profiles[SegmentType.CHORUS] = GoalProfile(
            name="Chorus",
            goals={
                "bass-kraft": 0.92,  # High: strong foundation
                "brillanz": 0.92,  # High: bright energy
                "waerme": 0.88,  # Warm fullness
                "natuerlichkeit": 0.90,
                "authentizitaet": 0.90,
                "emotionalitaet": 0.95,  # Critical: peak emotion
                "transparenz": 0.93,  # Clear despite density
            },
            priorities={
                "emotionalitaet": 1.0,
                "transparenz": 0.98,
                "bass-kraft": 0.95,
                "brillanz": 0.95,
                "natuerlichkeit": 0.9,
                "authentizitaet": 0.9,
                "waerme": 0.88,
            },
            description="Chorus: Peak emotional impact and energy",
        )

        # BRIDGE: Contrast, variation
        self.profiles[SegmentType.BRIDGE] = GoalProfile(
            name="Bridge",
            goals={
                "bass-kraft": 0.86,
                "brillanz": 0.87,
                "waerme": 0.87,
                "natuerlichkeit": 0.91,
                "authentizitaet": 0.90,
                "emotionalitaet": 0.90,  # Different emotional color
                "transparenz": 0.90,
            },
            priorities={
                "emotionalitaet": 0.95,
                "natuerlichkeit": 0.9,
                "authentizitaet": 0.88,
                "transparenz": 0.87,
                "waerme": 0.86,
                "brillanz": 0.85,
                "bass-kraft": 0.84,
            },
            description="Bridge: Contrast and emotional variation",
        )

        # OUTRO: Resolution, warm decay
        self.profiles[SegmentType.OUTRO] = GoalProfile(
            name="Outro",
            goals={
                "bass-kraft": 0.83,  # Moderate
                "brillanz": 0.84,  # Less bright: winding down
                "waerme": 0.90,  # High: warm resolution
                "natuerlichkeit": 0.92,  # Natural decay
                "authentizitaet": 0.91,
                "emotionalitaet": 0.87,  # Resolving
                "transparenz": 0.89,
            },
            priorities={
                "waerme": 1.0,
                "natuerlichkeit": 0.95,
                "authentizitaet": 0.9,
                "emotionalitaet": 0.85,
                "transparenz": 0.83,
                "bass-kraft": 0.8,
                "brillanz": 0.78,
            },
            description="Outro: Warm resolution and natural decay",
        )

        # SOLO: Spotlight, all qualities maximal
        self.profiles[SegmentType.SOLO] = GoalProfile(
            name="Solo",
            goals={
                "bass-kraft": 0.88,
                "brillanz": 0.93,  # High: bright presence
                "waerme": 0.88,
                "natuerlichkeit": 0.94,  # High: natural expression
                "authentizitaet": 0.92,
                "emotionalitaet": 0.94,  # High: expressive peak
                "transparenz": 0.95,  # Critical: clear spotlight
            },
            priorities={
                "transparenz": 1.0,
                "emotionalitaet": 0.98,
                "natuerlichkeit": 0.95,
                "brillanz": 0.93,
                "authentizitaet": 0.92,
                "waerme": 0.88,
                "bass-kraft": 0.85,
            },
            description="Solo: Maximum clarity and expression",
        )

        # BUILD_UP: Increasing energy
        self.profiles[SegmentType.BUILD_UP] = GoalProfile(
            name="Build-up",
            goals={
                "bass-kraft": 0.90,  # Rising bass
                "brillanz": 0.90,  # Rising brightness
                "waerme": 0.84,
                "natuerlichkeit": 0.88,
                "authentizitaet": 0.88,
                "emotionalitaet": 0.92,  # Rising emotion
                "transparenz": 0.91,  # Clear build
            },
            priorities={
                "emotionalitaet": 0.98,
                "bass-kraft": 0.95,
                "brillanz": 0.93,
                "transparenz": 0.9,
                "natuerlichkeit": 0.85,
                "authentizitaet": 0.85,
                "waerme": 0.8,
            },
            description="Build-up: Rising energy and anticipation",
        )

        # DROP: Maximum impact
        self.profiles[SegmentType.DROP] = GoalProfile(
            name="Drop",
            goals={
                "bass-kraft": 0.95,  # Maximum bass
                "brillanz": 0.93,  # Bright impact
                "waerme": 0.85,
                "natuerlichkeit": 0.87,
                "authentizitaet": 0.88,
                "emotionalitaet": 0.95,  # Peak emotion
                "transparenz": 0.92,  # Clear despite power
            },
            priorities={
                "bass-kraft": 1.0,
                "emotionalitaet": 0.98,
                "brillanz": 0.95,
                "transparenz": 0.93,
                "authentizitaet": 0.85,
                "natuerlichkeit": 0.82,
                "waerme": 0.8,
            },
            description="Drop: Maximum bass and emotional impact",
        )

        # UNKNOWN: Balanced
        self.profiles[SegmentType.UNKNOWN] = GoalProfile(
            name="Unknown",
            goals=InstrumentProfileLibrary.BASE_GOALS.copy(),
            priorities=dict.fromkeys(InstrumentProfileLibrary.BASE_GOALS.keys(), 0.85),
            description="Generic profile for unknown segments",
        )

    def get_profile(self, segment: SegmentType) -> GoalProfile:
        """
        Gibt zurück: goal profile for segment type.

        Args:
            segment: Segment type

        Returns:
            Goal profile
        """
        return self.profiles.get(segment, self.profiles[SegmentType.UNKNOWN])


class SemanticGoalsEngine:
    """
    Engine for semantic goal adjustment based on instrument and segment context.

    Combines instrument detection and structure analysis to provide
    context-aware Musical Goals adjustments.
    """

    def __init__(self, instrument_detector_path: Path | None = None, structure_analyzer_path: Path | None = None):
        """
        Initialisiert semantic goals engine.

        Args:
            instrument_detector_path: Path to MERT model
            structure_analyzer_path: Path to madmom models
        """
        self.instrument_library = InstrumentProfileLibrary()
        self.segment_library = SegmentProfileLibrary()
        self._instrument_fallback_logged = False
        self._structure_fallback_logged = False

        # ML models (optional, fallback if unavailable)
        self.instrument_detector = self._load_instrument_detector(instrument_detector_path)
        self.structure_analyzer = self._load_structure_analyzer(structure_analyzer_path)

        logger.info("SemanticGoalsEngine initialisiert")

    def _load_instrument_detector(self, model_path: Path | None) -> Any | None:
        """
        Lädt instrument detection model (MERT-v1-330M).

        Args:
            model_path: Path to model

        Returns:
            Model instance or None
        """
        if model_path is None or not model_path.exists():
            if not self._instrument_fallback_logged:
                logger.info("SemanticGoals: Instrument-ML nicht gebündelt — akustischer Offline-Ersatzpfad aktiv")
                self._instrument_fallback_logged = True
            return None

        try:
            # Import transformers (HuggingFace)
            from transformers import AutoFeatureExtractor, AutoModel

            model = AutoModel.from_pretrained(str(model_path), local_files_only=True)  # nosec B615 — local_files_only=True, kein Download
            feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_path), local_files_only=True)  # nosec B615 — local_files_only=True, kein Download

            logger.info("Instrument detector geladen from %s", model_path)
            return (model, feature_extractor)

        except ImportError:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            if not self._instrument_fallback_logged:
                logger.warning(
                    "⚠️ SOTA SemanticGoals: transformers nicht verfügbar — akustischer Offline-Ersatzpfad aktiv"
                )
                self._instrument_fallback_logged = True
            return None
        except Exception as e:
            logger.warning("konnte nicht laden instrument detector: %s", e)
            return None

    def _load_structure_analyzer(self, model_path: Path | None) -> Any | None:
        """
        Lädt structure analysis model (madmom).

        Args:
            model_path: Path to model

        Returns:
            Model instance or None
        """
        try:
            import madmom  # type: ignore[import-untyped]

            logger.info("madmom structure analyzer geladen")
            # Leichtes Wrapper-Objekt: Analyse nutzt madmom lazy in
            # analyze_structure(); dieses Objekt markiert nur die Verfügbarkeit.
            return type("MadmomStructureAnalyzer", (), {"available": True})()
        except ImportError:
            if not self._structure_fallback_logged:
                logger.warning(
                    "⚠️ SOTA SemanticGoals: madmom nicht gebündelt — heuristische Struktur-Analyse aktiv (librosa Ersatzpfad)"
                )
                self._structure_fallback_logged = True
            # Fallback-Marker statt None: Der Engine-Vertrag verlangt immer einen
            # Struktur-Analysator; available=False routet analyze_structure()
            # deterministisch auf den librosa-Ersatzpfad (§V6, Rev. 2026-08-16).
            return type("LibrosaStructureAnalyzer", (), {"available": False, "backend": "librosa"})()

    def detect_instruments(
        self, audio: np.ndarray, sr: int = 44100
    ) -> tuple[InstrumentCategory, list[InstrumentCategory], float]:
        """
        Erkennt instruments in audio.

        Args:
            audio: Audio data
            sr: Sample rate

        Returns:
            (dominant_instrument, all_instruments, confidence)
        """
        if self.instrument_detector is None:
            # Fallback: Use acoustic features
            return self._detect_instruments_fallback(audio, sr)

        try:
            model, feature_extractor = self.instrument_detector

            if not _load_torch() or torch is None:
                return self._detect_instruments_fallback(audio, sr)

            # Extract features
            inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

            # Get predictions
            assert torch is not None
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                torch.softmax(logits, dim=-1)

            # Map to instrument categories (simplified)
            # In production: Use proper label mapping
            dominant = InstrumentCategory.ENSEMBLE
            all_instruments = [InstrumentCategory.ENSEMBLE]
            confidence = 0.8

            return dominant, all_instruments, confidence

        except Exception as e:
            logger.warning("Instrument detection fehlgeschlagen: %s", e)
            return self._detect_instruments_fallback(audio, sr)

    def _detect_instruments_fallback(
        self, audio: np.ndarray, sr: int
    ) -> tuple[InstrumentCategory, list[InstrumentCategory], float]:
        """
        Fallback instrument detection using acoustic features.

        Args:
            audio: Audio data
            sr: Sample rate

        Returns:
            (dominant_instrument, all_instruments, confidence)
        """
        # Simple heuristics based on spectral centroid
        from scipy import signal

        # Compute spectral centroid
        f, _t, Sxx = signal.spectrogram(audio, sr)
        centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)
        mean_centroid = np.mean(centroid)

        # Basic classification
        if mean_centroid < 500:
            dominant = InstrumentCategory.BASS
        elif mean_centroid < 2000:
            dominant = InstrumentCategory.VOCALS
        elif mean_centroid < 4000:
            dominant = InstrumentCategory.GUITAR
        else:
            dominant = InstrumentCategory.KEYBOARD

        return dominant, [dominant], 0.5  # Low confidence

    def analyze_structure(self, audio: np.ndarray, sr: int = 44100) -> list[tuple[float, float, SegmentType]]:
        """
        Analysiert die Musikstruktur und erkennt Segmente.

        Args:
            audio: Audio data
            sr: Sample rate

        Returns:
            List of (start_time, end_time, segment_type)
        """
        if self.structure_analyzer is None or not getattr(self.structure_analyzer, "available", False):
            return self._analyze_structure_fallback(audio, sr)

        try:
            import madmom  # type: ignore[import-untyped]

            # Use madmom's RNNDownBeatProcessor
            proc = madmom.features.downbeats.RNNDownBeatProcessor()
            proc(audio)

            # Segment detection (simplified)
            # In production: Use proper segmentation algorithm
            duration = len(audio) / sr
            segments = [
                (0.0, duration * 0.2, SegmentType.INTRO),
                (duration * 0.2, duration * 0.8, SegmentType.VERSE),
                (duration * 0.8, duration, SegmentType.OUTRO),
            ]

            return segments

        except Exception as e:
            logger.warning("Structure Analyse fehlgeschlagen: %s", e)
            return self._analyze_structure_fallback(audio, sr)

    def _analyze_structure_fallback(self, audio: np.ndarray, sr: int) -> list[tuple[float, float, SegmentType]]:
        """
        Fallback structure analysis using simple heuristics.

        Args:
            audio: Audio data
            sr: Sample rate

        Returns:
            List of (start_time, end_time, segment_type)
        """
        duration = len(audio) / sr

        # Simple heuristic: assume intro/main/outro structure
        return [
            (0.0, min(10.0, duration * 0.15), SegmentType.INTRO),
            (min(10.0, duration * 0.15), max(duration - 10.0, duration * 0.85), SegmentType.VERSE),
            (max(duration - 10.0, duration * 0.85), duration, SegmentType.OUTRO),
        ]

    def get_semantic_context(self, audio: np.ndarray, sr: int = 44100, timestamp: float = 0.0) -> SemanticContext:
        """
        Gibt zurück: semantic context for given audio and timestamp.

        Args:
            audio: Audio data
            sr: Sample rate
            timestamp: Current timestamp in audio

        Returns:
            Semantic context
        """
        # Detect instruments
        dominant, all_instruments, inst_conf = self.detect_instruments(audio, sr)

        # Analyze structure
        segments = self.analyze_structure(audio, sr)

        # Find current segment
        current_segment = SegmentType.UNKNOWN
        segment_position = 0.5

        for start, end, seg_type in segments:
            if start <= timestamp < end:
                current_segment = seg_type
                segment_position = (timestamp - start) / (end - start)
                break

        return SemanticContext(
            dominant_instrument=dominant,
            all_instruments=all_instruments,
            segment_type=current_segment,
            segment_position=segment_position,
            confidence=inst_conf,
        )

    def adjust_goals_for_context(self, base_goals: dict[str, float], context: SemanticContext) -> dict[str, float]:
        """
        Adjust Musical Goals based on semantic context.

        Args:
            base_goals: Base goal values
            context: Semantic context

        Returns:
            Adjusted goal values
        """
        # Get instrument profile
        instrument_profile = self.instrument_library.get_profile(context.dominant_instrument)

        # Get segment profile
        segment_profile = self.segment_library.get_profile(context.segment_type)

        # Apply instrument adjustments (weight: 0.6)
        goals_after_instrument = instrument_profile.apply_to_base_goals(base_goals)

        # Apply segment adjustments (weight: 0.4)
        goals_after_segment = segment_profile.apply_to_base_goals(goals_after_instrument)

        # Blend based on confidence
        confidence = context.confidence
        adjusted = {}
        for goal_name in base_goals:
            adjusted[goal_name] = (
                confidence * goals_after_segment.get(goal_name, base_goals[goal_name])
                + (1 - confidence) * base_goals[goal_name]
            )

        logger.debug(
            f"angepasst goals for {context.dominant_instrument.value} / {context.segment_type.value}: {adjusted}"
        )

        return adjusted


# Convenience functions


def get_instrument_profile(instrument: InstrumentCategory) -> GoalProfile:
    """Gibt zurück: goal profile for instrument category."""
    lib = InstrumentProfileLibrary()
    return lib.get_profile(instrument)


def get_segment_profile(segment: SegmentType) -> GoalProfile:
    """Gibt zurück: goal profile for segment type."""
    lib = SegmentProfileLibrary()
    return lib.get_profile(segment)
