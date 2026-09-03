"""
forensics/adaptive_chain_builder.py
Adaptive Processing Chain Builder
===================================

Erstellt optimale Processing Chains basierend auf Forensics-Analyse:
- Material-spezifische Templates (Vinyl, Tape, CD, Digital)
- Defekt-basierte Modul-Auswahl
- Automatische Parameter-Konfiguration
- Dynamic Chain Optimization

Features:
- Template-basierte Chain Generation
- Forensics-guided Configuration
- Module Priority Ordering
- Parameter Inference
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.core.forensics.unified_analyzer import UnifiedForensicAnalysis

logger = logging.getLogger(__name__)


@dataclass
class ProcessingModule:
    """
    Single processing module configuration.
    """

    name: str  # Module name (e.g., "DCBlocker", "ClickRemover")
    enabled: bool = True  # Module enabled
    priority: int = 100  # Execution priority (lower = earlier)
    parameters: dict[str, Any] = field(default_factory=dict)  # Module parameters
    reason: str = ""  # Why this module was added


@dataclass
class ProcessingChain:
    """
    Complete processing chain configuration.
    """

    modules: list[ProcessingModule]  # Ordered list of modules
    material_type: str  # VINYL, TAPE, CD, DIGITAL
    era: str  # 1950s-2020s
    defects_addressed: list[str]  # Defect types addressed
    confidence: float  # Chain confidence (0-1)
    description: str  # Human-readable description

    def get_ordered_modules(self) -> list[ProcessingModule]:
        """Gibt zurück: modules sorted by priority."""
        return sorted([m for m in self.modules if m.enabled], key=lambda x: x.priority)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary."""
        return {
            "modules": [
                {
                    "name": m.name,
                    "enabled": m.enabled,
                    "priority": m.priority,
                    "parameters": m.parameters,
                    "reason": m.reason,
                }
                for m in self.modules
            ],
            "material_type": self.material_type,
            "era": self.era,
            "defects_addressed": self.defects_addressed,
            "confidence": self.confidence,
            "description": self.description,
        }


class AdaptiveChainBuilder:
    """
    Adaptive Processing Chain Builder.

    Generates optimized processing chains based on forensic analysis:
    1. Select material-specific template
    2. Add defect-specific modules
    3. Configure module parameters
    4. Optimize chain ordering
    """

    VERSION = "1.0.0"

    # Chain templates per material type
    CHAIN_TEMPLATES = {
        "VINYL": {
            "base_modules": ["DCBlocker", "RumbleFilter"],
            "defect_modules": {"CLICKS": "ClickRemover", "HUM": "HumRemover", "NOISE_BURST": "ImpulseNoiseRemover"},
            "enhancement": "VinylEnhancement",
        },
        "TAPE": {
            "base_modules": ["DCBlocker", "TapeCorrector"],
            "defect_modules": {"DROPOUT": "DropoutCorrector", "HUM": "HumRemover", "DISTORTION": "DistortionReducer"},
            "enhancement": "TapeEnhancement",
        },
        "CASSETTE": {
            "base_modules": ["DCBlocker", "TapeCorrector", "NoiseReducer"],
            "defect_modules": {"DROPOUT": "DropoutCorrector", "HUM": "HumRemover", "NOISE_BURST": "NoiseGate"},
            "enhancement": "CassetteEnhancement",
        },
        "CD": {
            "base_modules": ["DCBlocker", "DigitalCorrector"],
            "defect_modules": {"CLICKS": "ClickRemover", "DISTORTION": "DistortionReducer"},
            "enhancement": "DigitalEnhancement",
        },
        "DIGITAL": {
            "base_modules": ["DCBlocker"],
            "defect_modules": {"DISTORTION": "DistortionReducer", "CLICKS": "ClickRemover"},
            "enhancement": "DigitalEnhancement",
        },
        "LOSSY": {
            "base_modules": ["DCBlocker", "CodecArtifactRemover"],
            "defect_modules": {"DISTORTION": "DistortionReducer", "NOISE_BURST": "DeNoiser"},
            "enhancement": "LossyEnhancement",
        },
    }

    # Module priorities (lower = executed earlier)
    MODULE_PRIORITIES = {
        "DCBlocker": 10,
        "RumbleFilter": 20,
        "HumRemover": 25,
        "ClickRemover": 30,
        "ImpulseNoiseRemover": 35,
        "DropoutCorrector": 40,
        "TapeCorrector": 45,
        "DigitalCorrector": 50,
        "CodecArtifactRemover": 55,
        "DistortionReducer": 60,
        "NoiseReducer": 65,
        "NoiseGate": 70,
        "DeNoiser": 75,
        "VinylEnhancement": 90,
        "TapeEnhancement": 90,
        "CassetteEnhancement": 90,
        "DigitalEnhancement": 90,
        "LossyEnhancement": 90,
        "Enhancement": 95,
    }

    def __init__(self) -> None:
        """Initialisiert chain builder."""
        self.last_chain: ProcessingChain | None = None

    def build_chain(
        self, forensic_analysis: UnifiedForensicAnalysis, aggressive: bool = False, verbose: bool = True
    ) -> ProcessingChain:
        """
        Erstellt adaptive processing chain from forensic analysis.

        Args:
            forensic_analysis: Unified forensic analysis result
            aggressive: Use aggressive processing (more modules, stronger params)
            verbose: Print chain building progress

        Returns:
            ProcessingChain with configured modules
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("   Adaptive Chain Builder")
            logger.info("=" * 60)
            logger.info("   Material: %s", forensic_analysis.medium_type)
            logger.info("   Era: %s", forensic_analysis.era)
            logger.info("   Restoration Priority: %s", forensic_analysis.restoration_priority)

        modules = []
        defects_addressed = []

        # 1. Get material-specific template
        material_type = forensic_analysis.medium_type
        if material_type not in self.CHAIN_TEMPLATES:
            # Fallback to DIGITAL for unknown types - INFO (not WARNING) since fallback is graceful
            logger.debug("§2.46b Unknown medium type %s detected, falling back to DIGITAL template per §3.0", material_type)
            material_type = "DIGITAL"

        template = self.CHAIN_TEMPLATES[material_type]

        # 2. Add base modules
        if verbose:
            logger.info("\n   [1/4] Adding base modules...")

        for module_name in template["base_modules"]:
            module = self._create_module(module_name, forensic_analysis, reason=f"Base module for {material_type}")
            modules.append(module)
            if verbose:
                logger.info("         + %s", module_name)

        # 3. Add defect-specific modules
        if verbose:
            logger.info("\n   [2/4] Adding defect-specific modules...")

        detected_defects = {
            defect: detected for defect, detected in forensic_analysis.defects_detected.items() if detected
        }

        for defect_type, detected in detected_defects.items():
            if defect_type in template["defect_modules"]:
                module_name = template["defect_modules"][defect_type]  # type: ignore[index]

                # Check severity
                severity = forensic_analysis.defect_severities.get(defect_type, "LOW")
                confidence = forensic_analysis.defect_confidences.get(defect_type, 0.0)

                # Only add if confidence is reasonable or aggressive mode
                if confidence > 0.3 or aggressive:
                    module = self._create_module(
                        module_name,
                        forensic_analysis,
                        defect_type=defect_type,
                        severity=severity,
                        aggressive=aggressive,
                        reason=f"Address {defect_type} ({severity} severity)",
                    )
                    modules.append(module)
                    defects_addressed.append(defect_type)

                    if verbose:
                        logger.info("         + %s (for %s, %s)", module_name, defect_type, severity)

        # 4. Add enhancement module
        if verbose:
            logger.info("\n   [3/4] Adding enhancement...")

        enhancement_module = template.get("enhancement", "Enhancement")
        module = self._create_module(enhancement_module, forensic_analysis, reason="Final enhancement")  # type: ignore[arg-type]
        modules.append(module)
        if verbose:
            logger.info("         + %s", enhancement_module)

        # 5. Optimize chain
        if verbose:
            logger.info("\n   [4/4] Optimizing chain...")

        modules = self._optimize_chain(modules, forensic_analysis, aggressive)

        # Build description
        description = self._generate_description(material_type, forensic_analysis, defects_addressed)

        # Create chain
        chain = ProcessingChain(
            modules=modules,
            material_type=material_type,
            era=forensic_analysis.era,
            defects_addressed=defects_addressed,
            confidence=forensic_analysis.overall_confidence,
            description=description,
        )

        if verbose:
            logger.info("\n   Chain erstellt: %s modules", len(chain.modules))
            logger.info("   Confidence: %.1f", chain.confidence)
            logger.info("=" * 60)

        self.last_chain = chain
        return chain

    def _create_module(
        self,
        module_name: str,
        forensic_analysis: UnifiedForensicAnalysis,
        defect_type: str | None = None,
        severity: str = "MEDIUM",
        aggressive: bool = False,
        reason: str = "",
    ) -> ProcessingModule:
        """
        Erstellt and configure processing module.

        Parameters are inferred from forensic analysis.
        """
        # Get priority
        priority = self.MODULE_PRIORITIES.get(module_name, 100)

        # Infer parameters
        parameters = self._infer_parameters(module_name, forensic_analysis, defect_type, severity, aggressive)

        return ProcessingModule(name=module_name, enabled=True, priority=priority, parameters=parameters, reason=reason)

    def _infer_parameters(
        self,
        module_name: str,
        forensic_analysis: UnifiedForensicAnalysis,
        defect_type: str | None,
        severity: str,
        aggressive: bool,
    ) -> dict[str, Any]:
        """
        Infer module parameters from forensic analysis.

        Uses defect severity, era characteristics, and material type.
        """
        params = {}

        # Severity-based strength
        strength_map = {
            "LOW": 0.3 if not aggressive else 0.5,
            "MEDIUM": 0.5 if not aggressive else 0.7,
            "HIGH": 0.7 if not aggressive else 0.9,
        }
        strength = strength_map.get(severity, 0.5)

        # Module-specific parameters
        if module_name == "DCBlocker":
            params = {"cutoff_hz": 20}

        elif module_name == "RumbleFilter":
            # Vinyl: stronger rumble filter for older eras
            if "1950s" in forensic_analysis.era or "1960s" in forensic_analysis.era:
                params = {"cutoff_hz": 40, "slope": 12}
            else:
                params = {"cutoff_hz": 30, "slope": 12}

        elif module_name == "HumRemover":
            # Detect 50Hz vs 60Hz from forensic data
            # (In real implementation, would check hum detection features)
            params = {
                "fundamental_hz": 50,  # Or 60 for North America
                "harmonics": 5,
                "bandwidth_hz": 2,
                "strength": strength,  # type: ignore[dict-item]
            }

        elif module_name == "ClickRemover":
            params = {"sensitivity": strength, "max_click_length_ms": 3.0, "interpolation": "cubic"}  # type: ignore[dict-item]

        elif module_name == "ImpulseNoiseRemover":
            params = {"threshold_db": 15 - (strength * 5), "window_ms": 10}  # type: ignore[dict-item]  # Lower threshold = more aggressive

        elif module_name == "DropoutCorrector":
            params = {"threshold_db": -40, "min_dropout_ms": 5, "interpolation": "linear"}  # type: ignore[dict-item]

        elif module_name == "TapeCorrector":
            # Tape speed correction, azimuth, etc.
            params = {"speed_correction": True, "azimuth_correction": True, "deemphasis": True}

        elif module_name == "DigitalCorrector":
            params = {"error_concealment": True, "jitter_correction": True}

        elif module_name == "CodecArtifactRemover":
            # For lossy codecs (MP3, AAC, etc.)
            params = {
                "pre_echo_reduction": True,
                "quantization_noise_reduction": True,
                "bandwidth_extension": forensic_analysis.era in ["2010s", "2020s"],
            }

        elif module_name == "DistortionReducer":
            params = {"threshold": 0.9, "strength": strength, "harmonic_restoration": True}  # type: ignore[dict-item]

        elif module_name == "NoiseReducer":
            # Cassette noise reduction
            params = {"strength": strength, "preserve_transients": True, "noise_profile": "tape_hiss"}  # type: ignore[dict-item]

        elif module_name == "NoiseGate":
            params = {"threshold_db": -50 + (strength * 10), "attack_ms": 5, "release_ms": 50}  # type: ignore[dict-item]

        elif module_name == "DeNoiser":
            params = {"strength": strength, "algorithm": "spectral_subtraction"}  # type: ignore[dict-item]

        elif "Enhancement" in module_name:
            # Era-specific enhancement
            if forensic_analysis.era in ["1950s", "1960s", "1970s"]:
                # Vintage enhancement
                params = {"brightness": 0.3, "warmth": 0.4, "stereo_enhancement": 0.2, "vintage_character": True}  # type: ignore[dict-item]
            else:
                # Modern enhancement
                params = {"brightness": 0.2, "clarity": 0.3, "stereo_enhancement": 0.3, "modern_character": True}  # type: ignore[dict-item]

        return params

    def _optimize_chain(
        self, modules: list[ProcessingModule], forensic_analysis: UnifiedForensicAnalysis, aggressive: bool
    ) -> list[ProcessingModule]:
        """
        Optimiert processing chain.

        - Remove redundant modules
        - Adjust parameters based on confidence
        - Disable low-confidence modules (unless aggressive)
        """
        # Remove exact duplicates
        seen = set()
        optimized = []

        for module in modules:
            key = (module.name, module.priority)
            if key not in seen:
                seen.add(key)

                # Disable low-confidence modules in non-aggressive mode
                if not aggressive:
                    # Check if module addresses a low-confidence defect
                    for defect in forensic_analysis.defects_detected:
                        if defect in module.reason:
                            confidence = forensic_analysis.defect_confidences.get(defect, 0.0)
                            if confidence < 0.3:
                                module.enabled = False
                                module.reason += " (low confidence, disabled)"

                optimized.append(module)

        return optimized

    def _generate_description(
        self, material_type: str, forensic_analysis: UnifiedForensicAnalysis, defects_addressed: list[str]
    ) -> str:
        """Generiert human-readable chain description."""
        parts = [f"Processing chain for {material_type}"]

        if forensic_analysis.era != "UNKNOWN":
            parts.append(f"from {forensic_analysis.era}")

        if defects_addressed:
            parts.append(f"addressing {', '.join(defects_addressed)}")

        return " ".join(parts)

    def visualize_chain(self, chain: ProcessingChain | None = None) -> str:
        """
        Generiert ASCII visualization of processing chain.

        Args:
            chain: Chain to visualize (uses last_chain if None)

        Returns:
            ASCII art representation
        """
        if chain is None:
            chain = self.last_chain

        if chain is None:
            return "No chain available"

        lines = []
        lines.append("=" * 70)
        lines.append(f"PROCESSING CHAIN: {chain.material_type} ({chain.era})")
        lines.append("=" * 70)
        lines.append(f"Description: {chain.description}")
        lines.append(f"Confidence: {chain.confidence:.1%}")
        lines.append(f"Defects Addressed: {', '.join(chain.defects_addressed) if chain.defects_addressed else 'None'}")
        lines.append("")
        lines.append("MODULES:")
        lines.append("-" * 70)

        ordered_modules = chain.get_ordered_modules()

        for i, module in enumerate(ordered_modules, 1):
            status = "✓" if module.enabled else "❌"
            lines.append(f"  {i}. [{status}] {module.name} (priority: {module.priority})")
            lines.append(f"       Reason: {module.reason}")

            if module.parameters:
                lines.append("       Parameters:")
                for key, value in module.parameters.items():
                    lines.append(f"         - {key}: {value}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def export_chain(self, chain: ProcessingChain, filepath: str) -> None:
        """
        Export chain to JSON file.

        Args:
            chain: Chain to export
            filepath: Output file path
        """
        import json

        data = chain.to_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Chain exported to %s", filepath)

    def load_chain(self, filepath: str) -> ProcessingChain:
        """
        Lädt chain from JSON file.

        Args:
            filepath: Input file path

        Returns:
            ProcessingChain
        """
        import json

        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct modules
        modules = []
        for m in data["modules"]:
            module = ProcessingModule(
                name=m["name"],
                enabled=m["enabled"],
                priority=m["priority"],
                parameters=m["parameters"],
                reason=m["reason"],
            )
            modules.append(module)

        chain = ProcessingChain(
            modules=modules,
            material_type=data["material_type"],
            era=data["era"],
            defects_addressed=data["defects_addressed"],
            confidence=data["confidence"],
            description=data["description"],
        )

        logger.info("Chain geladen from %s", filepath)
        return chain
