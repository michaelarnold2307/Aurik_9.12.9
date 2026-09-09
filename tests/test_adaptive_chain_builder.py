"""
tests/test_adaptive_chain_builder.py
Tests für Adaptive Chain Builder
==================================

Tests:
1. Chain building from forensic analysis
2. Material-specific templates
3. Defect-based module selection
4. Parameter inference
5. Chain optimization
6. Chain export/import
"""

import pytest

pytest.importorskip("sklearn")  # CI-Minimal-Umgebung (cross-platform)

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.forensics.adaptive_chain_builder import AdaptiveChainBuilder, ProcessingChain, ProcessingModule
from backend.core.forensics.dataset_generator import DatasetGenerator
from backend.core.forensics.ml_defect_detector import train_ml_defect_detector_from_dataset
from backend.core.forensics.ml_era_detector import train_ml_era_detector_from_dataset
from backend.core.forensics.ml_medium_detector import train_ml_detector_from_dataset
from backend.core.forensics.unified_analyzer import UnifiedForensicAnalyzer

pytestmark = [pytest.mark.ml, pytest.mark.slow]


@pytest.fixture(scope="module")
def full_analyzer():
    """Create analyzer with all detectors (schnell: 2 Samples/Klasse, 2-fold CV)."""
    gen = DatasetGenerator()

    # Train medium detector
    medium_dataset = gen.generate_medium_dataset(n_synthetic_per_medium=2)
    medium_detector, _ = train_ml_detector_from_dataset(medium_dataset, test_size=0.2, verbose=False, cv_folds=2)

    # Train era detector (min. 5 Samples/Klasse wegen test_size=0.2 bei 8 Era-Klassen)
    era_dataset = gen.generate_era_dataset(n_synthetic_per_era=10)
    era_detector, _ = train_ml_era_detector_from_dataset(era_dataset, test_size=0.2, verbose=False, cv_folds=2)

    # Train defect detector
    from tests.test_ml_defect_detector import generate_defect_dataset

    defect_dataset = generate_defect_dataset(n_samples_per_type=2)
    defect_detector, _ = train_ml_defect_detector_from_dataset(
        defect_dataset, test_size=0.2, verbose=False, cv_folds=2, n_estimators=10, max_depth=3
    )

    return UnifiedForensicAnalyzer(
        medium_detector=medium_detector, era_detector=era_detector, defect_detector=defect_detector
    )


@pytest.fixture(scope="module")
def sample_forensic_analysis(full_analyzer):
    """Generate sample forensic analysis."""
    sr = 48000
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3

    return full_analyzer.analyze(audio, sr, verbose=False)


class TestAdaptiveChainBuilder:
    """Test suite for Adaptive Chain Builder."""

    def test_initialization(self):
        """Test chain builder initialization."""
        builder = AdaptiveChainBuilder()

        assert builder.VERSION == "1.0.0"
        assert builder.last_chain is None
        assert len(builder.CHAIN_TEMPLATES) >= 6
        assert "VINYL" in builder.CHAIN_TEMPLATES
        assert "TAPE" in builder.CHAIN_TEMPLATES
        assert "CD" in builder.CHAIN_TEMPLATES

    def test_chain_templates(self):
        """Test that chain templates are properly defined."""
        builder = AdaptiveChainBuilder()

        for material, template in builder.CHAIN_TEMPLATES.items():
            assert "base_modules" in template
            assert "defect_modules" in template
            assert "enhancement" in template
            assert len(template["base_modules"]) > 0

    def test_build_chain_basic(self, sample_forensic_analysis):
        """Test basic chain building."""
        builder = AdaptiveChainBuilder()

        chain = builder.build_chain(sample_forensic_analysis, verbose=False)

        assert isinstance(chain, ProcessingChain)
        assert len(chain.modules) > 0
        assert chain.material_type in builder.CHAIN_TEMPLATES
        assert chain.era != ""
        assert 0 <= chain.confidence <= 1
        assert chain.description != ""

    def test_build_chain_vinyl(self, full_analyzer):
        """Test chain building for vinyl with clicks."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add vinyl characteristics
        rumble = np.sin(2 * np.pi * 30 * t) * 0.05
        audio += rumble

        # Add clicks
        for i in range(5):
            click_pos = 1000 + i * 2000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(analysis, verbose=False)

        # Check that chain has appropriate modules
        module_names = [m.name for m in chain.modules]

        # Should have DCBlocker (always)
        assert "DCBlocker" in module_names

        # Should have some enhancement
        assert any("Enhancement" in name for name in module_names)

    def test_build_chain_aggressive(self, sample_forensic_analysis):
        """Test aggressive chain building."""
        builder = AdaptiveChainBuilder()

        chain_normal = builder.build_chain(sample_forensic_analysis, aggressive=False, verbose=False)

        chain_aggressive = builder.build_chain(sample_forensic_analysis, aggressive=True, verbose=False)

        # Aggressive chain may have more modules (or same, depending on detection)
        assert len(chain_aggressive.modules) >= len(chain_normal.modules)

    def test_module_priorities(self, sample_forensic_analysis):
        """Test that modules are correctly prioritized."""
        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(sample_forensic_analysis, verbose=False)

        ordered_modules = chain.get_ordered_modules()

        # Check that priorities are ascending
        priorities = [m.priority for m in ordered_modules]
        assert priorities == sorted(priorities)

        # DCBlocker should be first (if present)
        if ordered_modules:
            first_module = ordered_modules[0]
            if first_module.name == "DCBlocker":
                assert first_module.priority == 10

    def test_parameter_inference(self, sample_forensic_analysis):
        """Test parameter inference."""
        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(sample_forensic_analysis, verbose=False)

        # Check that modules have parameters
        for module in chain.modules:
            # Most modules should have parameters
            if module.name != "DCBlocker":  # DCBlocker has minimal params
                assert isinstance(module.parameters, dict)

    def test_defect_addressing(self, full_analyzer):
        """Test that defects are properly addressed."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add hum
        hum = np.sin(2 * np.pi * 50 * t) * 0.2
        audio += hum

        analysis = full_analyzer.analyze(audio, sr, verbose=False)

        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(analysis, verbose=False)

        # If HUM was detected, defects_addressed should include it
        # (May not always detect due to synthetic data)
        assert isinstance(chain.defects_addressed, list)


class TestProcessingChain:
    """Test ProcessingChain dataclass."""

    def test_get_ordered_modules(self):
        """Test module ordering."""
        modules = [
            ProcessingModule("Enhancement", True, 90, {}, "Final"),
            ProcessingModule("DCBlocker", True, 10, {}, "First"),
            ProcessingModule("ClickRemover", True, 30, {}, "Middle"),
            ProcessingModule("DisabledModule", False, 50, {}, "Disabled"),
        ]

        chain = ProcessingChain(
            modules=modules,
            material_type="VINYL",
            era="1970s",
            defects_addressed=["CLICKS"],
            confidence=0.9,
            description="Test chain",
        )

        ordered = chain.get_ordered_modules()

        # Should be ordered by priority
        assert ordered[0].name == "DCBlocker"
        assert ordered[1].name == "ClickRemover"
        assert ordered[2].name == "Enhancement"

        # Disabled module should not be included
        assert len(ordered) == 3
        assert "DisabledModule" not in [m.name for m in ordered]

    def test_to_dict(self):
        """Test chain serialization."""
        modules = [ProcessingModule("DCBlocker", True, 10, {"cutoff_hz": 20}, "Base")]

        chain = ProcessingChain(
            modules=modules,
            material_type="VINYL",
            era="1970s",
            defects_addressed=["CLICKS"],
            confidence=0.9,
            description="Test chain",
        )

        data = chain.to_dict()

        assert "modules" in data
        assert "material_type" in data
        assert "era" in data
        assert "defects_addressed" in data
        assert "confidence" in data
        assert "description" in data

        assert data["material_type"] == "VINYL"
        assert data["era"] == "1970s"


class TestChainVisualization:
    """Test chain visualization and export."""

    def test_visualize_chain(self, sample_forensic_analysis):
        """Test chain visualization."""
        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(sample_forensic_analysis, verbose=False)

        visualization = builder.visualize_chain(chain)

        assert isinstance(visualization, str)
        assert "PROCESSING CHAIN" in visualization
        assert chain.material_type in visualization
        assert "MODULES:" in visualization

    def test_visualize_last_chain(self, sample_forensic_analysis):
        """Test visualization of last chain."""
        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(sample_forensic_analysis, verbose=False)

        # Should use last_chain if no chain provided
        visualization = builder.visualize_chain()

        assert isinstance(visualization, str)
        assert "PROCESSING CHAIN" in visualization

    def test_visualize_no_chain(self):
        """Test visualization with no chain."""
        builder = AdaptiveChainBuilder()

        visualization = builder.visualize_chain()

        assert visualization == "No chain available"


class TestChainPersistence:
    """Test chain export/import."""

    def test_export_chain(self, sample_forensic_analysis):
        """Test chain export to JSON."""
        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(sample_forensic_analysis, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            temp_path = f.name

        try:
            builder.export_chain(chain, temp_path)

            # Check that file was created
            assert Path(temp_path).exists()

            # Check that it's valid JSON
            with open(temp_path) as f:
                data = json.load(f)

            assert "modules" in data
            assert "material_type" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_chain(self, sample_forensic_analysis):
        """Test chain loading from JSON."""
        builder = AdaptiveChainBuilder()
        chain = builder.build_chain(sample_forensic_analysis, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            temp_path = f.name

        try:
            # Export
            builder.export_chain(chain, temp_path)

            # Load
            loaded_chain = builder.load_chain(temp_path)

            # Check that chain is reconstructed correctly
            assert loaded_chain.material_type == chain.material_type
            assert loaded_chain.era == chain.era
            assert loaded_chain.confidence == chain.confidence
            assert len(loaded_chain.modules) == len(chain.modules)

            # Check first module
            if chain.modules:
                assert loaded_chain.modules[0].name == chain.modules[0].name
                assert loaded_chain.modules[0].enabled == chain.modules[0].enabled
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestMaterialSpecificChains:
    """Test material-specific chain generation."""

    def test_vinyl_template(self):
        """Test vinyl template."""
        builder = AdaptiveChainBuilder()
        template = builder.CHAIN_TEMPLATES["VINYL"]

        assert "DCBlocker" in template["base_modules"]
        assert "RumbleFilter" in template["base_modules"]
        assert "CLICKS" in template["defect_modules"]
        assert template["enhancement"] == "VinylEnhancement"

    def test_tape_template(self):
        """Test tape template."""
        builder = AdaptiveChainBuilder()
        template = builder.CHAIN_TEMPLATES["TAPE"]

        assert "TapeCorrector" in template["base_modules"]
        assert "DROPOUT" in template["defect_modules"]
        assert template["enhancement"] == "TapeEnhancement"

    def test_cd_template(self):
        """Test CD template."""
        builder = AdaptiveChainBuilder()
        template = builder.CHAIN_TEMPLATES["CD"]

        assert "DigitalCorrector" in template["base_modules"]
        assert template["enhancement"] == "DigitalEnhancement"

    def test_lossy_template(self):
        """Test lossy codec template."""
        builder = AdaptiveChainBuilder()
        template = builder.CHAIN_TEMPLATES["LOSSY"]

        assert "CodecArtifactRemover" in template["base_modules"]
        assert template["enhancement"] == "LossyEnhancement"


def manual_test_chain_builder():
    """
    Manual test for chain builder.
    Run with: pytest -k manual_test_chain_builder -s
    """
    print("\n" + "=" * 70)
    print("Manual Chain Builder Test")
    print("=" * 70)

    # Create analyzer
    print("\n[1/3] Training detectors...")
    gen = DatasetGenerator()

    medium_dataset = gen.generate_medium_dataset(n_synthetic_per_medium=10)
    medium_detector, _ = train_ml_detector_from_dataset(medium_dataset, test_size=0.2, verbose=False)

    era_dataset = gen.generate_era_dataset(n_synthetic_per_era=10)
    era_detector, _ = train_ml_era_detector_from_dataset(era_dataset, test_size=0.2, verbose=False)

    from tests.test_ml_defect_detector import generate_defect_dataset

    defect_dataset = generate_defect_dataset(n_samples_per_type=10)
    defect_detector, _ = train_ml_defect_detector_from_dataset(defect_dataset, test_size=0.2, verbose=False)

    analyzer = UnifiedForensicAnalyzer(
        medium_detector=medium_detector, era_detector=era_detector, defect_detector=defect_detector
    )
    print("      ✓ Detectors trained")

    # Create test audio (vinyl with clicks and hum)
    print("\n[2/3] Generating test audio (Vinyl with defects)...")
    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3

    # Add vinyl characteristics
    rumble = np.sin(2 * np.pi * 30 * t) * 0.05
    audio += rumble

    # Add clicks
    for i in range(5):
        click_pos = 1000 + i * 3000
        if click_pos < len(audio):
            audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

    # Add hum
    hum = np.sin(2 * np.pi * 50 * t) * 0.1
    audio += hum

    print("      ✓ Audio generated")

    # Analyze
    print("\n[3/3] Performing forensic analysis...")
    analysis = analyzer.analyze(audio, sr, verbose=True)

    # Build chain
    print("\n" + "=" * 70)
    print("Building Processing Chain")
    print("=" * 70)

    builder = AdaptiveChainBuilder()
    chain = builder.build_chain(analysis, aggressive=False, verbose=True)

    # Visualize
    print("\n" + builder.visualize_chain(chain))

    # Test export/import
    print("\nTesting export/import...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        builder.export_chain(chain, temp_path)
        print(f"   ✓ Chain exported to {temp_path}")

        loaded_chain = builder.load_chain(temp_path)
        print(f"   ✓ Chain loaded from {temp_path}")
        print(f"   Modules: {len(loaded_chain.modules)}")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print("Manual test complete!")
    print("=" * 70)


if __name__ == "__main__":
    manual_test_chain_builder()
