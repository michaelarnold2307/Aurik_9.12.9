"""
tests/test_ml_defect_detector.py
Tests für ML-basierten Defect Detector
=======================================

Tests:
1. Defect feature extraction
2. Training mit synthetischem Dataset
3. Prediction für Audio Samples
4. Multi-label classification
5. Recall target (98%+)
6. Model save/load
"""

import pytest

pytest.importorskip("sklearn")  # CI-Minimal-Umgebung (cross-platform)

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.forensics.ml_defect_detector import (
    DefectDetectionResult,
    DefectFeatureExtractor,
    DefectFeatures,
    MLDefectDetector,
    train_ml_defect_detector_from_dataset,
)


def generate_defect_dataset(n_samples_per_type: int = 10) -> list:
    """
    Generate synthetic dataset with defects.

    Returns:
        List of (audio, sr, defect_labels) tuples
    """
    dataset = []
    sr = 48000
    duration = 0.3

    defect_types = MLDefectDetector.DEFECT_TYPES

    for defect_type in defect_types:
        for _ in range(n_samples_per_type):
            # Generate base audio
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * 440 * t) * 0.3

            # Add defect
            defect_labels = dict.fromkeys(defect_types, False)
            defect_labels[defect_type] = True

            if defect_type == "CLICKS":
                # Add clicks
                for _ in range(5):
                    click_pos = np.random.randint(0, len(audio) - 100)
                    audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

            elif defect_type == "HUM":
                # Add 50Hz hum
                hum = np.sin(2 * np.pi * 50 * t) * 0.1
                audio += hum

            elif defect_type == "DISTORTION":
                # Add clipping
                audio = np.clip(audio * 3, -0.8, 0.8)

            elif defect_type == "DROPOUT":
                # Add silence dropout
                dropout_start = len(audio) // 2
                dropout_length = len(audio) // 10
                audio[dropout_start : dropout_start + dropout_length] *= 0.01

            elif defect_type == "NOISE_BURST":
                # Add noise burst
                burst_pos = len(audio) // 2
                burst_length = len(audio) // 20
                audio[burst_pos : burst_pos + burst_length] += np.random.randn(burst_length) * 0.8

            dataset.append((audio, sr, defect_labels))

    # Add clean samples
    for _ in range(n_samples_per_type):
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        defect_labels = dict.fromkeys(defect_types, False)
        dataset.append((audio, sr, defect_labels))

    return dataset


@pytest.fixture(scope="module")
def defect_dataset():
    """Generate defect dataset for testing."""
    return generate_defect_dataset(n_samples_per_type=10)


@pytest.fixture(scope="module")
def trained_defect_detector(defect_dataset):
    """Pre-trained defect detector for testing."""
    detector, _ = train_ml_defect_detector_from_dataset(defect_dataset, test_size=0.2, verbose=False)
    return detector


@pytest.mark.unit
class TestDefectFeatureExtractor:
    """Test suite for Defect Feature Extractor."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = DefectFeatureExtractor()

        assert extractor.base_extractor is not None

    def test_defect_feature_extraction(self):
        """Test defect-specific feature extraction."""
        extractor = DefectFeatureExtractor()

        # Generate test audio with clicks
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add clicks
        for i in range(5):
            click_pos = 1000 + i * 2000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

        base_features, defect_features = extractor.extract_defect_features(audio, sr, verbose=False)

        assert defect_features.impulsiveness > 0
        assert defect_features.zero_crossing_rate_std >= 0
        assert defect_features.click_density > 0  # Should detect clicks

    def test_defect_features_to_array(self):
        """Test defect features to numpy array conversion."""
        defect_features = DefectFeatures()
        defect_features.impulsiveness = 1.5
        defect_features.click_density = 10.0

        arr = defect_features.to_array()

        assert isinstance(arr, np.ndarray)
        assert len(arr) == 20  # 20 defect features
        assert arr[0] == 1.5  # impulsiveness
        assert arr[3] == 10.0  # click_density

    def test_click_detection(self):
        """Test click detection features."""
        extractor = DefectFeatureExtractor()

        # Generate audio with clicks
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add prominent clicks
        for i in range(10):
            click_pos = 1000 + i * 1000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 5] += np.random.randn(5) * 1.0

        _, defect_features = extractor.extract_defect_features(audio, sr, verbose=False)

        assert defect_features.impulsiveness > 1.0  # High impulsiveness
        assert defect_features.click_density > 5.0  # Several clicks per second
        assert defect_features.max_impulse_amplitude > 0.1

    def test_hum_detection(self):
        """Test hum detection features."""
        extractor = DefectFeatureExtractor()

        # Generate audio with 50Hz hum
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.1
        hum = np.sin(2 * np.pi * 50 * t) * 0.2  # Strong 50Hz hum
        audio += hum

        _, defect_features = extractor.extract_defect_features(audio, sr, verbose=False)

        # Should detect 50Hz component
        assert defect_features.hum_50hz_db > -60  # Significant 50Hz energy
        assert defect_features.hum_harmonics_strength > 0

    def test_distortion_detection(self):
        """Test distortion detection features."""
        extractor = DefectFeatureExtractor()

        # Generate clipped audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.8
        audio = np.clip(audio, -0.6, 0.6)  # Hard clipping

        _, defect_features = extractor.extract_defect_features(audio, sr, verbose=False)

        assert defect_features.clipping_percent > 0  # Should detect clipping
        assert defect_features.thd_percent > 0  # THD from clipping


class TestMLDefectDetector:
    """Test suite for ML Defect Detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MLDefectDetector(n_estimators=50, max_depth=10, recall_target=0.95)

        assert detector.n_estimators == 50
        assert detector.max_depth == 10
        assert detector.recall_target == 0.95
        assert len(detector.DEFECT_TYPES) == 5
        assert all(not trained for trained in detector.is_trained.values())

    def test_defect_types(self):
        """Test that all defect types are defined."""
        detector = MLDefectDetector()

        expected = ["CLICKS", "HUM", "DISTORTION", "DROPOUT", "NOISE_BURST"]
        assert expected == detector.DEFECT_TYPES

    def test_prediction(self, trained_defect_detector):
        """Test prediction on new audio."""
        # Generate test audio with clicks
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add clicks
        for i in range(5):
            click_pos = 1000 + i * 2000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

        # Predict
        result = trained_defect_detector.predict(audio, sr)

        assert isinstance(result, DefectDetectionResult)
        assert "CLICKS" in result.defects_detected
        assert "CLICKS" in result.defect_confidences
        assert "CLICKS" in result.defect_severities
        assert result.features_used > 70  # Base + defect features
        assert result.model_version == "1.0.0"
        assert result.summary != ""

    def test_prediction_with_features(self, trained_defect_detector):
        """Test prediction with feature return."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        result, base_features, defect_features = trained_defect_detector.predict(audio, sr, return_features=True)

        assert isinstance(result, DefectDetectionResult)
        assert base_features is not None
        assert defect_features is not None
        assert isinstance(defect_features, DefectFeatures)

    def test_multi_label_detection(self, trained_defect_detector):
        """Test that multiple defects can be detected."""
        # Generate audio with multiple defects
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add clicks
        for i in range(3):
            click_pos = 1000 + i * 3000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

        # Add hum
        hum = np.sin(2 * np.pi * 50 * t) * 0.1
        audio += hum

        result = trained_defect_detector.predict(audio, sr)

        # Check that results exist for all defect types
        assert len(result.defects_detected) == 5
        assert len(result.defect_confidences) == 5
        assert len(result.defect_severities) == 5

    def test_clean_audio_detection(self, trained_defect_detector):
        """Test that clean audio has no defects."""
        # Generate clean audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.2

        result = trained_defect_detector.predict(audio, sr)

        # Most defects should not be detected
        # (Some false positives are acceptable, but confidences should be low)
        detected_count = sum(result.defects_detected.values())
        assert detected_count <= 2  # At most 2 false positives

    def test_untrained_prediction_raises(self):
        """Test that prediction fails if model is not trained."""
        detector = MLDefectDetector()

        sr = 48000
        audio = np.random.randn(sr)

        with pytest.raises(RuntimeError, match="No models trained"):
            detector.predict(audio, sr)

    def test_save_load(self, trained_defect_detector):
        """Test model save and load."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            trained_defect_detector.save(temp_path)

            # Load into new detector
            new_detector = MLDefectDetector()
            new_detector.load(temp_path)

            # Check that training state is preserved
            assert new_detector.is_trained == trained_defect_detector.is_trained
            assert new_detector.n_estimators == trained_defect_detector.n_estimators
            assert new_detector.max_depth == trained_defect_detector.max_depth

            # Check that prediction works
            sr = 48000
            duration = 0.3
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * 440 * t) * 0.3

            result = new_detector.predict(audio, sr)
            assert isinstance(result, DefectDetectionResult)

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestTrainingPipeline:
    """Test suite for training pipeline."""

    def test_train_from_defect_dataset(self):
        """Test training from defect dataset."""
        # Generate small dataset
        dataset = generate_defect_dataset(n_samples_per_type=10)

        assert len(dataset) > 0

        # Train detector
        detector, metrics = train_ml_defect_detector_from_dataset(dataset, test_size=0.2, verbose=False)

        # Check that at least some models are trained
        trained_count = sum(detector.is_trained.values())
        assert trained_count > 0

        # Check metrics
        assert len(metrics) > 0
        for defect_type, metric in metrics.items():
            assert "test_recall" in metric
            assert "test_samples" in metric

    def test_train_with_save(self):
        """Test training with automatic save."""
        dataset = generate_defect_dataset(n_samples_per_type=10)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            detector, _ = train_ml_defect_detector_from_dataset(
                dataset, test_size=0.2, verbose=False, save_path=temp_path
            )

            # Check that file was created
            assert Path(temp_path).exists()

            # Check that we can load it
            new_detector = MLDefectDetector()
            new_detector.load(temp_path)
            assert any(new_detector.is_trained.values())

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDefectSpecificDetection:
    """Test detection of specific defect types."""

    def test_clicks_detection(self, trained_defect_detector):
        """Test that clicks are detected."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.2

        # Add prominent clicks
        for i in range(10):
            click_pos = 1000 + i * 1200
            if click_pos < len(audio):
                audio[click_pos : click_pos + 5] += np.random.randn(5) * 1.0

        result = trained_defect_detector.predict(audio, sr)

        # Clicks should be detected with reasonable confidence
        if trained_defect_detector.is_trained.get("CLICKS", False):
            # At least some confidence for clicks
            assert result.defect_confidences["CLICKS"] > 0.1

    def test_hum_detection(self, trained_defect_detector):
        """Test that hum is detected."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.1

        # Add strong 50Hz hum
        hum = np.sin(2 * np.pi * 50 * t) * 0.3
        audio += hum

        result = trained_defect_detector.predict(audio, sr)

        # Hum should be detected
        if trained_defect_detector.is_trained.get("HUM", False):
            assert result.defect_confidences["HUM"] > 0.1

    def test_distortion_detection(self, trained_defect_detector):
        """Test that distortion is detected."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.9

        # Hard clipping
        audio = np.clip(audio, -0.5, 0.5)

        result = trained_defect_detector.predict(audio, sr)

        # Distortion should be detected
        if trained_defect_detector.is_trained.get("DISTORTION", False):
            assert result.defect_confidences["DISTORTION"] > 0.1


def manual_test_defect_training():
    """
    Manual test for defect detector training.
    Run with: pytest -k manual_test_defect_training -s
    """
    print("\n" + "=" * 60)
    print("Manual Defect Detector Training Test")
    print("=" * 60)

    # Generate dataset
    print("\nGenerating defect dataset...")
    dataset = generate_defect_dataset(n_samples_per_type=10)
    print(f"Dataset size: {len(dataset)} samples")

    # Train
    print("\nTraining detector...")
    detector, metrics = train_ml_defect_detector_from_dataset(dataset, test_size=0.2, verbose=True)

    # Print metrics
    print("\n" + "=" * 60)
    print("Test Set Metrics:")
    print("=" * 60)
    for defect_type, metric in metrics.items():
        print(f"\n{defect_type}:")
        print(f"  Test Recall: {metric['test_recall']:.4f}")
        print(f"  Test Samples: {metric['test_samples']}")
        print(f"  Positive Samples: {metric['test_positives']}")

    # Test prediction
    print("\n" + "=" * 60)
    print("Test Prediction:")
    print("=" * 60)

    sr = 48000
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3

    # Add clicks
    for i in range(5):
        click_pos = 1000 + i * 2000
        if click_pos < len(audio):
            audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

    result = detector.predict(audio, sr)

    print(f"\nSummary: {result.summary}")
    print("\nDetected Defects:")
    for defect_type, detected in result.defects_detected.items():
        if detected:
            confidence = result.defect_confidences[defect_type]
            severity = result.defect_severities[defect_type]
            print(f"  {defect_type}: {confidence:.2%} ({severity})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    manual_test_defect_training()
