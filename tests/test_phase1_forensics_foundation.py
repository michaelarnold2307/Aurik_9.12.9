import pytest

# §Fix 2026-09-08: Diese Datei läuft Modul-Level-Code (Script-Stil) —
# ohne librosa (CI-Minimal-Umgebung, cross-platform) schlug der
# FeatureExtractor fehl und sys.exit(1) brach die Collection ab.
pytest.importorskip("librosa")

"""
Quick validation test für Phase 1 Signal Forensics Foundation
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🧪 Testing Signal Forensics Phase 1 Foundation...")
print()

# Test 1: Era Signatures
print("1️⃣ Testing Era Signatures...")
try:
    from backend.core.forensics.signatures import ERA_SIGNATURES, EraType

    assert len(ERA_SIGNATURES) == 8, f"Expected 8 eras, got {len(ERA_SIGNATURES)}"

    # Test 1980s signature
    era_1980s = ERA_SIGNATURES[EraType.ERA_1980s]
    assert era_1980s.year_range == (1980, 1989)
    assert era_1980s.peak_limiting == True
    assert era_1980s.brick_wall_limiting == False

    print(f"   ✅ Loaded {len(ERA_SIGNATURES)} era signatures")
    print(f"   ✅ 1980s: {era_1980s.year_range}, typical media: {len(era_1980s.typical_media)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print()

# Test 2: Dataset Generator
print("2️⃣ Testing Dataset Generator...")
try:
    from backend.core.forensics.dataset_generator import DatasetGenerator, SyntheticSample

    gen = DatasetGenerator()

    # Generate small test dataset
    dataset = gen.generate_medium_dataset(n_synthetic_per_medium=5, real_samples_only=False)

    assert "samples" in dataset
    assert "n_synthetic" in dataset
    assert "medium_distribution" in dataset

    n_samples = len(dataset["samples"])
    assert n_samples >= 20, f"Expected >= 20 samples, got {n_samples}"

    # Check sample structure
    sample = dataset["samples"][0]
    assert isinstance(sample, SyntheticSample)
    assert hasattr(sample, "audio")
    assert hasattr(sample, "sample_rate")
    assert hasattr(sample, "medium_type")

    print(f"   ✅ Generated {n_samples} samples")
    print(f"   ✅ Distribution: {dataset['medium_distribution']}")
    print(f"   ✅ Real: {dataset['n_real']}, Synthetic: {dataset['n_synthetic']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Feature Extractor
print("3️⃣ Testing Feature Extractor...")
try:
    from backend.core.forensics.feature_extractor import AudioFeatures, FeatureExtractor

    extractor = FeatureExtractor()

    # Generate test audio (1 second sine wave)
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # A4

    # Extract features
    features = extractor.extract_all(audio, sr, verbose=False)

    assert isinstance(features, AudioFeatures)

    # Check feature values
    assert features.spectral_centroid_mean > 0
    assert features.rms_energy_mean > 0
    assert features.bandwidth_3db_low > 0
    assert features.bandwidth_3db_high > 0

    # Convert to array
    feature_array = features.to_array()
    n_features = len(feature_array)

    assert n_features >= 50, f"Expected >= 50 features, got {n_features}"

    print(f"   ✅ Extracted {n_features} features")
    print(f"   ✅ Spectral Centroid: {features.spectral_centroid_mean:.1f} Hz")
    print(f"   ✅ Bandwidth: {features.bandwidth_3db_low:.0f} - {features.bandwidth_3db_high:.0f} Hz")
    print(f"   ✅ RMS Energy: {features.rms_energy_mean:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Integration Test
print("4️⃣ Testing Integration (Generator + Extractor)...")
try:
    # Generate synthetic sample
    gen = DatasetGenerator()
    dataset = gen.generate_medium_dataset(n_synthetic_per_medium=2, real_samples_only=False)

    # Extract features from first 3 samples
    extractor = FeatureExtractor()
    features_list = []

    for sample in dataset["samples"][:3]:
        features = extractor.extract_all(sample.audio, sample.sample_rate, verbose=False)
        features_list.append(features)

    # Convert to feature matrix
    feature_matrix = extractor.features_to_matrix(features_list)

    assert feature_matrix.shape[0] == 3, f"Expected 3 rows, got {feature_matrix.shape[0]}"
    assert feature_matrix.shape[1] >= 50, f"Expected >= 50 columns, got {feature_matrix.shape[1]}"

    print(f"   ✅ Feature Matrix shape: {feature_matrix.shape}")
    print("   ✅ Integration working: Generator → Extractor → ML-ready matrix")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("🎉 Phase 1 Foundation: ALL TESTS PASSED!")
print("=" * 60)
print()
print("📊 Summary:")
print(f"  - Era Signatures: {len(ERA_SIGNATURES)} eras defined")
print("  - Dataset Generator: Synthetic + real sample support")
print(f"  - Feature Extractor: {n_features} features extracted")
print("  - Integration: Ready for ML Model Training")
print()
print("✅ Phase 1 (Foundation) is COMPLETE and validated!")
print()
print("🚀 Next: Phase 2 - ML Medium Detector (3-4 days)")
