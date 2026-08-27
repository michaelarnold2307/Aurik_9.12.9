"""Tests für das flache Medium-/Depth-Klassifikator-Artefakt (Empfehlung 9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import train_medium_classifier as tm

ARTIFACT = _ROOT / "models" / "medium_shallow_v1.joblib"


@pytest.fixture(scope="module")
def artifact():
    import joblib

    if not ARTIFACT.exists():
        pytest.skip("Artefakt fehlt — erst scripts/train_medium_classifier.py ausführen")
    return joblib.load(ARTIFACT)


def test_artifact_has_expected_structure(artifact) -> None:
    assert artifact["version"] == 1
    assert set(artifact["material"]) >= {"model", "labels", "cv"}
    assert set(artifact["depth"]) >= {"model", "labels", "cv"}


def test_material_accuracy_beats_detector_baseline(artifact) -> None:
    """CV-Accuracy >= 50 % — der MediumDetector erreicht auf demselben Corpus 10.7 %."""
    cv_acc = float(artifact["material"]["cv"]["accuracy"])
    baseline = float(artifact["baseline_medium_detector"]["material_accuracy"])
    assert cv_acc >= 0.50, f"Material-CV {cv_acc:.2f} unter Schwelle"
    assert cv_acc > 3.0 * baseline, f"{cv_acc:.2f} nicht deutlich über Baseline {baseline:.2f}"


def test_depth_accuracy_beats_detector_baseline(artifact) -> None:
    """CV-Accuracy >= 70 % — der MediumDetector erreicht 51.8 %."""
    cv_acc = float(artifact["depth"]["cv"]["accuracy"])
    baseline = float(artifact["baseline_medium_detector"]["depth_accuracy"])
    assert cv_acc >= 0.70, f"Depth-CV {cv_acc:.2f} unter Schwelle"
    assert cv_acc > 1.3 * baseline, f"{cv_acc:.2f} nicht deutlich über Baseline {baseline:.2f}"


def test_feature_extraction_deterministic() -> None:
    rng = np.random.RandomState(9)
    audio = rng.randn(48000).astype(np.float32)
    f1 = tm.extract_features(audio)
    f2 = tm.extract_features(audio)
    np.testing.assert_array_equal(f1, f2)
    assert f1.dtype == np.float32
    assert np.all(np.isfinite(f1))


def test_predict_returns_valid_labels(artifact) -> None:
    rng = np.random.RandomState(3)
    audio = (0.2 * rng.randn(48000)).astype(np.float32)
    feats = tm.extract_features(audio)
    era_feats = np.concatenate([feats, np.asarray([1965.0, 1960.0], dtype=np.float32)])
    mat_pred = artifact["material"]["model"].predict(era_feats.reshape(1, -1))[0]
    dep_pred = artifact["depth"]["model"].predict(era_feats.reshape(1, -1))[0]
    assert mat_pred in artifact["material"]["labels"]
    assert dep_pred in artifact["depth"]["labels"]
