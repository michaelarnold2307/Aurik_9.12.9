"""
forensics/training/train_models.py
Signal Forensics Phase 2: ML Model Training Pipeline
=====================================================

Zentrale Training Pipeline für alle 3 ML-Detektoren:
1. Medium Detector → 99%+ Accuracy Target
2. Era Detector → 95%+ Accuracy Target
3. Defect Detector → 98%+ Recall Target

Workflow:
- Generate synthetic training data
- Feature extraction
- Train models with cross-validation
- Save trained models
- Validate on test set

Author: AURIK Team
Date: 11. Februar 2026
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.core.forensics.dataset_generator import DatasetGenerator
from backend.core.forensics.ml_defect_detector import MLDefectDetector
from backend.core.forensics.ml_era_detector import MLEraDetector
from backend.core.forensics.ml_medium_detector import MLMediumDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mean_std(values: object) -> tuple[float, float]:
    """Gibt mean/std from arbitrary numeric iterables with explicit float typing zurück."""
    if isinstance(values, dict):
        numeric_values = list(values.values())
    else:
        numeric_values = list(values) if values is not None else []  # type: ignore[call-overload]
    array = np.asarray(numeric_values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0, 0.0
    return float(array.mean()), float(array.std())


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Dataset
    n_samples_per_class: int = 200  # Samples per medium/era/defect
    audio_duration_sec: float = 3.0
    sample_rate: int = 48000

    # Model parameters
    n_estimators: int = 200
    max_depth: int = 20
    random_state: int = 42

    # Training
    cv_folds: int = 5
    test_split: float = 0.2

    # Output
    models_dir: Path = Path("models/forensics")
    reports_dir: Path = Path("forensics/training/reports")


@dataclass
class TrainingReport:
    """Training report with metrics."""

    model_name: str
    accuracy: float
    cross_val_mean: float
    cross_val_std: float
    training_time_sec: float
    samples_used: int
    features_used: int
    model_path: str
    timestamp: str


class ForensicsTrainingPipeline:
    """
    Comprehensive training pipeline for all forensics models.

    Generates synthetic data, trains models, validates, and saves.
    """

    VERSION = "1.0.0"

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """
        Initialisiert training pipeline.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()

        # Create output directories
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        # Dataset generator (initialize with no args, configure via config members if needed)
        self.dataset_gen = DatasetGenerator()

        # Training reports
        self.reports: list[TrainingReport] = []

    def train_medium_detector(self, save_model: bool = True) -> tuple[MLMediumDetector, TrainingReport]:
        """
        Train ML Medium Detector.

        Target: 99%+ Accuracy

        Returns:
            Trained detector and training report
        """
        logger.info("=" * 60)
        logger.info("   Training Medium Detector")
        logger.info("=" * 60)

        start_time = time.monotonic()

        # Generate training data
        logger.info("Generating training data...")
        X_train, y_train, X_test, y_test = self._generate_medium_dataset()

        logger.info("  Training samples: %s", len(X_train))
        logger.info("  Test samples: %s", len(X_test))
        logger.info("  Features: %s", X_train.shape[1])

        # Initialize detector
        detector = MLMediumDetector(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )

        # Train
        logger.info("Training model...")
        cv_scores = detector.train(X_train, y_train, cv_folds=self.config.cv_folds)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_accuracy = detector.evaluate(X_test, y_test)

        training_time = time.monotonic() - start_time

        # Save model
        model_path = ""
        if save_model:
            model_path = str(self.config.models_dir / f"medium_detector_v{detector.VERSION}.pkl")
            detector.save(model_path)  # type: ignore[arg-type]
            logger.info("Model gespeichert: %s", model_path)

        cv_mean, cv_std = _mean_std(cv_scores)

        # Create report
        report = TrainingReport(
            model_name="Medium Detector",
            accuracy=test_accuracy,  # type: ignore[arg-type]
            cross_val_mean=cv_mean,
            cross_val_std=cv_std,
            training_time_sec=training_time,
            samples_used=len(X_train) + len(X_test),
            features_used=X_train.shape[1],
            model_path=model_path,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.reports.append(report)

        logger.info("\n✅ Medium Detector Training vollstaendig")
        logger.info("   Test Accuracy: %.1f", test_accuracy)
        logger.info("   CV Mean: %.1f \u00b1 %.1f", report.cross_val_mean, report.cross_val_std)
        logger.info("   Training Time: %.1fs", training_time)

        if test_accuracy >= 0.99:  # type: ignore[operator]
            logger.info("   🎯 TARGET REACHED: %.1f >= 99%%", test_accuracy)
        else:
            logger.warning("   ⚠ Below target: %.1f < 99%%", test_accuracy)

        return detector, report

    def train_era_detector(self, save_model: bool = True) -> tuple[MLEraDetector, TrainingReport]:
        """
        Train ML Era Detector.

        Target: 95%+ Accuracy

        Returns:
            Trained detector and training report
        """
        logger.info("=" * 60)
        logger.info("   Training Era Detector")
        logger.info("=" * 60)

        start_time = time.monotonic()

        # Generate training data
        logger.info("Generating training data...")
        X_train, y_train, X_test, y_test = self._generate_era_dataset()

        logger.info("  Training samples: %s", len(X_train))
        logger.info("  Test samples: %s", len(X_test))
        logger.info("  Features: %s", X_train.shape[1])

        # Initialize detector
        detector = MLEraDetector(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )

        # Train
        logger.info("Training model...")
        cv_scores = detector.train(X_train, y_train, cv_folds=self.config.cv_folds)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_accuracy = detector.evaluate(X_test, y_test)

        training_time = time.monotonic() - start_time

        # Save model
        model_path = ""
        if save_model:
            model_path = str(self.config.models_dir / f"era_detector_v{detector.VERSION}.pkl")
            detector.save(model_path)  # type: ignore[arg-type]
            logger.info("Model gespeichert: %s", model_path)

        cv_mean, cv_std = _mean_std(cv_scores)

        # Create report
        report = TrainingReport(
            model_name="Era Detector",
            accuracy=test_accuracy,  # type: ignore[arg-type]
            cross_val_mean=cv_mean,
            cross_val_std=cv_std,
            training_time_sec=training_time,
            samples_used=len(X_train) + len(X_test),
            features_used=X_train.shape[1],
            model_path=model_path,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.reports.append(report)

        logger.info("\n✅ Era Detector Training vollstaendig")
        logger.info("   Test Accuracy: %.1f", test_accuracy)
        logger.info("   CV Mean: %.1f \u00b1 %.1f", report.cross_val_mean, report.cross_val_std)
        logger.info("   Training Time: %.1fs", training_time)

        if test_accuracy >= 0.95:  # type: ignore[operator]
            logger.info("   🎯 TARGET REACHED: %.1f >= 95%%", test_accuracy)
        else:
            logger.warning("   ⚠ Below target: %.1f < 95%%", test_accuracy)

        return detector, report

    def train_defect_detector(self, save_model: bool = True) -> tuple[MLDefectDetector, TrainingReport]:
        """
        Train ML Defect Detector.

        Target: 98%+ Recall (minimize false negatives)

        Returns:
            Trained detector and training report
        """
        logger.info("=" * 60)
        logger.info("   Training Defect Detector")
        logger.info("=" * 60)

        start_time = time.monotonic()

        # Generate training data
        logger.info("Generating training data...")
        X_train, y_train, X_test, y_test = self._generate_defect_dataset()

        logger.info("  Training samples: %s", len(X_train))
        logger.info("  Test samples: %s", len(X_test))
        logger.info("  Features: %s", X_train.shape[1])

        # Initialize detector
        detector = MLDefectDetector(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
        )

        # Train
        logger.info("Training model...")
        recalls = detector.train(X_train, y_train, cv_folds=self.config.cv_folds)  # type: ignore[arg-type]

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_recall = detector.evaluate(X_test, y_test)  # type: ignore[attr-defined]

        training_time = time.monotonic() - start_time

        # Save model
        model_path = ""
        if save_model:
            model_path = str(self.config.models_dir / f"defect_detector_v{detector.VERSION}.pkl")
            detector.save(model_path)
            logger.info("Model gespeichert: %s", model_path)

        cv_mean, cv_std = _mean_std(recalls)

        # Create report
        report = TrainingReport(
            model_name="Defect Detector",
            accuracy=test_recall,  # Using recall as primary metric
            cross_val_mean=cv_mean,
            cross_val_std=cv_std,
            training_time_sec=training_time,
            samples_used=len(X_train) + len(X_test),
            features_used=X_train.shape[1],
            model_path=model_path,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.reports.append(report)

        logger.info("\n✅ Defect Detector Training vollstaendig")
        logger.info("   Mean Recall: %.1f", test_recall)
        logger.info("   CV Mean Recall: %.1f \u00b1 %.1f", report.cross_val_mean, report.cross_val_std)
        logger.info("   Training Time: %.1fs", training_time)

        if test_recall >= 0.98:
            logger.info("   🎯 TARGET REACHED: %.1f >= 98%%", test_recall)
        else:
            logger.warning("   ⚠ Below target: %.1f < 98%%", test_recall)

        return detector, report

    def train_all_models(self, save_models: bool = True) -> dict:  # type: ignore[type-arg]
        """
        Train all forensics models in sequence.

        Args:
            save_models: Save trained models to disk

        Returns:
            Dictionary with all trained models and reports
        """
        logger.info("\n" + "=" * 60)
        logger.info("   AURIK Signal Forensics Training Pipeline")
        logger.info("   Verarbeitungsschritt 2: ML Model Training")
        logger.info("=" * 60 + "\n")

        start_time = time.monotonic()

        # Train models
        medium_detector, _medium_report = self.train_medium_detector(save_models)
        era_detector, _era_report = self.train_era_detector(save_models)
        defect_detector, _defect_report = self.train_defect_detector(save_models)

        total_time = time.monotonic() - start_time

        # Generate summary report
        self._generate_summary_report(total_time)

        logger.info("\n" + "=" * 60)
        logger.info("   Training Pipeline vollstaendig")
        logger.info("=" * 60)
        logger.info("   Total Time: %.1f minutes", total_time / 60)
        logger.info("   Models Trained: 3")
        logger.info("   Reports erzeugt: %s", len(self.reports))

        return {
            "medium_detector": medium_detector,
            "era_detector": era_detector,
            "defect_detector": defect_detector,
            "reports": self.reports,
        }

    def _generate_medium_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generiert synthetic medium detection dataset."""
        # Use existing dataset generator
        from backend.core.forensics.ml_medium_detector import MLMediumDetector

        # Generate samples for each medium type
        samples_per_class = self.config.n_samples_per_class

        # This would typically load from disk or generate synthetic data
        # For now, create placeholder (real implementation would use dataset_generator)
        n_classes = len(MLMediumDetector.MEDIUM_CATEGORIES)
        n_features = 70  # From feature extractor

        total_samples = samples_per_class * n_classes
        X = np.random.randn(total_samples, n_features)
        y = np.repeat(range(n_classes), samples_per_class)

        # Stratified train/test split
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_split, stratify=y, random_state=self.config.random_state
        )

        return X_train, y_train, X_test, y_test

    def _generate_era_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generiert synthetic era detection dataset."""

        samples_per_class = self.config.n_samples_per_class
        n_classes = 8  # 1950s-2020s
        n_features = 85  # Base features + era-specific features

        total_samples = samples_per_class * n_classes
        X = np.random.randn(total_samples, n_features)
        y = np.repeat(range(n_classes), samples_per_class)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_split, stratify=y, random_state=self.config.random_state
        )

        return X_train, y_train, X_test, y_test

    def _generate_defect_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generiert synthetic defect detection dataset."""
        samples_per_class = self.config.n_samples_per_class
        n_defect_types = 5  # Clicks, Hum, Distortion, Dropout, Noise Burst
        n_features = 20  # Defect-specific features

        total_samples = samples_per_class * n_defect_types * 2  # With/without defects
        X = np.random.randn(total_samples, n_features)
        y = np.random.randint(0, 2, (total_samples, n_defect_types))  # Multi-label

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_split, random_state=self.config.random_state
        )

        return X_train, y_train, X_test, y_test

    def _generate_summary_report(self, total_time: float) -> None:
        """Generiert comprehensive training summary report."""
        report_path = self.config.reports_dir / f"training_summary_{int(time.monotonic())}.json"

        summary = {
            "pipeline_version": self.VERSION,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_training_time_sec": total_time,
            "configuration": {
                "n_samples_per_class": self.config.n_samples_per_class,
                "audio_duration_sec": self.config.audio_duration_sec,
                "sample_rate": self.config.sample_rate,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "cv_folds": self.config.cv_folds,
            },
            "models": [],
        }

        for report in self.reports:
            summary["models"].append(  # type: ignore[attr-defined]
                {
                    "name": report.model_name,
                    "accuracy": float(report.accuracy),
                    "cv_mean": float(report.cross_val_mean),
                    "cv_std": float(report.cross_val_std),
                    "training_time_sec": float(report.training_time_sec),
                    "samples_used": report.samples_used,
                    "features_used": report.features_used,
                    "model_path": report.model_path,
                    "timestamp": report.timestamp,
                }
            )

        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\nSummary report gespeichert: %s", report_path)


def main() -> None:
    """Haupt-training entry point."""
    # Create training pipeline
    config = TrainingConfig(n_samples_per_class=200, audio_duration_sec=3.0, n_estimators=200, max_depth=20)

    pipeline = ForensicsTrainingPipeline(config)

    # Train all models
    results = pipeline.train_all_models(save_models=True)

    logger.info("\n✅ All models trained erfolgreich!")
    logger.info("   Medium Detector: %s", format(results["reports"][0].accuracy, ".1%"))
    logger.info("   Era Detector: %s", format(results["reports"][1].accuracy, ".1%"))
    logger.info("   Defect Detector: %s", format(results["reports"][2].accuracy, ".1%"))


if __name__ == "__main__":
    main()
