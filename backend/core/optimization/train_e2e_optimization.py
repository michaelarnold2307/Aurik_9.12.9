"""
Training Script für Aurik 10.0.0 End-to-End Optimization

Trainiert die differenzierbaren DSP-Module und optimiert Hyperparameter.

Usage:
    python train_e2e_optimization.py --dataset /path/to/dataset --epochs 100

Autor: Aurik Backend-Team
Version: 8.1
Datum: 14. Februar 2026
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

try:
    import soundfile as sf

    _HAS_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore[assignment]
    _HAS_SOUNDFILE = False

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
import yaml
from torch.utils.data import DataLoader, Dataset

from backend.core.optimization.e2e_optimizer import E2EOptimizationFramework
from backend.core.optimization.hyperparameter_optimizer import MaterialSpecificOptimizer, MultiMaterialOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training_e2e.log")],
)
logger = logging.getLogger(__name__)


class AudioRestorationDataset(Dataset):
    """
    Dataset for audio restoration training.

    Expects pairs of (degraded_audio, clean_reference).
    """

    def __init__(
        self, data_dir: Path, split: str = "train", sr: int = 48000, duration: float = 2.0, normalize: bool = True
    ) -> None:
        """
        Initialisiert dataset.

        Args:
            data_dir: Directory with audio files
            split: 'train', 'val', or 'test'
            sr: Sample rate
            duration: Duration of audio chunks in seconds
            normalize: Whether to normalize audio
        """
        self.data_dir = data_dir / split
        self.sr = sr
        self.duration = duration
        self.normalize = normalize
        self.samples = int(sr * duration)

        # Find all audio pairs
        self.audio_pairs = self._find_audio_pairs()

        logger.info("AudioRestorationDataset(%s): %s pairs", split, len(self.audio_pairs))

    def _find_audio_pairs(self) -> list[tuple[Path, Path]]:
        """Findet Paare aus degradierten und sauberen Audiodateien."""
        pairs: list[tuple[Path, Path]] = []

        degraded_dir = self.data_dir / "degraded"
        clean_dir = self.data_dir / "clean"

        if not degraded_dir.exists() or not clean_dir.exists():
            logger.warning("Dataset directories not found: %s, %s", degraded_dir, clean_dir)
            return pairs

        for degraded_file in degraded_dir.glob("*.wav"):
            clean_file = clean_dir / degraded_file.name
            if clean_file.exists():
                pairs.append((degraded_file, clean_file))

        return pairs

    def __len__(self) -> int:
        return len(self.audio_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gibt zurück: audio pair.

        Returns:
            (degraded_audio, clean_audio) as tensors [1, samples]
        """
        degraded_path, clean_path = self.audio_pairs[idx]

        # Load actual audio files using soundfile (SOTA)
        try:
            if not _HAS_SOUNDFILE or sf is None:
                logger.warning("soundfile nicht verfügbar — Fallback auf synthetische Daten")
                degraded_audio = torch.randn(1, self.samples)
                clean_audio = torch.randn(1, self.samples)
            else:
                # Load real audio at native sample rate, then resample to target SR
                deg_data, deg_sr = sf.read(str(degraded_path), dtype="float32", always_2d=True)
                cln_data, cln_sr = sf.read(str(clean_path), dtype="float32", always_2d=True)

                # Mono: Mittelwert über Kanäle (psychoakustisch korrekt für Training)
                if deg_data.ndim == 2 and deg_data.shape[0] > 1:
                    deg_mono = np.mean(deg_data, axis=0).astype(np.float32)
                else:
                    deg_mono = deg_data.flatten().astype(np.float32)

                if cln_data.ndim == 2 and cln_data.shape[0] > 1:
                    cln_mono = np.mean(cln_data, axis=0).astype(np.float32)
                else:
                    cln_mono = cln_data.flatten().astype(np.float32)

                # Resample to target sample rate (linear interpolation — deterministisch)
                if deg_sr != self.sr:
                    deg_resampled = librosa.resample(deg_mono, orig_sr=deg_sr, target_sr=self.sr)  # type: ignore[no-untyped-call]
                else:
                    deg_resampled = deg_mono

                if cln_sr != self.sr:
                    cln_resampled = librosa.resample(cln_mono, orig_sr=cln_sr, target_sr=self.sr)  # type: ignore[no-untyped-call]
                else:
                    cln_resampled = cln_mono

                # Trim or pad to exact sample count
                if len(deg_resampled) > self.samples:
                    deg_resampled = deg_resampled[: self.samples]
                elif len(deg_resampled) < self.samples:
                    deg_resampled = np.pad(deg_resampled, (0, self.samples - len(deg_resampled)), mode="constant")

                if len(cln_resampled) > self.samples:
                    cln_resampled = cln_resampled[: self.samples]
                elif len(cln_resampled) < self.samples:
                    cln_resampled = np.pad(cln_resampled, (0, self.samples - len(cln_resampled)), mode="constant")

                degraded_audio = torch.from_numpy(deg_resampled).unsqueeze(0)
                clean_audio = torch.from_numpy(cln_resampled).unsqueeze(0)

        except Exception as e:
            logger.warning("Audio-Laden fehlgeschlagen (%s, %s): %s — Fallback auf synthetische Daten", degraded_path, clean_path, e)
            degraded_audio = torch.randn(1, self.samples)
            clean_audio = torch.randn(1, self.samples)

        if self.normalize:
            max_deg = degraded_audio.abs().max() + 1e-8
            max_cln = clean_audio.abs().max() + 1e-8
            degraded_audio = degraded_audio / max_deg
            clean_audio = clean_audio / max_cln

        return degraded_audio, clean_audio


def train_e2e_optimization(
    dataset_path: Path,
    output_path: Path,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cpu",  # CPU-only Policy (Section 9.5) — kein CUDA
) -> None:
    """
    Train end-to-end optimization framework.

    Args:
        dataset_path: Path to training dataset
        output_path: Path to save checkpoints and results
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device for training
    """
    logger.info("=" * 80)
    logger.info("Starte E2E Optimization Training")
    logger.info("=" * 80)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Ausgabe: %s", output_path)
    logger.info("Epochs: %s", epochs)
    logger.info("Batch size: %s", batch_size)
    logger.info("Learning rate: %s", learning_rate)
    logger.info("Device: %s", device)
    logger.info("=" * 80)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = AudioRestorationDataset(dataset_path, split="train")
    val_dataset = AudioRestorationDataset(dataset_path, split="val")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,  # §9.5 CPU-only — pin_memory nur bei CUDA relevant
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    # Initialize framework
    framework = E2EOptimizationFramework(sr=48000, device=device, checkpoint_dir=output_path / "checkpoints")

    framework.setup_optimizer(learning_rate=learning_rate, weight_decay=1e-5)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        logger.info("\n%s", "=" * 80)
        logger.info("Epoch %s/%s", epoch, epochs)
        logger.info("%s", "=" * 80)

        # Train
        train_metrics = framework.train_epoch(train_loader, epoch)

        logger.info("\nTraining metrics:")
        for key, value in train_metrics.items():
            logger.info("  %s: %.4f", key, value)

        # Validate
        val_metrics = framework.validate(val_loader)

        logger.info("\nValidierung metrics:")
        for key, value in val_metrics.items():
            logger.info("  %s: %.4f", key, value)

        # Save checkpoint
        if epoch % 10 == 0 or val_metrics["total_perceptual_loss"] < best_val_loss:
            is_best = val_metrics["total_perceptual_loss"] < best_val_loss
            best_val_loss = min(best_val_loss, val_metrics["total_perceptual_loss"])

            framework.save_checkpoint(epoch, {"train": train_metrics, "val": val_metrics})  # type: ignore[dict-item]

            if is_best:
                logger.info("  *** New best Validierung loss: %.4f ***", best_val_loss)

    # Export optimized parameters
    final_params = framework.export_optimized_parameters()

    params_file = output_path / "optimized_dsp_parameters.yaml"
    with open(params_file, "w", encoding="utf-8") as f:
        yaml.dump(final_params, f, default_flow_style=False)

    logger.info("\noptimiert parameters gespeichert: %s", params_file)
    logger.info("\nTraining abgeschlossen!")


def train_hyperparameter_optimization(  # type: ignore[return]
    dataset_path: Path, output_path: Path, material_type: str, n_trials: int = 100, n_jobs: int = 4
) -> np.ndarray:
    """
    Train hyperparameter optimization for specific material.

    Args:
        dataset_path: Path to evaluation dataset
        output_path: Path to save results
        material_type: Material type to optimize
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
    """
    logger.info("=" * 80)
    logger.info("Starte Hyperparameter Optimization for %s", material_type)
    logger.info("=" * 80)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Ausgabe: %s", output_path)
    logger.info("Trials: %s", n_trials)
    logger.info("Jobs: %s", n_jobs)
    logger.info("=" * 80)

    # Create optimizer
    optimizer = MaterialSpecificOptimizer(
        material_type=material_type, storage_path=output_path / material_type, n_trials=n_trials, n_jobs=n_jobs
    )

    # Load evaluation dataset
    # Placeholder - would load actual audio files
    eval_dataset = [(np.random.randn(48000 * 2), np.random.randn(48000 * 2)) for _ in range(20)]

    # Dummy process function (would use actual Aurik pipeline)
    def process_audio(audio: np.ndarray, config: dict[str, object]) -> np.ndarray:
        # Interface-Vertrag des Optimizers erwartet (audio, config).
        del config
        # Simulate processing
        return audio * 0.9  # type: ignore[no-any-return]

    # Run optimization
    results = optimizer.optimize(evaluation_dataset=eval_dataset, process_function=process_audio)

    logger.info("\nOptimization abgeschlossen!")
    logger.info("Best Wert: %.4f", results["best_score"])
    logger.info("Best params gespeichert to: %s", output_path / material_type)


def train_all_materials(dataset_path: Path, output_path: Path, n_trials: int = 100) -> None:
    """
    Train hyperparameter optimization for all materials.

    Args:
        dataset_path: Path to datasets
        output_path: Path to save results
        n_trials: Number of trials per material
    """
    logger.info("=" * 80)
    logger.info("Starte Multi-Material Optimization")
    logger.info("=" * 80)
    logger.info("Dataset root: %s", dataset_path)

    # Create multi-material optimizer
    optimizer = MultiMaterialOptimizer(storage_path=output_path, n_trials_per_material=n_trials)

    # Prepare datasets for each material
    # Placeholder - would load actual audio files
    datasets = {
        material: [(np.random.randn(48000 * 2), np.random.randn(48000 * 2)) for _ in range(20)]
        for material in ["vinyl", "tape_shellac", "tape_cassette", "tape_reel", "digital", "live", "mp3"]
    }

    # Prepare process functions (would use actual Aurik pipeline)
    process_functions = {material: lambda audio, config: audio * 0.9 for material in datasets}

    # Run optimization for all materials
    optimizer.optimize_all(datasets=datasets, process_functions=process_functions)

    logger.info("\nAll materials optimiert!")
    logger.info("Results gespeichert to: %s", output_path)


def main() -> None:
    """Haupt-training script."""
    parser = argparse.ArgumentParser(description="Train Aurik 10.0.0 End-to-End Optimization")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["e2e", "hyperopt", "all"],
        default="e2e",
        help="Training mode: e2e (differentiable DSP), hyperopt (single material), all (all materials)",
    )

    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset")

    parser.add_argument("--output", type=str, default="optimization_results", help="Path to save results")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (for e2e mode)")

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (for e2e mode)")

    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (for e2e mode)")

    parser.add_argument(
        "--material",
        type=str,
        choices=["vinyl", "tape_shellac", "tape_cassette", "tape_reel", "digital", "live", "mp3"],
        help="Material type (for hyperopt mode)",
    )

    parser.add_argument(
        "--trials", type=int, default=100, help="Number of optimization trials (for hyperopt/all modes)"
    )

    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs (for hyperopt/all modes)")

    args = parser.parse_args()

    def _validate_cli_path_text(raw: str, param_name: str) -> str:
        """Validiert rohe CLI-Pfade gegen Traversal-/Steuerzeichenmuster."""
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{param_name} must be a non-empty path string")
        if "\x00" in raw:
            raise ValueError(f"{param_name} contains forbidden null byte")
        # Erlaubt u.a. Leerzeichen, verhindert aber Steuerzeichen und Pipes.
        if not re.fullmatch(r"[\w\-./\\ :]+", raw):
            raise ValueError(f"{param_name} contains forbidden characters")
        return raw

    def _resolve_dataset_path(raw: str) -> Path:
        raw = _validate_cli_path_text(raw, "--dataset")
        return Path(raw).expanduser().resolve(strict=False)

    def _resolve_output_path(raw: str, workspace_root: Path) -> Path:
        raw = _validate_cli_path_text(raw, "--output")
        if any(part == ".." for part in Path(raw).parts):
            raise ValueError("--output must not contain parent traversal ('..')")

        raw_path = Path(raw).expanduser()
        candidate = (
            raw_path.resolve(strict=False)
            if raw_path.is_absolute()
            else (workspace_root / raw_path).resolve(strict=False)
        )

        # Sicherheitsgrenze: Output bleibt innerhalb des aktuellen Workspace-Roots.
        try:
            candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError("--output must resolve inside the current workspace") from exc

        return candidate

    workspace_root = Path.cwd().resolve(strict=False)
    try:
        dataset_path = _resolve_dataset_path(args.dataset)
        output_path = _resolve_output_path(args.output, workspace_root)
    except ValueError as exc:
        logger.error("Ungültiger Pfadparameter: %s", exc)
        sys.exit(1)

    if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)

    # Run training based on mode
    if args.mode == "e2e":
        train_e2e_optimization(
            dataset_path=dataset_path,
            output_path=output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    elif args.mode == "hyperopt":
        if not args.material:
            logger.error("--material is required for hyperopt Betriebsart")
            sys.exit(1)

        train_hyperparameter_optimization(
            dataset_path=dataset_path,
            output_path=output_path,
            material_type=args.material,
            n_trials=args.trials,
            n_jobs=args.jobs,
        )

    elif args.mode == "all":
        train_all_materials(dataset_path=dataset_path, output_path=output_path, n_trials=args.trials)


if __name__ == "__main__":
    main()
