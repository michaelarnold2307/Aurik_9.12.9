from typing import Any

import pytest

"""
Test Dataset Creator for Real-World Validation

Creates and manages test audio files for validation across 4 categories:
- Vinyl: Scratches, pops, clicks
- Tape: Dropouts, wow/flutter
- Digital: Clipping, MP3 artifacts
- Vocals: Sibilance, plosives, breaths

Usage:
    python test_dataset_creator.py --mode placeholder --count 10
    python test_dataset_creator.py --category vinyl --source /path/to/files
"""

import argparse
import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("librosa")  # CI-Minimal-Umgebung (cross-platform) ohne librosa
import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _workspace_directory(path_value: str) -> Path:
    """Keep generated test datasets within the caller's working directory."""
    workspace = Path.cwd().resolve()
    library_path = Path(path_value).resolve()
    try:
        library_path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Test library must be inside the working directory: {path_value}") from exc
    return library_path


class DatasetCreator:
    """Creates and validates test audio datasets."""

    CATEGORIES = ["vinyl", "tape", "digital", "vocals"]
    TARGET_SR = 44100
    TARGET_DURATION = 30.0  # seconds

    def __init__(self, library_path: str = "test_library"):
        self.library_path = _workspace_directory(library_path)
        self.metadata: dict[Any, Any] = {}

    def create_placeholder_dataset(self, count_per_category: int = 3):
        """
        Create placeholder test files with synthesized defects.
        Useful for testing infrastructure before acquiring real archival recordings.
        """
        logger.info(f"Creating placeholder dataset ({count_per_category} per category)")

        for category in self.CATEGORIES:
            category_path = self.library_path / category
            category_path.mkdir(parents=True, exist_ok=True)

            for i in range(count_per_category):
                # Generate test signal
                audio, sr = self._generate_test_signal(category, i)

                # Save to file
                filename = f"{category}_test_{i + 1:02d}.wav"
                filepath = category_path / filename
                sf.write(filepath, audio, sr)

                # Store metadata
                self._store_metadata(
                    category,
                    filename,
                    {
                        "type": "placeholder",
                        "defects": self._get_defect_list(category),
                        "duration": len(audio) / sr,
                        "sample_rate": sr,
                    },
                )

                logger.info(f"Created {filepath}")

        # Save metadata
        self._save_metadata()
        logger.info(f"✓ Created {count_per_category * len(self.CATEGORIES)} placeholder files")

    def _generate_test_signal(self, category: str, index: int) -> tuple[np.ndarray, int]:
        """Generate synthetic test signal with category-specific defects."""
        sr = self.TARGET_SR
        duration = self.TARGET_DURATION
        t = np.linspace(0, duration, int(sr * duration))

        # Base signal: Sine wave + harmonics (musical content)
        freq = 440 * (1.5 ** (index % 12))  # Musical scale
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
        signal += 0.2 * np.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic

        # Add category-specific defects
        if category == "vinyl":
            signal = self._add_vinyl_defects(signal, sr)
        elif category == "tape":
            signal = self._add_tape_defects(signal, sr)
        elif category == "digital":
            signal = self._add_digital_defects(signal, sr)
        elif category == "vocals":
            signal = self._add_vocal_characteristics(signal, sr)

        # Normalize to -6dB peak
        signal = signal / np.max(np.abs(signal)) * 0.5

        return signal.astype(np.float32), sr

    def _add_vinyl_defects(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Add vinyl-specific defects: clicks, pops, surface noise."""
        # Surface noise (pink noise)
        noise = np.random.randn(len(signal))
        # Pink noise filter (1/f spectrum)
        from scipy.signal import lfilter

        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink_noise = lfilter(b, a, noise)
        signal += 0.05 * pink_noise

        # Clicks (impulsive noise)
        num_clicks = np.random.randint(10, 30)
        click_positions = np.random.randint(0, len(signal), num_clicks)
        for pos in click_positions:
            if pos < len(signal) - 100:
                # Exponentially decaying click
                click_env = np.exp(-np.arange(100) / 10)
                signal[pos : pos + 100] += 0.3 * click_env * np.random.randn()

        # Rumble (low-frequency noise)
        rumble_freq = 20  # Hz
        t = np.arange(len(signal)) / sr
        rumble = 0.02 * np.sin(2 * np.pi * rumble_freq * t)
        signal += rumble

        return signal

    def _add_tape_defects(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Add tape-specific defects: dropouts, wow/flutter."""
        # Dropouts (brief signal loss)
        num_dropouts = np.random.randint(2, 5)
        for _ in range(num_dropouts):
            start = np.random.randint(0, len(signal) - sr // 10)
            duration = np.random.randint(sr // 100, sr // 20)  # 10-50ms
            signal[start : start + duration] *= 0.1  # 90% amplitude reduction

        # Wow/flutter (pitch modulation)
        t = np.arange(len(signal)) / sr
        wow_freq = 0.5  # Hz (slow pitch drift)
        flutter_freq = 6  # Hz (fast pitch variation)
        pitch_mod = 1.0 + 0.01 * np.sin(2 * np.pi * wow_freq * t) + 0.002 * np.sin(2 * np.pi * flutter_freq * t)

        # Apply pitch modulation (time-stretching approximation)
        # For real implementation, use librosa.effects.pitch_shift
        # Here we approximate with amplitude modulation
        signal *= pitch_mod

        # Tape hiss (high-frequency noise)
        hiss = np.random.randn(len(signal))
        from scipy.signal import butter, filtfilt

        b, a = butter(4, 3000 / (sr / 2), btype="high", output="ba")  # type: ignore[call-overload]
        hiss = filtfilt(b, a, hiss)
        signal += 0.03 * hiss

        return signal

    def _add_digital_defects(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Add digital defects: clipping, quantization noise."""
        # Hard clipping at 0.7
        signal = np.clip(signal, -0.7, 0.7)

        # Quantization noise (simulate 8-bit audio)
        bits = 8
        levels = 2**bits
        signal_quantized = np.round(signal * levels) / levels
        signal = signal_quantized

        # Digital "zipper" noise (buffer underruns)
        num_glitches = np.random.randint(1, 3)
        for _ in range(num_glitches):
            pos = np.random.randint(0, len(signal) - sr // 20)
            duration = np.random.randint(sr // 100, sr // 50)  # 10-20ms
            # Repeat last sample (buffer underrun)
            signal[pos : pos + duration] = signal[pos]

        return signal

    def _add_vocal_characteristics(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Add vocal-specific characteristics: sibilance, plosives, breaths."""
        # Sibilance (6-10 kHz boost)
        from scipy.signal import butter, filtfilt

        b, a = butter(4, [6000 / (sr / 2), 10000 / (sr / 2)], btype="band", output="ba")  # type: ignore[call-overload]
        sibilance_band = filtfilt(b, a, signal)

        # Add sibilance bursts
        num_sibilance = np.random.randint(5, 10)
        for _ in range(num_sibilance):
            pos = np.random.randint(0, len(signal) - sr // 4)
            duration = sr // 10  # 100ms
            envelope = np.hanning(duration)
            signal[pos : pos + duration] += 0.3 * envelope * sibilance_band[pos : pos + duration]

        # Plosives (low-frequency bursts)
        num_plosives = np.random.randint(3, 6)
        for _ in range(num_plosives):
            pos = np.random.randint(0, len(signal) - sr // 10)
            duration = sr // 50  # 20ms
            # Low-frequency impulse
            b, a = butter(4, 200 / (sr / 2), btype="low", output="ba")  # type: ignore[call-overload]
            plosive = filtfilt(b, a, np.random.randn(duration))
            signal[pos : pos + len(plosive)] += 0.5 * plosive

        # Breath noise (band-limited noise)
        num_breaths = np.random.randint(4, 8)
        for _ in range(num_breaths):
            pos = np.random.randint(0, len(signal) - sr)
            duration = sr // 2  # 500ms
            breath = np.random.randn(duration)
            b, a = butter(4, [500 / (sr / 2), 3000 / (sr / 2)], btype="band", output="ba")  # type: ignore[call-overload]
            breath = filtfilt(b, a, breath)
            envelope = np.hanning(duration)
            signal[pos : pos + duration] += 0.1 * envelope * breath

        return signal

    def _get_defect_list(self, category: str) -> list[str]:
        """Get list of defects for category."""
        defect_map = {
            "vinyl": ["surface_noise", "clicks", "pops", "rumble"],
            "tape": ["dropouts", "wow_flutter", "tape_hiss", "azimuth_error"],
            "digital": ["clipping", "quantization_noise", "buffer_underruns"],
            "vocals": ["sibilance", "plosives", "breaths", "resonances"],
        }
        return defect_map.get(category, [])

    def _store_metadata(self, category: str, filename: str, data: dict):
        """Store file metadata."""
        if category not in self.metadata:
            self.metadata[category] = {}
        self.metadata[category][filename] = data

    def _save_metadata(self):
        """Save metadata to JSON."""
        metadata_path = self.library_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    def validate_dataset(self):
        """Validate test dataset integrity."""
        logger.info("Validating test dataset...")

        valid_count = 0
        invalid_files = []

        for category in self.CATEGORIES:
            category_path = self.library_path / category
            if not category_path.exists():
                logger.warning(f"Category {category} directory not found")
                continue

            wav_files = list(category_path.glob("*.wav"))
            logger.info(f"{category}: {len(wav_files)} files")

            for wav_file in wav_files:
                try:
                    audio, sr = librosa.load(wav_file, sr=None)
                    duration = len(audio) / sr

                    # Validation checks
                    if sr != self.TARGET_SR:
                        logger.warning(f"{wav_file.name}: Sample rate {sr} != {self.TARGET_SR}")
                    if duration < 10.0:
                        logger.warning(f"{wav_file.name}: Duration {duration:.1f}s < 10s")

                    valid_count += 1

                except Exception as e:
                    logger.error(f"Failed to load {wav_file}: {e}")
                    invalid_files.append(wav_file.name)

        logger.info(f"✓ Validated {valid_count} files")
        if invalid_files:
            logger.error(f"❌ Invalid files: {invalid_files}")

        return valid_count, invalid_files


def main():
    parser = argparse.ArgumentParser(description="Test Dataset Creator")
    parser.add_argument(
        "--mode",
        choices=["placeholder", "validate"],
        default="placeholder",
        help="Mode: placeholder (create synthetic) or validate (check existing)",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of files per category (placeholder mode)")
    parser.add_argument("--library", type=str, default="test_library", help="Path to test library directory")

    args = parser.parse_args()

    creator = DatasetCreator(library_path=args.library)

    if args.mode == "placeholder":
        creator.create_placeholder_dataset(count_per_category=args.count)
    elif args.mode == "validate":
        creator.validate_dataset()


if __name__ == "__main__":
    main()
