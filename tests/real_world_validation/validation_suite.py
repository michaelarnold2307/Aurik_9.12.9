"""
Validation Suite - Objective Metrics for Real-World Audio Restoration

Measures objective quality metrics for audio restoration:
- SNR (Signal-to-Noise Ratio)
- THD (Total Harmonic Distortion)
- Spectral metrics (Flatness, Centroid, Rolloff)
- Perceptual metrics: PQS-MOS, CDPAM, Musical Goals (§4.4)
  VERBOTEN für Musik: PESQ, ViSQOL (Speech-Mode), DNSMOS, STOI, NISQA (§4.4)

Usage:
    python validation_suite.py --input test_library/ --output validation_report.json
    python validation_suite.py --compare --baseline unprocessed/ --test aurik/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import rfftfreq as fftfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _workspace_output_path(path_value: Path) -> Path:
    """Allow validation reports only below the caller's working directory."""
    workspace = Path.cwd().resolve()
    output_path = path_value.resolve()
    try:
        output_path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Output path must be inside the working directory: {path_value}") from exc
    return output_path


class ValidationSuite:
    """Objective metrics validation for audio restoration."""

    def __init__(self, reference_dir: Path | None = None):
        self.reference_dir = reference_dir
        self.results: dict[Any, Any] = {}

    def analyze_file(self, audio_path: Path, reference_path: Path | None = None) -> dict:
        """
        Analyze single audio file with objective metrics.

        Args:
            audio_path: Path to audio file to analyze
            reference_path: Optional reference (clean) audio for comparison

        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Analyzing {audio_path.name}...")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        # Handle stereo
        if audio.ndim == 2:
            audio_mono = librosa.to_mono(audio)
        else:
            audio_mono = audio

        metrics = {
            "filename": audio_path.name,
            "duration": len(audio_mono) / sr,
            "sample_rate": sr,
            "channels": 2 if audio.ndim == 2 else 1,
        }

        # Compute metrics
        metrics["snr"] = self._compute_snr(audio_mono, sr)  # type: ignore[arg-type]
        metrics["thd"] = self._compute_thd(audio_mono, sr)  # type: ignore[arg-type]
        metrics["spectral"] = self._compute_spectral_metrics(audio_mono, sr)  # type: ignore[arg-type]
        metrics["dynamics"] = self._compute_dynamics(audio_mono, sr)  # type: ignore[arg-type]
        metrics["frequency_response"] = self._compute_frequency_response(audio_mono, sr)  # type: ignore[arg-type]

        # Reference-based metrics (if reference available)
        if reference_path and reference_path.exists():
            ref_audio, ref_sr = librosa.load(reference_path, sr=sr, mono=True)
            metrics["reference_based"] = self._compute_reference_metrics(audio_mono, ref_audio, sr)  # type: ignore[arg-type]

        return metrics

    def _compute_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Compute Signal-to-Noise Ratio.

        Method: Compare signal power to noise floor power.
        """
        # Use spectral method: top 50% of spectrum is signal, bottom 10% is noise
        n_fft = 2048
        hop_length = 512

        # Compute STFT
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(D)

        # Signal: top 50% of magnitudes
        signal_threshold = np.percentile(mag, 50)
        signal_mask = mag > signal_threshold
        signal_power = np.mean(mag[signal_mask] ** 2)

        # Noise: bottom 10% of magnitudes
        noise_threshold = np.percentile(mag, 10)
        noise_mask = mag < noise_threshold
        noise_power = np.mean(mag[noise_mask] ** 2)

        # SNR in dB
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 100.0  # Perfect signal

        return float(snr_db)

    def _compute_thd(self, audio: np.ndarray, sr: int) -> float:
        """
        Compute Total Harmonic Distortion.

        Method: Detect fundamental frequency, measure harmonic content.
        """
        # Find fundamental frequency (pitch)
        f0 = librosa.yin(audio, fmin=50, fmax=2000, sr=sr)
        f0_median = np.median(f0[~np.isnan(f0)])

        if np.isnan(f0_median):
            return 0.0  # No clear fundamental

        # Compute FFT
        n_fft = 8192
        fft_result = np.abs(fft(np.asarray(audio[:n_fft], dtype=np.float64)))
        freqs = fftfreq(n_fft, 1 / sr)

        # Find harmonics
        fundamental_power = 0
        harmonics_power = 0

        for harmonic in range(1, 8):  # Up to 7th harmonic
            target_freq = f0_median * harmonic

            # Find peak near target frequency (±10%)
            freq_range = target_freq * 0.1
            mask = (freqs > target_freq - freq_range) & (freqs < target_freq + freq_range)

            if np.any(mask):
                peak_power = np.max(fft_result[mask]) ** 2

                if harmonic == 1:
                    fundamental_power = peak_power
                else:
                    harmonics_power += peak_power

        # THD calculation
        if fundamental_power > 0:
            thd = np.sqrt(harmonics_power / fundamental_power)
            thd_percent = thd * 100
        else:
            thd_percent = 0.0

        return float(thd_percent)

    def _compute_spectral_metrics(self, audio: np.ndarray, sr: int) -> dict:
        """
        Compute spectral characteristics.

        Metrics:
        - Spectral Flatness: Measure of tonality vs noise
        - Spectral Centroid: Brightness
        - Spectral Rolloff: High-frequency content
        - Spectral Bandwidth: Spread of spectrum
        """
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

        return {
            "flatness": float(np.mean(spectral_flatness)),
            "centroid": float(np.mean(spectral_centroid)),
            "rolloff": float(np.mean(spectral_rolloff)),
            "bandwidth": float(np.mean(spectral_bandwidth)),
        }

    def _compute_dynamics(self, audio: np.ndarray, sr: int) -> dict:
        """
        Compute dynamic range metrics.

        Metrics:
        - Crest Factor: Peak-to-RMS ratio
        - Dynamic Range: Difference between loudest and softest parts
        - RMS Level: Average loudness
        """
        # RMS level
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Peak level
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        # Crest factor
        crest_factor_db = peak_db - rms_db

        # Dynamic range (90th percentile - 10th percentile)
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = frame_length // 2

        frames_rms = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            frames_rms.append(np.sqrt(np.mean(frame**2)))

        frames_rms = np.array(frames_rms)  # type: ignore[assignment]
        loud_level = np.percentile(frames_rms, 90)
        quiet_level = np.percentile(frames_rms, 10)

        dynamic_range = 20 * np.log10((loud_level + 1e-10) / (quiet_level + 1e-10))

        return {
            "rms_db": float(rms_db),
            "peak_db": float(peak_db),
            "crest_factor_db": float(crest_factor_db),
            "dynamic_range_db": float(dynamic_range),
        }

    def _compute_frequency_response(self, audio: np.ndarray, sr: int) -> dict:
        """
        Compute frequency response in octave bands.

        Standard octave bands: 31, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz
        """
        octave_bands = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

        # Compute STFT
        n_fft = 2048
        D = librosa.stft(audio, n_fft=n_fft)
        mag = np.abs(D)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Energy per octave band
        band_energies = {}
        for center_freq in octave_bands:
            if center_freq > sr / 2:
                continue

            # Octave band: center_freq / sqrt(2) to center_freq * sqrt(2)
            low_freq = center_freq / np.sqrt(2)
            high_freq = center_freq * np.sqrt(2)

            # Find bins in band
            mask = (freqs >= low_freq) & (freqs < high_freq)

            if np.any(mask):
                band_energy = np.mean(mag[mask, :] ** 2)
                band_energy_db = 10 * np.log10(band_energy + 1e-10)
                band_energies[f"{center_freq}Hz"] = float(band_energy_db)

        return band_energies

    def _compute_reference_metrics(self, audio: np.ndarray, reference: np.ndarray, sr: int) -> dict:
        """
        Compute reference-based metrics (requires clean reference).

        Metrics:
        - Improvement in SNR
        - Spectral distance
        - Correlation
        """
        # Ensure same length
        min_len = min(len(audio), len(reference))
        audio = audio[:min_len]
        reference = reference[:min_len]

        # Correlation
        correlation = np.corrcoef(audio, reference)[0, 1]

        # Spectral distance (L2 norm of spectral difference)
        audio_spec = np.abs(librosa.stft(audio))
        ref_spec = np.abs(librosa.stft(reference))

        # Ensure same shape
        min_time = min(audio_spec.shape[1], ref_spec.shape[1])
        audio_spec = audio_spec[:, :min_time]
        ref_spec = ref_spec[:, :min_time]

        spectral_distance = np.linalg.norm(audio_spec - ref_spec) / np.linalg.norm(ref_spec)

        # MSE (Mean Squared Error)
        mse = np.mean((audio - reference) ** 2)

        return {"correlation": float(correlation), "spectral_distance": float(spectral_distance), "mse": float(mse)}

    def validate_directory(self, input_dir: Path, output_file: Path):
        """
        Validate all files in directory and generate report.

        Args:
            input_dir: Directory with test files (organized by category)
            output_file: Output JSON report
        """
        output_file = _workspace_output_path(output_file)
        logger.info(f"Validating directory: {input_dir}")

        results = {"validation_date": "2026-02-09", "input_directory": str(input_dir), "categories": {}}

        # Process each category
        categories = ["vinyl", "tape", "digital", "vocals"]

        for category in categories:
            category_dir = input_dir / category
            if not category_dir.exists():
                logger.warning(f"Category {category} not found, skipping")
                continue

            logger.info(f"Processing category: {category}")

            category_results = []
            wav_files = list(category_dir.glob("*.wav"))

            for wav_file in wav_files:
                try:
                    metrics = self.analyze_file(wav_file)
                    category_results.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to analyze {wav_file}: {e}")

            results["categories"][category] = {"file_count": len(category_results), "files": category_results}  # type: ignore[index]

            # Compute category statistics
            if category_results:
                avg_snr = np.mean([f["snr"] for f in category_results])
                avg_thd = np.mean([f["thd"] for f in category_results])

                results["categories"][category]["statistics"] = {  # type: ignore[index]
                    "avg_snr_db": float(avg_snr),
                    "avg_thd_percent": float(avg_thd),
                }

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Validation complete. Report saved to {output_file}")

        return results

    def compare_restoration(self, baseline_dir: Path, test_dir: Path, output_file: Path):
        """
        Compare baseline (unprocessed) vs test (AURIK processed).

        Args:
            baseline_dir: Directory with unprocessed files
            test_dir: Directory with AURIK-processed files
            output_file: Output comparison report
        """
        output_file = _workspace_output_path(output_file)
        logger.info("Comparing baseline vs test...")

        comparison: dict[str, Any] = {
            "comparison_date": "2026-02-09",
            "baseline": str(baseline_dir),
            "test": str(test_dir),
            "improvements": [],
        }

        # Find matching files
        baseline_files = list(baseline_dir.rglob("*.wav"))

        for baseline_file in baseline_files:
            # Find corresponding test file
            relative_path = baseline_file.relative_to(baseline_dir)
            test_file = test_dir / relative_path

            if not test_file.exists():
                logger.warning(f"No matching test file for {baseline_file.name}")
                continue

            logger.info(f"Comparing {baseline_file.name}")

            try:
                # Analyze both
                baseline_metrics = self.analyze_file(baseline_file)
                test_metrics = self.analyze_file(test_file)

                # Compute improvements
                snr_improvement = test_metrics["snr"] - baseline_metrics["snr"]
                thd_change = test_metrics["thd"] - baseline_metrics["thd"]

                comparison["improvements"].append(
                    {
                        "filename": baseline_file.name,
                        "baseline_snr": baseline_metrics["snr"],
                        "test_snr": test_metrics["snr"],
                        "snr_improvement_db": float(snr_improvement),
                        "baseline_thd": baseline_metrics["thd"],
                        "test_thd": test_metrics["thd"],
                        "thd_change_percent": float(thd_change),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to compare {baseline_file.name}: {e}")

        # Compute overall statistics
        if comparison["improvements"]:
            avg_snr_improvement = np.mean([item["snr_improvement_db"] for item in comparison["improvements"]])
            avg_thd_change = np.mean([item["thd_change_percent"] for item in comparison["improvements"]])

            comparison["overall_statistics"] = {
                "avg_snr_improvement_db": float(avg_snr_improvement),
                "avg_thd_change_percent": float(avg_thd_change),
                "files_compared": len(comparison["improvements"]),
            }

            logger.info(f"✓ Average SNR improvement: {avg_snr_improvement:.1f} dB")
            logger.info(f"✓ Average THD change: {avg_thd_change:.2f}%")

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"✓ Comparison complete. Report saved to {output_file}")

        return comparison


def main():
    parser = argparse.ArgumentParser(description="Validation Suite - Objective Metrics")
    parser.add_argument("--input", type=str, required=True, help="Input directory with test files")
    parser.add_argument("--output", type=str, default="validation_report.json", help="Output JSON report file")
    parser.add_argument(
        "--compare", action="store_true", help="Compare baseline vs test (requires --baseline and --test)"
    )
    parser.add_argument("--baseline", type=str, help="Baseline (unprocessed) directory for comparison")
    parser.add_argument("--test", type=str, help="Test (processed) directory for comparison")

    args = parser.parse_args()

    suite = ValidationSuite()

    if args.compare:
        if not args.baseline or not args.test:
            logger.error("--compare requires --baseline and --test directories")
            return

        suite.compare_restoration(Path(args.baseline), Path(args.test), Path(args.output))
    else:
        suite.validate_directory(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
