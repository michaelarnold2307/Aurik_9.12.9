"""
ProcessingLogger System - Systematisches Logging aller Processing-Steps

Version: 1.0 (8. Februar 2026)
Purpose: Foundation für Selbstoptimierung, A/B Testing, Regression Detection

Features:
- Audio-Snapshot Logging (Before/After jeder Phase)
- Quality-Metrics (SNR, THD, LUFS, Spectral Centroid)
- Processing-Trace (JSON + Markdown)
- A/B Testing Infrastructure
- Regression-Detection
- Adaptive Learning Foundation

Use Cases:
- A/B Testing zwischen verschiedenen Algorithmen
- Parameter-Tuning basierend auf Quality-Metriken
- Regression-Detection bei Code-Updates
- Kontinuierliche Verbesserung
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Quality metrics computation
try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


@dataclass
class QualityMetrics:
    """Quality metrics for a single processing step."""

    snr_db: float  # Signal-to-Noise Ratio in dB
    thd_percent: float  # Total Harmonic Distortion in %
    lufs: float  # Loudness Units Full Scale
    spectral_centroid_hz: float  # Spectral center of mass
    peak_db: float  # Peak level in dB
    rms_db: float  # RMS level in dB
    dynamic_range_db: float  # Dynamic range (peak - RMS)

    def to_dict(self) -> dict[str, float]:
        """Konvertiert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProcessingStep:
    """Single processing step in the pipeline."""

    step_id: str  # Unique identifier (e.g., "phase_1a_declipping")
    phase: str  # Phase name (e.g., "Phase 1A: Declipping")
    module_name: str  # Module used (e.g., "apollo_declipping")

    # Quality metrics
    metrics_before: QualityMetrics
    metrics_after: QualityMetrics

    # Timings
    processing_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Audio snapshots (file paths)
    audio_before_path: str | None = None
    audio_after_path: str | None = None

    # Additional context
    parameters: dict[str, Any] = field(default_factory=dict)

    def improvement_snr_db(self) -> float:
        """Calculate SNR improvement."""
        return self.metrics_after.snr_db - self.metrics_before.snr_db

    def improvement_thd_percent(self) -> float:
        """Calculate THD reduction (negative = improvement)."""
        return self.metrics_after.thd_percent - self.metrics_before.thd_percent

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary for JSON serialization."""
        return {
            "step_id": self.step_id,
            "phase": self.phase,
            "module_name": self.module_name,
            "metrics_before": self.metrics_before.to_dict(),
            "metrics_after": self.metrics_after.to_dict(),
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "audio_before_path": self.audio_before_path,
            "audio_after_path": self.audio_after_path,
            "parameters": self.parameters,
            "improvements": {"snr_db": self.improvement_snr_db(), "thd_percent": self.improvement_thd_percent()},
        }


@dataclass
class ProcessingTrace:
    """Complete processing trace for one audio file."""

    session_id: str  # Unique session identifier
    input_file: str  # Input audio file path
    output_file: str | None = None  # Output audio file path (set at end_session)

    # Processing steps
    steps: list[ProcessingStep] = field(default_factory=list)

    # Overall metrics
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None
    total_processing_time_sec: float = 0.0

    # Configuration
    processing_mode: str = "restoration"
    sample_rate: int = 44100

    def overall_snr_improvement(self) -> float:
        """Calculate overall SNR improvement across all steps."""
        if not self.steps:
            return 0.0
        first_step = self.steps[0]
        last_step = self.steps[-1]
        return last_step.metrics_after.snr_db - first_step.metrics_before.snr_db

    def overall_thd_reduction(self) -> float:
        """Calculate overall THD reduction."""
        if not self.steps:
            return 0.0
        first_step = self.steps[0]
        last_step = self.steps[-1]
        return first_step.metrics_before.thd_percent - last_step.metrics_after.thd_percent

    def average_processing_time_per_step(self) -> float:
        """Calculate average processing time per step."""
        if not self.steps:
            return 0.0
        return sum(s.processing_time_ms for s in self.steps) / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_processing_time_sec": self.total_processing_time_sec,
            "processing_mode": self.processing_mode,
            "sample_rate": self.sample_rate,
            "steps": [step.to_dict() for step in self.steps],
            "overall_metrics": {
                "snr_improvement_db": self.overall_snr_improvement(),
                "thd_reduction_percent": self.overall_thd_reduction(),
                "average_time_per_step_ms": self.average_processing_time_per_step(),
                "total_steps": len(self.steps),
            },
        }

    def to_markdown(self) -> str:
        """Generiert Markdown summary report."""
        md = "# Processing Trace Report\n\n"
        md += f"**Session ID:** {self.session_id}\n"
        md += f"**Input File:** {self.input_file}\n"
        md += f"**Processing Mode:** {self.processing_mode}\n"
        md += f"**Sample Rate:** {self.sample_rate} Hz\n"
        md += f"**Total Steps:** {len(self.steps)}\n"
        md += f"**Total Time:** {self.total_processing_time_sec:.2f}s\n\n"

        md += "## Overall Improvements\n\n"
        md += f"- **SNR Improvement:** {self.overall_snr_improvement():+.2f} dB\n"
        md += f"- **THD Reduction:** {self.overall_thd_reduction():+.2f}%\n"
        md += f"- **Avg Time/Step:** {self.average_processing_time_per_step():.1f} ms\n\n"

        md += "## Processing Steps\n\n"
        md += "| Step | Phase | Module | SNR Δ | THD Δ | Time |\n"
        md += "|------|-------|--------|-------|-------|------|\n"

        for step in self.steps:
            snr_delta = step.improvement_snr_db()
            thd_delta = step.improvement_thd_percent()
            md += f"| {step.step_id} | {step.phase} | {step.module_name} | "
            md += f"{snr_delta:+.1f} dB | {thd_delta:+.2f}% | {step.processing_time_ms:.0f} ms |\n"

        return md


class ProcessingLogger:
    """
    Systematic logging of all processing steps for:
    - A/B Testing
    - Parameter Tuning
    - Regression Detection
    - Adaptive Learning
    """

    def __init__(
        self,
        session_id: str | None = None,
        output_dir: Path | None = None,
        save_audio_snapshots: bool = True,
        compress_audio: bool = False,
        save_json: bool = True,
        save_markdown: bool = True,
    ):
        """
        Initialisiert ProcessingLogger.

        Args:
            session_id: Unique session identifier (auto-generated if None)
            output_dir: Directory for log files (default: ./logs/processing/)
            save_audio_snapshots: Save before/after audio for each step
            compress_audio: Use FLAC compression (vs WAV)
            save_json: Save JSON trace file
            save_markdown: Save Markdown report
        """
        self.session_id = session_id or f"session_{int(time.time() * 1000)}"
        self.output_dir = output_dir or Path("./logs/processing") / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_audio_snapshots = save_audio_snapshots
        self.compress_audio = compress_audio
        self.save_json = save_json
        self.save_markdown = save_markdown

        # Current trace
        self.trace: ProcessingTrace | None = None

    def start_session(self, input_file: str, processing_mode: str = "restoration", sample_rate: int = 44100):
        """Start a new processing session."""
        self.trace = ProcessingTrace(
            session_id=self.session_id, input_file=input_file, processing_mode=processing_mode, sample_rate=sample_rate
        )

    def log_step(
        self,
        step_id: str,
        phase: str,
        module_name: str,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        processing_time_ms: float,
        parameters: dict[str, Any] | None = None,
    ):
        """
        Protokolliert a single processing step.

        Args:
            step_id: Unique step identifier
            phase: Phase name
            module_name: Module used
            audio_before: Audio before processing
            audio_after: Audio after processing
            sr: Sample rate
            processing_time_ms: Processing time in milliseconds
            parameters: Additional parameters
        """
        if self.trace is None:
            raise RuntimeError("Must call start_session() first")

        # Compute quality metrics
        metrics_before = self._compute_metrics(audio_before, sr)
        metrics_after = self._compute_metrics(audio_after, sr)

        # Save audio snapshots
        audio_before_path = None
        audio_after_path = None
        if self.save_audio_snapshots:
            ext = "flac" if self.compress_audio else "wav"
            audio_before_path = str(self.output_dir / f"{step_id}_before.{ext}")
            audio_after_path = str(self.output_dir / f"{step_id}_after.{ext}")

            sf.write(audio_before_path, audio_before, sr)
            sf.write(audio_after_path, audio_after, sr)

        # Create step
        step = ProcessingStep(
            step_id=step_id,
            phase=phase,
            module_name=module_name,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            processing_time_ms=processing_time_ms,
            audio_before_path=audio_before_path,
            audio_after_path=audio_after_path,
            parameters=parameters or {},
        )

        self.trace.steps.append(step)

    def end_session(self, output_file: str | None = None):
        """End the current session and save logs."""
        if self.trace is None:
            raise RuntimeError("No active session")

        self.trace.end_time = datetime.now().isoformat()
        self.trace.output_file = output_file

        # Calculate total time
        step_times = [s.processing_time_ms for s in self.trace.steps]
        self.trace.total_processing_time_sec = sum(step_times) / 1000.0

        # Save logs
        if self.save_json:
            json_path = self.output_dir / "trace.json"
            with open(json_path, "w") as f:
                json.dump(self.trace.to_dict(), f, indent=2)

        if self.save_markdown:
            md_path = self.output_dir / "report.md"
            with open(md_path, "w") as f:
                f.write(self.trace.to_markdown())

        return self.trace

    def _compute_metrics(self, audio: np.ndarray, sr: int) -> QualityMetrics:
        """Berechnet quality metrics for audio."""
        # Ensure mono for consistent metrics
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # SNR (estimate using high-pass filtered energy vs original)
        snr_db = self._estimate_snr(audio, sr)

        # THD (Total Harmonic Distortion)
        thd_percent = self._estimate_thd(audio, sr)

        # LUFS (Loudness)
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            lufs = meter.integrated_loudness(audio)
        except Exception:
            # Fallback: approximate using RMS
            rms = np.sqrt(np.mean(audio**2))
            lufs = 20 * np.log10(rms + 1e-10) - 23.0  # Rough approximation

        # Spectral Centroid
        if librosa is not None:
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        else:
            freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
            mag = np.abs(np.fft.rfft(audio))
            spectral_centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-10))

        # Peak and RMS
        peak_db = 20 * np.log10(np.abs(audio).max() + 1e-10)
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        dynamic_range_db = peak_db - rms_db

        return QualityMetrics(
            snr_db=snr_db,
            thd_percent=thd_percent,
            lufs=lufs,
            spectral_centroid_hz=spectral_centroid,
            peak_db=peak_db,
            rms_db=rms_db,
            dynamic_range_db=dynamic_range_db,
        )

    def _estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt Signal-to-Noise Ratio."""
        # Simple estimation: high-pass filtered energy / full signal energy
        from scipy import signal

        # High-pass filter at 100 Hz (remove low-frequency noise)
        sos = signal.butter(4, 100, "high", fs=sr, output="sos")
        # sosfiltfilt (zero-phase) required: filtered is subtracted from audio for noise estimation;
        # causal sosfilt would introduce group delay → inaccurate SNR estimate (§2.51, V11)
        filtered = signal.sosfiltfilt(sos, audio)

        signal_power = np.mean(filtered**2)
        noise_power = np.mean((audio - filtered) ** 2)

        if noise_power < 1e-10:
            return 60.0  # Very clean signal

        snr = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr, -20, 80))  # Reasonable bounds

    def _estimate_thd(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt Total Harmonic Distortion."""
        # Simple estimation using spectral analysis
        # Real THD would require fundamental frequency detection

        # Compute FFT
        fft = np.fft.rfft(audio)
        magnitudes = np.abs(fft)

        # Find fundamental (peak in 80-800 Hz range for vocals)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        vocal_range = (freqs >= 80) & (freqs <= 800)

        if not np.any(vocal_range):
            return 0.1  # No fundamental detected

        fundamental_idx = np.argmax(magnitudes[vocal_range])
        fundamental_power = magnitudes[vocal_range][fundamental_idx] ** 2

        # Estimate harmonics power (rough approximation)
        total_power: float = float(np.sum(magnitudes**2))
        harmonic_power = total_power - fundamental_power

        if fundamental_power < 1e-10:
            return 0.1

        thd = 100 * np.sqrt(harmonic_power / fundamental_power)
        return float(np.clip(thd, 0.0, 50.0))  # Reasonable bounds


# Convenience function
def create_logger(
    session_id: str | None = None, output_dir: Path | None = None, save_audio: bool = True, compress: bool = False
) -> ProcessingLogger:
    """
    Erstellt a ProcessingLogger instance.

    Args:
        session_id: Optional session ID
        output_dir: Optional output directory
        save_audio: Save audio snapshots
        compress: Use FLAC compression

    Returns:
        ProcessingLogger instance
    """
    return ProcessingLogger(
        session_id=session_id, output_dir=output_dir, save_audio_snapshots=save_audio, compress_audio=compress
    )


logger = logging.getLogger(__name__)
