"""
ONNX Runtime for optimized model inference

Provides OptimizedONNXModel class for CPU-optimized ONNX inference
with 1.5-2× speedup over PyTorch.
"""

import contextlib
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

# ── §D4 + D1: Suppress ONNX Runtime C-level warnings ────────────────
# 1. Set ONNX logger severity to ERROR (suppresses harmless WARNINGs like
#    MIOpen epsilon, 40+ lines per model load).
ort.set_default_logger_severity(3)  # 0=VERBOSE 1=INFO 2=WARNING 3=ERROR 4=FATAL

# 2. Install stderr deduplication wrapper for any remaining C-level output
#    that bypasses ONNX's logger (CUDA, ROCm, system libraries).
try:
    from backend.core.stderr_dedup import install_stderr_dedup

    install_stderr_dedup()
except Exception as _e:
    logging.warning("onnx_runtime: non-critical exception: %s", _e)

logger = logging.getLogger(__name__)


class ONNXProvider(Enum):
    """ONNX Runtime execution providers.

    §Spec: Aurik ist CPU-only (§3.x, map_location='cpu').
    CUDA/TensorRT existieren NICHT - nur CPUExecutionProvider erlaubt.
    """

    CPU = "CPUExecutionProvider"


@dataclass
class ModelInfo:
    """ONNX model metadata"""

    model_path: Path
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    sample_rate: int
    model_type: str  # 'denoising', 'separation', 'enhancement'
    quantized: bool = False
    opset_version: int = 14


class ONNXModelStatus(Enum):
    """Model availability status"""

    READY = "ready"
    LOADING = "loading"
    ERROR = "error"
    NOT_FOUND = "not_found"


class ONNXInferenceSession:
    """
    Manages ONNX Runtime session with optimizations.

    Features:
    - CPU-optimized session options
    - Automatic batch handling
    - Warmup for consistent performance
    - Performance tracking
    """

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
        intra_op_num_threads: int = 4,
        inter_op_num_threads: int = 2,
        enable_profiling: bool = False,
    ):
        """
        Initialisiert ONNX Runtime session.

        Args:
            model_path: Path to ONNX model file
            providers: Execution providers (default: CPU-only)
            intra_op_num_threads: Intra-op parallelism threads
            inter_op_num_threads: Inter-op parallelism threads
            enable_profiling: Enable ONNX Runtime profiling
        """
        self.model_path = Path(model_path)
        self.providers = providers or [ONNXProvider.CPU.value]

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Configure session options for CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.inter_op_num_threads = inter_op_num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if enable_profiling:
            sess_options.enable_profiling = True

        # §2.37 / Checkliste: try_allocate() vor jedem InferenceSession-Load (RELEASE_MUST)
        _session_name = f"ONNX_{self.model_path.stem}"
        try:
            from backend.core.ml_memory_budget import release as _ml_release
            from backend.core.ml_memory_budget import try_allocate as _try_allocate

            _model_size_gb = self.model_path.stat().st_size / (1024**3) if self.model_path.exists() else 0.1
            if not _try_allocate(_session_name, size_gb=_model_size_gb):
                raise MemoryError(
                    f"ML-Budget erschöpft — ONNX-Session '{self.model_path.name}' kann nicht geladen werden. "
                    "Lösung: bitte andere ML-Modelle zuerst entladen."
                )
        except MemoryError:
            raise
        except Exception as _budget_exc:
            logger.debug("ml_memory_Grenze nicht verfügbar: %s — Sitzung wird ohne Grenze geladen.", _budget_exc)
            _ml_release = None  # type: ignore[assignment]

        try:
            # Try MIGraphX GPU backend first if requested
            # (§v10.40 Compile-Zeit-Regel: Modelle > 200 MB → ORT statt MIGraphX)
            if "MIGraphXExecutionProvider" in self.providers:
                try:
                    from backend.core.migraphx_adapter import MIGraphXSession, is_migraphx_size_eligible

                    if not is_migraphx_size_eligible(self.model_path):
                        logger.debug(
                            "MIGraphX übersprungen (§v10.40 Größenlimit > 200 MB): %s",
                            self.model_path.name,
                        )
                        self.session = ort.InferenceSession(
                            str(self.model_path), sess_options, providers=self.providers
                        )
                    else:
                        self.session = MIGraphXSession(
                            self.model_path,
                            providers=self.providers,
                            sess_options=sess_options,
                        )
                        logger.info("ONNX Sitzung via MIGraphX (GPU): %s", self.model_path.name)
                except Exception as _mgx_exc:
                    logger.debug("MIGraphX session fallback: %s", _mgx_exc)
                    self.session = ort.InferenceSession(str(self.model_path), sess_options, providers=self.providers)
            else:
                self.session = ort.InferenceSession(str(self.model_path), sess_options, providers=self.providers)

            # Cache input/output names and shapes
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape

            logger.info("ONNX Sitzung initialisiert: %s", self.model_path.name)
            logger.info("Eingabe: %s %s", self.input_name, self.input_shape)
            logger.info("Ausgabe: %s %s", self.output_name, self.output_shape)
            logger.info("Providers: %s", self.session.get_providers())

        except Exception as e:
            if _ml_release is not None:
                with contextlib.suppress(Exception):
                    _ml_release(_session_name)
            logger.error("konnte nicht initialisieren ONNX Sitzung: %s", e)
            raise

        # Performance tracking
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.is_warmed_up = False

    def warmup(self, num_iterations: int = 3) -> None:
        """
        Warmup ONNX session for consistent performance.

        First few inferences are typically slower due to initialization.
        Warmup ensures stable performance benchmarking.

        Args:
            num_iterations: Number of warmup iterations
        """
        if self.is_warmed_up:
            logger.debug("Sitzung already warmed up")
            return

        logger.info("Warming up ONNX Sitzung (%s iterations)...", num_iterations)

        # Create dummy input matching expected shape
        dummy_shape = list(self.input_shape)
        # Replace dynamic dimensions (-1) with concrete values
        for i, dim in enumerate(dummy_shape):
            if isinstance(dim, str) or dim == -1:
                dummy_shape[i] = 1 if i == 0 else 16000  # batch=1, audio=16000 samples

        dummy_input = np.random.randn(*dummy_shape).astype(np.float32)

        for i in range(num_iterations):
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})

        self.is_warmed_up = True
        logger.info("Warmup vollstaendig")

    def run(self, inputs: dict[str, np.ndarray], profile: bool = False) -> list[np.ndarray]:
        """
        Führt aus: ONNX inference.

        Args:
            inputs: Input tensors as {name: array}
            profile: Enable profiling for this run

        Returns:
            List of output tensors
        """
        start_time = time.time()

        try:
            outputs = self.session.run(None, inputs)

            inference_time = time.time() - start_time
            self.total_inferences += 1
            self.total_inference_time += inference_time

            if profile:
                logger.info("Inference time: %.2f ms", inference_time * 1000)

            return outputs  # type: ignore[no-any-return]

        except Exception as e:
            logger.error("ONNX inference fehlgeschlagen: %s", e)
            raise

    def get_stats(self) -> dict[str, Any]:
        """Gibt zurück: inference statistics."""
        avg_time = self.total_inference_time / self.total_inferences if self.total_inferences > 0 else 0.0

        return {
            "total_inferences": self.total_inferences,
            "total_time_seconds": self.total_inference_time,
            "average_time_ms": avg_time * 1000,
            "model_path": str(self.model_path),
            "providers": self.session.get_providers(),
        }

    def reset_stats(self) -> None:
        """Setzt zurück: inference statistics."""
        self.total_inferences = 0
        self.total_inference_time = 0.0


class OptimizedONNXModel:
    """
    High-level ONNX model wrapper for audio processing.

    Provides simple interface for ONNX inference with:
    - Automatic shape handling
    - Batch processing support
    - Performance tracking
    - Error handling

    Expected speedup: 1.5-2× vs PyTorch
    """

    def __init__(
        self,
        model_path: Path,
        model_type: str = "denoising",
        sample_rate: int = 48000,
        providers: list[str] | None = None,
        enable_warmup: bool = True,
    ):
        """
        Initialisiert optimized ONNX model.

        Args:
            model_path: Path to ONNX model file
            model_type: Model type ('denoising', 'separation', 'enhancement')
            sample_rate: Expected audio sample rate
            providers: ONNX execution providers
            enable_warmup: Warmup session on initialization
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.sample_rate = sample_rate

        # Initialize ONNX session
        self.session = ONNXInferenceSession(model_path=self.model_path, providers=providers)

        if enable_warmup:
            self.session.warmup()

        self.status = ONNXModelStatus.READY
        logger.info("OptimizedONNXModel ready: %s", self.model_path.name)

    def process(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """
        Verarbeitet audio through ONNX model.

        Args:
            audio: Input audio array (1D or 2D)
            sr: Sample rate (optional, uses model default)

        Returns:
            Processed audio array
        """
        if sr is not None and sr != self.sample_rate:
            logger.warning("Sample rate mismatch: expected %s, got %s", self.sample_rate, sr)

        # Handle input shape
        audio = self._prepare_input(audio)

        # Run inference
        inputs = {self.session.input_name: audio}
        outputs = self.session.run(inputs)

        # Extract output
        output_audio = outputs[0]

        # Handle output shape
        output_audio = self._prepare_output(output_audio)

        return output_audio

    def process_batch(self, audio_batch: list[np.ndarray], sr: int | None = None) -> list[np.ndarray]:
        """
        Verarbeitet batch of audio samples.

        Args:
            audio_batch: List of audio arrays
            sr: Sample rate

        Returns:
            List of processed audio arrays
        """
        results = []

        for audio in audio_batch:
            processed = self.process(audio, sr)
            results.append(processed)

        return results

    def _prepare_input(self, audio: np.ndarray) -> np.ndarray:
        """
        Prepare audio for ONNX inference.

        Handles:
        - Shape conversion (1D → 2D with batch dimension)
        - Data type conversion (float32)
        - Normalization (if needed)
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # Add batch dimension if needed [batch, samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim == 2:
            # Assume [batch, samples] or [samples, channels]
            if audio.shape[1] < audio.shape[0]:
                # Likely [samples, channels], transpose to [channels, samples]
                audio = audio.T

        return audio

    def _prepare_output(self, output: np.ndarray) -> np.ndarray:
        """
        Prepare ONNX output for return.

        Handles:
        - Shape conversion (remove batch dimension)
        - Data type conversion
        """
        # Remove batch dimension if batch size = 1
        if output.ndim >= 2 and output.shape[0] == 1:
            output = output.squeeze(0)

        return output

    def get_stats(self) -> dict[str, Any]:
        """Gibt zurück: model inference statistics."""
        stats = self.session.get_stats()
        stats.update({"model_type": self.model_type, "sample_rate": self.sample_rate, "status": self.status.value})
        return stats

    def reset_stats(self) -> None:
        """Setzt zurück: inference statistics."""
        self.session.reset_stats()

    def __repr__(self) -> str:
        return f"OptimizedONNXModel(model={self.model_path.name}, type={self.model_type}, status={self.status.value})"
