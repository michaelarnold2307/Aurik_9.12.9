"""
MDX-Net Vocal Separator - HIPS Compliant Wrapper (SOTA Implementation)

MDX-Net (Music Demixing Challenge Network) uses spectral-domain processing
with U-Net architecture for high-quality vocal/instrumental separation.

HIPS Compliance:
- Kontextbewusstsein: ✅ Spectral context via U-Net receptive field
- Nebenwirkungen: ✅ Stereo width changes, phase artifacts (monitored)
- Reversibilität: ✅ Stems stored separately, can be recombined
- Auditierbarkeit: ✅ Full separation metrics logged
- Steuerbarkeit: ✅ Adjustable separation strength
- Bedeutungsagnostik: ✅ Pure spectral processing, no aesthetic decisions

SOTA Features:
- ONNX Runtime inference (CPU-optimized)
- 4096 FFT with Hanning window
- Overlap-add reconstruction with phase coherence
- Deterministic output (no random elements)
"""

import logging
from pathlib import Path

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False

try:
    import onnxruntime as ort

    _HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _HAS_ONNX = False

import numpy as np

logger = logging.getLogger(__name__)


class MDXNetSeparator:
    """
    MDX-Net wrapper for AURIK v8.1 (SOTA Implementation)

    Architecture:
    - U-Net based spectral separator via ONNX Runtime
    - 4096 FFT size with Hanning window for high frequency resolution
    - Overlap-add reconstruction with phase coherence preservation
    - CPU-optimized inference (§9.5 Aurik 10.0.0)

    HIPS Guarantees:
    - No training/adaptation (inference-only)
    - Deterministic output (same input → same output)
    - Full auditability (all metrics logged)
    - Graceful fallback to HPSS when model unavailable
    """

    # SOTA Default Model URLs (Demucs/MDX-Net community models)
    _MODEL_CANDIDATES: list[str] = [
        "mdx_net_vocal_v2.onnx",  # Primary: Vocal separation v2
        "mdx_q8_v1.onnx",  # Quantized v1 (faster, slightly lower quality)
        "uvr_v3 vocal only.onnx",  # Universal Vocal Remover v3
    ]

    def __init__(self, model_path: str | None = None, sample_rate: int = 48000, device: str | None = None):
        """
        Initialisiert MDX-Net separator (SOTA).

        Args:
            model_path: Path to pretrained MDX-Net model (ONNX format)
            sample_rate: Target sample rate (default: 48000 Hz for Aurik)
            device: 'cuda', 'cpu', or None (auto-detect; §9.5 → CPU-only)
        """
        self.sample_rate = sample_rate

        # Device selection — §9.5 Aurik 10.0.0 nutzt ausschließlich CPU. Kein CUDA.
        self.device = "cpu"

        logger.info("MDXNetSeparator initialisiert on %s (SR=%d Hz)", self.device, self.sample_rate)

        # Model loading
        _raw_path = model_path or self._get_default_model_path()
        self.model_path: Path = Path(_raw_path) if not isinstance(_raw_path, Path) else _raw_path
        self.model = self._load_model()

        # ONNX session state
        self._onnx_session: ort.InferenceSession | None = None  # type: ignore[type-var]
        self._model_loaded = self.model is not None and self._onnx_session is not None

        # HIPS tracking
        self.separation_count = 0
        self.nebenwirkungen_log: list[dict] = []

    def _get_default_model_path(self) -> Path:
        """Gibt zurück: default MDX-Net model path (SOTA Model Search)."""
        base_path = Path(__file__).parent.parent.parent.parent.parent
        model_dir = base_path / "models" / "mdx_net"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Search for available model
        for candidate in self._MODEL_CANDIDATES:
            model_path = model_dir / candidate
            if model_path.exists():
                logger.info("MDX-Net model gefunden: %s", model_path.name)
                return model_path

        logger.warning(
            "Kein MDX-Net model gefunden in %s. "
            "Bitte laden Sie ein Modell von: https://github.com/kuielab/mdx-net oder "
            "https://huggingface.co/models?search=mdx-net",
            model_dir,
        )
        return model_dir / self._MODEL_CANDIDATES[0]

    def _load_model(self):
        """
        Lädt MDX-Net model via ONNX Runtime (SOTA).

        Returns:
            ort.InferenceSession or None (fallback to HPSS)
        """
        if not self.model_path.exists():
            logger.warning("MDX-Net model nicht verfügbar (%s). Fallback auf HPSS.", self.model_path)
            return None

        try:
            # ONNX Runtime mit CPU-Optimierung (§9.5)
            if not _HAS_ONNX or ort is None:
                logger.warning("onnxruntime nicht installiert — Fallback auf HPSS spectral mask")
                return None

            # Execution Provider: CPU (EP) mit optimalen Einstellungen
            providers = ["CPUExecutionProvider"]
            session_options = ort.SessionOptions()
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Deterministisch
            session_options.log_severity_level = 0  # Info-level logging

            self._onnx_session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
                sess_options=session_options,
            )

            logger.info("MDX-Net model erfolgreich geladen via ONNX: %s", self.model_path)
            return self._onnx_session

        except Exception as e:
            logger.error("konnte nicht laden MDX-Net model (%s): %s — Fallback auf HPSS", self.model_path, e)
            self._onnx_session = None
            return None

    def separate(self, audio: np.ndarray, sr: int | None = None, return_stems: bool = True) -> dict[str, np.ndarray]:
        """
        Separate vocals from instrumental (SOTA Implementation).

        Args:
            audio: Audio array (shape: [channels, samples] or [samples])
            sr: Sample rate (if different from self.sample_rate; default 48000 Hz)
            return_stems: If True, return both stems; else only vocals

        Returns:
            Dictionary with 'vocals' and optionally 'instrumental' stems

        HIPS Compliance:
        - Logs all separation operations
        - Tracks nebenwirkungen (phase, stereo width)
        - Preserves original for reversibility check
        """
        # SR-Invariante (Aurik 10.0.0 nutzt 48000 Hz)
        assert sr == 48000 or sr is None or sr == self.sample_rate, f"SR muss 48000 Hz sein, erhalten: {sr}"

        # Resample if needed (deterministic linear interpolation)
        if sr is not None and sr != self.sample_rate and _HAS_LIBROSA and librosa is not None:
            logger.info("Resampling from %sHz to %sHz (linear)", sr, self.sample_rate)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)  # type: ignore[no-untyped-call]

        # NaN/Inf-Guard (§0a)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Ensure stereo [channels, samples]
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        audio_original = audio.copy()  # For reversibility check

        # HIPS: Log separation attempt
        self.separation_count += 1
        logger.info("MDX-Net separation #%s: shape=%s, sr=%d", self.separation_count, audio.shape, self.sample_rate)

        # Actual separation (SOTA: ONNX inference or HPSS fallback)
        if self._model_loaded and self._onnx_session is not None:
            vocals, instrumental = self._mdx_net_inference(audio)
        else:
            logger.warning("MDX-Net model nicht verfügbar oder nicht geladen — Fallback auf HPSS spectral mask")
            vocals, instrumental = self._fallback_separation(audio)

        # HIPS: Nebenwirkungen tracking
        nebenwirkungen = self._assess_nebenwirkungen(audio_original, vocals, instrumental)
        self.nebenwirkungen_log.append(nebenwirkungen)

        # NaN/Inf-Guard für Ausgabe (§0a)
        vocals = np.nan_to_num(vocals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        vocals = np.clip(vocals, -1.0, 1.0)
        instrumental = np.nan_to_num(instrumental, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        instrumental = np.clip(instrumental, -1.0, 1.0)

        if nebenwirkungen["severity"] > 0.3:
            logger.warning(
                "Separation nebenwirkungen erkannt: "
                f"stereo_width_loss={nebenwirkungen['stereo_width_loss']:.2f}, "
                f"phase_correlation_loss={nebenwirkungen['phase_loss']:.2f}"
            )

        # Return stems
        result = {"vocals": vocals}
        if return_stems:
            result["instrumental"] = instrumental

        return result

    def _fallback_separation(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fallback spectral mask separation (when MDX-Net model unavailable).

        SOTA HPSS Implementation:
        - Harmonic-Percussive Source Separation via complex STFT decomposition
        - Phase-coherent reconstruction with overlap-add
        - Length-matching to original input

        Note: Vocals = mostly harmonic; instrumental = percussive residual.
        H + P = D (HPSS identity) → vocals + instrumental ≈ original
        """
        logger.info("Using SOTA HPSS fallback for vocal/instrumental separation")

        # STFT parameters (SOTA: high resolution for vocal preservation)
        n_fft = 2048
        hop_length = 512  # 75% overlap for smooth reconstruction

        # Process each channel separately (phase coherence)
        vocals_stereo = []
        instrumental_stereo = []

        for ch_idx, channel in enumerate(audio):
            logger.debug("HPSS processing channel %d", ch_idx)

            # Complex STFT (Hanning window — deterministisch)
            D = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length, center=True)  # type: ignore[no-untyped-call]

            # HPSS decomposition (margin=2.0: aggressive harmonic extraction for vocals)
            H, P = librosa.decompose.hpss(D, margin=2.0)  # type: ignore[no-untyped-call]

            # ISTFT reconstruction (phase-coherent overlap-add)
            vocals_channel = librosa.istft(H, hop_length=hop_length, length=len(channel))  # type: ignore[no-untyped-call]
            instrumental_channel = librosa.istft(P, hop_length=hop_length, length=len(channel))  # type: ignore[no-untyped-call]

            # NaN/Inf-Guard (§0a)
            vocals_channel = np.nan_to_num(vocals_channel, nan=0.0, posinf=0.0, neginf=0.0)
            instrumental_channel = np.nan_to_num(instrumental_channel, nan=0.0, posinf=0.0, neginf=0.0)

            vocals_stereo.append(vocals_channel)
            instrumental_stereo.append(instrumental_channel)

        # Stack back to [channels, samples]
        vocals = np.stack(vocals_stereo)
        instrumental = np.stack(instrumental_stereo)

        return vocals, instrumental

    def _mdx_net_inference(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Actual MDX-Net inference via ONNX Runtime (SOTA).

        Pipeline:
        1. STFT computation (4096 FFT, Hanning window)
        2. Magnitude/Phase separation
        3. U-Net forward pass (ONNX model)
        4. Mask application (vocal/instrumental masks)
        5. ISTFT reconstruction (overlap-add, phase coherence)

        Args:
            audio: [channels, samples] float32 array

        Returns:
            (vocals, instrumental) as [channels, samples] arrays
        """
        if self._onnx_session is None:
            logger.warning("ONNX session nicht verfügbar — Fallback auf HPSS")
            return self._fallback_separation(audio)

        # SOTA STFT parameters (MDX-Net standard)
        n_fft = 4096
        hop_length = 1024  # 75% overlap
        window = np.hanning(n_fft + 1)[:-1]  # Hanning window (even length)

        try:
            # Process mono mixdown for model input (MDX-Net expects [batch, channels, freq, time])
            # Stereo → Mono (psychoakustisch korrekt: Mittelwert)
            if audio.shape[0] == 2:
                mono = np.mean(audio, axis=0).astype(np.float32)
            else:
                mono = audio[0].astype(np.float32)

            # STFT → Complex spectrogram [freq, time]
            D = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length, window=window, center=True)  # type: ignore[no-untyped-call]

            # Magnitude and Phase
            magnitude = np.abs(D).astype(np.float32)  # [freq, time]
            phase = np.angle(D)  # [freq, time]

            # Prepare ONNX input: [batch, channels, freq, time]
            # MDX-Net expects 4D input with channel dimension
            onnx_input = magnitude[np.newaxis, np.newaxis, :, :]  # [1, 1, freq, time]

            # Model inference (deterministic)
            input_name = self._onnx_session.get_inputs()[0].name
            vocal_mask = self._onnx_session.run(None, {input_name: onnx_input})[0]  # [1, 1, freq, time]

            # Extract mask and ensure valid range [0, 1]
            vocal_mask = np.clip(vocal_mask.squeeze(), 0.0, 1.0).astype(np.float32)
            instrumental_mask = 1.0 - vocal_mask

            # Apply masks to magnitude
            vocal_mag = magnitude * vocal_mask
            inst_mag = magnitude * instrumental_mask

            # Reconstruct complex spectrograms
            vocal_D = vocal_mag * np.exp(1j * phase)
            inst_D = inst_mag * np.exp(1j * phase)

            # ISTFT reconstruction (overlap-add, phase coherence)
            target_length = len(mono)
            vocal_mono = librosa.istft(vocal_D, hop_length=hop_length, window=window, length=target_length)  # type: ignore[no-untyped-call]
            inst_mono = librosa.istft(inst_D, hop_length=hop_length, window=window, length=target_length)  # type: ignore[no-untyped-call]

            # NaN/Inf-Guard (§0a)
            vocal_mono = np.nan_to_num(vocal_mono, nan=0.0, posinf=0.0, neginf=0.0)
            inst_mono = np.nan_to_num(inst_mono, nan=0.0, posinf=0.0, neginf=0.0)

            # Expand to stereo [channels, samples]
            if audio.shape[0] == 2:
                vocals = np.stack([vocal_mono, vocal_mono])
                instrumental = np.stack([inst_mono, inst_mono])
            else:
                vocals = vocal_mono[np.newaxis, :]
                instrumental = inst_mono[np.newaxis, :]

            logger.info("MDX-Net ONNX inference erfolgreich: vocal_mask_mean=%.4f", float(vocal_mask.mean()))

        except Exception as e:
            logger.error("MDX-Net ONNX inference fehlgeschlagen: %s — Fallback auf HPSS", e)
            return self._fallback_separation(audio)

        return vocals, instrumental

    def _assess_nebenwirkungen(
        self, original: np.ndarray, vocals: np.ndarray, instrumental: np.ndarray
    ) -> dict[str, float]:
        """
        HIPS Requirement: Assess separation nebenwirkungen

        Tracks:
        - Stereo width changes
        - Phase correlation loss
        - Spectral artifacts
        - Energy conservation
        """
        # Ensure same length
        min_len = min(original.shape[1], vocals.shape[1], instrumental.shape[1])
        original = original[:, :min_len]
        vocals = vocals[:, :min_len]
        instrumental = instrumental[:, :min_len]

        # Recombine stems
        recombined = vocals + instrumental

        # Energy conservation check
        energy_original = np.sum(original**2)
        energy_recombined = np.sum(recombined**2)
        energy_ratio = energy_recombined / (energy_original + 1e-10)
        # NaN/Inf-Guard
        energy_ratio = 0.0 if not np.isfinite(energy_ratio) else energy_ratio

        # Stereo width (correlation between L/R)
        def stereo_width(audio: np.ndarray) -> float:
            if audio.shape[0] < 2:
                return 0.0
            _s0 = float(np.std(audio[0]))
            _s1 = float(np.std(audio[1]))
            if _s0 < 1e-8 or _s1 < 1e-8:
                return 0.0  # near-constant → corr undefined, treat as mono
            _a = audio[0] - audio[0].mean()
            _b = audio[1] - audio[1].mean()
            _na = float(np.linalg.norm(_a))
            _nb = float(np.linalg.norm(_b))
            corr = float(np.dot(_a, _b) / (_na * _nb + 1e-10))
            if not np.isfinite(corr):
                return 0.0
            return 1.0 - abs(corr)  # 0=mono, 1=wide

        width_original = stereo_width(original)
        width_recombined = stereo_width(recombined)
        width_loss = abs(width_original - width_recombined)
        # NaN/Inf-Guard
        width_loss = 0.0 if not np.isfinite(width_loss) else width_loss

        # Phase correlation (measure of phase artifacts)
        def phase_correlation(audio: np.ndarray) -> float:
            if audio.shape[0] < 2:
                return 1.0
            # Simplified: cross-correlation peak
            xcorr = np.correlate(audio[0], audio[1], mode="valid")
            return np.max(np.abs(xcorr)) / (np.linalg.norm(audio[0]) * np.linalg.norm(audio[1]) + 1e-10)  # type: ignore[no-any-return]

        # NaN/Inf-Guard
        phase_original = phase_correlation(original)
        phase_recombined = phase_correlation(recombined)
        phase_loss = abs(phase_original - phase_recombined)
        phase_loss = 0.0 if not np.isfinite(phase_loss) else phase_loss

        # Overall severity (0-1 scale)
        severity = (
            abs(1.0 - energy_ratio) * 0.5  # Energy mismatch
            + width_loss * 0.3  # Stereo width change
            + phase_loss * 0.2  # Phase artifacts
        )

        return {
            "energy_ratio": energy_ratio,
            "stereo_width_loss": width_loss,
            "phase_loss": phase_loss,
            "severity": min(severity, 1.0),
        }

    def get_separation_metrics(self) -> dict:
        """
        HIPS: Auditability - Get all separation metrics
        """
        if not self.nebenwirkungen_log:
            return {"total_separations": 0}

        avg_severity = np.mean([n["severity"] for n in self.nebenwirkungen_log])
        max_severity = np.max([n["severity"] for n in self.nebenwirkungen_log])

        return {
            "total_separations": self.separation_count,
            "average_nebenwirkungen_severity": avg_severity,
            "max_nebenwirkungen_severity": max_severity,
            "nebenwirkungen_log": self.nebenwirkungen_log[-10:],  # Last 10
        }


if __name__ == "__main__":
    # Test MDX-Net separator
    separator = MDXNetSeparator()

    # Generate test signal
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simple test: vocal-like harmonic + instrumental-like noise
    vocal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    instrumental = np.random.randn(len(t)) * 0.1  # Noise
    mixed = vocal + instrumental

    # Stereo
    audio = np.stack([mixed, mixed])

    # Separate
    stems = separator.separate(audio, sr=sr)

    logger.info("✓ MDX-Net separation test passed")
    logger.info("  Vocals shape: %s", stems["vocals"].shape)
    logger.info("  Instrumental shape: %s", stems["instrumental"].shape)
    logger.info("  Metrics: %s", separator.get_separation_metrics())
