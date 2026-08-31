"""EAR_VAE Plugin — Neural Audio Reconstruction via ONNX.

εar-VAE (earlab/EAR_VAE, Apache 2.0): high-fidelity stereo audio VAE with
perceptual K-weighting, phase-derivative loss, and stereo correlation loss.

Exported to ONNX from PyTorch (no runtime PyTorch dependency):
  - Encoder: models/ear_vae/encoder.onnx (321 MB) — audio → latent
  - Decoder: models/ear_vae/decoder.onnx (322 MB) — latent → audio

Architecture:
  Input:  stereo float32 (2, N) @ 48 kHz
  Latent: 64-dim, 960× compression (~50 fps)
  Output: stereo float32 (2, N) @ 48 kHz

Aurik integration: Phase-0 preprocessor / neural denoiser / M-S reconstruction.
License: Apache 2.0 — freely usable for commercial purposes.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent / "models" / "ear_vae"
_ENCODER_ONNX = _MODEL_DIR / "encoder.onnx"
_DECODER_ONNX = _MODEL_DIR / "decoder.onnx"

# ---------------------------------------------------------------------------
# ONNX sessions (lazy-loaded, thread-safe)
# ---------------------------------------------------------------------------
_encoder_session = None
_decoder_session = None
_sessions_failed: bool = False
_sessions_lock = threading.Lock()

# Singleton
_lock = threading.Lock()
_instance: EarVAEPlugin | None = None


def _load_session(path: Path, name: str):
    """Thread-safe lazy load of an ONNX session."""
    global _encoder_session, _decoder_session, _sessions_failed  # pylint: disable=global-statement

    if name == "encoder" and _encoder_session is not None:
        return _encoder_session
    if name == "decoder" and _decoder_session is not None:
        return _decoder_session

    with _sessions_lock:
        if name == "encoder" and _encoder_session is not None:
            return _encoder_session
        if name == "decoder" and _decoder_session is not None:
            return _decoder_session
        if _sessions_failed:
            return None

        if not path.exists():
            logger.info("EAR_VAE ONNX not found: %s", path)
            _sessions_failed = True
            return None

        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            logger.info("EAR_VAE %s ONNX geladen (%s)", name, path)

            if name == "encoder":
                _encoder_session = session
            else:
                _decoder_session = session
            return session
        except Exception as exc:
            logger.warning("EAR_VAE %s ONNX laden fehlgeschlagen: %s", name, exc)
            _sessions_failed = True
            return None


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class EarVAEPlugin:
    """Neural audio reconstruction via EAR_VAE ONNX.

    Public API:
        _ok / _model_loaded: bool — True if both ONNX sessions loaded.
        TARGET_SR: int = 48000
        encode(audio) → latent np.ndarray
        decode(latent) → audio np.ndarray
        process(audio)  → audio np.ndarray (encode→decode clean pass)
    """

    TARGET_SR: int = 48000

    def __init__(self) -> None:
        self._enc = _load_session(_ENCODER_ONNX, "encoder")
        self._dec = _load_session(_DECODER_ONNX, "decoder")
        self._ok = self._enc is not None and self._dec is not None

    @property
    def _model_loaded(self) -> bool:
        return self._ok

    # ── Core encode/decode ──────────────────────────────────────────

    def encode(self, audio: np.ndarray) -> np.ndarray | None:
        """Encode stereo audio to latent.

        Args:
            audio: (2, samples) float32 at 48 kHz.

        Returns:
            (64, latent_frames) float32, or None.
        """
        if not self._ok or self._enc is None:
            return None
        try:
            audio = self._prepare_input(audio)
            # ONNX expects (batch=1, 2, N)
            inp = audio[np.newaxis, :, :].astype(np.float32)
            latent = self._enc.run(None, {"audio": inp})[0]
            return latent[0].astype(np.float32)  # type: ignore  # (128 or 64, T')
        except Exception as exc:
            logger.debug("EAR_VAE encode fehlgeschlagen: %s", exc)
            return None

    def decode(self, latent: np.ndarray) -> np.ndarray | None:
        """Decode latent to stereo audio.

        Args:
            latent: (64, latent_frames) or (128, latent_frames) float32.

        Returns:
            (2, samples) float32 at 48 kHz, or None.
        """
        if not self._ok or self._dec is None:
            return None
        try:
            latent = np.asarray(latent, dtype=np.float32)
            if latent.ndim == 2:
                latent = latent[np.newaxis, :, :]  # (1, C, T')
            elif latent.ndim == 4:
                latent = latent[0]  # strip extra dim
            audio = self._dec.run(None, {"latent": latent})[0]
            result = audio[0].astype(np.float32)  # (2, N)
            return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("EAR_VAE decode fehlgeschlagen: %s", exc)
            return None

    def process(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray | None:
        """Neural clean pass: encode → decode.

        The VAE's perceptual bottleneck removes noise while preserving
        phase coherence and stereo imaging (trained with K-filter +
        phase-derivative + stereo-correlation losses).

        Args:
            audio: Float32 audio, any channel layout.
            sample_rate: Input sample rate.

        Returns:
            (2, samples) float32 at 48 kHz, or None.
        """
        latent = self.encode(audio)
        if latent is None:
            return None
        return self.decode(latent)

    # ── Helpers ─────────────────────────────────────────────────────

    def _prepare_input(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to (2, N) float32 stereo."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2 and audio.shape[1] <= 2 and audio.shape[1] < audio.shape[0]:
            audio = audio.T  # (N, 2) → (2, N)
        if audio.shape[0] > 2:
            audio = audio[:2, :]
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        return cast(np.ndarray, np.ascontiguousarray(audio))

    def unload(self) -> None:
        """Release ONNX sessions."""
        global _encoder_session, _decoder_session  # pylint: disable=global-statement
        _encoder_session = None
        _decoder_session = None
        self._enc = None
        self._dec = None
        self._ok = False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def get_ear_vae_plugin() -> EarVAEPlugin:
    """Thread-safe singleton."""
    global _instance  # pylint: disable=global-statement
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = EarVAEPlugin()
        return _instance


def unload_ear_vae() -> None:
    """Release ONNX sessions."""
    global _instance  # pylint: disable=global-statement
    if _instance is not None:
        _instance.unload()
        _instance = None
