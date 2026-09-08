"""
§v10.124: Aurik SOTA Pipeline — Denoiser → Music Enhancer → Vocal Enhancer.

Kette:
  1. DFN Expanded Denoiser (7.0 dB SNR, PyTorch GPU)
  2. MelBandRoformer Music Enhancer (860M, ONNX GPU) — optional, CPU-intensiv
  3. Vocal Enhancer DSP (Mid/Side + Transient-Breath + De-Essing)

Alle drei Module sind einzeln aufrufbar oder als Pipeline.
Fallback: Jedes Modul hat try/except → graceful degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


def _get_session_manager():
    """§15.9/P1-1: Zentrale ONNX-Session-Residency (kein Reload je Aufruf)."""
    from backend.core.ml.session_manager import get_session_manager

    return get_session_manager()


@dataclass
class PipelineResult:
    audio: np.ndarray
    denoiser_active: bool = False
    music_enhancer_active: bool = False
    vocal_enhancer_active: bool = False
    denoiser_snr_db: float = 0.0


def denoise(audio: np.ndarray, sample_rate: int = 48000) -> tuple[np.ndarray, float]:
    """Step 1: DFN Expanded Denoising (7.0 dB SNR)."""
    try:
        from backend.core.dfn_expanded_inference import DFNExpandedDenoiser

        denoiser = DFNExpandedDenoiser()
        result = denoiser.denoise(audio, sample_rate)
        # Estimate SNR improvement (simplified)
        noise_before = np.mean((audio - result[: len(audio)]) ** 2)
        snr_db = float(10 * np.log10((np.mean(audio**2) + 1e-12) / (noise_before + 1e-12)))
        logger.info("DFN Expanded: %.1f dB SNR", snr_db)
        return result.astype(np.float32), snr_db
    except Exception as e:
        logger.debug("DFN Expanded nicht verfügbar: %s", e)
        return audio.astype(np.float32), 0.0


def enhance_music(audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """Step 2: MelBandRoformer Music Enhancement (860M, ONNX GPU)."""
    try:
        from pathlib import Path

        model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "models"
            / "melbandroformer"
            / "melbandroformer_optimized.onnx"
        )
        if not model_path.exists():
            logger.debug("MelBandRoformer nicht gefunden")
            return audio

        session = _get_session_manager().acquire("melbandroformer", str(model_path))
        logger.info("MelBandRoformer: %s", session.get_providers()[0])

        # MelBandRoformer needs Mel spectrogram [1, duration, 60, 384]
        # Simplified path: use librosa mel transformation
        import librosa

        mel = librosa.feature.melspectrogram(y=audio.astype(np.float64), sr=sample_rate, n_mels=60, hop_length=512)
        mel_db = np.log1p(mel).astype(np.float32)

        # Process in 384-frame chunks
        T = mel_db.shape[1]
        out_mel = np.zeros_like(mel_db)
        chunk_size = 384
        overlap = 192

        for pos in range(0, T, overlap):
            end = min(pos + chunk_size, T)
            chunk = mel_db[:, pos:end]
            if chunk.shape[1] < chunk_size:
                chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])), mode="edge")
            inp = chunk[np.newaxis, np.newaxis, :, :]
            out = session.run(None, {"input": inp})[0]
            actual = min(chunk_size, T - pos)
            out_mel[:, pos : pos + actual] = out[0, 0, :, :actual]

        # Inverse mel to audio
        enhanced = librosa.feature.inverse.mel_to_audio(np.exp(out_mel) - 1, sr=sample_rate, hop_length=512, n_fft=2048)
        # Match length
        if len(enhanced) < len(audio):
            enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
        else:
            enhanced = enhanced[: len(audio)]

        logger.info("MelBandRoformer: Verbesserung angewendet")
        return cast(np.ndarray, enhanced.astype(np.float32))

    except Exception as e:
        logger.debug("MelBandRoformer nicht verfügbar: %s", e)
        return audio


def enhance_vocals(audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Step 3: Vocal Enhancement (DSP: Mid/Side + Breath + De-Ess)."""
    try:
        from backend.core.vocal_enhancer import enhance_vocals as _enhance_vocals

        result = _enhance_vocals(audio, sr=sample_rate, breath_reduction_db=3.0, sibilance_reduction_db=2.0)
        _audio_out = cast(Any, result).audio if hasattr(result, "audio") else cast(Any, result)
        logger.info("Vocal Enhancer: angewendet")
        return cast(np.ndarray, _audio_out.astype(np.float32))
    except Exception as e:
        logger.debug("Vocal Enhancer DSP nicht verfügbar: %s", e)
        # Fallback: Aurik's built-in vocal enhancer
        try:
            from backend.core.vocal_ai_enhancement import UnifiedVocalAIEnhancer

            ve = UnifiedVocalAIEnhancer(sample_rate=sample_rate)
            result2 = ve.enhance(audio, breath_preservation=0.7, sibilance_reduction=True)
            _audio_out = cast(Any, result2).audio if hasattr(result2, "audio") else cast(Any, result2)
            logger.info("Vocal Enhancer: Aurik-Rückfall angewendet")
            return cast(np.ndarray, _audio_out.astype(np.float32))
        except Exception as e2:
            logger.debug("Aurik Vocal Enhancer nicht verfügbar: %s", e2)
            return audio


def process(
    audio: np.ndarray,
    sample_rate: int = 48000,
    enable_denoiser: bool = True,
    enable_music_enhancer: bool = True,
    enable_vocal_enhancer: bool = True,
) -> PipelineResult:
    """Full Aurik SOTA Pipeline.

    Args:
        audio: float32 [samples] or [channels, samples]
        sample_rate: Sample rate
        enable_denoiser: Step 1
        enable_music_enhancer: Step 2 (CPU/GPU-intensiv)
        enable_vocal_enhancer: Step 3

    Returns:
        PipelineResult with enhanced audio
    """
    result = PipelineResult(audio=audio.astype(np.float32))

    if enable_denoiser:
        result.audio, result.denoiser_snr_db = denoise(result.audio, sample_rate)
        result.denoiser_active = True

    if enable_music_enhancer:
        # Resample to 44100 if needed (MelBandRoformer is 44.1kHz)
        sr_music = 44100
        if sample_rate != sr_music:
            import librosa

            audio_44k = librosa.resample(result.audio.astype(np.float64), orig_sr=sample_rate, target_sr=sr_music)
        else:
            audio_44k = result.audio

        enhanced = enhance_music(audio_44k, sr_music)
        if enhanced is not result.audio:
            # Resample back
            if sample_rate != sr_music:
                import librosa

                enhanced = librosa.resample(enhanced.astype(np.float64), orig_sr=sr_music, target_sr=sample_rate)
            result.audio = enhanced.astype(np.float32)
            result.music_enhancer_active = True

    if enable_vocal_enhancer:
        result.audio = enhance_vocals(result.audio, sample_rate)
        result.vocal_enhancer_active = True

    result.audio = np.clip(result.audio, -1.0, 1.0).astype(np.float32)
    return result
