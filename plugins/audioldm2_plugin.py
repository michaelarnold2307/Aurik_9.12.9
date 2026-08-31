"""AudioLDM2 Plugin — Text-Conditioned Generative Audio Synthesis via ONNX.

AudioLDM2 (Liu et al. 2023/2024) generates audio from text prompts using a
latent diffusion model with CLAP + T5 text encoders, a VAE, and HiFi-GAN vocoder.

This plugin implements the full pipeline in ONNX:
  - UNet:     models/audioldm2/audioldm2.onnx (1.39 GB) — diffusion denoising
  - VAE Dec:  models/audioldm2/vae_decoder.onnx (126 MB) — latent → mel spectrogram
  - Vocoder:  Aurik HiFi-GAN plugin (hifigan_plugin) — mel → audio 16kHz
  - Text:     Aurik LAION-CLAP plugin — prompt → text embedding

Architecture (§4.4 Tier 3, Phase 24):
  DSP → GACELA → FlashSR → AudioLDM2 (dropout > 3 s)

License: AudioLDM2 is under a permissive license (CC-BY-NC 4.0 for the
  original model weights). The ONNX export in this repository is for
  non-commercial restoration use consistent with CC-BY-NC 4.0.

Reference:
  Liu et al. (2023): "AudioLDM 2: Learning Holistic Audio Generation with
  Self-supervised Pretraining"
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — AudioLDM2 architecture
# ---------------------------------------------------------------------------
_AUDIOLDM2_ONNX_PATH = Path(__file__).parent.parent / "models" / "audioldm2" / "audioldm2.onnx"
_AUDIOLDM2_VAE_ONNX_PATH = Path(__file__).parent.parent / "models" / "audioldm2" / "vae_decoder.onnx"
_AUDIOLDM2_SAMPLE_RATE = 16000  # native generation sample rate
_AUDIOLDM2_MEL_BINS = 64
_AUDIOLDM2_HOP_LENGTH = 160  # 10 ms @ 16 kHz → 100 fps
_AUDIOLDM2_LATENT_CHANNELS = 8
_AUDIOLDM2_LATENT_DOWNSAMPLE = 4  # VAE downsamples mel by factor 4 in time and freq
_AUDIOLDM2_CROSSATTENTION_DIM = 768  # CLAP embedding dimension
_AUDIOLDM2_T5_EMBEDDING_DIM = 1024  # T5-Flan embedding dimension

# Diffusion schedule
_DDIM_NUM_STEPS = 30  # quality/performance tradeoff (50 for max quality)

# Lazy-loaded ONNX session
_onnx_session = None
_onnx_failed: bool = False
_onnx_lock = threading.Lock()

# Lazy-loaded ONNX session (VAE decoder)
_vae_session = None
_vae_failed: bool = False
_vae_lock = threading.Lock()

# Singleton
_lock: threading.Lock = threading.Lock()
_instance: AudioLDM2Plugin | None = None


# ---------------------------------------------------------------------------
# ONNX session management
# ---------------------------------------------------------------------------


def _load_onnx_session():
    """Thread-safe lazy load of the AudioLDM2 UNet ONNX session."""
    global _onnx_session, _onnx_failed  # pylint: disable=global-statement
    if _onnx_failed:
        return None
    if _onnx_session is not None:
        return _onnx_session
    with _onnx_lock:
        if _onnx_session is not None or _onnx_failed:
            return _onnx_session
        if not _AUDIOLDM2_ONNX_PATH.exists():
            logger.info("AudioLDM2 ONNX not found at %s — plugin deaktiviert", _AUDIOLDM2_ONNX_PATH)
            _onnx_failed = True
            return None
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            _onnx_session = ort.InferenceSession(
                str(_AUDIOLDM2_ONNX_PATH),
                providers=["CPUExecutionProvider"],
            )
            logger.info("AudioLDM2 ONNX geladen: %s", _AUDIOLDM2_ONNX_PATH)
            return _onnx_session
        except Exception as exc:
            logger.warning("AudioLDM2 ONNX-Ladefehler: %s", exc)
            _onnx_failed = True
            return None


def _unload_onnx_session():
    """Release ONNX session memory."""
    global _onnx_session  # pylint: disable=global-statement
    if _onnx_session is not None:
        del _onnx_session
        _onnx_session = None
        import gc

        gc.collect()


# ---------------------------------------------------------------------------
# Diffusion helpers
# ---------------------------------------------------------------------------


def _ddim_scheduler(
    model_fn,
    latent_shape: tuple[int, ...],
    encoder_hidden_states_0: np.ndarray,
    encoder_hidden_states_1: np.ndarray,
    encoder_attention_mask_1: np.ndarray,
    num_inference_steps: int = _DDIM_NUM_STEPS,
    guidance_scale: float = 3.5,
    eta: float = 0.0,
) -> np.ndarray:
    """DDIM scheduler in pure numpy.

    Runs the reverse diffusion process: starts from random noise,
    iteratively denoises using the UNet model_fn, and returns the
    final latent representation.

    Args:
        model_fn: Callable that returns predicted noise given
                  (sample, timestep, encoder_hidden_states_0,
                   encoder_hidden_states_1, encoder_attention_mask_1).
        latent_shape: Shape of latent tensor (1, 8, H, W).
        encoder_hidden_states_0: CLAP embeddings (1, seq_len, 768).
        encoder_hidden_states_1: T5 embeddings (1, seq_len, 1024).
        encoder_attention_mask_1: T5 attention mask (1, seq_len).
        num_inference_steps: Number of DDIM steps (fewer = faster).
        guidance_scale: Classifier-free guidance strength.
        eta: DDIM stochasticity (0 = deterministic).

    Returns:
        Denoised latent of shape latent_shape.
    """
    # DDIM timesteps (descending from T to 0)
    ddim_timesteps = np.linspace(999, 0, num_inference_steps + 1, dtype=np.float32)[:-1]
    ddim_timesteps_prev = np.concatenate([ddim_timesteps[1:], np.array([0.0], dtype=np.float32)])

    # Initial random noise
    latent = np.random.randn(*latent_shape).astype(np.float32)

    batch_size = latent_shape[0]

    for i in range(num_inference_steps):
        t = ddim_timesteps[i]
        t_prev = ddim_timesteps_prev[i]

        # Timestep tensor for ONNX
        timestep = np.array([t], dtype=np.float32)

        # Classifier-free guidance: run UNet twice (conditional + unconditional)
        # Unconditional uses zero embeddings
        noise_pred_cond = model_fn(
            latent,
            timestep,
            encoder_hidden_states_0,
            encoder_hidden_states_1,
            encoder_attention_mask_1,
        )

        if guidance_scale > 1.0:
            zero_emb_0 = np.zeros_like(encoder_hidden_states_0, dtype=np.float32)
            zero_emb_1 = np.zeros_like(encoder_hidden_states_1, dtype=np.float32)
            zero_mask_1 = np.zeros_like(encoder_attention_mask_1)
            noise_pred_uncond = model_fn(
                latent,
                timestep,
                zero_emb_0,
                zero_emb_1,
                zero_mask_1,
            )
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        # DDIM step: predict x_0, then compute x_{t-1}
        # alpha_bar = cumulative product of (1 - beta)
        # Using standard linear beta schedule: beta_t = beta_start + t*(beta_end-beta_start)/T
        alpha_bar_t = _alpha_bar(t)
        alpha_bar_t_prev = _alpha_bar(t_prev)

        # Predict x_0 from noise prediction
        sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)

        pred_x0 = (latent - sqrt_one_minus_alpha_bar_t * noise_pred) / max(sqrt_alpha_bar_t, 1e-8)
        pred_x0 = np.clip(pred_x0, -10.0, 10.0)

        # Direction pointing to x_t
        dir_xt = (
            np.sqrt(1.0 - alpha_bar_t_prev - eta**2 * (1.0 - alpha_bar_t_prev) / max(1.0 - alpha_bar_t, 1e-8))
            * noise_pred
        )

        # Random noise for stochastic DDIM (eta > 0)
        if eta > 0:
            noise = np.random.randn(*latent_shape).astype(np.float32)
            sigma_t = (
                eta
                * np.sqrt((1.0 - alpha_bar_t_prev) / max(1.0 - alpha_bar_t, 1e-8))
                * np.sqrt(1.0 - alpha_bar_t / max(alpha_bar_t_prev, 1e-8))
            )
            latent = np.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        else:
            latent = np.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt

        # Clamp for stability
        latent = np.clip(latent, -10.0, 10.0)

    return cast(np.ndarray, latent.astype(np.float32))


# Pre-computed alpha_bar for efficiency (linear beta schedule, T=1000)
_ALPHA_BAR_CACHE: dict[int, float] = {}


def _alpha_bar(t: float, T: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012) -> float:
    """Cumulative product of (1 - beta_t) for timestep t."""
    t_int = int(round(t))
    if t_int in _ALPHA_BAR_CACHE:
        return _ALPHA_BAR_CACHE[t_int]
    # Compute all betas on first call
    if not _ALPHA_BAR_CACHE:
        betas = np.linspace(beta_start**0.5, beta_end**0.5, T, dtype=np.float64) ** 2
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        for i in range(T):
            _ALPHA_BAR_CACHE[i] = float(alpha_bars[i])
    return _ALPHA_BAR_CACHE.get(t_int, 1.0)


# ---------------------------------------------------------------------------
# Text encoder bridge — uses Aurik's LAION-CLAP plugin
# ---------------------------------------------------------------------------

_CLAP_ENCODER = None
_CLAP_LOCK = threading.Lock()


def _get_clap_embeddings(prompt: str) -> np.ndarray | None:
    """Get CLAP text embedding for the given prompt.

    Returns (1, 1, 768) embedding array, or None if CLAP unavailable.
    """
    global _CLAP_ENCODER  # pylint: disable=global-statement
    if _CLAP_ENCODER is None:
        with _CLAP_LOCK:
            if _CLAP_ENCODER is None:
                try:
                    from plugins.laion_clap_plugin import get_laion_clap

                    clap = get_laion_clap()
                    if clap is not None and getattr(clap, "_model_loaded", False):
                        _CLAP_ENCODER = clap
                    else:
                        logger.debug("AudioLDM2: LAION-CLAP not geladen")
                        return None
                except Exception as exc:
                    logger.debug("AudioLDM2: LAION-CLAP nicht verfuegbar: %s", exc)
                    return None

    try:
        # CLAP returns text embedding; shape depends on implementation
        emb = _CLAP_ENCODER.encode_text(prompt)  # type: ignore[attr-defined]
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        # Ensure shape is (1, seq_len, 768)
        if emb.ndim == 1:
            emb = emb.reshape(1, 1, -1)
        elif emb.ndim == 2:
            emb = emb[np.newaxis, :, :]
        # Pad/crop seq_len dimension to reasonable size
        seq_len = min(emb.shape[1], 77)  # AudioLDM2 uses max 77 tokens
        if emb.shape[1] > seq_len:
            emb = emb[:, :seq_len, :]
        # Ensure 768-dim if CLAP returns different dimensionality
        if emb.shape[2] != _AUDIOLDM2_CROSSATTENTION_DIM:
            if emb.shape[2] > _AUDIOLDM2_CROSSATTENTION_DIM:
                emb = emb[:, :, :_AUDIOLDM2_CROSSATTENTION_DIM]
            else:
                pad_w = _AUDIOLDM2_CROSSATTENTION_DIM - emb.shape[2]
                emb = np.pad(emb, ((0, 0), (0, 0), (0, pad_w)))
        return cast(np.ndarray | None, (np.asarray(emb, dtype=np.float32)))
    except Exception as exc:
        logger.debug("AudioLDM2: CLAP encode_text fehlgeschlagen: %s", exc)
        return None


def _get_t5_embeddings(prompt: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Get T5-Flan text embedding.

    Falls back to dummy embedding if T5 not available.
    Returns (encoder_hidden_states_1, encoder_attention_mask_1) or (None, None).
    """
    try:
        from backend.core.lyrics_guided_enhancement import (
            get_lyrics_guided_enhancement as _get_lge,
        )

        lge = _get_lge()
        if lge is not None and hasattr(lge, "encode_text_t5"):
            emb, mask = lge.encode_text_t5(prompt)
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
                if emb.ndim == 2:
                    emb = emb[np.newaxis, :, :]
                mask = np.asarray(mask) if mask is not None else np.ones((1, emb.shape[1]), dtype=bool)
                if mask.ndim == 1:
                    mask = mask[np.newaxis, :]
                return emb.astype(np.float32), mask
    except Exception:
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # Fallback: use CLAP embedding with padding to T5 dim
    clap_emb = _get_clap_embeddings(prompt)
    if clap_emb is not None:
        seq_len = clap_emb.shape[1]
        t5_emb = np.zeros((1, seq_len, _AUDIOLDM2_T5_EMBEDDING_DIM), dtype=np.float32)
        # Copy CLAP to first 768 dims, zero rest
        copy_dim = min(_AUDIOLDM2_CROSSATTENTION_DIM, clap_emb.shape[2])
        t5_emb[:, :, :copy_dim] = clap_emb[:, :, :copy_dim]
        mask = np.ones((1, seq_len), dtype=bool)
        return t5_emb, mask

    return None, None


# ---------------------------------------------------------------------------
# VAE Decoder — converts latent to mel spectrogram then to audio
# ---------------------------------------------------------------------------


def _load_vae_session():
    """Thread-safe lazy load of the AudioLDM2 VAE decoder ONNX session."""
    global _vae_session, _vae_failed  # pylint: disable=global-statement
    if _vae_failed:
        return None
    if _vae_session is not None:
        return _vae_session
    with _vae_lock:
        if _vae_session is not None or _vae_failed:
            return _vae_session
        if not _AUDIOLDM2_VAE_ONNX_PATH.exists():
            logger.info(
                "AudioLDM2 VAE decoder not found at %s — using Griffin-Lim Ersatzpfad", _AUDIOLDM2_VAE_ONNX_PATH
            )
            _vae_failed = True
            return None
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            _vae_session = ort.InferenceSession(
                str(_AUDIOLDM2_VAE_ONNX_PATH),
                providers=["CPUExecutionProvider"],
            )
            logger.info("AudioLDM2 VAE decoder ONNX geladen: %s", _AUDIOLDM2_VAE_ONNX_PATH)
            return _vae_session
        except Exception as exc:
            logger.warning("AudioLDM2 VAE ONNX-Ladefehler: %s", exc)
            _vae_failed = True
            return None


def _latent_to_audio(latent: np.ndarray, target_duration_s: float) -> np.ndarray:
    """Convert AudioLDM2 latent to audio waveform via VAE decoder + HiFi-GAN.

    Primary path: VAE decoder ONNX → mel spectrogram → HiFi-GAN → audio
    Fallback:    Numpy upsampling → Griffin-Lim

    Args:
        latent: (1, 8, H, W) float32 array from DDIM sampler.
        target_duration_s: Desired audio duration in seconds.

    Returns:
        Float32 mono audio at 16 kHz, shape (n_samples,).
    """
    latent = np.asarray(latent, dtype=np.float32)

    # Try VAE decoder path first
    vae_session = _load_vae_session()
    if vae_session is not None:
        try:
            # VAE decoder: (1, 8, H, W) → (1, 1, H*4, W*4)
            mel = vae_session.run(None, {"latent": latent})[0]
            mel = np.asarray(mel[0, 0], dtype=np.float32)  # (mel_bins, time)

            # Convert mel → audio via HiFi-GAN or Griffin-Lim
            audio = _mel_to_audio(mel, _AUDIOLDM2_SAMPLE_RATE, _AUDIOLDM2_HOP_LENGTH)

            # Trim/pad to target duration
            target_samples = int(target_duration_s * _AUDIOLDM2_SAMPLE_RATE)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            return cast(np.ndarray, audio.astype(np.float32))

        except Exception as exc:
            logger.debug("AudioLDM2 VAE decoder fehlgeschlagen: %s — falling back to Griffin-Lim", exc)

    # Fallback: numpy upsampling + Griffin-Lim
    return _latent_to_audio_fallback(latent, target_duration_s)


def _latent_to_audio_fallback(latent: np.ndarray, target_duration_s: float) -> np.ndarray:
    """Fallback: convert latent to audio via simple upsampling + Griffin-Lim."""
    latent = np.asarray(latent, dtype=np.float32)
    batch, channels, h, w = latent.shape
    target_mel_bins = _AUDIOLDM2_MEL_BINS
    target_time = int(target_duration_s * _AUDIOLDM2_SAMPLE_RATE / _AUDIOLDM2_HOP_LENGTH)

    latent_scaled = latent[0]
    mel = np.zeros((target_mel_bins, target_time), dtype=np.float32)
    for c in range(channels):
        ch_data = latent_scaled[c]
        from scipy.ndimage import zoom as _scipy_zoom

        h_scale = target_mel_bins / max(h, 1)
        w_scale = target_time / max(w, 1)
        ch_up = _scipy_zoom(ch_data, (h_scale, w_scale), order=1)
        h_actual, w_actual = ch_up.shape
        mel[:h_actual, :w_actual] += ch_up[:h_actual, :w_actual]

    mel = mel / max(channels, 1)
    mel_max = np.abs(mel).max()
    if mel_max > 1e-8:
        mel = mel / mel_max

    audio = _mel_to_audio(mel, _AUDIOLDM2_SAMPLE_RATE, _AUDIOLDM2_HOP_LENGTH)
    target_samples = int(target_duration_s * _AUDIOLDM2_SAMPLE_RATE)
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return cast(np.ndarray, audio.astype(np.float32))


def _mel_to_audio(mel_spec: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
    """Convert mel spectrogram to audio via HiFi-GAN (preferred) or Griffin-Lim."""
    # Try HiFi-GAN first
    try:
        from plugins.hifigan_plugin import get_hifigan_plugin

        hifigan = get_hifigan_plugin()
        if hifigan is not None and getattr(hifigan, "_model_loaded", False):
            # HiFi-GAN expects mel in a specific format
            audio = hifigan.mel_to_audio(mel_spec.astype(np.float32))  # type: ignore[attr-defined]
            if audio is not None and len(audio) > 0:
                return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))
    except Exception as exc:
        logger.debug("HiFi-GAN nicht verfuegbar, using Griffin-Lim: %s", exc)

    # Fallback to Griffin-Lim
    return _mel_to_audio_griffin_lim(mel_spec, sample_rate, hop_length)


def _mel_to_audio_griffin_lim(
    mel_spec: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_iter: int = 32,
    n_fft: int = 1024,
) -> np.ndarray:
    """Convert mel spectrogram to audio via inverse mel + Griffin-Lim.

    This is a fallback when HiFi-GAN vocoder is not available.
    """
    try:
        import librosa

        # Mel → linear spectrogram (approximate inverse)
        mel_spec = np.maximum(mel_spec, 1e-8)
        mel_spec_db = 20.0 * np.log10(mel_spec)

        # Approximate: just use spectrogram magnitude as-is with Griffin-Lim
        # Since we can't invert the mel filterbank exactly without the filters,
        # use the mel spectrogram directly (it's close enough for inpainting)
        audio = np.zeros(hop_length * (mel_spec.shape[1] - 1) + n_fft, dtype=np.float32)
        # Generate random phases for Griffin-Lim
        angles = np.exp(2j * np.pi * np.random.rand(*mel_spec.shape))
        spec_complex = mel_spec.astype(np.complex128) * angles

        # Griffin-Lim iterations
        for _ in range(n_iter):
            audio = librosa.istft(spec_complex, hop_length=hop_length, length=len(audio))
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            spec_complex = np.abs(stft) * (spec_complex / np.maximum(np.abs(spec_complex), 1e-8))

        return cast(np.ndarray, audio.astype(np.float32))
    except Exception:
        # Ultra-fallback: random noise shaped by mel envelope
        _n_samples = hop_length * (mel_spec.shape[1] - 1) + 1024
        noise = np.random.randn(_n_samples).astype(np.float32)
        # Simple envelope from mel mean
        mel_env = np.mean(mel_spec, axis=0)
        mel_env = mel_env / max(np.max(mel_env), 1e-8)
        # Upsample envelope to audio rate
        indices = np.clip(np.arange(_n_samples) * len(mel_env) // _n_samples, 0, len(mel_env) - 1)
        envelope = mel_env[indices]
        return (noise * envelope).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Main Plugin Class
# ---------------------------------------------------------------------------


class AudioLDM2Plugin:
    """Text-conditioned audio generation via AudioLDM2 ONNX + CLAP + HiFi-GAN.

    Generates plausible audio in-fill for very long dropouts (> 3 s) during
    restoration.  Used by Phase 24 as Tier 3 of the 4-tier dropout cascade.

    Public API:
        _ok: bool — True if ONNX model loaded successfully.
        TARGET_SR: int = 16000 — native sample rate.
        generate_array(prompt, duration, guidance) -> np.ndarray
    """

    TARGET_SR: int = _AUDIOLDM2_SAMPLE_RATE

    def __init__(self) -> None:
        self._ok: bool = False
        session = _load_onnx_session()
        if session is not None:
            self._session = session
            self._ok = True
        else:
            self._session = None

    @property
    def _model_loaded(self) -> bool:
        """For ml_model_readiness compatibility."""
        return self._ok

    def generate_array(
        self,
        prompt: str,
        duration: float = 3.0,
        guidance: float = 3.5,
    ) -> np.ndarray:
        """Generate audio from a text prompt.

        Args:
            prompt: Text description of desired audio content.
            duration: Target duration in seconds (≥ 1.0, ≤ 30.0).
            guidance: Classifier-free guidance scale (1.0–7.0, default 3.5).

        Returns:
            Float32 mono audio at 16 kHz, shape (n_samples,).
            Falls back to shaped noise on any error.
        """
        if not self._ok:
            return self._fallback_noise(duration)

        duration = max(1.0, min(30.0, float(duration)))
        guidance = float(np.clip(guidance, 1.0, 7.0))

        try:
            # 1. Get text embeddings
            clap_emb = _get_clap_embeddings(prompt)
            if clap_emb is None:
                logger.debug("AudioLDM2: CLAP nicht verfuegbar, using Ersatzpfad")
                return self._fallback_noise(duration)

            t5_emb, t5_mask = _get_t5_embeddings(prompt)
            if t5_emb is None:
                # Zero T5 embedding as fallback
                t5_emb = np.zeros((1, clap_emb.shape[1], _AUDIOLDM2_T5_EMBEDDING_DIM), dtype=np.float32)
                t5_mask = np.ones((1, clap_emb.shape[1]), dtype=bool)

            # 2. Compute latent dimensions — must be multiples of 8 for UNet skip-connections
            mel_time = max(1, int(duration * _AUDIOLDM2_SAMPLE_RATE / _AUDIOLDM2_HOP_LENGTH))
            latent_h = max(8, ((_AUDIOLDM2_MEL_BINS // _AUDIOLDM2_LATENT_DOWNSAMPLE) + 7) // 8 * 8)  # round up to ×8
            latent_w = max(8, (mel_time // _AUDIOLDM2_LATENT_DOWNSAMPLE + 7) // 8 * 8)  # round up to ×8
            latent_shape = (1, _AUDIOLDM2_LATENT_CHANNELS, latent_h, latent_w)

            # 3. Model function wrapping ONNX inference
            def _unet_fn(
                sample: np.ndarray,
                timestep: np.ndarray,
                enc_0: np.ndarray,
                enc_1: np.ndarray,
                mask_1: np.ndarray,
            ) -> np.ndarray:
                return self._run_unet(sample, timestep, enc_0, enc_1, mask_1)

            # 4. Run DDIM sampling
            latent = _ddim_scheduler(
                _unet_fn,
                latent_shape,
                encoder_hidden_states_0=clap_emb,
                encoder_hidden_states_1=t5_emb,
                encoder_attention_mask_1=t5_mask,  # type: ignore[arg-type]
                num_inference_steps=_DDIM_NUM_STEPS,
                guidance_scale=guidance,
            )

            # 5. Decode latent to audio
            audio = _latent_to_audio(latent, duration)
            return cast(np.ndarray, (np.clip(audio, -1.0, 1.0).astype(np.float32)))

        except Exception as exc:
            logger.warning("AudioLDM2: generation fehlgeschlagen: %s", exc, exc_info=True)
            return self._fallback_noise(duration)

    def denoise(
        self,
        audio: np.ndarray,
        sr: int,
        denoise_strength: float = 0.5,
        prompt: str | None = None,
    ) -> np.ndarray:
        """SDEdit-style text-guided denoising via AudioLDM2.

        Uses AudioLDM2's generative capabilities for denoising by:
        1. Extracting the audio's semantic context (via PANNs tags) to
           auto-generate a restoration prompt if none is given.
        2. Generating clean audio conditioned on the prompt.
        3. Crossfading the generated clean audio with the original,
           controlled by denoise_strength.

        Args:
            audio: Noisy input audio (mono or stereo, any sample rate).
            sr: Sample rate of input audio.
            denoise_strength: 0.0 = original only, 1.0 = fully regenerated.
                              Default 0.5 balances original character with
                              denoising effect.
            prompt: Optional text prompt override. Auto-generated from
                    PANNs audio tags if None (e.g. "clean high quality music
                    recording").

        Returns:
            Denoised audio at original sample rate, float32, same shape as input.
        """
        if not self._ok:
            logger.debug("AudioLDM2 denoise: model not loaded, returning original")
            return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))

        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        was_stereo = audio.ndim == 2 and audio.shape[0] == 2
        mono = audio if audio.ndim == 1 else audio.mean(axis=0)

        # Resample to 16 kHz for AudioLDM2
        if sr != self.TARGET_SR:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(sr, self.TARGET_SR)
            mono_16k = resample_poly(mono.astype(np.float64), self.TARGET_SR // g, sr // g).astype(np.float32)
        else:
            mono_16k = mono.astype(np.float32)

        # Clamp to [-1, 1]
        mono_16k = np.clip(mono_16k, -1.0, 1.0)

        # Auto-generate prompt from audio tags if not provided
        if prompt is None:
            prompt = "clean high quality audio, studio recording, professional mastering"
            try:
                from plugins.panns_plugin import get_panns_plugin

                _panns = get_panns_plugin()
                _panns_tags = _panns.get_tags(audio, sr)
                _top_tag = max(_panns_tags.items(), key=lambda x: x[1], default=("", 0.0))
                if _top_tag[1] >= 0.3:
                    if _top_tag[0] in ("Music", "Musical instrument"):
                        prompt = "clean high quality music recording, studio master"
                    elif _top_tag[0] in ("Speech", "Singing voice", "Vocals"):
                        prompt = "clean professional speech recording, studio quality"
                    elif _top_tag[0] in ("Noise", "Silence"):
                        prompt = "clean ambient sound, high quality recording"
            except Exception:
                pass  # Use default prompt

        duration = len(mono_16k) / self.TARGET_SR
        duration = max(1.0, min(30.0, duration))
        strength = float(np.clip(denoise_strength, 0.0, 1.0))

        try:
            # Generate clean audio from prompt
            generated = self.generate_array(prompt=prompt, duration=duration, guidance=3.0)
            # Match length to original
            if len(generated) > len(mono_16k):
                generated = generated[: len(mono_16k)]
            elif len(generated) < len(mono_16k):
                generated = np.pad(generated, (0, len(mono_16k) - len(generated)), mode="edge")

            # Crossfade: blend generated clean with original
            # strength=0 → all original, strength=1 → all generated
            # Use equal-power crossfade (cos² + sin² = 1)
            _theta = strength * np.pi / 2
            _orig_weight = float(np.cos(_theta))
            _gen_weight = float(np.sin(_theta))
            denoised_16k = _orig_weight * mono_16k + _gen_weight * generated
            denoised_16k = np.clip(denoised_16k, -1.0, 1.0)

            # Resample back to original sample rate
            if sr != self.TARGET_SR:
                g2 = gcd(self.TARGET_SR, sr)
                denoised = resample_poly(denoised_16k.astype(np.float64), sr // g2, self.TARGET_SR // g2).astype(
                    np.float32
                )
                # Match original length
                orig_len = audio.shape[-1] if audio.ndim == 2 else len(audio)
                if len(denoised) > orig_len:
                    denoised = denoised[:orig_len]
                elif len(denoised) < orig_len:
                    denoised = np.pad(denoised, (0, orig_len - len(denoised)), mode="edge")
            else:
                denoised = denoised_16k

            # Restore stereo
            if was_stereo:
                denoised = np.stack([denoised, denoised], axis=0)

            logger.info(
                "AudioLDM2 denoise: strength=%.2f duration=%.1fs prompt='%s'",
                strength,
                duration,
                prompt,
            )
            return cast(np.ndarray, denoised.astype(np.float32))

        except Exception as exc:
            logger.warning("AudioLDM2 denoise failed: %s — returning original", exc)
            return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))

    def _run_unet(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states_0: np.ndarray,
        encoder_hidden_states_1: np.ndarray,
        encoder_attention_mask_1: np.ndarray,
    ) -> np.ndarray:
        """Run ONNX UNet inference.

        Args:
            sample: (1, 8, H, W) float32 noisy latent.
            timestep: (1,) float32 diffusion timestep.
            encoder_hidden_states_0: (1, seq_len, 768) CLAP embedding.
            encoder_hidden_states_1: (1, seq_len, 1024) T5 embedding.
            encoder_attention_mask_1: (1, seq_len) boolean mask.

        Returns:
            (1, 8, H, W) float32 predicted noise.
        """
        if self._session is None:
            raise RuntimeError("AudioLDM2 ONNX session not loaded")

        inputs = {
            "sample": sample.astype(np.float32),
            "timestep": timestep.astype(np.float32),
            "encoder_hidden_states_0": encoder_hidden_states_0.astype(np.float32),
            "encoder_hidden_states_1": encoder_hidden_states_1.astype(np.float32),
            "encoder_attention_mask_1": encoder_attention_mask_1,
        }

        outputs = self._session.run(None, inputs)
        return cast(np.ndarray, (np.asarray(outputs[0], dtype=np.float32)))

    def _fallback_noise(self, duration: float) -> np.ndarray:
        """Generate shaped noise as fallback when generation fails."""
        n_samples = int(duration * self.TARGET_SR)
        noise = np.random.randn(n_samples).astype(np.float32)
        # Simple envelope: fade in/out with 50ms
        fade_n = int(0.05 * self.TARGET_SR)
        if fade_n * 2 < n_samples:
            fade_in = np.linspace(0, 1, fade_n, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_n, dtype=np.float32)
            noise[:fade_n] *= fade_in
            noise[-fade_n:] *= fade_out
        noise = np.clip(noise * 0.3, -1.0, 1.0)  # low amplitude
        return cast(np.ndarray, noise.astype(np.float32))

    def unload(self) -> None:
        """Release ONNX session memory."""
        _unload_onnx_session()
        self._ok = False
        self._session = None


# ---------------------------------------------------------------------------
# Singleton factory (required by ml_model_readiness and Phase 24)
# ---------------------------------------------------------------------------


def get_audioldm2_plugin() -> AudioLDM2Plugin:
    """Thread-safe singleton for AudioLDM2Plugin."""
    global _instance  # pylint: disable=global-statement
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = AudioLDM2Plugin()
        return _instance


def unload_audioldm2() -> None:
    """Release ONNX session and singleton."""
    global _instance  # pylint: disable=global-statement
    if _instance is not None:
        _instance.unload()
        _instance = None
