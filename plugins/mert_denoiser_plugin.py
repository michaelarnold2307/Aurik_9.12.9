#!/usr/bin/env python3
"""
Aurik: MERT Denoiser Plugin.
§v10.127: MERT (117M) features → Decoder (3.5M) → Clean Audio Mask.

Architecture:
  - MERT Feature Extractor (ONNX, frozen, 117M): 48kHz→16kHz→768-dim features.
  - Decoder (ONNX, 3.5M): STFT + MERT features → masking filter.
  - Total: 120.5M params, 3.5M trainable.

Training: 8,572 music files (FMA-Small + MUSDB18 + MTG-Jamendo).
Best Val Loss: 0.0001 (normalized STFT MSE).
"""

import io
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Union, cast

import numpy as np
import onnxruntime as ort
import soundfile as sf

log = logging.getLogger(__name__)

# Constants
MERT_SAMPLE_RATE = 16000
TARGET_SAMPLE_RATE = 48000
N_FFT = 960
HOP_LEN = 480  # 10ms @ 48kHz
MERT_HOP = 160  # 10ms @ 16kHz
CHUNK_DURATION = 2.0  # seconds
CHUNK_SAMPLES_48K = int(CHUNK_DURATION * TARGET_SAMPLE_RATE)
CHUNK_SAMPLES_16K = int(CHUNK_DURATION * MERT_SAMPLE_RATE)
DECODER_CONTEXT = 0.5  # seconds of MERT context on each side
CONTEXT_FRAMES_16K = int(DECODER_CONTEXT * MERT_SAMPLE_RATE / MERT_HOP)


class MERTDenoiserPlugin:
    """Denoises audio using MERT features + trained decoder."""

    def __init__(
        self,
        mert_onnx: str = "models/mert/mert.onnx",
        decoder_onnx: str = "models/mert_denoiser/mert_decoder.onnx",
        chunk_duration: float = 2.0,
        device: str = "cuda",
        gpu_id: int = 0,
        enable: bool = True,
    ):
        self.enabled = enable
        self.gpu_id = gpu_id
        self.chunk_duration = chunk_duration
        if not enable:
            return

        providers = ["ROCMExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        provider_options = [{"device_id": str(gpu_id)}, {}] if device == "cuda" else []

        # Load MERT ONNX
        mert_path = Path(mert_onnx)
        if not mert_path.exists():
            raise FileNotFoundError(f"MERT ONNX not found: {mert_path}")
        self.mert_session = ort.InferenceSession(
            str(mert_path),
            providers=providers,
            provider_options=provider_options,
        )
        log.info(f"MERT Denoiser: MERT loaded from {mert_path}")

        # Load Decoder ONNX
        decoder_path = Path(decoder_onnx)
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder ONNX not found: {decoder_path}")
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            providers=["CPUExecutionProvider"],  # Decoder is lightweight, CPU is fine
        )
        log.info(f"MERT Denoiser: Decoder loaded from {decoder_path}")

        # Window for STFT/ISTFT
        self.window = np.hanning(N_FFT).astype(np.float32)
        self.frame_samples = CHUNK_SAMPLES_48K // HOP_LEN  # 200 frames per chunk

        log.info("MERT Denoiser: Initialized")

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise audio in chunks with 50% overlap, using MERT features + decoder."""
        if not self.enabled or len(audio) == 0:
            return audio

        start_time = time.time()

        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        n_channels, total_samples = audio.shape

        # Process each channel independently
        outputs = []
        for ch in range(n_channels):
            output = self._process_channel(audio[ch])
            outputs.append(output)

        result = np.stack(outputs)
        if n_channels == 1:
            result = result[0]

        elapsed = time.time() - start_time
        log.debug(f"MERT Denoiser: {total_samples / TARGET_SAMPLE_RATE:.1f}s in {elapsed:.1f}s")
        return cast(np.ndarray, result)

    def _process_channel(self, audio: np.ndarray) -> np.ndarray:
        """Process a single channel with overlapping chunks."""
        total = len(audio)
        hop = CHUNK_SAMPLES_48K // 2  # 50% overlap
        window = self._overlap_window(CHUNK_SAMPLES_48K, hop)
        output = np.zeros(total, dtype=np.float32)
        weight = np.zeros(total, dtype=np.float32)

        pos = 0
        while pos < total:
            end = min(pos + CHUNK_SAMPLES_48K, total)
            chunk = audio[pos:end]
            if len(chunk) < CHUNK_SAMPLES_48K:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES_48K - len(chunk)))

            # Process chunk
            clean_chunk = self._denoise_chunk(chunk)
            clean_chunk = clean_chunk[:CHUNK_SAMPLES_48K]

            # Overlap-add
            chunk_len = min(len(clean_chunk), total - pos)
            w = window[:chunk_len]
            output[pos : pos + chunk_len] += clean_chunk[:chunk_len] * w
            weight[pos : pos + chunk_len] += w
            pos += hop

        # Normalize by overlap weight
        weight[weight < 1e-8] = 1.0
        output /= weight
        return cast(np.ndarray, output)

    def _denoise_chunk(self, audio_48k: np.ndarray) -> np.ndarray:
        """Denoise a single chunk: MERT features → Decoder → clean spectrogram → audio."""
        # 1. Downsample to 16kHz for MERT
        audio_16k = self._resample(audio_48k, TARGET_SAMPLE_RATE, MERT_SAMPLE_RATE)

        # 2. Extract MERT features (ONNX)
        mert_input = audio_16k[np.newaxis, :]  # [1, T_16k]
        mert_output = self.mert_session.run(None, {"input_values": mert_input})[0]
        # mert_output: [1, T_m, 768] — check and transpose if needed
        if mert_output.shape[1] == 768 and mert_output.shape[2] != 768:
            mert_output = mert_output.transpose(0, 2, 1)  # [1, T_m, 768]

        mert_feat = mert_output  # [1, T_m, 768]

        # 3. STFT of 48kHz noisy audio
        n_frames = 1 + (len(audio_48k) - N_FFT) // HOP_LEN
        spec = np.zeros((n_frames, N_FFT // 2 + 1), dtype=np.complex64)
        for i in range(n_frames):
            start = i * HOP_LEN
            frame = audio_48k[start : start + N_FFT] * self.window
            spec[i] = np.fft.rfft(frame)

        # Build real/imag input [1, 2, F, T]
        F = N_FFT // 2 + 1
        T_frames = n_frames
        spec_ri = np.zeros((1, 2, F, T_frames), dtype=np.float32)
        spec_ri[0, 0] = spec.T.real
        spec_ri[0, 1] = spec.T.imag

        # 4. Run decoder → enhanced spectrogram [1, 2, F, T]
        enhanced = self.decoder_session.run(
            None,
            {"spec_ri": spec_ri, "mert_features": mert_feat},
        )[0]

        # 5. Reconstruct complex spectrum and ISTFT
        enhanced_spec = enhanced[0, 0] + 1j * enhanced[0, 1]  # [F, T] complex
        enhanced_spec = enhanced_spec.T  # [T, F]

        # ISTFT
        output = self._istft(enhanced_spec, len(audio_48k))
        return output

    def _istft(self, stft_matrix: np.ndarray, output_length: int) -> np.ndarray:
        """Inverse STFT with Hann window, overlap-add."""
        n_frames, n_freq = stft_matrix.shape
        hop = HOP_LEN
        result = np.zeros(output_length, dtype=np.float32)
        window_sq = self.window**2 if np.sum(self.window**2) > 0 else self.window
        norm_window = np.zeros(output_length, dtype=np.float32)

        for i in range(n_frames):
            start = i * hop
            frame = np.fft.irfft(stft_matrix[i])
            frame = frame * self.window
            frame = frame[:N_FFT]
            end = min(start + N_FFT, output_length)
            result[start:end] += frame[: end - start]
            norm_window[start:end] += window_sq[: end - start]

        norm_window[norm_window < 1e-8] = 1.0
        result /= norm_window
        return cast(np.ndarray, result)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear resampling — adequate for non-audiophile quality."""
        if orig_sr == target_sr:
            return audio
        from scipy import signal as scipy_signal

        # Use polyphase resampling for quality
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return cast(np.ndarray, (scipy_signal.resample_poly(audio, up, down).astype(np.float32)))

    @staticmethod
    def _overlap_window(chunk_size: int, hop: int) -> np.ndarray:
        """Create overlapping Hann windows for smooth chunk blending."""
        window = np.hanning(chunk_size)
        return cast(np.ndarray, window**2)  # Hann^2 for perfect reconstruction with 50% overlap

    # ── Static helpers for compatibility ──

    @staticmethod
    def load_audio(data: str | bytes | np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
        """Load audio from file path, bytes, or numpy array.

        §11 VERBOTEN: sf.read direkt — kanonischer Import über backend.file_import.
        """
        from backend.file_import import load_audio_file

        if isinstance(data, np.ndarray):
            return data, sample_rate
        if isinstance(data, bytes):
            # load_audio_file() erwartet einen echten Dateipfad (os.path.isfile-Check) —
            # BytesIO ist hier nicht kompatibel, daher über eine Temp-Datei laden.
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                loaded = load_audio_file(tmp_path, target_sr=sample_rate, mono=False, do_carrier_analysis=False)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            loaded = load_audio_file(data, target_sr=sample_rate, mono=False, do_carrier_analysis=False)
        if loaded and loaded.get("audio") is not None:
            return np.asarray(loaded["audio"], dtype=np.float32), int(loaded.get("sr") or sample_rate)
        raise ValueError("Audio konnte nicht geladen werden (load_audio_file leer)")

    @staticmethod
    def save_audio(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
        """Save audio to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        return buf.getvalue()
