"""
§v10.121: KIM Music Enhancer Plugin — MDX23C-kompatibel (§B12).

Format (aus mdx23c_plugin.py reverse-engineered):
  Input:  STFT [B=1, 4, 3072, 256] — L-real, L-imag, R-real, R-imag
  Output: Maske [B=1, 4, 3072, 256] — auf Input multipliziert
  SR=44100, n_fft=6144, hop=1024, dim_t=256, overlap=128

Model: models/kim_inst/kim_inst.onnx (64 MB, vortrainiert)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_ROOT, "models", "kim_inst", "kim_inst.onnx")

SR = 44100
N_FFT = 6144
HOP = 1024
DIM_F = 3072
DIM_T = 256
OVERLAP = 128  # DIM_T // 2


def _get_session():
    import onnxruntime as ort

    _sess = getattr(_get_session, "_sess", None)
    if _sess is None:
        with _lock:
            _sess = getattr(_get_session, "_sess", None)
            if _sess is None:
                _sess = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
                setattr(_get_session, "_sess", _sess)
    return _sess


def _stft_stereo(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stereo STFT → (spec_L, spec_R) je [DIM_F, frames]."""
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    window = np.hanning(N_FFT).astype(np.float32)
    specs = []
    for ch in range(2):
        x = audio[ch].astype(np.float64)
        n_frames = (len(x) - N_FFT) // HOP + 1
        spec = np.zeros((N_FFT // 2 + 1, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            frame = x[i * HOP : i * HOP + N_FFT] * window
            spec[:, i] = np.fft.rfft(frame)
        specs.append(spec)
    return specs[0], specs[1]


def _istft_stereo(spec_L: np.ndarray, spec_R: np.ndarray, orig_len: int) -> np.ndarray:
    """Inverse STFT → Stereo."""
    window = np.hanning(N_FFT).astype(np.float32)
    out = np.zeros((2, orig_len), dtype=np.float32)
    for ch_idx, spec in enumerate([spec_L, spec_R]):
        audio = np.zeros(orig_len + N_FFT, dtype=np.float32)
        weight = np.zeros(orig_len + N_FFT, dtype=np.float32)
        for i in range(spec.shape[1]):
            frame = np.fft.irfft(spec[:, i], n=N_FFT).real
            start = i * HOP
            audio[start : start + N_FFT] += frame * window
            weight[start : start + N_FFT] += window**2
        audio = audio[:orig_len] / np.maximum(weight[:orig_len], 1e-8)
        out[ch_idx] = audio
    return cast(np.ndarray, out)


def enhance_music(audio: np.ndarray) -> np.ndarray:
    """Music enhancement via KIM Inst."""
    session = _get_session()
    spec_L, spec_R = _stft_stereo(audio)
    n_frames = spec_L.shape[1]

    out_L = np.zeros_like(spec_L, dtype=np.complex64)
    out_R = np.zeros_like(spec_R, dtype=np.complex64)

    pos = 0
    while pos < n_frames:
        end = min(pos + DIM_T, n_frames)
        sl_L = spec_L[:DIM_F, pos:end].copy()
        sl_R = spec_R[:DIM_F, pos:end].copy()

        # Pad time if needed
        pad_t = DIM_T - sl_L.shape[1]
        if pad_t > 0:
            sl_L = np.pad(sl_L, ((0, 0), (0, pad_t)), mode="edge")
            sl_R = np.pad(sl_R, ((0, 0), (0, pad_t)), mode="edge")

        # Build [1, 4, DIM_F, DIM_T]: L-real, L-imag, R-real, R-imag
        inp = np.stack(
            [
                sl_L.real.astype(np.float32),
                sl_L.imag.astype(np.float32),
                sl_R.real.astype(np.float32),
                sl_R.imag.astype(np.float32),
            ],
            axis=0,
        )[np.newaxis, :, :, :]

        mask = session.run(None, {"input": inp})[0]  # [1, 4, DIM_F, DIM_T]
        mask = np.squeeze(mask)  # [4, DIM_F, DIM_T]
        if mask.ndim == 2:
            mask = np.stack([mask, mask, mask, mask], axis=0)

        # Apply mask: multiply input spectrogram by mask
        mL_real = mask[0]
        mL_imag = mask[1]
        mR_real = mask[2]
        mR_imag = mask[3]

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        enhanced_L = (sl_L.real * mL_real - sl_L.imag * mL_imag) + 1j * (sl_L.real * mL_imag + sl_L.imag * mL_real)
        enhanced_R = (sl_R.real * mR_real - sl_R.imag * mR_imag) + 1j * (sl_R.real * mR_imag + sl_R.imag * mR_real)

        actual = min(DIM_T, n_frames - pos)
        out_L[:DIM_F, pos : pos + actual] = enhanced_L[:, :actual]
        out_R[:DIM_F, pos : pos + actual] = enhanced_R[:, :actual]

        pos += OVERLAP

    orig_len = audio.shape[-1] if audio.ndim == 2 else len(audio)
    return cast(np.ndarray, (_istft_stereo(out_L, out_R, orig_len).astype(np.float32)))
