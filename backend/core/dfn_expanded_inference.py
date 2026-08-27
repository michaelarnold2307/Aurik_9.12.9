"""
§v10.123: DFN Expanded Inference — PyTorch GPU Denoiser mit besten Gewichten.

Wrapper für DeepFilterNet3 mit dfn_expanded_best.pt (Val Loss 0.298, 7.0 dB SNR).
Läuft auf PyTorch GPU (ONNX-Export durch complex ops blockiert).

Nutzung:
    from backend.core.dfn_expanded_inference import DFNExpandedDenoiser
    denoiser = DFNExpandedDenoiser()
    clean = denoiser.denoise(noisy_audio, sample_rate)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "models" / "deepfilternet_v3_ii" / "DeepFilterNet"))
sys.path.insert(0, str(_PROJECT / "models" / "deepfilternet_v3_ii" / "pyDF-data"))

SR = 48_000
N_FFT = 960
HOP = 480
N_ERB = 32
DF_BINS = 96
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)

_CHECKPOINT = _PROJECT / "models" / "deepfilternet_v3_ii" / "finetuned" / "dfn_expanded_best.pt"


class DFNExpandedDenoiser:
    """PyTorch GPU Denoiser mit DFN Expanded Gewichten (0.298 Val Loss)."""

    def __init__(self):
        from df.config import config

        config.use_defaults()
        from df.deepfilternet3 import init_model

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = init_model().to(self._device).eval()

        ckpt = torch.load(str(_CHECKPOINT), map_location=self._device, weights_only=True)
        self._model.load_state_dict(ckpt["model_state_dict"])
        logger.info("DFN Expanded geladen: epoch %s, val_loss=%.4f", ckpt.get("epoch", "?"), ckpt.get("val_loss", 0.0))

        self._window = torch.hann_window(N_FFT, device=self._device)
        self._erb_fb = torch.from_numpy(self._build_erb()).to(self._device)

    @staticmethod
    def _build_erb() -> np.ndarray:
        n_bins = N_FFT // 2 + 1
        freqs = np.linspace(0, SR / 2, n_bins)

        def hz2erb(f):
            return 21.4 * np.log10(1.0 + f / 229.0 + 1e-9)

        erb_max = hz2erb(np.array([SR / 2]))[0]
        edges = np.linspace(hz2erb(np.array([0.0]))[0], erb_max, N_ERB + 1)
        fb = np.zeros((N_ERB, n_bins), dtype=np.float32)
        for b in range(N_ERB):
            lo, hi = edges[b], edges[b + 1]
            mask = (hz2erb(freqs) >= lo) & (hz2erb(freqs) < hi)
            if mask.sum() > 0:
                fb[b, mask] = 1.0 / mask.sum()
        return fb

    def _extract_features(self, audio: torch.Tensor) -> tuple:
        spec = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, window=self._window, return_complex=True)
        mag = spec.abs()
        erb_e = torch.matmul(self._erb_fb, mag)
        feat_erb = torch.log1p(erb_e).unsqueeze(1).transpose(2, 3)
        spec96 = spec[:, :DF_BINS, :]
        feat_spec = torch.stack([spec96.real, spec96.imag], dim=-1)
        feat_spec = feat_spec.permute(0, 2, 1, 3).unsqueeze(1)
        full_spec = torch.stack([spec.real, spec.imag], dim=-1)
        full_spec = full_spec.permute(0, 2, 1, 3).unsqueeze(1)
        return feat_erb, feat_spec, full_spec

    def denoise(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Entrauscht Audio mit DFN Expanded Gewichten.

        Args:
            audio: float32 [samples] oder [channels, samples]
            sample_rate: Sample-Rate (wird auf 48kHz resampelt)

        Returns:
            Denoised audio, gleiche Form wie Input
        """
        import librosa

        is_stereo = audio.ndim == 2 and audio.shape[0] == 2

        if is_stereo:
            left = self._denoise_mono(audio[0], sample_rate)
            right = self._denoise_mono(audio[1], sample_rate)
            return np.stack([left, right], axis=0)

        return self._denoise_mono(audio if audio.ndim == 1 else audio[0], sample_rate)

    def _denoise_mono(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != SR:
            import librosa

            audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=SR)

        orig_len = len(audio)
        audio = audio.astype(np.float32)

        # Process in chunks for long audio
        if orig_len <= CHUNK_SAMPLES:
            return self._process_chunk(audio, orig_len)

        hop = CHUNK_SAMPLES // 2
        result = np.zeros(orig_len, dtype=np.float32)
        weight = np.zeros(orig_len, dtype=np.float32)
        window = np.hanning(CHUNK_SAMPLES).astype(np.float32)

        for pos in range(0, orig_len, hop):
            end = min(pos + CHUNK_SAMPLES, orig_len)
            chunk = audio[pos:end].copy()
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode="reflect")
            processed = self._process_chunk(chunk, CHUNK_SAMPLES)
            w = window[: len(processed)]
            result[pos : pos + len(processed)] += processed * w
            weight[pos : pos + len(processed)] += w

        weight = np.maximum(weight, 1e-8)
        return (result / weight).astype(np.float32)

    def _process_chunk(self, audio: np.ndarray, target_len: int) -> np.ndarray:
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)
        feb, fsp, spec = self._extract_features(audio_t)

        with torch.no_grad():
            enh, _, _, _ = self._model.forward(spec=spec, feat_erb=feb, feat_spec=fsp)

        enhanced = torch.complex(enh[0, 0, :, :, 0], enh[0, 0, :, :, 1]).T.unsqueeze(0)
        out = torch.istft(enhanced, n_fft=N_FFT, hop_length=HOP, window=self._window, length=target_len)
        return out.cpu().numpy().squeeze()
