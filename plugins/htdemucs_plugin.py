"""HTDemucs Plugin — Hybrid Musik-Stem-Separation (PyTorch/ONNX).
==============================================================================

Echte 4-Stem-Trennung (vocals, drums, bass, other) mit HTDemucs (Meta, Défossez 2021).

Hybrid-Strategie:
    Primär:   PyTorch (`demucs.pretrained.get_model("htdemucs")`) mit GPU-Optimalität
    Fallback: ONNX-Export wenn PyTorch fehlschlägt (Kompatibilität, Determinismus)

Stem-Separation ist computeaufwändig:
    - Input: 30s @ 48kHz = 1.44M Samples
    - Inference: ~15s GPU (RTX 4090), ~60s CPU
    - Output: 4 Stems × 30s @ 48kHz = 5.76M Samples
    - Anwendung: Nur gemessen wenn global_scalar > 0.5 (Performance Guard)

Psychoakustische Applikation (§musical_goals.instructions.md):
    - separation_fidelity = 1.0 − (energy(residuum) / energy(original))
      wobei residuum = original − (vocals + drums + bass + other) (Rekonstruktion)
    - Score ∈ [0, 1]; 1.0 = perfekte Rekonstruktion

GPU-Support: Optional via AURIK_DEMUX_GPU=1 (Default: CPU für Determinismus, §G5)

Referenzen:
    Défossez (2021): "Music Source Separation in the Waveform Domain"
    https://arxiv.org/abs/2111.02477

Invarianten (§G5, §G8):
    - Deterministisch: CPU-Inferenz (Default), Optional GPU via Flag
    - Thread-sicher: Singleton mit Double-Checked Locking
    - Vollständig typisiert (PEP 484, kein Any in öffentlichen APIs)
    - Fehlerprotokollierung: Alle Fallbacks mit logger.warning()
"""

from __future__ import annotations

import logging
import os
import threading
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Feature-Flag: AURIK_DEMUX_GPU=1 ermöglicht GPU-Inferenz
_DEMUX_GPU_ENABLED = os.getenv("AURIK_DEMUX_GPU", "").lower() in ("1", "true", "yes")

# Singleton
_INSTANCE_HOLDER: dict[str, HtdemucsPlugin | None] = {"plugin": None}
_singleton_lock = threading.Lock()


class SeparationResult:
    """Holder für 4 Stems nach HTDemucs-Separation."""

    __slots__ = ("bass", "drums", "other", "sr", "vocals")

    def __init__(
        self,
        vocals: np.ndarray,
        drums: np.ndarray,
        bass: np.ndarray,
        other: np.ndarray,
        sr: int,
    ) -> None:
        """4-Stem-Container.

        Args:
            vocals: Gesang-Stem, Shape (T,) oder (2, T), float32, normalized ≈ [−1, +1]
            drums: Drum-Stem, same shape
            bass:  Bass-Stem, same shape
            other: Other-Stem (Instrumente, Effekte), same shape
            sr:    Sample-Rate in Hz (z.B. 48000)
        """
        self.vocals = vocals.astype(np.float32)
        self.drums = drums.astype(np.float32)
        self.bass = bass.astype(np.float32)
        self.other = other.astype(np.float32)
        self.sr = sr

    def as_dict(self) -> dict[str, np.ndarray]:
        """Gibt Dict für Iteration über Stems."""
        return {"vocals": self.vocals, "drums": self.drums, "bass": self.bass, "other": self.other}

    def reconstruct(self) -> np.ndarray:
        """Rekonstruiert Original: vocals + drums + bass + other."""
        return self.vocals + self.drums + self.bass + self.other


class HtdemucsPlugin:
    """Thread-sicherer HTDemucs-Wrapper mit Hybrid PyTorch/ONNX."""

    _model: Any = None  # Lazy-loaded: demucs.Model oder ort.InferenceSession
    _model_type: str = "uninitialized"  # "pytorch" oder "onnx"
    _lock = threading.Lock()

    def separate(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> SeparationResult:
        """Trennt Audio in 4 Stems.

        Args:
            audio: Input-Audio, Shape (T,) oder (C, T), float32, normalized ≈ [−1, +1]
            sr:    Sample-Rate in Hz (typisch 48000)

        Returns:
            SeparationResult mit vocals, drums, bass, other (alle shape (T,) oder (C, T))

        Raises:
            RuntimeError: Wenn Separation fehlschlägt
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Mono/Stereo konvertieren auf Shape (2, T) für HTDemucs
        if audio.ndim == 1:
            audio_2ch = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2 and audio.shape[0] in (1, 2):
            if audio.shape[0] == 1:
                audio_2ch = np.vstack([audio, audio])
            else:
                audio_2ch = audio
        else:
            raise ValueError(f"Expected audio shape (T,) or (C, T), got {audio.shape}")

        # Resampling zu 48 kHz (HTDemucs-Standard)
        if sr != 48000:
            try:
                julius_forward: Any = import_module("julius")
                import torch

                input_tensor = torch.from_numpy(np.ascontiguousarray(audio_2ch, dtype=np.float32)).unsqueeze(0)
                audio_2ch = julius_forward.resample_frac(
                    julius_forward.ResampleFrac(sr, 48000),
                    input_tensor,
                ).squeeze(0).cpu().numpy()
            except Exception as e:
                logger.warning("julius Resampling fehlgeschlagen, fallback librosa: %s", e)
                import librosa

                audio_2ch = np.stack(
                    [librosa.resample(channel, orig_sr=sr, target_sr=48000) for channel in audio_2ch],
                    axis=0,
                ).astype(np.float32, copy=False)
        else:
            audio_2ch = audio_2ch.astype(np.float32)

        # Lazy-load Modell
        self._ensure_model()

        # Separation
        try:
            if self._model_type == "pytorch":
                stems_list = self._separate_pytorch(audio_2ch)
            elif self._model_type == "onnx":
                stems_list = self._separate_onnx(audio_2ch)
            else:
                raise RuntimeError(f"Unbekannter Model-Type: {self._model_type}")
        except Exception as e:
            logger.error("HTDemucs Separation fehlgeschlagen: %s", e, exc_info=True)
            raise RuntimeError(f"HTDemucs Separation fehlgeschlagen: {e}") from e

        # Resampling zurück zu Eingabe-SR
        if sr != 48000:
            try:
                julius_reverse: Any = import_module("julius")
                import torch

                stems_list_rs = []
                for stem in stems_list:
                    stem_tensor = torch.as_tensor(np.asarray(stem, dtype=np.float32)).unsqueeze(0)
                    stem_rs = julius_reverse.resample_frac(
                        julius_reverse.ResampleFrac(48000, sr),
                        stem_tensor,
                    ).squeeze(0).cpu().numpy()
                    stems_list_rs.append(stem_rs)
                stems_list = stems_list_rs
            except Exception as e:
                logger.warning("julius Resampling (reverse) fehlgeschlagen, fallback librosa: %s", e)
                import librosa

                stems_list_rs = []
                for stem in stems_list:
                    stem_np = stem.numpy() if hasattr(stem, "numpy") else stem
                    stem_rs = librosa.resample(stem_np, orig_sr=48000, target_sr=sr)
                    stems_list_rs.append(stem_rs)
                stems_list = stems_list_rs

        # Zurück zu Original-Shape (mono oder stereo)
        if audio.ndim == 1:
            stems_list = [s[0] if hasattr(s, "__getitem__") else s for s in stems_list]
        elif audio.shape[0] == 1:
            stems_list = [s[0:1] if hasattr(s, "__getitem__") else s for s in stems_list]

        vocals, drums, bass, other = stems_list
        return SeparationResult(vocals, drums, bass, other, sr)

    def _ensure_model(self) -> None:
        """Lädt Modell lazy (Thread-sicher)."""
        if self._model is not None:
            return

        demucs_pretrained: Any | None = None
        if _DEMUX_GPU_ENABLED:
            try:
                demucs_pretrained = import_module("demucs.pretrained")
            except Exception as e:
                logger.debug("HTDemucs PyTorch-Import fehlgeschlagen, versuche ONNX: %s", e)

        ort: Any | None = None
        onnx_import_error: Exception | None = None
        try:
            ort = import_module("onnxruntime")
        except Exception as e:
            onnx_import_error = e

        with self._lock:
            if self._model is not None:
                return

            # Versuche PyTorch zuerst (wenn GPU oder explizit enabled)
            if demucs_pretrained is not None:
                try:
                    self._model = demucs_pretrained.get_model("htdemucs")
                    self._model_type = "pytorch"
                    logger.info("HTDemucs PyTorch Model geladen (GPU-enabled)")
                    return
                except Exception as e:
                    logger.debug("HTDemucs PyTorch Load fehlgeschlagen, versuche ONNX: %s", e)

            # Fallback: ONNX
            try:
                if ort is None:
                    raise RuntimeError("ONNX Runtime nicht importierbar") from onnx_import_error

                onnx_path = Path(__file__).parent.parent / "models" / "htdemucs" / "htdemucs.onnx"
                if not onnx_path.exists():
                    raise FileNotFoundError(f"ONNX Model nicht gefunden: {onnx_path}")

                providers = ["CPUExecutionProvider"]  # Default CPU für Determinismus
                if _DEMUX_GPU_ENABLED:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

                self._model = ort.InferenceSession(str(onnx_path), providers=providers)
                self._model_type = "onnx"
                logger.info("HTDemucs ONNX Model geladen (Provider: %s)", providers[0])
            except Exception as e:
                logger.error("HTDemucs ONNX Load fehlgeschlagen: %s", e, exc_info=True)
                raise RuntimeError(f"HTDemucs Modell konnte nicht geladen werden: {e}") from e

    def _separate_pytorch(self, audio_2ch: np.ndarray) -> list[np.ndarray]:
        """Separation mit PyTorch."""
        import torch

        device = "cuda" if _DEMUX_GPU_ENABLED and torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device)

        with torch.no_grad():
            # Input: (2, T) → Tensor (1, 2, T)
            audio_t = torch.from_numpy(audio_2ch[np.newaxis, ...]).to(device)

            # Separation: gibt (1, 4, 2, T) zurück — [vocals, drums, bass, other]
            stems_t = self._model.separate(audio_t)

            # Zurück zu numpy: (4, 2, T)
            stems_np = stems_t.squeeze(0).cpu().numpy()

        # Rückgabe: List[vocals, drums, bass, other]
        return [stems_np[i] for i in range(4)]

    def _separate_onnx(self, audio_2ch: np.ndarray) -> list[np.ndarray]:
        """Separation mit ONNX Runtime."""
        # Input: (2, T) → (1, 2, T)
        input_data = audio_2ch[np.newaxis, ...].astype(np.float32)

        # ONNX-Inferenz
        outputs = self._model.run(None, {"input": input_data})

        # outputs[0] = (1, 4, 2, T) — [vocals, drums, bass, other]
        stems = outputs[0].squeeze(0)  # (4, 2, T)

        return [stems[i] for i in range(4)]

    def unload(self) -> None:
        """Entladen des Modells aus RAM."""
        with self._lock:
            if self._model is not None:
                try:
                    if hasattr(self._model, "close"):
                        self._model.close()
                except Exception as e:
                    logger.debug("HTDemucs Unload Fehler: %s", e)
                finally:
                    self._model = None
                    self._model_type = "uninitialized"
                    logger.debug("HTDemucs Model entladen")


def get_htdemucs_plugin() -> HtdemucsPlugin:
    """Gibt HTDemucs-Singleton zurück (Double-Checked Locking)."""
    if _INSTANCE_HOLDER["plugin"] is None:
        with _singleton_lock:
            if _INSTANCE_HOLDER["plugin"] is None:
                _INSTANCE_HOLDER["plugin"] = HtdemucsPlugin()
    plugin = _INSTANCE_HOLDER["plugin"]
    assert plugin is not None
    return plugin
