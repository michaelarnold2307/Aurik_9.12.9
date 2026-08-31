#!/usr/bin/env python3
"""AERO-Plugin — Challenger-Kandidat: Bandbreiten-Extension 12 kHz → 48 kHz.

Bewertung gegen den Incumbent (FlashSR) über scripts/challenger_round.py auf
dem goldenen Hör-Set. NICHT in die Produktions-Routing-Pipeline verdrahtet —
Aufnahme erst nach bestandener Challenger-Runde (ADOPT).

Quelle: slp-rl/aero (MIT, vendored unter plugins/_vendor_aero/ mit LICENSE).
Checkpoint: models/aero/checkpoint_12-48_hl256.th (offizieller Google-Drive-Link
aus dem Upstream-README).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "_vendor_aero"
_MODEL_DIR = _ROOT.parent / "models" / "aero"
_CHECKPOINT = _MODEL_DIR / "checkpoint_12-48_hl256.th"
_LR_SR = 12000
_HR_SR = 48000
_SEGMENT_S = 10

_inst: AeroPlugin | None = None


class AeroPlugin:
    """AERO Super-Resolution (12 kHz → 48 kHz) als Challenger-Kandidat."""

    def __init__(self, checkpoint: Path | None = None, device: str = "cpu") -> None:
        self._model: Any = None
        self._device = device
        self.checkpoint = Path(checkpoint) if checkpoint else _CHECKPOINT
        self._try_load()

    def _try_load(self) -> None:
        if not self.checkpoint.exists():
            logger.warning(
                "AERO-Checkpoint fehlt: %s — Plugin ohne Modell (Challenger nicht lauffähig).",
                self.checkpoint,
            )
            return
        if str(_VENDOR) not in sys.path:
            sys.path.insert(0, str(_VENDOR))
        try:
            import torch
            from src.models.aero import Aero
        except Exception as exc:
            logger.warning("AERO-Vendor-Import fehlgeschlagen: %s", exc)
            return
        try:
            import inspect

            package = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            kwargs = dict(package["models"]["generator"].get("kwargs") or {})
            sig = inspect.signature(Aero.__init__).parameters
            kwargs = {k: v for k, v in kwargs.items() if k in sig and k != "self"}
            model = Aero(**kwargs)
            model.load_state_dict(package["models"]["generator"]["state"])
            model.eval()
            model.to(self._device)
            self._model = model
            logger.info("AERO 12-48 geladen (%s, device=%s)", self.checkpoint.name, self._device)
        except Exception as exc:
            logger.warning("AERO-Ladevorgang fehlgeschlagen: %s", exc)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def enhance(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """12 kHz → 48 kHz Bandbreiten-Extension. None bei fehlendem Modell."""
        if self._model is None:
            return None
        import torch

        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != _LR_SR:
            from scipy.signal import resample_poly

            g = int(np.gcd(sr, _LR_SR))
            audio = resample_poly(audio, _LR_SR // g, sr // g).astype(np.float32)
        seg = _SEGMENT_S * _LR_SR
        chunks = [audio[i : i + seg] for i in range(0, len(audio), seg)]
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for ch in chunks:
                # Checkpoint ist mono trainiert (audio_channels=1):
                # Input (B=1, C=1, T), Output (1, 1, T_hr).
                mono = torch.from_numpy(ch.astype(np.float32))
                x = mono.unsqueeze(0).unsqueeze(0).to(self._device)
                out = self._model(x).squeeze(0).squeeze(0).cpu().numpy()
                outs.append(out.astype(np.float32))
        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(out))) if out.size else 1.0
        if peak > 1.0:
            out = out / peak
        return cast(np.ndarray | None, out[: int(round(len(audio) * (_HR_SR / _LR_SR)))])


def get_aero_plugin(device: str = "cpu") -> AeroPlugin:
    """Thread-sicherer Singleton für Challenger-Runs."""
    global _inst
    if _inst is None:
        _inst = AeroPlugin(device=device)
    return _inst
