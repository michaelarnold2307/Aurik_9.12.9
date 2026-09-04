"""
backend/core/spectrogram_snapshot.py — Per-Phase Spektrogramm-Snapshot (§v10.10)
================================================================================

Speichert 2s STFT-Magnitude als 256×256 Graustufen-PNG pro Phase.
Pfad: .aurik/snapshots/{run_id}/{phase_id}_pre.png und _post.png

Usage:
    from backend.core.spectrogram_snapshot import SpectrogramSnapshotter
    snap = SpectrogramSnapshotter(run_id="20260719_001")
    snap.capture("phase_03_denoise", audio_pre, audio_post, sr)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SNAPSHOT_DIR = Path.home() / ".aurik" / "snapshots"
_SNAPSHOT_SECONDS: float = 2.0  # 2s Ausschnitt (Mitte des Audios)
_SNAPSHOT_SIZE: int = 256  # 256×256 Pixel
_N_FFT: int = 2048
_HOP: int = 512


class SpectrogramSnapshotter:
    """Erfasst und speichert Spektrogramm-Snapshots für Debugging."""

    def __init__(self, run_id: str) -> None:
        self._run_dir = _SNAPSHOT_DIR / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        phase_id: str,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Speichert Pre/Post-Spektrogramme für eine Phase."""
        try:
            _pre_png = self._render_spectrogram(audio_pre, sample_rate)
            _post_png = self._render_spectrogram(audio_post, sample_rate)
            _pre_png.save(self._run_dir / f"{phase_id}_pre.png", "PNG")
            _post_png.save(self._run_dir / f"{phase_id}_post.png", "PNG")
            _diff = np.abs(np.asarray(_pre_png, dtype=np.float32) - np.asarray(_post_png, dtype=np.float32)).mean()
            logger.debug(
                "📸 Spek-Snapshot %s: pre/post gespeichert (Diff=%.1f)",
                phase_id,
                _diff,
            )
        except Exception as exc:
            logger.debug("Spektrogramm-Snapshot fehlgeschlagen: %s", exc)

    def _render_spectrogram(self, audio: np.ndarray, sr: int):
        """Rendert STFT-Magnitude als 256×256 Graustufen-Bild (PIL Image)."""
        try:
            from PIL import Image
        except Exception as e:
            logger.debug("§V6 PIL-Import fehlgeschlagen — Dummy-Bild zurückgegeben: %s", e)
            return _dummy_image()

        _mono = audio
        if _mono.ndim > 1:
            _mono = _mono.mean(axis=0) if _mono.shape[0] <= 2 else _mono.mean(axis=1)

        # 2s aus der Mitte
        _center = len(_mono) // 2
        _half = int(_SNAPSHOT_SECONDS * sr / 2)
        _start = max(0, _center - _half)
        _end = min(len(_mono), _center + _half)
        _segment = _mono[_start:_end]

        if len(_segment) < _N_FFT:
            return _dummy_image()

        # STFT
        try:
            from scipy.signal import stft

            _f, _t, Zxx = stft(_segment.astype(np.float64), fs=sr, nperseg=_N_FFT, noverlap=_N_FFT - _HOP)
        except Exception as e:
            logger.debug("§V6 scipy.signal.stft fehlgeschlagen — Dummy-Bild zurückgegeben: %s", e)
            return _dummy_image()

        # Magnitude → dB → 0-255 Graustufen
        _mag = np.abs(Zxx)
        _mag_db = 20.0 * np.log10(_mag + 1e-10)
        _mag_db = np.clip(_mag_db, -80, 0)
        _mag_norm = ((_mag_db + 80) / 80 * 255).astype(np.uint8)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level

        # Resize auf 256×256
        _img = Image.fromarray(_mag_norm.T, mode="L")
        _img = _img.resize((_SNAPSHOT_SIZE, _SNAPSHOT_SIZE), Image.LANCZOS)  # type: ignore[attr-defined]
        return _img


def _dummy_image():
    """1×1 schwarzes Platzhalter-Bild."""
    try:
        from PIL import Image

        return Image.new("L", (1, 1), 0)
    except Exception as e:
        logger.debug("§V6 _dummy_image: PIL-Import fehlgeschlagen — None zurückgegeben: %s", e)
        return None
