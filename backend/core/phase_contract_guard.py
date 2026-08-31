"""
§v10.15 Phase Contract Guards
=============================
Centralized type and shape validation for all phase entry/exit points.
Imported by UnifiedRestorerV3 to validate every phase call.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

from .phases.phase_interface import PhaseResult

logger = logging.getLogger(__name__)


def guard_phase_input(audio: np.ndarray, sample_rate: int, phase_id: str) -> np.ndarray:
    """Validate phase input and normalize audio shape.

    Returns normalized audio (N, 2) for stereo, (N,) for mono.
    Raises TypeError on invalid input types.
    """
    if not isinstance(audio, np.ndarray):
        if isinstance(audio, (tuple, list)):
            logger.error(
                "PhaseContract [%s]: received %s instead of ndarray — extrahiere",
                phase_id,
                type(audio).__name__,
            )
            audio = audio[0] if len(audio) > 0 else np.zeros(1, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32)

    if not isinstance(sample_rate, (int, float)):
        raise TypeError(f"PhaseContract [{phase_id}]: sample_rate must be int, got {type(sample_rate)}")

    if audio.ndim not in (1, 2):
        raise ValueError(f"PhaseContract [{phase_id}]: audio must be 1D or 2D, got {audio.ndim}D shape={audio.shape}")

    if audio.ndim == 2 and audio.shape[0] > 2 and audio.shape[1] > 2:
        raise ValueError(f"PhaseContract [{phase_id}]: ambiguous shape {audio.shape}")

    return cast(np.ndarray, (np.ascontiguousarray(audio, dtype=np.float32)))


def guard_phase_output(result, audio_in: np.ndarray, phase_id: str) -> PhaseResult:
    """Validate phase output is a proper PhaseResult with valid audio.

    Returns the PhaseResult (passes through if valid).
    §v10.304: Normalisiert Audio-Shape auf Input-Shape (eliminiert P5-Exceptions).
    """
    from .phases.phase_interface import PhaseResult

    if not isinstance(result, PhaseResult):
        raise TypeError(
            f"PhaseContract [{phase_id}]: expected PhaseResult, got {type(result).__name__}. "
            f"Phase must return create_phase_result(audio=..., modifications=..., warnings=...)"
        )

    if result.audio is not None:
        if not isinstance(result.audio, np.ndarray):
            raise TypeError(f"PhaseContract [{phase_id}]: PhaseResult.audio must be ndarray, got {type(result.audio)}")
        if result.audio.ndim not in (1, 2):
            raise ValueError(f"PhaseContract [{phase_id}]: result audio must be 1D or 2D, got {result.audio.ndim}D")
        # §v10.304: Shape-Normalisierung verhindert P5-Exceptions
        # (STFT/ISTFT kann Frame-Grenzen verschieben → Kanäle unterschiedlich lang)
        if audio_in is not None and result.audio.ndim == audio_in.ndim:
            _in_len = audio_in.shape[-1] if audio_in.ndim == 2 else len(audio_in)
            _out_len = result.audio.shape[-1] if result.audio.ndim == 2 else len(result.audio)
            if _out_len != _in_len:
                _min_len = min(_out_len, _in_len)
                if result.audio.ndim == 2:
                    result = PhaseResult(
                        audio=result.audio[..., :_min_len].copy()
                        if _out_len > _in_len
                        else np.pad(result.audio, ((0, 0), (0, _in_len - _out_len))),
                        modifications=result.modifications,
                        warnings=result.warnings + [f"Shape normalized: {_out_len}→{_in_len}"],
                    )
                else:
                    result = PhaseResult(
                        audio=result.audio[:_min_len].copy()
                        if _out_len > _in_len
                        else np.pad(result.audio, (0, _in_len - _out_len)),
                        modifications=result.modifications,
                        warnings=result.warnings + [f"Shape normalized: {_out_len}→{_in_len}"],
                    )

    return result  # type: ignore[no-any-return]


def guard_phase_shape_consistency(audio_out: np.ndarray, audio_in: np.ndarray, phase_id: str) -> None:
    """Warn if phase output length differs from input length by > 0.1%."""
    if audio_out is None or audio_in is None:
        return
    len_in = audio_in.shape[-1] if audio_in.ndim == 2 else len(audio_in)
    len_out = audio_out.shape[-1] if audio_out.ndim == 2 else len(audio_out)
    delta_pct = abs(len_out - len_in) / max(len_in, 1) * 100
    if delta_pct > 0.1:
        logger.warning(
            "PhaseContract [%s]: Ausgabe length %d differs from Eingabe %d by %.2f%%",
            phase_id,
            len_out,
            len_in,
            delta_pct,
        )
