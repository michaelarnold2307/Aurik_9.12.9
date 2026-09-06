from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_SEGMENTS = 200
DEFAULT_MIN_SEGMENT_SECONDS = 0.5
MIN_FILE_DURATION_S: float = 5.0  # Mindestlänge für Segment-Adaptive-Processing (§2.10)
MIN_SEGMENT_DURATION_S: float = 0.5  # Minimale Segmentlänge in Sekunden (§2.10)
CROSSFADE_MS: float = 20.0  # OLA-Crossfade in ms (§2.10 Hanning)

# §7.6 Adaptive Chunk-Verarbeitung (Dateien ≥ 5 Minuten)
_CHUNK_MIN_S: float = 2.0  # absolutes Minimum
_CHUNK_MAX_S: float = 120.0  # absolutes Maximum (Stille-Segmente)


def adaptive_chunk_size(defect_severity: float, segment_type: str) -> float:
    """§7.6 Liefert die optimale Chunk-Größe in Sekunden basierend auf Defektschwere.

    Spec-Tabelle (§7.6):
        segment_type=="silence"  → 120.0 s  (NR aggressiv, kein Transient-Risiko)
        defect_severity >= 0.6   →   5.0 s  (Feingranular - hohes Defektniveau)
        defect_severity >= 0.3   →  15.0 s
        sonst                    →  60.0 s  (sauberes Material, Kontext-Kohärenz)

    Grenzen: Minimum 2 s, Maximum 120 s.

    Args:
        defect_severity: Defektschwere [0.0, 1.0].
        segment_type: "silence" | "mixed" | sonstiger Typ.

    Returns:
        Chunk-Dauer in Sekunden, geclampt auf [2.0, 120.0].
    """
    if segment_type == "silence":
        size_s = 120.0
    elif defect_severity >= 0.6:
        size_s = 5.0
    elif defect_severity >= 0.3:
        size_s = 15.0
    else:
        size_s = 60.0
    return float(max(_CHUNK_MIN_S, min(_CHUNK_MAX_S, size_s)))


@dataclass
class AudioSegment:
    """Semantisches Audio-Segment mit lokalem Verarbeitungskontext (§2.10)."""

    start_sample: int
    end_sample: int
    segment_type: str = "mixed"
    noise_level_db: float = -60.0
    defect_severity: float = 0.0
    optimal_params: dict[str, float] = field(default_factory=dict)

    @property
    def duration_samples(self) -> int:
        return max(0, self.end_sample - self.start_sample)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "segment_type": self.segment_type,
            "noise_level_db": self.noise_level_db,
            "defect_severity": self.defect_severity,
            "duration_samples": self.duration_samples,
        }


@dataclass
class AdaptiveProcessingResult:
    """Ergebnis-Container für segment-adaptive Verarbeitung (§2.10)."""

    audio: np.ndarray
    segments: list[AudioSegment]
    enabled: bool
    used_fallback: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_segments(self) -> int:
        """Anzahl der verarbeiteten Segmente."""
        return len(self.segments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_segments": self.n_segments,
            "enabled": self.enabled,
            "used_fallback": self.used_fallback,
            **self.metadata,
        }


class SegmentAdaptiveProcessor:
    """Segment-adaptiver Prozessor (§2.10) — verarbeitet jedes Segment mit
    individuell optimierten Parametern und OLA-Crossfade an den Grenzen."""

    def __init__(self, min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS) -> None:
        self.min_segment_seconds = max(0.1, float(min_segment_seconds))

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------
    def _default_params(self) -> dict[str, float]:
        """Standard-Verarbeitungsparameter (konservativ, sichere Defaults)."""
        return {
            "noise_reduction_strength": 0.5,
            "harmonic_boost_db": 1.0,
            "ola_crossfade_ms": CROSSFADE_MS,
            "compression_ratio": 1.5,
            "eq_high_shelf_db": 0.0,
        }

    def _adaptive_params(self, segment_type: str, defect_severity: float) -> dict[str, float]:
        """Liefert segment-spezifische Parameter basierend auf Typ und Severity.

        §2.10 Aktivierungsregel: Stille <− 40 dBFS → NR max. 30 %.
        """
        params = self._default_params()
        if segment_type == "silence":
            params["noise_reduction_strength"] = 0.15  # ≤ 0.20 für Stille
            params["harmonic_boost_db"] = 0.0
        elif defect_severity >= 0.6:
            params["noise_reduction_strength"] = float(np.clip(0.5 + 0.4 * defect_severity, 0.5, 0.95))
        elif defect_severity >= 0.3:
            params["noise_reduction_strength"] = float(np.clip(0.3 + 0.4 * defect_severity, 0.3, 0.7))
        else:
            params["noise_reduction_strength"] = float(np.clip(0.1 + defect_severity, 0.1, 0.5))
        return params

    def _estimate_defect_severity(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt lokale Defektschwere im Bereich [0.0, 1.0]."""
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        if mono.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        rms_db = 20.0 * np.log10(rms + 1e-12)
        # Schätze Severity aus Rauschboden: leiser = weniger Defekte in diesem Segment
        severity = float(np.clip(max(0.0, (-rms_db - 20.0) / 60.0), 0.0, 1.0))
        return severity

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def segment_audio(self, audio: np.ndarray, sr: int) -> list[AudioSegment]:
        """Teilt Audio in semantische Segmente (§2.10).

        Args:
            audio: float32 mono oder stereo
            sr:    MUSS 48000 sein

        Returns:
            Liste von AudioSegment-Objekten
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        min_len = int(self.min_segment_seconds * sr)
        step = max(min_len, 1)
        segments: list[AudioSegment] = []
        for i, start in enumerate(range(0, int(mono.shape[0]), step)):
            if i >= MAX_SEGMENTS:
                break
            end = min(int(mono.shape[0]), start + step)
            if end <= start:
                continue
            chunk = mono[start:end]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12))
            noise_db = float(20.0 * np.log10(rms + 1e-12))
            seg_type = "silence" if noise_db < -40.0 else "mixed"
            severity = self._estimate_defect_severity(chunk, sr)
            segments.append(
                AudioSegment(
                    start_sample=int(start),
                    end_sample=int(end),
                    segment_type=seg_type,
                    noise_level_db=noise_db,
                    defect_severity=severity,
                    optimal_params=self._adaptive_params(seg_type, severity),
                )
            )
        return segments

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        process_fn: Callable[[np.ndarray, int, dict[str, float]], np.ndarray],
        material: str = "unknown",
        enabled: bool = True,
    ) -> AdaptiveProcessingResult:
        """Verarbeitet Audio segmentadaptiv.

        Fallback (used_fallback=True) bei:
          - enabled=False
          - Audio kürzer als MIN_FILE_DURATION_S
          - Exception in process_fn

        SR-Invariante: assert sr == 48000
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))

        # --- Fallback: disabled oder leer ---
        if not enabled or arr.size == 0:
            return AdaptiveProcessingResult(
                audio=np.clip(arr, -1.0, 1.0),
                segments=[],
                enabled=False,
                used_fallback=True,
                metadata={"reason": "disabled_or_empty", "material": material},
            )

        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        min_len = int(MIN_FILE_DURATION_S * sr)  # 5 s

        # --- Fallback: zu kurz für Segmentierung ---
        if mono.shape[0] < min_len:
            try:
                out = process_fn(arr, sr, self._default_params())
                out = np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
            except Exception:
                out = np.clip(arr, -1.0, 1.0)
            return AdaptiveProcessingResult(
                audio=out,
                segments=[
                    AudioSegment(
                        start_sample=0,
                        end_sample=int(mono.shape[0]),
                    )
                ],
                enabled=True,
                used_fallback=True,
                metadata={"material": material, "reason": "too_short"},
            )

        # --- Segmentierung ---
        segments = self.segment_audio(audio, sr)

        # --- Verarbeitung je Segment ---
        processed = np.zeros_like(arr)
        fallback_used = False
        for seg in segments:
            if arr.ndim == 2:
                seg_audio = arr[:, seg.start_sample : seg.end_sample]
            else:
                seg_audio = arr[seg.start_sample : seg.end_sample]

            try:
                out_seg = process_fn(seg_audio, sr, seg.optimal_params)
                out_seg = np.nan_to_num(np.asarray(out_seg, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
            except Exception:
                # Exception-Fallback: Passthrough für dieses Segment
                out_seg = np.clip(seg_audio, -1.0, 1.0)
                fallback_used = True

            # ── §v10 OLA-Crossfade: Hanning-Fenster an Segmentgrenzen ──
            cf_samples = int(CROSSFADE_MS * sr / 1000.0)  # 20 ms → 960 samples @ 48 kHz
            cf_samples = max(1, min(cf_samples, seg.duration_samples // 3))
            fade_in = np.hanning(2 * cf_samples)[:cf_samples].astype(np.float32)
            fade_out = np.hanning(2 * cf_samples)[cf_samples:].astype(np.float32)

            n = min(out_seg.shape[-1] if out_seg.ndim == 2 else len(out_seg), seg.duration_samples)
            seg_start = seg.start_sample

            if cf_samples >= 2 and n > 2 * cf_samples:
                if arr.ndim == 2:
                    if out_seg.ndim == 2:
                        seg_processed = out_seg[:, :n].copy()
                    else:
                        seg_processed = np.tile(out_seg[:n], (arr.shape[0], 1))
                    for ch in range(arr.shape[0]):
                        seg_processed[ch, :cf_samples] *= fade_in
                        seg_processed[ch, -cf_samples:] *= fade_out
                    # OLA: add to existing, normalize overlap
                    processed[:, seg_start : seg_start + n] += seg_processed
                    if seg_start > 0:
                        overlap = min(cf_samples, n)
                        processed[:, seg_start : seg_start + overlap] *= 0.5
                else:
                    seg_processed = out_seg[:n].copy()
                    seg_processed[:cf_samples] *= fade_in
                    seg_processed[-cf_samples:] *= fade_out
                    processed[seg_start : seg_start + n] += seg_processed
                    if seg_start > 0:
                        overlap = min(cf_samples, n)
                        processed[seg_start : seg_start + overlap] *= 0.5
            else:
                if arr.ndim == 2:
                    if out_seg.ndim == 2:
                        processed[:, seg_start : seg_start + n] = out_seg[:, :n]
                    else:
                        processed[:, seg_start : seg_start + n] = out_seg[:n]
                else:
                    processed[seg_start : seg_start + n] = out_seg[:n]

        processed = np.clip(processed, -1.0, 1.0)
        return AdaptiveProcessingResult(
            audio=processed,
            segments=segments,
            enabled=True,
            used_fallback=fallback_used,
            metadata={"material": material, "n_segments": len(segments)},
        )


_instance: SegmentAdaptiveProcessor | None = None
_lock = threading.Lock()


def get_segment_adaptive_processor() -> SegmentAdaptiveProcessor:
    """Thread-sicherer Singleton-Accessor (§3.2 Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SegmentAdaptiveProcessor()
    return _instance


# Alias für ältere Importe
get_segment_processor = get_segment_adaptive_processor


def process_segment_adaptive(
    audio: np.ndarray,
    sr: int,
    process_fn: Callable[[np.ndarray, int, dict[str, float]], np.ndarray],
    material: str = "unknown",
    enabled: bool = True,
) -> AdaptiveProcessingResult:
    """Convenience-Wrapper für segment-adaptive Verarbeitung."""
    return get_segment_adaptive_processor().process(audio, sr, process_fn, material, enabled)


# Alias: process_adaptive (von Tests verwendet)
process_adaptive = process_segment_adaptive


__all__ = [
    "DEFAULT_MIN_SEGMENT_SECONDS",
    "MAX_SEGMENTS",
    "MIN_FILE_DURATION_S",
    "AdaptiveProcessingResult",
    "AudioSegment",
    "SegmentAdaptiveProcessor",
    "adaptive_chunk_size",
    "get_segment_adaptive_processor",
    "get_segment_processor",
    "process_adaptive",
    "process_segment_adaptive",
]

# Aliase für Rückwärtskompatibilität (Spec §3.2)
get_segment_processor = get_segment_adaptive_processor
process_adaptive = process_segment_adaptive

__all__ += ["_CHUNK_MAX_S", "_CHUNK_MIN_S", "adaptive_chunk_size"]
logger = logging.getLogger(__name__)
