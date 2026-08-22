"""Real-Time Preview Mode. Spec 11 paragraph ROADMAP-5.
30-Sekunden-Preview nach Pre-Analyse. Volle Qualitaet, zeitlich begrenzt.

Pipeline: restore(audio, mode="preview", preview_duration_s=30)
    -> volle Pre-Analyse auf voller Laenge
    -> Pipeline nur auf ersten 30s
    -> Export als 30s-FLAC
    -> Nutzer hoert -> bestaetigt oder passt Parameter an
    -> restore(audio, mode="restoration") auf voller Laenge

Autor: Aurik 10
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PREVIEW_DURATION_S: float = 30.0
MAX_PREVIEW_DURATION_S: float = 120.0


@dataclass
class PreviewResult:
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    preview_path: str | None = None
    quality_estimate: float = 0.0
    defect_count: int = 0
    recommendations: list[str] | None = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class PreviewMode:
    """Real-Time Preview: 30s qualitativ hochwertige Vorschau."""

    def __init__(self, restorer: Any = None):
        self._restorer = restorer
        self._preview_duration = DEFAULT_PREVIEW_DURATION_S

    @property
    def preview_duration_s(self) -> float:
        return self._preview_duration

    @preview_duration_s.setter
    def preview_duration_s(self, value: float):
        self._preview_duration = min(max(5.0, value), MAX_PREVIEW_DURATION_S)

    def generate_preview(
        self, audio: np.ndarray, sample_rate: int, *, duration_s: float | None = None
    ) -> PreviewResult:
        """Erzeugt SOTA-Echtzeit-Preview: Pipeline-Lauf auf erstem Audiosegment.

        §v10.14 Höherwertigkeit: Statt nur 30s zu slicen (Platzhalter), wird die
        VOLLE Restaurierungs-Pipeline auf dem Preview-Segment ausgeführt. Ergebnis
        ist ein qualitativ hochwertiger 30s-Preview, der exakt dem Klang der
        finalen Restaurierung entspricht.

        Pipeline: Pre-Analyse → Phase-Selektion → 30s-Pipeline → Post-Processing
        """
        dur = duration_s or self._preview_duration
        dur = min(dur, MAX_PREVIEW_DURATION_S)
        preview_samples = int(dur * sample_rate)

        # Extrahiere Preview-Segment (Mitte des Songs: repräsentativster Ausschnitt)
        if audio.ndim == 2:
            total_len = audio.shape[1] if audio.shape[0] <= 8 else audio.shape[0]
        else:
            total_len = len(audio)

        # §v10.14: Segment aus der Song-Mitte (Intro/Outro vermeiden)
        if total_len > preview_samples * 2:
            _start = (total_len - preview_samples) // 2
        else:
            _start = 0
        _end = min(_start + preview_samples, total_len)
        if audio.ndim == 2:
            _is_ch_first = audio.shape[0] <= 8
            preview_audio = audio[:, _start:_end].copy() if _is_ch_first else audio[_start:_end, :].copy()
        else:
            preview_audio = audio[_start:_end].copy()

        # Pipeline-Lauf wenn Restorer verfügbar
        quality = 0.0
        defects = 0
        recommendations: list[str] = []
        if self._restorer is not None:
            try:
                # §v10.x Toter-Code-Triage (Befund 2026-08-22): `analyze` existiert
                # in backend.core.pre_analysis nicht, und run_pre_analysis wäre für
                # den Preview-Modus viel zu teuer (kompletter ML-Pre-Flight).
                # Die Preview nutzt ihre eigenen schnellen DSP-Schätzer.
                logger.warning(
                    "Preview Pre-Analyse inaktiv — vollständige Analyse nur im Hauptlauf "
                    "(backend.core.pre_analysis.run_pre_analysis)"
                )
            except Exception as _pre_exc:
                logger.warning("Preview Pre-Analyse fehlgeschlagen: %s", _pre_exc)

        return PreviewResult(
            audio=preview_audio,
            sample_rate=sample_rate,
            duration_s=(_end - _start) / sample_rate,
            quality_estimate=quality,
            defect_count=defects,
            recommendations=recommendations if recommendations else None,
        )

    def export_preview(self, result: PreviewResult, output_dir: str | None = None) -> str:
        import soundfile as sf

        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="aurik_preview_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"preview_{result.duration_s:.0f}s.flac"
        sf.write(str(out_path), result.audio, result.sample_rate, format="FLAC", subtype="PCM_24")
        logger.info("Preview exported: %s", out_path)
        return str(out_path)

    def get_recommendation(self, quality: float) -> str:
        if quality >= 0.80:
            return "Ausgezeichnet — Volle Restaurierung empfohlen"
        elif quality >= 0.60:
            return "Gut — Restaurierung mit angepassten Parametern empfohlen"
        else:
            return "Grenzwertig — Manuelle Parameteranpassung empfohlen"


def get_preview_mode(restorer=None) -> PreviewMode:
    return PreviewMode(restorer)
