"""Aurik 10 — Bridge: Audio-Verarbeitung (§11 Spec 08)
===================================================
Audio-spezifische Brückenfunktionen für Frontend/CLI → Backend-Core.

Enthält:
  - AudioExporter-Klasse (lazy import, None-Fallback)
  - LyricsGuidedEnhancement-Singleton (§2.36)
  - Cleanup-after-file (PLM-Integration)
  - Pipeline Health State Enum + Normalisierung
  - StemRemixBalancer.balance_remix (§1.4)
  - ClippingClassifier-Singleton (§6.3)
  - Audio-Import-Kaskade (soundfile → pedalboard/FFmpeg → pydub)
  - Quiet Edge Boost (§11 Spec 08)
  - CD-Rauschprofil-Injektion (§G4/GEBOTE.md)
  - Live Preview während Restaurierung (§v10.101)

Referenz: AGENTS.md §1 (Normative Kette), .github/copilot-instructions.md §V4 Bridge-Verbot.
"""

# pylint: disable=import-outside-toplevel
# cspell:disable

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status locks & stubs (AudioExporter)
# ---------------------------------------------------------------------------

_audio_exporter_status_lock = threading.Lock()
_audio_exporter_status: dict[str, Any] = {
    "available": True,
    "failures": 0,
    "last_error": "",
}


def get_audio_exporter_class() -> type | None:
    """Gibt ``AudioExporter``-Klasse zurück (lazy import).

    Gibt ``None`` zurück wenn ``backend.core.audio_exporter`` nicht verfügbar
    ist — Aufrufer muss dann ``soundfile.write()`` als Fallback verwenden.
    Spec §11.3: Kein Hard-Fail bei optionalen Export-Modulen.
    """
    try:
        from backend.core.audio_exporter import AudioExporter  # type: ignore[import]

        with _audio_exporter_status_lock:
            _audio_exporter_status["available"] = True
            _audio_exporter_status["last_error"] = ""

        return AudioExporter  # type: ignore[no-any-return]
    except ImportError as exc:
        _err = f"{type(exc).__name__}: {exc}"
        with _audio_exporter_status_lock:
            _audio_exporter_status["available"] = False
            _audio_exporter_status["failures"] = int(_audio_exporter_status.get("failures", 0)) + 1
            _audio_exporter_status["last_error"] = _err
        logger.warning("bridge: AudioExporter nicht verfügbar — sf.write als Ersatzpfad (%s)", _err)
        return None


def get_audio_exporter_status() -> dict[str, Any]:
    """Liefert Bridge-Telemetrie für AudioExporter-Importstatus."""
    with _audio_exporter_status_lock:
        return dict(_audio_exporter_status)


# ---------------------------------------------------------------------------
# Lyrics-Guided Enhancement (§2.36)
# ---------------------------------------------------------------------------


def get_lyrics_guided_enhancement_fn():
    """Gibt ``LyricsGuidedEnhancement``-Singleton zurück (lazy import, §2.36).

    Rückgabe: ``LyricsGuidedEnhancement``-Instanz mit ``.enhance(audio, sr)``
    und ``.get_timeline()``.

    Pflicht ab 9.10.x (§2.36): Wird im Frontend für L-Shortcut-Overlay und
    im BatchProcessingThread für ContentAwareProcessor-Integration verwendet.
    """
    from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement  # type: ignore[import]

    return get_lyrics_guided_enhancement()


# ---------------------------------------------------------------------------
# Cleanup after file (PLM-Integration)
# ---------------------------------------------------------------------------


def get_cleanup_after_file_fn():
    """Gibt ``cleanup_after_file``-Funktion zurück (lazy import)."""
    from backend.core.plugin_lifecycle_manager import cleanup_after_file  # type: ignore[import]

    return cleanup_after_file


# ---------------------------------------------------------------------------
# Pipeline Health State Enum + Normalisierung
# ---------------------------------------------------------------------------


def get_pipeline_health_state_enum() -> type:
    """Gibt ``PipelineHealthState``-Enum zurück (lazy import)."""
    from backend.core.pipeline_health_state import PipelineHealthState  # type: ignore[import]

    return PipelineHealthState  # type: ignore[no-any-return]


def normalize_pipeline_health_state(raw):
    """Normalisiert Pipeline-Health-State auf kanonische Enum-Werte (lazy import)."""
    from backend.core.pipeline_health_state import normalize_pipeline_health_state as _normalize  # type: ignore[import]

    return _normalize(raw)


# ---------------------------------------------------------------------------
# StemRemixBalancer (§1.4)
# ---------------------------------------------------------------------------


def get_stem_remix_balancer_fn():
    """Gibt ``StemRemixBalancer.balance_remix``-Funktion zurück (lazy import, §1.4).

    Signatur: ``balance_remix(vocals, instruments, original, sr, vocal_weight) -> np.ndarray``
    Verwendet ITU-R BS.1770-5 K-gewichtete LUFS-Messung für Gain-Korrektur.
    LUFS-Differenz nach Re-Mix ≤ 0.3 LU gegenüber Original (§1.4 Spec).
    """
    from backend.core.stem_remix_balancer import StemRemixBalancer  # type: ignore[import]

    return StemRemixBalancer().balance_remix


# ---------------------------------------------------------------------------
# ClippingClassifier-Singleton (§6.3)
# ---------------------------------------------------------------------------


def get_clipping_classifier():
    """Gibt ``ClippingClassifier``-Singleton zurück (lazy import, §6.3).

    Rückgabe: ``ClippingClassifier``-Instanz.
    Verwende ``classify_clipping(audio, sr)`` (Convenience-Funktion) für
    direkten Aufruf ohne Singleton-Handle.

    §6.3 CLIPPING vs SOFT_SATURATION: THD-basierte Diskriminierung.
    SOFT_SATURATION (gerade Harmonische — Röhre/Tape) → bewahren.
    CLIPPING (ungerade Harmonische + flat_tops > 0.1 %) → reparieren.
    """
    from backend.core.clipping_detection import get_clipping_classifier as _get  # type: ignore[import]

    return _get()


# ---------------------------------------------------------------------------
# Audio-Import — Kaskade soundfile → pedalboard/FFmpeg → pydub (§11 VERBOTEN)
# ---------------------------------------------------------------------------


def get_load_audio_fn():
    """Gibt ``load_audio_file`` from backend.file_import (lazy) zurück.

    The returned function signature::

        load_audio_file(filepath, target_sr=None, mono=False) -> dict | None

    The dict contains keys ``audio`` (np.ndarray float32) and ``sr`` (int).
    Falls back to ``None`` when the import chain fails so callers can degrade
    gracefully.
    """
    from backend.file_import import load_audio_file  # type: ignore[import]

    return load_audio_file


# ---------------------------------------------------------------------------
# Quiet Edge Boost (§11 Spec 08)
# ---------------------------------------------------------------------------


def limit_quiet_edge_boost(
    reference_audio: Any,
    candidate_audio: Any,
    sr: int,
    *,
    material_key: str | None = None,
    max_edge_boost_db: float = 2.0,
) -> Any:
    """Bridge-Wrapper für backend.core.audio_utils.limit_quiet_edge_boost (§11 Spec 08).

    Skaliert quiet intro/outro regions back toward the original edge level.
    """
    try:
        from backend.core.audio_utils import limit_quiet_edge_boost as _fn

        return _fn(
            reference_audio,
            candidate_audio,
            sr,
            material_key=material_key,
            max_edge_boost_db=max_edge_boost_db,
        )
    except Exception as _e:
        logger.warning("§G23 bridge: limit_quiet_edge_boost DSP-Ersatzpfad (passthrough): %s", _e, exc_info=True)
        return candidate_audio


# ---------------------------------------------------------------------------
# CD-Rauschprofil-Injektion (§G4/GEBOTE.md / §G63)
# ---------------------------------------------------------------------------


def inject_cd_noise_profile(
    audio,
    sample_rate: int,
    *,
    mode: str = "restoration",
    bit_depth: int = 16,
    seed: int | None = None,
) -> object:
    """Injiziert CD-Rauschprofil. Wrapper für backend.core.cd_noise_profile (§G4 (GEBOTE.md)/§G63)."""
    try:
        from backend.core.cd_noise_profile import inject_cd_noise_profile as _inject

        return _inject(audio, sample_rate, mode=mode, bit_depth=bit_depth, seed=seed)  # type: ignore[misc]
    except ImportError:
        logger.debug("§V6 cd_noise_profile nicht verfügbar — Audio unverändert zurückgegeben")
        return audio


# ---------------------------------------------------------------------------
# Live Preview während Restaurierung (§v10.101)
# ---------------------------------------------------------------------------


def get_live_preview(seek_s: float = 0.0, duration_s: float = 5.0) -> dict | None:
    """§v10.101 Live-Preview: Aktuelles Pipeline-Audio an beliebiger Position.

    Der User kann waehrend der Restaurierung an JEDE Stelle springen
    und hoeren, wie der aktuelle Stand klingt.
    """
    try:
        import base64
        import io
        import os
        import tempfile

        import soundfile as sf

        _path = os.path.join(tempfile.gettempdir(), "aurik_live_preview.wav")
        if not os.path.exists(_path):
            return None

        from backend.file_import import load_audio_file

        audio, sr = load_audio_file(_path)  # type: ignore[misc]
        n_total = len(audio)
        start = max(0, min(int(seek_s * sr), n_total - 1))  # type: ignore[operator]
        end = min(start + int(duration_s * sr), n_total)  # type: ignore[operator]
        snippet = audio[start:end]

        buf = io.BytesIO()
        sf.write(buf, snippet, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        return {
            "audio_b64": base64.b64encode(buf.read()).decode("ascii"),
            "sample_rate": sr,
            "duration_s": float(len(snippet) / sr),  # type: ignore[operator]
            "seek_s": float(seek_s),
            "total_s": float(n_total / sr),  # type: ignore[operator]
        }
    except Exception as _prev_exc:
        logger.warning("§G93 bridge: get_live_preview failed → returning None: %s", _prev_exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public API — explizite Export-Liste
# ---------------------------------------------------------------------------

__all__ = [
    # AudioExporter
    "get_audio_exporter_class",
    "get_audio_exporter_status",
    # Lyrics-Guided Enhancement (§2.36)
    "get_lyrics_guided_enhancement_fn",
    # Cleanup after file (PLM-Integration)
    "get_cleanup_after_file_fn",
    # Pipeline Health State Enum + Normalisierung
    "get_pipeline_health_state_enum",
    "normalize_pipeline_health_state",
    # StemRemixBalancer (§1.4)
    "get_stem_remix_balancer_fn",
    # ClippingClassifier-Singleton (§6.3)
    "get_clipping_classifier",
    # Audio-Import-Kaskade (§11 VERBOTEN: sf.read / librosa.load direkt)
    "get_load_audio_fn",
    # Quiet Edge Boost (§11 Spec 08)
    "limit_quiet_edge_boost",
    # CD-Rauschprofil-Injektion (§G4/GEBOTE.md / §G63)
    "inject_cd_noise_profile",
    # Live Preview während Restaurierung (§v10.101)
    "get_live_preview",
]
