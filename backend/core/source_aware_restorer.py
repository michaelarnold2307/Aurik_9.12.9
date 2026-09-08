"""§3.0b SourceAwareRestorer — Demucs → Per-Stem-UV3 → Remix.

Der Orchestrator:
  1. Trennt Audio via Demucs v5 in 4 Stems (vocals, drums, bass, other)
  2. Erstellt pro Stem einen reduzierten Phasenplan (SourceAwareFahrplan)
  3. Führt UV3.restore() für jeden Stem separat aus
  4. Remixt die Ergebnisse mit Stem-spezifischen Gains

Non-blocking: Fehler in einem Stem → dieser Stem bleibt unbearbeitet.
GPU-Fallback: Wenn Demucs nicht verfügbar → kein Source-Split → UV3 auf Vollmix.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

AVAILABLE_STEMS = ["vocals", "drums", "bass", "other"]

# §15.9/P1-1: ONNX-Sessions laufen über den zentralen InferenceSessionManager
# (Residency je Session, LRU-Eviction, Memory-Limit — kein Prozess-Eigen-Cache).
def _get_ort_session(model_path: str) -> Any:
    """ONNX-Session über den zentralen InferenceSessionManager (§15.9, P1-1).

    Residency: Ein zweiter Lauf in derselben Session bekommt die bereits
    geladene Session zurück (kein Modell-Nachladen); LRU-Eviction und
    Memory-Limit setzt der Manager zentral durch. Der Manager-Loader
    nutzt zusätzlich den MIGraphX-Adapter (§v10.40) und ORT_ENABLE_ALL.
    """
    from backend.core.ml.session_manager import get_session_manager

    return get_session_manager().acquire("demucs_htdemucs_6s", model_path)


def restore_per_source(
    audio: np.ndarray,
    sample_rate: int,
    restore_fn: Any,  # restorer.restore()
    restore_kwargs: dict[str, Any],
    *,
    material: str = "unknown",
    progress_callback: Any = None,
    skip_stems: list[str] | None = None,
) -> np.ndarray:
    """Führt Source-Separation + Per-Stem-Restoration aus.

    Args:
        audio: Eingabe-Audio (samples,) oder (channels, samples)
        sample_rate: Sample-Rate
        restore_fn: UV3.restore (oder RestaurierDenker.restauriere)
        restore_kwargs: kwargs für restore_fn
        material: Trägermedium (für Logging/Priorisierung)
        progress_callback: optionale Progress-Funktion(elapsed_s, msg)
        skip_stems: Stems die übersprungen werden (z.B. ["bass"] für Vintage ohne Bass)

    Returns:
        Restauriertes Audio, gleiche Shape wie Input
    """
    t0 = time.perf_counter()
    skip = set(skip_stems or [])

    # ── 1. Demucs Source-Separation ──────────────────────────────
    _emit_progress(progress_callback, 0.0, "§3.0: Demucs source separation...")

    try:
        stems = _separate_sources(audio, sample_rate)
    except Exception as exc:
        logger.warning("§3.0 Demucs fehlgeschlagen: %s — Ersatzpfad auf Vollmix-UV3", exc)
        _emit_progress(progress_callback, 10.0, "§3.0: Demucs failed, using full-mix")
        return _restore_fullmix(audio, sample_rate, restore_fn, restore_kwargs)

    _emit_progress(progress_callback, 8.0, f"§3.0: separated {len(stems)} stems")

    if not stems:
        return _restore_fullmix(audio, sample_rate, restore_fn, restore_kwargs)

    # ── 2. Per-Stem UV3-Restoration ──────────────────────────────
    restored_stems: dict[str, np.ndarray] = {}
    active_stems = [s for s in AVAILABLE_STEMS if s in stems and s not in skip]
    n_stems = max(len(active_stems), 1)

    from backend.core.source_aware_fahrplan import STEM_REMIX_GAINS, get_stem_config

    for i, stem_name in enumerate(active_stems):
        stem_audio = stems[stem_name]
        stem_pct = 8.0 + (i / n_stems) * 82.0  # 8–90% Progress

        _emit_progress(progress_callback, stem_pct, f"§3.0: restoring {stem_name}...")

        try:
            # Stem-spezifische Phase-Konfiguration
            stem_cfg = get_stem_config(stem_name)

            # Passe restore_kwargs an für diesen Stem
            stem_kwargs = dict(restore_kwargs)

            # Filtere Phasenplan: nur Phasen die für diesen Stem erlaubt sind
            _orig_phase_plan = restore_kwargs.get("precomputed_phase_plan")
            if _orig_phase_plan and isinstance(_orig_phase_plan, list) and stem_cfg.phase_strengths:
                _default_str = stem_cfg.phase_strengths.get("_default", 0.0 if stem_cfg.skip_all_default else 1.0)
                _filtered = [p for p in _orig_phase_plan if stem_cfg.phase_strengths.get(p, _default_str) > 0.0]
                stem_kwargs["precomputed_phase_plan"] = _filtered

            # Injiziere Stem-Info in denker_policy_input (für Logging/Fahrplan)
            denker_input = dict(stem_kwargs.get("denker_policy_input", {}) or {})
            phase_interaction = dict(denker_input.get("phase_interaction", {}) or {})
            phase_interaction["_source_stem"] = stem_name
            phase_interaction["_source_phase_strengths"] = stem_cfg.phase_strengths
            denker_input["phase_interaction"] = phase_interaction
            stem_kwargs["denker_policy_input"] = denker_input

            # Führe UV3 auf diesem Stem aus
            stem_result = restore_fn(stem_audio, **stem_kwargs)

            # Extrahiere Audio aus Result
            if hasattr(stem_result, "audio"):
                restored_audio = np.asarray(stem_result.audio)
            elif isinstance(stem_result, np.ndarray):
                restored_audio = stem_result
            else:
                restored_audio = stem_audio  # Fallback

            # Shape-Normalisierung
            if restored_audio.ndim != stem_audio.ndim:
                if stem_audio.ndim == 1 and restored_audio.ndim == 2:
                    restored_audio = restored_audio.mean(axis=0)
                elif stem_audio.ndim == 2 and restored_audio.ndim == 1:
                    restored_audio = np.stack([restored_audio, restored_audio])

            # Gain anwenden
            gain = STEM_REMIX_GAINS.get(stem_name, 1.0)
            restored_stems[stem_name] = restored_audio.astype(np.float32) * gain

            logger.debug("§3.0 %s wiederhergestellt in %.1fs", stem_name, time.perf_counter() - t0)

        except Exception as exc:
            logger.warning("§3.0 %s wiederherstellen fehlgeschlagen: %s — keeping Originalsignal", stem_name, exc)
            restored_stems[stem_name] = stem_audio.astype(np.float32)

    _emit_progress(progress_callback, 92.0, "§3.0: remixing stems...")

    # ── 3. Remix ─────────────────────────────────────────────────
    result = _remix_stems(restored_stems, stems, audio.shape)
    _emit_progress(progress_callback, 96.0, "§3.0: complete")

    elapsed = time.perf_counter() - t0
    logger.info("§3.0 Source-aware restoration vollstaendig in %.1fs (%d stems)", elapsed, len(restored_stems))
    return cast(np.ndarray, result.astype(np.float32))


def _separate_sources(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Trennt Audio in 4 Stems via Demucs ONNX (htdemucs_6s).

    Nutzt onnxruntime direkt — umgeht die kaputte demucs.pretrained-Kette.
    Modell: models/demucs/htdemucs_6s.onnx (6-Source: drums,bass,other,vocals,guitar,piano)
    """
    import os as _os

    _mono = audio.ndim == 1
    audio = np.atleast_2d(audio).astype(np.float32)

    # Resample auf 44100 Hz (Demucs ONNX native SR)
    if sr != 44100:
        import librosa

        audio_441 = np.stack(
            [
                librosa.resample(audio[ch].astype(np.float64), orig_sr=sr, target_sr=44100)
                for ch in range(audio.shape[0])
            ]
        ).astype(np.float32)
    else:
        audio_441 = audio.copy()

    # Modell-Pfad relativ zum Projekt-Root (3x dirname: core → backend → .)
    model_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "models",
        "demucs",
        "htdemucs_6s.onnx",
    )
    if not _os.path.exists(model_path):
        raise FileNotFoundError(f"Demucs ONNX nicht gefunden: {model_path}")

    sess = _get_ort_session(model_path)

    # Input: [1, 2, N] + encoder state x: [1, 4, 2048, 336]
    # Modell hat fixe Eingabelänge 343980 Samples (~7.8s @44100).
    # Padding/Trimming für beliebige Längen.
    MODEL_SAMPLES = 343980
    n_in = audio_441.shape[1]

    if n_in < MODEL_SAMPLES:
        padded = np.zeros((2, MODEL_SAMPLES), dtype=np.float32)
        padded[:, :n_in] = audio_441
        input_tensor = padded[np.newaxis, :, :]
    elif n_in > MODEL_SAMPLES:
        input_tensor = audio_441[np.newaxis, :, :MODEL_SAMPLES]
    else:
        input_tensor = audio_441[np.newaxis, :, :]

    x_dummy = np.zeros((1, 4, 2048, 336), dtype=np.float32)

    outputs = sess.run(None, {"input": input_tensor.astype(np.float32), "x": x_dummy})
    all_sources = outputs[1][0]  # [6, 2, MODEL_SAMPLES]

    # Trim zurück auf originale Länge
    if n_in < MODEL_SAMPLES:
        all_sources = all_sources[:, :, :n_in]

    # htdemucs_6s: drums, bass, other, vocals, guitar, piano
    source_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]
    stems: dict[str, np.ndarray] = {}
    for i, name in enumerate(source_names):
        src = all_sources[i]  # [2, N]
        if name in ("guitar", "piano"):
            stems["other"] = stems.get("other", 0.0) + src  # type: ignore[assignment]
        else:
            stems[name] = src

    # Resample zurück auf originale SR
    if sr != 44100:
        import librosa

        for k in list(stems):
            stems[k] = np.stack(
                [
                    librosa.resample(stems[k][ch].astype(np.float64), orig_sr=44100, target_sr=sr)
                    for ch in range(stems[k].shape[0])
                ]
            ).astype(np.float32)

    if _mono:
        for k in stems:
            stems[k] = stems[k].mean(axis=0)

    return stems


def _restore_fullmix(
    audio: np.ndarray,
    sample_rate: int,
    restore_fn: Any,
    restore_kwargs: dict[str, Any],
) -> np.ndarray:
    """Fallback: UV3 auf Vollmix ohne Source-Separation."""
    result = restore_fn(audio, **restore_kwargs)
    if hasattr(result, "audio"):
        return cast(np.ndarray, np.asarray(result.audio))
    if isinstance(result, np.ndarray):
        return result
    return audio


def _remix_stems(
    restored: dict[str, np.ndarray],
    original_stems: dict[str, np.ndarray],
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Kombiniert Stems zu finalem Mix."""
    if not restored:
        # Fallback: Original-Stems summieren
        result = np.zeros(target_shape, dtype=np.float32)
        for stem in original_stems.values():
            result += stem.astype(np.float32)
        return cast(np.ndarray, result)

    # Baue Mix aus verfügbaren Stems
    first_stem = next(iter(restored.values()))
    result = np.zeros_like(first_stem)

    for stem_name, stem_audio in restored.items():
        if stem_audio.shape == result.shape:
            result += stem_audio
        else:
            # Resize auf Target-Shape
            if result.ndim == 1 and stem_audio.ndim == 2:
                result += stem_audio.mean(axis=0)
            elif result.ndim == 2 and stem_audio.ndim == 1:
                result += np.stack([stem_audio, stem_audio])
            else:
                result += stem_audio[: result.shape[0]] if result.ndim == 1 else stem_audio[:, : result.shape[1]]

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0)))


def _emit_progress(cb: Any, pct: float, msg: str) -> None:
    """Non-blocking Progress-Emitter."""
    if cb is not None:
        try:
            cb(pct, msg, 0.0)
        except Exception as e:
            logger.warning("source_aware_restorer.py::_emit_progress Ersatzpfad: %s", e)
