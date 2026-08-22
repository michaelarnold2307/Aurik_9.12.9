"""
Phase 53: Semantic Audio Analysis v3.0 — ML Tier-1 + DSP Baseline
==================================================================

Drei-stufige Semantik-Kaskade (beste Qualität → robuster Fallback):
  Tier-1: LAION-CLAP  (text-audio aligned, 512-dim, genre/instrument)
  Tier-0: BEATs iter3  (AudioSet-527, sound-event tagging)
  DSP:    Chromagramm + Spektralzentroid (kein Modell erforderlich)

KATEGORIE: METADATA — Audio wird NICHT verändert, nur analysiert.

EXTRAHIERTE FEATURES:
  - BPM:            Auto-Korrelation der Onset-Stärke (Bereich 60–180 BPM)
  - Tonart (Key):   Chromagramm-Projektion auf Dur/Moll-Profil (Krumhansl 1990)
  - Genre-Hint:     CLAP > BEATs > DSP-Heuristik (Prioritätskaskade)
  - Loudness-Klasse: LUFS-Näherung + Crest Factor
  - CLAP:           top_genres, top_instruments, 32-dim embedding
  - BEATs:          top_tags (AudioSet-527), 32-dim embedding

WICHTIG:
  - process() gibt audio UNVERÄNDERT zurück
  - Alle Analysen landen in PhaseResult.metadata + PhaseResult.metrics

Author: Aurik Development Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import safe_to_mono
from backend.core.ml_memory_budget import release as _release_ml_budget
from backend.core.ml_memory_budget import try_allocate as _try_allocate_ml_budget
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

_clap_factory_impl: Any = None
try:
    from plugins.laion_clap_plugin import get_laion_clap as _clap_factory_loaded

    _clap_factory_impl = _clap_factory_loaded
except Exception as e:
    logger.warning("Verarbeitungsschritt_53_semantic_audio.py::unbekannter Ersatzpfad: %s", e)
_clap_factory: Any = _clap_factory_impl

_beats_factory_impl: Any = None
try:
    from plugins.beats_plugin import get_beats_plugin as _beats_factory_loaded

    _beats_factory_impl = _beats_factory_loaded
except Exception as e:
    logger.warning("Verarbeitungsschritt_53_semantic_audio.py::unbekannter Ersatzpfad: %s", e)
_beats_factory: Any = _beats_factory_impl


_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_CANONICAL_GENRE_FALLBACK = "Unbekannt"
_GENRE_ALIAS_MAP = {
    "classical": "Klassik",
    "classical_orchestral": "Klassik",
    "orchestra": "Klassik",
    "orchestral": "Klassik",
    "opera": "Oper",
    "jazz": "Jazz",
    "jazz_acoustic": "Jazz",
    "rock": "Rock",
    "rock_metal": "Rock",
    "metal": "Rock",
    "pop": "Pop",
    "pop_ballad": "Pop",
    "blues": "Blues",
    "folk": "Folk",
    "country": "Folk",
    "electronic": "Electronic",
    "electronic_edm": "Electronic",
    "ambient": "Electronic",
    "hip_hop": "Hip-Hop",
    "hip-hop": "Hip-Hop",
    "rap": "Hip-Hop",
    "reggae": "Reggae",
    "gospel": "Gospel",
    "rnb": "Soul/R&B",
    "soul": "Soul/R&B",
    "soul/r&b": "Soul/R&B",
    "r&b": "Soul/R&B",
    "rhythm_and_blues": "Soul/R&B",
    "schlager": "Schlager",
    "general": _CANONICAL_GENRE_FALLBACK,
    "unknown": _CANONICAL_GENRE_FALLBACK,
    "unbekannt": _CANONICAL_GENRE_FALLBACK,
}


def _clap_allowed_in_current_context() -> bool:
    """Gibt an, ob der schwere CLAP-Pfad in diesem Kontext ausgeführt werden darf."""
    if os.getenv("AURIK_FORCE_CLAP_PHASE53", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if os.getenv("AURIK_SAFE_VALIDATION_PROFILE", "0") == "1":
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return True


def _mono(audio: np.ndarray) -> np.ndarray:
    return safe_to_mono(audio) if audio.ndim == 2 else audio


def _estimate_bpm(mono: np.ndarray, sr: int) -> float:
    """Auto-Korrelation der Onset-Hüllkurve für BPM-Schätzung."""
    # **GUARD: Short-Audio-Buffer (§2.47, §0 Primum non nocere)**
    MIN_AUDIO_SAMPLES = 512  # 10 ms @ 48 kHz
    if len(mono) < MIN_AUDIO_SAMPLES:
        return 120.0  # Default fallback for ultra-short audio

    # Hüllkurve über RMS-Fenster (23 ms)
    frame = max(1, int(0.023 * sr))
    hop = max(1, frame // 2)
    n_frames = (len(mono) - frame) // hop
    if n_frames < 4:
        return 120.0
    onset = np.array([float(np.sqrt(np.mean(mono[i * hop : i * hop + frame] ** 2))) for i in range(n_frames)])
    # Auto-Korrelation — FFT-based O(N log N)
    onset -= onset.mean()
    from backend.core.core_utils import fft_autocorr  # pylint: disable=import-outside-toplevel

    ac = fft_autocorr(onset)
    # Suchbereich: 60–180 BPM → Periode in Frames
    min_period = int(60.0 / 180.0 * sr / hop)
    max_period = int(60.0 / 60.0 * sr / hop)
    min_period = max(1, min(min_period, len(ac) - 2))
    max_period = max(min_period + 1, min(max_period, len(ac) - 1))
    best_period = min_period + int(np.argmax(ac[min_period:max_period]))
    bpm = 60.0 * sr / (best_period * hop)
    return float(np.clip(bpm, 60.0, 180.0))


def _estimate_key(mono: np.ndarray, sr: int) -> str:
    """Chromagramm + Krumhansl-Profile → Tonart-Schätzung."""
    # **GUARD: Short-Audio-Buffer (§2.47, §0 Primum non nocere)**
    MIN_AUDIO_SAMPLES = 512  # 10 ms @ 48 kHz
    if len(mono) < MIN_AUDIO_SAMPLES:
        return "C major"  # Default fallback for ultra-short audio

    n_fft = 4096
    hop = 1024
    # §v10.103 noverlap-Guard: clamp noverlap < nperseg für kurzes Audio
    _noverlap = min(n_fft - hop, max(0, n_fft - 1))
    f, _t, Zxx = sig.stft(mono, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann")
    mag = np.abs(Zxx)

    # Frequenz → Chroma-Bin (12-stufige gleichmäßige Stimmung, A4=440 Hz)
    eps = 1e-8
    freqs = f[1:]  # Gleichstromanteil überspringen
    mag = mag[1:, :]  # entsprechend kürzen
    chroma = np.zeros(12)
    for i, freq in enumerate(freqs):
        if freq < 27.5:
            continue
        midi = 69 + 12 * np.log2(freq / 440.0 + eps)
        chroma_bin = round(midi) % 12
        chroma[chroma_bin] += float(np.mean(mag[i]))

    if chroma.sum() < eps:
        return "C major"

    chroma = chroma / (chroma.sum() + eps)

    # Verschiebe das Profil für alle 12 Tonarten, wähle besten Pearson-r
    best_r = -2.0
    best_key = "C"
    best_mode = "major"
    # Pre-compute profile vectors for guarded Pearson correlation (§VERBOTEN: np.corrcoef)
    _maj_g = np.asarray(_MAJOR_PROFILE, dtype=np.float64)
    _min_g = np.asarray(_MINOR_PROFILE, dtype=np.float64)
    _maj_g = _maj_g - _maj_g.mean()
    _min_g = _min_g - _min_g.mean()
    _maj_norm_g = np.linalg.norm(_maj_g)
    _min_norm_g = np.linalg.norm(_min_g)
    for root in range(12):
        shifted = np.roll(chroma, -root)
        _shf_g = np.asarray(shifted, dtype=np.float64)
        _shf_g = _shf_g - _shf_g.mean()
        _shf_norm_g = np.linalg.norm(_shf_g)
        r_maj = float(np.dot(_shf_g, _maj_g) / (_shf_norm_g * _maj_norm_g + 1e-12))
        r_min = float(np.dot(_shf_g, _min_g) / (_shf_norm_g * _min_norm_g + 1e-12))
        if r_maj > best_r:
            best_r = r_maj
            best_key = _NOTE_NAMES[root]
            best_mode = "major"
        if r_min > best_r:
            best_r = r_min
            best_key = _NOTE_NAMES[root]
            best_mode = "minor"

    return f"{best_key} {best_mode}"


def _estimate_genre_hint(mono: np.ndarray, sr: int) -> str:
    """Grober Genre-Hint via Spektralzentroid + Crest Factor."""
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    peak = float(np.max(np.abs(mono))) + 1e-12
    crest = peak / rms

    # Spektralzentroid
    f, psd = sig.periodogram(mono, fs=sr, window="hann")
    total_psd = np.sum(psd) + 1e-12
    centroid = float(np.sum(f * psd) / total_psd)

    if centroid < 1200.0 and crest > 8.0:
        return "classical_orchestral"
    if centroid < 1500.0 and crest > 5.0:
        return "jazz_acoustic"
    if centroid > 3000.0 and crest < 5.0:
        return "electronic_edm"
    if centroid > 2500.0:
        return "rock_metal"
    if centroid < 2000.0:
        return "pop_ballad"
    return "general"


def _canonicalize_genre_hint(label: str | None) -> str:
    """Map raw DSP/CLAP/BEATs genre tags to Aurik canonical labels."""
    if not label:
        return _CANONICAL_GENRE_FALLBACK
    key = str(label).strip().lower().replace(" ", "_")
    if key in _GENRE_ALIAS_MAP:
        return _GENRE_ALIAS_MAP[key]
    if "opera" in key:
        return "Oper"
    if "class" in key or "orch" in key:
        return "Klassik"
    if "jazz" in key:
        return "Jazz"
    if "gospel" in key:
        return "Gospel"
    if "blues" in key:
        return "Blues"
    if "folk" in key or "country" in key:
        return "Folk"
    if "reggae" in key or "dub" in key:
        return "Reggae"
    if "hip" in key or "rap" in key:
        return "Hip-Hop"
    if "rnb" in key or "r&b" in key or "soul" in key:
        return "Soul/R&B"
    if "electro" in key or "edm" in key or "ambient" in key or "techno" in key:
        return "Electronic"
    if "metal" in key or "rock" in key or "punk" in key:
        return "Rock"
    if "pop" in key:
        return "Pop"
    return _CANONICAL_GENRE_FALLBACK


class SemanticAudioPhase(PhaseInterface):
    """Reine Analyse-Phase: DSP Feature Extraction (BPM, Key, Genre, Loudness)."""

    PHASE_ID = "phase_53_semantic_audio"
    PHASE_NAME = "Semantic Audio Analysis"
    PHASE_DESCRIPTION = (
        "Analysiert Audio semantisch auf drei Ebenen: "
        "Tier-1 LAION-CLAP (text-audio-aligned 512-dim Embeddings, Genre/Instrument-Tagging), "
        "Tier-0 BEATs iter3 (AudioSet-527), DSP-Fallback (Chromagramm/Spektralzentroid). "
        "Extrahiert BPM, Tonart, Genre-Hint, Loudness-Klasse — OHNE Audio zu verändern."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name=self.PHASE_NAME,
            category=PhaseCategory.METADATA,
            priority=2,
            version="3.0.0",
            dependencies=[],
            estimated_time_factor=0.08,
            memory_requirement_mb=80,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.75,
            description=self.PHASE_DESCRIPTION,
        )

    def process(
        self, audio: np.ndarray, sample_rate: int = 48000, material_type: str = "unknown", **kwargs
    ) -> PhaseResult:
        check_ml_model_ready("BEATs", phase_name="53")
        check_ml_model_ready("LAION-CLAP", phase_name="53")
        """
        Analysiert Audio semantisch, gibt unverändertes Audio zurück.

        PhaseResult.metadata enthält:
            bpm, key, genre_hint, loudness_class, crest_factor, spectral_centroid_hz
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        if effective_strength <= 1e-6:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        mono = _mono(audio.astype(np.float64))

        bpm = _estimate_bpm(mono, sample_rate)
        key = _estimate_key(mono, sample_rate)
        genre_hint_raw = _estimate_genre_hint(mono, sample_rate)
        genre_hint = _canonicalize_genre_hint(genre_hint_raw)
        genre_hint_source = "dsp"
        genre_hint_confidence = 0.25 if genre_hint != _CANONICAL_GENRE_FALLBACK else 0.0

        rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
        peak = float(np.max(np.abs(mono))) + 1e-12
        crest_factor = peak / rms
        # Annähernde LUFS-Näherung (nur Energiebasis)
        lufs_approx = 20.0 * np.log10(rms) - 0.691  # vereinfacht

        if lufs_approx > -9.0:
            loudness_class = "very_loud_mastered"
        elif lufs_approx > -14.0:
            loudness_class = "loud_streaming"
        elif lufs_approx > -23.0:
            loudness_class = "moderate"
        else:
            loudness_class = "quiet_dynamic"

        f, psd = sig.periodogram(mono, fs=sample_rate, window="hann")
        centroid = float(np.sum(f * psd) / (np.sum(psd) + 1e-12))

        # ── Tier-1: LAION-CLAP (text-audio aligned, 512-dim embeddings) ─────────────
        # Primary ML semantic path. Provides richer genre/instrument classification
        # than BEATs because CLAP is trained with natural-language supervision —
        # enabling zero-shot genre inference beyond AudioSet ontology.
        # Falls back to BEATs (Tier-0) transparently on OOM / missing model.
        _clap_top_genres: list[tuple[str, float]] = []
        _clap_instruments: list[str] = []
        _clap_embedding_32: list[float] = []
        _clap_model_used = "none"
        _clap_confidence = 0.0
        _clap_succeeded = False
        _clap_enabled = _clap_allowed_in_current_context()
        if _clap_enabled:
            try:
                if _clap_factory is None:
                    raise ImportError("LAION-CLAP Plugin nicht verfügbar")

                if _try_allocate_ml_budget("CLAP_phase53", 0.40):
                    try:
                        _clap_result = _clap_factory().tag(audio.astype(np.float32), sample_rate)
                        _clap_top_genres = sorted(_clap_result.genre_tags.items(), key=lambda x: x[1], reverse=True)[:5]
                        _clap_instruments = _clap_result.top_instruments(n=5)
                        emb = _clap_result.embedding
                        _clap_embedding_32 = [float(x) for x in emb.flatten()[:32].tolist()]
                        _clap_model_used = _clap_result.model_used
                        _clap_confidence = float(_clap_result.confidence)
                        # §v10.x Fallback-Hygiene (Befund 2026-08-22): Der DSP-
                        # Fallback synthetisiert Pseudo-Labels aus DSP-Features
                        # („blues“/„cello“ für einen Schlager) — diese dürfen
                        # einen existierenden Genre-Hint NICHT überschreiben.
                        # Nur echte Modell-Ergebnisse (laion_clap) überschreiben.
                        _clap_is_fallback = str(_clap_model_used) == "panns_fallback"
                        # Override DSP genre_hint when CLAP is confident enough
                        if _clap_top_genres and _clap_top_genres[0][1] >= 0.35 and not _clap_is_fallback:
                            genre_hint = _canonicalize_genre_hint(_clap_top_genres[0][0])
                            genre_hint_source = "clap"
                            genre_hint_confidence = float(_clap_top_genres[0][1])
                            _clap_succeeded = True
                        elif _clap_is_fallback:
                            logger.info(
                                "Verarbeitungsschritt 53: CLAP-DSP-Heuristik (advisory) — Genre-Hint bleibt: %s",
                                genre_hint if isinstance(genre_hint, str) else "kein Hint",
                            )
                        logger.info(
                            "Verarbeitungsschritt 53: CLAP OK (model=%s, conf=%.2f, top_genre=%s, instruments=%s)",
                            _clap_model_used,
                            _clap_confidence,
                            _clap_top_genres[:2],
                            _clap_instruments[:2],
                        )
                    except Exception as _clap_err:
                        logger.debug(
                            "Verarbeitungsschritt 53: CLAP tagging fehlgeschlagen (%s) — BEATs-Ersatzpfad", _clap_err
                        )
                    finally:
                        _release_ml_budget("CLAP_phase53")
            except Exception as _clap_imp_err:
                logger.debug(
                    "Verarbeitungsschritt 53: CLAP-Import nicht verfügbar (%s) — BEATs-Ersatzpfad", _clap_imp_err
                )
        else:
            _clap_model_used = "disabled_runtime_context"
            logger.info("Verarbeitungsschritt 53: CLAP deaktiviert (pytest/safe-Validierung) — BEATs/DSP aktiv")

        # ── Tier-0: BEATs iter3 Audio Tagging (SOTA §4.4) ───────────────────────────
        # AudioSet-527-Klassifikation für semantisch reichere Pipeline-Metadaten.
        # Runs after CLAP; only overrides genre_hint when CLAP did not succeed.
        _beats_tags: dict[str, float] = {}
        _beats_model_used = "dsp_only"
        _beats_embedding: list[float] = []
        _beats_top_k: list[tuple[str, float]] = []
        try:
            if _beats_factory is None:
                raise ImportError("BEATs Plugin nicht verfügbar")

            if _try_allocate_ml_budget("BEATs_phase53", 0.09):
                try:
                    _beats_result = _beats_factory().get_tags(audio, sample_rate, top_k=15)
                    _ = _beats_result.tags
                    _beats_model_used = _beats_result.model_used
                    _beats_top_k = _beats_result.top_k
                    # Speichere die ersten 32 Embedding-Dimensionen (Transport-safe)
                    _beats_embedding = [float(x) for x in _beats_result.embeddings[:32].tolist()]
                    # Überschreibe Genre-Hint nur wenn CLAP nicht erfolgreich war
                    if not _clap_succeeded and _beats_top_k:
                        _top_tag, _top_conf = _beats_top_k[0]
                        if _top_conf >= 0.40:
                            genre_hint = _canonicalize_genre_hint(_top_tag)
                            genre_hint_source = "beats"
                            genre_hint_confidence = float(_top_conf)
                    logger.info(
                        "Verarbeitungsschritt 53: BEATs OK (model=%s, top=%s)",
                        _beats_model_used,
                        [f"{t}({c:.2f})" for t, c in _beats_top_k[:3]],
                    )
                except Exception as _beats_err:
                    logger.debug(
                        "Verarbeitungsschritt 53: BEATs tagging fehlgeschlagen (%s) — DSP-Ersatzpfad", _beats_err
                    )
                finally:
                    _release_ml_budget("BEATs_phase53")
        except Exception as _imp_err:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("Verarbeitungsschritt 53: BEATs-Import nicht verfügbar (%s) — DSP-only", _imp_err)

        meta = {
            "bpm": round(bpm, 1),
            "key": key,
            "genre_hint": genre_hint,
            "genre_hint_raw_dsp": genre_hint_raw,
            "genre_hint_source": genre_hint_source,
            "genre_hint_confidence": round(float(genre_hint_confidence), 3),
            "loudness_class": loudness_class,
            "lufs_approx": round(float(lufs_approx), 1),
            "crest_factor": round(float(crest_factor), 2),
            "spectral_centroid_hz": round(float(centroid), 1),
            "phase_locality_factor": phase_locality_factor,
            "effective_strength": effective_strength,
            # CLAP semantic tags — Tier-1 (text-audio aligned, 512-dim)
            "clap_model_used": _clap_model_used,
            "clap_confidence": round(_clap_confidence, 3),
            "clap_top_genres": [{"genre": g, "confidence": round(c, 3)} for g, c in _clap_top_genres[:5]],
            "clap_top_instruments": _clap_instruments[:5],
            "clap_embedding_32": _clap_embedding_32,
            "clap_enabled": bool(_clap_enabled),
            # BEATs semantic tags — Tier-0 (AudioSet-527)
            "beats_model_used": _beats_model_used,
            "beats_top_tags": [{"tag": t, "confidence": round(c, 3)} for t, c in _beats_top_k[:10]],
            "beats_embedding_32": _beats_embedding,
        }

        logger.info(
            "Verarbeitungsschritt 53 SemanticAudio: BPM=%.1f, Key=%s, Genre=%s, Loudness=%s",
            bpm,
            key,
            genre_hint,
            loudness_class,
        )

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=audio,  # UNVERÄNDERT — Kategorie METADATA
            execution_time_seconds=time.time() - t0,
            metadata=meta,
            metrics={"bpm": bpm, "crest_factor": crest_factor, "effective_strength": effective_strength},
        )
