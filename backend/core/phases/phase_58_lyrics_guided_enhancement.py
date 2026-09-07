"""Phase 58 — Lyrics-Guided Enhancement (§2.36 PFLICHT).

Whisper-Tiny ONNX → wav2vec2 phoneme alignment → ContentAwareProcessor.
Runs as post-processing module AFTER the PMGG chain (§7.7 invariant).
NOT ML-deterministic in the PMGG sense — excluded from PMGG delta-checks.

Privacy invariant: lyrics text MUST NEVER appear in any log, metadata field,
or RestorationResult. Only phoneme-class labels (fricative/plosive/…) are kept.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .phase_interface import (
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
    create_phase_result,
)

logger = logging.getLogger(__name__)

# Minimum vocal probability (PANNs) to activate LGE (§2.9 invariant).
_VOCAL_PROB_MIN: float = 0.30

# Latency budget: 8 s per minute of audio (§2.36).
_LATENCY_BUDGET_S_PER_MIN: float = 8.0


class Phase58LyricsGuidedEnhancement(PhaseInterface):
    """§2.36 Lyrics-guided saliency enhancement via phoneme-class boost factors.

    Pipeline position: post-PMGG-chain, active only when vocals are detected.
    The phase wraps ``LyricsGuidedEnhancement.enhance()`` which applies
    segment-specific saliency boosts (fricative ×1.55, plosive ×1.40,
    vowel_stressed ×1.35, silence ×0.70) to achieve physical vocal intimacy.

    ML components loaded lazily via ``get_lyrics_guided_enhancement()`` (singleton).
    Memory: ~512 MB (Whisper-Tiny 39 MB + wav2vec2 125 MB + buffer overhead).
    """

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_58_lyrics_guided_enhancement",
            name="Lyrics-gestütztes Enhancement (§2.36)",
            category=PhaseCategory.ENHANCEMENT,
            priority=6,
            version="1.0.0",
            dependencies=[],
            estimated_time_factor=0.20,
            memory_requirement_mb=512,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description=(
                "§2.36 Whisper-Tiny ONNX → wav2vec2 forced-alignment → "
                "ContentAwareProcessor; phoneme-class saliency boosts; "
                "runs post-PMGG; latency ≤ 8 s/min audio."
            ),
            defect_types=["vocal_noise", "sibilant_distortion"],
            musical_goals=["artikulation", "natuerlichkeit"],
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48_000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        assert sample_rate == 48_000, f"phase_58: expected sr=48000, got {sample_rate}"

        # §2.47 PMGG-Retry: locality_factor skaliert LGE-Intensität bei Retries
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_58 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_58 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_58,
                )

                _fmz_58 = kwargs.get("forward_masking_zones") or _fmg_fn_58().compute_zones(audio, sample_rate)
                if _fmz_58:
                    _n_s_58 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_58 = sum(z.end_sample - z.start_sample for z in _fmz_58)
                    _zone_frac_58 = float(np.clip(_zone_s_58 / max(1, _n_s_58), 0.0, 1.0))
                    _effective_strength = float(np.clip(_effective_strength + _zone_frac_58 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_58:
                logger.debug("Verarbeitungsschritt58 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_58)

        if _effective_strength <= 0.0:
            return create_phase_result(
                audio=audio,
                modifications={
                    "lge_skipped": True,
                    "reason": "zero effective strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                },
                warnings=["LGE skipped: zero effective strength"],
                metadata={
                    "lge_active": False,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                },
                ml_used=False,
                quality_estimate=1.0,
                execution_time_seconds=0.0,
            )

        # §2.36 Pre-computed transcription from original audio (passed by UV3 via phase_kwargs).
        # If available and not a fallback, use saliency path instead of full re-transcription.
        pre_transcription: Any = kwargs.get("pre_transcription")

        # ── Vocal-gate (§2.9) ────────────────────────────────────────────────
        vocal_prob: float = float(kwargs.get("vocal_probability", 1.0))
        if vocal_prob < _VOCAL_PROB_MIN:
            logger.debug(
                "Verarbeitungsschritt_58_lyrics_guided_enhancement: uebersprungen — vocal_prob=%.3f < %.2f",
                vocal_prob,
                _VOCAL_PROB_MIN,
            )
            return create_phase_result(
                audio=audio,
                modifications={"lge_skipped": True, "vocal_prob": vocal_prob},
                warnings=["LGE skipped: vocal probability below threshold"],
                metadata={"lge_active": False, "vocal_probability": vocal_prob},
                ml_used=False,
                quality_estimate=1.0,
                execution_time_seconds=0.0,
            )

        # ── Latency budget check ──────────────────────────────────────────────
        dur_s: float = float(audio.shape[0] if audio.ndim == 1 else audio.shape[0]) / max(1, sample_rate)
        latency_budget_s: float = (dur_s / 60.0) * _LATENCY_BUDGET_S_PER_MIN

        # ── Load singleton LGE (lazy, DSP-fallback on ImportError) ───────────
        load_attempts = 0
        retry_attempted = False
        lge = None
        load_error: Exception | None = None
        for attempt in range(2):
            load_attempts = attempt + 1
            retry_attempted = retry_attempted or attempt > 0
            try:
                import importlib as _importlib  # pylint: disable=import-outside-toplevel

                _lge_mod = _importlib.import_module("backend.core.lyrics_guided_enhancement")
                lge = _lge_mod.get_lyrics_guided_enhancement()
                break
            except Exception as exc:
                load_error = exc

        if lge is None:
            _load_exc: BaseException = load_error if load_error is not None else RuntimeError("lge unavailable")
            logger.warning(
                "Verarbeitungsschritt_58_lyrics_guided_enhancement: LGE nicht verfuegbar (%s) — DSP passthrough",
                type(_load_exc).__name__,
            )
            return create_phase_result(
                audio=audio,
                modifications={"lge_fallback": "passthrough", "reason": type(_load_exc).__name__},
                warnings=[f"LGE unavailable ({type(_load_exc).__name__}) — passthrough"],
                metadata={
                    "lge_active": False,
                    "lge_error": type(_load_exc).__name__,
                    "retry_attempted": retry_attempted,
                    "load_attempts": load_attempts,
                },
                ml_used=False,
                quality_estimate=0.95,
            )

        # ── Apply LGE (saliency path if pre_transcription available, else full enhance) ─────
        t0 = time.perf_counter()
        enhance_attempts = 0
        try:
            if pre_transcription is not None and not getattr(pre_transcription, "fallback_used", True):
                # Fast path: use pre-computed transcription from original audio (§2.36 Phase-Gate).
                import importlib as _importlib  # pylint: disable=import-outside-toplevel

                _lge_mod2 = _importlib.import_module("backend.core.lyrics_guided_enhancement")
                cap = _lge_mod2.get_content_aware_processor()
                n_smp = audio.shape[-1] if audio.ndim == 2 else len(audio)
                base_sal = np.ones(n_smp, dtype=np.float32)
                saliency = cap.compute_lyrics_saliency(base_sal, pre_transcription, sample_rate)
                if audio.ndim == 2 and audio.shape[0] <= 2:
                    audio_out = audio * saliency[np.newaxis, :]
                elif audio.ndim == 2:
                    audio_out = audio * saliency[:, np.newaxis]
                else:
                    audio_out = audio * saliency
                audio_out = np.clip(np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
                    np.float32
                )
                transcription = pre_transcription
                logger.info(
                    "Verarbeitungsschritt_58_lyrics_guided_enhancement: saliency path (pre-transcription) — %d segments",
                    len(pre_transcription.words) if pre_transcription is not None else 0,
                )
            else:
                last_enhance_exc: Exception | None = None
                for attempt in range(2):
                    enhance_attempts = attempt + 1
                    retry_attempted = retry_attempted or attempt > 0
                    try:
                        audio_out, transcription = lge.enhance(audio, sample_rate)
                        break
                    except Exception as exc:
                        last_enhance_exc = exc
                else:
                    assert last_enhance_exc is not None
                    raise last_enhance_exc
        except Exception as exc:
            logger.warning(
                "Verarbeitungsschritt_58_lyrics_guided_enhancement: verbessern() fehlgeschlagen (%s) — passthrough",
                type(exc).__name__,
            )
            return create_phase_result(
                audio=audio,
                modifications={"lge_fallback": "passthrough_on_error", "reason": type(exc).__name__},
                warnings=[f"LGE enhance() failed ({type(exc).__name__}) — passthrough"],
                metadata={
                    "lge_active": False,
                    "lge_error": type(exc).__name__,
                    "retry_attempted": retry_attempted,
                    "load_attempts": load_attempts,
                    "enhance_attempts": enhance_attempts,
                },
                ml_used=True,
                quality_estimate=0.95,
            )

        elapsed = time.perf_counter() - t0

        # ── Latency warning (§2.36) ───────────────────────────────────────────
        warnings: list[str] = []
        if elapsed > latency_budget_s and latency_budget_s > 0.0:
            logger.warning(
                "Verarbeitungsschritt_58_lyrics_guided_enhancement: latency %.2f s > Grenze %.2f s for %.1f s audio",
                elapsed,
                latency_budget_s,
                dur_s,
            )
            warnings.append(f"LGE latency {elapsed:.1f}s exceeds 8 s/min budget ({latency_budget_s:.1f}s)")

        # ── NaN/Inf guard + Soft-Knee clip (§III: 6 dB, 200 ms Hanning) ───
        audio_out = _soft_knee_clip(
            np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0),
        ).astype(np.float32)

        # §2.47 PMGG-Retry: phase_locality_factor als Post-hoc-Wet/Dry-Regler
        if _effective_strength < 1.0:
            audio_out = audio + _effective_strength * (audio_out - audio)
            audio_out = _soft_knee_clip(audio_out).astype(np.float32)

        # §2.36 Privacy-Pflicht: word count only — no text in metadata.
        n_words: int = len(transcription.words) if transcription is not None else 0

        # §2.46e HallucinationGuard: Additive Phase darf kein halluziniertes Material einführen (VERBOTEN)
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _chk_hg58,
            )

            _hg_mode_58 = str(kwargs.get("mode", "restoration"))
            _pre58_mono = (
                audio.mean(axis=0)
                if (audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2)
                else (audio.mean(axis=1) if audio.ndim == 2 else audio)
            )
            _post58_mono = (
                audio_out.mean(axis=0)
                if (audio_out.ndim == 2 and audio_out.shape[0] == 2 and audio_out.shape[1] > 2)
                else (audio_out.mean(axis=1) if audio_out.ndim == 2 else audio_out)
            )
            _hg58 = _chk_hg58(
                _pre58_mono.astype(np.float32),
                _post58_mono.astype(np.float32),
                sr=sample_rate,
                mode=_hg_mode_58,
                bw_extension_context=True,
            )
            if _hg58.requires_rollback:
                audio_out = audio.copy()
                logger.warning("§2.46e Verarbeitungsschritt_58 HallucinationGuard: rollback (spectral_novelty > 0.15)")
        except Exception as _hg58_exc:
            logger.debug("§2.46e Verarbeitungsschritt_58 HallucinationGuard (nicht blockierend): %s", _hg58_exc)

        return create_phase_result(
            audio=audio_out,
            modifications={
                "lge_active": True,
                "vocal_probability": vocal_prob,
                "n_phoneme_segments": n_words,
                "latency_s": round(elapsed, 3),
                "latency_budget_s": round(latency_budget_s, 3),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
            warnings=warnings,
            metadata={
                "lge_active": True,
                "vocal_probability": vocal_prob,
                "n_phoneme_segments": n_words,
                "retry_attempted": retry_attempted,
                "load_attempts": load_attempts,
                "enhance_attempts": enhance_attempts,
                # §2.36 Datenschutz: no lyrics text stored here
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
            ml_used=True,
            quality_estimate=min(1.0, 0.90 + 0.10 * min(1.0, vocal_prob)),
            execution_time_seconds=elapsed,
        )


# ── Soft-Knee Clip Helper (§III: 6 dB, 200 ms Hanning) ───────────────────────
def _soft_knee_clip(x: np.ndarray, knee_db: float = 6.0) -> np.ndarray:
    """Sigmoid-Soft-Knee statt Hard-Clamp (§III)."""
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak <= 1.0:
        return x
    overshoot_db = 20.0 * np.log10(peak / 1.0)
    # Sigmoid: σ((x - 1)/knee_width) → sanfte Begrenzung über 1.0 hinaus
    gain = 1.0 / (1.0 + np.exp(-overshoot_db / knee_db))
    return x * gain  # type: ignore[no-any-return]


# ── Singleton convenience accessor ───────────────────────────────────────────
_PHASE_58_SINGLETON: dict[str, Phase58LyricsGuidedEnhancement | None] = {"instance": None}
_PHASE_58_LOCK = threading.Lock()


def get_phase_58_lge() -> Phase58LyricsGuidedEnhancement:
    """Gibt the singleton Phase58LyricsGuidedEnhancement instance (thread-safe) zurück."""
    instance = _PHASE_58_SINGLETON["instance"]
    if instance is None:
        with _PHASE_58_LOCK:
            instance = _PHASE_58_SINGLETON["instance"]
            if instance is None:
                instance = Phase58LyricsGuidedEnhancement()
                _PHASE_58_SINGLETON["instance"] = instance
    return instance
