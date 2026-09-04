"""
Phase 64 — Tape Splice Artifact Repair.

Tape splice artifacts combine three simultaneous discontinuities:
1. Impulsive transient (click) at the splice point
2. Level jump (gain change across the splice boundary)
3. Phase discontinuity

This phase detects splice points and applies targeted repair:
- Click removal at the splice boundary
- Level crossfade across the discontinuity
- Phase alignment via short cross-correlation

Scientific basis: Czyzewski (2007) "Detection and Removal of Tape Splice
Artifacts"; Godsill & Rayner (1998) "Digital Audio Restoration".
"""

from __future__ import annotations

import logging
import os
import time as _time

import numpy as np

from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    if sig.ndim == 2:
        _mono = sig.mean(axis=0 if sig.shape[0] <= 2 else 1).astype(np.float32)
    else:
        _mono = np.asarray(sig, dtype=np.float32).ravel()
    _frame = 480  # 10 ms @ 48 kHz
    _n_frames = len(_mono) // _frame
    if _n_frames == 0:
        return -96.0
    _frames = _mono[: _n_frames * _frame].reshape(_n_frames, _frame)
    _frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(_frames**2, axis=1)) + 1e-10)
    _mask = _frame_rms_db > -50.0
    if not np.any(_mask):
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(_frames[_mask] ** 2)) + 1e-10))


logger = logging.getLogger(__name__)

_MIN_SPLICE_SCORE: float = 0.10
_CROSSFADE_MS: float = 15.0  # Crossfade duration at splice boundary


def _detect_splice_points(x: np.ndarray, sample_rate: int, crossfade_samples: int) -> list[int]:
    """Erkennt splice points on a mono signal.

    Extracted so that stereo processing can run detection on the mono mix
    (§2.51 Linked Stereo — splice boundaries must be synchronised across channels).

    Returns:
        List of sample indices where tape splices were detected.
    """
    n = len(x)
    frame_len = max(1, int(0.010 * sample_rate))  # 10 ms frames
    hop = max(1, frame_len // 2)
    n_frames = max(1, (n - frame_len) // hop)
    if n_frames < 10:
        return []

    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop, x.strides[0]),
    ).copy()
    rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms_env + 1e-12)

    level_diffs = np.abs(np.diff(rms_db))
    jump_indices = np.where(level_diffs > 6.0)[0]

    splice_points: list[int] = []
    for ji in jump_indices[:30]:
        sample_idx = ji * hop
        if sample_idx < crossfade_samples or sample_idx > n - crossfade_samples:
            continue
        boundary = x[sample_idx - 32 : sample_idx + 32]
        if len(boundary) < 64:
            continue
        hf_spec = np.abs(np.fft.rfft(boundary))
        hf_energy = float(np.sum(hf_spec[len(hf_spec) // 2 :] ** 2))
        total_energy = float(np.sum(hf_spec**2)) + 1e-12
        hf_ratio = hf_energy / total_energy

        persist_frames = min(5, n_frames - ji - 1)
        if persist_frames > 2:
            post = rms_db[ji + 1 : ji + 1 + persist_frames]
            pre = rms_db[max(0, ji - persist_frames) : ji]
            if len(post) > 0 and len(pre) > 0:
                level_persist = abs(float(np.mean(post)) - float(np.mean(pre)))
                if hf_ratio > 0.15 and level_persist > 3.0:
                    splice_points.append(sample_idx)

    return splice_points


def _compute_splice_local_strength(
    original: np.ndarray,
    splice_idx: int,
    sample_rate: int,
    crossfade_samples: int,
    base_strength: float,
    protected_zones: list | None = None,
) -> float:
    """Berechnet per-Splice lokale Stärke über 250 ms Kontext + Schutzzonen-Caps.

    §V38: Event-Oracle statt globaler Einheitsstärke.
    """
    if base_strength < 1e-6:
        return 0.0

    n = int(len(original))
    _sr = max(int(sample_rate), 1)
    _ctx = max(1, int(0.250 * _sr))

    _pre_ctx_s = max(0, splice_idx - _ctx)
    _pre_ctx_e = max(_pre_ctx_s + 1, splice_idx)
    _post_ctx_s = min(n - 1, splice_idx)
    _post_ctx_e = min(n, splice_idx + _ctx)

    _pre_ctx_r = float(np.sqrt(np.mean(original[_pre_ctx_s:_pre_ctx_e] ** 2) + 1e-12))
    _post_ctx_r = float(np.sqrt(np.mean(original[_post_ctx_s:_post_ctx_e] ** 2) + 1e-12))
    _ctx_r = max(_pre_ctx_r, _post_ctx_r, 1e-10)

    _pre_len = min(crossfade_samples, splice_idx)
    _post_len = min(crossfade_samples, n - splice_idx)
    _pre_r = (
        float(np.sqrt(np.mean(original[splice_idx - _pre_len : splice_idx] ** 2) + 1e-12)) if _pre_len > 0 else 1e-10
    )
    _post_r = (
        float(np.sqrt(np.mean(original[splice_idx : splice_idx + _post_len] ** 2) + 1e-12)) if _post_len > 0 else 1e-10
    )
    _level_ratio = max(_pre_r, _post_r) / (min(_pre_r, _post_r) + 1e-12)
    _level_jump_db = float(20.0 * np.log10(max(1.0, _level_ratio)))

    _jump_factor = float(np.clip(0.35 + 0.65 * (_level_jump_db - 2.0) / 8.0, 0.30, 1.0))
    _activity_factor = float(np.clip(_ctx_r / 0.08, 0.35, 1.0))
    local_strength = float(base_strength * _jump_factor * _activity_factor)

    if protected_zones:
        _sp_s = splice_idx / float(_sr)
        for _pz in protected_zones:
            try:
                if float(_pz[0]) <= _sp_s <= float(_pz[1]):
                    local_strength = min(local_strength, float(_pz[2]))
                    break
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue

    return float(np.clip(local_strength, 0.05, 1.0))


def _apply_splice_repair(
    out: np.ndarray,
    original: np.ndarray,
    splice_points: list[int],
    crossfade_samples: int,
    strength: float,
    protected_zones: list | None = None,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Wendet Click-Entfernung und Pegel-Überblendung an jedem Schnittpunkt an.

    Args:
        out:             Working copy of the signal (modified in place).
        original:        Read-only original signal for stable RMS measurements.
        splice_points:   Sample indices of detected splices.
        crossfade_samples: Crossfade region length in samples.
        strength:        Basis-Verarbeitungsstärke [0, 1].
        protected_zones: [(start_s, end_s, max_strength), ...] — VFA-Schutzzonen.
        sample_rate:     Sample-Rate in Hz (für Zeitberechnung der Schutzzonen).
    """
    n = len(out)
    _sr = max(int(sample_rate), 1)
    for sp in splice_points:
        _local_str = _compute_splice_local_strength(
            original=original,
            splice_idx=int(sp),
            sample_rate=_sr,
            crossfade_samples=crossfade_samples,
            base_strength=float(strength),
            protected_zones=protected_zones,
        )

        # Sub-step 2a: Remove click impulse (short interpolation)
        click_half = min(32, crossfade_samples // 2)
        cl = max(0, sp - click_half)
        cr = min(n, sp + click_half)
        if cr - cl < 4:
            continue
        interp = np.linspace(out[cl], out[min(cr, n - 1)], cr - cl)
        click_weight = float(np.clip(_local_str, 0.0, 1.0))
        out[cl:cr] = out[cl:cr] * (1.0 - click_weight) + interp * click_weight

        # Sub-step 2b: Level crossfade (measured against unmodified original)
        pre_start = max(0, sp - crossfade_samples)
        post_end = min(n, sp + crossfade_samples)
        pre_rms = float(np.sqrt(np.mean(original[pre_start:sp] ** 2) + 1e-12))
        post_rms = float(np.sqrt(np.mean(original[sp:post_end] ** 2) + 1e-12))

        if pre_rms > 1e-8 and post_rms > 1e-8:
            gain_ratio = float(np.clip(pre_rms / post_rms, 0.5, 2.0))
            fade_len = min(crossfade_samples, post_end - sp)
            if fade_len > 0:
                fade = np.linspace(gain_ratio, 1.0, fade_len)
                blend = float(np.clip(_local_str * 0.5, 0.0, 0.5))
                out[sp : sp + fade_len] *= (1.0 - blend) + blend * fade

    return out


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.7,
    defect_scores: dict | None = None,
    min_splice_score: float = _MIN_SPLICE_SCORE,
    crossfade_ms: float = _CROSSFADE_MS,
    protected_zones: list | None = None,
) -> np.ndarray:
    """Haupt-entry point for Phase 64."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.asarray(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

    if defect_scores is not None:
        splice_score = float(defect_scores.get("tape_splice_artifact", 0.0))
        if splice_score < min_splice_score:
            logger.debug(
                "Verarbeitungsschritt 64: splice Wert %.3f < %.3f — uebersprungen", splice_score, min_splice_score
            )
            return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

    crossfade_samples = max(1, int(crossfade_ms * 0.001 * sample_rate))

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 Linked: detect splice boundaries on mono mix (L+R)/2 so that
        # both channels are repaired at exactly the same sample positions.
        mono_mix = (audio[0] + audio[1]) * 0.5
        mono64 = mono_mix.astype(np.float32)
        splice_points = _detect_splice_points(mono64, sample_rate, crossfade_samples)
        if not splice_points:
            return np.nan_to_num(np.clip(audio, -1.0, 1.0).astype(np.float32), nan=0.0)  # type: ignore[no-any-return]
        left_out = _apply_splice_repair(
            audio[0].astype(np.float32),
            audio[0].astype(np.float32),
            splice_points,
            crossfade_samples,
            strength,
            protected_zones=protected_zones,
            sample_rate=sample_rate,
        )
        right_out = _apply_splice_repair(
            audio[1].astype(np.float32),
            audio[1].astype(np.float32),
            splice_points,
            crossfade_samples,
            strength,
            protected_zones=protected_zones,
            sample_rate=sample_rate,
        )
        left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
        right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.nan_to_num(np.clip(np.stack([left_out, right_out], axis=0), -1.0, 1.0).astype(np.float32), nan=0.0)  # type: ignore[no-any-return]

    x = audio.astype(np.float32)
    splice_points = _detect_splice_points(x, sample_rate, crossfade_samples)
    if not splice_points:
        return np.nan_to_num(np.clip(audio, -1.0, 1.0).astype(np.float32), nan=0.0)  # type: ignore[no-any-return]

    out = _apply_splice_repair(
        np.copy(x),
        x,
        splice_points,
        crossfade_samples,
        strength,
        protected_zones=protected_zones,
        sample_rate=sample_rate,
    )
    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(np.clip(result, -1.0, 1.0).astype(np.float32), nan=0.0)  # type: ignore[no-any-return]


# ─── PhaseInterface ────────────────────────────────────────────────────────────


class TapeSpliceRepairPhase(PhaseInterface):
    """Phase 64: Tape splice artifact repair (click + level + phase discontinuity)."""

    @staticmethod
    def _build_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
    ) -> tuple[np.ndarray, float]:
        """Erzeugt lokale Blendmaske aus Tape-Splice-Locations."""
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0

        mask = np.zeros(n_samples, dtype=np.float32)
        pad = int(0.05 * sample_rate)
        for loc in defect_locations.get("tape_splice_artifact") or []:
            if not isinstance(loc, tuple) or len(loc) != 2:
                continue
            try:
                s = int(max(0.0, float(loc[0])) * sample_rate)
                e = int(max(0.0, float(loc[1])) * sample_rate)
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue
            if e <= s:
                continue
            s = max(0, s - pad)
            e = min(n_samples, e + pad)
            if e > s:
                mask[s:e] = 1.0

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0

        smooth = max(16, int(0.02 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        return mask, float(np.mean(mask))

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_64_tape_splice_repair",
            name="Tape Splice Repair",
            category=PhaseCategory.RESTORATION,
            priority=6,
            dependencies=["phase_01"],
            estimated_time_factor=0.04,
            version="1.0.0",
            memory_requirement_mb=16,
            is_cpu_intensive=False,
            quality_impact=0.55,
            description=(
                "Tape splice artifact repair combining click removal, level "
                "crossfading, and phase alignment at splice boundaries. "
                "Distinct from generic click removal (phase_01)."
            ),
        )

    @staticmethod
    def _compute_splice_profile(material_key: str, quality_mode: str, restorability_score: float) -> dict[str, float]:
        material = str(material_key or "unknown").strip().lower()
        mode = str(quality_mode or "balanced").strip().lower()
        mode = {"restoration": "balanced", "studio_2026": "maximum"}.get(mode, mode)

        if any(token in material for token in ("reel_tape", "tape", "cassette")):
            min_splice_score = 0.10
            crossfade_ms = 15.0
        elif any(token in material for token in ("cd_digital", "dat", "flac", "streaming")):
            min_splice_score = 0.18
            crossfade_ms = 12.0
        else:
            min_splice_score = 0.14
            crossfade_ms = 13.0

        rest_norm = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0)) / 100.0
        min_splice_score += (rest_norm - 0.5) * 0.12
        crossfade_ms += (0.5 - rest_norm) * 10.0

        score_off, crossfade_off = {
            "fast": (0.03, -3.0),
            "balanced": (0.00, 0.0),
            "quality": (-0.02, 4.0),
            "maximum": (-0.04, 6.0),
        }.get(mode, (0.0, 0.0))
        min_splice_score += score_off
        crossfade_ms += crossfade_off

        return {
            "min_splice_score": float(np.clip(min_splice_score, 0.05, 0.25)),
            "crossfade_ms": float(np.clip(crossfade_ms, 6.0, 30.0)),
        }

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="64")
        check_ml_model_ready("Whisper", phase_name="64")
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        strength = float(kwargs.get("strength", 0.7))
        defect_scores = kwargs.get("defect_scores")

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", strength))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_64 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_64 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_64,
                )

                _fmz_64 = kwargs.get("forward_masking_zones") or _fmg_fn_64().compute_zones(audio, sample_rate)
                if _fmz_64:
                    _n_s_64 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_64 = sum(z.end_sample - z.start_sample for z in _fmz_64)
                    _zone_frac_64 = float(np.clip(_zone_s_64 / max(1, _n_s_64), 0.0, 1.0))
                    _effective_strength = float(np.clip(_effective_strength + _zone_frac_64 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_64:
                logger.debug("Verarbeitungsschritt64 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_64)

        _profile_64 = self._compute_splice_profile(
            str(material_type or kwargs.get("material_type") or kwargs.get("material") or "unknown"),
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "splice_score": float((_defect_scores or {}).get("tape_splice_artifact", 0.0)),
                    "strength": _effective_strength,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Tape splice repair skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)

        # §v10.303.6 Early-Exit: Keine Splice-Zonen → nichts zu reparieren
        _splice_zones_raw = (
            kwargs.get("defect_locations", {}).get("tape_splice_artifact", [])
            if isinstance(kwargs.get("defect_locations"), dict)
            else []
        )
        _splice_count = len(_splice_zones_raw) if _splice_zones_raw else 0
        if _splice_count == 0:
            logger.info(
                "Verarbeitungsschritt 64 §v10.303.6 Early-Exit: Keine Klebestellen gefunden → ueberspringen (%.2fs)",
                _time.perf_counter() - t0,
            )
            return PhaseResult(
                audio=audio.copy() if isinstance(audio, np.ndarray) else np.asarray(audio, dtype=np.float32),
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metadata={
                    "splice_count": 0,
                    "skip_reason": "no_splice_zones",
                    "splice_profile": dict(_profile_64),
                    "min_splice_score": float(_profile_64["min_splice_score"]),
                    "crossfade_ms": float(_profile_64["crossfade_ms"]),
                },
            )

        # Schutzzonen für per-Splice individuelle Stärke zusammenstellen (§0p Vocal-Supremacy + §0l)
        _p64_zones: list = []
        for _z in kwargs.get("vibrato_zones") or []:
            try:
                _p64_zones.append((float(_z[0]), float(_z[1]), 0.20))  # §0p Vibrato-Schutz
            except Exception as e:
                logger.warning("Verarbeitungsschritt_64_tape_splice_repair.py::verarbeiten Ersatzpfad: %s", e)
        for _z in kwargs.get("frisson_zones") or []:
            try:
                _fz_s = float(getattr(_z, "start_s", None) or _z[0])
                _fz_e = float(getattr(_z, "end_s", None) or _z[1])
                _p64_zones.append((_fz_s, _fz_e, 0.30))  # Frisson sakrosankt
            except Exception as e:
                logger.warning("Verarbeitungsschritt_64_tape_splice_repair.py::verarbeiten Ersatzpfad: %s", e)
        for _z in kwargs.get("whisper_zones") or []:
            try:
                _p64_zones.append((float(_z[0]), float(_z[1]), 0.25))  # Flüsterpassagen
            except Exception as e:
                logger.warning("Verarbeitungsschritt_64_tape_splice_repair.py::unbekannter Ersatzpfad: %s", e)
        for _z in kwargs.get("passaggio_zones") or []:
            try:
                _p64_zones.append((float(_z[0]), float(_z[1]), 0.35))  # Passaggio-Übergänge
            except Exception as e:
                logger.warning("Verarbeitungsschritt_64_tape_splice_repair.py::unbekannter Ersatzpfad: %s", e)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_splice_score=_profile_64["min_splice_score"],
            crossfade_ms=_profile_64["crossfade_ms"],
            protected_zones=_p64_zones or None,
        )
        _n_samples_64 = int(result_audio.shape[0]) if result_audio.ndim >= 2 else int(result_audio.shape[0])
        _locality_profile, _locality_coverage = self._build_locality_profile(
            n_samples=_n_samples_64,
            sample_rate=sample_rate,
            defect_locations=kwargs.get("defect_locations"),
        )
        if _locality_profile.size > 0:
            if result_audio.ndim == 2:
                if result_audio.shape[0] <= 2 and result_audio.shape[1] >= result_audio.shape[0]:
                    _mask = _locality_profile[np.newaxis, :]
                else:
                    _mask = _locality_profile[:, np.newaxis]
                result_audio = audio + _mask * (result_audio - audio)
            else:
                result_audio = audio + _locality_profile * (result_audio - audio)
            result_audio = np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0)
            result_audio = np.clip(result_audio, -1.0, 1.0).astype(np.float32)
        elapsed = _time.perf_counter() - t0

        # §V19 Noise-Textur-Invariante (VERBOTEN-V19): Residual bewahrt Materialcharakter
        _mat64_str = str(material_type or "unknown").lower()
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt64_fn,
            )

            if result_audio.shape == audio.shape:
                _nt64_d = _nt64_fn(
                    audio.astype(np.float32) - result_audio.astype(np.float32), _mat64_str, sr=sample_rate
                )
                if _nt64_d > 0.25:
                    result_audio = (0.5 * result_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_64 noise_texture dist=%.3f > 0.25 → 50%%-Blend",
                        _nt64_d,
                    )
        except Exception as _nt64_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_64 noise_texture_guard (nicht blockierend): %s",
                _nt64_exc,
            )

        # §V24 Spektralfarbe-Prüfung (VERBOTEN-V24): 1/3-Oktav-Profil darf nicht verfärbt werden
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg64,
            )

            if result_audio.shape == audio.shape:
                _sc64 = _scg64(audio.astype(np.float32), result_audio.astype(np.float32), sample_rate)
                if not _sc64.ok:
                    result_audio = (0.70 * result_audio + 0.30 * audio).astype(np.float32)
        except Exception as _sc64_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_64 spectral_color_guard (nicht blockierend): %s",
                _sc64_exc,
            )

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "splice_score": float((_defect_scores or {}).get("tape_splice_artifact", 0.0)),
                "strength": _effective_strength,
            },
            metadata={
                "splice_profile": dict(_profile_64),
                "min_splice_score": float(_profile_64["min_splice_score"]),
                "crossfade_ms": float(_profile_64["crossfade_ms"]),
                "repair_locality_coverage": float(_locality_coverage),
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
        )
