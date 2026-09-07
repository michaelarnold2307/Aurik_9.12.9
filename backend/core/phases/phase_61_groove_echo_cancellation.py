"""
Phase 61 — Groove Echo Cancellation.

Groove echo (pre-echo) occurs when a loud passage deforms the adjacent groove
wall, creating a ghost image ~1.8 s before (at 33⅓ rpm).  This is fundamentally
different from codec pre-echo (5–35 ms, handled by phase_23).

Algorithm:
1. Detect transient peaks (loud passages)
2. For each peak, compute revolution delay (RPM-dependent: 1.8s/1.35s/0.77s)
3. Template-match the ghost signal at the expected delay
4. Subtract the ghost component using adaptive spectral subtraction
5. Apply spectral gating to remove residual echo energy

Scientific basis: McDermott (2005) "Record Groove Physics";
Cannam (2006) "Echo Removal in Gramophone Recordings".
"""

from __future__ import annotations

import logging
import os
import time as _time

import numpy as np

logger = logging.getLogger(__name__)


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


_MIN_GROOVE_ECHO_SCORE: float = 0.10
_REVOLUTION_DELAYS_S: list[float] = [1.8, 1.35, 0.77]  # 33⅓, 45, 78 RPM
_SPECTRAL_SUBTRACTION_FLOOR_DB: float = -40.0


def _apply_groove_echo_mono(
    x_mono: np.ndarray,
    sample_rate: int,
    strength: float,
    defect_scores: dict | None,  # pylint: disable=unused-argument
    spectral_subtraction_floor_db: float,
) -> np.ndarray:
    """Echo-Cancellation auf einem Mono-Kanal. Interne Hilfsfunktion für §2.51."""
    x = np.asarray(x_mono, dtype=np.float32)
    n = len(x)
    sr = sample_rate
    out = np.copy(x)

    # Find transient peaks (top 5% of envelope)
    win = max(1, int(0.010 * sr))
    envelope = np.convolve(np.abs(x), np.ones(win) / win, mode="same")
    peak_thresh = float(np.percentile(envelope, 95))
    peak_indices = np.where(envelope > peak_thresh)[0]

    # Deduplicate peaks (>500 ms apart)
    min_gap = int(0.5 * sr)
    deduped: list[int] = []
    if len(peak_indices) > 0:
        deduped = [int(peak_indices[0])]
        for p in peak_indices[1:]:
            if int(p) - deduped[-1] > min_gap:
                deduped.append(int(p))
    peaks = deduped[:30]

    if not peaks:
        return _soft_knee_clip(x).astype(np.float32)  # type: ignore[no-any-return]

    for delay_s in _REVOLUTION_DELAYS_S:
        delay_samples = int(delay_s * sr)
        search_window = int(0.05 * sr)

        for peak_idx in peaks:
            ghost_center = peak_idx - delay_samples
            if ghost_center < search_window or ghost_center + search_window >= n:
                continue

            template_len = int(0.03 * sr)
            t_start = peak_idx
            t_end = min(n, t_start + template_len)
            template = x[t_start:t_end]
            if len(template) < 64:
                continue

            g_start = max(0, ghost_center - search_window)
            g_end = min(n, ghost_center + search_window + len(template))
            ghost_region = x[g_start:g_end]

            if len(ghost_region) < len(template):
                continue

            corr = np.correlate(ghost_region, template, mode="valid")
            if len(corr) == 0:
                continue
            best_offset = int(np.argmax(np.abs(corr)))
            best_corr = float(np.abs(corr[best_offset]))
            template_energy = float(np.sum(template**2))
            if template_energy < 1e-12:
                continue
            norm_corr = best_corr / (np.sqrt(template_energy * np.sum(ghost_region**2)) + 1e-12)

            if norm_corr < 0.15:
                continue

            ghost_start = g_start + best_offset
            ghost_end = min(n, ghost_start + len(template))
            ghost_len = ghost_end - ghost_start

            if ghost_len < 32:
                continue

            alpha = float(np.clip(strength * norm_corr * 1.5, 0.0, 0.8))
            ghost = x[ghost_start:ghost_end]
            ref = template[:ghost_len]

            n_fft = min(2048, ghost_len)
            spec_ghost = np.fft.rfft(ghost[:n_fft])
            spec_ref = np.fft.rfft(ref[:n_fft])

            mag_ghost = np.abs(spec_ghost)
            mag_ref = np.abs(spec_ref)
            ratio = mag_ghost / (mag_ref + 1e-12)
            scale = float(np.median(ratio[ratio < np.percentile(ratio, 80)]))
            scale = float(np.clip(scale, 0.01, 0.5))

            spec_clean = spec_ghost - alpha * scale * spec_ref
            floor = 10 ** (spectral_subtraction_floor_db / 20.0)
            mag_clean = np.maximum(floor, np.abs(spec_clean))
            spec_clean = mag_clean * np.exp(1j * np.angle(spec_ghost))

            cleaned = np.fft.irfft(spec_clean, n=n_fft)
            out[ghost_start : ghost_start + n_fft] = cleaned

    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return _soft_knee_clip(result).astype(np.float32)  # type: ignore[no-any-return]


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.6,
    defect_scores: dict | None = None,
    min_groove_echo_score: float = _MIN_GROOVE_ECHO_SCORE,
    spectral_subtraction_floor_db: float = _SPECTRAL_SUBTRACTION_FLOOR_DB,
) -> np.ndarray:
    """Haupt-entry point for Phase 61."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        ge_score = float(defect_scores.get("groove_echo", 0.0))
        if ge_score < min_groove_echo_score:
            logger.debug(
                "Verarbeitungsschritt 61: groove_echo Wert %.3f < %.3f — uebersprungen", ge_score, min_groove_echo_score
            )
            return _soft_knee_clip(audio)  # type: ignore[no-any-return]

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 M/S-Domain: Groove-Echo ist ein physikalisches Nadelphänomen,
        # das symmetrisch in beiden Kanälen auftritt → im Mid-Kanal konzentriert.
        # Verarbeitung: Echo-Cancellation auf Mid; Side mit 30 % Stärke (korreliert).
        if audio.shape[0] == 2 and audio.shape[1] != 2:
            left = audio[0].astype(np.float32)
            right = audio[1].astype(np.float32)
        else:
            left = audio[:, 0].astype(np.float32)
            right = audio[:, 1].astype(np.float32)
        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        # Echo-Erkennung + Subtraktion auf Mid-Kanal
        mid_clean = _apply_groove_echo_mono(mid, sample_rate, strength, defect_scores, spectral_subtraction_floor_db)
        # Side mit reduzierter Stärke (korreliertes, symmetrisches Phänomen)
        side_clean = _apply_groove_echo_mono(
            side, sample_rate, strength * 0.3, defect_scores, spectral_subtraction_floor_db
        )

        # M/S → L/R rekonstruieren
        left_out = (mid_clean + side_clean).astype(np.float32)
        right_out = (mid_clean - side_clean).astype(np.float32)
        result = np.clip(np.stack([left_out, right_out], axis=0), -1.0, 1.0).astype(np.float32)
        return result  # type: ignore[no-any-return]

    # Mono-Verarbeitung via Hilfsfunktion
    return _apply_groove_echo_mono(audio, sample_rate, strength, defect_scores, spectral_subtraction_floor_db)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import (  # pylint: disable=wrong-import-position
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)


class GrooveEchoCancellationPhase(PhaseInterface):
    """Phase 61: Template-based groove echo (pre-echo) cancellation for vinyl."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_61_groove_echo_cancellation",
            name="Groove Echo Cancellation",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_01"],
            estimated_time_factor=0.06,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.55,
            description=(
                "Template-based groove echo cancellation via spectral subtraction. "
                "Removes pre-echo artifacts (~1.8 s delay at 33⅓ rpm) caused by "
                "adjacent groove deformation on vinyl records."
            ),
        )

    @staticmethod
    def _compute_groove_echo_profile(
        material_key: str, quality_mode: str | None, restorability_score: float
    ) -> dict[str, float]:
        material = str(material_key or "unknown").strip().lower()
        mode = str(quality_mode or "balanced").strip().lower()
        mode = {"restoration": "balanced", "studio_2026": "maximum"}.get(mode, mode)

        if "vinyl" in material:
            min_score = 0.10
            floor_db = -44.0
        elif "shellac" in material:
            min_score = 0.12
            floor_db = -42.0
        elif any(token in material for token in ("cd_digital", "dat", "flac", "streaming")):
            min_score = 0.18
            floor_db = -32.0
        else:
            min_score = 0.15
            floor_db = -38.0

        rest_norm = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0)) / 100.0
        min_score += (rest_norm - 0.5) * 0.10
        floor_db += (rest_norm - 0.5) * 8.0

        score_off, floor_off = {
            "fast": (0.03, 8.0),
            "balanced": (0.00, 0.0),
            "quality": (-0.02, -6.0),
            "maximum": (-0.04, -10.0),
        }.get(mode, (0.0, 0.0))
        min_score += score_off
        floor_db += floor_off

        return {
            "min_groove_echo_score": float(np.clip(min_score, 0.05, 0.25)),
            "spectral_subtraction_floor_db": float(np.clip(floor_db, -60.0, -20.0)),
        }

    @staticmethod
    def _build_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
        defect_event_metadata: dict[str, dict] | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Eventadaptive Blendmaske fuer Groove-Echo-Subtraktion."""
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0

        keys = ("groove_echo", "vinyl_pre_echo", "groove_pre_echo")
        mask = np.zeros(n_samples, dtype=np.float32)
        duration_s = float(n_samples) / float(sample_rate)
        for key in keys:
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start_s = max(0.0, float(loc[0]))
                    end_s = max(start_s, float(loc[1]))
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
                if end_s <= start_s:
                    continue
                meta_obj = (defect_event_metadata or {}).get(key, {})
                severity = (
                    float(np.clip(float(meta_obj.get("severity", 0.60)), 0.0, 1.0))
                    if isinstance(meta_obj, dict)
                    else 0.60
                )
                confidence = (
                    float(np.clip(float(meta_obj.get("confidence", 0.80)), 0.0, 1.0))
                    if isinstance(meta_obj, dict)
                    else 0.80
                )
                duration_factor = float(np.clip((end_s - start_s) / 0.90, 0.35, 1.0))
                event_strength = float(
                    np.clip((0.36 + 0.44 * severity + 0.20 * confidence) * duration_factor, 0.18, 1.0)
                )
                # Groove-Echo ist breit um den Geisterbereich, aber nicht songweit.
                s = int(max(0.0, start_s - 0.35) * sample_rate)
                e = int(min(duration_s, end_s + 0.55) * sample_rate)
                if e > s:
                    mask[s:e] = np.maximum(mask[s:e], event_strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0
        smooth = max(16, int(0.060 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if protected_zones:
            for start_s, end_s, cap in protected_zones:
                s = int(max(0.0, float(start_s)) * sample_rate)
                e = int(max(0.0, float(end_s)) * sample_rate)
                if e > s:
                    mask[s : min(n_samples, e)] = np.minimum(mask[s : min(n_samples, e)], float(cap))
        return mask, float(np.mean(mask))

    @staticmethod
    def _collect_protected_zones(kwargs: dict) -> list[tuple[float, float, float]]:
        zones: list[tuple[float, float, float]] = []
        for key, cap in (
            ("vibrato_zones", 0.20),
            ("frisson_zones", 0.30),
            ("whisper_zones", 0.25),
            ("passaggio_zones", 0.35),
        ):
            for zone in kwargs.get(key) or []:
                try:
                    start_s = float(getattr(zone, "start_s", None) or zone[0])
                    end_s = float(getattr(zone, "end_s", None) or zone[1])
                    if end_s > start_s:
                        zones.append((start_s, end_s, cap))
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
        return zones

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        _defect_scores = kwargs.get("defect_scores") or kwargs.get("defect_analysis", {})
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 0.6))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        _profile_61 = self._compute_groove_echo_profile(
            str(material_type or kwargs.get("material") or "unknown"),
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        if _effective_strength <= 0.0:
            passthrough = _soft_knee_clip(np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0))
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "groove_echo_score": float((_defect_scores or {}).get("groove_echo", 0.0)),
                    "strength": _pmgg_strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Groove echo cancellation skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_groove_echo_score=_profile_61["min_groove_echo_score"],
            spectral_subtraction_floor_db=_profile_61["spectral_subtraction_floor_db"],
        )
        elapsed = _time.perf_counter() - t0
        _locality_coverage61 = 0.0
        _loc_n61 = int(
            result_audio.shape[-1] if result_audio.ndim == 2 and result_audio.shape[0] <= 2 else result_audio.shape[0]
        )
        _locality_profile61, _locality_coverage61 = self._build_locality_profile(
            n_samples=_loc_n61,
            sample_rate=sample_rate,
            defect_locations=kwargs.get("defect_locations"),
            defect_event_metadata=kwargs.get("defect_event_metadata"),
            protected_zones=self._collect_protected_zones(kwargs),
        )
        if _locality_profile61.size > 0 and _locality_coverage61 > 0.0:
            if result_audio.ndim == 2 and audio.ndim == 2:
                if result_audio.shape[0] <= 2 and result_audio.shape[1] > 2:
                    _profile61 = _locality_profile61[np.newaxis, :]
                else:
                    _profile61 = _locality_profile61[:, np.newaxis]
                result_audio = np.clip(audio + _profile61 * (result_audio - audio), -1.0, 1.0).astype(np.float32)
            elif result_audio.ndim == 1 and audio.ndim == 1:
                result_audio = np.clip(audio + _locality_profile61 * (result_audio - audio), -1.0, 1.0).astype(
                    np.float32
                )

        # V19 Noise-Textur-Invariante (§NTI): Residual aus Spektral-Subtraktion darf kein
        # material-fremdes Spektralprofil (Whitening) aufweisen (§2.75 VERBOTEN-V19).
        try:
            from backend.core.dsp.noise_texture_guard import (
                compute_noise_texture_distance,  # pylint: disable=import-outside-toplevel
            )

            _nt61_residual = audio.astype(np.float32) - result_audio.astype(np.float32)
            _nt61_dist = compute_noise_texture_distance(_nt61_residual, str(material_type or "unknown"), sr=sample_rate)
            if _nt61_dist > 0.25:
                result_audio = (0.5 * result_audio + 0.5 * audio).astype(np.float32)
                logger.warning(
                    "Verarbeitungsschritt61 V19 Noise-Textur-Dist=%.3f > 0.25 → 50%%-Blend (Träger-Textur bewahrt)",
                    _nt61_dist,
                )
        except Exception as _nt61_exc:
            logger.debug("Verarbeitungsschritt61 V19 Noise-Textur-Guard (nicht blockierend): %s", _nt61_exc)

        # V20 Mikrodynamik-Korrelation (§2.75): voiced Frames dürfen durch Spektral-
        # Subtraktion nicht in ihrer Frame-Energie degradiert werden.
        _panns61 = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if _panns61 >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (
                    frame_energy_correlation,  # pylint: disable=import-outside-toplevel
                    recommend_mikrodynamik_wet,
                )

                _corr61 = frame_energy_correlation(audio, result_audio, sample_rate, frame_ms=10.0)
                if _corr61 < 0.97:
                    _need61 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet61 = recommend_mikrodynamik_wet(_corr61, _panns61, global_need=_need61)
                    result_audio = (_wet61 * result_audio + (1.0 - _wet61) * audio).astype(np.float32)
                    logger.warning(
                        "Verarbeitungsschritt61 V20 Mikrodynamik-Korr=%.3f < 0.97 → wet=%.3f Blend",
                        _corr61,
                        _wet61,
                    )
            except Exception as _dyn61_exc:
                logger.debug("Verarbeitungsschritt61 V20 Mikrodynamik-Guard (nicht blockierend): %s", _dyn61_exc)

        # V21 Mindestrauschboden (§2.76): Analog-Material darf nach Subtraktion keine
        # digitale Stille (-∞ dBFS) aufweisen — Rauschboden ist Naturalness-Marker.
        _mat61_str = str(material_type or "unknown").lower()
        if any(t in _mat61_str for t in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (
                    apply_noise_floor_minimum,  # pylint: disable=import-outside-toplevel
                )

                result_audio = apply_noise_floor_minimum(result_audio, sample_rate, _mat61_str)
            except Exception as _nf61_exc:
                logger.debug("Verarbeitungsschritt61 V21 Noise-Floor-Guard (nicht blockierend): %s", _nf61_exc)

        # V26 Onset-Guard (§2.77): HPSS-Onset-Fenster (0–20 ms nach Transient) dürfen
        # durch Groove-Echo-Subtraktion nicht energetisch beeinflusst werden.
        try:
            from backend.core.dsp.onset_guard import (
                apply_onset_protection_mask,  # pylint: disable=import-outside-toplevel
            )

            result_audio = apply_onset_protection_mask(audio, result_audio, None, max_delta_db=1.5)
        except Exception as _on61_exc:
            logger.debug("Verarbeitungsschritt61 V26 Onset-Guard (nicht blockierend): %s", _on61_exc)

        # §2.46f NPA-Guard: Early-Reflections (0-50 ms nach Onset) sind Recording-Chain-Signatur
        # und dürfen nicht durch Groove-Echo-Subtraktion entfernt werden (§2.46f Kategorie 3).
        try:
            from backend.core.natural_performance_detector import (
                get_natural_performance_detector,  # pylint: disable=import-outside-toplevel
            )

            _mono61 = audio.mean(axis=0) if audio.ndim == 2 else audio
            _npa_mask61 = (
                get_natural_performance_detector()
                .detect(_mono61, sample_rate)
                .get_protected_mask(len(_mono61), sample_rate)
            )
            if _npa_mask61 is not None and _npa_mask61.any():
                if result_audio.ndim == 2:
                    result_audio[:, _npa_mask61] = audio[:, _npa_mask61]
                else:
                    result_audio[_npa_mask61] = audio[_npa_mask61]
        except Exception as _npa61_exc:
            logger.debug("§2.46f Verarbeitungsschritt61 NPA-Guard (nicht blockierend): %s", _npa61_exc)

        # §2.62 Psychoakustischer Masking-Guard: Spektral-Subtraktion entfernt
        # keine Komponenten die vom Musiksignal maskiert werden (G_floor ≥ 0.10).
        try:
            from backend.core.dsp.psychoacoustics import (
                apply_psychoacoustic_masking_clamp,  # pylint: disable=import-outside-toplevel
            )

            result_audio = apply_psychoacoustic_masking_clamp(audio, result_audio, sample_rate, mode="restoration")
        except Exception as _pmask61_exc:
            logger.debug("§2.62 Verarbeitungsschritt61 Masking-Guard (nicht blockierend): %s", _pmask61_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0

        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "groove_echo_score": float((_defect_scores or {}).get("groove_echo", 0.0)),
                "strength": _pmgg_strength,
            },
            metadata={
                "groove_echo_profile": dict(_profile_61),
                "min_groove_echo_score": float(_profile_61["min_groove_echo_score"]),
                "spectral_subtraction_floor_db": float(_profile_61["spectral_subtraction_floor_db"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "repair_locality_coverage": float(_locality_coverage61),
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,  # Targeted, kein Makeup-Gain nötig
                "strength": _pmgg_strength,
            },
        )


# ── Soft-Knee Clip Helper (§III: 6 dB, 200 ms Hanning) ───────────────────────
def _soft_knee_clip(x: np.ndarray, knee_db: float = 6.0) -> np.ndarray:
    """Sigmoid-Soft-Knee statt Hard-Clamp (§III)."""
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak <= 1.0:
        return x
    overshoot_db = 20.0 * np.log10(peak / 1.0)
    gain = 1.0 / (1.0 + np.exp(-overshoot_db / knee_db))
    return x * gain  # type: ignore[no-any-return]


# ── End of file ───────────────────────────────────────────────────────────────
