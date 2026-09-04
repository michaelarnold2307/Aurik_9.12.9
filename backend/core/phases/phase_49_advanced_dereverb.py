"""
Phase 49: Advanced Dereverberation v3.0 — WPE/OMLSA Consistent
===============================================================

Vollständige DSP-Implementierung ohne ML-Abhängigkeiten.
Ersetzt den kaputten ML-Stub aus v1.0 und die np.fft.rfft-Schleife aus v2.0.

ALGORITHMUS — Weighted Prediction Error (WPE), vereinfachte Variante:
----------------------------------------------------------------------
WPE (Nakatani et al. 2010) modelliert den beobachteten Hallsignal y(t,f)
als Summe aus Direktsignal d(t,f) und spätem Nachhall r(t,f):

    y(t,f) = d(t,f) + r(t,f)

Der Nachhall-Anteil wird als gewichtete Summe vergangener Frames geschätzt:

    r(t,f) ≈ Σ_k  g_k(f) · y(t - D - k, f)

mit Systemverzögerung D (typisch 2–3 Frames ≈ 30–50 ms) und
Prädiktionsordnung K (typisch 5–10 Frames ≈ 80–160 ms Nachhall-Modell).

Die Koeffizienten g_k werden per Minimum-Varianz-Schätzung (MVDR-Prinzip)
iterativ ermittelt. Implementiert wird eine vereinfachte Single-Pass-Version:

1. scipy.signal.stft (Hann-Fenster, 75 % Überlappung) → phasenkonsistentes TF
2. Verzögertes Autoregressive Prediction per Frequenzband
3. Nachhall-Subtraktion mit Consistent-Wiener-Postfilterung (le Roux 2013)
4. Transientenmaske: Transienten umgehen die Dereverberation vollständig
5. scipy.signal.istft (OLA-konsistent, PGHI-konform) + nan_to_num + clip

Komplementär zu Phase 20 (OMLSA/IMCRA):
  - Phase 20: schnell, für moderaten Nachhall (RT60 < 0.6 s)
  - Phase 49: präziser, für starken Nachhall (RT60 0.4–2.0 s)
  → ARE aktiviert beide wenn REVERB_EXCESS severity >= 0.4

WISSENSCHAFTLICHE GRUNDLAGEN:
  - Nakatani et al. (2010): "Speech Dereverberation Based on Variance-Normalized
    Delayed Linear Prediction" — WPE Algorithmus
  - Kinoshita et al. (2016): "The REVERB Challenge" — Evaluierung
  - Habets (2007): "Multi-channel speech dereverberation based on a statistical
    model of late reverberation"
  - Le Roux & Vincent (2013): "Consistent Wiener Filtering" — Postfilter-Gain
  - Perraudin et al. (2013): PGHI — scipy.signal.stft/istft sichert OLA-Konsistenz

Author: Aurik Development Team
Version: 3.0.0 (scipy.signal.stft/istft, kein np.fft.rfft mehr)
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import scipy.signal as sig
from scipy.ndimage import median_filter

from backend.core.audio_utils import (
    restore_layout,
    safe_stft,  # §v10.115 explicit wrapper (no monkey-patch)
    to_channels_last,
)
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.restoration_policy import get_effective_song_goal_weights

from .phase_interface import (
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)

logger = logging.getLogger(__name__)


class AdvancedDereverbPhase(PhaseInterface):
    """
    WPE-basierte Dereverberation für starken Nachhall (RT60 > 0.4 s).

    Komplementär zu Phase 20 (Spectral Gating).
    Arbeitet rein mit DSP — kein ML-Import benötigt.
    """

    _PHASE_ID = "phase_49_advanced_dereverb"
    _NAME = "Advanced Dereverb (WPE DSP v3 — scipy.signal.stft)"
    description = (
        "Weighted Prediction Error Dereverberation v3: scipy.signal.stft/istft "
        "(OLA-konsistent, PGHI-konform) + Consistent-Wiener-Postfilter "
        "(Le Roux 2013). Kein ML — reine DSP-Implementierung."
    )

    # STFT-Parameter
    _WINDOW_SIZE: int = 2048  # ~46 ms bei 44.1 kHz
    _HOP_SIZE: int = 512  # 75 % Überlappung
    # WPE-Parameter
    _WPE_DELAY: int = 3  # Systemverzögerung D (Frames): ~35 ms
    _WPE_ORDER: int = 5  # Prädiktionsordnung K (war 8): ~58 ms — rt_factor ≤ 3.0
    _WPE_ITERATIONS: int = 1  # Iterationen (war 2): 1 Iteration reicht für Restaurierung
    _WIENER_FLOOR: float = 0.1  # Minimale Gain-Floor für Wiener-Postfilter
    _MAX_RMS_DROP_DB = {
        "tape": 2.5,
        "reel_tape": 2.2,
        "cassette": 2.8,
        "vinyl": 2.0,
        "shellac": 1.8,
        "wax_cylinder": 1.5,
        "cd_digital": 1.8,
        "mp3_low": 1.5,
        "mp3_high": 1.8,
        "aac": 1.8,
        "m4a": 1.8,
        "ogg": 1.8,
        "flac": 1.8,
        "streaming": 1.6,
        "dat": 1.8,
        "minidisc": 1.8,
        "unknown": 2.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self._current_material: str = "unknown"

    def _adaptive_clarity_limits(self, kwargs: dict[str, object]) -> tuple[float, float, float, float]:
        """Berechnet song-adaptive C80/D50 guard limits.

        Keeps §4.5c behavior but adapts limits with §2.56 goal-importance context.
        Returns: (c80_down_limit_db, c80_soft_limit_db, c80_hard_limit_db, d50_limit)
        """
        _gw = get_effective_song_goal_weights(kwargs)
        _w_nat = 1.0
        _w_auth = 1.0
        _w_timbre = 1.0
        _w_trans = 1.0
        _w_art = 1.0
        _w_bril = 1.0
        if isinstance(_gw, dict):
            _w_nat = float(np.clip(float(_gw.get("natuerlichkeit", 1.0)), 0.30, 2.00))
            _w_auth = float(np.clip(float(_gw.get("authentizitaet", 1.0)), 0.30, 2.00))
            _w_timbre = float(np.clip(float(_gw.get("timbre_authentizitaet", 1.0)), 0.30, 2.00))
            _w_trans = float(np.clip(float(_gw.get("transparenz", 1.0)), 0.30, 2.00))
            _w_art = float(np.clip(float(_gw.get("artikulation", 1.0)), 0.30, 2.00))
            _w_bril = float(np.clip(float(_gw.get("brillanz", 1.0)), 0.30, 2.00))

        _preserve_w = float(np.clip((_w_nat + _w_auth + _w_timbre) / 3.0, 0.30, 2.00))
        _clarity_w = float(np.clip((_w_trans + _w_art + _w_bril) / 3.0, 0.30, 2.00))

        _rest_raw = kwargs.get("restorability_score", 65.0)
        _rest_num: float = _rest_raw if isinstance(_rest_raw, (int, float)) else 65.0  # type: ignore[assignment]
        _rest = float(np.clip(_rest_num, 0.0, 100.0))
        _rest_factor = float(np.clip(1.0 + (50.0 - _rest) / 250.0, 0.85, 1.20))

        _ratio = float(np.sqrt(_clarity_w / max(_preserve_w, 1e-6)))
        c80_down_limit = float(np.clip((-2.0 * _rest_factor) / np.sqrt(max(_preserve_w, 1e-6)), -3.2, -1.2))
        c80_soft_limit = float(np.clip(4.0 * _ratio * _rest_factor, 2.8, 5.2))
        c80_hard_limit = float(np.clip(6.0 * _ratio * _rest_factor, 4.2, 7.5))
        d50_limit = float(np.clip(0.12 * _ratio * _rest_factor, 0.08, 0.18))
        return c80_down_limit, c80_soft_limit, c80_hard_limit, d50_limit

    @staticmethod
    def _adaptive_wet_mix_guard_profile(
        material_key: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive wet-mix guard profile for dereverb safety."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(
            token in _material for token in ("shellac", "wax_cylinder", "wire_recording", "lacquer_disc", "acoustic_78")
        ):
            wet_curve_exp = 1.00
            attenuation_guard_floor = 0.27
            rescue_wet_floor = 0.16
            scratch_guard_floor = 0.20
        elif any(token in _material for token in ("vinyl", "reel_tape", "tape", "cassette")):
            wet_curve_exp = 1.08
            attenuation_guard_floor = 0.31
            rescue_wet_floor = 0.19
            scratch_guard_floor = 0.24
        else:
            wet_curve_exp = 1.16
            attenuation_guard_floor = 0.37
            rescue_wet_floor = 0.24
            scratch_guard_floor = 0.29

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        wet_curve_exp += (_rest_norm - 0.5) * 0.20
        attenuation_guard_floor += (_rest_norm - 0.5) * 0.06
        rescue_wet_floor += (_rest_norm - 0.5) * 0.04
        scratch_guard_floor += (_rest_norm - 0.5) * 0.05

        _mode_offsets = {
            "fast": (0.10, 0.05, 0.04, 0.05),
            "balanced": (0.02, 0.01, 0.00, 0.01),
            "quality": (0.00, 0.00, 0.00, 0.00),
            "maximum": (-0.04, -0.02, -0.02, -0.02),
        }
        _exp_off, _att_off, _rescue_off, _scratch_off = _mode_offsets.get(_mode, (0.0, 0.0, 0.0, 0.0))
        wet_curve_exp += _exp_off
        attenuation_guard_floor += _att_off
        rescue_wet_floor += _rescue_off
        scratch_guard_floor += _scratch_off

        return {
            "wet_curve_exp": float(np.clip(wet_curve_exp, 0.95, 1.35)),
            "attenuation_guard_floor": float(np.clip(attenuation_guard_floor, 0.25, 0.45)),
            "rescue_wet_floor": float(np.clip(rescue_wet_floor, 0.15, 0.30)),
            "scratch_guard_floor": float(np.clip(scratch_guard_floor, 0.18, 0.35)),
        }

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=7,
            version="3.0.0",
            dependencies=["phase_03_denoise"],
            estimated_time_factor=0.12,
            memory_requirement_mb=160,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.91,
            description=self.description,
        )

    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="49")
        check_ml_model_ready("WPE-Dereverb", phase_name="49")
        """
        Führt WPE-Dereverberation durch.

        Args:
            audio:         Mono- oder Stereo-Audiodaten (float32/64, ±1 normiert)
            sample_rate:   Abtastrate in Hz
            material_type: Materialtyp-String
            **kwargs:      strength (float, 0–1, Default 0.7),
                           protect_transients (bool, Default True)

        Returns:
            PhaseResult mit dereverberiertem Audio.
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        # ── §v10 PIM: Per-Band-Intensität kalibrieren ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim = apply_pim_intensity(kwargs, "adv_dereverb", default_nr=0.5, default_de_ess=0.3, default_comp=1.0)
            if kwargs.get("pim_intensity_map") is not None:
                for _key in ("noise_reduction_strength", "nr_strength", "strength", "wet"):
                    if _key in kwargs:
                        kwargs[_key] = _pim["nr_strength"]
        except Exception as e:
            logger.warning("Verarbeitungsschritt_49_advanced_dereverb.py::verarbeiten Ersatzpfad: %s", e)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        # §2.51 Stereo-Kohärenz: normalize to channels-last (N,2) so all M/S
        # processing uses audio[:,0]/audio[:,1] correctly.
        audio, _p49_transposed = to_channels_last(audio)
        t0 = time.time()

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                get_plugin_lifecycle_manager as _get_plm_evict49,
            )

            _get_plm_evict49().evict_for_phase("phase_49_advanced_dereverb")
        except Exception as e:
            logger.warning("Verarbeitungsschritt_49_advanced_dereverb.py::verarbeiten Ersatzpfad: %s", e)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 0.7)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # §V40 NMR-Feedback: NR-Stärke adaptiv anpassen (FeedbackChain-aware).
        try:
            from backend.core.dsp.nmr_feedback import (
                compute_nmr_score as _nmr_fn_49,  # pylint: disable=import-outside-toplevel
            )

            _nmr_result_49 = _nmr_fn_49(audio, sample_rate)
            if not _nmr_result_49.ok:
                logger.warning(
                    "Verarbeitungsschritt49 §V40 NMR: nmr_above_masking → §2.45 Minimal-Intervention prüfen",
                )
            effective_strength = float(
                np.clip(
                    effective_strength + _nmr_result_49.recommended_nr_strength_delta,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "Verarbeitungsschritt49 §V40 NMR: delta=%.3f → eff_str=%.3f",
                _nmr_result_49.recommended_nr_strength_delta,
                effective_strength,
            )
        except Exception as _nmr_exc_49:  # pylint: disable=broad-except
            logger.debug("Verarbeitungsschritt49 §V40 NMR nicht blockierend: %s", _nmr_exc_49)

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=restore_layout(dry, _p49_transposed),
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        # §2.46f / §0 Primum non nocere — Reverb-Presence Gate:
        # WPE only delivers benefit when genuine room reverberation is present.
        # Gate is based solely on DefectScanner reverb_excess severity — not on
        # a C80-proxy from the music signal (C80 is a room-acoustics concept for
        # impulse responses; for continuous music early/late energy are nearly equal
        # so the proxy always reads ~0 dB and never fires, rendering it useless).
        # If DefectScanner finds reverb_excess < 0.15, no significant reverb is
        # detected → phase would produce artifacts without perceptual gain → skip.
        _dur_s_49 = float(len(audio)) / max(1, sample_rate)
        if _dur_s_49 >= 0.5:
            try:
                _defect_locs_49 = kwargs.get("defect_locations") or {}
                _reverb_sev_49 = 0.0
                if isinstance(_defect_locs_49, dict):
                    _reverb_sev_49 = float(
                        _defect_locs_49.get("reverb_excess", 0.0) or _defect_locs_49.get("room_reverb", 0.0)
                    )
                if _reverb_sev_49 < 0.15:
                    logger.info(
                        "Verarbeitungsschritt 49: Reverb-Presence Gate → ueberspringen "
                        "(reverb_sev=%.3f < 0.15, dur=%.1fs) "
                        "— DefectScanner finds no significant reverb, WPE would erstellen artifacts",
                        _reverb_sev_49,
                        _dur_s_49,
                    )
                    _clean = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                    _clean = np.clip(_clean, -1.0, 1.0)
                    return PhaseResult(
                        success=True,
                        audio=restore_layout(_clean, _p49_transposed),
                        execution_time_seconds=time.time() - t0,
                        metadata={
                            "algorithm": "skipped_no_reverb",
                            "reverb_severity": float(_reverb_sev_49),
                        },
                        metrics={"reverb_severity": float(_reverb_sev_49)},
                    )
                logger.debug(
                    "Verarbeitungsschritt 49: Reverb-Presence Gate → PROCEED (reverb_sev=%.3f, dur=%.1fs)",
                    _reverb_sev_49,
                    _dur_s_49,
                )
            except Exception as _rp_gate_err:
                logger.debug(
                    "Verarbeitungsschritt 49: Reverb-Presence Gate fehlgeschlagen (nicht blockierend): %s", _rp_gate_err
                )

        strength = effective_strength
        _quality_mode_49 = str(kwargs.get("quality_mode", "quality")).strip().lower()
        _quality_first_unleashed_49 = bool(
            kwargs.get("quality_first_unleashed", _quality_mode_49 in ("quality", "maximum"))
        )
        self._current_material = material_type or str(kwargs.get("material_type", "unknown"))
        _wet_mix_profile_49 = self._adaptive_wet_mix_guard_profile(
            self._current_material,
            _quality_mode_49,
            float(kwargs.get("restorability_score", 65.0)),
        )
        _vocal_conf_49 = float(kwargs.get("vocal_confidence", kwargs.get("panns_singing_confidence", 0.0)))
        _vocal_detected_49 = bool(kwargs.get("vocal_detected", False)) or (_vocal_conf_49 >= 0.35)

        # §0p Formant-Integrity pre-snapshot — §0p: F1–F4 dürfen
        # durch keine Phase um mehr als ±15% verschoben werden
        _f1_pre_49 = None
        if _vocal_conf_49 >= 0.35:
            try:
                from backend.core.dsp.lpc_formant_tracker import (  # pylint: disable=import-outside-toplevel
                    get_lpc_formant_tracker as _get_lfc_49,
                )

                _ft_in_49 = audio.mean(axis=0) if audio.ndim == 2 else audio
                _f1_pre_49 = (
                    float(_get_lfc_49().track(_ft_in_49.astype(np.float32), sample_rate).get("f1_mean", 0.0)) or None
                )
            except Exception as e:
                logger.warning("Verarbeitungsschritt_49_advanced_dereverb.py::unbekannter Ersatzpfad: %s", e)

        # §2.20 Genre-adaptive dereverb hardcap (defense-in-depth — SongCal is primary guard).
        _genre_label_49 = str(kwargs.get("genre_label", ""))
        if _genre_label_49 == "Reggae":
            strength = min(strength, 0.20)  # Dub/Echo-Reverb = genre-definierend
            logger.debug("Verarbeitungsschritt 49: Genre=Reggae → strength capped to %.2f", strength)
        elif _genre_label_49 == "Gospel":
            strength = min(strength, 0.30)  # Kirchenhall = Authentizität
            logger.debug("Verarbeitungsschritt 49: Genre=Gospel → strength capped to %.2f", strength)
        elif _genre_label_49 in ("Klassik", "Oper"):
            strength = min(strength, 0.25)  # Konzerthall = integraler Klanganteil
            logger.debug("Verarbeitungsschritt 49: Genre=%s → strength capped to %.2f", _genre_label_49, strength)
        elif _genre_label_49 == "Folk":
            strength = min(strength, 0.40)  # Natürlicher Kleiner-Raum-Klang
            logger.debug("Verarbeitungsschritt 49: Genre=Folk → strength capped to %.2f", strength)
        elif _genre_label_49 == "Jazz":
            strength = min(strength, 0.35)  # Club-Atmosphäre bewahren
            logger.debug("Verarbeitungsschritt 49: Genre=Jazz → strength capped to %.2f", strength)

        if _vocal_detected_49:
            _vocal_cap_49 = float(np.clip(0.36 - 0.10 * _vocal_conf_49, 0.26, 0.36))
            if strength > _vocal_cap_49:
                logger.debug(
                    "Verarbeitungsschritt 49: vocal guard active (conf=%.2f) → strength %.2f -> %.2f",
                    _vocal_conf_49,
                    strength,
                    _vocal_cap_49,
                )
                strength = _vocal_cap_49

        # §2.46f Room-Acoustics-Fingerprint guard — authentic room character protection.
        # Injected by UV3 from room_acoustics_fingerprinter into _restoration_context.
        _raf_49 = kwargs.get("room_acoustics_fingerprint") or {}
        _raf_cap_49 = float(_raf_49.get("dereverb_strength_cap", 1.0))
        if _raf_cap_49 < 1.0 and strength > _raf_cap_49:
            logger.debug(
                "Verarbeitungsschritt 49 §2.46f RoomAcoustics guard: rt60=%.2fs room=%s → strength %.2f → %.2f",
                float(_raf_49.get("rt60_s", 0.0)),
                _raf_49.get("room_type", "?"),
                strength,
                _raf_cap_49,
            )
            strength = _raf_cap_49

        # §Lücke5 Era-authentische Raumcharakter-Protection (Dereverb-Ceiling nach Ära)
        # Akustische Aufnahmen: kein Dereverb. Shellac: max 10 %. Vinyl-Frühphase: 20 %.
        _era_decade_49 = int(kwargs.get("era_decade", 0) or 0)
        _rt60_49 = float(_raf_49.get("rt60_s", 0.0))
        # Ceiling-Tabelle: (era_from_inclusive, era_to_exclusive) → max_strength
        _ERA_DEREVERB_CEILING: list[tuple[int, int, float]] = [
            (0, 1930, 0.0),  # Akustische Aufnahmen (Trichter): kein Dereverb
            (1930, 1950, 0.10),  # Elektrische Shellac: sehr konservativ
            (1950, 1965, 0.20),  # Vinyl-Frühphase
            (1965, 1980, 0.40),  # Analog-Bandzeitalter: nur bei starkem Nachhall
            (1980, 9999, 1.0),  # Moderne: kein Ceiling
        ]
        if _era_decade_49 > 0:
            for _era_lo, _era_hi, _era_ceil in _ERA_DEREVERB_CEILING:
                if _era_lo <= _era_decade_49 < _era_hi:
                    # Vinyl 1965–1980: nur bei RT60 > 2.5 s aktiv; sonst sehr konservativ
                    if _era_lo == 1965 and _rt60_49 < 2.5:
                        _era_ceil = 0.15
                    if strength > _era_ceil:
                        logger.debug(
                            "Verarbeitungsschritt 49 §Lücke5 Era-Ceiling: era=%d → strength %.2f → %.2f (rt60=%.2fs)",
                            _era_decade_49,
                            strength,
                            _era_ceil,
                            _rt60_49,
                        )
                        strength = _era_ceil
                    break

        protect_transients: bool = bool(kwargs.get("protect_transients", True))
        # Store material type for EMA-alpha selection in _dereverb_channel
        # Sub-phase progress callback: scoped to this phase's range (injected by UV3).
        # Emitting keeps the UI progress bar moving during slow WPE computation.
        _progress_sub_cb = kwargs.get("progress_sub_callback")

        # ── Tier-0: SGMSE+ (Richter 2022) — SOTA für starken Nachhall RT60 > 0.4 s ────
        # Dereverb-Kaskade §2.47: SGMSE+ → WPE DSP-Fallback
        processed: np.ndarray = audio.copy()  # safe fallback if all paths are skipped
        _sgmse_used = False
        _ml_model_name = "WPE-DSP"
        # Mindest-Signallänge für SGMSE+: RT60 > 0.4 s ist nur aus ≥ 1 s Audio
        # zuverlässig messbar; kürzere Signale haben unzureichenden Kontext und
        # erzeugen hohe ML-Overhead (> 10 s/call) ohne Qualitätsgewinn.
        _audio_dur_mono_s = float(audio.shape[-1] if audio.ndim > 1 else len(audio)) / max(1, sample_rate)
        _sgmse_min_dur_s = 1.0  # < 1 s → direkt WPE DSP
        _sgmse_skipped_short = _audio_dur_mono_s < _sgmse_min_dur_s
        if _sgmse_skipped_short:
            logger.debug(
                "Verarbeitungsschritt 49: SGMSE+ übersprungen (Signaldauer %.2fs < %.1fs Mindest) → WPE-DSP",
                _audio_dur_mono_s,
                _sgmse_min_dur_s,
            )
        try:
            if _sgmse_skipped_short:
                raise ImportError("short-signal-skip")  # → WPE-DSP direkt
            from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                release as _release_49,
            )
            from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                try_allocate as _alloc_49,
            )
            from plugins.sgmse_plugin import (  # pylint: disable=import-outside-toplevel
                get_sgmse_plus_plugin as _sgmse_factory_49,
            )

            if _alloc_49("SGMSE+_phase49", 0.25):
                try:
                    _plm49 = None
                    try:
                        from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                            get_plugin_lifecycle_manager as _get_plm49,
                        )

                        _plm49 = _get_plm49()
                        _plm49.set_active("SGMSE+", True)
                    except Exception:
                        _plm49 = None
                    # sigma: adaptiv aus strength — stärkerer Nachhall braucht höheres sigma
                    _sigma = float(np.clip(0.25 + strength * 0.65, 0.25, 0.90))
                    # Realistic CPU budgets: audio-duration-adaptive, capped to prevent
                    # wall-time dominance. SGMSE+ on CPU runs ~15× real-time; a 30 s clip
                    # would require ~450 s per channel without a cap → unacceptable.
                    # Cap: 2× audio_duration per channel (max 60 s normal / 90 s unleashed).
                    # Any remaining audio after budget falls back to WPE-DSP within SGMSE+.
                    _audio_dur_s_49 = float(len(audio)) / max(1, sample_rate)
                    _rt_cap_default = 90.0 if _quality_first_unleashed_49 else 60.0
                    _rt_cap_max = 120.0 if _quality_first_unleashed_49 else 90.0
                    _ml_runtime_default = float(np.clip(2.0 * _audio_dur_s_49, 15.0, _rt_cap_default))
                    _ml_runtime_max = float(np.clip(3.0 * _audio_dur_s_49, 20.0, _rt_cap_max))
                    _ml_runtime_budget_s = float(
                        np.clip(kwargs.get("ml_runtime_budget_s", _ml_runtime_default), 20.0, _ml_runtime_max)
                    )
                    if _plm49 is not None:
                        try:
                            _plm49.touch_plugin("SGMSE+")  # type: ignore[attr-defined]
                        except Exception as e:
                            logger.warning(
                                "Verarbeitungsschritt_49_advanced_dereverb.py::unbekannter Ersatzpfad: %s", e
                            )
                    _sgmse_result = _sgmse_factory_49().enhance(
                        audio,
                        sample_rate,
                        sigma=_sigma,
                        max_runtime_s=_ml_runtime_budget_s,
                    )
                    processed = np.asarray(_sgmse_result.audio, dtype=np.float32)
                    _sgmse_used = True
                    _ml_model_name = f"SGMSE+ ({_sgmse_result.model_used})"
                    logger.info(
                        "Verarbeitungsschritt 49: Tier-0 SGMSE+ OK (sigma=%.2f, model=%s, Grenze=%.1fs)",
                        _sigma,
                        _sgmse_result.model_used,
                        _ml_runtime_budget_s,
                    )
                    if _progress_sub_cb is not None:
                        _progress_sub_cb(80.0, "Nachhall-Entfernung (SGMSE+-Nachbearbeitung)", 0.0)
                except Exception as _sgmse_err:
                    logger.info(
                        "Verarbeitungsschritt 49: SGMSE+ verbessern fehlgeschlagen (%s) → WPE DSP-Ersatzpfad",
                        _sgmse_err,
                    )
                finally:
                    if _plm49 is not None:
                        try:
                            _plm49.set_active("SGMSE+", False)
                        except Exception as e:
                            logger.warning(
                                "Verarbeitungsschritt_49_advanced_dereverb.py::unbekannter Ersatzpfad: %s", e
                            )
                    _release_49("SGMSE+_phase49")
        except Exception as _imp_err:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("Verarbeitungsschritt 49: SGMSE+-Import nicht verfügbar (%s) → WPE DSP-Ersatzpfad", _imp_err)

        if not _sgmse_used:
            is_stereo = audio.ndim == 2
            if is_stereo:
                # §2.51 M/S: derive room envelope from Mid-channel only so both
                # channels are attenuated by the SAME gain curve — independent L/R
                # processing assigns different room estimates and causes stereo-field
                # asymmetry that triggers §2.49 phase-cancellation rollbacks.
                _sqrt2 = np.sqrt(2.0)
                _mid = (audio[:, 0] + audio[:, 1]) / _sqrt2
                _side = (audio[:, 0] - audio[:, 1]) / _sqrt2
                _mid_dry = self._dereverb_channel(_mid, sample_rate, strength, protect_transients)
                if _progress_sub_cb is not None:
                    _progress_sub_cb(55.0, "Nachhall-Entfernung (Side-Kanal)", 0.0)
                # Side: apply weaker dereverb (side information is already less reverberant)
                _side_strength = strength * 0.5
                _side_dry = self._dereverb_channel(_side, sample_rate, _side_strength, protect_transients)
                if _progress_sub_cb is not None:
                    _progress_sub_cb(85.0, "Nachhall-Entfernung (Nachbearbeitung)", 0.0)
                _n = min(len(_mid_dry), len(_side_dry))
                _l = (_mid_dry[:_n] + _side_dry[:_n]) / _sqrt2
                _r = (_mid_dry[:_n] - _side_dry[:_n]) / _sqrt2
                processed = np.column_stack([_l, _r])
            else:
                _mid_dry_mono = self._dereverb_channel(audio, sample_rate, strength, protect_transients)
                if _progress_sub_cb is not None:
                    _progress_sub_cb(80.0, "Nachhall-Entfernung (Nachbearbeitung)", 0.0)
                processed = _mid_dry_mono

        # Make PMGG strength retries audibly monotonic: explicit wet/dry blend at
        # the phase output, independent from internal WPE model scaling.
        wet_mix = float(np.clip(effective_strength, 0.0, 1.0))
        wet_mix = float(np.clip(wet_mix ** _wet_mix_profile_49["wet_curve_exp"], 0.0, 1.0))
        attenuation_guard_triggered = False
        attenuation_guard_factor = 1.0

        rms_before_db = self._rms_dbfs_gated(audio)
        rms_processed_db = self._rms_dbfs_gated(processed)
        rms_drop_db = rms_processed_db - rms_before_db if rms_before_db > -80.0 else 0.0

        # Safety rescue for catastrophic dereverb attenuation. If the raw processed
        # signal collapses energy too aggressively, reduce wet mix preemptively.
        _max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(self._current_material, self._MAX_RMS_DROP_DB["unknown"]))
        if rms_before_db > -80.0 and rms_drop_db < -_max_rms_drop_db and wet_mix > 0.0:
            attenuation_guard_triggered = True
            attenuation_guard_factor = float(
                np.clip(
                    _max_rms_drop_db / (abs(rms_drop_db) + 1e-9),
                    _wet_mix_profile_49["attenuation_guard_floor"],
                    1.0,
                )
            )
            wet_mix *= attenuation_guard_factor

        # --- §4.5c Early-Reflection-Guard: C80/D50 clarity-based wet-mix limiting ---
        # §4.5c Early-Reflection-Guard: C80/D50 clarity-based wet-mix limiting
        # Measure C80 (Kuttruff 2009) on original and processed signal.
        # Spec limits: ΔC80 ≤ 6 dB, ΔD50 ≤ 0.12.  Values exceeding bounds → reduce wet_mix
        # or blend early reflections back (alpha=0.35 for first 50 ms).
        _c80_down_lim, _c80_soft_lim, _c80_hard_lim, _d50_lim = self._adaptive_clarity_limits(kwargs)
        c80_pre = self._measure_c80_proxy(audio, sample_rate)
        c80_post = self._measure_c80_proxy(processed, sample_rate)
        delta_c80 = c80_post - c80_pre
        c80_guard_triggered = False
        early_blend_triggered = False

        # §4.5c D50 Guard — ΔD50 ≤ 0.12 (Deutlichkeitsmaß)
        d50_pre = self._measure_d50_proxy(audio, sample_rate)
        d50_post = self._measure_d50_proxy(processed, sample_rate)
        delta_d50 = d50_post - d50_pre

        if delta_c80 < _c80_down_lim:
            # C80 degraded — full rollback to original dry signal
            logger.warning(
                "Verarbeitungsschritt 49 C80-guard: ΔC80=%.2f dB below %.2f dB → rollback to dry",
                delta_c80,
                _c80_down_lim,
            )
            processed = audio.copy()
            wet_mix = 0.0
            c80_guard_triggered = True
        elif delta_c80 > _c80_hard_lim:
            # Excessive clarity boost → scale wet_mix proportionally
            _c80_scale = float(
                np.clip(_c80_hard_lim / (delta_c80 + 1e-9), _wet_mix_profile_49["scratch_guard_floor"], 1.0)
            )
            wet_mix *= _c80_scale
            c80_guard_triggered = True
            logger.info(
                "Verarbeitungsschritt 49 C80-guard: ΔC80=%.2f dB > %.2f dB → wet_mix scaled to %.3f",
                delta_c80,
                _c80_hard_lim,
                wet_mix,
            )
        elif delta_c80 > _c80_soft_lim:
            # Moderate clarity boost → blend 35 % of original early reflections
            # back into the first 50 ms (spec §4.5c alpha=0.35 early-reflection blend)
            early_blend_triggered = True
            _early_win = int(sample_rate * 0.050)
            _alpha = 0.35
            _proc64 = processed.copy().astype(np.float32)
            _orig64 = audio.copy().astype(np.float32)
            if _proc64.ndim == 2:
                for _ch in range(_proc64.shape[1]):
                    _e = min(_early_win, _proc64.shape[0])
                    _proc64[:_e, _ch] = (1.0 - _alpha) * _proc64[:_e, _ch] + _alpha * _orig64[:_e, _ch]
            else:
                _e = min(_early_win, len(_proc64))
                _proc64[:_e] = (1.0 - _alpha) * _proc64[:_e] + _alpha * _orig64[:_e]
            processed = _proc64.astype(np.float32)
            logger.info(
                "Verarbeitungsschritt 49 C80-guard: ΔC80=%.2f dB — early-reflection blend 35 %% angewendet (50 ms)",
                delta_c80,
            )

        # §4.5c D50 secondary guard: adaptive ΔD50 limit
        if abs(delta_d50) > _d50_lim and not c80_guard_triggered:
            _d50_scale = float(
                np.clip(_d50_lim / (abs(delta_d50) + 1e-9), _wet_mix_profile_49["scratch_guard_floor"], 1.0)
            )
            wet_mix *= _d50_scale
            logger.info(
                "Verarbeitungsschritt 49 D50-guard: ΔD50=%.3f > %.3f → wet_mix scaled to %.3f",
                delta_d50,
                _d50_lim,
                wet_mix,
            )

        if wet_mix < 1.0:
            processed = audio + wet_mix * (processed - audio)

        # Final dry/wet rescue: keep dereverb effective, but do not allow an
        # audible loudness collapse after all clarity guards have been applied.
        rms_after_blend_db = self._rms_dbfs_gated(processed)
        rms_drop_after_blend_db = rms_after_blend_db - rms_before_db if rms_before_db > -80.0 else 0.0
        if rms_before_db > -80.0 and rms_drop_after_blend_db < -_max_rms_drop_db and wet_mix > 0.0:
            _rescue_wet = float(
                np.clip(
                    wet_mix * (_max_rms_drop_db / (abs(rms_drop_after_blend_db) + 1e-9)),
                    _wet_mix_profile_49["rescue_wet_floor"],
                    wet_mix,
                )
            )
            processed = audio + _rescue_wet * (processed - audio)
            wet_mix = _rescue_wet

        elapsed = time.time() - t0
        rms_after_db = self._rms_dbfs_gated(processed)
        rms_change_db = rms_after_db - rms_before_db if rms_before_db > -80.0 else 0.0

        logger.info(
            "Verarbeitungsschritt 49 WPE-Dereverb: strength=%.2f, RMS-Δ=%.2f dB, t=%.2fs",
            strength,
            rms_change_db,
            elapsed,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # §4.5 Psychoacoustic Masking Clamp — preserve inaudible reverb tail
        try:
            from backend.core.dsp.psychoacoustics import (  # pylint: disable=import-outside-toplevel
                apply_psychoacoustic_masking_clamp,
            )

            processed = apply_psychoacoustic_masking_clamp(
                audio,
                processed,
                sample_rate,
                strength=effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt49 masking clamp nicht blockierend: %s", _pm_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — Dereverb darf Atemgeräusche
        # zwischen Phrasen und Early-Reflections des Aufnahmestudios nicht tilgen.
        try:
            from backend.core.natural_performance_detector import (  # pylint: disable=import-outside-toplevel
                get_natural_performance_detector,
            )

            _npa_a49 = processed
            if _npa_a49.ndim == 2 and _npa_a49.shape[0] == 2 and _npa_a49.shape[1] > _npa_a49.shape[0]:
                _npa_a49 = _npa_a49.T
            _npa_r49 = get_natural_performance_detector().detect(_npa_a49, sample_rate)
            _npa_n49 = (
                processed.shape[1]
                if (processed.ndim == 2 and processed.shape[0] == 2 and processed.shape[1] > 2)
                else processed.shape[0]
            )
            _npa_m49 = _npa_r49.get_protected_mask(_npa_n49, sample_rate)
            if np.any(_npa_m49):
                _npa_orig49 = audio
                if processed.ndim == 2 and _npa_orig49.ndim == 2:
                    if processed.shape[0] == 2 and processed.shape[1] > 2:
                        processed[:, _npa_m49] = _npa_orig49[:, _npa_m49]
                    elif processed.shape == _npa_orig49.shape:
                        processed[_npa_m49, :] = _npa_orig49[_npa_m49, :]
                elif processed.ndim == 1 and _npa_orig49.ndim == 1:
                    processed[_npa_m49] = _npa_orig49[_npa_m49]
        except Exception as _npa49_exc:
            logger.debug("§2.46f Verarbeitungsschritt_49 NPA-Guard (nicht blockierend): %s", _npa49_exc)

        # §2.36 Phonem-Schutz: Dereverb-Nass-Anteil kann Konsonanten-Burst-Transienten
        # glätten wenn Hüllkurven-Schätzer plosive Einsätze als Reverb-Onset klassifiziert.
        # Plosiv-Burst-Frames aus Original restaurieren (sample-level).
        try:
            from backend.core.lyrics_guided_enhancement import (  # pylint: disable=import-outside-toplevel
                get_phoneme_mask as _get_pmask_49,
            )

            _hop_49 = 512
            _mono_49 = (
                processed.mean(axis=0)
                if (processed.ndim == 2 and processed.shape[0] == 2 and processed.shape[1] > 2)
                else (processed.mean(axis=1) if processed.ndim == 2 else processed)
            )
            _pmask_49 = _get_pmask_49(_mono_49.astype(np.float32), sample_rate, hop_length=_hop_49)
            if np.any(_pmask_49):
                _n_49 = _mono_49.shape[0]
                _smask_49 = np.zeros(_n_49, dtype=bool)
                for _fi49, _fp49 in enumerate(_pmask_49):
                    if _fp49:
                        _fs49 = _fi49 * _hop_49
                        _fe49 = min(_n_49, _fs49 + _hop_49)
                        _smask_49[_fs49:_fe49] = True
                if processed.ndim == 2 and audio.ndim == 2:
                    if processed.shape[0] == 2 and processed.shape[1] > 2:
                        processed[:, _smask_49] = audio[:, _smask_49]
                    elif processed.shape == audio.shape:
                        processed[_smask_49, :] = audio[_smask_49, :]
                elif processed.ndim == 1 and audio.ndim == 1:
                    processed[_smask_49] = audio[_smask_49]
        except Exception as _pm49_exc:
            logger.debug("§2.36 Verarbeitungsschritt_49 Phonem-Mask (nicht blockierend): %s", _pm49_exc)

        # §0p HNR-Blend nach ML-Dereverb (RELEASE_MUST §0p): ΔHNR > 3 dB → Dry-Wet-Blend
        _p49_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", _vocal_conf_49)))
        if _p49_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import (  # pylint: disable=import-outside-toplevel
                    apply_hnr_blend as _apply_hnr_p49,
                )

                _hnr_blended_p49, _hnr_diag_p49 = _apply_hnr_p49(
                    audio.astype(np.float32), processed.astype(np.float32), sample_rate
                )
                if _hnr_diag_p49.get("over_cleaned"):
                    processed = _hnr_blended_p49
            except Exception as _hnr_exc_p49:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_49 (nicht blockierend): %s", _hnr_exc_p49)

        # §0p Formant-Integrity post-check — rollback if F1 shifted >±15%
        if _f1_pre_49 is not None:
            try:
                from backend.core.dsp.lpc_formant_tracker import (  # pylint: disable=import-outside-toplevel
                    get_lpc_formant_tracker as _get_lfc_49_post,
                )

                _ft_out_49 = processed.mean(axis=0) if processed.ndim == 2 else processed
                _f1_post_49 = float(
                    _get_lfc_49_post().track(_ft_out_49.astype(np.float32), sample_rate).get("f1_mean", 0.0)
                )
                if _f1_post_49 > 0 and abs(_f1_post_49 - _f1_pre_49) > _f1_pre_49 * 0.15:
                    logger.warning(
                        "§0p Formant drift phase_49 (F1 %.0f→%.0f Hz, delta=%.0f Hz) — rollback",
                        _f1_pre_49,
                        _f1_post_49,
                        abs(_f1_post_49 - _f1_pre_49),
                    )
                    processed = audio.copy()
            except Exception as e:
                logger.warning("Verarbeitungsschritt_49_advanced_dereverb.py::unbekannter Ersatzpfad: %s", e)

        # §Gap3 PhraseBoundaryGuard — taper artifacts at phrase transitions (§0p Vocal-Supremacy)
        try:
            from backend.core.dsp.phrase_boundary_guard import (  # pylint: disable=import-outside-toplevel
                apply_phrase_boundary_taper as _apply_pbg_49,
            )
            from backend.core.dsp.phrase_boundary_guard import (  # pylint: disable=import-outside-toplevel
                detect_phrase_boundaries as _detect_pbg_49,
            )

            _pbg_bounds_49 = _detect_pbg_49(audio, sample_rate)
            if _pbg_bounds_49:
                _pbg_env_49 = _apply_pbg_49(audio, _pbg_bounds_49, sample_rate, taper_ms=20.0).astype(np.float32)
                if processed.ndim == 1:
                    processed = audio + (processed - audio) * _pbg_env_49
                elif processed.ndim == 2 and processed.shape[0] == 2 and processed.shape[1] > 2:
                    processed = audio + (processed - audio) * _pbg_env_49[np.newaxis, :]
                else:
                    processed = audio + (processed - audio) * _pbg_env_49[:, np.newaxis]
                processed = np.clip(np.nan_to_num(processed, nan=0.0), -1.0, 1.0).astype(np.float32)
                logger.debug("§Gap3 PhraseBoundaryGuard Verarbeitungsschritt_49: %d boundaries", len(_pbg_bounds_49))
        except Exception as _pbg_exc_49:
            logger.debug("PhraseBoundaryGuard Verarbeitungsschritt_49 (nicht blockierend): %s", _pbg_exc_49)

        # §2.46f [RELEASE_MUST] Atemgeräusch-Schutz — dereverb-Algorithmen greifen diffuse
        # Spektralanteile an (Early-Reflections, Raum-Rauschen), die Atemgeräusche
        # charakterisieren. Originalklang in Atemzonen zurückblenden (§0p Vocal-Supremacy).
        _breath_segs_p49 = list(kwargs.get("breath_segments", []) or [])
        if _breath_segs_p49 and _p49_panns >= 0.25:
            try:
                _n_samp_p49 = int(processed.shape[-1] if processed.ndim == 2 else len(processed))
                _n_in_p49 = int(audio.shape[-1] if audio.ndim == 2 else len(audio))
                _n_blend_p49 = min(_n_samp_p49, _n_in_p49)
                _result_blend_p49 = np.array(processed, copy=True)
                _blended_p49 = False
                for _bs_p49 in _breath_segs_p49:
                    _bs_start_p49 = float(getattr(_bs_p49, "start_s", 0.0))
                    _bs_end_p49 = float(getattr(_bs_p49, "end_s", 0.0))
                    if _bs_end_p49 <= _bs_start_p49:
                        continue
                    # Konservativeres Dry-Ratio für Dereverb: 0.70 (mehr Originalerhalt als NR)
                    # Dereverb entfernt Raumhall, aber Atemgeräusche HABEN keinen störenden Hall.
                    _g_fl_p49 = float(np.clip(getattr(_bs_p49, "recommended_g_floor", 0.70), 0.50, 0.95))
                    _si_p49 = max(0, min(int(round(_bs_start_p49 * sample_rate)), _n_blend_p49))
                    _ei_p49 = max(0, min(int(round(_bs_end_p49 * sample_rate)), _n_blend_p49))
                    if _si_p49 >= _ei_p49:
                        continue
                    if _result_blend_p49.ndim == 2 and audio.ndim == 2:
                        _result_blend_p49[:, _si_p49:_ei_p49] = (
                            _g_fl_p49 * audio[:, _si_p49:_ei_p49] + (1.0 - _g_fl_p49) * processed[:, _si_p49:_ei_p49]
                        )
                    elif _result_blend_p49.ndim == 1 and audio.ndim == 1:
                        _result_blend_p49[_si_p49:_ei_p49] = (
                            _g_fl_p49 * audio[_si_p49:_ei_p49] + (1.0 - _g_fl_p49) * processed[_si_p49:_ei_p49]
                        )
                    _blended_p49 = True
                if _blended_p49:
                    processed = np.clip(np.nan_to_num(_result_blend_p49.astype(np.float32), nan=0.0), -1.0, 1.0)
                    logger.debug(
                        "§2.46f BreathGuard Verarbeitungsschritt_49: %d Atemzonen geschützt (dry_Verhaeltnis=%.2f)",
                        len(
                            [
                                s
                                for s in _breath_segs_p49
                                if float(getattr(s, "end_s", 0.0)) > float(getattr(s, "start_s", 0.0))
                            ]
                        ),
                        _g_fl_p49 if _breath_segs_p49 else 0.0,
                    )
            except Exception as _breath_exc_p49:
                logger.debug("§2.46f BreathGuard Verarbeitungsschritt_49 (nicht blockierend): %s", _breath_exc_p49)

        # §0p [RELEASE_MUST] VQI per-Phase Gate — vokal-beeinflussende Phasen (phases.instructions.md):
        # Dereverb verändert diffuse Spektralanteile und kann Stimmqualität degradieren.
        # Threshold: material-adaptiv (§0p: Shellac 0.62, Vinyl 0.72, CD 0.82).
        # FALSCH wäre 0.95 — Dereverb ist Carrier-Repair, kein Enhancement; 0.95 führt
        # auf historischem Material fast immer zum Rollback (VQI 0.72–0.94 ist akzeptabel).
        if _p49_panns >= 0.35:
            try:
                from backend.core.musical_goals.era_vocal_profile import (  # pylint: disable=import-outside-toplevel
                    get_era_vocal_profile,
                )
                from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
                    compute_vqi,
                )

                _era_profile_49 = get_era_vocal_profile(_era_decade_49) if _era_decade_49 > 0 else None
                _vqi_result_49 = compute_vqi(
                    audio_orig=audio,
                    audio_restored=processed,
                    sr=sample_rate,
                    era_profile=_era_profile_49,  # §EraVocalProfile: historisches Material
                    # braucht angepasste Toleranzen
                )
                _vqi_p49 = _vqi_result_49["vqi"]
                # Material-adaptiver Rollback-Schwellwert (§0p: Shellac 0.62, Vinyl 0.72, CD 0.82)
                _mat_49 = str(self._current_material).lower()
                if "shellac" in _mat_49:
                    _vqi_floor_49 = 0.62
                elif any(x in _mat_49 for x in ("vinyl", "tape", "cassette")):
                    _vqi_floor_49 = 0.72
                else:
                    _vqi_floor_49 = 0.72  # konservativ für unbekanntes Material
                if _vqi_p49 < _vqi_floor_49:
                    logger.info(
                        "Verarbeitungsschritt_49: VQI per-Verarbeitungsschritt rollback (vqi=%.3f < %.2f [%s], panns=%.2f) — Dereverb zurückgesetzt",
                        _vqi_p49,
                        _vqi_floor_49,
                        _mat_49,
                        _p49_panns,
                    )
                    processed = audio  # Rollback auf Phase-Input
            except Exception as _vqi_exc_p49:
                logger.debug("Verarbeitungsschritt_49 VQI-Gate (nicht blockierend): %s", _vqi_exc_p49)

        # §V19 (Spec-Vintage-Guard)/V20/V21/V26/§2.72 Vokal- + Textur-Guards nach Advanced Dereverb (RELEASE_MUST §0p V19-V26)
        _mat49_guards = str(self._current_material or "unknown").lower()
        _nt49_residual = audio - processed
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt49_dist_fn,
            )

            if _nt49_residual.shape == audio.shape:
                _nt49_d = _nt49_dist_fn(_nt49_residual, _mat49_guards, sr=sample_rate)
                if _nt49_d > 0.25:
                    processed = (0.5 * processed + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_49: noise_texture_dist=%.3f > 0.25 → 50%% dry-blend",
                        _nt49_d,
                    )
        except Exception as _nt49_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_49 noise_texture nicht blockierend: %s", _nt49_exc
            )

        if _p49_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (  # pylint: disable=import-outside-toplevel
                    frame_energy_correlation as _fec49,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr49 = _fec49(audio, processed, sample_rate, frame_ms=10.0)
                if _corr49 < 0.97:
                    _need49 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet49 = _recommend_mkk_wet(_corr49, _p49_panns, global_need=_need49)
                    processed = (_wet49 * processed + (1.0 - _wet49) * audio).astype(np.float32)
                    logger.warning(
                        "§V20 Verarbeitungsschritt_49: mikrodynamik_corr=%.4f < 0.97 → wet=%.3f", _corr49, _wet49
                    )
            except Exception as _v20_49_exc:
                logger.debug("§V20 Verarbeitungsschritt_49 mikrodynamik nicht blockierend: %s", _v20_49_exc)

        if any(x in _mat49_guards for x in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (  # pylint: disable=import-outside-toplevel
                    apply_noise_floor_minimum as _nfmin49,
                )

                processed = _nfmin49(processed, sample_rate, _mat49_guards, original_audio=audio)
            except Exception as _v21_49_exc:
                logger.debug("§V21 Verarbeitungsschritt_49 noise_floor nicht blockierend: %s", _v21_49_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_49,
            )

            _sc_result_49 = _scg_49(audio, processed, sample_rate)
            if not _sc_result_49.ok:
                _sc_wet_49 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                processed = (_sc_wet_49 * processed + (1.0 - _sc_wet_49) * audio).astype(np.float32)
        except Exception as _sc_exc_49:  # pylint: disable=broad-except
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_49 spectral_color nicht blockierend: %s", _sc_exc_49
            )

        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opm49,
            )

            processed = _opm49(audio, processed, None, max_delta_db=1.5)
        except Exception as _v26_49_exc:
            logger.debug("§V26 Verarbeitungsschritt_49 onset_guard nicht blockierend: %s", _v26_49_exc)

        if _p49_panns >= 0.25:
            try:
                from backend.core.dsp.vibrato_guard import (  # pylint: disable=import-outside-toplevel
                    check_vibrato_depth_preservation as _vib49_fn,
                )

                _vibr49 = _vib49_fn(audio, processed, sample_rate)
                if not _vibr49.ok:
                    processed = (0.5 * processed + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§2.72 Verarbeitungsschritt_49: vibrato_reduction=%.1f%% → 50%% dry-blend",
                        _vibr49.depth_reduction_pct,
                    )
            except Exception as _vib49_exc:
                logger.debug("§2.72 Verarbeitungsschritt_49 vibrato nicht blockierend: %s", _vib49_exc)

        return PhaseResult(
            success=True,
            audio=restore_layout(processed, _p49_transposed),
            execution_time_seconds=elapsed,
            resolved_defects={
                "REVERB_EXCESS": float(np.clip(max(0.0, 1.0 - effective_strength), 0.0, 0.3)),
            },
            metadata={
                "algorithm": _ml_model_name,
                "ml_used": _sgmse_used,
                "ml_model": _ml_model_name,
                "strength": strength,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "wpe_delay": "adaptive_schroeder",
                "wpe_order": "adaptive_schroeder",
                "wpe_iterations": self._WPE_ITERATIONS,
                "window_size": self._WINDOW_SIZE,
                "hop_size": self._HOP_SIZE,
                "wet_mix": wet_mix,
                "wet_mix_guard_profile": dict(_wet_mix_profile_49),
                "wet_curve_exp": float(_wet_mix_profile_49["wet_curve_exp"]),
                "attenuation_guard_floor": float(_wet_mix_profile_49["attenuation_guard_floor"]),
                "rescue_wet_floor": float(_wet_mix_profile_49["rescue_wet_floor"]),
                "scratch_guard_floor": float(_wet_mix_profile_49["scratch_guard_floor"]),
                "attenuation_guard_triggered": attenuation_guard_triggered,
                "attenuation_guard_factor": attenuation_guard_factor,
                "rms_change_db": rms_change_db,
                "rms_drop_db": round(float(min(0.0, rms_change_db)), 3),  # §2.45a telemetry
                "loudness_makeup_db": 0.0,  # phase_49 uses wet-rescue, not makeup gain
                "protect_transients": protect_transients,
                "c80_pre": float(c80_pre),
                "c80_post": float(c80_post),
                "delta_c80": float(delta_c80),
                "c80_down_limit_db": float(_c80_down_lim),
                "c80_soft_limit_db": float(_c80_soft_lim),
                "c80_hard_limit_db": float(_c80_hard_lim),
                "d50_limit": float(_d50_lim),
                "c80_guard_triggered": c80_guard_triggered,
                "early_blend_triggered": early_blend_triggered,
                "vocal_guard_active": bool(_vocal_detected_49),
                "vocal_confidence": float(_vocal_conf_49),
            },
            metrics={"rms_change_db": rms_change_db, "strength": strength, "effective_strength": effective_strength},
        )

    # ------------------------------------------------------------------
    # Kern-Implementierung
    # ------------------------------------------------------------------

    def _rms_dbfs_gated(self, audio: np.ndarray, gate_dbfs: float = -50.0) -> float:
        """Frame-wise RMS over active music frames only (§2.45a-I)."""
        _x = np.asarray(audio, dtype=np.float32)
        if _x.ndim == 2:
            _mono = _x.mean(axis=0) if (_x.shape[0] <= 2 and _x.shape[1] > 2) else _x.mean(axis=1)
        else:
            _mono = _x
        if _mono.size < 480:
            _rms = float(np.sqrt(np.mean(_mono.astype(np.float64) ** 2) + 1e-12))
            return float(20.0 * np.log10(_rms + 1e-10))
        _frame = 480
        _vals = []
        for _i in range(0, len(_mono) - _frame + 1, _frame):
            _seg = _mono[_i : _i + _frame]
            _r = float(np.sqrt(np.mean(_seg.astype(np.float64) ** 2) + 1e-12))
            _db = float(20.0 * np.log10(_r + 1e-10))
            if _db > gate_dbfs:
                _vals.append(_r * _r)
        if not _vals:
            return -96.0
        _rms = float(np.sqrt(np.mean(_vals) + 1e-12))
        return float(20.0 * np.log10(_rms + 1e-10))

    def _measure_c80_proxy(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt Clarity C80 from time-domain energy ratio (Kuttruff 2009).

        C80 = 10 × log10(E_early / E_late)
        where E_early = energy in first 80 ms, E_late = energy from 80 ms onward.

        Used by spec §4.5c Early-Reflection-Guard to limit dereverberation depth.
        Returns 0.0 when audio is silent or shorter than 80 ms.
        """
        mono = audio[:, 0] if audio.ndim == 2 else audio
        early_win = int(sr * 0.080)  # 80 ms
        if len(mono) <= early_win:
            return 0.0
        e_early = float(np.sum(mono[:early_win] ** 2))
        e_late = float(np.sum(mono[early_win:] ** 2))
        if e_late < 1e-12:
            return 0.0
        return 10.0 * float(np.log10(max(e_early, 1e-12) / e_late))

    def _measure_d50_proxy(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt Definition D50 from time-domain energy ratio.

        D50 = E_early50 / E_total  where E_early50 = energy in first 50 ms.
        Returns 0.0 when audio is silent.
        """
        mono = audio[:, 0] if audio.ndim == 2 else audio
        early_win = int(sr * 0.050)  # 50 ms
        e_total = float(np.sum(mono**2))
        if e_total < 1e-12:
            return 0.0
        e_early50 = float(np.sum(mono[:early_win] ** 2))
        return float(np.clip(e_early50 / e_total, 0.0, 1.0))

    def _dereverb_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float,
        protect_transients: bool,
    ) -> np.ndarray:
        """WPE-Dereverberation für einen einzelnen Kanal."""
        n_orig = len(audio)

        # 0. Schroeder T60-Schätzung — Vorab-Prüfung für Early-Exit
        t60 = self._estimate_t60_schroeder(audio, sample_rate)
        if t60 < 0.15:
            logger.info(
                "Verarbeitungsschritt 49 WPE: T60=%.3fs < 0.15s — kein signifikanter Nachhall, WPE übersprungen (Kanal %d Samples)",
                t60,
                n_orig,
            )
            return np.clip(audio.copy(), -1.0, 1.0)  # type: ignore[no-any-return]

        # §C2 Blind RIR: refine WPE delay D using predelay estimate
        _rir_params = self._estimate_blind_rir(audio, sample_rate)
        _predelay_samples = max(0, int(_rir_params["predelay_ms"] / 1000.0 * sample_rate))
        _drr_db = float(_rir_params["drr_db"])
        logger.debug(
            "§C2 Blind RIR: rt60=%.2fs predelay=%.1fms EDT=%.2fs DRR=%.1fdB",
            _rir_params["rt60_s"],
            _rir_params["predelay_ms"],
            _rir_params["edt_s"],
            _drr_db,
        )

        # 1. Transientenmaske
        transient_mask: np.ndarray | None = None
        if protect_transients:
            transient_mask = self._compute_transient_mask(audio, sample_rate)

        # 2. STFT
        win = np.hanning(self._WINDOW_SIZE)
        stft_matrix = self._stft(audio, win)  # (T, F)

        # 3. WPE: iterative Nachhall-Schätzung & Subtraktion
        enhanced = stft_matrix.copy()
        t60_frames = max(1, int(t60 * sample_rate / self._HOP_SIZE))
        # §C2: Add predelay frames to WPE delay D for more accurate onset alignment
        _predelay_frames = max(0, _predelay_samples // self._HOP_SIZE)
        D = max(2, min(6, int(t60_frames * 0.25) + _predelay_frames))  # ~25% T60 + predelay
        K = max(3, min(12, int(t60_frames * 0.60)))  # ~60 % von T60
        # §C2: DRR-adaptive iteration count — anechoic signal (DRR > 15 dB) → 1 iteration enough
        _iterations = 1 if _drr_db > 15.0 else self._WPE_ITERATIONS
        logger.debug(
            "Verarbeitungsschritt 49 Schroeder T60=%.2fs + §C2 predelay=%dframes → WPE D=%d K=%d iter=%d (t60_frames=%d)",
            t60,
            _predelay_frames,
            D,
            K,
            _iterations,
            t60_frames,
        )
        _, F = stft_matrix.shape

        # §2.61 Wall-Time-Guard: WPE DSP must not run unbounded on long audio.
        # Budget per channel = audio_duration × 2.0 (up to 60s cap, min 10s).
        # Stereo calls _dereverb_channel twice (Mid + Side), so total = 2 × budget.
        # For 30s audio: 2 × 60s = 120s total, well within pipeline wall-time budget.
        # After budget is exhausted, we stop at the current bin and use whatever
        # `enhanced` has been computed so far — partial WPE is still better than nothing.
        _audio_dur_s = float(len(audio)) / max(1, sample_rate)
        _wpe_max_runtime_s = float(np.clip(_audio_dur_s * 2.0, 10.0, 60.0))
        _wpe_channel_start_t = time.monotonic()
        _wpe_budget_exhausted = False

        for _iteration in range(_iterations):
            power = np.abs(enhanced) ** 2
            # alpha=0.90 is fast (high noise-floor reactivity) — good for vinyl/cassette
            # reverb bursts.  Tape material has a longer reverb tail and gentler room
            # impulse; alpha=0.93 averages over more frames, preventing over-dereverb
            # that strips authentic Tape room character (§0 Authenticity).
            _ema_alpha = 0.93 if getattr(self, "_current_material", "") in ("tape", "reel_tape") else 0.90
            smoothed_power = self._smooth_power(power, alpha=_ema_alpha)

            reverb_estimate = np.zeros_like(stft_matrix)
            for f in range(F):
                # §2.61 Wall-Time-Guard: abort bin loop if WPE runtime exceeds budget.
                if f % 64 == 0 and f > 0:
                    _wpe_elapsed = time.monotonic() - _wpe_channel_start_t
                    if _wpe_elapsed > _wpe_max_runtime_s:
                        logger.warning(
                            "Verarbeitungsschritt 49 WPE: Grenze %.1fs exhausted at bin %d/%d "
                            "(elapsed=%.1fs, audio=%.1fs) — partial WPE Ergebnis used",
                            _wpe_max_runtime_s,
                            f,
                            F,
                            _wpe_elapsed,
                            _audio_dur_s,
                        )
                        _wpe_budget_exhausted = True
                        break
                if smoothed_power[:, f].max() < 1e-12:
                    continue
                reverb_estimate[:, f] = self._predict_reverb_band(
                    stft_matrix[:, f], smoothed_power[:, f], D, K, strength
                )
            enhanced = stft_matrix - reverb_estimate
            if _wpe_budget_exhausted:
                break

        # 4. Wiener-Postfilter
        enhanced = self._apply_wiener_postfilter(enhanced, stft_matrix, floor=self._WIENER_FLOOR)

        # 5. ISTFT (OLA)
        output = self._istft(enhanced, win, n_orig)

        # 6. Transientenrestauration
        if protect_transients and transient_mask is not None:
            mask_res = sig.resample(transient_mask.astype(float), n_orig)
            mask_res = np.clip(mask_res, 0.0, 1.0)
            output = output * (1.0 - mask_res) + audio[:n_orig] * mask_res

        # Pegel-Erhalt (§2.45a — gated RMS: silence sections after reverb removal must not
        # inflate the correction factor; ungated mean would amplify silence dramatically)
        _rms_in_db = self._rms_dbfs_gated(audio)
        _rms_out_db = self._rms_dbfs_gated(output)
        if _rms_out_db > -80.0 and _rms_in_db > -80.0:
            _level_gain = float(10.0 ** ((_rms_in_db - _rms_out_db) / 20.0))
            _level_gain = min(_level_gain, 4.0)  # hard cap +12 dB — prevents runaway amplification
            if abs(_level_gain - 1.0) > 0.001:
                output = output * _level_gain

        return np.clip(output, -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # WPE-Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_blind_rir(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        """§C2 Blind Room Impulse Response estimation via autocorrelation.

        Estimates key RIR parameters from the reverberant speech/music signal
        without requiring a clean reference (no anechoic chamber needed).

        Method:
        1. T60 via Schroeder (1965) backward integration (existing method).
        2. Predelay: first significant autocorrelation peak lag after t=0
           (delay before room reflections arrive, typically 0-15 ms in studios).
        3. Early decay time (EDT, 0 to -10 dB) via EDC slope (Beranek 2016).
        4. Direct-to-reverberant ratio (DRR) proxy: ratio of first 5 ms RMS
           to full signal RMS (Vesa & Harma 2005 blind DRR estimate).

        Returns dict with 'rt60_s', 'predelay_ms', 'edt_s', 'drr_db'.

        References:
            Schroeder (1965) JASA 37:409.
            Vesa & Harma (2005) Proc. ICASSP — blind DRR.
            Beranek (2016) JASA 139:1548 — EDT and RT relationship.
        """
        result = {"rt60_s": 0.4, "predelay_ms": 0.0, "edt_s": 0.25, "drr_db": 6.0}
        try:
            x = np.asarray(audio, dtype=np.float64)
            x = x - float(np.mean(x))  # DC removal

            # 1. T60 via EDC
            energy = x**2
            edc = np.cumsum(energy[::-1])[::-1]
            peak = float(edc.max())
            if peak < 1e-14:
                return result
            edc_db = 10.0 * np.log10(edc / peak + 1e-14)

            below5 = np.where(edc_db <= -5.0)[0]
            below35 = np.where(edc_db <= -35.0)[0]
            if len(below5) > 0 and len(below35) > 0 and below35[0] > below5[0]:
                t30 = (below35[0] - below5[0]) / float(sample_rate)
                result["rt60_s"] = float(np.clip(2.0 * t30, 0.05, 3.0))

            # 2. Predelay via normalized autocorrelation — FFT-based O(N log N)
            max_lag = int(0.020 * sample_rate)  # Search up to 20 ms
            if len(x) > 2 * max_lag:
                from backend.core.core_utils import fft_autocorr  # pylint: disable=import-outside-toplevel

                ac = fft_autocorr(x[: max_lag * 4])
                ac_norm = ac / (float(ac[0]) + 1e-14)
                # Find first local peak after lag=0
                for lag in range(2, min(max_lag, len(ac_norm) - 1)):
                    if ac_norm[lag] > ac_norm[lag - 1] and ac_norm[lag] > ac_norm[lag + 1] and ac_norm[lag] > 0.05:
                        result["predelay_ms"] = float(lag / sample_rate * 1000.0)
                        break

            # 3. EDT: 0 to -10 dB slope of EDC (Beranek 2016 definition)
            below0 = np.where(edc_db <= 0.0)[0]
            below10 = np.where(edc_db <= -10.0)[0]
            if len(below0) > 0 and len(below10) > 0 and below10[0] > below0[0]:
                edt_t = (below10[0] - below0[0]) / float(sample_rate)
                result["edt_s"] = float(np.clip(edt_t * 6.0, 0.02, 2.0))  # Extrapolate to -60 dB

            # 4. DRR proxy: first 5 ms vs full (Vesa & Harma 2005)
            direct_window = int(0.005 * sample_rate)
            if direct_window > 0:
                rms_direct = float(np.sqrt(np.mean(x[:direct_window] ** 2)) + 1e-14)
                rms_total = float(np.sqrt(np.mean(x**2)) + 1e-14)
                ddRR = 20.0 * np.log10(rms_direct / rms_total)
                result["drr_db"] = float(np.clip(ddRR, -20.0, 30.0))

        except Exception as _rir_exc:
            logger.debug("§C2 Blind RIR estimation nicht blockierend: %s", _rir_exc)
        return result

    @staticmethod
    def _estimate_t60_schroeder(audio: np.ndarray, sample_rate: int) -> float:
        """Schroeder (1965) Backward Integration — blinde T60-Schätzung.

        Berechnet die Nachhallzeit T60 aus der Energy Decay Curve (EDC):

            EDC(t) = ∫[t..∞] x²(τ) dτ  ≈  Σ[n=t..N] x²[n]  (Rückwärts-Kumulativsumme)

        Dann gilt: T60 = 2 × T30  (Abfall von -5 dB auf -35 dB in der EDC).

        Referenz:
            Schroeder (1965) "New Method of Measuring Reverberation Time"
            — JASA 37(3): 409–412.

        Args:
            audio:       Mono-Audiosignal (float32/64).
            sample_rate: Abtastrate (Hz).

        Returns:
            T60-Schätzung in Sekunden, geklämmt auf [0.1, 3.0].
            Fallback 0.4 s bei Stille oder nicht-konvergenter Kurve.
        """
        x = audio.astype(np.float64)
        energy = x**2
        # Energy Decay Curve: rückwärtige Kumulativsumme
        edc = np.cumsum(energy[::-1])[::-1]
        peak = edc.max()
        if peak < 1e-12:
            return 0.4  # Stille → konservativer Fallback
        edc_db = 10.0 * np.log10(edc / peak + 1e-12)
        # -5 dB → -35 dB Schnittpunkte → T30 × 2 = T60
        below5 = np.where(edc_db <= -5.0)[0]
        below35 = np.where(edc_db <= -35.0)[0]
        if len(below5) == 0 or len(below35) == 0:
            return 0.4
        idx5 = int(below5[0])
        idx35 = int(below35[0])
        if idx35 <= idx5:
            return 0.4
        t30 = (idx35 - idx5) / float(sample_rate)
        return float(np.clip(2.0 * t30, 0.1, 3.0))

    @staticmethod
    def _smooth_power(power: np.ndarray, alpha: float = 0.90) -> np.ndarray:
        """Exponential Moving Average über Zeitachse — vectorized via lfilter."""
        # EMA: y[t] = alpha * y[t-1] + (1 - alpha) * x[t]
        # As IIR filter: b = [1 - alpha], a = [1, -alpha]
        b = np.array([1.0 - alpha])
        a = np.array([1.0, -alpha])
        # Apply per frequency bin (power shape: (T, F) or (T,))
        if power.ndim == 1:
            smoothed = sig.lfilter(b, a, power)
        else:
            smoothed = sig.lfilter(b, a, power, axis=0)
        _smoothed_arr: np.ndarray = np.asarray(smoothed + 1e-12, dtype=np.float64)
        return _smoothed_arr

    @staticmethod
    def _predict_reverb_band(
        y: np.ndarray,
        power: np.ndarray,
        D: int,
        K: int,
        strength: float,
    ) -> np.ndarray:
        """
        Schätzt den Nachhall-Anteil r(t) für ein einzelnes Frequenzband f
        via gewichteter Least-Squares-Regression (vereinfachtes WPE).

        y(t) ≈ Σ_k g_k · y(t - D - k)   →  r(t) = Σ_k g_k · y(t-D-k)

        Args:
            y:        Komplexes STFT-Spektrum, shape (T,)
            power:    Geglättete Leistungsschätzung, shape (T,)
            D:        Systemverzögerung (Frames)
            K:        Prädiktionsordnung
            strength: Skalierung der Subtraktion [0–1]

        Returns:
            reverb_estimate, shape (T,), komplex
        """
        T = len(y)
        n_eq = T - D - K
        if n_eq < K:
            return np.zeros_like(y)  # type: ignore[no-any-return]

        # Regressionsmatrix X (n_eq × K)
        X = np.zeros((n_eq, K), dtype=complex)
        b = y[D + K :]
        for k in range(K):
            X[:, k] = y[K - k - 1 : T - D - k - 1]

        # Gewichtung: low-power Frames bevorzugen (Direktschall-Selektion)
        w = 1.0 / (power[D + K :] + 1e-8)
        w = w / (w.max() + 1e-12)

        Xw = X * w[:, np.newaxis]
        try:
            XhXw = Xw.conj().T @ X
            XhBw = Xw.conj().T @ b
            reg = 1e-4 * np.eye(K)
            g = np.linalg.solve(XhXw + reg, XhBw)
        except np.linalg.LinAlgError as exc:
            logger.debug("§V6 LS-Reverb-Prediction fehlgeschlagen — Nullen zurückgegeben (LinAlgError): %s", exc)
            return np.zeros_like(y)  # type: ignore[no-any-return]

        # Vectorized reverb prediction: r(t) = Σ_k g_k · y(t-D-k-1)
        # np.convolve flips the kernel internally, so we pass g directly.
        conv_result = np.convolve(y, g, mode="full")  # length T + K - 1
        # conv_result[s] = Σ_k g[k] * y[s-k], valid when s >= K-1
        # reverb[t] = conv_result[t - D - 1] for t >= D + K (i.e. t-D-1 >= K-1)
        reverb = np.zeros_like(y)
        valid_start = D + K
        valid_end = min(T, T)  # conv_result has enough entries
        src_start = valid_start - D - 1  # = K - 1
        src_end = valid_end - D - 1
        if src_end > src_start and src_end <= len(conv_result):
            reverb[valid_start:valid_end] = conv_result[src_start:src_end]

        return reverb * strength  # type: ignore[no-any-return]

    @staticmethod
    def _apply_wiener_postfilter(
        enhanced: np.ndarray,
        original: np.ndarray,
        floor: float = 0.10,
    ) -> np.ndarray:
        """
        Wiener-Postfilter gegen Musical Noise nach WPE-Subtraktion.

        Gain(t,f) = |enhanced|² / (|original|² + ε), geklämmt auf [floor, 1].
        Zusätzlich 3-Frame-Median-Glättung in der Zeitachse.
        """
        eps = 1e-10
        gain = np.abs(enhanced) ** 2 / (np.abs(original) ** 2 + eps)
        gain = np.clip(gain, floor, 1.0)
        gain = median_filter(gain, size=(3, 1))
        # STFT bleibt bis zur ISTFT komplex; ein fruehes Float-Cast wirft den
        # Imaginaerteil weg und destabilisiert die Rueckprojektion unnoetig.
        _wiener_out: np.ndarray = np.asarray(enhanced * gain, dtype=enhanced.dtype)
        return _wiener_out  # type: ignore[no-any-return]

    # ------------------------------------------------------------------

    def _stft(self, audio: np.ndarray, window: np.ndarray) -> np.ndarray:
        """Short-Time Fourier Transform → komplexe Matrix (T, F).

        Implementiert via scipy.signal.stft (OLA-konsistent, PGHI-konform).
        Ersetzt die verbotene np.fft.rfft-Frame-Schleife aus v2.0.

        Args:
            audio:  1D-Audio-Signal.
            window: Hann-Fensterfunktion (Länge = _WINDOW_SIZE) — für
                    Konsistenz mit ISTFT übergeben, aber scipy nutzt intern
                    die Hann-Parameterisierung.

        Returns:
            np.ndarray: Komplexe STFT-Matrix, Form (T, F).
        """
        _, _, Zxx = safe_stft(
            audio,
            fs=1,  # normierte Frequenzachse — absolute Werte nicht benötigt
            window=window,  # type: ignore[arg-type]
            nperseg=self._WINDOW_SIZE,
            noverlap=self._WINDOW_SIZE - self._HOP_SIZE,
            boundary="even",
            padded=True,
        )
        _stft_out: np.ndarray = np.asarray(Zxx.T, dtype=np.complex128)  # type: ignore[no-any-return]
        return _stft_out  # scipy: (F, T) → intern (T, F)

    def _istft(self, stft: np.ndarray, window: np.ndarray, orig_len: int) -> np.ndarray:
        """OLA-Rücksynthese via scipy.signal.istft → Signal der Länge orig_len.

        Ersetzt die verbotene np.fft.irfft-Frame-Schleife aus v2.0.
        PGHI-Phasenkonsistenz durch scipy-interne OLA-Normierung gewährleistet.

        Args:
            stft:     Komplexe STFT-Matrix, Form (T, F).
            window:   Hann-Fensterfunktion (muss identisch zu _stft sein).
            orig_len: Ziel-Länge des Ausgangssignals.

        Returns:
            np.ndarray: 1D-Ausgangssignal, Länge = orig_len, clip[-1, 1].
        """
        _, out = sig.safe_istft(
            stft.T,  # intern (T, F) → scipy erwartet (F, T)
            fs=1,
            window=window,
            nperseg=self._WINDOW_SIZE,
            noverlap=self._WINDOW_SIZE - self._HOP_SIZE,
            boundary=True,
        )
        out = np.real(out)
        if len(out) > orig_len:
            out = out[:orig_len]
        elif len(out) < orig_len:
            out = np.pad(out, (0, orig_len - len(out)), mode="edge")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        _istft_out: np.ndarray = np.asarray(out, dtype=np.float32)
        return _istft_out

    # ------------------------------------------------------------------
    # Transientenmaske
    # ------------------------------------------------------------------

    def _compute_transient_mask(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Einsatz-Detektion via Energie-Anstieg (>9.5 dB in 5 ms).

        Returns:
            mask: Float-Array [0, 1], Länge = Anzahl RMS-Frames.
                  1.0 = Transiente (bleibt unverändert).
        """
        win_s = int(0.010 * sample_rate)  # 10 ms Fenster
        hop_s = max(1, int(0.005 * sample_rate))  # 5 ms Hop
        n_frames = (len(audio) - win_s) // hop_s + 1

        rms = np.zeros(n_frames)
        for i in range(n_frames):
            s = i * hop_s
            rms[i] = np.sqrt(np.mean(audio[s : s + win_s] ** 2))

        mask = np.zeros(n_frames)
        for i in range(1, n_frames):
            if rms[i - 1] > 1e-8 and rms[i] / rms[i - 1] > 3.0:
                # Energie-Anstieg > 9.5 dB in 5 ms → Transiente
                extend = int(0.025 * sample_rate / hop_s)  # 25 ms schützen
                hi = min(i + extend, n_frames)
                mask[i:hi] = 1.0

        return mask  # type: ignore[no-any-return]
