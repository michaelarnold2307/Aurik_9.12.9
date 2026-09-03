"""
Phase 44: Guitar Enhancement v2.0 — Transient Boost + Harmonic Exciter + Presence EQ
======================================================================================

Korrektur des alten Stubs: falscher Rückgabetyp (rohes ndarray statt PhaseResult),
defekte Local-Imports (np nicht in Scope in Helper-Methoden), kein aurik_ml.

ALGORITHMUS:
  1. Spektralzentroid-basierte Genre-Einstufung (Rock / Jazz / Pop)
     → amplitude-gewichteter Schwerpunkt des Magnitude-Spektrums

  2. Transienten-Boost:
     - Hilbert-Hüllkurve × 0.15 zum Signal addieren
     - Betont Anschlag/Pick-Transiente

  3. Genre-adaptiver Harmonic Exciter:
     - Rock:  Soft-Clip via tanh (ungerade Harmonics, Gain 0.12)
     - Jazz:  Subtile Sinus-Verzerrung (Gain 0.07)
     - Pop:   Vollwellengleichrichter (gerade Harmonics, Gain 0.09)

  4. Presence-EQ:
     - Rock/Pop: +2 dB Peak bei 3 kHz, Q=1.8
     - Jazz:     +1 dB Peak bei 1.5 kHz, Q=2.5

  5. Normalisierungs-Pass, np.clip, PhaseResult

Author: Aurik Development Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import safe_to_mono
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls

    _FORMANT_SYSTEM_GUITAR = None
except Exception:
    _FormantSystemCls = None  # type: ignore[assignment,misc]
    _FORMANT_SYSTEM_GUITAR = None

try:
    from dsp.instrument_formant_corrector import correct_instrument_formant_drift
except Exception:
    correct_instrument_formant_drift = None  # type: ignore[assignment]

try:
    from backend.core.sub_stem_processor import process_sub_stems
except Exception:
    process_sub_stems = None  # type: ignore[assignment]

try:
    from backend.core.physics_resonance_enhancer import enhance_physics_resonance
except Exception:
    enhance_physics_resonance = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _get_formant_system_guitar():
    """Gibt the cached guitar formant system instance when available zurück."""
    if _FormantSystemCls is None:
        return None
    if _FORMANT_SYSTEM_GUITAR is None:
        return _FormantSystemCls(enhance_singers_formant=False)
    return _FORMANT_SYSTEM_GUITAR


def _spectral_centroid(audio: np.ndarray) -> float:
    """Amplitudengewichteter Spektralzentroid (normiert 0–1)."""
    mono = safe_to_mono(audio) if audio.ndim == 2 else audio
    mag = np.abs(np.fft.rfft(mono[: min(len(mono), 65536)]))
    total = mag.sum() + 1e-12
    freqs = np.arange(len(mag), dtype=np.float64)
    return float((freqs * mag).sum() / total / len(mag))


def _peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    """Biquad Peaking-EQ (Audio-EQ-Cookbook)."""
    w0 = 2.0 * np.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = b1
    a2 = 1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    # §2.51 zero-phase: filtfilt eliminates causal group-delay → no L/R interchannel lag.
    # padlen = 3*max(len(a),len(b))-1 = 8 — safe for 48 kHz audio (always >> 8 samples).
    if x.ndim == 1:
        return sig.filtfilt(b, a, x)  # type: ignore[no-any-return]
    # Handle both (2,N) channels-first and (N,2) channels-last
    if x.shape[0] == 2 and x.shape[1] > 2:
        return np.vstack([sig.filtfilt(b, a, x[ch, :]) for ch in range(x.shape[0])])  # type: ignore[no-any-return]
    return np.column_stack([sig.filtfilt(b, a, x[:, ch]) for ch in range(x.shape[1])])  # type: ignore[no-any-return]


class GuitarEnhancementPhase(PhaseInterface):
    """Transient-Boost + genre-adaptiver Harmonic Exciter + Presence-EQ für Gitarre."""

    PHASE_ID = "phase_44_guitar_enhancement"
    PHASE_NAME = "Guitar Enhancement (Transient + Exciter + EQ)"
    PHASE_DESCRIPTION = (
        "Genre-adaptive Gitarren-Verbesserung: Hilbert-Transient-Boost, "
        "Harmonic Exciter (tanh/sin/abs je nach Genre) und Presence-EQ. "
        "Kein aurik_ml, vollständig DSP."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name=self.PHASE_NAME,
            category=PhaseCategory.ENHANCEMENT,
            priority=4,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.03,
            memory_requirement_mb=50,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.82,
            description=self.PHASE_DESCRIPTION,
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("BEATs", phase_name="44")
        check_ml_model_ready("LAION-CLAP", phase_name="44")
        check_ml_model_ready("PANNs", phase_name="44")
        """
        Guitar Enhancement: Transient + Exciter + EQ.

        Args:
            audio:        Mono oder Stereo
            sample_rate:  Hz
            material_type: Träger-Material z.B. "tape", "vinyl", "unknown"
            **kwargs:     transient_gain (float, default 0.15)
                          exciter_gain   (float, default 1.0 = Genre-Default)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor: float = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength: float = float(kwargs.get("strength", 1.0))
        _effective_strength: float = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_44 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_44 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_44,
                )

                _fmz_44 = kwargs.get("forward_masking_zones") or _fmg_fn_44().compute_zones(audio, sample_rate)
                if _fmz_44:
                    _n_s_44 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_44 = sum(z.end_sample - z.start_sample for z in _fmz_44)
                    _zone_frac_44 = float(np.clip(_zone_s_44 / max(1, _n_s_44), 0.0, 1.0))
                    _effective_strength = float(np.clip(_effective_strength + _zone_frac_44 * 0.15, 0.0, 1.0))  # type: ignore[no-redef]
            except Exception as _fmg_exc_44:
                logger.debug("Verarbeitungsschritt44 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_44)

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.astype(audio.dtype),
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"spectral_centroid": 0.0},
            )

        # §2.46g soft_saturation-Guard: Guitar-Exciter und Transient-Boost auf gesättigtem
        # Material addieren Harmonics in bereits übersteuerten Regionen. Hard-Cap: 35 %.
        _p44_soft_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        _p44_soft_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        if _p44_soft_sat_preserve or _p44_soft_sat_sev > 0.30:
            _p44_sat_scale = 1.0
            if _p44_soft_sat_sev > 0.30:
                _p44_sat_scale = float(np.clip(1.0 - (_p44_soft_sat_sev - 0.30) * 1.0, 0.20, 1.0))
            if _p44_soft_sat_preserve and _p44_sat_scale > 0.35:
                _p44_sat_scale = 0.35
            _effective_strength = float(_effective_strength * _p44_sat_scale)
            logger.debug(
                "Verarbeitungsschritt 44 soft_saturation guard: severity=%.2f preserve=%s → scale=%.2f (strength=%.3f)",
                _p44_soft_sat_sev,
                _p44_soft_sat_preserve,
                _p44_sat_scale,
                _effective_strength,
            )

        transient_gain: float = float(kwargs.get("transient_gain", 0.15))
        exciter_gain: float = float(kwargs.get("exciter_gain", 1.0))
        transient_gain = float(transient_gain * _effective_strength)
        exciter_gain = float(exciter_gain * _effective_strength)

        x = audio.astype(np.float64)
        mono = safe_to_mono(x) if x.ndim == 2 else x

        # 1. Genre-Klassifikation via Spektralzentroid + Crest Factor
        centroid = _spectral_centroid(audio)
        rms = float(np.sqrt(np.mean(mono**2)))
        # V08: np.percentile statt np.max — robust gegen Einzel-Sample-Spikes im Crest-Factor.
        peak = float(np.percentile(np.abs(mono), 99.9))
        crest_db = 20.0 * np.log10(peak / (rms + 1e-10) + 1e-10)
        # Rock: bright centroid + moderate crest (distorted pickups)
        # Jazz: dark centroid + high crest (clean dynamics)
        # Pop: mid centroid
        if centroid > 0.40 and crest_db < 20.0:
            genre = "Rock"
        elif centroid < 0.28 and crest_db > 16.0:
            genre = "Jazz"
        else:
            genre = "Pop"

        peak_in = float(np.percentile(np.abs(audio), 99.9))

        # 2. Transient-Boost via attack-envelope differentiation (NOT Hilbert envelope)
        # True transient boost: detect fast amplitude rises and amplify the onset
        # HP-filter → envelope derivative → onset mask → mix scaled signal
        sos_hp = sig.butter(2, 200.0, btype="high", fs=sample_rate, output="sos")

        def _transient_boost_channel(ch_audio: np.ndarray) -> np.ndarray:
            hp = sig.sosfiltfilt(sos_hp, ch_audio)
            # Rectified envelope
            env = np.abs(hp)
            smooth_win = max(1, int(0.003 * sample_rate))  # 3ms smoothing
            env_smooth = np.convolve(env, np.ones(smooth_win) / smooth_win, mode="same")
            # Attack detector: positive derivative of smoothed envelope
            d_env = np.diff(env_smooth, prepend=env_smooth[0])
            attack = np.maximum(0.0, d_env)
            # §copilot Peak-Guard: percentile(99.9) instead of np.max so that a single
            # click/impulse artefact in defect-laden material does not drive a_max to an
            # extreme value that zeroes out all genuine transient boosts.
            a_max = float(np.percentile(attack, 99.9))
            if a_max > 1e-10:
                attack = attack / a_max
            # Boost signal at attack moments (wet/dry)
            return ch_audio * (1.0 + transient_gain * attack)  # type: ignore[no-any-return]

        if x.ndim == 1:
            x = _transient_boost_channel(x)
        else:
            # §2.51 M/S-Domain: Transient Boost nur auf Mid
            # Handle both (2,N) channels-first and (N,2) channels-last
            if x.shape[0] == 2 and x.shape[1] > 2:
                _x_ch0, _x_ch1 = x[0], x[1]  # (2,N)
            else:
                _x_ch0, _x_ch1 = x[:, 0], x[:, 1]  # (N,2)
            mid = (_x_ch0 + _x_ch1) / np.sqrt(2.0)
            side = (_x_ch0 - _x_ch1) / np.sqrt(2.0)
            mid = _transient_boost_channel(mid)
            _out_l = (mid + side) / np.sqrt(2.0)
            _out_r = (mid - side) / np.sqrt(2.0)
            # Preserve original orientation
            if x.shape[0] == 2 and x.shape[1] > 2:
                x = np.vstack([_out_l, _out_r])  # (2,N)
            else:
                x = np.column_stack([_out_l, _out_r])  # (N,2)

        # 3. Genre-adaptiver Harmonic Exciter (band-limited to body resonance region)
        # Limit exciter to 200-5000 Hz to avoid mud (< 200 Hz) and hash (> 5 kHz)
        # §2.51 Anti-Zeitversatz: sosfiltfilt — body_band wird als Harmonic Exciter
        # auf ch_audio aufaddiert; sosfilt erzeugt Zeitversatz → Kammfilter-Artefakt.
        sos_body = sig.butter(4, [200.0, 5000.0], btype="band", fs=sample_rate, output="sos")
        g = exciter_gain

        def _excite_channel(ch_audio: np.ndarray) -> np.ndarray:
            body_band = sig.sosfiltfilt(sos_body, ch_audio)
            if genre == "Rock":
                # Soft-clip for odd harmonics (pick attack grit)
                excited = np.tanh(body_band * 2.5) * 0.12 * g
            elif genre == "Jazz":
                # Subtle even harmonics (warmth, not distortion)
                excited = (np.abs(body_band) - body_band**2 * 0.5) * 0.07 * g
            else:  # Pop
                # Mild saturation
                excited = np.tanh(body_band * 1.5) * 0.09 * g
            return ch_audio + excited  # type: ignore[no-any-return]

        if x.ndim == 1:
            x = _excite_channel(x)
        else:
            # §2.51 M/S-Domain: Exciter nur auf Mid
            # Handle both (2,N) channels-first and (N,2) channels-last
            if x.shape[0] == 2 and x.shape[1] > 2:
                _xc0, _xc1 = x[0], x[1]  # (2,N)
            else:
                _xc0, _xc1 = x[:, 0], x[:, 1]  # (N,2)
            mid = (_xc0 + _xc1) / np.sqrt(2.0)
            side = (_xc0 - _xc1) / np.sqrt(2.0)
            mid = _excite_channel(mid)
            _xl = (mid + side) / np.sqrt(2.0)
            _xr = (mid - side) / np.sqrt(2.0)
            if x.shape[0] == 2 and x.shape[1] > 2:
                x = np.vstack([_xl, _xr])  # (2,N)
            else:
                x = np.column_stack([_xl, _xr])  # (N,2)

        # 4. Presence-EQ (genre-adaptive)
        if genre == "Rock":
            # Pick attack clarity at 3 kHz, string bite at 5 kHz
            x = _peaking_eq(x, sample_rate, freq=3000.0, gain_db=2.0, q=1.8)
            x = _peaking_eq(x, sample_rate, freq=5000.0, gain_db=1.2, q=2.5)
        elif genre == "Jazz":
            # Warm body at 800 Hz, subtle presence at 2 kHz
            x = _peaking_eq(x, sample_rate, freq=800.0, gain_db=1.0, q=2.0)
            x = _peaking_eq(x, sample_rate, freq=2000.0, gain_db=1.0, q=2.5)
        else:  # Pop
            x = _peaking_eq(x, sample_rate, freq=3000.0, gain_db=1.5, q=2.0)

        # 5. Normalisierung + Clip — §2.49 Peak-Guard: percentile(99.9) for peak-preservation
        peak_in = float(np.percentile(np.abs(audio), 99.9))
        peak_out = float(np.percentile(np.abs(x), 99.9))
        if peak_out > 1e-8 and peak_in > 1e-8:
            x = x * (peak_in / peak_out)
        processed = np.clip(x, -1.0, 1.0).astype(audio.dtype)

        logger.info(
            "Verarbeitungsschritt 44 GuitarEnhancement: genre=%s, centroid=%.3f, transient_gain=%.2f",
            genre,
            centroid,
            transient_gain,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # Instrument-guided formant enhancement (InstrumentFormantTargets — guitar body resonances)
        igt_frames = 0
        try:
            formant_system = _get_formant_system_guitar()
            if formant_system is not None:
                processed, igt_report = formant_system.instrument_guided_enhance(
                    processed, sample_rate, instrument="guitar", correction_strength=0.20
                )
                igt_frames = igt_report.get("frames_processed", 0)
                logger.debug("Verarbeitungsschritt 44 InstrumentFormant: guitar frames=%d", igt_frames)
        except Exception as _igt_exc:
            logger.debug("Verarbeitungsschritt 44 instrument_guided_verbessern uebersprungen: %s", _igt_exc)

        # Formant-Drift-Korrektur via DTW (Schritt 3)
        try:
            _drift_corrector = correct_instrument_formant_drift
            if _drift_corrector is not None:
                drift_result = _drift_corrector(processed, sample_rate, instrument="guitar")
                processed = drift_result.audio
                logger.debug(
                    "Verarbeitungsschritt 44 drift correction: erkannt=%s frames=%d/%d drift=%.1fHz",
                    drift_result.drift_detected,
                    drift_result.n_frames_corrected,
                    drift_result.total_frames,
                    drift_result.mean_drift_hz,
                )
        except Exception as _drift_exc:
            logger.debug("Verarbeitungsschritt 44 drift correction uebersprungen: %s", _drift_exc)

        # Sub-Stem-Verarbeitung (Schritt 4)
        try:
            if process_sub_stems is not None:
                # MAX_STRENGTH cap for SubStemProcessor is 0.60; outer blend (_effective_strength)
                # controls final PMGG mix, so internal processing runs at full capacity.
                ss_result = process_sub_stems(
                    processed,
                    sample_rate,
                    instrument="guitar",
                    processing_strength=0.60,
                )
                processed = ss_result.audio
                logger.debug(
                    "Verarbeitungsschritt 44 sub-stem: bands=%d strength=%.2f",
                    ss_result.n_bands,
                    ss_result.processing_strength,
                )
        except Exception as _ss_exc:
            logger.debug("Verarbeitungsschritt 44 sub-stem uebersprungen: %s", _ss_exc)

        # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
        try:
            if enhance_physics_resonance is not None:
                # PhysicsResonanceEnhancer MAX_STRENGTH=1.0; 0.70 = conservative-max for guitar
                # (internal MAX_GAIN_DB=4.0 provides hard per-peak ceiling).
                pr_result = enhance_physics_resonance(
                    processed, sample_rate, instrument="guitar", enhancement_strength=0.70
                )
                processed = pr_result.audio
                logger.debug(
                    "Verarbeitungsschritt 44 physics resonance: peaks=%d strength=%.2f",
                    pr_result.n_peaks,
                    pr_result.enhancement_strength,
                )
        except Exception as _pr_exc:
            logger.debug("Verarbeitungsschritt 44 physics resonance uebersprungen: %s", _pr_exc)

        if 0.0 < _effective_strength < 1.0 and processed.shape == audio.shape:
            processed = audio + _effective_strength * (processed - audio)

        # §2.46e HallucinationGuard: Additive Phase darf kein halluziniertes Material einführen (VERBOTEN)
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _chk_hg44,
            )

            _hg_mode_44 = str(kwargs.get("mode", "restoration"))
            _pre44_mono = (
                audio.mean(axis=0)
                if (audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2)
                else (audio.mean(axis=1) if audio.ndim == 2 else audio)
            )
            _post44_mono = (
                processed.mean(axis=0)
                if (processed.ndim == 2 and processed.shape[0] == 2 and processed.shape[1] > 2)
                else (processed.mean(axis=1) if processed.ndim == 2 else processed)
            )
            _hg44 = _chk_hg44(
                _pre44_mono.astype(np.float32),
                _post44_mono.astype(np.float32),
                sr=sample_rate,
                mode=_hg_mode_44,
                bw_extension_context=True,
            )
            if _hg44.requires_rollback:
                processed = audio.copy()
                logger.warning("§2.46e Verarbeitungsschritt_44 HallucinationGuard: rollback (spectral_novelty > 0.15)")
        except Exception as _hg44_exc:
            logger.debug("§2.46e Verarbeitungsschritt_44 HallucinationGuard (nicht blockierend): %s", _hg44_exc)

        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "genre": genre,
                "spectral_centroid": centroid,
                "transient_gain": transient_gain,
                "instrument_formant_frames": igt_frames,
                # Semantic metadata contract (transport-safe defaults).
                "clap_model_used": "dsp_only",
                "clap_confidence": 0.0,
                "clap_top_genres": [],
                "clap_top_instruments": [],
                "clap_embedding_32": [],
                "beats_model_used": "dsp_only",
                "beats_top_tags": [],
                "beats_embedding_32": [],
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            metrics={"genre": genre, "spectral_centroid": centroid},
        )
