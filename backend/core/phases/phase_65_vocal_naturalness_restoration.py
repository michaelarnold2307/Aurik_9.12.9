"""
Phase 65 — DSP-Vocal-Naturalness-Restaurierung (KORREKTIV, §0a Restoration-only).

Spec §7.10 [RELEASE_MUST] (Spec 06, v10.0.0)

Zweck: Behebt VQI-Abfall nach ML-NR-Phasen (phase_03, phase_29, phase_20) durch
3-stufigen DSP-Korrektiv-Prozess ohne Halluzination, §0a-konform für Restoration-Modus.

§0a-Invariante:
    VERBOTEN in studio_2026-Modus (dort: phase_42_vocal_enhancement).
    KORREKTIV — kein Energiegewinn über Input hinaus erlaubt (subraktiver Fallback).

3-stufiger Algorithmus (§7.10a):
    Stufe 1 — Spektral-Tilt-Korrektur:
        estimate_spectral_tilt(pre_nr) − estimate_spectral_tilt(post_nr) → Δtilt
        wenn |Δtilt| > 1.5 dB: Shelving-EQ ±3 dB, F_shelf=2000 Hz
    Stufe 2 — HNR-Blend:
        compute_hnr(pre_nr) − compute_hnr(post_nr) → ΔHNR
        wenn ΔHNR > 2.5 dB: blend = clip(ΔHNR/10, 0, 0.35)
        apply_hnr_blend() (kanonisch via backend.core.dsp.hnr_guard)
    Stufe 3 — Formant-Tilt-Korrektur:
        LPC F1–F4 pre/post via check_formant_shift_db() + lpc_formant_enhance()
        wenn |Δ| > 1.5 dB: narrow shelving ±2.5 dB, Q=6

VQI-Guard: compute_vqi(pre_nr, result) < compute_vqi(pre_nr, post_nr) → Rollback
Aktivierungsgate: panns_singing ≥ 0.25 AND (ΔHNR > 2.5 OR |tilt_delta| > 1.5)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np
import scipy.signal as sps

from backend.core.audio_utils import to_channels_last
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.phase_strength_contract import resolve_phase_strength_contract

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithmus-Konstanten (§7.10a normativ)
# ---------------------------------------------------------------------------
_HNR_DELTA_THRESHOLD: float = 2.5  # dB — ab wann HNR-Blend aktiv
_TILT_DELTA_THRESHOLD: float = 1.5  # dB — ab wann Spektral-Tilt-Korrektur aktiv
_TILT_SHELF_HZ: float = 2000.0  # Hz — Shelving-Frequenz
_TILT_MAX_BOOST_DB: float = 3.0  # dB — Maximaler Tilt-Boost (limitierend)
_FORMANT_MAX_BOOST_DB: float = 1.0  # dB — Maximaler Formant-Boost (§2.71: F1/F2 ≤ ±1 dB Pflicht)
_FORMANT_SHIFT_THRESHOLD_DB: float = 1.5  # dB — Formant-Shift-Gate
_HNR_BLEND_MAX: float = 0.35  # Maximaler Dry-Wet-Blend-Faktor
_PANNS_SINGING_GATE: float = 0.25  # Mindest-Vocal-Confidence


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_mono_float64(audio: np.ndarray) -> np.ndarray:
    """Kanal-unabhängige Mono-Konvertierung nach float64."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a  # type: ignore[no-any-return]
    if a.ndim == 2:
        if a.shape[0] <= 2 and a.shape[1] > a.shape[0]:
            # [C, T] Layout (nach to_channels_last)
            return a.mean(axis=0)  # type: ignore[no-any-return]
        return a.mean(axis=1)  # type: ignore[no-any-return]
    return a.flatten()  # type: ignore[no-any-return]


def _estimate_spectral_tilt_db(audio_mono: np.ndarray, sr: int) -> float:
    """Schätzt Spektral-Tilt als Steigung der Power-Spektrumdichte (dB/oct).

    Positive Werte = mehr Energie bei tiefen Frequenzen (dunkler Klang).
    Negative Werte = mehr Energie bei hohen Frequenzen (heller Klang).
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 512:
        return 0.0
    window = np.hanning(n_fft)
    # Mittelwert über mehrere Frames für Stabilität
    n_frames = max(1, len(audio_mono) // n_fft)
    spectra = []
    for i in range(min(n_frames, 8)):
        start = i * n_fft
        seg = audio_mono[start : start + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spectra.append(np.abs(np.fft.rfft(seg * window)) ** 2)

    if not spectra:
        return 0.0

    mean_spec = np.mean(spectra, axis=0) + 1e-20
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Lineare Regression über log-Frequenz vs dB-Energie (nur 100 Hz – 8 kHz)
    mask = (freqs >= 100.0) & (freqs <= 8000.0)
    if mask.sum() < 8:
        return 0.0

    log_f = np.log2(freqs[mask] + 1e-9)
    db_e = 10.0 * np.log10(mean_spec[mask])
    # Lineare Regression
    coef = np.polyfit(log_f, db_e, 1)
    return float(coef[0])  # dB/octave Steigung


def _apply_shelving_eq(
    audio: np.ndarray,
    sr: int,
    shelf_hz: float,
    boost_db: float,
    shelf_type: str = "low",
) -> np.ndarray:
    """Wendet ein einfaches Low- oder High-Shelf-EQ an.

    Args:
        audio: [T] oder [C, T] Layout
        sr:    Abtastrate
        shelf_hz: Shelf-Frequenz
        boost_db: Gain in dB (positiv = Boost, negativ = Cut)
        shelf_type: "low" oder "high"
    """
    # Biquad-Shelf-Filter via scipy.signal
    # A = 10^(dB/40) (Standard-Shelf-Gain)
    A = 10.0 ** (boost_db / 40.0)
    w0 = 2.0 * np.pi * shelf_hz / sr
    S = 1.0  # Shelf-Slope = 1 (flache Flanke)
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

    if shelf_type == "low":
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    else:  # "high"
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    # Normalisieren
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    sos = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])

    audio_f64 = np.asarray(audio, dtype=np.float64)
    if audio_f64.ndim == 1:
        return np.nan_to_num(sps.sosfiltfilt(sos, audio_f64).astype(audio.dtype), nan=0.0)  # type: ignore[no-any-return]
    if audio_f64.ndim == 2:
        if audio_f64.shape[0] <= 2 and audio_f64.shape[1] > audio_f64.shape[0]:
            # [C, T] — nach to_channels_last
            return np.nan_to_num(np.stack([sps.sosfiltfilt(sos, audio_f64[c]) for c in range(audio_f64.shape[0])]).astype(audio.dtype), nan=0.0)  # type: ignore[no-any-return]
        return np.nan_to_num(np.stack(  # type: ignore[no-any-return]
            [sps.sosfiltfilt(sos, audio_f64[:, c]) for c in range(audio_f64.shape[1])],
            axis=1,
        ).astype(audio.dtype), nan=0.0)
    return np.nan_to_num(audio, nan=0.0)


# ---------------------------------------------------------------------------
# PhaseInterface-Klasse
# ---------------------------------------------------------------------------


class VocalNaturalnessRestorationPhase(PhaseInterface):
    """Phase 65: DSP-Vocal-Naturalness-Restaurierung.

    Spec §7.10 (Spec 06 v10.0.0).
    KORREKTIV — Restoration-only (§0a). Kein Energiegewinn über Input hinaus.
    """

    _PHASE_ID = "phase_65_vocal_naturalness_restoration"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name="Vocal Naturalness Restoration (DSP-Korrektiv)",
            category=PhaseCategory.RESTORATION,
            priority=6,
            version="1.0.0",
            dependencies=["phase_03", "phase_29", "phase_20"],
            estimated_time_factor=0.03,
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.88,
            description=(
                "§7.10 DSP-Korrektiv (3-Stufen): Spektral-Tilt + HNR-Blend + Formant-Tilt-Korrektur. "
                "Activation gate: panns_singing ≥ 0.25 AND (ΔHNR > 2.5 OR |tilt_delta| > 1.5). "
                "Restoration-only — §0a VERBOTEN in Studio 2026. "
                "VQI-Guard: Rollback wenn VQI nach Verarbeitung < VQI vorher."
            ),
        )

    def process(  # type: ignore[override]
        self, audio: np.ndarray, sample_rate: int = 48000, material_type: str = "unknown", **kwargs: Any
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="65")
        check_ml_model_ready("Whisper", phase_name="65")
        """DSP-Vocal-Naturalness-Restaurierung.

        Args:
            audio:       Mono oder Stereo
            sample_rate: Abtastrate Hz (MUSS 48000)
            **kwargs:
                panns_singing (float): PANNs-Singing-Konfidenz [0, 1]
                pre_nr_audio (np.ndarray): Audio VOR NR-Phasen (aus _restoration_context)
                quality_mode (str): "restoration" | "studio_2026"
                strength (float): Phase-Strength [0, 1]
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"Phase65 SR MUSS 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p65_transposed = to_channels_last(audio)
        self.validate_input(audio)
        t0 = time.time()

        _p65_meta: dict = {
            "algorithm": "vocal_naturalness_restoration_dsp",
            "stages_applied": [],
            "activation_triggered": False,
        }
        _strength_ctx = resolve_phase_strength_contract(kwargs)
        phase_locality_factor = float(_strength_ctx["phase_locality_factor"])
        _p65_meta["phase_locality_factor"] = phase_locality_factor

        # §0a Guard: Phase_65 ist VERBOTEN in Studio 2026
        quality_mode = str(kwargs.get("quality_mode", "restoration")).strip().lower()
        if quality_mode in ("studio_2026", "studio2026"):
            logger.debug(
                "§0a Verarbeitungsschritt65: uebersprungen in studio_2026 Betriebsart (use Verarbeitungsschritt_42_vocal_enhancement instead)"
            )
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta["algorithm"] = "skipped_studio2026_mode"
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0},
            )

        # Vocal-Gate: nur bei panns_singing >= 0.25
        panns_singing = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if panns_singing < _PANNS_SINGING_GATE:
            logger.debug(
                "Verarbeitungsschritt65: vocal gate not met (panns_singing=%.3f < %.2f)",
                panns_singing,
                _PANNS_SINGING_GATE,
            )
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta["algorithm"] = "skipped_no_vocal"
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0},
            )

        # Pre-NR-Checkpoint (erforderlich für alle 3 Stufen)
        pre_nr_audio_raw = kwargs.get("pre_nr_audio")
        if pre_nr_audio_raw is None:
            logger.debug("Verarbeitungsschritt65: pre_nr_audio nicht in kwargs — uebersprungen")
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta["algorithm"] = "skipped_no_pre_nr_audio"
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0},
            )

        pre_nr_audio = np.asarray(pre_nr_audio_raw, dtype=np.float32)
        # to_channels_last für pre_nr_audio
        pre_nr_ch, _ = to_channels_last(pre_nr_audio)

        # Mono für Analyse
        pre_mono = _to_mono_float64(pre_nr_ch.astype(np.float64))
        post_mono = _to_mono_float64(audio.astype(np.float64))

        # --- Vorab: ΔHNR und Tilt-Delta berechnen ---
        hnr_pre = 0.0
        hnr_post = 0.0
        tilt_pre = 0.0
        tilt_post = 0.0

        try:
            from backend.core.dsp.hnr_guard import (  # pylint: disable=import-outside-toplevel
                compute_hnr as _compute_hnr65,
            )

            hnr_pre = float(_compute_hnr65(pre_mono.astype(np.float32), sample_rate))
            hnr_post = float(_compute_hnr65(post_mono.astype(np.float32), sample_rate))
        except Exception as _hnr_exc:
            logger.debug("Verarbeitungsschritt65 berechnen_hnr nicht blockierend: %s", _hnr_exc)

        tilt_pre = _estimate_spectral_tilt_db(pre_mono, sample_rate)
        tilt_post = _estimate_spectral_tilt_db(post_mono, sample_rate)
        delta_hnr = hnr_pre - hnr_post
        tilt_delta = tilt_post - tilt_pre  # positiv = post wurde dunkler

        # Aktivierungsgate (§7.10a)
        activation_triggered = (delta_hnr > _HNR_DELTA_THRESHOLD) or (abs(tilt_delta) > _TILT_DELTA_THRESHOLD)

        if not activation_triggered:
            logger.debug(
                "Verarbeitungsschritt65: activation gate not met (delta_hnr=%.2f, tilt_delta=%.2f)",
                delta_hnr,
                tilt_delta,
            )
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta.update(
                {
                    "activation_triggered": False,
                    "delta_hnr": round(delta_hnr, 3),
                    "tilt_delta": round(tilt_delta, 3),
                    "algorithm": "skipped_gate_not_met",
                }
            )
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0, "delta_hnr": delta_hnr, "tilt_delta": tilt_delta},
            )

        _p65_meta["activation_triggered"] = True
        _p65_meta["delta_hnr"] = round(delta_hnr, 3)
        _p65_meta["tilt_delta"] = round(tilt_delta, 3)

        result = audio.copy().astype(np.float32)
        effective_strength = float(_strength_ctx["effective_strength"])
        _p65_meta["effective_strength"] = round(effective_strength, 4)

        # §M-2 §0p VFA-Schutzzonen: Vibrato-Passagen (4–7 Hz F0) auf max. 0.20 begrenzen.
        # phase_65 greift via HNR-Blend und Spektral-Tilt aktiv ins Vokal-Signal ein —
        # ohne Vibrato-Cap würde das die F0-Modulationstiefe reduzieren (§2.72).
        _p65_vibrato_zones = kwargs.get("vibrato_zones") or []
        _p65_passaggio_zones = kwargs.get("passaggio_zones") or []
        if _p65_vibrato_zones or _p65_passaggio_zones:
            # Prüfe ob die Signalmitte in einer Schutzzone liegt
            _n65 = audio.shape[0]
            _center_s = float(_n65 // 2) / float(max(sample_rate, 1))
            _p65_vib_cap = 1.0
            for _vz in _p65_vibrato_zones:
                try:
                    _vzs = float(getattr(_vz, "start_sample", 0)) / float(max(sample_rate, 1))
                    _vze = float(getattr(_vz, "end_sample", 0)) / float(max(sample_rate, 1))
                    if _vzs <= _center_s <= _vze:
                        _p65_vib_cap = min(_p65_vib_cap, 0.20)
                        break
                except Exception as e:
                    logger.warning(
                        "Verarbeitungsschritt_65_vocal_naturalness_restoration.py::unbekannter Ersatzpfad: %s", e
                    )
            for _pz in _p65_passaggio_zones:
                try:
                    _pzs = float(getattr(_pz, "start_sample", getattr(_pz, "start_s", 0)) or 0)
                    _pze = float(getattr(_pz, "end_sample", getattr(_pz, "end_s", 0)) or 0)
                    if _pzs > 1.0:  # Samples → Sekunden
                        _pzs /= float(max(sample_rate, 1))
                        _pze /= float(max(sample_rate, 1))
                    if _pzs <= _center_s <= _pze:
                        _p65_vib_cap = min(_p65_vib_cap, 0.35)
                        break
                except Exception as e:
                    logger.warning(
                        "Verarbeitungsschritt_65_vocal_naturalness_restoration.py::unbekannter Ersatzpfad: %s", e
                    )
            if _p65_vib_cap < effective_strength:
                effective_strength = _p65_vib_cap
                _p65_meta["vibrato_zone_cap_applied"] = True
                _p65_meta["effective_strength"] = round(effective_strength, 4)
                logger.debug(
                    "Verarbeitungsschritt65 §M-2 VFA-Cap: effective_strength → %.2f (Vibrato/Passaggio-Schutz)",
                    effective_strength,
                )

        # ── §SVM-1 SingerVoiceModel: Stimm-Modell für natürliche Vokal-Restaurierung ──
        _svm_65 = kwargs.get("singer_voice_model")
        if _svm_65 is not None and isinstance(_svm_65, dict) and _svm_65.get("confidence", 0.0) > 0.3:
            try:
                _svm65_formants = _svm_65.get("formant_targets", {}) or {}
                _svm65_vr = float(_svm_65.get("vibrato_rate_hz", 0.0) or 0.0)
                _svm65_vd = float(_svm_65.get("vibrato_depth_cents", 0.0) or 0.0)
                _svm65_hnr = float(_svm_65.get("hnr_db", 20.0) or 20.0)
                _svm65_tilt = float(_svm_65.get("spectral_tilt_db_per_octave", 0.0) or 0.0)
                # Vibrato-Erhalt: SVM-Vibrato als Ziel für Tilt-Korrektur
                if _svm65_vr > 0 and _svm65_vd > 30.0:
                    kwargs.setdefault("vibrato_zones_preserve", True)
                # HNR-Schutz: bei bereits rauer Stimme HNR-Blend konservativer
                if _svm65_hnr < 18.0:
                    effective_strength = float(
                        np.clip(effective_strength * (0.65 + 0.35 * _svm65_hnr / 18.0), 0.0, 1.0)
                    )
                    _p65_meta["effective_strength"] = round(effective_strength, 4)
                # Formant-Ziele als Referenz für Tilt-Korrektur
                if _svm65_formants and _svm65_tilt != 0:
                    _p65_meta["formant_targets_from_svm"] = True
                logger.debug(
                    "Verarbeitungsschritt65 §SVM-1 SVM: hnr=%.1fdB vibrato=%.1fHz/%.1fcent formants=%d → eff=%.3f",
                    _svm65_hnr,
                    _svm65_vr,
                    _svm65_vd,
                    len(_svm65_formants),
                    effective_strength,
                )
            except Exception as _svm_exc_65:
                logger.debug("Verarbeitungsschritt65 §SVM-1 nicht blockierend: %s", _svm_exc_65)

        # ---- Stufe 1: Spektral-Tilt-Korrektur ----
        if abs(tilt_delta) > _TILT_DELTA_THRESHOLD:
            try:
                # tilt_delta > 0 → post wurde dunkler (HF verloren) → High-Shelf-Boost
                # tilt_delta < 0 → post wurde heller → Low-Shelf-Boost
                boost_db = float(np.clip(-tilt_delta * 0.5, -_TILT_MAX_BOOST_DB, _TILT_MAX_BOOST_DB))
                boost_db *= effective_strength
                if abs(boost_db) > 0.3:
                    shelf_type = "high" if boost_db > 0 else "low"
                    result_tilted = _apply_shelving_eq(result, sample_rate, _TILT_SHELF_HZ, abs(boost_db), shelf_type)
                    result = result_tilted.astype(np.float32)
                    _p65_meta["stages_applied"].append(f"tilt_correction_{shelf_type}_{boost_db:.1f}dB")
                    logger.debug("Verarbeitungsschritt65 Stufe1 Tilt: boost=%.2f dB %s-shelf", boost_db, shelf_type)
            except Exception as _tilt_exc:
                logger.debug("Verarbeitungsschritt65 Stufe1 tilt nicht blockierend: %s", _tilt_exc)

        # ---- Stufe 2: HNR-Blend ----
        if delta_hnr > _HNR_DELTA_THRESHOLD:
            try:
                from backend.core.dsp.hnr_guard import (  # pylint: disable=import-outside-toplevel
                    apply_hnr_blend as _apply_hnr65,
                )

                _hnr_blended65, _hnr_diag65 = _apply_hnr65(
                    pre_nr_ch.astype(np.float32),
                    result,
                    sample_rate,
                )
                if _hnr_diag65.get("over_cleaned"):
                    # Angepasster Blend basierend auf ΔHNR
                    manual_blend = float(np.clip(delta_hnr / 10.0, 0.0, _HNR_BLEND_MAX)) * effective_strength
                    result = ((1.0 - manual_blend) * result + manual_blend * pre_nr_ch.astype(np.float32)).astype(
                        np.float32
                    )
                    _p65_meta["stages_applied"].append(f"hnr_blend_{manual_blend:.3f}")
                    logger.debug("Verarbeitungsschritt65 Stufe2 HNR-Blend: blend=%.3f", manual_blend)
                elif _hnr_diag65.get("blend_applied"):
                    result = _hnr_blended65
                    _p65_meta["stages_applied"].append("hnr_blend_auto")
            except Exception as _hnr65_exc:
                logger.debug("Verarbeitungsschritt65 Stufe2 HNR-Blend nicht blockierend: %s", _hnr65_exc)

        # ---- Stufe 3: Formant-Tilt-Korrektur ----
        try:
            from backend.core.dsp.lpc_formant_tracker import (  # pylint: disable=import-outside-toplevel
                check_formant_shift_db as _check_fshift65,
            )
            from backend.core.dsp.lpc_formant_tracker import (  # pylint: disable=import-outside-toplevel
                lpc_formant_enhance as _lpc_enhance65,
            )

            _rollback_needed, _max_shift = _check_fshift65(
                pre_nr_ch.astype(np.float32),
                result,
                sample_rate,
                threshold_db=_FORMANT_SHIFT_THRESHOLD_DB,
            )
            if _rollback_needed and _max_shift > _FORMANT_SHIFT_THRESHOLD_DB:
                # Formant-Tilt-Korrektur: sanfte LPC-Boost-Rekonstruktion
                _formant_boost_db = float(np.clip(_max_shift * 0.4, 0.1, _FORMANT_MAX_BOOST_DB)) * effective_strength
                # §Lücke3 WLPC: era-Info für noise-robusten Formant-Pfad
                _era_p65_lpc = kwargs.get("decade") or (kwargs.get("_restoration_context") or {}).get("era_decade")
                result_enhanced = _lpc_enhance65(
                    result,
                    sample_rate,
                    max_boost_db=_formant_boost_db,
                    era_decade=int(_era_p65_lpc) if _era_p65_lpc is not None else None,  # type: ignore[call-arg]
                )
                result = result_enhanced.astype(np.float32)
                _p65_meta["stages_applied"].append(f"formant_tilt_{_formant_boost_db:.2f}dB")
                logger.debug(
                    "Verarbeitungsschritt65 Stufe3 Formant-Tilt: shift=%.2f dB, boost=%.2f dB",
                    _max_shift,
                    _formant_boost_db,
                )
        except Exception as _ft65_exc:
            logger.debug("Verarbeitungsschritt65 Stufe3 formant-tilt nicht blockierend: %s", _ft65_exc)

        # NaN-Safety + Clip
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)

        # §2.36 LyricsGuided-Phonemgrenzen-Schutz (RELEASE_MUST ab 9.10.x):
        # Konsonanten-Bursts (Plosive/Frikative < 20 ms) müssen in phase_65 erhalten bleiben —
        # DSP-Eingriffe (Tilt, HNR-Blend) dürfen Artikulation nicht glätten.
        # Phonem-Frames werden vollständig auf das Original zurückgeblendet.
        try:
            from backend.core.lyrics_guided_enhancement import (  # pylint: disable=import-outside-toplevel
                get_phoneme_mask as _get_pmask65,
            )

            _p65_audio_1ch = audio if audio.ndim == 1 else audio[0]
            _p65_pmask_frames = _get_pmask65(_p65_audio_1ch, sample_rate)
            if _p65_pmask_frames is not None and len(_p65_pmask_frames) > 0:
                # Frame-Maske → Sample-Maske (nearest-neighbour + Decay-Fenster 5 ms)
                _p65_hop = max(1, len(_p65_audio_1ch) // max(1, len(_p65_pmask_frames)))
                _p65_smask = np.repeat(_p65_pmask_frames.astype(np.float32), _p65_hop)
                _p65_smask = _p65_smask[: len(_p65_audio_1ch)]
                if len(_p65_smask) < len(_p65_audio_1ch):
                    _p65_smask = np.pad(_p65_smask, (0, len(_p65_audio_1ch) - len(_p65_smask)))
                # Sanftes Decay-Fenster (5 ms = _p65_decay_samples) an Phonemgrenzen
                _p65_decay_samples = max(1, int(0.005 * sample_rate))
                _p65_kernel = np.ones(_p65_decay_samples, dtype=np.float32) / _p65_decay_samples
                _p65_smask = np.convolve(_p65_smask, _p65_kernel, mode="same").clip(0.0, 1.0)
                # Blend: Phonem-Frames → Original; Rest → DSP-Ergebnis
                _p65_orig_1ch = audio if audio.ndim == 1 else (audio[0] if result.ndim == 1 else audio)
                if result.ndim == 2:
                    for _ch65 in range(result.shape[0]):
                        _orig_ch = audio[_ch65] if audio.ndim == 2 else audio
                        result[_ch65] = (_p65_smask * _orig_ch + (1.0 - _p65_smask) * result[_ch65]).astype(np.float32)
                else:
                    result = (_p65_smask * _p65_orig_1ch + (1.0 - _p65_smask) * result).astype(np.float32)
                _p65_phoneme_frames_protected = int(np.sum(_p65_pmask_frames))
                _p65_meta["phoneme_frames_protected"] = _p65_phoneme_frames_protected
                logger.debug(
                    "§2.36 Verarbeitungsschritt_65 Phonemschutz: %d Frames geschützt (%.1f%%)",
                    _p65_phoneme_frames_protected,
                    100.0 * float(np.mean(_p65_pmask_frames)),
                )
        except Exception as _pmask65_exc:
            logger.debug("§2.36 Verarbeitungsschritt_65 Phonemschutz nicht blockierend: %s", _pmask65_exc)

        # ---- VQI-Guard: Rollback wenn VQI schlechter geworden ----
        _vqi_before: float = -1.0
        _vqi_after: float = -1.0
        try:
            from backend.core.musical_goals.era_vocal_profile import (
                get_era_vocal_profile as _gevp_p65,  # pylint: disable=import-outside-toplevel  # §EraVocalProfile
            )
            from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
                compute_vqi as _compute_vqi65,
            )

            _era_dec_p65 = kwargs.get("decade") or (kwargs.get("_restoration_context") or {}).get("era_decade")
            _era_prof_p65 = _gevp_p65(int(_era_dec_p65)) if _era_dec_p65 else None
            _vqi_res_before = _compute_vqi65(
                pre_nr_ch.astype(np.float32), audio.astype(np.float32), sample_rate, era_profile=_era_prof_p65
            )
            _vqi_res_after = _compute_vqi65(
                pre_nr_ch.astype(np.float32), result, sample_rate, era_profile=_era_prof_p65
            )
            _vqi_before = float(_vqi_res_before.get("vqi", -1.0))
            _vqi_after = float(_vqi_res_after.get("vqi", -1.0))
            if _vqi_before > 0 and _vqi_after > 0 and _vqi_after < _vqi_before - 0.005:
                logger.warning(
                    "Verarbeitungsschritt65 VQI-Guard: vqi_after=%.3f < vqi_before=%.3f → Rollback",
                    _vqi_after,
                    _vqi_before,
                )
                result = audio.copy().astype(np.float32)
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                result = np.clip(result, -1.0, 1.0)
                _p65_meta["vqi_rollback"] = True
        except Exception as _vqi65_exc:
            logger.debug("Verarbeitungsschritt65 VQI-Guard nicht blockierend: %s", _vqi65_exc)

        _p65_meta["vqi_before"] = round(_vqi_before, 4)
        _p65_meta["vqi_after"] = round(_vqi_after, 4)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach Vokal-Naturalness-Restaurierung (§2.74, non-blocking)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_65,
            )

            _sc_result_65 = _scg_65(audio, result, sample_rate)
            if not _sc_result_65.ok:
                _sc_wet_65 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                result = (_sc_wet_65 * result + (1.0 - _sc_wet_65) * audio).astype(np.float32)
        except Exception as _sc_exc_65:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_65 spectral_color nicht blockierend: %s", _sc_exc_65
            )

        # V26 Onset-Guard (§2.77): Vokal-Onset-Transients schützen (non-blocking)
        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opg65,
            )

            result = _opg65(audio, result, None, max_delta_db=1.5)
        except Exception as _on65_exc:
            logger.debug("Verarbeitungsschritt65 V26 Onset-Guard (nicht blockierend): %s", _on65_exc)

        # §2.46e HallucinationGuard: Additive Phase darf kein halluziniertes Material einführen (VERBOTEN)
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _chk_hg65,
            )

            _hg_mode_65 = str(kwargs.get("mode", "restoration"))
            # channels-last [N,2] → mono
            _pre65_mono = (
                audio.mean(axis=1)
                if (audio.ndim == 2 and audio.shape[1] == 2 and audio.shape[0] > 2)
                else (audio.mean(axis=0) if audio.ndim == 2 else audio)
            )
            _post65_mono = (
                result.mean(axis=1)
                if (result.ndim == 2 and result.shape[1] == 2 and result.shape[0] > 2)
                else (result.mean(axis=0) if result.ndim == 2 else result)
            )
            _hg65 = _chk_hg65(
                _pre65_mono.astype(np.float32),
                _post65_mono.astype(np.float32),
                sr=sample_rate,
                mode=_hg_mode_65,
                bw_extension_context=True,
            )
            if _hg65.requires_rollback:
                result = audio.copy().astype(np.float32)
                logger.warning("§2.46e Verarbeitungsschritt_65 HallucinationGuard: rollback (spectral_novelty > 0.15)")
        except Exception as _hg65_exc:
            logger.debug("§2.46e Verarbeitungsschritt_65 HallucinationGuard (nicht blockierend): %s", _hg65_exc)

        # V19 Noise-Textur-Invariante (§NTI): Residual-Rauschen darf Material-Profil nicht ändern (non-blocking)
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt65_fn,
            )

            # channels-last [N,2] → channels-first [2,N] für Guard
            _a65cf = (
                audio.T.astype(np.float32)
                if (audio.ndim == 2 and audio.shape[1] == 2 and audio.shape[0] > 2)
                else audio.astype(np.float32)
            )
            _r65cf = (
                result.T.astype(np.float32)
                if (result.ndim == 2 and result.shape[1] == 2 and result.shape[0] > 2)
                else result.astype(np.float32)
            )
            _mat65_str = str(material_type) if material_type else "unknown"
            _nt65_thr = 0.18 if panns_singing >= 0.35 else 0.25
            _nt65_d = _nt65_fn(_a65cf - _r65cf, _mat65_str, sr=sample_rate)
            if _nt65_d > _nt65_thr:
                result = (0.5 * result + 0.5 * audio).astype(np.float32)
                logger.warning(
                    "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_65 noise_texture_distance=%.3f > %.2f → 50%% Dry-Blend",
                    _nt65_d,
                    _nt65_thr,
                )
        except Exception as _nt65_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_65 noise_texture (nicht blockierend): %s", _nt65_exc
            )

        if _p65_transposed:
            result = result.T

        return PhaseResult(
            success=True,
            audio=result,
            execution_time_seconds=time.time() - t0,
            metadata=_p65_meta,
            metrics={
                "activation_triggered": 1 if activation_triggered else 0,
                "phase_locality_factor": round(phase_locality_factor, 4),
                "effective_strength": round(effective_strength, 4),
                "delta_hnr": round(delta_hnr, 3),
                "tilt_delta": round(tilt_delta, 3),
                "vqi_before": round(_vqi_before, 4),
                "vqi_after": round(_vqi_after, 4),
                "stages_applied": len(_p65_meta.get("stages_applied", [])),
            },
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: VocalNaturalnessRestorationPhase | None = None
_lock = threading.Lock()


def get_phase_65() -> VocalNaturalnessRestorationPhase:
    """Thread-safe Singleton accessor (§0 Kopilot-Instructions Singleton-Pattern)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VocalNaturalnessRestorationPhase()
    assert _instance is not None
    return _instance
