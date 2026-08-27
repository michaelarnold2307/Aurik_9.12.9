"""
Hybrid Wow/Flutter Correction - AURIK 9.0 Phase 12 ML-Hybrid
=============================================================

Zwei-Stufen-Pitch-Detektion: pYIN (DSP) + RMVPE/PESTO (ML) für überlegene Genauigkeit.
§4.2-konform: klassisches YIN (de Cheveigné 2002) ist VERBOTEN als primäre Methode.

Architektur:
1. Stufe 1: pYIN Pitch-Detektion (Mauch & Dixon 2014)
   - HMM-basierte Voiced/Unvoiced-Klassifikation
   - Kumulative mittlere normalisierte Differenz (probabilistisch)
   - Robust bei verrauschten/historischen Signalen

2. Stufe 2: RMVPE/PESTO ML Pitch-Detektion
   - CNN-basiertes Pitch-Tracking
   - ±1 Cent Genauigkeit
   - Verhindert Oktavfehler
   - Besser bei komplexen/harmonisch dichten Signalen

Strategy-Modi:
- PYIN_ONLY: Schnelle pYIN-DSP-Detektion
- CREPE_ONLY: Reines ML (kompatibler Name; nutzt bestes verfügbares ML-Modell)
- HYBRID: pYIN -> ML-Verfeinerung für unsichere Regionen
- ADAPTIVE: Auswahl nach Konfidenz-Scores

Korrektur-Pipeline:
1. Pitch-Detektion (pYIN oder CREPE)
2. Wow/Flutter-Trennung (< 4 Hz vs. 4-100 Hz)
3. Phase-Vocoder Zeitstreckung (Korrektur)

Author: Aurik 10.0.0 Development Team
Version: 1.0.0
Date: 16. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PitchDetectionStrategy(Enum):
    """Pitch detection strategy selection."""

    PYIN_ONLY = "pyin_only"  # pYIN DSP (Mauch & Dixon 2014) — §4.2 konform
    CREPE_ONLY = "crepe_only"  # Pure ML CNN
    HYBRID = "hybrid"  # pYIN → CREPE-Verfeinerung
    ADAPTIVE = "adaptive"  # Auto-Auswahl nach Konfidenz
    POLYPHONIC = "polyphonic"  # Multi-F0 consensus (Capstan-kompetitiv, §2.12)
    # Backward-Alias (deprecated)
    YIN_ONLY = "pyin_only"  # Alias → PYIN_ONLY


@dataclass
class WowFlutterConfig:
    """Configuration for hybrid wow/flutter correction."""

    strategy: PitchDetectionStrategy = PitchDetectionStrategy.ADAPTIVE
    pyin_confidence_threshold: float = 0.4  # pYIN Mindest-Konfidenz
    crepe_model: str = "full"  # CREPE model size
    confidence_threshold: float = 0.7  # Mindest-Konfidenz für Pitch-Schätzwert
    enable_preprocessing: bool = True  # pYIN-Preprocessing aktivieren
    # Backward-Alias
    yin_threshold: float = 0.15  # unused — nur für Kompatibilität


@dataclass
class WowFlutterResult:
    """Result from hybrid wow/flutter correction."""

    pitch_trajectory: np.ndarray  # Pitch-Schätzwerte (Hz)
    confidence: np.ndarray  # Konfidenz-Scores (0-1)
    strategy_used: PitchDetectionStrategy
    pyin_applied: bool  # pYIN (Mauch & Dixon 2014) angewendet
    crepe_applied: bool
    processing_time: float
    mean_confidence: float
    metadata: dict[str, Any]

    @property
    def yin_applied(self) -> bool:
        """Backward-Alias für pyin_applied."""
        return self.pyin_applied


class PolyphonicSpeedCurveEstimator:
    """Polyphonic consensus speed curve estimator (Capstan-competitive, §2.12).

    Tracks all simultaneous tonal voices in the mix via BasicPitch (multi-F0 ONNX)
    and derives a robust speed-deviation curve through confidence-weighted median
    consensus across all K voices per frame.

    Advantages over mono pYIN:
    - Percussion/noise confusing a single-voice tracker cannot dominate the result.
    - Multiple voices vote on the speed curve — outliers are suppressed.
    - Works even when individual voices drop out; others continue tracking.

    Algorithm (Klapuri 2003; Salamon & Gómez 2012 MELODIA; Plangent consensus):
    1. BasicPitch[T, K] → K simultaneous pitch tracks (K ≤ 6).
    2. Per track k: global reference pitch via robust median over all voiced frames.
    3. Per frame t, voiced slot k: deviation_cents[t,k] = 1200·log₂(hz/ref_hz[k]).
    4. Per frame t: confidence-weighted median of voiced deviations → speed_curve[t].
       Requires ≥ 2 simultaneously voiced slots; otherwise marked as NaN.
    5. Gap-fill: linear interpolation ≤ 1 s; zero-fill for longer gaps.
    6. Savitzky-Golay smoothing: SG(51, 3) for smoothed speed curve.
    7. Output: virtual pitch trajectory (REF_HZ · 2^(speed_cents/1200)) compatible
       with the existing phase_12 _separate_wow_flutter / _calculate_stretch_factors
       pipeline without any further changes to downstream code.

    Fallback chain:
    - BasicPitch ONNX unavailable / fails → pYIN DSP (via WowFlutterFix).
    """

    _REF_HZ: float = 440.0  # Canonical reference for virtual pitch output
    _MIN_VOICES: int = 2  # Minimum simultaneous voices for consensus
    _SG_WINDOW: int = 51  # Savitzky-Golay window (≈ 0.5 s at 10 ms hop)
    _SG_POLY: int = 3  # Savitzky-Golay polynomial order
    _MIN_HZ: float = 20.0  # Below this: treat slot as unvoiced
    _MIN_CONF: float = 0.20  # Minimum per-slot confidence to include in consensus

    def __init__(self) -> None:
        self._bp = None
        self._init_basicpitch()

    def _init_basicpitch(self) -> None:
        try:
            from plugins.basicpitch_plugin import get_basicpitch_plugin

            self._bp = get_basicpitch_plugin()  # type: ignore[assignment]
            _was_loaded = getattr(self._bp, "_model_loaded", False)
            if _was_loaded:
                logger.debug("PolyphonicSpeedCurveEstimator: BasicPitch bereits geladen (model_geladen=True)")
            else:
                logger.info(
                    "PolyphonicSpeedCurveEstimator: BasicPitch geladen (model_geladen=%s)",
                    _was_loaded,
                )
        except Exception as exc:
            logger.warning("BasicPitch nicht verfügbar (%s) — pYIN-Ersatzpfad aktiv", exc)
            self._bp = None

    def estimate(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Schätzt a speed-deviation curve from polyphonic content.

        Returns a virtual pitch trajectory and confidence array that are
        drop-in compatible with phase_12's _separate_wow_flutter /
        _calculate_stretch_factors pipeline.

        virtual_pitch[t] = REF_HZ · 2^(speed_deviation_cents[t] / 1200)
        where speed_deviation_cents is the zero-centred consensus curve.

        Returns:
            (virtual_pitch, confidence): shape [T], dtype float32.
        """
        if self._bp is None or not getattr(self._bp, "_model_loaded", False):
            return self._pyin_fallback(audio, sr)
        try:
            return self._estimate_polyphonic(audio, sr)
        except Exception as exc:
            logger.warning(
                "PolyphonicSpeedCurveEstimator._estimate_polyphonic fehlgeschlagen (%s) — pYIN-Ersatzpfad",
                exc,
            )
            return self._pyin_fallback(audio, sr)

    def _estimate_polyphonic(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Kern-polyphonic consensus estimation."""

        from scipy.signal import savgol_filter

        if self._bp is None:
            return self._pyin_fallback(audio, sr)

        mono = np.mean(audio, axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
        result = self._bp.analyze(mono, sr, max_polyphony=6)
        pitches_hz: np.ndarray = result.pitches_hz  # [T, K]
        confidences: np.ndarray = result.confidences  # [T, K]

        T, K = pitches_hz.shape
        if T < 4:
            return self._pyin_fallback(audio, sr)

        # Step 2: per-voice global reference pitch (robust median)
        ref_hz = np.zeros(K, dtype=np.float32)
        for k in range(K):
            voiced = pitches_hz[:, k]
            voiced = voiced[voiced > self._MIN_HZ]
            if len(voiced) >= 4:
                ref_hz[k] = float(np.median(voiced))

        # Step 2b: Octave-align reference pitches across voices.
        # BasicPitch may track harmonics (2f, 3f) instead of fundamentals
        # for some voices.  Cluster ref_hz in log space, find the modal
        # octave, then shift outlier voices to the same octave band.
        _valid_refs = ref_hz[ref_hz > self._MIN_HZ]
        if len(_valid_refs) >= 2:
            _log_refs = np.log2(_valid_refs)
            # Compute the median fractional (within-octave) position
            _fracs = _log_refs - np.floor(_log_refs)
            _cluster = float(np.median(_fracs))
            for k in range(K):
                if ref_hz[k] < self._MIN_HZ:
                    continue
                _log_r = np.log2(ref_hz[k])
                _oct_shift = round(_log_r - _cluster)
                if abs(_oct_shift) >= 1:
                    ref_hz[k] = float(ref_hz[k] / (2.0**_oct_shift))

        # Step 3: per-frame per-voice deviation in cents
        deviation_cents = np.zeros((T, K), dtype=np.float32)
        voiced_mask = np.zeros((T, K), dtype=bool)
        for k in range(K):
            if ref_hz[k] < self._MIN_HZ:
                continue
            hz = pitches_hz[:, k]
            valid = (hz > self._MIN_HZ) & (confidences[:, k] > self._MIN_CONF)
            voiced_mask[:, k] = valid
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(valid, hz / ref_hz[k], 1.0)
                ratio = np.clip(ratio, 1e-6, 1e6)
                deviation_cents[:, k] = np.where(valid, 1200.0 * np.log2(ratio), 0.0)

        # Step 3b: Octave-error correction with inter-voice consistency.
        # After ref_hz alignment, per-frame octave errors can still occur
        # (transient harmonics, noise).  For each frame with ≥2 voices:
        # fold large deviations only if folding brings the voice closer
        # to the consensus of the other voices.  Otherwise exclude it.
        _octave_mask = np.abs(deviation_cents) > 600.0
        for t in range(T):
            _active = np.where(voiced_mask[t])[0]
            if len(_active) < 2:
                continue
            for k in _active:
                if not _octave_mask[t, k]:
                    continue
                _folded = ((deviation_cents[t, k] + 600.0) % 1200.0) - 600.0
                _others = [j for j in _active if j != k]
                if len(_others) == 0:
                    deviation_cents[t, k] = _folded
                    continue
                _other_median = float(np.median(deviation_cents[t, _others]))
                _dist_orig = abs(deviation_cents[t, k] - _other_median)
                _dist_folded = abs(_folded - _other_median)
                if _dist_folded < _dist_orig and abs(_folded) < 300.0:
                    deviation_cents[t, k] = _folded
                elif _dist_folded < _dist_orig:
                    # Fold helps but still far off → outlier
                    voiced_mask[t, k] = False
                else:
                    # Fold doesn't help → tracker failure, exclude
                    voiced_mask[t, k] = False

        # Step 3c: Clamp per-voice deviations to ±500 cents before consensus.
        # Values beyond ±500 cents are physically implausible for wow/flutter.
        deviation_cents = np.clip(deviation_cents, -500.0, 500.0)

        # Step 4: confidence-weighted median per frame
        # §Spec 24 Root-Fix (2026-08-17): Degradationsstufen statt Alles-oder-Nichts.
        # Befund: „consensus NEVER reached (0/8184 frames)“ bei realen Songs —
        # BasicPitch liefert pro Frame oft nur EINE dominante Stimme; _MIN_VOICES=2
        # machte den polyphonen Pfad wertlos (pYIN-Ersatz lief immer, das geladene
        # BasicPitch-Modell wurde nie genutzt).
        #   Stufe 1: ≥2 Stimmen → confidence-weighted median (unverändert).
        #   Stufe 2: exakt 1 Stimme mit Konfidenz ≥ 0.50 → Single-Voice-Kurve
        #     (strengeres Gate; Konfidenz 0.60 statt 0.85 — downstream bleibt
        #     dadurch vorsichtiger).
        #   Stufe 3: keine stimmhaften Frames → pYIN-Ersatzpfad.
        speed_curve = np.full(T, np.nan, dtype=np.float32)
        consensus_conf = np.zeros(T, dtype=np.float32)
        n_multi = 0
        n_single = 0
        for t in range(T):
            active = np.where(voiced_mask[t])[0]
            if len(active) >= self._MIN_VOICES:
                devs = deviation_cents[t, active]
                wgts = confidences[t, active]
                wgts = wgts / (wgts.sum() + 1e-10)
                speed_curve[t] = self._weighted_median(devs, wgts)
                consensus_conf[t] = 1.0
                n_multi += 1
            elif len(active) == 1 and float(confidences[t, active[0]]) >= 0.50:
                speed_curve[t] = float(deviation_cents[t, active[0]])
                consensus_conf[t] = 0.60
                n_single += 1

        # Step 5: fill NaN gaps
        nan_mask = np.isnan(speed_curve)
        if nan_mask.all():
            # §Fix: zero real per-frame consensus was EVER achieved — weder
            # multi- noch single-voice (Stufe 1+2 leer). _fill_gaps() würde
            # zero-fillen („0 cents deviation“) — eine falsch-plausible Lesart,
            # die den Step-6b-Guard unterläuft. Ohne echten Datenpunkt ist die
            # Kurve definitiongemäß unvertrauenswürdig → pYIN-Fallback.
            logger.info(
                "PolyphonicSpeedCurveEstimator: keine stimmhaften Frames (multi=0, single=0 von %d) — pYIN-Ersatzpfad",
                T,
            )
            return self._pyin_fallback(audio, sr)
        if n_multi == 0 and n_single > 0:
            logger.info(
                "PolyphonicSpeedCurveEstimator: multi-Voice-Konsens nicht erreichbar — Single-Voice-Speed-Curve (%d/%d Frames, Stufe 2)",
                n_single,
                T,
            )
        if nan_mask.any():
            speed_curve = self._fill_gaps(speed_curve, result.frame_times_s)

        # Step 6: Savitzky-Golay smoothing
        sg_win = self._SG_WINDOW
        if len(speed_curve) > sg_win:
            speed_curve = savgol_filter(speed_curve, sg_win, self._SG_POLY).astype(np.float32)

        speed_curve = np.nan_to_num(speed_curve, nan=0.0).astype(np.float32)

        # Step 6b: Plausibility guard — speed deviations > 200 cents (2 semitones)
        # are physically implausible for wow/flutter and indicate inference failure.
        # Use a deterministic pYIN fallback instead of forcing a zero-curve to avoid
        # masking hard tracker failures and to preserve musical timing information.
        # Ana­logue to the Peak-Guard rule (spec §Verboten): use 99th percentile
        # instead of max — a single crackle/impulse frame after SG-smoothing must
        # not trigger a full fallback for a crackle-heavy vinyl recording.
        _max_abs_cents = float(np.percentile(np.abs(speed_curve), 99.0)) if len(speed_curve) > 0 else 0.0
        if _max_abs_cents > 200.0:
            logger.info(
                "PolyphonicSpeedCurveEstimator: speed_range implausible (max |%.1f| cents > 200) — switching to pYIN Ersatzpfad",
                _max_abs_cents,
            )
            return self._pyin_fallback(audio, sr)
        else:
            final_conf = None  # computed below

        # Step 7: virtual pitch trajectory (phase_12 pipeline compatible)
        virtual_pitch = (self._REF_HZ * np.power(2.0, speed_curve / 1200.0)).astype(np.float32)
        virtual_pitch = np.clip(virtual_pitch, self._MIN_HZ, 4000.0)

        if final_conf is None:
            # multi-Voice-Konsens: 0.85; Single-Voice (Stufe 2): 0.60; unvoiced: 0.30
            final_conf = np.where(consensus_conf >= 1.0, 0.85, 0.60).astype(np.float32)
            final_conf[consensus_conf <= 0.0] = 0.30
            final_conf[nan_mask] *= 0.5

        logger.info(
            "PolyphonicSpeedCurveEstimator: T=%d frames, K=%d voices, "
            "consensus=%d/%d frames, speed_range=[%.2f, %.2f] cents",
            T,
            K,
            int(np.sum(~nan_mask)),
            T,
            float(np.min(speed_curve)),
            float(np.max(speed_curve)),
        )
        return virtual_pitch, final_conf

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Confidence-weighted median (1D robust estimator)."""
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        idx = np.argsort(values)
        vals_s = values[idx]
        wgts_s = weights[idx]
        cum = np.cumsum(wgts_s)
        total = cum[-1]
        if total < 1e-10:
            return float(np.mean(values))
        mid = np.searchsorted(cum, 0.5 * total)
        return float(vals_s[min(mid, len(vals_s) - 1)])

    @staticmethod
    def _fill_gaps(speed_curve: np.ndarray, frame_times_s: np.ndarray) -> np.ndarray:
        """Fill NaN gaps: linear interpolation ≤ 1 s, zero-fill for longer spans."""
        frame_step = float(frame_times_s[1] - frame_times_s[0]) if len(frame_times_s) > 1 else 0.01
        max_interp = int(1.0 / max(frame_step, 1e-4))
        result = speed_curve.copy()
        T = len(result)
        i = 0
        while i < T:
            if np.isnan(result[i]):
                j = i
                while j < T and np.isnan(result[j]):
                    j += 1
                v_before = result[i - 1] if i > 0 else 0.0
                v_after = result[j] if j < T else 0.0
                if j - i <= max_interp:
                    result[i:j] = np.linspace(v_before, v_after, j - i)
                else:
                    result[i:j] = 0.0
                i = j
            else:
                i += 1
        return result

    def _pyin_fallback(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """pYIN DSP fallback via WowFlutterFix._estimate_pitch_pyin."""
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        mono = np.mean(audio, axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
        return WowFlutterFix()._estimate_pitch_pyin(mono, sr)


class HybridWowFlutter:
    """
    Hybrid Wow/Flutter Detection: YIN + CREPE.

    Combines fast YIN DSP detection with high-accuracy CREPE ML refinement.
    Adaptive strategy selects optimal detection based on signal characteristics.
    """

    def __init__(self, config: WowFlutterConfig | None = None) -> None:
        """
        Initialisiert hybrid wow/flutter detector.

        Args:
            config: Wow/Flutter configuration
        """
        self.config = config or WowFlutterConfig()

        # Lazy-load ML pitch plugin chain
        self.crepe = None
        if self.config.strategy in [
            PitchDetectionStrategy.CREPE_ONLY,
            PitchDetectionStrategy.HYBRID,
            PitchDetectionStrategy.ADAPTIVE,
        ]:
            self._init_crepe()

    def _init_crepe(self) -> None:
        """Initialisiert pitch plugin: FCPE -> RMVPE -> PESTO -> pYIN (§4.4 Spec).

        Order: Tier-1 FCPE, Tier-2 RMVPE (Wei et al. ICASSP 2023, ~30 % lower pitch
        error for vocals), Tier-3 PESTO, Tier-4 pYIN (DSP-Endfall, §4.4).
        CREPE als Produktions-Tier VERBOTEN (Spec 04, Z. 1129) — Rev. 2026-08-16.
        VERBOTEN: FCPE -> CREPE -> RMVPE (RMVPE muss vor CREPE stehen — §4.4).
        """
        try:
            from plugins.fcpe_plugin import get_fcpe_plugin

            self.crepe = get_fcpe_plugin()  # type: ignore[assignment]
            logger.info("FCPE pitch plugin geladen for wow/flutter detection (model=%s)", self.crepe.model_used)  # type: ignore[attr-defined]
            return
        except Exception as e:
            logger.debug("FCPE-Plugin nicht verfügbar (%s) — RMVPE-Ersatzpfad (§4.4 Tier-2)", e)
        # Tier-2: RMVPE — before CREPE per §4.4 (30 % lower pitch error, Wei ICASSP 2023)
        try:
            from plugins.rmvpe_plugin import get_rmvpe_plugin

            self.crepe = get_rmvpe_plugin()  # type: ignore[assignment]
            logger.info("RMVPE plugin geladen für wow/flutter-Detektion (§4.4 Tier-2)")
            return
        except Exception as e:
            logger.debug("RMVPE nicht verfügbar (%s) — CREPE-Ersatzpfad (§4.4 Tier-3)", e)
        # Tier-3: PESTO
        try:
            from plugins.pesto_plugin import get_pesto_plugin  # pylint: disable=no-name-in-module

            self.crepe = get_pesto_plugin()  # type: ignore[assignment]
            logger.info("PESTO plugin geladen für wow/flutter-Detektion (§4.4 Tier-3)")
            return
        except Exception as e:
            logger.debug("PESTO nicht verfügbar (%s) — pYIN-DSP-Ersatzpfad (§4.4 Tier-4)", e)
        # Tier-4: pYIN (DSP-Endfall, §4.4). CREPE als Produktions-Tier VERBOTEN
        # (Spec 04, Z. 1129) — Rev. 2026-08-16.
        logger.warning("Kein Pitch-ML-Plugin verfügbar — pYIN-DSP-Ersatzpfad (§V6)")
        try:
            from backend.core.fallback_auditor import get_fallback_auditor

            get_fallback_auditor().record("PitchDetection", "fcpe_rmvpe_pesto", "pyin_dsp", "all_ml_unavailable")
        except Exception:
            logger.debug("FallbackAuditor nicht verfügbar (unkritisch)", exc_info=True)
        self.crepe = None

    def detect_pitch(self, audio: np.ndarray, sample_rate: int = 48000) -> WowFlutterResult:
        """
        Pitch-Trajektorie via hybrides pYIN + CREPE detektieren.

        Args:
            audio: Eingangs-Audio (mono oder stereo)
            sample_rate: Abtastrate in Hz

        Returns:
            WowFlutterResult mit Pitch-Trajektorie und Metadaten
        """
        import time

        start_time = time.time()

        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        strategy = self._determine_strategy(audio, sample_rate)

        pyin_applied = False
        crepe_applied = False
        metadata = {}
        pitch_trajectory: np.ndarray = np.array([])
        confidence: np.ndarray = np.array([])

        # Stufe 1: pYIN-Detektion (Mauch & Dixon 2014) — §4.2 konform
        if strategy in [PitchDetectionStrategy.PYIN_ONLY, PitchDetectionStrategy.HYBRID]:
            logger.info("Stufe 1: pYIN-Pitch-Detektion (Mauch & Dixon 2014)...")
            pitch_pyin, confidence_pyin = self._apply_pyin(audio, sample_rate)
            pyin_applied = True
            # Direkt als Basis setzen (wird von CREPE ggf. überschrieben)
            pitch_trajectory = pitch_pyin
            confidence = confidence_pyin
            valid_pyin = confidence_pyin[confidence_pyin > 0]
            metadata["pyin"] = {
                "mean_confidence": float(np.mean(valid_pyin)) if len(valid_pyin) > 0 else 0.0,
                "num_estimates": int(np.sum(pitch_pyin > 0)),
            }

            mean_confidence = float(np.mean(valid_pyin)) if len(valid_pyin) > 0 else 0.0
            logger.info("pYIN abgeschlossen: mean confidence=%.3f", mean_confidence)

            # §v10.18: CREPE-Skip entfernt. pYIN confidence (voicing probability) ist
            # kein verlässlicher Indikator für Pitch-Genauigkeit — bei voiced frames
            # meldet pYIN fast immer ~1.0, auch wenn die F0-Schätzung ungenau ist.
            # CREPE muss in HYBRID immer laufen, die Blend-Logik entscheidet.
            # (Mauch & Dixon 2014: pYIN confidence = voicing probability ≠ pitch accuracy)

        # Stufe 2: ML-Verfeinerung (RMVPE/PESTO/CREPE; falls nötig)
        if strategy in [PitchDetectionStrategy.CREPE_ONLY, PitchDetectionStrategy.HYBRID]:
            if self.crepe is not None:
                logger.info("Stufe 2: ML-Pitch-Detektion (RMVPE/PESTO/CREPE)...")
                pitch_crepe, confidence_crepe = self._apply_crepe(audio, sample_rate)
                crepe_applied = True
                valid_crepe = confidence_crepe[confidence_crepe > 0]
                metadata["crepe"] = {
                    "mean_confidence": float(np.mean(valid_crepe)) if len(valid_crepe) > 0 else 0.0,
                    "num_estimates": int(np.sum(pitch_crepe > 0)),
                    "model": getattr(self.crepe, "model_used", self.config.crepe_model),
                }

                mean_confidence_crepe = float(np.mean(valid_crepe)) if len(valid_crepe) > 0 else 0.0
                logger.info("ML-Stufe abgeschlossen: mean confidence=%.3f", mean_confidence_crepe)

                pitch_trajectory = pitch_crepe
                confidence = confidence_crepe

                if strategy == PitchDetectionStrategy.HYBRID and pyin_applied:
                    logger.info("pYIN + CREPE Ergebnisse werden gemischt...")
                    pitch_trajectory, confidence = self._blend_pitch_estimates(
                        pitch_pyin, confidence_pyin, pitch_crepe, confidence_crepe
                    )
            else:
                logger.warning("ML-Pitch-Plugin nicht verfügbar, nutze pYIN-Ergebnis")
                if not pyin_applied:
                    pitch_trajectory, confidence = self._apply_pyin(audio, sample_rate)
                    pyin_applied = True

        # Sicherheitsnetz: pYIN Fallback
        if not pyin_applied and not crepe_applied:
            pitch_trajectory, confidence = self._apply_pyin(audio, sample_rate)
            pyin_applied = True

        processing_time = time.time() - start_time
        valid = confidence[confidence > 0]
        mean_confidence = float(np.mean(valid)) if len(valid) > 0 else 0.0
        metadata["processing_time"] = processing_time  # type: ignore[assignment]

        return WowFlutterResult(
            pitch_trajectory=pitch_trajectory,
            confidence=confidence,
            strategy_used=strategy,
            pyin_applied=pyin_applied,
            crepe_applied=crepe_applied,
            processing_time=processing_time,
            mean_confidence=mean_confidence,
            metadata=metadata,
        )

    def _determine_strategy(self, audio: np.ndarray, sample_rate: int) -> PitchDetectionStrategy:
        """Optimale Pitch-Detektions-Strategie bestimmen."""
        if self.config.strategy != PitchDetectionStrategy.ADAPTIVE:
            return self.config.strategy

        if self.crepe is not None:
            logger.info("Adaptiv: CREPE verfügbar, nutze HYBRID-Modus")
            return PitchDetectionStrategy.HYBRID
        else:
            logger.info("Adaptiv: CREPE nicht verfügbar, nutze PYIN_ONLY-Modus")
            return PitchDetectionStrategy.PYIN_ONLY

    def _apply_pyin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        pYIN-Pitch-Detektion via Phase 12 (Mauch & Dixon 2014).

        Delegiert an WowFlutterFix._estimate_pitch_yin() welche intern
        _estimate_pitch_pyin() via librosa.pyin aufruft.

        Args:
            audio: Mono-Audio
            sample_rate: Abtastrate

        Returns:
            (pitch_trajectory, confidence) als np.ndarray
        """
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        phase = WowFlutterFix()
        pitch_trajectory, confidence = phase._estimate_pitch_yin(audio, sample_rate)
        return pitch_trajectory, confidence

    # Backward-Compat Alias
    def _apply_yin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Backward-Compat: delegiert an _apply_pyin."""
        return self._apply_pyin(audio, sample_rate)

    def _apply_crepe(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Wendet an: FCPE/CREPE ML pitch detection (numpy-API, kein Subprocess)."""
        if self.crepe is None:
            return self._apply_pyin(audio, sample_rate)
        try:
            result = self.crepe.analyze(audio, sample_rate)
            # CrepeResult.f0_hz / .voiced_prob sind die finalen Arrays
            f0 = np.nan_to_num(result.f0_hz.astype(np.float32))
            conf = np.clip(np.nan_to_num(result.voiced_prob.astype(np.float32)), 0.0, 1.0)
            return f0, conf
        except Exception as exc:
            logger.warning("FCPE/CREPE Pitch-Inferenz fehlgeschlagen (%s) — pYIN Ersatzpfad", exc)
            return self._apply_pyin(audio, sample_rate)

    def _blend_pitch_estimates(
        self, pitch_pyin: np.ndarray, conf_pyin: np.ndarray, pitch_crepe: np.ndarray, conf_crepe: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        pYIN- und CREPE-Pitch-Schätzwerte konfidenzgewichtet mischen.

        Strategie:
        - CREPE bei hoher Konfidenz (> 0.8)
        - pYIN bei niedriger CREPE-Konfidenz oder fehlenden Schätzwerten
        - Gewichtetes Mischen in unsicheren Regionen
        """
        if len(pitch_crepe) != len(pitch_pyin):
            from typing import cast

            from scipy import signal as sp_signal

            pitch_crepe = cast(np.ndarray, sp_signal.resample(pitch_crepe, len(pitch_pyin)))
            conf_crepe = cast(np.ndarray, sp_signal.resample(conf_crepe, len(pitch_pyin)))

        blended_pitch = np.zeros_like(pitch_pyin)
        blended_conf = np.zeros_like(conf_pyin)

        for i in range(len(pitch_pyin)):
            if conf_crepe[i] > 0.8:
                blended_pitch[i] = pitch_crepe[i]
                blended_conf[i] = conf_crepe[i]
            elif conf_pyin[i] > conf_crepe[i]:
                blended_pitch[i] = pitch_pyin[i]
                blended_conf[i] = conf_pyin[i]
            else:
                total = conf_crepe[i] + conf_pyin[i] + 1e-10
                w_crepe = conf_crepe[i] / total
                w_pyin = 1.0 - w_crepe
                blended_pitch[i] = w_crepe * pitch_crepe[i] + w_pyin * pitch_pyin[i]
                blended_conf[i] = max(conf_crepe[i], conf_pyin[i])

        return blended_pitch, blended_conf


if __name__ == "__main__":
    """Test hybrid wow/flutter detection."""

    logger.debug("=" * 80)
    logger.debug("Hybrid Wow/Flutter Detection Test")
    logger.debug("=" * 80)

    # Generate test audio with pitch variation (simulated wow/flutter)
    duration = 5.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base frequency (440 Hz A4)
    base_freq = 440.0

    # Add simulated wow (slow pitch drift, <4 Hz)
    wow_freq = 2.0  # 2 Hz wow
    wow_amount = 0.02  # 2% pitch variation
    pitch_variation = 1.0 + wow_amount * np.sin(2 * np.pi * wow_freq * t)

    # Generate audio with pitch variation
    phase = np.cumsum(2 * np.pi * base_freq * pitch_variation / sample_rate)
    audio = 0.5 * np.sin(phase)

    logger.debug("erzeugt %ss test audio @ %s Hz", duration, sample_rate)
    logger.debug("Base frequency: %s Hz with %.1f%% wow at %s Hz", base_freq, wow_amount * 100, wow_freq)
    logger.debug("")

    # Test strategies
    strategies = [
        (PitchDetectionStrategy.PYIN_ONLY, "pYIN Only (Mauch & Dixon 2014)"),
        (PitchDetectionStrategy.HYBRID, "Hybrid (pYIN + CREPE)"),
    ]

    for strategy, name in strategies:
        logger.debug("-" * 80)
        logger.debug("Strategy: %s", name)
        logger.debug("-" * 80)

        config = WowFlutterConfig(strategy=strategy)
        detector = HybridWowFlutter(config)

        result = detector.detect_pitch(audio, sample_rate)

        logger.debug("✅ Strategy used: %s", result.strategy_used.value)
        logger.debug("   pYIN angewendet: %s", result.pyin_applied)
        logger.debug("   CREPE angewendet: %s", result.crepe_applied)
        logger.debug("   Mean confidence: %.3f", result.mean_confidence)
        logger.debug("   Pitch estimates: %s", len(result.pitch_trajectory[result.pitch_trajectory > 0]))
        logger.debug("   Processing time: %.2fs", result.processing_time)
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test vollstaendig")
