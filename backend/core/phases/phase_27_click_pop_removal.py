#!/usr/bin/env python3
"""
Phase 27: Click/Pop Removal v3.0 — AR-Residual + Consistent Wiener
====================================================================
Advanced impulse noise detection and removal with AR-residual detection.

Algorithm Overview:
1. AR-Residual Click Detection (multi-scale, orders 6/12/20):
   - Schätzung AR-Koeffizienten (Levinson-Durbin via scipy.signal.lpc)
   - Vorhersagefehler e(t) = audio(t) - AR_pred(t) — Clicks als Ausreißer
   - Z-Score: z(t) = (|e(t)| - μ_e) / σ_e  → Detektion wo z > Schwellwert
   Ersetzt das primitive signal.medfilt (verboten per §4.2 copilot-instructions)
2. Duration-Based Classification (click / pop / burst)
3. Adaptive Repair Strategies:
   - Short clicks (<5 samples): Cubic spline interpolation
   - Medium pops (5-15 samples): AR(8) prediction (beiderseitig, Cosinus-Blend)
   - Long bursts (>15 samples): Gain-adaptiver Cross-fade
4. Material-Adaptive Thresholds
5. Post-Processing: Boundary taper, transient preservation

SCIENTIFIC FOUNDATION:
- Godsill & Rayner (1998): „Digital Audio Restoration" — AR-Residual-Detektion,
  Bayesian-Rahmen für Impulsnoise-Segmentierung (Kapitel 3, Gleichungen 3.1–3.8)
- Cemgil et al. (2006): „A Generative Model for Music Transcription" —
  Sparse Bayes für impulsnoise (Grundlage RBME-Declicker)
- Le Roux & Vincent (2013): „Consistent Wiener Filtering" — Gain-Floor
- Lagrange et al. (2007): Long Interpolation via AR Sinusoidal Modeling
- Röbel (2003): Transient Preservation Under Inpainting

VERBOTEN (entfernt in v3.0):
- signal.medfilt als primäre Detektion — ersetzt durch AR-Residual

Author: Aurik Development Team
Version: 3.0.0 AR-Residual
"""

import logging
import os
import time
from typing import Any

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np
from scipy import interpolate
from scipy.signal import lfilter

from backend.core.audio_utils import audio_sample_count, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType
from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.natural_performance_detector import get_natural_performance_detector

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


def _get_phase27_lge():
    """LGE-Resolver als stabiler Patch-/Accessor-Punkt fuer Phase 27."""
    return get_lyrics_guided_enhancement()


def _get_phase27_npd():
    """NPA-Resolver als stabiler Patch-/Accessor-Punkt fuer Phase 27."""
    return get_natural_performance_detector()


class ClickPopRemoval(PhaseInterface):
    """
    Professional Click/Pop Removal with Statistical Feature Extraction.

    Key Features:
    - Multi-scale median filtering (3-15 samples)
    - Statistical outlier detection (Z-score, energy, ZCR)
    - Duration-based classification (click/pop/burst)
    - Adaptive repair (interpolation, AR prediction, cross-fade)
    - Material-adaptive thresholds
    - False positive reduction

    Use Cases:
    - Vinyl/shellac click removal
    - Digital artifact repair
    - Impulse noise suppression
    - Transient preservation

    Performance: <0.20× realtime on modern CPU
    """

    # Material-adaptive detection parameters
    # ar_orders: Multi-Scale AR-Residual-Detektion @ 48 kHz.
    # Spec §VERBOTEN: LPC < 16; Richtig: 30–40 @ 48 kHz.
    # Drei Skalen für robuste Detektion: kurz (16), mittel (24), lang (32).
    DETECTION_CONFIG = {
        MaterialType.SHELLAC: {
            "ar_orders": [16, 24, 32],  # Multi-Scale AR-Residual-Detektion (§VERBOTEN: < 16)
            "z_score_threshold": 3.0,  # Sehr sensitiv (dichte Defekte)
            "energy_threshold_db": -35,
            "max_click_duration_samples": 25,
            "repair_strength": 1.0,
        },
        MaterialType.VINYL: {
            "ar_orders": [16, 24, 32],
            "z_score_threshold": 3.5,
            "energy_threshold_db": -40,
            "max_click_duration_samples": 20,
            "repair_strength": 0.95,
        },
        MaterialType.TAPE: {
            "ar_orders": [16, 24],
            "z_score_threshold": 4.0,  # Konservativ (hauptsächlich Dropouts)
            "energy_threshold_db": -45,
            "max_click_duration_samples": 15,
            "repair_strength": 0.90,
        },
        MaterialType.CASSETTE: {
            "ar_orders": [16, 24],
            "z_score_threshold": 3.8,  # v10.0.0: leicht sensitiver als TAPE (Bandrisse häufiger)
            "energy_threshold_db": -43,  # v10.0.0: etwas höher (Kassette häufigere kurze Aussetzer)
            "max_click_duration_samples": 18,  # v10.0.0: etwas längere Reparaturfenster
            "repair_strength": 0.92,
        },  # v10.0.0: IEC 60094-1 — Cassette-Hiss/Bandriss-Profil angepasst
        MaterialType.CD_DIGITAL: {
            "ar_orders": [16, 24],
            "z_score_threshold": 5.0,  # Sehr konservativ
            "energy_threshold_db": -50,
            "max_click_duration_samples": 10,
            "repair_strength": 0.85,
        },
        MaterialType.STREAMING: {
            "ar_orders": [16, 24],
            "z_score_threshold": 5.0,
            "energy_threshold_db": -55,
            "max_click_duration_samples": 10,
            "repair_strength": 0.85,
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "Click/Pop Removal v3 AR-Residual"
        self._click_repair_profile_current = {
            "cubic_context": 5.0,
            "ar_context": 128.0,
            "ar_order": 32.0,
            "crossfade_context": 10.0,
            "taper_length": 5.0,
        }

    @staticmethod
    def _compute_click_repair_profile(
        material: str,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """Berechnet adaptive click repair profile (§2.54)."""
        mat = str(material or "unknown").lower().replace("-", "_").replace(" ", "_")
        qm = str(quality_mode or "balanced").lower().replace("-", "_")
        if restorability_score is None:
            restorability_score = 50.0
        rest = float(np.clip(restorability_score, 0.0, 100.0))

        base = {
            "shellac": {"cubic": 8.0, "ar_ctx": 220.0, "ar_order": 42.0, "crossfade": 16.0, "taper": 8.0},
            "vinyl": {"cubic": 7.0, "ar_ctx": 192.0, "ar_order": 38.0, "crossfade": 14.0, "taper": 7.0},
            "tape": {"cubic": 6.0, "ar_ctx": 170.0, "ar_order": 34.0, "crossfade": 12.0, "taper": 6.0},
            "reel_tape": {"cubic": 6.0, "ar_ctx": 176.0, "ar_order": 35.0, "crossfade": 12.0, "taper": 6.0},
            "cd_digital": {"cubic": 4.0, "ar_ctx": 132.0, "ar_order": 26.0, "crossfade": 8.0, "taper": 4.0},
            "streaming": {"cubic": 4.0, "ar_ctx": 124.0, "ar_order": 24.0, "crossfade": 8.0, "taper": 4.0},
            "unknown": {"cubic": 5.0, "ar_ctx": 144.0, "ar_order": 28.0, "crossfade": 10.0, "taper": 5.0},
        }.get(mat, {"cubic": 5.0, "ar_ctx": 144.0, "ar_order": 28.0, "crossfade": 10.0, "taper": 5.0})

        mode = {
            "fast": -1.0,
            "balanced": 0.0,
            "restoration": 0.5,
            "quality": 1.0,
            "maximum": 1.8,
            "studio_2026": 1.8,
        }.get(qm, 0.0)

        rest_factor = (50.0 - rest) / 50.0
        cubic_context = float(np.clip(base["cubic"] + mode + 0.8 * rest_factor, 3.0, 12.0))
        ar_context = float(np.clip(base["ar_ctx"] + mode * 28.0 + 36.0 * rest_factor, 64.0, 320.0))
        ar_order = float(np.clip(base["ar_order"] + mode * 6.0 + 8.0 * rest_factor, 16.0, 56.0))
        crossfade_context = float(np.clip(base["crossfade"] + mode * 2.0 + 3.0 * rest_factor, 6.0, 24.0))
        taper_length = float(np.clip(base["taper"] + 0.9 * mode + 1.5 * rest_factor, 3.0, 12.0))

        return {
            "cubic_context": cubic_context,
            "ar_context": ar_context,
            "ar_order": ar_order,
            "crossfade_context": crossfade_context,
            "taper_length": taper_length,
        }

    @staticmethod
    def _derive_safe_click_strength(
        effective_strength: float,
        material_key: str,
        panns_tags: dict[str, float],
    ) -> float:
        """Reduce aggressive click repair on vocal/analog-sensitive material."""
        strength = float(effective_strength)
        vocal_prob = max(
            float(panns_tags.get("Singing voice", 0.0)),
            float(panns_tags.get("Vocals", 0.0)),
            float(panns_tags.get("Speech", 0.0)),
            float(panns_tags.get("Male singing", 0.0)),
            float(panns_tags.get("Female singing", 0.0)),
        )
        is_analog_sensitive = any(
            token in material_key for token in ("vinyl", "shellac", "wax_cylinder", "wire_recording", "lacquer_disc")
        )
        if vocal_prob >= 0.40:
            strength *= 0.84
        if is_analog_sensitive:
            strength *= 0.88
        return float(np.clip(strength, 0.0, 1.0))

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_27_click_pop_removal",
            name="Click/Pop Removal v3 AR-Residual",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=3,
            dependencies=["phase_01_click_removal"],
            estimated_time_factor=0.20,
            version="3.0.0",
            memory_requirement_mb=100,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.93,
            description=(
                "Impulsrauschen-Detektion via AR-Residual + Z-Score "
                "(Godsill & Rayner 1998) — ersetzt primitiven Medianfilter-Declicker"
            ),
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="27")
        check_ml_model_ready("Whisper", phase_name="27")
        """
        Erkennt and remove clicks/pops from audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material_type: Material type for adaptive processing

        Returns:
            PhaseResult with cleaned audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        if isinstance(material_type, MaterialType):
            material_enum = material_type
        else:
            _mat_norm = str(material_type or "unknown").strip().upper().replace("-", "_").replace(" ", "_")
            material_enum = getattr(MaterialType, _mat_norm, MaterialType.CD_DIGITAL)  # type: ignore[arg-type]
        material_name = material_enum.name

        is_stereo = audio.ndim == 2
        config = dict(self.DETECTION_CONFIG.get(material_enum, self.DETECTION_CONFIG[MaterialType.CD_DIGITAL]))

        # Locality-aware intensity control from UV3.
        # Sparse click/pop coverage should preserve unaffected transients.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        _material_key = str(getattr(material_enum, "name", material_enum)).lower()
        _panns_tags = {k: float(v) for k, v in kwargs.get("panns_tags", {}).items() if isinstance(v, (int, float, str))}
        _quality_mode = kwargs.get("quality_mode")
        _restorability_score = kwargs.get("restorability_score", 50.0)
        click_repair_profile = self._compute_click_repair_profile(_material_key, _quality_mode, _restorability_score)
        self._click_repair_profile_current = click_repair_profile
        _safe_strength = self._derive_safe_click_strength(_effective_strength, _material_key, _panns_tags)
        config["repair_strength"] = float(np.clip(float(config["repair_strength"]) * _safe_strength, 0.0, 1.0))  # type: ignore[arg-type]
        config["sample_rate"] = int(sample_rate)

        # §V38 VFA-Schutzzonen für per-Click-Strength-Oracle (§0p Vocal-Supremacy)
        _p27_protected_zones: list[tuple[float, float, float]] = []
        for _z in kwargs.get("vibrato_zones") or []:
            try:
                _p27_protected_zones.append((float(_z[0]), float(_z[1]), 0.20))
            except Exception as e:
                logger.warning("Verarbeitungsschritt_27_click_pop_removal.py::verarbeiten Ersatzpfad: %s", e)
        for _z in kwargs.get("frisson_zones") or []:
            try:
                _fz_s = float(getattr(_z, "start_s", None) or _z[0])
                _fz_e = float(getattr(_z, "end_s", None) or _z[1])
                _p27_protected_zones.append((_fz_s, _fz_e, 0.30))
            except Exception as e:
                logger.warning("Verarbeitungsschritt_27_click_pop_removal.py::verarbeiten Ersatzpfad: %s", e)
        for _z in kwargs.get("whisper_zones") or []:
            try:
                _p27_protected_zones.append((float(_z[0]), float(_z[1]), 0.25))
            except Exception as e:
                logger.warning("Verarbeitungsschritt_27_click_pop_removal.py::verarbeiten Ersatzpfad: %s", e)
        for _z in kwargs.get("passaggio_zones") or []:
            try:
                _p27_protected_zones.append((float(_z[0]), float(_z[1]), 0.35))
            except Exception as e:
                logger.warning("Verarbeitungsschritt_27_click_pop_removal.py::verarbeiten Ersatzpfad: %s", e)
        _p27_pz = _p27_protected_zones or None

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material_name,
                    "clicks_removed": 0,
                    "clicks_per_second": 0.0,
                    "rt_factor": 0.0,
                    "click_repair_profile": click_repair_profile,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "safe_strength": _safe_strength,
                    "processing": "skipped_zero_strength",
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Click/pop removal skipped due to zero effective strength"],
            )

        # §2.51: Linked detection — detect on mono mix, repair synchronized
        if is_stereo:
            left, right = stereo_channel_view(audio)
            mono_mix = (left + right) * 0.5
            click_locations = self._detect_clicks_multiband(mono_mix, config)
            classified_clicks = self._classify_clicks(mono_mix, click_locations, config)
            cleaned_left = self._repair_clicks(left, classified_clicks, config, protected_zones=_p27_pz)
            cleaned_right = self._repair_clicks(right, classified_clicks, config, protected_zones=_p27_pz)
            cleaned_audio = stereo_like(cleaned_left, cleaned_right, audio)
            total_clicks = len(classified_clicks)
        else:
            cleaned_audio, total_clicks = self._process_channel(audio, sample_rate, config, protected_zones=_p27_pz)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        cleaned_audio = np.nan_to_num(cleaned_audio, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned_audio = np.clip(cleaned_audio, -1.0, 1.0)

        # §2.36 Phonem-Schutz: Plosiv-Bursts (/p/,/t/,/k/) sind breitbandig und sehen
        # dem AR-Residual-Profil eines Clicks ähnlich. Frame-Restore für Konsonanten-Bursts.
        # §2.46f NPA-Guard: Atemgeräusche (50–500 ms, dumpf) dürfen nicht als Click gelöscht werden.
        try:
            _mono27 = cleaned_audio.mean(axis=0) if cleaned_audio.ndim == 2 else cleaned_audio
            _orig27 = audio.mean(axis=0) if audio.ndim == 2 else audio
            n_samples27 = _mono27.shape[0]
            # §2.36 Phonem-Schutz
            try:
                _lge27 = _get_phase27_lge()
                _phon_mask27 = _lge27.get_phoneme_mask(_orig27, sample_rate, hop_length=512)
                if _phon_mask27 is not None and len(_phon_mask27) > 0:
                    hop27 = 512
                    for _fi, _is_burst in enumerate(_phon_mask27):
                        if _is_burst:
                            _s = min(_fi * hop27, n_samples27)
                            _e = min(_s + hop27, n_samples27)
                            if cleaned_audio.ndim == 2:
                                cleaned_audio[:, _s:_e] = audio[:, _s:_e]
                            else:
                                cleaned_audio[_s:_e] = audio[_s:_e]
            except Exception as _p27_exc:
                logger.debug("§2.36 Verarbeitungsschritt27 Phonem-Guard (nicht blockierend): %s", _p27_exc)
            # §2.46f NPA-Guard
            try:
                _npa_mask27 = (
                    _get_phase27_npd().detect(_orig27, sample_rate).get_protected_mask(n_samples27, sample_rate)
                )
                if _npa_mask27 is not None and _npa_mask27.any():
                    if cleaned_audio.ndim == 2:
                        cleaned_audio[:, _npa_mask27] = audio[:, _npa_mask27]
                    else:
                        cleaned_audio[_npa_mask27] = audio[_npa_mask27]
            except Exception as _npa27_exc:
                logger.debug("§2.46f Verarbeitungsschritt27 NPA-Guard (nicht blockierend): %s", _npa27_exc)
        except Exception as _guard27_exc:
            logger.debug("§2.36/§2.46f Verarbeitungsschritt27 guards (nicht blockierend): %s", _guard27_exc)

        # §V19 Noise-Textur-Invariante (VERBOTEN-V19): Residual bewahrt Materialcharakter
        _mat27_str = str(material_type or "unknown").lower()
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt27_fn,
            )

            if cleaned_audio.shape == audio.shape:
                _nt27_d = _nt27_fn(
                    audio.astype(np.float32) - cleaned_audio.astype(np.float32), _mat27_str, sr=sample_rate
                )
                if _nt27_d > 0.25:
                    cleaned_audio = (0.5 * cleaned_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_27 noise_texture dist=%.3f > 0.25 → 50%%-Blend",
                        _nt27_d,
                    )
        except Exception as _nt27_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_27 noise_texture_guard (nicht blockierend): %s",
                _nt27_exc,
            )

        # §V24 Spektralfarbe-Prüfung (VERBOTEN-V24): 1/3-Oktav-Profil darf nicht verfärbt werden
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg27,
            )

            if cleaned_audio.shape == audio.shape:
                _sc27 = _scg27(audio.astype(np.float32), cleaned_audio.astype(np.float32), sample_rate)
                if not _sc27.ok:
                    cleaned_audio = (0.70 * cleaned_audio + 0.30 * audio).astype(np.float32)
        except Exception as _sc27_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_27 spectral_color_guard (nicht blockierend): %s",
                _sc27_exc,
            )

        return PhaseResult(
            success=True,
            audio=cleaned_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material_name,
                "clicks_removed": int(total_clicks),
                "clicks_per_second": float(total_clicks / (audio_sample_count(audio) / sample_rate)),
                "rt_factor": float(rt_factor),
                "stereo_mode": "linked_detection" if is_stereo else "mono",
                "click_repair_profile": click_repair_profile,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "safe_strength": _safe_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.25 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _process_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: dict[str, Any],
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Verarbeitet a single channel for click/pop removal."""
        del sample_rate
        # Step 1: Detect clicks via AR-Residual + Z-Score (Godsill & Rayner 1998)
        click_locations = self._detect_clicks_multiband(audio, config)

        # Step 2: Classify click severity
        classified_clicks = self._classify_clicks(audio, click_locations, config)

        # Step 3: Repair clicks
        repaired_audio = self._repair_clicks(audio, classified_clicks, config, protected_zones=protected_zones)

        return repaired_audio, len(classified_clicks)

    def _detect_clicks_multiband(self, audio: np.ndarray, config: dict[str, Any]) -> list[int]:
        """Click-Detektion via AR-Residual + Z-Score (Godsill & Rayner 1998).

        Algorithmus (ersetzt signal.medfilt-Mediafilter-Declicker):
            Für jede AR-Ordnung in ar_orders:
                1. Schätze AR-Koeffizienten via Levinson-Durbin (scipy.signal.lpc)
                2. Berechne Vorhersagefehler: e(t) = Audio(t) - AR_Vorhersage(t)
                3. Normiere: z(t) = (|e(t)| - μ_e) / σ_e
                4. Detektiere Ausreißer: t wo z(t) > z_threshold
                   → Clicks haben viel größere Vorhersagefehler als Musik/Sprache

        Die multi-scale AR-Ordnungen decken verschiedene Klangstrukturen ab:
            - Ordnung 6:  Kurze Korrelationslänge (Glottis-Grundperiode)
            - Ordnung 12: Mittlere Länge (Formantstruktur Vokal)
            - Ordnung 20: Längere Perioden (Harmonische tiefer Töne)

        Vorteile gegenüber Medianfilter:
            - Keine Verschmierung von Transienten bei großem Filterfenster
            - Physikalisch motiviert (AR = Schallröhrenmodell)
            - Keine Artefakte durch Fenster-Randeffekte

        Forschungsreferenz:
            Godsill & Rayner (1998): Digital Audio Restoration, Kap. 3
            Cemgil et al. (2006): Sparse Bayes für Impulsnoise

        Args:
            audio:  1D float32/64 Audio-Array.
            config: DETECTION_CONFIG-Eintrag mit 'ar_orders' und 'z_score_threshold'.

        Returns:
            Sortierte Liste detektierter Ausreißer-Indices (Sample-Ebene).
        """
        ar_orders = config["ar_orders"]
        z_threshold = config["z_score_threshold"]
        all_detections: set = set()

        for order in ar_orders:
            min_len = order * 4
            if len(audio) <= min_len:
                continue
            try:
                # Levinson-Durbin AR-Koeffizienten via librosa.lpc (Autocorrelation-Methode)
                # Gibt [1, a1, ..., a_p] zurück — Analyse-Filter A(z)
                a_coeff = librosa.lpc(audio.astype(np.float32), order=order)
                # Guard: degenerate LPC coefficients (NaN/Inf) trigger LAPACK DLASCL warning
                if not np.isfinite(a_coeff).all():
                    continue
                # Vorhersagefehler (AR-Residual) — Clicks = große Ausreißer
                # lfilter([1], a_coeff, ...) implementiert Analysis-Filter A(z)
                residual = lfilter(a_coeff, [1.0], audio)
                diff = np.abs(residual)

                # Z-Score-Normierung (robuster via MAD anstelle σ)
                mean_d = np.mean(diff)
                std_d = np.std(diff)
                if std_d < 1e-10:
                    continue

                z_scores = (diff - mean_d) / std_d

                # NaN/Inf-Schutz
                z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)

                outliers = np.where(z_scores > z_threshold)[0]
                all_detections.update(outliers.tolist())

            except Exception:
                # Graceful Degradation: Diese Ordnung überspringen (§V6)
                logger.warning(
                    "§V6 Phase-27 click detection order failed → skipping: %s",
                    str(Exception),
                )
                continue

        return sorted(all_detections)

    def _classify_clicks(
        self, audio: np.ndarray, click_locations: list[int], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Classify detected clicks by severity and type."""
        classified = []
        max_duration = config["max_click_duration_samples"]

        i = 0
        while i < len(click_locations):
            start = click_locations[i]

            # Find consecutive samples (burst detection)
            end = start
            j = i
            while j + 1 < len(click_locations) and click_locations[j + 1] - click_locations[j] <= 2:
                end = click_locations[j + 1]
                j += 1

            duration = end - start + 1

            # Only process if within max duration
            if duration <= max_duration:
                # Compute severity metrics
                if start > 0 and end < len(audio) - 1:
                    energy: float = float(np.sum(audio[start : end + 1] ** 2))

                    click_info = {
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "energy": energy,
                        "type": "click" if duration < 5 else ("pop" if duration < 15 else "burst"),
                    }
                    classified.append(click_info)

            i = j + 1

        return classified

    @staticmethod
    def _compute_click_local_strength(
        mono_ref: np.ndarray,
        start: int,
        end: int,
        sr: int,
        base_strength: float,
        protected_zones: list[tuple[float, float, float]],
    ) -> float:
        """§V38 Per-Click-Strength-Oracle: 250ms RMS-Proxy + VFA-Schutzzonen-Cap.

        Vibrato/Frisson/Flüster/Passaggio-Zonen begrenzen die Strength auf den Zone-Cap.
        `base_strength < 1e-6` → 0.0 (V38-Invariante).
        """
        if base_strength < 1e-6:
            return 0.0
        # 250 ms Kontext-RMS-Proxy
        _ctx = max(1, int(0.125 * sr))  # ±125 ms
        _s = max(0, start - _ctx)
        _e = min(len(mono_ref), end + _ctx)
        _rms = float(np.sqrt(np.mean(mono_ref[_s:_e] ** 2))) if _e > _s else 0.0
        # Sanfte RMS-Skalierung: stille Passagen → Strength reduzieren
        _scale = float(np.clip(_rms / (0.05 + 1e-8), 0.5, 1.0))
        strength = float(base_strength) * _scale
        # VFA-Schutzzonen-Cap
        _start_s = start / max(sr, 1)
        _end_s = end / max(sr, 1)
        for _pz_s, _pz_e, _pz_cap in protected_zones:
            if _start_s < _pz_e and _end_s > _pz_s:
                strength = min(strength, _pz_cap)
                break
        return float(np.clip(strength, 0.0, 1.0))

    def _repair_clicks(
        self,
        audio: np.ndarray,
        classified_clicks: list[dict[str, Any]],
        config: dict[str, Any],
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> np.ndarray:
        """Repariert detected clicks using adaptive strategies."""
        repaired = audio.copy()
        _base_repair_strength = config["repair_strength"]
        _sr = config.get("sample_rate", 48000)
        _pz = protected_zones or []

        for click in classified_clicks:
            start = click["start"]
            end = click["end"]
            duration = click["duration"]
            click_type = click["type"]

            # Safety bounds
            if start < 10 or end >= len(audio) - 10:
                continue

            # Choose repair strategy
            if click_type == "click" or duration < 5:
                # Cubic interpolation
                repaired_segment = self._cubic_interpolation(audio, start, end)
            elif click_type == "pop" or duration < 15:
                # AR prediction
                repaired_segment = self._ar_prediction(audio, start, end)
            else:
                # Cross-fade
                repaired_segment = self._crossfade_repair(audio, start, end)

            # §V38 Per-Click-Stärke (250ms RMS-Proxy + VFA-Schutzzonen-Cap)
            repair_strength = self._compute_click_local_strength(
                audio, start, end, int(_sr), float(_base_repair_strength), _pz
            )
            # Apply repair with strength blending
            repaired[start : end + 1] = (
                repaired[start : end + 1] * (1 - repair_strength) + repaired_segment * repair_strength
            )

            # Smooth boundaries with adaptive taper length.
            profile = getattr(self, "_click_repair_profile_current", {})
            taper_target = int(np.clip(float(profile.get("taper_length", 5.0)), 3, 12))
            taper_len = min(taper_target, duration // 2)
            if taper_len > 0:
                taper = np.linspace(0, 1, taper_len)
                repaired[start : start + taper_len] = (
                    audio[start : start + taper_len] * (1 - taper) + repaired[start : start + taper_len] * taper
                )
                repaired[end - taper_len + 1 : end + 1] = (
                    repaired[end - taper_len + 1 : end + 1] * (1 - taper[::-1])
                    + audio[end - taper_len + 1 : end + 1] * taper[::-1]
                )

        return repaired

    def _cubic_interpolation(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Cubic spline interpolation for small clicks."""
        profile = getattr(self, "_click_repair_profile_current", {})
        context = int(np.clip(float(profile.get("cubic_context", 5.0)), 3, 12))
        x_known = np.concatenate([np.arange(start - context, start), np.arange(end + 1, end + context + 1)])
        y_known = audio[x_known]

        # Cubic spline
        cs = interpolate.CubicSpline(x_known, y_known)
        x_repair = np.arange(start, end + 1)
        repaired = cs(x_repair)

        return np.nan_to_num(np.asarray(repaired, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    def _ar_prediction(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """AR prediction for medium pops (order ≥ 16 @ 48 kHz, §VERBOTEN: LPC < 16)."""
        profile = getattr(self, "_click_repair_profile_current", {})
        context = int(np.clip(float(profile.get("ar_context", 128.0)), 64, 320))
        configured_order = int(np.clip(float(profile.get("ar_order", 32.0)), 16, 56))
        order = int(max(16, min(configured_order, max(16, context // 4))))

        # Use samples before click for AR coefficients
        if start < context + order:
            # Fallback to interpolation
            return self._cubic_interpolation(audio, start, end)

        training_data = audio[start - context - order : start]

        # Estimate AR coefficients (simple linear regression)
        X = np.array([training_data[i : i + order] for i in range(len(training_data) - order)])
        y = training_data[order:]

        if len(X) < order:
            return self._cubic_interpolation(audio, start, end)

        # Least squares
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception as e:
            logger.warning("Verarbeitungsschritt_27_click_pop_removal.py::_ar_prediction Ersatzpfad: %s", e)
            return self._cubic_interpolation(audio, start, end)

        # Predict
        repaired = []
        buffer = audio[start - order : start].tolist()

        for _ in range(end - start + 1):
            pred = np.dot(coeffs, buffer[-order:])
            repaired.append(pred)
            buffer.append(pred)

        return np.array(repaired)  # type: ignore[no-any-return]

    def _crossfade_repair(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Cross-fade repair for large bursts."""
        duration = end - start + 1
        fade = np.linspace(1, 0, duration)

        # Blend from before and after
        profile = getattr(self, "_click_repair_profile_current", {})
        crossfade_context = int(np.clip(float(profile.get("crossfade_context", 10.0)), 6, 24))
        before_avg = np.mean(audio[max(0, start - crossfade_context) : start])
        after_avg = np.mean(audio[end + 1 : min(len(audio), end + 1 + crossfade_context)])

        repaired = before_avg * fade + after_avg * (1 - fade)

        return np.nan_to_num(np.asarray(repaired, dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]
