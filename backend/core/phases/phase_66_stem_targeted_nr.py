"""
Phase 66 — Stem-Targeted NR (Vokal + Begleitung, getrennte Restaurierung).

Spec §7.11 [RELEASE_MUST] (v10.0.0)

Zweck: Qualitätsstufe, die mit einer klassischen wideband-NR auf dem Gesamtmix
nicht erreichbar ist: Vokal-Stem und Begleitungs-Stem werden getrennt rausch-
unterdrückt mit stemspezifischen Parametern.

Algorithmus:
    1. BS-RoFormer Stem-Separation → vocals + other (Begleitung)
       (Fallback: MDX23C Kim_Vocal_2 → BSRoFormer-Fallback)
    2. Confidence-Gate: SDRi-Schätzwert < 5.0 dB → skip (Separation zu unsicher)
    3. Vokal-Stem → DFN v3 II mit energy_bias=−6 dB (VFA-Schutzzone-aware)
    4. Begleitungs-Stem → DFN v3 II mit energy_bias=−9 dB (Instrumental-Optimum)
    5. Rekombination: vocals_clean + acc_clean
    6. VQI-Guard: VQI_post < VQI_pre → Rollback auf Input (§0p)

§0a-Invariante:
    ERLAUBT in Restoration UND Studio 2026 (kein §0a-Ausschluss).
    Restoration: kein additiver Energiegewinn über Input-RMS hinaus.
    Studio 2026: vollständige Enhancement-Kette nach NR.

Aktivierungs-Gates:
    - panns_singing ≥ 0.40 (klares Stimmaterial — Separation lohnt sich)
    - Material NICHT in {shellac, wax_cylinder} (zu viel Rauschen für BSRoFormer)
    - Eingabedauer ≥ 3.0 s (Mindestlänge für stabile Separation)

§0h Music-Death-Shield: artifact_freedom-Guard nach Rekombination.
§0p Vocal-Supremacy: VQI-Gate für vokales Material.
§2.46e HallucinationGuard: nach additiver Verarbeitung (Studio 2026).
§2.69 TemporalContinuityGuard: Post-Phase-Hook.
§2.63 Reflect-Padding: nicht nötig (kein STFT in dieser Phase direkt).

Wissenschaftliche Grundlage:
    - BS-RoFormer (Lu et al. 2023): Band-Split RoPE Transformer, beste SDR auf MusDB18
    - MDX23C (Kim et al.): Fallback, Kim_Vocal_2 Modell
    - DeepFilterNet v3 (Schröter et al. 2022/2023): state-of-the-art NR für Sprache/Musik
    - §4.4a Spec: BSRoFormer = Primär, MDX23C = Fallback (Spec 04, SOTA-Matrix)

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0)
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from backend.core.audio_utils import to_channels_last
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.phase_strength_contract import resolve_phase_strength_contract

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithmus-Konstanten (§7.11a normativ)
# ---------------------------------------------------------------------------
_PANNS_SINGING_GATE: float = 0.40  # Mindest-Vocal-Konfidenz für Aktivierung
_SDR_CONFIDENCE_GATE: float = 5.0  # Mindest-SDRi in dB (Separationsqualität)
_MIN_DURATION_SECONDS: float = 3.0  # Mindest-Eingabedauer
_ENERGY_BIAS_VOCAL_DB: float = -6.0  # DFN energy_bias für Vokal-Stem (§0j)
_ENERGY_BIAS_ACC_DB: float = -9.0  # DFN energy_bias für Begleitungs-Stem (§0j)
_VQI_ROLLBACK_THRESHOLD: float = -0.02  # VQI-Delta-Minimum (Regression → Rollback)

# Materialien bei denen Separation zu risikoreich ist (zu laut für BSRoFormer)
_SKIP_MATERIALS: frozenset[MaterialType] = frozenset(
    {
        MaterialType.SHELLAC,
        MaterialType.WAX_CYLINDER,
    }
)

# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------
_instance: StemTargetedNRPhase | None = None
_lock = threading.Lock()


class StemTargetedNRPhase(PhaseInterface):
    """Phase 66: Stem-Targeted NR — Vokal/Begleitung getrennte Restaurierung.

    Trennt Audio in Vokal-Stem (BSRoFormer) und Begleitungs-Stem, wendet
    stem-spezifische NR an und kombiniert die saubereren Stems zurück.
    §0a-konform in beiden Modi. VQI-Guard schützt Stimmqualität.
    """

    _PHASE_ID = "phase_66_stem_targeted_nr"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name="Stem-Targeted NR (Vokal + Begleitung)",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=5,
            version="1.0.0",
            dependencies=["phase_03_denoise", "phase_29_tape_hiss_reduction"],
            estimated_time_factor=0.25,
            memory_requirement_mb=1200,  # BSRoFormer + 2x DFN
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.91,
            description=(
                "§7.11 Stem-Targeted NR: BSRoFormer Separation → Vokal-DFN (−6 dB bias) + "
                "Acc-DFN (−9 dB bias) → Rekombination. "
                "Aktiviert wenn panns_singing ≥ 0.40 AND SDRi ≥ 5 dB. "
                "VQI-Guard: Rollback wenn VQI_post < VQI_pre. "
                "§0a: Erlaubt in Restoration + Studio 2026."
            ),
        )

    # ------------------------------------------------------------------
    # Lazy-load Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _get_separator():
        """Lazy-load BS-RoFormer (primär) oder MDX23C (Fallback)."""
        try:
            from plugins.bs_roformer_plugin import get_bs_roformer

            sep = get_bs_roformer()
            if sep is not None:
                return sep, "bs_roformer"
        except Exception as _e:
            logger.debug("Verarbeitungsschritt_66: BS-RoFormer nicht verfügbar: %s", _e)
        try:
            from plugins.htdemucs_plugin import get_htdemucs_plugin

            _sep_fb = get_htdemucs_plugin()
            if _sep_fb is not None:
                return _sep_fb, "mdx23c_fallback"
        except Exception as _e2:
            logger.debug("Verarbeitungsschritt_66: MDX23C-Ersatzpfad nicht verfügbar: %s", _e2)
        return None, "unavailable"

    @staticmethod
    def _get_dfn():
        """Lazy-load DeepFilterNet v3 II Plugin (§7.11, Spec-04-NR-Kette).

        Reihenfolge: DFN (primär) → None (OMLSA-DSP-Fallback in _apply_dsp_nr).
        Der Whisper-Denoiser ist deprecated (Rev. 2026-08-16) und nicht mehr
        Teil der Kette — Denoising tragen DFN/SGMSE+/OMLSA gemäß Spec 04.
        """
        try:
            from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin

            return DeepFilterNetV3IIPlugin()
        except Exception as _e:
            logger.debug("Verarbeitungsschritt_66: DFN nicht verfügbar: %s", _e)
        return None

    # ------------------------------------------------------------------
    # Kernlogik
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono_mix(audio: np.ndarray) -> np.ndarray:
        """Channels-last [N, C] → Mono [N]."""
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            return np.nan_to_num(a, nan=0.0)  # type: ignore[no-any-return]
        if a.ndim == 2 and a.shape[1] <= 2:
            return np.nan_to_num(np.asarray(a.mean(axis=1), dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]
        return np.nan_to_num(np.asarray(a.mean(axis=-1), dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]

    @staticmethod
    def _apply_dsp_nr(
        stem: np.ndarray,
        sr: int,
        energy_bias_db: float,
        dfn_plugin,
    ) -> np.ndarray:
        """Wendet DFN-NR oder OMLSA-Fallback auf einen Stem an.

        Args:
            stem:          Audio-Array [N] Mono oder [N, 2] Stereo.
            sr:            Sample-Rate (muss 48000 Hz sein).
            energy_bias_db: Energy-Bias für DFN (−6 dB Vokal, −9 dB Instrumental).
            dfn_plugin:    DFN-Plugin-Instanz oder None (OMLSA-Fallback).
        """
        stem_f = np.asarray(stem, dtype=np.float32)
        if dfn_plugin is not None:
            try:
                return np.nan_to_num(np.asarray(dfn_plugin.enhance(stem_f, sr, energy_bias_db=energy_bias_db), dtype=np.float32), nan=0.0)  # type: ignore[no-any-return]
            except Exception as _e:
                logger.debug("Verarbeitungsschritt_66: DFN verbessern Fehler (OMLSA-Ersatzpfad): %s", _e)

        # OMLSA-DSP-Fallback (AiDehiss OMLSA/MMSE-LSA — kanonische Implementierung in dsp/dehiss.py)
        try:
            from dsp.dehiss import AiDehiss

            mono_f = stem_f[:, 0] if stem_f.ndim == 2 else stem_f
            cleaned: np.ndarray = np.asarray(AiDehiss().dehiss(mono_f, sr), dtype=np.float32)
            if stem_f.ndim == 2:
                # Beide Kanäle mit identischem Gain
                diff = cleaned - mono_f
                return np.nan_to_num(np.clip(stem_f + diff[:, np.newaxis], -1.0, 1.0).astype(np.float32), nan=0.0)  # type: ignore[no-any-return]
            return np.nan_to_num(np.clip(cleaned, -1.0, 1.0).astype(np.float32), nan=0.0)  # type: ignore[no-any-return]
        except Exception as _e2:
            logger.debug("Verarbeitungsschritt_66: OMLSA-Ersatzpfad Fehler: %s", _e2)
        return np.nan_to_num(stem_f, nan=0.0)  # type: ignore[no-any-return]

    def process(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str | MaterialType = MaterialType.UNKNOWN,
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("BS-RoFormer", phase_name="66")
        check_ml_model_ready("DeepFilterNetV3", phase_name="66")
        check_ml_model_ready("Demucs", phase_name="66")
        check_ml_model_ready("PANNs", phase_name="66")
        """Stem-Targeted NR.

        Args:
            audio:       Mono [N] oder Stereo [N, 2] (channels-last)
            sample_rate: Abtastrate Hz (MUSS 48000)
            **kwargs:
                panns_singing (float): PANNs-Singing-Konfidenz [0, 1]
                quality_mode (str): "restoration" | "studio_2026"
                strength (float): Phase-Strength [0, 1]
                vfa_result (dict): VocalFocusAnalyzer-Ergebnis
        """
        assert sample_rate == 48000, f"Phase66 SR MUSS 48000 Hz sein, erhalten: {sample_rate}"
        # ── §v10 PIM: Per-Band-Intensität kalibrieren ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim = apply_pim_intensity(kwargs, "stem_nr", default_nr=0.5, default_de_ess=0.3, default_comp=1.0)
            if kwargs.get("pim_intensity_map") is not None:
                for _key in ("noise_reduction_strength", "nr_strength", "strength", "wet"):
                    if _key in kwargs:
                        kwargs[_key] = _pim["nr_strength"]
        except Exception as e:
            logger.warning("Verarbeitungsschritt_66_stem_targeted_nr.py::verarbeiten Ersatzpfad: %s", e)
        audio, _p66_transposed = to_channels_last(audio)
        self.validate_input(audio)
        t0 = time.time()

        _p66_meta: dict = {
            "algorithm": "stem_targeted_nr",
            "separator_used": "unavailable",
            "sdri_db": 0.0,
            "separation_confidence": 0.0,
            "vocal_nr_applied": False,
            "acc_nr_applied": False,
            "activation_triggered": False,
            "vqi_before": None,
            "vqi_after": None,
            "rollback_reason": None,
        }

        _strength_ctx = resolve_phase_strength_contract(kwargs)
        strength = float(_strength_ctx["effective_strength"])
        quality_mode = str(kwargs.get("quality_mode", "restoration")).strip().lower()
        panns_singing = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if not isinstance(material_type, MaterialType):
            try:
                material_type = MaterialType(str(material_type).strip().lower())
            except Exception:
                material_type = MaterialType.UNKNOWN

        def _passthrough(reason: str) -> PhaseResult:
            """Gibt Input unverändert zurück (Gate nicht erfüllt oder Rollback)."""
            audio_out = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p66_transposed:
                audio_out = audio_out.T
            _p66_meta["rollback_reason"] = reason
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p66_meta,
            )

        # --- Aktivierungs-Gates ---
        if panns_singing < _PANNS_SINGING_GATE:
            return _passthrough(f"panns_singing={panns_singing:.2f} < {_PANNS_SINGING_GATE}")

        if material_type in _SKIP_MATERIALS:
            return _passthrough(f"material={material_type} in skip-list (zu verrauscht für Separation)")

        n_samples = audio.shape[0] if audio.ndim >= 1 else len(audio)
        duration_s = n_samples / sample_rate
        if duration_s < _MIN_DURATION_SECONDS:
            return _passthrough(f"Dauer={duration_s:.1f}s < {_MIN_DURATION_SECONDS}s Minimum")

        # --- VQI-Baseline (vor Eingriff) ---
        vqi_before: float | None = None
        if panns_singing >= 0.35:
            try:
                from backend.core.musical_goals.vocal_quality_index import compute_vqi

                _audio_mono_ref = self._to_mono_mix(audio)
                _vqi_ref = compute_vqi(_audio_mono_ref, _audio_mono_ref, sample_rate)
                vqi_before = float(_vqi_ref.get("vqi", 0.0))
                _p66_meta["vqi_before"] = vqi_before
            except Exception as _vqi_e:
                logger.debug("Verarbeitungsschritt_66: VQI-Baseline Fehler (nicht blockierend): %s", _vqi_e)

        # --- Stem-Separation ---
        separator, sep_name = self._get_separator()
        if separator is None:
            return _passthrough("Kein Stem-Separator verfügbar")

        try:
            audio_for_sep = np.asarray(audio, dtype=np.float32)
            sep_result = separator.separate(audio_for_sep, sample_rate, stems=["vocals", "other"])
            _p66_meta["separator_used"] = sep_name
            _p66_meta["sdri_db"] = float(sep_result.sdri_db)
            _p66_meta["separation_confidence"] = float(sep_result.confidence)

            if sep_result.sdri_db < _SDR_CONFIDENCE_GATE or sep_result.confidence < 0.40:
                return _passthrough(
                    f"Separation SDRi={sep_result.sdri_db:.1f} dB / confidence={sep_result.confidence:.2f} unter Gate"
                )

            vocals_stem = sep_result.stems.get("vocals")
            other_stem = sep_result.stems.get("other")
            if vocals_stem is None or other_stem is None:
                return _passthrough("Kein vocals/other Stem im Ergebnis")

        except Exception as _sep_e:
            logger.debug("Verarbeitungsschritt_66: Stem-Separation Fehler: %s", _sep_e)
            return _passthrough(f"Separation Exception: {_sep_e}")

        # --- Stem-weise NR ---
        dfn_plugin = self._get_dfn()

        # Vocals: vocal-optimierter energy_bias (§0j)
        try:
            vocals_clean = self._apply_dsp_nr(vocals_stem, sample_rate, _ENERGY_BIAS_VOCAL_DB, dfn_plugin)
            _p66_meta["vocal_nr_applied"] = True
        except Exception as _vnr_e:
            logger.debug("Verarbeitungsschritt_66: Vokal-NR Fehler, Vokal-Originalsignal verwendet: %s", _vnr_e)
            vocals_clean = np.asarray(vocals_stem, dtype=np.float32)

        # Begleitung: instrumental-optimierter energy_bias (§0j)
        try:
            acc_clean = self._apply_dsp_nr(other_stem, sample_rate, _ENERGY_BIAS_ACC_DB, dfn_plugin)
            _p66_meta["acc_nr_applied"] = True
        except Exception as _anr_e:
            logger.debug("Verarbeitungsschritt_66: Acc-NR Fehler, Acc-Originalsignal verwendet: %s", _anr_e)
            acc_clean = np.asarray(other_stem, dtype=np.float32)

        # --- Rekombination ---
        try:
            # Längenangleichung
            _n = min(
                vocals_clean.shape[0] if vocals_clean.ndim >= 1 else len(vocals_clean),
                acc_clean.shape[0] if acc_clean.ndim >= 1 else len(acc_clean),
                audio.shape[0],
            )
            vc = np.asarray(vocals_clean, dtype=np.float32)
            ac = np.asarray(acc_clean, dtype=np.float32)

            # Dimensionen angleichen
            if vc.ndim == 1 and ac.ndim == 2:
                vc = vc[:, np.newaxis]
            elif vc.ndim == 2 and ac.ndim == 1:
                ac = ac[:, np.newaxis]

            audio_combined = vc[:_n] + ac[:_n]
        except Exception as _comb_e:
            logger.debug("Verarbeitungsschritt_66: Rekombination Fehler: %s", _comb_e)
            return _passthrough(f"Rekombination Exception: {_comb_e}")

        # Längen- und Formatsicherung
        if audio_combined.shape[0] < audio.shape[0]:
            _pad_len = audio.shape[0] - audio_combined.shape[0]
            if audio_combined.ndim == 2:
                audio_combined = np.pad(audio_combined, ((0, _pad_len), (0, 0)))
            else:
                audio_combined = np.pad(audio_combined, (0, _pad_len))
        else:
            audio_combined = audio_combined[: audio.shape[0]]

        # Restoration: kein additiver Energiegewinn über Input-RMS (§0a-Invariante)
        if "restoration" in quality_mode:
            _in_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
            _out_rms = float(np.sqrt(np.mean(audio_combined**2) + 1e-12))
            if _out_rms > _in_rms * 1.05:
                _scale = _in_rms / (_out_rms + 1e-12)
                audio_combined = audio_combined * _scale

        audio_combined = np.nan_to_num(audio_combined, nan=0.0, posinf=0.0, neginf=0.0)
        audio_combined = np.clip(audio_combined, -1.0, 1.0)

        # --- Strength-Blend: wet/dry Mix ---
        if strength < 1.0:
            _n_blend = min(audio.shape[0], audio_combined.shape[0])
            audio_combined[:_n_blend] = strength * audio_combined[:_n_blend] + (1.0 - strength) * audio[:_n_blend]
            audio_combined = np.clip(audio_combined, -1.0, 1.0)

        # --- §2.46e HallucinationGuard (Studio 2026: additive Operation) ---
        if "studio" in quality_mode or "2026" in quality_mode:
            try:
                from backend.core.dsp.hallucination_guard import check_hallucination

                _hg = check_hallucination(
                    self._to_mono_mix(audio),
                    self._to_mono_mix(audio_combined),
                )
                if _hg.requires_rollback:
                    logger.debug("Verarbeitungsschritt_66: HallucinationGuard → Rollback")
                    return _passthrough("HallucinationGuard Rollback (Studio 2026)")
            except Exception as _hg_e:
                logger.debug("Verarbeitungsschritt_66: HallucinationGuard nicht blockierend: %s", _hg_e)

        # --- §0p VQI-Guard ---
        if panns_singing >= 0.35 and vqi_before is not None:
            try:
                from backend.core.musical_goals.vocal_quality_index import compute_vqi

                _audio_mono_out = self._to_mono_mix(audio_combined)
                _audio_mono_in = self._to_mono_mix(audio)
                _vqi_res = compute_vqi(_audio_mono_in, _audio_mono_out, sample_rate)
                vqi_after = float(_vqi_res.get("vqi", 0.0))
                _p66_meta["vqi_after"] = vqi_after
                if vqi_after < vqi_before + _VQI_ROLLBACK_THRESHOLD:
                    logger.debug(
                        "Verarbeitungsschritt_66: VQI-Rollback: %.3f → %.3f (Delta=%.3f)",
                        vqi_before,
                        vqi_after,
                        vqi_after - vqi_before,
                    )
                    return _passthrough(f"VQI-Guard Rollback: VQI {vqi_before:.3f}→{vqi_after:.3f}")
            except Exception as _vqi_after_e:
                logger.debug("Verarbeitungsschritt_66: VQI-After Fehler (nicht blockierend): %s", _vqi_after_e)

        _p66_meta["activation_triggered"] = True

        # V19 Noise-Textur-Invariante (§NTI): Residual nach Stem-NR darf kein
        # material-fremdes Spektralprofil aufweisen (VERBOTEN-V19).
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt66_dist_fn,
            )

            _nt66_residual = audio.astype(np.float32) - audio_combined.astype(np.float32)
            _nt66_mat = material_type.value if hasattr(material_type, "value") else str(material_type)
            _nt66_dist = _nt66_dist_fn(_nt66_residual, _nt66_mat, sr=sample_rate)
            if _nt66_dist > 0.25:
                audio_combined = (0.5 * audio_combined + 0.5 * audio).astype(np.float32)
                logger.warning("Verarbeitungsschritt66 V19 Noise-Textur-Dist=%.3f > 0.25 → 50%%-Blend", _nt66_dist)
        except Exception as _nt66_exc:
            logger.debug("Verarbeitungsschritt66 V19 Noise-Textur-Guard (nicht blockierend): %s", _nt66_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach Stem-NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_66,
            )

            _sc_result_66 = _scg_66(audio, audio_combined, sample_rate)
            if not _sc_result_66.ok:
                _sc_wet_66 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                audio_combined = (_sc_wet_66 * audio_combined + (1.0 - _sc_wet_66) * audio).astype(np.float32)
        except Exception as _sc_exc_66:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_66 spectral_color nicht blockierend: %s", _sc_exc_66
            )

        # V26 Onset-Guard (§2.77): Transients nach Stem-NR schützen (non-blocking)
        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opg66,
            )

            audio_combined = _opg66(audio, audio_combined, None, max_delta_db=1.5)
        except Exception as _on66_exc:
            logger.debug("Verarbeitungsschritt66 V26 Onset-Guard (nicht blockierend): %s", _on66_exc)

        if _p66_transposed:
            audio_combined = audio_combined.T

        logger.info("Verarbeitungsschritt=%s Wert=%.2f", self._PHASE_ID, 1.0)

        return PhaseResult(
            success=True,
            audio=audio_combined.astype(np.float32),
            execution_time_seconds=time.time() - t0,
            metadata=_p66_meta,
        )


# ---------------------------------------------------------------------------
# Singleton-Accessor
# ---------------------------------------------------------------------------


def get_phase_66() -> StemTargetedNRPhase:
    """Thread-safe Singleton-Accessor für StemTargetedNRPhase (phase_66)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StemTargetedNRPhase()
    return _instance
