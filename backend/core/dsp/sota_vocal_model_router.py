"""SOTA vocal model router for vocal-first music restoration (§SMR-1).

Centralizes model choice for pre-phase vocal/instrumental stem restoration:
BS-RoFormer → Demucs v4 → MDX23C for separation, and MIIPHER → SGMSE+ →
DeepFilterNet for vocal NR.  The router is deliberately adapter-only: it does
not invent new DSP, it selects the strongest available local plugin and returns
explicit fallback metadata.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from operator import methodcaller
from typing import Any

import numpy as np

from backend.core.dsp.stem_routing_policy import prefer_demucs_native_from_ctx

logger = logging.getLogger(__name__)


@dataclass
class StemSeparationRouteResult:
    """Result of routed vocal/instrumental separation."""

    vocal: np.ndarray
    instrumental: np.ndarray
    success: bool
    model_used: str
    fallback_chain: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class EnhancementRouteResult:
    """Result of routed stem enhancement."""

    audio: np.ndarray
    success: bool
    model_used: str
    fallback_chain: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class SotaVocalModelRouter:
    """Routes vocal-centric separation and NR through the best local model."""

    _SEPARATION_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    _ROFORMER_MIN_MEM_GB = 10.0
    _DEMUCS_MIN_MEM_GB = 5.0
    _MDX_MIN_MEM_GB = 3.0

    def separate_vocal_instrumental(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        panns_singing: float = 0.0,
        ctx: dict[str, Any] | None = None,
    ) -> StemSeparationRouteResult:
        """Separate mix into vocal + full instrumental stems using routed SOTA plugins."""
        route_ctx = ctx or {}
        reference = np.asarray(audio, dtype=np.float32)
        attempts: list[str] = []
        capability_report = self._capability_report()
        prefer_demucs_native = self._prefer_demucs_native(route_ctx)
        score_routing_enabled = bool(route_ctx.get("score_routing_enabled", True))
        pressure_factor = self._runtime_pressure_multiplier(route_ctx)

        if panns_singing >= 0.35:
            _available_mem = self._available_memory_gb()
            _required_roformer = self._required_memory_gb("bs_roformer", reference, sr, pressure_factor=pressure_factor)
            if _available_mem is not None and _available_mem < _required_roformer:
                attempts.append(f"bs_roformer:preflight_low_ram_{_available_mem:.1f}GB_req_{_required_roformer:.1f}GB")
                logger.info(
                    "§SMR-1 BS-RoFormer preflight ueberspringen (verfuegbar=%.1fGB < required=%.1fGB)",
                    _available_mem,
                    _required_roformer,
                )
            else:
                try:
                    from plugins.bs_roformer_plugin import get_bs_roformer  # pylint: disable=import-outside-toplevel

                    separator = get_bs_roformer()
                    result = separator.separate(reference, sr, stems=self._SEPARATION_STEMS)
                    stems_obj = getattr(result, "stems", {})
                    model_used = str(getattr(result, "model_used", "bs_roformer"))
                    if isinstance(stems_obj, dict) and stems_obj and self._is_primary_roformer_model(model_used):
                        vocal, instrumental = self._stems_to_vocal_instr(stems_obj, reference)
                        return StemSeparationRouteResult(
                            vocal=vocal,
                            instrumental=instrumental,
                            success=True,
                            model_used=model_used,
                            fallback_chain=attempts.copy(),
                            metadata={
                                "confidence": float(getattr(result, "confidence", 0.0)),
                                "sdri_db": float(getattr(result, "sdri_db", 0.0)),
                                "roformer_model_loaded": bool(getattr(separator, "_model_loaded", False)),
                                "capability_status": self._capability_status(capability_report, "melbandroformer"),
                            },
                        )
                    if isinstance(stems_obj, dict) and stems_obj:
                        attempts.append(f"bs_roformer:{model_used}")
                        logger.debug(
                            "§SMR-1 BS/MelBand-RoFormer returned Ersatzpfad model=%s; trying next separator",
                            model_used,
                        )
                        raise RuntimeError("roformer_fallback_result")
                    attempts.append("bs_roformer:empty_stems")
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
                    if str(exc) != "roformer_fallback_result":
                        attempts.append(f"bs_roformer:{type(exc).__name__}")
                    logger.debug("§SMR-1 BS-RoFormer separation nicht verfuegbar: %s", exc)

        def _try_demucs() -> StemSeparationRouteResult | None:
            _available_mem = self._available_memory_gb()
            _required_demucs = self._required_memory_gb("demucs_v4", reference, sr, pressure_factor=pressure_factor)
            if _available_mem is not None and _available_mem < _required_demucs:
                attempts.append(f"demucs_v4:preflight_low_ram_{_available_mem:.1f}GB_req_{_required_demucs:.1f}GB")
                logger.info(
                    "§SMR-1 Demucs preflight ueberspringen (verfuegbar=%.1fGB < required=%.1fGB)",
                    _available_mem,
                    _required_demucs,
                )
                return None
            try:
                from plugins.demucs_v4_plugin import get_demucs_plugin  # pylint: disable=import-outside-toplevel

                demucs = get_demucs_plugin()
                if getattr(demucs, "_session", None) is None:
                    attempts.append("demucs_v4:no_native_session")
                    return None
                try:
                    stems = demucs.separate(reference, sr, prefer_mdx23c=False)
                except TypeError:
                    # Backward compatibility for older plugin stubs in tests.
                    stems = demucs.separate(reference, sr)
                if isinstance(stems, dict) and stems:
                    vocal, instrumental = self._stems_to_vocal_instr(stems, reference)
                    return StemSeparationRouteResult(
                        vocal=vocal,
                        instrumental=instrumental,
                        success=True,
                        model_used="demucs_v4_htdemucs",
                        fallback_chain=attempts.copy(),
                        metadata={"capability_status": self._capability_status(capability_report, "demucs_v4")},
                    )
                attempts.append("demucs_v4:empty_stems")
            except Exception as exc:  # pylint: disable=broad-except
                attempts.append(f"demucs_v4:{type(exc).__name__}")
                logger.debug("§SMR-1 Demucs separation nicht verfuegbar: %s", exc)
            return None

        def _try_mdx23c() -> StemSeparationRouteResult | None:
            _available_mem = self._available_memory_gb()
            _required_mdx = self._required_memory_gb("mdx23c", reference, sr, pressure_factor=pressure_factor)
            if _available_mem is not None and _available_mem < _required_mdx:
                attempts.append(f"mdx23c:preflight_low_ram_{_available_mem:.1f}GB_req_{_required_mdx:.1f}GB")
                logger.info(
                    "§SMR-1 MDX23C preflight ueberspringen (verfuegbar=%.1fGB < required=%.1fGB)",
                    _available_mem,
                    _required_mdx,
                )
                return None
            try:
                mdx_module = __import__("plugins.mdx23c_plugin", fromlist=["get_mdx23c_plugin"])
                get_mdx23c_plugin = mdx_module.get_mdx23c_plugin
                get_loaded_mdx23c_plugin = getattr(mdx_module, "get_loaded_mdx23c_plugin", None)

                mdx = get_loaded_mdx23c_plugin() if callable(get_loaded_mdx23c_plugin) else None
                if mdx is None:
                    mdx = get_mdx23c_plugin()
                stems = mdx.separate_all_stems(reference, sr, stems=["vocals", "inst"])
                if isinstance(stems, dict) and stems:
                    vocal = self._coerce_like(stems.get("vocals", np.zeros_like(reference)), reference)
                    instrumental = self._coerce_like(stems.get("inst", reference - vocal), reference)
                    return StemSeparationRouteResult(
                        vocal=vocal,
                        instrumental=instrumental,
                        success=True,
                        model_used="mdx23c",
                        fallback_chain=attempts.copy(),
                        metadata={"capability_status": "sota_fallback"},
                    )
                attempts.append("mdx23c:empty_stems")
            except Exception as exc:  # pylint: disable=broad-except
                attempts.append(f"mdx23c:{type(exc).__name__}")
                logger.debug("§SMR-1 MDX23C separation nicht verfuegbar: %s", exc)
            return None

        if prefer_demucs_native:
            demucs_result = _try_demucs()
            mdx_result = _try_mdx23c()
            _ranked = self._rank_separation_candidates(
                [demucs_result, mdx_result],
                reference,
                panns_singing=panns_singing,
                prefer_demucs_native=prefer_demucs_native,
                score_routing_enabled=score_routing_enabled,
            )
            if _ranked is not None:
                return _ranked
        else:
            mdx_result = _try_mdx23c()
            demucs_result = _try_demucs()
            _ranked = self._rank_separation_candidates(
                [mdx_result, demucs_result],
                reference,
                panns_singing=panns_singing,
                prefer_demucs_native=prefer_demucs_native,
                score_routing_enabled=score_routing_enabled,
            )
            if _ranked is not None:
                return _ranked

        return StemSeparationRouteResult(
            vocal=np.zeros_like(reference, dtype=np.float32),
            instrumental=reference.copy(),
            success=False,
            model_used="dsp_fallback_required",
            fallback_chain=attempts,
        )

    def _prefer_demucs_native(self, ctx: dict[str, Any]) -> bool:
        """Aktiviert nativen HTDemucs-Pfad nur für live/crowd-nahe Quellen."""
        return prefer_demucs_native_from_ctx(ctx)

    @staticmethod
    def _available_memory_gb() -> float | None:
        """Liefert verfügbaren RAM in GB oder None, wenn nicht ermittelbar."""
        try:
            import psutil as _psutil  # pylint: disable=import-outside-toplevel

            return float(_psutil.virtual_memory().available / (1024**3))
        except Exception:  # pylint: disable=broad-except
            return None

    @classmethod
    def _required_memory_gb(
        cls,
        model_name: str,
        audio: np.ndarray,
        sr: int,
        *,
        pressure_factor: float = 1.0,
    ) -> float:
        """Schätzt RAM-Bedarf für Separatoren adaptiv (Basis + Dauer + Kanalfaktor)."""
        name = str(model_name).lower()
        if "roformer" in name:
            base = cls._ROFORMER_MIN_MEM_GB
            per_min = 1.0
        elif "demucs" in name:
            base = cls._DEMUCS_MIN_MEM_GB
            per_min = 0.5
        else:
            base = cls._MDX_MIN_MEM_GB
            per_min = 0.3

        duration_s = cls._audio_duration_seconds(audio, sr)
        duration_factor = max(0.0, duration_s / 60.0 - 1.0)
        channels = cls._channel_count(audio)
        channel_factor = 1.0 + (0.15 * max(0, channels - 1))
        required = float(base + per_min * duration_factor * channel_factor)
        return float(required * max(1.0, float(pressure_factor)))

    def _runtime_pressure_multiplier(self, ctx: dict[str, Any]) -> float:
        """Schätzt Lastfaktor aus RAM/Swap-Druck und aktiver ML-Parallelität."""
        factor = 1.0

        try:
            import psutil as _psutil  # pylint: disable=import-outside-toplevel

            vm = _psutil.virtual_memory()
            ram_pct = float(getattr(vm, "percent", 0.0))
            if ram_pct >= 90.0:
                factor += 0.30
            elif ram_pct >= 80.0:
                factor += 0.15

            swap = _psutil.swap_memory()
            swap_pct = float(getattr(swap, "percent", 0.0))
            if swap_pct >= 80.0:
                factor += 0.35
            elif swap_pct >= 60.0:
                factor += 0.20
            elif swap_pct >= 40.0:
                factor += 0.10
        except Exception:  # pylint: disable=broad-except
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        active_ml_plugins = int(max(0, self._active_ml_plugin_count(ctx)))
        if active_ml_plugins >= 4:
            factor += 0.30
        elif active_ml_plugins >= 2:
            factor += 0.15
        elif active_ml_plugins >= 1:
            factor += 0.05

        return float(max(1.0, factor))

    @staticmethod
    def _active_ml_plugin_count(ctx: dict[str, Any]) -> int:
        """Ermittelt aktive ML-Plugins aus Kontext oder PluginLifecycleManager."""
        try:
            if isinstance(ctx, dict) and "active_ml_plugins" in ctx:
                return int(max(0, int(ctx.get("active_ml_plugins", 0))))
        except Exception:  # pylint: disable=broad-except
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        try:
            from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                get_plugin_lifecycle_manager,
            )

            plm = get_plugin_lifecycle_manager()
            entries = getattr(plm, "_entries", {})
            if isinstance(entries, dict):
                return int(sum(1 for entry in entries.values() if bool(getattr(entry, "active", False))))
        except Exception:  # pylint: disable=broad-except
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        return 0

    @staticmethod
    def _audio_duration_seconds(audio: np.ndarray, sr: int) -> float:
        arr = np.asarray(audio)
        if sr <= 0:
            return 0.0
        if arr.ndim <= 1:
            return float(arr.shape[0] / sr)
        # Robust for [n, c] and [c, n]: longest axis is treated as time.
        return float(max(arr.shape) / sr)

    @staticmethod
    def _channel_count(audio: np.ndarray) -> int:
        arr = np.asarray(audio)
        if arr.ndim <= 1:
            return 1
        return int(min(arr.shape))

    def _rank_separation_candidates(
        self,
        candidates: list[StemSeparationRouteResult | None],
        reference: np.ndarray,
        *,
        panns_singing: float,
        prefer_demucs_native: bool,
        score_routing_enabled: bool,
    ) -> StemSeparationRouteResult | None:
        """Wählt den besten Kandidaten anhand eines robusten Route-Scores."""
        available = [c for c in candidates if c is not None]
        if not available:
            return None
        if len(available) == 1:
            return available[0]
        if not score_routing_enabled:
            return available[0]

        best: StemSeparationRouteResult | None = None
        best_score = -1.0
        for candidate in available:
            score, parts = self._score_separation_candidate(
                candidate,
                reference,
                panns_singing=panns_singing,
                prefer_demucs_native=prefer_demucs_native,
            )
            candidate.metadata = dict(candidate.metadata)
            candidate.metadata["route_score"] = score
            candidate.metadata["route_score_parts"] = parts
            if score > best_score:
                best = candidate
                best_score = score

        if best is not None:
            best.metadata = dict(best.metadata)
            best.metadata["route_score_selected"] = True
        return best

    def _score_separation_candidate(
        self,
        candidate: StemSeparationRouteResult,
        reference: np.ndarray,
        *,
        panns_singing: float,
        prefer_demucs_native: bool,
    ) -> tuple[float, dict[str, float]]:
        """Bewertet Separator-Kandidaten über Rekonstruktion + Vocal-Energie-Plausibilität."""
        ref = self._coerce_like(reference, reference)
        vocal = self._coerce_like(candidate.vocal, ref)
        instrumental = self._coerce_like(candidate.instrumental, ref)
        reconstructed = np.clip(vocal + instrumental, -1.0, 1.0)

        ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2) + 1e-12))
        rec_err = float(np.mean((reconstructed.astype(np.float64) - ref.astype(np.float64)) ** 2))
        rec_norm = rec_err / float(np.mean(ref.astype(np.float64) ** 2) + 1e-12)
        reconstruction_score = float(np.clip(1.0 - 2.0 * rec_norm, 0.0, 1.0))

        vocal_rms = float(np.sqrt(np.mean(vocal.astype(np.float64) ** 2) + 1e-12))
        vocal_ratio = float(np.clip(vocal_rms / (ref_rms + 1e-12), 0.0, 1.0))
        target_ratio = float(np.clip(0.15 + 0.55 * float(np.clip(panns_singing, 0.0, 1.0)), 0.10, 0.75))
        vocal_plausibility = float(np.clip(1.0 - abs(vocal_ratio - target_ratio) / (target_ratio + 1e-6), 0.0, 1.0))

        model = str(candidate.model_used).lower()
        prior = 0.0
        if ("demucs" in model and prefer_demucs_native) or ("mdx" in model and not prefer_demucs_native):
            prior = 0.08
        elif "roformer" in model:
            prior = 0.10

        score = float(
            np.clip(
                0.55 * reconstruction_score + 0.35 * vocal_plausibility + 0.10 * float(np.clip(prior, 0.0, 1.0)),
                0.0,
                1.0,
            )
        )
        parts = {
            "reconstruction": reconstruction_score,
            "vocal_plausibility": vocal_plausibility,
            "model_prior": float(np.clip(prior, 0.0, 1.0)),
            "vocal_ratio": vocal_ratio,
            "target_ratio": target_ratio,
        }
        return score, parts

    def enhance_vocal(
        self,
        vocal_stem: np.ndarray,
        sr: int,
        *,
        energy_bias_db: float = -6.0,
        noise_snr_db: float = 0.0,
    ) -> EnhancementRouteResult:
        """Deep-Noise-Vokal-Kette via SGMSE+/DFN (Legacy-MIIPHER-Adapter) mit expliziten Fallback-Metadaten."""
        reference = np.asarray(vocal_stem, dtype=np.float32)
        attempts: list[str] = []
        capability_report = self._capability_report()

        # §v10.14: MIIPHER-DiT als primärer Open-Source-Ersatz
        # §v10.19 TODO: MP-SENet Musik als Post-Enhancer nach DFN einbauen
        # §v10.19 TODO: Proprietäre MIIPHER-Stufe (Zeile 474) entfernen für proprietäres MIIPHER
        try:
            from plugins.miipher_dit_plugin import get_miipher_dit

            _miipher_dit = get_miipher_dit()
            _dit_loaded = getattr(_miipher_dit, "_model_loaded", False)
            _dit_fallback = getattr(_miipher_dit, "_fallback_active", True)
            if _dit_loaded and not _dit_fallback:
                _dit_result = _miipher_dit.enhance(
                    reference,
                    sr,
                    material="unknown",
                    restorability_score=20.0,
                )
                if _dit_result.applied and _dit_result.model_used == "miipher_dit":
                    attempts.append("miipher_dit:primary")
                    logger.info(
                        "§v10.14 MIIPHER-DiT: Gesangsverbesserung erfolgreich (novelty=%.3f)", _dit_result.novelty
                    )
                    return EnhancementRouteResult(
                        audio=np.asarray(_dit_result.audio, dtype=np.float32),
                        success=True,
                        model_used="miipher_dit",
                        fallback_chain=list(attempts),
                        metadata={"route_taken": "miipher_dit", "implicit_harmonic_recovery": True},
                    )
                else:
                    attempts.append(f"miipher_dit:{_dit_result.model_used}")
            else:
                attempts.append("miipher_dit:not_loaded")
        except Exception as _dit_exc:
            logger.debug("MIIPHER-DiT nicht verfügbar: %s", _dit_exc)
            attempts.append("miipher_dit:import_error")

        # §v10.14: Proprietäres MIIPHER als Fallback (falls Modell vorhanden)
        try:
            from plugins.miipher_plugin import get_miipher_plugin  # pylint: disable=import-outside-toplevel

            miipher = get_miipher_plugin()
            model_loaded = bool(getattr(miipher, "_model_loaded", False))
            is_productive = False
            try:
                is_productive = bool(methodcaller("is_productive")(miipher))
            except Exception as prod_exc:  # pylint: disable=broad-except
                logger.debug("§SMR-1 MIIPHER adapter productivity Pruefung fehlgeschlagen: %s", prod_exc)
            if not model_loaded and not is_productive:
                attempts.append("miipher:not_loaded")
                raise RuntimeError("miipher_model_not_loaded")
            out = miipher.enhance(reference, sr, noise_snr_db=noise_snr_db)
            out_arr = self._coerce_like(out, reference)
            route_metadata = getattr(miipher, "route_metadata", {})
            if not isinstance(route_metadata, dict):
                route_metadata = {}
            route_status = str(
                route_metadata.get(
                    "capability_status",
                    self._capability_status(capability_report, "miipher"),
                )
            )
            route_model = str(route_metadata.get("model_used", "miipher" if model_loaded else "miipher_adapter"))
            if route_status == "dsp_fallback":
                attempts.append(f"miipher:{route_model}")
                raise RuntimeError("miipher_adapter_dsp_fallback")
            return EnhancementRouteResult(
                audio=out_arr,
                success=True,
                model_used=route_model,
                fallback_chain=attempts.copy(),
                metadata={
                    "miipher_model_loaded": model_loaded,
                    "miipher_adapter_productive": is_productive,
                    "energy_bias_db": float(energy_bias_db),
                    "capability_status": route_status,
                    "miipher_route_metadata": dict(route_metadata),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            if str(exc) not in {"miipher_model_not_loaded", "miipher_adapter_dsp_fallback"}:
                attempts.append(f"miipher:{type(exc).__name__}")
            logger.debug("§SMR-1 MIIPHER nicht verfuegbar: %s", exc)

        try:
            from plugins.sgmse_plugin import get_sgmse_plugin  # pylint: disable=import-outside-toplevel

            sgmse = get_sgmse_plugin()
            raw = sgmse.enhance(reference, sr)
            out = getattr(raw, "audio", raw)
            sgmse_audio = self._coerce_like(out, reference)
            _sgmse_model_used = str(getattr(raw, "model_used", "sgmse_plus"))
            compensated = self._compensate_missing_miipher(
                sgmse_audio,
                reference,
                sr,
                energy_bias_db=energy_bias_db,
                base_model_used=_sgmse_model_used,
            )
            metadata = {
                "energy_bias_db": float(energy_bias_db),
                "miipher_model_loaded": False,
                "sgmse_model_loaded": bool(getattr(sgmse, "_model_loaded", False)),
                "capability_status": self._capability_status(capability_report, "sgmse_plus"),
            }
            compensated_metadata = compensated.get("metadata")
            if isinstance(compensated_metadata, dict):
                metadata.update(compensated_metadata)
            return EnhancementRouteResult(
                audio=self._coerce_like(compensated.get("audio", sgmse_audio), reference),
                success=True,
                model_used=str(compensated.get("model_used", _sgmse_model_used)),
                fallback_chain=attempts.copy(),
                metadata=metadata,
            )
        except Exception as exc:  # pylint: disable=broad-except
            attempts.append(f"sgmse_plus:{type(exc).__name__}")
            logger.debug("§SMR-1 SGMSE+ nicht verfuegbar: %s", exc)

        dfn_result = self.enhance_instrumental(reference, sr, energy_bias_db=energy_bias_db)
        dfn_result.fallback_chain = attempts + dfn_result.fallback_chain
        if dfn_result.success:
            dfn_result.model_used = f"vocal_{dfn_result.model_used}"
        return dfn_result

    def _compensate_missing_miipher(
        self,
        sgmse_audio: np.ndarray,
        reference: np.ndarray,
        sr: int,
        *,
        energy_bias_db: float,
        base_model_used: str,
    ) -> dict[str, object]:
        """Kompensiert fehlendes MIIPHER bestmöglich via DFN + optionalem HNR-Blend."""
        base_audio = self._coerce_like(sgmse_audio, reference)
        metadata: dict[str, object] = {
            "miipher_compensation_active": True,
            "miipher_compensation_dfn_applied": False,
            "miipher_compensation_hnr_applied": False,
        }
        model_used = str(base_model_used or "sgmse_plus")

        dfn_result = self.enhance_instrumental(base_audio, sr, energy_bias_db=energy_bias_db)
        if not dfn_result.success:
            metadata["miipher_compensation_dfn_reason"] = "deepfilternet_unavailable"
            return {"audio": base_audio, "model_used": model_used, "metadata": metadata}

        compensated_audio = self._coerce_like(dfn_result.audio, reference)
        metadata["miipher_compensation_dfn_applied"] = True
        metadata["miipher_compensation_dfn_model"] = str(dfn_result.model_used)
        model_used = f"sgmse_plus+{dfn_result.model_used}"

        try:
            from backend.core.dsp.hnr_guard import apply_hnr_blend  # pylint: disable=import-outside-toplevel

            compensated_audio = self._coerce_like(apply_hnr_blend(reference, compensated_audio, sr), reference)
            metadata["miipher_compensation_hnr_applied"] = True
            model_used = f"{model_used}+hnr_blend"
        except Exception as exc:  # pylint: disable=broad-except
            metadata["miipher_compensation_hnr_reason"] = type(exc).__name__

        return {"audio": compensated_audio, "model_used": model_used, "metadata": metadata}

    def enhance_instrumental(
        self,
        stem: np.ndarray,
        sr: int,
        *,
        energy_bias_db: float = -9.0,
    ) -> EnhancementRouteResult:
        """Enhance instrumental stem via DeepFilterNet v3.II."""
        reference = np.asarray(stem, dtype=np.float32)
        attempts: list[str] = []
        capability_report = self._capability_report()
        try:
            from plugins.deepfilternet_v3_ii_plugin import (  # pylint: disable=import-outside-toplevel
                get_deepfilternet_plugin,
            )

            out = get_deepfilternet_plugin().enhance(reference, sr, energy_bias_db=energy_bias_db)
            return EnhancementRouteResult(
                audio=self._coerce_like(out, reference),
                success=True,
                model_used="deepfilternet_v3_ii",
                fallback_chain=attempts,
                metadata={
                    "energy_bias_db": float(energy_bias_db),
                    "capability_status": self._capability_status(capability_report, "deepfilternet_v3_ii"),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            attempts.append(f"deepfilternet_v3_ii:{type(exc).__name__}")
            logger.debug("§SMR-1 DeepFilterNet nicht verfuegbar: %s", exc)
            return EnhancementRouteResult(
                audio=reference.copy(),
                success=False,
                model_used="none",
                fallback_chain=attempts,
                metadata={"energy_bias_db": float(energy_bias_db), "capability_status": "dsp_fallback"},
            )

    @staticmethod
    def _capability_report() -> dict[str, object]:
        try:
            from backend.core.dsp.model_capability_gate import (  # pylint: disable=import-outside-toplevel
                get_model_capability_gate,
            )

            return dict(get_model_capability_gate().build_report())
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§SMR-1 model capability report nicht verfuegbar: %s", exc)
            return {}

    @staticmethod
    def _capability_status(report: dict[str, object], name: str) -> str:
        try:
            capabilities = report.get("capabilities", {}) if isinstance(report, dict) else {}
            if isinstance(capabilities, dict):
                cap = capabilities.get(name, {})
                if isinstance(cap, dict):
                    return str(cap.get("status", "unknown"))
        except Exception:  # pylint: disable=broad-except
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        return "unknown"

    @classmethod
    def _is_primary_roformer_model(cls, model_used: str) -> bool:
        """True only for actual BS/MelBand-RoFormer inference, not plugin fallbacks."""
        normalized = model_used.lower()
        return normalized in {"bs_roformer", "melbandroformer", "melband_roformer"}

    @classmethod
    def _stems_to_vocal_instr(
        cls,
        stems: Mapping[str, object],
        reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        vocal = cls._coerce_like(stems.get("vocals", np.zeros_like(reference)), reference)
        instr_parts = [
            cls._coerce_like(stem_audio, reference) for stem_name, stem_audio in stems.items() if stem_name != "vocals"
        ]
        if instr_parts:
            instrumental = np.sum(np.stack(instr_parts, axis=0), axis=0).astype(np.float32)
        else:
            instrumental = (np.asarray(reference, dtype=np.float32) - vocal).astype(np.float32)
        instrumental = np.clip(np.nan_to_num(instrumental, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return vocal.astype(np.float32), instrumental.astype(np.float32)

    @staticmethod
    def _coerce_like(candidate: object, reference: np.ndarray) -> np.ndarray:
        """Gibt candidate as finite float32 audio with reference shape/layout zurück."""
        ref = np.asarray(reference, dtype=np.float32)
        try:
            arr = np.asarray(candidate, dtype=np.float32)
        except Exception:  # pylint: disable=broad-except
            return np.zeros_like(ref, dtype=np.float32)  # type: ignore[no-any-return]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2 and ref.ndim == 2 and arr.T.shape == ref.shape:
            arr = arr.T
        if ref.ndim == 1 and arr.ndim == 2:
            arr = arr.mean(axis=1 if arr.shape[0] == ref.shape[0] else 0)
        elif ref.ndim == 2 and arr.ndim == 1:
            arr = np.repeat(arr[:, None], ref.shape[1], axis=1)

        if arr.ndim != ref.ndim:
            return np.zeros_like(ref, dtype=np.float32)  # type: ignore[no-any-return]
        if arr.shape[0] < ref.shape[0]:
            pad_shape = list(arr.shape)
            pad_shape[0] = ref.shape[0] - arr.shape[0]
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        elif arr.shape[0] > ref.shape[0]:
            arr = arr[: ref.shape[0]]

        if arr.shape != ref.shape:
            return np.zeros_like(ref, dtype=np.float32)  # type: ignore[no-any-return]
        return np.clip(arr.astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]


_instance: SotaVocalModelRouter | None = None
_lock = threading.Lock()


def get_sota_vocal_model_router() -> SotaVocalModelRouter:
    """Thread-safe singleton accessor for :class:`SotaVocalModelRouter`."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SotaVocalModelRouter()
    return _instance


__all__ = [
    "EnhancementRouteResult",
    "SotaVocalModelRouter",
    "StemSeparationRouteResult",
    "get_sota_vocal_model_router",
]
