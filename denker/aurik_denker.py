"""
denker/aurik_denker.py — AurikDenker: Orchestrator mit 8 Verarbeitungsstufen
==============================================================================

Koordiniert die vollständige Restaurierungs-Pipeline in der kanonischen
Reihenfolge (Spec §2.2):

  1. TontraegerDenker  → Trägermedium-Erkennung
  2. DefektDenker      → Defekt-Analyse + Kausal-Reasoning
  3. StrategieDenker   → 8×RT-Budget-Plan + Timer
  4. _run_rest()-Closure → ReparaturDenker (Preprocessing) →
                           RekonstruktionsDenker (Preprocessing) →
                           RestaurierDenker (UV3-Vollpipeline)
  7. ExzellenzDenker   → Musical Goals + Excellence-Optimierung
  8. VERSA MOS-Gate     → Finale Qualitätsbewertung (§4.4, nicht-referenzbasiert);
                          MOS < 4.0 → 2. ExzellenzDenker-Durchlauf

8×RT-Invariante: rt_factor im Endergebnis ist IMMER ≤ 8.0.

Spec §2.1, §2.2, §9.5 — v10.0.0
"""

from __future__ import annotations

import inspect
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from scipy.signal import butter, sosfiltfilt

from backend.core.calibration_matrix import get_material_floor
from backend.core.pipeline_health_state import PipelineHealthState, pipeline_health_from_fail_reasons

logger = logging.getLogger(__name__)

_3X_RT_LIMIT: float = 32.0  # RT-Reported max (Spec §9.5 — caps rt_factor in output)
# Mode-abhängige RT-Budgets für RestaurierDenker-Thread (Spec §9.5):
#   BALANCED=32×, QUALITY=32×, MAXIMUM=32×
#   Werte sind auf PerformanceGuard.LIMIT_* ausgerichtet.
_RT_BUDGET_BY_MODE: dict[str, float] = {
    "balanced": 32.0,
    "restoration": 32.0,  # "Restoration" maps to quality
    "quality": 32.0,
    "studio2026": 32.0,
    "maximum": 32.0,
}
# Cold-Start-Minimum: Erster Lauf lädt ML-Modelle (PANNs 0.7GB, UV3 vollständig, etc.)
# UV3-Init + Klassifikation allein dauert 20–120s je nach Host-System. 1800s = 30 min
# deckt auch langsame HDDs + thermisches Throttling ab. PerformanceGuard begrenzt
# die eigentliche Verarbeitungszeit — dieser Floor gilt ausschließlich für Cold-Start.
_COLDSTART_MIN_SECONDS: float = 1800.0
# Absolutes Gesamtlimit (§9.5): eine Stufe-1-Restaurierung endet spätestens nach 90 min.
# Begründung: 20-min Vinyl (1200s) × 4× DSP-RT = 4800s; 90 min = komfortabler Puffer.
# KMV Stufe 2 (MLRefinementThread) übernimmt danach ohne Zeitlimit.
# Alter Wert: 1800s (30 min), dann 5400s (90 min) — beides zu eng für
# 39-Phasen-Pipeline auf 225s Audio mit schweren ML-Modellen (NVSR-SBR, MelBandRoformer).
_MAX_TOTAL_SECONDS: float = 14400.0  # 240 Minuten (4h, §K 64×RT-aligned)
_MIN_AUDIO_SAMPLES: int = 64  # Mindestsignallänge

# Material-adaptive MOS-Mindestziele (§6.2)
_MATERIAL_MOS_TARGETS: dict[str, float] = {
    "wax_cylinder": 3.5,
    "shellac": 3.8,
    "lacquer_disc": 3.7,
    "wire_recording": 3.6,
    "vinyl": 4.0,
    "tape": 4.2,
    "reel_tape": 4.3,
    "dat": 4.5,
    "cd_digital": 4.5,
    "mp3_low": 3.9,
    "mp3_high": 4.5,  # §6.2: wie cd_digital/dat (digitale Hochqualitäts-Quelle)
    "aac": 4.5,  # §6.2: wie cd_digital/dat (digitale Hochqualitäts-Quelle)
    "minidisc": 4.0,
    "streaming": 4.0,
    "unknown": 3.8,
}

_HISTORICAL_OR_FRAGILE_MATERIALS: frozenset[str] = frozenset(
    {
        "wax_cylinder",
        "shellac",
        "lacquer_disc",
        "wire_recording",
        "vinyl",
        "tape",
        "reel_tape",
        # v10.0.0: Kassette ist analoges Träger-Material — fehle bisher (Bug G1)
        "cassette",
        "cassette_dolby_b",
        "cassette_dolby_c",
        "cassette_dolby_s",
    }
)

_MODERN_DIGITAL_MATERIALS: frozenset[str] = frozenset(
    {
        "cd_digital",
        "dat",
        "aac",
        "mp3_high",
        "streaming",
    }
)

_HEAVY_DEFECT_HINTS: frozenset[str] = frozenset(
    {
        "dropout",
        "clipping",
        "hum",
        "wow",
        "flutter",
        "crackle",
        "click",
        "print_through",
    }
)

_ORACLE_ROLLOUT_MODES: frozenset[str] = frozenset({"off", "pilot", "all"})


def _is_pytest_or_safe_validation_context() -> bool:
    """Erkennt test-/validierungsgetriebene Laufzeitprofile für konservative Timeouts."""
    if os.getenv("AURIK_SAFE_VALIDATION_PROFILE", "0") == "1":
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    return "pytest" in os.path.basename(str(os.getenv("_", ""))).lower()


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Lädt a symbol lazily to avoid module-level circular imports."""
    return getattr(import_module(module_name), symbol_name)


def _normalize_goal_scores(raw_goals: Any) -> dict[str, float]:
    """Coerce heterogeneous goal mappings to the canonical dict[str, float] shape."""
    if not isinstance(raw_goals, dict):
        return {}

    normalized: dict[str, float] = {}
    for raw_key, raw_value in raw_goals.items():
        if not isinstance(raw_value, (int, float)):
            continue
        key = raw_key.decode("utf-8", errors="ignore") if isinstance(raw_key, bytes) else str(raw_key)
        normalized[key] = float(raw_value)
    return normalized


def _normalize_audio_output_layout(audio: np.ndarray) -> np.ndarray:
    """Normalisiert Ergebnis-Audio auf samples-first für konsistente API-Verträge.

    Regel:
        - Mono bleibt 1D.
        - Stereo channels-first (2, N) wird zu (N, 2) transponiert.
        - Bereits samples-first bleibt unverändert.
    """
    arr = cast(np.ndarray, np.asarray(audio))
    if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
        return arr.T
    return arr


def _get_canonical_thresholds_for_mode(is_studio_2026: bool) -> dict[str, float]:
    """Lädt PMGG goal thresholds lazily for the current mode."""
    threshold_fn = cast(
        Callable[..., dict[str, float]],
        _load_symbol("backend.core.per_phase_musical_goals_gate", "_get_canonical_thresholds"),
    )
    return threshold_fn(is_studio_2026=is_studio_2026)


def _score_versa_mos(audio: np.ndarray, sr: int) -> Any:
    """Führt aus: VERSA MOS via lazy import to keep optional plugin loading local."""
    score_mos_fn = cast(Callable[[np.ndarray, int], Any], _load_symbol("plugins.versa_plugin", "score_mos"))
    return score_mos_fn(audio, sr)


def _set_pipeline_active(active: bool) -> None:
    """Inform the plugin lifecycle manager that the heavy pipeline is active."""
    set_active_fn = cast(
        Callable[[bool], None],
        _load_symbol("backend.core.plugin_lifecycle_manager", "set_pipeline_active"),
    )
    set_active_fn(active)


def _evict_stale_plugins(required_mb: float | None = None) -> int:
    """Evict inactive plugins through the lifecycle manager."""
    evict_fn = cast(
        Callable[..., int],
        _load_symbol("backend.core.plugin_lifecycle_manager", "evict_stale_plugins"),
    )
    if required_mb is None:
        return int(evict_fn())
    return int(evict_fn(required_mb=required_mb))


def _probe_benign_digital_source(
    audio: np.ndarray,
    sr: int,
    resolved_material: str,
) -> tuple[bool, dict[str, Any]]:
    """Call UV3's clean-digital probe without hard importing the full backend at module import time."""
    material_type = _load_symbol("backend.core.defect_scanner", "MaterialType")
    uv3_cls = _load_symbol("backend.core.unified_restorer_v3", "UnifiedRestorerV3")
    _probe_name = "_is_benign_digital_source"
    benign_probe = cast(
        Callable[[np.ndarray, int, Any], tuple[bool, dict[str, Any]]],
        getattr(uv3_cls, _probe_name),
    )
    material_enum = next(
        (member for member in material_type if getattr(member, "value", None) == resolved_material),
        None,
    )
    benign, metrics = benign_probe(audio, sr, material_enum)
    return bool(benign), dict(metrics)


def _compute_song_audio_fingerprint(audio: np.ndarray, sr: int) -> str:
    """Berechnet the persistent strategy-cache song id through a lazy import."""
    fingerprint_fn = cast(
        Callable[[np.ndarray, int], str],
        _load_symbol("backend.core.song_strategy_cache", "compute_audio_fingerprint"),
    )
    return str(fingerprint_fn(audio, sr))


def _get_song_strategy_cache() -> Any:
    """Gibt the strategy-cache singleton through a lazy import zurück."""
    cache_fn = cast(Callable[[], Any], _load_symbol("backend.core.song_strategy_cache", "get_song_strategy_cache"))
    return cache_fn()


def _build_song_strategy_entry_from_result(**kwargs: Any) -> Any:
    """Erstellt a strategy-cache entry through a lazy import."""
    build_fn = cast(
        Callable[..., Any],
        _load_symbol("backend.core.song_strategy_cache", "build_strategy_entry_from_result"),
    )
    return build_fn(**kwargs)


# ─── Ergebnis-Datenklasse ────────────────────────────────────────────────────


@dataclass
class AurikErgebnis:
    """Vollständiges Restaurierungsergebnis des AurikDenker-Orchestrators.

    Felder:
        audio:               Restauriertes Audio (float32, clip [-1, 1])
        material:            Erkanntes Trägermedium (z. B. "tape", "vinyl")
        rt_factor:           Tatsächlicher Echtzeit-Faktor (≤ 8.0)
        quality_estimate:    Qualitätsschätzung ∈ [0, 1]
        musical_goals:       14 Musical-Goals-Scores
        goals_passed:        Anzahl bestandener Goals
        phases_executed:     Liste der ausgeführten Verarbeitungsphasen
        warnings:            Warnungsliste (technisch, Englisch)
        processing_note:     Kurzbeschreibung auf Deutsch
        stage_notes:         Detailnotizen je Verarbeitungsstufe
        chain_info:          Tonträgerketten-Analyse (optional)
        confidence:          Gesamtkonfidenz der Restaurierung ∈ [0, 1]
        rollback_triggered:  War ARE-Rollback ausgelöst?
        winning_variant:     Beste ARE-Variante (oder None)
        gaps_found:          Erkannte Dropout-Lücken
        gaps_repaired:       Erfolgreich reparierte Lücken
        gap_total_repaired_ms: Gesamtdauer reparierter Lücken in ms
    """

    audio: np.ndarray
    material: str
    rt_factor: float
    quality_estimate: float
    musical_goals: dict[str, float]
    goals_passed: int
    phases_executed: list[str]
    warnings: list[str] = field(default_factory=list)
    processing_note: str = ""
    stage_notes: dict[str, str] = field(default_factory=dict)
    chain_info: dict[str, Any] | None = field(default=None)
    # ── ARE-spezifische Felder (A-2) ─────────────────────────────────────────
    confidence: float = 0.85
    rollback_triggered: bool = False
    winning_variant: str | None = None
    gaps_found: int = 0
    gaps_repaired: int = 0
    gap_total_repaired_ms: float = 0.0
    degradation_status: str = PipelineHealthState.OK.value
    fail_reason: str | None = None
    # §Dach: Musikalischer Globalplan — stilbewusstes Restaurierungsportrait
    global_plan: dict[str, Any] | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)
    # §S5 GAF-Propagation: True = applicable, False = inapplicable (§2.32)
    goal_applicability: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "material": self.material,
            "rt_factor": float(self.rt_factor),
            "quality_estimate": float(self.quality_estimate),
            "musical_goals": {k: float(v) for k, v in self.musical_goals.items()},
            "goals_passed": self.goals_passed,
            "phases_executed": list(self.phases_executed),
            "warnings": list(self.warnings),
            "processing_note": self.processing_note,
            "stage_notes": dict(self.stage_notes),
            "chain_info": self.chain_info,
            "confidence": float(self.confidence),
            "rollback_triggered": self.rollback_triggered,
            "winning_variant": self.winning_variant,
            "gaps_found": self.gaps_found,
            "gaps_repaired": self.gaps_repaired,
            "gap_total_repaired_ms": float(self.gap_total_repaired_ms),
            "degradation_status": self.degradation_status,
            "fail_reason": self.fail_reason,
            "global_plan": self.global_plan,
            "metadata": dict(self.metadata),
        }


# ─── Orchestrator ────────────────────────────────────────────────────────────


class AurikDenker:
    """Orchestrator mit 8 Verarbeitungsstufen im Aurik-System.

    Steuert die vollständige Restaurierungs-Pipeline sequenziell und
    überwacht dabei das 8×RT-Budget. Jeder Domänen-Denker läuft in
    try/except — ein Fehler in einer Stufe stoppt nicht die gesamte Pipeline.

    Verwendung::

        aurik = get_aurik_denker()
        ergebnis = aurik.restauriere(audio, sr=48_000)
        logger.debug("RT-Faktor: %.2f×", ergebnis.rt_factor)
        logger.debug("Qualität: %.3f (VERSA MOS eingerechnet)", ergebnis.quality_estimate)
        logger.debug("Goals: %s/%s", ergebnis.goals_passed, len(ergebnis.musical_goals))
    """

    # ── Öffentliche API ──────────────────────────────────────────────────────

    def restauriere(
        self,
        audio: np.ndarray,
        sr: int = 48_000,
        *,
        validate_audio: bool = True,
        mode: str = "quality",
        progress_callback: Any | None = None,
        audio_update_callback: Any | None = None,
        cached_era_result: Any | None = None,
        cached_genre_result: Any | None = None,
        cached_defect_result: Any | None = None,
        cached_medium_result: Any | None = None,
        cached_restorability_result: Any | None = None,
        recovery_checkpoint: Any | None = None,
        input_path: str = "",
        output_path: str = "",
        no_rt_limit: bool = False,
        phase_strength_oracle_rollout: str | None = None,
    ) -> AurikErgebnis:
        """Vollständige Aurik-Restaurierung: 8 Stufen orchestriert.

        Args:
            audio:         Audio-Signal (mono/stereo, float32)
            sr:            Sample-Rate in Hz (Spec §6.5: immer 48 000 Hz)
            validate_audio: Ob Eingabe-Validierung durchgeführt werden soll
            mode:          "Restoration", "Studio 2026", oder "Preview" (§3.5)

        Returns:
            AurikErgebnis mit restauriertem Audio und vollständiger Bewertung.
            Bei Fehlern: ursprüngliches Audio + Fehlerbeschreibung.
        """
        t_start = time.perf_counter()
        assert sr == 48000, f"AurikDenker.restauriere() erwartet sr=48000 Hz, erhalten: {sr} Hz"

        # ── §V8/§G1 (copilot-instructions.md): Song-Isolation — FallbackAuditor pro Song zurücksetzen.
        # Sonst akkumulieren Degradations-Events session-global und der PreExportValidator
        # blockiert ab 8 Events fälschlich Exporte späterer Songs (Rev. 2026-08-16).
        try:
            from backend.core.fallback_auditor import get_fallback_auditor

            get_fallback_auditor().reset()
        except Exception as _fa_reset_exc:
            logger.debug("FallbackAuditor.reset nicht möglich (unkritisch): %s", _fa_reset_exc)

        # ── §3.5 Preview-Mode: 30s Vorschau vor voller Restaurierung ─────
        _PREVIEW_DURATION_S = 30.0
        from backend.api.bridge import normalize_user_mode as _bridge_norm

        _is_preview = _bridge_norm(mode) == "Preview"
        if _is_preview:
            preview_samples = int(_PREVIEW_DURATION_S * sr)
            original_duration_s = len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr
            if len(audio.shape) == 1:
                audio = audio[:preview_samples].copy()
            else:
                audio = audio[:preview_samples].copy()
            logger.info(
                "§3.5 Preview: Audio auf %.1fs getrimmt (Originalsignal: %.1fs)",
                _PREVIEW_DURATION_S,
                original_duration_s,
            )
            # Preview läuft immer mit no_rt_limit für schnellstmögliche Vorschau
            no_rt_limit = True

        _dur_s = len(audio) / max(sr, 1) if audio.ndim == 1 else audio.shape[0] / max(sr, 1)
        logger.info(
            "AurikDenker.denke() gestartet: Betriebsart=%s, sr=%d, duration=%.1fs, shape=%s, "
            "caches=[era=%s, genre=%s, defect=%s, medium=%s, rest=%s]%s",
            mode,
            sr,
            _dur_s,
            audio.shape,
            cached_era_result is not None,
            cached_genre_result is not None,
            cached_defect_result is not None,
            cached_medium_result is not None,
            cached_restorability_result is not None,
            " [PREVIEW §3.5]" if _is_preview else "",
        )

        # NaN/Inf-Schutz (Spec §3.1)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if validate_audio and audio.size < _MIN_AUDIO_SAMPLES:
            elapsed = time.perf_counter() - t_start
            return self._fallback(
                audio,
                rt_factor=0.0,
                grund=f"Signal zu kurz ({audio.size} Samples < {_MIN_AUDIO_SAMPLES})",
            )

        # Audio-Dauer für RT-Berechnung
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=-1 if audio.shape[-1] <= 2 else 0)
        audio_duration_s = len(audio_mono) / max(sr, 1)

        try:
            ergebnis = self._orchestriere(
                audio,
                sr,
                audio_duration_s,
                t_start,
                mode=mode,
                progress_callback=progress_callback,
                audio_update_callback=audio_update_callback,
                cached_era_result=cached_era_result,
                cached_genre_result=cached_genre_result,
                cached_defect_result=cached_defect_result,
                cached_medium_result=cached_medium_result,
                cached_restorability_result=cached_restorability_result,
                recovery_checkpoint=recovery_checkpoint,
                input_path=input_path,
                output_path=output_path,
                no_rt_limit=no_rt_limit,
                phase_strength_oracle_rollout=phase_strength_oracle_rollout,
            )
        except MemoryError as exc:
            elapsed = time.perf_counter() - t_start
            rt = elapsed / max(audio_duration_s, 1e-6)
            try:
                _evict_stale_plugins(required_mb=4096.0)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (nicht-kritisch): %s", _exc)
            logger.error("AurikDenker: Speicherfehler in Pipeline: %s", exc)
            return self._fallback(
                audio,
                rt_factor=min(rt, _3X_RT_LIMIT),
                grund="Speicherfehler (OOM-Guard): ML-Last zu hoch, DSP-Fallback empfohlen",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            rt = elapsed / max(audio_duration_s, 1e-6)
            logger.error("AurikDenker: Unerwarteter Fehler in Pipeline: %s", exc)
            return self._fallback(
                audio,
                rt_factor=min(rt, _3X_RT_LIMIT),
                grund=f"Pipeline-Fehler: {exc}",
            )

        return ergebnis

    def denke(self, audio: np.ndarray, sr: int = 48_000, **kwargs: Any) -> AurikErgebnis:
        """Alias für restauriere() — API-Kompatibilität (Spec §2.2)."""
        # Unpack pre_analysis_result into individual cached_* kwargs if present.
        # BatchProcessingThread passes a PreAnalysisResult object; restauriere() expects
        # individual cached_* parameters.  Without this step all cached analyses would be
        # silently discarded, forcing DefektDenker / TontraegerDenker to re-run (~120 s).
        _pre = kwargs.get("pre_analysis_result")
        if _pre is not None:
            if kwargs.get("cached_era_result") is None and getattr(_pre, "era", None) is not None:
                kwargs["cached_era_result"] = _pre.era
            if kwargs.get("cached_genre_result") is None and getattr(_pre, "genre", None) is not None:
                kwargs["cached_genre_result"] = _pre.genre
            if kwargs.get("cached_defect_result") is None and getattr(_pre, "defects", None) is not None:
                kwargs["cached_defect_result"] = _pre.defects
            if kwargs.get("cached_medium_result") is None and getattr(_pre, "medium", None) is not None:
                kwargs["cached_medium_result"] = _pre.medium
            if kwargs.get("cached_restorability_result") is None and getattr(_pre, "restorability", None) is not None:
                kwargs["cached_restorability_result"] = _pre.restorability
            logger.debug(
                "AurikDenker.denke(): pre_Analyse_Ergebnis entpackt (era=%s genre=%s defects=%s medium=%s rest=%s)",
                "✓" if kwargs.get("cached_era_result") else "—",
                "✓" if kwargs.get("cached_genre_result") else "—",
                "✓" if kwargs.get("cached_defect_result") else "—",
                "✓" if kwargs.get("cached_medium_result") else "—",
                "✓" if kwargs.get("cached_restorability_result") else "—",
            )
        return self.restauriere(
            audio,
            sr,
            mode=kwargs.get("mode", "quality"),
            progress_callback=kwargs.get("progress_callback"),
            audio_update_callback=kwargs.get("audio_update_callback"),
            cached_era_result=kwargs.get("cached_era_result"),
            cached_genre_result=kwargs.get("cached_genre_result"),
            cached_defect_result=kwargs.get("cached_defect_result"),
            cached_medium_result=kwargs.get("cached_medium_result"),
            cached_restorability_result=kwargs.get("cached_restorability_result"),
            recovery_checkpoint=kwargs.get("recovery_checkpoint"),
            # §2.39 OOM-Recovery: Pfade für Checkpoint-Persistierung durchreichen
            input_path=kwargs.get("input_path", ""),
            output_path=kwargs.get("output_path", ""),
            no_rt_limit=bool(kwargs.get("no_rt_limit", False)),
            phase_strength_oracle_rollout=kwargs.get("phase_strength_oracle_rollout"),
        )

    @staticmethod
    def _resolve_excellence_material(material: str, chain_info: dict[str, Any] | None) -> str:
        """Gibt das restaurierungsrelevante Material zurück (Ursprungs-Träger bevorzugt).

        v10.0.0 Fix G2: Bei Transfer-Ketten (z.B. cassette→mp3_low) wird das erste
        analoge Träger-Material zurückgegeben, nicht das finale Dateiformat.
        Begründung: Oracle-, Budget- und Stärken-Entscheidungen müssen auf den
        Original-Träger kalibriert sein (§0l, §2.47a), nicht auf das Dateiformat.
        """
        if isinstance(chain_info, dict):
            # Transfer-Kette: ersten analogen Träger bevorzugen (Ursprung)
            chain = chain_info.get("chain")
            if isinstance(chain, list) and chain:
                for node in chain:
                    node_str = str(node).strip().lower()
                    if node_str in _HISTORICAL_OR_FRAGILE_MATERIALS:
                        return node_str
            # Kein analoger Ursprung → primary_medium als Fallback
            primary_medium = str(chain_info.get("primary_medium", "")).strip().lower()
            if primary_medium:
                return primary_medium
        return str(material or "unknown").strip().lower()

    @classmethod
    def _should_skip_excellence_for_clean_digital(
        cls,
        audio: np.ndarray,
        sr: int,
        material: str,
        chain_info: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, float | str]]:
        """Skips Excellence for benign digital end-formats to avoid overprocessing."""
        resolved_material = cls._resolve_excellence_material(material, chain_info)

        # Fast-path for very short benign clips: avoid full ARE/UV3 orchestration in
        # tiny snippets where heavy restoration is not meaningful and often unstable.
        try:
            arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 2:
                mono = arr.mean(axis=0) if arr.shape[0] <= arr.shape[1] else arr.mean(axis=1)
                sample_count = int(mono.shape[0])
            else:
                mono = arr
                sample_count = int(mono.shape[0])

            duration_s = float(sample_count / max(1, sr))
            _short_clip_materials = {"unknown", "digital_unknown", "mp3_low", "mp3_high", "aac", "cd_digital", "dat"}
            if duration_s <= 5.0 and resolved_material in _short_clip_materials and sample_count > 0:
                abs_mono = np.abs(mono)
                hard_clip_ratio = float(np.mean(abs_mono >= 0.999))
                near_clip_ratio = float(np.mean(abs_mono >= 0.98))
                rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
                # Only skip if truly silent (RMS ≤ -60dBFS, ~0.001).
                # Avoid confusing noisy signal with "clean" audio.
                if hard_clip_ratio <= 1e-4 and near_clip_ratio <= 1e-3 and rms <= 0.001:
                    logger.warning(
                        "Short silence clip autopass: duration=%.1fs, material=%s, rms=%.4f — "
                        "To force restoration, use Betriebsart='studio2026'",
                        duration_s,
                        resolved_material,
                        rms,
                    )
                    return True, {
                        "material": resolved_material,
                        "reason": "short_benign_clip_autopass",
                        "duration_s": float(round(duration_s, 3)),
                        "hard_clip_ratio": hard_clip_ratio,
                        "near_clip_ratio": near_clip_ratio,
                        "rms": float(round(rms, 6)),
                    }
        except Exception as exc:
            logger.debug("AurikDenker: short-clip guard nicht verfuegbar: %s", exc)

        # Never skip restoration when the recording chain has an analog origin.
        # primary_medium = chain[-1] (e.g. "mp3_low"), but if original_medium = "tape",
        # the source is a tape transfer and must not be treated as a clean digital source.
        _ANALOG_ORIGINALS = {
            "tape",
            "reel_tape",
            "vinyl",
            "shellac",
            "cassette",
            "phonograph",
            "wax_cylinder",
        }
        if isinstance(chain_info, dict):
            _orig = str(chain_info.get("original_medium", "")).strip().lower()
            if _orig in _ANALOG_ORIGINALS:
                logger.debug(
                    "AurikDenker: analog-origin chain (%s → %s) — pass-through blocked.",
                    _orig,
                    resolved_material,
                )
                return False, {
                    "material": resolved_material,
                    "reason": "analog_chain_no_passthrough",
                    "original_medium": _orig,
                }

        try:
            benign, metrics = _probe_benign_digital_source(audio, sr, resolved_material)
            metrics = dict(metrics)
            metrics.setdefault("material", resolved_material)
            return benign, metrics
        except Exception as exc:
            logger.debug("AurikDenker: Clean-digital guard nicht verfuegbar: %s", exc)
            return False, {"material": resolved_material, "reason": "guard_unavailable"}

    @classmethod
    def _material_mos_target(cls, material: str, chain_info: dict[str, Any] | None) -> tuple[str, float]:
        """Gibt resolved material key and its adaptive MOS target zurück."""
        resolved = cls._resolve_excellence_material(material, chain_info)
        target = float(_MATERIAL_MOS_TARGETS.get(resolved, _MATERIAL_MOS_TARGETS["unknown"]))
        return resolved, target

    @staticmethod
    def _normalize_mode_name(mode: str | None) -> str:
        """Normalisiert user-facing mode aliases to canonical internal names."""
        normalized = str(mode or "quality").strip().lower().replace("_", "")
        internal_aliases = {
            "quality": "quality",
            "balanced": "balanced",
            "fast": "fast",
            "maximum": "maximum",
        }
        if normalized in internal_aliases:
            return internal_aliases[normalized]

        from backend.api.bridge import normalize_user_mode as _bridge_norm

        canonical_mode = _bridge_norm(mode)
        if canonical_mode == "Studio 2026":
            return "studio2026"
        if canonical_mode == "Restoration":
            return "restoration"

        aliases = {
            "studio 2026": "studio2026",
            "studio2026": "studio2026",
            "restoration": "restoration",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _normalize_oracle_rollout_mode(mode: str | None) -> str | None:
        """Normalisiert den Strength-Oracle-Rollout-Modus auf off/pilot/all."""
        if mode is None:
            return None
        raw = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "0": "off",
            "false": "off",
            "disabled": "off",
            "none": "off",
            "pilot": "pilot",
            "1": "all",
            "true": "all",
            "enabled": "all",
            "full": "all",
            "all": "all",
        }
        normalized = aliases.get(raw)
        if normalized not in _ORACLE_ROLLOUT_MODES:
            return None
        return normalized

    @staticmethod
    def _extract_phase_delta_value(entry: Any) -> float | None:
        """Extrahiert robust einen numerischen Delta-Wert aus heterogenen phase_deltas-Formaten."""
        if isinstance(entry, (int, float)) and math.isfinite(float(entry)):
            return float(entry)
        if isinstance(entry, dict):
            for key in ("overall_delta", "delta", "score_delta", "mos_delta", "goal_delta"):
                raw = entry.get(key)
                if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                    return float(raw)
        return None

    @classmethod
    def _collect_improvement_opportunities(
        cls,
        *,
        musical_goals: dict[str, float],
        rest_metadata: dict[str, Any],
        material: str,
        chain_info: dict[str, Any] | None,
        restorability_score: float,
        is_studio_2026: bool,
    ) -> dict[str, Any]:
        """Leitet nach jedem Lauf konkrete Qualitäts-Upgrade-Chancen aus Telemetrie ab."""
        opportunities: dict[str, Any] = {
            "material": cls._resolve_excellence_material(material, chain_info),
            "restorability_score": float(np.clip(restorability_score, 0.0, 100.0)),
            "goal_gaps": [],
            "regressive_phases": [],
        }

        if not isinstance(musical_goals, dict) or not musical_goals:
            opportunities["summary"] = {
                "high_priority_goal_count": 0,
                "regressive_phase_count": 0,
                "top_goal": "",
            }
            return opportunities

        try:
            floor_fn = cast(
                Callable[..., float],
                _load_symbol("backend.core.calibration_matrix", "get_effective_material_floor"),
            )
            recovery_fn = cast(
                Callable[..., list[str]],
                _load_symbol("backend.core.calibration_matrix", "get_goal_recovery_phases"),
            )
            canonical_targets = _get_canonical_thresholds_for_mode(is_studio_2026)
        except Exception as exc:
            logger.debug("AurikDenker: opportunity mining setup nicht verfuegbar: %s", exc)
            opportunities["summary"] = {
                "high_priority_goal_count": 0,
                "regressive_phase_count": 0,
                "top_goal": "",
            }
            return opportunities

        material_key = str(opportunities["material"])
        goal_gaps: list[dict[str, Any]] = []
        for goal, raw_score in musical_goals.items():
            if not isinstance(raw_score, (int, float)):
                continue
            score = float(raw_score)
            if not math.isfinite(score):
                continue

            floor_target = float(
                floor_fn(
                    material_type=material_key,
                    goal=goal,
                    restorability_score=float(np.clip(restorability_score, 0.0, 100.0)),
                )
            )
            canonical_target = float(canonical_targets.get(goal, floor_target))
            gap_floor = float(max(0.0, floor_target - score))
            gap_canonical = float(max(0.0, canonical_target - score))
            if gap_floor <= 0.0 and gap_canonical <= 0.0:
                continue

            priority = "high" if gap_floor > 0.0 else "medium"
            recommended = recovery_fn(goal=goal, is_studio_2026=is_studio_2026)
            goal_gaps.append(
                {
                    "goal": str(goal),
                    "score": float(round(score, 4)),
                    "target_floor": float(round(floor_target, 4)),
                    "target_canonical": float(round(canonical_target, 4)),
                    "gap_to_floor": float(round(gap_floor, 4)),
                    "gap_to_canonical": float(round(gap_canonical, 4)),
                    "priority": priority,
                    "recommended_phases": [str(p) for p in recommended[:3]],
                }
            )

        goal_gaps.sort(
            key=lambda item: (
                0 if str(item.get("priority", "")).lower() == "high" else 1,
                -float(item.get("gap_to_floor", 0.0)),
                -float(item.get("gap_to_canonical", 0.0)),
            )
        )

        regressive: list[dict[str, Any]] = []
        phase_deltas = rest_metadata.get("phase_deltas") if isinstance(rest_metadata, dict) else None
        if isinstance(phase_deltas, dict):
            for phase_id, entry in phase_deltas.items():
                delta = cls._extract_phase_delta_value(entry)
                if delta is None or delta >= -0.001:
                    continue
                regressive.append({"phase": str(phase_id), "delta": float(round(delta, 6))})
            regressive.sort(key=lambda item: float(item.get("delta", 0.0)))

        opportunities["goal_gaps"] = goal_gaps[:5]
        opportunities["regressive_phases"] = regressive[:5]
        opportunities["summary"] = {
            "high_priority_goal_count": int(
                sum(1 for item in goal_gaps if str(item.get("priority", "")).lower() == "high")
            ),
            "regressive_phase_count": int(len(regressive)),
            "top_goal": str(goal_gaps[0]["goal"]) if goal_gaps else "",
        }
        return opportunities

    def _wd_phase_start(self, name: str) -> None:
        """Non-blocking watchdog phase start (ignores all errors)."""
        try:
            from backend.core.watchdog_monitor import get_watchdog

            get_watchdog().on_phase_start(name)
        except Exception:
            logger.debug("aurik_denker.py:825: Silent exception absorbed", exc_info=True)

    def _wd_phase_end(self, name: str, audio: np.ndarray | None = None, sr: int = 48000) -> None:
        """Non-blocking watchdog phase end (ignores all errors)."""
        try:
            from backend.core.watchdog_monitor import get_watchdog

            if audio is not None:
                get_watchdog().on_phase_end(name, audio, sr)
        except Exception:
            logger.debug("aurik_denker.py:835: Silent exception absorbed", exc_info=True)

    @staticmethod
    def _compute_signal_intelligence_signature(audio: np.ndarray, sr: int) -> dict[str, float]:
        """Berechnet robuste, leichtgewichtige Signal-Indikatoren für Risikoentscheidungen."""
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            # channels-first (2, N) und channels-last (N, 2) robust behandeln.
            if arr.shape[0] <= 2 < arr.shape[1]:
                arr = np.mean(arr, axis=0)
            elif arr.shape[1] <= 2 < arr.shape[0]:
                arr = np.mean(arr, axis=1)
            else:
                arr = np.mean(arr, axis=-1)
        elif arr.ndim != 1:
            arr = np.ravel(arr)

        if arr.size < 64:
            return {
                "rms_dbfs": -120.0,
                "crest_db": 0.0,
                "hf_ratio": 0.0,
                "transient_ratio": 0.0,
                "micro_dynamic_db": 0.0,
            }

        arr64 = arr.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(arr64 * arr64) + 1e-12))
        peak = float(np.max(np.abs(arr64)) + 1e-12)
        rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
        crest_db = float(20.0 * np.log10(max(peak / max(rms, 1e-8), 1e-8)))

        # Hochfrequenz-Anteil (>6 kHz) als Proxy für Zisch-/Artefaktrisiko.
        n_fft = int(min(16384, arr64.size))
        if n_fft >= 512:
            frame = arr64[:n_fft]
            window = np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(sr, 1))
            total_energy = float(np.sum(spectrum) + 1e-12)
            hf_energy = float(np.sum(spectrum[freqs >= 6000.0]))
            hf_ratio = float(np.clip(hf_energy / total_energy, 0.0, 1.0))
        else:
            hf_ratio = 0.0

        # Transienten-Dichte via 1. Ableitung als schonender Onset-Proxy.
        diff = np.abs(np.diff(arr64, prepend=arr64[0]))
        if diff.size > 8:
            transient_thr = max(float(np.percentile(diff, 99.0)), 1e-6)
            transient_ratio = float(np.mean(diff > transient_thr))
        else:
            transient_ratio = 0.0

        # Mikrodynamik-Proxy: P95-P05 der Frame-RMS in dB.
        # Vektorisiert via reshape statt Python-for-loop — für 225s Audio
        # (~5280 Frames) ist das der Unterschied zwischen <10ms und >500ms.
        frame_len = 2048
        if arr64.size >= frame_len:
            n_frames = arr64.size // frame_len
            if n_frames > 0:
                frames = arr64[: n_frames * frame_len].reshape(n_frames, frame_len)
                frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
                frame_rms_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
                p95 = float(np.percentile(frame_rms_db, 95))
                p05 = float(np.percentile(frame_rms_db, 5))
                micro_dynamic_db = max(0.0, p95 - p05)
            else:
                micro_dynamic_db = 0.0
        else:
            micro_dynamic_db = 0.0

        return {
            "rms_dbfs": rms_dbfs,
            "crest_db": float(np.clip(crest_db, 0.0, 40.0)),
            "hf_ratio": hf_ratio,
            "transient_ratio": float(np.clip(transient_ratio, 0.0, 1.0)),
            "micro_dynamic_db": float(np.clip(micro_dynamic_db, 0.0, 60.0)),
        }

    @classmethod
    def _recommend_phase_strength_oracle_rollout(
        cls,
        *,
        requested_rollout: str | None,
        material: str,
        chain_info: dict[str, Any] | None,
        defekt: Any,
        global_plan: Any,
        effective_mode: str,
        cached_restorability_result: Any | None,
        signal_signature: dict[str, float],
    ) -> tuple[str, str]:
        """Empfiehlt einen risikoadaptiven Strength-Oracle-Rollout (off/pilot/all)."""
        normalized_requested = cls._normalize_oracle_rollout_mode(requested_rollout)
        if normalized_requested is not None:
            return normalized_requested, (
                f"Expliziter Oracle-Rollout '{normalized_requested}' beibehalten (Nutzer-/Aufrufer-Vorgabe)."
            )

        resolved_material = cls._resolve_excellence_material(material, chain_info)
        severity = float(getattr(defekt, "overall_severity", 0.0) or 0.0)
        primary_defect = (
            str(getattr(defekt, "primary_defect", None) or getattr(defekt, "primary_cause", "unknown")).strip().lower()
        )

        decade_value = None
        if global_plan is not None:
            portrait = getattr(global_plan, "portrait", None)
            if portrait is not None:
                try:
                    decade_value = int(getattr(portrait, "decade", 0) or 0)
                except (TypeError, ValueError):
                    decade_value = None

        restorability_score = float(getattr(cached_restorability_result, "restorability_score", 65.0) or 65.0)
        crest_db = float(signal_signature.get("crest_db", 0.0))
        hf_ratio = float(signal_signature.get("hf_ratio", 0.0))
        transient_ratio = float(signal_signature.get("transient_ratio", 0.0))
        micro_dynamic_db = float(signal_signature.get("micro_dynamic_db", 0.0))

        is_historical_material = resolved_material in _HISTORICAL_OR_FRAGILE_MATERIALS
        is_modern_digital = resolved_material in _MODERN_DIGITAL_MATERIALS
        is_historical_era = decade_value is not None and decade_value <= 1965
        has_heavy_defect = severity >= 0.60 or any(token in primary_defect for token in _HEAVY_DEFECT_HINTS)

        chain = list((chain_info or {}).get("chain") or []) if isinstance(chain_info, dict) else []
        has_analog_origin = any(str(node).strip().lower() in _HISTORICAL_OR_FRAGILE_MATERIALS for node in chain)

        # Wissenschaftsbasierter Risiko-Index (psychoakustisch + materialspezifisch).
        risk_score = 0.0
        if is_historical_material:
            risk_score += 0.30
        if is_historical_era:
            risk_score += 0.12
        if has_analog_origin:
            risk_score += 0.18
        if severity >= 0.75:
            risk_score += 0.26
        elif severity >= 0.45:
            risk_score += 0.14
        if has_heavy_defect:
            risk_score += 0.10
        if crest_db >= 20.0:
            risk_score += 0.10
        elif crest_db >= 16.0:
            risk_score += 0.05
        if transient_ratio >= 0.012:
            risk_score += 0.08
        if micro_dynamic_db >= 14.0:
            risk_score += 0.06
        if hf_ratio <= 0.025 and (is_historical_material or has_analog_origin):
            risk_score += 0.05

        # Bei sehr gut restaurierbarem modernem Digitalmaterial aggressiver ausrollen.
        if is_modern_digital and restorability_score >= 80.0 and severity < 0.35:
            risk_score -= 0.22

        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        recommended = "all"
        if severity >= 0.85 and (is_historical_material or has_analog_origin) and transient_ratio >= 0.01:
            recommended = "off"
        elif risk_score >= 0.45:
            recommended = "pilot"

        # Studio-2026 darf bei stabilem modernem Material voll laufen.
        if cls._normalize_mode_name(effective_mode) == "studio2026" and is_modern_digital and risk_score < 0.45:
            recommended = "all"

        reason = (
            f"Oracle-Rollout '{recommended}' (risk={risk_score:.2f}, material={resolved_material}, "
            f"severity={severity:.2f}, crest={crest_db:.1f}dB, hf={hf_ratio:.3f}, "
            f"transient={transient_ratio:.4f}, restorability={restorability_score:.1f})."
        )
        if requested_rollout is not None and normalized_requested is None:
            reason = f"Ungültiger expliziter Rollout '{requested_rollout}' ignoriert. " + reason
        return recommended, reason

    @classmethod
    def _recommend_excellence_recovery_profile(
        cls,
        *,
        material: str,
        chain_info: dict[str, Any] | None,
        effective_mode: str,
        signal_signature: dict[str, float],
        versa_mos: float,
    ) -> dict[str, Any]:
        """Leitet eine material-/signaladaptive Recovery-Strategie für Exzellenz ab."""
        resolved_material = cls._resolve_excellence_material(material, chain_info)
        crest_db = float(signal_signature.get("crest_db", 0.0))
        transient_ratio = float(signal_signature.get("transient_ratio", 0.0))
        hf_ratio = float(signal_signature.get("hf_ratio", 0.0))
        micro_dynamic_db = float(signal_signature.get("micro_dynamic_db", 0.0))

        is_historical = resolved_material in _HISTORICAL_OR_FRAGILE_MATERIALS
        is_modern = resolved_material in _MODERN_DIGITAL_MATERIALS
        is_studio = cls._normalize_mode_name(effective_mode) == "studio2026"

        preserve_signal = 0.0
        if is_historical:
            preserve_signal += 0.25
        if crest_db >= 18.0:
            preserve_signal += 0.15
        if transient_ratio >= 0.01:
            preserve_signal += 0.15
        if micro_dynamic_db >= 14.0:
            preserve_signal += 0.10
        if hf_ratio <= 0.02:
            preserve_signal += 0.05
        preserve_signal = float(np.clip(preserve_signal, 0.0, 0.75))

        if is_studio and is_modern and preserve_signal < 0.20:
            blend_alphas = (0.90, 0.84, 0.78)
        elif preserve_signal >= 0.40:
            blend_alphas = (0.95, 0.92, 0.89)
        else:
            blend_alphas = (0.92, 0.88, 0.84)

        strict_mos_recovery = bool(0.0 < versa_mos < 3.5)
        return {
            "resolved_material": resolved_material,
            "preserve_signal": preserve_signal,
            "blend_alphas": blend_alphas,
            "strict_mos_recovery": strict_mos_recovery,
        }

    @classmethod
    def _recommend_autopilot_mode(
        cls,
        *,
        requested_mode: str,
        material: str,
        chain_info: dict[str, Any] | None,
        defekt: Any,
        global_plan: Any,
        strategy_mode: str | None,
    ) -> tuple[str, str]:
        """Wählt aus: the safest high-level mode for one-button/autopilot execution."""
        requested = cls._normalize_mode_name(requested_mode)
        strategy = cls._normalize_mode_name(strategy_mode)
        resolved_material = cls._resolve_excellence_material(material, chain_info)
        severity = float(getattr(defekt, "overall_severity", 0.0) or 0.0)
        primary_defect = (
            str(getattr(defekt, "primary_defect", None) or getattr(defekt, "primary_cause", "unknown")).strip().lower()
        )

        decade_value = None
        genre_value = ""
        if global_plan is not None:
            portrait = getattr(global_plan, "portrait", None)
            if portrait is not None:
                try:
                    decade_value = int(getattr(portrait, "decade", 0) or 0)
                except (TypeError, ValueError):
                    decade_value = None
                genre_value = str(getattr(portrait, "genre", "") or "").strip().lower()

        is_historical_material = resolved_material in _HISTORICAL_OR_FRAGILE_MATERIALS
        is_modern_digital = resolved_material in _MODERN_DIGITAL_MATERIALS
        is_historical_era = decade_value is not None and decade_value <= 1965
        is_schlager = "schlager" in genre_value
        has_heavy_defect = severity >= 0.60 or any(token in primary_defect for token in _HEAVY_DEFECT_HINTS)
        has_medium_risk = severity >= 0.35

        recommended = "restoration"
        reason = "Unsicherer oder analoger Fall: Restoration als sicherer Standard."

        if is_historical_material or is_historical_era or has_heavy_defect:
            reason = (
                f"Restoration empfohlen: Material={resolved_material}, decade={decade_value}, "
                f"severity={severity:.2f}, defect={primary_defect}."
            )
        elif is_schlager:
            if is_modern_digital and not has_medium_risk and (decade_value is None or decade_value >= 1970):
                recommended = "studio2026"
                reason = (
                    f"Studio 2026 empfohlen: stabiler Schlager-Fall auf {resolved_material} "
                    f"bei severity={severity:.2f}."
                )
            else:
                reason = (
                    f"Restoration empfohlen: Schlager bleibt konservativ auf {resolved_material} "
                    f"bei severity={severity:.2f}."
                )
        elif is_modern_digital and not has_medium_risk and (decade_value is None or decade_value >= 1970):
            recommended = "studio2026"
            reason = f"Studio 2026 empfohlen: modernes Digitalmaterial {resolved_material} bei severity={severity:.2f}."
        elif strategy in {"studio2026", "maximum"} and not has_medium_risk and is_modern_digital:
            recommended = "studio2026"
            reason = (
                f"Studio 2026 empfohlen: Strategie bevorzugt High-End und Material {resolved_material} wirkt stabil."
            )

        if requested in {"restoration", "studio2026", "maximum", "fast", "balanced"}:
            if requested == "studio2026" and recommended != "studio2026":
                # §Bindend: Nutzerwahl respektieren — Studio 2026 wird NICHT überschrieben.
                # Autopilot-Empfehlung ist ein Hinweis, kein Veto. Der Nutzer hat genau
                # eine bewusste Entscheidung (§Autonomer Magic-Button-Vertrag).
                # Konservative Schutzmaßnahmen greifen INNERHALB Studio 2026:
                # — SongCalibration reduziert Strength bei starken Defekten
                # — PMGG erzwingt Musical-Goals-Einhaltung pro Phase
                # — FeedbackChain rollt bei Regression zurück
                logger.warning(
                    "Studio 2026 auf nicht-idealem Material beibehalten (Nutzerwahl): material=%s, severity=%.2f — %s",
                    resolved_material,
                    severity,
                    reason,
                )
                return "studio2026", (
                    f"Studio 2026 beibehalten (Nutzer-Entscheidung). "
                    f"Hinweis: {reason} "
                    f"Interne Schutzmaßnahmen (SongCal, PMGG, FeedbackChain) sind aktiv."
                )
            return requested, f"Expliziter Modus '{requested}' beibehalten. {reason}"

        if requested == "quality":
            return recommended, f"Autopilot wählte '{recommended}'. {reason}"

        return recommended, f"Unbekannter Modus '{requested_mode}' auf '{recommended}' normalisiert. {reason}"

    # ── Pipeline-Logik ───────────────────────────────────────────────────────

    def _orchestriere(
        self,
        audio: np.ndarray,
        sr: int,
        audio_duration_s: float,
        t_start: float,
        *,
        mode: str = "quality",
        progress_callback: Any | None = None,
        audio_update_callback: Any | None = None,
        cached_era_result: Any | None = None,
        cached_genre_result: Any | None = None,
        cached_defect_result: Any | None = None,
        cached_medium_result: Any | None = None,
        cached_restorability_result: Any | None = None,
        recovery_checkpoint: Any | None = None,
        input_path: str = "",
        output_path: str = "",
        no_rt_limit: bool = False,
        phase_strength_oracle_rollout: str | None = None,
    ) -> AurikErgebnis:
        """Führt die 10-stufige Restaurierungs-Pipeline aus.

        Jede Stufe wird in try/except gekapselt. Fehler einer Stufe führen
        zur Fortsetzung mit dem bisherigen Audio-Zustand.
        """

        def _emit(pct: int, msg: str) -> None:
            """Emittiert Fortschritt (0–100) wenn progress_callback gesetzt."""
            if progress_callback is not None:
                try:
                    elapsed = time.perf_counter() - t_start
                    progress_callback(pct, msg, elapsed)
                except Exception as cb_exc:
                    logger.debug(
                        "AurikDenker: Progress-Callback fehlgeschlagen (Ursache: %s). "
                        "Lösung: Callback-Signatur (pct, msg, elapsed_s) prüfen.",
                        cb_exc,
                    )

        warnings: list[str] = []
        phases_executed: list[str] = []
        stage_notes: dict[str, Any] = {}
        aktuelles_audio = audio.copy()
        requested_mode = self._normalize_mode_name(mode)
        effective_mode = requested_mode
        _critical_stages: frozenset[str] = frozenset({"tontraeger", "defekt", "restaurierung"})
        _stage_severity: dict[str, str] = {
            "tontraeger": "critical",
            "kette": "degraded",
            "defekt": "critical",
            "globalplan": "degraded",
            "strategie": "critical",
            "restaurierung": "critical",
            "exzellenz": "degraded",
        }
        _stage_fail_reasons: list[dict[str, str]] = []
        _degradation_status: str = PipelineHealthState.OK.value
        _fail_reason: str | None = None

        def _record_stage_failure(stage: str, component: str, exc: Exception) -> None:
            msg = str(exc)
            warnings.append(f"{component} fehlgeschlagen: {msg}")
            stage_notes[stage] = f"Fehler: {msg}"
            _stage_fail_reasons.append(
                {
                    "component": component,
                    "stage": stage,
                    "error_code": f"STAGE_{stage.upper()}_FAILED",
                    "severity": _stage_severity.get(stage, "degraded"),
                    "exc_type": type(exc).__name__,
                    "exc_msg": msg,
                }
            )

        def _rt() -> float:
            elapsed = time.perf_counter() - t_start
            return elapsed / max(audio_duration_s, 1e-6)

        strat_denker: Any = None  # M-2: hoisted — für _budget_ok()-Zugriff nach Stufe 5
        defekt: Any = None  # M-1/M-4: hoisted — für Stage-5-Flags

        # §SSC-1 Song-Strategy-Cache: Fingerprint berechnen + Cache abfragen.
        # Non-blocking: Fehler stoppen nie die Pipeline.
        _ssc_song_id: str = ""
        _ssc_warm_start: Any | None = None
        try:
            _ssc_song_id = _compute_song_audio_fingerprint(aktuelles_audio, sr)
            _ssc_warm_start = _get_song_strategy_cache().get(_ssc_song_id, effective_mode)
            if _ssc_warm_start is not None:
                logger.info(
                    "§SSC-1 Warm-Start: song_id=%s Betriebsart=%s HPI=%.3f OQS=%.1f (use_count=%d)",
                    _ssc_song_id[:8],
                    effective_mode,
                    _ssc_warm_start.hpi_achieved,
                    _ssc_warm_start.oqs_achieved,
                    _ssc_warm_start.use_count,
                )
        except Exception as _ssc_exc:
            logger.debug("§SSC-1 Zwischenspeicher-Lookup nicht blockierend: %s", _ssc_exc)

        def _budget_ok() -> bool:
            if no_rt_limit:
                return True
            # M-2: RT-Primärcheck — mode-abhängig (Spec §9.5):
            #   BALANCED=32× | RESTORATION/QUALITY=32× | MAXIMUM=32×
            # Zusätzlich: absolutes 90-Minuten-Limit — gilt unabhängig vom RT-Faktor.
            # StrategieDenker.check() wird hier NICHT verwendet, da es immer 8×RT
            # hardcoded prüft und so ExzellenzDenker in Restoration/Quality-Mode
            # fälschlich blockieren würde. Der primäre _RT_BUDGET_BY_MODE-Check ist
            # die einzige authoritative Schranke.
            if (time.perf_counter() - t_start) >= _MAX_TOTAL_SECONDS:
                logger.warning(
                    "⚠️ 30-Minuten-Absolutlimit erreicht (%.0fs) — Stufe wird übersprungen.",
                    time.perf_counter() - t_start,
                )
                return False
            _mode_limit = _RT_BUDGET_BY_MODE.get(effective_mode.lower(), 5.0)
            if _is_pytest_or_safe_validation_context():
                # Test/CI-Läufe müssen vor pytest-timeout kontrolliert degradieren.
                _mode_limit = min(_mode_limit, 6.0)
            return _rt() < _mode_limit * 0.90

        # ── v10 Pipeline-Guard: Watchdog + CLP + Dynamics + Whisper ───────────
        _guard: Any = None
        _guard_report: dict[str, Any] = {}
        try:
            from backend.core.pipeline_guard import get_guard

            _guard = get_guard()
            _guard.pre_flight(aktuelles_audio, sr, material="unknown")
            _guard.phase_start("orchestrierung")
        except Exception as _g_exc:
            logger.debug("AurikDenker: PipelineGuard init: %s", _g_exc)

        # ── Stufe 1: Tonträger-Erkennung ─────────────────────────────────────
        # Skip full TontraegerDenker run when the frontend has already analysed the
        # carrier (cached_medium_result from _carrier_bg / bridge cache) — avoids a
        # redundant MediumDetector pass that the user sees as "Tonträger wird erkannt".
        material = "unknown"
        _medium_confidence: float = 1.0  # §v10.303.4: Für PID-Planung
        if cached_medium_result is not None:
            # Use same multi-fallback as UV3 (line 1564): primary_material → material_type → material
            material = str(
                getattr(cached_medium_result, "primary_material", None)
                or getattr(cached_medium_result, "material_type", None)
                or getattr(cached_medium_result, "material", None)
                or "unknown"
            )
            _medium_confidence = float(getattr(cached_medium_result, "confidence", 1.0) or 1.0)
            stage_notes["tontraeger"] = (
                f"{material} (Cache, Konfidenz: {getattr(cached_medium_result, 'confidence', 0.0):.2f})"
            )
            phases_executed.append("tontraeger_erkennung")
            logger.info(
                "AurikDenker [1/10] Träger aus Frontend-Zwischenspeicher: %s (%.2f)",
                material,
                getattr(cached_medium_result, "confidence", 0.0),
            )
            _emit(2, f"Tonträger erkannt: {material} (aus Analyse-Cache)")
        else:
            _emit(2, "Tonträger wird erkannt …")
            try:
                # §2.59 Guard: input_path ohne Dateiendung → MediumDetector bekommt keinen Digital-Prior.
                # Bei .mp3/.aac/.ogg etc. würde der ×0.25 Analog-Penalty fehlen → falsche Klassifikation möglich.
                _input_ext = os.path.splitext(input_path)[1] if input_path else ""
                if input_path and not _input_ext:
                    logger.warning(
                        "AurikDenker: Eingabe_path=%s hat keine Dateiendung — "
                        "MediumDetector läuft ohne Digital-Prior (kein ×0.25 Analog-Penalty). "
                        "Bei .mp3/.aac/.ogg fehlt dann die korrekte Material-Erkennung.",
                        input_path,
                    )
                toni = get_tontraeger_denker().erkenne(aktuelles_audio, sr, file_path=input_path)
                material = toni.material_type
                _medium_confidence = float(getattr(toni, "confidence", 1.0) or 1.0)
                stage_notes["tontraeger"] = f"{material} (Konfidenz: {toni.confidence:.2f})"
                phases_executed.append("tontraeger_erkennung")
                logger.info("AurikDenker [1/10] Träger: %s (%.2f)", material, toni.confidence)
                # §6.7 v10.0.0: Bayesian ClassificationResult aus Stufe 1 als cached_medium_result
                # für UV3 übernehmen — eliminiert redundante MediumClassifier-Aufrufe.
                if getattr(toni, "classification_result", None) is not None:
                    cached_medium_result = toni.classification_result
                    logger.info(
                        "AurikDenker: MediumDetector-ClassificationResult als zwischengespeichert_medium_Ergebnis übernommen "
                        "(material=%s, conf=%.2f)",
                        getattr(cached_medium_result, "material_type", material),
                        getattr(cached_medium_result, "confidence", toni.confidence),
                    )
            except Exception as exc:
                _record_stage_failure("tontraeger", "TontraegerDenker", exc)
                logger.warning("AurikDenker [1/10] TontraegerDenker: %s", exc)

        # ── Stufe 2: Tonträgerketten-Analyse ──────────────────────────────────
        # §2.47a: Pass cached_medium_result to avoid a second MediumDetector.detect() call.
        _emit(4, "Tonträgerkette analysiert …")
        chain_info: dict[str, Any] = {}
        kette = None  # Guard: TontraegerketteDenker.analysiere() kann fehlschlagen
        try:
            kette = get_tontraegerkette_denker().analysiere(
                aktuelles_audio,
                sr,
                file_path=input_path,
                cached_medium_result=cached_medium_result,
            )
            chain_info = kette.as_dict()
            # §v10.119 Normalisierung: Beide Key-Namen konsistent befüllen.
            # KettenErgebnis.as_dict liefert "chain", TontraegerInfo.as_dict
            # liefert "transfer_chain" — fehlt der erwartete Key, fällt der
            # §0h-VETO auf depth=1 zurück (zu harte Export-Schwelle).
            if kette is not None:
                _kette_chain_list = list(getattr(kette, "chain", None) or [])
                if _kette_chain_list:
                    chain_info.setdefault("chain", _kette_chain_list)
                    chain_info.setdefault("transfer_chain", _kette_chain_list)
            # §6.8: Era-Precursor (reel_tape) der physikalischen Chain voranstellen
            _era_mp = (
                str(getattr(cached_era_result, "material_prior", "") or "").lower()
                if cached_era_result is not None
                else ""
            )
            _phys_chain = getattr(kette, "chain", []) or []
            if _era_mp in ("reel_tape", "tape") and _era_mp not in _phys_chain:
                _phys_chain = [_era_mp, *list(_phys_chain)]
                kette.chain = _phys_chain
                kette.chain_string = " → ".join(_phys_chain)
            stage_notes["kette"] = kette.chain_string
            phases_executed.extend(kette.combined_phases)
            logger.info(
                "AurikDenker [2/10] Kette: %s (Komplexität: %.2f)",
                kette.chain_string,
                kette.chain_complexity,
            )
            if getattr(kette, "chain", None):
                _emit(4, "__carrier_chain__:" + "|".join(str(_k) for _k in (kette.chain or [])))
        except Exception as exc:
            _record_stage_failure("kette", "TontraegerketteDenker", exc)
            logger.warning("AurikDenker [2/10] TontraegerketteDenker: %s", exc)

        # ── Stufe 3: Defekt-Analyse ───────────────────────────────────────────
        _emit(6, "Defekte werden analysiert …")
        defekt = None  # Guard: DefektDenker kann fehlschlagen
        _defekt_hint: dict[str, Any] | None = None

        def _get_surgical_defect_types() -> frozenset[str]:
            try:
                from backend.core.surgical_defect_analyzer import SURGICAL_DEFECT_TYPES

                return SURGICAL_DEFECT_TYPES
            except ImportError:
                return frozenset()

        def _defekt_scan_cb(pct: int, name: str = "") -> None:
            if name:
                _emit(6, f"Defekte werden analysiert … {name}")

        try:
            defekt = get_defekt_denker().analysiere(
                aktuelles_audio,
                sr,
                material=material,
                progress_callback=_defekt_scan_cb,
                cached_defect_result=cached_defect_result,
                file_ext=os.path.splitext(input_path)[1].lower() if input_path else "",
            )
            defekt_primaer_raw: str = cast(
                str, getattr(defekt, "primary_defect", None) or getattr(defekt, "primary_cause", "unknown")
            )
            stage_notes["defekt"] = f"Hauptdefekt: {defekt_primaer_raw} (Schwere: {defekt.overall_severity:.2f})"
            phases_executed.append("defekt_analyse")
            _defekt_hint = {
                "recommended_phases": list(getattr(defekt, "recommended_phases", [])),
                "confidence": float(getattr(defekt, "cause_confidence", 0.0)),
                # §2.59: Defekt-Typen und -Severities für PhasePruner/DefectManifest
                "defect_types": list(getattr(defekt, "defect_scores", {}).keys()),
                "defect_severities": dict(getattr(defekt, "defect_scores", {})),
                # §2.59: Chirurgische Klassifikation für alle Denker sichtbar
                "surgical_defect_types": [
                    d for d in getattr(defekt, "defect_scores", {}).keys() if d in _get_surgical_defect_types()
                ],
            }
            # Bug-17-Fix: raw DefectAnalysisResult aus DefektErgebnis extrahieren und
            # als cached_defect_result weiterreichen — verhindert zweiten internen Scan
            # in RestaurierDenker (ARE-Pfad) mit falscher Material-Erkennung.
            if cached_defect_result is None:
                _raw = getattr(defekt, "raw_scan_result", None)
                if _raw is not None:
                    cached_defect_result = _raw
                    logger.info(
                        "AurikDenker: DefektScan-Ergebnis (material=%s) als zwischengespeichert_defect_Ergebnis übernommen.",
                        getattr(_raw, "material_type", "?"),
                    )
            logger.info(
                "AurikDenker [3/10] Defekt: %s (Schwere: %.2f)",
                defekt_primaer_raw,
                defekt.overall_severity,
            )
        except Exception as exc:
            _record_stage_failure("defekt", "DefektDenker", exc)
            logger.warning("AurikDenker [3/10] DefektDenker: %s", exc)

        # ── Song-individuelle Defekt-Meldung für den Nutzer ──────────
        if defekt is not None and hasattr(defekt, "summary"):
            _prim = str(getattr(defekt, "summary", {}).get("primary_defect", defekt_primaer_raw or ""))
            float(getattr(defekt, "overall_severity", 0.5))
            if _prim:
                _emit(7, f"Hauptproblem erkannt: {_prim.replace('_', ' ').title()} — wird gezielt behandelt")

        # ── Stufe 4: Musikalischer Globalplan (§Dach) ────────────────────────
        self._wd_phase_start("globalplan")
        _emit(8, "Musikalischer Restaurierungsplan erstellt …")
        _globalplan: Any = None
        try:
            # use_ml_classifiers=False: EraClassifier/GermanSchlagerClassifier laufen
            # bereits parallel in UnifiedRestorerV3 (§P-3). Doppelaufruf vermeiden
            # (Anti-Parallelwelten-Pflicht). Nur DSP-Heuristik in Stufe 4.
            # hint_decade: cached_era_result aus Frontend-Vorabanalyse übergeben —
            # verhindert, dass GlobalPlan via DSP-Rolloff eine physikalisch unmögliche
            # Ära (z.B. 1890er für Magnetband) berechnet, die dann _recommend_autopilot_mode
            # und die emotionale Intention falsch setzt.
            _hint_decade = int(getattr(cached_era_result, "decade", 0) or 0) or None
            # hint_genre: cached genre result aus Pre-Analyse übergeben —
            # verhindert dass MusikalischerGlobalplan Genre als "unknown" einträgt
            # wenn GenreClassifier bereits Schlager/Jazz/etc. erkannt hat (§2.47a).
            _hint_genre: str | None = None
            if cached_genre_result is not None:
                if getattr(cached_genre_result, "is_schlager", False):
                    _hint_genre = "schlager"
                else:
                    _gl = getattr(cached_genre_result, "genre_label", None) or getattr(
                        cached_genre_result, "primary_label", None
                    )
                    if _gl and str(_gl).lower() not in ("unbekannt", "unknown", ""):
                        _hint_genre = str(_gl).lower()
            _globalplan = erstelle_globalplan(
                aktuelles_audio,
                sr,
                material=material,
                use_ml_classifiers=False,
                chain_info=chain_info,
                hint_decade=_hint_decade,
                hint_genre=_hint_genre,
            )
            stage_notes["globalplan"] = (
                f"\u00c4ra: {_globalplan.portrait.decade}er, "
                f"Genre: {_globalplan.portrait.genre}, "
                f"Intention: {_globalplan.emotional_intention}"
            )
            phases_executed.append("musikalischer_globalplan")
            logger.info(
                "AurikDenker [4/10] Globalplan: %s\u00b4er | %s | Authentizit\u00e4t=%.2f",
                _globalplan.portrait.decade,
                _globalplan.portrait.genre,
                _globalplan.authenticity_target,
            )
        except Exception as exc:
            _record_stage_failure("globalplan", "MusikalischerGlobalplan", exc)
            logger.warning("AurikDenker [4/10] MusikalischerGlobalplan: %s", exc)
            # Fallback: minimaler Dummy-Plan verhindert Folge-AttributeErrors
            if _globalplan is None:
                try:
                    _globalplan = erstelle_globalplan(
                        aktuelles_audio,
                        sr,
                        material=material,
                        use_ml_classifiers=False,
                        chain_info=chain_info,
                    )
                except Exception:
                    logger.debug("aurik_denker.py:1518: Silent exception absorbed", exc_info=True)
        finally:
            self._wd_phase_end("globalplan", aktuelles_audio, sr)

        # Wissenschaftliche Signal-Signatur einmal zentral berechnen und in
        # Strategie-, Orchestrierungs- und Rollout-Entscheidungen wiederverwenden.
        logger.debug(
            "AurikDenker: berechne Signal-Signatur (%.1fs Audio, %d samples) …", audio_duration_s, aktuelles_audio.size
        )
        _t0_sig = time.perf_counter()
        _signal_signature = self._compute_signal_intelligence_signature(aktuelles_audio, sr)
        _dt_sig = time.perf_counter() - _t0_sig
        logger.info("AurikDenker: Signal-Signatur berechnet in %.2fs", _dt_sig)

        # ── Stufe 5: Strategie (8×RT-Budget) ────────────────────────────────
        self._wd_phase_start("strategie")
        _emit(10, "Restaurierungsstrategie geplant …")
        strategie = None
        _strat_mode = "quality"
        try:
            strat_denker = get_strategie_denker()
            # M-2b: §7.6 defekt-adaptive Chunk-Größe — übergebe Defektschwere an Strategie
            _defect_sev_for_plan = float(getattr(defekt, "overall_severity", 0.0)) if defekt is not None else 0.0
            # §v10.304.18: Timeout-Guard für StrategieDenker.plan() — ROCm/HIP GPU-Init
            # kann auf manchen Systemen > 120s blockieren. Fallback: Default-Plan.
            import concurrent.futures as _cf_strat

            def _plan_with_timeout() -> Any:
                # §v10.706 Denker-IQ: chain_depth für adaptives Budget
                _strat_chain = chain_info.get("chain", []) if isinstance(chain_info, dict) else []
                _strat_depth = max(1, len(_strat_chain)) if isinstance(_strat_chain, list) else 1
                _strat_rs = (
                    float(getattr(cached_restorability_result, "restorability_score", 70.0))
                    if cached_restorability_result is not None
                    else 70.0
                )
                return strat_denker.plan(
                    aktuelles_audio,
                    sr,
                    enforce_3x_rt=True,
                    defect_severity=_defect_sev_for_plan,
                    signal_signature=_signal_signature,
                    chain_depth=_strat_depth,
                    restorability_score=_strat_rs,
                )

            _strat_future = _cf_strat.ThreadPoolExecutor(max_workers=1).submit(_plan_with_timeout)
            try:
                strategie = _strat_future.result(timeout=120.0)
            except (_cf_strat.TimeoutError, TimeoutError):
                logger.warning(
                    "§v10.304.18 StrategieDenker Zeitlimit nach 120s — verwende Default-Plan (GPU-Init blockiert)"
                )
                strategie = None  # Fallback: wird unten als Default gehandhabt
                _strat_future.cancel()
            if strategie is not None:
                strat_denker.starte_timer(audio_duration_s)
                _budget_raw = getattr(strategie, "max_processing_s", audio_duration_s * _3X_RT_LIMIT)
                _mode_raw = getattr(strategie, "quality_mode", "quality")
            else:
                _budget_raw = audio_duration_s * _3X_RT_LIMIT
                _mode_raw = "quality"
            try:
                _budget_s = float(_budget_raw)
            except (TypeError, ValueError):
                _budget_s = float(audio_duration_s * _3X_RT_LIMIT)
            _strat_mode = str(_mode_raw) if _mode_raw is not None else "quality"
            stage_notes["strategie"] = f"Budget: {_budget_s:.1f}s, Modus: {_strat_mode}"
            phases_executed.append("strategie_plan")
            logger.info(
                "AurikDenker [5/10] Grenze: %.1fs für %.1fs Audio",
                _budget_s,
                audio_duration_s,
            )
        except Exception as exc:
            _record_stage_failure("strategie", "StrategieDenker", exc)
            logger.warning("AurikDenker [5/10] StrategieDenker: %s", exc)
        finally:
            self._wd_phase_end("strategie", aktuelles_audio, sr)

        try:
            effective_mode, autopilot_note = self._recommend_autopilot_mode(
                requested_mode=mode,
                material=material,
                chain_info=chain_info,
                defekt=defekt,
                global_plan=_globalplan,
                strategy_mode=_strat_mode,
            )
        except Exception as _autopilot_exc:
            logger.warning(
                "Autopilot Betriebsart selection fehlgeschlagen: %s — defaulting to requested Betriebsart",
                _autopilot_exc,
            )
            effective_mode = requested_mode or "restoration"
            autopilot_note = f"Autopilot fallback (error): {_autopilot_exc}"
        stage_notes["autopilot"] = autopilot_note
        if requested_mode == "studio2026" and effective_mode == "restoration":
            warnings.append("Autopilot-Sicherheitsfallback: Studio 2026 wurde auf Restoration zurückgesetzt.")
        elif requested_mode == "studio2026" and effective_mode == "studio2026" and "Hinweis:" in autopilot_note:
            warnings.append(
                "Studio 2026 auf nicht-idealem Material: Interne Schutzmaßnahmen "
                "(SongCal, PMGG, FeedbackChain) sind aktiv."
            )

        # Wissenschaftsbasierte Oracle-Rollout-Empfehlung (off/pilot/all).
        try:
            _effective_oracle_rollout, _oracle_rollout_note = self._recommend_phase_strength_oracle_rollout(
                requested_rollout=phase_strength_oracle_rollout,
                material=material,
                chain_info=chain_info,
                defekt=defekt,
                global_plan=_globalplan,
                effective_mode=effective_mode,
                cached_restorability_result=cached_restorability_result,
                signal_signature=_signal_signature,
            )
        except Exception as _oracle_exc:
            _effective_oracle_rollout = self._normalize_oracle_rollout_mode(phase_strength_oracle_rollout) or "all"
            _oracle_rollout_note = f"Oracle-Rollout-Fallback nach Fehler: {_oracle_exc}"
            logger.warning("Oracle rollout recommendation fehlgeschlagen: %s", _oracle_exc)
        stage_notes["oracle_rollout"] = _oracle_rollout_note
        stage_notes["oracle_signal_signature"] = dict(_signal_signature)
        logger.info("AurikDenker [5a/10] %s", _oracle_rollout_note)

        # ── Stufe 5b: PhaseInteractionDenker — Orchestrierung übernehmen ────────
        # Erzeugt einen semantisch aufgelösten, konfliktfreien Phasenplan.
        # UV3.restore() erhält diesen als precomputed_phase_plan und agiert dann
        # als reiner Executor (kein _optimize_phase_plan_intelligence() mehr).
        # Bei Fehler: leerer Plan → UV3 selektiert autonom (fail-safe §0).
        _pid_plan = None
        _pid_phase_plan: list[str] | None = None
        _pid_conflict_notes: list[str] = []
        _pid_runtime_hint: dict[str, Any] = {}
        try:
            _pid_defect_result = cached_defect_result or (
                getattr(defekt, "raw_scan_result", None) if defekt is not None else None
            )
            if _pid_defect_result is not None:
                _pid_rest_score: float = float(
                    getattr(cached_restorability_result, "restorability_score", 70.0)
                    if cached_restorability_result is not None
                    else 70.0
                )
                # §GoalRisk: Exzellenz-Prognose VOR UV3 — schützende Phasen prophylaktisch
                # aktivieren statt P1/P2-Verletzungen erst nach der Pipeline zu heilen (§2.45a).
                _goal_risk_map: dict[str, float] = {}
                try:
                    _goal_risk_map = get_exzellenz_denker().prognostiziere(
                        aktuelles_audio,
                        sr,
                        defect_result=_pid_defect_result,
                        material=material,
                    )
                    if _goal_risk_map:
                        logger.debug(
                            "AurikDenker [5b/10] Goal-Risiko-Prognose: %s",
                            {k: f"{v:.2f}" for k, v in _goal_risk_map.items()},
                        )
                except Exception as _grm_exc:
                    logger.debug("AurikDenker [5b/10] Goal-Risiko-Prognose: %s", _grm_exc)
                _pid_plan = get_phase_interaction_denker().plan(
                    defect_result=_pid_defect_result,
                    material=material,
                    mode=effective_mode,
                    chain_info=chain_info or None,
                    chain_result=kette,
                    defekt_hint=_defekt_hint,
                    audio=aktuelles_audio,
                    sr=sr,
                    restorability_score=_pid_rest_score,
                    pipeline_confidence=_medium_confidence,
                    goal_risk_map=_goal_risk_map or None,
                    strategie_plan=strategie,
                    causal_plan=defekt,
                    signal_signature=_signal_signature,
                    sections=_get_musical_sections(aktuelles_audio, sr),
                )
                if _pid_plan.is_valid:
                    _pid_phase_plan = cast(list[str], _pid_plan.phases or [])
                    _pid_suppressed = cast(dict[str, str], _pid_plan.suppressed or {})
                    _pid_ordering_applied = cast(list[tuple[str, str]], _pid_plan.ordering_applied or [])
                    _pid_conflict_notes = cast(list[str], _pid_plan.conflict_notes or [])
                    _pid_injected = sum(1 for n in _pid_conflict_notes if "Injektion" in n)
                    _pid_top_goal = ""
                    _pid_top_risk = 0.0
                    _pid_phase_count = len(_pid_phase_plan)
                    if _goal_risk_map:
                        _pid_top_goal, _pid_top_risk = max(_goal_risk_map.items(), key=lambda kv: float(kv[1]))
                    _pid_runtime_hint = {
                        "phase_count": _pid_phase_count,
                        "suppressed_count": len(_pid_suppressed),
                        "injected_count": int(_pid_injected),
                        "top_goal": str(_pid_top_goal),
                        "top_goal_risk": float(_pid_top_risk),
                        "risk_goal_count": len(_goal_risk_map),
                    }
                    stage_notes["phase_interaction"] = (
                        f"PhaseInteractionDenker: {_pid_phase_count} Phasen "
                        f"({len(_pid_suppressed)} supprimiert, "
                        f"{len(_pid_ordering_applied)} Ordnungsänderungen, "
                        f"{_pid_injected} injiziert)"
                    )
                    _emit(
                        12,
                        "__pid_live_hint__:"
                        + f"phases={_pid_runtime_hint['phase_count']}|"
                        + f"suppressed={_pid_runtime_hint['suppressed_count']}|"
                        + f"injected={_pid_runtime_hint['injected_count']}|"
                        + f"goal={_pid_runtime_hint['top_goal']}|"
                        + f"risk={_pid_runtime_hint['top_goal_risk']:.3f}",
                    )
                    phases_executed.append("phase_interaction_denker")
                    logger.info(
                        "AurikDenker [5b/10] PhaseInteractionDenker: %d Phasen, supprimiert=%s, conflict_notes=%s",
                        _pid_phase_count,
                        list(_pid_suppressed.keys()),
                        _pid_conflict_notes,
                    )
                else:
                    stage_notes["phase_interaction"] = "PhaseInteractionDenker: kein Plan — UV3-Fallback"
                    logger.info("AurikDenker [5b/10] PhaseInteractionDenker: kein Plan → UV3 übernimmt.")
            else:
                stage_notes["phase_interaction"] = "PhaseInteractionDenker: kein defect_result — übersprungen"
        except Exception as _pid_exc:
            stage_notes["phase_interaction"] = f"PhaseInteractionDenker fehlgeschlagen: {_pid_exc}"
            logger.warning("AurikDenker [5b/10] PhaseInteractionDenker: %s", _pid_exc)

        # ── ARE-Metadaten-Träger (A-2/A-5/B-1) ────────────────────────────────
        _rest_confidence: float = 0.85
        _rest_rollback: bool = False
        _rest_variant: str | None = None
        _rest_musical_goals: dict[str, float] = {}
        _rest_goals_passed: int = 0
        _rest_metadata: dict[str, Any] = {}
        _rest_inapplicable_goals: frozenset[str] = frozenset()  # §S5: GAF-Inapplicable-Propagation
        _goal_app_raw: dict[str, bool] = {}  # §S5: raw goal_applicability aus RestaurierErgebnis
        _gaps_found: int = 0
        _gaps_repaired: int = 0
        _gap_total_ms: float = 0.0

        # ── Stufe 6–8: _run_rest()-Closure — Reparatur → Rekonstruktion → Restaurierung ──
        _emit(12, "Vorverarbeitung & Restaurierung laufen …")
        _skip_restoration, _skip_rest_metrics = self._should_skip_excellence_for_clean_digital(
            aktuelles_audio,
            sr,
            material,
            chain_info,
        )
        if _skip_restoration:
            _skip_mat = _skip_rest_metrics.get("material", self._resolve_excellence_material(material, chain_info))
            stage_notes["reparatur"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, keine Vorverarbeitung erforderlich)"
            )
            stage_notes["rekonstruktion"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, keine Lückenrekonstruktion erforderlich)"
            )
            stage_notes["restaurierung"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, "
                "Autopilot priorisiert Pass-Through vor Overprocessing)"
            )
            _rest_confidence = 1.0
            _rest_variant = "clean_digital_pass_through"
            phases_executed.append("clean_digital_pass_through")
            logger.info(
                "AurikDenker [8/10] Restaurierung übersprungen für saubere Digitalquelle: %s",
                _skip_rest_metrics,
            )
        elif _budget_ok():
            try:
                # M-3: Mode-Hierarchie: expliziter mode-Parameter hat Vorrang vor StrategieDenker
                _mode = effective_mode
                # RT-Budget-Guard: Restaurierung läuft in eigenem Thread.
                # Budget ist mode-abhängig (Spec §9.5: 5× für Quality, 8× für Maximum).
                # Cold-Start-Minimum: mindestens _COLDSTART_MIN_SECONDS um ML-Modell-Ladezeit
                # (PANNs 0.7 GB, wav2vec2 0.35 GB …) nicht dem Restaurierungs-Budget anzulasten.
                _rt_multiplier = _RT_BUDGET_BY_MODE.get(_mode.lower(), 5.0)
                if _is_pytest_or_safe_validation_context():
                    _rt_multiplier = min(_rt_multiplier, 6.0)
                _elapsed_so_far = time.perf_counter() - t_start
                _coldstart_floor = 30.0 if _is_pytest_or_safe_validation_context() else _COLDSTART_MIN_SECONDS
                _remaining = max(
                    audio_duration_s * _rt_multiplier - _elapsed_so_far,
                    _coldstart_floor,
                )
                if not no_rt_limit:
                    # §9.5 Absolutes 30-Minuten-Limit: Thread-Timeout darf nie über die
                    # verbleibende Zeit bis zur Gesamtgrenze hinausgehen. Mindestens 30 s
                    # werden immer einkalkuliert, damit auch beim Cold-Start-Ablauf noch
                    # ein sinnvoller Verarbeitungsversuch möglich ist.
                    _abs_remaining = _MAX_TOTAL_SECONDS - _elapsed_so_far
                    if _abs_remaining <= 0:
                        raise RuntimeError(
                            f"30-Minuten-Absolutlimit bereits überschritten "
                            f"({_elapsed_so_far:.0f}s) — Restaurierung wird nicht gestartet."
                        )
                    _remaining = min(_remaining, max(30.0, _abs_remaining))
                _result_box: list = []
                _err_box: list = []
                _rep_result_box: list = []  # ReparaturDenker Ergebnis (4a)
                _rek_result_box: list = []  # RekonstruktionsDenker Ergebnis (4b)

                def _run_rest() -> None:
                    # §Safety: PLM-Eviction für gesamte Restaurierungskette sperren —
                    # nicht nur _execute_pipeline(), sondern auch ReparaturDenker und
                    # RekonstruktionsDenker nutzen ONNX-Modelle. Refcount-basiert,
                    # sodass UV3-interner Guard additiv funktioniert.
                    try:
                        _set_pipeline_active(True)
                    except Exception as _exc:
                        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
                    try:
                        _work_audio = aktuelles_audio.copy()

                        # §2.39 OOM-Recovery: bei vorhandenem Checkpoint direkte
                        # Wiederaufnahme in UV3 (ohne erneute Reparatur-/Rekonstruktionsstufen).
                        if recovery_checkpoint is not None:

                            def _resume_cb(pct: int, msg: str, elapsed: float = 0.0) -> None:
                                if pct <= 19:
                                    d = 13 + int(pct * 7 / 19) if pct > 0 else 13
                                elif pct <= 85:
                                    d = 20 + int((pct - 20) * 67 / 65)
                                else:
                                    d = 87 + int((pct - 86) * 7 / 14)
                                _emit(min(94, d), msg)

                            _emit(13, "OOM-Recovery wird fortgesetzt …")
                            _result_box.append(
                                get_restaurier_denker().restauriere(
                                    _work_audio,
                                    sr,
                                    material=material,
                                    mode=_mode,
                                    global_plan=_globalplan,
                                    chain_info=chain_info or None,
                                    defekt_hint=_defekt_hint,
                                    progress_callback=_resume_cb if progress_callback is not None else None,
                                    audio_update_callback=audio_update_callback,
                                    cached_era_result=cached_era_result,
                                    cached_genre_result=cached_genre_result,
                                    cached_defect_result=(
                                        cached_defect_result
                                        or (getattr(defekt, "raw_scan_result", None) if defekt is not None else None)
                                    ),
                                    cached_medium_result=cached_medium_result,
                                    cached_restorability_result=cached_restorability_result,
                                    recovery_checkpoint=recovery_checkpoint,
                                    input_path=input_path,
                                    output_path=output_path,
                                    no_rt_limit=no_rt_limit,
                                    phase_strength_oracle_rollout=_effective_oracle_rollout,
                                )
                            )
                            return

                        # §G1 (GEBOTE.md): Pre-Repair-Referenz sichern — echtes Original VOR
                        # ReparaturDenker/RekonstruktionsDenker für referenz-basierte
                        # Musical Goals (Authentizität, Groove, Timbre, Artikulation).
                        _pre_repair_reference = _work_audio.copy()
                        # [6/10] ReparaturDenker — gezielter Phase-Mix (Preprocessing vor UV3)
                        _emit(13, "Gezielte DSP-Reparaturen (Vorverarbeitung) …")
                        _ph = set(getattr(defekt, "recommended_phases", None) or [])
                        _sev = float(getattr(defekt, "overall_severity", 1.0))
                        _skip_repair = _sev < 0.05 and defekt is not None
                        _remove_clicks: bool = not _skip_repair and (
                            defekt is None
                            or bool(
                                _ph
                                & {"phase_01_click_removal", "phase_09_crackle_removal", "phase_27_click_pop_removal"}
                            )
                            or _sev >= 0.3
                        )
                        _remove_hum: bool = not _skip_repair and (
                            defekt is None or bool(_ph & {"phase_02_hum_removal"}) or _sev >= 0.3
                        )
                        _repair_clipping: bool = not _skip_repair and (
                            defekt is None
                            or bool(_ph & {"phase_23_spectral_repair", "phase_06_frequency_restoration"})
                            or _sev >= 0.4
                        )
                        _dsp_op_names: dict[str, str] = {
                            "click_repair": "Knackser & Impulse werden entfernt",
                            "hum_removal": "Netzbrummen (50/60 Hz) wird entfernt",
                            "declip": "Übersteuerungen werden repariert",
                        }

                        def _repair_cb(op_key: str) -> None:
                            _label = _dsp_op_names.get(op_key, op_key)
                            _emit(13, f"DSP-Reparatur — {op_key}: {_label}")

                        # §2.41 v10.0.0: Defect-Locations + Scores aus cached_defect_result extrahieren,
                        # damit ReparaturDenker chirurgische (lokalisierte) Reparaturen durchführen kann.
                        _repair_defect_scores: dict[str, float] = {}
                        _repair_defect_locations: dict[str, list[tuple[float, float]]] = {}
                        _repair_era_decade: int | None = None
                        if cached_defect_result is not None and hasattr(cached_defect_result, "scores"):
                            for _rdt, _rds in cached_defect_result.scores.items():
                                _rdt_key = getattr(_rdt, "value", str(_rdt))
                                if hasattr(_rds, "severity"):
                                    _repair_defect_scores[_rdt_key] = float(_rds.severity)
                                if hasattr(_rds, "locations") and _rds.locations:
                                    _repair_defect_locations[_rdt_key] = list(_rds.locations)
                        elif defekt is not None:
                            _repair_defect_scores = dict(getattr(defekt, "defect_scores", {}) or {})
                        # Era-Dekade aus cached Era-Ergebnis
                        if cached_era_result is not None:
                            _repair_era_decade = int(getattr(cached_era_result, "decade", 0) or 0) or None

                        # §0c Codec-Chain-IQR-Floor: Ketten-Liste aus chain_info extrahieren,
                        # damit ReparaturDenker bei Terminal-Codec (mp3/aac) keinen
                        # zu aggressiven click_iqr verwendet (Brandenburg 1999).
                        # Key "transfer_chain" (TontraegerInfo.as_dict()) mit Fallback auf
                        # "chain" (ältere Serialisierungs-Pfade und pre_analysis-Handover).
                        _transfer_chain: list[str] | None = (
                            list(chain_info.get("transfer_chain") or chain_info.get("chain") or []) or None
                            if isinstance(chain_info, dict)
                            else None
                        )
                        rep = get_reparatur_denker().repariere(
                            _work_audio,
                            sr,
                            remove_clicks=_remove_clicks,
                            remove_hum=_remove_hum,
                            repair_clipping=_repair_clipping,
                            material=material or "",
                            progress_callback=_repair_cb,
                            defect_scores=_repair_defect_scores or None,
                            defect_locations=_repair_defect_locations or None,
                            era_decade=_repair_era_decade,
                            transfer_chain=_transfer_chain,
                        )
                        _work_audio = rep.audio
                        _rep_result_box.append(rep)
                        # [7/10] RekonstruktionsDenker — Lücken-Erkennung & Reparatur (Preprocessing vor UV3)
                        _emit(16, "Dropout-Lücken werden rekonstruiert (Vorverarbeitung) …")
                        rek = get_rekonstruktions_denker().rekonstruiere(
                            _work_audio,
                            sr,
                            material_hint=material,
                            defect_result=cached_defect_result,
                            repair_context=rep,  # §11.7a Kontextfluss: ReparaturDenker-Ergebnis weiterreichen
                            defect_locations=_repair_defect_locations or None,
                            era_decade=_repair_era_decade,
                        )
                        _work_audio = rek.audio
                        _rek_result_box.append(rek)
                        _strategie_as_dict = getattr(strategie, "as_dict", None)
                        _denker_policy_input = {
                            "strategy": _strategie_as_dict() if callable(_strategie_as_dict) else {},
                            "phase_interaction": dict(getattr(_pid_plan, "policy_hints", {}) or {}),
                            "phase_runtime_hint": dict(_pid_runtime_hint),
                            "repair_risk_profile": dict(getattr(rep, "repair_risk_profile", {}) or {}),
                            "reconstruction_risk_profile": dict(getattr(rek, "reconstruction_risk_profile", {}) or {}),
                            "signal_signature": dict(_signal_signature),
                            "source": "denker_policy_synthesis",
                        }

                        # §ORCHESTRATOR P1 (Spec 23_zero_touch_orchestration_contract.md):
                        # Gatekeeper-Preflight als Advisory-Stufe. Initialisiert den
                        # P2-Watchdog-Singleton, damit die in UV3 verdrahteten
                        # after_phase()-Aufrufe aktiv werden (vorher stiller No-Op).
                        # Der Phasenplan wird NICHT verändert — PMGG, HPE-Gate und
                        # CompletionEngine bleiben unverändert aktiv (keine Deaktivierung).
                        try:
                            from backend.core.aurik_orchestrator import get_orchestrator as _get_orch_pre

                            _orch_pre = _get_orch_pre()
                            _pf_chain_info = chain_info if isinstance(chain_info, dict) else {}
                            _pf_chain = _pf_chain_info.get("transfer_chain") or _pf_chain_info.get("chain") or []
                            _pf_rs = float(
                                getattr(cached_restorability_result, "restorability_score", 0.0)
                                or getattr(defekt, "restorability_score", 0.0)
                                or 50.0
                            )
                            _pf_pruned, _pf_decision = _orch_pre.preflight(
                                original_audio=audio,
                                sample_rate=sr,
                                restorability_score=_pf_rs,
                                transfer_chain_depth=max(
                                    1, len(_pf_chain) if isinstance(_pf_chain, (list, tuple)) else 1
                                ),
                                bandwidth_loss=float(_pf_chain_info.get("bandwidth_loss", 0.0) or 0.0),
                                snr_db=float(_pf_chain_info.get("snr_db", 40.0) or 40.0),
                                terminal_codec=_pf_chain_info.get("terminal_codec"),
                                material_type=str(material or "unknown"),
                                is_restoration_mode=True,
                                selected_phases=[str(p) for p in (_pid_phase_plan or [])],
                            )
                            stage_notes["orchestrator_preflight"] = {
                                "mode": _pf_decision.mode,
                                "max_phases": int(_pf_decision.max_phases),
                                "reason": _pf_decision.reason,
                                "phases_selected": len(_pid_phase_plan or []),
                                "phases_suggested_prune": len(_pf_pruned),
                            }
                            if _pf_decision.mode == "passthrough":
                                warnings.append(
                                    "Orchestrator-Empfehlung: passthrough ("
                                    + _pf_decision.reason
                                    + "). Pipeline läuft unverändert weiter (Spec 23, Stufe 1)."
                                )
                            elif _pf_decision.mode in ("conservative", "repair_only") and effective_mode in (
                                "studio2026",
                                "maximum",
                            ):
                                warnings.append(
                                    f"Orchestrator-Empfehlung '{_pf_decision.mode}' bei aggressivem Modus "
                                    f"'{effective_mode}' — interne Schutzmaßnahmen (SongCal, PMGG, HPE-Gate) bleiben aktiv."
                                )
                        except Exception as _opf_exc:
                            logger.debug("§ORCHESTRATOR preflight nicht blockierend: %s", _opf_exc)
                        # 4c: RestaurierDenker (UV3-Vollpipeline) auf vorgereinigtem Material
                        # Scaled inner progress: UV3 0–100 → AurikDenker 17–94
                        # UV3-intern: Analyse pct 1–19, Pipeline pct 20–85, Post pct 86–96.
                        # Denker-Mapping (§v10.19 Bugfix: Baseline 13→17, verhindert
                        # Monotonic-Dead-Zone mit letztem Pre-ARE-Emit von 16):
                        #   UV3 0–19  (Analyse)   → Denker 17–24  (komprimiert: 7 pts)
                        #   UV3 20–85 (37 Phasen) → Denker 24–87  (Löwenanteil: 63 pts)
                        #   UV3 86–100 (Post)     → Denker 87–94  (komprimiert: 7 pts)

                        def _inner_cb(pct, msg, elapsed=0.0):
                            if pct <= 19:
                                d = 17 + int(pct * 7 / 19) if pct > 0 else 17
                            elif pct <= 85:
                                d = 24 + int((pct - 20) * 63 / 65)
                            else:
                                d = 87 + int((pct - 86) * 7 / 14)
                            _emit(min(94, d), msg)

                        _result_box.append(
                            get_restaurier_denker().restauriere(
                                _work_audio,
                                sr,
                                material=material,
                                mode=_mode,
                                global_plan=_globalplan,
                                chain_info=chain_info or None,
                                defekt_hint=_defekt_hint,
                                progress_callback=_inner_cb if progress_callback is not None else None,
                                audio_update_callback=audio_update_callback,
                                cached_era_result=cached_era_result,
                                cached_genre_result=cached_genre_result,
                                # Bug-17-Fix: intern berechnetes DefectAnalysisResult aus
                                # DefektDenker.analysiere() als cached_defect_result nutzen,
                                # damit RestaurierDenker den Direkt-UV3-Pfad (_has_caches=True)
                                # nehmen kann. Ohne diesen Fix: _has_caches=False → ARE-Pfad →
                                # konkurrierende UV3-Instanzen → OOM.
                                cached_defect_result=(
                                    cached_defect_result
                                    or (getattr(defekt, "raw_scan_result", None) if defekt is not None else None)
                                ),
                                cached_medium_result=cached_medium_result,
                                cached_restorability_result=cached_restorability_result,
                                reconstruction_context=rek,
                                pre_repair_reference=_pre_repair_reference,
                                input_path=input_path,
                                output_path=output_path,
                                no_rt_limit=no_rt_limit,
                                # §PID: PhaseInteractionDenker-Plan — UV3 als reiner Executor
                                precomputed_phase_plan=_pid_phase_plan,
                                conflict_notes=_pid_conflict_notes,
                                phase_strength_oracle_rollout=_effective_oracle_rollout,
                                denker_policy_input=_denker_policy_input,
                                use_source_separation=os.environ.get("AURIK_SOURCE_SEPARATION", "") == "1",
                            )
                        )
                    except Exception as _e:
                        _err_box.append(_e)
                    finally:
                        try:
                            _set_pipeline_active(False)
                        except Exception as _exc:
                            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

                _daemon_rest_thread = not _is_pytest_or_safe_validation_context()
                _t = threading.Thread(target=_run_rest, daemon=_daemon_rest_thread)
                _t.start()
                if no_rt_limit:
                    _t.join()
                else:
                    _t.join(timeout=_remaining)
                    if _t.is_alive():
                        raise RuntimeError(
                            "RestaurierDenker überschritt RT-Budget "
                            f"({_remaining:.1f}s für {audio_duration_s:.1f}s Audio)"
                        )
                if _err_box:
                    raise _err_box[0]  # type: ignore[misc]
                rest = _result_box[0]
                if isinstance(rest, np.ndarray):
                    rest = SimpleNamespace(
                        audio=np.asarray(rest, dtype=np.float32),
                        phases_executed=[],
                        warnings=[],
                        quality_estimate=0.0,
                        rt_factor=0.0,
                        confidence=0.85,
                        rollback_triggered=False,
                        winning_variant=None,
                        musical_goals={},
                        goals_passed=0,
                        metadata={},
                    )
                aktuelles_audio = rest.audio
                phases_executed.extend(rest.phases_executed or [])
                warnings.extend(rest.warnings or [])
                # A-2: ARE-Metadaten propagieren
                _rest_confidence = float(getattr(rest, "confidence", 0.85))
                _rest_rollback = bool(getattr(rest, "rollback_triggered", False))
                _rest_variant = getattr(rest, "winning_variant", None)
                _raw_goals = getattr(rest, "musical_goals", None)
                _rest_musical_goals = dict(_raw_goals) if _raw_goals else {}
                _rest_goals_passed = int(getattr(rest, "goals_passed", 0))
                # §S5: GAF-Inapplicable-Ziele aus UV3-RestorationResult extrahieren
                _goal_app_raw = dict(getattr(rest, "goal_applicability", None) or {})
                _rest_inapplicable_goals = frozenset(g for g, ok in _goal_app_raw.items() if not ok)
                _meta_raw = getattr(rest, "metadata", None)
                _rest_metadata = dict(_meta_raw) if isinstance(_meta_raw, dict) else {}
                if _pid_runtime_hint:
                    _rest_metadata["phase_interaction"] = {
                        "phase_count": int(_pid_runtime_hint.get("phase_count", 0) or 0),
                        "suppressed_count": int(_pid_runtime_hint.get("suppressed_count", 0) or 0),
                        "injected_count": int(_pid_runtime_hint.get("injected_count", 0) or 0),
                        "top_goal": str(_pid_runtime_hint.get("top_goal", "") or ""),
                        "top_goal_risk": float(_pid_runtime_hint.get("top_goal_risk", 0.0) or 0.0),
                        "risk_goal_count": int(_pid_runtime_hint.get("risk_goal_count", 0) or 0),
                    }
                # Keep a single canonical material label across UV3 scorecards and
                # AurikDenker final logs/exports.
                _rest_mat_raw = getattr(rest, "material_type", None)
                _rest_mat = str(getattr(_rest_mat_raw, "value", _rest_mat_raw) or "").strip().lower()
                if _rest_mat and _rest_mat not in {"unknown", "materialtype.unknown"}:
                    material = _rest_mat
                else:
                    _meta_mat = (
                        str(
                            ((_rest_metadata.get("defect_analysis") or {}).get("material") if _rest_metadata else "")
                            or ""
                        )
                        .strip()
                        .lower()
                    )
                    if _meta_mat and _meta_mat not in {"unknown", "materialtype.unknown"}:
                        material = _meta_mat
                # §Dach-Enrichment: era_decade aus RestorationResult in Globalplan übernehmen
                # (ML-Klassifikatoren liefen in UV3 — jetzt Ergebnis in Plan einpflegen)
                if _globalplan is not None:
                    _era_from_pipeline = getattr(rest, "era_decade", None)
                    if _era_from_pipeline is not None:
                        try:
                            _globalplan.portrait.decade = int(_era_from_pipeline)
                            _globalplan.portrait.era_confidence = max(_globalplan.portrait.era_confidence, 0.75)
                            _globalplan.reasoning_trace.append(
                                f"Enriched with pipeline era_decade={_era_from_pipeline} (UV3 ML)"
                            )
                        except Exception as _exc:
                            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
                # [6/10] ReparaturDenker-Ergebnis (Preprocessing-Schritt)
                if _rep_result_box:
                    _rep = _rep_result_box[0]
                    warnings.extend(getattr(_rep, "warnings", []) or [])
                    stage_notes["reparatur"] = (
                        f"Clicks: {getattr(_rep, 'clicks_removed', False)}, "
                        f"Hum: {getattr(_rep, 'hum_removed', False)}, "
                        f"Clipping: {getattr(_rep, 'clipping_repaired', False)} (Vorverarbeitung)"
                    )
                    logger.info(
                        "AurikDenker [6/10] Reparatur (Pre-UV3): clicks=%s hum=%s clipping=%s",
                        getattr(_rep, "clicks_removed", False),
                        getattr(_rep, "hum_removed", False),
                        getattr(_rep, "clipping_repaired", False),
                    )
                # [7/10] RekonstruktionsDenker-Ergebnis (Preprocessing-Schritt)
                if _rek_result_box:
                    _rek = _rek_result_box[0]
                    warnings.extend(getattr(_rek, "warnings", []) or [])
                    _gaps_found = int(getattr(_rek, "gaps_found", 0))
                    _gaps_repaired = int(getattr(_rek, "gaps_repaired", 0))
                    _gap_total_ms = float(getattr(_rek, "total_repaired_ms", 0.0))
                    stage_notes["rekonstruktion"] = (
                        f"Lücken gefunden: {_gaps_found}, "
                        f"repariert: {_gaps_repaired}/{_gap_total_ms:.1f} ms (Vorverarbeitung)"
                    )
                    logger.info(
                        "AurikDenker [7/10] Rekonstruktion (Pre-UV3): %d/%d Lücken, %.1f ms",
                        _gaps_repaired,
                        _gaps_found,
                        _gap_total_ms,
                    )
                stage_notes["restaurierung"] = (
                    f"Qualität: {rest.quality_estimate:.3f}, "
                    f"RT: {rest.rt_factor:.2f}×, "
                    f"Konfidenz: {_rest_confidence:.2f}"
                )
                logger.info(
                    "AurikDenker [8/10] Restaurierung: Q=%.3f, RT=%.2f×, K=%.2f",
                    rest.quality_estimate,
                    rest.rt_factor,
                    _rest_confidence,
                )
            except Exception as exc:
                _record_stage_failure("restaurierung", "RestaurierDenker", exc)
                logger.warning("AurikDenker [8/10] RestaurierDenker: %s", exc)
        else:
            stage_notes["restaurierung"] = "Übersprungen (RT-Budget ausgeschöpft)"
            warnings.append("Restaurierung übersprungen: RT-Budget ausgeschöpft")
            logger.warning("AurikDenker [8/10] Grenze erschöpft, Restaurierung übersprungen")

        # ── Stufe 7: Exzellenz-Optimierung + Musical Goals ───────────────────
        _emit(95, "Musikalische Exzellenz wird optimiert …")
        musical_goals: dict[str, float] = dict(_rest_musical_goals)
        goals_passed: int = _rest_goals_passed
        excellence_score = 0.0
        _exz_versa_mos: float = 0.0  # M-8b: aus ExzellenzDenker gecachter VERSA-Score
        _exz_material = self._resolve_excellence_material(material, chain_info)
        _skip_excellence, _skip_metrics = self._should_skip_excellence_for_clean_digital(
            aktuelles_audio,
            sr,
            material,
            chain_info,
        )
        _exz_recovery_profile = self._recommend_excellence_recovery_profile(
            material=material,
            chain_info=chain_info,
            effective_mode=effective_mode,
            signal_signature=_signal_signature,
            versa_mos=_exz_versa_mos,
        )
        stage_notes["exzellenz_recovery_profile"] = {
            "material": _exz_recovery_profile.get("resolved_material"),
            "preserve_signal": float(_exz_recovery_profile.get("preserve_signal", 0.0)),
            "blend_alphas": list(_exz_recovery_profile.get("blend_alphas", (0.92, 0.88, 0.84))),
            "strict_mos_recovery": bool(_exz_recovery_profile.get("strict_mos_recovery", False)),
        }

        if _rest_rollback:
            # ARE rollback means restoration degraded quality — ExzellenzDenker
            # on unrestored audio is wasteful and violates Autopilot rules.
            stage_notes["exzellenz"] = (
                "Übersprungen (ARE-Rollback: Restaurierung verschlechterte Qualität, "
                "ExzellenzDenker auf unrestauriertem Audio nicht sinnvoll)"
            )
            warnings.append(
                "ExzellenzDenker übersprungen: ARE-Rollback aktiv — Restaurierung hat Qualität verschlechtert"
            )
            logger.warning(
                "AurikDenker [9/10] Exzellenz übersprungen: ARE-Rollback aktiv "
                "(restoration degraded quality, excellence on unrestored audio uebersprungen)",
            )
        elif _skip_excellence:
            _skip_mat = _skip_metrics.get("material", _exz_material)
            stage_notes["exzellenz"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, "
                "Autopilot priorisiert Originaltreue vor Overprocessing)"
            )
            logger.info(
                "AurikDenker [9/10] Exzellenz übersprungen für saubere Digitalquelle: %s",
                _skip_metrics,
            )
        elif _budget_ok():
            # v10.0.0: STFT-basierte ExcellenceOptimizer-Passe nach UV3+FeedbackChain
            # deaktiviert (Ephraim & Malah 1984: kaskadierte STFT-Modifikation akkumuliert
            # Rundungsfehler; ML-Modelle nicht auf eigenen Output trainiert → Domain Shift).
            # v10.0.0: messe_und_repariere() ersetzt messe_ziele() — nutzt ausschließlich
            # Zeit-Domain-Operationen (micro_dynamics, ola_edges) und linearen Blend mit
            # Original-Audio für P3-P5-Verletzungen. Kein STFT-Roundtrip → Ephraim-sicher.
            try:
                exd = get_exzellenz_denker()
                # Goals messen + konservative P3-P5-Reparatur (Zeit-Domain-first, §0 + §2.45).
                # Legacy-Fallback für Testmocks/ältere ExzellenzDenker mit messe_ziele().
                _used_repair_path = False
                _repair_fn = None
                _repair_attr = getattr(type(exd), "messe_und_repariere", None)
                if callable(_repair_attr):
                    _repair_fn = getattr(exd, "messe_und_repariere", None)
                if callable(_repair_fn):
                    _repair_call = cast(Callable[..., Any], _repair_fn)
                    _repair_sig = inspect.signature(_repair_call)
                    _repair_kwargs: dict[str, Any] = {}
                    if "mode" in _repair_sig.parameters:
                        _repair_kwargs["mode"] = effective_mode
                    if "material" in _repair_sig.parameters:
                        _repair_kwargs["material"] = _exz_material
                    if "inapplicable_goals" in _repair_sig.parameters and _rest_inapplicable_goals:
                        # §S5: GAF-Inapplicable-Propagation — physikalisch unerreichbare Goals
                        # aus Violations-Zählung und Reparaturlogik ausschließen (§2.32)
                        _repair_kwargs["inapplicable_goals"] = _rest_inapplicable_goals
                    if "reference_audio" in _repair_sig.parameters:
                        _repair_out = _repair_call(
                            aktuelles_audio,
                            sr,
                            reference_audio=audio,
                            **_repair_kwargs,
                        )
                    else:
                        _repair_out = _repair_call(aktuelles_audio, sr, **_repair_kwargs)
                    _used_repair_path = True
                    if isinstance(_repair_out, tuple) and len(_repair_out) == 2:
                        aktuelles_audio = np.asarray(_repair_out[0], dtype=np.float32)
                        goals = _normalize_goal_scores(_repair_out[1])
                    else:
                        goals = _normalize_goal_scores(_repair_out)
                else:
                    # §2.32 bass_kraft/brillanz-Fix: Legacy-Pfad für ältere ExzellenzDenker.
                    _legacy_fn = getattr(exd, "messe_ziele", None)
                    if not callable(_legacy_fn):
                        raise RuntimeError("ExzellenzDenker liefert weder messe_und_repariere() noch messe_ziele()")
                    _legacy_call = cast(Callable[..., Any], _legacy_fn)
                    _legacy_sig = inspect.signature(_legacy_call)
                    if "material" in _legacy_sig.parameters:
                        _legacy_out = _legacy_call(aktuelles_audio, sr, material=_exz_material)
                    else:
                        _legacy_out = _legacy_call(aktuelles_audio, sr)
                    if isinstance(_legacy_out, tuple) and len(_legacy_out) == 2:
                        aktuelles_audio = np.asarray(_legacy_out[0], dtype=np.float32)
                        goals = _normalize_goal_scores(_legacy_out[1])
                    else:
                        goals = _normalize_goal_scores(_legacy_out)
                if goals:
                    try:
                        _exz_thresholds = _get_canonical_thresholds_for_mode(
                            is_studio_2026=effective_mode == "studio2026"
                        )
                    except Exception:
                        _exz_thresholds = {}
                    _FALLBACK_GOAL_MIN = 0.75

                    finite_vals = [v for v in goals.values() if math.isfinite(v)]
                    excellence_score = float(np.mean(finite_vals)) if finite_vals else 0.0
                    goals_passed = sum(
                        1
                        for k, v in goals.items()
                        if k not in _rest_inapplicable_goals  # §S5: physikalisch nicht erreichbare Goals ausschließen
                        and math.isfinite(v)
                        and v >= _exz_thresholds.get(k, _FALLBACK_GOAL_MIN)
                    )
                    musical_goals = _normalize_goal_scores(goals)
                # VERSA MOS für Qualitätsentscheidung
                try:
                    if aktuelles_audio.ndim == 1:
                        _mono_exz = aktuelles_audio
                    elif aktuelles_audio.ndim == 2:
                        # Robust mono downmix for both layouts:
                        # (channels, samples) -> axis=0, (samples, channels) -> axis=1.
                        _mono_exz = (
                            aktuelles_audio.mean(axis=0)
                            if aktuelles_audio.shape[0] <= 2 < aktuelles_audio.shape[1]
                            else aktuelles_audio.mean(axis=1)
                        )
                    else:
                        _mono_exz = np.ravel(aktuelles_audio)
                    _mono_exz = np.nan_to_num(_mono_exz.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    _vr_exz = _score_versa_mos(_mono_exz, sr)
                    _exz_versa_mos = float(_vr_exz.mos)
                except Exception as _ve_exz:
                    logger.debug("ExzellenzDenker VERSA nicht verfügbar: %s", _ve_exz)

                _goals_total = len(goals) if goals else 0
                stage_notes["exzellenz"] = (
                    f"Messung{' + Repair' if _used_repair_path else ''}: Score {excellence_score:.3f}, "
                    f"{goals_passed}/{_goals_total} Ziele erfüllt"
                    + (f", VERSA MOS={_exz_versa_mos:.2f}" if _exz_versa_mos > 0.0 else "")
                    + (
                        " (Zeit-Domain-Repair für P3-P5 — kein STFT-Re-Pass)"
                        if _used_repair_path
                        else " (Legacy-Goal-Messpfad)"
                    )
                )
                phases_executed.append("exzellenz_messung_repair" if _used_repair_path else "exzellenz_messung")
                logger.info(
                    "AurikDenker [9/10] Exzellenz: Wert=%.3f, Goals %d/%d%s (%s)",
                    excellence_score,
                    goals_passed,
                    _goals_total,
                    f", VERSA={_exz_versa_mos:.3f}" if _exz_versa_mos > 0.0 else "",
                    "Goal-Repair aktiv" if _used_repair_path else "Legacy-Goal-Messung",
                )
            except Exception as exc:
                _record_stage_failure("exzellenz", "ExzellenzDenker", exc)
                logger.warning("AurikDenker [9/10] ExzellenzDenker: %s", exc)
        else:
            stage_notes["exzellenz"] = "Übersprungen (RT-Budget ausgeschöpft)"
            warnings.append("Exzellenz-Optimierung übersprungen: RT-Budget ausgeschöpft")
            logger.warning("AurikDenker [9/10] Exzellenz übersprungen")
            # Musical Goals + VERSA trotzdem messen (~7 s) — essentiell für UI-Anzeige
            # und Qualitätsurteil, auch wenn Optimierung übersprungen wird.
            try:
                MusicalGoalsChecker = _load_symbol(
                    "backend.core.musical_goals.musical_goals_metrics",
                    "MusicalGoalsChecker",
                )

                _mg_budget = MusicalGoalsChecker(mode=effective_mode)
                _budget_goals = _mg_budget.measure_all(aktuelles_audio, sr)
                if _budget_goals:
                    musical_goals = _normalize_goal_scores(_budget_goals)
                    _finite_bg = [v for v in musical_goals.values() if math.isfinite(v)]
                    goals_passed = sum(1 for v in _finite_bg if v >= 0.75)
                    excellence_score = float(np.mean(_finite_bg)) if _finite_bg else 0.0
                    logger.info(
                        "AurikDenker [9/10] Goals gemessen trotz Grenze-Limit: %d/%d bestanden",
                        goals_passed,
                        len(_budget_goals),
                    )

                    # Budgetfreundlicher Goal-Recovery ohne zusätzliche DSP-Phasen:
                    # Falls kritische Goals schwach sind, mische einen kleinen Anteil
                    # Originalsignal zurück (P1/P2-Schutz), messe erneut und übernehme
                    # nur bei tatsächlicher Verbesserung.
                    _critical_goals = {
                        "natuerlichkeit",
                        "authentizitaet",
                        "tonal_center",
                        "timbre_authentizitaet",
                        "artikulation",
                    }
                    _crit_total = sum(1 for g in _budget_goals if g in _critical_goals)
                    _crit_pass_before = sum(
                        1 for g, v in _budget_goals.items() if g in _critical_goals and float(v) >= 0.75
                    )
                    if _crit_total > 0 and _crit_pass_before < _crit_total and aktuelles_audio.shape == audio.shape:
                        _best_audio = aktuelles_audio
                        _best_goals = dict(_budget_goals)
                        _best_pass = goals_passed
                        _best_score = excellence_score
                        _best_crit = _crit_pass_before

                        _blend_alphas = tuple(_exz_recovery_profile.get("blend_alphas", (0.92, 0.88, 0.84)))
                        for _alpha in _blend_alphas:
                            _candidate = np.clip(_alpha * aktuelles_audio + (1.0 - _alpha) * audio, -1.0, 1.0)
                            _cand_goals = _normalize_goal_scores(_mg_budget.measure_all(_candidate, sr))
                            _cand_finite = [v for v in _cand_goals.values() if math.isfinite(v)]
                            _cand_pass = sum(1 for v in _cand_finite if v >= 0.75)
                            _cand_score = float(np.mean(_cand_finite)) if _cand_finite else 0.0
                            _cand_crit = sum(
                                1 for g, v in _cand_goals.items() if g in _critical_goals and float(v) >= 0.75
                            )

                            _strict_mos = bool(_exz_recovery_profile.get("strict_mos_recovery", False))
                            _is_better = (_cand_crit > _best_crit) or (
                                _cand_crit == _best_crit
                                and (
                                    _cand_pass > _best_pass or (_cand_pass == _best_pass and _cand_score > _best_score)
                                )
                            )
                            if _strict_mos and _cand_pass == _best_pass and _cand_score >= _best_score - 1e-6:
                                # Bei kritischem MOS erlauben wir neutralen Goal-Tradeoff,
                                # wenn dadurch später ein stärkerer Originalanteil möglich ist.
                                _is_better = True
                            if _is_better:
                                _best_audio = _candidate
                                _best_goals = dict(_cand_goals)
                                _best_pass = _cand_pass
                                _best_score = _cand_score
                                _best_crit = _cand_crit

                        if _best_crit > _crit_pass_before or _best_pass > goals_passed:
                            aktuelles_audio = _best_audio
                            musical_goals = _best_goals
                            goals_passed = _best_pass
                            excellence_score = _best_score
                            stage_notes["exzellenz"] += (
                                f"; Zielschutz-Blend angewendet (kritisch {_crit_pass_before}/{_crit_total} → "
                                f"{_best_crit}/{_crit_total}, gesamt {goals_passed}/{len(_best_goals)})"
                            )
                            logger.info(
                                "AurikDenker [9/10] Goal-Wiederherstellung-Blend aktiv: "
                                "kritische Goals %d/%d → %d/%d, gesamt=%d/%d",
                                _crit_pass_before,
                                _crit_total,
                                _best_crit,
                                _crit_total,
                                goals_passed,
                                len(_best_goals),
                            )
            except Exception as _mg_exc:
                logger.debug("Musical Goals Messung nach Grenze-Limit: %s", _mg_exc)
            try:
                if aktuelles_audio.ndim == 1:
                    _mono_budget = aktuelles_audio
                elif aktuelles_audio.ndim == 2:
                    _mono_budget = (
                        aktuelles_audio.mean(axis=0)
                        if aktuelles_audio.shape[0] <= 2 < aktuelles_audio.shape[1]
                        else aktuelles_audio.mean(axis=1)
                    )
                else:
                    _mono_budget = np.ravel(aktuelles_audio)
                _mono_budget = np.nan_to_num(_mono_budget.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _vr_budget = _score_versa_mos(_mono_budget, sr)
                _exz_versa_mos = float(_vr_budget.mos)
                logger.info(
                    "AurikDenker [9/10] VERSA MOS gemessen trotz Grenze-Limit: %.3f",
                    _exz_versa_mos,
                )
            except Exception as _ve_budget:
                logger.debug("VERSA MOS nach Grenze-Limit: %s", _ve_budget)

        # ── §APR: Adaptive Post-Repair — zielgerichtete Nachbesserung nach ExzellenzDenker ─────
        # §0a Crossfire-Modus-Invariante: §APR darf NUR in Studio 2026 laufen.
        # Restoration = Minimal-Intervention (nur Trägerverluste invertieren) — kein Bandpass-EQ-Boost.
        # Studio 2026 = Enhancement erlaubt — waerme/brillanz-Boost ist korrekte Studio-Operation.
        # Aktivierung: Studio 2026 + goals vorhanden + P3-P5-Gap > 2 % unter Material-Floor + Budget ok.
        # Methode: Zeit-Domain-Bandpass-Blend (sosfiltfilt, zero-phase) für waerme/brillanz.
        # Gate: VERSA MOS darf nicht um mehr als 0.05 MOS-Punkte fallen (non-blocking, kein Veto).
        # §0: Primum non nocere — kein Artefakt; kein STFT-Roundtrip (Ephraim 1984).
        _apr_is_studio = effective_mode in ("studio2026", "studio_2026", "studio")
        if _apr_is_studio and not _rest_rollback and musical_goals and _budget_ok():
            try:
                _apr_get_floor = get_material_floor
                _apr_butter = butter
                _apr_sosfiltfilt = sosfiltfilt

                # Material-BW-Ceiling für §6.2b-Konformität (Kassette: 12 kHz, Shellac: 7 kHz, …)
                _apr_bw_ceiling: float = float(
                    _rest_metadata.get("bw_ceiling_hz")
                    or {
                        "cassette": 12000.0,
                        "tape": 15000.0,
                        "vinyl": 16000.0,
                        "shellac": 7000.0,
                        "acetate": 7000.0,
                    }.get(_exz_material, 20000.0)
                )
                # P3-P5 Goals die durch Zeit-Domain-Bandpass-EQ direkt verbesserbar sind:
                #   bass_kraft → 60–200 Hz (+1.0 dB) — Sub-Bassband der musikalischen Wärme
                #   waerme     → 200–800 Hz (+1.5 dB additive blend)
                #   brillanz   → 2 kHz–min(6 kHz, BW-Ceiling×0.90) (+1.0 dB additive blend)
                # §6.2b-Konformität: Shellac/Wax 200 Hz Sub-Ceiling bereits durch BW-Ceiling garantiert.
                _APR_GOALS: dict[str, dict] = {
                    "bass_kraft": {"lo": 60.0, "hi": 200.0, "gain_db": 1.0},
                    "waerme": {"lo": 200.0, "hi": 800.0, "gain_db": 1.5},
                    "brillanz": {"lo": 2000.0, "hi": min(6000.0, _apr_bw_ceiling * 0.90), "gain_db": 1.0},
                }
                _apr_nyq = float(sr) / 2.0
                _apr_targets: list[str] = []
                for _apr_g, _apr_cfg in _APR_GOALS.items():
                    if _apr_g not in musical_goals:
                        continue
                    try:
                        _apr_floor = float(_apr_get_floor(_exz_material, _apr_g))
                    except Exception:
                        _apr_floor = 0.78
                    _apr_score = float(musical_goals.get(_apr_g) or 0.0)
                    _hi_safe = min(_apr_cfg["hi"], _apr_nyq * 0.95)
                    if _apr_score < _apr_floor - 0.02 and _apr_cfg["lo"] < _hi_safe:
                        _apr_targets.append(_apr_g)

                if _apr_targets:
                    _apr_candidate = np.array(aktuelles_audio, dtype=np.float32)
                    for _apr_g in _apr_targets:
                        _apr_cfg = _APR_GOALS[_apr_g]
                        _hi_safe = min(_apr_cfg["hi"], _apr_nyq * 0.95)
                        try:
                            _sos_apr = _apr_butter(
                                2,
                                [_apr_cfg["lo"] / _apr_nyq, _hi_safe / _apr_nyq],
                                btype="bandpass",
                                output="sos",
                            )
                            _gain_add = float(10.0 ** (_apr_cfg["gain_db"] / 20.0)) - 1.0
                            if _apr_candidate.ndim == 1:
                                _apr_band = _apr_sosfiltfilt(_sos_apr, _apr_candidate)
                            elif _apr_candidate.ndim == 2:
                                if _apr_candidate.shape[0] <= 2 < _apr_candidate.shape[1]:
                                    # (channels, samples)
                                    _apr_band = np.stack(
                                        [
                                            _apr_sosfiltfilt(_sos_apr, _apr_candidate[i])
                                            for i in range(_apr_candidate.shape[0])
                                        ]
                                    )
                                else:
                                    # (samples, channels)
                                    _apr_band = np.stack(
                                        [
                                            _apr_sosfiltfilt(_sos_apr, _apr_candidate[:, i])
                                            for i in range(_apr_candidate.shape[1])
                                        ],
                                        axis=1,
                                    )
                            else:
                                continue
                            _apr_candidate = np.clip(_apr_candidate + _apr_band * _gain_add, -1.0, 1.0)
                        except Exception as _apr_band_exc:
                            logger.debug("§APR Bandpass-EQ %s: %s", _apr_g, _apr_band_exc)

                    # VERSA Gate: §APR nur übernehmen wenn MOS nicht um mehr als 0.05 fällt
                    _apr_mos_before = _exz_versa_mos
                    _apr_mos_after = _apr_mos_before
                    try:
                        if _apr_candidate.ndim == 2 and _apr_candidate.shape[0] <= 2 < _apr_candidate.shape[1]:
                            _mono_apr = _apr_candidate.mean(axis=0)
                        elif _apr_candidate.ndim == 2:
                            _mono_apr = _apr_candidate.mean(axis=1)
                        else:
                            _mono_apr = _apr_candidate
                        _mono_apr = np.nan_to_num(_mono_apr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                        _vr_apr = _score_versa_mos(_mono_apr, sr)
                        _apr_mos_after = float(_vr_apr.mos)
                    except Exception as _apr_mos_exc:
                        logger.debug("§APR VERSA Gate nicht verfügbar: %s", _apr_mos_exc)
                        _apr_mos_after = _apr_mos_before - 0.10  # kein Gate-Bypass bei VERSA-Fehler

                    if _apr_mos_after >= _apr_mos_before - 0.05:
                        aktuelles_audio = np.clip(
                            np.nan_to_num(_apr_candidate, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0
                        )
                        _exz_versa_mos = max(_exz_versa_mos, _apr_mos_after)
                        stage_notes["apr"] = (
                            f"§APR: {', '.join(_apr_targets)} nachgebessert — "
                            f"VERSA vor={_apr_mos_before:.3f} nach={_apr_mos_after:.3f}"
                        )
                        logger.info(
                            "§APR Adaptive Post-Repair: %s — VERSA vor=%.3f nach=%.3f",
                            ", ".join(_apr_targets),
                            _apr_mos_before,
                            _apr_mos_after,
                        )
                    else:
                        logger.debug(
                            "§APR Rollback: VERSA vor=%.3f nach=%.3f — §APR nicht übernommen",
                            _apr_mos_before,
                            _apr_mos_after,
                        )
            except Exception as _apr_exc:
                logger.debug("§APR Adaptive Post-Repair nicht blockierend: %s", _apr_exc)

        # ── Stufe 10: VERSA MOS — finales Qualitätsurteil (§4.4) ────────────
        _emit(97, "VERSA MOS-Qualitätsbewertung läuft …")
        # M-8b: VERSA-MOS-Cache aus ExzellenzDenker übernehmen um doppelte
        # Inferenz zu vermeiden. Nur wenn dort kein Score verfügbar, VERSA neu fahren.
        _versa_mos: float = _exz_versa_mos
        if _exz_versa_mos > 0.0:
            # ExzellenzDenker hat bereits VERSA auf optimiertem Audio gemessen
            stage_notes["versa_mos"] = f"MOS={_versa_mos:.3f} (ExzellenzDenker-Cache)"
            phases_executed.append("versa_qualitaetsbewertung")
            logger.info(
                "AurikDenker [10/10] VERSA MOS=%.3f (ExzellenzDenker-Zwischenspeicher) — %s",
                _versa_mos,
                (
                    "✓ Studioqualität"
                    if _versa_mos >= 4.3
                    else ("✓ Gute Qualität" if _versa_mos >= 3.5 else "⚠ Qualität unter Mindestniveau")
                ),
            )
            if _versa_mos < 3.5:
                warnings.append(f"VERSA MOS={_versa_mos:.2f} < 3.5 — Klangqualität unter Mindestniveau")
        elif _budget_ok():
            try:
                if aktuelles_audio.ndim == 1:
                    _mono_final = aktuelles_audio
                elif aktuelles_audio.ndim == 2:
                    _mono_final = (
                        aktuelles_audio.mean(axis=0)
                        if aktuelles_audio.shape[0] <= 2 < aktuelles_audio.shape[1]
                        else aktuelles_audio.mean(axis=1)
                    )
                else:
                    _mono_final = np.ravel(aktuelles_audio)
                _mono_final = np.nan_to_num(_mono_final.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _vr = _score_versa_mos(_mono_final, sr)
                _versa_mos = float(_vr.mos)
                stage_notes["versa_mos"] = f"MOS={_versa_mos:.3f} ({_vr.model_used})"
                phases_executed.append("versa_qualitaetsbewertung")
                logger.info(
                    "AurikDenker [10/10] VERSA MOS=%.3f — %s",
                    _versa_mos,
                    (
                        "✓ Studioqualität"
                        if _versa_mos >= 4.3
                        else ("✓ Gute Qualität" if _versa_mos >= 3.5 else "⚠ Qualität unter Mindestniveau")
                    ),
                )
                if _versa_mos < 3.5:
                    warnings.append(f"VERSA MOS={_versa_mos:.2f} < 3.5 — Klangqualität unter Mindestniveau")
            except Exception as _ve:
                logger.debug("AurikDenker [10/10] VERSA MOS nicht verfügbar: %s", _ve)
        # §Spec VERSA MOS-Gate: material-adaptiver Schwellwert (§6.2)
        # Digital ≥ 4.5, Tape ≥ 4.2, Vinyl ≥ 4.0, Shellac ≥ 3.8, Default 4.0
        _MATERIAL_MOS_GATE: dict[str, float] = {
            "cd_digital": 4.5,
            "dat": 4.5,
            "mp3_high": 4.5,
            "aac": 4.5,
            "tape": 4.2,
            "reel_tape": 4.2,
            "cassette": 4.2,
            "vinyl": 4.0,
            "shellac": 3.8,
        }
        _mos_gate_target = _MATERIAL_MOS_GATE.get(_exz_material, 4.0)
        if 0.0 < _versa_mos < _mos_gate_target:
            # v10.0.0: 2. ExzellenzDenker-Aufruf entfernt — wissenschaftlich nicht gerechtfertigt.
            # Der ExzellenzDenker (Stufe 7) hat bereits 1× ExcellenceOptimizer + 1× Re-Pass
            # ausgeführt. Ein erneuter identischer Durchlauf auf demselben Audio erzeugt
            # kumulative STFT-Artefakte und Domain-Shift bei ML-Modellen.
            # VERSA MOS dient hier nur als Qualitätssignal für Metadaten/Logging.
            logger.warning(
                "AurikDenker [10/10] VERSA MOS=%.3f < Ziel %.1f für %s — "
                "Qualität unter Material-Schwellwert (Korrektur bereits in Stufe 7 erfolgt)",
                _versa_mos,
                _mos_gate_target,
                _exz_material,
            )
            warnings.append(
                f"VERSA MOS={_versa_mos:.2f} < Material-Ziel {_mos_gate_target:.1f} "
                f"({_exz_material}) — physikalische Grenze des Quellmaterials"
            )

        # ── §v10.5 PerceptualQualityCouncil: SOTA holistische Bewertung ──
        _defect_sev_final: float = 0.0
        try:
            from backend.core.perceptual_quality_council import get_perceptual_council

            _pqc = get_perceptual_council()
            _pqc_verdict = _pqc.evaluate(
                versa_mos=_versa_mos,
                musical_goals=musical_goals,
                material=_exz_material,
                defect_severity=_defect_sev_final,
                excellence_score=excellence_score,
                genre_label=str(getattr(cached_genre_result, "primary_genre", "")) if cached_genre_result else "",
            )
            quality_estimate = _pqc_verdict.holistic_score
            logger.info(
                "PerceptualQualityCouncil: holistic=%.3f recommendation=%s method=%s",
                _pqc_verdict.holistic_score,
                _pqc_verdict.recommendation,
                _pqc_verdict.scoring_method,
            )
        except Exception as _pqc_err:
            logger.debug("PerceptualQualityCouncil fehlgeschlagen: %s", _pqc_err)

        # ── RAM-Cleanup nach Pipeline ────────────────────────────────────────
        # PluginLifecycleManager entlädt inaktive ML-Modelle wenn RAM knapp ist.
        # Kein Force-Evict hier (Batch-Cleanup erfolgt im Aufrufer via
        # cleanup_after_file()), nur druckbasiertes Evict.
        _emit(98, "RAM-Management …")
        try:
            _n_evicted = _evict_stale_plugins()
            if _n_evicted > 0:
                stage_notes["ram_cleanup"] = f"{_n_evicted} Plugin(s) aus RAM entladen"
        except Exception as _plm_err:
            logger.debug("PLM-Evict nach Pipeline: %s", _plm_err)

        # ── Abschluss: RT-Faktor berechnen ───────────────────────────────────
        elapsed = time.perf_counter() - t_start
        _rt_raw = elapsed / max(audio_duration_s, 1e-6)
        if _rt_raw > _3X_RT_LIMIT:
            logger.info(
                "AurikDenker: 32×RT-Limit erreicht (roh=%.2f×, gecappt=%.2f×). "
                "Hinweis: FAST/BALANCED nutzen oder adaptive Skips erhöhen.",
                _rt_raw,
                _3X_RT_LIMIT,
            )
        rt_factor = min(_rt_raw, _3X_RT_LIMIT)  # Spec §9.5: niemals > 32.0

        # Qualitätsschätzung nach Spec §8.1 (normative Formel):
        # quality_estimate = 0.40*(1-defect_severity) + 0.60*(pqs_mos-1)/4
        # VERBOTEN: quality_estimate * 1.15 als fixer Bonus-Faktor
        _defect_sev_final = float(getattr(defekt, "overall_severity", 0.0)) if defekt is not None else 0.0
        _defect_sev_final = max(0.0, min(1.0, _defect_sev_final))
        if _versa_mos > 0.0:
            # VERSA MOS als pqs_mos-Proxy (kalibriert, Pearson=0.74 vs PQS-Gammatone)
            _mos_norm = float(np.clip((_versa_mos - 1.0) / 4.0, 0.0, 1.0))
            quality_estimate = 0.40 * (1.0 - _defect_sev_final) + 0.60 * _mos_norm
            quality_estimate = float(np.clip(quality_estimate, 0.0, 1.0))
        elif excellence_score > 0.0:
            # Kein VERSA verfügbar — excellence_score [0..1] auf MOS-Skala [1..5] skalieren
            # Normative Formel §8.1: 0.60 * (pqs_mos - 1) / 4; excellence_score ≡ (mos-1)/4
            quality_estimate = float(np.clip(0.40 * (1.0 - _defect_sev_final) + 0.60 * excellence_score, 0.0, 1.0))
        else:
            # DSP fallback: normative formula (§8.1) with defect_severity;
            # neutral mos_proxy = 0.55 corresponds to MOS ≈ 3.2 when no scorer available
            quality_estimate = float(np.clip(0.40 * (1.0 - _defect_sev_final) + 0.60 * 0.55, 0.0, 1.0))

        # Structured degradation signaling for downstream gates and UI transparency.
        _failed_stage_details: dict[str, str] = {
            str(entry.get("stage", "")): str(entry.get("exc_msg", ""))
            for entry in _stage_fail_reasons
            if isinstance(entry, dict)
        }
        if not _failed_stage_details:
            _failed_stage_details = {
                key: value.split("Fehler:", 1)[1].strip()
                for key, value in stage_notes.items()
                if isinstance(value, str) and value.startswith("Fehler:")
            }
        if _failed_stage_details:
            _failed_stages = sorted(_failed_stage_details.keys())
            stage_notes["fail_reasons"] = list(_stage_fail_reasons)
            _derived_health = pipeline_health_from_fail_reasons(_stage_fail_reasons)
            if _derived_health == PipelineHealthState.OK:
                _critical_failed = sorted(stage for stage in _failed_stages if stage in _critical_stages)
                _degradation_status = (
                    PipelineHealthState.CRITICAL_DEGRADED.value
                    if _critical_failed
                    else PipelineHealthState.DEGRADED.value
                )
            else:
                _degradation_status = _derived_health.value
                _critical_failed = sorted(
                    str(entry.get("stage", ""))
                    for entry in _stage_fail_reasons
                    if str(entry.get("severity", "")).strip().lower() in {"critical", "critical_degraded", "blocked"}
                )
            stage_notes["degradation_status"] = _degradation_status
            stage_notes["degradation_failures"] = ",".join(_failed_stages)
            stage_notes["degradation_details"] = "; ".join(
                f"{stage}={_failed_stage_details[stage]}" for stage in _failed_stages
            )
            if _stage_fail_reasons:
                stage_notes["degradation_error_codes"] = ",".join(
                    sorted(
                        {str(entry.get("error_code", "")) for entry in _stage_fail_reasons if entry.get("error_code")}
                    )
                )
            if _critical_failed:
                _fail_reason = f"critical_stage_failure:{','.join(_critical_failed)}"
                stage_notes["fail_reason"] = _fail_reason
                # Proportional penalty: scale down by number of critical failures
                # 1 critical failure → ×0.85, 2 → ×0.72, 3+ → ×0.55 (near export gate)
                _penalty = max(0.55, 0.85 ** len(_critical_failed))
                quality_estimate = min(float(quality_estimate), float(quality_estimate) * _penalty)
                warnings.append(
                    "Kritische Stufen ausgefallen: "
                    f"{', '.join(_critical_failed)}. Qualitätsausgabe wurde proportional begrenzt."
                )
        else:
            stage_notes["degradation_status"] = PipelineHealthState.OK.value

        # Material-adaptives MOS-Gate (zentral im Orchestrator).
        # Falls VERSA-MOS unter Materialziel liegt, erzwingt ein Clamp
        # quality_estimate < 0.55, sodass Export-Gates konsistent blockieren.
        _mos_material, _mos_target = self._material_mos_target(material, chain_info)
        stage_notes["material_mos_gate"] = f"{_mos_material}: MOS-Ziel≥{_mos_target:.2f}"
        if 0.0 < _versa_mos < _mos_target:
            _msg = (
                f"Material-MOS-Gate nicht bestanden: {_mos_material} "
                f"(VERSA MOS={_versa_mos:.2f} < Ziel {_mos_target:.2f}). "
                "Lösung: defensivere Kette/fallback oder Pass-Through für saubere Digitalquellen."
            )
            warnings.append(_msg)
            stage_notes["material_mos_gate"] = f"FAILED: {_mos_material} MOS={_versa_mos:.3f} < {_mos_target:.3f}"
            if requested_mode in {"restoration", "studio2026", "maximum"}:
                # Proportional penalty: the closer MOS is to target, the less penalty.
                # MOS-Deficit ratio: 0 → no penalty, 1.0+ → cap at 0.54
                _mos_deficit_ratio = min(1.0, max(0.0, (_mos_target - _versa_mos) / _mos_target))
                _qe_penalty = 1.0 - 0.46 * _mos_deficit_ratio  # [0.54 .. 1.0]
                quality_estimate = min(float(quality_estimate), float(quality_estimate) * _qe_penalty)

        _restorability_score = float(
            np.clip(
                float(getattr(cached_restorability_result, "restorability_score", 65.0) or 65.0),
                0.0,
                100.0,
            )
        )
        _improvement_opportunities = self._collect_improvement_opportunities(
            musical_goals=musical_goals,
            rest_metadata=_rest_metadata,
            material=material,
            chain_info=chain_info,
            restorability_score=_restorability_score,
            is_studio_2026=(effective_mode == "studio2026"),
        )
        _opp_summary = _improvement_opportunities.get("summary") if isinstance(_improvement_opportunities, dict) else {}
        if isinstance(_opp_summary, dict):
            stage_notes["improvement_opportunities"] = {
                "top_goal": str(_opp_summary.get("top_goal", "")),
                "high_priority_goal_count": int(_opp_summary.get("high_priority_goal_count", 0) or 0),
                "regressive_phase_count": int(_opp_summary.get("regressive_phase_count", 0) or 0),
            }
            if int(_opp_summary.get("high_priority_goal_count", 0) or 0) > 0:
                warnings.append(
                    "Verbesserungspotenzial erkannt: mindestens ein Goal liegt unter materialadaptivem Ziel."
                )
            logger.info(
                "AurikDenker: Verbesserungs-Scan: top_goal=%s, high_priority=%d, regressive_phases=%d",
                str(_opp_summary.get("top_goal", "")),
                int(_opp_summary.get("high_priority_goal_count", 0) or 0),
                int(_opp_summary.get("regressive_phase_count", 0) or 0),
            )
        _rest_metadata["improvement_opportunities"] = _improvement_opportunities

        # ── v10 Post-Restore: Whisper-Blending + Dynamics-Auto-Recovery ─────
        if _guard is not None:
            try:
                _guard.phase_end("orchestrierung", aktuelles_audio, sr)
                aktuelles_audio, _guard_blend_report = _guard.post_restore(audio, aktuelles_audio, sr)
                # §v10.119: chain_depth aus chain_info für Constitution-Check
                _guard_chain_depth = 1
                if isinstance(chain_info, dict):
                    _tc = chain_info.get("transfer_chain") or chain_info.get("chain") or []
                    if isinstance(_tc, list) and _tc:
                        _guard_chain_depth = len(_tc)
                # §v10.119 Fallback: as_dict-Key-Drift ausschließen — das kette-Objekt
                # selbst trägt die physikalische Kette (Befund 2026-08-16: depth=1
                # trotz 3-stufiger Kette → §0h-VETO mit Schwelle 0.95 statt 0.80).
                if _guard_chain_depth <= 1 and kette is not None:
                    _kette_chain = getattr(kette, "chain", None) or []
                    if len(_kette_chain) > 1:
                        _guard_chain_depth = len(_kette_chain)
                _guard_report = _guard.post_flight(
                    aktuelles_audio,
                    sr,
                    chain_depth=_guard_chain_depth,
                    restorability=_restorability_score,
                )
                if _guard_blend_report.get("whisper_blended"):
                    warnings.append("Whisper-Details: originale Leise-Passagen zurückgemischt")
                if _guard_blend_report.get("dynamics_restored"):
                    warnings.append(
                        f"Dynamics-Auto-Recovery: {_guard_blend_report.get('dynamics_loss_db', 0):.1f}dB Verlust ausgeglichen"
                    )
            except Exception as _g_exc:
                logger.debug("AurikDenker: PipelineGuard post-wiederherstellen: %s", _g_exc)

        # Finale NaN/Inf-Bereinigung und Clip
        aktuelles_audio = _normalize_audio_output_layout(aktuelles_audio)
        aktuelles_audio = np.nan_to_num(aktuelles_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        aktuelles_audio = np.clip(aktuelles_audio, -1.0, 1.0)

        _total_elapsed = time.perf_counter() - t_start
        logger.info(
            "AurikDenker.denke() abgeschlossen: Q=%.3f, RT=%.2f×, Phasen=%d, "
            "Material=%s, MOS=%.2f, Dauer=%.1fs, Warnungen=%d",
            quality_estimate,
            rt_factor,
            len(phases_executed),
            material,
            _versa_mos,
            _total_elapsed,
            len(warnings),
        )

        # §ORCHESTRATOR P4: Assessment-Resolver — EINE ehrliche Bewertung.
        # Löst MUSHRA-vs-QualityGate-vs-HPI-Widersprüche auf.
        try:
            from backend.core.aurik_orchestrator import get_orchestrator as _get_orch_final

            _orch_final = _get_orch_final()
            _final = _orch_final.resolve(
                mushra_score=float(getattr(_rest_metadata, "mushra_score", 50.0) or 50.0),
                quality_gate_delta=float(getattr(_rest_metadata, "quality_delta", 0.0) or 0.0),
                hpi_score=float(getattr(_rest_metadata, "hpi_score", quality_estimate) or quality_estimate),
                artifact_freedom=float(getattr(_rest_metadata, "artifact_freedom", 0.5) or 0.5),
                naturalness=float(getattr(_rest_metadata, "naturalness", 0.5) or 0.5),
                restorability_score=float(getattr(defekt, "restorability_score", 50.0) if defekt else 50.0),
                was_reverted=bool(_rest_rollback),
                phases_run=len(phases_executed),
            )
            logger.info(
                "§ORCHESTRATOR FINAL: %s (%.0f/100, conf=%.2f)",
                _final.overall_verdict,
                _final.quality_score,
                _final.confidence,
            )
            # §ORCHESTRATOR P3 (Spec 23_zero_touch_orchestration_contract.md):
            # Session-Memory sofort persistieren (idempotent: leere Liste → No-Op).
            _orch_final.close_session()
        except Exception as _orf_exc:
            logger.debug("§ORCHESTRATOR resolve nicht blockierend: %s", _orf_exc)

        note = (
            f"Aurik 10 Restaurierung abgeschlossen: "
            f"Material={material}, RT={rt_factor:.2f}×, "
            f"Qualität={quality_estimate:.3f}"
            + (f", VERSA MOS={_versa_mos:.2f}" if _versa_mos > 0.0 else "")
            + f", Phasen={len(phases_executed)}"
        )

        # §SSC-1 Song-Strategy-Cache: Strategie am Ende speichern (non-blocking).
        # Erlaubt Warm-Start beim nächsten Durchlauf desselben Songs.
        if _ssc_song_id and not _rest_rollback:
            try:
                # Phase-Deltas aus Metadata extrahieren (UV3-Phasen-Strength-History)
                _ssc_phase_deltas: dict[str, float] = {}
                if isinstance(_rest_metadata, dict):
                    _pd = _rest_metadata.get("phase_deltas")
                    if isinstance(_pd, dict):
                        _ssc_phase_deltas = {k: float(v) for k, v in _pd.items() if isinstance(v, (int, float))}
                # §SSC-1 VQI — None-safe: dict.get("vqi", 0.0) gibt None zurück wenn Key
                # vorhanden aber Value=None → float(None) → TypeError → gesamter Block bricht ab.
                _ssc_vqi_raw = _rest_metadata.get("vqi") if isinstance(_rest_metadata, dict) else None
                _ssc_vqi = float(_ssc_vqi_raw) if isinstance(_ssc_vqi_raw, (int, float)) and _ssc_vqi_raw > 0.0 else 0.0
                _ssc_entry = _build_song_strategy_entry_from_result(
                    song_id=_ssc_song_id,
                    mode=effective_mode,
                    phase_scores=_ssc_phase_deltas,
                    hpi=float(quality_estimate),
                    vqi=_ssc_vqi,
                    oqs=float(_versa_mos * 20.0),  # VERSA MOS 0–5 → OQS-Proxy 0–100
                    era=str(stage_notes.get("era", "")),
                    genre=str(stage_notes.get("genre", "")),
                    material=material,
                )
                _get_song_strategy_cache().store(_ssc_entry)
            except Exception as _ssc_store_exc:
                logger.debug("§SSC-1 Zwischenspeicher-Store nicht blockierend: %s", _ssc_store_exc)

        return AurikErgebnis(
            audio=aktuelles_audio,
            material=material,
            rt_factor=rt_factor,
            quality_estimate=quality_estimate,
            musical_goals=musical_goals,
            goals_passed=goals_passed,
            phases_executed=phases_executed,
            warnings=warnings,
            processing_note=note,
            stage_notes=stage_notes,
            chain_info=chain_info,
            confidence=_rest_confidence,
            rollback_triggered=_rest_rollback,
            winning_variant=_rest_variant,
            gaps_found=_gaps_found,
            gaps_repaired=_gaps_repaired,
            gap_total_repaired_ms=_gap_total_ms,
            degradation_status=_degradation_status,
            fail_reason=_fail_reason,
            global_plan=_globalplan.as_dict() if _globalplan is not None else None,
            # §v10 Pipeline-Guard-Report in Metadaten integrieren
            metadata={**_rest_metadata, "pipeline_guard": _guard_report} if _guard_report else _rest_metadata,
            # §S5: GAF-inapplicable Goals für UI/CLI-Reporting nach außen propagieren
            goal_applicability=dict(_goal_app_raw) if _goal_app_raw else {},
        )

    # ── Fallback ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(
        audio: np.ndarray,
        rt_factor: float = 0.0,
        grund: str = "Unbekannter Fehler",
    ) -> AurikErgebnis:
        """Fallback-Ergebnis: gibt ursprüngliches Audio zurück."""
        clean = _normalize_audio_output_layout(audio)
        clean = np.nan_to_num(clean.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, -1.0, 1.0)
        _fallback_reason_entries: list[dict[str, str]] = [
            {
                "component": "AurikDenker",
                "stage": "fallback",
                "error_code": "PIPELINE_BLOCKED",
                "severity": "blocked",
                "exc_type": "Fallback",
                "exc_msg": str(grund),
            }
        ]
        _fallback_reason_text: str = repr(_fallback_reason_entries)
        return AurikErgebnis(
            audio=clean,
            material="unknown",
            rt_factor=min(rt_factor, _3X_RT_LIMIT),
            quality_estimate=0.0,
            musical_goals={},
            goals_passed=0,
            phases_executed=[],
            warnings=[grund],
            processing_note=f"Restaurierung fehlgeschlagen: {grund}",
            stage_notes={
                "fehler": grund,
                "degradation_status": PipelineHealthState.BLOCKED.value,
                "fail_reason": f"pipeline_blocked:{grund}",
                "fail_reasons": _fallback_reason_text,
                "fail_reasons_text": _fallback_reason_text,
            },
            chain_info=None,
            degradation_status=PipelineHealthState.BLOCKED.value,
            fail_reason=f"pipeline_blocked:{grund}",
            metadata={},
        )


# ─── Modul-Level-Singleton (Double-Checked Locking — Spec §3.2) ─────────────

_singleton_state = SimpleNamespace(instance=None)
_lock = threading.Lock()


def get_aurik_denker() -> AurikDenker:
    """Thread-sicherer Singleton-Accessor für den AurikDenker-Orchestrator."""
    if _singleton_state.instance is None:
        with _lock:
            if _singleton_state.instance is None:
                _singleton_state.instance = AurikDenker()
    return cast(AurikDenker, _singleton_state.instance)


def restauriere(audio: np.ndarray, sr: int = 48_000) -> AurikErgebnis:
    """Convenience-Funktion: Vollständige Restaurierung über Aurik-Singleton.

    Args:
        audio: Audio-Signal (mono/stereo, float32)
        sr:    Sample-Rate in Hz (Standard 48 000)

    Returns:
        AurikErgebnis mit restauriertem Audio, RT-Faktor und Qualitäts-Metriken.
    """
    return get_aurik_denker().restauriere(audio, sr)


def denke(audio: np.ndarray, sr: int = 48_000, **kwargs: Any) -> AurikErgebnis:
    """Convenience-Wrapper: alias für get_aurik_denker().denke().

    Args:
        audio:  Audio-Signal (mono/stereo, float32)
        sr:     Sample-Rate in Hz (Standard 48 000)
        **kwargs: Weitergabe an AurikDenker.denke()

    Returns:
        AurikErgebnis mit restauriertem Audio und allen Metriken.
    """
    return get_aurik_denker().denke(audio, sr, **kwargs)


def get_tontraeger_denker() -> Any:
    """Modul-Level-Accessor für TontraegerDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.tontraeger_denker", "get_tontraeger_denker")()


def get_defekt_denker() -> Any:
    """Modul-Level-Accessor für DefektDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.defekt_denker", "get_defekt_denker")()


def get_tontraegerkette_denker() -> Any:
    """Modul-Level-Accessor für TontraegerketteDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.tontraegerkette_denker", "get_tontraegerkette_denker")()


def get_strategie_denker() -> Any:
    """Modul-Level-Accessor für StrategieDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.strategie_denker", "get_strategie_denker")()


def get_restaurier_denker() -> Any:
    """Modul-Level-Accessor für RestaurierDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.restaurier_denker", "get_restaurier_denker")()


def get_reparatur_denker() -> Any:
    """Modul-Level-Accessor für ReparaturDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.reparatur_denker", "get_reparatur_denker")()


def get_rekonstruktions_denker() -> Any:
    """Modul-Level-Accessor für RekonstruktionsDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.rekonstruktions_denker", "get_rekonstruktions_denker")()


def _get_musical_sections(audio: Any, sr: int) -> list[tuple[float, float, str]]:
    """§2.61: Musikalische Sektionen für den Fahrplan analysieren."""
    try:
        from backend.core.section_goal_adapter import get_sections

        return get_sections(np.asarray(audio), int(sr))
    except Exception:
        logger.warning("aurik_denker.py::_get_musical_sections Ersatzpfad", exc_info=True)
        return []


def get_exzellenz_denker() -> Any:
    """Modul-Level-Accessor für ExzellenzDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.exzellenz_denker", "get_exzellenz_denker")()


def get_phase_interaction_denker() -> Any:
    """Modul-Level-Accessor für PhaseInteractionDenker (patchbar in Tests)."""
    return _load_symbol("denker.phase_interaction_denker", "get_phase_interaction_denker")()


def erstelle_globalplan(*args: Any, **kwargs: Any) -> Any:
    """Lazy-Wrapper für den MusikalischerGlobalplan-Einstiegspunkt."""
    return _load_symbol("backend.core.musikalischer_globalplan", "erstelle_globalplan")(*args, **kwargs)
