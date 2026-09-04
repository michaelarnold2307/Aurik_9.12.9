"""Aurik 10 — Bridge: Infrastruktur / Speicher-Management (§2.37)
===================================================================
Infrastruktur-Funktionen für Frontend/CLI → Backend-Core.

Enthält:
  - PluginLifecycleManager-Singleton (LRU-Eviction, RAM-Trigger 82%)
  - ML-Memory-Budget-Status + Import-Telemetrie (§2.37)
  - Warmup Models Background (§9.7.4, Tier-1/Tier-2 mit RAM-Guard)
  - ROCm GPU-Warmup (HIP JIT cold-start-Latenz eliminieren)
  - DeferredRefinementJob-Klasse (§2.38 KMV Stufe 2)
  - Recovery Checkpoint Functions (§2.39 OOM-Recovery)
  - Era/Medium Constraint (MEDIUM_DECADE_FLOOR, constrain_era_to_medium)
  - ML-Memory-Budget-Singleton (§2.37 try_allocate/release)
  - ModelDownloader-Singleton (§9.x / §13.x startup self-heal)
  - PresenceEmbedding + EraAuthenticPerceptualCompletion (§G90)
  - RollbackSanityGuard (§G92)
  - PreviewMode für 30s-Vorschau (§ROADMAP-5)
  - ArtistFingerprintStore (§13.11)
  - PluginRegistry (Aurik10/ui/plugin_manager.py)

Referenz: AGENTS.md §1 (Normative Kette), .github/copilot-instructions.md §V4 Bridge-Verbot.
"""

# pylint: disable=import-outside-toplevel
# cspell:disable

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status locks & stubs (ML-Memory-Budget)
# ---------------------------------------------------------------------------

_ml_memory_budget_status_lock = threading.Lock()
_ml_memory_budget_import_status: dict[str, Any] = {
    "available": True,
    "failures": 0,
    "last_error": "",
}


# ---------------------------------------------------------------------------
# PluginLifecycleManager-Singleton (LRU-Eviction, RAM-Trigger 82%)
# ---------------------------------------------------------------------------


def get_plugin_lifecycle_manager():
    """Gibt den ``PluginLifecycleManager``-Singleton zurück (lazy import, §2.37).

    Der PLM ist **Schicht 2** des zweischichtigen OOM-Schutzsystems:

    - **Schicht 1**: ``ml_memory_budget.try_allocate()`` — logisch
    - **Schicht 2**: ``PluginLifecycleManager`` — physisch (LRU-Eviction)

    RAM-Trigger: 82 % Systemauslastung → LRU-Eviction bis < 70 % oder
    ≥ 1,5 GB frei. Monitoring-Thread alle 10 Sekunden.

    Verwendung::

        plm = get_plugin_lifecycle_manager()
        plm.register("MeinPlugin", size_gb=0.10, unload_fn=lambda: ...)
        plm.set_active("MeinPlugin", True)   # schützt vor Eviction

    VERBOTEN: ``plm.try_allocate()`` — Methode existiert nicht!
    Verwende stattdessen ``ml_memory_budget.try_allocate()``.
    """
    from backend.core.plugin_lifecycle_manager import (  # type: ignore[import]
        get_plugin_lifecycle_manager as _get,
    )

    return _get()


# ---------------------------------------------------------------------------
# ML-Memory-Budget-Status + Import-Telemetrie (§2.37)
# ---------------------------------------------------------------------------


def get_ml_memory_budget_status() -> dict:
    """Gibt den aktuellen ML-Speicherbudget-Status als Dict zurück (lazy import, §2.37).

    Rückgabe-Keys (Beispiel)::

        {
            "budget_gb": 10.7,
            "allocated_gb": 3.2,
            "free_gb": 7.5,
            "plugins": {"fcpe": 0.12, "panns": 0.44, ...},
        }

    Das Budget wird automatisch auf ``RAM/3, capped [4–12 GB]`` gesetzt.
    Auf 32-GB-System: ≈ 10.7 GB (Cap: 12 GB).

    WARNUNG: Fehlt ``psutil``, sind physische RAM-Checks deaktiviert —
    ``psutil`` muss im AppImage gebündelt sein.
    """
    try:
        from backend.core.ml_memory_budget import get_status  # type: ignore[import]

        _status = get_status()
        with _ml_memory_budget_status_lock:
            _ml_memory_budget_import_status["available"] = True
            _ml_memory_budget_import_status["last_error"] = ""
        return _status  # type: ignore[no-any-return]
    except Exception as _e:
        _err = f"{type(_e).__name__}: {_e}"
        with _ml_memory_budget_status_lock:
            _ml_memory_budget_import_status["available"] = False
            _ml_memory_budget_import_status["failures"] = int(_ml_memory_budget_import_status.get("failures", 0)) + 1
            _ml_memory_budget_import_status["last_error"] = _err
        logger.warning("bridge: ml_memory_Grenze.get_status() nicht verfügbar: %s", _err)
        return {"max_gb": 0.0, "allocated_gb": 0.0, "free_gb": 0.0, "models": {}}


def get_ml_memory_budget_import_status() -> dict[str, Any]:
    """Liefert Bridge-Telemetrie für ml_memory_budget-Importstatus."""
    with _ml_memory_budget_status_lock:
        return dict(_ml_memory_budget_import_status)


# ---------------------------------------------------------------------------
# Warmup Models Background (§9.7.4, Tier-1/Tier-2 mit RAM-Guard)
# ---------------------------------------------------------------------------


def warmup_models_background() -> None:
    """Initialisiert häufig genutzte ML-Modelle im Hintergrund vor.

    Kanonische Warmup-Funktion (§9.7.4). Wird 2 Sekunden nach App-Start
    als Daemon-Thread gestartet — aus ``ModernMainWindow.__init__`` via
    ``QTimer.singleShot(2000, ...)``. Fehler werden nur geloggt, kein Absturz.

    Der Caller (QTimer) steuert das Timing — kein zusätzliches sleep().
    Warmup berührt keinerlei UI-Objekte (kein GUI-Zugriff aus dem Thread).

    Plugin-Reihenfolge spiegelt §4.4-Priorisierung:
    Tier-1-Primär-Plugins zuerst (VAD/Pitch/Tagging), Fallbacks danach.

    §v10.305 G73: Warmup-Plugin-Namen MÜSSEN vor dem ersten Lauf validiert werden.
    §v10.306: RAM-bewusstes Staggered-Loading — kleine Modelle sofort,
    große Modelle nur bei ausreichend RAM (>20% frei).
    """
    import gc as _warmup_gc
    import importlib

    # §Spec 24 Root-Fix: Defensiv für Caller ohne startup_model_check —
    # serialisiert librosa, bevor dieser Daemon-Thread neben dem Hauptthread
    # läuft. Idempotent (No-Op, wenn der Hauptthread bereits fertig war).
    from backend.core.librosa_bootstrap import ensure_librosa_ready

    ensure_librosa_ready()

    # ── Tier-1: Kritische Sofort-Plugins (<100 MB, immer laden) ──────────
    _plugins_tier1 = [
        ("plugins.silero_plugin", "get_silero_plugin"),  # VAD (~1 MB)
        ("plugins.fcpe_plugin", "get_fcpe_plugin"),  # Pitch (~7 MB)
        (
            "plugins.crepe_plugin",
            "get_crepe_plugin",
        ),  # Pitch-Rückfall-Kaskade FCPE→RMVPE→PESTO→pYIN (RMVPE Tier-2, §4.4)
        ("plugins.beats_plugin", "get_beats_plugin"),  # Audio-Tagging (~10 MB)
        ("backend.core.noise_reduction", "get_noise_reducer"),  # DeepFilterNet v3.II (~15 MB)
        ("plugins.panns_plugin", "get_panns_plugin"),  # Audio-Tagging Primär (~66 MB)
        ("plugins.sgmse_plugin", "get_sgmse_plus_plugin"),  # Dereverb/Denoising (~12 MB)
    ]

    # ── Tier-2: Große Modelle — nur bei RAM-Reserve laden ───────────────
    _plugins_tier2 = [
        ("plugins.apollo_plugin", "get_apollo"),  # ~800 MB
        ("plugins.bs_roformer_plugin", "get_bs_roformer"),  # ~860 MB
        ("plugins.mdx23c_plugin", "get_mdx23c_plugin"),  # ~900 MB
        ("plugins.mert_plugin", "get_mert_plugin"),  # ~1.2 GB (async)
    ]

    logger.info("bridge: warmup gestartet (%d+%d plugins) …", len(_plugins_tier1), len(_plugins_tier2))
    _loaded = 0
    _failed = 0
    _deferred = 0

    # §v10.304.30: Keine GPU-Detection im Warmup. torch.zeros("cuda") hängt
    # auf manchen ROCm-Systemen → Warmup-Thread tot. GPU-Plugins werden
    # trotzdem geladen — wenn GPU nicht verfügbar, crashen sie und werden
    # von try/except gefangen. Warmup läuft GARANTIERT durch.
    def _load_one(_m: str, _a: str) -> bool:
        try:
            m = importlib.import_module(_m)
            fn = getattr(m, _a, None)
            if fn is not None:
                fn()
            return True
        except Exception as _e:
            logger.warning("bridge: %s.%s FEHLGESCHLAGEN: %s", _m, _a, _e)
            return False

    # ── RAM-Check-Helper ─────────────────────────────────────────────────
    def _ram_ok_for_large() -> bool:
        try:
            import psutil

            vm = psutil.virtual_memory()
            avail_pct = vm.available / max(vm.total, 1)
            # >28% frei (>8.7 GB bei 31 GB) = genug für große Modelle + Swap-Puffer
            if avail_pct < 0.28:
                return False
            # Swap-Guard: bei >65% Swap keine großen Modelle mehr laden
            # (Swap-Thrashing → C-Allokatoren crashen mit SIGSEGV)
            try:
                sw = psutil.swap_memory()
                if sw.percent > 65.0:
                    return False
            except Exception:
                logger.warning("§G23 bridge: ML-Modell-Validierung DSP-Ersatzpfad: %s", exc_info=True)
            return True
        except Exception:
            logger.warning(
                "§G93 bridge: ML-Modell-Ladeversuch fehlgeschlagen (Segfault-Risiko) → returning False", exc_info=True
            )
            return False  # §v10.306: Im Zweifel NICHT laden — Segfault-Risiko

    # ── Tier 1: Alle kleinen Modelle sofort laden ────────────────────────
    for _mod, _accessor in _plugins_tier1:
        try:
            if _load_one(_mod, _accessor):
                _loaded += 1
            else:
                _failed += 1
        except Exception as _sync_exc:
            _failed += 1
            logger.warning("bridge: %s.%s CRASH: %s", _mod, _accessor, _sync_exc)

    # GC nach Tier-1 — gibt temporäre Allokationen frei
    _warmup_gc.collect()

    # ── Tier 2: Große Modelle nur bei RAM-Reserve ────────────────────────
    for _mod, _accessor in _plugins_tier2:
        if "mert" in _mod.lower():
            # MERT immer async (1.2 GB, 160s Kaltstart)
            if _ram_ok_for_large():
                try:
                    _mert_thread = threading.Thread(
                        target=lambda m=_mod, a=_accessor: _load_one(m, a),
                        daemon=True,
                        name="aurik_warmup_mert",
                    )
                    _mert_thread.start()
                    _loaded += 1
                    logger.info("bridge: MERT async warmup gestartet")
                except Exception:
                    _failed += 1
            else:
                _deferred += 1
                logger.info("bridge: MERT deferred — RAM zu knapp")
            continue

        if _ram_ok_for_large():
            try:
                if _load_one(_mod, _accessor):
                    _loaded += 1
                else:
                    _failed += 1
            except Exception as _sync_exc:
                _failed += 1
                logger.warning("bridge: %s.%s CRASH: %s", _mod, _accessor, _sync_exc)
        else:
            _deferred += 1
            logger.info("bridge: %s deferred — RAM zu knapp", _mod.split(".")[-1])

    logger.info("bridge: warmup vollstaendig — %d geladen, %d fehlgeschlagen, %d deferred", _loaded, _failed, _deferred)
    # §v10.305 G73: Validiere alle Plugin-Zugriffsnamen (einmal pro Prozess)
    if _failed > 0:
        logger.warning(
            "bridge: %d Warmup-Plugins FEHLGESCHLAGEN — Zugriffsnamen in warmup_models_background() prüfen!",
            _failed,
        )


def warmup_rocm() -> None:
    """AMD ROCm GPU-Warmup — eliminiert HIP JIT cold-start-Latenz.

    Delegiert an ``ml_device_manager.warmup_rocm_gpu()``.
    Sicheres No-op auf CPU-only und non-AMD Systemen.
    """
    try:
        from backend.core.ml_device_manager import warmup_rocm_gpu as _wup

        _wup()
    except Exception as _exc:
        logger.debug("bridge.warmup_rocm: unkritisch: %s", _exc)


# ---------------------------------------------------------------------------
# §2.38 KMV + §2.39 OOM-Recovery + §2.37 RAM-Budget  (Lazy-Wrapper)
# ---------------------------------------------------------------------------


def get_deferred_refinement_job_class() -> type:
    """Gibt ``DeferredRefinementJob`` class (lazy import, §2.38 KMV Stufe 2) zurück.

    Used by MLRefinementThread and ModernMainWindow._maybe_start_kmv_refinement.
    """
    from backend.core.deferred_refinement_job import DeferredRefinementJob  # type: ignore[import]

    return DeferredRefinementJob  # type: ignore[no-any-return]


def get_save_checkpoint_fn():
    """Gibt ``save_checkpoint`` from recovery_checkpoint (lazy, §2.39) zurück."""
    from backend.core.recovery_checkpoint import save_checkpoint  # type: ignore[import]

    return save_checkpoint


def get_recovery_checkpoint_fns() -> tuple:
    """Gibt ``(cleanup_expired_checkpoints, find_pending_checkpoints, delete_checkpoint)`` (lazy, §2.39) zurück.

    Usage::

        cleanup_fn, find_fn, delete_fn = get_recovery_checkpoint_fns()
        cleanup_fn()
        checkpoints = find_fn()
        delete_fn(input_path)
    """
    from backend.core.recovery_checkpoint import (  # type: ignore[import]
        cleanup_expired_checkpoints,
        delete_checkpoint,
        find_pending_checkpoints,
    )

    return cleanup_expired_checkpoints, find_pending_checkpoints, delete_checkpoint


def get_era_medium_constraint() -> tuple:
    """Gibt ``(MEDIUM_DECADE_FLOOR, constrain_era_to_medium)`` from era_classifier (lazy import) zurück.

    Usage::

        floor_map, constrain_fn = get_era_medium_constraint()
        era = constrain_fn(era_result, medium_type)
        floor = floor_map.get(medium_type)
    """
    from backend.core.era_classifier import (  # type: ignore[import]
        MEDIUM_DECADE_FLOOR,
        constrain_era_to_medium,
    )

    return MEDIUM_DECADE_FLOOR, constrain_era_to_medium


def get_ml_memory_budget():
    """Gibt the ``MlMemoryBudget`` singleton (lazy import, §2.37) zurück.

    Usage::

        budget = get_ml_memory_budget()
        ok = budget.try_allocate("kmv_job", size_gb)
        budget.release("kmv_job")

    VERBOTEN: ``get_plugin_lifecycle_manager().try_allocate()`` — existiert nicht.
    """
    from backend.core.ml_memory_budget import get_ml_memory_budget as _get  # type: ignore[import]

    return _get()


def get_model_downloader():
    """Gibt the ``ModelDownloader`` singleton (lazy import, §9.x / §13.x) zurück.

    Used in Aurik startup self-heal to repair missing/corrupted bundled models.
    """
    from backend.core.model_downloader import get_model_downloader as _get  # type: ignore[import]

    return _get()


# ---------------------------------------------------------------------------
# §G90 PresenceEmbedding + EraAuthenticPerceptualCompletion (Aurik 10.14)
# ---------------------------------------------------------------------------


def get_presence_embedding():
    """Gibt die globale PresenceEmbedding-Instanz zurück (§G90)."""
    from backend.core.presence_embedding import get_presence_embedding as _fn  # type: ignore[import]

    return _fn()


def get_era_completion():
    """Gibt die globale EraAuthenticPerceptualCompletion-Instanz zurück (§G90)."""
    from backend.core.era_authentic_completion import get_era_completion as _fn  # type: ignore[import]

    return _fn()


def get_rollback_sanity_guard():
    """Gibt den globalen RollbackSanityGuard zurück (§G92)."""
    from backend.core.rollback_sanity_check import get_rollback_sanity_guard as _fn  # type: ignore[import]

    return _fn()


def get_preview_mode():
    """Gibt den PreviewMode für 30s-Vorschau zurück (§ROADMAP-5)."""
    from backend.core.preview_mode import get_preview_mode as _fn  # type: ignore[import]

    return _fn()


def get_artist_fingerprint_store():
    """Gibt den globalen ArtistFingerprintStore zurück (§13.11)."""
    from backend.core.artist_fingerprint import get_artist_fingerprint_store as _fn  # type: ignore[import]

    return _fn()


# ---------------------------------------------------------------------------
# Plugin-Registry — via bridge (§V4 (copilot-instructions.md) Bridge-Bypass-Verbot)
# ---------------------------------------------------------------------------


def get_plugin_registry():
    """Gibt die globale PluginRegistry-Instanz zurück (Aurik10/ui/plugin_manager.py)."""
    from backend.core.plugin_registry import get_plugin_registry as _get_plugin_registry

    return _get_plugin_registry()


# ---------------------------------------------------------------------------
# Public API — explizite Export-Liste
# ---------------------------------------------------------------------------

__all__ = [
    # PluginLifecycleManager-Singleton (LRU-Eviction, RAM-Trigger 82%)
    "get_plugin_lifecycle_manager",
    # ML-Memory-Budget-Status + Import-Telemetrie (§2.37)
    "get_ml_memory_budget_status",
    "get_ml_memory_budget_import_status",
    # Warmup Models Background (§9.7.4, Tier-1/Tier-2 mit RAM-Guard)
    "warmup_models_background",
    "warmup_rocm",
    # §2.38 KMV + §2.39 OOM-Recovery + §2.37 RAM-Budget
    "get_deferred_refinement_job_class",
    "get_save_checkpoint_fn",
    "get_recovery_checkpoint_fns",
    "get_era_medium_constraint",
    "get_ml_memory_budget",
    "get_model_downloader",
    # §G90 PresenceEmbedding + EraAuthenticPerceptualCompletion (Aurik 10.14)
    "get_presence_embedding",
    "get_era_completion",
    "get_rollback_sanity_guard",
    "get_preview_mode",
    "get_artist_fingerprint_store",
    # Plugin-Registry (§V4 (copilot-instructions.md) Bridge-Bypass-Verbot)
    "get_plugin_registry",
]
