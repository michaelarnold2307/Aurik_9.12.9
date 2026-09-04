from __future__ import annotations

"""Tests für backend/api/bridge_infrastructure.py — Infrastruktur-Bridge (§2.37, §9.7.4).

Prüft:
- Alle __all__-Einträge sind importierbar und aufrufbar
- PluginLifecycleManager-Singleton gibt konsistente Instanz zurück
- ML-Memory-Budget-Status ist immer ein Dict mit Pflicht-Keys
- Warmup-Funktionen laufen durch ohne Exception (defensiv)
- Recovery Checkpoint Functions sind callable
- Era/Medium Constraint liefert korrekte Typen
- ML-Memory-Budget Singleton hat try_allocate/release
- ModelDownloader-Singleton ist verfügbar
- PresenceEmbedding, EraCompletion, RollbackSanityGuard, PreviewMode, ArtistFingerprintStore
- PluginRegistry ist verfügbar
"""

import importlib
import pytest


@pytest.fixture(scope="module")
def bridge_infra():
    """Importiert die Infrastruktur-Bridge einmalig pro Modul."""
    return importlib.import_module("backend.api.bridge_infrastructure")


# ---------------------------------------------------------------------------
# 1. Grundlegender Import + __all__
# ---------------------------------------------------------------------------


class TestBridgeInfrastructureImport:
    """bridge_infrastructure ist importierbar und hat valides __all__."""

    def test_imports_cleanly(self, bridge_infra):
        assert bridge_infra is not None

    def test_has_all(self, bridge_infra):
        assert hasattr(bridge_infra, "__all__"), "__all__ fehlt"

    def test_all_entries_exist(self, bridge_infra):
        missing = [name for name in bridge_infra.__all__ if not hasattr(bridge_infra, name)]
        assert not missing, f"In __all__ deklariert, aber nicht im Modul: {missing}"

    def test_all_entries_callable_or_data(self, bridge_infra):
        for name in bridge_infra.__all__:
            obj = getattr(bridge_infra, name)
            assert obj is not None or name.startswith("_"), f"__all__ enthält None-Eintrag '{name}'"


# ---------------------------------------------------------------------------
# 2. PluginLifecycleManager-Singleton (LRU-Eviction, RAM-Trigger 82%)
# ---------------------------------------------------------------------------


class TestPluginLifecycleManager:
    """get_plugin_lifecycle_manager() gibt konsistente Singleton-Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        plm = bridge_infra.get_plugin_lifecycle_manager()
        assert plm is not None, "get_plugin_lifecycle_manager() gibt None zurück"

    def test_singleton_consistency(self, bridge_infra):
        """Zwei Aufrufe liefern dieselbe Instanz."""
        plm1 = bridge_infra.get_plugin_lifecycle_manager()
        plm2 = bridge_infra.get_plugin_lifecycle_manager()
        assert plm1 is plm2, "PluginLifecycleManager ist kein Singleton"

    def test_has_register_method(self, bridge_infra):
        plm = bridge_infra.get_plugin_lifecycle_manager()
        assert hasattr(plm, "register"), "PluginLifecycleManager fehlt .register()-Methode"


# ---------------------------------------------------------------------------
# 3. ML-Memory-Budget-Status + Import-Telemetrie (§2.37)
# ---------------------------------------------------------------------------


class TestMlMemoryBudgetStatus:
    """get_ml_memory_budget_status() gibt immer ein Dict mit Pflicht-Keys zurück."""

    def test_returns_dict(self, bridge_infra):
        result = bridge_infra.get_ml_memory_budget_status()
        assert isinstance(result, dict), f"Muss dict zurückgeben, nicht {type(result)}"

    def test_has_required_keys(self, bridge_infra):
        result = bridge_infra.get_ml_memory_budget_status()
        for key in ("max_gb", "allocated_gb", "free_gb", "models"):
            assert key in result, f"Pflicht-Key '{key}' fehlt"


class TestMlMemoryBudgetImportStatus:
    """get_ml_memory_budget_import_status() liefert Telemetrie."""

    def test_returns_dict_with_stable_keys(self, bridge_infra):
        status = bridge_infra.get_ml_memory_budget_import_status()
        assert isinstance(status, dict)
        for key in ("available", "failures", "last_error"):
            assert key in status


# ---------------------------------------------------------------------------
# 4. Warmup Models Background (§9.7.4)
# ---------------------------------------------------------------------------


class TestWarmupModelsBackground:
    """warmup_models_background() läuft durch ohne Exception."""

    def test_is_callable(self, bridge_infra):
        assert callable(bridge_infra.warmup_models_background)

    def test_completes_without_exception(self, bridge_infra):
        try:
            bridge_infra.warmup_models_background()
        except Exception as e:
            pytest.fail(f"warmup_models_background() wirft Exception: {e}")


class TestWarmupRocm:
    """warmup_rocm() ist sicherer No-op auf CPU-only Systemen."""

    def test_is_callable(self, bridge_infra):
        assert callable(bridge_infra.warmup_rocm)

    def test_completes_without_exception(self, bridge_infra):
        try:
            bridge_infra.warmup_rocm()
        except Exception as e:
            pytest.fail(f"warmup_rocm() wirft Exception: {e}")


# ---------------------------------------------------------------------------
# 5. §2.38 KMV + §2.39 OOM-Recovery + §2.37 RAM-Budget
# ---------------------------------------------------------------------------


class TestDeferredRefinementJobClass:
    """get_deferred_refinement_job_class() gibt eine Klasse zurück."""

    def test_returns_type(self, bridge_infra):
        result = bridge_infra.get_deferred_refinement_job_class()
        assert isinstance(result, type), f"Muss Klasse zurückgeben, nicht {type(result)}"


class TestSaveCheckpointFn:
    """get_save_checkpoint_fn() gibt einen Callable zurück."""

    def test_returns_callable(self, bridge_infra):
        fn = bridge_infra.get_save_checkpoint_fn()
        assert callable(fn), "Muss Callable zurückgeben"


class TestRecoveryCheckpointFns:
    """get_recovery_checkpoint_fns() gibt Tuple von 3 Callables zurück."""

    def test_returns_tuple_of_three(self, bridge_infra):
        result = bridge_infra.get_recovery_checkpoint_fns()
        assert isinstance(result, tuple) and len(result) == 3
        for fn in result:
            assert callable(fn), "Jedes Tuple-Element muss callable sein"


class TestEraMediumConstraint:
    """get_era_medium_constraint() gibt (MEDIUM_DECADE_FLOOR, constrain_fn) zurück."""

    def test_returns_tuple_of_two(self, bridge_infra):
        floor_map, constrain_fn = bridge_infra.get_era_medium_constraint()
        assert isinstance(floor_map, dict), "MEDIUM_DECADE_FLOOR muss ein Dict sein"
        assert callable(constrain_fn), "constrain_era_to_medium muss callable sein"


# ---------------------------------------------------------------------------
# 6. ML-Memory-Budget-Singleton (§2.37)
# ---------------------------------------------------------------------------


class TestMlMemoryBudgetSingleton:
    """get_ml_memory_budget() gibt MlMemoryBudget-Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        budget = bridge_infra.get_ml_memory_budget()
        assert budget is not None

    def test_has_try_allocate(self, bridge_infra):
        budget = bridge_infra.get_ml_memory_budget()
        assert hasattr(budget, "try_allocate"), "MlMemoryBudget fehlt .try_allocate()"

    def test_has_release(self, bridge_infra):
        budget = bridge_infra.get_ml_memory_budget()
        assert hasattr(budget, "release"), "MlMemoryBudget fehlt .release()"


# ---------------------------------------------------------------------------
# 7. ModelDownloader-Singleton (§9.x / §13.x)
# ---------------------------------------------------------------------------


class TestModelDownloader:
    """get_model_downloader() gibt ModelDownloader-Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        downloader = bridge_infra.get_model_downloader()
        assert downloader is not None


# ---------------------------------------------------------------------------
# 8. §G90 PresenceEmbedding + EraAuthenticPerceptualCompletion
# ---------------------------------------------------------------------------


class TestPresenceEmbedding:
    """get_presence_embedding() gibt Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        pe = bridge_infra.get_presence_embedding()
        assert pe is not None


class TestEraCompletion:
    """get_era_completion() gibt Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        ec = bridge_infra.get_era_completion()
        assert ec is not None


# ---------------------------------------------------------------------------
# 9. RollbackSanityGuard (§G92)
# ---------------------------------------------------------------------------


class TestRollbackSanityGuard:
    """get_rollback_sanity_guard() gibt Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        rsg = bridge_infra.get_rollback_sanity_guard()
        assert rsg is not None


# ---------------------------------------------------------------------------
# 10. PreviewMode (§ROADMAP-5)
# ---------------------------------------------------------------------------


class TestPreviewMode:
    """get_preview_mode() gibt Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        pm = bridge_infra.get_preview_mode()
        assert pm is not None


# ---------------------------------------------------------------------------
# 11. ArtistFingerprintStore (§13.11)
# ---------------------------------------------------------------------------


class TestArtistFingerprintStore:
    """get_artist_fingerprint_store() gibt Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        afs = bridge_infra.get_artist_fingerprint_store()
        assert afs is not None


# ---------------------------------------------------------------------------
# 12. PluginRegistry (§V4 Bridge-Bypass-Verbot)
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    """get_plugin_registry() gibt Instanz zurück."""

    def test_returns_non_none(self, bridge_infra):
        pr = bridge_infra.get_plugin_registry()
        assert pr is not None
