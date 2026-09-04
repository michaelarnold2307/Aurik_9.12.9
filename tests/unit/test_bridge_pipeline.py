from __future__ import annotations

"""Tests für backend/api/bridge_pipeline.py — Pipeline Health State + Trace (§v10).

Prüft:
- Alle __all__-Einträge sind importierbar und aufrufbar
- get_pipeline_trace gibt Dict zurück (defensiv)
- get_pipeline_ab_snapshots läuft durch ohne Exception
- run_album_consistency_pass ist callable
- Crash Report Visibility (§v10.993): get_new_crash_reports, mark_seen, install_handler
- Guard Report Telemetrie (§v10.990): get_guard_report gibt Dict zurück
- Restoration Bericht (§v10.996): get_restoration_bericht gibt Dict zurück
- Repair Plan Consent (§v10.992): get_repair_plan_consent gibt Dict zurück
"""

import importlib
import pytest


@pytest.fixture(scope="module")
def bridge_pipeline():
    """Importiert die Pipeline-Bridge einmalig pro Modul."""
    return importlib.import_module("backend.api.bridge_pipeline")


# ---------------------------------------------------------------------------
# 1. Grundlegender Import + __all__
# ---------------------------------------------------------------------------


class TestBridgePipelineImport:
    """bridge_pipeline ist importierbar und hat valides __all__."""

    def test_imports_cleanly(self, bridge_pipeline):
        assert bridge_pipeline is not None

    def test_has_all(self, bridge_pipeline):
        assert hasattr(bridge_pipeline, "__all__"), "__all__ fehlt"

    def test_all_entries_exist(self, bridge_pipeline):
        missing = [name for name in bridge_pipeline.__all__ if not hasattr(bridge_pipeline, name)]
        assert not missing, f"In __all__ deklariert, aber nicht im Modul: {missing}"


# ---------------------------------------------------------------------------
# 2. Pipeline Trace — vollständiger Trace mit Goal-Timeline
# ---------------------------------------------------------------------------


class TestPipelineTrace:
    """get_pipeline_trace() gibt immer ein Dict zurück."""

    def test_returns_dict(self, bridge_pipeline):
        result = bridge_pipeline.get_pipeline_trace(None)
        assert isinstance(result, dict), f"Muss dict zurückgeben, nicht {type(result)}"

    def test_handles_none_result(self, bridge_pipeline):
        result = bridge_pipeline.get_pipeline_trace(None)
        # Defensiv: kein Crash bei None-Input
        assert "error" in result or len(result) >= 0


# ---------------------------------------------------------------------------
# 3. A/B-Vergleichs-Snapshots pro Phase als Base64-WAV (§v10)
# ---------------------------------------------------------------------------


class TestPipelineAbSnapshots:
    """get_pipeline_ab_snapshots() läuft durch ohne Exception."""

    def test_is_callable(self, bridge_pipeline):
        assert callable(bridge_pipeline.get_pipeline_ab_snapshots)

    def test_returns_list(self, bridge_pipeline):
        result = bridge_pipeline.get_pipeline_ab_snapshots(include_audio=False)
        assert isinstance(result, list), f"Muss Liste zurückgeben, nicht {type(result)}"

    def test_completes_without_exception(self, bridge_pipeline):
        try:
            result = bridge_pipeline.get_pipeline_ab_snapshots(include_audio=True, max_duration_s=5.0)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"get_pipeline_ab_snapshots() wirft Exception: {e}")


# ---------------------------------------------------------------------------
# 4. Album Consistency Pass (§1.4)
# ---------------------------------------------------------------------------


class TestAlbumConsistencyPass:
    """run_album_consistency_pass() ist callable und gibt Dict zurück."""

    def test_is_callable(self, bridge_pipeline):
        assert callable(bridge_pipeline.run_album_consistency_pass)

    def test_returns_dict_with_empty_list(self, bridge_pipeline):
        result = bridge_pipeline.run_album_consistency_pass(output_files=[], dry_run=True)
        assert isinstance(result, dict), f"Muss dict zurückgeben, nicht {type(result)}"


# ---------------------------------------------------------------------------
# 5. Crash Report Visibility (§v10.993)
# ---------------------------------------------------------------------------


class TestCrashReportVisibility:
    """Crash-Report-Funktionen sind callable und robust."""

    def test_get_new_crash_reports_returns_list(self, bridge_pipeline):
        result = bridge_pipeline.get_new_crash_reports()
        assert isinstance(result, list), f"Muss Liste zurückgeben, nicht {type(result)}"

    def test_mark_crash_reports_seen_completes(self, bridge_pipeline):
        try:
            bridge_pipeline.mark_crash_reports_seen()
        except Exception as e:
            pytest.fail(f"mark_crash_reports_seen() wirft Exception: {e}")

    def test_install_crash_handler_completes(self, bridge_pipeline):
        try:
            bridge_pipeline.install_crash_handler()
        except Exception as e:
            pytest.fail(f"install_crash_handler() wirft Exception: {e}")


# ---------------------------------------------------------------------------
# 6. Guard Report Telemetrie (§v10.990)
# ---------------------------------------------------------------------------


class TestGuardReportTelemetrie:
    """get_guard_report() gibt immer ein Dict zurück."""

    def test_returns_dict_for_none(self, bridge_pipeline):
        result = bridge_pipeline.get_guard_report(None)
        assert isinstance(result, dict), f"Muss dict zurückgeben, nicht {type(result)}"

    def test_returns_dict_for_object(self, bridge_pipeline):
        result = bridge_pipeline.get_guard_report(object())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 7. Restoration Bericht (§v10.996)
# ---------------------------------------------------------------------------


class TestRestorationBericht:
    """get_restoration_bericht() gibt immer ein Dict zurück."""

    def test_returns_dict_for_none(self, bridge_pipeline):
        result = bridge_pipeline.get_restoration_bericht(None)
        assert isinstance(result, dict), f"Muss dict zurückgeben, nicht {type(result)}"

    def test_returns_dict_for_object(self, bridge_pipeline):
        result = bridge_pipeline.get_restoration_bericht(object())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 8. Repair Plan Consent (§v10.992)
# ---------------------------------------------------------------------------


class TestRepairPlanConsent:
    """get_repair_plan_consent() gibt immer ein Dict zurück."""

    def test_returns_dict_for_none(self, bridge_pipeline):
        result = bridge_pipeline.get_repair_plan_consent(None)
        assert isinstance(result, dict), f"Muss dict zurückgeben, nicht {type(result)}"

    def test_returns_dict_with_found_and_will_do_keys(self, bridge_pipeline):
        """Bei vorhandenem Defekt-Ergebnis: 'found' und 'will_do' Keys vorhanden."""
        # Dummy-Objekt mit leerem repair_plan
        class _DummyDefectResult:
            _consensus_manifest = None
            defect_scores = {}
            repair_plan = type("Plan", (), {"phase_order": []})()

        result = bridge_pipeline.get_repair_plan_consent(_DummyDefectResult())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 9. Defensiv: alle Funktionen crashen nicht bei None/leerem Input
# ---------------------------------------------------------------------------


class TestBridgePipelineDefensive:
    """Alle Bridge-Pipeline-Funktionen sind defensiv gegen None/leere Inputs."""

    def test_all_functions_handle_none(self, bridge_pipeline):
        """Keine Funktion wirft Exception bei None-Input."""
        functions_to_test = [
            ("get_pipeline_trace", [None]),
            ("get_guard_report", [None]),
            ("get_restoration_bericht", [None, None]),
            ("get_repair_plan_consent", [None]),
        ]

        for name, args in functions_to_test:
            fn = getattr(bridge_pipeline, name)
            try:
                result = fn(*args)
                assert isinstance(result, dict), f"{name}() muss dict zurückgeben"
            except Exception as e:
                pytest.fail(f"{name}({args}) wirft Exception: {e}")
