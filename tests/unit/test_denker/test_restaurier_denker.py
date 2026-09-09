from __future__ import annotations

import pytest

"""tests/unit/test_denker/test_restaurier_denker.py

Tests für RestaurierDenker — Hauptrestaurierung via UnifiedRestorerV3.
"""


import math
from unittest.mock import patch

import numpy as np

SR = 48_000
np.random.seed(42)


def _sine(dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


# ─── RestaurierErgebnis ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestRestaurierErgebnisFields:
    def _make(self):
        from denker.restaurier_denker import RestaurierErgebnis

        audio = _sine()
        return RestaurierErgebnis(
            audio=audio,
            quality_delta=0.12,
            phases_applied=["phase_03_denoise"],
            material="tape",
            processing_note="Rauschen entfernt",
        )

    def test_01_audio_ndarray(self):
        e = self._make()
        assert isinstance(e.audio, np.ndarray)

    def test_02_quality_delta_finite(self):
        e = self._make()
        assert math.isfinite(e.quality_delta)

    def test_03_phases_applied_list(self):
        e = self._make()
        assert isinstance(e.phases_applied, list)

    def test_04_material_str(self):
        e = self._make()
        assert isinstance(e.material, str)

    def test_05_processing_note_str(self):
        e = self._make()
        assert isinstance(e.processing_note, str)

    def test_06_audio_finite(self):
        e = self._make()
        assert np.all(np.isfinite(e.audio))


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestRestaurierDenkerSingleton:
    def test_07_returns_instance(self):
        from denker.restaurier_denker import RestaurierDenker, get_restaurier_denker

        assert isinstance(get_restaurier_denker(), RestaurierDenker)

    def test_08_singleton_identity(self):
        from denker.restaurier_denker import get_restaurier_denker

        assert get_restaurier_denker() is get_restaurier_denker()

    def test_09_thread_safe(self):
        import concurrent.futures

        from denker.restaurier_denker import get_restaurier_denker

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            insts = list(ex.map(lambda _: get_restaurier_denker(), range(12)))
        assert all(i is insts[0] for i in insts)


# ─── restauriere() — Graceful-Degradation-Tests ──────────────────────────────


class TestRestaurierDenkerMock:
    """Testet restauriere() durch Patchen der Methoden auf Klassen-Ebene.

    ``_build_restorer`` wird via setup_method auf ``return_value=None`` gesetzt,
    damit keine echten Heavy-Imports stattfinden. Der ARE-Legacy-Pfad wurde
    2026-09-08 entfernt (Parallelversion-Verbot) — es gibt nur noch den
    direkten UV3-Pfad. Alle Tests prüfen das garantierte
    Graceful-Degradation-Verhalten: ``_fallback()`` liefert stets ein valides
    ``RestaurierErgebnis``.
    """

    def setup_method(self):
        from denker.restaurier_denker import RestaurierDenker

        self._p_restorer = patch.object(RestaurierDenker, "_build_restorer", return_value=None)
        self._p_restorer.start()

    def teardown_method(self):
        self._p_restorer.stop()

    # ── 10–12: Rückgabe-Typ und Audio-Integrität ──────────────────────────

    def test_10_returns_restaurier_ergebnis(self):
        from denker.restaurier_denker import RestaurierDenker, RestaurierErgebnis

        result = RestaurierDenker().restauriere(_sine(), SR)
        assert isinstance(result, RestaurierErgebnis)

    def test_11_audio_is_ndarray(self):
        from denker.restaurier_denker import RestaurierDenker

        result = RestaurierDenker().restauriere(_sine(), SR)
        assert isinstance(result.audio, np.ndarray)

    def test_12_audio_finite(self):
        from denker.restaurier_denker import RestaurierDenker

        result = RestaurierDenker().restauriere(_sine(), SR)
        assert np.all(np.isfinite(result.audio))

    # ── 13–15: Fallback-Felder sind valide ───────────────────────────────

    def test_13_audio_clipped_to_one(self):
        """_fallback() clippt das Audio zwingend auf [-1, 1]."""
        from denker.restaurier_denker import RestaurierDenker

        loud = _sine() * 5.0  # über-laut; muss auf ±1 geclippt werden
        result = RestaurierDenker().restauriere(loud, SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-5

    def test_14_phases_applied_is_list(self):
        from denker.restaurier_denker import RestaurierDenker

        result = RestaurierDenker().restauriere(_sine(), SR)
        assert isinstance(result.phases_applied, list)

    def test_15_quality_delta_finite(self):
        from denker.restaurier_denker import RestaurierDenker

        result = RestaurierDenker().restauriere(_sine(), SR)
        assert math.isfinite(result.quality_delta)

    # ── 16–18: Parameter-Durchleitung und Randfälle ───────────────────────

    def test_16_material_forwarded(self):
        """material-Parameter landet im Ergebnis-Feld."""
        from denker.restaurier_denker import RestaurierDenker

        result = RestaurierDenker().restauriere(_sine(), SR, material="vinyl")
        assert result.material == "vinyl"

    def test_17_quality_estimate_finite(self):
        from denker.restaurier_denker import RestaurierDenker

        result = RestaurierDenker().restauriere(_sine(), SR)
        assert math.isfinite(result.quality_estimate)

    def test_18_short_audio_returns_ergebnis(self):
        """Kurzes Audio (512 Samples) darf keinen Absturz erzeugen."""
        from denker.restaurier_denker import RestaurierDenker, RestaurierErgebnis

        result = RestaurierDenker().restauriere(np.zeros(512, dtype=np.float32), SR)
        assert isinstance(result, RestaurierErgebnis)

    # ── 19–20: Weitere Invarianten ────────────────────────────────────────

    def test_19_warnings_is_list(self):
        """_fallback() trägt in warnings ein warum kein Restorer verfügbar war."""
        from denker.restaurier_denker import RestaurierDenker

        result = RestaurierDenker().restauriere(_sine(), SR)
        assert isinstance(result.warnings, list)
        assert len(result.warnings) >= 1

    def test_20_stereo_input_returns_ergebnis(self):
        """Stereo-Input (2 × N) darf keinen Absturz erzeugen."""
        from denker.restaurier_denker import RestaurierDenker, RestaurierErgebnis

        audio = np.stack([_sine(), _sine()], axis=0)
        result = RestaurierDenker().restauriere(audio, SR)
        assert isinstance(result, RestaurierErgebnis)


class TestRestaurierDenkerCheckpointResume:
    def test_21_uses_restore_from_checkpoint_when_provided(self):
        from denker.restaurier_denker import RestaurierDenker

        class _Material:
            value = "vinyl"

        class _Raw:
            def __init__(self):
                self.audio = _sine(0.5)
                self.rt_factor = 1.2
                self.quality_estimate = 0.9
                self.phases_executed = ["phase_01_click_removal"]
                self.phases_skipped = []
                self.musical_goals = {"natuerlichkeit": 0.95}
                self.warnings = []
                self.confidence = 0.92
                self.winning_variant = None
                self.rollback_triggered = False
                self.total_time_seconds = 0.5
                self.material_type = _Material()

        class _FakeRestorer:
            def __init__(self):
                self.called = False
                self.received_checkpoint = None

            def restore_from_checkpoint(self, checkpoint, **kwargs):
                self.called = True
                self.received_checkpoint = checkpoint
                return _Raw()

        denker = RestaurierDenker()
        fake_restorer = _FakeRestorer()
        checkpoint_obj = object()

        with patch.object(RestaurierDenker, "_get_restorer", return_value=fake_restorer):
            result = denker.restauriere(_sine(), SR, recovery_checkpoint=checkpoint_obj)

        assert fake_restorer.called is True
        assert fake_restorer.received_checkpoint is checkpoint_obj
        assert isinstance(result.audio, np.ndarray)
        assert result.phases_executed == ["phase_01_click_removal"]


class TestRestaurierDenkerOracleRolloutForwarding:
    def test_22_forwards_phase_strength_oracle_rollout_to_uv3_restore(self):
        from denker.restaurier_denker import RestaurierDenker

        class _Material:
            value = "vinyl"

        class _Raw:
            def __init__(self):
                self.audio = _sine(0.5)
                self.rt_factor = 1.0
                self.quality_estimate = 0.8
                self.phases_executed = []
                self.phases_skipped = []
                self.musical_goals = {}
                self.warnings = []
                self.confidence = 0.9
                self.winning_variant = None
                self.rollback_triggered = False
                self.total_time_seconds = 0.5
                self.material_type = _Material()

        class _FakeRestorer:
            def __init__(self):
                self.restore_kwargs: dict[str, object] = {}

            def restore(self, _audio, **kwargs):
                self.restore_kwargs = dict(kwargs)
                return _Raw()

        denker = RestaurierDenker()
        fake_restorer = _FakeRestorer()

        with patch.object(RestaurierDenker, "_get_restorer", return_value=fake_restorer):
            result = denker.restauriere(
                _sine(),
                SR,
                cached_defect_result=object(),  # direkt in den UV3-Pfad
                phase_strength_oracle_rollout="pilot",
            )

        assert isinstance(result.audio, np.ndarray)
        assert fake_restorer.restore_kwargs.get("phase_strength_oracle_rollout") == "pilot"
