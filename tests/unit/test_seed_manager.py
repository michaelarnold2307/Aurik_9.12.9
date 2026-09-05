"""§G5 SeedManager — Unit-Tests für deterministischen Seed-Verwaltung.

Testet Session-Initialisierung, phasenspezifische Seeds, Determinismus (gleicher
Master-Seed → gleiche Phase-Seeds), Reset und Edge-Cases.

Spec: .github/specs/01_musical_goals.md §G5 / AGENTS.md §G5
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestSeedManager:
    """§G5 Seed-Manager funktioniert korrekt."""

    def test_singleton_factory(self):
        from backend.core.seed_manager import get_seed_manager

        m1 = get_seed_manager()
        m2 = get_seed_manager()
        assert m1 is m2, "Singleton-Factory sollte dieselbe Instanz zurückgeben"

    def test_start_session_returns_master_seed(self):
        """start_session() sollte Master-Seed zurückgeben."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        seed = manager.start_session(song_id="test_track_001")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31
        assert manager.master_seed == seed

    def test_master_seed_deterministic_from_song_id(self):
        """Gleiche song_id → gleicher Master-Seed (deterministisch)."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        seed1 = manager.start_session(song_id="track_A")
        manager.reset()
        seed2 = manager.start_session(song_id="track_A")
        assert seed1 == seed2, "Gleiche song_id sollte gleichen Master-Seed produzieren"

    def test_phase_seed_deterministic(self):
        """Gleicher Master-Seed + Phase-ID → gleicher Phase-Seed."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.start_session(song_id="track_A")

        seed1 = manager.get_phase_seed("phase_03_denoise")
        seed2 = manager.get_phase_seed("phase_03_denoise")
        assert seed1 == seed2, "Gleiche Phase-ID sollte gleichen Seed produzieren"

    def test_different_phases_different_seeds(self):
        """Verschiedene Phasen sollten verschiedene Seeds haben."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.start_session(song_id="track_A")

        seed1 = manager.get_phase_seed("phase_03_denoise")
        seed2 = manager.get_phase_seed("phase_07_harmonic_exc")
        assert seed1 != seed2, "Verschiedene Phasen sollten verschiedene Seeds haben"

    def test_phase_seed_bounds(self):
        """Phase-Seed liegt in [0, 2^31)."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.start_session(song_id="track_A")

        for phase_id in ["phase_03_denoise", "phase_07_harmonic_exc", "phase_12_vocal"]:
            seed = manager.get_phase_seed(phase_id)
            assert 0 <= seed < 2**31, f"Seed für {phase_id}={seed}"

    def test_reset_clears_session(self):
        """reset() sollte Session zurücksetzen."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.start_session(song_id="track_A")
        assert manager.master_seed is not None

        manager.reset()
        assert manager.master_seed is None
        assert manager.session_id is None

    def test_explicit_master_seed(self):
        """Expliziter Master-Seed sollte verwendet werden."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        explicit_seed = 12345
        result = manager.start_session(master_seed=explicit_seed)
        assert result == (explicit_seed & 0x7FFFFFFF)

    def test_fallback_when_no_session(self):
        """get_phase_seed ohne Session sollte Fallback-Seed produzieren."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.reset()

        seed = manager.get_phase_seed("phase_03_denoise")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31


@pytest.mark.unit
class TestSeedManagerEdgeCases:
    """Edge-Cases für Seed-Manager."""

    def setup_method(self) -> None:
        """Reset Singleton vor jedem Test."""
        from backend.core.seed_manager import (
            _manager_instance,
            get_seed_manager,
        )
        # pylint: disable=global-statement
        global _manager_instance  # noqa: F841
        import backend.core.seed_manager as sm_module
        sm_module._manager_instance = None

    def test_none_song_id(self):
        """song_id=None sollte Session starten können."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        seed = manager.start_session(song_id=None)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31

    def test_empty_phase_id(self):
        """Leere Phase-ID sollte Seed produzieren."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.start_session(song_id="track_A")
        seed = manager.get_phase_seed("")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31

    def test_repeated_calls_same_seed(self):
        """Wiederholte Aufrufe für gleiche Phase sollten gleichen Seed zurückgeben."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.start_session(song_id="track_A")

        seeds = [manager.get_phase_seed("phase_03_denoise") for _ in range(10)]
        assert all(s == seeds[0] for s in seeds), "Wiederholte Aufrufe sollten gleichen Seed produzieren"

    def test_session_id_tracking(self):
        """session_id sollte korrekt gesetzt werden."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        manager.start_session(song_id="my_song_123")
        assert manager.session_id == "my_song_123"

    def test_master_seed_property(self):
        """master_seed-Eigenschaft sollte korrekt funktionieren."""
        from backend.core.seed_manager import get_seed_manager

        manager = get_seed_manager()
        assert manager.master_seed is None  # Vor Session-Start
        manager.start_session(song_id="test")
        assert manager.master_seed is not None
