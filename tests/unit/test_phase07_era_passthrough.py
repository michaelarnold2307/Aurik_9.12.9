"""§v10.730 (2026-09-09): era_result muss die Phasen erreichen.

Befund: §ERA_HARMONIC Verarbeitungsschritt_07 loggte era=None obwohl
material=vinyl klassifiziert war — der Pipeline reichte nur decade/era_vocal_profile
an die Phasen durch, das vollständige EraResult (mit .decade) ging verloren →
Phase 07 fiel auf material-adaptive H2-Defaults zurück statt era-authentischer
H2-Targets. Fix: restore() legt era_result in den Phasen-Kontext, alle drei
Phase-Aufrufpfade (PMGG-Primär, Fallback, Direkt, Parallel) reichen es durch.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_SR = 48000


def _vinyl_like(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(_SR * seconds), dtype=np.float64) / _SR
    # Grundton + starker H2 (Vinyl-Typisch) für den §ERA_HARMONIC-Steering-Block
    # Grundton dominant, H2 SCHWACH (H2/H1 ≈ 0.1 < 0.15): der Sättigungs-Guard
    # (§v10.111/§v10.114) darf die Strength NICHT drosseln, sonst schaltet die
    # Phase vor dem §ERA_HARMONIC-Block auf Passthrough.
    sig = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.03 * np.sin(2 * np.pi * 440.0 * t)
    sig += 0.02 * np.sin(2 * np.pi * 660.0 * t)
    rng = np.random.default_rng(11)
    sig += 0.005 * rng.standard_normal(len(t))
    return sig.astype(np.float32)


def test_pipeline_passes_era_result_to_phases_all_paths() -> None:
    """Quellen-Vertrag: alle Phase-Aufrufpfade reichen era_result durch."""
    _src = (Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text(
        encoding="utf-8"
    )
    assert 'self._restoration_context["era_result"] = _era_result' in _src
    assert '"era_result": _ctx_phase_vocal.get("era_result"),' in _src  # PMGG-Primär
    assert "era_result=_ctx_phase_vocal.get(\"era_result\")," in _src  # Fallback + Direkt
    assert "era_result=(getattr(self, \"_restoration_context\", {}) or {}).get(\"era_result\")," in _src  # Parallel


def test_phase07_uses_era_decade_from_era_result() -> None:
    """Phase 07 übergibt era_decade aus kwargs['era_result'] an den Tonal-Profiler."""
    from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

    _fake_era = SimpleNamespace(
        decade=1968,
        confidence=0.8,
        material_prior="vinyl",
        noise_profile=np.zeros(24, dtype=np.float32),
    )

    class _NullCurve:
        def apply_snr_adaptive_ceiling(self, audio, restored, sr):
            return restored

        def apply_target_steering(self, audio, restored, sr, *a, **k):
            return restored

    _captured: dict = {}

    def _fake_get_curve(**kwargs):
        _captured.update(kwargs)
        return _NullCurve()

    _prof = SimpleNamespace()
    _prof.get_curve = _fake_get_curve

    with patch("backend.core.tonal_reference_profile.get_tonal_reference_profiler", return_value=_prof):
        phase = HarmonicRestorationPhase()
        phase.process(_vinyl_like(), _SR, material_type="vinyl", era_result=_fake_era, strength=0.5)

    assert _captured.get("era_decade") == 1968, (
        f"era_decade aus era_result wurde nicht durchgereicht: {_captured}"
    )


def test_phase07_era_none_keeps_material_adaptive_default() -> None:
    """Ohne era_result bleibt era_decade=None (material-adaptiver Fallback, kein Crash)."""
    from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

    class _NullCurve:
        def apply_snr_adaptive_ceiling(self, audio, restored, sr):
            return restored

        def apply_target_steering(self, audio, restored, sr, *a, **k):
            return restored

    _captured: dict = {}

    def _fake_get_curve(**kwargs):
        _captured.update(kwargs)
        return _NullCurve()

    _prof = SimpleNamespace()
    _prof.get_curve = _fake_get_curve

    with patch("backend.core.tonal_reference_profile.get_tonal_reference_profiler", return_value=_prof):
        phase = HarmonicRestorationPhase()
        phase.process(_vinyl_like(), _SR, material_type="vinyl", strength=0.5)

    assert _captured.get("era_decade") is None, f"erwartet era_decade=None, war: {_captured.get('era_decade')}"
