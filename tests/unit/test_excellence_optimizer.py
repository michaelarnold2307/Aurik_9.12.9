from __future__ import annotations

import pytest

"""Unit-Tests für core/excellence_optimizer.py — ExcellenceOptimizer.

Spec §2.x: Kontext-bewusstes Post-Processing (Spectral Continuity,
Micro-Dynamic Re-injection, Harmonic Reinforcement, OLA Smoothing).
≥ 14 Tests: Shape, NaN, Mono/Stereo, Material-Profile, Context-Analyse.
"""


import math

import numpy as np

np.random.seed(42)

from backend.core.excellence_optimizer import (
    MATERIAL_PROFILES,
    ExcellenceContext,
    ExcellenceOptimizer,
    ExcellenceResult,
    MaterialProfile,
    analyze_context,
    map_panns_to_profile,
    optimize_for_excellence,
)

SR = 48000


def _sine(freq: float = 440.0, secs: float = 1.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(secs: float = 1.0) -> np.ndarray:
    mono = _sine(secs=secs)
    return np.stack([mono, mono * 0.9])


def _noise(secs: float = 1.0, amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(SR * secs)) * amp).astype(np.float32)


# ---------------------------------------------------------------------------
# Klasse 1: Import und Material-Profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExcellenceOptimizerImport:
    def test_01_class_importable(self):
        assert ExcellenceOptimizer is not None

    def test_02_result_class_importable(self):
        assert ExcellenceResult is not None

    def test_03_material_profiles_present(self):
        for key in ("auto", "vinyl", "tape", "shellac", "broadcast"):
            assert key in MATERIAL_PROFILES, f"Profil '{key}' fehlt"

    def test_04_material_profile_fields(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(MaterialProfile)}
        required = {"name", "flux_smoothing_max", "target_cv_min", "modulation_strength", "harm_boost_db", "ola_ms"}
        assert required.issubset(fields)

    def test_05_instantiate_default(self):
        opt = ExcellenceOptimizer(sample_rate=SR)
        assert opt is not None

    def test_06_instantiate_with_material(self):
        opt = ExcellenceOptimizer(sample_rate=SR, material="vinyl")
        assert opt.material == "vinyl"


# ---------------------------------------------------------------------------
# Klasse 2: analyze_context
# ---------------------------------------------------------------------------


class TestAnalyzeContext:
    def test_07_context_returns_dataclass(self):
        ctx = analyze_context(_sine(), SR)
        assert isinstance(ctx, ExcellenceContext)

    def test_08_context_fields_finite(self):
        ctx = analyze_context(_sine(), SR)
        assert math.isfinite(ctx.rms_db)
        assert math.isfinite(ctx.snr_estimate_db)
        assert 0.0 <= ctx.harmonicity <= 1.0
        assert 0.0 <= ctx.transient_density <= 1.0

    def test_09_context_stereo(self):
        ctx = analyze_context(_stereo(), SR)
        assert ctx.is_stereo is True

    def test_10_context_mono_flag_false(self):
        ctx = analyze_context(_sine(), SR)
        assert ctx.is_stereo is False

    def test_11_context_silence_low_rms(self):
        silence = np.zeros(SR, dtype=np.float32)
        ctx_silence = analyze_context(silence, SR)
        ctx_sine = analyze_context(_sine(), SR)
        assert ctx_silence.rms_db < ctx_sine.rms_db


# ---------------------------------------------------------------------------
# Klasse 3: optimize() — Shape und Wertebereiche
# ---------------------------------------------------------------------------


class TestExcellenceOptimizerOptimize:
    def setup_method(self):
        self.opt = ExcellenceOptimizer(sample_rate=SR)

    def test_12_mono_shape_preserved(self):
        audio = _sine()
        out, result = self.opt.optimize(audio)
        assert out.shape == audio.shape

    def test_13_stereo_shape_preserved(self):
        audio = _stereo()
        out, result = self.opt.optimize(audio)
        assert out.shape == audio.shape

    def test_14_output_clipped_to_minus1_plus1(self):
        audio = _sine()
        out, _ = self.opt.optimize(audio)
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)

    def test_15_output_is_finite(self):
        audio = _sine()
        out, _ = self.opt.optimize(audio)
        assert np.all(np.isfinite(out))

    def test_16_result_is_excellence_result(self):
        _, result = self.opt.optimize(_sine())
        assert isinstance(result, ExcellenceResult)

    def test_17_empty_audio_passthrough(self):
        empty = np.array([], dtype=np.float32)
        out, _ = self.opt.optimize(empty)
        assert out.size == 0


# ---------------------------------------------------------------------------
# Klasse 4: NaN/Inf-Guard
# ---------------------------------------------------------------------------


class TestExcellenceOptimizerNaNGuard:
    def setup_method(self):
        self.opt = ExcellenceOptimizer(sample_rate=SR)

    def test_18_nan_input_handled(self):
        nan_audio = np.full(SR, float("nan"), dtype=np.float32)
        out, _ = self.opt.optimize(nan_audio)
        assert np.all(np.isfinite(out))

    def test_19_inf_input_handled(self):
        inf_audio = np.full(SR, float("inf"), dtype=np.float32)
        out, _ = self.opt.optimize(inf_audio)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Klasse 5: Material-spezifisches Verhalten
# ---------------------------------------------------------------------------


class TestExcellenceOptimizerMaterials:
    def test_20_vinyl_material_runs(self):
        opt = ExcellenceOptimizer(sample_rate=SR, material="vinyl")
        out, _ = opt.optimize(_sine())
        assert out.shape == (SR,)
        assert np.all(np.isfinite(out))

    def test_21_tape_material_runs(self):
        opt = ExcellenceOptimizer(sample_rate=SR, material="tape")
        out, _ = opt.optimize(_sine())
        assert np.all(np.isfinite(out))

    def test_21b_reel_tape_alias_uses_tape_profile(self):
        """ExcellenceOptimizer: 'reel_tape' must resolve to 'tape' profile without warning."""
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        opt = ExcellenceOptimizer(sample_rate=SR, material="reel_tape")
        # Should not fall back to 'auto' — profile must equal 'tape'
        assert opt._profile == MATERIAL_PROFILES["tape"]
        # Optimize must complete without error
        out, _ = opt.optimize(_sine())
        assert np.all(np.isfinite(out))

    def test_22_shellac_material_runs(self):
        opt = ExcellenceOptimizer(sample_rate=SR, material="shellac")
        out, _ = opt.optimize(_sine())
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Klasse 6: Convenience-Funktion und PANNs-Mapping
# ---------------------------------------------------------------------------


class TestExcellenceConvenienceAndMapping:
    def test_23_optimize_for_excellence_returns_audio(self):
        audio = _sine()
        out, result = optimize_for_excellence(audio, SR)
        assert isinstance(out, np.ndarray)
        assert out.shape == audio.shape
        assert np.all(np.isfinite(out))

    def test_24_map_panns_vinyl(self):
        profile = map_panns_to_profile({"Vinyl": 0.9, "Radio": 0.1})
        assert profile == "vinyl"

    def test_25_map_panns_low_confidence_returns_auto(self):
        profile = map_panns_to_profile({"Vinyl": 0.1})
        assert profile == "auto"

    def test_26_map_panns_empty_returns_auto(self):
        profile = map_panns_to_profile({})
        assert profile == "auto"


# ---------------------------------------------------------------------------
# Klasse 7: OLA Edge Crossfade Energie-Guard (§v10.x, Befund 2026-08-22)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOlaEdgeCrossfadeGuard:
    """Der Fade multiplizierte Sample 0 mit 0.0 — bei Musik ohne Lead-in-
    Silence wurde der erste Onset bis zu 100 % gedämpft. Der Energie-Guard
    darf Fades nur in stille Randbereiche anwenden; aktive Kanten bleiben
    bit-identisch. Deterministisch: reine RMS-Schwelle, kein Zufall."""

    def _click_train(self, sr: int = SR, secs: float = 2.0) -> np.ndarray:
        n = int(sr * secs)
        audio = np.zeros(n, dtype=np.float64)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.98):
            idx = min(int(t * sr), n - 1)
            decay = np.exp(-np.arange(min(480, n - idx)) * 0.02)
            audio[idx : idx + len(decay)] += 0.9 * decay
        return np.clip(audio, -1.0, 1.0)

    def test_27_active_edges_untouched(self):
        """Click-Train mit Onset bei t=0 → kein Fade (n_xfades=0, bit-identisch)."""
        from backend.core.excellence_optimizer import _ola_crossfade_edges

        audio = self._click_train()
        out, n_xfades = _ola_crossfade_edges(audio, SR)
        assert n_xfades == 0
        assert np.array_equal(out, audio)

    def test_28_silent_edges_faded(self):
        """Stille Ränder → 2 Fades, Signalmitte unverändert."""
        from backend.core.excellence_optimizer import _ola_crossfade_edges

        n = int(SR * 2)
        audio = np.zeros(n, dtype=np.float64)
        # Aktiver Teil nur in der Mitte (1.0s–1.5s)
        mid = np.sin(2 * np.pi * 440.0 * np.arange(int(0.5 * SR)) / SR) * 0.5
        audio[int(1.0 * SR) : int(1.5 * SR)] = mid
        out, n_xfades = _ola_crossfade_edges(audio, SR)
        assert n_xfades == 2
        assert out[0] == 0.0  # fade-in startet bei 0
        assert out[-1] == 0.0  # fade-out endet bei 0
        # Mitte bit-identisch (außerhalb der 20-ms-Fades)
        xfade_samples = int(0.020 * SR)
        assert np.array_equal(out[xfade_samples:-xfade_samples], audio[xfade_samples:-xfade_samples])

    def test_29_layout_determinism_channels_first_vs_samples_first(self):
        """(2, N) und (N, 2) liefern transponiert identische Ergebnisse."""
        from backend.core.excellence_optimizer import _ola_crossfade_edges

        n = int(SR * 2)
        mono = np.zeros(n, dtype=np.float64)
        mid = np.sin(2 * np.pi * 440.0 * np.arange(int(0.5 * SR)) / SR) * 0.5
        mono[int(1.0 * SR) : int(1.5 * SR)] = mid
        stereo_cf = np.stack([mono, mono], axis=0)  # (2, N)
        stereo_sf = np.stack([mono, mono], axis=1)  # (N, 2)
        out_cf, n_cf = _ola_crossfade_edges(stereo_cf, SR)
        out_sf, n_sf = _ola_crossfade_edges(stereo_sf, SR)
        assert n_cf == n_sf == 2
        assert np.allclose(out_cf, out_sf.T)

    def test_30_short_audio_untouched(self):
        """Kürzer als 4×Fade-Länge → Passthrough (n_xfades=0)."""
        from backend.core.excellence_optimizer import _ola_crossfade_edges

        audio = self._click_train(secs=0.05)  # 50 ms < 80 ms
        out, n_xfades = _ola_crossfade_edges(audio, SR)
        assert n_xfades == 0
        assert np.array_equal(out, audio)
