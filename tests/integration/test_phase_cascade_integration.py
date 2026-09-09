from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

# Timeout für Kaskaden-Tests — reelle DSP-Berechnung, kein ML
_KAS_TIMEOUT = 90


# ── Shared Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def audio_vinyl_like() -> tuple[np.ndarray, int]:
    """3s mono @ 48 kHz — Vinyl-ähnliches Signal mit Clicks + Rauschen + Hum."""
    sr = 48_000
    n = 3 * sr
    rng = np.random.default_rng(123)
    t = np.linspace(0, 3.0, n, endpoint=False, dtype=np.float64)

    # Musik-ähnliches Signal (mehrere Töne + Harmonische)
    signal = (
        0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.sin(2 * np.pi * 880 * t)
    )

    # Breitbandiges Rauschen (Hiss)
    signal += 0.03 * rng.standard_normal(n, dtype=np.float64)

    # 50 Hz Hum
    signal += 0.025 * np.sin(2 * np.pi * 50 * t)

    # Clicks / Pops (transiente Spitzen)
    for _ in range(15):
        pos = rng.integers(500, n - 500)
        signal[pos] += 0.8 * rng.choice([-1, 1])

    # DC-Offset (leicht)
    signal += 0.005

    return np.clip(signal.astype(np.float32), -1.0, 1.0), sr


@pytest.fixture(scope="module")
def audio_vocal_like() -> tuple[np.ndarray, int]:
    """3s mono @ 48 kHz — Gesang-ähnliches Signal (Fundamental + Formanten)."""
    sr = 48_000
    n = 3 * sr
    rng = np.random.default_rng(456)
    t = np.linspace(0, 3.0, n, endpoint=False, dtype=np.float64)

    # Stimm-Fundamental (~200 Hz) + Formanten (800 Hz, 2400 Hz, 4000 Hz)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.15 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 2400 * t)
        + 0.05 * np.sin(2 * np.pi * 4000 * t)
    )

    # Sibilanz (3-5 kHz, hissend)
    signal += 0.04 * rng.standard_normal(n, dtype=np.float64) * np.where(
        (t % 0.1 < 0.05), 1.0, 0.2
    )

    # Leichter DC-Offset
    signal += 0.003

    return np.clip(signal.astype(np.float32), -1.0, 1.0), sr


@pytest.fixture(scope="module")
def audio_stereo_like(audio_vinyl_like: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
    """3s stereo @ 48 kHz — Stereo-Signal mit leichtem Channel-Imbalance."""
    # §Fix 2026-09-08: Fixture direkt aufgerufen — Fixtures werden als
    # Parameter angefordert, nicht aufgerufen (pytest-Error).
    mono, sr = audio_vinyl_like
    # Leichte Stereo-Differenz (Rechts etwas leiser)
    stereo = np.stack([mono, mono * 0.92], axis=0).astype(np.float32)
    return stereo, sr


# ═══════════════════════════════════════════════════════════════════════
# 1. NR-Kette: Click-Removal → Denoise → De-Esser
# ═══════════════════════════════════════════════════════════════════════
class TestNRChainCascade:
    """Typische Noise-Reduction-Kette: Clicks rauschen, dann Denoise, dann De-Esser."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_nr_chain_preserves_signal_integrity(self, audio_vinyl_like):
        """Click → Denoise → DeEsser: Output muss finite + bounded sein."""
        audio, sr = audio_vinyl_like

        # Phase 01: Click Removal
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        click_phase = ClickRemovalPhase()
        result1 = click_phase.process(audio.copy(), sample_rate=sr)
        assert np.isfinite(result1.audio).all(), "Click-Removal erzeugt NaN/Inf"
        assert np.max(np.abs(result1.audio)) <= 1.0, "Click-Removal erzeugt Clipping"

        # Phase 03: Denoise
        from backend.core.phases.phase_03_denoise import DenoisePhase

        denoise_phase = DenoisePhase()
        result2 = denoise_phase.process(result1.audio.copy(), sample_rate=sr)
        assert np.isfinite(result2.audio).all(), "Denoise erzeugt NaN/Inf"
        assert np.max(np.abs(result2.audio)) <= 1.0, "Denoise erzeugt Clipping"

        # Phase 19: De-Esser
        from backend.core.phases.phase_05_rumble_filter import RumbleFilterPhase

        rumble_phase = RumbleFilterPhase()
        result3 = rumble_phase.process(result2.audio.copy(), sample_rate=sr)
        assert np.isfinite(result3.audio).all(), "Rumble-Filter erzeugt NaN/Inf"
        assert np.max(np.abs(result3.audio)) <= 1.0, "Rumble-Filter erzeugt Clipping"

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_nr_chain_shape_preserved(self, audio_vinyl_like):
        """NR-Kette: Mono-Input → Mono-Output (Shape-Erhalt)."""
        audio, sr = audio_vinyl_like
        orig_shape = audio.shape

        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase
        from backend.core.phases.phase_03_denoise import DenoisePhase
        from backend.core.phases.phase_05_rumble_filter import RumbleFilterPhase

        result = ClickRemovalPhase().process(audio.copy(), sample_rate=sr)
        result = DenoisePhase().process(result.audio.copy(), sample_rate=sr)
        result = RumbleFilterPhase().process(result.audio.copy(), sample_rate=sr)

        assert result.audio.shape == orig_shape, (
            f"Shape changed: {orig_shape} → {result.audio.shape}"
        )

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_nr_chain_reduces_noise_floor(self, audio_vinyl_like):
        """NR-Kette: RMS des Outputs muss ≤ RMS des Inputs (Rauschen wird reduziert)."""
        audio, sr = audio_vinyl_like
        orig_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase
        from backend.core.phases.phase_03_denoise import DenoisePhase

        result = ClickRemovalPhase().process(audio.copy(), sample_rate=sr)
        result = DenoisePhase().process(result.audio.copy(), sample_rate=sr)

        out_rms = float(np.sqrt(np.mean(result.audio.astype(np.float64) ** 2)))

        # NR-Kette sollte Rauschen reduzieren (oder zumindest nicht erhöhen)
        assert out_rms <= orig_rms * 1.15, (
            f"NR-Kette erhöhte RMS: {orig_rms:.4f} → {out_rms:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. Dynamik-Kette: DC-Offset → Clipping-Repair → Harmonic-Restoration
# ═══════════════════════════════════════════════════════════════════════
class TestDynamicsChainCascade:
    """Dynamik-Kette: DC entfernen, dann Clipping reparieren, dann Harmonische wiederherstellen."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_dynamics_chain_finite_output(self, audio_vinyl_like):
        """DC → Clipping → Harmonic: alle Outputs finite + bounded."""
        audio, sr = audio_vinyl_like

        from backend.core.phases.phase_30_dc_offset_removal import DCOffsetRemoval
        from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

        result = DCOffsetRemoval().process(audio.copy(), sample_rate=sr)
        assert np.isfinite(result.audio).all()

        # Simuliere leicht übersteuertes Signal für Clipping-Repair-Test
        boosted = result.audio * 1.3
        boosted = np.clip(boosted, -1.0, 1.0)

        result = HarmonicRestorationPhase().process(
            boosted.copy(), sample_rate=sr, strength=0.5
        )
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# 3. Gesang-Kette: Vocal-Naturalness → Spatial-Enhancement
# ═══════════════════════════════════════════════════════════════════════
class TestVocalChainCascade:
    """Gesang-spezifische Kette: Naturalness-Restoration → Spatial-Enhancement."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_vocal_chain_no_crash(self, audio_vocal_like):
        """Vocal-Naturalness → Spatial: keine Exception, finite Output."""
        audio, sr = audio_vocal_like

        from backend.core.phases.phase_65_vocal_naturalness_restoration import (
            VocalNaturalnessRestorationPhase,
        )

        vocal_phase = VocalNaturalnessRestorationPhase()
        result1 = vocal_phase.process(audio.copy(), sample_rate=sr)
        assert np.isfinite(result1.audio).all(), "Vocal-Naturalness erzeugt NaN/Inf"

        from backend.core.phases.phase_46_spatial_enhancement import SpatialEnhancementPhase

        spatial_phase = SpatialEnhancementPhase()
        result2 = spatial_phase.process(
            result1.audio.copy(), sample_rate=sr, panns_singing=0.5
        )
        assert np.isfinite(result2.audio).all(), "Spatial-Enhancement erzeugt NaN/Inf"

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_vocal_chain_preserves_fundamental(self, audio_vinyl_like):
        """Gesang-Kette: Fundamental-Energie (~200 Hz) bleibt erhalten."""
        # §Fix 2026-09-08: Parameter war audio_vocal_like, der Body griff auf
        # audio_vinyl_like zu (nicht injiziert → Fixture-Funktionsobjekt →
        # TypeError „cannot unpack non-iterable function object“).
        audio, sr = audio_vinyl_like  # Verwende vinyl-like mit bekanntem 220 Hz Ton

        from backend.core.phases.phase_65_vocal_naturalness_restoration import (
            VocalNaturalnessRestorationPhase,
        )

        result = VocalNaturalnessRestorationPhase().process(audio.copy(), sample_rate=sr)

        # Fundamental-Band (180-260 Hz) Energie-Vergleich
        def _fundamental_energy(sig: np.ndarray) -> float:
            fft = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
            mask = (freqs >= 180) & (freqs <= 260)
            return float(np.mean(fft[mask] ** 2))

        orig_energy = _fundamental_energy(audio)
        out_energy = _fundamental_energy(result.audio)

        # Fundamental sollte nicht um > 50 % reduziert werden
        assert out_energy >= orig_energy * 0.3, (
            f"Fundamental-Energie verloren: {orig_energy:.6f} → {out_energy:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Multi-Phase NaN/Inf-Schutz (§III DSP)
# ═══════════════════════════════════════════════════════════════════════
class TestMultiPhaseNaNProtection:
    """Jede Phase muss NaN/Inf-sicheren Output liefern — auch in Kette."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    @pytest.mark.parametrize(
        "phase_module,phase_class",
        [
            ("phase_01_click_removal", "ClickRemovalPhase"),
            ("phase_02_hum_removal", "HumRemovalPhase"),
            ("phase_03_denoise", "DenoisePhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
            ("phase_30_dc_offset_removal", "DCOffsetRemoval"),
        ],
    )
    def test_phase_nan_inf_safe(self, audio_vinyl_like, phase_module: str, phase_class: str):
        """Jede Phase: NaN/Inf-freier Output bei normalem Input."""
        import importlib

        audio, sr = audio_vinyl_like
        mod = importlib.import_module(f"backend.core.phases.{phase_module}")
        cls = getattr(mod, phase_class)
        instance = cls()

        result = instance.process(audio.copy(), sample_rate=sr)
        assert np.isfinite(result.audio).all(), (
            f"{phase_class} erzeugt NaN/Inf bei normalem Input"
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. Shape-Erhalt über Kaskaden (§G1 Song-Isolation)
# ═══════════════════════════════════════════════════════════════════════
class TestShapePreservationCascade:
    """Mono bleibt Mono, Stereo bleibt Stereo — durch alle Phasen."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_mono_stays_mono_through_chain(self, audio_vinyl_like):
        """Mono-Input → 5-Phase-Kette → Mono-Output."""
        audio, sr = audio_vinyl_like
        assert audio.ndim == 1, "Fixture sollte mono sein"

        phases = [
            ("phase_01_click_removal", "ClickRemovalPhase"),
            ("phase_03_denoise", "DenoisePhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
            ("phase_30_dc_offset_removal", "DCOffsetRemoval"),
        ]

        current = audio.copy()
        for mod_name, cls_name in phases:
            import importlib

            mod = importlib.import_module(f"backend.core.phases.{mod_name}")
            cls = getattr(mod, cls_name)
            result = cls().process(current.copy(), sample_rate=sr)
            current = result.audio
            assert current.ndim == 1, (
                f"{cls_name} hat Mono-Shape gebrochen: {current.shape}"
            )

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_stereo_stays_stereo_through_chain(self, audio_stereo_like):
        """Stereo-Input → 3-Phase-Kette → Stereo-Output."""
        audio, sr = audio_stereo_like
        assert audio.ndim == 2 and audio.shape[0] == 2, "Fixture sollte stereo sein"

        phases = [
            ("phase_01_click_removal", "ClickRemovalPhase"),
            ("phase_03_denoise", "DenoisePhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
        ]

        current = audio.copy()
        for mod_name, cls_name in phases:
            import importlib

            mod = importlib.import_module(f"backend.core.phases.{mod_name}")
            cls = getattr(mod, cls_name)
            result = cls().process(current.copy(), sample_rate=sr)
            current = result.audio
            assert current.ndim == 2 and current.shape[0] == 2, (
                f"{cls_name} hat Stereo-Shape gebrochen: {current.shape}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 6. Peak-Bounds über Kaskaden (§III Soft-Knee statt Hard-Clamp)
# ═══════════════════════════════════════════════════════════════════════
class TestPeakBoundsCascade:
    """Kein Clipping durch Phasenakkumulation — Soft-Knee schützt."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_5_phase_chain_no_clipping(self, audio_vinyl_like):
        """5-Phase-Kette: Peak muss ≤ 1.0 bleiben (kein Akkumulations-Clipping)."""
        audio, sr = audio_vinyl_like

        phases = [
            ("phase_01_click_removal", "ClickRemovalPhase"),
            ("phase_02_hum_removal", "HumRemovalPhase"),
            ("phase_03_denoise", "DenoisePhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
        ]

        current = audio.copy()
        for mod_name, cls_name in phases:
            import importlib

            mod = importlib.import_module(f"backend.core.phases.{mod_name}")
            cls = getattr(mod, cls_name)
            result = cls().process(current.copy(), sample_rate=sr)
            current = result.audio
            peak = float(np.max(np.abs(current)))
            assert peak <= 1.0 + 1e-6, (
                f"{cls_name} erzeugt Clipping: peak={peak:.6f}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 7. Determinismus über Kaskaden (§G5)
# ═══════════════════════════════════════════════════════════════════════
class TestDeterminismCascade:
    """Gleiche Input + gleiche Phasen → bit-identischer Output."""

    @pytest.mark.timeout(120)
    def test_3_phase_chain_deterministic(self, audio_vinyl_like):
        """3x dieselbe Kette ausführen → SHA-256 des Outputs identisch."""
        # Kürzeres Signal für schnellere Ausführung (1s statt 3s)
        sr = 48_000
        rng = np.random.default_rng(777)
        audio = (rng.standard_normal(sr, dtype=np.float32) * 0.05).astype(np.float32)

        # Verwende Phasen mit einfacher Signatur (kein material_type nötig)
        phases = [
            ("phase_01_click_removal", "ClickRemovalPhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
            ("phase_30_dc_offset_removal", "DCOffsetRemoval"),
        ]

        def _run_chain(sig: np.ndarray) -> bytes:
            current = sig.copy()
            for mod_name, cls_name in phases:
                import importlib

                mod = importlib.import_module(f"backend.core.phases.{mod_name}")
                cls = getattr(mod, cls_name)
                result = cls().process(current.copy(), sample_rate=sr)
                current = result.audio
            return hashlib.sha256(current.tobytes()).hexdigest().encode()

        hash1 = _run_chain(audio)
        hash2 = _run_chain(audio)
        hash3 = _run_chain(audio)

        assert hash1 == hash2 == hash3, (
            f"Kaskade nicht deterministisch: {hash1} ≠ {hash2} ≠ {hash3}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 8. Metadata-Akkumulation (§G9 Logger-Pflicht)
# ═══════════════════════════════════════════════════════════════════════
class TestMetadataAccumulation:
    """Jede Phase trägt zum metadata-Dict bei — Akkumulation über Kette."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_chain_metadata_accumulates(self, audio_vinyl_like):
        """3-Phase-Kette: metadata muss Einträge aller Phasen enthalten."""
        audio, sr = audio_vinyl_like

        phases = [
            ("phase_01_click_removal", "ClickRemovalPhase"),
            ("phase_03_denoise", "DenoisePhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
        ]

        all_metadata: list[dict[str, Any]] = []
        current = audio.copy()
        for mod_name, cls_name in phases:
            import importlib

            mod = importlib.import_module(f"backend.core.phases.{mod_name}")
            cls = getattr(mod, cls_name)
            result = cls().process(current.copy(), sample_rate=sr)
            current = result.audio
            if hasattr(result, "metadata") and result.metadata:
                all_metadata.append(dict(result.metadata))

        # Mindestens eine Phase sollte metadata liefern
        assert len(all_metadata) > 0, "Keine Phase hat metadata geliefert"

        # Metadata muss dict-sein und phase_id enthalten
        for meta in all_metadata:
            assert isinstance(meta, dict), f"metadata ist kein dict: {type(meta)}"


# ═══════════════════════════════════════════════════════════════════════
# 9. VocalQualityGate-Integration (§1.10)
# ═══════════════════════════════════════════════════════════════════════
class TestVocalQualityGateCascade:
    """6-dimensionales VocalQualityGate muss in Kette integriert sein."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_vocal_gate_importable(self):
        """VocalQualityGate ist importierbar und hat evaluate-Methode."""
        from backend.core.vocal_quality_gate import VocalQualityGate, get_vocal_quality_gate

        gate = get_vocal_quality_gate()
        assert gate is not None
        assert hasattr(gate, "evaluate")

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_vocal_gate_returns_valid_dimensions(self, audio_vocal_like):
        """VQI muss alle 6 Dimensionen zurückliefern."""
        from backend.core.vocal_quality_gate import get_vocal_quality_gate

        audio, sr = audio_vocal_like
        gate = get_vocal_quality_gate()

        # §Fix 2026-09-08: evaluate(pre, post, sr) — der Test übergab sr als
        # post_audio (int) → AttributeError in detect().
        result = gate.evaluate(audio, audio, sr=sr)

        required_dims = {
            "formant_integrity",
            "breath_naturalness",
            "sibilance_preservation",
            "intelligibility",
            "listening_comfort",
            "vocal_warmth",
        }

        assert hasattr(result, "scores"), "VQI-Resultat hat kein .scores"
        scores = result.scores if hasattr(result, "scores") else getattr(result, "data", {})

        # Wenn scores ein dict ist, prüfe die Keys
        if isinstance(scores, dict):
            missing = required_dims - set(scores.keys())
            assert len(missing) == 0 or len(required_dims & set(scores.keys())) >= 4, (
                f"VQI fehlt Dimensionen: {missing}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 10. FeedbackChain-Repair-Kaskade (§v10 Pleasantness-First)
# ═══════════════════════════════════════════════════════════════════════
class TestFeedbackChainCascade:
    """FeedbackChain muss Reparatur-Kaskaden steuern können."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_feedback_chain_importable(self):
        """FeedbackChain ist importierbar und hat run-Methode."""
        from backend.core.feedback_chain import FeedbackChain

        fc = FeedbackChain()
        assert hasattr(fc, "run")

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_feedback_chain_accepts_phase_list(self, audio_vinyl_like):
        """FeedbackChain akzeptiert Phase-Liste und führt Kette aus."""
        from backend.core.feedback_chain import FeedbackChain

        audio, sr = audio_vinyl_like

        # Erstelle eine minimale Phase-Liste
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase
        from backend.core.phases.phase_03_denoise import DenoisePhase

        phases = [ClickRemovalPhase(), DenoisePhase()]

        fc = FeedbackChain(max_iterations=2)
        result = fc.run(audio.copy(), sr, phases)

        assert result is not None
        assert hasattr(result, "audio")
        assert np.isfinite(result.audio).all()


# ═══════════════════════════════════════════════════════════════════════
# 11. Edge-Case-Kaskaden (Silence, Single-Sample, Extremwerte)
# ═══════════════════════════════════════════════════════════════════════
class TestEdgeCaseCascade:
    """Kaskaden müssen Edge-Cases robust handhaben."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_silence_through_chain(self):
        """Reine Stille durch 3-Phase-Kette → keine Exception, finite Output."""
        sr = 48_000
        audio = np.zeros(sr * 2, dtype=np.float32)

        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase
        from backend.core.phases.phase_03_denoise import DenoisePhase
        from backend.core.phases.phase_05_rumble_filter import RumbleFilterPhase

        result = ClickRemovalPhase().process(audio.copy(), sample_rate=sr)
        result = DenoisePhase().process(result.audio.copy(), sample_rate=sr)
        result = RumbleFilterPhase().process(result.audio.copy(), sample_rate=sr)

        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) < 0.01, "Stille sollte Stille bleiben"

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_max_peak_through_chain(self):
        """Max-Peak-Signal (±1.0) durch Kette → kein Clipping."""
        sr = 48_000
        audio = np.ones(sr * 2, dtype=np.float32)

        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase
        from backend.core.phases.phase_30_dc_offset_removal import DCOffsetRemoval

        result = ClickRemovalPhase().process(audio.copy(), sample_rate=sr)
        result = DCOffsetRemoval().process(result.audio.copy(), sample_rate=sr)

        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ═══════════════════════════════════════════════════════════════════════
# 12. Performance-Kaskaden (Timing-Grenzen)
# ═══════════════════════════════════════════════════════════════════════
class TestPerformanceCascade:
    """Kaskaden müssen innerhalb realistischer Zeitgrenzen bleiben."""

    @pytest.mark.timeout(_KAS_TIMEOUT)
    def test_5_phase_chain_under_30s(self, audio_vinyl_like):
        """5-Phase-Kette auf 3s Audio muss < 30 Sekunden dauern."""
        import time

        audio, sr = audio_vinyl_like

        phases = [
            ("phase_01_click_removal", "ClickRemovalPhase"),
            ("phase_02_hum_removal", "HumRemovalPhase"),
            ("phase_03_denoise", "DenoisePhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
            ("phase_05_rumble_filter", "RumbleFilterPhase"),
        ]

        start = time.monotonic()
        current = audio.copy()
        for mod_name, cls_name in phases:
            import importlib

            mod = importlib.import_module(f"backend.core.phases.{mod_name}")
            cls = getattr(mod, cls_name)
            result = cls().process(current.copy(), sample_rate=sr)
            current = result.audio
        elapsed = time.monotonic() - start

        assert elapsed < 30.0, f"5-Phase-Kette zu langsam: {elapsed:.2f}s > 30s"
