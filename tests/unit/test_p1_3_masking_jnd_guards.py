"""P1-3 — Audibility: lokale Maskierungs-JND auf §SCK/§WBG/§ATI-Guards.

Vertrag (dsp.instructions.md §WBG/§ATI/§SCK, Hörordnung Ebene 2):
    Ein maskierter Phasen-Delta ist unhörbar → Guards lösen dafür keine
    Rücknahmen/Rollbacks aus (effektive Toleranz = max(fest, lokale JND)).

- §WBG: Der maskierte Anteil des aktuellen Phasen-Verlusts zählt nicht zum
  kumulativen Verlust (vorherige Verluste bleiben voll wirksam).
- §ATI: Onset-Toleranz = max(1.5 dB, JND).
- §SCK: Korrelations-Schwelle relaxiert um min(0.20, JND/30).

Regressionstests: maskierte vs. unmaskierte Fälle (JND über Monkeypatch
deterministisch gesetzt — die JND-Schätzung selbst wird separat getestet).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

SR = 48000


def _jnd(jnd_db: float, above_db: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(jnd_db=jnd_db, delta_above_db=above_db, threshold_db=-60.0)


# ── JND-Schätzung (residuum_masking) ───────────────────────────────────────


class TestDeltaMaskingJND:
    def test_tiny_delta_is_masked(self) -> None:
        from backend.core.residuum_masking import estimate_delta_masking_jnd_db

        rng = np.random.default_rng(0)
        n = SR * 2
        t = np.arange(n) / SR
        pre = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.standard_normal(n)).astype(np.float32)
        post = pre + 0.001 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        r = estimate_delta_masking_jnd_db(pre, post, SR)
        assert r.jnd_db > 3.0, f"Tiny Delta muss maskiert sein, jnd={r.jnd_db}"
        assert r.delta_above_db == 0.0

    def test_loud_delta_is_exposed(self) -> None:
        from backend.core.residuum_masking import estimate_delta_masking_jnd_db

        rng = np.random.default_rng(1)
        n = SR * 2
        t = np.arange(n) / SR
        pre = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.standard_normal(n)).astype(np.float32)
        post = pre + (0.5 * rng.standard_normal(n)).astype(np.float32)
        r = estimate_delta_masking_jnd_db(pre, post, SR)
        assert r.jnd_db == 0.0
        assert r.delta_above_db > 5.0

    def test_noop_delta_is_masked(self) -> None:
        from backend.core.residuum_masking import estimate_delta_masking_jnd_db

        rng = np.random.default_rng(2)
        pre = (0.2 * rng.standard_normal(SR)).astype(np.float32)
        r = estimate_delta_masking_jnd_db(pre, pre, SR)
        assert r.jnd_db > 3.0

    def test_deterministic_and_nan_safe(self) -> None:
        from backend.core.residuum_masking import estimate_delta_masking_jnd_db

        rng = np.random.default_rng(3)
        pre = rng.standard_normal(8192).astype(np.float32)
        post = pre + 0.01 * rng.standard_normal(8192).astype(np.float32)
        r1 = estimate_delta_masking_jnd_db(pre, post, SR)
        r2 = estimate_delta_masking_jnd_db(pre, post, SR)
        assert (r1.jnd_db, r1.delta_above_db) == (r2.jnd_db, r2.delta_above_db)
        bad = estimate_delta_masking_jnd_db(np.full(512, np.nan), np.full(512, np.inf), SR)
        assert bad.jnd_db == 0.0 and bad.delta_above_db == 0.0


# ── §WBG: maskierter Verlust zählt nicht kumulativ ─────────────────────────


def _warmth_signal() -> np.ndarray:
    rng = np.random.default_rng(7)
    n = SR
    t = np.arange(n) / SR
    x = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.25 * np.sin(2 * np.pi * 550 * t) + 0.05 * rng.standard_normal(n)
    out: np.ndarray = np.asarray(x, dtype=np.float32)
    return out


class TestWarmthMaskingJND:
    def test_masked_loss_does_not_trigger_blend(self) -> None:
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _warmth_signal()
        post = audio * 0.7  # ≈ −3.1 dB im Band
        with patch(
            "backend.core.dsp.warmth_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ):
            result = measure_warmth_band_delta(audio, post, SR, cumulative_loss_db=0.0)
        assert result.loss_db > 2.0
        assert result.warmth_blend_factor == pytest.approx(1.0)  # maskiert → kein Blend

    def test_unmasked_loss_triggers_blend(self) -> None:
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _warmth_signal()
        post = audio * 0.7
        with patch(
            "backend.core.dsp.warmth_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(0.0),
        ):
            result = measure_warmth_band_delta(audio, post, SR, cumulative_loss_db=0.0)
        assert result.warmth_blend_factor < 1.0  # hörbar → Blend

    def test_prior_cumulative_loss_stays_effective_on_noop_phase(self) -> None:
        from backend.core.dsp.warmth_guard import WARMTH_LOSS_THRESHOLD_DB, measure_warmth_band_delta

        audio = _warmth_signal()
        # No-op-Phase mit bereits überschrittenem kumulativem Verlust:
        # die Antwort auf VORHERIGE Verluste bleibt erhalten (Blend < 1).
        with patch(
            "backend.core.dsp.warmth_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ):
            result = measure_warmth_band_delta(
                audio,
                audio.copy(),
                SR,
                cumulative_loss_db=WARMTH_LOSS_THRESHOLD_DB + 1.0,
            )
        assert result.warmth_blend_factor < 1.0


# ── §ATI: Onset-Toleranz = max(1.5 dB, JND) ────────────────────────────────


def _transient_signal() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    n = 4096
    x = (0.02 * rng.standard_normal(n)).astype(np.float32)
    for start in (512, 2048):
        x[start : start + 64] = np.linspace(0.0, 0.8, 64).astype(np.float32) * np.hanning(64)
    mask = np.zeros(n, dtype=bool)
    mask[512:1024] = True
    mask[2048:2560] = True
    return x, mask


class TestOnsetMaskingJND:
    def test_unmasked_boost_is_limited(self) -> None:
        from backend.core.dsp.onset_guard import apply_onset_protection_mask

        pre, mask = _transient_signal()
        post = pre.copy()
        post[512:1024] *= 2.0  # +6 dB in Onset-Fenstern
        with patch(
            "backend.core.dsp.onset_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(0.0, above_db=6.0),
        ):
            result = apply_onset_protection_mask(pre, post, mask, max_delta_db=1.5)
        assert not np.allclose(result[512:1024], post[512:1024])  # begrenzt

    def test_masked_boost_is_allowed(self) -> None:
        from backend.core.dsp.onset_guard import apply_onset_protection_mask

        pre, mask = _transient_signal()
        post = pre.copy()
        post[512:1024] *= 1.9  # +5,6 dB — bei JND 6 dB maskiert (Ratio < 10^(6/20))
        with patch(
            "backend.core.dsp.onset_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ):
            result = apply_onset_protection_mask(pre, post, mask, max_delta_db=1.5)
        assert np.allclose(result, post)  # kein Blend — Delta maskiert


# ── §SCK: maskierte Abweichung relaxiert die Korrelations-Schwelle ─────────


def _spectral_color_signals() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    n = 65536
    x = rng.standard_normal(n).astype(np.float32)
    # 1/f-Form → definierte Spektralfarbe (pre_std > 0.5 dB)
    spec = np.fft.rfft(x.astype(np.float64))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec *= 1.0 / (1.0 + freqs / 300.0) ** 0.7
    pre = np.fft.irfft(spec, n=n).astype(np.float32)
    pre /= np.max(np.abs(pre)) + 1e-9
    # post: zusätzliches Band-Rauschen 4–8 kHz (spektrale Abweichung)
    band = np.fft.rfft(rng.standard_normal(n).astype(np.float64))
    band[(freqs < 4000.0) | (freqs > 8000.0)] = 0.0
    addition = np.fft.irfft(band, n=n).astype(np.float32)
    addition *= 0.8 / (np.max(np.abs(addition)) + 1e-9)  # corr ≈ 0.85 (kalibriert)
    post = (pre + addition).astype(np.float32)
    return pre, post


class TestSpectralColorMaskingJND:
    def test_unmasked_deviation_fails(self) -> None:
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        pre, post = _spectral_color_signals()
        with patch("backend.core.calibration_context.get_calibration_context", return_value=None), patch(
            "backend.core.dsp.spectral_color_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(0.0, above_db=8.0),
        ):
            result = check_spectral_color_preservation(pre, post, SR, threshold=0.97)
        assert result.correlation < 0.97
        assert result.ok is False

    def test_masked_deviation_passes(self) -> None:
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        pre, post = _spectral_color_signals()
        with patch("backend.core.calibration_context.get_calibration_context", return_value=None), patch(
            "backend.core.dsp.spectral_color_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ):
            result_masked = check_spectral_color_preservation(pre, post, SR, threshold=0.97)
        assert result_masked.ok is True  # Schwelle 0.97 − 0.20 = 0.77
        assert result_masked.correlation >= 0.77


# ── §0p Formant-Schutz: Toleranz = max(fest/JND-Frequenz, lokale Maskierung) ─


def _vowel_formant_signals() -> tuple[np.ndarray, np.ndarray]:
    """Synthetischer Vokal (F1≈740, F2≈1830, F3≈3080 Hz) + F1-Band-Boost.

    Kalibriert: Der Boost hebt die F1-Band-Energie um ≈2,6 dB — über der
    festen Frequenz-JND-Toleranz (~1,8 dB), unter der 6-dB-Maskierungs-Obergrenze.
    """
    from scipy.signal import butter, lfilter, sosfiltfilt

    rng = np.random.default_rng(5)
    n = SR * 2
    glott = np.zeros(n)
    for k in range(0, n, SR // 120):
        seg = np.hanning(240)
        end = min(k + 240, n)
        glott[k:end] += seg[: end - k]

    def resonator(x: np.ndarray, f: float, bw: float) -> np.ndarray:
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * f / SR
        return lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], x)  # type: ignore[no-any-return]

    x = glott
    for f, bw in ((600, 90), (1500, 110), (2500, 150)):
        x = resonator(x, f, bw)
    x = x / (np.max(np.abs(x)) + 1e-9)
    pre = (x + 0.02 * rng.standard_normal(n)).astype(np.float32)
    sos = butter(4, [700 / (SR / 2), 790 / (SR / 2)], btype="band", output="sos")
    band = sosfiltfilt(sos, pre)
    post = (pre + 0.7 * band).astype(np.float32)
    return pre, post


class TestFormantMaskingJND:
    def test_unmasked_formant_shift_triggers_rollback(self) -> None:
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        pre, post = _vowel_formant_signals()
        with patch(
            "backend.core.dsp.lpc_formant_tracker.delta_masking_margin_db_per_band",
            return_value=np.zeros(24, dtype=np.float64),
        ):
            rollback, shift = check_formant_shift_db(pre, post, SR, threshold_db=2.0)
        assert shift > 2.0, f"Erwarteter F1-Shift > 2 dB, erhalten {shift:.2f}"
        assert rollback is True

    def test_masked_formant_shift_no_rollback(self) -> None:
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        pre, post = _vowel_formant_signals()
        with patch(
            "backend.core.dsp.lpc_formant_tracker.delta_masking_margin_db_per_band",
            return_value=np.full(24, 6.0, dtype=np.float64),
        ):
            rollback, _ = check_formant_shift_db(pre, post, SR, threshold_db=2.0)
        assert rollback is False  # Toleranz max(1,8 dB; 6 dB JND) = 6 dB

    def test_identical_signal_no_rollback(self) -> None:
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        pre, _ = _vowel_formant_signals()
        rollback, shift = check_formant_shift_db(pre, pre, SR)
        assert rollback is False
        assert shift == 0.0

    def test_burg_lpc_long_segment_no_crash(self) -> None:
        """§V6-Fix: _burg_lpc lief bei langen Segmenten in einen Shape-Mismatch
        (Original-n statt schrumpfender Länge) → Guard war stummer No-op."""
        from backend.core.dsp.lpc_formant_tracker import _burg_lpc

        rng = np.random.default_rng(7)
        a = _burg_lpc(rng.standard_normal(32000), order=16)
        assert a.shape == (17,)
        assert np.all(np.isfinite(a))


# ── §Gain-Step: effektive Schwelle = max(1,5 dB, JND) ───────────────────────


class TestGainStepMaskingJND:
    def test_unmasked_gain_step_fixed_threshold_and_warning(self, caplog) -> None:
        from backend.core.temporal_continuity_guard import check_temporal_continuity

        rng = np.random.default_rng(21)
        pre = (0.4 * rng.standard_normal(SR * 2)).astype(np.float32)
        post = (pre * 2.0).astype(np.float32)  # +6 dB Gain-Sprung
        with patch(
            "backend.core.temporal_continuity_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(0.0, above_db=6.0),
        ), caplog.at_level("WARNING", logger="backend.core.temporal_continuity_guard"):
            r = check_temporal_continuity(pre, post, "phase_test", SR)
        assert r.gain_step_threshold_db == 1.5
        assert r.gain_step_db > 5.0
        assert any("Mikro-Klick-Risiko" in rec.message for rec in caplog.records)

    def test_masked_gain_step_raises_threshold_no_warning(self, caplog) -> None:
        from backend.core.temporal_continuity_guard import check_temporal_continuity

        rng = np.random.default_rng(21)
        pre = (0.4 * rng.standard_normal(SR * 2)).astype(np.float32)
        post = (pre * 1.9).astype(np.float32)  # ≈ +5,6 dB — unter dem 6-dB-JND-Cap
        with patch(
            "backend.core.temporal_continuity_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ), caplog.at_level("WARNING", logger="backend.core.temporal_continuity_guard"):
            r = check_temporal_continuity(pre, post, "phase_test", SR)
        assert r.gain_step_threshold_db == 6.0  # max(1,5; 6,0)
        assert not any("Mikro-Klick-Risiko" in rec.message for rec in caplog.records)
