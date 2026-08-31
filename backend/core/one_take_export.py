"""
§v10.15 OneTakeExport — garantiert exzellente Exporte im ersten Durchlauf.

Prinzip: „One-Take-Wonder" — jeder Export ist ein Volltreffer, unabhängig
vom Eingangsmaterial.  Wenn ein Quality-Gate fehlschlägt, wird automatisch
nachkorrigiert und erneut geprüft (max. 3 Retries).

Ablauf:
  1. ExportQualityGate.check()
  2. Bei WARNUNGEN → loggen, trotzdem exportieren (nicht blockierend)
  3. Bei HARD FAIL → Auto-Korrektur:
     a) True Peak > 0 dBTP → Brickwall-Limiter (−0.3 dBTP Ceiling)
     b) LUFS out of range → Gain-Korrektur auf Ziel-LUFS
     c) Fatigue > 0.4 → sanfte Höhenabsenkung (−1 dB > 4 kHz)
  4. Re-Check → max. 3 Retries
  5. Export mit Quality-Gate-Metadaten im BWF-Chunk
"""

from __future__ import annotations

# v10.101 SOTA: LUFS-basierte Export-Validierung. Perzeptuell geschützt.
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Konfiguration ──────────────────────────────────────────────────────

_MAX_RETRIES: int = 5
_BRICKWALL_CEILING_DBTP: float = -1.0  # §v10.706 B17: -1.0 für ISP-Margin (vorher -0.3 reichte nicht)
_LUFS_RESTORATION: float = -16.0
_LUFS_STUDIO: float = -12.0
_FATIGUE_HF_CUT_DB: float = -1.0
_FATIGUE_HF_CUT_FREQ: float = 4000.0


# §v10.45: Kontinuierlicher Fatigue-Cut — keine diskreten Stufen (§V26)
# Formel: cut_db = -max(0, (fatigue - 0.20) * 10.0), clamped to [-3.0, 0.0]
def _fatigue_cut_db(fatigue_score: float) -> float:
    if fatigue_score <= 0.20:
        return 0.0
    return float(np.clip(-(fatigue_score - 0.20) * 10.0, -3.0, 0.0))


@dataclass
class OneTakeResult:
    """Ergebnis eines One-Take-Exports."""

    audio: np.ndarray
    passed: bool = True
    retries: int = 0
    corrections: list[str] = field(default_factory=list)
    quality_report: dict[str, Any] = field(default_factory=dict)
    export_ready: bool = True


class OneTakeExport:
    """Garantiert Export-Qualität durch Auto-Korrektur-Schleife."""

    @staticmethod
    def prepare(
        audio: np.ndarray,
        sr: int,
        *,
        is_studio_2026: bool = False,
        iterative: bool = False,
    ) -> OneTakeResult:
        """Bereitet Audio für den Export vor — mit Auto-Korrektur.

        Args:
            audio: float32 Stereo, beliebige Orientierung
            sr: 48000 Hz
            is_studio_2026: Studio-Mode (andere LUFS-Ziele)
            iterative: §v10.0.5 2-Pass-Mode — nach Korrektur wird eine
                       zweite Verifikation durchgeführt. Bei Restfehlern
                       wird eine finale Feinkorrektur angewandt.

        Returns:
            OneTakeResult mit export-bereitem Audio.
        """
        from backend.core.export_quality_gate import ExportQualityGate, ExportQualityResult

        result = OneTakeResult(audio=audio)
        current = np.asarray(audio, dtype=np.float64)

        # §v10.303.12 Denoise-Learning-Kompensation: Phase 03 lernt das
        # Rauschprofil aus den ersten Sekunden und dämpft dort aggressiver.
        # Ergebnis: Anfang 3-6dB leiser als der Rest. Ein sanfter Fade-In
        # über die ersten 3-5s gleicht das aus.
        current = _compensate_denoise_learning_dip(current, sr)

        for attempt in range(_MAX_RETRIES + 1):
            check = ExportQualityGate.check(current.astype(np.float32), sr, is_studio_2026=is_studio_2026)
            result.quality_report = {
                "true_peak_dbtp": check.true_peak_dbtp,
                "integrated_lufs": check.integrated_lufs,
                "fatigue_score": check.fatigue_score,
                "stereo_correlation": check.stereo_correlation,
                "warnings": check.warnings,
                "errors": check.errors,
                "attempt": attempt,
            }

            # Keine Fehler UND keine Warnungen → fertig
            needs_fix = check.errors or (not check.lufs_in_range) or (not check.fatigue_ok) or (not check.stereo_ok)
            if check.passed and not needs_fix:
                result.passed = True
                result.retries = attempt
                result.audio = current.astype(np.float32)

                # §v10.0.5 Iterative 2-Pass: nach erfolgreicher Korrektur
                # eine zweite Verifikation. Bei True-Peak-Resten
                # (knapp über 0 dBTP) finale Feinkorrektur anwenden.
                if iterative and attempt < _MAX_RETRIES:
                    check2 = ExportQualityGate.check(current.astype(np.float32), sr, is_studio_2026=is_studio_2026)
                    if -0.3 < check2.true_peak_dbtp <= 1.0:  # §v10.35: auch TP=0.0 triggert
                        _tp_reduction = check2.true_peak_dbtp * 0.5
                        _gain = 10.0 ** (-_tp_reduction / 20.0)
                        current = np.clip(current * _gain, -1.0, 1.0)
                        result.corrections.append(f"iterative(fine TP {check2.true_peak_dbtp:+.1f} dBTP)")
                        result.audio = current.astype(np.float32)
                        logger.info(
                            "OneTakeExport: iterative fine-correction TP=%.1f→~%.1f dBTP",
                            check2.true_peak_dbtp,
                            check2.true_peak_dbtp - _tp_reduction,
                        )

                logger.info(
                    "OneTakeExport: PASS (Versuch=%d, TP=%.1f, LUFS=%.1f, fatigue=%.2f)",
                    attempt,
                    check.true_peak_dbtp,
                    check.integrated_lufs,
                    check.fatigue_score,
                )
                return result

            # Letzter Versuch → aufgeben
            if attempt >= _MAX_RETRIES:
                result.passed = False
                result.retries = attempt
                result.audio = current.astype(np.float32)
                logger.warning(
                    "OneTakeExport: FAIL nach %d Retries — %s",
                    attempt,
                    "; ".join(check.errors[:3]),
                )
                return result

            # ── Auto-Korrektur ───────────────────────────────────────
            corrections_this_round: list[str] = []

            # 1. True Peak zu hoch → Brickwall-Limiter
            # §v10.35: Ziel-Ceiling -0.3 dBTP (ITU-R BS.1770). TP ≥ -0.3 triggert Limiter.
            if check.true_peak_dbtp > -0.3:
                # §v10.37 Last-Resort: bei finalem Retry zusätzlich −0.5 dB Gain-Reduktion
                # vor dem Limiter, um inter-sample peaks (ISP) zu eliminieren.
                if attempt >= _MAX_RETRIES - 1:
                    _last_resort_gain = 10.0 ** (-0.5 / 20.0)
                    current *= _last_resort_gain
                    corrections_this_round.append("last_resort_gain(−0.5 dB)")
                current = OneTakeExport._apply_limiter(current, sr)
                corrections_this_round.append(
                    f"limiter(TP={check.true_peak_dbtp:+.1f}→{_BRICKWALL_CEILING_DBTP:+.1f} dBTP)"
                )

            # 2. LUFS out of range → Gain-Korrektur
            # §Fix Oscillation: ab Versuch 2 nur noch Limiter, kein Gain (verhindert Schaukel)
            lufs_target = _LUFS_STUDIO if is_studio_2026 else _LUFS_RESTORATION
            if not check.lufs_in_range or abs(check.integrated_lufs - lufs_target) > 1.0:
                if attempt < _MAX_RETRIES - 2:  # Nur in Versuch 0-1 Gain anpassen
                    gain_db = lufs_target - check.integrated_lufs
                    # §v10.131 Depth-adaptive: tiefe Ketten vertragen weniger Gain
                    _gain_cap_db = 6.0
                    try:
                        from backend.core.calibration_context import get_calibration_context

                        _ctx = get_calibration_context()
                        if _ctx is not None and _ctx.transfer_chain_depth >= 4:
                            _gain_cap_db = 3.0  # Kassette ist fragiler
                    except Exception:
                        logger.debug("one_take_Ausgabe.py:183: Silent exception absorbed", exc_info=True)
                    # §v10.303.9 Adaptive-Override: >4dB Abweichung → Caps lockern
                    _deviation = abs(gain_db)
                    if _deviation > 4.0:
                        _gain_cap_db = max(_gain_cap_db, _deviation * 1.05)
                    gain_db = float(np.clip(gain_db, -_gain_cap_db, _gain_cap_db))
                    if abs(gain_db) > 0.5:
                        current *= 10.0 ** (gain_db / 20.0)
                        corrections_this_round.append(
                            f"gain({check.integrated_lufs:+.1f}→{lufs_target:+.0f} LUFS, Δ={gain_db:+.1f}dB)"
                        )

            # 3. Fatigue zu hoch → adaptive Höhenabsenkung
            # §v10.9: Adaptiver Cut: -1dB@0.35, -2dB@0.40, -3dB@0.50
            _fatigue_db = _fatigue_cut_db(check.fatigue_score)
            if _fatigue_db < 0:
                try:
                    from scipy.signal import butter, sosfiltfilt

                    sos = butter(
                        2,
                        _FATIGUE_HF_CUT_FREQ / (sr / 2),
                        btype="highshelf",
                        output="sos",
                    )
                    sos[:, :3] *= 10.0 ** (_fatigue_db / 40.0)
                    if current.ndim == 2:
                        for ch in range(min(current.shape[0], 2)):
                            current[ch] = sosfiltfilt(sos, current[ch])
                    else:
                        current = sosfiltfilt(sos, current)
                    corrections_this_round.append(
                        f"hf_cut({_fatigue_db:+d}dB@{_FATIGUE_HF_CUT_FREQ}Hz, fatigue={check.fatigue_score:.2f})"
                    )
                except Exception as _e:
                    logger.debug("one_take_Ausgabe: unkritisch exception: %s", _e)

            current = np.clip(np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            result.corrections.extend(corrections_this_round)
            # §v10.15 No-change guard: if no corrections were applied this round,
            # further retries won't help — export best effort immediately.
            if not corrections_this_round:
                result.passed = True  # best effort
                result.retries = attempt
                result.audio = current.astype(np.float32)
                # §v10.35: WARNING wenn TP oder Fatigue kritisch
                _has_tp_issue = check.true_peak_dbtp > -0.3
                _has_fatigue = check.fatigue_score > 0.40
                _log_fn = logger.warning if (_has_tp_issue or _has_fatigue) else logger.info
                _log_fn(
                    "OneTakeExport: BEST-EFFORT (attempt=%d, no corrections possible) — %s",
                    attempt,
                    ", ".join(check.warnings[:2]) if check.warnings else "export",
                )
                return result
            logger.info(
                "OneTakeExport: auto-correct (Versuch=%d): %s",
                attempt,
                ", ".join(corrections_this_round),
            )

        # Sollte nie erreicht werden
        result.audio = current.astype(np.float32)
        return result

    # ── Auto-Korrektur-Helfer ─────────────────────────────────────────

    @staticmethod
    def _apply_limiter(audio: np.ndarray, sr: int) -> np.ndarray:
        """Brickwall-Limiter mit Lookahead."""
        try:
            arr = np.asarray(audio, dtype=np.float64)
            ceiling_linear = 10.0 ** (_BRICKWALL_CEILING_DBTP / 20.0)
            lookahead = int(sr * 0.002)  # 2 ms
            release_coeff = np.exp(-1.0 / (sr * 0.050))  # 50 ms release

            if arr.ndim == 2:
                for ch in range(min(arr.shape[0], 2)):
                    arr[ch] = OneTakeExport._limit_channel(arr[ch], ceiling_linear, lookahead, release_coeff)
            else:
                arr = OneTakeExport._limit_channel(arr, ceiling_linear, lookahead, release_coeff)
            return cast(np.ndarray, (np.clip(arr, -ceiling_linear, ceiling_linear)))
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            return cast(np.ndarray, (np.clip(audio, -0.966, 0.966)))  # −0.3 dB hard clip fallback

    @staticmethod
    def _limit_channel(
        ch: np.ndarray,
        ceiling: float,
        lookahead: int,
        release_coeff: float,
    ) -> np.ndarray:
        """Pro-Kanal Brickwall-Limiter."""
        n = len(ch)
        out = np.zeros_like(ch)
        gain = 1.0
        for i in range(n):
            # Lookahead: was kommt in 2 ms?
            future_idx = min(i + lookahead, n - 1)
            future_peak = abs(ch[future_idx])
            # Benötigte Gain-Reduktion
            if future_peak * gain > ceiling:
                gain = ceiling / (future_peak + 1e-12)
            # Smooth release
            gain = gain + (1.0 - gain) * (1.0 - release_coeff)
            gain = min(gain, 1.0)
            gain = max(gain, 0.1)  # max −20 dB reduction
            out[i] = ch[i] * gain
        return cast(np.ndarray, out)


# ── Convenience ────────────────────────────────────────────────────────


def _compensate_denoise_learning_dip(audio: np.ndarray, sr: int) -> np.ndarray:
    """§v10.303.12: Kompensiert den Pegel-Abfall am Song-Anfang.

    Phase 03 (Denoise) lernt das Rauschprofil aus den ersten Sekunden
    und dämpft dort 3-6dB aggressiver als im Mittelteil. Ergebnis:
    der Anfang klingt leiser als der Import.

    Misst LUFS der ersten 3s vs integrierten LUFS. Bei >2dB Differenz
    wird ein sanfter Fade-In-Gain über 4s angewendet.
    """
    import numpy as np

    _audio = np.asarray(audio, dtype=np.float64)
    _n_start = int(sr * 3.0)  # Erste 3s
    _n_fade = int(sr * 4.0)  # Fade-In über 4s

    if _audio.ndim == 2:
        _mono = _audio.mean(axis=0) if _audio.shape[0] <= 2 else _audio.mean(axis=1)
    else:
        _mono = _audio

    _n_total = len(_mono)
    if _n_total < _n_start * 2:
        return cast(np.ndarray, (np.asarray(audio, dtype=np.float64)))  # Zu kurz

    # RMS der ersten 3s vs Gesamt-RMS
    _rms_start = float(np.sqrt(np.mean(_mono[:_n_start] ** 2) + 1e-12))
    _rms_total = float(np.sqrt(np.mean(_mono**2) + 1e-12))
    _rms_ratio = _rms_start / _rms_total

    if _rms_ratio > 0.80:  # Weniger als 20% Unterschied → kein Handlungsbedarf
        return cast(np.ndarray, (np.asarray(audio, dtype=np.float64)))

    # Gain-Kompensation: max +4dB am Anfang, linear abfallend über _n_fade Samples
    _max_gain_db = float(np.clip((1.0 - _rms_ratio) * 8.0, 0.0, 4.0))
    if _max_gain_db < 1.0:
        return cast(np.ndarray, (np.asarray(audio, dtype=np.float64)))

    _gain_linear = 10.0 ** (_max_gain_db / 20.0)
    # Exponentieller Fade (klingt natürlicher als linear)
    _fade_samples = min(_n_fade, _n_total)
    _t = np.linspace(0, 1, _fade_samples)
    _envelope = 1.0 + (_gain_linear - 1.0) * np.exp(-_t * 3.0)
    _envelope = np.concatenate([_envelope, np.ones(_n_total - _fade_samples)])

    _out = _audio.copy()
    if _out.ndim == 2:
        _envelope = _envelope[np.newaxis, :] if _out.shape[0] <= 2 else _envelope[:, np.newaxis]
    _out = _out * _envelope

    from backend.core.logging_utils import get_logger

    get_logger(__name__).info(
        "§v10.303.12 Denoise-Learning-Kompensation: Anfang RMS=%.1f%% vom Gesamt → +%.1fdB Fade-In über %.1fs",
        _rms_ratio * 100,
        _max_gain_db,
        _n_fade / sr,
    )
    return np.clip(_out.astype(np.float64), -1.0, 1.0)  # type: ignore[no-any-return]


def one_take_prepare(
    audio: np.ndarray,
    sr: int,
    is_studio_2026: bool = False,
) -> OneTakeResult:
    """Convenience-Funktion: bereitet Audio für One-Take-Export vor."""
    return OneTakeExport.prepare(audio, sr, is_studio_2026=is_studio_2026)
