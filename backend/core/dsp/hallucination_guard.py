"""
§2.46e [RELEASE_MUST] Hallucination-Guard DSP-API — backend/core/dsp/hallucination_guard.py

Lightweight wrapper around the primary hallucination guard
(backend/core/hallucination_guard.py) that exposes the
`check_hallucination(pre, post)` interface required by the VERBOTEN table §2.46e:

    > `check_hallucination(pre, post)` aus `backend/core/dsp/hallucination_guard.py`
    > nach jeder ADDITIVE-Phase;
    > `spectral_novelty > 0.15` → Phase-Rollback (Restoration);
    > `> 0.08` → Score-Penalty 0.3

Threshold semantics (§2.46e normativ):
    spectral_novelty > 0.15  — rollback required in Restoration mode
    spectral_novelty > 0.08  — score penalty −0.3 (both modes)
    harmonic_ceiling_violation == True — hard rollback (BW-Ceiling violated)

Thread-safe via module-level functions (no singleton needed — pure DSP).

Author: Aurik 10.0.0 Engineering
Version: 1.0.0 (§2.46e RELEASE_MUST, BUG-FIX v10.0.0)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# §2.46e normative thresholds (floors — may be raised by SFT adaptation)
_ROLLBACK_THRESHOLD_FLOOR: float = 0.15  # spectral_novelty > this → rollback (Restoration)
_PENALTY_FLOOR: float = 0.08  # spectral_novelty > this → score penalty -0.3


def _get_adaptive_penalty_threshold() -> float:
    """§v10.122 Penalty-Schwellwert skaliert mit dem adaptiven Rollback-Schwellwert.

    Bei depth 4 (Rollback=0.40): Penalty = max(0.08, 0.40*0.5) = 0.20.
    Verhindert Score-Penalties für legitime Carrier-Inversion-Neuheit.
    """
    return max(_PENALTY_FLOOR, _get_adaptive_rollback_threshold() * 0.5)


def _get_adaptive_rollback_threshold() -> float:
    """§v10.122 Liest den depth-adaptiven Schwellwert aus der SFT-Kalibrierung.

    Der Wert wird von calibrate_sft_thresholds() pro Song gesetzt und
    skaliert mit der Transfer-Chain-Tiefe (0.15 bei depth 1, 0.55 bei depth 5+).
    Fallback auf 0.15 wenn die SFT-Kalibrierung nicht geladen werden kann.
    """
    try:
        from backend.core.signal_flow_tracer import get_hallucination_guard_threshold

        return max(_ROLLBACK_THRESHOLD_FLOOR, get_hallucination_guard_threshold())
    except Exception as exc:
        logger.debug("§V6 SFT-Halluzinationsschwelle nicht verfügbar — Floor-Wert zurückgegeben (0.15): %s", exc)
        return _ROLLBACK_THRESHOLD_FLOOR


@dataclass
class HallucinationCheckResult:
    """Result of check_hallucination(pre, post).

    Attributes:
        spectral_novelty: Fraction of energy in new/hallucinated spectral bins [0, 1].
        requires_rollback: True if spectral_novelty > 0.15 (Restoration hard limit §2.46e).
        score_penalty: Score deduction to apply; 0.3 when spectral_novelty > 0.08.
        harmonic_ceiling_violation: True if energy above BW-Ceiling grew > 8× (§2.46e).
        metadata: Additional diagnostic information.
    """

    spectral_novelty: float
    requires_rollback: bool
    score_penalty: float
    harmonic_ceiling_violation: bool
    metadata: dict


def check_hallucination(
    pre: npt.NDArray[np.float32],
    post: npt.NDArray[np.float32],
    *,
    sr: int = 48000,
    mode: str = "restoration",
    material_bw_ceiling_hz: float | None = None,
    bw_extension_context: bool = False,
) -> HallucinationCheckResult:
    """§2.46e: Check whether an additive phase introduced hallucinated material.

    Delegates spectral_novelty measurement to
    `backend.core.hallucination_guard.compute_spectral_novelty` and applies
    the normative threshold logic defined in §2.46e.

    The rollback threshold is now depth-adaptive (§v10.122): the SFT
    calibration sets a per-song base threshold (0.15 at depth 1,
    0.55 at depth 5+) which is used as the floor.  Existing bandwidth
    and BW-extension modifiers continue to raise the threshold further.

    Args:
        pre:   Audio array before the additive phase (mono float32).
        post:  Audio array after the additive phase (mono float32).
        sr:    Sample rate in Hz (must be 48 000 Hz in phase context).
        mode:  Processing mode — "restoration" enforces hard rollback;
               "studio_2026" enforces MUSHRA check instead.
        material_bw_ceiling_hz: Physical BW ceiling of the carrier medium (Hz).
               If provided, energy growth above this frequency is checked for
               harmonic_ceiling_violation (> 8× increase = hard rollback).
        bw_extension_context: When True the phase intentionally adds new HF content
               as part of carrier-inverse BW restoration (AudioSR / NVSR). In this
               case the rollback threshold is relaxed, because new
               spectral energy below the material ceiling is physically expected and
               not a hallucination. harmonic_ceiling_violation veto remains absolute.

    Returns:
        HallucinationCheckResult with decision flags and diagnostic metadata.
    """
    pre_arr = np.asarray(pre, dtype=np.float32)
    post_arr = np.asarray(post, dtype=np.float32)

    # Mono-ify if stereo
    if pre_arr.ndim == 2:
        pre_arr = np.mean(pre_arr, axis=1 if pre_arr.shape[1] <= 8 else 0).astype(np.float32)
    if post_arr.ndim == 2:
        post_arr = np.mean(post_arr, axis=1 if post_arr.shape[1] <= 8 else 0).astype(np.float32)

    spectral_novelty: float = 0.0
    harmonic_ceiling_violation: bool = False
    meta: dict = {}

    # --- Primary: delegate to backend.core.hallucination_guard ---
    try:
        _primary_hallucination_guard: Any = importlib.import_module("backend.core.hallucination_guard")

        spectral_novelty, sn_meta = _primary_hallucination_guard.compute_spectral_novelty(pre_arr, post_arr, sr=sr)
        spectral_novelty = float(np.nan_to_num(spectral_novelty, nan=0.0, posinf=0.0, neginf=0.0))
        meta.update(sn_meta)

        # BW-Ceiling check (§2.46e harmonic_ceiling_violation)
        if material_bw_ceiling_hz is not None and material_bw_ceiling_hz > 0:
            try:
                _chv = _primary_hallucination_guard.check_harmonic_ceiling_violation  # pyright: ignore[reportCallIssue]
                harmonic_ceiling_violation, ceiling_meta = _chv(
                    pre_arr,
                    post_arr,
                    material_bw_ceiling_hz,
                    sr=sr,
                )
                # §v10.122 Depth-adaptive ceiling: tiefere Ketten haben niedrigeres
                # BW-Ceiling → selbst kleine HF-Restauration triggert 8×-Schwelle.
                # Skaliere die effektive Schwelle mit der Chain-Depth.
                if harmonic_ceiling_violation:
                    _hg_base = _get_adaptive_rollback_threshold()
                    _ceiling_factor = _hg_base / _ROLLBACK_THRESHOLD_FLOOR  # 1.0–3.7
                    _ceiling_ratio = float(ceiling_meta.get("ceiling_band_ratio", 999.0))
                    if _ceiling_ratio < 8.0 * _ceiling_factor:
                        harmonic_ceiling_violation = False
                        logger.info(
                            "§v10.122 Ceiling-Veto aufgehoben: Verhaeltnis=%.1f < %.1f (depth-factor=%.2f)",
                            _ceiling_ratio,
                            8.0 * _ceiling_factor,
                            _ceiling_factor,
                        )
                meta.update(ceiling_meta)
                meta["bw_ceiling_hz"] = material_bw_ceiling_hz
                meta["harmonic_ceiling_violation"] = harmonic_ceiling_violation
            except Exception as exc:
                logger.debug("Pruefung_harmonic_ceiling_violation fehlgeschlagen (unkritisch): %s", exc)

    except ImportError:
        # DSP fallback: simple spectral energy delta
        logger.warning("§G23 hallucination_guard primary import fehlgeschlagen; using DSP Ersatzpfad")
        spectral_novelty, meta = _compute_spectral_novelty_dsp(pre_arr, post_arr, sr)

    except Exception as exc:
        logger.warning(
            "Pruefung_hallucination: primary computation fehlgeschlagen (%s) — returning safe defaults.", exc
        )
        spectral_novelty = 0.0
        meta["error"] = str(exc)

    # --- Apply §2.46e threshold logic ---
    requires_rollback = False
    score_penalty = 0.0

    # §v10.122 SFT-Integration: Depth-adaptiver Basis-Schwellwert statt hartem 0.15.
    _base_rollback_threshold = _get_adaptive_rollback_threshold()

    # §2.46b Adaptive-Threshold: Je schmaler die Eingangs-Bandbreite,
    # desto mehr "neue" Spektralenergie ist legitime Restauration.
    _pre_bw = float(meta.get("pre_effective_bandwidth_hz", 20000.0) or 20000.0)
    # Fallback: estimate from pre audio if meta doesn't have bandwidth
    if _pre_bw >= 20000.0 and pre_arr.size > 0:
        try:
            _pre_fft = np.abs(np.fft.rfft(pre_arr[: min(len(pre_arr), sr)]))
            _cumsum = np.cumsum(_pre_fft)
            _total = _cumsum[-1] + 1e-12
            _bw_idx = int(np.searchsorted(_cumsum, 0.95 * _total))
            _pre_bw = float(_bw_idx * sr / len(_pre_fft))
        except Exception as _e:
            logger.debug("hallucination_guard: unkritisch exception: %s", _e)
    if _pre_bw < 1000.0:
        _base_rollback_threshold = max(_base_rollback_threshold, 0.30)
    elif _pre_bw < 4000.0:
        _base_rollback_threshold = max(_base_rollback_threshold, 0.20)

    # BW-extension context: carrier-inverse HF restoration is expected to add new
    # spectral content below the material ceiling — raise rollback threshold.
    # harmonic_ceiling_violation (above-ceiling energy growth) veto remains absolute.
    _effective_rollback_threshold = _base_rollback_threshold
    if bw_extension_context:
        if material_bw_ceiling_hz is None:
            # Codec-Material ohne BW-Ceiling (mp3_low, mp3_high, aac, streaming):
            # HF-Rekonstruktion ist erwartete Carrier-Inversion des Codec-Verlustes —
            # AudioSR/NVSR synthetisiert Inhalt, der durch Encoder verworfen wurde.
            # §v10.122: Skaliert mit depth-adaptivem Basis-Schwellwert (×2.5, min 0.50).
            _effective_rollback_threshold = max(_base_rollback_threshold * 2.5, 0.50)
        else:
            # Analog-Material mit BW-Ceiling (Shellac, Vinyl, Kassette …):
            # §v10.122: Skaliert mit depth-adaptivem Basis-Schwellwert (×1.5, min 0.20).
            # Depth 4 (base=0.40) → 0.60 Toleranz für legitime HF-Restauration.
            _effective_rollback_threshold = max(_base_rollback_threshold * 1.5, 0.20)
        meta["bw_extension_context"] = True
        meta["effective_rollback_threshold"] = _effective_rollback_threshold

    if harmonic_ceiling_violation:
        # Hard rollback regardless of spectral_novelty
        requires_rollback = True
        score_penalty = 0.3
        logger.warning(
            "§2.46e HallucinationGuard: harmonic_ceiling_violation=True (ceiling=%.0f Hz) → hard rollback.",
            material_bw_ceiling_hz or -1,
        )
    elif spectral_novelty > _effective_rollback_threshold:
        if mode == "restoration":
            requires_rollback = True
        score_penalty = 0.3
        logger.warning(
            "§2.46e HallucinationGuard: spectral_novelty=%.3f > %.2f (Betriebsart=%s) → %s, Wert_penalty=%.1f",
            spectral_novelty,
            _effective_rollback_threshold,
            mode,
            "rollback" if requires_rollback else "penalty_only",
            score_penalty,
        )
    elif spectral_novelty > _get_adaptive_penalty_threshold():
        score_penalty = 0.3
        logger.debug(
            "§2.46e HallucinationGuard: spectral_novelty=%.3f > %.2f → Wert_penalty=%.1f",
            spectral_novelty,
            _get_adaptive_penalty_threshold(),
            score_penalty,
        )

    meta["spectral_novelty"] = spectral_novelty
    meta["requires_rollback"] = requires_rollback
    meta["score_penalty"] = score_penalty
    meta["mode"] = mode

    return HallucinationCheckResult(
        spectral_novelty=spectral_novelty,
        requires_rollback=requires_rollback,
        score_penalty=score_penalty,
        harmonic_ceiling_violation=harmonic_ceiling_violation,
        metadata=meta,
    )


def _compute_spectral_novelty_dsp(
    pre: npt.NDArray[np.float32],
    post: npt.NDArray[np.float32],
    sr: int,
) -> tuple[float, dict]:
    """DSP fallback for spectral_novelty when primary module is unavailable.

    Uses STFT-based energy delta: fraction of energy in bins that grew
    more than 5% (> 1.05×) after the additive phase.
    """
    try:
        from scipy import signal as _sp_signal  # pylint: disable=import-outside-toplevel

        n_fft = min(2048, len(pre), len(post))
        if n_fft < 4:
            return 0.0, {"error": "audio_too_short_dsp"}

        _, _, Pxx_pre = _sp_signal.spectrogram(pre, fs=sr, nperseg=n_fft)
        _, _, Pxx_post = _sp_signal.spectrogram(post, fs=sr, nperseg=n_fft)

        E_pre = np.mean(Pxx_pre, axis=1)
        E_post = np.mean(Pxx_post, axis=1)

        novel_mask = E_post > E_pre * 1.05
        E_novel = float(np.sum(E_post[novel_mask]))
        E_total = float(np.sum(E_post)) + 1e-12
        novelty = float(np.clip(E_novel / E_total, 0.0, 1.0))
        return novelty, {"method": "dsp_fallback"}
    except Exception as exc:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        logger.debug("_berechnen_spectral_novelty_dsp fehlgeschlagen: %s", exc)
        return 0.0, {"error": str(exc), "method": "dsp_fallback_failed"}
