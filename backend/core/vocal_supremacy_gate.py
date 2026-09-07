"""
§0p Vokal-Supremacy-Gate — Aurik 10

Zweck: Harte Gate-Instanz, die vor jeder vokal-relevanten Phase prüft, ob
die 6-dimensionale Gesangsqualität (Formant-Integrität, HNR, Vibrato-Tiefe,
Atem-Natürlichkeit, Sibilanz-Erhalt, Stimmwärme) erhalten bleibt.

Rollt zurück oder reduziert Phasenstärke bei Δ < −10 oder Einzelkriterien-Verletzung.

Implementiert nach Spec 01 §1.10 VocalQualityGate (v10.0.0-Phantom).

Usage:
    from backend.core.vocal_supremacy_gate import VocalSupremacyGate

    gate = VocalSupremacyGate()
    decision = gate.evaluate(pre_audio, post_audio, sr=48000)
    if decision.rollback:
        # Phase skippen oder Stärke halbieren
        ...
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import cast

import numpy as np

# ── Imports der bestehenden Guards ────────────────────────────────────────
from backend.core.dsp.hnr_guard import check_hnr_delta, compute_hnr
from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db, resolve_jnd_tolerance_db
from backend.core.dsp.vibrato_guard import (
    VIBRATO_MAX_REDUCTION_PCT,
    VibratoDepthResult,
    check_vibrato_depth_preservation,
)

logger = logging.getLogger(__name__)


# ── Ergebnis-Datenstruktur ───────────────────────────────────────────────


@dataclass
class VocalSupremacyDecision:
    """Entscheidung des Vokal-Supremacy-Gates.

    Attributes:
        rollback: True wenn Phase zurückgerollt werden soll.
        strength_scalar: Multiplikator für Phasenstärke [0, 1].
        formant_ok: Formant-Integrität (±2 dB JND-Toleranz).
        hnr_ok: HNR-Delta ≤ +3 dB (keine Over-Cleaning).
        vibrato_ok: Vibrato-Tiefe-Reduktion ≤ 10 %.
        breath_ok: Atem-Natürlichkeit erhalten.
        sibilance_ok: Sibilanz-Erhalt (S/H-Bereich 4–8 kHz).
        warmth_ok: Stimmwärme (Bark 3–6, 300–2500 Hz) erhalten.
        composite_score: Gesamtscore [0, 1] — alle Kriterien gewichtet.
    """

    rollback: bool
    strength_scalar: float
    formant_ok: bool
    hnr_ok: bool
    vibrato_ok: bool
    breath_ok: bool
    sibilance_ok: bool
    warmth_ok: bool
    composite_score: float


# ── Gewichtung der 6 Dimensionen (psychoakustisch kalibriert) ────────────

_WEIGHT_FORMANT = 0.25  # F1–F3 sind Vokal-Identität
_WEIGHT_HNR = 0.15  # Rauigkeit ist natürlich, aber weniger kritisch
_WEIGHT_VIBRATO = 0.15  # Vibrato = Lebendigkeit
_WEIGHT_BREATH = 0.10  # Atem-Natürlichkeit
_WEIGHT_SIBILANCE = 0.15  # Sibilanz-Erhalt (Konsonanten)
_WEIGHT_WARMTH = 0.20  # Stimmwärme (Bark 3–6)


def _measure_sibilance_preservation(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
) -> bool:
    """Prüft Sibilanz-Erhalt im S/H-Bereich (4–8 kHz).

    §0p: De-Esser darf Sibilanten nicht vollständig entfernen — das würde
    Konsonanten-Intelligibilität zerstören.

    Returns:
        True wenn Energie-Verhältnis in 4–8 kHz ≥ 0.7 (≤ 3 dB Verlust).
    """
    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)

    # Mono
    if pre.ndim == 2:
        pre_mono = np.mean(pre, axis=0).astype(np.float64)
    else:
        pre_mono = pre.astype(np.float64)

    if post.ndim == 2:
        post_mono = np.mean(post, axis=0).astype(np.float64)
    else:
        post_mono = post.astype(np.float64)

    n = min(len(pre_mono), len(post_mono))
    pre_seg = pre_mono[:n]
    post_seg = post_mono[:n]

    # FFT-basierte Energie-Messung in 4–8 kHz Band
    n_fft = max(4096, int(sr * 0.5))  # mind. 0.5 s
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # S/H-Bereich: 4–8 kHz
    sibilance_mask = (freqs >= 4000.0) & (freqs <= 8000.0)

    if not np.any(sibilance_mask):
        return True  # Band nicht erreichbar → kein Check nötig

    pre_buf = pre_seg[:n_fft] if len(pre_seg) >= n_fft else np.pad(pre_seg, (0, n_fft - len(pre_seg)), mode="edge")
    post_buf = post_seg[:n_fft] if len(post_seg) >= n_fft else np.pad(post_seg, (0, n_fft - len(post_seg)), mode="edge")

    spec_pre = np.abs(np.fft.rfft(pre_buf)) ** 2
    spec_post = np.abs(np.fft.rfft(post_buf)) ** 2

    e_pre = float(np.mean(spec_pre[sibilance_mask]))
    e_post = float(np.mean(spec_post[sibilance_mask]))

    if e_pre < 1e-20:
        return True  # Keine Sibilanz im Original → kein Check nötig

    ratio = e_post / (e_pre + 1e-20)
    ok = ratio >= 0.7  # ≤ 3 dB Verlust erlaubt

    if not ok:
        logger.info(
            "§0p Sibilanz-Erhalt: Energie-Verhältnis %.3f < 0.7 (4–8 kHz)",
            ratio,
        )
    return ok


def _measure_warmth_preservation(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
) -> bool:
    """Prüft Stimmwärme-Erhalt (Bark 3–6 ≈ 300–2500 Hz).

    §0p: NR-Phasen dürfen die warme Grundtonregion nicht übermäßig absenken.

    Returns:
        True wenn Energie-Verhältnis in 300–2500 Hz ≥ 0.6 (≤ 4.2 dB Verlust).
    """
    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)

    if pre.ndim == 2:
        pre_mono = np.mean(pre, axis=0).astype(np.float64)
    else:
        pre_mono = pre.astype(np.float64)

    if post.ndim == 2:
        post_mono = np.mean(post, axis=0).astype(np.float64)
    else:
        post_mono = post.astype(np.float64)

    n = min(len(pre_mono), len(post_mono))
    pre_seg = pre_mono[:n]
    post_seg = post_mono[:n]

    n_fft = max(4096, int(sr * 0.5))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Warmth-Bereich: 300–2500 Hz (Bark 3–6)
    warmth_mask = (freqs >= 300.0) & (freqs <= 2500.0)

    if not np.any(warmth_mask):
        return True

    pre_buf = pre_seg[:n_fft] if len(pre_seg) >= n_fft else np.pad(pre_seg, (0, n_fft - len(pre_seg)), mode="edge")
    post_buf = post_seg[:n_fft] if len(post_seg) >= n_fft else np.pad(post_seg, (0, n_fft - len(post_seg)), mode="edge")

    spec_pre = np.abs(np.fft.rfft(pre_buf)) ** 2
    spec_post = np.abs(np.fft.rfft(post_buf)) ** 2

    e_pre = float(np.mean(spec_pre[warmth_mask]))
    e_post = float(np.mean(spec_post[warmth_mask]))

    if e_pre < 1e-20:
        return True

    ratio = e_post / (e_pre + 1e-20)
    ok = ratio >= 0.6  # ≤ 4.2 dB Verlust erlaubt

    if not ok:
        logger.info(
            "§0p Stimmwärme-Erhalt: Energie-Verhältnis %.3f < 0.6 (300–2500 Hz)",
            ratio,
        )
    return ok


def _measure_breath_naturalness(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
) -> bool:
    """Prüft Atem-Natürlichkeit (Rauschen in Stille-Lücken).

    §0p: Aggressives NR entfernt auch das natürliche Atemrauschen zwischen
    Phrasen → „klinischer" Klang.

    Returns:
        True wenn RMS-Verhältnis in leisen Segmenten ≥ 0.5 (≤ 6 dB Verlust).
    """
    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)

    if pre.ndim == 2:
        pre_mono = np.mean(pre, axis=0).astype(np.float64)
    else:
        pre_mono = pre.astype(np.float64)

    if post.ndim == 2:
        post_mono = np.mean(post, axis=0).astype(np.float64)
    else:
        post_mono = post.astype(np.float64)

    n = min(len(pre_mono), len(post_mono))
    pre_seg = pre_mono[:n]
    post_seg = post_mono[:n]

    # Gleitende Fenster (200 ms, 100 ms Hop) um leise Segmente zu finden
    win_len = int(0.2 * sr)
    hop_len = int(0.1 * sr)

    if n < win_len:
        return True

    # Finde leise Segmente (RMS < 5% des Peak-RMS)
    peak_rms = float(np.sqrt(np.mean(pre_seg**2)))
    quiet_threshold = peak_rms * 0.05

    ratios = []
    for i in range(0, n - win_len, hop_len):
        pre_frame = pre_seg[i : i + win_len]
        post_frame = post_seg[i : i + win_len]
        pre_rms = float(np.sqrt(np.mean(pre_frame**2)))

        if pre_rms < quiet_threshold:
            # Leises Segment — prüfe ob NR zu viel entfernt hat
            post_rms = float(np.sqrt(np.mean(post_frame**2)))
            ratio = post_rms / (pre_rms + 1e-10)
            ratios.append(ratio)

    if not ratios:
        return True  # Keine leisen Segmente gefunden → kein Check nötig

    median_ratio = float(np.median(ratios))
    ok = median_ratio >= 0.5  # ≤ 6 dB Verlust in leisen Segmenten erlaubt

    if not ok:
        logger.info(
            "§0p Atem-Natürlichkeit: RMS-Median-Verhältnis %.3f < 0.5 (leise Segmente)",
            median_ratio,
        )
    return ok


# ── Haupt-Gate-Klasse ────────────────────────────────────────────────────


class VocalSupremacyGate:
    """§0p Vokal-Supremacy-Gate — harte Gate-Instanz.

    Aggregiert Formant-, HNR-, Vibrato-, Atem-, Sibilanz- und Wärme-Guards
    zu einer einzigen Entscheidung mit gewichtetem Composite-Score.

    Invarianten:
        - Bei rollback=True MUSS die Phase übersprungen oder auf 50% Stärke reduziert werden.
        - strength_scalar ∈ [0, 1] — nur Dämpfung, kein Boost (§1.4b).
        - composite_score < 0.6 → Rollback empfohlen.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def evaluate(
        self,
        pre_audio: np.ndarray,
        post_audio: np.ndarray,
        sr: int,
    ) -> VocalSupremacyDecision:
        """Bewertet die vokal-relevante Qualität vor/nach einer Phase.

        Args:
            pre_audio: Audio VOR der Phase (float32).
            post_audio: Audio NACH der Phase.
            sr: Sample-Rate (48000).

        Returns:
            VocalSupremacyDecision mit Rollback-Empfehlung und Stärke-Scalar.
        """
        with self._lock:
            # NaN/Inf-Schutz (§0a)
            pre = np.nan_to_num(pre_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            post = np.nan_to_num(post_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # ── 1. Formant-Integrität (±JND-Toleranz pro Formant) ───────
            formant_rollback, max_shift_db = check_formant_shift_db(pre, post, sr, threshold_db=2.0)
            formant_ok = not formant_rollback

            # ── 2. HNR-Guard (ΔHNR ≤ +3 dB) ─────────────────────────────
            hnr_diag = check_hnr_delta(pre, post, sr)
            hnr_ok = not bool(hnr_diag.get("over_cleaned", False))

            # ── 3. Vibrato-Tiefe-Erhalt (≤ 10% Reduktion) ───────────────
            vibrato_result: VibratoDepthResult = check_vibrato_depth_preservation(pre, post, sr)
            vibrato_ok = vibrato_result.ok

            # ── 4. Atem-Natürlichkeit ────────────────────────────────────
            breath_ok = _measure_breath_naturalness(pre, post, sr)

            # ── 5. Sibilanz-Erhalt (4–8 kHz) ────────────────────────────
            sibilance_ok = _measure_sibilance_preservation(pre, post, sr)

            # ── 6. Stimmwärme (Bark 3–6) ────────────────────────────────
            warmth_ok = _measure_warmth_preservation(pre, post, sr)

            # ── Composite-Score (gewichtete Summe) ───────────────────────
            score = 0.0
            if formant_ok:
                score += _WEIGHT_FORMANT * max(0.0, 1.0 - max_shift_db / 6.0)
            else:
                score -= _WEIGHT_FORMANT * 0.5  # Penalty bei Formant-Shift

            if hnr_ok:
                score += _WEIGHT_HNR
            else:
                delta = float(cast(float, hnr_diag.get("delta_hnr", 0.0)))
                score += _WEIGHT_HNR * max(0.0, 1.0 - delta / 6.0)

            if vibrato_ok:
                score += _WEIGHT_VIBRATO
            else:
                reduction = vibrato_result.depth_reduction_pct
                score += _WEIGHT_VIBRATO * max(0.0, 1.0 - reduction / 20.0)

            if breath_ok:
                score += _WEIGHT_BREATH
            if sibilance_ok:
                score += _WEIGHT_SIBILANCE
            if warmth_ok:
                score += _WEIGHT_WARMTH

            composite = float(np.clip(score, 0.0, 1.0))

            # ── Rollback-Entscheidung ───────────────────────────────────
            rollback = formant_rollback or not hnr_ok or not vibrato_ok or composite < 0.6

            # Stärke-Scalar: proportional zum Composite-Score
            strength_scalar = float(np.clip(composite, 0.0, 1.0))
            if rollback:
                strength_scalar *= 0.5  # §0p: Rollback → 50% Stärke

            decision = VocalSupremacyDecision(
                rollback=rollback,
                strength_scalar=strength_scalar,
                formant_ok=formant_ok,
                hnr_ok=hnr_ok,
                vibrato_ok=vibrato_ok,
                breath_ok=breath_ok,
                sibilance_ok=sibilance_ok,
                warmth_ok=warmth_ok,
                composite_score=composite,
            )

            if rollback:
                logger.info(
                    "§0p Vokal-Supremacy-Gate: ROLLBACK empfohlen — score=%.3f, formant=%s, hnr=%s, vibrato=%s",
                    composite,
                    "OK" if formant_ok else f"FAIL ({max_shift_db:.1f} dB)",
                    "OK" if hnr_ok else f"FAIL (Δ{float(cast(float, hnr_diag.get('delta_hnr', 0.0))):.1f} dB)",
                    "OK" if vibrato_ok else f"FAIL ({vibrato_result.depth_reduction_pct:.1f}%)",
                )

            return decision


# ── Thread-safe Singleton ────────────────────────────────────────────────

_gate_instance: VocalSupremacyGate | None = None
_gate_lock = threading.Lock()


def get_vocal_supremacy_gate() -> VocalSupremacyGate:
    """Singleton-Zugriff auf das Vokal-Supremacy-Gate."""
    global _gate_instance  # pylint: disable=global-statement
    if _gate_instance is None:
        with _gate_lock:
            if _gate_instance is None:
                _gate_instance = VocalSupremacyGate()
    return _gate_instance
