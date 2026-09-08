"""TemporalContinuityGuard — Post-Phase-Hook (§2.69, v10.0.0).

Erkennt Frame-by-Frame-Diskontinuitäten nach jeder Phase anhand des
Frame-RMS-Varianz-Verhältnisses (pre/post). Kein Veto — ausschließlich
Protokollierung in metadata["temporal_continuity"][phase_id].

Schwellwerte:
    variance_ratio > 2.5  → WARNING + ok=False
    variance_ratio > 8.0  → zusätzlich "critical" im result

Kanonischer Aufruf in _profiled_phase_call_with_delta():
    from backend.core.temporal_continuity_guard import check_temporal_continuity
    tc = check_temporal_continuity(pre=pre_phase_audio, post=audio, phase_id=phase_id, sr=sr)
    metadata.setdefault("temporal_continuity", {})[phase_id] = {
        "variance_ratio": tc.variance_ratio, "ok": tc.ok
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from backend.core.residuum_masking import estimate_delta_masking_jnd_db

logger = logging.getLogger(__name__)

# Schwellwerte (kanonisch, §2.69)
_VARIANCE_RATIO_WARN: float = 2.5
_VARIANCE_RATIO_CRITICAL: float = 8.0
_GAIN_STEP_THRESHOLD_DB: float = 1.5  # §2.69: fester Gain-Sprung-Schwellwert

# Frame-Parameter für librosa.feature.rms
_FRAME_LENGTH: int = 2048
_HOP_LENGTH: int = 512


@dataclass
class TemporalContinuityResult:
    """Ergebnis eines TemporalContinuityGuard-Checks.

    Attributes:
        ok:             True wenn variance_ratio <= 2.5 (kein Problem).
        variance_ratio: Frame-RMS-Varianz(post) / Frame-RMS-Varianz(pre).
        phase_id:       ID der geprüften Phase.
        critical:       True wenn variance_ratio > 8.0.
        gain_step_db:   Abrupter Gain-Sprung an der Phasengrenze (§2.69 v10.0.0).
                        > gain_step_threshold_db → WARNING (Mikro-Klick-Risiko). Kein Veto.
        gain_step_threshold_db: Effektive Schwelle = max(1.5 dB, lokale
                        Maskierungs-JND, §P1-3 Hörordnung Ebene 2).
    """

    ok: bool
    variance_ratio: float
    phase_id: str
    critical: bool = False
    gain_step_db: float = 0.0
    gain_step_threshold_db: float = _GAIN_STEP_THRESHOLD_DB


def check_temporal_continuity(
    pre: np.ndarray,
    post: np.ndarray,
    phase_id: str,
    sr: int,
) -> TemporalContinuityResult:
    """Misst Frame-RMS-Varianz-Verhältnis pre/post einer Phase.

    Args:
        pre:      Audio vor der Phase (float32, 1D oder 2D [channels, samples]).
        post:     Audio nach der Phase (float32, 1D oder 2D [channels, samples]).
        phase_id: Phasen-ID für Log-Ausgaben.
        sr:       Sample-Rate (Hz) — für zukünftige adaptive Schwellwerte reserviert.

    Returns:
        TemporalContinuityResult mit ok, variance_ratio, phase_id, critical.
        Bei Fehler: ok=True, variance_ratio=1.0 (neutral, kein False-Positive).
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        _ = sr  # reserviert für adaptive Schwellwerte (z.B. Mindest-Frame-Länge bei 16 kHz)
        # Mono-Downmix (Varianz-Messung auf Summe der Kanäle)
        pre_mono = np.mean(pre, axis=0) if pre.ndim == 2 else pre
        post_mono = np.mean(post, axis=0) if post.ndim == 2 else post

        pre_mono = np.nan_to_num(pre_mono.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        post_mono = np.nan_to_num(post_mono.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if len(pre_mono) < _FRAME_LENGTH or len(post_mono) < _FRAME_LENGTH:
            # Zu kurz für sinnvolle Messung
            return TemporalContinuityResult(ok=True, variance_ratio=1.0, phase_id=phase_id)

        rms_pre = librosa.feature.rms(y=pre_mono, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH)[0]
        rms_post = librosa.feature.rms(y=post_mono, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH)[0]

        var_pre = float(np.var(rms_pre))
        var_post = float(np.var(rms_post))

        if var_pre < 1e-12:
            # Stille/Konstant-Signal: Division durch Null vermeiden → neutral
            return TemporalContinuityResult(ok=True, variance_ratio=1.0, phase_id=phase_id)

        ratio = float(var_post / var_pre)
        ratio = float(np.nan_to_num(ratio, nan=1.0, posinf=_VARIANCE_RATIO_CRITICAL + 1.0))

        ok = ratio <= _VARIANCE_RATIO_WARN
        critical = ratio > _VARIANCE_RATIO_CRITICAL

        # §2.69 v10.0.0: Gain-Sprung einer Phase messen (abrupter RMS-Anstieg/-Abfall).
        # FIX v10.0.0c: Vergleich MITTLERER RMS (pre vs. post) — nicht rms_pre[-N:] vs.
        # rms_post[:N] (Lied-Ende vs. Lied-Anfang). Bei Fade-out-Material lieferte die
        # alte Logik immer ~120 dB (Near-Stille vs. Vollpegel) → systematisches false-
        # positive für JEDE Phase, doppelter Rescue-Aufruf, nie hilfreich. Korrekte
        # Messung: Gesamtpegel-Änderung durch die Phase (mean-RMS pre vs. mean-RMS post).
        pre_boundary_rms = float(np.mean(rms_pre) + 1e-12)
        post_boundary_rms = float(np.mean(rms_post) + 1e-12)
        gain_step_db = float(abs(20.0 * np.log10(post_boundary_rms / pre_boundary_rms)))
        gain_step_db = float(np.nan_to_num(gain_step_db, nan=0.0, posinf=0.0, neginf=0.0))
        # §P1-3 (Hörordnung Ebene 2): Effektive Schwelle = max(fest, lokale
        # Maskierungs-JND). Ein maskierter Gain-Sprung ist unhörbar → kein
        # Mikro-Klick-Risiko, keine Warnung, kein Rescue-Trigger.
        _jnd = estimate_delta_masking_jnd_db(pre, post, sr)
        _eff_threshold_db = max(_GAIN_STEP_THRESHOLD_DB, float(_jnd.jnd_db))
        if gain_step_db > _eff_threshold_db:
            logger.warning(
                "TemporalContinuityGuard §2.69 gain_step_db=%.2f dB > %.2f dB Verarbeitungsschritt=%s — Mikro-Klick-Risiko (§P1-3 JND=%.2f dB)",
                gain_step_db,
                _eff_threshold_db,
                phase_id,
                _jnd.jnd_db,
            )

        if critical:
            logger.warning(
                "TemporalContinuityGuard CRITICAL Verarbeitungsschritt=%s variance_Verhaeltnis=%.2f",
                phase_id,
                ratio,
            )
        elif not ok:
            logger.warning(
                "TemporalContinuityGuard WARNING Verarbeitungsschritt=%s variance_Verhaeltnis=%.2f",
                phase_id,
                ratio,
            )
        else:
            logger.debug(
                "TemporalContinuityGuard OK Verarbeitungsschritt=%s variance_Verhaeltnis=%.2f",
                phase_id,
                ratio,
            )

        return TemporalContinuityResult(
            ok=ok,
            variance_ratio=ratio,
            phase_id=phase_id,
            critical=critical,
            gain_step_db=round(gain_step_db, 3),
            gain_step_threshold_db=round(_eff_threshold_db, 3),
        )

    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("TemporalContinuityGuard nicht blockierend: Verarbeitungsschritt=%s err=%s", phase_id, exc)
        return TemporalContinuityResult(ok=True, variance_ratio=1.0, phase_id=phase_id, gain_step_db=0.0)
