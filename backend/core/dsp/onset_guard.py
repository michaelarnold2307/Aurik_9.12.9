"""§ATI (V26) Onset-Protection-Guard.

Schützt Transient-Onset-Fenster (0–20 ms nach Transient) vor übermäßiger
Einflussnahme durch NR/EQ-Phasen. Puls-Onset-Frames werden mit max. 1.5 dB
Energiedifferenz begrenzt.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.onset_guard import apply_onset_protection_mask
    result.audio = apply_onset_protection_mask(pre, result.audio, onset_mask, max_delta_db=1.5)
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import librosa  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore[assignment]

from backend.core.residuum_masking import estimate_delta_masking_jnd_db

logger = logging.getLogger(__name__)

_ONSET_WINDOW_MS = 20.0  # Schutzfenster nach Transient


def apply_onset_protection_mask(
    pre: np.ndarray,
    post: np.ndarray,
    onset_mask: np.ndarray | None,
    max_delta_db: float = 1.5,
) -> np.ndarray:
    """Begrenzt Energie-Änderungen in Onset-Fenstern auf max_delta_db.

    Für jeden Transient-Frame (onset_mask == True) wird ein 20 ms Schutzfenster
    aufgespannt. Wenn |Δ_rms| > max_delta_db → Dry-Wet-Blend in Richtung Input.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase (same shape as pre).
        onset_mask: Boolean-Array der Länge N (sample-genau). True = Onset-Bereich.
            Wenn None → Berechnung direkt aus pre via librosa (lazy fallback).
        max_delta_db: Maximale erlaubte Energiedifferenz in dB. Standard: 1.5 dB.

    Returns:
        Geschütztes Audio (Float32, geclippt auf [-1.0, 1.0]).
    """
    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if pre.shape != post.shape:
            return post

        # §2.51 Stereo-Axis-Invariante: channels-last (N, 2) → channels-first (2, N)
        _was_transposed = False
        if pre.ndim == 2 and pre.shape[1] == 2 and pre.shape[0] > 2:
            pre = pre.T
            post = post.T
            _was_transposed = True

        n = pre.shape[-1] if pre.ndim == 2 else len(pre)

        # Onset-Maske bestimmen
        if onset_mask is None or len(onset_mask) == 0:
            try:
                if librosa is None:
                    return post

                pre_mono = pre.mean(axis=0).astype(np.float32) if pre.ndim == 2 else pre
                # sr aus Pufferlänge kann nicht inferiert werden → hart auf 48000 setzen
                # (Guard wird nur aus UV3 mit sr=48000 aufgerufen)
                _sr = 48000
                onset_frames = librosa.onset.onset_detect(y=pre_mono, sr=_sr, hop_length=256, backtrack=True)  # type: ignore[attr-defined]
                # Sample-genaue Maske erzeugen
                onset_mask_arr = np.zeros(n, dtype=bool)
                window_samples = int(_ONSET_WINDOW_MS / 1000.0 * _sr)
                for frame in onset_frames:
                    start = int(frame * 256)
                    end = min(n, start + window_samples)
                    if start < n:
                        onset_mask_arr[start:end] = True
            except Exception as e:
                logger.warning("onset_guard.py::anwenden_onset_protection_mask Ersatzpfad: %s", e)
                return post  # Ohne Onset-Daten: kein Eingriff
        else:
            onset_mask_arr = np.asarray(onset_mask, dtype=bool)
            if len(onset_mask_arr) != n:
                return post

        if not onset_mask_arr.any():
            return post

        # §P1-3 (Hörordnung Ebene 2): Effektive Toleranz = max(fest, lokale
        # Maskierungs-JND). Transient-Onsets sind laut → maskieren den
        # Phasen-Delta stark; dann begrenzen wir erst bei größeren
        # Abweichungen (weniger unnötige Dry-Wet-Blends).
        _jnd = estimate_delta_masking_jnd_db(pre, post, 48000)
        _eff_max_db = max(float(max_delta_db), float(_jnd.jnd_db))
        if _jnd.jnd_db > 0.1:
            logger.info(
                "§ATI Onset-Guard: §P1-3 Maskierungs-JND=%.2f dB → Toleranz %.2f dB (fest %.1f dB)",
                _jnd.jnd_db,
                _eff_max_db,
                max_delta_db,
            )

        # Schutz in Onset-Fenstern anwenden
        max_ratio = float(10.0 ** (_eff_max_db / 20.0))  # Linearer Schwellwert

        if post.ndim == 2:
            result = post.copy()
            for c in range(post.shape[0]):
                pre_c = pre[c]
                post_c = post[c]
                # RMS pre/post in Onset-Fenstern messen und begrenzen
                # Sample-weise: zu aufwändig → Frame-weise (512 Samples)
                frame_len = 512
                n_frames = n // frame_len
                for fi in range(n_frames):
                    start = fi * frame_len
                    end = start + frame_len
                    if not onset_mask_arr[start:end].any():
                        continue
                    pre_rms = float(np.sqrt(np.mean(pre_c[start:end] ** 2) + 1e-12))
                    post_rms = float(np.sqrt(np.mean(post_c[start:end] ** 2) + 1e-12))
                    if pre_rms < 1e-9:
                        continue
                    ratio = post_rms / (pre_rms + 1e-12)
                    if ratio > max_ratio or ratio < 1.0 / max_ratio:
                        # Wet-Faktor so setzen, dass ratio → 1.0 tendiert
                        wet = float(np.clip(1.0 - abs(ratio - 1.0) / (abs(ratio - 1.0) + 0.5), 0.1, 0.9))
                        result[c, start:end] = wet * post_c[start:end] + (1.0 - wet) * pre_c[start:end]
        else:
            result = post.copy()
            frame_len = 512
            n_frames = n // frame_len
            for fi in range(n_frames):
                start = fi * frame_len
                end = start + frame_len
                if not onset_mask_arr[start:end].any():
                    continue
                pre_rms = float(np.sqrt(np.mean(pre[start:end] ** 2) + 1e-12))
                post_rms = float(np.sqrt(np.mean(result[start:end] ** 2) + 1e-12))
                if pre_rms < 1e-9:
                    continue
                ratio = post_rms / (pre_rms + 1e-12)
                if ratio > max_ratio or ratio < 1.0 / max_ratio:
                    wet = float(np.clip(1.0 - abs(ratio - 1.0) / (abs(ratio - 1.0) + 0.5), 0.1, 0.9))
                    result[start:end] = wet * post[start:end] + (1.0 - wet) * pre[start:end]

        result = np.clip(result, -1.0, 1.0).astype(np.float32)
        if _was_transposed:
            result = result.T
        return result  # type: ignore[no-any-return]

    except Exception as exc:
        logger.debug("anwenden_onset_protection_mask nicht blockierend: %s", exc)
        return post
