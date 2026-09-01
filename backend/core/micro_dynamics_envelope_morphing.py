"""
backend/core/micro_dynamics_envelope_morphing.py
Aurik 10.0.0 -- Spec §2.30: MicroDynamicsEnvelopeMorphing (MDEM)

Stellt origales Mikro-Dynamik-Profil im restaurierten Signal wieder her.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


def _lufs_frame(frame: np.ndarray) -> float:
    """ITU-R BS.1770: momentane LUFS eines Audio-Frames (vereinfacht)."""
    if frame.ndim == 2:
        # §v10.x Layout-sicher: Kanal-Achse statt Achse 0 ((N, C) → 1, (C, N) → 0).
        frame = frame.mean(axis=1) if frame.shape[1] <= 2 else frame.mean(axis=0)
    rms = math.sqrt(max(1e-15, float(np.mean(frame.astype(np.float64) ** 2))))
    lufs = 20.0 * math.log10(rms + 1e-15)
    return lufs


def _savgol_smooth(arr: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Vereinfachte Savitzky-Golay Glaettung (boxcar wenn scipy nicht verfuegbar)."""
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(arr, window_length=window, polyorder=polyorder).astype(np.float32)  # type: ignore[no-any-return]
    except Exception:
        # Boxcar-Fallback
        half = window // 2
        out = arr.copy()
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            out[i] = np.mean(arr[lo:hi])
        return out


@dataclass
class MorphResult:
    """Ergebnis des Envelope-Morphing."""

    pearson_correlation: float
    max_gain_applied_lu: float
    retried: bool
    audio: np.ndarray


class MicroDynamicsEnvelopeMorphing:
    """Spec §2.30: Mikro-Dynamik-Profil aus Original im Restaurierten wiederherstellen.

    Algorithmus:
        1. 400-ms-LUFS-Profile beider Signale (hop 200 ms, 50 % Ueberlappung)
        2. G[k] = L_orig[k] - L_rest[k], geclippt auf ±MAX_GAIN_LU (mode-adaptive)
        3. Savitzky-Golay-Glaettung
        4. Frame-weise lineare Gain-Interpolation
        5. True-Peak 1 dBTP nach Morphing kontrollen
    """

    MAX_GAIN_LU: float = 6.0  # Studio 2026 mode; Restoration nutzt 4.0 (siehe morph())
    FRAME_SIZE_SAMPLES: int = 19200  # 400 ms @ 48000 Hz
    HOP_SIZE_SAMPLES: int = 9600  # 200 ms
    PEARSON_TARGET: float = 0.93
    MIN_LEVEL_LUFS: float = -60.0
    TRUE_PEAK_LIMIT: float = 0.98  # ~-1.0 dBTP linear

    def compute_lufs_profile(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """400-ms-momentane LUFS-Kurve, float32 [n_frames]."""
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if arr.ndim == 2:
            # §v10.x Layout-sicher: Kanal-Achse statt Achse 0 (Befund 2026-08-22).
            arr = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)

        n = len(arr)
        hop = self.HOP_SIZE_SAMPLES
        fsize = self.FRAME_SIZE_SAMPLES
        frames = max(1, (n - fsize) // hop + 1)
        profile = np.zeros(frames, dtype=np.float32)
        for i in range(frames):
            start = i * hop
            end = start + fsize
            frame = arr[start : min(end, n)]
            profile[i] = _lufs_frame(frame)
        return profile  # type: ignore[no-any-return]

    def morph(
        self,
        restored: np.ndarray,
        original: np.ndarray,
        sr: int = 48000,
        mode: str = "restoration",
        phoneme_timeline=None,
        frisson_zones=None,
    ) -> np.ndarray:
        """Morphed restauriertes Signal auf Original-Mikrodynamik. NaN/Inf-sicher.

        §2.30 v10.0.0 Mode-basierte MAX_GAIN_LU-Kalibrierung:
        - Restoration: 4.0 dB (bewahrt emotionale Spitzen, aber konservativ für degradierte Quellen)
        - Studio 2026: 6.0 dB (maximale Dynamik-Restauration für hochwertige Eingaben)

        §2.36a phoneme_timeline: stressed vowel frames erhalten +0.5 dB extra max_gain
        (bis zum Klassen-Maximum MAX_GAIN_LU), um Vokal-Intensität zu bewahren.
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        max_gain = 4.0 if mode == "restoration" else 6.0  # §2.30 v10.0.0 Psychoakustische Kalibrierung

        # §2.36a: Vorberechnung stressed-vowel frame mask (frame index → extra gain headroom)
        _stressed_frames: set[int] = set()
        if phoneme_timeline is not None:
            try:
                _sv_segs = phoneme_timeline.stressed_vowel_segments()
                for _sv_seg in _sv_segs:
                    _f_start = int(_sv_seg.start_s * sr / self.HOP_SIZE_SAMPLES)
                    _f_end = int(_sv_seg.end_s * sr / self.HOP_SIZE_SAMPLES) + 1
                    for _fi in range(max(0, _f_start), _f_end):
                        _stressed_frames.add(_fi)
            except Exception as _sv_exc:
                logger.debug("MDEM stressed_vowel_segments fehlgeschlagen: %s", _sv_exc)

        # §Frisson: pre-compute frame set where downward gain is capped at -1.0 LU
        # (Blood & Zatorre 2001: expectation-violation peaks must not be attenuated away).
        # frisson_zones is a list of objects with .start_s and .end_s attributes.
        _frisson_frame_set: set[int] = set()
        if frisson_zones:
            try:
                _fhop_s = self.HOP_SIZE_SAMPLES / float(sr)
                for _fz in frisson_zones:
                    _fz_start = float(getattr(_fz, "start_s", 0.0))
                    _fz_end = float(getattr(_fz, "end_s", 0.0))
                    _fi_start = max(0, int(_fz_start / _fhop_s))
                    _fi_end = int(_fz_end / _fhop_s) + 1
                    for _fi in range(_fi_start, _fi_end):
                        _frisson_frame_set.add(_fi)
                if _frisson_frame_set:
                    logger.debug("MDEM §Frisson: %d Frames in Schutzzone", len(_frisson_frame_set))
            except Exception as _friz_exc:
                logger.debug("MDEM Frisson-Frame-Set fehlgeschlagen (nicht blockierend): %s", _friz_exc)
                _frisson_frame_set = set()

        res = np.nan_to_num(np.asarray(restored, dtype=np.float32))
        orig = np.nan_to_num(np.asarray(original, dtype=np.float32))

        # §v10.14.1 Defensive-Length-Normalization:
        # OLA/STFT-Rounding, TDP-Crossfade und Chunk-Grenzen können
        # restored und original um wenige Samples (352 hier) divergieren
        # lassen. Abschneiden auf gemeinsame Mindestlänge verhindert
        # Broadcast-Errors in der Gain-Envelope-Application.
        _r_n = res.shape[-1] if res.ndim >= 1 else 0
        _o_n = orig.shape[-1] if orig.ndim >= 1 else 0
        if _r_n != _o_n and _r_n > 0 and _o_n > 0:
            _common_len = min(_r_n, _o_n)
            if res.ndim == 2:
                res = res[..., :_common_len]
            else:
                res = res[:_common_len]
            if orig.ndim == 2:
                orig = orig[..., :_common_len]
            else:
                orig = orig[:_common_len]

        # §Bug-Fix: interne Längen-Absicherung — schlägt fehl wenn UV3-Alignment nicht greift
        # (z.B. bei TDP-OLA-Crossfade, STFT-Rounding, phase_09 AR-Interpolation).
        # Gilt für (channels, samples) = (2, N) UND (N, 2) Format.
        _res_len = res.shape[1] if (res.ndim == 2 and res.shape[0] <= 8) else (res.shape[0] if res.ndim >= 1 else 0)
        _orig_len = (
            orig.shape[1] if (orig.ndim == 2 and orig.shape[0] <= 8) else (orig.shape[0] if orig.ndim >= 1 else 0)
        )
        if _res_len != _orig_len and _res_len > 0 and _orig_len > 0:
            _min_len = min(_res_len, _orig_len)
            if res.ndim == 2 and res.shape[0] <= 8:
                res = res[:, :_min_len]
            elif res.ndim == 2:
                res = res[:_min_len]
            else:
                res = res[:_min_len]
            if orig.ndim == 2 and orig.shape[0] <= 8:
                orig = orig[:, :_min_len]
            elif orig.ndim == 2:
                orig = orig[:_min_len]
            else:
                orig = orig[:_min_len]
            logger.debug(
                "MDEM internal length alignment: res=%d orig=%d → min=%d samples",
                _res_len,
                _orig_len,
                _min_len,
            )

        is_stereo = res.ndim == 2
        res_channel_first = bool(is_stereo and res.shape[0] <= 8)
        orig_channel_first = bool(orig.ndim == 2 and orig.shape[0] <= 8)
        if is_stereo:
            res_mono = res.mean(axis=0) if res_channel_first else res.mean(axis=1)
            if orig.ndim == 2:
                orig_mono = orig.mean(axis=0) if orig_channel_first else orig.mean(axis=1)
            else:
                orig_mono = orig
        else:
            res_mono = res
            orig_mono = orig if orig.ndim == 1 else (orig.mean(axis=0) if orig_channel_first else orig.mean(axis=1))

        # §v10.x Degenerate-Guard (Befund 2026-08-22): „cannot reshape array of
        # size 2 into shape (1,480)“ — ein 2-Sample-Signal (Layout-Mix-Fehler
        # upstream) darf nicht in die Frame-Reshapes laufen. Morphing ist dann
        # sinnlos; Eingabe unverändert zurückgeben (Non-Blocking, §V6 (copilot-instructions.md) mit Log).
        if res_mono.size < self.FRAME_SIZE_SAMPLES or orig_mono.size < self.FRAME_SIZE_SAMPLES:
            logger.warning(
                "MDEM: degenerierte Eingabe (res=%d, orig=%d Samples) — Morphing übersprungen, Eingabe unverändert",
                int(res_mono.size),
                int(orig_mono.size),
            )
            return cast(np.ndarray, res)

        L_orig = self.compute_lufs_profile(orig_mono, sr)
        L_rest = self.compute_lufs_profile(res_mono, sr)

        # Gain-Profil berechnen
        n_frames = min(len(L_orig), len(L_rest))
        G = np.zeros(n_frames, dtype=np.float32)

        # §2.30b Adaptiver Quiet-Zone-Schwellwert (identisch zu correct_arc):
        # Vinyl/Shellac-Trägerrauschen liegt bei −35 bis −25 dBFS — der feste −36-dBFS-Schwellwert
        # lässt diese Frames durch, weil −35 dBFS > −36 dBFS.
        # Lösung: 5th-Percentile der 400ms-LUFS-Profile (Proxy für Rauschboden) + 8 dB Margin,
        # begrenzt auf [−36, −18] dBFS. Bei vollständig entrauschtem Material (p5 ≈ −65 dBFS)
        # bleibt der Schwellwert bei −36 dBFS. Bei Vinyl-Rauschen (p5 ≈ −35 dBFS) → −27 dBFS.
        #
        # §v10.51 HARD_FLOOR_GUARD: ZUSÄTZLICH zum adaptiven Schwellwert ein absoluter
        # −36 dBFS Hard-Floor. Verhindert positive Gains in echten Stille-Zonen auch dann,
        # wenn der adaptive Schwellwert durch Vinyl-Rauschboden auf > −36 dBFS angehoben wurde.
        _HARD_QUIET_FLOOR_DBFS: float = -36.0
        _p5_L = float(np.percentile(L_rest[:n_frames], 5)) if n_frames > 2 else -36.0
        _FADEOUT_QUIET_LUFS = float(np.clip(_p5_L + 8.0, -36.0, -18.0))
        # Moderate quiet zone: 6 dB above quiet zone, min -30 dBFS
        _MODERATE_QUIET_LUFS = float(max(-30.0, _FADEOUT_QUIET_LUFS + 6.0))

        # §0a Original-Music-Floor: frames below median(L_orig) − 18 dB are carrier noise,
        # not music. Positive MDEM gain on these frames boosts denoised carrier noise back
        # to its original level, causing Pegelexplosion in intro/outro/fadeout.
        # Median-18 dB is robust: pop music median ≈ −12 dBFS → floor = −30 dBFS;
        # quiet folk median ≈ −25 dBFS → floor = −43 dBFS (won't block soft music).
        _lo_active_mdem = L_orig[:n_frames][L_orig[:n_frames] > -60.0]
        if len(_lo_active_mdem) > 4:
            _lo_median_mdem = float(np.median(_lo_active_mdem))
            _ORIG_MUSIC_FLOOR = float(np.clip(_lo_median_mdem - 18.0, -48.0, -20.0))
        else:
            _ORIG_MUSIC_FLOOR = -42.0

        for k in range(n_frames):
            lo = L_orig[k]
            lr = L_rest[k]
            # §2.36a: +0.5 dB headroom for stressed-vowel frames (capped at MAX_GAIN_LU)
            _frame_max = min(max_gain + 0.5, self.MAX_GAIN_LU) if k in _stressed_frames else max_gain
            if not (math.isfinite(lo) and math.isfinite(lr)):
                G[k] = 0.0
                continue
            if lo < self.MIN_LEVEL_LUFS:
                G[k] = 0.0  # Original silent: no adjustment
                continue
            # §0a Guard 0: Original frame at carrier noise floor — no positive boost.
            # If lo is below the estimated music floor of the original, this is a
            # carrier-noise frame (vinyl/tape/shellac surface noise), not music.
            # Applying positive gain would re-amplify removed carrier noise → Pegelexplosion.
            if lo < _ORIG_MUSIC_FLOOR:
                G[k] = float(np.clip(lo - lr, -_frame_max, 0.0))  # attenuation only
                continue
            if lr < self.MIN_LEVEL_LUFS:
                # Restored is near-silent (denoising removed noise floor).
                # Never apply positive boost here — that would re-amplify the
                # cleaned noise floor, inflate integrated LUFS and force
                # phase_40 to attenuate the musical content (regression).
                G[k] = float(np.clip(lo - lr, -_frame_max, 0.0))
                continue
            # §2.30b Guard 1: Adaptive quiet zone — no positive boost.
            # _FADEOUT_QUIET_LUFS is adaptive: p5(L_rest) + 8 dB, capped at [−36, −18] dBFS.
            # This catches vinyl carrier noise at −35 dBFS which bypassed the fixed −36 dBFS.
            # §v10.51 HARD_FLOOR_GUARD: Additional absolute −36 dBFS check — blocks positive
            # gain even when adaptive threshold has been raised by vinyl noise floor.
            if lr < _FADEOUT_QUIET_LUFS or lr < _HARD_QUIET_FLOOR_DBFS:
                # Restored is in the quiet zone → no positive boost allowed
                G[k] = float(np.clip(lo - lr, -_frame_max, 0.0))
                continue
            # §2.30b Guard 2: Moderate quiet zone combined with large orig−rest gap.
            if lr < _MODERATE_QUIET_LUFS and (lo - lr) > _frame_max:
                G[k] = 0.0
                continue
            # §2.30b Guard 3 — Any-level noise-removal guard (lr >= _MODERATE_QUIET_LUFS):
            # diff > 2× max_gain is acoustically impossible for genuine 400ms dynamics.
            if (lo - lr) > 2.0 * _frame_max:
                G[k] = 0.0
                continue
            # §Frisson: in detected high-potential moments, cap downward correction
            # at -1.0 LU so the restored peak is never attenuated more than 1 LU.
            _atten_floor = float(-1.0 if k in _frisson_frame_set else -_frame_max)
            G[k] = np.clip(lo - lr, _atten_floor, _frame_max)

        # Glaettung
        G_smooth = _savgol_smooth(G)

        # §Frisson post-smooth guard: Savitzky-Golay may redistribute attenuation
        # back into frisson frames from their neighbours. Re-apply the floor.
        if _frisson_frame_set:
            for _fk in _frisson_frame_set:
                if 0 <= _fk < len(G_smooth) and G_smooth[_fk] < -1.0:
                    G_smooth[_fk] = -1.0

        # §2.30 Post-Smoothing Quiet-Zone-Clamp: SG-Glaettung kann den Boost aus
        # Musik-Frames in angrenzende Fadeout-Frames verschleppen (window=7 → 1.4 s).
        # Nach der Glaettung alle Frames nochmals auf kein-positiver-Boost prüfen.
        # Verwendet denselben adaptiven _FADEOUT_QUIET_LUFS-Schwellwert wie oben.
        for k in range(n_frames):
            if G_smooth[k] > 0.0:
                if L_orig[k] < _ORIG_MUSIC_FLOOR:
                    G_smooth[k] = 0.0  # Post-SG: original noise floor — no boost
                elif L_rest[k] < _FADEOUT_QUIET_LUFS or L_rest[k] < _HARD_QUIET_FLOOR_DBFS:
                    G_smooth[k] = 0.0  # Adaptive quiet zone + hard floor: no boost
                elif L_rest[k] < _MODERATE_QUIET_LUFS and (L_orig[k] - L_rest[k]) > max_gain:
                    G_smooth[k] = 0.0  # Moderate quiet zone post-SG: diff too large
                elif (L_orig[k] - L_rest[k]) > 2.0 * max_gain:
                    # §2.30b Guard 3 post-smoothing: SG may redistribute Guard-3 gain
                    # back into frames that passed Guard 3 pre-smoothing.
                    G_smooth[k] = 0.0

        # Gain-Anwendung: frame-weise lineare Interpolation
        hop = self.HOP_SIZE_SAMPLES
        fsize = self.FRAME_SIZE_SAMPLES
        n = len(res_mono)

        gain_envelope = np.ones(n, dtype=np.float32)
        for k in range(n_frames):
            start = k * hop
            end = start + fsize
            linear_gain = 10.0 ** (G_smooth[k] / 20.0)
            ce = min(end, n)
            if start < n:
                # Lineare Interpolation zu naechstem Frame
                nxt_gain = 10.0 ** (G_smooth[k + 1] / 20.0) if k + 1 < n_frames else linear_gain
                ramp = np.linspace(linear_gain, nxt_gain, ce - start, dtype=np.float32)
                gain_envelope[start:ce] = ramp

        # §2.30b Per-Sample Quiet-Zone Guard (Step 5 — Drei-Stufen-Invariante):
        # linspace interpolation between a positive music frame and a zeroed quiet frame
        # creates a positive ramp in the transition region. Clamp gain to 1.0 (0 dB)
        # wherever res_mono is below the adaptive threshold — prevents Pegelexplosion at
        # music/fade-out boundary. Uses same adaptive threshold as correct_arc():
        # p5(L_rest) + 8 dB, capped at [−36, −18] dBFS. Catches vinyl carrier noise
        # at −35 dBFS which bypassed the fixed −36 dBFS threshold.
        _quiet_thresh_ps = float(10.0 ** (_FADEOUT_QUIET_LUFS / 20.0))  # adaptive dBFS linear
        # §v10.51: Hard floor at −36 dBFS — blocks positive gain when adaptive threshold is higher
        _HARD_QUIET_THRESH_PS = float(10.0 ** (_HARD_QUIET_FLOOR_DBFS / 20.0))
        _ps_frame = 480  # 10 ms @ 48 kHz for fine-grained fade-out detection
        _n_full_ps = n // _ps_frame
        if _n_full_ps > 0:
            _segs_ps = res_mono[: _n_full_ps * _ps_frame].reshape(_n_full_ps, _ps_frame)
            _rms_ps = np.sqrt(np.mean(_segs_ps**2, axis=1) + 1e-12)
            # §v10.51: Both adaptive + hard floor thresholds
            _quiet_mask = np.repeat((_rms_ps < _quiet_thresh_ps) | (_rms_ps < _HARD_QUIET_THRESH_PS), _ps_frame)
            if _n_full_ps * _ps_frame < n:
                _tail_rms_ps2 = float(np.sqrt(np.mean(res_mono[_n_full_ps * _ps_frame :] ** 2) + 1e-12))
                _quiet_mask = np.concatenate(
                    [
                        _quiet_mask,
                        np.full(
                            n - _n_full_ps * _ps_frame,
                            _tail_rms_ps2 < _quiet_thresh_ps or _tail_rms_ps2 < _HARD_QUIET_THRESH_PS,
                            dtype=bool,
                        ),
                    ]
                )
        else:
            # n < 480 (sehr kurzes Segment): einfachen Gesamt-RMS als Quiet-Indikator nutzen
            _tail_rms_short = float(np.sqrt(np.mean(res_mono**2) + 1e-12))
            _quiet_mask = np.full(
                n,
                _tail_rms_short < _quiet_thresh_ps or _tail_rms_short < _HARD_QUIET_THRESH_PS,
                dtype=bool,
            )
        _qmask = _quiet_mask[:n] & (gain_envelope[:n] > 1.0)
        if np.any(_qmask):
            gain_envelope[:n] = np.where(_qmask, np.float32(1.0), gain_envelope[:n])

        # §2.30 tail-gap fix (§9.10.119 improved): Samples nach dem letzten
        # Frame-Ende erhalten eine sanfte Rückkehr statt eines harten Wert-Kopierens.
        # §9.11.2: Tail-Gain darf in stillen Regionen NICHT auf 1.0 (Unity) hoch-
        # interpolieren — das hebt den Rauschboden an und erzeugt hörbaren Pegelsprung
        # am Musikende.  Stattdessen: Wenn das Tail-Audio leise ist (Stille/Rauschboden),
        # bleibt der letzte Gain erhalten oder wird sanft auf _last_gain gehalten.
        _last_covered = min((n_frames - 1) * hop + fsize, n)
        if _last_covered < n and _last_covered > 0:
            _tail_len = n - _last_covered
            _last_gain = gain_envelope[_last_covered - 1]
            # Check if tail audio is silence/noise floor (RMS < -50 dBFS ≈ 0.003)
            _tail_audio = res_mono[_last_covered:] if _last_covered < len(res_mono) else np.zeros(1)
            _tail_rms = float(np.sqrt(np.mean(_tail_audio**2) + 1e-12))
            _tail_rms_dbfs = 20.0 * math.log10(_tail_rms + 1e-12)
            # §2.30 Tail-Guard: covers both digital silence (<-50 dBFS) and
            # vinyl/tape fade-out noise floor (<-36 dBFS). In both cases do NOT
            # boost — clamp gain to min(last_gain, 1.0) to avoid Pegelexplosion.
            # §v10.51: Uses HARD floor −36 dBFS, not just adaptive threshold.
            _tail_in_quiet_zone = _tail_rms_dbfs < -36.0  # §v10.51 hard floor
            if _tail_in_quiet_zone:
                # Tail is silence or noise floor: cap gain at unity (no boost)
                _safe_gain = min(_last_gain, 1.0)
                gain_envelope[_last_covered:] = _safe_gain
            else:
                # Tail has musical content: smooth interpolation to unity,
                # but never exceed 1.0 when last_gain > 1.0 (avoids fade-out boost)
                _interp = np.linspace(_last_gain, 1.0, _tail_len, dtype=np.float32)
                if _last_gain > 1.0:
                    _interp = np.minimum(_interp, 1.0)
                gain_envelope[_last_covered:] = _interp

        # Auf Stereo/Mono anwenden
        if is_stereo:
            out = np.zeros_like(res)
            if res_channel_first:
                for ch in range(res.shape[0]):
                    n_samp = min(res.shape[1], len(gain_envelope))
                    out[ch, :n_samp] = res[ch, :n_samp] * gain_envelope[:n_samp]
                    if n_samp < res.shape[1]:
                        out[ch, n_samp:] = res[ch, n_samp:]
            else:
                n_samp = min(res.shape[0], len(gain_envelope))
                # §v10.14: Robustheits-Clamp — bei OLA/STFT-Rounding kann
                # gain_envelope wenige Samples länger sein als res (Bug: Broadcast-Error).
                _ge = gain_envelope[:n_samp]
                out[:n_samp, :] = res[:n_samp, :] * _ge[:n_samp, np.newaxis]
                if n_samp < res.shape[0]:
                    out[n_samp:, :] = res[n_samp:, :]
        else:
            n_samp = min(n, len(gain_envelope))
            out = res.copy()
            # §v10.14: Robustheits-Clamp — bei OLA/STFT-Rounding kann
            # gain_envelope wenige Samples länger sein als res.
            _ge_mono = gain_envelope[:n_samp]
            _r_mono = res[:n_samp]
            out[:n_samp] = _r_mono * _ge_mono

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        # §v10.51 Post-MDEM Quiet-Zone Peak Guard (Layer 2 — Defense-in-Depth):
        # After all gain envelope logic, scan the output for isolated sample-level
        # spikes in quiet regions (< −36 dBFS RMS). These spikes originate from
        # Savitzky-Golay edge effects + linspace interpolation at fade-out boundaries
        # and are too short (1–3 samples) to be caught by the 10ms per-sample guard.
        #
        # Algorithm: For each 10ms segment where RMS < −36 dBFS, clamp any sample
        # exceeding local_RMS + 12 dB back to local_RMS + 6 dB. The 12 dB threshold
        # discriminates genuine transient attacks (>12 dB above noise floor) from
        # interpolation artifacts (3–9 dB above noise floor in a silent context).
        _peak_guard_frame = 480  # 10 ms @ 48 kHz
        _n_peak_segs = n // _peak_guard_frame
        if _n_peak_segs > 0:
            _peak_mono = out if out.ndim == 1 else out.mean(axis=-1) if not res_channel_first else out.mean(axis=0)
            _peak_segs = _peak_mono[: _n_peak_segs * _peak_guard_frame].reshape(_n_peak_segs, _peak_guard_frame)
            _peak_rms = np.sqrt(np.mean(_peak_segs**2, axis=1) + 1e-12)
            _peak_quiet = _peak_rms < float(10.0 ** (-36.0 / 20.0))  # −36 dBFS linear
            if np.any(_peak_quiet):
                _quiet_indices = np.where(_peak_quiet)[0]
                for _qi in _quiet_indices:
                    _seg_start = _qi * _peak_guard_frame
                    _seg_end = min(_seg_start + _peak_guard_frame, n)
                    _seg_rms = float(_peak_rms[_qi])
                    _clamp_ceiling = _seg_rms * 2.0  # RMS + 6 dB
                    if is_stereo:
                        if res_channel_first:
                            for _ch in range(out.shape[0]):
                                _ch_seg = out[_ch, _seg_start:_seg_end]
                                _exceed = np.abs(_ch_seg) > _clamp_ceiling
                                if np.any(_exceed):
                                    _ch_seg[_exceed] = np.sign(_ch_seg[_exceed]) * _clamp_ceiling
                                    out[_ch, _seg_start:_seg_end] = _ch_seg
                        else:
                            for _ch in range(out.shape[1]):
                                _ch_seg = out[_seg_start:_seg_end, _ch]
                                _exceed = np.abs(_ch_seg) > _clamp_ceiling
                                if np.any(_exceed):
                                    _ch_seg[_exceed] = np.sign(_ch_seg[_exceed]) * _clamp_ceiling
                                    out[_seg_start:_seg_end, _ch] = _ch_seg
                    else:
                        _seg = out[_seg_start:_seg_end]
                        _exceed = np.abs(_seg) > _clamp_ceiling
                        if np.any(_exceed):
                            _seg[_exceed] = np.sign(_seg[_exceed]) * _clamp_ceiling
                            out[_seg_start:_seg_end] = _seg
                _n_clamped = int(np.sum(_peak_quiet))
                logger.debug("§v10.51 Post-MDEM Peak Guard: %d stille Segmente geclampt", _n_clamped)

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = np.clip(out, -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT)

        # Pearson-Korrelation pruefen, ggf. Retry
        out_mono = (
            out.mean(axis=1) if (out.ndim == 2 and out.shape[1] <= 2) else (out.mean(axis=0) if out.ndim == 2 else out)
        )
        r = self._pearson(orig_mono[: len(out_mono)], out_mono[: len(orig_mono)])

        if r < self.PEARSON_TARGET and max_gain < self.MAX_GAIN_LU:
            _retry_gain = min(max(max_gain * 1.5, 4.0), self.MAX_GAIN_LU)  # §2.54: mindestens 4.0, max MAX_GAIN_LU
            logger.debug("MDEM Wiederholung mit erweitertem MAX_GAIN=%.1f dB (aktuell r=%.3f)", _retry_gain, r)
            # Einmaliger Retry mit erweitertem Gain — kein weiterer rekursiver Aufruf
            out2 = self._morph_internal(res_mono, orig_mono, max_gain=_retry_gain)
            out2 = np.nan_to_num(out2, nan=0.0, posinf=1.0, neginf=-1.0)
            out2 = np.clip(out2, -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT)
            if is_stereo and res.ndim == 2:
                # Gain-Kurve aus Mono-Ergebnis zurückrechnen und auf alle Kanäle anwenden.
                # §2.30b: gain2 kann > 1.0 sein — Per-Sample Quiet-Zone Guard auf JEDEM Kanal.
                _safe_res_mono = np.where(np.abs(res_mono) > 1e-8, res_mono, 1.0)
                gain2 = out2 / _safe_res_mono
                _quiet_thresh_retry = float(10.0 ** (-36.0 / 20.0))
                out_retry = np.zeros_like(res)
                if res_channel_first:
                    for ch in range(res.shape[0]):
                        n_s = min(res.shape[1], len(gain2))
                        _ch_out = res[ch, :n_s] * gain2[:n_s]
                        # §2.30b Per-Channel Quiet-Zone Guard: supprimiere positiven Gain wo
                        # der jeweilige Kanal in der Quiet-Zone liegt (<-36 dBFS).
                        _ch_rms_frames = max(1, n_s // 480)
                        _ch_seg = res[ch, : _ch_rms_frames * 480].reshape(_ch_rms_frames, 480)
                        _ch_rms = np.sqrt(np.mean(_ch_seg**2, axis=1) + 1e-12)
                        _ch_quiet = np.repeat(_ch_rms < _quiet_thresh_retry, 480)[:n_s]
                        _ch_gain_pos = gain2[:n_s] > 1.0
                        _ch_out[_ch_quiet & _ch_gain_pos] = res[ch, :n_s][_ch_quiet & _ch_gain_pos]
                        out_retry[ch, :n_s] = _ch_out
                        out_retry[ch, n_s:] = res[ch, n_s:]
                else:
                    n_s = min(res.shape[0], len(gain2))
                    _ch_out = res[:n_s, :] * gain2[:n_s, np.newaxis]
                    _ch_rms_frames = max(1, n_s // 480)
                    _ch_seg = res[: _ch_rms_frames * 480, :].reshape(_ch_rms_frames, 480, res.shape[1])
                    _ch_rms = np.sqrt(np.mean(_ch_seg**2, axis=1) + 1e-12)
                    _ch_quiet = np.repeat(_ch_rms < _quiet_thresh_retry, 480, axis=0)[:n_s, :]
                    _ch_gain_pos = gain2[:n_s, np.newaxis] > 1.0
                    _ch_out[_ch_quiet & _ch_gain_pos] = res[:n_s, :][_ch_quiet & _ch_gain_pos]
                    out_retry[:n_s, :] = _ch_out
                    out_retry[n_s:, :] = res[n_s:, :]
                final = np.clip(np.nan_to_num(out_retry), -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT).astype(
                    np.float32
                )
            else:
                final = out2.astype(np.float32)
            # §8.2 Observability: log final pearson after retry (universal guarantee ≥ 0.92)
            _final_mono = (
                final.mean(axis=1)
                if (final.ndim == 2 and final.shape[1] <= 2)
                else (final.mean(axis=0) if final.ndim == 2 else final)
            )
            r_final = self._pearson(orig_mono[: len(_final_mono)], _final_mono[: len(orig_mono)])
            if r_final < 0.92:
                # Pearson still < 0.92 after retry — signal structure limits correlation
                # (e.g. heavily compressed 1970s vinyl).  Only warn if retry made no progress.
                _delta = r_final - r
                _pearson_improved = _delta > 0.001  # < 0.001 = konvergiert, kein weiterer Retry sinnvoll
                _log_fn = logger.debug if _pearson_improved else logger.warning
                _log_fn(
                    "§8.2 MDEM Mikro-Dynamik-Guarantee VERFEHLT: pearson=%.4f < 0.92 "
                    "(max_gain=%.1f dB, mode=retry, Δ=%.4f) — Rohausgabe nicht unterdrückt",
                    r_final,
                    max_gain,
                    _delta,
                )
            else:
                logger.info("§8.2 MDEM Micro-Dynamics pearson=%.4f ≥ 0.92 (Wiederholung, r_before=%.4f)", r_final, r)
            return final  # type: ignore[no-any-return]

        # §8.2 Observability: log final pearson (universal guarantee ≥ 0.92)
        if r < 0.92:
            logger.warning(
                "§8.2 MDEM Mikro-Dynamik-Guarantee VERFEHLT: pearson=%.4f < 0.92 "
                "(max_gain=%.1f dB) — Rohausgabe nicht unterdrückt",
                r,
                max_gain,
            )
        else:
            logger.debug("§8.2 MDEM Micro-Dynamics pearson=%.4f ≥ 0.92 (max_gain=%.1f dB)", r, max_gain)

        return out.astype(np.float32)  # type: ignore[no-any-return]

    def _morph_internal(
        self,
        res_mono: np.ndarray,
        orig_mono: np.ndarray,
        max_gain: float = 4.0,
    ) -> np.ndarray:
        """Interne Gain-Envelope-Berechnung und -Anwendung auf Mono-Signale (kein Retry).

        Default max_gain = 4.0 LU (Restoration mode, §2.30 v10.0.0).
        VERBOTEN: einheitliches 3.0 LU — nicht spec-konform (Fix X2 §2.30).
        """
        L_orig = self.compute_lufs_profile(orig_mono)
        L_rest = self.compute_lufs_profile(res_mono)
        n_frames = min(len(L_orig), len(L_rest))
        G = np.zeros(n_frames, dtype=np.float32)
        # §2.30b Adaptiver Quiet-Zone-Schwellwert (identisch zu morph() und correct_arc()):
        # Vinyl/Shellac-Rauschen bei −35 dBFS liegt über dem festen −36-dBFS-Schwellwert.
        # p5(L_rest) + 8 dB, capped [−36, −18] dBFS.
        _p5_L_mi = float(np.percentile(L_rest[:n_frames], 5)) if n_frames > 2 else -36.0
        _QUIET_LUFS = float(np.clip(_p5_L_mi + 8.0, -36.0, -18.0))
        for k in range(n_frames):
            lo, lr = L_orig[k], L_rest[k]
            if not (math.isfinite(lo) and math.isfinite(lr)) or lo < self.MIN_LEVEL_LUFS:
                G[k] = 0.0
            elif lr < self.MIN_LEVEL_LUFS:
                # Restored near-silent: suppress positive boost (see morph() §2.30 guard)
                G[k] = float(np.clip(lo - lr, -max_gain, 0.0))
            elif lr < _QUIET_LUFS or lr < -36.0:
                # §2.30 adaptive quiet-zone guard + §v10.51 hard floor: no positive boost
                # No positive gain allowed in quiet zone.
                G[k] = float(np.clip(lo - lr, -max_gain, 0.0))
            elif (lo - lr) > 2.0 * max_gain:
                # §2.30b Guard 3 (_morph_internal): any-level noise-removal guard.
                G[k] = 0.0
            else:
                G[k] = float(np.clip(lo - lr, -max_gain, max_gain))
        G_smooth = _savgol_smooth(G)
        # §2.30 post-smoothing quiet-zone clamp — uses adaptive _QUIET_LUFS.
        for k in range(n_frames):
            if G_smooth[k] > 0.0:
                if L_rest[k] < _QUIET_LUFS or L_rest[k] < -36.0:
                    G_smooth[k] = 0.0  # §v10.51: adaptive + hard floor
                elif (L_orig[k] - L_rest[k]) > 2.0 * max_gain:
                    # §2.30b Guard 3 post-smoothing (_morph_internal)
                    G_smooth[k] = 0.0
        hop = self.HOP_SIZE_SAMPLES
        n = len(res_mono)
        gain_envelope = np.ones(n, dtype=np.float32)
        for k in range(n_frames):
            start = k * hop
            ce = min(start + self.FRAME_SIZE_SAMPLES, n)
            if start >= n:
                break
            lg = float(10.0 ** (G_smooth[k] / 20.0))
            nxt = float(10.0 ** (G_smooth[k + 1] / 20.0)) if k + 1 < n_frames else lg
            gain_envelope[start:ce] = np.linspace(lg, nxt, ce - start, dtype=np.float32)

        # §2.30b Per-Sample Quiet-Zone Guard (Step 5 — Drei-Stufen-Invariante):
        # mirrors the identical guard in morph() — uses adaptive _QUIET_LUFS threshold.
        _quiet_thresh_mi = float(10.0 ** (_QUIET_LUFS / 20.0))
        _ps_fr_mi = 480  # 10 ms @ 48 kHz
        _n_full_mi = n // _ps_fr_mi
        if _n_full_mi > 0:
            _segs_mi = res_mono[: _n_full_mi * _ps_fr_mi].reshape(_n_full_mi, _ps_fr_mi)
            _rms_mi = np.sqrt(np.mean(_segs_mi**2, axis=1) + 1e-12)
            _quiet_mask_mi = np.repeat(_rms_mi < _quiet_thresh_mi, _ps_fr_mi)
            if _n_full_mi * _ps_fr_mi < n:
                _tail_rms_mi = float(np.sqrt(np.mean(res_mono[_n_full_mi * _ps_fr_mi :] ** 2) + 1e-12))
                _quiet_mask_mi = np.concatenate(
                    [
                        _quiet_mask_mi,
                        np.full(
                            n - _n_full_mi * _ps_fr_mi,
                            _tail_rms_mi < _quiet_thresh_mi,
                            dtype=bool,
                        ),
                    ]
                )
        else:
            # n < 480: Gesamt-RMS als Quiet-Indikator
            _quiet_mask_mi = np.full(
                n,
                float(np.sqrt(np.mean(res_mono**2) + 1e-12)) < _quiet_thresh_mi,
                dtype=bool,
            )
        _qmask_mi = _quiet_mask_mi[:n] & (gain_envelope[:n] > 1.0)
        if np.any(_qmask_mi):
            gain_envelope[:n] = np.where(_qmask_mi, np.float32(1.0), gain_envelope[:n])

        # §2.30 tail-gap fix: Samples jenseits des letzten Frame-Endes fortsetzen.
        _last_covered = min((n_frames - 1) * hop + self.FRAME_SIZE_SAMPLES, n)
        if _last_covered < n and _last_covered > 0:
            _tail_audio = res_mono[_last_covered:]
            _tail_rms = float(np.sqrt(np.mean(_tail_audio**2) + 1e-12))
            _tail_rms_dbfs = 20.0 * math.log10(_tail_rms + 1e-12)
            _tail_in_quiet_zone = _tail_rms_dbfs < _QUIET_LUFS or _tail_rms_dbfs < -36.0
            if _tail_in_quiet_zone:
                # §2.30 tail quiet-zone guard: restored tail is vinyl/tape noise floor.
                # Copy last gain but cap at 1.0 (no positive boost on tail).
                gain_envelope[_last_covered:] = min(float(gain_envelope[_last_covered - 1]), 1.0)
            else:
                gain_envelope[_last_covered:] = gain_envelope[_last_covered - 1]

        out = res_mono.copy()
        n_s = min(n, len(gain_envelope))
        out[:n_s] = res_mono[:n_s] * gain_envelope[:n_s]
        return out

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        if n < 2:
            return 1.0
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        am, bm = a.mean(), b.mean()
        num = float(np.mean((a - am) * (b - bm)))
        den = max(1e-15, float(np.std(a) * np.std(b)))
        val = num / den
        return float(np.clip(val, -1.0, 1.0)) if math.isfinite(val) else 0.0


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: MicroDynamicsEnvelopeMorphing | None = None
_lock = threading.Lock()


def get_mdem() -> MicroDynamicsEnvelopeMorphing:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MicroDynamicsEnvelopeMorphing()
    return _instance


def morph_micro_dynamics(
    restored: np.ndarray,
    original: np.ndarray,
    sr: int = 48000,
    mode: str = "restoration",
) -> np.ndarray:
    """Convenience-Wrapper."""
    return get_mdem().morph(restored, original, sr, mode)


# ── Modul-Level-Konstanten (für direkten Import durch Tests und Consumer-Code) ──
# Spiegeln die Klassen-Attribute wider, sodass `from backend.core.micro_dynamics_envelope_morphing
# import FRAME_SIZE_SAMPLES` ohne Klassen-Instantiierung funktioniert.
FRAME_SIZE_SAMPLES: int = MicroDynamicsEnvelopeMorphing.FRAME_SIZE_SAMPLES
HOP_SIZE_SAMPLES: int = MicroDynamicsEnvelopeMorphing.HOP_SIZE_SAMPLES
MAX_GAIN_LU: float = MicroDynamicsEnvelopeMorphing.MAX_GAIN_LU
MIN_LEVEL_LUFS: float = MicroDynamicsEnvelopeMorphing.MIN_LEVEL_LUFS
PEARSON_TARGET: float = MicroDynamicsEnvelopeMorphing.PEARSON_TARGET


def compute_lufs_profile(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Convenience-Wrapper für MicroDynamicsEnvelopeMorphing.compute_lufs_profile()."""
    return get_mdem().compute_lufs_profile(audio, sr)


__all__ = [
    # Modul-Level-Konstanten:
    "FRAME_SIZE_SAMPLES",
    "HOP_SIZE_SAMPLES",
    "MAX_GAIN_LU",
    "MIN_LEVEL_LUFS",
    "PEARSON_TARGET",
    "MicroDynamicsEnvelopeMorphing",
    "MorphResult",
    "compute_lufs_profile",
    "get_mdem",
    "morph_micro_dynamics",
]
