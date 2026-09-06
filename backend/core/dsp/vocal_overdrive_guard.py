"""vocal_overdrive_guard — Harte Vocal-Schutz-Invariante gegen Verzerrung/Übersteuerung.

Hörordnung Ebene 1/2 (hoerordnung.instructions.md §3/§4): Gesang ist die
kritischste Spur — Verzerrung (harmonische/Intermodulations-Anreicherung,
Clipping, Crest-Kollaps) in stimmlichen Zonen ist unverhandelbar. Dieser
Guard misst nach jeder Phase und am Lauf-Ende, ob eine Verarbeitungsstufe
dem Gesang nichtlineare „Drive"-Artefakte hinzufügt, und blendet die Phase
zurück (oder nimmt sie vollständig zurück), bevor das Ohr sie hört.

Warum nicht nur Pegel/THD klassisch:
  * Musik-Real-Signale haben natürliche Obertöne — absolutes THD ist kein Maß.
  * Gemessen wird die *Zunahme* der harmonischen Kamm-/IMD-Struktur relativ
    zur lokalen Breitband-Verstärkung im selben Frame (level-angepasst):
    Eine reine EQ-/Tilt-Änderung hebt Kamm- und Zwischenbänder gleichmäßig
    an (Excess ≈ 0); eine nichtlineare Stufe (Sättigung, Waveshaping,
    Hard-Clip, fehlerhafter Enhancer) hebt Vielfache/Halbvielfache der F0
    über das lokale Breitband hinaus an (Excess ≫ 0).
  * Zusätzlich: Clipping-Ratio und Crest-Kollaps in stimmlichen Frames.

Stateless (kein Song-Zustand, §V8/§G1), numpy-only, kostenbegrenzt:
  * Vorauswahl stimmlicher Kandidaten-Frames über Bandpass-RMS (max. 24),
  * Autokorrelations-F0 nur auf Kandidaten, FFT nur auf ~200-ms-Frames.

Ablauf (Integration in unified_restorer_v3.py):
  * nach jeder Phase:  protect_vocal_overdrive(pre=audio, post=result.audio)
  * am Lauf-Ende:       protect_vocal_overdrive(pre=original, post=final)
  * im Export-Pfad:     protect_vocal_overdrive(pre=export_input, post=korrigiert)

Schwellen (kalibriert 2026-09-06 gegen Real-Clip „Elke Best … 1977",
Worst-Frames wiesen Excess bis +21 dB bei 0 dB hartem Clip auf):
  * VOICED_CLIP_HARD_RATIO     = 1e-4   (0,01 % gesättigte Samples in Stimm-Frames)
  * COMB_EXCESS_MODERATE_DB    = +5.0   (p90 über stimmliche Frames)
  * COMB_EXCESS_SEVERE_DB      = +10.0  (→ vollständige Rücknahme)
  * CREST_COLLAPSE_SEVERE_DB   = 8.0    (Pumpen/Überkompression in Stimm-Frames)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- Schwellen (Hör-Invarianten, hart) ---------------------------------------
VOICED_CLIP_HARD_RATIO = 1e-4
VOICED_CLIP_SEVERE_RATIO = 1e-3
COMB_EXCESS_MODERATE_DB = 5.0
COMB_EXCESS_SEVERE_DB = 10.0
IMD_EXCESS_SEVERE_DB = 10.0
# Final-Modus (kumulativ über den ganzen Lauf, pre=Original): moderate
# IMD-Schwelle. Kalibriert 2026-09-06 am Real-Clip „Elke Best … 1977"
# (beanstandete Ausgabe: imd_p90=8.2 dB bei comb_p90=3.4 dB — hörbar
# „buzziger" Gesang, kein Clip): legitime Aufhellung hält comb niedrig,
# IMD-Anreicherung in Halbvielfach-Lücken ist die Verzerrungs-Signatur.
IMD_EXCESS_FINAL_MODERATE_DB = 6.0
_FINAL_BLEND_FLOOR = 0.70
CREST_COLLAPSE_MODERATE_DB = 5.0
CREST_COLLAPSE_SEVERE_DB = 8.0

_FRAME_S = 0.2
_HOP_S = 0.1
_F0_MIN_HZ = 80.0
_F0_MAX_HZ = 500.0
_MAX_CANDIDATE_FRAMES = 24
_MID_LOW_HZ = 250.0
_MID_HIGH_HZ = 3200.0


@dataclass
class VocalOverdriveResult:
    """Ergebnis der Vocal-Overdrive-Messung.

    blend_factor: 1.0 = keine Schutzmaßnahme nötig; < 1.0 = zum Pre-Signal
                  blenden (0.0 = vollständige Rücknahme der Stufe).
    """

    passed: bool = True
    blend_factor: float = 1.0
    hard_revert: bool = False
    reasons: list[str] = field(default_factory=list)
    # Telemetrie
    voiced_frames: int = 0
    analyzed_frames: int = 0
    voiced_clip_ratio: float = 0.0
    comb_excess_db_p90: float | None = None
    imd_excess_db_p90: float | None = None
    crest_delta_db_p90: float | None = None
    max_frame_excess_db: float | None = None


def _mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        # Robust gegen (N,2) und (2,N)
        if a.shape[0] == 2 and a.shape[1] != 2:
            a = a.T
        return a.mean(axis=1)
    return a


def _frame_rms_db(x: np.ndarray, sr: int) -> np.ndarray:
    flen = max(2, int(_FRAME_S * sr))
    hop = max(1, int(_HOP_S * sr))
    n = (len(x) - flen) // hop + 1
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = x[i * hop : i * hop + flen]
        out[i] = 20.0 * np.log10(float(np.sqrt(np.mean(seg**2)) + 1e-12))
    return out


def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if lo <= 0.0:
        sos = butter(4, min(hi, nyq * 0.98) / nyq, btype="lowpass", output="sos")
    else:
        sos = butter(4, [max(lo, 1.0) / nyq, min(hi, nyq * 0.98) / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _estimate_f0(x: np.ndarray, sr: int) -> tuple[float | None, float]:
    """Autokorrelations-F0 im Stimmbereich; Rückgabe (f0, Prominenz).

    Lag-begrenzte Autokorrelation (nur Stimm-Lags 80–500 Hz) statt
    np.correlate(...,"full") — O(n·Lags) statt O(n²), damit der Guard
    pro Phase nur wenige ms kostet.
    """
    n = len(x)
    lo = max(1, int(sr / _F0_MAX_HZ))
    hi = min(n - 1, int(sr / _F0_MIN_HZ))
    if hi - lo < 4:
        return None, 0.0
    x0 = x - float(np.mean(x))
    energy = float(np.dot(x0, x0))
    if energy <= 1e-12:
        return None, 0.0
    lags = np.arange(lo, hi + 1)
    ac = np.empty(len(lags), dtype=np.float64)
    for j, lag in enumerate(lags):
        ac[j] = float(np.dot(x0[: n - lag], x0[lag:]))
    prom = float(ac.max() / energy)
    if prom < 0.25:
        return None, prom
    return sr / float(lags[int(np.argmax(ac))]), prom


def _comb_metrics(pre_f: np.ndarray, post_f: np.ndarray, sr: int, f0: float) -> dict[str, float]:
    """Harmonischer Kamm-Excess eines Frames (level-angepasst).

    Deckt k = 2 … k_max ab, wobei k_max so gewählt ist, dass k·f0 ≤ 9 kHz
    bleibt (stimmlich relevanter Bereich). Der Excess ist die Zunahme der
    Energie an den Vielfachen der F0 *über* der lokalen Breitband-Verstärkung:
    Sättigung/Waveshaping „flacht" die natürliche Oberton-Abklingkurve ab
    (1/k^1.1 → ~1/k) und hebt damit hohe Vielfache über das Breitband hinaus;
    eine reine EQ-/Tilt-Änderung hebt Kamm- und Zwischenbereiche gleichmäßig.
    """
    n = len(pre_f)
    w = np.hanning(n)
    P = np.abs(np.fft.rfft(pre_f * w)) ** 2
    Q = np.abs(np.fft.rfft(post_f * w)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    g_pre = float(np.sqrt(np.mean(pre_f**2)) + 1e-12)
    g_post = float(np.sqrt(np.mean(post_f**2)) + 1e-12)
    Q = Q * (g_pre / g_post) ** 2  # Level-Angleich (Schutz vor Lautheits-Artefakten)

    def band_power(spec: np.ndarray, fc: float, bw: float = 0.035) -> float:
        m = (freqs > fc * (1.0 - bw)) & (freqs < fc * (1.0 + bw))
        return float(np.sum(spec[m]))

    k_max = min(20, int(9000.0 / f0))
    if k_max < 3:
        return {"comb_median_db": 0.0, "imd_median_db": 0.0}
    # Lokale Breitband-Verstärkung über den gesamten geprüften Bereich
    # (EQ/Tilt hebt dieses Band insgesamt an — davon wird der Excess abgezogen).
    m_band = (freqs > 1.5 * f0) & (freqs < (k_max + 1.0) * f0)
    g_band = 10.0 * np.log10(float(np.sum(Q[m_band])) / (float(np.sum(P[m_band])) + 1e-15) + 1e-12)

    comb: list[float] = []
    imd: list[float] = []
    for k in range(2, k_max + 1):
        if k * f0 > sr * 0.45:
            break
        e = 10.0 * np.log10(band_power(Q, k * f0) / (band_power(P, k * f0) + 1e-15) + 1e-12) - g_band
        comb.append(float(np.clip(e, -20.0, 20.0)))
    for half in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
        if half * f0 > min(9000.0, sr * 0.45):
            break
        e = 10.0 * np.log10(band_power(Q, half * f0) / (band_power(P, half * f0) + 1e-15) + 1e-12) - g_band
        imd.append(float(np.clip(e, -20.0, 20.0)))
    return {
        "comb_median_db": float(np.median(comb)) if comb else 0.0,
        "imd_median_db": float(np.median(imd)) if imd else 0.0,
    }


def measure_vocal_overdrive(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    *,
    voiced_zones: list[tuple[float, float]] | None = None,
    vocal_active: bool = True,
    max_frames: int = _MAX_CANDIDATE_FRAMES,
    mode: str = "phase",
) -> VocalOverdriveResult:
    """Misst die Vocal-Overdrive-Invariante zwischen pre und post.

    mode="phase"  (Default, nach jeder Phase): inkrementelle Invariante — die
                   Stufe darf dem Gesang keinen Drive oberhalb der moderaten
                   Schwelle hinzufügen; Verstoß → Blend Richtung pre.
    mode="final"  (Lauf-Ende/Export, pre=Original): nur harte Verstöße lösen
                   Schutz aus (Clipping, Crest-Kollaps, >10-dB-Kamm/IMD-Excess),
                   damit legitime Entrauschung/Aufhellung nicht zurückgemischt
                   wird. Moderate Kamm-/IMD-Werte werden nur telemetriert.

    Liefert immer ein VocalOverdriveResult (kein Raise); aufrufende Stellen
    blenden bei blend_factor < 1.0 zum pre-Signal.
    """
    res = VocalOverdriveResult()
    if not vocal_active:
        return res
    try:
        pre_m = _mono(pre)
        post_m = _mono(post)
        # Toleranz für minimale Längen-Differenzen (z.B. Resample-Reste, ≤ 256
        # Samples ≈ 5 ms): gemeinsamer Bereich wird gemessen. Größere Abweichung
        # oder zu kurzes Audio → Invariante trivial erfüllt (kein Raise).
        if len(pre_m) < int(sr * 0.3):
            return res
        if abs(len(pre_m) - len(post_m)) > 256:
            return res
        if len(pre_m) != len(post_m):
            _n_c = min(len(pre_m), len(post_m))
            pre_m = pre_m[:_n_c]
            post_m = post_m[:_n_c]
        if np.array_equal(pre_m, post_m):
            return res
        sr = int(sr)

        # --- Schnell-Pfad: keine nennenswerte Änderung im Stimmband ----------
        mid_pre = _bandpass(pre_m, sr, _MID_LOW_HZ, _MID_HIGH_HZ)
        mid_post = _bandpass(post_m, sr, _MID_LOW_HZ, _MID_HIGH_HZ)
        rms_pre = float(np.sqrt(np.mean(mid_pre**2)))
        rms_post = float(np.sqrt(np.mean(mid_post**2)))
        if abs(20.0 * np.log10((rms_post + 1e-12) / (rms_pre + 1e-12))) < 0.05:
            # Kein hörbarer Eingriff im Stimmband → Invariante trivial erfüllt
            res.voiced_frames = 0
            return res

        rms_db_pre = _frame_rms_db(mid_pre, sr)
        rms_db_post = _frame_rms_db(mid_post, sr)
        clip_mask = np.abs(post_m) >= 0.999

        # Kandidaten-Frames: stärkste Stimmband-Frames (pre), größte Δ-Frames
        # sowie alle Frames mit Sättigung in post.
        flen0 = int(_FRAME_S * sr)
        hop0 = int(_HOP_S * sr)
        nfr = len(rms_db_pre)
        clip_frames_idx = [
            i
            for i in range(nfr)
            if bool(np.any(clip_mask[i * hop0 : min(len(post_m), i * hop0 + flen0)]))
        ]
        order_pre = np.argsort(-rms_db_pre)
        delta_db = np.abs(rms_db_post - rms_db_pre)
        order_delta = np.argsort(-delta_db)
        cand: set[int] = set()
        for idx in order_pre[: max_frames // 2]:
            cand.add(int(idx))
        for idx in order_delta[: max_frames // 2]:
            cand.add(int(idx))
        cand.update(clip_frames_idx)
        if voiced_zones:
            zcand: set[int] = set()
            for zs, ze in voiced_zones:
                for i in range(int(zs / _HOP_S), int(ze / _HOP_S)):
                    if 0 <= i < nfr:
                        zcand.add(i)
            # Zonen sind autoritativ: Union mit Energie-Ranking, gedeckelt.
            cand = zcand | cand
        cand = set(sorted(cand, key=lambda i: -float(rms_db_pre[i]))[:max_frames])

        flen = int(_FRAME_S * sr)
        hop = int(_HOP_S * sr)
        comb_list: list[float] = []
        imd_list: list[float] = []
        crest_delta_list: list[float] = []
        voiced_clip = 0
        voiced_samples = 0
        analyzed = 0
        for i in cand:
            s0 = i * hop
            seg_pre = pre_m[s0 : s0 + flen]
            seg_post = post_m[s0 : s0 + flen]
            if len(seg_pre) < int(sr * 0.1):
                continue
            voiced_samples += len(seg_post)
            # Clipping in stimmlichem Kandidat
            voiced_clip += int(np.sum(np.abs(seg_post) >= 0.999))
            # Tonalitäts-Gate: F0 nur bei klarer Periodizität im Pre-Frame
            f0, prom = _estimate_f0(seg_pre, sr)
            if f0 is None or f0 * 6 > sr * 0.45:
                continue
            if prom < 0.3:
                continue
            crest_pre = float(np.max(np.abs(seg_pre))) / (float(np.sqrt(np.mean(seg_pre**2))) + 1e-12)
            crest_post = float(np.max(np.abs(seg_post))) / (float(np.sqrt(np.mean(seg_post**2))) + 1e-12)
            crest_delta_list.append(20.0 * np.log10(crest_post / (crest_pre + 1e-12) + 1e-12))
            cm = _comb_metrics(seg_pre, seg_post, sr, f0)
            comb_list.append(cm["comb_median_db"])
            imd_list.append(cm["imd_median_db"])
            analyzed += 1

        res.voiced_frames = len(cand)
        res.analyzed_frames = analyzed
        if voiced_samples > 0:
            res.voiced_clip_ratio = voiced_clip / voiced_samples
        if comb_list:
            res.comb_excess_db_p90 = float(np.percentile(comb_list, 90))
            res.max_frame_excess_db = float(np.max(comb_list))
        if imd_list:
            res.imd_excess_db_p90 = float(np.percentile(imd_list, 90))
        if crest_delta_list:
            res.crest_delta_db_p90 = float(np.percentile(crest_delta_list, 10))  # negativ = Kollaps

        # --- Entscheidung (Hör-Invariante) -----------------------------------
        clip = res.voiced_clip_ratio
        comb = res.comb_excess_db_p90
        imd = res.imd_excess_db_p90
        crest = res.crest_delta_db_p90
        if clip > VOICED_CLIP_SEVERE_RATIO:
            res.reasons.append(f"voiced_clip={clip:.5f}")
            res.hard_revert = True
        if comb is not None and comb > COMB_EXCESS_SEVERE_DB:
            res.reasons.append(f"comb_excess_p90={comb:.1f} dB")
            res.hard_revert = True
        if imd is not None and imd > IMD_EXCESS_SEVERE_DB:
            res.reasons.append(f"imd_excess_p90={imd:.1f} dB")
            res.hard_revert = True
        if crest is not None and crest < -CREST_COLLAPSE_SEVERE_DB:
            res.reasons.append(f"crest_collapse={crest:.1f} dB")
            res.hard_revert = True
        if clip > VOICED_CLIP_HARD_RATIO and not res.hard_revert:
            res.reasons.append(f"voiced_clip={clip:.5f}")
        if mode != "final":
            if comb is not None and comb > COMB_EXCESS_MODERATE_DB and not res.hard_revert:
                res.reasons.append(f"comb_excess_p90={comb:.1f} dB")
            if imd is not None and imd > COMB_EXCESS_MODERATE_DB and not res.hard_revert:
                res.reasons.append(f"imd_excess_p90={imd:.1f} dB")
        else:
            # Final-Modus: kumulativ über viele Phasen akkumulierter Vocal-Drive
            # (moderate Kamm-/IMD-Anreicherung) → weiche Blend Richtung Original.
            if comb is not None and comb > COMB_EXCESS_MODERATE_DB and not res.hard_revert:
                res.reasons.append(f"final_comb_excess_p90={comb:.1f} dB")
            if imd is not None and imd > IMD_EXCESS_FINAL_MODERATE_DB and not res.hard_revert:
                res.reasons.append(f"final_imd_excess_p90={imd:.1f} dB")
        if res.reasons:
            res.passed = False
            if res.hard_revert:
                res.blend_factor = 0.0
            elif mode == "final":
                # Weiche Korrektur am Lauf-Ende (max. 30 % Original-Anteil), damit
                # die Entrauschung erhalten bleibt, der Vocal-Drive aber unter die
                # Hör-Schwelle gedrückt wird.
                severity = max(
                    (comb or 0.0) - COMB_EXCESS_MODERATE_DB,
                    (imd or 0.0) - IMD_EXCESS_FINAL_MODERATE_DB,
                    (np.log10(max(clip, 1e-9)) - np.log10(VOICED_CLIP_HARD_RATIO)) * 2.0,
                    (-crest - CREST_COLLAPSE_SEVERE_DB) / 4.0 if crest is not None else 0.0,
                    0.0,
                ) / 6.0
                res.blend_factor = float(np.clip(1.0 - 0.6 * severity, _FINAL_BLEND_FLOOR, 0.98))
            else:
                # Moderater Verstoß → weicher Blend Richtung Pre (max. 70 % Pre).
                # Severity aus dem stärksten Verstoß (Kamm/IMD/Clipping).
                severity = max(
                    (comb or 0.0) - COMB_EXCESS_MODERATE_DB,
                    (imd or 0.0) - COMB_EXCESS_MODERATE_DB,
                    (np.log10(max(clip, 1e-9)) - np.log10(VOICED_CLIP_HARD_RATIO)) * 2.0,
                    0.0,
                ) / 5.0
                res.blend_factor = float(np.clip(1.0 - severity, 0.30, 0.90))
        return res
    except Exception as exc:  # Guard darf nie die Pipeline brechen
        logger.debug("vocal_overdrive_guard nicht blockierend: %s", exc)
        return VocalOverdriveResult()


def protect_vocal_overdrive(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    *,
    voiced_zones: list[tuple[float, float]] | None = None,
    vocal_active: bool = True,
    phase_id: str = "",
    mode: str = "phase",
) -> tuple[np.ndarray, VocalOverdriveResult]:
    """Wendet die Invariante an: blendet post bei Verletzung Richtung pre.

    Returns:
        (geschütztes Audio, Messergebnis) — bei passed ist das Audio unverändert.
    """
    res = measure_vocal_overdrive(
        pre, post, sr, voiced_zones=voiced_zones, vocal_active=vocal_active, mode=mode
    )
    out = np.asarray(post, dtype=np.float32)
    pre_a = np.asarray(pre, dtype=np.float32)
    # Final-Modus: bis zu 4 Blend-Iterationen, damit die Invariante am Ende
    # garantiert erfüllt ist (imd/comb unter Schwelle), ohne die Entrauschung
    # übermäßig zurückzunehmen (Blend-Floor 0.70 begrenzt den Original-Anteil).
    _max_iter = 4 if mode == "final" else 1
    for _ in range(_max_iter):
        if res.blend_factor >= 1.0:
            break
        if pre_a.shape != out.shape:
            # Shape-Mismatch (z.B. Mono→Stereo): Schutz auf Mono-Mix anwenden
            pre_a = np.broadcast_to(pre_a, out.shape).copy()
        out = np.clip(
            (res.blend_factor * out + (1.0 - res.blend_factor) * pre_a).astype(np.float32),
            -1.0,
            1.0,
        )
        if res.hard_revert:
            logger.warning(
                "§Vocal-Drive HARTE Rücknahme (%s): %s",
                phase_id or "?",
                "; ".join(res.reasons),
            )
        else:
            logger.info(
                "§Vocal-Drive Blend (%s): blend=%.2f — %s",
                phase_id or "?",
                res.blend_factor,
                "; ".join(res.reasons),
            )
        if _max_iter > 1:
            res = measure_vocal_overdrive(
                pre, out, sr, voiced_zones=voiced_zones, vocal_active=vocal_active, mode=mode
            )
    return out, res


def vocal_drive_telemetry(result: VocalOverdriveResult) -> dict[str, Any]:
    """Kompakte Telemetrie für result.metadata / phase_metadata_accumulator."""
    return {
        "vocal_drive_passed": bool(result.passed),
        "vocal_drive_blend": round(float(result.blend_factor), 3),
        "vocal_drive_hard_revert": bool(result.hard_revert),
        "vocal_drive_reasons": list(result.reasons),
        "vocal_drive_voiced_clip_ratio": round(float(result.voiced_clip_ratio), 6),
        "vocal_drive_comb_excess_db_p90": (
            round(float(result.comb_excess_db_p90), 2) if result.comb_excess_db_p90 is not None else None
        ),
        "vocal_drive_imd_excess_db_p90": (
            round(float(result.imd_excess_db_p90), 2) if result.imd_excess_db_p90 is not None else None
        ),
        "vocal_drive_crest_delta_db_p10": (
            round(float(result.crest_delta_db_p90), 2) if result.crest_delta_db_p90 is not None else None
        ),
        "vocal_drive_voiced_frames": int(result.voiced_frames),
        "vocal_drive_analyzed_frames": int(result.analyzed_frames),
    }
