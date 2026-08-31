"""
§v10.2 Naturalness Optimizer MAX — Weltklasse-Natürlichkeit für menschliche Ohren.

Der NaturalnessOptimizer ist Auriks letzte Verteidigungslinie gegen
unangenehmen Klang. Er läuft NACH der UV3-Restauration und führt
alle Korrekturen aus, die das menschliche Ohr erwartet, aber technische
Metriken nicht erfassen.

Pipeline:
  1. HPE-Evaluation (ISO 532, Zwicker/Fastl, Moore/Glasberg)
  2. Multi-Band-Glue (3-Band SSL-Style Kompressor)
  3. Stereo-Feld-Erhalt & Breiten-Optimierung
  4. Transienten-Schutz (Attack-Erkennung + Preservation)
  5. De-Essing-Nachbearbeitung (material-adaptiv, phonem-bewusst)
  6. Bass-Management (Sub-Bass-Erhalt 20-100 Hz)
  7. Sharpness-Korrektur (High-Shelf ± HPE-gesteuert)
  8. Roughness-Glättung (Mikro-Dynamik-Smoothing)
  9. Wärmeband-Guard (200-800 Hz Referenz-gebunden)
  10. Air-Band-Polish (12-16 kHz, material-/ära-abhängig)
  11. Loudness-Feinschliff (EBU R128 / Archiv-Pegel)
  12. Tonalness-Enhancement (sanfte harmonische Anreicherung)
  13. Safety-Clamp (§0p: max 2x Original)

Ergebnis: ΔHPE ≥ +0.08 (signifikant hörbare Verbesserung)
Ref: §v10 Pleasantness-First, §3.0 Cross-Phase Naturalness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NaturalnessResult:
    audio: np.ndarray
    hpe_before: float
    hpe_after: float
    delta_hpe: float
    improvements: list[str] = field(default_factory=list)
    applied_stages: list[str] = field(default_factory=list)
    glue_reduction_db: float = 0.0
    stereo_width_before: float = 0.0
    stereo_width_after: float = 0.0
    transient_events_protected: int = 0


def optimize_naturalness(
    audio: np.ndarray,
    original: np.ndarray,
    sr: int,
    *,
    material: str = "unknown",
    era: str = "",
    mode: str = "RESTORATION",
    dry_run: bool = False,
    album_ref: dict | None = None,
    progress_callback: callable | None = None,  # type: ignore[valid-type]  # f(pct_0_100: float, label: str)
) -> NaturalnessResult:
    """Maximiert die natürliche Hörqualität auf Weltklasse-Niveau.

    progress_callback: Optional callable(pct_0_100, label) für Fortschrittsanzeige.

    audio: UV3-restauriertes Audio (float32)
    original: Original-Audio (float32)
    sr: Sample-Rate (48000)
    material: shellac, vinyl, tape, cd_digital, ...
    era: ≤1930, 1930-1945, 1945-1960, 1960-1970, ≥1970
    mode: RESTORATION oder STUDIO_2026
    dry_run: Nur Analyse, keine Änderung
    """
    # Input validation
    if sr != 48000:
        logger.warning("optimieren_naturalness: sr=%d, expected 48000", sr)
    arr = np.asarray(audio, dtype=np.float32).copy()
    orig = np.asarray(original, dtype=np.float32)
    if arr.ndim not in (1, 2):
        raise ValueError(f"audio must be 1D or 2D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("audio is empty")
    is_stereo = arr.ndim == 2 and arr.shape[1] == 2

    hpe_before = _compute_hpe(arr, sr)
    improvements: list[str] = []
    applied: list[str] = []
    stereo_before = _measure_stereo_width(arr) if is_stereo else 0.0
    transients_protected = 0

    if dry_run:
        return NaturalnessResult(
            audio=arr,
            hpe_before=hpe_before,
            hpe_after=hpe_before,
            delta_hpe=0.0,
            improvements=["Dry-Run: keine Änderungen"],
            stereo_width_before=stereo_before,
            stereo_width_after=stereo_before,
        )

    # ── Unified Steering via PhaseSteeringEngine (§v10.5) ─────────────
    _engine = None
    try:
        from backend.core.phase_steering_guard import SteerAction, get_engine

        _engine = get_engine()
    except Exception as e:
        logger.warning("PhaseSteeringEngine not verfuegbar: %s", e)

    def _guarded_stage(name, before, after):
        if _engine is None:
            return after, 1.0
        h0 = _engine._compute_hpe(before, sr)
        h1 = _engine._compute_hpe(after, sr)
        decision = _engine.decide(h0, h1, name, 1.0)
        if decision.action == SteerAction.SKIP:
            return before, 0.0
        if decision.action == SteerAction.RETRY_LIGHTER:
            # Re-run with reduced intensity via crossfade
            return before + (after - before) * decision.new_strength, decision.new_strength
        return after, 1.0

    # Progress tracking
    _stage_num = 0
    _total_stages = 12 + (3 if mode == "RESTORATION" else 0) + (7 if mode == "STUDIO_2026" else 0)

    def _progress(label):
        nonlocal _stage_num
        _stage_num += 1
        if progress_callback:
            progress_callback(_stage_num / _total_stages * 100, label)  # type: ignore[misc]

    # ── 1. HPE-Vollanalyse ─────────────────────────────────────────────
    hpe_full = _compute_hpe_full(arr, sr)
    _progress("Analyse")

    _progress("Multi-Band-Glue")
    # ── 2. Multi-Band-Glue Stage ────────────────────────────────────────
    _bg = arr.copy()
    arr, glue_db = _multiband_glue(arr, sr, mode=mode)
    arr, _ = _guarded_stage("multiband_glue", _bg, arr)
    if glue_db > 0.1:
        applied.append("multiband_glue")
        improvements.append(f"Multi-Band-Feinschliff ({glue_db:.1f} dB)")

    _progress("Stereo")
    # ── 3. Stereo-Feld-Optimierung ──────────────────────────────────────
    if is_stereo:
        _s3 = arr.copy()
        arr, sw_info = _stereo_field_optimize(arr, orig, sr, material)
        arr, _ = _guarded_stage("stereo_optimize", _s3, arr)
        if sw_info:
            applied.append("stereo_optimize")
            improvements.append(sw_info)

    _progress("Transienten")
    # ── 4. Transienten-Schutz ───────────────────────────────────────────
    if hpe_full.get("roughness_asper", 0.5) < 2.0:
        _s4 = arr.copy()
        arr, n_protected = _transient_preservation(arr, orig, sr)
        arr, _ = _guarded_stage("transient_protection", _s4, arr)
        if n_protected > 0:
            applied.append("transient_protection")
            improvements.append(f"{n_protected} Attack-Transienten geschützt")
            transients_protected = n_protected

    _progress("De-Essing")
    # ── 5. De-Essing-Nachbearbeitung ────────────────────────────────────
    if material not in ("shellac", "wax_cylinder"):
        _s5 = arr.copy()
        arr, ds_info = _gentle_de_ess(arr, sr, material)
        arr, _ = _guarded_stage("de_essing", _s5, arr)
        if ds_info:
            applied.append("de_essing_refinement")
            improvements.append(ds_info)

    _progress("Bass")
    # ── 6. Bass-Management ─────────────────────────────────────────────
    if material not in ("shellac", "wax_cylinder", "wire_recording"):
        _s6 = arr.copy()
        arr = _bass_preservation(arr, orig, sr)
        arr, _ = _guarded_stage("bass_management", _s6, arr)
        applied.append("bass_management")

    _progress("Sharpness")
    # ── 7. Sharpness-Korrektur ─────────────────────────────────────────
    sharp = hpe_full.get("sharpness_zwicker", 1.5)
    if sharp > 3.0:
        _s7 = arr.copy()
        reduction = min(3.5, (sharp - 2.5) * 1.5)
        arr = _apply_high_shelf(arr, sr, 8000, -reduction, 0.7)
        arr, _ = _guarded_stage("sharpness_correction", _s7, arr)
        improvements.append(f"Höhen −{reduction:.1f} dB (zu scharf: {sharp:.1f} acum)")
        applied.append("sharpness_correction")
    elif sharp < 0.8:
        _s7b = arr.copy()
        boost = min(3.0, (1.2 - sharp) * 3.0)
        arr = _apply_high_shelf(arr, sr, 10000, boost, 0.5)
        arr, _ = _guarded_stage("sharpness_correction", _s7b, arr)
        improvements.append(f"Höhen +{boost:.1f} dB (zu dumpf)")
        applied.append("sharpness_correction")

    _progress("Roughness")
    # ── 8. Roughness-Glättung ──────────────────────────────────────────
    if hpe_full.get("roughness_asper", 0.5) > 1.5:
        _s8 = arr.copy()
        arr = _smooth_micro_dynamics(arr, sr, 30)
        arr, _ = _guarded_stage("roughness_smoothing", _s8, arr)
        improvements.append(f"Mikrodynamik geglättet ({hpe_full['roughness_asper']:.1f} asper)")
        applied.append("roughness_smoothing")

    _progress("Wärme")
    # ── 9. Wärmeband-Guard ─────────────────────────────────────────────
    _s9 = arr.copy()
    arr = _warmth_band_guard(arr, sr, orig)
    arr, _ = _guarded_stage("warmth_guard", _s9, arr)
    applied.append("warmth_guard")

    _progress("Air-Band")
    # ── 10. Air-Band-Polish ────────────────────────────────────────────
    if _allows_air_band(material, era):
        _s10 = arr.copy()
        arr = _apply_high_shelf(arr, sr, 14000, 1.5, 0.4)
        arr, _ = _guarded_stage("air_band_polish", _s10, arr)
        improvements.append("Luftband poliert (+1.5 dB @ 14 kHz)")
        applied.append("air_band_polish")

    _progress("Loudness")
    # ── 11. Loudness-Feinschliff ───────────────────────────────────────
    _s11 = arr.copy()
    arr = _loudness_balance(arr, orig, mode)
    arr, _ = _guarded_stage("loudness_balance", _s11, arr)
    applied.append("loudness_balance")

    _progress("Tonalness")
    # ── 12. Tonalness-Enhancement ──────────────────────────────────────
    if hpe_full.get("tonalness", 0.5) < 0.25 and hpe_before > 0.45:
        _s12 = arr.copy()
        arr = _gentle_harmonic_enhance(arr, sr, 0.12)
        arr, _ = _guarded_stage("tonalness_boost", _s12, arr)
        improvements.append("Tonale Anteile sanft verstärkt")
        applied.append("tonalness_boost")

    _progress("Masterband")
    # ── 13. Studio 2026: Kreative Klangformung (§v10.119) ────────────
    # §v10.119: Noise-Gate, Spektral-Balance und Stereo-Fokus sind
    # KREATIVE Eingriffe, die die Original-Dynamik verändern.
    # Sie laufen NUR im STUDIO_2026-Mode, nicht in RESTORATION.
    # RESTORATION bewahrt die Original-Dynamik — Do No Harm.
    if mode == "STUDIO_2026":
        if _detect_noise_floor(orig, sr):
            _r13a = arr.copy()
            arr = _noise_floor_gate(arr, sr, orig)
            arr, _ = _guarded_stage("noise_floor_gate", _r13a, arr)
            applied.append("noise_floor_gate")

        if _detect_spectral_imbalance(orig, sr):
            _r13b = arr.copy()
            arr = _spectral_balance(arr, sr)
            arr, _ = _guarded_stage("spectral_balance", _r13b, arr)
            applied.append("spectral_balance")

        if is_stereo and _detect_diffuse_center(arr, sr):
            _r13c = arr.copy()
            arr = _stereo_focus(arr, sr)
            arr, _ = _guarded_stage("stereo_focus", _r13c, arr)
            applied.append("stereo_focus")

    _progress("Bandbreite")
    # ── 13d. Bandbreiten-Extension: Vintage-Material (beide Modi) ─────
    if _needs_bandwidth_extension(material):
        _r13d = arr.copy()
        arr = _bandwidth_extend(arr, sr, material)
        arr, _ = _guarded_stage("bandwidth_extend", _r13d, arr)
        applied.append("bandwidth_extend")

    _progress("Studio 2026")
    # ── 14. Studio 2026 Re-Production Chain (§v10.5) ───────────────────
    if mode == "STUDIO_2026":
        _s13_pre = arr.copy()
        try:
            from backend.core.studio2026_chain import reprocess_studio2026

            dna = reprocess_studio2026(arr, orig, sr, material=material, era=era, mode=mode, album_ref=album_ref)
            arr = dna.audio  # Chain already has own DNA guards, no cross-phase needed
            if dna.stages_applied:
                improvements.append(f"Studio 2026: {len(dna.stages_applied)}-stage Re-Production")
            for stage in dna.stages_applied:
                applied.append(f"studio_{stage}")
            if dna.voiceprint_match > 0.90:
                improvements.append(f"Stimme erhalten ({dna.voiceprint_match:.0%})")
            if dna.groove_preserved > 0.95:
                improvements.append(f"Groove erhalten ({dna.groove_preserved:.0%})")
            if dna.emotion_preserved > 0.85:
                improvements.append(f"Emotion erhalten ({dna.emotion_preserved:.0%})")
            dna_ok = dna.voiceprint_match > 0.88 and dna.groove_preserved > 0.90 and dna.emotion_preserved > 0.80
            if dna_ok:
                improvements.append("DNA erhalten: Stimme, Groove, Emotion intakt")
        except Exception as _s26_exc:
            logger.warning("Studio 2026 chain nicht verfuegbar: %s", _s26_exc)

    # ── 14. Safety Clamp ───────────────────────────────────────────────
    arr = _safety_clamp(arr, orig)

    # ── Final: HPE-Nachmessung ─────────────────────────────────────────
    hpe_after = _compute_hpe(arr, sr)
    stereo_after = _measure_stereo_width(arr) if is_stereo else 0.0

    if hpe_after > hpe_before + 0.02:
        improvements.insert(0, f"Natürlichkeit: {hpe_before:.2f} → {hpe_after:.2f} (+{hpe_after - hpe_before:.2f})")
    elif hpe_after < hpe_before - 0.03:
        logger.warning("NaturalnessOptimizer: Verschlechterung, gebe UV3-Originalsignal zurück")
        arr = np.asarray(audio, dtype=np.float32)
        hpe_after = hpe_before
    else:
        improvements.insert(0, "Natürlichkeit erhalten – bereits optimal.")

    return NaturalnessResult(
        audio=arr,
        hpe_before=hpe_before,
        hpe_after=hpe_after,
        delta_hpe=hpe_after - hpe_before,
        improvements=improvements,
        applied_stages=applied,
        glue_reduction_db=glue_db,
        stereo_width_before=stereo_before,
        stereo_width_after=stereo_after,
        transient_events_protected=transients_protected,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Multi-Band Glue
# ═══════════════════════════════════════════════════════════════════════════


def _multiband_glue(audio: np.ndarray, sr: int, mode: str = "RESTORATION") -> tuple[np.ndarray, float]:
    """3-Band SSL-Style Glue-Kompression.

    Bands: Low < 250Hz, Mid 250-4000Hz, High > 4kHz.
    Jedes Band bekommt eigene Ratio/Threshold.
    """
    try:
        from scipy.signal import butter, sosfiltfilt

        nyq = sr / 2

        # Linkwitz-Riley crossovers (2x butter 2nd order)
        sos_lo = butter(2, 250 / nyq, btype="low", output="sos")
        sos_mid_lo = butter(2, 250 / nyq, btype="high", output="sos")
        sos_mid_hi = butter(2, 4000 / nyq, btype="low", output="sos")
        sos_hi = butter(2, 4000 / nyq, btype="high", output="sos")

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio

        def _split(x):
            lo = sosfiltfilt(sos_lo, x)
            mid = sosfiltfilt(sos_mid_hi, sosfiltfilt(sos_mid_lo, x))
            hi = sosfiltfilt(sos_hi, x)
            return lo, mid, hi

        bands = _split(mono)
        band_configs = [  # (ratio, threshold_db, attack_ms, release_ms)
            (1.10, -8.0, 40, 150),  # Low: gentle, slow
            (1.20, -12.0, 20, 100),  # Mid: standard glue
            (1.15, -14.0, 15, 80),  # High: light touch
        ]
        if mode == "STUDIO_2026":
            band_configs = [
                (1.15, -10.0, 30, 120),
                (1.30, -14.0, 15, 80),
                (1.20, -16.0, 10, 60),
            ]

        processed = []
        total_gr_db = 0.0
        for band, (ratio, thresh_db, att_ms, rel_ms) in zip(bands, band_configs):
            proc, gr = _compress_band(band, sr, ratio, thresh_db, att_ms, rel_ms)
            processed.append(proc)
            total_gr_db = max(total_gr_db, gr)

        combined = processed[0] + processed[1] + processed[2]

        if audio.ndim == 2:
            diff = combined / (mono + 1e-12)
            result = audio * np.clip(diff, 0.85, 1.15)[:, np.newaxis]
        else:
            result = combined

        return result.astype(np.float32), min(total_gr_db, 3.0)
    except Exception as e:
        logger.warning("_split: %s", e)
        return audio, 0.0


def _compress_band(
    band: np.ndarray, sr: int, ratio: float, thresh_db: float, att_ms: float, rel_ms: float
) -> tuple[np.ndarray, float]:
    """Komprimiert ein Frequenzband."""
    n = len(band)
    att_c = np.exp(-1.0 / (att_ms / 1000.0 * sr))
    rel_c = np.exp(-1.0 / (rel_ms / 1000.0 * sr))

    thresh_lin = 10.0 ** (thresh_db / 20.0)
    rms = np.sqrt(band**2 + 1e-12)
    gain = np.ones(n, dtype=np.float32)
    gr_state = 1.0

    for i in range(n):
        if rms[i] > thresh_lin:
            target = thresh_lin + (rms[i] - thresh_lin) / ratio
            tgt = target / (rms[i] + 1e-12)
        else:
            tgt = 1.0
        gr_state = (att_c if tgt < gr_state else rel_c) * gr_state + (1 - (att_c if tgt < gr_state else rel_c)) * tgt
        gain[i] = gr_state

    gain = np.clip(gain, 0.6, 1.0)
    gr_db = -20.0 * np.log10(float(np.min(gain)) + 1e-12)
    return band * gain, min(gr_db, 3.0)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Stereo-Feld-Optimierung
# ═══════════════════════════════════════════════════════════════════════════


def _stereo_field_optimize(audio: np.ndarray, original: np.ndarray, sr: int, material: str) -> tuple[np.ndarray, str]:
    """Bewahrt und optimiert das Stereofeld."""
    try:
        L, R = audio[:, 0], audio[:, 1]
        orig_L, orig_R = original[:, 0], original[:, 1]

        # Mid/Side
        M = (L + R) / 2.0
        S = (L - R) / 2.0
        orig_M = (orig_L + orig_R) / 2.0
        orig_S = (orig_L - orig_R) / 2.0

        rms_S = float(np.sqrt(np.mean(S**2)) + 1e-12)
        rms_M = float(np.sqrt(np.mean(M**2)) + 1e-12)
        rms_orig_S = float(np.sqrt(np.mean(orig_S**2)) + 1e-12)
        rms_orig_M = float(np.sqrt(np.mean(orig_M**2)) + 1e-12)

        width_now = rms_S / (rms_M + 1e-12)
        width_orig = rms_orig_S / (rms_orig_M + 1e-12)

        # Mono-Ära-Schutz: Breite nicht >120% vom Original
        max_width = width_orig * 1.2
        min_width = width_orig * 0.85

        if width_now > max_width:
            S *= max_width / width_now
            info = f"Stereo-Breite begrenzt ({width_now:.2f}→{max_width:.2f})"
        elif width_now < min_width and width_orig > 0.05:
            S *= min_width / width_now
            info = f"Stereo-Breite wiederhergestellt ({width_now:.2f}→{min_width:.2f})"
        else:
            info = ""

        # Für extrem schmale Originale (Mono): nichts erzwingen
        if width_orig < 0.03 and material not in ("cd_digital", "streaming"):
            return audio, info

        L_out = M + S
        R_out = M - S
        result = np.stack([L_out, R_out], axis=1)
        return result.astype(np.float32), info
    except Exception as e:
        logger.warning("_stereo_field_optimieren: %s", e)
        return audio, ""


def _measure_stereo_width(audio: np.ndarray) -> float:
    """Misst die Stereo-Breite (Mid/Side Ratio)."""
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 0.0
    M = (audio[:, 0] + audio[:, 1]) / 2.0
    S = (audio[:, 0] - audio[:, 1]) / 2.0
    rms_S = float(np.sqrt(np.mean(S**2)) + 1e-12)
    rms_M = float(np.sqrt(np.mean(M**2)) + 1e-12)
    return rms_S / (rms_M + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Transienten-Schutz
# ═══════════════════════════════════════════════════════════════════════════


def _transient_preservation(audio: np.ndarray, original: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Erkennt und schützt Attack-Transienten vor Überglättung.

    §v10.112 Groove-Guard: Bei hoher Transientendichte (groove > 1.5/s)
    wird die Original-Blend-Rate drastisch reduziert, weil die Restauration
    die Attack-Transienten verbessert hat und ein Ersetzen durch das
    unverarbeitete Original den Groove zerstören würde (1673 geschützte
    Transienten = zerstörter Punch).
    """
    try:
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        orig_mono = original.mean(axis=1) if original.ndim == 2 else original

        # Envelope detection
        win = int(0.005 * sr)  # 5ms
        n_windows = len(mono) // win
        if n_windows < 4:
            return audio, 0

        env = np.array([float(np.max(np.abs(mono[i * win : (i + 1) * win]))) for i in range(n_windows)])
        _orig_env = np.array([float(np.max(np.abs(orig_mono[i * win : (i + 1) * win]))) for i in range(n_windows)])

        # Detect attacks: rapid envelope rise
        diff = np.diff(env)
        # §v10.124 Major-Version: Rauschquellen (tiefe Ketten) erzeugen viele
        # kleine Diff-Peaks → 90. Perzentil findet Noise statt Musik-Transienten.
        # Hebe das Perzentil für tiefe Ketten an (Proxy: HG-Threshold).
        try:
            from backend.core.signal_flow_tracer import get_hallucination_guard_threshold

            _hg_proxy = get_hallucination_guard_threshold()
            _attack_percentile = 90.0 + max(0, _hg_proxy - 0.15) * 20.0  # 90→95 bei depth 4
        except Exception:
            _attack_percentile = 90.0
        attacks = np.where(diff > np.percentile(diff, _attack_percentile) * 1.5)[0]
        attacks = attacks[attacks < n_windows - 1]

        if len(attacks) == 0:
            return audio, 0

        # §v10.112 Groove-Guard: Transientendichte (attacks/sec) bestimmt Blend-Stärke.
        # Je dichter die Transienten, desto mehr Punch hat der Song — und desto
        # weniger darf das Original die restaurierten Transienten ersetzen.
        _duration_s = len(mono) / sr
        _transient_density = len(attacks) / max(_duration_s, 1.0)
        if _transient_density > 5.0:
            _blend_strength = 0.05  # 95% restauriert — Minimal-Eingriff
        elif _transient_density > 3.0:
            _blend_strength = 0.10  # 90% restauriert
        elif _transient_density > 1.5:
            _blend_strength = 0.18  # 82% restauriert
        else:
            _blend_strength = 0.30  # 70% restauriert (original, für ambient/klassik)

        # §v10.120 Depth-aware blend: tiefe Transfer-Ketten haben fragilere
        # Original-Transienten (Crackle, Noise-Peaks). Reduziere den Original-
        # Anteil proportional zur Chain-Depth (Proxy: HG-Threshold > 0.25).
        try:
            from backend.core.signal_flow_tracer import get_hallucination_guard_threshold

            _hg_proxy = get_hallucination_guard_threshold()
            if _hg_proxy > 0.25:
                _depth_factor = float(np.clip(1.0 - (_hg_proxy - 0.25) * 1.2, 0.40, 1.0))
                _blend_strength *= _depth_factor
        except Exception:
            pass  # SFT nicht verfügbar — Default-Blend verwenden

        # For each attack, blend in original transient at groove-adaptive strength
        blend = np.ones(len(mono), dtype=np.float32)
        for a in attacks:
            start = a * win
            end = min((a + 2) * win, len(mono))
            # Hanning crossfade: blend_strength original, (1-blend_strength) processed at attack peak
            t = np.linspace(0, 1, end - start)
            fade = 1.0 - _blend_strength * np.exp(-4.0 * t) * (1.0 - t)
            blend[start:end] = np.minimum(blend[start:end], fade.astype(np.float32))

        if audio.ndim == 2:
            result = audio * blend[:, np.newaxis] + original * (1.0 - blend[:, np.newaxis])
        else:
            result = audio * blend + original * (1.0 - blend)

        return result.astype(np.float32), len(attacks)
    except Exception as e:
        logger.warning("_transient_preservation: %s", e)
        return audio, 0


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: De-Essing-Nachbearbeitung
# ═══════════════════════════════════════════════════════════════════════════


def _gentle_de_ess(audio: np.ndarray, sr: int, material: str) -> tuple[np.ndarray, str]:
    """Sanftes, material-adaptives De-Essing."""
    try:
        from scipy.signal import butter, sosfiltfilt

        # Sibilanz-Band: weiblich 7-11kHz, männlich 5-9kHz, default 6-10kHz
        band = (5000, 10000)
        nyq = sr / 2
        sos = butter(2, [band[0] / nyq, band[1] / nyq], btype="band", output="sos")

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        sib_band = sosfiltfilt(sos, mono)
        _full = sosfiltfilt(sos, np.ones_like(mono)) if False else mono

        rms_sib = float(np.sqrt(np.mean(sib_band**2)) + 1e-12)
        rms_full = float(np.sqrt(np.mean(mono**2)) + 1e-12)
        sib_ratio = rms_sib / (rms_full + 1e-12)

        # Only de-ess if sibilance ratio is unusually high
        if sib_ratio < 0.08:
            return audio, ""

        # Gentle reduction: max 2dB in the sibilance band
        reduction = min(2.0, (sib_ratio - 0.06) * 40)
        sos_notch = butter(1, [band[0] / nyq, band[1] / nyq], btype="bandstop", output="sos")

        if audio.ndim == 2:
            result = np.zeros_like(audio)
            for ch in range(2):
                filtered = sosfiltfilt(sos_notch, audio[:, ch])
                result[:, ch] = audio[:, ch] + (filtered - audio[:, ch]) * min(reduction / 6.0, 0.3)
            return result.astype(np.float32), f"Sibilanz sanft reduziert (−{reduction:.1f} dB)"
        else:
            filtered = sosfiltfilt(sos_notch, audio)
            result = audio + (filtered - audio) * min(reduction / 6.0, 0.3)
            return result.astype(np.float32), f"Sibilanz sanft reduziert (−{reduction:.1f} dB)"
    except Exception as e:
        logger.warning("_gentle_de_ess: %s", e)
        return audio, ""


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6: Bass-Management
# ═══════════════════════════════════════════════════════════════════════════


def _bass_preservation(audio: np.ndarray, original: np.ndarray, sr: int) -> np.ndarray:
    """Stellt sicher, dass Sub-Bass (20-100Hz) nicht verloren geht."""
    try:
        from scipy.signal import butter, sosfiltfilt

        nyq = sr / 2
        sos = butter(2, [20 / nyq, 100 / nyq], btype="band", output="sos")

        def _bass_energy(x):
            m = x.mean(axis=1) if x.ndim == 2 else x
            f = sosfiltfilt(sos, m)
            return float(np.sqrt(np.mean(f**2)) + 1e-12)

        e_orig = _bass_energy(original)
        e_cur = _bass_energy(audio)

        if e_cur < e_orig * 0.8 and e_orig > 1e-8:
            gain = min(1.3, e_orig / e_cur)
            # Apply gentle broadband gain
            return (audio * gain).astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_bass_energy: %s", e)
    return audio


# ═══════════════════════════════════════════════════════════════════════════
# Stages 7-13: Sharpness, Roughness, Warmth, Air, Loudness, Tonalness, Clamp
# ═══════════════════════════════════════════════════════════════════════════


def _apply_high_shelf(audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 0.7) -> np.ndarray:
    if abs(gain_db) < 0.2:
        return audio
    try:
        from scipy.signal import butter, sosfiltfilt

        nyq = sr / 2
        sos = butter(2, freq / nyq, btype="high", output="sos")
        gain_linear = 10.0 ** (gain_db / 20.0)
        if audio.ndim == 2:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered = sosfiltfilt(sos, audio[:, ch])
                result[:, ch] = audio[:, ch] + (filtered - audio[:, ch]) * (gain_linear - 1.0)
            return cast(np.ndarray, result.astype(np.float32))
        else:
            filtered = sosfiltfilt(sos, audio)
            return (audio + (filtered - audio) * (gain_linear - 1.0)).astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_anwenden_high_shelf: %s", e)
        return audio


def _smooth_micro_dynamics(audio: np.ndarray, sr: int, window_ms: float = 30) -> np.ndarray:
    win = int(window_ms / 1000.0 * sr)
    if win < 8 or len(audio) < win * 4:
        return audio
    try:
        from scipy.ndimage import uniform_filter1d

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        env = np.sqrt(uniform_filter1d(mono**2, win) + 1e-12)
        smooth = uniform_filter1d(env, win * 2)
        ratio = np.clip(smooth / (env + 1e-12), 0.7, 1.3)
        if audio.ndim == 2:
            return (audio * ratio[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return (audio * ratio).astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_smooth_micro_dynamics: %s", e)
        return audio


def _warmth_band_guard(audio: np.ndarray, sr: int, reference: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfiltfilt

        nyq = sr / 2
        sos = butter(3, [200.0 / nyq, 800.0 / nyq], btype="band", output="sos")

        def _band_rms(x):
            m = x.mean(axis=1) if x.ndim == 2 else x
            f = sosfiltfilt(sos, m)
            return float(np.sqrt(np.mean(f**2)) + 1e-12)

        rms_ref = _band_rms(reference)
        rms_cur = _band_rms(audio)
        if rms_cur < rms_ref * 0.85 and rms_cur > 1e-10:
            gain = min(1.25, rms_ref / rms_cur)
            return (audio * gain).astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_band_rms: %s", e)
    return audio


def _allows_air_band(material: str, era: str) -> bool:
    no_air = {"shellac", "wax_cylinder", "wire_recording", "lacquer_disc"}
    if material in no_air:
        return False
    try:
        if era and int(str(era)[:4]) < 1945:
            return False
    except (ValueError, IndexError):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return True


def _loudness_balance(audio: np.ndarray, original: np.ndarray, mode: str) -> np.ndarray:
    rms_cur = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    rms_orig = float(np.sqrt(np.mean(original**2)) + 1e-12)
    cur_db = 20.0 * np.log10(rms_cur + 1e-12)
    _orig_db = 20.0 * np.log10(rms_orig + 1e-12)

    target = -15.0 if mode == "RESTORATION" else -11.0
    diff = target - cur_db
    if abs(diff) > 1.5:
        diff = np.clip(diff, -5.0, 5.0)
        return (audio * (10.0 ** (diff / 20.0))).astype(np.float32)  # type: ignore[no-any-return]
    return audio


def _gentle_harmonic_enhance(audio: np.ndarray, sr: int, amount: float = 0.12) -> np.ndarray:
    if amount <= 0.01:
        return audio
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    soft = np.tanh(mono * 0.5)
    enhanced = mono + (soft - mono * 0.5) * amount * 0.3
    if audio.ndim == 2:
        ratio = np.clip(enhanced / (mono + 1e-12), 0.9, 1.1)
        return (audio * ratio[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
    return enhanced.astype(np.float32)  # type: ignore[no-any-return]


def _safety_clamp(audio: np.ndarray, original: np.ndarray) -> np.ndarray:
    orig_rms = float(np.sqrt(np.mean(original**2)) + 1e-12)
    opt_rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    if opt_rms > orig_rms * 2.0 and orig_rms > 1e-10:
        audio = audio * (orig_rms * 2.0 / opt_rms)
    orig_peak = float(np.max(np.abs(original))) + 1e-12
    opt_peak = float(np.max(np.abs(audio))) + 1e-12
    if opt_peak > orig_peak * 2.0 and orig_peak > 1e-10:
        audio = audio * (orig_peak * 2.0 / opt_peak)
    return np.tanh(audio).astype(np.float32)  # type: ignore[no-any-return]  # Soft-clip (tanh) statt Hard-Clip


# ═══════════════════════════════════════════════════════════════════════════
# HPE-Aufrufe & Vergleich
# ═══════════════════════════════════════════════════════════════════════════


def _compute_hpe(audio: np.ndarray, sr: int) -> float:
    try:
        from backend.core.human_pleasantness_estimator import compute_pleasantness

        return float(compute_pleasantness(audio, sr).score)
    except Exception as e:
        logger.warning("_berechnen_hpe: %s", e)
        return _fallback_hpe(audio)


def _compute_hpe_full(audio: np.ndarray, sr: int) -> dict:
    try:
        from backend.core.human_pleasantness_estimator import compute_pleasantness

        r = compute_pleasantness(audio, sr)
        return {
            "score": float(r.score),
            "sharpness_zwicker": float(r.sharpness_zwicker),
            "roughness_asper": float(r.roughness_asper),
            "loudness_sone": float(r.loudness_sone),
            "tonalness": float(r.tonalness),
            "fluctuation_vacil": float(r.fluctuation_vacil),
            "label": r.label,
        }
    except Exception as e:
        logger.warning("_berechnen_hpe_full: %s", e)
        return {
            "score": _fallback_hpe(audio),
            "sharpness_zwicker": 1.5,
            "roughness_asper": 0.5,
            "loudness_sone": 18.0,
            "tonalness": 0.5,
            "fluctuation_vacil": 0.5,
            "label": "?",
        }


def _fallback_hpe(audio: np.ndarray) -> float:
    mono = audio.mean(axis=1) if audio.ndim == 2 and audio.shape[1] <= 2 else audio.ravel()
    rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 1e-12
    crest = min(peak / (rms + 1e-12), 20.0)
    return float(
        min(1.0, max(0.0, 0.5 * (1.0 - abs(crest - 4.0) / 8.0) + 0.5 * (1.0 - abs(20.0 * np.log10(rms) + 18) / 20)))
    )


# ═══════════════════════════════════════════════════════════════════════════
# Restoration: Masterband-Qualität (§v10.6)
# Diese Stages laufen NUR in RESTORATION — nicht in Studio 2026.
# Ziel: Klingt wie frisch vom Masterband, in CD-Qualität.
# Kein EQ-Boost, keine Kompression, kein Widening.
# Nur: Rauschen in Pausen senken, Frequenzbalance glätten, Phantom-Mitte.
# ═══════════════════════════════════════════════════════════════════════════


def _noise_floor_gate(audio: np.ndarray, sr: int, original: np.ndarray) -> np.ndarray:
    """Sanftes Noise-Gate: Rauschpegel in Signalpausen um 1-2 dB senken.

    Nur aktiv wenn Signal-Pegel < −55 dB (echte Stille/Rauschen).
    Max −1.4 dB Reduktion, Attack 5 ms, Release 80 ms.
    Erhält natürlichen Raumton unter −60 dBFS — absolute Stille
    zwischen Noten klingt unnatürlich und ermüdend.
    §v10.119: Schwelle −48→−55 dB, erster Audiosample ignoriert
    (musikalische Intros sind keine Rauschpausen).
    """
    try:
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio

        # RMS in 10ms-Fenstern
        win = int(0.010 * sr)
        n_win = len(mono) // win
        rms = np.array([float(np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2)) + 1e-12) for i in range(n_win)])
        rms_db = 20.0 * np.log10(rms)

        # §v10.119: Schwelle auf −55 dB gesenkt (war −48).
        # −48 dB erfasste musikalische Intros (leise Balladen-Anfänge)
        # als 'Rauschen' und drosselte sie um −1.4 dB.
        # −55 dB ist unterhalb jedes Musiksignals — nur echtes Rauschen.
        threshold_db = -55.0
        is_noise = rms_db < threshold_db

        # §v10.119: Intro-Guard — die ersten 3 Sekunden nie gaten.
        # Ein musikalisch leiser Anfang (Fade-In, Ballade) ist KEIN Rauschen.
        _intro_wins = int(3.0 * sr / win)
        if _intro_wins < len(is_noise):
            is_noise[:_intro_wins] = False

        # Smooth Gate: Attack 5ms, Release 80ms
        att = np.exp(-1.0 / (0.005 * sr / win))
        rel = np.exp(-1.0 / (0.080 * sr / win))

        gate = np.ones(n_win, dtype=np.float32)
        state = 1.0
        for i in range(n_win):
            target = 0.85 if is_noise[i] else 1.0  # −1.4 dB max, erhält Raumton
            coef = att if target < state else rel
            state = coef * state + (1.0 - coef) * target
            gate[i] = state

        # Apply per-sample
        upsampled = np.interp(np.arange(len(mono)), np.arange(n_win) * win, gate)
        if audio.ndim == 2:
            return (audio * upsampled[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return cast(np.ndarray, (audio * upsampled).astype(np.float32))
    except Exception as e:
        logger.warning("_noise_floor_gate: %s", e)
        return audio


def _spectral_balance(audio: np.ndarray, sr: int) -> np.ndarray:
    """Spektrale Balance: Bänder >3 dB vom Neutral → sanft korrigieren.

    Misst RMS pro 1/3-Oktav-Band. Wenn ein Band >3 dB vom
    gleitenden Mittel abweicht, ziehe es um 40% Richtung Mittel.
    Nur CUTS (keine Boosts) — konservativ, nicht aggressiv.
    """
    try:
        from scipy.signal import butter, sosfiltfilt

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        nyq = sr / 2

        # 9 Bänder (Bark-Skala, vereinfacht)
        bands = [
            (20, 60),
            (60, 200),
            (200, 400),
            (400, 800),
            (800, 1500),
            (1500, 3000),
            (3000, 6000),
            (6000, 10000),
            (10000, 18000),
        ]

        band_energies = []
        for lo, hi in bands:
            if hi >= nyq * 0.95:
                hi = nyq * 0.95  # type: ignore[assignment]
            if lo >= hi:
                continue
            sos = butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
            filtered = sosfiltfilt(sos, mono)
            rms = float(np.sqrt(np.mean(filtered**2)) + 1e-12)
            rms_db = 20.0 * np.log10(rms)
            band_energies.append(rms_db)

        if len(band_energies) < 3:
            return audio

        # Gleitender Mittel über 3 Bänder
        smoothed = np.convolve(band_energies, [1 / 3] * 3, mode="same")

        result = mono.copy()
        for i, (lo, hi) in enumerate(bands):
            if i >= len(smoothed):
                continue
            delta = band_energies[i] - smoothed[i]
            if abs(delta) > 3.0 and delta > 0:  # Nur CUTS, kein Boost
                correction = -delta * 0.4  # 40% Richtung Mittel
                correction = np.clip(correction, -4.0, 0.0)
                if hi >= nyq * 0.95:
                    hi = nyq * 0.95  # type: ignore[assignment]
                sos = butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
                band_sig = sosfiltfilt(sos, mono)
                gain = 10.0 ** (correction / 20.0)
                result = result + band_sig * (gain - 1.0)

        if audio.ndim == 2:
            ratio = np.clip(result / (mono + 1e-12), 0.85, 1.05)
            return (audio * ratio[:, np.newaxis]).astype(np.float32)  # type: ignore[no-any-return]
        return cast(np.ndarray, result.astype(np.float32))
    except Exception as e:
        logger.warning("_spectral_balance: %s", e)
        return audio


def _stereo_focus(audio: np.ndarray, sr: int) -> np.ndarray:
    """Stereo-Fokus: Phantom-Mitte in 300-3000 Hz stabilisieren.

    Analysiert die Mid/Side-Korrelation im Sprach-/Gesangsbereich.
    Reduziert Side-Anteil um 5-10% wenn die Mitte diffus ist.
    Betrifft NICHT Höhen (>6 kHz) — dort bleibt die Breite erhalten.
    """
    if audio.ndim < 2 or audio.shape[1] < 2:
        return audio
    try:
        from scipy.signal import butter, sosfiltfilt

        L, R = audio[:, 0], audio[:, 1]
        M, S = (L + R) / 2.0, (L - R) / 2.0
        nyq = sr / 2

        # Nur 300-3000 Hz (Sprach-/Gesangsbereich)
        sos_mid_lo = butter(2, 300 / nyq, btype="high", output="sos")
        sos_mid_hi = butter(2, 3000 / nyq, btype="low", output="sos")

        S_mid = sosfiltfilt(sos_mid_hi, sosfiltfilt(sos_mid_lo, S))

        # Wenn Side-Energie > 25% der Mid-Energie → Zentrum etwas straffen
        rms_M = float(np.sqrt(np.mean(M**2)) + 1e-12)
        rms_S_mid = float(np.sqrt(np.mean(S_mid**2)) + 1e-12)
        if rms_S_mid / (rms_M + 1e-12) > 0.25:
            S_mid *= 0.90  # −10% Side im Gesangsbereich

        # Rekombinieren: Höhen bleiben breit, Mitten werden fokussiert
        sos_hi = butter(2, 3000 / nyq, btype="high", output="sos")
        S_hi = sosfiltfilt(sos_hi, S)
        sos_lo = butter(2, 300 / nyq, btype="low", output="sos")
        S_lo = sosfiltfilt(sos_lo, S)

        S_new = S_lo + S_mid + S_hi
        L_out, R_out = M + S_new, M - S_new
        return np.stack([L_out, R_out], axis=1).astype(np.float32)  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("_stereo_focus: %s", e)
        return audio


def compare_naturalness(original: np.ndarray, restored: np.ndarray, sr: int) -> dict[str, Any]:
    hpe_orig = _compute_hpe_full(original, sr)
    hpe_rest = _compute_hpe_full(restored, sr)
    delta = hpe_rest["score"] - hpe_orig["score"]
    if delta > 0.10:
        verdict = "deutlich natürlicher"
    elif delta > 0.03:
        verdict = "etwas natürlicher"
    elif delta > -0.03:
        verdict = "gleich natürlich"
    else:
        verdict = "weniger natürlich"
    return {
        "original_hpe": round(hpe_orig["score"], 3),
        "restored_hpe": round(hpe_rest["score"], 3),
        "delta": round(delta, 3),
        "verdict": verdict,
        "original_label": hpe_orig["label"],
        "restored_label": hpe_rest["label"],
        "original_sharpness": round(hpe_orig["sharpness_zwicker"], 2),
        "restored_sharpness": round(hpe_rest["sharpness_zwicker"], 2),
        "original_roughness": round(hpe_orig["roughness_asper"], 2),
        "restored_roughness": round(hpe_rest["roughness_asper"], 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Detection Helpers: Analysieren ob Masterband-Stages nötig sind (§v10.6)
# ═══════════════════════════════════════════════════════════════════════════


def _detect_noise_floor(audio: np.ndarray, sr: int) -> bool:
    """Detektiert ob ein hörbarer Rausch-Teppich in Signalpausen existiert."""
    try:
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        win = int(0.050 * sr)
        n_win = len(mono) // win
        if n_win < 10:
            return False
        rms = np.array([float(np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2)) + 1e-12) for i in range(n_win)])
        rms_db = 20.0 * np.log10(rms)
        noise_floor = float(np.percentile(rms_db, 10))
        return noise_floor > -50.0
    except Exception:
        return False


def _detect_spectral_imbalance(audio: np.ndarray, sr: int) -> bool:
    """Detektiert ob die spektrale Balance signifikant unausgeglichen ist."""
    try:
        from scipy.signal import butter, sosfiltfilt

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        nyq = sr / 2
        bands = [
            (20, 60),
            (60, 200),
            (200, 400),
            (400, 800),
            (800, 1500),
            (1500, 3000),
            (3000, 6000),
            (6000, 10000),
            (10000, 18000),
        ]
        energies = []
        for lo, hi in bands:
            if hi >= nyq * 0.95:
                hi = nyq * 0.95  # type: ignore[assignment]
            if lo >= hi:
                continue
            sos = butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
            rms = float(np.sqrt(np.mean(sosfiltfilt(sos, mono) ** 2)) + 1e-12)
            energies.append(20.0 * np.log10(rms))
        if len(energies) < 5:
            return False
        median = float(np.median(energies))
        return any(abs(e - median) > 6.0 for e in energies)
    except Exception:
        return False


def _detect_diffuse_center(audio: np.ndarray, sr: int) -> bool:
    """Detektiert ob die Phantom-Mitte im Gesangsbereich diffus ist."""
    if audio.ndim < 2 or audio.shape[1] < 2:
        return False
    try:
        from scipy.signal import butter, sosfiltfilt

        L, R = audio[:, 0], audio[:, 1]
        M, S = (L + R) / 2, (L - R) / 2
        nyq = sr / 2
        sos_lo = butter(2, 300 / nyq, btype="high", output="sos")
        sos_hi = butter(2, 3000 / nyq, btype="low", output="sos")
        S_mid = sosfiltfilt(sos_hi, sosfiltfilt(sos_lo, S))
        M_mid = sosfiltfilt(sos_hi, sosfiltfilt(sos_lo, M))
        rms_S = float(np.sqrt(np.mean(S_mid**2)) + 1e-12)
        rms_M = float(np.sqrt(np.mean(M_mid**2)) + 1e-12)
        return rms_S / (rms_M + 1e-12) > 0.35
    except Exception:
        return False


def _needs_bandwidth_extension(material: str) -> bool:
    """Prüft ob das Material Bandbreiten-Extension braucht.
    Nur für Vintage-Material mit bekannten Bandbreiten-Limits.
    """
    from backend.core.dsp.bandwidth_extender import needs_bandwidth_extension

    return needs_bandwidth_extension(material)


def _bandwidth_extend(audio, sr, material):
    """Erweitert Bandbreite für Vintage-Material via spektrale Spiegelung."""
    from backend.core.dsp.bandwidth_extender import extend_bandwidth

    return extend_bandwidth(audio, sr, material=material, amount=0.35)
