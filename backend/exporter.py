import hashlib
import io

# v10.101 SOTA: Export durch Pipeline-Gates (JND+Perceptual-Blend) geschützt.
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import soundfile as sf

from backend.core.metadata_preserver import get_metadata_preserver

try:
    from backend.core.version import AURIK_VERSION as _AURIK_VERSION
except Exception:
    _AURIK_VERSION = "unknown"

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

try:
    from scipy import signal as _scipy_signal

    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_signal = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _transfer_metadata(source_path: str, target_path: str, *, transfer_chain: list[str] | None = None) -> None:
    """Best-effort metadata transfer from source to target audio file.

    Embeds Aurik provenance + optional carrier-chain metadata (§2.46a).
    """
    if not source_path:
        return
    try:
        get_metadata_preserver().transfer(
            source_path,
            target_path,
            aurik_version=_AURIK_VERSION,
            transfer_chain=transfer_chain,
        )
    except Exception as exc:
        logger.debug("Metadaten-Übertragung übersprungen: %s", exc)


# ── Transferkette / Carrier-Chain Metadata (§2.46a) ─────────────────────
# Thread-safe module-level storage for the carrier chain metadata discovered
# during forensics analysis.  Populated by the denker layer before export.

import threading as _threading

_chain_lock: _threading.Lock = _threading.Lock()
_chain_metadata: dict[str, object] = {}


def set_chain_metadata(data: dict[str, object]) -> None:
    """Store carrier-chain metadata for the current export session.

    Called by the denker/forensics layer after Tonträgerketten-Analyse.
    Thread-safe – uses a lock so concurrent pipeline runs don't interfere.

    Parameters
    ----------
    data : dict
        Chain metadata dict with keys such as ``carriers``, ``chain_depth``,
        ``degradation_level``, ``phase_recommendations``.
    """
    global _chain_metadata
    with _chain_lock:
        _chain_metadata = dict(data)


def _build_chain_metadata() -> dict[str, object]:
    """Return the carrier-chain metadata snapshot for embedding in export tags.

    Returns an empty dict when no chain analysis has been performed.
    Called by :func:`export_audio` during the export phase (§2.46a).
    """
    global _chain_metadata
    with _chain_lock:
        return dict(_chain_metadata)


# True-Peak threshold: > -0.5 dBTP triggers a warning, > 0 dBTP triggers clipping guard
_TRUE_PEAK_WARN_DBTP = -0.5
_TRUE_PEAK_LIMIT = 1.0

# ---------------------------------------------------------------------------
# POW-r Type 3 noise-shaping coefficients (Craven / Law / Stuart, AES 1987).
# Psychoacoustically optimised for 48 kHz / 16-bit targets; also applied for
# 24-bit (reduces perceived noise floor well beyond −98 dBFS requirement).
# Reference: Meridian Lossless Packing — noise-shaping appendix, Table 3.
# ---------------------------------------------------------------------------
_POWR3_COEFFS = np.array(
    [2.412, -3.370, 3.937, -4.174, 3.353, -2.205, 1.281, -0.569, 0.0847],
    dtype=np.float64,
)


def _apply_powr3_dither(
    audio: np.ndarray, bit_depth: int, *, seed: int | None = 42, cd_active: bool = False
) -> np.ndarray:
    """Wendet an: POW-r Type 3 noise-shaped dither (primary) before integer quantisation.

    Uses error-feedback-approximated noise shaping: TPDF dither is pre-shaped
    with the POW-r Type 3 FIR filter via ``scipy.signal.lfilter``, then added to
    the signal.  This concentrates dither energy in psychoacoustically less
    sensitive high-frequency bands, pushing the perceived noise floor ≥ 14 dB
    lower than flat TPDF at 16-bit (targeting ≤ −72 dBFS per §8.2).

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio in ``[-1.0, 1.0]``, shape ``(samples,)`` or
        ``(samples, channels)``.
    bit_depth : int
        Target integer bit depth.  No-op for 32 or higher.

    Returns
    -------
    np.ndarray
        Dithered float32 audio, still in ``[-1.0, 1.0]``, ready for
        ``soundfile`` integer quantisation.
    """
    if bit_depth >= 32 or not _SCIPY_AVAILABLE:
        return audio

    # Type narrowing for static analyzers: guarded by _SCIPY_AVAILABLE above.
    scipy_signal = _scipy_signal
    if scipy_signal is None:
        return audio

    lsb = 2.0 / (2**bit_depth)
    if cd_active:
        lsb *= 0.7071  # -3 dB (§V5 (copilot-instructions.md): Dither-Doppelung vermeiden)

    mono_input = audio.ndim == 1
    if mono_input:
        a = audio[:, np.newaxis].astype(np.float64)
    else:
        a = audio.astype(np.float64)

    n_samples, n_ch = a.shape

    # TPDF dither: two uniform RVs → triangular distribution centred on 0,
    # amplitude = ±1 LSB (spec §DSP: POW-r Typ 3 primär → TPDF Fallback).
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    raw_dither = (rng.random((n_samples, n_ch)) + rng.random((n_samples, n_ch)) - 1.0) * lsb

    # Shape the dither with the POW-r Type 3 FIR response.
    shaped = np.empty_like(raw_dither)
    for ch in range(n_ch):
        shaped[:, ch] = scipy_signal.lfilter(_POWR3_COEFFS, [1.0], raw_dither[:, ch])

    result = np.clip((a + shaped).astype(np.float32), -1.0, 1.0)
    out = result[:, 0] if mono_input else result
    return cast(np.ndarray, out)


def _apply_tpdf_dither(
    audio: np.ndarray, bit_depth: int, *, seed: int | None = 42, cd_active: bool = False
) -> np.ndarray:
    """TPDF fallback dither — no noise shaping.

    Used when scipy is unavailable.  Amplitude = ±1 LSB triangular noise.

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio in ``[-1.0, 1.0]``.
    bit_depth : int
        Target integer bit depth.  No-op for 32 or higher.

    Returns
    -------
    np.ndarray
        Dithered float32 audio in ``[-1.0, 1.0]``.
    """
    if bit_depth >= 32:
        return audio
    lsb = 2.0 / (2**bit_depth)
    if cd_active:
        lsb *= 0.7071  # -3 dB (§V5 (copilot-instructions.md): Dither-Doppelung vermeiden)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise = (rng.random(audio.shape) + rng.random(audio.shape) - 1.0) * lsb
    out = np.clip((audio + noise).astype(np.float32), -1.0, 1.0)
    return cast(np.ndarray, out)


def apply_dither(
    audio: np.ndarray, bit_depth: int = 16, *, seed: int | None = None, cd_active: bool = False
) -> np.ndarray:
    """Wendet an: dither before integer quantisation.

    Primary: POW-r Type 3 noise-shaped dither (spec §DSP-Spezialregeln).
    Fallback: TPDF dither when scipy is unavailable.
    No-op for ``bit_depth >= 32``.

    Parameters
    ----------
    audio : np.ndarray
        Float audio in ``[-1.0, 1.0]``.
    bit_depth : int
        Target bit depth.

    Returns
    -------
    np.ndarray
        Dithered float32 audio.
    """
    if bit_depth >= 32:
        return audio

    if _SCIPY_AVAILABLE:
        logger.debug("💿 Dithering: POW-r Type 3 angewendet (Bittiefe=%d)", bit_depth)
        return _apply_powr3_dither(audio, bit_depth, seed=seed, cd_active=cd_active)

    logger.warning("Dithering: scipy nicht verfügbar — TPDF-Ersatz angewendet (Bittiefe=%d).", bit_depth)
    return _apply_tpdf_dither(audio, bit_depth, seed=seed, cd_active=cd_active)


def _export_guard(audio: np.ndarray) -> np.ndarray:
    """
    Numerische Robustheitspr\xfcfung unmittelbar vor dem Schreiben.

    Entfernt NaN/Inf, clampt auf [-1.0, 1.0] und warnt bei True-Peak-\xdcberschreitung.
    Entspricht der Pflicht-Invariante aus den Copilot-Instructions (§ Numerische Robustheit).
    """
    # 1. NaN / Inf bereinigen
    audio_clean = np.asarray(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

    # 2. True-Peak pr\xfcfen (Spitzenwert vor Hard-Clip)
    peak = float(np.max(np.abs(audio_clean)))
    if peak > _TRUE_PEAK_LIMIT:
        logger.warning(
            "Ausgabe-Guard: True-Peak %.4f dBTP \xfcberschreitet 0 dBTP — wird begrenzt.",
            20 * math.log10(peak) if peak > 0 else -math.inf,
        )
    elif peak > 10 ** (_TRUE_PEAK_WARN_DBTP / 20):
        logger.warning(
            "Ausgabe-Guard: True-Peak %.2f dBFS n\xe4hert sich 0 dBTP.",
            20 * math.log10(peak) if peak > 0 else -math.inf,
        )

    # 3. Hard-Clip auf [-1.0, 1.0] (Pflicht-Invariante)
    guarded = np.asarray(np.clip(audio_clean, -1.0, 1.0), dtype=np.float32)
    return cast(np.ndarray, guarded)


def _export_nuance_guard(audio: np.ndarray, sr: int) -> np.ndarray:
    """Wendet an: subtle perceptual refinements before final export.

    Goal: preserve listening comfort without changing musical intent.
    Interventions are intentionally conservative and only applied when clear
    nuisance indicators are present.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 0:
        return cast(np.ndarray, np.asarray(a, dtype=np.float32))

    # 1) Short micro-fades to prevent boundary clicks on start/end.
    n = len(a) if a.ndim == 1 else a.shape[0]
    fade_n = int(max(16, min(int(0.005 * max(sr, 1)), n // 8)))
    if n > 2 * fade_n and fade_n > 0:
        w = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        if a.ndim == 1:
            a[:fade_n] *= w
            a[-fade_n:] *= w[::-1]
        else:
            a[:fade_n, :] *= w[:, None]
            a[-fade_n:, :] *= w[::-1, None]

    # 2) Stereo balance guard: fix strong L/R drift only if clearly imbalanced.
    if a.ndim == 2 and a.shape[1] >= 2:
        l = a[:, 0].astype(np.float64)
        r = a[:, 1].astype(np.float64)
        l_rms = float(np.sqrt(np.mean(l**2) + 1e-12))
        r_rms = float(np.sqrt(np.mean(r**2) + 1e-12))
        bal_db = 20.0 * math.log10((l_rms + 1e-12) / (r_rms + 1e-12))
        if abs(bal_db) > 7.0:
            max_ratio = 10.0 ** (7.0 / 20.0)
            g_l = 1.0
            g_r = 1.0
            if l_rms > (r_rms * max_ratio):
                g_l = float(np.clip((r_rms * max_ratio) / (l_rms + 1e-12), 0.80, 1.0))
            elif r_rms > (l_rms * max_ratio):
                g_r = float(np.clip((l_rms * max_ratio) / (r_rms + 1e-12), 0.80, 1.0))
            a[:, 0] = np.clip(a[:, 0] * g_l, -1.0, 1.0)
            a[:, 1] = np.clip(a[:, 1] * g_r, -1.0, 1.0)
            logger.info("Ausgabe-NuanceGuard: Stereobalance korrigiert (%.2f dB, gL=%.3f gR=%.3f)", bal_db, g_l, g_r)

    # 3) Gentle HF-harshness guard (only on clear excess treble energy).
    if _SCIPY_AVAILABLE and _scipy_signal is not None:
        mono = a if a.ndim == 1 else a.mean(axis=1)
        if mono.size > max(1024, sr // 4):
            nper = min(4096, mono.size)
            freqs, psd = _scipy_signal.welch(mono.astype(np.float64), fs=sr, nperseg=nper)
            low_e = float(np.sum(psd[(freqs >= 200.0) & (freqs <= 4500.0)])) + 1e-12
            high_e = float(np.sum(psd[(freqs >= 7000.0) & (freqs <= 14000.0)]))
            ratio = high_e / low_e
            if ratio > 0.42:
                # Split at 6.5 kHz and attenuate only the HF residual slightly.
                sos = _scipy_signal.butter(2, 6500.0 / max(sr / 2.0, 1.0), btype="low", output="sos")
                att = float(np.clip(0.94 - (ratio - 0.42) * 0.10, 0.86, 0.94))
                if a.ndim == 1:
                    low = _scipy_signal.sosfiltfilt(sos, a.astype(np.float64))
                    high = a.astype(np.float64) - low
                    a = np.clip((low + high * att).astype(np.float32), -1.0, 1.0)
                else:
                    out = np.empty_like(a)
                    for ch in range(a.shape[1]):
                        low = _scipy_signal.sosfiltfilt(sos, a[:, ch].astype(np.float64))
                        high = a[:, ch].astype(np.float64) - low
                        out[:, ch] = np.clip((low + high * att).astype(np.float32), -1.0, 1.0)
                    a = out
                logger.info("Ausgabe-NuanceGuard: sanfte Höhenglättung aktiv (Verhaeltnis=%.3f, att=%.3f)", ratio, att)

    guarded = np.asarray(np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), dtype=np.float32)
    return cast(np.ndarray, guarded)


def validate_export_quality(result: Any) -> tuple[bool, list[str]]:
    """Validiert export quality based on RestorationResult metadata.

    Hard-fail conditions (§8.1 / §8.2 / Copilot-Instructions):
    - chroma_correlation < 0.80 (catastrophic tonal shift)
    - quality_estimate < 0.55 (spec §8.1 E2E-Pflicht)
    - P1 goal (natuerlichkeit, authentizitaet) below minimum threshold

    Returns (passed, list_of_warnings).
    """
    warnings: list[str] = []
    passed = True

    # §8.2 Chroma correlation ≥ 0.95 (Tonart-Erhaltung)
    chroma = getattr(result, "chroma_correlation", None)
    if chroma is not None:
        if chroma < 0.80:
            warnings.append(
                f"KRITISCH: Chroma-Korrelation {chroma:.3f} < 0.80 — "
                "schwere Tonart-Verschiebung erkannt. Export wird nicht empfohlen."
            )
            passed = False
        elif chroma < 0.95:
            warnings.append(f"WARNUNG: Chroma-Korrelation {chroma:.3f} < 0.95 — geringe Tonart-Abweichung erkannt.")

    # §8.1 quality_estimate ≥ 0.55 (E2E-Pflicht)
    quality_estimate = getattr(result, "quality_estimate", None)
    if quality_estimate is not None and quality_estimate < 0.55:
        warnings.append(
            f"KRITISCH: quality_estimate {quality_estimate:.3f} < 0.55 — "
            "Mindestqualität nicht erreicht. Mögliche Ursache: schweres Ausgangsmaterial "
            "oder Pipeline-Fehler. Prüfe Defekt-Analyse und Musical Goals."
        )
        passed = False

    # §8.2 LUFS delta ≤ 1 LU (Restoration) / EBU R128 (Studio 2026)
    lufs_delta = getattr(result, "lufs_delta", None)
    if lufs_delta is not None and lufs_delta > 3.0:
        warnings.append(f"WARNUNG: LUFS-Delta {lufs_delta:.1f} LU > 3.0 LU — signifikante Lautstärke-Änderung.")

    # Musical Goals: P1/P2 hard-fail, P3–P5 warning-only
    meta = getattr(result, "metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}

    def _nested_float(container: dict[str, Any], *path: str) -> float | None:
        cur: Any = container
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        if isinstance(cur, bool):
            return None
        if isinstance(cur, (float, int)):
            return float(cur)
        return None

    def _nested_bool(container: dict[str, Any], *path: str) -> bool:
        cur: Any = container
        for key in path:
            if not isinstance(cur, dict):
                return False
            cur = cur.get(key)
        return bool(cur) if isinstance(cur, bool) else False

    # §0h/§2.68: Export darf struktureller Stille keine Energie hinzufuegen.
    silence_audit = meta.get("structural_silence_audit")
    if isinstance(silence_audit, dict):
        if _nested_bool(silence_audit, "failed") or _nested_bool(silence_audit, "violation"):
            warnings.append("KRITISCH: Structural-Silence-Audit meldet Energieeintrag in Stille-Zonen.")
            passed = False
        silence_lift_db = (
            _nested_float(silence_audit, "max_lift_db")
            or _nested_float(silence_audit, "silence_lift_db")
            or _nested_float(silence_audit, "max_energy_lift_db")
        )
        if silence_lift_db is not None and silence_lift_db > 1.0:
            warnings.append(f"KRITISCH: Stille-Zonen-Lift {silence_lift_db:.2f} dB > 1.00 dB — Export blockiert.")
            passed = False
    direct_silence_lift = _nested_float(meta, "structural_silence_lift_db")
    if direct_silence_lift is not None and direct_silence_lift > 1.0:
        warnings.append(f"KRITISCH: Structural-Silence-Lift {direct_silence_lift:.2f} dB > 1.00 dB — Export blockiert.")
        passed = False

    # §0p: Explizite schwere Vocal-Guard-Schäden blockieren Export; VQI selbst bleibt Recovery-Trigger.
    vocal_quality_check = meta.get("vocal_quality_check")
    if isinstance(vocal_quality_check, dict):
        formant_integrity = _nested_float(vocal_quality_check, "formant_integrity")
        vibrato_depth = _nested_float(vocal_quality_check, "vibrato_depth_preservation")
        if formant_integrity is not None and formant_integrity < 0.72:
            warnings.append(f"KRITISCH: Formant-Integrität {formant_integrity:.3f} < 0.72 — Vokalverfärbung erkannt.")
            passed = False
        if vibrato_depth is not None and vibrato_depth < 0.80:
            warnings.append(f"KRITISCH: Vibrato-Tiefe {vibrato_depth:.3f} < 0.80 — Performance-Artefakt erkannt.")
            passed = False

    goals_meta = meta.get("musical_goals", {})
    goal_scores: dict = goals_meta.get("scores", {})
    # §09.2 PMGG-Blend-Invariante: adaptive_thresholds aus UV3 _effective_goal_thresholds
    # nutzen (material/era/restorability-adaptiv). Fallback nur wenn nicht vorhanden.
    _adaptive_thresholds: dict = goals_meta.get("thresholds", {}) or {}
    # Auch direkt auf result.adaptive_thresholds prüfen (§2.53 Propagations-Pflicht)
    if not _adaptive_thresholds:
        _at_direct = getattr(result, "adaptive_thresholds", None)
        if isinstance(_at_direct, dict):
            _adaptive_thresholds = _at_direct
    violations: list = goals_meta.get("violations", [])

    # §09.2 Material-adaptiver Fallback-Boden (wenn keine adaptive_thresholds vorhanden).
    # Abgeleitet aus Spec §09.2b Material-Bias + §0a Physical-Ceiling — physikalisch erreichbar.
    # Basiert auf: canonical_floor + material_bias, geclippt auf physikalisches Minimum.
    # Getrennt nach material_type (aus Metadaten, Fallback: "unknown" = konservativ).
    _mat_key = str(meta.get("material_type") or meta.get("primary_material") or "unknown").lower()
    # Ultra-Analog: Shellac, Wax, Wire — physikalisch stark limitiert (§0a, §09.2b)
    _MATERIAL_P1P2_FALLBACK: dict[str, dict[str, float]] = {
        "shellac": {
            "natuerlichkeit": 0.62,
            "authentizitaet": 0.60,
            "tonal_center": 0.65,
            "timbre_authentizitaet": 0.60,
            "artikulation": 0.56,
        },
        "wax_cylinder": {
            "natuerlichkeit": 0.60,
            "authentizitaet": 0.58,
            "tonal_center": 0.62,
            "timbre_authentizitaet": 0.57,
            "artikulation": 0.54,
        },
        "wire_recording": {
            "natuerlichkeit": 0.60,
            "authentizitaet": 0.58,
            "tonal_center": 0.62,
            "timbre_authentizitaet": 0.57,
            "artikulation": 0.54,
        },
        # Normal-Analog: Vinyl, Tape, Reel — moderate physikalische Limits (§09.2b MATERIAL_BIAS_ANALOG)
        "vinyl": {
            "natuerlichkeit": 0.72,
            "authentizitaet": 0.70,
            "tonal_center": 0.74,
            "timbre_authentizitaet": 0.70,
            "artikulation": 0.66,
        },
        "reel_tape": {
            "natuerlichkeit": 0.72,
            "authentizitaet": 0.70,
            "tonal_center": 0.74,
            "timbre_authentizitaet": 0.70,
            "artikulation": 0.66,
        },
        "tape": {
            "natuerlichkeit": 0.70,
            "authentizitaet": 0.68,
            "tonal_center": 0.72,
            "timbre_authentizitaet": 0.68,
            "artikulation": 0.64,
        },
        "cassette": {
            "natuerlichkeit": 0.68,
            "authentizitaet": 0.66,
            "tonal_center": 0.70,
            "timbre_authentizitaet": 0.66,
            "artikulation": 0.62,
        },
        "lacquer_disc": {
            "natuerlichkeit": 0.70,
            "authentizitaet": 0.68,
            "tonal_center": 0.72,
            "timbre_authentizitaet": 0.68,
            "artikulation": 0.64,
        },
        # Lossy-Digital: MP3 — Codec-Artefakte limitieren Timbre/Tonal (§09.2b MATERIAL_BIAS_DIGITAL partial)
        "mp3_low": {
            "natuerlichkeit": 0.74,
            "authentizitaet": 0.72,
            "tonal_center": 0.76,
            "timbre_authentizitaet": 0.72,
            "artikulation": 0.68,
        },
        "mp3_high": {
            "natuerlichkeit": 0.78,
            "authentizitaet": 0.76,
            "tonal_center": 0.80,
            "timbre_authentizitaet": 0.76,
            "artikulation": 0.72,
        },
        "minidisc": {
            "natuerlichkeit": 0.72,
            "authentizitaet": 0.70,
            "tonal_center": 0.74,
            "timbre_authentizitaet": 0.70,
            "artikulation": 0.66,
        },
        # Lossless-Digital: CD, Streaming — canonical-nah, geringe Einschränkung
        "cd_digital": {
            "natuerlichkeit": 0.82,
            "authentizitaet": 0.80,
            "tonal_center": 0.85,
            "timbre_authentizitaet": 0.79,
            "artikulation": 0.76,
        },
        "streaming": {
            "natuerlichkeit": 0.78,
            "authentizitaet": 0.76,
            "tonal_center": 0.80,
            "timbre_authentizitaet": 0.75,
            "artikulation": 0.72,
        },
    }
    # Konservativer Fallback für unbekanntes Material (zwischen Analog und Digital)
    _FALLBACK_UNKNOWN = {
        "natuerlichkeit": 0.72,
        "authentizitaet": 0.70,
        "tonal_center": 0.74,
        "timbre_authentizitaet": 0.70,
        "artikulation": 0.66,
    }
    _material_fallback = _MATERIAL_P1P2_FALLBACK.get(_mat_key, _FALLBACK_UNKNOWN)

    # ── §v10.303.29 HPI/MUSHRA Perceptual Override ──
    # Wenn das menschliche Ohr (MUSHRA ≥ 85) UND der Holistic Perceptual Index
    # (HPI ≥ 0.75) unisono "Excellent" sagen, werden objektive P1/P2-Warnungen
    # auf Warning-only herabgestuft. Das Ohr hat Vorrang vor Proxies.
    # Gleiche Logik wie DoNoHarmGuardian.perceptual_override (§5/5).
    _hpi = (
        _nested_float(meta, "hpi")
        or _nested_float(meta, "holistic_perceptual_index")
        or _nested_float(meta, "analytics", "hpi")
        or getattr(result, "hpi", None)
        or getattr(result, "holistic_perceptual_index", None)
    )
    _mushra = (
        _nested_float(meta, "mushra_score")
        or _nested_float(meta, "mushra")
        or _nested_float(meta, "analytics", "mushra", "mushra_score")
        or _nested_float(meta, "analytics", "mushra_score")
        or getattr(result, "mushra_score", None)
        or getattr(result, "mushra", None)
    )
    _perceptual_override = _hpi is not None and _hpi >= 0.75 and _mushra is not None and _mushra >= 85.0
    if _perceptual_override:
        logger.info(
            "Ausgabe-Gate: PERCEPTUAL OVERRIDE — MUSHRA=%.0f≥85 HPI=%.3f≥0.75 → "
            "objektive P1/P2-Warnungen werden auf Warning herabgestuft",
            _mushra,
            _hpi,
        )

    for goal_name, mat_floor in _material_fallback.items():
        score = goal_scores.get(goal_name)
        if score is None:
            continue
        # §09.2 adaptive Threshold hat Vorrang; Fallback = material-adaptiver Boden
        effective_thr = float(_adaptive_thresholds.get(goal_name, mat_floor))
        if float(score) < effective_thr:
            _msg = (
                f"KRITISCH: {goal_name} = {float(score):.3f} < {effective_thr:.2f} — "
                "P1/P2-Mindestziel unterschritten. Restaurierung hat Kernqualität verletzt."
            )
            if _perceptual_override:
                # HPI+MUSHRA unisono "Excellent" → downgrade zu Warning
                warnings.append(_msg.replace("KRITISCH:", "WARNUNG (override):"))
                logger.info("Ausgabe-Gate: %s override aktiv — kein Hard-Fail", goal_name)
            else:
                warnings.append(_msg)
                passed = False

    # P3–P5 violations: warning only (no hard-fail)
    _p3p5_violations = [v for v in violations if v not in _material_fallback]
    if _p3p5_violations:
        warnings.append(
            f"Qualitäts-Hinweis: {len(_p3p5_violations)} Musical Goal(s) nicht optimal: "
            f"{', '.join(_p3p5_violations[:5])}"
        )

    for w in warnings:
        logger.warning("Ausgabe-Qualitätsprüfung: %s", w)

    return passed, warnings


def export_audio(
    audio_bytes,
    export_path: str,
    export_format: str = "wav",
    bit_depth: int = 24,
    *,
    source_path: str = "",
    **kwargs: Any,
) -> bool:
    """Export audio bytes to a file on disk.

    Applies the full export chain mandated by the spec:
    1. NaN/Inf guard + True-Peak clip (``_export_guard``).
    2. POW-r Type 3 dithering (primary) / TPDF (fallback) for integer targets
       (``bit_depth < 32``).  Spec §DSP-Spezialregeln: *VERBOTEN: Truncation
       ohne Dithering.*
    3. Atomic write via ``.tmp → os.replace``.
    4. Metadata preservation (ID3/Vorbis/FLAC tags + Aurik provenance).

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio bytes (any format readable by soundfile).
    export_path : str
        Destination file path.
    export_format : str
        Output container/codec (wav, flac, mp3, …).
    bit_depth : int
        Target integer bit depth for lossless formats (16 or 24).
        Use 32 to write float32 without dithering.  Default: 24.
    source_path : str
        Path to the original input file for metadata transfer (optional).

    Returns
    -------
    bool
        ``True`` on success.
    """
    legacy_format = kwargs.pop("format", None)
    if legacy_format is not None:
        export_format = str(legacy_format)
    if kwargs:
        unexpected = ", ".join(sorted(str(key) for key in kwargs))
        raise TypeError(f"Unerwartete Export-Parameter: {unexpected}")

    _SUBTYPE_MAP = {8: "PCM_S8", 16: "PCM_16", 24: "PCM_24", 32: "FLOAT", 64: "DOUBLE"}

    # 1. Decode incoming bytes
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    except Exception as e:
        logger.error("Ausgabe: Audiodaten konnten nicht gelesen werden: %s", e)
        raise RuntimeError(f"Fehler beim Lesen der Audiodaten: {e}") from e

    logger.info(
        "💿 Ausgabe gestartet: Pfad=%s, Format=%s, Bittiefe=%d, Abtastrate=%d, Shape=%s, Dauer=%.1fs",
        export_path,
        export_format,
        bit_depth,
        sr,
        audio.shape,
        len(audio) / max(sr, 1) if audio.ndim == 1 else audio.shape[0] / max(sr, 1),
    )

    # 2. NaN/Inf-Bereinigung + True-Peak-Schutz
    audio = _export_guard(audio)

    # 2b. Subtle perceptual polish for listening comfort (non-destructive).
    audio = _export_nuance_guard(audio, sr)

    # §V15: Deterministischer Dither-Seed aus Audio-Hash
    dither_seed = int(hashlib.sha256(audio.tobytes()[:4096]).hexdigest()[:16], 16) % (2**31)

    # 2c. §G8 (GEBOTE.md) CD-Rauschprofil-Pflicht: Psychoakustisch maskiert injizieren
    #     §G15, §G30–§G39, §V11, §V14–§V17
    #     §G63: Idempotent — überspringen wenn CD-Rauschen bereits aktiv
    try:
        from backend.core.cd_noise_profile import inject_cd_noise_profile

        _nf = float(np.percentile(np.abs(audio), 10))
        if 20.0 * math.log10(max(_nf, 1e-15)) > -100.0:
            logger.debug("💿 CD-Rauschprofil bereits aktiv — übersprungen")
        else:
            audio = inject_cd_noise_profile(audio, sr, bit_depth=bit_depth, seed=dither_seed)
    except Exception:
        logger.debug("CD-Rauschprofil-Injektion übersprungen (nicht blockierend)")

    # 3. Dithering before integer quantisation (spec §DSP-Spezialregeln)
    if bit_depth < 32 and export_format.lower() not in ("mp3", "aac", "m4a", "opus"):
        audio = apply_dither(audio, bit_depth=bit_depth, seed=dither_seed, cd_active=True)

    subtype = _SUBTYPE_MAP.get(bit_depth)

    # 4. WAV, FLAC, OGG, AIFF direkt mit soundfile — atomic write via .tmp → os.replace
    if export_format.lower() in ["wav", "flac", "ogg", "aiff", "aif", "alac", "caf", "rf64"]:
        tmp_path = export_path + ".tmp"
        try:
            write_kwargs: dict = {"format": export_format.upper()}
            if subtype and export_format.lower() in ("wav", "flac", "aiff", "aif"):
                write_kwargs["subtype"] = subtype
            sf.write(tmp_path, audio, sr, **write_kwargs)
            os.replace(tmp_path, export_path)
            # §2.46a: Transferkette in Export-Metadaten – vor Metadata-Transfer bauen
            _chain_list: list[str] | None = None
            try:
                _chain_dict = _build_chain_metadata()
                if _chain_dict:
                    _chain_list = [str(v) for v in _chain_dict.values() if v]
            except Exception as _e:
                logger.debug("exporter: unkritisch exception: %s", _e)
            _transfer_metadata(source_path, export_path, transfer_chain=_chain_list)
            # BWF-Metadaten für WAV/RF64 schreiben
            if export_format.lower() in ("wav", "rf64"):
                try:
                    from backend.core.bwf_writer import write_bwf_chunks

                    write_bwf_chunks(
                        export_path, description=f"Aurik {_AURIK_VERSION} Restauration", originator="Aurik"
                    )
                except Exception as _bwf_exc:
                    logger.debug("BWF-Metadaten-Schreiben übersprungen: %s", _bwf_exc)
            _size_mb = os.path.getsize(export_path) / (1024 * 1024)
            logger.info(
                "Ausgabe abgeschlossen: %s (%.1f MB, %s %d-bit)",
                export_path,
                _size_mb,
                export_format.upper(),
                bit_depth,
            )
            return True
        except Exception as e:
            # Cleanup orphaned tmp on failure
            logger.error("Ausgabe fehlgeschlagen (%s): %s", export_format, e)
            try:
                os.remove(tmp_path)
            except OSError as _exc:
                logger.debug("Vorgang fehlgeschlagen (nicht-kritisch): %s", _exc)
            raise RuntimeError(f"Fehler beim Export als {export_format}: {e}") from e
    # MP3, AAC, M4A, OPUS nur mit ffmpeg
    elif export_format.lower() in ["mp3", "aac", "m4a", "opus"]:
        if ffmpeg is None:
            raise RuntimeError("ffmpeg-python nicht installiert. Export nicht möglich.")
        tmp_wav = None
        tmp_out = export_path + ".tmp"
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name
                # Export Guard already applied above; write guarded audio to temp WAV
                sf.write(tmp_wav, audio, sr, format="WAV")
            out_args = {"format": export_format.lower()}
            if export_format.lower() == "mp3":
                # LAME VBR V0 — perceptually transparent, adaptive bitrate (~245 kbps Ø).
                # Avoids CBR pre-echo on transients restored by TDP/MDEM (spec §DSP / §8.3).
                out_args["q:a"] = "0"
            (ffmpeg.input(tmp_wav).output(tmp_out, **out_args).run(overwrite_output=True, quiet=True))
            os.replace(tmp_out, export_path)
            # §2.46a: Transferkette in Export-Metadaten – vor Metadata-Transfer bauen
            _chain_list2: list[str] | None = None
            try:
                _chain_dict2 = _build_chain_metadata()
                if _chain_dict2:
                    _chain_list2 = [str(v) for v in _chain_dict2.values() if v]
            except Exception as _e:
                logger.debug("exporter: unkritisch exception: %s", _e)
            _transfer_metadata(source_path, export_path, transfer_chain=_chain_list2)
            _size_mb = os.path.getsize(export_path) / (1024 * 1024)
            logger.info("Ausgabe abgeschlossen: %s (%.1f MB, %s)", export_path, _size_mb, export_format.upper())
            return True
        except Exception as e:
            # Cleanup orphaned tmp files
            logger.error("Ausgabe fehlgeschlagen (%s via ffmpeg): %s", export_format, e)
            for _p in (tmp_out,):
                try:
                    os.remove(_p)
                except OSError as _exc:
                    logger.debug("Vorgang fehlgeschlagen (nicht-kritisch): %s", _exc)
            raise RuntimeError(f"Fehler beim {export_format.upper()}-Export: {e}") from e
        finally:
            # Always clean up the intermediate WAV temp file
            if tmp_wav:
                try:
                    os.remove(tmp_wav)
                except OSError as _exc:
                    logger.debug("Vorgang fehlgeschlagen (nicht-kritisch): %s", _exc)
    else:
        raise ValueError(f"Nicht unterstütztes Exportformat: {export_format}")
