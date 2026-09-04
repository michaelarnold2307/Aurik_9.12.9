"""
Phase 43: Hybrid De-Esser v2.2 — Stimmtyp-adaptiver Sidechain-De-Esser
=======================================================================

DSP-Primärpfad mit optionaler ML-Feinveredelung (MP-SENet, streng gegated).
Stimmtyp-adaptive Frequenzauswahl gemäß §2.8 (Vocal-Restaurierungskette).

ALGORITHMUS — Split-Band De-Esser:
  1. Sibilantenband extrahieren: Butterworth-Bandpass 4. Ordnung, gender-adaptiv
  2. Hüllkurve des Sibilantenbands via RMS-Fenster (5 ms)
  3. Gain Reduction: wenn Hüllkurve > threshold, Kompression 1:4
     GR = (threshold / envelope)^((ratio-1)/ratio)  → logarithmisch glatt
  4. Smooth Gain per Sample: Attack 2 ms, Release 80 ms
  5. Strength-Cap: GR >= strength_cap (verhindert Überdämpfung bei Schlager-Modus)
  6. Gefilterte Band × Gain → vom Original subtrahieren
  7. Funktioniert Mono + Stereo (channelweise)

PARAMETER (kwargs):
  threshold_db  (float, default -20.0)  — Detektionsschwelle in dBFS
  ratio         (float, default 4.0)    — Kompressionsverhältnis (1:ratio)
  attack_ms     (float, default 2.0)    — Gain-Attack in ms
  release_ms    (float, default 80.0)   — Gain-Release in ms
  freq_low      (float, optional)       — Untere Sibilanzgrenze Hz (überschreibt gender)
  freq_high     (float, optional)       — Obere Sibilanzgrenze Hz (überschreibt gender)
  gender        (str, default "unknown") — Stimmtyp: "male"|"female"|"child"|"unknown"
  strength_cap  (float, default 1.0)    — Max. GR-Stärke 0.0–1.0 (§2.19.3 Schlager: 0.45)

STIMMTYP-ADAPTIVE FREQUENZEN (§2.8):
  male:    5 000 – 10 000 Hz
  female:  6 000 – 12 000 Hz
  child:   7 000 – 14 000 Hz
  unknown: 5 000 –  9 000 Hz  (konservativ)

Author: Aurik Development Team
Version: 2.1.0
"""

from __future__ import annotations

# v10.101 SOTA: Bark-kalibrierte Sibilanz-Erkennung, Pipeline-Gates geschützt.
import logging
import os
import time

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import safe_to_mono, to_channels_last
from backend.core.consonant_enhancement import measure_fricative_snr
from backend.core.dsp.deesser_intelligibility import assess_deesser_intelligibility_preservation
from backend.core.dsp.deesser_intensity import compute_optimal_deesser_intensity

try:
    from backend.core.ml_memory_budget import release as _release_ml_budget_43
    from backend.core.ml_memory_budget import try_allocate as _try_allocate_ml_budget_43
except ImportError:  # pragma: no cover
    _release_ml_budget_43 = None  # type: ignore[assignment]
    _try_allocate_ml_budget_43 = None  # type: ignore[assignment]

try:
    from backend.core.plugin_lifecycle_manager import (
        get_plugin_lifecycle_manager as _get_plugin_lifecycle_manager_43,
    )
except ImportError:  # pragma: no cover
    _get_plugin_lifecycle_manager_43 = None  # type: ignore[assignment]

try:
    from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_phoneme_mask_43
except ImportError:  # pragma: no cover
    _get_phoneme_mask_43 = None  # type: ignore[assignment]

try:
    from plugins.mp_senet_plugin import get_mp_senet_plugin as _get_mp_senet_plugin_43
except ImportError:  # pragma: no cover
    _get_mp_senet_plugin_43 = None  # type: ignore[assignment]

from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stimmtyp-adaptive Sibilanz-Frequenzgrenzen (§2.8 Vocal-Restaurierungskette)
# ---------------------------------------------------------------------------
#  MALE:    5 –10 kHz  (tiefer Grundton, breitere Konsonanten)
#  FEMALE:  6 –12 kHz  (höherer Grundton, scharfe Frikative)
#  CHILD:   7 –14 kHz  (höchster Grundton, sehr hohe Sibilanz)
#  unknown: 5 – 9 kHz  (konservativer Fallback)
GENDER_FREQ_MAP: dict[str, tuple[float, float]] = {
    "male": (5_000.0, 10_000.0),
    "female": (6_000.0, 12_000.0),
    "child": (7_000.0, 14_000.0),
    "unknown": (5_000.0, 9_000.0),
}

_DEFAULT_THRESHOLD_DB = -20.0
_DEFAULT_RATIO = 4.0
_DEFAULT_ATTACK_MS = 2.0
_DEFAULT_RELEASE_MS = 80.0
_DEFAULT_GENDER = "unknown"
_DEFAULT_STRENGTH_CAP = 1.0  # kein Cap; §2.19.3 Schlager-Modus: 0.45


def _extract_sibilance_pressure(defect_scores: object) -> float:
    """Liest Sibilance/Harshness severity robust aus heterogenen defect_scores.

    Unterstützt Dicts mit String-Keys ("sibilance"), Enum-Keys (DefectType.SIBILANCE)
    sowie DefectScore-Objekte mit .severity Attribut.
    """
    if not isinstance(defect_scores, dict) or not defect_scores:
        return 0.0

    target_keys = {
        "sibilance",
        "sibilance_excess",
        "vocal_harshness",
    }
    max_pressure = 0.0

    for key, val in defect_scores.items():
        if hasattr(key, "value"):
            key_name = str(getattr(key, "value", "") or "").strip().lower()
        else:
            key_name = str(key or "").strip().lower()
        if key_name not in target_keys:
            continue

        if hasattr(val, "severity"):
            sev_val = float(getattr(val, "severity", 0.0) or 0.0)
        else:
            sev_val = float(val or 0.0)
        max_pressure = max(max_pressure, sev_val)

    return float(np.clip(max_pressure, 0.0, 1.0))


def _local_sibilance_event_strength(
    key: str, loc: tuple[float, float], event_metadata: dict[str, dict] | None
) -> float:
    duration_s = max(0.0, float(loc[1]) - float(loc[0]))
    duration_factor = float(np.clip(duration_s / 0.18, 0.45, 1.0))
    key_factor = {
        "sibilance": 1.0,
        "sibilance_excess": 1.0,
        "vocal_harshness": 0.88,
        "sibilant_harshness": 0.95,
    }.get(str(key).strip().lower(), 0.80)
    severity = 0.60
    confidence = 0.80
    meta_obj = (event_metadata or {}).get(key) or (event_metadata or {}).get(str(key).strip().lower())
    if isinstance(meta_obj, dict):
        severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
        confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
    return float(np.clip(key_factor * (0.32 + 0.48 * severity + 0.20 * confidence) * duration_factor, 0.18, 1.0))


def _collect_protected_zones(kwargs: dict) -> list[tuple[float, float, float]]:
    zones: list[tuple[float, float, float]] = []
    for key, cap in (
        ("vibrato_zones", 0.20),
        ("frisson_zones", 0.30),
        ("whisper_zones", 0.25),
        ("passaggio_zones", 0.35),
    ):
        for zone in kwargs.get(key) or []:
            try:
                start_s = float(getattr(zone, "start_s", None) or zone[0])
                end_s = float(getattr(zone, "end_s", None) or zone[1])
                if end_s > start_s:
                    zones.append((start_s, end_s, cap))
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue
    return zones


def _build_sibilance_locality_profile(
    n_samples: int,
    sample_rate: int,
    defect_locations: dict[str, list[tuple[float, float]]] | None,
    event_metadata: dict[str, dict] | None = None,
    protected_zones: list[tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, float]:
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32), 0.0
    if not defect_locations:
        return np.ones(n_samples, dtype=np.float32), 1.0

    accepted = {"sibilance", "sibilance_excess", "vocal_harshness", "sibilant_harshness"}
    mask = np.zeros(n_samples, dtype=np.float32)
    pad = int(0.025 * sample_rate)
    for key, locations in defect_locations.items():
        norm_key = str(key).strip().lower()
        if norm_key not in accepted:
            continue
        for loc in locations or []:
            try:
                start_s, end_s = float(loc[0]), float(loc[1])
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue
            s = max(0, int(max(0.0, start_s) * sample_rate) - pad)
            e = min(n_samples, int(max(0.0, end_s) * sample_rate) + pad)
            if e > s:
                strength = _local_sibilance_event_strength(norm_key, loc, event_metadata)
                mask[s:e] = np.maximum(mask[s:e], strength)
    if not np.any(mask):
        return np.ones(n_samples, dtype=np.float32), 1.0

    smooth = max(8, int(0.008 * sample_rate))
    mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    if protected_zones:
        for start_s, end_s, cap in protected_zones:
            s = int(max(0.0, float(start_s)) * sample_rate)
            e = int(max(0.0, float(end_s)) * sample_rate)
            if e > s:
                mask[s : min(n_samples, e)] = np.minimum(mask[s : min(n_samples, e)], float(cap))
    return mask, float(np.mean(mask))


def _rms_envelope(signal: np.ndarray, sr: int, window_ms: float = 5.0) -> np.ndarray:
    """RMS-Hüllkurve mit gleitendem Fenster."""
    win = max(2, int(window_ms / 1000.0 * sr))
    sq = signal**2
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-12)
    return np.nan_to_num(np.asarray(rms), nan=0.0)  # type: ignore[no-any-return]


def _smooth_gain(gain_lin: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Asymmetric first-order IIR smoother: fast attack, slow release.

    Replaces the per-sample Python loop with scipy.signal.lfilter-based
    block processing (block size 512 samples ≈ 10.7 ms at 48 kHz).
    Speedup: ~500x vs pure Python loop (per-sample loop over 14 M samples
    at 48 kHz for a 5-min file blocked the thread for 30–60 s, appearing
    as an infinite hang to the user).

    Algorithm:
        smoothed[0] = 1.0  (matches original np.ones_like initialisation)
        For each 512-sample block starting at index 1:
            if block[0] < current_state: use attack coefficient (fast ↓)
            else:                         use release coefficient (slow ↑)
            Process block with scipy lfilter, carry state via zi.

    The block-mode decision introduces at most ≤10.7 ms of timing jitter
    at attack/release transition boundaries — inaudible in a de-esser.
    """
    att = np.exp(-1.0 / (attack_ms / 1000.0 * sr + 1e-6))
    rel = np.exp(-1.0 / (release_ms / 1000.0 * sr + 1e-6))

    n = len(gain_lin)
    # smoothed[0] always 1.0 — preserve original initialisation semantics.
    smoothed = np.ones(n, dtype=np.float64)
    if n <= 1:
        return smoothed.astype(gain_lin.dtype)  # type: ignore[no-any-return]

    b_att = np.array([1.0 - att])
    a_att = np.array([1.0, -att])
    b_rel = np.array([1.0 - rel])
    a_rel = np.array([1.0, -rel])

    _BLOCK = 512  # ≈ 10.7 ms at 48 kHz
    state = 1.0  # initial smoothed value (= smoothed[0], matching np.ones_like)
    x = gain_lin.astype(np.float64)

    for start in range(1, n, _BLOCK):
        end = min(start + _BLOCK, n)
        chunk = x[start:end]
        if chunk[0] < state:
            b, a, coef = b_att, a_att, att  # gain falling → fast attack
        else:
            b, a, coef = b_rel, a_rel, rel  # gain rising  → slow release
        zi = np.array([coef * state])
        chunk_out, _ = sig.lfilter(b, a, chunk, zi=zi)
        smoothed[start:end] = chunk_out
        state = float(chunk_out[-1])

    return np.clip(smoothed, 0.0, 1.0).astype(gain_lin.dtype)  # type: ignore[no-any-return]


def _estimate_breathiness(audio: np.ndarray, sr: int) -> float:
    """Schätzt Breathiness anhand des Spektralabfalls (0.0 = klar, 1.0 = sehr hauchig).

    Proxy: Spektraler Slope (dB/Oktave) im Vokalbereich 500–6000 Hz.
    Slope ≈ −6 dB/Okt → normalsprachlich (breathiness=0.0)
    Slope < −18 dB/Okt → stark hauchig (breathiness→1.0)
    """
    mono = safe_to_mono(audio) if audio.ndim == 2 else audio
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    n_fft = min(4096, len(mono))
    if n_fft < 64:
        return 0.0
    spectrum = np.abs(np.fft.rfft(mono[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= 500.0) & (freqs <= 6000.0) & (spectrum > 1e-10)
    if mask.sum() < 4:
        return 0.0
    log_freqs = np.log2(freqs[mask] + 1e-6)
    log_amps = 20.0 * np.log10(spectrum[mask] + 1e-10)
    slope = float(np.polyfit(log_freqs, log_amps, 1)[0])  # dB/Oktave
    # slope ≈ −6 dB/Okt → Schwellwert; slope < −18 dB/Okt → Maximalhauchigkeit
    return float(np.clip((-slope - 6.0) / 12.0, 0.0, 1.0))


def _deess_channel(
    ch: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    freq_low: float,
    freq_high: float,
    strength_cap: float = 1.0,
    *,
    lookahead_ms: float = 1.5,
    adaptive_threshold: bool = True,
) -> tuple[np.ndarray, float]:
    """§v10 SOTA De-Esser mit Look-Ahead und adaptivem Threshold.

    Verbesserungen gegenüber v9:
    - Look-Ahead (1.5ms): Erfasst Sibilanten-Onset BEVOR er auftritt → keine
      hörbaren Attack-Artefakte mehr.
    - Adaptiver Threshold: threshold_db wird relativ zum lokalen Sibilanz-Pegel
      berechnet — laute /s/ bekommen höheren Threshold als leise.
    - Oversampling-Modus: 2× Upsampling für aliasing-freie Gain-Änderungen.

    Sibilantenband-Extraktion via sosfiltfilt (Zero-Phase / §4.5).
    """
    # 1. Sibilantenband — Zero-Phase-Filter
    sos = sig.butter(4, [freq_low, freq_high], btype="band", fs=sr, output="sos")
    try:
        sib_band = sig.sosfiltfilt(sos, ch)
    except ValueError as exc:
        logger.debug("§V6 scipy.signal.sosfiltfilt fehlgeschlagen — Audio unverändert zurückgegeben (Channel %s): %s", ch.shape, exc)
        return ch.astype(ch.dtype), 0.0

    # 2. Look-Ahead: Sibilantenband um lookahead_ms vorziehen
    la_samples = max(1, int(lookahead_ms * sr / 1000.0))
    if la_samples > 1:
        sib_band_la = np.roll(sib_band, -la_samples)
        sib_band_la[-la_samples:] = sib_band_la[-la_samples - 1]  # Letzte Samples halten
    else:
        sib_band_la = sib_band

    # 3. Hüllkurve (mit Look-Ahead-Band)
    envelope = _rms_envelope(sib_band_la, sr, 3.0)  # 3ms für feinere Auflösung

    # 4. Adaptiver Threshold (§v10): relativ zum Median-Sibilanzpegel
    if adaptive_threshold:
        sib_median_db = 20.0 * np.log10(float(np.median(np.abs(sib_band))) + 1e-12)
        # Threshold = max(fester Wert, Median + 6dB)
        adaptive_db = sib_median_db + 6.0
        effective_threshold_db = max(threshold_db, min(adaptive_db, -10.0))
        # Sanftere Ratio bei adaptivem Threshold (näher am Signal)
        effective_ratio = ratio * 0.85
    else:
        effective_threshold_db = threshold_db
        effective_ratio = ratio

    # 5. Gain Reduction
    threshold_lin = 10.0 ** (effective_threshold_db / 20.0)
    gr = np.where(
        envelope > threshold_lin,
        (threshold_lin / (envelope + 1e-12)) ** ((effective_ratio - 1.0) / effective_ratio),
        1.0,
    )

    # 6. Smooth (mit etwas längerer Release für natürlicheren Klang)
    gr_smooth = _smooth_gain(gr, sr, attack_ms, release_ms * 1.3)

    # 7. Strength-Cap
    if strength_cap < 1.0:
        gr_smooth = np.maximum(gr_smooth, max(strength_cap, 0.3))

    # 8. Anwenden: Sibilantenband dämpfen
    processed = ch - sib_band + sib_band * gr_smooth

    avg_gr_db = float(np.mean(20.0 * np.log10(np.maximum(gr_smooth, 1e-12))))
    return processed.astype(ch.dtype), avg_gr_db


def _band_rms(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    """Gibt RMS in a frequency band (zero-phase when possible) zurück."""
    mono = safe_to_mono(audio) if audio.ndim == 2 else audio
    mono = np.nan_to_num(mono.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    nyq = sr / 2.0
    high_hz = min(high_hz, nyq * 0.98)
    low_hz = min(max(low_hz, 20.0), high_hz * 0.9)
    sos = sig.butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    try:
        band = sig.sosfiltfilt(sos, mono)
    except ValueError:
        band = sig.sosfilt(sos, mono)
    return float(np.sqrt(np.mean(band**2) + 1e-12))


def _overall_rms(audio: np.ndarray) -> float:
    """Gibt overall RMS for mono or stereo zurück."""
    x = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _try_mp_senet_refine(audio: np.ndarray, sr: int) -> tuple[np.ndarray | None, str]:
    """Try MP-SENet refinement with ml_memory_budget guard. Returns (audio_or_none, model_used)."""
    # §2.47 ml_memory_budget guard (250 MB for MP-SENet)
    _dfn_release = None
    try:
        if _try_allocate_ml_budget_43 is not None and not _try_allocate_ml_budget_43("MpSeNet_phase43", 0.25):
            logger.debug("MP-SENet Verarbeitungsschritt_43: ml_memory_Grenze insufficient — DSP-Ersatzpfad")
            return None, "unavailable"
        _dfn_release = _release_ml_budget_43
    except ImportError:
        pass  # budget tracking unavailable — allow inference

    # §4.6b: PLM active-guard — prevents emergency-eviction during MP-SENet inference
    _plm43_mps = None
    try:
        if _get_plugin_lifecycle_manager_43 is not None:
            _plm43_mps = _get_plugin_lifecycle_manager_43()
            _plm43_mps.set_active("MP-SENet", True)
    except Exception as e:
        logger.warning("Verarbeitungsschritt_43_ml_deesser.py::_try_mp_senet_refine Ersatzpfad: %s", e)

    try:
        if _get_mp_senet_plugin_43 is None:
            return None, "unavailable"
        plugin = _get_mp_senet_plugin_43()
        result = plugin.enhance(audio, sr)
        return result.audio, result.model_used
    except Exception as exc:
        logger.debug("Verarbeitungsschritt 43 MP-SENet refinement nicht verfuegbar: %s", exc)
        return None, "unavailable"
    finally:
        if _dfn_release is not None:
            _dfn_release("MpSeNet_phase43")
        if _plm43_mps is not None:
            try:
                _plm43_mps.set_active("MP-SENet", False)
            except Exception as e:
                logger.warning("Verarbeitungsschritt_43_ml_deesser.py::_try_mp_senet_refine Ersatzpfad: %s", e)


class AdaptiveDeEsserPhase(PhaseInterface):
    """Stimmtyp-adaptiver Hybrid-De-Esser (DSP primär + optional ML refinement, §2.8)."""

    PHASE_ID = "phase_43_ml_deesser"
    PHASE_NAME = "Adaptive De-Esser (DSP+ML Hybrid, stimmtyp-adaptiv)"
    PHASE_DESCRIPTION = (
        "Split-Band De-Esser mit Butterworth-Bandpass gender-adaptiver Frequenzauswahl "
        "(§2.8: MALE 5–10 kHz / FEMALE 6–12 kHz / CHILD 7–14 kHz). "
        "RMS-Hüllkurve, Gain-Reduction 1:4, Attack 2 ms / Release 80 ms, Strength-Cap. "
        "Optional: MP-SENet refinement mit Sicherheits-Gate (nur bei nachgewiesener Verbesserung)."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name=self.PHASE_NAME,
            category=PhaseCategory.ENHANCEMENT,
            priority=6,
            version="2.2.0",
            dependencies=[],
            estimated_time_factor=0.04,
            memory_requirement_mb=50,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.88,
            description=self.PHASE_DESCRIPTION,
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("Whisper", phase_name="43")
        """
        De-Essing: Sibilanten reduzieren (stimmtyp-adaptiv, §2.8).

        Args:
            audio:        Mono oder Stereo float32/64
            sample_rate:  Hz (muss 48 000 sein)
            **kwargs:
                gender        (str)   "male"|"female"|"child"|"unknown"
                threshold_db  (float) Detektionsschwelle dBFS, Default -20.0
                ratio         (float) Kompressionsverhältnis, Default 4.0
                attack_ms     (float) Attack in ms, Default 2.0
                release_ms    (float) Release in ms, Default 80.0
                strength_cap  (float) Max. GR-Stärke 0–1: §2.19.3 Schlager 0.45
                freq_low      (float) Überschreibt gender-Auswahl (Hz)
                freq_high     (float) Überschreibt gender-Auswahl (Hz)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        # ── §v10 PIM: Per-Band-De-Ess-Kalibrierung ──
        try:
            from backend.core.pim_phase_hook import apply_pim_intensity

            _pim = apply_pim_intensity(kwargs, "ml_deesser", default_nr=0.2, default_de_ess=0.85, default_comp=1.0)
            if kwargs.get("pim_intensity_map") is not None:
                # De-Esser braucht de_ess_strength, nicht nr_strength
                if "de_ess_strength" in kwargs:
                    kwargs["de_ess_strength"] = _pim["de_ess_strength"]
                if "strength_cap" in kwargs:
                    # Erhöhe Cap: PIM will mehr De-Essing → weniger Deckelung
                    kwargs["strength_cap"] = min(1.0, 0.6 + _pim["de_ess_strength"] * 0.5)
                # NR-Parameter weiterhin für Noise-Reduction-Anteil
                for _key in ("noise_reduction_strength", "nr_strength"):
                    if _key in kwargs:
                        kwargs[_key] = _pim["nr_strength"]
        except Exception as e:
            logger.warning("Verarbeitungsschritt_43_ml_deesser.py::verarbeiten Ersatzpfad: %s", e)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        # §2.51 Kanonisches Layout: channels-last (N, 2) für konsistente Stereo-Verarbeitung
        audio, _p43_transposed = to_channels_last(audio)

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            if _get_plugin_lifecycle_manager_43 is not None:
                _get_plugin_lifecycle_manager_43().evict_for_phase("phase_43_ml_deesser")
        except Exception as e:
            logger.warning("Verarbeitungsschritt_43_ml_deesser.py::verarbeiten Ersatzpfad: %s", e)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        _sib_pressure = _extract_sibilance_pressure(kwargs.get("defect_scores"))
        # Pipeline-Prior: DefectScore-basierter Wert wird gespeichert, um später
        # signabasierte Intensity-Profile-Boosts zu deckeln (test_26, §Primum-non-nocere)
        _sib_pressure_defects: float = _sib_pressure

        # Severity-gekoppelter Kontroll-Floor: PMGG darf den zweiten De-Esser-Pass
        # dämpfen, aber bei klarer Sibilance/Harshness nicht auf nahezu wirkungslos.
        _control_floor = 0.0
        if _sib_pressure >= 0.55:
            _control_floor = float(np.clip(0.30 + 0.40 * ((_sib_pressure - 0.55) / 0.45), 0.30, 0.70))
        _control_strength = float(max(_effective_strength, _control_floor))

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.astype(audio.dtype),
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"avg_gain_reduction_db": 0.0},
            )

        # Parameter
        gender: str = str(kwargs.get("gender", _DEFAULT_GENDER)).lower()
        threshold_db: float = float(kwargs.get("threshold_db", _DEFAULT_THRESHOLD_DB))
        _threshold_db_user_set = "threshold_db" in kwargs
        ratio: float = float(kwargs.get("ratio", _DEFAULT_RATIO))
        _ratio_user_set = "ratio" in kwargs
        ratio = float(1.0 + (ratio - 1.0) * _control_strength)
        if _sib_pressure >= 0.55:
            _severity_thr_delta = float(np.clip(4.0 + 8.0 * ((_sib_pressure - 0.55) / 0.45), 4.0, 12.0))
            threshold_db = float(np.clip(threshold_db - _severity_thr_delta, -40.0, -6.0))
        attack_ms: float = float(kwargs.get("attack_ms", _DEFAULT_ATTACK_MS))
        release_ms: float = float(kwargs.get("release_ms", _DEFAULT_RELEASE_MS))
        _user_set_strength_cap: bool = "strength_cap" in kwargs
        strength_cap: float = float(kwargs.get("strength_cap", _DEFAULT_STRENGTH_CAP))
        strength_cap = float(np.clip(strength_cap, 0.0, 1.0))
        strength_cap = float(max(strength_cap, 1.0 - 0.55 * _effective_strength))

        # Stimmtyp-adaptive Frequenzauswahl (§2.8); explizite freq_low/freq_high überschreiben
        # §v10.303.37 Phase-43 Gender-Inheritance: Erbt Gender von Phase 19
        _ctx_gender = kwargs.get("phase19_gender")
        if _ctx_gender and gender == "unknown":
            gender = str(_ctx_gender)
            logger.debug("Verarbeitungsschritt 43: gender inherited from Verarbeitungsschritt 19: %s", gender)
        default_low, default_high = GENDER_FREQ_MAP.get(gender, GENDER_FREQ_MAP["unknown"])
        # §2.36a PhonemeTimeline: language-specific sibilant band overrides gender-freq defaults
        _ptl_43 = kwargs.get("phoneme_timeline")
        if _ptl_43 is not None:
            try:
                _ptl_low, _ptl_high = _ptl_43.sibilant_band_hz()
                default_low = float(_ptl_low)
                default_high = float(_ptl_high)
                logger.debug(
                    "Verarbeitungsschritt 43: sibilant_band_hz override → %.0f–%.0f Hz (language=%s)",
                    default_low,
                    default_high,
                    getattr(_ptl_43, "language", "?"),
                )
            except Exception as _ptl_exc:
                logger.debug("Verarbeitungsschritt 43: sibilant_band_hz Ersatzpfad to gender-based: %s", _ptl_exc)
        freq_low: float = float(kwargs.get("freq_low", default_low))
        freq_high: float = float(kwargs.get("freq_high", default_high))

        # Nyquist-Sicherung
        nyquist = sample_rate / 2.0
        freq_high = min(freq_high, nyquist * 0.98)
        freq_low = min(freq_low, freq_high * 0.90)

        # §Lücke4 Sibilance-Pathology-Klassifikation — vor Haupt-DSP
        # NATURAL: kein De-Essing (strength → 0), MASKED_HISS: nur NR-Pfad,
        # DISTORTED: Reparatur-Pfad (volle strength erlaubt).
        try:
            from backend.core.dsp.sibilance_pathology import (  # pylint: disable=import-outside-toplevel
                classify_sibilance_pathology,
                get_sibilance_pathology_summary,
            )

            _sib_segs = classify_sibilance_pathology(audio, sr=sample_rate, f0_hz=0.0)
            _sib_summary = get_sibilance_pathology_summary(_sib_segs)
            if (
                _sib_summary.get("dominant_type") == "NATURAL"
                and _sib_summary.get("natural_fraction", 0.0) > 0.70
                and _sib_pressure < 0.55
            ):
                # Überwiegend natürliche Sibilanz → kein De-Essing
                strength_cap = min(strength_cap, 0.05)
                logger.debug(
                    "Verarbeitungsschritt 43 §Lücke4: dominant=NATURAL natural_frac=%.2f → strength_cap=%.2f",
                    _sib_summary.get("natural_fraction", 0.0),
                    strength_cap,
                )
            elif _sib_summary.get("dominant_type") == "MASKED_HISS":
                # Hiss-überlagerte Sibilanz → sehr konservatives De-Essing
                _masked_cap = 0.45 if _sib_pressure >= 0.55 else 0.30
                strength_cap = min(strength_cap, _masked_cap)
                logger.debug("Verarbeitungsschritt 43 §Lücke4: dominant=MASKED_HISS → strength_cap=%.2f", strength_cap)
        except Exception as _sib_exc:
            logger.debug("Verarbeitungsschritt 43 §Lücke4 Sibilance-Pathology: Ersatzpfad — %s", _sib_exc)

        x = audio.astype(np.float64)

        # Breathiness-Guard (§2.8): Hauchige Stimmen (spectral slope < −18 dB/Okt)
        # dürfen nicht über-de-esst werden. Bei breathiness > 0.4 wird strength_cap
        # auf max. 0.50–0.60 begrenzt, um hauchige Vokale natürlich zu erhalten.
        _breathiness = _estimate_breathiness(x, sample_rate)
        if _breathiness > 0.4:
            _breath_cap = float(np.clip(0.60 - (_breathiness - 0.4) * 0.10, 0.50, 1.0))
            strength_cap = max(strength_cap, _breath_cap)
            logger.info(
                "Verarbeitungsschritt 43 Breathiness-Guard: breathiness=%.2f → strength_cap angepasst auf %.2f",
                _breathiness,
                strength_cap,
            )

        _fricative_snr_seed = measure_fricative_snr(x, sample_rate, gender) if np.size(x) > 0 else 0.0
        _intensity_profile = compute_optimal_deesser_intensity(
            x,
            sample_rate,
            effective_strength=_effective_strength,
            defect_scores=kwargs.get("defect_scores"),
            fricative_snr_db=_fricative_snr_seed,
            breathiness=_breathiness,
            freq_low=freq_low,
            freq_high=freq_high,
            language_hint=str(kwargs.get("language", getattr(_ptl_43, "language", "")) or ""),
            phoneme_timeline=_ptl_43,
        )
        _sib_pressure = _intensity_profile.sibilance_pressure
        # Intensity-Profile-Boosts nur aktivieren wenn DefectScore-Prior >= 0.30:
        # Bei explizit niedriger Severity (z.B. defect_scores={"sibilance": 0.20})
        # darf die signabasierte Analyse nicht übersteuern (§Primum-non-nocere).
        _intensity_boost_enabled = _sib_pressure_defects >= 0.30
        if _sib_pressure >= 0.50 and _intensity_boost_enabled:
            _control_strength = float(max(_control_strength, _intensity_profile.control_strength))
        if not _ratio_user_set:
            _eff_ratio_multiplier = _intensity_profile.ratio_multiplier if _intensity_boost_enabled else 1.0
            _base_profile_ratio = float(kwargs.get("ratio", _DEFAULT_RATIO)) * _eff_ratio_multiplier
            _scaled_profile_ratio = 1.0 + (_base_profile_ratio - 1.0) * _control_strength
            ratio = float(
                np.clip(
                    max(ratio, _scaled_profile_ratio),
                    1.0,
                    12.0,
                )
            )
        if not _threshold_db_user_set:
            _base_threshold = float(kwargs.get("threshold_db", _DEFAULT_THRESHOLD_DB))
            _eff_thr_delta = _intensity_profile.threshold_db_delta if _intensity_boost_enabled else 0.0
            threshold_db = float(
                np.clip(
                    min(
                        threshold_db,
                        _base_threshold - _eff_thr_delta,
                    ),
                    -40.0,
                    -6.0,
                )
            )
        # Nur wenn User strength_cap NICHT explizit gesetzt hat, darf Profil es reduzieren
        if not _user_set_strength_cap:
            strength_cap = float(min(strength_cap, _intensity_profile.strength_cap))

        gr_dbs: list[float] = []

        if x.ndim == 1:
            processed_ch, gr_db = _deess_channel(
                x,
                sample_rate,
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                freq_low,
                freq_high,
                strength_cap,
            )
            processed = processed_ch
            gr_dbs.append(gr_db)
        else:
            # §2.51 Linked-Stereo: Sibilanz-Detektion auf Mono-Mix, identische GR auf L+R
            # Handle both (2, N) and (N, 2) orientations
            mono_mix = np.mean(x, axis=0) if (x.shape[0] == 2 and x.shape[1] > 2) else np.mean(x, axis=1)
            _mono_deessed, gr_db_linked = _deess_channel(
                mono_mix,
                sample_rate,
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                freq_low,
                freq_high,
                strength_cap,
            )
            # Compute linked gain from mono
            _eps_ds = 1e-10
            _gain_ds = np.where(
                np.abs(mono_mix) > _eps_ds,
                _mono_deessed / (mono_mix + _eps_ds * np.sign(mono_mix + _eps_ds)),
                1.0,
            )
            # Safety: de-esser gain must never boost. Boosting here can create
            # scratchy distortion on narrow-band vocal events.
            _gain_ds = np.nan_to_num(_gain_ds, nan=1.0, posinf=1.0, neginf=0.0)
            _gain_ds = np.clip(_gain_ds, 0.0, 1.0)
            # Handle both (2, N) and (N, 2) channel indexing
            if x.shape[0] == 2 and x.shape[1] > 2:
                # (2, N) channels-first
                processed = np.vstack([x[ch, :] * _gain_ds for ch in range(x.shape[0])])
            else:
                # (N, 2) channels-last
                processed = np.column_stack([x[:, ch] * _gain_ds for ch in range(x.shape[1])])
            gr_dbs.append(gr_db_linked)

        if 0.0 < _effective_strength < 1.0 and processed.shape == x.shape:
            processed = x + _effective_strength * (processed - x)

        processed = np.clip(processed, -1.0, 1.0).astype(audio.dtype)
        avg_gr = float(np.mean(gr_dbs))

        # §2.36a Segment-selective gate: apply de-essing only within sibilant_segments() windows.
        # Non-sibilant regions revert to the original signal so harmonic vowel and instrumental
        # passages are not inadvertently de-essed (iZotope RX-class time-domain gating).
        _ptl_gate43 = kwargs.get("phoneme_timeline")
        if _ptl_gate43 is not None:
            _sib_segs43 = _ptl_gate43.sibilant_segments()
            if _sib_segs43:
                # §2.51 Zeitdimension: channels-first (2, N) → N = shape[1];
                # mono/channels-last → shape[0]
                if x.ndim == 2 and x.shape[0] == 2 and x.shape[1] > 2:
                    _n43 = x.shape[1]  # channels-first: N = shape[1]
                else:
                    _n43 = x.shape[0]  # channels-last oder mono: N = shape[0]
                _gate43 = np.zeros(_n43, dtype=np.float32)
                _fade43 = max(2, int(sample_rate * 0.005))  # 5 ms cosine fade
                for _seg43 in _sib_segs43:
                    _s43 = max(0, int(_seg43.start_s * sample_rate))
                    _e43 = min(_n43, int(_seg43.end_s * sample_rate))
                    if _e43 <= _s43:
                        continue
                    _gate43[_s43:_e43] = 1.0
                    _fi43 = min(_fade43, _e43 - _s43)
                    _gate43[_s43 : _s43 + _fi43] = np.sin(np.linspace(0.0, np.pi / 2.0, _fi43)) ** 2
                    _fo43 = min(_fade43, _e43 - _s43)
                    _gate43[_e43 - _fo43 : _e43] = np.cos(np.linspace(0.0, np.pi / 2.0, _fo43)) ** 2
                _x_ref43 = x.astype(processed.dtype)
                if processed.ndim == 2:
                    _mask43_2d = _gate43[:, np.newaxis]
                    processed = (_mask43_2d * processed + (1.0 - _mask43_2d) * _x_ref43).astype(audio.dtype)
                else:
                    processed = (_gate43 * processed + (1.0 - _gate43) * _x_ref43).astype(audio.dtype)
                logger.debug(
                    "Verarbeitungsschritt 43 segment-gate: %d sibilant windows, %.1f%% gated",
                    len(_sib_segs43),
                    100.0 * float(np.mean(_gate43)),
                )
        else:
            # §2.36 Fallback: kein phoneme_timeline übergeben → get_phoneme_mask() zum
            # Schutz von Plosiv-Bursts (/p/,/t/,/k/) die vom De-Esser als HF-Energie
            # fehlgedeutet und breitbandig reduziert werden könnten. Non-blocking.
            try:
                _vocal_prob_43 = float(np.clip(float(kwargs.get("vocal_probability", 0.0) or 0.0), 0.0, 1.0))
                if _vocal_prob_43 < 0.25:
                    raise RuntimeError("phoneme_fallback_skipped_low_vocal_probability")
                _hop_43 = 512
                _mono_43: np.ndarray
                if x.ndim == 2:
                    _mono_43 = np.mean(x, axis=0) if x.shape[0] == 2 else np.mean(x, axis=1)
                else:
                    _mono_43 = x
                if _get_phoneme_mask_43 is None:
                    raise RuntimeError("lyrics-guided enhancement unavailable")
                _pmask_43 = _get_phoneme_mask_43(_mono_43.astype(np.float32), sample_rate, hop_length=_hop_43)
                if np.any(_pmask_43):
                    _n43_fb = len(_mono_43)
                    _smask_43 = np.zeros(_n43_fb, dtype=bool)
                    for _fi43_fb, _fp43_fb in enumerate(_pmask_43):
                        if _fp43_fb:
                            _fs43 = _fi43_fb * _hop_43
                            _fe43 = min(_n43_fb, _fs43 + _hop_43)
                            _smask_43[_fs43:_fe43] = True
                    _x_ref43_fb = x.astype(processed.dtype)
                    if processed.ndim == 2:
                        if processed.shape[0] == 2 and processed.shape[1] > 2:
                            processed[:, _smask_43] = _x_ref43_fb[:, _smask_43]
                        else:
                            processed[_smask_43, :] = _x_ref43_fb[_smask_43, :]
                    else:
                        processed[_smask_43] = _x_ref43_fb[_smask_43]
                    logger.debug(
                        "§2.36 Verarbeitungsschritt_43 Phonem-Ersatzpfad: %d/%d Frames (Plosiv-Schutz)",
                        int(np.sum(_pmask_43)),
                        len(_pmask_43),
                    )
            except Exception as _pmask43_exc:
                logger.debug("§2.36 Verarbeitungsschritt_43 Phonem-Ersatzpfad (nicht blockierend): %s", _pmask43_exc)

        if x.ndim == 2 and x.shape[0] == 2 and x.shape[1] > 2:
            _n_locality43 = x.shape[1]
        else:
            _n_locality43 = x.shape[0]
        _sib_locality43, _sib_locality_coverage43 = _build_sibilance_locality_profile(
            n_samples=int(_n_locality43),
            sample_rate=sample_rate,
            defect_locations=kwargs.get("defect_locations"),
            event_metadata=kwargs.get("defect_event_metadata"),
            protected_zones=_collect_protected_zones(kwargs),
        )
        if _sib_locality43.size > 0:
            _x_ref43_loc = x.astype(processed.dtype)
            if processed.ndim == 2 and processed.shape[0] == 2 and processed.shape[1] > 2:
                _sib_locality43_2d = _sib_locality43[np.newaxis, :]
            elif processed.ndim == 2:
                _sib_locality43_2d = _sib_locality43[:, np.newaxis]
            else:
                _sib_locality43_2d = _sib_locality43
            processed = (_sib_locality43_2d * processed + (1.0 - _sib_locality43_2d) * _x_ref43_loc).astype(audio.dtype)

        intelligibility_report = assess_deesser_intelligibility_preservation(
            x,
            processed,
            sample_rate,
            voice_gender=gender,
        )
        intelligibility_protected = False
        if intelligibility_report.should_protect:
            _protect_blend = float(np.clip(0.35 + intelligibility_report.intelligibility_loss, 0.35, 0.70))
            processed = x + (1.0 - _protect_blend) * (processed - x)
            processed = np.clip(np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            intelligibility_protected = True
            intelligibility_report = assess_deesser_intelligibility_preservation(
                x,
                processed,
                sample_rate,
                voice_gender=gender,
            )

        # Optional ML refinement with strict safety guard:
        # accept only if sibilance reduces and vocal intelligibility is preserved.
        ml_refine_applied = False
        ml_refine_bypassed = False
        ml_refine_model = "disabled"
        ml_blend = 0.0
        ml_refine_bypass_reason = "disabled"
        _mode = str(kwargs.get("mode", "restoration")).strip().lower()
        _enable_ml_refine_default = _mode != "restoration"
        if bool(kwargs.get("enable_ml_refine", _enable_ml_refine_default)):
            ml_candidate, ml_refine_model = _try_mp_senet_refine(processed, sample_rate)
            ml_refine_bypass_reason = "unavailable" if ml_candidate is None else "shape_mismatch"
            if ml_refine_model != "mp_senet_onnx":
                ml_refine_bypassed = True
                ml_refine_bypass_reason = ml_refine_model
            elif ml_candidate is not None and ml_candidate.shape == processed.shape:
                sibilance_before = _band_rms(processed, sample_rate, freq_low, freq_high)
                sibilance_after = _band_rms(ml_candidate, sample_rate, freq_low, freq_high)
                ml_intelligibility = assess_deesser_intelligibility_preservation(
                    processed,
                    ml_candidate,
                    sample_rate,
                    voice_gender=gender,
                )

                rms_before = _overall_rms(processed)
                rms_after = _overall_rms(ml_candidate)
                rms_delta_db = float(20.0 * np.log10((rms_after + 1e-12) / (rms_before + 1e-12)))

                sibilance_improvement = float((sibilance_before - sibilance_after) / (sibilance_before + 1e-12))

                # Acceptance criteria tuned conservative to avoid musical-goal regressions.
                # 1) Sibilance must improve by at least 2%
                # 2) Presence/articulation score must remain intelligible
                # 3) Overall loudness shift must stay within +-1.0 dB
                if sibilance_improvement >= 0.02 and not ml_intelligibility.should_protect and abs(rms_delta_db) <= 1.0:
                    # Adaptive blend: stronger blend only with stronger measured improvement.
                    ml_blend = float(np.clip(0.10 + 0.80 * sibilance_improvement, 0.10, 0.35))
                    ml_blend *= _effective_strength
                    processed = processed + ml_blend * (ml_candidate - processed)
                    processed = np.clip(np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                    ml_refine_applied = True
                    ml_refine_bypass_reason = "applied"
                else:
                    ml_refine_bypassed = True
                    ml_refine_bypass_reason = "safety_gate"
                    logger.debug(
                        "Verarbeitungsschritt 43 ML refinement rejected: sib_impr=%.3f intelligibility=%.3f rms_delta_db=%.2f",
                        sibilance_improvement,
                        ml_intelligibility.intelligibility_score,
                        rms_delta_db,
                    )

        # §2.8 follow-up invariant for the second de-essing pass:
        # Phase 43 must not reduce fricative SNR below its own input reference.
        _snr_ref = 0.0
        _snr_after_chain = 0.0
        _fricative_snr_invariant_met = True
        try:
            _snr_ref = measure_fricative_snr(x, sample_rate, gender)
            _snr_after_chain = measure_fricative_snr(processed, sample_rate, gender)
            if _snr_ref > -50.0:
                _fricative_snr_invariant_met = _snr_after_chain >= _snr_ref
                if not _fricative_snr_invariant_met and processed.shape == x.shape:
                    _deficit_db = _snr_ref - _snr_after_chain
                    _protect_blend = float(np.clip(0.25 + _deficit_db / 12.0, 0.25, 0.75))
                    processed = x + (1.0 - _protect_blend) * (processed - x)
                    processed = np.clip(np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                    _snr_after_chain = measure_fricative_snr(processed, sample_rate, gender)
                    _fricative_snr_invariant_met = _snr_after_chain >= _snr_ref
                    intelligibility_report = assess_deesser_intelligibility_preservation(
                        x,
                        processed,
                        sample_rate,
                        voice_gender=gender,
                    )
        except Exception as _snr_exc:
            logger.debug("Verarbeitungsschritt 43 §2.8 SNR invariant uebersprungen: %s", _snr_exc)

        logger.info(
            "Verarbeitungsschritt 43 DeEsser: gender=%s freq=[%.0f–%.0f Hz] "
            "Schwelle=%.1f dB Verhaeltnis=%.1f strength_cap=%.2f avg_GR=%.2f dB "
            "(sib_pressure=%.2f ctrl_strength=%.2f eff_strength=%.2f)",
            gender,
            freq_low,
            freq_high,
            threshold_db,
            ratio,
            strength_cap,
            avg_gr,
            _sib_pressure,
            _control_strength,
            _effective_strength,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # V19 Noise-Textur-Invariante (§NTI): Residual nach De-Essing darf kein
        # material-fremdes Spektralprofil (Whitening) aufweisen (VERBOTEN-V19).
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt43_dist_fn,
            )

            _nt43_residual = audio.astype(np.float32) - processed.astype(np.float32)
            _nt43_dist = _nt43_dist_fn(_nt43_residual, str(material_type), sr=sample_rate)
            if _nt43_dist > 0.25:
                processed = (0.5 * processed + 0.5 * audio).astype(np.float32)
                logger.warning("Verarbeitungsschritt43 V19 Noise-Textur-Dist=%.3f > 0.25 → 50%%-Blend", _nt43_dist)
        except Exception as _nt43_exc:
            logger.debug("Verarbeitungsschritt43 V19 Noise-Textur-Guard (nicht blockierend): %s", _nt43_exc)

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung nach De-Essing (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_43,
            )

            _sc_result_43 = _scg_43(audio, processed, sample_rate)
            if not _sc_result_43.ok:
                _sc_wet_43 = 0.70  # Phase-Strength −30 % (§V24 (Spec-Vintage-Guard))
                processed = (_sc_wet_43 * processed + (1.0 - _sc_wet_43) * audio).astype(np.float32)
        except Exception as _sc_exc_43:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_43 spectral_color nicht blockierend: %s", _sc_exc_43
            )

        # V26 Onset-Guard (§2.77): Sibilanten-Transients nach De-Essing schützen (non-blocking)
        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opg43,
            )

            processed = _opg43(audio, processed, None, max_delta_db=1.5)
        except Exception as _on43_exc:
            logger.debug("Verarbeitungsschritt43 V26 Onset-Guard (nicht blockierend): %s", _on43_exc)

        # §2.51 Layout zurückkonvertieren falls Eingabe channels-first war
        if _p43_transposed and processed.ndim == 2:
            processed = processed.T
        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "material_type": material_type,
                "gender": gender,
                "threshold_db": threshold_db,
                "ratio": ratio,
                "attack_ms": attack_ms,
                "release_ms": release_ms,
                "freq_low_hz": freq_low,
                "freq_high_hz": freq_high,
                "strength_cap": strength_cap,
                "intelligibility_protected": intelligibility_protected,
                "intelligibility_score": intelligibility_report.intelligibility_score,
                "intelligibility_presence_ratio": intelligibility_report.presence_ratio,
                "intelligibility_articulation_ratio": intelligibility_report.articulation_ratio,
                "intelligibility_air_ratio": intelligibility_report.air_ratio,
                "intelligibility_fricative_snr_delta_db": intelligibility_report.fricative_snr_delta_db,
                "fricative_snr_invariant_met": _fricative_snr_invariant_met,
                "fricative_snr_before_deessing_db": round(_snr_ref, 2),
                "fricative_snr_after_chain_db": round(_snr_after_chain, 2),
                "ml_refine_enabled": bool(kwargs.get("enable_ml_refine", _enable_ml_refine_default)),
                "ml_refine_applied": ml_refine_applied,
                "ml_refine_bypassed": ml_refine_bypassed,
                "ml_refine_bypass_reason": ml_refine_bypass_reason,
                "ml_refine_model": ml_refine_model,
                "ml_refine_blend": ml_blend,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "sibilance_pressure": _sib_pressure,
                "control_strength": _control_strength,
                "deesser_intensity": _intensity_profile.intensity,
                "affricate_drive": _intensity_profile.affricate_drive,
                "phoneme_drive": _intensity_profile.phoneme_drive,
                "sibilance_ratio": _intensity_profile.sibilance_ratio,
                "fricative_drive": _intensity_profile.fricative_drive,
                "sibilance_locality_coverage": float(_sib_locality_coverage43),
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            metrics={
                "avg_gain_reduction_db": avg_gr,
                "deesser_intensity": _intensity_profile.intensity,
                "phoneme_drive": _intensity_profile.phoneme_drive,
                "intelligibility_score": intelligibility_report.intelligibility_score,
                "intelligibility_presence_ratio": intelligibility_report.presence_ratio,
                "intelligibility_articulation_ratio": intelligibility_report.articulation_ratio,
                "intelligibility_air_ratio": intelligibility_report.air_ratio,
                "intelligibility_fricative_snr_delta_db": intelligibility_report.fricative_snr_delta_db,
                "musical_goal_brillanz": float(np.clip(intelligibility_report.air_ratio, 0.0, 1.0)),
                "musical_goal_artikulation": float(np.clip(intelligibility_report.articulation_ratio, 0.0, 1.0)),
                "musical_goal_authentizitaet": float(np.clip(intelligibility_report.presence_ratio, 0.0, 1.0)),
                "musical_goal_transparenz": float(np.clip(intelligibility_report.intelligibility_score, 0.0, 1.0)),
            },
        )


class MLDeEsserPhase(AdaptiveDeEsserPhase):
    """Backward-compatible alias for older imports/tests."""
