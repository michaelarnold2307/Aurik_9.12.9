"""
DSP-Applier-Modul für AURIK: Wendet eine konfigurierbare Kette von DSP-Effekten
(EQ, Kompressor, Limiter, Enhancer etc.) auf Audiodaten an.
SOTA-Architektur, modular und erweiterbar.

Kanonisch zusammengeführt aus backend/_dsp_applier.py (Phase-Skip-Gate, Modul-Dispatcher)
und backend/core/regulator/_dsp_applier.py (Biquad-EQ, Kompressor, Limiter, Enhancer).
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy.signal

from backend.core.audio_utils import safe_filtfilt  # §v10.101 padlen-guard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase-Skip-Gate (portiert aus backend/_dsp_applier.py)
# Module werden revertiert, wenn sie den SNR-Wert um mehr als den Schwellwert senken.
# ---------------------------------------------------------------------------
_SKIP_GATE_THRESHOLD_DB: float = -0.2

_ALWAYS_APPLY: frozenset = frozenset(
    {
        "DCBlocker",  # Muss immer zuerst (DC-Offset kann andere Module beschädigen)
        "MultibandLimiter",  # Clipping-Schutz am Ende ist immer sicher
        "Dither",  # Quantisierungsrauschen-Kompensation
        "TransientProtectionGuard",  # Schutz, kein destruktiver Eingriff
    }
)

# §v10.1 Mode-aware Plugin-Routing
_RESTORATION_BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "HarmonicExciterStudio",
        "StereoEnhancer",
        "SpeakerEnhancement",
    }
)
_current_processing_mode: str = "restoration"


def set_dsp_processing_mode(mode: str) -> None:
    global _current_processing_mode
    _current_processing_mode = str(mode or "restoration")


def _compute_snr_db(audio: np.ndarray) -> float:
    """Schätzt SNR des Audio-Signals in dB via spektraler Flachheit (Wiener-Entropie)."""
    x = np.ravel(audio).astype(np.float64)
    if x.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(x**2)))
    if rms < 1e-10:
        return 0.0
    frame_len = min(2048, len(x))
    frames = [x[i : i + frame_len] for i in range(0, len(x) - frame_len, frame_len // 2)]
    if not frames:
        return 0.0
    power_frames = np.maximum(np.array([np.mean(f**2) for f in frames]), 1e-20)
    geo_mean = float(np.exp(np.mean(np.log(power_frames))))
    arith_mean = float(np.mean(power_frames))
    spectral_flatness = max(1e-10, min(geo_mean / arith_mean if arith_mean > 1e-20 else 0.0, 1.0))
    return float(-10.0 * np.log10(spectral_flatness))


# ---------------------------------------------------------------------------
# EQ — Parametrischer Mehr-Band-Equalizer
# ---------------------------------------------------------------------------
def eq(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    """
    Parametrischer Multi-Band-EQ (Audio-EQ-Cookbook Biquad-Peaking-Filter,
    R. Bristow-Johnson, https://webaudio.github.io/Audio-EQ-Cookbook/).

    Jedes Band ist ein 2nd-order Peaking-EQ mit exakt dem angegebenen Gain_dB
    bei der Mittenfrequenz und Einheits-Gain abseits der Bandbreite.

    params:
        bands: Liste von Dicts mit Schlüsseln
            - freq     (Hz, float)   — Mittenfrequenz
            - gain_db  (dB, float)   — Verstärkung (positiv = Boost, negativ = Cut)
            - q        (float, >0)   — Gütefaktor (default: 1.0)
        Beispiel: [{"freq": 120, "gain_db": -6.0, "q": 0.7},
                   {"freq": 8000, "gain_db": 3.0,  "q": 1.5}]

    Falls keine Bänder übergeben: Audio unverändert.
    """
    bands = params.get("bands", [])
    if not bands:
        return audio

    audio_f64 = audio.astype(np.float64)
    nyq = sr / 2.0

    for band in bands:
        freq = float(band.get("freq", 1000))
        gain_db = float(band.get("gain_db", 0.0))
        q = max(float(band.get("q", 1.0)), 0.01)

        if freq <= 0 or freq >= nyq or gain_db == 0.0:
            continue

        # --- Bristow-Johnson Peaking-EQ Biquad ---
        # A = sqrt(10^(dB/20))
        # w0 = 2π·f / sr
        # alpha = sin(w0) / (2·Q)
        # b0 =  1 + alpha*A    b1 = -2·cos(w0)   b2 = 1 - alpha*A
        # a0 =  1 + alpha/A    a1 = -2·cos(w0)   a2 = 1 - alpha/A
        try:
            w0 = 2.0 * np.pi * freq / sr
            A = 10.0 ** (gain_db / 40.0)  # sqrt(linear gain)
            alpha = np.sin(w0) / (2.0 * q)

            b0 = 1.0 + alpha * A
            b1 = -2.0 * np.cos(w0)
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = b1  # gleich für Peaking
            a2 = 1.0 - alpha / A

            # SOS-Format: [b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]
            sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
            audio_f64 = scipy.signal.sosfilt(sos, audio_f64)
        except Exception:
            continue  # Numerisch instabiles Band überspringen

    return np.clip(audio_f64, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Compressor — Dynamikkompression (Peak-Sidechain, Attack/Release-Hüllkurve)
# ---------------------------------------------------------------------------
def compressor(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    """
    Dynamisch-komprimierender Prozessor mit Sidechain-Hüllkurve.

    params:
        threshold_db  (float, default -20.0) — Schwellwert in dBFS
        ratio         (float, default 4.0)   — Kompressionsverhältnis (n:1)
        attack_ms     (float, default 10.0)  — Einschwingzeit in ms
        release_ms    (float, default 100.0) — Ausschwingzeit in ms
        makeup_db     (float, default 0.0)   — Makeup-Gain in dB
        knee_db       (float, default 3.0)   — Soft-Knee-Breite in dB
    """
    threshold_db = float(params.get("threshold_db", -20.0))
    ratio = max(float(params.get("ratio", 4.0)), 1.0)
    attack_ms = max(float(params.get("attack_ms", 10.0)), 0.01)
    release_ms = max(float(params.get("release_ms", 100.0)), 0.01)
    makeup_db = float(params.get("makeup_db", 0.0))
    knee_db = max(float(params.get("knee_db", 3.0)), 0.0)

    makeup_lin = 10 ** (makeup_db / 20.0)
    knee_half = knee_db / 2.0

    # Zeitkonstanten → Filterkoeffizienten (1-pol RC-Filter)
    tau_attack = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    tau_release = np.exp(-1.0 / (release_ms * 1e-3 * sr))

    audio_f64 = audio.astype(np.float64)
    n = len(audio_f64)

    # Sidechain: gleitender Gleichrichter (Peak-Detektor)
    envelope = np.zeros(n)
    env_val = 0.0
    for i in range(n):
        x_abs = abs(audio_f64[i])
        if x_abs > env_val:
            env_val = tau_attack * env_val + (1 - tau_attack) * x_abs
        else:
            env_val = tau_release * env_val + (1 - tau_release) * x_abs
        envelope[i] = env_val

    # Gain-Berechnung mit Soft-Knee
    gain = np.ones(n)
    eps = 1e-30
    for i in range(n):
        env_db = 20 * np.log10(envelope[i] + eps)
        above = env_db - threshold_db
        if above < -knee_half:
            # Unterhalb des Knee: keine Kompression
            cs = 1.0
        elif above <= knee_half:
            # Im Soft-Knee-Bereich: sanfter Übergang
            slope = (1.0 - 1.0 / ratio) * ((above + knee_half) ** 2) / (2 * knee_db) if knee_db > 0 else 0.0
            cs = 1.0 - slope / max(above + knee_half, eps)
            # Gain-Reduktion (dB)
            gain_db_reduction = -((above + knee_half) ** 2) / (2 * knee_db) * (1 - 1 / ratio) if knee_db > 0 else 0.0
            cs = 10 ** (gain_db_reduction / 20.0) if gain_db_reduction < 0 else 1.0
        else:
            # Oberhalb des Knee: volle Kompression
            gain_db_reduction = (threshold_db + above / ratio) - env_db
            cs = 10 ** (gain_db_reduction / 20.0)
        gain[i] = cs

    result = audio_f64 * gain * makeup_lin
    return np.clip(result, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Limiter — Lookahead True-Peak-Limiter
# ---------------------------------------------------------------------------
def limiter(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    """
    Lookahead True-Peak-Limiter (hard ceiling mit Soft-Knee).

    params:
        ceiling_db    (float, default -1.0)  — Maximalpegel in dBFS
        lookahead_ms  (float, default 5.0)   — Vorausschau in ms
        release_ms    (float, default 50.0)  — Ausschwingzeit in ms
    """
    ceiling_db = float(params.get("ceiling_db", -1.0))
    lookahead_ms = max(float(params.get("lookahead_ms", 5.0)), 0.1)
    release_ms = max(float(params.get("release_ms", 50.0)), 1.0)

    ceiling_lin = 10 ** (ceiling_db / 20.0)
    lookahead_samp = int(lookahead_ms * 1e-3 * sr)
    tau_release = np.exp(-1.0 / (release_ms * 1e-3 * sr))

    audio_f64 = audio.astype(np.float64)
    n = len(audio_f64)

    # Lookahead: maximale Hüllkurve im Vorausfenster
    padded = np.pad(np.abs(audio_f64), (lookahead_samp, 0))
    peak_ahead = np.array([np.max(padded[max(0, i) : i + lookahead_samp + 1]) for i in range(n)])

    # Gain-Hüllkurve berechnen (Release-Glättung)
    gain = np.ones(n)
    current_gain = 1.0
    for i in range(n):
        needed_gain = ceiling_lin / (peak_ahead[i] + 1e-30) if peak_ahead[i] > ceiling_lin else 1.0
        # Gain darf nur langsam steigen (Release), sofort fallen
        if needed_gain < current_gain:
            current_gain = needed_gain  # sofortiger Gain-Down
        else:
            current_gain = tau_release * current_gain + (1 - tau_release) * needed_gain
        gain[i] = current_gain

    result = audio_f64 * gain
    return np.clip(result, -ceiling_lin, ceiling_lin).astype(audio.dtype)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Enhancer — Harmonischer Exciter (Sättigung + HF-Boost)
# ---------------------------------------------------------------------------
def enhancer(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    """
    Harmonischer Exciter: Erzeugt Obertöne durch leichte Sättigung des
    Hochfrequenzanteils und mischt diese zurück ins Signal.

    params:
        drive         (float, default 0.3)   — Sättigungsstärke [0.0, 1.0]
        mix           (float, default 0.2)   — Anteil Harmonische am Original [0..1]
        freq_hz       (float, default 3000)  — Einsatzfrequenz des Exciters (HP)
    """
    drive = float(np.clip(params.get("drive", 0.3), 0.0, 2.0))
    mix = float(np.clip(params.get("mix", 0.2), 0.0, 1.0))
    freq_hz = float(params.get("freq_hz", 3000.0))

    audio_f64 = audio.astype(np.float64)
    nyq = sr / 2.0

    # Hochpass-Filter: nur HF-Anteil wird gesättigt
    try:
        w_norm = min(freq_hz / nyq, 0.99)
        b_hp, a_hp = scipy.signal.butter(2, w_norm, btype="high")
        hf = scipy.safe_filtfilt(b_hp, a_hp, audio_f64)
    except Exception:
        hf = audio_f64.copy()

    # Sättigung: soft-clipping via tanh (drive skaliert Eingang)
    hf_saturated = np.tanh(hf * (1.0 + drive * 5.0)) / (1.0 + drive * 5.0) * (1.0 + drive * 5.0)
    # Normalisierung: Energie des Exciters an HF anpassen
    rms_hf = np.sqrt(np.mean(hf**2) + 1e-30)
    rms_sat = np.sqrt(np.mean(hf_saturated**2) + 1e-30)
    if rms_sat > 1e-10:
        hf_saturated *= rms_hf / rms_sat

    # Mix: Original + Exziter-Anteil
    result = audio_f64 + mix * (hf_saturated - hf)
    return np.clip(result, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Registry & Chain-Applier
# ---------------------------------------------------------------------------
# Mapping von Effekt-Namen zu Funktionen
dsp_effects: dict[str, Callable[[np.ndarray, int, dict[str, Any]], np.ndarray]] = {
    "eq": eq,
    "compressor": compressor,
    "limiter": limiter,
    "enhancer": enhancer,
}


def apply_dsp_chain(
    audio: np.ndarray,
    sr: int,
    chain: list[dict[str, Any]],
    enable_skip_gate: bool = True,
) -> np.ndarray:
    """
    Wendet eine Kette von DSP-Effekten auf das Audio an.
    chain: Liste von Effekten, z.B. [{'type': 'eq', 'params': {...}}, ...]
    enable_skip_gate: Aktiviert Phase-Skip-Gate (ΔSNR-Wächter, Standard: True).
    """
    for effect in chain:
        effect_type = effect.get("type")
        if effect_type is None:
            continue
        params = effect.get("params", {})
        func = dsp_effects.get(effect_type)
        if func is None:
            continue

        if enable_skip_gate and effect_type not in _ALWAYS_APPLY:
            snr_before = _compute_snr_db(audio)
            try:
                processed = func(audio, sr, params)
            except Exception as exc:
                logger.warning("DSP-Modul %s fehlgeschlagen, übersprungen: %s", effect_type, exc)
                continue
            delta = _compute_snr_db(processed) - snr_before
            if delta < _SKIP_GATE_THRESHOLD_DB:
                logger.info(
                    "⏭️ Verarbeitungsschritt-ueberspringen-Gate: %s revertiert (ΔSNR=%+.2f dB < %.2f dB)",
                    effect_type,
                    delta,
                    _SKIP_GATE_THRESHOLD_DB,
                )
                continue
            audio = processed
        else:
            try:
                audio = func(audio, sr, params)
            except Exception as exc:
                logger.warning("DSP-Modul %s fehlgeschlagen, übersprungen: %s", effect_type, exc)
    return audio


def apply_dsp_chain_tuple(
    audio: np.ndarray,
    sr: int,
    dsp_chain: list[tuple[str, dict[str, Any]]],
    enable_skip_gate: bool = True,
) -> np.ndarray:
    """
    Tuple-Listen-Variante von apply_dsp_chain (portiert aus backend._dsp_applier).
    dsp_chain: Liste von (module_name, params)-Tupeln.
    Delegiert intern an _apply_dsp_module() für plug-in-basiertes Dispatching.
    """
    applied: list[str] = []
    skipped: list[tuple[str, float | None]] = []

    for module_name, params in dsp_chain:
        snr_before = _compute_snr_db(audio) if enable_skip_gate and module_name not in _ALWAYS_APPLY else None
        try:
            processed = _apply_dsp_module(audio, sr, module_name, params)
        except Exception as exc:
            logger.warning("DSP-Modul %s fehlgeschlagen, übersprungen: %s", module_name, exc)
            skipped.append((module_name, None))
            continue

        if snr_before is not None:
            delta = _compute_snr_db(processed) - snr_before
            if delta < _SKIP_GATE_THRESHOLD_DB:
                logger.info(
                    "⏭️ Verarbeitungsschritt-ueberspringen-Gate: %s revertiert (ΔSNR=%+.2f dB)", module_name, delta
                )
                skipped.append((module_name, delta))
                continue

        audio = processed
        applied.append(module_name)

    if skipped:
        logger.info(
            "Verarbeitungsschritt-ueberspringen-Gate: %d angewendet, %d übersprungen: %s",
            len(applied),
            len(skipped),
            [s[0] for s in skipped],
        )
    return audio


def _apply_dsp_module(audio: np.ndarray, sr: int, module_name: str, params: dict[str, Any]) -> np.ndarray:
    """Plug-in-basierter DSP-Modul-Dispatcher (portiert aus backend._dsp_applier)."""
    try:
        if module_name == "DCBlocker":
            from dsp.dc_blocker import DCBlocker  # type: ignore[import]

            return DCBlocker().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name in ("HighpassFilter", "LinearPhaseHighpass"):
            from dsp.highpass_filter import HighpassFilter  # type: ignore[import]

            return HighpassFilter(cutoff_hz=params.get("cutoff_hz", 20.0)).process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "RumbleFilter":
            from dsp.rumble_filter import RumbleFilter  # type: ignore[import]

            return RumbleFilter().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "WowFlutterRemover":
            from dsp.wow_flutter_remover import WowFlutterRemover  # type: ignore[import]

            return WowFlutterRemover().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "AutomaticDeclicker":
            from dsp.automatic_declicker import AutomaticDeclicker  # type: ignore[import]

            return AutomaticDeclicker(aggressive=params.get("aggressive", False)).process(audio, sr)  # type: ignore[no-any-return,call-arg]
        elif module_name == "ClickpopRemover":
            from dsp.clickpop_remover import ClassicClickPopRemover

            return ClassicClickPopRemover().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name in ("AutomaticDeclipperVoice", "AutomaticDeclipperMusic", "AutomaticDeclipper"):
            from dsp.automatic_declipper import AutomaticDeclipper  # type: ignore[import]

            return AutomaticDeclipper().declip(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "AutomaticDehum":
            from dsp.automatic_dehum import AutomaticDehum  # type: ignore[import]

            return AutomaticDehum().dehum(audio, sr)  # type: ignore[no-any-return]
        elif module_name in ("AdaptiveOMLSA", "AdaptiveMCRA"):
            from dsp.adaptive_noise_reduction import AdaptiveNoiseReduction  # type: ignore[import]

            return AdaptiveNoiseReduction(algorithm="omlsa" if "OMLSA" in module_name else "mcra").process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "AdaptiveMusicalNoiseReduction":
            from dsp.musical_noise_reduction import MusicalNoiseReduction  # type: ignore[import]

            return MusicalNoiseReduction().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "TapeNoiseReduction":
            from dsp.tape_noise_reduction import TapeNoiseReduction  # type: ignore[import]

            return TapeNoiseReduction().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "Dehiss":
            from dsp.dehiss import AiDehiss

            return AiDehiss().dehiss(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "SpectralGate":
            from dsp.spectral_gate import SpectralGate  # type: ignore[import]

            return SpectralGate(threshold=params.get("threshold", -40)).process(audio, sr)  # type: ignore[no-any-return,call-arg]
        elif module_name == "RIAAEqualizer":
            from dsp.riaa_equalizer import RIAAEqualizer  # type: ignore[import]

            _curve = params.get("curve", "auto")
            return RIAAEqualizer(mode="invert", curve=_curve).process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "TapeEqualizer":
            from dsp.tape_equalizer import TapeEqualizer  # type: ignore[import]

            return TapeEqualizer().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "CDDeemphasis":
            from dsp.cd_deemphasis import CDDeemphasis  # type: ignore[import]

            return CDDeemphasis().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "AutoEQ":
            from dsp.auto_eq import AutoEQ  # type: ignore[import]

            return AutoEQ().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "CustomCompressor":
            from dsp.custom_compressor import CustomCompressor  # type: ignore[import]

            return CustomCompressor(  # type: ignore[no-any-return,call-arg]
                ratio=params.get("ratio", 2.0),
                threshold=params.get("threshold", -20.0),
            ).process(audio, sr)
        elif module_name == "TransientProtectionGuard":
            from dsp.transient_protection_guard import TransientProtectionGuard  # type: ignore[import]

            return cast(np.ndarray, TransientProtectionGuard().process(audio, sr))
        elif module_name == "HarmonicExciterStudio":
            from dsp.harmonic_exciter import HarmonicExciter  # type: ignore[import]

            return HarmonicExciter(mode="studio").process(audio, sr)  # type: ignore[no-any-return,call-arg]
        elif module_name == "SpeakerEnhancement":
            from dsp.speaker_enhancement import AiSpeakerEnhancement

            return AiSpeakerEnhancement().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "StereoEnhancer":
            from dsp.stereo_enhancer import AiStereoEnhancer

            return AiStereoEnhancer().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "StereoImageCorrection":
            from dsp.stereo_image_correction import StereoImageCorrection  # type: ignore[import]

            return StereoImageCorrection().process(audio, sr)  # type: ignore[no-any-return]
        elif module_name == "MultibandLimiter":
            from dsp.multiband_limiter import MultibandLimiter  # type: ignore[import]

            return MultibandLimiter(ceiling=params.get("ceiling", 0.95)).process(audio, sr)  # type: ignore[no-any-return,call-arg]
        elif module_name == "Dither":
            from dsp.dither import Dither  # type: ignore[import]

            return Dither().process(audio, sr)  # type: ignore[no-any-return]
        # §v10.1 Mode-check: Studio-only Modules in Restoration blockieren
        is_studio_mode = "studio" in _current_processing_mode.lower() or "2026" in _current_processing_mode.lower()
        if not is_studio_mode and module_name in _RESTORATION_BLOCKED_MODULES:
            logger.info("§v10.1 DSP: %s blockiert in Restoration → uebersprungen", module_name)
            return audio
        elif module_name in dsp_effects:
            return dsp_effects[module_name](audio, sr, params)
        else:
            logger.warning("Unbekanntes DSP-Modul: %s — übersprungen", module_name)
            return audio
    except ImportError:
        # DSP-Modul nicht installiert → Signal unverändert zurückgeben
        logger.debug("DSP-Modul %s nicht verfügbar (ImportError) — übersprungen", module_name)
        return audio
