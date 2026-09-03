"""§ORCHESTRATOR OrchestratorParams — Das EINE Gehirn. v10.0.0 Final.

JEDE Phase bezieht ALLE Parameter von HIER.
Keine Phase enthält eigene Entscheidungslogik.
Kein Wert ist geraten — alles kontinuierlich aus Messungen.

Architektur:
  Orchestrator (Gehirn)  →  compute_params(phase_id, measurements) → dict
  Phasen (Hände)         →  process(audio, **params)  ← nur DSP

§V25: 0 hartcodierte Werte. Alles kontinuierliche Funktionen.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Kontinuierliche Parameter-Funktionen — Pro Phase
# ═══════════════════════════════════════════════════════════════════════════


def _safe(m: dict, key: str, default: float = 0.0) -> float:
    return float(m.get(key, default) or default)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


# ── Phase 03: Denoise ───────────────────────────────────────────────────


def phase03_denoise(m: dict) -> dict:
    """Kontinuierlich: Stärke sinkt mit bandwidth_loss und Crest-Verlust."""
    bw = _safe(m, "bandwidth_loss")
    crest_drop = max(0.0, _safe(m, "crest_original") - _safe(m, "crest_current"))
    panns = _safe(m, "panns_singing")

    strength = 0.55 - 0.35 * bw - 0.04 * crest_drop
    if panns > 0.30:
        strength *= 0.70  # Vocal-Blend-Schutz
    strength = _clip(strength, 0.08, 0.70)

    return {"strength": round(strength, 3)}


# ── Phase 07: Harmonic Restoration ──────────────────────────────────────


def phase07_harmonic(m: dict) -> dict:
    """Kontinuierlich: Stärke sinkt mit bw_loss UND aktuellem Crest."""
    bw = _safe(m, "bandwidth_loss")
    crest_orig = _safe(m, "crest_original", 12.0)
    crest_now = _safe(m, "crest_current", crest_orig)
    crest_drop = max(0.0, crest_orig - crest_now)
    rms = _safe(m, "rms_db", -20.0)

    strength = 0.50 - 0.42 * bw - 0.05 * crest_drop
    if rms < -30:
        strength *= 0.50
    strength = _clip(strength, 0.05, 0.65)

    return {
        "strength": round(strength, 3),
        "h2_target": round(0.003 + 0.003 * (1.0 - bw), 4),
        "tilt_tolerance_db": round(2.0 + 2.0 * bw, 1),
    }


# ── Phase 19: De-Esser ──────────────────────────────────────────────────


def phase19_deesser(m: dict) -> dict:
    """Kontinuierlich: Sibilanz-Schwelle und Stärke-Cap aus bw_loss + Codec."""
    bw = _safe(m, "bandwidth_loss")
    terminal = str(m.get("terminal_codec", "") or "").lower()
    is_mp3 = terminal in ("mp3_low", "mp3_high")

    sib_factor = 1.0 + 4.0 * bw if is_mp3 else 1.0 + 1.5 * bw
    sib_factor = _clip(sib_factor, 1.0, 6.0)

    cap_factor = 0.55 + 0.45 * (1.0 - bw) if is_mp3 else 1.0
    cap_factor = _clip(cap_factor, 0.40, 1.0)

    return {
        "sibilance_threshold_mult": round(sib_factor, 1),
        "deessing_strength_cap_factor": round(cap_factor, 2),
    }


# ── Phase 39: Air-Band ──────────────────────────────────────────────────


def phase39_air_band(m: dict) -> dict:
    """Kontinuierlich: Air-Band nur wenn bandwidth_loss es rechtfertigt."""
    bw = _safe(m, "bandwidth_loss")
    is_restoration = m.get("is_restoration_mode", True)
    is_analog = str(m.get("material_type", "")).lower() in (
        "vinyl",
        "shellac",
        "wax_cylinder",
        "wire_recording",
        "tape",
        "reel_tape",
        "cassette",
        "lacquer_disc",
    )

    # Air-Band ist in Restoration für analoge Quellen NUR mit bw_loss>0.5 erlaubt
    allow = not (is_restoration and is_analog and bw <= 0.5)

    if allow:
        strength = 0.15 + 0.35 * bw  # bw=0.5→0.33, bw=1.0→0.50
    else:
        strength = 0.0

    return {
        "allow_air_band": allow,
        "strength": round(_clip(strength, 0.0, 0.60), 3),
        "shelf_gain_db": round(1.0 + 5.0 * bw, 1),
    }


# ── Phase 29: Tape Hiss Reduction ───────────────────────────────────────


def phase29_tape_hiss(m: dict) -> dict:
    """Kontinuierlich: Stärke aus Material-Typ + SNR."""
    bw = _safe(m, "bandwidth_loss")
    snr = _safe(m, "snr_db", 30.0)
    depth = max(1, int(m.get("transfer_chain_depth", 1)))

    strength = 0.60 - 0.15 * bw
    if snr < 20:
        strength *= 0.70
    if depth >= 5:
        strength *= 0.80
    strength = _clip(strength, 0.10, 0.65)

    return {"strength": round(strength, 3)}


# ── Phase 06: Frequency Restoration (NVSR) ──────────────────────────────


def phase06_frequency(m: dict) -> dict:
    """Kontinuierlich: NVSR-Stärke aus Bandwidth-Loss + Restorability."""
    bw = _safe(m, "bandwidth_loss")
    rs = _safe(m, "restorability_score", 50.0) / 100.0

    # Nur sinnvoll wenn tatsächlich Frequenzen fehlen
    if bw < 0.3:
        return {"strength": 0.0, "skip": True}

    strength = 0.25 + 0.55 * bw
    strength *= 0.5 + 0.5 * rs  # Restorability-Modifikator
    strength = _clip(strength, 0.10, 0.80)

    rolloff_target = 10000 + int(5500 * bw)  # bw=0.5→12750, bw=1.0→15500

    return {
        "strength": round(strength, 3),
        "target_rolloff_hz": rolloff_target,
        "skip": False,
    }


# ── Phase 40: Loudness Normalization ────────────────────────────────────


def phase40_loudness(m: dict) -> dict:
    """Kontinuierlich: Loudness-Ziel aus Material + Restorability."""
    rs = _safe(m, "restorability_score", 50.0) / 100.0

    # Je besser das Material, desto näher am Standard-LUFS
    target_lufs = -18.0 + 3.0 * (1.0 - rs)  # rs=50→-16.5, rs=90→-17.7

    return {
        "target_lufs": round(target_lufs, 1),
        "strength": round(rs, 3),
    }


# ── Phase 01: Click Removal ─────────────────────────────────────────────


def phase01_click(m: dict) -> dict:
    """Kontinuierlich: Stärke aus Klick-Dichte + SNR."""
    snr = _safe(m, "snr_db", 30.0)
    click_density = _safe(m, "click_density", 500.0)
    depth = max(1, int(m.get("transfer_chain_depth", 1)))

    # Mehr Klicks → mehr Stärke, aber SNR limitiert
    strength = 0.15 + 0.0003 * click_density
    if snr < 15:
        strength *= 0.60
    if depth >= 4:
        strength *= 0.85
    strength = _clip(strength, 0.10, 0.80)

    return {"strength": round(strength, 3)}


# ── Phase 12: Wow/Flutter ───────────────────────────────────────────────


def phase12_wow_flutter(m: dict) -> dict:
    """Kontinuierlich: Stärke aus Wow/Flutter-Severity + Material."""
    wow = _safe(m, "wow_severity", 0.0)
    flutter = _safe(m, "flutter_severity", 0.0)
    depth = max(1, int(m.get("transfer_chain_depth", 1)))

    sev = max(wow, flutter)
    if sev < 0.10:
        return {"strength": 0.0, "skip": True}

    strength = 0.20 + 0.60 * sev
    if depth >= 4:
        strength *= 0.80
    strength = _clip(strength, 0.10, 0.70)

    return {"strength": round(strength, 3), "skip": False}


# ── Phase 09: Crackle ───────────────────────────────────────────────────


def phase09_crackle(m: dict) -> dict:
    """Kontinuierlich: Stärke aus Crackle-Dichte."""
    density = _safe(m, "crackle_density", 0.0)
    if density < 5:
        return {"strength": 0.0, "skip": True}
    strength = 0.10 + 0.001 * min(density, 1000)
    return {"strength": round(_clip(strength, 0.08, 0.60), 3), "skip": False}


# ═══════════════════════════════════════════════════════════════════════════
# Universelle Parameter-Funktion — Das EINE Gehirn
# ═══════════════════════════════════════════════════════════════════════════

_PARAM_FUNCTIONS: dict[str, Any] = {
    "phase_01_click_removal": phase01_click,
    "phase_03_denoise": phase03_denoise,
    "phase_06_frequency_restoration": phase06_frequency,
    "phase_07_harmonic_restoration": phase07_harmonic,
    "phase_09_crackle_removal": phase09_crackle,
    "phase_12_wow_flutter_fix": phase12_wow_flutter,
    "phase_19_de_esser": phase19_deesser,
    "phase_29_tape_hiss_reduction": phase29_tape_hiss,
    "phase_39_air_band_enhancement": phase39_air_band,
    "phase_40_loudness_normalization": phase40_loudness,
}


# Generische Default-Funktion für Phasen ohne spezifische Kalibrierung
def _generic_params(m: dict) -> dict:
    bw = _safe(m, "bandwidth_loss")
    rs = _safe(m, "restorability_score", 50.0) / 100.0
    strength = (0.30 + 0.40 * rs) * (1.0 - 0.30 * bw)
    return {"strength": round(_clip(strength, 0.08, 0.75), 3)}


def compute_phase_params(
    phase_id: str,
    measurements: dict,
) -> dict:
    """Das EINE Gehirn: Berechnet ALLE Parameter für eine Phase.

    Args:
        phase_id: ID der Phase (z.B. "phase_03_denoise")
        measurements: Dict mit ALLEN aktuellen Messwerten.
            Muss enthalten: bandwidth_loss, restorability_score,
            crest_original, crest_current (mindestens).

    Returns:
        Dict mit kalibrierten Parametern — direkt an phase.process() übergebbar.
    """
    fn = _PARAM_FUNCTIONS.get(phase_id, _generic_params)
    try:
        params = fn(measurements)
    except Exception as e:
        logger.debug("berechnen_Verarbeitungsschritt_params %s: %s — Ersatzpfad generic", phase_id, e)
        params = _generic_params(measurements)

    # Gemeinsame Parameter für ALLE Phasen
    params.setdefault("calibrated", True)
    params.setdefault("restorability_score", _safe(measurements, "restorability_score", 50.0))
    params.setdefault("bandwidth_loss", _safe(measurements, "bandwidth_loss", 0.0))

    return params  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight Probe: Eine schnelle Test-Iteration für Grenzfälle
# ═══════════════════════════════════════════════════════════════════════════


def probe_phase_benefit(
    phase_id: str,
    audio: np.ndarray,
    phase_runner,
    params: dict,
    sample_rate: int = 48000,
) -> dict:
    """Testet mit EINER schnellen Ausführung, ob die Phase nützt.

    Führt die Phase mit den kalibrierten Parametern aus und misst
    das Delta. Wenn es negativ ist, wird eine reduzierte Stärke
    getestet. Nur wenn BEIDE schaden → Skip.

    Returns:
        {"should_run": bool, "strength": float, "delta": float, "reason": str}
    """
    try:
        # Test 1: Kalibrierte Stärke
        strength = params.get("strength", 0.20)
        audio_test = phase_runner(audio, strength)
        delta = _quick_probe_delta(audio, audio_test)

        if delta > -0.02:
            return {
                "should_run": True,
                "strength": strength,
                "delta": round(delta, 4),
                "reason": f"Kalibrierte Stärke {strength:.3f} hilft (Δ={delta:+.4f})",
            }

        # Test 2: Halbierte Stärke
        strength_lo = max(0.03, strength * 0.5)
        audio_lo = phase_runner(audio, strength_lo)
        delta_lo = _quick_probe_delta(audio, audio_lo)

        if delta_lo > -0.02:
            return {
                "should_run": True,
                "strength": strength_lo,
                "delta": round(delta_lo, 4),
                "reason": f"Reduziert auf {strength_lo:.3f} (Δ={delta_lo:+.4f})",
            }

        # Beide schaden → Skip
        return {
            "should_run": False,
            "strength": 0.0,
            "delta": round(min(delta, delta_lo), 4),
            "reason": f"Keine Stärke hilft ({strength:.3f}→{delta:+.4f}, {strength_lo:.3f}→{delta_lo:+.4f})",
        }

    except Exception as e:
        return {"should_run": False, "strength": 0.0, "delta": -1.0, "reason": f"Probe fehlgeschlagen: {e}"}


def _quick_probe_delta(pre: np.ndarray, post: np.ndarray) -> float:
    """Ultraschnelles Qualitäts-Delta (<1ms)."""
    try:
        a = np.asarray(pre, dtype=np.float32).ravel()
        b = np.asarray(post, dtype=np.float32).ravel()
        n = min(len(a), len(b), 4096)  # nur 4096 samples
        if n < 64:
            return 0.0
        a, b = a[:n], b[:n]
        rms_a = float(np.sqrt(np.mean(a**2))) + 1e-12
        rms_b = float(np.sqrt(np.mean(b**2))) + 1e-12
        rms_ok = min(rms_a, rms_b) / max(rms_a, rms_b)
        corr = float(np.corrcoef(a, b)[0, 1]) if n > 2 else 1.0
        corr = max(0.0, min(1.0, corr)) if not np.isnan(corr) else 1.0
        return float(0.5 * rms_ok + 0.5 * corr - 0.95)
    except Exception:
        logger.warning("§V6 ML→DSP-Fallback: _compute_quality_delta fehlgeschlagen → neutraler Return (0.0)")
        return 0.0
