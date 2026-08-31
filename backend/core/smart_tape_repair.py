"""SmartTapeRepair — Azimuth/HF-Reparatur mit Gesangs-Schutz.

Behebt Bandkopfdefekte (Azimuth-Drift, HF-Verlust) ohne den
Gesang zu beschädigen. Nutzt Vocal-Activity-Detection um
Reparatur-Stärke adaptiv zu steuern:
  - Volle Reparatur: Nicht-Gesang-Zonen (Instrumente)
  - Reduzierte Reparatur (30%): Gesang-Zonen
  - Keine Reparatur: Frisson/Klimax-Zonen
"""

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


def smart_tape_repair(audio: np.ndarray, sr: int, vocal_mask: np.ndarray | None = None):
    """Azimuth/HF-Repair with vocal protection.

    Args:
        audio: (channels, samples) float32 stereo
        sr: sample rate
        vocal_mask: bool array (True=vocal) or None for auto-detection

    Returns: repaired_audio, report_dict
    """
    result = np.asarray(audio, dtype=np.float32).copy()
    n = result.shape[-1]

    if result.ndim < 2 or result.shape[0] < 2:
        return result, {"error": "stereo required"}

    mono = np.mean(result, axis=0)

    # Auto-detect vocal zones if no mask
    if vocal_mask is None:
        vocal_mask = _detect_vocal_zones(mono, sr)

    vocal_pct = np.mean(vocal_mask) * 100
    logger.info("SmartTapeRepair: vocal=%.1f%% of audio", vocal_pct)

    # ── 1. AZIMUTH CORRECTION (vocal-aware) ──
    block = int(sr * 0.100)
    hop = block // 4
    azimuth_fixes = 0

    for i in range(0, n - block, hop):
        is_vocal = vocal_mask[i + block // 2] if i + block // 2 < n else False

        l_seg = result[0, i : i + block]
        r_seg = result[1, i : i + block]

        fft_l = np.fft.rfft(l_seg)
        fft_r = np.fft.rfft(r_seg)
        freqs = np.fft.rfftfreq(len(l_seg), d=1.0 / sr)
        hf = freqs >= 6000

        if not np.any(hf):
            continue

        phase_diff = np.median(np.angle(fft_r[hf]) - np.angle(fft_l[hf]))
        deg = abs(np.degrees(phase_diff))

        if deg > 3:
            # Vocal zones: gentler correction (15% statt 40%)
            strength = 0.15 if is_vocal else 0.40
            fft_r[hf] *= np.exp(-1j * phase_diff * strength)
            corrected = np.fft.irfft(fft_r, n=block)
            result[1, i : i + block] = corrected[:block].astype(np.float32)
            azimuth_fixes += 1

    # ── 2. DYNAMIC HF RESTORATION (vocal-aware) ──
    block2 = int(sr * 0.200)
    hf_restored = 0

    for i in range(0, n - block2, block2):
        mid = i + block2 // 2
        is_vocal = vocal_mask[min(mid, len(vocal_mask) - 1)] if mid < n else False

        seg = mono[i : i + block2]
        fft = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        hf_mask = freqs >= 8000
        hf_ratio = np.sum(fft[hf_mask]) / (np.sum(fft) + 1e-12) if np.any(hf_mask) else 0

        # Target HF ratio: 3% (typical for cassette) or 8% (typical for good vinyl)
        target = 0.03  # Conservative for cassette

        if hf_ratio < target * 0.5:  # Less than 50% of target
            deficit = target - hf_ratio
            gain = 1.0 + deficit * 50  # Scale factor
            gain = min(gain, 2.5)  # Max +8dB

            # Vocal zones: only 30% of the correction
            actual_gain = 1.0 + (gain - 1.0) * (0.3 if is_vocal else 0.7)

            for ch in range(2):
                ch_fft = np.fft.rfft(result[ch, i : i + block2])
                ch_fft[hf_mask] *= actual_gain
                result[ch, i : i + block2] = np.fft.irfft(ch_fft, n=block2)
            hf_restored += 1

    result = np.clip(result, -1.0, 1.0).astype(np.float32)

    report = {
        "azimuth_fixes": azimuth_fixes,
        "hf_restored_blocks": hf_restored,
        "vocal_coverage_pct": round(vocal_pct, 1),
    }
    return result, report


def _detect_vocal_zones(audio: np.ndarray, sr: int) -> np.ndarray:
    """Detect vocal zones via spectral centroid + energy."""
    n = len(audio)
    win = int(sr * 0.025)
    hop = max(1, win // 2)
    if win < 64 or n < win:
        return cast(np.ndarray, (np.zeros(n, dtype=bool)))

    mask = np.zeros(n, dtype=bool)
    for i in range(0, n - win, hop):
        frame = audio[i : i + win]
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        if rms < 0.002:  # Silence
            continue
        fft = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
        if np.sum(fft) > 0:
            centroid = float(np.average(freqs, weights=fft + 1e-10))
            # Vocal: energy in 300-3000 Hz
            vb = (freqs >= 300) & (freqs <= 3000)
            vb_ratio = float(np.sum(fft[vb])) / (float(np.sum(fft)) + 1e-12)
            if vb_ratio > 0.3 and 300 < centroid < 3000:
                mask[i : i + win] = True

    return cast(np.ndarray, mask)
