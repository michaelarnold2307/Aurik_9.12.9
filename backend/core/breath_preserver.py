"""BreathPreservationGate — Atem-Erhalt bei Noise Reduction.

§Rolls-Royce-Phantom: Atemgeräusche sind Teil der Stimme, kein Rauschen.
Dieser Gate wird VOR Noise-Reduction-Phasen eingehängt und schützt den
4–8 kHz-Bereich mit einer spektralen Maske.

Algorithmus:
  1. Detektiere Atem-Energie im 4–8 kHz-Bereich
  2. Erstelle spektrale Maske (Soft-Mask, nicht Hard-Gate)
  3. NR-Phase läuft (die Maske ist im Audio encodiert)
  4. Optional: Post-NR die Maske entfernen und Original-Atem rekonstruieren

Nutzung:
    from backend.core.breath_preserver import protect_breath, restore_breath

    masked, mask = protect_breath(audio, sr)
    cleaned = noise_reduction_phase(masked)  # NR-Phase
    audio = restore_breath(cleaned, mask, original_breath)

Autor: Aurik 10 — Rolls-Royce Phantom Edition, 11. Juli 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

BREATH_LOW_HZ: float = 4000.0
BREATH_HIGH_HZ: float = 8000.0
BREATH_THRESHOLD: float = 0.0005  # Min Energie-Ratio für Atem-Schutz


@dataclass
class BreathMask:
    """Spektrale Atem-Maske."""

    frequencies: np.ndarray  # Frequenz-Achse
    mask_values: np.ndarray  # Maske [0–1] pro Frequenz-Bin
    original_breath_energy: float  # Energie VOR NR
    sr: int
    fft_size: int


def extract_breath_mask(
    audio: np.ndarray,
    sr: int = 48000,
    fft_size: int = 4096,
    softness: float = 0.7,  # 1.0 = Hard-Mask, 0.0 = kein Schutz
) -> BreathMask | None:
    """Extrahiert Atem-Maske aus dem Audio.

    Args:
        audio:    float32, mono oder stereo.
        sr:       Abtastrate.
        fft_size: FFT-Größe.
        softness: Weichheit der Maske (0–1).

    Returns:
        BreathMask oder None wenn keine Atem-Energie detektiert wurde.
    """
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
    mono = mono.astype(np.float32).flatten()

    if len(mono) < sr * 0.5:
        return None

    # ── Spektrum ───────────────────────────────────────────────────────
    spec = np.abs(np.fft.rfft(mono[:fft_size]))
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sr)

    breath_range = (freqs >= BREATH_LOW_HZ) & (freqs <= BREATH_HIGH_HZ)
    breath_energy = float(np.sum(spec[breath_range]))
    total_energy = float(np.sum(spec)) + 1e-10
    breath_ratio = breath_energy / total_energy

    if breath_ratio < BREATH_THRESHOLD:
        logger.debug(
            "BreathPreserver: Atem-Energie %.6f < %.6f — kein Schutz nötig",
            breath_ratio,
            BREATH_THRESHOLD,
        )
        return None

    # ── Maske bauen ────────────────────────────────────────────────────
    mask = np.ones(len(freqs), dtype=np.float32)

    # Soft-Mask im Atembereich: Erhöhtes Signal für NR → NR entfernt weniger
    # NR-Algorithmen sehen das erhöhte Signal und behandeln es NICHT als Rauschen
    mask[breath_range] = 1.0 + (softness * 0.5)  # +50% im Atembereich

    logger.info(
        "BreathPreserver: Atem-Energie %.5f (%.1f%%) → Maske aktiv (softness=%.1f)",
        breath_ratio,
        breath_ratio * 100,
        softness,
    )

    return BreathMask(
        frequencies=freqs,
        mask_values=mask,
        original_breath_energy=breath_energy,
        sr=sr,
        fft_size=fft_size,
    )


def reconstruct_breath(
    cleaned: np.ndarray,
    original: np.ndarray,
    mask: BreathMask,
    blend: float = 0.3,  # Wieviel Original-Atem zurückgemischt wird
) -> np.ndarray:
    """Rekonstruiert Atem-Energie nach Noise Reduction.

    Mischt einen Teil der Original-Atem-Energie zurück ins gereinigte Signal.
    Verhindert den „sterilen" Klang von über-entrauschtem Gesang.

    Args:
        cleaned:  Audio NACH Noise Reduction.
        original: Audio VOR Noise Reduction.
        mask:     BreathMask aus extract_breath_mask().
        blend:    Blend-Faktor für Original-Atem (0.0–1.0).

    Returns:
        Audio mit rekonstruierter Atem-Natürlichkeit.
    """
    mono_clean = np.mean(cleaned, axis=-1) if cleaned.ndim > 1 else cleaned
    mono_orig = np.mean(original, axis=-1) if original.ndim > 1 else original

    if len(mono_clean) < mask.fft_size:
        return cleaned

    # ── Spektral-Domain Rekonstruktion ─────────────────────────────────
    spec_clean = np.fft.rfft(mono_clean[: mask.fft_size])
    spec_orig = np.fft.rfft(mono_orig[: mask.fft_size])

    # Maske auf Original und Cleaned anwenden
    breath_range = (mask.frequencies >= BREATH_LOW_HZ) & (mask.frequencies <= BREATH_HIGH_HZ)

    # Blend: Mixe Original-Atem in cleaned Signal
    spec_blend = spec_clean.copy()
    spec_blend[breath_range] = (1 - blend) * spec_clean[breath_range] + blend * spec_orig[breath_range]

    # Überprüfen: Blend nicht lauter als Original machen
    blend_energy = float(np.sum(np.abs(spec_blend[breath_range])))
    orig_energy = float(np.sum(np.abs(spec_orig[breath_range])))

    if blend_energy > orig_energy * 1.5:  # Nicht mehr als +50%
        scale = (orig_energy * 1.5) / (blend_energy + 1e-10)
        spec_blend[breath_range] *= scale
        logger.debug("BreathPreserver: Blend-Energie begrenzt (×%.2f)", scale)

    # Zurück in Time-Domain
    waveform_blend = np.fft.irfft(spec_blend, n=len(mono_clean[: mask.fft_size * 2]))

    # Nur ersten Teil ersetzen, Rest unverändert
    result = cleaned.copy()
    if cleaned.ndim > 1:
        n_samples = min(len(waveform_blend), result.shape[0])
        for ch in range(result.shape[1]):
            result[:n_samples, ch] = (
                result[:n_samples, ch] * (1 - blend * 0.3) + waveform_blend[:n_samples] * blend * 0.3
            )
    else:
        n_samples = min(len(waveform_blend), len(result))
        result[:n_samples] = result[:n_samples] * (1 - blend * 0.3) + waveform_blend[:n_samples] * blend * 0.3

    logger.info("BreathPreserver: Atem rekonstruiert (blend=%.1f)", blend)
    return cast(np.ndarray, result.astype(np.float32))


def protect_breath(audio: np.ndarray, sr: int = 48000) -> tuple[np.ndarray, BreathMask | None]:
    """Pre-Noise-Reduction: Atem-Bereich vor NR schützen.

    Erhöht die Energie im 4–8 kHz-Bereich leicht, sodass NR-Algorithmen
    diesen Bereich NICHT als Rauschen klassifizieren.

    Args:
        audio: float32 Audio.
        sr:    Abtastrate.

    Returns:
        (masked_audio, breath_mask) — mask=None wenn kein Atem detektiert.
    """
    mask = extract_breath_mask(audio, sr)
    if mask is None:
        return audio, None

    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
    spec = np.fft.rfft(mono[: mask.fft_size].astype(np.float32))

    # Maske anwenden (Boosten, nicht Cutten)
    spec_masked = spec * mask.mask_values

    waveform = np.fft.irfft(spec_masked, n=len(mono[: mask.fft_size * 2]))

    result = audio.copy()
    n = min(len(waveform), result.shape[0] if result.ndim == 1 else result.shape[0])
    if result.ndim > 1:
        for ch in range(result.shape[1]):
            result[:n, ch] = waveform[:n]
    else:
        result[:n] = waveform[:n]

    return result.astype(np.float32), mask


def restore_breath(
    cleaned: np.ndarray,
    mask: BreathMask | None,
    original: np.ndarray,
) -> np.ndarray:
    """Post-Noise-Reduction: Atem-Natürlichkeit wiederherstellen.

    Args:
        cleaned:  Audio NACH NR.
        mask:     BreathMask (oder None).
        original: Audio VOR NR (für Atem-Rekonstruktion).

    Returns:
        Audio mit rekonstruiertem Atem.
    """
    if mask is None:
        return cleaned
    return reconstruct_breath(cleaned, original, mask)


__all__ = [
    "protect_breath",
    "restore_breath",
    "extract_breath_mask",
    "reconstruct_breath",
    "BreathMask",
]
