"""§2.72 VitalityRestorer: Post-Pipeline Klang-Lebendigkeit.

Drei Maßnahmen, die nach allen 42 Phasen den musikalischen Atem
wiederherstellen — ohne Training, rein DSP:

  1. Stereo-Breite:   IACC/Panorama vor Pipeline messen, per M/S-Gain zurückführen
  2. Mikrodynamik:    Crest-Faktor-Expansion — Leises leiser, Lautes lauter
  3. Transient-Punch: Lokales Dry/Wet-Override an Attack-Punkten (5-15ms)

Alle drei arbeiten auf dem Original-Referenzsignal (pre-repair reference)
und dem final restaurierten Signal.

Reference: ITU-R BS.1770 (Loudness), BS.1387 (PEAQ), Zwicker/Fastl.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Stereo-Width ──────────────────────────────────────────────
_MID_GAIN_MAX_BOOST: float = 1.12  # +1 dB Mid-Kanal
_SIDE_GAIN_MAX_BOOST: float = 1.35  # +2.6 dB Side-Kanal
_IACC_TARGET_MIN: float = 0.75  # Untergrenze IACC (Mono-Schutz)

# ── Mikrodynamik ──────────────────────────────────────────────
_EXPAND_RATIO: float = 1.15  # Sanfte Expansion (1:1.15)
_EXPAND_THRESHOLD_DBFS: float = -18.0
_EXPAND_KNEE_DB: float = 3.0
_EXPAND_MAX_GAIN_DB: float = 2.5

# ── Transient-Punch ───────────────────────────────────────────
_TRANSIENT_ATTACK_MS: float = 8.0  # Attack-Schutz-Fenster
_TRANSIENT_DRY_BOOST: float = 0.25  # +25% Dry-Anteil an Transienten


def restore_vitality(
    reference: np.ndarray,
    restored: np.ndarray,
    sample_rate: int,
    *,
    stereo: bool = True,
    microdynamics: bool = True,
    transients: bool = True,
) -> np.ndarray:
    """Haupt-API: Stellt Stereo-Breite, Mikrodynamik und Transient-Punch wieder her.

    Args:
        reference: Pre-Repair-Referenz (Original vor allen Phasen)
        restored:  Final restauriertes Signal (nach allen 42 Phasen)
        sample_rate: Abtastrate
        stereo: Stereo-Breite wiederherstellen
        microdynamics: Crest-Faktor-Expansion
        transients: Transienten-Punch erhalten

    Returns:
        Vitalisiertes Signal (selbe Shape wie restored)
    """
    ref = np.asarray(reference, dtype=np.float64)
    rst = np.asarray(restored, dtype=np.float64)
    is_stereo = ref.ndim == 2 and ref.shape[0] == 2
    result = rst.copy()

    if stereo and is_stereo:
        result = _restore_stereo_width(ref, result, sample_rate)

    if microdynamics:
        result = _restore_microdynamics(ref, result, sample_rate)

    if transients:
        result = _restore_transient_punch(ref, result, sample_rate)

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))


# ═══════════════════════════════════════════════════════════════
# Stereo-Breite: M/S-Gain-Rückführung
# ═══════════════════════════════════════════════════════════════


def _restore_stereo_width(
    reference: np.ndarray,
    restored: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Misst IACC + Panorama der Referenz und führt es im Restaurierten zurück.

    Nach 42 Phasen (M/S-Verarbeitung, per-Kanal-DSP) kollabiert das Stereobild.
    Mid/Side-Energie-Verhältnis wird gemessen und sanft korrigiert.
    """
    try:
        # Mid/Side der Referenz
        ref_mid = (reference[0] + reference[1]) / 1.41421356
        ref_side = (reference[0] - reference[1]) / 1.41421356

        # Mid/Side des Restaurierten
        rst_mid = (restored[0] + restored[1]) / 1.41421356
        rst_side = (restored[0] - restored[1]) / 1.41421356

        # Energie-Verhältnis Side/Mid (Stereo-Breite-Proxy)
        eps = 1e-10
        ref_side_rms = float(np.sqrt(np.mean(ref_side**2)) + eps)
        ref_mid_rms = float(np.sqrt(np.mean(ref_mid**2)) + eps)
        rst_side_rms = float(np.sqrt(np.mean(rst_side**2)) + eps)
        rst_mid_rms = float(np.sqrt(np.mean(rst_mid**2)) + eps)

        ref_ratio = ref_side_rms / ref_mid_rms
        rst_ratio = rst_side_rms / rst_mid_rms

        # Nur korrigieren wenn Restaurierung das Bild verengt hat
        if rst_ratio < ref_ratio * 0.98 and ref_ratio > 0.02:
            # Wie viel Side-Gain brauchen wir?
            target_side_rms = rst_mid_rms * ref_ratio
            side_gain = float(np.clip(target_side_rms / rst_side_rms, 1.0, _SIDE_GAIN_MAX_BOOST))
            # §v10.63 mid_gain=1.0: Keine Mid-Kompensation. Der vorherige Ansatz
            # (mid_gain = 1/√side_gain) war eine Heuristik die bei side_gain=1.5
            # +1.6 dB Gesamtenergie-Überschuss erzeugte — hörbare Lautstärke-Änderung.
            # Nur Side korrigieren, Loudness-Normalizer fängt Pegeländerung auf.
            mid_gain = 1.0

            # Sanfte Korrektur (nur 85 % des Wegs — konservativ, war 70 %)
            side_gain = 1.0 + (side_gain - 1.0) * 0.85

            rst_side_corrected = rst_side * side_gain
            rst_mid_corrected = rst_mid * mid_gain

            left = (rst_mid_corrected + rst_side_corrected) / 1.41421356
            right = (rst_mid_corrected - rst_side_corrected) / 1.41421356

            logger.info(
                "§2.72 Stereo: side/mid %.3f→%.3f (ref=%.3f) → side_gain=%.2f",
                rst_ratio,
                rst_side_rms * side_gain / (rst_mid_rms * mid_gain),
                ref_ratio,
                side_gain,
            )
            return np.stack([left, right], axis=0).astype(np.float64)  # type: ignore[no-any-return]

    except Exception as exc:
        logger.debug("§2.72 Stereo-Width nicht blockierend: %s", exc)

    return restored


# ═══════════════════════════════════════════════════════════════
# Mikrodynamik: Crest-Faktor-Expansion
# ═══════════════════════════════════════════════════════════════


def _restore_microdynamics(
    reference: np.ndarray,
    restored: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Sanfte Expansion: Leises leiser, Lautes lauter — Atmung für den Mix.

    Misst den Crest-Faktor-Unterschied (Peak/RMS) zwischen Original und
    Restauriertem. Wendet einen sanften Downward-Expander an, der das
    natürliche Dynamik-Profil wiederherstellt.
    """
    try:
        is_stereo = restored.ndim == 2
        rst_mono = np.mean(restored, axis=0) if is_stereo else restored
        ref_mono = np.mean(reference, axis=0) if is_stereo else reference

        # RMS in 50ms-Fenstern für Dynamik-Profil
        win = int(0.050 * sample_rate)
        if win < 16:
            return restored

        kernel = np.hanning(win)
        kernel /= kernel.sum()

        ref_rms = np.sqrt(np.convolve(ref_mono**2, kernel, mode="same") + 1e-12)
        rst_rms = np.sqrt(np.convolve(rst_mono**2, kernel, mode="same") + 1e-12)

        # Wo wurde das Signal komprimiert? (Restauriert-RMS < Referenz-RMS)
        # Mindest-RMS: −40 dBFS — verhindert Expansion von Stille/Intros
        compressed = (rst_rms < ref_rms * 0.95) & (ref_rms > 0.01)

        if not np.any(compressed):
            return restored

        # Expansion: Gain = (ref_rms / rst_rms) ^ (1 - 1/ratio) im komprimierten Bereich
        expand_ratio = _EXPAND_RATIO
        gain = np.ones(len(rst_mono), dtype=np.float64)

        for i in np.where(compressed)[0]:
            ratio_db = 20.0 * np.log10(ref_rms[i] / max(rst_rms[i], 1e-10))
            if ratio_db > 0.5:  # Nur wenn signifikante Kompression
                # Sanfte Expansion: je stärker komprimiert, desto mehr Gain
                expand_db = min(_EXPAND_MAX_GAIN_DB, ratio_db * (1.0 - 1.0 / expand_ratio))
                gain[i] = 10.0 ** (expand_db / 20.0)

        # Glätten des Gain-Signals (Attack: 10ms, Release: 100ms)
        attack_coeff = np.exp(-1.0 / max(1, 0.010 * sample_rate / win))
        release_coeff = np.exp(-1.0 / max(1, 0.100 * sample_rate / win))

        gain_smooth = np.copy(gain)
        for i in range(1, len(gain)):
            if gain[i] > gain_smooth[i - 1]:
                gain_smooth[i] = attack_coeff * gain_smooth[i - 1] + (1.0 - attack_coeff) * gain[i]
            else:
                gain_smooth[i] = release_coeff * gain_smooth[i - 1] + (1.0 - release_coeff) * gain[i]

        n_expanded = int(np.sum(gain_smooth > 1.01))
        if n_expanded > 0:
            logger.info(
                "§2.72 Mikrodynamik: %d/%d Samples expandiert (max +%.1f dB)",
                n_expanded,
                len(gain),
                20.0 * np.log10(np.max(gain_smooth)),
            )

        if is_stereo:
            gain_bc = gain_smooth[np.newaxis, :]
        else:
            gain_bc = gain_smooth

        return np.clip(restored * gain_bc, -1.0, 1.0).astype(np.float64)  # type: ignore[no-any-return]

    except Exception as exc:
        logger.debug("§2.72 Mikrodynamik nicht blockierend: %s", exc)
        return restored


# ═══════════════════════════════════════════════════════════════
# Transient-Punch: Lokales Dry/Wet-Override
# ═══════════════════════════════════════════════════════════════


def _restore_transient_punch(
    reference: np.ndarray,
    restored: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """An Transienten: mehr Original-Signal beimischen → Punch erhalten.

    Jede Transiente (Drum, Plosiv) bekommt für 8ms einen höheren Dry-Anteil.
    Die Referenz-Transiente ersetzt nicht komplett, sondern boostet den
    Original-Anteil um +25 %.
    """
    try:
        is_stereo = restored.ndim == 2
        rst_mono = np.mean(restored, axis=0) if is_stereo else restored
        _ref_mono = np.mean(reference, axis=0) if is_stereo else reference

        # Einfacher Transient-Detektor: Energie-Spitzen > 3× lokaler Mittelwert
        win_short = int(_TRANSIENT_ATTACK_MS * sample_rate / 1000)
        win_long = win_short * 4

        if win_short < 4 or len(rst_mono) < win_long:
            return restored

        energy = rst_mono.astype(np.float64) ** 2
        e_short = np.convolve(energy, np.ones(win_short) / win_short, mode="same")
        e_long = np.convolve(energy, np.ones(win_long) / win_long, mode="same")

        ratio = e_short / (e_long + 1e-10)
        transients = ratio > 3.0

        if not np.any(transients):
            return restored

        # An jeder Transiente: 8ms Dry-Boost
        blend = np.ones(len(rst_mono), dtype=np.float64)
        for i in np.where(transients)[0]:
            t0 = max(0, i - win_short // 2)
            t1 = min(len(blend), i + win_short // 2)
            # Hann-Fenster um den Transienten-Kern
            w = np.hanning(t1 - t0)
            blend[t0:t1] = np.minimum(blend[t0:t1], 1.0 - _TRANSIENT_DRY_BOOST * w)

        # Blend: (1-blend)*ref + blend*rst → mehr Ref an Transienten
        if is_stereo:
            blend_bc = blend[np.newaxis, :]
        else:
            blend_bc = blend

        n_punched = int(np.sum(transients))
        logger.info("§2.72 Transient-Punch: %d Transienten mit Dry-Boost", n_punched)

        return np.clip(  # type: ignore[no-any-return]
            blend_bc * restored + (1.0 - blend_bc) * reference,
            -1.0,
            1.0,
        ).astype(np.float64)

    except Exception as exc:
        logger.debug("§2.72 Transient-Punch nicht blockierend: %s", exc)
        return restored
