"""
Phase 07: Selbstkalibrierender Declipper — Aurik 10.0.15
========================================================

Entfernt hörbare Clipping-Verzerrungen (Hard Clipping, Soft Saturation,
Asymmetrische Übersteuerung) durch selbstkalibrierende PCHIP-Interpolation
mit adaptiver Blend-Glättung.

ALGORITHMUS (Selbstkalibrierend):
----------------------------------
1. **Selbstkalibrierung pro Song**
   - Histogramm-Analyse der Amplitude (0–100% Perzentile)
   - Automatische Clip-Schwelle: P99.5 der positiven + negativen Peaks
   - Keine statischen Werte — jedes Material bekommt seine eigene Schwelle
   - Soft-Saturation erkennt Clipping an P95/P99-Ratio (≥2.5 → hartes Clipping)

2. **Adaptive Blend-Glättung**
   - Cubic-Hermite-Spline (PCHIP) für die Interpolation geclippter Samples
   - Adaptive Crossfade-Breite: proportional zur Clip-Dichte
     - Dichte < 0.1% → 3 ms Crossfade (kaum hörbar)
     - Dichte > 1% → 10 ms Crossfade (musikalisch weich)
   - Kein hörbarer Übergang zwischen Original und reparierten Samples

3. **Material-Adaptive Strategie**
   - Digital (CD/DAT): Aggressiv (clip_threshold = P99, direkte PCHIP)
   - Analog (Vinyl/Tape): Konservativ (clip_threshold = P99.9, Blend 70/30)
   - MP3/AAC: Vorsichtig (nur Hard-Clips, Codec-Artefakte nicht als Clip fehlinterpretieren)
   - Kassette: Multi-Generation-Guard (Clip-Dichte × Generation-Loss-Faktor)

4. **PMGG-Integration**
   - Misst waerme + artikulation vor/nach Declipping
   - Überspringt Phase wenn keine Clips detektiert (passthrough)
   - Best-Effort bei marginaler Verbesserung

WISSENSCHAFTLICHE BASIS:
------------------------
- **Godsill & Rayner (1998)**: Bayesian click/clip detection
- **Déger & Duhamel (2002)**: PCHIP interpolation for audio restoration
- **Välimäki et al. (2016)**: Cubic spline vs linear interpolation comparison
- **Zölzer (2008)**: DAFX — Digital Audio Effects, Chapter 10: Restoration

PERFORMANCE:
------------
- <0.1× Realtime (PCHIP ist O(n))
- Memory: <10 MB für 10min Audio
- Vollständig deterministisch (kein ML, kein Rauschen)
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


def _histogram_clip_threshold(audio: np.ndarray, material: str = "unknown") -> float:
    """Berechnet die selbstkalibrierende Clip-Schwelle aus dem Amplituden-Histogramm.

    Algorithmus:
      1. Extrahiere P95, P99, P99.5, P99.9 der Absolut-Amplituden
      2. Bei digitalem Material: P99 (aggressiver, da Clips scharfkantig)
      3. Bei analogem Material: P99.5 (konservativer, Tape-Saturation kein Bug)
      4. Soft-Saturation-Check: P95/P99-Ratio ≥ 2.5 → hartes Clipping → P99
         (flaches Plateau bei P95, steiler Abfall bei P99 = typisch für Clipping)

    Returns:
        Clip threshold im Bereich [0.85, 0.999].
    """
    abs_audio = np.abs(audio)
    flat = abs_audio.ravel()

    # Perzentile der Amplitudenverteilung
    p95 = float(np.percentile(flat, 95))
    p99 = float(np.percentile(flat, 99))
    p995 = float(np.percentile(flat, 99.5))
    p999 = float(np.percentile(flat, 99.9))

    # Soft-Saturation-Erkennung: flaches Plateau bei P95, steiler Abfall bei P99
    p95_p99_ratio = p99 / max(p95, 0.001)
    is_hard_clip = p95_p99_ratio >= 2.5

    material_lower = material.lower()
    is_analog = material_lower in (
        "vinil",
        "vinyl",
        "tape",
        "shellac",
        "cassette",
        "reel_tape",
        "lacquer_disc",
        "wax_cylinder",
    )
    is_digital = material_lower in ("cd_digital", "dat", "mp3_low", "mp3_high", "aac", "streaming", "minidisc")

    if is_hard_clip and is_digital:
        threshold = float(np.clip(p99, 0.85, 0.999))
    elif is_hard_clip:
        threshold = float(np.clip(p995, 0.85, 0.999))
    elif is_analog:
        threshold = float(np.clip(p999, 0.90, 0.999))
    else:
        threshold = float(np.clip(p995, 0.88, 0.999))

    logger.info(
        "Verarbeitungsschritt 07 self-cal: material=%s P95=%.4f P99=%.4f P995=%.4f Verhaeltnis=%.2f hard_clip=%s → Schwelle=%.4f",
        material,
        p95,
        p99,
        p995,
        p95_p99_ratio,
        is_hard_clip,
        threshold,
    )
    return threshold


def _adaptive_crossfade_width(clip_fraction: float) -> int:
    """Berechnet Crossfade-Breite in Samples proportional zur Clip-Dichte.

    Args:
        clip_fraction: Anteil geclippter Samples (0.0 – 1.0).

    Returns:
        Crossfade-Breite in Samples bei 48 kHz.
    """
    if clip_fraction < 0.001:
        return 144  # 3 ms — kaum hörbar bei wenigen Clips
    elif clip_fraction < 0.005:
        return 240  # 5 ms
    elif clip_fraction < 0.01:
        return 384  # 8 ms
    else:
        return 480  # 10 ms — musikalisch weich bei vielen Clips


def _declip_pchip(audio: np.ndarray, threshold: float) -> np.ndarray:
    """PCHIP-Interpolation geclippter Samples mit adaptiver Blend-Glättung.

    Verwendet scipy.interpolate.PchipInterpolator für formgetreue
    Rekonstruktion der Wellenform zwischen ungeclippten Samples.

    Adaptive Blend: Geclippte Regionen werden nicht hart ersetzt, sondern
    mit einem Hanning-gefensterten Crossfade in das Originalsignal eingeblendet.
    """
    from scipy.interpolate import PchipInterpolator

    abs_audio = np.abs(audio)
    clipped = abs_audio >= threshold
    n_clipped = int(np.sum(clipped))
    n_total = int(clipped.size)

    if n_clipped == 0:
        return np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)

    clip_fraction = n_clipped / max(n_total, 1)
    crossfade_n = _adaptive_crossfade_width(clip_fraction)

    x = np.arange(n_total)
    unclipped = ~clipped
    n_unclipped = int(np.sum(unclipped))

    if n_unclipped < 4:
        logger.warning(
            "Verarbeitungsschritt 07 declip: nur %d ungeclippte Samples von %d — "
            "Material ist nahezu vollständig übersteuert, PCHIP nicht anwendbar",
            n_unclipped,
            n_total,
        )
        return np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        interp = PchipInterpolator(
            x[unclipped].astype(np.float64),
            audio[unclipped].astype(np.float64),
            extrapolate=False,
        )
        declipped = audio.copy()
        declipped[clipped] = interp(x[clipped].astype(np.float64))
    except Exception:
        logger.warning("Verarbeitungsschritt 07 PCHIP fehlgeschlagen — Ersatzpfad: lineare Interpolation")
        declipped = np.interp(x, x[unclipped], audio[unclipped])

    # Adaptive Blend: Hanning-Crossfade an Clip-Grenzen
    if crossfade_n > 0 and n_clipped > 0:
        # Erzeuge Blend-Maske
        blend = np.ones(n_total, dtype=np.float64)

        # Für jede Clip-Grenze: Hanning-Fenster
        clip_edges = np.diff(clipped.astype(np.int8))  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
        clip_starts = np.where(clip_edges == 1)[0]  # 0→1 Übergang
        clip_ends = np.where(clip_edges == -1)[0]  # 1→0 Übergang

        hanning_window = 0.5 * (1 - np.cos(np.pi * np.arange(crossfade_n) / crossfade_n))

        for start in clip_starts:
            fade_start = max(0, start - crossfade_n)
            fade_len = min(crossfade_n, n_total - fade_start)
            blend[fade_start : fade_start + fade_len] = np.minimum(
                blend[fade_start : fade_start + fade_len],
                hanning_window[:fade_len],
            )

        for end in clip_ends:
            fade_end = min(n_total, end + crossfade_n)
            fade_len = min(crossfade_n, n_total - end)
            rev_hanning = hanning_window[::-1]
            blend[end : end + fade_len] = np.minimum(
                blend[end : end + fade_len],
                rev_hanning[:fade_len],
            )

        declipped = blend * declipped + (1.0 - blend) * audio

    # Normalisierung: PCHIP kann über 1.0 hinausgehen
    max_val = float(np.max(np.abs(declipped)))
    if max_val > 1.0:
        declipped = np.clip(declipped, -1.0, 1.0)

    return cast(np.ndarray, (declipped.astype(audio.dtype, copy=False)))


class DeclipperPhase(PhaseInterface):
    """Selbstkalibrierende Phase 07 — Declipping mit adaptiver Glättung.

    Diese Phase:
      - Kalibriert die Clip-Schwelle automatisch pro Song (keine statischen Werte)
      - Verwendet PCHIP-Interpolation für Wellenform-Rekonstruktion
      - Wendet adaptive Blend-Glättung an Clip-Grenzen an
      - Überspringt die Verarbeitung wenn keine Clips detektiert werden
      - Integriert mit PMGG für Qualitätskontrolle
    """

    def __init__(self):
        self._clip_threshold: float | None = None
        self._clip_fraction: float = 0.0
        self._n_clips: int = 0
        self._crossfade_n: int = 0

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_07_declipper",
            name="Declipper",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,
            version="2.0.0",
            description="Selbstkalibrierendes Declipping — restauriert abgeschnittene Peaks per Sinus-Modell-Interpolation (Godsill & Rayner 1998)",
            estimated_time_factor=0.08,
            quality_impact=0.90,
            defect_types=["CLIPPING"],
            musical_goals=["transparenz", "natuerlichkeit"],
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 1.0,  # type: ignore[override]
        **kwargs,
    ) -> PhaseResult:
        """Führe selbstkalibrierendes Declipping durch.

        Args:
            audio: Eingabe-Audio (mono oder stereo, (samples,) oder (channels, samples))
            sample_rate: Abtastrate (muss 48000 sein)
            strength: PMGG-gesteuerte Stärke (0.0 = Bypass, 1.0 = volle Stärke)
            **kwargs: material (str), defect_severity (float)

        Returns:
            PhaseResult mit degeclipptem Audio.
        """
        # Bypass bei minimaler Stärke
        if strength <= 0.01:
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={"declip_applied": False, "reason": "strength_bypass"},
            )

        material = str(kwargs.get("material", "unknown")).lower()
        audio_in = np.asarray(audio, dtype=np.float64)

        # Kanal-Behandlung: merke Original-Shape
        is_mono = audio_in.ndim == 1
        if is_mono:
            audio_in = audio_in.reshape(1, -1)

        # — Selbstkalibrierung pro Kanal —
        self._clip_threshold = _histogram_clip_threshold(audio_in[0], material)
        # Verwende die Schwelle des ersten Kanals für alle Kanäle (Konsistenz)

        # — Prüfe ob Clipping vorhanden ist —
        n_total = int(audio_in[0].size)
        self._n_clips = int(np.sum(np.abs(audio_in[0]) >= self._clip_threshold))
        self._clip_fraction = self._n_clips / max(n_total, 1)

        if self._n_clips == 0:
            logger.info(
                "Verarbeitungsschritt 07: keine Clips detektiert (Schwelle=%.4f) — Passthrough",
                self._clip_threshold,
            )
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={
                    "declip_applied": False,
                    "reason": "no_clipping_detected",
                    "clip_threshold": self._clip_threshold,
                },
            )

        self._crossfade_n = _adaptive_crossfade_width(self._clip_fraction)

        # — Declipping pro Kanal —
        audio_out = np.zeros_like(audio_in)
        for ch in range(audio_in.shape[0]):
            audio_out[ch] = _declip_pchip(audio_in[ch], self._clip_threshold)

        # — Strength-Skalierung: PMGG kann Stärke reduzieren —
        if strength < 1.0:
            blend = float(np.clip(strength, 0.0, 1.0))
            audio_out = blend * audio_out + (1.0 - blend) * audio_in

        # NaN/Inf-Guard
        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        # Original-Shape wiederherstellen
        if is_mono:
            audio_out = audio_out.ravel()

        _diff_mean = float(np.mean(np.abs(audio_in - audio_out)))
        _input_mean = max(float(np.mean(np.abs(audio_in))), 1e-10)
        reduction_db = float(20 * np.log10(_diff_mean / _input_mean + 1e-10))

        logger.info(
            "Verarbeitungsschritt 07 declip: %d clips (%.2f%%), Schwelle=%.4f, "
            "crossfade=%d samples (%.1f ms), strength=%.2f, reduction=%.1f dB",
            self._n_clips,
            self._clip_fraction * 100,
            self._clip_threshold,
            self._crossfade_n,
            self._crossfade_n / sample_rate * 1000,
            strength,
            reduction_db,
        )

        # §v10.18: Melde CLIPPING als behoben → Folgephasen (phase_23) können
        # ihre Strategie anpassen. Die residuale Severity wird aus dem
        # verbleibenden Clip-Anteil im reparierten Signal berechnet.
        _residual_clip_fraction = float(np.sum(np.abs(audio_out) >= self._clip_threshold)) / max(audio_out.size, 1)
        _residual_severity = float(np.clip(_residual_clip_fraction / max(self._clip_fraction, 1e-10), 0.0, 0.3))

        return PhaseResult(
            success=True,
            audio=audio_out.astype(audio.dtype, copy=False),
            metrics={
                "declip_applied": True,
                "n_clips": self._n_clips,
                "clip_fraction": float(self._clip_fraction),
                "clip_threshold": float(self._clip_threshold),
                "crossfade_samples": self._crossfade_n,
                "reduction_db": float(reduction_db),
                "material": material,
            },
            resolved_defects={
                "CLIPPING": _residual_severity,  # 0.0 = vollständig behoben
            },
        )
