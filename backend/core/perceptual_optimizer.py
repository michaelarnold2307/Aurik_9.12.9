"""
PerceptualOptimizer — §CROWN Closed-Loop Optimization
=======================================================

Ersetzt die lineare Pipeline durch einen Optimierungs-Loop:
  Statt: Phase1 → Phase2 → ... → Phase66
  Sondern: Parallele Strategien → Per-Segment-Auswahl → Iteration

Prinzip:
  1. Segmentiere Audio in 3-10s Blöcke
  2. Pro Segment: Wende 3-5 Strategien parallel an
  3. Wähle die Strategie mit dem höchsten Perceptual-Score
  4. Cross-Fade zwischen Segmenten
  5. Iteriere bis Konvergenz (ΔMOS < 0.01)

Kein Training. Keine externen Daten. Reine Optimierung.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Eine Restaurierungs-Strategie = spezifische Phase-Kombination."""

    name: str = ""
    phases: list[str] = field(default_factory=list)  # Phase-IDs
    strength: float = 1.0
    description: str = ""


@dataclass
class SegmentResult:
    """Ergebnis einer Strategie auf einem Segment."""

    strategy_name: str = ""
    audio: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    perceptual_score: float = 0.0
    harmonic_score: float = 0.0
    naturalness_score: float = 0.0
    rms_delta_db: float = 0.0


@dataclass
class OptimizationResult:
    """Ergebnis der Optimierung."""

    audio: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    iterations: int = 0
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement: float = 0.0
    segment_strategies: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


class PerceptualOptimizer:
    """Geschlossener Optimierungs-Regelkreis.

    Verwendung:
        opt = PerceptualOptimizer(restorer)
        result = opt.optimize(audio, sr, material="vinyl", era=1970)
    """

    # ── Konfiguration ─────────────────────────────────────────────────
    SEGMENT_DURATION_S: float = 8.0  # Segment-Länge
    CROSSFADE_S: float = 0.15  # Cross-Fade zwischen Segmenten
    MAX_ITERATIONS: int = 3  # Max Optimierungs-Iterationen
    CONVERGENCE_THRESHOLD: float = 0.01  # ΔMOS < 0.01 → konvergiert
    MIN_IMPROVEMENT: float = 0.02  # Min Verbesserung pro Iteration

    # Strategie-Definitionen (increasing intensity)
    STRATEGIES: list[Strategy] = [
        Strategy("passthrough", [], 0.0, "Keine Änderung — Original"),
        Strategy(
            "light",
            ["phase_01_click_removal", "phase_03_denoise", "phase_05_rumble_filter"],
            0.5,
            "Leichte Defekt-Entfernung",
        ),
        Strategy(
            "balanced",
            [
                "phase_01_click_removal",
                "phase_03_denoise",
                "phase_04_eq_correction",
                "phase_05_rumble_filter",
                "phase_09_crackle_removal",
            ],
            0.7,
            "Ausgewogene Restaurierung",
        ),
        Strategy(
            "deep",
            [
                "phase_01_click_removal",
                "phase_03_denoise",
                "phase_04_eq_correction",
                "phase_05_rumble_filter",
                "phase_08_transient_preservation",
                "phase_09_crackle_removal",
                "phase_12_wow_flutter_fix",
            ],
            0.9,
            "Tiefe Restaurierung",
        ),
        Strategy(
            "full",
            [
                "phase_01_click_removal",
                "phase_03_denoise",
                "phase_04_eq_correction",
                "phase_05_rumble_filter",
                "phase_08_transient_preservation",
                "phase_09_crackle_removal",
                "phase_12_wow_flutter_fix",
                "phase_19_de_esser",
                "phase_28_surface_noise_profiling",
            ],
            1.0,
            "Vollständige Restaurierung",
        ),
    ]

    def __init__(self, restorer: Any | None = None):
        self._restorer = restorer

    def optimize(
        self,
        audio: np.ndarray,
        sr: int,
        material: str = "unknown",
        era: int = 0,
        genre: str = "unknown",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> OptimizationResult:
        """Führt Closed-Loop-Optimierung durch.

        Args:
            audio: Input-Audio
            sr: Sample-Rate
            material: Material-Typ
            era: Jahrzehnt
            genre: Genre
            progress_callback: (pct, msg) → None

        Returns:
            OptimizationResult mit optimiertem Audio
        """
        t0 = time.time()
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)

        # 1. Initial-Score messen
        initial_score = self._perceptual_score(mono, sr)
        current_audio = mono.copy()
        current_score = initial_score

        if progress_callback:
            progress_callback(0.0, "Optimiere...")

        segment_strategies: list[str] = []
        iterations = 0

        for iteration in range(self.MAX_ITERATIONS):
            iterations += 1
            iteration_improved = False

            # 2. Segmentieren
            segments = self._segment(current_audio, sr)

            # 3. Pro Segment: beste Strategie finden
            optimized_segments = []
            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                seg_audio = current_audio[seg_start:seg_end]

                # Wende alle Strategien parallel an
                candidates: list[SegmentResult] = []
                for strategy in self.STRATEGIES:
                    try:
                        result = self._apply_strategy(seg_audio, sr, strategy, material, era)
                        candidates.append(result)
                    except Exception as e:
                        logger.debug("Strategy %s fehlgeschlagen: %s", strategy.name, e)

                if not candidates:
                    optimized_segments.append(seg_audio)
                    continue

                # Wähle beste Strategie
                best = max(candidates, key=lambda r: r.perceptual_score)
                optimized_segments.append(best.audio)

                if best.strategy_name != "passthrough":
                    segment_strategies.append(best.strategy_name)
                    if best.perceptual_score > current_score + self.MIN_IMPROVEMENT:
                        iteration_improved = True

                if progress_callback:
                    pct = (iteration * len(segments) + seg_idx + 1) / (self.MAX_ITERATIONS * len(segments))
                    progress_callback(
                        pct, f"Iter {iteration + 1} Seg {seg_idx + 1}/{len(segments)}: {best.strategy_name}"
                    )

            # 4. Stitche Segmente mit Cross-Fade
            current_audio = self._stitch(optimized_segments, sr)

            # 5. Score nach Iteration
            new_score = self._perceptual_score(current_audio, sr)

            logger.info(
                "Iteration %d: %.3f → %.3f (Δ=%.3f)",
                iteration + 1,
                current_score,
                new_score,
                new_score - current_score,
            )

            # 6. Konvergenz-Prüfung
            if abs(new_score - current_score) < self.CONVERGENCE_THRESHOLD:
                logger.info("Konvergenz erreicht nach %d Iterationen", iteration + 1)
                break

            if not iteration_improved and iteration > 0:
                logger.info("Keine Verbesserung — Abbruch")
                break

            current_score = new_score

        elapsed = time.time() - t0
        improvement = current_score - initial_score

        return OptimizationResult(
            audio=current_audio.astype(np.float32),
            iterations=iterations,
            initial_score=initial_score,
            final_score=current_score,
            improvement=improvement,
            segment_strategies=segment_strategies,
            elapsed_s=elapsed,
        )

    # ── Interne Methoden ──────────────────────────────────────────────

    def _segment(self, audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
        """Segmentiert Audio in überlappende Blöcke."""
        seg_samples = int(self.SEGMENT_DURATION_S * sr)
        n = len(audio)
        if n <= seg_samples:
            return [(0, n)]

        segments = []
        overlap = int(self.CROSSFADE_S * sr)
        pos = 0
        while pos < n:
            end = min(pos + seg_samples, n)
            segments.append((pos, end))
            pos += seg_samples - overlap
        return segments

    def _apply_strategy(self, audio: np.ndarray, sr: int, strategy: Strategy, material: str, era: int) -> SegmentResult:
        """Wendet eine Strategie auf ein Segment an."""
        if strategy.name == "passthrough" or not strategy.phases:
            score = self._perceptual_score(audio, sr)
            return SegmentResult(
                strategy_name=strategy.name,
                audio=audio.copy(),
                perceptual_score=score,
            )

        # Nutze Restorer wenn verfügbar
        if self._restorer is not None:
            try:
                result = self._restorer.restore(
                    audio,
                    sr,
                    override_phases=strategy.phases,
                    strength=strategy.strength,
                    material=material,
                    era=era,
                )
                processed = result.audio if hasattr(result, "audio") else audio
            except Exception:
                processed = audio
        else:
            processed = audio  # Fallback: kein Restorer verfügbar

        score = self._perceptual_score(processed, sr)
        return SegmentResult(
            strategy_name=strategy.name,
            audio=processed,
            perceptual_score=score,
        )

    def _stitch(self, segments: list[np.ndarray], sr: int) -> np.ndarray:
        """Sticht Segmente mit Cross-Fade zusammen."""
        if len(segments) == 1:
            return segments[0]

        crossfade_samples = int(self.CROSSFADE_S * sr)
        total_len = sum(len(s) for s in segments) - crossfade_samples * (len(segments) - 1)
        result = np.zeros(total_len, dtype=np.float32)

        pos = 0
        for i, seg in enumerate(segments):
            seg_len = len(seg)
            if i == 0:
                result[pos : pos + seg_len] = seg
                pos += seg_len - crossfade_samples
            elif i == len(segments) - 1:
                # Letztes Segment: Cross-Fade mit Vorgänger
                fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
                fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
                result[pos : pos + crossfade_samples] *= fade_out
                result[pos : pos + crossfade_samples] += seg[:crossfade_samples] * fade_in
                result[pos + crossfade_samples : pos + seg_len] = seg[crossfade_samples:]
            else:
                fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
                fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
                result[pos : pos + crossfade_samples] *= fade_out
                result[pos : pos + crossfade_samples] += seg[:crossfade_samples] * fade_in
                result[pos + crossfade_samples : pos + seg_len - crossfade_samples] = seg[
                    crossfade_samples:-crossfade_samples
                ]
                pos += seg_len - 2 * crossfade_samples

        return cast(np.ndarray, result)

    def _perceptual_score(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet einen kombinierten Perceptual-Score (0-1)."""
        mono = np.asarray(audio, dtype=np.float32).flatten()
        n = len(mono)
        if n < sr // 4:
            # §v10.93: < 250ms — RMS-basierter Perceptual-Proxy
            _rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
            return float(np.clip(_rms * 5.0, 0.30, 0.70))

        # 1. Brightness (HF-Energie)
        n_fft = min(4096, n)
        spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        hf = float(np.sum(spec[freqs >= 4000] ** 2))
        total = float(np.sum(spec**2)) + 1e-10
        brightness = float(np.clip(hf / total, 0.0, 1.0))

        # 2. Naturalness (spektrale Entropie = 1 - flatness)
        log_mean = np.exp(np.mean(np.log(spec + 1e-10)))
        arith_mean = np.mean(spec)
        flatness = log_mean / max(arith_mean, 1e-10)
        naturalness = float(np.clip(1.0 - flatness, 0.0, 1.0))

        # 3. Dynamic Range (P99/P1)
        abs_mono = np.abs(mono)
        p99 = float(np.percentile(abs_mono, 99))
        p1 = float(np.percentile(abs_mono, 1)) + 1e-10
        dr = 20.0 * np.log10(p99 / p1)
        dr_score = float(np.clip(dr / 40.0, 0.0, 1.0))  # 40dB dynamic range = optimal

        # 4. Clipping-Freiheit
        clipped = float(np.mean(np.abs(mono) > 0.98))
        clip_score = float(np.clip(1.0 - clipped * 100, 0.0, 1.0))

        return float(
            np.clip(
                brightness * 0.20 + naturalness * 0.35 + dr_score * 0.25 + clip_score * 0.20,
                0.0,
                1.0,
            )
        )


# ── Singleton ─────────────────────────────────────────────────────────

_optimizer: PerceptualOptimizer | None = None


def get_perceptual_optimizer(restorer: Any | None = None) -> PerceptualOptimizer:
    global _optimizer
    if _optimizer is None or restorer is not None:
        _optimizer = PerceptualOptimizer(restorer)
    return _optimizer
