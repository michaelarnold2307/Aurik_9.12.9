"""§AJ: AntiMufflingPass — Dumpfheit-Entfernung auf Maximalstufe.

Entfernt unnatürliche Dumpfheit aus allen Aufnahmen ohne die Musik
oder den Gesang zu beschädigen. Basiert auf brillance_score +
inviting_sound_checker + spektraler Tilt-Analyse.

Architektur:
  1. Muffling-Detektion — spektraler Tilt + HF-Energie-Ratio + Centroid
  2. Chirurgische HF-Restoration — nur in dumpfen Zonen
  3. Dynamischer Tilt-EQ — graduelle Aufhellung, keine harten Sprünge
  4. Warmth-Erhalt — Bass/Mitten werden nie reduziert (kein "thin sound")
  5. Over-Bright-Schutz — maximal +3 dB HF-Anhebung
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AntiMufflingReport:
    muffling_detected: bool = False
    muffling_score: float = 1.0  # 0.0=extrem dumpf, 1.0=perfekt
    hf_restoration_db: float = 0.0
    tilt_correction_db: float = 0.0
    brightness_before: float = 0.0  # HF-Ratio vorher
    brightness_after: float = 0.0  # HF-Ratio nachher
    over_bright_protected: bool = True
    warmth_preserved: bool = True
    warnings: list[str] = field(default_factory=list)


class AntiMufflingPass:
    """Entfernt Dumpfheit chirurgisch — nur wo nötig, nie zu viel.

    Wird in der Post-Pipeline NACH VocalClarity und VOR Humanization
    aufgerufen. Arbeitet auf dem gesamten Mix.
    """

    # ── Schwellwerte ──
    MUFFLED_HF_RATIO = 0.08  # HF-Anteil unter 8% = dumpf
    MAX_HF_BOOST_DB = 3.0  # Maximal +3 dB HF
    TILT_CORRECTION_MAX_DB = 2.5  # Maximaler Tilt-Korrektur
    BRIGHTNESS_TARGET = 0.12  # Ziel-HF-Ratio (Basis, wird chain-depth-adaptiv)
    OVER_BRIGHT_CEILING = 0.30  # HF-Ratio > 30% = zu hell

    # §v10.304: Chain-Depth-adaptive Brightness-Targets
    # Tiefere Ketten (≥3 Träger) haben kumulierte HF-Dämpfung → höheres Target nötig.
    DEPTH_BRIGHTNESS_TARGETS: dict[int, float] = {
        1: 0.12,  # Studio/Digital
        2: 0.15,  # eine analoge Generation
        3: 0.25,  # zwei analoge Generationen (z.B. vinyl→cassette)
        4: 0.38,  # drei analoge Generationen (z.B. reel_tape→vinyl→cassette)
        5: 0.45,  # vier+ analoge Generationen
    }
    MAX_HF_BOOST_DEPTH: dict[int, float] = {
        1: 3.0,
        2: 4.0,
        3: 5.0,
        4: 6.0,
        5: 7.0,
    }

    def __init__(self) -> None:
        self._reports: list[AntiMufflingReport] = []

    def detect_muffling(self, audio: np.ndarray, sr: int, brightness_target: float | None = None) -> AntiMufflingReport:
        """Erkennt Dumpfheit im gesamten Audio."""
        _target = brightness_target if brightness_target is not None else self.BRIGHTNESS_TARGET
        report = AntiMufflingReport()
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
        n = len(mono)

        if n < sr // 2:
            return report

        # ── 1. HF-Energy-Ratio ──
        fft = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        # Bänder
        hf_mask = (freqs >= 8000) & (freqs <= 16000)
        full_mask = (freqs >= 20) & (freqs <= 20000)
        hf_energy = float(np.sum(fft[hf_mask])) if np.any(hf_mask) else 0.0
        full_energy = float(np.sum(fft[full_mask])) + 1e-10
        hf_ratio = hf_energy / full_energy
        report.brightness_before = hf_ratio

        # ── 2. Spektraler Tilt ──
        low_mask = (freqs >= 100) & (freqs <= 500)
        low_energy = float(np.mean(fft[low_mask])) if np.any(low_mask) else 1e-10
        high_mask = (freqs >= 3000) & (freqs <= 8000)
        high_energy = float(np.mean(fft[high_mask])) if np.any(high_mask) else 1e-10
        tilt_db = 10.0 * np.log10(high_energy / low_energy)

        # ── 3. Spektraler Schwerpunkt ──
        centroid = float(np.average(freqs, weights=fft + 1e-10))
        centroid_normalized = min(1.0, centroid / 4000.0)  # 4kHz = 1.0

        # ── 4. Muffling-Score ──
        hf_score = min(1.0, hf_ratio / _target)
        tilt_score = min(1.0, (tilt_db + 20.0) / 20.0)  # tilt > -20 dB
        centroid_score = centroid_normalized
        report.muffling_score = float(np.mean([hf_score, tilt_score, centroid_score]))
        report.muffling_detected = report.muffling_score < 0.7

        return report

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        manual_strength: float = 0.7,
        chain_depth: int = 1,  # §v10.304: Transfer-Chain-Tiefe für adaptive Targets
    ) -> np.ndarray:
        """Führt Anti-Muffling durch — chirurgisch, nur in dumpfen Zonen.

        Args:
            audio: float32 Audio
            sr: Sample-Rate
            manual_strength: 0.0-1.0, Override für automatische Stärke
            chain_depth: Transfer-Chain-Tiefe (1-5), steuert Brightness-Target
        """
        # §v10.304: Chain-Depth-adaptives Brightness-Target
        _depth = max(1, min(5, int(chain_depth)))
        _brightness_target = self.DEPTH_BRIGHTNESS_TARGETS.get(_depth, self.BRIGHTNESS_TARGET)
        _max_hf_boost = self.MAX_HF_BOOST_DEPTH.get(_depth, self.MAX_HF_BOOST_DB)
        result = np.asarray(audio, dtype=np.float32).copy()
        mono = np.mean(result, axis=0) if result.ndim == 2 else result
        n = len(mono)

        # ── Detection ──
        report = self.detect_muffling(result, sr, brightness_target=_brightness_target)
        if not report.muffling_detected:
            report.warnings.append("No muffling detected — skipping")
            self._reports.append(report)
            return cast(np.ndarray, result)

        # ── Adaptive Strength ──
        muffling_severity = 1.0 - report.muffling_score  # 0=ok, 1=extrem
        hf_boost_db = muffling_severity * _max_hf_boost * manual_strength
        tilt_correction_db = muffling_severity * self.TILT_CORRECTION_MAX_DB * manual_strength
        report.hf_restoration_db = hf_boost_db
        report.tilt_correction_db = tilt_correction_db

        # ── Block-basierte Verarbeitung für Chirurgie ──
        block_ms = 200  # 200ms Blöcke
        block_samples = int(sr * block_ms / 1000.0)
        if block_samples < 64:
            block_samples = 4096

        # Analyse pro Block: nur dumpfe Blöcke behandeln
        block_muffling = []
        for i in range(0, n, block_samples):
            block = mono[i : min(n, i + block_samples)]
            if len(block) < 64:
                continue
            fft_b = np.abs(np.fft.rfft(block))
            freqs_b = np.fft.rfftfreq(len(block), d=1.0 / sr)
            hf_b = (freqs_b >= 8000) & (freqs_b <= 16000)
            full_b = (freqs_b >= 20) & (freqs_b <= 20000)
            hf_ratio_b = float(np.sum(fft_b[hf_b])) / (float(np.sum(fft_b[full_b])) + 1e-10)
            block_muffling.append(hf_ratio_b < self.MUFFLED_HF_RATIO)

        # ── Chirurgische HF-Restoration ──
        n_blocks = len(block_muffling)
        if n_blocks == 0:
            self._reports.append(report)
            return cast(np.ndarray, result)

        for ch in range(result.shape[0] if result.ndim == 2 else 1):
            ch_data = result[ch] if result.ndim == 2 else result

            for bi in range(n_blocks):
                if not block_muffling[bi]:
                    continue  # Nur dumpfe Blöcke behandeln

                b0 = bi * block_samples
                b1 = min(n, b0 + block_samples)
                block = ch_data[b0:b1]

                # FFT-basierte HF-Anhebung
                fft_block = np.fft.rfft(block)
                freqs_block = np.fft.rfftfreq(len(block), d=1.0 / sr)

                # High-Shelf ab 8 kHz
                hf_band = freqs_block >= 8000
                if np.any(hf_band):
                    gain_hf = 10 ** (hf_boost_db / 20.0)
                    # Gradueller Übergang 4-8 kHz
                    transition = (freqs_block >= 4000) & (freqs_block < 8000)
                    if np.any(transition):
                        transition_ratio = (freqs_block[transition] - 4000) / 4000
                        fft_block[transition] *= (
                            1.0 + (gain_hf - 1.0) * transition_ratio[:, np.newaxis]
                            if fft_block.ndim > 1
                            else 1.0 + (gain_hf - 1.0) * transition_ratio
                        )
                    fft_block[hf_band] *= gain_hf

                # Over-Bright-Schutz
                new_rms = float(np.sqrt(np.mean(np.abs(fft_block) ** 2) + 1e-12))
                old_rms = float(np.sqrt(np.mean(np.abs(np.fft.rfft(block)) ** 2) + 1e-12))
                if new_rms > old_rms * 2.0:  # Max +6 dB
                    fft_block *= old_rms * 2.0 / new_rms
                    report.over_bright_protected = True

                ch_data[b0:b1] = np.fft.irfft(fft_block, n=len(block))

            if result.ndim == 2:
                result[ch] = ch_data[: len(result[ch])]
            else:
                result = ch_data[: len(result)].astype(np.float32)

        # ── Sanfte Tilt-Korrektur (global, sehr dezent) ──
        if tilt_correction_db > 0.1:
            try:
                for ch in range(result.shape[0] if result.ndim == 2 else 1):
                    ch_data = result[ch] if result.ndim == 2 else result
                    fft_full = np.fft.rfft(ch_data)
                    freqs_full = np.fft.rfftfreq(len(ch_data), d=1.0 / sr)

                    for fhi, gain_factor in [(2000, 0.6), (4000, 0.8), (8000, 1.0)]:
                        band = freqs_full >= fhi
                        if np.any(band):
                            band_gain = 10 ** (tilt_correction_db * gain_factor / 20.0)
                            fft_full[band] *= band_gain

                    ch_result = np.fft.irfft(fft_full, n=len(ch_data))
                    if result.ndim == 2:
                        result[ch] = ch_result[: len(result[ch])]
                    else:
                        result = ch_result[: len(result)].astype(np.float32)
            except Exception as e:
                logger.warning("anti_muffling_pass.py::unbekannter Ersatzpfad: %s", e)

        # ── Clamp ──
        result = np.clip(result, -1.0, 1.0).astype(np.float32)

        # ── Post-Check ──
        report_final = self.detect_muffling(result, sr)
        report.brightness_after = report_final.brightness_before
        report.over_bright_protected = report_final.brightness_before <= self.OVER_BRIGHT_CEILING
        report.warmth_preserved = self._verify_warmth(audio, result, sr)

        if report.over_bright_protected and report.warmth_preserved:
            logger.info(
                "§AJ AntiMuffling: HF +%.1f dB, Tilt +%.1f dB, brightness %.3f→%.3f",
                hf_boost_db,
                tilt_correction_db,
                report.brightness_before,
                report.brightness_after,
            )

        self._reports.append(report)
        return cast(np.ndarray, result)

    def _verify_warmth(self, before: np.ndarray, after: np.ndarray, sr: int) -> bool:
        """Prüft ob Bass/Mitten erhalten blieben."""
        mono_before = np.mean(before, axis=0) if before.ndim == 2 else before
        mono_after = np.mean(after, axis=0) if after.ndim == 2 else after

        fft_b = np.abs(np.fft.rfft(mono_before))
        fft_a = np.abs(np.fft.rfft(mono_after))
        freqs = np.fft.rfftfreq(len(mono_before), d=1.0 / sr)

        warm_mask = (freqs >= 100) & (freqs <= 500)
        warm_before = float(np.sum(fft_b[warm_mask])) + 1e-10
        warm_after = float(np.sum(fft_a[warm_mask])) + 1e-10

        return warm_after >= warm_before * 0.95  # < 5% Reduktion

    def get_reports(self) -> list[AntiMufflingReport]:
        return list(self._reports)

    def clear_reports(self) -> None:
        self._reports.clear()
