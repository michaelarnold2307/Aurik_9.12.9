"""§AP: TapeHeadArtifactRepair — Bandkopf-Fehler-Reparatur auf Maximalstufe.

Behebt hörbare Bandkopf-Artefakte die der Scanner nicht oder zu schwach erkannt hat:
  1. Head-Clog — kurzzeitige (<10ms) Pegel-Einbrüche durch Schmutz auf dem Kopf
  2. Head-Contact-Dropout — Amplituden-Modulation durch schwankenden Band-Kopf-Kontakt
  3. Azimuth-Misalignment — Phasen-Drift zwischen L/R bei hohen Frequenzen
  4. HF-Remanence-Loss — periodischer Verlust hoher Frequenzen (Kopf-Verschleiß)

Algorithmus:
  1. Short-Dropout-Detektion (<10ms Pegel-Abfall >6dB)
  2. Chirurgische Interpolation (kubisch, 3-Punkt)
  3. Azimuth-Korrektur via L/R-Phasen-Abgleich >8kHz
  4. HF-Envelope-Glättung bei periodischen Einbrüchen
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


class TapeHeadArtifactRepair:
    """Repariert Bandkopf-Artefakte — direkt, ohne auf Scanner-Erkennung zu warten."""

    def __init__(self) -> None:
        pass

    def repair(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        dropout_threshold_db: float = 4.0,  # Aggressiver: Pegel-Abfall ab 4dB
        dropout_max_ms: float = 15.0,  # Aggressiver: bis 15ms Dropout
        azimuth_correct: bool = True,
        ast_musical_confidence: float = 0.0,  # §v10.304: 0-1, senkt Aggressivität bei Musik
    ) -> np.ndarray:
        """Führt alle Bandkopf-Reparaturen durch.

        Args:
            audio: (channels, samples) float32
            sr: Sample-Rate
            dropout_threshold_db: Pegel-Abfall für Dropout-Erkennung
            dropout_max_ms: Maximale Dropout-Dauer
            azimuth_correct: Azimuth-Korrektur aktivieren
        """
        # §v10.304 AST-Guard: Bei hohem Musik-Instrument-Anteil Schwelle anheben
        # Tiefes chain-Material (depth≥3) hat MP3-Artefakte die als Dropouts fehlklassifiziert werden.
        _ast_factor = 1.0 + ast_musical_confidence * 1.5  # 0→1.0, 0.5→1.75, 1.0→2.5
        _effective_threshold_db = dropout_threshold_db * _ast_factor
        _effective_max_ms = dropout_max_ms * (1.0 - ast_musical_confidence * 0.5)  # kürzere Max-Dauer bei Musik
        _effective_max_ms = max(3.0, _effective_max_ms)  # Minimum 3ms

        result = np.asarray(audio, dtype=np.float32).copy()

        # 1. Short-Dropout-Repair (mit AST-adaptierten Schwellen)
        result = self._repair_short_dropouts(result, sr, _effective_threshold_db, _effective_max_ms)
        if azimuth_correct and result.ndim == 2 and result.shape[0] >= 2:
            result = self._correct_azimuth(result, sr)

        # 3. HF-Envelope-Glättung
        result = self._smooth_hf_envelope(result, sr)

        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    def _repair_short_dropouts(self, audio: np.ndarray, sr: int, threshold_db: float, max_ms: float) -> np.ndarray:
        """Erkennt und repariert kurze Pegel-Einbrüche (Head-Clog/Contact)."""
        result = np.asarray(audio, dtype=np.float32).copy()
        mono = np.mean(result, axis=0) if result.ndim == 2 else result
        n = len(mono)

        # RMS-Hüllkurve (5ms Fenster, 2.5ms Hop)
        rms_win = int(sr * 0.005)
        hop = rms_win // 2
        if rms_win < 16:
            return cast(np.ndarray, result)

        max_dropout_samples = int(sr * max_ms / 1000.0)
        threshold_linear = 10 ** (-threshold_db / 20.0)

        # Gleitende RMS
        rms_env = np.zeros(n // hop + 1, dtype=np.float32)
        for i in range(0, n - rms_win, hop):
            rms_env[i // hop] = float(np.sqrt(np.mean(mono[i : i + rms_win] ** 2) + 1e-12))

        # Median-Filter für robuste Baseline
        median_win = 10  # 10 Frames = 25ms
        rms_baseline = np.zeros_like(rms_env)
        for i in range(len(rms_env)):
            lo = max(0, i - median_win)
            hi = min(len(rms_env), i + median_win)
            rms_baseline[i] = float(np.median(rms_env[lo:hi])) + 1e-12

        # Dropout-Erkennung: lokales RMS < Baseline * threshold
        dropouts = []
        in_dropout = False
        dropout_start = 0
        for i in range(len(rms_env)):
            is_drop = rms_env[i] < rms_baseline[i] * threshold_linear
            if is_drop and not in_dropout:
                dropout_start = i * hop
                in_dropout = True
            elif not is_drop and in_dropout:
                dropout_end = i * hop
                duration = dropout_end - dropout_start
                if 4 <= duration <= max_dropout_samples:
                    dropouts.append((dropout_start, dropout_end))
                in_dropout = False

        if not dropouts:
            return cast(np.ndarray, result)

        logger.info("§AP TapeHeadRepair: %d short dropouts found", len(dropouts))

        # Reparatur: kubische Interpolation für jeden Kanal
        for ch in range(result.shape[0] if result.ndim == 2 else 1):
            ch_data = result[ch] if result.ndim == 2 else result
            for s0, s1 in dropouts:
                # 3-Punkt-Interpolation: vor, nach, und Mittelpunkt
                pre_idx = max(0, s0 - 4)
                post_idx = min(n - 1, s1 + 4)
                mid_idx = (s0 + s1) // 2

                # Kubische Spline über 3 Stützpunkte
                x = np.array([pre_idx, mid_idx, post_idx], dtype=np.float64)
                y = np.array([ch_data[pre_idx], ch_data[mid_idx], ch_data[post_idx]], dtype=np.float64)
                xi = np.arange(s0, s1, dtype=np.float64)

                try:
                    # Quadratische Interpolation
                    coeffs = np.polyfit(x, y, 2)
                    yi = np.polyval(coeffs, xi)
                    # Cross-fade mit Original (1ms)
                    xfade = min(int(sr * 0.001), (s1 - s0) // 3)
                    if xfade >= 2:
                        win = np.ones(s1 - s0, dtype=np.float32)
                        win[:xfade] = np.linspace(0, 1, xfade)
                        win[-xfade:] = np.linspace(1, 0, xfade)
                        ch_data[s0:s1] = yi.astype(np.float32) * win + ch_data[s0:s1] * (1 - win)
                    else:
                        ch_data[s0:s1] = yi.astype(np.float32)
                except Exception as e:
                    logger.warning("tape_head_artifact_repair.py::_repair_short_dropouts Ersatzpfad: %s", e)

            if result.ndim == 2:
                result[ch] = ch_data[: len(result[ch])]
            else:
                result = ch_data[: len(result)].astype(np.float32)

        return cast(np.ndarray, result)

    def _correct_azimuth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Korrigiert Azimuth-Fehler via L/R-Phasen-Abgleich >8kHz."""
        result = np.asarray(audio, dtype=np.float32).copy()
        if result.ndim < 2 or result.shape[0] < 2:
            return cast(np.ndarray, result)

        left = result[0]
        right = result[1]
        n = len(left)

        # Hochpass >8kHz (Azimuth-Fehler sind am stärksten bei hohen Frequenzen)
        fft_l = np.fft.rfft(left)
        fft_r = np.fft.rfft(right)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        hf_mask = freqs >= 8000
        if not np.any(hf_mask):
            return cast(np.ndarray, result)

        # Phasen-Differenz im HF-Bereich
        phase_diff = np.angle(fft_r[hf_mask]) - np.angle(fft_l[hf_mask])
        mean_phase_diff = float(np.median(phase_diff))

        # Nur korrigieren wenn Abweichung > 5°
        if abs(mean_phase_diff) < np.radians(5):
            return cast(np.ndarray, result)

        # §v10.16 Coherence-Guard: Stereo-Panning erzeugt Phasendifferenzen
        # die KEINE Azimuth-Fehler sind. Nur korrigieren, wenn die Kanäle
        # tatsächlich das gleiche Signal enthalten (korreliert sind).
        # Unkorreliertes Material → Phasenkorrektur = Kammfilter = zerstoerte Baesse.
        _mean_corr_az = float(np.corrcoef(left[: min(n, sr * 5)], right[: min(n, sr * 5)]).flat[1])
        _mean_corr_az = abs(_mean_corr_az) if np.isfinite(_mean_corr_az) else 1.0
        if _mean_corr_az < 0.40:
            logger.info(
                "§AP Azimuth: inter-channel correlation=%.3f < 0.40 — "
                "stereo panning, NOT azimuth error — skipping correction",
                _mean_corr_az,
            )
            return cast(np.ndarray, result)

        logger.info("§AP Azimuth correction: %.1f° Verarbeitungsschritt shift", np.degrees(mean_phase_diff))

        # Phasen-Korrektur auf rechtem Kanal
        fft_r[hf_mask] *= np.exp(-1j * mean_phase_diff)
        right_corrected = np.fft.irfft(fft_r, n=n)
        result[1] = right_corrected[:n].astype(np.float32)

        return cast(np.ndarray, result)

    def _smooth_hf_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Glättet periodische HF-Energie-Einbrüche (Head-Wear)."""
        result = np.asarray(audio, dtype=np.float32).copy()

        for ch in range(result.shape[0] if result.ndim == 2 else 1):
            ch_data = result[ch] if result.ndim == 2 else result
            n = len(ch_data)

            # HF-Hüllkurve (8-16 kHz) via STFT-Approximation
            block = int(sr * 0.050)  # 50ms Blöcke
            hop = block // 2
            hf_energy = []

            for i in range(0, n - block, hop):
                frame = ch_data[i : i + block]
                fft = np.abs(np.fft.rfft(frame))
                freqs_f = np.fft.rfftfreq(len(frame), d=1.0 / sr)
                hf = (freqs_f >= 8000) & (freqs_f <= 16000)
                hf_energy.append(float(np.mean(fft[hf])) if np.any(hf) else 0.0)

            if len(hf_energy) < 4:
                continue

            hf_env = np.array(hf_energy, dtype=np.float32)
            median_hf = float(np.median(hf_env)) + 1e-12

            # Suche nach Einbrüchen >50% unter Median
            for i, energy in enumerate(hf_env):
                if energy < median_hf * 0.5 and energy > 0:
                    # HF-Einbruch: sanft auf Median anheben
                    s0 = i * hop
                    s1 = min(n, s0 + block)
                    gain = median_hf / (energy + 1e-12)
                    gain = min(gain, 2.0)  # Max +6dB

                    # Graduelle Anhebung im HF-Bereich
                    fft_full = np.fft.rfft(ch_data[s0:s1])
                    freqs_full = np.fft.rfftfreq(len(ch_data[s0:s1]), d=1.0 / sr)
                    hf_band = freqs_full >= 8000
                    if np.any(hf_band):
                        fft_full[hf_band] *= 1.0 + (gain - 1.0) * 0.5  # 50% der Korrektur
                        ch_data[s0:s1] = np.fft.irfft(fft_full, n=s1 - s0)

            if result.ndim == 2:
                result[ch] = ch_data[: len(result[ch])]
            else:
                result = ch_data[: len(result)].astype(np.float32)

        return cast(np.ndarray, result)
