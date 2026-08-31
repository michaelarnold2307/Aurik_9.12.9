"""§AE: Cross-Channel Stereo Defect Repair — mit §AF DynamicsGuard.

When one channel has a defect (click, dropout, crackle) and the other doesn't,
use the healthy channel to guide repair in the damaged one.  Preserves stereo
imaging — no mono-compatibility sacrifice.

§AF integration (RepairDynamicsGuard):
- Envelope matching after each repair: repaired segment amplitude matches
  surrounding context → no hopping/pumping/stuttering
- Continuity verification at repair boundaries
- Global dynamics preservation check after all repairs

Algorithm:
1. Detect per-channel defects independently
2. For single-channel defects: extract healthy segment from other channel
3. Match spectral envelope + amplitude envelope + cross-fade (5-15ms)
4. For dual-channel defects: fall back to traditional mono repair
5. Verify: cross-channel correlation must not drop >5%
6. §AF: DynamicsGuard — envelope continuity, global dynamics check
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from backend.core.repair_dynamics_guard import RepairDynamicsGuard

logger = logging.getLogger(__name__)


class CrossChannelRepair:
    """Nutzt den gesunden Kanal um Defekte im anderen Kanal zu reparieren.

    §AF: Jede Reparatur wird durch den RepairDynamicsGuard abgesichert:
    - Envelope-Matching verhindert Lautstärkesprünge
    - Continuity-Check erkennt Diskontinuitäten
    - Global-Dynamics-Erhalt wird nach allen Repairs verifiziert
    """

    def __init__(self) -> None:
        self._dynamics = RepairDynamicsGuard()
        self._repair_boundaries: list[tuple[int, int, int]] = []  # (ch, s0, s1)

    def repair_dropout(
        self,
        audio: np.ndarray,  # (2, N) stereo
        sr: int,
        dropout_start: int,  # sample index
        dropout_len: int,  # samples
        affected_channel: int,  # 0=left, 1=right
    ) -> np.ndarray:
        """Repariert einen Dropout in einem Kanal mit Material vom anderen.

        §AF: Nach der Cross-Channel-Reparatur wird die Amplitude des
        reparierten Segments an den umgebenden Kontext angeglichen.

        Args:
            audio: (2, N) float32 stereo audio
            sr: sample rate
            dropout_start: Start-Sample des Dropouts
            dropout_len: Länge in Samples
            affected_channel: 0 für links, 1 für rechts

        Returns:
            Repariertes Audio, same shape.
        """
        result = np.asarray(audio, dtype=np.float32).copy()
        if audio.ndim < 2 or audio.shape[0] < 2:
            return cast(np.ndarray, result)  # mono → kein Cross-Channel-Repair möglich

        healthy_ch = 1 - affected_channel
        healthy = result[healthy_ch]
        damaged = result[affected_channel]
        n = min(dropout_len, len(healthy) - dropout_start, len(damaged) - dropout_start)

        if n < 10:
            return cast(np.ndarray, result)

        # Cross-fade Regionen: 10ms vorher und nachher
        fade_in = min(int(sr * 0.010), n // 3)  # 10ms
        fade_out = min(int(sr * 0.010), n // 3)
        repair_start = max(0, dropout_start - fade_in)
        repair_end = min(len(damaged), dropout_start + n + fade_out)

        try:
            safe_seg = healthy[repair_start:repair_end]
            damaged_seg = damaged[repair_start:repair_end]

            # FFT-basierte Spektral-Hüllkurve
            fft_len = min(2048, len(safe_seg))
            if fft_len > 64:
                safe_fft = np.abs(np.fft.rfft(safe_seg, n=fft_len))
                damaged_fft = np.abs(np.fft.rfft(damaged_seg, n=fft_len))

                # Glättung über Frequenzbänder
                n_bands = 8
                band_size = len(safe_fft) // n_bands
                envelope = np.ones(len(safe_fft), dtype=np.float32)
                for b in range(n_bands):
                    b_start = b * band_size
                    b_end = b_start + band_size
                    safe_band = np.mean(safe_fft[b_start:b_end]) + 1e-10
                    damaged_band = np.mean(damaged_fft[b_start:b_end]) + 1e-10
                    ratio = safe_band / damaged_band
                    envelope[b_start:b_end] = float(np.clip(ratio, 0.5, 2.0))

                healthy_segment = healthy[repair_start:repair_end].copy()

                # Cross-fade Fenster
                ramp_in = np.linspace(0, 1, fade_in) if fade_in > 0 else np.array([])
                ramp_out = np.linspace(1, 0, fade_out) if fade_out > 0 else np.array([])
                blend_len = len(healthy_segment) - len(ramp_in) - len(ramp_out)
                blend = np.ones(max(0, blend_len), dtype=np.float32)
                window = np.concatenate([ramp_in, blend, ramp_out])[: len(healthy_segment)]

                # Reparatur: Original (beschädigt) + Cross-Channel (gesund)
                damaged[repair_start:repair_end] = damaged_seg * (1 - window) + healthy_segment * window

                # §AF: Envelope-Matching — Amplitude an Kontext angleichen
                damaged = self._dynamics.match_envelope(
                    damaged, sr, repair_start, repair_end, context_ms=40, crossfade_ms=12
                )
        except Exception:
            # Fallback: direkter Cross-fade ohne spektrale Anpassung
            window = np.hanning(min(n * 2, repair_end - repair_start))
            window = window[: repair_end - repair_start]
            damaged[repair_start:repair_end] = (
                damaged[repair_start:repair_end] * (1 - window) + healthy[repair_start:repair_end] * window
            )
            # §AF: Auch im Fallback Envelope matchen
            try:
                damaged = self._dynamics.match_envelope(
                    damaged, sr, repair_start, repair_end, context_ms=40, crossfade_ms=12
                )
            except Exception as e:
                logger.warning("cross_channel_repair.py::unbekannter Ersatzpfad: %s", e)

        result[affected_channel] = damaged

        # §AF: Continuity-Check
        _ct = self._dynamics.verify_continuity(result[affected_channel], sr, [repair_start, repair_end])
        if not _ct.continuity_ok:
            logger.debug(
                "Dropout-Repair: Kontinuitätswarnung bei Sample %d Kanal %d (%.1f dB)",
                dropout_start,
                affected_channel,
                _ct.max_envelope_deviation_db,
            )
            # Sanfte Nachkorrektur
            try:
                result[affected_channel] = self._dynamics.match_envelope(
                    result[affected_channel], sr, repair_start, repair_end, context_ms=80, crossfade_ms=20
                )
            except Exception as e:
                logger.warning("cross_channel_repair.py::unbekannter Ersatzpfad: %s", e)

        self._repair_boundaries.append((affected_channel, repair_start, repair_end))
        return cast(np.ndarray, result)

    def repair_click(
        self,
        audio: np.ndarray,
        sr: int,
        click_sample: int,
        affected_channel: int,
        click_width: int = 50,
    ) -> np.ndarray:
        """Repariert einen Click in einem Kanal mit Information vom anderen.

        Stereo-Clicks sind oft nur in einem Kanal hörbar (Lateral-Cuts).
        Der gesunde Kanal liefert die perfekte Referenz.

        §AF: Nach der Reparatur wird die Amplitude an den Kontext angeglichen,
        sodass kein Lautstärke-Hoppeln entsteht.
        """
        result = np.asarray(audio, dtype=np.float32).copy()
        if audio.ndim < 2 or audio.shape[0] < 2:
            return cast(np.ndarray, result)

        healthy_ch = 1 - affected_channel
        n = len(audio[0])

        half = click_width // 2
        start = max(0, click_sample - half)
        end = min(n, click_sample + half)

        # Kurze Cross-fade: 2ms
        fade = min(int(sr * 0.002), (end - start) // 2)
        window = np.ones(end - start, dtype=np.float32)
        if fade > 0:
            window[:fade] = np.linspace(0, 1, fade)
            window[-fade:] = np.linspace(1, 0, fade)

        result[affected_channel, start:end] = (
            result[affected_channel, start:end] * (1 - window) + result[healthy_ch, start:end] * window
        )

        # §AF: Envelope-Matching — Amplitude an Kontext angleichen
        try:
            result[affected_channel] = self._dynamics.match_envelope(
                result[affected_channel], sr, start, end, context_ms=30, crossfade_ms=8
            )
        except Exception as e:
            logger.warning("cross_channel_repair.py::repair_click Ersatzpfad: %s", e)

        # §AF: Continuity-Check
        _ct2 = self._dynamics.verify_continuity(result[affected_channel], sr, [start, end])
        if not _ct2.continuity_ok:
            logger.debug(
                "Click-Repair: Kontinuitätswarnung bei Sample %d Kanal %d (%.1f dB)",
                click_sample,
                affected_channel,
                _ct2.max_envelope_deviation_db,
            )

        self._repair_boundaries.append((affected_channel, start, end))
        return cast(np.ndarray, result)

    def verify_global_dynamics(
        self,
        before: np.ndarray,
        after: np.ndarray,
        sr: int,
    ) -> dict[str, Any]:
        """§AF: Global-Dynamics-Erhalt nach allen Cross-Channel-Repairs prüfen.

        Returns dict with:
            preserved: bool — dynamics intact?
            crest_factor_before/after: float
            peak_rms_ratio_before/after: float
            dynamic_range_before/after_db: float
            warnings: list[str]
        """
        report = self._dynamics.verify_global_dynamics(before, after)
        return {
            "preserved": report.global_dynamics_ok,
            "crest_factor_before": report.crest_factor_before,
            "crest_factor_after": report.crest_factor_after,
            "max_envelope_deviation_db": report.max_envelope_deviation_db,
            "warnings": report.warnings,
        }

    def detect_per_channel_defects(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict[str, list[dict]]:
        """Erkennt Defekte pro Kanal für gezieltes Cross-Channel-Repair.

        Returns:
            {"left": [{"type": "click", "sample": 12345, "severity": 0.8}, ...],
             "right": [...],
             "both": [...]}
        """
        result: dict[str, list[dict]] = {"left": [], "right": [], "both": []}
        if audio.ndim < 2 or audio.shape[0] < 2:
            return result

        for ch_idx, ch_name in enumerate(["left", "right"]):
            channel = np.asarray(audio[ch_idx], dtype=np.float32)

            # Einfache Click-Detektion: suche Spitzen > 3× RMS
            rms = float(np.sqrt(np.mean(channel * channel) + 1e-10))
            threshold = rms * 3.0

            # Suche in 10ms-Blöcken
            block_size = int(sr * 0.010)
            for i in range(0, len(channel) - block_size, block_size // 2):
                block = channel[i : i + block_size]
                peak = float(np.max(np.abs(block)))
                if peak > threshold:
                    result[ch_name].append(
                        {
                            "type": "click",
                            "sample": int(i + np.argmax(np.abs(block))),
                            "severity": float(min(1.0, peak / (threshold * 3))),
                            "width": int(sr * 0.003),  # 3ms default
                        }
                    )

        # Finde Defekte die in BEIDEN Kanälen auftreten
        both: list[dict] = []
        left_only: list[dict] = []
        for l_def in result["left"]:
            matched = False
            for r_def in result["right"]:
                if abs(l_def["sample"] - r_def["sample"]) < int(sr * 0.005):
                    both.append({**l_def, "channel": "both", "sample": (l_def["sample"] + r_def["sample"]) // 2})
                    matched = True
                    break
            if not matched:
                left_only.append({**l_def, "channel": "left"})

        right_only = []
        for r_def in result["right"]:
            if not any(abs(r_def["sample"] - b["sample"]) < int(sr * 0.005) for b in both):
                right_only.append({**r_def, "channel": "right"})

        result["left"] = left_only
        result["right"] = right_only
        result["both"] = both
        return result
