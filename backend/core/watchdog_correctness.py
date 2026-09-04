"""§v10.17 WatchdogCorrectness — prüft nicht nur Liveness, sondern auch Output-Qualität."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class WatchdogCorrectness:
    """Ergänzt den Liveness-Watchdog um Output-Validierung."""

    @staticmethod
    def check_audio_valid(audio: np.ndarray, label: str = "") -> tuple[bool, str]:
        """Prüft ob Audio nach einer Phase noch valide ist."""
        try:
            arr = np.asarray(audio)
            if arr.size == 0:
                return False, f"{label}: empty array"
            if np.any(np.isnan(arr)):
                return False, f"{label}: contains NaN"
            if np.any(np.isinf(arr)):
                return False, f"{label}: contains Inf"
            if np.max(np.abs(arr)) > 2.0:
                return False, f"{label}: amplitude {np.max(np.abs(arr)):.2f} > 2.0"
            # Check for all-zeros (phase produced silence)
            rms = float(np.sqrt(np.mean(arr**2)))
            if rms < 1e-10:
                return False, f"{label}: total silence (RMS={rms:.2e})"
            return True, "ok"
        except Exception as e:
            logger.debug("§V6 check_audio_valid fehlgeschlagen — (False, error) zurückgegeben für %s: %s", label, e)
            return False, f"{label}: validation error: {e}"

    @staticmethod
    def check_not_identical(original: np.ndarray, processed: np.ndarray, label: str = "") -> tuple[bool, str]:
        """Prüft dass eine Phase das Audio tatsächlich verändert hat (nicht Passthrough)."""
        if np.array_equal(np.asarray(original), np.asarray(processed)):
            return False, f"{label}: output identical to input — possible passthrough"
        return True, "ok"
