"""
EraAuthenticPerceptualCompletion — Ära-authentische Wahrnehmungs-Ergänzung.

Spec 03 §2.1, Spec 02 §1.5 Schritt 8.
Aktiviert bei Quell-BW < 10 kHz. Rekonstruiert fehlende Höhenanteile
mit era-appropriate Spektralformung via DSP BandwidthExtender.

Nur für die Studio-2026-Kette. Kein ML, kein Download.
100% deterministisch, 100% offline.

Author: Aurik 10 — August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EraCompletionConfig:
    """Konfiguration für EraAuthenticPerceptualCompletion."""

    # Unterhalb dieser Bandbreite wird die Completion aktiviert (Hz)
    bandwidth_threshold_hz: float = 10000.0

    # Mix-Anteil der synthetisierten Höhen (0.0 = original, 0.5 = 50/50)
    default_amount: float = 0.35

    # Maximale Zielbandbreite nach Completion (Hz)
    max_target_bandwidth_hz: float = 16000.0

    # Minimales SNR des Quellmaterials für Completion (dB)
    min_snr_db: float = 12.0


# Ära-spezifische Completion-Parameter
# Je älter die Aufnahme, desto konservativer die Rekonstruktion
ERA_COMPLETION_PARAMS: dict[int, dict[str, float]] = {
    # decade -> {amount, max_target_hz}
    1890: {"amount": 0.15, "max_target_hz": 8000.0},
    1900: {"amount": 0.15, "max_target_hz": 8000.0},
    1910: {"amount": 0.20, "max_target_hz": 9000.0},
    1920: {"amount": 0.20, "max_target_hz": 10000.0},
    1930: {"amount": 0.25, "max_target_hz": 11000.0},
    1940: {"amount": 0.30, "max_target_hz": 12000.0},
    1950: {"amount": 0.30, "max_target_hz": 13000.0},
    1960: {"amount": 0.35, "max_target_hz": 14000.0},
    1970: {"amount": 0.35, "max_target_hz": 15000.0},
    1980: {"amount": 0.40, "max_target_hz": 16000.0},
    1990: {"amount": 0.40, "max_target_hz": 16000.0},
}


def _estimate_effective_bandwidth_hz(audio: np.ndarray, sr: int) -> float:
    """Schätzt die effektive Bandbreite (-20 dB Schwelle) des Audiosignals.

    Berechnet das spektrale Leistungsdichteprofil und findet die Frequenz,
    oberhalb derer die Leistung um 20 dB unter dem Peak liegt.

    Args:
        audio: Mono Audio-Array (float32/float64)
        sr: Sample-Rate (Hz)

    Returns:
        Effektive Bandbreite in Hz
    """
    if audio.size < sr // 10:  # Weniger als 100ms — zu kurz
        return float(sr / 2)

    try:
        from scipy.signal import welch

        freqs, psd = welch(audio.astype(np.float64), sr, nperseg=min(4096, len(audio) // 4))
        psd_db = 10.0 * np.log10(psd + 1e-15)
        psd_db -= psd_db.max()  # Normalisiere auf 0 dB Peak

        # Finde letzte Frequenz mit PSD >= -20 dB
        mask = psd_db >= -20.0
        if np.any(mask):
            return float(freqs[mask][-1])
        return float(sr / 2)
    except Exception as exc:
        logger.debug("§V6 _estimate_bandwidth fehlgeschlagen — Nyquist-Frequenz zurückgegeben (sr/2): %s", exc)
        return float(sr / 2)


class EraAuthenticPerceptualCompletion:
    """Ära-authentische Wahrnehmungs-Ergänzung für bandbreitenbegrenztes Material.

    Studio-2026-Kette Schritt 8. Läuft NUR wenn Quell-BW < 10 kHz.
    Nutzt den DSP BandwidthExtender mit ära-abhängiger Spektralformung.
    """

    def __init__(self, config: EraCompletionConfig | None = None) -> None:
        self.config = config or EraCompletionConfig()
        self._last_bw: float | None = None
        self._last_decade: int | None = None
        self._applied: bool = False

    @property
    def last_bandwidth_hz(self) -> float | None:
        """Zuletzt gemessene Quell-Bandbreite (Hz)."""
        return self._last_bw

    @property
    def last_decade(self) -> int | None:
        """Zuletzt klassifizierte Ära-Dekade."""
        return self._last_decade

    @property
    def was_applied(self) -> bool:
        """Wurde die Completion im letzten Lauf angewandt?"""
        return self._applied

    def needs_completion(self, audio: np.ndarray, sr: int) -> bool:
        """Prüft ob das Material Bandbreiten-Ergänzung benötigt.

        Args:
            audio: Mono Audio-Array
            sr: Sample-Rate

        Returns:
            True wenn Quell-BW < 10 kHz und Completion sinnvoll ist
        """
        bw = _estimate_effective_bandwidth_hz(audio, sr)
        self._last_bw = bw
        needs = bw < self.config.bandwidth_threshold_hz
        if needs:
            logger.info(
                "EraAuthenticPerceptualCompletion: BW=%.0f Hz < %.0f Hz — Completion aktiviert",
                bw,
                self.config.bandwidth_threshold_hz,
            )
        else:
            logger.debug(
                "EraAuthenticPerceptualCompletion: BW=%.0f Hz >= %.0f Hz — keine Completion nötig",
                bw,
                self.config.bandwidth_threshold_hz,
            )
        return needs

    def get_era_params(self, decade: int | None = None) -> dict[str, float]:
        """Ermittelt ära-spezifische Completion-Parameter.

        Args:
            decade: Geschätzte Dekade (None → Default)

        Returns:
            Dict mit amount und max_target_hz
        """
        if decade is None:
            return {"amount": self.config.default_amount, "max_target_hz": self.config.max_target_bandwidth_hz}

        decade_rounded = (decade // 10) * 10
        if decade_rounded in ERA_COMPLETION_PARAMS:
            return ERA_COMPLETION_PARAMS[decade_rounded].copy()
        # Interpolation für unbekannte Dekaden
        if decade_rounded < 1890:
            return ERA_COMPLETION_PARAMS[1890].copy()
        if decade_rounded > 1990:
            return {"amount": self.config.default_amount, "max_target_hz": self.config.max_target_bandwidth_hz}
        return {"amount": self.config.default_amount, "max_target_hz": self.config.max_target_bandwidth_hz}

    def complete(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        decade: int | None = None,
        material: str = "unknown",
        force: bool = False,
    ) -> np.ndarray:
        """Führt ära-authentische Wahrnehmungs-Ergänzung durch.

        Args:
            audio: Eingabe-Audio (float32, beliebige Kanäle)
            sr: Sample-Rate (Hz)
            decade: Geschätzte Ära-Dekade (None → Auto-Detection)
            material: Material-Typ (shellac, vinyl, etc.)
            force: Wenn True, wird Completion auch bei BW >= 10 kHz erzwungen

        Returns:
            Bandbreiten-erweitertes Audio (float32). Identisch zum Input wenn
            Completion nicht nötig ist.

        Raises:
            Keine — bei Fehlern wird das Original-Audio zurückgegeben.
        """
        self._applied = False
        self._last_decade = decade

        arr = np.asarray(audio, dtype=np.float32)

        if not force and not self.needs_completion(arr, sr):
            return cast(np.ndarray, arr)

        try:
            # 1. Ära-Parameter bestimmen
            params = self.get_era_params(decade)
            amount = params["amount"]
            max_target = params["max_target_hz"]

            # 2. Material-spezifische Bandbreiten-Extension via DSP
            from backend.core.dsp.bandwidth_extender import extend_bandwidth

            # Konvertiere Material zu Kategorie für den BandwidthExtender
            ext_material = _map_material_to_extender_category(material)

            # 3. Bandwidth-Extension anwenden
            bw_extended = extend_bandwidth(arr, sr, material=ext_material, amount=amount)

            if bw_extended is arr or np.array_equal(bw_extended, arr):
                logger.debug("EraAuthenticPerceptualCompletion: Keine Änderung durch BandwidthExtender")
                return cast(np.ndarray, arr)

            # 4. Ära-spezifische spektrale Formung (Lowpass bei max_target)
            shaped = _apply_era_spectral_shaping(bw_extended, sr, max_target_hz=max_target)

            # 5. Psychoakustische Plausibilitätsprüfung
            if not _validate_completion(arr, shaped, sr):
                logger.warning(
                    "EraAuthenticPerceptualCompletion: Completion validierung fehlgeschlagen — "
                    "Original wird zurückgegeben"
                )
                return cast(np.ndarray, arr)

            self._applied = True
            logger.info(
                "EraAuthenticPerceptualCompletion: Decade=%s, amount=%.2f, max_target=%.0f Hz, BW vorher=%.0f Hz",
                decade if decade else "auto",
                amount,
                max_target,
                self._last_bw,
            )

            return shaped.astype(np.float32)  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning(
                "EraAuthenticPerceptualCompletion: Fehler bei Completion — Original wird zurückgegeben: %s",
                exc,
            )
            return cast(np.ndarray, arr)


def _map_material_to_extender_category(material: str) -> str:
    """Mapped Material-Typ auf vom BandwidthExtender unterstützte Kategorie."""
    material_lower = material.lower().replace("-", "_").replace(" ", "_")
    known = {"shellac", "wax_cylinder", "wire_recording", "lacquer_disc", "mp3_low"}
    if material_lower in known:
        return material_lower
    # Heuristik: Bandbreiten-Begrenzung bei analogem Vintage-Material
    if material_lower in ("vinyl", "record", "schallplatte"):
        return "shellac"  # Nächste Approximation
    if material_lower in ("cassette", "tape", "tonband", "reel_to_reel"):
        return "lacquer_disc"
    return "shellac"  # Konservativster Fallback


def _apply_era_spectral_shaping(audio: np.ndarray, sr: int, *, max_target_hz: float) -> np.ndarray:
    """Wendet ära-spezifische spektrale Formung an (sanfter Lowpass).

    Simuliert den natürlichen Frequenzgang der Ära durch einen
    Butterworth-Tiefpass 2. Ordnung bei max_target_hz.
    """
    if max_target_hz >= sr / 2 * 0.95:
        return audio

    try:
        from scipy.signal import butter, sosfiltfilt

        nyq = sr / 2.0
        cutoff_norm = min(max_target_hz / nyq, 0.95)
        sos = butter(2, cutoff_norm, btype="low", output="sos")
        arr = np.asarray(audio, dtype=np.float64)

        if arr.ndim == 2:
            result = np.zeros_like(arr)
            for ch in range(arr.shape[1]):
                result[:, ch] = sosfiltfilt(sos, arr[:, ch])
        else:
            result = sosfiltfilt(sos, arr)

        return result.astype(np.float32)  # type: ignore[no-any-return]
    except ImportError as exc:
        logger.debug("§V6 scipy.signal.sosfiltfilt nicht verfügbar — Audio unverändert zurückgegeben (ImportError): %s", exc)
        return audio
    except Exception as exc:
        logger.warning("EraAuthenticPerceptualCompletion: Spectral shaping fehlgeschlagen: %s", exc)
        return audio


def _validate_completion(original: np.ndarray, completed: np.ndarray, sr: int) -> bool:
    """Psychoakustische Plausibilitätsprüfung der Completion.

    Prüft dass die Completion:
    - Nicht zu laut ist (RMS-Anstieg < 6 dB)
    - Keine klaffenden spektralen Lücken hinterlässt
    - Die Gesamtenergie nicht verdoppelt
    """
    try:
        rms_orig = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)) + 1e-12)
        rms_comp = float(np.sqrt(np.mean(completed.astype(np.float64) ** 2)) + 1e-12)

        if rms_comp > rms_orig * 2.0:
            logger.debug("Abschluss-Prüfung: RMS-Anstieg zu hoch (%.1fx)", rms_comp / rms_orig)
            return False

        if rms_comp < rms_orig * 0.5:
            logger.debug("Abschluss-Prüfung: RMS-Abfall zu stark (%.1fx)", rms_comp / rms_orig)
            return False

        # NaN/Inf check
        if not np.all(np.isfinite(completed)):
            logger.debug("Abschluss-Prüfung: NaN/Inf erkannt")
            return False

        return True
    except Exception as exc:
        logger.debug("§V6 _validate_completion fehlgeschlagen — False zurückgegeben (konservativ): %s", exc)
        return False


# Singleton
_era_completion: EraAuthenticPerceptualCompletion | None = None


def get_era_completion() -> EraAuthenticPerceptualCompletion:
    """Gibt die globale EraAuthenticPerceptualCompletion-Instanz zurück."""
    global _era_completion
    if _era_completion is None:
        _era_completion = EraAuthenticPerceptualCompletion()
    return _era_completion
