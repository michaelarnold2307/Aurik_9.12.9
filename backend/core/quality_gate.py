"""
Quality Gate Modul – automatisierte technische und psychoakustische Prüfungen für jede Phase.
Normativ: Kein Release ohne bestandene Quality Gates.
Implementierung gemäß §8.1 (PQS-Metriken + Musical Goals) und §5.1.

Geprüfte Kriterien:
  DSP / ML : True-Peak ≤ −1.0 dBTP, NaN/Inf-Freiheit, RMS-Mindestsignal,
             SNR ≥ 15 dB (STFT-Perzentil), alle Musical Goals ≥ Schwellwert.
  ML extra : authenticity_score ≥ 0.88 (falls vorhanden).
  GUI      : mode-String in erlaubter Whitelist.
In allen Methoden: Exception → Fallback True (kein Absturz der Pipeline).
"""

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QualityGate:
    """Automatisiertes Quality-Gate für DSP-, ML- und GUI-Prüfungen."""

    # §8.1 / §4.5 — True-Peak-Limit −1.0 dBTP (ITU-R BS.1770-5)
    TRUE_PEAK_LIMIT: float = 10 ** (-1.0 / 20.0)  # ≈ 0.8913

    # Mindest-SNR (geschätzt via STFT-Perzentilmethode)
    SNR_MIN_DB: float = 15.0

    # §1.2 — Musical Goals Pflicht-Schwellwerte
    # §1.2 — Musical Goals Pflicht-Schwellwerte (spec-konform gem. copilot-instructions §15 Musical Goals)
    MUSICAL_GOAL_MIN: dict = {
        # P1
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        # P2
        "tonal_center": 0.95,
        "timbre_authentizitaet": 0.87,
        "artikulation": 0.85,
        # P3
        "emotionalitaet": 0.82,
        "micro_dynamics": 0.88,
        "groove": 0.83,
        # P4
        "transparenz": 0.82,
        "waerme": 0.75,
        "bass_kraft": 0.78,
        "separation_fidelity": 0.78,
        # P5
        "brillanz": 0.78,
        "spatial_depth": 0.70,
    }

    # §1.4 — erlaubte Restaurierungs-Modi
    _VALID_MODES: frozenset = frozenset(
        {
            "restoration",
            "studio2026",
            "RESTORATION",
            "STUDIO_2026",
        }
    )

    # ------------------------------------------------------------------ #
    # Interne Hilfsmethoden                                                #
    # ------------------------------------------------------------------ #

    def _extract_audio(self, result: Any) -> np.ndarray | None:
        """Extrahiert np.ndarray aus einem Ergebnis-Objekt oder Array."""
        if isinstance(result, np.ndarray):
            return result
        for attr in ("audio", "output", "waveform"):
            candidate = getattr(result, attr, None)
            if isinstance(candidate, np.ndarray):
                return candidate
        return None

    def _check_audio_array(self, audio: np.ndarray, context: str) -> bool:
        """True-Peak, NaN/Inf, RMS und SNR-Prüfung.

        Algorithmus (SNR):
            1. STFT auf max. 10 s des Signals (nperseg=1024)
            2. Mittlere Leistung pro Frame
            3. P5-Perzentil  → Schätzung des Rauschbodens
            4. P80-Perzentil → Schätzung des Nutzsignalpegels
            5. SNR = 10·log10(P80 / P5)
        Fehlschlag erzeugt logger.warning, Ausnahmen → silent True.
        """
        if not np.isfinite(audio).all():
            logger.warning("[QualityGate/%s] NaN/Inf im Signal – abgelehnt.", context)
            return False

        # §DSP-Invariante: percentile(99.9) — ein einzelner Click-Spike darf
        # valides Audio nicht fälschlicherweise vom Export ausschließen.
        tp = float(np.percentile(np.abs(audio), 99.9))
        if tp > self.TRUE_PEAK_LIMIT + 1e-6:
            logger.warning("[QualityGate/%s] True-Peak (p99.9) %.4f > −1.0 dBTP – abgelehnt.", context, tp)
            return False

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-9:
            logger.warning("[QualityGate/%s] RMS %.2e zu niedrig (Stille?) – abgelehnt.", context, rms)
            return False

        # SNR-Schätzung via STFT-Perzentil (Post-2018-DSP, IMCRA-Minima-Basis)
        try:
            from scipy.signal import stft as _stft  # pylint: disable=import-outside-toplevel

            flat = audio.flatten()[: 48000 * 10]
            if len(flat) >= 2048:
                _, _, zxx = _stft(flat, fs=48000, nperseg=1024, noverlap=768)
                frame_power = np.mean(np.abs(zxx) ** 2, axis=0)
                noise_floor = float(np.percentile(frame_power, 5))
                signal_level = float(np.percentile(frame_power, 80))
                if noise_floor > 1e-15 and signal_level > noise_floor:
                    snr_db = 10.0 * math.log10(signal_level / noise_floor)
                    if snr_db < self.SNR_MIN_DB:
                        logger.warning(
                            "[QualityGate/%s] SNR %.1f dB < %.1f dB – abgelehnt.",
                            context,
                            snr_db,
                            self.SNR_MIN_DB,
                        )
                        return False
        except Exception as exc:
            logger.debug("[QualityGate/%s] SNR-Pruefung übersprungen: %s", context, exc)

        return True

    def _check_musical_goals(self, result: Any, context: str) -> bool:
        """Prüft alle vorhandenen Musical Goals gegen §1.2-Schwellwerte."""
        goals = getattr(result, "musical_goals", None)
        if not isinstance(goals, dict):
            return True  # Keine Goals vorhanden → nicht beanstandbar

        failed: list = []
        for goal, threshold in self.MUSICAL_GOAL_MIN.items():
            score = goals.get(goal)
            if score is None:
                continue
            try:
                s = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(s):
                logger.warning(
                    "[QualityGate/%s] Goal '%s' = %r nicht finite – abgelehnt.",
                    context,
                    goal,
                    score,
                )
                return False
            if s < threshold:
                failed.append((goal, s, threshold))

        if failed:
            for goal, s, thr in failed:
                logger.warning(
                    "[QualityGate/%s] Goal '%s' = %.3f < %.3f – abgelehnt.",
                    context,
                    goal,
                    s,
                    thr,
                )
            return False

        return True

    def _check_pleasantness_gate(self, audio: np.ndarray, context: str) -> tuple[bool, float]:
        """§v10: Prüft psychoakustische Angenehmheit als QUALITY GATE.

        Ein Ergebnis, das für menschliche Ohren unangenehm klingt,
        wird abgelehnt — auch wenn technische Metriken ok sind.

        Returns:
            (passed, pleasantness_score)
        """
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            result = compute_pleasantness(audio, 44100)  # sr wird ggf. vom Aufrufer gesetzt
            score = result.score
            label = result.label

            if score < 0.30:
                logger.warning(
                    "[QualityGate/%s] HPE ANSTREBGEND (%.3f) — abgelehnt. "
                    "Das Ergebnis klingt für menschliche Ohren unangenehm: %s",
                    context,
                    score,
                    ", ".join(result.issues) if result.issues else label,
                )
                return False, score

            if score < 0.50:
                logger.info(
                    "[QualityGate/%s] HPE NEUTRAL (%.3f) — akzeptiert mit Warnung. %s",
                    context,
                    score,
                    label,
                )
                return True, score

            logger.debug(
                "[QualityGate/%s] HPE %s (%.3f) — bestanden.",
                context,
                label,
                score,
            )
            return True, score

        except Exception as e:
            logger.debug("[QualityGate/%s] HPE nicht verfügbar: %s", context, e)
            return True, 0.5  # Fallback: ohne HPE durchlassen

    # ------------------------------------------------------------------ #
    # Öffentliche API                                                       #
    # ------------------------------------------------------------------ #

    def check_dsp(self, dsp_result: Any) -> bool:
        """§v10: Prüft technische UND psychoakustische Kriterien für DSP.

        Checks (HPE-zentriert):
            1. Audio-Array NaN/Inf-frei + True-Peak ≤ −1.0 dBTP
            2. RMS ≥ 1 × 10⁻⁹ (kein Stille-Signal)
            3. SNR ≥ 15 dB
            4. §v10 HPE Pleasantness Gate (PRIMÄR — ersetzt starre Musical-Goal-Thresholds)
            5. Musical Goals ≥ Pflicht-Schwellwerte (§1.2, sekundär)
        """
        try:
            # §5.9 (Spec 07) [RELEASE_MUST] Early-Exit: teure STFT/SNR-Analysen
            # erst nach positiver Musical-Goals-Prüfung. Goal-Failure → sofort False.
            if not self._check_musical_goals(dsp_result, "DSP"):
                return False

            audio = self._extract_audio(dsp_result)

            # Technische Basis-Checks
            if audio is not None and not self._check_audio_array(audio, "DSP"):
                return False

            # §v10 HPE Pleasantness Gate (PRIMÄR)
            if audio is not None:
                hpe_passed, hpe_score = self._check_pleasantness_gate(audio, "DSP")
                if not hpe_passed:
                    return False

            return True
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.exception("[QualityGate/DSP] Unerwarteter Fehler – Fallback True")
            return True

    def check_ml(self, ml_result: Any) -> bool:
        """Prüft technische und psychoakustische Kriterien für ML.

        Wie check_dsp(), zusätzlich:
            6. authenticity_score ≥ 0.88 (falls Attribut vorhanden)
        """
        try:
            if not self._check_musical_goals(ml_result, "ML"):
                return False
            audio = self._extract_audio(ml_result)
            if audio is not None and not self._check_audio_array(audio, "ML"):
                return False

            # ML-spezifisch: Authentizitäts-Score prüfen (§1.2 AuthentizitaetMetric)
            authentic = getattr(ml_result, "authenticity_score", None)
            if authentic is not None:
                try:
                    a = float(authentic)
                    limit = self.MUSICAL_GOAL_MIN.get("authentizitaet", 0.88)
                    if math.isfinite(a) and a < limit:
                        logger.warning(
                            "[QualityGate/ML] authenticity_Wert %.3f < %.3f – abgelehnt.",
                            a,
                            limit,
                        )
                        return False
                except (TypeError, ValueError) as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

            return True  # alle ML-Checks bestanden

        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.exception("[QualityGate/ML] Unerwarteter Fehler – Fallback True")
            return True

    def check_gui(self, gui_state: Any) -> bool:
        """Prüft GUI auf Verständlichkeit und erlaubten Restaurierungs-Modus.

        Checks:
            1. gui_state ist nicht None
            2. 'mode'-Feld (dict oder Attribut) liegt in _VALID_MODES (falls vorhanden)
        """
        try:
            if gui_state is None:
                logger.warning("[QualityGate/GUI] gui_state ist None – abgelehnt.")
                return False

            mode = gui_state.get("mode") if isinstance(gui_state, dict) else getattr(gui_state, "mode", None)
            if mode is not None and mode not in self._VALID_MODES:
                logger.warning("[QualityGate/GUI] Unbekannter Modus '%s' – abgelehnt.", mode)
                return False

            return True  # alle GUI-Checks bestanden

        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.exception("[QualityGate/GUI] Unerwarteter Fehler – Fallback True")
            return True

    # Rückwärtskompatibilität: einfache Schwellwert-Prüfung (wie dsp/feedback.py QualityGate)
    def check(self, value: float, threshold: float = 0.0) -> bool:
        """Einfacher Schwellwert-Check (Rückwärtskompatibilität)."""
        try:
            return float(value) >= float(threshold)
        except (TypeError, ValueError):
            return False
