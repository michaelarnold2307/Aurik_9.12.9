"""
§v10 Reflective Listening Pass (RLP) — Der „nochmal hinhören"-Durchlauf.

Ein menschlicher Toningenieur hört sich das Ergebnis seiner Arbeit an,
erkennt verbleibende Schwächen und bessert gezielt nach — NUR an den
Stellen, die es brauchen, und NUR mit der minimal nötigen Intensität.

Der RLP automatisiert genau das:
  1. Analysiere V1 auf verbleibende Probleme (Spectral Tilt, Sibilanz,
     Bass-Druck, Stereo-Breite, Dynamik-Verlust, Rausch-Modulation)
  2. Priorisiere NUR die Top-2-Probleme („nicht alles auf einmal")
  3. Wende GEZIELTE Mikro-Korrekturen an (EQ ±1dB, sanfte Kompression,
     Stereo-Weite ±5%, Tiefpass-Glättung >14kHz)
  4. Vergleiche V1 vs V2 objektiv (RMS, Peak, Spectral Correlation)
  5. V2 besser → ersetze. V2 schlechter → verwerfe.
  6. Maximal 2 Iterationen („nicht totpolieren")

Der RLP modifiziert NIE den künstlerischen Charakter. Er korrigiert nur
technische Schwächen, die in der ersten Pipeline unentdeckt blieben.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RLPIssue:
    """Ein vom RLP erkanntes Problem."""

    category: str  # spectral_tilt, sibilance, bass_loss, stereo_collapse, dynamic_loss, noise_modulation
    severity: float  # 0.0 = nicht vorhanden, 1.0 = kritisch
    detail: str  # Mensch-lesbare Beschreibung
    correction: dict[str, float]  # DSP-Parameter für die Korrektur


@dataclass
class RLPCorrection:
    """Eine vom RLP angewandte Mikro-Korrektur."""

    issue: RLPIssue
    applied: bool = False
    delta_rms_db: float = 0.0
    delta_peak_db: float = 0.0
    delta_spectral_corr: float = 0.0
    accepted: bool = False  # True wenn V2 objektiv besser als V1


@dataclass
class RLPResult:
    """Ergebnis des Reflective Listening Pass."""

    audio: np.ndarray
    issues_found: list[RLPIssue] = field(default_factory=list)
    corrections: list[RLPCorrection] = field(default_factory=list)
    iterations: int = 0
    overall_improved: bool = False
    summary: str = ""


# ═══════════════════════════════════════════════════════════════════════════


class ReflectiveListeningPass:
    """§v10 Zweite-Hörrunde — analysiert V1 und bessert gezielt nach."""

    # Konfiguration
    MAX_ITERATIONS: int = 2
    MAX_ISSUES_PER_ITERATION: int = 2
    MIN_SEVERITY: float = 0.15  # Probleme unter diesem Schwellwert werden ignoriert
    IMPROVEMENT_THRESHOLD: float = 0.001  # Mindest-Verbesserung, um V2 zu akzeptieren

    # Korrektur-Limits (absichtlich KLEIN — Mikro-Korrekturen!)
    MAX_EQ_DB: float = 1.5  # Max ±1.5 dB EQ
    MAX_COMP_RATIO: float = 1.3  # Max 1.3:1 Ratio
    MAX_STEREO_ADJUST: float = 0.05  # Max ±5% Stereo-Breite
    NR_GENTLE_STRENGTH: float = 0.15  # Sanfte NR für RLP

    def __init__(self) -> None:
        pass

    # ── Öffentliche API ──────────────────────────────────────────────────

    def listen_and_refine(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        reference_audio: np.ndarray | None = None,
        artistic_intent: Any | None = None,
        material: str = "unknown",
        max_iterations: int | None = None,
    ) -> RLPResult:
        """Führt den Reflective Listening Pass aus.

        Args:
            audio:            V1 — Output der Haupt-Pipeline
            sr:               Sample-Rate
            reference_audio:  Original (vor Pipeline) für Vergleich
            artistic_intent:  Von get_artistic_intent()
            material:         Material-Typ
            max_iterations:   Override für MAX_ITERATIONS

        Returns:
            RLPResult mit V2-Audio (oder V1 falls keine Verbesserung)
        """
        max_iter = max_iterations or self.MAX_ITERATIONS
        current = np.asarray(audio, dtype=np.float64).copy()
        issues_found: list[RLPIssue] = []
        corrections: list[RLPCorrection] = []
        overall_improved = False

        for iteration in range(max_iter):
            # Schritt 1: Analysiere V1 auf verbleibende Probleme
            issues = self._diagnose(current, sr, reference_audio, artistic_intent)
            issues_found.extend(issues)

            if not issues:
                logger.info("RLP Iteration %d: Keine Probleme gefunden — Abbruch.", iteration + 1)
                break

            # Schritt 2: Priorisiere Top-2
            top_issues = sorted(issues, key=lambda i: i.severity, reverse=True)[: self.MAX_ISSUES_PER_ITERATION]

            # Schritt 3: Wende Mikro-Korrekturen an
            corrected = self._apply_corrections(current, sr, top_issues, material)

            # Schritt 4: Vergleiche V1 vs V2
            better, score_delta = self._is_improvement(current, corrected, sr)
            correction = RLPCorrection(
                issue=top_issues[0] if top_issues else RLPIssue("none", 0, "", {}),
                applied=True,
                delta_rms_db=score_delta.get("rms_delta", 0.0),
                delta_peak_db=score_delta.get("peak_delta", 0.0),
                delta_spectral_corr=score_delta.get("spectral_corr", 1.0),
                accepted=better,
            )
            corrections.append(correction)

            # Schritt 5: V2 besser → übernehmen, sonst verwerfen
            if better:
                current = corrected
                overall_improved = True
                logger.info(
                    "RLP Iteration %d: Verbesserung (Δspec=%.4f, Δrms=%.2fdB) — übernommen.",
                    iteration + 1,
                    score_delta.get("spectral_corr", 0.0),
                    score_delta.get("rms_delta", 0.0),
                )
            else:
                logger.info(
                    "RLP Iteration %d: Keine objektive Verbesserung — Korrektur verworfen.",
                    iteration + 1,
                )
                break  # Wenn die erste Korrektur nicht hilft, brechen wir ab

        # Baue Summary
        summary_parts = []
        n_fixed = sum(1 for c in corrections if c.accepted)
        n_issues = len(issues_found)
        if n_issues == 0:
            summary_parts.append("Keine hörbaren Restprobleme gefunden.")
        elif n_fixed > 0:
            summary_parts.append(f"{n_fixed} von {n_issues} erkannten Restproblemen gezielt korrigiert.")
        else:
            summary_parts.append(
                f"{n_issues} Restprobleme erkannt — aber keine automatische Korrektur "
                "brachte objektive Verbesserung. Manuelle Prüfung empfohlen."
            )

        return RLPResult(
            audio=np.asarray(current, dtype=np.float32),
            issues_found=issues_found,
            corrections=corrections,
            iterations=min(iteration + 1, max_iter),
            overall_improved=overall_improved,
            summary=" ".join(summary_parts),
        )

    # ── Diagnose-Methoden ────────────────────────────────────────────────

    def _diagnose(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | None,
        intent: Any | None,
    ) -> list[RLPIssue]:
        """Analysiert Audio auf verbleibende Probleme.

        Prüft 6 Dimensionen:
        1. Spectral Tilt — zu hell/dunkel?
        2. Sibilanz — zu scharf?
        3. Bass-Druck — zu wenig/zu viel?
        4. Stereo-Breite — zu schmal/zu breit?
        5. Dynamik — zu komprimiert?
        6. Rausch-Modulation — Gate-Pumpen?

        Returns nur Probleme mit severity >= MIN_SEVERITY.
        """
        issues: list[RLPIssue] = []

        # 1. Spectral Tilt
        tilt = self._measure_spectral_tilt(audio, sr)
        tilt_target = 0.0  # Flaches Spektrum ideal
        if intent and hasattr(intent, "brilliance_target"):
            # Je brillianter das Ziel, desto weniger negativer Tilt toleriert
            tilt_target = (intent.brilliance_target - 0.5) * 3.0

        tilt_error = abs(tilt - tilt_target)
        if tilt_error > 1.5:  # >1.5 dB/Oktave Abweichung
            direction = "hell" if tilt > tilt_target else "dunkel"
            eq_correction = -0.8 if tilt > tilt_target else 0.8
            issues.append(
                RLPIssue(
                    category="spectral_tilt",
                    severity=min(1.0, tilt_error / 5.0),
                    detail=f"Spektrum zu {direction} ({tilt:+.1f} dB/Oktave, Ziel: {tilt_target:+.1f})",
                    correction={"eq_high_shelf_db": eq_correction, "eq_freq_hz": 8000.0},
                )
            )

        # 2. Sibilanz (5-10 kHz Energie relativ zu 1-4 kHz)
        sib_ratio = self._measure_sibilance_ratio(audio, sr)
        sib_target = 0.15  # ~18 dB Unterschied (natürlicher Mix)
        sib_error = sib_ratio - sib_target
        if sib_error > 0.10:  # Zu viel Sibilanz
            severity = min(1.0, sib_error * 5.0)
            issues.append(
                RLPIssue(
                    category="sibilance",
                    severity=severity,
                    detail=f"Sibilanz zu präsent (Ratio={sib_ratio:.3f}, Ziel={sib_target:.3f})",
                    correction={"de_ess_strength": severity * 0.5, "de_ess_freq_hz": 7000.0},
                )
            )

        # 3. Bass-Druck (Energie unter 150 Hz relativ zu 200-2000 Hz)
        bass_ratio = self._measure_bass_presence(audio, sr)
        bass_target = 0.12  # ~18 dB unter den Mitten
        bass_error = bass_target - bass_ratio  # Positiv = zu wenig Bass
        if abs(bass_error) > 0.06:
            direction = "wenig" if bass_error > 0 else "viel"
            eq_correction = min(bass_error * 5.0, 1.5) if bass_error > 0 else max(bass_error * 5.0, -1.5)
            issues.append(
                RLPIssue(
                    category="bass_loss",
                    severity=min(1.0, abs(bass_error) * 8.0),
                    detail=f"Bass-Druck zu {direction} (Ratio={bass_ratio:.3f}, Ziel={bass_target:.3f})",
                    correction={"eq_low_shelf_db": eq_correction, "eq_freq_hz": 150.0},
                )
            )

        # 4. Stereo-Breite (Side/Mid Energie-Verhältnis)
        stereo_width = self._measure_stereo_width(audio)
        stereo_target = 0.25  # ~12 dB M/S-Unterschied
        stereo_error = abs(stereo_width - stereo_target)
        if stereo_error > 0.15:
            direction = "schmal" if stereo_width < stereo_target else "breit"
            issues.append(
                RLPIssue(
                    category="stereo_collapse",
                    severity=min(1.0, stereo_error * 3.0),
                    detail=f"Stereo-Bild zu {direction} (Width={stereo_width:.3f}, Ziel={stereo_target:.3f})",
                    correction={"stereo_width_adjust": 0.03 if stereo_width < stereo_target else -0.03},
                )
            )

        # 5. Dynamik-Verlust (LRA-Prüfung)
        if reference is not None:
            lra_v1 = self._estimate_lra(audio, sr)
            lra_ref = self._estimate_lra(reference, sr)
            if lra_v1 < lra_ref * 0.5 and lra_v1 < 3.0:  # Starke Kompression
                issues.append(
                    RLPIssue(
                        category="dynamic_loss",
                        severity=min(1.0, (lra_ref - lra_v1) / lra_ref),
                        detail=f"Dynamik komprimiert (LRA: {lra_v1:.1f} → Ziel: {lra_ref:.1f} LU)",
                        correction={"compression_ratio": 1.0},  # Keine weitere Kompression!
                    )
                )

        # 6. Rausch-Modulation (Varianz des Noise-Floors)
        nf_modulation = self._measure_noise_floor_modulation(audio, sr)
        if nf_modulation > 3.0:  # >3 dB Modulation
            issues.append(
                RLPIssue(
                    category="noise_modulation",
                    severity=min(1.0, nf_modulation / 10.0),
                    detail=f"Rauschboden moduliert ({nf_modulation:.1f} dB) — mögliches Gate-Pumpen",
                    correction={"nr_strength": 0.05, "nr_freq_hz": 14000.0},  # Sanftes HF-NR
                )
            )

        return issues

    @staticmethod
    def _filter_time_axis(sos, arr: np.ndarray) -> np.ndarray:
        """SOS-Filter entlang der ZEITACHSE — layout-sicher (§v10.x).

        Befund 2026-08-22: ``arr[:, ch]`` setzt Samples-first-Layout voraus.
        Bei Channels-first (2, N) ist ``arr[:, ch]`` eine Spalte der Länge 2
        → sosfiltfilt crasht mit „length must be greater than padlen“. Der
        frühe Kurzsignal-Guard (≤16 Samples) schützt dagegen nicht, weil er
        die Zeitachse korrekt erkennt — die Filterzweige taten es nicht.
        Zeitachse: (N, C) → Achse 0, (C, N) → Achse 1, mono → Achse 0.
        """
        from scipy import signal as scipy_signal

        if arr.ndim == 2:
            # Zeitachse: (N, C) samples-first → Achse 0; (C, N) channels-first → Achse 1.
            _axis = 0 if arr.shape[1] <= 2 else 1
        else:
            _axis = 0
        return cast(np.ndarray, (scipy_signal.sosfiltfilt(sos, arr, axis=_axis)))

    def _apply_corrections(self, audio: np.ndarray, sr: int, issues: list[RLPIssue], material: str) -> np.ndarray:
        """Wendet Mikro-Korrekturen an — kumulativ, aber mit strengen Limits."""
        arr = np.asarray(audio, dtype=np.float64).copy()
        # §III RLP-last: Die Korrektur-Filter (SOS, Ordnung 2–4) brauchen
        # padlen 9–15 Samples. Bei kürzeren Signalen ist die Identität die
        # einzige korrekte Antwort — kein Exception-Fallback (Befund
        # 2026-08-22: „length of input vector x must be greater than padlen, 9“
        # → kompletter RLP-Pass fiel aus, V1 beibehalten).
        _time_len = arr.shape[0] if (arr.ndim == 1 or arr.shape[-1] <= 2) else arr.shape[-1]
        if _time_len <= 16:
            logger.debug("RLP: Mikro-Korrekturen übersprungen (Signal zu kurz: %d Samples)", _time_len)
            return cast(np.ndarray, arr)

        for issue in issues:
            corr = issue.correction

            if "eq_high_shelf_db" in corr:
                from scipy import signal as scipy_signal

                gain_db = float(np.clip(corr["eq_high_shelf_db"], -self.MAX_EQ_DB, self.MAX_EQ_DB))
                freq_hz = float(corr.get("eq_freq_hz", 8000.0))
                if abs(gain_db) > 0.1:
                    # High-Shelf EQ (sanft)
                    sos = self._make_high_shelf(sr, freq_hz, gain_db)
                    arr = self._filter_time_axis(sos, arr)
                    logger.debug("RLP: High-Shelf %.1f dB @ %.0f Hz", gain_db, freq_hz)

            if "eq_low_shelf_db" in corr:
                from scipy import signal as scipy_signal

                gain_db = float(np.clip(corr["eq_low_shelf_db"], -self.MAX_EQ_DB, self.MAX_EQ_DB))
                freq_hz = float(corr.get("eq_freq_hz", 150.0))
                if abs(gain_db) > 0.1:
                    sos = self._make_low_shelf(sr, freq_hz, gain_db)
                    arr = self._filter_time_axis(sos, arr)
                    logger.debug("RLP: Low-Shelf %.1f dB @ %.0f Hz", gain_db, freq_hz)

            if "de_ess_strength" in corr:
                strength = float(np.clip(corr["de_ess_strength"], 0.0, 0.5))
                freq_hz = float(corr.get("de_ess_freq_hz", 7000.0))
                if strength > 0.05 and arr.ndim >= 1:
                    arr = self._gentle_de_ess(arr, sr, freq_hz, strength)
                    logger.debug("RLP: De-Essing strength=%.1f @ %.0f Hz", strength, freq_hz)

            if "stereo_width_adjust" in corr and arr.ndim == 2 and arr.shape[1] >= 2:
                adj = float(np.clip(corr["stereo_width_adjust"], -self.MAX_STEREO_ADJUST, self.MAX_STEREO_ADJUST))
                if abs(adj) > 0.001:
                    mid = (arr[:, 0] + arr[:, 1]) / 2
                    side = (arr[:, 0] - arr[:, 1]) / 2
                    # §v10.63 Multiplier 5.0→2.0: ±25% war zu aggressiv, erzeugte
                    # hörbare Stereo-Bild-Veränderungen statt subtiler Korrektur.
                    # 2.0× gibt ±10% Breitenänderung — hörbar aber unaufdringlich.
                    side *= 1.0 + adj * 2.0  # Skaliere Side-Kanal
                    arr[:, 0] = mid + side
                    arr[:, 1] = mid - side
                    logger.debug("RLP: Stereo-Breite %+.1f%%", adj * 100)

            if "nr_strength" in corr:
                strength = float(np.clip(corr["nr_strength"], 0.0, self.NR_GENTLE_STRENGTH))
                freq_hz = float(corr.get("nr_freq_hz", 14000.0))
                if strength > 0.01:
                    arr = self._gentle_hf_noise_reduction(arr, sr, freq_hz, strength)
                    logger.debug("RLP: HF-NR strength=%.2f @ %.0f Hz", strength, freq_hz)

        return cast(np.ndarray, (np.clip(arr, -1.0, 1.0)))

    def _is_improvement(self, v1: np.ndarray, v2: np.ndarray, sr: int) -> tuple[bool, dict[str, float]]:
        """Objektiver Vergleich V1 vs V2 mit psychoakustischer Angenehmheits-Prüfung.

        §v10 HPE: V2 wird nur akzeptiert wenn es für menschliche Ohren
        mindestens gleich angenehm klingt wie V1. Technische Metriken
        (spektrale Korrelation, RMS, Peak) sind NOTWENDIGE aber nicht
        HINREICHENDE Bedingungen.

        Entscheidungslogik:
        1. Technische Sicherheits-Prüfung (Clipping, Pegel-Explosion)
        2. Psychoakustische Angenehmheit: P(V2) >= P(V1) - 0.02
        3. Falls P(V2) < P(V1): V2 wird V2RWERFEN, selbst wenn
           technische Metriken ok sind
        """
        v1m = np.asarray(v1, dtype=np.float64)
        v2m = np.asarray(v2, dtype=np.float64)

        # Sicherstellen, dass gleiche Länge
        min_len = min(len(v1m), len(v2m))
        if v1m.ndim > 1:
            v1_mono = v1m[:min_len].mean(axis=-1) if v1m.shape[-1] <= 2 else v1m[:, :min_len].mean(axis=0)
            v2_mono = v2m[:min_len].mean(axis=-1) if v2m.shape[-1] <= 2 else v2m[:, :min_len].mean(axis=0)
        else:
            v1_mono = v1m[:min_len]
            v2_mono = v2m[:min_len]

        # RMS-Änderung
        rms1 = np.sqrt(np.mean(v1_mono**2)) + 1e-12
        rms2 = np.sqrt(np.mean(v2_mono**2)) + 1e-12
        rms_delta = 20.0 * np.log10(rms2 / rms1)

        # Peak-Änderung
        peak1 = float(np.max(np.abs(v1_mono)))
        peak2 = float(np.max(np.abs(v2_mono)))
        peak_delta = 20.0 * np.log10((peak2 + 1e-12) / (peak1 + 1e-12))

        # Spektrale Korrelation (Klangfarbe erhalten?)
        n_fft = min(2048, min_len // 4)
        if n_fft >= 64:
            spec1 = np.abs(np.fft.rfft(v1_mono[: n_fft * 4] * np.hanning(n_fft * 4)))[: n_fft // 2]
            spec2 = np.abs(np.fft.rfft(v2_mono[: n_fft * 4] * np.hanning(n_fft * 4)))[: n_fft // 2]
            spectral_corr = float(np.corrcoef(spec1, spec2)[0, 1]) if len(spec1) > 1 else 1.0
        else:
            spectral_corr = 1.0

        # ── §v10 HPE: Psychoakustische Angenehmheit ──
        try:
            from backend.core.human_pleasantness_estimator import (
                compare_pleasantness,
            )

            hpe_cmp = compare_pleasantness(v1m.astype(np.float32), v2m.astype(np.float32), sr)
            pleasantness_delta = float(hpe_cmp.get("delta_score", 0.0))
            pleasantness_improved = hpe_cmp.get("improved", False)
        except Exception:
            # Fallback: Wenn HPE nicht verfügbar, nur technische Prüfung
            logger.debug("HPE-Vergleich fehlgeschlagen — Fallback auf technische Prüfung", exc_info=True)
            pleasantness_delta = 0.0
            pleasantness_improved = True

        # Entscheidung: Technik ODER Angenehmheit
        # V2 wird akzeptiert wenn:
        # - Kein Clipping UND keine Pegel-Explosion (SICHERHEIT)
        # - UND entweder: spektral ähnlich ODER angenehmer fürs Ohr
        tech_safe = (
            peak2 < 1.0  # Kein Clipping
            and abs(rms_delta) < 2.0  # Keine Pegel-Explosion
        )
        sounds_acceptable = (
            spectral_corr >= 0.85  # Klangfarbe erhalten
            or pleasantness_improved  # Oder: klingt angenehmer
        )

        better = tech_safe and sounds_acceptable

        return better, {
            "rms_delta": float(rms_delta),
            "peak_delta": float(peak_delta),
            "spectral_corr": float(spectral_corr),
            "pleasantness_delta": float(pleasantness_delta),
            "pleasantness_improved": bool(pleasantness_improved),
        }

    # ── Mess-Methoden ────────────────────────────────────────────────────

    def _measure_spectral_tilt(self, audio: np.ndarray, sr: int) -> float:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
        else:
            mono = arr
        mono = np.atleast_1d(mono).ravel()  # ensure 1D
        n_fft = 4096
        if len(mono) < n_fft * 2:
            return 0.0
        spec = np.abs(np.fft.rfft(mono[: n_fft * 10] * np.hanning(n_fft * 10)))
        freqs = np.fft.rfftfreq(n_fft * 10, 1.0 / sr)
        mask = (freqs > 100) & (freqs < 10000)
        if mask.sum() > 10:
            spec_db = 20.0 * np.log10(np.maximum(spec[: len(freqs)], 1e-10))
            coeffs = np.polyfit(freqs[mask], spec_db[mask], 1)
            return float(coeffs[0] * 1000)  # dB/kHz
        return 0.0

    def _measure_sibilance_ratio(self, audio: np.ndarray, sr: int) -> float:
        from scipy import signal as scipy_signal

        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
        else:
            mono = arr
        mono = np.atleast_1d(mono).ravel()
        sos_sib = scipy_signal.butter(4, [5000, 10000], "bandpass", fs=sr, output="sos")
        sos_voice = scipy_signal.butter(4, [1000, 4000], "bandpass", fs=sr, output="sos")
        sib_energy = float(np.sum(scipy_signal.sosfilt(sos_sib, mono) ** 2))
        voice_energy = float(np.sum(scipy_signal.sosfilt(sos_voice, mono) ** 2))
        return sib_energy / (voice_energy + 1e-12)

    def _measure_bass_presence(self, audio: np.ndarray, sr: int) -> float:
        from scipy import signal as scipy_signal

        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
        else:
            mono = arr
        mono = np.atleast_1d(mono).ravel()
        sos_bass = scipy_signal.butter(4, [20, 150], "bandpass", fs=sr, output="sos")
        sos_mid = scipy_signal.butter(4, [200, 2000], "bandpass", fs=sr, output="sos")
        bass_energy = float(np.sum(scipy_signal.sosfilt(sos_bass, mono) ** 2))
        mid_energy = float(np.sum(scipy_signal.sosfilt(sos_mid, mono) ** 2))
        return bass_energy / (mid_energy + 1e-12)

    def _measure_stereo_width(self, audio: np.ndarray) -> float:
        if audio.ndim < 2 or audio.shape[-1] < 2:
            return 0.0
        left = audio[..., 0] if audio.shape[-1] <= 2 else audio[0, :]
        right = audio[..., 1] if audio.shape[-1] <= 2 else audio[1, :]
        side = left - right
        mid = left + right
        side_rms = float(np.sqrt(np.mean(side**2)) + 1e-12)
        mid_rms = float(np.sqrt(np.mean(mid**2)) + 1e-12)
        return side_rms / (mid_rms + 1e-12)

    def _estimate_lra(self, audio: np.ndarray, sr: int) -> float:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
        else:
            mono = arr
        mono = np.atleast_1d(mono).ravel()
        win = int(3.0 * sr)
        hop = int(1.0 * sr)
        st_vals = []
        for i in range(0, len(mono) - win, hop):
            chunk = mono[i : i + win]
            rms = np.sqrt(np.mean(chunk**2)) + 1e-12
            st_vals.append(20.0 * np.log10(rms))
        if len(st_vals) >= 4:
            st_vals = np.array(st_vals)  # type: ignore[assignment]
            gate = np.max(st_vals) - 20
            gated = st_vals[st_vals > gate]
            if len(gated) >= 4:
                return float(np.percentile(gated, 95) - np.percentile(gated, 10))
        return 10.0

    def _measure_noise_floor_modulation(self, audio: np.ndarray, sr: int) -> float:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
        else:
            mono = arr
        mono = np.atleast_1d(mono).ravel()
        win = int(0.1 * sr)
        rms_vals = []
        for i in range(0, len(mono) - win, win):
            chunk = mono[i : i + win]
            rms_vals.append(20.0 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-12))
        rms_vals = np.array(rms_vals)  # type: ignore[assignment]
        # Nur leise Abschnitte betrachten
        quiet_mask = rms_vals < (np.mean(rms_vals) - 6)
        if quiet_mask.sum() >= 4:
            np.percentile(rms_vals[quiet_mask], 10)
            # Varianz der leisen Abschnitte
            quiet_stds = []
            for i in range(0, len(mono) - int(2 * sr), int(2 * sr)):
                chunk_rms = []
                for j in range(0, int(2 * sr), win):
                    if i + j + win <= len(mono):
                        c = mono[i + j : i + j + win]
                        chunk_rms.append(20.0 * np.log10(np.sqrt(np.mean(c**2)) + 1e-12))
                if chunk_rms:
                    quiet_chunks = [v for v in chunk_rms if v < (np.mean(chunk_rms) - 3)]
                    if len(quiet_chunks) >= 3:
                        quiet_stds.append(np.std(quiet_chunks))
            if quiet_stds:
                return float(np.mean(quiet_stds))
        return 0.0

    # ── DSP-Hilfsmethoden ────────────────────────────────────────────────

    @staticmethod
    def _make_high_shelf(sr: int, freq: float, gain_db: float):
        """RBJ-High-Shelf (Audio-EQ-Cookbook-Formeln).

        scipy.butter kennt KEINE Bandtypen 'highshelf'/'lowshelf' — die alte
        Butter-Variante warf ValueError ("'lowshelf' is an invalid bandtype for
        filter") und ließ RLP-last regelmäßig fehlschlagen (Befund 2026-08-16).
        """
        from scipy import signal as scipy_signal

        w0 = 2.0 * np.pi * freq / sr
        A = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(w0) / (2.0 * 0.7)
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        return scipy_signal.tf2sos([b0, b1, b2], [a0, a1, a2])

    @staticmethod
    def _make_low_shelf(sr: int, freq: float, gain_db: float):
        """RBJ-Low-Shelf (Audio-EQ-Cookbook) — komplementär zum High-Shelf."""
        from scipy import signal as scipy_signal

        w0 = 2.0 * np.pi * freq / sr
        A = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(w0) / (2.0 * 0.7)
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        return scipy_signal.tf2sos([b0, b1, b2], [a0, a1, a2])

    @staticmethod
    def _gentle_de_ess(audio: np.ndarray, sr: int, freq: float, strength: float) -> np.ndarray:
        from scipy import signal as scipy_signal

        # Einfaches De-Essing: Low-Pass-Filter oberhalb der Zielfrequenz mit sanfter Stärke
        sos = scipy_signal.butter(2, freq, "lowpass", fs=sr, output="sos")
        # Layout-sicher über die ZEIT-Achse filtern: (N, 2) → axis 0,
        # kanal-orientiert (2, N) → axis -1 (Befund 2026-08-22: axis=0 auf
        # (2, N) filtert 2-Sample-Vektoren → padlen-Crash).
        _axis = -1 if (audio.ndim == 2 and audio.shape[-1] > 2) else 0
        filtered = scipy_signal.sosfiltfilt(sos, audio, axis=_axis)
        mix = 1.0 - strength * 0.8  # Max 40% Mix des gefilterten Signals
        return audio * mix + filtered * (1.0 - mix)  # type: ignore[no-any-return]

    @staticmethod
    def _gentle_hf_noise_reduction(audio: np.ndarray, sr: int, freq: float, strength: float) -> np.ndarray:
        from scipy import signal as scipy_signal

        # Sanfte Hochton-Rauschunterdrückung via Low-Pass + Mix
        sos = scipy_signal.butter(2, freq, "lowpass", fs=sr, output="sos")
        # Layout-sicher über die ZEIT-Achse filtern (wie _gentle_de_ess).
        _axis = -1 if (audio.ndim == 2 and audio.shape[-1] > 2) else 0
        filtered = scipy_signal.sosfiltfilt(sos, audio, axis=_axis)
        mix = 1.0 - strength  # Sanftes Blending
        return audio * mix + filtered * (1.0 - mix)  # type: ignore[no-any-return]


# Singleton
_instance: ReflectiveListeningPass | None = None


def get_reflective_listening_pass() -> ReflectiveListeningPass:
    global _instance
    if _instance is None:
        _instance = ReflectiveListeningPass()
    return _instance
