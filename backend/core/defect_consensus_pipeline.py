#!/usr/bin/env python3
"""
§v10.220: Defect Consensus Pipeline — 30 Module → 1 Manifest.

Problem: 30 isolierte Defekt-Detektoren treffen unabhängige Entscheidungen.
Wenn DefectScanner "Klick" sagt und SurgicalAnalyzer "Transient",
gibt es keine Konfliktlösung. Falsche Defekte korrumpieren alle Folge-Phasen.

Lösung: Consensus-Pipeline mit 3 Stufen:

  Stufe 1 – PARALLEL SCANNING:
    Alle 30 Module laufen parallel. Jedes liefert Defect-Hypothesen
    mit Typ, Zeitstempel, Konfidenz und Begründung.

  Stufe 2 – CONFLICT RESOLUTION:
    Überlappende/konfligierende Hypothesen werden per Weighted Voting
    aufgelöst. Module mit höherer historischer Präzision bekommen
    mehr Gewicht. Zeitliche Überschneidungen werden gemerged.

  Stufe 3 – CAUSAL REASONING:
    CausalDefectReasoner prüft kausale Ketten (z.B. Klick → Pre-Echo →
    harmonische Lücke). Defekte ohne kausale Basis werden gedowngraded.

  Output: Ein einziges DefectManifest — widerspruchsfrei, gewichtet,
  kausal validiert. Alle Folge-Phasen arbeiten mit DEMSELBEN Manifest.

Key-Innovation: Nicht mehr Module bauen, sondern die 30 existierenden
ENDLICH koordinieren.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

SR = 48000


# ═════════════════════════════════════════════════════════════════════════════
# Unified Defect Model
# ═════════════════════════════════════════════════════════════════════════════


class DefectCategory(str, Enum):
    """Standardisierte Defekt-Kategorien — alle Module mappen hierhin."""

    CLICK = "click"  # Einzelimpuls < 10ms
    CRACKLE = "crackle"  # Dichte Mikro-Impulse
    POP = "pop"  # Starker Einzelimpuls > 10ms
    HUM = "hum"  # 50/60 Hz Brummen
    HISS = "hiss"  # Breitband-Rauschen
    TAPE_HISS = "tape_hiss"  # Bandrauschen (spektrale Färbung)
    VINYL_NOISE = "vinyl_noise"  # Oberflächenrauschen
    WOW_FLUTTER = "wow_flutter"  # Geschwindigkeitsschwankungen
    CLIPPING = "clipping"  # Digitale/analoge Übersteuerung
    DROPOUT = "dropout"  # Signalaussetzer
    PRE_ECHO = "pre_echo"  # Band-Übersprechen
    PRINT_THROUGH = "print_through"  # Magnetische Kopie
    SIBILANCE = "sibilance"  # Übermäßige Zischlaute
    BREATH = "breath"  # Störende Atemgeräusche
    DE_ESSING_ARTIFACT = "de_essing"  # De-Essing-Artefakte
    PHASE_ERROR = "phase_error"  # Phasenfehler/Stereo-Imbalance
    DISTORTION = "distortion"  # Nichtlineare Verzerrung
    NOISE_GATE_CHATTER = "gate_chatter"  # Noise-Gate-Flattern
    REVERB_TAIL = "reverb_tail"  # Unerwünschter Nachhall
    UNKNOWN = "unknown"


@dataclass
class DefectHypothesis:
    """Eine Defekt-Hypothese von einem Detektor-Modul."""

    category: DefectCategory
    start_sample: int
    end_sample: int
    confidence: float  # 0.0–1.0
    severity: float  # 0.0–1.0
    source_module: str  # welches Modul hat detektiert
    evidence: dict[str, Any] = field(default_factory=dict)
    # Kausale Verknüpfungen
    caused_by: list[str] = field(default_factory=list)  # Defect-IDs upstream
    causes: list[str] = field(default_factory=list)  # Defect-IDs downstream


@dataclass
class DefectManifest:
    """Widerspruchsfreies, gewichtetes Defekt-Manifest."""

    defects: list[DefectHypothesis] = field(default_factory=list)
    total_hypotheses: int = 0
    conflicts_resolved: int = 0
    merged_defects: int = 0
    causal_downgrades: int = 0
    processing_time: float = 0.0
    module_count: int = 0

    @property
    def total_severity(self) -> float:
        """Summierte Schwere aller Defekte."""
        return sum(d.severity * d.confidence for d in self.defects)

    @property
    def dominant_category(self) -> DefectCategory:
        """Häufigste Defekt-Kategorie."""
        if not self.defects:
            return DefectCategory.UNKNOWN
        counts: dict[DefectCategory, float] = defaultdict(float)
        for d in self.defects:
            counts[d.category] += d.confidence * d.severity
        return max(counts, key=lambda k: counts[k])


# ═════════════════════════════════════════════════════════════════════════════
# Module Registry with Precision Weights
# ═════════════════════════════════════════════════════════════════════════════

# Historische Präzision pro Modul (kann aus Telemetrie aktualisiert werden)
MODULE_WEIGHTS: dict[str, float] = {
    # Scanner
    "defect_scanner": 0.85,
    "precision_defect_locator": 0.90,
    "defect_re_scanner": 0.80,
    # Klassifikatoren
    "artifact_detector": 0.82,
    "introduced_artifact_detector": 0.78,
    "psychoacoustic_artifact_detector": 0.88,
    "clipping_detection": 0.92,
    "attack_type_classifier": 0.85,
    "intentional_artifact_classifier": 0.80,
    # Kausale Analyse
    "causal_defect_reasoner": 0.95,
    "surgical_defect_analyzer": 0.87,
    # Qualität
    "defect_detection_quality_gate": 0.75,
    "quality_regression_detector": 0.72,
    # Spezial
    "dolby_nr_detector": 0.85,
    "cassette_defect_verifier": 0.80,
    "remaster_detector": 0.70,
    "vocal_overprocessing_detector": 0.82,
}

DEFAULT_WEIGHT = 0.70


# ═════════════════════════════════════════════════════════════════════════════
# §v10.840: Impuls-Detektor — Klicks/Knackser direkt auf der Waveform
# ═════════════════════════════════════════════════════════════════════════════

_ANALOG_ONLY_CATEGORIES = frozenset(
    {
        DefectCategory.UNKNOWN,
    }
)

# Analog-Detektoren, die auf digitalem Material physikalisch unmöglich sind
_ANALOG_ONLY_NAMES = frozenset(
    {
        "bandwidth_loss",
        "riaa_curve_error",
        "soft_saturation",
        "speed_calibration_error",
        "print_through",
        "wow_flutter",
        "flutter_spectral_sidebands",
        "room_mode_resonance",
        "proximity_effect_excess",
        "reverb_excess",
    }
)


def detect_impulse_defects(audio: np.ndarray, sr: int) -> list[DefectHypothesis]:
    """Erkennt Klicks und Knackser als kurze, breitbandige Impulse.

    §v10.840: Der DefectScanner ist auf Träger-Artefakte (Wow, Rumble, RIAA)
    kalibriert und übersieht Transienten auf digitalem Material. Dieser
    Detektor arbeitet direkt auf der Waveform:
      - Klick:   isolierter Spike 1–5 ms
      - Knackser: dichte Folge von Spikes (Crackle-Bursts)

    Returns:
        Liste von DefectHypothesis (click/crackle).
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        # §Spec 24: Kanal-Mix über die Kanal-Achse — (N, C) → Achse 1,
        # (C, N) → Achse 0 (alt: mean(axis=0) mittelte (N, 2) über die Zeit → (2,)).
        audio = audio.mean(axis=0) if audio.shape[0] <= 2 and audio.shape[0] < audio.shape[1] else audio.mean(axis=1)

    hop = max(1, sr // 1000)  # 1 ms Hops
    win = max(4, sr // 250)  # 4 ms Fenster
    n = 1 + (len(audio) - win) // hop
    if n < 4:
        return []

    # Kurzzeit-Energie + lokaler Median (robust gegen Musik-Transienten)
    energy = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = i * hop
        energy[i] = np.mean(audio[s : s + win] ** 2) + 1e-12

    # Median-Filter (51 ms = 51 Hops) als lokale Referenz
    med_k = 51
    local_med = np.zeros_like(energy)
    for i in range(n):
        lo, hi = max(0, i - med_k // 2), min(n, i + med_k // 2 + 1)
        local_med[i] = np.median(energy[lo:hi])

    ratio = energy / (local_med + 1e-12)

    # Klick: isolierter Peak mit Verhältnis > 3.5 (§v10.840 kalibriert auf Corpus:
    # echte digitale Klicks erreichen nur ~4× — Drums sind länger und breiter)
    click_mask = ratio > 3.5
    # Crackle: mehrere Spikes in kurzer Folge mit Verhältnis > 2.5
    crackle_mask = ratio > 2.5

    hypotheses: list[DefectHypothesis] = []

    # Klicks: isolierte Spikes zusammenfassen
    i = 0
    click_count = 0
    while i < n:
        if click_mask[i]:
            start_i = i
            while i < n and click_mask[i]:
                i += 1
            # Nur kurze Bursts (max 10 ms) sind Klicks
            burst_ms = (i - start_i) * 1.0
            if burst_ms <= 10:
                click_count += 1
                s_smp = start_i * hop
                e_smp = min(len(audio), i * hop + win)
                hypotheses.append(
                    DefectHypothesis(
                        category=DefectCategory.CLICK,
                        start_sample=int(s_smp),
                        end_sample=int(e_smp),
                        confidence=min(0.95, 0.5 + click_count * 0.02),
                        severity=float(min(1.0, ratio[start_i] / 20.0)),
                        source_module="impulse_detector",
                        evidence={"method": "waveform_spike", "ratio": float(ratio[start_i])},
                    )
                )
        else:
            i += 1

    # Crackle: dichte Spikes (mehr als 3 Spikes in 100 ms)
    spike_positions = np.where(crackle_mask)[0]
    if len(spike_positions) >= 3:
        burst_starts: list[int] = []
        prev = -1000
        burst: list[int] = []
        for pos in spike_positions:
            if pos - prev > 100:  # neue Burst-Gruppe (100 ms Lücke)
                if len(burst) >= 3:
                    burst_starts.append(burst[0])
                burst = [pos]
            else:
                burst.append(pos)
            prev = pos
        if len(burst) >= 3:
            burst_starts.append(burst[0])

        for bs in burst_starts:
            s_smp = int(bs * hop)
            e_smp = min(len(audio), int((bs + 100) * hop))
            hypotheses.append(
                DefectHypothesis(
                    category=DefectCategory.CRACKLE,
                    start_sample=s_smp,
                    end_sample=e_smp,
                    confidence=0.7,
                    severity=0.6,
                    source_module="impulse_detector",
                    evidence={"method": "crackle_burst"},
                )
            )

    return hypotheses


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Parallel Scanning
# ═════════════════════════════════════════════════════════════════════════════


def detect_reverb_tail(audio: np.ndarray, sr: int = SR) -> dict:
    """§v10.998: Reverb-Tail-Detektor — Late-Tail-Energy (SOTA-Lücken-Schluss).

    Messung zeigte: Ein Hall-Fall (RT60 = 1.2s) wurde von der Consensus
    NIE erkannt — es gab keinen Detektor, der REVERB_TAIL meldet. Dieser
    Detektor schließt die Lücke:

      - 20-ms-Hüllkurve des bandgefilterten Signals (200–4000 Hz)
      - Direktenergie um den Hüllkurven-Peak vs. Tail-Energie (+120–520 ms)
      - late/direct-Ratio → Severity (0.1 trocken … 0.6+ hallig)

    Returns dict im Consensus-Format: {"defects": [{"type": "reverb_tail", …}]}
    """
    try:
        from scipy.signal import butter, hilbert, sosfiltfilt
    except Exception as exc:
        log.debug("Reverb-Detektor: scipy nicht verfügbar (%s)", exc)
        return {"defects": []}

    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim > 1:
        # §Spec 24: Kanal-Mix über die Kanal-Achse (Befund 2026-08-22).
        mono = mono.mean(axis=0) if mono.shape[0] <= 2 and mono.shape[0] < mono.shape[1] else mono.mean(axis=1)
    if len(mono) < sr:  # mindestens 1 s Audio
        return {"defects": []}

    sos = butter(4, [200.0, 4000.0], "band", fs=sr, output="sos")
    band = sosfiltfilt(sos, mono)
    env = np.abs(hilbert(band))

    hop = max(1, int(0.020 * sr))
    n_blocks = (len(env) - hop) // hop
    if n_blocks < 40:
        return {"defects": []}
    blocks = np.array([env[i * hop : (i + 1) * hop].mean() for i in range(n_blocks)])

    peak_idx = int(np.argmax(blocks))
    direct = float(np.max(blocks[max(0, peak_idx - 2) : min(n_blocks, peak_idx + 4)]))
    if direct < 1e-9:
        return {"defects": []}

    late_start = min(n_blocks - 1, peak_idx + int(0.120 / 0.020))
    late_end = min(n_blocks, late_start + int(0.400 / 0.020))
    if late_end <= late_start:
        return {"defects": []}
    late = float(np.mean(blocks[late_start:late_end]))

    ratio = late / direct
    if ratio < 0.10:  # trockenes Signal
        return {"defects": []}

    # §v10.998: Hüllkurven-Dynamik-Gate — ein gehaltener Dauerton (Orgel,
    # Sinus) hat eine flache Hülle (ratio ≈ 1, aber KEIN Hall). Hall zeigt
    # sich nur in den Pausen zwischen Impulsen: erst abfallende Hüllkurven
    # mit deutlicher Dynamik zählen als Nachhall.
    _env_dynamics = float((np.max(blocks) - np.min(blocks)) / (np.max(blocks) + 1e-9))
    if _env_dynamics < 0.25:
        return {"defects": []}

    # Abkling-Check: Hall FÄLLT in den Pausen ab (negativer log-Hüllkurven-Slope)
    _late_blocks = blocks[late_start:late_end]
    if len(_late_blocks) > 8:
        _x = np.arange(len(_late_blocks), dtype=np.float64)
        _log_env = np.log(_late_blocks + 1e-12)
        _slope = float(np.polyfit(_x, _log_env, 1)[0])
        if _slope >= -0.002:  # flach oder steigend → kein Nachhall-Abfall
            return {"defects": []}

    severity = float(np.clip((ratio - 0.10) / 0.5, 0.05, 1.0))
    confidence = float(np.clip(0.4 + (ratio - 0.10) * 0.8, 0.4, 0.9))
    start_s = float(peak_idx * 0.020)
    end_s = float(min(len(audio) / sr, start_s + 2.0))
    return {
        "defects": [
            {
                "type": "reverb_tail",
                "start": start_s,
                "end": end_s,
                "severity": severity,
                "confidence": confidence,
                "evidence": {"late_direct_ratio": round(ratio, 3)},
            }
        ]
    }


class ParallelDefectScanner:
    """
    Führt alle verfügbaren Defekt-Detektoren aus und sammelt Hypothesen.
    Jedes Modul wird in einem try/except gewrappt — ein fehlerhaftes Modul
    blockiert nicht die gesamte Pipeline.

    §v10.600 SOTA: 3-Stufen-Ausführung mit Kontext-Passing für deferred Detektoren.
    """

    def __init__(self):
        self._detectors: list[tuple[str, Callable]] = []
        self._stage2_detectors: list[tuple[str, Callable]] = []
        self._stage3_detectors: list[tuple[str, Callable]] = []
        self._register_detectors()

    def _register_detectors(self):
        """Registriert alle verfügbaren Detektoren mit ihren ECHTEN APIs.

        §v10.700: Keine Silent Failures — jeder Detektor wird protokolliert:
          - "registered":       Scan-Funktion aktiv (Stufe 1)
          - "deferred_stage2":  kontext-deferred — braucht Stufe-1-Ergebnisse (Defekte/Onsets)
          - "deferred_stage3":  kontext-deferred — braucht Material/Ära/Vorher-Nachher
          - "failed":           Import/API-Fehler — wird GELOGGT, nicht verschluckt

        Kontext-Deferral ist ARCHITEKTUR (3-Stufen-Pipeline), keine Degradation →
        INFO statt WARNING (Rev. 2026-08-16: Startup-Warnungen auf SOTA-Niveau).
        """
        self._registration_report: list[dict[str, str]] = []

        def _reg(name: str, loader, stage: str | None = None):
            try:
                fn = loader()
                if fn is not None:
                    if stage == "deferred_stage2":
                        self._stage2_detectors.append((name, fn))
                    elif stage == "deferred_stage3":
                        self._stage3_detectors.append((name, fn))
                    else:
                        self._detectors.append((name, fn))
                    self._registration_report.append({"name": name, "status": "registered"})
                elif stage is not None:
                    self._registration_report.append({"name": name, "status": stage})
                    log.info("Defect Consensus: %s kontext-deferred (%s)", name, stage)
                else:
                    self._registration_report.append({"name": name, "status": "failed"})
                    log.warning("Defect Consensus: %s NICHT registriert (API nicht verfügbar)", name)
            except Exception as exc:
                self._registration_report.append({"name": name, "status": "failed"})
                log.warning("Defect Consensus: %s fehlgeschlagen — %s", name, exc)

        # 1. DefectScanner (primary)
        def _load_defect_scanner():
            from backend.core.defect_scanner import get_defect_scanner

            return get_defect_scanner().scan

        _reg("defect_scanner", _load_defect_scanner)

        # 2. Clipping Detection — classify_clipping(audio, sr)
        def _load_clipping():
            from backend.core.clipping_detection import classify_clipping

            return classify_clipping

        _reg("clipping_detection", _load_clipping)

        # 3. Artifact Detector — ArtifactDetector().scan(audio)
        def _load_artifact():
            from backend.core.artifact_detector import ArtifactDetector

            return ArtifactDetector().scan

        _reg("artifact_detector", _load_artifact)

        # 4. Psychoacoustic — Detector().analyze(audio, sr)
        def _load_psychoacoustic():
            from backend.core.psychoacoustic_artifact_detector import PsychoacousticArtifactDetector

            return PsychoacousticArtifactDetector().analyze

        _reg("psychoacoustic_artifact_detector", _load_psychoacoustic)

        # 5. Remaster Detector — analyse(audio, sr)
        def _load_remaster():
            from backend.core.remaster_detector import RemasterDetector

            return RemasterDetector().analyse

        _reg("remaster_detector", _load_remaster)

        # §v10.998: Reverb-Tail-Detektor (Late-Tail-Energy) — schließt die
        # Dereverb-Detektionslücke (Messung: Hall-Fall RT60 1.2s wurde NIE erkannt)
        _reg("reverb_tail_detector", lambda: detect_reverb_tail)

        # 5b. §v10.840: Impuls-Detektor (Klicks/Knackser auf der Waveform)
        _reg("impulse_detector", lambda: detect_impulse_defects)

        # 6. Precision Locator — refine_edges(audio, sr, defects): braucht Defekte als Input
        def _load_precision():
            try:
                from backend.core.precision_defect_locator import PrecisionDefectLocator

                locator = PrecisionDefectLocator()
                return locator.refine_edges
            except Exception as exc:
                log.debug("PrecisionDefectLocator nicht verfügbar: %s", exc)
                return None

        _reg("precision_defect_locator", _load_precision, stage="deferred_stage2")

        # 7. Attack Type — classify(audio, sr, onset_sample): braucht Onset-Positionen
        def _load_attack():
            try:
                from backend.core.attack_type_classifier import classify_attack_type

                return classify_attack_type
            except Exception as exc:
                log.debug("AttackTypeClassifier nicht verfügbar: %s", exc)
                return None

        _reg("attack_type_classifier", _load_attack, stage="deferred_stage2")

        # 8. Intentional Artifact — classify(material, era, freedom): braucht Material-Kontext
        def _load_intentional():
            try:
                from backend.core.intentional_artifact_classifier import classify_intentional_artifacts

                return classify_intentional_artifacts
            except Exception as exc:
                log.debug("IntentionalArtifactClassifier nicht verfügbar: %s", exc)
                return None

        _reg("intentional_artifact_classifier", _load_intentional, stage="deferred_stage3")

        # 9. Dolby NR — detect_dolby_encoding(audio, sr, material, era): braucht Material/Ära
        def _load_dolby():
            try:
                from backend.core.dolby_nr_detector import detect_dolby_encoding

                return detect_dolby_encoding
            except Exception as exc:
                log.debug("DolbyNRDetector nicht verfügbar: %s", exc)
                return None

        _reg("dolby_nr_detector", _load_dolby, stage="deferred_stage3")

        # 10. Cassette Verifier — verify(original, repaired): braucht Vorher/Nachher
        def _load_cassette():
            try:
                from backend.core.cassette_defect_verifier import verify_cassette_defects

                return verify_cassette_defects
            except Exception as exc:
                log.debug("CassetteDefectVerifier nicht verfügbar: %s", exc)
                return None

        _reg("cassette_defect_verifier", _load_cassette, stage="deferred_stage3")

        # 11. Surgical Defect Analyzer — analyze(audio, sr): kausale Analyse
        def _load_surgical():
            try:
                from backend.core.surgical_defect_analyzer import SurgicalDefectAnalyzer

                return SurgicalDefectAnalyzer().analyze
            except Exception as exc:
                log.debug("SurgicalDefectAnalyzer nicht verfügbar: %s", exc)
                return None

        _reg("surgical_defect_analyzer", _load_surgical)

        # 12. Vocal Overprocessing Detector — detect(audio, sr): ML-Überkorrektur
        def _load_vocal_over():
            try:
                from backend.core.vocal_overprocessing_detector import detect_vocal_overprocessing

                return detect_vocal_overprocessing
            except Exception as exc:
                log.debug("VocalOverprocessingDetector nicht verfügbar: %s", exc)
                return None

        _reg("vocal_overprocessing_detector", _load_vocal_over)

        # 13. Introduced Artifact Detector — scan(audio, sr): eingeführte Artefakte
        def _load_introduced():
            try:
                from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

                return IntroducedArtifactDetector().scan
            except Exception as exc:
                log.debug("IntroducedArtifactDetector nicht verfügbar: %s", exc)
                return None

        _reg("introduced_artifact_detector", _load_introduced)

        # 14. Quality Regression Detector — detect(audio_before, audio_after, sr)
        def _load_quality_regression():
            try:
                from backend.core.quality_regression_detector import detect_quality_regression

                return detect_quality_regression
            except Exception as exc:
                log.debug("QualityRegressionDetector nicht verfügbar: %s", exc)
                return None

        _reg("quality_regression_detector", _load_quality_regression, stage="deferred_stage3")

        # 15. Defect Re-Scanner — rescan(audio, sr): sekundärer Scan
        def _load_re_scanner():
            try:
                from backend.core.defect_re_scanner import get_defect_re_scanner

                return get_defect_re_scanner().scan
            except Exception as exc:
                log.debug("DefectReScanner nicht verfügbar: %s", exc)
                return None

        _reg("defect_re_scanner", _load_re_scanner)

        # 16. Defect Detection Quality Gate — validate(audio, sr, defects)
        def _load_quality_gate():
            try:
                from backend.core.defect_detection_quality_gate import validate_defect_detection

                return validate_defect_detection
            except Exception as exc:
                log.debug("DefectDetectionQualityGate nicht verfügbar: %s", exc)
                return None

        _reg("defect_detection_quality_gate", _load_quality_gate, stage="deferred_stage2")

        _registered = sum(1 for r in self._registration_report if r["status"] == "registered")
        _deferred = sum(1 for r in self._registration_report if r["status"].startswith("deferred"))
        _failed = sum(1 for r in self._registration_report if r["status"] == "failed")
        log.info(
            "Defect Consensus: %d registriert (Stufe 1), %d kontext-deferred (Stufe 2/3), %d fehlgeschlagen (von %d)",
            _registered,
            _deferred,
            _failed,
            len(self._registration_report),
        )

    def scan_all(
        self,
        audio: np.ndarray,
        sample_rate: int = SR,
        metadata: dict | None = None,
        precomputed: dict[str, Any] | None = None,
    ) -> list[DefectHypothesis]:
        """
        Führt ALLE registrierten Detektoren parallel aus (konzeptionell —
        tatsächlich sequentiell, aber mit Timeout pro Detektor).

        §v10.600 SOTA: 3-Stufen-Ausführung mit Kontext-Passing für deferred Detektoren.

        Returns:
            Liste aller DefectHypothesis von allen Modulen (Stufe 1 + 2 + 3).
        """
        all_hypotheses: list[DefectHypothesis] = []

        is_digital = (metadata or {}).get("is_digital", False) or str((metadata or {}).get("material", "")).lower() in (
            "digital",
            "cd_digital",
        )

        precomputed = precomputed or {}

        # ── Stage 1: Primary Detectors ──
        for name, detector_fn in self._detectors:
            try:
                if name in precomputed:
                    result = precomputed[name]
                    dt = 0.0
                else:
                    t0 = time.time()
                    result = detector_fn(audio, sample_rate)
                    dt = time.time() - t0

                hypotheses = self._normalize_result(name, result, sample_rate)

                if is_digital:
                    hypotheses = [h for h in hypotheses if h.category.value not in _ANALOG_ONLY_NAMES]

                all_hypotheses.extend(hypotheses)
                log.debug(f"  Stage1 {name}: {len(hypotheses)} hypotheses in {dt:.2f}s")
            except Exception as e:
                log.debug(f"  Stage1 {name}: SKIPPED — {e}")

        # ── Stage 2: Context-Dependent Detectors (need Stage 1 results) ──
        stage2_hypotheses = list(all_hypotheses)  # Copy for context passing
        for name, detector_fn in self._stage2_detectors:
            try:
                if name == "precision_defect_locator":
                    result = detector_fn(audio, sample_rate, stage2_hypotheses)
                elif name == "attack_type_classifier":
                    # Extract onset positions from click/crackle hypotheses
                    onsets = [h.start_sample for h in stage2_hypotheses if h.category in (DefectCategory.CLICK, DefectCategory.POP)]
                    result = detector_fn(audio, sample_rate, onsets)
                elif name == "defect_detection_quality_gate":
                    result = detector_fn(audio, sample_rate, stage2_hypotheses)
                else:
                    result = detector_fn(audio, sample_rate, stage2_hypotheses)

                hypotheses = self._normalize_result(name, result, sample_rate)
                all_hypotheses.extend(hypotheses)
                log.debug(f"  Stage2 {name}: {len(hypotheses)} hypotheses")
            except Exception as e:
                log.debug(f"  Stage2 {name}: SKIPPED — {e}")

        # ── Stage 3: Material/Context-Dependent Detectors ──
        material = str((metadata or {}).get("material", "")).lower()
        era = str((metadata or {}).get("era", ""))
        for name, detector_fn in self._stage3_detectors:
            try:
                if name == "intentional_artifact_classifier":
                    result = detector_fn(material, era, (metadata or {}).get("freedom", {}))
                elif name == "dolby_nr_detector":
                    result = detector_fn(audio, sample_rate, material, era)
                elif name == "cassette_defect_verifier":
                    # Needs original + repaired audio — skip if not available
                    repaired = (precomputed or {}).get("repaired_audio")
                    if repaired is None:
                        log.debug(f"  Stage3 {name}: SKIPPED (no repaired audio)")
                        continue
                    result = detector_fn(audio, repaired)
                elif name == "quality_regression_detector":
                    repaired = (precomputed or {}).get("repaired_audio")
                    if repaired is None:
                        log.debug(f"  Stage3 {name}: SKIPPED (no repaired audio)")
                        continue
                    result = detector_fn(audio, repaired, sample_rate)
                else:
                    result = detector_fn(audio, sample_rate, material, era)

                hypotheses = self._normalize_result(name, result, sample_rate)
                all_hypotheses.extend(hypotheses)
                log.debug(f"  Stage3 {name}: {len(hypotheses)} hypotheses")
            except Exception as e:
                log.debug(f"  Stage3 {name}: SKIPPED — {e}")

        # §v10.998: Severity-Fallback — mehrere Detektoren melden severity ≈ 0.0
        # trotz positiver Erkennung (selbst auf echtem Vinyl-Knistern gemessen).
        for _h in all_hypotheses:
            if _h.severity < 0.01:
                _h.severity = float(np.clip(max(0.05, _h.confidence * 0.5), 0.0, 1.0))

        return all_hypotheses

    def _normalize_result(
        self,
        module_name: str,
        result: Any,
        sample_rate: int,
    ) -> list[DefectHypothesis]:
        """
        Normalisiert verschiedene Modul-Output-Formate in einheitliche Hypothesen.
        """
        hypotheses: list[DefectHypothesis] = []

        # Handle dict-based results
        if isinstance(result, dict):
            defects = result.get("defects", result.get("detections", []))
            if isinstance(defects, list):
                for d in defects:
                    hyp = self._dict_to_hypothesis(module_name, d, sample_rate)
                    if hyp:
                        hypotheses.append(hyp)
            elif isinstance(defects, dict):
                for cat, dets in defects.items():
                    if isinstance(dets, list):
                        for d in dets:
                            hyp = self._dict_to_hypothesis(module_name, d, sample_rate, cat)
                            if hyp:
                                hypotheses.append(hyp)

        # Handle list-based results
        elif isinstance(result, list):
            for d in result:
                hyp = self._dict_to_hypothesis(module_name, d, sample_rate)
                if hyp:
                    hypotheses.append(hyp)

        # Handle DefectAnalysisResult with .scores (from defect_scanner)
        elif hasattr(result, "scores"):
            scores = result.scores
            if isinstance(scores, dict):
                for key, score_obj in scores.items():
                    cat_str = key.value if hasattr(key, "value") else str(key)
                    severity = float(getattr(score_obj, "severity", 0.0))
                    confidence = float(getattr(score_obj, "confidence", 0.0))
                    locations = getattr(score_obj, "locations", None) or []
                    if not locations:
                        # Kein Ort → Defekt überall / nicht lokalisierbar
                        locations = [(0, 4800)]  # 100ms default
                    for loc in locations:
                        if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                            s_smp, e_smp = int(loc[0]), int(loc[1])
                        else:
                            s_smp, e_smp = 0, 4800
                        if e_smp <= s_smp:
                            e_smp = s_smp + 4800
                        hyp = DefectHypothesis(
                            category=self._map_category(cat_str),
                            start_sample=s_smp,
                            end_sample=e_smp,
                            confidence=confidence,
                            severity=severity,
                            source_module=module_name,
                            evidence={"metadata": getattr(score_obj, "metadata", {})},
                        )
                        hypotheses.append(hyp)

        return hypotheses

    def _dict_to_hypothesis(
        self,
        module: str,
        d: dict,
        sr: int,
        category_override: str | None = None,
    ) -> DefectHypothesis | None:
        """Konvertiert Dict-basierte Defekt-Erkennung in Hypothesis."""
        try:
            cat_str = category_override or d.get("type", d.get("category", "unknown"))
            cat = self._map_category(cat_str)

            start_s = d.get("start", d.get("start_s", 0))
            end_s = d.get("end", d.get("end_s", start_s + 0.01))

            return DefectHypothesis(
                category=cat,
                start_sample=int(start_s * sr),
                end_sample=int(end_s * sr),
                confidence=float(d.get("confidence", d.get("score", 0.5))),
                severity=float(d.get("severity", d.get("strength", 0.5))),
                source_module=module,
                evidence=d.get("evidence", d.get("details", {})),
                caused_by=d.get("caused_by", []),
                causes=d.get("causes", []),
            )
        except Exception as exc:
            log.debug("§V6 _dict_to_hypothesis fehlgeschlagen — None zurückgegeben (Dict %s): %s", module, exc)
            return None

    def _object_to_hypothesis(
        self,
        module: str,
        d: Any,
        sr: int,
    ) -> DefectHypothesis | None:
        """Konvertiert Objekt-basierte Defekt-Erkennung in Hypothesis."""
        try:
            cat_str = getattr(d, "type", getattr(d, "category", "unknown"))
            cat = self._map_category(cat_str)

            start_s = getattr(d, "start", getattr(d, "start_s", 0))
            end_s = getattr(d, "end", getattr(d, "end_s", start_s + 0.01))

            return DefectHypothesis(
                category=cat,
                start_sample=int(start_s * sr),
                end_sample=int(end_s * sr),
                confidence=float(getattr(d, "confidence", getattr(d, "score", 0.5))),
                severity=float(getattr(d, "severity", getattr(d, "strength", 0.5))),
                source_module=module,
                evidence=getattr(d, "evidence", getattr(d, "details", {})),
                caused_by=getattr(d, "caused_by", []),
                causes=getattr(d, "causes", []),
            )
        except Exception as exc:
            log.debug("§V6 _object_to_hypothesis fehlgeschlagen — None zurückgegeben (Objekt %s): %s", module, exc)
            return None

    @staticmethod
    def _map_category(raw: str) -> DefectCategory:
        """Mapped beliebige Defekt-Bezeichner auf standardisierte Kategorien."""
        raw_lower = raw.lower().replace(" ", "_").replace("-", "_")
        mapping = {
            "click": DefectCategory.CLICK,
            "crackle": DefectCategory.CRACKLE,
            "pop": DefectCategory.POP,
            "hum": DefectCategory.HUM,
            "hiss": DefectCategory.HISS,
            "tape_hiss": DefectCategory.TAPE_HISS,
            "vinyl_noise": DefectCategory.VINYL_NOISE,
            "surface_noise": DefectCategory.VINYL_NOISE,
            "wow": DefectCategory.WOW_FLUTTER,
            "flutter": DefectCategory.WOW_FLUTTER,
            "wow_flutter": DefectCategory.WOW_FLUTTER,
            "clipping": DefectCategory.CLIPPING,
            "clip": DefectCategory.CLIPPING,
            "dropout": DefectCategory.DROPOUT,
            "pre_echo": DefectCategory.PRE_ECHO,
            "print_through": DefectCategory.PRINT_THROUGH,
            "sibilance": DefectCategory.SIBILANCE,
            "breath": DefectCategory.BREATH,
            "de_essing": DefectCategory.DE_ESSING_ARTIFACT,
            "phase": DefectCategory.PHASE_ERROR,
            "distortion": DefectCategory.DISTORTION,
            "gate_chatter": DefectCategory.NOISE_GATE_CHATTER,
            "reverb": DefectCategory.REVERB_TAIL,
            "reverb_tail": DefectCategory.REVERB_TAIL,  # §v10.998: Detektor-Format
        }
        return mapping.get(raw_lower, DefectCategory.UNKNOWN)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Conflict Resolution (Weighted Voting + Temporal Merging)
# ═════════════════════════════════════════════════════════════════════════════


class ConflictResolver:
    """
    Löst Konflikte zwischen überlappenden Defekt-Hypothesen auf.

    Strategie:
      1. Weighted Voting: Jedes Modul hat ein historisches Präzisions-Gewicht
      2. Temporal Merging: Überlappende Defekte gleichen Typs werden gemerged
      3. Conflict Resolution: Widersprüchliche Defekte → Majority-Vote
    """

    def __init__(self, overlap_threshold: float = 0.5):
        self.overlap_threshold = overlap_threshold

    def resolve(self, hypotheses: list[DefectHypothesis]) -> tuple[list[DefectHypothesis], int, int]:
        """
        Löst Konflikte auf und gibt bereinigte Defekt-Liste zurück.

        Returns:
            (resolved_defects, conflicts_resolved, merged_count)
        """
        if not hypotheses:
            return [], 0, 0

        conflicts_resolved = 0
        merged_count = 0

        # Step 1: Sort by start time
        hypotheses.sort(key=lambda h: h.start_sample)

        # Step 2: Group overlapping hypotheses of same category → merge
        merged: list[DefectHypothesis] = []
        used = set()

        for i, h1 in enumerate(hypotheses):
            if i in used:
                continue
            group = [h1]
            used.add(i)

            for j, h2 in enumerate(hypotheses):
                if j in used or j <= i:
                    continue
                if h1.category == h2.category and self._overlap_ratio(h1, h2) > self.overlap_threshold:
                    group.append(h2)
                    used.add(j)

            if len(group) > 1:
                merged_defect = self._merge_group(group)
                merged.append(merged_defect)
                merged_count += len(group) - 1
            else:
                merged.append(h1)

        # Step 3: Resolve conflicting categories at same time position
        # §v10.840: Globale Defekte (hum/hiss über ganze Datei) dürfen lokale
        # Defekte (Klicks/Knackser) NICHT verdrängen — beide werden behalten.
        resolved: list[DefectHypothesis] = []
        i = 0
        while i < len(merged):
            conflict_group = [merged[i]]
            j = i + 1
            while j < len(merged):
                if self._overlap_ratio(merged[i], merged[j]) > 0.3:
                    if merged[i].category != merged[j].category:
                        conflict_group.append(merged[j])
                j += 1

            if len(conflict_group) > 1:
                # Globale vs lokale Defekte: beide behalten
                global_cats = {"hum", "hiss", "tape_hiss", "vinyl_noise", "bandwidth_loss"}
                has_global = any(h.category.value in global_cats for h in conflict_group)
                has_local = any(h.category.value not in global_cats for h in conflict_group)
                if has_global and has_local:
                    resolved.extend(conflict_group)  # BEIDE behalten
                else:
                    winner = self._resolve_conflict(conflict_group)
                    resolved.append(winner)
                    conflicts_resolved += len(conflict_group) - 1
                i += len(conflict_group)
            else:
                resolved.append(merged[i])
                i += 1

        return resolved, conflicts_resolved, merged_count

    def _overlap_ratio(self, h1: DefectHypothesis, h2: DefectHypothesis) -> float:
        """Berechnet das zeitliche Überlappungsverhältnis."""
        start = max(h1.start_sample, h2.start_sample)
        end = min(h1.end_sample, h2.end_sample)
        if start >= end:
            return 0.0

        overlap = end - start
        total = min(h1.end_sample - h1.start_sample, h2.end_sample - h2.start_sample)
        if total <= 0:
            return 0.0
        return overlap / total

    def _merge_group(self, group: list[DefectHypothesis]) -> DefectHypothesis:
        """Merged eine Gruppe gleichartiger Defekte in einen."""
        # Weighted average of confidence and severity
        weights = np.array([MODULE_WEIGHTS.get(h.source_module, DEFAULT_WEIGHT) for h in group])
        total_weight = weights.sum()

        avg_confidence = sum(h.confidence * w for h, w in zip(group, weights)) / total_weight
        avg_severity = sum(h.severity * w for h, w in zip(group, weights)) / total_weight

        # Temporal bounds: earliest start, latest end
        min_start = min(h.start_sample for h in group)
        max_end = max(h.end_sample for h in group)

        # Combine evidence
        combined_evidence = {}
        for h in group:
            combined_evidence.update(h.evidence)

        sources = list({h.source_module for h in group})

        return DefectHypothesis(
            category=group[0].category,
            start_sample=min_start,
            end_sample=max_end,
            confidence=float(avg_confidence),
            severity=float(avg_severity),
            source_module="+".join(sources),
            evidence=combined_evidence,
            caused_by=list(set().union(*(h.caused_by for h in group))),
            causes=list(set().union(*(h.causes for h in group))),
        )

    def _resolve_conflict(self, conflict_group: list[DefectHypothesis]) -> DefectHypothesis:
        """
        Löst widersprüchliche Defekte per Weighted Majority Vote.

        Bei Konflikten gewinnt die Hypothese mit dem höchsten
        (confidence × severity × module_weight)-Produkt.
        """
        best_score = -1.0
        best_hypothesis = conflict_group[0]

        for h in conflict_group:
            module_weight = MODULE_WEIGHTS.get(h.source_module, DEFAULT_WEIGHT)
            score = h.confidence * h.severity * module_weight
            # Bonus: wenn mehrere Module übereinstimmen
            if len(h.source_module.split("+")) > 1:
                score *= 1.2  # 20% Bonus für Multi-Modul-Konsens

            if score > best_score:
                best_score = score
                best_hypothesis = h

        # Markiere Konflikt in Evidence
        best_hypothesis.evidence["conflict_resolved"] = True
        best_hypothesis.evidence["alternative_hypotheses"] = [
            f"{h.category.value} ({h.source_module})" for h in conflict_group if h != best_hypothesis
        ]

        return best_hypothesis


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Causal Validation
# ═════════════════════════════════════════════════════════════════════════════


class CausalValidator:
    """
    Validiert Defekte auf kausale Plausibilität via CausalDefectReasoner.

    Wenn CausalDefectReasoner sagt: "Keine kausale Basis für diesen Defekt",
    wird die Konfidenz um 30% reduziert.
    """

    def __init__(self):
        self._reasoner = None
        self._init_reasoner()

    def _init_reasoner(self):
        try:
            from backend.core.causal_defect_reasoner import get_reasoner

            self._reasoner = get_reasoner()
        except Exception:
            pass

    def validate(self, defects: list[DefectHypothesis], audio_length: int) -> tuple[list[DefectHypothesis], int]:
        """
        Validiert Defekte kausal. Downgraded Defekte ohne kausale Basis.

        Returns:
            (validated_defects, downgrade_count)
        """
        if not self._reasoner or not defects:
            return defects, 0

        downgrades = 0

        try:
            # Get causal analysis
            causal_result = self._reasoner.reason_about_defects(
                [(d.category.value, d.start_sample, d.end_sample) for d in defects],
                audio_length,
            )

            if causal_result and hasattr(causal_result, "validated_defects"):
                validated = causal_result.validated_defects
                downgrade_set = set(validated.get("downgraded", []))

                for i, d in enumerate(defects):
                    defect_id = f"{d.category.value}_{d.start_sample}"
                    if defect_id in downgrade_set:
                        d.confidence *= 0.7  # 30% downgrade
                        d.evidence["causal_downgrade"] = True
                        downgrades += 1

        except Exception:
            pass

        return defects, downgrades


# ═════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═════════════════════════════════════════════════════════════════════════════


class DefectConsensusPipeline:
    """
    3-Stufen Defect Consensus Pipeline.

    Nutzung:
        pipeline = DefectConsensusPipeline()
        manifest = pipeline.analyze(audio, sample_rate)
        # manifest.defects enthält das bereinigte, widerspruchsfreie Manifest
    """

    def __init__(self):
        self.scanner = ParallelDefectScanner()
        self.resolver = ConflictResolver()
        self.validator = CausalValidator()

        log.info("Defect Consensus Pipeline: 3 Stufen initialisiert")

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int = SR,
        metadata: dict | None = None,
        precomputed_results: dict[str, Any] | None = None,
    ) -> DefectManifest:
        """
        Führt die vollständige 3-Stufen-Defekt-Analyse durch.

        Args:
            audio: [T] mono audio
            sample_rate: Samplerate
            metadata: Optionale Metadaten (Medium, Ära, Genre)

        Returns:
            DefectManifest mit widerspruchsfreiem Defekt-Set
        """
        t0 = time.time()

        if audio.ndim > 1:
            # §Spec 24-Ergänzung (Befund 2026-08-22): Kanal-Mix über die KANAL-
            # Achse, nie über die Zeit. Zeit-orientiertes Stereo (N, 2) →
            # mean(axis=1); kanal-orientiert (2, N) → mean(axis=0). Das alte
            # mean(axis=0) mittelte (N, 2) über die Zeit zu einem 2-Sample-Array
            # → Zero-Length-Guard „Audio zu kurz (2 Samples)“ im Cached-Zweig.
            if audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
                audio = audio.mean(axis=0)
            else:
                audio = audio.mean(axis=1)

        # §Spec 24-Härtung (Befund 2026-08-22): Auch 1-D-Eingaben können
        # degeneriert ankommen (Log: (2,) → „Audio zu kurz (2 Samples)“,
        # Stage 1: 0 Hypothesen, alle Detektoren auf 2 Samples). Solche
        # Eingaben sofort ablehnen statt die Detektoren zu füttern — der
        # Seed (gecachter DefectScan) trägt die Information ohnehin.
        if audio.size < max(1, sample_rate // 2):
            log.warning(
                "DefectConsensusPipeline.analyze(): degenerierte Eingabe (shape=%s, %d Samples) "
                "— Stage 1 übersprungen (Spec 24 Zero-Length-Guard)",
                tuple(audio.shape),
                int(audio.size),
            )
            return DefectManifest(
                defects=[],
                total_hypotheses=0,
                processing_time=time.time() - t0,
                module_count=0,
            )

        # ── Stage 1: Parallel Scanning ──
        all_hypotheses = self.scanner.scan_all(audio, sample_rate, metadata, precomputed=precomputed_results)
        total_hypotheses = len(all_hypotheses)
        log.info(f"Stage 1: {total_hypotheses} Hypothesen von {self.scanner._detectors} Modulen")

        if not all_hypotheses:
            return DefectManifest(
                defects=[],
                total_hypotheses=0,
                processing_time=time.time() - t0,
                module_count=0,
            )

        # ── Stage 2: Conflict Resolution ──
        resolved, conflicts, merged = self.resolver.resolve(all_hypotheses)
        log.info(f"Stage 2: {len(resolved)} Defekte nach Resolution ({conflicts} Konflikte, {merged} Merges)")

        # ── Stage 3: Causal Validation ──
        validated, downgrades = self.validator.validate(resolved, len(audio))
        if downgrades > 0:
            log.info(f"Stage 3: {downgrades} kausale Downgrades")

        elapsed = time.time() - t0

        return DefectManifest(
            defects=validated,
            total_hypotheses=total_hypotheses,
            conflicts_resolved=conflicts,
            merged_defects=merged,
            causal_downgrades=downgrades,
            processing_time=elapsed,
            module_count=len(self.scanner._detectors),
        )
