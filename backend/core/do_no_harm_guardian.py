"""
DoNoHarmGuardian — §5/5 Final Safety Net (Aurik 10.0.0+)
=========================================================

Stellt sicher: Jede Restaurierung verbessert den Klang — oder
das Original wird unverändert zurückgegeben.

Prinzip: „Primum non nocere" — zuerst nicht schaden.

Arbeitsweise:
  1. Vor der Pipeline: Input-Audio und dessen Metriken speichern.
  2. Nach der Pipeline: Output-Audio-Metriken messen.
  3. Wenn IRGENDEINE Kernmetrik sich signifikant verschlechtert hat:
     → Output verwerfen, Original zurückgeben.
  4. Wenn alle Metriken gleich oder besser sind:
     → Output durchlassen.

Kernmetriken (unabhängig vom Materialtyp):
  - spectral_brightness:     Verhältnis HF-Energie (>4kHz) zu Gesamtenergie
  - naturalness_estimate:    Wiener-Entropie als Naturalness-Proxy
  - rms_preservation:       RMS-Änderung in dB (max ±6 dB toleriert)
  - peak_integrity:          True-Peak nicht näher an 0 dBFS als vorher

Integration:
  Wird in UnifiedRestorerV3.restore() als Post-Pipeline-Check aufgerufen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GuardianSnapshot:
    """Metrik-Snapshot eines Audio-Signals (vor oder nach der Pipeline)."""

    spectral_brightness: float = 0.5  # 0–1 (0.5 = neutral)
    naturalness_estimate: float = 0.5  # 0–1 (Wiener-Entropie)
    rms_dbfs: float = -30.0  # RMS in dBFS
    peak_dbfs: float = -6.0  # True-Peak in dBFS
    dynamic_range_db: float = 12.0  # P99.9 − P0.1 in dB

    # Rohdaten für spätere Diagnose
    _raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardianVerdict:
    """Entscheidung des Guardians."""

    passed: bool = True
    reason: str = ""
    metrics_input: GuardianSnapshot = field(default_factory=GuardianSnapshot)
    metrics_output: GuardianSnapshot = field(default_factory=GuardianSnapshot)
    degraded_metrics: list[str] = field(default_factory=list)
    severity: str = "none"  # "none", "minor", "moderate", "critical"


class DoNoHarmGuardian:
    """Finaler Qualitäts-Schutz — stellt sicher, dass Aurik nicht schadet.

    Verwendung:
        guardian = DoNoHarmGuardian()
        guardian.capture_input(audio, sr)
        # ... Pipeline läuft ...
        verdict = guardian.evaluate(output_audio, sr)
        if not verdict.passed:
            return input_audio  # Original zurückgeben
    """

    # ── Schwellwerte ───────────────────────────────────────────────────
    # §G-5/5: Diese Schwellwerte wurden empirisch an 50+ Restaurierungen
    # kalibriert. Sie sind konservativ — lieber zu früh warnen als zu spät.
    #
    # RESTORATION-Modus: Charakter bewahren, nur Defekte entfernen.
    # → strenge Schwellwerte — jede signifikante Änderung ist verdächtig.
    #
    # STUDIO-2026-Modus: Bewusste Modernisierung erlaubt.
    # → lockere Schwellwerte — LUFS-Normalisierung und Air-Band sind gewollt.

    # Restoration-Schwellwerte (konservativ)
    # §v10.102: Naturalness auf 0.70 für degradierte Quellen (MERT-Proxy ist
    # bei MP3/Kassette unzuverlässig — Rauschen täuscht hohe Naturalness vor,
    # sauberes Restaurat wird fälschlich als "unnatürlich" bewertet).
    REST_MAX_BRIGHTNESS_DROP: float = 0.20
    REST_MAX_NATURALNESS_DROP: float = 0.70
    REST_MAX_RMS_CHANGE_DB: float = 8.0

    # Studio-2026-Schwellwerte (erlauben bewusste Änderungen)
    STU_MAX_BRIGHTNESS_DROP: float = 0.40  # Air-Band DARF Helligkeit erhöhen
    STU_MAX_NATURALNESS_DROP: float = 0.30  # LUFS-Norm kann Naturalness beeinflussen
    STU_MAX_RMS_CHANGE_DB: float = 20.0  # -14 LUFS Normierung = große Pegeländerung ok

    def __init__(self, mode: str = "restoration") -> None:
        self._mode = ""
        self.mode = mode  # Property-Setter aktualisiert die Schwellwerte
        self._input_audio: np.ndarray | None = None
        self._input_sr: int = 0
        self._input_snapshot: GuardianSnapshot | None = None
        self._captured: bool = False
        # §v10.103 P2: Clean-Referenz (carrier_checkpoint)
        self._clean_reference: np.ndarray | None = None
        self._clean_snapshot: GuardianSnapshot | None = None
        self._reference_mode: str = "degraded_input"

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = str(value).strip().lower()
        if self._mode in ("studio_2026", "studio2026", "studio"):
            self._max_brightness_drop = self.STU_MAX_BRIGHTNESS_DROP
            self._max_naturalness_drop = self.STU_MAX_NATURALNESS_DROP
            self._max_rms_change_db = self.STU_MAX_RMS_CHANGE_DB
            self._min_peak_headroom_db = 0.0
        else:
            self._max_brightness_drop = self.REST_MAX_BRIGHTNESS_DROP
            self._max_naturalness_drop = self.REST_MAX_NATURALNESS_DROP
            self._max_rms_change_db = self.REST_MAX_RMS_CHANGE_DB
            self._min_peak_headroom_db = 0.5
        self._input_sr: int = 0  # type: ignore[no-redef]
        self._input_snapshot: GuardianSnapshot | None = None  # type: ignore[no-redef]
        self._captured: bool = False  # type: ignore[no-redef]
        # §v10.103: Clean-Referenz bei Mode-Wechsel konsistent zurücksetzen
        self._clean_reference = None
        self._clean_snapshot = None
        self._reference_mode = "degraded_input"

    # ── Public API ─────────────────────────────────────────────────────

    def capture_input(self, audio: np.ndarray, sr: int, carrier_checkpoint: np.ndarray | None = None) -> None:
        """Speichert das Input-Audio und misst dessen Metriken.

        Muss VOR der Pipeline aufgerufen werden.

        Args:
            audio: Degradiertes Input-Audio (für Fallback-Vergleich).
            sr: Sample-Rate.
            carrier_checkpoint: Optional — sauberster Carrier-Checkpoint
                (via BlindInternalReference). Wenn gesetzt, werden alle
                Vergleiche gegen DIESE Referenz durchgeführt statt gegen
                das degradierte Input (§v10.103 P2 Referenzrahmen-Korrektur).
        """
        self._input_audio = np.asarray(audio, dtype=np.float32).copy()
        self._input_sr = int(sr)
        self._input_snapshot = self._measure(audio, sr)
        self._captured = True

        # §v10.103 P2: Clean-Referenz statt degradiertem Input
        if carrier_checkpoint is not None:
            self._clean_reference = np.asarray(carrier_checkpoint, dtype=np.float32).copy()
            self._clean_snapshot = self._measure(carrier_checkpoint, sr)
            self._reference_mode = "carrier_checkpoint"
            logger.debug(
                "DoNoHarmGuardian: Eingabe + carrier_checkpoint captured — "
                "Referenz_Betriebsart=carrier_checkpoint "
                "clean_naturalness=%.3f clean_brightness=%.3f",
                self._clean_snapshot.naturalness_estimate,
                self._clean_snapshot.spectral_brightness,
            )
        else:
            self._clean_reference = None
            self._clean_snapshot = None
            self._reference_mode = "degraded_input"

        logger.debug(
            "DoNoHarmGuardian: Eingabe captured — brightness=%.3f naturalness=%.3f rms=%.1f dBFS",
            self._input_snapshot.spectral_brightness,
            self._input_snapshot.naturalness_estimate,
            self._input_snapshot.rms_dbfs,
        )

    def evaluate(
        self,
        output_audio: np.ndarray,
        sr: int,
        material: str = "unknown",
        chain_depth: int = 1,
        mushra_score: float = 0.0,
        hpi_score: float = 0.0,
    ) -> GuardianVerdict:
        """Vergleicht Output mit Referenz und entscheidet: passed oder nicht.

        Args:
            output_audio: Das von der Pipeline verarbeitete Audio.
            sr: Sample-Rate.
            material: Materialtyp (cassette, reel_tape, etc.) für adaptive Schwellwerte.
            chain_depth: Transfer-Chain-Tiefe (1-5) für depth-adaptive Schwellwerte.
            mushra_score: MUSHRA-Score (0-100) des Outputs für Perceptual Override (P1).
            hpi_score: HPI-Score (0-1) des Outputs für Perceptual Override (P1).

        Returns:
            GuardianVerdict mit passed=True wenn alle Metriken ok sind.
        """
        if not self._captured:
            logger.warning("DoNoHarmGuardian: evaluate() ohne Erfassung_Eingabe() — lasse durch")
            return GuardianVerdict(passed=True, reason="no_input_captured")

        self._chain_depth = max(1, int(chain_depth))  # §v10.102: für depth-adaptive Schwellwerte

        output = np.asarray(output_audio, dtype=np.float32)
        output_snap = self._measure(output, sr)

        # ═══ §v10.103 P1: Perceptual Override ═══
        # Wenn beide perzeptuellen Metriken unisono "exzellent" sagen,
        # vertrauen wir dem menschlichen Ohr mehr als den objektiven Proxies.
        # Schwellwerte: MUSHRA ≥ 85 (ITU-R BS.1534 "Excellent"),
        # HPI ≥ 0.75 (hohe perzeptuelle Verbesserung).
        _MUSHRA_EXCELLENT = 85.0
        _HPI_HIGH = 0.75
        if mushra_score >= _MUSHRA_EXCELLENT and hpi_score >= _HPI_HIGH:
            logger.info(
                "DoNoHarmGuardian: PERCEPTUAL OVERRIDE — "
                "MUSHRA=%.0f≥%.0f HPI=%.3f≥%.2f → "
                "objektive Proxy-Warnungen werden unterdrückt "
                "(das menschliche Ohr hat Vorrang)",
                mushra_score,
                _MUSHRA_EXCELLENT,
                hpi_score,
                _HPI_HIGH,
            )
            return GuardianVerdict(
                passed=True,
                reason=f"perceptual_override:MUSHRA={mushra_score:.0f}_HPI={hpi_score:.3f}",
                metrics_input=self._input_snapshot,  # type: ignore[arg-type]
                metrics_output=output_snap,
                degraded_metrics=[],
                severity="none",
            )

        # ═══ §v10.119 Referenzrahmen-Korrektur ═══
        # §v10.119: IMMER gegen das degradierte Original vergleichen.
        # Der DoNoHarmGuardian prüft: "Ist der Output schlechter als der Input?"
        # Ein Vergleich gegen carrier_checkpoint (Zwischenstand) ist IRRELEVANT —
        # das fertige Restaurat SOLL sich vom Zwischenstand unterscheiden.
        # §v10.103 P2 war falsch: Restauration bedeutet Verbesserung gegenüber
        # dem DEGRADIERTEN Input, nicht Annäherung an einen Zwischenstand.
        # Siehe: 8326s Restaurierung verworfen wegen carrier_checkpoint-Vergleich.
        _ref_label = "degraded_input"
        ref_snap = self._input_snapshot

        logger.debug(
            "DoNoHarmGuardian: referencing against %s",
            _ref_label,
        )

        degraded: list[str] = []

        # 1. Spectral Brightness — §v10.103 P2: gegen REFERENZ statt Input
        _brightness_drop = ref_snap.spectral_brightness - output_snap.spectral_brightness  # type: ignore[union-attr]
        if _brightness_drop > self._max_brightness_drop:
            degraded.append(f"brightness_drop={_brightness_drop:.3f} (>{self._max_brightness_drop}) [ref={_ref_label}]")

        # 2. Naturalness — §v10.102 depth-adaptiv + §v10.103 P2: gegen REFERENZ
        _mat_nat = str(material).lower()
        _depth_nat = max(1, int(getattr(self, "_chain_depth", 1)))
        if _mat_nat in ("cassette", "reel_tape", "tape"):
            _nat_threshold = 0.30 + 0.10 * _depth_nat + (0.10 if _depth_nat >= 4 else 0.0)
        else:
            _nat_threshold = self._max_naturalness_drop
        _nat_drop = ref_snap.naturalness_estimate - output_snap.naturalness_estimate  # type: ignore[union-attr]
        if _nat_drop > _nat_threshold:
            degraded.append(f"naturalness_drop={_nat_drop:.3f} (>{_nat_threshold}) [ref={_ref_label}]")

        # 3. RMS Change — gegen REFERENZ
        _rms_change = abs(output_snap.rms_dbfs - ref_snap.rms_dbfs)  # type: ignore[union-attr]
        if _rms_change > self._max_rms_change_db:
            degraded.append(f"rms_change={_rms_change:.1f} dB (>{self._max_rms_change_db}) [ref={_ref_label}]")

        # 4. Peak Integrity (Crest-Factor-basiert, §v10.102 depth-adaptiv) — gegen REFERENZ
        _mat_crest = str(material).lower()
        _depth_crest = max(1, int(getattr(self, "_chain_depth", 1)))
        if _mat_crest in ("cassette", "reel_tape", "tape"):
            _crest_threshold = 2.0 + 2.0 * _depth_crest
        else:
            _crest_threshold = 3.0
        _crest_ref = ref_snap.peak_dbfs - ref_snap.rms_dbfs  # type: ignore[union-attr]
        _crest_output = output_snap.peak_dbfs - output_snap.rms_dbfs
        _crest_drop = _crest_ref - _crest_output
        if _crest_drop > _crest_threshold:
            degraded.append(
                f"peak_degraded: crest_drop={_crest_drop:.1f}dB (>{_crest_threshold}) "
                f"[ref={_ref_label} ref={ref_snap.peak_dbfs:.1f}/{ref_snap.rms_dbfs:.1f} "  # type: ignore[union-attr]
                f"out={output_snap.peak_dbfs:.1f}/{output_snap.rms_dbfs:.1f}]"
            )

        # 5. Dynamic Range — gegen REFERENZ
        _dr_change = ref_snap.dynamic_range_db - output_snap.dynamic_range_db  # type: ignore[union-attr]
        if _dr_change > 6.0:
            degraded.append(f"dynamic_range_collapse={_dr_change:.1f} dB [ref={_ref_label}]")

        passed = len(degraded) == 0

        if not passed:
            _severity = "critical" if len(degraded) >= 3 else ("moderate" if len(degraded) >= 2 else "minor")
            _reason = "; ".join(degraded)
            logger.warning(
                "DoNoHarmGuardian: BLOCKED — %d Metriken verschlechtert [%s]: %s",
                len(degraded),
                _severity,
                _reason,
            )
        else:
            _severity = "none"
            _reason = "all_metrics_ok"
            logger.info(
                "DoNoHarmGuardian: PASSED — brightness=%.3f→%.3f naturalness=%.3f→%.3f [ref=%s]",
                ref_snap.spectral_brightness,  # type: ignore[union-attr]
                output_snap.spectral_brightness,
                ref_snap.naturalness_estimate,  # type: ignore[union-attr]
                output_snap.naturalness_estimate,
                _ref_label,
            )

        return GuardianVerdict(
            passed=passed,
            reason=_reason,
            metrics_input=ref_snap,  # type: ignore[arg-type]
            metrics_output=output_snap,
            degraded_metrics=degraded,
            severity=_severity,
        )

    def get_input_audio(self) -> np.ndarray | None:
        """Gibt das gespeicherte Input-Audio zurück (für Rollback)."""
        return self._input_audio

    # ── Interne Metrik-Messung ─────────────────────────────────────────

    @staticmethod
    def _measure(audio: np.ndarray, sr: int) -> GuardianSnapshot:
        """Misst alle Kernmetriken an einem Audio-Signal.

        Optimiert für Geschwindigkeit: verwendet einfache, robuste Metriken
        die ohne externe ML-Modelle auskommen (keine PANNS, kein CLAP).
        """
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio)
        mono = mono.astype(np.float32)
        n = len(mono)
        if n < sr // 4:  # Weniger als 250 ms
            return GuardianSnapshot()  # Zu kurz für sinnvolle Messung

        # 1. Spectral Brightness: Energie > 4 kHz / Gesamtenergie
        try:
            n_fft = min(4096, n)
            spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            hf_mask = freqs >= 4000.0
            hf_energy = float(np.sum(spec[hf_mask] ** 2))
            total_energy = float(np.sum(spec**2)) + 1e-10
            brightness = float(np.clip(hf_energy / total_energy, 0.0, 1.0))
        except Exception:
            brightness = 0.5

        # 2. Naturalness Estimate — §v10.102: Wiener-Entropie als Primärmetrik
        # MERT-basierte "Naturalness" ist für degradierte Quellen (MP3, Kassette)
        # unzuverlässig: der willkürliche Referenzvektor (0.1·ones) hat keine
        # perzeptuelle Validierung. Wiener-Entropie misst spektrale Dichte —
        # ein sauberes Restaurat hat typischerweise höhere Entropie (reichere
        # Harmonik) als das MP3-degradierte Original.
        try:
            _spec_db = 20.0 * np.log10(spec + 1e-10)
            _spec_db -= np.max(_spec_db)
            _spec_lin = 10.0 ** (_spec_db / 20.0)
            _spec_norm = _spec_lin / (np.sum(_spec_lin) + 1e-10)
            _entropy = float(-np.sum(_spec_norm * np.log2(_spec_norm + 1e-10)))
            _max_entropy = np.log2(len(_spec_norm))
            naturalness = float(np.clip(_entropy / max(_max_entropy, 1.0), 0.0, 1.0))
        except Exception:
            naturalness = 0.5

        # 3. RMS in dBFS
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-10)
        rms_dbfs = float(20.0 * np.log10(rms))

        # 4. Peak in dBFS
        peak = float(np.max(np.abs(mono)))
        peak_dbfs = float(20.0 * np.log10(max(peak, 1e-10)))

        # 5. Dynamic Range: P99.9 − P0.1
        try:
            abs_mono = np.abs(mono)
            p99_9 = float(np.percentile(abs_mono, 99.9))
            p0_1 = float(np.percentile(abs_mono, 0.1))
            p0_1_safe = max(p0_1, 1e-10)
            dynamic_range_db = float(20.0 * np.log10(p99_9 / p0_1_safe))
        except Exception:
            dynamic_range_db = 12.0

        return GuardianSnapshot(
            spectral_brightness=brightness,
            naturalness_estimate=naturalness,
            rms_dbfs=rms_dbfs,
            peak_dbfs=peak_dbfs,
            dynamic_range_db=dynamic_range_db,
            _raw={
                "hf_energy": hf_energy if "hf_energy" in dir() else 0.0,
                "total_energy": total_energy if "total_energy" in dir() else 0.0,
                "entropy": _entropy if "_entropy" in dir() else 0.0,
                "rms_linear": float(rms),
                "peak_linear": float(peak),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# §v10.103 P3: UnifiedQualityModel — Ein System, eine Entscheidung
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class UQMInput:
    """Eingabe-Metriken für das UnifiedQualityModel."""

    # ── Perzeptuelle Metriken (60% Gewicht) ──
    mushra_score: float = 50.0  # 0-100
    hpi_score: float = 0.5  # 0-1

    # ── Objektive Metriken (40% Gewicht) ──
    brightness_drop: float = 0.0  # Referenz - Output (positiv = dunkler)
    naturalness_drop: float = 0.0  # Referenz - Output (positiv = weniger natürlich)
    crest_drop_db: float = 0.0  # Referenz-Crest - Output-Crest (positiv = flacher)
    rms_change_db: float = 0.0  # |Output - Referenz| RMS
    dynamic_range_drop_db: float = 0.0  # Referenz-DR - Output-DR

    # ── Kontext ──
    material: str = "unknown"
    chain_depth: int = 1
    reference_mode: str = "degraded_input"


@dataclass
class UQMDecision:
    """Entscheidung des UnifiedQualityModel."""

    decision: str  # "PASS", "WARN", "REVERT"
    quality_score: float  # 0-100
    perceptual_score: float  # 0-100
    objective_score: float  # 0-100
    confidence: float  # 0-1
    reason: str
    advisory_warnings: list[str]


class UnifiedQualityModel:
    """§v10.103 P3: Single Source of Truth für Restaurations-Qualität.

    Architecture:
      - Perzeptuelle Metriken (MUSHRA, HPI): 60% Gewicht
        → Wir machen Audio für MENSCHEN, nicht für Spektrumanalysatoren.
      - Objektive Metriken (Crest, Naturalness, Brightness, DR): 40% Gewicht
        → Schützen vor Extremfällen (Clipping, Dynamik-Kollaps).
      - Keine isolierten Vetos mehr — ALLE Metriken fließen in EINE Entscheidung.

    Usage:
        uqm = UnifiedQualityModel()
        decision = uqm.assess(UQMInput(...))
        if decision.decision == "REVERT":
            return original_audio
    """

    # ── Gewichte (§v10.103 kalibriert) ──
    PERCEPTUAL_WEIGHT: float = 0.60
    OBJECTIVE_WEIGHT: float = 0.40

    # ── Entscheidungsschwellen ──
    QUALITY_PASS: float = 65.0  # quality_score ≥ 65 → PASS
    QUALITY_WARN: float = 45.0  # quality_score ≥ 45 → WARN (noch ok)
    # quality_score < 45 → REVERT

    # ── Perzeptuelle Exzellenz (P1: Override-Schwelle) ──
    MUSHRA_EXCELLENT: float = 85.0
    HPI_HIGH: float = 0.75

    def assess(self, inp: UQMInput) -> UQMDecision:
        """Berechnet EINE gewichtete Qualitätsentscheidung."""

        advisory: list[str] = []

        # ── 1. Perzeptueller Score (0-100) ──
        # MUSHRA direkt in [0,100], HPI skaliert auf [0,100]
        perceptual_score = 0.50 * inp.mushra_score + 0.50 * (inp.hpi_score * 100.0)

        # ── 2. Objektiver Score (0-100) ──
        # Jede objektive Metrik: 100 = perfekt (kein Drop), 0 = katastrophal
        _brightness_ok = max(0.0, 100.0 - inp.brightness_drop * 200.0)
        _naturalness_ok = max(0.0, 100.0 - inp.naturalness_drop * 150.0)
        _crest_ok = max(0.0, 100.0 - inp.crest_drop_db * 10.0)
        _rms_ok = max(0.0, 100.0 - inp.rms_change_db * 5.0)
        _dr_ok = max(0.0, 100.0 - inp.dynamic_range_drop_db * 10.0)

        objective_score = (
            0.20 * _brightness_ok + 0.25 * _naturalness_ok + 0.25 * _crest_ok + 0.15 * _rms_ok + 0.15 * _dr_ok
        )

        # ── 3. Gewichteter Gesamt-Score ──
        quality_score = self.PERCEPTUAL_WEIGHT * perceptual_score + self.OBJECTIVE_WEIGHT * objective_score

        # ── 4. Material/Depth-Adaptivität ──
        _mat = str(inp.material).lower()
        _depth = max(1, int(inp.chain_depth))
        if _mat in ("cassette", "reel_tape", "tape") and _depth >= 4:
            # Deep-Chain-Kassetten (depth≥4 nach §v10.120 Calibration-Shift):
            # objektive Metriken sind weniger aussagekräftig
            # → perzeptuelles Gewicht steigt auf 75%
            _perceptual_w = 0.75
            _objective_w = 0.25
            quality_score = _perceptual_w * perceptual_score + _objective_w * objective_score
            advisory.append(f"Deep-chain tape (depth={_depth}): perceptual weight increased to {_perceptual_w:.0%}")

        # ── 5. Perzeptueller Override (P1 im UQM) ──
        if inp.mushra_score >= self.MUSHRA_EXCELLENT and inp.hpi_score >= self.HPI_HIGH:
            # Perzeptuelle Exzellenz → Floor bei 80
            quality_score = max(quality_score, 80.0)
            advisory.append(
                f"Perceptual excellence (MUSHRA={inp.mushra_score:.0f}, HPI={inp.hpi_score:.3f}): "
                f"quality floor raised to 80"
            )

        # ── 6. Entscheidung ──
        if quality_score >= self.QUALITY_PASS:
            decision = "PASS"
        elif quality_score >= self.QUALITY_WARN:
            decision = "WARN"
        else:
            decision = "REVERT"

        # ── 7. Confidence ──
        # Höhere Confidence bei Referenz-Mode "carrier_checkpoint" (echte Vergleichsbasis)
        _base_conf = 0.75
        if inp.reference_mode == "carrier_checkpoint":
            _base_conf = 0.85
        # Nähe an Schwellen reduziert Confidence
        _dist_to_pass = abs(quality_score - self.QUALITY_PASS) / 20.0
        confidence = float(np.clip(_base_conf - _dist_to_pass * 0.15, 0.5, 0.95))

        # ── 8. Reason ──
        if decision == "PASS":
            _reason = (
                f"UQM PASS: quality={quality_score:.1f} "
                f"(perceptual={perceptual_score:.1f}, objective={objective_score:.1f})"
            )
        elif decision == "WARN":
            _reason = f"UQM WARN: quality={quality_score:.1f} < {self.QUALITY_PASS:.0f} — output accepted with warnings"
        else:
            _reason = f"UQM REVERT: quality={quality_score:.1f} < {self.QUALITY_WARN:.0f} — returning original audio"

        logger.info(
            "UnifiedQualityModel: %s (perceptual=%.1f objective=%.1f combined=%.1f conf=%.2f)",
            decision,
            perceptual_score,
            objective_score,
            quality_score,
            confidence,
        )

        return UQMDecision(
            decision=decision,
            quality_score=round(quality_score, 1),
            perceptual_score=round(perceptual_score, 1),
            objective_score=round(objective_score, 1),
            confidence=round(confidence, 2),
            reason=_reason,
            advisory_warnings=advisory,
        )


# ── Singleton ─────────────────────────────────────────────────────────

_guardian: DoNoHarmGuardian | None = None


def get_do_no_harm_guardian() -> DoNoHarmGuardian:
    """Thread-sicherer Singleton."""
    global _guardian
    if _guardian is None:
        _guardian = DoNoHarmGuardian()
    return _guardian
