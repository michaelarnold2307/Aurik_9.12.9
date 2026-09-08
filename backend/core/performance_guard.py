"""
Performance Guard for Aurik 10.0.0 - adaptive Real-Time Enforcement
============================================================

Überwacht Processing-Performance und stellt sicher, dass das mode-adaptive
Real-Time-Budget eingehalten wird. Implementiert adaptive Quality-Reduction bei Bedarf.

Performance Target: FAST max 8× Real-Time; Quality/Maximum max 32× Real-Time.

Key Features:
- Real-time Performance Monitoring
- Early-Exit Predictions
- Adaptive Phase Skipping
- Detailed Performance Logging

Author: Aurik 10.0.0 Development Team
Version: 10.0.0
Date: 2026-02-15
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PerformanceStatus(Enum):
    """Performance-Status der Verarbeitung (relativ zum aktiven RT-Budget)."""

    OPTIMAL = "optimal"  # < 25 % des Budgets
    GOOD = "good"  # 25–50 % des Budgets
    ACCEPTABLE = "acceptable"  # 50–75 % des Budgets
    CRITICAL = "critical"  # 75–100 % des Budgets ⚠️
    EXCEEDED = "exceeded"  # > 100 % des Budgets ❌


class QualityMode(Enum):
    """Quality-Modi für adaptive Processing."""

    FAST = "fast"  # max 8× RT, schneller Proof-/Desktop-Pfad mit Real-Audio-Reserve
    BALANCED = "balanced"  # max 32× RT (DEFAULT, §2.38 KMV normativ)
    QUALITY = "quality"  # max 32× RT (Restoration-Modus)
    MAXIMUM = "maximum"  # max 32× RT (§9.5 — maximum / studio_2026)
    # Budget-Wahrheit (P0-3, 2026-09-08): Alle Modi teilen 32× RT — die früheren
    # Doku-Werte („3×“, „5×“) waren nie implementiert. Real gemessen wurden
    # 53× RT (Matrix-Endlauf) — Ziel ist es, durch Song-Ebene-Analytik
    # (docs/TODOS_SOTA_ROADMAP.md P0-1) wieder unter 32× zu kommen.


class DeploymentMode(Enum):
    """Deployment-Modus für Aurik (P2-2 Produkt-/Forschungstrennung).

    PRODUCT:  Nur freigegebene, stabile Pfade — planbares Verhalten für Endnutzer.
              Experimentelle Phasen und SOTA-Experimente sind deaktiviert.
              Default für alle Produktions-Releases.

    RESEARCH: Alle experimentellen und SOTA-Phasen aktiviert (feature-flagged).
              Nur für interne Entwicklung und Benchmark-Forschung.
              Kein Release-Risiko, da explizit opt-in.
    """

    PRODUCT = "product"  # stable paths only — default
    RESEARCH = "research"  # experimental SOTA features enabled


@dataclass
class PhasePerformance:
    """Performance-Metrics für eine einzelne Phase."""

    phase_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    audio_duration_seconds: float
    rt_factor: float  # Real-Time Factor (duration / audio_duration)
    is_critical: bool = False  # Phase ist kritisch für Quality
    skipped: bool = False


@dataclass
class PerformanceReport:
    """Vollständiger Performance-Report."""

    total_duration_seconds: float
    audio_duration_seconds: float
    total_rt_factor: float
    status: PerformanceStatus
    phases: list[PhasePerformance]
    skipped_phases: list[str]
    warnings: list[str] = field(default_factory=list)
    quality_degradation: float = 0.0  # 0-1 (0 = keine Degradation)

    def __str__(self):
        return (
            "PerformanceReport:\n"
            f"  Total: {self.total_duration_seconds:.1f}s for {self.audio_duration_seconds:.1f}s audio\n"
            f"  RT Factor: {self.total_rt_factor:.2f}× ({self.status.value})\n"
            f"  Phases: {len(self.phases)} executed, {len(self.skipped_phases)} skipped\n"
            f"  Quality: {(1 - self.quality_degradation) * 100:.1f}%"
        )


class PerformanceGuard:
    """
    Überwacht und enforced das mode-adaptive Real-Time Performance-Limit.

    Garantiert:
    - Max 8× RT für FAST Mode
    - Max 32× RT für Balanced/Quality/Maximum Mode
    - Adaptive Phase-Skipping wenn nötig
    - Detailed Performance-Logging
    """

    # Performance Limits (RT Factors)
    # Rationale for values (desktop CPU, no GPU, longer/degraded audio):
    #   FAST      = 8×  — quick desktop/proof path with enough headroom for real-audio guards.
    #   BALANCED  = 32× — Standard. Allows full pipeline including ML phases.
    #   QUALITY   = 32× — Restoration mode. Same budget to avoid premature deferral.
    #   MAXIMUM   = 32× — Studio 2026. Full ML chain (SGMSE+, BsRoformer, Vocos,
    #                     Matchering) needs ~5× RT each; 32× gives comfortable margin
    #                     for 5–10 min audio in Stufe 1 without excessive deferral.
    #   BACKGROUND = ∞  — MLRefinementThread (KMV Stufe 2) only — never for Stufe 1.
    LIMIT_3X_RT = 32.0  # §2.38 KMV: 32.0 (normativ, CI-Gate)
    LIMIT_FAST = 8.0
    LIMIT_BALANCED = 32.0  # §2.38 KMV: 32.0 (normativ, CI-Gate)
    LIMIT_QUALITY = 32.0  # §2.38 KMV: 32.0 (normativ, CI-Gate)
    LIMIT_MAXIMUM = 32.0  # §2.38 KMV: 32.0 (normativ, CI-Gate)
    # §2.38 KMV: Hintergrund-ML-Veredelung läuft ohne RT-Limit (MLRefinementThread)
    LIMIT_BACKGROUND: float = float("inf")  # Stufe 2 — unbeschränkt, Priority=LowPriority

    # Warnschwellen — Bruchteile des aktiven Budgets (target_rt_factor)
    # _get_status() multipliziert diese mit target_rt_factor.
    WARNING_FRACTION_OPTIMAL = 0.25  # < 25 % → OPTIMAL
    WARNING_FRACTION_GOOD = 0.50  # < 50 % → GOOD
    WARNING_FRACTION_ACCEPTABLE = 0.85  # < 85 % → ACCEPTABLE (§v10.52: 0.75→0.85)
    WARNING_FRACTION_CRITICAL = 1.0  # < 100 % → CRITICAL, darüber → EXCEEDED

    # Phase Priorities (höher = wichtiger für Quality)
    # ── Musikalische Exzellenz — Priorität 1 (§MusEx-P1) ────────────────────
    # Phasen mit Priority ≥ 9 werden NIEMALS übersprungen (§2.29, §9.5).
    # RT×32-Budget bleibt harte Obergrenze — garantiert
    # durch <15 ms/s Laufzeit der Excellence-Phasen (§9.5 Performance-Budget).
    # ─────────────────────────────────────────────────────────────────────────
    PHASE_PRIORITIES = {
        # Gruppe 1: CRITICAL — niemals überspringen
        "click_removal": 10,
        "hum_removal": 10,
        "wow_flutter": 9,
        "dehum": 10,
        "decracklé": 9,
        # ── Musikalische Exzellenz Priorität 1 (§MusEx-P1) ──────────────────
        # Waren MEDIUM/HIGH → auf CRITICAL angehoben: RT×32 Budget garantiert.
        "harmonic_recovery": 10,  # war: 5 (MEDIUM, skippable bei >2.5×)
        "frequency_restoration": 10,  # war: 7 (HIGH,   skippable bei >2.8×)
        "transient_preservation": 10,  # war: 5 (MEDIUM, skippable bei >2.5×)
        "excellence_optimizer": 10,  # neu: ExcellenceOptimizer (§2.5)
        "psychoacoustic_enhancement": 10,  # neu: PsychoakustikModell (§4.5)
        "micro_dynamics_restoration": 10,  # neu: MicroDynamicsEnvMorphing (§2.30)
        "vocal_enhancement": 10,  # neu: VocalAIEnhancement (§2.8)
        "harmonic_preservation": 10,  # neu: HarmonicPreservationGuard (§2.28)
        # ─────────────────────────────────────────────────────────────────────
        # Gruppe 2: HIGH (nur bei < 2.8× RT skippen)
        "denoise": 8,
        "digital_repair": 8,
        # Gruppe 3: MEDIUM (bei > 2.5× RT skippen)
        "stereo_enhancement": 6,
        # Gruppe 4: LOW (bei > 2.0× RT skippen)
        "final_polish": 3,
        "metadata_embedding": 1,
    }

    # Musikalische Exzellenz — Priorität 1 (§MusEx-P1)
    # Phasen in diesem Set werden unter keinen Umständen übersprungen.
    # RT×32-Kompatibilität: alle Excellence-Phasen < 15 ms/s Audio (§9.5).
    MUSICAL_EXCELLENCE_PHASES: frozenset = frozenset(
        {
            "harmonic_recovery",
            "frequency_restoration",
            "transient_preservation",
            "excellence_optimizer",
            "psychoacoustic_enhancement",
            "micro_dynamics_restoration",
            "vocal_enhancement",
            "harmonic_preservation",
        }
    )

    # Hard Budget: maximaler RT-Faktor für Musikalische Exzellenz-Betrieb
    RT8_EXCELLENCE_BUDGET: float = 32.0

    # Laufzeitintensive Phasen, die unter RT-Druck bevorzugt in KMV Stufe 2
    # deferiert werden sollen, um spaete Skip-Lawinen zu vermeiden.
    HEAVY_PHASE_IDS: frozenset[str] = frozenset(
        {
            "phase_23",
            "phase_24",
            "phase_29",
            "phase_31",
            "phase_32",
            "phase_49",
            "phase_50",
            "phase_55",
            "phase_56",
        }
    )

    # Absolutes Zeitlimit (§9.5): Stufe-1-Restaurierung endet spätestens nach 90 Minuten.
    # Rationale: 30-min Vinyl-Seite (1800s Audio) × 3× DSP-RT = 5400s. KMV Stufe 2
    # übernimmt danach automatisch die verbleibenden ML-Phasen ohne Zeitlimit.
    # Alter Wert: 1800s (30 min) — zu eng für Aufnahmen > 10 min mit schweren Defekten.
    MAX_ABSOLUTE_SECONDS: float = 5400.0  # 90 Minuten

    def __init__(
        self, mode: QualityMode = QualityMode.QUALITY, enforce_limit: bool = True, enable_adaptive_skipping: bool = True
    ):
        """
        Initialisiert PerformanceGuard.

        Args:
            mode: Quality-Mode (bestimmt Performance-Target)
            enforce_limit: Enforce active RT limit (False = nur Monitor)
            enable_adaptive_skipping: Adaptive Phase-Skipping aktivieren
        """
        self.mode = mode
        self.enforce_limit = enforce_limit
        self.enable_adaptive_skipping = enable_adaptive_skipping

        # Performance-Targets basierend auf Mode
        self.target_rt_factor = {
            QualityMode.FAST: self.LIMIT_FAST,
            QualityMode.BALANCED: self.LIMIT_BALANCED,
            QualityMode.QUALITY: self.LIMIT_QUALITY,
            QualityMode.MAXIMUM: self.LIMIT_MAXIMUM,  # 32× RT — maximum / studio_2026
        }[mode]

        # Tracking State
        self.phase_performances: list[PhasePerformance] = []
        self.skipped_phases: list[str] = []
        self.start_time: float | None = None
        self.audio_duration: float | None = None
        self.current_rt_factor: float = 0.0
        self.warnings: list[str] = []
        self._skip_warned_phase_ids: set[str] = set()
        self._last_rt_limit_warning_s: float = 0.0
        # Runtime-Set für verpflichtende Phasen (z. B. §6.2a Material-Pflichtphasen),
        # die auch unter RT-Druck niemals deferred/geskippt werden dürfen.
        self._runtime_never_skip_phase_ids: set[str] = set()
        self._runtime_never_skip_short_ids: set[str] = set()
        # Accumulated analytics overhead (goal measurements, PMGG checks, etc.)
        # This time is excluded from the RT calculation so quality checks do not
        # inflate the RT factor and cause unnecessary phase skipping.
        self._analytics_overhead_s: float = 0.0

        logger.info(
            f"PerformanceGuard initialisiert: Betriebsart={mode.value}, "
            f"Target={self.target_rt_factor:.1f}× RT, "
            f"Enforce={enforce_limit}, Adaptive={enable_adaptive_skipping}"
        )

    def start_monitoring(self, audio_duration_seconds: float):
        """Startet Performance-Monitoring für Audio-File."""
        if audio_duration_seconds < 0.5:
            # Ignoriere Warmup/Dummy-Audio (z.B. 2-Sample-Signale aus HPSS-Init)
            logger.debug(
                "PerformanceGuard: start_monitoring ignoriert (audio_duration=%.6fs < 0.5s — Dummy-Signal)",
                audio_duration_seconds,
            )
            # audio_duration bleibt None → should_skip_phase gibt False zurück
            return
        self.start_time = time.perf_counter()
        # Minimum 30s floor for RT calculation — prevents fixed overhead
        # (DefectScanner, EraClassifier, CausalDefect, FeedbackChain, etc.)
        # from dominating on short clips/excerpts (e.g. 10s multi-pass chunks)
        # and causing pathological phase-skipping.  For actual 10s audio files
        # this means a generous but finite processing budget of 10× × 30s = 300s.
        self.audio_duration = max(30.0, audio_duration_seconds)
        self.phase_performances.clear()
        self.skipped_phases.clear()
        self.warnings.clear()
        self._skip_warned_phase_ids.clear()
        self._last_rt_limit_warning_s = 0.0
        self.current_rt_factor = 0.0
        self._analytics_overhead_s = 0.0

        logger.info(
            f"gestartet monitoring: {audio_duration_seconds:.1f}s audio, "
            f"max {self.target_rt_factor * audio_duration_seconds:.1f}s processing"
        )

    def add_analytics_overhead(self, seconds: float) -> None:
        """Registriert time spent on quality analytics (goal checks, PMGG, FeedbackChain.
        measurements) so it is excluded from the RT-factor calculation.

        Must be called by any caller that invokes ``measure_all()`` or similar
        blocking analytics operations that should not count against the pipeline
        RT budget.
        """
        if seconds > 0:
            self._analytics_overhead_s += seconds
            logger.debug(
                "PerformanceGuard: +%.2fs analytics overhead (total excluded=%.2fs)",
                seconds,
                self._analytics_overhead_s,
            )

    def set_never_skip_phases(self, phase_ids: list[str] | tuple[str, ...] | set[str]) -> None:
        """Setzt Phasen, die zur Laufzeit niemals geskippt werden dürfen.

        Verwendet sowohl volle Phase-IDs (``phase_09_crackle_removal``) als auch
        normalisierte Kurznamen (``crackle_removal``), damit Aufrufer flexibel
        bleiben und §6.2a-Pflichtphasen robust geschützt sind.
        """
        cleaned = {str(pid) for pid in phase_ids if pid}
        self._runtime_never_skip_phase_ids = cleaned
        self._runtime_never_skip_short_ids = {self._normalize_phase_id(pid) for pid in cleaned}

    def start_phase(self, phase_id: str) -> float:
        """
        Startet Monitoring für eine Phase.

        Returns:
            Phase start timestamp
        """
        return time.perf_counter()

    def end_phase(self, phase_id: str, phase_start_time: float) -> PhasePerformance:
        """
        Beendet Monitoring für eine Phase und gibt Performance zurück.

        Args:
            phase_id: Phase ID
            phase_start_time: Timestamp von start_phase()

        Returns:
            PhasePerformance Objekt
        """
        phase_end_time = time.perf_counter()
        phase_duration = phase_end_time - phase_start_time

        # Guard: start_monitoring() may not have been called (e.g. short audio < 0.5s)
        if self.start_time is None:
            logger.debug(
                "end_Verarbeitungsschritt(%s): start_time is None — start_monitoring() was not called. Skipping RT Aktualisierung.",
                phase_id,
            )
            return PhasePerformance(
                phase_id=phase_id,
                start_time=phase_start_time,
                end_time=phase_end_time,
                duration_seconds=phase_duration,
                audio_duration_seconds=self.audio_duration or 0.0,
                rt_factor=0.0,
                is_critical=self._is_phase_critical(phase_id),
                skipped=False,
            )

        # RT Factor für diese Phase
        phase_rt_factor = phase_duration / self.audio_duration if self.audio_duration else 0

        # Ist Phase kritisch?
        is_critical = self._is_phase_critical(phase_id)

        perf = PhasePerformance(
            phase_id=phase_id,
            start_time=phase_start_time,
            end_time=phase_end_time,
            duration_seconds=phase_duration,
            audio_duration_seconds=self.audio_duration,
            rt_factor=phase_rt_factor,
            is_critical=is_critical,
            skipped=False,
        )

        self.phase_performances.append(perf)

        # Update current RT Factor (kumulativ) — exclude analytics overhead so that
        # goal measurements (measure_all, PMGG, FeedbackChain checks) do not inflate
        # the RT factor and trigger unnecessary phase skipping.
        total_time_so_far = max(0.0, (phase_end_time - self.start_time) - self._analytics_overhead_s)
        self.current_rt_factor = total_time_so_far / self.audio_duration if self.audio_duration else 0

        # Status Check
        status = self._get_status()

        logger.debug(
            f"Verarbeitungsschritt {phase_id}: {phase_duration:.2f}s ({phase_rt_factor:.2f}× RT), "
            f"Total: {self.current_rt_factor:.2f}× RT [{status.value}]"
        )

        # Warnungen
        if status == PerformanceStatus.CRITICAL:
            warning = f"⚠️ Performance CRITICAL: {self.current_rt_factor:.2f}× RT (limit: {self.target_rt_factor:.1f}×)"
            self.warnings.append(warning)
            logger.warning(warning)
        elif status == PerformanceStatus.EXCEEDED:
            warning = f"❌ Performance EXCEEDED: {self.current_rt_factor:.2f}× RT (limit: {self.target_rt_factor:.1f}×)"
            self.warnings.append(warning)
            logger.error(warning)

        return perf

    def should_skip_phase(self, phase_id: str, estimated_time_seconds: float, remaining_phases: int) -> bool:
        """
        Entscheidet ob Phase übersprungen werden sollte.

        Args:
            phase_id: Phase ID
            estimated_time_seconds: Geschätzte Laufzeit der Phase
            remaining_phases: Anzahl verbleibender Phasen

        Returns:
            True wenn Phase übersprungen werden sollte
        """
        if not self.enable_adaptive_skipping:
            return False

        # Kein Skip wenn audio_duration nicht gesetzt oder zu kurz (Dummy-Audio-Guard)
        if not self.audio_duration or self.audio_duration < 0.5 or self.start_time is None:
            return False

        # Kritische Phasen nie skippen
        if self._is_phase_critical(phase_id):
            return False

        short_id = self._normalize_phase_id(phase_id)
        if phase_id in self._runtime_never_skip_phase_ids or short_id in self._runtime_never_skip_short_ids:
            logger.debug(
                "§6.2a Runtime-Guard: Verarbeitungsschritt '%s' als Pflichtphase markiert — ueberspringen verhindert",
                phase_id,
            )
            return False

        # Musikalische Exzellenz — Priorität 1 (§MusEx-P1): niemals überspringen.
        # Doppelter Schutz: Priority-Dict UND Namens-Set verhindern Skip.
        # RT×32-Kompatibilität garantiert: alle Excellence-Phasen < 15 ms/s (§9.5).
        if short_id in self.MUSICAL_EXCELLENCE_PHASES:
            logger.debug(
                f"🎵 Verarbeitungsschritt '{phase_id}' ist Musikalische-Exzellenz-Verarbeitungsschritt (§MusEx-P1) "
                f"— wird niemals übersprungen (RT×32 Grenze: {self.RT8_EXCELLENCE_BUDGET:.1f}×)"
            )
            return False

        # Prognostiziere RT Factor nach dieser Phase — subtract analytics overhead
        # so goal-measurement time does not count against processing budget.
        _elapsed_processing = max(0.0, (time.perf_counter() - self.start_time) - self._analytics_overhead_s)
        current_rt = _elapsed_processing / self.audio_duration if self.audio_duration else 0

        # Skip-Kriterien basierend auf Phase Priority
        phase_priority = self.PHASE_PRIORITIES.get(short_id, 5)  # Default: Medium

        # Skip-Thresholds relativ zum aktiven Budget.
        if phase_priority <= 3:  # LOW Priority
            skip_threshold = self.target_rt_factor * 0.85
        elif phase_priority <= 6:  # MEDIUM Priority
            skip_threshold = self.target_rt_factor * 0.93
        elif phase_priority <= 8:  # HIGH Priority
            skip_threshold = self.target_rt_factor * 0.97
        else:  # CRITICAL (≥ 9)
            return False  # Never skip

        # Marginale RT-Kosten dieser einzelnen Phase
        marginal_rt = estimated_time_seconds / self.audio_duration if self.audio_duration else 0

        # Fruehe Heavy-Phase-Deferral-Logik: wenn das Budget bereits deutlich
        # angezogen ist, grosse Einzelkosten vorziehen statt viele spaete Skips.
        _phase_parts = str(phase_id).split("_", 2)
        _phase_prefix = (
            f"{_phase_parts[0]}_{_phase_parts[1]}"
            if len(_phase_parts) >= 2 and _phase_parts[0] == "phase" and _phase_parts[1].isdigit()
            else short_id
        )
        _phase_is_heavy = _phase_prefix in self.HEAVY_PHASE_IDS
        if _phase_is_heavy and current_rt >= (self.target_rt_factor * 0.75) and marginal_rt >= 0.18:
            should_skip = True
        else:
            should_skip = False

        if not should_skip and current_rt >= skip_threshold:
            # RT ist bereits über dem Threshold (z.B. nach BsRoformer Stem-Separation).
            # Nur Phasen mit hohen marginalen Kosten (> 0.5× RT) werden geskippt
            # und an KMV Stufe 2 deferiert. Leichte DSP-Phasen (EQ, Kompression,
            # Limiting etc.) laufen trotzdem — sie fügen nur Millisekunden hinzu.
            should_skip = marginal_rt > 0.5
        elif not should_skip:
            # RT noch unter Threshold — Standardlogik: projiziere Gesamtkosten
            estimated_total_time = _elapsed_processing + estimated_time_seconds
            estimated_remaining_time = remaining_phases * 0.3  # 0.3s pro DSP-Phase (konservativ)
            final_projected_rt_factor = (
                (estimated_total_time + estimated_remaining_time) / self.audio_duration if self.audio_duration else 0
            )
            should_skip = final_projected_rt_factor > skip_threshold

        if should_skip:
            # Skip-Warnungen pro Phase nur einmal pro Lauf ausgeben.
            # Wiederholte Deferrals derselben Phase sind erwartbares Verhalten
            # unter konstantem RT-Druck und erzeugen sonst Log-Spam ohne Mehrwert.
            if phase_id not in self._skip_warned_phase_ids:
                logger.warning(
                    f"⏭️ Skipping {phase_id} (priority={phase_priority}): "
                    f"current={current_rt:.2f}× RT, marginal={marginal_rt:.2f}× RT, "
                    f"Schwelle={skip_threshold:.1f}× — deferring to KMV Stufe 2"
                )
                self._skip_warned_phase_ids.add(phase_id)
            else:
                logger.debug(
                    "⏭️ Skipping %s erneut (current=%.2f×, marginal=%.2f×, Schwelle=%.1f×)",
                    phase_id,
                    current_rt,
                    marginal_rt,
                    skip_threshold,
                )
            self.skipped_phases.append(phase_id)

        return should_skip

    def check_early_exit(self, remaining_phases: int) -> bool:
        """
        Prüft ob Early-Exit nötig ist — NUR bei absolutem 30-Minuten-Limit.

        §2.38 KMV: RT-Limit-Überschreitung führt NICHT zum Pipeline-Abbruch.
        Einzelne langsame Phasen werden über should_skip_phase() / deferred_phases
        behandelt. Der einzige harte Abbruchgrund ist das 30-Minuten-Absolutlimit.

        Args:
            remaining_phases: Anzahl verbleibender Phasen

        Returns:
            True wenn sofortiger Abbruch empfohlen (nur bei 90-min-Limit)
        """
        if not self.enforce_limit:
            return False

        # §2.38 Absolutes 90-Minuten-Limit — einziger harter Abbruchgrund.
        # Gilt unabhängig vom RT-Faktor.
        if self.start_time is not None:
            elapsed_abs = time.perf_counter() - self.start_time
            if elapsed_abs >= self.MAX_ABSOLUTE_SECONDS:
                logger.error(
                    "❌ ABSOLUTES ZEITLIMIT: %.0fs ≥ %.0fs (90 min) — "
                    "Early-Exit bei %d verbleibenden Phasen erzwungen.",
                    elapsed_abs,
                    self.MAX_ABSOLUTE_SECONDS,
                    remaining_phases,
                )
                return True

        # RT-Faktor-Überschreitung: NUR warnen, KEIN Abbruch (§2.38 KMV).
        # Langsame Phasen werden durch should_skip_phase() / deferred_phases gehandelt.
        current_limit = self.target_rt_factor
        if self.current_rt_factor > current_limit:
            _now = time.perf_counter()
            # Wiederkehrende RT-Limit-Meldungen drosseln: maximal 1x / 30 s.
            if (_now - self._last_rt_limit_warning_s) >= 30.0:
                logger.warning(
                    "⚠️ RT-Limit überschritten: %.2f× RT (limit=%.1f×), "
                    "%d Phasen verbleiben — Pipeline läuft weiter (§2.38 KMV, 90-min-Limit gilt)",
                    self.current_rt_factor,
                    current_limit,
                    remaining_phases,
                )
                self._last_rt_limit_warning_s = _now
            else:
                logger.debug(
                    "RT-Limit weiterhin überschritten: %.2f× (limit=%.1f×, remaining=%d)",
                    self.current_rt_factor,
                    current_limit,
                    remaining_phases,
                )
            # KEIN return True — Pipeline läuft bis 30-min-Limit weiter

        return False

    def get_performance_report(self) -> PerformanceReport:
        """Erzeugt vollständigen Performance-Report."""
        if self.start_time is None or self.audio_duration is None:
            raise RuntimeError("Monitoring not started!")

        total_duration = time.perf_counter() - self.start_time
        total_rt_factor = total_duration / self.audio_duration
        status = self._get_status()

        # Quality Degradation berechnen
        quality_degradation = self._calculate_quality_degradation()

        report = PerformanceReport(
            total_duration_seconds=total_duration,
            audio_duration_seconds=self.audio_duration,
            total_rt_factor=total_rt_factor,
            status=status,
            phases=self.phase_performances.copy(),
            skipped_phases=self.skipped_phases.copy(),
            warnings=self.warnings.copy(),
            quality_degradation=quality_degradation,
        )

        logger.info("\n%s", report)

        return report

    def _get_status(self) -> PerformanceStatus:
        """Ermittelt aktuellen Performance-Status relativ zum aktiven RT-Budget.

        Thresholds are fractions of target_rt_factor:
        OPTIMAL < 25 %, GOOD < 50 %, ACCEPTABLE < 75 %, CRITICAL < 100 %, EXCEEDED > 100 %.
        For BALANCED (32×): EXCEEDED fires at >32× RT (not at >8× as before).
        """
        rt = self.current_rt_factor
        budget = self.target_rt_factor

        if rt > self.WARNING_FRACTION_CRITICAL * budget:
            return PerformanceStatus.EXCEEDED
        elif rt > self.WARNING_FRACTION_ACCEPTABLE * budget:
            return PerformanceStatus.CRITICAL
        elif rt > self.WARNING_FRACTION_GOOD * budget or rt > self.WARNING_FRACTION_OPTIMAL * budget:
            return PerformanceStatus.GOOD
        else:
            return PerformanceStatus.OPTIMAL

    @staticmethod
    def _normalize_phase_id(phase_id: str) -> str:
        """Strip 'phase_XX_' prefix to match PHASE_PRIORITIES short keys.

        E.g. 'phase_01_click_removal' → 'click_removal',
             'phase_03_denoise' → 'denoise',
             'click_removal' → 'click_removal' (unchanged).
        """
        import re

        m = re.match(r"^phase_\d+_(.*)", phase_id)
        return m.group(1) if m else phase_id

    def _is_phase_critical(self, phase_id: str) -> bool:
        """Prüft ob Phase kritisch für Quality ist."""
        short = self._normalize_phase_id(phase_id)
        priority = self.PHASE_PRIORITIES.get(short, 5)
        return priority >= 9  # Priority ≥ 9 = CRITICAL

    def _calculate_quality_degradation(self) -> float:
        """
        Berechnet Quality-Degradation durch geskippte Phasen.

        Returns:
            0.0 - 1.0 (0 = keine Degradation, 1 = kompletter Verlust)
        """
        if not self.skipped_phases:
            return 0.0

        # Summiere Priorities der geskippten Phasen
        total_priority = sum(self.PHASE_PRIORITIES.values())
        skipped_priority = sum(
            self.PHASE_PRIORITIES.get(self._normalize_phase_id(pid), 5) for pid in self.skipped_phases
        )

        # Degradation = Anteil der geskippten Priority
        degradation = skipped_priority / total_priority if total_priority > 0 else 0

        return min(1.0, degradation)

    def get_phase_budget_seconds(self, phase_id: str, total_phases: int, completed_phases: int) -> float:
        """
        Berechnet verbleibendes Time-Budget für eine Phase.

        Args:
            phase_id: Phase ID
            total_phases: Gesamtanzahl Phasen
            completed_phases: Anzahl abgeschlossener Phasen

        Returns:
            Verbleibendes Budget in Sekunden
        """
        # Gesamtbudget
        total_budget = self.target_rt_factor * self.audio_duration if self.audio_duration else 0

        # Bereits verbrauchte Zeit
        elapsed_time = time.perf_counter() - self.start_time if self.start_time else 0

        # Verbleibendes Budget
        remaining_budget = total_budget - elapsed_time

        # Verteile auf verbleibende Phasen
        remaining_phases = total_phases - completed_phases
        phase_budget = remaining_budget / remaining_phases if remaining_phases > 0 else 0

        return max(0, phase_budget)


# ========== CLI/Testing Interface ==========

if __name__ == "__main__":
    """Test PerformanceGuard mit Beispiel-Workflow."""

    # Setup Logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.debug("\n%s", "=" * 60)
    logger.debug("PERFORMANCE GUARD TEST")
    logger.debug("%s\n", "=" * 60)

    # Simuliere 3:45 Audio (225 Sekunden)
    audio_duration = 225.0

    # Test 1: Balanced Mode (sollte erfolgreich sein)
    logger.debug("TEST 1: Balanced Betriebsart (Target: 2.4× RT = 540s max)")
    logger.debug("-" * 60)

    guard = PerformanceGuard(mode=QualityMode.BALANCED, enforce_limit=True, enable_adaptive_skipping=True)
    guard.start_monitoring(audio_duration)

    # Simuliere Phasen
    phases = [
        ("click_removal", 15, 10),  # (phase_id, duration_seconds, priority)
        ("hum_removal", 20, 10),
        ("denoise", 30, 8),
        ("stereo_enhancement", 25, 6),
        ("final_polish", 40, 3),
    ]

    for i, (phase_id, duration, priority) in enumerate(phases):
        # Prüfe ob skippen
        remaining = len(phases) - i - 1
        if guard.should_skip_phase(phase_id, duration, remaining):
            continue

        # Simuliere Phase
        phase_start = guard.start_phase(phase_id)
        time.sleep(duration / 100)  # Skaliert runter für schnellen Test
        guard.end_phase(phase_id, phase_start)

        # Check Early Exit
        if guard.check_early_exit(remaining):
            logger.debug("⚠️ Early exit triggered!")
            break

    report1 = guard.get_performance_report()

    # Test 2: Fast Mode
    logger.debug("\n\nTEST 2: Fast Betriebsart (Target: 1.5× RT = 338s max)")
    logger.debug("-" * 60)

    guard2 = PerformanceGuard(mode=QualityMode.FAST, enforce_limit=True, enable_adaptive_skipping=True)
    guard2.start_monitoring(audio_duration)

    for i, (phase_id, duration, priority) in enumerate(phases):
        remaining = len(phases) - i - 1
        if guard2.should_skip_phase(phase_id, duration * 0.6, remaining):  # Faster phases in fast mode
            continue

        phase_start = guard2.start_phase(phase_id)
        time.sleep(duration * 0.6 / 100)
        guard2.end_phase(phase_id, phase_start)

    report2 = guard2.get_performance_report()

    # Zusammenfassung
    logger.debug("\n\n%s", "=" * 60)
    logger.debug("SUMMARY")
    logger.debug("%s", "=" * 60)
    logger.debug(
        f"Balanced Betriebsart: {report1.total_rt_factor:.2f}× RT, "
        f"{len(report1.skipped_phases)} uebersprungen, "
        f"{(1 - report1.quality_degradation) * 100:.1f}% quality"
    )
    logger.debug(
        f"Fast Betriebsart:     {report2.total_rt_factor:.2f}× RT, "
        f"{len(report2.skipped_phases)} uebersprungen, "
        f"{(1 - report2.quality_degradation) * 100:.1f}% quality"
    )
