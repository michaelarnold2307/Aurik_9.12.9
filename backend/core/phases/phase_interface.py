"""
Phase Interface — Basisklassen für alle Aurik-Verarbeitungsphasen (§7.1).

Definiert:
  - PhaseCategory  (Enum): Kategorisierung der Phasen
  - PhaseMetadata  (dataclass): Beschreibende Metadaten einer Phase
  - PhaseResult    (dataclass): Ausgabe einer Phase
    - PhaseInterface (ABC): Abstrakte Basisklasse für alle 64 Phasen
  - create_phase_result(): Convenience-Factory

Aurik 10.0.0 — Kanonische Implementierung (core/phases/phase_interface.py)
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhaseCategory — Kategorisiert jede Phase nach ihrer Funktion (§7.1)
# ---------------------------------------------------------------------------
class PhaseCategory(Enum):
    """Funktionale Kategorie einer Restaurierungsphase."""

    DEFECT_REMOVAL = auto()  # Klicks, Rauschen, Brumm, Crackle …
    FREQUENCY = auto()  # EQ, Bandbreiten-Erweiterung, Rumble …
    RESTORATION = auto()  # Dropout, Inpainting, Spektralreparatur …
    DYNAMICS = auto()  # Kompression, Expansion, Limiting …
    ENHANCEMENT = auto()  # Exciter, Gesang, Instrumente, Air …
    STEREO = auto()  # Stereo-Balance, Mid/Side, Breite …
    METADATA = auto()  # Normalisierung, Format-Optimierung …


# ---------------------------------------------------------------------------
# PhaseMode — Deklariert in welchem Modus eine Phase laufen darf (§v10.70)
# ---------------------------------------------------------------------------
class PhaseMode(Enum):
    """Modus-Zugehörigkeit einer Phase — architektonische Garantie.

    RESTORATION_ONLY:  Läuft NUR im Restoration-Mode.
                       Defekt-Entfernung, Geometrie-Korrektur, Reparatur.
                       Darf NICHTS hinzufügen was nicht vor der Beschädigung da war.

    STUDIO_ONLY:       Läuft NUR im Studio-2026-Mode.
                       Kreative Enhancement, Mastering, Exciter, Sub-Bass-Synthese.
                       Darf Frequenzen, Raum, Dynamik NEU gestalten.

    BOTH:              Läuft in BEIDEN Modi, aber mit modus-spezifischer Konfiguration.
                       Restoration → konservativ (nur Reparatur).
                       Studio 2026 → voll (kreative Gestaltung).
    """

    RESTORATION_ONLY = "restoration_only"
    STUDIO_ONLY = "studio_only"
    BOTH = "both"


# ---------------------------------------------------------------------------
# PhaseMetadata — Beschreibende Informationen einer Phase
# ---------------------------------------------------------------------------
@dataclass
class PhaseMetadata:
    """Metadaten zu einer Verarbeitungsphase (unveränderlich nach Erstellung)."""

    phase_id: str  # z.B. "phase_01_click_removal"
    name: str  # Anzeigename
    category: PhaseCategory  # Funktionale Kategorie
    priority: int  # 1 (niedrig) – 10 (hoch)
    version: str = "1.0.0"
    dependencies: list[str] = field(default_factory=list)
    estimated_time_factor: float = 0.05  # Anteil Verarbeitungszeit (0–1)
    memory_requirement_mb: int = 64
    is_cpu_intensive: bool = True
    is_io_intensive: bool = False
    quality_impact: float = 0.85  # Erwarteter Qualitätsbeitrag (0–1)
    description: str = ""
    defect_types: list[str] = field(default_factory=list)
    musical_goals: list[str] = field(default_factory=list)
    phase_mode: PhaseMode = PhaseMode.RESTORATION_ONLY  # §v10.70 Modus-Zugehörigkeit

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert phase metadata to a plain dictionary."""
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "category": self.category.name,
            "priority": self.priority,
            "version": self.version,
            "dependencies": self.dependencies,
            "estimated_time_factor": self.estimated_time_factor,
            "memory_requirement_mb": self.memory_requirement_mb,
            "is_cpu_intensive": self.is_cpu_intensive,
            "is_io_intensive": self.is_io_intensive,
            "quality_impact": self.quality_impact,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# PhaseResult — Ausgabe einer Verarbeitungsphase
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    """Ergebnis einer Phase-Verarbeitung — immer NaN/Inf-frei und geclippt.

    §2.59: time_range ermöglicht chirurgische Verarbeitung.
    Wenn gesetzt, wurde die Phase NUR auf diesen Zeitbereich angewendet.
    None = Phase hat gesamtes Audio verarbeitet (global).

    §v10.18: resolved_defects erlaubt Phasen, dem Pipeline-Kontext zu melden,
    dass ein Defekt behoben wurde. Nachfolgende Phasen können dadurch ihre
    Strategie anpassen (z.B. Phase 23 drosselt Stärke, wenn Phase 07 bereits
    Clipping entfernt hat).
    """

    audio: np.ndarray  # Verarbeitetes Audio (float32, [-1,1])
    modifications: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metrics ist ein echtes Feld als Alias fuer metadata-Inhalte.
    # Wird es beim Konstruktor-Aufruf uebergeben, landet der Inhalt
    # in metadata (via __post_init__).
    time_range: tuple[float, float] | None = None  # §2.59: (start_s, end_s) oder None=global
    metrics: dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    ml_used: bool = False
    quality_estimate: float = 1.0  # 0–1
    success: bool = True  # True = Phase erfolgreich abgeschlossen
    # §v10.15: 2s-Ausschnitt VOR der Phase für A/B-Vergleich im UI
    audio_before_snippet: np.ndarray | None = None
    # §v10.18: Defekte, die diese Phase behoben hat (DefectType → neue Severity)
    # Wird vom UV3/Denker konsumiert, um den Defect-Context für Folgephasen zu aktualisieren
    resolved_defects: dict[str, float] = field(default_factory=dict)
    # §0a Passthrough-Semantik: Zero-Strength-/Skip-Pfade dürfen das Signal bit-identisch
    # zurückgeben (kein §v10.62-Soft-Clip). Nur für echte Passthrough-Ergebnisse setzen.
    _skip_soft_clip: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # Sicherheits-Invarianten: NaN/Inf bereinigen, soft-clipping (§v10.62)
        if not isinstance(self.audio, np.ndarray):
            # §v10.95: Tuple→ndarray-Normalisierung. Phasen können versehentlich
            # (audio_ndarray, metadata_dict) als self.audio setzen.
            # np.asarray auf inhomogene Tupel crasht mit "setting an array element
            # with a sequence". Daher erstes ndarray im Tupel extrahieren.
            if isinstance(self.audio, (tuple, list)):
                _candidates = [x for x in self.audio if isinstance(x, np.ndarray)]
                self.audio = _candidates[0] if _candidates else np.zeros(1, dtype=np.float32)
            else:
                self.audio = np.asarray(self.audio, dtype=np.float32)
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        if not self._skip_soft_clip:
            # §v10.62: apply_soft_clip statt Hard-Clamp — verhindert hörbare
            # Rechteck-Clipping-Artefakte in allen 68 Phasen.
            try:
                from backend.core.audio_utils import apply_soft_clip

                self.audio = apply_soft_clip(self.audio, ceiling=1.0)
            except ImportError:
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
                self.audio = np.clip(self.audio, -1.0, 1.0)  # Fallback
        if self.audio.dtype != np.float32:
            self.audio = self.audio.astype(np.float32)
        # metrics und metadata synchronisieren: metrics erhaelt Vorrang
        # wenn explizit gesetzt, sonst wird metrics mit metadata befüllt.
        if self.metrics and not self.metadata:
            self.metadata = self.metrics
        elif self.metadata and not self.metrics:
            self.metrics = self.metadata

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert the phase result payload to a plain dictionary."""
        return {
            "modifications": self.modifications,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "execution_time_seconds": self.execution_time_seconds,
            "ml_used": self.ml_used,
            "quality_estimate": self.quality_estimate,
        }


# ---------------------------------------------------------------------------
# create_phase_result — Convenience-Factory (NaN/Inf-sicher)
# ---------------------------------------------------------------------------
def create_phase_result(
    audio: np.ndarray,
    modifications: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_time_seconds: float = 0.0,
    ml_used: bool = False,
    quality_estimate: float = 1.0,
    phase_id: str = "",
    phase_name: str = "",
    resolved_defects: dict[str, float] | None = None,
) -> PhaseResult:
    """Erzeugt ein NaN/Inf-bereinigtes PhaseResult mit Fazit-Log.

    Args:
        audio:                  Verarbeitetes Audio-Signal (float32)
        modifications:          Dict mit Phase-spezifischen Änderungen
        warnings:               Liste von Warnungen
        metadata:               Zusätzliche Metadaten
        execution_time_seconds: Verarbeitungszeit in Sekunden
        ml_used:                Ob ML-Modell verwendet wurde
        quality_estimate:       Qualitätsschätzung 0–1
        phase_id:               Phasen-Nummer (z.B. "03", "09")
        phase_name:             Menschlicher Name (z.B. "Entrauschen")
        resolved_defects:       §v10.18: {DefectType: residual_severity} nach Reparatur

    Returns:
        PhaseResult mit bereinigtem Audio und Fazit-Log
    """
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # §v10.62: apply_soft_clip statt Hard-Clamp
    try:
        from backend.core.audio_utils import apply_soft_clip

        audio = apply_soft_clip(audio, ceiling=1.0)
    except ImportError:
        audio = np.clip(audio, -1.0, 1.0)

    # ── Fazit-Log ────────────────────────────────────────────────────
    if phase_id and phase_name:
        try:
            from backend.core.phase_fazit import log_phase_fazit

            _score = float(np.clip(quality_estimate * 10.0, 0.0, 10.0))
            _mods = modifications or {}
            # Build human-readable summary from modifications dict
            _summary_parts = []
            for k, v in _mods.items():
                if isinstance(v, (int, float)):
                    _summary_parts.append(f"{k}={v:.1f}" if isinstance(v, float) and v == v else f"{k}={v}")
                elif isinstance(v, str) and len(v) < 40:
                    _summary_parts.append(f"{k}={v}")
            _summary = ", ".join(_summary_parts[:4]) if _summary_parts else "Phase abgeschlossen"

            _details = {}
            for k, v in list(_mods.items())[:3]:
                if isinstance(v, (int, float, str)):
                    _details[str(k)] = str(v)
            if ml_used:
                _details["ML"] = "ja"

            log_phase_fazit(
                phase=phase_id,
                name=phase_name,
                score=_score,
                summary=_summary,
                details=_details if _details else None,
            )
        except Exception as _e:
            logger.debug(
                "Verarbeitungsschritt_Verarbeitungsschritt_interface: unkritisch exception: %s", _e
            )  # Fazit-Log ist optional, darf Phase nicht blockieren

    return PhaseResult(
        audio=audio,
        modifications=modifications or {},
        warnings=warnings or [],
        metadata=metadata or {},
        execution_time_seconds=execution_time_seconds,
        ml_used=ml_used,
        quality_estimate=float(np.clip(quality_estimate, 0.0, 1.0)),
        resolved_defects=resolved_defects or {},
    )


# ---------------------------------------------------------------------------
# PhaseInterface — Abstrakte Basisklasse für alle 64 Phasen (§7.1)
# ---------------------------------------------------------------------------
class PhaseInterface(abc.ABC):
    """Abstrakte Basisklasse für alle Aurik-Verarbeitungsphasen.

    Jede Phase implementiert:
        get_metadata() -> PhaseMetadata
        process(audio, sample_rate, material_type, **kwargs) -> PhaseResult

    Invarianten (§3.1):
        - Ausgang immer float32 im Bereich [-1, 1]
        - Kein NaN/Inf in Ausgang
        - sample_rate == 48000 wird vorausgesetzt
        - Kein direktes Netzwerk-I/O
    """

    def __init__(self, sample_rate: int = 48000, **_kwargs) -> None:
        """Basisinitialisierung für alle Phasen.

        Args:
            sample_rate: Sample-Rate (Standard 48000 Hz). Wird von Subklassen
                         via super().__init__(sample_rate) weitergegeben.
            **_kwargs:   Zusätzliche Konfigurations-Parameter (werden ignoriert,
                         aber akzeptiert, damit Subklassen **kwargs weiterreichen).
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._name_override: str | None = None
        # sample_rate für Subklassen verfügbar machen (ohne Pflicht es zu nutzen)
        self._sample_rate: int = sample_rate

    @property
    def sample_rate(self) -> int:
        """Sample-Rate dieser Phase (Standard 48000 Hz).

        Property für Rückwärtskompatibilität: Phasen die ``self.sample_rate``
        nutzen, funktionieren ohne Änderung. Intern gespeichert als ``_sample_rate``.
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        """Setzt sample_rate (ermöglicht phase_02-Muster: self.sample_rate = sr)."""
        self._sample_rate = value

    @property
    def metadata(self) -> PhaseMetadata:
        """Gibt Phasen-Metadaten als Attribut zurück (delegiert an get_metadata())."""
        return self.get_metadata()

    @abc.abstractmethod
    def get_metadata(self) -> PhaseMetadata:
        """Gibt beschreibende Metadaten dieser Phase zurück."""

    @abc.abstractmethod
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        """Verarbeitet Audio und gibt PhaseResult zurück.

        Args:
            audio:        float32 np.ndarray, mono [N] oder stereo [2, N] / [N, 2]
            sample_rate:  Sample-Rate in Hz (intern immer 48000)
            material_type: Träger-Material z.B. "tape", "vinyl", "unknown"
            **kwargs:     Phase-spezifische Parameter

        Returns:
            PhaseResult mit bereinigtem Audio, NaN/Inf-frei, geclippt auf [-1, 1]
        """

    # ------------------------------------------------------------------
    # Konkrete Hilfsmethoden (von allen Phasen geerbt)
    # ------------------------------------------------------------------

    @staticmethod
    def surgical_dispatch(
        phase: PhaseInterface,
        audio: np.ndarray,
        sample_rate: int,
        material_type: str,
        time_ranges: list[tuple[float, float]],
        context_ms: float = 20.0,
        crossfade_ms: float = 5.0,
        **kwargs,
    ) -> np.ndarray:
        """§2.59.14: Führt eine Phase chirurgisch aus — nur auf Zeitfenstern.

        Extrahiert jedes Zeitfenster mit Kontext, ruft phase.process()
        auf das Fenster auf, und blended das Ergebnis via Cosine-Crossfade
        nahtlos zurück ins Gesamtsignal.

        Args:
            phase: Die auszuführende Phase (muss PhaseInterface sein)
            audio: Vollständiges Audio (channels, samples) oder (samples,)
            sample_rate: Sample-Rate in Hz
            material_type: Material-Typ für die Phase
            time_ranges: Liste von (start_s, end_s) Zeitfenstern
            context_ms: Kontext vor/nach jedem Fenster
            crossfade_ms: Dauer des Crossfades an den Rändern
            **kwargs: Werden an phase.process() weitergereicht

        Returns:
            Audio mit chirurgisch reparierten Zonen (gleiche Shape wie Input)
        """
        import numpy as np

        was_mono = audio.ndim == 1
        if was_mono:
            audio = audio.reshape(1, -1)
        result = audio.copy()
        total_samples = audio.shape[1]
        ctx_samples = int(context_ms * sample_rate / 1000)
        fade_samples = int(crossfade_ms * sample_rate / 1000)
        repaired = 0
        skipped = 0

        for start_s, end_s in sorted(time_ranges, key=lambda x: x[0]):
            s0 = max(0, int(start_s * sample_rate) - ctx_samples)
            s1 = min(total_samples, int(end_s * sample_rate) + ctx_samples)
            if s1 - s0 < 32:  # Minimum für DSP
                skipped += 1
                continue

            segment = audio[:, s0:s1].copy()
            original = segment.copy()

            try:
                proc_result = phase.process(segment, sample_rate, material_type, **kwargs)
                if isinstance(proc_result, np.ndarray):
                    segment = proc_result
                elif hasattr(proc_result, "audio"):
                    segment = proc_result.audio
            except Exception:
                skipped += 1
                continue

            # Safety-Clamp: ≤2× Original-Amplitude
            import numpy as _np

            abs_orig = _np.maximum(_np.abs(original), 1e-10)
            limit = abs_orig * 2.0
            _np.clip(segment, -limit, limit, out=segment)

            # Cosine-Crossfade an den Rändern
            if segment.shape[1] >= fade_samples * 2:
                ramp_in = 0.5 * (1 - _np.cos(_np.pi * _np.arange(fade_samples) / fade_samples))
                ramp_out = ramp_in[::-1]
                for ch in range(segment.shape[0]):
                    segment[ch, :fade_samples] = (
                        original[ch, :fade_samples] * (1 - ramp_in) + segment[ch, :fade_samples] * ramp_in
                    )
                    segment[ch, -fade_samples:] = (
                        original[ch, -fade_samples:] * (1 - ramp_out) + segment[ch, -fade_samples:] * ramp_out
                    )

            result[:, s0:s1] = segment
            repaired += 1

        if was_mono:
            result = result[0]
        return cast(np.ndarray, result.astype(np.float32))

    def _safe_process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        """Wrapper mit Timing, Exception-Handling, NaN-Guard, ComfortGuard und VocalQualityGate.

        §Rolls-Royce-Phantom: Jede Phase wird automatisch auf Hörkomfort und
        Gesangsqualität geprüft. Kein manuelles Eingreifen nötig.
        """
        assert sample_rate == 48000, f"Interne SR muss 48000 Hz sein, erhalten: {sample_rate}"

        # ── BreathPreserver: Atem-Erhalt vor NR-Phasen ─────────────────
        _breath_mask = None
        _is_nr = any(kw in self.get_metadata().phase_id for kw in ("denoise", "hiss", "noise", "nr", "03", "29"))
        if _is_nr:
            try:
                from backend.core.breath_preserver import protect_breath

                audio, _breath_mask = protect_breath(audio, sample_rate)
            except Exception as _bp_exc:
                self._logger.debug("BreathPreserver protect skipped: %s", _bp_exc)

        t0 = time.monotonic()
        # §v10.118 FeedbackChain-Awareness: Phasen erkennen zweiten Durchlauf.
        # Wenn _fc_active im restoration_context gesetzt ist, wird das Flag
        # an die Phase weitergereicht. Additive Phasen (07, 23, 39) können
        # daraufhin ihre Stärke drosseln.
        _is_fc_pass = bool(kwargs.get("_feedback_chain_pass", False))
        if not _is_fc_pass:
            try:
                _rest_ctx = kwargs.get("_restoration_context")
                if isinstance(_rest_ctx, dict) and _rest_ctx.get("_fc_active", False):
                    kwargs["_feedback_chain_pass"] = True
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        try:
            result = self.process(audio, sample_rate, material_type, **kwargs)
        except Exception as exc:
            self._logger.warning(
                "Phase %s fehlgeschlagen (%s) — Pass-Through",
                self.get_metadata().phase_id,
                exc,
            )
            result = create_phase_result(
                audio=audio,
                warnings=[f"Phase fehlgeschlagen: {exc}"],
                quality_estimate=0.95,
            )

        # ── BreathPreserver: Atem-Natürlichkeit wiederherstellen ────────
        if _breath_mask is not None:
            try:
                from backend.core.breath_preserver import restore_breath

                result.audio = restore_breath(result.audio, _breath_mask, audio)
            except Exception as _bp_exc:
                self._logger.debug("BreathPreserver restore skipped: %s", _bp_exc)

        # ── ComfortGuard: Automatische Hörmüdungs-Prävention ──────────
        try:
            from backend.core.comfort_guard import apply_comfort_guard

            result.audio = apply_comfort_guard(result.audio, sample_rate)
        except Exception as _cg_exc:
            self._logger.debug("ComfortGuard skipped: %s", _cg_exc)

        # ═══════════════════════════════════════════════════════════════
        # §v10.115 Universal Phase Safety Wrapper — hebt ALLE 65+ Phasen
        # auf das gleiche SOTA-Sicherheitsniveau. Drei systemische Guards:
        #
        #  1. RMS-Preservation-Guard: Rollback bei >30 dB Pegelabfall
        #  2. §V22 Transient-Shift-Guard: Erkennt destruktive Transient-
        #     Verschiebungen in additiven/Enhancement-Phasen
        #  3. §2.46e Spectral-Novelty-Guard: Verhindert HF-Halluzinationen
        #     in Synthese-/Spektral-Phasen
        #
        # Alle Guards sind non-blocking: Fehler → Debug-Log, kein Crash.
        # Alle Guards enrichieren result.metadata für Audit/Tracing.
        # ═══════════════════════════════════════════════════════════════
        phase_id = self.get_metadata().phase_id
        _input_rms = None
        try:
            _input_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2)) + 1e-12)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        if _input_rms is not None:
            # ── Guard 1: Universal RMS-Preservation ─────────────────
            try:
                _output_rms = float(np.sqrt(np.mean(np.asarray(result.audio, dtype=np.float32) ** 2)) + 1e-12)
                _rms_drop_db = float(20.0 * np.log10(_output_rms / _input_rms)) if _input_rms > 1e-12 else 0.0
                result.metadata["rms_drop_db"] = round(float(min(0.0, _rms_drop_db)), 3)
                if _rms_drop_db < -30.0:
                    self._logger.warning(
                        "§v10.115 RMS-Guard: %s RMS-Drop %.1f dB → Rollback auf Eingangs-Audio",
                        phase_id,
                        _rms_drop_db,
                    )
                    result.audio = np.asarray(audio, dtype=np.float32)
                    result.warnings.append(f"RMS-Guard: Pegelabfall {_rms_drop_db:.1f} dB rückgängig gemacht")
                    result.metadata["rms_guard_rollback"] = True
            except Exception as _rms_exc:
                self._logger.debug("§v10.115 RMS-Guard non-blocking: %s", _rms_exc)

            # ── Guard 2: §V22 Transient-Shift-Detection ─────────────
            # Nur für additive/Enhancement-Phasen, die Transienten
            # verändern könnten (Harmonik, Exciter, Air-Band, Bass etc.).
            _phase_cat = self.get_metadata().category.name if hasattr(self, "get_metadata") else ""
            _is_additive = _phase_cat in ("ENHANCEMENT", "RESTORATION") or any(
                kw in phase_id
                for kw in (
                    "harmonic",
                    "exciter",
                    "air_band",
                    "bass_enhance",
                    "presence",
                    "transient",
                    "spectral",
                    "frequency",
                    "drums",
                    "guitar",
                    "brass",
                    "piano",
                    "vocal",
                    "saturation",
                    "spatial",
                    "stereo_enhance",
                )
            )
            if _is_additive:
                try:
                    from backend.core.dsp.transient_guard import detect_transient_shifts

                    _ts_result = detect_transient_shifts(
                        np.asarray(audio, dtype=np.float32),
                        np.asarray(result.audio, dtype=np.float32),
                        sample_rate,
                    )
                    if _ts_result is not None and hasattr(_ts_result, "onset_shift_ms"):
                        _shift_ms = float(_ts_result.onset_shift_ms)
                        result.metadata["onset_shift_ms"] = round(_shift_ms, 2)
                        if _shift_ms > 5.0:
                            self._logger.warning(
                                "§V22 Transient-Guard: %s onset_shift=%.1f ms > 5 ms → Detektion",
                                phase_id,
                                _shift_ms,
                            )
                            result.warnings.append(f"Transient-Guard: onset_shift={_shift_ms:.1f} ms detektiert")
                except Exception as _ts_exc:
                    self._logger.debug("§V22 Transient-Guard non-blocking: %s", _ts_exc)

            # ── Guard 3: §2.46e Spectral-Novelty-Hallucination-Guard ─
            # Nur für Synthese-/Spektral-Phasen, die neuen Spektral-
            # Inhalt erzeugen können (Harmonik, Inpainting, Exciter).
            _is_synthesis = any(
                kw in phase_id
                for kw in (
                    "harmonic",
                    "spectral_repair",
                    "inpainting",
                    "exciter",
                    "frequency_restoration",
                    "air_band",
                    "diffusion",
                    "band_gap",
                    "dropout",
                )
            )
            if _is_synthesis:
                try:
                    _n_fft = min(2048, audio.shape[-1] // 4 if audio.ndim == 1 else audio.shape[-1] // 4)
                    if _n_fft >= 64:
                        _mono_in = audio.mean(axis=0) if audio.ndim == 2 else audio
                        _mono_out = result.audio.mean(axis=0) if result.audio.ndim == 2 else result.audio
                        _spec_in = np.abs(np.fft.rfft(_mono_in[: _n_fft * 4]))
                        _spec_out = np.abs(np.fft.rfft(_mono_out[: _n_fft * 4]))
                        _spec_in_norm = _spec_in / (np.max(_spec_in) + 1e-12)
                        _spec_out_norm = _spec_out / (np.max(_spec_out) + 1e-12)
                        _novelty = float(np.mean(np.abs(_spec_out_norm - _spec_in_norm)))
                        result.metadata["spectral_novelty"] = round(_novelty, 4)
                        if _novelty > 0.15:
                            self._logger.warning(
                                "§2.46e Hallucination-Guard: %s spectral_novelty=%.3f > 0.15",
                                phase_id,
                                _novelty,
                            )
                            result.warnings.append(f"Hallucination-Guard: spectral_novelty={_novelty:.3f}")
                except Exception as _hg_exc:
                    self._logger.debug("§2.46e Hallucination-Guard non-blocking: %s", _hg_exc)

            # ── Guard 4: §v10.117 Universal Formant-Guard ─────────────
            # Leichtgewichtige Formant-Stabilitäts-Prüfung für ALLE Phasen.
            # Nutzt spektrale Band-Vektor-Korrelation (10 Bänder, 300–3500 Hz)
            # um Formant-Verschiebungen zu erkennen — die häufigste Ursache
            # für unnatürlich klingenden Gesang nach DSP-Verarbeitung.
            # Laufzeit: < 0.5 ms, non-blocking.
            try:
                _n_guard4 = min(len(audio), 8192)
                if _n_guard4 >= 256:
                    _mono_pre4 = (
                        np.asarray(audio, dtype=np.float32).mean(axis=0)
                        if audio.ndim == 2
                        else np.asarray(audio, dtype=np.float32)
                    )[:_n_guard4]
                    _mono_post4 = (
                        np.asarray(result.audio, dtype=np.float32).mean(axis=0)
                        if result.audio.ndim == 2
                        else np.asarray(result.audio, dtype=np.float32)
                    )[:_n_guard4]
                    # 10-Band spectral envelope 300–3500 Hz (Formant-Bereich)
                    _bands_hz4 = np.logspace(np.log10(300), np.log10(3500), 11)
                    _bands_bin4 = np.round(_bands_hz4 * _n_guard4 / sample_rate).astype(int)
                    _bands_bin4 = np.clip(_bands_bin4, 1, _n_guard4 // 2 - 1)
                    _spec_pre4 = np.abs(np.fft.rfft(_mono_pre4))
                    _spec_post4 = np.abs(np.fft.rfft(_mono_post4))
                    _env_pre4 = np.array(
                        [float(np.mean(_spec_pre4[_bands_bin4[i] : _bands_bin4[i + 1]])) for i in range(10)]
                    )
                    _env_post4 = np.array(
                        [float(np.mean(_spec_post4[_bands_bin4[i] : _bands_bin4[i + 1]])) for i in range(10)]
                    )
                    _env_pre4 = _env_pre4 / (np.max(_env_pre4) + 1e-12)
                    _env_post4 = _env_post4 / (np.max(_env_post4) + 1e-12)
                    _formant_corr4 = float(
                        np.dot(_env_pre4, _env_post4) / (np.linalg.norm(_env_pre4) * np.linalg.norm(_env_post4) + 1e-12)
                    )
                    result.metadata["formant_stability"] = round(_formant_corr4, 4)
                    if _formant_corr4 < 0.85:
                        self._logger.warning(
                            "§v10.117 Formant-Guard: %s formant_stability=%.3f < 0.85 → mögliche Gesangsdegradation",
                            phase_id,
                            _formant_corr4,
                        )
                        result.warnings.append(
                            f"Formant-Guard: spektrale Hüllkurve verschoben (corr={_formant_corr4:.3f})"
                        )
            except Exception as _fg_exc:
                self._logger.debug("§v10.117 Formant-Guard non-blocking: %s", _fg_exc)

        # ── §v10.300 Universal NaN/Inf Final Guard ────────────────────────
        # Letzte Verteidigungslinie: Stellt sicher dass KEIN NaN/Inf das System
        # verlässt. 50/68 Phasen haben nur nan_to_num ohne isfinite-Warnung —
        # dieser Guard schließt die Lücke für ALLE Phasen.
        try:
            _post_audio = np.asarray(result.audio, dtype=np.float32)
            if not np.isfinite(_post_audio).all():
                _n_nan = int(np.sum(np.isnan(_post_audio)))
                _n_inf = int(np.sum(np.isinf(_post_audio)))
                self._logger.warning(
                    "§v10.300 NaN/Inf-Guard: %s Output enthält %d NaN + %d Inf → bereinigt",
                    phase_id,
                    _n_nan,
                    _n_inf,
                )
                result.audio = np.nan_to_num(_post_audio, nan=0.0, posinf=0.0, neginf=0.0)
                result.warnings.append(f"NaN/Inf-Guard: {_n_nan} NaN + {_n_inf} Inf bereinigt")
        except Exception as _nan_exc:
            self._logger.debug("§v10.300 NaN/Inf-Guard non-blocking: %s", _nan_exc)

        # ── Guard 6: §G144/§G145 MUSHRA-Proxy Per-Phase Check + Rollback ─
        # SOTA perzeptueller Guard: Vergleich des Pre-Phase- mit dem
        # Post-Phase-Audio via leichtgewichtigem MERT-Embedding-Vergleich
        # (MERT verfügbar) oder Bark-Band-Heuristik (Fallback).
        # §G144: JEDE Phase MUSS nach Ausführung den MUSHRAProxy konsultieren.
        # §G145: delta ≤ 0 → Phase MUSS zurückgerollt werden.
        # Keine Ausnahme, kein "war nur eine kleine Verschlechterung".
        try:
            from backend.core.mushra_proxy import get_mushra_proxy

            _mushra_proxy = get_mushra_proxy()
            _verdict = _mushra_proxy.evaluate(
                phase_id=phase_id,
                audio_before=audio,
                audio_after=result.audio,
                sample_rate=sample_rate,
            )
            # Telemetrie immer speichern — auch bei erfolgreicher Phase
            result.metadata["mushra_proxy_delta"] = round(_verdict.delta, 4)
            result.metadata["mushra_proxy_before"] = round(_verdict.mushra_before, 2)
            result.metadata["mushra_proxy_after"] = round(_verdict.mushra_after, 2)
            result.metadata["mushra_proxy_latency_ms"] = round(_verdict.latency_ms, 2)
            result.metadata["mushra_proxy_version"] = "mert" if _mushra_proxy._mert_available else "bark-fallback"

            # §G145: delta ≤ 0 → kompromissloser Rollback
            if _verdict.should_rollback or _verdict.delta <= 0.0:
                _rollback_reason = _verdict.rollback_reason or (
                    f"delta={_verdict.delta:.4f} ≤ 0 — Phase verschlechtert perzeptuelle Qualität"
                )
                self._logger.warning(
                    "§G144/§G145 MUSHRA-Proxy %s: ROLLBACK — %s (mushra: %.1f→%.1f, Δ=%.3f, latency=%.1fms, proxy=%s)",
                    phase_id,
                    _rollback_reason,
                    _verdict.mushra_before,
                    _verdict.mushra_after,
                    _verdict.delta,
                    _verdict.latency_ms,
                    result.metadata["mushra_proxy_version"],
                )
                # §G145: result.audio auf Pre-Phase-Audio zurücksetzen
                result.audio = audio.astype(np.float32, copy=True)
                result.metadata["mushra_proxy_rollback"] = True
                result.metadata["mushra_proxy_rollback_reason"] = _rollback_reason
                result.warnings.append(f"§G144/§G145 MUSHRA-Proxy Rollback: {_rollback_reason}")
                # Qualitäts-Schätzung konservativ abwerten
                result.quality_estimate = max(0.4, result.quality_estimate - 0.15)
            else:
                self._logger.info(
                    "§G144 MUSHRA-Proxy %s: PASS — mushra %.1f→%.1f (Δ=+.3f, latency=%.1fms, proxy=%s)",
                    phase_id,
                    _verdict.mushra_before,
                    _verdict.mushra_after,
                    _verdict.delta,
                    _verdict.latency_ms,
                    result.metadata["mushra_proxy_version"],
                )
                result.metadata["mushra_proxy_rollback"] = False
        except ImportError:
            self._logger.debug(
                "§G144 MUSHRA-Proxy nicht verfügbar (mushra_proxy Import fehlgeschlagen) — "
                "perzeptueller Per-Phase-Guard deaktiviert"
            )
            result.metadata["mushra_proxy_available"] = False
        except Exception as _mp_exc:
            self._logger.debug("§G144 MUSHRA-Proxy non-blocking: %s", _mp_exc)
            result.metadata["mushra_proxy_error"] = str(_mp_exc)[:200]

        # ── VocalQualityGate: Gesangsqualität prüfen (nur bei Vokal-Phasen) ─
        if any(kw in phase_id for kw in ("42", "65", "vocal", "voice", "deess")):
            try:
                from backend.core.vocal_quality_gate import get_vocal_quality_gate

                gate = get_vocal_quality_gate()
                decision = gate.evaluate(
                    pre_audio=audio,
                    post_audio=result.audio,
                    sr=sample_rate,
                    phase_name=phase_id,
                )
                if decision.rollback_needed:
                    result.warnings.append(f"VocalQualityGate: Rollback empfohlen (Δ={decision.naturalness_delta:.1f})")
                    result.warnings.extend(decision.warnings)
                    # Leichte Qualitätsabwertung bei Rollback
                    result.quality_estimate = max(0.5, result.quality_estimate - 0.1)
                if decision.recommendations:
                    result.metadata["vocal_recommendations"] = decision.recommendations
            except Exception as _vqg_exc:
                self._logger.debug("VocalQualityGate skipped: %s", _vqg_exc)

        result.execution_time_seconds = time.monotonic() - t0
        return result

    @property
    def phase_id(self) -> str:
        """Kurz-ID dieser Phase."""
        return self.get_metadata().phase_id

    @property
    def name(self) -> str:
        """Anzeigename dieser Phase."""
        if self._name_override is not None:
            return self._name_override
        return self.get_metadata().name

    @name.setter
    def name(self, value: str) -> None:
        """Erlaubt Subklassen self.name = '...' im __init__ zu setzen."""
        self._name_override = value

    def validate_input(self, audio: np.ndarray) -> tuple[bool, str | None]:
        """Validiert Eingangs-Audio auf Korrektheits-Invarianten.

        Returns:
            (True, None) wenn valide, (False, Fehlermeldung) sonst.
        """
        if audio.size == 0:
            return False, "Empty audio input"
        if not np.isfinite(audio).all():
            return False, "Audio contains NaN or Inf values"
        if audio.ndim > 2:
            return False, "Audio must be mono or stereo"
        return True, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.phase_id!r})"
