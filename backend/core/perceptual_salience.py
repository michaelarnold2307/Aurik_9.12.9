"""Perceptual salience estimator for defect locations.

Assigns a perceptual salience score (0.0–1.0) to each defect event based on
psychoacoustic masking models.  Defects masked by louder surrounding content
score low; defects in quiet/exposed passages score high.

Scientific basis:
- Simultaneous masking: Fastl & Zwicker (2007) "Psychoacoustics: Facts and Models"
- Temporal masking: forward masking ~200 ms, backward masking ~20 ms (ISO 226:2003)
- Loudness model: ITU-R BS.1770-5 momentary loudness (400 ms windows)

The estimator does NOT modify audio — it only annotates DefectScore metadata
with a ``perceptual_salience`` field that downstream stages can use to:
1. Prioritize repair of perceptually salient defects
2. Skip repair of masked (inaudible) defects to reduce artefact risk
3. Report to the user which defects were audible

Module invariants (§3.x compliant):
- Thread-safe singleton via double-checked locking
- NaN/Inf guard on all numeric outputs
- No sample-rate assertion (analysis module — works at native import SR)
- English docstrings and log messages
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from backend.core.defect_scanner import DefectAnalysisResult, DefectType

logger = logging.getLogger(__name__)


@dataclass
class SalienceAnnotation:
    """Salience annotation for a single defect event."""

    defect_type: DefectType
    location: tuple[float, float]  # (start_s, end_s)
    salience: float  # 0.0 = completely masked, 1.0 = fully exposed
    local_loudness_lufs: float  # momentary loudness at defect location
    surrounding_loudness_lufs: float  # loudness of masking context (±400 ms)
    masking_type: str  # "simultaneous" | "temporal_forward" | "temporal_backward" | "none"


@dataclass
class SalienceResult:
    """Result of perceptual salience analysis for all defects."""

    annotations: list[SalienceAnnotation] = field(default_factory=list)
    mean_salience: float = 0.0
    n_salient: int = 0  # events with salience >= 0.5
    n_masked: int = 0  # events with salience < 0.3
    pass_through_detected: bool = False  # Hörordnung Ebene 2: Filter maskiert nichts


class PerceptualSalienceEstimator:
    """Schätzt perceptual salience of detected defect events.

    Uses momentary loudness (ITU-R BS.1770-5, 400 ms windows) to determine
    whether a defect is perceptually masked by surrounding audio content.

    Masking model:
    - Simultaneous: defect during loud passage (defect loudness < context - 12 dB)
    - Temporal forward: defect within 200 ms after loud transient (context - 8 dB)
    - Temporal backward: defect within 20 ms before loud passage (context - 6 dB)
    """

    _WINDOW_S = 0.4  # ITU-R BS.1770-5 momentary loudness (400 ms)
    _HOP_S = 0.1  # 100 ms hop for loudness profile
    _FORWARD_MASK_S = 0.200  # forward masking duration (200 ms at 1 kHz, varies with frequency)
    _BACKWARD_MASK_S = 0.020  # backward masking duration (20 ms)
    _SIMULTANEOUS_THRESHOLD_DB = 12.0  # dB below context = masked
    _FORWARD_THRESHOLD_DB = 8.0
    _BACKWARD_THRESHOLD_DB = 6.0

    @staticmethod
    def _forward_mask_duration_ms(dominant_freq_hz: float = 1000.0) -> float:
        """Frequency-dependent forward masking duration (Fastl & Zwicker 2007, Fig. 8.5).

        Low-frequency maskers produce longer forward masking:
        - 100 Hz  → ~400 ms
        - 1000 Hz → ~200 ms (baseline)
        - 8000 Hz → ~50 ms

        Logarithmic interpolation between 100 Hz and 8 kHz, clamped to [50, 500] ms.
        """
        f = float(np.clip(dominant_freq_hz, 100.0, 8000.0))
        log_f = np.log10(f / 100.0)
        mask_ms = 400.0 - log_f * 185.0
        return float(np.clip(mask_ms, 50.0, 500.0))

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        defect_result: DefectAnalysisResult,
    ) -> SalienceResult:
        """Annotate all defect events with perceptual salience scores.

        Parameters
        ----------
        audio : np.ndarray
            Mono or stereo audio at native sample rate.
        sr : int
            Sample rate in Hz.
        defect_result : DefectAnalysisResult
            Output of DefectScanner.scan() with locations.

        Returns
        -------
        SalienceResult with per-event annotations.
        """
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        mono = np.nan_to_num(mono.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # Build momentary loudness profile (ITU-R BS.1770-5 simplified: RMS in dBFS)
        loudness_profile = self._compute_loudness_profile(mono, sr)
        duration_s = len(mono) / sr

        annotations: list[SalienceAnnotation] = []

        for defect_type, defect_score in defect_result.scores.items():
            if not defect_score.locations:
                continue
            for loc_start, loc_end in defect_score.locations:
                salience, local_lufs, context_lufs, mask_type = self._score_event(
                    loudness_profile,
                    sr,
                    duration_s,
                    loc_start,
                    loc_end,
                )
                annotations.append(
                    SalienceAnnotation(
                        defect_type=defect_type,
                        location=(loc_start, loc_end),
                        salience=salience,
                        local_loudness_lufs=local_lufs,
                        surrounding_loudness_lufs=context_lufs,
                        masking_type=mask_type,
                    )
                )

        mean_sal = float(np.mean([a.salience for a in annotations])) if annotations else 0.0
        n_salient = sum(1 for a in annotations if a.salience >= 0.5)
        n_masked = sum(1 for a in annotations if a.salience < 0.3)

        result = SalienceResult(
            annotations=annotations,
            mean_salience=float(np.nan_to_num(mean_sal, nan=0.0)),
            n_salient=n_salient,
            n_masked=n_masked,
        )

        # Hörordnung Ebene 2 (hoerordnung.instructions.md §4) — Mindestanforderung:
        # Ein Salience-Filter, der real maskiert, darf auf breitbandigem Musikmaterial
        # keinen Pass-Through liefern. Befund 2026-08-23: 12969/12969 salient,
        # mean=1.000 — das Modell vergleicht Spitze-gegen-Spitze (die Defekt-Spitze
        # IST das lokale Maximum), daher maskiert der ±400 ms-Kontext praktisch nie.
        # Solange das der Fall ist, darf der Filter KEINE Audibility-Entscheidung
        # tragen (kein Skip „inaudible“, keine Severity-Absenkung auf Basis Salience).
        _n_ann = len(annotations)
        _pass_through = False
        if _n_ann >= 50:
            _salient_ratio = n_salient / _n_ann
            if _salient_ratio >= 0.99 and result.mean_salience >= 0.99:
                _pass_through = True
                logger.warning(
                    "Hörordnung Ebene 2: PerceptualSalience wirkt als Pass-Through "
                    "(%d/%d salient, mean=%.3f) — trägt keine Audibility-Entscheidung "
                    "(Defekt-Spitze ist lokales Maximum, Kontext maskiert nicht)",
                    n_salient,
                    _n_ann,
                    result.mean_salience,
                )
        result.pass_through_detected = _pass_through  # type: ignore[attr-defined]

        logger.info(
            "PerceptualSalience: %d events analysed, %d salient (>=0.5), %d masked (<0.3), mean=%.3f",
            len(annotations),
            n_salient,
            n_masked,
            result.mean_salience,
        )

        # §SOTA #10: Binaural Masking — Inter-Aural Cross-Correlation (IACC)
        # Bei breitem Stereo (IACC<0.7) können Defekte in einem Kanal durch
        # das kontralaterale Ohr partiell maskiert werden → Salience↓
        if audio.ndim == 2 and audio.shape[0] == 2:
            try:
                _l = audio[0, : min(len(audio[0]), sr * 10)].astype(np.float64)
                _r = audio[1, : min(len(audio[1]), sr * 10)].astype(np.float64)
                _iacc = float(np.corrcoef(_l, _r)[0, 1]) if len(_l) > 100 else 1.0
                _iacc = max(0.0, min(1.0, _iacc))
                if _iacc < 0.85:
                    _binaural_factor = float(np.clip(0.90 + 0.10 * _iacc, 0.90, 1.0))
                    result.mean_salience *= _binaural_factor
                    logger.debug(
                        "§SOTA #10 Binaural: IACC=%.3f → salience ×%.2f",
                        _iacc,
                        _binaural_factor,
                    )
            except Exception as e:
                logger.warning("perceptual_salience.py::unbekannter Ersatzpfad: %s", e)

        return result

    def annotate_defect_scores(
        self,
        audio: np.ndarray,
        sr: int,
        defect_result: DefectAnalysisResult,
        use_erb_model: bool = True,
    ) -> DefectAnalysisResult:
        """Annotate DefectAnalysisResult in-place with salience metadata.

        Adds to each DefectScore.metadata:
        - ``perceptual_salience``: mean salience across events (0.0–1.0)
        - ``n_salient_events``: count of events with salience >= 0.5
        - ``n_masked_events``: count of events with salience < 0.3

        Additionally scales severity by mean salience:
        ``adjusted_severity = severity * (0.3 + 0.7 * mean_salience)``
        This preserves a base severity (30%) even for fully masked defects
        while boosting exposed defects to near-original severity.

        Parameters
        ----------
        use_erb_model : bool
            If True (default), uses the ERB auditory masking model for
            frequency-dependent masking thresholds (Glasberg & Moore 1990).
            Falls back to broadband model on import error.
        """
        # First run broadband analysis (always needed for annotations)
        salience_result = self.estimate(audio, sr, defect_result)

        # Optionally enhance with ERB model for frequency-dependent masking
        erb_saliences: dict[tuple[DefectType, tuple[float, float]], float] = {}
        if use_erb_model:
            try:
                from backend.core.erb_auditory_masking import get_erb_auditory_masking_model

                erb_model = get_erb_auditory_masking_model()
                mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
                mono = np.nan_to_num(np.asarray(mono, dtype=np.float64), nan=0.0)

                # ERB masking is expensive (3 FFTs per annotation).
                # Quality-first default: keep full budget for maximum detection fidelity.
                # Reduced budgets are opt-in only via env var to avoid hidden quality drift.
                # Strategy: take up to _ERB_PER_TYPE highest-salience annotations per
                # defect type so all types are represented, then fill remaining budget
                # with remaining annotations sorted by broadband salience descending.
                _duration_s = float(len(mono)) / float(max(1, sr))
                _erb_budget_mode = str(os.getenv("AURIK_ERB_BUDGET_MODE", "quality")).strip().lower()
                if _erb_budget_mode in {"fast", "balanced"}:
                    if _duration_s >= 300.0:
                        _ERB_MAX_ANNOTATIONS = 180
                    elif _duration_s >= 180.0:
                        _ERB_MAX_ANNOTATIONS = 260
                    elif _duration_s >= 90.0:
                        _ERB_MAX_ANNOTATIONS = 360
                    else:
                        _ERB_MAX_ANNOTATIONS = 500
                else:
                    _ERB_MAX_ANNOTATIONS = 500
                _ERB_PER_TYPE = 20
                all_anns = salience_result.annotations
                if len(all_anns) > _ERB_MAX_ANNOTATIONS:
                    # Tier 1: pro Defekttyp die Salience-Verteilung an BEIDEN Enden
                    # abdecken — die lautesten (exponierten) UND die leisesten
                    # (potentiell maskierbaren) Events. Vorher wurden nur die
                    # top-salienten gezogen → die Stichprobe war systematisch
                    # verzerrt Richtung „alles salient“ (Befund 2026-08-23:
                    # 13695/13695 salient trotz ERB+Residuum-Blend).
                    _by_type: dict = {}
                    for _a in all_anns:
                        _by_type.setdefault(_a.defect_type, []).append(_a)
                    _selected: list = []
                    _half = max(1, _ERB_PER_TYPE // 2)
                    for _type_anns in _by_type.values():
                        _type_anns_sorted = sorted(_type_anns, key=lambda a: a.salience, reverse=True)
                        _picked = _type_anns_sorted[:_half]
                        if len(_type_anns_sorted) > _half:
                            _picked = _picked + _type_anns_sorted[-_half:]
                        _selected.extend(_picked)
                    # Tier 2: fill remaining budget from all annotations not yet selected
                    _selected_set = {id(a) for a in _selected}
                    _remaining = sorted(
                        (a for a in all_anns if id(a) not in _selected_set),
                        key=lambda a: a.salience,
                        reverse=True,
                    )
                    _selected.extend(_remaining[: max(0, _ERB_MAX_ANNOTATIONS - len(_selected))])
                    erb_anns = _selected
                    logger.debug(
                        "ERB masking: capped %d → %d annotations (Grenze=%d, duration=%.1fs, Betriebsart=%s)",
                        len(all_anns),
                        len(erb_anns),
                        _ERB_MAX_ANNOTATIONS,
                        _duration_s,
                        _erb_budget_mode,
                    )
                else:
                    erb_anns = all_anns

                for ann in erb_anns:
                    erb_result = erb_model.compute_masking_threshold(
                        mono,
                        sr,
                        ann.location[0],
                        ann.location[1],
                    )
                    # Blend: 70% ERB model (frequency-aware) + 30% broadband (robust)
                    blended = 0.7 * erb_result.salience + 0.3 * ann.salience
                    # Hörordnung Ebene 2: Residuum-Bark-Masking (Defekt-Anteil vs.
                    # maskierender Inhalt) als dritter Term — unter demselben
                    # Budget-Cap wie ERB (keine Zusatzkosten bei großen Scans).
                    try:
                        from backend.core.residuum_masking import estimate_residuum_salience as _residuum_sal

                        _rs = _residuum_sal(mono, sr, ann.location[0], ann.location[1])
                        # Diskrepanz-Regel (Hörordnung Ebene 2): Sagt das Residuum-
                        # Modell „maskiert“ (Defekt hebt sich nicht vom Kontext ab),
                        # während Broadband/ERB „exponiert“ melden, ist das Residuum
                        # die richtige Autorität — der Defekt-Anteil über dem
                        # maskierenden Inhalt IST die Hörbarkeits-Frage. Sonst
                        # überstimmen die Spitzen-basierten Terme die Maskierung
                        # (Befund 2026-08-23: Blend lieferte 1.0 trotz Residuum≈0).
                        if _rs.salience < 0.30 and ann.salience >= 0.90:
                            blended = 0.20 * erb_result.salience + 0.10 * ann.salience + 0.70 * _rs.salience
                        else:
                            blended = 0.5 * erb_result.salience + 0.3 * ann.salience + 0.2 * _rs.salience
                    except Exception as _rs_exc:
                        logger.debug("Residuum-Masking nicht verfügbar (ERB-Blend aktiv): %s", _rs_exc)
                    ann.salience = float(np.clip(blended, 0.0, 1.0))
                    erb_saliences[(ann.defect_type, ann.location)] = erb_result.salience

                logger.info(
                    "ERB masking model verbessert %d salience annotations",
                    len(erb_saliences),
                )
                # Hörordnung Ebene 2: Post-Blend-Auswertung — die ehrliche Frage
                # ist, ob NACH ERB+Residuum weiterhin alles salient ist. Erst dann
                # ist die Pass-Through-Warnung entscheidungsrelevant (die
                # estimate()-Warnung prüft nur den Broadband-Vorzustand).
                _post_salient = sum(1 for _a in salience_result.annotations if _a.salience >= 0.5)
                _post_masked = sum(1 for _a in salience_result.annotations if _a.salience < 0.3)
                _post_n = len(salience_result.annotations)
                if _post_n >= 50 and _post_masked == 0 and _post_salient >= _post_n * 0.99:
                    logger.warning(
                        "Hörordnung Ebene 2: auch nach ERB+Residuum-Blend Pass-Through "
                        "(%d/%d salient, 0 maskiert) — Audibility bleibt eingeschränkt; "
                        "Wurzel: Residuum-Modell auf %d/%d Events begrenzt (Budget-Cap)",
                        _post_salient,
                        _post_n,
                        len(erb_saliences),
                        _post_n,
                    )
                elif _post_masked > 0:
                    logger.info(
                        "Hörordnung Ebene 2: ERB+Residuum maskiert %d/%d Events (%d über Cap verfeinert)",
                        _post_masked,
                        _post_n,
                        len(erb_saliences),
                    )
            except ImportError:
                logger.debug("ERB masking model not verfuegbar, using broadband only")

        # Group annotations by defect type
        by_type: dict[DefectType, list[SalienceAnnotation]] = {}
        for ann in salience_result.annotations:
            by_type.setdefault(ann.defect_type, []).append(ann)

        for dt, type_annotations in by_type.items():
            if dt not in defect_result.scores:
                continue
            ds = defect_result.scores[dt]
            mean_sal = float(np.mean([a.salience for a in type_annotations]))
            mean_sal = float(np.nan_to_num(mean_sal, nan=0.5))
            # Hörordnung Ebene 2: Neutralisierung nur, wenn der Filter auch NACH
            # ERB+Residuum-Blend nichts maskiert hat. Hat der Blend real maskiert,
            # ist die Salience informativ und wird NICHT neutralisiert.
            _type_masked_ho = sum(1 for a in type_annotations if a.salience < 0.3)
            if getattr(salience_result, "pass_through_detected", False) and _type_masked_ho == 0:
                mean_sal = 1.0
            ds.metadata["perceptual_salience"] = round(mean_sal, 3)
            ds.metadata["n_salient_events"] = sum(1 for a in type_annotations if a.salience >= 0.5)
            ds.metadata["n_masked_events"] = sum(1 for a in type_annotations if a.salience < 0.3)

            # Timing-Defekte sind Lautstärken-unabhängig: Loudness-Masking gilt NICHT für
            # Pitch-/Zeitmodulationen (Houtsma et al. 1980; Hartmann 1991 — JND für
            # Frequenzmodulation ist signalpegel-unabhängig). v10.0.0
            _TIMING_DEFECTS_NO_SALIENCE_SCALE = frozenset(
                {
                    DefectType.WOW,
                    DefectType.FLUTTER,
                    DefectType.MULTIBAND_WOW_FLUTTER,
                    DefectType.PITCH_DRIFT,
                }
            )

            # Scale severity: masked defects get reduced priority
            old_sev = ds.severity
            if dt in _TIMING_DEFECTS_NO_SALIENCE_SCALE:
                # Timing-Defekte: Severity wird NICHT durch Lautstärkekontext skaliert.
                # Nur Metadaten-Annotation, keine Severity-Reduktion.
                ds.metadata["salience_scale_skipped"] = "timing_defect"
            else:
                ds.severity = float(
                    np.nan_to_num(
                        min(1.0, old_sev * (0.3 + 0.7 * mean_sal)),
                        nan=0.0,
                    )
                )
            if abs(ds.severity - old_sev) > 0.01:
                logger.debug(
                    "Salience adjustment: %s severity %.3f → %.3f (salience=%.3f)",
                    dt.value,
                    old_sev,
                    ds.severity,
                    mean_sal,
                )

        return defect_result

    # ------------------------------------------------------------------
    # Internal: Loudness profile
    # ------------------------------------------------------------------

    def _compute_loudness_profile(
        self,
        mono: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Berechnet momentary loudness profile (dBFS, 400 ms windows, 100 ms hop).

        Returns array of shape (n_frames,) with loudness in dBFS per frame.
        """
        win_samples = max(1, int(self._WINDOW_S * sr))
        hop_samples = max(1, int(self._HOP_S * sr))

        n_frames = max(1, (len(mono) - win_samples) // hop_samples + 1)
        loudness = np.full(n_frames, -100.0, dtype=np.float64)

        for i in range(n_frames):
            start = i * hop_samples
            end = start + win_samples
            if end > len(mono):
                break
            rms = np.sqrt(np.mean(mono[start:end] ** 2) + 1e-12)
            loudness[i] = 20.0 * np.log10(max(rms, 1e-10))

        return loudness  # type: ignore[no-any-return]

    def _time_to_frame(self, t: float, sr: int) -> int:
        """Konvertiert time in seconds to loudness profile frame index."""
        hop_samples = max(1, int(self._HOP_S * sr))
        return max(0, int(t * sr / hop_samples))

    def _score_event(
        self,
        loudness_profile: np.ndarray,
        sr: int,
        duration_s: float,
        loc_start: float,
        loc_end: float,
    ) -> tuple[float, float, float, str]:
        """Bewertet a single defect event for perceptual salience.

        Returns (salience, local_lufs, context_lufs, masking_type).
        """
        n_frames = len(loudness_profile)
        if n_frames == 0:
            return 1.0, -100.0, -100.0, "none"

        # Frame indices for the defect location
        f_start = min(self._time_to_frame(loc_start, sr), n_frames - 1)
        f_end = min(self._time_to_frame(loc_end, sr), n_frames - 1)
        f_end = max(f_end, f_start)

        # Local loudness at defect location
        local_lufs = float(np.max(loudness_profile[f_start : f_end + 1]))

        # Context: surrounding ±400 ms window (excluding the defect itself)
        ctx_start_t = max(0.0, loc_start - self._WINDOW_S)
        ctx_end_t = min(duration_s, loc_end + self._WINDOW_S)
        cf_start = min(self._time_to_frame(ctx_start_t, sr), n_frames - 1)
        cf_end = min(self._time_to_frame(ctx_end_t, sr), n_frames - 1)

        # Context frames excluding defect region
        ctx_frames = np.concatenate(
            [
                loudness_profile[cf_start:f_start],
                loudness_profile[f_end + 1 : cf_end + 1],
            ]
        )
        if len(ctx_frames) == 0:
            return 1.0, local_lufs, local_lufs, "none"

        context_lufs = float(np.max(ctx_frames))

        # Check masking conditions
        diff_db = context_lufs - local_lufs

        # Forward masking: loud content just before the defect
        fwd_start_t = max(0.0, loc_start - self._FORWARD_MASK_S)
        ff_start = min(self._time_to_frame(fwd_start_t, sr), n_frames - 1)
        pre_lufs = float(np.max(loudness_profile[ff_start : f_start + 1])) if ff_start < f_start else -100.0

        # Backward masking: loud content just after the defect
        bwd_end_t = min(duration_s, loc_end + self._BACKWARD_MASK_S)
        bf_end = min(self._time_to_frame(bwd_end_t, sr), n_frames - 1)
        post_lufs = float(np.max(loudness_profile[f_end : bf_end + 1])) if f_end < bf_end else -100.0

        masking_type = "none"
        salience = 1.0

        # Simultaneous masking (context louder than defect by threshold)
        if diff_db >= self._SIMULTANEOUS_THRESHOLD_DB:
            masking_type = "simultaneous"
            # Salience decreases with increasing masking margin
            salience = max(0.0, 1.0 - (diff_db - self._SIMULTANEOUS_THRESHOLD_DB) / 20.0)

        # Forward masking (loud content before defect)
        elif (pre_lufs - local_lufs) >= self._FORWARD_THRESHOLD_DB:
            masking_type = "temporal_forward"
            margin = pre_lufs - local_lufs - self._FORWARD_THRESHOLD_DB
            salience = max(0.0, 1.0 - margin / 15.0)

        # Backward masking (loud content after defect)
        elif (post_lufs - local_lufs) >= self._BACKWARD_THRESHOLD_DB:
            masking_type = "temporal_backward"
            margin = post_lufs - local_lufs - self._BACKWARD_THRESHOLD_DB
            salience = max(0.0, 1.0 - margin / 15.0)

        salience = float(np.nan_to_num(np.clip(salience, 0.0, 1.0), nan=0.5))
        return salience, local_lufs, context_lufs, masking_type


# ---------------------------------------------------------------------------
# Thread-safe singleton (double-checked locking — §3.2)
# ---------------------------------------------------------------------------

_instance: PerceptualSalienceEstimator | None = None
_lock = threading.Lock()


def get_perceptual_salience_estimator() -> PerceptualSalienceEstimator:
    """Gibt thread-safe singleton PerceptualSalienceEstimator zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PerceptualSalienceEstimator()
    return _instance
