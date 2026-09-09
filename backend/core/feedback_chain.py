from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modul-Konstanten (Spec §2.16 / §9.5)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_SCORE: float = 0.70  # Standard-Qualitätsschwelle (MOS-normalisiert)
EXCELLENCE_TARGET_SCORE: float = 0.85  # Verschärftes Ziel im Excellence-Modus
MUSIC_OVR_EXCELLENCE_THRESHOLD: float = 0.90  # Musik-OVR Schwelle für Excellence
HEADROOM_THRESHOLD: float = 0.03  # §2.33 PhysicalCeilingEstimator: Δ < 3 % → früher Abbruch


@dataclass
class FeedbackChainResult:
    """Ergebnis-Datenklasse der FeedbackChain nach allen Optimierungsiterationen."""

    audio: np.ndarray
    iterations: int
    converged: bool
    mos_history: list[float] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    phase_executions: list[dict] = field(default_factory=list)  # Ausgeführte Phasen-Protokolle
    overall_score: float = 0.0  # Gesamt-Score nach allen Iterationen
    total_retries: int = 0  # Kompatibilitaet: Alias fuer iterations
    total_time_s: float = 0.0  # Gesamtdauer der Feedback-Schleife in Sekunden
    ceiling_reached: bool = False  # §2.33: True wenn PhysicalCeiling frühzeitig erreicht
    analytics_overhead_s: float = 0.0  # Time spent in goal-measurement calls (excluded from RT budget)


class FeedbackChain:
    """Iterative quality loop with conservative convergence control."""

    # §Hörordnung-Pre-Filter (hoerordnung.instructions.md §5, "Teamwork statt
    # Dominanz", Produktionsbefund 2026-09-07): Primäre Ziel-Goals der
    # FC-Phasen (numerische Phase-ID → Goal-Namen). Unbekannte Goal-Namen
    # fallen auf hearing_tier()=3 zurück — der Filter bleibt damit konservativ.
    # Die GoalPriorityProtocol/GPP-Prüfung bleibt der autoritative Gatekeeper;
    # dieser Pre-Filter überspringt nur Kandidaten, deren Ziel-Stufe über der
    # niedrigsten Defizit-Stufe liegt (lexikografische Ordnung).
    FC_PHASE_PRIMARY_GOALS: dict[int, tuple[str, ...]] = {
        7: ("waerme", "natuerlichkeit"),  # Harmonic Restoration
        14: ("transparenz",),  # Phase correction
        16: ("brillanz", "waerme"),  # Final EQ
        17: ("brillanz", "groove"),  # Mastering polish
        40: ("emotionalitaet", "loudness_consistency"),  # Loudness normalization
        19: ("artikulation", "transparenz"),  # De-esser
        22: ("micro_dynamics", "groove"),  # Dynamic enhancement
        36: ("micro_dynamics", "emotionalitaet"),  # Log-envelope smoothing
        47: ("waerme", "timbre_authentizitaet"),  # Timbre restoration
        48: ("spatial_depth", "separation_fidelity"),  # Stereo imaging
        55: ("brillanz", "bass_kraft"),  # Spectral band enhancement
    }

    # §Hörordnung-Pre-Filter Baseline: uv3 setzt den Goal-Snapshot VOR der
    # FC-Iteration hier ein (dynamisches Attribut war untypisiert — mypy).
    baseline_goals: dict[str, object] | None = None

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_delta: float = 0.02,
        *,
        sample_rate: int = 48000,
        target_score: float | None = None,
        excellence_mode: bool = False,
        material: str = "auto",
        use_mert: bool = False,
        use_pqs_in_loop: bool = False,
        use_versa_in_loop: bool = True,  # §VERBOTEN: VERSA muss immer aktiv sein (§2.44)
        max_retries: int | None = None,
        restorability_score: float = 50.0,
        defect_severity_mean: float = 0.3,
        panns_singing: float = 0.0,
        max_runtime_s: float | None = None,
    ) -> None:
        # Legacy-Kompatibilitaet: max_retries entspricht max_iterations.
        if max_retries is not None:
            max_iterations = int(max_retries)
        self.max_iterations = max(1, int(max_iterations))
        self.convergence_delta = max(1e-6, float(convergence_delta))
        self.max_runtime_s = float(max_runtime_s) if max_runtime_s is not None else 180.0
        self.max_runtime_s = max(0.05, self.max_runtime_s)
        self.sample_rate = int(sample_rate)
        self.excellence_mode = bool(excellence_mode)
        self.material = str(material)
        self.restorability_score = float(np.clip(restorability_score, 0.0, 100.0))
        self.defect_severity_mean = float(np.clip(defect_severity_mean, 0.0, 1.0))
        self.use_mert = bool(use_mert)
        self.use_pqs_in_loop = bool(use_pqs_in_loop)
        self.use_versa_in_loop = bool(use_versa_in_loop)
        self.panns_singing: float = float(np.clip(panns_singing, 0.0, 1.0))  # §0p VQI-Gate
        self.era_decade: int = 1975  # §EraVocalProfile: optional von UV3 überschrieben
        self._vqi_orig_audio: np.ndarray | None = None  # gesetzt in run() — §0p Dual-Objective
        self.frisson_zones: list | None = None  # §Frisson: gesetzt von UV3 vor FC-Loop
        self.frisson_orig_audio: np.ndarray | None = None  # §Frisson: Original-Audio-Referenz
        self.goal_priority_callback: Callable[[np.ndarray, np.ndarray], tuple[bool, str]] | None = None
        self.goal_weights: dict[str, float] | None = None  # §2.56 Song-Goal-Importance
        self.adaptive_goal_thresholds: dict[str, float] | None = None  # §09.2 per-song adaptive targets
        # §Hebel5: Sub-JND phase IDs from UV3 main pipeline — skip immediately in FC loop
        self.pre_pruned_phase_ids: frozenset[str] = frozenset()
        self._pqs_score_fn: Callable[[np.ndarray, int], object] | None = None
        self._versa_score_fn: Callable[[np.ndarray, int], object] | None = None
        self._last_score_source: str = "heuristic_rms"
        self._last_analytics_overhead_s: float = 0.0  # §perf: accumulated goal-measurement overhead
        if self.use_pqs_in_loop:
            try:
                from backend.core.perceptual_quality_scorer import (  # pylint: disable=import-outside-toplevel
                    score_audio_absolute,
                )

                self._pqs_score_fn = score_audio_absolute
            except Exception as exc:
                logger.warning(
                    "§G23 FeedbackChain: PQS scorer nicht verfügbar, heuristic Ersatzpfad: %s", exc, exc_info=True
                )
        if self.use_versa_in_loop:
            try:
                from plugins.versa_plugin import (  # pylint: disable=import-outside-toplevel
                    get_loaded_versa_plugin,
                    get_versa_plugin,
                )

                _versa_plugin = get_loaded_versa_plugin()
                if _versa_plugin is None:
                    _versa_plugin = get_versa_plugin()
                if _versa_plugin is not None:
                    self._versa_score_fn = _versa_plugin.score
            except Exception as exc:
                logger.warning(
                    "FeedbackChain: VERSA scorer nicht verfügbar — PQS/RMS-Ersatzpfad (§V6 (copilot-instructions.md)): %s",
                    exc,
                )
                try:
                    from backend.core.fallback_auditor import get_fallback_auditor

                    get_fallback_auditor().record("FeedbackChain", "versa_mos", "pqs_rms_dsp", "versa_load_failed")
                except Exception:
                    logger.debug("FallbackAuditor nicht verfügbar (unkritisch)", exc_info=True)
        # target_score: explizit gesetzt oder aus excellence_mode abgeleitet
        excellence_target = EXCELLENCE_TARGET_SCORE if excellence_mode else DEFAULT_TARGET_SCORE
        if target_score is not None:
            self.target_score = float(max(target_score, excellence_target))
        else:
            self.target_score = excellence_target

    _SCORE_EXCERPT_S = 90.0
    _SCORE_WINDOW_S = 30.0
    _SCORE_WINDOWS_N = 3

    @staticmethod
    def compute_perceptual_score(audio: np.ndarray) -> float:
        """Berechnet RMS-basierte Näherung des Wahrnehmungsqualitäts-Scores (1.0–5.0)."""
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        return float(np.clip(1.0 + 4.0 * (1.0 - np.exp(-8.0 * rms)), 1.0, 5.0))

    @classmethod
    def _score_windows(cls, audio: np.ndarray, sr: int) -> list[np.ndarray]:
        """Deterministische Analyse-Fenster für ML-Scorer (§G5 (copilot-instructions.md), §9 Performance-Budget).

        Bei Signalen > 90 s: 3 Fenster à 30 s (Anfang/Mitte/Ende). Gleicher Input
        ⇒ gleiche Fenster ⇒ bit-identische Scores. Kürzere Signale: als Ganzes.
        """
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            _axis = 0 if arr.shape[-1] <= 2 else -1
        else:
            _axis = 0
        total = arr.shape[_axis]
        win = int(cls._SCORE_WINDOW_S * sr)
        if total <= win * cls._SCORE_WINDOWS_N or total <= int(cls._SCORE_EXCERPT_S * sr):
            return [arr]
        starts = [0, (total - win) // 2, total - win]
        return [np.take(arr, np.arange(s, s + win), axis=_axis) for s in starts]

    def _score_single(self, audio: np.ndarray, sr: int) -> float:
        """VERSA → PQS → RMS-Heuristik für EIN Audio-Signal."""
        if self._versa_score_fn is not None:
            try:
                versa_mos = self._compute_versa_segmented_score(audio, sr)
                if np.isfinite(versa_mos):
                    self._last_score_source = "versa_segmented"
                    return float(np.clip(versa_mos, 1.0, 5.0))
            except Exception as exc:
                logger.debug("FeedbackChain: VERSA loop Wert fehlgeschlagen, trying PQS Ersatzpfad: %s", exc)
        if self._pqs_score_fn is not None:
            try:
                pqs = self._pqs_score_fn(audio, sr)
                pqs_mos = float(getattr(pqs, "pqs_mos", getattr(pqs, "mos", np.nan)))
                if np.isfinite(pqs_mos):
                    self._last_score_source = "pqs_absolute"
                    return float(np.clip(pqs_mos, 1.0, 5.0))
            except Exception as exc:
                logger.debug("FeedbackChain: PQS loop Wert fehlgeschlagen, Ersatzpfad active: %s", exc)
        self._last_score_source = "heuristic_rms"
        return self.compute_perceptual_score(audio)

    def _compute_iteration_score(self, audio: np.ndarray, sr: int) -> float:
        """Loop-Score: ML-Scorer auf deterministischen Fenstern bei langen Signalen.

        §9 Performance-Budget BUG-FIX 2026-08-22: Der ML-Scorer hat kein
        Längen-Cap — 224 s kosteten 37.3 s pro Aufruf und erschöpften das
        Iterations-Budget. Fenster-Scoring (Mittelwert der Fenster-Scores)
        senkt die Kosten ~2.5× und bleibt deterministisch.
        """
        _windows = self._score_windows(audio, sr)
        _scores = [self._score_single(w, sr) for w in _windows]
        base_mos = float(np.mean(_scores)) if _scores else 1.0
        return self._apply_vqi_dual_objective(audio, sr, base_mos)

    def _apply_vqi_dual_objective(self, audio: np.ndarray, sr: int, base_mos: float) -> float:
        """§0p: Dual-Objective VQI-Gewichtung wenn panns_singing ≥ 0.35.

        Loop-Score = VERSA_MOS × VQI^0.5 — sqrt-Gewichtung verhindert VQI-Dominanz,
        aber stellt sicher dass Vokal-Verschlechterungen den Score deutlich reduzieren
        (F-06 v10.0.0: Exponent 0.3→0.5 für stärkere Penalty bei VQI < 0.82).
        Zusätzlich: §Frisson-Gewichtung — Energy-Abfall in Frisson-Zonen (Gänsehaut-Passagen)
        reduziert den Loop-Score, damit FeedbackChain diese Klimax-Momente nicht wegoptimiert.
        Non-blocking: Exception → base_mos unverändert zurückgeben.
        """
        if self.panns_singing < 0.35 or self._vqi_orig_audio is None:
            return base_mos
        try:
            from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
                compute_vqi,
            )

            # §EraVocalProfile: era_decade für korrekte historische Formant-Toleranzen
            _era_profile = None
            try:
                from backend.core.musical_goals.era_vocal_profile import (  # pylint: disable=import-outside-toplevel
                    get_era_vocal_profile,
                )

                _era_profile = get_era_vocal_profile(int(getattr(self, "era_decade", 1975) or 1975))
            except Exception as _ep_exc:
                logger.debug("FeedbackChain era_Profil nicht geladen: %s", _ep_exc)

            # §9 Performance-Budget (Befund 2026-08-22): compute_vqi hat kein
            # Längen-Cap — Voll-Audio-VQI dominierte die Iterations-Latenz
            # (224 s ⇒ ~30+ s pro Aufruf). VQI daher NUR auf den denselben
            # deterministischen Analyse-Fenstern wie der ML-Scorer berechnen:
            # ≤90 s ⇒ 1 Fenster (Voll-Audio, identisch zum alten Verhalten),
            # >90 s ⇒ 3×30 s statt Voll-Länge (Kosten-Bound, §G5 (copilot-instructions.md) deterministisch).
            _vqi_windows_o = self._score_windows(self._vqi_orig_audio, sr)
            _vqi_windows_c = self._score_windows(audio, sr)
            _vqi_scores: list[float] = []
            for _ow, _cw in zip(_vqi_windows_o, _vqi_windows_c):
                _w_result = compute_vqi(_ow, _cw, sr, era_profile=_era_profile)
                _vqi_scores.append(float(np.clip(_w_result.get("vqi", 1.0), 0.01, 1.0)))
            vqi_score = float(np.mean(_vqi_scores)) if _vqi_scores else 1.0
            # §0p F-06: VQI^0.5 (sqrt) — stärkere Penalty bei sub-threshold VQI.
            # VQI=0.60 → Faktor 0.775 (war 0.849); VQI=0.72 → 0.849 (war 0.906).
            # Verhindert, dass VERSA-MOS-Gewinn (+0.15) die VQI-Penalty neutralisiert
            # → FeedbackChain konvergiert nicht mehr in VQI < 0.72 Region.
            loop_score = float(np.clip(base_mos * (vqi_score**0.5), 1.0, 5.0))
            logger.debug(
                "FeedbackChain §0p VQI-Dual-Objective: base_mos=%.3f vqi=%.3f loop_Wert=%.3f era=%d",
                base_mos,
                vqi_score,
                loop_score,
                int(getattr(self, "era_decade", 1975) or 1975),
            )
            # §Frisson [RELEASE_MUST]: Energie in Gänsehaut-Zonen überwachen.
            # Signifikanter Energie-Abfall in Frisson-Passagen → Loop-Score-Penalty.
            # Motivation: FC darf Klimax-Passagen (Falsett-Einsatz, Vibrato-Höhepunkt,
            # letzter Akkord) nicht durch Over-Processing komprimieren.
            _frisson_zones = getattr(self, "frisson_zones", None)
            _frisson_orig: np.ndarray | None = getattr(self, "frisson_orig_audio", None)
            if _frisson_zones and _frisson_orig is not None:
                _fo: np.ndarray = _frisson_orig  # type: ignore[assignment]  # narrowed: is not None guard oben
                try:
                    _frisson_penalty = 1.0
                    _n_frisson = int(audio.shape[-1] if audio.ndim == 2 else len(audio))
                    _n_orig = int(_fo.shape[-1] if _fo.ndim == 2 else len(_fo))
                    _n_cmp = min(_n_frisson, _n_orig)
                    _zones_checked = 0
                    for _fz in _frisson_zones:  # pylint: disable=not-an-iterable
                        _fz_s = float(getattr(_fz, "start_s", 0.0))
                        _fz_e = float(getattr(_fz, "end_s", 0.0))
                        if _fz_e <= _fz_s:
                            continue
                        _si = max(0, min(int(round(_fz_s * sr)), _n_cmp))
                        _ei = max(0, min(int(round(_fz_e * sr)), _n_cmp))
                        if _si >= _ei:
                            continue
                        if audio.ndim == 2:
                            _rms_out = float(np.sqrt(np.mean(audio[:, _si:_ei] ** 2) + 1e-14))
                        else:
                            _rms_out = float(np.sqrt(np.mean(audio[_si:_ei] ** 2) + 1e-14))
                        if _fo.ndim == 2:
                            _rms_ref = float(np.sqrt(np.mean(_fo[:, _si:_ei] ** 2) + 1e-14))  # type: ignore[index]  # pylint: disable=unsubscriptable-object
                        else:
                            _rms_ref = float(np.sqrt(np.mean(_fo[_si:_ei] ** 2) + 1e-14))  # type: ignore[index]  # pylint: disable=unsubscriptable-object
                        if _rms_ref > 1e-10:
                            _ratio = _rms_out / _rms_ref
                            # Penalty wenn Energie > 3 dB abgefallen (ratio < 0.708) —
                            # sanfte Gewichtung damit einzelne Frisson-Zone nicht dominiert
                            if _ratio < 0.708:  # -3 dB
                                _frisson_penalty = min(
                                    _frisson_penalty,
                                    float(np.clip(0.92 + 0.08 * _ratio / 0.708, 0.88, 1.0)),
                                )
                        _zones_checked += 1
                    if _frisson_penalty < 1.0:
                        loop_score = float(np.clip(loop_score * _frisson_penalty, 1.0, 5.0))
                        logger.debug(
                            "FeedbackChain §Frisson-Penalty: %.4f → loop_Wert=%.3f (%d Zonen)",
                            _frisson_penalty,
                            loop_score,
                            _zones_checked,
                        )
                except Exception as _frisson_exc:
                    logger.debug("FeedbackChain §Frisson nicht blockierend: %s", _frisson_exc)
            return loop_score
        except Exception as exc:
            logger.debug("FeedbackChain VQI dual-objective nicht blockierend: %s", exc)
            return base_mos

    def _compute_versa_segmented_score(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet VERSA MOS on up to 5 representative segments, energie-gewichtet aggregiert.

        Motivation: avoid local quality collapses being hidden by a single global MOS,
        but also avoid silence/fade segments dominating and masking good content quality.
        Stille-Segmente (RMS < −48 dBFS) werden vor der Aggregation ausgeschlossen.
        """
        if self._versa_score_fn is None:
            return float("nan")

        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            if arr.shape[1] <= 2 and arr.shape[0] > arr.shape[1]:
                mono = arr.mean(axis=1)
            elif arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
                mono = arr.mean(axis=0)
            else:
                mono = arr.mean(axis=-1)
        else:
            mono = arr.ravel()

        win = int(sr * 30)  # align with SingMOS 30 s design window
        if mono.size <= win:
            versa = self._versa_score_fn(mono, sr)
            return float(getattr(versa, "mos", np.nan))

        n_segments = int(np.clip(np.ceil(mono.size / win), 3, 5))
        half = win // 2
        centers = np.linspace(half, mono.size - half, n_segments, dtype=int)

        # RMS-Schwelle für Stille-Ausschluss: −48 dBFS ≈ 0.004
        _SILENCE_RMS_FLOOR: float = 10.0 ** (-48.0 / 20.0)

        seg_scores: list[float] = []
        seg_rms: list[float] = []
        for c in centers:
            s = int(max(0, c - half))
            e = int(min(mono.size, s + win))
            seg = mono[s:e]
            if seg.size < int(sr * 5):
                continue
            versa = self._versa_score_fn(seg, sr)
            mos = float(getattr(versa, "mos", np.nan))
            if np.isfinite(mos):
                seg_scores.append(float(np.clip(mos, 1.0, 5.0)))
                seg_rms.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-14)))

        if not seg_scores:
            versa = self._versa_score_fn(mono, sr)
            return float(getattr(versa, "mos", np.nan))

        # Stille-Segmente (Intro/Outro/Fade) ausschließen — sie verzerren den Qualitätsscore.
        # Fallback: alle Segmente wenn alle unter Schwelle liegen.
        _active_scores = [s for s, r in zip(seg_scores, seg_rms) if r >= _SILENCE_RMS_FLOOR]
        _active_rms = [r for r in seg_rms if r >= _SILENCE_RMS_FLOOR]
        if not _active_scores:
            _active_scores = seg_scores
            _active_rms = seg_rms

        # Energie-gewichtetes Mittel: laute Segmente repräsentieren Musikinhalt stärker.
        # Konservative Untergrenze: 20 % Gewicht auf Minimum verhindert dass ein schlechtes
        # Segment komplett ignoriert wird (Qualitätskollapsse sind noch sichtbar).
        _rms_weights = [max(r, 1e-10) for r in _active_rms]
        _total_w = sum(_rms_weights)
        _weighted_mean = sum(s * w for s, w in zip(_active_scores, _rms_weights)) / _total_w
        _min_score = min(_active_scores)
        return float(np.clip(0.20 * _min_score + 0.80 * _weighted_mean, 1.0, 5.0))

    def _adaptive_convergence_delta(self, current_mos: float) -> float:
        """Adaptive convergence threshold based on current MOS level and §2.56 goal_weights.

        High-quality audio (MOS > 4.0) uses tighter delta to squeeze out
        remaining improvements. Low-quality uses relaxed delta to avoid
        wasting iterations on negligible gains.

        §2.54 Material-Adaptive Plateau (VERBOTEN: fixed _PLATEAU_THRESHOLD=0.005):
        Shellac produces ~0.002 improvements per iteration; a universal 0.005
        threshold causes early termination. Material-specific deltas applied:
            shellac/wax_cylinder: 0.002   (tiny gains are real progress)
            reel_tape/tape:       0.003
            cassette:             0.004
            vinyl/acetate_disc:   0.004
            mp3_low/streaming:    0.008
            cd_digital and above: 0.010
        Additionally: restorability < 40 → floor the delta at 0.002 (heavily
        degraded songs must not be stopped on marginally improving iterations).

        §2.56: P1/P2-heavy songs (naturalness/authenticity) get a tighter
        delta at high MOS to extract maximum perceptual quality on the
        goals that matter most. P4/P5-heavy songs remain at standard delta.
        """
        # §2.54 Material-adaptive base delta (plateau threshold)
        _mat = (self.material or "").lower()
        _MATERIAL_DELTA: dict[str, float] = {
            "shellac": 0.002,
            "wax_cylinder": 0.002,
            "acetate_disc": 0.003,
            "wire_recording": 0.002,
            "reel_tape": 0.003,
            "tape": 0.003,
            "cassette": 0.004,
            "vinyl": 0.004,
            "mp3_low": 0.008,
            "streaming": 0.008,
            "cd_digital": 0.010,
        }
        _mat_delta = _MATERIAL_DELTA.get(_mat, self.convergence_delta)

        # Restorability floor: heavily degraded songs (restorability < 40) use minimum delta
        if self.restorability_score < 40:
            _mat_delta = min(_mat_delta, 0.002)

        # MOS-level scaling on top of material base
        if current_mos >= 4.0:
            base_delta = max(1e-6, _mat_delta * 0.5)  # tighten at near-ceiling quality
        elif current_mos >= 3.5:
            base_delta = _mat_delta
        else:
            base_delta = min(_mat_delta * 3.0, 0.05)  # relax for poor-quality audio

        # §2.56: tighten convergence for P1/P2-dominant songs at high quality
        if current_mos >= 4.0 and isinstance(self.goal_weights, dict) and self.goal_weights:
            _P1P2_KEYS = ("natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation")
            _p1p2_vals = [self.goal_weights.get(k, 1.0) for k in _P1P2_KEYS]
            _p1p2_mean = float(sum(_p1p2_vals) / max(len(_p1p2_vals), 1))
            if _p1p2_mean > 1.1:
                # Tighten by up to 40% for strongly P1/P2-dominant songs
                _tighten = float(min(0.40, (_p1p2_mean - 1.0) * 0.40))
                base_delta = max(1e-8, base_delta * (1.0 - _tighten))

        return base_delta

    # -------------------------------------------------------------------
    # §2.54 Adaptive thresholds — material/restorability/defect-aware
    # -------------------------------------------------------------------
    _POOR_MATERIALS = frozenset(
        {
            "shellac",
            "wax_cylinder",
            "wire_recording",
            "acetate_disc",
        }
    )
    _ANALOG_MATERIALS = frozenset(
        {
            "vinyl",
            "tape",
            "reel_tape",
            "cassette",
            "minidisc",
        }
    )

    def _compute_adaptive_prune_threshold(self, is_restorative: bool) -> float:
        """§2.54 + §2.56: Material- and goal-importance-adaptive pruning threshold.

        Restorative phases on severely degraded material need much more
        lenient thresholds — their MOS-proxy drop is expected (removing
        energy that was defect, not content).

        §2.56: When P1/P2 goals (natuerlichkeit, authentizitaet, tonal_center,
        timbre_authentizitaet, artikulation) carry high weight for this song,
        we apply a *conservative bias*: the threshold is tightened (less negative)
        so that phases that improve these critical goals are less likely to be
        pruned based on an incomplete MOS proxy.  Conversely, a P4/P5-dominated
        profile (brillanz, raumtiefe) loosens the threshold because minor MOS
        fluctuations there are tolerable.

        Returns a negative float (more negative = more lenient).
        """
        # Base: -0.01 enhancement, -0.05 restorative (legacy fallback)
        base = -0.05 if is_restorative else -0.01

        # Material factor: poor materials get 2x–3x more lenient
        mat = self.material.lower() if self.material else "unknown"
        if mat in self._POOR_MATERIALS:
            mat_factor = 3.0
        elif mat in self._ANALOG_MATERIALS:
            mat_factor = 2.0
        else:
            mat_factor = 1.0

        # Restorability factor: lower restorability → more lenient
        # restorability_score 0–100: 0=pristine, 100=heavily degraded
        rest_factor = 1.0 + (self.restorability_score / 100.0) * 1.5  # [1.0, 2.5]

        # Defect severity: higher → more lenient
        sev_factor = 1.0 + self.defect_severity_mean * 1.0  # [1.0, 2.0]

        # Combined: base * max(factors) — use max to avoid over-compounding
        adaptive = base * max(mat_factor, rest_factor, sev_factor)

        # §2.56 Goal-importance bias: P1/P2 heavy → tighten threshold (conservative).
        # A song where naturalness/authenticity matter most should not aggressively prune
        # phases that might be nudging those delicate goals in the right direction.
        _gw_bias = 0.0
        if isinstance(self.goal_weights, dict) and self.goal_weights:
            _P1P2_KEYS = (
                "natuerlichkeit",
                "authentizitaet",
                "tonal_center",
                "timbre_authentizitaet",
                "artikulation",
            )
            _P4P5_KEYS = (
                "brillanz",
                "spatial_depth",
                "waerme",
                "bass_kraft",
            )
            _p1p2_vals = [self.goal_weights.get(k, 1.0) for k in _P1P2_KEYS]
            _p4p5_vals = [self.goal_weights.get(k, 1.0) for k in _P4P5_KEYS]
            _p1p2_mean = float(sum(_p1p2_vals) / max(len(_p1p2_vals), 1))
            _p4p5_mean = float(sum(_p4p5_vals) / max(len(_p4p5_vals), 1))
            # bias ∈ [-0.05, +0.05]: positive bias = tighten (less pruning for P1/P2 songs)
            _gw_bias = float((_p1p2_mean - _p4p5_mean) * 0.025)
            _gw_bias = float(max(-0.05, min(0.05, _gw_bias)))

        # Apply bias: tighten (toward 0) when P1/P2 heavy, loosen when P4/P5 heavy.
        adaptive_biased = adaptive + _gw_bias
        # Clamp: never more lenient than -0.30, never stricter than -0.005
        return float(max(-0.30, min(-0.005, adaptive_biased)))

    def _compute_adaptive_mos_regression_tolerance(self) -> float:
        """§2.54 + §2.56: Material- and goal-importance-adaptive MOS regression tolerance.

        Poor material with heavy defects needs more tolerance — each
        iteration may transiently worsen MOS as it repairs deeper damage.

        §2.56: Songs with dominant P1/P2 goals get a small tolerance reduction
        to prevent accepting spurious regressions on critical perceptual goals.

        Returns a positive float (higher = more tolerant).
        """
        # Base: 0.05 (legacy fallback)
        base = 0.05

        mat = self.material.lower() if self.material else "unknown"
        if mat in self._POOR_MATERIALS:
            mat_bonus = 0.10  # shellac/wax allow up to 0.15 regression
        elif mat in self._ANALOG_MATERIALS:
            mat_bonus = 0.05  # vinyl/tape allow up to 0.10
        else:
            mat_bonus = 0.0

        # Higher restorability (more degraded) → more tolerance
        rest_bonus = (self.restorability_score / 100.0) * 0.08  # up to +0.08

        # Higher defect severity → more tolerance
        sev_bonus = self.defect_severity_mean * 0.05  # up to +0.05

        tolerance = base + max(mat_bonus, rest_bonus, sev_bonus)

        # §2.56: P1/P2 heavy songs → tighten tolerance slightly (max -0.015)
        # so we don't accept regressions in the most perceptually critical goals.
        if isinstance(self.goal_weights, dict) and self.goal_weights:
            _P1P2_KEYS = ("natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation")
            _p1p2_vals = [self.goal_weights.get(k, 1.0) for k in _P1P2_KEYS]
            _p1p2_mean = float(sum(_p1p2_vals) / max(len(_p1p2_vals), 1))
            if _p1p2_mean > 1.0:
                # Over-weighted P1/P2 → reduce tolerance proportionally, capped at -0.015
                _p1p2_reduction = float(min(0.015, (_p1p2_mean - 1.0) * 0.01))
                tolerance = max(base, tolerance - _p1p2_reduction)

        # Clamp: [0.03, 0.25] — never allow unlimited regression
        return float(np.clip(tolerance, 0.03, 0.25))

    def _filter_phases_by_hoerordnung_tiers(
        self,
        active_phases: list,
        goal_scores: dict,
    ) -> list:
        """§Hörordnung-Pre-Filter: lexikografische Ziel-Stufen vor der Messung.

        Regel (hoerordnung.instructions.md §5): Solange ein Goal der niedrigsten
        Defizit-Stufe unter Ziel liegt, werden FC-Phasen übersprungen, die
        ausschließlich höhere Stufen adressieren und KEIN Defizit-Goal
        direkt bedienen (Teamwork statt Dominanz). Phasen mit unbekannter ID
        oder unbekannten Goals bleiben erhalten (konservativ). GPP/Wohlklang-
        Ordnung-Gate prüfen danach weiterhin autoritativ.
        """
        if not goal_scores:
            return active_phases
        try:
            from backend.core.goal_priority_protocol import GoalPriorityProtocol as _TierGPP

            _tier_gpp = _TierGPP()
            _thr_map = getattr(self, "adaptive_goal_thresholds", None)
            _thr = _thr_map if isinstance(_thr_map, dict) and _thr_map else {}
            _deficits = [
                str(g)
                for g, v in goal_scores.items()
                if _thr.get(str(g)) is not None and float(v) < float(_thr[str(g)])
            ]
            if not _deficits:
                return active_phases
            _lowest_tier = min(int(_tier_gpp.hearing_tier(g)) for g in _deficits)

            kept: list = []
            dropped: list[int] = []
            for (_pid, _fn, _kw) in active_phases:
                _goals = self.FC_PHASE_PRIMARY_GOALS.get(int(_pid))
                if not _goals:
                    kept.append((_pid, _fn, _kw))  # unbekannt → konservativ behalten
                    continue
                _targets_deficit = any(g in _deficits for g in _goals)
                _min_tier = min(int(_tier_gpp.hearing_tier(g)) for g in _goals)
                if _targets_deficit or _min_tier <= _lowest_tier:
                    kept.append((_pid, _fn, _kw))
                else:
                    dropped.append(int(_pid))
            if dropped and kept:
                logger.info(
                    "FeedbackChain §Hörordnung-Pre-Filter: %s übersprungen (Ziel-Stufe > niedrigste Defizit-Stufe %d)",
                    sorted(dropped),
                    _lowest_tier,
                )
            return kept if kept else active_phases
        except Exception as _tier_exc:
            logger.debug("FeedbackChain §Hörordnung-Pre-Filter nicht verfügbar: %s", _tier_exc)
            return active_phases

    def run(
        self,
        audio: np.ndarray,
        phases_or_fn: Callable[[np.ndarray, int], np.ndarray] | list,
        sr: int | None = None,
        ceiling: float | None = None,
    ) -> FeedbackChainResult:
        """Führt die Feedback-Schleife aus.

        Akzeptiert zwei Aufruf-Varianten:
          - run(audio, improve_fn, sr)           – klassisch
          - run(audio, [(phase_id, fn, kwargs)]) – Phasen-Listen-Modus
        """
        _sr = sr if sr is not None else self.sample_rate
        assert _sr == 48000, f"FeedbackChain.run() erwartet SR=48000, erhalten: {_sr}"

        # §0p: Original-Audio für VQI-Dual-Objective sichern (einmalig, vor allen Iterationen)
        if self.panns_singing >= 0.35:
            try:
                self._vqi_orig_audio = np.asarray(audio, dtype=np.float32).copy()
            except Exception as _vqi_exc:
                logger.debug("FeedbackChain: VQI orig-audio Erfassung fehlgeschlagen: %s", _vqi_exc)

        # --- Adaptive Per-Phase Pruning for phase-list mode ---
        # In the first iteration, evaluate each phase individually.
        # Phases that degrade MOS (Δ < -0.01) are pruned from subsequent iterations.
        # This prevents a harmful phase from cancelling gains of helpful ones.
        _phase_list_mode = isinstance(phases_or_fn, list)
        _active_phases: list
        if isinstance(phases_or_fn, list):
            _active_phases = list(phases_or_fn)
        else:
            _active_phases = []
        _pruned_phases: list[str] = []
        _phase_deltas: dict[str, float] = {}  # phase_id → MOS delta from first iteration

        # ── §Hebel5: Sub-JND pre-pruning — phases already known sub-threshold from UV3 ────
        if _phase_list_mode and self.pre_pruned_phase_ids:
            _pre_prune_count = 0
            _filtered: list = []
            for _entry in _active_phases:
                _pid_str = str(_entry[0])
                if any(_pid_str.startswith(_ppid) for _ppid in self.pre_pruned_phase_ids):
                    _pruned_phases.append(_pid_str)
                    _pre_prune_count += 1
                    logger.debug("FeedbackChain §Hebel5: pre-pruned sub-JND Verarbeitungsschritt %s", _pid_str)
                else:
                    _filtered.append(_entry)
            if _pre_prune_count:
                _active_phases = _filtered
                logger.info(
                    "FeedbackChain §Hebel5: %d sub-JND phases pre-pruned (from UV3 metadata)",
                    _pre_prune_count,
                )

        # ── §2.47 GP-Advisory Strength Lookup (§Hebel1: propose_pareto statt propose) ──
        # Consult GP memory for material-genre-specific strength priors before
        # running the loop.  Uses propose_pareto() (MOO, §2.5) — falls back to
        # legacy propose() when insufficient MOO-data exists (< n_init entries).
        _gp_advisory_applied = False
        if _phase_list_mode and len(_active_phases) > 0:
            try:
                from backend.core.gp_parameter_optimizer import (  # pylint: disable=import-outside-toplevel
                    get_optimizer as _get_gp_opt,
                )

                _gp_opt = _get_gp_opt()
                _mat = self.material if self.material != "auto" else "unknown"
                # §Hebel1: MOO Pareto-Proposals für alle 15 Goals simultan
                _pareto_prop_list = None
                try:
                    _pareto_prop_list = _gp_opt.propose_pareto(material=_mat, n_init=5)
                except Exception as _pareto_not_avail:
                    logger.debug(
                        "FeedbackChain §Hebel1: propose_pareto nicht verfügbar, Legacy-Ersatzpfad: %s",
                        _pareto_not_avail,
                    )
                # Wähle besten Pareto-Kandidaten (höchster UCB-Score), fallback auf propose()
                _proposal = None
                if _pareto_prop_list:
                    _proposal = _pareto_prop_list[0]  # crowding-distance best
                    logger.debug(
                        "FeedbackChain §Hebel1: propose_pareto liefert %d Kandidaten, nehme besten",
                        len(_pareto_prop_list),
                    )
                else:
                    _proposal = _gp_opt.propose(material=_mat, n_init=5)
                if _proposal is not None and hasattr(_proposal, "params") and _proposal.params:
                    _gp_proposal = dict(_proposal.params)
                    _strength_keys = {
                        "noise_reduction_strength": ("phase_03",),
                        "reverb_reduction_strength": ("phase_49", "phase_20"),
                        "eq_correction_strength": ("phase_04", "phase_06"),
                        "harmonic_preservation": ("phase_07", "phase_08"),
                        "transient_strength": ("phase_08",),
                    }
                    _hints_applied = 0
                    for gp_key, phase_prefixes in _strength_keys.items():
                        if gp_key in _gp_proposal:
                            gp_val = float(np.clip(_gp_proposal[gp_key], 0.1, 1.0))
                            for idx, (_pid, _fn, _kw) in enumerate(_active_phases):
                                pid_str = str(_pid)
                                if any(pid_str.startswith(pp) for pp in phase_prefixes):
                                    if "strength" not in (_kw or {}):
                                        _kw_new = dict(_kw) if _kw else {}
                                        _kw_new["gp_advisory_strength"] = gp_val
                                        _active_phases[idx] = (_pid, _fn, _kw_new)
                                        _hints_applied += 1
                    if _hints_applied > 0:
                        _gp_advisory_applied = True
                        logger.info(
                            "FeedbackChain: GP advisory angewendet %d strength hints (material=%s, pareto=%s)",
                            _hints_applied,
                            _mat,
                            bool(_pareto_prop_list),
                        )
            except Exception as _gp_exc:
                logger.debug("FeedbackChain: GP advisory lookup nicht blockierend: %s", _gp_exc)

        if _phase_list_mode:

            def _build_combined_fn(active_phase_list: list):
                """Erstellt improve_fn from currently active phases."""

                def _combined_fn(a: np.ndarray, _sr2: int) -> np.ndarray:
                    out = a
                    for _pid, _fn, _kw in active_phase_list:
                        try:
                            out = _fn(out, _sr2, **_kw) if _kw else _fn(out, _sr2)
                        except Exception as phase_exc:
                            logger.debug(
                                "FeedbackChain: Verarbeitungsschritt callable fehlgeschlagen (%s): %s",
                                _pid,
                                phase_exc,
                            )
                    return out

                return _combined_fn

            improve_fn: Callable[[np.ndarray, int], np.ndarray] = _build_combined_fn(_active_phases)
        else:
            improve_fn = phases_or_fn  # type: ignore[assignment]

        _t0 = time.perf_counter()
        _hard_deadline = _t0 + self.max_runtime_s

        current = np.nan_to_num(np.asarray(audio, dtype=np.float32))

        # §Performance-Budget (copilot-instructions.md, Tabelle „pro Minute Audio“):
        # FeedbackChain (alle Iterationen) ≤ 120 s pro Minute Audio.
        # Für 30-s-Fragmente: 60 s. Floor 60 s für Kurzfragmente (dokumentierte
        # Abweichung: darunter kann kein vollständiger Phasen-Durchlauf stattfinden).
        _audio_dur_s = float(max(current.shape) if current.ndim == 2 else len(current)) / float(_sr)
        _time_budget_s = max(60.0, (_audio_dur_s / 60.0) * 120.0)
        # §performance-hard-stop: even with long or pathological audio, the feedback loop
        # must not run for hours. The adaptive budget is still used, but a strict ceiling
        # prevents runaway iterations in the deep restore path.
        _time_budget_s = min(_time_budget_s, self.max_runtime_s)
        best = current.copy()
        _t_before_init_score = time.perf_counter()
        best_mos = self._compute_iteration_score(best, _sr)
        _init_score_elapsed = time.perf_counter() - _t_before_init_score
        if _init_score_elapsed > 30.0:
            logger.warning(
                "FeedbackChain: initial Wert call took %.1fs (audio=%.0fs) — "
                "likely ML scorer without length cap; iterations will be uebersprungen if Grenze exhausted",
                _init_score_elapsed,
                _audio_dur_s,
            )
        history = [best_mos]
        _score_sources = [self._last_score_source]
        _ceiling_reached = False

        # §2.34 GoalPriorityProtocol — Stufe-1/2-Regression löst sofortigen Rollback aus
        _gpp = None
        try:
            from backend.core.goal_priority_protocol import (  # pylint: disable=import-outside-toplevel
                GoalPriorityProtocol,
            )

            _gpp = GoalPriorityProtocol()
        except Exception as gpp_exc:
            # §0 Primum non nocere: P1/P2-Schutz ist RELEASE_MUST. Fehler = strukturelles Problem.
            logger.warning(
                "§2.34 GoalPriorityProtocol NICHT VERFÜGBAR — P1/P2-Schutz deaktiviert (§0-Risiko): %s",
                gpp_exc,
            )

        _prev_goals: dict[str, float] = {}
        # §Hörordnung-Pre-Filter (Punkt 3, 2026-09-07): UV3 kann eine DSP-only
        # Baseline (UnifiedRestorerV3._fast_goal_snapshot) injizieren — damit
        # wirkt der Tier-Filter schon vor Iteration 1 statt erst nach der
        # ersten (teuren) Messung. Ohne Injektion: Filter ab Iteration 2 aktiv.
        _filter_goal_baseline: dict[str, float] = {}
        try:
            _inj_baseline = getattr(self, "baseline_goals", None)
            if isinstance(_inj_baseline, dict) and _inj_baseline:
                _filter_goal_baseline = {str(k): float(v) for k, v in _inj_baseline.items()}
        except Exception as _bl_exc:
            logger.debug("FeedbackChain baseline_goals nicht nutzbar: %s", _bl_exc)
        _goal_priority_log: list[str] = []
        _phase_executions: list[dict] = []

        # Max audio window for goal regression checks — 30 s is sufficient to
        # detect P1/P2 regressions; measuring the full signal wastes CPU budget.
        _GOAL_WINDOW_SAMPLES = int(_sr * 30.0)

        def _goal_window(a: np.ndarray) -> np.ndarray:
            """Gibt a centre-slice ≤ 30 s for goal measurement zurück."""
            total = a.shape[-1] if a.ndim == 2 else len(a)
            if total <= _GOAL_WINDOW_SAMPLES:
                return a
            start = (total - _GOAL_WINDOW_SAMPLES) // 2
            return (
                a[..., start : start + _GOAL_WINDOW_SAMPLES] if a.ndim == 2 else a[start : start + _GOAL_WINDOW_SAMPLES]
            )

        # §9.8 Goal-vector candidate selection — track how many goals pass thresholds
        _best_goal_pass_count: int = -1  # -1 = not yet measured
        _curr_goal_pass_count: int = -1

        converged = False
        for i in range(1, self.max_iterations + 1):
            # §Performance-Budget: abort if time budget exceeded
            _elapsed = time.perf_counter() - _t0
            if _elapsed >= self.max_runtime_s or _elapsed >= _time_budget_s:
                logger.warning(
                    "FeedbackChain: time Grenze exceeded (%.1fs >= %.1fs) — aborting at iteration %d",
                    _elapsed,
                    self.max_runtime_s,
                    i,
                )
                break

            # --- Adaptive Pruning: after iteration 1, evaluate each phase individually ---
            # On the first iteration, run all phases as a bundle to get the combined effect.
            # Then measure each phase's individual contribution and prune harmful ones.
            if _phase_list_mode and i == 2 and len(_active_phases) > 1:
                _pre_prune_audio = current.copy()
                # §Perf: reuse already-computed iter i-1 score (same audio, deterministic scorer)
                _base_mos = history[-1]
                # §Goal-deficit retention: use iter-1 goal scores to make pruning more lenient
                # when goals are still below their adaptive thresholds. This prevents pruning
                # phases that have small MOS impact but are critical for specific goal metrics
                # (e.g., a phase targeting bass_kraft or raumtiefe may have near-zero MOS delta
                # while being the only FC phase nudging that goal above threshold).
                _fc_deficit_factor = 1.0
                if _prev_goals and isinstance(self.adaptive_goal_thresholds, dict) and self.adaptive_goal_thresholds:
                    _agt_local: dict[str, float] = self.adaptive_goal_thresholds or {}
                    _deficits = [
                        float(_agt_local[_dg] - _prev_goals[_dg])
                        for _dg in _prev_goals
                        if _dg in _agt_local and _prev_goals[_dg] < _agt_local[_dg]
                    ]
                    if _deficits:
                        _max_deficit = max(_deficits)
                        # Scale up to 2× more lenient: deficit=0.05→factor 1.20; 0.25→2.0 (cap)
                        _fc_deficit_factor = float(min(2.0, 1.0 + 4.0 * _max_deficit))
                        logger.info(
                            "FeedbackChain §goal-deficit: %d/%d goals below Schwelle "
                            "(max_deficit=%.3f) → pruning %.2f× more lenient",
                            len(_deficits),
                            len(_prev_goals),
                            _max_deficit,
                            _fc_deficit_factor,
                        )
                # §Perf: import _RESTORATIVE_PHASES once before the per-phase loop (not per-phase)
                try:
                    from backend.core.per_phase_musical_goals_gate import (  # pylint: disable=import-outside-toplevel
                        _RESTORATIVE_PHASES as _RP_SET,
                    )
                except Exception:
                    _RP_SET = frozenset(
                        (
                            "phase_01",
                            "phase_02",
                            "phase_03",
                            "phase_05",
                            "phase_09",
                            "phase_12",
                            "phase_18",
                            "phase_20",
                            "phase_23",
                            "phase_24",
                            "phase_27",
                            "phase_28",
                            "phase_29",
                            "phase_30",
                            "phase_49",
                            "phase_50",
                            "phase_55",
                            "phase_56",
                            "phase_57",
                        )
                    )
                _surviving_phases = []
                for _pid, _fn, _kw in _active_phases:
                    try:
                        _test_out = _fn(_pre_prune_audio, _sr, **_kw) if _kw else _fn(_pre_prune_audio, _sr)
                        _test_out = np.clip(np.nan_to_num(np.asarray(_test_out, dtype=np.float32)), -1.0, 1.0)
                        _test_mos = self._compute_iteration_score(_test_out, _sr)
                        _delta = _test_mos - _base_mos
                        _phase_deltas[str(_pid)] = float(_delta)
                        # §2.54: Restorative phases (denoise, click, dropout) intentionally
                        # remove energy → MOS proxy may drop slightly vs. defect-laden
                        # reference. Use material-adaptive threshold (§2.54) to avoid
                        # pruning legitimate carrier-repair phases.
                        # §2.55 Single Source of Truth: _RESTORATIVE_PHASES aus PMGG-Ontologie,
                        # nicht hardcodiert — neue Phasen werden automatisch erkannt.
                        _is_restorative = any(str(_pid).startswith(rp) for rp in _RP_SET)
                        _prune_threshold = self._compute_adaptive_prune_threshold(_is_restorative)
                        # §Goal-deficit leniency: scale threshold when goals are below target.
                        # A phase that slightly reduces MOS but is the only one targeting a
                        # deficit goal should not be pruned — MOS proxy doesn't capture all goals.
                        if _fc_deficit_factor > 1.0:
                            _prune_threshold = float(np.clip(_prune_threshold * _fc_deficit_factor, -0.30, -0.005))
                        if _delta >= _prune_threshold:
                            # §2.56 FeedbackChain Strength-Adaptation:
                            # Phasen mit klar positivem Delta (> 0.01) werden leicht geboostet;
                            # marginal hilfreiche Nicht-Restaurierungs-Phasen leicht gedämpft.
                            # Nur aktiv wenn Phase einen expliziten 'strength'-Kwarg hat.
                            _kw_adapted = dict(_kw) if _kw else {}
                            if "strength" in _kw_adapted:
                                _cur_str = float(_kw_adapted["strength"])
                                if _delta > 0.010:
                                    # Clearly beneficial: modest boost (capped at 1.0)
                                    _kw_adapted["strength"] = float(np.clip(_cur_str * 1.12, 0.1, 1.0))
                                elif _delta < 0.002 and not _is_restorative:
                                    # Marginally helpful: slight reduction to prevent over-processing
                                    _kw_adapted["strength"] = float(np.clip(_cur_str * 0.90, 0.1, 1.0))
                            _surviving_phases.append((_pid, _fn, _kw_adapted))
                            logger.debug(
                                "FeedbackChain: Verarbeitungsschritt %s kept (Δ=%.4f)",
                                _pid,
                                _delta,
                            )
                        else:
                            _pruned_phases.append(str(_pid))
                            logger.info(
                                "FeedbackChain: Verarbeitungsschritt %s pruned — degraded MOS by %.4f",
                                _pid,
                                _delta,
                            )
                    except Exception as _eval_exc:
                        _surviving_phases.append((_pid, _fn, _kw))
                        logger.debug(
                            "FeedbackChain: Verarbeitungsschritt %s evaluation fehlgeschlagen (%s) — keeping",
                            _pid,
                            _eval_exc,
                        )
                if _surviving_phases and len(_surviving_phases) < len(_active_phases):
                    _active_phases = _surviving_phases
                    improve_fn = _build_combined_fn(_active_phases)
                    logger.info(
                        "FeedbackChain: pruned %d/%d phases, %d remaining",
                        len(_pruned_phases),
                        len(_pruned_phases) + len(_active_phases),
                        len(_active_phases),
                    )
                elif not _surviving_phases:
                    logger.info("FeedbackChain: all phases degrade quality — converging early")
                    converged = True
                    break

            # §Hörordnung-Pre-Filter (Punkt 3, 2026-09-07): Vor der
            # Kandidaten-Konstruktion Phasen mit Ziel-Stufe über der niedrigsten
            # Defizit-Stufe überspringen (lexikografische Ordnung,
            # hoerordnung.instructions.md §5). GPP bleibt autoritativ.
            if _phase_list_mode and _active_phases:
                _tier_scores = _prev_goals if _prev_goals else _filter_goal_baseline
                if isinstance(_tier_scores, dict) and _tier_scores:
                    _active_phases = self._filter_phases_by_hoerordnung_tiers(_active_phases, _tier_scores)
                    improve_fn = _build_combined_fn(_active_phases)

            candidate = improve_fn(current, _sr)
            candidate = np.clip(np.nan_to_num(np.asarray(candidate, dtype=np.float32)), -1.0, 1.0)
            mos = self._compute_iteration_score(candidate, _sr)
            history.append(mos)
            _score_sources.append(self._last_score_source)
            _phase_executions.append({"iteration": i, "mos": float(mos)})

            # Optionaler externer Priority-Callback (z.B. aus UnifiedRestorerV3).
            if callable(self.goal_priority_callback):
                try:
                    _cb_abort, _cb_reason = self.goal_priority_callback(current, candidate)  # pylint: disable=not-callable
                    if _cb_abort:
                        _log_entry = (
                            f"FeedbackChain Iteration {i} abgebrochen: {_cb_reason or 'goal-priority callback'}"
                        )
                        _goal_priority_log.append(_log_entry)
                        logger.warning("⚠ %s", _log_entry)
                        break
                except Exception as _cb_exc:
                    logger.debug("FeedbackChain goal_priority_callback fehlgeschlagen: %s", _cb_exc)

            # §2.34 GoalPriorityProtocol: Stufe-1/2-Ziele schützen
            # Skip internal GPP check when external goal_priority_callback is wired
            # (UV3 provides its own GPP callback that already calls measure_all).
            # §v10.101/D3: JND-validierter Iterations-Abbruch — stoppt wenn Verbesserung unhörbar.
            if _gpp is not None and _prev_goals and not callable(self.goal_priority_callback):
                try:
                    from backend.core.musical_goals.musical_goals_metrics import (  # pylint: disable=import-outside-toplevel
                        get_checker,
                    )

                    _checker = get_checker()
                    _t_goals = time.perf_counter()
                    _curr_goals = _checker.measure_all(_goal_window(candidate), _sr)
                    _analytics_dt = time.perf_counter() - _t_goals
                    self._last_analytics_overhead_s = getattr(self, "_last_analytics_overhead_s", 0.0) + _analytics_dt
                    abort_result = _gpp.should_abort_iteration(_prev_goals, _curr_goals, goal_weights=self.goal_weights)
                    if abort_result.should_abort:
                        _log_entry = f"FeedbackChain Iteration {i} abgebrochen: {abort_result.reason}"
                        _goal_priority_log.append(_log_entry)
                        logger.warning("⚠ %s", _log_entry)
                        break  # Rollback auf best (§2.34)
                    _prev_goals = _curr_goals
                    # §09.2: use song-adaptive targets if available, else canonical thresholds
                    _fc_agt = self.adaptive_goal_thresholds
                    _curr_goal_pass_count = sum(
                        1
                        for g, v in _curr_goals.items()
                        if v
                        >= (
                            _fc_agt.get(g, _checker.thresholds.get(g, 0.85))
                            if _fc_agt
                            else _checker.thresholds.get(g, 0.85)
                        )
                    )
                    # Hörordnung Ebene 3 (hoerordnung.instructions.md §5): strikte
                    # lexikografische Dominanz. Ein Kandidat, der ein Ziel einer
                    # höheren Hörstufe (kleinere Zahl) senkt, während nur Ziele
                    # niedrigerer Stufen gewinnen, wird NICHT akzeptiert — kein
                    # Brillanz-Transparenz-Boost auf Kosten von Wärme/Natürlichkeit.
                    _ho_dropped_tier = 99
                    _ho_gained_tier = 99
                    for _g_ho, _v_prev in _prev_goals.items():
                        _v_curr = float(_curr_goals.get(_g_ho, _v_prev))
                        _d_ho = _v_curr - float(_v_prev)
                        _t_ho = int(_gpp.hearing_tier(str(_g_ho)))
                        if _d_ho < -_gpp.REGRESSION_EPSILON:
                            _ho_dropped_tier = min(_ho_dropped_tier, _t_ho)
                        elif _d_ho > _gpp.REGRESSION_EPSILON:
                            _ho_gained_tier = min(_ho_gained_tier, _t_ho)
                    # Hörordnung Ebene 3 Audit (hoerordnung.instructions.md §5/§8):
                    # Maschinelle Dominanz-Verifikation via WohlklangOrdnungGate
                    # (Reihenfolge recycelt aus GoalPriorityProtocol, keine neuen
                    # Schwellwerte). Ergebnis für UV3-Metadaten/GUI-Ampel.
                    try:
                        from backend.core.wohlklang_ordnung_gate import (  # pylint: disable=import-outside-toplevel
                            WohlklangOrdnungGate as _WOGate,
                        )

                        _wo_deltas = {
                            str(_g_ho): float(_curr_goals.get(_g_ho, _v_prev)) - float(_v_prev)
                            for _g_ho, _v_prev in _prev_goals.items()
                        }
                        self.last_wohlklang_audit = _WOGate().evaluate(_wo_deltas).to_dict()
                    except Exception as _wo_exc:
                        logger.debug("WohlklangOrdnungGate in FeedbackChain nicht verfügbar: %s", _wo_exc)
                    if _ho_dropped_tier < _ho_gained_tier:
                        _ho_entry = (
                            f"FeedbackChain Iteration {i}: Hörordnungs-Verstoß "
                            f"(Senkung Stufe {_ho_dropped_tier} gegen Gewinn Stufe {_ho_gained_tier}) "
                            "— Kandidat verworfen (§Hörordnung Ebene 3)"
                        )
                        _goal_priority_log.append(_ho_entry)
                        logger.warning("⚠ %s", _ho_entry)
                        continue
                except Exception as _gpp_exc:
                    logger.debug("GoalPriorityProtocol in FeedbackChain nicht verfügbar: %s", _gpp_exc)
            elif _gpp is not None and not _prev_goals and not callable(self.goal_priority_callback):
                try:
                    from backend.core.musical_goals.musical_goals_metrics import (  # pylint: disable=import-outside-toplevel
                        get_checker,
                    )

                    _checker = get_checker()
                    _t_goals = time.perf_counter()
                    _prev_goals = _checker.measure_all(_goal_window(candidate), _sr)
                    _analytics_dt = time.perf_counter() - _t_goals
                    self._last_analytics_overhead_s = getattr(self, "_last_analytics_overhead_s", 0.0) + _analytics_dt
                    # §09.2: use song-adaptive targets if available, else canonical thresholds
                    _fc_agt = self.adaptive_goal_thresholds
                    _curr_goal_pass_count = sum(
                        1
                        for g, v in _prev_goals.items()
                        if v
                        >= (
                            _fc_agt.get(g, _checker.thresholds.get(g, 0.85))
                            if _fc_agt
                            else _checker.thresholds.get(g, 0.85)
                        )
                    )
                except Exception as mg_exc:
                    logger.debug("FeedbackChain: initial musical-goals read fehlgeschlagen: %s", mg_exc)

            # §9.8 Goal-aware candidate selection: prefer candidates passing more goals
            _candidate_better = mos > best_mos
            if _curr_goal_pass_count >= 0 and _best_goal_pass_count >= 0:
                if _curr_goal_pass_count > _best_goal_pass_count:
                    _candidate_better = True  # more goals passed → accept
                elif _curr_goal_pass_count < _best_goal_pass_count:
                    _candidate_better = mos > best_mos + 0.05  # need significant MOS gain
            if _candidate_better:
                best_mos = mos
                best = candidate.copy()
                if _curr_goal_pass_count >= 0:
                    _best_goal_pass_count = _curr_goal_pass_count

            # §2.33 PhysicalCeilingEstimator: Frühzeitiger Abbruch wenn Ceiling erreicht
            # Tight headroom: allow iterations to push closer to ceiling
            _adaptive_headroom = 0.01
            if ceiling is not None and best_mos >= ceiling - _adaptive_headroom:
                _ceiling_reached = True
                converged = True
                logger.debug(
                    "FeedbackChain: ceiling=%.3f reached (MOS=%.3f) — Frühzeitiger Abbruch",
                    ceiling,
                    best_mos,
                )
                break

            if abs(history[-1] - history[-2]) < self._adaptive_convergence_delta(best_mos):
                converged = True
                break

            # §2.54 Target-Score-Gate: Qualitätsziel erreicht → frühzeitig beenden
            # Verhindert unnötige Iterationen wenn das Ziel schon erfüllt ist.
            # Mapping: target_score [0,1] → MOS-Skala [1,5]
            _mos_target = 1.0 + self.target_score * 4.0
            if best_mos >= _mos_target:
                converged = True
                logger.info(
                    "FeedbackChain: Qualitätsziel erreicht (target=%.2f → MOS≥%.2f, erreicht=%.3f) nach %d Iterationen",
                    self.target_score,
                    _mos_target,
                    best_mos,
                    i,
                )
                break

            # §Hebel2: Oszillationsdetection — n und n-2 nahezu identisch → Loop oszilliert
            # JND-Schwelle: 0.010 MOS (kaum wahrnehmbar). Bei ≥ 3 Iterationen prüfbar.
            if len(history) >= 4:
                _osc_delta = abs(history[-1] - history[-3])  # Iteration n vs. n-2
                _jnd_osc = 0.010  # Minimale wahrnehmbare Differenz
                if _osc_delta < _jnd_osc:
                    logger.info(
                        "FeedbackChain §Hebel2: Oszillation erkannt (|iter%d - iter%d|=%.4f < JND=%.3f)"
                        " — beende mit bestem Checkpoint",
                        len(history) - 1,
                        len(history) - 3,
                        _osc_delta,
                        _jnd_osc,
                    )
                    converged = True
                    break

            # §2.54 Adaptive regression guard — material/restorability-aware
            _mos_regression_tol = self._compute_adaptive_mos_regression_tolerance()
            if history[-1] < history[-2] - _mos_regression_tol:
                break

            current = candidate

        _final_elapsed = min(float(time.perf_counter() - _t0), self.max_runtime_s)
        return FeedbackChainResult(
            audio=best,
            iterations=len(history) - 1,
            converged=converged,
            mos_history=history,
            metadata={
                "best_mos": best_mos,
                "goal_priority_log": _goal_priority_log,
                "score_source": _score_sources[-1] if _score_sources else self._last_score_source,
                "score_sources_seen": list(dict.fromkeys(_score_sources)),
                "score_fallback_used": bool(
                    (self.use_pqs_in_loop or self.use_versa_in_loop)
                    and any(src == "heuristic_rms" for src in _score_sources)
                ),
                "pruned_phases": _pruned_phases,
                "phase_deltas": _phase_deltas,
                "gp_advisory_applied": _gp_advisory_applied,
                "gp_pareto_used": bool(_gp_advisory_applied and locals().get("_pareto_prop_list")),
                "oscillation_stopped": bool(
                    converged
                    and len(history) >= 4
                    and abs(history[-1] - history[-3]) < 0.010
                    and len(history) - 1 < self.max_iterations
                ),
            },
            phase_executions=_phase_executions,
            overall_score=float(best_mos),
            total_retries=max(0, len(history) - 1),
            total_time_s=_final_elapsed,
            ceiling_reached=_ceiling_reached,
            analytics_overhead_s=float(getattr(self, "_last_analytics_overhead_s", 0.0)),
        )


_instance: FeedbackChain | None = None
_lock = threading.Lock()


def get_feedback_chain() -> FeedbackChain:
    """Gibt den globalen FeedbackChain-Singleton zurück (lazy init)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FeedbackChain()
    return _instance


def compute_perceptual_score(
    original: np.ndarray,
    degraded: np.ndarray,
    *,
    sample_rate: int = 48000,
) -> dict:
    """Berechnet Perceptual-Score-Dict für original vs. degraded Audio (Spec §2.6).

    Rückgabe-Schlüssel:
        sisnr_db       SI-SNR in dB (Scale-Invariant SNR)
        spectral_flatness  Spektrale Flachheit ∈ [0, 1]
        snr_db         SNR in dB
        transient_score    Hüllkurven-Korrelation ∈ [0, 1]
        combined       Gewichteter Gesamt-Score ∈ [0, 1]
    """
    _ = sample_rate  # wird für zukünftige SR-abhängige Metriken genutzt

    orig = np.nan_to_num(np.asarray(original, dtype=np.float32)).ravel()
    deg = np.nan_to_num(np.asarray(degraded, dtype=np.float32)).ravel()
    n = min(len(orig), len(deg))
    orig, deg = orig[:n], deg[:n]

    # — SI-SNR (Scale-Invariant SNR) ——————————————————————————————————————
    orig64 = orig.astype(np.float64)
    deg64 = deg.astype(np.float64)
    dot = float(np.dot(orig64, orig64)) + 1e-12
    s_target = (np.dot(deg64, orig64) / dot) * orig64
    e_noise = deg64 - s_target
    sisnr = 10.0 * float(np.log10((np.dot(s_target, s_target) + 1e-12) / (np.dot(e_noise, e_noise) + 1e-12)))

    # — SNR ——————————————————————————————————————————————————————————————
    signal_power = float(np.mean(orig64**2)) + 1e-12
    noise_power = float(np.mean((deg64 - orig64) ** 2)) + 1e-12
    raw_snr = 10.0 * np.log10(signal_power / noise_power)
    snr_db = float(np.nan_to_num(raw_snr, nan=0.0, posinf=60.0, neginf=-60.0))

    # — Spectral Flatness ——————————————————————————————————————————————
    n_fft = min(2048, max(4, len(deg) // 4))
    spec = np.abs(np.fft.rfft(deg, n=n_fft)) + 1e-12
    spectral_flatness = float(
        np.clip(
            np.exp(float(np.mean(np.log(spec)))) / (float(np.mean(spec)) + 1e-12),
            0.0,
            1.0,
        )
    )

    # — Transient Score (Hüllkurven-Korrelation) ——————————————————————
    hop = max(1, len(orig) // 200)
    env_o = np.array(
        [float(np.max(np.abs(orig[i : i + hop]))) for i in range(0, len(orig) - hop, hop)],
        dtype=np.float64,
    )
    env_d = np.array(
        [float(np.max(np.abs(deg[i : i + hop]))) for i in range(0, len(deg) - hop, hop)],
        dtype=np.float64,
    )
    ml = min(len(env_o), len(env_d))
    if ml > 1 and np.std(env_o[:ml]) > 1e-10 and np.std(env_d[:ml]) > 1e-10:
        _eo = env_o[:ml] - env_o[:ml].mean()
        _ed = env_d[:ml] - env_d[:ml].mean()
        _no = float(np.linalg.norm(_eo))
        _nd = float(np.linalg.norm(_ed))
        _raw_corr = float(np.dot(_eo, _ed) / (_no * _nd + 1e-10))
        transient_score = float(np.clip((_raw_corr + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(_raw_corr) else 0.5
    elif ml > 1 and np.std(env_o[:ml]) < 1e-10 and np.std(env_d[:ml]) < 1e-10:
        transient_score = 1.0  # Both silent — trivially matched
    else:
        transient_score = 0.5

    # — Spectral Correlation (Kosinus-Ähnlichkeit der Magnitude-Spektren) ——————
    spec_o = np.abs(np.fft.rfft(orig, n=n_fft)) + 1e-12
    _norm_so = float(np.linalg.norm(spec_o))
    _norm_sd = float(np.linalg.norm(spec))
    spectral_corr = float(np.clip(np.dot(spec_o, spec) / (_norm_so * _norm_sd + 1e-12), 0.0, 1.0))

    # — Combined ———————————————————————————————————————————————————————
    sisnr_norm = float(np.clip((sisnr + 20.0) / 80.0, 0.0, 1.0))
    combined = float(
        np.clip(
            0.4 * sisnr_norm + 0.3 * transient_score + 0.3 * (1.0 - spectral_flatness),
            0.0,
            1.0,
        )
    )

    return {
        "sisnr_db": float(sisnr),
        "spectral_flatness": spectral_flatness,
        "snr_db": snr_db,
        "transient_score": transient_score,
        "spectral_corr": spectral_corr,
        "combined": combined,
    }


# Convenience-Konstante: kritische Phasen-IDs (Spec §2.2 — TIER_1 + Dropout-Repair)
FEEDBACK_CRITICAL_PHASES: frozenset[int] = frozenset(
    {
        1,  # click_removal
        2,  # hum_removal
        3,  # denoise
        9,  # crackle_removal
        12,  # wow_flutter_fix
        24,  # dropout_repair
        29,  # tape_hiss_reduction
        55,  # diffusion_inpainting
    }
)

__all__ = [
    "DEFAULT_TARGET_SCORE",
    "EXCELLENCE_TARGET_SCORE",
    "FEEDBACK_CRITICAL_PHASES",
    "MUSIC_OVR_EXCELLENCE_THRESHOLD",
    "FeedbackChain",
    "FeedbackChainResult",
    "compute_perceptual_score",
    "get_feedback_chain",
]
