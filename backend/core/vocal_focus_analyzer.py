"""
VocalFocusAnalyzer (VFA) — §0p [RELEASE_MUST] Aurik 10.0.0
===========================================================

Aggregiert alle vokalspezifischen Analysen zu einem einzelnen VFAResult,
der in ``_restoration_context["vfa_result"]`` injiziert wird und allen
nachgelagerten Phasen zur Verfügung steht.

Läuft nach SongCalibration, vor GoalApplicabilityFilter (§0p / §0g).
Aktivierung: PANNs Singing ≥ 0.25.

Komponenten:
  - PANNs-Singing-Konfidenz (aus UV3 übergeben)
  - VocalRegisterDetector → dominant_register + energy_bias_db
  - FrissonCandidateDetector → frisson_zones
  - LPC-Formant-Tracker → f1_mean, f2_mean, formant_stable

Singleton-Pattern (thread-safe double-checked locking).
Performance: ≤ 3 s für 30 s Stereo (48 kHz) ohne ML-Modelle.

Spec: copilot-instructions.md §0p (v10.0.0)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from backend.core.dsp.vocal_register_detector import detect_vocal_register as _detect_vocal_register_fn
except Exception:
    _detect_vocal_register_fn = None  # type: ignore[assignment]

try:
    from backend.core.frisson_candidate_detector import get_frisson_detector as _get_frisson_detector_fn
except Exception:
    _get_frisson_detector_fn = None  # type: ignore[assignment]

try:
    from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker as _get_lpc_formant_tracker_fn
except Exception:
    _get_lpc_formant_tracker_fn = None  # type: ignore[assignment]

try:
    import librosa as _librosa
except Exception:
    _librosa = None  # type: ignore[assignment]

try:
    from backend.core.dsp.style_intent_detector import get_style_intent_detector as _get_style_intent_detector_fn
except Exception:
    _get_style_intent_detector_fn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VFAResult — Ergebnis-Container
# ---------------------------------------------------------------------------


@dataclass
class VFAResult:
    """Gebündeltes Ergebnis der Vokal-Fokus-Analyse.

    Alle Felder sind serialisierbar (dict-kompatibel für ``_restoration_context``).
    """

    panns_singing: float = 0.0
    """PANNs Singing-Konfidenz [0, 1]."""

    vocal_present: bool = False
    """True wenn panns_singing ≥ 0.25."""

    dominant_register: str = "chest"
    """Vokalregister: "head", "chest", "fry_whisper", "falsetto", "unknown"."""

    energy_bias_db: float = -6.0
    """Register-adaptiver Energy-Bias für NR-Algorithmen (dB, negativ)."""

    frisson_zones: list[tuple[float, float]] = field(default_factory=list)
    """Liste von (start_s, end_s) Klimax/Gänsehaut-Passagen."""

    formant_f1_mean: float = 0.0
    """Mittlere F1-Formantfrequenz (Hz); 0.0 wenn nicht ermittelbar."""

    formant_f2_mean: float = 0.0
    """Mittlere F2-Formantfrequenz (Hz); 0.0 wenn nicht ermittelbar."""

    formant_stable: bool = True
    """True wenn F1-Varianz gering (Opera/Klassik); False bei stark fluktuierenden Formanten."""

    passaggio_zones: list[tuple[float, float]] = field(default_factory=list)
    """Register-Übergangszonen (Brust→Kopf). Leer wenn nicht erkannt."""

    vibrato_zones: list[tuple[float, float]] = field(default_factory=list)
    """Vibrato-geschützte Passagen (F0-Modulation 4–7 Hz). Leer wenn nicht erkannt."""

    # §Gap1 Emotional Context — Tension/Release/Climax/Whisper (v10.0.0.x)
    tension_zones: list[tuple[float, float]] = field(default_factory=list)
    """Spannungs-Passagen (start_s, end_s): steigende Energie + hoher Spectral Centroid.
    DSP-Schutz: Strength-Reduktion in Tension-Zonen um 20 % (kein Over-NR vor Klimax)."""

    release_zones: list[tuple[float, float]] = field(default_factory=list)
    """Entspannungs-Passagen (start_s, end_s): nach Tension-Peak oder Frisson-Abfall.
    Schutz: Keine additive Verstärkung in Release-Zonen (§0p Emotionale Authentizität)."""

    whisper_zones: list[tuple[float, float]] = field(default_factory=list)
    """Flüster-Passagen (start_s, end_s): niedrige Energie + hohe Spectral Flatness.
    §0p Atemgeschützte Segmente: Spectral Flatness > 0.35, RMS < −35 dBFS."""

    climax_type: str = "none"
    """Klimax-Charakter: "none" | "peak" | "sustained" | "dynamic".
    Beeinflusst MDEM-Stärke-Entscheidung und frisson_zone-Schutz."""

    vqi_gate_active: bool = False
    """True wenn panns_singing ≥ 0.35 → VQI-Gate in HolisticPerceptualGate aktiv."""

    style_intent_zones: list[tuple[float, float]] = field(default_factory=list)
    """§P2 Style-Intent-Zonen (start_s, end_s): intentionale Pitch-Abweichungen (Blue Notes,
    Microtonal Bends etc.). Phase_31 + Phase_42 reduzieren dort ihre Stärke."""

    style_confidence: float = 0.0
    """§P2 Style-Intent-Konfidenz [0, 1]: Anteil intentionaler Voicing-Events / Gesamt-Voiced."""

    singer_school: str = "unknown"
    """Gesangsschule — klassifiziert aus F1/F2-Formanten, Vibrato, Register, Style-Konfidenz.
    Werte:
    - "classical"   Bel Canto/Oper: stabile Formanten, gleichmäßiges Vibrato, F1 < 600 Hz
    - "jazz"        Jazz-Standard: variable Formanten, Blue Notes (style_confidence > 0.15)
    - "soul_rnb"    Soul/R&B/Gospel: breite Vibrato-Kurve, Chest-Dominant, style_confidence > 0.20
    - "schlager"    Schlager/Chanson: stabile Formanten, enge Vokal-Räume, kein Style-Intent
    - "folk_country" Folk/Country: nasale Resonanz (hohes F1), geringe Vibrato-Dichte
    - "pop"         Pop/Crossover: minimales Vibrato, breitere Formant-Varianz
    - "unknown"     Nicht klassifizierbar (kein Gesang oder Formanten nicht messbar)
    """

    phoneme_protection_level: str = "standard"
    """Phonem-Schutz-Stufe (aus singer_school abgeleitet):
    - "strict"   Klassik/Oper: Formant-Rollback-Toleranz ±1 dB, alle Atemgeräusche
                 als Naturalness-Marker, kein aggressives De-Essing.
    - "standard" Jazz/Schlager/Soul: Spec-Standard ±2 dB (§0p).
    - "relaxed"  Pop/Rock: ±3 dB, weniger Vibrato-Schutz, normales De-Essing.
    """

    intonation_events: list = field(default_factory=list)
    """§Lücke1 IntonationEvents: Klassifizierte F0-Abweichungen (INTENTIONAL/DEGRADATION/AMBIGUOUS).
    Liste von `IntonationEvent`-Objekten aus `backend.core.dsp.intonation_classifier`.
    Pitch-Korrektur-geschützte Zonen: `[e for e in intonation_events if not e.pitch_correction_allowed]`.
    """

    breath_segments: list = field(default_factory=list)
    """§Lücke-F BreathEmotionClassifier: Klassifizierte Atemgeräusche.
    Liste von `BreathSegment`-Objekten aus `backend.core.dsp.breath_emotion_classifier`.
    EMOTIONAL_TENSION-Segmente sind Frisson-Vorboten → G_floor 0.85 in NR-Phasen.
    """

    analysis_duration_s: float = 0.0
    """Tatsächlich analysierte Segmentlänge (s)."""

    def to_dict(self) -> dict:
        """Serialisiert für ``_restoration_context``."""
        return {
            "panns_singing": self.panns_singing,
            "vocal_present": self.vocal_present,
            "dominant_register": self.dominant_register,
            "energy_bias_db": self.energy_bias_db,
            "frisson_zones": list(self.frisson_zones),
            "formant_f1_mean": self.formant_f1_mean,
            "formant_f2_mean": self.formant_f2_mean,
            "formant_stable": self.formant_stable,
            "passaggio_zones": list(self.passaggio_zones),
            "vibrato_zones": list(self.vibrato_zones),
            "tension_zones": list(self.tension_zones),
            "release_zones": list(self.release_zones),
            "whisper_zones": list(self.whisper_zones),
            "climax_type": self.climax_type,
            "vqi_gate_active": self.vqi_gate_active,
            "style_intent_zones": list(self.style_intent_zones),
            "style_confidence": self.style_confidence,
            "singer_school": self.singer_school,
            "phoneme_protection_level": self.phoneme_protection_level,
            "analysis_duration_s": self.analysis_duration_s,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: VocalFocusAnalyzer | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# VocalFocusAnalyzer
# ---------------------------------------------------------------------------


class VocalFocusAnalyzer:
    """Koordiniert alle Vokal-Analysen und gibt VFAResult zurück.

    Alle Teilanalysen sind non-blocking: Exceptions → sicherer Fallback.
    """

    # Analyse-Segment-Länge (s) — Balance zwischen Genauigkeit und Speed
    _ANALYSIS_MAX_S: float = 30.0
    _FORMANT_SEGMENT_S: float = 4.0

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing: float = 0.0,
        *,
        panns_tags: dict[str, float] | None = None,
        panns_vocals_confidence: float | None = None,
        is_schlager: bool = False,
        genre_label: str = "",
        vocal_material_prior: bool = False,
    ) -> VFAResult:
        """Vollständige Vokal-Fokus-Analyse.

        Args:
            audio:         Mono oder Stereo float32 (SR = sr).
            sr:            Abtastrate.
            panns_singing: PANNs-Singing-Konfidenz aus UV3 (bereits berechnet).

        Returns:
            VFAResult mit allen Vokal-Analysedaten.
        """
        _t0 = time.perf_counter()

        effective_panns = self._resolve_vocal_presence_confidence(
            panns_singing,
            panns_tags=panns_tags,
            panns_vocals_confidence=panns_vocals_confidence,
            is_schlager=is_schlager,
            genre_label=genre_label,
            vocal_material_prior=vocal_material_prior,
        )

        result = VFAResult(
            panns_singing=effective_panns,
            vocal_present=effective_panns >= 0.25,
            vqi_gate_active=effective_panns >= 0.35,
        )

        # Mono-Segment für Analyse (Zentrum, max. _ANALYSIS_MAX_S)
        mono = self._to_mono(audio)
        n_max = int(self._ANALYSIS_MAX_S * sr)
        if len(mono) > n_max:
            _start = (len(mono) - n_max) // 2
            mono_seg = mono[_start : _start + n_max]
        else:
            mono_seg = mono
        result.analysis_duration_s = len(mono_seg) / max(sr, 1)

        if not result.vocal_present:
            # Kein Gesang → Defaults, keine Analyse nötig
            return result

        # 1. Vokalregister + Energy-Bias
        result.dominant_register, result.energy_bias_db = self._detect_register(mono_seg, sr, effective_panns)

        # 2. Frisson-Zonen (non-blocking, max. 150 ms)
        result.frisson_zones = self._detect_frisson(audio, sr)

        # 3. Formant-Analyse (non-blocking, max. 500 ms)
        f1_mean, f2_mean, stable = self._analyze_formants(mono_seg, sr)
        result.formant_f1_mean = f1_mean
        result.formant_f2_mean = f2_mean
        # §Spec 24-Konsistenz (Befund 2026-08-22): f1=0Hz bei stable=True ist
        # widersprüchlich (Log: „formant_f1=0Hz stable=True“) — ohne gültige
        # F1-Schätzung gibt es keine Stabilitätsaussage.
        if f1_mean is not None and float(f1_mean) <= 0.0:
            stable = False
        result.formant_stable = stable

        # 4. Passaggio-Zonen (F0-basiert, optional — leert sich bei Fehler)
        result.passaggio_zones = self._detect_passaggio(mono_seg, sr)

        # 5. Vibrato-Zonen (F0-Modulation 4–7 Hz, §0p Vibrato-Schutz)
        result.vibrato_zones = self._detect_vibrato(mono_seg, sr)

        # 6. Tension-Zonen (steigende Energie + Spectral Centroid, §Gap1)
        result.tension_zones = self._detect_tension_zones(mono_seg, sr)

        # 7. Release-Zonen (nach Tension-Peak oder Frisson-Abfall, §Gap1)
        result.release_zones = self._detect_release_zones(mono_seg, sr, result.tension_zones, result.frisson_zones)

        # 8. Whisper-Zonen (niedrige Energie + hohe Spectral Flatness, §0p §Gap1)
        result.whisper_zones = self._detect_whisper_zones(mono_seg, sr)

        # 9. Klimax-Typ (beeinflusst MDEM + Frisson-Schutz, §Gap1)
        result.climax_type = self._detect_climax_type(result.frisson_zones, result.tension_zones, mono_seg, sr)

        # 10. Style-Intent-Zonen (§P2: intentionale Pitch-Abweichungen schützen)
        result.style_intent_zones, result.style_confidence = self._detect_style_intent(mono_seg, sr)

        # 11. Singer-School-Klassifikation (aus Formant/Vibrato/Register/Style, non-blocking)
        result.singer_school, result.phoneme_protection_level = self._classify_singer_school(result)

        # 12. §Lücke1 Intonations-Intentionality: F0-Kontur klassifizieren
        # Ergebnis in result.intonation_events → UV3 nutzt pitch_correction_allowed-Flag
        if result.vocal_present:
            try:
                from backend.core.dsp.intonation_classifier import (  # pylint: disable=import-outside-toplevel
                    classify_intonation_events,
                )

                _f0_contour = getattr(result, "_f0_contour_internal", None)
                if _f0_contour is not None and len(_f0_contour) >= 4:
                    result.intonation_events = classify_intonation_events(
                        f0_hz=_f0_contour,
                        sr=sr,
                        hop=512,
                    )
                    logger.debug(
                        "VFA step 12: %d intonation_events (%d protected zones)",
                        len(result.intonation_events),
                        sum(1 for e in result.intonation_events if not e.pitch_correction_allowed),
                    )
            except Exception as _ie_exc:
                logger.debug("VFA step 12 IntonationClassifier: nicht blockierend Ersatzpfad — %s", _ie_exc)

        # 13. §Lücke-F Breath Emotion Classification: Atemgeräusche kategorisieren
        # Emotionale Atemgeräusche (EMOTIONAL_TENSION) sind Frisson-Vorboten → G_floor 0.85
        try:
            from backend.core.dsp.breath_emotion_classifier import (  # pylint: disable=import-outside-toplevel
                classify_breath_emotions,
            )

            _breath_segs = classify_breath_emotions(audio, sr)
            result.breath_segments = _breath_segs
            if _breath_segs:
                _n_tension = sum(1 for s in _breath_segs if s.category.value == "emotional_tension")
                logger.debug(
                    "VFA step 13: %d breath segments (%d emotional_tension)",
                    len(_breath_segs),
                    _n_tension,
                )
        except Exception as _bec_exc:
            logger.debug("VFA step 13 BreathEmotionClassifier: nicht blockierend Ersatzpfad — %s", _bec_exc)

        elapsed = time.perf_counter() - _t0
        logger.info(
            "VocalFocusAnalyzer: panns_singing=%.2f register=%s energy_bias=%.1fdB "
            "frisson=%d formant_f1=%.0fHz stable=%s passaggio=%d vibrato=%d "
            "tension=%d release=%d whisper=%d climax=%s style_zones=%d style_conf=%.2f "
            "singer_school=%s phoneme_prot=%s in %.2fs",
            effective_panns,
            result.dominant_register,
            result.energy_bias_db,
            len(result.frisson_zones),
            result.formant_f1_mean,
            result.formant_stable,
            len(result.passaggio_zones),
            len(result.vibrato_zones),
            len(result.tension_zones),
            len(result.release_zones),
            len(result.whisper_zones),
            result.climax_type,
            len(result.style_intent_zones),
            result.style_confidence,
            result.singer_school,
            result.phoneme_protection_level,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Private Hilfsmethoden — alle non-blocking
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.astype(np.float32)  # type: ignore[no-any-return]
        if audio.ndim == 2:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                return audio.mean(axis=0).astype(np.float32)  # type: ignore[no-any-return]
            if audio.shape[1] == 2:
                return audio.mean(axis=1).astype(np.float32)  # type: ignore[no-any-return]
            if audio.shape[0] == 1:
                return audio[0].astype(np.float32)  # type: ignore[no-any-return]
        return audio.flatten().astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _resolve_vocal_presence_confidence(
        panns_singing: float,
        *,
        panns_tags: dict[str, float] | None = None,
        panns_vocals_confidence: float | None = None,
        is_schlager: bool = False,
        genre_label: str = "",
        vocal_material_prior: bool = False,
    ) -> float:
        """Normalisiert rohe PANNs-Singing-Werte mit vorhandenen Vokal-Hinweisen.

        Ziel: VFA darf bei degradiertem Vokal-Material nicht auf reinem Rohwert
        hängenbleiben, wenn bereits konsistente Musik-/Genre-/Vocal-Tags vorliegen.
        """

        confidence = float(np.clip(float(panns_singing), 0.0, 1.0))
        tags = panns_tags if isinstance(panns_tags, dict) else {}

        def _tag_value(*names: str) -> float:
            values: list[float] = []
            for name in names:
                raw_value = tags.get(name, tags.get(name.lower(), 0.0))
                try:
                    values.append(float(raw_value or 0.0))
                except (TypeError, ValueError):
                    values.append(0.0)
            return max(values) if values else 0.0

        vocal_tag_conf = max(
            _tag_value("Singing", "Singing voice"),
            _tag_value("Vocals"),
            _tag_value("Male singing", "Female singing"),
            _tag_value("Choir", "Chant", "A cappella", "Gospel choir"),
            _tag_value("Soprano", "Alto", "Tenor", "Baritone"),
            _tag_value("Opera", "Aria"),
            _tag_value("Yodeling", "Scat singing"),
        )
        speech_conf = _tag_value("Speech", "Narration", "Male speech", "Female speech")
        music_conf = _tag_value("Music", "Musical instrument", "Pop music", "Music of Latin America")
        raw_vocals = _tag_value("Vocals")

        confidence = max(confidence, vocal_tag_conf)

        if confidence < speech_conf:
            confidence = max(confidence, speech_conf * 0.5)

        if panns_vocals_confidence is not None:
            try:
                confidence = max(confidence, float(panns_vocals_confidence or 0.0))
            except (TypeError, ValueError):
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        if music_conf >= 0.40 and raw_vocals > 0.08 and speech_conf < 0.30:
            confidence = max(confidence, 0.35)

        genre_key = str(genre_label or "").strip().lower()
        genre_key_norm = genre_key.replace("_", " ").replace("-", " ")
        choir_like = any(
            key in genre_key_norm
            for key in ("choir", "choral", "chormusik", "chor", "kantate", "oratorium", "motette", "a cappella")
        )
        strict_vocal_genre = genre_key_norm in {
            "opera",
            "oper",
            "aria",
            "chanson",
            "lied",
            "art song",
            "crooner",
            "vocal",
            "vocals only",
            "vocal jazz",
            "singer songwriter",
        }
        if choir_like or strict_vocal_genre:
            confidence = max(confidence, 0.35)

        schlager_like = bool(is_schlager) or any(
            key in genre_key_norm for key in ("schlager", "chanson", "vocal", "folk", "gospel", "lied")
        )
        if schlager_like and (is_schlager or vocal_tag_conf >= 0.10):
            confidence = max(confidence, 0.35)

        # Expliziter Vocal-Prior aus Pre-Analysis/Bridge muss VFA auch ohne PANNs-Tags
        # aktivieren; sonst bleibt der gesamte Vokal-Kontextpfad trotz bekannter
        # Vokalmaterial-Markierung stumm.
        if vocal_material_prior:
            confidence = max(confidence, 0.35)

        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _detect_register(mono: np.ndarray, sr: int, panns_singing: float) -> tuple[str, float]:
        """VocalRegisterDetector → (register, energy_bias_db). Fallback: ("chest", -6.0)."""
        try:
            register, bias = _detect_vocal_register_fn(mono, sr, panns_singing)  # type: ignore[misc]
            return register, bias
        except Exception as exc:
            logger.debug("VFA._erkennen_register Ersatzpfad: %s", exc)
            return "chest", -6.0

    @staticmethod
    def _detect_frisson(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """FrissonCandidateDetector → [(start_s, end_s)]. Non-blocking."""
        try:
            zones_raw = _get_frisson_detector_fn().detect(audio, sr)  # type: ignore[misc]
            # Normalisiere auf (start_s, end_s) — FrissonZone hat start_s / end_s
            result = []
            for z in zones_raw:
                if hasattr(z, "start_s") and hasattr(z, "end_s"):
                    result.append((float(z.start_s), float(z.end_s)))
                elif isinstance(z, (tuple, list)) and len(z) >= 2:
                    result.append((float(z[0]), float(z[1])))
            return result
        except Exception as exc:
            logger.debug("VFA._erkennen_frisson Ersatzpfad: %s", exc)
            return []

    def _analyze_formants(self, mono: np.ndarray, sr: int) -> tuple[float, float, bool]:
        """LPC-Formant-Analyse → (f1_mean_hz, f2_mean_hz, is_stable).

        Verwendet 4 s Segment aus der Mitte. Fallback: (0.0, 0.0, True).
        """
        try:
            # 4s Analyse-Segment (Zentrum)
            n4 = int(self._FORMANT_SEGMENT_S * sr)
            if len(mono) > n4:
                start4 = (len(mono) - n4) // 2
                seg = mono[start4 : start4 + n4]
            else:
                seg = mono

            tracker = _get_lpc_formant_tracker_fn()  # type: ignore[misc]
            result = tracker.track(seg.astype(np.float32), sr)

            f1_vals: list[float] = []
            f2_vals: list[float] = []
            for frame in result.formant_tracks:  # type: ignore[attr-defined]
                freqs = frame.frequencies if hasattr(frame, "frequencies") else []
                confs = frame.confidences if hasattr(frame, "confidences") else [1.0] * len(freqs)
                if len(freqs) >= 1 and len(confs) >= 1 and float(confs[0]) > 0.25:
                    f1_vals.append(float(freqs[0]))
                if len(freqs) >= 2 and len(confs) >= 2 and float(confs[1]) > 0.25:
                    f2_vals.append(float(freqs[1]))

            f1_mean = float(np.mean(f1_vals)) if f1_vals else 0.0
            f2_mean = float(np.mean(f2_vals)) if f2_vals else 0.0

            # §SOTA #2: LPC-Formant-Fallback — wenn Tracker keine validen Frames
            # liefert (Vintage-Rauschen), plausible Defaults aus F0 ableiten.
            if f1_mean <= 0.0 or f2_mean <= 0.0:
                _f0_est = _estimate_f0_simple(seg, sr)
                if _f0_est > 0:
                    if _f0_est < 160.0:  # Male range
                        f1_mean = f1_mean if f1_mean > 0 else 500.0
                        f2_mean = f2_mean if f2_mean > 0 else 1500.0
                    elif _f0_est < 300.0:  # Female range
                        f1_mean = f1_mean if f1_mean > 0 else 700.0
                        f2_mean = f2_mean if f2_mean > 0 else 2000.0
                    else:  # Child / high voice
                        f1_mean = f1_mean if f1_mean > 0 else 800.0
                        f2_mean = f2_mean if f2_mean > 0 else 2300.0
                else:
                    f1_mean = f1_mean if f1_mean > 0 else 600.0
                    f2_mean = f2_mean if f2_mean > 0 else 1800.0
                logger.debug("VFA formant Ersatzpfad: F0=%.0f → F1=%.0f F2=%.0f", _f0_est, f1_mean, f2_mean)
            # Stabilität: F1-Standardabweichung < 15 % des Mittelwerts → stabil (Opera)
            f1_std = float(np.std(f1_vals)) if len(f1_vals) >= 4 else 0.0
            is_stable = (f1_std < f1_mean * 0.15) if f1_mean > 0 else True
            return f1_mean, f2_mean, is_stable
        except Exception as exc:
            logger.debug("VFA._analyze_formants Ersatzpfad: %s", exc)
            return 0.0, 0.0, True

    @staticmethod
    def _detect_passaggio(mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Erkennt Registerübergangszonen (Brust→Kopf) via F0-Sprünge.

        Gibt [(start_s, end_s)] zurück. Non-blocking.
        Passaggio-Kriterium: F0-Sprung > 2 Halbtöne innerhalb 200 ms.
        """
        try:
            hop = 512
            # Für pyin: auf 16 kHz heruntersampeln — F0-Bereich C2–C7 (65–2093 Hz)
            # liegt weit unter Nyquist 8 kHz. 48 kHz × 30 s = 1,44 M Samples → zu langsam.
            # 16 kHz × max 8 s = 128 K Samples → schnell und ausreichend für Passaggio-Erkennung.
            _PYIN_TARGET_SR = 16000
            _PYIN_MAX_S = 8.0
            mono_f = mono.astype(np.float32)
            _pyin_sr = sr
            if _librosa is not None and sr != _PYIN_TARGET_SR:
                mono_f = _librosa.resample(mono_f, orig_sr=sr, target_sr=_PYIN_TARGET_SR)  # type: ignore[union-attr]
                _pyin_sr = _PYIN_TARGET_SR
            _max_samp = int(_PYIN_MAX_S * _pyin_sr)
            if len(mono_f) > _max_samp:
                mono_f = mono_f[:_max_samp]
            f0, voiced_flag, _ = _librosa.pyin(  # type: ignore[union-attr]
                mono_f,
                fmin=_librosa.note_to_hz("C2"),  # type: ignore[arg-type]
                fmax=_librosa.note_to_hz("C7"),  # type: ignore[arg-type]
                sr=_pyin_sr,
                frame_length=2048,
                hop_length=hop,
            )
            if f0 is None or voiced_flag is None:
                return []

            zones: list[tuple[float, float]] = []
            times = _librosa.frames_to_time(np.arange(len(f0)), sr=_pyin_sr, hop_length=hop)  # type: ignore[union-attr]
            # Halbtöne-Differenz zwischen aufeinanderfolgenden voiced Frames
            prev_f0: float | None = None
            prev_t: float = 0.0
            in_passaggio = False
            zone_start: float = 0.0
            for i, (f, v) in enumerate(zip(f0, voiced_flag)):
                t = float(times[i])
                if not v or not np.isfinite(f) or f <= 0:
                    if in_passaggio:
                        zones.append((zone_start, t))
                        in_passaggio = False
                    prev_f0 = None
                    continue
                f = float(f)
                if prev_f0 is not None and prev_f0 > 0:
                    semitones = abs(12.0 * np.log2(f / prev_f0 + 1e-12))
                    if semitones >= 2.0 and (t - prev_t) <= 0.25:
                        if not in_passaggio:
                            zone_start = max(0.0, prev_t - 0.05)
                            in_passaggio = True
                    elif in_passaggio and semitones < 1.0:
                        zones.append((zone_start, t + 0.05))
                        in_passaggio = False
                prev_f0 = f
                prev_t = t
            if in_passaggio:
                zones.append((zone_start, float(times[-1])))
            return zones
        except Exception as exc:
            logger.debug("VFA._erkennen_passaggio Ersatzpfad: %s", exc)
            return []

    @staticmethod
    def _detect_vibrato(mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Erkennt Vibrato-Passagen (F0-Modulation 4–7 Hz) für §0p Strength-Cap.

        Nutzt ``natural_performance_detector._detect_vibrato_zones`` als Backend.
        Konvertiert Sample-basierte VibratoZone-Objekte in (start_s, end_s)-Tupel.
        Non-blocking: Fehler → [].
        """
        try:
            from backend.core.dsp.natural_performance_detector import (  # pylint: disable=import-outside-toplevel
                detect_vibrato_zones as _dvz,
            )

            raw_zones = _dvz(mono, sr)
            result: list[tuple[float, float]] = []
            for z in raw_zones:
                if hasattr(z, "start_sample") and hasattr(z, "end_sample"):
                    start_s = float(z.start_sample) / max(sr, 1)
                    end_s = float(z.end_sample) / max(sr, 1)
                    result.append((start_s, end_s))
                elif isinstance(z, (tuple, list)) and len(z) >= 2:
                    result.append((float(z[0]), float(z[1])))
            return result
        except Exception as exc:
            logger.debug("VFA._erkennen_vibrato Ersatzpfad: %s", exc)
            return []

    # ------------------------------------------------------------------
    # §Gap1 Emotional Context — Tension/Release/Whisper/Climax
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_tension_zones(mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Erkennt Spannungsaufbau-Passagen: steigende RMS-Energie + hoher Spectral Centroid.

        Algorithm:
        - Frame-weise RMS + Spectral Centroid (hop 512 Samples)
        - tension_value[t] = 0.5 * rms_norm[t] + 0.5 * centroid_norm[t]
        - Threshold ≥ 0.60 → Tension-Zone (mind. 1.0 s Dauer)
        Non-blocking: Fehler → [].
        """
        try:
            _hop = 512
            _frame = 2048
            n = len(mono)
            if n < _frame:
                return []

            n_frames = (n - _frame) // _hop + 1
            rms_frames = np.empty(n_frames, dtype=np.float32)
            centroid_frames = np.empty(n_frames, dtype=np.float32)

            for i in range(n_frames):
                seg = mono[i * _hop : i * _hop + _frame]
                rms_frames[i] = float(np.sqrt(np.mean(seg**2) + 1e-10))
                # Weighted spectral centroid via FFT magnitude
                mag = np.abs(np.fft.rfft(seg * np.hanning(_frame)))
                freqs = np.fft.rfftfreq(_frame, d=1.0 / sr)
                denom = float(np.sum(mag) + 1e-10)
                centroid_frames[i] = float(np.sum(freqs * mag) / denom)

            # Normalize to [0, 1]
            rms_min, rms_max = rms_frames.min(), rms_frames.max()
            c_min, c_max = centroid_frames.min(), centroid_frames.max()
            rms_norm = (rms_frames - rms_min) / max(rms_max - rms_min, 1e-10)
            c_norm = (centroid_frames - c_min) / max(c_max - c_min, 1e-10)

            tension_val = 0.5 * rms_norm + 0.5 * c_norm
            _thr = 0.60
            _min_dur_frames = max(1, int(1.0 * sr / _hop))

            zones: list[tuple[float, float]] = []
            in_zone = False
            zone_start = 0
            for i, tv in enumerate(tension_val):
                if tv >= _thr and not in_zone:
                    in_zone = True
                    zone_start = i
                elif tv < _thr and in_zone:
                    in_zone = False
                    dur = i - zone_start
                    if dur >= _min_dur_frames:
                        zones.append((zone_start * _hop / sr, i * _hop / sr))
            if in_zone:
                dur = n_frames - zone_start
                if dur >= _min_dur_frames:
                    zones.append((zone_start * _hop / sr, n_frames * _hop / sr))

            return zones
        except Exception as exc:
            logger.debug("VFA._erkennen_tension_zones Ersatzpfad: %s", exc)
            return []

    @staticmethod
    def _detect_release_zones(
        mono: np.ndarray,
        sr: int,
        tension_zones: list[tuple[float, float]],
        frisson_zones: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Erkennt Release-Passagen: 1–3 s nach Tension-Peak oder Frisson-Ende.

        Algorithm:
        - Für jede Tension-Zone: Ende + 0.1 s → Fenster [end, end+3.0 s]
        - Für jede Frisson-Zone: [frisson_end, frisson_end+2.5 s]
        - Merged, max. 3.0 s Dauer, überlappend zusammengefasst.
        Non-blocking: Fehler → [].
        """
        try:
            total_s = len(mono) / max(sr, 1)
            candidates: list[tuple[float, float]] = []
            for _start, _end in tension_zones:
                rel_s = _end + 0.1
                rel_e = min(rel_s + 3.0, total_s)
                if rel_e > rel_s:
                    candidates.append((rel_s, rel_e))
            for _start, _end in frisson_zones:
                rel_s = _end + 0.05
                rel_e = min(rel_s + 2.5, total_s)
                if rel_e > rel_s:
                    candidates.append((rel_s, rel_e))
            if not candidates:
                return []
            # Merge overlapping
            candidates.sort(key=lambda x: x[0])
            merged: list[tuple[float, float]] = [candidates[0]]
            for s, e in candidates[1:]:
                ms, me = merged[-1]
                if s <= me:
                    merged[-1] = (ms, max(me, e))
                else:
                    merged.append((s, e))
            return merged
        except Exception as exc:
            logger.debug("VFA._erkennen_release_zones Ersatzpfad: %s", exc)
            return []

    @staticmethod
    def _detect_whisper_zones(mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Erkennt Flüster-Passagen: niedrige Energie (< −35 dBFS) + hohe Spectral Flatness.

        Algorithm:
        - Frames (hop 512, frame 2048): RMS in dBFS + Spectral Flatness
        - Whisper: rms_dbfs < −35 AND flatness > 0.35
        - Mindestdauer 0.3 s, max. Stille-Cutoff: rms < −60 dBFS → übersprungen
        Non-blocking: Fehler → [].
        """
        try:
            _hop = 512
            _frame = 2048
            n = len(mono)
            if n < _frame:
                return []

            n_frames = (n - _frame) // _hop + 1
            _min_dur_frames = max(1, int(0.3 * sr / _hop))
            zones: list[tuple[float, float]] = []
            in_zone = False
            zone_start = 0

            for i in range(n_frames):
                seg = mono[i * _hop : i * _hop + _frame]
                rms = float(np.sqrt(np.mean(seg**2) + 1e-10))
                rms_db = 20.0 * np.log10(rms + 1e-10)
                if rms_db < -60.0:
                    # Pure silence — nicht Flüstern
                    if in_zone:
                        in_zone = False
                        dur = i - zone_start
                        if dur >= _min_dur_frames:
                            zones.append((zone_start * _hop / sr, i * _hop / sr))
                    continue

                # Spectral flatness via geometric mean / arithmetic mean
                mag = np.abs(np.fft.rfft(seg * np.hanning(_frame))) + 1e-10
                geo_mean = float(np.exp(np.mean(np.log(mag))))
                arith_mean = float(np.mean(mag))
                flatness = float(np.clip(geo_mean / (arith_mean + 1e-10), 0.0, 1.0))

                is_whisper = rms_db < -35.0 and flatness > 0.35
                if is_whisper and not in_zone:
                    in_zone = True
                    zone_start = i
                elif not is_whisper and in_zone:
                    in_zone = False
                    dur = i - zone_start
                    if dur >= _min_dur_frames:
                        zones.append((zone_start * _hop / sr, i * _hop / sr))

            if in_zone:
                dur = n_frames - zone_start
                if dur >= _min_dur_frames:
                    zones.append((zone_start * _hop / sr, n_frames * _hop / sr))

            return zones
        except Exception as exc:
            logger.debug("VFA._erkennen_whisper_zones Ersatzpfad: %s", exc)
            return []

    @staticmethod
    def _detect_climax_type(
        frisson_zones: list[tuple[float, float]],
        _tension_zones: list[tuple[float, float]],
        _mono: np.ndarray,
        _sr: int,
    ) -> str:
        """Klassifiziert den Klimax-Charakter für MDEM-Stärke-Entscheidung.

        Typen:
        - "none"       → keine Frisson-Zonen
        - "peak"       → einzelne kurze Frisson-Zone (< 5 s), hohe Intensität
        - "sustained"  → einzelne lange Frisson-Zone (≥ 5 s) oder dominante Tension
        - "dynamic"    → mehrere Frisson-Zonen oder Tension-Frisson-Wechsel
        Non-blocking: Fehler → "none".
        """
        try:
            if not frisson_zones:
                return "none"
            if len(frisson_zones) >= 3:
                return "dynamic"
            if len(frisson_zones) == 2:
                # Weit getrennte Klimax-Passagen = dynamisch
                gap = frisson_zones[1][0] - frisson_zones[0][1]
                return "dynamic" if gap > 5.0 else "sustained"
            # Einzelne Zone
            _s, _e = frisson_zones[0]
            dur = _e - _s
            if dur >= 5.0:
                return "sustained"
            # Viele Tension-Zonen mit kurzer Frisson = peak
            return "peak"
        except Exception as exc:
            logger.debug("VFA._erkennen_climax_type Ersatzpfad: %s", exc)
            return "none"

    @staticmethod
    def _classify_singer_school(vfa: VFAResult) -> tuple[str, str]:
        """Klassifiziert die Gesangsschule aus vorhandenen VFA-Messwerten.

        Entscheidungslogik (hierarchisch, erster Match gewinnt):
            1. Kein Gesang / keine Formanten → "unknown" / "standard"
            2. Classical:  formant_stable=True, F1 < 600 Hz, F2 > 1100 Hz,
                           Vibrato vorhanden, style_confidence < 0.12
            3. Soul/R&B:   Chest-Dominant, style_confidence > 0.20,
                           F1 > 650 Hz, breite Vibrato-Zonen
            4. Jazz:       style_confidence > 0.12, variable Formanten,
                           Vibrato vorhanden
            5. Folk/Country: F1 > 700 Hz, style_confidence < 0.08,
                             wenig Vibrato, instabile Formanten
            6. Schlager:   formant_stable=True, F1 450–680 Hz, kein Style-Intent
            7. Pop:        Default-Fallback

        Returns:
            (singer_school_str, phoneme_protection_level_str)

        Non-blocking: jede Exception → ("unknown", "standard").
        """
        _STRICT = "strict"
        _STANDARD = "standard"
        _RELAXED = "relaxed"
        try:
            if not vfa.vocal_present:
                return "unknown", _STANDARD

            f1 = vfa.formant_f1_mean
            f2 = vfa.formant_f2_mean
            stable = vfa.formant_stable
            style_conf = vfa.style_confidence
            register = vfa.dominant_register
            n_vibrato = len(vfa.vibrato_zones)
            vibrato_coverage = sum(e - s for s, e in vfa.vibrato_zones) if vfa.vibrato_zones else 0.0

            # Formanten nicht messbar → Pop-Fallback
            if f1 < 50.0:
                return "pop", _RELAXED

            # 1. Classical / Bel Canto
            #    Merkmale: gleichmäßiges Vibrato, enge Vokalformanträume (F1 < 600),
            #    hohe Formant-Stabilität, kein Blue-Note-Style
            if stable and f1 < 600.0 and f2 > 1100.0 and n_vibrato >= 1 and style_conf < 0.12:
                return "classical", _STRICT

            # 2. Soul / R&B / Gospel
            #    Merkmale: Chest-Dominant, hoher Style-Intent (Melisma, Bends),
            #    breite Formant-Varianz, viel Vibrato
            if register in {"chest", "head"} and style_conf > 0.20 and f1 > 600.0:
                return "soul_rnb", _STANDARD

            # 3. Jazz Standard
            #    Merkmale: Blue Notes (style_confidence > 0.12), Vibrato vorhanden,
            #    instabilere Formanten als Klassik
            if style_conf > 0.12 and (n_vibrato >= 1 or vibrato_coverage > 1.0):
                return "jazz", _STANDARD

            # 4. Folk / Country
            #    Merkmale: nasale Resonanz (F1 > 700, F2 < 1500), wenig Style-Intent,
            #    geringe Vibrato-Dichte, instabile Formanten
            if f1 > 700.0 and style_conf < 0.08 and not stable and n_vibrato < 2:
                return "folk_country", _STANDARD

            # 5. Schlager / Chanson
            #    Merkmale: stabile Formanten, enge Vokal-Räume (F1 450–680),
            #    kein Blue-Note-Style, typische Gesangsschule "Belting/Chanson"
            if stable and 450.0 < f1 < 680.0 and style_conf < 0.10:
                return "schlager", _STANDARD

            # 6. Pop (Default)
            return "pop", _RELAXED

        except Exception as exc:
            logger.debug("VFA._classify_singer_school Ersatzpfad: %s", exc)
            return "unknown", _STANDARD

    @staticmethod
    def _detect_style_intent(mono: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], float]:
        """§P2 Style-Intent-Detektion — intentionale Pitch-Abweichungen (Blue Notes etc.).

        Returns:
            (style_intent_zones, style_confidence)
        """
        try:
            if _get_style_intent_detector_fn is None:
                return [], 0.0
            detector = _get_style_intent_detector_fn()
            sid_result = detector.analyze(mono, sr)
            return list(sid_result.style_intent_zones), float(sid_result.style_confidence)
        except Exception as exc:
            logger.debug("VFA._erkennen_style_intent Ersatzpfad: %s", exc)
            return [], 0.0


# ---------------------------------------------------------------------------
# Singleton-Zugriff
# ---------------------------------------------------------------------------


def _estimate_f0_simple(audio: np.ndarray, sr: int) -> float:
    """Einfache F0-Schätzung via Autocorrelation für Formant-Fallback."""
    try:
        if len(audio) < sr // 20:
            return 0.0
        n = min(len(audio), sr * 2)  # Max 2s
        seg = audio[:n].astype(np.float64)
        autocorr = np.correlate(seg, seg, mode="full")
        autocorr = autocorr[len(seg) - 1 :] / max(autocorr[len(seg) - 1], 1e-10)
        min_lag = int(sr / 450)
        max_lag = int(sr / 80)
        if max_lag >= len(autocorr):
            return 0.0
        peak_idx = np.argmax(autocorr[min_lag:max_lag]) + min_lag
        return float(sr / peak_idx)
    except Exception as e:
        logger.warning("vocal_focus_analyzer.py::_estimate_f0_simple Ersatzpfad: %s", e)
        return 0.0


def get_vocal_focus_analyzer() -> VocalFocusAnalyzer:
    """Thread-sicherer Singleton-Zugriff."""
    # pylint: disable-next=global-statement
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VocalFocusAnalyzer()
                logger.info("VocalFocusAnalyzer initialisiert (§0p)")
    return _instance
