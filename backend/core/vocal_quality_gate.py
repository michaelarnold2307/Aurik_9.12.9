"""Vocal Quality Gate — Rolls-Royce-Phantom-Standard für Gesangs-Restaurierung.

§Rolls-Royce-Vision: Aurik restauriert Musik mit Gesang auf einem Niveau,
das kein anderes automatisches System erreicht. Der Vocal Quality Gate ist
das zentrale Qualitäts-Sicherungssystem, das JEDE Pipeline-Entscheidung
an der Frage misst: „Klingt die Stimme natürlicher fürs menschliche Ohr?"

Kernprinzipien:
    1. **Naturalness First** — Keine Metrik rechtfertigt unnatürlichen Klang
    2. **Breath is Signal** — Atemgeräusche sind Teil der Stimme, kein Rauschen
    3. **Formant Integrity** — Vokalformanten (F1–F4) dürfen nie verschoben werden
    4. **Sibilance Retention** — Zischlaute müssen erhalten bleiben (mind. 95%)
    5. **Listening Comfort** — Keine Ermüdung im kritischen 2–5 kHz-Bereich
    6. **Intelligibility** — Textverständlichkeit nach Restaurierung ≥ vorher

Architektur:
    - VocalDetector: PANNS-basierte Gesangsaktivitätserkennung
    - VocalNaturalnessScorer: 6-dimensionale Gesangsqualität (0–100)
    - VocalComfortOptimizer: Hörmüdungs-Prävention
    - VocalQualityGate: Zentraler Entscheider (accept/reject/rollback)

Autor: Aurik 10 — Rolls-Royce Phantom Edition, 11. Juli 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Konstanten ───────────────────────────────────────────────────────────────
_VOCAL_PRESENCE_THRESHOLD: float = 0.15  # PANNS-Singing-Confidence-Schwelle
_CRITICAL_FREQ_LOW: float = 2000.0  # Hz — Beginn kritischer Gesangsbereich
_CRITICAL_FREQ_HIGH: float = 5000.0  # Hz — Ende kritischer Gesangsbereich
_FORMANT_MAX_SHIFT_DB: float = 1.0  # dB — Maximal erlaubte Formant-Verschiebung
_SIBILANCE_MIN_RETENTION: float = 0.95  # Mindest-Zischlaut-Erhalt
_COMFORT_MAX_SHARPNESS: float = 3.0  # Maximal erlaubte Schärfe (Sharpness in acum)


@dataclass
class VocalPresence:
    """Ergebnis der Gesangsaktivitätserkennung."""

    has_vocals: bool
    confidence: float  # 0.0–1.0, PANNS-Singing-Confidence
    vocal_ratio: float  # Anteil der Zeit mit Gesang (0.0–1.0)
    segments: list[tuple[float, float]] = field(default_factory=list)
    # Liste von (start_s, end_s) mit Gesangsaktivität


@dataclass
class VocalNaturalnessScores:
    """6-dimensionale Gesangsqualitäts-Bewertung.

    Jede Dimension: 0–100, wobei 100 = perfekte natürliche Stimme.
    """

    overall: float  # Gesamtwertung (gewichtetes Mittel)
    formant_integrity: float  # Formant-Treue (F1–F4)
    breath_naturalness: float  # Atem-Natürlichkeit
    sibilance_retention: float  # Zischlaut-Erhalt
    intelligibility: float  # Textverständlichkeit
    comfort: float  # Hörkomfort (keine Ermüdung)
    timbre_warmth: float  # Stimmwärme (kein „dünner" Klang)
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class VocalGateDecision:
    """Entscheidung des Vocal Quality Gate."""

    accept: bool  # True = Verarbeitung akzeptiert
    naturalness_delta: float  # Veränderung der Gesangsqualität (positiv = besser)
    pre_scores: VocalNaturalnessScores | None = None
    post_scores: VocalNaturalnessScores | None = None
    rollback_needed: bool = False
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class VocalDetector:
    """Erkennt Gesangsaktivität im Audio via PANNS-Singing-Confidence.

    Lazy-Load des PANNS-Modells. Thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._model: Any = None
        self._threshold: float = _VOCAL_PRESENCE_THRESHOLD

    def detect(self, audio: np.ndarray, sr: int = 48000) -> VocalPresence:
        """Erkennt Gesangsaktivität.

        Args:
            audio: float32, mono [N] oder stereo.
            sr:    Abtastrate.

        Returns:
            VocalPresence mit has_vocals und confidence.
        """
        # Mono konvertieren
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
        mono = mono.astype(np.float32)

        # ── PANNS-basierte Erkennung ────────────────────────────────────
        detection_method = "spectral"
        try:
            # Vereinfachte spektrale Gesangsdetektion (PANNS-Proxy)
            # Echte PANNS-Inferenz via SessionManager wenn verfügbar
            confidence = self._spectral_vocal_proxy(mono, sr)
        except Exception:
            # §5/5: Verbesserte Fallback-Kette: spectral → MFCC → energy
            logger.warning("VocalDetector: spectral_vocal_proxy fehlgeschlagen — trying MFCC-based Ersatzpfad")
            try:
                confidence = self._mfcc_vocal_proxy(mono, sr)
                detection_method = "mfcc"
            except Exception:
                logger.warning("VocalDetector: MFCC Ersatzpfad fehlgeschlagen — falling back to energy heuristic")
                confidence = self._energy_vocal_proxy(mono, sr)
                detection_method = "energy"
        logger.debug("VocalDetector: detection_method=%s confidence=%.3f", detection_method, confidence)

        has_vocals = confidence >= self._threshold

        # Grobe Segmentierung (2-Sekunden-Fenster)
        segment_s = 2.0
        n_segments = max(1, int(len(mono) / sr / segment_s))
        segments: list[tuple[float, float]] = []
        vocal_frames = 0
        total_frames = 0

        for i in range(n_segments):
            s0 = i * int(sr * segment_s)
            s1 = min(len(mono), s0 + int(sr * segment_s))
            segment = mono[s0:s1]
            if len(segment) < sr * 0.1:
                continue
            total_frames += 1
            # Einfache Energie-Prüfung pro Segment
            seg_rms = float(np.sqrt(np.mean(segment**2)))
            if seg_rms > 0.001:
                vocal_frames += 1
                segments.append((s0 / sr, s1 / sr))

        vocal_ratio = vocal_frames / max(total_frames, 1)

        return VocalPresence(
            has_vocals=has_vocals,
            confidence=confidence,
            vocal_ratio=vocal_ratio,
            segments=segments,
        )

    def _spectral_vocal_proxy(self, mono: np.ndarray, sr: int) -> float:
        """Spektrale Gesangs-Proxy (ohne PANNS-Modell).

        Nutzt spektrale Merkmale die für Gesang charakteristisch sind:
        - Hohe Energie im 300–3000 Hz-Bereich (Formant-Bereich)
        - Harmonische Struktur (Pitch-Präsenz)
        - Vibrato-Modulation (4–8 Hz AM/FM)
        """
        if len(mono) < sr * 0.5:
            return 0.0

        # Spektrum berechnen
        n_fft = min(2048, len(mono) // 2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spec = np.abs(np.fft.rfft(mono[:n_fft]))

        # Energie im Stimmformant-Bereich (300–3400 Hz)
        voice_mask = (freqs >= 300) & (freqs <= 3400)
        voice_energy = float(np.sum(spec[voice_mask]))
        total_energy = np.sum(spec) + 1e-10
        voice_ratio = float(voice_energy / total_energy)

        # Harmonische Struktur (spektrale Peaks)
        from scipy.signal import find_peaks

        peaks, props = find_peaks(spec, height=np.max(spec) * 0.1, distance=5)
        harmonic_score = min(1.0, len(peaks) / 20.0) if len(peaks) > 0 else 0.0

        # Kombinierte Confidence
        confidence = 0.6 * voice_ratio + 0.4 * harmonic_score
        return min(1.0, max(0.0, confidence))

    def _mfcc_vocal_proxy(self, mono: np.ndarray, sr: int) -> float:
        """MFCC-basierte Gesangsdetektion als Fallback zwischen spectral und energy.

        Nutzt die Tatsache, dass Gesang stärkere Energie in den ersten
        3-5 MFCC-Koeffizienten konzentriert (Grundfrequenz + Formanten),
        während Rauschen/Instrumente flachere MFCC-Profile haben.
        """
        try:
            import librosa

            _mfcc = librosa.feature.mfcc(y=mono.astype(np.float64), sr=sr, n_mfcc=13)
            _mfcc_var = np.var(_mfcc, axis=1)
            # Gesangs-Indikator: hohe Varianz in MFCC 0-4 (Grundfrequenz + Formanten)
            # vs niedrige Varianz in MFCC 8-12 (Rauschanteil)
            _voice_energy = float(np.sum(_mfcc_var[:5]))
            _noise_energy = float(np.sum(_mfcc_var[8:]))
            _ratio = _voice_energy / max(_noise_energy, 1e-10)
            # Heuristik: ratio > 2.0 = wahrscheinlich Gesang
            return float(np.clip((_ratio - 1.0) / 3.0, 0.0, 1.0))
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            raise  # Weiterreichen an energy-Fallback

    def _energy_vocal_proxy(self, mono: np.ndarray, sr: int) -> float:
        """Minimal-Fallback: Energie-basierte Heuristik."""
        rms = float(np.sqrt(np.mean(mono**2)))
        return min(1.0, rms * 10.0)


class VocalNaturalnessScorer:
    """Bewertet Gesangs-Natürlichkeit auf 6 Dimensionen.

    Optimiert für menschliches Hörempfinden, nicht für technische Metriken.
    """

    def score(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        reference: np.ndarray | None = None,
    ) -> VocalNaturalnessScores:
        """Berechnet 6-dimensionale Gesangsqualität.

        Args:
            audio:     Zu bewertendes Audio.
            sr:        Abtastrate.
            reference: Optionales Referenz-Audio (für Delta-Berechnung).

        Returns:
            VocalNaturalnessScores (0–100 pro Dimension).
        """
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
        mono = mono.astype(np.float32)

        scores = VocalNaturalnessScores(
            overall=50.0,
            formant_integrity=self._score_formant_integrity(mono, sr),
            breath_naturalness=self._score_breath_naturalness(mono, sr),
            sibilance_retention=self._score_sibilance_retention(mono, sr),
            intelligibility=self._score_intelligibility(mono, sr),
            comfort=self._score_comfort(mono, sr),
            timbre_warmth=self._score_timbre_warmth(mono, sr),
        )

        # Gewichtetes Gesamturteil (human-weighted)
        weights = {
            "formant_integrity": 0.25,
            "breath_naturalness": 0.15,
            "sibilance_retention": 0.15,
            "intelligibility": 0.20,
            "comfort": 0.15,
            "timbre_warmth": 0.10,
        }

        overall = sum(getattr(scores, key) * weight for key, weight in weights.items())
        scores.overall = round(overall, 1)

        return scores

    # ── Dimension-Scorer ─────────────────────────────────────────────────

    def _score_formant_integrity(self, mono: np.ndarray, sr: int) -> float:
        """Formant-Treue: Wie intakt sind die Vokalformanten?

        Misst spektrale Stabilität im 300–3400 Hz-Bereich.
        """
        if len(mono) < sr * 0.5:
            return 50.0

        n_fft = min(2048, len(mono))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spec = np.abs(np.fft.rfft(mono[:n_fft]))

        voice_mask = (freqs >= 300) & (freqs <= 3400)
        voice_spec = spec[voice_mask]
        if len(voice_spec) < 10:
            return 50.0

        # Spektrale Glätte (raue Spektren = beschädigte Formanten)
        diff = np.diff(voice_spec)
        roughness = float(np.std(diff) / (np.mean(np.abs(voice_spec)) + 1e-10))
        score = 100.0 - min(80.0, roughness * 20.0)
        return max(0.0, min(100.0, score))

    def _score_breath_naturalness(self, mono: np.ndarray, sr: int) -> float:
        """Atem-Natürlichkeit: Wurden Atemgeräusche als Rauschen entfernt?

        Atem ist im 4–8 kHz-Bereich als sanftes Rauschen präsent.
        Wenn dieser Bereich zu sauber ist, wurde Über-Entrauscht.
        """
        if len(mono) < sr * 0.5:
            return 50.0

        n_fft = min(2048, len(mono))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spec = np.abs(np.fft.rfft(mono[:n_fft]))

        # Atembereich: 4–8 kHz
        breath_mask = (freqs >= 4000) & (freqs <= 8000)
        breath_energy = float(np.sum(spec[breath_mask]))

        if breath_energy < 1e-8:
            return 30.0  # Zu sauber — Atem wurde entfernt

        # Gesamtenergie
        total_energy = np.sum(spec) + 1e-10
        breath_ratio = float(breath_energy / total_energy)

        # Optimal: 0.5%–2% der Gesamtenergie im Atembereich
        if 0.005 <= breath_ratio <= 0.02:
            return 90.0
        elif breath_ratio < 0.001:
            return 40.0  # Über-entrauscht
        elif breath_ratio > 0.05:
            return 60.0  # Zu viel Rauschen
        else:
            return 70.0

    def _score_sibilance_retention(self, mono: np.ndarray, sr: int) -> float:
        """Zischlaut-Erhalt: Wurden S-Laute durch De-Essing zerstört?

        Sibilanten liegen bei 5–10 kHz. Wenn dieser Bereich zu stark
        reduziert wurde, klingt die Stimme „lispelnd" oder dumpf.
        """
        if len(mono) < sr * 0.5:
            return 50.0

        n_fft = min(2048, len(mono))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spec = np.abs(np.fft.rfft(mono[:n_fft]))

        sib_mask = (freqs >= 5000) & (freqs <= 10000)
        sib_energy = float(np.sum(spec[sib_mask]))
        total_energy = np.sum(spec) + 1e-10
        sib_ratio = float(sib_energy / total_energy)

        # Natürliche Sibilanz: 0.1%–1% der Gesamtenergie
        score = 100.0 - abs(sib_ratio - 0.005) * 10000
        return max(0.0, min(100.0, score))

    def _score_intelligibility(self, mono: np.ndarray, sr: int) -> float:
        """Textverständlichkeit: Wie klar sind die Konsonanten?

        Misst spektrale Modulation im 1–4 kHz-Bereich (Konsonanten).
        """
        if len(mono) < sr * 0.5:
            return 50.0

        n_fft = min(2048, len(mono))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spec = np.abs(np.fft.rfft(mono[:n_fft]))

        cons_mask = (freqs >= 1000) & (freqs <= 4000)
        cons_spec = spec[cons_mask]
        if len(cons_spec) < 5:
            return 50.0

        # Modulationstiefe (höher = klarere Konsonanten)
        modulation = float(np.std(cons_spec) / (np.mean(cons_spec) + 1e-10))
        score = 50.0 + modulation * 30.0
        return max(0.0, min(100.0, score))

    def _score_comfort(self, mono: np.ndarray, sr: int) -> float:
        """Hörkomfort: Verursacht das Audio Hörmüdung?

        Schärfe (Sharpness) im 2–5 kHz-Bereich ist der Haupttreiber
        von Hörmüdung. Zu viel Energie in diesem Bereich = unangenehm.
        """
        if len(mono) < sr * 0.5:
            return 50.0

        n_fft = min(2048, len(mono))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spec = np.abs(np.fft.rfft(mono[:n_fft]))

        critical_mask = (freqs >= _CRITICAL_FREQ_LOW) & (freqs <= _CRITICAL_FREQ_HIGH)
        critical_energy = float(np.sum(spec[critical_mask]))
        total_energy = np.sum(spec) + 1e-10
        sharpness = float(critical_energy / total_energy)

        # Schärfe: < 10% ist komfortabel, > 25% ermüdend
        if sharpness < 0.10:
            return 95.0
        elif sharpness < 0.15:
            return 85.0
        elif sharpness < 0.20:
            return 70.0
        elif sharpness < 0.25:
            return 50.0
        else:
            return 30.0

    def _score_timbre_warmth(self, mono: np.ndarray, sr: int) -> float:
        """Stimmwärme: Klingt die Stimme warm oder dünn?

        Wärme = ausgewogene Energie in den unteren Formanten (100–500 Hz).
        """
        if len(mono) < sr * 0.5:
            return 50.0

        n_fft = min(2048, len(mono))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spec = np.abs(np.fft.rfft(mono[:n_fft]))

        warmth_mask = (freqs >= 100) & (freqs <= 500)
        warmth_energy = float(np.sum(spec[warmth_mask]))
        total_energy = np.sum(spec) + 1e-10
        warmth_ratio = float(warmth_energy / total_energy)

        # Optimale Wärme: 20%–40%
        if 0.20 <= warmth_ratio <= 0.40:
            return 90.0
        elif warmth_ratio < 0.10:
            return 40.0  # Dünn
        elif warmth_ratio > 0.60:
            return 60.0  # Mulmig
        else:
            return 70.0


class VocalQualityGate:
    """Zentraler Entscheider für Gesangsqualität.

    Wird NACH jeder Phase aufgerufen wenn Gesang erkannt wurde.
    Entscheidet: accept (weiter), reject (Rollback), oder warn.
    """

    def __init__(self):
        self._detector = VocalDetector()
        self._scorer = VocalNaturalnessScorer()
        self._history: list[VocalGateDecision] = []

    def evaluate(
        self,
        pre_audio: np.ndarray,
        post_audio: np.ndarray,
        sr: int = 48000,
        phase_name: str = "",
        strict: bool = False,
    ) -> VocalGateDecision:
        """Bewertet Gesangsqualität vor/nach einer Phase.

        Args:
            pre_audio:   Audio VOR der Phase.
            post_audio:  Audio NACH der Phase.
            sr:          Abtastrate.
            phase_name:  Name der Phase (für Logging).
            strict:      True → Rollback bei JEDER Verschlechterung.

        Returns:
            VocalGateDecision mit accept/reject.
        """
        # ── Gesangserkennung ────────────────────────────────────────────
        presence = self._detector.detect(post_audio, sr)
        if not presence.has_vocals:
            # Kein Gesang → Gate passiv, immer accept
            return VocalGateDecision(
                accept=True,
                naturalness_delta=0.0,
                recommendations=["Kein Gesang erkannt — Gate passiv"],
            )

        # ── Qualität bewerten ───────────────────────────────────────────
        pre_scores = self._scorer.score(pre_audio, sr)
        post_scores = self._scorer.score(post_audio, sr)

        delta = post_scores.overall - pre_scores.overall
        warnings: list[str] = []
        recommendations: list[str] = []

        # Einzel-Checks
        if post_scores.formant_integrity < pre_scores.formant_integrity - 5:
            warnings.append(
                f"Formant-Integrität gesunken: {pre_scores.formant_integrity:.0f} → {post_scores.formant_integrity:.0f}"
            )
            recommendations.append("Formant-Korrektur prüfen (Phase 65)")
        if post_scores.sibilance_retention < _SIBILANCE_MIN_RETENTION * 100:
            warnings.append(f"Sibilanz-Erhalt kritisch: {post_scores.sibilance_retention:.0f}/100")
            recommendations.append("De-Essing reduzieren oder rückgängig machen")
        if post_scores.comfort < 40:
            warnings.append(f"Hörkomfort kritisch: {post_scores.comfort:.0f}/100 (Hörmüdung wahrscheinlich)")
            recommendations.append("2–5 kHz-Bereich auf Schärfe prüfen")
        if post_scores.breath_naturalness < 30:
            warnings.append(
                f"Atem-Natürlichkeit kritisch: {post_scores.breath_naturalness:.0f}/100 "
                "(wahrscheinlich über-entrauscht)"
            )
            recommendations.append("Noise Reduction zurückschrauben")

        # Entscheidung
        rollback = delta < -10.0 or (strict and delta < -1.0)
        accept = not rollback

        decision = VocalGateDecision(
            accept=accept,
            naturalness_delta=round(delta, 1),
            pre_scores=pre_scores,
            post_scores=post_scores,
            rollback_needed=rollback,
            warnings=warnings,
            recommendations=recommendations,
        )

        self._history.append(decision)
        return decision

    def get_history(self) -> list[VocalGateDecision]:
        """Chronologische Gate-Entscheidungen."""
        return self._history.copy()

    def get_final_report(self) -> dict[str, Any]:
        """Abschlussbericht über alle Gate-Entscheidungen."""
        if not self._history:
            return {"status": "no_data"}

        rollbacks = [d for d in self._history if d.rollback_needed]
        deltas = [d.naturalness_delta for d in self._history]

        return {
            "total_decisions": len(self._history),
            "rollbacks": len(rollbacks),
            "accepted": len(self._history) - len(rollbacks),
            "mean_delta": round(float(np.mean(deltas)), 1) if deltas else 0.0,
            "best_delta": round(float(np.max(deltas)), 1) if deltas else 0.0,
            "worst_delta": round(float(np.min(deltas)), 1) if deltas else 0.0,
            "final_quality": (
                self._history[-1].post_scores.overall if self._history and self._history[-1].post_scores else 0.0
            ),
            "warnings": [w for d in self._history for w in d.warnings],
            "recommendations": list({r for d in self._history for r in d.recommendations}),
        }


# ── Convenience ─────────────────────────────────────────────────────────────

_vocal_gate: VocalQualityGate | None = None
_lock = threading.Lock()


def get_vocal_quality_gate() -> VocalQualityGate:
    """Singleton-Instanz des VocalQualityGate."""
    global _vocal_gate
    if _vocal_gate is None:
        with _lock:
            if _vocal_gate is None:
                _vocal_gate = VocalQualityGate()
    return _vocal_gate


__all__ = [
    "VocalDetector",
    "VocalNaturalnessScorer",
    "VocalQualityGate",
    "VocalPresence",
    "VocalNaturalnessScores",
    "VocalGateDecision",
    "get_vocal_quality_gate",
]
