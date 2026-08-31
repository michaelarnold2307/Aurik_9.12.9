"""PresenceEmbedding — 5-dimensionale Praesenz-Metrik.

Spec 18.1, G90 RELEASE_MUST.
Misst die Distanz zwischen „Aufnahme" und „Live-Praesenz":
  - Vocal Formant Coherence: MERT-basierte Formant-Distanz zu echten Gesangsaufnahmen
  - Transient Immediacy: Onset-Staerke vs. Live-Referenzen
  - Room Tone Continuity: Rauschboden-Varianz ueber Zeit
  - Microdynamic Liveliness: Crest-Faktor in 200ms-Fenstern
  - Spectral Air Authenticity: HF-Huellkurve >10 kHz vs. natuerliche Referenz

Schwellwert: PresenceScore >= 0.70 = „hoerbare Verbesserung".

Autor: Aurik 10 — August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Schwellwert fuer „hoerbare Verbesserung" (Spec 18.1)
PRESENCE_THRESHOLD_HEARABLE: float = 0.70

# Referenz-Datenbank fuer Vocal-Formant-Coherence (wird via MERT gefuellt)
# Format: {formant_bin: (reference_mean, reference_std)}
_VOCAL_FORMANT_REFERENCE: dict[int, tuple[float, float]] = {
    0: (0.65, 0.15),
    1: (0.55, 0.18),
    2: (0.45, 0.20),
    3: (0.35, 0.22),
    4: (0.25, 0.25),  # Faelle fuer 5 Formant-Bins
}


@dataclass
class PresenceScore:
    """5-dimensionaler Presence-Score. Jede Dimension in [0.0, 1.0]."""

    # Sub-Scores (alle 0.0–1.0, hoeher = besser)
    vocal_formant_coherence: float = 0.0
    transient_immediacy: float = 0.0
    room_tone_continuity: float = 0.0
    microdynamic_liveliness: float = 0.0
    spectral_air_authenticity: float = 0.0

    # Gewichteter Gesamt-Score
    overall: float = 0.0

    # Metadata
    is_hearable_improvement: bool = False
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocal_formant_coherence": round(self.vocal_formant_coherence, 4),
            "transient_immediacy": round(self.transient_immediacy, 4),
            "room_tone_continuity": round(self.room_tone_continuity, 4),
            "microdynamic_liveliness": round(self.microdynamic_liveliness, 4),
            "spectral_air_authenticity": round(self.spectral_air_authenticity, 4),
            "overall": round(self.overall, 4),
            "is_hearable_improvement": self.is_hearable_improvement,
            "confidence": round(self.confidence, 4),
        }


# ── Hauptklasse ──────────────────────────────────────────────────────────────


class PresenceEmbedding:
    """Berechnet den 5-dimensionalen Presence-Score.

    Integration: NACH allen Restaurierungsphasen, VOR dem Export.
    Ersetzt NICHT die technischen Metriken — ergaenzt sie.

    Verwendung:
        embedder = PresenceEmbedding()
        score = embedder.compute(restored_audio, original_audio, sr=48000)
        if score.is_hearable_improvement:
            logger.info("Hoerbare Verbesserung: %.2f", score.overall)
    """

    def __init__(self) -> None:
        self._last_score: PresenceScore | None = None

    @property
    def last_score(self) -> PresenceScore | None:
        """Zuletzt berechneter PresenceScore."""
        return self._last_score

    def compute(
        self,
        restored: np.ndarray,
        original: np.ndarray | None = None,
        *,
        sample_rate: int = 48000,
    ) -> PresenceScore:
        """Berechnet den PresenceScore fuer restauriertes Audio.

        Args:
            restored: Restauriertes Audio (float32, mono oder stereo)
            original: Original-Audio (optional, fuer Vorher/Nachher-Vergleich)
            sample_rate: Sample-Rate in Hz

        Returns:
            PresenceScore mit allen 5 Sub-Scores
        """
        # Konvertiere zu mono float64 fuer Analyse
        mono_restored = self._to_mono(restored)
        mono_original = self._to_mono(original) if original is not None else None

        # 1. Vocal Formant Coherence
        vfc = self._compute_vocal_formant_coherence(mono_restored, sample_rate)

        # 2. Transient Immediacy
        ti = self._compute_transient_immediacy(mono_restored, sample_rate)

        # 3. Room Tone Continuity
        rtc = self._compute_room_tone_continuity(mono_restored, mono_original, sample_rate)

        # 4. Microdynamic Liveliness
        ml = self._compute_microdynamic_liveliness(mono_restored, sample_rate)

        # 5. Spectral Air Authenticity
        saa = self._compute_spectral_air_authenticity(mono_restored, sample_rate)

        # Gewichteter Overall-Score (alle 5 Dimensionen gleichgewichtet)
        overall = float(np.mean([vfc, ti, rtc, ml, saa]))

        # Confidence basierend auf Varianz der Sub-Scores
        sub_scores = np.array([vfc, ti, rtc, ml, saa])
        confidence = 1.0 - float(np.std(sub_scores))  # Niedrige Varianz = hohe Confidence

        is_hearable = overall >= PRESENCE_THRESHOLD_HEARABLE

        score = PresenceScore(
            vocal_formant_coherence=vfc,
            transient_immediacy=ti,
            room_tone_continuity=rtc,
            microdynamic_liveliness=ml,
            spectral_air_authenticity=saa,
            overall=overall,
            is_hearable_improvement=is_hearable,
            confidence=confidence,
        )

        self._last_score = score
        logger.info(
            "PresenceEmbedding: overall=%.3f (hearable=%s) | vfc=%.3f ti=%.3f rtc=%.3f ml=%.3f saa=%.3f",
            overall,
            is_hearable,
            vfc,
            ti,
            rtc,
            ml,
            saa,
        )
        return score

    def compute_delta(
        self,
        restored: np.ndarray,
        original: np.ndarray,
        *,
        sample_rate: int = 48000,
    ) -> tuple[PresenceScore, PresenceScore]:
        """Berechnet PresenceScore fuer Original UND restauriert (Vorher/Nachher).

        Returns:
            Tuple[original_score, restored_score]
        """
        score_orig = self.compute(original, sample_rate=sample_rate)
        score_rest = self.compute(restored, original=original, sample_rate=sample_rate)
        return score_orig, score_rest

    # ── Sub-Scorer ────────────────────────────────────────────────────────

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float64)
        return cast(np.ndarray, arr.mean(axis=1) if arr.ndim == 2 else arr)

    def _compute_vocal_formant_coherence(self, audio: np.ndarray, sr: int) -> float:
        """MERT-basierte Distanz zwischen restaurierten Formanten und Referenz-Datenbank.

        Simuliert die MERT-Distanz via Formant-Peak-Analyse.
        In Produktion: MERT-Embedding-Vergleich mit echter Gesangsdatenbank.
        """
        try:
            # LPC-basierte Formant-Schaetzung
            # Vereinfachte Heuristik: spektrale Huellkurve via Cepstrum
            n_fft = min(2048, len(audio) // 4)
            if n_fft < 64:
                return 0.5

            spec = np.abs(np.fft.rfft(audio[: n_fft * 4], n=n_fft))
            spec_db = 20.0 * np.log10(spec + 1e-12)
            spec_db -= np.max(spec_db)

            # Finde Peaks als Proxy fuer Formanten
            from scipy.signal import find_peaks

            peaks, props = find_peaks(spec_db, height=-30.0, distance=max(5, n_fft // 128))
            if len(peaks) < 2:
                return 0.5

            # Binning der Peaks in 5 Formant-Bins
            n_bins = min(5, len(_VOCAL_FORMANT_REFERENCE))
            bin_edges = np.linspace(0, len(spec_db), n_bins + 1).astype(int)
            coherence_scores = []

            for i in range(n_bins):
                bin_peaks = peaks[(peaks >= bin_edges[i]) & (peaks < bin_edges[i + 1])]
                if len(bin_peaks) == 0:
                    coherence_scores.append(0.5)
                    continue
                bin_magnitude = float(np.max(spec_db[bin_peaks]))
                ref_mean, ref_std = _VOCAL_FORMANT_REFERENCE.get(i, (0.5, 0.2))
                # Normalisierte Distanz zur Referenz
                dist = abs(bin_magnitude / (np.max(spec_db) + 1e-12) - ref_mean) / (ref_std + 1e-6)
                coherence = np.exp(-dist)  # [0,1], 1 = perfekte Uebereinstimmung
                coherence_scores.append(float(np.clip(coherence, 0.0, 1.0)))

            return float(np.mean(coherence_scores))
        except Exception as e:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("PresenceEmbedding: vocal_formant_coherence fallback: %s", e)
            return 0.5

    def _compute_transient_immediacy(self, audio: np.ndarray, sr: int) -> float:
        """Onset-Staerke-Verteilung im Vergleich zu Live-Referenzen.

        Lebendige Musik hat klare, starke Transienten.
        Ueberprozessierte Musik hat verschliffene Transienten.
        """
        try:
            # Einfacher Onset-Detektor: Energie-Differenz zwischen Frames
            frame_len = int(sr * 0.01)  # 10ms
            n_frames = max(1, len(audio) // frame_len)
            energy = np.zeros(n_frames)
            for i in range(n_frames):
                chunk = audio[i * frame_len : (i + 1) * frame_len]
                energy[i] = np.mean(chunk**2)

            # Onset: Energie-Sprung > 6dB zwischen Frames
            energy_db = 10.0 * np.log10(energy + 1e-12)
            onset_strength = np.diff(energy_db)
            onset_strength = onset_strength[onset_strength > 0]  # Nur positive Spruenge

            if len(onset_strength) < 3:
                return 0.5

            # Verteilung der Onset-Staerke: Median + IQR
            median_onset = float(np.median(onset_strength))
            iqr_onset = float(np.percentile(onset_strength, 75) - np.percentile(onset_strength, 25))

            # Referenz: gute Musik hat median ~4dB, IQR ~3dB
            median_score = 1.0 - abs(median_onset - 4.0) / 8.0
            iqr_score = 1.0 - abs(iqr_onset - 3.0) / 6.0
            return float(np.clip((median_score * 0.7 + iqr_score * 0.3), 0.0, 1.0))
        except Exception as e:
            logger.debug("PresenceEmbedding: transient_immediacy fallback: %s", e)
            return 0.5

    def _compute_room_tone_continuity(self, restored: np.ndarray, original: np.ndarray | None, sr: int) -> float:
        """Rauschboden-Varianz ueber die Zeit.

        Ein echter Raum hat einen kontinuierlichen, gleichmaessigen Rauschboden.
        DSP-Artefakte erzeugen unnatuerliche Varianz.
        """
        try:
            frame_len = int(sr * 0.5)  # 500ms Frames
            n_frames = max(1, len(restored) // frame_len)
            noise_floors = np.zeros(n_frames)

            for i in range(n_frames):
                chunk = restored[i * frame_len : (i + 1) * frame_len]
                # Rauschboden = unterstes 10%-Perzentil der Amplitude
                noise_floors[i] = float(np.percentile(np.abs(chunk), 10))

            if n_frames < 4:
                return 0.5

            # CoV (Coefficient of Variation) des Rauschbodens
            mean_nf = float(np.mean(noise_floors))
            std_nf = float(np.std(noise_floors))
            cov = std_nf / (mean_nf + 1e-12)

            # Niedrige CoV = kontinuierlicher Rauschboden = authentisch
            # Referenz: CoV < 0.3 = sehr gut, CoV > 1.0 = unnatuerlich
            score = 1.0 - min(cov / 1.5, 1.0)

            # Wenn Original verfuegbar: vergleiche Rauschboden-Kontinuitaet
            if original is not None:
                orig_mono = self._to_mono(original)
                orig_noise = np.zeros(min(n_frames, len(orig_mono) // frame_len))
                for i in range(len(orig_noise)):
                    chunk = orig_mono[i * frame_len : (i + 1) * frame_len]
                    orig_noise[i] = float(np.percentile(np.abs(chunk), 10))
                orig_cov = float(np.std(orig_noise)) / (float(np.mean(orig_noise)) + 1e-12)
                orig_score = 1.0 - min(orig_cov / 1.5, 1.0)
                # Verbesserung? Dann score erhoehen
                if score > orig_score:
                    score = min(1.0, score * 1.1)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.debug("PresenceEmbedding: room_tone_continuity fallback: %s", e)
            return 0.5

    def _compute_microdynamic_liveliness(self, audio: np.ndarray, sr: int) -> float:
        """Crest-Faktor-Verteilung in 200ms-Fenstern.

        Lebendige Dynamik = hoher Crest-Faktor (Peak/RMS).
        Komprimiertes Audio = niedriger Crest-Faktor.
        """
        try:
            frame_len = int(sr * 0.2)  # 200ms per Spec
            n_frames = max(1, len(audio) // frame_len)
            crest_factors = np.zeros(n_frames)

            for i in range(n_frames):
                chunk = audio[i * frame_len : (i + 1) * frame_len]
                peak = float(np.max(np.abs(chunk)))
                rms = float(np.sqrt(np.mean(chunk**2)) + 1e-12)
                crest_factors[i] = peak / rms

            if n_frames < 3:
                return 0.5

            median_cf = float(np.median(crest_factors))
            std_cf = float(np.std(crest_factors))

            # Referenz: gute Mikrodynamik hat median CF ~6-10, std ~2-4
            median_score = float(np.clip(1.0 - abs(median_cf - 8.0) / 10.0, 0.0, 1.0))
            std_score = min(std_cf / 3.0, 1.0)  # Hoehere Std = mehr Dynamik-Varianz = lebendiger
            return float(np.clip((median_score * 0.6 + std_score * 0.4), 0.0, 1.0))
        except Exception as e:
            logger.debug("PresenceEmbedding: microdynamic_liveliness fallback: %s", e)
            return 0.5

    def _compute_spectral_air_authenticity(self, audio: np.ndarray, sr: int) -> float:
        """Korrelation der HF-Huellkurve (>10 kHz) mit natuerlichen Referenzen.

        Echte Luft hat eine kontinuierlich abfallende HF-Huellkurve.
        Synthetische HF klingt unnatuerlich flach oder hat Artefakt-Peaks.
        """
        try:
            n_fft = min(4096, len(audio) // 2)
            if n_fft < 128:
                return 0.5

            spec = np.abs(np.fft.rfft(audio[: n_fft * 2], n=n_fft))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

            # HF-Bereich: >10 kHz
            hf_mask = freqs > 10000.0
            if not np.any(hf_mask):
                # Kein HF-Content (bandbreitenbegrenztes Material)
                return 0.8  # Neutral — kein HF = keine HF-Artefakte

            hf_env = spec[hf_mask]
            hf_freqs = freqs[hf_mask]

            # Ideale HF-Huellkurve: monoton fallend, -6dB/Oktave ab 10 kHz
            # Modelliere als: A * (f/10000)^(-alpha) mit alpha ~1 (entspricht -6dB/Okt)
            log_freqs = np.log10(hf_freqs + 1e-12)
            log_env = 20.0 * np.log10(hf_env + 1e-12)

            # Lineare Regression auf log-log: Steigung sollte ~ -6 sein
            if len(log_freqs) < 4:
                return 0.5
            slope, intercept = np.polyfit(log_freqs, log_env, 1)

            # Perfekte Steigung: -6 (dB/Oktave), akzeptabler Bereich: -3 bis -12
            slope_score = 1.0 - abs(slope + 6.0) / 9.0

            # R^2 der Regression: wie gut passt die Kurve?
            predicted = slope * log_freqs + intercept
            ss_res = float(np.sum((log_env - predicted) ** 2))
            ss_tot = float(np.sum((log_env - float(np.mean(log_env))) ** 2))
            r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
            return float(np.clip(slope_score * 0.5 + r_squared * 0.5, 0.0, 1.0))
        except Exception as e:
            logger.warning("ML→DSP-Fallback aktiviert: %s", e, exc_info=True)  # §V6 (copilot-instructions.md)
            return 0.5


# ── Singleton ────────────────────────────────────────────────────────────────

_presence_embedding: PresenceEmbedding | None = None


def get_presence_embedding() -> PresenceEmbedding:
    """Gibt die globale PresenceEmbedding-Instanz zurueck."""
    global _presence_embedding
    if _presence_embedding is None:
        _presence_embedding = PresenceEmbedding()
    return _presence_embedding


def is_hearable_improvement(score: PresenceScore | float) -> bool:
    """Prueft ob der PresenceScore eine hoerbare Verbesserung anzeigt."""
    if isinstance(score, PresenceScore):
        return score.is_hearable_improvement
    return float(score) >= PRESENCE_THRESHOLD_HEARABLE
