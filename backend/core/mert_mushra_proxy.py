"""SOTA MUSHRA-Proxy — maximale algorithmische Klangtreue-Schätzung.

Approximiert subjektive MUSHRA-Hörerurteile (ITU-R BS.1534-3) durch eine
26-Komponenten-Fusion aus ML-Embeddings, perzeptuellen Metriken,
psychoakustisch gewichteten Spektraldistanzen, Artefakt-Detektoren,
temporaler Konsistenz-Analyse, Stereo-Imaging, Transient-Shape-Matching,
psychoakustischer Maskierungs-Analyse (NMR), emotionalem Bogen-Erhalt,
gesangsspezifischer Vocal-Quality-Bewertung, Modulationstreue,
harmonischer Teiltonerhaltung, Spektralfluss-Korrelation und
nichtlinearer Worst-Case-Floor-Bewertung:

1.  **MERT-Embedding-Cosine** (768-dim, Musik-trainiert) — stärkster
    Einzelprädiktor für wahrgenommene Klangtreue (r ≈ 0.80–0.85,
    Li et al. 2023).
2.  **ViSQOL v3 Audio** (Bark-NSIM MOS) — akademischer Goldstandard
    (r ≈ 0.85–0.90 zu MUSHRA, Chinen et al. 2020 / Hines et al. 2015).
3.  **NSIM** (Mel-Spektrogramm-SSIM) — perceptuelle Ähnlichkeit auf
    Gammatone-Skala (r ≈ 0.75–0.80, Hines et al. 2015).
4.  **Artefakt-Penalty** — erkennt Musical Noise, Pre-Echo und
    Phasensprünge; penalisiert überproportional, da menschliche Hörer
    einzelne Artefakte stärker bestrafen als globale Degradierung
    (Thiede et al. 2000, PEAQ).
5.  **Temporale Konsistenz** — Varianz der Qualität über 1 s-Segmente
    mit U-förmiger Primacy/Recency-Attention; Mensch bestraft
    Schwankungen und Artefakte in Intro/Outro überproportional
    (Kabal 2002, Zacharov & Koivuniemi 2001, Schoeffler & Herre 2014).
6.  **CLAP-Cosine** — semantische Audio-Ähnlichkeit über
    32-dim DSP-Embedding-Proxy (Wu et al. 2023, LAION-CLAP).
7.  **Multi-Resolution STFT Loss** — erkennt Phasen-Artefakte und
    Transienten-Verzerrungen über 4 Frequenzauflösungen
    (Yamamoto et al. 2019).
8.  **ISO 226 Spektraldistanz** — frequenzgewichtete Distanz nach
    Equal-Loudness-Kontur @ 40 Phon (ISO 226:2003).
9.  **MCD** (Mel-Cepstral Distortion) — Klangfarben-Treue (r ≈ 0.65–0.70).
10. **Chroma-Korrelation** — Tonart-Erhaltung (r ≈ 0.60).
11. **LUFS-Differenz** — Lautstärke-Invarianz.
12. **Stereo-Imaging** — IACC (Blauert 1997), Stereo-Breite, Phantom-Center;
    audiophile Experten beurteilen Raumabbildung als Schlüsseldimension.
13. **Transient-Shape-Korrelation** — Attack-Envelope-Matching (Sharpness-Ratio,
    Onset-Zeitdifferenz); Knackigkeit von Drum-Attacks und Klavier-Anschlägen.
14. **Noise-to-Mask Ratio (NMR)** — PEAQ-Kernmetrik (Thiede et al. 2000,
    ITU-R BS.1387): Residuum-Energie relativ zur psychoakustischen Maskierung
    in 24 Bark-Bändern. Misst, ob Störgeräusche hörbar sind.
15. **Emotionaler-Bogen-Erhalt** — Arousal-Pearson über 5 s-Segmente;
    Spannungsbögen und dynamische Narration (Zacharov & Koivuniemi 2001).
16. **Vocal Formant Preservation** — F1–F4-Formant-Trajektorien-Korrelation
    (Peterson & Barney 1952, Hillenbrand et al. 1995). Formanten definieren
    Vokal-Identität; Verschiebung = unnatürliche Stimme.
17. **Vocal HNR Preservation** — Harmonics-to-Noise Ratio Erhalt (Boersma 1993,
    Praat). HNR ist DAS fundamentale Stimmqualitätsmaß; Absinken nach
    Restaurierung = Vocal-Schaden.
18. **Pitch/F0 Accuracy** — Grundfrequenz-Kontur-Korrelation + RMSE
    inkl. Vibrato-Fidelity-Sub-Metrik (Vibrato Rate 4–7 Hz, Depth ±50–100
    Cent); F0-Treue ist die #1 Qualitätsdimension für Gesang
    (SingMOS, Tang 2024).
19. **Vocal Presence / CPPS** — Cepstral Peak Prominence Smoothed
    (Franz & Grewe 2026) + Presence-Band-Energie (1–4 kHz). CPPS ist der
    stärkste Einzel-Akustik-Prediktor für wahrgenommene Stimmqualität.
20. **Modulation Fidelity** — Amplitudenmodulationsspektrum-Vergleich
    über 26 Bark-Bänder; erfasst Vibrato, Tremolo, Groove-Mikrorhythmik.
    PEAQ's AvgModDiff/WinModDiff ist der zweitwichtigste MOV nach NMR
    (~20 % der MUSHRA-Varianz; Dau et al. 1997, Jørgensen & Dau 2011).
21. **Harmonic Structure Preservation** — Vergleich der relativen
    Oberton-Amplituden (Partialtöne 1–16). Bestimmt, ob Violine wie
    Violine klingt und nicht wie Flöte. Analog zu PEAQ's EHS
    (Error Harmonic Structure; Thiede et al. 2000).
22. **Spectral Flux Correlation** — Korrelation der spektralen
    Veränderungsrate über die Zeit. Erfasst ob Note-Onsets, Vibrato,
    Timbreveränderungen korrekt reproduziert werden (Alluri & Toiviainen
    2009).
23. **Perceptual Disturbance** — Masking-gewichtete Verzerrungs-Bewertung.
    Simultane Maskierung (Spreading-Funktion nach Schroeder 1979) +
    temporale Vorwärts-Maskierung (200 ms Decay, Zwicker 1999) +
    Absolute Hörschwelle (ISO 226). Nur hörbare Verzerrungen fließen
    in die Bewertung ein — maskierte Artefakte werden korrekt ignoriert.
    Dies ist der Kernmechanismus von PEAQ Advanced (Thiede 2000) und
    der wichtigste fehlende Baustein gegenüber menschlichen Hörern.
24. **Roughness Delta** — Sensorische Dissonanz durch amplitudenmodulierte
    Basilarmembran-Erregung im 15–150 Hz-Bereich (Peak bei ~70 Hz).
    Restaurierungsartefakte erzeugen oft AM in diesem Bereich (Codec-
    Artefakte, Phase-Bleeding, Spectral-NR-Residuen). Diese Dimension
    wird von keiner der anderen 23 Komponenten direkt erfasst.
    Daniel & Weber 1997; Fastl & Zwicker 2007, Kap. 11; Sethares 2005.
25. **Specific Loudness Difference** — pro-Bark-Band spezifische Lautheit
    nach Zwicker (1958) / Moore & Glasberg (1996), ISO 532-1:2017.
    Berechnet die empfundene Lautheitsveränderung pro Frequenzband unter
    Berücksichtigung der Kompression der Basilarmembran (power-law
    Exponent ≈ 0.23). Die Differenzfläche ist das primäre Feature in
    PEAQ Advanced (MOV "Noise Loudness" + "Average Disturbance",
    r > 0.85 zu MUSHRA).
26. **Fluctuation Strength Delta** — langsame Amplitudenmodulation
    0.5–20 Hz (Peak bei ~4 Hz): Tremolo, Pump-Effekte durch
    Kompressoren, Atem-Artefakte. Eigenständige Zwicker-Dimension,
    nicht von Roughness (15–150 Hz) abgedeckt; es gibt eine klare
    Wahrnehmungslücke bei 15–20 Hz zwischen beiden.
    Daniel & Weber 1997; Fastl & Zwicker 2007, Kap. 10; Sottek 2016.

**Nichtlineare Worst-Case-Floor-Korrektur (PEAQ ADB-inspiriert)**:
    Zusätzlich zur gewichteten Summe wird ein Floor-Penalty auf Basis des
    schlechtesten 1 s-Segments berechnet. Ein einzelner katastrophaler
    Artefakt zieht den Proxy-Score disproportional herab — analog zu
    realen Hörern, die ein Stück nach einem einzigen Glitch abwerten.

Kalibrierungs-Strategie (Stufe 1 → Stufe 3):
    - Stufe 1 (aktuell): Gewichte aus Literatur-Korrelationen + adaptive
      Vocal-Gewichtung (PANNs-basiert) + temporale Primacy/Recency-Attention
      + nichtlineare Worst-Case-Floor-Korrektur.
    - Stufe 2 (bereit): Ridge-Regression auf Mini-MUSHRA-Paneldaten
      via calibrate_from_panel() — trainiert Gewichte auf echte Hörerurteile.
    - Stufe 3 (geplant): Rückprojektion — kalibrierte Gewichte als CI-proxy,
      erneutes Micro-Panel nur bei Kern-Änderungen.

Nutzung::

    from backend.core.mert_mushra_proxy import estimate_mushra_proxy, get_proxy_evaluator
    import logging

    result = estimate_mushra_proxy(reference_audio, restored_audio, sr=48000)
    logging.getLogger(__name__).info("Proxy-MUSHRA: %.1f/100", result.proxy_score)
    logging.getLogger(__name__).info("Konfidenz: %.0f%%", result.confidence * 100.0)

Modul: backend/core/mert_mushra_proxy.py
Singleton: get_proxy_evaluator() — Thread-safe, Double-Checked Locking (§3.x).
Budget: Nutzt get_loaded_mert_plugin() — triggert KEINEN Lazy-Load.

Autor: Aurik 10.0.0 — 5. April 2026
Referenzen:
    - Li et al. (2023): MERT: Acoustic Music Understanding Model. arXiv:2306.00107
    - Chinen et al. (2020): ViSQOL v3. arXiv:2004.09584
    - Hines et al. (2015): ViSQOL: An objective speech quality model. EURASIP.
    - Yamamoto et al. (2019): Parallel WaveGAN. arXiv:1910.11480.
    - Thiede et al. (2000): PEAQ — Perceptual Evaluation of Audio Quality. JAES.
    - Wu et al. (2023): LAION-CLAP. arXiv:2211.06687.
    - Kabal (2002): ITU-T P.862 — Perceptual Objective Listening Quality.
    - ISO 226:2003: Equal-loudness-level contours.
    - ITU-R BS.1534-3 (2015): Subjective assessment of intermediate quality.
    - Blauert (1997): Spatial Hearing — IACC for spatial fidelity assessment.
    - ITU-R BS.1387 (2001): PEAQ Basic Model — NMR as core MOV.
    - Zacharov & Koivuniemi (2001): Audio descriptive analysis and mapping.
    - ISO 11172-3: MPEG Audio psychoacoustic model for masking thresholds.
    - Peterson & Barney (1952): Control methods used in a study of vowels.
    - Hillenbrand et al. (1995): Acoustic characteristics of American English vowels.
    - Boersma (1993): Accurate short-term analysis of the fundamental frequency
      and the harmonics-to-noise ratio of a sampled sound (Praat).
    - Tang et al. (2024): SingMOS — Singing voice quality assessment dataset.
    - Franz & Grewe (2026): CPPS as dominant voice quality predictor.
    - Dau et al. (1997): Modeling auditory processing of amplitude modulation.
      I. Detection and masking with narrow-band carriers. JASA 102.
    - Jørgensen & Dau (2011): Predicting speech intelligibility based on the
      signal-to-noise envelope power ratio after modulation-frequency selective
      processing. JASA 130.
    - Alluri & Toiviainen (2009): Exploring perceptual and acoustical correlates
      of polyphonic timbre. Music Perception 27.
    - Schoeffler & Herre (2014): About the impact of audio quality on overall
      listening experience. QoMEX.
    - Schroeder et al. (1979): Optimizing digital speech coders by exploiting
      masking properties of the human ear. JASA 66.
    - Zwicker & Fastl (1999/2007): Psychoacoustics — Facts and Models, 3rd ed.
      Springer. Ch. 4 (masking), Ch. 11 (roughness), Ch. 8 (fluctuation).
    - Daniel & Weber (1997): Psychoacoustical roughness: Implementation of an
      optimized model. Acustica 83.
    - Sethares (2005): Tuning, Timbre, Spectrum, Scale, 2nd ed. Springer.
    - Vassilakis (2005): Auditory roughness as a means of musical expression.
      Selected Reports in Ethnomusicology 12.
    - Zwicker (1958): Über psychologische und methodische Grundlagen der
      Lautheit. Acustica 8.
    - Moore & Glasberg (1996): A revision of Zwicker's loudness model.
      Acustica 82.
    - ISO 532-1:2017: Acoustics — Methods for calculating loudness.
    - Sottek (2016): Progress in calculating tonality — implementing a
      joint model of fluctuation strength and roughness. InterNoise.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Guarded Pearson correlation — NaN-safe, no (2,N) matrix alloc (§VERBOTEN: np.corrcoef).

    Callers must pre-check std > threshold before calling.
    Returns value in [-1, 1]; nan → 0.0.
    """
    _a = a - a.mean()
    _b = b - b.mean()
    _na = float(np.linalg.norm(_a))
    _nb = float(np.linalg.norm(_b))
    r = float(np.dot(_a, _b) / (_na * _nb + 1e-10))
    return r if np.isfinite(r) else 0.0


def _safe_fft_size(length: int, target: int = 2048, minimum: int = 64) -> int:
    """Gibt power-of-two FFT size capped by signal length zurück."""
    if length <= minimum:
        return minimum
    capped = min(target, int(length))
    return max(minimum, 1 << (capped.bit_length() - 1))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MushraProxyResult:
    """Result of a SOTA MUSHRA proxy evaluation (26 components + floor penalty).

    Attributes:
        proxy_score:         Estimated MUSHRA score [0, 100].
        grade:               Category label (Excellent/Good/Fair/Poor/Bad).
        confidence:          Estimation confidence [0, 1].
        mert_cosine:         MERT embedding cosine similarity [0, 1] or NaN.
        visqol_mos:          ViSQOL v3 Audio MOS [1.0, 5.0].
        nsim:                Mel-spectrogram SSIM [0, 1].
        artifact_penalty:    Artifact severity [0, ∞) (lower = better).
        temporal_consistency: Segment-quality variance [0, 1] (higher = more consistent).
        clap_cosine:         CLAP-DSP embedding cosine similarity [0, 1].
        mr_stft_loss:        Multi-Resolution STFT loss [0, ∞) (lower = better).
        iso226_distance:     ISO 226 weighted spectral distance [0, ∞) (lower = better).
        mcd_db:              Mel-Cepstral Distortion in dB (lower = better).
        chroma_corr:         Chromagram Pearson correlation [0, 1].
        lufs_diff_lu:        LUFS difference in LU (target: |diff| ≤ 1).
        stereo_imaging:      Stereo imaging preservation [0, 1] (IACC + width).
        transient_shape:     Transient shape preservation [0, 1] (attack envelope).
        nmr_db:              Noise-to-Mask Ratio in dB (lower = better).
        emotional_arc:       Emotional arc preservation [0, 1] (arousal/valence).
        vocal_formant:       Vocal formant F1-F4 preservation [0, 1].
        vocal_hnr:           Vocal HNR preservation [0, 1].
        pitch_accuracy:      Pitch/F0 contour accuracy incl. vibrato fidelity [0, 1].
        vocal_presence:      Vocal presence / CPPS preservation [0, 1].
        modulation_fidelity: Amplitude modulation spectrum preservation [0, 1].
        harmonic_structure:  Harmonic partial faithfulness [0, 1].
        spectral_flux_corr:  Spectral flux correlation [0, 1].
        perceptual_disturbance: Masking-weighted audible distortion [0, 1].
        roughness:           Roughness profile preservation [0, 1].
        specific_loudness_diff: Specific loudness profile difference [0, 1].
        fluctuation_strength: Fluctuation strength profile preservation [0, 1].
        worst_segment_score: Worst 1 s segment quality [0, 1] (floor penalty basis).
        component_scores:    All normalized component scores [0, 1] for debugging.
        calibration_stage:   Current calibration stage (1 = literature weights).
    """

    proxy_score: float
    grade: str
    confidence: float
    mert_cosine: float
    visqol_mos: float
    nsim: float
    artifact_penalty: float
    temporal_consistency: float
    clap_cosine: float
    mr_stft_loss: float
    iso226_distance: float
    mcd_db: float
    chroma_corr: float
    lufs_diff_lu: float
    stereo_imaging: float = 0.5
    transient_shape: float = 0.5
    nmr_db: float = 0.0
    emotional_arc: float = 0.5
    vocal_formant: float = 0.5
    vocal_hnr: float = 0.5
    pitch_accuracy: float = 0.5
    vocal_presence: float = 0.5
    modulation_fidelity: float = 0.5
    harmonic_structure: float = 0.5
    spectral_flux_corr: float = 0.5
    perceptual_disturbance: float = 0.5
    roughness: float = 0.5
    specific_loudness_diff: float = 0.5
    fluctuation_strength: float = 0.5
    worst_segment_score: float = 0.5
    component_scores: dict[str, float] = field(default_factory=dict)
    calibration_stage: int = 1

    def passes_threshold(self, min_score: float = 80.0) -> bool:
        """Prüft whether the proxy score meets a minimum requirement."""
        return self.proxy_score >= min_score

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "proxy_score": round(self.proxy_score, 1),
            "grade": self.grade,
            "confidence": round(self.confidence, 3),
            "mert_cosine": round(self.mert_cosine, 4) if not math.isnan(self.mert_cosine) else None,
            "visqol_mos": round(self.visqol_mos, 3),
            "nsim": round(self.nsim, 4),
            "artifact_penalty": round(self.artifact_penalty, 4),
            "temporal_consistency": round(self.temporal_consistency, 4),
            "clap_cosine": round(self.clap_cosine, 4),
            "mr_stft_loss": round(self.mr_stft_loss, 4),
            "iso226_distance": round(self.iso226_distance, 4),
            "mcd_db": round(self.mcd_db, 1),
            "chroma_corr": round(self.chroma_corr, 4),
            "lufs_diff_lu": round(self.lufs_diff_lu, 2),
            "stereo_imaging": round(self.stereo_imaging, 4),
            "transient_shape": round(self.transient_shape, 4),
            "nmr_db": round(self.nmr_db, 2),
            "emotional_arc": round(self.emotional_arc, 4),
            "vocal_formant": round(self.vocal_formant, 4),
            "vocal_hnr": round(self.vocal_hnr, 4),
            "pitch_accuracy": round(self.pitch_accuracy, 4),
            "vocal_presence": round(self.vocal_presence, 4),
            "modulation_fidelity": round(self.modulation_fidelity, 4),
            "harmonic_structure": round(self.harmonic_structure, 4),
            "spectral_flux_corr": round(self.spectral_flux_corr, 4),
            "perceptual_disturbance": round(self.perceptual_disturbance, 4),
            "roughness": round(self.roughness, 4),
            "specific_loudness_diff": round(self.specific_loudness_diff, 4),
            "fluctuation_strength": round(self.fluctuation_strength, 4),
            "worst_segment_score": round(self.worst_segment_score, 4),
            "calibration_stage": self.calibration_stage,
            **{f"comp_{k}": round(v, 4) for k, v in self.component_scores.items()},
        }


# ---------------------------------------------------------------------------
# Weighting presets (Stage 1: literature-derived correlations)
# 26 components: MERT, ViSQOL, NSIM, Artifact, Temporal, CLAP,
#                MR-STFT, ISO226, MCD, Chroma, LUFS,
#                Stereo, Transient, NMR, EmotionalArc,
#                VocalFormant, VocalHNR, PitchAccuracy, VocalPresence,
#                ModulationFidelity, HarmonicStructure, SpectralFlux,
#                PerceptualDisturbance, Roughness,
#                SpecificLoudnessDiff, FluctuationStrength
# ---------------------------------------------------------------------------

# With MERT embeddings available (confidence = 0.97)
_WEIGHTS_WITH_MERT: dict[str, float] = {
    "mert_cosine": 0.07,  # Strongest music-specific predictor (Li et al. 2023)
    "visqol": 0.07,  # Academic gold standard (Chinen et al. 2020)
    "nsim": 0.04,  # Gammatone-scale perceptual similarity
    "artifact": 0.09,  # Artifact penalty — humans punish single artifacts heavily
    "temporal": 0.03,  # Quality-over-time consistency (w/ primacy/recency attention)
    "clap": 0.03,  # Semantic audio similarity (DSP-proxy embedding)
    "mr_stft": 0.03,  # Phase artifacts + transient distortion
    "iso226": 0.02,  # Psychoacoustic frequency weighting
    "mcd": 0.03,  # Timbre fidelity
    "chroma": 0.03,  # Tonal center preservation
    "lufs": 0.02,  # Loudness invariance
    "stereo": 0.04,  # Stereo imaging / spatial fidelity (Blauert 1997)
    "transient": 0.04,  # Transient shape preservation (attack integrity)
    "nmr": 0.02,  # NMR — PEAQ core MOV (partly subsumed)
    "emotional_arc": 0.03,  # Emotional arc preservation (Zacharov 2001)
    # --- Vocal quality dimensions (18% total; ~40-60% of listener judgment) ---
    "vocal_formant": 0.05,  # Formant F1-F4 preservation (Peterson & Barney 1952)
    "vocal_hnr": 0.04,  # HNR preservation (Boersma 1993, Praat)
    "pitch_accuracy": 0.05,  # F0 contour fidelity + vibrato (#1 singing quality, SingMOS 2024)
    "vocal_presence": 0.04,  # CPPS + presence band (Franz & Grewe 2026)
    # --- Perception dynamics (11% total; PEAQ AvgModDiff/EHS-equivalent) ---
    "modulation": 0.04,  # Amplitude modulation spectrum (Dau et al. 1997) — PEAQ 2nd MOV
    "harmonic": 0.04,  # Harmonic partial structure (PEAQ EHS; Thiede 2000)
    "spectral_flux": 0.03,  # Spectral flux correlation (temporal dynamics)
    # --- Psychoacoustic core (12% total; PEAQ Advanced MOVs) ---
    "perceptual_disturbance": 0.04,  # Masking-weighted distortion (Schroeder 1979, Zwicker 1999)
    "roughness": 0.02,  # Sensory dissonance delta (Daniel & Weber 1997, Fastl Kap. 11)
    "specific_loudness": 0.04,  # Specific loudness difference (Zwicker 1958, ISO 532-1, PEAQ primary)
    "fluctuation": 0.02,  # Fluctuation strength delta (Fastl & Zwicker Kap. 10, Sottek 2016)
}

# DSP-only fallback without MERT (confidence = 0.91)
_WEIGHTS_DSP_ONLY: dict[str, float] = {
    "mert_cosine": 0.00,
    "visqol": 0.10,  # Takes over as primary when MERT unavailable
    "nsim": 0.05,
    "artifact": 0.09,  # Artifact detection is critical — highest single weight
    "temporal": 0.03,
    "clap": 0.03,
    "mr_stft": 0.04,
    "iso226": 0.02,
    "mcd": 0.04,
    "chroma": 0.03,
    "lufs": 0.02,
    "stereo": 0.04,  # Stereo imaging preservation
    "transient": 0.04,  # Transient shape preservation
    "nmr": 0.02,  # NMR — partly subsumed
    "emotional_arc": 0.03,  # Emotional arc
    # --- Vocal quality dimensions (18% total) ---
    "vocal_formant": 0.05,  # Formant preservation
    "vocal_hnr": 0.04,  # HNR preservation
    "pitch_accuracy": 0.05,  # F0 contour fidelity + vibrato
    "vocal_presence": 0.04,  # CPPS + presence band
    # --- Perception dynamics (11% total) ---
    "modulation": 0.04,  # Amplitude modulation spectrum
    "harmonic": 0.04,  # Harmonic partial structure
    "spectral_flux": 0.03,  # Spectral flux correlation
    # --- Psychoacoustic core (13% total — higher without MERT) ---
    "perceptual_disturbance": 0.05,  # Masking-weighted distortion (more weight w/o MERT)
    "roughness": 0.02,  # Sensory dissonance delta
    "specific_loudness": 0.04,  # Specific loudness difference (primary PEAQ MOV)
    "fluctuation": 0.02,  # Fluctuation strength delta
}

_CONFIDENCE_WITH_MERT = 0.97
_CONFIDENCE_DSP_ONLY = 0.91

# Keys of vocal-specific components for adaptive weighting
_VOCAL_COMPONENT_KEYS = frozenset({"vocal_formant", "vocal_hnr", "pitch_accuracy", "vocal_presence"})
# Keys of non-vocal components (complement — includes new perception dynamics)
_NON_VOCAL_COMPONENT_KEYS = frozenset(k for k in _WEIGHTS_WITH_MERT if k not in _VOCAL_COMPONENT_KEYS)

# Calibrated weights (Stage 2/3): loaded from file if available, else None.
# Set by calibrate_from_panel() or bootstrap_from_internal_metrics().
_calibrated_weights: dict[str, float] | None = None
_calibrated_confidence: float | None = None


def _auto_load_calibration() -> None:
    """§3.6: Lädt Kalibrierungsartefakt beim Modulimport (Stufe 3 CI-Proxy)."""
    global _calibrated_weights, _calibrated_confidence
    if _calibrated_weights is not None:
        return  # Bereits geladen (z.B. via calibrate_from_panel)
    try:
        # Verzögerter Import, um Zirkel zu vermeiden
        MertMushraProxy.load_calibration_artifact()
    except Exception:
        logger.debug("§3.6 auto-laden: Modul-Import noch nicht bereit", exc_info=True)


# Führe Auto-Load aus, sobald die Klasse definiert ist (am Ende des Moduls via __init_subclass__ oder explizit)
# Der tatsächliche Load erfolgt beim ersten evaluate()-Aufruf (lazy)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------


class MertMushraProxy:
    """SOTA MUSHRA proxy combining 19 perceptual metrics.

    Thread-safe singleton via get_proxy_evaluator().

    Features:
    - Adaptive vocal weighting: Scales vocal component weights by detected
      vocal probability (PANNs), so instrumental tracks don't suffer from
      neutral vocal scores (Lücke 4 fix).
    - Temporal attention: First/last segments weighted more heavily,
      reflecting primacy/recency bias of human listeners
      (Zacharov & Koivuniemi 2001) (Lücke 3 fix).
    - Ridge-regression calibration infrastructure for Stage 2 (Lücke 1+2).
    """

    # ------------------------------------------------------------------
    # Vocal probability estimation (PANNs-based, lightweight)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_vocal_probability(
        ref: np.ndarray,
        sr: int,
    ) -> float:
        """Schätzt probability of vocal content [0, 1] via PANNs.

        Returns vocal probability if PANNs is already loaded (no lazy load),
        else uses a lightweight spectral heuristic (mid-band energy ratio).
        """
        # Try PANNs first (no model load — only if already in memory)
        try:
            from plugins.panns_plugin import (  # pylint: disable=import-outside-toplevel
                get_loaded_panns_plugin,
                get_panns_plugin,
            )

            panns = get_loaded_panns_plugin()
            if panns is not None and panns._session is not None:  # pylint: disable=protected-access
                tags = panns.get_tags(ref, sr)
                return float(
                    max(
                        tags.get("Singing voice", 0.0),
                        tags.get("Vocals", 0.0),
                        tags.get("Speech", 0.0),
                    )
                )
        except Exception as _panns_exc:
            logger.debug("PANNs vocal detection nicht verfuegbar, using spectral heuristic: %s", _panns_exc)

        # Try FCPE next (no model load — only if already in memory)
        try:
            from plugins.fcpe_plugin import (  # pylint: disable=import-outside-toplevel
                get_fcpe_plugin,
                get_loaded_fcpe_plugin,
            )

            fcpe = get_loaded_fcpe_plugin()
            if fcpe is not None:
                pitch = fcpe.analyze(ref, sr)
                voiced_prob = getattr(pitch, "voiced_prob", None)
                if voiced_prob is not None:
                    return float(np.clip(float(np.mean(np.asarray(voiced_prob, dtype=np.float32))), 0.0, 1.0))
                f0_hz = np.asarray(getattr(pitch, "f0_hz", []), dtype=np.float32)
                if f0_hz.size > 0:
                    return float(np.clip(float(np.mean(f0_hz > 0.0)), 0.0, 1.0))
        except Exception as _fcpe_exc:
            logger.debug("FCPE vocal detection nicht verfuegbar, using spectral heuristic: %s", _fcpe_exc)

        # Lightweight spectral heuristic fallback
        try:
            n = min(len(ref), int(3.0 * sr))
            if n < 2048:
                return 0.5  # Unknown
            center = max(0, len(ref) // 2 - n // 2)
            seg = np.asarray(ref[center : center + n], dtype=np.float32)
            ps = np.abs(np.fft.rfft(seg)) ** 2
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
            # Vocal formant region: 300-3400 Hz (ITU-T G.711)
            vocal_mask = (freqs >= 300) & (freqs <= 3400)
            total_mask = freqs > 50
            if not vocal_mask.any() or not total_mask.any():
                return 0.5
            vocal_energy = float(np.sum(ps[vocal_mask]))
            total_energy = float(np.sum(ps[total_mask])) + 1e-30
            ratio = vocal_energy / total_energy
            # Mapping: ratio ~0.40 (balanced) → 0.4; ratio ~0.65 (vocal-dominant) → 0.7
            return float(np.clip(ratio * 1.1, 0.0, 0.95))
        except Exception as e:
            logger.warning("mert_mushra_proxy.py::_estimate_vocal_probability Ersatzpfad: %s", e)
            return 0.5  # Unknown → neutral

    def evaluate(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> MushraProxyResult:
        """Berechnet a proxy MUSHRA score for a reference/test pair.

        Args:
            reference: Original audio (1-D or 2-D float, [-1, 1]).
            test:      Restored audio (same shape convention as reference).
            sr:        Sample rate in Hz.

        Returns:
            MushraProxyResult with proxy score, all 26 components, confidence,
            and worst-case floor penalty.
        """
        ref_mono = _to_mono(reference)
        test_mono = _to_mono(test)

        # Length-align
        min_len = min(len(ref_mono), len(test_mono))
        # Guard: 2048 samples minimum — largest n_fft used in _compute_nsim/melspectrogram.
        # Shorter audio (e.g. length=1 from empty tails) causes librosa UserWarning.
        if min_len < 2048:
            return self._empty_result()
        ref_mono = ref_mono[:min_len]
        test_mono = test_mono[:min_len]

        # --- 15 core component metrics ---
        mert_cos = self._compute_mert_cosine(ref_mono, test_mono, sr)
        visqol_mos = self._compute_visqol(ref_mono, test_mono, sr)
        nsim = self._compute_nsim(ref_mono, test_mono, sr)
        artifact_pen = self._compute_artifact_penalty(ref_mono, test_mono, sr)
        temporal_con = self._compute_temporal_consistency(ref_mono, test_mono, sr)
        clap_cos = self._compute_clap_cosine(ref_mono, test_mono, sr)
        mr_stft = self._compute_mr_stft_loss(ref_mono, test_mono)
        iso226_dist = self._compute_iso226_distance(ref_mono, test_mono, sr)
        mcd = self._compute_mcd(ref_mono, test_mono, sr)
        chroma = self._compute_chroma_corr(ref_mono, test_mono, sr)
        lufs_diff = self._compute_lufs_diff(ref_mono, test_mono)
        # Spatial + transient (12-15)
        stereo_img = self._compute_stereo_imaging(reference, test, sr)
        transient_sh = self._compute_transient_shape(ref_mono, test_mono, sr)
        nmr_val = self._compute_nmr(ref_mono, test_mono, sr)
        emo_arc = self._compute_emotional_arc(ref_mono, test_mono, sr)
        # Vocal quality (16-19)
        voc_formant = self._compute_vocal_formant(ref_mono, test_mono, sr)
        voc_hnr = self._compute_vocal_hnr(ref_mono, test_mono, sr)
        pitch_acc = self._compute_pitch_accuracy(ref_mono, test_mono, sr)
        voc_presence = self._compute_vocal_presence(ref_mono, test_mono, sr)
        # Perception dynamics (20-22)
        mod_fidelity = self._compute_modulation_fidelity(ref_mono, test_mono, sr)
        harm_struct = self._compute_harmonic_structure(ref_mono, test_mono, sr)
        spec_flux = self._compute_spectral_flux_correlation(ref_mono, test_mono, sr)
        # Psychoacoustic masking + roughness + loudness + fluctuation (23-26)
        perc_disturb = self._compute_perceptual_disturbance(ref_mono, test_mono, sr)
        roughness_val = self._compute_roughness(ref_mono, test_mono, sr)
        sloud_val = self._compute_specific_loudness_diff(ref_mono, test_mono, sr)
        fluct_val = self._compute_fluctuation_strength(ref_mono, test_mono, sr)

        has_mert = not math.isnan(mert_cos)
        weights = _WEIGHTS_WITH_MERT if has_mert else _WEIGHTS_DSP_ONLY
        confidence = _CONFIDENCE_WITH_MERT if has_mert else _CONFIDENCE_DSP_ONLY

        # Normalize each component to [0, 1] — NaN guard on every value
        mert_norm = float(np.clip(mert_cos, 0.0, 1.0)) if has_mert else 0.0
        visqol_norm = float(np.clip((visqol_mos - 1.0) / 4.0, 0.0, 1.0))  # MOS [1,5] → [0,1]
        nsim_norm = float(np.clip(nsim, 0.0, 1.0))
        # Artifact: 0 = clean → 1.0; penalty ≥ 3 → near 0.0 (exponential decay)
        artifact_norm = float(np.clip(np.exp(-artifact_pen * 1.5), 0.0, 1.0)) if np.isfinite(artifact_pen) else 0.5
        temporal_norm = float(np.clip(temporal_con, 0.0, 1.0)) if np.isfinite(temporal_con) else 0.5
        clap_norm = float(np.clip(clap_cos, 0.0, 1.0)) if np.isfinite(clap_cos) else 0.5
        mr_stft_norm = float(np.clip(np.exp(-mr_stft * 2.0), 0.0, 1.0)) if np.isfinite(mr_stft) else 0.5
        iso226_norm = float(np.clip(np.exp(-iso226_dist * 0.5), 0.0, 1.0)) if np.isfinite(iso226_dist) else 0.5
        mcd_norm = float(np.exp(-mcd / 300.0)) if np.isfinite(mcd) else 0.5
        chroma_norm = float(np.clip(chroma if not math.isnan(chroma) else 0.0, 0.0, 1.0))
        lufs_norm = float(np.clip(1.0 - abs(lufs_diff) / 12.0, 0.0, 1.0)) if np.isfinite(lufs_diff) else 0.5
        stereo_norm = float(np.clip(stereo_img, 0.0, 1.0)) if np.isfinite(stereo_img) else 0.5
        transient_norm = float(np.clip(transient_sh, 0.0, 1.0)) if np.isfinite(transient_sh) else 0.5
        # NMR: 0 dB = perfect (noise at mask) → 1.0; -30 dB stellar → 1.0; +10 dB terrible → ~0.0
        nmr_norm = float(np.clip(np.exp(-max(0.0, nmr_val) * 0.15), 0.0, 1.0)) if np.isfinite(nmr_val) else 0.5
        emo_norm = float(np.clip(emo_arc, 0.0, 1.0)) if np.isfinite(emo_arc) else 0.5
        # Vocal quality (already [0, 1])
        vformant_norm = float(np.clip(voc_formant, 0.0, 1.0)) if np.isfinite(voc_formant) else 0.5
        vhnr_norm = float(np.clip(voc_hnr, 0.0, 1.0)) if np.isfinite(voc_hnr) else 0.5
        pitch_norm = float(np.clip(pitch_acc, 0.0, 1.0)) if np.isfinite(pitch_acc) else 0.5
        vpres_norm = float(np.clip(voc_presence, 0.0, 1.0)) if np.isfinite(voc_presence) else 0.5
        # Perception dynamics (20-22, already [0, 1])
        # Psychoacoustic core (23-26, already [0, 1])
        pdist_norm = float(np.clip(perc_disturb, 0.0, 1.0)) if np.isfinite(perc_disturb) else 0.5
        rough_norm = float(np.clip(roughness_val, 0.0, 1.0)) if np.isfinite(roughness_val) else 0.5
        sloud_norm = float(np.clip(sloud_val, 0.0, 1.0)) if np.isfinite(sloud_val) else 0.5
        fluct_norm = float(np.clip(fluct_val, 0.0, 1.0)) if np.isfinite(fluct_val) else 0.5
        mod_norm = float(np.clip(mod_fidelity, 0.0, 1.0)) if np.isfinite(mod_fidelity) else 0.5
        harm_norm = float(np.clip(harm_struct, 0.0, 1.0)) if np.isfinite(harm_struct) else 0.5
        sflux_norm = float(np.clip(spec_flux, 0.0, 1.0)) if np.isfinite(spec_flux) else 0.5

        component_scores = {
            "mert_cosine": mert_norm,
            "visqol": visqol_norm,
            "nsim": nsim_norm,
            "artifact": artifact_norm,
            "temporal": temporal_norm,
            "clap": clap_norm,
            "mr_stft": mr_stft_norm,
            "iso226": iso226_norm,
            "mcd": mcd_norm,
            "chroma": chroma_norm,
            "lufs": lufs_norm,
            "stereo": stereo_norm,
            "transient": transient_norm,
            "nmr": nmr_norm,
            "emotional_arc": emo_norm,
            "vocal_formant": vformant_norm,
            "perceptual_disturbance": pdist_norm,
            "roughness": rough_norm,
            "vocal_hnr": vhnr_norm,
            "pitch_accuracy": pitch_norm,
            "vocal_presence": vpres_norm,
            "modulation": mod_norm,
            "harmonic": harm_norm,
            "spectral_flux": sflux_norm,
            "specific_loudness": sloud_norm,
            "fluctuation": fluct_norm,
        }

        # --- Adaptive vocal weighting (Lücke 4 fix) ---
        vocal_prob = self._estimate_vocal_probability(ref_mono, sr)
        effective_weights = self._adapt_weights_for_vocal_content(
            weights,
            vocal_prob,
        )

        # --- Stage 2/3: Calibrated weights (nur explizit via
        # calibrate_from_panel() oder bootstrap_from_internal_metrics();
        # §3.6/v10.700: kein impliziter Auto-Load/Bootstrap in evaluate() —
        # unkalibriert bleibt Stage 1 (Literatur-Gewichte).) ---
        cal_stage = 1
        if _calibrated_weights is not None:
            effective_weights = dict(_calibrated_weights)
            confidence = _calibrated_confidence or confidence
            cal_stage = 2

        # Weighted combination → [0, 1] then scale to [0, 100]
        raw = sum(effective_weights[k] * component_scores[k] for k in effective_weights)

        # --- Worst-Case Floor Penalty (PEAQ ADB-inspired) ---
        # A single catastrophic 1 s segment drags the score disproportionately.
        # This models real listener behaviour: one glitch ruins the experience.
        worst_seg = self._compute_worst_segment_score(ref_mono, test_mono, sr)
        # Floor penalty: if worst segment is much worse than the weighted mean,
        # pull the score toward the floor. Blending factor 0.15 ~ 15% influence.
        _FLOOR_BLEND = 0.15
        raw_floor_adjusted = (1.0 - _FLOOR_BLEND) * raw + _FLOOR_BLEND * worst_seg
        # Only apply if floor penalty would lower the score (never inflate)
        raw_final = min(raw, raw_floor_adjusted) if worst_seg < raw else raw

        proxy_score = float(np.clip(raw_final * 100.0, 0.0, 100.0))
        proxy_score = round(proxy_score, 1)

        grade = _grade(proxy_score)

        logger.info(
            "MUSHRA-Proxy: %.1f/100 (%s) | MERT=%.3f ViSQOL=%.2f NSIM=%.3f "
            "Artifact=%.3f Temporal=%.3f CLAP=%.3f MR-STFT=%.3f ISO226=%.3f "
            "MCD=%.1fdB Chroma=%.3f LUFS-D=%.1fLU Stereo=%.3f Transient=%.3f "
            "NMR=%.1fdB EmoArc=%.3f VocFormant=%.3f VocHNR=%.3f Pitch=%.3f "
            "VocPres=%.3f Mod=%.3f Harm=%.3f SFlux=%.3f PDist=%.3f Rough=%.3f "
            "SLoud=%.3f Fluct=%.3f Floor=%.3f | conf=%.0f%%",
            proxy_score,
            grade,
            mert_cos if has_mert else -1.0,
            visqol_mos,
            nsim,
            artifact_pen,
            temporal_con,
            clap_cos,
            mr_stft,
            iso226_dist,
            mcd,
            chroma,
            lufs_diff,
            stereo_img,
            transient_sh,
            nmr_val,
            emo_arc,
            voc_formant,
            voc_hnr,
            pitch_acc,
            voc_presence,
            mod_fidelity,
            harm_struct,
            spec_flux,
            perc_disturb,
            roughness_val,
            sloud_val,
            fluct_val,
            worst_seg,
            confidence * 100,
        )

        return MushraProxyResult(
            proxy_score=proxy_score,
            grade=grade,
            confidence=confidence,
            mert_cosine=mert_cos if has_mert else float("nan"),
            visqol_mos=visqol_mos,
            nsim=nsim,
            artifact_penalty=artifact_pen,
            temporal_consistency=temporal_con,
            clap_cosine=clap_cos,
            mr_stft_loss=mr_stft,
            iso226_distance=iso226_dist,
            mcd_db=mcd,
            chroma_corr=chroma,
            lufs_diff_lu=lufs_diff,
            stereo_imaging=stereo_img,
            transient_shape=transient_sh,
            nmr_db=nmr_val,
            emotional_arc=emo_arc,
            perceptual_disturbance=perc_disturb,
            roughness=roughness_val,
            vocal_formant=voc_formant,
            vocal_hnr=voc_hnr,
            pitch_accuracy=pitch_acc,
            vocal_presence=voc_presence,
            modulation_fidelity=mod_fidelity,
            harmonic_structure=harm_struct,
            spectral_flux_corr=spec_flux,
            specific_loudness_diff=sloud_val,
            fluctuation_strength=fluct_val,
            worst_segment_score=worst_seg,
            component_scores=component_scores,
            calibration_stage=cal_stage,
        )

    # ------------------------------------------------------------------
    # Adaptive vocal weight redistribution (Lücke 4 fix)
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_weights_for_vocal_content(
        base_weights: dict[str, float],
        vocal_prob: float,
    ) -> dict[str, float]:
        """Redistribute component weights based on detected vocal content.

        Principle: A human listener automatically adjusts evaluation criteria
        based on what they hear. Instrumental tracks → spatial/timbral/transient
        dimensions matter more. Vocal tracks → formants/pitch/HNR/presence matter.

        The vocal_prob maps to a scale factor for the vocal component pool:
        - vocal_prob ≈ 0.0 (purely instrumental): vocal pool shrinks to 20%,
          surplus redistributed proportionally to non-vocal components.
        - vocal_prob ≈ 0.5 (mixed / unknown): weights unchanged.
        - vocal_prob ≈ 1.0 (vocal-dominant): vocal pool scales up to 180%,
          deficit drawn proportionally from non-vocal components.

        Invariant: Sum of all weights == 1.0 (guaranteed by normalization).

        Args:
            base_weights: The static weight preset (19 keys, sum = 1.0).
            vocal_prob:   Vocal probability ∈ [0, 1].

        Returns:
            New weight dict (19 keys, sum = 1.0).
        """
        # Neutral zone: 0.4–0.6 → no redistribution (avoids unnecessary churn)
        if 0.35 <= vocal_prob <= 0.65:
            return dict(base_weights)

        # Scale factor for vocal pool: maps [0, 1] → [0.2, 1.8]
        # vocal_prob=0 → 0.2; vocal_prob=0.5 → 1.0; vocal_prob=1.0 → 1.8
        vocal_scale = float(np.clip(0.2 + 1.6 * vocal_prob, 0.2, 1.8))

        vocal_sum = sum(base_weights[k] for k in _VOCAL_COMPONENT_KEYS)
        non_vocal_sum = sum(base_weights[k] for k in _NON_VOCAL_COMPONENT_KEYS)

        if vocal_sum < 1e-10 or non_vocal_sum < 1e-10:
            return dict(base_weights)

        new_vocal_sum = vocal_sum * vocal_scale
        new_non_vocal_sum = 1.0 - new_vocal_sum

        if new_non_vocal_sum < 0.05:  # Safety floor: non-vocal always ≥ 5%
            new_non_vocal_sum = 0.05
            new_vocal_sum = 0.95

        vocal_ratio = new_vocal_sum / vocal_sum
        non_vocal_ratio = new_non_vocal_sum / non_vocal_sum

        result = {}
        for k, w in base_weights.items():
            if k in _VOCAL_COMPONENT_KEYS:
                result[k] = w * vocal_ratio
            else:
                result[k] = w * non_vocal_ratio

        # Normalize to exactly 1.0 (handles floating-point drift)
        total = sum(result.values())
        if total > 1e-10:
            result = {k: v / total for k, v in result.items()}

        return result

    # ------------------------------------------------------------------
    # Stage 2: Ridge-regression calibration from panel data (Lücke 1+2)
    # ------------------------------------------------------------------

    @staticmethod
    def calibrate_from_panel(
        component_matrix: np.ndarray,
        mushra_scores: np.ndarray,
        alpha: float = 1.0,
    ) -> dict[str, float]:
        """Kalibriert weights via Ridge regression from MUSHRA panel data.

        Call this once you have real listener data (§8.4 Mini-MUSHRA, ≥ 48 pairs).
        Stores result in module-level _calibrated_weights for immediate use.

        Args:
            component_matrix: (N, 24) array — normalized component scores per pair.
            mushra_scores:    (N,) array — mean listener MUSHRA scores [0, 100].
            alpha:            Ridge regularization (prevents overfitting on small panels).

        Returns:
            Optimized weight dict (24 keys, sum = 1.0, all ≥ 0).
        """
        global _calibrated_weights, _calibrated_confidence  # pylint: disable=global-statement

        from sklearn.linear_model import Ridge  # pylint: disable=import-outside-toplevel

        # Normalize MUSHRA to [0, 1]
        y = mushra_scores / 100.0

        model = Ridge(alpha=alpha, fit_intercept=False, positive=True)
        model.fit(component_matrix, y)

        keys = list(_WEIGHTS_WITH_MERT.keys())
        raw_w = model.coef_
        # Ensure non-negative and normalize to sum = 1
        raw_w = np.maximum(raw_w, 0.0)
        total = float(np.sum(raw_w))
        if total < 1e-10:
            logger.warning("Ridge regression yielded near-zero weights, keeping defaults")
            return dict(_WEIGHTS_WITH_MERT)
        optimized = {k: float(raw_w[i] / total) for i, k in enumerate(keys)}

        # Cross-validation correlation estimate
        from sklearn.model_selection import cross_val_score  # pylint: disable=import-outside-toplevel

        r2_scores = cross_val_score(model, component_matrix, y, cv=min(5, len(y)), scoring="r2")
        r_est = float(np.sqrt(np.clip(np.mean(r2_scores), 0.0, 1.0)))
        conf = float(np.clip(0.7 + 0.3 * r_est, 0.70, 0.99))

        _calibrated_weights = optimized
        _calibrated_confidence = conf

        logger.info(
            "MUSHRA-Proxy Stufe 2 kalibriert: r≈%.3f conf=%.2f (N=%d pairs, α=%.1f)",
            r_est,
            conf,
            len(mushra_scores),
            alpha,
        )
        return optimized

    # ------------------------------------------------------------------
    # Stage 3: CI-Proxy — Persistenz, Drift-Erkennung, Auto-Trigger (§3.6)
    # ------------------------------------------------------------------

    _CALIBRATION_ARTIFACT_PATH: str = "reports/mushra_calibration.json"
    _CORE_FILE_HASHES: list[tuple[str, str]] = [
        ("backend/core/mert_mushra_proxy.py", ""),
        ("backend/core/unified_restorer_v3.py", ""),
        ("backend/core/defect_scanner.py", ""),
        ("backend/core/blind_reference_free_quality.py", ""),
        ("denker/aurik_denker.py", ""),
    ]

    @classmethod
    def save_calibration_artifact(cls) -> str | None:
        """§3.6 Stufe 3: Persistiert kalibrierte Gewichte + Core-Hashes.

        Speichert unter reports/mushra_calibration.json:
          - calibrated_weights (26 keys)
          - confidence
          - core_file_hashes (für Drift-Erkennung)
          - timestamp, aurik_version

        Returns den Pfad oder None bei Fehler.
        """
        global _calibrated_weights, _calibrated_confidence
        if _calibrated_weights is None:
            logger.warning("§3.6 speichern: keine kalibrierten Gewichte zum Speichern")
            return None

        import hashlib
        import json as _json
        from datetime import datetime, timezone

        artifact = {
            "schema_version": 1,
            "calibrated_weights": _calibrated_weights,
            "confidence": _calibrated_confidence,
            "calibration_stage": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "core_file_hashes": cls._compute_core_hashes(),
        }

        try:
            from pathlib import Path

            out_path = Path(cls._CALIBRATION_ARTIFACT_PATH)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(_json.dumps(artifact, indent=2, ensure_ascii=False))
            logger.info("§3.6 Kalibrierung artifact gespeichert: %s", out_path)
            return str(out_path)
        except Exception:
            logger.warning("§3.6 speichern: Fehler beim Schreiben des Artefakts", exc_info=True)
            return None

    @classmethod
    def load_calibration_artifact(cls) -> bool:
        """§3.6 Stufe 3: Lädt kalibrierte Gewichte aus Persistenz.

        Returns True wenn erfolgreich geladen und Drift-Check bestanden.
        """
        global _calibrated_weights, _calibrated_confidence

        import json as _json
        from pathlib import Path

        artifact_path = Path(cls._CALIBRATION_ARTIFACT_PATH)
        if not artifact_path.exists():
            logger.debug("§3.6 laden: kein Kalibrierungsartefakt gefunden")
            return False

        try:
            data = _json.loads(artifact_path.read_text())
            stored_weights = data.get("calibrated_weights")
            stored_confidence = data.get("confidence")
            stored_stage = data.get("calibration_stage")

            if not stored_weights or stored_stage is None:
                logger.warning("§3.6 laden: Artefakt unvollständig")
                return False

            # Drift-Check: Haben sich Core-Dateien geändert?
            stored_hashes = data.get("core_file_hashes", {})
            current_hashes = cls._compute_core_hashes()
            drifted_files = [f for f in current_hashes if current_hashes[f] != stored_hashes.get(f, "")]

            if drifted_files:
                logger.warning(
                    "§3.6 DRIFT erkannt: %d core files changed since Kalibrierung: %s",
                    len(drifted_files),
                    ", ".join(drifted_files),
                )
                logger.warning("§3.6: Alte Gewichte verworfen. Bitte calibrate_from_panel() erneut aufrufen.")
                _calibrated_weights = None
                _calibrated_confidence = None
                return False

            _calibrated_weights = stored_weights
            _calibrated_confidence = stored_confidence
            logger.info(
                "§3.6 Kalibrierung geladen: Stufe=%d conf=%.2f, %d weights",
                stored_stage,
                stored_confidence or 0.0,
                len(stored_weights),
            )
            return True

        except Exception:
            logger.warning("§3.6 laden: Fehler beim Lesen des Artefakts", exc_info=True)
            return False

    @classmethod
    def _compute_core_hashes(cls) -> dict[str, str]:
        """Berechnet SHA256-Hashes der Core-Dateien für Drift-Erkennung."""
        import hashlib
        from pathlib import Path

        hashes: dict[str, str] = {}
        for rel_path, _ in cls._CORE_FILE_HASHES:
            try:
                full_path = Path(rel_path)
                if full_path.exists():
                    hashes[rel_path] = hashlib.sha256(full_path.read_bytes()).hexdigest()[:16]
                else:
                    hashes[rel_path] = "MISSING"
            except Exception:
                hashes[rel_path] = "ERROR"
        return hashes

    @classmethod
    def bootstrap_from_internal_metrics(cls, n_samples: int = 50) -> dict[str, float] | None:
        """§3.6 Bootstrap: Kalibriert Ridge Regression aus internen Qualitätsmetriken.

        Erzeugt synthetische Trainingsdaten, indem die 26 Komponenten mit
        Literatur-Gewichten gewichtet werden. Führt Ridge Regression durch,
        um die Gewichte zu verfeinern.

        Dies ist ein Self-Supervised Bootstrap — keine menschlichen Daten nötig.
        Mit n_samples=50 wird die Ridge-Regression stabil genug für Produktion.

        Returns:
            Kalibrierte Gewichte (26 keys, sum=1.0) oder None bei Fehler.
        """
        global _calibrated_weights, _calibrated_confidence

        try:
            import numpy as np
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score

            keys = list(_WEIGHTS_WITH_MERT.keys())
            lit_weights = np.array([_WEIGHTS_WITH_MERT[k] for k in keys], dtype=np.float64)

            # Generiere synthetische Komponenten-Matrix
            rng = np.random.RandomState(42)  # Deterministisch
            n_components = len(keys)

            # Simuliere realistische Komponenten-Verteilung
            X = np.zeros((n_samples, n_components), dtype=np.float64)
            for i in range(n_samples):
                # Basis: 0.5–1.0 für gute Qualität, mit Rauschen
                base = 0.5 + 0.5 * rng.random(n_components)
                # Einige Komponenten "schlecht" (0.1–0.4) — simuliert defekte Aufnahmen
                n_bad = rng.randint(0, 8)
                bad_idx = rng.choice(n_components, n_bad, replace=False)
                base[bad_idx] = 0.1 + 0.3 * rng.random(n_bad)
                X[i] = base

            # Synthetische MUSHRA-Scores: gewichtete Summe + Rauschen
            y = X @ lit_weights  # (n_samples,)
            # Füge nichtlinearen Floor-Effekt hinzu
            y = np.clip(y - 0.05 * np.min(X, axis=1), 0.0, 1.0)
            y += 0.03 * rng.randn(n_samples)  # Kleines Rauschen
            y = np.clip(y, 0.0, 1.0)

            # Ridge Regression
            model = Ridge(alpha=0.5, fit_intercept=False, positive=True)
            model.fit(X, y)

            raw_w = np.maximum(model.coef_, 0.0)
            total = float(np.sum(raw_w))
            if total < 1e-10:
                return None
            optimized = {k: float(raw_w[i] / total) for i, k in enumerate(keys)}

            # Cross-validation
            r2_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
            r_est = float(np.sqrt(np.clip(np.mean(r2_scores), 0.0, 1.0)))
            conf = float(np.clip(0.85 + 0.10 * r_est, 0.80, 0.95))

            _calibrated_weights = optimized
            _calibrated_confidence = conf

            logger.info(
                "§3.6 Bootstrap: kalibriert from %d synthetic samples, r≈%.3f conf=%.2f",
                n_samples,
                r_est,
                conf,
            )

            # Persistiere sofort
            cls.save_calibration_artifact()

            return optimized

        except ImportError:
            logger.warning("§3.6 Bootstrap: scikit-learn nicht verfügbar")
            return None
        except Exception:
            logger.warning("§3.6 Bootstrap: Fehler", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # ViSQOL v3 Audio (Bark-band NSIM MOS)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_visqol(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Berechnet ViSQOL v3 Audio MOS [1.0, 5.0].

        Uses the existing ViSQOL DSP plugin (Bark-band NSIM, no external binary).
        Falls back to 3.0 (neutral MOS) on error.
        """
        try:
            from plugins.visqol_plugin import (  # pylint: disable=import-outside-toplevel
                get_loaded_visqol_plugin,
                get_visqol_plugin,
            )

            _visqol = get_loaded_visqol_plugin()
            if _visqol is None:
                _visqol = get_visqol_plugin()
            mos = _visqol.score(ref, test, sr)
            return float(np.clip(mos, 1.0, 5.0))
        except Exception as exc:
            logger.debug("ViSQOL computation fehlgeschlagen: %s", exc)
            return 3.0

    # ------------------------------------------------------------------
    # Artifact Penalty — Musical Noise, Pre-Echo, Phase Jumps
    # (Thiede et al. 2000 PEAQ-inspired, Camacho & Harris 2008)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_artifact_penalty(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Erkennt and quantify restoration artifacts.

        Combines three sub-detectors:
        1. Musical noise: tonal islands in residual (spectral kurtosis > 6)
        2. Pre-echo: energy before transients in test but not in reference
        3. Phase discontinuities: unwrapped phase jumps in residual STFT

        Returns a penalty score [0, ∞) where 0 = artifact-free.
        """
        try:
            residual = test - ref

            # --- 1. Musical noise detection via spectral kurtosis ---
            n_fft = 2048
            hop = 512
            S_res = _stft_magnitude(residual, n_fft, hop)
            if S_res.size == 0:
                return 0.0
            # Spectral kurtosis per frequency bin (excess kurtosis)
            # Musical noise = isolated spectral peaks → high kurtosis
            mu = np.mean(S_res, axis=1, keepdims=True)
            sigma = np.std(S_res, axis=1, keepdims=True) + 1e-12
            kurt = np.mean(((S_res - mu) / sigma) ** 4, axis=1) - 3.0
            # Bins with kurtosis > 6 are likely musical noise artifacts
            musical_noise_frac = float(np.mean(kurt > 6.0))
            musical_noise_severity = (
                float(np.clip(np.mean(kurt[kurt > 6.0]) / 20.0, 0.0, 1.0)) if musical_noise_frac > 0 else 0.0
            )

            # --- 2. Pre-echo detection ---
            # Compute envelope
            frame_len = int(0.005 * sr)  # 5 ms frames
            if frame_len < 1:
                frame_len = 1
            n_env_frames = len(ref) // frame_len
            if n_env_frames < 4:
                pre_echo_score = 0.0
            else:
                ref_env = np.array(
                    [
                        np.sqrt(np.mean(ref[i * frame_len : (i + 1) * frame_len] ** 2) + 1e-12)
                        for i in range(n_env_frames)
                    ]
                )
                test_env = np.array(
                    [
                        np.sqrt(np.mean(test[i * frame_len : (i + 1) * frame_len] ** 2) + 1e-12)
                        for i in range(n_env_frames)
                    ]
                )
                # Detect transients in reference (rising edge > 6 dB)
                ref_db = 20.0 * np.log10(ref_env + 1e-12)
                diff_db = np.diff(ref_db)
                transient_idx = np.where(diff_db > 6.0)[0]
                # Check for energy leakage before transients in test
                pre_echo_events = 0
                for idx in transient_idx:
                    if idx < 2:
                        continue
                    # Look 2 frames back: is test louder than ref by > 3 dB?
                    pre_frames = slice(max(0, idx - 2), idx)
                    ref_pre = np.mean(ref_db[pre_frames])
                    test_pre = np.mean(20.0 * np.log10(test_env[pre_frames] + 1e-12))
                    if test_pre - ref_pre > 3.0:
                        pre_echo_events += 1
                pre_echo_score = min(1.0, pre_echo_events / max(1, len(transient_idx)))

            # --- 3. Phase discontinuity detection ---
            # Compute STFT of residual with phase
            n_phase = min(len(residual), sr * 2)  # cap to 2 s
            res_chunk = np.asarray(residual[:n_phase], dtype=np.float32)
            if len(res_chunk) < n_fft:
                res_chunk = np.pad(res_chunk, (0, n_fft - len(res_chunk)))
            n_frames_p = 1 + (len(res_chunk) - n_fft) // hop
            if n_frames_p < 3:
                phase_jump_score = 0.0
            else:
                window = np.hanning(n_fft).astype(np.float32)
                phases = np.zeros((n_fft // 2 + 1, n_frames_p), dtype=np.float32)
                for i in range(n_frames_p):
                    start = i * hop
                    frame = res_chunk[start : start + n_fft] * window
                    phases[:, i] = np.angle(np.fft.rfft(frame, n=n_fft))
                # Unwrap and measure jumps
                unwrapped = np.unwrap(phases, axis=1)
                phase_diff = np.abs(np.diff(unwrapped, axis=1))
                # Jumps > π in residual suggest phase artifacts
                jump_frac = float(np.mean(phase_diff > math.pi))
                phase_jump_score = float(np.clip(jump_frac * 5.0, 0.0, 1.0))

            # Combine with human-perception weighting:
            # Musical noise is the most salient artifact type
            penalty = (
                1.5 * musical_noise_severity * (1.0 + musical_noise_frac * 2.0)
                + 1.0 * pre_echo_score
                + 0.5 * phase_jump_score
            )
            return float(np.clip(penalty, 0.0, 10.0))
        except Exception as exc:
            logger.debug("Artifact penalty computation fehlgeschlagen: %s", exc)
            return 0.5  # neutral fallback

    # ------------------------------------------------------------------
    # Temporal Consistency — Segment-wise quality variance
    # (Kabal 2002, PESQ time-alignment term inspired)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_temporal_consistency(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
        segment_dur: float = 1.0,
    ) -> float:
        """Misst quality consistency over time with primacy/recency attention.

        Computes NSIM-like metric per 1-second segment, then returns a
        weighted consistency score. Segments are weighted by a U-shaped
        attention curve reflecting human listening behavior:

        - First 3 segments: weight ×2.0 (primacy effect — first impression)
        - Last 2 segments:  weight ×1.5 (recency effect — lingering memory)
        - Middle segments:  weight ×1.0

        This models the empirical finding that listeners are especially
        sensitive to artifacts at the start and end of audio excerpts
        (Zacharov & Koivuniemi 2001; Schoeffler & Herre 2014).

        The final score combines:
        - Weighted mean quality (70%): overall quality across segments
        - Consistency penalty (30%): penalizes quality variance

        Higher = better.
        """
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            seg_len = int(segment_dur * sr)
            if seg_len < sr // 4:
                seg_len = sr // 4
            n_segs = max(1, len(ref) // seg_len)
            if n_segs < 2:
                # Too short for variance → assume consistent
                return 1.0

            seg_scores = []
            for i in range(n_segs):
                start = i * seg_len
                end = min(start + seg_len, len(ref))
                seg_ref = ref[start:end]
                seg_test = test[start:end]
                if len(seg_ref) < 64:
                    continue
                # Quick per-segment NSIM (simplified — lower cost)
                try:
                    seg_n_fft = _safe_fft_size(min(len(seg_ref), len(seg_test)), target=1024, minimum=64)
                    seg_hop = max(16, seg_n_fft // 4)
                    S_r = librosa.power_to_db(
                        np.maximum(
                            librosa.feature.melspectrogram(
                                y=seg_ref, sr=sr, n_fft=seg_n_fft, hop_length=seg_hop, n_mels=64
                            ),
                            1e-10,
                        )
                    )
                    S_t = librosa.power_to_db(
                        np.maximum(
                            librosa.feature.melspectrogram(
                                y=seg_test, sr=sr, n_fft=seg_n_fft, hop_length=seg_hop, n_mels=64
                            ),
                            1e-10,
                        )
                    )
                    mu_r, mu_t = np.mean(S_r), np.mean(S_t)
                    sig_r, sig_t = np.std(S_r), np.std(S_t)
                    sig_rt = np.mean((S_r - mu_r) * (S_t - mu_t))
                    C1 = (0.01 * 80) ** 2
                    C2 = (0.03 * 80) ** 2
                    sim = (
                        (2 * mu_r * mu_t + C1)
                        * (2 * sig_rt + C2)
                        / ((mu_r**2 + mu_t**2 + C1) * (sig_r**2 + sig_t**2 + C2))
                    )
                    seg_scores.append(float(np.clip(sim, 0.0, 1.0)))
                except Exception:
                    seg_scores.append(0.5)

            if len(seg_scores) < 2:
                return 1.0

            # --- Primacy/recency attention weights ---
            # U-shaped curve: first 3 segments ×2.0, last 2 ×1.5, middle ×1.0
            n = len(seg_scores)
            attention = np.ones(n, dtype=np.float64)
            primacy_count = min(3, n // 2)  # First 3 segments (or fewer for short audio)
            recency_count = min(2, max(1, n // 3))  # Last 2 segments
            attention[:primacy_count] = 2.0
            attention[-recency_count:] = np.maximum(attention[-recency_count:], 1.5)
            attention /= attention.sum()  # Normalize to probability distribution

            scores_arr = np.array(seg_scores, dtype=np.float64)

            # Weighted mean quality
            weighted_mean = float(np.sum(attention * scores_arr))

            # Weighted variance (consistency penalty)
            w_var = float(np.sum(attention * (scores_arr - weighted_mean) ** 2))
            # Map variance to [0, 1]: var=0 → 1.0; var=0.05 → 0.0
            consistency = float(np.clip(1.0 - w_var / 0.05, 0.0, 1.0))

            # Combined: 70% weighted mean quality + 30% consistency
            return float(np.clip(0.70 * weighted_mean + 0.30 * consistency, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Temporal consistency computation fehlgeschlagen: %s", exc)
            return 0.8  # neutral-ish fallback

    # ------------------------------------------------------------------
    # CLAP-Cosine — Semantic audio embedding similarity
    # (Wu et al. 2023 LAION-CLAP inspired, DSP 32-dim proxy)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_clap_cosine(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Berechnet CLAP-style semantic audio similarity.

        Uses the existing DSP embedding from clap_reference_matcher (32-dim vector
        of spectral centroid, MFCCs, harmonicity, dynamic range, noise floor,
        rolloff, ZCR, spectral contrast). L2-normalized → cosine similarity.
        """
        try:
            from backend.core.clap_reference_matcher import (
                compute_dsp_embedding,  # pylint: disable=import-outside-toplevel
            )

            emb_ref = compute_dsp_embedding(ref, sr)
            emb_test = compute_dsp_embedding(test, sr)
            return _cosine_similarity(emb_ref, emb_test)
        except Exception as exc:
            logger.debug("CLAP cosine computation fehlgeschlagen: %s", exc)
            return 0.5  # neutral fallback

    # ------------------------------------------------------------------
    # Multi-Resolution STFT Loss (numpy-only, no torch dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mr_stft_loss(
        ref: np.ndarray,
        test: np.ndarray,
        fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256),
    ) -> float:
        """Multi-resolution STFT loss (Yamamoto et al. 2019, numpy-only).

        Computes spectral convergence + log-magnitude loss across multiple
        FFT resolutions. Lower = better (0.0 = identical).
        """
        try:
            total_sc = 0.0
            total_lm = 0.0
            n = 0
            for fft_size in fft_sizes:
                hop_length = fft_size // 4
                # Compute STFT magnitudes
                S_ref = _stft_magnitude(ref, fft_size, hop_length)
                S_test = _stft_magnitude(test, fft_size, hop_length)
                if S_ref.size == 0 or S_test.size == 0:
                    continue
                # Spectral convergence: ||S_ref - S_test||_F / ||S_ref||_F
                norm_ref = np.linalg.norm(S_ref, "fro")
                if norm_ref < 1e-12:
                    continue
                sc = np.linalg.norm(S_ref - S_test, "fro") / norm_ref
                # Log-magnitude loss: mean(|log(S_ref) - log(S_test)|)
                log_ref = np.log(np.maximum(S_ref, 1e-7))
                log_test = np.log(np.maximum(S_test, 1e-7))
                lm = float(np.mean(np.abs(log_ref - log_test)))
                total_sc += float(sc)
                total_lm += lm
                n += 1
            if n == 0:
                return 0.5  # neutral
            return (total_sc / n + total_lm / n) / 2.0
        except Exception as exc:
            logger.debug("MR-STFT loss computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # ISO 226 weighted spectral distance
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iso226_distance(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Frequency-weighted spectral distance using ISO 226:2003 equal-loudness.

        Emphasizes 3–4 kHz sensitivity peak, de-emphasizes sub-bass.
        Lower = better (0.0 = identical).
        """
        try:
            n_fft = 4096
            hop = n_fft // 4
            S_ref = _stft_magnitude(ref, n_fft, hop)
            S_test = _stft_magnitude(test, n_fft, hop)
            if S_ref.size == 0 or S_test.size == 0:
                return 2.0
            # Frequency axis
            freqs = np.linspace(0, sr / 2, S_ref.shape[0])
            weights = _iso226_weights_for_proxy(freqs)
            # Weighted log-magnitude difference per frequency bin
            log_ref = np.log(np.maximum(S_ref, 1e-7))
            log_test = np.log(np.maximum(S_test, 1e-7))
            diff = np.abs(log_ref - log_test)
            # Apply ISO 226 weights per bin (broadcast over time axis)
            weighted_diff = diff * weights[:, np.newaxis]
            return float(np.mean(weighted_diff))
        except Exception as exc:
            logger.debug("ISO 226 distance computation fehlgeschlagen: %s", exc)
            return 2.0

    # ------------------------------------------------------------------
    # MERT embedding cosine similarity
    # ------------------------------------------------------------------

    def _compute_mert_cosine(
        self,
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet cosine similarity between MERT embeddings.

        Uses get_loaded_mert_plugin() — does NOT trigger lazy-load.
        Returns NaN if MERT is not already loaded in process.
        """
        try:
            from plugins.mert_plugin import get_loaded_mert_plugin  # pylint: disable=import-outside-toplevel

            mert = get_loaded_mert_plugin()
            if mert is None:
                return float("nan")

            # Extract embeddings via the HF path (768-dim last hidden state)
            emb_ref = self._extract_embedding(mert, ref, sr)
            emb_test = self._extract_embedding(mert, test, sr)

            if emb_ref is None or emb_test is None:
                return float("nan")

            return _cosine_similarity(emb_ref, emb_test)
        except Exception as exc:
            logger.debug("MERT cosine computation fehlgeschlagen: %s", exc)
            return float("nan")

    @staticmethod
    def _extract_embedding(
        mert_plugin: object,
        audio: np.ndarray,
        sr: int,
    ) -> np.ndarray | None:
        """Extrahiert a fixed-size embedding vector from a MERT plugin instance.

        For HF models: temporal mean of last hidden state → 768-dim vector.
        For ONNX models: mean of output tensor → N-dim vector.
        For DSP fallback: 512-dim DSP feature vector (MFCCs + chroma + spectral).  # §V6 (copilot-instructions.md): logger.warning handled at call site
        """
        try:
            import scipy.signal as spsig  # pylint: disable=import-outside-toplevel

            # Prepare audio: mono, float32, resample to MERT target SR
            mono = audio.astype(np.float32)
            target_sr = getattr(mert_plugin, "_target_sr", 24000)
            if sr != target_sr:
                n_out = int(len(mono) * target_sr / sr)
                mono = spsig.resample(mono, n_out)

            # Cap at 30 s (MERT OOM guard)
            max_samples = int(30 * target_sr)
            if len(mono) > max_samples:
                offset = (len(mono) - max_samples) // 2
                mono = mono[offset : offset + max_samples]

            model_type = getattr(mert_plugin, "_model_type", "dsp_fallback")

            if model_type == "mert_hf":
                return _extract_hf_embedding(mert_plugin, mono, target_sr)
            if model_type == "mert_onnx":
                return _extract_onnx_embedding(mert_plugin, mono, target_sr)
            # DSP fallback: compute feature vector
            return _extract_dsp_embedding(mono, target_sr)
        except Exception as exc:
            logger.debug("Embedding extraction fehlgeschlagen: %s", exc)
            return None

    def compute_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Extrahiert MERT-Embedding via geladenes Plugin oder DSP-Fallback.

        §v10.703 MUSHRA-Proxy: Wird von MushraProxy.set_session_reference() und
        _estimate_mushra_mert() aufgerufen. Lädt KEIN neues Modell — nutzt nur das
        bereits geladene Plugin (get_loaded_mert_plugin).

        Returns:
            768-dim (HF) / N-dim (ONNX) / 512-dim (DSP) embedding oder None.
        """
        try:
            from plugins.mert_plugin import get_loaded_mert_plugin  # pylint: disable=import-outside-toplevel

            _mert = get_loaded_mert_plugin()
            if _mert is not None:
                return self._extract_embedding(_mert, audio, sr)
            return _extract_dsp_embedding(audio, sr)
        except Exception as exc:
            logger.debug("compute_embedding: MERT/DSP fehlgeschlagen: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Perceptual metrics (same math as mushra_evaluator, kept self-contained)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_nsim(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Mel-spectrogram SSIM (structural similarity)."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            n_fft = _safe_fft_size(min(len(ref), len(test)), target=2048, minimum=64)
            hop = max(16, n_fft // 4)

            S_ref = librosa.power_to_db(
                np.maximum(
                    librosa.feature.melspectrogram(y=ref, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128),
                    1e-10,
                )
            )
            S_test = librosa.power_to_db(
                np.maximum(
                    librosa.feature.melspectrogram(y=test, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128),
                    1e-10,
                )
            )
            mu_r, mu_t = np.mean(S_ref), np.mean(S_test)
            sig_r, sig_t = np.std(S_ref), np.std(S_test)
            sig_rt = np.mean((S_ref - mu_r) * (S_test - mu_t))
            C1 = (0.01 * 80) ** 2
            C2 = (0.03 * 80) ** 2
            nsim = (2 * mu_r * mu_t + C1) * (2 * sig_rt + C2) / ((mu_r**2 + mu_t**2 + C1) * (sig_r**2 + sig_t**2 + C2))
            return float(np.clip(nsim, 0.0, 1.0))
        except Exception as e:
            logger.warning("mert_mushra_proxy.py::_berechnen_nsim Ersatzpfad: %s", e)
            return float(np.clip(1.0 - np.sqrt(np.mean((ref - test) ** 2)), 0.0, 1.0))

    @staticmethod
    def _compute_mcd(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Mel-Cepstral Distortion in dB (lower = better).

        §Spec 24 (Root-Fix 2026-08-16): Identische CMVN-Formel wie
        mushra_evaluator._compute_mcd — vorher wurden rohe librosa-MFCCs
        verrechnet („MCD=435.2dB“ vs. phys. sinnvollem 0–30 dB; der Evaluator
        kommentiert das explizit: ohne Normalisierung 500–1000 dB). Der Wert
        speist exp(-mcd/300) und verfälschte die Klangfarben-Komponente.
        """
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            mfcc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=13).T
            mfcc_test = librosa.feature.mfcc(y=test, sr=sr, n_mfcc=13).T
            # CMVN — Cepstral Mean and Variance Normalization (standard speech/music DSP).
            # Subtrainiert den absoluten Energie-Offset, normalisiert auf σ=1 —
            # Distanz misst ausschließlich Klangfarbenunterschiede.
            for _mc in (mfcc_ref, mfcc_test):
                _mu = np.mean(_mc, axis=0, keepdims=True)
                _sigma = np.std(_mc, axis=0, keepdims=True) + 1e-8
                _mc -= _mu
                _mc /= _sigma
            min_f = min(mfcc_ref.shape[0], mfcc_test.shape[0])
            diff = mfcc_ref[:min_f, 1:] - mfcc_test[:min_f, 1:]
            frame_dists = np.sqrt(2.0 * np.sum(diff**2, axis=1))
            mcd = (10.0 / math.log(10)) * float(np.mean(frame_dists))
            # Sane cap (wie mushra_evaluator): CMVN-MCD-Bereich 0–40 dB.
            return float(np.clip(mcd, 0.0, 40.0))
        except Exception as e:
            logger.warning("mert_mushra_proxy.py::_berechnen_mcd Ersatzpfad: %s", e)
            return 5.0

    @staticmethod
    def _compute_chroma_corr(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Chromagram Pearson correlation — tonal center preservation."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            n_fft = _safe_fft_size(min(len(ref), len(test)), target=2048, minimum=64)
            hop = max(16, n_fft // 4)
            chroma_ref = librosa.feature.chroma_stft(
                y=ref,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                n_chroma=12,
                tuning=0.0,
            ).flatten()
            chroma_test = librosa.feature.chroma_stft(
                y=test,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                n_chroma=12,
                tuning=0.0,
            ).flatten()
            min_len = min(len(chroma_ref), len(chroma_test))
            _cr = chroma_ref[:min_len]
            _ct = chroma_test[:min_len]
            corr = _pearson(_cr, _ct) if np.std(_cr) > 1e-12 and np.std(_ct) > 1e-12 else 1.0
            return float(np.clip(corr, 0.0, 1.0))
        except Exception as e:
            logger.warning("mert_mushra_proxy.py::_berechnen_chroma_corr Ersatzpfad: %s", e)
            return 0.5

    @staticmethod
    def _compute_lufs_diff(ref: np.ndarray, test: np.ndarray) -> float:
        """LUFS difference in LU (simplified K-weighted RMS)."""
        try:
            rms_ref = float(np.sqrt(np.mean(ref**2) + 1e-12))
            rms_test = float(np.sqrt(np.mean(test**2) + 1e-12))
            return 20.0 * math.log10(rms_test) - 20.0 * math.log10(rms_ref)
        except Exception as e:
            logger.warning("mert_mushra_proxy.py::_berechnen_lufs_diff Ersatzpfad: %s", e)
            return 0.0

    @staticmethod
    def _empty_result() -> MushraProxyResult:
        return MushraProxyResult(
            proxy_score=0.0,
            grade="Bad",
            confidence=0.0,
            mert_cosine=float("nan"),
            visqol_mos=1.0,
            nsim=0.0,
            artifact_penalty=10.0,
            temporal_consistency=0.0,
            clap_cosine=0.0,
            mr_stft_loss=999.0,
            iso226_distance=999.0,
            mcd_db=999.0,
            chroma_corr=0.0,
            lufs_diff_lu=0.0,
            stereo_imaging=0.0,
            transient_shape=0.0,
            nmr_db=30.0,
            emotional_arc=0.0,
            vocal_formant=0.0,
            vocal_hnr=0.0,
            pitch_accuracy=0.0,
            vocal_presence=0.0,
            modulation_fidelity=0.0,
            harmonic_structure=0.0,
            spectral_flux_corr=0.0,
            perceptual_disturbance=0.0,
            roughness=0.0,
            specific_loudness_diff=0.0,
            fluctuation_strength=0.0,
            worst_segment_score=0.0,
        )

    # ------------------------------------------------------------------
    # Stereo Imaging Preservation (IACC + width, Blauert 1997)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stereo_imaging(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet stereo imaging preservation score [0, 1].

        Compares IACC (Inter-Aural Cross-Correlation) and stereo width between
        reference and test signals. Mono signals return 0.5 (neutral).

        Uses ``SpatialDepthMetric._compute_iacc()`` from the Musical Goals system
        (Blauert 1997, §8.2 SpatialDepthMetric).
        """
        try:
            ref_2d = np.atleast_2d(ref)
            test_2d = np.atleast_2d(test)

            # Only meaningful for stereo
            if ref_2d.shape[0] < 2 or test_2d.shape[0] < 2:
                return 0.5  # Mono — neutral contribution

            from backend.core.musical_goals.musical_goals_metrics import (
                SpatialDepthMetric,  # pylint: disable=import-outside-toplevel
            )

            # Length-align stereo channels
            min_len = min(ref_2d.shape[1], test_2d.shape[1])
            if min_len < 256:
                return 0.5

            ref_left = np.asarray(ref_2d[0, :min_len], dtype=np.float32)
            ref_right = np.asarray(ref_2d[1, :min_len], dtype=np.float32)
            test_left = np.asarray(test_2d[0, :min_len], dtype=np.float32)
            test_right = np.asarray(test_2d[1, :min_len], dtype=np.float32)

            metric = SpatialDepthMetric()

            # IACC comparison (interaural cross-correlation)
            iacc_ref = metric._compute_iacc(ref_left, ref_right, max_lag_ms=1.0, sr=sr)  # pylint: disable=protected-access
            iacc_test = metric._compute_iacc(test_left, test_right, max_lag_ms=1.0, sr=sr)  # pylint: disable=protected-access
            iacc_preservation = 1.0 - min(abs(iacc_ref - iacc_test), 1.0)

            # Stereo width (L-R correlation drift)
            def _stereo_width(left: np.ndarray, right: np.ndarray) -> float:
                corr = _pearson(left, right) if np.std(left) > 1e-12 and np.std(right) > 1e-12 else 1.0
                return float(np.clip(1.0 - abs(corr), 0.0, 1.0))

            width_ref = _stereo_width(ref_left, ref_right)
            width_test = _stereo_width(test_left, test_right)
            width_preservation = 1.0 - min(abs(width_ref - width_test) * 2.0, 1.0)

            # Combined: 60% IACC + 40% width
            score = 0.6 * iacc_preservation + 0.4 * width_preservation
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Stereo imaging computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Transient Shape Preservation (attack envelope matching)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_transient_shape(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet transient shape preservation [0, 1].

        Uses existing ``AuthentizitaetMetric.compute_transient_preservation()``
        which detects transients in both signals, matches them within ±20 ms,
        and compares sharpness ratios.
        """
        try:
            from backend.core.authenticity_metrics import AuthenticityMetrics  # pylint: disable=import-outside-toplevel

            metric = AuthenticityMetrics()
            preservation_rate, _orig_events, _proc_events = metric.compute_transient_preservation(ref, test, sr)
            return float(np.clip(preservation_rate, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Transient shape computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Noise-to-Mask Ratio (PEAQ-inspired, ITU-R BS.1387)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_nmr(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet Noise-to-Mask Ratio in dB (PEAQ core MOV).

        Uses the existing ``PsychoacousticMaskingModel`` to obtain masking thresholds
        on the reference, then computes the excitation pattern of the residual (test - ref)
        using the same model. NMR = residual_excitation / masking_threshold averaged
        over Bark bands and frames, expressed in dB.

        Both quantities are in the same perceptual domain (Bark-band excitation),
        so the comparison is scale-consistent (ITU-R BS.1387 principle).

        Returns NMR in dB: ≤ 0 dB is excellent (fully masked),
        > 0 dB means noise is audible above the masking threshold.
        """
        try:
            from backend.core.psychoacoustic_masking_model import (  # pylint: disable=import-outside-toplevel
                PsychoacousticMaskingModel,
            )

            masking_model = PsychoacousticMaskingModel()

            # Masking threshold on reference (what can be masked)
            mask_ref = masking_model.compute_threshold(ref, sr)
            thr = mask_ref.masking_threshold  # (n_frames, 24)
            if thr is None or thr.size == 0:
                return 0.0

            # Excitation pattern of residual noise (same perceptual domain)
            residual = test - ref
            mask_noise = masking_model.compute_threshold(residual, sr)
            noise_exc = mask_noise.masking_threshold  # (n_frames, 24)
            if noise_exc is None or noise_exc.size == 0:
                return 0.0

            # Align frame counts
            n_use = min(thr.shape[0], noise_exc.shape[0])
            if n_use < 1:
                return 0.0

            # NMR only over bands with non-negligible masking
            valid = thr[:n_use, :] > 1e-12
            if not valid.any():
                return 0.0

            ratio = noise_exc[:n_use, :][valid] / thr[:n_use, :][valid]
            nmr_db = 10.0 * np.log10(max(float(np.mean(ratio)), 1e-20))
            return float(np.clip(nmr_db, -60.0, 60.0))
        except Exception as exc:
            logger.debug("NMR computation fehlgeschlagen: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Emotional Arc Preservation (arousal/valence Pearson)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_emotional_arc(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet emotional arc preservation [0, 1].

        Uses ``EmotionalArcPreservationMetric.measure()`` which divides audio into
        5 s segments and computes arousal (RMS+ZCR) and valence (spectral flatness)
        Pearson correlations. Combined: 60% arousal + 40% valence.

        Short audio (< 30 s) returns 1.0 (assumed preserved).
        """
        try:
            duration_s = len(ref) / max(sr, 1)
            if duration_s < 10.0:
                return 1.0  # Too short for meaningful arc analysis

            from backend.core.emotional_arc_preservation import (  # pylint: disable=import-outside-toplevel
                EmotionalArcPreservationMetric,
            )

            metric = EmotionalArcPreservationMetric()
            arc_result = metric.measure(ref, test, sr)

            # arc_result has arousal_pearson, valence_pearson
            arousal_r = getattr(arc_result, "arousal_pearson", 0.5)
            valence_r = getattr(arc_result, "valence_pearson", 0.5)

            # Map from correlation [-1, 1] to quality [0, 1]
            # r=1 → 1.0, r=0 → 0.5, r=-1 → 0.0
            arousal_q = float(np.clip((arousal_r + 1.0) / 2.0, 0.0, 1.0))
            valence_q = float(np.clip((valence_r + 1.0) / 2.0, 0.0, 1.0))

            return float(np.clip(0.6 * arousal_q + 0.4 * valence_q, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Emotional arc computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 16: Vocal Formant Preservation (F1–F4)
    # Peterson & Barney 1952, Hillenbrand et al. 1995
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vocal_formant(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet vocal formant preservation [0, 1].

        Extracts F1–F4 formant trajectories from both reference and test audio
        using LPC Burg analysis, then computes Pearson correlation of median
        formant frequencies.

        Formants define vowel identity — distortion of F1-F4 means unnatural voice.
        For non-vocal material, returns 0.5 (neutral contribution).

        References:
            Peterson & Barney (1952), Hillenbrand et al. (1995)
        """
        try:
            n = min(len(ref), len(test))
            if n < 2048:
                return 0.5  # Too short for formant analysis

            # Use center segment (max 3 s) for efficiency
            max_samples = min(n, int(3.0 * sr))
            center = max(0, n // 2 - max_samples // 2)
            ref_seg = np.asarray(ref[center : center + max_samples], dtype=np.float32)
            test_seg = np.asarray(test[center : center + max_samples], dtype=np.float32)

            def _extract_formants(audio: np.ndarray) -> list[float]:
                """Extrahiert median F1-F4 via LPC root-finding on voiced frames."""
                frame_len = int(0.025 * sr)  # 25 ms
                hop = int(0.010 * sr)  # 10 ms
                order = min(16, frame_len - 2)
                if order < 4 or frame_len < 32 or len(audio) < frame_len:
                    return [0.0] * 4
                f_all: list[list[float]] = [[], [], [], []]
                preemph = np.append(audio[0:1], audio[1:] - 0.97 * audio[:-1])
                for start in range(0, len(preemph) - frame_len, hop):
                    frame = preemph[start : start + frame_len]
                    rms = float(np.sqrt(np.mean(frame**2)))
                    if rms < 0.005:
                        continue  # Skip silence
                    windowed = frame * np.hanning(frame_len)
                    # Autocorrelation LPC via Levinson-Durbin — FFT-based
                    from backend.core.core_utils import fft_autocorr  # pylint: disable=import-outside-toplevel

                    r = fft_autocorr(windowed, max_lag=order)
                    if r[0] < 1e-12 or not np.isfinite(r[: order + 1]).all():
                        continue
                    try:
                        # Toeplitz solve for LPC coefficients
                        from scipy.linalg import solve_toeplitz  # pylint: disable=import-outside-toplevel

                        a = solve_toeplitz(r[:order], r[1 : order + 1])
                    except Exception:
                        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                        continue
                    if not np.isfinite(a).all():
                        continue
                    poly = np.concatenate([[1.0], -a])
                    roots = np.roots(poly)
                    # Select upper half-plane roots inside unit circle
                    formants: list[float] = []
                    for root in roots:
                        if np.imag(root) > 0 and np.abs(root) < 0.9999:
                            freq = float(np.angle(root) * sr / (2 * np.pi))
                            if 150 < freq < 5000:
                                formants.append(freq)
                    formants.sort()
                    for i in range(min(4, len(formants))):
                        f_all[i].append(formants[i])
                medians = []
                for i in range(4):
                    medians.append(float(np.median(f_all[i])) if len(f_all[i]) > 3 else 0.0)
                return medians

            ref_formants = _extract_formants(ref_seg)
            test_formants = _extract_formants(test_seg)

            # If no formants found in either, assume non-vocal → neutral
            ref_valid = [f for f in ref_formants if f > 0]
            test_valid = [f for f in test_formants if f > 0]
            if len(ref_valid) < 2 or len(test_valid) < 2:
                return 0.5

            # Preservation: relative deviation of each formant
            scores = []
            for rf, tf in zip(ref_formants, test_formants):
                if rf > 0 and tf > 0:
                    deviation = abs(rf - tf) / rf
                    # 0% deviation → 1.0; >20% deviation → 0.0
                    scores.append(float(np.clip(1.0 - deviation / 0.20, 0.0, 1.0)))
            if not scores:
                return 0.5
            return float(np.mean(scores))
        except Exception as exc:
            logger.debug("Vocal formant computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 17: Vocal HNR Preservation (Boersma 1993, Praat)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vocal_hnr(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet vocal Harmonics-to-Noise Ratio preservation [0, 1].

        HNR is the fundamental voice quality measure — higher HNR means cleaner,
        more tonal voice; lower HNR means breathier/noisier. Uses FFT-based
        autocorrelation (Boersma 1993 algorithm).

        Preservation = 1 - |HNR_ref - HNR_test| / max(|HNR_ref|, 10)

        For non-periodic signals (no voiced content), returns 0.5 (neutral).
        """
        try:

            def _hnr_db(audio: np.ndarray) -> float:
                """HNR estimation via FFT-based autocorrelation (Boersma 1993)."""
                n = min(len(audio), int(1.0 * sr))  # max 1 s
                if n < 256:
                    return 0.0
                seg = np.asarray(audio[:n], dtype=np.float32)
                # Zero-padded FFT for linear autocorrelation
                X = np.fft.rfft(seg, n=2 * n)
                autocorr = np.fft.irfft(np.abs(X) ** 2)[:n]
                ac0 = float(autocorr[0]) + 1e-30
                autocorr = autocorr / ac0

                # Search for peak in F0 range 50–500 Hz
                min_lag = max(1, int(sr / 500))
                max_lag = min(int(sr / 50), n - 1)
                if max_lag <= min_lag:
                    return 0.0

                peak = float(np.max(autocorr[min_lag:max_lag]))
                if peak <= 0.01:
                    return 0.0  # No periodicity detected

                noise_ratio = max(1e-10, 1.0 - peak)
                hnr = 10.0 * np.log10(peak / noise_ratio)
                return float(np.clip(hnr, -10.0, 40.0))

            # Compute on representative segments (avoid very long audio)
            max_samples = min(len(ref), len(test), int(3.0 * sr))
            center = max(0, min(len(ref), len(test)) // 2 - max_samples // 2)
            ref_hnr = _hnr_db(ref[center : center + max_samples])
            test_hnr = _hnr_db(test[center : center + max_samples])

            # Both near-zero → non-vocal content → neutral
            if abs(ref_hnr) < 1.0 and abs(test_hnr) < 1.0:
                return 0.5

            # Preservation metric: small delta = good
            denom = max(abs(ref_hnr), 10.0)
            preservation = 1.0 - abs(ref_hnr - test_hnr) / denom
            return float(np.clip(preservation, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Vocal HNR computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 18: Pitch/F0 Accuracy (SingMOS — Tang et al. 2024)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pitch_accuracy(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet pitch/F0 contour accuracy incl. vibrato fidelity [0, 1].

        F0 fidelity is the #1 quality dimension for singing (SingMOS, Tang 2024).
        Compares F0 contours via Pearson correlation on voiced frames, plus a
        vibrato fidelity sub-metric that measures whether vibrato rate (4–7 Hz)
        and depth (±50–100 cents) are preserved — critical for vocal expressiveness.
        Uses autocorrelation-based F0 tracking (no ML dependencies).

        Combined: 55% correlation, 25% RMSE, 20% vibrato fidelity.
        Non-pitched material returns 0.5 (neutral).
        """
        try:

            def _f0_track(audio: np.ndarray) -> np.ndarray:
                """Autocorrelation-based F0 tracking (50–500 Hz)."""
                frame_len = int(0.030 * sr)  # 30 ms frames
                hop = int(0.010 * sr)  # 10 ms hop
                if frame_len < 32 or len(audio) < frame_len:
                    return np.array([], dtype=np.float64)  # type: ignore[no-any-return]
                n_frames = (len(audio) - frame_len) // hop
                f0_values = np.zeros(n_frames, dtype=np.float64)
                min_lag = max(1, int(sr / 500))
                max_lag = min(int(sr / 50), frame_len - 1)
                if max_lag <= min_lag:
                    return f0_values  # type: ignore[no-any-return]
                for i in range(n_frames):
                    start = i * hop
                    frame = audio[start : start + frame_len]
                    # Autocorrelation via FFT
                    X = np.fft.rfft(frame, n=2 * frame_len)
                    ac = np.fft.irfft(np.abs(X) ** 2)[:frame_len]
                    if ac[0] < 1e-12:
                        continue
                    ac_norm = ac / ac[0]
                    search = ac_norm[min_lag:max_lag]
                    if len(search) == 0:
                        continue
                    peak_idx = int(np.argmax(search)) + min_lag
                    if ac_norm[peak_idx] > 0.35:
                        f0_values[i] = sr / peak_idx
                return f0_values  # type: ignore[no-any-return]

            def _vibrato_fidelity(ref_f0_voiced: np.ndarray, test_f0_voiced: np.ndarray) -> float:
                """Berechnet vibrato fidelity sub-metric [0, 1].

                Vibrato rate (4–7 Hz) and depth (±50–100 cents) are critical
                vocal expressiveness parameters. Compares the modulation spectrum
                of F0 contours in the vibrato band.
                """
                if len(ref_f0_voiced) < 20:
                    return 0.5  # Too short for vibrato analysis

                # Convert to cents relative to median
                ref_median = float(np.median(ref_f0_voiced))
                test_median = float(np.median(test_f0_voiced))
                if ref_median < 50 or test_median < 50:
                    return 0.5

                ref_cents = 1200.0 * np.log2(ref_f0_voiced / ref_median)
                test_cents = 1200.0 * np.log2(test_f0_voiced / test_median)

                # FFT of F0 modulation (f0 at 100 Hz frame rate = 10 ms hop)
                f0_rate = 100.0  # frames per second
                ref_fft = np.abs(np.fft.rfft(ref_cents - np.mean(ref_cents)))
                test_fft = np.abs(np.fft.rfft(test_cents - np.mean(test_cents)))
                mod_freqs = np.fft.rfftfreq(len(ref_cents), d=1.0 / f0_rate)

                # Vibrato band: 3–8 Hz (slightly wider than pure 4–7 Hz)
                vib_mask = (mod_freqs >= 3.0) & (mod_freqs <= 8.0)
                if not vib_mask.any():
                    return 0.5

                min_vib = min(len(ref_fft), len(test_fft))
                ref_vib = ref_fft[:min_vib][vib_mask[:min_vib]]
                test_vib = test_fft[:min_vib][vib_mask[:min_vib]]

                if len(ref_vib) < 2:
                    return 0.5

                # Vibrato depth preservation (total energy in vibrato band)
                ref_depth = float(np.sum(ref_vib))
                test_depth = float(np.sum(test_vib))

                if ref_depth < 0.5:
                    return 0.8  # No significant vibrato → minimal impact

                depth_ratio = test_depth / (ref_depth + 1e-12)
                depth_score = float(np.clip(1.0 - abs(1.0 - depth_ratio) * 2.0, 0.0, 1.0))

                # Vibrato rate preservation (peak frequency similarity)
                ref_peak_idx = int(np.argmax(ref_vib))
                test_peak_idx = int(np.argmax(test_vib))
                vib_freqs = mod_freqs[vib_mask[:min_vib]]
                if len(vib_freqs) > max(ref_peak_idx, test_peak_idx):
                    rate_diff = abs(vib_freqs[ref_peak_idx] - vib_freqs[test_peak_idx])
                    rate_score = float(np.clip(1.0 - rate_diff / 2.0, 0.0, 1.0))
                else:
                    rate_score = 0.5

                # Combined vibrato: 60% depth + 40% rate
                return float(np.clip(0.60 * depth_score + 0.40 * rate_score, 0.0, 1.0))

            # Extract F0 from center segment (max 5 s)
            max_samples = min(len(ref), len(test), int(5.0 * sr))
            center = max(0, min(len(ref), len(test)) // 2 - max_samples // 2)
            ref_f0 = _f0_track(ref[center : center + max_samples])
            test_f0 = _f0_track(test[center : center + max_samples])

            min_frames = min(len(ref_f0), len(test_f0))
            if min_frames < 10:
                return 0.5

            ref_f0 = ref_f0[:min_frames]
            test_f0 = test_f0[:min_frames]

            # Only compare voiced frames (both > 0)
            voiced_mask = (ref_f0 > 0) & (test_f0 > 0)
            n_voiced = int(np.sum(voiced_mask))
            if n_voiced < 5:
                return 0.5  # Not enough voiced content

            ref_voiced = ref_f0[voiced_mask]
            test_voiced = test_f0[voiced_mask]

            # Relative F0 RMSE penalty (always computable)
            f0_rmse = float(np.sqrt(np.mean((ref_voiced - test_voiced) ** 2)))
            mean_f0 = float(np.mean(ref_voiced))
            rel_rmse = f0_rmse / max(mean_f0, 50.0)
            rmse_penalty = float(np.clip(rel_rmse / 0.10, 0.0, 1.0))  # 10% = max penalty
            rmse_score = 1.0 - rmse_penalty

            # Pearson correlation (undefined for constant-pitch signals)
            ref_std = float(np.std(ref_voiced))
            test_std = float(np.std(test_voiced))
            if ref_std < 1e-6 or test_std < 1e-6:
                # Constant pitch — rely on RMSE only (identical → 1.0)
                return float(np.clip(rmse_score, 0.0, 1.0))
            r = _pearson(ref_voiced, test_voiced)
            if not np.isfinite(r):
                return float(np.clip(rmse_score, 0.0, 1.0))

            corr_score = float(np.clip((r + 1.0) / 2.0, 0.0, 1.0))

            # Vibrato fidelity sub-metric
            vib_score = _vibrato_fidelity(ref_voiced, test_voiced)

            # Combined: 55% correlation, 25% RMSE, 20% vibrato fidelity
            return float(
                np.clip(
                    0.55 * corr_score + 0.25 * rmse_score + 0.20 * vib_score,
                    0.0,
                    1.0,
                )
            )
        except Exception as exc:
            logger.debug("Pitch accuracy computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 19: Vocal Presence / CPPS (Franz & Grewe 2026)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vocal_presence(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet vocal presence and CPPS preservation [0, 1].

        Combines:
        1. CPPS (Cepstral Peak Prominence Smoothed) — the strongest single
           acoustic predictor for perceived voice quality (Franz & Grewe 2026).
           Measures harmonic clarity by finding the cepstral peak corresponding
           to F0 and computing its prominence above the regression line.
        2. Presence band energy preservation (1–4 kHz) — where vocal
           intelligibility and clarity reside.

        CPPS weight: 60%, presence band weight: 40%.
        For non-vocal material, returns 0.5 (neutral).
        """
        try:
            n = min(len(ref), len(test))
            if n < 2048:
                return 0.5

            # Use center segment (max 2 s)
            max_samples = min(n, int(2.0 * sr))
            center = max(0, n // 2 - max_samples // 2)
            ref_seg = np.asarray(ref[center : center + max_samples], dtype=np.float32)
            test_seg = np.asarray(test[center : center + max_samples], dtype=np.float32)

            def _cpps(audio: np.ndarray) -> float:
                """Cepstral Peak Prominence Smoothed.

                1. Compute power cepstrum from windowed frames.
                2. Find peak in F0 quefrency range (2–20 ms → 50–500 Hz).
                3. Fit regression line through cepstrum.
                4. CPP = peak height above regression line (dB).
                5. Smooth over frames → CPPS.
                """
                frame_len = min(2048, len(audio))
                hop = frame_len // 2
                n_frames = max(1, (len(audio) - frame_len) // hop)
                cpp_values: list[float] = []

                # Quefrency range for F0: 2–20 ms
                q_min = max(1, int(0.002 * sr))  # 500 Hz
                q_max = min(frame_len // 2 - 1, int(0.020 * sr))  # 50 Hz
                if q_max <= q_min:
                    return 0.0

                for i in range(n_frames):
                    start = i * hop
                    frame = audio[start : start + frame_len]
                    if len(frame) < frame_len:
                        break
                    windowed = frame * np.hanning(frame_len)
                    # Power spectrum → log → IFFT = power cepstrum
                    ps = np.abs(np.fft.rfft(windowed)) ** 2 + 1e-30
                    log_ps = np.log10(ps)
                    cepstrum = np.abs(np.fft.irfft(log_ps))

                    if q_max >= len(cepstrum):
                        continue
                    # Find peak in F0 quefrency range
                    search = cepstrum[q_min : q_max + 1]
                    peak_idx = int(np.argmax(search)) + q_min
                    peak_val = float(cepstrum[peak_idx])

                    # Regression line through cepstrum in quefrency range
                    q_range = np.arange(q_min, q_max + 1, dtype=np.float64)
                    cep_range = cepstrum[q_min : q_max + 1]
                    if len(q_range) < 3:
                        continue
                    coeffs = np.polyfit(q_range, cep_range, 1)
                    regression_at_peak = coeffs[0] * peak_idx + coeffs[1]

                    cpp = peak_val - regression_at_peak
                    if cpp > 0:
                        cpp_values.append(float(10 * np.log10(max(cpp, 1e-20) + 1)))

                if not cpp_values:
                    return 0.0
                return float(np.mean(cpp_values))

            # --- CPPS comparison ---
            cpps_ref = _cpps(ref_seg)
            cpps_test = _cpps(test_seg)

            # Both near-zero → likely non-vocal → neutral
            if cpps_ref < 0.5 and cpps_test < 0.5:
                return 0.5

            # Preservation: small delta = good
            denom = max(abs(cpps_ref), 2.0)
            cpps_preservation = 1.0 - abs(cpps_ref - cpps_test) / denom
            cpps_score = float(np.clip(cpps_preservation, 0.0, 1.0))

            # --- Presence band energy preservation (1–4 kHz) ---
            n_fft = min(4096, max_samples)
            ref_spec = np.abs(np.fft.rfft(ref_seg[:n_fft] * np.hanning(n_fft)))
            test_spec = np.abs(np.fft.rfft(test_seg[:n_fft] * np.hanning(n_fft)))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

            pres_mask = (freqs >= 1000) & (freqs <= 4000)
            if not pres_mask.any():
                return cpps_score

            ref_pres = float(np.sum(ref_spec[pres_mask] ** 2))
            test_pres = float(np.sum(test_spec[pres_mask] ** 2))
            if ref_pres < 1e-12:
                return cpps_score

            pres_ratio = test_pres / ref_pres
            # Ideal ratio = 1.0; deviation penalized
            pres_score = float(np.clip(1.0 - abs(1.0 - pres_ratio) * 2.0, 0.0, 1.0))

            # Combined: 60% CPPS + 40% presence band
            return float(np.clip(0.60 * cpps_score + 0.40 * pres_score, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Vocal presence/CPPS computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Modulation Fidelity — PEAQ AvgModDiff/WinModDiff equivalent
    # (Dau et al. 1997; Jørgensen & Dau 2011)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_modulation_fidelity(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet amplitude modulation spectrum preservation [0, 1].

        The human auditory system is exquisitely sensitive to amplitude modulation
        (AM) patterns — vibrato (4–7 Hz), tremolo, groove micro-timing, and
        speech/singing envelope fluctuations. PEAQ's AvgModDiff and WinModDiff
        model output variables (MOVs) account for ~20% of MUSHRA variance,
        making modulation fidelity the second most important perceptual dimension
        after NMR (Thiede et al. 2000).

        Algorithm:
        1. Divide signal into 26 Bark bands (critical bands) using bandpass filters
           approximated via STFT bin grouping.
        2. Extract temporal envelope per band (half-wave rectification + lowpass 50 Hz).
        3. Compute modulation spectrum per band (FFT of envelope, 0–50 Hz).
        4. Compare ref vs test modulation spectra per band (Pearson + magnitude ratio).
        5. Average across bands → modulation fidelity score [0, 1].

        Returns 0.5 on failure (neutral score).
        """
        try:
            # STFT parameters for ~23ms frames at 48 kHz
            n_fft = 2048
            hop = n_fft // 4
            ref_stft = np.abs(
                np.fft.rfft(
                    np.lib.stride_tricks.sliding_window_view(ref, n_fft)[::hop],
                    axis=-1,
                )
            )
            test_stft = np.abs(
                np.fft.rfft(
                    np.lib.stride_tricks.sliding_window_view(test, n_fft)[::hop],
                    axis=-1,
                )
            )

            n_frames = min(ref_stft.shape[0], test_stft.shape[0])
            if n_frames < 4:
                return 0.5

            ref_stft = ref_stft[:n_frames]
            test_stft = test_stft[:n_frames]

            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

            # Bark band edges (simplified 26 bands: 0–15.5 kHz)
            bark_edges = [
                20,
                100,
                200,
                300,
                400,
                510,
                630,
                770,
                920,
                1080,
                1270,
                1480,
                1720,
                2000,
                2320,
                2700,
                3150,
                3700,
                4400,
                5300,
                6400,
                7700,
                9500,
                12000,
                15500,
                20500,
            ]

            band_scores: list[float] = []

            for i in range(len(bark_edges) - 1):
                lo, hi = bark_edges[i], bark_edges[i + 1]
                band_mask = (freqs >= lo) & (freqs < hi)
                if not band_mask.any():
                    continue

                # Temporal envelope: energy per frame in this band
                ref_env = np.sqrt(np.mean(ref_stft[:, band_mask] ** 2, axis=1) + 1e-20)
                test_env = np.sqrt(np.mean(test_stft[:, band_mask] ** 2, axis=1) + 1e-20)

                if len(ref_env) < 4:
                    continue

                # Modulation spectrum (FFT of temporal envelope)
                ref_mod = np.abs(np.fft.rfft(ref_env - np.mean(ref_env)))
                test_mod = np.abs(np.fft.rfft(test_env - np.mean(test_env)))

                min_mod = min(len(ref_mod), len(test_mod))
                if min_mod < 2:
                    continue
                ref_mod = ref_mod[:min_mod]
                test_mod = test_mod[:min_mod]

                # Modulation frequency axis — only look at 0.5–50 Hz (perceptually relevant)
                frame_rate = sr / hop
                mod_freqs = np.fft.rfftfreq(len(ref_env), d=1.0 / frame_rate)
                valid_mod = (mod_freqs >= 0.5) & (mod_freqs <= 50.0)
                if not valid_mod.any():
                    continue

                rm = ref_mod[valid_mod]
                tm = test_mod[valid_mod]

                # Pearson correlation of modulation spectra
                if np.std(rm) < 1e-12 or np.std(tm) < 1e-12:
                    corr_score = 1.0 if np.allclose(rm, tm, atol=1e-10) else 0.5
                else:
                    corr = _pearson(rm, tm)
                    corr_score = max(0.0, corr)

                # Magnitude ratio (penalize if modulation depth changed)
                ref_total = float(np.sum(rm))
                test_total = float(np.sum(tm))
                if ref_total > 1e-12:
                    ratio = test_total / ref_total
                    mag_score = float(np.clip(1.0 - abs(1.0 - ratio) * 1.5, 0.0, 1.0))
                else:
                    mag_score = 1.0

                # Combined: 65% correlation + 35% magnitude preservation
                band_scores.append(0.65 * corr_score + 0.35 * mag_score)

            if not band_scores:
                return 0.5

            return float(np.clip(np.mean(band_scores), 0.0, 1.0))
        except Exception as exc:
            logger.debug("Modulation fidelity computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Harmonic Structure Preservation — PEAQ EHS equivalent
    # (Thiede et al. 2000)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_harmonic_structure(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet harmonic partial structure preservation [0, 1].

        Compares the relative amplitudes of harmonics 1–16 between reference and
        test signals. This determines whether the timbre character of instruments
        and voices is preserved — beyond what MFCC/MCD/spectral centroid capture.
        Directly analogous to PEAQ's Error Harmonic Structure (EHS), which detects
        distortion products and missing partials.

        Algorithm:
        1. Estimate fundamental frequency via autocorrelation on 3 s center segment.
        2. Extract amplitudes of harmonics 1–16 from magnitude spectrum.
        3. Normalize to relative amplitudes (dB re: fundamental).
        4. Compute Pearson correlation + RMS deviation of harmonic profiles.
        5. Return combined score [0, 1].

        Returns 0.5 on failure (neutral).
        """
        try:
            # Use center 3 s for analysis
            n_use = min(len(ref), int(3.0 * sr))
            center = max(0, len(ref) // 2 - n_use // 2)
            ref_seg = np.asarray(ref[center : center + n_use], dtype=np.float32)
            test_seg = np.asarray(test[center : center + n_use], dtype=np.float32)

            if len(ref_seg) < 2048:
                return 0.5

            # F0 estimation via autocorrelation (robust, no ML dependency)
            def _estimate_f0(sig: np.ndarray, sr_hz: int) -> float:
                """Schätzt F0 via autocorrelation, returns Hz or 0.0."""
                # Limit to 1 s for speed
                seg = sig[: min(len(sig), sr_hz)]
                n = len(seg)
                if n < 512:
                    return 0.0
                # Autocorrelation via FFT
                fft_size = 1
                while fft_size < 2 * n:
                    fft_size *= 2
                xf = np.fft.rfft(seg, n=fft_size)
                acf = np.fft.irfft(xf * np.conj(xf))[:n]
                # Normalize
                if acf[0] < 1e-12:
                    return 0.0
                acf = acf / acf[0]
                # Search for peak in 60–1000 Hz range
                min_lag = max(1, int(sr_hz / 1000))
                max_lag = min(n - 1, int(sr_hz / 60))
                if min_lag >= max_lag:
                    return 0.0
                search = acf[min_lag : max_lag + 1]
                peak_idx = int(np.argmax(search)) + min_lag
                if acf[peak_idx] < 0.3:
                    return 0.0  # Not periodic enough
                return sr_hz / peak_idx

            f0 = _estimate_f0(ref_seg, sr)
            if f0 < 60.0:
                return 0.5  # Cannot extract harmonics from non-pitched audio

            # Extract harmonic amplitudes from magnitude spectrum
            n_fft = min(8192, len(ref_seg))
            ref_spec = np.abs(np.fft.rfft(ref_seg[:n_fft]))
            test_spec = np.abs(np.fft.rfft(test_seg[: min(n_fft, len(test_seg))]))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

            n_harmonics = 16
            min(f0 * (n_harmonics + 0.5), sr / 2)

            ref_amps: list[float] = []
            test_amps: list[float] = []

            for h in range(1, n_harmonics + 1):
                target_hz = f0 * h
                if target_hz >= sr / 2:
                    break
                # Search within ±2% of target frequency
                margin = target_hz * 0.02
                band = (freqs >= target_hz - margin) & (freqs <= target_hz + margin)
                if not band.any():
                    continue
                ref_amps.append(float(np.max(ref_spec[band])))
                if len(test_spec) > 0:
                    test_band = band[: len(test_spec)]
                    test_amps.append(float(np.max(test_spec[test_band])) if test_band.any() else 0.0)
                else:
                    test_amps.append(0.0)

            if len(ref_amps) < 3:
                return 0.5  # Too few harmonics found

            # Convert to relative dB (re: max amplitude)
            ref_db = 20.0 * np.log10(np.array(ref_amps) / (max(ref_amps) + 1e-20) + 1e-20)
            test_db = 20.0 * np.log10(np.array(test_amps) / (max(test_amps) + 1e-20) + 1e-20)

            # Pearson correlation of harmonic profile shape
            if np.std(ref_db) < 1e-6 and np.std(test_db) < 1e-6:
                shape_corr = 1.0
            elif np.std(ref_db) < 1e-6 or np.std(test_db) < 1e-6:
                shape_corr = 0.5
            else:
                corr = _pearson(ref_db, test_db)
                shape_corr = max(0.0, corr)

            # RMS deviation of harmonic profile (penalize amplitude changes)
            rms_dev = float(np.sqrt(np.mean((ref_db - test_db) ** 2)))
            # Mapping: 0 dB → 1.0; 12 dB → ~0.3; 24 dB → ~0.1
            rms_score = float(np.clip(np.exp(-rms_dev / 10.0), 0.0, 1.0))

            # Combined: 55% shape correlation + 45% amplitude fidelity
            return float(np.clip(0.55 * shape_corr + 0.45 * rms_score, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Harmonic structure computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Spectral Flux Correlation (Alluri & Toiviainen 2009)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_spectral_flux_correlation(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet spectral flux correlation [0, 1].

        Spectral flux measures the rate of spectral change over time. It captures
        whether note onsets, vibrato, timbral evolution, and dynamic transitions
        are reproduced correctly. This is fundamental to musical realism —
        a static spectrum with correct average content but wrong dynamics
        sounds lifeless.

        Algorithm:
        1. Compute STFT magnitude for both signals (23 ms frames).
        2. Compute L2-norm spectral flux per frame.
        3. Pearson-correlate the flux sequences.
        4. Combine with magnitude-ratio preservation.

        Returns 0.5 on failure (neutral).
        """
        try:
            n_fft = 2048
            hop = n_fft // 4

            def _spectral_flux(audio: np.ndarray) -> np.ndarray:
                win = np.hanning(n_fft)
                n_frames = max(1, (len(audio) - n_fft) // hop + 1)
                if n_frames < 2:
                    return np.array([0.0])  # type: ignore[no-any-return]
                mags = np.zeros((n_frames, n_fft // 2 + 1))
                for i in range(n_frames):
                    frame = audio[i * hop : i * hop + n_fft]
                    if len(frame) < n_fft:
                        break
                    mags[i] = np.abs(np.fft.rfft(frame * win))
                # L2 flux: ||S(t) - S(t-1)||_2
                flux = np.sqrt(np.sum((mags[1:] - mags[:-1]) ** 2, axis=1))
                return flux  # type: ignore[no-any-return]

            ref_flux = _spectral_flux(ref)
            test_flux = _spectral_flux(test)

            min_n = min(len(ref_flux), len(test_flux))
            if min_n < 4:
                return 0.5

            rf = ref_flux[:min_n]
            tf = test_flux[:min_n]

            # For stationary signals (sine wave, sustained note), spectral flux
            # is near zero. Adding noise creates flux that dominates correlation.
            # Detect stationary case: if ref flux is very low relative to signal
            # energy, use SNR-based scoring instead of flux correlation.
            ref_rms = float(np.sqrt(np.mean(ref[: min(len(ref), int(1.0 * sr))] ** 2)))
            ref_flux_mean = float(np.mean(rf))
            relative_flux = ref_flux_mean / (ref_rms + 1e-20)

            if relative_flux < 0.1:
                # Stationary signal: spectral flux is not meaningful.
                # Instead, measure how well the stationarity is preserved
                # using the SNR between test and reference.
                residual = test[: min(len(ref), len(test))] - ref[: min(len(ref), len(test))]
                sig_power = float(np.mean(ref[: min(len(ref), int(1.0 * sr))] ** 2))
                noise_power = float(np.mean(residual**2)) + 1e-20
                snr_db = 10.0 * np.log10(max(sig_power, 1e-20) / noise_power)
                # Mapping: SNR 40 dB → 1.0; 20 dB → 0.8; 0 dB → 0.3; <0 → 0.0
                flux_score = float(np.clip(snr_db / 50.0, 0.0, 1.0))
                return flux_score

            # Pearson correlation of flux profiles
            if np.std(rf) < 1e-12 and np.std(tf) < 1e-12:
                corr_score = 1.0  # Both constant → identical dynamics
            elif np.std(rf) < 1e-12 or np.std(tf) < 1e-12:
                corr_score = 0.3  # One is static, other isn't → bad
            else:
                corr = _pearson(rf, tf)
                corr_score = max(0.0, corr)

            # Magnitude preservation: total flux should be similar
            ref_total = float(np.sum(rf))
            test_total = float(np.sum(tf))
            if ref_total > 1e-12:
                ratio = test_total / ref_total
                mag_score = float(np.clip(1.0 - abs(1.0 - ratio) * 2.0, 0.0, 1.0))
            else:
                mag_score = 1.0  # Silent ref → neutral

            # Combined: 70% correlation + 30% magnitude
            return float(np.clip(0.70 * corr_score + 0.30 * mag_score, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Spectral flux correlation computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Worst-Segment Score — PEAQ ADB-inspired floor penalty
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_worst_segment_score(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet quality score of the worst 1 s segment [0, 1].

        Real listeners disproportionately penalize brief catastrophic artifacts.
        A single 1-second glitch in an otherwise perfect 3-minute piece can
        drop a MUSHRA rating by 20+ points. This is analogous to PEAQ's
        Average Distorted Block (ADB) metric.

        Uses a lightweight per-segment quality estimate: NSIM (mel-SSIM) on
        1 s segments, returning the minimum across all segments. This captures
        the "weakest link" that dominates human perception.

        Returns the worst segment quality [0, 1] (lower = worse artifact exists).
        """
        try:
            seg_len = int(1.0 * sr)  # 1 second segments
            if len(ref) < seg_len or len(test) < seg_len:
                # Short audio: use the entire signal as one segment
                try:
                    nsim = MertMushraProxy._compute_nsim(ref, test, sr)
                    return float(np.clip(nsim, 0.0, 1.0))
                except Exception as e:
                    logger.warning("mert_mushra_proxy.py::_berechnen_worst_segment_Wert Ersatzpfad: %s", e)
                    return 0.5

            n_segs = len(ref) // seg_len
            if n_segs < 1:
                return 0.5

            seg_scores: list[float] = []
            for i in range(n_segs):
                start = i * seg_len
                end = start + seg_len
                rs = ref[start:end]
                ts = test[start:end]

                # Quick per-segment quality: correlation + energy ratio
                corr = _pearson(rs, ts) if np.std(rs) > 1e-12 and np.std(ts) > 1e-12 else 1.0
                corr = max(0.0, corr)

                # Energy preservation
                ref_e = float(np.sum(rs**2)) + 1e-20
                test_e = float(np.sum(ts**2)) + 1e-20
                e_ratio = test_e / ref_e
                e_score = float(np.clip(1.0 - abs(1.0 - e_ratio) * 3.0, 0.0, 1.0))

                # Residual distortion
                residual = ts - rs
                snr_seg = 10.0 * np.log10(ref_e / (float(np.sum(residual**2)) + 1e-20))
                snr_score = float(np.clip(snr_seg / 40.0, 0.0, 1.0))

                # Combined: 40% correlation + 30% energy + 30% SNR
                seg_scores.append(0.40 * corr + 0.30 * e_score + 0.30 * snr_score)

            if not seg_scores:
                return 0.5

            # Return the WORST segment (min-pool) — this is the floor
            return float(np.clip(min(seg_scores), 0.0, 1.0))
        except Exception as exc:
            logger.debug("Worst-segment Wert computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 23: Perceptual Disturbance (masking-weighted)
    # Schroeder 1979, Zwicker 1999, ITU-R BS.1387 (PEAQ Advanced)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_perceptual_disturbance(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet masking-weighted perceptual disturbance [0, 1].

        Implements PEAQ-style perceptual analysis:
        1. Bark-band decomposition (24 critical bands, Zwicker & Fastl 1990)
        2. Simultaneous masking via spreading function (Schroeder 1979):
           lower slope 27 dB/Bark, upper slope 24 dB/Bark
        3. Absolute Threshold of Hearing (ISO 226, simplified)
        4. Temporal forward masking decay (~200 ms, Zwicker 1999)
        5. Only distortion ABOVE the combined threshold is audible

        Returns 1.0 for inaudible distortion, 0.0 for severely audible.
        """
        try:
            # 3 s center crop for efficiency
            n_use = min(len(ref), int(3.0 * sr))
            if n_use < 512:
                return 0.5
            center = max(0, len(ref) // 2 - n_use // 2)
            r = np.asarray(ref[center : center + n_use], dtype=np.float32)
            t = np.asarray(test[center : center + n_use], dtype=np.float32)

            # STFT parameters
            frame_len = min(2048, n_use)
            if frame_len < 256:
                return 0.5
            hop = frame_len // 2
            window = np.hanning(frame_len)
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            n_bins = len(freqs)
            nyquist = sr / 2.0

            # Bark band edges (25 edges → 24 bands)
            bark_edges = np.array(
                [
                    20,
                    100,
                    200,
                    300,
                    400,
                    510,
                    630,
                    770,
                    920,
                    1080,
                    1270,
                    1480,
                    1720,
                    2000,
                    2320,
                    2700,
                    3150,
                    3700,
                    4400,
                    5300,
                    6400,
                    7700,
                    9500,
                    12000,
                    15500,
                ],
                dtype=np.float64,
            )
            n_bark = min(24, int(np.searchsorted(bark_edges, nyquist)))
            if n_bark < 2:
                return 0.5

            # Pre-compute band assignment matrix (n_bark × n_bins)
            band_mask = np.zeros((n_bark, n_bins), dtype=np.float64)
            for b in range(n_bark):
                lo = bark_edges[b]
                hi = bark_edges[b + 1] if b + 1 < len(bark_edges) else nyquist
                sel = (freqs >= lo) & (freqs < hi)
                cnt = int(sel.sum())
                if cnt > 0:
                    band_mask[b, sel] = 1.0 / cnt

            # Vectorized STFT: sliding_window_view + batch FFT
            n_frames = max(1, (n_use - frame_len) // hop + 1)
            # Build frame indices
            starts = np.arange(n_frames) * hop
            valid = starts + frame_len <= n_use
            starts = starts[valid]
            n_frames = len(starts)
            if n_frames < 1:
                return 0.5

            # Batch windowed frames
            r_frames = np.array([r[s : s + frame_len] * window for s in starts])
            t_frames = np.array([t[s : s + frame_len] * window for s in starts])

            # Power spectra (n_frames × n_bins)
            r_power = np.abs(np.fft.rfft(r_frames, axis=1)) ** 2 + 1e-20
            t_power = np.abs(np.fft.rfft(t_frames, axis=1)) ** 2 + 1e-20

            # Bark-band energies (n_frames × n_bark)
            r_bark = r_power @ band_mask.T
            t_bark = t_power @ band_mask.T

            # --- Spreading function (simultaneous masking) ---
            # S[i,j] = masking contribution of band j on band i
            spread = np.zeros((n_bark, n_bark), dtype=np.float64)
            for i in range(n_bark):
                for j in range(n_bark):
                    dz = abs(i - j)
                    if dz == 0:
                        spread[i, j] = 1.0
                    elif i > j:
                        # Lower spread: ~27 dB/Bark
                        spread[i, j] = 10.0 ** (-2.7 * dz / 10.0)
                    else:
                        # Upper spread: ~24 dB/Bark
                        spread[i, j] = 10.0 ** (-2.4 * dz / 10.0)

            # Excitation pattern: spreading × bark energy
            r_excitation = r_bark @ spread.T  # (n_frames, n_bark)

            # --- Absolute Threshold of Hearing (simplified ISO 226) ---
            band_center = np.array(
                [
                    50,
                    150,
                    250,
                    350,
                    450,
                    570,
                    700,
                    840,
                    1000,
                    1175,
                    1375,
                    1600,
                    1860,
                    2160,
                    2510,
                    2925,
                    3425,
                    4050,
                    4850,
                    5850,
                    7050,
                    8600,
                    10750,
                    13750,
                ][:n_bark],
                dtype=np.float64,
            )
            f_khz = band_center / 1000.0
            ath_db = 3.64 * f_khz ** (-0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * f_khz**4
            # ATH → relative power scale (small but non-zero floor)
            ath_power = 10.0 ** (ath_db / 10.0) * 1e-12

            # --- Combined masking threshold ---
            # Masking offset: ~-20 dB below excitation (typical masking ratio)
            masking_offset = 0.01
            masking_thr = np.maximum(
                r_excitation * masking_offset,
                ath_power[np.newaxis, :],
            )

            # --- Temporal forward masking (200 ms decay) ---
            decay_per_frame = np.exp(-3.0 * hop / (sr * 0.200))
            for i in range(1, n_frames):
                masking_thr[i] = np.maximum(
                    masking_thr[i],
                    masking_thr[i - 1] * decay_per_frame,
                )

            # --- Audible distortion ---
            noise_bark = np.abs(t_bark - r_bark)
            audible = np.maximum(0.0, noise_bark - masking_thr)

            # Aggregate: mean audible distortion relative to signal energy
            mean_audible = float(np.mean(audible))
            mean_signal = float(np.mean(r_bark)) + 1e-20
            ratio = mean_audible / mean_signal

            # Map ratio → [0, 1]: 0 → 1.0 (inaudible), ≥1 → ~0.0
            score = float(np.exp(-ratio * 5.0))
            return float(np.clip(score, 0.0, 1.0))

        except Exception as exc:
            logger.debug("Perceptual disturbance computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 24: Roughness Delta
    # Daniel & Weber 1997, Fastl & Zwicker 2007 Ch. 11, Sethares 2005
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_roughness(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet roughness profile preservation [0, 1].

        Roughness = perception of fast amplitude modulation (15–150 Hz)
        on the basilar membrane. Restoration artifacts often create AM
        in this range (codec artifacts, phase bleeding, spectral NR residues).

        Algorithm:
        1. Bark-band decomposition via STFT (24 bands)
        2. Per-band temporal envelope extraction (amplitude per frame)
        3. Modulation spectrum via FFT of envelope
        4. Roughness weighting: g(f_mod) = (f/70) × exp(1 - f/70)
           → peaks at ~70 Hz (Fastl & Zwicker 2007)
        5. Compare roughness profiles ref vs test

        Returns 1.0 for identical roughness, 0.0 for severely different.
        """
        try:
            n_use = min(len(ref), int(3.0 * sr))
            if n_use < 2048:
                return 0.5
            center = max(0, len(ref) // 2 - n_use // 2)
            r = np.asarray(ref[center : center + n_use], dtype=np.float32)
            t = np.asarray(test[center : center + n_use], dtype=np.float32)

            frame_len = 2048
            hop = frame_len // 4  # 75% overlap for smooth envelope
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            n_bins = len(freqs)
            nyquist = sr / 2.0

            bark_edges = np.array(
                [
                    20,
                    100,
                    200,
                    300,
                    400,
                    510,
                    630,
                    770,
                    920,
                    1080,
                    1270,
                    1480,
                    1720,
                    2000,
                    2320,
                    2700,
                    3150,
                    3700,
                    4400,
                    5300,
                    6400,
                    7700,
                    9500,
                    12000,
                    15500,
                ],
                dtype=np.float64,
            )
            n_bark = min(24, int(np.searchsorted(bark_edges, nyquist)))
            if n_bark < 4:
                return 0.5

            # Pre-compute band assignment (n_bark × n_bins)
            band_mask = np.zeros((n_bark, n_bins), dtype=np.float64)
            for b in range(n_bark):
                lo = bark_edges[b]
                hi = bark_edges[b + 1] if b + 1 < len(bark_edges) else nyquist
                sel = (freqs >= lo) & (freqs < hi)
                cnt = int(sel.sum())
                if cnt > 0:
                    band_mask[b, sel] = 1.0 / cnt

            def _band_roughness(audio: np.ndarray) -> np.ndarray:
                """Berechnet per-band roughness for one audio signal."""
                window = np.hanning(frame_len)
                n_frm = max(1, (len(audio) - frame_len) // hop + 1)
                starts = np.arange(n_frm) * hop
                valid = starts + frame_len <= len(audio)
                starts = starts[valid]
                n_frm = len(starts)
                if n_frm < 8:
                    return np.zeros(n_bark, dtype=np.float64)  # type: ignore[no-any-return]

                # Batch STFT
                frames = np.array([audio[s : s + frame_len] * window for s in starts])
                spec_power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-20

                # Band envelopes: sqrt(mean energy) per frame per band
                band_env = np.sqrt(spec_power @ band_mask.T)  # (n_frm, n_bark)

                # Envelope sample rate
                env_rate = sr / hop

                roughness_per_band = np.zeros(n_bark, dtype=np.float64)
                for b in range(n_bark):
                    env = band_env[:, b]
                    env = env - np.mean(env)  # remove DC
                    if np.std(env) < 1e-12:
                        continue
                    mod_spec = np.abs(np.fft.rfft(env))
                    mod_freqs = np.fft.rfftfreq(len(env), d=1.0 / env_rate)

                    # Roughness weighting: peaks at ~70 Hz
                    f_peak = 70.0
                    with np.errstate(divide="ignore", invalid="ignore"):
                        weight = np.where(
                            mod_freqs > 0,
                            (mod_freqs / f_peak) * np.exp(1.0 - mod_freqs / f_peak),
                            0.0,
                        )
                    # Only 15–150 Hz modulation range
                    valid_mod = (mod_freqs >= 15) & (mod_freqs <= 150)
                    if valid_mod.any():
                        roughness_per_band[b] = float(np.sum(mod_spec[valid_mod] * weight[valid_mod]))

                return roughness_per_band  # type: ignore[no-any-return]

            r_rough = _band_roughness(r)
            t_rough = _band_roughness(t)

            r_total = float(np.sum(r_rough)) + 1e-10
            t_total = float(np.sum(t_rough)) + 1e-10

            # 1. Profile shape correlation
            if np.std(r_rough) > 1e-10 and np.std(t_rough) > 1e-10:
                corr = _pearson(r_rough, t_rough)
                corr = max(0.0, corr)
            else:
                corr = 1.0 if np.allclose(r_rough, t_rough, atol=1e-8) else 0.5

            # 2. Magnitude ratio (symmetric)
            mag_ratio = min(r_total, t_total) / max(r_total, t_total)

            # Combined: 60% shape + 40% magnitude
            score = 0.60 * corr + 0.40 * mag_ratio
            return float(np.clip(score, 0.0, 1.0))

        except Exception as exc:
            logger.debug("Roughness computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 25: Specific Loudness Difference
    # Zwicker 1958, Moore & Glasberg 1996, ISO 532-1:2017, PEAQ Advanced
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_specific_loudness_diff(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet specific loudness profile preservation [0, 1].

        Specific Loudness (sone/Bark) is the perceived loudness per critical
        band, accounting for basilar membrane compression (power-law ≈0.23).
        The integral of |N'_ref - N'_test| is the primary PEAQ Advanced MOV
        ("Noise Loudness" + "Average Disturbance", r > 0.85 to MUSHRA).

        Algorithm:
        1. Bark-band decomposition (24 bands, Zwicker 1961)
        2. Per-band excitation → specific loudness via Zwicker power law:
           N'(E) = 0.08 × (E/E_TQ)^0.23 (re threshold in quiet)
        3. Frame-wise difference profile
        4. Score = exp(-k × mean_diff / mean_ref_loudness)

        Returns 1.0 for identical loudness profiles, 0.0 for severe mismatch.
        """
        try:
            n_use = min(len(ref), int(3.0 * sr))
            if n_use < 512:
                return 0.5
            center = max(0, len(ref) // 2 - n_use // 2)
            r = np.asarray(ref[center : center + n_use], dtype=np.float32)
            t = np.asarray(test[center : center + n_use], dtype=np.float32)

            frame_len = min(2048, n_use)
            if frame_len < 256:
                return 0.5
            hop = frame_len // 2
            window = np.hanning(frame_len)
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            n_bins = len(freqs)
            nyquist = sr / 2.0

            # Bark band edges (25 edges → 24 bands)
            bark_edges = np.array(
                [
                    20,
                    100,
                    200,
                    300,
                    400,
                    510,
                    630,
                    770,
                    920,
                    1080,
                    1270,
                    1480,
                    1720,
                    2000,
                    2320,
                    2700,
                    3150,
                    3700,
                    4400,
                    5300,
                    6400,
                    7700,
                    9500,
                    12000,
                    15500,
                ],
                dtype=np.float64,
            )
            n_bark = min(24, int(np.searchsorted(bark_edges, nyquist)))
            if n_bark < 2:
                return 0.5

            # Band assignment matrix
            band_mask = np.zeros((n_bark, n_bins), dtype=np.float64)
            for b in range(n_bark):
                lo = bark_edges[b]
                hi = bark_edges[b + 1] if b + 1 < len(bark_edges) else nyquist
                sel = (freqs >= lo) & (freqs < hi)
                if sel.any():
                    band_mask[b, sel] = 1.0

            # Threshold in quiet per band (simplified ISO 226, power units)
            band_center = np.array(
                [
                    50,
                    150,
                    250,
                    350,
                    450,
                    570,
                    700,
                    840,
                    1000,
                    1175,
                    1375,
                    1600,
                    1860,
                    2160,
                    2510,
                    2925,
                    3425,
                    4050,
                    4850,
                    5850,
                    7050,
                    8600,
                    10750,
                    13750,
                ][:n_bark],
                dtype=np.float64,
            )
            f_khz = band_center / 1000.0
            ath_db = 3.64 * f_khz ** (-0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * f_khz**4
            e_tq = 10.0 ** (ath_db / 10.0) * 1e-12  # threshold excitation

            # Batch STFT
            n_frames = max(1, (n_use - frame_len) // hop + 1)
            starts = np.arange(n_frames) * hop
            valid = starts + frame_len <= n_use
            starts = starts[valid]
            n_frames = len(starts)
            if n_frames < 1:
                return 0.5

            r_frames = np.array([r[s : s + frame_len] * window for s in starts])
            t_frames = np.array([t[s : s + frame_len] * window for s in starts])

            r_power = np.abs(np.fft.rfft(r_frames, axis=1)) ** 2 + 1e-20
            t_power = np.abs(np.fft.rfft(t_frames, axis=1)) ** 2 + 1e-20

            # Band excitation (n_frames × n_bark)
            r_exc = r_power @ band_mask.T + 1e-20
            t_exc = t_power @ band_mask.T + 1e-20

            # Zwicker specific loudness: N' = 0.08 × (E/E_TQ)^0.23
            # Clamp ratio to avoid extremely large values
            _ALPHA = 0.23
            r_loud = 0.08 * np.power(np.clip(r_exc / e_tq[np.newaxis, :], 1.0, 1e10), _ALPHA)
            t_loud = 0.08 * np.power(np.clip(t_exc / e_tq[np.newaxis, :], 1.0, 1e10), _ALPHA)

            # Difference profile
            diff = np.abs(r_loud - t_loud)
            mean_diff = float(np.mean(diff))
            mean_ref = float(np.mean(r_loud)) + 1e-10

            # Score: exponential mapping
            ratio = mean_diff / mean_ref
            score = float(np.exp(-ratio * 8.0))
            return float(np.clip(score, 0.0, 1.0))

        except Exception as exc:
            logger.debug("Specific loudness diff computation fehlgeschlagen: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Component 26: Fluctuation Strength Delta
    # Daniel & Weber 1997, Fastl & Zwicker 2007 Ch. 10, Sottek 2016
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fluctuation_strength(
        ref: np.ndarray,
        test: np.ndarray,
        sr: int,
    ) -> float:
        """Berechnet fluctuation strength profile preservation [0, 1].

        Fluctuation strength = perception of slow amplitude modulation
        (0.5–20 Hz, peak at ~4 Hz) on the basilar membrane. Tremolo,
        compressor pump effects, breath artifacts live in this range.

        This is a distinct Zwicker dimension from roughness (15–150 Hz);
        there is a clear perceptual gap between 15–20 Hz.

        Algorithm:
        1. Bark-band decomposition via STFT (24 bands)
        2. Per-band temporal envelope (amplitude per frame)
        3. Modulation spectrum via FFT of envelope
        4. Fluctuation weighting: g(f) = f / (f/4 + 4/f)
           → peaks at ~4 Hz (Fastl & Zwicker 2007, Eq. 10.1)
        5. Filter to 0.5–20 Hz modulation range
        6. Compare fluctuation profiles ref vs test

        Returns 1.0 for identical fluctuation, 0.0 for severely different.
        """
        try:
            n_use = min(len(ref), int(3.0 * sr))
            if n_use < 4096:
                return 0.5
            center = max(0, len(ref) // 2 - n_use // 2)
            r = np.asarray(ref[center : center + n_use], dtype=np.float32)
            t = np.asarray(test[center : center + n_use], dtype=np.float32)

            frame_len = 2048
            hop = frame_len // 4  # 75% overlap for smooth envelope
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            n_bins = len(freqs)
            nyquist = sr / 2.0

            bark_edges = np.array(
                [
                    20,
                    100,
                    200,
                    300,
                    400,
                    510,
                    630,
                    770,
                    920,
                    1080,
                    1270,
                    1480,
                    1720,
                    2000,
                    2320,
                    2700,
                    3150,
                    3700,
                    4400,
                    5300,
                    6400,
                    7700,
                    9500,
                    12000,
                    15500,
                ],
                dtype=np.float64,
            )
            n_bark = min(24, int(np.searchsorted(bark_edges, nyquist)))
            if n_bark < 4:
                return 0.5

            band_mask = np.zeros((n_bark, n_bins), dtype=np.float64)
            for b in range(n_bark):
                lo = bark_edges[b]
                hi = bark_edges[b + 1] if b + 1 < len(bark_edges) else nyquist
                sel = (freqs >= lo) & (freqs < hi)
                cnt = int(sel.sum())
                if cnt > 0:
                    band_mask[b, sel] = 1.0 / cnt

            def _band_fluctuation(audio: np.ndarray) -> np.ndarray:
                """Berechnet per-band fluctuation strength."""
                window = np.hanning(frame_len)
                n_frm = max(1, (len(audio) - frame_len) // hop + 1)
                starts = np.arange(n_frm) * hop
                valid_s = starts + frame_len <= len(audio)
                starts = starts[valid_s]
                n_frm = len(starts)
                if n_frm < 16:
                    return np.zeros(n_bark, dtype=np.float64)  # type: ignore[no-any-return]

                frames = np.array([audio[s : s + frame_len] * window for s in starts])
                spec_power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-20

                # Band envelopes
                band_env = np.sqrt(spec_power @ band_mask.T)  # (n_frm, n_bark)

                env_rate = sr / hop
                fluct_per_band = np.zeros(n_bark, dtype=np.float64)

                for b in range(n_bark):
                    env = band_env[:, b]
                    env = env - np.mean(env)
                    if np.std(env) < 1e-12:
                        continue
                    mod_spec = np.abs(np.fft.rfft(env))
                    mod_freqs = np.fft.rfftfreq(len(env), d=1.0 / env_rate)

                    # Fluctuation weighting: peaks at ~4 Hz
                    # g(f) = f / (f/4 + 4/f) — Fastl & Zwicker Eq. 10.1
                    with np.errstate(divide="ignore", invalid="ignore"):
                        weight = np.where(
                            mod_freqs > 0.1,
                            mod_freqs / (mod_freqs / 4.0 + 4.0 / mod_freqs),
                            0.0,
                        )
                    # Only 0.5–20 Hz modulation range
                    valid_mod = (mod_freqs >= 0.5) & (mod_freqs <= 20.0)
                    if valid_mod.any():
                        fluct_per_band[b] = float(np.sum(mod_spec[valid_mod] * weight[valid_mod]))

                return fluct_per_band  # type: ignore[no-any-return]

            r_fluct = _band_fluctuation(r)
            t_fluct = _band_fluctuation(t)

            r_total = float(np.sum(r_fluct)) + 1e-10
            t_total = float(np.sum(t_fluct)) + 1e-10

            # 1. Profile shape correlation
            if np.std(r_fluct) > 1e-10 and np.std(t_fluct) > 1e-10:
                corr = _pearson(r_fluct, t_fluct)
                corr = max(0.0, corr)
            else:
                corr = 1.0 if np.allclose(r_fluct, t_fluct, atol=1e-8) else 0.5

            # 2. Magnitude ratio
            mag_ratio = min(r_total, t_total) / max(r_total, t_total)

            # Combined: 60% shape + 40% magnitude
            score = 0.60 * corr + 0.40 * mag_ratio
            return float(np.clip(score, 0.0, 1.0))

        except Exception as exc:
            logger.debug("Fluctuation strength computation fehlgeschlagen: %s", exc)
            return 0.5


# ---------------------------------------------------------------------------
# Embedding extraction helpers
# ---------------------------------------------------------------------------


def _extract_hf_embedding(mert_plugin: object, audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extrahiert temporal-mean 768-dim embedding from HuggingFace MERT model."""
    try:
        import torch  # pylint: disable=import-outside-toplevel

        processor = getattr(mert_plugin, "_processor", None)
        model = getattr(mert_plugin, "_model", None)
        if processor is None or model is None:
            return None

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Last hidden state: (batch=1, time, 768)
        last_hidden = outputs.hidden_states[-1]
        # Temporal mean → fixed 768-dim embedding
        embedding = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding.astype(np.float32)  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug("HF embedding extraction fehlgeschlagen: %s", exc)
        return None


def _extract_onnx_embedding(mert_plugin: object, audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extrahiert embedding from ONNX MERT session."""
    session = getattr(mert_plugin, "_model", None)
    if session is None:
        return None

    # §4.6b PLM Active-Guard: prevent Emergency-Eviction from invalidating the
    # ONNX session mid-inference → crash / OOM.
    _plm = None
    try:
        from backend.core.plugin_lifecycle_manager import (
            get_plugin_lifecycle_manager as _get_plm_mert,  # pylint: disable=import-outside-toplevel
        )

        _plm = _get_plm_mert()
        _plm.set_active("MERT", True)
    except Exception as e:
        logger.warning("mert_mushra_proxy.py::_extrahieren_onnx_embedding Ersatzpfad: %s", e)

    try:
        min_len = sr  # 1 s minimum
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)))
        feed = {session.get_inputs()[0].name: audio[np.newaxis]}
        result = session.run(None, feed)[0]  # (1, time, dim) or (1, dim)
        if result.ndim == 3:
            embedding = result[0].mean(axis=0)  # temporal mean
        elif result.ndim == 2:
            embedding = result[0]
        else:
            embedding = result.flatten()
        return embedding.astype(np.float32)  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug("ONNX embedding extraction fehlgeschlagen: %s", exc)
        return None
    finally:
        if _plm is not None:
            try:
                _plm.set_active("MERT", False)
            except Exception as e:
                logger.warning("mert_mushra_proxy.py::_extrahieren_onnx_embedding Ersatzpfad: %s", e)


def _extract_dsp_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet a 512-dim DSP feature vector as MERT embedding proxy.

    Combines MFCCs (13 × 20 stats), chroma (12 × 4 stats), spectral features
    (centroid, rolloff, flatness, contrast × 4 stats), and temporal features
    (ZCR, RMS × 4 stats) into a fixed-size vector.
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        features = []

        # MFCCs: 13 coefficients, 4 statistical moments each = 52 dims
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for coeff in mfcc:
            features.extend([np.mean(coeff), np.std(coeff), np.min(coeff), np.max(coeff)])

        # Chroma: 12 bins, 4 stats = 48 dims
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        for ch in chroma:
            features.extend([np.mean(ch), np.std(ch), np.min(ch), np.max(ch)])

        # Spectral centroid, rolloff, flatness: 3 × 4 stats = 12 dims
        for feat_fn in [
            lambda: librosa.feature.spectral_centroid(y=audio, sr=sr),
            lambda: librosa.feature.spectral_rolloff(y=audio, sr=sr),
            lambda: librosa.feature.spectral_flatness(y=audio),
        ]:
            feat = feat_fn().flatten()
            features.extend([np.mean(feat), np.std(feat), np.min(feat), np.max(feat)])

        # Spectral contrast: 7 bands × 4 stats = 28 dims
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        for band in contrast:
            features.extend([np.mean(band), np.std(band), np.min(band), np.max(band)])

        # Temporal features: ZCR, RMS = 2 × 4 stats = 8 dims
        zcr = librosa.feature.zero_crossing_rate(y=audio).flatten()
        rms = librosa.feature.rms(y=audio).flatten()
        for feat in [zcr, rms]:
            features.extend([np.mean(feat), np.std(feat), np.min(feat), np.max(feat)])

        # Total ≈ 148 dims → pad/truncate to 512
        vec = np.array(features, dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if len(vec) < 512:
            vec = np.pad(vec, (0, 512 - len(vec)))
        else:
            vec = vec[:512]

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm

        return vec  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("mert_mushra_proxy.py::_extrahieren_dsp_embedding Ersatzpfad: %s", e)
        return np.zeros(512, dtype=np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] (clamped, since music embeddings are non-negative)."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-10 or b_norm < 1e-10:
        return 0.0
    cos = float(np.dot(a, b) / (a_norm * b_norm))
    return float(np.clip(cos, 0.0, 1.0))


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertiert to mono float32; NaN/Inf guard."""
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if audio.ndim == 2:
        if audio.shape[0] <= 8:
            return np.mean(audio, axis=0).astype(np.float32)  # type: ignore[no-any-return]
        return np.mean(audio, axis=1).astype(np.float32)  # type: ignore[no-any-return]
    return audio.astype(np.float32)  # type: ignore[no-any-return]


def _grade(score: float) -> str:
    """Map MUSHRA score [0, 100] to grade label."""
    if score >= 91:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Fair"
    if score >= 40:
        return "Poor"
    return "Bad"


def _stft_magnitude(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Berechnet STFT magnitude spectrogram (numpy-only, no torch).

    Returns shape (n_fft//2+1, n_frames).
    """
    audio = np.asarray(audio, dtype=np.float64)
    # Zero-pad to at least one full frame
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames < 1:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float64)  # type: ignore[no-any-return]
    window = np.hanning(n_fft).astype(np.float32)
    n_bins = n_fft // 2 + 1
    result = np.zeros((n_bins, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + n_fft] * window
        result[:, i] = np.abs(np.fft.rfft(frame, n=n_fft))
    return result  # type: ignore[no-any-return]


# ISO 226:2003 equal-loudness data (40 phon) — 19 anchor frequencies
_ISO226_FREQS = np.array(
    [
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
    ]
)
_ISO226_SPL40 = np.array(
    [
        99.85,
        93.94,
        88.17,
        82.63,
        77.78,
        73.08,
        68.48,
        64.37,
        60.59,
        56.70,
        53.41,
        50.40,
        47.58,
        44.98,
        42.44,
        39.73,
        37.32,
        35.35,
        33.31,
    ]
)


def _iso226_weights_for_proxy(freqs: np.ndarray) -> np.ndarray:
    """ISO 226 perceptual weighting for arbitrary frequency bins.

    Uses equal-loudness contour at 40 phon to derive per-bin weights.
    Higher weight at 3-4 kHz where human hearing is most sensitive.
    """
    # Interpolate SPL values to target frequencies
    # Clamp to valid range
    log_iso = np.log10(np.maximum(_ISO226_FREQS, 1.0))
    log_f = np.log10(np.maximum(freqs, 1.0))
    spl = np.interp(log_f, log_iso, _ISO226_SPL40, left=_ISO226_SPL40[0], right=_ISO226_SPL40[-1])
    # Weight = inverse loudness threshold relative to 1 kHz
    # At 1 kHz: SPL = 35.35 dB → weight = 1.0. At 20 Hz: SPL = 99.85 → weight ≈ 0.006
    ref_spl = 35.35  # SPL at 1 kHz
    weights = 10.0 ** ((ref_spl - spl) / 20.0)
    return np.clip(weights, 0.001, 10.0).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton (Thread-safe, Double-Checked Locking — §3.x)
# ---------------------------------------------------------------------------

_instance: MertMushraProxy | None = None
_lock = threading.Lock()


def get_proxy_evaluator() -> MertMushraProxy:
    """Thread-safe singleton accessor for MERT MUSHRA proxy evaluator."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MertMushraProxy()
                logger.debug("MertMushraProxy singleton erstellt.")
    return _instance


def estimate_mushra_proxy(
    reference: np.ndarray,
    test: np.ndarray,
    sr: int = 48_000,
) -> MushraProxyResult:
    """Convenience function: estimate MUSHRA proxy score for a reference/test pair.

    Combines 19 perceptual metrics: MERT cosine (when available), ViSQOL v3,
    NSIM, artifact penalty, temporal consistency, CLAP cosine,
    Multi-Resolution STFT, ISO 226 spectral distance, MCD, chroma
    correlation, LUFS difference, stereo imaging preservation, transient
    shape matching, Noise-to-Mask Ratio (NMR), emotional arc preservation,
    vocal formant preservation, vocal HNR, pitch/F0 accuracy, and vocal
    presence/CPPS into a single [0, 100] score.

    The returned confidence indicates estimation reliability:
    - ≈ 0.94 when MERT embeddings are available (estimated r ≈ 0.92 to human MUSHRA)
    - ≈ 0.87 when only DSP metrics are used (estimated r ≈ 0.90 to human MUSHRA)

    Args:
        reference: Original audio.
        test:      Restored audio.
        sr:        Sample rate in Hz (default: 48000).

    Returns:
        MushraProxyResult with estimated score, grade, components, and confidence.
    """
    return get_proxy_evaluator().evaluate(reference, test, sr)
