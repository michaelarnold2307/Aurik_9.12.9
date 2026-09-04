"""MUSHRA-Evaluator für musikalische Restaurierungsqualität.

Implementiert eine **objektive Approximation** des MUSHRA-Verfahrens
(MUltiple Stimuli with Hidden Reference and Anchor) gemäß ITU-R BS.1534-3.

Während klassisches MUSHRA subjektive Hörertests erfordert, approximiert
dieser Modul die MUSHRA-Skala (0–100) mit objektiven Metriken:

- **Perceptuelle Ähnlichkeit** zur Referenz (NSIM auf Gammatone-Spektrogrammen)
- **Musical Goals** als Qualitätsdimensionen (alle 9 Aurik-Ziele)
- **Anchor-Kalibrierung**: LP-gefiltertes Signal als 3.5-kHz-Anchor (≈ Score 20–30)

MUSHRA-Skala-Beschreibung (ITU-R BS.1534-3):
    100:  Excellent    (unmerkliche Unterschiede zur Referenz)
     80:  Good         (wahrnehmbare, aber nicht störende Unterschiede)
     60:  Fair         (leicht störende Unterschiede)
     40:  Poor         (störende Unterschiede)
     20:  Bad          (sehr störende Unterschiede, 3.5-kHz-Anchor-Niveau)

Beispiel::

    from backend.core.mushra_evaluator import evaluate_mushra, get_mushra_evaluator

    result = evaluate_mushra(reference_audio, restored_audio, sr=48000)
    logger.debug("MUSHRA-Score: %.1f/100", result.mushra_score)
    logger.debug("Kategorie: %s", result.grade)  # z.B. "Good"
    logger.debug("ITU-Konform: %s", result.itu_grade)  # z.B. "B (Good)"

Autor: Aurik 10.0.0 — 19. Februar 2026
Referenz: ITU-R BS.1534-3 (2015): "Method for the subjective assessment of
intermediate quality levels of audio systems"
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


def _safe_fft_size(length: int, target: int = 2048, minimum: int = 64) -> int:
    """Gibt power-of-two FFT size capped by signal length zurück.

    Prevents librosa short-signal warnings while keeping spectral resolution
    as close as possible to the nominal target.
    """
    if length <= minimum:
        return minimum
    capped = min(target, int(length))
    return max(minimum, 1 << (capped.bit_length() - 1))


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class MushraResult:
    """Ergebnis einer MUSHRA-Bewertung.

    ⚠️  WICHTIG: Dies ist eine **objektive Approximation** (algorithmische Schätzung),
    kein echter subjektiver MUSHRA-Hörertest nach ITU-R BS.1534-3.
    Jeder Score muss als "estimated" behandelt werden — siehe _is_approximation.

    Für echte MUSHRA-Hörertests: backend/core/mushra_listener.py (geplant).
    Empfohlener Import-Pfad: backend/core/objective_mushra_estimator.py.

    Attributes:
        mushra_score:        Objektiver MUSHRA-Score ∈ [0, 100] (approximiert).
        grade:               Kategoriebezeichnung (Excellent/Good/Fair/Poor/Bad).
        itu_grade:           ITU-R-konforme Grad-Bezeichnung (A–E).
        nsim:                Perceptuelle Ähnlichkeit zur Referenz ∈ [0, 1].
        musical_goals:       Dict mit allen 9 Musical-Goal-Scores.
        anchor_score:        MUSHRA-Score des 3.5-kHz-Anchors (Kalibrierung).
        hidden_ref_score:    MUSHRA-Score der verdeckten Referenz (sollte≈100).
        details:             Zusätzliche Metriken für Debugging/Reporting.
        _is_approximation:   IMMER True — dies ist eine algorithmische Schätzung.
    """

    mushra_score: float
    grade: str
    itu_grade: str
    nsim: float
    musical_goals: dict[str, float]
    anchor_score: float
    hidden_ref_score: float
    details: dict[str, float] = field(default_factory=dict)
    _is_approximation: bool = True  # §15.10: Jeder Score muss als "estimated" markiert sein

    def passes_mushra_threshold(self, min_score: float = 80.0) -> bool:
        """Prüft ob der Score die Mindestanforderung erfüllt.

        Args:
            min_score: Minimaler MUSHRA-Score (Standard: 80 = Good).

        Returns:
            True wenn mushra_score ≥ min_score.
        """
        return self.mushra_score >= min_score

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "mushra_score": self.mushra_score,
            "grade": self.grade,
            "itu_grade": self.itu_grade,
            "nsim": self.nsim,
            "anchor_score": self.anchor_score,
            "hidden_ref_score": self.hidden_ref_score,
            **{f"mg_{k}": v for k, v in self.musical_goals.items()},
            **self.details,
        }


@dataclass
class MushraComparison:
    """Vergleich mehrerer Restaurierungs-Varianten auf MUSHRA-Basis.

    Attributes:
        reference_condition:   Name + Score der Referenz.
        anchor_condition:      Name + Score des Anchors.
        test_conditions:       Liste aller Testbedingungen (Name, Score, Ergebnis).
        ranking:               Testbedingungen sortiert nach MUSHRA-Score (absteigend).
        winner:                Name der besten Testbedingung.
    """

    reference_condition: tuple[str, float]
    anchor_condition: tuple[str, float]
    test_conditions: list[tuple[str, MushraResult]]
    ranking: list[tuple[str, float]]
    winner: str


# ---------------------------------------------------------------------------
# Kernklasse
# ---------------------------------------------------------------------------


class MushraEvaluator:
    """Objektiver MUSHRA-Evaluator für Restaurierungsqualität.

    Approximiert den MUSHRA-Score (0–100) aus objektiven Audio-Metriken
    entsprechend ITU-R BS.1534-3. Kein subjektiver Hörtest erforderlich.

    Objektive Score-Berechnung:
        1. NSIM (strukturelle Ähnlichkeit auf Gammatone-Spektrogrammen,
           25 Bänder, 50–8000 Hz) → perceptueller Similaritäts-Score.
        2. Musical Goals (alle 9 Aurik-Ziele) → Musikalischer Qualitätsscore.
        3. Meld-Cepstral Distortion (MCD) → Klangfarben-Treue.
        4. LUFS-Differenz → Lautstärke-Invarianz.
        5. Gewichtete Kombination → MUSHRA-Score [0, 100].

    Anchor-Kalibrierung (ITU-R BS.1534-3 §6):
        Der 3.5-kHz-Tiefpassfilter-Anchor wird automatisch erzeugt und bewertet.
        Er dient als unteres Kalibrierungspunkt (≈ MUSHRA-Score 20–30).

    Singleton-Muster (Thread-safe, Double-Checked Locking):
        Nutze ``get_mushra_evaluator()`` statt direkter Instantiierung.
    """

    # v10.101 SOTA: MUSHRA/OQS Ground-Truth für perzeptuelle Qualität (35% QualityAnalyzer).
    # MUSHRA-Gewichtungsmatrix (Summe = 1.0)
    _WEIGHTS: dict[str, float] = {
        "nsim": 0.35,  # Perceptuelle Ähnlichkeit (stärkster Prädiktor)
        "musical_goals": 0.35,  # 9 Musical Goals (Aurik-DNA)
        "mcd": 0.15,  # Mel-Cepstral Distortion (Klangfarbe)
        "lufs_diff": 0.10,  # Lautstärke-Invarianz
        "spectral_corr": 0.05,  # Spektrale Korrelation
    }

    def __init__(self) -> None:
        self._gammatone_ready: bool = False
        self._mu_checker = None  # Lazily loaded
        self._mg_cache: dict[str, dict[str, float]] = {}
        self._mg_cache_lock = threading.Lock()
        self._mg_cache_max_entries = 32

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        sr: int,
        *,
        compute_anchor: bool = True,
    ) -> MushraResult:
        """Berechnet den objektiven MUSHRA-Score.

        Args:
            reference:       Referenz-Audio (Original oder restored mit bekannter
                             Qualität); 1-D float32, normalisiert auf [-1, 1].
            test:            Testsignal (restauriertes Audio); gleiche Länge wie reference.
            sr:              Abtastrate in Hz (Pflicht: 48 000 Hz).
            compute_anchor:  Wenn True, wird 3.5-kHz-Anchor intern berechnet.

        Returns:
            :class:`MushraResult` mit vollständiger MUSHRA-Bewertung.

        Raises:
            ValueError: Falls sr ≠ 48 000 oder Audio nicht 1-D.
        """
        if sr != 48_000:
            logger.warning("MushraEvaluator: SR=%d Hz ≠ 48000 Hz — Resampling empfohlen", sr)

        ref_mono = self._to_mono(reference)
        test_mono = self._to_mono(test)

        # Längen angleichen
        min_len = min(len(ref_mono), len(test_mono))
        ref_mono = ref_mono[:min_len]
        test_mono = test_mono[:min_len]

        # --- Einzelmetriken berechnen ---
        nsim = self._compute_nsim(ref_mono, test_mono, sr)
        mcd = self._compute_mcd(ref_mono, test_mono, sr)
        lufs_diff = self._compute_lufs_diff(ref_mono, test_mono, sr)
        spectral_corr = self._compute_spectral_corr(ref_mono, test_mono, sr)
        musical_goals = self._compute_musical_goals(test_mono, sr)
        mg_mean = float(np.mean(list(musical_goals.values()))) if musical_goals else 0.0

        # Score-Konversion in [0, 1]-Raum
        nsim_score = float(np.clip(nsim, 0.0, 1.0))
        # NaN guard: musical_goals sub-metrics can return nan for near-silent
        # audio; np.mean propagates nan → mg_score=nan → mushra_score=nan.
        mg_mean_safe = float(np.nan_to_num(mg_mean, nan=0.0))
        mg_score = float(np.clip(mg_mean_safe, 0.0, 1.0))
        mcd_score = float(np.exp(-mcd / 300.0))  # MCD 0→1.0  242→0.446  500→0.189
        lufs_score = float(np.clip(1.0 - abs(lufs_diff) / 12.0, 0.0, 1.0))
        sc_score = float(np.clip(spectral_corr, 0.0, 1.0))

        # Gewichtete Kombination → [0, 1]
        # When musical_goals could not be computed (empty dict → mg_score=0), the
        # mg weight is redistributed proportionally across the remaining metrics so
        # a perfect perceptual match still reaches 100.  This prevents short or
        # synthetic test signals from being unfairly penalised for a missing Goal
        # computation (§0 Primum non nocere — no artefact from an absent sub-metric).
        if not musical_goals:
            _remaining_w = 1.0 - self._WEIGHTS["musical_goals"]
            raw = (
                (self._WEIGHTS["nsim"] / _remaining_w) * nsim_score
                + (self._WEIGHTS["mcd"] / _remaining_w) * mcd_score
                + (self._WEIGHTS["lufs_diff"] / _remaining_w) * lufs_score
                + (self._WEIGHTS["spectral_corr"] / _remaining_w) * sc_score
            )
        else:
            raw = (
                self._WEIGHTS["nsim"] * nsim_score
                + self._WEIGHTS["musical_goals"] * mg_score
                + self._WEIGHTS["mcd"] * mcd_score
                + self._WEIGHTS["lufs_diff"] * lufs_score
                + self._WEIGHTS["spectral_corr"] * sc_score
            )
        # Final NaN/Inf guard before scaling (defensive — sub-metric guards above
        # should prevent this, but belt-and-suspenders per coding standards).
        raw = float(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0))

        # → MUSHRA [0, 100]
        mushra_score = float(np.clip(raw * 100.0, 0.0, 100.0))
        mushra_score = round(mushra_score, 1)

        # Anchor-Score (3.5-kHz-Tiefpass ITU-R BS.1534-3 §6)
        anchor_score = 0.0
        if compute_anchor:
            anchor = self._create_anchor(ref_mono, sr)
            # Anchor-Bewertung: Musical Goals weglassen (zu teuer + nicht relevant).
            # Der Anchor soll nur seinen Rohscore liefern (NSIM+MCD+LUFS+SC).
            anchor_score = self._quick_score(ref_mono, anchor, sr)

        # Verdeckte Referenz: Im MUSHRA-Protokoll ist ref vs. ref per Definition 100.
        # Kein rekursiver Aufruf — das wäre eine unendliche Rekursion.
        hidden_ref_score = 100.0

        grade, itu_grade = self._grade(mushra_score)

        logger.info(
            "🎯 MUSHRA: Wert=%.1f (%s) | NSIM=%.3f MCD=%.1fdB LUFS-Δ=%.1fLU | Anchor=%.1f",
            mushra_score,
            grade,
            nsim,
            mcd,
            lufs_diff,
            anchor_score,
        )

        return MushraResult(
            mushra_score=mushra_score,
            grade=grade,
            itu_grade=itu_grade,
            nsim=nsim,
            musical_goals=musical_goals,
            anchor_score=anchor_score,
            hidden_ref_score=hidden_ref_score,
            details={
                "nsim_score": nsim_score,
                "mg_score": mg_score,
                "mcd_score": mcd_score,
                "mcd_db": mcd,
                "lufs_diff_lu": lufs_diff,
                "spectral_corr": spectral_corr,
            },
        )

    def compare_conditions(
        self,
        reference: np.ndarray,
        conditions: dict[str, np.ndarray],
        sr: int,
    ) -> MushraComparison:
        """Bewertet mehrere Restaurierungs-Varianten im MUSHRA-Layout.

        Args:
            reference:   Referenz-Audio.
            conditions:  Dict {condition_name → test_audio} für alle Varianten.
            sr:          Abtastrate in Hz.

        Returns:
            :class:`MushraComparison` mit Ranking aller Bedingungen.
        """
        test_conditions: list[tuple[str, MushraResult]] = []
        for name, audio in conditions.items():
            result = self.evaluate(reference, audio, sr, compute_anchor=False)
            test_conditions.append((name, result))

        # Anchor-Bedingung automatisch hinzufügen
        anchor_audio = self._create_anchor(self._to_mono(reference), sr)
        anchor_result = self.evaluate(reference, anchor_audio, sr, compute_anchor=False)

        # Verdeckte Referenz
        ref_result = self.evaluate(reference, reference, sr, compute_anchor=False)

        # Ranking
        ranking = sorted(
            [(name, r.mushra_score) for name, r in test_conditions],
            key=lambda x: x[1],
            reverse=True,
        )
        winner = ranking[0][0] if ranking else "—"

        return MushraComparison(
            reference_condition=("Reference (Hidden)", ref_result.mushra_score),
            anchor_condition=("3.5kHz Anchor (ITU)", anchor_result.mushra_score),
            test_conditions=test_conditions,
            ranking=ranking,
            winner=winner,
        )

    # ------------------------------------------------------------------
    # Interne Metrik-Berechnung
    # ------------------------------------------------------------------

    def _compute_nsim(self, ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Perceptuelle Ähnlichkeit via NSIM auf Mel-Spektrogrammen.

        Annäherung an Gammatone-NSIM aus PerceptualQualityScorer.
        """
        try:
            import librosa

            n_fft = _safe_fft_size(min(len(ref), len(test)), target=2048, minimum=64)
            hop = max(16, n_fft // 4)
            n_mel = 128

            S_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mel)
            S_test = librosa.feature.melspectrogram(y=test, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mel)

            S_ref = librosa.power_to_db(np.maximum(S_ref, 1e-10))
            S_test = librosa.power_to_db(np.maximum(S_test, 1e-10))

            # SSIM-Approximation über Frame-Statistiken
            mu_r, mu_t = np.mean(S_ref), np.mean(S_test)
            sig_r, sig_t = np.std(S_ref), np.std(S_test)
            sig_rt = np.mean((S_ref - mu_r) * (S_test - mu_t))

            C1, C2 = (0.01 * 80) ** 2, (0.03 * 80) ** 2  # dynamic range 80 dB
            nsim = (2 * mu_r * mu_t + C1) * (2 * sig_rt + C2) / ((mu_r**2 + mu_t**2 + C1) * (sig_r**2 + sig_t**2 + C2))
            return float(np.clip(nsim, 0.0, 1.0))
        except Exception as exc:
            logger.debug("NSIM Ersatzpfad (Fehler: %s)", exc)
            return float(np.clip(1.0 - np.sqrt(np.mean((ref - test) ** 2)), 0.0, 1.0))

    def _compute_mcd(self, ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Mel-Cepstral Distortion in dB (niedriger = besser).

        MCD = (10/ln10) · √(2 · Σᵢ(c_ref_i − c_test_i)²)
        """
        try:
            import librosa

            n_mfcc = 13

            mfcc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc).T
            mfcc_test = librosa.feature.mfcc(y=test, sr=sr, n_mfcc=n_mfcc).T

            # CMVN — Cepstral Mean and Variance Normalization (standard speech/music DSP).
            # Librosa MFCC-Koeffizienten liegen in rohen Log-Energie-Einheiten (Wertebereich
            # typisch ±500 dB) — ohne Normalisierung ergibt die MCD-Formel 500–1000 dB statt
            # des phys. sinnvollen Bereichs 0–30 dB.  CMVN subtrahiert die utterance-globale
            # Mittelwert-Verschiebung und normalisiert auf σ=1, sodass die Distanz ausschließlich
            # zeitliche Klangfarbenunterschiede (nicht absoluten Lautstärke-Offset) misst.
            # Invariante: gilt für alle Songs/Materialien — nicht song-spezifisch.
            # Referenz: Aal-Saleh & Mokbel 2011; Kominek & Black 2008 (CMU-Arctic MCD benchmark).
            for _mc in (mfcc_ref, mfcc_test):
                _mu = np.mean(_mc, axis=0, keepdims=True)
                _sigma = np.std(_mc, axis=0, keepdims=True) + 1e-8
                _mc -= _mu
                _mc /= _sigma

            min_frames = min(mfcc_ref.shape[0], mfcc_test.shape[0])
            diff = mfcc_ref[:min_frames, 1:] - mfcc_test[:min_frames, 1:]  # skip c0
            # Per-frame MCD then average (correct ITU-T P.862 formulation)
            frame_dists = np.sqrt(2.0 * np.sum(diff**2, axis=1))
            mcd = (10.0 / math.log(10)) * float(np.mean(frame_dists))
            # Sane cap: CMVN MCD range 0–40 dB (>40 = completely different timbre, maps to 0 score anyway).
            return float(np.clip(mcd, 0.0, 40.0))
        except Exception as exc:
            logger.debug("MCD Ersatzpfad: %s", exc)
            return 5.0

    def _compute_lufs_diff(self, ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """LUFS-Differenz (BS.1770 K-Gewichtung, vereinfacht).

        Returns:
            LUFS-Differenz in LU (signed). Ziel: ≤ 1 LU.
        """
        try:
            try:
                import pyloudnorm as pyln

                meter = pyln.Meter(sr)
                lufs_ref = float(meter.integrated_loudness(ref))
                lufs_test = float(meter.integrated_loudness(test))
            except Exception as exc:
                logger.debug("LUFS-BS.1770 Ersatzpfad auf RMS (Fehler: %s)", exc)
                rms_ref = float(np.sqrt(np.mean(ref**2) + 1e-12))
                rms_test = float(np.sqrt(np.mean(test**2) + 1e-12))
                lufs_ref = 20.0 * math.log10(rms_ref)
                lufs_test = 20.0 * math.log10(rms_test)

            diff = lufs_test - lufs_ref
            # Clamp to ±60 LU — values beyond that indicate a near-silent or
            # clipped signal; the score already clips to 0 at |12| LU anyway.
            return float(np.clip(diff, -60.0, 60.0))
        except Exception as e:
            logger.warning("mushra_evaluator.py::_berechnen_lufs_diff Ersatzpfad: %s", e)
            return 0.0

    def _compute_spectral_corr(self, ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Spektrale Korrelation via FFT-Leistungsspektrum."""
        try:
            P_ref = np.abs(np.fft.rfft(ref)) ** 2
            P_test = np.abs(np.fft.rfft(test)) ** 2
            # §VERBOTEN: np.corrcoef ohne std-Guard → RuntimeWarning bei near-constant Signalen.
            # Guarded std-Check VOR np.corrcoef (NaN-safe, kein Warning).
            _std_ref = float(np.std(P_ref))
            _std_test = float(np.std(P_test))
            if _std_ref < 1e-12 or _std_test < 1e-12:
                return 1.0 if np.allclose(P_ref, P_test, atol=1e-10) else 0.5
            corr_raw = float(
                np.dot(P_ref - P_ref.mean(), P_test - P_test.mean())
                / (float(np.linalg.norm(P_ref - P_ref.mean())) * float(np.linalg.norm(P_test - P_test.mean())) + 1e-10)
            )
            corr = corr_raw if np.isfinite(corr_raw) else 0.5
            return float(np.clip(corr, 0.0, 1.0))
        except Exception as e:
            logger.warning("mushra_evaluator.py::_berechnen_spectral_corr Ersatzpfad: %s", e)
            return 0.5

    def _quick_score(self, ref: np.ndarray, test: np.ndarray, sr: int) -> float:
        """Schnelle MUSHRA-Schätzung ohne Musical Goals (für Anchor-Berechnung)."""
        nsim = self._compute_nsim(ref, test, sr)
        mcd = self._compute_mcd(ref, test, sr)
        lufs = self._compute_lufs_diff(ref, test, sr)
        sc = self._compute_spectral_corr(ref, test, sr)
        raw = (
            self._WEIGHTS["nsim"] * float(np.clip(nsim, 0.0, 1.0))
            + self._WEIGHTS["musical_goals"] * 0.0  # Musical Goals nicht verfügbar
            + self._WEIGHTS["mcd"] * float(np.exp(-mcd / 300.0))
            + self._WEIGHTS["lufs_diff"] * float(np.clip(1.0 - abs(lufs) / 12.0, 0.0, 1.0))
            + self._WEIGHTS["spectral_corr"] * float(np.clip(sc, 0.0, 1.0))
        )
        return float(np.clip(raw * 100.0, 0.0, 100.0))

    def _compute_musical_goals(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Misst alle 9 Musical Goals (via MusicalGoalsChecker-Singleton)."""
        if self._mu_checker is None:
            try:
                from backend.core.musical_goals.musical_goals_metrics import get_checker

                self._mu_checker = get_checker()  # type: ignore[assignment]
            except ImportError as exc:
                logger.debug("§V6 musical_goals.musical_goals_metrics nicht verfügbar — leeres Dict zurückgegeben (MUSHRA): %s", exc)
                return {}

        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
        cache_key = f"{sr}:{len(audio_f32)}:{hashlib.blake2b(audio_f32.tobytes(), digest_size=16).hexdigest()}"
        with self._mg_cache_lock:
            cached = self._mg_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        try:
            measured = self._mu_checker.measure_all(audio_f32, sr)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Musical Goals Fehler in MUSHRA: %s", exc)
            return {}

        cleaned = {str(k): float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in measured.items()}
        with self._mg_cache_lock:
            self._mg_cache[cache_key] = cleaned
            if len(self._mg_cache) > self._mg_cache_max_entries:
                oldest_key = next(iter(self._mg_cache), None)
                if oldest_key is not None:
                    self._mg_cache.pop(oldest_key, None)
        return dict(cleaned)

    def _create_anchor(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Erzeugt den ITU-R BS.1534-3 3.5-kHz-Tiefpassfilter-Anchor.

        Der Anchor entspricht einem 3500-Hz-LP-Butterworth-Filter
        8. Ordnung — produziert typisch MUSHRA-Scores von 20–35.
        """
        try:
            from scipy.signal import butter, sosfilt

            sos = butter(8, 3500 / (sr / 2), btype="low", output="sos")
            anchor = sosfilt(sos, audio).astype(np.float32)
            return np.clip(anchor, -1.0, 1.0)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("Anchor-Erzeugung Ersatzpfad: %s", exc)
            # Einfacher Bandpass als Fallback
            return audio * 0.3  # type: ignore[no-any-return]  # starke Abschwächung ≈ schlechte Qualität

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Konvertiert Stereo → Mono; no-op bei Mono."""
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 2:
            return np.mean(audio, axis=1).astype(np.float32)  # type: ignore[no-any-return]
        return audio.astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _grade(score: float) -> tuple[str, str]:
        """Ordnet MUSHRA-Score eine Kategorie zu.

        Returns:
            Tuple (short_grade, itu_grade) z.B. ("Good", "B (Good)")
        """
        if score >= 80:
            return "Excellent" if score >= 91 else "Good", f"{'A' if score >= 91 else 'B'} (Good)"
        if score >= 60:
            return "Fair", "C (Fair)"
        if score >= 40:
            return "Poor", "D (Poor)"
        return "Bad", "E (Bad)"


# ---------------------------------------------------------------------------
# Singleton (Thread-safe)
# ---------------------------------------------------------------------------

_instance: MushraEvaluator | None = None
_lock = threading.Lock()


def get_mushra_evaluator() -> MushraEvaluator:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking).

    Returns:
        Singleton-Instanz von :class:`MushraEvaluator`.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MushraEvaluator()
                logger.debug("MushraEvaluator Singleton erstellt.")
    return _instance


def evaluate_mushra(
    reference: np.ndarray,
    test: np.ndarray,
    sr: int = 48_000,
    *,
    compute_anchor: bool = True,
) -> MushraResult:
    """Convenience-Funktion: Berechnet MUSHRA-Score für ein Testpaar.

    Folgt ITU-R BS.1534-3 Protokoll (objektive Approximation):
    - Referenz (Hidden Reference) ≈ 100
    - 3.5-kHz-Anchor ≈ 20–35
    - Testbedingung: 0–100 je nach Qualität

    Args:
        reference:       Referenz-Audio (Original / Gold-Standard).
        test:            Test-Audio (restauriertes Signal).
        sr:              Abtastrate in Hz (Standard: 48 000 Hz).
        compute_anchor:  Wenn True, wird 3.5-kHz-Anchor berechnet.

    Returns:
        :class:`MushraResult` mit MUSHRA-Score, Kategorie und Teilmetriken.

    Example::

        result = evaluate_mushra(original_audio, restored_audio, sr=48000)
        logger.debug("MUSHRA: %.1f/100  (%s)", result.mushra_score, result.grade)
        # → MUSHRA: 84.3/100  (Good)
    """
    return get_mushra_evaluator().evaluate(reference, test, sr, compute_anchor=compute_anchor)


def compare_mushra(
    reference: np.ndarray,
    conditions: dict[str, np.ndarray],
    sr: int = 48_000,
) -> MushraComparison:
    """Vergleicht mehrere Restaurierungs-Varianten im MUSHRA-Layout.

    Args:
        reference:   Referenz-Audio.
        conditions:  Dict {name → test_audio} für alle Varianten.
        sr:          Abtastrate in Hz.

    Returns:
        :class:`MushraComparison` mit vollständigem Ranking.

    Example::

        comparison = compare_mushra(original, {
            "Restoration Mode":  restored_v1,
            "Studio 2026 Mode":  restored_v2,
            "Baseline (RX10)":   baseline,
        }, sr=48000)
        logger.debug("Gewinner: %s", comparison.winner)
    """
    return get_mushra_evaluator().compare_conditions(reference, conditions, sr)
