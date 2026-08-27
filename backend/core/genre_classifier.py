from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)
# pylint: disable=import-outside-toplevel


@dataclass
class SchlagerClassificationResult:
    """Ergebnis der Schlager-Klassifikation mit Beitrag jeder Erkennungsschicht."""

    # Required fields — must be provided by all callers (including UV3 §Dach GlobalPlan-Prior)
    is_schlager: bool
    confidence: float
    genre_label: str
    subgenre: str
    bpm: float
    # Optional tier-score fields — default to 0.0 when reconstructed from GlobalPlan context
    clap_score: float = 0.0
    accordion_score: float = 0.0
    harmonic_simplicity: float = 0.0
    rhythm_score: float = 0.0
    vocal_german_prior: float = 0.0
    melodic_repetition: float = 0.0
    vocal_language_score: float = 0.5  # 1.0 = klar Deutsch, 0.0 = klar Englisch
    dsp_language_score: float = 0.5
    lyrics_language_hint: float = 0.0
    genre_family: str = "unknown"
    genre_family_confidence: float = 0.0
    top_genres: list[tuple[str, float]] = field(default_factory=list)
    open_set_unknown: bool = False
    key: str = ""
    reasoning: str = ""

    @property
    def primary_genre_label(self) -> str:
        """Primäres Genre-Label — genre-unabhängig (Schlager oder Nicht-Schlager)."""
        return self.genre_label

    @property
    def language_code(self) -> str:
        """ISO 639-1 language code inferred from vocal language score.

        Returns 'de' when German is likely (score ≥ 0.40), 'en' otherwise.
        Used by MediumDetector for language-aware chain building.
        """
        return "de" if self.vocal_language_score >= 0.40 else "en"

    @property
    def primary_genre_confidence(self) -> float:
        """Primäre Genre-Klassifikationssicherheit — maximaler verfügbarer Score.

        Für Schlager-Material: confidence (Schlager-Wahrscheinlichkeit).
        Für Nicht-Schlager-Material: genre_family_confidence (Genre-Family-Score).
        Immer der semantisch richtige Wert für 'wie sicher ist das Genre-Ergebnis'.
        """
        return max(self.confidence, self.genre_family_confidence)


#: Backward-compatible alias — UV3 and other callsites import as ``GenreResult``
GenreResult = SchlagerClassificationResult


# ── §Spec 24: Deterministische DSP-Ersatzpfade für numba-defekte librosa-Aufrufe ──
# Im ROCm-Venv ist der numba-Dispatcher ein plain function ohne get_call_template —
# librosa.onset/chroma werfen dann AttributeError. Konstanten-Fallbacks
# (2.0, „Unbekannt“) degradieren Genre/BPM dauerhaft; diese DSP-Pfade liefern
# echte Messungen ohne numba.


def _onset_rate_dsp(mono: np.ndarray, sr: int) -> float:
    """Onset-Dichte (Onsets/s) via Energie-Flux — deterministisch, kein numba."""
    hop = max(256, int(sr * 0.01))  # 10 ms Frames
    n_frames = (len(mono) - hop) // hop
    if n_frames < 8:
        return 2.0
    frames = mono[: n_frames * hop].reshape(n_frames, hop)
    energies = np.mean(frames.astype(np.float64) ** 2, axis=1)
    flux = np.diff(energies, prepend=energies[0])
    thr = max(float(np.percentile(np.abs(flux), 90)), 1e-9)
    onsets = int(np.sum(flux > thr))
    duration_s = float(len(mono)) / float(sr)
    return float(onsets / max(duration_s, 1.0))


_KEY_NAMES_DSP = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "H"]


def _estimate_key_dsp(mono: np.ndarray, sr: int) -> str:
    """Tonart via Pitch-Class-Profile (FFT) — deterministisch, kein numba."""
    n = len(mono)
    if n < 4096:
        return "Unbekannt"
    n_fft = 1 << int(np.log2(min(n, 16384)))
    frame = mono[-n_fft:] * np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(frame.astype(np.float64)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    # MIDI-Pitch-Klassen: p = round(69 + 12*log2(f/440)) mod 12
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = np.round(69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-9) / 440.0))
    valid = (freqs >= 55.0) & np.isfinite(midi)
    pc = midi[valid].astype(int) % 12
    pcp = np.bincount(pc, weights=mag[valid], minlength=12)
    idx = int(np.argmax(pcp))
    return f"{_KEY_NAMES_DSP[idx]}-Dur"


class GermanSchlagerClassifier:
    """Erkennt Deutschen Schlager zuverlässig ohne vortrainiertes Genre-Modell.

    Erkennungskaskade (6 Schichten):
        Tier-1: LAION-CLAP Zero-Shot (optional, weicher Prior)
        Tier-2: Akkordeon-Reed-Beating-Fingerprint (DSP, physikalisch)
        Tier-3: Harmonischer Simplizitäts-Index (HSI, CQT-Chroma)
        Tier-4: Rhythmus-Muster-Klassifikation (madmom / librosa)
        Tier-5: Deutsch-Vokal-Formant-Prior (LPC-Burg, SAMPA)
        Tier-6: Melodische Wiederholungsrate (MFCC-SSM)
    """

    # ---- CLAP Zero-Shot Prompts ----
    SCHLAGER_CLAP_PROMPTS: list[tuple[str, float]] = [
        ("Deutscher Schlager mit Akkordeon und Melodie", 0.25),
        ("German Schlager music with accordion and folk singing", 0.20),
        ("Volksmusik mit Schlagzeug und Bläsern", 0.15),
        ("German folk pop music with simple chord progression", 0.15),
        ("Schunkelmusik Blaskapelle Volksfest", 0.12),
        ("Oompah music accordion brass band", 0.08),
        ("Marschmusik Deutschland Blasorchester", 0.05),
    ]

    NON_SCHLAGER_NEGATIVE_PROMPTS: list[str] = [
        "jazz improvisation complex harmony",
        "orchestral classical music symphony",
        "electronic dance music synthesizer",
        "hip hop rap rhythm and blues",
        "heavy metal electric guitar distortion",
        "English pop singing british accent",
        "American country music english vocals",
        "English language pop ballad singing",
    ]

    # ---- Schwellwerte ----
    # §2.2.2 normative Invariante: Aktivierungsschwelle MUSS 0.52 sein (keine Abweichung).
    SCHLAGER_CONFIDENCE_THRESHOLD: float = 0.52
    CLAP_POSITIVE_THRESHOLD: float = 0.26
    ACCORDION_AM_FREQ_RANGE: tuple[float, float] = (5.0, 15.0)
    ACCORDION_TREMOLO_RANGE: tuple[float, float] = (4.0, 8.0)
    ACCORDION_FREQ_BAND: tuple[float, float] = (150.0, 2500.0)
    HSI_THRESHOLD: float = 0.82
    REPETITION_THRESHOLD: float = 0.42
    BPM_RANGES: dict[str, tuple[float, float]] = {
        "schunkel": (108.0, 162.0),
        "walzer": (140.0, 200.0),
        "marsch": (96.0, 132.0),
        "discoschlager": (116.0, 134.0),
    }

    # Individuelle Tier-Schwellwerte für Voting
    _TIER_THRESHOLDS: list[float] = [0.50, 0.75, 0.55, 0.50, 0.42]
    _NON_SCHLAGER_MIN_SCORE: float = 0.35
    _OPEN_SET_MIN_SCORE: float = 0.38
    _OPEN_SET_MARGIN: float = 0.08

    def classify(self, audio: np.ndarray, sr: int) -> SchlagerClassificationResult:
        """Klassifiziert Audio als Schlager oder Non-Schlager.

        Args:
            audio: float32/64 nd-array, mono oder stereo
            sr:    Sample-Rate in Hz — muss exakt 48000 sein (Spec §3.x).

        Returns:
            SchlagerClassificationResult mit allen Schicht-Scores.
        """
        # SR-agnostic: analysis modules work at native import SR (Spec §Performance-Budget)
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Mono-Konvertierung
        mono = self._to_mono(audio)

        # Spektrale Flachheit (Wiener-Entropie) als Rausch-Vorfilter — VOR Resampling.
        # After downsampling from 48 kHz to 22050 Hz the anti-alias LP filter causes
        # band-limited noise to have lower flatness (~0.38) than true white noise (~0.57).
        # Running the check on the PRE-resampled native-SR mono avoids this false pass.
        if not self._is_music_like(mono):
            return SchlagerClassificationResult(
                is_schlager=False,
                confidence=0.0,
                genre_label="Unbekannt",
                clap_score=0.0,
                accordion_score=0.0,
                harmonic_simplicity=0.5,
                rhythm_score=0.0,
                vocal_german_prior=0.5,
                melodic_repetition=0.35,
                subgenre="unknown",
                bpm=0.0,
                key="?",
                reasoning="Signal ist rauschähnlich (hohe spektrale Flachheit) — kein Schlager.",
            )

        # Resampling to 22050 Hz for all downstream analysis tiers.
        mono = self._resample(mono, sr, 22050)
        sr_a = 22050

        # Tier-1: CLAP (optional). Das Verfügbarkeits-Flag wird vor dem Aufruf
        # zurückgesetzt; nur die echte _compute_clap_score()-Methode setzt es auf
        # True, wenn CLAP nicht geladen werden konnte (offline/Speicherbudget).
        self._clap_score_is_fallback = False
        clap_score = self._compute_clap_score(mono, sr_a)

        # Tier-2: Akkordeon
        accordion_score = self._compute_accordion_score(mono, sr_a)

        # Tier-3: Harmonische Simplizität
        hsi = self._compute_harmonic_simplicity(mono, sr_a)

        # Tier-4: Rhythmus
        rhythm_score, subgenre, bpm = self._classify_rhythm_pattern(mono, sr_a)

        # Tier-5: Vokal-Prior
        vocal_prior = self._compute_german_vocal_prior(mono, sr_a)

        # Tier-6: Melodische Wiederholung
        melodic_rep = self._compute_melodic_repetition(mono, sr_a)

        # Tier-7: Vokalsprach-Erkennung (Deutsch vs. Englisch)
        lang_de_score = self._detect_vocal_language(mono, sr_a)
        dsp_lang_score = float(lang_de_score)
        lyrics_lang_hint = self._compute_lyrics_language_hint(audio, sr)
        if lyrics_lang_hint > 0.0:
            # Fuse DSP language cue with §2.36 lyrics-guided cue.
            lang_de_score = float(np.clip(max(lang_de_score, lyrics_lang_hint), 0.0, 1.0))

        # Ensemble-Voting
        tier_scores = [accordion_score, hsi, rhythm_score, vocal_prior, melodic_rep]
        n_active = sum(1 for s, t in zip(tier_scores, self._TIER_THRESHOLDS) if s >= t)
        weighted_mean = (
            0.20 * accordion_score
            + 0.20 * hsi
            + 0.20 * rhythm_score
            + 0.10 * vocal_prior
            + 0.15 * melodic_rep
            + 0.15 * hsi  # doppeltes Gewicht für HSI (wichtigstes DSP-Merkmal)
        ) / 1.0  # Normalisierung bereits 1.0
        if getattr(self, "_clap_score_is_fallback", False):
            # CLAP nicht verfügbar (offline-Desktop/Speicherbudget/Plugin) → das für
            # CLAP reservierte 0.30-Gewicht auf die tatsächlich GEMESSENE DSP-Evidenz
            # umverteilen, statt einen neutralen Platzhalter (0.35) als echte Messung
            # einzumischen. Fehlende OPTIONALE Evidenz darf die Konfidenz nicht unter
            # das drücken, was die DSP-Merkmale belegen — sonst bleibt z. B. eindeutiges
            # bandbegrenztes Schlager-Material künstlich unsicher (conf≈0.507 trotz
            # konvergenter DSP-Tiers). WISSENSCHAFTLICHE INVARIANTE (§0g, §0c): rein
            # materialunabhängig, keine Schwellwert-Absenkung, keine künstliche
            # Inflation — die Konfidenz spiegelt exakt die verfügbare Evidenz.
            confidence = float(np.clip(weighted_mean, 0.0, 1.0))
        else:
            confidence = float(np.clip(0.30 * clap_score + 0.70 * weighted_mean, 0.0, 1.0))
            # §2.13 CLAP-DSP-Konsistenz-Gate: CLAP ist ein general-purpose
            # Audio-Modell und kein Genre-Spezialist. Wenn CLAP den Schlager-Score
            # systematisch unter das DSP-Ensemble drückt (CLAP < 0.35, DSP ≥ 0.50),
            # ist CLAP für dieses Material nicht aussagekräftig → pure DSP verwenden.
            # Sonst würde ein funktionierendes DSP-Ensemble durch irrelevante
            # CLAP-Embeddings künstlich verschlechtert.
            if clap_score < 0.35 and weighted_mean >= 0.50:
                logger.info(
                    "GenreClassifier: CLAP-Wert %.3f < 0.35 bei DSP-Ensemble %.3f ≥ 0.50 "
                    "→ CLAP nicht aussagekräftig, pure-DSP-Konfidenz",
                    clap_score,
                    weighted_mean,
                )
                confidence = float(np.clip(weighted_mean, 0.0, 1.0))

        # Sprach-Penalty: klar englischer Gesang (lang_de_score < 0.30) → confidence −15 %
        if lang_de_score < 0.30:
            confidence = float(np.clip(confidence * 0.85, 0.0, 1.0))

        is_schlager = (n_active >= 3) and (confidence >= self.SCHLAGER_CONFIDENCE_THRESHOLD)
        if not is_schlager and self._is_schlager_near_miss(
            n_active=n_active,
            confidence=confidence,
            hsi=hsi,
            rhythm_score=rhythm_score,
            vocal_prior=vocal_prior,
            melodic_rep=melodic_rep,
            lang_de_score=lang_de_score,
        ):
            # Prevent non-Schlager fallback mislabels (e.g. "Jazz") for German
            # Schlager tracks that narrowly miss the strict threshold gate.
            is_schlager = True
            confidence = float(max(confidence, self.SCHLAGER_CONFIDENCE_THRESHOLD))

        centroid = self._spectral_centroid_hz(mono, sr_a)
        onset = self._onset_rate(mono, sr_a)
        dr_db = self._dynamic_range_db(mono, sr_a)
        non_schlager_scores = self._compute_non_schlager_scores(centroid, onset, hsi, dr_db, bpm)
        # Kong et al. (2020) PANNs: learned tag priors are fused conservatively
        # and only allowed to increase confidence where DSP evidence is compatible.
        _panns_prior = self._compute_panns_genre_prior(mono, sr_a)
        if _panns_prior:
            non_schlager_scores = self._fuse_non_schlager_with_panns(non_schlager_scores, _panns_prior)
        alt_genre, alt_conf = self._pick_non_schlager_genre(non_schlager_scores)

        # Post-classification veto: Schlager features + German vocal override misleading
        # alternative labels.  Jazz, Soul/R&B, and Gospel share DSP feature overlap with
        # Deutscher Schlager (moderate HSI, warm centroid, analog dynamics, BPM 90–130).
        # When at least 1 Schlager tier is active AND either German vocal OR Schlager-typical
        # harmonics are detected, these genres are wrong labels for a German Schlager track.
        # Guard: n_active >= 1 — veto only when *some* Schlager evidence exists; prevents
        # misclassifying genuine (English) Soul/R&B or Jazz as Schlager.
        if (
            not is_schlager
            and n_active >= 1
            and alt_genre in {"Jazz", "Soul/R&B", "Gospel"}
            and (hsi >= 0.50 or lang_de_score >= 0.30)
        ):
            is_schlager = True
            confidence = float(max(confidence, self.SCHLAGER_CONFIDENCE_THRESHOLD))

        # Latin/Reggae-Veto: Diese Genres sind bei deutschem Sprachmaterial (lang_de_score >= 0.30)
        # und vorhandener Schlager-Evidenz (n_active >= 1) physikalisch ausgeschlossen.
        # hsi >= 0.50 wird hier NICHT als Alternative akzeptiert, weil echte Latin-Musik
        # ebenfalls moderate HSI-Werte aufweisen kann — nur Deutsch-Sprach-Evidenz ist sicher.
        if not is_schlager and n_active >= 1 and alt_genre in {"Latin", "Reggae"} and lang_de_score >= 0.30:
            is_schlager = True
            confidence = float(max(confidence, self.SCHLAGER_CONFIDENCE_THRESHOLD))

        # Hip-Hop-Veto: Deutscher Schlager und Hip-Hop teilen DSP-Merkmale
        # (einfache Harmonik, moderate Onset-Dichte, komprimierte Dynamik).
        # Hip-Hop ist bei deutschsprachigem Material mit Schlager-Evidenz
        # physikalisch ausgeschlossen — die Genre-Merkmale entstehen durch
        # die analoge Kette (Vinyl→Cassette→mp3), nicht durch Hip-Hop-Produktion.
        if not is_schlager and n_active >= 1 and alt_genre == "Hip-Hop" and lang_de_score >= 0.30:
            is_schlager = True
            confidence = float(max(confidence, self.SCHLAGER_CONFIDENCE_THRESHOLD))

        # Genre-Label + Subgenre
        if is_schlager:
            genre_label = self._determine_genre_label(subgenre, bpm, lang_de_score)
        else:
            genre_label = alt_genre
            # Use the higher confidence (schlager near-miss vs. alternative genre)
            if alt_conf > confidence:
                confidence = alt_conf

        schlager_family_score = float(
            np.clip(
                0.30 * rhythm_score + 0.30 * hsi + 0.25 * vocal_prior + 0.15 * lang_de_score,
                0.0,
                1.0,
            )
        )
        family_label, family_confidence = self._infer_genre_family(non_schlager_scores, schlager_family_score)

        top_genres = self._build_top_genres(
            is_schlager=is_schlager,
            primary_label=genre_label,
            primary_confidence=confidence,
            non_schlager_scores=non_schlager_scores,
        )
        open_set_unknown = self._is_open_set_unknown(top_genres)
        if not is_schlager and open_set_unknown:
            # PANNs-rescue: when DSP fails to exceed open-set threshold, try PANNs
            # prior as a fallback (Kong et al. 2020). Only applies when a single genre
            # has unambiguously high PANNs confidence (≥ 0.60) to avoid overconfident
            # rescue on ambiguous material. Advisory-only: Schlager decision is immune.
            _rescue_genre, _rescue_conf = self._panns_open_set_rescue(_panns_prior)
            if _rescue_genre:
                genre_label = _rescue_genre
                confidence = _rescue_conf
                open_set_unknown = False
                logger.debug(
                    "GenreClassifier: PANNs open-set rescue → %s (conf=%.2f)",
                    _rescue_genre,
                    _rescue_conf,
                )
            else:
                genre_label = "Unbekannt"
                confidence = 0.0

        # Tonart (einfache Schätzung)
        key = self._estimate_key(mono, sr_a)

        reasoning = self._build_reasoning(
            is_schlager,
            confidence,
            clap_score,
            accordion_score,
            hsi,
            rhythm_score,
            vocal_prior,
            melodic_rep,
            n_active,
            subgenre,
            lang_de_score,
        )

        if is_schlager:
            logger.info(
                "🎵 %s erkannt — melodische Lead-Stimme und "
                "Schunkelrhythmus werden sorgfältig bewahrt. "
                "Konfidenz=%.2f, Subgenre=%s, Sprache=%.2f",
                genre_label,
                confidence,
                subgenre,
                lang_de_score,
            )

        return SchlagerClassificationResult(
            is_schlager=is_schlager,
            confidence=confidence,
            genre_label=genre_label,
            clap_score=float(np.clip(clap_score, 0.0, 1.0)),
            accordion_score=float(np.clip(accordion_score, 0.0, 1.0)),
            harmonic_simplicity=float(np.clip(hsi, 0.0, 1.0)),
            rhythm_score=float(np.clip(rhythm_score, 0.0, 1.0)),
            vocal_german_prior=float(np.clip(vocal_prior, 0.0, 1.0)),
            melodic_repetition=float(np.clip(melodic_rep, 0.0, 1.0)),
            vocal_language_score=float(np.clip(lang_de_score, 0.0, 1.0)),
            dsp_language_score=float(np.clip(dsp_lang_score, 0.0, 1.0)),
            lyrics_language_hint=float(np.clip(lyrics_lang_hint, 0.0, 1.0)),
            genre_family=family_label,
            genre_family_confidence=float(np.clip(family_confidence, 0.0, 1.0)),
            top_genres=top_genres,
            open_set_unknown=open_set_unknown,
            subgenre=subgenre,
            bpm=float(bpm),
            key=key,
            reasoning=reasoning,
        )

    def _compute_lyrics_language_hint(self, audio: np.ndarray, sr: int) -> float:
        """Derive a German-language hint from §2.36 lyrics-guided transcription.

        This is only used as an additive cue for borderline genre decisions.
        It must never log or persist lyric text.
        """
        if audio.size < max(sr * 8, 1):
            return 0.0
        # Resample to 48 kHz if needed — LGE requires 48 kHz input.
        audio_48k = audio
        sr_48k = sr
        if sr != 48_000:
            try:
                import librosa

                # §VERBOTEN: audio[0] liefert bei (samples×channels)-Shape nur 2 Samples → audio[:, 0] korrekt
                audio_48k = librosa.resample(audio if audio.ndim == 1 else audio[:, 0], orig_sr=sr, target_sr=48_000)
                sr_48k = 48_000
            except Exception as _exc:
                logger.debug("Lyrics hint: resample to 48k fehlgeschlagen: %s", _exc)
                return 0.0

        try:
            import sys as _sys

            _lge_mod = _sys.modules.get("backend.core.lyrics_guided_enhancement")

            # Skip LGE load during pre-analysis / file-open scanning.
            # Whisper + wav2vec2 (~390 MB) must not be loaded on-demand here;
            # only use LGE if the singleton is already initialised from a
            # previous processing run.
            # Guard logic:
            #   - Module not imported yet → return 0.0 (won't trigger lazy load)
            #   - Module imported AND has is_lyrics_guided_loaded → honour its result
            #   - Module imported BUT no is_lyrics_guided_loaded (e.g. test mock) → proceed
            if _lge_mod is None:
                logger.debug("Lyrics hint uebersprungen — LGE module not imported (pre-Analyse guard)")
                return 0.0
            _is_loaded_fn = getattr(_lge_mod, "is_lyrics_guided_loaded", None)
            if _is_loaded_fn is not None and not _is_loaded_fn():
                logger.debug("Lyrics hint uebersprungen — LGE not yet geladen (pre-Analyse guard)")
                return 0.0

            from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement

            lge = get_lyrics_guided_enhancement()
            transcription = lge.transcribe(audio_48k, sr_48k)
        except Exception as exc:
            logger.debug("Lyrics hint nicht verfuegbar for genre classification: %s", exc)
            return 0.0

        words = getattr(transcription, "words", []) or []
        lang = str(getattr(transcription, "language", "") or "").lower()
        if not words:
            return 0.0

        # Start from neutral and then lift for confident German language cues.
        score = 0.5
        if lang.startswith("de"):
            score += 0.20
        elif lang.startswith("en"):
            score -= 0.12

        # German diction often carries clear fricative/plosive articulation.
        n_words = max(1, len(words))
        fric_plosive = 0
        stressed = 0
        conf_sum = 0.0
        for w in words:
            ptype = str(getattr(w, "phoneme_type", "") or "")
            if "fricative" in ptype or ptype == "plosive":
                fric_plosive += 1
            if "stressed" in ptype:
                stressed += 1
            conf_sum += float(getattr(w, "confidence", 0.0) or 0.0)

        fp_ratio = fric_plosive / n_words
        stress_ratio = stressed / n_words
        avg_conf = conf_sum / n_words
        score += 0.10 * min(1.0, fp_ratio / 0.30)
        score += 0.08 * min(1.0, stress_ratio / 0.40)
        score += 0.08 * min(1.0, avg_conf / 0.60)

        return float(np.clip(score, 0.0, 1.0))

    # ---- Tier-2: Akkordeon-Reed-Beating-Fingerprint ----

    def _is_music_like(self, mono: np.ndarray) -> bool:
        """Prüft via spectral flatness (Wiener entropy) whether the signal is music-like.

        White noise yields periodogram flatness ≈ 0.56 (empirical, Blackman window,
        2048 samples — NOT 1.0 as the theoretical ideal; chi-squared(2) periodogram
        variance causes this bias).  Tonal music typically < 0.35.
        Silence is treated as non-music.

        The threshold 0.50 is calibrated against three sample positions to reduce
        statistical variance and reliably separate white/coloured noise from music.

        Returns:
            True  → signal is music-like, continue classification.
            False → signal is noise-like/silent, return non-Schlager immediately.
        """
        if len(mono) < 32:
            return True  # too short for analysis — err on the side of caution
        rms = float(np.sqrt(np.mean(mono**2)))
        if rms < 1e-6:
            return False  # silence
        try:
            n = min(2048, len(mono))
            window = np.blackman(n)
            # Sample at 9 evenly-spaced positions across the signal.
            # Using only 3 positions produced high variance (σ≈0.06 for white noise), which let
            # seed-42 Gaussian noise (flatness ≈ 0.50 at unlucky positions) slip below the
            # former 0.50 threshold and produce a false-positive Schlager classification.
            # 9 positions reduce the standard error ~3× (σ/√9) → reliable separation.
            n_pos = min(9, max(1, (len(mono) - n) // max(1, n // 2) + 1))
            step = max(1, (len(mono) - n) // max(1, n_pos - 1)) if n_pos > 1 else 0
            positions = [min(i * step, len(mono) - n) for i in range(n_pos)]
            flatnesses: list[float] = []
            for pos in positions:
                pos = max(0, min(pos, len(mono) - n))
                segment = mono[pos : pos + n] * window
                spectrum = np.abs(np.fft.rfft(segment)) ** 2
                spectrum = np.clip(spectrum, 1e-30, None)
                log_mean = float(np.exp(np.mean(np.log(spectrum))))
                arith_mean = float(np.mean(spectrum))
                flatnesses.append(log_mean / (arith_mean + 1e-30))
            flatness = float(np.mean(flatnesses))
            # Empirical calibration: white noise ≈ 0.56 (Blackman window, 2048 pts),
            # music typically ≤ 0.40.  Threshold 0.42 increases safety margin against
            # stochastic borderline-noise segments while keeping tonal/music material.
            return flatness <= 0.42
        except Exception as e:
            logger.warning("genre_classifier.py::_is_music_like Ersatzpfad: %s", e)
            return True  # on error: conservative — continue classification

    def _compute_accordion_score(self, mono: np.ndarray, sr: int) -> float:
        """Akkordeon-Reed-Beating via AM-Demodulation.

        Physikalischer Hintergrund: Akkordeon-Reeds sind paarweise leicht verstimmt
        (5–15 Hz Schwebung), sichtbar als Amplitudenmodulation.
        """
        try:
            from scipy.signal import butter, hilbert, sosfilt

            # Bandpass [150, 2500] Hz
            low, high = self.ACCORDION_FREQ_BAND
            nyq = sr / 2.0
            lo_n = float(np.clip(low / nyq, 1e-6, 0.9999))
            hi_n = float(np.clip(high / nyq, 1e-6, 0.9999))
            if lo_n >= hi_n:
                return 0.0
            sos = butter(4, [lo_n, hi_n], btype="band", output="sos")
            filtered = sosfilt(sos, mono)

            # Hüllkurve via Hilbert
            analytic: np.ndarray = hilbert(np.asarray(filtered, dtype=np.float64))  # type: ignore[assignment]
            envelope = np.abs(analytic)

            # Subsampling der Hüllkurve auf 100 Hz
            hop = max(1, sr // 100)
            env_sub = envelope[::hop].astype(np.float32)
            env_sub = np.nan_to_num(env_sub)

            if len(env_sub) < 10:
                return 0.0

            # FFT der Hüllkurve
            fft_env = np.abs(np.fft.rfft(env_sub))
            freqs = np.fft.rfftfreq(len(env_sub), d=1.0 / 100)

            total_energy = float(np.sum(fft_env**2)) + 1e-12

            # Reed-Beating [5, 15] Hz
            rb_lo, rb_hi = self.ACCORDION_AM_FREQ_RANGE
            rb_mask = (freqs >= rb_lo) & (freqs <= rb_hi)
            reed_energy = float(np.sum(fft_env[rb_mask] ** 2))

            # Balgzug-Tremolo [4, 8] Hz
            tr_lo, tr_hi = self.ACCORDION_TREMOLO_RANGE
            tr_mask = (freqs >= tr_lo) & (freqs <= tr_hi)
            tremolo_energy = float(np.sum(fft_env[tr_mask] ** 2))

            score = float(np.clip((reed_energy + 0.5 * tremolo_energy) / total_energy * 20.0, 0.0, 1.0))

            # ---- Tremolo-Diskriminator: Inter-Band-AM-Kohärenz ----
            # Physik: Akkordeon-Reeds haben paarweise unabhängige Verstimmung →
            # AM-Schwebungsfrequenzen UNTERSCHEIDEN sich zwischen Frequenzbändern.
            # Tremolo-Gitarre / Vibrato-Violine hat einen EINZIGEN Modulator →
            # Alle Bänder zeigen denselben AM-Peak (hohe Inter-Band-Kohärenz).
            # Hohe Kohärenz → kein Akkordeon → Score reduzieren.
            try:
                sub_bands = [(150.0, 600.0), (600.0, 1500.0), (1500.0, 2500.0)]
                band_peak_freqs: list[float] = []
                am_band_mask = (freqs >= 4.0) & (freqs <= 15.0)
                if np.any(am_band_mask) and score > 0.10:
                    for blo, bhi in sub_bands:
                        lo_s = float(np.clip(blo / nyq, 1e-6, 0.9999))
                        hi_s = float(np.clip(bhi / nyq, 1e-6, 0.9999))
                        if lo_s >= hi_s:
                            continue
                        sos_s = butter(4, [lo_s, hi_s], btype="band", output="sos")
                        filt_s = sosfilt(sos_s, mono)
                        _filt_s64 = np.asarray(filt_s, dtype=np.float64)
                        env_s = np.abs(hilbert(_filt_s64))[::hop].astype(  # type: ignore[arg-type]
                            np.float32
                        )
                        env_s = np.nan_to_num(env_s)
                        if len(env_s) < 10:
                            continue
                        fft_s = np.abs(np.fft.rfft(env_s))
                        # Peak AM frequency within [4, 15] Hz for this sub-band
                        peak_idx = int(np.argmax(fft_s[am_band_mask]))
                        band_peak_freqs.append(float(freqs[am_band_mask][peak_idx]))

                    if len(band_peak_freqs) == 3:
                        freq_spread = float(np.std(band_peak_freqs))
                        # freq_spread > 2 Hz → different reed-pairs beating independently → Akkordeon
                        # freq_spread < 1 Hz → single modulator → Tremolo/Vibrato → reduce score
                        coherence = float(np.clip(1.0 - freq_spread / 2.0, 0.0, 1.0))
                        score *= 1.0 - 0.40 * coherence
                        logger.debug(
                            "AccordionDiscriminator: band_peaks=%s spread=%.2f Hz coherence=%.2f → Wert×%.2f",
                            [f"{f:.1f}" for f in band_peak_freqs],
                            freq_spread,
                            coherence,
                            1.0 - 0.40 * coherence,
                        )
            except Exception as _disc_exc:
                logger.debug("AccordionDiscriminator uebersprungen: %s", _disc_exc)

            return float(np.nan_to_num(np.clip(score, 0.0, 1.0)))

        except Exception as e:
            logger.debug("AccordionScore Ersatzpfad: %s", e)
            return 0.0

    # ---- Tier-3: Harmonischer Simplizitäts-Index ----

    def _compute_harmonic_simplicity(self, audio: np.ndarray, sr: int) -> float:
        """Harmonischer Simplizitäts-Index (HSI) via CQT-Chroma-Analyse.

        Schlager: HSI ≥ 0.82 (I-IV-V-Dominanz, einfache Harmonik)
        Jazz: HSI ≤ 0.60 (komplexe Harmonik)
        """
        try:
            import librosa

            if len(audio) < max(int(sr * 0.5), 8192):
                return 0.5  # neutral bei sehr kurzem Audio

            hop_len = int(sr * 0.5)  # 500-ms-Hop
            hop_len = max(512, hop_len)

            # Use STFT chroma for short clips to avoid CQT internals requesting
            # large FFT windows (n_fft > signal length) on sparse test inputs.
            if len(audio) < 12000:
                _n_fft = max(512, min(2048, len(audio)))
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_len, n_fft=_n_fft)
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
                        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_len)
                except Exception:
                    _n_fft = max(512, min(4096, len(audio)))
                    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_len, n_fft=_n_fft)
            chroma = np.nan_to_num(chroma)

            if chroma.shape[1] < 2:
                return 0.5

            # Chromatische Übergänge
            chroma_idx = np.argmax(chroma, axis=0)  # Dominante Klasse pro Frame
            n_total = len(chroma_idx) - 1
            if n_total < 1:
                return 0.5

            # Quintenkreis-Abstand
            transitions = np.abs(np.diff(chroma_idx.astype(int)))
            # Minimum-Abstand im Kreissinn (Wrapping bei 12)
            transitions = np.minimum(transitions, 12 - transitions)
            n_simple = int(np.sum(transitions <= 2))
            hsi = float(n_simple / n_total)

            return float(np.clip(np.nan_to_num(hsi), 0.0, 1.0))

        except Exception as e:
            logger.debug("HSI Ersatzpfad: %s", e)
            return 0.5

    # ---- Tier-4: Rhythmus-Muster-Klassifikation ----

    def _classify_rhythm_pattern(self, audio: np.ndarray, sr: int) -> tuple[float, str, float]:
        """Schunkel/Marsch/Walzer-Klassifikation via Beat-Tracking.

        Returns: (rhythm_score, subgenre_label, bpm)
        """
        try:
            import librosa

            if len(audio) < sr:
                return 0.35, "unknown", 120.0

            tempo, _beats = librosa.beat.beat_track(y=audio, sr=sr)  # type: ignore[attr-defined]
            bpm = float(np.asarray(tempo, dtype=np.float64).flat[0])
            if bpm <= 0:
                return 0.35, "unknown", 120.0

            # Half/double-tempo robustness: librosa sometimes estimates
            # double or half the true tempo.  Try original, half, and double
            # candidates and pick the one with the best sub-genre match.
            candidates = [bpm]
            if bpm > 60:
                candidates.append(bpm / 2.0)
            if bpm < 200:
                candidates.append(bpm * 2.0)

            best_score = 0.0
            best_subgenre = "unknown"
            best_bpm = bpm

            for candidate_bpm in candidates:
                for subgenre, (lo, hi) in self.BPM_RANGES.items():
                    if lo <= candidate_bpm <= hi:
                        center = (lo + hi) / 2.0
                        width = (hi - lo) / 2.0
                        dist = abs(candidate_bpm - center) / (width + 1e-8)
                        score = float(np.clip(1.0 - dist * 0.5, 0.5, 1.0))
                        # Penalize half/double tempo slightly (prefer original)
                        if candidate_bpm != bpm:
                            score *= 0.85
                        if score > best_score:
                            best_score = score
                            best_subgenre = subgenre
                            best_bpm = candidate_bpm

            if best_score == 0.0:
                best_score = 0.25
                best_subgenre = "unknown"

            return float(np.nan_to_num(best_score)), best_subgenre, best_bpm

        except Exception as e:
            logger.debug("RhythmPattern Ersatzpfad: %s", e)
            return 0.35, "unknown", 120.0

    # ---- Tier-5: Deutsch-Vokal-Formant-Prior ----

    def _compute_german_vocal_prior(self, audio: np.ndarray, sr: int) -> float:
        """Deutsch-Vokal-Formantraum-Overlap (SAMPA-Referenz).

        Nur als Tie-Breaker: max. ±0.08 Einfluss auf Gesamt-Score.
        """
        try:
            if len(audio) < sr // 2:
                return 0.5  # neutral

            # Vokal-Segmente via Energie-Schwelle + ZCR
            frame_len = int(sr * 0.025)  # 25 ms
            all_frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
            # Distribute 200 frames evenly over the full song (not just the first 5 s).
            # Long songs with instrumental intros would otherwise yield vocal_prior ≈ 0.5
            # (no vocal data in first 5 s) and miss the Schlager classification.
            _n_want = 200
            if len(all_frames) <= _n_want:
                frames = all_frames
            else:
                _step = len(all_frames) / _n_want
                frames = [all_frames[int(i * _step)] for i in range(_n_want)]

            f1_vals, f2_vals = [], []

            for frame in frames:  # evenly distributed across full song
                rms = float(np.sqrt(np.mean(frame**2)))
                if not np.isfinite(rms) or rms < 0.01:
                    continue  # Stille

                # Einfaches LPC-Formant-Tracking via Autokorrelations-Methode
                order = 16
                if len(frame) <= order:
                    continue
                try:
                    # Autokorrelations-LPC via Levinson-Durbin (O(order²)) — FFT-based
                    from backend.core.core_utils import fft_autocorr

                    r_full = fft_autocorr(frame, max_lag=order)
                    if not np.isfinite(r_full).all() or r_full[0] < 1e-12:
                        continue
                    lpc_coefs = self._lpc_levinson(r_full, order)
                    if not np.isfinite(lpc_coefs).all():
                        continue

                    # Wurzeln des LPC-Polynoms
                    poly = np.concatenate([[1.0], -lpc_coefs])
                    roots = np.roots(poly)

                    # Nur komplexe Wurzeln mit positivem Imaginärteil
                    formants = []
                    for root in roots:
                        if np.imag(root) > 0:
                            freq = np.angle(root) * sr / (2 * np.pi)
                            if 200 < freq < 3500:
                                formants.append(freq)
                    formants.sort()

                    if len(formants) >= 2:
                        f1_vals.append(formants[0])
                        f2_vals.append(formants[1])
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue

            if len(f1_vals) < 5:
                return 0.5  # zu wenig Daten

            f1_arr = np.array(f1_vals)
            f2_arr = np.array(f2_vals)

            # Deutsche Vokal-Polygone (SAMPA)
            german_regions = [
                # ä: F1 ∈ [600, 900], F2 ∈ [1700, 2200]
                ((600, 900), (1700, 2200)),
                # ö: F1 ∈ [380, 520], F2 ∈ [1300, 1700]
                ((380, 520), (1300, 1700)),
                # ü: F1 ∈ [270, 380], F2 ∈ [1900, 2300]
                ((270, 380), (1900, 2300)),
                # a: F1 ∈ [700, 1100], F2 ∈ [1000, 1600]
                ((700, 1100), (1000, 1600)),
            ]

            n_in = 0
            for (f1lo, f1hi), (f2lo, f2hi) in german_regions:
                mask = (f1_arr >= f1lo) & (f1_arr <= f1hi) & (f2_arr >= f2lo) & (f2_arr <= f2hi)
                n_in += int(np.sum(mask))

            overlap = n_in / len(f1_vals)
            prior = float(np.clip(2.0 * overlap, 0.0, 1.0))
            return float(np.nan_to_num(prior))

        except Exception as e:
            logger.debug("VocalPrior Ersatzpfad: %s", e)
            return 0.5

    # ---- Tier-7: Vokalsprach-Erkennung (Deutsch vs. Englisch) ----

    def _detect_vocal_language(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt ob Vokalinhalt eher Deutsch (1.0) oder Englisch (0.0) ist.

        Drei DSP-Merkmale (gewichtetes Mittel):

        1. Umlaut-F2-F1-Gap (Gewicht 0.50):
           Deutsch ü/ö haben F2-F1 > 1400 Hz bei gleichzeitig F1 < 550 Hz.
           Kein englisches Vokal-Phonem besetzt diesen Bereich systematisch.
           Hoher Anteil solcher Frames → klar Deutsch.

        2. F2-Varianz-Bimodalität (Gewicht 0.30):
           Deutsch kontrastiert stark zwischen front-gerundeten Vokalen (ü, ö → hohe F2)
           und Rückenvokalen (u, o → niedrige F2). Englisch hat weniger front-gerundete
           Vokale → niedrigere F2-Standardabweichung relativ zum Mittelwert.
           Normierte F2-Std (σ/µ) > 0.35 → eher Deutsch.

        3. Konsonant-Cluster-Fricative-Band (Gewicht 0.20):
           Deutsch /ch/ (Ich-Laut ~2.5 kHz, Ach-Laut ~1.5 kHz) erzeugt charakteristische
           Energie im 1.2–3.5 kHz Band während stimmloser Passagen. Englisch fehlt dieses
           Paar weitgehend (/ʃ/ konzentriert sich in 3–8 kHz).
           Ratio E(1.2–3.5kHz) / E(3.5–8kHz) in stillen Segmenten > 1.2 → eher Deutsch.

        Args:
            audio: Mono float32, bereits auf 22050 Hz umgetastet.
            sr:    22050.

        Returns:
            lang_de_score ∈ [0.0, 1.0] — 1.0 = klar Deutsch, 0.0 = klar Englisch.
            0.5 = neutral / kein Gesang erkennbar.
        """
        try:
            if len(audio) < sr // 2:
                return 0.5

            frame_len = int(sr * 0.025)  # 25 ms
            hop = frame_len
            all_frames_lang = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, hop)]
            # Distribute 300 frames evenly over the full song (not just the first 7.5 s).
            _n_want_lang = 300
            if len(all_frames_lang) <= _n_want_lang:
                frames = all_frames_lang
            else:
                _step_lang = len(all_frames_lang) / _n_want_lang
                frames = [all_frames_lang[int(i * _step_lang)] for i in range(_n_want_lang)]

            f1_vals: list[float] = []
            f2_vals: list[float] = []
            order = 16

            for frame in frames:  # evenly distributed across full song
                rms = float(np.sqrt(np.mean(frame**2)))
                if not np.isfinite(rms) or rms < 0.01:
                    continue
                if len(frame) <= order:
                    continue
                try:
                    # Levinson-Durbin LPC — FFT-based O(N log N) autocorrelation
                    from backend.core.core_utils import fft_autocorr

                    r_full = fft_autocorr(frame, max_lag=order)
                    if not np.isfinite(r_full).all() or r_full[0] < 1e-12:
                        continue
                    lpc_coefs = self._lpc_levinson(r_full, order)
                    if not np.isfinite(lpc_coefs).all():
                        continue
                    poly = np.concatenate([[1.0], -lpc_coefs])
                    roots = np.roots(poly)
                    formants: list[float] = []
                    for root in roots:
                        if np.imag(root) > 0:
                            freq = np.angle(root) * sr / (2 * np.pi)
                            if 200 < freq < 3500:
                                formants.append(freq)
                    formants.sort()
                    if len(formants) >= 2:
                        f1_vals.append(formants[0])
                        f2_vals.append(formants[1])
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue

            # --- Merkmal 1: Umlaut-Score (F2-F1 > 1400, F1 < 550) ---
            umlaut_score = 0.5
            if len(f1_vals) >= 5:
                f1_arr = np.array(f1_vals)
                f2_arr = np.array(f2_vals)
                umlaut_mask = (f2_arr - f1_arr > 1400.0) & (f1_arr < 550.0)
                umlaut_frac = float(np.sum(umlaut_mask)) / len(f1_vals)
                # 0 Frames → 0.15 (neutral-negativ); > 20 % Frames → 1.0
                umlaut_score = float(np.clip(0.15 + umlaut_frac * 4.25, 0.0, 1.0))

            # --- Merkmal 2: F2-Bimodalität (normierte F2-Standardabweichung) ---
            f2_bimodal_score = 0.5
            if len(f2_vals) >= 10:
                f2_arr = np.array(f2_vals)
                f2_mean = float(np.mean(f2_arr))
                if f2_mean > 100.0:
                    f2_cv = float(np.std(f2_arr)) / f2_mean  # Variationskoeffizient
                    # Deutsch: σ/µ typisch 0.35–0.60; Englisch: 0.20–0.35
                    f2_bimodal_score = float(np.clip((f2_cv - 0.20) / 0.25, 0.0, 1.0))

            # --- Merkmal 3: Konsonant-Cluster /ch/-Band-Ratio ---
            ch_score = 0.5
            try:
                spec = np.abs(np.fft.rfft(audio, n=min(len(audio), 4096 * 8)))
                freqs = np.fft.rfftfreq(min(len(audio), 4096 * 8), d=1.0 / sr)
                ch_band = (freqs >= 1200.0) & (freqs <= 3500.0)
                hf_band = (freqs > 3500.0) & (freqs <= 8000.0)
                e_ch = float(np.mean(spec[ch_band] ** 2)) if np.any(ch_band) else 1e-12
                e_hf = float(np.mean(spec[hf_band] ** 2)) if np.any(hf_band) else 1e-12
                ratio = e_ch / (e_hf + 1e-12)
                # Ratio > 1.2 → eher Deutsch; < 0.8 → eher Englisch
                ch_score = float(np.clip((ratio - 0.8) / 0.8, 0.0, 1.0))
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

            lang_de_score = float(
                np.clip(
                    0.50 * umlaut_score + 0.30 * f2_bimodal_score + 0.20 * ch_score,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "VocalLanguage: umlaut=%.2f f2_bimodal=%.2f ch_Verhaeltnis=%.2f → lang_de=%.2f",
                umlaut_score,
                f2_bimodal_score,
                ch_score,
                lang_de_score,
            )
            return float(np.nan_to_num(lang_de_score, nan=0.5))

        except Exception as exc:
            logger.debug("VocalLanguage Ersatzpfad: %s", exc)
            return 0.5

    # ---- Tier-6: Melodische Wiederholungsrate ----

    def _compute_melodic_repetition(self, audio: np.ndarray, sr: int) -> float:
        """Melodische Wiederholungsrate via MFCC-Self-Similarity-Matrix.

        Schlager (Refrain 3-6×): 0.42 – 0.70
        Jazz (Improvisation): 0.10 – 0.25
        """
        try:
            import librosa

            min_duration_s = 30.0
            if len(audio) < sr * min_duration_s:
                return 0.35  # neutral bei kurzen Dateien

            int(sr * 1.0)  # 1-s-Frames
            hop_len = int(sr * 0.5)  # 0.5-s-Hop
            n_mfcc = 20
            min_gap_frames = 16  # ≥ 8 s bei 0.5s-Hop

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_len)
            mfcc = np.nan_to_num(mfcc)

            T = mfcc.shape[1]
            if min_gap_frames * 2 > T:
                return 0.35

            # SSM (Kosinus-Ähnlichkeit)
            norms = np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-8
            mfcc_n = mfcc / norms  # [n_mfcc, T]

            # Nur Stichprobe für Performance
            max_frames = min(T, 200)
            idx = np.linspace(0, T - 1, max_frames, dtype=int)
            mfcc_s = mfcc_n[:, idx].T  # [max_frames, n_mfcc]

            # Kosinus-SSM
            ssm = mfcc_s @ mfcc_s.T  # [max_frames, max_frames]

            # Ähnliche Paare mit Mindestabstand — vektorisiert (O(n²) Python-Loop vermieden)
            upper_mask = np.triu(np.ones((max_frames, max_frames), dtype=bool), k=min_gap_frames)
            n_total = int(np.sum(upper_mask))
            if n_total == 0:
                return 0.35
            n_similar = int(np.sum((ssm >= 0.85) & upper_mask))

            score = float(n_similar / n_total)
            score = float(np.clip(score * 2.0, 0.0, 1.0))
            return float(np.nan_to_num(score))

        except Exception as e:
            logger.debug("MelodicRepetition Ersatzpfad: %s", e)
            return 0.35

    # ---- Multi-Genre Scoring (Rock / Jazz / Klassik / Oper) ----

    @staticmethod
    def _spectral_centroid_hz(mono: np.ndarray, sr: int) -> float:
        """Weighted mean frequency of the power spectrum (brightness indicator)."""
        n_fft = min(4096, len(mono))
        if n_fft < 256:
            return 2000.0
        hop = n_fft // 2
        centroids: list[float] = []
        for start in range(0, max(1, len(mono) - n_fft), hop):
            frame = mono[start : start + n_fft] * np.hanning(n_fft)
            mag = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            total = float(np.sum(mag)) + 1e-12
            c = float(np.sum(freqs * mag) / total)
            centroids.append(c)
            if len(centroids) >= 200:
                break
        return float(np.median(centroids)) if centroids else 2000.0

    @staticmethod
    def _onset_rate(mono: np.ndarray, sr: int) -> float:
        """Transient onset density (onsets per second)."""
        try:
            import librosa

            if len(mono) < sr:
                return 2.0
            onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="time")  # type: ignore[attr-defined]
            duration_s = len(mono) / sr
            return float(len(onsets) / max(duration_s, 1.0))
        except Exception as e:
            # §Spec 24: librosa-Ausfall (numba-Defekt o. a.) → echter DSP-Messwert
            # statt Konstante 2.0 (BPM/Groove-Features degradierten sonst still).
            logger.warning("genre_classifier.py::_onset_rate DSP-Ersatzpfad: %s", e)
            try:
                return _onset_rate_dsp(mono, sr)
            except Exception as _dsp_exc:
                logger.warning("genre_classifier.py::_onset_rate Ersatzpfad: %s", _dsp_exc)
                return 2.0

    @staticmethod
    def _dynamic_range_db(mono: np.ndarray, sr: int) -> float:
        """Frame-energy P95-P5 spread in dB."""
        import math as _math

        frame_size = max(1, sr // 10)
        n_frames = len(mono) // frame_size
        if n_frames < 5:
            return 25.0
        energies = np.array([np.mean(mono[i * frame_size : (i + 1) * frame_size] ** 2) for i in range(n_frames)])
        energies = energies[energies > 1e-6]
        if len(energies) < 5:
            return 25.0
        p95 = float(np.percentile(energies, 95))
        p5 = float(np.percentile(energies, 5))
        if p5 < 1e-18:
            return 40.0
        return float(np.clip(10.0 * _math.log10(max(p95 / p5, 1.0)), 5.0, 70.0))

    def _score_rock(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        bpm: float,
    ) -> float:
        """Rock genre score: bright spectrum + dense transients + moderate harmony."""
        score = 0.0
        # High spectral centroid → bright/aggressive sound
        if centroid_hz > 2800:
            score += 0.30
        elif centroid_hz > 2200:
            score += 0.15
        # High onset density (drum attacks, power chords)
        if onset_rate > 3.5:
            score += 0.25
        elif onset_rate > 2.5:
            score += 0.12
        # Moderate harmonic complexity (not simple Schlager, not complex Jazz)
        if 0.40 <= hsi <= 0.72:
            score += 0.20
        # Typical Rock BPM range (90–170)
        if 90 <= bpm <= 170:
            score += 0.15
        return float(np.clip(score, 0.0, 1.0))

    def _score_jazz(
        self,
        centroid_hz: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Jazz genre score: complex harmony + wide dynamics + moderate tempo.

        Jazz requires genuinely complex harmony (low HSI). Songs with high harmonic
        simplicity (hsi >= 0.68, typical for Schlager/Folk/Pop) cannot be Jazz —
        the threshold is intentionally strict to avoid false Jazz labels for simple-
        harmony German music that has been degraded through analogue chain + MP3.
        """
        score = 0.0
        # Low HSI = complex harmony (quintessential Jazz feature).
        # Threshold tightened from 0.65 → 0.55: songs with hsi 0.55-0.65 have
        # intermediate simplicity that is NOT characteristic of Jazz (harmonic
        # complexity is the single most defining Jazz feature). Schlager and similar
        # simple-harmony styles sit in the 0.60-0.82 range even after analogue degradation.
        if hsi < 0.50:
            score += 0.40
        elif hsi < 0.55:
            score += 0.20
        # Anti-Jazz guard: harmonic simplicity in the Schlager/Pop/Folk range → zero Jazz.
        # Threshold tightened from 0.68 → 0.58: any song with hsi ≥ 0.58 lacks the
        # complex harmony that is the single defining feature of Jazz.
        if hsi >= 0.58:
            return 0.0
        # Wide dynamic range (expressive playing).
        # Thresholds raised from 35/25 → 40/32: analogue-chain recordings (Vinyl→Kassette→MP3)
        # naturally exhibit wide DR; this must NOT be credited as Jazz.
        if dr_db > 40:
            score += 0.20
        elif dr_db > 32:
            score += 0.08
        # Moderate spectral centroid (warm, not aggressive); tightened to 1500-2800 Hz
        # (the original 1400-3200 range is too broad and catches Schlager warm-vocal tone).
        if 1500 < centroid_hz < 2800:
            score += 0.15
        # Jazz BPM range is extremely variable; moderate tempos common
        if 80 <= bpm <= 200:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_classical(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
    ) -> float:
        """Classical genre score: extreme dynamics + low onset density + diatonic."""
        score = 0.0
        # Very high dynamic range (orchestral pianissimo → fortissimo)
        if dr_db > 42:
            score += 0.35
        elif dr_db > 32:
            score += 0.15
        # Low onset density (no percussion-heavy rhythm)
        if onset_rate < 1.5:
            score += 0.25
        elif onset_rate < 2.5:
            score += 0.10
        # Diatonic but not trivially simple harmony
        if 0.55 <= hsi <= 0.88:
            score += 0.15
        # Lower spectral centroid (rich mids, warm strings)
        if centroid_hz < 2200:
            score += 0.15
        return float(np.clip(score, 0.0, 1.0))

    def _is_schlager_near_miss(
        self,
        *,
        n_active: int,
        confidence: float,
        hsi: float,
        rhythm_score: float,
        vocal_prior: float,
        melodic_rep: float,
        lang_de_score: float,
    ) -> bool:
        """Identify German Schlager near-miss cases to avoid wrong fallback labels.

        Thresholds for hsi and rhythm_score are intentionally relaxed from their
        original values (0.60 / 0.55) to 0.55 / 0.45 to handle recordings that
        have been degraded through an analogue chain (Vinyl→Kassette→MP3): codec
        artefacts and generational noise reduce apparent harmonic simplicity and
        can disturb BPM detection, causing the strict gate to be narrowly missed
        even for unambiguous Schlager material.
        """
        if n_active < 2:
            return False
        if confidence < (self.SCHLAGER_CONFIDENCE_THRESHOLD - 0.12):
            return False
        # Relaxed from 0.60→0.55 (hsi) and 0.55→0.45 (rhythm) to tolerate
        # analogue+codec degradation in recordings like Vinyl→Kassette→MP3.
        if hsi < 0.55 or rhythm_score < 0.45:
            return False
        # vocal_prior and lang_de_score: require only ONE of the two to be strong
        # (long songs with instrumental intros have fewer voiced frames → lower scores).
        # Fallback: HSI >= 0.68 (unambiguous Schlager harmonic simplicity) alone suffices.
        if vocal_prior < 0.50 and lang_de_score < 0.45 and hsi < 0.68:
            return False
        if melodic_rep < 0.36:
            return False
        return True

    def _score_oper(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
    ) -> float:
        """Opera genre score: extreme dynamics + vocal-range centroid + diatonic harmony.

        Key differentiators from Klassik (pure orchestral):
        - Singer's formant (2–3 kHz) raises spectral centroid above purely orchestral material.
        - Very wide DR (singer piano/forte contrasts exceed orchestral range).
        - Moderate onset density: vocal consonants + orchestra, but less than rock.
        """
        score = 0.0
        # Very high dynamic range — even wider than Klassik (singer's piano/forte extremes)
        if dr_db > 48:
            score += 0.35
        elif dr_db > 38:
            score += 0.15
        # Vocal-range spectral centroid: singer's formant (2–3 kHz) elevates centroid
        # above purely orchestral material (Klassik: centroid < 2200 Hz)
        if 1800 < centroid_hz < 3200:
            score += 0.25
        elif 1400 < centroid_hz <= 1800:
            score += 0.10
        # Moderate onset density — vocal consonants + orchestra; less than rock
        if 0.8 < onset_rate < 2.8:
            score += 0.15
        # Diatonic harmony (tonal, like Klassik)
        if 0.55 <= hsi <= 0.88:
            score += 0.15
        return float(np.clip(score, 0.0, 1.0))

    def _score_pop(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Pop genre score: bright centroid + dense onsets + compressed dynamics."""
        score = 0.0
        # Bright, polished sound (mix engineer-boosted highs)
        if centroid_hz > 3000:
            score += 0.25
        elif centroid_hz > 2500:
            score += 0.12
        # High onset density (modern production, programmed beats)
        if onset_rate > 3.5:
            score += 0.25
        elif onset_rate > 2.5:
            score += 0.12
        # Moderately simple harmony (pop songwriting convention)
        if 0.58 <= hsi <= 0.85:
            score += 0.20
        # Compressed loudness-war dynamics (Pop is more compressed than Rock: DR typically < 18)
        if dr_db < 14:
            score += 0.20
        elif dr_db < 18:
            score += 0.10
        # Typical Pop BPM range (90–145)
        if 90 <= bpm <= 145:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_blues(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Blues genre score: pentatonic harmony + expressive dynamics + warm centroid.

        Intermediate harmonic complexity between Jazz (hsi < 0.50) and Schlager (hsi > 0.72).
        Blues musicians use the pentatonic/blues scale — not as chromatic as Jazz but
        significantly more than Schlager's simple I-IV-V diatonism.
        """
        score = 0.0
        # Pentatonic/blues scale: hsi 0.38–0.65; looser upper bound 0.72 for blues-rock
        if 0.38 <= hsi <= 0.65:
            score += 0.35
        elif 0.65 < hsi <= 0.72:
            score += 0.15
        # Warm, mid-focused centroid (guitar body resonance, vocal warmth)
        if 1500 < centroid_hz < 2800:
            score += 0.20
        elif 1200 < centroid_hz <= 1500:
            score += 0.08
        # Wide dynamic range (expressive guitar; no heavy compression)
        if dr_db > 28:
            score += 0.20
        elif dr_db > 20:
            score += 0.10
        # Moderate onset density (guitar + drums; not as dense as Rock)
        if 1.5 <= onset_rate <= 3.5:
            score += 0.15
        # Blues BPM range (60–130, shuffle tempos common)
        if 60 <= bpm <= 130:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_soul_rnb(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Soul/R&B score: gospel-influenced harmony + vocal warmth + moderate compression."""
        score = 0.0
        # Intermediate harmonic complexity (gospel chords, 7ths; not pure Jazz)
        if 0.45 <= hsi <= 0.72:
            score += 0.30
        # Warm vocal centroid range (singer's formant prominent in the mix)
        if 1800 < centroid_hz < 3200:
            score += 0.25
        elif 1400 < centroid_hz <= 1800:
            score += 0.10
        # Moderate onset density (groove-based rhythm section)
        if 2.0 <= onset_rate <= 4.5:
            score += 0.20
        # Moderate dynamics (analog studio warmth, not heavily compressed like modern pop)
        if 18 <= dr_db <= 38:
            score += 0.15
        # Soul/R&B BPM range (70–120)
        if 70 <= bpm <= 120:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_country(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Country-Genre-Score: einfache diatonische Harmonie + helles Twang + weite Dynamik."""
        score = 0.0
        # Simple diatonic harmony (I-IV-V progressions; similar to Schlager but higher centroid)
        if 0.62 <= hsi <= 0.88:
            score += 0.30
        # Bright, twangy spectral centroid (steel guitar, banjo, fiddle presence)
        if 2200 < centroid_hz < 4000:
            score += 0.25
        elif 1800 < centroid_hz <= 2200:
            score += 0.10
        # Wide dynamics (organic, live-sounding recording)
        if dr_db > 22:
            score += 0.20
        # Moderate onset density (pick/strum rhythm patterns)
        if 1.8 <= onset_rate <= 4.0:
            score += 0.15
        # Country BPM range (85–160, two-step to slow ballad)
        if 85 <= bpm <= 160:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_folk(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Folk genre score: very simple harmony + acoustic warmth + low onset density.

        Disambiguation from Klassik: Folk recordings rarely exceed ~38 dB DR.
        Classical/orchestral recordings regularly achieve 40–55 dB DR (pianissimo
        to fortissimo). A penalty is applied for extreme DR to avoid misclassifying
        orchestral or chamber music as Folk.
        """
        score = 0.0
        # Very simple, diatonic harmony (three-chord folk tradition)
        if 0.65 <= hsi <= 0.92:
            score += 0.35
        # Warm, acoustic centroid (acoustic guitar, voice, fiddle)
        if 1400 < centroid_hz < 2600:
            score += 0.25
        elif 1000 < centroid_hz <= 1400:
            score += 0.10
        # Low onset density (strumming, fingerpicking; no programmed beats)
        if onset_rate < 2.0:
            score += 0.25
        elif onset_rate < 2.8:
            score += 0.12
        # Folk BPM range (relaxed tempos, 60–140)
        if 60 <= bpm <= 140:
            score += 0.15
        # Orchestral-range DR (> 40 dB) is not typical of acoustic folk recordings
        # and strongly indicates Klassik/orchestral material instead.
        if dr_db > 40:
            score = float(np.clip(score - 0.25, 0.0, 1.0))
        return float(np.clip(score, 0.0, 1.0))

    def _score_funk(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Funk genre score: dense syncopated 16th-note rhythm + warm-bright centroid + compressed groove.

        Disambiguation from Rock/Metal: Funk is warm-bright, not Rock-aggressive.
        Centroid of 3000–3500 Hz is typical for Rock/Metal distortion; genuine Funk
        has a warmer mix (horn section + slap bass complement = 1800–3500 Hz but
        with a characteristic upper bound). An explicit centroid window ensures
        Rock-parametrized signals are not misclassified as Funk.
        """
        score = 0.0
        # High onset density (syncopated 16th-note patterns, slap bass attacks)
        if onset_rate > 4.0:
            score += 0.30
        elif onset_rate > 3.0:
            score += 0.15
        # Intermediate harmonic complexity (7th chords, sus chords; not Schlager-simple)
        if 0.38 <= hsi <= 0.64:
            score += 0.25
        # Warm-bright presence (horn section, slap bass attack).
        # Rock/Metal brightness (centroid >= 2800 Hz) does NOT indicate Funk.
        # Funk peak = 1800–2800 Hz (horns, wah-guitar); above 2800 Hz is Rock/Metal.
        if 1800 < centroid_hz < 2800:
            score += 0.20
        elif 1400 < centroid_hz <= 1800:
            score += 0.08
        # Funk dynamics: compressed but punchy
        if 12 <= dr_db <= 28:
            score += 0.15
        # Funk BPM range (75–120, groove-based)
        if 75 <= bpm <= 120:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_electronic(
        self,
        centroid_hz: float,
        onset_rate: float,
        _hsi: float = 0.0,
        dr_db: float = 20.0,
        bpm: float = 120.0,
        *,
        hsi: float | None = None,
    ) -> float:
        """Electronic/Dance genre score: very bright + dense onsets + extreme compression.

        Disambiguation from acoustic styles (Folk, Blues, Jazz, Classical):
        Synthesizers and digital production create inherently bright spectral content
        (centroid typically > 2200 Hz). A dark/bass-heavy centroid (< 1800 Hz) combined
        with low DR simply indicates a compressed acoustic recording, NOT Electronic.
        Centroid gate (>= 2200 Hz) prevents false positives on compressed shellac/tape.
        """
        # 'hsi' ist Alias für '_hsi' (Rückwärtskompatibilität)
        if hsi is not None:
            pass
        score = 0.0
        # Electronic gate: synthesis is inherently bright.
        # Without significant high-frequency content (centroid >= 2200 Hz),
        # low DR alone indicates compressed acoustic music, not synthesis.
        if centroid_hz < 2200:
            return 0.0
        # Very bright, artificial synthesis
        if centroid_hz > 3500:
            score += 0.30
        elif centroid_hz > 2800:
            score += 0.15
        # High onset density (kick, clap, hi-hat programming)
        if onset_rate > 4.0:
            score += 0.25
        elif onset_rate > 3.0:
            score += 0.12
        # Extreme loudness-war compression (DR typically very low in electronic music)
        if dr_db < 12:
            score += 0.25
        elif dr_db < 18:
            score += 0.12
        # Electronic dance BPM ranges (120–180 for house/techno/drum and bass)
        if 115 <= bpm <= 185:
            score += 0.20
        return float(np.clip(score, 0.0, 1.0))

    def _score_hiphop(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Hip-Hop genre score: heavy bass + compressed dynamics + moderate onset density.

        Disambiguation from compressed acoustic music:
        Hip-Hop features vocal presence in the mid-range (centroid > 1400 Hz due to
        rapper's formant). A very low centroid (< 1400 Hz) with low DR indicates
        bass-heavy acoustic compression, not Hip-Hop.
        """
        score = 0.0
        # Hip-Hop gate: vocal/sample presence requires centroid > 1400 Hz.
        if centroid_hz < 1400:
            return 0.0
        # Compressed dynamics (heavily produced, mastered loud)
        if dr_db < 15:
            score += 0.30
        elif dr_db < 22:
            score += 0.15
        # Moderate-high onset density (trap hi-hats, boom-bap kick patterns)
        if 2.5 <= onset_rate <= 6.0:
            score += 0.25
        # Moderate spectral centroid (bass-forward but not dull; vocal presence in mid-range)
        if 1500 < centroid_hz < 3200:
            score += 0.20
        # Relatively simple harmonic loops (sample-based, modal)
        if 0.55 <= hsi <= 0.85:
            score += 0.15
        # Hip-Hop BPM range (70–110 boom-bap; 130–180 trap)
        if (70 <= bpm <= 110) or (130 <= bpm <= 180):
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_metal(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Metal genre score: very high centroid + extreme onset density + distorted timbres."""
        score = 0.0
        # Very bright, distortion-rich spectral content
        if centroid_hz > 3200:
            score += 0.30
        elif centroid_hz > 2600:
            score += 0.15
        # Very high onset density (blast beats, rapid power-chord riffing)
        if onset_rate > 5.0:
            score += 0.30
        elif onset_rate > 3.5:
            score += 0.15
        # Complex or power-chord harmony (distortion blurs tonal clarity → low HSI)
        if hsi < 0.55:
            score += 0.20
        elif hsi < 0.67:
            score += 0.10
        # Metal BPM (100–260: doom to blast beat)
        if bpm >= 100:
            score += 0.15
        # Some dynamic range (live recording feel; not full brick-wall)
        if dr_db > 20:
            score += 0.05
        return float(np.clip(score, 0.0, 1.0))

    def _score_latin(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        bpm: float,
    ) -> float:
        """Latin genre score: rhythmic density + moderate harmony + bright centroid.

        Covers Salsa, Bossa Nova, Cumbia, Latin Jazz, Merengue.
        Key differentiators: dense clave-based rhythms, brass/percussion brightness.

        Disambiguation from Electronic/Rock:
        Latin music requires BOTH high onset density AND bright spectral content
        (brass section, percussion transients). Onset density alone without the
        characteristic brass brightness (centroid > 1800 Hz) indicates Rock/Metal,
        not Latin. This prevents false positives on dark-centroid signals (e.g.
        pure tones, bass-heavy material) that happen to have high onset density.
        """
        score = 0.0
        # Latin gate 1: brass/percussion brightness ist Pflicht.
        # Ein dunkles Spektrum (< 1800 Hz) bedeutet keine Blechbläser → kein Latin.
        if centroid_hz < 1800:
            return 0.0
        # Latin gate 2: Latin erfordert dichte synkopierte Rhythmik (Clave-Muster).
        # Schlager-typische Onset-Dichte liegt bei 1.5–3.0; echter Latin-Groove > 2.0.
        # Ein onset_rate < 2.0 deutet auf ruhige Begleitung hin → kein Latin.
        if onset_rate < 2.0:
            return 0.0
        # Dense, syncopated clave-based rhythms (congas, timbales, güiro)
        if onset_rate > 3.5:
            score += 0.30
        elif onset_rate > 2.5:
            score += 0.15
        # BPM-context-aware centroid bonus — prevents Rock/Electric false positives:
        # Salsa (bpm > 150): bright brass is expected at centroid > 2200 Hz.
        # Bossa nova/cumbia (bpm <= 150): dark acoustic spectrum → centroid 1800–2500.
        # A signal at centroid > 2500 Hz AND bpm 90–150 is typical Rock, not Latin.
        if bpm > 150 and centroid_hz > 2200:
            score += 0.25  # salsa / merengue brass (high BPM + bright)
        elif bpm <= 150 and 1800 < centroid_hz < 2500:
            score += 0.25  # bossa nova / cumbia (moderate BPM, darker spectrum)
        elif bpm <= 150 and centroid_hz >= 2500:
            score += 0.05  # marginal; centroid too bright for moderate-BPM Latin
        # Moderate harmonic complexity (salsa = simple; Latin jazz = complex)
        if 0.42 <= hsi <= 0.75:
            score += 0.25
        # Latin BPM ranges: salsa 160–240, bossa nova 80–130, cumbia 80–120
        if (80 <= bpm <= 130) or (160 <= bpm <= 250):
            score += 0.20
        return float(np.clip(score, 0.0, 1.0))

    def _score_gospel(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Gospel genre score: choir-rich vocals + extended chords + expressive dynamics."""
        score = 0.0
        # Gospel harmony: richer than Schlager (major 7ths, extended) but less complex than Jazz
        if 0.45 <= hsi <= 0.72:
            score += 0.30
        # Vocal-prominent centroid (choir raises upper-mid energy)
        if 1800 < centroid_hz < 3200:
            score += 0.25
        # Wide dynamics (call-and-response, emotional climaxes)
        if dr_db > 25:
            score += 0.20
        elif dr_db > 18:
            score += 0.10
        # Moderate onset density (choir consonants + organ/piano rhythmic stabs)
        if 1.5 <= onset_rate <= 3.5:
            score += 0.15
        # Gospel BPM range (60–130)
        if 60 <= bpm <= 130:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _score_reggae(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> float:
        """Reggae genre score: offbeat skank rhythm + bass-heavy warmth + simple harmony.

        Disambiguation from Electronic/Hip-Hop:
        Reggae's defining feature is the sub-bass weight: low centroid (< 2500 Hz)
        combined with slow/moderate tempo (55–95 BPM). High-centroid bright signals
        indicate synthesis or distortion, not Reggae. Electronic music at 120+ BPM
        must not be mistaken for Reggae.
        """
        score = 0.0
        # Reggae gate: BPM must be in the reggae/dub groove range.
        # Electronic/Hip-Hop at 120+ BPM is excluded explicitly.
        if bpm > 100:
            return 0.0
        # Simple, repetitive harmony (I-IV-V, minor pentatonic; reggae chords diatonic)
        if 0.58 <= hsi <= 0.85:
            score += 0.30
        # Bass-heavy, warm spectral profile (sub-bass dominant in reggae/dub mixes)
        if 1200 < centroid_hz < 2600:
            score += 0.25
        elif centroid_hz <= 1200:
            score += 0.10  # very bass-forward sub-style (dub)
        # Moderate dynamics (analog tape warmth, dub mixing headroom)
        if 15 <= dr_db <= 35:
            score += 0.20
        # Reggae BPM range (55–95; characteristic slow groove with heavy sub-bass)
        if 55 <= bpm <= 95:
            score += 0.15
        # Moderate onset density (offbeat skank guitar + steady bass walk)
        if 1.5 <= onset_rate <= 3.5:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _compute_non_schlager_scores(
        self,
        centroid_hz: float,
        onset_rate: float,
        hsi: float,
        dr_db: float,
        bpm: float,
    ) -> dict[str, float]:
        rock_s = self._score_rock(centroid_hz, onset_rate, hsi, bpm)
        jazz_s = self._score_jazz(centroid_hz, hsi, dr_db, bpm)
        classical_s = self._score_classical(centroid_hz, onset_rate, hsi, dr_db)
        oper_s = self._score_oper(centroid_hz, onset_rate, hsi, dr_db)
        pop_s = self._score_pop(centroid_hz, onset_rate, hsi, dr_db, bpm)
        blues_s = self._score_blues(centroid_hz, onset_rate, hsi, dr_db, bpm)
        soul_s = self._score_soul_rnb(centroid_hz, onset_rate, hsi, dr_db, bpm)
        country_s = self._score_country(centroid_hz, onset_rate, hsi, dr_db, bpm)
        folk_s = self._score_folk(centroid_hz, onset_rate, hsi, dr_db, bpm)
        funk_s = self._score_funk(centroid_hz, onset_rate, hsi, dr_db, bpm)
        electronic_s = self._score_electronic(centroid_hz, onset_rate, hsi, dr_db, bpm)
        hiphop_s = self._score_hiphop(centroid_hz, onset_rate, hsi, dr_db, bpm)
        metal_s = self._score_metal(centroid_hz, onset_rate, hsi, dr_db, bpm)
        latin_s = self._score_latin(centroid_hz, onset_rate, hsi, bpm)
        gospel_s = self._score_gospel(centroid_hz, onset_rate, hsi, dr_db, bpm)
        reggae_s = self._score_reggae(centroid_hz, onset_rate, hsi, dr_db, bpm)
        return {
            "Rock": float(np.clip(rock_s, 0.0, 1.0)),
            "Jazz": float(np.clip(jazz_s, 0.0, 1.0)),
            "Klassik": float(np.clip(classical_s, 0.0, 1.0)),
            "Oper": float(np.clip(oper_s, 0.0, 1.0)),
            "Pop": float(np.clip(pop_s, 0.0, 1.0)),
            "Blues": float(np.clip(blues_s, 0.0, 1.0)),
            "Soul/R&B": float(np.clip(soul_s, 0.0, 1.0)),
            "Country": float(np.clip(country_s, 0.0, 1.0)),
            "Folk": float(np.clip(folk_s, 0.0, 1.0)),
            "Funk": float(np.clip(funk_s, 0.0, 1.0)),
            "Electronic": float(np.clip(electronic_s, 0.0, 1.0)),
            "Hip-Hop": float(np.clip(hiphop_s, 0.0, 1.0)),
            "Metal": float(np.clip(metal_s, 0.0, 1.0)),
            "Latin": float(np.clip(latin_s, 0.0, 1.0)),
            "Gospel": float(np.clip(gospel_s, 0.0, 1.0)),
            "Reggae": float(np.clip(reggae_s, 0.0, 1.0)),
        }

    def _compute_panns_genre_prior(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Berechnet conservative genre priors from PANNs tags.

        Literature: Kong et al. (2020) PANNs (IEEE TASLP) and Won et al. (2020)
        show robust cross-dataset music-tag priors. We use them as a weak prior
        only (advisory): priors can increase compatible DSP scores, never force
        a label on their own.
        """
        try:
            from plugins.panns_plugin import classify_audio as _panns_classify_audio

            _tags = _panns_classify_audio(audio, sr)
        except Exception as e:
            logger.warning("genre_classifier.py::_berechnen_panns_genre_prior Ersatzpfad: %s", e)
            return {}
        if not isinstance(_tags, dict) or not _tags:
            return {}

        def _t(name: str) -> float:
            return float(np.clip(float(_tags.get(name, 0.0) or 0.0), 0.0, 1.0))

        _guitar = max(_t("Guitar"), _t("Electric guitar"))
        _drum = max(_t("Drum"), _t("Percussion"))
        _keys = max(_t("Keyboard (musical)"), _t("Piano"))
        _voc = max(_t("Singing voice"), _t("Vocals"))
        _brass = max(_t("Brass instrument"), _t("Trumpet"), _t("Saxophone"))

        # Priors are intentionally soft and sparse to avoid destabilizing DSP rules.
        return {
            "Rock": float(np.clip(max(_t("Rock music"), 0.65 * _guitar + 0.45 * _drum), 0.0, 1.0)),
            "Jazz": float(np.clip(max(_t("Jazz"), 0.55 * _brass + 0.35 * _keys), 0.0, 1.0)),
            "Klassik": float(
                np.clip(max(_t("Classical music"), 0.45 * _keys + 0.35 * _t("Bowed string instrument")), 0.0, 1.0)
            ),
            "Electronic": float(np.clip(max(_t("Electronic music"), 0.60 * _keys + 0.25 * _drum), 0.0, 1.0)),
            "Pop": float(np.clip(0.50 * _voc + 0.25 * _t("Music"), 0.0, 1.0)),
        }

    def _fuse_non_schlager_with_panns(
        self,
        dsp_scores: dict[str, float],
        panns_priors: dict[str, float],
    ) -> dict[str, float]:
        """Fuse DSP scores with PANNs priors using a conservative max-only blend.

        75/25 blend is applied only as an upper candidate and then maxed with
        original DSP score. This means priors cannot reduce scores and therefore
        cannot undo existing validated DSP behavior.
        """
        if not isinstance(dsp_scores, dict) or not isinstance(panns_priors, dict):
            return dsp_scores
        fused = dict(dsp_scores)
        for genre, dsp_val in dsp_scores.items():
            prior = float(np.clip(float(panns_priors.get(genre, 0.0) or 0.0), 0.0, 1.0))
            blended = float(np.clip(0.75 * float(dsp_val) + 0.25 * prior, 0.0, 1.0))
            fused[genre] = float(max(float(dsp_val), blended))
        return fused

    def _panns_open_set_rescue(self, panns_prior: dict[str, float]) -> tuple[str, float]:
        """Rescue open-set 'Unbekannt' using a clear PANNs signal (Kong et al. 2020).

        Only rescues when a single genre exceeds 0.60 and no other genre is within
        0.20 of it (unambiguous signal). Returned confidence is deliberately reduced
        to 0.40 (just above open-set min) to mark low DSP agreement.

        Returns:
            (genre_label, confidence) or ("", 0.0) when no rescue applies.
        """
        if not panns_prior:
            return "", 0.0
        sorted_priors = sorted(panns_prior.items(), key=lambda x: x[1], reverse=True)
        if not sorted_priors:
            return "", 0.0
        best_genre, best_score = sorted_priors[0]
        if best_score < 0.60:
            return "", 0.0
        second_score = sorted_priors[1][1] if len(sorted_priors) > 1 else 0.0
        if (best_score - second_score) < 0.20:
            return "", 0.0  # ambiguous — no rescue
        # Map PANNs Genre-keys (English) to internal labels
        _panns_to_internal = {
            "Rock": "Rock",
            "Jazz": "Jazz",
            "Klassik": "Klassik",
            "Electronic": "Electronic",
            "Pop": "Pop",
        }
        label = _panns_to_internal.get(best_genre, "")
        if not label:
            return "", 0.0
        return label, float(np.clip(0.40 + 0.10 * (best_score - 0.60) / 0.40, 0.40, 0.50))

    def _pick_non_schlager_genre(self, scores: dict[str, float]) -> tuple[str, float]:
        if not scores:
            return "Unbekannt", 0.0
        best_genre = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = float(scores[best_genre])
        if best_score >= self._NON_SCHLAGER_MIN_SCORE:
            return best_genre, best_score
        return "Unbekannt", 0.0

    def _infer_genre_family(
        self,
        non_schlager_scores: dict[str, float],
        schlager_family_score: float,
    ) -> tuple[str, float]:
        g = non_schlager_scores  # shorthand
        family_scores = {
            "schlager_folk": float(
                np.clip(max(schlager_family_score, g.get("Folk", 0.0), g.get("Country", 0.0)), 0.0, 1.0)
            ),
            "rock": float(np.clip(max(g.get("Rock", 0.0), g.get("Metal", 0.0)), 0.0, 1.0)),
            "jazz": float(np.clip(max(g.get("Jazz", 0.0), g.get("Blues", 0.0)), 0.0, 1.0)),
            "klassik": float(np.clip(g.get("Klassik", 0.0), 0.0, 1.0)),
            "oper": float(np.clip(g.get("Oper", 0.0), 0.0, 1.0)),
            "pop": float(np.clip(max(g.get("Pop", 0.0), g.get("Soul/R&B", 0.0)), 0.0, 1.0)),
            "funk_soul": float(
                np.clip(max(g.get("Funk", 0.0), g.get("Soul/R&B", 0.0), g.get("Gospel", 0.0)), 0.0, 1.0)
            ),
            "electronic": float(np.clip(max(g.get("Electronic", 0.0), g.get("Hip-Hop", 0.0)), 0.0, 1.0)),
            "latin": float(np.clip(max(g.get("Latin", 0.0), g.get("Reggae", 0.0)), 0.0, 1.0)),
        }
        label = max(family_scores, key=family_scores.get)  # type: ignore[arg-type]
        score = float(family_scores[label])
        if score < self._OPEN_SET_MIN_SCORE:
            return "unknown", 0.0
        return label, score

    def _build_top_genres(
        self,
        *,
        is_schlager: bool,
        primary_label: str,
        primary_confidence: float,
        non_schlager_scores: dict[str, float],
    ) -> list[tuple[str, float]]:
        top: list[tuple[str, float]] = []
        if primary_label and primary_label.lower() not in ("unknown", "unbekannt"):
            top.append((str(primary_label), float(np.clip(primary_confidence, 0.0, 1.0))))
        ranked = sorted(non_schlager_scores.items(), key=lambda x: x[1], reverse=True)
        for label, score in ranked:
            if score < self._NON_SCHLAGER_MIN_SCORE:
                continue
            if any(lbl.lower() == label.lower() for lbl, _ in top):
                continue
            top.append((label, float(np.clip(score, 0.0, 1.0))))
            if len(top) >= 3:
                break
        if not top and is_schlager:
            top.append(("Schlager", float(np.clip(primary_confidence, 0.0, 1.0))))
        return top

    def _is_open_set_unknown(self, top_genres: list[tuple[str, float]]) -> bool:
        if not top_genres:
            return True
        scores = sorted((float(score) for _, score in top_genres), reverse=True)
        best = scores[0]
        if best < self._OPEN_SET_MIN_SCORE:
            return True
        second = scores[1] if len(scores) > 1 else 0.0
        return (best - second) < self._OPEN_SET_MARGIN

    # ---- Tier-1: CLAP (optional) ----

    def _compute_clap_score(self, audio: np.ndarray, sr: int) -> float:
        """LAION-CLAP Zero-Shot (optionaler weicher Prior).

        Lädt LAION-CLAP via ml_memory_budget.try_allocate() — identisches Muster
        wie alle anderen ML-Plugins (§2.47 ML-Failure-Degradationskaskade).
        Kein Env-Gate: CLAP wird geladen wenn Speicher verfügbar ist, sonst 0.35.

        Setzt ``self._clap_score_is_fallback`` auf True, solange keine echte
        positive CLAP-Messung vorliegt; erst eine erfolgreiche Tag-Inferenz setzt
        das Flag auf False. Der Aufrufer nutzt das Flag, um das CLAP-Gewicht bei
        Nichtverfügbarkeit auf die DSP-Evidenz umzuverteilen.
        """
        # Standardannahme: kein genuiner CLAP-Score → Fallback. Erst eine
        # erfolgreiche Tag-Inferenz (unten) setzt das Flag zurück auf False.
        self._clap_score_is_fallback = True
        # §Spec 24: CLAP embed_audio verlangt exakt 48000 Hz. Analyse-Audio kann
        # mit 22050 Hz ankommen (CQT-Analysepfad) — sonst fällt die Tag-Inferenz
        # auf den Prior zurück (Befund 2026-08-16: „SR muss 48000 Hz sein,
        # erhalten: 22050“). Root-Fix: vor clap.tag resampeln.
        if sr != 48000:
            try:
                audio = self._resample(audio, sr, 48000)
                sr = 48000
            except Exception as _rs48_exc:
                logger.warning(
                    "GenreClassifier CLAP: Resample auf 48k fehlgeschlagen (%s) — neutraler Prior 0.35",
                    _rs48_exc,
                )
                return 0.35
        try:
            from backend.core.ml_memory_budget import release as _release_clap_genre
            from backend.core.ml_memory_budget import try_allocate as _alloc_clap_genre
            from plugins.laion_clap_plugin import get_laion_clap, get_loaded_laion_clap

            if not _alloc_clap_genre("LAION_CLAP_genre", 2.2):
                logger.debug("GenreClassifier CLAP: Speicherbudget nicht verfügbar — neutraler Prior 0.35")
                return 0.35

            try:
                clap = get_loaded_laion_clap()
                if clap is None:
                    clap = get_laion_clap()
                if clap is None:
                    logger.debug("GenreClassifier CLAP: Plugin nicht verfügbar — neutraler Prior 0.35")
                    return 0.35

                # Schlager-Ähnlichkeit via Genre-Tags schätzen
                schlager_prompts = [p for p, _ in self.SCHLAGER_CLAP_PROMPTS[:3]]
                try:
                    tag_result = clap.tag(audio, sr, text_queries=schlager_prompts)
                    genre_scores_dict = tag_result.genre_tags
                    proxy_keys = ["schlager", "volksmusik", "folk", "german", "pop"]
                    proxy_score = 0.0
                    for key in proxy_keys:
                        if key in genre_scores_dict:
                            proxy_score = max(proxy_score, genre_scores_dict[key])
                    clap_score = float(np.clip(proxy_score, 0.0, 1.0))
                    # Genuine positive CLAP-Messung erhalten → kein Fallback.
                    self._clap_score_is_fallback = False
                except Exception as _tag_exc:
                    # §V6 (copilot-instructions.md): ML→DSP-Fallback mit Warnung
                    # + Begründung — hier fiel bisher STILL 0.35 (Befund
                    # 2026-08-16: Resample-Defekt → assert sr==48000 in
                    # embed_audio → unsichtbare Genre-Degradation).
                    logger.warning(
                        "GenreClassifier CLAP: Tag-Inferenz (positiv) fehlgeschlagen (%s) — neutraler Prior 0.35",
                        _tag_exc,
                    )
                    clap_score = 0.35

                pos_total = clap_score

                neg_scores: list[float] = []
                try:
                    neg_tag = clap.tag(audio, sr, text_queries=self.NON_SCHLAGER_NEGATIVE_PROMPTS[:3])
                    _neg_dict = neg_tag.genre_tags if hasattr(neg_tag, "genre_tags") else {}
                    for v in _neg_dict.values():
                        neg_scores.append(float(v))
                except Exception as _neg_exc:
                    # §V6: Negativ-Tag-Fehler sichtbar machen (sonst neg_mean=0.0
                    # ohne Spur — optimistischer Margin ohne Evidenz).
                    logger.warning(
                        "GenreClassifier CLAP: Tag-Inferenz (negativ) fehlgeschlagen (%s) — neg_mean=0.0",
                        _neg_exc,
                    )
                    neg_scores = []
                neg_mean = float(np.mean(neg_scores)) if neg_scores else 0.0
                result_score = float(np.clip(pos_total - 0.5 * neg_mean, 0.0, 1.0))
                logger.info(
                    "GenreClassifier CLAP: Wert=%.3f (pos=%.3f neg_mean=%.3f)", result_score, pos_total, neg_mean
                )
                return float(np.nan_to_num(result_score))

            except Exception as _clap_err:
                logger.debug("GenreClassifier CLAP: Inferenz fehlgeschlagen (%s) — neutraler Prior 0.35", _clap_err)
                return 0.35
            finally:
                _release_clap_genre("LAION_CLAP_genre")

        except (ImportError, Exception) as e:
            logger.debug("GenreClassifier CLAP: Import nicht verfügbar (%s) — neutraler Prior 0.35", e)
            return 0.35

    # ---- Hilfsfunktionen ----

    @staticmethod
    def _lpc_levinson(r: np.ndarray, order: int) -> np.ndarray:
        """Levinson-Durbin Yule-Walker solver — O(order²) vs O(order³) for lstsq.

        Solves the autocorrelation normal equations and returns lpc_coefs such that
        ``poly = np.concatenate([[1.0], -lpc_coefs])`` is the all-pole filter whose
        roots correspond to formant frequencies.  Replaces the former
        ``np.linalg.lstsq`` on the full Toeplitz matrix (16×16 → up to 300 calls per
        classify() call, each O(order³) ≈ 4096 FLOPs → Levinson reduces to ≈256).
        """
        r = np.asarray(r, dtype=np.float64)
        a = np.zeros(order, dtype=np.float64)  # AR coefficients a_ar[1..p]
        e = float(r[0])
        for m in range(order):
            if e < 1e-18:
                break
            # Reflection coefficient: k = -(r[m+1] + a[:m] · r[m:0:-1]) / e
            k = -(float(r[m + 1]) + float(np.dot(a[:m], r[m:0:-1]))) / e
            a_new = a.copy()
            if m > 0:
                a_new[:m] = a[:m] + k * a[m - 1 :: -1]
            a_new[m] = k
            a = a_new
            e *= 1.0 - k * k
            if e < 1e-18:
                break
        # Return -a_ar so that poly = [1, -lpc_coefs] = [1, a_ar] (standard all-pole filter)
        return -a  # type: ignore[no-any-return]

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Konvertiert Stereo → Mono.

        Aurik-Konvention: ``(samples, channels)`` (File-Import liefert z. B.
        ``(1_323_000, 2)``). Der Downmix muss über die **Kanal-Achse** mitteln,
        nicht über die Sample-Achse. Da nur Mono/Stereo unterstützt wird, ist die
        Kanal-Achse stets die kleinere — robust gegen beide Layouts
        (``(samples, channels)`` und ``(channels, samples)``). Spaltenvektoren
        ``(N, 1)`` werden als Mono behandelt.
        """
        if audio.ndim == 2:
            if min(audio.shape) < 2:
                return np.asarray(audio, dtype=np.float32).reshape(-1)  # type: ignore[no-any-return]
            ch_axis = 1 if audio.shape[1] <= audio.shape[0] else 0
            return np.asarray(audio.mean(axis=ch_axis), dtype=np.float32)  # type: ignore[no-any-return]
        return np.asarray(audio, dtype=np.float32)  # type: ignore[no-any-return]

    def _resample(self, audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        """Resampelt auf Ziel-Sample-Rate (numba-Guard via resampling_utils)."""
        if sr_in == sr_out:
            return audio
        try:
            from backend.core.resampling_utils import resample_audio

            return resample_audio(audio, sr_in, sr_out)
        except Exception as e:
            # §V6 (copilot-instructions.md): Fallback mit Warnung + Begründung.
            # Pass-through nur als letzter Ausweg, wenn auch der SciPy-Pfad
            # scheitert (z. B. fehlende scipy-Installation).
            logger.warning("genre_classifier.py::_resample Ersatzpfad (beide Resample-Pfade fehlgeschlagen): %s", e)
            return audio

    def _estimate_key(self, audio: np.ndarray, sr: int) -> str:
        """Einfache Tonart-Schätzung via Chroma."""
        try:
            import librosa

            # chroma_cqt -> vqt -> pitch_tuning may request large FFT windows on
            # short clips; use chroma_stft fallback in that case.
            if len(audio) < 2048:
                return "Unbekannt"
            if len(audio) < 12000:
                _n_fft = max(512, min(2048, len(audio)))
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=_n_fft)
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
                        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
                except Exception:
                    _n_fft = max(512, min(4096, len(audio)))
                    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=_n_fft)
            chroma_mean = np.nan_to_num(chroma.mean(axis=1))
            key_idx = int(np.argmax(chroma_mean))
            key_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "H"]
            return f"{key_names[key_idx]}-Dur"
        except Exception as e:
            # §Spec 24: librosa-Ausfall (numba-Defekt o. a.) → DSP-Pitch-Class-Profile
            # statt „Unbekannt“ (Genre-/Schlager-Merkmale degradierten sonst still).
            logger.warning("genre_classifier.py::_estimate_key DSP-Ersatzpfad: %s", e)
            try:
                return _estimate_key_dsp(audio, sr)
            except Exception as _dsp_exc:
                logger.warning("genre_classifier.py::_estimate_key Ersatzpfad: %s", _dsp_exc)
                return "Unbekannt"

    def _determine_genre_label(self, subgenre: str, _bpm: float, lang_de_score: float = 0.5) -> str:
        """Bestimmt das Genre-Label aus Subgenre, BPM und Sprachscore.

        lang_de_score >= 0.55 → Deutscher Schlager (eindeutig deutschsprachig).
        lang_de_score < 0.30 → Internationaler Schlager (englischsprachig vermutet).
        0.30–0.55 → Schlager (Sprache unsicher).
        """
        mapping = {
            "schunkel": "Schlager",
            "walzer": "Walzer",
            "marsch": "Marsch",
            "discoschlager": "Disco-Schlager",
            "unknown": "Schlager",
        }
        base_label = mapping.get(subgenre, "Schlager")
        if lang_de_score < 0.30:
            return f"Internationaler {base_label}"
        if lang_de_score >= 0.55 and base_label == "Schlager":
            return f"Deutscher {base_label}"
        return base_label

    def _build_reasoning(
        self,
        is_schlager: bool,
        confidence: float,
        _clap: float,
        accordion: float,
        hsi: float,
        rhythm: float,
        _vocal: float,
        melodic: float,
        n_active: int,
        subgenre: str,
        lang_de_score: float = 0.5,
    ) -> str:
        parts = []
        if accordion >= 0.50:
            parts.append(f"Akkordeon-Charakteristik erkannt ({accordion:.2f})")
        if hsi >= self.HSI_THRESHOLD:
            parts.append(f"Harmonische Simplizität hoch ({hsi:.2f})")
        if rhythm >= 0.55:
            parts.append(f"Schlager-Rhythmus '{subgenre}' erkannt ({rhythm:.2f})")
        if melodic >= self.REPETITION_THRESHOLD:
            parts.append(f"Hohe melodische Wiederholungsrate ({melodic:.2f})")
        if lang_de_score >= 0.55:
            parts.append(f"Deutschsprachiger Gesang erkannt ({lang_de_score:.2f})")
        elif lang_de_score < 0.30:
            parts.append(f"Englischsprachiger Gesang vermutet ({lang_de_score:.2f}) → Sprach-Penalty")
        verdict = "Schlager erkannt" if is_schlager else "Kein Schlager"
        return f"{verdict} (Konfidenz={confidence:.2f}, {n_active}/5 DSP-Schichten aktiv). " + "; ".join(parts)


# ---------------------------------------------------------------------------
# SCHLAGER_RESTORATION_PROFILE — Pipeline-Anpassungen bei Schlager-Erkennung (§2.19.3)
# ---------------------------------------------------------------------------

SCHLAGER_RESTORATION_PROFILE: dict = {
    # Akkordeon-Sättigung BEWAHREN (→ DefectType.SOFT_SATURATION-Schutz)
    "soft_saturation_preserve": True,
    "clipping_repair_threshold_db": -3.5,  # Konservativer als Standard (−2.0)
    # TonalCenterMetric verschärft (kein Tonart-Shift bei Schlager)
    "tonal_center_threshold": 0.97,  # Statt Standard 0.95
    # Harmonischer Exciter: im Restoration-Modus UNIVERSAL verboten (§0a, UV3 _restoration_forbidden_stem_enhancement).
    # Dieser Flag ist dokumentarisch — die Durchsetzung erfolgt in UV3, nicht hier.
    "phase_21_exciter_enabled": False,
    # Groove-Erhalt kritisch (Schunkelrhythmus darf nicht begradigt werden)
    "groove_dtw_max_ms": 5.0,  # Strenger als Standard 8.0 ms
    # De-Esser an typischen Schlager-Gesang angepasst
    "deessing_target_hz": 6500,
    "deessing_strength_cap": 0.45,  # Max. 45 % (Standard: 80 %)
    # Brillanz-Ziel leicht gesenkt (Schlager klingt warm, nicht "modern crisp")
    "brillanz_target": 0.82,  # Statt Standard 0.85
    # Wärme-Ziel angehoben (charakteristisch für das Genre)
    "waerme_target": 0.88,  # Statt Standard 0.80
    # Stereo-Breite: historischer Schlager oft Mono/Narrow-Stereo
    "stereo_width_max_era_aware": True,
    # GP-Optimizer Warmstart aus Schlager-spezifischem Gedächtnis
    "gp_memory_key": "schlager",  # ~/.aurik/gp_memory/schlager.json
    # §v10.0.5: Kompressions-Cap für vokal-zentrierten Schlager
    "compression_ratio_cap": 1.8,
}

# Subgenre-Erweiterungen (werden über das Basis-Profil gelegt)
_SUBGENRE_EXTENSIONS: dict = {
    "schlager_1950s": {"flashsr_disabled": True, "max_bandwidth_hz": 12000},
    "schlager_modern": {"flashsr_disabled": True},
    "volksmusik": {"phase_45_priority": "high"},
    "marsch": {"transient_preservation_strength": 1.0, "snare_attack_max_ms": 1.0},
    "walzer": {"groove_meter": "3/4"},
    "discoschlager": {"bass_kraft_target": 0.90, "kick_preserve": True},
}


# ── Genre-Restaurierungsprofile (Spec §2.20) ────────────────────────────────
POP_RESTORATION_PROFILE: dict = {
    "compression_ratio_cap": 2.0,
    "brillanz_target": 0.88,
    "deessing_strength_cap": 0.60,
    "groove_dtw_max_ms": 7.0,
    "gp_memory_key": "pop",
}

BLUES_RESTORATION_PROFILE: dict = {
    "soft_saturation_preserve": True,
    "clipping_repair_threshold_db": -3.0,
    "compression_ratio_cap": 1.5,
    "waerme_target": 0.88,
    "groove_dtw_max_ms": 6.0,
    "gp_memory_key": "blues",
}

SOUL_RNB_RESTORATION_PROFILE: dict = {
    "compression_ratio_cap": 1.8,
    "waerme_target": 0.85,
    "deessing_strength_cap": 0.50,
    "groove_dtw_max_ms": 5.0,
    "gp_memory_key": "soul_rnb",
}

COUNTRY_RESTORATION_PROFILE: dict = {
    "soft_saturation_preserve": True,
    "transient_preservation_strength": 1.0,
    "compression_ratio_cap": 1.5,
    "brillanz_target": 0.85,
    "groove_dtw_max_ms": 7.0,
    "gp_memory_key": "country",
}

FOLK_RESTORATION_PROFILE: dict = {
    "soft_saturation_preserve": True,
    "transient_preservation_strength": 1.0,
    "compression_ratio_cap": 1.3,
    "waerme_target": 0.85,
    "groove_dtw_max_ms": 8.0,
    "gp_memory_key": "folk",
}

FUNK_RESTORATION_PROFILE: dict = {
    "transient_preservation_strength": 1.0,
    "brillanz_target": 0.88,
    "bass_kraft_target": 0.88,
    "groove_dtw_max_ms": 4.0,
    "compression_ratio_cap": 2.2,
    "gp_memory_key": "funk",
}

ELECTRONIC_RESTORATION_PROFILE: dict = {
    "brillanz_target": 0.92,
    "bass_kraft_target": 0.88,
    "compression_ratio_cap": 2.5,
    "groove_dtw_max_ms": 3.0,
    "gp_memory_key": "electronic",
}

HIPHOP_RESTORATION_PROFILE: dict = {
    "bass_kraft_target": 0.90,
    "transient_preservation_strength": 1.0,
    "compression_ratio_cap": 2.0,
    "groove_dtw_max_ms": 4.0,
    "gp_memory_key": "hiphop",
}

METAL_RESTORATION_PROFILE: dict = {
    "transient_preservation_strength": 1.0,
    "clipping_repair_threshold_db": -1.5,
    "soft_saturation_preserve": True,
    "brillanz_target": 0.90,
    "compression_ratio_cap": 2.5,
    "groove_dtw_max_ms": 5.0,
    "gp_memory_key": "metal",
}

LATIN_RESTORATION_PROFILE: dict = {
    "transient_preservation_strength": 1.0,
    "groove_dtw_max_ms": 4.0,
    "brillanz_target": 0.87,
    "compression_ratio_cap": 1.8,
    "gp_memory_key": "latin",
}

GOSPEL_RESTORATION_PROFILE: dict = {
    "waerme_target": 0.86,
    "deessing_strength_cap": 0.45,
    "compression_ratio_cap": 1.6,
    "groove_dtw_max_ms": 6.0,
    "gp_memory_key": "gospel",
}

REGGAE_RESTORATION_PROFILE: dict = {
    "soft_saturation_preserve": True,
    "bass_kraft_target": 0.88,
    "waerme_target": 0.86,
    "groove_dtw_max_ms": 5.0,
    "compression_ratio_cap": 1.8,
    "gp_memory_key": "reggae",
}

# §v10.0.5: Ambient — atmosphärische Klangflächen, lange Hallfahnen,
# keine perkussiven Transienten, oft instrumental. Sub-bass-dominant,
# Hall ist intentional (kein Dereverb), weicher Frequenzgang.
AMBIENT_RESTORATION_PROFILE: dict = {
    "transient_preservation_strength": 0.6,
    "brillanz_target": 0.86,
    "waerme_target": 0.88,
    "bass_kraft_target": 0.85,
    "spatial_depth_threshold": 0.60,
    "groove_dtw_max_ms": 14.0,
    "compression_ratio_cap": 1.2,
    "deessing_strength_cap": 0.60,
    "phase_20_dereverb_enabled": False,
    "phase_49_dereverb_enabled": False,
    "soft_saturation_preserve": True,
    "gp_memory_key": "ambient",
}

# §v10.0.5: World — ethnische Musik, akustische Instrumente,
# oft vokal-lastig mit traditionellen Gesangsstilen. Natürliche
# Dynamik (keine starke Kompression), organischer Klang.
WORLD_RESTORATION_PROFILE: dict = {
    "transient_preservation_strength": 0.9,
    "brillanz_target": 0.85,
    "waerme_target": 0.86,
    "groove_dtw_max_ms": 7.0,
    "compression_ratio_cap": 1.4,
    "deessing_strength_cap": 0.50,
    "soft_saturation_preserve": True,
    "gp_memory_key": "world",
}

JAZZ_RESTORATION_PROFILE: dict = {
    "groove_dtw_max_ms": 4.0,
    "tonal_center_threshold": 0.92,
    "harmonic_exciter_enabled": False,
    "dereverb_strength_cap": 0.30,
    "deessing_strength_cap": 0.50,
    "compression_ratio_cap": 1.8,
    "gp_memory_key": "jazz",
}

KLASSIK_RESTORATION_PROFILE: dict = {
    "phase_20_dereverb_enabled": False,
    "phase_49_dereverb_enabled": False,
    "transient_preservation_strength": 1.0,
    "compression_ratio_cap": 1.3,
    "brillanz_target": 0.88,
    "waerme_target": 0.82,
    "spatial_depth_threshold": 0.82,
    "groove_dtw_max_ms": 10.0,
    "gp_memory_key": "orchestral",
}

OPER_RESTORATION_PROFILE: dict = {
    "deessing_target_hz": 7000,
    "deessing_strength_cap": 0.35,
    "formant_pearson_threshold": 0.97,
    "phase_20_dereverb_enabled": False,
    "vibrato_rate_tolerance_hz": 0.20,
    "de_esser_voice_adaptive": True,
    # §v10.0.5: fehlende Schlüsselparameter ergänzt
    "groove_dtw_max_ms": 12.0,
    "compression_ratio_cap": 1.2,
    "brillanz_target": 0.85,
    "waerme_target": 0.82,
    "transient_preservation_strength": 0.9,
    "gp_memory_key": "opera",
}

ROCK_RESTORATION_PROFILE: dict = {
    "transient_preservation_strength": 1.0,
    "brillanz_target": 0.90,
    "soft_saturation_preserve": True,
    "clipping_repair_threshold_db": -2.0,
    "groove_dtw_max_ms": 6.0,
    "compression_ratio_cap": 2.5,
    "gp_memory_key": "rock",
}

# Alle Profile in einem Dict — für Tests und Iteration
# Keys: Kleinschreibung (intern) UND Großschreibung (Test-Kompatibilität / genre_label)
GENRE_RESTORATION_PROFILES: dict[str, dict] = {
    # Kleinschreibung (intern)
    "schlager": SCHLAGER_RESTORATION_PROFILE,
    "jazz": JAZZ_RESTORATION_PROFILE,
    "klassik": KLASSIK_RESTORATION_PROFILE,
    "oper": OPER_RESTORATION_PROFILE,
    "rock": ROCK_RESTORATION_PROFILE,
    "pop": POP_RESTORATION_PROFILE,
    "blues": BLUES_RESTORATION_PROFILE,
    "soul/r&b": SOUL_RNB_RESTORATION_PROFILE,
    "soul_rnb": SOUL_RNB_RESTORATION_PROFILE,
    "country": COUNTRY_RESTORATION_PROFILE,
    "folk": FOLK_RESTORATION_PROFILE,
    "funk": FUNK_RESTORATION_PROFILE,
    "electronic": ELECTRONIC_RESTORATION_PROFILE,
    "hip-hop": HIPHOP_RESTORATION_PROFILE,
    "hiphop": HIPHOP_RESTORATION_PROFILE,
    "metal": METAL_RESTORATION_PROFILE,
    "latin": LATIN_RESTORATION_PROFILE,
    "gospel": GOSPEL_RESTORATION_PROFILE,
    "reggae": REGGAE_RESTORATION_PROFILE,
    "ambient": AMBIENT_RESTORATION_PROFILE,
    "world": WORLD_RESTORATION_PROFILE,
    # Kapitalisierte Aliases (GermanSchlagerClassifier.genre_label-Format)
    "Schlager": SCHLAGER_RESTORATION_PROFILE,
    "Jazz": JAZZ_RESTORATION_PROFILE,
    "Klassik": KLASSIK_RESTORATION_PROFILE,
    "Oper": OPER_RESTORATION_PROFILE,
    "Rock": ROCK_RESTORATION_PROFILE,
    "Pop": POP_RESTORATION_PROFILE,
    "Blues": BLUES_RESTORATION_PROFILE,
    "Soul/R&B": SOUL_RNB_RESTORATION_PROFILE,
    "Country": COUNTRY_RESTORATION_PROFILE,
    "Folk": FOLK_RESTORATION_PROFILE,
    "Funk": FUNK_RESTORATION_PROFILE,
    "Electronic": ELECTRONIC_RESTORATION_PROFILE,
    "Hip-Hop": HIPHOP_RESTORATION_PROFILE,
    "Metal": METAL_RESTORATION_PROFILE,
    "Latin": LATIN_RESTORATION_PROFILE,
    "Gospel": GOSPEL_RESTORATION_PROFILE,
    "Reggae": REGGAE_RESTORATION_PROFILE,
    "Ambient": AMBIENT_RESTORATION_PROFILE,
    "World": WORLD_RESTORATION_PROFILE,
}


def get_restoration_profile(subgenre: str = "unknown") -> dict:
    """Gibt das Restaurierungsprofil für ein Genre/Subgenre zurück.

    Unterstützte Genre-Label (exakt wie GermanSchlagerClassifier.genre_label):
        'Schlager', 'Walzer', 'Marsch', 'Disco-Schlager', 'Volksmusik',
        'Jazz', 'Klassik', 'Oper', 'Rock',
        'Pop', 'Blues', 'Soul/R&B', 'Country', 'Folk',
        'Funk', 'Electronic', 'Hip-Hop', 'Metal',
        'Latin', 'Gospel', 'Reggae'
    Unterstützte Subgenre-Keys (SCHLAGER_SUBGENRE_EXTENSIONS):
        'schunkel', 'walzer', 'marsch', 'discoschlager', 'schlager_1950s',
        'schlager_modern', 'volksmusik'

    Args:
        subgenre: Genre-Label oder Subgenre-Key (Groß-/Kleinschreibung egal).

    Returns:
        Profil-Dict; leeres Dict wenn unbekannt.
    """
    key = subgenre.strip().lower()
    # Genre-Label → kanonisches Profil
    label_map: dict[str, dict] = {
        # Schlager-Varianten
        "schlager": SCHLAGER_RESTORATION_PROFILE,
        "walzer": {**SCHLAGER_RESTORATION_PROFILE, **_SUBGENRE_EXTENSIONS.get("walzer", {})},
        "marsch": {**SCHLAGER_RESTORATION_PROFILE, **_SUBGENRE_EXTENSIONS.get("marsch", {})},
        "disco-schlager": {**SCHLAGER_RESTORATION_PROFILE, **_SUBGENRE_EXTENSIONS.get("discoschlager", {})},
        "discoschlager": {**SCHLAGER_RESTORATION_PROFILE, **_SUBGENRE_EXTENSIONS.get("discoschlager", {})},
        "volksmusik": {**SCHLAGER_RESTORATION_PROFILE, **_SUBGENRE_EXTENSIONS.get("volksmusik", {})},
        "schlager_1950s": {**SCHLAGER_RESTORATION_PROFILE, **_SUBGENRE_EXTENSIONS.get("schlager_1950s", {})},
        "schlager_modern": {**SCHLAGER_RESTORATION_PROFILE, **_SUBGENRE_EXTENSIONS.get("schlager_modern", {})},
        # Klassische Genres
        "jazz": JAZZ_RESTORATION_PROFILE,
        "klassik": KLASSIK_RESTORATION_PROFILE,
        "oper": OPER_RESTORATION_PROFILE,
        "rock": ROCK_RESTORATION_PROFILE,
        # Neue Genres
        "pop": POP_RESTORATION_PROFILE,
        "blues": BLUES_RESTORATION_PROFILE,
        "soul/r&b": SOUL_RNB_RESTORATION_PROFILE,
        "soul_rnb": SOUL_RNB_RESTORATION_PROFILE,
        "soul": SOUL_RNB_RESTORATION_PROFILE,
        "r&b": SOUL_RNB_RESTORATION_PROFILE,
        "rnb": SOUL_RNB_RESTORATION_PROFILE,
        "country": COUNTRY_RESTORATION_PROFILE,
        "folk": FOLK_RESTORATION_PROFILE,
        "funk": FUNK_RESTORATION_PROFILE,
        "electronic": ELECTRONIC_RESTORATION_PROFILE,
        "dance": ELECTRONIC_RESTORATION_PROFILE,
        "hip-hop": HIPHOP_RESTORATION_PROFILE,
        "hiphop": HIPHOP_RESTORATION_PROFILE,
        "hip hop": HIPHOP_RESTORATION_PROFILE,
        "rap": HIPHOP_RESTORATION_PROFILE,
        "metal": METAL_RESTORATION_PROFILE,
        "latin": LATIN_RESTORATION_PROFILE,
        "gospel": GOSPEL_RESTORATION_PROFILE,
        "reggae": REGGAE_RESTORATION_PROFILE,
        "ambient": AMBIENT_RESTORATION_PROFILE,
        "world": WORLD_RESTORATION_PROFILE,
        "world music": WORLD_RESTORATION_PROFILE,
    }
    return dict(label_map.get(key, {}))  # leere Kopie wenn unbekannt


# ---- Thread-sicherer Singleton (Double-Checked Locking, §3.2) ----
_instance: GermanSchlagerClassifier | None = None
_lock = threading.Lock()


def get_genre_classifier() -> GermanSchlagerClassifier:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GermanSchlagerClassifier()
    return _instance


def classify_genre(audio: np.ndarray, sr: int) -> SchlagerClassificationResult:
    """Convenience-Wrapper für Genre-Klassifikation."""
    return get_genre_classifier().classify(audio, sr)
