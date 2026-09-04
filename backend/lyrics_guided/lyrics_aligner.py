"""
Lyrics Aligner for Lyrics-Guided Vocal Enhancement

Aligns lyrics to audio using Whisper (ASR) + Montreal Forced Aligner (phoneme-level).
Provides word-level and phoneme-level timestamps for content-aware processing.

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-11
"""

import logging
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class PhonemeAlignment:
    """single phoneme with timing."""

    phoneme: str
    """Phoneme symbol (IPA or ARPABET)."""

    start_time: float
    """Start time in seconds."""

    end_time: float
    """End time in seconds."""

    confidence: float
    """Alignment confidence (0.0-1.0)."""

    phoneme_type: str
    """Type: vowel, consonant, sibilant, nasal, glide, etc."""


@dataclass
class WordAlignment:
    """single word with timing and phonemes."""

    word: str
    """Word text."""

    start_time: float
    """Start time in seconds."""

    end_time: float
    """End time in seconds."""

    confidence: float
    """Recognition confidence (0.0-1.0)."""

    phonemes: list[PhonemeAlignment]
    """Phoneme-level breakdown."""


@dataclass
class LyricsAlignment:
    """Complete lyrics alignment result."""

    text: str
    """Full transcript."""

    language: str
    """Detected language code."""

    words: list[WordAlignment]
    """Word-level alignments."""

    has_vocals: bool
    """Whether vocals were detected."""

    vocal_segments: list[tuple[float, float]]
    """List of (start, end) times with vocals."""

    instrumental_segments: list[tuple[float, float]]
    """List of (start, end) times without vocals."""


class LyricsAligner:
    """
    Lyrics alignment using Whisper (ASR) + Montreal Forced Aligner (phonemes).

    Features:
    - Multi-language support (99 languages via Whisper)
    - Word-level timestamps
    - Phoneme-level timestamps (via MFA)
    - Vocal/instrumental segmentation
    """

    # Phoneme type classification (ARPABET symbols)
    PHONEME_TYPES = {
        # Vowels
        "vowel": ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"],
        # Sibilants (harsh, need de-essing)
        "sibilant": ["S", "Z", "SH", "ZH"],
        # Fricatives (non-sibilant)
        "fricative": ["F", "V", "TH", "DH", "HH"],
        # Plosives (transient-heavy)
        "plosive": ["P", "B", "T", "D", "K", "G"],
        # Nasals
        "nasal": ["M", "N", "NG"],
        # Liquids
        "liquid": ["L", "R"],
        # Glides
        "glide": ["W", "Y"],
    }

    # MFA models for different languages
    MFA_MODELS = {
        "en": {"dictionary": "english_us_arpa", "acoustic": "english_us_arpa", "name": "English (US)"},
        "de": {"dictionary": "german_mfa", "acoustic": "german_mfa", "name": "German"},
        "fr": {"dictionary": "french_mfa", "acoustic": "french_mfa", "name": "French"},
        "es": {"dictionary": "spanish_mfa", "acoustic": "spanish_mfa", "name": "Spanish"},
        "it": {"dictionary": "italian_mfa", "acoustic": "italian_mfa", "name": "Italian"},
        # Add more languages as needed
    }

    LEGACY_ENABLE_ENV = "AURIK_ENABLE_LEGACY_LYRICS"

    def __init__(
        self,
        use_whisper: bool = True,
        use_mfa: bool = True,
        whisper_model: str = "large-v3",
        language: str | None = None,
    ):
        """
        Initialisiert lyrics aligner.

        Args:
            use_whisper: Whether to use Whisper for ASR
            use_mfa: Whether to use Montreal Forced Aligner
            whisper_model: Whisper model size (tiny, base, small, medium, large, large-v3)
            language: Force specific language (None = auto-detect)
                      Supported: 'en' (English), 'de' (German), 'fr' (French), 'es' (Spanish)
        """
        self.use_whisper = use_whisper
        self.use_mfa = use_mfa
        self.whisper_model = whisper_model
        self.language = language
        self._legacy_enabled = os.getenv(self.LEGACY_ENABLE_ENV, "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        # Production policy: legacy lyrics stack is opt-in for dev/research only.
        if not self._legacy_enabled:
            self.use_whisper = False
            self.use_mfa = False
            logger.info(
                "Lyrics Aligner: legacy path deaktiviert by policy (set %s=1 for dev only)",
                self.LEGACY_ENABLE_ENV,
            )

        self._whisper_available = False
        self._mfa_available = False
        self._mfa_models_available: set[Any] = set()  # Track which MFA models are installed

        self._initialize()

    def _initialize(self) -> None:
        """Initialisiert ASR and alignment systems."""
        if not self._legacy_enabled:
            return

        if self.use_whisper:
            self._check_whisper_availability()

        if self.use_mfa:
            self._check_mfa_availability()

    def _check_whisper_availability(self) -> None:
        """SOTA-Update (Rev. 2026-09-04): Lädt Whisper via faster-whisper oder openai-whisper."""
        self._whisper_model_obj = None
        self._whisper_available = False

        # Try faster-whisper first (3x faster, lower memory)
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            model_size = "medium" if self.whisper_model in {"large", "large-v3"} else self.whisper_model
            logger.info("Lyrics Aligner: Loading faster-whisper %s...", model_size)
            self._whisper_model_obj = WhisperModel(model_size, device="cpu")
            self._whisper_available = True
            logger.info("✅ Lyrics Aligner: faster-whisper verfuegbar (%s)", model_size)
        except Exception as e1:
            logger.debug("faster-whisper nicht verfügbar: %s", e1)

        # Fallback to openai-whisper
        if not self._whisper_available:
            try:
                import whisper  # type: ignore[import-untyped]

                model_size = "medium" if self.whisper_model in {"large", "large-v3"} else self.whisper_model
                logger.info("Lyrics Aligner: Loading openai-whisper %s...", model_size)
                self._whisper_model_obj = whisper.load_model(model_size)
                self._whisper_available = True
                logger.info("✅ Lyrics Aligner: openai-whisper verfuegbar (%s)", model_size)
            except Exception as e2:
                logger.warning("Lyrics Aligner: Whisper nicht verfügbar (%s) — VAD-Fallback aktiv", e2)

    def _check_mfa_availability(self) -> None:
        """Prüft if MFA is available and which models are installed."""
        if not self._legacy_enabled:
            self._mfa_available = False
            logger.info("Lyrics Aligner: MFA path deaktiviert by policy")
            return

        try:
            import subprocess

            result = subprocess.run(["which", "mfa"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self._mfa_available = True
                logger.info("✅ Lyrics Aligner: MFA verfuegbar")

                # Check which models are available
                self._check_mfa_models()
            else:
                logger.warning("⚠️ Lyrics Aligner: MFA not found")
        except Exception as e:
            logger.warning("MFA Pruefung fehlgeschlagen: %s", e)

    def _check_mfa_models(self) -> None:
        """Prüft which MFA acoustic models are installed."""
        try:
            import subprocess

            result = subprocess.run(["mfa", "model", "list", "acoustic"], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                output = result.stdout.lower()

                # Check each language's model
                for lang_code, models in self.MFA_MODELS.items():
                    model_name = models["acoustic"].lower()
                    if model_name in output:
                        self._mfa_models_available.add(lang_code)
                        logger.info("  ✅ MFA model for %s: %s", models["name"], models["acoustic"])

                if not self._mfa_models_available:
                    logger.warning("  ⚠️ No MFA acoustic models found. Install with:")
                    logger.warning("     mfa model download acoustic english_us_arpa")
                    logger.warning("     mfa model download acoustic german_mfa")

        except Exception as e:
            logger.warning("Could not Pruefung MFA models: %s", e)

    def align(
        self, audio: npt.NDArray[np.float32], sr: int, lyrics: str | None = None, language: str | None = None
    ) -> LyricsAlignment:
        """
        Align lyrics to audio.

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate
            lyrics: Optional known lyrics (if None, will transcribe)
            language: Optional language code ('en', 'de', 'fr', 'es', etc.)
                     If None, uses language from __init__ or auto-detect

        Returns:
            LyricsAlignment with word and phoneme timestamps
        """
        # Convert to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Determine language
        lang = language or self.language or "en"

        # 1. Transcribe with Whisper (if no lyrics provided)
        if lyrics is None:
            logger.info("🎤 Transcribing with Whisper (language: %s)...", lang)
            transcript, detected_lang, word_segments = self._whisper_transcribe(audio, sr)
            # Use detected language if auto-detection was used
            if not language and not self.language:
                lang = detected_lang if detected_lang != "unknown" else "en"
                logger.info("   erkannt language: %s", lang)
        else:
            logger.info("🎤 Using provided lyrics (language: %s)...", lang)
            transcript = lyrics
            word_segments = []  # Will need forced alignment

        # 2. Detect vocal segments
        logger.info("🎵 Detecting vocal segments...")
        vocal_segments, instrumental_segments = self._detect_vocal_segments(audio, sr, word_segments)

        # 3. Phoneme-level alignment (if MFA available)
        if self._mfa_available and len(word_segments) > 0:
            logger.info("🔤 Phoneme-level alignment with MFA (%s)...", lang)
            words = self._mfa_align(audio, sr, word_segments, lang)
        else:
            logger.info("🔤 Using word-level alignment only...")
            words = self._word_level_only(word_segments)

        has_vocals = len(vocal_segments) > 0

        return LyricsAlignment(
            text=transcript,
            language=lang,
            words=words,
            has_vocals=has_vocals,
            vocal_segments=vocal_segments,
            instrumental_segments=instrumental_segments,
        )

    def _whisper_transcribe(self, audio: npt.NDArray[np.float32], sr: int) -> tuple[str, str, list[dict]]:
        """Transcribe audio with Whisper (SOTA: faster-whisper oder openai-whisper)."""
        if self._whisper_available and self._whisper_model_obj is not None:
            try:
                # Normalize to float32 [-1, 1] → PCM16 for Whisper (Intermediate-Format, kein Audio-Dither nötig §V5)
                pcm16 = (audio * 32767.0).astype(np.int16)

                if "faster_whisper" in str(type(self._whisper_model_obj)):
                    # faster-whisper API
                    segments, info = self._whisper_model_obj.transcribe(
                        pcm16,
                        language=self.language or None,
                        task="transcribe",
                        beam_size=5,
                        temperature=[0.0],
                    )
                    detected_lang = getattr(info, "language", "unknown") or "unknown"
                    lang_name = getattr(info, "language_name", "") if hasattr(info, "language_name") else ""

                    word_segments = []
                    for seg in segments:
                        # faster-whisper returns segments with text and timestamps
                        start = float(seg.start)
                        end = float(seg.end)
                        text = str(seg.text).strip()
                        confidence = float(seg.no_ts_prob) if hasattr(seg, "no_ts_prob") else 0.8

                        for word in text.split():
                            word_segments.append({
                                "start": start,
                                "end": end,
                                "word": word,
                                "confidence": min(confidence, 0.95),
                            })

                    transcript = " ".join(seg.text.strip() for seg in segments)
                    return transcript, detected_lang, word_segments

                else:
                    # openai-whisper API
                    result = self._whisper_model_obj.transcribe(
                        pcm16,
                        language=self.language or None,
                        task="transcribe",
                        beam_size=5,
                        temperature=0.0,
                    )
                    detected_lang = result.get("language", "unknown")
                    text_segments = result.get("segments", [])

                    word_segments = []
                    for seg in text_segments:
                        start = float(seg["start"])
                        end = float(seg["end"])
                        text = str(seg["text"]).strip()

                        for word in text.split():
                            word_segments.append({
                                "start": start,
                                "end": end,
                                "word": word,
                                "confidence": 0.85,
                            })

                    transcript = result.get("text", "")
                    return transcript.strip(), detected_lang, word_segments

            except Exception as e:
                logger.warning("Whisper Transkription fehlgeschlagen (%s) — VAD-Fallback aktiv", e)
                self._whisper_available = False

        # Fallback to VAD-only transcription
        return self._fallback_transcribe(audio, sr)

    def _whisper_docker_inference(self, audio: npt.NDArray[np.float32], sr: int) -> tuple[str, str, list[dict]]:
        """Legacy method retained for API compatibility; Docker execution is disabled."""
        del audio, sr
        raise RuntimeError("Docker-based Whisper inference is disabled by policy")

    def _fallback_transcribe(self, audio: npt.NDArray[np.float32], sr: int) -> tuple[str, str, list[dict]]:
        """Fallback transcription (simple voice activity detection)."""
        # Simple VAD-based segmentation
        from scipy import signal

        # Compute envelope
        analytic = signal.hilbert(audio)
        analytic_arr = np.asarray(analytic, dtype=np.complex128)
        envelope = np.abs(analytic_arr)

        # Smooth
        window_size = int(sr * 0.05)  # 50ms
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Threshold
        threshold = np.mean(envelope_smooth) + 0.5 * np.std(envelope_smooth)
        is_voice = envelope_smooth > threshold

        # Find speech segments
        segments = []
        in_segment = False
        start = 0

        for i, val in enumerate(is_voice):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                end = i
                segments.append({"start": start / sr, "end": end / sr, "word": "[speech]", "confidence": 0.5})
                in_segment = False

        if in_segment:
            segments.append({"start": start / sr, "end": len(audio) / sr, "word": "[speech]", "confidence": 0.5})

        transcript = " ".join([seg["word"] for seg in segments])  # type: ignore[misc]
        language = "unknown"

        return transcript, language, segments

    def _detect_vocal_segments(
        self, audio: npt.NDArray[np.float32], sr: int, word_segments: list[dict]
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Erkennt vocal vs instrumental segments."""
        if len(word_segments) > 0:
            # Use word segments as vocal segments
            vocal_segs = [(seg["start"], seg["end"]) for seg in word_segments]

            # Merge nearby segments (< 0.5s gap)
            merged_vocals = []
            if len(vocal_segs) > 0:
                current_start, current_end = vocal_segs[0]

                for start, end in vocal_segs[1:]:
                    if start - current_end < 0.5:
                        current_end = end
                    else:
                        merged_vocals.append((current_start, current_end))
                        current_start, current_end = start, end

                merged_vocals.append((current_start, current_end))

            # Instrumental = gaps between vocals
            duration = len(audio) / sr
            instrumental_segs = []

            if len(merged_vocals) > 0:
                # Before first vocal
                if merged_vocals[0][0] > 1.0:
                    instrumental_segs.append((0.0, merged_vocals[0][0]))

                # Between vocals
                for i in range(len(merged_vocals) - 1):
                    gap_start = merged_vocals[i][1]
                    gap_end = merged_vocals[i + 1][0]
                    if gap_end - gap_start > 1.0:
                        instrumental_segs.append((gap_start, gap_end))

                # After last vocal
                if duration - merged_vocals[-1][1] > 1.0:
                    instrumental_segs.append((merged_vocals[-1][1], duration))

            return merged_vocals, instrumental_segs
        else:
            # No words detected → assume instrumental
            duration = len(audio) / sr
            return [], [(0.0, duration)]

    def _mfa_align(
        self, audio: npt.NDArray[np.float32], sr: int, word_segments: list[dict], language: str = "en"
    ) -> list[WordAlignment]:
        """Use MFA for phoneme-level alignment."""
        # Check if MFA model for this language is available
        if language not in self._mfa_models_available:
            logger.warning(
                f"MFA model for language '{language}' not verfuegbar. "
                f"Using word-level only. verfuegbar: {self._mfa_models_available}"
            )
            return self._word_level_only(word_segments)

        try:
            return self._mfa_alignment_process(audio, sr, word_segments, language)
        except Exception as e:
            logger.warning("MFA alignment fehlgeschlagen: %s. Using word-level only.", e)
            return self._word_level_only(word_segments)

    def _mfa_alignment_process(
        self, audio: npt.NDArray[np.float32], sr: int, word_segments: list[dict], language: str = "en"
    ) -> list[WordAlignment]:
        """Führt aus: Montreal Forced Aligner for phoneme-level timestamps."""
        import subprocess
        import tempfile

        import soundfile as sf

        # Get MFA models for this language
        if language not in self.MFA_MODELS:
            raise ValueError(f"Language '{language}' not supported for MFA. Supported: {list(self.MFA_MODELS.keys())}")

        models = self.MFA_MODELS[language]
        logger.info("laeuft MFA phoneme-level alignment for %s...", models["name"])

        # Create temporary directory for MFA
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save audio
            audio_path = f"{temp_dir}/audio.wav"
            sf.write(audio_path, audio, sr)

            # Create transcript file (TextGrid format expected by MFA)
            transcript = " ".join([seg["word"] for seg in word_segments])
            transcript_path = f"{temp_dir}/audio.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            # Run MFA align with language-specific models
            output_dir = f"{temp_dir}/output"
            cmd = [
                "mfa",
                "align",
                "--clean",
                "--single_speaker",
                temp_dir,
                models["dictionary"],  # Language-specific dictionary
                models["acoustic"],  # Language-specific acoustic model
                output_dir,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes

            if result.returncode != 0:
                raise RuntimeError(f"MFA failed: {result.stderr}")

            # Parse MFA output (TextGrid format)
            textgrid_path = f"{output_dir}/audio.TextGrid"
            if not os.path.exists(textgrid_path):
                raise FileNotFoundError("MFA output not found")

            # Parse TextGrid to extract phoneme alignments
            words = self._parse_textgrid(textgrid_path, word_segments)

            logger.info("✅ MFA aligned: %s words with phonemes", len(words))

            return words

    def _parse_textgrid(self, textgrid_path: str, word_segments: list[dict]) -> list[WordAlignment]:
        """Parst MFA-TextGrid-Ausgabe zur Extraktion von Phonem-Ausrichtungen."""
        try:
            textgrid_module = import_module("textgrid")
            tg = textgrid_module.TextGrid.fromFile(textgrid_path)
        except ImportError:
            logger.warning("textgrid not verfuegbar. Falling back to simplified TextGrid parser.")
            return self._parse_textgrid_simple(textgrid_path, word_segments)
        except Exception as e:
            logger.warning("TextGrid parsing fehlgeschlagen: %s. Using simplified parser.", e)
            return self._parse_textgrid_simple(textgrid_path, word_segments)

        words = []

        # MFA creates two tiers: 'words' and 'phones'
        word_tier = None
        phone_tier = None

        for tier in tg.tiers:
            if tier.name.lower() in ["words", "word"]:
                word_tier = tier
            elif tier.name.lower() in ["phones", "phone", "phonemes"]:
                phone_tier = tier

        if not word_tier or not phone_tier:
            raise ValueError("Required tiers not found in TextGrid")

        # Build word alignments with phonemes
        for word_interval in word_tier:
            if not word_interval.mark or word_interval.mark.strip() == "":
                continue

            word_text = word_interval.mark.strip()
            word_start = word_interval.minTime
            word_end = word_interval.maxTime

            # Find phonemes within this word's timespan
            phonemes = []
            for phone_interval in phone_tier:
                if not phone_interval.mark or phone_interval.mark.strip() == "":
                    continue

                phone_start = phone_interval.minTime
                phone_end = phone_interval.maxTime

                # Check if phoneme is within word boundaries
                if phone_start >= word_start and phone_end <= word_end + 0.01:  # Small tolerance
                    phoneme_symbol = phone_interval.mark.strip()
                    phoneme_type = self.get_phoneme_type(phoneme_symbol)

                    phonemes.append(
                        PhonemeAlignment(
                            phoneme=phoneme_symbol,
                            start_time=phone_start,
                            end_time=phone_end,
                            confidence=0.9,  # High confidence from MFA
                            phoneme_type=phoneme_type,
                        )
                    )

            # Match with original word segment to get confidence
            confidence = 0.8
            for orig_seg in word_segments:
                if abs(orig_seg["start"] - word_start) < 0.1:  # Match by timing
                    confidence = orig_seg.get("confidence", 0.8)
                    break

            words.append(
                WordAlignment(
                    word=word_text, start_time=word_start, end_time=word_end, confidence=confidence, phonemes=phonemes
                )
            )

        return words

    def _parse_textgrid_simple(self, textgrid_path: str, word_segments: list[dict]) -> list[WordAlignment]:
        """Einfacher TextGrid-Parser (Fallback wenn textgrid-Bibliothek nicht verfügbar)."""
        # Parse manually
        with open(textgrid_path) as f:
            f.read()

        # Very simple parsing - just extract intervals
        # This is a simplified version. Production should use proper TextGrid library.

        # For now, return word-level only
        logger.warning("Using simplified TextGrid parser - phoneme details may be limited")
        return self._word_level_only(word_segments)

    def _word_level_only(self, word_segments: list[dict]) -> list[WordAlignment]:
        """Erstellt WordAlignment without phoneme details."""
        words = []

        for seg in word_segments:
            word_text = seg["word"]
            start = seg["start"]
            end = seg["end"]
            conf = seg.get("confidence", 0.5)

            # Estimate phonemes (simple heuristic)
            phonemes = self._estimate_phonemes(word_text, start, end)

            words.append(
                WordAlignment(word=word_text, start_time=start, end_time=end, confidence=conf, phonemes=phonemes)
            )

        return words

    def _estimate_phonemes(self, word: str, start: float, end: float) -> list[PhonemeAlignment]:
        """Schätzt phonemes from word (very simple)."""
        # Simple: split word duration by letter count
        duration = end - start
        letters = len(word)

        if letters == 0:
            return []

        phoneme_duration = duration / letters
        phonemes = []

        for i, letter in enumerate(word.lower()):
            phoneme_start = start + i * phoneme_duration
            phoneme_end = phoneme_start + phoneme_duration

            # Classify phoneme type based on letter
            ptype = self._classify_letter(letter)

            phonemes.append(
                PhonemeAlignment(
                    phoneme=letter,
                    start_time=phoneme_start,
                    end_time=phoneme_end,
                    confidence=0.3,  # Low confidence (estimated)
                    phoneme_type=ptype,
                )
            )

        return phonemes

    def _classify_letter(self, letter: str) -> str:
        """Classify letter as vowel/consonant/etc."""
        vowels = "aeiou"
        sibilants = "sz"
        nasals = "mn"
        liquids = "lr"

        if letter in vowels:
            return "vowel"
        elif letter in sibilants:
            return "sibilant"
        elif letter in nasals:
            return "nasal"
        elif letter in liquids:
            return "liquid"
        else:
            return "consonant"

    def get_phoneme_type(self, phoneme: str) -> str:
        """Gibt zurück: phoneme type (vowel, consonant, etc.)."""
        phoneme_upper = phoneme.upper()

        for ptype, phonemes in self.PHONEME_TYPES.items():
            if phoneme_upper in phonemes:
                return ptype

        # Default: consonant
        return "consonant"


if __name__ == "__main__":
    # Demo
    logger.info(str("=" * 80))
    logger.info("AURIK Lyrics Aligner Demo")
    logger.info(str("=" * 80))

    # Create test audio with simulated vocals (5 seconds)
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Simulated vocal pattern: speech at 0.5-1.5s, 2.5-3.5s
    audio = np.zeros_like(t)

    # Vocal segment 1 (0.5-1.5s)
    mask1 = (t >= 0.5) & (t < 1.5)
    audio[mask1] = 0.3 * np.sin(2 * np.pi * 300 * t[mask1])  # Fundamental
    audio[mask1] += 0.1 * np.sin(2 * np.pi * 3000 * t[mask1])  # Formant

    # Vocal segment 2 (2.5-3.5s)
    mask2 = (t >= 2.5) & (t < 3.5)
    audio[mask2] = 0.4 * np.sin(2 * np.pi * 350 * t[mask2])
    audio[mask2] += 0.15 * np.sin(2 * np.pi * 2800 * t[mask2])

    # Background noise
    audio += 0.02 * np.random.randn(len(audio)).astype(np.float32)

    # Align
    logger.info("\n🎤 Aligning lyrics...")
    aligner = LyricsAligner(use_whisper=False, use_mfa=False)
    result = aligner.align(audio, sr, lyrics="Hello world testing")

    logger.info('\n📝 Transcript: "%s"', result.text)
    logger.info("🌍 Language: %s", result.language)
    logger.info("🎵 Has vocals: %s", result.has_vocals)

    logger.info("\n🎤 Vocal segments (%s):", len(result.vocal_segments))
    for start, end in result.vocal_segments:
        logger.info("   %.2fs - %.2fs (%.2fs)", start, end, end - start)

    logger.info("\n🎸 Instrumental segments (%s):", len(result.instrumental_segments))
    for start, end in result.instrumental_segments:
        logger.info("   %.2fs - %.2fs (%.2fs)", start, end, end - start)

    logger.info("\n📖 Words (%s):", len(result.words))
    for word in result.words:
        logger.info('   [%.2fs - %.2fs] "%s" (conf: %.1f)', word.start_time, word.end_time, word.word, word.confidence)
        logger.info("      Phonemes: %s", len(word.phonemes))
        for phoneme in word.phonemes[:3]:  # Show first 3
            logger.info(
                "        \u2022 %s (%s): %.2fs - %.2fs",
                phoneme.phoneme,
                phoneme.phoneme_type,
                phoneme.start_time,
                phoneme.end_time,
            )
        if len(word.phonemes) > 3:
            logger.info("        ... and %s more", len(word.phonemes) - 3)

    logger.info(str("\n" + "=" * 80))
    logger.info("✅ Lyrics Aligner Demo vollstaendig")
