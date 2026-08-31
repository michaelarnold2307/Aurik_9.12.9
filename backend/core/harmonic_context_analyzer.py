"""
backend/core/harmonic_context_analyzer.py — HarmonicContextAnalyzer (Aurik 10.0.0 §HCA-1)
======================================================================================
Chord-progression and harmonic-density analysis for NR-gate awareness.

A world-class human engineer always knows the harmonic language of a recording
before touching any processing.  This module gives Aurik the same awareness:

    - Chroma-CQT chord template matching (24 major/minor chords, KS key algorithm)
    - Per-(frequency × time) harmonic mask — frequency bins that carry *real* harmonic
      content should not be gated by noise reduction
    - Harmonic density per frame — more complex polyphony → gentler NR gate
    - Modulation detection — key-change boundaries are protected zones

Integration (UV3 §HCA-1):
    Called after VocalFocusAnalyzer, before GoalApplicabilityFilter.
    Injects into _restoration_context:
        "harmonic_context": HarmonicContextResult.to_dict()
        "harmonic_mask": np.ndarray (F × T)  — used by NR phases
        "harmonic_density": np.ndarray (T,)  — 0..1 polyphonic complexity per frame

Scientific references:
    - Krumhansl & Schmuckler (1990): key-finding algorithm
    - Foote (1999): self-similarity matrix for musical structure
    - Müller (2015): FMP §5.3 chroma features
    - Cohen (2004): OMLSA — harmonic_mask guides NR gain floor
"""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass, field

import numpy as np

from backend.core.audio_utils import safe_to_mono

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chord templates (12-dim chroma, normalised to unit L2 norm)
# ---------------------------------------------------------------------------

# Major chord: root, M3, P5
_MAJ_INTERVALS = [0, 4, 7]
# Minor chord: root, m3, P5
_MIN_INTERVALS = [0, 3, 7]

_CHORD_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_chord_templates() -> tuple[np.ndarray, list[str]]:
    """Gibt (24 × 12) template matrix and chord-name list zurück."""
    templates: list[np.ndarray] = []
    names: list[str] = []
    for root in range(12):
        # Major
        tmpl = np.zeros(12, dtype=np.float32)
        for iv in _MAJ_INTERVALS:
            tmpl[(root + iv) % 12] = 1.0
        tmpl /= np.linalg.norm(tmpl) + 1e-8
        templates.append(tmpl)
        names.append(f"{_CHORD_ROOTS[root]}maj")
        # Minor
        tmpl = np.zeros(12, dtype=np.float32)
        for iv in _MIN_INTERVALS:
            tmpl[(root + iv) % 12] = 1.0
        tmpl /= np.linalg.norm(tmpl) + 1e-8
        templates.append(tmpl)
        names.append(f"{_CHORD_ROOTS[root]}min")
    return np.stack(templates, axis=0), names  # (24, 12)


_CHORD_TEMPLATES, _CHORD_NAMES = _build_chord_templates()

# Krumhansl-Schmuckler major/minor profiles (normalised)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
_KS_MAJOR = _KS_MAJOR / _KS_MAJOR.sum()
_KS_MINOR = _KS_MINOR / _KS_MINOR.sum()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HarmonicContextResult:
    """Output of HarmonicContextAnalyzer.analyze()."""

    # Chord sequence (one label per analysis frame)
    chord_sequence: list[str] = field(default_factory=list)
    # Per-frame chord confidence (0..1)
    chord_confidence: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    # Estimated global key, e.g. "A_min"
    key_root: str = "C"
    key_mode: str = "maj"
    # Frame positions (in analysis frames) where a key/harmonic change is detected
    modulation_frames: list[int] = field(default_factory=list)
    # Per-frame harmonic density (0=single note / silence, 1=rich polyphony)
    harmonic_density: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    # Per-bin × per-frame harmonic mask (shape: n_fft_bins × T)
    # True where the spectrum likely carries real harmonic energy
    harmonic_mask: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=bool))
    # Sample rate used during analysis
    sr: int = 48000
    # Hop length in samples (for sample-accurate frame→sample conversion)
    hop_length: int = 512
    # Overall confidence of the harmonic analysis (0..1)
    analysis_confidence: float = 0.0

    def frame_to_sample(self, frame_idx: int) -> int:
        """Konvertiert analysis frame index to audio sample index."""
        return int(frame_idx * self.hop_length)

    def sample_to_frame(self, sample_idx: int) -> int:
        """Konvertiert audio sample index to nearest analysis frame index."""
        return int(round(sample_idx / max(1, self.hop_length)))

    def to_dict(self) -> dict:
        """Gibt JSON-serializable metadata for UV3 restoration context zurück."""
        return {
            "chord_sequence": self.chord_sequence,
            "chord_confidence": self.chord_confidence.astype(float).tolist(),
            "key": f"{self.key_root}_{self.key_mode}",
            "key_root": self.key_root,
            "key_mode": self.key_mode,
            "modulation_frames": self.modulation_frames,
            "harmonic_density": self.harmonic_density.astype(float).tolist(),
            "harmonic_mask_shape": list(self.harmonic_mask.shape),
            "harmonic_mask_protected_ratio": float(np.mean(self.harmonic_mask)) if self.harmonic_mask.size else 0.0,
            "n_frames": len(self.chord_sequence),
            "analysis_confidence": float(self.analysis_confidence),
            "hop_length": self.hop_length,
            "sr": self.sr,
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

# Performance budget (§0k): max analysis duration
_MAX_ANALYZE_S: float = 300.0  # 5 min upper cap


class HarmonicContextAnalyzer:
    """Chord-progression and harmonic-density analyzer (§HCA-1).

    All operations are CPU-only and use only numpy + scipy + (optionally)
    librosa CQT for chroma extraction.  Falls back to STFT chroma if librosa
    is unavailable.
    """

    # Chroma analysis parameters
    HOP_LENGTH: int = 512  # ~10 ms @ 48 kHz
    N_CHROMA: int = 12
    # Harmonic mask parameters
    N_FFT: int = 2048  # bins for the harmonic mask STFT
    MASK_SMOOTHING_FRAMES: int = 3  # temporal smoothing of harmonic mask
    # Modulation detection: cosine-distance threshold between successive chord vectors
    MODULATION_THRESHOLD: float = 0.45
    # Minimum frames between modulation events (anti-chatter)
    MODULATION_MIN_FRAMES: int = 16  # ~160 ms @ 10 ms hop

    def analyze(self, audio: np.ndarray, sr: int) -> HarmonicContextResult:
        """Full harmonic context analysis.

        Args:
            audio: mono or stereo float32 array in either (samples, channels) or (channels, samples) layout
            sr: sample rate (expected 48000 Hz)

        Returns:
            HarmonicContextResult with chord_sequence, harmonic_mask, density.
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        # Collapse to mono
        mono = np.asarray(safe_to_mono(audio), dtype=np.float32)
        n = mono.shape[0]
        if n == 0:
            return HarmonicContextResult(sr=sr, hop_length=self.HOP_LENGTH)

        # Cap analysis length for performance budget
        max_samples = int(_MAX_ANALYZE_S * sr)
        if n > max_samples:
            mono = mono[:max_samples]

        # 1. Chroma extraction
        chroma = self._compute_chroma(mono, sr)  # (12, T)
        n_frames = chroma.shape[1]
        if n_frames == 0:
            return HarmonicContextResult(sr=sr, hop_length=self.HOP_LENGTH)

        # 2. Chord sequence via template matching
        chord_ids, chord_conf = self._match_chords(chroma)  # (T,), (T,)
        chord_sequence = [_CHORD_NAMES[c] for c in chord_ids]

        # 3. Key estimation (Krumhansl-Schmuckler)
        key_root, key_mode = self._estimate_key(chroma)

        # 4. Modulation detection
        modulation_frames = self._detect_modulations(chroma)

        # 5. Harmonic density per frame (number of active chroma bins / 12)
        chroma_norm = np.clip(chroma, 0.0, None)
        threshold = np.max(chroma_norm, axis=0, keepdims=True) * 0.25
        active_bins = (chroma_norm > threshold).sum(axis=0)  # (T,)
        harmonic_density = np.clip(active_bins.astype(np.float32) / 12.0, 0.0, 1.0)

        # 6. Harmonic mask in frequency × time space
        harmonic_mask = self._build_harmonic_mask(mono, sr, chroma, chord_ids)

        # 7. Analysis confidence: based on mean chord-match quality
        analysis_confidence = float(np.clip(chord_conf.mean() * 1.5, 0.0, 1.0))

        return HarmonicContextResult(
            chord_sequence=chord_sequence,
            chord_confidence=chord_conf,
            key_root=key_root,
            key_mode=key_mode,
            modulation_frames=modulation_frames,
            harmonic_density=harmonic_density,
            harmonic_mask=harmonic_mask,
            sr=sr,
            hop_length=self.HOP_LENGTH,
            analysis_confidence=analysis_confidence,
        )

    # ------------------------------------------------------------------
    # Chroma extraction
    # ------------------------------------------------------------------

    def _compute_chroma(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """Berechnet 12-dimensional chroma features (STFT-based, no librosa dep)."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
                return librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=self.HOP_LENGTH, bins_per_octave=36).astype(  # type: ignore[no-any-return]
                    np.float32
                )
        except Exception as e:
            logger.warning("harmonic_context_analyzer.py::_berechnen_chroma Ersatzpfad: %s", e)
        # Fallback: STFT chroma
        return self._stft_chroma(mono, sr)

    def _stft_chroma(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """STFT-based chroma fallback (no librosa required)."""
        from scipy.signal import stft  # pylint: disable=import-outside-toplevel

        n_fft = min(4096, max(64, int(mono.shape[0])))
        hop = self.HOP_LENGTH
        noverlap = min(max(0, n_fft - hop), n_fft - 1)
        if len(mono) < n_fft:
            return  # type: ignore[return-value]
        _, _, Zxx = stft(mono, sr, nperseg=n_fft, noverlap=noverlap, boundary="even")
        mag = np.abs(Zxx).astype(np.float32)  # (F, T)
        n_freqs, n_frames = mag.shape

        # Map each FFT bin to a chroma class
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
        chroma = np.zeros((12, n_frames), dtype=np.float32)
        for k, f in enumerate(freqs[:n_freqs]):
            if f < 27.5:  # below A0
                continue
            pitch_class = int(round(12 * np.log2(f / 440.0 + 1e-12))) % 12
            chroma[pitch_class] += mag[k]

        # Normalise each frame
        col_max = np.max(chroma, axis=0, keepdims=True)
        chroma = chroma / (col_max + 1e-8)
        return chroma  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Chord matching
    # ------------------------------------------------------------------

    def _match_chords(self, chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gibt per-frame chord index (0..23) and confidence (0..1) zurück."""
        # chroma: (12, T); templates: (24, 12)
        # Normalise chroma per frame
        col_norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
        chroma_n = chroma / col_norm  # (12, T)
        # Cosine similarity: (24, T)
        similarity = _CHORD_TEMPLATES @ chroma_n  # (24, T)
        chord_ids = similarity.argmax(axis=0).astype(np.int32)  # type: ignore[arg-type]  # (T,)  # §V5 (copilot-instructions.md) Dither applied at export level
        chord_conf = similarity.max(axis=0).astype(np.float32)
        chord_conf = np.clip(chord_conf, 0.0, 1.0)
        return chord_ids, chord_conf

    # ------------------------------------------------------------------
    # Key estimation (Krumhansl-Schmuckler)
    # ------------------------------------------------------------------

    def _estimate_key(self, chroma: np.ndarray) -> tuple[str, str]:
        """Schätzt global key using KS correlation profiles."""
        # Global chroma histogram
        pc = chroma.mean(axis=1)  # (12,)
        pc = pc / (pc.sum() + 1e-8)

        best_r = -np.inf
        best_root = 0
        best_mode = "maj"
        for root in range(12):
            # Rotate profiles to align with this root
            ks_maj = np.roll(_KS_MAJOR, root)
            ks_min = np.roll(_KS_MINOR, root)
            # Guarded dot-product correlation (NaN-safe, avoids np.corrcoef)
            pc_c = pc - pc.mean()
            maj_c = ks_maj - ks_maj.mean()
            min_c = ks_min - ks_min.mean()
            denom_maj = float(np.sqrt((pc_c**2).sum() * (maj_c**2).sum()))
            denom_min = float(np.sqrt((pc_c**2).sum() * (min_c**2).sum()))
            r_maj = float(np.dot(pc_c, maj_c) / denom_maj) if denom_maj > 1e-10 else 0.0
            r_min = float(np.dot(pc_c, min_c) / denom_min) if denom_min > 1e-10 else 0.0
            if r_maj > best_r:
                best_r, best_root, best_mode = r_maj, root, "maj"
            if r_min > best_r:
                best_r, best_root, best_mode = r_min, root, "min"

        return _CHORD_ROOTS[best_root], best_mode

    # ------------------------------------------------------------------
    # Modulation detection
    # ------------------------------------------------------------------

    def _detect_modulations(self, chroma: np.ndarray) -> list[int]:
        """Erkennt harmonic modulation points using cosine-distance of chroma vectors."""
        n_frames = chroma.shape[1]
        modulations: list[int] = []
        last_mod = -self.MODULATION_MIN_FRAMES

        for t in range(1, n_frames):
            v_prev = chroma[:, t - 1]
            v_curr = chroma[:, t]
            norm_p = np.linalg.norm(v_prev) + 1e-8
            norm_c = np.linalg.norm(v_curr) + 1e-8
            cosine = float(np.dot(v_prev / norm_p, v_curr / norm_c))
            dist = 1.0 - cosine  # 0=identical, 2=opposite
            if dist > self.MODULATION_THRESHOLD and (t - last_mod) >= self.MODULATION_MIN_FRAMES:
                modulations.append(t)
                last_mod = t

        return modulations

    # ------------------------------------------------------------------
    # Harmonic mask
    # ------------------------------------------------------------------

    def _build_harmonic_mask(
        self,
        mono: np.ndarray,
        sr: int,
        chroma: np.ndarray,
        chord_ids: np.ndarray,
    ) -> np.ndarray:
        """Erstellt a (n_fft_bins × T_mask) boolean mask marking harmonic content.

        For each analysis frame we know the dominant chord → we know which
        frequencies are fundamentals + overtones of that chord's notes.
        These bins are protected from NR gating.

        Returns:
            mask: (n_fft_bins, T_mask) bool — True = protect from NR
        """
        from scipy.signal import stft  # pylint: disable=import-outside-toplevel

        # §Fix: scipy.signal.stft silently caps nperseg to len(mono) when the
        # signal is shorter than N_FFT (no error raised) — using the static
        # self.N_FFT for `freqs`/pitch-class computation would then mismatch
        # the actual Zxx bin count. Cap n_fft up-front so both stay consistent.
        n_fft = min(self.N_FFT, max(1, int(mono.shape[0])))
        hop = self.HOP_LENGTH
        _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103
        try:
            _, _, Zxx = stft(mono, sr, nperseg=n_fft, noverlap=_noverlap, boundary="even")
        except Exception:
            # Fallback: full protect (never gate anything)
            n_fft_bins = n_fft // 2 + 1
            n_frames = chroma.shape[1]
            return np.ones((n_fft_bins, n_frames), dtype=bool)  # type: ignore[no-any-return]

        n_fft_bins, n_stft_frames = Zxx.shape
        mag = np.abs(Zxx).astype(np.float32)
        # Derive freqs from the actual bin count returned by stft (not the
        # static self.N_FFT) so it always matches n_fft_bins exactly, even if
        # scipy internally adjusted nperseg further (e.g. odd-length signals).
        _actual_n_fft = 2 * (n_fft_bins - 1)
        freqs = np.fft.rfftfreq(_actual_n_fft, d=1.0 / sr).astype(np.float32)

        # Resample chord_ids to match stft frame count (nearest-neighbour)
        t_chroma = np.arange(len(chord_ids))
        t_stft = np.linspace(0.0, len(chord_ids) - 1, n_stft_frames)
        chord_stft = np.round(np.interp(t_stft, t_chroma, chord_ids.astype(float))).astype(int)
        chord_stft = np.clip(chord_stft, 0, len(_CHORD_NAMES) - 1)

        # For each chord: which pitch classes are active?
        chord_pitch_classes = np.zeros((len(_CHORD_NAMES), 12), dtype=bool)
        for ci, _ in enumerate(_CHORD_NAMES):
            root = ci // 2
            intervals = _MAJ_INTERVALS if (ci % 2 == 0) else _MIN_INTERVALS
            for iv in intervals:
                chord_pitch_classes[ci, (root + iv) % 12] = True

        # Pre-compute MIDI pitch class for each FFT bin
        with np.errstate(divide="ignore", invalid="ignore"):
            midi_bins = np.where(
                freqs > 0.0,
                69.0 + 12.0 * np.log2(freqs / 440.0 + 1e-12),
                -999.0,
            ).astype(np.float32)
        pitch_class_bins = np.round(midi_bins).astype(int) % 12

        # Vectorised mask — pitch-class matching implicitly covers octave-identical
        # overtones (2f, 4f, 8f share pitch class with the root); the P5 (3rd harmonic,
        # +7 st) and M3 (5th harmonic, +4 st) are already in every major/minor chord
        # template; the 7th harmonic (m7, +10 st) is caught by the 75th-percentile
        # energy gate below.
        active_pcs_all = chord_pitch_classes[chord_stft]  # (n_stft_frames, 12)
        bin_active = active_pcs_all[:, pitch_class_bins].T  # (n_fft_bins, n_stft_frames)
        energy_threshold = np.percentile(mag, 75.0, axis=0, keepdims=True)  # (1, n_stft_frames)
        energy_active = mag >= energy_threshold  # (n_fft_bins, n_stft_frames)
        mask = bin_active | energy_active

        # Temporal smoothing: a bin is protected if it was active in any of
        # MASK_SMOOTHING_FRAMES surrounding frames
        from scipy.ndimage import uniform_filter1d  # pylint: disable=import-outside-toplevel

        mask_f = mask.astype(np.float32)
        mask_f = uniform_filter1d(mask_f, size=self.MASK_SMOOTHING_FRAMES, axis=1)
        mask = np.asarray(mask_f > 0.0)

        return mask  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: HarmonicContextAnalyzer | None = None
_lock = threading.Lock()


def get_harmonic_context_analyzer() -> HarmonicContextAnalyzer:
    """Thread-safe singleton (double-checked locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HarmonicContextAnalyzer()
    return _instance


__all__ = [
    "HarmonicContextResult",
    "HarmonicContextAnalyzer",
    "get_harmonic_context_analyzer",
]
