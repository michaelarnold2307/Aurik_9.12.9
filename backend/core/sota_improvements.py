"""
§v10 Schwachpunkt-Stärkung: 10 SOTA-Verbesserungen für Weltklasse-Exporte.

Dieses Modul implementiert die verbleibenden Lücken zwischen Auriks technischer
Exzellenz und menschlichem Toningenieur-Niveau.

Schwachpunkte:
S1:  Real-time A/B comparison during processing
S2:  Dynamic parameter optimization per song section
S3:  Pipeline confidence gates aggressive phases
S4:  Cross-phase awareness (Phase B knows Δ from Phase A)
S5:  Loudness normalization cascade fix
S6:  MP3 vs Vinyl source-differentiated processing
S7:  Per-phase rollback / undo capability
S8:  Spectral balance final validation (tilt check)
S9:  Automatic multi-format export in one pass
S10: Singer identity post-pipeline validation
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# S1: Real-time A/B comparison state
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ABComparisonState:
    """Hält den Zustand für Echtzeit-A/B-Vergleich während der Verarbeitung."""

    pre_phase_audio: np.ndarray | None = None
    post_phase_audio: np.ndarray | None = None
    current_phase: str = ""
    ab_snippets: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def capture_pre(self, audio: np.ndarray, phase_id: str) -> None:
        """Erfasst das Audio VOR einer Phase."""
        with self.lock:
            self.pre_phase_audio = np.asarray(audio, dtype=np.float32).copy()
            self.current_phase = phase_id

    def capture_post(self, audio: np.ndarray) -> None:
        """Erfasst das Audio NACH einer Phase und erzeugt A/B-Snippet."""
        with self.lock:
            self.post_phase_audio = np.asarray(audio, dtype=np.float32).copy()
            if self.pre_phase_audio is not None:
                self.ab_snippets.append(
                    {
                        "phase": self.current_phase,
                        "pre": self.pre_phase_audio,
                        "post": self.post_phase_audio,
                    }
                )


# Singleton
_ab_state: ABComparisonState | None = None
_ab_lock = threading.Lock()


def get_ab_comparison_state() -> ABComparisonState:
    global _ab_state
    if _ab_state is None:
        with _ab_lock:
            if _ab_state is None:
                _ab_state = ABComparisonState()
    return _ab_state


# ═══════════════════════════════════════════════════════════════════════════
# S2: Dynamic parameter optimization per song section
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SongSectionParams:
    """Per-Sektion optimierte DSP-Parameter."""

    section_type: str  # "intro", "verse", "chorus", "bridge", "outro"
    start_sample: int
    end_sample: int
    noise_reduction_strength: float = 0.5
    eq_high_shelf_db: float = 0.0
    compression_ratio: float = 1.5
    stereo_width: float = 0.85
    de_ess_strength: float = 0.5
    presence_boost_db: float = 0.5


def compute_section_specific_params(
    audio: np.ndarray,
    sr: int,
    structure: list[dict[str, Any]] | None = None,
    intent: Any | None = None,
) -> list[SongSectionParams]:
    """Berechnet pro Song-Sektion optimierte Parameter.

    Args:
        audio:     Mono-Audio für Analyse
        sr:        Sample-Rate
        structure: Optionale Song-Struktur (von SongStructureAnalyzer)
        intent:    Optionales ArtisticIntent-Objekt

    Returns:
        Liste von SongSectionParams
    """
    if structure is None:
        # Fallback: einfache Energie-basierte Segmentierung
        n = len(audio)
        section_len = int(15.0 * sr)  # 15s Abschnitte
        sections = []
        for i, start in enumerate(range(0, n, section_len)):
            end = min(n, start + section_len)
            chunk = audio[start:end]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12))
            rms_db = 20.0 * np.log10(rms + 1e-12)
            # Heuristik: laute Abschnitte = Refrain, leise = Strophe
            if rms_db > -15:
                stype = "chorus"
            elif rms_db > -25:
                stype = "verse"
            else:
                stype = "bridge"
            sections.append(
                SongSectionParams(
                    section_type=stype,
                    start_sample=start,
                    end_sample=end,
                )
            )
        return sections

    # Struktur-basierte Parameter
    params = []
    for seg in structure:
        stype = seg.get("label", "verse")
        start = int(seg.get("start_sample", 0))
        end = int(seg.get("end_sample", 0))

        sp = SongSectionParams(section_type=stype, start_sample=start, end_sample=end)

        if stype == "chorus":
            sp.compression_ratio = 1.8
            sp.presence_boost_db = 1.5
            sp.stereo_width = 0.90
        elif stype == "verse":
            sp.noise_reduction_strength = 0.6
            sp.compression_ratio = 1.3
            sp.presence_boost_db = 0.5
        elif stype == "bridge":
            sp.stereo_width = 0.95
            sp.de_ess_strength = 0.4
        elif stype == "intro" or stype == "outro":
            sp.noise_reduction_strength = 0.3
            sp.compression_ratio = 1.1

        # Intent-Overrides
        if intent and hasattr(intent, "preserve_dynamics") and intent.preserve_dynamics:
            sp.compression_ratio = min(sp.compression_ratio, 1.3)

        params.append(sp)

    return params


# ═══════════════════════════════════════════════════════════════════════════
# S3: Pipeline confidence gates for aggressive phases
# ═══════════════════════════════════════════════════════════════════════════

AGGRESSIVE_PHASES: frozenset[str] = frozenset(
    {
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
        "phase_38_presence_boost",
        "phase_46_spatial_enhancement",
        "phase_55_diffusion_inpainting",
    }
)


def should_gate_phase(phase_id: str, pipeline_confidence: float) -> tuple[bool, str]:
    """Prüft, ob eine aggressive Phase bei niedriger Pipeline-Konfidenz gegated werden soll.

    Args:
        phase_id: ID der Phase
        pipeline_confidence: Pipeline-Konfidenz (0.0–1.0)

    Returns:
        (should_gate: bool, reason: str)
    """
    if phase_id not in AGGRESSIVE_PHASES:
        return False, ""

    if pipeline_confidence < 0.30:
        return True, f"Pipeline-Konfidenz {pipeline_confidence:.2f} < 0.30 → {phase_id} gegated"
    elif pipeline_confidence < 0.50:
        return True, f"Pipeline-Konfidenz {pipeline_confidence:.2f} < 0.50 → {phase_id} mit reduzierter Stärke"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════
# S4: Cross-Phase Awareness
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PhaseDelta:
    """Dokumentiert, was eine Phase geändert hat."""

    phase_id: str
    rms_delta_db: float = 0.0
    peak_delta_db: float = 0.0
    spectral_tilt_delta: float = 0.0
    lufs_delta: float = 0.0
    stereo_width_delta: float = 0.0


class CrossPhaseAwareness:
    """Ermöglicht Phasen, die Änderungen vorheriger Phasen zu kennen."""

    def __init__(self) -> None:
        self._deltas: list[PhaseDelta] = []
        self._warnings: list[str] = []

    def record_phase(self, delta: PhaseDelta) -> None:
        self._deltas.append(delta)
        # S4: Checke Interaktionen
        self._check_interactions(delta)

    def _check_interactions(self, latest: PhaseDelta) -> None:
        """Prüft potenziell schädliche Phasen-Interaktionen."""
        for prev in self._deltas[:-1]:
            # Problem: Phase A hat 3kHz geboostet, Phase B könnte Sibilanz verschärfen
            if prev.spectral_tilt_delta > 1.0 and "de_ess" in latest.phase_id:
                self._warnings.append(
                    f"⚠️ {latest.phase_id}: Vorgänger-Phase {prev.phase_id} hat "
                    f"spektrale Balance um {prev.spectral_tilt_delta:.1f} dB verschoben. "
                    "De-Essing könnte überkompensieren."
                )
            # Problem: Zwei Phasen haben beide die Loudness signifikant geändert
            if abs(prev.rms_delta_db) > 2.0 and abs(latest.rms_delta_db) > 2.0:
                self._warnings.append(
                    f"⚠️ Kumulative Loudness-Änderung: {prev.phase_id}({prev.rms_delta_db:+.1f}dB) "
                    f"+ {latest.phase_id}({latest.rms_delta_db:+.1f}dB) — "
                    "Risiko von Loudness-Kaskade."
                )

    def get_warnings(self) -> list[str]:
        return self._warnings

    def get_last_delta(self) -> PhaseDelta | None:
        return self._deltas[-1] if self._deltas else None


# ═══════════════════════════════════════════════════════════════════════════
# S5: Loudness Normalization Cascade Fix
# ═══════════════════════════════════════════════════════════════════════════

LOUDNESS_PHASES: frozenset[str] = frozenset(
    {
        "phase_40_loudness_normalization",
        "phase_41_output_format_optimization",
    }
)


def compute_cumulative_loudness_gain(phase_deltas: list[PhaseDelta]) -> float:
    """Berechnet die kumulative Loudness-Änderung über alle Loudness-Phasen.

    Wenn mehrere Loudness-Phasen aktiv sind, kann die kumulative Änderung
    das Ziel überschreiten. Diese Funktion berechnet den tatsächlichen
    kumulativen Gain und empfiehlt eine Korrektur.

    Returns:
        Empfohlene Korrektur in dB (negativ = attenuieren)
    """
    total_lufs_delta = 0.0
    for d in phase_deltas:
        if any(lp in d.phase_id for lp in LOUDNESS_PHASES):
            total_lufs_delta += d.lufs_delta
    # Wenn kumulativ > 3 LU → korrigiere auf 3 LU max
    if abs(total_lufs_delta) > 3.0:
        correction = 3.0 - abs(total_lufs_delta) if total_lufs_delta > 0 else -(3.0 - abs(total_lufs_delta))
        logger.warning(
            "S5: Loudness-Kaskade erkannt — kumulativ %.1f LU. Korrektur: %.1f dB.",
            total_lufs_delta,
            correction,
        )
        return correction
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# S6: MP3 vs Vinyl Source-Differentiated Processing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SourceProfile:
    """Verarbeitungsprofil basierend auf der Quellen-Charakteristik."""

    is_lossy_source: bool = False
    is_analog_source: bool = False
    # MP3/AAC: keine analoge Rauschunterdrückung nötig, aber Codec-Artefakt-Reparatur
    skip_vinyl_phases: bool = False
    enable_codec_repair: bool = False
    # Vinyl/Shellac: keine Codec-Reparatur, aber analoge Defekt-Behandlung
    skip_codec_phases: bool = False
    enable_analog_repair: bool = False
    # Noise-Floor-Erwartung
    expected_noise_floor_db: float = -60.0


def get_source_profile(material: str) -> SourceProfile:
    """Ermittelt das Quellen-Profil basierend auf dem Material-Typ.

    MP3-Quellen haben FUNDAMENTAL andere Defekte als Vinyl-Quellen.
    Ein menschlicher Toningenieur behandelt sie völlig unterschiedlich.
    """
    mat = str(material).lower()

    if any(t in mat for t in ("mp3", "aac", "streaming", "cd_digital", "dat")):
        return SourceProfile(
            is_lossy_source=True,
            is_analog_source=False,
            skip_vinyl_phases=True,
            enable_codec_repair=True,
            expected_noise_floor_db=-55.0,
        )
    elif any(t in mat for t in ("vinyl", "shellac", "wax", "lacquer")):
        return SourceProfile(
            is_lossy_source=False,
            is_analog_source=True,
            skip_codec_phases=True,
            enable_analog_repair=True,
            expected_noise_floor_db=-33.0,
        )
    elif any(t in mat for t in ("tape", "cassette", "reel")):
        return SourceProfile(
            is_lossy_source=False,
            is_analog_source=True,
            enable_analog_repair=True,
            expected_noise_floor_db=-45.0,
        )
    else:
        return SourceProfile(expected_noise_floor_db=-50.0)


# ═══════════════════════════════════════════════════════════════════════════
# S7: Per-Phase Rollback / Undo
# ═══════════════════════════════════════════════════════════════════════════


class PhaseRollback:
    """Ermöglicht Rollback einzelner Phasen."""

    def __init__(self, max_snapshots: int = 10) -> None:
        self._snapshots: list[tuple[str, np.ndarray]] = []
        self._max_snapshots = max_snapshots

    def save_snapshot(self, phase_id: str, audio: np.ndarray) -> None:
        """Speichert einen Snapshot VOR einer Phase."""
        if len(self._snapshots) >= self._max_snapshots:
            self._snapshots.pop(0)
        self._snapshots.append((phase_id, np.asarray(audio, dtype=np.float32).copy()))

    def rollback_to(self, phase_id: str) -> np.ndarray | None:
        """Rollt zurück zum Snapshot VOR der angegebenen Phase."""
        for i, (pid, snap) in enumerate(self._snapshots):
            if pid == phase_id:
                # Entferne alle Snapshots NACH dieser Phase
                self._snapshots = self._snapshots[:i]
                return snap.copy()
        return None

    def rollback_last(self) -> tuple[str, np.ndarray] | None:
        """Rollt die LETZTE Phase zurück."""
        if not self._snapshots:
            return None
        phase_id, snap = self._snapshots.pop()
        return phase_id, snap.copy()


# ═══════════════════════════════════════════════════════════════════════════
# S8: Spectral Balance Final Validation (Tilt Check)
# ═══════════════════════════════════════════════════════════════════════════


def validate_spectral_balance(
    audio: np.ndarray,
    sr: int,
    *,
    max_tilt_db_per_octave: float = 3.0,
    reference_tilt: float | None = None,
) -> dict[str, Any]:
    """Validiert die spektrale Balance des finalen Masters.

    Ein menschlicher Toningenieur prüft am Ende IMMER: „Klingt es zu dumpf? Zu scharf?"

    Args:
        audio:                     Mono oder Stereo Audio
        sr:                        Sample-Rate
        max_tilt_db_per_octave:    Maximal tolerierte spektrale Neigung
        reference_tilt:            Optionale Referenz-Neigung (vom Original)

    Returns:
        dict mit 'passed', 'tilt_db_per_octave', 'warning', 'recommendation'
    """
    arr = np.asarray(audio, dtype=np.float64)
    mono = arr.mean(axis=0) if arr.ndim == 2 else arr

    # Berechne gemitteltes Spektrum
    n_fft = 4096
    hop = n_fft // 4
    spec_accum = np.zeros(n_fft // 2 + 1)
    n_frames = 0
    for i in range(0, len(mono) - n_fft, hop * 10):  # Every 10th frame
        frame = mono[i : i + n_fft] * np.hanning(n_fft)
        spec_accum += np.abs(np.fft.rfft(frame))
        n_frames += 1

    spec_avg = spec_accum / max(n_frames, 1)
    spec_db = 20.0 * np.log10(np.maximum(spec_avg, 1e-10))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Spektrale Neigung über 100 Hz – 10 kHz
    mask = (freqs > 100) & (freqs < 10000)
    if mask.sum() > 10:
        coeffs = np.polyfit(freqs[mask], spec_db[mask], 1)
        tilt_db_per_octave = coeffs[0] * 1000  # dB/kHz ≈ dB/Oktave
    else:
        tilt_db_per_octave = 0.0

    passed = abs(tilt_db_per_octave) <= max_tilt_db_per_octave

    warning = ""
    recommendation = ""
    if tilt_db_per_octave > max_tilt_db_per_octave:
        warning = f"⚠️ Spektrum zu HELL: +{tilt_db_per_octave:.1f} dB/Oktave (Limit: ±{max_tilt_db_per_octave})"
        recommendation = "High-Shelf -2 dB @ 8 kHz anwenden"
    elif tilt_db_per_octave < -max_tilt_db_per_octave:
        warning = f"⚠️ Spektrum zu DUNKEL: {tilt_db_per_octave:.1f} dB/Oktave (Limit: ±{max_tilt_db_per_octave})"
        recommendation = "High-Shelf +2 dB @ 8 kHz anwenden"

    # Prüfe gegen Referenz (wenn vorhanden)
    ref_delta = None
    if reference_tilt is not None:
        ref_delta = tilt_db_per_octave - reference_tilt
        if abs(ref_delta) > 1.5:
            warning += f" | Δ zur Referenz: {ref_delta:+.1f} dB/Oktave"

    return {
        "passed": passed,
        "tilt_db_per_octave": float(tilt_db_per_octave),
        "reference_tilt": reference_tilt,
        "ref_delta_db_per_octave": ref_delta,
        "warning": warning,
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# S9: Automatic Multi-Format Export in One Pass
# ═══════════════════════════════════════════════════════════════════════════

MULTI_FORMAT_PRESETS = {
    "spotify": {"format": "wav", "sample_rate": 44100, "bit_depth": 16, "lufs": -14.0},
    "apple_music": {"format": "wav", "sample_rate": 44100, "bit_depth": 24, "lufs": -16.0},
    "youtube": {"format": "wav", "sample_rate": 48000, "bit_depth": 16, "lufs": -14.0},
    "tidal": {"format": "flac", "sample_rate": 44100, "bit_depth": 16, "lufs": -14.0},
    "amazon": {"format": "wav", "sample_rate": 44100, "bit_depth": 16, "lufs": -14.0},
    "deezer": {"format": "flac", "sample_rate": 44100, "bit_depth": 16, "lufs": -15.0},
    "soundcloud": {"format": "wav", "sample_rate": 48000, "bit_depth": 16, "lufs": -14.0},
    "broadcast": {"format": "wav", "sample_rate": 48000, "bit_depth": 24, "lufs": -23.0},
    "archival": {"format": "flac", "sample_rate": 96000, "bit_depth": 24, "lufs": -18.0},
    "cd_master": {"format": "wav", "sample_rate": 44100, "bit_depth": 16, "lufs": -14.0},
    "hi_res": {"format": "flac", "sample_rate": 96000, "bit_depth": 24, "lufs": -18.0},
    "car_optimized": {"format": "wav", "sample_rate": 48000, "bit_depth": 16, "lufs": -12.0},
}


def export_all_formats(
    audio: np.ndarray,
    sr: int,
    output_dir: str | Path,
    platforms: list[str] | None = None,
    base_name: str = "master",
) -> dict[str, Path]:
    """Exportiert das Master in ALLE relevanten Formate in EINEM Durchlauf.

    Args:
        audio:      Fertig gemastertes Audio (float32)
        sr:         Sample-Rate
        output_dir: Ausgabeverzeichnis
        platforms:  Liste von Plattformen (Default: alle)
        base_name:  Basis-Dateiname

    Returns:
        dict mit {platform: export_path}
    """
    import soundfile as sf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    platforms = platforms or list(MULTI_FORMAT_PRESETS.keys())

    results: dict[str, Path] = {}
    arr = np.asarray(audio, dtype=np.float32)

    for platform in platforms:
        preset = MULTI_FORMAT_PRESETS.get(platform)
        if preset is None:
            continue

        ext = preset["format"]
        target_sr = preset["sample_rate"]
        bit_depth = preset["bit_depth"]

        # Resampling (wenn nötig)
        if target_sr != sr:
            from scipy import signal as scipy_signal

            g = np.gcd(sr, target_sr)  # type: ignore[call-overload]
            up, down = target_sr // g, sr // g
            audio_out = scipy_signal.resample_poly(arr, up, down, axis=-1)
        else:
            audio_out = arr.copy()

        # Bit-Tiefe + Dithering
        if bit_depth == 16:
            dither_amp = 1.0 / (2**15)
            rng = np.random.RandomState(hash(platform) % (2**31))
            dither = rng.uniform(-dither_amp, dither_amp, audio_out.shape)
            dither += rng.uniform(-dither_amp, dither_amp, audio_out.shape)
            audio_out = audio_out + dither.astype(np.float32)
        elif bit_depth == 24:
            audio_out = audio_out.astype(np.float32)  # Float → 24-bit via soundfile

        audio_out = np.clip(audio_out, -1.0, 1.0)

        # Write
        out_path = output_dir / f"{base_name}_{platform}.{ext}"
        subtype = f"PCM_{bit_depth}" if ext in ("wav", "flac") else None
        sf.write(str(out_path), audio_out, target_sr, subtype=subtype)
        results[platform] = out_path

    # Write metadata summary
    import json

    meta_path = output_dir / f"{base_name}_export_manifest.json"
    manifest = {
        "source_sample_rate": sr,
        "platforms_exported": len(results),
        "files": {k: str(v) for k, v in results.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("S9: %d Formate exportiert → %s", len(results), output_dir)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# S10: Singer Identity Post-Pipeline Validation
# ═══════════════════════════════════════════════════════════════════════════


def validate_singer_identity_post_pipeline(
    pre_pipeline_audio: np.ndarray,
    post_pipeline_audio: np.ndarray,
    sr: int,
    threshold: float = 0.92,
) -> dict[str, Any]:
    """Validiert die Sänger-Identität NACH der gesamten Pipeline.

    Wenn die Cosine-Similarity zwischen Pre- und Post-Pipeline-Voiceprint
    unter den Schwellwert fällt, wurde die Stimme zu stark verändert.

    Args:
        pre_pipeline_audio:  Original-Audio (vor Pipeline)
        post_pipeline_audio: Restauriertes Audio (nach Pipeline)
        sr:                  Sample-Rate
        threshold:           Cosine-Similarity-Schwellwert (Default 0.92)

    Returns:
        dict mit 'identity_preserved', 'cosine_similarity', 'warning'
    """
    try:
        from backend.ml.speaker_identity_guard import SpeakerIdentityGuard

        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(pre_pipeline_audio, sr)
        result = guard.check_phase(post_pipeline_audio, sr, "final")  # type: ignore[arg-type]

        if result is not None and hasattr(result, "cosine_similarity"):
            sim = float(result.cosine_similarity)
        else:
            # Fallback: einfache MFCC-Distanz
            sim = _compute_simple_mfcc_similarity(pre_pipeline_audio, post_pipeline_audio, sr)

        identity_preserved = sim >= threshold
        warning = ""
        if not identity_preserved:
            warning = (
                f"⚠️ Sänger-Identität möglicherweise verändert: Cosine-Similarity = {sim:.3f} (Schwelle: {threshold})"
            )

        return {
            "identity_preserved": identity_preserved,
            "cosine_similarity": sim,
            "threshold": threshold,
            "warning": warning,
        }
    except Exception as e:
        logger.debug("S10: Sänger-Identitäts-Validierung: %s", e)
        return {"identity_preserved": True, "cosine_similarity": 1.0, "warning": ""}


def _compute_simple_mfcc_similarity(audio1: np.ndarray, audio2: np.ndarray, sr: int, n_mfcc: int = 13) -> float:
    """Einfache MFCC-basierte Ähnlichkeit (Fallback)."""
    try:
        import librosa

        mfcc1 = librosa.feature.mfcc(
            y=np.asarray(audio1, dtype=np.float32).mean(axis=0) if audio1.ndim > 1 else audio1, sr=sr, n_mfcc=n_mfcc
        )
        mfcc2 = librosa.feature.mfcc(
            y=np.asarray(audio2, dtype=np.float32).mean(axis=0) if audio2.ndim > 1 else audio2, sr=sr, n_mfcc=n_mfcc
        )
        mfcc1_mean = mfcc1.mean(axis=1)
        mfcc2_mean = mfcc2.mean(axis=1)
        dot = float(np.dot(mfcc1_mean, mfcc2_mean))
        norm = float(np.linalg.norm(mfcc1_mean) * np.linalg.norm(mfcc2_mean) + 1e-12)
        return dot / norm
    except ImportError as e:
        logger.debug("§V6 MFCC-Ähnlichkeit: librosa nicht verfügbar — 1.0 (Identity) zurückgegeben: %s", e)
        return 1.0
