"""Artist/Track-Fingerprint — Persistente Kuenstler- und Track-Modelle.

Spec 13 paragraph 13.11, Spec 11 paragraph ROADMAP-3.
Speichert Stimm-Modelle (Formanten, Vibrato-Rate, HNR, spektrale Huellkurve)
und Track-Modelle (Genre, Era, Aufnahmekette, typische Defekte) pro song_id.
Wiederverwendung beschleunigt wiederholte Restaurierungen desselben Kuenstlers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FINGERPRINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "sessions", "fingerprints")
MAX_FINGERPRINT_AGE_DAYS = 90


@dataclass
class SingerVoiceFingerprint:
    """Stimm-Modell eines Kuenstlers: Formanten, Vibrato, HNR, spektrale Huellkurve."""

    artist_id: str
    formant_f1_hz: float = 0.0
    formant_f2_hz: float = 0.0
    vibrato_rate_hz: float = 0.0
    vibrato_extent_semitones: float = 0.0
    hnr_db: float = 0.0
    spectral_envelope: list[float] = field(default_factory=list)
    f0_mean_hz: float = 0.0
    f0_range_hz: tuple[float, float] = (0.0, 0.0)
    gender: str = "unknown"
    last_updated: float = 0.0
    observation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artist_id": self.artist_id,
            "formant_f1_hz": self.formant_f1_hz,
            "formant_f2_hz": self.formant_f2_hz,
            "vibrato_rate_hz": self.vibrato_rate_hz,
            "vibrato_extent_semitones": self.vibrato_extent_semitones,
            "hnr_db": self.hnr_db,
            "spectral_envelope": self.spectral_envelope,
            "f0_mean_hz": self.f0_mean_hz,
            "f0_range_hz": list(self.f0_range_hz),
            "gender": self.gender,
            "last_updated": self.last_updated,
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SingerVoiceFingerprint:
        return cls(
            artist_id=d["artist_id"],
            formant_f1_hz=d.get("formant_f1_hz", 0.0),
            formant_f2_hz=d.get("formant_f2_hz", 0.0),
            vibrato_rate_hz=d.get("vibrato_rate_hz", 0.0),
            vibrato_extent_semitones=d.get("vibrato_extent_semitones", 0.0),
            hnr_db=d.get("hnr_db", 0.0),
            spectral_envelope=d.get("spectral_envelope", []),
            f0_mean_hz=d.get("f0_mean_hz", 0.0),
            f0_range_hz=tuple(d.get("f0_range_hz", [0.0, 0.0])),
            gender=d.get("gender", "unknown"),
            last_updated=d.get("last_updated", 0.0),
            observation_count=d.get("observation_count", 0),
        )

    def update_from_observation(self, other: SingerVoiceFingerprint) -> None:
        """Gleitender Durchschnitt: neuer Wert = (n * alt + neu) / (n + 1)."""
        n = self.observation_count
        w_old = n / (n + 1)
        w_new = 1.0 / (n + 1)
        self.formant_f1_hz = w_old * self.formant_f1_hz + w_new * other.formant_f1_hz
        self.formant_f2_hz = w_old * self.formant_f2_hz + w_new * other.formant_f2_hz
        self.vibrato_rate_hz = w_old * self.vibrato_rate_hz + w_new * other.vibrato_rate_hz
        self.vibrato_extent_semitones = w_old * self.vibrato_extent_semitones + w_new * other.vibrato_extent_semitones
        self.hnr_db = w_old * self.hnr_db + w_new * other.hnr_db
        self.f0_mean_hz = w_old * self.f0_mean_hz + w_new * other.f0_mean_hz
        self.gender = other.gender if other.gender != "unknown" else self.gender
        self.last_updated = time.time()
        self.observation_count += 1
        if other.spectral_envelope and self.spectral_envelope:
            if len(other.spectral_envelope) == len(self.spectral_envelope):
                self.spectral_envelope = [
                    w_old * s + w_new * o for s, o in zip(self.spectral_envelope, other.spectral_envelope)
                ]
        elif other.spectral_envelope:
            self.spectral_envelope = other.spectral_envelope


@dataclass
class TrackFingerprint:
    """Track-Modell: Genre, Era, Aufnahmekette, typische Defekte."""

    track_id: str
    genre: str = "unknown"
    era_decade: int = 2000
    material: str = "unknown"
    transfer_chain_depth: int | None = None
    typical_defects: list[str] = field(default_factory=list)
    bandwidth_hz: float = 20000.0
    snr_db: float = 60.0
    stereo_width: float = 0.5
    dynamic_range_db: float = 30.0
    last_updated: float = 0.0
    restoration_count: int = 0

    def __post_init__(self) -> None:
        """§G86 (GEBOTE.md): transfer_chain_depth-Default nur aus CalibrationContext."""
        if self.transfer_chain_depth is None:
            from backend.core.defect_to_audibility import _resolve_transfer_chain_depth

            self.transfer_chain_depth = _resolve_transfer_chain_depth(None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "genre": self.genre,
            "era_decade": self.era_decade,
            "material": self.material,
            "transfer_chain_depth": self.transfer_chain_depth,
            "typical_defects": self.typical_defects,
            "bandwidth_hz": self.bandwidth_hz,
            "snr_db": self.snr_db,
            "stereo_width": self.stereo_width,
            "dynamic_range_db": self.dynamic_range_db,
            "last_updated": self.last_updated,
            "restoration_count": self.restoration_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackFingerprint:
        return cls(
            track_id=d["track_id"],
            genre=d.get("genre", "unknown"),
            era_decade=d.get("era_decade", 2000),
            material=d.get("material", "unknown"),
            transfer_chain_depth=d.get("transfer_chain_depth", 1),
            typical_defects=d.get("typical_defects", []),
            bandwidth_hz=d.get("bandwidth_hz", 20000.0),
            snr_db=d.get("snr_db", 60.0),
            stereo_width=d.get("stereo_width", 0.5),
            dynamic_range_db=d.get("dynamic_range_db", 30.0),
            last_updated=d.get("last_updated", 0.0),
            restoration_count=d.get("restoration_count", 0),
        )


class ArtistFingerprintStore:
    """Persistente Speicherung von Artist/Track-Fingerprints.

    Speichert pro artist_id/track_id im JSON-Format.
    Automatisches Bereinigen alter Eintraege (>90 Tage).
    """

    def __init__(self, store_dir: str | None = None) -> None:
        self._store_dir = store_dir or DEFAULT_FINGERPRINT_DIR
        os.makedirs(self._store_dir, exist_ok=True)
        self._voice_cache: dict[str, SingerVoiceFingerprint] = {}
        self._track_cache: dict[str, TrackFingerprint] = {}

    def store_voice(self, fingerprint: SingerVoiceFingerprint) -> None:
        """Persistiert ein Stimm-Modell."""
        fingerprint.last_updated = time.time()
        self._voice_cache[fingerprint.artist_id] = fingerprint
        path = self._voice_path(fingerprint.artist_id)
        try:
            with open(path, "w") as f:
                json.dump(fingerprint.to_dict(), f, indent=2)
        except OSError as e:
            logger.warning("ArtistFingerprintStore: voice save failed for %s: %s", fingerprint.artist_id, e)

    def load_voice(self, artist_id: str) -> SingerVoiceFingerprint | None:
        """Laedt ein Stimm-Modell, mit Cache."""
        if artist_id in self._voice_cache:
            return self._voice_cache[artist_id]
        path = self._voice_path(artist_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            fp = SingerVoiceFingerprint.from_dict(data)
            if time.monotonic() - fp.last_updated > MAX_FINGERPRINT_AGE_DAYS * 86400:
                os.remove(path)
                return None
            self._voice_cache[artist_id] = fp
            return fp
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning("ArtistFingerprintStore: voice load failed for %s: %s", artist_id, e)
            return None

    def store_track(self, fingerprint: TrackFingerprint) -> None:
        """Persistiert ein Track-Modell."""
        fingerprint.last_updated = time.monotonic()
        self._track_cache[fingerprint.track_id] = fingerprint
        path = self._track_path(fingerprint.track_id)
        try:
            with open(path, "w") as f:
                json.dump(fingerprint.to_dict(), f, indent=2)
        except OSError as e:
            logger.warning("ArtistFingerprintStore: track save failed for %s: %s", fingerprint.track_id, e)

    def load_track(self, track_id: str) -> TrackFingerprint | None:
        """Laedt ein Track-Modell, mit Cache."""
        if track_id in self._track_cache:
            return self._track_cache[track_id]
        path = self._track_path(track_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            fp = TrackFingerprint.from_dict(data)
            if time.monotonic() - fp.last_updated > MAX_FINGERPRINT_AGE_DAYS * 86400:
                os.remove(path)
                return None
            self._track_cache[track_id] = fp
            return fp
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning("ArtistFingerprintStore: track load failed for %s: %s", track_id, e)
            return None

    def update_or_create_voice(self, artist_id: str, new_fp: SingerVoiceFingerprint) -> SingerVoiceFingerprint:
        """Laedt existierenden Fingerprint und updated ihn, oder erstellt neuen."""
        existing = self.load_voice(artist_id)
        if existing is not None:
            existing.update_from_observation(new_fp)
            self.store_voice(existing)
            return existing
        self.store_voice(new_fp)
        return new_fp

    def find_artist_by_voice_similarity(self, candidate: SingerVoiceFingerprint, threshold: float = 0.85) -> str | None:
        """Findet einen aehnlichen Kuenstler-Fingerprint via Cosine-Similarity."""
        best_id: str | None = None
        best_sim = 0.0
        for artist_id in self._list_voice_ids():
            stored = self.load_voice(artist_id)
            if stored is None:
                continue
            sim = self._voice_similarity(candidate, stored)
            if sim > best_sim:
                best_sim = sim
                best_id = artist_id
        if best_sim >= threshold:
            return best_id
        return None

    @staticmethod
    def _voice_similarity(a: SingerVoiceFingerprint, b: SingerVoiceFingerprint) -> float:
        """Cosine-Similarity zwischen zwei Stimm-Fingerprints."""
        vec_a = np.array(
            [
                a.formant_f1_hz / 1000.0,
                a.formant_f2_hz / 2000.0,
                a.vibrato_rate_hz / 10.0,
                a.vibrato_extent_semitones / 2.0,
                a.hnr_db / 40.0,
                a.f0_mean_hz / 500.0,
            ]
        )
        vec_b = np.array(
            [
                b.formant_f1_hz / 1000.0,
                b.formant_f2_hz / 2000.0,
                b.vibrato_rate_hz / 10.0,
                b.vibrato_extent_semitones / 2.0,
                b.hnr_db / 40.0,
                b.f0_mean_hz / 500.0,
            ]
        )
        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a) + 1e-12)
        norm_b = float(np.linalg.norm(vec_b) + 1e-12)
        return dot / (norm_a * norm_b)

    def _voice_path(self, artist_id: str) -> str:
        safe = "".join(c for c in artist_id if c.isalnum() or c in "._-")
        return os.path.join(self._store_dir, f"voice_{safe}.json")

    def _track_path(self, track_id: str) -> str:
        safe = "".join(c for c in track_id if c.isalnum() or c in "._-")
        return os.path.join(self._store_dir, f"track_{safe}.json")

    def _list_voice_ids(self) -> list[str]:
        ids: list[str] = []
        try:
            for fname in os.listdir(self._store_dir):
                if fname.startswith("voice_") and fname.endswith(".json"):
                    ids.append(fname[6:-5])
        except OSError:
            pass
        return ids

    def cleanup_expired(self, max_age_days: int = MAX_FINGERPRINT_AGE_DAYS) -> int:
        """Entfernt alle Fingerprints, die aelter als max_age_days sind."""
        removed = 0
        cutoff = time.time() - max_age_days * 86400
        try:
            for fname in os.listdir(self._store_dir):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(self._store_dir, fname)
                try:
                    mtime = os.path.getmtime(path)
                    if mtime < cutoff:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass
        except OSError:
            pass
        if removed > 0:
            logger.info("ArtistFingerprintStore: removed %d expired fingerprints", removed)
        return removed


_global_store: ArtistFingerprintStore | None = None


def get_artist_fingerprint_store() -> ArtistFingerprintStore:
    """Singleton-Zugriff auf den ArtistFingerprintStore."""
    global _global_store
    if _global_store is None:
        _global_store = ArtistFingerprintStore()
    return _global_store


def extract_singer_voice_fingerprint(
    audio: np.ndarray,
    sample_rate: int,
    *,
    artist_id: str = "unknown",
    f0_hz: np.ndarray | None = None,
    formants: tuple[float, float] | None = None,
    hnr_db: float | None = None,
) -> SingerVoiceFingerprint:
    """Extrahiert ein Stimm-Fingerprint aus Audio-Daten.

    Nutzt existierende Analyse-Daten (f0, Formanten, HNR) wenn verfuegbar,
    sonst einfache Heuristiken.
    """
    fp = SingerVoiceFingerprint(artist_id=artist_id)
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    fp.last_updated = time.time()
    fp.observation_count = 1

    if formants is not None:
        fp.formant_f1_hz = formants[0]
        fp.formant_f2_hz = formants[1]
    else:
        fp.formant_f1_hz = 600.0
        fp.formant_f2_hz = 1200.0

    if hnr_db is not None:
        fp.hnr_db = hnr_db

    if f0_hz is not None and len(f0_hz) > 0:
        voiced = f0_hz[f0_hz > 0]
        if len(voiced) > 0:
            fp.f0_mean_hz = float(np.median(voiced))
            fp.f0_range_hz = (float(np.min(voiced)), float(np.max(voiced)))
            fp.vibrato_rate_hz = max(4.0, min(8.0, fp.f0_mean_hz * 0.02))
            fp.vibrato_extent_semitones = 0.5

    if fp.f0_mean_hz < 165:
        fp.gender = "male"
    elif fp.f0_mean_hz > 165:
        fp.gender = "female"

    n_bands = 16
    spec_env: list[float] = []
    try:
        from scipy.signal import spectrogram  # type: ignore[import]

        f, _, Sxx = spectrogram(mono, fs=sample_rate, nperseg=1024)  # type: ignore[no-any-return]
        band_edges = np.logspace(np.log10(80), np.log10(8000), n_bands + 1)
        for i in range(n_bands):
            mask = (f >= band_edges[i]) & (f < band_edges[i + 1])
            if mask.any():
                spec_env.append(float(np.mean(Sxx[mask])))
            else:
                spec_env.append(0.0)
        total = sum(spec_env) + 1e-12
        spec_env = [v / total for v in spec_env]
    except Exception:
        spec_env = [1.0 / n_bands] * n_bands

    fp.spectral_envelope = spec_env
    return fp


def extract_track_fingerprint(
    audio: np.ndarray,
    sample_rate: int,
    *,
    track_id: str = "unknown",
    genre: str = "unknown",
    era_decade: int = 2000,
    material: str = "unknown",
    transfer_chain_depth: int | None = None,
    defects: list[str] | None = None,
) -> TrackFingerprint:
    """Extrahiert ein Track-Fingerprint aus Audio-Daten."""
    fp = TrackFingerprint(
        track_id=track_id,
        genre=genre,
        era_decade=era_decade,
        material=material,
        transfer_chain_depth=transfer_chain_depth,
        typical_defects=defects or [],
        last_updated=time.time(),
        restoration_count=1,
    )
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio

    try:
        spec = np.fft.rfft(mono[: min(len(mono), sample_rate * 10)])
        mag = np.abs(spec)
        cumsum = np.cumsum(mag)
        total = cumsum[-1] + 1e-12
        for i in range(len(cumsum)):
            if cumsum[i] / total >= 0.95:
                fp.bandwidth_hz = float(i) / len(mag) * (sample_rate / 2)
                break

        rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
        quiet_mask = np.abs(mono) < rms * 0.1
        if quiet_mask.any():
            noise_rms = float(np.sqrt(np.mean(mono[quiet_mask] ** 2)))
            fp.snr_db = float(20 * np.log10(rms / (noise_rms + 1e-12)))
        fp.dynamic_range_db = float(20 * np.log10(np.max(np.abs(mono)) / (rms + 1e-12)))

        if audio.ndim == 2 and audio.shape[1] == 2:
            corr = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
            fp.stereo_width = max(0.0, min(1.0, 1.0 - abs(corr)))
    except Exception as e:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        logger.debug("Track-Fingerprint extraction fallback: %s", e)

    return fp
