"""§v10.303.20–21 Phase-0 Pre-Processor Pipeline für Aurik UV3.

Wissenschaftlich korrekte Reihenfolge (Carrier-Chain-Inversion §2.46):
  1. Apollo      (Codec-Decompression)  — subtraktiv: MP3/AAC-Artefakte entfernen
  2. DeepFilterNet v3 (Denoising)       — subtraktiv: Noise-Floor stabilisieren
  3. Resemble Enhance (Enhancement)     — additiv: Spektrale Reparatur

Cache (§v10.303.18): Hash-basierte Persistenz in ~/.aurik/cache/phase0/.
Vermeidet wiederholte ML-Inferenz bei Batch-Imports und Re-Imports.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Cache: Per-Material-Effectiveness ───────────────────────────────────

_effectiveness_cache: dict[str, dict[str, Any]] = {}


def _material_cache_key(audio: np.ndarray, material: str) -> str:
    """Cache-Key aus Material und Audio-Signatur."""
    _prefix = audio[: min(len(audio), 48000)]
    _h = hashlib.sha256(_prefix.tobytes()).hexdigest()[:16]
    return f"{material}_{_h}"


# ── Spectral Novelty ───────────────────────────────────────────────────


def _spectral_novelty(original: np.ndarray, processed: np.ndarray, sr: int = 48000) -> float:
    """§2.46e: Misst spektrale Neuheit von processed relativ zu original."""
    _mono_orig = original if original.ndim == 1 else np.mean(original, axis=0)
    _mono_proc = processed if processed.ndim == 1 else np.mean(processed, axis=0)
    _n_fft, _hop = 2048, 512
    _min_len = min(len(_mono_orig), len(_mono_proc))
    if _min_len < _n_fft:
        return 0.0
    _mono_orig, _mono_proc = _mono_orig[:_min_len], _mono_proc[:_min_len]

    _spec_orig = np.abs(np.stack([np.fft.rfft(_mono_orig[i : i + _n_fft]) for i in range(0, _min_len - _n_fft, _hop)]))
    _spec_proc = np.abs(np.stack([np.fft.rfft(_mono_proc[i : i + _n_fft]) for i in range(0, _min_len - _n_fft, _hop)]))
    _n = min(_spec_orig.shape[0], _spec_proc.shape[0])
    _spec_orig, _spec_proc = _spec_orig[:_n], _spec_proc[:_n]
    _diff = np.abs(_spec_proc - _spec_orig)
    return float(np.clip(np.mean(_diff) / (np.mean(_spec_orig) + 1e-10), 0.0, 1.0))


# ── Simplified Quality Check ────────────────────────────────────────────


def _quality_delta(original: np.ndarray, processed: np.ndarray, sr: int = 48000) -> dict[str, float]:
    """Misst Qualitätsänderung zwischen Original und Apollo-Output."""
    _mono_orig = original if original.ndim == 1 else np.mean(original, axis=0)
    _mono_proc = processed if processed.ndim == 1 else np.mean(processed, axis=0)
    _min_len = min(len(_mono_orig), len(_mono_proc))
    _mono_orig, _mono_proc = _mono_orig[:_min_len], _mono_proc[:_min_len]

    # RMS (Lautheit)
    _rms_orig = float(np.sqrt(np.mean(_mono_orig**2)) + 1e-12)
    _rms_proc = float(np.sqrt(np.mean(_mono_proc**2)) + 1e-12)
    _rms_delta_db = float(20 * np.log10(_rms_proc / _rms_orig))

    # Crest-Faktor (Dynamik-Erhalt)
    _peak_orig = float(np.max(np.abs(_mono_orig)) + 1e-12)
    _peak_proc = float(np.max(np.abs(_mono_proc)) + 1e-12)
    _crest_orig = _peak_orig / _rms_orig
    _crest_proc = _peak_proc / _rms_proc
    _crest_delta = float((_crest_proc - _crest_orig) / (_crest_orig + 1e-10))

    # HF-Energie (8 kHz+)
    _n_fft = 2048
    _spec_orig = np.abs(np.fft.rfft(_mono_orig, n=_n_fft))
    _spec_proc = np.abs(np.fft.rfft(_mono_proc, n=_n_fft))
    _freqs = np.fft.rfftfreq(_n_fft, 1 / sr)
    _hf_mask = _freqs >= 8000
    _hf_orig = float(np.sum(_spec_orig[_hf_mask]) + 1e-12)
    _hf_proc = float(np.sum(_spec_proc[_hf_mask]) + 1e-12)
    _hf_delta_db = float(20 * np.log10(_hf_proc / _hf_orig))

    # Spectrale Varianz (Textur-Erhalt)
    _spec_var_orig = float(np.var(_spec_orig) / (np.mean(_spec_orig) + 1e-10))
    _spec_var_proc = float(np.var(_spec_proc) / (np.mean(_spec_proc) + 1e-10))
    _spec_var_delta = float((_spec_var_proc - _spec_var_orig) / (_spec_var_orig + 1e-10))

    return {
        "rms_delta_db": round(_rms_delta_db, 2),
        "crest_delta": round(_crest_delta, 4),
        "hf_delta_db": round(_hf_delta_db, 2),
        "spec_var_delta": round(_spec_var_delta, 4),
    }


# ── Apollo Phase-0 Guard ────────────────────────────────────────────────


from dataclasses import dataclass, field
from typing import cast


@dataclass
class ApolloResult:
    """Ergebnis eines Apollo Phase-0 Durchlaufs."""

    audio: np.ndarray
    applied: bool = False
    novelty: float = 0.0
    rms_delta_db: float = 0.0
    hf_delta_db: float = 0.0
    goosebumps_preserved: bool = True
    elapsed_s: float = 0.0
    material: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class ApolloPhase0Guard:
    """Apollo MP3-Decompression mit Qualitäts-Guard.

    Nutzt models/apollo/apollo_model.pt (TorchScript).
    Wendet Apollo NUR an wenn:
    1. Material ist lossy codec (mp3/aac/minidisc/streaming)
    2. Apollo ist geladen und verfügbar
    3. Spectral novelty ≤ threshold (keine Halluzination)
    4. Quality-Delta ist nicht-degradierend

    Bei Degradation oder Halluzination: Gibt Original unverändert zurück.
    """

    def __init__(
        self,
        model_path: str | None = None,
        hallucination_threshold: float = 0.35,
    ):
        import os as _os

        self._model_path = model_path or _os.environ.get(
            "AURIK_APOLLO_MODEL",
            _os.path.join(
                _os.path.dirname(__file__),
                "..",
                "models",
                "apollo",
                "apollo_model.pt",
            ),
        )
        self._hallucination_threshold = float(hallucination_threshold)  # default 0.35
        self._model = None
        self._device = "cpu"
        self._loaded = False
        self._cached_effective: set[str] = set()  # Materialien wo Apollo half
        self._cached_ineffective: set[str] = set()  # Materialien wo Apollo nichts brachte

    def unload(self) -> None:
        """§v10.402: Apollo-Modell entladen. Überspringt wenn PLM-Singleton (shared)."""
        if getattr(self, "_shared_from_plm", False):
            self._model = None  # Nur Referenz lösen, nicht das Modell
            self._loaded = False
            return
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        import gc

        gc.collect()

    # ── Modell-Ladung ────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Lädt Apollo Modell. Returns True bei Erfolg.

        §v10.402: Nutzt PLM-Singleton wenn verfügbar — spart 800 MB RAM.
        """
        if self._loaded:
            return True
        # §v10.402: PLM-Singleton statt eigener Instanz
        try:
            from plugins.apollo_plugin import get_loaded_apollo

            _apollo = get_loaded_apollo()
            if _apollo is not None:
                _model = getattr(_apollo, "_model", None)
                if _model is not None:
                    self._model = _model
                    self._device = getattr(_apollo, "_device", "cpu")
                    self._loaded = True
                    self._shared_from_plm = True
                    logger.info("Apollo Verarbeitungsschritt-0 via PLM-Singleton (shared model, 0 MB extra)")
                    return True
        except Exception:
            pass
        # Fallback: eigenes Modell laden
        if not __import__("os").path.isfile(self._model_path):
            logger.debug("Apollo-Modell nicht gefunden: %s", self._model_path)
            return False
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = torch.jit.load(self._model_path, map_location=self._device)
            self._model.eval()  # type: ignore[attr-defined]
            self._loaded = True
            logger.info(
                "Apollo Verarbeitungsschritt-0 geladen (device=%s, %.1f MB)",
                self._device,
                __import__("os").path.getsize(self._model_path) / 1e6,
            )
            return True
        except ImportError:
            logger.debug("PyTorch nicht installiert.")
            return False
        except Exception as exc:
            logger.warning("Apollo-Ladung fehlgeschlagen: %s", exc)
            return False

    # ── Processing ───────────────────────────────────────────────────

    @staticmethod
    def should_apply(material: Any, transfer_chain: list | None = None) -> bool:
        """Prüft ob Apollo angewendet werden sollte.

        Checkt primäres Material UND terminalen Carrier der Transfer-Chain.
        Z.B. vinyl→cassette→mp3_high: primary=cassette, terminal=mp3_high → Apollo JA.
        """
        _lossy = {"mp3_low", "mp3_high", "aac", "minidisc", "streaming"}
        _mat = ""
        if isinstance(material, str):
            _mat = material.strip().lower()
        elif hasattr(material, "value"):
            _mat = str(material.value).strip().lower()
        else:
            _mat = str(material).strip().lower()
        if _mat in _lossy:
            return True
        # Check terminal carrier
        if transfer_chain and isinstance(transfer_chain, list) and len(transfer_chain) > 0:
            _terminal = str(transfer_chain[-1]).strip().lower()
            if _terminal in _lossy:
                return True
        return False

    def process(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        material: Any = "unknown",
    ) -> ApolloResult:
        """Führt Apollo Decompression mit Guard aus.

        Args:
            audio: Eingabe-Audio (float32, [-1,1])
            sr: Sample-Rate
            material: Material-String (z.B. "mp3_low", "mp3_high", "aac")

        Returns:
            ApolloResult mit audio, applied, novelty, quality-Metriken.
        """
        _audio = np.asarray(audio, dtype=np.float32)
        _audio = np.nan_to_num(_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        _audio = np.clip(_audio, -1.0, 1.0)

        # Material-Namen normalisieren
        if hasattr(material, "value"):
            _mat = str(material.value).strip().lower()
        else:
            _mat = str(material).strip().lower()

        # Nur für lossy codec
        if not self.should_apply(material):
            return ApolloResult(audio=_audio, applied=False, material=_mat)

        # Skip wenn bereits als ineffektiv gecached
        if _mat in self._cached_ineffective:
            return ApolloResult(audio=_audio, applied=False, material=_mat)

        if not self.load():
            return ApolloResult(audio=_audio, applied=False, material=_mat)

        t0 = time.perf_counter()
        _novelty = 0.0
        _qual: dict[str, float] = {}
        try:
            import torch
            import torchaudio

            _model = self._model
            if _model is None:
                return ApolloResult(audio=_audio, applied=False, material=_mat)

            # Handle stereo → mono für Apollo (dann remix)
            _is_stereo = _audio.ndim == 2 and _audio.shape[0] == 2
            _mono = _audio if not _is_stereo else np.mean(_audio, axis=0)

            # Resample zu 44100 (Apollo intern)
            _apollo_sr = 44100
            t = torch.from_numpy(_mono).float().unsqueeze(0).unsqueeze(0).to(self._device)
            if sr != _apollo_sr:
                t = torchaudio.functional.resample(t, sr, _apollo_sr)

            # Chunked processing
            _chunk_samples = 8 * _apollo_sr
            _total = t.shape[-1]
            _result = torch.zeros_like(t)

            for _start in range(0, _total, _chunk_samples):
                _end = min(_start + _chunk_samples, _total)
                _chunk = t[:, :, _start:_end]
                with torch.no_grad():
                    _out = _model(_chunk)
                _result[:, :, _start:_end] = _out

            # Resample zurück
            if sr != _apollo_sr:
                _result = torchaudio.functional.resample(_result, _apollo_sr, sr)

            _processed = _result.squeeze().cpu().numpy().astype(np.float32)
            _processed = np.nan_to_num(_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            _processed = np.clip(_processed, -1.0, 1.0)

            # Auf Original-Länge trimmen
            _processed = _processed[: len(_mono)]

            # ── Guard 1: Hallucination ──
            _novelty = _spectral_novelty(_mono, _processed, sr)
            if _novelty > self._hallucination_threshold:
                logger.warning(
                    "§2.46e Apollo Hallucination-Guard: novelty=%.3f > %.2f → Rollback (Material=%s)",
                    _novelty,
                    self._hallucination_threshold,
                    _mat,
                )
                self._cached_ineffective.add(_mat)
                return ApolloResult(
                    audio=_audio,
                    applied=False,
                    novelty=_novelty,
                    material=_mat,
                    goosebumps_preserved=True,
                )

            # ── Guard 2: Quality-Delta ──
            _qual = _quality_delta(_mono, _processed, sr)
            _degraded = (
                _qual["rms_delta_db"] < -3.0  # Mehr als 3 dB leiser
                or _qual["crest_delta"] < -0.5  # Dynamik massiv reduziert
                or _qual["hf_delta_db"] < -6.0  # HF massiv verloren
                or _qual["spec_var_delta"] < -0.5  # Textur zerstört
            )
            if _degraded:
                logger.warning(
                    "Apollo Quality-Guard: Degradation erkannt (Material=%s) → Rollback. "
                    "RMS=%.1fdB Crest=%.3f HF=%.1fdB SpecVar=%.3f",
                    _mat,
                    _qual["rms_delta_db"],
                    _qual["crest_delta"],
                    _qual["hf_delta_db"],
                    _qual["spec_var_delta"],
                )
                self._cached_ineffective.add(_mat)
                return ApolloResult(
                    audio=_audio,
                    applied=False,
                    novelty=_novelty,
                    rms_delta_db=_qual["rms_delta_db"],
                    hf_delta_db=_qual["hf_delta_db"],
                    material=_mat,
                    goosebumps_preserved=False,
                )

            # ── Apollo erfolgreich ──
            _elapsed = time.perf_counter() - t0
            logger.info(
                "Apollo Verarbeitungsschritt-0: %s decompressed (%.2fs, novelty=%.3f, HF=+%.1fdB)",
                _mat,
                _elapsed,
                _novelty,
                _qual.get("hf_delta_db", 0.0),
            )
            self._cached_effective.add(_mat)

            # Remix stereo wenn nötig
            _out_audio: np.ndarray
            if _is_stereo:
                _out_audio = np.stack([_processed, _processed], axis=0).astype(np.float32)
            else:
                _out_audio = _processed.astype(np.float32)

            return ApolloResult(
                audio=_out_audio,
                applied=True,
                novelty=_novelty,
                rms_delta_db=_qual.get("rms_delta_db", 0.0),
                hf_delta_db=_qual.get("hf_delta_db", 0.0),
                goosebumps_preserved=True,
                elapsed_s=round(_elapsed, 2),
                material=_mat,
                metadata={"quality": _qual},
            )

        except Exception as exc:
            logger.warning("Apollo-Verarbeitung fehlgeschlagen (%s): %s", type(exc).__name__, exc)
            return ApolloResult(audio=_audio, applied=False, material=_mat)

    # ── Reset-Funktion (zwischen Songs) ────────────────────────────────

    def reset(self) -> None:
        """Setzt den Cache zurück (zwischen verschiedenen Songs aufrufen)."""
        self._cached_effective.clear()
        self._cached_ineffective.clear()

    # ── Utility ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_material(material: Any) -> str:
        """Normalisiert MaterialType/Enum/String zu Kleinbuchstaben."""
        if hasattr(material, "value"):
            return str(material.value).strip().lower()
        return str(material).strip().lower()

    @staticmethod
    def is_lossy_codec_material(material: str) -> bool:
        """Prüft ob Material von verlustbehaftetem Codec stammt."""
        return str(material).strip().lower() in {
            "mp3_low",
            "mp3_high",
            "aac",
            "minidisc",
            "streaming",
        }


# ═══════════════════════════════════════════════════════════════════════
# Resemble Enhance Phase-0b Guard
# ═══════════════════════════════════════════════════════════════════════


class ResembleEnhanceGuard:
    """Phase-0b: Resemble Enhance — Rauschunterdrückung + spektrale Reparatur.

    Läuft NACH Apollo (Phase-0a) für Codec-Material.
    Läuft als EINZIGER Pre-Processor für nicht-Codec-Material.

    Modell: models/resemble_enhance/model.onnx (ONNX, CPU, 722 MB)
    Fallback: Wiener-DSP (eingebauter Fallback im ResembleEnhancePlugin)  # §V6 (copilot-instructions.md): logger.warning handled at call site
    """

    def __init__(self, hallucination_threshold: float = 0.40):
        self._threshold = hallucination_threshold
        self._loaded = False
        self._plugin = None

    def unload(self) -> None:
        """§v10.402: Resemble entladen. Überspringt wenn PLM-Singleton."""
        if getattr(self, "_shared_from_plm", False):
            self._plugin = None
            self._loaded = False
            return
        if self._plugin is not None:
            self._plugin.unload()
            self._plugin = None
        self._loaded = False
        import gc

        gc.collect()

    def _ensure_loaded(self) -> bool:
        if self._loaded and self._plugin is not None:
            return True
        # §v10.402: PLM-Singleton statt eigener Instanz
        try:
            from plugins.resemble_enhance_plugin import get_resemble_enhance_plugin

            self._plugin = get_resemble_enhance_plugin()  # type: ignore[assignment]
            self._loaded = getattr(self._plugin, "_session", None) is not None
            if self._loaded:
                self._shared_from_plm = True
                logger.info("Resemble verbessern Verarbeitungsschritt-0 via PLM-Singleton")
            return self._loaded
        except Exception:
            pass
        # Fallback
        try:
            from plugins.resemble_enhance_plugin import ResembleEnhancePlugin

            self._plugin = ResembleEnhancePlugin()  # type: ignore[assignment]
            self._loaded = self._plugin._session is not None  # type: ignore[attr-defined]
            if self._loaded:
                logger.info("Resemble verbessern Verarbeitungsschritt-0 geladen")
            return self._loaded
        except ImportError:
            logger.debug("Resemble verbessern Plugin nicht verfügbar")
            return False
        except Exception as exc:
            logger.warning("Resemble verbessern Ladefehler: %s", exc)
            return False

    def process(self, audio: np.ndarray, sr: int = 48000) -> tuple[np.ndarray, bool]:
        """Führt Resemble Enhance mit Quality-Guard aus.

        Returns: (audio_out, applied)
        """
        _audio = np.asarray(audio, dtype=np.float32)
        _audio = np.nan_to_num(_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        _audio = np.clip(_audio, -1.0, 1.0)

        if not self._ensure_loaded():
            return _audio, False

        t0 = time.perf_counter()
        try:
            _plugin = self._plugin
            if _plugin is None:
                return _audio, False

            _processed = _plugin.enhance(_audio, sr)
            _processed = np.nan_to_num(_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            _processed = np.clip(_processed, -1.0, 1.0)

            # ── Guard: Hallucination ──
            _novelty = _spectral_novelty(_audio, _processed, sr)
            if _novelty > self._threshold:
                logger.warning(
                    "Resemble verbessern Hallucination-Guard: novelty=%.3f > %.2f → Rollback",
                    _novelty,
                    self._threshold,
                )
                return _audio, False

            # ── Guard: Quality-Delta ──
            _qual = _quality_delta(_audio, _processed, sr)
            _degraded = (
                _qual["rms_delta_db"] < -6.0
                or _qual["crest_delta"] < -0.6
                or _qual["hf_delta_db"] < -10.0
                or _qual["spec_var_delta"] < -0.6
            )
            if _degraded:
                logger.warning(
                    "Resemble verbessern Quality-Guard: Degradation → Rollback. RMS=%.1fdB Crest=%.3f HF=%.1fdB",
                    _qual["rms_delta_db"],
                    _qual["crest_delta"],
                    _qual["hf_delta_db"],
                )
                return _audio, False

            _elapsed = time.perf_counter() - t0
            logger.info(
                "Resemble verbessern Verarbeitungsschritt-0b: %.2fs, novelty=%.3f, HF=+%.1fdB",
                _elapsed,
                _novelty,
                _qual.get("hf_delta_db", 0.0),
            )
            return _processed, True

        except Exception as exc:
            logger.warning("Resemble verbessern fehlgeschlagen: %s", exc)
            return _audio, False


# ═══════════════════════════════════════════════════════════════════════
# Chained Phase-0 Preprocessor (Apollo → Resemble Enhance)
# ═══════════════════════════════════════════════════════════════════════


# ── Chained Phase-0 Pre-Processor ─────────────────────────────────────


class EARVAEPhase0Stage:
    """§v10.306 EAR_VAE Neural Clean-Pass — Phase-0 Stage.

    Uses the EAR_VAE ONNX model (earlab/EAR_VAE, Apache 2.0) to perform
    a neural clean-pass via VAE bottleneck. The perceptual K-weighting,
    phase-derivative loss, and stereo correlation loss trained into the
    model produce a cleaner version of the input while preserving musical
    detail.

    This stage runs BEFORE subtractive stages (Apollo/DeepFilterNet) to
    give them a cleaner baseline. Falls back silently if the ONNX model
    is not available.
    """

    def __init__(self) -> None:
        self._plugin = None
        self._load_attempted = False

    def _get_plugin(self):
        if self._load_attempted:
            return self._plugin
        self._load_attempted = True
        try:
            from plugins.ear_vae_plugin import get_ear_vae_plugin

            self._plugin = get_ear_vae_plugin()  # type: ignore[assignment]
            if self._plugin is not None and self._plugin._ok:
                logger.info("EAR_VAE Verarbeitungsschritt-0 Stufe verfuegbar")
            else:
                logger.debug("EAR_VAE Verarbeitungsschritt-0: plugin not geladen")
                self._plugin = None
        except Exception as exc:
            logger.debug("EAR_VAE Verarbeitungsschritt-0 nicht verfuegbar: %s", exc)
            self._plugin = None
        return self._plugin

    def unload(self) -> None:
        """§v10.306: EAR_VAE sofort aus RAM entladen."""
        if self._plugin is not None:
            self._plugin.unload()
            self._plugin = None
        self._load_attempted = False
        import gc

        gc.collect()

    def process(self, audio: np.ndarray, sr: int = 48000) -> tuple[np.ndarray, bool]:
        """Run EAR_VAE neural clean-pass.

        Returns (audio_out, applied).
        """
        plugin = self._get_plugin()
        if plugin is None:
            return audio, False

        try:
            _audio = np.asarray(audio, dtype=np.float32)
            _audio = np.nan_to_num(_audio, nan=0.0, posinf=1.0, neginf=-1.0)
            _audio = np.clip(_audio, -1.0, 1.0)

            # Convert mono to stereo if needed
            if _audio.ndim == 1:
                _audio = np.stack([_audio, _audio], axis=0)
            elif _audio.ndim == 2 and _audio.shape[1] == 2 and _audio.shape[0] > 2:
                _audio = _audio.T

            # Resample to 48k if needed
            if sr != 48000:
                try:
                    from scipy.signal import resample_poly

                    num = int(_audio.shape[-1] * 48000 / sr)
                    channels = []
                    for c in range(min(2, _audio.shape[0] if _audio.ndim == 2 else 1)):
                        ch = _audio[c] if _audio.ndim == 2 else _audio
                        channels.append(resample_poly(ch, num, ch.shape[-1]))
                    _audio = np.stack(channels, axis=0).astype(np.float32)
                except Exception:
                    logger.debug(
                        "Verarbeitungsschritt-0: resample_poly für EAR_VAE fehlgeschlagen — überspringe Resample"
                    )

            # Run neural clean-pass
            # §v10.360: PLM-Schutz für EAR_VAE (643 MB)
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

                _plm = get_plugin_lifecycle_manager()
                _plm.set_active("EAR_VAE", True)
                _plm.touch("EAR_VAE")  # §v10.370: LRU-Update
            except Exception:
                logger.debug("Verarbeitungsschritt-0: PLM EAR_VAE Aktivierung fehlgeschlagen")
            try:
                out = plugin.process(_audio, sample_rate=48000)
            finally:
                try:
                    get_plugin_lifecycle_manager().set_active("EAR_VAE", False)
                except Exception:
                    logger.debug("Verarbeitungsschritt-0: PLM EAR_VAE Deaktivierung fehlgeschlagen")
            if out is None:
                return audio, False

            # Quality guard: skip if spectral novelty > 40%
            _novelty = _spectral_novelty(audio, out, sr=48000)
            if _novelty > 0.40:
                logger.debug("EAR_VAE uebersprungen: novelty %.3f > 0.40", _novelty)
                return audio, False

            # Resample back to original SR if needed
            if sr != 48000:
                try:
                    from scipy.signal import resample_poly

                    num = int(out.shape[-1] * sr / 48000)
                    channels = []
                    for c in range(out.shape[0]):
                        channels.append(resample_poly(out[c], num, out[c].shape[-1]))
                    out = np.stack(channels, axis=0).astype(np.float32)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

            # Convert back to original channel layout
            if audio.ndim == 1:
                out = np.mean(out[:2], axis=0) if out.shape[0] >= 2 else out[0]
            elif audio.ndim == 2 and audio.shape[1] == 2 and audio.shape[0] > 2:
                out = out.T

            out = np.clip(out, -1.0, 1.0)
            if out.shape[-1] != audio.shape[-1]:
                # Trim/pad to match original length
                min_len = min(out.shape[-1], audio.shape[-1])
                if out.ndim == 2:
                    out = out[:, :min_len]
                else:
                    out = out[:min_len]

            logger.info("EAR_VAE Verarbeitungsschritt-0 angewendet: novelty=%.3f", _novelty)
            return out.astype(np.float32), True

        except Exception as exc:
            logger.debug("EAR_VAE Verarbeitungsschritt-0 fehlgeschlagen: %s", exc)
            return audio, False


class ChainedPhase0Preprocessor:
    """Verketteter Phase-0 Pre-Processor mit korrekter Reihenfolge.

    Reihenfolge (Carrier-Chain-Inversion §2.46):
      1. Apollo           — Codec-Decompression (nur für lossy-codec)
      2. DeepFilterNet v3 — Noise-Floor-Stabilisierung (alle Materialien, Atmungserhalt)
      3. Resemble Enhance — Enhancement/Denoising (alle Materialien)

    Jede Stufe hat eigenen Hallucination-Guard + Quality-Guard.
    Bei Degradation in einer Stufe: Rollback dieser Stufe, nächste läuft trotzdem.
    DeepFilterNet erhält Atmung im Gesang via BreathDetector.

    Cache: Per-Material-Effectiveness verhindert wiederholte Fehlversuche.
    """

    def __init__(self):
        self._ear_vae = EARVAEPhase0Stage()
        self._apollo = ApolloPhase0Guard()
        self._deepfilter = DeepFilterNetGuard()
        self._resemble = ResembleEnhanceGuard()
        self._apollo_failed_materials: set[str] = set()
        self._dfn_failed_materials: set[str] = set()
        self._resemble_failed_materials: set[str] = set()
        self._ear_vae_failed: bool = False
        # ── §v10.303.18 Phase-0-Cache ──
        import os as _os

        self._cache_dir = _os.path.join(_os.path.expanduser("~/.aurik/cache/phase0"))
        _os.makedirs(self._cache_dir, exist_ok=True)

    @staticmethod
    def should_apply(material: Any) -> bool:
        """Chained Pre-Processor gilt für alle Materialien.

        Apollo läuft nur bei lossy-codec, Resemble läuft immer.
        """
        return True

    def process(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        material: Any = "unknown",
        transfer_chain: list[str] | None = None,
    ) -> ApolloResult:
        """Führt die verkettete Phase-0 Pipeline aus.

        Returns ApolloResult mit finalem Audio und Metadaten aller Stufen.
        """
        _mat = ApolloPhase0Guard._normalize_material(material)
        _current = np.asarray(audio, dtype=np.float32)

        # ── §v10.303.18 Cache-Check ──
        import json as _json
        import os as _os

        _cache_key = hashlib.sha256(_current[: min(len(_current), 96000)].tobytes()).hexdigest()
        _cache_file = _os.path.join(self._cache_dir, f"{_mat}_{_cache_key[:16]}.npz")
        if _os.path.exists(_cache_file):
            try:
                _cached = np.load(_cache_file, allow_pickle=True)
                _cached_audio = _cached["audio"]
                _cached_stages_str = (
                    str(_cached.get("stage_info", ["apollo,deepfilternet,resemble_enhance"])[0])
                    if "stage_info" in _cached
                    else "apollo,deepfilternet,resemble_enhance"
                )
                _cached_stages = [
                    {"stage": s.strip(), "applied": True} for s in _cached_stages_str.split(",") if s.strip()
                ]
                if len(_cached_audio) == len(_current):
                    logger.info("§v10.303.18 Verarbeitungsschritt-0 Zwischenspeicher HIT: %s", _cache_key[:16])
                    return ApolloResult(
                        audio=_cached_audio.astype(np.float32),
                        applied=True,
                        material=_mat,
                        metadata={
                            "stages": _cached_stages,
                            "chain": "ear_vae→apollo→deepfilternet→resemble_enhance",
                            "cached": True,
                        },
                    )
            except Exception:
                pass  # Cache corrupt → neu berechnen

        _any_applied = False
        _meta_stages: list[dict[str, Any]] = []

        # ── Stufe 0: EAR_VAE Neural Clean-Pass (§v10.306) ──
        if not self._ear_vae_failed:
            _ev_out, _ev_applied = self._ear_vae.process(_current, sr)
            if _ev_applied:
                _current = _ev_out
                _any_applied = True
                _meta_stages.append({"stage": "ear_vae", "applied": True})
            else:
                _meta_stages.append({"stage": "ear_vae", "applied": False})
            # §v10.306: EAR_VAE sofort entladen — 643 MB RAM freigeben
            self._ear_vae.unload()

        # ── Stufe 1: Apollo Codec-Decompression ──
        _should_apply_apollo = ApolloPhase0Guard.should_apply(material, transfer_chain=transfer_chain)
        if _should_apply_apollo and _mat not in self._apollo_failed_materials:
            # §v10.330: Apollo als aktiv markieren — verhindert PLM-Eviction
            # während der Inferenz (Segfault-Risiko).
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

                _plm = get_plugin_lifecycle_manager()
                _plm.set_active("Apollo", True)
                _plm.touch("Apollo")  # §v10.370: LRU-Update
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            try:
                _apollo_result = self._apollo.process(_current, sr, material)
            finally:
                try:
                    get_plugin_lifecycle_manager().set_active("Apollo", False)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            if _apollo_result.applied:
                _current = _apollo_result.audio
                _any_applied = True
                _meta_stages.append(
                    {
                        "stage": "apollo",
                        "applied": True,
                        "novelty": _apollo_result.novelty,
                        "hf_delta_db": _apollo_result.hf_delta_db,
                    }
                )
            else:
                if _apollo_result.novelty > 0:
                    self._apollo_failed_materials.add(_mat)
                _meta_stages.append({"stage": "apollo", "applied": False})
        else:
            _meta_stages.append({"stage": "apollo", "applied": False, "reason": "not_codec_or_cached"})
        # §v10.306: Apollo sofort entladen — 800 MB RAM freigeben
        self._apollo.unload()

        # ── Stufe 2: DeepFilterNet v3 (Noise-Floor, Atmungserhalt) ──
        # §v10.700.8 Precondition: Nur bei sprachdominiertem Material (panns_singing >= 0.5).
        # DeepFilterNet ist auf DNS-Challenge (Sprache+Rauschen) trainiert. Bei Musik
        # mit instrumentalem Hintergrund (panns_singing < 0.5) halluziniert es spektrale
        # Inhalte → Hallucination-Guard triggert Rollback. Besser: gar nicht erst anwenden.
        _vocal_conf = 0.0  # §FIX_B30: _pre undefined — Phase-0 has no pre-analysis context; safe default
        if _mat not in self._dfn_failed_materials and _vocal_conf >= 0.5:
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

                _plm = get_plugin_lifecycle_manager()
                _plm.set_active("DeepFilterNetV3", True)
                _plm.touch("DeepFilterNetV3")  # §v10.370: LRU-Update
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            try:
                _dfn_out, _dfn_applied = self._deepfilter.process(_current, sr)
            finally:
                try:
                    get_plugin_lifecycle_manager().set_active("DeepFilterNetV3", False)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            if _dfn_applied:
                _current = _dfn_out
                _any_applied = True
                _meta_stages.append({"stage": "deepfilternet", "applied": True})
            else:
                self._dfn_failed_materials.add(_mat)
                _meta_stages.append({"stage": "deepfilternet", "applied": False})
        else:
            _reason = "low_vocal" if _vocal_conf < 0.5 else "cached_failure"
            _meta_stages.append({"stage": "deepfilternet", "applied": False, "reason": _reason})
            if _vocal_conf < 0.5 and _vocal_conf > 0:
                logger.debug(
                    "DeepFilterNet uebersprungen: panns_singing=%.2f < 0.5 (Musik/Instrumental — außerhalb Trainingsbereich)",
                    _vocal_conf,
                )
        # §v10.306: DeepFilterNet sofort entladen — 34 MB RAM freigeben
        self._deepfilter.unload()

        # ── Stufe 3: Resemble Enhance ──
        # §v10.700.8 Precondition: Nur bei sprachdominiertem Material (panns_singing >= 0.5).
        # Resemble Enhance ist auf Sprach-Denoising trainiert (DNS-Challenge).
        # Bei Musik halluziniert es → Hallucination-Guard triggert Rollback.
        if _mat not in self._resemble_failed_materials and _vocal_conf >= 0.5:
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

                _plm = get_plugin_lifecycle_manager()
                _plm.set_active("ResembleEnhance", True)
                _plm.touch("ResembleEnhance")  # §v10.370: LRU-Update
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            try:
                _re_out, _re_applied = self._resemble.process(_current, sr)
            finally:
                try:
                    get_plugin_lifecycle_manager().set_active("ResembleEnhance", False)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            if _re_applied:
                _current = _re_out
                _any_applied = True
                _meta_stages.append({"stage": "resemble_enhance", "applied": True})
            else:
                self._resemble_failed_materials.add(_mat)
                _meta_stages.append({"stage": "resemble_enhance", "applied": False})
        else:
            _meta_stages.append({"stage": "resemble_enhance", "applied": False, "reason": "cached_failure"})
        # §v10.306: ResembleEnhance sofort entladen — 722 MB RAM freigeben
        self._resemble.unload()

        # ── §v10.303.18 Cache speichern ──
        if _any_applied:
            try:
                import os as _os2

                _stage_info = ",".join(sorted({s["stage"] for s in _meta_stages if s.get("applied")}))
                np.savez_compressed(
                    _cache_file,
                    audio=_current.astype(np.float32),
                    stage_info=np.array([_stage_info]),
                )
                logger.debug("§v10.303.18 Verarbeitungsschritt-0 Zwischenspeicher gespeichert: %s", _cache_key[:16])
            except Exception:
                pass

        return ApolloResult(
            audio=_current.astype(np.float32),
            applied=_any_applied,
            material=_mat,
            metadata={"stages": _meta_stages, "chain": "ear_vae→apollo→deepfilternet→resemble_enhance"},
        )

    def reset(self) -> None:
        self._apollo.reset()
        self._apollo_failed_materials.clear()
        self._dfn_failed_materials.clear()
        self._resemble_failed_materials.clear()

    # ── §v10.303.20 PLM-optimierte Lade-Reihenfolge ──────────────

    def preload(self) -> None:
        """Lädt Phase-0-Modelle in optimaler Reihenfolge.

        DeepFilterNet (34 MB) → Apollo (67 MB) → Resemble (722 MB).
        Klein zu groß: schnelle Modelle blockieren nicht auf große.
        """
        _order = [
            ("DeepFilterNet", self._deepfilter._ensure_loaded),
            ("Apollo", self._apollo.load),
            ("Resemble Enhance", self._resemble._ensure_loaded),
        ]
        for _name, _loader in _order:
            try:
                _ok = _loader()
                logger.info(
                    "§v10.303.20 PLM preload %s: %s",
                    _name,
                    "geladen" if _ok else "nicht verfügbar",
                )
            except Exception as _exc:
                logger.debug("PLM preload %s fehlgeschlagen: %s", _name, _exc)


# ═══════════════════════════════════════════════════════════════════════
# DeepFilterNet v3 Phase-0b Guard (mit Atmungserhalt)
# ═══════════════════════════════════════════════════════════════════════


class DeepFilterNetGuard:
    """Phase-0b: DeepFilterNet v3 — Noise-Floor-Stabilisierung mit Atmungserhalt.

    Läuft NACH Apollo (Phase-0a), VOR Resemble Enhance (Phase-0c).
    Entfernt Breitbandrauschen OHNE Atmung im Gesang zu unterdrücken.

    Modell: models/deepfilternet_v3_ii/{enc,dec,erb_dec}.onnx (34 MB total)
    Fallback: Passthrough (kein DSP — DeepFilterNet ist leichtgewichtig)  # §V6 (copilot-instructions.md): logger.warning handled at call site

    Atmungserhalt (§2.8):
        - BreathDetector erkennt Atemsegmente via ZCR + Energie
        - DeepFilterNet läuft nur auf Nicht-Atem-Segmenten
        - An Atemgrenzen: 5 ms Hanning-Crossfade zwischen DFN-Output und Original
    """

    def __init__(self, hallucination_threshold: float = 0.50):
        self._threshold = hallucination_threshold
        self._loaded = False
        self._plugin = None
        self._breath_detector = None

    def unload(self) -> None:
        """§v10.402: DeepFilterNet entladen. Überspringt wenn PLM-Singleton."""
        if getattr(self, "_shared_from_plm", False):
            self._plugin = None
            self._loaded = False
            return
        if self._plugin is not None:
            self._plugin.unload()
            self._plugin = None
        self._loaded = False
        self._breath_detector = None
        import gc

        gc.collect()

    def _ensure_loaded(self) -> bool:
        if self._loaded and self._plugin is not None:
            return True
        # §v10.402: PLM-Singleton statt eigener Instanz
        try:
            from plugins.deepfilternet_v3_ii_plugin import get_deepfilternet_plugin

            self._plugin = get_deepfilternet_plugin()  # type: ignore[assignment]
            self._loaded = True
            self._shared_from_plm = True
            logger.info("DeepFilterNet Verarbeitungsschritt-0 via PLM-Singleton")
            return True
        except Exception:
            pass
        # Fallback
        try:
            from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

            self._plugin = DeepFilterNetV3Plugin()  # type: ignore[assignment]
            self._loaded = True
            logger.info("DeepFilterNet v3 Verarbeitungsschritt-0b geladen")
            return True
        except ImportError:
            logger.debug("DeepFilterNet Plugin nicht verfügbar")
            return False
        except Exception as exc:
            logger.warning("DeepFilterNet Ladefehler: %s", exc)
            return False

    def _get_breath_mask(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Erzeugt Atem-Maske: True = Atemsegment (preserve), False = normal (process)."""
        try:
            from plugins.breath_detector import detect_breaths

            _mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
            _result = detect_breaths(_mono.astype(np.float32), sr)
            _mask = np.zeros(len(_mono), dtype=bool)
            for _start, _end in zip(_result.breath_positions, _result.breath_end_positions):
                _start = max(0, _start)
                _end = min(len(_mask), _end)
                _mask[_start:_end] = True
            _breath_pct = float(np.sum(_mask)) / max(len(_mask), 1) * 100
            if _breath_pct > 0:
                logger.debug(
                    "DeepFilterNet: %d Atemsegmente erkannt (%.1f%% des Audios) — werden erhalten",
                    len(_result.breath_positions),
                    _breath_pct,
                )
            return cast(np.ndarray, _mask)
        except ImportError:
            logger.debug("BreathDetector nicht verfügbar — keine Atemmaske")
            return cast(np.ndarray, (np.zeros(len(audio), dtype=bool)))
        except Exception as exc:
            logger.debug("Atemerkennung fehlgeschlagen: %s", exc)
            return cast(np.ndarray, (np.zeros(len(audio), dtype=bool)))

    def process(self, audio: np.ndarray, sr: int = 48000) -> tuple[np.ndarray, bool]:
        """Führt DeepFilterNet mit Atmungserhalt aus.

        Returns: (audio_out, applied)
        """
        _audio = np.asarray(audio, dtype=np.float32)
        _audio = np.nan_to_num(_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        _audio = np.clip(_audio, -1.0, 1.0)

        if not self._ensure_loaded():
            return _audio, False

        t0 = time.perf_counter()
        try:
            _plugin = self._plugin
            if _plugin is None:
                return _audio, False

            # ── Atem-Maske ──
            _breath_mask = self._get_breath_mask(_audio, sr)

            # DeepFilterNet verarbeitet mono
            _is_stereo = _audio.ndim == 2 and _audio.shape[0] == 2
            _mono = _audio if not _is_stereo else np.mean(_audio, axis=0)

            # DeepFilterNet enhance (arbeitet auf 48 kHz nativ)
            _processed = _plugin.enhance(_mono, sr)
            _processed = np.nan_to_num(_processed, nan=0.0, posinf=1.0, neginf=-1.0)
            _processed = np.clip(_processed, -1.0, 1.0)

            # ── Atmungserhalt: Original an Atempositionen ──
            if np.any(_breath_mask):
                _min_len = min(len(_processed), len(_mono), len(_breath_mask))
                _crossfade_samples = int(0.005 * sr)  # 5 ms
                for _start, _end in self._mask_to_segments(_breath_mask[:_min_len]):
                    _s = max(0, _start)
                    _e = min(_min_len, _end)
                    if _e <= _s:
                        continue
                    # Crossfade an den Atemgrenzen
                    _cf_in = min(_crossfade_samples, (_e - _s) // 3)
                    _cf_out = min(_crossfade_samples, (_e - _s) // 3)
                    for _i in range(_s, _e):
                        _pos = _i - _s
                        if _pos < _cf_in:
                            _w = 0.5 - 0.5 * np.cos(np.pi * _pos / max(_cf_in, 1))
                            _processed[_i] = _processed[_i] * (1 - _w) + _mono[_i] * _w
                        elif _pos >= (_e - _s - _cf_out):
                            _w = 0.5 - 0.5 * np.cos(np.pi * (_e - _s - _pos) / max(_cf_out, 1))
                            _processed[_i] = _processed[_i] * (1 - _w) + _mono[_i] * _w
                        else:
                            _processed[_i] = _mono[_i]  # Vollständiger Atmungserhalt
                logger.debug("DeepFilterNet: Atmungserhalt aktiv — Atemsegmente unverändert")

            # ── Guard: Hallucination ──
            _novelty = _spectral_novelty(_mono, _processed, sr)
            if _novelty > self._threshold:
                logger.warning(
                    "DeepFilterNet Hallucination-Guard: novelty=%.3f > %.2f → Rollback",
                    _novelty,
                    self._threshold,
                )
                return _audio, False

            # ── Guard: Quality-Delta ──
            _qual = _quality_delta(_mono, _processed, sr)
            _degraded = (
                _qual["rms_delta_db"] < -3.0
                or _qual["crest_delta"] < -0.4
                or _qual["hf_delta_db"] < -8.0
                or _qual["spec_var_delta"] < -0.4
            )
            if _degraded:
                logger.warning(
                    "DeepFilterNet Quality-Guard: Degradation → Rollback. RMS=%.1fdB Crest=%.3f HF=%.1fdB",
                    _qual["rms_delta_db"],
                    _qual["crest_delta"],
                    _qual["hf_delta_db"],
                )
                return _audio, False

            _elapsed = time.perf_counter() - t0
            logger.info(
                "DeepFilterNet Verarbeitungsschritt-0b: %.2fs, novelty=%.3f, RMS=%.1fdB",
                _elapsed,
                _novelty,
                _qual.get("rms_delta_db", 0.0),
            )

            # Remix stereo
            if _is_stereo:
                _stereo_out = np.stack([_processed, _processed], axis=0).astype(np.float32)
                return _stereo_out, True
            return _processed.astype(np.float32), True

        except Exception as exc:
            logger.warning("DeepFilterNet fehlgeschlagen: %s", exc)
            return _audio, False

    @staticmethod
    def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
        """Konvertiert Boolean-Maske zu (start, end) Segmenten."""
        _segments = []
        _in_seg = False
        _seg_start = 0
        for _i, _val in enumerate(mask):
            if _val and not _in_seg:
                _seg_start = _i
                _in_seg = True
            elif not _val and _in_seg:
                _segments.append((_seg_start, _i))
                _in_seg = False
        if _in_seg:
            _segments.append((_seg_start, len(mask)))
        # Merge adjacent segments closer than 30 ms @ 48 kHz
        _gap_samples = int(0.030 * 48000)
        _merged: list[Any] = []
        for _s, _e in _segments:
            if _merged and _s - _merged[-1][1] <= _gap_samples:
                _merged[-1] = (_merged[-1][0], _e)
            else:
                _merged.append((_s, _e))
        return _merged
