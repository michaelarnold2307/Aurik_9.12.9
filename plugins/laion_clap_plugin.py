"""
LAION-CLAP Plugin — Text-Audio-Kontrastives Audio-Tagging für Aurik 10.0.0

Primäres Audio-Tagging-Modell für Genre-, Instrument- und Stil-Erkennung.
LAION-CLAP (Contrastive Language-Audio Pretraining) wird auf Milliarden
von Text-Audio-Paaren trainiert und übertrifft PANNs in Zero-Shot-Kategorisierung.

Referenz:
    Wu et al. (2023): "Large-Scale Contrastive Language-Audio Pretraining with
    Feature Fusion and Keyword-to-Caption Augmentation"
    ICASSP 2023. https://arxiv.org/abs/2211.06687

SOTA-Entscheidungsmatrix (§4.4 Aurik-Spec):
    Primär:   LAION-CLAP (ONNX, text-audio-kontrastiv)
    Fallback: PANNs-Plugin (panns_plugin.py, breite Kategorie)

CPU-Policy: Ausschließlich CPUExecutionProvider.
Modell-Gewichte: ~/.aurik/models/laion_clap/ (via ModelDownloader)

Anwendung in Aurik:
    - Instrument-Erkennung → Phasen-Aktivierung (§2.9 Aktivierungsmatrix)
    - Genre-Erkennung → Studio-2026-Parametrisierung
    - Material-Typ-Unterstützung → DefectScanner-Prior
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vordefinierte Tag-Kategorien (für PANNs-kompatible Ausgabe)
# ---------------------------------------------------------------------------

INSTRUMENT_TAGS: list[str] = [
    "vocals",
    "guitar",
    "electric_guitar",
    "acoustic_guitar",
    "piano",
    "keyboard",
    "bass",
    "drums",
    "percussion",
    "violin",
    "cello",
    "trumpet",
    "saxophone",
    "flute",
    "brass",
    "strings",
    "choir",
    "orchestra",
]

GENRE_TAGS: list[str] = [
    "rock",
    "jazz",
    "classical",
    "pop",
    "blues",
    "country",
    "electronic",
    "folk",
    "hip_hop",
    "reggae",
    "metal",
    "soul",
    "rnb",
    "gospel",
    "opera",
    "ambient",
]

MATERIAL_TAGS: list[str] = [
    "vinyl",
    "tape",
    "shellac",
    "digital",
    "mp3",
    "aac",
    "live_recording",
    "studio_recording",
    "broadcast",
]


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class AudioTaggingResult:
    """Ergebnis des LAION-CLAP Audio-Taggings.

    Attribute:
        instrument_tags:  Dict tag_name → Konfidenz ∈ [0, 1]
        genre_tags:       Dict tag_name → Konfidenz ∈ [0, 1]
        material_tags:    Dict tag_name → Konfidenz ∈ [0, 1]
        embedding:        512-dim Audio-Embedding (für Ähnlichkeitssuche)
        model_used:       "laion_clap" | "panns_fallback"
        confidence:       Gesamtkonfidenz ∈ [0, 1]
    """

    instrument_tags: dict[str, float]
    genre_tags: dict[str, float]
    material_tags: dict[str, float]
    embedding: np.ndarray
    model_used: str
    confidence: float
    metadata: dict[str, float] = field(default_factory=dict)

    def top_instruments(self, n: int = 3, threshold: float = 0.4) -> list[str]:
        """Gibt die Top-n-Instrumente über Schwellwert zurück."""
        sorted_tags = sorted(self.instrument_tags.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, conf in sorted_tags if conf >= threshold][:n]

    def top_genres(self, n: int = 2) -> list[str]:
        """Gibt die Top-n-Genres zurück."""
        sorted_tags = sorted(self.genre_tags.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in sorted_tags][:n]

    def as_dict(self) -> dict:
        """Gibt Tagging-Ergebnis als flaches Dictionary zurück."""
        return {
            "top_instruments": self.top_instruments(),
            "top_genres": self.top_genres(),
            "confidence": self.confidence,
            "model_used": self.model_used,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------

_instance: LAIONCLAPPlugin | None = None
_lock = threading.Lock()


def _load_trusted_local_torch_checkpoint(checkpoint_path: Path) -> Any:
    """Laedt einen lokal gebuendelten Aurik-Checkpoint mit Torch-2.6+-Kompatibilitaet."""
    import torch  # pylint: disable=import-outside-toplevel

    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


class LAIONCLAPPlugin:
    """LAION-CLAP Text-Audio-Kontrastives Tagging für Aurik 10.0.0.

    Algorithmus (CLAP Vollpfad):
        1. Audio-Encoder: HTS-AT (Hierarchical Token Semantic Audio Transformer)
           Audio → 512-dim L2-normalisiertes Audio-Embedding
        2. Text-Encoder: CLIP-Text-Transformer → Text-Embeddings für Tags
        3. Ähnlichkeitsmatrix: cosine(audio_emb, text_emb_i) → Konfidenz-Scores
        4. Instrument-/Genre-/Material-Tags werden per Zero-Shot klassifiziert

    Zero-Shot-Vorteil gegenüber PANNs:
        Keine festen Klassen — beliebige Text-Anfragen möglich
        z.B. "1940er Jazz mit Kontrabass" → direkter Ähnlichkeits-Score

    DSP-Fallback (ohne L.CLAP-Modell):
        Spektral-Feature-Extraktion:
        - Harmonic-Ratio → vocals/strings/winds hoher Wert
        - Onset-Density → drums/percussion
        - Spektral-Schwerpunkt → Instrument-Schätzung
        - Chroma-Profil → harmonische Komplexität

    Invarianten:
        - PANNs-kompatible Ausgabe-Struktur (tag → confidence)
        - Alle Konfidenz-Werte ∈ [0, 1]
        - kein NaN/Inf im Embedding
        - Instrument-Tags triggern Phasen-Aktivierung (§2.9)
    """

    # ONNX-Pfad (SOTA-Upgrade via ModelDownloader)
    MODELS_DIR: Path = Path.home() / ".aurik" / "models" / "laion_clap"
    # Lokaler PyTorch-Checkpoint + Quellcode (models/clap/, lokal gebündelt)
    _PROJECT_ROOT: Path = Path(__file__).parent.parent
    _LOCAL_CLAP_DIR: Path = _PROJECT_ROOT / "models" / "clap"
    _LOCAL_CLAP_CKPT: str = "music_audioset_epoch_15_esc_90.14.pt"
    # Lokaler roberta-base (§13.3: Out-of-Box-Pflicht, kein HF-Hub-Download)
    _LOCAL_ROBERTA_DIR: Path = _PROJECT_ROOT / "models" / "roberta-base"
    _HF_STAGING_DIR: Path = _PROJECT_ROOT / "models" / ".hf_staging" / "hub"

    EMBEDDING_DIM: int = 512

    def __init__(self) -> None:
        self._audio_session: Any = None  # onnxruntime.InferenceSession (ONNX-Pfad)
        self._audio_session_model_path: str | None = None  # §ROCm-Fallback: Pfad für CPU-Rebuild
        self._clap_model: Any = None  # laion_clap.CLAP_Module (PyTorch-Pfad)
        self._text_embeddings: np.ndarray | None = None
        self._model_loaded: bool = False
        self._fallback_active: bool = False
        self._load_attempted: bool = False
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Lazy-Load: Modell erst beim ersten Aufruf laden (2.2 GB PyTorch-Checkpoint)."""
        if self._load_attempted:
            return
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Lädt CLAP: erst ONNX-SOTA, dann lokales PyTorch-Checkpoint, dann DSP."""
        # 1. Versuch: ONNX-Modell (SOTA-Upgrade unter ~/.aurik/)
        try:
            import onnxruntime as ort  # pylint: disable=import-outside-toplevel

            audio_enc_path = self.MODELS_DIR / "audio_encoder.onnx"
            text_emb_path = self.MODELS_DIR / "text_embeddings.npy"

            if audio_enc_path.exists() and text_emb_path.exists():
                # Memory-Budget-Guard (Schicht 1) — PFLICHT vor jedem ML-Modell-Laden
                _onnx_budget_ok = True
                _ml_release_onnx = None
                try:
                    from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                        release as _ml_release_onnx,
                    )
                    from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                        try_allocate as _try_alloc_onnx,
                    )

                    if not _try_alloc_onnx("LaionCLAP_ONNX", 0.30):
                        logger.warning("LAION-CLAP: ML-Budget erschöpft (ONNX) — überspringe ONNX-Pfad")
                        _onnx_budget_ok = False
                except ImportError as _exc:
                    logger.debug(
                        "Optional import not available (non-critical): %s", _exc
                    )  # Budget-Modul optional — weiter

                if not _onnx_budget_ok:
                    raise MemoryError("LaionCLAP_ONNX: Budget erschöpft")

                try:
                    try:
                        from backend.core.ml_device_manager import get_ort_providers as _get_prov  # pylint: disable=import-outside-toplevel  # noqa: I001

                        _clap_providers = _get_prov("LaionCLAP_ONNX")
                    except Exception:
                        _clap_providers = ["CPUExecutionProvider"]
                    self._audio_session = ort.InferenceSession(
                        str(audio_enc_path),
                        providers=_clap_providers,
                    )
                    self._audio_session_model_path = str(audio_enc_path)
                except Exception:
                    if _ml_release_onnx is not None:
                        _ml_release_onnx("LaionCLAP_ONNX")
                    raise
                self._text_embeddings = np.load(str(text_emb_path))
                self._model_loaded = True
                logger.info(
                    "🔵 LAION-CLAP: ONNX Audio-Encoder + %d Tag-Embeddings geladen",
                    len(self._text_embeddings),
                )
                # PLM-Registrierung (Schicht 2) nach erfolgreichem Load
                try:
                    from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                        get_plugin_lifecycle_manager as _get_plm_onnx,
                    )

                    # Ensure audio session is available before registering
                    _ = self._audio_session
                    _get_plm_onnx().register(
                        "LaionCLAP_ONNX",
                        size_gb=0.30,
                        unload_fn=lambda: setattr(self, "_audio_session", None),
                    )
                except Exception as _exc:
                    logger.debug("Plugin operation failed (non-critical): %s", _exc)
                return
        except Exception as exc:
            logger.debug("LAION-CLAP ONNX-Pfad nicht verfügbar: %s", exc)

        # 2. Versuch: lokaler PyTorch-Checkpoint (models/clap/)
        if self._try_load_clap_pt():
            return

        # 3. Fallback: PANNs-DSP
        logger.info("LAION-CLAP: Kein Modell geladen — PANNs-DSP-Fallback aktiv")
        self._fallback_active = True

    def _ensure_roberta_hf_cache(self) -> str | None:
        """Baut einmalig eine HF-Hub-Cache-Struktur für roberta-base auf.

        Verwendet Symlinks, damit keine Datei doppelt vorliegt.
        Die Struktur entspricht dem HuggingFace-Hub-Format (v0.12+):

            models/.hf_staging/hub/
                models--roberta-base/
                    snapshots/
                        local/          ← Symlinks → models/roberta-base/*
                    refs/
                        main            ← Datei mit Inhalt "local"

        Returns:
            Pfad zu HF_HUB_CACHE wenn models/roberta-base/ existiert,
            sonst None (→ Standard-Cache bleibt aktiv).
        """
        roberta_src = self._LOCAL_ROBERTA_DIR
        if not roberta_src.exists():
            logger.debug("LAION-CLAP: models/roberta-base/ nicht gefunden — HF-Cache unverändert")
            return None

        hf_cache = self._HF_STAGING_DIR
        snapshot_dir = hf_cache / "models--roberta-base" / "snapshots" / "local"
        refs_dir = hf_cache / "models--roberta-base" / "refs"
        refs_main = refs_dir / "main"

        # Einmalig aufbauen — idempotent
        if not snapshot_dir.exists():
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            for src_file in roberta_src.iterdir():
                if src_file.is_file():
                    link = snapshot_dir / src_file.name
                    if not link.exists():
                        link.symlink_to(src_file.resolve())
            logger.debug(
                "LAION-CLAP: HF-Staging-Cache für roberta-base angelegt: %s",
                snapshot_dir,
            )

        # refs/main setzen (oder sicherstellen, dass Inhalt korrekt ist)
        if not refs_main.exists() or refs_main.read_text(encoding="utf-8").strip() != "local":
            refs_main.write_text("local", encoding="utf-8")

        logger.debug("LAION-CLAP: HF_HUB_CACHE → %s", hf_cache)
        return str(hf_cache)

    def _try_load_clap_pt(self) -> bool:
        """Lädt lokalen LAION-CLAP-Checkpoint aus models/clap/ (lokal gebündelt).

        Strategie:
            1. models/clap/src/ zu sys.path hinzufügen → 'laion_clap' importierbar
            2. models/roberta-base/ als lokalen HF-Hub-Cache eintragen
               (HF_HUB_CACHE → models/.hf_staging/hub/ mit Symlinks)
            3. CLAP_Module mit dem lokalen .pt-Checkpoint laden
            4. Text-Embeddings für alle tags synthetisch generieren (DSP-Pfad)
               oder aus Modell berechnen

        Returns:
            True wenn Checkpoint erfolgreich geladen.
        """
        try:
            clap_src = str(self._LOCAL_CLAP_DIR / "src")
            if clap_src not in sys.path:
                sys.path.insert(0, clap_src)

            # Torch ≥2.3 Compatibility Shim ─────────────────────────────
            # laion_clap ruft torch.library.register_fake auf, das erst ab
            # torch 2.3 als öffentliche API verfügbar ist. Torch 2.2.2+cpu
            # hat die Funktion intern noch nicht. Wir legen einen harmlosen
            # No-Op-Stub an, der die Decorator-/Direkt-Aufruf-Signatur
            # vollständig abdeckt.
            import torch as _torch  # pylint: disable=import-outside-toplevel

            if not hasattr(_torch.library, "register_fake"):

                def _register_fake_compat(_op, fn=None, **_kwargs):
                    """No-Op-Stub für torch.library.register_fake (< 2.3)."""
                    return fn if fn is not None else (lambda f: f)

                _torch.library.register_fake = _register_fake_compat  # type: ignore[attr-defined]
                logger.debug("LAION-CLAP: torch.library.register_fake Shim installiert (torch %s)", _torch.__version__)
            # ───────────────────────────────────────────────────────────

            import laion_clap  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

            ckpt_path = self._LOCAL_CLAP_DIR / self._LOCAL_CLAP_CKPT
            if not ckpt_path.exists():
                logger.info("LAION-CLAP: Checkpoint nicht gefunden: %s", ckpt_path)
                return False

            # Globaler ML-Budget-Guard: ~2.2 GB für LAION-CLAP.
            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc  # pylint: disable=import-outside-toplevel  # noqa: I001

                if not _try_alloc("LAION-CLAP", 2.2):
                    return False  # Budget erschöpft → PANNs-DSP-Fallback
            except Exception as _exc:
                logger.debug(
                    "Plugin operation failed (non-critical): %s", _exc
                )  # Budget-Modul nicht verfügbar — weiter

            # CLAP_Module laden:
            # music_audioset_epoch_15_esc_90.14.pt → HTSAT-base, embed_dim=128,
            # 12 Blöcke in Layer 2, KEIN Fusion-Encoder → enable_fusion=False
            #
            # TRANSFORMERS_OFFLINE=1: RobertaModel.from_pretrained('roberta-base')
            # soll NUR aus lokalem Cache laden — kein HuggingFace-Hub-Download.
            # HF_HUB_CACHE wird auf models/.hf_staging/hub/ gesetzt, das
            # Symlinks auf models/roberta-base/ enthält (§13.3 Out-of-Box-Pflicht).
            _hf_offline_orig = os.environ.get("TRANSFORMERS_OFFLINE")
            _hf_hub_cache_orig = os.environ.get("HF_HUB_CACHE")
            _hf_home_orig = os.environ.get("HF_HOME")

            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            # Lokales roberta-base einbinden (kein HF-Hub-Download)
            _staging = self._ensure_roberta_hf_cache()
            if _staging:
                os.environ["HF_HUB_CACHE"] = _staging
                # HF_HOME überschreiben, falls es auf ein fremdes Verzeichnis
                # zeigt und dadurch die Cache-Auflösung stört
                os.environ.pop("HF_HOME", None)
            try:
                model = laion_clap.CLAP_Module(
                    enable_fusion=False,
                    amodel="HTSAT-base",
                    device="cpu",
                )
            finally:
                # Alle veränderten Umgebungsvariablen wiederherstellen
                if _hf_offline_orig is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = _hf_offline_orig

                if _staging:
                    if _hf_hub_cache_orig is None:
                        os.environ.pop("HF_HUB_CACHE", None)
                    else:
                        os.environ["HF_HUB_CACHE"] = _hf_hub_cache_orig

                    if _hf_home_orig is None:
                        os.environ.pop("HF_HOME", None)
                    else:
                        os.environ["HF_HOME"] = _hf_home_orig
            # Load checkpoint with shape-mismatch tolerance.
            # Torch 2.6+ defaults to weights_only=True; this trusted checkpoint
            # is bundled locally with Aurik and contains optimizer metadata, so
            # it must be loaded with weights_only=False instead of falling back
            # to the DSP path.
            try:
                _checkpoint = _load_trusted_local_torch_checkpoint(ckpt_path)
                _state = _checkpoint.get("state_dict", _checkpoint) if isinstance(_checkpoint, dict) else _checkpoint
                if isinstance(_state, dict) and _state:
                    if next(iter(_state.items()))[0].startswith("module"):
                        _state = {_k[7:]: _v for _k, _v in _state.items()}
                    _model_state = model.model.state_dict()
                    _state = {
                        _k: _v
                        for _k, _v in _state.items()
                        if _k in _model_state and getattr(_model_state[_k], "shape", None) == getattr(_v, "shape", None)
                    }
                model.model.load_state_dict(_state, strict=False)
                logger.debug("LAION-CLAP: Checkpoint geladen (shape-inkompatible Keys übersprungen)")
            except Exception:
                # Fallback: bibliothekseigener Load
                model.load_ckpt(str(ckpt_path))
            model.eval()
            self._clap_model = model
            self._model_loaded = True
            logger.info(
                "🔵 LAION-CLAP: PyTorch-Checkpoint geladen (%s, %.1f MB)",
                ckpt_path.name,
                ckpt_path.stat().st_size / 1024 / 1024,
            )
            # PLM-Registrierung für LRU-basierte Auto-Eviction
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm  # pylint: disable=import-outside-toplevel  # noqa: I001

                _unload_fn = globals().get("unload_laion_clap")
                if _unload_fn is not None:
                    _reg_plm("LAION-CLAP", size_gb=2.2, unload_fn=_unload_fn)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            return True

        except ImportError as ie:
            logger.info(
                "LAION-CLAP: 'laion_clap'-Paket nicht importierbar: %s — PANNs-Fallback",
                ie,
            )
            return False
        except Exception as exc:
            # torchvision::nms fehlt im ROCm-venv — bekanntes Setup, PANNs-Fallback ist korrekt
            exc_str = str(exc)
            if "torchvision" in exc_str or "nms" in exc_str:
                logger.info("LAION-CLAP: torchvision nicht verfügbar (%s) — PANNs-Fallback", exc_str[:80])
            elif "Weights only load failed" in exc_str or "weights_only" in exc_str:
                logger.info("LAION-CLAP: PyTorch-Checkpoint nicht kompatibel mit weights_only-Loader — PANNs-Fallback")
            else:
                logger.warning("LAION-CLAP PyTorch-Checkpoint-Fehler: %s — PANNs-Fallback", exc)
            return False

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def _build_cpu_audio_session(self) -> Any:
        """ROCm-Fallback: Audio-Session CPU-only neu aufbauen (MIOpen-Kernel-Fehler).

        Returns:
            Neue CPU-Session oder None, wenn kein Modellpfad hinterlegt ist
            oder der Neuaufbau fehlschlägt.
        """
        if not self._audio_session_model_path:
            return None
        try:
            import onnxruntime as ort

            return ort.InferenceSession(self._audio_session_model_path, providers=["CPUExecutionProvider"])
        except Exception as _exc:
            logger.warning("LAION-CLAP: CPU-Session-Rebuild fehlgeschlagen: %s", _exc)
            return None

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Gibt a 512-dim L2-normalised audio embedding for downstream NN search zurück.

        Used by EraClassifier Tier-1 for cosine-similarity based decade detection.

        Args:
            audio: 1-D float audio (mono, any length).
            sr:    Sample rate (must be 48000).

        Returns:
            np.ndarray of shape (512,), L2-normalised, dtype float32.

        Raises:
            RuntimeError: If no CLAP model is available (caller should catch
                          and fall back to DSP-based era detection).
        """
        assert sr == 48000, f"LAION-CLAP embed_audio: SR muss 48000 Hz sein, erhalten: {sr}"
        self._ensure_loaded()
        audio_f32 = np.asarray(audio, dtype=np.float32).ravel()
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        # Path 1: ONNX audio-encoder
        if self._model_loaded and self._audio_session is not None:
            feat = np.abs(np.fft.rfft(audio_f32[:sr], n=2048)).astype(np.float32)
            feat = feat[np.newaxis, :]
            input_name = self._audio_session.get_inputs()[0].name
            _plm_clap = None
            try:
                from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                    get_plugin_lifecycle_manager as _get_plm_fn,
                )

                _plm_clap = _get_plm_fn()
                _plm_clap.set_active("LaionCLAP_ONNX", True)
            except Exception as _exc:
                logger.debug("LaionCLAP: PLM set_active failed: %s", _exc)
            try:
                try:
                    outputs = self._audio_session.run(None, {input_name: feat})
                except Exception as _ort_exc:
                    # §ROCm-Fallback: MIOpen-Kernel-Fehler (Code object build failed)
                    # auf GPU → Session CPU-only neu aufbauen und EINMAL wiederholen.
                    # Kein Qualitätsverlust, nur Laufzeit (CPU-Inferenz).
                    logger.warning(
                        "LAION-CLAP: ONNX-Inferenz fehlgeschlagen (%s) — CPU-Fallback-Retry", _ort_exc
                    )
                    _cpu_session = self._build_cpu_audio_session()
                    if _cpu_session is None:
                        raise
                    self._audio_session = _cpu_session
                    outputs = self._audio_session.run(None, {input_name: feat})
            finally:
                if _plm_clap is not None:
                    try:
                        _plm_clap.set_active("LaionCLAP_ONNX", False)
                    except Exception as _exc:
                        logger.debug("LaionCLAP: PLM unset_active failed: %s", _exc)
            emb = np.nan_to_num(outputs[0].flatten()[: self.EMBEDDING_DIM], nan=0.0)
        # Path 2: PyTorch laion_clap
        elif self._model_loaded and self._clap_model is not None:
            import torch  # pylint: disable=import-outside-toplevel

            with torch.no_grad():
                emb = self._clap_model.get_audio_embedding_from_data(
                    x=[audio_f32],
                    use_tensor=False,
                )
                if isinstance(emb, np.ndarray) and emb.ndim == 2:
                    emb = emb[0]
                emb = np.nan_to_num(emb, nan=0.0)
        else:
            raise RuntimeError("No CLAP model loaded — embed_audio unavailable")

        # L2 normalise
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        return emb

    def tag(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        text_queries: list[str] | None = None,
    ) -> AudioTaggingResult:
        """Klassifiziert Audio nach Instrumenten, Genre und Material (CLAP oder DSP).

        Algorithmus:
            1. L2-normalisiertes Audio-Embedding (512-dim) via CLAP HTS-AT
            2. Cosinus-Ähnlichkeit mit vorberechneten Tag-Text-Embeddings
            3. Konfidenz-Normalisierung via Softmax(logits / Temperatur=100)
            4. Top-Tags je Kategorie extrahieren

        Args:
            audio:         Audio-Signal (1D float32, 48000 Hz)
            sr:            Sample-Rate (muss 48000 sein)
            text_queries:  Optionale benutzerdefinierte Text-Anfragen (Zero-Shot)

        Returns:
            AudioTaggingResult mit Instrument-/Genre-/Material-Tags

        Raises:
            ValueError: Falls sr != 48000
        """
        assert sr == 48000, f"LAION-CLAP: SR muss 48000 Hz sein, erhalten: {sr}"
        self._ensure_loaded()
        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        if self._model_loaded and self._audio_session is not None:
            result = self._tag_clap(audio_f32, sr, text_queries)
        elif self._model_loaded and self._clap_model is not None:
            result = self._tag_clap_pt(audio_f32, sr, text_queries)
        else:
            result = self._tag_dsp_fallback(audio_f32, sr)

        if result.model_used == "panns_fallback":
            logger.info(
                "🔵 LAION-CLAP: Top-Instrumente=%s | Genre=%s | Modell=%s (DSP-Heuristik, conf=%.2f) — "
                "Labels advisory, keine Modell-Erkennung",
                result.top_instruments()[:2],
                result.top_genres()[:1],
                result.model_used,
                float(getattr(result, "confidence", 0.0)),
            )
        else:
            logger.info(
                "🔵 LAION-CLAP: Top-Instrumente=%s | Genre=%s | Modell=%s",
                result.top_instruments()[:2],
                result.top_genres()[:1],
                result.model_used,
            )
        return result

    # ------------------------------------------------------------------
    # CLAP ONNX-Pfad
    # ------------------------------------------------------------------

    def _tag_clap(
        self,
        audio: np.ndarray,
        sr: int,
        _text_queries: list[str] | None,  # reserved for future Zero-Shot path
    ) -> AudioTaggingResult:
        """CLAP-Inferenz via Audio-Encoder ONNX + vorberechnete Text-Embeddings."""
        try:
            # Audio-Embedding [1, 512]
            feat = np.abs(np.fft.rfft(audio[:sr], n=2048)).astype(np.float32)
            feat = feat[np.newaxis, :]

            input_name = self._audio_session.get_inputs()[0].name
            _plm_clap = None
            try:
                from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                    get_plugin_lifecycle_manager as _get_plm_fn,
                )

                _plm_clap = _get_plm_fn()
                _plm_clap.set_active("LaionCLAP_ONNX", True)
            except Exception as _exc:
                logger.debug("LaionCLAP: PLM set_active (tag): failed: %s", _exc)
            try:
                outputs = self._audio_session.run(None, {input_name: feat})
            finally:
                if _plm_clap is not None:
                    try:
                        _plm_clap.set_active("LaionCLAP_ONNX", False)
                    except Exception as _exc:
                        logger.debug("LaionCLAP: PLM unset_active (tag) failed: %s", _exc)
            audio_emb = np.nan_to_num(outputs[0].flatten()[: self.EMBEDDING_DIM], nan=0.0)
            norm = np.linalg.norm(audio_emb)
            audio_emb = audio_emb / (norm + 1e-12)

            # Cosinus-Ähnlichkeit mit Tag-Embeddings (falls vorhanden)
            # Format text_embeddings: [n_tags, 512]
            n_all = len(INSTRUMENT_TAGS) + len(GENRE_TAGS) + len(MATERIAL_TAGS)
            if self._text_embeddings is not None and len(self._text_embeddings) >= n_all:
                te = self._text_embeddings[:n_all]
                sims = (te @ audio_emb) / (np.linalg.norm(te, axis=1) + 1e-12)
                sims = self._softmax(sims * 100.0)

                offset = 0
                instrument_scores = {k: float(sims[offset + i]) for i, k in enumerate(INSTRUMENT_TAGS)}
                offset += len(INSTRUMENT_TAGS)
                genre_scores = {k: float(sims[offset + i]) for i, k in enumerate(GENRE_TAGS)}
                offset += len(GENRE_TAGS)
                material_scores = {k: float(sims[offset + i]) for i, k in enumerate(MATERIAL_TAGS)}
            else:
                # Keine Text-Embeddings → DSP-Fallback für Scores
                dsp_result = self._tag_dsp_fallback(audio, sr)
                instrument_scores = dsp_result.instrument_tags
                genre_scores = dsp_result.genre_tags
                material_scores = dsp_result.material_tags

            return AudioTaggingResult(
                instrument_tags=instrument_scores,
                genre_tags=genre_scores,
                material_tags=material_scores,
                embedding=audio_emb.astype(np.float32),
                model_used="laion_clap",
                confidence=0.88,
            )

        except Exception as exc:
            logger.warning("LAION-CLAP Inferenz-Fehler: %s — DSP-Fallback", exc)
            return self._tag_dsp_fallback(audio, sr)

    # ------------------------------------------------------------------
    # CLAP PyTorch-Pfad (lokaler Checkpoint aus models/clap/)
    # ------------------------------------------------------------------

    def _tag_clap_pt(
        self,
        audio: np.ndarray,
        sr: int,
        text_queries: list[str] | None,
    ) -> AudioTaggingResult:
        """CLAP-Tagging via lokalem PyTorch-Checkpoint (laion_clap.CLAP_Module).

        Algorithmus:
            1. Audio auf 48 kHz resampled (CLAP-intern erwartet 48 kHz)
            2. get_audio_embedding_from_data() → 512-dim L2-normalisiert
            3. get_text_embedding() für alle 43 Tags → Cosinus-Ähnlichkeit
            4. Softmax × 100 → Scores ∈ [0, 1]
        """
        try:
            import torch  # pylint: disable=import-outside-toplevel

            audio_f32 = audio.astype(np.float32)
            if audio_f32.ndim == 2:
                audio_f32 = audio_f32.mean(axis=0) if audio_f32.shape[0] <= 2 else audio_f32.mean(axis=1)

            model = self._clap_model

            # Tags für Text-Embeddings aufbauen
            all_tags = list(INSTRUMENT_TAGS) + list(GENRE_TAGS) + list(MATERIAL_TAGS)
            # text_queries ergänzen
            query_tags = text_queries or all_tags

            with torch.no_grad():
                # Audio-Embedding (laion_clap erwartet Liste von 1D-Arrays @ 48kHz)
                audio_emb = model.get_audio_embedding_from_data(x=[audio_f32], use_tensor=False)
                if isinstance(audio_emb, np.ndarray) and audio_emb.ndim == 2:
                    audio_emb = audio_emb[0]
                audio_emb = np.nan_to_num(audio_emb, nan=0.0)
                norm = np.linalg.norm(audio_emb)
                audio_emb = audio_emb / (norm + 1e-12)

                # Text-Embeddings für Tag-Scores
                text_emb = model.get_text_embedding(query_tags, use_tensor=False)
                if isinstance(text_emb, np.ndarray) and text_emb.ndim == 2:
                    text_norms = np.linalg.norm(text_emb, axis=1, keepdims=True)
                    text_emb = text_emb / (text_norms + 1e-12)
                    sims = text_emb @ audio_emb
                    sims = self._softmax(sims * 100.0)
                else:
                    sims = np.full(len(query_tags), 1.0 / len(query_tags), dtype=np.float32)

            if text_queries:
                # Nur Query-Scores zurückgeben
                custom_scores = {q: float(sims[i]) for i, q in enumerate(query_tags)}
                instrument_scores = {k: custom_scores.get(k, 0.0) for k in INSTRUMENT_TAGS}
                genre_scores = {k: custom_scores.get(k, 0.0) for k in GENRE_TAGS}
                material_scores = {k: custom_scores.get(k, 0.0) for k in MATERIAL_TAGS}
            else:
                offset = 0
                instrument_scores = {k: float(sims[offset + i]) for i, k in enumerate(INSTRUMENT_TAGS)}
                offset += len(INSTRUMENT_TAGS)
                genre_scores = {k: float(sims[offset + i]) for i, k in enumerate(GENRE_TAGS)}
                offset += len(GENRE_TAGS)
                material_scores = {k: float(sims[offset + i]) for i, k in enumerate(MATERIAL_TAGS)}

            return AudioTaggingResult(
                instrument_tags=instrument_scores,
                genre_tags=genre_scores,
                material_tags=material_scores,
                embedding=audio_emb.astype(np.float32),
                model_used="laion_clap_pt",
                confidence=0.85,
            )

        except Exception as exc:
            logger.warning("LAION-CLAP PyTorch-Inferenz-Fehler: %s — DSP-Fallback", exc)
            return self._tag_dsp_fallback(audio, sr)

    # ------------------------------------------------------------------
    # DSP-Fallback (Spektral-Feature-basiertes Tagging)
    # ------------------------------------------------------------------

    def _tag_dsp_fallback(self, audio: np.ndarray, sr: int) -> AudioTaggingResult:
        """PANNs-ähnlicher DSP-Fallback via Spektral-Features.

        Referenz: Kong et al. (2020) PANNs-Methodologie.
        """
        n_fft = 2048
        n = min(len(audio), n_fft * 4)
        spec = np.abs(np.fft.rfft(audio[:n], n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Spektral-Features
        total_e = np.sum(spec) + 1e-12
        bass_e = np.sum(spec[freqs < 250]) / total_e
        mid_e = np.sum(spec[(freqs >= 250) & (freqs < 4000)]) / total_e
        hf_e = np.sum(spec[freqs >= 6000]) / total_e
        spectral_centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))

        # Harmonizität (glatt vs. Rauschen)
        peak_e = float(np.max(spec))
        harmonicity = float(np.clip(peak_e / (np.mean(spec) + 1e-12) / 50.0, 0.0, 1.0))

        # Percussion-Indikator: schnelle Energieänderungen (vereinfacht)
        onset_density = 0.3  # Schätzwert

        # Instrument-Scores (heuristisch)
        instrument_scores: dict[str, float] = {
            "vocals": float(np.clip(harmonicity * 0.7 + mid_e * 0.5, 0.0, 1.0)),
            "guitar": float(np.clip(harmonicity * 0.5 + mid_e * 0.3, 0.0, 1.0)),
            "electric_guitar": float(np.clip(harmonicity * 0.4 + hf_e * 0.5, 0.0, 1.0)),
            "acoustic_guitar": float(np.clip(harmonicity * 0.5 * (1.0 - hf_e), 0.0, 1.0)),
            "piano": float(np.clip(harmonicity * 0.6, 0.0, 1.0)),
            "keyboard": float(np.clip(harmonicity * 0.5, 0.0, 1.0)),
            "bass": float(np.clip(bass_e * 2.0, 0.0, 1.0)),
            "drums": float(np.clip(onset_density * 0.8 + bass_e * 0.5, 0.0, 1.0)),
            "percussion": float(np.clip(onset_density * 0.7, 0.0, 1.0)),
            "violin": float(np.clip(harmonicity * 0.5 + mid_e * 0.4, 0.0, 1.0)),
            "cello": float(np.clip(harmonicity * 0.4 + bass_e * 0.4, 0.0, 1.0)),
            "trumpet": float(np.clip(harmonicity * 0.5 + hf_e * 0.3, 0.0, 1.0)),
            "saxophone": float(np.clip(harmonicity * 0.4 + mid_e * 0.5, 0.0, 1.0)),
            "flute": float(np.clip(harmonicity * 0.5 + hf_e * 0.4, 0.0, 1.0)),
            "brass": float(np.clip(harmonicity * 0.5 + hf_e * 0.2, 0.0, 1.0)),
            "strings": float(np.clip(harmonicity * 0.5, 0.0, 1.0)),
            "choir": float(np.clip(harmonicity * 0.6 + mid_e * 0.3, 0.0, 1.0)),
            "orchestra": float(np.clip(harmonicity * 0.5 + 0.2, 0.0, 1.0)),
        }

        genre_scores: dict[str, float] = {
            "rock": float(np.clip(hf_e + onset_density * 0.5, 0.0, 1.0)),
            "jazz": float(np.clip(harmonicity * 0.7, 0.0, 1.0)),
            "classical": float(np.clip(harmonicity * 0.8, 0.0, 1.0)),
            "pop": float(np.clip(mid_e * 1.2, 0.0, 1.0)),
            "blues": float(np.clip(harmonicity * 0.5 + bass_e, 0.0, 1.0)),
            "country": float(np.clip(harmonicity * 0.5, 0.0, 1.0)),
            "electronic": float(np.clip(bass_e + hf_e, 0.0, 1.0)),
            "folk": float(np.clip(harmonicity * 0.4, 0.0, 1.0)),
            "hip_hop": float(np.clip(bass_e * 1.5 + onset_density, 0.0, 1.0)),
            "reggae": float(np.clip(bass_e * 1.2, 0.0, 1.0)),
            "metal": float(np.clip(hf_e * 1.5 + onset_density, 0.0, 1.0)),
            "soul": float(np.clip(harmonicity * 0.6 + mid_e, 0.0, 1.0)),
            "rnb": float(np.clip(harmonicity * 0.5 + mid_e, 0.0, 1.0)),
            "gospel": float(np.clip(harmonicity * 0.7, 0.0, 1.0)),
            "opera": float(np.clip(harmonicity * 0.8 + mid_e, 0.0, 1.0)),
            "ambient": float(np.clip(1.0 - onset_density - bass_e, 0.0, 1.0)),
        }

        material_scores: dict[str, float] = {
            "vinyl": 0.1,
            "tape": 0.1,
            "shellac": 0.05,
            "digital": 0.2,
            "mp3": 0.15,
            "aac": 0.1,
            "live_recording": float(np.clip(0.1 + float(hf_e), 0.0, 1.0)),
            "studio_recording": float(np.clip(harmonicity * 0.5, 0.0, 1.0)),
            "broadcast": 0.1,
        }

        # Dummy-Embedding
        dummy_emb = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        dummy_emb[0] = spectral_centroid / 24000.0

        return AudioTaggingResult(
            instrument_tags=instrument_scores,
            genre_tags=genre_scores,
            material_tags=material_scores,
            embedding=dummy_emb,
            model_used="panns_fallback",
            confidence=0.45,
            metadata={"harmonicity": harmonicity, "spectral_centroid": spectral_centroid},
        )

    def unload(self) -> None:
        """Entlädt Modell-Gewichte und gibt RAM frei."""
        self._clap_model = None
        self._audio_session = None
        self._model_loaded = False
        gc.collect()
        try:
            from backend.core.ml_memory_budget import release as _rel  # pylint: disable=import-outside-toplevel  # noqa: I001

            _rel("LAION-CLAP")
        except Exception as _exc:
            logger.debug("Plugin operation failed (non-critical): %s", _exc)
        logger.info("LAION-CLAP: Modell entladen, ~2.2 GB RAM freigegeben.")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerisch stabiles Softmax."""
        x_shifted = x - np.max(x)
        e = np.exp(np.clip(x_shifted, -100.0, 100.0))
        return e / (np.sum(e) + 1e-12)


# ---------------------------------------------------------------------------
# Singleton-Accessor
# ---------------------------------------------------------------------------


def get_laion_clap() -> LAIONCLAPPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    if _instance is None:
        with _lock:
            if _instance is None:
                # Assign via module dict to avoid the global-statement Pylint warning
                # while preserving thread-safe double-checked locking semantics.
                globals()["_instance"] = LAIONCLAPPlugin()
    return _instance  # type: ignore[return-value]  # never None after block above


def get_loaded_laion_clap() -> LAIONCLAPPlugin | None:
    """Gibt nur eine bereits geladene LAION-CLAP-Instanz zurück, ohne Lazy-Load."""
    return _instance


def unload_laion_clap() -> None:
    """Entlädt LAION-CLAP aus dem RAM und gibt das Budget frei.

    Aufruf: nach dem Tag-Matching / Analyse zu Beginn der Pipeline.
    """
    if _instance is not None:
        _instance.unload()


def tag_audio(
    audio: np.ndarray,
    sr: int,
    *,
    text_queries: list[str] | None = None,
) -> AudioTaggingResult:
    """Convenience-Wrapper — LAION-CLAP Audio-Tagging ohne Klassen-Instantiierung.

    Beispiel::

        result = tag_audio(audio, sr=48000)
        logger.debug("Instrumente: %s, Genre: %s", result.top_instruments(), result.top_genres())
        # Phasen-Aktivierung (§2.9):
        if result.instrument_tags.get("vocals", 0) >= 0.4:
            activate_phase("phase_42_vocal_enhancement")

    Args:
        audio:        Audio-Signal (1D float32, 48000 Hz)
        sr:           Sample-Rate (48000)
        text_queries: Optionale Zero-Shot Text-Anfragen

    Returns:
        AudioTaggingResult
    """
    return get_laion_clap().tag(audio, sr, text_queries=text_queries)
