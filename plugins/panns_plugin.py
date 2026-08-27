"""
PANNs Plugin — Direkte ONNX-Inferenz auf Audio-Array (§4.4 Aurik Spec v10.0.0).

Dieses Plugin ersetzt die dateibasierte panns_inference.AudioTagging durch
vollständige ONNX-basierte Inferenz auf np.ndarray-Signalen (kein Datei-I/O
zur Laufzeit, keine Container-Abhängigkeit).

Modell: models/panns/panns_wavegram_logmel_cnn14.onnx
  - Input:  audio [1, 320000] float32 (10 s @ 32 kHz)
  - Output: clipwise_output [1, 527] float32 (AudioSet-527-Klassen, Sigmoid-Scores)

Spec-Referenzen:
  §4.4: PANNs CNN14 — Audio-Tagging / Instrument-Erkennung
        Top-K=10 Tags; Konfidenz-Schwellen: Vocals ≥ 0.40, Drums ≥ 0.50,
        Instruments ≥ 0.50; steuert Phasen-Aktivierungsmatrix
  §2.9: Aktivierungsmatrix (PANNs-Tag → Phase)
  §3.2: Singleton + Convenience-Pattern (Double-Checked Locking, thread-sicher)
  §3.1: NaN/Inf-Guard am Ausgang
  §3.7: Type-Annotations (PEP 484)
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.plugin_base import MLPluginBase  # §A2

# ---------------------------------------------------------------------------
# §9.7.1 SHA256-Ergebnis-Cache für PANNs-Inferenz (§3.8)
# ---------------------------------------------------------------------------
_tags_cache: dict[str, dict[str, float]] = {}
_tags_cache_lock = threading.Lock()
_TAGS_CACHE_MAX = 128  # FIFO-Trim


def _audio_tags_cache_key(audio: np.ndarray, sr: int) -> str:
    """Deterministischer Cache-Key für PANNsPlugin.get_tags()."""
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(sr.to_bytes(4, "little"))
    return f"panns:{h.hexdigest()[:16]}"


logger = logging.getLogger(__name__)

# ROCm-MIOpen-Prozess-Latch: Der MIOpen-Pooling-Kernel-Build schlägt auf
# gfx1100 (RX 7900 XTX) reproduzierbar fehl ("Code object build failed",
# miopenStatusUnknownError). Das ist eine UMWELT-Fähigkeit, kein Song-Zustand —
# §V8 (copilot-instructions.md) Song-Isolation bleibt unberührt: Der Latch gilt
# pro Prozess (nicht pro Song) und trifft keine Audio-Entscheidung, sondern nur
# die Provider-Wahl. Nach dem ersten Fehler bauen ALLE weiteren
# PANNs-Instanzen in diesem Prozess direkt eine CPU-Session — kein erneuter
# GPU-Versuch, kein MIOpen-Log-Spam (Befund 2026-08-22: Fehler wiederholte
# sich bei jeder neuen Plugin-Instanz).
_MIOPEN_POOL_FAILED_THIS_PROCESS: bool = False

# ---------------------------------------------------------------------------
# AudioSet-527-Klassen — Tag-Name → Indizes in clipwise_output[527]
# Quelle: models/clap/class_labels/audioset_class_labels_indices.json
# Schlüsselnamen entsprechen exakt den Bezeichnern im Orchestrator
# (unified_restorer_v3.py, _select_phases()).
# ---------------------------------------------------------------------------
_TAG_INDEX_MAP: dict[str, list[int]] = {
    # Gesang & Stimme (§2.9: Threshold ≥ 0.40 für Vocals/Speech)
    "Singing voice": [27, 32, 33, 34, 254, 255, 266],
    # Idx 27=Singing, 32=Male singing, 33=Female singing, 34=Child singing,
    #     254=Vocal music, 255=A capella, 266=Song
    "Vocals": [27, 32, 33, 34],
    "Speech": [0, 1, 2, 3],
    # Idx 0=Speech, 1=Male speech, 2=Female speech, 3=Child speech
    # Saiteninstrumente (§2.9: Threshold ≥ 0.50)
    "Guitar": [140, 143, 144, 146],
    # Idx 140=Guitar, 143=Acoustic guitar, 144=Steel guitar, 146=Strum
    "Electric guitar": [141],
    "Bass guitar": [142],
    # Tasteninstrumente (§2.9: Threshold ≥ 0.50)
    "Piano": [153, 154],  # Piano, Electric piano
    "Keyboard (musical)": [152, 155, 157, 158],
    # Idx 152=Keyboard, 155=Organ, 157=Hammond organ, 158=Synthesizer
    # Blasinstrumente (§2.9: Threshold ≥ 0.50)
    "Brass instrument": [185, 186, 188],  # Brass, French horn, Trombone
    "Trumpet": [187],
    "Saxophone": [197],
    # Perkussion (§2.9: Threshold ≥ 0.50)
    "Drum": [164, 162, 163, 165, 168],
    # Idx 164=Drum, 162=Drum kit, 163=Drum machine, 165=Snare, 168=Bass drum
    "Percussion": [161, 174, 176, 179, 180, 181],
    # Idx 161=Percussion, 174=Tambourine, 176=Maraca, 179=Mallet, 180=Marimba,
    #     181=Glockenspiel
    # Streicher
    "Bowed string instrument": [189, 191, 193],  # Bowed string, Violin, Cello
    # Akkordeon (Schlager-Erkennung §2.19.2)
    "Accordion": [209],
    # Genre-Überblick
    "Music": [137, 138],  # Music, Musical instrument
    "Classical music": [237],
    "Jazz": [235],
    "Rock music": [219],
    "Electronic music": [239],
}


# ---------------------------------------------------------------------------
# Singleton-Lock (§3.2 Aurik Spec — Double-Checked Locking)
# ---------------------------------------------------------------------------
_instance: PANNsPlugin | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Haupt-Plugin-Klasse
# ---------------------------------------------------------------------------


class PANNsPlugin(MLPluginBase):  # §A2
    """PANNs CNN14 — Audio-Tagging via lokaler ONNX-Inferenz.

    Erkennt Instrumente, Gesang und Genre aus np.ndarray-Signalen.
    Kein Datei-I/O, kein Netzwerk-Zugriff, kein Docker.

    Algorithmus:
        1. Audio → mono, resample auf 32 000 Hz (Modell-SR)
        2. Pad / Truncate auf genau 320 000 Samples (10 s @ 32 kHz)
        3. ONNX-Inferenz → clipwise_output [1, 527] float32 (Sigmoid-Scores ∈ [0,1])
        4. Pro Tag: max(scores[indices]) → confidence ∈ [0, 1]

    Invarianten (§3.1):
        - Alle Ausgabe-Scores ∈ [0, 1], NaN/Inf → 0.0
        - Fallback auf leeres Dict wenn ONNX nicht verfügbar (kein Absturz)
        - Thread-safe Singleton via Double-Checked Locking (§3.2)
    """

    _MODEL_SR: int = 32_000
    _MODEL_SAMPLES: int = 320_000  # 10 s × 32 000 Hz
    _ONNX_PATH: Path = Path(__file__).parent.parent / "models" / "panns" / "panns_wavegram_logmel_cnn14.onnx"

    def __init__(self) -> None:
        self._session: object | None = None
        self._torch_model: Any | None = None  # PyTorch fallback model
        self._device: str = "cpu"
        self._use_fp16: bool = False
        self._load_onnx()

    @property
    def _model_loaded(self) -> bool:
        """Für ml_model_readiness-Kompatibilität (§2.47)."""
        return self._session is not None

    # ------------------------------------------------------------------
    # GPU-Erkennung
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gpu() -> tuple[str, bool]:
        """Erkennt verfügbare GPU (CUDA oder ROCm).

        Returns:
            (device_name, fp16_supported)
        """
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if torch.cuda.is_available():
                device = "cuda"
                # Prüfe auf fp16-Unterstützung (ab Compute Capability 5.3 bzw. Volta+)
                fp16_ok = torch.cuda.get_device_capability(0)[0] >= 7
                logger.info(
                    "PANNs GPU: CUDA/ROCm erkannt (capability=%s), fp16=%s",
                    torch.cuda.get_device_capability(0),
                    fp16_ok,
                )
                return device, fp16_ok
        except Exception:
            logger.warning("panns_plugin.py::_erkennen_gpu Ersatzpfad", exc_info=True)

        try:
            import torch  # pylint: disable=import-outside-toplevel

            # ROCm: torch.cuda.is_available() returns True for AMD GPUs too
            # but we also check for MIOpen/ROCm via HIP
            if hasattr(torch, "hip") and torch.hip.is_available():
                logger.info("PANNs GPU: ROCm/HIP erkannt")
                return "cuda", True  # ROCm GPUs support fp16 well
        except Exception:
            logger.warning("panns_plugin.py::_erkennen_gpu Ersatzpfad", exc_info=True)

        logger.info("PANNs GPU: Keine GPU erkannt — CPU-Inferenz")
        return "cpu", False

    # ------------------------------------------------------------------
    # ONNX-Laden
    # ------------------------------------------------------------------

    def _load_onnx(self) -> None:
        """Lädt ONNX-Session einmalig lazy; GPU-beschleunigt wenn verfügbar; warnt bei Fehler, kein Absturz."""
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                budget_gb = 0.66
                if not _try_alloc("PANNs", size_gb=budget_gb):
                    logger.warning("PANNs: ML-Grenze erschöpft — Spektral-Ersatzpfad.")
                    return
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

            # GPU-Provider-Präferenz: CUDA > ROCm > CPU
            device, fp16_ok = self._detect_gpu()
            self._device = device
            self._use_fp16 = fp16_ok

            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                _providers = _get_prov("PANNs")
            except Exception:
                # §v10.304 GPU-Provider-Fallback: HIP-Backend = ROCm-EP, sonst CUDA-EP.
                _providers = []
                if device == "cuda":
                    try:
                        import torch as _torch_probe

                        _is_hip = bool(getattr(_torch_probe.version, "hip", None))
                    except Exception:
                        _is_hip = False
                    _providers.append("ROCMExecutionProvider" if _is_hip else "CUDAExecutionProvider")
                _providers.append("CPUExecutionProvider")

            if _MIOPEN_POOL_FAILED_THIS_PROCESS and device == "cuda":
                logger.info(
                    "PANNs ONNX: MIOPEN-Pooling in diesem Prozess als defekt erkannt — "
                    "CPU-Inferenz für alle weiteren PANNs-Sessions (Prozess-Latch)"
                )
                device = "cpu"
                self._device = "cpu"
                self._use_fp16 = False
                _providers = ["CPUExecutionProvider"]

            # ONNX Session options for GPU optimization
            sess_options = ort.SessionOptions()
            if device == "cuda":
                # Enable graph optimization for GPU
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                logger.info("PANNs ONNX: GPU-Inferenz angefordert (providers=%s)", _providers)
            else:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                logger.info("PANNs ONNX: CPU-Inferenz")

            self._session = ort.InferenceSession(
                str(self._ONNX_PATH),
                sess_options=sess_options,
                providers=_providers,
            )
            _active_providers = self._session.get_providers()
            if device == "cuda" and not any(
                "CUDAExecutionProvider" in _p or "ROCMExecutionProvider" in _p for _p in _active_providers
            ):
                # §V6 (copilot-instructions.md): GPU angefordert, ORT still auf CPU zurückgefallen — nie still.
                logger.warning(
                    "PANNs: GPU-Inferenz angefordert (%s), aber Session nutzt nur %s — Provider-Fallback (§v10.304)",
                    _providers,
                    _active_providers,
                )
            logger.info(
                "panns_plugin: CNN14 ONNX model geladen (%s, device=%s, fp16=%s, §4.4 primary genre/tagging)",
                self._ONNX_PATH.name,
                self._device,
                self._use_fp16,
            )
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm("PANNs", size_gb=0.66, unload_fn=lambda s=self: setattr(s, "_session", None))  # type: ignore[misc]
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        except Exception as exc:
            logger.warning(
                "PANNs ONNX nicht verfügbar — versuche PyTorch-Ersatzpfad: %s",
                exc,
            )
            self._session = None
            self._try_load_torch_panns()
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("PANNs")
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    def _try_load_torch_panns(self) -> None:
        """Versucht PANNs CNN14 als PyTorch-Modell zu laden (GPU-Fallback).

        Nutzt `torch.hub.load` für das pretrained CNN14-Modell von
        qiuqiangkong/audioset_tagging_cnn. Bei Erfolg wird das Modell
        auf GPU geschoben (falls verfügbar).
        """
        try:
            import torch  # pylint: disable=import-outside-toplevel

            device, fp16_ok = self._detect_gpu()
            self._device = device
            self._use_fp16 = fp16_ok

            # Load pretrained CNN14 from torch hub
            self._torch_model = torch.hub.load(
                "qiuqiangkong/audioset_tagging_cnn",
                "Cnn14",
                pretrained=True,
                trust_repo=True,
            )
            self._torch_model.eval()  # type: ignore[union-attr]

            # Move to GPU if available
            if device == "cuda":
                self._torch_model = self._torch_model.to(device)  # type: ignore[union-attr]
                if fp16_ok:
                    self._torch_model = self._torch_model.half()  # type: ignore[union-attr]
                    logger.info("PANNs PyTorch: GPU fp16-Inferenz aktiviert")
                else:
                    logger.info("PANNs PyTorch: GPU fp32-Inferenz aktiviert")
            else:
                logger.info("PANNs PyTorch: CPU-Inferenz")

            logger.info("PANNs PyTorch CNN14 geladen (device=%s, fp16=%s)", self._device, self._use_fp16)
        except Exception as exc:
            logger.warning(
                "PANNs auch als PyTorch-Modell nicht verfügbar — Instrument-Gate inaktiv: %s",
                exc,
            )
            self._torch_model = None
            self._device = "cpu"
            self._use_fp16 = False

    # ------------------------------------------------------------------
    # Audio-Aufbereitung
    # ------------------------------------------------------------------

    def _to_model_input(self, audio: np.ndarray, sr: int, position_ratio: float = 0.5) -> np.ndarray:
        """Konvertiert Audio-Array auf [1, 320000] float32 @ 32 kHz.

        Algorithmus:
            1. Stereo → Mono (arithmetisches Mittel)
            2. NaN/Inf bereinigen
            3. Resample auf 32 000 Hz (scipy.signal.resample_poly, Fallback: linspace-interp)
            4. Amplituden-Normalisierung auf Spitze ≈ 0.9
            5. 10-s-Fenster bei `position_ratio` extrahieren (0.0=Anfang, 0.5=Mitte, 1.0=Ende)
            6. Pad/Truncate auf genau 320 000 Samples

        Args:
            audio:          Mono oder Stereo, beliebige Sample-Rate, float32/64.
            sr:             Quell-Sample-Rate in Hz.
            position_ratio: Fensterposition [0.0, 1.0]; Standard 0.5 (Mitte).

        Returns:
            ndarray [1, 320000] float32, NaN/Inf-frei.
        """
        # Stereo → Mono
        if audio.ndim == 2:
            # [channels, samples] oder [samples, channels]
            audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)

        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Resample auf Modell-SR
        if sr != self._MODEL_SR:
            try:
                from scipy.signal import resample_poly

                g = math.gcd(self._MODEL_SR, sr)
                audio = resample_poly(
                    audio,
                    self._MODEL_SR // g,
                    sr // g,
                ).astype(np.float32)
            except Exception:
                n_out = max(1, round(len(audio) * self._MODEL_SR / sr))
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n_out),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

        # Amplituden-Normalisierung (§VERBOTEN: np.max → np.percentile 99.9 — Impuls-Artefakt darf Normalisierung nicht blockieren)
        peak = float(np.percentile(np.abs(audio), 99.9))
        if peak > 1e-7:
            audio = (audio / peak * 0.9).astype(np.float32)

        # Pad oder Fenster-Extraktion bei gewünschter Position
        n = len(audio)
        if n < self._MODEL_SAMPLES:
            audio = np.pad(audio, (0, self._MODEL_SAMPLES - n))
        elif n > self._MODEL_SAMPLES:
            start = max(0, int((n - self._MODEL_SAMPLES) * float(np.clip(position_ratio, 0.0, 1.0))))
            audio = audio[start : start + self._MODEL_SAMPLES]

        return audio[np.newaxis, :].astype(np.float32)  # type: ignore[no-any-return]  # [1, 320000]

    def _to_model_input_from_resampled(self, audio_mono_rs: np.ndarray, position_ratio: float) -> np.ndarray:
        """Extrahiert Fenster aus bereits resampeltem Mono-Array (spart Resample-Overhead bei Multi-Window).

        Args:
            audio_mono_rs:  Mono-Array @ _MODEL_SR, float32, bereits NaN/Inf-bereinigt.
            position_ratio: Fensterposition [0.0, 1.0].

        Returns:
            ndarray [1, 320000] float32, amplitude-normiert.
        """
        # Amplituden-Normalisierung
        peak = float(np.percentile(np.abs(audio_mono_rs), 99.9))
        chunk = (audio_mono_rs / peak * 0.9).astype(np.float32) if peak > 1e-7 else audio_mono_rs.astype(np.float32)

        n = len(chunk)
        if n < self._MODEL_SAMPLES:
            chunk = np.pad(chunk, (0, self._MODEL_SAMPLES - n))
        elif n > self._MODEL_SAMPLES:
            start = max(0, int((n - self._MODEL_SAMPLES) * float(np.clip(position_ratio, 0.0, 1.0))))
            chunk = chunk[start : start + self._MODEL_SAMPLES]

        return chunk[np.newaxis, :].astype(np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Haupt-Inferenz-Methode
    # ------------------------------------------------------------------

    def _build_cpu_session(self) -> object | None:
        """ROCm-Fallback: Session CPU-only neu aufbauen (MIOpen-Kernel-Fehler).

        Returns:
            Neue CPU-Session oder None bei Fehlschlag.
        """
        try:
            import onnxruntime as ort

            _sess_opts = ort.SessionOptions()
            _sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            return ort.InferenceSession(
                str(self._ONNX_PATH),
                sess_options=_sess_opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as _exc:
            logger.warning("PANNs: CPU-Session-Rebuild fehlgeschlagen: %s", _exc)
            return None

    def get_tags(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Gibt Konfidenz-Dict zurück (Tag-Name → Score ∈ [0, 1]).

        Schlüssel entsprechen den Bezeichnern im Orchestrator
        (unified_restorer_v3.py, _select_phases()). §4.4 Konfidenz-Schwellen:
            - "Singing voice" / "Vocals" / "Speech": threshold ≥ 0.40
            - "Drum" / "Percussion":                 threshold ≥ 0.50
            - alle Instrument-Tags:                  threshold ≥ 0.50

        Args:
            audio: Audio-Signal als np.ndarray (mono oder stereo), float32/64.
            sr:    Sample-Rate in Hz.

        Returns:
            Dict[str, float] oder leeres Dict wenn ONNX nicht verfügbar.
        """
        if self._session is None:
            return {}

        # §9.7.1 SHA256-Cache — teure ONNX-Inferenz nur bei neuem Signal
        _cache_key = _audio_tags_cache_key(audio, sr)
        with _tags_cache_lock:
            if _cache_key in _tags_cache:
                logger.debug("PANNs Zwischenspeicher-Hit: %s", _cache_key)
                return _tags_cache[_cache_key]

        _plm_panns = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_fn

            _plm_panns = _get_plm_fn()
            _plm_panns.set_active("PANNs", True)
        except Exception as _exc:
            logger.debug("PANNs: PLM set_active fehlgeschlagen: %s", _exc)

        try:
            # §PANNs-Multi-Window: Resampled Mono einmal aufbereiten, dann
            # bis zu 3 Fenster analysieren (25 %, 50 %, 75 %) und element-weises Maximum
            # bilden — verhindert, dass instrumentale Bridges/Intros den Vokal-Score
            # auf 0.00 ziehen (vocal=False für Schlager/Pop mit Vokalsolo).
            _mono_rs: np.ndarray | None = None
            try:
                _audio_mono = (
                    audio.mean(axis=0).astype(np.float32)
                    if (audio.ndim == 2 and audio.shape[0] <= 2)
                    else audio.mean(axis=1).astype(np.float32)
                    if audio.ndim == 2
                    else np.asarray(audio, dtype=np.float32)
                )
                _audio_mono = np.nan_to_num(_audio_mono, nan=0.0, posinf=0.0, neginf=0.0)
                if sr != self._MODEL_SR:
                    from scipy.signal import resample_poly as _rsp

                    _g = math.gcd(self._MODEL_SR, sr)
                    _mono_rs = _rsp(_audio_mono, self._MODEL_SR // _g, sr // _g).astype(np.float32)
                else:
                    _mono_rs = _audio_mono
            except Exception as _rs_exc:
                logger.debug("PANNs: Resample für Multi-Window fehlgeschlagen: %s", _rs_exc)

            # Primär-Inferenz: Mitte (position_ratio=0.5)
            model_input = self._to_model_input(audio, sr, position_ratio=0.5)
            try:
                ort_out = self._session.run(  # type: ignore[attr-defined]
                    None,
                    {self._session.get_inputs()[0].name: model_input},  # type: ignore[attr-defined]
                )
            except Exception as _ort_exc:
                # §ROCm-Fallback: MIOpen-Kernel-Fehler (AveragePool, "Code object
                # build failed") → Session CPU-only neu aufbauen und EINMAL
                # wiederholen. Sonst: PANNs tags: {} — Singing/Genre-Erkennung
                # fällt aus und degradiert alle panns_singing-Guards.
                global _MIOPEN_POOL_FAILED_THIS_PROCESS
                _is_miopen = "miopen" in str(_ort_exc).lower()
                if _is_miopen:
                    _MIOPEN_POOL_FAILED_THIS_PROCESS = True
                logger.warning(
                    "PANNs: ONNX-Inferenz fehlgeschlagen (%s) — CPU-Fallback-Retry%s",
                    _ort_exc,
                    " (MIOPEN defekt — Prozess-Latch gesetzt)" if _is_miopen else "",
                )
                _cpu_sess = self._build_cpu_session()
                if _cpu_sess is None:
                    raise
                self._session = _cpu_sess
                self._device = "cpu"
                self._use_fp16 = False
                model_input = self._to_model_input(audio, sr, position_ratio=0.5)
                ort_out = self._session.run(  # type: ignore[attr-defined]
                    None,
                    {self._session.get_inputs()[0].name: model_input},  # type: ignore[attr-defined]
                )
            scores: np.ndarray = ort_out[0][0].copy()  # [527] float32

            # Multi-Window-Fallback: nur wenn Singing-Score < 0.35 UND Song > 20 s bei Model-SR
            _singing_indices = _TAG_INDEX_MAP.get("Singing voice", [27])
            _singing_score_mid = max(
                (float(scores[i]) for i in _singing_indices if 0 <= i < len(scores)),
                default=0.0,
            )
            _is_long_song = _mono_rs is not None and len(_mono_rs) > 2 * self._MODEL_SAMPLES
            if _singing_score_mid < 0.35 and _is_long_song:
                for _pos in (0.25, 0.75):
                    try:
                        assert _mono_rs is not None
                        _inp = self._to_model_input_from_resampled(_mono_rs, _pos)
                        _out = self._session.run(None, {self._session.get_inputs()[0].name: _inp})  # type: ignore[attr-defined]
                        scores = np.maximum(scores, _out[0][0])
                    except Exception as _mw_exc:
                        logger.debug("PANNs Multi-Window pos=%.2f fehlgeschlagen: %s", _pos, _mw_exc)
                _singing_score_final = max(
                    (float(scores[i]) for i in _singing_indices if 0 <= i < len(scores)),
                    default=0.0,
                )
                if _singing_score_final > _singing_score_mid:
                    logger.info(
                        "PANNs Multi-Window: Singing %.2f→%.2f (Mitte-10s hatte Bridge/Instrumental)",
                        _singing_score_mid,
                        _singing_score_final,
                    )

            result: dict[str, float] = {}
            for tag, indices in _TAG_INDEX_MAP.items():
                raw = [float(scores[i]) for i in indices if 0 <= i < len(scores)]
                val = max(raw) if raw else 0.0
                result[tag] = max(0.0, min(1.0, val if math.isfinite(val) else 0.0))

            logger.debug(
                "PANNs-Tags: Vocals=%.2f Speech=%.2f Guitar=%.2f Drum=%.2f Piano=%.2f Brass=%.2f Accordion=%.2f",
                result.get("Singing voice", 0.0),
                result.get("Speech", 0.0),
                result.get("Guitar", 0.0),
                result.get("Drum", 0.0),
                result.get("Piano", 0.0),
                result.get("Brass instrument", 0.0),
                result.get("Accordion", 0.0),
            )
            # §9.7.1 Cache-Write — teure Inferenz nur einmal pro Audio
            with _tags_cache_lock:
                if len(_tags_cache) >= _TAGS_CACHE_MAX:
                    _tags_cache.pop(next(iter(_tags_cache)))  # FIFO-Trim
                _tags_cache[_cache_key] = result
            return result

        except Exception as exc:
            logger.debug("PANNs-Inferenz fehlgeschlagen — leeres Tag-Dict: %s", exc)
            return {}
        finally:
            if _plm_panns is not None:
                try:
                    _plm_panns.set_active("PANNs", False)
                except Exception as _exc:
                    logger.debug("PANNs: PLM unset_active fehlgeschlagen: %s", _exc)

    # ------------------------------------------------------------------
    # Backward-Kompatibilität: dateibasierter Aufruf (Legacy-API)
    # ------------------------------------------------------------------

    def tag(
        self,
        input_wav: str,
        output_json: str | None = None,
    ) -> dict[str, float]:
        """Legacy-Methode: Lädt Audio-Datei und delegiert an get_tags().

        Args:
            input_wav:   Pfad zur Audio-Datei (beliebiges soundfile-Format).
            output_json: Optionaler Pfad für JSON-Ausgabe (wird erstellt falls nötig).

        Returns:
            Selbes Format wie get_tags().
        """
        import json

        try:
            from backend.file_import import load_audio_file

            _res = load_audio_file(str(input_wav), do_carrier_analysis=False)
            if not isinstance(_res, dict):
                raise TypeError("load_audio_file() muss ein Dict mit audio/sr liefern")
            _payload: dict[str, Any] = _res
            audio = np.asarray(_payload["audio"], dtype=np.float32)
            sr = int(_payload["sr"])
        except Exception as exc:
            logger.error("PANNsPlugin.tag: Datei nicht lesbar '%s': %s", input_wav, exc)
            return {}

        result = self.get_tags(audio, sr)

        if output_json:
            out_path = Path(output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        return result


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 Aurik Spec)
# ---------------------------------------------------------------------------


def get_panns_plugin() -> PANNsPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PANNsPlugin()
    return _instance


def get_loaded_panns_plugin() -> PANNsPlugin | None:
    """Gibt nur eine bereits geladene PANNs-Instanz zurück, ohne Lazy-Load."""
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------


def classify_audio(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Convenience-Wrapper für get_panns_plugin().get_tags()."""
    return get_panns_plugin().get_tags(audio, sr)


# ---------------------------------------------------------------------------
# Backward-Kompatibilität: alter Klassenname PANNSPlugin (alle Großbuchstaben)
# Verwendet von: backend/adaptive_pipeline.py und Legacy-Code
# ---------------------------------------------------------------------------
PANNSPlugin = PANNsPlugin


# ---------------------------------------------------------------------------
# CLI-Nutzung
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.debug("Verwendung: panns_plugin.py <audio_datei> [Ausgabe.json]")
        sys.exit(1)

    from backend.file_import import load_audio_file

    _res = load_audio_file(sys.argv[1])
    if not isinstance(_res, dict):
        raise TypeError("load_audio_file() muss ein Dict mit audio/sr liefern")
    _payload: dict[str, Any] = _res
    _audio, _sr = _payload["audio"], int(_payload["sr"])
    _tags = classify_audio(_audio, _sr)
    logger.debug("PANNs CNN14 — %s", sys.argv[1])
    for _tag, _score in sorted(_tags.items(), key=lambda x: -x[1]):
        bar = "█" * int(_score * 20)
        logger.debug("  %s  %.4f  %s", _tag, _score, bar)

    if len(sys.argv) > 2:
        import json

        with open(sys.argv[2], "w", encoding="utf-8") as _f:
            json.dump(_tags, _f, indent=2)
        logger.debug("→ JSON gespeichert: %s", sys.argv[2])
